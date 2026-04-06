/*
 * foo_dsd_trellis — Convolution filter (room correction)
 *
 * Uniform-Partitioned Overlap-Save (UPOLS) with IPP DFT in fp64.
 *
 * Algorithm:
 *   1. IR is loaded from WAV, resampled to convolution rate, partitioned
 *      into blocks of P samples, each partition FFT'd and stored.
 *   2. Per block: input accumulated into 2P-sample overlap-save buffer,
 *      FFT'd, stored in Frequency Domain Delay Line (FDL), multiplied
 *      with each IR partition and accumulated, IFFT'd, second half extracted.
 *   3. For DSD paths: decimate signal to 176.4/192 kHz, convolve, interpolate back.
 *
 * References:
 *   - Garcia, "Optimal filter partition for efficient convolution with
 *     short input/output delay", AES 113th, 2002
 *   - HiFi-LoFi/FFTConvolver (C++ reference implementation)
 */

#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include "../include/convolution.h"
#include "../include/gpu_compute.h"
#include "../include/fir.h"
#include "../include/wav_io.h"
#include "../include/resample.h"
#include <ipps.h>
#include <stdlib.h>
#include <wchar.h>
#include <io.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

extern void trellis_log_c(const char *msg);

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

static int log2i(int v) {
    int r = 0;
    while (v > 1) { v >>= 1; r++; }
    return r;
}

static int choose_partition_size(int ir_length) {
    if (ir_length <= 1024)  return 256;
    if (ir_length <= 4096)  return 512;
    if (ir_length <= 16384) return 1024;
    if (ir_length <= 65536) return 2048;
    return 4096;
}

/* ═══════════════════════════════════════════════════════════════════════
 * IPP DFT wrappers
 * ═══════════════════════════════════════════════════════════════════════ */

static int fft_init(conv_channel_t *ch, int fft_size) {
    int specSize = 0, specBufSize = 0, workBufSize = 0;
    IppStatus st = ippsDFTGetSize_C_64fc(fft_size, IPP_FFT_DIV_INV_BY_N,
                                          ippAlgHintAccurate,
                                          &specSize, &specBufSize, &workBufSize);
    if (st != ippStsNoErr) return -1;

    IppsDFTSpec_C_64fc *spec = (IppsDFTSpec_C_64fc *)ippsMalloc_8u(specSize);
    if (!spec) return -1;

    Ipp8u *specBuf = specBufSize > 0 ? ippsMalloc_8u(specBufSize) : NULL;
    st = ippsDFTInit_C_64fc(fft_size, IPP_FFT_DIV_INV_BY_N,
                             ippAlgHintAccurate, spec, specBuf);
    if (st != ippStsNoErr) {
        ippsFree(spec);
        if (specBuf) ippsFree(specBuf);
        return -1;
    }

    Ipp8u *workBuf = workBufSize > 0 ? ippsMalloc_8u(workBufSize) : NULL;
    ch->fft_spec = spec;
    ch->fft_work = workBuf;
    return 0;
}

static void fft_forward(conv_channel_t *ch, const Ipp64fc *in, Ipp64fc *out, int fft_size) {
    (void)fft_size;
    ippsDFTFwd_CToC_64fc(in, out, (IppsDFTSpec_C_64fc *)ch->fft_spec,
                          (Ipp8u *)ch->fft_work);
}

static void fft_inverse(conv_channel_t *ch, const Ipp64fc *in, Ipp64fc *out, int fft_size) {
    (void)fft_size;
    ippsDFTInv_CToC_64fc(in, out, (IppsDFTSpec_C_64fc *)ch->fft_spec,
                          (Ipp8u *)ch->fft_work);
}

/* ═══════════════════════════════════════════════════════════════════════
 * IR loading and preprocessing
 * ═══════════════════════════════════════════════════════════════════════ */

/* Resample IR from wav_rate to target_rate. Returns new sample count. */
static int resample_ir(const float *in, int in_count, uint32_t wav_rate,
                        uint32_t target_rate, double **out_samples) {
    if (wav_rate == target_rate) {
        double *out = (double *)malloc((size_t)in_count * sizeof(double));
        if (!out) return -1;
        for (int i = 0; i < in_count; i++)
            out[i] = (double)in[i];
        *out_samples = out;
        return in_count;
    }

    uint32_t hi = (target_rate > wav_rate) ? target_rate : wav_rate;
    uint32_t lo = (target_rate > wav_rate) ? wav_rate : target_rate;
    bool is_upsample = (target_rate > wav_rate);
    bool is_p2 = (hi % lo == 0) && ((hi / lo) & ((hi / lo) - 1)) == 0;

    if (is_p2) {
        fir_chain_t fir;
        memset(&fir, 0, sizeof(fir));
        if (fir_chain_init_ex(&fir, wav_rate, target_rate, true) != 0) {
            is_p2 = false;
        } else {
            double *in_d = (double *)malloc((size_t)in_count * sizeof(double));
            if (!in_d) { fir_chain_free(&fir); return -1; }
            for (int i = 0; i < in_count; i++)
                in_d[i] = (double)in[i];

            int ratio = is_upsample ? (int)(target_rate / wav_rate) : 1;
            int out_max = in_count * ratio + 1024;
            double *out = (double *)malloc((size_t)out_max * sizeof(double));
            if (!out) { free(in_d); fir_chain_free(&fir); return -1; }

            size_t out_count = fir_chain_process_d(&fir, in_d, out, (size_t)in_count);
            fir_chain_free(&fir);
            free(in_d);
            *out_samples = out;
            return (int)out_count;
        }
    }

    if (!is_p2) {
        resample_ctx_t *rs = resample_create(wav_rate, target_rate,
                                              -1 /* auto */, 1 /* HQ */);
        if (!rs) {
            /* Fallback: linear interpolation */
            double ratio_d = (double)target_rate / wav_rate;
            int out_count = (int)(in_count * ratio_d);
            double *out = (double *)malloc((size_t)out_count * sizeof(double));
            if (!out) return -1;
            for (int i = 0; i < out_count; i++) {
                double src = (double)i / ratio_d;
                int s0 = (int)src;
                double frac = src - s0;
                int s1 = s0 + 1;
                if (s1 >= in_count) s1 = in_count - 1;
                if (s0 >= in_count) s0 = in_count - 1;
                out[i] = (double)in[s0] * (1.0 - frac) + (double)in[s1] * frac;
            }
            *out_samples = out;
            return out_count;
        }

        int out_max = (int)((double)in_count * target_rate / wav_rate) + 1024;
        float *out_f = (float *)malloc((size_t)out_max * sizeof(float));
        if (!out_f) { resample_free(rs); return -1; }

        size_t out_count = resample_process(rs, in, out_f, (size_t)in_count);
        resample_free(rs);

        double *out = (double *)malloc(out_count * sizeof(double));
        if (!out) { free(out_f); return -1; }
        for (size_t i = 0; i < out_count; i++)
            out[i] = (double)out_f[i];
        free(out_f);

        *out_samples = out;
        return (int)out_count;
    }

    return -1;
}

/* Forward declaration */
static void conv_process_direct(conv_state_t *state, double *buf, size_t count);

/* Partition IR and pre-compute DFTs */
static int prepare_ir(conv_ir_t *ir, const double *samples, int count,
                       uint32_t target_rate) {
    int P = choose_partition_size(count);
    int fft_size = P * 2;
    int fft_order = log2i(fft_size);
    int num_partitions = (count + P - 1) / P;

    size_t total_complex = (size_t)num_partitions * fft_size;
    size_t alloc_bytes = total_complex * sizeof(Ipp64fc);
    Ipp64fc *freq = (Ipp64fc *)ippsMalloc_8u((int)alloc_bytes);
    if (!freq) return -1;
    memset(freq, 0, alloc_bytes);

    int specSize = 0, specBufSize = 0, workBufSize = 0;
    IppStatus st = ippsDFTGetSize_C_64fc(fft_size, IPP_FFT_DIV_INV_BY_N, ippAlgHintAccurate,
                           &specSize, &specBufSize, &workBufSize);
    if (st != ippStsNoErr) { ippsFree(freq); return -1; }

    IppsDFTSpec_C_64fc *spec = (IppsDFTSpec_C_64fc *)ippsMalloc_8u(specSize);
    Ipp8u *specBuf = specBufSize > 0 ? ippsMalloc_8u(specBufSize) : NULL;
    st = ippsDFTInit_C_64fc(fft_size, IPP_FFT_DIV_INV_BY_N,
                        ippAlgHintAccurate, spec, specBuf);
    if (st != ippStsNoErr) {
        ippsFree(spec);
        if (specBuf) ippsFree(specBuf);
        ippsFree(freq);
        return -1;
    }
    Ipp8u *workBuf = workBufSize > 0 ? ippsMalloc_8u(workBufSize) : NULL;

    Ipp64fc *temp_in = (Ipp64fc *)ippsMalloc_8u(fft_size * (int)sizeof(Ipp64fc));

    for (int p = 0; p < num_partitions; p++) {
        int offset = p * P;
        int avail = count - offset;
        int n = (avail < P) ? avail : P;

        ippsZero_64fc(temp_in, fft_size);
        for (int i = 0; i < n; i++) {
            temp_in[i].re = samples[offset + i];
            temp_in[i].im = 0.0;
        }

        ippsDFTFwd_CToC_64fc(temp_in, &freq[p * fft_size], spec, workBuf);
    }

    ippsFree(temp_in);
    if (workBuf) ippsFree(workBuf);

    ir->freq_partitions = freq;
    ir->num_partitions = num_partitions;
    ir->partition_size = P;
    ir->fft_size = fft_size;
    ir->fft_order = fft_order;
    ir->ir_length = count;
    ir->target_rate = target_rate;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Channel init / free / reset
 * ═══════════════════════════════════════════════════════════════════════ */

static int channel_init(conv_channel_t *ch, conv_ir_t *ir) {
    memset(ch, 0, sizeof(*ch));
    if (!ir || !ir->freq_partitions) return 0;

    ch->ir = ir;
    int P = ir->partition_size;
    int fft_size = ir->fft_size;

    ch->input_buf = (double *)calloc(2 * P, sizeof(double));
    if (!ch->input_buf) return -1;

    ch->fft_buf = (Ipp64fc *)ippsMalloc_8u(fft_size * (int)sizeof(Ipp64fc));
    ch->accum = (Ipp64fc *)ippsMalloc_8u(fft_size * (int)sizeof(Ipp64fc));
    if (!ch->fft_buf || !ch->accum) return -1;

    ch->out_buf = (double *)calloc(P, sizeof(double));
    if (!ch->out_buf) return -1;

    ch->fdl = (void **)calloc(ir->num_partitions, sizeof(void *));
    if (!ch->fdl) return -1;
    for (int i = 0; i < ir->num_partitions; i++) {
        ch->fdl[i] = (Ipp64fc *)ippsMalloc_8u(fft_size * (int)sizeof(Ipp64fc));
        if (!ch->fdl[i]) return -1;
        ippsZero_64fc((Ipp64fc *)ch->fdl[i], fft_size);
    }

    if (fft_init(ch, ir->fft_size) != 0) return -1;
    return 0;
}

static void channel_free(conv_channel_t *ch) {
    if (ch->input_buf) { free(ch->input_buf); ch->input_buf = NULL; }
    if (ch->out_buf) { free(ch->out_buf); ch->out_buf = NULL; }
    if (ch->fft_buf) { ippsFree(ch->fft_buf); ch->fft_buf = NULL; }
    if (ch->accum) { ippsFree(ch->accum); ch->accum = NULL; }
    if (ch->fdl) {
        int np = ch->ir ? ch->ir->num_partitions : 0;
        for (int i = 0; i < np; i++)
            if (ch->fdl[i]) ippsFree(ch->fdl[i]);
        free(ch->fdl);
        ch->fdl = NULL;
    }
    ch->fft_spec = NULL;
    ch->fft_work = NULL;
    ch->ir = NULL;
}

static void channel_reset(conv_channel_t *ch) {
    if (!ch->ir) return;
    int P = ch->ir->partition_size;
    int fft_size = ch->ir->fft_size;

    memset(ch->input_buf, 0, 2 * P * sizeof(double));
    memset(ch->out_buf, 0, P * sizeof(double));
    ch->buf_pos = 0;
    ch->out_avail = 0;
    ch->out_read = 0;
    ch->fdl_pos = 0;
    ch->fdl_filled = 0;

    for (int i = 0; i < ch->ir->num_partitions; i++)
        ippsZero_64fc((Ipp64fc *)ch->fdl[i], fft_size);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Core UPOLS processing
 * ═══════════════════════════════════════════════════════════════════════ */

static void process_partition(conv_channel_t *ch, double *out) {
    conv_ir_t *ir = ch->ir;
    int P = ir->partition_size;
    int fft_size = ir->fft_size;
    int np = ir->num_partitions;
    Ipp64fc *fft_buf = (Ipp64fc *)ch->fft_buf;
    Ipp64fc *accum = (Ipp64fc *)ch->accum;
    Ipp64fc *ir_freq = (Ipp64fc *)ir->freq_partitions;

    /* Pack real input into complex */
    for (int i = 0; i < fft_size; i++) {
        fft_buf[i].re = (i < 2 * P) ? ch->input_buf[i] : 0.0;
        fft_buf[i].im = 0.0;
    }

    /* Forward DFT of input block → store in FDL */
    Ipp64fc *input_fft = (Ipp64fc *)ch->fdl[ch->fdl_pos];
    fft_forward(ch, fft_buf, input_fft, fft_size);

    if (ch->fdl_filled < np)
        ch->fdl_filled++;

    /* Complex multiply-accumulate across filled partitions */
    ippsZero_64fc(accum, fft_size);
    int active_parts = ch->fdl_filled < np ? ch->fdl_filled : np;

    for (int k = 0; k < active_parts; k++) {
        int fdl_idx = (ch->fdl_pos - k + np) % np;
        Ipp64fc *fdl_k = (Ipp64fc *)ch->fdl[fdl_idx];
        Ipp64fc *ir_k = &ir_freq[k * fft_size];
        ippsMul_64fc(fdl_k, ir_k, fft_buf, fft_size);
        ippsAdd_64fc_I(fft_buf, accum, fft_size);
    }

    /* Inverse DFT → extract second half (overlap-save valid output) */
    fft_inverse(ch, accum, fft_buf, fft_size);
    for (int i = 0; i < P; i++)
        out[i] = fft_buf[P + i].re;

    ch->fdl_pos = (ch->fdl_pos + 1) % np;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════ */

int conv_init(conv_state_t *state, uint32_t signal_rate, uint32_t conv_rate) {
    memset(state, 0, sizeof(*state));
    state->signal_rate = signal_rate;
    state->conv_rate = conv_rate;

    if (signal_rate != conv_rate) {
        if (signal_rate < conv_rate) return -1;
        uint32_t ratio = signal_rate / conv_rate;
        if (signal_rate != conv_rate * ratio) return -1;
        state->dec_ratio = (int)ratio;
    } else {
        state->dec_ratio = 1;
    }
    return 0;
}

int conv_load_ir(conv_state_t *state, const char *wav_path) {
    if (!wav_path || wav_path[0] == '\0') return -1;

    /* Load WAV file */
    wav_data_t wav = {0};
    if (wav_read(wav_path, &wav) != 0) {
        /* Try ANSI path for Windows codepage compatibility */
        wchar_t wpath[CONV_PATH_MAX];
        char ansi_path[CONV_PATH_MAX];
        int wlen = MultiByteToWideChar(CP_UTF8, 0, wav_path, -1, wpath, CONV_PATH_MAX);
        if (wlen > 0) {
            WideCharToMultiByte(CP_ACP, 0, wpath, -1, ansi_path, CONV_PATH_MAX, NULL, NULL);
            if (wav_read(ansi_path, &wav) != 0) {
                char msg[384];
                snprintf(msg, sizeof(msg), "conv: failed to load IR: %s", wav_path);
                trellis_log_c(msg);
                return -1;
            }
        } else {
            char msg[384];
            snprintf(msg, sizeof(msg), "conv: failed to load IR: %s", wav_path);
            trellis_log_c(msg);
            return -1;
        }
    }

    /* Extract first channel if multi-channel */
    int ir_count = (int)wav.num_frames;
    float *mono = wav.samples;
    float *mono_alloc = NULL;
    if (wav.channels > 1) {
        mono_alloc = (float *)malloc((size_t)ir_count * sizeof(float));
        if (!mono_alloc) { wav_free(&wav); return -1; }
        for (int i = 0; i < ir_count; i++)
            mono_alloc[i] = wav.samples[i * wav.channels];
        mono = mono_alloc;
    }

    /* Compute original IR DC gain before resampling */
    double orig_sum = 0.0;
    for (int i = 0; i < ir_count; i++)
        orig_sum += (double)mono[i];

    /* Resample IR to convolution rate */
    double *ir_d = NULL;
    int ir_resampled = resample_ir(mono, ir_count, wav.sample_rate,
                                    state->conv_rate, &ir_d);
    free(mono_alloc);
    wav_free(&wav);

    if (ir_resampled <= 0 || !ir_d) {
        trellis_log_c("conv: IR resample failed");
        return -1;
    }

    /* Normalize resampled IR to preserve original DC gain.
     * Polyphase upsampling preserves amplitude but scales the sum by the
     * upsample ratio. Without normalization, the output is amplified,
     * causing SDM overload. */
    {
        double resampled_sum = 0.0;
        for (int i = 0; i < ir_resampled; i++)
            resampled_sum += ir_d[i];
        if (fabs(resampled_sum) > 1e-10 && fabs(orig_sum) > 1e-10) {
            double scale = orig_sum / resampled_sum;
            for (int i = 0; i < ir_resampled; i++)
                ir_d[i] *= scale;
        }
    }

    if (ir_resampled > CONV_MAX_IR_TAPS) {
        trellis_log_c("conv: IR too long");
        free(ir_d);
        return -1;
    }

    /* Prepare FFT'd partitions */
    if (prepare_ir(&state->ir, ir_d, ir_resampled, state->conv_rate) != 0) {
        free(ir_d);
        return -1;
    }
    free(ir_d);

    /* Init channel processing state */
    if (channel_init(&state->ch, &state->ir) != 0)
        return -1;

    state->active = true;

    /* Prime the output FIFO with one partition of silence */
    {
        int P = state->ir.partition_size;
        double *silence = (double *)calloc(P, sizeof(double));
        if (silence) {
            conv_process_direct(state, silence, P);
            free(silence);
        }
    }

    /* Log success */
    {
        char msg[384];
        snprintf(msg, sizeof(msg),
                 "conv: loaded '%s' — %d taps @ %u Hz, P=%d, %d partitions, dec=%dx",
                 wav_path, state->ir.ir_length, state->ir.target_rate,
                 state->ir.partition_size, state->ir.num_partitions,
                 state->dec_ratio);
        trellis_log_c(msg);
    }
    return 0;
}

/* FIFO-based UPOLS: feed all input, drain exactly count output */
static void conv_process_direct(conv_state_t *state, double *buf, size_t count) {
    conv_channel_t *ch = &state->ch;
    int P = ch->ir->partition_size;

    int max_new = (int)((ch->buf_pos + count) / (size_t)P) + 1;
    int fifo_existing = ch->out_avail - ch->out_read;
    if (fifo_existing < 0) fifo_existing = 0;

    size_t total_need = (size_t)fifo_existing + (size_t)max_new * P;
    static __declspec(thread) double *tls_fifo = NULL;
    static __declspec(thread) size_t tls_fifo_sz = 0;
    if (tls_fifo_sz < total_need) {
        free(tls_fifo);
        tls_fifo = (double *)malloc(total_need * sizeof(double));
        tls_fifo_sz = tls_fifo ? total_need : 0;
    }
    if (!tls_fifo) return;

    /* Copy existing FIFO content */
    if (fifo_existing > 0)
        memcpy(tls_fifo, ch->out_buf + ch->out_read, fifo_existing * sizeof(double));
    int fifo_len = fifo_existing;

    /* Feed ALL input */
    for (size_t in_pos = 0; in_pos < count; ) {
        int need = P - ch->buf_pos;
        int avail = (int)(count - in_pos);
        int n = (avail < need) ? avail : need;

        memcpy(&ch->input_buf[P + ch->buf_pos], &buf[in_pos], n * sizeof(double));
        ch->buf_pos += n;
        in_pos += n;

        if (ch->buf_pos >= P) {
            process_partition(ch, &tls_fifo[fifo_len]);
            fifo_len += P;
            memmove(ch->input_buf, &ch->input_buf[P], P * sizeof(double));
            ch->buf_pos = 0;
        }
    }

    /* Drain exactly count samples */
    if ((size_t)fifo_len >= count) {
        memcpy(buf, tls_fifo, count * sizeof(double));
        int remain = fifo_len - (int)count;
        if (remain > 0) {
            if ((size_t)remain > (size_t)P) {
                free(ch->out_buf);
                ch->out_buf = (double *)malloc(remain * sizeof(double));
            }
            if (ch->out_buf)
                memcpy(ch->out_buf, &tls_fifo[count], remain * sizeof(double));
            ch->out_avail = remain;
        } else {
            ch->out_avail = 0;
        }
        ch->out_read = 0;
    } else {
        /* Not enough output (startup only) */
        if (fifo_len > 0)
            memcpy(buf, tls_fifo, fifo_len * sizeof(double));
        ch->out_avail = 0;
        ch->out_read = 0;
    }
}

/* Decimated convolution for DSD paths */
static void conv_process_decimated(conv_state_t *state, double *buf, size_t count) {
    int ratio = state->dec_ratio;
    size_t dec_count = count / (size_t)ratio;
    if (dec_count == 0) return;

    size_t dec_need = dec_count + 16;
    if (dec_need > state->dec_buf_sz) {
        free(state->dec_buf);
        state->dec_buf = (double *)malloc(dec_need * sizeof(double));
        state->dec_buf_sz = state->dec_buf ? dec_need : 0;
    }
    if (!state->dec_buf) return;

    /* Step 1: Decimate via block averaging */
    for (size_t i = 0; i < dec_count; i++) {
        double sum = 0.0;
        size_t base = i * (size_t)ratio;
        for (int j = 0; j < ratio; j++)
            sum += buf[base + j];
        state->dec_buf[i] = sum / (double)ratio;
    }

    /* Step 2: Convolve at decimated rate */
    conv_process_direct(state, state->dec_buf, dec_count);

    /* Step 3: Sample-hold interpolation back to DSD rate */
    for (size_t i = 0; i < count; i++) {
        size_t idx = i / (size_t)ratio;
        buf[i] = (idx < dec_count) ? state->dec_buf[idx] : 0.0;
    }
}

void conv_process(conv_state_t *state, double *buf, size_t count) {
    if (!state || !state->active) return;

    /* GPU path: full-rate convolution via cuFFT */
    if (state->use_gpu && state->gpu_conv) {
        gpu_conv_process((gpu_context_t *)state->gpu_ctx,
                          (gpu_conv_state_t *)state->gpu_conv, buf, count);
        return;
    }

    /* CPU path */
    if (!state->ch.ir) return;
    if (state->dec_ratio > 1)
        conv_process_decimated(state, buf, count);
    else
        conv_process_direct(state, buf, count);
}

void conv_reset(conv_state_t *state) {
    if (!state) return;
    if (state->ch.ir) channel_reset(&state->ch);
}

void conv_free(conv_state_t *state) {
    if (!state) return;

    if (state->gpu_conv) {
        gpu_conv_free((gpu_context_t *)state->gpu_ctx,
                       (gpu_conv_state_t *)state->gpu_conv);
        state->gpu_conv = NULL;
    }

    channel_free(&state->ch);

    if (state->ir.freq_partitions) {
        ippsFree(state->ir.freq_partitions);
        state->ir.freq_partitions = NULL;
    }

    free(state->dec_buf);
    state->dec_buf = NULL;
    state->dec_buf_sz = 0;

    state->active = false;
}
