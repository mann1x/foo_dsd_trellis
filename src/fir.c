/*
 * foo_dsd_trellis — FIR half-band filter for DSD rate conversion
 *
 * Uses Intel IPP ippsFIRMR (multi-rate polyphase) with a 63-tap Kaiser
 * half-band filter. Multi-stage 2x up/down-sampling without zero-stuffing.
 * Polyphase decomposition only computes needed output samples (~2x faster
 * than zero-stuff + single-rate FIR for upsample, and filter-all + decimate
 * for downsample).
 */

#include "../include/fir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ipps.h>
#include <ippcore.h>

/* Force IPP to use AVX2+FMA dispatch on AMD CPUs.
 * Without this, IPP's auto-dispatcher may select SSE4 on AMD,
 * leaving up to 2x performance on the table. */
static void ipp_force_avx2(void) {
    extern void trellis_log_c(const char *);
    Ipp64u mask = 0;
    ippGetCpuFeatures(&mask, NULL);

    char msg[256];
    snprintf(msg, sizeof(msg), "IPP dispatch: mask=0x%llx AVX2=%s AVX=%s SSE42=%s AVX512F=%s",
             (unsigned long long)mask,
             (mask & ippCPUID_AVX2) ? "yes" : "NO",
             (mask & ippCPUID_AVX) ? "yes" : "no",
             (mask & ippCPUID_SSE42) ? "yes" : "no",
             (mask & ippCPUID_AVX512F) ? "yes" : "no");
    trellis_log_c(msg);

    if (!(mask & ippCPUID_AVX2)) {
        /* IPP didn't detect AVX2 — force it if CPU supports it */
        int cpuInfo[4] = {0};
        __cpuidex(cpuInfo, 7, 0);
        if (cpuInfo[1] & (1 << 5)) {  /* EBX bit 5 = AVX2 */
            ippSetCpuFeatures(mask | ippCPUID_AVX2);
            trellis_log_c("IPP: forced AVX2 dispatch on AMD");
        } else {
            trellis_log_c("IPP: CPU does not support AVX2");
        }
    }
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define IPP_HB_KAISER_BETA 12.0   /* ~120 dB stopband */

/* Global half-band taps for GPU sharing */
float  g_hb_taps[IPP_HB_NTAPS_MAX] = {0};
double g_hb_taps_d[IPP_HB_NTAPS_MAX] = {0};
int    g_hb_ntaps = IPP_HB_NTAPS;
static bool g_hb_taps_initialized = false;

/* ═══════════════════════════════════════════════════════════════════════
 * Half-band filter design
 * ═══════════════════════════════════════════════════════════════════════ */

static double bessel_I0(double x) {
    double sum = 1.0;
    double term = 1.0;
    double x_half_sq = (x / 2.0) * (x / 2.0);

    for (int k = 1; k <= 25; k++) {
        term *= x_half_sq / ((double)k * (double)k);
        sum += term;
        if (term < 1e-20 * sum)
            break;
    }
    return sum;
}

static void design_halfband_kaiser(double *h, int ntaps, double beta) {
    int center = (ntaps - 1) / 2;
    double I0b = bessel_I0(beta);
    double sum = 0.0;

    for (int n = 0; n < ntaps; n++) {
        double x = (double)(n - center) / 2.0;
        double sinc_val = (fabs(x) < 1e-15) ? 1.0 : sin(M_PI * x) / (M_PI * x);

        double t = (double)(n - center) / center;
        double arg = 1.0 - t * t;
        double w = bessel_I0(beta * sqrt(arg > 0.0 ? arg : 0.0)) / I0b;

        h[n] = sinc_val * w;
        sum += h[n];
    }

    for (int n = 0; n < ntaps; n++)
        h[n] /= sum;
}

/* ═══════════════════════════════════════════════════════════════════════
 * IPP FIRMR per-stage init/free (polyphase multi-rate)
 * ═══════════════════════════════════════════════════════════════════════ */

static int ipp_firmr_stage_init(fir_chain_t *chain, int stage_idx,
                                 bool upsample) {
    double hd[IPP_HB_NTAPS];
    design_halfband_kaiser(hd, IPP_HB_NTAPS, IPP_HB_KAISER_BETA);

    Ipp32f taps[IPP_HB_NTAPS];
    for (int i = 0; i < IPP_HB_NTAPS; i++)
        taps[i] = (Ipp32f)hd[i];

    /* Cache taps globally for GPU backend */
    if (!g_hb_taps_initialized) {
        memcpy(g_hb_taps, taps, sizeof(g_hb_taps));
        memcpy(g_hb_taps_d, hd, sizeof(g_hb_taps_d));
        g_hb_taps_initialized = true;
    }

    /* Upsample: pre-scale taps by 2 to compensate for polyphase energy split */
    if (upsample) {
        for (int i = 0; i < IPP_HB_NTAPS; i++)
            taps[i] *= 2.0f;
    }

    int upFactor   = upsample ? 2 : 1;
    int downFactor = upsample ? 1 : 2;

    int specSize = 0, bufSize = 0;
    IppStatus st = ippsFIRMRGetSize(IPP_HB_NTAPS, upFactor, downFactor,
                                     ipp32f, &specSize, &bufSize);
    if (st != ippStsNoErr)
        return -1;

    IppsFIRSpec_32f *spec = (IppsFIRSpec_32f *)ippsMalloc_8u(specSize);
    if (!spec)
        return -1;

    st = ippsFIRMRInit_32f(taps, IPP_HB_NTAPS, upFactor, 0, downFactor, 0,
                            spec);
    if (st != ippStsNoErr) {
        ippsFree(spec);
        return -1;
    }

    Ipp8u *buf = ippsMalloc_8u(bufSize);
    if (!buf) {
        ippsFree(spec);
        return -1;
    }
    ippsZero_8u(buf, bufSize);  /* Zero work buffer — IPP may read before write on some codepaths */

    int dlyLen = IPP_HB_NTAPS - 1;
    Ipp32f *dly = ippsMalloc_32f(dlyLen);
    if (!dly) {
        ippsFree(buf);
        ippsFree(spec);
        return -1;
    }
    ippsZero_32f(dly, dlyLen);

    chain->ipp_spec[stage_idx] = spec;
    chain->ipp_buf[stage_idx] = buf;
    chain->ipp_dly[stage_idx] = dly;
    chain->ipp_taps_len = IPP_HB_NTAPS;

    return 0;
}

static void ipp_firmr_stage_free(fir_chain_t *chain, int stage_idx) {
    if (chain->ipp_spec[stage_idx]) {
        ippsFree(chain->ipp_spec[stage_idx]);
        chain->ipp_spec[stage_idx] = NULL;
    }
    if (chain->ipp_buf[stage_idx]) {
        ippsFree(chain->ipp_buf[stage_idx]);
        chain->ipp_buf[stage_idx] = NULL;
    }
    if (chain->ipp_dly[stage_idx]) {
        ippsFree(chain->ipp_dly[stage_idx]);
        chain->ipp_dly[stage_idx] = NULL;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * DSD-Wide demodulation stage init/free
 *
 * Applies the same half-band LP filter at the input DSD rate to convert
 * 1-bit ±1.0 into multi-bit "DSD-Wide" before rate conversion.
 * This suppresses the massive out-of-band noise that otherwise creates
 * un-filterable spectral images during zero-stuff upsampling.
 * ═══════════════════════════════════════════════════════════════════════ */

static int ipp_firsr_demod_init(fir_chain_t *chain) {
    double hd[IPP_HB_NTAPS];
    design_halfband_kaiser(hd, IPP_HB_NTAPS, IPP_HB_KAISER_BETA);

    Ipp32f taps[IPP_HB_NTAPS];
    for (int i = 0; i < IPP_HB_NTAPS; i++)
        taps[i] = (Ipp32f)hd[i];

    int specSize = 0, bufSize = 0;
    IppStatus st = ippsFIRSRGetSize(IPP_HB_NTAPS, ipp32f, &specSize, &bufSize);
    if (st != ippStsNoErr)
        return -1;

    IppsFIRSpec_32f *spec = (IppsFIRSpec_32f *)ippsMalloc_8u(specSize);
    if (!spec)
        return -1;

    st = ippsFIRSRInit_32f(taps, IPP_HB_NTAPS, ippAlgDirect, spec);
    if (st != ippStsNoErr) {
        ippsFree(spec);
        return -1;
    }

    Ipp8u *buf = ippsMalloc_8u(bufSize);
    if (!buf) {
        ippsFree(spec);
        return -1;
    }

    int dlyLen = IPP_HB_NTAPS - 1;
    Ipp32f *dly = ippsMalloc_32f(dlyLen);
    if (!dly) {
        ippsFree(buf);
        ippsFree(spec);
        return -1;
    }
    ippsZero_32f(dly, dlyLen);

    chain->demod_spec = spec;
    chain->demod_buf = buf;
    chain->demod_dly = dly;
    return 0;
}

static void ipp_firsr_demod_free(fir_chain_t *chain) {
    if (chain->demod_spec) { ippsFree(chain->demod_spec); chain->demod_spec = NULL; }
    if (chain->demod_buf)  { ippsFree(chain->demod_buf);  chain->demod_buf = NULL; }
    if (chain->demod_dly)  { ippsFree(chain->demod_dly);  chain->demod_dly = NULL; }
    if (chain->demod_tmp)  { ippsFree(chain->demod_tmp);  chain->demod_tmp = NULL; }
    chain->demod_tmp_sz = 0;
    chain->has_demod = false;
}

static int ensure_demod_tmp(fir_chain_t *chain, size_t need) {
    if (chain->demod_tmp_sz >= need)
        return 0;
    if (chain->demod_tmp)
        ippsFree(chain->demod_tmp);
    chain->demod_tmp = ippsMalloc_32f((int)need);
    chain->demod_tmp_sz = chain->demod_tmp ? need : 0;
    return chain->demod_tmp ? 0 : -1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Upsample / Downsample 2x via IPP FIRMR (polyphase)
 *
 * No zero-stuffing or decimation needed — FIRMR handles polyphase
 * decomposition internally, only computing the output samples needed.
 * ═══════════════════════════════════════════════════════════════════════ */

static size_t ipp_upsample2(fir_chain_t *chain, int stage_idx,
                             const float *in, float *out, size_t count) {
    /* FIRMR: numIters = input_count / downFactor = count / 1 = count.
     * Each iteration consumes 1 input, produces 2 outputs. */
    IppsFIRSpec_32f *spec = (IppsFIRSpec_32f *)chain->ipp_spec[stage_idx];
    Ipp32f *dly = chain->ipp_dly[stage_idx];
    Ipp8u *buf = (Ipp8u *)chain->ipp_buf[stage_idx];

    ippsFIRMR_32f(in, out, (int)count, spec, dly, dly, buf);

    return count * 2;
}

static size_t ipp_downsample2(fir_chain_t *chain, int stage_idx,
                               const float *in, float *out, size_t in_count) {
    /* FIRMR: numIters = input_count / downFactor = in_count / 2.
     * Each iteration consumes 2 inputs, produces 1 output. */
    size_t numIters = in_count / 2;

    IppsFIRSpec_32f *spec = (IppsFIRSpec_32f *)chain->ipp_spec[stage_idx];
    Ipp32f *dly = chain->ipp_dly[stage_idx];
    Ipp8u *buf = (Ipp8u *)chain->ipp_buf[stage_idx];

    ippsFIRMR_32f(in, out, (int)numIters, spec, dly, dly, buf);

    /* IPP AMD AVX2 dispatch: NaN guard for warmup region (same as fp64 path) */
    int ntaps = chain->ipp_taps_len;
    for (int i = 0; i < ntaps && i < (int)numIters; i++) {
        if (out[i] != out[i]) out[i] = 0.0f;
    }

    return numIters;
}

/* ═══════════════════════════════════════════════════════════════════════
 * FP64 IPP FIRSR per-stage init/free
 * ═══════════════════════════════════════════════════════════════════════ */

static void ensure_hb_taps_d(void) {
    /* Make sure g_hb_taps_d is populated even if fp32 init was never called */
    if (!g_hb_taps_initialized) {
        double hd[IPP_HB_NTAPS];
        design_halfband_kaiser(hd, IPP_HB_NTAPS, IPP_HB_KAISER_BETA);
        Ipp32f taps_f[IPP_HB_NTAPS];
        for (int i = 0; i < IPP_HB_NTAPS; i++)
            taps_f[i] = (Ipp32f)hd[i];
        memcpy(g_hb_taps, taps_f, sizeof(g_hb_taps));
        memcpy(g_hb_taps_d, hd, sizeof(g_hb_taps_d));
        g_hb_taps_initialized = true;
    }
}

static int ipp_firmr_stage_init_d(fir_chain_t *chain, int stage_idx,
                                   bool upsample) {
    ensure_hb_taps_d();

    Ipp64f taps[IPP_HB_NTAPS];
    for (int i = 0; i < IPP_HB_NTAPS; i++)
        taps[i] = g_hb_taps_d[i];

    /* Upsample: pre-scale taps by 2 to compensate for polyphase energy split */
    if (upsample) {
        for (int i = 0; i < IPP_HB_NTAPS; i++)
            taps[i] *= 2.0;
    }

    int upFactor   = upsample ? 2 : 1;
    int downFactor = upsample ? 1 : 2;

    int specSize = 0, bufSize = 0;
    IppStatus st = ippsFIRMRGetSize(IPP_HB_NTAPS, upFactor, downFactor,
                                     ipp64f, &specSize, &bufSize);
    if (st != ippStsNoErr)
        return -1;

    IppsFIRSpec_64f *spec = (IppsFIRSpec_64f *)ippsMalloc_8u(specSize);
    if (!spec)
        return -1;

    st = ippsFIRMRInit_64f(taps, IPP_HB_NTAPS, upFactor, 0, downFactor, 0,
                            spec);
    if (st != ippStsNoErr) {
        ippsFree(spec);
        return -1;
    }

    Ipp8u *buf = ippsMalloc_8u(bufSize);
    if (!buf) {
        ippsFree(spec);
        return -1;
    }
    ippsZero_8u(buf, bufSize);  /* Zero work buffer — IPP may read before write on some codepaths */

    int dlyLen = IPP_HB_NTAPS - 1;
    Ipp64f *dly = ippsMalloc_64f(dlyLen);
    if (!dly) {
        ippsFree(buf);
        ippsFree(spec);
        return -1;
    }
    ippsZero_64f(dly, dlyLen);

    chain->ipp_spec_d[stage_idx] = spec;
    chain->ipp_buf_d[stage_idx] = buf;
    chain->ipp_dly_d[stage_idx] = dly;
    chain->ipp_taps_len = IPP_HB_NTAPS;

    return 0;
}

static void ipp_firmr_stage_free_d(fir_chain_t *chain, int stage_idx) {
    if (chain->ipp_spec_d[stage_idx]) {
        ippsFree(chain->ipp_spec_d[stage_idx]);
        chain->ipp_spec_d[stage_idx] = NULL;
    }
    if (chain->ipp_buf_d[stage_idx]) {
        ippsFree(chain->ipp_buf_d[stage_idx]);
        chain->ipp_buf_d[stage_idx] = NULL;
    }
    if (chain->ipp_dly_d[stage_idx]) {
        ippsFree(chain->ipp_dly_d[stage_idx]);
        chain->ipp_dly_d[stage_idx] = NULL;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * FP64 Upsample / Downsample 2x via IPP FIRMR (polyphase)
 * ═══════════════════════════════════════════════════════════════════════ */

static size_t ipp_upsample2_d(fir_chain_t *chain, int stage_idx,
                               const double *in, double *out, size_t count) {
    IppsFIRSpec_64f *spec = (IppsFIRSpec_64f *)chain->ipp_spec_d[stage_idx];
    Ipp64f *dly = chain->ipp_dly_d[stage_idx];
    Ipp8u *buf = (Ipp8u *)chain->ipp_buf_d[stage_idx];

    ippsFIRMR_64f(in, out, (int)count, spec, dly, dly, buf);

    return count * 2;
}

static size_t ipp_downsample2_d(fir_chain_t *chain, int stage_idx,
                                 const double *in, double *out, size_t in_count) {
    size_t numIters = in_count / 2;

    IppsFIRSpec_64f *spec = (IppsFIRSpec_64f *)chain->ipp_spec_d[stage_idx];
    Ipp64f *dly = chain->ipp_dly_d[stage_idx];
    Ipp8u *buf = (Ipp8u *)chain->ipp_buf_d[stage_idx];

    ippsFIRMR_64f(in, out, (int)numIters, spec, dly, dly, buf);

    /* IPP AMD AVX2 dispatch bug: ippsFIRMR_64f can produce NaN in the first
     * ~32 output samples (delay line warmup region) when heap-allocated buffers
     * have certain alignment. Replace any NaN with 0.0 (warmup period anyway). */
    int ntaps = chain->ipp_taps_len;
    for (int i = 0; i < ntaps && i < (int)numIters; i++) {
        if (out[i] != out[i]) out[i] = 0.0;
    }

    return numIters;
}

/* ═══════════════════════════════════════════════════════════════════════
 * FP64 scratch buffer management
 * ═══════════════════════════════════════════════════════════════════════ */

static int ensure_scratch_d(fir_chain_t *chain, size_t need) {
    if (chain->scratch_d_size >= need)
        return 0;
    double *p = (double *)realloc(chain->scratch_d, need * sizeof(double));
    if (!p)
        return -1;
    chain->scratch_d = p;
    chain->scratch_d_size = need;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Scratch buffer management
 * ═══════════════════════════════════════════════════════════════════════ */

static int ensure_scratch(fir_chain_t *chain, size_t need) {
    if (chain->scratch_size >= need)
        return 0;
    float *p = (float *)realloc(chain->scratch, need * sizeof(float));
    if (!p)
        return -1;
    chain->scratch = p;
    chain->scratch_size = need;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════ */

int fir_chain_init_ex(fir_chain_t *chain, uint32_t fs_in, uint32_t fs_out,
                      bool use_fp64) {
    static bool ipp_checked = false;
    if (!ipp_checked) {
        ipp_force_avx2();
        ipp_checked = true;
    }

    memset(chain, 0, sizeof(*chain));
    chain->use_fp64 = use_fp64;

    if (fs_in == fs_out) {
        chain->num_stages = 0;
        return 0;
    }

    /* Determine number of x2 or /2 stages needed */
    uint32_t ratio;

    if (fs_out > fs_in) {
        ratio = fs_out / fs_in;
        chain->upsample = true;
    } else {
        ratio = fs_in / fs_out;
        chain->upsample = false;
    }

    /* Must be power of 2 */
    if (ratio == 0 || (ratio & (ratio - 1)) != 0)
        return -1;

    int stages = 0;
    uint32_t r = ratio;
    while (r > 1) {
        stages++;
        r >>= 1;
    }

    if (stages > FIR_MAX_STAGES)
        return -1;

    chain->num_stages = stages;

    /* Init per-stage IPP FIRMR specs (polyphase multi-rate) */
    if (use_fp64) {
        for (int i = 0; i < stages; i++) {
            if (ipp_firmr_stage_init_d(chain, i, chain->upsample) != 0) {
                for (int j = 0; j < i; j++)
                    ipp_firmr_stage_free_d(chain, j);
                return -1;
            }
        }
    } else {
        for (int i = 0; i < stages; i++) {
            if (ipp_firmr_stage_init(chain, i, chain->upsample) != 0) {
                for (int j = 0; j < i; j++)
                    ipp_firmr_stage_free(chain, j);
                return -1;
            }
        }
    }

    /* DSD-Wide demod disabled: the unfiltered DSD noise acts as natural
     * dithering for the SDM re-encoder, producing better end-to-end SINAD
     * than pre-filtered (demod) input despite worse FIR-only SINAD. */

    return 0;
}

int fir_chain_init(fir_chain_t *chain, uint32_t fs_in, uint32_t fs_out) {
    return fir_chain_init_ex(chain, fs_in, fs_out, false);
}

size_t fir_chain_process(fir_chain_t *chain,
                         const float *in, float *out,
                         size_t in_count) {
    if (chain->num_stages == 0) {
        memcpy(out, in, in_count * sizeof(float));
        return in_count;
    }

    /* DSD-Wide demod: LP filter at input rate to convert 1-bit → multi-bit */
    const float *rate_in = in;
    if (chain->has_demod) {
        if (ensure_demod_tmp(chain, in_count) != 0)
            return 0;
        ippsFIRSR_32f(in, chain->demod_tmp, (int)in_count,
                       (IppsFIRSpec_32f *)chain->demod_spec,
                       chain->demod_dly, chain->demod_dly,
                       (Ipp8u *)chain->demod_buf);
        rate_in = chain->demod_tmp;
    }

    /* Single stage: direct rate_in → out */
    if (chain->num_stages == 1) {
        if (chain->upsample)
            return ipp_upsample2(chain, 0, rate_in, out, in_count);
        else
            return ipp_downsample2(chain, 0, rate_in, out, in_count);
    }

    /* Multi-stage: ping-pong between scratch and out */
    size_t max_intermediate;
    if (chain->upsample)
        max_intermediate = in_count << (chain->num_stages - 1);
    else
        max_intermediate = in_count / 2;

    if (ensure_scratch(chain, max_intermediate) != 0)
        return 0;

    float *bufs[2] = { out, chain->scratch };
    const float *src = rate_in;
    size_t count = in_count;

    for (int i = 0; i < chain->num_stages; i++) {
        int remaining = chain->num_stages - 1 - i;
        float *dst = bufs[remaining & 1];

        if (chain->upsample)
            count = ipp_upsample2(chain, i, src, dst, count);
        else
            count = ipp_downsample2(chain, i, src, dst, count);

        src = dst;
    }

    return count;
}

size_t fir_chain_process_d(fir_chain_t *chain,
                            const double *in, double *out,
                            size_t in_count) {
    if (chain->num_stages == 0) {
        memcpy(out, in, in_count * sizeof(double));
        return in_count;
    }

    /* Single stage: direct in -> out */
    if (chain->num_stages == 1) {
        if (chain->upsample)
            return ipp_upsample2_d(chain, 0, in, out, in_count);
        else
            return ipp_downsample2_d(chain, 0, in, out, in_count);
    }

    /* Multi-stage: ping-pong between scratch_d and out */
    size_t max_intermediate;
    if (chain->upsample)
        max_intermediate = in_count << (chain->num_stages - 1);
    else
        max_intermediate = in_count / 2;

    if (ensure_scratch_d(chain, max_intermediate) != 0)
        return 0;

    double *bufs[2] = { out, chain->scratch_d };
    const double *src = in;
    size_t count = in_count;

    for (int i = 0; i < chain->num_stages; i++) {
        int remaining = chain->num_stages - 1 - i;
        double *dst = bufs[remaining & 1];

        if (chain->upsample)
            count = ipp_upsample2_d(chain, i, src, dst, count);
        else
            count = ipp_downsample2_d(chain, i, src, dst, count);

        src = dst;
    }

    return count;
}

void fir_chain_reset(fir_chain_t *chain) {
    if (chain->has_demod && chain->demod_dly)
        ippsZero_32f(chain->demod_dly, chain->ipp_taps_len - 1);

    int dlyLen = chain->ipp_taps_len - 1;
    if (dlyLen <= 0) return;

    if (chain->use_fp64) {
        for (int i = 0; i < chain->num_stages; i++) {
            if (chain->ipp_dly_d[i])
                ippsZero_64f(chain->ipp_dly_d[i], dlyLen);
        }
    } else {
        for (int i = 0; i < chain->num_stages; i++) {
            if (chain->ipp_dly[i])
                ippsZero_32f(chain->ipp_dly[i], dlyLen);
        }
    }
}

void fir_chain_free(fir_chain_t *chain) {
    ipp_firsr_demod_free(chain);

    for (int i = 0; i < FIR_MAX_STAGES; i++) {
        ipp_firmr_stage_free(chain, i);
        ipp_firmr_stage_free_d(chain, i);
    }

    free(chain->scratch);
    chain->scratch = NULL;
    chain->scratch_size = 0;

    free(chain->scratch_d);
    chain->scratch_d = NULL;
    chain->scratch_d_size = 0;

    chain->num_stages = 0;
    chain->use_fp64 = false;
}

const char *fir_ipp_version(void) {
    static char ver[128] = {0};
    if (ver[0] == 0) {
        const IppLibraryVersion *v = ippGetLibVersion();
        if (v)
            snprintf(ver, sizeof(ver), "%s %s", v->Name, v->Version);
        else
            snprintf(ver, sizeof(ver), "IPP (unknown version)");
    }
    return ver;
}

const char *fir_ipp_kernel_name(void) {
    return "IPP FIRSR (63-tap Kaiser)";
}

/* ═══════════════════════════════════════════════════════════════════════
 * Same-rate lowpass FIR for DSD-Wide re-encoding
 * ═══════════════════════════════════════════════════════════════════════
 * Replaces the boxcar with a proper windowed sinc lowpass.
 * Produces smooth multi-bit output that Trellis SDM can track.
 * Cutoff at ~50 kHz (audio band + margin) regardless of DSD rate.
 */

#define LP_NTAPS 127         /* more taps = sharper transition, less noise leakage */
#define LP_KAISER_BETA 10.0  /* ~100 dB stopband */

int fir_lowpass_init(fir_lowpass_t *lp, uint32_t dsd_rate) {
    memset(lp, 0, sizeof(*lp));

    /* Design lowpass kernel: cutoff at 50 kHz.
     * With 127 taps the transition band is ~22 kHz (50-72 kHz),
     * rejecting most ultrasonic noise while preserving SDM headroom.
     * 22 kHz cutoff caused Gibbs ringing artifacts (pops). */
    double fc = 50000.0 / ((double)dsd_rate / 2.0);  /* normalized cutoff */
    if (fc > 0.5) fc = 0.5;  /* can't exceed Nyquist/2 */

    double taps_d[LP_NTAPS];
    float taps_f[LP_NTAPS];
    int M = LP_NTAPS - 1;
    double I0_beta = bessel_I0(LP_KAISER_BETA);

    for (int n = 0; n <= M; n++) {
        double x = (double)n - (double)M / 2.0;
        double sinc = (fabs(x) < 1e-10) ? 2.0 * fc : sin(2.0 * M_PI * fc * x) / (M_PI * x);
        double arg = 1.0 - (2.0 * n / M - 1.0) * (2.0 * n / M - 1.0);
        double w = bessel_I0(LP_KAISER_BETA * sqrt(fabs(arg))) / I0_beta;
        taps_d[n] = sinc * w;
    }

    /* Normalize to unity gain at DC (fp64) */
    double sum = 0.0;
    for (int n = 0; n < LP_NTAPS; n++) sum += taps_d[n];
    if (sum > 0.0)
        for (int n = 0; n < LP_NTAPS; n++) taps_d[n] /= sum;

    /* Keep fp32 copy for GPU upload */
    for (int n = 0; n < LP_NTAPS; n++) taps_f[n] = (float)taps_d[n];

    /* Create IPP FIRSR spec (fp64) */
    int specSize = 0, bufSize = 0;
    if (ippsFIRSRGetSize(LP_NTAPS, ipp64f, &specSize, &bufSize) != ippStsNoErr)
        return -1;

    lp->spec = ippsMalloc_8u(specSize);
    lp->buf = ippsMalloc_8u(bufSize);
    lp->dly = ippsMalloc_64f(LP_NTAPS - 1);
    if (!lp->spec || !lp->buf || !lp->dly) {
        fir_lowpass_free(lp);
        return -1;
    }

    ippsZero_64f(lp->dly, LP_NTAPS - 1);

    if (ippsFIRSRInit_64f(taps_d, LP_NTAPS, ippAlgAuto,
                           (IppsFIRSpec_64f *)lp->spec) != ippStsNoErr) {
        fir_lowpass_free(lp);
        return -1;
    }

    lp->coeffs = (float *)malloc(LP_NTAPS * sizeof(float));
    if (lp->coeffs)
        memcpy(lp->coeffs, taps_f, LP_NTAPS * sizeof(float));

    lp->coeffs_d = (double *)malloc(LP_NTAPS * sizeof(double));
    if (lp->coeffs_d)
        memcpy(lp->coeffs_d, taps_d, LP_NTAPS * sizeof(double));

    lp->taps = LP_NTAPS;
    lp->initialized = true;
    return 0;
}

size_t fir_lowpass_process(fir_lowpass_t *lp, const double *in, double *out, size_t count) {
    if (!lp->initialized || count == 0)
        return 0;

    ippsFIRSR_64f(in, out, (int)count,
                   (IppsFIRSpec_64f *)lp->spec,
                   lp->dly, lp->dly,
                   (Ipp8u *)lp->buf);
    return count;
}

void fir_lowpass_reset(fir_lowpass_t *lp) {
    if (lp->initialized && lp->dly)
        ippsZero_64f(lp->dly, lp->taps - 1);
}

void fir_lowpass_free(fir_lowpass_t *lp) {
    if (lp->spec) { ippsFree(lp->spec); lp->spec = NULL; }
    if (lp->buf)  { ippsFree(lp->buf);  lp->buf = NULL; }
    if (lp->dly)  { ippsFree(lp->dly);  lp->dly = NULL; }
    if (lp->coeffs) { free(lp->coeffs); lp->coeffs = NULL; }
    if (lp->coeffs_d) { free(lp->coeffs_d); lp->coeffs_d = NULL; }
    lp->initialized = false;
}
