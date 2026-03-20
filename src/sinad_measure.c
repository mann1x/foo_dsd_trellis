/*
 * foo_dsd_trellis — Audio quality measurement
 *
 * Four metrics for DSD same-rate re-encode quality:
 *   1. A-weighted SINAD (theoretical): psychoacoustically weighted noise
 *   2. Multitone SINAD: 32-tone complex signal through pipeline
 *   3. Noise Modulation Index: noise floor variation vs signal level
 *   4. NMR (simplified PEAQ): noise vs masking threshold
 */

#include <windows.h>
#include "../include/sinad_measure.h"
#include "../include/trellis.h"
#include "../include/ntf.h"
#include "../include/fir.h"
#include "../include/dsd_types.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── Goertzel single-bin power ─── */

static double goertzel_power(const float *x, size_t n, double freq_hz,
                             double sample_rate) {
    double k = freq_hz * (double)n / sample_rate;
    double w = 2.0 * M_PI * k / (double)n;
    double coeff = 2.0 * cos(w);
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (size_t i = 0; i < n; i++) {
        s0 = (double)x[i] + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    double real = s1 - s2 * cos(w);
    double imag = s2 * sin(w);
    return (real * real + imag * imag) / ((double)n * (double)n);
}

/* ─── A-weighting (IEC 61672) ─── */

double a_weight_factor(double freq_hz) {
    if (freq_hz <= 0.0) return 0.0;
    double f2 = freq_hz * freq_hz;
    double num = 12194.0 * 12194.0 * f2 * f2;
    double den = (f2 + 20.6 * 20.6)
               * sqrt((f2 + 107.7 * 107.7) * (f2 + 737.9 * 737.9))
               * (f2 + 12194.0 * 12194.0);
    double a = num / den;
    /* Normalize so A(1kHz) = 1.0 (0 dB) */
    double a_1k = (12194.0*12194.0 * 1e12) /
                  ((1e6 + 20.6*20.6) * sqrt((1e6 + 107.7*107.7)*(1e6 + 737.9*737.9))
                   * (1e6 + 12194.0*12194.0));
    double normalized = a / a_1k;
    return normalized * normalized;  /* power factor = amplitude² */
}

/* ─── Measure SINAD with Goertzel (0-20kHz) ─── */

static double measure_goertzel_sinad(const float *out, size_t produced,
                                      double freq, uint32_t dsd_rate,
                                      double *awtd_out) {
    double actual_bw = (double)dsd_rate / (double)produced;
    unsigned sig_bin = (unsigned)(freq / actual_bw + 0.5);
    double signal_power = goertzel_power(out, produced, freq, (double)dsd_rate);
    unsigned max_bin = (unsigned)(20000.0 / actual_bw);

    double noise = 0.0, noise_awtd = 0.0;
    double sig_aw = signal_power * a_weight_factor(freq);

    for (unsigned b = 1; b <= max_bin; b++) {
        if (b >= sig_bin - 1 && b <= sig_bin + 1) continue;
        double f = b * actual_bw;
        double pwr = goertzel_power(out, produced, f, (double)dsd_rate);
        noise += pwr;
        noise_awtd += pwr * a_weight_factor(f);
    }
    if (noise <= 0.0) noise = 1e-30;
    if (noise_awtd <= 0.0) noise_awtd = 1e-30;

    if (awtd_out)
        *awtd_out = 10.0 * log10(sig_aw / noise_awtd);

    return 10.0 * log10(signal_power / noise);
}

/* ─── Multitone frequencies (32 tones, 1/10-decade spacing) ─── */

static const double g_multitone_freqs[32] = {
    17, 21, 27, 34, 42, 53, 67, 85, 107, 134, 169, 213, 268, 337,
    424, 534, 672, 846, 1065, 1340, 1687, 2124, 2674, 3366, 4237,
    5334, 6714, 8452, 10640, 13396, 16863, 21228
};

static double measure_multitone(uint32_t dsd_rate, const ntf_filter_t *f,
                                 int depth, int cands, int lat,
                                 int use_fir_lowpass, double gain) {
    unsigned rate_mult = dsd_rate / 44100;
    const unsigned n_dsd = (rate_mult <= 64) ? 262144u :
                           (rate_mult <= 128) ? 524288u :
                           (rate_mult <= 256) ? 1048576u : 2097152u;

    sdm_context_t ctx;
    if (sdm_context_init(&ctx, f, (unsigned)depth, (unsigned)cands, (unsigned)lat) != 0)
        return -999.0;

    float  *dsd_in = (float  *)malloc(n_dsd * sizeof(float));
    double *smooth = (double *)malloc(n_dsd * sizeof(double));
    float  *out    = (float  *)malloc(n_dsd * sizeof(float));
    if (!dsd_in || !smooth || !out) {
        free(dsd_in); free(smooth); free(out);
        sdm_context_free(&ctx); return -999.0;
    }

    /* Feed clean analog multitone directly to SDM (like theoretical SINAD).
     * DSD-quantized input gives ~6 dB (1-bit noise limit, not meaningful). */
    double amp = 0.5 / 32.0;
    for (unsigned i = 0; i < n_dsd; i++) {
        double s = 0.0;
        for (int t = 0; t < 32; t++)
            s += amp * sin(2.0 * M_PI * g_multitone_freqs[t] * (double)i / (double)dsd_rate);
        smooth[i] = s;
    }
    (void)dsd_in;  /* not used for multitone */

    size_t produced = sdm_process_block(&ctx, smooth, out, n_dsd);
    sdm_context_free(&ctx);

    if (produced < 1024) {
        free(dsd_in); free(smooth); free(out);
        return -999.0;
    }

    /* Goertzel at each of the 32 tones */
    double actual_bw = (double)dsd_rate / (double)produced;
    unsigned max_bin = (unsigned)(20000.0 / actual_bw);
    double total_signal = 0.0;

    /* Collect signal bin indices for exclusion */
    unsigned sig_bins[32];
    for (int t = 0; t < 32; t++) {
        sig_bins[t] = (unsigned)(g_multitone_freqs[t] / actual_bw + 0.5);
        total_signal += goertzel_power(out, produced, g_multitone_freqs[t], (double)dsd_rate);
    }

    /* Sum noise at all non-signal bins */
    double noise = 0.0;
    for (unsigned b = 1; b <= max_bin; b++) {
        int is_signal = 0;
        for (int t = 0; t < 32; t++) {
            if (b >= sig_bins[t] - 1 && b <= sig_bins[t] + 1) {
                is_signal = 1; break;
            }
        }
        if (!is_signal)
            noise += goertzel_power(out, produced, b * actual_bw, (double)dsd_rate);
    }
    if (noise <= 0.0) noise = 1e-30;

    free(dsd_in); free(smooth); free(out);
    return 10.0 * log10(total_signal / noise);
}

/* ─── Noise Modulation Index ─── */

static double measure_noise_modulation(uint32_t dsd_rate, const ntf_filter_t *f,
                                        int depth, int cands, int lat,
                                        int use_fir_lowpass, double gain) {
    static const double amplitudes[4] = { 0.1, 0.3, 0.5, 0.7 };
    double noise_floors[4];

    unsigned rate_mult = dsd_rate / 44100;
    /* Use shorter samples for speed */
    const unsigned n_dsd = (rate_mult <= 64) ? 65536u :
                           (rate_mult <= 128) ? 131072u :
                           (rate_mult <= 256) ? 262144u : 524288u;

    double freq = 1000.0;

    for (int a = 0; a < 4; a++) {
        sdm_context_t ctx;
        if (sdm_context_init(&ctx, f, (unsigned)depth, (unsigned)cands, (unsigned)lat) != 0)
            return -999.0;

        float  *dsd_in = (float  *)malloc(n_dsd * sizeof(float));
        double *smooth = (double *)malloc(n_dsd * sizeof(double));
        float  *out    = (float  *)malloc(n_dsd * sizeof(float));
        if (!dsd_in || !smooth || !out) {
            free(dsd_in); free(smooth); free(out);
            sdm_context_free(&ctx); return -999.0;
        }

        /* Feed clean analog sine at this amplitude directly to SDM */
        for (unsigned i = 0; i < n_dsd; i++)
            smooth[i] = amplitudes[a] * sin(2.0 * M_PI * freq * (double)i / (double)dsd_rate);
        (void)dsd_in;

        size_t produced = sdm_process_block(&ctx, smooth, out, n_dsd);
        sdm_context_free(&ctx);

        if (produced < 256) {
            free(dsd_in); free(smooth); free(out);
            return -999.0;
        }

        /* Measure noise floor (sum of all non-signal bins) */
        double actual_bw = (double)dsd_rate / (double)produced;
        unsigned sig_bin = (unsigned)(freq / actual_bw + 0.5);
        unsigned max_bin = (unsigned)(20000.0 / actual_bw);
        double noise = 0.0;
        for (unsigned b = 1; b <= max_bin; b++) {
            if (b >= sig_bin - 1 && b <= sig_bin + 1) continue;
            noise += goertzel_power(out, produced, b * actual_bw, (double)dsd_rate);
        }
        if (noise <= 0.0) noise = 1e-30;
        noise_floors[a] = 10.0 * log10(noise);

        free(dsd_in); free(smooth); free(out);
    }

    /* Noise modulation = max - min noise floor across amplitudes */
    double max_nf = noise_floors[0], min_nf = noise_floors[0];
    for (int a = 1; a < 4; a++) {
        if (noise_floors[a] > max_nf) max_nf = noise_floors[a];
        if (noise_floors[a] < min_nf) min_nf = noise_floors[a];
    }
    return max_nf - min_nf;
}

/* ─── Simple radix-2 FFT ─── */

static void fft_radix2(double *re, double *im, int n) {
    /* Bit-reversal permutation */
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
            double t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    /* Cooley-Tukey */
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / len;
        double wre = cos(ang), wim = sin(ang);
        for (int i = 0; i < n; i += len) {
            double ure = 1.0, uim = 0.0;
            for (int j = 0; j < len / 2; j++) {
                double tre = re[i+j+len/2]*ure - im[i+j+len/2]*uim;
                double tim = re[i+j+len/2]*uim + im[i+j+len/2]*ure;
                re[i+j+len/2] = re[i+j] - tre;
                im[i+j+len/2] = im[i+j] - tim;
                re[i+j] += tre;
                im[i+j] += tim;
                double t = ure*wre - uim*wim;
                uim = ure*wim + uim*wre;
                ure = t;
            }
        }
    }
}

/* ─── NMR (Noise-to-Mask Ratio) ─── */

/* Bark critical band edge frequencies (25 bands) */
static const double g_bark_edges[26] = {
    0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480,
    1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700,
    9500, 12000, 15500, 20000
};

/* NMR thread work structure */
typedef struct {
    const float *dsd_out;
    size_t       produced;
    uint32_t     dsd_rate;
    double       freq;
    double       result_nmr;
} nmr_work_t;

static DWORD WINAPI nmr_thread(LPVOID param) {
    nmr_work_t *w = (nmr_work_t *)param;
    w->result_nmr = -999.0;

    /* Decimate output DSD to ~48kHz via FIR chain */
    /* Use closest integer ratio to 48kHz: DSD64→44.1k (64x) */
    uint32_t target = 44100;
    fir_chain_t dec_ch;
    memset(&dec_ch, 0, sizeof(dec_ch));
    if (fir_chain_init(&dec_ch, w->dsd_rate, target) != 0)
        return 1;

    size_t pcm_sz = w->produced;  /* large enough for intermediates */
    float *pcm = (float *)calloc(pcm_sz, sizeof(float));
    if (!pcm) { fir_chain_free(&dec_ch); return 1; }

    size_t pcm_n = fir_chain_process(&dec_ch, w->dsd_out, pcm, w->produced);
    fir_chain_free(&dec_ch);

    if (pcm_n < 4096) { free(pcm); return 1; }

    /* Take 4096 samples for FFT (skip warmup) */
    size_t fft_n = 4096;
    size_t skip = 512;
    if (skip + fft_n > pcm_n) skip = 0;
    if (fft_n > pcm_n - skip) { free(pcm); return 1; }

    /* Generate reference sine at target rate */
    double *ref_re = (double *)calloc(fft_n, sizeof(double));
    double *ref_im = (double *)calloc(fft_n, sizeof(double));
    double *tst_re = (double *)calloc(fft_n, sizeof(double));
    double *tst_im = (double *)calloc(fft_n, sizeof(double));
    if (!ref_re || !ref_im || !tst_re || !tst_im) {
        free(ref_re); free(ref_im); free(tst_re); free(tst_im);
        free(pcm); return 1;
    }

    for (size_t i = 0; i < fft_n; i++) {
        ref_re[i] = 0.5 * sin(2.0 * M_PI * w->freq * (double)i / (double)target);
        tst_re[i] = (double)pcm[skip + i];
    }

    /* FFT both */
    fft_radix2(ref_re, ref_im, (int)fft_n);
    fft_radix2(tst_re, tst_im, (int)fft_n);

    /* Compute NMR per Bark band */
    double bw = (double)target / (double)fft_n;
    double total_nmr = 0.0;
    int band_count = 0;

    for (int band = 0; band < 25; band++) {
        double f_lo = g_bark_edges[band];
        double f_hi = g_bark_edges[band + 1];
        unsigned bin_lo = (unsigned)(f_lo / bw + 0.5);
        unsigned bin_hi = (unsigned)(f_hi / bw + 0.5);
        if (bin_hi > fft_n / 2) bin_hi = (unsigned)(fft_n / 2);
        if (bin_lo >= bin_hi) continue;

        /* Reference energy in this band */
        double ref_energy = 0.0;
        for (unsigned b = bin_lo; b < bin_hi; b++) {
            ref_energy += ref_re[b]*ref_re[b] + ref_im[b]*ref_im[b];
        }

        /* Noise energy = |ref - test|² per bin */
        double noise_energy = 0.0;
        for (unsigned b = bin_lo; b < bin_hi; b++) {
            double dr = ref_re[b] - tst_re[b];
            double di = ref_im[b] - tst_im[b];
            noise_energy += dr*dr + di*di;
        }

        /* Masking threshold: simplified — reference energy / 100 (-20 dB below) */
        double mask = ref_energy * 0.01;
        if (mask <= 0.0) mask = 1e-30;

        if (noise_energy > 0.0) {
            total_nmr += 10.0 * log10(noise_energy / mask);
            band_count++;
        }
    }

    if (band_count > 0)
        w->result_nmr = total_nmr / (double)band_count;

    free(ref_re); free(ref_im); free(tst_re); free(tst_im);
    free(pcm);
    return 0;
}

/* ─── Main measurement function ─── */

void sinad_measure(uint32_t dsd_rate, int ntf_id,
                   int cands, int depth, int lat,
                   int use_fir_lowpass, float fir_gain,
                   sinad_result_t *result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    result->sinad_theoretical = -999.0;
    result->sinad_awtd_theo = -999.0;
    result->multitone_sinad_db = -999.0;
    result->noise_mod_db = -999.0;
    result->nmr_db = -999.0;

    unsigned rate_mult = dsd_rate / 44100;
    const unsigned n_dsd = (rate_mult <= 64) ? 262144u :
                           (rate_mult <= 128) ? 524288u :
                           (rate_mult <= 256) ? 1048576u : 2097152u;

    /* Bin-align frequency */
    const unsigned produced_est = n_dsd - (unsigned)lat;
    double bin_width = (double)dsd_rate / (double)produced_est;
    unsigned sig_bin = (unsigned)(1000.0 / bin_width + 0.5);
    double freq = sig_bin * bin_width;

    const ntf_filter_t *f = ntf_get_filter((ntf_filter_id_t)ntf_id, dsd_rate);
    if (!f) f = ntf_auto_select(dsd_rate);
    if (!f) return;

    double gain = (double)fir_gain;
    if (gain <= 0.0) gain = 0.708;

    /* ── Metric 1: Theoretical SINAD + A-weighted ── */
    {
        sdm_context_t ctx;
        if (sdm_context_init(&ctx, f, (unsigned)depth, (unsigned)cands, (unsigned)lat) != 0)
            return;

        double *in = (double *)malloc(n_dsd * sizeof(double));
        float *out = (float *)malloc(n_dsd * sizeof(float));
        if (!in || !out) {
            free(in); free(out); sdm_context_free(&ctx); return;
        }

        for (unsigned i = 0; i < n_dsd; i++)
            in[i] = 0.5 * sin(2.0 * M_PI * freq * (double)i / (double)dsd_rate);

        size_t produced = sdm_process_block(&ctx, in, out, n_dsd);

        result->conv_fail = ctx.conv_fail;
        result->cands_collapse = ctx.cands_collapse;
        result->drop_pct = ctx.total_children > 0
            ? 100.0 * (double)ctx.next_filter_drops / (double)ctx.total_children : 0.0;

        if (produced >= 1024) {
            result->sinad_theoretical = measure_goertzel_sinad(
                out, produced, freq, dsd_rate, &result->sinad_awtd_theo);

            /* ── Metric 4: NMR on theoretical output ── */
            {
                nmr_work_t work;
                work.dsd_out = out;
                work.produced = produced;
                work.dsd_rate = dsd_rate;
                work.freq = freq;
                work.result_nmr = -999.0;

                HANDLE h = CreateThread(NULL, 4 * 1024 * 1024,
                                         nmr_thread, &work, 0, NULL);
                if (h) {
                    WaitForSingleObject(h, 30000);
                    CloseHandle(h);
                    result->nmr_db = work.result_nmr;
                }
            }
        }

        result->ok = 1;
        free(in); free(out);
        sdm_context_free(&ctx);
    }

    /* ── Metric 2: Multitone SINAD ── */
    result->multitone_sinad_db = measure_multitone(
        dsd_rate, f, depth, cands, lat, use_fir_lowpass, gain);

    /* ── Metric 3: Noise Modulation ── */
    result->noise_mod_db = measure_noise_modulation(
        dsd_rate, f, depth, cands, lat, use_fir_lowpass, gain);
}
