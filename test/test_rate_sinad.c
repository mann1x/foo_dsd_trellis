/*
 * foo_dsd_trellis — SINAD tests for all up/down rate conversion paths
 *
 * Tests all 12 DSD rate conversion combinations.
 * Method: DSD encode at fs_in → FIR rate convert → SDM re-encode at fs_out
 */

#include "test.h"
#include "../include/trellis.h"
#include "../include/ntf.h"
#include "../include/fir.h"
#include "../include/engine.h"
#include "../include/sinad_measure.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Helper: convert float buffer to double for sdm_process_block input */
static double *float_to_double(const float *src, size_t count) {
    double *dst = (double *)malloc(count * sizeof(double));
    if (dst) {
        for (size_t i = 0; i < count; i++)
            dst[i] = (double)src[i];
    }
    return dst;
}

#define SINAD_TRELLIS_DEPTH  8
#define SINAD_TRELLIS_CANDS  16
#define SINAD_TRELLIS_LAT    512

/* Fast sweep uses fewer samples + candidates for relative ranking */
#define SWEEP_TRELLIS_CANDS  8
#define SWEEP_TRELLIS_LAT    256

/* ─── Goertzel single-bin power measurement ─── */

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

/* Measure in-band SINAD on a signal (flat + A-weighted) */
static double measure_sinad_ex(const float *x, size_t n, double freq_hz,
                                double sample_rate, double *awtd_out) {
    double signal_power = goertzel_power(x, n, freq_hz, sample_rate);

    double bw = sample_rate / (double)n;
    unsigned max_bin = (unsigned)(22050.0 / bw);
    unsigned sig_bin = (unsigned)(freq_hz / bw + 0.5);

    double noise = 0.0, noise_aw = 0.0;
    double sig_aw = signal_power * a_weight_factor(freq_hz);
    for (unsigned b = 1; b <= max_bin; b++) {
        if (b >= sig_bin - 1 && b <= sig_bin + 1) continue;
        double pwr = goertzel_power(x, n, b * bw, sample_rate);
        noise += pwr;
        noise_aw += pwr * a_weight_factor(b * bw);
    }
    if (noise <= 0.0) noise = 1e-30;
    if (noise_aw <= 0.0) noise_aw = 1e-30;
    if (awtd_out)
        *awtd_out = 10.0 * log10(sig_aw / noise_aw);
    return 10.0 * log10(signal_power / noise);
}

static double measure_sinad(const float *x, size_t n, double freq_hz,
                            double sample_rate) {
    return measure_sinad_ex(x, n, freq_hz, sample_rate, NULL);
}

/* ─── Generate DSD-encoded sine at a given DSD rate ─── */

static double bin_align_freq(double target_hz, double sample_rate,
                              size_t n_produced) {
    double bw = sample_rate / (double)n_produced;
    unsigned bin = (unsigned)(target_hz / bw + 0.5);
    return bin * bw;
}

static size_t generate_dsd_sine(uint32_t dsd_rate, double freq_hz,
                                 double amplitude, size_t n_samples,
                                 float *dsd_out) {
    const ntf_filter_t *f = ntf_auto_select(dsd_rate);
    if (!f) return 0;

    sdm_context_t ctx;
    if (sdm_context_init(&ctx, f, SINAD_TRELLIS_DEPTH, SINAD_TRELLIS_CANDS,
                         SINAD_TRELLIS_LAT) != 0)
        return 0;

    double *sine = (double *)malloc(n_samples * sizeof(double));
    if (!sine) { sdm_context_free(&ctx); return 0; }

    for (size_t i = 0; i < n_samples; i++)
        sine[i] = amplitude * sin(2.0 * M_PI * freq_hz *
                                   (double)i / (double)dsd_rate);

    size_t produced = sdm_process_block(&ctx, sine, dsd_out, n_samples);

    free(sine);
    sdm_context_free(&ctx);
    return produced;
}

/* ─── Measure SINAD for a rate conversion path ─── */

/* Single-frequency measurement core (no printf).
 * If awtd_out is non-NULL, stores A-weighted SINAD there. */
static double measure_rate_sinad_at(uint32_t fs_in, uint32_t fs_out,
                                     double target_hz,
                                     const engine_path_info_t *pi,
                                     int cands, int lat, int depth,
                                     double *awtd_out) {
    unsigned base = rate_is_48k_family(fs_in) ? 48000 : 44100;
    unsigned mult_in = fs_in / base;
    size_t n_in;
    if (mult_in <= 64)       n_in = 262144;
    else if (mult_in <= 128) n_in = 524288;
    else if (mult_in <= 256) n_in = 1048576;
    else                     n_in = 2097152;

    size_t max_out;
    if (fs_out >= fs_in)
        max_out = n_in * (fs_out / fs_in) + 4096;
    else
        max_out = n_in / 2 + 4096;

    float *dsd_in  = (float *)malloc(n_in * sizeof(float));
    float *fir_buf = (float *)malloc(max_out * sizeof(float));
    float *dsd_out = (float *)malloc(max_out * sizeof(float));
    if (!dsd_in || !fir_buf || !dsd_out) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    size_t est_in = n_in - SINAD_TRELLIS_LAT;
    size_t est_fir = (fs_out >= fs_in) ?
        est_in * (fs_out / fs_in) : est_in / (fs_in / fs_out);
    size_t est_sdm = (est_fir > (size_t)lat) ? est_fir - (size_t)lat : 1024;
    double freq = bin_align_freq(target_hz, (double)fs_out, est_sdm);
    size_t dsd_in_count = generate_dsd_sine(fs_in, freq, 0.5, n_in, dsd_in);

    if (dsd_in_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    /* FIR processing: rate conversion uses fir_chain (fp64), same-rate uses lowpass.
     * Production engine keeps fp64 FIR output as double all the way to SDM —
     * no float truncation. Match that here. */
    size_t fir_count;
    double *fir_d_out = NULL;  /* fp64 output (kept alive for SDM input) */
    if (fs_in == fs_out) {
        /* Same-rate: boxcar DSD-Wide smoothing (matches engine production path).
         * Boxcar preserves DSD noise as dither → +30 dB over FIR lowpass. */
        unsigned base_r = rate_is_48k_family(fs_in) ? 48000 : 44100;
        unsigned mult_r = fs_in / base_r;
        int box_taps = (mult_r >= 512) ? 16 : (mult_r >= 128) ? 64 : 32;

        fir_d_out = (double *)malloc(max_out * sizeof(double));
        if (!fir_d_out) {
            free(dsd_in); free(fir_buf); free(dsd_out);
            fir_d_out = NULL; return -999.0;
        }
        /* Boxcar running average */
        double bsum = 0.0;
        double *ring = (double *)calloc(box_taps, sizeof(double));
        if (!ring) { free(fir_d_out); free(dsd_in); free(fir_buf); free(dsd_out); fir_d_out = NULL; return -999.0; }
        int bpos = 0;
        double inv_n = 1.0 / (double)box_taps;
        for (size_t i = 0; i < dsd_in_count; i++) {
            double s = (double)dsd_in[i];
            bsum -= ring[bpos];
            ring[bpos] = s;
            bsum += s;
            bpos = (bpos + 1) % box_taps;
            fir_d_out[i] = bsum * inv_n;
        }
        free(ring);

        /* Pre-SDM pre-emphasis DISABLED: optimal k is signal-dependent
         * (k=0.007 for 1kHz, k=0.05 for 100Hz, k=0.02 for 10kHz).
         * Fixed k helps test tones but hurts diverse music content.
         * Needs adaptive ML model — see train_pre_sdm.py. */

        fir_count = dsd_in_count;
    } else {
        /* Rate conversion: fp64 FIR chain (matches production Auto=fp64).
         * Keep output as double — no float truncation, matching engine.c. */
        fir_chain_t fir;
        if (fir_chain_init_ex(&fir, fs_in, fs_out, true) != 0) {
            free(dsd_in); free(fir_buf); free(dsd_out);
            return -999.0;
        }
        double *fir_d_in = (double *)malloc(dsd_in_count * sizeof(double));
        fir_d_out = (double *)calloc(max_out, sizeof(double));
        if (!fir_d_in || !fir_d_out) {
            free(fir_d_in); free(fir_d_out); free(dsd_in); free(fir_buf); free(dsd_out);
            fir_chain_free(&fir); fir_d_out = NULL; return -999.0;
        }

        /* Boxcar pre-smooth for paths where it helps (tested 2026-03-27):
         * DSD64→256: box=4 (+33 dB), DSD256→128: box=32 (+18 dB).
         * Other paths: raw DSD (noise as natural dither is optimal). */
        unsigned base_r = rate_is_48k_family(fs_in) ? 48000 : 44100;
        unsigned mult_in_r = fs_in / base_r;
        unsigned mult_out_r = fs_out / base_r;
        int pre_box = 0;
        if (mult_in_r == 256 && mult_out_r == 128) pre_box = 32;  /* +11 dB /48, +1.5 /44 */

        if (pre_box > 0) {
            double bsum = 0.0;
            double *ring = (double *)calloc(pre_box, sizeof(double));
            int bpos = 0;
            double inv = 1.0 / (double)pre_box;
            for (size_t i = 0; i < dsd_in_count; i++) {
                double s = (double)dsd_in[i];
                bsum -= ring[bpos]; ring[bpos] = s; bsum += s;
                bpos = (bpos + 1) % pre_box;
                fir_d_in[i] = bsum * inv;
            }
            free(ring);
        } else {
            for (size_t i = 0; i < dsd_in_count; i++)
                fir_d_in[i] = (double)dsd_in[i];
        }

        fir_count = fir_chain_process_d(&fir, fir_d_in, fir_d_out, dsd_in_count);
        free(fir_d_in);
        fir_chain_free(&fir);
    }

    if (fir_count < 1024) {
        free(fir_d_out); free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    /* Apply path-adaptive FIR gain (in fp64 when available, matching engine) */
    if (fir_d_out) {
        /* fp64 path: apply gain in double precision */
        double gain = (double)pi->fir_gain;
        if (gain != 1.0) {
            for (size_t i = 0; i < fir_count; i++)
                fir_d_out[i] *= gain;
        }
    } else {
        /* fp32/lowpass path: apply gain in float */
        if (pi->fir_gain != 1.0f) {
            for (size_t i = 0; i < fir_count; i++)
                fir_buf[i] *= pi->fir_gain;
        }
    }

    /* SDM requantize at fs_out with path-adaptive NTF/cands/lat */
    const ntf_filter_t *f_out;
    if (pi->ntf_filter != NTF_AUTO)
        f_out = ntf_get_filter((ntf_filter_id_t)pi->ntf_filter, fs_out);
    else
        f_out = ntf_auto_select(fs_out);
    if (!f_out) {
        free(fir_d_out); free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    sdm_context_t sdm;
    if (sdm_context_init(&sdm, f_out, depth, cands, lat) != 0) {
        free(fir_d_out); free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    if (pi->state_limit > 0.0)
        sdm.state_limit = pi->state_limit;

    /* Feed SDM: use fp64 FIR output directly (no float truncation) or
     * widen fp32/lowpass output to double */
    double *sdm_in_d;
    if (fir_d_out) {
        sdm_in_d = fir_d_out;  /* already fp64 — use directly */
    } else {
        sdm_in_d = float_to_double(fir_buf, fir_count);
        if (!sdm_in_d) { sdm_context_free(&sdm); free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    }
    size_t out_count = sdm_process_block(&sdm, sdm_in_d, dsd_out, fir_count);
    if (!fir_d_out) free(sdm_in_d);  /* only free if we allocated via float_to_double */
    free(fir_d_out);
    sdm_context_free(&sdm);

    if (out_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    /* Align frequency to actual output bin grid for clean Goertzel measurement. */
    freq = bin_align_freq(freq, (double)fs_out, out_count);

    /* Measure SINAD directly at DSD output rate (bin-by-bin Goertzel up to 22 kHz).
     * This is the correct end-to-end measurement: DSD→FIR→SDM→Goertzel. */
    double awtd = -999.0;
    double sinad_db = (out_count > 1024) ?
        measure_sinad_ex(dsd_out, out_count, freq, (double)fs_out, &awtd) : -999.0;

    if (awtd_out) *awtd_out = awtd;

    free(dsd_in);
    free(fir_buf);
    free(dsd_out);

    return sinad_db;
}

/* Helper: median of 3 doubles */
static double median3(double a, double b, double c) {
    if (a > b) { double t = a; a = b; b = t; }
    if (b > c) { double t = b; b = c; c = t; }
    if (a > b) { double t = a; a = b; b = t; }
    return b;
}

/* Multi-frequency median SINAD measurement.
 * Trellis SDMs are chaotic dynamical systems — a few Hz frequency shift
 * causes dramatically different quantization noise patterns. Measuring at
 * 3 adjacent frequencies (900, 1000, 1100 Hz) and taking the median gives
 * a robust quality estimate that is insensitive to individual limit cycles. */
static double measure_rate_sinad(uint32_t fs_in, uint32_t fs_out) {
    /* Query path-adaptive settings once */
    dsd_config_t test_cfg;
    memset(&test_cfg, 0, sizeof(test_cfg));
    test_cfg.fs_in = fs_in;
    test_cfg.fs_out = fs_out;
    test_cfg.trellis_depth = SINAD_TRELLIS_DEPTH;
    test_cfg.trellis_cands = SINAD_TRELLIS_CANDS;
    test_cfg.trellis_lat = SINAD_TRELLIS_LAT;

    engine_path_info_t pi;
    engine_get_path_info(fs_in, fs_out, NTF_AUTO, SDM_MODE_TRELLIS, &test_cfg, &pi);

    int cands = pi.cands > 0 ? pi.cands : SINAD_TRELLIS_CANDS;
    int lat   = pi.lat > 0 ? pi.lat : SINAD_TRELLIS_LAT;
    int depth = pi.depth > 0 ? pi.depth : SINAD_TRELLIS_DEPTH;

    /* Measure at 3 frequencies to average out SDM chaotic sensitivity */
    double a1, a2, a3;
    double s1 = measure_rate_sinad_at(fs_in, fs_out, 900.0, &pi, cands, lat, depth, &a1);
    double s2 = measure_rate_sinad_at(fs_in, fs_out, 1000.0, &pi, cands, lat, depth, &a2);
    double s3 = measure_rate_sinad_at(fs_in, fs_out, 1100.0, &pi, cands, lat, depth, &a3);
    double sinad_db = median3(s1, s2, s3);
    double awtd_db = median3(a1, a2, a3);

    unsigned base_in  = rate_is_48k_family(fs_in)  ? 48000 : 44100;
    unsigned base_out = rate_is_48k_family(fs_out) ? 48000 : 44100;
    unsigned rate_in_mult  = fs_in  / base_in;
    unsigned rate_out_mult = fs_out / base_out;
    const char *dir;
    if (fs_out > fs_in) dir = "UP";
    else if (fs_out < fs_in) dir = "DN";
    else dir = "same";
    /* MT and NMod: encoding quality at output rate (supplementary) */
    sinad_result_t r;
    memset(&r, 0, sizeof(r));
    sinad_measure(fs_out, pi.ntf_filter, cands, depth, lat, 1, pi.fir_gain, &r);

    const char *ntf_name = (pi.ntf_filter != NTF_AUTO) ?
        ntf_get_filter((ntf_filter_id_t)pi.ntf_filter, fs_out)->name : "auto";

    if (fs_in == fs_out) {
        printf("    [SINAD] DSD%u->DSD%u (%s): SINAD=%.1f dB A-wtd=%.1f MT=%.1f NMod=%.1f"
               "  [%s, gain=%.2f, cands=%d, depth=%d]\n",
               rate_in_mult, rate_out_mult, dir, sinad_db, awtd_db,
               r.multitone_sinad_db, r.noise_mod_db,
               ntf_name, pi.fir_gain, cands, depth);
    } else {
        printf("    [SINAD] DSD%u->DSD%u (%s): SINAD=%.1f dB A-wtd=%.1f MT=%.1f NMod=%.1f"
               "  [%s, gain=%.2f, lim=%s, cands=%d, depth=%d]\n",
               rate_in_mult, rate_out_mult, dir, sinad_db, awtd_db,
               r.multitone_sinad_db, r.noise_mod_db,
               ntf_name, pi.fir_gain,
               pi.state_limit > 0.0 ? "on" : "off",
               cands, depth);
    }

    return sinad_db;
}

/* ─── DSD→DSD upsample tests ─── */
/* DSD→DSD rate conversion goes through FIR + SDM re-encode.
 * SINAD is inherently limited by 1-bit quantization noise (~6 dB per
 * re-encode) plus FIR conversion artifacts. Threshold 12 dB reflects
 * the fundamental DSD→DSD quality floor. */

static void test_sinad_up_64_128(void) {
    double sinad = measure_rate_sinad(DSD_RATE_64, DSD_RATE_128);
    TEST_ASSERT_TRUE(sinad > 12.0,
                     "DSD64->DSD128 SINAD should exceed 12 dB");
}

static void test_sinad_up_64_256(void) {
    double sinad = measure_rate_sinad(DSD_RATE_64, DSD_RATE_256);
    TEST_ASSERT_TRUE(sinad > 12.0,
                     "DSD64->DSD256 SINAD should exceed 12 dB");
}

static void test_sinad_up_64_512(void) {
    double sinad = measure_rate_sinad(DSD_RATE_64, DSD_RATE_512);
    TEST_ASSERT_TRUE(sinad > 12.0,
                     "DSD64->DSD512 SINAD should exceed 12 dB");
}

static void test_sinad_up_128_256(void) {
    double sinad = measure_rate_sinad(DSD_RATE_128, DSD_RATE_256);
    TEST_ASSERT_TRUE(sinad > 12.0,
                     "DSD128->DSD256 SINAD should exceed 12 dB");
}

static void test_sinad_up_128_512(void) {
    double sinad = measure_rate_sinad(DSD_RATE_128, DSD_RATE_512);
    TEST_ASSERT_TRUE(sinad > 12.0,
                     "DSD128->DSD512 SINAD should exceed 12 dB");
}

static void test_sinad_up_256_512(void) {
    double sinad = measure_rate_sinad(DSD_RATE_256, DSD_RATE_512);
    TEST_ASSERT_TRUE(sinad > 12.0,
                     "DSD256->DSD512 SINAD should exceed 12 dB");
}

/* ─── DSD→DSD downsample tests ─── */

static void test_sinad_dn_128_64(void) {
    double sinad = measure_rate_sinad(DSD_RATE_128, DSD_RATE_64);
    TEST_ASSERT_TRUE(sinad > 12.0,
                     "DSD128->DSD64 SINAD should exceed 12 dB");
}

static void test_sinad_dn_256_64(void) {
    double sinad = measure_rate_sinad(DSD_RATE_256, DSD_RATE_64);
    TEST_ASSERT_TRUE(sinad > 12.0,
                     "DSD256->DSD64 SINAD should exceed 12 dB");
}

static void test_sinad_dn_512_64(void) {
    double sinad = measure_rate_sinad(DSD_RATE_512, DSD_RATE_64);
    TEST_ASSERT_TRUE(sinad > 12.0,
                     "DSD512->DSD64 SINAD should exceed 12 dB");
}

static void test_sinad_dn_256_128(void) {
    double sinad = measure_rate_sinad(DSD_RATE_256, DSD_RATE_128);
    TEST_ASSERT_TRUE(sinad > 12.0,
                     "DSD256->DSD128 SINAD should exceed 12 dB");
}

static void test_sinad_dn_512_128(void) {
    double sinad = measure_rate_sinad(DSD_RATE_512, DSD_RATE_128);
    TEST_ASSERT_TRUE(sinad > 12.0,
                     "DSD512->DSD128 SINAD should exceed 12 dB");
}

static void test_sinad_dn_512_256(void) {
    double sinad = measure_rate_sinad(DSD_RATE_512, DSD_RATE_256);
    TEST_ASSERT_TRUE(sinad > 12.0,
                     "DSD512->DSD256 SINAD should exceed 12 dB");
}

/* ─── DSD → PCM decimation tests (FIR only, no SDM) ─── */

static double measure_dsd_to_pcm_sinad(uint32_t dsd_rate, uint32_t pcm_rate) {
    unsigned dsd_base = rate_is_48k_family(dsd_rate) ? 48000 : 44100;
    unsigned mult_in = dsd_rate / dsd_base;
    size_t n_in;
    if (mult_in <= 64)       n_in = 262144;
    else if (mult_in <= 128) n_in = 524288;
    else if (mult_in <= 256) n_in = 1048576;
    else                     n_in = 2097152;

    uint32_t ratio = dsd_rate / pcm_rate;
    /* fir_chain_process uses out buffer for intermediate stages (ping-pong),
     * so it must be large enough for the first intermediate result (n_in/2) */
    size_t max_out = n_in / 2 + 4096;

    float *dsd_in  = (float *)malloc(n_in * sizeof(float));
    float *pcm_out = (float *)malloc(max_out * sizeof(float));
    if (!dsd_in || !pcm_out) {
        free(dsd_in); free(pcm_out);
        return -999.0;
    }

    /* Skip FIR startup transient: 63-tap filter × num_stages × 2 (safety margin)
     * at the output rate */
    size_t fir_stages = 0;
    { uint32_t r = ratio; while (r > 1) { fir_stages++; r >>= 1; } }
    size_t skip = IPP_HB_NTAPS * fir_stages * 2;

    size_t est_in_produced = n_in - SINAD_TRELLIS_LAT;
    size_t est_pcm_out = est_in_produced / ratio;
    size_t est_measured = (est_pcm_out > skip) ? est_pcm_out - skip : est_pcm_out;
    double freq = bin_align_freq(1000.0, (double)pcm_rate, est_measured);

    size_t dsd_in_count = generate_dsd_sine(dsd_rate, freq, 0.5, n_in, dsd_in);
    if (dsd_in_count < 1024) {
        free(dsd_in); free(pcm_out);
        return -999.0;
    }

    /* FIR decimation (no SDM — output is multi-bit PCM) */
    fir_chain_t fir;
    if (fir_chain_init(&fir, dsd_rate, pcm_rate) != 0) {
        free(dsd_in); free(pcm_out);
        return -999.0;
    }
    size_t pcm_count = fir_chain_process(&fir, dsd_in, pcm_out, dsd_in_count);
    fir_chain_free(&fir);

    if (pcm_count <= skip + 1024) {
        free(dsd_in); free(pcm_out);
        return -999.0;
    }

    /* Measure on steady-state portion (skip startup transient) */
    float *meas_ptr = pcm_out + skip;
    size_t meas_count = pcm_count - skip;

    float mn = meas_ptr[0], mx = meas_ptr[0];
    for (size_t i = 0; i < meas_count; i++) {
        if (meas_ptr[i] < mn) mn = meas_ptr[i];
        if (meas_ptr[i] > mx) mx = meas_ptr[i];
    }

    double sig_pwr = goertzel_power(meas_ptr, meas_count, freq, (double)pcm_rate);
    double pcm_amp = sqrt(sig_pwr * 4.0);
    double sinad_db = measure_sinad(meas_ptr, meas_count, freq, (double)pcm_rate);

    unsigned rate_mult = dsd_rate / dsd_base;
    printf("    [SINAD] DSD%u%s->PCM%u: SINAD=%.1f dB  [%ux, n=%zu, skip=%zu, amp=%.4f, range=[%.4f,%.4f]]\n",
           rate_mult, dsd_base == 48000 ? "/48" : "", pcm_rate, sinad_db, ratio, meas_count, skip, pcm_amp, mn, mx);

    free(dsd_in);
    free(pcm_out);
    return sinad_db;
}

static void test_sinad_dsd64_pcm44(void) {
    double sinad = measure_dsd_to_pcm_sinad(DSD_RATE_64, 44100);
    TEST_ASSERT_TRUE(sinad > 90.0, "DSD64->PCM44 SINAD should exceed 90 dB");
}

static void test_sinad_dsd64_pcm88(void) {
    double sinad = measure_dsd_to_pcm_sinad(DSD_RATE_64, 88200);
    TEST_ASSERT_TRUE(sinad > 90.0, "DSD64->PCM88 SINAD should exceed 90 dB");
}

static void test_sinad_dsd64_pcm176(void) {
    double sinad = measure_dsd_to_pcm_sinad(DSD_RATE_64, 176400);
    TEST_ASSERT_TRUE(sinad > 85.0, "DSD64->PCM176 SINAD should exceed 85 dB");
}

static void test_sinad_dsd128_pcm44(void) {
    double sinad = measure_dsd_to_pcm_sinad(DSD_RATE_128, 44100);
    TEST_ASSERT_TRUE(sinad > 100.0, "DSD128->PCM44 SINAD should exceed 100 dB");
}

static void test_sinad_dsd128_pcm88(void) {
    double sinad = measure_dsd_to_pcm_sinad(DSD_RATE_128, 88200);
    TEST_ASSERT_TRUE(sinad > 100.0, "DSD128->PCM88 SINAD should exceed 100 dB");
}

static void test_sinad_dsd128_pcm176(void) {
    double sinad = measure_dsd_to_pcm_sinad(DSD_RATE_128, 176400);
    TEST_ASSERT_TRUE(sinad > 90.0, "DSD128->PCM176 SINAD should exceed 90 dB");
}

static void test_sinad_dsd256_pcm44(void) {
    double sinad = measure_dsd_to_pcm_sinad(DSD_RATE_256, 44100);
    TEST_ASSERT_TRUE(sinad > 100.0, "DSD256->PCM44 SINAD should exceed 100 dB");
}

static void test_sinad_dsd256_pcm88(void) {
    double sinad = measure_dsd_to_pcm_sinad(DSD_RATE_256, 88200);
    TEST_ASSERT_TRUE(sinad > 100.0, "DSD256->PCM88 SINAD should exceed 100 dB");
}

static void test_sinad_dsd256_pcm176(void) {
    double sinad = measure_dsd_to_pcm_sinad(DSD_RATE_256, 176400);
    TEST_ASSERT_TRUE(sinad > 100.0, "DSD256->PCM176 SINAD should exceed 100 dB");
}

static void test_sinad_dsd512_pcm44(void) {
    double sinad = measure_dsd_to_pcm_sinad(DSD_RATE_512, 44100);
    TEST_ASSERT_TRUE(sinad > 100.0, "DSD512->PCM44 SINAD should exceed 100 dB");
}

static void test_sinad_dsd512_pcm88(void) {
    double sinad = measure_dsd_to_pcm_sinad(DSD_RATE_512, 88200);
    TEST_ASSERT_TRUE(sinad > 100.0, "DSD512->PCM88 SINAD should exceed 100 dB");
}

static void test_sinad_dsd512_pcm176(void) {
    double sinad = measure_dsd_to_pcm_sinad(DSD_RATE_512, 176400);
    TEST_ASSERT_TRUE(sinad > 100.0, "DSD512->PCM176 SINAD should exceed 100 dB");
}

static void test_sinad_dsd512_pcm352(void) {
    double sinad = measure_dsd_to_pcm_sinad(DSD_RATE_512, 352800);
    TEST_ASSERT_TRUE(sinad > 100.0, "DSD512->PCM352 SINAD should exceed 100 dB");
}

/* ─── Diagnostic: PCM sine through FIR decimation (control test) ─── */

static void test_diag_pcm_fir_control(void) {
    printf("\n    --- PCM sine through FIR decimation (control) ---\n");

    /* Generate a pure multi-bit PCM sine at DSD64 rate, decimate to PCM44100 */
    uint32_t fs_in = DSD_RATE_64;   /* 2822400 */
    uint32_t fs_out = 44100;
    uint32_t ratio = fs_in / fs_out; /* 64 */
    size_t n_in = 262144;
    size_t max_out = n_in / 2 + 4096;

    float *pcm_in  = (float *)malloc(n_in * sizeof(float));
    float *pcm_out = (float *)malloc(max_out * sizeof(float));
    if (!pcm_in || !pcm_out) { free(pcm_in); free(pcm_out); return; }

    size_t est_out = n_in / ratio;
    double freq = bin_align_freq(1000.0, (double)fs_out, est_out);

    /* Skip FIR startup transient */
    size_t skip = IPP_HB_NTAPS * 6 * 2;  /* 6 stages, safety margin */
    size_t est_measured = est_out > skip ? est_out - skip : est_out;
    freq = bin_align_freq(1000.0, (double)fs_out, est_measured);

    /* Generate multi-bit sine (NOT ±1.0 DSD) at the DSD rate */
    for (size_t i = 0; i < n_in; i++)
        pcm_in[i] = (float)(0.5 * sin(2.0 * M_PI * freq * (double)i / (double)fs_in));

    /* FIR decimate */
    fir_chain_t fir;
    fir_chain_init(&fir, fs_in, fs_out);
    size_t out_count = fir_chain_process(&fir, pcm_in, pcm_out, n_in);
    fir_chain_free(&fir);

    /* Measure on steady-state portion */
    float *meas = pcm_out + skip;
    size_t meas_n = out_count - skip;
    double out_sig = goertzel_power(meas, meas_n, freq, (double)fs_out);
    double out_amp = sqrt(out_sig * 4.0);
    double sinad = measure_sinad(meas, meas_n, freq, (double)fs_out);

    printf("    PCM sine 2822400->44100: out_amp=%.4f, SINAD=%.1f dB (no-skip: %.1f dB), n=%zu, skip=%zu\n",
           out_amp, sinad,
           measure_sinad(pcm_out, out_count, freq, (double)fs_out),
           meas_n, skip);

    /* Also test single-stage: 88200 -> 44100 */
    size_t n_in2 = 262144;
    float *pcm_in2 = (float *)malloc(n_in2 * sizeof(float));
    float *pcm_out2 = (float *)malloc((n_in2 / 2 + 4096) * sizeof(float));
    if (pcm_in2 && pcm_out2) {
        size_t skip2 = IPP_HB_NTAPS * 2;  /* 1 stage */
        size_t est2 = n_in2 / 2 - skip2;
        double freq2 = bin_align_freq(1000.0, 44100.0, est2);
        for (size_t i = 0; i < n_in2; i++)
            pcm_in2[i] = (float)(0.5 * sin(2.0 * M_PI * freq2 * (double)i / 88200.0));

        fir_chain_t fir2;
        fir_chain_init(&fir2, 88200, 44100);
        size_t out_count2 = fir_chain_process(&fir2, pcm_in2, pcm_out2, n_in2);
        fir_chain_free(&fir2);

        float *meas2 = pcm_out2 + skip2;
        size_t meas_n2 = out_count2 - skip2;
        double sinad2 = measure_sinad(meas2, meas_n2, freq2, 44100.0);

        printf("    PCM sine 88200->44100: SINAD=%.1f dB (no-skip: %.1f dB), n=%zu\n",
               sinad2,
               measure_sinad(pcm_out2, out_count2, freq2, 44100.0),
               meas_n2);
    }

    free(pcm_in); free(pcm_out); free(pcm_in2); free(pcm_out2);
    TEST_ASSERT_TRUE(1, "PCM FIR control diagnostic completed");
}

/* ─── Diagnostic: FIR-only SINAD (no SDM re-encode) ─── */

static void test_diag_fir_only(void) {
    uint32_t pairs[][2] = {
        {DSD_RATE_64,  DSD_RATE_128},
        {DSD_RATE_64,  DSD_RATE_256},
        {DSD_RATE_64,  DSD_RATE_512},
        {DSD_RATE_128, DSD_RATE_256},
        {DSD_RATE_256, DSD_RATE_512},
        {DSD_RATE_512, DSD_RATE_128},
        {DSD_RATE_512, DSD_RATE_64},
    };
    int n_pairs = sizeof(pairs) / sizeof(pairs[0]);

    printf("\n    --- FIR-only SINAD (no SDM re-encode) ---\n");
    for (int p = 0; p < n_pairs; p++) {
        uint32_t fs_in = pairs[p][0], fs_out = pairs[p][1];
        size_t n_in = 262144;
        if (fs_in > DSD_RATE_64) n_in = 524288;
        if (fs_in > DSD_RATE_128) n_in = 1048576;
        if (fs_in > DSD_RATE_256) n_in = 2097152;

        size_t max_out = (fs_out >= fs_in)
            ? n_in * (fs_out / fs_in) + 4096
            : n_in / 2 + 4096;

        float *dsd_in  = (float *)malloc(n_in * sizeof(float));
        float *fir_buf = (float *)malloc(max_out * sizeof(float));
        if (!dsd_in || !fir_buf) { free(dsd_in); free(fir_buf); continue; }

        /* Estimate output count for bin alignment */
        size_t est_produced = n_in - SINAD_TRELLIS_LAT;
        size_t est_fir;
        if (fs_out >= fs_in)
            est_fir = est_produced * (fs_out / fs_in);
        else
            est_fir = est_produced / (fs_in / fs_out);
        double freq = bin_align_freq(1000.0, (double)fs_out, est_fir);

        size_t dsd_count = generate_dsd_sine(fs_in, freq, 0.5, n_in, dsd_in);
        if (dsd_count < 1024) { free(dsd_in); free(fir_buf); continue; }

        fir_chain_t fir;
        fir_chain_init(&fir, fs_in, fs_out);
        size_t fir_count = fir_chain_process(&fir, dsd_in, fir_buf, dsd_count);
        fir_chain_free(&fir);

        /* Signal stats */
        float mn = fir_buf[0], mx = fir_buf[0];
        double rms = 0.0;
        for (size_t i = 0; i < fir_count; i++) {
            if (fir_buf[i] < mn) mn = fir_buf[i];
            if (fir_buf[i] > mx) mx = fir_buf[i];
            rms += (double)fir_buf[i] * (double)fir_buf[i];
        }
        rms = sqrt(rms / (double)fir_count);

        double sinad = measure_sinad(fir_buf, fir_count, freq, (double)fs_out);

        printf("    DSD%u->DSD%u: FIR_SINAD=%.1f dB, range=[%.3f, %.3f], rms=%.4f, n=%zu\n",
               fs_in / 44100, fs_out / 44100, sinad, mn, mx, rms, fir_count);

        free(dsd_in);
        free(fir_buf);
    }
    TEST_ASSERT_TRUE(1, "FIR-only diagnostic completed");
}

/* ─── Diagnostic: SDM limiter sweep ─── */

static double measure_rate_sinad_with_limit(uint32_t fs_in, uint32_t fs_out,
                                             double state_limit) {
    unsigned mult_in = fs_in / 44100;
    size_t n_in;
    if (mult_in <= 64)       n_in = 262144;
    else if (mult_in <= 128) n_in = 524288;
    else if (mult_in <= 256) n_in = 1048576;
    else                     n_in = 2097152;

    size_t max_out;
    if (fs_out >= fs_in)
        max_out = n_in * (fs_out / fs_in) + 4096;
    else
        max_out = n_in / 2 + 4096;

    float *dsd_in  = (float *)malloc(n_in * sizeof(float));
    float *fir_buf = (float *)malloc(max_out * sizeof(float));
    float *dsd_out = (float *)malloc(max_out * sizeof(float));
    if (!dsd_in || !fir_buf || !dsd_out) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    size_t est_in_produced = n_in - SINAD_TRELLIS_LAT;
    size_t est_fir_out;
    if (fs_out >= fs_in)
        est_fir_out = est_in_produced * (fs_out / fs_in);
    else
        est_fir_out = est_in_produced / (fs_in / fs_out);
    size_t est_sdm_out = est_fir_out - SINAD_TRELLIS_LAT;
    double freq = bin_align_freq(1000.0, (double)fs_out, est_sdm_out);
    size_t dsd_in_count = generate_dsd_sine(fs_in, freq, 0.5, n_in, dsd_in);

    if (dsd_in_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    fir_chain_t fir;
    if (fir_chain_init(&fir, fs_in, fs_out) != 0) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    size_t fir_count = fir_chain_process(&fir, dsd_in, fir_buf, dsd_in_count);
    fir_chain_free(&fir);

    if (fir_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    const ntf_filter_t *f_out = ntf_auto_select(fs_out);
    if (!f_out) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    sdm_context_t sdm;
    if (sdm_context_init(&sdm, f_out, SINAD_TRELLIS_DEPTH, SINAD_TRELLIS_CANDS,
                         SINAD_TRELLIS_LAT) != 0) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    sdm.state_limit = state_limit;
    double *sdm_in_d = float_to_double(fir_buf, fir_count);
    if (!sdm_in_d) { sdm_context_free(&sdm); free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    size_t out_count = sdm_process_block(&sdm, sdm_in_d, dsd_out, fir_count);
    free(sdm_in_d);
    sdm_context_free(&sdm);

    if (out_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    double sinad_db = measure_sinad(dsd_out, out_count, freq, (double)fs_out);

    free(dsd_in);
    free(fir_buf);
    free(dsd_out);
    return sinad_db;
}

/* ─── Measure SINAD with specific NTF filter and limiter value ─── */

static double measure_rate_sinad_with_filter_limit(uint32_t fs_in, uint32_t fs_out,
                                                     ntf_filter_id_t filter_id,
                                                     double state_limit) {
    /* Reduced sample counts for sweep — 8x smaller than full tests.
       Sufficient for relative ranking (±2 dB vs full), completes in minutes not hours. */
    unsigned mult_in = fs_in / 44100;
    size_t n_in;
    if (mult_in <= 64)       n_in = 32768;
    else if (mult_in <= 128) n_in = 65536;
    else if (mult_in <= 256) n_in = 131072;
    else                     n_in = 262144;

    size_t max_out;
    if (fs_out >= fs_in)
        max_out = n_in * (fs_out / fs_in) + 4096;
    else
        max_out = n_in / 2 + 4096;

    float *dsd_in  = (float *)malloc(n_in * sizeof(float));
    float *fir_buf = (float *)malloc(max_out * sizeof(float));
    float *dsd_out = (float *)malloc(max_out * sizeof(float));
    if (!dsd_in || !fir_buf || !dsd_out) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    size_t est_in_produced = n_in - SWEEP_TRELLIS_LAT;
    size_t est_fir_out;
    if (fs_out >= fs_in)
        est_fir_out = est_in_produced * (fs_out / fs_in);
    else
        est_fir_out = est_in_produced / (fs_in / fs_out);
    size_t est_sdm_out = est_fir_out - SWEEP_TRELLIS_LAT;
    double freq = bin_align_freq(1000.0, (double)fs_out, est_sdm_out);

    /* Generate input DSD with reduced candidates for speed */
    const ntf_filter_t *f_in = ntf_auto_select(fs_in);
    if (!f_in) { free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    sdm_context_t gen;
    if (sdm_context_init(&gen, f_in, SINAD_TRELLIS_DEPTH, SWEEP_TRELLIS_CANDS,
                         SWEEP_TRELLIS_LAT) != 0) {
        free(dsd_in); free(fir_buf); free(dsd_out); return -999.0;
    }
    double *sine = (double *)malloc(n_in * sizeof(double));
    if (!sine) { sdm_context_free(&gen); free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    for (size_t i = 0; i < n_in; i++)
        sine[i] = 0.5 * sin(2.0 * M_PI * freq * (double)i / (double)fs_in);
    size_t dsd_in_count = sdm_process_block(&gen, sine, dsd_in, n_in);
    free(sine);
    sdm_context_free(&gen);

    if (dsd_in_count < 512) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    fir_chain_t fir;
    if (fir_chain_init(&fir, fs_in, fs_out) != 0) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    size_t fir_count = fir_chain_process(&fir, dsd_in, fir_buf, dsd_in_count);
    fir_chain_free(&fir);

    if (fir_count < 512) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    const ntf_filter_t *f_out = ntf_get_filter(filter_id, fs_out);
    if (!f_out) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    sdm_context_t sdm;
    if (sdm_context_init(&sdm, f_out, SINAD_TRELLIS_DEPTH, SWEEP_TRELLIS_CANDS,
                         SWEEP_TRELLIS_LAT) != 0) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    sdm.state_limit = state_limit;
    double *sdm_in_d = float_to_double(fir_buf, fir_count);
    if (!sdm_in_d) { sdm_context_free(&sdm); free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    size_t out_count = sdm_process_block(&sdm, sdm_in_d, dsd_out, fir_count);
    free(sdm_in_d);
    sdm_context_free(&sdm);

    if (out_count < 512) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    double sinad_db = measure_sinad(dsd_out, out_count, freq, (double)fs_out);

    free(dsd_in);
    free(fir_buf);
    free(dsd_out);
    return sinad_db;
}

static void test_diag_limiter_sweep(void) {
    uint32_t pairs[][2] = {
        {DSD_RATE_64,  DSD_RATE_128},
        {DSD_RATE_64,  DSD_RATE_256},
        {DSD_RATE_64,  DSD_RATE_512},
        {DSD_RATE_128, DSD_RATE_256},
        {DSD_RATE_128, DSD_RATE_512},
        {DSD_RATE_256, DSD_RATE_512},
        {DSD_RATE_128, DSD_RATE_64},
        {DSD_RATE_512, DSD_RATE_64},
        {DSD_RATE_512, DSD_RATE_128},
        {DSD_RATE_512, DSD_RATE_256},
    };
    double limits[] = { 0.0, 6.0, 8.0, 10.0, 12.0, 16.0 };
    int n_pairs = sizeof(pairs) / sizeof(pairs[0]);
    int n_limits = sizeof(limits) / sizeof(limits[0]);

    printf("\n    --- SDM Limiter Sweep (upsample paths) ---\n");
    printf("    %-16s", "Conversion");
    for (int l = 0; l < n_limits; l++) {
        if (limits[l] == 0.0)
            printf("  %8s", "off");
        else
            printf("  lim=%-3.0f", limits[l]);
    }
    printf("\n");

    for (int p = 0; p < n_pairs; p++) {
        uint32_t fs_in = pairs[p][0], fs_out = pairs[p][1];
        printf("    DSD%u->DSD%u", fs_in / 44100, fs_out / 44100);
        int pad = 16 - 10;  /* rough alignment */
        for (int i = 0; i < pad; i++) printf(" ");

        for (int l = 0; l < n_limits; l++) {
            double sinad = measure_rate_sinad_with_limit(fs_in, fs_out, limits[l]);
            printf("  %6.1f", sinad);
        }
        printf(" dB\n");
    }
    TEST_ASSERT_TRUE(1, "Limiter sweep completed");
}

/* ─── Comprehensive NTF × Limiter Sweep ─── */

static void test_comprehensive_ntf_limiter_sweep(void) {
    /* All 12 rate conversion paths */
    static const uint32_t pairs[][2] = {
        /* Upsample */
        {DSD_RATE_64,  DSD_RATE_128},
        {DSD_RATE_64,  DSD_RATE_256},
        {DSD_RATE_64,  DSD_RATE_512},
        {DSD_RATE_128, DSD_RATE_256},
        {DSD_RATE_128, DSD_RATE_512},
        {DSD_RATE_256, DSD_RATE_512},
        /* Downsample */
        {DSD_RATE_128, DSD_RATE_64},
        {DSD_RATE_256, DSD_RATE_64},
        {DSD_RATE_256, DSD_RATE_128},
        {DSD_RATE_512, DSD_RATE_64},
        {DSD_RATE_512, DSD_RATE_128},
        {DSD_RATE_512, DSD_RATE_256},
    };
    static const int n_pairs = sizeof(pairs) / sizeof(pairs[0]);

    /* All 10 NTF filter IDs */
    static const ntf_filter_id_t filter_ids[] = {
        NTF_CLANS_4, NTF_SDM_4,
        NTF_CLANS_5, NTF_SDM_5,
        NTF_CLANS_6, NTF_SDM_6,
        NTF_CLANS_7, NTF_SDM_7,
        NTF_CLANS_8, NTF_SDM_8,
    };
    static const char *filter_names[] = {
        "clans-4", "sdm-4",
        "clans-5", "sdm-5",
        "clans-6", "sdm-6",
        "clans-7", "sdm-7",
        "clans-8", "sdm-8",
    };
    static const int n_filters = sizeof(filter_ids) / sizeof(filter_ids[0]);

    /* Limiter values to sweep */
    static const double limits[] = { 0.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0 };
    static const int n_limits = sizeof(limits) / sizeof(limits[0]);

    printf("\n    ╔══════════════════════════════════════════════════════╗\n");
    printf("    ║  Comprehensive NTF × Limiter Sweep                  ║\n");
    printf("    ║  %d paths × %d filters × %d limits = %d measurements  ║\n",
           n_pairs, n_filters, n_limits, n_pairs * n_filters * n_limits);
    printf("    ╚══════════════════════════════════════════════════════╝\n");

    /* Track best result per path */
    double best_sinad[12];
    int    best_filter[12];
    int    best_limit_idx[12];
    for (int p = 0; p < n_pairs; p++) {
        best_sinad[p] = -999.0;
        best_filter[p] = -1;
        best_limit_idx[p] = -1;
    }

    for (int p = 0; p < n_pairs; p++) {
        uint32_t fs_in = pairs[p][0], fs_out = pairs[p][1];
        const char *dir = (fs_out > fs_in) ? "UP" : "DN";
        printf("\n    --- DSD%u -> DSD%u (%s) ---\n",
               fs_in / 44100, fs_out / 44100, dir);

        /* Header */
        printf("    %-10s", "Filter");
        for (int l = 0; l < n_limits; l++) {
            if (limits[l] == 0.0)
                printf("  %6s", "off");
            else
                printf("  lim=%2.0f", limits[l]);
        }
        printf("   BEST\n");

        for (int f = 0; f < n_filters; f++) {
            printf("    %-10s", filter_names[f]);

            double path_best = -999.0;
            int path_best_l = 0;

            for (int l = 0; l < n_limits; l++) {
                double sinad = measure_rate_sinad_with_filter_limit(
                    fs_in, fs_out, filter_ids[f], limits[l]);
                printf("  %6.1f", sinad);

                if (sinad > path_best) {
                    path_best = sinad;
                    path_best_l = l;
                }
            }

            /* Mark best limiter for this filter */
            if (limits[path_best_l] == 0.0)
                printf("   %.1f@off", path_best);
            else
                printf("   %.1f@%.0f", path_best, limits[path_best_l]);
            printf("\n");

            /* Update overall best for this path */
            if (path_best > best_sinad[p]) {
                best_sinad[p] = path_best;
                best_filter[p] = f;
                best_limit_idx[p] = path_best_l;
            }
        }
    }

    /* ─── Summary: best configuration per path ─── */
    printf("\n    ╔══════════════════════════════════════════════════════╗\n");
    printf("    ║  OPTIMAL CONFIGURATION PER PATH                     ║\n");
    printf("    ╠══════════════════════════════════════════════════════╣\n");
    printf("    ║  %-18s %-10s %-8s %8s     ║\n",
           "Path", "Filter", "Limiter", "SINAD");
    printf("    ╠══════════════════════════════════════════════════════╣\n");
    for (int p = 0; p < n_pairs; p++) {
        uint32_t fs_in = pairs[p][0], fs_out = pairs[p][1];
        char path[20];
        sprintf_s(path, sizeof(path), "DSD%u->DSD%u", fs_in / 44100, fs_out / 44100);

        char lim_str[16];
        if (limits[best_limit_idx[p]] == 0.0)
            sprintf_s(lim_str, sizeof(lim_str), "off");
        else
            sprintf_s(lim_str, sizeof(lim_str), "%.0f", limits[best_limit_idx[p]]);

        printf("    ║  %-18s %-10s %-8s %7.1f dB   ║\n",
               path, filter_names[best_filter[p]], lim_str, best_sinad[p]);
    }
    printf("    ╚══════════════════════════════════════════════════════╝\n");

    TEST_ASSERT_TRUE(1, "Comprehensive NTF x limiter sweep completed");
}

/* ─── Phase 2: Candidates × Latency sweep for winning configs ─── */

static double measure_rate_sinad_cands_lat(uint32_t fs_in, uint32_t fs_out,
                                            ntf_filter_id_t filter_id,
                                            double state_limit,
                                            int cands, int latency) {
    unsigned mult_in = fs_in / 44100;
    size_t n_in;
    if (mult_in <= 64)       n_in = 32768;
    else if (mult_in <= 128) n_in = 65536;
    else if (mult_in <= 256) n_in = 131072;
    else                     n_in = 262144;

    size_t max_out;
    if (fs_out >= fs_in)
        max_out = n_in * (fs_out / fs_in) + 4096;
    else
        max_out = n_in / 2 + 4096;

    float *dsd_in  = (float *)malloc(n_in * sizeof(float));
    float *fir_buf = (float *)malloc(max_out * sizeof(float));
    float *dsd_out = (float *)malloc(max_out * sizeof(float));
    if (!dsd_in || !fir_buf || !dsd_out) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    size_t est_in_produced = n_in - (size_t)latency;
    if (est_in_produced < 1024) { free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    size_t est_fir_out;
    if (fs_out >= fs_in)
        est_fir_out = est_in_produced * (fs_out / fs_in);
    else
        est_fir_out = est_in_produced / (fs_in / fs_out);
    size_t est_sdm_out = est_fir_out - (size_t)latency;
    if (est_sdm_out < 512) { free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    double freq = bin_align_freq(1000.0, (double)fs_out, est_sdm_out);

    /* Generate input DSD */
    const ntf_filter_t *f_in = ntf_auto_select(fs_in);
    if (!f_in) { free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    sdm_context_t gen;
    if (sdm_context_init(&gen, f_in, SINAD_TRELLIS_DEPTH, cands, latency) != 0) {
        free(dsd_in); free(fir_buf); free(dsd_out); return -999.0;
    }
    double *sine = (double *)malloc(n_in * sizeof(double));
    if (!sine) { sdm_context_free(&gen); free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    for (size_t i = 0; i < n_in; i++)
        sine[i] = 0.5 * sin(2.0 * M_PI * freq * (double)i / (double)fs_in);
    size_t dsd_in_count = sdm_process_block(&gen, sine, dsd_in, n_in);
    free(sine);
    sdm_context_free(&gen);

    if (dsd_in_count < 512) { free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }

    /* FIR rate conversion */
    fir_chain_t fir;
    if (fir_chain_init(&fir, fs_in, fs_out) != 0) {
        free(dsd_in); free(fir_buf); free(dsd_out); return -999.0;
    }
    size_t fir_count = fir_chain_process(&fir, dsd_in, fir_buf, dsd_in_count);
    fir_chain_free(&fir);
    if (fir_count < 512) { free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }

    /* SDM re-encode with specified filter/cands/latency */
    const ntf_filter_t *f_out = ntf_get_filter(filter_id, fs_out);
    if (!f_out) { free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    sdm_context_t sdm;
    if (sdm_context_init(&sdm, f_out, SINAD_TRELLIS_DEPTH, cands, latency) != 0) {
        free(dsd_in); free(fir_buf); free(dsd_out); return -999.0;
    }
    sdm.state_limit = state_limit;
    double *sdm_in_d = float_to_double(fir_buf, fir_count);
    if (!sdm_in_d) { sdm_context_free(&sdm); free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    size_t out_count = sdm_process_block(&sdm, sdm_in_d, dsd_out, fir_count);
    free(sdm_in_d);
    sdm_context_free(&sdm);
    if (out_count < 512) { free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }

    double sinad_db = measure_sinad(dsd_out, out_count, freq, (double)fs_out);
    free(dsd_in); free(fir_buf); free(dsd_out);
    return sinad_db;
}

static void test_cands_latency_sweep(void) {
    /* Winners from NTF×Limiter sweep */
    typedef struct {
        uint32_t fs_in, fs_out;
        ntf_filter_id_t filter;
        double limit;
        const char *name;
    } winner_t;

    static const winner_t winners[] = {
        { DSD_RATE_64,  DSD_RATE_128, NTF_CLANS_6, 0.0,  "DSD64->128"  },
        { DSD_RATE_64,  DSD_RATE_256, NTF_SDM_7,   0.0,  "DSD64->256"  },
        { DSD_RATE_64,  DSD_RATE_512, NTF_SDM_8,   10.0, "DSD64->512"  },
        { DSD_RATE_128, DSD_RATE_256, NTF_SDM_4,   12.0, "DSD128->256" },
        { DSD_RATE_128, DSD_RATE_512, NTF_CLANS_8, 12.0, "DSD128->512" },
        { DSD_RATE_256, DSD_RATE_512, NTF_CLANS_8, 6.0,  "DSD256->512" },
        { DSD_RATE_128, DSD_RATE_64,  NTF_CLANS_4, 0.0,  "DSD128->64"  },
        { DSD_RATE_256, DSD_RATE_64,  NTF_CLANS_8, 0.0,  "DSD256->64"  },
        { DSD_RATE_256, DSD_RATE_128, NTF_CLANS_4, 0.0,  "DSD256->128" },
        { DSD_RATE_512, DSD_RATE_64,  NTF_SDM_6,   0.0,  "DSD512->64"  },
        { DSD_RATE_512, DSD_RATE_128, NTF_SDM_4,   16.0, "DSD512->128" },
        { DSD_RATE_512, DSD_RATE_256, NTF_SDM_6,   16.0, "DSD512->256" },
    };
    static const int n_winners = sizeof(winners) / sizeof(winners[0]);

    static const int cands_vals[] = { 4, 8, 16, 32 };
    static const int lat_vals[]   = { 64, 128, 256, 512 };
    static const int n_cands = 4, n_lat = 4;

    printf("\n    ╔══════════════════════════════════════════════════════╗\n");
    printf("    ║  Candidates × Latency Sweep (best NTF+limiter)      ║\n");
    printf("    ║  %d paths × %d cands × %d latencies = %d measurements  ║\n",
           n_winners, n_cands, n_lat, n_winners * n_cands * n_lat);
    printf("    ╚══════════════════════════════════════════════════════╝\n");

    for (int w = 0; w < n_winners; w++) {
        const winner_t *win = &winners[w];
        printf("\n    --- %s (filter=%s, lim=%s) ---\n",
               win->name,
               win->filter == NTF_CLANS_4 ? "clans-4" :
               win->filter == NTF_CLANS_5 ? "clans-5" :
               win->filter == NTF_CLANS_6 ? "clans-6" :
               win->filter == NTF_CLANS_7 ? "clans-7" :
               win->filter == NTF_CLANS_8 ? "clans-8" :
               win->filter == NTF_SDM_4   ? "sdm-4"   :
               win->filter == NTF_SDM_5   ? "sdm-5"   :
               win->filter == NTF_SDM_6   ? "sdm-6"   :
               win->filter == NTF_SDM_7   ? "sdm-7"   :
               win->filter == NTF_SDM_8   ? "sdm-8"   : "?",
               win->limit == 0.0 ? "off" : "on");

        /* Header */
        printf("    %-8s", "cands\\lat");
        for (int l = 0; l < n_lat; l++)
            printf("  lat=%3d", lat_vals[l]);
        printf("   BEST\n");

        double path_best = -999.0;
        int best_c = 0, best_l = 0;

        for (int c = 0; c < n_cands; c++) {
            printf("    cands=%-3d", cands_vals[c]);

            for (int l = 0; l < n_lat; l++) {
                double sinad = measure_rate_sinad_cands_lat(
                    win->fs_in, win->fs_out, win->filter, win->limit,
                    cands_vals[c], lat_vals[l]);
                printf("  %7.1f", sinad);

                if (sinad > path_best) {
                    path_best = sinad;
                    best_c = c;
                    best_l = l;
                }
            }
            printf("\n");
        }
        printf("    BEST: cands=%d, lat=%d -> %.1f dB\n",
               cands_vals[best_c], lat_vals[best_l], path_best);
    }

    TEST_ASSERT_TRUE(1, "Candidates x latency sweep completed");
}

/* ─── Diagnostic: input gain + limiter effect on problem paths ─── */

static double measure_rate_sinad_gain_limit(uint32_t fs_in, uint32_t fs_out,
                                              ntf_filter_id_t filter_id,
                                              float input_gain,
                                              double state_limit) {
    unsigned mult_in = fs_in / 44100;
    size_t n_in;
    if (mult_in <= 64)       n_in = 262144;
    else if (mult_in <= 128) n_in = 524288;
    else if (mult_in <= 256) n_in = 1048576;
    else                     n_in = 2097152;

    size_t max_out;
    if (fs_out >= fs_in)
        max_out = n_in * (fs_out / fs_in) + 4096;
    else
        max_out = n_in / 2 + 4096;

    float *dsd_in  = (float *)malloc(n_in * sizeof(float));
    float *fir_buf = (float *)malloc(max_out * sizeof(float));
    float *dsd_out = (float *)malloc(max_out * sizeof(float));
    if (!dsd_in || !fir_buf || !dsd_out) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    size_t est_in_produced = n_in - SINAD_TRELLIS_LAT;
    size_t est_fir_out;
    if (fs_out >= fs_in)
        est_fir_out = est_in_produced * (fs_out / fs_in);
    else
        est_fir_out = est_in_produced / (fs_in / fs_out);
    size_t est_sdm_out = est_fir_out - SINAD_TRELLIS_LAT;
    double freq = bin_align_freq(1000.0, (double)fs_out, est_sdm_out);
    size_t dsd_in_count = generate_dsd_sine(fs_in, freq, 0.5, n_in, dsd_in);

    if (dsd_in_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    fir_chain_t fir;
    if (fir_chain_init(&fir, fs_in, fs_out) != 0) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    size_t fir_count = fir_chain_process(&fir, dsd_in, fir_buf, dsd_in_count);
    fir_chain_free(&fir);

    if (fir_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    /* Apply input gain to FIR output */
    if (input_gain != 1.0f) {
        for (size_t i = 0; i < fir_count; i++)
            fir_buf[i] *= input_gain;
    }

    const ntf_filter_t *f_out = ntf_get_filter(filter_id, fs_out);
    if (!f_out) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    sdm_context_t sdm;
    if (sdm_context_init(&sdm, f_out, SINAD_TRELLIS_DEPTH, SINAD_TRELLIS_CANDS,
                         SINAD_TRELLIS_LAT) != 0) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    sdm.state_limit = state_limit;
    double *sdm_in_d = float_to_double(fir_buf, fir_count);
    if (!sdm_in_d) { sdm_context_free(&sdm); free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    size_t out_count = sdm_process_block(&sdm, sdm_in_d, dsd_out, fir_count);
    free(sdm_in_d);
    sdm_context_free(&sdm);

    if (out_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    double sinad_db = measure_sinad(dsd_out, out_count, freq, (double)fs_out);
    free(dsd_in); free(fir_buf); free(dsd_out);
    return sinad_db;
}

static void test_diag_gain_limiter(void) {
    typedef struct {
        uint32_t fs_in, fs_out;
        const char *name;
    } path_t;

    static const path_t paths[] = {
        { DSD_RATE_64,  DSD_RATE_128, "DSD64->128"  },
        { DSD_RATE_64,  DSD_RATE_256, "DSD64->256"  },
        { DSD_RATE_64,  DSD_RATE_512, "DSD64->512"  },
        { DSD_RATE_128, DSD_RATE_256, "DSD128->256" },
        { DSD_RATE_128, DSD_RATE_512, "DSD128->512" },
        { DSD_RATE_256, DSD_RATE_512, "DSD256->512" },
    };
    int n_paths = sizeof(paths) / sizeof(paths[0]);

    static const float gains[] = { 1.0f, 0.5f, 0.25f };
    static const double limits[] = { 0.0, 10.0 };
    static const ntf_filter_id_t filters[] = { NTF_CLANS_6, NTF_CLANS_8 };
    static const char *fnames[] = { "CLANS6", "CLANS8" };

    printf("\n    --- Input Gain × Limiter × NTF diagnostic ---\n");

    for (int p = 0; p < n_paths; p++) {
        printf("\n    %s:\n", paths[p].name);
        printf("    %-7s %-5s %-5s  SINAD\n", "Filter", "Gain", "Limit");

        for (int f = 0; f < 2; f++) {
            for (int g = 0; g < 3; g++) {
                for (int l = 0; l < 2; l++) {
                    double sinad = measure_rate_sinad_gain_limit(
                        paths[p].fs_in, paths[p].fs_out,
                        filters[f], gains[g], limits[l]);
                    printf("    %-7s %-5.2f %-5s  %.1f dB\n",
                           fnames[f], gains[g],
                           limits[l] == 0.0 ? "off" : "10.0",
                           sinad);
                }
            }
        }
    }
    TEST_ASSERT_TRUE(1, "Gain/limiter diagnostic completed");
}

/* ─── Gain Sweep: NTF × fir_gain × state_limit for problematic paths ─── */

/* Production-like settings */
#define GAIN_SWEEP_DEPTH  4
#define GAIN_SWEEP_CANDS  2
#define GAIN_SWEEP_LAT    512

static double measure_gain_sweep_sinad(uint32_t fs_in, uint32_t fs_out,
                                        ntf_filter_id_t filter_id,
                                        float fir_gain,
                                        double state_limit) {
    unsigned mult_in = fs_in / 44100;
    size_t n_in;
    if (mult_in <= 64)       n_in = 262144;
    else if (mult_in <= 128) n_in = 524288;
    else if (mult_in <= 256) n_in = 1048576;
    else                     n_in = 2097152;

    size_t max_out;
    if (fs_out >= fs_in)
        max_out = n_in * (fs_out / fs_in) + 4096;
    else
        max_out = n_in / 2 + 4096;

    float *dsd_in  = (float *)malloc(n_in * sizeof(float));
    float *fir_buf = (float *)malloc(max_out * sizeof(float));
    float *dsd_out = (float *)malloc(max_out * sizeof(float));
    if (!dsd_in || !fir_buf || !dsd_out) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    size_t est_in_produced = n_in - GAIN_SWEEP_LAT;
    size_t est_fir_out;
    if (fs_out >= fs_in)
        est_fir_out = est_in_produced * (fs_out / fs_in);
    else
        est_fir_out = est_in_produced / (fs_in / fs_out);
    size_t est_sdm_out = est_fir_out - GAIN_SWEEP_LAT;
    double freq = bin_align_freq(1000.0, (double)fs_out, est_sdm_out);
    size_t dsd_in_count = generate_dsd_sine(fs_in, freq, 0.5, n_in, dsd_in);

    if (dsd_in_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    fir_chain_t fir;
    if (fir_chain_init(&fir, fs_in, fs_out) != 0) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    size_t fir_count = fir_chain_process(&fir, dsd_in, fir_buf, dsd_in_count);
    fir_chain_free(&fir);

    if (fir_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    /* Apply FIR gain */
    if (fir_gain != 1.0f) {
        for (size_t i = 0; i < fir_count; i++)
            fir_buf[i] *= fir_gain;
    }

    const ntf_filter_t *f_out = ntf_get_filter(filter_id, fs_out);
    if (!f_out) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    sdm_context_t sdm;
    if (sdm_context_init(&sdm, f_out, GAIN_SWEEP_DEPTH, GAIN_SWEEP_CANDS,
                         GAIN_SWEEP_LAT) != 0) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    if (state_limit > 0.0)
        sdm.state_limit = state_limit;
    double *sdm_in_d = float_to_double(fir_buf, fir_count);
    if (!sdm_in_d) { sdm_context_free(&sdm); free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    size_t out_count = sdm_process_block(&sdm, sdm_in_d, dsd_out, fir_count);
    free(sdm_in_d);
    sdm_context_free(&sdm);

    if (out_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    double sinad_db = measure_sinad(dsd_out, out_count, freq, (double)fs_out);
    free(dsd_in); free(fir_buf); free(dsd_out);
    return sinad_db;
}

static void test_gain_sweep(void) {
    typedef struct {
        uint32_t fs_in, fs_out;
        const char *name;
    } path_t;

    static const path_t paths[] = {
        { DSD_RATE_64,  DSD_RATE_128, "DSD64->DSD128"  },
        { DSD_RATE_128, DSD_RATE_512, "DSD128->DSD512"  },
        { DSD_RATE_64,  DSD_RATE_512, "DSD64->DSD512"  },
    };
    static const int n_paths = sizeof(paths) / sizeof(paths[0]);

    static const ntf_filter_id_t filter_ids[] = {
        NTF_CLANS_4, NTF_SDM_4,
        NTF_CLANS_5, NTF_SDM_5,
        NTF_CLANS_6, NTF_SDM_6,
        NTF_CLANS_7, NTF_SDM_7,
        NTF_CLANS_8, NTF_SDM_8,
    };
    static const char *filter_names[] = {
        "clans-4", "sdm-4",
        "clans-5", "sdm-5",
        "clans-6", "sdm-6",
        "clans-7", "sdm-7",
        "clans-8", "sdm-8",
    };
    static const int n_filters = sizeof(filter_ids) / sizeof(filter_ids[0]);

    static const float gains[] = { 1.0f, 0.9f, 0.8f, 0.71f };
    static const int n_gains = sizeof(gains) / sizeof(gains[0]);

    static const double limits[] = { 0.0, 3.0, 6.0, 10.0, 15.0, 20.0 };
    static const int n_limits = sizeof(limits) / sizeof(limits[0]);

    int total = n_paths * n_filters * n_gains * n_limits;
    printf("\n    ╔══════════════════════════════════════════════════════════════╗\n");
    printf("    ║  Gain Sweep: problematic paths (cands=%d, depth=%d, lat=%d)  ║\n",
           GAIN_SWEEP_CANDS, GAIN_SWEEP_DEPTH, GAIN_SWEEP_LAT);
    printf("    ║  %d paths x %d filters x %d gains x %d limits = %d combos       ║\n",
           n_paths, n_filters, n_gains, n_limits, total);
    printf("    ║  Only showing results with SINAD > 50 dB                    ║\n");
    printf("    ╚══════════════════════════════════════════════════════════════╝\n");

    /* Track best per path */
    double best_sinad[3] = { -999.0, -999.0, -999.0 };
    int    best_filter[3], best_gain[3], best_limit[3];
    memset(best_filter, 0, sizeof(best_filter));
    memset(best_gain, 0, sizeof(best_gain));
    memset(best_limit, 0, sizeof(best_limit));

    int count = 0;
    for (int p = 0; p < n_paths; p++) {
        printf("\n    --- %s ---\n", paths[p].name);
        printf("    %-10s %-6s %-6s  SINAD\n", "Filter", "Gain", "Limit");

        int hits = 0;
        for (int f = 0; f < n_filters; f++) {
            for (int g = 0; g < n_gains; g++) {
                for (int l = 0; l < n_limits; l++) {
                    double sinad = measure_gain_sweep_sinad(
                        paths[p].fs_in, paths[p].fs_out,
                        filter_ids[f], gains[g], limits[l]);
                    count++;

                    if (sinad > 50.0) {
                        char lim_str[16];
                        if (limits[l] == 0.0)
                            sprintf_s(lim_str, sizeof(lim_str), "off");
                        else
                            sprintf_s(lim_str, sizeof(lim_str), "%.1f", limits[l]);

                        printf("    %-10s %-6.2f %-6s  %.1f dB\n",
                               filter_names[f], gains[g], lim_str, sinad);
                        hits++;
                    }

                    if (sinad > best_sinad[p]) {
                        best_sinad[p] = sinad;
                        best_filter[p] = f;
                        best_gain[p] = g;
                        best_limit[p] = l;
                    }
                }
            }
        }
        if (hits == 0)
            printf("    (no combinations exceeded 50 dB)\n");

        printf("    [%d/%d done]\n", count, total);
    }

    /* Summary */
    printf("\n    ╔══════════════════════════════════════════════════════════════╗\n");
    printf("    ║  BEST CONFIGURATION PER PATH                                ║\n");
    printf("    ╠══════════════════════════════════════════════════════════════╣\n");
    printf("    ║  %-16s %-10s %-6s %-6s %8s         ║\n",
           "Path", "Filter", "Gain", "Limit", "SINAD");
    printf("    ╠══════════════════════════════════════════════════════════════╣\n");
    for (int p = 0; p < n_paths; p++) {
        char lim_str[16];
        if (limits[best_limit[p]] == 0.0)
            sprintf_s(lim_str, sizeof(lim_str), "off");
        else
            sprintf_s(lim_str, sizeof(lim_str), "%.1f", limits[best_limit[p]]);

        printf("    ║  %-16s %-10s %-6.2f %-6s %7.1f dB       ║\n",
               paths[p].name, filter_names[best_filter[p]],
               gains[best_gain[p]], lim_str, best_sinad[p]);
    }
    printf("    ╚══════════════════════════════════════════════════════════════╝\n");

    TEST_ASSERT_TRUE(1, "Gain sweep completed");
}

/* ─── Weak paths sweep with cands/depth as parameters ─── */

static double measure_weak_path_sinad(uint32_t fs_in, uint32_t fs_out,
                                       ntf_filter_id_t filter_id,
                                       float fir_gain,
                                       double state_limit,
                                       int cands, int depth, int lat) {
    unsigned mult_in = fs_in / 44100;
    size_t n_in;
    if (mult_in <= 64)       n_in = 262144;
    else if (mult_in <= 128) n_in = 524288;
    else if (mult_in <= 256) n_in = 1048576;
    else                     n_in = 2097152;

    size_t max_out;
    if (fs_out >= fs_in)
        max_out = n_in * (fs_out / fs_in) + 4096;
    else
        max_out = n_in / 2 + 4096;

    float *dsd_in  = (float *)malloc(n_in * sizeof(float));
    float *fir_buf = (float *)malloc(max_out * sizeof(float));
    float *dsd_out = (float *)malloc(max_out * sizeof(float));
    if (!dsd_in || !fir_buf || !dsd_out) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    /* Approximate frequency alignment. Exact alignment done post-SDM
     * using actual output count (avoids multi-stage FIR delay estimation). */
    size_t est_in = n_in - SINAD_TRELLIS_LAT;
    size_t est_fir = (fs_out >= fs_in) ?
        est_in * (fs_out / fs_in) : est_in / (fs_in / fs_out);
    size_t est_sdm = (est_fir > (size_t)lat) ? est_fir - (size_t)lat : 1024;
    double freq = bin_align_freq(1000.0, (double)fs_out, est_sdm);
    size_t dsd_in_count = generate_dsd_sine(fs_in, freq, 0.5, n_in, dsd_in);

    if (dsd_in_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    fir_chain_t fir;
    if (fir_chain_init(&fir, fs_in, fs_out) != 0) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    size_t fir_count = fir_chain_process(&fir, dsd_in, fir_buf, dsd_in_count);
    fir_chain_free(&fir);

    if (fir_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    /* Apply FIR gain */
    if (fir_gain != 1.0f) {
        for (size_t i = 0; i < fir_count; i++)
            fir_buf[i] *= fir_gain;
    }

    const ntf_filter_t *f_out = ntf_get_filter(filter_id, fs_out);
    if (!f_out) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    sdm_context_t sdm;
    if (sdm_context_init(&sdm, f_out, depth, cands, lat) != 0) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    if (state_limit > 0.0)
        sdm.state_limit = state_limit;
    double *sdm_in_d = float_to_double(fir_buf, fir_count);
    if (!sdm_in_d) { sdm_context_free(&sdm); free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    size_t out_count = sdm_process_block(&sdm, sdm_in_d, dsd_out, fir_count);
    free(sdm_in_d);
    sdm_context_free(&sdm);

    if (out_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    double sinad_db = measure_sinad(dsd_out, out_count, freq, (double)fs_out);
    free(dsd_in); free(fir_buf); free(dsd_out);
    return sinad_db;
}

static void test_weak_paths_sweep(void) {
    typedef struct {
        uint32_t fs_in, fs_out;
        const char *name;
    } path_t;

    static const path_t paths[] = {
        { DSD_RATE_64,  DSD_RATE_512, "DSD64->DSD512"   },
        { DSD_RATE_128, DSD_RATE_512, "DSD128->DSD512"  },
        { DSD_RATE_128, DSD_RATE_64,  "DSD128->DSD64"   },
        { DSD_RATE_256, DSD_RATE_64,  "DSD256->DSD64"   },
        { DSD_RATE_512, DSD_RATE_64,  "DSD512->DSD64"   },
        { DSD_RATE_256, DSD_RATE_128, "DSD256->DSD128"  },
    };
    static const int n_paths = sizeof(paths) / sizeof(paths[0]);

    static const ntf_filter_id_t filter_ids[] = {
        NTF_CLANS_4, NTF_SDM_4,
        NTF_CLANS_5, NTF_SDM_5,
        NTF_CLANS_6, NTF_SDM_6,
        NTF_CLANS_7, NTF_SDM_7,
        NTF_CLANS_8, NTF_SDM_8,
    };
    static const char *filter_names[] = {
        "clans-4", "sdm-4",
        "clans-5", "sdm-5",
        "clans-6", "sdm-6",
        "clans-7", "sdm-7",
        "clans-8", "sdm-8",
    };
    static const int n_filters = sizeof(filter_ids) / sizeof(filter_ids[0]);

    static const double limits[] = { 0.0, 3.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0 };
    static const int n_limits = sizeof(limits) / sizeof(limits[0]);

    static const int cands_vals[] = { 2, 4, 8 };
    static const int n_cands = sizeof(cands_vals) / sizeof(cands_vals[0]);

    static const int depth_vals[] = { 4, 8, 16 };
    static const int n_depths = sizeof(depth_vals) / sizeof(depth_vals[0]);

    const float gain = 0.708f;
    const int lat = 512;

    int combos_per_path = n_filters * n_limits * n_cands * n_depths;
    int total = n_paths * combos_per_path;

    printf("\n    ╔══════════════════════════════════════════════════════════════════╗\n");
    printf("    ║  Weak Paths Sweep: gain=%.3f, lat=%d                           ║\n",
           gain, lat);
    printf("    ║  %d paths x %d filters x %d limits x %d cands x %d depths = %d  ║\n",
           n_paths, n_filters, n_limits, n_cands, n_depths, total);
    printf("    ║  Showing top 5 results per path (best SINAD)                   ║\n");
    printf("    ╚══════════════════════════════════════════════════════════════════╝\n");

    #define WEAK_TOP_N 5

    for (int p = 0; p < n_paths; p++) {
        /* Top-N tracking */
        double top_sinad[WEAK_TOP_N];
        int    top_filter[WEAK_TOP_N];
        int    top_limit[WEAK_TOP_N];
        int    top_cands[WEAK_TOP_N];
        int    top_depth[WEAK_TOP_N];
        for (int i = 0; i < WEAK_TOP_N; i++)
            top_sinad[i] = -999.0;

        int count = 0;
        for (int fi = 0; fi < n_filters; fi++) {
            for (int li = 0; li < n_limits; li++) {
                for (int ci = 0; ci < n_cands; ci++) {
                    for (int di = 0; di < n_depths; di++) {
                        double sinad = measure_weak_path_sinad(
                            paths[p].fs_in, paths[p].fs_out,
                            filter_ids[fi], gain, limits[li],
                            cands_vals[ci], depth_vals[di], lat);
                        count++;

                        /* Insert into top-N if better than worst */
                        if (sinad > top_sinad[WEAK_TOP_N - 1]) {
                            top_sinad[WEAK_TOP_N - 1]  = sinad;
                            top_filter[WEAK_TOP_N - 1] = fi;
                            top_limit[WEAK_TOP_N - 1]  = li;
                            top_cands[WEAK_TOP_N - 1]  = ci;
                            top_depth[WEAK_TOP_N - 1]  = di;
                            /* Bubble up */
                            for (int k = WEAK_TOP_N - 1; k > 0; k--) {
                                if (top_sinad[k] > top_sinad[k - 1]) {
                                    double ts = top_sinad[k];
                                    int tf = top_filter[k], tl = top_limit[k];
                                    int tc = top_cands[k], td = top_depth[k];
                                    top_sinad[k]  = top_sinad[k - 1];
                                    top_filter[k] = top_filter[k - 1];
                                    top_limit[k]  = top_limit[k - 1];
                                    top_cands[k]  = top_cands[k - 1];
                                    top_depth[k]  = top_depth[k - 1];
                                    top_sinad[k - 1]  = ts;
                                    top_filter[k - 1] = tf;
                                    top_limit[k - 1]  = tl;
                                    top_cands[k - 1]  = tc;
                                    top_depth[k - 1]  = td;
                                } else break;
                            }
                        }
                    }
                }
            }
        }

        printf("\n    --- %s (top %d of %d combos) ---\n",
               paths[p].name, WEAK_TOP_N, combos_per_path);
        printf("    %-10s %-6s %-6s %-6s  SINAD\n",
               "Filter", "Limit", "Cands", "Depth");
        for (int i = 0; i < WEAK_TOP_N; i++) {
            if (top_sinad[i] <= -999.0) break;
            char lim_str[16];
            if (limits[top_limit[i]] == 0.0)
                sprintf_s(lim_str, sizeof(lim_str), "off");
            else
                sprintf_s(lim_str, sizeof(lim_str), "%.1f", limits[top_limit[i]]);

            printf("    %-10s %-6s %-6d %-6d  %.1f dB\n",
                   filter_names[top_filter[i]], lim_str,
                   cands_vals[top_cands[i]], depth_vals[top_depth[i]],
                   top_sinad[i]);
        }
        printf("    [%d/%d paths done]\n", p + 1, n_paths);
    }

    #undef WEAK_TOP_N

    TEST_ASSERT_TRUE(1, "Weak paths sweep completed");
}

/* ─── DSD/48 same-rate tests ─── */

/* Same-rate tests use sinad_measure (the proven quality suite measurement)
 * instead of measure_rate_sinad which was broken for same-rate paths
 * (no FIR lowpass smoothing, raw DSD Goertzel spectral leakage). */
static void test_sinad_same_rate(uint32_t rate, const char *name, double min_sinad) {
    engine_path_info_t pi;
    dsd_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.fs_in = rate; cfg.fs_out = rate;
    engine_get_path_info(rate, rate, NTF_AUTO, SDM_MODE_TRELLIS, &cfg, &pi);
    int cands = pi.cands > 0 ? pi.cands : 2;
    int lat = pi.lat > 0 ? pi.lat : 32;
    int depth = pi.depth > 0 ? pi.depth : 4;
    int ntf_id = pi.ntf_filter;
    sinad_result_t r;
    memset(&r, 0, sizeof(r));
    sinad_measure(rate, ntf_id, cands, depth, lat, 1, pi.fir_gain, &r);
    unsigned base = rate_is_48k_family(rate) ? 48000 : 44100;
    printf("    [SINAD] DSD%u->DSD%u (same): SINAD=%.1f dB A-wtd=%.1f MT=%.1f NMod=%.1f"
           "  [%s, gain=%.2f, cands=%d, lat=%d]\n",
           rate/base, rate/base, r.sinad_theoretical, r.sinad_awtd_theo,
           r.multitone_sinad_db, r.noise_mod_db,
           ntf_id != NTF_AUTO ? "path" : "auto", pi.fir_gain, cands, lat);
    TEST_ASSERT_TRUE(r.sinad_theoretical > min_sinad, name);
}

static void test_sinad_dsd64_48_same(void) {
    double s = measure_rate_sinad(DSD48_RATE_64, DSD48_RATE_64);
    TEST_ASSERT_TRUE(s > 55.0, "DSD64/48 same-rate");
}
static void test_sinad_dsd128_48_same(void) {
    double s = measure_rate_sinad(DSD48_RATE_128, DSD48_RATE_128);
    TEST_ASSERT_TRUE(s > 70.0, "DSD128/48 same-rate");
}
static void test_sinad_dsd256_48_same(void) {
    double s = measure_rate_sinad(DSD48_RATE_256, DSD48_RATE_256);
    TEST_ASSERT_TRUE(s > 90.0, "DSD256/48 same-rate");
}
static void test_sinad_dsd512_48_same(void) {
    double s = measure_rate_sinad(DSD48_RATE_512, DSD48_RATE_512);
    TEST_ASSERT_TRUE(s > 90.0, "DSD512/48 same-rate");
}

/* ─── DSD/48 upsample tests ─── */

static void test_sinad_up_64_48_128_48(void) {
    double sinad = measure_rate_sinad(DSD48_RATE_64, DSD48_RATE_128);
    TEST_ASSERT_TRUE(sinad > 12.0, "DSD64/48->DSD128/48 SINAD > 12 dB");
}

static void test_sinad_up_64_48_256_48(void) {
    double sinad = measure_rate_sinad(DSD48_RATE_64, DSD48_RATE_256);
    TEST_ASSERT_TRUE(sinad > 12.0, "DSD64/48->DSD256/48 SINAD > 12 dB");
}

static void test_sinad_up_128_48_256_48(void) {
    double sinad = measure_rate_sinad(DSD48_RATE_128, DSD48_RATE_256);
    TEST_ASSERT_TRUE(sinad > 12.0, "DSD128/48->DSD256/48 SINAD > 12 dB");
}

/* ─── DSD/48 downsample tests ─── */

static void test_sinad_dn_128_48_64_48(void) {
    double sinad = measure_rate_sinad(DSD48_RATE_128, DSD48_RATE_64);
    TEST_ASSERT_TRUE(sinad > 12.0, "DSD128/48->DSD64/48 SINAD > 12 dB");
}

static void test_sinad_dn_256_48_128_48(void) {
    double sinad = measure_rate_sinad(DSD48_RATE_256, DSD48_RATE_128);
    TEST_ASSERT_TRUE(sinad > 12.0, "DSD256/48->DSD128/48 SINAD > 12 dB");
}

static void test_sinad_dn_256_48_64_48(void) {
    double sinad = measure_rate_sinad(DSD48_RATE_256, DSD48_RATE_64);
    TEST_ASSERT_TRUE(sinad > 12.0, "DSD256/48->DSD64/48 SINAD > 12 dB");
}

/* ─── DSD/48 → PCM tests ─── */

static void test_sinad_dsd64_48_pcm48(void) {
    double sinad = measure_dsd_to_pcm_sinad(DSD48_RATE_64, 48000);
    TEST_ASSERT_TRUE(sinad > 90.0, "DSD64/48->PCM48k SINAD > 90 dB");
}

static void test_sinad_dsd128_48_pcm96(void) {
    double sinad = measure_dsd_to_pcm_sinad(DSD48_RATE_128, 96000);
    TEST_ASSERT_TRUE(sinad > 90.0, "DSD128/48->PCM96k SINAD > 90 dB");
}

/* ─── Suites ─── */

void test_rate_sinad_suite(void) {
    TEST_SUITE("Rate Conversion SINAD");

    /* DSD/44 same-rate — full pipeline: DSD→FIR lowpass→SDM→Goertzel (median) */
    { double s = measure_rate_sinad(DSD_RATE_64,  DSD_RATE_64);
      TEST_ASSERT_TRUE(s > 55.0, "DSD64 same-rate"); }
    { double s = measure_rate_sinad(DSD_RATE_128, DSD_RATE_128);
      TEST_ASSERT_TRUE(s > 70.0, "DSD128 same-rate"); }
    { double s = measure_rate_sinad(DSD_RATE_256, DSD_RATE_256);
      TEST_ASSERT_TRUE(s > 90.0, "DSD256 same-rate"); }
    { double s = measure_rate_sinad(DSD_RATE_512, DSD_RATE_512);
      TEST_ASSERT_TRUE(s > 100.0, "DSD512 same-rate"); }

    TEST_RUN(test_sinad_up_64_128);
    TEST_RUN(test_sinad_up_64_256);
    TEST_RUN(test_sinad_up_64_512);
    TEST_RUN(test_sinad_up_128_256);
    TEST_RUN(test_sinad_up_128_512);
    TEST_RUN(test_sinad_up_256_512);
    TEST_RUN(test_sinad_dn_128_64);
    TEST_RUN(test_sinad_dn_256_64);
    TEST_RUN(test_sinad_dn_512_64);
    TEST_RUN(test_sinad_dn_256_128);
    TEST_RUN(test_sinad_dn_512_128);
    TEST_RUN(test_sinad_dn_512_256);
    TEST_RUN(test_sinad_dsd64_pcm44);
    TEST_RUN(test_sinad_dsd64_pcm88);
    TEST_RUN(test_sinad_dsd64_pcm176);
    TEST_RUN(test_sinad_dsd128_pcm44);
    TEST_RUN(test_sinad_dsd128_pcm88);
    TEST_RUN(test_sinad_dsd128_pcm176);
    TEST_RUN(test_sinad_dsd256_pcm44);
    TEST_RUN(test_sinad_dsd256_pcm88);
    TEST_RUN(test_sinad_dsd256_pcm176);
    TEST_RUN(test_sinad_dsd512_pcm44);
    TEST_RUN(test_sinad_dsd512_pcm88);
    TEST_RUN(test_sinad_dsd512_pcm176);
    TEST_RUN(test_sinad_dsd512_pcm352);
    /* DSD/48 tests */
    TEST_RUN(test_sinad_dsd64_48_same);
    TEST_RUN(test_sinad_dsd128_48_same);
    TEST_RUN(test_sinad_dsd256_48_same);
    TEST_RUN(test_sinad_dsd512_48_same);
    TEST_RUN(test_sinad_up_64_48_128_48);
    TEST_RUN(test_sinad_up_64_48_256_48);
    TEST_RUN(test_sinad_up_128_48_256_48);
    TEST_RUN(test_sinad_dn_128_48_64_48);
    TEST_RUN(test_sinad_dn_256_48_128_48);
    TEST_RUN(test_sinad_dn_256_48_64_48);
    TEST_RUN(test_sinad_dsd64_48_pcm48);
    TEST_RUN(test_sinad_dsd128_48_pcm96);
}

/* ─── Quick depth+NTF+nc sweep for weak rate conversion paths ─── */
static void test_depth16_spot_check(void) {
    typedef struct {
        uint32_t fs_in, fs_out;
        double limit;
        const char *name;
    } spot_t;

    /* Rate conversion paths to re-sweep */
    static const spot_t spots[] = {
        /* Downsample → DSD64 (weakest, ~85 dB) */
        { DSD_RATE_128, DSD_RATE_64,  0.0,  "DSD128->64"  },
        { DSD_RATE_256, DSD_RATE_64,  0.0,  "DSD256->64"  },
        { DSD_RATE_512, DSD_RATE_64,  0.0,  "DSD512->64"  },
        /* Downsample → DSD128 */
        { DSD_RATE_256, DSD_RATE_128, 0.0,  "DSD256->128" },
        { DSD_RATE_512, DSD_RATE_128, 16.0, "DSD512->128" },
        /* Upsample (weaker ones) */
        { DSD_RATE_64,  DSD_RATE_128, 0.0,  "DSD64->128"  },
    };
    int n_spots = sizeof(spots) / sizeof(spots[0]);

    /* NTFs to test */
    static const ntf_filter_id_t test_ntfs[] = {
        NTF_CLANS_4, NTF_CLANS_5, NTF_CLANS_6, NTF_SDM_4, NTF_SDM_6,
    };
    static const char *ntf_names[] = {
        "clans-4", "clans-5", "clans-6", "sdm-4", "sdm-6",
    };
    int n_ntfs = sizeof(test_ntfs) / sizeof(test_ntfs[0]);

    static const int depths[] = { 4, 8, 16 };
    int n_depths = 3;

    /* nc values: nc=2 (same-rate optimal for DSD64) + nc=4/8 */
    static const int nc_vals[] = { 2, 4, 8 };
    int n_nc = 3;

    int total = n_spots * n_ntfs * n_depths * n_nc;
    printf("\n    ╔══════════════════════════════════════════════════════╗\n");
    printf("    ║  Depth+NTF+nc Sweep (DSD-level Goertzel)            ║\n");
    printf("    ║  %d paths × %d NTFs × %d depths × %d nc = %d meas   ║\n",
           n_spots, n_ntfs, n_depths, n_nc, total);
    printf("    ╚══════════════════════════════════════════════════════╝\n");

    for (int s = 0; s < n_spots; s++) {
        const spot_t *sp = &spots[s];
        printf("\n    --- %s (lim=%.0f) ---\n", sp->name, sp->limit);

        double path_best = -999.0;
        const char *best_ntf = "?";
        int best_depth = 0, best_nc = 0;

        for (int ci = 0; ci < n_nc; ci++) {
            int nc = nc_vals[ci];
            int lat = nc * 8;
            if (lat < 32) lat = 32;

            printf("    nc=%-2d %-10s  d=4     d=8     d=16\n", nc, "NTF");
            for (int f = 0; f < n_ntfs; f++) {
                printf("    nc=%-2d %-10s", nc, ntf_names[f]);

                for (int d = 0; d < n_depths; d++) {
                    double sinad = measure_weak_path_sinad(
                        sp->fs_in, sp->fs_out, test_ntfs[f],
                        0.708f, sp->limit, nc, depths[d], lat);
                    printf("  %6.1f", sinad);

                    if (sinad > path_best) {
                        path_best = sinad;
                        best_ntf = ntf_names[f];
                        best_depth = depths[d];
                        best_nc = nc;
                    }
                }
                printf("\n");
            }
        }

        printf("    >>> BEST: %s nc=%d d=%d → %.1f dB\n",
               best_ntf, best_nc, best_depth, path_best);
    }

    TEST_ASSERT_TRUE(1, "Depth+NTF+nc sweep completed");
}

static void sweep_48k_rate(uint32_t rate, const char *rate_name) {
    static const ntf_filter_id_t ntfs[] = {
        NTF_CLANS_4, NTF_SDM_4, NTF_CLANS_5, NTF_SDM_5,
        NTF_CLANS_6, NTF_SDM_6, NTF_CLANS_7, NTF_SDM_7,
        NTF_CLANS_8, NTF_SDM_8,
    };
    static const char *names[] = {
        "clans-4","sdm-4","clans-5","sdm-5",
        "clans-6","sdm-6","clans-7","sdm-7",
        "clans-8","sdm-8",
    };
    static const int depths[] = { 4, 8, 16 };
    static const int lats[] = { 16, 32, 64, 128 };

    printf("\n    %s NTF × Depth × Lat sweep (nc=2, gain=0.708)\n", rate_name);
    double best = -999.0;
    const char *best_ntf = "?";
    int best_d = 0, best_l = 0;

    for (int li = 0; li < 4; li++) {
        printf("\n    lat=%d:\n    %-10s  d=4     d=8     d=16\n", lats[li], "NTF");
        for (int f = 0; f < 10; f++) {
            printf("    %-10s", names[f]);
            for (int d = 0; d < 3; d++) {
                sinad_result_t r;
                sinad_measure(rate, ntfs[f], 2, depths[d], lats[li], 1, 0.708f, &r);
                double s = r.ok ? r.sinad_theoretical : -999.0;
                printf("  %6.1f", s);
                if (s > best) { best = s; best_ntf = names[f]; best_d = depths[d]; best_l = lats[li]; }
            }
            printf("\n");
        }
    }
    printf("\n    >>> %s BEST: %s d=%d lat=%d → %.1f dB\n",
           rate_name, best_ntf, best_d, best_l, best);
}

static void test_48k_ntf_sweep(void) {
    sweep_48k_rate(DSD48_RATE_64,  "DSD64/48");
    sweep_48k_rate(DSD48_RATE_128, "DSD128/48");
    sweep_48k_rate(DSD48_RATE_256, "DSD256/48");
    sweep_48k_rate(DSD48_RATE_512, "DSD512/48");
    TEST_ASSERT_TRUE(1, "48k family NTF sweep completed");
}

/* ─── PreCorr intermediate step experiment ───
 * Compare direct FIR downsample vs PreCorr intermediate for weak paths.
 * Path A: DSD_in → FIR(full ratio) → Trellis@DSD_out
 * Path B: DSD_in → FIR(half) → PreCorr@DSD_mid → FIR(half) → Trellis@DSD_out */
static void test_precorr_intermediate(void) {
    #include "../include/precorr.h"

    typedef struct {
        uint32_t fs_in, fs_mid, fs_out;
        const char *name;
    } path_t;

    static const path_t paths[] = {
        { DSD_RATE_256, DSD_RATE_128, DSD_RATE_64, "DSD256→128→64" },
        { DSD_RATE_512, DSD_RATE_256, DSD_RATE_128, "DSD512→256→128" },
        { DSD_RATE_128, 0, DSD_RATE_64, "DSD128→64 (no mid)" },
    };
    int n_paths = sizeof(paths) / sizeof(paths[0]);

    printf("\n    ╔══════════════════════════════════════════════════════╗\n");
    printf("    ║  PreCorr Intermediate Step Experiment                ║\n");
    printf("    ╚══════════════════════════════════════════════════════╝\n");

    for (int p = 0; p < n_paths; p++) {
        uint32_t fs_in = paths[p].fs_in;
        uint32_t fs_out = paths[p].fs_out;
        uint32_t fs_mid = paths[p].fs_mid;

        /* Generate clean DSD at fs_in */
        size_t n_in = fs_in;  /* 1 second */
        double freq = 997.0;
        double *sine = (double *)malloc(n_in * sizeof(double));
        if (!sine) continue;
        for (size_t i = 0; i < n_in; i++)
            sine[i] = 0.5 * sin(2.0 * M_PI * freq * (double)i / (double)fs_in);

        const ntf_filter_t *f_in = ntf_auto_select(fs_in);
        sdm_context_t enc;
        sdm_context_init(&enc, f_in, f_in->order, 16, 512);
        float *dsd_in = (float *)calloc(n_in, sizeof(float));
        size_t enc_n = sdm_process_block(&enc, sine, dsd_in, n_in);
        sdm_context_free(&enc);
        free(sine);

        /* ── Path A: Direct FIR + Trellis ── */
        fir_chain_t fir_a;
        fir_chain_init(&fir_a, fs_in, fs_out);
        size_t max_a = enc_n;
        float *fir_a_out = (float *)malloc(max_a * sizeof(float));
        size_t fir_a_n = fir_chain_process(&fir_a, dsd_in, fir_a_out, enc_n);
        fir_chain_free(&fir_a);

        /* Apply gain */
        for (size_t i = 0; i < fir_a_n; i++) fir_a_out[i] *= 0.708f;

        const ntf_filter_t *f_out = ntf_auto_select(fs_out);
        sdm_context_t sdm_a;
        sdm_context_init(&sdm_a, f_out, 4, 2, 32);
        double *sdm_a_in = (double *)malloc(fir_a_n * sizeof(double));
        float *dsd_a_out = (float *)calloc(fir_a_n, sizeof(float));
        for (size_t i = 0; i < fir_a_n; i++) sdm_a_in[i] = (double)fir_a_out[i];
        size_t out_a = sdm_process_block(&sdm_a, sdm_a_in, dsd_a_out, fir_a_n);
        sdm_context_free(&sdm_a);
        double sinad_a = (out_a > 1024) ?
            measure_sinad(dsd_a_out, out_a, freq, (double)fs_out) : -999.0;
        free(fir_a_out); free(sdm_a_in); free(dsd_a_out);

        /* ── Path B: FIR(half) → PreCorr@mid → FIR(half) → Trellis ── */
        double sinad_b = -999.0;
        if (fs_mid > 0) {
            /* Step 1: FIR downsample to mid rate */
            fir_chain_t fir_b1;
            fir_chain_init(&fir_b1, fs_in, fs_mid);
            float *fir_b1_out = (float *)malloc(enc_n * sizeof(float));
            size_t fir_b1_n = fir_chain_process(&fir_b1, dsd_in, fir_b1_out, enc_n);
            fir_chain_free(&fir_b1);

            for (size_t i = 0; i < fir_b1_n; i++) fir_b1_out[i] *= 0.708f;

            /* Step 2: PreCorr SDM at mid rate */
            const ntf_filter_t *f_mid = ntf_auto_select_precorr(fs_mid);
            precorr_context_t pc;
            precorr_context_init(&pc, f_mid);
            double *pc_in = (double *)malloc(fir_b1_n * sizeof(double));
            float *dsd_mid = (float *)calloc(fir_b1_n, sizeof(float));
            for (size_t i = 0; i < fir_b1_n; i++) pc_in[i] = (double)fir_b1_out[i];
            size_t mid_n = precorr_process_block(&pc, pc_in, dsd_mid, fir_b1_n);
            precorr_context_free(&pc);
            free(fir_b1_out); free(pc_in);

            /* Step 3: FIR downsample mid → out */
            fir_chain_t fir_b2;
            fir_chain_init(&fir_b2, fs_mid, fs_out);
            float *fir_b2_out = (float *)malloc(mid_n * sizeof(float));
            size_t fir_b2_n = fir_chain_process(&fir_b2, dsd_mid, fir_b2_out, mid_n);
            fir_chain_free(&fir_b2);
            free(dsd_mid);

            for (size_t i = 0; i < fir_b2_n; i++) fir_b2_out[i] *= 0.708f;

            /* Step 4: Trellis SDM at out rate */
            sdm_context_t sdm_b;
            sdm_context_init(&sdm_b, f_out, 4, 2, 32);
            double *sdm_b_in = (double *)malloc(fir_b2_n * sizeof(double));
            float *dsd_b_out = (float *)calloc(fir_b2_n, sizeof(float));
            for (size_t i = 0; i < fir_b2_n; i++) sdm_b_in[i] = (double)fir_b2_out[i];
            size_t out_b = sdm_process_block(&sdm_b, sdm_b_in, dsd_b_out, fir_b2_n);
            sdm_context_free(&sdm_b);
            sinad_b = (out_b > 1024) ?
                measure_sinad(dsd_b_out, out_b, freq, (double)fs_out) : -999.0;
            free(fir_b2_out); free(sdm_b_in); free(dsd_b_out);
        }

        free(dsd_in);

        printf("\n    %s:\n", paths[p].name);
        printf("      Direct:              %.1f dB\n", sinad_a);
        if (fs_mid > 0)
            printf("      PreCorr intermediate: %.1f dB (delta: %+.1f)\n",
                   sinad_b, sinad_b - sinad_a);
    }

    TEST_ASSERT_TRUE(1, "PreCorr intermediate experiment completed");
}

/* ─── FIR tap count experiment for downsample ───
 * Test if longer FIR filters improve the weak downsample paths.
 * The 63-tap half-band has ~120 dB stopband but DSD ultrasonic noise
 * is enormous. Transition band leakage may be the bottleneck. */
static void test_fir_taps_experiment(void) {
    static const int tap_counts[] = { 63, 127, 255, 511 };
    static const double betas[] = { 12.0, 12.0, 12.0, 12.0 };
    int n_taps = sizeof(tap_counts) / sizeof(tap_counts[0]);

    typedef struct { uint32_t fs_in, fs_out; const char *name; } path_t;
    static const path_t paths[] = {
        { DSD_RATE_128, DSD_RATE_64, "DSD128→64" },
        { DSD_RATE_256, DSD_RATE_64, "DSD256→64" },
        { DSD_RATE_512, DSD_RATE_64, "DSD512→64" },
    };
    int n_paths = sizeof(paths) / sizeof(paths[0]);

    printf("\n    ╔══════════════════════════════════════════════════════╗\n");
    printf("    ║  FIR Tap Count Experiment (downsample paths)         ║\n");
    printf("    ╚══════════════════════════════════════════════════════╝\n");
    printf("    %-14s", "Path");
    for (int t = 0; t < n_taps; t++)
        printf("  %d-tap", tap_counts[t]);
    printf("\n");

    for (int p = 0; p < n_paths; p++) {
        uint32_t fs_in = paths[p].fs_in, fs_out = paths[p].fs_out;

        /* Generate clean DSD at fs_in */
        size_t n_in = fs_in / 2;  /* 0.5 seconds */
        double freq = 997.0;
        double *sine = (double *)malloc(n_in * sizeof(double));
        if (!sine) continue;
        for (size_t i = 0; i < n_in; i++)
            sine[i] = 0.5 * sin(2.0 * M_PI * freq * (double)i / (double)fs_in);

        const ntf_filter_t *f_in = ntf_auto_select(fs_in);
        sdm_context_t enc;
        sdm_context_init(&enc, f_in, f_in->order, 16, 512);
        float *dsd_in = (float *)calloc(n_in, sizeof(float));
        size_t enc_n = sdm_process_block(&enc, sine, dsd_in, n_in);
        sdm_context_free(&enc);
        free(sine);

        printf("    %-14s", paths[p].name);

        for (int t = 0; t < n_taps; t++) {
            int ntaps = tap_counts[t];

            /* Design half-band filter with specified taps */
            double *hd = (double *)calloc((size_t)ntaps, sizeof(double));
            float *hf = (float *)calloc((size_t)ntaps, sizeof(float));
            {
                int center = (ntaps - 1) / 2;
                double beta = betas[t];
                /* Inline Kaiser design */
                double sum = 0.0;
                for (int n = 0; n < ntaps; n++) {
                    double x = (double)(n - center) / 2.0;
                    double sinc_val = (fabs(x) < 1e-15) ? 1.0 : sin(M_PI * x) / (M_PI * x);
                    double tt = (double)(n - center) / center;
                    double arg = 1.0 - tt * tt;
                    if (arg < 0) arg = 0;
                    /* Bessel I0 approximation */
                    double bx = beta * sqrt(arg);
                    double I0_val = 1.0, term = 1.0, bx2 = (bx/2)*(bx/2);
                    for (int k = 1; k <= 25; k++) {
                        term *= bx2 / ((double)k * k);
                        I0_val += term;
                        if (term < 1e-20 * I0_val) break;
                    }
                    double I0_beta_val;
                    {
                        double bb = beta;
                        double I0b = 1.0, term2 = 1.0, bb2 = (bb/2)*(bb/2);
                        for (int k = 1; k <= 25; k++) {
                            term2 *= bb2 / ((double)k * k);
                            I0b += term2;
                        }
                        I0_beta_val = I0b;
                    }
                    double w = I0_val / I0_beta_val;
                    hd[n] = sinc_val * w;
                    sum += hd[n];
                }
                for (int n = 0; n < ntaps; n++) hd[n] /= sum;
                for (int n = 0; n < ntaps; n++) hf[n] = (float)hd[n];
            }

            /* Manual FIR downsample chain (can't use fir_chain since tap count is fixed) */
            uint32_t ratio = fs_in / fs_out;
            int stages = 0;
            { uint32_t r = ratio; while (r > 1) { stages++; r >>= 1; } }

            /* Simple direct-form FIR downsample (no IPP, just measure quality) */
            float *cur_buf = (float *)malloc(enc_n * sizeof(float));
            memcpy(cur_buf, dsd_in, enc_n * sizeof(float));
            size_t cur_n = enc_n;

            for (int s = 0; s < stages; s++) {
                size_t out_n = cur_n / 2;
                float *out_buf = (float *)calloc(out_n, sizeof(float));
                for (size_t i = 0; i < out_n; i++) {
                    double acc = 0.0;
                    int ii = (int)(i * 2);
                    for (int k = 0; k < ntaps; k++) {
                        int si = ii - k;
                        if (si >= 0 && si < (int)cur_n)
                            acc += hd[k] * (double)cur_buf[si];
                    }
                    out_buf[i] = (float)acc;
                }
                free(cur_buf);
                cur_buf = out_buf;
                cur_n = out_n;
            }

            /* Apply gain */
            for (size_t i = 0; i < cur_n; i++) cur_buf[i] *= 0.708f;

            /* Trellis SDM at output rate */
            const ntf_filter_t *f_out = ntf_auto_select(fs_out);
            sdm_context_t sdm;
            sdm_context_init(&sdm, f_out, 4, 2, 32);
            double *sdm_in = (double *)malloc(cur_n * sizeof(double));
            float *dsd_out = (float *)calloc(cur_n, sizeof(float));
            for (size_t i = 0; i < cur_n; i++) sdm_in[i] = (double)cur_buf[i];
            size_t out_count = sdm_process_block(&sdm, sdm_in, dsd_out, cur_n);
            sdm_context_free(&sdm);

            double sinad = (out_count > 1024) ?
                measure_sinad(dsd_out, out_count, freq, (double)fs_out) : -999.0;

            printf("  %6.1f", sinad);

            free(cur_buf); free(sdm_in); free(dsd_out);
            free(hd); free(hf);
        }
        printf("\n");
        free(dsd_in);
    }

    TEST_ASSERT_TRUE(1, "FIR taps experiment completed");
}

/* ─── FIR quality experiment: what limits the downsample ceiling? ───
 * Test the FIR-only SINAD (no SDM re-encode) to isolate FIR quality. */
static void test_fir_ceiling(void) {
    typedef struct { uint32_t fs_in, fs_out; const char *name; } path_t;
    static const path_t paths[] = {
        { DSD_RATE_128, DSD_RATE_64, "DSD128→64" },
        { DSD_RATE_256, DSD_RATE_64, "DSD256→64" },
        { DSD_RATE_256, DSD_RATE_128, "DSD256→128" },
        { DSD_RATE_512, DSD_RATE_64, "DSD512→64" },
        { DSD_RATE_512, DSD_RATE_128, "DSD512→128" },
    };
    int n_paths = sizeof(paths) / sizeof(paths[0]);

    printf("\n    ╔══════════════════════════════════════════════════════╗\n");
    printf("    ║  FIR Ceiling Test: FIR-only vs FIR+SDM              ║\n");
    printf("    ╚══════════════════════════════════════════════════════╝\n");
    printf("    %-14s  FIR-only  FIR+SDM   SDM cost\n", "Path");

    for (int p = 0; p < n_paths; p++) {
        uint32_t fs_in = paths[p].fs_in, fs_out = paths[p].fs_out;
        size_t n_in = fs_in;  /* 1 second */
        double freq = 997.0;

        /* Generate high-quality DSD at fs_in */
        double *sine = (double *)malloc(n_in * sizeof(double));
        for (size_t i = 0; i < n_in; i++)
            sine[i] = 0.5 * sin(2.0 * M_PI * freq * (double)i / (double)fs_in);
        const ntf_filter_t *f_in = ntf_auto_select(fs_in);
        sdm_context_t enc;
        sdm_context_init(&enc, f_in, f_in->order, 16, 512);
        float *dsd_in = (float *)calloc(n_in, sizeof(float));
        size_t enc_n = sdm_process_block(&enc, sine, dsd_in, n_in);
        sdm_context_free(&enc); free(sine);

        /* FIR downsample */
        fir_chain_t fir;
        fir_chain_init(&fir, fs_in, fs_out);
        float *fir_out = (float *)malloc(enc_n * sizeof(float));
        size_t fir_n = fir_chain_process(&fir, dsd_in, fir_out, enc_n);
        fir_chain_free(&fir); free(dsd_in);

        for (size_t i = 0; i < fir_n; i++) fir_out[i] *= 0.708f;

        /* Measure FIR-only SINAD (at output DSD rate, on multi-bit signal) */
        double sinad_fir = measure_sinad(fir_out, fir_n, freq, (double)fs_out);

        /* SDM re-encode + measure */
        const ntf_filter_t *f_out = ntf_auto_select(fs_out);
        sdm_context_t sdm;
        sdm_context_init(&sdm, f_out, 4, 8, 128);
        double *sdm_in = (double *)malloc(fir_n * sizeof(double));
        float *dsd_out = (float *)calloc(fir_n, sizeof(float));
        for (size_t i = 0; i < fir_n; i++) sdm_in[i] = (double)fir_out[i];
        size_t out_n = sdm_process_block(&sdm, sdm_in, dsd_out, fir_n);
        sdm_context_free(&sdm);
        double sinad_sdm = (out_n > 1024) ?
            measure_sinad(dsd_out, out_n, freq, (double)fs_out) : -999.0;

        printf("    %-14s  %6.1f    %6.1f    %+.1f dB\n",
               paths[p].name, sinad_fir, sinad_sdm, sinad_sdm - sinad_fir);

        free(fir_out); free(sdm_in); free(dsd_out);
    }

    TEST_ASSERT_TRUE(1, "FIR ceiling test completed");
}

/* ─── Lowpass before SDM experiment ───
 * Same-rate paths use 50kHz lowpass → SDM. Rate conversion doesn't.
 * Test if adding a lowpass between FIR chain and SDM improves quality. */
static void test_lowpass_before_sdm(void) {
    typedef struct { uint32_t fs_in, fs_out; const char *name; } path_t;
    static const path_t paths[] = {
        { DSD_RATE_128, DSD_RATE_64,  "DSD128→64" },
        { DSD_RATE_256, DSD_RATE_64,  "DSD256→64" },
        { DSD_RATE_256, DSD_RATE_128, "DSD256→128" },
        { DSD_RATE_512, DSD_RATE_128, "DSD512→128" },
    };
    int n_paths = sizeof(paths) / sizeof(paths[0]);

    printf("\n    ╔══════════════════════════════════════════════════════╗\n");
    printf("    ║  Lowpass Before SDM Experiment                       ║\n");
    printf("    ║  FIR chain → [optional 50kHz LP] → Trellis SDM      ║\n");
    printf("    ╚══════════════════════════════════════════════════════╝\n");
    printf("    %-14s  no-LP    with-LP   delta\n", "Path");

    for (int p = 0; p < n_paths; p++) {
        uint32_t fs_in = paths[p].fs_in, fs_out = paths[p].fs_out;
        size_t n_in = fs_in;
        double freq = 997.0;

        /* Generate clean DSD at fs_in */
        double *sine = (double *)malloc(n_in * sizeof(double));
        for (size_t i = 0; i < n_in; i++)
            sine[i] = 0.5 * sin(2.0 * M_PI * freq * (double)i / (double)fs_in);
        const ntf_filter_t *f_in = ntf_auto_select(fs_in);
        sdm_context_t enc;
        sdm_context_init(&enc, f_in, f_in->order, 16, 512);
        float *dsd_in = (float *)calloc(n_in, sizeof(float));
        size_t enc_n = sdm_process_block(&enc, sine, dsd_in, n_in);
        sdm_context_free(&enc); free(sine);

        /* FIR downsample */
        fir_chain_t fir;
        fir_chain_init(&fir, fs_in, fs_out);
        float *fir_out = (float *)malloc(enc_n * sizeof(float));
        size_t fir_n = fir_chain_process(&fir, dsd_in, fir_out, enc_n);
        fir_chain_free(&fir); free(dsd_in);
        for (size_t i = 0; i < fir_n; i++) fir_out[i] *= 0.708f;

        /* Path A: Direct FIR → SDM (no lowpass) */
        const ntf_filter_t *f_out = ntf_auto_select(fs_out);
        sdm_context_t sdm_a;
        sdm_context_init(&sdm_a, f_out, 4, 8, 128);
        double *in_a = (double *)malloc(fir_n * sizeof(double));
        float *out_a = (float *)calloc(fir_n, sizeof(float));
        for (size_t i = 0; i < fir_n; i++) in_a[i] = (double)fir_out[i];
        size_t n_a = sdm_process_block(&sdm_a, in_a, out_a, fir_n);
        sdm_context_free(&sdm_a);
        double sinad_a = (n_a > 1024) ? measure_sinad(out_a, n_a, freq, (double)fs_out) : -999.0;
        free(in_a); free(out_a);

        /* Path B: FIR → Lowpass@50kHz → SDM */
        fir_lowpass_t lp;
        fir_lowpass_init(&lp, fs_out);
        double *lp_in = (double *)malloc(fir_n * sizeof(double));
        double *lp_out_d = (double *)malloc(fir_n * sizeof(double));
        for (size_t i = 0; i < fir_n; i++) lp_in[i] = (double)fir_out[i];
        fir_lowpass_process(&lp, lp_in, lp_out_d, fir_n);
        fir_lowpass_free(&lp);
        free(lp_in);

        sdm_context_t sdm_b;
        sdm_context_init(&sdm_b, f_out, 4, 8, 128);
        float *out_b = (float *)calloc(fir_n, sizeof(float));
        size_t n_b = sdm_process_block(&sdm_b, lp_out_d, out_b, fir_n);
        sdm_context_free(&sdm_b);
        double sinad_b = (n_b > 1024) ? measure_sinad(out_b, n_b, freq, (double)fs_out) : -999.0;
        free(lp_out_d); free(out_b); free(fir_out);

        printf("    %-14s  %6.1f    %6.1f    %+.1f dB\n",
               paths[p].name, sinad_a, sinad_b, sinad_b - sinad_a);
    }

    TEST_ASSERT_TRUE(1, "Lowpass before SDM experiment completed");
}

void test_depth16_suite(void) {
    TEST_SUITE("Lowpass Before SDM");
    TEST_RUN(test_lowpass_before_sdm);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Downsample Path NTF+nc+depth+lat Sweep
 * ═══════════════════════════════════════════════════════════════════════
 * Comprehensive sweep for downsample rate conversion paths.
 * Input encoded at high quality (nc=16, lat=512), then FIR downsampled,
 * then SDM re-encoded with variable NTF/nc/depth/lat.
 * Measurement: bin-by-bin Goertzel at output DSD rate (audio band). */

static double measure_downsample_sinad(uint32_t fs_in, uint32_t fs_out,
                                        ntf_filter_id_t filter_id,
                                        int nc, int depth, int lat,
                                        float gain) {
    unsigned base = rate_is_48k_family(fs_in) ? 48000 : 44100;
    unsigned mult_in = fs_in / base;
    /* Use larger samples for stable measurements */
    size_t n_in;
    if (mult_in <= 64)       n_in = 131072;
    else if (mult_in <= 128) n_in = 262144;
    else if (mult_in <= 256) n_in = 524288;
    else                     n_in = 1048576;

    /* For multi-stage downsample, fir_chain_process uses out buffer for
     * intermediate ping-pong. Stage 0 writes n_in/2 to out. Must size
     * for the largest intermediate, not just the final output. */
    size_t max_out = n_in / 2 + 4096;

    float *dsd_in  = (float *)malloc(n_in * sizeof(float));
    float *fir_buf = (float *)malloc(max_out * sizeof(float));
    float *dsd_out = (float *)malloc(max_out * sizeof(float));
    if (!dsd_in || !fir_buf || !dsd_out) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    /* Estimate output sample count for frequency alignment */
    size_t est_in = n_in - 512;
    size_t est_fir = est_in / (fs_in / fs_out);
    size_t est_sdm = (est_fir > (size_t)lat) ? est_fir - (size_t)lat : 512;
    if (est_sdm < 512) est_sdm = 512;
    double freq = bin_align_freq(1000.0, (double)fs_out, est_sdm);

    /* Generate high-quality DSD input (nc=16, lat=512) */
    const ntf_filter_t *f_in = ntf_auto_select(fs_in);
    if (!f_in) { free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    sdm_context_t gen;
    if (sdm_context_init(&gen, f_in, 8, 16, 512) != 0) {
        free(dsd_in); free(fir_buf); free(dsd_out); return -999.0;
    }
    double *sine = (double *)malloc(n_in * sizeof(double));
    if (!sine) { sdm_context_free(&gen); free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    for (size_t i = 0; i < n_in; i++)
        sine[i] = 0.5 * sin(2.0 * M_PI * freq * (double)i / (double)fs_in);
    size_t dsd_in_count = sdm_process_block(&gen, sine, dsd_in, n_in);
    free(sine);
    sdm_context_free(&gen);
    if (dsd_in_count < 512) { free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }

    /* FIR downsample (fp64, matching production Auto=fp64) */
    fir_chain_t fir;
    if (fir_chain_init_ex(&fir, fs_in, fs_out, true) != 0) {
        free(dsd_in); free(fir_buf); free(dsd_out); return -999.0;
    }
    double *fir_d_in = (double *)malloc(dsd_in_count * sizeof(double));
    double *fir_d_out = (double *)malloc(max_out * sizeof(double));
    if (!fir_d_in || !fir_d_out) {
        free(fir_d_in); free(fir_d_out); free(dsd_in); free(fir_buf); free(dsd_out);
        fir_chain_free(&fir); return -999.0;
    }
    for (size_t i = 0; i < dsd_in_count; i++)
        fir_d_in[i] = (double)dsd_in[i];
    size_t fir_count = fir_chain_process_d(&fir, fir_d_in, fir_d_out, dsd_in_count);
    free(fir_d_in);
    fir_chain_free(&fir);
    if (fir_count < 512) { free(fir_d_out); free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }

    /* Apply gain in fp64 */
    if (gain != 1.0f) {
        double g = (double)gain;
        for (size_t i = 0; i < fir_count; i++)
            fir_d_out[i] *= g;
    }

    /* SDM re-encode with specified params */
    const ntf_filter_t *f_out = ntf_get_filter(filter_id, fs_out);
    if (!f_out) { free(fir_d_out); free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    sdm_context_t sdm;
    if (sdm_context_init(&sdm, f_out, (unsigned)depth, (unsigned)nc, (unsigned)lat) != 0) {
        free(fir_d_out); free(dsd_in); free(fir_buf); free(dsd_out); return -999.0;
    }
    size_t out_count = sdm_process_block(&sdm, fir_d_out, dsd_out, fir_count);
    free(fir_d_out);
    sdm_context_free(&sdm);
    if (out_count < 512) { free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }

    double sinad_db = measure_sinad(dsd_out, out_count, freq, (double)fs_out);
    free(dsd_in); free(fir_buf); free(dsd_out);
    return sinad_db;
}

/* General rate-conversion SINAD with configurable NTF/nc/depth/lat, fp64 FIR.
 * Works for both upsample and downsample (buffer sizing handles both). */
static double measure_rateconv_sinad(uint32_t fs_in, uint32_t fs_out,
                                      ntf_filter_id_t filter_id,
                                      int nc, int depth, int lat,
                                      double state_limit, float gain) {
    unsigned base = rate_is_48k_family(fs_in) ? 48000 : 44100;
    unsigned mult_in = fs_in / base;
    size_t n_in;
    if (mult_in <= 64)       n_in = 131072;
    else if (mult_in <= 128) n_in = 262144;
    else if (mult_in <= 256) n_in = 524288;
    else                     n_in = 1048576;

    size_t max_out;
    if (fs_out >= fs_in)
        max_out = n_in * (fs_out / fs_in) + 4096;
    else
        max_out = n_in / 2 + 4096;

    float *dsd_in  = (float *)malloc(n_in * sizeof(float));
    float *dsd_out = (float *)malloc(max_out * sizeof(float));
    if (!dsd_in || !dsd_out) {
        free(dsd_in); free(dsd_out); return -999.0;
    }

    size_t est_in = n_in - 512;
    size_t est_fir = (fs_out >= fs_in) ?
        est_in * (fs_out / fs_in) : est_in / (fs_in / fs_out);
    size_t est_sdm = (est_fir > (size_t)lat) ? est_fir - (size_t)lat : 512;
    if (est_sdm < 512) est_sdm = 512;
    double freq = bin_align_freq(1000.0, (double)fs_out, est_sdm);

    const ntf_filter_t *f_in = ntf_auto_select(fs_in);
    if (!f_in) { free(dsd_in); free(dsd_out); return -999.0; }
    sdm_context_t gen;
    if (sdm_context_init(&gen, f_in, 8, 16, 512) != 0) {
        free(dsd_in); free(dsd_out); return -999.0;
    }
    double *sine = (double *)malloc(n_in * sizeof(double));
    if (!sine) { sdm_context_free(&gen); free(dsd_in); free(dsd_out); return -999.0; }
    for (size_t i = 0; i < n_in; i++)
        sine[i] = 0.5 * sin(2.0 * M_PI * freq * (double)i / (double)fs_in);
    size_t dsd_in_count = sdm_process_block(&gen, sine, dsd_in, n_in);
    free(sine);
    sdm_context_free(&gen);
    if (dsd_in_count < 512) { free(dsd_in); free(dsd_out); return -999.0; }

    /* FIR rate conversion (fp64, matching production) */
    fir_chain_t fir;
    if (fir_chain_init_ex(&fir, fs_in, fs_out, true) != 0) {
        free(dsd_in); free(dsd_out); return -999.0;
    }
    double *fir_d_in = (double *)malloc(dsd_in_count * sizeof(double));
    double *fir_d_out = (double *)malloc(max_out * sizeof(double));
    if (!fir_d_in || !fir_d_out) {
        free(fir_d_in); free(fir_d_out); free(dsd_in); free(dsd_out);
        fir_chain_free(&fir); return -999.0;
    }
    for (size_t i = 0; i < dsd_in_count; i++)
        fir_d_in[i] = (double)dsd_in[i];
    size_t fir_count = fir_chain_process_d(&fir, fir_d_in, fir_d_out, dsd_in_count);
    free(fir_d_in);
    fir_chain_free(&fir);
    if (fir_count < 512) { free(fir_d_out); free(dsd_in); free(dsd_out); return -999.0; }

    /* Apply gain in fp64 */
    if (gain != 1.0f) {
        double g = (double)gain;
        for (size_t i = 0; i < fir_count; i++)
            fir_d_out[i] *= g;
    }

    /* SDM re-encode */
    const ntf_filter_t *f_out = ntf_get_filter(filter_id, fs_out);
    if (!f_out) { free(fir_d_out); free(dsd_in); free(dsd_out); return -999.0; }
    sdm_context_t sdm;
    if (sdm_context_init(&sdm, f_out, (unsigned)depth, (unsigned)nc, (unsigned)lat) != 0) {
        free(fir_d_out); free(dsd_in); free(dsd_out); return -999.0;
    }
    if (state_limit > 0.0)
        sdm.state_limit = state_limit;
    size_t out_count = sdm_process_block(&sdm, fir_d_out, dsd_out, fir_count);
    free(fir_d_out);
    sdm_context_free(&sdm);
    if (out_count < 512) { free(dsd_in); free(dsd_out); return -999.0; }

    freq = bin_align_freq(freq, (double)fs_out, out_count);
    double sinad_db = measure_sinad(dsd_out, out_count, freq, (double)fs_out);
    free(dsd_in); free(dsd_out);
    return sinad_db;
}

static void test_downsample_sweep(void) {
    typedef struct {
        uint32_t fs_in, fs_out;
        const char *name;
    } dn_path_t;

    static const dn_path_t paths[] = {
        { DSD_RATE_128, DSD_RATE_64,  "DSD128->64"  },
        { DSD_RATE_256, DSD_RATE_64,  "DSD256->64"  },
        { DSD_RATE_512, DSD_RATE_64,  "DSD512->64"  },
        { DSD_RATE_256, DSD_RATE_128, "DSD256->128" },
        { DSD_RATE_512, DSD_RATE_128, "DSD512->128" },
        { DSD_RATE_512, DSD_RATE_256, "DSD512->256" },
    };
    static const int n_paths = sizeof(paths) / sizeof(paths[0]);

    static const ntf_filter_id_t filters[] = {
        NTF_CLANS_4, NTF_SDM_4,
        NTF_CLANS_5, NTF_SDM_5,
        NTF_CLANS_6, NTF_SDM_6,
        NTF_CLANS_7, NTF_SDM_7,
        NTF_CLANS_8, NTF_SDM_8,
    };
    static const char *filter_names[] = {
        "clans-4", "sdm-4",
        "clans-5", "sdm-5",
        "clans-6", "sdm-6",
        "clans-7", "sdm-7",
        "clans-8", "sdm-8",
    };
    static const int n_filters = 10;

    /* ─── Phase 1: NTF × nc sweep (d=4, lat=128) ─── */
    /* nc is the dominant parameter for downsample quality.
     * nc=2 gives 65-75 dB, production nc=32 gives 85 dB. */
    static const int nc_vals[]  = { 2, 4, 8, 16, 32 };
    static const int n_nc = 5;

    printf("\n    ╔══════════════════════════════════════════════════════════════╗\n");
    printf("    ║  Downsample NTF × nc Sweep (Phase 1)                        ║\n");
    printf("    ║  %d paths × %d NTFs × %d nc = %d measurements                ║\n",
           n_paths, n_filters, n_nc, n_paths * n_filters * n_nc);
    printf("    ║  Fixed: depth=4, lat=128, gain=0.708                        ║\n");
    printf("    ╚══════════════════════════════════════════════════════════════╝\n");

    /* Track best per path for Phase 2 */
    typedef struct {
        ntf_filter_id_t filter;
        int nc;
        double sinad;
    } winner_t;
    winner_t best_p1[6];
    for (int p = 0; p < n_paths; p++) best_p1[p].sinad = -999.0;

    for (int p = 0; p < n_paths; p++) {
        printf("\n    --- %s ---\n", paths[p].name);
        printf("    %-10s", "NTF\\nc");
        for (int n = 0; n < n_nc; n++)
            printf("  nc=%-3d", nc_vals[n]);
        printf("   best\n");

        for (int f = 0; f < n_filters; f++) {
            printf("    %-10s", filter_names[f]);
            double row_best = -999.0;
            int row_best_nc = 0;

            for (int n = 0; n < n_nc; n++) {
                fflush(stdout);
                double s = measure_downsample_sinad(
                    paths[p].fs_in, paths[p].fs_out,
                    filters[f], nc_vals[n], 4, 128, 0.708f);
                printf("  %6.1f", s);
                if (s > row_best) { row_best = s; row_best_nc = nc_vals[n]; }
                if (s > best_p1[p].sinad) {
                    best_p1[p].filter = filters[f];
                    best_p1[p].nc = nc_vals[n];
                    best_p1[p].sinad = s;
                }
            }
            printf("  %6.1f (nc=%d)\n", row_best, row_best_nc);
            fflush(stdout);
        }
    }

    /* Print Phase 1 winners */
    printf("\n    ╔══════════════════════════════════════════════════════════════╗\n");
    printf("    ║  Phase 1 Winners                                            ║\n");
    printf("    ╠══════════════════════════════════════════════════════════════╣\n");
    for (int p = 0; p < n_paths; p++) {
        const char *fn = "?";
        for (int f = 0; f < n_filters; f++)
            if (filters[f] == best_p1[p].filter) { fn = filter_names[f]; break; }
        printf("    ║  %-14s %-10s nc=%-2d         %7.1f dB            ║\n",
               paths[p].name, fn, best_p1[p].nc, best_p1[p].sinad);
    }
    printf("    ╚══════════════════════════════════════════════════════════════╝\n");

    /* ─── Phase 2: depth × lat sweep on Phase 1 winners ─── */
    static const int depths[]   = { 2, 4, 8, 16 };
    static const int lat_vals[] = { 32, 64, 128, 256, 512 };
    static const int n_depths = 4, n_lat = 5;

    printf("\n    ╔══════════════════════════════════════════════════════════════╗\n");
    printf("    ║  Downsample depth × lat Sweep (Phase 2)                     ║\n");
    printf("    ║  Best NTF+nc per path × %d depth × %d lat = %d measurements  ║\n",
           n_depths, n_lat, n_paths * n_depths * n_lat);
    printf("    ╚══════════════════════════════════════════════════════════════╝\n");

    typedef struct {
        ntf_filter_id_t filter;
        int nc, depth, lat;
        double sinad;
    } best_config_t;
    best_config_t best[6];
    for (int p = 0; p < n_paths; p++) best[p].sinad = -999.0;

    for (int p = 0; p < n_paths; p++) {
        if (best_p1[p].sinad < -900.0) continue;
        const char *fn = "?";
        for (int f = 0; f < n_filters; f++)
            if (filters[f] == best_p1[p].filter) { fn = filter_names[f]; break; }

        printf("\n    --- %s (NTF=%s, nc=%d, Phase1=%.1f dB) ---\n",
               paths[p].name, fn, best_p1[p].nc, best_p1[p].sinad);
        printf("    %-10s", "depth\\lat");
        for (int l = 0; l < n_lat; l++)
            printf("  lat=%-3d", lat_vals[l]);
        printf("   best\n");

        for (int d = 0; d < n_depths; d++) {
            printf("    d=%-6d", depths[d]);
            double row_best = -999.0;
            int row_best_lat = 0;

            for (int l = 0; l < n_lat; l++) {
                double s = measure_downsample_sinad(
                    paths[p].fs_in, paths[p].fs_out,
                    best_p1[p].filter, best_p1[p].nc, depths[d],
                    lat_vals[l], 0.708f);
                printf("  %7.1f", s);
                if (s > row_best) { row_best = s; row_best_lat = lat_vals[l]; }
                if (s > best[p].sinad) {
                    best[p].filter = best_p1[p].filter;
                    best[p].nc = best_p1[p].nc;
                    best[p].depth = depths[d];
                    best[p].lat = lat_vals[l];
                    best[p].sinad = s;
                }
            }
            printf("  %7.1f (lat=%d)\n", row_best, row_best_lat);
        }
    }

    /* Print final results */
    printf("\n    ╔══════════════════════════════════════════════════════════════╗\n");
    printf("    ║  Optimal Downsample Configurations                          ║\n");
    printf("    ╠══════════════════════════════════════════════════════════════╣\n");
    for (int p = 0; p < n_paths; p++) {
        if (best[p].sinad < -900.0) continue;
        const char *fn = "?";
        for (int f = 0; f < n_filters; f++)
            if (filters[f] == best[p].filter) { fn = filter_names[f]; break; }
        printf("    ║  %-14s %-10s nc=%-2d d=%-2d lat=%-3d  %7.1f dB  ║\n",
               paths[p].name, fn, best[p].nc, best[p].depth,
               best[p].lat, best[p].sinad);
    }
    printf("    ╚══════════════════════════════════════════════════════════════╝\n");

    TEST_ASSERT_TRUE(1, "Downsample NTF+nc+depth+lat sweep completed");
}

/* ═══════════════════════════════════════════════════════════════════════
 * FIR Chain Improvement Experiments
 * ═══════════════════════════════════════════════════════════════════════
 * Test approaches to improve DSD→DSD rate conversion SINAD (currently 15-68 dB).
 * Baseline: fp32 FIR chain (63-tap Kaiser beta=12). */

static double measure_fir_experiment(uint32_t fs_in, uint32_t fs_out,
                                      int use_fp64, int use_lowpass_pre) {
    unsigned base = rate_is_48k_family(fs_in) ? 48000 : 44100;
    unsigned mult_in = fs_in / base;
    size_t n_in;
    if (mult_in <= 64)       n_in = 262144;
    else if (mult_in <= 128) n_in = 524288;
    else if (mult_in <= 256) n_in = 1048576;
    else                     n_in = 2097152;

    /* Buffer for largest intermediate (multi-stage ping-pong) */
    size_t max_buf;
    if (fs_out >= fs_in)
        max_buf = n_in * (fs_out / fs_in) + 4096;
    else
        max_buf = n_in / 2 + 4096;

    float *dsd_in  = (float *)malloc(n_in * sizeof(float));
    float *dsd_out = (float *)malloc(max_buf * sizeof(float));
    if (!dsd_in || !dsd_out) { free(dsd_in); free(dsd_out); return -999.0; }

    /* Estimate output for frequency alignment */
    dsd_config_t cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.fs_in = fs_in; cfg.fs_out = fs_out;
    engine_path_info_t pi;
    engine_get_path_info(fs_in, fs_out, NTF_AUTO, SDM_MODE_TRELLIS, &cfg, &pi);
    int cands = pi.cands > 0 ? pi.cands : 2;
    int lat = pi.lat > 0 ? pi.lat : 32;
    int depth = pi.depth > 0 ? pi.depth : 4;

    size_t est_in = n_in - 512;  /* input SDM latency */
    size_t est_fir = (fs_out >= fs_in) ?
        est_in * (fs_out / fs_in) : est_in / (fs_in / fs_out);
    size_t est_sdm = (est_fir > (size_t)lat) ? est_fir - (size_t)lat : 1024;
    double freq = bin_align_freq(1000.0, (double)fs_out, est_sdm);

    /* Generate high-quality DSD input */
    size_t dsd_in_count = generate_dsd_sine(fs_in, freq, 0.5, n_in, dsd_in);
    if (dsd_in_count < 1024) { free(dsd_in); free(dsd_out); return -999.0; }

    /* Optional: FIR lowpass BEFORE rate conversion (removes DSD shaped noise) */
    double *lp_buf = NULL;
    if (use_lowpass_pre) {
        fir_lowpass_t lp;
        memset(&lp, 0, sizeof(lp));
        if (fir_lowpass_init(&lp, fs_in) != 0) { free(dsd_in); free(dsd_out); return -999.0; }
        double *lp_in = (double *)malloc(dsd_in_count * sizeof(double));
        lp_buf = (double *)malloc(dsd_in_count * sizeof(double));
        if (!lp_in || !lp_buf) { free(lp_in); free(lp_buf); free(dsd_in); free(dsd_out); fir_lowpass_free(&lp); return -999.0; }
        for (size_t i = 0; i < dsd_in_count; i++) lp_in[i] = (double)dsd_in[i];
        size_t lp_n = fir_lowpass_process(&lp, lp_in, lp_buf, dsd_in_count);
        free(lp_in);
        fir_lowpass_free(&lp);
        /* Copy back to dsd_in as float for FIR chain */
        for (size_t i = 0; i < lp_n; i++) dsd_in[i] = (float)lp_buf[i];
        dsd_in_count = lp_n;
    }

    /* FIR rate conversion */
    size_t fir_count;
    if (use_fp64) {
        fir_chain_t fir;
        if (fir_chain_init_ex(&fir, fs_in, fs_out, true) != 0) {
            free(dsd_in); free(dsd_out); free(lp_buf); return -999.0;
        }
        double *din = (double *)malloc(dsd_in_count * sizeof(double));
        double *dout = (double *)malloc(max_buf * sizeof(double));
        if (!din || !dout) { free(din); free(dout); free(dsd_in); free(dsd_out); free(lp_buf); fir_chain_free(&fir); return -999.0; }
        if (use_lowpass_pre && lp_buf) {
            memcpy(din, lp_buf, dsd_in_count * sizeof(double));
        } else {
            for (size_t i = 0; i < dsd_in_count; i++) din[i] = (double)dsd_in[i];
        }
        fir_count = fir_chain_process_d(&fir, din, dout, dsd_in_count);
        /* Convert back to float for SDM */
        float *fir_buf = (float *)malloc(fir_count * sizeof(float));
        if (fir_buf) {
            for (size_t i = 0; i < fir_count; i++) fir_buf[i] = (float)dout[i];
            free(dsd_in);
            dsd_in = fir_buf;  /* reuse pointer */
        }
        free(din); free(dout);
        fir_chain_free(&fir);
    } else {
        float *fir_buf = (float *)malloc(max_buf * sizeof(float));
        if (!fir_buf) { free(dsd_in); free(dsd_out); free(lp_buf); return -999.0; }
        fir_chain_t fir;
        if (fir_chain_init(&fir, fs_in, fs_out) != 0) {
            free(dsd_in); free(fir_buf); free(dsd_out); free(lp_buf); return -999.0;
        }
        fir_count = fir_chain_process(&fir, dsd_in, fir_buf, dsd_in_count);
        fir_chain_free(&fir);
        free(dsd_in);
        dsd_in = fir_buf;
    }
    free(lp_buf);

    if (fir_count < 512) { free(dsd_in); free(dsd_out); return -999.0; }

    /* Apply gain */
    if (pi.fir_gain != 1.0f)
        for (size_t i = 0; i < fir_count; i++)
            dsd_in[i] *= pi.fir_gain;

    /* SDM re-encode */
    const ntf_filter_t *f_out = (pi.ntf_filter != NTF_AUTO) ?
        ntf_get_filter((ntf_filter_id_t)pi.ntf_filter, fs_out) : ntf_auto_select(fs_out);
    if (!f_out) { free(dsd_in); free(dsd_out); return -999.0; }
    sdm_context_t sdm;
    if (sdm_context_init(&sdm, f_out, depth, cands, lat) != 0) {
        free(dsd_in); free(dsd_out); return -999.0;
    }
    if (pi.state_limit > 0.0) sdm.state_limit = pi.state_limit;
    double *sdm_in_d = float_to_double(dsd_in, fir_count);
    if (!sdm_in_d) { sdm_context_free(&sdm); free(dsd_in); free(dsd_out); return -999.0; }
    size_t out_count = sdm_process_block(&sdm, sdm_in_d, dsd_out, fir_count);
    free(sdm_in_d); free(dsd_in);
    sdm_context_free(&sdm);

    if (out_count < 512) { free(dsd_out); return -999.0; }

    /* Re-align frequency to actual output count */
    double freq_pre = freq;
    freq = bin_align_freq(freq, (double)fs_out, out_count);

    double sinad = measure_sinad(dsd_out, out_count, freq, (double)fs_out);


    free(dsd_out);
    return sinad;
}

static void test_fir_improvement_experiment(void) {
    typedef struct { uint32_t fs_in, fs_out; const char *name; } path_t;
    static const path_t paths[] = {
        /* /44 family */
        { DSD_RATE_64,  DSD_RATE_128, "64->128 UP"     },
        { DSD_RATE_64,  DSD_RATE_256, "64->256 UP"     },
        { DSD_RATE_64,  DSD_RATE_512, "64->512 UP"     },
        { DSD_RATE_128, DSD_RATE_256, "128->256 UP"    },
        { DSD_RATE_128, DSD_RATE_512, "128->512 UP"    },
        { DSD_RATE_256, DSD_RATE_512, "256->512 UP"    },
        { DSD_RATE_128, DSD_RATE_64,  "128->64 DN"     },
        { DSD_RATE_256, DSD_RATE_128, "256->128 DN"    },
        { DSD_RATE_512, DSD_RATE_256, "512->256 DN"    },
        /* /48 family */
        { DSD48_RATE_64,  DSD48_RATE_128, "64/48->128/48 UP"  },
        { DSD48_RATE_64,  DSD48_RATE_256, "64/48->256/48 UP"  },
        { DSD48_RATE_128, DSD48_RATE_256, "128/48->256/48 UP" },
        { DSD48_RATE_128, DSD48_RATE_64,  "128/48->64/48 DN"  },
        { DSD48_RATE_256, DSD48_RATE_64,  "256/48->64/48 DN"  },
        { DSD48_RATE_256, DSD48_RATE_128, "256/48->128/48 DN" },
    };
    int n_paths = sizeof(paths) / sizeof(paths[0]);

    printf("\n    ╔══════════════════════════════════════════════════════════════╗\n");
    printf("    ║  FIR Improvement: fp32 vs fp64 vs lowpass+fp32/fp64         ║\n");
    printf("    ╚══════════════════════════════════════════════════════════════╝\n");
    printf("    %-18s %8s %8s %8s %8s\n", "Path", "A(fp32)", "B(fp64)", "C(lp32)", "D(lp64)");

    for (int p = 0; p < n_paths; p++) {
        printf("    %-18s", paths[p].name);
        fflush(stdout);
        double a = measure_fir_experiment(paths[p].fs_in, paths[p].fs_out, 0, 0);
        printf(" %7.1f", a); fflush(stdout);
        double b = measure_fir_experiment(paths[p].fs_in, paths[p].fs_out, 1, 0);
        printf(" %7.1f", b); fflush(stdout);
        double c = measure_fir_experiment(paths[p].fs_in, paths[p].fs_out, 0, 1);
        printf(" %7.1f", c); fflush(stdout);
        double d = measure_fir_experiment(paths[p].fs_in, paths[p].fs_out, 1, 1);
        printf(" %7.1f\n", d); fflush(stdout);
    }

    TEST_ASSERT_TRUE(1, "FIR precision/lowpass experiment completed");
}

/* ═══════════════════════════════════════════════════════════════════════
 * NTF Candidate Testing (pydelsig-generated NTFs)
 * ═══════════════════════════════════════════════════════════════════════ */

static double test_ntf_sinad(const ntf_filter_t *f, uint32_t dsd_rate,
                              int depth, int cands, int lat) {
    size_t n = (dsd_rate / 44100 <= 64) ? 262144 :
               (dsd_rate / 44100 <= 128) ? 524288 : 1048576;
    float *out = (float *)calloc(n, sizeof(float));
    if (!out) return -999.0;

    sdm_context_t ctx;
    if (sdm_context_init(&ctx, f, depth, cands, lat) != 0) { free(out); return -999.0; }

    double *sine = (double *)malloc(n * sizeof(double));
    if (!sine) { sdm_context_free(&ctx); free(out); return -999.0; }
    double freq = bin_align_freq(1000.0, (double)dsd_rate, n - lat);
    for (size_t i = 0; i < n; i++)
        sine[i] = 0.5 * sin(2.0 * M_PI * freq * (double)i / (double)dsd_rate);
    size_t produced = sdm_process_block(&ctx, sine, out, n);
    free(sine);
    sdm_context_free(&ctx);

    if (produced < 1024) { free(out); return -999.0; }
    freq = bin_align_freq(freq, (double)dsd_rate, produced);
    double sinad = measure_sinad(out, produced, freq, (double)dsd_rate);
    free(out);
    return sinad;
}

static void test_pydelsig_ntf_candidates(void) {
    /* pydelsig-generated NTFs (reversed a[], zero-interleaved g[]) */
    static const ntf_filter_t candidates[] = {
      { { 5.56285181e-01, 2.50809320e-01, 5.67247036e-02, 1.00372674e-02, 1.05328380e-03, 5.59904858e-05 },
        { 2.09475548e-03, 0, 1.05336853e-03, 0, 1.37198910e-04, 0 },
        6, 2822400, "opt1-h15" },
      { { 7.50410415e-01, 6.23302293e-01, 2.18968155e-01, 6.86702761e-02, 1.15867896e-02, 1.05133287e-03 },
        { 2.09475548e-03, 0, 1.05336853e-03, 0, 1.37198910e-04, 0 },
        6, 2822400, "opt1-h20" },
      { { 8.40262670e-01, 9.65403526e-01, 4.16382876e-01, 1.75614505e-01, 3.75659037e-02, 4.45647920e-03 },
        { 2.09475548e-03, 0, 1.05336853e-03, 0, 1.37198910e-04, 0 },
        6, 2822400, "opt1-h25" },
      { { 7.50000000e-01, 6.25393860e-01, 2.19242324e-01, 6.91331866e-02, 1.15576735e-02, 1.05422766e-03 },
        { 0, 0, 0, 0, 0, 0 },
        6, 2822400, "opt0-h20" },
      { { 8.40000005e-01, 9.67376466e-01, 4.16556688e-01, 1.76250195e-01, 3.74850761e-02, 4.46051626e-03 },
        { 0, 0, 0, 0, 0, 0 },
        6, 2822400, "opt0-h25" },
      { { 5.56553351e-01, 2.50929241e-01, 5.76763359e-02, 1.11272300e-02, 1.42854237e-03, 1.35736774e-04, 8.51829433e-06, 2.54758348e-07 },
        { 2.22159111e-03, 0, 1.52910719e-03, 0, 6.65451549e-04, 0, 8.10795197e-05, 0 },
        8, 2822400, "o8-opt1-h15" },
      { { 7.50561643e-01, 6.27518403e-01, 2.24613788e-01, 7.82652033e-02, 1.60789148e-02, 2.70150491e-03, 2.73158861e-04, 1.41349628e-05 },
        { 2.22159111e-03, 0, 1.52910719e-03, 0, 6.65451549e-04, 0, 8.10795197e-05, 0 },
        8, 2822400, "o8-opt1-h20" },
    };
    int n_cands = sizeof(candidates) / sizeof(candidates[0]);

    /* Also test existing best: CLANS-6 @ DSD64 */
    const ntf_filter_t *clans6 = ntf_get_filter(NTF_CLANS_6, DSD_RATE_64);

    printf("\n    NTF Candidate SINAD Test (DSD64, depth=16, nc=2, lat=32)\n");
    printf("    %-16s %8s\n", "Name", "SINAD");

    if (clans6) {
        double s = test_ntf_sinad(clans6, DSD_RATE_64, 16, 2, 32);
        printf("    %-16s %7.1f dB  (existing)\n", clans6->name, s);
    }

    for (int i = 0; i < n_cands; i++) {
        int d = (candidates[i].order <= 6) ? 16 : 8;
        double s = test_ntf_sinad(&candidates[i], DSD_RATE_64, d, 2, 32);
        printf("    %-16s %7.1f dB\n", candidates[i].name, s);
        fflush(stdout);
    }

    TEST_ASSERT_TRUE(1, "NTF candidate test completed");
}

void test_fir_experiment_suite(void) {
    TEST_SUITE("FIR Experiment");
    TEST_RUN(test_fir_improvement_experiment);
    TEST_RUN(test_pydelsig_ntf_candidates);
}

/* ═══════════════════════════════════════════════════════════════════════
 * DSD/48 Downsample NTF+nc Sweep (same methodology as /44 sweep)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_48k_downsample_sweep(void) {
    typedef struct { uint32_t fs_in, fs_out; const char *name; } dn_path_t;
    static const dn_path_t paths[] = {
        { DSD48_RATE_128, DSD48_RATE_64,  "128/48->64/48"  },
        { DSD48_RATE_256, DSD48_RATE_64,  "256/48->64/48"  },
        { DSD48_RATE_256, DSD48_RATE_128, "256/48->128/48" },
    };
    int n_paths = sizeof(paths) / sizeof(paths[0]);

    static const ntf_filter_id_t filters[] = {
        NTF_CLANS_4, NTF_SDM_4, NTF_CLANS_5, NTF_SDM_5,
        NTF_CLANS_6, NTF_SDM_6, NTF_CLANS_7, NTF_SDM_7,
        NTF_CLANS_8, NTF_SDM_8,
    };
    static const char *fnames[] = {
        "clans-4","sdm-4","clans-5","sdm-5","clans-6","sdm-6",
        "clans-7","sdm-7","clans-8","sdm-8",
    };
    int n_filters = 10;
    static const int nc_vals[] = { 2, 4, 8, 16, 32 };
    int n_nc = 5;

    printf("\n    DSD/48 Downsample NTF x nc Sweep (d=4, lat=128, gain=0.708)\n");
    printf("    %d paths x %d NTFs x %d nc = %d measurements\n\n",
           n_paths, n_filters, n_nc, n_paths * n_filters * n_nc);

    typedef struct { ntf_filter_id_t filter; int nc; double sinad; } best_t;
    best_t best[3];
    for (int p = 0; p < n_paths; p++) best[p].sinad = -999.0;

    for (int p = 0; p < n_paths; p++) {
        printf("    --- %s ---\n", paths[p].name);
        printf("    %-10s", "NTF\\nc");
        for (int n = 0; n < n_nc; n++) printf("  nc=%-3d", nc_vals[n]);
        printf("   best\n");

        for (int f = 0; f < n_filters; f++) {
            printf("    %-10s", fnames[f]);
            double row_best = -999.0; int row_nc = 0;
            for (int n = 0; n < n_nc; n++) {
                fflush(stdout);
                double s = measure_downsample_sinad(
                    paths[p].fs_in, paths[p].fs_out,
                    filters[f], nc_vals[n], 4, 128, 0.708f);
                printf("  %6.1f", s);
                if (s > row_best) { row_best = s; row_nc = nc_vals[n]; }
                if (s > best[p].sinad) {
                    best[p].filter = filters[f]; best[p].nc = nc_vals[n]; best[p].sinad = s;
                }
            }
            printf("  %6.1f (nc=%d)\n", row_best, row_nc);
            fflush(stdout);
        }
    }

    printf("\n    Winners:\n");
    for (int p = 0; p < n_paths; p++) {
        const char *fn = "?";
        for (int f = 0; f < n_filters; f++)
            if (filters[f] == best[p].filter) { fn = fnames[f]; break; }
        printf("    %-16s %-10s nc=%-2d  %.1f dB\n",
               paths[p].name, fn, best[p].nc, best[p].sinad);
    }

    TEST_ASSERT_TRUE(1, "48k downsample sweep completed");
}

/* ═══════════════════════════════════════════════════════════════════════
 * fp64 Re-sweep: paths that regressed after removing float truncation
 * ═══════════════════════════════════════════════════════════════════════ */
static void test_fp64_resweep(void) {
    typedef struct {
        uint32_t fs_in, fs_out;
        const char *name;
    } path_t;

    /* Paths that regressed when switching to fp64-direct pipeline */
    static const path_t paths[] = {
        { DSD_RATE_128, DSD_RATE_256,   "128->256 /44"    },
        { DSD48_RATE_256, DSD48_RATE_128, "256->128 /48"  },
        /* Also check these for potential improvement */
        { DSD_RATE_256, DSD_RATE_128,   "256->128 /44"    },
        { DSD_RATE_512, DSD_RATE_128,   "512->128 /44"    },
        { DSD48_RATE_128, DSD48_RATE_256, "128->256 /48"  },
    };
    static const int n_paths = sizeof(paths) / sizeof(paths[0]);

    static const ntf_filter_id_t filters[] = {
        NTF_CLANS_4, NTF_SDM_4, NTF_CLANS_5, NTF_SDM_5,
        NTF_CLANS_6, NTF_SDM_6, NTF_CLANS_7, NTF_SDM_7,
        NTF_CLANS_8, NTF_SDM_8
    };
    static const int n_filters = sizeof(filters) / sizeof(filters[0]);

    static const int nc_vals[] = { 2, 4, 8, 16, 32 };
    static const int n_nc = sizeof(nc_vals) / sizeof(nc_vals[0]);

    printf("\n=== fp64 Re-sweep: NTF x nc (Phase 1) ===\n");

    struct { ntf_filter_id_t filter; int nc; double sinad; } best[5];
    for (int p = 0; p < n_paths; p++)
        best[p].sinad = -999.0;

    for (int f = 0; f < n_filters; f++) {
        const ntf_filter_t *ft = ntf_get_filter(filters[f], DSD_RATE_128);
        const char *fn = ft ? ft->name : "?";

        for (int n = 0; n < n_nc; n++) {
            printf("  %-9s nc=%-2d:", fn, nc_vals[n]);
            fflush(stdout);

            for (int p = 0; p < n_paths; p++) {
                double s = measure_rateconv_sinad(
                    paths[p].fs_in, paths[p].fs_out,
                    filters[f], nc_vals[n], 4, 128, 0.0, 0.708f);
                printf("  %6.1f", s);
                if (s > best[p].sinad) {
                    best[p].sinad = s;
                    best[p].filter = filters[f];
                    best[p].nc = nc_vals[n];
                }
            }
            printf("\n");
            fflush(stdout);
        }
    }

    printf("\n  Phase 1 winners:\n");
    for (int p = 0; p < n_paths; p++) {
        const ntf_filter_t *ft = ntf_get_filter(best[p].filter, DSD_RATE_128);
        printf("    %-16s %-10s nc=%-2d  %.1f dB\n",
               paths[p].name, ft ? ft->name : "?", best[p].nc, best[p].sinad);
    }

    /* Phase 2: depth x lat on winners */
    static const int depths[] = { 4, 8, 16 };
    static const int n_depths = sizeof(depths) / sizeof(depths[0]);
    static const int lat_vals[] = { 16, 32, 64, 128 };
    static const int n_lats = sizeof(lat_vals) / sizeof(lat_vals[0]);

    printf("\n=== fp64 Re-sweep: depth x lat (Phase 2) ===\n");

    struct { int depth; int lat; double sinad; } best2[5];
    for (int p = 0; p < n_paths; p++) {
        best2[p].sinad = best[p].sinad;
        best2[p].depth = 4;
        best2[p].lat = 128;
    }

    for (int d = 0; d < n_depths; d++) {
        for (int l = 0; l < n_lats; l++) {
            printf("  d=%-2d lat=%-3d:", depths[d], lat_vals[l]);
            fflush(stdout);

            for (int p = 0; p < n_paths; p++) {
                double s = measure_rateconv_sinad(
                    paths[p].fs_in, paths[p].fs_out,
                    best[p].filter, best[p].nc, depths[d],
                    lat_vals[l], 0.0, 0.708f);
                printf("  %6.1f", s);
                if (s > best2[p].sinad) {
                    best2[p].sinad = s;
                    best2[p].depth = depths[d];
                    best2[p].lat = lat_vals[l];
                }
            }
            printf("\n");
            fflush(stdout);
        }
    }

    printf("\n  Final winners (fp64 pipeline):\n");
    for (int p = 0; p < n_paths; p++) {
        const ntf_filter_t *ft = ntf_get_filter(best[p].filter, DSD_RATE_128);
        printf("    %-16s %-10s nc=%-2d d=%-2d lat=%-3d  %.1f dB\n",
               paths[p].name, ft ? ft->name : "?", best[p].nc,
               best2[p].depth, best2[p].lat, best2[p].sinad);
    }

    TEST_ASSERT_TRUE(1, "fp64 re-sweep completed");
}

/* ═══════════════════════════════════════════════════════════════════════
 * DSD64/48→256/48 targeted sweep (still at 54.6 dB with clans-8/nc=2)
 * ═══════════════════════════════════════════════════════════════════════ */
static void test_64_48_to_256_48_sweep(void) {
    static const ntf_filter_id_t filters[] = {
        NTF_CLANS_4, NTF_SDM_4, NTF_CLANS_5, NTF_SDM_5,
        NTF_CLANS_6, NTF_SDM_6, NTF_CLANS_7, NTF_SDM_7,
        NTF_CLANS_8, NTF_SDM_8
    };
    static const int n_filters = sizeof(filters) / sizeof(filters[0]);
    static const int nc_vals[] = { 2, 4, 8, 16 };
    static const int n_nc = sizeof(nc_vals) / sizeof(nc_vals[0]);

    printf("\n=== DSD64/48->256/48 NTF x nc sweep ===\n");
    ntf_filter_id_t best_f = NTF_CLANS_8;
    int best_nc = 2;
    double best_s = -999.0;

    for (int f = 0; f < n_filters; f++) {
        const ntf_filter_t *ft = ntf_get_filter(filters[f], DSD_RATE_128);
        for (int n = 0; n < n_nc; n++) {
            double s = measure_rateconv_sinad(
                DSD48_RATE_64, DSD48_RATE_256,
                filters[f], nc_vals[n], 4, 128, 0.0, 0.708f);
            printf("  %-9s nc=%-2d: %.1f dB\n",
                   ft ? ft->name : "?", nc_vals[n], s);
            fflush(stdout);
            if (s > best_s) { best_s = s; best_f = filters[f]; best_nc = nc_vals[n]; }
        }
    }
    const ntf_filter_t *bft = ntf_get_filter(best_f, DSD_RATE_128);
    printf("\n  Winner: %s nc=%d → %.1f dB\n",
           bft ? bft->name : "?", best_nc, best_s);

    /* Phase 2: depth x lat on winner */
    static const int depths[] = { 4, 8, 16 };
    static const int lats[] = { 32, 64, 128 };
    double best_s2 = best_s;
    int best_d = 4, best_l = 128;
    for (int d = 0; d < 3; d++) {
        for (int l = 0; l < 3; l++) {
            double s = measure_rateconv_sinad(
                DSD48_RATE_64, DSD48_RATE_256,
                best_f, best_nc, depths[d], lats[l], 0.0, 0.708f);
            printf("  d=%-2d lat=%-3d: %.1f dB\n", depths[d], lats[l], s);
            fflush(stdout);
            if (s > best_s2) { best_s2 = s; best_d = depths[d]; best_l = lats[l]; }
        }
    }
    printf("\n  Final: %s nc=%d d=%d lat=%d → %.1f dB\n",
           bft ? bft->name : "?", best_nc, best_d, best_l, best_s2);
    TEST_ASSERT_TRUE(1, "64/48->256/48 sweep completed");
}

/* ═══════════════════════════════════════════════════════════════════════
 * →DSD512 NTF × limiter sweep (fp64 pipeline)
 * ═══════════════════════════════════════════════════════════════════════ */
static void test_dsd512_limiter_sweep(void) {
    typedef struct { uint32_t fs_in, fs_out; const char *name; } path_t;
    static const path_t paths[] = {
        { DSD_RATE_64,  DSD_RATE_512, "64->512"  },
        { DSD_RATE_128, DSD_RATE_512, "128->512" },
        { DSD_RATE_256, DSD_RATE_512, "256->512" },
    };
    static const int n_paths = 3;

    static const ntf_filter_id_t filters[] = {
        NTF_CLANS_5, NTF_SDM_5, NTF_CLANS_6, NTF_SDM_6,
        NTF_CLANS_7, NTF_SDM_7, NTF_CLANS_8, NTF_SDM_8
    };
    static const int n_filters = sizeof(filters) / sizeof(filters[0]);

    static const double limits[] = { 0.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0 };
    static const int n_limits = sizeof(limits) / sizeof(limits[0]);

    printf("\n=== DSD512 NTF x limiter sweep (fp64) ===\n");
    printf("  %-9s %8s", "NTF", "lim");
    for (int p = 0; p < n_paths; p++)
        printf(" %10s", paths[p].name);
    printf("\n");

    struct { ntf_filter_id_t filter; double limit; double sinad; } best[3];
    for (int p = 0; p < n_paths; p++) best[p].sinad = -999.0;

    for (int f = 0; f < n_filters; f++) {
        const ntf_filter_t *ft = ntf_get_filter(filters[f], DSD_RATE_512);
        for (int l = 0; l < n_limits; l++) {
            printf("  %-9s %6.1f:", ft ? ft->name : "?", limits[l]);
            fflush(stdout);
            for (int p = 0; p < n_paths; p++) {
                double s = measure_rateconv_sinad(
                    paths[p].fs_in, paths[p].fs_out,
                    filters[f], 2, 4, 128, limits[l], 0.708f);
                printf("  %8.1f", s);
                if (s > best[p].sinad) {
                    best[p].sinad = s;
                    best[p].filter = filters[f];
                    best[p].limit = limits[l];
                }
            }
            printf("\n");
            fflush(stdout);
        }
    }

    printf("\n  Winners:\n");
    for (int p = 0; p < n_paths; p++) {
        const ntf_filter_t *ft = ntf_get_filter(best[p].filter, DSD_RATE_512);
        printf("    %-10s %-10s lim=%.1f → %.1f dB\n",
               paths[p].name, ft ? ft->name : "?", best[p].limit, best[p].sinad);
    }
    TEST_ASSERT_TRUE(1, "DSD512 limiter sweep completed");
}

/* ═══════════════════════════════════════════════════════════════════════
 * DSD64→DSD512 focused sweep: NTF × nc × limiter (with 127-tap FIR)
 * ═══════════════════════════════════════════════════════════════════════ */
static void test_dsd64_to_512_sweep(void) {
    static const ntf_filter_id_t filters[] = {
        NTF_CLANS_5, NTF_SDM_5, NTF_CLANS_6, NTF_SDM_6,
        NTF_CLANS_7, NTF_SDM_7, NTF_CLANS_8, NTF_SDM_8
    };
    static const int n_filters = sizeof(filters) / sizeof(filters[0]);
    static const int nc_vals[] = { 2, 4, 8 };
    static const int n_nc = 3;
    static const double limits[] = { 0.0, 6.0, 10.0, 16.0 };
    static const int n_limits = 4;

    printf("\n=== DSD64->DSD512 NTF x nc x limiter sweep ===\n");

    ntf_filter_id_t best_f = NTF_CLANS_6;
    int best_nc = 2;
    double best_lim = 10.0;
    double best_s = -999.0;

    for (int f = 0; f < n_filters; f++) {
        const ntf_filter_t *ft = ntf_get_filter(filters[f], DSD_RATE_512);
        for (int n = 0; n < n_nc; n++) {
            for (int l = 0; l < n_limits; l++) {
                double s = measure_rateconv_sinad(
                    DSD_RATE_64, DSD_RATE_512,
                    filters[f], nc_vals[n], 4, 128, limits[l], 0.708f);
                printf("  %-9s nc=%-2d lim=%4.1f: %.1f dB\n",
                       ft ? ft->name : "?", nc_vals[n], limits[l], s);
                fflush(stdout);
                if (s > best_s) {
                    best_s = s; best_f = filters[f];
                    best_nc = nc_vals[n]; best_lim = limits[l];
                }
            }
        }
    }

    const ntf_filter_t *bft = ntf_get_filter(best_f, DSD_RATE_512);
    printf("\n  Winner: %s nc=%d lim=%.1f → %.1f dB\n",
           bft ? bft->name : "?", best_nc, best_lim, best_s);

    /* Phase 2: depth x lat on winner */
    static const int depths[] = { 4, 8, 16 };
    static const int lats[] = { 32, 64, 128 };
    double best_s2 = best_s;
    int best_d = 4, best_l = 128;
    printf("\n  Phase 2: depth x lat\n");
    for (int d = 0; d < 3; d++) {
        for (int ll = 0; ll < 3; ll++) {
            double s = measure_rateconv_sinad(
                DSD_RATE_64, DSD_RATE_512,
                best_f, best_nc, depths[d], lats[ll], best_lim, 0.708f);
            printf("  d=%-2d lat=%-3d: %.1f dB\n", depths[d], lats[ll], s);
            fflush(stdout);
            if (s > best_s2) { best_s2 = s; best_d = depths[d]; best_l = lats[ll]; }
        }
    }
    printf("\n  Final: %s nc=%d lim=%.1f d=%d lat=%d → %.1f dB\n",
           bft ? bft->name : "?", best_nc, best_lim, best_d, best_l, best_s2);
    TEST_ASSERT_TRUE(1, "DSD64->512 sweep completed");
}

/* ═══════════════════════════════════════════════════════════════════════
 * Median-aware re-sweep: uses rate-suite sample sizes + 3-freq median
 * for paths that regressed with multi-frequency median measurement.
 * ═══════════════════════════════════════════════════════════════════════ */
static double measure_median_sweep(uint32_t fs_in, uint32_t fs_out,
                                    ntf_filter_id_t filter_id,
                                    int nc, int depth, int lat,
                                    double state_limit, float gain) {
    /* Build a fake path_info for measure_rate_sinad_at */
    engine_path_info_t pi;
    memset(&pi, 0, sizeof(pi));
    pi.ntf_filter = (int)filter_id;
    pi.cands = nc;
    pi.lat = lat;
    pi.depth = depth;
    pi.state_limit = state_limit;
    pi.fir_gain = gain;

    double s1 = measure_rate_sinad_at(fs_in, fs_out, 900.0, &pi, nc, lat, depth, NULL);
    double s2 = measure_rate_sinad_at(fs_in, fs_out, 1000.0, &pi, nc, lat, depth, NULL);
    double s3 = measure_rate_sinad_at(fs_in, fs_out, 1100.0, &pi, nc, lat, depth, NULL);
    return median3(s1, s2, s3);
}

static void test_median_resweep(void) {
    typedef struct { uint32_t fs_in, fs_out; const char *name; } path_t;
    static const path_t paths[] = {
        { DSD_RATE_64,  DSD_RATE_256,  "64->256 /44"  },
        { DSD_RATE_128, DSD_RATE_64,   "128->64 /44"  },
        { DSD_RATE_64,  DSD_RATE_128,  "64->128 /44"  },
    };
    static const int n_paths = 3;

    static const ntf_filter_id_t filters[] = {
        NTF_CLANS_4, NTF_SDM_4, NTF_CLANS_5, NTF_SDM_5,
        NTF_CLANS_6, NTF_SDM_6, NTF_CLANS_7, NTF_SDM_7,
        NTF_CLANS_8, NTF_SDM_8
    };
    static const int n_filters = 10;
    static const int nc_vals[] = { 2, 4, 8, 16 };
    static const int n_nc = 4;

    printf("\n=== Median re-sweep (rate-suite samples, 3-freq median) ===\n");

    struct { ntf_filter_id_t filter; int nc; double sinad; } best[3];
    for (int p = 0; p < n_paths; p++) best[p].sinad = -999.0;

    for (int f = 0; f < n_filters; f++) {
        const ntf_filter_t *ft = ntf_get_filter(filters[f], DSD_RATE_128);
        for (int n = 0; n < n_nc; n++) {
            printf("  %-9s nc=%-2d:", ft ? ft->name : "?", nc_vals[n]);
            fflush(stdout);
            for (int p = 0; p < n_paths; p++) {
                double s = measure_median_sweep(
                    paths[p].fs_in, paths[p].fs_out,
                    filters[f], nc_vals[n], 4, 128, 0.0, 0.708f);
                printf("  %6.1f", s);
                if (s > best[p].sinad) {
                    best[p].sinad = s;
                    best[p].filter = filters[f];
                    best[p].nc = nc_vals[n];
                }
            }
            printf("\n"); fflush(stdout);
        }
    }

    printf("\n  Phase 1 winners:\n");
    for (int p = 0; p < n_paths; p++) {
        const ntf_filter_t *ft = ntf_get_filter(best[p].filter, DSD_RATE_128);
        printf("    %-14s %-10s nc=%-2d  %.1f dB\n",
               paths[p].name, ft ? ft->name : "?", best[p].nc, best[p].sinad);
    }

    /* Phase 2: depth x lat on winners */
    static const int depths[] = { 4, 8, 16 };
    static const int lats[] = { 32, 64, 128 };
    struct { int depth; int lat; double sinad; } best2[3];
    for (int p = 0; p < n_paths; p++) { best2[p].sinad = best[p].sinad; best2[p].depth = 4; best2[p].lat = 128; }

    printf("\n  Phase 2: depth x lat\n");
    for (int d = 0; d < 3; d++) {
        for (int l = 0; l < 3; l++) {
            printf("  d=%-2d lat=%-3d:", depths[d], lats[l]);
            fflush(stdout);
            for (int p = 0; p < n_paths; p++) {
                double s = measure_median_sweep(
                    paths[p].fs_in, paths[p].fs_out,
                    best[p].filter, best[p].nc, depths[d], lats[l], 0.0, 0.708f);
                printf("  %6.1f", s);
                if (s > best2[p].sinad) { best2[p].sinad = s; best2[p].depth = depths[d]; best2[p].lat = lats[l]; }
            }
            printf("\n"); fflush(stdout);
        }
    }

    printf("\n  Final winners (median, rate-suite samples):\n");
    for (int p = 0; p < n_paths; p++) {
        const ntf_filter_t *ft = ntf_get_filter(best[p].filter, DSD_RATE_128);
        printf("    %-14s %-10s nc=%-2d d=%-2d lat=%-3d  %.1f dB\n",
               paths[p].name, ft ? ft->name : "?", best[p].nc,
               best2[p].depth, best2[p].lat, best2[p].sinad);
    }
    TEST_ASSERT_TRUE(1, "median re-sweep completed");
}

/* ═══════════════════════════════════════════════════════════════════════
 * Lowpass cutoff sweep for same-rate re-encode quality
 * ═══════════════════════════════════════════════════════════════════════ */
static double measure_samerate_with_lp(uint32_t rate, double cutoff_hz, int ntaps) {
    unsigned base = rate_is_48k_family(rate) ? 48000 : 44100;
    unsigned mult = rate / base;
    size_t n_in = (mult <= 64) ? 262144 : (mult <= 128) ? 524288 :
                  (mult <= 256) ? 1048576 : 2097152;
    size_t max_out = n_in + 4096;

    /* Get path config */
    dsd_config_t cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.fs_in = rate; cfg.fs_out = rate;
    engine_path_info_t pi;
    engine_get_path_info(rate, rate, NTF_AUTO, SDM_MODE_TRELLIS, &cfg, &pi);
    int cands = pi.cands > 0 ? pi.cands : 2;
    int lat = pi.lat > 0 ? pi.lat : 32;
    int depth = pi.depth > 0 ? pi.depth : 4;

    size_t est_out = n_in - 512 - (size_t)lat;
    double freq = bin_align_freq(1000.0, (double)rate, est_out);

    /* Generate DSD input */
    float *dsd_in = (float *)malloc(n_in * sizeof(float));
    float *dsd_out = (float *)malloc(max_out * sizeof(float));
    if (!dsd_in || !dsd_out) { free(dsd_in); free(dsd_out); return -999.0; }
    size_t dsd_in_count = generate_dsd_sine(rate, freq, 0.5, n_in, dsd_in);
    if (dsd_in_count < 1024) { free(dsd_in); free(dsd_out); return -999.0; }

    /* FIR lowpass with custom cutoff */
    fir_lowpass_t lp;
    if (fir_lowpass_init_ex(&lp, rate, cutoff_hz, ntaps) != 0) {
        free(dsd_in); free(dsd_out); return -999.0;
    }
    double *lp_in = (double *)malloc(dsd_in_count * sizeof(double));
    double *lp_out = (double *)malloc(max_out * sizeof(double));
    if (!lp_in || !lp_out) {
        free(lp_in); free(lp_out); free(dsd_in); free(dsd_out);
        fir_lowpass_free(&lp); return -999.0;
    }
    for (size_t i = 0; i < dsd_in_count; i++) lp_in[i] = (double)dsd_in[i];
    size_t fir_count = fir_lowpass_process(&lp, lp_in, lp_out, dsd_in_count);
    free(lp_in);
    fir_lowpass_free(&lp);
    if (fir_count < 1024) { free(lp_out); free(dsd_in); free(dsd_out); return -999.0; }

    /* Apply gain */
    double gain = (double)pi.fir_gain;
    if (gain != 1.0) for (size_t i = 0; i < fir_count; i++) lp_out[i] *= gain;

    /* SDM re-encode */
    const ntf_filter_t *f = (pi.ntf_filter != NTF_AUTO) ?
        ntf_get_filter((ntf_filter_id_t)pi.ntf_filter, rate) : ntf_auto_select(rate);
    if (!f) { free(lp_out); free(dsd_in); free(dsd_out); return -999.0; }
    sdm_context_t sdm;
    if (sdm_context_init(&sdm, f, depth, cands, lat) != 0) {
        free(lp_out); free(dsd_in); free(dsd_out); return -999.0;
    }
    size_t out_count = sdm_process_block(&sdm, lp_out, dsd_out, fir_count);
    free(lp_out);
    sdm_context_free(&sdm);

    freq = bin_align_freq(freq, (double)rate, out_count);
    double sinad = (out_count > 1024) ?
        measure_sinad(dsd_out, out_count, freq, (double)rate) : -999.0;
    free(dsd_in); free(dsd_out);
    return sinad;
}

static void test_lowpass_cutoff_sweep(void) {
    static const uint32_t rates[] = {
        DSD_RATE_64, DSD_RATE_128, DSD_RATE_256, DSD_RATE_512
    };
    static const char *names[] = { "DSD64", "DSD128", "DSD256", "DSD512" };
    static const double cutoffs[] = { 22000, 25000, 30000, 35000, 40000, 50000 };
    static const int ntaps_vals[] = { 127, 255 };
    static const int n_rates = 4, n_cutoffs = 6, n_ntaps = 2;

    printf("\n=== Lowpass cutoff sweep (same-rate, single 1kHz) ===\n");
    printf("  %-8s", "cutoff");
    for (int r = 0; r < n_rates; r++) printf(" %8s", names[r]);
    printf("\n");

    for (int t = 0; t < n_ntaps; t++) {
        printf("\n  --- %d taps ---\n", ntaps_vals[t]);
        for (int c = 0; c < n_cutoffs; c++) {
            printf("  %5.0f Hz:", cutoffs[c]);
            fflush(stdout);
            for (int r = 0; r < n_rates; r++) {
                double s = measure_samerate_with_lp(rates[r], cutoffs[c], ntaps_vals[t]);
                printf("  %7.1f", s);
            }
            printf("\n"); fflush(stdout);
        }
    }
    TEST_ASSERT_TRUE(1, "lowpass cutoff sweep completed");
}

/* ═══════════════════════════════════════════════════════════════════════
 * Boxcar (DSD-Wide) vs FIR lowpass for same-rate re-encode
 * Tests whether 4-bit DSD-Wide intermediate improves over FIR lowpass
 * ═══════════════════════════════════════════════════════════════════════ */
static double measure_samerate_boxcar(uint32_t rate, int box_taps) {
    unsigned base = rate_is_48k_family(rate) ? 48000 : 44100;
    unsigned mult = rate / base;
    size_t n_in = (mult <= 64) ? 262144 : (mult <= 128) ? 524288 :
                  (mult <= 256) ? 1048576 : 2097152;

    /* Get path config */
    dsd_config_t cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.fs_in = rate; cfg.fs_out = rate;
    engine_path_info_t pi;
    engine_get_path_info(rate, rate, NTF_AUTO, SDM_MODE_TRELLIS, &cfg, &pi);
    int cands = pi.cands > 0 ? pi.cands : 2;
    int lat = pi.lat > 0 ? pi.lat : 32;
    int depth = pi.depth > 0 ? pi.depth : 4;

    size_t est_out = n_in - 512 - (size_t)lat;
    double freq = bin_align_freq(1000.0, (double)rate, est_out);

    /* Generate DSD input */
    float *dsd_in = (float *)malloc(n_in * sizeof(float));
    float *dsd_out = (float *)malloc(n_in * sizeof(float));
    if (!dsd_in || !dsd_out) { free(dsd_in); free(dsd_out); return -999.0; }
    size_t dsd_count = generate_dsd_sine(rate, freq, 0.5, n_in, dsd_in);
    if (dsd_count < 1024) { free(dsd_in); free(dsd_out); return -999.0; }

    /* Boxcar smoothing (DSD-Wide) — running average of box_taps 1-bit samples */
    double *smooth = (double *)calloc(dsd_count, sizeof(double));
    if (!smooth) { free(dsd_in); free(dsd_out); return -999.0; }
    double bsum = 0.0;
    double *ring = (double *)calloc(box_taps, sizeof(double));
    if (!ring) { free(smooth); free(dsd_in); free(dsd_out); return -999.0; }
    int bpos = 0;
    double inv_n = 1.0 / (double)box_taps;
    double gain = (double)pi.fir_gain;

    for (size_t i = 0; i < dsd_count; i++) {
        double s = (double)dsd_in[i];
        bsum -= ring[bpos];
        ring[bpos] = s;
        bsum += s;
        bpos = (bpos + 1) % box_taps;
        smooth[i] = bsum * inv_n * gain;
    }
    free(ring);

    /* SDM re-encode */
    const ntf_filter_t *f = (pi.ntf_filter != NTF_AUTO) ?
        ntf_get_filter((ntf_filter_id_t)pi.ntf_filter, rate) : ntf_auto_select(rate);
    if (!f) { free(smooth); free(dsd_in); free(dsd_out); return -999.0; }
    sdm_context_t sdm;
    if (sdm_context_init(&sdm, f, depth, cands, lat) != 0) {
        free(smooth); free(dsd_in); free(dsd_out); return -999.0;
    }
    size_t out_count = sdm_process_block(&sdm, smooth, dsd_out, dsd_count);
    free(smooth);
    sdm_context_free(&sdm);

    freq = bin_align_freq(freq, (double)rate, out_count);
    double sinad = (out_count > 1024) ?
        measure_sinad(dsd_out, out_count, freq, (double)rate) : -999.0;
    free(dsd_in); free(dsd_out);
    return sinad;
}

static void test_boxcar_vs_lowpass(void) {
    static const uint32_t rates[] = {
        DSD_RATE_64, DSD_RATE_128, DSD_RATE_256, DSD_RATE_512
    };
    static const char *names[] = { "DSD64", "DSD128", "DSD256", "DSD512" };
    static const int box_taps[] = { 8, 16, 32, 64 };
    static const int n_rates = 4, n_box = 4;

    printf("\n=== Boxcar (DSD-Wide) vs FIR lowpass for same-rate ===\n");
    printf("  %-12s", "method");
    for (int r = 0; r < n_rates; r++) printf(" %8s", names[r]);
    printf("\n");

    /* FIR lowpass baseline (current production) */
    printf("  FIR LP 50k:");
    fflush(stdout);
    for (int r = 0; r < n_rates; r++) {
        double s = measure_samerate_with_lp(rates[r], 50000.0, 127);
        printf("  %7.1f", s);
    }
    printf("\n"); fflush(stdout);

    /* Boxcar at different tap counts */
    for (int b = 0; b < n_box; b++) {
        printf("  Box %3d:   ", box_taps[b]);
        fflush(stdout);
        for (int r = 0; r < n_rates; r++) {
            double s = measure_samerate_boxcar(rates[r], box_taps[b]);
            printf("  %7.1f", s);
        }
        printf("\n"); fflush(stdout);
    }

    /* Also test boxcar with higher nc (nc=4, nc=8) at DSD64 */
    printf("\n  --- DSD64 boxcar nc sweep ---\n");
    static const int nc_vals[] = { 2, 4, 8, 16 };
    for (int n = 0; n < 4; n++) {
        /* Quick inline test with custom nc */
        size_t n_in = 262144;
        dsd_config_t cfg; memset(&cfg, 0, sizeof(cfg));
        cfg.fs_in = DSD_RATE_64; cfg.fs_out = DSD_RATE_64;
        engine_path_info_t pi;
        engine_get_path_info(DSD_RATE_64, DSD_RATE_64, NTF_AUTO, SDM_MODE_TRELLIS, &cfg, &pi);
        int lat = pi.lat > 0 ? pi.lat : 32;
        int depth = pi.depth > 0 ? pi.depth : 16;

        size_t est = n_in - 512 - (size_t)lat;
        double freq = bin_align_freq(1000.0, (double)DSD_RATE_64, est);

        float *din = (float *)malloc(n_in * sizeof(float));
        float *dout = (float *)malloc(n_in * sizeof(float));
        size_t dc = generate_dsd_sine(DSD_RATE_64, freq, 0.5, n_in, din);

        /* Boxcar 32 taps */
        double *sm = (double *)calloc(dc, sizeof(double));
        double *rng = (double *)calloc(32, sizeof(double));
        double bs = 0.0; int bp = 0;
        for (size_t i = 0; i < dc; i++) {
            double v = (double)din[i];
            bs -= rng[bp]; rng[bp] = v; bs += v;
            bp = (bp + 1) % 32;
            sm[i] = bs / 32.0 * (double)pi.fir_gain;
        }
        free(rng);

        const ntf_filter_t *f = (pi.ntf_filter != NTF_AUTO) ?
            ntf_get_filter((ntf_filter_id_t)pi.ntf_filter, DSD_RATE_64) : ntf_auto_select(DSD_RATE_64);
        sdm_context_t sdm;
        sdm_context_init(&sdm, f, depth, nc_vals[n], lat);
        size_t oc = sdm_process_block(&sdm, sm, dout, dc);
        free(sm);
        sdm_context_free(&sdm);

        freq = bin_align_freq(freq, (double)DSD_RATE_64, oc);
        double sinad = measure_sinad(dout, oc, freq, (double)DSD_RATE_64);
        printf("  Box32 nc=%-2d: %.1f dB\n", nc_vals[n], sinad);
        fflush(stdout);
        free(din); free(dout);
    }

    TEST_ASSERT_TRUE(1, "boxcar vs lowpass completed");
}

/* ═══════════════════════════════════════════════════════════════════════
 * Boxcar pre-smoothing for rate conversion paths
 * Test: boxcar → FIR → SDM vs raw DSD → FIR → SDM
 * ═══════════════════════════════════════════════════════════════════════ */
static double measure_rateconv_boxcar(uint32_t fs_in, uint32_t fs_out, int box_taps) {
    unsigned base = rate_is_48k_family(fs_in) ? 48000 : 44100;
    unsigned mult_in = fs_in / base;
    size_t n_in = (mult_in <= 64) ? 262144 : (mult_in <= 128) ? 524288 :
                  (mult_in <= 256) ? 1048576 : 2097152;
    size_t max_out = (fs_out >= fs_in) ?
        n_in * (fs_out / fs_in) + 4096 : n_in / 2 + 4096;

    dsd_config_t cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.fs_in = fs_in; cfg.fs_out = fs_out;
    engine_path_info_t pi;
    engine_get_path_info(fs_in, fs_out, NTF_AUTO, SDM_MODE_TRELLIS, &cfg, &pi);
    int cands = pi.cands > 0 ? pi.cands : 2;
    int lat = pi.lat > 0 ? pi.lat : 128;
    int depth = pi.depth > 0 ? pi.depth : 4;

    size_t est_in = n_in - 512;
    size_t est_fir = (fs_out >= fs_in) ?
        est_in * (fs_out / fs_in) : est_in / (fs_in / fs_out);
    size_t est_sdm = (est_fir > (size_t)lat) ? est_fir - (size_t)lat : 512;
    double freq = bin_align_freq(1000.0, (double)fs_out, est_sdm);

    /* Generate DSD input */
    float *dsd_in = (float *)malloc(n_in * sizeof(float));
    float *dsd_out = (float *)malloc(max_out * sizeof(float));
    if (!dsd_in || !dsd_out) { free(dsd_in); free(dsd_out); return -999.0; }
    size_t dsd_count = generate_dsd_sine(fs_in, freq, 0.5, n_in, dsd_in);
    if (dsd_count < 1024) { free(dsd_in); free(dsd_out); return -999.0; }

    /* Boxcar pre-smooth → multi-bit at input rate */
    double *smooth = (double *)malloc(dsd_count * sizeof(double));
    if (!smooth) { free(dsd_in); free(dsd_out); return -999.0; }
    if (box_taps > 0) {
        double bsum = 0.0;
        double *ring = (double *)calloc(box_taps, sizeof(double));
        int bpos = 0;
        double inv = 1.0 / (double)box_taps;
        for (size_t i = 0; i < dsd_count; i++) {
            double s = (double)dsd_in[i];
            bsum -= ring[bpos]; ring[bpos] = s; bsum += s;
            bpos = (bpos + 1) % box_taps;
            smooth[i] = bsum * inv;
        }
        free(ring);
    } else {
        /* No boxcar — raw DSD (baseline) */
        for (size_t i = 0; i < dsd_count; i++)
            smooth[i] = (double)dsd_in[i];
    }

    /* FIR rate conversion (fp64) */
    fir_chain_t fir;
    if (fir_chain_init_ex(&fir, fs_in, fs_out, true) != 0) {
        free(smooth); free(dsd_in); free(dsd_out); return -999.0;
    }
    double *fir_out = (double *)calloc(max_out, sizeof(double));
    if (!fir_out) { free(smooth); free(dsd_in); free(dsd_out); fir_chain_free(&fir); return -999.0; }
    size_t fir_count = fir_chain_process_d(&fir, smooth, fir_out, dsd_count);
    free(smooth);
    fir_chain_free(&fir);
    if (fir_count < 512) { free(fir_out); free(dsd_in); free(dsd_out); return -999.0; }

    /* Apply gain */
    double gain = (double)pi.fir_gain;
    if (gain != 1.0) for (size_t i = 0; i < fir_count; i++) fir_out[i] *= gain;

    /* SDM re-encode */
    const ntf_filter_t *f = (pi.ntf_filter != NTF_AUTO) ?
        ntf_get_filter((ntf_filter_id_t)pi.ntf_filter, fs_out) : ntf_auto_select(fs_out);
    if (!f) { free(fir_out); free(dsd_in); free(dsd_out); return -999.0; }
    sdm_context_t sdm;
    if (sdm_context_init(&sdm, f, depth, cands, lat) != 0) {
        free(fir_out); free(dsd_in); free(dsd_out); return -999.0;
    }
    if (pi.state_limit > 0.0) sdm.state_limit = pi.state_limit;
    size_t out_count = sdm_process_block(&sdm, fir_out, dsd_out, fir_count);
    free(fir_out);
    sdm_context_free(&sdm);

    freq = bin_align_freq(freq, (double)fs_out, out_count);
    double sinad = (out_count > 512) ?
        measure_sinad(dsd_out, out_count, freq, (double)fs_out) : -999.0;
    free(dsd_in); free(dsd_out);
    return sinad;
}

static void test_boxcar_rateconv(void) {
    typedef struct { uint32_t fs_in, fs_out; const char *name; } path_t;
    static const path_t paths[] = {
        { DSD_RATE_64,  DSD_RATE_128, "64->128" },
        { DSD_RATE_64,  DSD_RATE_256, "64->256" },
        { DSD_RATE_128, DSD_RATE_256, "128->256" },
        { DSD_RATE_128, DSD_RATE_64,  "128->64"  },
        { DSD_RATE_256, DSD_RATE_128, "256->128" },
    };
    static const int n_paths = 5;
    static const int box_vals[] = { 0, 4, 8, 16, 32 };
    static const int n_box = 5;

    printf("\n=== Boxcar pre-smooth for rate conversion ===\n");
    printf("  %-8s", "box");
    for (int p = 0; p < n_paths; p++) printf(" %9s", paths[p].name);
    printf("\n");

    for (int b = 0; b < n_box; b++) {
        printf("  %3d:   ", box_vals[b]);
        fflush(stdout);
        for (int p = 0; p < n_paths; p++) {
            double s = measure_rateconv_boxcar(
                paths[p].fs_in, paths[p].fs_out, box_vals[b]);
            printf("  %8.1f", s);
        }
        printf("\n"); fflush(stdout);
    }
    TEST_ASSERT_TRUE(1, "boxcar rateconv completed");
}

/* ═══════════════════════════════════════════════════════════════════════
 * Pre-SDM enhancement concept test
 * Pipeline: DSD → boxcar → [transform] → Trellis SDM → SINAD
 * Tests whether signal conditioning before SDM improves quality.
 * ═══════════════════════════════════════════════════════════════════════ */
static double measure_pre_sdm(uint32_t rate, int box_taps,
                                int transform_type, double param) {
    unsigned base = rate_is_48k_family(rate) ? 48000 : 44100;
    unsigned mult = rate / base;
    size_t n_in = (mult <= 64) ? 262144 : (mult <= 128) ? 524288 :
                  (mult <= 256) ? 1048576 : 2097152;

    dsd_config_t cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.fs_in = rate; cfg.fs_out = rate;
    engine_path_info_t pi;
    engine_get_path_info(rate, rate, NTF_AUTO, SDM_MODE_TRELLIS, &cfg, &pi);
    int cands = pi.cands > 0 ? pi.cands : 2;
    int lat = pi.lat > 0 ? pi.lat : 32;
    int depth = pi.depth > 0 ? pi.depth : 4;

    size_t est = n_in - 512 - (size_t)lat;
    double freq = bin_align_freq(1000.0, (double)rate, est);

    float *dsd_in = (float *)malloc(n_in * sizeof(float));
    float *dsd_out = (float *)malloc(n_in * sizeof(float));
    if (!dsd_in || !dsd_out) { free(dsd_in); free(dsd_out); return -999.0; }
    size_t dsd_count = generate_dsd_sine(rate, freq, 0.5, n_in, dsd_in);
    if (dsd_count < 1024) { free(dsd_in); free(dsd_out); return -999.0; }

    /* Boxcar smoothing */
    double *smooth = (double *)calloc(dsd_count, sizeof(double));
    if (!smooth) { free(dsd_in); free(dsd_out); return -999.0; }
    {
        double bsum = 0.0;
        double *ring = (double *)calloc(box_taps, sizeof(double));
        int bp = 0;
        double inv = 1.0 / (double)box_taps;
        double gain = (double)pi.fir_gain;
        for (size_t i = 0; i < dsd_count; i++) {
            double s = (double)dsd_in[i];
            bsum -= ring[bp]; ring[bp] = s; bsum += s;
            bp = (bp + 1) % box_taps;
            smooth[i] = bsum * inv * gain;
        }
        free(ring);
    }

    /* Pre-SDM transform */
    switch (transform_type) {
    case 0: /* Baseline: no transform */
        break;
    case 1: /* Pre-emphasis: y[n] = x[n] + param * (x[n] - x[n-1]) */
        for (size_t i = dsd_count - 1; i > 0; i--)
            smooth[i] = smooth[i] + param * (smooth[i] - smooth[i-1]);
        break;
    case 2: /* De-emphasis: first-order IIR lowpass, param = alpha */
        for (size_t i = 1; i < dsd_count; i++)
            smooth[i] = param * smooth[i] + (1.0 - param) * smooth[i-1];
        break;
    case 3: /* Gain scaling */
        for (size_t i = 0; i < dsd_count; i++)
            smooth[i] *= param;
        break;
    case 4: /* Shaped noise injection (param = noise level) */
        {
            unsigned seed = 12345;
            double prev_noise = 0.0;
            for (size_t i = 0; i < dsd_count; i++) {
                seed = seed * 1103515245 + 12345;
                double white = ((double)(seed >> 16) / 32768.0) - 1.0;
                double shaped = white - prev_noise;  /* first-order HPF noise */
                prev_noise = white;
                smooth[i] += shaped * param;
            }
        }
        break;
    case 5: /* Soft clip (tanh compression, param = drive) */
        for (size_t i = 0; i < dsd_count; i++)
            smooth[i] = tanh(smooth[i] * param) / tanh(param);
        break;
    }

    /* SDM re-encode */
    const ntf_filter_t *f = (pi.ntf_filter != NTF_AUTO) ?
        ntf_get_filter((ntf_filter_id_t)pi.ntf_filter, rate) : ntf_auto_select(rate);
    if (!f) { free(smooth); free(dsd_in); free(dsd_out); return -999.0; }
    sdm_context_t sdm;
    if (sdm_context_init(&sdm, f, depth, cands, lat) != 0) {
        free(smooth); free(dsd_in); free(dsd_out); return -999.0;
    }
    size_t out_count = sdm_process_block(&sdm, smooth, dsd_out, dsd_count);
    free(smooth);
    sdm_context_free(&sdm);

    freq = bin_align_freq(freq, (double)rate, out_count);
    double sinad = (out_count > 1024) ?
        measure_sinad(dsd_out, out_count, freq, (double)rate) : -999.0;
    free(dsd_in); free(dsd_out);
    return sinad;
}

static void test_pre_sdm_enhancement(void) {
    static const uint32_t rates[] = { DSD_RATE_64, DSD_RATE_128, DSD_RATE_256 };
    static const char *names[] = { "DSD64", "DSD128", "DSD256" };
    static const int box[] = { 32, 64, 64 };
    int n_rates = 3;

    printf("\n=== Pre-SDM Enhancement Concept Test ===\n");

    /* Baseline */
    printf("\n  Baseline (boxcar only):\n");
    for (int r = 0; r < n_rates; r++) {
        double s = measure_pre_sdm(rates[r], box[r], 0, 0.0);
        printf("    %-8s: %.1f dB\n", names[r], s);
    }

    /* Pre-emphasis sweep */
    printf("\n  Pre-emphasis (y += k*(x[n]-x[n-1])):\n");
    static const double pre_vals[] = { 0.01, 0.05, 0.1, 0.2, 0.5 };
    for (int p = 0; p < 5; p++) {
        printf("    k=%-5.2f:", pre_vals[p]);
        for (int r = 0; r < n_rates; r++) {
            double s = measure_pre_sdm(rates[r], box[r], 1, pre_vals[p]);
            printf("  %7.1f", s);
        }
        printf("\n");
    }

    /* De-emphasis sweep */
    printf("\n  De-emphasis (IIR LP, alpha):\n");
    static const double de_vals[] = { 0.99, 0.95, 0.9, 0.8 };
    for (int p = 0; p < 4; p++) {
        printf("    a=%-5.2f:", de_vals[p]);
        for (int r = 0; r < n_rates; r++) {
            double s = measure_pre_sdm(rates[r], box[r], 2, de_vals[p]);
            printf("  %7.1f", s);
        }
        printf("\n");
    }

    /* Gain scaling */
    printf("\n  Gain scaling:\n");
    static const double gain_vals[] = { 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5 };
    for (int p = 0; p < 7; p++) {
        printf("    g=%-5.2f:", gain_vals[p]);
        for (int r = 0; r < n_rates; r++) {
            double s = measure_pre_sdm(rates[r], box[r], 3, gain_vals[p]);
            printf("  %7.1f", s);
        }
        printf("\n");
    }

    /* Shaped noise injection */
    printf("\n  Shaped noise (HP dither):\n");
    static const double noise_vals[] = { 0.001, 0.005, 0.01, 0.05, 0.1 };
    for (int p = 0; p < 5; p++) {
        printf("    n=%-6.3f:", noise_vals[p]);
        for (int r = 0; r < n_rates; r++) {
            double s = measure_pre_sdm(rates[r], box[r], 4, noise_vals[p]);
            printf("  %7.1f", s);
        }
        printf("\n");
    }

    /* Soft clip */
    printf("\n  Soft clip (tanh drive):\n");
    static const double clip_vals[] = { 1.0, 2.0, 3.0, 5.0, 10.0 };
    for (int p = 0; p < 5; p++) {
        printf("    d=%-5.1f:", clip_vals[p]);
        for (int r = 0; r < n_rates; r++) {
            double s = measure_pre_sdm(rates[r], box[r], 5, clip_vals[p]);
            printf("  %7.1f", s);
        }
        printf("\n");
    }

    /* Fine-grained pre-emphasis sweep for all rates including DSD512 */
    printf("\n  --- Fine pre-emphasis sweep (all rates) ---\n");
    static const uint32_t all_rates[] = {
        DSD_RATE_64, DSD_RATE_128, DSD_RATE_256, DSD_RATE_512
    };
    static const char *all_names[] = { "DSD64", "DSD128", "DSD256", "DSD512" };
    static const int all_box[] = { 32, 64, 64, 16 };
    static const double fine_k[] = {
        0.0, 0.001, 0.002, 0.005, 0.007, 0.01, 0.015, 0.02,
        0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3
    };
    int n_fine = sizeof(fine_k) / sizeof(fine_k[0]);

    printf("  %-8s", "k");
    for (int r = 0; r < 4; r++) printf("  %7s", all_names[r]);
    printf("\n");

    double best_k[4] = {0}; double best_s[4] = {-999,-999,-999,-999};
    for (int p = 0; p < n_fine; p++) {
        printf("  %-8.3f", fine_k[p]);
        fflush(stdout);
        for (int r = 0; r < 4; r++) {
            double s = measure_pre_sdm(all_rates[r], all_box[r], 1, fine_k[p]);
            printf("  %7.1f", s);
            if (s > best_s[r]) { best_s[r] = s; best_k[r] = fine_k[p]; }
        }
        printf("\n"); fflush(stdout);
    }
    printf("\n  Optimal k per rate:\n");
    for (int r = 0; r < 4; r++)
        printf("    %-8s: k=%.3f → %.1f dB\n", all_names[r], best_k[r], best_s[r]);

    /* Signal-type dependency test: does optimal k vary with signal content?
     * If yes → adaptive ML is worth pursuing. If no → parametric is optimal. */
    printf("\n  --- Signal-type dependency (DSD512, k sweep) ---\n");
    {
        uint32_t rate = DSD_RATE_512;
        int btap = 16;
        size_t n = 2097152;
        dsd_config_t cfg2; memset(&cfg2, 0, sizeof(cfg2));
        cfg2.fs_in = rate; cfg2.fs_out = rate;
        engine_path_info_t pi2;
        engine_get_path_info(rate, rate, NTF_AUTO, SDM_MODE_TRELLIS, &cfg2, &pi2);
        int c2 = pi2.cands > 0 ? pi2.cands : 2;
        int l2 = pi2.lat > 0 ? pi2.lat : 32;
        int d2 = pi2.depth > 0 ? pi2.depth : 4;

        /* Test signals: sine, multitone, noise */
        static const char *sig_names[] = { "1kHz sine", "100Hz sine", "10kHz sine", "multitone 8" };
        static const double sig_freqs[] = { 1000, 100, 10000, 0 };  /* 0 = multitone */
        static const double test_k[] = { 0.0, 0.005, 0.007, 0.01, 0.02, 0.05 };

        printf("  %-14s", "signal\\k");
        for (int ki = 0; ki < 6; ki++) printf("  %7.3f", test_k[ki]);
        printf("\n");

        for (int si = 0; si < 4; si++) {
            printf("  %-14s", sig_names[si]);
            fflush(stdout);
            for (int ki = 0; ki < 6; ki++) {
                /* Generate signal */
                float *din = (float *)malloc(n * sizeof(float));
                float *dout = (float *)malloc(n * sizeof(float));
                double *smooth = (double *)calloc(n, sizeof(double));

                size_t est = n - 512 - (size_t)l2;
                double freq = bin_align_freq(sig_freqs[si] > 0 ? sig_freqs[si] : 1000.0, (double)rate, est);

                size_t dc;
                if (sig_freqs[si] > 0) {
                    dc = generate_dsd_sine(rate, freq, 0.5, n, din);
                } else {
                    /* Multitone: 8 tones */
                    const ntf_filter_t *fi = ntf_auto_select(rate);
                    sdm_context_t gen;
                    sdm_context_init(&gen, fi, 8, 16, 512);
                    double *mt = (double *)malloc(n * sizeof(double));
                    double amp = 0.5 / sqrt(8.0);
                    double freqs[8] = {100, 300, 1000, 2000, 4000, 7000, 10000, 14000};
                    for (size_t i = 0; i < n; i++) {
                        mt[i] = 0;
                        for (int t = 0; t < 8; t++)
                            mt[i] += amp * sin(2.0 * 3.14159265358979 * freqs[t] * (double)i / (double)rate);
                    }
                    dc = sdm_process_block(&gen, mt, din, n);
                    free(mt);
                    sdm_context_free(&gen);
                }

                /* Boxcar */
                double bsum = 0.0;
                double rng[128] = {0};
                int bp = 0;
                double inv = 1.0 / (double)btap;
                double gain = (double)pi2.fir_gain;
                for (size_t i = 0; i < dc; i++) {
                    double s = (double)din[i];
                    bsum -= rng[bp]; rng[bp] = s; bsum += s;
                    bp = (bp + 1) % btap;
                    smooth[i] = bsum * inv * gain;
                }

                /* Pre-emphasis */
                if (test_k[ki] > 0) {
                    for (size_t i = dc - 1; i > 0; i--)
                        smooth[i] += test_k[ki] * (smooth[i] - smooth[i-1]);
                }

                /* SDM */
                const ntf_filter_t *fo = (pi2.ntf_filter != NTF_AUTO) ?
                    ntf_get_filter((ntf_filter_id_t)pi2.ntf_filter, rate) : ntf_auto_select(rate);
                sdm_context_t sdm;
                sdm_context_init(&sdm, fo, d2, c2, l2);
                size_t oc = sdm_process_block(&sdm, smooth, dout, dc);
                sdm_context_free(&sdm);

                freq = bin_align_freq(freq, (double)rate, oc);
                double sinad = measure_sinad(dout, oc, freq, (double)rate);
                printf("  %7.1f", sinad);

                free(din); free(dout); free(smooth);
            }
            printf("\n"); fflush(stdout);
        }
    }

    TEST_ASSERT_TRUE(1, "pre-SDM enhancement test completed");
}

/* ═══════════════════════════════════════════════════════════════════════
 * CLI evaluator for CMA-ES optimizer: --preemph <rate_hz> <freq_hz> [taps...]
 * Outputs: single SINAD value (machine-parseable)
 * Pipeline: generate DSD sine → boxcar → apply FIR taps → SDM → Goertzel
 * ═══════════════════════════════════════════════════════════════════════ */
int preemph_eval_main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: test.exe --preemph [--precorr] <rate_hz> <freq_hz> [tap0 tap1 ...]\n");
        fprintf(stderr, "  --precorr: use PreCorr SDM instead of Trellis\n");
        fprintf(stderr, "  rate_hz: DSD rate (2822400, 5644800, 11289600, 22579200, + /48 variants)\n");
        fprintf(stderr, "  freq_hz: test frequency (e.g. 1000)\n");
        fprintf(stderr, "  taps: FIR pre-emphasis taps (default: 1.0 = identity)\n");
        fprintf(stderr, "Output: SINAD in dB (single number)\n");
        return 1;
    }

    /* Check for flags */
    int use_precorr = 0;
    int arg_start = 1;
    while (arg_start < argc && argv[arg_start][0] == '-') {
        if (strcmp(argv[arg_start], "--precorr") == 0)
            use_precorr = 1;
        else
            break;
        arg_start++;
    }

    if (argc - arg_start < 1) {
        fprintf(stderr, "Missing rate_hz\n");
        return 1;
    }

    uint32_t rate = (uint32_t)atoi(argv[arg_start]);
    double freq_hz = (argc - arg_start) >= 2 ? atof(argv[arg_start + 1]) : 1000.0;

    /* Parse FIR taps */
    int num_taps = argc - arg_start - 2;
    double taps[64] = {0};
    if (num_taps <= 0) {
        taps[0] = 1.0;
        num_taps = 1;
    } else {
        if (num_taps > 64) num_taps = 64;
        for (int i = 0; i < num_taps; i++)
            taps[i] = atof(argv[arg_start + 2 + i]);
    }

    /* Determine boxcar taps */
    unsigned base = rate_is_48k_family(rate) ? 48000 : 44100;
    unsigned mult = rate / base;
    int box_taps = (mult >= 512) ? 16 : (mult >= 128) ? 64 : 32;

    /* Get path config */
    int sdm_mode = use_precorr ? SDM_MODE_PRECORR : SDM_MODE_TRELLIS;
    dsd_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.fs_in = rate; cfg.fs_out = rate;
    engine_path_info_t pi;
    engine_get_path_info(rate, rate, NTF_AUTO, sdm_mode, &cfg, &pi);
    int cands = pi.cands > 0 ? pi.cands : 2;
    int lat = pi.lat > 0 ? pi.lat : 32;
    int depth = pi.depth > 0 ? pi.depth : 4;

    /* Sample count */
    size_t n_in = (mult <= 64) ? 262144 : (mult <= 128) ? 524288 :
                  (mult <= 256) ? 1048576 : 2097152;
    size_t est = n_in - 512 - (size_t)lat;
    double freq = bin_align_freq(freq_hz, (double)rate, est);

    /* Generate DSD sine */
    float *dsd_in = (float *)malloc(n_in * sizeof(float));
    float *dsd_out = (float *)malloc(n_in * sizeof(float));
    if (!dsd_in || !dsd_out) { free(dsd_in); free(dsd_out); fprintf(stderr, "OOM\n"); return 1; }
    size_t dsd_count = generate_dsd_sine(rate, freq, 0.5, n_in, dsd_in);
    if (dsd_count < 1024) { free(dsd_in); free(dsd_out); fprintf(stderr, "gen fail\n"); return 1; }

    /* Boxcar smoothing */
    double *smooth = (double *)calloc(dsd_count, sizeof(double));
    if (!smooth) { free(dsd_in); free(dsd_out); return 1; }
    {
        double bsum = 0.0;
        double *ring = (double *)calloc(box_taps, sizeof(double));
        int bp = 0;
        double inv = 1.0 / (double)box_taps;
        double gain = (double)pi.fir_gain;
        for (size_t i = 0; i < dsd_count; i++) {
            double s = (double)dsd_in[i];
            bsum -= ring[bp]; ring[bp] = s; bsum += s;
            bp = (bp + 1) % box_taps;
            smooth[i] = bsum * inv * gain;
        }
        free(ring);
    }

    /* Apply FIR pre-emphasis (causal convolution) */
    if (num_taps > 1 || taps[0] != 1.0) {
        double *tmp = (double *)malloc(dsd_count * sizeof(double));
        if (tmp) {
            for (size_t i = 0; i < dsd_count; i++) {
                double y = 0.0;
                for (int t = 0; t < num_taps; t++) {
                    if (i >= (size_t)t)
                        y += taps[t] * smooth[i - t];
                }
                tmp[i] = y;
            }
            memcpy(smooth, tmp, dsd_count * sizeof(double));
            free(tmp);
        }
    }

    /* SDM re-encode */
    size_t out_count;
    {
        const ntf_filter_t *f;
        if (use_precorr) {
            f = ntf_auto_select_precorr(rate);
        } else {
            f = (pi.ntf_filter != NTF_AUTO) ?
                ntf_get_filter((ntf_filter_id_t)pi.ntf_filter, rate) : ntf_auto_select(rate);
        }
        if (!f) { free(smooth); free(dsd_in); free(dsd_out); return 1; }

        if (use_precorr) {
            precorr_context_t pc;
            if (precorr_context_init(&pc, f) != 0) {
                free(smooth); free(dsd_in); free(dsd_out); return 1;
            }
            out_count = precorr_process_block(&pc, smooth, dsd_out, dsd_count);
            free(smooth);
            precorr_context_free(&pc);
        } else {
            sdm_context_t sdm;
            if (sdm_context_init(&sdm, f, depth, cands, lat) != 0) {
                free(smooth); free(dsd_in); free(dsd_out); return 1;
            }
            out_count = sdm_process_block(&sdm, smooth, dsd_out, dsd_count);
            free(smooth);
            sdm_context_free(&sdm);
        }
    }

    /* Measure SINAD (flat + A-weighted) — end-to-end pipeline */
    freq = bin_align_freq(freq, (double)rate, out_count);
    double awtd = -999.0;
    double sinad = (out_count > 1024) ?
        measure_sinad_ex(dsd_out, out_count, freq, (double)rate, &awtd) : -999.0;

    /* Output: SINAD A-wtd */
    printf("%.2f %.2f\n", sinad, awtd);

    free(dsd_in); free(dsd_out);
    return 0;
}

/* ─── Full-pipeline MT sweep: NTF × nc × depth × lat for same-rate ───
 * Real pipeline: DSD multitone → boxcar → SDM re-encode → MT measurement.
 * Sweeps Trellis and PreCorr. Ranks by MT. */

static const double g_mt_freqs[32] = {
    17, 21, 27, 34, 42, 53, 67, 85, 107, 134, 169, 213, 268, 337,
    424, 534, 672, 846, 1065, 1340, 1687, 2124, 2674, 3366, 4237,
    5334, 6714, 8452, 10640, 13396, 16863, 21228
};

/* Measure MT through full boxcar pipeline for a given SDM config.
 * Returns MT SINAD in dB. Also fills awtd_out with A-weighted single-tone SINAD. */
static double measure_pipeline_mt(uint32_t rate, ntf_filter_id_t ntf_id,
                                   int nc, int depth, int lat, int use_precorr,
                                   double *awtd_out) {
    unsigned base = rate_is_48k_family(rate) ? 48000 : 44100;
    unsigned mult = rate / base;
    unsigned n_dsd = (mult <= 64) ? 262144u : (mult <= 128) ? 524288u :
                     (mult <= 256) ? 1048576u : 2097152u;
    int box_taps = (mult >= 512) ? 16 : (mult >= 128) ? 64 : 32;

    /* Use a safe low-order NTF for the source SDM generator.
     * ntf_auto_select may return aggressive NTFs that diverge on multitone.
     * CLANS-4 is stable at all rates with nc=16 and high-amplitude input. */
    const ntf_filter_t *f_gen = ntf_get_filter(NTF_CLANS_4, rate);
    if (!f_gen) f_gen = ntf_auto_select(rate);
    if (!f_gen) return -999.0;

    const ntf_filter_t *f_re = ntf_get_filter(ntf_id, rate);
    if (!f_re) {
        if (use_precorr)
            f_re = ntf_auto_select_precorr(rate);
        else
            f_re = ntf_auto_select(rate);
    }
    if (!f_re) return -999.0;

    float  *dsd_src = (float *)malloc(n_dsd * sizeof(float));
    float  *dsd_out = (float *)malloc(n_dsd * sizeof(float));
    double *smooth  = (double *)calloc(n_dsd, sizeof(double));
    if (!dsd_src || !dsd_out || !smooth) {
        free(dsd_src); free(dsd_out); free(smooth);
        return -999.0;
    }

    /* Bin-align frequencies to the CANDIDATE SDM output count.
     * This ensures Goertzel measures at exact bin centers. */
    #define GEN_LAT 512
    size_t out_est = n_dsd - GEN_LAT - (unsigned)lat;  /* source removes GEN_LAT, candidate removes lat */
    if (use_precorr) out_est = n_dsd - GEN_LAT;  /* PreCorr has no latency */
    double out_bw = (double)rate / (double)out_est;

    double aligned_freqs[32];
    unsigned aligned_bins[32];
    for (int t = 0; t < 32; t++) {
        aligned_bins[t] = (unsigned)(g_mt_freqs[t] / out_bw + 0.5);
        aligned_freqs[t] = aligned_bins[t] * out_bw;
    }

    /* ── Generate DSD multitone source ── */
    size_t gen_count;
    {
        sdm_context_t gen;
        if (sdm_context_init(&gen, f_gen, 8, 16, GEN_LAT) != 0) {
            free(dsd_src); free(dsd_out); free(smooth); return -999.0;
        }
        double *pcm = (double *)malloc(n_dsd * sizeof(double));
        if (!pcm) { sdm_context_free(&gen); free(dsd_src); free(dsd_out); free(smooth); return -999.0; }
        double amp = 0.3 / sqrt(32.0);  /* Conservative: peak ~0.6, safe for all NTFs */
        for (unsigned i = 0; i < n_dsd; i++) {
            double s = 0.0;
            for (int t = 0; t < 32; t++)
                s += amp * sin(2.0 * M_PI * aligned_freqs[t] * (double)i / (double)rate);
            pcm[i] = s;
        }
        gen_count = sdm_process_block(&gen, pcm, dsd_src, n_dsd);
        free(pcm);
        sdm_context_free(&gen);
        if (gen_count < 1024) { free(dsd_src); free(dsd_out); free(smooth); return -999.0; }
    }

    /* ── Boxcar smooth with -3 dB gain (matches production pipeline) ── */
    {
        double bsum = 0.0, inv = 1.0 / (double)box_taps;
        double *ring = (double *)calloc(box_taps, sizeof(double));
        int bp = 0;
        double gain = 0.708;  /* -3 dB, production default */
        for (size_t i = 0; i < gen_count; i++) {
            double s = (double)dsd_src[i];
            bsum -= ring[bp]; ring[bp] = s; bsum += s;
            bp = (bp + 1) % box_taps;
            smooth[i] = bsum * inv * gain;
        }
        free(ring);
    }

    /* ── SDM re-encode ── */
    size_t produced;
    if (use_precorr) {
        precorr_context_t pc;
        if (precorr_context_init(&pc, f_re) != 0) {
            free(dsd_src); free(dsd_out); free(smooth); return -999.0;
        }
        produced = precorr_process_block(&pc, smooth, dsd_out, gen_count);
        precorr_context_free(&pc);
    } else {
        sdm_context_t sdm;
        if (sdm_context_init(&sdm, f_re, (unsigned)depth, (unsigned)nc, (unsigned)lat) != 0) {
            free(dsd_src); free(dsd_out); free(smooth); return -999.0;
        }
        produced = sdm_process_block(&sdm, smooth, dsd_out, gen_count);
        sdm_context_free(&sdm);
    }

    if (produced < 1024) {
        free(dsd_src); free(dsd_out); free(smooth); return -999.0;
    }

    /* ── Measure MT on output ── */
    double actual_bw = (double)rate / (double)produced;
    unsigned max_bin = (unsigned)(20000.0 / actual_bw);
    double total_signal = 0.0;
    unsigned sig_bins[32];
    for (int t = 0; t < 32; t++) {
        /* Re-align to actual produced bin grid */
        sig_bins[t] = (unsigned)(aligned_freqs[t] / actual_bw + 0.5);
        double meas_freq = sig_bins[t] * actual_bw;
        total_signal += goertzel_power(dsd_out, produced, meas_freq, (double)rate);
    }
    double noise = 0.0;
    for (unsigned b = 1; b <= max_bin; b++) {
        int is_sig = 0;
        for (int t = 0; t < 32; t++) {
            if (b >= sig_bins[t] - 1 && b <= sig_bins[t] + 1) { is_sig = 1; break; }
        }
        if (!is_sig)
            noise += goertzel_power(dsd_out, produced, b * actual_bw, (double)rate);
    }
    double mt = (noise > 0.0) ? 10.0 * log10(total_signal / noise) : -999.0;

    /* Debug: print per-tone signal power for DSD128 to diagnose MT floor */
    if (rate == 5644800 && mt < 60.0) {
        printf("\n      [MT-DBG] produced=%zu actual_bw=%.4f max_bin=%u sig=%.1f noise=%.1f mt=%.1f\n",
               produced, actual_bw, max_bin, 10.0*log10(total_signal), 10.0*log10(noise > 0 ? noise : 1e-30), mt);
        for (int t = 0; t < 32; t++) {
            double mf = sig_bins[t] * actual_bw;
            double pwr = goertzel_power(dsd_out, produced, mf, (double)rate);
            printf("      [MT-DBG]  f=%.1f bin=%u pwr=%.1e (%.1f dB)\n",
                   g_mt_freqs[t], sig_bins[t], pwr, 10.0*log10(pwr > 0 ? pwr : 1e-30));
        }
    }

    /* ── Also measure single-tone A-weighted SINAD ── */
    if (awtd_out) {
        /* Re-run with 1kHz sine through the same pipeline */
        double freq = bin_align_freq(1000.0, (double)rate, out_est);
        size_t gen2_count;
        {
            sdm_context_t gen2;
            sdm_context_init(&gen2, f_gen, 8, 16, GEN_LAT);
            double *pcm1k = (double *)malloc(n_dsd * sizeof(double));
            for (unsigned i = 0; i < n_dsd; i++)
                pcm1k[i] = 0.5 * sin(2.0 * M_PI * freq * (double)i / (double)rate);
            gen2_count = sdm_process_block(&gen2, pcm1k, dsd_src, n_dsd);
            free(pcm1k);
            sdm_context_free(&gen2);
        }
        /* Boxcar with -3 dB gain */
        {
            double bsum = 0.0, inv = 1.0 / (double)box_taps;
            double *ring = (double *)calloc(box_taps, sizeof(double));
            int bp = 0;
            for (size_t i = 0; i < gen2_count; i++) {
                double s = (double)dsd_src[i];
                bsum -= ring[bp]; ring[bp] = s; bsum += s;
                bp = (bp + 1) % box_taps;
                smooth[i] = bsum * inv * 0.708;
            }
            free(ring);
        }
        /* Re-encode */
        if (use_precorr) {
            precorr_context_t pc2;
            precorr_context_init(&pc2, f_re);
            produced = precorr_process_block(&pc2, smooth, dsd_out, gen2_count);
            precorr_context_free(&pc2);
        } else {
            sdm_context_t sdm2;
            sdm_context_init(&sdm2, f_re, (unsigned)depth, (unsigned)nc, (unsigned)lat);
            produced = sdm_process_block(&sdm2, smooth, dsd_out, gen2_count);
            sdm_context_free(&sdm2);
        }
        freq = bin_align_freq(freq, (double)rate, produced);
        measure_sinad_ex(dsd_out, produced, freq, (double)rate, awtd_out);
    }
    #undef GEN_LAT

    free(dsd_src); free(dsd_out); free(smooth);
    return mt;
}

/* Two-pass smart sweep:
 * Pass 1: NTF × lat (nc=2, depth=4) — find best NTF and lat
 * Pass 2: depth × nc for winning NTF+lat — fine-tune
 * Skips unstable combos: order-8 NTFs at DSD64, sdm-8 everywhere. */
static void test_mt_pipeline_sweep(void) {
    uint32_t rates[] = {
        DSD_RATE_64, DSD_RATE_128, DSD_RATE_256, DSD_RATE_512,
        DSD48_RATE_64, DSD48_RATE_128, DSD48_RATE_256, DSD48_RATE_512,
    };
    const char *rate_names[] = {
        "DSD64", "DSD128", "DSD256", "DSD512",
        "DSD64/48", "DSD128/48", "DSD256/48", "DSD512/48",
    };
    ntf_filter_id_t ntfs[] = {
        NTF_CLANS_4, NTF_SDM_4, NTF_CLANS_5, NTF_SDM_5,
        NTF_CLANS_6, NTF_SDM_6, NTF_CLANS_7, NTF_SDM_7,
        NTF_CLANS_8,
    };
    const char *ntf_names[] = {
        "clans-4", "sdm-4", "clans-5", "sdm-5",
        "clans-6", "sdm-6", "clans-7", "sdm-7",
        "clans-8",
    };
    int n_ntfs = sizeof(ntfs) / sizeof(ntfs[0]);
    int n_rates = sizeof(rates) / sizeof(rates[0]);

    int pass1_lats[] = { 32, 64, 128, 256 };
    int n_p1_lats = sizeof(pass1_lats) / sizeof(pass1_lats[0]);

    int pass2_depths[] = { 4, 8, 16 };
    int pass2_nc[] = { 2, 4, 8 };
    int n_p2_depths = sizeof(pass2_depths) / sizeof(pass2_depths[0]);
    int n_p2_nc = sizeof(pass2_nc) / sizeof(pass2_nc[0]);

    for (int sdm = 0; sdm < 2; sdm++) {
        int use_precorr = (sdm == 0) ? 0 : 1;
        printf("\n=== %s Full-Pipeline MT Sweep ===\n",
               use_precorr ? "PRECORR" : "TRELLIS");

        for (int r = 0; r < n_rates; r++) {
            unsigned base = rate_is_48k_family(rates[r]) ? 48000 : 44100;
            unsigned mult = rates[r] / base;

            printf("\n  %s:\n", rate_names[r]);

            double best_mt = -999.0, best_awtd = -999.0;
            int best_ntf_idx = -1, best_nc = 2, best_depth = 4, best_lat = 32;

            if (use_precorr) {
                /* PreCorr: sweep NTF only (no nc/depth/lat) */
                printf("    Pass 1: NTF sweep\n");
                for (int fi = 0; fi < n_ntfs; fi++) {
                    const ntf_filter_t *fcheck = ntf_get_filter(ntfs[fi], rates[r]);
                    if (!fcheck) continue;
                    /* Skip order-8 at DSD64 (unstable) */
                    if (mult <= 64 && fcheck->order >= 8) continue;

                    double awtd = -999.0;
                    double mt = measure_pipeline_mt(rates[r], ntfs[fi],
                        2, 4, 32, 1, &awtd);
                    printf("      %-9s: MT=%6.1f A-wtd=%6.1f\n",
                           ntf_names[fi], mt, awtd);
                    if (mt > best_mt) {
                        best_mt = mt; best_awtd = awtd;
                        best_ntf_idx = fi;
                    }
                }
            } else {
                /* Trellis Pass 1: NTF × lat (nc=2, depth=4) */
                printf("    Pass 1: NTF x lat (nc=2, d=4)\n");
                for (int fi = 0; fi < n_ntfs; fi++) {
                    const ntf_filter_t *fcheck = ntf_get_filter(ntfs[fi], rates[r]);
                    if (!fcheck) continue;
                    if (mult <= 64 && fcheck->order >= 8) continue;

                    for (int li = 0; li < n_p1_lats; li++) {
                        double awtd = -999.0;
                        double mt = measure_pipeline_mt(rates[r], ntfs[fi],
                            2, 4, pass1_lats[li], 0, &awtd);
                        printf("      %-9s lat=%3d: MT=%6.1f A-wtd=%6.1f%s\n",
                               ntf_names[fi], pass1_lats[li], mt, awtd,
                               mt > best_mt ? " *" : "");
                        if (mt > best_mt) {
                            best_mt = mt; best_awtd = awtd;
                            best_ntf_idx = fi; best_lat = pass1_lats[li];
                        }
                    }
                }

                /* Trellis Pass 2: depth × nc for winning NTF+lat */
                if (best_ntf_idx >= 0) {
                    printf("    Pass 2: depth x nc (NTF=%s, lat=%d)\n",
                           ntf_names[best_ntf_idx], best_lat);
                    for (int di = 0; di < n_p2_depths; di++) {
                        for (int ni = 0; ni < n_p2_nc; ni++) {
                            if (pass2_depths[di] == 4 && pass2_nc[ni] == 2)
                                continue;  /* already tested in pass 1 */
                            double awtd = -999.0;
                            double mt = measure_pipeline_mt(rates[r], ntfs[best_ntf_idx],
                                pass2_nc[ni], pass2_depths[di], best_lat, 0, &awtd);
                            printf("      d=%2d nc=%d: MT=%6.1f A-wtd=%6.1f%s\n",
                                   pass2_depths[di], pass2_nc[ni], mt, awtd,
                                   mt > best_mt ? " *" : "");
                            if (mt > best_mt) {
                                best_mt = mt; best_awtd = awtd;
                                best_nc = pass2_nc[ni]; best_depth = pass2_depths[di];
                            }
                        }
                    }
                }
            }

            /* Print winner */
            if (use_precorr) {
                printf("    >>> BEST: %-9s MT=%.1f A-wtd=%.1f\n",
                       best_ntf_idx >= 0 ? ntf_names[best_ntf_idx] : "?",
                       best_mt, best_awtd);
            } else {
                printf("    >>> BEST: %-9s nc=%d d=%d lat=%d MT=%.1f A-wtd=%.1f\n",
                       best_ntf_idx >= 0 ? ntf_names[best_ntf_idx] : "?",
                       best_nc, best_depth, best_lat,
                       best_mt, best_awtd);
            }
        }
    }

    TEST_ASSERT_TRUE(1, "MT pipeline sweep completed");
}

void test_mt_sweep_suite(void) {
    TEST_SUITE("MT Pipeline Sweep");
    TEST_RUN(test_mt_pipeline_sweep);
}

void test_downsample_sweep_suite(void) {
    TEST_SUITE("Downsample Sweep");
    TEST_RUN(test_downsample_sweep);
    TEST_RUN(test_48k_downsample_sweep);
    TEST_RUN(test_fp64_resweep);
    TEST_RUN(test_64_48_to_256_48_sweep);
    TEST_RUN(test_dsd512_limiter_sweep);
    TEST_RUN(test_dsd64_to_512_sweep);
    TEST_RUN(test_median_resweep);
    TEST_RUN(test_lowpass_cutoff_sweep);
    TEST_RUN(test_boxcar_vs_lowpass);
    TEST_RUN(test_boxcar_rateconv);
    TEST_RUN(test_pre_sdm_enhancement);
}

void test_rate_sweep_suite(void) {
    TEST_SUITE("Rate Conversion Sweep");

    TEST_RUN(test_diag_pcm_fir_control);
    TEST_RUN(test_diag_fir_only);
    TEST_RUN(test_diag_gain_limiter);
    TEST_RUN(test_diag_limiter_sweep);
    TEST_RUN(test_comprehensive_ntf_limiter_sweep);
    TEST_RUN(test_cands_latency_sweep);
    TEST_RUN(test_gain_sweep);
    TEST_RUN(test_weak_paths_sweep);
    TEST_RUN(test_depth16_spot_check);
}
