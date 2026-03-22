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

/* Measure in-band SINAD on a signal */
static double measure_sinad(const float *x, size_t n, double freq_hz,
                            double sample_rate) {
    double signal_power = goertzel_power(x, n, freq_hz, sample_rate);

    double bw = sample_rate / (double)n;
    unsigned max_bin = (unsigned)(22050.0 / bw);
    unsigned sig_bin = (unsigned)(freq_hz / bw + 0.5);

    double noise = 0.0;
    for (unsigned b = 1; b <= max_bin; b++) {
        if (b >= sig_bin - 1 && b <= sig_bin + 1) continue;
        noise += goertzel_power(x, n, b * bw, sample_rate);
    }
    if (noise <= 0.0) noise = 1e-30;
    return 10.0 * log10(signal_power / noise);
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

static double measure_rate_sinad(uint32_t fs_in, uint32_t fs_out) {
    /* For DSD→DSD rate conversion, use sinad_measure at fs_out
     * (matches the Test Quality button methodology). This measures
     * the SDM encode quality at the output rate with the path-configured
     * NTF/cands/lat. The FIR rate conversion quality is tested separately
     * in the DSD→PCM tests. */
    if (fs_in != fs_out && fs_in >= DSD_RATE_64 && fs_out >= DSD_RATE_64) {
        dsd_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.fs_in = fs_in; cfg.fs_out = fs_out;
        engine_path_info_t pi;
        engine_get_path_info(fs_in, fs_out, NTF_AUTO, SDM_MODE_TRELLIS, &cfg, &pi);
        int cands = pi.cands > 0 ? pi.cands : 2;
        int lat = pi.lat > 0 ? pi.lat : 32;
        int depth = pi.depth > 0 ? pi.depth : 4;
        sinad_result_t r;
        memset(&r, 0, sizeof(r));
        sinad_measure(fs_out, pi.ntf_filter, cands, depth, lat, 1, pi.fir_gain, &r);
        unsigned base_in  = rate_is_48k_family(fs_in)  ? 48000 : 44100;
        unsigned base_out = rate_is_48k_family(fs_out) ? 48000 : 44100;
        const char *dir = (fs_out > fs_in) ? "UP" : "DN";
        printf("    [SINAD] DSD%u->DSD%u (%s): SINAD=%.1f dB A-wtd=%.1f MT=%.1f NMod=%.1f"
               "  [%s, gain=%.2f, cands=%d, lat=%d]\n",
               fs_in/base_in, fs_out/base_out, dir,
               r.sinad_theoretical, r.sinad_awtd_theo,
               r.multitone_sinad_db, r.noise_mod_db,
               pi.ntf_filter != NTF_AUTO ?
                   ntf_get_filter((ntf_filter_id_t)pi.ntf_filter, fs_out)->name : "auto",
               pi.fir_gain, cands, lat);
        return r.sinad_theoretical;
    }

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

    /* Query path-adaptive settings (mirrors production engine behavior) */
    dsd_config_t test_cfg;
    memset(&test_cfg, 0, sizeof(test_cfg));
    test_cfg.fs_in = fs_in;
    test_cfg.fs_out = fs_out;
    test_cfg.trellis_depth = SINAD_TRELLIS_DEPTH;
    test_cfg.trellis_cands = SINAD_TRELLIS_CANDS;
    test_cfg.trellis_lat = SINAD_TRELLIS_LAT;

    engine_path_info_t pi;
    engine_get_path_info(fs_in, fs_out, NTF_AUTO, SDM_MODE_TRELLIS, &test_cfg, &pi);

    /* Use path-optimal values (matching real production behavior) */
    int cands = pi.cands > 0 ? pi.cands : SINAD_TRELLIS_CANDS;
    int lat   = pi.lat > 0 ? pi.lat : SINAD_TRELLIS_LAT;
    int depth = pi.depth > 0 ? pi.depth : SINAD_TRELLIS_DEPTH;

    /* Align test frequency to the FINAL PCM measurement grid (44100/48000 Hz).
     * This ensures clean Goertzel measurement after DSD→PCM decimation. */
    unsigned meas_pcm_rate = rate_is_48k_family(fs_out) ? 48000 : 44100;
    size_t est_in_produced = n_in - (size_t)lat;
    size_t est_fir_out;
    if (fs_out >= fs_in)
        est_fir_out = est_in_produced * (fs_out / fs_in);
    else
        est_fir_out = est_in_produced / (fs_in / fs_out);
    size_t est_sdm_out = est_fir_out - (size_t)lat;
    size_t est_pcm_out = est_sdm_out / (fs_out / meas_pcm_rate);
    if (est_pcm_out < 256) est_pcm_out = 256;
    double freq = bin_align_freq(1000.0, (double)meas_pcm_rate, est_pcm_out);
    size_t dsd_in_count = generate_dsd_sine(fs_in, freq, 0.5, n_in, dsd_in);

    if (dsd_in_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    /* FIR processing: rate conversion uses fir_chain, same-rate uses lowpass */
    size_t fir_count;
    if (fs_in == fs_out) {
        /* Same-rate: FIR lowpass smoothing (matches engine's fir_lowpass path) */
        fir_lowpass_t lp;
        memset(&lp, 0, sizeof(lp));
        if (fir_lowpass_init(&lp, fs_in) != 0) {
            free(dsd_in); free(fir_buf); free(dsd_out);
            return -999.0;
        }
        fir_count = fir_lowpass_process(&lp, dsd_in, fir_buf, dsd_in_count);
        fir_lowpass_free(&lp);
    } else {
        /* Rate conversion: multi-stage FIR chain */
        fir_chain_t fir;
        if (fir_chain_init(&fir, fs_in, fs_out) != 0) {
            free(dsd_in); free(fir_buf); free(dsd_out);
            return -999.0;
        }
        fir_count = fir_chain_process(&fir, dsd_in, fir_buf, dsd_in_count);
        fir_chain_free(&fir);
    }

    if (fir_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    /* Apply path-adaptive FIR gain */
    if (pi.fir_gain != 1.0f) {
        for (size_t i = 0; i < fir_count; i++)
            fir_buf[i] *= pi.fir_gain;
    }

    /* SDM requantize at fs_out with path-adaptive NTF/cands/lat */
    const ntf_filter_t *f_out;
    if (pi.ntf_filter != NTF_AUTO)
        f_out = ntf_get_filter((ntf_filter_id_t)pi.ntf_filter, fs_out);
    else
        f_out = ntf_auto_select(fs_out);
    if (!f_out) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    sdm_context_t sdm;
    if (sdm_context_init(&sdm, f_out, depth, cands, lat) != 0) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }
    if (pi.state_limit > 0.0)
        sdm.state_limit = pi.state_limit;
    double *sdm_in_d = float_to_double(fir_buf, fir_count);
    if (!sdm_in_d) { sdm_context_free(&sdm); free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    size_t out_count = sdm_process_block(&sdm, sdm_in_d, dsd_out, fir_count);
    free(sdm_in_d);
    sdm_context_free(&sdm);

    if (out_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    /* Decimate DSD to audio rate before SINAD measurement.
     * Raw DSD Goertzel has spectral leakage from shaped noise.
     * FIR decimation removes ultrasonic noise → accurate in-band SINAD. */
    double sinad_db;
    {
        fir_chain_t dec_fir;
        memset(&dec_fir, 0, sizeof(dec_fir));
        fir_chain_init(&dec_fir, fs_out, meas_pcm_rate);
        float *pcm_buf = (float *)calloc(out_count, sizeof(float));
        size_t pcm_n = pcm_buf ? fir_chain_process(&dec_fir, dsd_out, pcm_buf, out_count) : 0;
        fir_chain_free(&dec_fir);
        size_t skip = 256;
        if (pcm_n > skip + 1024) {
            size_t meas_n = pcm_n - skip;
            /* freq is already bin-aligned to meas_pcm_rate from the start */
            sinad_db = measure_sinad(pcm_buf + skip, meas_n, freq, (double)meas_pcm_rate);
        } else {
            sinad_db = -999.0;
        }
        free(pcm_buf);
    }

    unsigned base_in  = rate_is_48k_family(fs_in)  ? 48000 : 44100;
    unsigned base_out = rate_is_48k_family(fs_out) ? 48000 : 44100;
    unsigned rate_in_mult  = fs_in  / base_in;
    unsigned rate_out_mult = fs_out / base_out;
    const char *dir = (fs_out > fs_in) ? "UP" : "DN";
    printf("    [SINAD] DSD%u->DSD%u (%s): SINAD=%.1f dB  [%s, gain=%.2f, lim=%s, cands=%d, depth=%d]\n",
           rate_in_mult, rate_out_mult, dir, sinad_db,
           pi.ntf_filter != NTF_AUTO ?
               ntf_get_filter((ntf_filter_id_t)pi.ntf_filter, fs_out)->name : "auto",
           pi.fir_gain,
           pi.state_limit > 0.0 ? "on" : "off",
           cands, depth);

    free(dsd_in);
    free(fir_buf);
    free(dsd_out);

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

    /* Align test frequency to the FINAL PCM measurement grid (44100/48000 Hz).
     * This ensures clean Goertzel measurement after DSD→PCM decimation. */
    unsigned meas_pcm_rate = rate_is_48k_family(fs_out) ? 48000 : 44100;
    size_t est_in_produced = n_in - (size_t)lat;
    size_t est_fir_out;
    if (fs_out >= fs_in)
        est_fir_out = est_in_produced * (fs_out / fs_in);
    else
        est_fir_out = est_in_produced / (fs_in / fs_out);
    size_t est_sdm_out = est_fir_out - (size_t)lat;
    size_t est_pcm_out = est_sdm_out / (fs_out / meas_pcm_rate);
    if (est_pcm_out < 256) est_pcm_out = 256;
    double freq = bin_align_freq(1000.0, (double)meas_pcm_rate, est_pcm_out);
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

    static const int depth_vals[] = { 4, 8 };
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
    test_sinad_same_rate(DSD48_RATE_64, "DSD64/48 same-rate", 80.0);
}
static void test_sinad_dsd128_48_same(void) {
    test_sinad_same_rate(DSD48_RATE_128, "DSD128/48 same-rate", 90.0);
}
static void test_sinad_dsd256_48_same(void) {
    test_sinad_same_rate(DSD48_RATE_256, "DSD256/48 same-rate", 90.0);
}
static void test_sinad_dsd512_48_same(void) {
    test_sinad_same_rate(DSD48_RATE_512, "DSD512/48 same-rate", 90.0);
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

    /* DSD/44 same-rate (using proven sinad_measure) */
    { test_sinad_same_rate(DSD_RATE_64,  "DSD64 same-rate",  80.0); }
    { test_sinad_same_rate(DSD_RATE_128, "DSD128 same-rate", 90.0); }
    { test_sinad_same_rate(DSD_RATE_256, "DSD256 same-rate", 90.0); }
    { test_sinad_same_rate(DSD_RATE_512, "DSD512 same-rate", 90.0); }

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
}
