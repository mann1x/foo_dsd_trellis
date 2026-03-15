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
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    float *sine = (float *)malloc(n_samples * sizeof(float));
    if (!sine) { sdm_context_free(&ctx); return 0; }

    for (size_t i = 0; i < n_samples; i++)
        sine[i] = (float)(amplitude * sin(2.0 * M_PI * freq_hz *
                                          (double)i / (double)dsd_rate));

    size_t produced = sdm_process_block(&ctx, sine, dsd_out, n_samples);

    free(sine);
    sdm_context_free(&ctx);
    return produced;
}

/* ─── Measure SINAD for a rate conversion path ─── */

static double measure_rate_sinad(uint32_t fs_in, uint32_t fs_out) {
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

    /* FIR rate conversion */
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

    /* SDM requantize at fs_out */
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
    size_t out_count = sdm_process_block(&sdm, fir_buf, dsd_out, fir_count);
    sdm_context_free(&sdm);

    if (out_count < 1024) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    double sinad_db = measure_sinad(dsd_out, out_count, freq, (double)fs_out);

    unsigned rate_in_mult  = fs_in  / 44100;
    unsigned rate_out_mult = fs_out / 44100;
    const char *dir = (fs_out > fs_in) ? "UP" : "DN";
    printf("    [SINAD] DSD%u->DSD%u (%s): SINAD=%.1f dB\n",
           rate_in_mult, rate_out_mult, dir, sinad_db);

    free(dsd_in);
    free(fir_buf);
    free(dsd_out);

    return sinad_db;
}

/* ─── Upsample tests ─── */

static void test_sinad_up_64_128(void) {
    double sinad = measure_rate_sinad(DSD_RATE_64, DSD_RATE_128);
    TEST_ASSERT_TRUE(sinad > 50.0,
                     "DSD64->DSD128 SINAD should exceed 50 dB");
}

static void test_sinad_up_64_256(void) {
    double sinad = measure_rate_sinad(DSD_RATE_64, DSD_RATE_256);
    TEST_ASSERT_TRUE(sinad > 50.0,
                     "DSD64->DSD256 SINAD should exceed 50 dB");
}

static void test_sinad_up_64_512(void) {
    double sinad = measure_rate_sinad(DSD_RATE_64, DSD_RATE_512);
    TEST_ASSERT_TRUE(sinad > 50.0,
                     "DSD64->DSD512 SINAD should exceed 50 dB");
}

static void test_sinad_up_128_256(void) {
    double sinad = measure_rate_sinad(DSD_RATE_128, DSD_RATE_256);
    TEST_ASSERT_TRUE(sinad > 50.0,
                     "DSD128->DSD256 SINAD should exceed 50 dB");
}

static void test_sinad_up_128_512(void) {
    double sinad = measure_rate_sinad(DSD_RATE_128, DSD_RATE_512);
    TEST_ASSERT_TRUE(sinad > 50.0,
                     "DSD128->DSD512 SINAD should exceed 50 dB");
}

static void test_sinad_up_256_512(void) {
    double sinad = measure_rate_sinad(DSD_RATE_256, DSD_RATE_512);
    TEST_ASSERT_TRUE(sinad > 50.0,
                     "DSD256->DSD512 SINAD should exceed 50 dB");
}

/* ─── Downsample tests ─── */

static void test_sinad_dn_128_64(void) {
    double sinad = measure_rate_sinad(DSD_RATE_128, DSD_RATE_64);
    TEST_ASSERT_TRUE(sinad > 50.0,
                     "DSD128->DSD64 SINAD should exceed 50 dB");
}

static void test_sinad_dn_256_64(void) {
    double sinad = measure_rate_sinad(DSD_RATE_256, DSD_RATE_64);
    TEST_ASSERT_TRUE(sinad > 50.0,
                     "DSD256->DSD64 SINAD should exceed 50 dB");
}

static void test_sinad_dn_512_64(void) {
    double sinad = measure_rate_sinad(DSD_RATE_512, DSD_RATE_64);
    TEST_ASSERT_TRUE(sinad > 50.0,
                     "DSD512->DSD64 SINAD should exceed 50 dB");
}

static void test_sinad_dn_256_128(void) {
    double sinad = measure_rate_sinad(DSD_RATE_256, DSD_RATE_128);
    TEST_ASSERT_TRUE(sinad > 50.0,
                     "DSD256->DSD128 SINAD should exceed 50 dB");
}

static void test_sinad_dn_512_128(void) {
    double sinad = measure_rate_sinad(DSD_RATE_512, DSD_RATE_128);
    TEST_ASSERT_TRUE(sinad > 50.0,
                     "DSD512->DSD128 SINAD should exceed 50 dB");
}

static void test_sinad_dn_512_256(void) {
    double sinad = measure_rate_sinad(DSD_RATE_512, DSD_RATE_256);
    TEST_ASSERT_TRUE(sinad > 50.0,
                     "DSD512->DSD256 SINAD should exceed 50 dB");
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
    size_t out_count = sdm_process_block(&sdm, fir_buf, dsd_out, fir_count);
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
    float *sine = (float *)malloc(n_in * sizeof(float));
    if (!sine) { sdm_context_free(&gen); free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    for (size_t i = 0; i < n_in; i++)
        sine[i] = (float)(0.5 * sin(2.0 * M_PI * freq * (double)i / (double)fs_in));
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
    size_t out_count = sdm_process_block(&sdm, fir_buf, dsd_out, fir_count);
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
    float *sine = (float *)malloc(n_in * sizeof(float));
    if (!sine) { sdm_context_free(&gen); free(dsd_in); free(fir_buf); free(dsd_out); return -999.0; }
    for (size_t i = 0; i < n_in; i++)
        sine[i] = (float)(0.5 * sin(2.0 * M_PI * freq * (double)i / (double)fs_in));
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
    size_t out_count = sdm_process_block(&sdm, fir_buf, dsd_out, fir_count);
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

/* ─── Suites ─── */

void test_rate_sinad_suite(void) {
    TEST_SUITE("Rate Conversion SINAD");

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
}

void test_rate_sweep_suite(void) {
    TEST_SUITE("Rate Conversion Sweep");

    TEST_RUN(test_diag_fir_only);
    TEST_RUN(test_diag_limiter_sweep);
    TEST_RUN(test_comprehensive_ntf_limiter_sweep);
    TEST_RUN(test_cands_latency_sweep);
}
