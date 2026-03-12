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

/*
 * Pre-compute a bin-aligned frequency near target_hz for Goertzel on
 * `n_produced` samples at `sample_rate`.  The frequency must land on an
 * exact DFT bin of the MEASURED signal to avoid spectral leakage.
 */
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

    /* For multi-stage FIR, the output buffer is used for intermediate
     * results (ping-pong).  For downsampling, the first stage writes
     * n_in/2 samples into 'out'.  Size for the worst case. */
    size_t max_out;
    if (fs_out >= fs_in)
        max_out = n_in * (fs_out / fs_in) + 4096;
    else
        max_out = n_in / 2 + 4096;  /* first downsample stage output */

    float *dsd_in  = (float *)malloc(n_in * sizeof(float));
    float *fir_buf = (float *)malloc(max_out * sizeof(float));
    float *dsd_out = (float *)malloc(max_out * sizeof(float));
    if (!dsd_in || !fir_buf || !dsd_out) {
        free(dsd_in); free(fir_buf); free(dsd_out);
        return -999.0;
    }

    /* Estimate output count to bin-align frequency to the MEASURED signal */
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

    printf("    [SINAD] DSD%u->DSD%u (%s): %zu->%zu samples, SINAD=%.1f dB\n",
           rate_in_mult, rate_out_mult, dir,
           dsd_in_count, out_count, sinad_db);

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

/* ─── Suite ─── */

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
