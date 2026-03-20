/*
 * foo_dsd_trellis — Polyphase resampler tests
 */

#include "test.h"
#include "../include/resample.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Goertzel single-bin power */
static double goertzel_power_rs(const float *x, size_t n, double freq_hz,
                                 double sample_rate) {
    double k = freq_hz * (double)n / sample_rate;
    double w = 2.0 * M_PI * k / (double)n;
    double coeff = 2.0 * cos(w);
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (size_t i = 0; i < n; i++) {
        s0 = (double)x[i] + coeff * s1 - s2;
        s2 = s1; s1 = s0;
    }
    double real = s1 - s2 * cos(w);
    double imag = s2 * sin(w);
    return (real * real + imag * imag) / ((double)n * (double)n);
}

/* SINAD measurement — limit to max_bins for speed */
static double measure_sinad_fast(const float *x, size_t n, double freq_hz,
                                  double sample_rate, unsigned max_bins) {
    double signal = goertzel_power_rs(x, n, freq_hz, sample_rate);
    double bw = sample_rate / (double)n;
    unsigned sig_bin = (unsigned)(freq_hz / bw + 0.5);
    unsigned top_bin = (unsigned)(20000.0 / bw);
    if (top_bin > max_bins) top_bin = max_bins;
    double noise = 0.0;
    for (unsigned b = 1; b <= top_bin; b++) {
        if (b >= sig_bin - 1 && b <= sig_bin + 1) continue;
        noise += goertzel_power_rs(x, n, b * bw, sample_rate);
    }
    if (noise <= 0.0) noise = 1e-30;
    return 10.0 * log10(signal / noise);
}

static double bin_align(double target, double fs, size_t n) {
    double bw = fs / (double)n;
    return (unsigned)(target / bw + 0.5) * bw;
}

static void generate_sine(float *out, size_t n, double freq, double fs, double amp) {
    for (size_t i = 0; i < n; i++)
        out[i] = (float)(amp * sin(2.0 * M_PI * freq * (double)i / fs));
}

/* Helper: resample and measure SINAD */
static double resample_sinad(uint32_t fs_in, uint32_t fs_out, int engine) {
    size_t n_in = fs_in;  /* 1 second of input */
    size_t n_out = fs_out + 1024;

    float *in  = (float *)malloc(n_in * sizeof(float));
    float *out = (float *)malloc(n_out * sizeof(float));
    if (!in || !out) { free(in); free(out); return -999.0; }

    /* First pass: discover actual output count */
    generate_sine(in, n_in, 1000.0, (double)fs_in, 0.9);
    resample_ctx_t *ctx = resample_create(fs_in, fs_out, engine, SOXR_QUALITY_HQ);
    if (!ctx) { free(in); free(out); return -999.0; }
    size_t produced = resample_process(ctx, in, out, n_in);
    resample_free(ctx);

    if (produced < 1000) { free(in); free(out); return -999.0; }

    /* Second pass: bin-aligned tone */
    double freq = bin_align(1000.0, (double)fs_out, produced);
    generate_sine(in, n_in, freq, (double)fs_in, 0.9);
    ctx = resample_create(fs_in, fs_out, engine, SOXR_QUALITY_HQ);
    if (!ctx) { free(in); free(out); return -999.0; }
    produced = resample_process(ctx, in, out, n_in);
    resample_free(ctx);

    double meas_freq = bin_align(freq, (double)fs_out, produced);
    /* Use max 2000 bins to keep Goertzel fast */
    double sinad = measure_sinad_fast(out, produced, meas_freq, (double)fs_out, 2000);

    free(in); free(out);
    return sinad;
}

/* ─── Tests ─── */

static void test_resample_needed(void) {
    TEST_ASSERT_FALSE(resample_needed(44100, 88200), "44.1k->88.2k power-of-2");
    TEST_ASSERT_FALSE(resample_needed(48000, 96000), "48k->96k power-of-2");
    TEST_ASSERT_FALSE(resample_needed(44100, 176400), "44.1k->176.4k power-of-2");
    TEST_ASSERT_FALSE(resample_needed(44100, 44100), "same rate");
    TEST_ASSERT_TRUE(resample_needed(44100, 48000), "44.1k->48k cross-family");
    TEST_ASSERT_TRUE(resample_needed(48000, 44100), "48k->44.1k cross-family");
    TEST_ASSERT_TRUE(resample_needed(88200, 96000), "88.2k->96k cross-family");
}

static void test_resample_ipp_create(void) {
    resample_ctx_t *ctx = resample_create(44100, 48000, RESAMPLE_IPP, SOXR_QUALITY_HQ);
    TEST_ASSERT_TRUE(ctx != NULL, "IPP create for 44.1k->48k");
    if (ctx) {
        TEST_ASSERT_TRUE(strcmp(resample_engine_name(ctx), "IPP") == 0, "engine=IPP");
        resample_free(ctx);
    }
}

static void test_resample_ipp_44_48(void) {
    double sinad = resample_sinad(44100, 48000, RESAMPLE_IPP);
    printf("    IPP 44.1k->48k SINAD: %.1f dB\n", sinad);
    TEST_ASSERT_TRUE(sinad > 50.0, "SINAD > 50 dB");
}

static void test_resample_ipp_48_44(void) {
    double sinad = resample_sinad(48000, 44100, RESAMPLE_IPP);
    printf("    IPP 48k->44.1k SINAD: %.1f dB\n", sinad);
    TEST_ASSERT_TRUE(sinad > 50.0, "SINAD > 50 dB");
}

static void test_resample_ipp_96_88(void) {
    double sinad = resample_sinad(96000, 88200, RESAMPLE_IPP);
    printf("    IPP 96k->88.2k SINAD: %.1f dB\n", sinad);
    TEST_ASSERT_TRUE(sinad > 50.0, "SINAD > 50 dB");
}

static void test_resample_ipp_192_176(void) {
    double sinad = resample_sinad(192000, 176400, RESAMPLE_IPP);
    printf("    IPP 192k->176.4k SINAD: %.1f dB\n", sinad);
    TEST_ASSERT_TRUE(sinad > 50.0, "SINAD > 50 dB");
}

static void test_resample_soxr_available(void) {
    bool avail = resample_soxr_available();
    printf("    libsoxr available: %s\n", avail ? "yes" : "no");
}

static void test_resample_soxr_if_available(void) {
    if (!resample_soxr_available()) {
        printf("    [SKIP] libsoxr not available\n");
        return;
    }
    double sinad = resample_sinad(44100, 48000, RESAMPLE_SOXR);
    printf("    soxr 44.1k->48k SINAD: %.1f dB\n", sinad);
    TEST_ASSERT_TRUE(sinad > 100.0, "soxr VHQ SINAD > 100 dB");
}

void test_resample_suite(void) {
    TEST_SUITE("Resample");
    TEST_RUN(test_resample_needed);
    TEST_RUN(test_resample_ipp_create);
    TEST_RUN(test_resample_ipp_44_48);
    TEST_RUN(test_resample_ipp_48_44);
    TEST_RUN(test_resample_ipp_96_88);
    TEST_RUN(test_resample_ipp_192_176);
    TEST_RUN(test_resample_soxr_available);
    TEST_RUN(test_resample_soxr_if_available);
}
