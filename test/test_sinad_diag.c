/*
 * Diagnostic SINAD tests — isolate anomalies:
 * 1. Trellis DSD256 (115.8 dB) < DSD128 (117.4 dB)
 * 2. PreCorr DSD128 (107.6 dB) too low
 *
 * Uses smaller sample counts for sweeps to keep runtime reasonable.
 * Goertzel noise measurement is O(n * bins), so n must stay modest.
 */
#include "test.h"
#include "../include/trellis.h"
#include "../include/precorr.h"
#include "../include/ntf.h"
#include "../include/dsd_types.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Goertzel single-bin power */
static double goertzel_pwr(const float *x, size_t n, double freq_hz,
                            double fs) {
    double k = freq_hz * (double)n / fs;
    double w = 2.0 * M_PI * k / (double)n;
    double coeff = 2.0 * cos(w);
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (size_t i = 0; i < n; i++) {
        s0 = (double)x[i] + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    double re = s1 - s2 * cos(w);
    double im = s2 * sin(w);
    return (re * re + im * im) / ((double)n * (double)n);
}

/* SINAD: signal bin vs all other bins in 0-22050 Hz */
static double calc_sinad(const float *x, size_t n, double freq_hz, double fs) {
    double sig = goertzel_pwr(x, n, freq_hz, fs);
    double bw = fs / (double)n;
    unsigned max_bin = (unsigned)(22050.0 / bw);
    unsigned sig_bin = (unsigned)(freq_hz / bw + 0.5);
    double noise = 0.0;
    for (unsigned b = 1; b <= max_bin; b++) {
        if (b >= sig_bin - 1 && b <= sig_bin + 1) continue;
        noise += goertzel_pwr(x, n, b * bw, fs);
    }
    if (noise <= 0.0) noise = 1e-30;
    return 10.0 * log10(sig / noise);
}

/* Helper: run trellis SDM and measure SINAD */
static double trellis_sinad(unsigned dsd_rate, const ntf_filter_t *f,
                             int depth, int cands, int lat, unsigned n_dsd) {
    unsigned produced_est = n_dsd - (unsigned)lat;
    double bw = (double)dsd_rate / (double)produced_est;
    unsigned sig_bin = (unsigned)(1000.0 / bw + 0.5);
    double freq = sig_bin * bw;

    sdm_context_t ctx;
    if (sdm_context_init(&ctx, f, depth, cands, lat) != 0)
        return -999.0;

    double *in = (double *)malloc(n_dsd * sizeof(double));
    float *out = (float *)malloc(n_dsd * sizeof(float));
    for (unsigned j = 0; j < n_dsd; j++)
        in[j] = 0.5 * sin(2.0 * M_PI * freq * j / dsd_rate);

    size_t produced = sdm_process_block(&ctx, in, out, n_dsd);
    double sinad = calc_sinad(out, produced, freq, (double)dsd_rate);

    free(in); free(out);
    sdm_context_free(&ctx);
    return sinad;
}

/*
 * Diag 1: Trellis at DSD256 with different NTF orders
 * Uses 262144 samples (~23ms at DSD256) for speed
 */
static void test_diag_trellis_dsd256_ntf_sweep(void) {
    unsigned dsd_rate = DSD_RATE_256;
    ntf_filter_id_t ids[] = {NTF_CLANS_5, NTF_CLANS_6, NTF_CLANS_7, NTF_CLANS_8};
    const char *names[] = {"clans-5", "clans-6", "clans-7", "clans-8"};

    printf("\n    --- Trellis DSD256 NTF sweep (depth=8, cands=16, lat=256) ---\n");
    for (int i = 0; i < 4; i++) {
        const ntf_filter_t *f = ntf_get_filter(ids[i], dsd_rate);
        if (!f) { printf("    %s: N/A\n", names[i]); continue; }
        double s = trellis_sinad(dsd_rate, f, 8, 16, 256, 262144);
        printf("    %s (order %d): SINAD=%.1f dB\n", names[i], f->order, s);
    }
    TEST_ASSERT_TRUE(1, "NTF sweep completed");
}

/*
 * Diag 2: Trellis DSD128 with different NTF orders (for comparison)
 */
static void test_diag_trellis_dsd128_ntf_sweep(void) {
    unsigned dsd_rate = DSD_RATE_128;
    ntf_filter_id_t ids[] = {NTF_CLANS_5, NTF_CLANS_6, NTF_CLANS_7};
    const char *names[] = {"clans-5", "clans-6", "clans-7"};

    printf("\n    --- Trellis DSD128 NTF sweep (depth=8, cands=16, lat=256) ---\n");
    for (int i = 0; i < 3; i++) {
        const ntf_filter_t *f = ntf_get_filter(ids[i], dsd_rate);
        if (!f) { printf("    %s: N/A\n", names[i]); continue; }
        double s = trellis_sinad(dsd_rate, f, 8, 16, 256, 262144);
        printf("    %s (order %d): SINAD=%.1f dB\n", names[i], f->order, s);
    }
    TEST_ASSERT_TRUE(1, "NTF sweep completed");
}

/*
 * Diag 3: Trellis DSD256 clans-7 with varying latency
 */
static void test_diag_trellis_dsd256_latency(void) {
    unsigned dsd_rate = DSD_RATE_256;
    const ntf_filter_t *f = ntf_auto_select(dsd_rate);
    int lats[] = {64, 128, 256, 512};

    printf("\n    --- Trellis DSD256 clans-7 latency sweep ---\n");
    for (int i = 0; i < 4; i++) {
        double s = trellis_sinad(dsd_rate, f, 8, 16, lats[i], 262144);
        printf("    lat=%d: SINAD=%.1f dB\n", lats[i], s);
    }
    TEST_ASSERT_TRUE(1, "latency sweep completed");
}

/*
 * Diag 4: Trellis DSD256 clans-7 with varying candidates
 */
static void test_diag_trellis_dsd256_cands(void) {
    unsigned dsd_rate = DSD_RATE_256;
    const ntf_filter_t *f = ntf_auto_select(dsd_rate);
    int cands[] = {4, 8, 16, 32};

    printf("\n    --- Trellis DSD256 clans-7 cands sweep ---\n");
    for (int i = 0; i < 4; i++) {
        double s = trellis_sinad(dsd_rate, f, 8, cands[i], 256, 262144);
        printf("    cands=%d: SINAD=%.1f dB\n", cands[i], s);
    }
    TEST_ASSERT_TRUE(1, "cands sweep completed");
}

/*
 * Diag 5: PreCorr DSD128 warmup sweep — does startup transient hurt SINAD?
 */
static void test_diag_precorr_dsd128_warmup(void) {
    unsigned dsd_rate = DSD_RATE_128;
    const ntf_filter_t *f = ntf_auto_select(dsd_rate);

    printf("\n    --- PreCorr DSD128 warmup sweep ---\n");

    size_t total_n = 564480;  /* 0.1 seconds */
    double bw = (double)dsd_rate / (double)total_n;
    unsigned bin = (unsigned)(1000.0 / bw + 0.5);
    double freq = bin * bw;

    double *in = (double *)malloc(total_n * sizeof(double));
    float *out = (float *)malloc(total_n * sizeof(float));
    for (size_t i = 0; i < total_n; i++)
        in[i] = 0.5 * sin(2.0 * M_PI * freq * i / dsd_rate);

    precorr_context_t ctx;
    precorr_context_init(&ctx, f);
    precorr_process_block(&ctx, in, out, total_n);

    size_t discards[] = {0, 256, 1024, 4096, 16384};
    for (int i = 0; i < 5; i++) {
        size_t d = discards[i];
        size_t mn = total_n - d;
        double mbw = (double)dsd_rate / (double)mn;
        unsigned mbin = (unsigned)(freq / mbw + 0.5);
        double mfreq = mbin * mbw;
        double s = calc_sinad(out + d, mn, mfreq, (double)dsd_rate);
        printf("    discard=%zu: SINAD=%.1f dB (%zu samples)\n", d, s, mn);
    }

    free(in); free(out);
    precorr_context_free(&ctx);
    TEST_ASSERT_TRUE(1, "warmup sweep completed");
}

/*
 * Diag 6: PreCorr all rates with warmup discard
 */
static void test_diag_precorr_all_rates_warmup(void) {
    uint32_t rates[] = { DSD_RATE_64, DSD_RATE_128, DSD_RATE_256, DSD_RATE_512 };
    const char *names[] = { "DSD64", "DSD128", "DSD256", "DSD512" };

    printf("\n    --- PreCorr all rates: no warmup vs 4096 discard ---\n");

    for (int r = 0; r < 4; r++) {
        const ntf_filter_t *f = ntf_auto_select(rates[r]);
        if (!f) continue;

        size_t warmup = 4096;
        size_t meas_n = (size_t)(rates[r] / 10);
        size_t total_n = meas_n + warmup;

        double bw = (double)rates[r] / (double)meas_n;
        unsigned bin = (unsigned)(1000.0 / bw + 0.5);
        double freq = bin * bw;

        double *in = (double *)malloc(total_n * sizeof(double));
        float *out = (float *)malloc(total_n * sizeof(float));
        for (size_t i = 0; i < total_n; i++)
            in[i] = 0.5 * sin(2.0 * M_PI * freq * i / rates[r]);

        precorr_context_t ctx;
        precorr_context_init(&ctx, f);
        precorr_process_block(&ctx, in, out, total_n);

        /* No warmup */
        double s0 = calc_sinad(out, meas_n, freq, (double)rates[r]);

        /* With warmup discard */
        double bw2 = (double)rates[r] / (double)meas_n;
        unsigned bin2 = (unsigned)(freq / bw2 + 0.5);
        double freq2 = bin2 * bw2;
        double s1 = calc_sinad(out + warmup, meas_n, freq2, (double)rates[r]);

        printf("    %s: no_warmup=%.1f dB, discard_4096=%.1f dB (delta=%+.1f)\n",
               names[r], s0, s1, s1 - s0);

        free(in); free(out);
        precorr_context_free(&ctx);
    }
    TEST_ASSERT_TRUE(1, "warmup comparison completed");
}

/*
 * Diag 7: PreCorr DSD128 NTF sweep
 */
static void test_diag_precorr_dsd128_ntf_sweep(void) {
    unsigned dsd_rate = DSD_RATE_128;
    ntf_filter_id_t ids[] = {NTF_CLANS_5, NTF_CLANS_6, NTF_CLANS_7};
    const char *names[] = {"clans-5", "clans-6", "clans-7"};

    printf("\n    --- PreCorr DSD128 NTF sweep ---\n");
    for (int i = 0; i < 3; i++) {
        const ntf_filter_t *f = ntf_get_filter(ids[i], dsd_rate);
        if (!f) { printf("    %s: N/A\n", names[i]); continue; }

        precorr_context_t ctx;
        precorr_context_init(&ctx, f);

        size_t n = (size_t)(dsd_rate / 10);
        double bw = (double)dsd_rate / (double)n;
        unsigned bin = (unsigned)(1000.0 / bw + 0.5);
        double freq = bin * bw;

        double *in = (double *)malloc(n * sizeof(double));
        float *out = (float *)malloc(n * sizeof(float));
        for (size_t j = 0; j < n; j++)
            in[j] = 0.5 * sin(2.0 * M_PI * freq * j / dsd_rate);

        precorr_process_block(&ctx, in, out, n);
        double s = calc_sinad(out, n, freq, (double)dsd_rate);
        printf("    %s (order %d): SINAD=%.1f dB\n", names[i], f->order, s);

        free(in); free(out);
        precorr_context_free(&ctx);
    }
    TEST_ASSERT_TRUE(1, "PreCorr NTF sweep completed");
}

void test_sinad_diag_suite(void) {
    TEST_SUITE("SINAD Diagnostics");
    TEST_RUN(test_diag_trellis_dsd128_ntf_sweep);
    TEST_RUN(test_diag_trellis_dsd256_ntf_sweep);
    TEST_RUN(test_diag_trellis_dsd256_latency);
    TEST_RUN(test_diag_trellis_dsd256_cands);
    TEST_RUN(test_diag_precorr_dsd128_warmup);
    TEST_RUN(test_diag_precorr_all_rates_warmup);
    TEST_RUN(test_diag_precorr_dsd128_ntf_sweep);
}
