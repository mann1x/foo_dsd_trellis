/*
 * foo_dsd_trellis — PreCorr SDM unit tests
 */

#include "test.h"
#include "../include/precorr.h"
#include "../include/ntf.h"
#include "../include/dsd_types.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void test_precorr_init(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_64);
    TEST_ASSERT_NOT_NULL(f, "auto-select filter should not be null");

    precorr_context_t ctx;
    int ret = precorr_context_init(&ctx, f);
    TEST_ASSERT_EQ(ret, 0, "precorr_context_init should succeed");
    TEST_ASSERT_EQ(ctx.order, f->order, "order should match filter");
    TEST_ASSERT_EQ(ctx.history, 0x69, "history should start at DSD silence");
    TEST_ASSERT_EQ(ctx.phase, 0, "phase should start at 0");

    /* Null filter should fail */
    ret = precorr_context_init(&ctx, NULL);
    TEST_ASSERT_EQ(ret, -1, "null filter should fail");

    precorr_context_free(&ctx);
}

static void test_precorr_output_binary(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_128);
    precorr_context_t ctx;
    precorr_context_init(&ctx, f);

    /* Feed a mix of signals */
    float in[1024];
    float out[1024];
    for (int i = 0; i < 1024; i++)
        in[i] = 0.3f * sinf((float)(2.0 * M_PI * 1000.0 * i / (5644800.0)));

    size_t n = precorr_process_block(&ctx, in, out, 1024);
    TEST_ASSERT_EQ(n, 1024u, "output count should equal input count");

    int all_binary = 1;
    for (size_t i = 0; i < n; i++) {
        if (out[i] != 1.0f && out[i] != -1.0f) {
            all_binary = 0;
            break;
        }
    }
    TEST_ASSERT_TRUE(all_binary, "all outputs must be +1.0 or -1.0");

    precorr_context_free(&ctx);
}

static void test_precorr_no_latency(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_256);
    precorr_context_t ctx;
    precorr_context_init(&ctx, f);

    /* Various block sizes */
    float in[512], out[512];
    memset(in, 0, sizeof(in));

    size_t sizes[] = {1, 7, 32, 128, 512};
    for (int t = 0; t < 5; t++) {
        size_t n = precorr_process_block(&ctx, in, out, sizes[t]);
        TEST_ASSERT_EQ(n, sizes[t], "output count must equal input count (no latency)");
    }

    /* Drain should return 0 */
    size_t drained = precorr_drain(&ctx, out, 512);
    TEST_ASSERT_EQ(drained, 0u, "drain should return 0 (no latency)");

    precorr_context_free(&ctx);
}

static void test_precorr_reset(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_64);
    precorr_context_t ctx;
    precorr_context_init(&ctx, f);

    /* Process some data to dirty state */
    float in[256], out[256];
    for (int i = 0; i < 256; i++)
        in[i] = 0.1f * sinf((float)(2.0 * M_PI * 440.0 * i / 2822400.0));
    precorr_process_block(&ctx, in, out, 256);

    /* Save prediction table before reset */
    float saved_table[PRECORR_HIST_SIZE][PRECORR_PHASES];
    memcpy(saved_table, ctx.pred_table, sizeof(saved_table));

    precorr_context_reset(&ctx);

    /* State should be zeroed */
    TEST_ASSERT_EQ(ctx.history, 0x69, "history should reset to DSD silence");
    TEST_ASSERT_EQ(ctx.phase, 0, "phase should reset to 0");

    /* Prediction table should be preserved */
    int table_preserved = (memcmp(saved_table, ctx.pred_table, sizeof(saved_table)) == 0);
    TEST_ASSERT_TRUE(table_preserved, "prediction table should survive reset");

    /* NTF state should be zeroed */
    int state_zero = 1;
    for (int i = 0; i < ctx.order; i++) {
        if (ctx.state[i] != 0.0f) {
            state_zero = 0;
            break;
        }
    }
    TEST_ASSERT_TRUE(state_zero, "NTF state should be zeroed after reset");

    precorr_context_free(&ctx);
}

static void test_precorr_dc_stability(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_128);
    precorr_context_t ctx;
    precorr_context_init(&ctx, f);

    /* DC=0 input should produce roughly balanced +1/-1 output */
    float in[4096], out[4096];
    memset(in, 0, sizeof(in));

    precorr_process_block(&ctx, in, out, 4096);

    int ones = 0;
    for (int i = 0; i < 4096; i++)
        if (out[i] > 0.0f) ones++;

    float ratio = (float)ones / 4096.0f;
    /* Should be roughly 50% ± 10% */
    TEST_ASSERT_TRUE(ratio > 0.40f && ratio < 0.60f,
                     "DC=0 should produce ~50% +1 outputs");
    printf("    DC balance: %.1f%% ones\n", ratio * 100.0f);

    precorr_context_free(&ctx);
}

static void test_precorr_pred_table_nonzero(void) {
    /* After training, at least some entries should be non-zero */
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_64);
    precorr_context_t ctx;
    precorr_context_init(&ctx, f);

    int nonzero = 0;
    for (int h = 0; h < PRECORR_HIST_SIZE; h++)
        for (int p = 0; p < PRECORR_PHASES; p++)
            if (ctx.pred_table[h][p] != 0.0f)
                nonzero++;

    TEST_ASSERT_TRUE(nonzero > 0, "prediction table should have non-zero entries");
    printf("    Non-zero table entries: %d / %d\n",
           nonzero, PRECORR_HIST_SIZE * PRECORR_PHASES);

    precorr_context_free(&ctx);
}

/* ─── Goertzel single-bin power measurement (same as test_rate_sinad.c) ─── */

static double goertzel_power_pc(const float *x, size_t n, double freq_hz,
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

static double measure_sinad_pc(const float *x, size_t n, double freq_hz,
                                double sample_rate) {
    double signal_power = goertzel_power_pc(x, n, freq_hz, sample_rate);

    double bw = sample_rate / (double)n;
    unsigned max_bin = (unsigned)(22050.0 / bw);
    unsigned sig_bin = (unsigned)(freq_hz / bw + 0.5);

    double noise = 0.0;
    for (unsigned b = 1; b <= max_bin; b++) {
        if (b >= sig_bin - 1 && b <= sig_bin + 1) continue;
        noise += goertzel_power_pc(x, n, b * bw, sample_rate);
    }
    if (noise <= 0.0) noise = 1e-30;
    return 10.0 * log10(signal_power / noise);
}

/* SINAD measurement for PreCorr at each DSD rate */
static void test_precorr_sinad(void) {
    uint32_t rates[] = { DSD_RATE_64, DSD_RATE_128, DSD_RATE_256, DSD_RATE_512 };
    const char *names[] = { "DSD64", "DSD128", "DSD256", "DSD512" };

    for (int r = 0; r < 4; r++) {
        const ntf_filter_t *f = ntf_auto_select_precorr(rates[r]);
        TEST_ASSERT_NOT_NULL(f, "filter should exist for rate");
        if (!f) continue;

        precorr_context_t ctx;
        precorr_context_init(&ctx, f);

        /* Generate test tone — bin-align frequency to avoid spectral leakage */
        size_t n = (size_t)(rates[r] / 10);  /* 0.1 seconds */
        double bw = (double)rates[r] / (double)n;
        unsigned bin = (unsigned)(1000.0 / bw + 0.5);
        double freq = bin * bw;
        double amp = 0.5;

        float *in  = (float *)malloc(n * sizeof(float));
        float *out = (float *)malloc(n * sizeof(float));
        if (!in || !out) { free(in); free(out); continue; }

        for (size_t i = 0; i < n; i++)
            in[i] = (float)(amp * sin(2.0 * M_PI * freq * i / rates[r]));

        precorr_process_block(&ctx, in, out, n);

        /* Measure SINAD using Goertzel (frequency-domain, correct for 1-bit PDM) */
        double sinad_db = measure_sinad_pc(out, n, freq, (double)rates[r]);
        printf("    [SINAD] %s: %zu samples, SINAD=%.1f dB (order %d, %s)\n",
               names[r], n, sinad_db, f->order, f->name);

        /* PreCorr SINAD should be >20 dB (lower than trellis which gets >60 dB) */
        TEST_ASSERT_TRUE(sinad_db > 20.0, "PreCorr SINAD should be >20 dB");

        free(in);
        free(out);
        precorr_context_free(&ctx);
    }
}

/* All DSD rates should initialize and process successfully */
static void test_precorr_all_rates(void) {
    uint32_t rates[] = { DSD_RATE_64, DSD_RATE_128, DSD_RATE_256, DSD_RATE_512 };
    for (int r = 0; r < 4; r++) {
        const ntf_filter_t *f = ntf_auto_select(rates[r]);
        precorr_context_t ctx;
        int ret = precorr_context_init(&ctx, f);
        TEST_ASSERT_EQ(ret, 0, "init should succeed for all DSD rates");

        float in[64], out[64];
        memset(in, 0, sizeof(in));
        size_t n = precorr_process_block(&ctx, in, out, 64);
        TEST_ASSERT_EQ(n, 64u, "should process 64 samples at each rate");

        precorr_context_free(&ctx);
    }
}

void test_precorr_suite(void) {
    TEST_SUITE("PreCorr SDM");
    TEST_RUN(test_precorr_init);
    TEST_RUN(test_precorr_output_binary);
    TEST_RUN(test_precorr_no_latency);
    TEST_RUN(test_precorr_reset);
    TEST_RUN(test_precorr_dc_stability);
    TEST_RUN(test_precorr_pred_table_nonzero);
    TEST_RUN(test_precorr_all_rates);
    TEST_RUN(test_precorr_sinad);
}
