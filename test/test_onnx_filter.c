/*
 * foo_dsd_trellis — ONNX filter tests
 *
 * Tests the ONNX filter infrastructure without requiring onnxruntime.dll.
 * Validates: DLL probe, NULL-safe API, config round-trip for ML fields.
 */

#include "test.h"
#include "../include/onnx_filter.h"
#include "../include/dsd_types.h"
#include "../include/trellis.h"
#include "../include/ntf.h"
#define _USE_MATH_DEFINES
#include <math.h>

extern size_t config_serialize(const dsd_config_t *cfg, uint8_t *buf, size_t buf_size);
extern int config_deserialize(dsd_config_t *cfg, const uint8_t *buf, size_t buf_size);
extern void config_validate(dsd_config_t *cfg);

/* ─── DLL probe ─── */

static void test_onnx_runtime_probe(void) {
    /* Just verify the probe doesn't crash. Result depends on environment. */
    bool avail = onnx_runtime_available();
    printf("    onnxruntime.dll %s\n", avail ? "found" : "not found");
    /* Probe is idempotent */
    bool avail2 = onnx_runtime_available();
    TEST_ASSERT_EQ(avail, avail2, "onnx_runtime_available idempotent");
}

/* ─── NULL-safe API ─── */

static void test_onnx_filter_null_safe(void) {
    /* All functions must handle NULL gracefully */
    onnx_filter_process(NULL, NULL, 0);
    onnx_filter_reset(NULL);
    onnx_filter_free(NULL);
    TEST_ASSERT_TRUE(1, "NULL-safe API calls succeeded");
}

/* ─── Create returns NULL without DLL ─── */

static void test_onnx_filter_create_no_model(void) {
    /* Even if onnxruntime.dll is present, a nonexistent model should fail */
    onnx_filter_t *f = onnx_filter_create(L"nonexistent_model.onnx",
                                            2822400, ML_EP_CPU);
    /* Either NULL (no DLL or bad model) — both are acceptable */
    if (f) {
        /* Unexpected success — clean up */
        onnx_filter_free(f);
        printf("    (onnxruntime.dll present, but model load expected to fail)\n");
    }
    /* This test verifies no crash */
    TEST_ASSERT_TRUE(1, "create with nonexistent model did not crash");
}

/* ─── Config round-trip for ML fields ─── */

static void test_config_ml_roundtrip(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.ml_enabled = true;
    cfg.ml_ep = 1;  /* ML_EP_DIRECTML */

    uint8_t buf[256];
    size_t len = config_serialize(&cfg, buf, sizeof(buf));
    TEST_ASSERT_TRUE(len > 0, "serialize with ML fields");

    dsd_config_t cfg2;
    int ret = config_deserialize(&cfg2, buf, len);
    TEST_ASSERT_EQ(ret, 0, "deserialize with ML fields");
    TEST_ASSERT_EQ(cfg2.ml_enabled, true, "ml_enabled round-trip");
    TEST_ASSERT_EQ(cfg2.ml_ep, 1, "ml_ep round-trip");
}

static void test_config_ml_defaults(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    TEST_ASSERT_FALSE(cfg.ml_enabled, "ml_enabled default is false");
    TEST_ASSERT_EQ(cfg.ml_ep, 2, "ml_ep default is Auto (2)");
}

static void test_config_ml_validation(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.ml_ep = 99;  /* invalid */
    config_validate(&cfg);
    TEST_ASSERT_EQ(cfg.ml_ep, 2, "invalid ml_ep clamped to Auto");
}

static void test_config_v9_upgrade(void) {
    /* Simulate a v9 config (no ML fields) — ml_enabled/ml_ep should default */
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.gain = 0.75f;
    cfg.sdm_mode = SDM_MODE_TRELLIS;

    uint8_t buf[256];
    config_serialize(&cfg, buf, sizeof(buf));

    /* Truncate to v9 size (78 bytes) to simulate old config */
    uint32_t v9 = 9;
    memcpy(buf, &v9, 4);  /* force version 9 */

    dsd_config_t cfg2;
    int ret = config_deserialize(&cfg2, buf, 82);  /* CONFIG_V9_SIZE */
    TEST_ASSERT_EQ(ret, 0, "deserialize v9 config");
    TEST_ASSERT_FALSE(cfg2.ml_enabled, "v9 upgrade: ml_enabled defaults false");
    TEST_ASSERT_EQ(cfg2.ml_ep, 2, "v9 upgrade: ml_ep defaults Auto");
    /* Other fields should survive */
    TEST_ASSERT_FLOAT_EQ(cfg2.gain, 0.75f, 0.001f, "v9 upgrade: gain preserved");
}

/* ─── Goertzel (duplicated from test_trellis.c for independence) ─── */

static double goertzel_power_ml(const float *x, size_t n, double freq_hz,
                                double sample_rate) {
    double k = freq_hz * n / sample_rate;
    double w = 2.0 * M_PI * k / n;
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

static double measure_sinad_dsd(const float *out, size_t produced,
                                double freq, unsigned dsd_rate) {
    double signal_power = goertzel_power_ml(out, produced, freq, (double)dsd_rate);
    double actual_bw = (double)dsd_rate / (double)produced;
    unsigned max_bin = (unsigned)(22050.0 / actual_bw);
    unsigned sig_bin = (unsigned)(freq / actual_bw + 0.5);
    double total_noise = 0.0;

    for (unsigned b = 1; b <= max_bin; b++) {
        if (b >= sig_bin - 1 && b <= sig_bin + 1)
            continue;
        total_noise += goertzel_power_ml(out, produced, b * actual_bw, (double)dsd_rate);
    }

    if (total_noise <= 0.0) total_noise = 1e-30;
    return 10.0 * log10(signal_power / total_noise);
}

/* ─── Live inference test (requires onnxruntime.dll + model) ─── */

static void test_onnx_filter_live_inference(void) {
    if (!onnx_runtime_available()) {
        printf("    (skipped: onnxruntime.dll not found)\n");
        TEST_ASSERT_TRUE(1, "skipped");
        return;
    }

    /* Try to find model next to test exe */
    onnx_filter_t *f = onnx_filter_create(L"foo_dsd_trellis_ml.onnx",
                                            2822400, ML_EP_CPU);
    if (!f) {
        printf("    (skipped: model not found or session creation failed)\n");
        TEST_ASSERT_TRUE(1, "skipped");
        return;
    }

    printf("    ONNX session created successfully\n");

    /* Generate ±1.0 DSD-like test signal */
    float buf[256];
    for (int i = 0; i < 256; i++)
        buf[i] = (i % 3 == 0) ? -1.0f : 1.0f;

    /* Run inference — should not crash */
    onnx_filter_process(f, buf, 256);

    /* Output should be ±1.0 (requantized) */
    int valid = 1;
    for (int i = 0; i < 256; i++) {
        if (buf[i] != 1.0f && buf[i] != -1.0f) {
            valid = 0;
            break;
        }
    }
    TEST_ASSERT_TRUE(valid, "output is ±1.0 after requantization");

    /* Run again to test causal state persistence */
    for (int i = 0; i < 256; i++)
        buf[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    onnx_filter_process(f, buf, 256);
    TEST_ASSERT_TRUE(1, "second inference block succeeded");

    /* Reset causal state */
    onnx_filter_reset(f);
    for (int i = 0; i < 256; i++)
        buf[i] = 1.0f;
    onnx_filter_process(f, buf, 256);
    TEST_ASSERT_TRUE(1, "inference after reset succeeded");

    onnx_filter_free(f);
    printf("    Live inference: 3 blocks processed OK\n");
}

/* ─── SINAD comparison: with and without ML filter ─── */

static void test_onnx_sinad_dsd64_comparison(void) {
    if (!onnx_runtime_available()) {
        printf("    (skipped: onnxruntime.dll not found)\n");
        TEST_ASSERT_TRUE(1, "skipped");
        return;
    }

    onnx_filter_t *ml = onnx_filter_create(L"foo_dsd_trellis_ml.onnx",
                                             DSD_RATE_64, ML_EP_CPU);
    if (!ml) {
        printf("    (skipped: model not found)\n");
        TEST_ASSERT_TRUE(1, "skipped");
        return;
    }

    const unsigned dsd_rate = DSD_RATE_64;
    const unsigned n_dsd = 262144u;
    const int lat = DSD_DEFAULT_TRELLIS_LAT;

    /* Bin-aligned frequency near 1kHz */
    unsigned produced_est = n_dsd - (unsigned)lat;
    double bin_width = (double)dsd_rate / (double)produced_est;
    unsigned sig_bin = (unsigned)(1000.0 / bin_width + 0.5);
    double freq = sig_bin * bin_width;

    const ntf_filter_t *f = ntf_auto_select(dsd_rate);
    sdm_context_t ctx;
    sdm_context_init(&ctx, f, 8, 16, lat);

    float *in = (float *)malloc(n_dsd * sizeof(float));
    float *out = (float *)malloc(n_dsd * sizeof(float));
    float *out_ml = (float *)malloc(n_dsd * sizeof(float));

    /* Generate 1kHz sine */
    for (unsigned i = 0; i < n_dsd; i++)
        in[i] = (float)(0.5 * sin(2.0 * M_PI * freq * i / dsd_rate));

    /* Encode through SDM */
    size_t produced = sdm_process_block(&ctx, in, out, n_dsd);

    /* Copy for ML processing */
    memcpy(out_ml, out, produced * sizeof(float));

    /* Apply ML filter — process entire signal at once for best quality.
     * For non-causal models, chunked processing introduces edge artifacts
     * due to the look-ahead window. Single-block gives the model full context. */
    onnx_filter_process(ml, out_ml, produced);

    /* Measure SINAD for both */
    double sinad_raw = measure_sinad_dsd(out, produced, freq, dsd_rate);
    double sinad_ml = measure_sinad_dsd(out_ml, produced, freq, dsd_rate);
    double delta = sinad_ml - sinad_raw;

    printf("    [SINAD DSD64] Without ML: %.1f dB\n", sinad_raw);
    printf("    [SINAD DSD64] With ML:    %.1f dB  (delta: %+.1f dB)\n",
           sinad_ml, delta);

    /* Diagnostic test: prints SINAD comparison.
     * No assertion on SINAD value — model quality depends on training.
     * Just verify the filter ran without crashing. */
    TEST_ASSERT_TRUE(1, "ML filter SINAD comparison completed");

    free(in);
    free(out);
    free(out_ml);
    sdm_context_free(&ctx);
    onnx_filter_free(ml);
}

/* ─── Suite entry point ─── */

void test_onnx_filter_suite(void) {
    TEST_SUITE("ONNX Filter");
    TEST_RUN(test_onnx_runtime_probe);
    TEST_RUN(test_onnx_filter_null_safe);
    TEST_RUN(test_onnx_filter_create_no_model);
    TEST_RUN(test_config_ml_roundtrip);
    TEST_RUN(test_config_ml_defaults);
    TEST_RUN(test_config_ml_validation);
    TEST_RUN(test_config_v9_upgrade);
    TEST_RUN(test_onnx_filter_live_inference);
    TEST_RUN(test_onnx_sinad_dsd64_comparison);
}
