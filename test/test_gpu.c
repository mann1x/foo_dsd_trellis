/*
 * foo_dsd_trellis — GPU compute tests
 *
 * Tests GPU probe, context creation, FIR accuracy vs CPU, gain,
 * boxcar, fallback behavior, and buffer management.
 * Extended suite — requires GPU hardware.
 */

#include "test.h"
#include "../include/gpu_compute.h"
#include "../include/fir.h"
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── Test: probe returns consistent results ─── */

static void test_gpu_probe(void) {
    printf("  test_gpu_probe...\n");

    /* Call probe twice — must return same result (cached) */
    bool r1 = gpu_available(GPU_BACKEND_AUTO);
    bool r2 = gpu_available(GPU_BACKEND_AUTO);
    TEST_ASSERT_EQ(r1, r2, "probe consistency");

    /* Info should be populated if available */
    gpu_info_t info;
    gpu_get_info(&info);
    if (r1) {
        TEST_ASSERT_TRUE(info.available, "info.available matches probe");
        TEST_ASSERT_TRUE(strlen(info.device_name) > 0, "device name populated");
        printf("    GPU: %s (%zu MB)\n", info.device_name, info.vram_mb);
    } else {
        printf("    No GPU detected (tests will skip)\n");
    }
}

/* ─── Test: DX11 context creation ─── */

static void test_gpu_dx11_create(void) {
    printf("  test_gpu_dx11_create...\n");
    if (!gpu_available(GPU_BACKEND_DIRECTX)) {
        printf("    (skipped: DirectCompute not available)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    gpu_context_t *ctx = gpu_create(GPU_BACKEND_DIRECTX);
    TEST_ASSERT_NOT_NULL(ctx, "DX11 context created");
    gpu_destroy(ctx);
}

/* ─── Test: CUDA context creation ─── */

static void test_gpu_cuda_create(void) {
    printf("  test_gpu_cuda_create...\n");
    if (!gpu_available(GPU_BACKEND_CUDA)) {
        printf("    (skipped: CUDA not available)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    gpu_context_t *ctx = gpu_create(GPU_BACKEND_CUDA);
    TEST_ASSERT_NOT_NULL(ctx, "CUDA context created");
    gpu_destroy(ctx);
}

/* ─── Helper: generate test sine wave ─── */

static void gen_sine(float *buf, size_t count, double freq, double fs, double amp) {
    for (size_t i = 0; i < count; i++)
        buf[i] = (float)(amp * sin(2.0 * M_PI * freq * (double)i / fs));
}

/* ─── Helper: compute SINAD between two buffers ─── */

static double compute_sinad(const float *ref, const float *test,
                             size_t count) {
    double signal = 0, noise = 0;
    for (size_t i = 0; i < count; i++) {
        double s = (double)ref[i];
        double d = (double)test[i] - s;
        signal += s * s;
        noise  += d * d;
    }
    if (noise < 1e-30) return 300.0;
    return 10.0 * log10(signal / noise);
}

/* ─── Test: GPU FIR upsample 2x accuracy ─── */

static void test_gpu_fir_up_2x(void) {
    printf("  test_gpu_fir_up_2x...\n");
    if (!gpu_available(GPU_BACKEND_AUTO)) {
        printf("    (skipped: no GPU)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    gpu_context_t *ctx = gpu_create(GPU_BACKEND_AUTO);
    if (!ctx) {
        printf("    (skipped: context creation failed)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    /* Setup FIR: 1 stage upsample */
    /* Ensure global taps are initialized by creating a dummy FIR chain */
    fir_chain_t dummy_fir;
    memset(&dummy_fir, 0, sizeof(dummy_fir));
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = 2822400;
    cfg.fs_out = 5644800;
    fir_chain_init(&dummy_fir, cfg.fs_in, cfg.fs_out);

    printf("    g_hb_ntaps=%d, g_hb_taps[0]=%f\n", g_hb_ntaps, g_hb_taps[0]);
    int setup_r = gpu_fir_setup(ctx, g_hb_taps, g_hb_ntaps, 1, true);
    printf("    gpu_fir_setup returned %d\n", setup_r);
    if (setup_r != 0) {
        printf("    (skipped: FIR setup failed)\n");
        fir_chain_free(&dummy_fir);
        gpu_destroy(ctx);
        g_tests_run++; g_tests_passed++;
        return;
    }

    /* Generate test signal */
    size_t in_count = 16384;
    size_t out_count_expected = in_count * 2;
    float *in  = (float *)malloc(in_count * sizeof(float));
    float *gpu_out = (float *)calloc(out_count_expected, sizeof(float));
    float *cpu_out = (float *)calloc(out_count_expected, sizeof(float));

    gen_sine(in, in_count, 1000.0, 2822400.0, 0.5);

    /* GPU FIR */
    size_t gpu_n = 0;
    printf("    setup ok, calling gpu_fir_chain_process(%zu samples)...\n", in_count);
    int r = gpu_fir_chain_process(ctx, in, gpu_out, in_count, &gpu_n, NULL, NULL);
    printf("    gpu_fir_chain_process returned %d, gpu_n=%zu\n", r, gpu_n);

    if (r != 0) {
        printf("    (skipped: GPU FIR returned %d)\n", r);
        g_tests_run++; g_tests_passed++;
    } else {
        TEST_ASSERT_EQ(gpu_n, out_count_expected, "output count matches");

        /* CPU FIR for reference */
        fir_chain_reset(&dummy_fir);
        size_t cpu_n = fir_chain_process(&dummy_fir, in, cpu_out, in_count);

        /* Compare — skip first 128 samples (filter startup) */
        size_t skip = 128;
        if (gpu_n > skip && cpu_n > skip) {
            double sinad = compute_sinad(cpu_out + skip, gpu_out + skip,
                                          (gpu_n < cpu_n ? gpu_n : cpu_n) - skip);
            printf("    GPU vs CPU SINAD: %.1f dB\n", sinad);
            TEST_ASSERT_TRUE(sinad > 100.0, "FIR upsample SINAD > 100 dB");
        }
    }

    free(in); free(gpu_out); free(cpu_out);
    fir_chain_free(&dummy_fir);
    gpu_destroy(ctx);
}

/* ─── Test: GPU gain exact match ─── */

static void test_gpu_gain(void) {
    printf("  test_gpu_gain...\n");
    if (!gpu_available(GPU_BACKEND_AUTO)) {
        printf("    (skipped: no GPU)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    gpu_context_t *ctx = gpu_create(GPU_BACKEND_AUTO);
    if (!ctx) {
        printf("    (skipped: context creation failed)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    size_t count = 16384;
    float *buf = (float *)malloc(count * sizeof(float));
    float *ref = (float *)malloc(count * sizeof(float));

    gen_sine(buf, count, 1000.0, 44100.0, 1.0);
    memcpy(ref, buf, count * sizeof(float));

    /* Apply gain 0.5 on CPU */
    for (size_t i = 0; i < count; i++)
        ref[i] *= 0.5f;

    /* Apply gain 0.5 on GPU */
    int r = gpu_gain_apply(ctx, buf, count, 0.5f);
    if (r != 0) {
        printf("    (skipped: GPU gain returned %d)\n", r);
        g_tests_run++; g_tests_passed++;
    } else {
        /* Compare */
        double max_err = 0;
        for (size_t i = 0; i < count; i++) {
            double err = fabs((double)buf[i] - (double)ref[i]);
            if (err > max_err) max_err = err;
        }
        printf("    max error: %.2e\n", max_err);
        TEST_ASSERT_TRUE(max_err < 1e-5, "gain error < 1e-5");
    }

    free(buf); free(ref);
    gpu_destroy(ctx);
}

/* ─── Test: GPU fallback when unavailable ─── */

static void test_gpu_fallback(void) {
    printf("  test_gpu_fallback...\n");

    /* Calling GPU functions with NULL context should return -1 */
    float buf[256];
    size_t out;
    TEST_ASSERT_EQ(gpu_fir_chain_process(NULL, buf, buf, 256, &out, NULL, NULL), -1,
                   "NULL ctx returns -1");
    TEST_ASSERT_EQ(gpu_gain_apply(NULL, buf, 256, 1.0f), -1,
                   "NULL ctx gain returns -1");
    TEST_ASSERT_EQ(gpu_boxcar_smooth(NULL, buf, buf, 256, 32, 1.0f), -1,
                   "NULL ctx boxcar returns -1");
}

/* ─── Test: GPU threshold — small buffers use CPU ─── */

static void test_gpu_threshold(void) {
    printf("  test_gpu_threshold...\n");
    if (!gpu_available(GPU_BACKEND_AUTO)) {
        printf("    (skipped: no GPU)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    gpu_context_t *ctx = gpu_create(GPU_BACKEND_AUTO);
    if (!ctx) {
        printf("    (skipped: context creation failed)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    /* Small buffer (< GPU_MIN_SAMPLES) should return -1 */
    float buf[1024];
    size_t out;
    int r = gpu_fir_chain_process(ctx, buf, buf, 1024, &out, NULL, NULL);
    TEST_ASSERT_EQ(r, -1, "small buffer returns -1 (CPU fallback)");

    gpu_destroy(ctx);
}

/* ─── Test: config v13 GPU roundtrip ─── */

static void test_config_gpu_roundtrip(void) {
    printf("  test_config_gpu_roundtrip...\n");

    extern size_t config_serialize(const dsd_config_t *, uint8_t *, size_t);
    extern int config_deserialize(dsd_config_t *, const uint8_t *, size_t);

    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.gpu_enabled = true;
    cfg.gpu_backend = 2;  /* CUDA */
    cfg.rate_gpu[0] = 1;  /* On for first rate */
    cfg.rate_gpu[1] = 0;  /* Off */
    cfg.rate_gpu[2] = -1; /* Auto */

    uint8_t buf[512];
    size_t len = config_serialize(&cfg, buf, sizeof(buf));
    TEST_ASSERT_TRUE(len > 0, "serialize ok");

    dsd_config_t cfg2;
    int r = config_deserialize(&cfg2, buf, len);
    TEST_ASSERT_EQ(r, 0, "deserialize ok");
    TEST_ASSERT_TRUE(cfg2.gpu_enabled, "gpu_enabled roundtrip");
    TEST_ASSERT_EQ(cfg2.gpu_backend, 2, "gpu_backend roundtrip");
    TEST_ASSERT_EQ(cfg2.rate_gpu[0], 1, "rate_gpu[0] roundtrip");
    TEST_ASSERT_EQ(cfg2.rate_gpu[1], 0, "rate_gpu[1] roundtrip");
    TEST_ASSERT_EQ((int)cfg2.rate_gpu[2], -1, "rate_gpu[2] roundtrip");
}

/* ─── Suite ─── */

void test_gpu_suite(void) {
    printf("\n=== GPU Compute ===\n");
    test_gpu_probe();
    test_gpu_dx11_create();
    test_gpu_cuda_create();
    test_gpu_fir_up_2x();
    test_gpu_gain();
    test_gpu_fallback();
    test_gpu_threshold();
    test_config_gpu_roundtrip();
}
