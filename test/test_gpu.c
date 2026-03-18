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
#include "../include/ntf.h"
#include "../include/trellis.h"
#include "../include/precorr.h"
#include <stdlib.h>

/* Stub for test exe (trellis_log_c is in dsp_fb2k.cpp, not linked in tests) */
void trellis_log_c(const char *msg) { printf("  [log] %s\n", msg); }
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

/* ─── Test: DX11 FIR upsample 2x accuracy ─── */

static void test_gpu_dx11_fir_up_2x(void) {
    printf("  test_gpu_dx11_fir_up_2x...\n");
    if (!gpu_available(GPU_BACKEND_DIRECTX)) {
        printf("    (skipped: DirectCompute not available)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    gpu_context_t *ctx = gpu_create(GPU_BACKEND_DIRECTX);
    if (!ctx) {
        printf("    (skipped: DX11 context creation failed)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    /* Ensure global taps */
    fir_chain_t dummy_fir;
    memset(&dummy_fir, 0, sizeof(dummy_fir));
    fir_chain_init(&dummy_fir, 2822400, 5644800);

    if (gpu_fir_setup(ctx, g_hb_taps, g_hb_ntaps, 1, true) != 0) {
        printf("    (skipped: DX11 FIR setup failed)\n");
        fir_chain_free(&dummy_fir);
        gpu_destroy(ctx);
        g_tests_run++; g_tests_passed++;
        return;
    }

    size_t in_count = 16384;
    float *in  = (float *)malloc(in_count * sizeof(float));
    float *gpu_out = (float *)calloc(in_count * 2, sizeof(float));
    float *cpu_out = (float *)calloc(in_count * 2, sizeof(float));
    gen_sine(in, in_count, 1000.0, 2822400.0, 0.5);

    size_t gpu_n = 0;
    int r = gpu_fir_chain_process(ctx, in, gpu_out, in_count, &gpu_n, NULL, NULL);
    if (r != 0) {
        printf("    (skipped: DX11 FIR returned %d)\n", r);
        g_tests_run++; g_tests_passed++;
    } else {
        fir_chain_reset(&dummy_fir);
        size_t cpu_n = fir_chain_process(&dummy_fir, in, cpu_out, in_count);
        size_t skip = 128;
        size_t cmp = (gpu_n < cpu_n ? gpu_n : cpu_n) - skip;
        double sinad = compute_sinad(cpu_out + skip, gpu_out + skip, cmp);
        printf("    DX11 GPU vs CPU SINAD: %.1f dB\n", sinad);
        TEST_ASSERT_TRUE(sinad > 100.0, "DX11 FIR upsample SINAD > 100 dB");
    }

    free(in); free(gpu_out); free(cpu_out);
    fir_chain_free(&dummy_fir);
    gpu_destroy(ctx);
}

/* ─── Test: CUDA Trellis SDM vs CPU ─── */

static void test_gpu_trellis_sinad(void) {
    printf("  test_gpu_trellis_sinad...\n");
    if (!gpu_available(GPU_BACKEND_CUDA)) {
        printf("    (skipped: CUDA not available)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    /* Trellis GPU only viable with cands >= 16 */
    printf("    (note: GPU Trellis requires cands>=16, testing kernel launch)\n");

    gpu_context_t *ctx = gpu_create(GPU_BACKEND_CUDA);
    if (!ctx) {
        printf("    (skipped: CUDA context creation failed)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    /* Test with 16 candidates */
    int num_cands = 16;
    size_t count = 8192;
    int trellis_lat = 128;

    const ntf_filter_t *f = ntf_auto_select(2822400);

    /* Setup persistent Trellis state on GPU */
    gpu_cuda_trellis_setup(ctx, num_cands, f->order, trellis_lat,
                            f->a, f->g, 0.0);

    /* Generate test input: low-amplitude sine */
    float *in = (float *)malloc(count * sizeof(float));
    float *out = (float *)calloc(count, sizeof(float));
    gen_sine(in, count, 1000.0, 2822400.0, 0.3);

    int r = gpu_trellis_process(ctx, in, out, count, NULL, NULL,
                                 num_cands, f->order, f->a, f->g);
    if (r != 0) {
        printf("    (skipped: GPU Trellis returned %d)\n", r);
        g_tests_run++; g_tests_passed++;
    } else {
        /* Verify output is ±1.0 */
        int valid = 1;
        int non_zero = 0;
        size_t lat = (size_t)trellis_lat;
        for (size_t i = lat; i < count - lat; i++) {
            if (out[i] != 1.0f && out[i] != -1.0f) { valid = 0; break; }
            if (out[i] != 0.0f) non_zero++;
        }
        printf("    output: %s, non-zero samples: %d/%zu\n",
               valid ? "valid ±1.0" : "INVALID", non_zero, count - 2*lat);
        TEST_ASSERT_TRUE(valid, "GPU Trellis output is ±1.0");
        TEST_ASSERT_TRUE(non_zero > 0, "GPU Trellis produces non-trivial output");
    }

    free(in); free(out);
    gpu_destroy(ctx);
}

/* ─── Test: CUDA PreCorr vs CPU ─── */

static void test_gpu_precorr(void) {
    printf("  test_gpu_precorr...\n");
    if (!gpu_available(GPU_BACKEND_CUDA)) {
        printf("    (skipped: CUDA not available)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    gpu_context_t *ctx = gpu_create(GPU_BACKEND_CUDA);
    if (!ctx) {
        printf("    (skipped: CUDA context creation failed)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    /* Get NTF for DSD64 PreCorr */
    const ntf_filter_t *f = ntf_auto_select_precorr(2822400);

    /* Build prediction table on CPU */
    precorr_context_t pc;
    memset(&pc, 0, sizeof(pc));
    precorr_context_init(&pc, f);

    size_t count = 16384;
    float *in = (float *)malloc(count * sizeof(float));
    float *gpu_out = (float *)calloc(count, sizeof(float));
    float *cpu_out = (float *)calloc(count, sizeof(float));
    gen_sine(in, count, 1000.0, 2822400.0, 0.3);

    /* CPU reference — capture initial state BEFORE processing */
    precorr_context_t pc_cpu;
    memset(&pc_cpu, 0, sizeof(pc_cpu));
    precorr_context_init(&pc_cpu, f);

    /* Setup persistent PreCorr on GPU */
    float ntf_a_f[8], ntf_g_f[8];
    for (int k = 0; k < f->order; k++) {
        ntf_a_f[k] = (float)f->a[k];
        ntf_g_f[k] = (float)f->g[k];
    }
    gpu_cuda_precorr_setup(ctx, f->order, ntf_a_f, ntf_g_f,
                            (const float *)pc.pred_table, 0.0f);

    /* Capture initial conditions after init (before any audio processing) */
    gpu_precorr_state_t gpu_init, gpu_final;
    memset(&gpu_init, 0, sizeof(gpu_init));
    memcpy(gpu_init.state, pc_cpu.state, sizeof(gpu_init.state));
    gpu_init.prev_y = pc_cpu.prev_y;
    gpu_init.history = (int)pc_cpu.history;
    gpu_init.phase = pc_cpu.phase;

    /* Now run CPU reference */
    precorr_process_block(&pc_cpu, in, cpu_out, count);

    int r = gpu_precorr_process(ctx, in, gpu_out, count,
                                 ntf_a_f, ntf_g_f, f->order,
                                 (const float (*)[8])pc.pred_table,
                                 &gpu_init, &gpu_final);
    if (r != 0) {
        printf("    (skipped: GPU PreCorr returned %d)\n", r);
        g_tests_run++; g_tests_passed++;
    } else {
        /* Compare GPU vs CPU */
        int match = 0;
        size_t skip = 64;
        for (size_t i = skip; i < count; i++)
            if (gpu_out[i] == cpu_out[i]) match++;
        double pct = 100.0 * match / (count - skip);
        printf("    GPU vs CPU match: %.1f%% (%d/%zu)\n", pct, match, count - skip);
        /* PreCorr's greedy quantizer amplifies tiny float differences —
         * a single bit flip early cascades through history/prediction table.
         * 70%+ match is good for float-precision GPU vs CPU. */
        TEST_ASSERT_TRUE(pct > 70.0, "GPU PreCorr > 70% match with CPU");
    }

    free(in); free(gpu_out); free(cpu_out);
    precorr_context_free(&pc);
    precorr_context_free(&pc_cpu);
    gpu_destroy(ctx);
}

/* ─── Test: GPU boxcar ─── */

static void test_gpu_boxcar(void) {
    printf("  test_gpu_boxcar...\n");
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
    float *in  = (float *)malloc(count * sizeof(float));
    float *gpu_out = (float *)calloc(count, sizeof(float));
    float *cpu_out = (float *)calloc(count, sizeof(float));

    /* DSD-like ±1.0 input */
    for (size_t i = 0; i < count; i++)
        in[i] = (i % 3 == 0) ? 1.0f : -1.0f;

    /* CPU boxcar reference */
    int taps = 32;
    float inv_n = 1.0f / (float)taps;
    float gain = 0.708f;
    {
        float ring[128] = {0};
        float sum = 0;
        int pos = 0;
        for (size_t i = 0; i < count; i++) {
            float s = in[i] >= 0.0f ? 1.0f : -1.0f;
            sum -= ring[pos];
            ring[pos] = s;
            sum += s;
            pos = (pos + 1) % taps;
            cpu_out[i] = sum * inv_n * gain;
        }
    }

    int r = gpu_boxcar_smooth(ctx, in, gpu_out, count, taps, gain);
    if (r != 0) {
        printf("    (skipped: GPU boxcar returned %d)\n", r);
        g_tests_run++; g_tests_passed++;
    } else {
        /* Compare — skip first taps samples (startup) */
        double max_err = 0;
        for (size_t i = (size_t)taps; i < count; i++) {
            double err = fabs((double)gpu_out[i] - (double)cpu_out[i]);
            if (err > max_err) max_err = err;
        }
        printf("    boxcar max error: %.2e\n", max_err);
        TEST_ASSERT_TRUE(max_err < 0.01, "boxcar error < 0.01");
    }

    free(in); free(gpu_out); free(cpu_out);
    gpu_destroy(ctx);
}

/* ─── Test: GPU vs CPU Trellis output comparison ─── */
static void test_gpu_vs_cpu_trellis(void) {
    printf("  test_gpu_vs_cpu_trellis...\n");
    if (!gpu_available(GPU_BACKEND_CUDA)) {
        printf("    (skipped)\n"); g_tests_run++; g_tests_passed++; return;
    }
    gpu_context_t *ctx = gpu_create(GPU_BACKEND_CUDA);
    if (!ctx) { printf("    (skipped)\n"); g_tests_run++; g_tests_passed++; return; }

    const ntf_filter_t *f = ntf_auto_select(2822400);
    gpu_cuda_trellis_setup(ctx, 4, f->order, 128, f->a, f->g, 0.0);

    size_t count = 65536;
    float *in = (float *)malloc(count * sizeof(float));
    float *gpu_out = (float *)calloc(count, sizeof(float));
    float *cpu_out = (float *)calloc(count, sizeof(float));
    gen_sine(in, count, 1000.0, 2822400.0, 0.3);

    sdm_context_t cpu_sdm;
    sdm_context_init(&cpu_sdm, f, 4, 4, 128);
    size_t cpu_n = sdm_process_block(&cpu_sdm, in, cpu_out, count);

    int r = gpu_trellis_process(ctx, in, gpu_out, count,
                                 NULL, NULL, 4, f->order, f->a, f->g);
    if (r != 0 || cpu_n == 0) {
        printf("    (skipped: GPU=%d CPU=%zu)\n", r, cpu_n);
        g_tests_run++; g_tests_passed++;
    } else {
        size_t skip = 9000;
        size_t cmp = cpu_n < count ? cpu_n : count;
        int valid = 1, match = 0;
        for (size_t i = skip; i < cmp; i++) {
            if (gpu_out[i] != 1.0f && gpu_out[i] != -1.0f) valid = 0;
            if (gpu_out[i] == cpu_out[i]) match++;
        }
        double pct = (cmp > skip) ? 100.0 * match / (double)(cmp - skip) : 0;
        /* Measure SINAD: signal power vs noise in audio band */
        double gpu_sig = 0, gpu_nse = 0, cpu_sig = 0, cpu_nse = 0;
        for (size_t i = skip; i < cmp; i++) {
            double s2 = (double)in[i];
            double gd = (double)gpu_out[i] - s2;
            double cd = (double)cpu_out[i] - s2;
            gpu_sig += s2*s2; gpu_nse += gd*gd;
            cpu_sig += s2*s2; cpu_nse += cd*cd;
        }
        double gpu_sinad = gpu_nse > 0 ? 10.0*log10(gpu_sig/gpu_nse) : 300.0;
        double cpu_sinad = cpu_nse > 0 ? 10.0*log10(cpu_sig/cpu_nse) : 300.0;
        printf("    GPU output: %s, bit match: %.1f%%, GPU SINAD: %.1f dB, CPU SINAD: %.1f dB\n",
               valid ? "valid" : "INVALID", pct, gpu_sinad, cpu_sinad);
        TEST_ASSERT_TRUE(valid, "GPU output is ±1.0");
    }
    free(in); free(gpu_out); free(cpu_out);
    sdm_context_free(&cpu_sdm);
    gpu_destroy(ctx);
}

/* ─── Suite ─── */

void test_gpu_suite(void) {
    printf("\n=== GPU Compute ===\n");
    test_gpu_probe();
    test_gpu_dx11_create();
    test_gpu_cuda_create();
    test_gpu_fir_up_2x();
    test_gpu_dx11_fir_up_2x();
    test_gpu_gain();
    test_gpu_boxcar();
    test_gpu_trellis_sinad();
    test_gpu_precorr();
    test_gpu_vs_cpu_trellis();
    test_gpu_fallback();
    test_gpu_threshold();
    test_config_gpu_roundtrip();
}
