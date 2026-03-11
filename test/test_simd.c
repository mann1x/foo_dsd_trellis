/*
 * foo_dsd_trellis — SIMD detection and FIR SIMD correctness tests
 */

#include "test.h"
#include "../include/simd_detect.h"
#include "../include/fir.h"
#include "../include/engine.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* External SIMD kernel functions for direct testing */
extern double fir_convolve_scalar(const double *coeffs, const float *delay,
                                  int pos, int ntaps);
extern double fir_convolve_sse2(const double *coeffs, const float *delay,
                                int pos, int ntaps);
extern double fir_convolve_avx2(const double *coeffs, const float *delay,
                                int pos, int ntaps);
extern double fir_convolve_avx128(const double *coeffs, const float *delay,
                                  int pos, int ntaps);

static void test_cpu_detect_sane(void) {
    const cpu_features_t *cpu = cpu_detect();
    TEST_ASSERT_NOT_NULL(cpu, "cpu_detect should return non-null");

    /* x64 always has SSE2 */
    TEST_ASSERT_TRUE(cpu->sse2, "x64 must have SSE2");

    /* Vendor should be detected */
    TEST_ASSERT_TRUE(cpu->vendor == CPU_VENDOR_INTEL ||
                     cpu->vendor == CPU_VENDOR_AMD ||
                     cpu->vendor == CPU_VENDOR_UNKNOWN,
                     "vendor should be valid enum");

    printf("    CPU: %s, SSE2=%d, AVX2=%d, FMA3=%d, Family=0x%X, Model=0x%X\n",
           cpu->vendor == CPU_VENDOR_INTEL ? "Intel" :
           cpu->vendor == CPU_VENDOR_AMD   ? "AMD" : "Unknown",
           cpu->sse2, cpu->avx2, cpu->fma3, cpu->family, cpu->model);
}

static void test_simd_dispatch_name(void) {
    const char *name = fir_simd_name();
    TEST_ASSERT_NOT_NULL(name, "SIMD name should not be null");
    printf("    FIR SIMD kernel: %s\n", name);

    /* Name should be one of the known values */
    int valid = (strcmp(name, "AVX2+FMA") == 0 ||
                 strcmp(name, "AVX128+FMA (Zen)") == 0 ||
                 strcmp(name, "SSE2") == 0 ||
                 strcmp(name, "Scalar") == 0);
    TEST_ASSERT_TRUE(valid, "SIMD name should be a known variant");
}

static void test_simd_sse2_matches_scalar(void) {
    /* Set up a small FIR test case */
    double coeffs[12];
    float delay[64];
    memset(delay, 0, sizeof(delay));

    for (int i = 0; i < 12; i++)
        coeffs[i] = 1.0 / (i + 1);
    for (int i = 0; i < 12; i++)
        delay[i] = (float)(i + 1);

    double scalar = fir_convolve_scalar(coeffs, delay, 11, 12);
    double sse2   = fir_convolve_sse2(coeffs, delay, 11, 12);

    TEST_ASSERT_FLOAT_EQ(scalar, sse2, 1e-10, "SSE2 should match scalar");
}

static void test_simd_avx2_matches_scalar(void) {
    const cpu_features_t *cpu = cpu_detect();
    if (!cpu->avx2) {
        printf("    [SKIP] AVX2 not available\n");
        /* Still count as passed */
        TEST_ASSERT_TRUE(1, "AVX2 skipped (not available)");
        return;
    }

    double coeffs[12];
    float delay[64];
    memset(delay, 0, sizeof(delay));

    for (int i = 0; i < 12; i++)
        coeffs[i] = 1.0 / (i + 1);
    for (int i = 0; i < 12; i++)
        delay[i] = (float)(i + 1);

    double scalar = fir_convolve_scalar(coeffs, delay, 11, 12);
    double avx2   = fir_convolve_avx2(coeffs, delay, 11, 12);
    double avx128 = fir_convolve_avx128(coeffs, delay, 11, 12);

    TEST_ASSERT_FLOAT_EQ(scalar, avx2, 1e-10, "AVX2 should match scalar");
    TEST_ASSERT_FLOAT_EQ(scalar, avx128, 1e-10, "AVX128 should match scalar");
}

static void test_simd_fir_chain_correctness(void) {
    /* Process through FIR chain with SIMD — verify output is valid DSD */
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_128;
    cfg.gain = 0.5f;

    engine_channel_t eng;
    int ret = engine_channel_init(&eng, 0, &cfg);
    TEST_ASSERT_EQ(ret, 0, "engine init for SIMD test");

    unsigned n = 1024;
    float *in  = (float *)malloc(n * sizeof(float));
    float *out = (float *)malloc(n * 4 * sizeof(float));

    for (unsigned i = 0; i < n; i++)
        in[i] = (float)(0.3 * sin(2.0 * M_PI * 1000.0 * i / DSD_RATE_64));

    size_t n_out = engine_process_block(&eng, in, out, n, &cfg);
    TEST_ASSERT_TRUE(n_out > 0, "SIMD FIR chain should produce output");

    /* All output must be +/-1.0 */
    int valid = 1;
    for (size_t i = 0; i < n_out; i++) {
        if (out[i] != 1.0f && out[i] != -1.0f) { valid = 0; break; }
    }
    TEST_ASSERT_TRUE(valid, "SIMD FIR chain output should be +/-1.0");

    free(in);
    free(out);
    engine_channel_free(&eng);
}

static void test_simd_circular_wrap(void) {
    /* Test with delay_pos near the wrap boundary */
    double coeffs[12];
    float delay[64];
    memset(delay, 0, sizeof(delay));

    for (int i = 0; i < 12; i++) {
        coeffs[i] = 0.1 * (i + 1);
        delay[(63 - i) & 63] = (float)(i * 0.5);
    }

    /* Position near end of circular buffer */
    double scalar = fir_convolve_scalar(coeffs, delay, 63, 12);
    double sse2   = fir_convolve_sse2(coeffs, delay, 63, 12);

    TEST_ASSERT_FLOAT_EQ(scalar, sse2, 1e-10,
                         "SSE2 should match scalar at wrap boundary");
}

static void test_prefer_128bit(void) {
    /* Just verify the function doesn't crash */
    bool pref = cpu_prefer_128bit_avx();
    const cpu_features_t *cpu = cpu_detect();

    if (cpu->vendor == CPU_VENDOR_AMD && cpu->family == 0x17) {
        TEST_ASSERT_TRUE(pref, "Zen 1/2 should prefer 128-bit");
    } else if (cpu->vendor == CPU_VENDOR_INTEL) {
        TEST_ASSERT_FALSE(pref, "Intel should not prefer 128-bit");
    }
    /* For other cases, just don't crash */
    TEST_ASSERT_TRUE(1, "prefer_128bit should not crash");
}

void test_simd_suite(void) {
    TEST_SUITE("SIMD Detection & FIR Kernels");
    TEST_RUN(test_cpu_detect_sane);
    TEST_RUN(test_simd_dispatch_name);
    TEST_RUN(test_simd_sse2_matches_scalar);
    TEST_RUN(test_simd_avx2_matches_scalar);
    TEST_RUN(test_simd_fir_chain_correctness);
    TEST_RUN(test_simd_circular_wrap);
    TEST_RUN(test_prefer_128bit);
}
