/*
 * foo_dsd_trellis — CPU detection and IPP FIR engine tests
 */

#include "test.h"
#include "../include/simd_detect.h"
#include "../include/fir.h"
#include "../include/engine.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

static void test_ipp_kernel_name(void) {
    const char *name = fir_ipp_kernel_name();
    TEST_ASSERT_NOT_NULL(name, "IPP kernel name should not be null");
    printf("    FIR engine: %s\n", name);
}

static void test_ipp_version(void) {
    const char *ver = fir_ipp_version();
    TEST_ASSERT_NOT_NULL(ver, "IPP version should not be null");
    printf("    IPP version: %s\n", ver);
}

static void test_fir_chain_correctness(void) {
    /* Process through FIR chain — verify output is valid DSD */
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_128;
    cfg.gain = 0.5f;

    engine_channel_t eng;
    int ret = engine_channel_init(&eng, 0, &cfg);
    TEST_ASSERT_EQ(ret, 0, "engine init for FIR test");

    unsigned n = 1024;
    float *in  = (float *)malloc(n * sizeof(float));
    float *out = (float *)malloc(n * 4 * sizeof(float));

    for (unsigned i = 0; i < n; i++)
        in[i] = (float)(0.3 * sin(2.0 * M_PI * 1000.0 * i / DSD_RATE_64));

    size_t n_out = engine_process_block(&eng, in, out, n, &cfg);
    TEST_ASSERT_TRUE(n_out > 0, "FIR chain should produce output");

    /* All output must be +/-1.0 */
    int valid = 1;
    for (size_t i = 0; i < n_out; i++) {
        if (out[i] != 1.0f && out[i] != -1.0f) { valid = 0; break; }
    }
    TEST_ASSERT_TRUE(valid, "FIR chain output should be +/-1.0");

    free(in);
    free(out);
    engine_channel_free(&eng);
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
    TEST_ASSERT_TRUE(1, "prefer_128bit should not crash");
}

void test_simd_suite(void) {
    TEST_SUITE("CPU & IPP FIR Engine");
    TEST_RUN(test_cpu_detect_sane);
    TEST_RUN(test_ipp_kernel_name);
    TEST_RUN(test_ipp_version);
    TEST_RUN(test_fir_chain_correctness);
    TEST_RUN(test_prefer_128bit);
}
