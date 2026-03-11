/*
 * foo_dsd_trellis — Trellis SDM tests
 * Phase 3 will add SINAD measurement.
 */

#include "test.h"
#include "../include/trellis.h"
#include "../include/ntf.h"

static void test_sdm_context_init(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_64);
    TEST_ASSERT_NOT_NULL(f, "auto-select should return a filter");

    sdm_context_t ctx;
    int ret = sdm_context_init(&ctx, f, 8, 16, 64);
    TEST_ASSERT_EQ(ret, 0, "sdm_context_init should succeed");
    TEST_ASSERT_EQ(ctx.num_cands, 1u, "initial candidate count should be 1");

    sdm_context_free(&ctx);
}

static void test_sdm_null_filter(void) {
    sdm_context_t ctx;
    int ret = sdm_context_init(&ctx, NULL, 8, 16, 64);
    TEST_ASSERT_NEQ(ret, 0, "null filter should fail init");
}

static void test_sdm_process_passthrough(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_64);
    sdm_context_t ctx;
    sdm_context_init(&ctx, f, 8, 16, 64);

    float in[16] = { 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
                     1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f };
    float out[16] = {0};

    size_t n = sdm_process_block(&ctx, in, out, 16);
    TEST_ASSERT_EQ(n, 16u, "output count should equal input count");

    /* With stub implementation, output should be sign-quantised ±1.0 */
    for (int i = 0; i < 16; i++) {
        TEST_ASSERT_TRUE(out[i] == 1.0f || out[i] == -1.0f,
                         "output should be ±1.0");
    }

    sdm_context_free(&ctx);
}

void test_trellis_suite(void) {
    TEST_SUITE("Trellis SDM");
    TEST_RUN(test_sdm_context_init);
    TEST_RUN(test_sdm_null_filter);
    TEST_RUN(test_sdm_process_passthrough);
}
