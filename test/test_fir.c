/*
 * foo_dsd_trellis — FIR rate conversion tests
 * Phase 4 will add frequency response validation.
 */

#include "test.h"
#include "../include/fir.h"

static void test_fir_passthrough_init(void) {
    fir_chain_t chain;
    int ret = fir_chain_init(&chain, DSD_RATE_64, DSD_RATE_64);
    TEST_ASSERT_EQ(ret, 0, "passthrough init should succeed");
    TEST_ASSERT_EQ(chain.num_stages, 0, "passthrough should have 0 stages");
    fir_chain_free(&chain);
}

static void test_fir_2x_upsample_init(void) {
    fir_chain_t chain;
    int ret = fir_chain_init(&chain, DSD_RATE_64, DSD_RATE_128);
    TEST_ASSERT_EQ(ret, 0, "2x upsample init should succeed");
    TEST_ASSERT_EQ(chain.num_stages, 1, "2x should need 1 stage");
    fir_chain_free(&chain);
}

static void test_fir_4x_upsample_init(void) {
    fir_chain_t chain;
    int ret = fir_chain_init(&chain, DSD_RATE_64, DSD_RATE_256);
    TEST_ASSERT_EQ(ret, 0, "4x upsample init should succeed");
    TEST_ASSERT_EQ(chain.num_stages, 2, "4x should need 2 stages");
    fir_chain_free(&chain);
}

static void test_fir_8x_upsample_init(void) {
    fir_chain_t chain;
    int ret = fir_chain_init(&chain, DSD_RATE_64, DSD_RATE_512);
    TEST_ASSERT_EQ(ret, 0, "8x upsample init should succeed");
    TEST_ASSERT_EQ(chain.num_stages, 3, "8x should need 3 stages");
    fir_chain_free(&chain);
}

static void test_fir_invalid_ratio(void) {
    fir_chain_t chain;
    /* 3x is not a power-of-2 ratio */
    int ret = fir_chain_init(&chain, DSD_RATE_64, DSD_RATE_64 * 3);
    TEST_ASSERT_NEQ(ret, 0, "non-power-of-2 ratio should fail");
}

void test_fir_suite(void) {
    TEST_SUITE("FIR");
    TEST_RUN(test_fir_passthrough_init);
    TEST_RUN(test_fir_2x_upsample_init);
    TEST_RUN(test_fir_4x_upsample_init);
    TEST_RUN(test_fir_8x_upsample_init);
    TEST_RUN(test_fir_invalid_ratio);
}
