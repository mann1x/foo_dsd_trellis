/*
 * foo_dsd_trellis — Rate map validation tests
 *
 * Tests rate_map_valid_output(), rate_map_index(), rate family helpers,
 * and resample_needed() for all input/output combinations.
 */

#include "test.h"
#include "../include/dsd_types.h"
#include "../include/resample.h"

/* ─── rate_map_index tests ─── */

static void test_rate_map_index_pcm(void) {
    TEST_ASSERT_EQ(rate_map_index(44100),   RATEIDX_44100,   "44100 index");
    TEST_ASSERT_EQ(rate_map_index(48000),   RATEIDX_48000,   "48000 index");
    TEST_ASSERT_EQ(rate_map_index(88200),   RATEIDX_88200,   "88200 index");
    TEST_ASSERT_EQ(rate_map_index(96000),   RATEIDX_96000,   "96000 index");
    TEST_ASSERT_EQ(rate_map_index(176400),  RATEIDX_176400,  "176400 index");
    TEST_ASSERT_EQ(rate_map_index(192000),  RATEIDX_192000,  "192000 index");
    TEST_ASSERT_EQ(rate_map_index(352800),  RATEIDX_352800,  "352800 index");
    TEST_ASSERT_EQ(rate_map_index(384000),  RATEIDX_384000,  "384000 index");
    TEST_ASSERT_EQ(rate_map_index(705600),  RATEIDX_705600,  "705600 index");
    TEST_ASSERT_EQ(rate_map_index(768000),  RATEIDX_768000,  "768000 index");
    TEST_ASSERT_EQ(rate_map_index(1411200), RATEIDX_1411200, "1411200 index");
    TEST_ASSERT_EQ(rate_map_index(1536000), RATEIDX_1536000, "1536000 index");
}

static void test_rate_map_index_dsd(void) {
    TEST_ASSERT_EQ(rate_map_index(DSD_RATE_64),    RATEIDX_DSD64,     "DSD64 index");
    TEST_ASSERT_EQ(rate_map_index(DSD_RATE_128),   RATEIDX_DSD128,    "DSD128 index");
    TEST_ASSERT_EQ(rate_map_index(DSD_RATE_256),   RATEIDX_DSD256,    "DSD256 index");
    TEST_ASSERT_EQ(rate_map_index(DSD_RATE_512),   RATEIDX_DSD512,    "DSD512 index");
    TEST_ASSERT_EQ(rate_map_index(DSD48_RATE_64),  RATEIDX_DSD64_48,  "DSD64/48 index");
    TEST_ASSERT_EQ(rate_map_index(DSD48_RATE_128), RATEIDX_DSD128_48, "DSD128/48 index");
    TEST_ASSERT_EQ(rate_map_index(DSD48_RATE_256), RATEIDX_DSD256_48, "DSD256/48 index");
    TEST_ASSERT_EQ(rate_map_index(DSD48_RATE_512), RATEIDX_DSD512_48, "DSD512/48 index");
}

static void test_rate_map_index_unknown(void) {
    TEST_ASSERT_EQ(rate_map_index(0),      -1, "0 unknown");
    TEST_ASSERT_EQ(rate_map_index(22050),  -1, "22050 unknown");
    TEST_ASSERT_EQ(rate_map_index(99999),  -1, "99999 unknown");
}

/* ─── rate_idx_to_hz roundtrip ─── */

static void test_rate_idx_to_hz(void) {
    for (int i = 0; i < RATE_MAP_COUNT; i++) {
        uint32_t hz = rate_idx_to_hz(i);
        TEST_ASSERT_TRUE(hz != 0, "rate_idx_to_hz should return nonzero");
        int back = rate_map_index(hz);
        TEST_ASSERT_EQ(back, i, "rate_idx_to_hz/rate_map_index roundtrip");
    }
}

/* ─── Family helpers ─── */

static void test_rate_family_44k(void) {
    TEST_ASSERT_TRUE(rate_is_44k_family(44100),   "44100 is 44k family");
    TEST_ASSERT_TRUE(rate_is_44k_family(88200),   "88200 is 44k family");
    TEST_ASSERT_TRUE(rate_is_44k_family(176400),  "176400 is 44k family");
    TEST_ASSERT_TRUE(rate_is_44k_family(352800),  "352800 is 44k family");
    TEST_ASSERT_TRUE(rate_is_44k_family(705600),  "705600 is 44k family");
    TEST_ASSERT_TRUE(rate_is_44k_family(1411200), "1411200 is 44k family");
    TEST_ASSERT_TRUE(rate_is_44k_family(DSD_RATE_64), "DSD64 is 44k family");
}

static void test_rate_family_48k(void) {
    TEST_ASSERT_TRUE(rate_is_48k_family(48000),   "48000 is 48k family");
    TEST_ASSERT_TRUE(rate_is_48k_family(96000),   "96000 is 48k family");
    TEST_ASSERT_TRUE(rate_is_48k_family(192000),  "192000 is 48k family");
    TEST_ASSERT_TRUE(rate_is_48k_family(384000),  "384000 is 48k family");
    TEST_ASSERT_TRUE(rate_is_48k_family(768000),  "768000 is 48k family");
    TEST_ASSERT_TRUE(rate_is_48k_family(1536000), "1536000 is 48k family");
    TEST_ASSERT_TRUE(rate_is_48k_family(DSD48_RATE_64), "DSD64/48 is 48k family");
    /* Cross-check: 48k should NOT be 44k family */
    TEST_ASSERT_FALSE(rate_is_44k_family(48000), "48000 not 44k family");
    TEST_ASSERT_FALSE(rate_is_48k_family(44100), "44100 not 48k family");
}

/* ─── rate_map_valid_output ─── */

static void test_valid_bypass(void) {
    /* Bypass is always valid for any input */
    for (int i = 0; i < RATE_MAP_COUNT; i++)
        TEST_ASSERT_TRUE(rate_map_valid_output(i, RATE_OUT_BYPASS), "bypass always valid");
}

static void test_valid_pcm_to_dsd_same_family(void) {
    /* 44.1k PCM → DSD/44: valid */
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_44100, RATE_OUT_DSD64), "44.1k→DSD64");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_88200, RATE_OUT_DSD128), "88.2k→DSD128");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_352800, RATE_OUT_DSD512), "352.8k→DSD512");
    /* 48k PCM → DSD/48: valid */
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_48000, RATE_OUT_DSD64_48), "48k→DSD64/48");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_96000, RATE_OUT_DSD128_48), "96k→DSD128/48");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_384000, RATE_OUT_DSD512_48), "384k→DSD512/48");
}

static void test_valid_pcm_to_dsd_cross_family(void) {
    /* Cross-family PCM→DSD: valid (polyphase resample + FIR upsample) */
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_44100, RATE_OUT_DSD64_48), "44.1k→DSD64/48 cross-family");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_88200, RATE_OUT_DSD128_48), "88.2k→DSD128/48 cross-family");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_48000, RATE_OUT_DSD64), "48k→DSD64 cross-family");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_96000, RATE_OUT_DSD128), "96k→DSD128 cross-family");
}

static void test_valid_pcm_to_pcm(void) {
    /* PCM→PCM: any combination valid (same-family FIR or cross-family polyphase) */
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_44100, RATE_OUT_PCM48), "44.1k→48k");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_48000, RATE_OUT_PCM44), "48k→44.1k");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_44100, RATE_OUT_PCM88), "44.1k→88.2k");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_96000, RATE_OUT_PCM176), "96k→176.4k");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_44100, RATE_OUT_PCM1536), "44.1k→1536k");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_1536000, RATE_OUT_PCM44), "1536k→44.1k");
}

static void test_valid_dsd_to_pcm(void) {
    /* DSD→PCM: any combination valid */
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_DSD64, RATE_OUT_PCM44), "DSD64→44.1k");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_DSD64, RATE_OUT_PCM48), "DSD64→48k cross-family");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_DSD128, RATE_OUT_PCM96), "DSD128→96k cross-family");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_DSD64_48, RATE_OUT_PCM48), "DSD64/48→48k");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_DSD64_48, RATE_OUT_PCM44), "DSD64/48→44.1k cross-family");
}

static void test_valid_dsd_to_dsd_same_family(void) {
    /* DSD/44→DSD/44: valid */
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_DSD64, RATE_OUT_DSD128), "DSD64→DSD128");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_DSD256, RATE_OUT_DSD64), "DSD256→DSD64");
    /* DSD/48→DSD/48: valid */
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_DSD64_48, RATE_OUT_DSD128_48), "DSD64/48→DSD128/48");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_DSD256_48, RATE_OUT_DSD64_48), "DSD256/48→DSD64/48");
}

static void test_invalid_dsd_to_dsd_cross_family(void) {
    /* DSD/44→DSD/48: INVALID */
    TEST_ASSERT_FALSE(rate_map_valid_output(RATEIDX_DSD64, RATE_OUT_DSD64_48), "DSD64→DSD64/48 invalid");
    TEST_ASSERT_FALSE(rate_map_valid_output(RATEIDX_DSD128, RATE_OUT_DSD128_48), "DSD128→DSD128/48 invalid");
    /* DSD/48→DSD/44: INVALID */
    TEST_ASSERT_FALSE(rate_map_valid_output(RATEIDX_DSD64_48, RATE_OUT_DSD64), "DSD64/48→DSD64 invalid");
    TEST_ASSERT_FALSE(rate_map_valid_output(RATEIDX_DSD128_48, RATE_OUT_DSD128), "DSD128/48→DSD128 invalid");
}

static void test_invalid_out_of_range(void) {
    TEST_ASSERT_FALSE(rate_map_valid_output(0, RATE_OUT_COUNT), "out of range output");
    TEST_ASSERT_FALSE(rate_map_valid_output(0, 99), "output 99 invalid");
    TEST_ASSERT_FALSE(rate_map_valid_output(-1, RATE_OUT_DSD64), "negative input index");
    TEST_ASSERT_FALSE(rate_map_valid_output(RATE_MAP_COUNT, RATE_OUT_DSD64), "input index too high");
}

/* ─── High PCM rates ─── */

static void test_valid_high_pcm_to_dsd(void) {
    /* 705600 (44.1k family) → DSD/44 */
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_705600, RATE_OUT_DSD512), "705.6k→DSD512");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_705600, RATE_OUT_DSD512_48), "705.6k→DSD512/48 cross-family");
    /* 768000 (48k family) → DSD/48 */
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_768000, RATE_OUT_DSD512_48), "768k→DSD512/48");
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_768000, RATE_OUT_DSD512), "768k→DSD512 cross-family");
    /* 1411200 (44.1k family) → DSD/44 */
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_1411200, RATE_OUT_DSD512), "1411.2k→DSD512");
    /* 1536000 (48k family) → DSD/48 */
    TEST_ASSERT_TRUE(rate_map_valid_output(RATEIDX_1536000, RATE_OUT_DSD512_48), "1536k→DSD512/48");
}

/* ─── rate_out conversion helpers ─── */

static void test_rate_out_to_dsd(void) {
    TEST_ASSERT_EQ(rate_out_to_dsd(RATE_OUT_DSD64),     DSD_RATE_64,    "DSD64 out→hz");
    TEST_ASSERT_EQ(rate_out_to_dsd(RATE_OUT_DSD512),    DSD_RATE_512,   "DSD512 out→hz");
    TEST_ASSERT_EQ(rate_out_to_dsd(RATE_OUT_DSD64_48),  DSD48_RATE_64,  "DSD64/48 out→hz");
    TEST_ASSERT_EQ(rate_out_to_dsd(RATE_OUT_DSD512_48), DSD48_RATE_512, "DSD512/48 out→hz");
    TEST_ASSERT_EQ(rate_out_to_dsd(RATE_OUT_PCM44),     0u,             "PCM44 not DSD");
    TEST_ASSERT_EQ(rate_out_to_dsd(RATE_OUT_BYPASS),    0u,             "bypass not DSD");
}

static void test_rate_out_to_pcm(void) {
    TEST_ASSERT_EQ(rate_out_to_pcm(RATE_OUT_PCM44),   44100u,   "PCM44 out→hz");
    TEST_ASSERT_EQ(rate_out_to_pcm(RATE_OUT_PCM48),   48000u,   "PCM48 out→hz");
    TEST_ASSERT_EQ(rate_out_to_pcm(RATE_OUT_PCM96),   96000u,   "PCM96 out→hz");
    TEST_ASSERT_EQ(rate_out_to_pcm(RATE_OUT_PCM706),  705600u,  "PCM706 out→hz");
    TEST_ASSERT_EQ(rate_out_to_pcm(RATE_OUT_PCM1536), 1536000u, "PCM1536 out→hz");
    TEST_ASSERT_EQ(rate_out_to_pcm(RATE_OUT_DSD64),   0u,       "DSD64 not PCM");
}

void test_validation_suite(void) {
    TEST_SUITE("Validation");
    TEST_RUN(test_rate_map_index_pcm);
    TEST_RUN(test_rate_map_index_dsd);
    TEST_RUN(test_rate_map_index_unknown);
    TEST_RUN(test_rate_idx_to_hz);
    TEST_RUN(test_rate_family_44k);
    TEST_RUN(test_rate_family_48k);
    TEST_RUN(test_valid_bypass);
    TEST_RUN(test_valid_pcm_to_dsd_same_family);
    TEST_RUN(test_valid_pcm_to_dsd_cross_family);
    TEST_RUN(test_valid_pcm_to_pcm);
    TEST_RUN(test_valid_dsd_to_pcm);
    TEST_RUN(test_valid_dsd_to_dsd_same_family);
    TEST_RUN(test_invalid_dsd_to_dsd_cross_family);
    TEST_RUN(test_invalid_out_of_range);
    TEST_RUN(test_valid_high_pcm_to_dsd);
    TEST_RUN(test_rate_out_to_dsd);
    TEST_RUN(test_rate_out_to_pcm);
}
