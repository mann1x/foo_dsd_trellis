/*
 * foo_dsd_trellis — DoP detection and pack/unpack tests
 */

#include "test.h"
#include "../include/dop.h"

#define SCALE_23 8388608.0f  /* 2^23 — same as dop.c */

/* Helper: build a float that encodes a 24-bit integer via the DoP convention */
static float make_dop_sample(uint8_t marker, uint16_t dsd_bits) {
    int32_t val = ((int32_t)marker << 16) | (int32_t)dsd_bits;
    return (float)((double)val / (double)SCALE_23);
}

/* ── dop_detect ─────────────────────────────────────────────── */

static void test_dop_detect_null(void) {
    TEST_ASSERT_FALSE(dop_detect(NULL, 0), "null input");
    TEST_ASSERT_FALSE(dop_detect(NULL, 100), "null input with frames");
}

static void test_dop_detect_single_frame(void) {
    float f = make_dop_sample(DOP_MARKER_A, 0x0000);
    TEST_ASSERT_FALSE(dop_detect(&f, 1), "single frame not enough");
}

static void test_dop_detect_valid_2(void) {
    float pcm[2];
    pcm[0] = make_dop_sample(DOP_MARKER_A, 0xAAAA);
    pcm[1] = make_dop_sample(DOP_MARKER_B, 0x5555);
    TEST_ASSERT_TRUE(dop_detect(pcm, 2), "2-frame valid DoP");
}

static void test_dop_detect_valid_8(void) {
    float pcm[8];
    for (int i = 0; i < 8; i++) {
        uint8_t marker = (i & 1) ? DOP_MARKER_B : DOP_MARKER_A;
        pcm[i] = make_dop_sample(marker, (uint16_t)(i * 0x1111));
    }
    TEST_ASSERT_TRUE(dop_detect(pcm, 8), "8-frame valid DoP");
}

static void test_dop_detect_valid_large(void) {
    /* Only first 8 frames are checked, rest can be anything */
    float pcm[64];
    memset(pcm, 0, sizeof(pcm));
    for (int i = 0; i < 8; i++) {
        uint8_t marker = (i & 1) ? DOP_MARKER_B : DOP_MARKER_A;
        pcm[i] = make_dop_sample(marker, 0);
    }
    TEST_ASSERT_TRUE(dop_detect(pcm, 64), "large buffer, valid first 8");
}

static void test_dop_detect_wrong_marker(void) {
    float pcm[2];
    pcm[0] = make_dop_sample(DOP_MARKER_A, 0);
    pcm[1] = make_dop_sample(DOP_MARKER_A, 0); /* should be B */
    TEST_ASSERT_FALSE(dop_detect(pcm, 2), "same marker both frames");
}

static void test_dop_detect_swapped_markers(void) {
    float pcm[2];
    pcm[0] = make_dop_sample(DOP_MARKER_B, 0); /* should be A */
    pcm[1] = make_dop_sample(DOP_MARKER_A, 0);
    TEST_ASSERT_FALSE(dop_detect(pcm, 2), "swapped markers");
}

static void test_dop_detect_random(void) {
    float pcm[8] = { 0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f };
    TEST_ASSERT_FALSE(dop_detect(pcm, 8), "random PCM data");
}

/* ── dop_unpack / dop_pack round-trip ───────────────────────── */

static void test_dop_unpack_null(void) {
    float bits[16];
    dop_unpack(NULL, bits, 1);     /* should not crash */
    dop_unpack(bits, NULL, 1);
    dop_unpack(bits, bits, 0);
    TEST_ASSERT_TRUE(1, "dop_unpack null/zero no crash");
}

static void test_dop_pack_null(void) {
    float pcm[1];
    dop_pack(NULL, pcm, 16);
    dop_pack(pcm, NULL, 16);
    dop_pack(pcm, pcm, 0);
    TEST_ASSERT_TRUE(1, "dop_pack null/zero no crash");
}

static void test_dop_unpack_known(void) {
    /* 0xFFFF → all 1-bits → all +1.0f */
    float pcm[1] = { make_dop_sample(DOP_MARKER_A, 0xFFFF) };
    float bits[16];
    dop_unpack(pcm, bits, 1);
    for (int i = 0; i < 16; i++) {
        TEST_ASSERT_FLOAT_EQ(bits[i], 1.0f, 0.001f, "all-ones unpack");
    }
}

static void test_dop_unpack_zeros(void) {
    /* 0x0000 → all 0-bits → all -1.0f */
    float pcm[1] = { make_dop_sample(DOP_MARKER_A, 0x0000) };
    float bits[16];
    dop_unpack(pcm, bits, 1);
    for (int i = 0; i < 16; i++) {
        TEST_ASSERT_FLOAT_EQ(bits[i], -1.0f, 0.001f, "all-zeros unpack");
    }
}

static void test_dop_unpack_alternating(void) {
    /* 0xAAAA = 1010 1010 1010 1010 → alternating +1/-1 starting +1 */
    float pcm[1] = { make_dop_sample(DOP_MARKER_A, 0xAAAA) };
    float bits[16];
    dop_unpack(pcm, bits, 1);
    for (int i = 0; i < 16; i++) {
        float expected = (i % 2 == 0) ? 1.0f : -1.0f;
        TEST_ASSERT_FLOAT_EQ(bits[i], expected, 0.001f, "alternating unpack");
    }
}

static void test_dop_roundtrip(void) {
    /* Build 4 DoP frames with known patterns, unpack → repack, compare */
    uint16_t patterns[4] = { 0xDEAD, 0xBEEF, 0x1234, 0x5678 };
    float pcm_in[4], pcm_out[4];
    float bits[4 * 16];

    for (int i = 0; i < 4; i++) {
        uint8_t marker = (i & 1) ? DOP_MARKER_B : DOP_MARKER_A;
        pcm_in[i] = make_dop_sample(marker, patterns[i]);
    }

    dop_unpack(pcm_in, bits, 4);
    dop_pack(bits, pcm_out, 4 * 16);

    /* Verify DSD payload round-trips exactly.
     * Markers should match (pack uses same alternation). */
    for (int i = 0; i < 4; i++) {
        /* Compare as int24 to avoid float precision issues */
        int32_t in_val  = (int32_t)(pcm_in[i]  * SCALE_23 + (pcm_in[i]  >= 0 ? 0.5f : -0.5f));
        int32_t out_val = (int32_t)(pcm_out[i] * SCALE_23 + (pcm_out[i] >= 0 ? 0.5f : -0.5f));
        TEST_ASSERT_EQ(in_val, out_val, "DoP round-trip frame match");
    }
}

static void test_dop_roundtrip_silence(void) {
    /* DSD silence = 0x69 repeating (1-0 pattern) = 0x6969 per frame */
    float pcm_in[8], pcm_out[8];
    float bits[8 * 16];

    for (int i = 0; i < 8; i++) {
        uint8_t marker = (i & 1) ? DOP_MARKER_B : DOP_MARKER_A;
        pcm_in[i] = make_dop_sample(marker, 0x6969);
    }

    dop_unpack(pcm_in, bits, 8);
    dop_pack(bits, pcm_out, 8 * 16);

    for (int i = 0; i < 8; i++) {
        int32_t in_val  = (int32_t)(pcm_in[i]  * SCALE_23 + (pcm_in[i]  >= 0 ? 0.5f : -0.5f));
        int32_t out_val = (int32_t)(pcm_out[i] * SCALE_23 + (pcm_out[i] >= 0 ? 0.5f : -0.5f));
        TEST_ASSERT_EQ(in_val, out_val, "DoP silence round-trip");
    }
}

/* ── bits_unpack / bits_pack round-trip ─────────────────────── */

static void test_bits_unpack_null(void) {
    float dst[8];
    uint8_t src[1] = {0};
    bits_unpack(NULL, dst, 8);
    bits_unpack(src, NULL, 8);
    bits_unpack(src, dst, 0);
    TEST_ASSERT_TRUE(1, "bits_unpack null/zero no crash");
}

static void test_bits_pack_null(void) {
    float src[8];
    uint8_t dst[1] = {0};
    bits_pack(NULL, dst, 8);
    bits_pack(src, NULL, 8);
    bits_pack(src, dst, 0);
    TEST_ASSERT_TRUE(1, "bits_pack null/zero no crash");
}

static void test_bits_unpack_0xff(void) {
    uint8_t src[1] = { 0xFF };
    float dst[8];
    bits_unpack(src, dst, 8);
    for (int i = 0; i < 8; i++) {
        TEST_ASSERT_FLOAT_EQ(dst[i], 1.0f, 0.001f, "0xFF → all +1.0");
    }
}

static void test_bits_unpack_0x00(void) {
    uint8_t src[1] = { 0x00 };
    float dst[8];
    bits_unpack(src, dst, 8);
    for (int i = 0; i < 8; i++) {
        TEST_ASSERT_FLOAT_EQ(dst[i], -1.0f, 0.001f, "0x00 → all -1.0");
    }
}

static void test_bits_unpack_0xa5(void) {
    /* 0xA5 = 1010 0101 */
    uint8_t src[1] = { 0xA5 };
    float dst[8];
    float expected[8] = { 1,-1,1,-1, -1,1,-1,1 };
    bits_unpack(src, dst, 8);
    for (int i = 0; i < 8; i++) {
        TEST_ASSERT_FLOAT_EQ(dst[i], expected[i], 0.001f, "0xA5 pattern");
    }
}

static void test_bits_roundtrip(void) {
    uint8_t src[4] = { 0xDE, 0xAD, 0xBE, 0xEF };
    float bits[32];
    uint8_t dst[4] = {0};

    bits_unpack(src, bits, 32);
    bits_pack(bits, dst, 32);

    TEST_ASSERT_EQ(memcmp(src, dst, 4), 0, "bits round-trip 4 bytes");
}

static void test_bits_roundtrip_partial(void) {
    /* Test partial byte: 5 bits from 0xF8 (= 1111 1000) */
    uint8_t src[1] = { 0xF8 };
    float bits[5];
    uint8_t dst[1] = {0};

    bits_unpack(src, bits, 5);
    bits_pack(bits, dst, 5);

    /* Only upper 5 bits should match: 0xF8 & 0xF8 = 0xF8 */
    uint8_t mask = 0xF8;  /* upper 5 bits */
    TEST_ASSERT_EQ(dst[0] & mask, src[0] & mask, "bits partial round-trip");
}

static void test_bits_roundtrip_large(void) {
    uint8_t src[32], dst[32];
    float bits[256];
    for (int i = 0; i < 32; i++) src[i] = (uint8_t)(i * 7 + 13);

    bits_unpack(src, bits, 256);
    bits_pack(bits, dst, 256);

    TEST_ASSERT_EQ(memcmp(src, dst, 32), 0, "bits round-trip 32 bytes");
}

/* ── suite entry point ──────────────────────────────────────── */

void test_dop_suite(void) {
    TEST_SUITE("DoP");

    TEST_RUN(test_dop_detect_null);
    TEST_RUN(test_dop_detect_single_frame);
    TEST_RUN(test_dop_detect_valid_2);
    TEST_RUN(test_dop_detect_valid_8);
    TEST_RUN(test_dop_detect_valid_large);
    TEST_RUN(test_dop_detect_wrong_marker);
    TEST_RUN(test_dop_detect_swapped_markers);
    TEST_RUN(test_dop_detect_random);

    TEST_RUN(test_dop_unpack_null);
    TEST_RUN(test_dop_pack_null);
    TEST_RUN(test_dop_unpack_known);
    TEST_RUN(test_dop_unpack_zeros);
    TEST_RUN(test_dop_unpack_alternating);
    TEST_RUN(test_dop_roundtrip);
    TEST_RUN(test_dop_roundtrip_silence);

    TEST_RUN(test_bits_unpack_null);
    TEST_RUN(test_bits_pack_null);
    TEST_RUN(test_bits_unpack_0xff);
    TEST_RUN(test_bits_unpack_0x00);
    TEST_RUN(test_bits_unpack_0xa5);
    TEST_RUN(test_bits_roundtrip);
    TEST_RUN(test_bits_roundtrip_partial);
    TEST_RUN(test_bits_roundtrip_large);
}
