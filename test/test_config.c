/*
 * foo_dsd_trellis — Config serialization tests
 */

#include "test.h"
#include "../include/dsd_types.h"

extern size_t config_serialize(const dsd_config_t *cfg, uint8_t *buf, size_t buf_size);
extern int config_deserialize(dsd_config_t *cfg, const uint8_t *buf, size_t buf_size);
extern void config_validate(dsd_config_t *cfg);

static void test_config_roundtrip(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_out = DSD_RATE_128;
    cfg.gain = 0.75f;
    cfg.trellis_depth = 16;
    cfg.trellis_cands = 24;
    cfg.trellis_lat = 128;
    cfg.ntf_filter = 3;
    cfg.thread_count = 4;
    cfg.mute = true;
    cfg.format = FORMAT_DOP;
    cfg.output_format = OUTPUT_PCM;

    uint8_t buf[128];
    size_t len = config_serialize(&cfg, buf, sizeof(buf));
    TEST_ASSERT_TRUE(len > 0, "serialize should write bytes");

    dsd_config_t cfg2;
    int ret = config_deserialize(&cfg2, buf, len);
    TEST_ASSERT_EQ(ret, 0, "deserialize should succeed");

    TEST_ASSERT_EQ(cfg2.fs_out, DSD_RATE_128, "fs_out roundtrip");
    TEST_ASSERT_FLOAT_EQ(cfg2.gain, 0.75f, 0.001f, "gain roundtrip");
    TEST_ASSERT_EQ(cfg2.trellis_depth, 16, "depth roundtrip");
    TEST_ASSERT_EQ(cfg2.trellis_cands, 24, "cands roundtrip");
    TEST_ASSERT_EQ(cfg2.trellis_lat, 128, "lat roundtrip");
    TEST_ASSERT_EQ(cfg2.ntf_filter, 3, "ntf roundtrip");
    TEST_ASSERT_EQ(cfg2.thread_count, 4, "threads roundtrip");
    TEST_ASSERT_TRUE(cfg2.mute, "mute roundtrip");
    TEST_ASSERT_EQ(cfg2.format, FORMAT_DOP, "format roundtrip");
    TEST_ASSERT_EQ(cfg2.output_format, OUTPUT_PCM, "output_format roundtrip");
}

static void test_config_defaults_roundtrip(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);

    uint8_t buf[128];
    size_t len = config_serialize(&cfg, buf, sizeof(buf));

    dsd_config_t cfg2;
    config_deserialize(&cfg2, buf, len);

    TEST_ASSERT_FLOAT_EQ(cfg2.gain, DSD_DEFAULT_GAIN, 0.001f, "default gain");
    TEST_ASSERT_EQ(cfg2.trellis_depth, DSD_DEFAULT_TRELLIS_N, "default depth");
    TEST_ASSERT_EQ(cfg2.trellis_cands, DSD_DEFAULT_TRELLIS_M, "default cands");
    TEST_ASSERT_EQ(cfg2.trellis_lat, DSD_DEFAULT_TRELLIS_LAT, "default lat");
    TEST_ASSERT_FALSE(cfg2.mute, "default mute");
    TEST_ASSERT_EQ(cfg2.output_format, OUTPUT_DOP, "default output format");
}

static void test_config_corrupt_falls_back(void) {
    uint8_t junk[4] = { 0xFF, 0xFF, 0xFF, 0xFF };
    dsd_config_t cfg;
    config_deserialize(&cfg, junk, sizeof(junk));
    /* Should fall back to defaults */
    TEST_ASSERT_FLOAT_EQ(cfg.gain, DSD_DEFAULT_GAIN, 0.001f,
                         "corrupt data should give default gain");
}

static void test_config_empty_buffer(void) {
    dsd_config_t cfg;
    int ret = config_deserialize(&cfg, NULL, 0);
    TEST_ASSERT_NEQ(ret, 0, "empty buffer should fail");
    TEST_ASSERT_FLOAT_EQ(cfg.gain, DSD_DEFAULT_GAIN, 0.001f,
                         "empty buffer should give defaults");
    (void)ret;
}

static void test_config_validate_clamp(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.gain = 5.0f;
    cfg.trellis_depth = 99;
    cfg.trellis_cands = 100;
    cfg.trellis_lat = 9999;
    cfg.output_format = 99;

    config_validate(&cfg);

    TEST_ASSERT_FLOAT_EQ(cfg.gain, 1.0f, 0.001f, "gain clamped to 1.0");
    TEST_ASSERT_EQ(cfg.trellis_depth, 32, "depth clamped to 32");
    TEST_ASSERT_EQ(cfg.trellis_cands, 32, "cands clamped to 32");
    TEST_ASSERT_EQ(cfg.trellis_lat, 2048, "lat clamped to 2048");
    TEST_ASSERT_EQ(cfg.output_format, OUTPUT_DOP, "invalid output_format clamped");
}

static void test_config_buffer_too_small(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    uint8_t buf[4]; /* way too small */
    size_t len = config_serialize(&cfg, buf, sizeof(buf));
    TEST_ASSERT_EQ(len, 0u, "too-small buffer should return 0");
}

void test_config_suite(void) {
    TEST_SUITE("Config Serialization");
    TEST_RUN(test_config_roundtrip);
    TEST_RUN(test_config_defaults_roundtrip);
    TEST_RUN(test_config_corrupt_falls_back);
    TEST_RUN(test_config_empty_buffer);
    TEST_RUN(test_config_validate_clamp);
    TEST_RUN(test_config_buffer_too_small);
}
