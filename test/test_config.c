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
    cfg.gain = 0.75f;
    cfg.trellis_depth = 16;
    cfg.trellis_cands = 24;
    cfg.trellis_lat = 128;
    cfg.ntf_filter = 3;
    cfg.thread_count = 4;
    cfg.mute = true;
    cfg.format = FORMAT_DOP;
    cfg.output_format = OUTPUT_PCM;

    uint8_t buf[256];
    size_t len = config_serialize(&cfg, buf, sizeof(buf));
    TEST_ASSERT_TRUE(len > 0, "serialize should write bytes");

    dsd_config_t cfg2;
    int ret = config_deserialize(&cfg2, buf, len);
    TEST_ASSERT_EQ(ret, 0, "deserialize should succeed");

    /* fs_out is runtime-only (set from rate_map), not serialized */
    TEST_ASSERT_FLOAT_EQ(cfg2.gain, 0.75f, 0.001f, "gain roundtrip");
    TEST_ASSERT_EQ(cfg2.trellis_depth, 16, "depth roundtrip");
    TEST_ASSERT_EQ(cfg2.trellis_cands, 24, "cands roundtrip");
    TEST_ASSERT_EQ(cfg2.trellis_lat, 128, "lat roundtrip");
    TEST_ASSERT_EQ(cfg2.ntf_filter, 3, "ntf roundtrip");
    TEST_ASSERT_EQ(cfg2.thread_count, 4, "threads roundtrip");
    TEST_ASSERT_TRUE(cfg2.mute, "mute roundtrip");
    TEST_ASSERT_EQ(cfg2.format, FORMAT_DOP, "format roundtrip");
    /* output_format is always forced to OUTPUT_DOP by config_validate */
    TEST_ASSERT_EQ(cfg2.output_format, OUTPUT_DOP, "output_format roundtrip");
    TEST_ASSERT_EQ(cfg2.sdm_mode, SDM_MODE_PRECORR, "sdm_mode roundtrip (default)");
    /* rate_map should be all bypass by default */
    for (int i = 0; i < RATE_MAP_COUNT; i++)
        TEST_ASSERT_EQ(cfg2.rate_map[i], RATE_OUT_BYPASS, "rate_map default bypass");
}

static void test_config_sdm_mode_roundtrip(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.sdm_mode = SDM_MODE_TRELLIS;

    uint8_t buf[256];
    size_t len = config_serialize(&cfg, buf, sizeof(buf));
    TEST_ASSERT_TRUE(len > 0, "serialize should write bytes");

    dsd_config_t cfg2;
    int ret = config_deserialize(&cfg2, buf, len);
    TEST_ASSERT_EQ(ret, 0, "deserialize should succeed");
    TEST_ASSERT_EQ(cfg2.sdm_mode, SDM_MODE_TRELLIS, "sdm_mode=TRELLIS roundtrip");
}

static void test_config_v4_compat(void) {
    /* Simulate a v4 config (no sdm_mode field) */
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.sdm_mode = SDM_MODE_PRECORR;

    uint8_t buf[256];
    size_t len = config_serialize(&cfg, buf, sizeof(buf));
    TEST_ASSERT_TRUE(len > 0, "serialize should work");

    /* Patch to look like v4: set version=4, truncate to 55 bytes */
    uint32_t v4 = 4;
    memcpy(buf, &v4, 4);
    size_t v4_len = 55;

    dsd_config_t cfg2;
    int ret = config_deserialize(&cfg2, buf, v4_len);
    TEST_ASSERT_EQ(ret, 0, "v4 deserialize should succeed");
    /* Pre-v5 should default to Trellis (preserve existing behavior) */
    TEST_ASSERT_EQ(cfg2.sdm_mode, SDM_MODE_TRELLIS,
                   "v4 config should default to Trellis mode");
}

static void test_config_defaults_roundtrip(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);

    uint8_t buf[256];
    size_t len = config_serialize(&cfg, buf, sizeof(buf));

    dsd_config_t cfg2;
    config_deserialize(&cfg2, buf, len);

    TEST_ASSERT_FLOAT_EQ(cfg2.gain, DSD_DEFAULT_GAIN, 0.001f, "default gain");
    TEST_ASSERT_EQ(cfg2.trellis_depth, DSD_DEFAULT_TRELLIS_N, "default depth");
    TEST_ASSERT_EQ(cfg2.trellis_cands, DSD_DEFAULT_TRELLIS_M, "default cands");
    TEST_ASSERT_EQ(cfg2.trellis_lat, DSD_DEFAULT_TRELLIS_LAT, "default lat");
    TEST_ASSERT_FALSE(cfg2.mute, "default mute");
    TEST_ASSERT_EQ(cfg2.output_format, OUTPUT_DOP, "default output format");
    TEST_ASSERT_EQ(cfg2.sdm_mode, SDM_MODE_PRECORR, "default sdm_mode");
    for (int i = 0; i < RATE_MAP_COUNT; i++)
        TEST_ASSERT_EQ(cfg2.rate_map[i], RATE_OUT_BYPASS, "default rate_map bypass");
}

static void test_config_corrupt_falls_back(void) {
    uint8_t junk[4] = { 0xFF, 0xFF, 0xFF, 0xFF };
    dsd_config_t cfg;
    config_deserialize(&cfg, junk, sizeof(junk));
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

    /* Invalid sdm_mode should clamp to PreCorr */
    cfg.sdm_mode = 99;
    config_validate(&cfg);
    TEST_ASSERT_EQ(cfg.sdm_mode, SDM_MODE_PRECORR, "invalid sdm_mode clamped");
}

static void test_config_buffer_too_small(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    uint8_t buf[4]; /* way too small */
    size_t len = config_serialize(&cfg, buf, sizeof(buf));
    TEST_ASSERT_EQ(len, 0u, "too-small buffer should return 0");
}

static void test_config_rate_map_roundtrip(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    /* Set some rate map entries */
    cfg.rate_map[RATEIDX_44100]  = RATE_OUT_DSD64;
    cfg.rate_map[RATEIDX_88200]  = RATE_OUT_DSD128;
    cfg.rate_map[RATEIDX_DSD64]  = RATE_OUT_DSD256;
    cfg.rate_map[RATEIDX_DSD128] = RATE_OUT_DSD512;

    uint8_t buf[256];
    size_t len = config_serialize(&cfg, buf, sizeof(buf));
    TEST_ASSERT_TRUE(len > 0, "serialize should write bytes");

    dsd_config_t cfg2;
    int ret = config_deserialize(&cfg2, buf, len);
    TEST_ASSERT_EQ(ret, 0, "deserialize should succeed");
    TEST_ASSERT_EQ(cfg2.rate_map[RATEIDX_44100],  RATE_OUT_DSD64,  "44100 rate_map roundtrip");
    TEST_ASSERT_EQ(cfg2.rate_map[RATEIDX_88200],  RATE_OUT_DSD128, "88200 rate_map roundtrip");
    TEST_ASSERT_EQ(cfg2.rate_map[RATEIDX_DSD64],  RATE_OUT_DSD256, "DSD64 rate_map roundtrip");
    TEST_ASSERT_EQ(cfg2.rate_map[RATEIDX_DSD128], RATE_OUT_DSD512, "DSD128 rate_map roundtrip");
    /* Unset entries should be bypass */
    TEST_ASSERT_EQ(cfg2.rate_map[RATEIDX_48000],  RATE_OUT_BYPASS, "48000 rate_map bypass");
    TEST_ASSERT_EQ(cfg2.rate_map[RATEIDX_DSD512], RATE_OUT_BYPASS, "DSD512 rate_map bypass");
}

static void test_config_rate_map_validate(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    /* Invalid output index should be clamped to bypass */
    cfg.rate_map[RATEIDX_44100] = 99;
    config_validate(&cfg);
    TEST_ASSERT_EQ(cfg.rate_map[RATEIDX_44100], RATE_OUT_BYPASS,
                   "invalid rate_map entry clamped to bypass");

    /* 48000-family can't convert to DSD — should be forced to bypass */
    cfg.rate_map[RATEIDX_48000] = RATE_OUT_DSD64;
    config_validate(&cfg);
    TEST_ASSERT_EQ(cfg.rate_map[RATEIDX_48000], RATE_OUT_BYPASS,
                   "48000 DSD output forced to bypass");
}

static void test_config_v7_migration(void) {
    /* Serialize a v8 config, then truncate to simulate v7 (remove rate_map).
     * Set fs_out=DSD128 and proc_mode=2 (DSD recode) so migration
     * should populate DSD input entries with RATE_OUT_DSD128. */
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);

    uint8_t buf[256];
    size_t len = config_serialize(&cfg, buf, sizeof(buf));
    TEST_ASSERT_TRUE(len > 0, "serialize should work");

    /* Truncate to v7 size (58 bytes) */
    size_t v7_len = 58;

    /* Patch version to 7 */
    uint32_t v7 = 7;
    memcpy(buf, &v7, 4);

    /* Set fs_out = DSD128 at offset 8 */
    uint32_t fs_out_dsd128 = DSD_RATE_128;
    memcpy(buf + 8, &fs_out_dsd128, 4);

    /* proc_mode is the last byte before rate_map (offset v7_len - 1) */
    buf[v7_len - 1] = 2; /* PROC_DSD_RECODE */

    dsd_config_t cfg2;
    int ret = config_deserialize(&cfg2, buf, v7_len);
    TEST_ASSERT_EQ(ret, 0, "v7 deserialize should succeed");
    /* DSD inputs should be migrated to DSD128 */
    TEST_ASSERT_EQ(cfg2.rate_map[RATEIDX_DSD64],  RATE_OUT_DSD128, "v7 DSD64 migrated");
    TEST_ASSERT_EQ(cfg2.rate_map[RATEIDX_DSD128], RATE_OUT_DSD128, "v7 DSD128 migrated");
    TEST_ASSERT_EQ(cfg2.rate_map[RATEIDX_DSD256], RATE_OUT_DSD128, "v7 DSD256 migrated");
    TEST_ASSERT_EQ(cfg2.rate_map[RATEIDX_DSD512], RATE_OUT_DSD128, "v7 DSD512 migrated");
    /* PCM inputs should remain bypass */
    TEST_ASSERT_EQ(cfg2.rate_map[RATEIDX_44100], RATE_OUT_BYPASS, "v7 PCM stays bypass");
}

static void test_config_rate_ntf_roundtrip(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    /* Set some NTF overrides */
    cfg.rate_ntf[RATEIDX_DSD64]  = 2;  /* specific NTF */
    cfg.rate_ntf[RATEIDX_DSD128] = 5;
    cfg.rate_ntf[RATEIDX_44100]  = 0;  /* first NTF */

    uint8_t buf[256];
    size_t len = config_serialize(&cfg, buf, sizeof(buf));
    TEST_ASSERT_TRUE(len > 0, "serialize should write bytes");

    dsd_config_t cfg2;
    int ret = config_deserialize(&cfg2, buf, len);
    TEST_ASSERT_EQ(ret, 0, "deserialize should succeed");
    TEST_ASSERT_EQ(cfg2.rate_ntf[RATEIDX_DSD64],  2, "DSD64 ntf roundtrip");
    TEST_ASSERT_EQ(cfg2.rate_ntf[RATEIDX_DSD128], 5, "DSD128 ntf roundtrip");
    TEST_ASSERT_EQ(cfg2.rate_ntf[RATEIDX_44100],  0, "44100 ntf roundtrip");
    /* Unset entries should be NTF_AUTO (-1) */
    TEST_ASSERT_EQ(cfg2.rate_ntf[RATEIDX_DSD256], NTF_AUTO, "DSD256 ntf auto");
    TEST_ASSERT_EQ(cfg2.rate_ntf[RATEIDX_48000],  NTF_AUTO, "48000 ntf auto");
}

static void test_config_rate_ntf_validate(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    /* Invalid NTF index should be clamped to NTF_AUTO */
    cfg.rate_ntf[RATEIDX_DSD64] = 99;
    config_validate(&cfg);
    TEST_ASSERT_EQ(cfg.rate_ntf[RATEIDX_DSD64], NTF_AUTO,
                   "invalid rate_ntf clamped to NTF_AUTO");

    /* Negative values other than NTF_AUTO should be clamped */
    cfg.rate_ntf[RATEIDX_DSD128] = -5;
    config_validate(&cfg);
    TEST_ASSERT_EQ(cfg.rate_ntf[RATEIDX_DSD128], NTF_AUTO,
                   "negative rate_ntf clamped to NTF_AUTO");
}

static void test_config_v8_migration(void) {
    /* Serialize a v9 config, truncate to v8 (remove rate_ntf).
     * rate_ntf should default to NTF_AUTO for all entries. */
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.rate_ntf[RATEIDX_DSD64] = 3;  /* will be lost in truncation */

    uint8_t buf[256];
    size_t len = config_serialize(&cfg, buf, sizeof(buf));
    TEST_ASSERT_TRUE(len > 0, "serialize should work");

    /* Truncate to v8 size (70 bytes) */
    size_t v8_len = 70;

    /* Patch version to 8 */
    uint32_t v8 = 8;
    memcpy(buf, &v8, 4);

    dsd_config_t cfg2;
    int ret = config_deserialize(&cfg2, buf, v8_len);
    TEST_ASSERT_EQ(ret, 0, "v8 deserialize should succeed");
    /* All rate_ntf should be NTF_AUTO (not preserved from v8) */
    for (int i = 0; i < RATE_MAP_COUNT; i++)
        TEST_ASSERT_EQ(cfg2.rate_ntf[i], NTF_AUTO, "v8 rate_ntf defaults to NTF_AUTO");
}

void test_config_suite(void) {
    TEST_SUITE("Config Serialization");
    TEST_RUN(test_config_roundtrip);
    TEST_RUN(test_config_defaults_roundtrip);
    TEST_RUN(test_config_corrupt_falls_back);
    TEST_RUN(test_config_empty_buffer);
    TEST_RUN(test_config_validate_clamp);
    TEST_RUN(test_config_buffer_too_small);
    TEST_RUN(test_config_sdm_mode_roundtrip);
    TEST_RUN(test_config_v4_compat);
    TEST_RUN(test_config_rate_map_roundtrip);
    TEST_RUN(test_config_rate_map_validate);
    TEST_RUN(test_config_v7_migration);
    TEST_RUN(test_config_rate_ntf_roundtrip);
    TEST_RUN(test_config_rate_ntf_validate);
    TEST_RUN(test_config_v8_migration);
}
