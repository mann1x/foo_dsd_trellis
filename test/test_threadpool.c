/*
 * foo_dsd_trellis — Thread pool tests
 * Phase 5: Concurrent processing, stress, correctness.
 */

#include "test.h"
#include "../include/threadpool.h"
#include "../include/engine.h"
#include "../include/ntf.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void test_threadpool_create_destroy(void) {
    threadpool_t *pool = threadpool_create(2, 0);
    TEST_ASSERT_NOT_NULL(pool, "pool creation should succeed");
    TEST_ASSERT_EQ(threadpool_get_thread_count(pool), 2,
                   "thread count should be 2");
    threadpool_destroy(pool);
}

static void test_threadpool_auto_count(void) {
    threadpool_t *pool = threadpool_create(0, 0);
    TEST_ASSERT_NOT_NULL(pool, "auto-count pool should succeed");
    TEST_ASSERT_TRUE(threadpool_get_thread_count(pool) >= 1,
                     "auto-count should give at least 1 thread");
    threadpool_destroy(pool);
}

static void test_threadpool_single_block(void) {
    /* Submit a single mute block and verify output */
    threadpool_t *pool = threadpool_create(2, 0);

    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.mute = true;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    float in[64], out[64];
    memset(in, 0, sizeof(in));
    memset(out, 0, sizeof(out));

    channel_block_t block = {
        .in = in, .out = out, .count = 64,
        .out_count = 0, .channel = 0,
        .eng = &eng, .cfg = &cfg
    };

    threadpool_submit(pool, &block);
    threadpool_wait(pool);

    TEST_ASSERT_EQ(block.out_count, 64u, "mute block should produce 64 samples");

    /* Verify mute pattern (alternating +/-1.0) */
    int valid = 1;
    for (int i = 0; i < 64; i++) {
        if (out[i] != 1.0f && out[i] != -1.0f) {
            valid = 0;
            break;
        }
    }
    TEST_ASSERT_TRUE(valid, "mute output should be +/-1.0 pattern");

    engine_channel_free(&eng);
    threadpool_destroy(pool);
}

static void test_threadpool_stereo_concurrent(void) {
    /* Process two channels concurrently — both mute for simplicity */
    threadpool_t *pool = threadpool_create(2, 0);

    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.mute = true;

    engine_channel_t eng_l, eng_r;
    engine_channel_init(&eng_l, 0, &cfg);
    engine_channel_init(&eng_r, 1, &cfg);

    float in_l[128], in_r[128], out_l[128], out_r[128];
    memset(in_l, 0, sizeof(in_l));
    memset(in_r, 0, sizeof(in_r));

    channel_block_t blocks[2] = {
        { .in = in_l, .out = out_l, .count = 128, .out_count = 0,
          .channel = 0, .eng = &eng_l, .cfg = &cfg },
        { .in = in_r, .out = out_r, .count = 128, .out_count = 0,
          .channel = 1, .eng = &eng_r, .cfg = &cfg },
    };

    threadpool_submit(pool, &blocks[0]);
    threadpool_submit(pool, &blocks[1]);
    threadpool_wait(pool);

    TEST_ASSERT_EQ(blocks[0].out_count, 128u, "left channel should produce 128");
    TEST_ASSERT_EQ(blocks[1].out_count, 128u, "right channel should produce 128");

    engine_channel_free(&eng_l);
    engine_channel_free(&eng_r);
    threadpool_destroy(pool);
}

static void test_threadpool_passthrough_concurrent(void) {
    /* Passthrough mode: same rate, unity gain, no mute */
    threadpool_t *pool = threadpool_create(2, 0);

    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_64;
    cfg.gain = 1.0f;

    engine_channel_t eng_l, eng_r;
    engine_channel_init(&eng_l, 0, &cfg);
    engine_channel_init(&eng_r, 1, &cfg);

    float in_l[256], in_r[256], out_l[256], out_r[256];
    /* Fill with known DSD-like pattern */
    for (int i = 0; i < 256; i++) {
        in_l[i] = (i % 3 == 0) ? 1.0f : -1.0f;
        in_r[i] = (i % 5 == 0) ? 1.0f : -1.0f;
    }

    channel_block_t blocks[2] = {
        { .in = in_l, .out = out_l, .count = 256, .out_count = 0,
          .channel = 0, .eng = &eng_l, .cfg = &cfg },
        { .in = in_r, .out = out_r, .count = 256, .out_count = 0,
          .channel = 1, .eng = &eng_r, .cfg = &cfg },
    };

    threadpool_submit(pool, &blocks[0]);
    threadpool_submit(pool, &blocks[1]);
    threadpool_wait(pool);

    TEST_ASSERT_EQ(blocks[0].out_count, 256u, "L passthrough count");
    TEST_ASSERT_EQ(blocks[1].out_count, 256u, "R passthrough count");

    /* Passthrough sign-requantises: output should be +/-1.0 matching input sign */
    int match_l = 1, match_r = 1;
    for (int i = 0; i < 256; i++) {
        float expected_l = in_l[i] >= 0.0f ? 1.0f : -1.0f;
        float expected_r = in_r[i] >= 0.0f ? 1.0f : -1.0f;
        if (out_l[i] != expected_l) match_l = 0;
        if (out_r[i] != expected_r) match_r = 0;
    }
    TEST_ASSERT_TRUE(match_l, "L passthrough should preserve sign");
    TEST_ASSERT_TRUE(match_r, "R passthrough should preserve sign");

    engine_channel_free(&eng_l);
    engine_channel_free(&eng_r);
    threadpool_destroy(pool);
}

static void test_threadpool_stress(void) {
    /* Submit many blocks in multiple batches */
    threadpool_t *pool = threadpool_create(4, 0);

    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.mute = true;

    #define STRESS_CHANNELS 8
    engine_channel_t eng[STRESS_CHANNELS];
    float in_buf[STRESS_CHANNELS][64];
    float out_buf[STRESS_CHANNELS][64];
    channel_block_t blocks[STRESS_CHANNELS];

    for (int i = 0; i < STRESS_CHANNELS; i++) {
        engine_channel_init(&eng[i], i, &cfg);
        memset(in_buf[i], 0, sizeof(in_buf[i]));
    }

    /* Run 10 batches */
    int all_ok = 1;
    for (int batch = 0; batch < 10; batch++) {
        for (int i = 0; i < STRESS_CHANNELS; i++) {
            blocks[i].in = in_buf[i];
            blocks[i].out = out_buf[i];
            blocks[i].count = 64;
            blocks[i].out_count = 0;
            blocks[i].channel = i;
            blocks[i].eng = &eng[i];
            blocks[i].cfg = &cfg;
            threadpool_submit(pool, &blocks[i]);
        }
        threadpool_wait(pool);

        for (int i = 0; i < STRESS_CHANNELS; i++) {
            if (blocks[i].out_count != 64) {
                all_ok = 0;
                break;
            }
        }
        if (!all_ok) break;
    }

    TEST_ASSERT_TRUE(all_ok, "all stress batches should produce correct counts");

    for (int i = 0; i < STRESS_CHANNELS; i++)
        engine_channel_free(&eng[i]);
    threadpool_destroy(pool);
    #undef STRESS_CHANNELS
}

static void test_threadpool_empty_wait(void) {
    /* Calling wait with nothing submitted should return immediately */
    threadpool_t *pool = threadpool_create(2, 0);
    threadpool_wait(pool);  /* Should not hang */
    TEST_ASSERT_TRUE(1, "empty wait should return immediately");
    threadpool_destroy(pool);
}

static void test_threadpool_sdm_concurrent(void) {
    /* Process two channels through full SDM pipeline concurrently */
    threadpool_t *pool = threadpool_create(2, 0);

    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_64;
    cfg.gain = 0.5f;  /* Non-unity gain forces SDM path */
    cfg.trellis_depth = 8;
    cfg.trellis_cands = 8;
    cfg.trellis_lat = 64;

    engine_channel_t eng_l, eng_r;
    int ret_l = engine_channel_init(&eng_l, 0, &cfg);
    int ret_r = engine_channel_init(&eng_r, 1, &cfg);
    TEST_ASSERT_EQ(ret_l, 0, "L engine init should succeed");
    TEST_ASSERT_EQ(ret_r, 0, "R engine init should succeed");

    unsigned n = 512;
    float *in_l  = (float *)malloc(n * sizeof(float));
    float *in_r  = (float *)malloc(n * sizeof(float));
    float *out_l = (float *)malloc(n * sizeof(float));
    float *out_r = (float *)malloc(n * sizeof(float));

    /* Different sine frequencies per channel */
    for (unsigned i = 0; i < n; i++) {
        in_l[i] = (float)(0.5 * sin(2.0 * M_PI * 1000.0 * i / DSD_RATE_64));
        in_r[i] = (float)(0.5 * sin(2.0 * M_PI * 2000.0 * i / DSD_RATE_64));
    }

    channel_block_t blocks[2] = {
        { .in = in_l, .out = out_l, .count = n, .out_count = 0,
          .channel = 0, .eng = &eng_l, .cfg = &cfg },
        { .in = in_r, .out = out_r, .count = n, .out_count = 0,
          .channel = 1, .eng = &eng_r, .cfg = &cfg },
    };

    threadpool_submit(pool, &blocks[0]);
    threadpool_submit(pool, &blocks[1]);
    threadpool_wait(pool);

    /* With latency=64, output should be 512-64 = 448 samples */
    size_t expected = n - (unsigned)cfg.trellis_lat;
    TEST_ASSERT_EQ(blocks[0].out_count, expected, "L SDM output count");
    TEST_ASSERT_EQ(blocks[1].out_count, expected, "R SDM output count");

    /* All output should be +/-1.0 */
    int valid = 1;
    for (size_t i = 0; i < blocks[0].out_count && valid; i++) {
        if (out_l[i] != 1.0f && out_l[i] != -1.0f) valid = 0;
        if (out_r[i] != 1.0f && out_r[i] != -1.0f) valid = 0;
    }
    TEST_ASSERT_TRUE(valid, "concurrent SDM output should be +/-1.0");

    free(in_l); free(in_r); free(out_l); free(out_r);
    engine_channel_free(&eng_l);
    engine_channel_free(&eng_r);
    threadpool_destroy(pool);
}

void test_threadpool_suite(void) {
    TEST_SUITE("Thread Pool");
    TEST_RUN(test_threadpool_create_destroy);
    TEST_RUN(test_threadpool_auto_count);
    TEST_RUN(test_threadpool_single_block);
    TEST_RUN(test_threadpool_stereo_concurrent);
    TEST_RUN(test_threadpool_passthrough_concurrent);
    TEST_RUN(test_threadpool_stress);
    TEST_RUN(test_threadpool_empty_wait);
    TEST_RUN(test_threadpool_sdm_concurrent);
}
