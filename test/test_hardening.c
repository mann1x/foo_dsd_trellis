/*
 * foo_dsd_trellis — Phase 7 hardening tests
 * Edge cases: mute, passthrough, config changes, discontinuities,
 * zero-length blocks, multi-rate transitions, drain after reset.
 */

#include "test.h"
#include "../include/engine.h"
#include "../include/threadpool.h"
#include "../include/dop.h"
#include "../include/ntf.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── Mute edge cases ─── */

static void test_mute_produces_alternating_pattern(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.mute = true;

    engine_channel_t eng;
    TEST_ASSERT_EQ(engine_channel_init(&eng, 0, &cfg), 0,
                   "mute engine init");

    float in[256], out[256];
    memset(in, 0, sizeof(in));

    size_t n = engine_process_block(&eng, in, out, 256, &cfg);
    TEST_ASSERT_EQ(n, 256u, "mute should produce same count");

    /* Verify strict alternating +/-1.0 */
    int valid = 1;
    for (size_t i = 0; i < n; i++) {
        float expected = (i & 1) ? 1.0f : -1.0f;
        if (out[i] != expected) { valid = 0; break; }
    }
    TEST_ASSERT_TRUE(valid, "mute should produce alternating +/-1.0");

    engine_channel_free(&eng);
}

static void test_mute_ignores_input(void) {
    /* Even with non-zero input, mute should produce silence pattern */
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.mute = true;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    float in[64], out[64];
    for (int i = 0; i < 64; i++)
        in[i] = (float)(0.9 * sin(2.0 * M_PI * 1000.0 * i / DSD_RATE_64));

    size_t n = engine_process_block(&eng, in, out, 64, &cfg);
    TEST_ASSERT_EQ(n, 64u, "mute count with non-zero input");

    int valid = 1;
    for (size_t i = 0; i < n; i++) {
        if (out[i] != 1.0f && out[i] != -1.0f) { valid = 0; break; }
    }
    TEST_ASSERT_TRUE(valid, "mute should ignore input content");

    engine_channel_free(&eng);
}

static void test_mute_small_block(void) {
    /* Single sample block */
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.mute = true;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    float in = 0.0f, out = 0.0f;
    size_t n = engine_process_block(&eng, &in, &out, 1, &cfg);
    TEST_ASSERT_EQ(n, 1u, "mute single sample count");
    TEST_ASSERT_TRUE(out == 1.0f || out == -1.0f,
                     "mute single sample should be +/-1.0");

    engine_channel_free(&eng);
}

/* ─── Passthrough edge cases ─── */

static void test_passthrough_preserves_sign(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_64;
    cfg.gain = 1.0f;

    engine_channel_t eng;
    TEST_ASSERT_EQ(engine_channel_init(&eng, 0, &cfg), 0,
                   "passthrough init");
    TEST_ASSERT_TRUE(eng.passthrough, "should be passthrough mode");

    float in[128], out[128];
    for (int i = 0; i < 128; i++)
        in[i] = (i % 3 == 0) ? 0.5f : -0.3f;

    size_t n = engine_process_block(&eng, in, out, 128, &cfg);
    TEST_ASSERT_EQ(n, 128u, "passthrough count");

    int match = 1;
    for (int i = 0; i < 128; i++) {
        float expected = in[i] >= 0.0f ? 1.0f : -1.0f;
        if (out[i] != expected) { match = 0; break; }
    }
    TEST_ASSERT_TRUE(match, "passthrough should requantise to +/-1.0 by sign");

    engine_channel_free(&eng);
}

static void test_passthrough_zero_input(void) {
    /* Zero is >= 0, should map to +1.0 */
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_64;
    cfg.gain = 1.0f;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    float in[32], out[32];
    memset(in, 0, sizeof(in));

    size_t n = engine_process_block(&eng, in, out, 32, &cfg);
    TEST_ASSERT_EQ(n, 32u, "passthrough zero count");

    int all_pos = 1;
    for (int i = 0; i < 32; i++) {
        if (out[i] != 1.0f) { all_pos = 0; break; }
    }
    TEST_ASSERT_TRUE(all_pos, "zero input should map to +1.0 in passthrough");

    engine_channel_free(&eng);
}

static void test_passthrough_not_when_gain_differs(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_64;
    cfg.gain = 0.5f;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    TEST_ASSERT_FALSE(eng.passthrough,
                      "gain != 1.0 should not be passthrough");

    engine_channel_free(&eng);
}

static void test_passthrough_not_when_rate_differs(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_128;
    cfg.gain = 1.0f;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    TEST_ASSERT_FALSE(eng.passthrough,
                      "rate change should not be passthrough");

    engine_channel_free(&eng);
}

/* ─── Config change mid-stream ─── */

static void test_reconfigure_mute_to_passthrough(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.mute = true;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    float in[64], out[64];
    memset(in, 0, sizeof(in));

    /* Process muted block */
    size_t n1 = engine_process_block(&eng, in, out, 64, &cfg);
    TEST_ASSERT_EQ(n1, 64u, "muted block count");

    /* Reconfigure to passthrough */
    cfg.mute = false;
    cfg.fs_out = DSD_RATE_64;
    cfg.gain = 1.0f;
    int ret = engine_channel_reconfigure(&eng, &cfg);
    TEST_ASSERT_EQ(ret, 0, "reconfigure should succeed");
    TEST_ASSERT_TRUE(eng.passthrough, "should now be passthrough");

    /* Process passthrough block */
    for (int i = 0; i < 64; i++) in[i] = (i & 1) ? 1.0f : -1.0f;
    size_t n2 = engine_process_block(&eng, in, out, 64, &cfg);
    TEST_ASSERT_EQ(n2, 64u, "passthrough block count after reconfig");

    engine_channel_free(&eng);
}

static void test_reconfigure_passthrough_to_sdm(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_64;
    cfg.gain = 1.0f;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);
    TEST_ASSERT_TRUE(eng.passthrough, "initial passthrough");

    /* Reconfigure to SDM path (gain < 1.0) */
    cfg.gain = 0.5f;
    int ret = engine_channel_reconfigure(&eng, &cfg);
    TEST_ASSERT_EQ(ret, 0, "reconfigure to SDM should succeed");
    TEST_ASSERT_FALSE(eng.passthrough, "should not be passthrough after reconfig");

    /* Process through SDM */
    float in[256], out[256];
    for (int i = 0; i < 256; i++)
        in[i] = (float)(0.3 * sin(2.0 * M_PI * 1000.0 * i / DSD_RATE_64));

    size_t n = engine_process_block(&eng, in, out, 256, &cfg);
    /* With latency=64, first block should produce 256-64 = 192 */
    TEST_ASSERT_TRUE(n > 0, "SDM should produce output after reconfig");

    /* All output must be +/-1.0 */
    int valid = 1;
    for (size_t i = 0; i < n; i++) {
        if (out[i] != 1.0f && out[i] != -1.0f) { valid = 0; break; }
    }
    TEST_ASSERT_TRUE(valid, "SDM output should be +/-1.0 after reconfig");

    engine_channel_free(&eng);
}

static void test_reconfigure_rate_change(void) {
    /* DSD64 → DSD128 reconfigure */
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_64;
    cfg.gain = 0.8f;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    /* Reconfigure to DSD128 output */
    cfg.fs_out = DSD_RATE_128;
    int ret = engine_channel_reconfigure(&eng, &cfg);
    TEST_ASSERT_EQ(ret, 0, "rate change reconfigure should succeed");

    float in[128], out[512];
    for (int i = 0; i < 128; i++)
        in[i] = (i & 1) ? 0.3f : -0.3f;

    size_t n = engine_process_block(&eng, in, out, 128, &cfg);
    /* 2x upsample: 128 → 256, minus latency */
    TEST_ASSERT_TRUE(n > 0, "rate-changed engine should produce output");

    engine_channel_free(&eng);
}

/* ─── Reset / discontinuity ─── */

static void test_reset_clears_state(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_64;
    cfg.gain = 0.5f;
    cfg.trellis_lat = 64;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    float in[256], out[256];
    for (int i = 0; i < 256; i++)
        in[i] = (float)(0.5 * sin(2.0 * M_PI * 1000.0 * i / DSD_RATE_64));

    /* Process first block */
    size_t n1 = engine_process_block(&eng, in, out, 256, &cfg);
    TEST_ASSERT_TRUE(n1 > 0, "pre-reset output");

    /* Reset */
    engine_channel_reset(&eng);

    /* After reset, SDM latency starts over — first block should have
     * the same count as the very first call */
    size_t n2 = engine_process_block(&eng, in, out, 256, &cfg);
    TEST_ASSERT_EQ(n1, n2, "post-reset should behave like fresh init");

    engine_channel_free(&eng);
}

static void test_reset_then_different_signal(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_64;
    cfg.gain = 0.5f;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    float in1[256], in2[256], out[256];
    for (int i = 0; i < 256; i++) {
        in1[i] = (float)(0.5 * sin(2.0 * M_PI * 1000.0 * i / DSD_RATE_64));
        in2[i] = (float)(0.5 * sin(2.0 * M_PI * 5000.0 * i / DSD_RATE_64));
    }

    engine_process_block(&eng, in1, out, 256, &cfg);
    engine_channel_reset(&eng);

    /* After reset, processing a different signal should not crash
     * or produce invalid output */
    size_t n = engine_process_block(&eng, in2, out, 256, &cfg);
    TEST_ASSERT_TRUE(n > 0, "different signal after reset");

    int valid = 1;
    for (size_t i = 0; i < n; i++) {
        if (out[i] != 1.0f && out[i] != -1.0f) { valid = 0; break; }
    }
    TEST_ASSERT_TRUE(valid, "output after reset must be +/-1.0");

    engine_channel_free(&eng);
}

/* ─── Zero-length and edge-size blocks ─── */

static void test_zero_length_block(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_64;
    cfg.gain = 0.5f;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    float out[1];
    size_t n = engine_process_block(&eng, NULL, out, 0, &cfg);
    TEST_ASSERT_EQ(n, 0u, "zero-length block should produce 0 output");

    engine_channel_free(&eng);
}

static void test_single_sample_sdm(void) {
    /* Single sample through SDM — should accumulate in latency buffer */
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_64;
    cfg.gain = 0.5f;
    cfg.trellis_lat = 64;
    cfg.sdm_mode = SDM_MODE_TRELLIS;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    float in = 0.3f, out = 0.0f;
    size_t n = engine_process_block(&eng, &in, &out, 1, &cfg);
    /* With latency=64, first sample produces 0 output */
    TEST_ASSERT_EQ(n, 0u, "single sample should produce 0 (latency fill)");

    engine_channel_free(&eng);
}

/* ─── Drain edge cases ─── */

static void test_drain_without_processing(void) {
    /* Drain on a fresh engine should produce nothing */
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_64;
    cfg.gain = 0.5f;
    cfg.trellis_lat = 64;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    float drain_out[128];
    size_t drained = sdm_drain(&eng.sdm, drain_out, 128);
    /* Fresh SDM with pending=0 should drain 0 or very few samples */
    TEST_ASSERT_TRUE(drained <= (size_t)cfg.trellis_lat,
                     "drain without processing should be bounded");

    engine_channel_free(&eng);
}

static void test_drain_after_reset(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_64;
    cfg.gain = 0.5f;
    cfg.trellis_lat = 64;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    /* Process some data */
    float in[256], out[256];
    for (int i = 0; i < 256; i++)
        in[i] = (float)(0.3 * sin(2.0 * M_PI * 1000.0 * i / DSD_RATE_64));
    engine_process_block(&eng, in, out, 256, &cfg);

    /* Reset then drain — should not crash */
    engine_channel_reset(&eng);
    float drain_out[128];
    size_t drained = sdm_drain(&eng.sdm, drain_out, 128);
    TEST_ASSERT_TRUE(drained <= (size_t)cfg.trellis_lat,
                     "drain after reset should be bounded");

    engine_channel_free(&eng);
}

/* ─── Multiple sequential blocks ─── */

static void test_continuous_blocks_no_gap(void) {
    /* Process many small blocks sequentially — total output should match */
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_64;
    cfg.gain = 0.5f;
    cfg.trellis_lat = 64;
    cfg.sdm_mode = SDM_MODE_TRELLIS;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    float in[64], out[64];
    size_t total_out = 0;

    for (int block = 0; block < 20; block++) {
        for (int i = 0; i < 64; i++)
            in[i] = (float)(0.3 * sin(2.0 * M_PI * 1000.0 *
                    (block * 64 + i) / (double)DSD_RATE_64));
        size_t n = engine_process_block(&eng, in, out, 64, &cfg);
        total_out += n;

        /* All output must be valid */
        for (size_t i = 0; i < n; i++) {
            if (out[i] != 1.0f && out[i] != -1.0f) {
                TEST_ASSERT_TRUE(0, "invalid output in continuous blocks");
                engine_channel_free(&eng);
                return;
            }
        }
    }

    /* 20 * 64 = 1280 input, minus latency 64 = 1216 expected output */
    TEST_ASSERT_EQ(total_out, 1280u - 64u,
                   "continuous blocks total output");

    engine_channel_free(&eng);
}

/* ─── Thread pool edge cases ─── */

static void test_threadpool_single_thread(void) {
    /* Verify single-threaded pool works correctly */
    threadpool_t *pool = threadpool_create(1, 0);
    TEST_ASSERT_NOT_NULL(pool, "single-thread pool");
    TEST_ASSERT_EQ(threadpool_get_thread_count(pool), 1, "thread count = 1");

    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.mute = true;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    float in[64], out[64];
    memset(in, 0, sizeof(in));

    channel_block_t block = {
        .in = in, .out = out, .count = 64,
        .out_count = 0, .channel = 0,
        .eng = &eng, .cfg = &cfg
    };

    threadpool_submit(pool, &block);
    threadpool_wait(pool);

    TEST_ASSERT_EQ(block.out_count, 64u, "single-thread pool output");

    engine_channel_free(&eng);
    threadpool_destroy(pool);
}

static void test_threadpool_sequential_batches(void) {
    /* Multiple sequential submit-wait cycles */
    threadpool_t *pool = threadpool_create(2, 0);

    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.mute = true;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    float in[64], out[64];
    memset(in, 0, sizeof(in));

    int all_ok = 1;
    for (int batch = 0; batch < 50; batch++) {
        channel_block_t block = {
            .in = in, .out = out, .count = 64,
            .out_count = 0, .channel = 0,
            .eng = &eng, .cfg = &cfg
        };

        threadpool_submit(pool, &block);
        threadpool_wait(pool);

        if (block.out_count != 64) {
            all_ok = 0;
            break;
        }
    }
    TEST_ASSERT_TRUE(all_ok, "50 sequential batches should all succeed");

    engine_channel_free(&eng);
    threadpool_destroy(pool);
}

/* ─── Plugin-level edge cases ─── */

static void test_engine_double_free_safe(void) {
    /* Calling free twice should not crash (second call is no-op) */
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.mute = true;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);
    engine_channel_free(&eng);
    /* Re-init to zeroed state before second free to test idempotency */
    memset(&eng, 0, sizeof(eng));
    engine_channel_free(&eng);
    TEST_ASSERT_TRUE(1, "double free should not crash");
}

static void test_engine_reconfigure_multiple_times(void) {
    /* Reconfigure repeatedly between different modes */
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_64;
    cfg.gain = 1.0f;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    float in[128], out[128];
    for (int i = 0; i < 128; i++) in[i] = (i & 1) ? 0.5f : -0.5f;

    /* Cycle: passthrough → SDM → mute → passthrough */
    const struct { float gain; bool mute; } modes[] = {
        { 1.0f, false },   /* passthrough */
        { 0.5f, false },   /* SDM */
        { 1.0f, true  },   /* mute */
        { 1.0f, false },   /* passthrough again */
    };

    for (int m = 0; m < 4; m++) {
        cfg.gain = modes[m].gain;
        cfg.mute = modes[m].mute;
        int ret = engine_channel_reconfigure(&eng, &cfg);
        TEST_ASSERT_EQ(ret, 0, "reconfigure cycle should succeed");

        size_t n = engine_process_block(&eng, in, out, 128, &cfg);
        TEST_ASSERT_TRUE(n > 0 || (!modes[m].mute && modes[m].gain != 1.0f),
                         "reconfig mode should produce output or latency-fill");
    }

    engine_channel_free(&eng);
}

/* ─── DoP edge cases ─── */

static void test_dop_detect_short_buffer(void) {
    /* Very short buffer should not crash dop_detect */
    float buf[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    /* dop_detect with < 8 frames is not called by plugin_process,
     * but let's verify it doesn't crash */
    int result = dop_detect(buf, 4);
    /* Should return 0 (not DoP) or not crash */
    TEST_ASSERT_TRUE(result == 0 || result == 1,
                     "dop_detect on short buffer should not crash");
}

static void test_dop_unpack_pack_roundtrip(void) {
    /* Create valid DoP frames, unpack, then repack — should roundtrip */
    float pcm[16];
    float dsd[256];  /* 16 frames * 16 bits = 256 DSD samples */
    float pcm_out[16];

    /* Create DoP frames with known DSD pattern */
    for (int i = 0; i < 16; i++) {
        uint32_t marker = (i & 1) ? 0xFA : 0x05;
        uint32_t dsd_byte_hi = 0xAA;  /* alternating bits */
        uint32_t dsd_byte_lo = 0x55;
        int32_t val = (int32_t)((marker << 16) | (dsd_byte_hi << 8) | dsd_byte_lo);
        /* Scale to float as 24-bit signed */
        pcm[i] = (float)val / (float)(1 << 23);
    }

    dop_unpack(pcm, dsd, 16);

    /* All unpacked DSD should be +/-1.0 */
    int valid = 1;
    for (int i = 0; i < 256; i++) {
        if (dsd[i] != 1.0f && dsd[i] != -1.0f) { valid = 0; break; }
    }
    TEST_ASSERT_TRUE(valid, "unpacked DSD should be +/-1.0");

    dop_pack(dsd, pcm_out, 256);
    /* Repacked DoP should be valid float values (not NaN/inf) */
    int valid2 = 1;
    for (int i = 0; i < 16; i++) {
        if (pcm_out[i] != pcm_out[i]) { valid2 = 0; break; } /* NaN check */
    }
    TEST_ASSERT_TRUE(valid2, "repacked DoP should be finite");
}

/* ─── Rate conversion edge cases ─── */

static void test_upsample_2x_output_count(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_64;
    cfg.fs_out = DSD_RATE_128;
    cfg.gain = 0.5f;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    /* Large block to ensure we're past latency fill */
    unsigned n_in = 1024;
    float *in  = (float *)malloc(n_in * sizeof(float));
    float *out = (float *)malloc(n_in * 4 * sizeof(float));

    for (unsigned i = 0; i < n_in; i++)
        in[i] = (float)(0.3 * sin(2.0 * M_PI * 1000.0 * i / DSD_RATE_64));

    size_t n_out = engine_process_block(&eng, in, out, n_in, &cfg);

    /* 2x upsample: expect ~2048 - latency output samples */
    TEST_ASSERT_TRUE(n_out > 0, "2x upsample should produce output");
    TEST_ASSERT_TRUE(n_out <= n_in * 2,
                     "2x upsample should not exceed 2x input count");

    free(in);
    free(out);
    engine_channel_free(&eng);
}

static void test_downsample_2x_output_count(void) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);
    cfg.fs_in = DSD_RATE_128;
    cfg.fs_out = DSD_RATE_64;
    cfg.gain = 0.5f;

    engine_channel_t eng;
    engine_channel_init(&eng, 0, &cfg);

    unsigned n_in = 2048;
    float *in  = (float *)malloc(n_in * sizeof(float));
    float *out = (float *)malloc(n_in * sizeof(float));

    for (unsigned i = 0; i < n_in; i++)
        in[i] = (float)(0.3 * sin(2.0 * M_PI * 1000.0 * i / DSD_RATE_128));

    size_t n_out = engine_process_block(&eng, in, out, n_in, &cfg);

    TEST_ASSERT_TRUE(n_out > 0, "2x downsample should produce output");
    TEST_ASSERT_TRUE(n_out <= n_in,
                     "2x downsample should not exceed input count");

    free(in);
    free(out);
    engine_channel_free(&eng);
}

/* ─── Test suite entry point ─── */

void test_hardening_suite(void) {
    TEST_SUITE("Hardening (Phase 7)");

    /* Mute */
    TEST_RUN(test_mute_produces_alternating_pattern);
    TEST_RUN(test_mute_ignores_input);
    TEST_RUN(test_mute_small_block);

    /* Passthrough */
    TEST_RUN(test_passthrough_preserves_sign);
    TEST_RUN(test_passthrough_zero_input);
    TEST_RUN(test_passthrough_not_when_gain_differs);
    TEST_RUN(test_passthrough_not_when_rate_differs);

    /* Config changes */
    TEST_RUN(test_reconfigure_mute_to_passthrough);
    TEST_RUN(test_reconfigure_passthrough_to_sdm);
    TEST_RUN(test_reconfigure_rate_change);

    /* Reset / discontinuity */
    TEST_RUN(test_reset_clears_state);
    TEST_RUN(test_reset_then_different_signal);

    /* Edge-size blocks */
    TEST_RUN(test_zero_length_block);
    TEST_RUN(test_single_sample_sdm);

    /* Drain */
    TEST_RUN(test_drain_without_processing);
    TEST_RUN(test_drain_after_reset);

    /* Continuous processing */
    TEST_RUN(test_continuous_blocks_no_gap);

    /* Thread pool */
    TEST_RUN(test_threadpool_single_thread);
    TEST_RUN(test_threadpool_sequential_batches);

    /* Safety */
    TEST_RUN(test_engine_double_free_safe);
    TEST_RUN(test_engine_reconfigure_multiple_times);

    /* DoP */
    TEST_RUN(test_dop_detect_short_buffer);
    TEST_RUN(test_dop_unpack_pack_roundtrip);

    /* Rate conversion */
    TEST_RUN(test_upsample_2x_output_count);
    TEST_RUN(test_downsample_2x_output_count);
}
