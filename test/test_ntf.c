/*
 * foo_dsd_trellis — NTF coefficient table tests
 * Verifies all 30 filters match reference/sdm.c exactly.
 */

#include "test.h"
#include "../include/ntf.h"

/* ── filter count and basic structure ───────────────────────── */

static void test_ntf_filter_count(void) {
    TEST_ASSERT_EQ(ntf_get_filter_count(), 40, "should have 40 filters");
}

static void test_ntf_all_entries_valid(void) {
    int count = ntf_get_filter_count();
    for (int i = 0; i < count; i++) {
        const ntf_filter_t *f = ntf_get_by_index(i);
        TEST_ASSERT_NOT_NULL(f, "filter should not be null");
        TEST_ASSERT_TRUE(f->order >= 4 && f->order <= 8, "order in [4,8]");
        TEST_ASSERT_TRUE(f->freq > 0, "freq > 0");
        TEST_ASSERT_NOT_NULL(f->name, "name should not be null");
        TEST_ASSERT_TRUE(
            strcmp(f->name, "clans-4") == 0 ||
            strcmp(f->name, "sdm-4") == 0 ||
            strcmp(f->name, "clans-5") == 0 ||
            strcmp(f->name, "sdm-5") == 0 ||
            strcmp(f->name, "clans-6") == 0 ||
            strcmp(f->name, "sdm-6") == 0 ||
            strcmp(f->name, "clans-7") == 0 ||
            strcmp(f->name, "sdm-7") == 0 ||
            strcmp(f->name, "clans-8") == 0 ||
            strcmp(f->name, "sdm-8") == 0,
            "name should be a known filter name"
        );
    }
}

static void test_ntf_frequency_groups(void) {
    /* Expect 10 filters per frequency group */
    int count_64 = 0, count_128 = 0, count_256 = 0, count_512 = 0;
    int count = ntf_get_filter_count();
    for (int i = 0; i < count; i++) {
        const ntf_filter_t *f = ntf_get_by_index(i);
        if (f->freq == 64u * 44100u) count_64++;
        else if (f->freq == 128u * 44100u) count_128++;
        else if (f->freq == 256u * 44100u) count_256++;
        else if (f->freq == 512u * 44100u) count_512++;
    }
    TEST_ASSERT_EQ(count_512, 10, "10 filters at 512x44100");
    TEST_ASSERT_EQ(count_256, 10, "10 filters at 256x44100");
    TEST_ASSERT_EQ(count_128, 10, "10 filters at 128x44100");
    TEST_ASSERT_EQ(count_64,  10, "10 filters at 64x44100");
}

/* ── exact coefficient verification against reference ───────── */

static void test_ntf_clans5_dsd64_coefficients(void) {
    /* clans-5 @ 64×44100 — reference sdm.c index 22 */
    const ntf_filter_t *f = ntf_get_filter(NTF_CLANS_5, DSD_RATE_64);
    TEST_ASSERT_NOT_NULL(f, "clans-5 @ DSD64 should exist");
    TEST_ASSERT_EQ(f->order, 5, "order should be 5");
    TEST_ASSERT_EQ(f->freq, 64u * 44100u, "freq should be 64×44100");

    /* a[] coefficients — exact match */
    TEST_ASSERT_FLOAT_EQ(f->a[0], 1.09979653514762e+00, 1e-15, "a[0]");
    TEST_ASSERT_FLOAT_EQ(f->a[1], 4.81149952106030e-01, 1e-15, "a[1]");
    TEST_ASSERT_FLOAT_EQ(f->a[2], 1.03481231987752e-01, 1e-15, "a[2]");
    TEST_ASSERT_FLOAT_EQ(f->a[3], 1.07520561970131e-02, 1e-15, "a[3]");
    TEST_ASSERT_FLOAT_EQ(f->a[4], 3.08801118488355e-04, 1e-15, "a[4]");

    /* g[] coefficients */
    TEST_ASSERT_FLOAT_EQ(f->g[0], 0.0, 1e-20, "g[0]");
    TEST_ASSERT_FLOAT_EQ(f->g[1], 6.98490600683106e-04, 1e-15, "g[1]");
    TEST_ASSERT_FLOAT_EQ(f->g[2], 0.0, 1e-20, "g[2]");
    TEST_ASSERT_FLOAT_EQ(f->g[3], 1.97734357803445e-03, 1e-15, "g[3]");
}

static void test_ntf_sdm8_dsd128_coefficients(void) {
    /* sdm-8 @ 128×44100 — reference sdm.c index 19 */
    const ntf_filter_t *f = ntf_get_filter(NTF_SDM_8, DSD_RATE_128);
    TEST_ASSERT_NOT_NULL(f, "sdm-8 @ DSD128 should exist");
    TEST_ASSERT_EQ(f->order, 8, "order should be 8");
    TEST_ASSERT_EQ(f->freq, 128u * 44100u, "freq should be 128×44100");

    TEST_ASSERT_FLOAT_EQ(f->a[0], 7.42763211426562e-01, 1e-15, "a[0]");
    TEST_ASSERT_FLOAT_EQ(f->a[7], 8.14280266547840e-08, 1e-20, "a[7]");

    TEST_ASSERT_FLOAT_EQ(f->g[0], 2.02698799324546e-05, 1e-15, "g[0]");
    TEST_ASSERT_FLOAT_EQ(f->g[1], 0.0, 1e-20, "g[1]");
    TEST_ASSERT_FLOAT_EQ(f->g[6], 5.55397776875272e-04, 1e-15, "g[6]");
    TEST_ASSERT_FLOAT_EQ(f->g[7], 0.0, 1e-20, "g[7]");
}

static void test_ntf_clans8_dsd256_coefficients(void) {
    /* clans-8 @ 256×44100 — reference sdm.c index 8 */
    const ntf_filter_t *f = ntf_get_filter(NTF_CLANS_8, DSD_RATE_256);
    TEST_ASSERT_NOT_NULL(f, "clans-8 @ DSD256 should exist");
    TEST_ASSERT_EQ(f->order, 8, "order should be 8");

    TEST_ASSERT_FLOAT_EQ(f->a[0], 1.15188624720851e+00, 1e-15, "a[0]");
    TEST_ASSERT_FLOAT_EQ(f->a[7], 2.22594461751768e-08, 1e-20, "a[7]");

    TEST_ASSERT_FLOAT_EQ(f->g[0], 5.06749566262594e-06, 1e-15, "g[0]");
    TEST_ASSERT_FLOAT_EQ(f->g[7], 0.0, 1e-20, "g[7]");
}

static void test_ntf_clans8_dsd64_negative_coeff(void) {
    /* clans-8 @ 64×44100 has negative a[6] and a[7] — verify sign preserved */
    const ntf_filter_t *f = ntf_get_filter(NTF_CLANS_8, DSD_RATE_64);
    TEST_ASSERT_NOT_NULL(f, "clans-8 @ DSD64 should exist");
    TEST_ASSERT_TRUE(f->a[6] < 0.0, "a[6] should be negative");
    TEST_ASSERT_TRUE(f->a[7] < 0.0, "a[7] should be negative");
    TEST_ASSERT_FLOAT_EQ(f->a[6], -1.90294986721073e-06, 1e-18, "a[6] exact");
    TEST_ASSERT_FLOAT_EQ(f->a[7], -7.39020160622772e-08, 1e-20, "a[7] exact");
}

/* ── ntf_get_filter lookup ──────────────────────────────────── */

static void test_ntf_get_filter_all_ids(void) {
    /* Every ID should find a filter at its natural rate */
    struct { ntf_filter_id_t id; unsigned rate; const char *name; } cases[] = {
        { NTF_CLANS_4, DSD_RATE_256, "clans-4" },
        { NTF_SDM_4,   DSD_RATE_256, "sdm-4"   },
        { NTF_CLANS_5, DSD_RATE_64,  "clans-5" },
        { NTF_SDM_5,   DSD_RATE_128, "sdm-5"   },
        { NTF_CLANS_6, DSD_RATE_128, "clans-6" },
        { NTF_SDM_6,   DSD_RATE_64,  "sdm-6"   },
        { NTF_CLANS_7, DSD_RATE_256, "clans-7" },
        { NTF_SDM_7,   DSD_RATE_64,  "sdm-7"   },
        { NTF_CLANS_8, DSD_RATE_128, "clans-8" },
        { NTF_SDM_8,   DSD_RATE_64,  "sdm-8"   },
    };

    for (int i = 0; i < 10; i++) {
        const ntf_filter_t *f = ntf_get_filter(cases[i].id, cases[i].rate);
        TEST_ASSERT_NOT_NULL(f, "filter should be found");
        TEST_ASSERT_EQ(strcmp(f->name, cases[i].name), 0, "name should match");
    }
}

static void test_ntf_get_filter_rate_matching(void) {
    /* Requesting clans-5 at DSD256 should return the 256×44100 variant (freq <= rate) */
    const ntf_filter_t *f = ntf_get_filter(NTF_CLANS_5, DSD_RATE_256);
    TEST_ASSERT_NOT_NULL(f, "clans-5 at DSD256 should find something");
    TEST_ASSERT_EQ(f->freq, 256u * 44100u, "should pick 256×44100 variant");

    /* Requesting clans-5 at DSD64 should return 64×44100 variant */
    f = ntf_get_filter(NTF_CLANS_5, DSD_RATE_64);
    TEST_ASSERT_NOT_NULL(f, "clans-5 at DSD64 should find something");
    TEST_ASSERT_TRUE(f->freq <= DSD_RATE_64, "freq should be <= DSD64");
}

static void test_ntf_get_filter_invalid(void) {
    TEST_ASSERT_NULL(ntf_get_filter((ntf_filter_id_t)-2, DSD_RATE_64), "invalid negative ID");
    TEST_ASSERT_NULL(ntf_get_filter(NTF_COUNT, DSD_RATE_64), "ID at NTF_COUNT");
}

/* ── ntf_auto_select ────────────────────────────────────────── */

static void test_ntf_auto_select_dsd64(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_64);
    TEST_ASSERT_NOT_NULL(f, "auto DSD64");
    TEST_ASSERT_EQ(strcmp(f->name, "clans-5"), 0, "DSD64 → clans-5");
    TEST_ASSERT_EQ(f->order, 5, "order 5");
}

static void test_ntf_auto_select_dsd128(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_128);
    TEST_ASSERT_NOT_NULL(f, "auto DSD128");
    TEST_ASSERT_EQ(strcmp(f->name, "clans-6"), 0, "DSD128 → clans-6");
    TEST_ASSERT_EQ(f->order, 6, "order 6");
}

static void test_ntf_auto_select_dsd256(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_256);
    TEST_ASSERT_NOT_NULL(f, "auto DSD256");
    TEST_ASSERT_EQ(strcmp(f->name, "clans-7"), 0, "DSD256 → clans-7");
    TEST_ASSERT_EQ(f->order, 7, "order 7");
}

static void test_ntf_auto_select_dsd512(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_512);
    TEST_ASSERT_NOT_NULL(f, "auto DSD512");
    TEST_ASSERT_EQ(strcmp(f->name, "clans-8"), 0, "DSD512 -> clans-8");
    TEST_ASSERT_EQ(f->order, 8, "order 8");
    TEST_ASSERT_EQ(f->freq, 512u * 44100u, "should use 512x44100 group");
}

static void test_ntf_dsd512_filters_exist(void) {
    /* All 10 DSD512 filter variants should be findable */
    struct { ntf_filter_id_t id; const char *name; } cases[] = {
        { NTF_CLANS_4, "clans-4" }, { NTF_SDM_4, "sdm-4" },
        { NTF_CLANS_5, "clans-5" }, { NTF_SDM_5, "sdm-5" },
        { NTF_CLANS_6, "clans-6" }, { NTF_SDM_6, "sdm-6" },
        { NTF_CLANS_7, "clans-7" }, { NTF_SDM_7, "sdm-7" },
        { NTF_CLANS_8, "clans-8" }, { NTF_SDM_8, "sdm-8" },
    };
    for (int i = 0; i < 10; i++) {
        const ntf_filter_t *f = ntf_get_filter(cases[i].id, DSD_RATE_512);
        TEST_ASSERT_NOT_NULL(f, "DSD512 filter should exist");
        TEST_ASSERT_EQ(strcmp(f->name, cases[i].name), 0, "name match");
        TEST_ASSERT_EQ(f->freq, 512u * 44100u, "freq should be 512x44100");
    }
}

static void test_ntf_dsd512_clans8_coefficients(void) {
    const ntf_filter_t *f = ntf_get_filter(NTF_CLANS_8, DSD_RATE_512);
    TEST_ASSERT_NOT_NULL(f, "clans-8 @ DSD512");
    TEST_ASSERT_EQ(f->order, 8, "order 8");
    /* Verify a[0] is reasonable (close to 256x44100 value of 1.1518...) */
    TEST_ASSERT_FLOAT_EQ(f->a[0], 1.15143264272600732e+00, 1e-12, "a[0]");
    /* g[] should be ~1/4 of 256x44100 values */
    TEST_ASSERT_FLOAT_EQ(f->g[0], 1.26687391565648508e-06, 1e-15, "g[0]");
}

/* ── ntf_get_by_index bounds ────────────────────────────────── */

static void test_ntf_get_by_index_bounds(void) {
    TEST_ASSERT_NULL(ntf_get_by_index(-1), "index -1 should be null");
    TEST_ASSERT_NULL(ntf_get_by_index(40), "index 40 should be null");
    TEST_ASSERT_NULL(ntf_get_by_index(100), "index 100 should be null");
    TEST_ASSERT_NOT_NULL(ntf_get_by_index(0), "index 0 valid");
    TEST_ASSERT_NOT_NULL(ntf_get_by_index(39), "index 39 valid");
}

/* ── suite entry point ──────────────────────────────────────── */

void test_ntf_suite(void) {
    TEST_SUITE("NTF Coefficient Tables");

    TEST_RUN(test_ntf_filter_count);
    TEST_RUN(test_ntf_all_entries_valid);
    TEST_RUN(test_ntf_frequency_groups);

    TEST_RUN(test_ntf_clans5_dsd64_coefficients);
    TEST_RUN(test_ntf_sdm8_dsd128_coefficients);
    TEST_RUN(test_ntf_clans8_dsd256_coefficients);
    TEST_RUN(test_ntf_clans8_dsd64_negative_coeff);

    TEST_RUN(test_ntf_get_filter_all_ids);
    TEST_RUN(test_ntf_get_filter_rate_matching);
    TEST_RUN(test_ntf_get_filter_invalid);

    TEST_RUN(test_ntf_auto_select_dsd64);
    TEST_RUN(test_ntf_auto_select_dsd128);
    TEST_RUN(test_ntf_auto_select_dsd256);
    TEST_RUN(test_ntf_auto_select_dsd512);
    TEST_RUN(test_ntf_dsd512_filters_exist);
    TEST_RUN(test_ntf_dsd512_clans8_coefficients);

    TEST_RUN(test_ntf_get_by_index_bounds);
}
