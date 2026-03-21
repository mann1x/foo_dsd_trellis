/*
 * foo_dsd_trellis — Audio quality measurement tests
 *
 * Tests the A-weighting curve, multitone, noise modulation, and NMR
 * measurement functions used by the "Test Quality" button.
 */

#include "test.h"
#include "../include/sinad_measure.h"
#include "../include/dsd_types.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── A-weighting curve tests ─── */

static void test_a_weight_1khz(void) {
    /* A-weighting is 0 dB at 1 kHz by definition */
    double w = a_weight_factor(1000.0);
    double db = 10.0 * log10(w);
    printf("    A-weight @ 1kHz: %.2f dB (expected 0.0)\n", db);
    TEST_ASSERT_TRUE(fabs(db) < 0.5, "A-weight at 1kHz should be ~0 dB");
}

static void test_a_weight_100hz(void) {
    /* A-weighting at 100 Hz is about -19.1 dB */
    double w = a_weight_factor(100.0);
    double db = 10.0 * log10(w);
    printf("    A-weight @ 100Hz: %.1f dB (expected ~-19.1)\n", db);
    TEST_ASSERT_TRUE(db < -15.0 && db > -25.0, "A-weight at 100Hz should be ~-19 dB");
}

static void test_a_weight_4khz(void) {
    /* A-weighting peaks slightly above 0 dB around 3-4 kHz (~+1.0 dB) */
    double w = a_weight_factor(4000.0);
    double db = 10.0 * log10(w);
    printf("    A-weight @ 4kHz: %.1f dB (expected ~+1.0)\n", db);
    TEST_ASSERT_TRUE(db > -1.0 && db < 3.0, "A-weight at 4kHz should be ~+1 dB");
}

static void test_a_weight_10khz(void) {
    /* A-weighting at 10 kHz is about -2.5 dB */
    double w = a_weight_factor(10000.0);
    double db = 10.0 * log10(w);
    printf("    A-weight @ 10kHz: %.1f dB (expected ~-2.5)\n", db);
    TEST_ASSERT_TRUE(db > -6.0 && db < 2.0, "A-weight at 10kHz should be ~-2.5 dB");
}

static void test_a_weight_monotonic_low(void) {
    /* A-weight should increase from 20 Hz to ~3 kHz */
    double prev = a_weight_factor(20.0);
    for (double f = 50.0; f <= 3000.0; f *= 1.5) {
        double w = a_weight_factor(f);
        TEST_ASSERT_TRUE(w >= prev * 0.99, "A-weight should increase 20Hz→3kHz");
        prev = w;
    }
}

/* ─── Full quality measurement tests ─── */

static void test_quality_dsd64(void) {
    sinad_result_t r;
    sinad_measure(DSD_RATE_64, -1, 2, 4, 32, 1, 0.708f, &r);
    printf("    DSD64: SINAD=%.1f A-wtd=%.1f MT=%.1f NMod=%.1f NMR=%.1f\n",
           r.sinad_theoretical, r.sinad_awtd_theo,
           r.multitone_sinad_db, r.noise_mod_db, r.nmr_db);
    TEST_ASSERT_TRUE(r.ok, "DSD64 measurement should succeed");
    TEST_ASSERT_TRUE(r.sinad_theoretical > 80.0, "DSD64 SINAD > 80 dB");
    TEST_ASSERT_TRUE(r.sinad_awtd_theo >= r.sinad_theoretical - 1.0,
                     "A-weighted >= flat (noise at LF/HF downweighted)");
    TEST_ASSERT_TRUE(r.multitone_sinad_db > 50.0, "DSD64 multitone > 50 dB");
    TEST_ASSERT_TRUE(r.noise_mod_db < 15.0, "DSD64 noise mod < 15 dB");
    TEST_ASSERT_TRUE(r.nmr_db < 0.0, "DSD64 NMR < 0 dB (noise below mask)");
}

static void test_quality_dsd128(void) {
    sinad_result_t r;
    sinad_measure(DSD_RATE_128, -1, 2, 4, 128, 1, 0.708f, &r);
    printf("    DSD128: SINAD=%.1f A-wtd=%.1f MT=%.1f NMod=%.1f NMR=%.1f\n",
           r.sinad_theoretical, r.sinad_awtd_theo,
           r.multitone_sinad_db, r.noise_mod_db, r.nmr_db);
    TEST_ASSERT_TRUE(r.ok, "DSD128 measurement should succeed");
    TEST_ASSERT_TRUE(r.sinad_theoretical > 110.0, "DSD128 SINAD > 110 dB");
    TEST_ASSERT_TRUE(r.sinad_awtd_theo > r.sinad_theoretical, "DSD128 A-wtd > flat");
    TEST_ASSERT_TRUE(r.multitone_sinad_db > 80.0, "DSD128 multitone > 80 dB");
    TEST_ASSERT_TRUE(r.nmr_db < -50.0, "DSD128 NMR < -50 dB");
}

static void test_quality_dsd256(void) {
    sinad_result_t r;
    sinad_measure(DSD_RATE_256, -1, 2, 4, 128, 1, 0.708f, &r);
    printf("    DSD256: SINAD=%.1f A-wtd=%.1f MT=%.1f NMod=%.1f NMR=%.1f\n",
           r.sinad_theoretical, r.sinad_awtd_theo,
           r.multitone_sinad_db, r.noise_mod_db, r.nmr_db);
    TEST_ASSERT_TRUE(r.ok, "DSD256 measurement should succeed");
    TEST_ASSERT_TRUE(r.sinad_theoretical > 100.0, "DSD256 SINAD > 100 dB");
    TEST_ASSERT_TRUE(r.multitone_sinad_db > 100.0, "DSD256 multitone > 100 dB");
    TEST_ASSERT_TRUE(r.nmr_db < -80.0, "DSD256 NMR < -80 dB");
}

/* ─── DSD→PCM quality test ─── */

static void test_quality_dsd64_to_pcm44(void) {
    sinad_result_t r;
    sinad_measure_dsd_to_pcm(DSD_RATE_64, 44100, &r);
    printf("    DSD64→PCM44.1k: SINAD=%.1f A-wtd=%.1f MT=%.1f NMod=%.1f\n",
           r.sinad_theoretical, r.sinad_awtd_theo,
           r.multitone_sinad_db, r.noise_mod_db);
    TEST_ASSERT_TRUE(r.ok, "DSD64→PCM44.1k should succeed");
    TEST_ASSERT_TRUE(r.sinad_theoretical > 90.0, "DSD64→PCM44.1k SINAD > 90 dB");
    TEST_ASSERT_TRUE(r.multitone_sinad_db > 50.0, "DSD64→PCM44.1k multitone > 50 dB");
}

/* ─── PCM→PCM quality test ─── */

static void test_quality_pcm_44_to_48(void) {
    sinad_result_t r;
    sinad_measure_pcm_to_pcm(44100, 48000, RESAMPLE_AUTO, SOXR_QUALITY_HQ, &r);
    printf("    PCM 44.1k→48k: SINAD=%.1f A-wtd=%.1f MT=%.1f\n",
           r.sinad_theoretical, r.sinad_awtd_theo, r.multitone_sinad_db);
    TEST_ASSERT_TRUE(r.ok, "PCM 44.1k→48k should succeed");
    TEST_ASSERT_TRUE(r.sinad_theoretical > 50.0, "PCM 44.1k→48k SINAD > 50 dB");
}

/* ─── DSD/48 quality test ─── */

static void test_quality_dsd64_48(void) {
    sinad_result_t r;
    sinad_measure(DSD48_RATE_64, -1, 2, 4, 32, 1, 0.708f, &r);
    printf("    DSD64/48: SINAD=%.1f A-wtd=%.1f MT=%.1f NMod=%.1f NMR=%.1f\n",
           r.sinad_theoretical, r.sinad_awtd_theo,
           r.multitone_sinad_db, r.noise_mod_db, r.nmr_db);
    TEST_ASSERT_TRUE(r.ok, "DSD64/48 measurement should succeed");
    TEST_ASSERT_TRUE(r.sinad_theoretical > 80.0, "DSD64/48 SINAD > 80 dB");
    TEST_ASSERT_TRUE(r.multitone_sinad_db > 50.0, "DSD64/48 multitone > 50 dB");
}

/* ─── Full matrix test ─── */

static void test_quality_matrix(void) {
    /* Test all path types with representative rate pairs */
    typedef struct { const char *name; uint32_t in; uint32_t out; int type; } qtest_t;
    /* type: 0=DSD→DSD, 1=DSD→PCM, 2=PCM→DSD, 3=PCM→PCM */
    static const qtest_t tests[] = {
        /* DSD→DSD same-rate */
        { "DSD128 re-encode",       DSD_RATE_128, DSD_RATE_128, 0 },
        { "DSD256 re-encode",       DSD_RATE_256, DSD_RATE_256, 0 },
        /* DSD→PCM same-family */
        { "DSD64->PCM44.1k",        DSD_RATE_64,  44100,    1 },
        { "DSD128->PCM88.2k",       DSD_RATE_128, 88200,    1 },
        { "DSD256->PCM176.4k",      DSD_RATE_256, 176400,   1 },
        /* DSD/48→PCM same-family */
        { "DSD64/48->PCM48k",       DSD48_RATE_64, 48000,   1 },
        /* PCM→PCM same-family */
        { "44.1k->88.2k FIR",       44100, 88200,   3 },
        { "96k->48k FIR",           96000, 48000,   3 },
        /* PCM→PCM cross-family */
        { "44.1k->48k polyphase",   44100, 48000,   3 },
        { "96k->88.2k polyphase",   96000, 88200,   3 },
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int ok = 0;
    printf("    %-28s  %7s  %7s  %7s  %7s\n", "Path", "SINAD", "A-wtd", "MT", "NMod");
    printf("    %-28s  %7s  %7s  %7s  %7s\n", "---", "---", "---", "---", "---");
    for (int i = 0; i < n; i++) {
        sinad_result_t r;
        memset(&r, 0, sizeof(r));
        if (tests[i].type == 0) {
            sinad_measure(tests[i].out, -1, 2, 4, 128, 1, 0.708f, &r);
        } else if (tests[i].type == 1) {
            sinad_measure_dsd_to_pcm(tests[i].in, tests[i].out, &r);
        } else if (tests[i].type == 3) {
            sinad_measure_pcm_to_pcm(tests[i].in, tests[i].out,
                                      RESAMPLE_AUTO, SOXR_QUALITY_HQ, &r);
        }
        printf("    %-28s  %6.1f  %6.1f  %6.1f  %6.1f  %s\n",
               tests[i].name,
               r.sinad_theoretical, r.sinad_awtd_theo,
               r.multitone_sinad_db, r.noise_mod_db,
               r.ok ? "OK" : "FAIL");
        if (r.ok) {
            TEST_ASSERT_TRUE(r.sinad_theoretical > 30.0, tests[i].name);
            if (r.multitone_sinad_db != 0.0)
                TEST_ASSERT_TRUE(r.multitone_sinad_db > 30.0, tests[i].name);
            ok++;
        }
    }
    printf("    %d/%d paths measured successfully\n", ok, n);
    TEST_ASSERT_EQ(ok, n, "all matrix paths should succeed");
}

void test_quality_suite(void) {
    TEST_SUITE("Quality Metrics");
    TEST_RUN(test_a_weight_1khz);
    TEST_RUN(test_a_weight_100hz);
    TEST_RUN(test_a_weight_4khz);
    TEST_RUN(test_a_weight_10khz);
    TEST_RUN(test_a_weight_monotonic_low);
    TEST_RUN(test_quality_dsd64);
    TEST_RUN(test_quality_dsd128);
    TEST_RUN(test_quality_dsd256);
    TEST_RUN(test_quality_dsd64_48);
    TEST_RUN(test_quality_dsd64_to_pcm44);
    TEST_RUN(test_quality_pcm_44_to_48);
    TEST_RUN(test_quality_matrix);
}
