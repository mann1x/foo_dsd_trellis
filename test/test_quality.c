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
}
