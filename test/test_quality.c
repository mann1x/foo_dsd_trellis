/*
 * foo_dsd_trellis — Audio quality measurement tests
 *
 * Tests the A-weighting curve, multitone, noise modulation, and NMR
 * measurement functions used by the "Test Quality" button.
 */

#include "test.h"
#include "../include/sinad_measure.h"
#include "../include/dsd_types.h"
#include "../include/fir.h"
#include <math.h>
#include <stdlib.h>

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
    sinad_measure(DSD_RATE_64, NTF_CLANS_5, 2, 4, 64, 1, 0.708f, &r);
    printf("    DSD64: SINAD=%.1f A-wtd=%.1f MT=%.1f NMod=%.1f NMR=%.1f\n",
           r.sinad_theoretical, r.sinad_awtd_theo,
           r.multitone_sinad_db, r.noise_mod_db, r.nmr_db);
    TEST_ASSERT_TRUE(r.ok, "DSD64 measurement should succeed");
    TEST_ASSERT_TRUE(r.sinad_theoretical > 80.0, "DSD64 SINAD > 80 dB");
    TEST_ASSERT_TRUE(r.sinad_awtd_theo >= r.sinad_theoretical - 1.0,
                     "A-weighted >= flat (noise at LF/HF downweighted)");
    TEST_ASSERT_TRUE(r.multitone_sinad_db > 50.0, "DSD64 multitone > 50 dB");
    TEST_ASSERT_TRUE(r.noise_mod_db < 25.0, "DSD64 noise mod < 25 dB");
    TEST_ASSERT_TRUE(r.nmr_db < 0.0, "DSD64 NMR < 0 dB (noise below mask)");
}

static void test_quality_dsd128(void) {
    sinad_result_t r;
    sinad_measure(DSD_RATE_128, NTF_CLANS_6, 2, 4, 128, 1, 0.708f, &r);
    printf("    DSD128: SINAD=%.1f A-wtd=%.1f MT=%.1f NMod=%.1f NMR=%.1f\n",
           r.sinad_theoretical, r.sinad_awtd_theo,
           r.multitone_sinad_db, r.noise_mod_db, r.nmr_db);
    TEST_ASSERT_TRUE(r.ok, "DSD128 measurement should succeed");
    TEST_ASSERT_TRUE(r.sinad_theoretical > 95.0, "DSD128 SINAD > 95 dB");
    TEST_ASSERT_TRUE(r.sinad_awtd_theo > r.sinad_theoretical, "DSD128 A-wtd > flat");
    TEST_ASSERT_TRUE(r.multitone_sinad_db > 80.0, "DSD128 multitone > 80 dB");
    TEST_ASSERT_TRUE(r.nmr_db < -50.0, "DSD128 NMR < -50 dB");
}

static void test_quality_dsd256(void) {
    sinad_result_t r;
    sinad_measure(DSD_RATE_256, NTF_CLANS_6, 2, 4, 128, 1, 0.708f, &r);
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
    sinad_measure(DSD48_RATE_64, NTF_CLANS_6, 2, 4, 64, 1, 0.708f, &r);
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
    typedef struct {
        const char *name; uint32_t in; uint32_t out; int type;
        int ntf; int cands; int depth; int lat;
    } qtest_t;
    /* type: 0=DSD→DSD, 1=DSD→PCM, 2=PCM→DSD, 3=PCM→PCM
     * DSD→DSD params match engine.c path_table + auto-lat for accurate measurement. */
    static const qtest_t tests[] = {
        /* DSD→DSD same-rate (params from path_table: NTF, nc=2, depth=4, auto-lat) */
        { "DSD128 re-encode",       DSD_RATE_128, DSD_RATE_128, 0, NTF_CLANS_6, 2, 4, 128 },
        { "DSD256 re-encode",       DSD_RATE_256, DSD_RATE_256, 0, NTF_CLANS_6, 2, 4, 128 },
        /* DSD→PCM same-family (no SDM params needed) */
        { "DSD64->PCM44.1k",        DSD_RATE_64,  44100,    1, 0,0,0,0 },
        { "DSD128->PCM88.2k",       DSD_RATE_128, 88200,    1, 0,0,0,0 },
        { "DSD256->PCM176.4k",      DSD_RATE_256, 176400,   1, 0,0,0,0 },
        /* DSD/48→PCM same-family */
        { "DSD64/48->PCM48k",       DSD48_RATE_64, 48000,   1, 0,0,0,0 },
        /* PCM→PCM same-family */
        { "44.1k->88.2k FIR",       44100, 88200,   3, 0,0,0,0 },
        { "96k->48k FIR",           96000, 48000,   3, 0,0,0,0 },
        /* PCM→PCM cross-family */
        { "44.1k->48k polyphase",   44100, 48000,   3, 0,0,0,0 },
        { "96k->88.2k polyphase",   96000, 88200,   3, 0,0,0,0 },
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int ok = 0;
    printf("    %-28s  %7s  %7s  %7s  %7s\n", "Path", "SINAD", "A-wtd", "MT", "NMod");
    printf("    %-28s  %7s  %7s  %7s  %7s\n", "---", "---", "---", "---", "---");
    for (int i = 0; i < n; i++) {
        sinad_result_t r;
        memset(&r, 0, sizeof(r));
        if (tests[i].type == 0) {
            sinad_measure(tests[i].out, tests[i].ntf,
                          tests[i].cands, tests[i].depth, tests[i].lat,
                          1, 0.708f, &r);
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

/* Goertzel for FIR diagnostic */
static double goertzel_power_q(const float *x, size_t n, double freq_hz,
                                double sample_rate) {
    double k = freq_hz * (double)n / sample_rate;
    double w = 2.0 * M_PI * k / (double)n;
    double coeff = 2.0 * cos(w);
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (size_t i = 0; i < n; i++) {
        s0 = (double)x[i] + coeff * s1 - s2;
        s2 = s1; s1 = s0;
    }
    double real = s1 - s2 * cos(w);
    double imag = s2 * sin(w);
    return (real * real + imag * imag) / ((double)n * (double)n);
}

/* Direct FIR PCM→PCM quality test (bypasses sinad_measure_pcm_to_pcm) */
static void test_fir_pcm_direct(void) {
    uint32_t pairs[][2] = { {44100,88200}, {96000,48000}, {44100,176400} };
    int n_pairs = 3;
    for (int p = 0; p < n_pairs; p++) {
        uint32_t fs_in = pairs[p][0], fs_out = pairs[p][1];
        size_t n_in = fs_in;
        size_t n_out_buf = (fs_out > fs_in ? fs_out * 2 : fs_in) + 4096;
        float *in  = (float *)malloc(n_in * sizeof(float));
        float *out = (float *)malloc(n_out_buf * sizeof(float));

        /* Pass 1: discover output count */
        for (size_t i = 0; i < n_in; i++)
            in[i] = (float)(0.5 * sin(2.0 * M_PI * 1000.0 * (double)i / (double)fs_in));
        fir_chain_t fir;
        int rc = fir_chain_init(&fir, fs_in, fs_out);
        if (rc != 0) { printf("    %u->%u: init failed\n", fs_in, fs_out); free(in); free(out); continue; }
        size_t produced = fir_chain_process(&fir, in, out, n_in);
        fir_chain_free(&fir);

        /* Pass 2: bin-aligned frequency for actual output N (skip transient) */
        size_t skip = 128;
        size_t meas_n = produced - skip;
        double bw = (double)fs_out / (double)meas_n;
        double gen_freq = (unsigned)(1000.0 / bw + 0.5) * bw;

        for (size_t i = 0; i < n_in; i++)
            in[i] = (float)(0.5 * sin(2.0 * M_PI * gen_freq * (double)i / (double)fs_in));
        rc = fir_chain_init(&fir, fs_in, fs_out);
        if (rc != 0) { free(in); free(out); continue; }
        produced = fir_chain_process(&fir, in, out, n_in);
        fir_chain_free(&fir);

        float *meas = out + skip;
        meas_n = produced - skip;

        float peak = 0;
        for (size_t i = 0; i < meas_n; i++) {
            float a = meas[i] > 0 ? meas[i] : -meas[i];
            if (a > peak) peak = a;
        }

        bw = (double)fs_out / (double)meas_n;
        unsigned sig_bin = (unsigned)(1000.0 / bw + 0.5);
        double freq = sig_bin * bw;
        double sig_pwr = goertzel_power_q(meas, meas_n, freq, (double)fs_out);
        unsigned max_bin = (unsigned)(20000.0 / bw);
        double noise = 0;
        for (unsigned b = 1; b <= max_bin; b++) {
            if (b >= sig_bin - 1 && b <= sig_bin + 1) continue;
            noise += goertzel_power_q(meas, meas_n, b * bw, (double)fs_out);
        }
        double sinad = 10.0 * log10(sig_pwr / (noise > 0 ? noise : 1e-30));
        /* Find top 3 noise bins */
        double top1 = 0; unsigned top1b = 0;
        for (unsigned b = 1; b <= max_bin; b++) {
            if (b >= sig_bin - 1 && b <= sig_bin + 1) continue;
            double np = goertzel_power_q(meas, meas_n, b * bw, (double)fs_out);
            if (np > top1) { top1 = np; top1b = b; }
        }
        printf("    FIR %u->%u: %zu samples, peak=%.4f, SINAD=%.1f dB, top noise: bin %u (%.0f Hz) = %.1f dB\n",
               fs_in, fs_out, produced, peak, sinad,
               top1b, top1b * bw, 10.0 * log10(top1 > 0 ? top1 / sig_pwr : 1e-30));
        TEST_ASSERT_TRUE(sinad > 30.0, "FIR PCM SINAD (investigating)");

        free(in); free(out);
    }
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
    TEST_RUN(test_fir_pcm_direct);
}
