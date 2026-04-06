/*
 * foo_dsd_trellis — Convolution filter unit tests
 */

#include "test.h"
#include "../include/convolution.h"
#include "../include/wav_io.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Test 1: Identity IR (Dirac delta) should pass signal unchanged */
static void test_conv_identity(void) {
    conv_state_t state;
    TEST_ASSERT_EQ(conv_init(&state, 44100, 44100), 0, "init at 44100");

    /* Load identity filter */
    int rc = conv_load_ir(&state, "test_filters/identity_L.wav");
    TEST_ASSERT_EQ(rc, 0, "load identity IR");

    /* Process a known signal: 1kHz sine */
    int N = 44100;  /* 1 second */
    double *buf = (double *)calloc(N, sizeof(double));
    for (int i = 0; i < N; i++)
        buf[i] = sin(2.0 * 3.14159265358979 * 1000.0 * i / 44100.0) * 0.5;

    /* Save copy of input */
    double *ref = (double *)malloc(N * sizeof(double));
    memcpy(ref, buf, N * sizeof(double));

    conv_process(&state, buf, N);

    /* The identity IR has the Dirac at tap ir_length/2, plus UPOLS
     * adds partition_size latency. Total delay ≈ ir_length/2 + P.
     * Check that the signal appears in the output with this delay. */
    int P = state.ir.partition_size;
    int delay = state.ir.ir_length / 2 + P;
    int skip_start = delay + P;  /* extra margin */
    int skip_end = P * 2;
    double max_err = 0;
    int compared = 0;
    for (int i = skip_start; i < N - skip_end; i++) {
        int ref_idx = i - delay;
        if (ref_idx >= 0 && ref_idx < N) {
            double err = fabs(buf[i] - ref[ref_idx]);
            if (err > max_err) max_err = err;
            compared++;
        }
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "identity max error = %.6f (should be < 0.01)", max_err);
    TEST_ASSERT_TRUE(max_err < 0.01, msg);

    free(buf);
    free(ref);
    conv_free(&state);
}

/* Test 2: No-mids filter should reduce energy in 500-4000 Hz */
static void test_conv_no_mids(void) {
    conv_state_t state;
    TEST_ASSERT_EQ(conv_init(&state, 44100, 44100), 0, "init at 44100");

    int rc = conv_load_ir(&state, "test_filters/no_mids_L.wav");
    TEST_ASSERT_EQ(rc, 0, "load no_mids IR");

    /* Process 1kHz sine (within the 500-4000 Hz reject band) */
    int N = 88200;  /* 2 seconds */
    double *buf = (double *)calloc(N, sizeof(double));
    for (int i = 0; i < N; i++)
        buf[i] = sin(2.0 * 3.14159265358979 * 1000.0 * i / 44100.0) * 0.5;

    /* Save unfiltered copy */
    double *ref = (double *)malloc(N * sizeof(double));
    memcpy(ref, buf, N * sizeof(double));

    conv_process(&state, buf, N);

    /* Measure RMS of output vs input.
     * Skip settling: 8 partitions × P samples + margin. */
    int skip = state.ir.num_partitions * state.ir.partition_size + 4096;
    double rms_in = 0, rms_out = 0;
    for (int i = skip; i < N; i++) {
        rms_in += ref[i] * ref[i];
        rms_out += buf[i] * buf[i];
    }
    rms_in = sqrt(rms_in / (N - skip));
    rms_out = sqrt(rms_out / (N - skip));

    double ratio_db = 20.0 * log10(rms_out / (rms_in + 1e-30));

    char msg[128];
    snprintf(msg, sizeof(msg),
             "no_mids: RMS in=%.6f out=%.6f ratio=%.1f dB (should be < -3 dB)",
             rms_in, rms_out, ratio_db);
    /* 1kHz sine through no_mids (rejects 500-4000 Hz) should be
     * heavily attenuated. Expect at least -40 dB. */
    TEST_ASSERT_TRUE(ratio_db < -40.0, msg);

    free(buf);
    free(ref);
    conv_free(&state);
}

/* Test 3: Verify conv_process_direct actually modifies the buffer */
static void test_conv_modifies_buffer(void) {
    conv_state_t state;
    TEST_ASSERT_EQ(conv_init(&state, 44100, 44100), 0, "init");

    int rc = conv_load_ir(&state, "test_filters/no_bass_L.wav");
    TEST_ASSERT_EQ(rc, 0, "load no_bass IR");

    /* Process 2 partitions worth of sine wave */
    int P = state.ir.partition_size;
    int N = P * 4;
    double *buf = (double *)calloc(N, sizeof(double));
    double *ref = (double *)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++)
        buf[i] = sin(2.0 * 3.14159265358979 * 100.0 * i / 44100.0) * 0.5;
    memcpy(ref, buf, N * sizeof(double));

    conv_process(&state, buf, N);

    /* Check if ANY sample changed */
    int changed = 0;
    for (int i = 0; i < N; i++) {
        if (fabs(buf[i] - ref[i]) > 1e-15)
            changed++;
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "buffer modified: %d/%d samples changed", changed, N);
    TEST_ASSERT_TRUE(changed > 0, msg);

    free(buf);
    free(ref);
    conv_free(&state);
}

/* Test 4: Direct check - process exactly P samples multiple times */
static void test_conv_direct_partition(void) {
    conv_state_t state;
    TEST_ASSERT_EQ(conv_init(&state, 44100, 44100), 0, "init");
    TEST_ASSERT_EQ(conv_load_ir(&state, "test_filters/no_bass_L.wav"), 0, "load IR");

    int P = state.ir.partition_size;
    /* Process multiple partitions of a 100 Hz sine */
    double *buf = (double *)malloc(P * sizeof(double));
    double total_energy_in = 0, total_energy_out = 0;
    int blocks = 40;  /* 40 blocks = 20480 samples = ~0.46s */
    int skip_blocks = 12;  /* skip startup (8 partitions + margin) */

    for (int b = 0; b < blocks; b++) {
        for (int i = 0; i < P; i++) {
            int t = b * P + i;
            buf[i] = sin(2.0 * 3.14159265358979 * 100.0 * t / 44100.0) * 0.5;
        }
        double e_in = 0;
        for (int i = 0; i < P; i++) e_in += buf[i] * buf[i];

        conv_process(&state, buf, P);

        double e_out = 0;
        for (int i = 0; i < P; i++) e_out += buf[i] * buf[i];

        if (b >= skip_blocks) {
            total_energy_in += e_in;
            total_energy_out += e_out;
        }

        if (b < 3 || b == blocks-1) {
            char msg[128];
            snprintf(msg, sizeof(msg), "  block %d: rms_in=%.4f rms_out=%.4f",
                     b, sqrt(e_in/P), sqrt(e_out/P));
            printf("%s\n", msg);
        }
    }

    int measured_blocks = blocks - skip_blocks;
    double rms_in = sqrt(total_energy_in / (measured_blocks * P));
    double rms_out = sqrt(total_energy_out / (measured_blocks * P));
    double ratio_db = 20.0 * log10(rms_out / (rms_in + 1e-30));

    char msg[128];
    snprintf(msg, sizeof(msg),
             "no_bass 100Hz: rms_in=%.6f rms_out=%.6f ratio=%.1f dB (should be < -20 dB)",
             rms_in, rms_out, ratio_db);
    /* A 100 Hz sine through a 200 Hz high-pass: -7 dB typical */
    TEST_ASSERT_TRUE(ratio_db < -3.0, msg);

    free(buf);
    conv_free(&state);
}

void test_convolution_suite(void) {
    TEST_SUITE("Convolution Filter");
    TEST_RUN(test_conv_direct_partition);
    TEST_RUN(test_conv_modifies_buffer);
    TEST_RUN(test_conv_identity);
    TEST_RUN(test_conv_no_mids);
}
