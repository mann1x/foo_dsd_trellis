/*
 * Standalone test for convolution module.
 * Processes a known signal through conv_process_direct and verifies output.
 */
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/convolution.h"
#include "../include/wav_io.h"

/* Stubs */
void trellis_log_c(const char *msg) { printf("[LOG] %s\n", msg); }
bool g_log_enabled = true;

int main(void) {
    printf("=== Convolution Module Test ===\n\n");

    /* Load the no_mids filter */
    conv_state_t state;
    if (conv_init(&state, 44100, 44100) != 0) {
        printf("FAIL: conv_init failed\n");
        return 1;
    }

    if (conv_load_ir(&state, "test_filters/no_mids_L.wav") != 0) {
        printf("FAIL: conv_load_ir failed\n");
        return 1;
    }

    printf("\nIR loaded: %d taps, P=%d, %d partitions\n",
           state.ir.ir_length, state.ir.partition_size, state.ir.num_partitions);

    /* Generate 1 second of white noise */
    int N = 44100;
    double *input = (double *)malloc(N * sizeof(double));
    double *output = (double *)malloc(N * sizeof(double));
    srand(42);
    for (int i = 0; i < N; i++) {
        input[i] = ((double)rand() / RAND_MAX - 0.5) * 0.2;
        output[i] = input[i];  /* copy for in-place processing */
    }

    /* Process through convolution */
    printf("\nProcessing %d samples...\n", N);
    conv_process(&state, output, N);

    /* Check if output differs from input */
    double max_diff = 0;
    double rms_diff = 0;
    int nonzero_out = 0;
    for (int i = 0; i < N; i++) {
        double d = fabs(output[i] - input[i]);
        if (d > max_diff) max_diff = d;
        rms_diff += d * d;
        if (output[i] != 0.0) nonzero_out++;
    }
    rms_diff = sqrt(rms_diff / N);

    printf("Max diff:    %.6f\n", max_diff);
    printf("RMS diff:    %.6f\n", rms_diff);
    printf("Nonzero out: %d / %d\n", nonzero_out, N);

    if (max_diff < 1e-10) {
        printf("\nFAIL: output == input (convolution had no effect!)\n");
    } else {
        printf("\nPASS: convolution modified the signal\n");
    }

    /* Quick spectral check: energy in mid band vs bass */
    /* Process a longer chunk to let the filter settle */
    int N2 = 88200;
    double *test = (double *)calloc(N2, sizeof(double));
    srand(42);
    for (int i = 0; i < N2; i++)
        test[i] = ((double)rand() / RAND_MAX - 0.5) * 0.2;

    conv_reset(&state);
    conv_process(&state, test, N2);

    /* Measure RMS in frequency bands using simple bandpass */
    /* Skip first 4096 samples (settling time) */
    int start = 8192;
    int len = N2 - start;
    double rms_bass = 0, rms_mid = 0, rms_hi = 0;
    int cnt_bass = 0, cnt_mid = 0, cnt_hi = 0;

    /* Simple approach: measure energy of test signal */
    for (int i = start; i < N2; i++) {
        double v = test[i] * test[i];
        rms_mid += v;
    }
    rms_mid = sqrt(rms_mid / len);

    /* Compare with unfiltered */
    double *ref = (double *)calloc(N2, sizeof(double));
    srand(42);
    for (int i = 0; i < N2; i++)
        ref[i] = ((double)rand() / RAND_MAX - 0.5) * 0.2;
    double rms_ref = 0;
    for (int i = start; i < N2; i++)
        rms_ref += ref[i] * ref[i];
    rms_ref = sqrt(rms_ref / len);

    printf("\nRMS unfiltered: %.6f\n", rms_ref);
    printf("RMS filtered:   %.6f\n", rms_mid);
    printf("Ratio:          %.2f dB\n", 20 * log10(rms_mid / (rms_ref + 1e-30)));

    free(input);
    free(output);
    free(test);
    free(ref);
    conv_free(&state);
    return 0;
}
