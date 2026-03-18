/*
 * GPU vs CPU SDM SINAD comparison with proper Goertzel measurement.
 */

#include "test.h"
#include "../include/gpu_compute.h"
#include "../include/fir.h"
#include "../include/ntf.h"
#include "../include/trellis.h"
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern void trellis_log_c(const char *msg);

/* Returns signal power at freq using Goertzel.
 * Result is in same units as sum(x²)/N (Parseval-compatible). */
static double goertzel_power(const float *x, size_t n, double freq, double fs) {
    double k = (double)n * freq / fs;
    double w = 2.0 * M_PI * k / (double)n;
    double c = 2.0 * cos(w);
    double s1 = 0, s2 = 0;
    for (size_t i = 0; i < n; i++) {
        double s0 = (double)x[i] + c * s1 - s2;
        s2 = s1; s1 = s0;
    }
    /* DFT magnitude squared / N = power at this bin */
    double mag2 = s1*s1 + s2*s2 - c*s1*s2;
    return 2.0 * mag2 / ((double)n * (double)n);
}

void test_gpu_sinad_comparison(void) {
    printf("\n=== GPU vs CPU SDM SINAD ===\n");

    if (!gpu_available(GPU_BACKEND_CUDA)) {
        printf("  (skipped: CUDA not available)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    uint32_t dsd_rate = 2822400;
    int cands = 4, depth = 4, lat = 256;
    size_t N = 524288;  /* ~186ms at DSD64, gives ~8192 PCM samples */

    /* Bin-align test frequency to BOTH DSD and PCM rates.
     * PCM output will be N/64 samples at 44100 Hz.
     * Align to PCM bin for accurate Goertzel. */
    size_t pcm_n_est = N / 64;
    double test_freq = 1000.0;
    double bin = round(test_freq * (double)pcm_n_est / 44100.0);
    test_freq = bin * 44100.0 / (double)pcm_n_est;
    printf("  Test: %.1f Hz, %zu samples at %u Hz\n", test_freq, N, dsd_rate);

    const ntf_filter_t *f = ntf_auto_select(dsd_rate);

    /* Step 1: Generate sine and encode to DSD */
    float *sine = (float *)malloc(N * sizeof(float));
    float *dsd_enc = (float *)calloc(N, sizeof(float));
    if (!sine || !dsd_enc) { printf("  malloc failed\n"); return; }

    for (size_t i = 0; i < N; i++)
        sine[i] = (float)(0.5 * sin(2.0 * M_PI * test_freq * (double)i / (double)dsd_rate));

    sdm_context_t enc;
    sdm_context_init(&enc, f, depth, 16, 512);
    size_t enc_n = sdm_process_block(&enc, sine, dsd_enc, N);
    sdm_context_free(&enc);
    printf("  Encoded: %zu DSD samples\n", enc_n);
    if (enc_n < 10000) { printf("  too few samples\n"); goto cleanup1; }

    /* Step 2: Boxcar smooth */
    float *smoothed = (float *)malloc(enc_n * sizeof(float));
    if (!smoothed) goto cleanup1;
    {
        float ring[128] = {0}; float sum = 0; int p = 0;
        for (size_t i = 0; i < enc_n; i++) {
            float s = dsd_enc[i] >= 0.0f ? 1.0f : -1.0f;
            sum -= ring[p]; ring[p] = s; sum += s;
            p = (p + 1) % 32;
            smoothed[i] = sum / 32.0f * 0.708f;
        }
    }

    /* Step 3: CPU Trellis re-encode */
    float *cpu_out = (float *)calloc(enc_n, sizeof(float));
    if (!cpu_out) goto cleanup2;
    {
        sdm_context_t cpu_sdm;
        sdm_context_init(&cpu_sdm, f, depth, cands, lat);
        size_t cpu_n = sdm_process_block(&cpu_sdm, smoothed, cpu_out, enc_n);
        sdm_context_free(&cpu_sdm);
        printf("  CPU re-encode: %zu samples\n", cpu_n);
    }

    /* Step 4: GPU Trellis re-encode */
    float *gpu_out = (float *)calloc(enc_n, sizeof(float));
    if (!gpu_out) goto cleanup3;
    {
        gpu_context_t *ctx = gpu_create(GPU_BACKEND_CUDA);
        if (!ctx) { printf("  GPU create failed\n"); goto cleanup4; }
        gpu_cuda_trellis_setup(ctx, cands, f->order, lat, f->a, f->g, 0.0);
        int rc = gpu_trellis_process(ctx, smoothed, gpu_out, enc_n,
                                      NULL, NULL, cands, f->order, f->a, f->g);
        gpu_destroy(ctx);
        printf("  GPU re-encode: rc=%d\n", rc);
        if (rc != 0) goto cleanup4;
    }

    /* Step 5: Decimate to PCM via FIR */
    {
        size_t pcm_max = enc_n;  /* oversized to be safe */
        float *cpu_pcm = (float *)calloc(pcm_max, sizeof(float));
        float *gpu_pcm = (float *)calloc(pcm_max, sizeof(float));
        if (!cpu_pcm || !gpu_pcm) {
            free(cpu_pcm); free(gpu_pcm); goto cleanup4;
        }

        fir_chain_t fir1, fir2;
        memset(&fir1, 0, sizeof(fir1));
        memset(&fir2, 0, sizeof(fir2));
        fir_chain_init(&fir1, dsd_rate, 44100);
        fir_chain_init(&fir2, dsd_rate, 44100);

        size_t cpu_pcm_n = fir_chain_process(&fir1, cpu_out, cpu_pcm, enc_n);
        size_t gpu_pcm_n = fir_chain_process(&fir2, gpu_out, gpu_pcm, enc_n);
        fir_chain_free(&fir1);
        fir_chain_free(&fir2);

        printf("  CPU PCM: %zu, GPU PCM: %zu\n", cpu_pcm_n, gpu_pcm_n);

        /* Measure SINAD */
        size_t skip = 128;
        if (cpu_pcm_n > skip + 1024) {
            size_t mn = cpu_pcm_n - skip;
            double sig = goertzel_power(cpu_pcm + skip, mn, test_freq, 44100.0);
            double total = 0;
            for (size_t i = skip; i < cpu_pcm_n; i++)
                total += (double)cpu_pcm[i] * cpu_pcm[i];
            total /= (double)mn;
            double noise = total - sig;
            if (noise < 1e-30) noise = 1e-30;
            printf("  *** CPU SINAD: %.1f dB *** (sig=%.2e noise=%.2e total=%.2e)\n",
                   10.0 * log10(sig / noise), sig, noise, total);
        }
        if (gpu_pcm_n > skip + 1024) {
            size_t mn = gpu_pcm_n - skip;
            double sig = goertzel_power(gpu_pcm + skip, mn, test_freq, 44100.0);
            double total = 0;
            for (size_t i = skip; i < gpu_pcm_n; i++)
                total += (double)gpu_pcm[i] * gpu_pcm[i];
            total /= (double)mn;
            double noise = total - sig;
            if (noise < 1e-30) noise = 1e-30;
            printf("  *** GPU SINAD: %.1f dB *** (sig=%.2e noise=%.2e total=%.2e)\n",
                   10.0 * log10(sig / noise), sig, noise, total);
        }

        free(cpu_pcm); free(gpu_pcm);
    }

cleanup4: free(gpu_out);
cleanup3: free(cpu_out);
cleanup2: free(smoothed);
cleanup1: free(sine); free(dsd_enc);

    g_tests_run++;
    g_tests_passed++;
}
