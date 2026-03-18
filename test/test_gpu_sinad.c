/*
 * GPU vs CPU SDM quality comparison with proper SINAD measurement.
 *
 * 1. Generate a test sine at DSD rate
 * 2. Encode to DSD via CPU Trellis (create valid DSD input)
 * 3. Boxcar-smooth the DSD (same as same-rate engine path)
 * 4. Re-encode through CPU SDM → measure SINAD
 * 5. Re-encode through GPU SDM → measure SINAD
 * 6. Compare
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

/* Goertzel SINAD: measure signal power at freq, noise = rest of audio band */
static double goertzel_sinad(const float *pcm, size_t count, double freq,
                              double sample_rate) {
    /* Goertzel for signal power at freq */
    double k = (double)count * freq / sample_rate;
    double w = 2.0 * M_PI * k / (double)count;
    double coeff = 2.0 * cos(w);
    double s0 = 0, s1 = 0, s2 = 0;
    for (size_t i = 0; i < count; i++) {
        s0 = (double)pcm[i] + coeff * s1 - s2;
        s2 = s1; s1 = s0;
    }
    double sig_power = (s1*s1 + s2*s2 - coeff*s1*s2) / ((double)count * count / 4.0);

    /* Total power in audio band via Parseval's */
    double total_power = 0;
    for (size_t i = 0; i < count; i++)
        total_power += (double)pcm[i] * (double)pcm[i];
    total_power /= (double)count;

    double noise_power = total_power - sig_power;
    if (noise_power < 1e-30) return 300.0;
    return 10.0 * log10(sig_power / noise_power);
}

void test_gpu_sinad_comparison(void) {
    printf("\n=== GPU vs CPU SDM SINAD ===\n");

    if (!gpu_available(GPU_BACKEND_CUDA)) {
        printf("  (skipped: CUDA not available)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    uint32_t dsd_rate = 2822400;
    double test_freq = 1000.0;
    /* Bin-align frequency */
    size_t N = 262144;
    double bin = round(test_freq * (double)N / (double)dsd_rate);
    test_freq = bin * (double)dsd_rate / (double)N;
    printf("  Test: %.1f Hz sine, %zu DSD samples at %u Hz\n", test_freq, N, dsd_rate);

    const ntf_filter_t *f = ntf_auto_select(dsd_rate);
    int cands = 4, depth = 4, lat = 256;

    /* Step 1: Generate DSD-rate sine and encode to 1-bit via CPU Trellis */
    float *sine = (float *)malloc(N * sizeof(float));
    float *dsd_input = (float *)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
        sine[i] = (float)(0.5 * sin(2.0 * M_PI * test_freq * (double)i / (double)dsd_rate));

    sdm_context_t enc_sdm;
    sdm_context_init(&enc_sdm, f, depth, 16, 512);  /* high quality encode */
    size_t enc_n = sdm_process_block(&enc_sdm, sine, dsd_input, N);
    printf("  Encoded %zu DSD samples\n", enc_n);

    /* Step 2: Boxcar smooth the DSD input (same as same-rate engine path) */
    int box_taps = 32;
    float *smoothed = (float *)malloc(enc_n * sizeof(float));
    {
        float ring[128] = {0};
        float sum = 0;
        int pos = 0;
        float inv = 1.0f / (float)box_taps;
        float gain = 0.708f;  /* -3 dB */
        for (size_t i = 0; i < enc_n; i++) {
            float s = dsd_input[i] >= 0.0f ? 1.0f : -1.0f;
            sum -= ring[pos];
            ring[pos] = s;
            sum += s;
            pos = (pos + 1) % box_taps;
            smoothed[i] = sum * inv * gain;
        }
    }

    /* Step 3: Re-encode through CPU Trellis */
    float *cpu_out = (float *)calloc(enc_n, sizeof(float));
    sdm_context_t cpu_sdm;
    sdm_context_init(&cpu_sdm, f, depth, cands, lat);
    size_t cpu_n = sdm_process_block(&cpu_sdm, smoothed, cpu_out, enc_n);
    printf("  CPU SDM: %zu output samples\n", cpu_n);

    /* Step 4: Re-encode through GPU Trellis */
    float *gpu_out = (float *)calloc(enc_n, sizeof(float));
    gpu_context_t *ctx = gpu_create(GPU_BACKEND_CUDA);
    if (!ctx) {
        printf("  (skipped: GPU context failed)\n");
        g_tests_run++; g_tests_passed++;
        free(sine); free(dsd_input); free(smoothed); free(cpu_out); free(gpu_out);
        sdm_context_free(&enc_sdm); sdm_context_free(&cpu_sdm);
        return;
    }
    gpu_cuda_trellis_setup(ctx, cands, f->order, lat, f->a, f->g, 0.0);
    int gpu_r = gpu_trellis_process(ctx, smoothed, gpu_out, enc_n,
                                     NULL, NULL, cands, f->order, f->a, f->g);
    printf("  GPU SDM: rc=%d\n", gpu_r);

    /* Step 5: Decimate both outputs with FIR lowpass for proper SINAD */
    /* Use the FIR chain to decimate DSD → PCM 44100 */
    fir_chain_t fir_cpu, fir_gpu;
    memset(&fir_cpu, 0, sizeof(fir_cpu));
    memset(&fir_gpu, 0, sizeof(fir_gpu));
    fir_chain_init(&fir_cpu, dsd_rate, 44100);
    fir_chain_init(&fir_gpu, dsd_rate, 44100);

    size_t pcm_est = cpu_n / 64 + 1024;
    float *cpu_pcm = (float *)calloc(pcm_est, sizeof(float));
    float *gpu_pcm = (float *)calloc(pcm_est, sizeof(float));

    size_t cpu_pcm_n = fir_chain_process(&fir_cpu, cpu_out, cpu_pcm, cpu_n);
    size_t gpu_pcm_n = (gpu_r == 0) ?
        fir_chain_process(&fir_gpu, gpu_out, gpu_pcm, enc_n) : 0;

    printf("  CPU PCM: %zu samples at 44100 Hz\n", cpu_pcm_n);
    printf("  GPU PCM: %zu samples at 44100 Hz\n", gpu_pcm_n);

    /* Step 6: Measure SINAD with Goertzel */
    size_t skip = 1024;  /* skip FIR startup transient */
    if (cpu_pcm_n > skip) {
        double cpu_sinad = goertzel_sinad(cpu_pcm + skip, cpu_pcm_n - skip,
                                           test_freq, 44100.0);
        printf("  CPU SINAD: %.1f dB\n", cpu_sinad);
    }
    if (gpu_pcm_n > skip) {
        double gpu_sinad = goertzel_sinad(gpu_pcm + skip, gpu_pcm_n - skip,
                                           test_freq, 44100.0);
        printf("  GPU SINAD: %.1f dB\n", gpu_sinad);
    }

    /* Step 7: Bit match */
    size_t cmp = cpu_n < enc_n ? cpu_n : enc_n;
    if (cmp > (size_t)lat + 8192) {
        int match = 0;
        size_t start = (size_t)lat + 8192;
        for (size_t i = start; i < cmp; i++)
            if (gpu_out[i] == cpu_out[i]) match++;
        printf("  Bit match: %d/%zu = %.1f%%\n", match, cmp - start,
               100.0 * match / (double)(cmp - start));
    }

    /* Cleanup */
    free(sine); free(dsd_input); free(smoothed);
    free(cpu_out); free(gpu_out);
    free(cpu_pcm); free(gpu_pcm);
    sdm_context_free(&enc_sdm);
    sdm_context_free(&cpu_sdm);
    fir_chain_free(&fir_cpu);
    fir_chain_free(&fir_gpu);
    gpu_destroy(ctx);

    g_tests_run++;
    g_tests_passed++;
}
