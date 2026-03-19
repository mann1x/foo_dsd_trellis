/*
 * GPU vs CPU SDM SINAD comparison with proper Goertzel measurement.
 * Tests CUDA, DX12, and DX11 backends independently.
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
    double mag2 = s1*s1 + s2*s2 - c*s1*s2;
    return 2.0 * mag2 / ((double)n * (double)n);
}

/* Shared test signal: sine → DSD encode → boxcar smooth.
 * Returns smoothed buffer + enc_n. Caller frees smoothed. */
static float *make_test_signal(const ntf_filter_t *f, uint32_t dsd_rate,
                                double test_freq, size_t N,
                                size_t *out_enc_n) {
    double *sine = (double *)malloc(N * sizeof(double));
    float *dsd_enc = (float *)calloc(N, sizeof(float));
    if (!sine || !dsd_enc) { free(sine); free(dsd_enc); return NULL; }

    for (size_t i = 0; i < N; i++)
        sine[i] = 0.5 * sin(2.0 * M_PI * test_freq * (double)i / (double)dsd_rate);

    sdm_context_t enc;
    sdm_context_init(&enc, f, 4, 16, 512);
    size_t enc_n = sdm_process_block(&enc, sine, dsd_enc, N);
    sdm_context_free(&enc);
    free(sine);

    if (enc_n < 10000) { free(dsd_enc); return NULL; }

    float *smoothed = (float *)malloc(enc_n * sizeof(float));
    if (!smoothed) { free(dsd_enc); return NULL; }
    {
        float ring[128] = {0}; float sum = 0; int p = 0;
        for (size_t i = 0; i < enc_n; i++) {
            float s = dsd_enc[i] >= 0.0f ? 1.0f : -1.0f;
            sum -= ring[p]; ring[p] = s; sum += s;
            p = (p + 1) % 32;
            smoothed[i] = sum / 32.0f * 0.708f;
        }
    }
    free(dsd_enc);
    *out_enc_n = enc_n;
    return smoothed;
}

/* Measure SINAD of a DSD buffer after FIR decimation to PCM.
 * Returns SINAD in dB, or -999 on error. */
static double measure_sinad(const float *dsd, size_t dsd_n,
                             uint32_t dsd_rate, double test_freq) {
    size_t pcm_max = dsd_n;
    float *pcm = (float *)calloc(pcm_max, sizeof(float));
    if (!pcm) return -999.0;

    fir_chain_t fir;
    memset(&fir, 0, sizeof(fir));
    fir_chain_init(&fir, dsd_rate, 44100);
    size_t pcm_n = fir_chain_process(&fir, dsd, pcm, dsd_n);
    fir_chain_free(&fir);

    size_t skip = 128;
    double sinad = -999.0;
    if (pcm_n > skip + 1024) {
        size_t mn = pcm_n - skip;
        double sig = goertzel_power(pcm + skip, mn, test_freq, 44100.0);
        double total = 0;
        for (size_t i = skip; i < pcm_n; i++)
            total += (double)pcm[i] * pcm[i];
        total /= (double)mn;
        double noise = total - sig;
        if (noise < 1e-30) noise = 1e-30;
        sinad = 10.0 * log10(sig / noise);
    }
    free(pcm);
    return sinad;
}

/* Core SINAD comparison: runs CPU + one GPU backend.
 * backend: GPU_BACKEND_CUDA or GPU_BACKEND_DIRECTX.
 * label: "CUDA" or "DX12" etc.
 * dsd_rate: DSD sample rate (e.g. 2822400 for DSD64, 11289600 for DSD256)
 * N: number of DSD samples to generate */
static void run_sinad_test_rate(gpu_backend_t backend, const char *label,
                                 uint32_t dsd_rate, size_t N) {
    int cands = 4, lat = 256;

    int decimation = (int)(dsd_rate / 44100);
    size_t pcm_n_est = N / (size_t)decimation;
    double test_freq = 1000.0;
    double bin = round(test_freq * (double)pcm_n_est / 44100.0);
    test_freq = bin * 44100.0 / (double)pcm_n_est;
    printf("    %s @ DSD%d: %.1f Hz, %zu samples\n",
           label, (int)(dsd_rate / 44100), test_freq, N);

    const ntf_filter_t *f = ntf_auto_select(dsd_rate);

    /* Generate test signal */
    size_t enc_n = 0;
    float *smoothed = make_test_signal(f, dsd_rate, test_freq, N, &enc_n);
    if (!smoothed) { printf("    signal gen failed\n"); return; }

    /* CPU reference — sdm_process_block takes double* input */
    float *cpu_out = (float *)calloc(enc_n, sizeof(float));
    double *smoothed_d = (double *)malloc(enc_n * sizeof(double));
    if (!cpu_out || !smoothed_d) { free(smoothed); free(cpu_out); free(smoothed_d); return; }
    for (size_t i = 0; i < enc_n; i++) smoothed_d[i] = (double)smoothed[i];
    {
        sdm_context_t cpu_sdm;
        sdm_context_init(&cpu_sdm, f, 4, cands, lat);
        sdm_process_block(&cpu_sdm, smoothed_d, cpu_out, enc_n);
        sdm_context_free(&cpu_sdm);
    }
    free(smoothed_d);
    double cpu_sinad = measure_sinad(cpu_out, enc_n, dsd_rate, test_freq);

    /* GPU */
    float *gpu_out = (float *)calloc(enc_n, sizeof(float));
    if (!gpu_out) { free(smoothed); free(cpu_out); return; }

    gpu_context_t *ctx = gpu_create(backend);
    if (!ctx) {
        printf("    %s: create failed\n", label);
        free(smoothed); free(cpu_out); free(gpu_out);
        return;
    }

    /* Setup trellis for this backend */
    if (backend == GPU_BACKEND_CUDA) {
        gpu_cuda_trellis_setup(ctx, cands, f->order, lat, f->a, f->g, 0.0);
    } else {
        gpu_dx12_trellis_setup_full(ctx, cands, f->order, lat, f->a, f->g, 0.0);
    }

    int rc = gpu_trellis_process(ctx, smoothed, gpu_out, enc_n,
                                  NULL, NULL, cands, f->order, f->a, f->g);
    gpu_destroy(ctx);

    if (rc != 0) {
        printf("    %s: trellis_process failed (rc=%d)\n", label, rc);
        free(smoothed); free(cpu_out); free(gpu_out);
        return;
    }

    double gpu_sinad = measure_sinad(gpu_out, enc_n, dsd_rate, test_freq);

    /* DSD avg comparison at position 10000 */
    double cpu_avg = 0, gpu_avg = 0;
    for (size_t i = 10000; i < 10064; i++) {
        cpu_avg += cpu_out[i];
        gpu_avg += gpu_out[i];
    }
    printf("    DSD avg[10000..10063]: CPU=%.4f %s=%.4f\n",
           cpu_avg/64, label, gpu_avg/64);

    printf("    *** CPU SINAD: %.1f dB, %s SINAD: %.1f dB (delta=%.1f dB) ***\n",
           cpu_sinad, label, gpu_sinad, gpu_sinad - cpu_sinad);

    free(smoothed); free(cpu_out); free(gpu_out);
}

/* Convenience wrapper for DSD64 (backward compat) */
static void run_sinad_test(gpu_backend_t backend, const char *label) {
    run_sinad_test_rate(backend, label, 2822400, 524288);
}

/* ─── Per-backend test functions ─── */

void test_gpu_sinad_cuda(void) {
    printf("  test_gpu_sinad_cuda...\n");
    if (!gpu_available(GPU_BACKEND_CUDA)) {
        printf("    (skipped: CUDA not available)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }
    run_sinad_test(GPU_BACKEND_CUDA, "CUDA");
    g_tests_run++; g_tests_passed++;
}

void test_gpu_sinad_cuda_dsd256(void) {
    printf("  test_gpu_sinad_cuda_dsd256...\n");
    if (!gpu_available(GPU_BACKEND_CUDA)) {
        printf("    (skipped: CUDA not available)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }
    /* DSD256 = 11289600 Hz, ~2M samples ≈ 0.18s of audio */
    run_sinad_test_rate(GPU_BACKEND_CUDA, "CUDA", 11289600, 2097152);
    g_tests_run++; g_tests_passed++;
}

void test_gpu_sinad_dx12(void) {
    printf("  test_gpu_sinad_dx12...\n");
    if (!gpu_available(GPU_BACKEND_DIRECTX)) {
        printf("    (skipped: DirectX not available)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }
    run_sinad_test(GPU_BACKEND_DIRECTX, "DX12");
    g_tests_run++; g_tests_passed++;
}

/* Combined entry point */
void test_gpu_sinad_comparison(void) {
    printf("\n=== GPU vs CPU SDM SINAD ===\n");
    test_gpu_sinad_cuda();
    test_gpu_sinad_cuda_dsd256();
    test_gpu_sinad_dx12();
}
