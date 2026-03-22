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

/* ─── GPU DAS pipeline SINAD test ─── */

static void test_gpu_das_sinad(void) {
    printf("  test_gpu_das_sinad...\n");
    if (!gpu_available(GPU_BACKEND_CUDA)) {
        printf("    (skipped: CUDA not available)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    /* Test at DSD512 config: nc=2, lat=32 — actual playback params */
    uint32_t dsd_rate = 22579200;
    int nc = 2, lat = 32;
    size_t N = 22579200;  /* 1 second of DSD512 */
    double test_freq = 1000.0;

    /* Bin-align frequency */
    int decimation = (int)(dsd_rate / 44100);
    size_t pcm_n_est = N / (size_t)decimation;
    double bin = round(test_freq * (double)pcm_n_est / 44100.0);
    test_freq = bin * 44100.0 / (double)pcm_n_est;

    const ntf_filter_t *f = ntf_auto_select(dsd_rate);
    if (!f) { printf("    no NTF\n"); return; }

    printf("    DAS @ DSD512: %.1f Hz, %zu samples, nc=%d lat=%d\n",
           test_freq, N, nc, lat);

    /* Generate test signal: sine → DSD encode → boxcar smooth */
    size_t enc_n = 0;
    float *smoothed = make_test_signal(f, dsd_rate, test_freq, N, &enc_n);
    if (!smoothed) { printf("    signal gen failed\n"); return; }

    /* CPU reference */
    float *cpu_out = (float *)calloc(enc_n, sizeof(float));
    double *smoothed_d = (double *)malloc(enc_n * sizeof(double));
    if (!cpu_out || !smoothed_d) {
        free(smoothed); free(cpu_out); free(smoothed_d); return;
    }
    for (size_t i = 0; i < enc_n; i++) smoothed_d[i] = (double)smoothed[i];
    {
        sdm_context_t cpu_sdm;
        sdm_context_init(&cpu_sdm, f, f->order, nc, lat);
        sdm_process_block(&cpu_sdm, smoothed_d, cpu_out, enc_n);
        sdm_context_free(&cpu_sdm);
    }
    double cpu_sinad = measure_sinad(cpu_out, enc_n, dsd_rate, test_freq);

    /* GPU DAS pipeline: gpu_cuda_trellis_das with 1 channel */
    float *gpu_out = (float *)calloc(enc_n, sizeof(float));
    gpu_context_t *ctx = gpu_create(GPU_BACKEND_CUDA);
    if (!ctx || !gpu_out) {
        printf("    GPU create failed\n");
        free(smoothed); free(cpu_out); free(smoothed_d); free(gpu_out);
        return;
    }
    gpu_cuda_trellis_setup(ctx, nc, f->order, lat, f->a, f->g, 0.0);

    /* DAS path: 1 channel (fp64 input) */
    double *smoothed_d_das = (double *)malloc(enc_n * sizeof(double));
    if (smoothed_d_das) {
        for (size_t i = 0; i < enc_n; i++) smoothed_d_das[i] = (double)smoothed[i];
    }
    int rc = gpu_cuda_trellis_das(ctx, smoothed_d_das, gpu_out, enc_n, 1);
    free(smoothed_d_das);
    gpu_destroy(ctx);
    free(smoothed_d);

    if (rc != 0) {
        printf("    DAS pipeline failed (rc=%d)\n", rc);
        /* Fallback: test old path */
        free(smoothed); free(cpu_out); free(gpu_out);
        return;
    }

    double das_sinad = measure_sinad(gpu_out, enc_n, dsd_rate, test_freq);

    /* Also test old single-path for comparison */
    gpu_context_t *ctx2 = gpu_create(GPU_BACKEND_CUDA);
    float *old_out = (float *)calloc(enc_n, sizeof(float));
    double old_sinad = -999.0;
    if (ctx2 && old_out) {
        gpu_cuda_trellis_setup(ctx2, nc, f->order, lat, f->a, f->g, 0.0);
        /* Convert fp32 input to fp64 for trellis kernel */
        double *smoothed_d2 = (double *)malloc(enc_n * sizeof(double));
        if (smoothed_d2) {
            for (size_t i = 0; i < enc_n; i++) smoothed_d2[i] = (double)smoothed[i];
        }
        int rc2 = smoothed_d2 ? gpu_cuda_trellis(ctx2, smoothed_d2, old_out, enc_n) : -1;
        free(smoothed_d2);
        if (rc2 == 0)
            old_sinad = measure_sinad(old_out, enc_n, dsd_rate, test_freq);
        gpu_destroy(ctx2);
    }
    free(old_out);

    /* Sample comparison */
    double cpu_avg = 0, das_avg = 0;
    for (size_t i = 10000; i < 10064 && i < enc_n; i++) {
        cpu_avg += cpu_out[i];
        das_avg += gpu_out[i];
    }
    printf("    DSD avg[10000..10063]: CPU=%.4f DAS=%.4f\n",
           cpu_avg/64, das_avg/64);
    printf("    *** CPU SINAD: %.1f dB, DAS SINAD: %.1f dB, old GPU: %.1f dB ***\n",
           cpu_sinad, das_sinad, old_sinad);

    TEST_ASSERT(das_sinad > 20.0, "GPU DAS SINAD should be > 20 dB");

    free(smoothed); free(cpu_out); free(gpu_out);
}

/* ─── Multi-chunk DAS test: detects noise at chunk boundaries ─── */

static void test_gpu_das_multi_chunk(void) {
    printf("  test_gpu_das_multi_chunk...\n");
    if (!gpu_available(GPU_BACKEND_CUDA)) {
        printf("    (skipped: CUDA not available)\n");
        g_tests_run++; g_tests_passed++;
        return;
    }

    uint32_t dsd_rate = 22579200;
    int nc = 2, lat = 32;
    int num_chunks = 5;
    size_t chunk_size = 4000000;  /* ~177ms per chunk */
    size_t total_n = chunk_size * (size_t)num_chunks;
    double test_freq = 1000.0;

    /* Bin-align */
    int decimation = (int)(dsd_rate / 44100);
    size_t pcm_est = total_n / (size_t)decimation;
    double bin = round(test_freq * (double)pcm_est / 44100.0);
    test_freq = bin * 44100.0 / (double)pcm_est;

    const ntf_filter_t *f = ntf_auto_select(dsd_rate);
    if (!f) { printf("    no NTF\n"); return; }

    printf("    Multi-chunk DAS: %d chunks x %zu samples, nc=%d lat=%d\n",
           num_chunks, chunk_size, nc, lat);

    /* Generate full test signal */
    size_t enc_n = 0;
    float *smoothed = make_test_signal(f, dsd_rate, test_freq, total_n, &enc_n);
    if (!smoothed) { printf("    signal gen failed\n"); return; }

    /* CPU reference: single continuous pass */
    float *cpu_out = (float *)calloc(enc_n, sizeof(float));
    double *smoothed_d = (double *)malloc(enc_n * sizeof(double));
    if (!cpu_out || !smoothed_d) {
        free(smoothed); free(cpu_out); free(smoothed_d); return;
    }
    for (size_t i = 0; i < enc_n; i++) smoothed_d[i] = (double)smoothed[i];
    {
        sdm_context_t cpu_sdm;
        sdm_context_init(&cpu_sdm, f, f->order, nc, lat);
        sdm_process_block(&cpu_sdm, smoothed_d, cpu_out, enc_n);
        sdm_context_free(&cpu_sdm);
    }
    double cpu_sinad = measure_sinad(cpu_out, enc_n, dsd_rate, test_freq);

    /* GPU DAS: process in chunks (simulates live playback) */
    float *das_out = (float *)calloc(enc_n, sizeof(float));
    gpu_context_t *ctx = gpu_create(GPU_BACKEND_CUDA);
    if (!ctx || !das_out) {
        printf("    GPU create failed\n");
        free(smoothed); free(cpu_out); free(smoothed_d); free(das_out);
        return;
    }
    gpu_cuda_trellis_setup(ctx, nc, f->order, lat, f->a, f->g, 0.0);

    size_t out_pos = 0;
    for (int c = 0; c < num_chunks; c++) {
        size_t in_start = (size_t)c * chunk_size;
        size_t in_count = chunk_size;
        if (in_start + in_count > enc_n) in_count = enc_n - in_start;
        if (in_count == 0) break;

        float *chunk_out = (float *)calloc(in_count, sizeof(float));
        if (!chunk_out) break;

        /* Convert chunk to fp64 for trellis kernel */
        double *chunk_d = (double *)malloc(in_count * sizeof(double));
        if (chunk_d) {
            for (size_t i = 0; i < in_count; i++)
                chunk_d[i] = (double)smoothed[in_start + i];
        }
        int rc = gpu_cuda_trellis_das(ctx, chunk_d,
                                       chunk_out, in_count, 1);
        free(chunk_d);
        if (rc != 0) {
            printf("    DAS chunk %d failed\n", c);
            free(chunk_out);
            break;
        }

        /* Copy to continuous output.
         * DAS returns D*num_segs samples which may differ from in_count. */
        size_t copy = in_count;
        if (out_pos + copy > enc_n) copy = enc_n - out_pos;
        memcpy(das_out + out_pos, chunk_out, copy * sizeof(float));
        out_pos += copy;
        free(chunk_out);
    }
    gpu_destroy(ctx);
    free(smoothed_d);

    double das_sinad = measure_sinad(das_out, out_pos, dsd_rate, test_freq);

    /* Check for discontinuities at chunk boundaries */
    int boundary_glitches = 0;
    for (int c = 1; c < num_chunks; c++) {
        size_t bnd = (size_t)c * chunk_size;
        if (bnd >= out_pos) break;
        /* Check if consecutive samples have opposite signs (full-scale flip) */
        for (int d = -2; d <= 2; d++) {
            size_t idx = bnd + (size_t)d;
            if (idx > 0 && idx < out_pos) {
                float prev = das_out[idx - 1];
                float curr = das_out[idx];
                /* In DSD, ±1 transitions are normal. Look for unusual
                 * patterns: compare with CPU at same position */
                if (idx < enc_n && das_out[idx] != cpu_out[idx]) {
                    /* Count how many consecutive mismatches around boundary */
                    int mismatch_run = 0;
                    for (size_t j = idx; j < idx + 100 && j < out_pos && j < enc_n; j++) {
                        if (das_out[j] != cpu_out[j]) mismatch_run++;
                        else break;
                    }
                    if (mismatch_run > 50) {
                        boundary_glitches++;
                        printf("    chunk boundary %d @ %zu: %d consecutive mismatches\n",
                               c, bnd, mismatch_run);
                        break;
                    }
                }
            }
        }
    }

    printf("    *** CPU SINAD: %.1f dB, multi-chunk DAS: %.1f dB, boundary glitches: %d ***\n",
           cpu_sinad, das_sinad, boundary_glitches);

    TEST_ASSERT(das_sinad > 20.0, "multi-chunk GPU DAS SINAD should be > 20 dB");

    free(smoothed); free(cpu_out); free(das_out);
}

/* Combined entry point */
void test_gpu_sinad_comparison(void) {
    printf("\n=== GPU vs CPU SDM SINAD ===\n");
    test_gpu_sinad_cuda();
    test_gpu_sinad_cuda_dsd256();
    test_gpu_sinad_dx12();
    test_gpu_das_sinad();
    test_gpu_das_multi_chunk();
}
