#define _CRT_SECURE_NO_WARNINGS
#include "test.h"
#include "../include/gpu_compute.h"
#include "../include/ntf.h"
#include "../include/trellis.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern void trellis_log_c(const char *msg);

/* Process a raw DSD WAV through GPU Trellis and save output as WAV.
 * Usage: test.exe --encode --gpu-trellis input.wav output.wav [num_segs] */
int gpu_trellis_offline(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: --gpu-trellis input.wav output.wav [num_segs]\n");
        return 1;
    }
    const char *in_path = argv[1];
    const char *out_path = argv[2];
    int force_segs = (argc > 3) ? atoi(argv[3]) : 0;

    /* Read input WAV (32-bit float mono DSD at 2822400 Hz) */
    FILE *fin = fopen(in_path, "rb");
    if (!fin) { printf("Cannot open %s\n", in_path); return 1; }
    fseek(fin, 0, SEEK_END);
    long fsize = ftell(fin);
    fseek(fin, 44, SEEK_SET);  /* skip WAV header */
    size_t n_samples = (fsize - 44) / sizeof(float);
    float *input = (float *)malloc(n_samples * sizeof(float));
    float *output = (float *)calloc(n_samples, sizeof(float));
    fread(input, sizeof(float), n_samples, fin);
    fclose(fin);
    printf("Input: %zu samples (%.1fs)\n", n_samples, (double)n_samples / 2822400.0);

    /* Boxcar smooth (32 taps, gain 0.708) — same as engine */
    float *smoothed = (float *)malloc(n_samples * sizeof(float));
    {
        float ring[128] = {0}; float sum = 0; int p = 0;
        for (size_t i = 0; i < n_samples; i++) {
            float s = input[i] >= 0.0f ? 1.0f : -1.0f;
            sum -= ring[p]; ring[p] = s; sum += s;
            p = (p + 1) % 32;
            smoothed[i] = sum / 32.0f * 0.708f;
        }
    }

    /* Create GPU context */
    if (!gpu_available(GPU_BACKEND_CUDA)) {
        printf("CUDA not available\n");
        free(input); free(output); free(smoothed);
        return 1;
    }
    gpu_context_t *ctx = gpu_create(GPU_BACKEND_CUDA);
    if (!ctx) { printf("GPU create failed\n"); return 1; }

    const ntf_filter_t *f = ntf_auto_select(2822400);
    int use_hawksford = (argc > 3 && strcmp(argv[3], "hawk") == 0);

    if (use_hawksford) {
        /* Hawksford: nc=64, single segment, intra-step parallel */
        extern int gpu_cuda_trellis_hawksford(void *, const float *, float *, size_t);
        gpu_cuda_trellis_setup(ctx, 4, f->order, 256, f->a, f->g, 0.0);
        printf("Processing %zu samples through Hawksford (nc=64)...\n", n_samples);
        int rc = gpu_cuda_trellis_hawksford(ctx, smoothed, output, n_samples);
        printf("Done, rc=%d\n", rc);
    } else {
        gpu_cuda_trellis_setup(ctx, 4, f->order, 256, f->a, f->g, 0.0);
        printf("Processing %zu samples through GPU Trellis...\n", n_samples);
        int rc = gpu_trellis_process(ctx, smoothed, output, n_samples,
                                      NULL, NULL, 4, f->order, f->a, f->g);
        printf("Done, rc=%d\n", rc);
    }
    gpu_destroy(ctx);
    free(smoothed);

    /* Write output WAV */
    FILE *fout = fopen(out_path, "wb");
    if (!fout) { printf("Cannot open %s\n", out_path); return 1; }
    /* WAV header: 32-bit float, mono, 2822400 Hz */
    uint32_t data_size = (uint32_t)(n_samples * sizeof(float));
    uint32_t file_size = 36 + data_size;
    uint32_t sr = 2822400;
    fwrite("RIFF", 1, 4, fout); fwrite(&file_size, 4, 1, fout);
    fwrite("WAVE", 1, 4, fout);
    fwrite("fmt ", 1, 4, fout);
    uint32_t fmt_size = 16; fwrite(&fmt_size, 4, 1, fout);
    uint16_t fmt_tag = 3; fwrite(&fmt_tag, 2, 1, fout);  /* float */
    uint16_t n_ch = 1; fwrite(&n_ch, 2, 1, fout);
    fwrite(&sr, 4, 1, fout);
    uint32_t byte_rate = sr * 4; fwrite(&byte_rate, 4, 1, fout);
    uint16_t block_align = 4; fwrite(&block_align, 2, 1, fout);
    uint16_t bps = 32; fwrite(&bps, 2, 1, fout);
    fwrite("data", 1, 4, fout); fwrite(&data_size, 4, 1, fout);
    fwrite(output, sizeof(float), n_samples, fout);
    fclose(fout);
    printf("Output: %s (%zu samples)\n", out_path, n_samples);

    free(input); free(output);
    return 0;
}
