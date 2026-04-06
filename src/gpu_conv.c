/*
 * foo_dsd_trellis — GPU Convolution via cuFFT + custom kernels
 *
 * Implements UPOLS at full DSD rate on GPU. The IR partitions and FDL
 * live in device memory. cuFFT handles FFT/IFFT, custom kernels handle
 * complex multiply-accumulate. FIFO logic mirrors the CPU path.
 *
 * This file is compiled as C (not CUDA) — all GPU calls go through
 * delay-loaded function pointers resolved in gpu_cuda.c.
 */

#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include "../include/gpu_compute.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

extern void trellis_log_c(const char *msg);

/* These are defined in gpu_cuda.c — we access via the dispatcher in gpu_compute.c */
int gpu_cuda_conv_max_partitions(void *ctx, uint32_t signal_rate, int P);
void *gpu_cuda_conv_init(void *ctx, int num_partitions,
                                      int partition_size, int fft_size,
                                      const void *ir_freq_host);
int gpu_cuda_conv_process(void *ctx, void *state,
                           double *buf, size_t count);
void gpu_cuda_conv_free(void *ctx, void *state);

/* ─── Public API (dispatches to CUDA backend) ─── */

int gpu_conv_max_partitions(gpu_context_t *ctx, uint32_t signal_rate,
                             int partition_size) {
    if (!ctx) return 0;
    /* Only CUDA backend supports convolution */
    return gpu_cuda_conv_max_partitions(ctx, signal_rate, partition_size);
}

gpu_conv_state_t *gpu_conv_init(gpu_context_t *ctx, int num_partitions,
                                 int partition_size, int fft_size,
                                 const void *ir_freq) {
    if (!ctx) return NULL;
    return (gpu_conv_state_t *)gpu_cuda_conv_init(ctx, num_partitions,
                                                    partition_size, fft_size, ir_freq);
}

int gpu_conv_process(gpu_context_t *ctx, void *state,
                      double *buf, size_t count) {
    if (!ctx || !state) return -1;
    return gpu_cuda_conv_process(ctx, state, buf, count);
}

void gpu_conv_free(gpu_context_t *ctx, void *state) {
    if (!ctx || !state) return;
    gpu_cuda_conv_free(ctx, state);
}
