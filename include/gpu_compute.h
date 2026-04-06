/*
 * foo_dsd_trellis — GPU Compute Offload API
 *
 * Abstract GPU compute interface for FIR convolution, gain, boxcar, and SDM.
 * Two backends: DirectCompute (D3D11 cs_5_0) and CUDA (driver API).
 * Both delay-loaded — plugin works without any GPU driver installed.
 *
 * All functions return 0 on success, -1 on failure (caller falls back to CPU).
 */

#ifndef GPU_COMPUTE_H
#define GPU_COMPUTE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpu_context gpu_context_t;

typedef enum {
    GPU_BACKEND_NONE     = 0,   /* GPU disabled */
    GPU_BACKEND_DIRECTX  = 1,   /* D3D11 Compute Shader cs_5_0 */
    GPU_BACKEND_CUDA     = 2,   /* CUDA Driver API (nvcuda.dll) */
    GPU_BACKEND_AUTO     = 3,   /* Try CUDA first, then DirectX */
} gpu_backend_t;

typedef struct {
    gpu_backend_t backend;          /* Active backend */
    char          device_name[128]; /* GPU device name */
    size_t        vram_mb;          /* Available VRAM in MB */
    bool          available;        /* true if GPU is usable */
} gpu_info_t;

/* Minimum sample count for GPU to be faster than CPU.
 * Below this, kernel launch + transfer overhead exceeds CPU FIR time. */
#define GPU_MIN_SAMPLES 8192

/* ─── Probe & Info ─── */

/* Probe GPU availability. Thread-safe, caches result. */
bool gpu_available(gpu_backend_t preferred);

/* Get info about the active GPU (call after gpu_available). */
void gpu_get_info(gpu_info_t *info);

/* ─── Lifecycle ─── */

/* Create a GPU compute context. Returns NULL on failure. */
gpu_context_t *gpu_create(gpu_backend_t backend);

/* Destroy GPU context. Safe to call with NULL. */
void gpu_destroy(gpu_context_t *ctx);

/* Reset per-chunk state (boxcar channel counter). Call at start of each audio chunk. */
void gpu_reset_chunk(gpu_context_t *ctx);

/* ─── FIR Convolution ─── */

/* Upload FIR coefficients (call once per rate configuration).
 * taps: filter coefficients, ntaps: tap count (63),
 * num_stages: number of 2x up/downsample stages,
 * upsample: true for upsample chain, false for downsample. */
int gpu_fir_setup(gpu_context_t *ctx, const float *taps, int ntaps,
                  int num_stages, bool upsample);

/* Process FIR chain for one channel.
 * delay_in/delay_out: FIR delay line state (62 floats per stage). */
int gpu_fir_chain_process(gpu_context_t *ctx, const float *in, float *out,
                          size_t in_count, size_t *out_count,
                          const float *delay_in, float *delay_out);

/* Process FIR chain in fp64. Input is float (DSD ±1.0), widened to double
 * internally. Output is double. CUDA only — DX11/DX12 fall back to fp32+widen. */
int gpu_fir_chain_process_f64(gpu_context_t *ctx, const float *in, double *out,
                               size_t in_count, size_t *out_count);

/* Batched FIR: process all channels in one kernel launch.
 * in_batch/out_batch: contiguous buffers [ch0_data | ch1_data | ...]. */
int gpu_fir_batch_process(gpu_context_t *ctx, const float *in_batch,
                          float *out_batch, size_t samples_per_ch,
                          int num_channels, size_t *out_count_per_ch);

/* ─── FIR Lowpass (same-rate pre-SDM filter) ─── */

/* Upload FIR lowpass coefficients (call once per rate configuration).
 * taps: filter coefficients, ntaps: tap count (e.g. 63).
 * Separate from rate conversion FIR taps. */
int gpu_fir_lowpass_setup(gpu_context_t *ctx, const float *taps, int ntaps);

/* Process FIR lowpass for one channel (same-rate, no up/downsampling).
 * Input is ±1.0 DSD, output is multi-bit smoothed + gained.
 * gain: combined FIR gain × volume gain. */
int gpu_fir_lowpass(gpu_context_t *ctx, const float *in, float *out,
                    size_t count, float gain);

/* FIR lowpass with fp64 output (for SDM pipeline).
 * CUDA: native fp64. DX12/DX11: float lowpass → widen to double. */
int gpu_fir_lowpass_f64(gpu_context_t *ctx, const float *in, double *out,
                        size_t count, double gain);

/* ─── Gain & Boxcar ─── */

/* Apply gain multiply in-place. */
int gpu_gain_apply(gpu_context_t *ctx, float *buf, size_t count, float gain);

/* Boxcar smoothing + gain in one pass.
 * Converts ±1.0 DSD → multi-bit via running average, applies gain. */
int gpu_boxcar_smooth(gpu_context_t *ctx, const float *in, float *out,
                      size_t count, int taps, float gain);

/* ─── Trellis SDM (cands ≥ 16 only) ─── */

/* Process chunk through Trellis SDM on GPU using sub-chunking.
 * Subdivides into small GPU dispatches (sub_chunk_size samples each)
 * to prevent GPU lockup. State persists on device between sub-chunks. */
int gpu_trellis_process(gpu_context_t *ctx, const float *in, float *out,
                        size_t count, const void *sdm_state_in,
                        void *sdm_state_out, int num_cands, int order,
                        const double *ntf_a, const double *ntf_g);

#define GPU_SDM_SUB_CHUNK 4096  /* samples per GPU SDM dispatch */

/* ─── PreCorr SDM ─── */

/* Per-channel PreCorr initial/final conditions (matches GPU struct layout) */
typedef struct {
    float state[8];
    float prev_y;
    int   history;
    int   phase;
    float pad;
} gpu_precorr_state_t;

/* Process entire chunk through PreCorr SDM on GPU.
 * Sequential loop, one thread per channel for multi-channel batch. */
int gpu_precorr_process(gpu_context_t *ctx, const float *in, float *out,
                        size_t count, const float *ntf_a, const float *ntf_g,
                        int order, const float pred_table[256][8],
                        const gpu_precorr_state_t *init,
                        gpu_precorr_state_t *final_state);

/* ─── Backend probe functions (implemented in gpu_dx11.c / gpu_cuda.c) ─── */

bool gpu_dx11_probe(void);
bool gpu_cuda_probe(void);

void gpu_dx11_get_info(gpu_info_t *info);
void gpu_cuda_get_info(gpu_info_t *info);

gpu_context_t *gpu_dx11_create(void);
gpu_context_t *gpu_cuda_create(void);

void gpu_dx11_destroy(void *ctx);
void gpu_cuda_destroy(void *ctx);
void gpu_cuda_reset_chunk(void *ctx);
void gpu_dx11_reset_chunk(void *ctx);
void gpu_dx12_reset_chunk(void *ctx);
void gpu_dx12_destroy_full(void *ctx);

/* DX12 full backend */
bool gpu_dx12_probe(void);
void gpu_dx12_get_info(gpu_info_t *info);
gpu_context_t *gpu_dx12_create_full(void);
int gpu_dx12_fir_setup(void *ctx, const float *taps, int ntaps,
                        int num_stages, bool upsample);
int gpu_dx12_fir_chain(void *ctx, const float *in, float *out,
                        size_t in_count, size_t *out_count);
int gpu_dx12_gain(void *ctx, float *buf, size_t count, float gain);
int gpu_dx12_boxcar(void *ctx, const float *in, float *out,
                     size_t count, int taps, float gain);
int gpu_dx12_trellis_setup_full(void *ctx, int nc, int ord, int lat,
                                  const double *a, const double *g, double sl);
int gpu_dx12_trellis_full(void *ctx, const float *in, float *out,
                           size_t count, int num_segs);

/* Backend-specific operations (called by gpu_compute.c dispatcher) */
int gpu_dx11_fir_lowpass_setup(void *ctx, const float *taps, int ntaps);
int gpu_dx11_fir_lowpass(void *ctx, const float *in, float *out,
                          size_t count, float gain);
int gpu_dx12_fir_lowpass_setup(void *ctx, const float *taps, int ntaps);
int gpu_dx12_fir_lowpass(void *ctx, const float *in, float *out,
                          size_t count, float gain);

int gpu_dx11_fir_setup(void *ctx, const float *taps, int ntaps,
                        int num_stages, bool upsample);
int gpu_dx11_fir_chain(void *ctx, const float *in, float *out,
                        size_t in_count, size_t *out_count);
int gpu_dx11_gain(void *ctx, float *buf, size_t count, float gain);
int gpu_dx11_boxcar(void *ctx, const float *in, float *out,
                     size_t count, int taps, float gain);

int gpu_cuda_fir_lowpass_setup(void *ctx, const float *taps, int ntaps);
int gpu_cuda_fir_lowpass(void *ctx, const float *in, float *out,
                          size_t count, float gain);
int gpu_cuda_fir_lowpass_f64(void *ctx, const float *in, double *out,
                               size_t count, double gain);

int gpu_cuda_fir_setup(void *ctx, const float *taps, int ntaps,
                        int num_stages, bool upsample);
int gpu_cuda_fir_chain(void *ctx, const float *in, float *out,
                        size_t in_count, size_t *out_count);
int gpu_cuda_fir_chain_f64(void *ctx, const float *in, double *out,
                            size_t in_count, size_t *out_count);
int gpu_cuda_gain(void *ctx, float *buf, size_t count, float gain);
int gpu_cuda_boxcar(void *ctx, const float *in, float *out,
                     size_t count, int taps, float gain);
int gpu_cuda_fir_batch(void *ctx, const float *in_batch,
                        float *out_batch, size_t samples_per_ch,
                        int num_channels, size_t *out_count_per_ch);
/* Persistent SDM setup (call once at engine init) */
int gpu_dx11_trellis(void *ctx, const float *in, float *out, size_t count);
int gpu_dx11_trellis_setup(void *ctx, int num_cands, int order,
                            int trellis_lat, const double *ntf_a,
                            const double *ntf_g, double state_limit);

int gpu_cuda_trellis_setup(void *ctx, int num_cands, int order,
                            int trellis_lat, const double *ntf_a,
                            const double *ntf_g, double state_limit,
                            int trellis_depth);
int gpu_cuda_precorr_setup(void *ctx, int order,
                            const float *ntf_a, const float *ntf_g,
                            const float *pred_table, float state_limit);

/* Persistent-buffer SDM process (no per-chunk alloc) */
int gpu_cuda_trellis(void *ctx, const double *in, float *out, size_t count);
/* DAS (Density-Aligned Stitching) GPU pipeline: SDM + stitch + assemble.
 * All channels processed simultaneously. Input/output: [num_ch × count] */
int gpu_cuda_trellis_das(void *ctx, const double *in, float *out,
                          size_t count, int num_channels);
/* Upload extended seed (history/path/traceback) for full state continuity */
int gpu_cuda_upload_ext_seed(void *ctx,
                              const unsigned char *hist, int hist_bytes, int nc,
                              const unsigned *path, const unsigned *next_stored,
                              int hist_pos, int pending);
/* Hybrid CPU+GPU: launch GPU segments async, finish downloads results */
int gpu_cuda_hybrid_async(void *ctx, const double *in, size_t count,
                           int num_channels,
                           const int *seg_starts, const int *seg_out_starts,
                           const int *seg_total_sizes, const int *seg_out_caps,
                           int num_gpu_segs, int overlap, size_t total_seg_out);
int gpu_cuda_hybrid_pass2(void *ctx, const double *cpu_states, const double *cpu_costs);
int gpu_cuda_hybrid_finish(void *ctx, float **seg_bufs, size_t *seg_out_counts,
                            int num_gpu_segs, int num_channels);
int gpu_cuda_precorr(void *ctx, const float *in, float *out, size_t count,
                      const gpu_precorr_state_t *init,
                      gpu_precorr_state_t *final_state);

/* ─── Multi-bit SDM experiment ─── */
int gpu_cuda_multibit_setup(void *ctx, int order,
                             const double *ntf_a, const double *ntf_g,
                             double state_limit, int num_levels);
int gpu_cuda_multibit_process(void *ctx, const double *in, float *out,
                               size_t count, int num_channels, float gain);

/* ─── GPU Convolution (cuFFT + custom kernels) ─── */

/* Opaque GPU convolution state */
typedef struct gpu_conv_state gpu_conv_state_t;

/* Query GPU convolution capability: max partitions for given rate.
 * Returns 0 if GPU convolution not available (no cuFFT). */
/* budget_level: 0=High (full), 1=Medium (50%), 2=Low (25%) */
int gpu_conv_max_partitions(gpu_context_t *ctx, uint32_t signal_rate,
                             int partition_size, int budget_level);

/* Initialize GPU convolution for one channel.
 * ir_freq: pre-FFT'd IR partitions (complex double, host memory)
 * Returns NULL on failure. */
gpu_conv_state_t *gpu_conv_init(gpu_context_t *ctx, int num_partitions,
                                 int partition_size, int fft_size,
                                 const void *ir_freq);

/* Process one block: convolve buf in-place at signal rate.
 * buf: fp64 samples (host), count samples. */
int gpu_conv_process(gpu_context_t *ctx, void *state,
                      double *buf, size_t count);

/* Free GPU convolution state. */
void gpu_conv_free(gpu_context_t *ctx, void *state);

#ifdef __cplusplus
}
#endif

#endif /* GPU_COMPUTE_H */
