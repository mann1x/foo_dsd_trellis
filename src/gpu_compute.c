/*
 * foo_dsd_trellis — GPU Compute backend dispatcher
 *
 * Probes for available GPU backends (CUDA, DirectCompute) via delay-loading.
 * Routes all gpu_* API calls to the active backend.
 * Falls back gracefully when no GPU is available.
 */

#include "../include/gpu_compute.h"
#include <string.h>

/* ─── Probe cache ─── */

static bool g_probed = false;
static bool g_cuda_available = false;
static bool g_dx12_available = false;
static bool g_dx11_available = false;
static gpu_backend_t g_active_backend = GPU_BACKEND_NONE;

bool gpu_available(gpu_backend_t preferred) {
    if (!g_probed) {
        g_cuda_available = gpu_cuda_probe();
        g_dx12_available = gpu_dx12_probe();     /* prefer DX12 over DX11 */
        g_dx11_available = gpu_dx11_probe();
        g_probed = true;
    }

    switch (preferred) {
    case GPU_BACKEND_CUDA:
        return g_cuda_available;
    case GPU_BACKEND_DIRECTX:
        return g_dx12_available || g_dx11_available;
    case GPU_BACKEND_AUTO:
        return g_cuda_available || (g_dx12_available || g_dx11_available);
    default:
        return false;
    }
}

void gpu_get_info(gpu_info_t *info) {
    memset(info, 0, sizeof(*info));
    info->backend = GPU_BACKEND_NONE;

    if (!g_probed) {
        gpu_available(GPU_BACKEND_AUTO);
    }

    /* Prefer CUDA info if available */
    if (g_cuda_available) {
        gpu_cuda_get_info(info);
        return;
    }
    if (g_dx12_available) {
        gpu_dx12_get_info(info);
        return;
    }
    if (g_dx11_available) {
        gpu_dx11_get_info(info);
        return;
    }
}

/* ─── Resolve backend for creation ─── */

static gpu_backend_t resolve_backend(gpu_backend_t preferred) {
    if (!g_probed)
        gpu_available(GPU_BACKEND_AUTO);

    switch (preferred) {
    case GPU_BACKEND_CUDA:
        return g_cuda_available ? GPU_BACKEND_CUDA : GPU_BACKEND_NONE;
    case GPU_BACKEND_DIRECTX:
        return (g_dx12_available || g_dx11_available) ? GPU_BACKEND_DIRECTX : GPU_BACKEND_NONE;
    case GPU_BACKEND_AUTO:
        if (g_cuda_available) return GPU_BACKEND_CUDA;
        if (g_dx12_available || g_dx11_available) return GPU_BACKEND_DIRECTX;
        return GPU_BACKEND_NONE;
    default:
        return GPU_BACKEND_NONE;
    }
}

/* ─── Lifecycle ─── */

gpu_context_t *gpu_create(gpu_backend_t backend) {
    gpu_backend_t resolved = resolve_backend(backend);

    switch (resolved) {
    case GPU_BACKEND_CUDA:
        g_active_backend = GPU_BACKEND_CUDA;
        return gpu_cuda_create();
    case GPU_BACKEND_DIRECTX:
        g_active_backend = GPU_BACKEND_DIRECTX;
        /* Prefer DX12 (async compute), fall back to DX11 */
        if (g_dx12_available) {
            gpu_context_t *ctx = gpu_dx12_create_full();
            if (ctx) return ctx;
        }
        return gpu_dx11_create();
    default:
        return NULL;
    }
}

/* ─── Backend dispatch helper ─── */

/* The first field of every backend context is gpu_backend_t */
typedef struct { gpu_backend_t backend; } gpu_base_t;

static gpu_backend_t ctx_backend(gpu_context_t *ctx) {
    return ((gpu_base_t *)ctx)->backend;
}

void gpu_destroy(gpu_context_t *ctx) {
    if (!ctx) return;
    /* DX12 and DX11 both use GPU_BACKEND_DIRECTX tag.
     * Check if it's a DX12 context by trying DX12 destroy first. */
    switch (ctx_backend(ctx)) {
    case GPU_BACKEND_DIRECTX:
        /* DX12 contexts have different internal structure but same backend tag.
         * For now, try both destroys — only one will match. */
        gpu_dx12_destroy_full(ctx);
        break;
    case GPU_BACKEND_CUDA:    gpu_cuda_destroy(ctx); break;
    default: break;
    }
}

/* ─── Dispatched implementations ─── */

int gpu_fir_setup(gpu_context_t *ctx, const float *taps, int ntaps,
                  int num_stages, bool upsample) {
    if (!ctx) return -1;
    switch (ctx_backend(ctx)) {
    case GPU_BACKEND_DIRECTX: {
        int r = gpu_dx12_fir_setup(ctx, taps, ntaps, num_stages, upsample);
        if (r == 0) return 0;
        return gpu_dx11_fir_setup(ctx, taps, ntaps, num_stages, upsample);
    }
    case GPU_BACKEND_CUDA:
        return gpu_cuda_fir_setup(ctx, taps, ntaps, num_stages, upsample);
    default: return -1;
    }
}

int gpu_fir_chain_process(gpu_context_t *ctx, const float *in, float *out,
                          size_t in_count, size_t *out_count,
                          const float *delay_in, float *delay_out) {
    if (!ctx) return -1;
    (void)delay_in; (void)delay_out; /* Delay handled internally by backend */
    switch (ctx_backend(ctx)) {
    case GPU_BACKEND_DIRECTX: {
        int r = gpu_dx12_fir_chain(ctx, in, out, in_count, out_count);
        if (r == 0) return 0;
        return gpu_dx11_fir_chain(ctx, in, out, in_count, out_count);
    }
    case GPU_BACKEND_CUDA:
        return gpu_cuda_fir_chain(ctx, in, out, in_count, out_count);
    default: return -1;
    }
}

int gpu_fir_batch_process(gpu_context_t *ctx, const float *in_batch,
                          float *out_batch, size_t samples_per_ch,
                          int num_channels, size_t *out_count_per_ch) {
    if (!ctx) return -1;
    /* True batched dispatch for CUDA, sequential fallback for DX11 */
    switch (ctx_backend(ctx)) {
    case GPU_BACKEND_CUDA:
        return gpu_cuda_fir_batch(ctx, in_batch, out_batch,
                                   samples_per_ch, num_channels, out_count_per_ch);
    default: {
        /* Sequential per-channel fallback */
        for (int ch = 0; ch < num_channels; ch++) {
            size_t out_n = 0;
            int r = gpu_fir_chain_process(ctx,
                in_batch + ch * samples_per_ch,
                out_batch + ch * (*out_count_per_ch),
                samples_per_ch, &out_n, NULL, NULL);
            if (r != 0) return -1;
            if (ch == 0) *out_count_per_ch = out_n;
        }
        return 0;
    }
    }
}

int gpu_gain_apply(gpu_context_t *ctx, float *buf, size_t count, float gain) {
    if (!ctx) return -1;
    switch (ctx_backend(ctx)) {
    case GPU_BACKEND_DIRECTX: {
        int r = gpu_dx12_gain(ctx, buf, count, gain);
        if (r == 0) return 0;
        return gpu_dx11_gain(ctx, buf, count, gain);
    }
    case GPU_BACKEND_CUDA:
        return gpu_cuda_gain(ctx, buf, count, gain);
    default: return -1;
    }
}

int gpu_boxcar_smooth(gpu_context_t *ctx, const float *in, float *out,
                      size_t count, int taps, float gain) {
    if (!ctx) return -1;
    switch (ctx_backend(ctx)) {
    case GPU_BACKEND_DIRECTX: {
        int r = gpu_dx12_boxcar(ctx, in, out, count, taps, gain);
        if (r == 0) return 0;
        return gpu_dx11_boxcar(ctx, in, out, count, taps, gain);
    }
    case GPU_BACKEND_CUDA:
        return gpu_cuda_boxcar(ctx, in, out, count, taps, gain);
    default: return -1;
    }
}

int gpu_trellis_process(gpu_context_t *ctx, const float *in, float *out,
                        size_t count, const void *sdm_state_in,
                        void *sdm_state_out, int num_cands, int order,
                        const double *ntf_a, const double *ntf_g) {
    if (!ctx || num_cands < 2) return -1;
    (void)sdm_state_in; (void)sdm_state_out; /* state on device now */
    (void)num_cands; (void)order; (void)ntf_a; (void)ntf_g;
    switch (ctx_backend(ctx)) {
    case GPU_BACKEND_CUDA:
        return gpu_cuda_trellis(ctx, in, out, count);
    case GPU_BACKEND_DIRECTX: {
        /* Try DX12 parallel-segment trellis first */
        int r = gpu_dx12_trellis_full(ctx, in, out, count, 64 /* segments */);
        if (r == 0) return 0;
        return gpu_dx11_trellis(ctx, in, out, count);
    }
    default: return -1;
    }
}

int gpu_precorr_process(gpu_context_t *ctx, const float *in, float *out,
                        size_t count, const float *ntf_a, const float *ntf_g,
                        int order, const float pred_table[256][8],
                        const gpu_precorr_state_t *init,
                        gpu_precorr_state_t *final_state) {
    if (!ctx) return -1;
    (void)ntf_a; (void)ntf_g; (void)order; (void)pred_table; /* setup done earlier */
    switch (ctx_backend(ctx)) {
    case GPU_BACKEND_CUDA:
        return gpu_cuda_precorr(ctx, in, out, count, init, final_state);
    default: return -1;
    }
}
