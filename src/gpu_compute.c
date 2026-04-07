/*
 * foo_dsd_trellis — GPU Compute backend dispatcher
 *
 * Probes for available GPU backends (CUDA, DirectCompute) via delay-loading.
 * Routes all gpu_* API calls to the active backend.
 * Falls back gracefully when no GPU is available.
 */

#include "../include/gpu_compute.h"
#include <windows.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ─── Probe cache ─── */

static bool g_cuda_probed = false;
static bool g_dx12_probed = false;
static bool g_dx11_probed = false;
static bool g_cuda_available = false;
static bool g_dx12_available = false;
static bool g_dx11_available = false;
static gpu_backend_t g_active_backend = GPU_BACKEND_NONE;

static void log_probe_result(const char *which, int avail, double ms) {
    extern void trellis_log_c(const char *);
    char msg[128];
    sprintf_s(msg, sizeof(msg), "GPU probe: %s=%d (%.0fms)", which, avail, ms);
    trellis_log_c(msg);
}

static bool probe_cuda(void) {
    if (g_cuda_probed) return g_cuda_available;
    LARGE_INTEGER f, t0, t1;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t0);
    g_cuda_available = gpu_cuda_probe();
    QueryPerformanceCounter(&t1);
    g_cuda_probed = true;
    log_probe_result("cuda", g_cuda_available,
                     (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)f.QuadPart);
    return g_cuda_available;
}

static bool probe_dx12(void) {
    if (g_dx12_probed) return g_dx12_available;
    LARGE_INTEGER f, t0, t1;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t0);
    g_dx12_available = gpu_dx12_probe();
    QueryPerformanceCounter(&t1);
    g_dx12_probed = true;
    log_probe_result("dx12", g_dx12_available,
                     (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)f.QuadPart);
    return g_dx12_available;
}

static bool probe_dx11(void) {
    if (g_dx11_probed) return g_dx11_available;
    LARGE_INTEGER f, t0, t1;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t0);
    g_dx11_available = gpu_dx11_probe();
    QueryPerformanceCounter(&t1);
    g_dx11_probed = true;
    log_probe_result("dx11", g_dx11_available,
                     (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)f.QuadPart);
    return g_dx11_available;
}

bool gpu_available(gpu_backend_t preferred) {
    /* Probe ONLY what's needed for the requested backend.
     * AUTO short-circuits on first available (CUDA → DX12 → DX11). */
    switch (preferred) {
    case GPU_BACKEND_CUDA:
        return probe_cuda();
    case GPU_BACKEND_DIRECTX:
        return probe_dx12() || probe_dx11();
    case GPU_BACKEND_AUTO:
        if (probe_cuda()) return true;
        if (probe_dx12()) return true;
        return probe_dx11();
    default:
        return false;
    }
}

void gpu_get_info(gpu_info_t *info) {
    memset(info, 0, sizeof(*info));
    info->backend = GPU_BACKEND_NONE;

    /* Probe lazily — only check what's needed (matches gpu_available). */
    if (probe_cuda()) {
        gpu_cuda_get_info(info);
        return;
    }
    if (probe_dx12()) {
        gpu_dx12_get_info(info);
        return;
    }
    if (probe_dx11()) {
        gpu_dx11_get_info(info);
        return;
    }
}

/* ─── Resolve backend for creation ─── */

static gpu_backend_t resolve_backend(gpu_backend_t preferred) {
    switch (preferred) {
    case GPU_BACKEND_CUDA:
        return probe_cuda() ? GPU_BACKEND_CUDA : GPU_BACKEND_NONE;
    case GPU_BACKEND_DIRECTX:
        return (probe_dx12() || probe_dx11()) ? GPU_BACKEND_DIRECTX : GPU_BACKEND_NONE;
    case GPU_BACKEND_AUTO:
        if (probe_cuda()) return GPU_BACKEND_CUDA;
        if (probe_dx12() || probe_dx11()) return GPU_BACKEND_DIRECTX;
        return GPU_BACKEND_NONE;
    default:
        return GPU_BACKEND_NONE;
    }
}

/* ─── Lifecycle ─── */

gpu_context_t *gpu_create(gpu_backend_t backend) {
    gpu_backend_t resolved = resolve_backend(backend);
    {
        extern void trellis_log_c(const char *);
        char msg[128];
        sprintf_s(msg, sizeof(msg), "gpu_create: requested=%d resolved=%d cuda_avail=%d",
                  backend, resolved, g_cuda_available);
        trellis_log_c(msg);
    }

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

void gpu_reset_chunk(gpu_context_t *ctx) {
    if (!ctx) return;
    switch (ctx_backend(ctx)) {
    case GPU_BACKEND_CUDA:
        gpu_cuda_reset_chunk(ctx);
        break;
    case GPU_BACKEND_DIRECTX:
        gpu_dx11_reset_chunk(ctx);
        gpu_dx12_reset_chunk(ctx);
        break;
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

int gpu_fir_chain_process_f64(gpu_context_t *ctx, const float *in, double *out,
                               size_t in_count, size_t *out_count) {
    if (!ctx) return -1;
    switch (ctx_backend(ctx)) {
    case GPU_BACKEND_CUDA:
        return gpu_cuda_fir_chain_f64(ctx, in, out, in_count, out_count);
    default: {
        /* DX11/DX12 fallback: fp32 chain + widen to fp64 */
        float *tmp = (float *)malloc(in_count * 8 * sizeof(float));
        if (!tmp) return -1;
        size_t n = 0;
        int r = gpu_fir_chain_process(ctx, in, tmp, in_count, &n, NULL, NULL);
        if (r == 0) {
            for (size_t i = 0; i < n; i++)
                out[i] = (double)tmp[i];
            *out_count = n;
        }
        free(tmp);
        return r;
    }
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

int gpu_fir_lowpass_setup(gpu_context_t *ctx, const float *taps, int ntaps) {
    if (!ctx) return -1;
    switch (ctx_backend(ctx)) {
    case GPU_BACKEND_DIRECTX: {
        int r = gpu_dx12_fir_lowpass_setup(ctx, taps, ntaps);
        if (r == 0) return 0;
        return gpu_dx11_fir_lowpass_setup(ctx, taps, ntaps);
    }
    case GPU_BACKEND_CUDA:
        return gpu_cuda_fir_lowpass_setup(ctx, taps, ntaps);
    default: return -1;
    }
}

int gpu_fir_lowpass(gpu_context_t *ctx, const float *in, float *out,
                    size_t count, float gain) {
    if (!ctx) return -1;
    switch (ctx_backend(ctx)) {
    case GPU_BACKEND_DIRECTX: {
        int r = gpu_dx12_fir_lowpass(ctx, in, out, count, gain);
        if (r == 0) return 0;
        return gpu_dx11_fir_lowpass(ctx, in, out, count, gain);
    }
    case GPU_BACKEND_CUDA:
        return gpu_cuda_fir_lowpass(ctx, in, out, count, gain);
    default: return -1;
    }
}

int gpu_fir_lowpass_f64(gpu_context_t *ctx, const float *in, double *out,
                        size_t count, double gain) {
    if (!ctx) return -1;
    switch (ctx_backend(ctx)) {
    case GPU_BACKEND_CUDA:
        return gpu_cuda_fir_lowpass_f64(ctx, in, out, count, gain);
    case GPU_BACKEND_DIRECTX: {
        /* DX12/DX11 lowpass is float32 only. Run float lowpass then
         * widen to double. Uses static TLS buffer to avoid per-call malloc. */
        static __declspec(thread) float *tls_f32 = NULL;
        static __declspec(thread) size_t tls_cap = 0;
        if (tls_cap < count) {
            free(tls_f32);
            tls_f32 = (float *)malloc(count * sizeof(float));
            tls_cap = tls_f32 ? count : 0;
        }
        if (!tls_f32) return -1;
        int r = gpu_dx12_fir_lowpass(ctx, in, tls_f32, count, (float)gain);
        if (r != 0)
            r = gpu_dx11_fir_lowpass(ctx, in, tls_f32, count, (float)gain);
        if (r != 0) return r;
        for (size_t i = 0; i < count; i++)
            out[i] = (double)tls_f32[i];
        return 0;
    }
    default: return -1;
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
        int r = gpu_dx12_trellis_full(ctx, in, out, count, 8 /* segments */);
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
