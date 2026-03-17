/*
 * foo_dsd_trellis — CUDA (Driver API) GPU backend
 *
 * Delay-loads nvcuda.dll via LoadLibrary/GetProcAddress.
 * PTX kernels embedded as C string (compiled offline with nvcc).
 * Triple-buffered async pipeline: 3 streams + pinned host memory.
 *
 * Implements: FIR upsample/downsample, gain, boxcar, Trellis SDM, PreCorr.
 */

#include "../include/gpu_compute.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <string.h>
#include <stdlib.h>

/* ─── Embedded PTX ─── */
#include "kernels/fir_kernels_ptx.h"
#include "kernels/sdm_kernels_ptx.h"

/* ─── CUDA Driver API types (no CUDA headers needed) ─── */

typedef int            CUresult;
typedef int            CUdevice;
typedef void          *CUcontext;
typedef void          *CUmodule;
typedef void          *CUfunction;
typedef unsigned long long CUdeviceptr;
typedef void          *CUstream;

#define CUDA_SUCCESS   0
#define CU_CTX_SCHED_AUTO 0

/* ─── Delay-loaded function pointers ─── */

#define DECL_PFN(ret, name, ...) typedef ret (*PFN_##name)(__VA_ARGS__)

DECL_PFN(CUresult, cuInit, unsigned int);
DECL_PFN(CUresult, cuDeviceGet, CUdevice *, int);
DECL_PFN(CUresult, cuDeviceGetName, char *, int, CUdevice);
DECL_PFN(CUresult, cuDeviceTotalMem, size_t *, CUdevice);
DECL_PFN(CUresult, cuCtxCreate, CUcontext *, unsigned int, CUdevice);
DECL_PFN(CUresult, cuCtxDestroy, CUcontext);
DECL_PFN(CUresult, cuCtxSetCurrent, CUcontext);
DECL_PFN(CUresult, cuModuleLoadData, CUmodule *, const void *);
DECL_PFN(CUresult, cuModuleUnload, CUmodule);
DECL_PFN(CUresult, cuModuleGetFunction, CUfunction *, CUmodule, const char *);
DECL_PFN(CUresult, cuModuleGetGlobal, CUdeviceptr *, size_t *, CUmodule, const char *);
DECL_PFN(CUresult, cuLaunchKernel, CUfunction, unsigned, unsigned, unsigned,
         unsigned, unsigned, unsigned, unsigned, CUstream, void **, void **);
DECL_PFN(CUresult, cuMemAlloc, CUdeviceptr *, size_t);
DECL_PFN(CUresult, cuMemFree, CUdeviceptr);
DECL_PFN(CUresult, cuMemcpyHtoD, CUdeviceptr, const void *, size_t);
DECL_PFN(CUresult, cuMemcpyDtoH, void *, CUdeviceptr, size_t);
DECL_PFN(CUresult, cuMemcpyHtoDAsync, CUdeviceptr, const void *, size_t, CUstream);
DECL_PFN(CUresult, cuMemcpyDtoHAsync, void *, CUdeviceptr, size_t, CUstream);
DECL_PFN(CUresult, cuMemAllocHost, void **, size_t);
DECL_PFN(CUresult, cuMemFreeHost, void *);
DECL_PFN(CUresult, cuStreamCreate, CUstream *, unsigned int);
DECL_PFN(CUresult, cuStreamDestroy, CUstream);
DECL_PFN(CUresult, cuStreamSynchronize, CUstream);

static HMODULE g_nvcuda = NULL;
static bool    g_probed = false;
static bool    g_available = false;
static char    g_device_name[128] = "";
static size_t  g_vram_bytes = 0;

/* Resolved function pointers */
static PFN_cuInit              pfn_cuInit;
static PFN_cuDeviceGet         pfn_cuDeviceGet;
static PFN_cuDeviceGetName     pfn_cuDeviceGetName;
static PFN_cuDeviceTotalMem    pfn_cuDeviceTotalMem;
static PFN_cuCtxCreate         pfn_cuCtxCreate;
static PFN_cuCtxDestroy        pfn_cuCtxDestroy;
static PFN_cuCtxSetCurrent     pfn_cuCtxSetCurrent;
static PFN_cuModuleLoadData    pfn_cuModuleLoadData;
static PFN_cuModuleUnload      pfn_cuModuleUnload;
static PFN_cuModuleGetFunction pfn_cuModuleGetFunction;
static PFN_cuModuleGetGlobal   pfn_cuModuleGetGlobal;
static PFN_cuLaunchKernel      pfn_cuLaunchKernel;
static PFN_cuMemAlloc          pfn_cuMemAlloc;
static PFN_cuMemFree           pfn_cuMemFree;
static PFN_cuMemcpyHtoD        pfn_cuMemcpyHtoD;
static PFN_cuMemcpyDtoH        pfn_cuMemcpyDtoH;
static PFN_cuMemcpyHtoDAsync   pfn_cuMemcpyHtoDAsync;
static PFN_cuMemcpyDtoHAsync   pfn_cuMemcpyDtoHAsync;
static PFN_cuMemAllocHost      pfn_cuMemAllocHost;
static PFN_cuMemFreeHost       pfn_cuMemFreeHost;
static PFN_cuStreamCreate      pfn_cuStreamCreate;
static PFN_cuStreamDestroy     pfn_cuStreamDestroy;
static PFN_cuStreamSynchronize pfn_cuStreamSynchronize;

#define RESOLVE(name) do { \
    pfn_##name = (PFN_##name)GetProcAddress(g_nvcuda, #name); \
    if (!pfn_##name) { ok = false; } \
} while(0)

#define RESOLVE_V2(name) do { \
    pfn_##name = (PFN_##name)GetProcAddress(g_nvcuda, #name "_v2"); \
    if (!pfn_##name) pfn_##name = (PFN_##name)GetProcAddress(g_nvcuda, #name); \
    if (!pfn_##name) { ok = false; } \
} while(0)

/* ─── CUDA context structure ─── */

#define NUM_STREAMS 3

typedef struct {
    gpu_backend_t  backend;        /* Must be first — GPU_BACKEND_CUDA */
    CUcontext      context;
    CUmodule       module;
    /* Kernel functions */
    CUfunction     fn_fir_up;
    CUfunction     fn_fir_down;
    CUfunction     fn_gain;
    CUfunction     fn_boxcar;
    /* Batched multi-channel kernels */
    CUfunction     fn_fir_up_batch;
    CUfunction     fn_fir_down_batch;
    CUfunction     fn_gain_batch;
    /* SDM kernels (from sdm_kernels.ptx) */
    CUmodule       mod_sdm;
    CUfunction     fn_trellis_chunk;
    CUfunction     fn_precorr_chunk;
    /* Triple-buffered device memory */
    CUdeviceptr    d_in[NUM_STREAMS];
    CUdeviceptr    d_out[NUM_STREAMS];
    size_t         cap_in;         /* floats per buffer */
    size_t         cap_out;
    /* Pinned host memory for async transfers */
    float         *h_in[NUM_STREAMS];
    float         *h_out[NUM_STREAMS];
    size_t         cap_h_in;
    size_t         cap_h_out;
    /* Streams */
    CUstream       streams[NUM_STREAMS];
    int            active;         /* Current stream index (0..2) */
    /* FIR config */
    int            num_stages;
    bool           upsample;
    int            ntaps;
    /* Intermediate device buffer for multi-stage */
    CUdeviceptr    d_inter;
    size_t         cap_inter;
} cuda_context_t;

/* ─── Helpers ─── */

static int ensure_d_in(cuda_context_t *c, size_t floats) {
    if (c->cap_in >= floats) return 0;
    for (int i = 0; i < NUM_STREAMS; i++) {
        if (c->d_in[i]) pfn_cuMemFree(c->d_in[i]);
        if (pfn_cuMemAlloc(&c->d_in[i], floats * sizeof(float)) != CUDA_SUCCESS)
            return -1;
    }
    c->cap_in = floats;
    return 0;
}

static int ensure_d_out(cuda_context_t *c, size_t floats) {
    if (c->cap_out >= floats) return 0;
    for (int i = 0; i < NUM_STREAMS; i++) {
        if (c->d_out[i]) pfn_cuMemFree(c->d_out[i]);
        if (pfn_cuMemAlloc(&c->d_out[i], floats * sizeof(float)) != CUDA_SUCCESS)
            return -1;
    }
    c->cap_out = floats;
    return 0;
}

static int ensure_h_in(cuda_context_t *c, size_t floats) {
    if (c->cap_h_in >= floats) return 0;
    for (int i = 0; i < NUM_STREAMS; i++) {
        if (c->h_in[i]) pfn_cuMemFreeHost(c->h_in[i]);
        if (pfn_cuMemAllocHost((void **)&c->h_in[i], floats * sizeof(float)) != CUDA_SUCCESS)
            return -1;
    }
    c->cap_h_in = floats;
    return 0;
}

static int ensure_h_out(cuda_context_t *c, size_t floats) {
    if (c->cap_h_out >= floats) return 0;
    for (int i = 0; i < NUM_STREAMS; i++) {
        if (c->h_out[i]) pfn_cuMemFreeHost(c->h_out[i]);
        if (pfn_cuMemAllocHost((void **)&c->h_out[i], floats * sizeof(float)) != CUDA_SUCCESS)
            return -1;
    }
    c->cap_h_out = floats;
    return 0;
}

static int ensure_d_inter(cuda_context_t *c, size_t floats) {
    if (c->cap_inter >= floats) return 0;
    if (c->d_inter) pfn_cuMemFree(c->d_inter);
    if (pfn_cuMemAlloc(&c->d_inter, floats * sizeof(float)) != CUDA_SUCCESS)
        return -1;
    c->cap_inter = floats;
    return 0;
}

/* Launch a kernel with given params. block_size=256. */
static int launch_kernel(cuda_context_t *c, CUfunction fn,
                          void **args, int num_elements) {
    unsigned grid = (unsigned)((num_elements + 255) / 256);
    return pfn_cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0,
                               c->streams[c->active], args, NULL)
           == CUDA_SUCCESS ? 0 : -1;
}

/* ─── Probe ─── */

bool gpu_cuda_probe(void) {
    if (g_probed) return g_available;
    g_probed = true;

    g_nvcuda = LoadLibraryA("nvcuda.dll");
    if (!g_nvcuda) { g_available = false; return false; }

    bool ok = true;
    RESOLVE(cuInit);
    RESOLVE(cuDeviceGet);
    RESOLVE(cuDeviceGetName);
    RESOLVE_V2(cuDeviceTotalMem);
    RESOLVE_V2(cuCtxCreate);
    RESOLVE_V2(cuCtxDestroy);
    RESOLVE_V2(cuCtxSetCurrent);
    RESOLVE(cuModuleLoadData);
    RESOLVE(cuModuleUnload);
    RESOLVE(cuModuleGetFunction);
    RESOLVE_V2(cuModuleGetGlobal);
    RESOLVE(cuLaunchKernel);
    RESOLVE_V2(cuMemAlloc);
    RESOLVE_V2(cuMemFree);
    RESOLVE_V2(cuMemcpyHtoD);
    RESOLVE_V2(cuMemcpyDtoH);
    RESOLVE_V2(cuMemcpyHtoDAsync);
    RESOLVE_V2(cuMemcpyDtoHAsync);
    RESOLVE(cuMemAllocHost);
    RESOLVE(cuMemFreeHost);
    RESOLVE(cuStreamCreate);
    RESOLVE_V2(cuStreamDestroy);
    RESOLVE(cuStreamSynchronize);

    if (!ok) { g_available = false; return false; }

    if (pfn_cuInit(0) != CUDA_SUCCESS) { g_available = false; return false; }

    CUdevice dev = 0;
    if (pfn_cuDeviceGet(&dev, 0) != CUDA_SUCCESS) { g_available = false; return false; }

    pfn_cuDeviceGetName(g_device_name, sizeof(g_device_name), dev);
    pfn_cuDeviceTotalMem(&g_vram_bytes, dev);

    g_available = true;
    return true;
}

void gpu_cuda_get_info(gpu_info_t *info) {
    memset(info, 0, sizeof(*info));
    info->backend = GPU_BACKEND_CUDA;
    info->available = g_available;
    strncpy_s(info->device_name, sizeof(info->device_name), g_device_name, _TRUNCATE);
    info->vram_mb = g_vram_bytes / (1024 * 1024);
}

/* ─── Create / Destroy ─── */

gpu_context_t *gpu_cuda_create(void) {
    if (!g_available) return NULL;

    CUdevice dev = 0;
    if (pfn_cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return NULL;

    cuda_context_t *c = (cuda_context_t *)calloc(1, sizeof(cuda_context_t));
    if (!c) return NULL;
    c->backend = GPU_BACKEND_CUDA;

    if (pfn_cuCtxCreate(&c->context, CU_CTX_SCHED_AUTO, dev) != CUDA_SUCCESS)
        goto fail;

    /* Load PTX module */
    if (pfn_cuModuleLoadData(&c->module, g_ptx_fir_kernels) != CUDA_SUCCESS)
        goto fail;

    /* Resolve kernel functions */
    if (pfn_cuModuleGetFunction(&c->fn_fir_up, c->module, "fir_upsample_2x") != CUDA_SUCCESS)
        goto fail;
    if (pfn_cuModuleGetFunction(&c->fn_fir_down, c->module, "fir_downsample_2x") != CUDA_SUCCESS)
        goto fail;
    if (pfn_cuModuleGetFunction(&c->fn_gain, c->module, "gain_apply") != CUDA_SUCCESS)
        goto fail;
    if (pfn_cuModuleGetFunction(&c->fn_boxcar, c->module, "boxcar_smooth") != CUDA_SUCCESS)
        goto fail;
    /* Batched multi-channel kernels */
    if (pfn_cuModuleGetFunction(&c->fn_fir_up_batch, c->module, "fir_upsample_2x_batch") != CUDA_SUCCESS)
        goto fail;
    if (pfn_cuModuleGetFunction(&c->fn_fir_down_batch, c->module, "fir_downsample_2x_batch") != CUDA_SUCCESS)
        goto fail;
    if (pfn_cuModuleGetFunction(&c->fn_gain_batch, c->module, "gain_apply_batch") != CUDA_SUCCESS)
        goto fail;

    /* Load SDM PTX module */
    if (pfn_cuModuleLoadData(&c->mod_sdm, g_ptx_sdm_kernels) != CUDA_SUCCESS)
        goto fail;
    if (pfn_cuModuleGetFunction(&c->fn_trellis_chunk, c->mod_sdm, "trellis_chunk") != CUDA_SUCCESS)
        goto fail;
    if (pfn_cuModuleGetFunction(&c->fn_precorr_chunk, c->mod_sdm, "precorr_chunk") != CUDA_SUCCESS)
        goto fail;

    /* Create streams */
    for (int i = 0; i < NUM_STREAMS; i++) {
        if (pfn_cuStreamCreate(&c->streams[i], 0) != CUDA_SUCCESS)
            goto fail;
    }

    return (gpu_context_t *)c;

fail:
    gpu_cuda_destroy(c);
    return NULL;
}

void gpu_cuda_destroy(void *ptr) {
    cuda_context_t *c = (cuda_context_t *)ptr;
    if (!c) return;

    for (int i = 0; i < NUM_STREAMS; i++) {
        if (c->streams[i]) pfn_cuStreamDestroy(c->streams[i]);
        if (c->d_in[i])    pfn_cuMemFree(c->d_in[i]);
        if (c->d_out[i])   pfn_cuMemFree(c->d_out[i]);
        if (c->h_in[i])    pfn_cuMemFreeHost(c->h_in[i]);
        if (c->h_out[i])   pfn_cuMemFreeHost(c->h_out[i]);
    }
    if (c->d_inter) pfn_cuMemFree(c->d_inter);
    if (c->mod_sdm) pfn_cuModuleUnload(c->mod_sdm);
    if (c->module)  pfn_cuModuleUnload(c->module);
    if (c->context) pfn_cuCtxDestroy(c->context);
    free(c);
}

/* ─── FIR Setup ─── */

int gpu_cuda_fir_setup(cuda_context_t *c, const float *taps, int ntaps,
                        int num_stages, bool upsample) {
    c->ntaps      = ntaps;
    c->num_stages = num_stages;
    c->upsample   = upsample;

    /* Upload taps to constant memory */
    CUdeviceptr d_taps;
    size_t taps_size;
    if (pfn_cuModuleGetGlobal(&d_taps, &taps_size, c->module, "c_taps") != CUDA_SUCCESS)
        return -1;
    if (pfn_cuMemcpyHtoD(d_taps, taps, ntaps * sizeof(float)) != CUDA_SUCCESS)
        return -1;

    /* Upload ntaps */
    CUdeviceptr d_ntaps;
    size_t ntaps_size;
    if (pfn_cuModuleGetGlobal(&d_ntaps, &ntaps_size, c->module, "c_ntaps") != CUDA_SUCCESS)
        return -1;
    if (pfn_cuMemcpyHtoD(d_ntaps, &ntaps, sizeof(int)) != CUDA_SUCCESS)
        return -1;

    return 0;
}

/* ─── Single-stage FIR ─── */

static int cuda_fir_stage(cuda_context_t *c, bool up,
                           CUdeviceptr d_in, CUdeviceptr d_out,
                           int in_count, int out_count) {
    CUfunction fn = up ? c->fn_fir_up : c->fn_fir_down;
    void *args[] = { &d_in, &d_out, &in_count, &out_count };
    return launch_kernel(c, fn, args, out_count);
}

/* ─── FIR Chain Process ─── */

int gpu_cuda_fir_chain(cuda_context_t *c, const float *in, float *out,
                        size_t in_count, size_t *out_count) {
    if (!c || in_count < GPU_MIN_SAMPLES) return -1;

    pfn_cuCtxSetCurrent(c->context);

    int s_idx = c->active;
    int stages = c->num_stages;
    if (stages <= 0) return -1;

    /* Calculate output size */
    size_t cur = in_count;
    for (int s = 0; s < stages; s++)
        cur = c->upsample ? cur * 2 : cur / 2;
    size_t final_out = cur;

    /* Calculate max intermediate */
    size_t max_size = in_count;
    cur = in_count;
    for (int s = 0; s < stages; s++) {
        cur = c->upsample ? cur * 2 : cur / 2;
        if (cur > max_size) max_size = cur;
    }

    /* Ensure buffers */
    if (ensure_d_in(c, max_size) != 0) return -1;
    if (ensure_d_out(c, max_size) != 0) return -1;
    if (ensure_h_in(c, in_count) != 0) return -1;
    if (ensure_h_out(c, final_out) != 0) return -1;
    if (stages > 1 && ensure_d_inter(c, max_size) != 0) return -1;

    CUstream stream = c->streams[s_idx];

    /* Upload input via pinned memory (async) */
    memcpy(c->h_in[s_idx], in, in_count * sizeof(float));
    if (pfn_cuMemcpyHtoDAsync(c->d_in[s_idx], c->h_in[s_idx],
                               in_count * sizeof(float), stream) != CUDA_SUCCESS)
        return -1;

    /* Multi-stage FIR dispatch */
    cur = in_count;
    for (int s = 0; s < stages; s++) {
        size_t next = c->upsample ? cur * 2 : cur / 2;

        CUdeviceptr src, dst;
        if (s == 0) {
            src = c->d_in[s_idx];
        } else {
            /* Ping-pong: even stages from d_inter, odd from d_out */
            src = (s & 1) ? c->d_inter : c->d_out[s_idx];
        }
        if (s == stages - 1) {
            dst = c->d_out[s_idx];
        } else {
            dst = (s & 1) ? c->d_out[s_idx] : c->d_inter;
        }

        /* For stage > 0, source is previous output — use proper ping-pong */
        if (stages == 1) {
            src = c->d_in[s_idx];
            dst = c->d_out[s_idx];
        } else if (s == 0) {
            src = c->d_in[s_idx];
            dst = c->d_inter;
        } else if (s == stages - 1) {
            src = (stages % 2 == 0) ? c->d_inter : c->d_out[s_idx];
            dst = c->d_out[s_idx];
        } else {
            /* Middle stages: alternate */
            if (s & 1) { src = c->d_inter; dst = c->d_out[s_idx]; }
            else       { src = c->d_out[s_idx]; dst = c->d_inter; }
        }

        cuda_fir_stage(c, c->upsample, src, dst, (int)cur, (int)next);
        cur = next;
    }

    /* Download result (async) */
    if (pfn_cuMemcpyDtoHAsync(c->h_out[s_idx], c->d_out[s_idx],
                               final_out * sizeof(float), stream) != CUDA_SUCCESS)
        return -1;

    /* Synchronize this stream */
    pfn_cuStreamSynchronize(stream);

    /* Copy from pinned to caller's buffer */
    memcpy(out, c->h_out[s_idx], final_out * sizeof(float));

    /* Rotate stream for next call (triple-buffering) */
    c->active = (s_idx + 1) % NUM_STREAMS;

    *out_count = final_out;
    return 0;
}

/* ─── Gain ─── */

int gpu_cuda_gain(cuda_context_t *c, float *buf, size_t count, float gain) {
    if (!c || count < GPU_MIN_SAMPLES) return -1;

    pfn_cuCtxSetCurrent(c->context);
    int s_idx = c->active;

    if (ensure_d_out(c, count) != 0) return -1;
    if (ensure_h_in(c, count) != 0) return -1;
    if (ensure_h_out(c, count) != 0) return -1;

    CUstream stream = c->streams[s_idx];

    memcpy(c->h_in[s_idx], buf, count * sizeof(float));
    pfn_cuMemcpyHtoDAsync(c->d_out[s_idx], c->h_in[s_idx],
                           count * sizeof(float), stream);

    int cnt = (int)count;
    void *args[] = { &c->d_out[s_idx], &cnt, &gain };
    launch_kernel(c, c->fn_gain, args, (int)count);

    pfn_cuMemcpyDtoHAsync(c->h_out[s_idx], c->d_out[s_idx],
                           count * sizeof(float), stream);
    pfn_cuStreamSynchronize(stream);
    memcpy(buf, c->h_out[s_idx], count * sizeof(float));

    c->active = (s_idx + 1) % NUM_STREAMS;
    return 0;
}

/* ─── Boxcar ─── */

int gpu_cuda_boxcar(cuda_context_t *c, const float *in, float *out,
                     size_t count, int taps, float gain) {
    if (!c || count < GPU_MIN_SAMPLES) return -1;

    pfn_cuCtxSetCurrent(c->context);
    int s_idx = c->active;

    if (ensure_d_in(c, count) != 0) return -1;
    if (ensure_d_out(c, count) != 0) return -1;
    if (ensure_h_in(c, count) != 0) return -1;
    if (ensure_h_out(c, count) != 0) return -1;

    CUstream stream = c->streams[s_idx];

    memcpy(c->h_in[s_idx], in, count * sizeof(float));
    pfn_cuMemcpyHtoDAsync(c->d_in[s_idx], c->h_in[s_idx],
                           count * sizeof(float), stream);

    int cnt = (int)count;
    void *args[] = { &c->d_in[s_idx], &c->d_out[s_idx], &cnt, &taps, &gain };
    launch_kernel(c, c->fn_boxcar, args, (int)count);

    pfn_cuMemcpyDtoHAsync(c->h_out[s_idx], c->d_out[s_idx],
                           count * sizeof(float), stream);
    pfn_cuStreamSynchronize(stream);
    memcpy(out, c->h_out[s_idx], count * sizeof(float));

    c->active = (s_idx + 1) % NUM_STREAMS;
    return 0;
}

/* ─── Batched multi-channel FIR ─── */

int gpu_cuda_fir_batch(cuda_context_t *c, const float *in_batch,
                        float *out_batch, size_t samples_per_ch,
                        int num_channels, size_t *out_count_per_ch) {
    if (!c || samples_per_ch < GPU_MIN_SAMPLES || num_channels <= 0)
        return -1;

    pfn_cuCtxSetCurrent(c->context);

    int stages = c->num_stages;
    if (stages <= 0) return -1;

    /* Calculate per-channel output size */
    size_t cur = samples_per_ch;
    for (int s = 0; s < stages; s++)
        cur = c->upsample ? cur * 2 : cur / 2;
    size_t out_per_ch = cur;

    /* Total sizes */
    size_t total_in  = samples_per_ch * (size_t)num_channels;
    size_t total_out = out_per_ch * (size_t)num_channels;

    /* Ensure device buffers (reuse d_in[0] and d_out[0] for batch) */
    size_t max_total = total_in > total_out ? total_in : total_out;
    if (ensure_d_in(c, max_total) != 0) return -1;
    if (ensure_d_out(c, max_total) != 0) return -1;
    if (ensure_h_in(c, total_in) != 0) return -1;
    if (ensure_h_out(c, total_out) != 0) return -1;
    if (stages > 1 && ensure_d_inter(c, max_total) != 0) return -1;

    int s_idx = c->active;
    CUstream stream = c->streams[s_idx];

    /* Upload all channels in one transfer */
    memcpy(c->h_in[s_idx], in_batch, total_in * sizeof(float));
    pfn_cuMemcpyHtoDAsync(c->d_in[s_idx], c->h_in[s_idx],
                           total_in * sizeof(float), stream);

    /* Multi-stage FIR: each stage uses 2D grid (samples, channels) */
    size_t in_per_ch = samples_per_ch;
    for (int s = 0; s < stages; s++) {
        size_t next_per_ch = c->upsample ? in_per_ch * 2 : in_per_ch / 2;

        CUdeviceptr src, dst;
        if (stages == 1) {
            src = c->d_in[s_idx];
            dst = c->d_out[s_idx];
        } else if (s == 0) {
            src = c->d_in[s_idx];
            dst = c->d_inter;
        } else if (s == stages - 1) {
            src = (stages % 2 == 0) ? c->d_inter : c->d_out[s_idx];
            dst = c->d_out[s_idx];
        } else {
            if (s & 1) { src = c->d_inter; dst = c->d_out[s_idx]; }
            else       { src = c->d_out[s_idx]; dst = c->d_inter; }
        }

        CUfunction fn = c->upsample ? c->fn_fir_up_batch : c->fn_fir_down_batch;
        unsigned grid_x = (unsigned)((next_per_ch + 255) / 256);
        unsigned grid_y = (unsigned)num_channels;
        int ipc = (int)in_per_ch;
        int opc = (int)next_per_ch;
        void *args[] = { &src, &dst, &ipc, &opc, &num_channels };

        pfn_cuLaunchKernel(fn, grid_x, grid_y, 1, 256, 1, 1,
                            0, stream, args, NULL);

        in_per_ch = next_per_ch;
    }

    /* Download all channels in one transfer */
    pfn_cuMemcpyDtoHAsync(c->h_out[s_idx], c->d_out[s_idx],
                           total_out * sizeof(float), stream);
    pfn_cuStreamSynchronize(stream);
    memcpy(out_batch, c->h_out[s_idx], total_out * sizeof(float));

    c->active = (s_idx + 1) % NUM_STREAMS;
    *out_count_per_ch = out_per_ch;
    return 0;
}

/* ─── Trellis SDM (cands >= 16) ─── */

int gpu_cuda_trellis(cuda_context_t *c, const float *in, float *out,
                      size_t count, const void *sdm_state_in,
                      void *sdm_state_out, int num_cands, int order,
                      const double *ntf_a, const double *ntf_g,
                      double state_limit, int trellis_lat) {
    if (!c || !c->fn_trellis_chunk || num_cands < 16)
        return -1;

    pfn_cuCtxSetCurrent(c->context);

    /* Upload NTF constants to SDM module */
    CUdeviceptr d_a, d_g, d_order, d_limit;
    size_t sz;
    pfn_cuModuleGetGlobal(&d_a, &sz, c->mod_sdm, "c_ntf_a");
    pfn_cuMemcpyHtoD(d_a, ntf_a, (size_t)order * sizeof(double));
    pfn_cuModuleGetGlobal(&d_g, &sz, c->mod_sdm, "c_ntf_g");
    pfn_cuMemcpyHtoD(d_g, ntf_g, (size_t)order * sizeof(double));
    pfn_cuModuleGetGlobal(&d_order, &sz, c->mod_sdm, "c_ntf_order");
    pfn_cuMemcpyHtoD(d_order, &order, sizeof(int));
    pfn_cuModuleGetGlobal(&d_limit, &sz, c->mod_sdm, "c_state_limit");
    pfn_cuMemcpyHtoD(d_limit, &state_limit, sizeof(double));

    /* Allocate device memory */
    CUdeviceptr d_in_buf, d_out_buf, d_init_states, d_init_costs;
    CUdeviceptr d_final_states, d_final_costs;
    size_t out_count = count > (size_t)trellis_lat ? count - (size_t)trellis_lat : 0;

    pfn_cuMemAlloc(&d_in_buf, count * sizeof(float));
    pfn_cuMemAlloc(&d_out_buf, (out_count > 0 ? out_count : 1) * sizeof(float));
    pfn_cuMemAlloc(&d_init_states, (size_t)num_cands * 8 * sizeof(double));
    pfn_cuMemAlloc(&d_init_costs, (size_t)num_cands * sizeof(double));
    pfn_cuMemAlloc(&d_final_states, (size_t)num_cands * 8 * sizeof(double));
    pfn_cuMemAlloc(&d_final_costs, (size_t)num_cands * sizeof(double));

    /* Upload input + initial state */
    pfn_cuMemcpyHtoD(d_in_buf, in, count * sizeof(float));
    pfn_cuMemcpyHtoD(d_init_states, sdm_state_in,
                      (size_t)num_cands * 8 * sizeof(double));
    const double *costs_in = (const double *)sdm_state_in + num_cands * 8;
    pfn_cuMemcpyHtoD(d_init_costs, costs_in, (size_t)num_cands * sizeof(double));

    /* Launch: block_size = max(64, 2*num_cands), 1 block */
    int block_size = 2 * num_cands;
    if (block_size < 64) block_size = 64;
    int cnt = (int)count;
    void *args[] = {
        &d_in_buf, &d_out_buf, &cnt,
        &num_cands, &trellis_lat,
        &d_init_states, &d_init_costs,
        &d_final_states, &d_final_costs
    };
    pfn_cuLaunchKernel(c->fn_trellis_chunk, 1, 1, 1,
                        (unsigned)block_size, 1, 1,
                        0, c->streams[0], args, NULL);
    pfn_cuStreamSynchronize(c->streams[0]);

    /* Download output + final state */
    if (out_count > 0)
        pfn_cuMemcpyDtoH(out, d_out_buf, out_count * sizeof(float));
    pfn_cuMemcpyDtoH(sdm_state_out, d_final_states,
                      (size_t)num_cands * 8 * sizeof(double));
    double *costs_out = (double *)sdm_state_out + num_cands * 8;
    pfn_cuMemcpyDtoH(costs_out, d_final_costs, (size_t)num_cands * sizeof(double));

    /* Cleanup */
    pfn_cuMemFree(d_in_buf);
    pfn_cuMemFree(d_out_buf);
    pfn_cuMemFree(d_init_states);
    pfn_cuMemFree(d_init_costs);
    pfn_cuMemFree(d_final_states);
    pfn_cuMemFree(d_final_costs);

    return 0;
}

/* ─── PreCorr batch ─── */

int gpu_cuda_precorr(cuda_context_t *c, const float *in, float *out,
                      size_t count, const float *ntf_a, const float *ntf_g,
                      int order, const float *pred_table,
                      const float *state_in, float *state_out,
                      int num_channels) {
    if (!c || !c->fn_precorr_chunk)
        return -1;

    pfn_cuCtxSetCurrent(c->context);

    /* Upload PreCorr NTF constants */
    CUdeviceptr d_a, d_g, d_order, d_limit;
    size_t sz;
    pfn_cuModuleGetGlobal(&d_a, &sz, c->mod_sdm, "c_precorr_a");
    pfn_cuMemcpyHtoD(d_a, ntf_a, (size_t)order * sizeof(float));
    pfn_cuModuleGetGlobal(&d_g, &sz, c->mod_sdm, "c_precorr_g");
    pfn_cuMemcpyHtoD(d_g, ntf_g, (size_t)order * sizeof(float));
    pfn_cuModuleGetGlobal(&d_order, &sz, c->mod_sdm, "c_precorr_order");
    pfn_cuMemcpyHtoD(d_order, &order, sizeof(int));
    float zero_limit = 0.0f;
    pfn_cuModuleGetGlobal(&d_limit, &sz, c->mod_sdm, "c_precorr_limit");
    pfn_cuMemcpyHtoD(d_limit, &zero_limit, sizeof(float));

    size_t total = count * (size_t)num_channels;

    /* Allocate device memory */
    CUdeviceptr d_in_buf, d_out_buf, d_pred, d_state_in, d_state_out;
    pfn_cuMemAlloc(&d_in_buf, total * sizeof(float));
    pfn_cuMemAlloc(&d_out_buf, total * sizeof(float));
    pfn_cuMemAlloc(&d_pred, 256 * 8 * sizeof(float));
    pfn_cuMemAlloc(&d_state_in, (size_t)num_channels * 8 * sizeof(float));
    pfn_cuMemAlloc(&d_state_out, (size_t)num_channels * 8 * sizeof(float));

    /* Upload */
    pfn_cuMemcpyHtoD(d_in_buf, in, total * sizeof(float));
    pfn_cuMemcpyHtoD(d_pred, pred_table, 256 * 8 * sizeof(float));
    pfn_cuMemcpyHtoD(d_state_in, state_in, (size_t)num_channels * 8 * sizeof(float));

    /* Launch: one thread per channel */
    int cnt = (int)count;
    void *args[] = {
        &d_in_buf, &d_out_buf, &cnt,
        &d_pred, &d_state_in, &d_state_out, &num_channels
    };
    pfn_cuLaunchKernel(c->fn_precorr_chunk, 1, 1, 1,
                        (unsigned)num_channels, 1, 1,
                        0, c->streams[0], args, NULL);
    pfn_cuStreamSynchronize(c->streams[0]);

    /* Download */
    pfn_cuMemcpyDtoH(out, d_out_buf, total * sizeof(float));
    pfn_cuMemcpyDtoH(state_out, d_state_out,
                      (size_t)num_channels * 8 * sizeof(float));

    /* Cleanup */
    pfn_cuMemFree(d_in_buf);
    pfn_cuMemFree(d_out_buf);
    pfn_cuMemFree(d_pred);
    pfn_cuMemFree(d_state_in);
    pfn_cuMemFree(d_state_out);

    return 0;
}
