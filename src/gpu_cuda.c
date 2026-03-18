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
#include "../include/trellis.h"
#include "../include/ntf.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ─── Embedded PTX ─── */
#include "kernels/fir_kernels_ptx.h"
#include "kernels/sdm_kernels_ptx.h"
#include "kernels/sdm_parallel_ptx.h"

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
DECL_PFN(CUresult, cuDeviceGetAttribute, int *, int, CUdevice);
DECL_PFN(CUresult, cuStreamCreateWithPriority, CUstream *, unsigned int, int);
DECL_PFN(CUresult, cuCtxGetStreamPriorityRange, int *, int *);
DECL_PFN(CUresult, cuStreamDestroy, CUstream);
DECL_PFN(CUresult, cuStreamSynchronize, CUstream);
DECL_PFN(CUresult, cuMemcpyDtoD, CUdeviceptr, CUdeviceptr, size_t);

static HMODULE g_nvcuda = NULL;
static bool    g_probed = false;
static bool    g_available = false;
static char    g_device_name[128] = "";
static size_t  g_vram_bytes = 0;
static int     g_sm_count = 0;  /* Streaming Multiprocessors */

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
static PFN_cuDeviceGetAttribute pfn_cuDeviceGetAttribute;
static PFN_cuStreamCreateWithPriority pfn_cuStreamCreateWithPriority;
static PFN_cuCtxGetStreamPriorityRange pfn_cuCtxGetStreamPriorityRange;
static PFN_cuStreamDestroy     pfn_cuStreamDestroy;
static PFN_cuStreamSynchronize pfn_cuStreamSynchronize;
static PFN_cuMemcpyDtoD        pfn_cuMemcpyDtoD;

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
    /* Parallel-segment SDM (from sdm_parallel.ptx) */
    CUmodule       mod_sdm_parallel;
    CUfunction     fn_trellis_parallel;
    int            num_sms;  /* GPU SM count for optimal parallelism */
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
    /* Low-priority stream for SDM (won't block display) */
    CUstream       sdm_stream;
    /* FIR delay line: last (ntaps-1) input samples from previous chunk.
     * Prepended to input for continuity across chunk boundaries. */
    float         *delay_buf;       /* [ntaps-1] floats, host memory */
    bool           delay_valid;     /* true after first chunk */
    /* ─── Persistent SDM state (pre-allocated, no per-chunk alloc) ─── */
    CUdeviceptr    d_sdm_in;        /* SDM input buffer (device) */
    CUdeviceptr    d_sdm_out;       /* SDM output buffer (device) */
    size_t         sdm_buf_cap;     /* current capacity in floats */
    /* Trellis persistent state on device */
    CUdeviceptr    d_trellis_states; /* [num_cands * 8] doubles */
    CUdeviceptr    d_trellis_costs;  /* [num_cands] doubles */
    int            trellis_cands;    /* cached num_cands */
    int            trellis_order;    /* cached NTF order */
    int            trellis_lat;      /* cached latency */
    double         trellis_ntf_a[8]; /* cached NTF coefficients */
    double         trellis_ntf_g[8];
    bool           trellis_state_valid; /* false until first chunk or after reset */
    /* PreCorr persistent state on device */
    CUdeviceptr    d_precorr_init;   /* precorr_init_t on device */
    CUdeviceptr    d_precorr_pred;   /* prediction table [256][8] on device */
    bool           precorr_pred_uploaded; /* true after first upload */
    bool           precorr_state_valid;
    /* Boundary re-encoding: all segments' final NTF states */
    CUdeviceptr    d_all_final_states;
    CUdeviceptr    d_all_final_costs;
    double        *h_all_final_states;
    double        *h_all_final_costs;
    int            boundary_alloc;  /* allocated num_segs * nc */
    /* Boxcar history for chunk continuity — per channel */
#define BOXCAR_MAX_CH 8
    CUdeviceptr    d_boxcar_hist;    /* last taps samples (shared device buf) */
    float         *h_boxcar_hist[BOXCAR_MAX_CH]; /* host-side per-channel */
    int            boxcar_hist_taps; /* allocated size (0 = none) */
    bool           boxcar_hist_valid[BOXCAR_MAX_CH];
    int            boxcar_ch;       /* current channel index for calls */
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
    RESOLVE_V2(cuMemAllocHost);
    RESOLVE_V2(cuMemFreeHost);
    RESOLVE(cuStreamCreate);
    RESOLVE(cuDeviceGetAttribute);
    /* Optional — may not exist on older drivers */
    pfn_cuStreamCreateWithPriority = (PFN_cuStreamCreateWithPriority)
        GetProcAddress(g_nvcuda, "cuStreamCreateWithPriority");
    pfn_cuCtxGetStreamPriorityRange = (PFN_cuCtxGetStreamPriorityRange)
        GetProcAddress(g_nvcuda, "cuCtxGetStreamPriorityRange");
    RESOLVE_V2(cuStreamDestroy);
    RESOLVE(cuStreamSynchronize);
    RESOLVE_V2(cuMemcpyDtoD);

    if (!ok) { g_available = false; return false; }

    if (pfn_cuInit(0) != CUDA_SUCCESS) { g_available = false; return false; }

    CUdevice dev = 0;
    if (pfn_cuDeviceGet(&dev, 0) != CUDA_SUCCESS) { g_available = false; return false; }

    pfn_cuDeviceGetName(g_device_name, sizeof(g_device_name), dev);
    pfn_cuDeviceTotalMem(&g_vram_bytes, dev);

    /* Query SM count for optimal segment parallelism */
    if (pfn_cuDeviceGetAttribute)
        pfn_cuDeviceGetAttribute(&g_sm_count, 16 /* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */, dev);

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

    /* CU_CTX_MAP_HOST enables mapped pinned memory (ReBAR/SAM compatible).
     * Allows cuMemAllocHost buffers to be directly accessible from GPU
     * without explicit HtoD copies on ReBAR-enabled systems. */
    if (pfn_cuCtxCreate(&c->context, CU_CTX_SCHED_AUTO | 0x08 /* CU_CTX_MAP_HOST */,
                          dev) != CUDA_SUCCESS)
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

    /* Load parallel-segment SDM module */
    if (pfn_cuModuleLoadData(&c->mod_sdm_parallel, g_ptx_sdm_parallel) != CUDA_SUCCESS)
        goto fail;
    if (pfn_cuModuleGetFunction(&c->fn_trellis_parallel, c->mod_sdm_parallel,
                                  "trellis_parallel_segments") != CUDA_SUCCESS)
        goto fail;
    c->num_sms = g_sm_count > 0 ? g_sm_count : 64;

    /* Create streams */
    for (int i = 0; i < NUM_STREAMS; i++) {
        if (pfn_cuStreamCreate(&c->streams[i], 0) != CUDA_SUCCESS)
            goto fail;
    }
    /* Create high-priority stream for SDM (CUDA async compute doesn't
     * block display — unlike DX11 which needed low priority).
     * High priority ensures SDM gets maximum GPU scheduling. */
    if (pfn_cuStreamCreateWithPriority && pfn_cuCtxGetStreamPriorityRange) {
        int lo_pri = 0, hi_pri = 0;
        pfn_cuCtxGetStreamPriorityRange(&lo_pri, &hi_pri);
        /* hi_pri = highest priority (lowest number) */
        if (pfn_cuStreamCreateWithPriority(&c->sdm_stream, 0, hi_pri) != CUDA_SUCCESS)
            c->sdm_stream = c->streams[0];
    } else {
        c->sdm_stream = c->streams[0];
    }

    return (gpu_context_t *)c;

fail:
    gpu_cuda_destroy(c);
    return NULL;
}

void gpu_cuda_reset_chunk(void *ptr) {
    cuda_context_t *c = (cuda_context_t *)ptr;
    if (c) c->boxcar_ch = 0;
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
    free(c->delay_buf);
    /* Persistent SDM buffers */
    if (c->d_sdm_in)  pfn_cuMemFree(c->d_sdm_in);
    if (c->d_sdm_out) pfn_cuMemFree(c->d_sdm_out);
    if (c->d_trellis_states) pfn_cuMemFree(c->d_trellis_states);
    if (c->d_trellis_costs)  pfn_cuMemFree(c->d_trellis_costs);
    if (c->d_precorr_init)   pfn_cuMemFree(c->d_precorr_init);
    if (c->d_precorr_pred)   pfn_cuMemFree(c->d_precorr_pred);
    if (c->d_all_final_states) pfn_cuMemFree(c->d_all_final_states);
    if (c->d_all_final_costs)  pfn_cuMemFree(c->d_all_final_costs);
    free(c->h_all_final_states);
    free(c->h_all_final_costs);
    if (c->d_boxcar_hist) pfn_cuMemFree(c->d_boxcar_hist);
    for (int i = 0; i < BOXCAR_MAX_CH; i++) free(c->h_boxcar_hist[i]);
    if (c->mod_sdm_parallel) pfn_cuModuleUnload(c->mod_sdm_parallel);
    if (c->sdm_stream && c->sdm_stream != c->streams[0])
        pfn_cuStreamDestroy(c->sdm_stream);
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

    /* Allocate delay buffer for continuity across chunks */
    free(c->delay_buf);
    c->delay_buf = (float *)calloc((size_t)(ntaps - 1), sizeof(float));
    c->delay_valid = false;

    return 0;
}

/* ─── Persistent SDM setup ─── */

static int ensure_sdm_bufs(cuda_context_t *c, size_t floats) {
    if (c->sdm_buf_cap >= floats) return 0;
    if (c->d_sdm_in)  pfn_cuMemFree(c->d_sdm_in);
    if (c->d_sdm_out) pfn_cuMemFree(c->d_sdm_out);
    if (pfn_cuMemAlloc(&c->d_sdm_in, floats * sizeof(float)) != CUDA_SUCCESS)
        return -1;
    if (pfn_cuMemAlloc(&c->d_sdm_out, floats * sizeof(float)) != CUDA_SUCCESS)
        return -1;
    c->sdm_buf_cap = floats;
    return 0;
}

int gpu_cuda_trellis_setup(cuda_context_t *c, int num_cands, int order,
                            int trellis_lat, const double *ntf_a,
                            const double *ntf_g, double state_limit) {
    pfn_cuCtxSetCurrent(c->context);

    /* Upload NTF constants */
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

    /* Allocate persistent state buffers */
    if (c->d_trellis_states) pfn_cuMemFree(c->d_trellis_states);
    if (c->d_trellis_costs)  pfn_cuMemFree(c->d_trellis_costs);
    pfn_cuMemAlloc(&c->d_trellis_states, (size_t)num_cands * 8 * sizeof(double));
    pfn_cuMemAlloc(&c->d_trellis_costs, (size_t)num_cands * sizeof(double));

    /* Zero-init state on device */
    double *zeros = (double *)calloc((size_t)num_cands * 8, sizeof(double));
    double *czeros = (double *)calloc((size_t)num_cands, sizeof(double));
    if (zeros) pfn_cuMemcpyHtoD(c->d_trellis_states, zeros, (size_t)num_cands * 8 * sizeof(double));
    if (czeros) pfn_cuMemcpyHtoD(c->d_trellis_costs, czeros, (size_t)num_cands * sizeof(double));
    free(zeros); free(czeros);

    c->trellis_cands = num_cands;
    c->trellis_order = order;
    c->trellis_lat = trellis_lat;
    memcpy(c->trellis_ntf_a, ntf_a, (size_t)order * sizeof(double));
    memcpy(c->trellis_ntf_g, ntf_g, (size_t)order * sizeof(double));
    c->trellis_state_valid = false;
    return 0;
}

int gpu_cuda_precorr_setup(cuda_context_t *c, int order,
                            const float *ntf_a, const float *ntf_g,
                            const float *pred_table, float state_limit) {
    pfn_cuCtxSetCurrent(c->context);

    /* Upload NTF constants */
    CUdeviceptr d_a, d_g, d_order, d_limit;
    size_t sz;
    pfn_cuModuleGetGlobal(&d_a, &sz, c->mod_sdm, "c_precorr_a");
    pfn_cuMemcpyHtoD(d_a, ntf_a, (size_t)order * sizeof(float));
    pfn_cuModuleGetGlobal(&d_g, &sz, c->mod_sdm, "c_precorr_g");
    pfn_cuMemcpyHtoD(d_g, ntf_g, (size_t)order * sizeof(float));
    pfn_cuModuleGetGlobal(&d_order, &sz, c->mod_sdm, "c_precorr_order");
    pfn_cuMemcpyHtoD(d_order, &order, sizeof(int));
    pfn_cuModuleGetGlobal(&d_limit, &sz, c->mod_sdm, "c_precorr_limit");
    pfn_cuMemcpyHtoD(d_limit, &state_limit, sizeof(float));

    /* Upload prediction table (persistent) */
    if (!c->d_precorr_pred)
        pfn_cuMemAlloc(&c->d_precorr_pred, 256 * 8 * sizeof(float));
    pfn_cuMemcpyHtoD(c->d_precorr_pred, pred_table, 256 * 8 * sizeof(float));

    /* Allocate persistent init/final state */
    if (!c->d_precorr_init)
        pfn_cuMemAlloc(&c->d_precorr_init, sizeof(gpu_precorr_state_t));

    c->precorr_pred_uploaded = true;
    c->precorr_state_valid = false;
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

    CUresult cr = pfn_cuCtxSetCurrent(c->context);
    if (cr != CUDA_SUCCESS) { OutputDebugStringA("CUDA: cuCtxSetCurrent failed\n"); return -1; }

    int s_idx = c->active;
    int stages = c->num_stages;
    if (stages <= 0) { OutputDebugStringA("CUDA: stages <= 0\n"); return -1; }

    /* FIR delay line: prepend previous chunk's tail for continuity.
     * Extended input = [delay_buf (ntaps-1)] [in (in_count)]
     * The FIR output for the extended input includes the transient,
     * but we only keep output starting from the original input position. */
    int dly_len = c->ntaps - 1;  /* 62 for 63-tap filter */
    size_t ext_in = in_count + (size_t)dly_len;

    /* Calculate output size (based on original in_count, not extended) */
    size_t cur = in_count;
    for (int s = 0; s < stages; s++)
        cur = c->upsample ? cur * 2 : cur / 2;
    size_t final_out = cur;

    /* Extended output includes the delay prefix */
    size_t ext_out = ext_in;
    for (int s = 0; s < stages; s++)
        ext_out = c->upsample ? ext_out * 2 : ext_out / 2;

    /* Calculate max intermediate */
    size_t max_size = ext_in;
    cur = ext_in;
    for (int s = 0; s < stages; s++) {
        cur = c->upsample ? cur * 2 : cur / 2;
        if (cur > max_size) max_size = cur;
    }

    /* Ensure buffers */
    if (ensure_d_in(c, max_size) != 0) return -1;
    if (ensure_d_out(c, max_size) != 0) return -1;
    if (ensure_h_in(c, ext_in) != 0) return -1;
    if (ensure_h_out(c, ext_out) != 0) return -1;
    if (stages > 1 && ensure_d_inter(c, max_size) != 0) return -1;

    CUstream stream = c->streams[s_idx];

    /* Build extended input: [delay | in] */
    if (c->delay_valid && c->delay_buf)
        memcpy(c->h_in[s_idx], c->delay_buf, (size_t)dly_len * sizeof(float));
    else
        memset(c->h_in[s_idx], 0, (size_t)dly_len * sizeof(float));
    memcpy(c->h_in[s_idx] + dly_len, in, in_count * sizeof(float));

    /* Save current input's tail as delay for next chunk */
    if (c->delay_buf) {
        if (in_count >= (size_t)dly_len)
            memcpy(c->delay_buf, in + in_count - dly_len,
                   (size_t)dly_len * sizeof(float));
        c->delay_valid = true;
    }

    if (pfn_cuMemcpyHtoDAsync(c->d_in[s_idx], c->h_in[s_idx],
                               ext_in * sizeof(float), stream) != CUDA_SUCCESS)
        return -1;

    /* Multi-stage FIR dispatch on extended input */
    cur = ext_in;
    for (int s = 0; s < stages; s++) {
        size_t next = c->upsample ? cur * 2 : cur / 2;

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

        cuda_fir_stage(c, c->upsample, src, dst, (int)cur, (int)next);
        cur = next;
    }
    /* cur now = ext_out (extended output including delay prefix) */

    /* Download full extended result */
    if (pfn_cuMemcpyDtoHAsync(c->h_out[s_idx], c->d_out[s_idx],
                               ext_out * sizeof(float), stream) != CUDA_SUCCESS)
        return -1;

    pfn_cuStreamSynchronize(stream);

    /* Strip delay prefix from output: skip first (ext_out - final_out) samples.
     * The delay prefix in the input gets scaled by the FIR chain ratio. */
    size_t out_skip = ext_out - final_out;
    memcpy(out, c->h_out[s_idx] + out_skip, final_out * sizeof(float));

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

    /* Per-channel boxcar history.
     * boxcar_ch increments per call, reset by gpu_cuda_reset_chunk(). */
    int ch = c->boxcar_ch % BOXCAR_MAX_CH;
    c->boxcar_ch++;

    /* Ensure boxcar history buffers exist.
     * Pre-fill with DSD silence (alternating ±1.0, sums to ~0)
     * so the very first chunk has valid history. */
    if (c->boxcar_hist_taps < taps) {
        if (c->d_boxcar_hist) pfn_cuMemFree(c->d_boxcar_hist);
        for (int i = 0; i < BOXCAR_MAX_CH; i++) {
            free(c->h_boxcar_hist[i]);
            c->h_boxcar_hist[i] = (float *)malloc((size_t)taps * sizeof(float));
            if (c->h_boxcar_hist[i]) {
                for (int j = 0; j < taps; j++)
                    c->h_boxcar_hist[i][j] = (j & 1) ? -1.0f : 1.0f;
                c->boxcar_hist_valid[i] = true;  /* valid from start */
            }
        }
        pfn_cuMemAlloc(&c->d_boxcar_hist, (size_t)taps * sizeof(float));
        c->boxcar_hist_taps = taps;
    }

    CUstream stream = c->streams[s_idx];

    memcpy(c->h_in[s_idx], in, count * sizeof(float));
    pfn_cuMemcpyHtoDAsync(c->d_in[s_idx], c->h_in[s_idx],
                           count * sizeof(float), stream);

    /* Upload THIS channel's history if available */
    CUdeviceptr hist_ptr = (CUdeviceptr)0;
    if (c->boxcar_hist_valid[ch] && c->h_boxcar_hist[ch]) {
        pfn_cuMemcpyHtoDAsync(c->d_boxcar_hist, c->h_boxcar_hist[ch],
                               (size_t)taps * sizeof(float), stream);
        hist_ptr = c->d_boxcar_hist;
    }

    int cnt = (int)count;
    void *args[] = { &c->d_in[s_idx], &c->d_out[s_idx], &cnt, &taps, &gain, &hist_ptr };
    launch_kernel(c, c->fn_boxcar, args, (int)count);

    pfn_cuMemcpyDtoHAsync(c->h_out[s_idx], c->d_out[s_idx],
                           count * sizeof(float), stream);
    pfn_cuStreamSynchronize(stream);
    memcpy(out, c->h_out[s_idx], count * sizeof(float));

    /* Save last `taps` samples as THIS channel's history */
    if (count >= (size_t)taps && c->h_boxcar_hist[ch]) {
        memcpy(c->h_boxcar_hist[ch], in + count - (size_t)taps,
               (size_t)taps * sizeof(float));
        c->boxcar_hist_valid[ch] = true;
    }

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

/* Persistent-buffer Trellis: no per-chunk alloc/free.
 * State stays on GPU between chunks. Only input/output transferred.
 * Call gpu_cuda_trellis_setup once at engine init. */
/* Parallel-segment GPU Trellis: launches num_sms independent segments,
 * each with overlap warmup. Massive parallelism across segments.
 * Same approach as CPU threadpool segmentation but with 84+ segments. */
int gpu_cuda_trellis(cuda_context_t *c, const float *in, float *out,
                      size_t count) {
    if (!c || !c->fn_trellis_parallel || c->trellis_cands < 2)
        return -1;

    pfn_cuCtxSetCurrent(c->context);

    int nc = c->trellis_cands;
    int lat = c->trellis_lat;

    /* SBVD parameters:
     * M = convergence region (trellis paths converge regardless of init state)
     * D = output region (valid samples kept)
     * L = traceback lookahead (ensures correct decisions at end of D)
     * Total processed per segment: M + D + L. Only D samples output. */
    /* Convergence depth scales with trellis latency but capped to keep
     * per-segment time reasonable. At DSD256 (lat=512), 32×lat=16384
     * makes segments too large for real-time. Cap M at 8192. */
    /* M=4096: convergence sufficient, allows more segments.
     * More segments = more independent trellis paths = noise averages
     * out better (31 dB vs 25 dB at 252 vs 84 segments). */
    int M = 16 * lat;
    if (M > 4096) M = 4096;
    int L = lat;

    /* Maximize segments: 3× SMs for better diversity averaging.
     * With MAX_CANDS=8 in kernel, ~10 blocks/SM fit in shared memory. */
    int num_segs = c->num_sms * 3;
    if (num_segs < 1) num_segs = 1;
    size_t min_D_sbvd = (size_t)(M + L + 1024);
    if (num_segs > (int)(count / min_D_sbvd)) num_segs = (int)(count / min_D_sbvd);
    if (num_segs < 1) num_segs = 1;

    int D = (int)(count / (size_t)num_segs);

    /* seg_total clamp: last segment must not read beyond input */
    int seg_total = M + D + L;
    int last_start = (num_segs - 1) * D - M;
    if (last_start < 0) last_start = 0;
    int max_total = (int)count - last_start;
    if (seg_total > max_total) seg_total = max_total;

    /* Upload NTF constants to parallel module */
    {
        CUdeviceptr d_a, d_g, d_order, d_limit;
        size_t sz;
        pfn_cuModuleGetGlobal(&d_a, &sz, c->mod_sdm_parallel, "c_ntf_a");
        pfn_cuMemcpyHtoD(d_a, c->trellis_ntf_a, (size_t)c->trellis_order * sizeof(double));
        pfn_cuModuleGetGlobal(&d_g, &sz, c->mod_sdm_parallel, "c_ntf_g");
        pfn_cuMemcpyHtoD(d_g, c->trellis_ntf_g, (size_t)c->trellis_order * sizeof(double));
        pfn_cuModuleGetGlobal(&d_order, &sz, c->mod_sdm_parallel, "c_ntf_order");
        pfn_cuMemcpyHtoD(d_order, &c->trellis_order, sizeof(int));
        pfn_cuModuleGetGlobal(&d_limit, &sz, c->mod_sdm_parallel, "c_state_limit");
        double sl = 0.0;
        pfn_cuMemcpyHtoD(d_limit, &sl, sizeof(double));
    }

    /* Build segment descriptors */
    int *h_seg_starts = (int *)malloc((size_t)num_segs * sizeof(int));
    int *h_seg_out_starts = (int *)malloc((size_t)num_segs * sizeof(int));
    if (!h_seg_starts || !h_seg_out_starts) {
        free(h_seg_starts); free(h_seg_out_starts);
        return -1;
    }

    size_t out_per_seg = (size_t)D;
    for (int i = 0; i < num_segs; i++) {
        int in_start;
        if (i == 0) {
            in_start = 0;
        } else {
            in_start = i * D - M;
            if (in_start < 0) in_start = 0;
        }
        if (in_start + seg_total > (int)count)
            in_start = (int)count - seg_total;
        if (in_start < 0) in_start = 0;
        h_seg_starts[i] = in_start;
        h_seg_out_starts[i] = (int)((size_t)i * out_per_seg);
    }

    CUstream stream = c->sdm_stream;
    size_t total_out = out_per_seg * (size_t)num_segs;

    /* Ensure device buffers */
    if (ensure_sdm_bufs(c, count > total_out ? count : total_out) != 0) {
        free(h_seg_starts); free(h_seg_out_starts);
        return -1;
    }

    /* Upload input + segment descriptors */
    CUdeviceptr d_seg_starts, d_seg_out_starts;
    pfn_cuMemAlloc(&d_seg_starts, (size_t)num_segs * sizeof(int));
    pfn_cuMemAlloc(&d_seg_out_starts, (size_t)num_segs * sizeof(int));
    pfn_cuMemcpyHtoDAsync(c->d_sdm_in, in, count * sizeof(float), stream);
    pfn_cuMemcpyHtoDAsync(d_seg_starts, h_seg_starts, (size_t)num_segs * sizeof(int), stream);
    pfn_cuMemcpyHtoDAsync(d_seg_out_starts, h_seg_out_starts, (size_t)num_segs * sizeof(int), stream);

    LARGE_INTEGER t_start_qpc, t_kernel, t_end, t_freq;
    QueryPerformanceFrequency(&t_freq);
    QueryPerformanceCounter(&t_start_qpc);

    /* Segment 0 persistent state */
    static CUdeviceptr d_seg0_final_s = 0, d_seg0_final_c = 0;
    static int seg0_alloc = 0;
    if (seg0_alloc < nc) {
        if (d_seg0_final_s) pfn_cuMemFree(d_seg0_final_s);
        if (d_seg0_final_c) pfn_cuMemFree(d_seg0_final_c);
        pfn_cuMemAlloc(&d_seg0_final_s, (size_t)nc * 8 * sizeof(double));
        pfn_cuMemAlloc(&d_seg0_final_c, (size_t)nc * sizeof(double));
        seg0_alloc = nc;
    }

    CUdeviceptr seg0_init_s = c->trellis_state_valid ? c->d_trellis_states : (CUdeviceptr)0;
    CUdeviceptr seg0_init_c = c->trellis_state_valid ? c->d_trellis_costs : (CUdeviceptr)0;

    /* Allocate all-segment final state buffers for boundary re-encoding */
    int needed = num_segs * nc;
    if (c->boundary_alloc < needed) {
        if (c->d_all_final_states) pfn_cuMemFree(c->d_all_final_states);
        if (c->d_all_final_costs)  pfn_cuMemFree(c->d_all_final_costs);
        free(c->h_all_final_states); free(c->h_all_final_costs);
        pfn_cuMemAlloc(&c->d_all_final_states, (size_t)needed * 8 * sizeof(double));
        pfn_cuMemAlloc(&c->d_all_final_costs, (size_t)needed * sizeof(double));
        c->h_all_final_states = (double *)malloc((size_t)needed * 8 * sizeof(double));
        c->h_all_final_costs = (double *)malloc((size_t)needed * sizeof(double));
        c->boundary_alloc = needed;
    }

    /* Launch kernel */
    int block_size = 2 * nc;
    if (block_size < 32) block_size = 32;
    int seg_total_i = seg_total;
    int M_param = M;
    int D_param = D;
    int overlap_param = 0;
    void *args[] = {
        &c->d_sdm_in, &c->d_sdm_out,
        &d_seg_starts, &d_seg_out_starts,
        &seg_total_i, &M_param, &D_param, &nc, &lat, &overlap_param,
        &seg0_init_s, &seg0_init_c,
        &d_seg0_final_s, &d_seg0_final_c,
        &c->d_all_final_states, &c->d_all_final_costs
    };
    pfn_cuLaunchKernel(c->fn_trellis_parallel,
                        (unsigned)num_segs, 1, 1,
                        (unsigned)block_size, 1, 1,
                        0, stream, args, NULL);
    pfn_cuStreamSynchronize(stream);

    /* Copy segment 0's final state to persistent buffer */
    pfn_cuMemcpyDtoD(c->d_trellis_states, d_seg0_final_s,
                      (size_t)nc * 8 * sizeof(double));
    pfn_cuMemcpyDtoD(c->d_trellis_costs, d_seg0_final_c,
                      (size_t)nc * sizeof(double));
    QueryPerformanceCounter(&t_kernel);

    /* Download output + all-segment final states */
    if (total_out > 0)
        pfn_cuMemcpyDtoHAsync(out, c->d_sdm_out, total_out * sizeof(float), stream);
    pfn_cuMemcpyDtoHAsync(c->h_all_final_states, c->d_all_final_states,
                           (size_t)needed * 8 * sizeof(double), stream);
    pfn_cuMemcpyDtoHAsync(c->h_all_final_costs, c->d_all_final_costs,
                           (size_t)needed * sizeof(double), stream);
    pfn_cuStreamSynchronize(stream);

    /* ─── Boundary re-encoding on CPU ───
     * For each boundary between segments i and i+1:
     * 1. Init a temp sdm_context from segment i's best-candidate final state
     * 2. Warmup: feed (half + lat) samples before boundary → discard
     * 3. Replace: feed half samples after boundary → overwrite GPU output */
    if (0 && num_segs > 1) {  /* TODO: boundary re-encoding disabled — needs D >> M to work */
        int half = 256;
        ntf_filter_t tmp_flt;
        memset(&tmp_flt, 0, sizeof(tmp_flt));
        tmp_flt.order = c->trellis_order;
        for (int k = 0; k < c->trellis_order; k++) {
            tmp_flt.a[k] = c->trellis_ntf_a[k];
            tmp_flt.g[k] = c->trellis_ntf_g[k];
        }
        /* Use the auto-selected filter name for init */
        const ntf_filter_t *real_flt = ntf_auto_select(0);
        if (real_flt) tmp_flt.name = real_flt->name;

        float *discard_buf = (float *)malloc(((size_t)half + (size_t)lat + (size_t)M) * sizeof(float));

        for (int i = 0; i < num_segs - 1 && discard_buf; i++) {
            int bnd_out = (i + 1) * D;  /* boundary in output */
            int bnd_in = (i + 1) * D;   /* boundary in input (same-rate) */

            /* Warmup: feed half+lat samples before boundary */
            int warmup = half + lat;
            int in_start = bnd_in - warmup;
            if (in_start < 0) in_start = 0;
            int actual_warmup = bnd_in - in_start;

            /* Replace: half samples after boundary */
            int replace_end = bnd_in + half;
            if (replace_end > (int)count) replace_end = (int)count;
            int replace_count = replace_end - bnd_in;
            if (replace_count <= 0 || bnd_out + replace_count > (int)total_out)
                continue;

            /* Init temp SDM from segment i's best candidate (idx 0) */
            sdm_context_t tmp;
            sdm_context_init(&tmp, &tmp_flt, 8, nc, lat);

            /* Inject segment i's best candidate (idx 0) final state
             * into the single initial candidate (SDM starts with num_cands=1) */
            int state_off = i * nc * 8;  /* best candidate = index 0 */
            {
                sdm_state_t *s = tmp.trellis[0].act[0];
                for (int k = 0; k < c->trellis_order; k++)
                    s->state[k] = c->h_all_final_states[state_off + k];
                s->cost = 0.0;
            }

            /* Feed warmup (output discarded) */
            size_t warmup_out = sdm_process_block(&tmp, in + in_start, discard_buf, (size_t)actual_warmup);

            /* Feed replacement region → overwrite GPU output at boundary */
            size_t replace_out = sdm_process_block(&tmp, in + bnd_in, out + bnd_out, (size_t)replace_count);

            if (i == 0) {
                extern void trellis_log_c(const char *);
                char dbg[256];
                sprintf_s(dbg, sizeof(dbg),
                    "boundary[0]: bnd_out=%d warmup_in=%d warmup_out=%zu replace_in=%d replace_out=%zu "
                    "state[0]=%.4f in[bnd]=%.4f gpu_out[bnd]=%.1f cpu_out[bnd]=%.1f",
                    bnd_out, actual_warmup, warmup_out, replace_count, replace_out,
                    c->h_all_final_states[state_off],
                    in[bnd_in], out[bnd_out], out[bnd_out]);
                trellis_log_c(dbg);
            }

            sdm_context_free(&tmp);
        }
        free(discard_buf);
    }

    QueryPerformanceCounter(&t_end);

    pfn_cuMemFree(d_seg_starts);
    pfn_cuMemFree(d_seg_out_starts);
    free(h_seg_starts);
    free(h_seg_out_starts);

    /* Log timing */
    {
        extern void trellis_log_c(const char *);
        double kernel_ms = (double)(t_kernel.QuadPart - t_start_qpc.QuadPart) * 1000.0 / (double)t_freq.QuadPart;
        double total_ms = (double)(t_end.QuadPart - t_start_qpc.QuadPart) * 1000.0 / (double)t_freq.QuadPart;
        double audio_ms = (double)count / 2822400.0 * 1000.0;
        char msg[256];
        sprintf_s(msg, sizeof(msg),
            "[GPU CUDA SBVD] %zu samples, %d/%d segs/SMs, %d cands, M=%d D=%d L=%d: kernel=%.1fms total=%.1fms (%.2fx RT)",
            count, num_segs, c->num_sms, nc, M, D, L, kernel_ms, total_ms, total_ms / audio_ms);
        trellis_log_c(msg);
    }

    c->trellis_state_valid = true;
    return 0;
}

/* ─── PreCorr batch ─── */

/* Persistent-buffer PreCorr: no per-chunk alloc/free.
 * Pred table and state stay on GPU. Only input/output transferred.
 * Call gpu_cuda_precorr_setup once at engine init. */
int gpu_cuda_precorr(cuda_context_t *c, const float *in, float *out,
                      size_t count,
                      const gpu_precorr_state_t *init,
                      gpu_precorr_state_t *final_state) {
    if (!c || !c->fn_precorr_chunk || !c->precorr_pred_uploaded)
        return -1;

    pfn_cuCtxSetCurrent(c->context);

    /* Ensure persistent I/O buffers */
    if (ensure_sdm_bufs(c, count) != 0)
        return -1;

    /* Upload input + initial state (state is small: 48 bytes) */
    pfn_cuMemcpyHtoD(c->d_sdm_in, in, count * sizeof(float));
    pfn_cuMemcpyHtoD(c->d_precorr_init, init, sizeof(gpu_precorr_state_t));

    /* Allocate a device-side final state buffer (reuse d_precorr_init area + offset) */
    /* For simplicity, allocate a second init struct if needed */
    static CUdeviceptr d_precorr_final = 0;
    if (!d_precorr_final)
        pfn_cuMemAlloc(&d_precorr_final, sizeof(gpu_precorr_state_t));

    int cnt = (int)count;
    int num_ch = 1;
    void *args[] = {
        &c->d_sdm_in, &c->d_sdm_out, &cnt,
        &c->d_precorr_pred, &c->d_precorr_init,
        &d_precorr_final, &num_ch
    };
    pfn_cuLaunchKernel(c->fn_precorr_chunk, 1, 1, 1,
                        1, 1, 1, /* 1 thread for 1 channel */
                        0, c->streams[0], args, NULL);
    pfn_cuStreamSynchronize(c->streams[0]);

    /* Download output + final state */
    pfn_cuMemcpyDtoH(out, c->d_sdm_out, count * sizeof(float));
    pfn_cuMemcpyDtoH(final_state, d_precorr_final, sizeof(gpu_precorr_state_t));

    c->precorr_state_valid = true;
    return 0;
}
