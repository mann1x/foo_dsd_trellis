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
#include "kernels/sdm_hawksford_ptx.h"
#include "kernels/das_stitch_ptx.h"
#include "kernels/sdm_multibit_ptx.h"
#include "kernels/conv_kernels_ptx.h"

/* ─── CUDA Driver API types (no CUDA headers needed) ─── */

typedef int            CUresult;
typedef int            CUdevice;
typedef void          *CUcontext;
typedef void          *CUmodule;
typedef void          *CUfunction;
typedef unsigned long long CUdeviceptr;
typedef void          *CUstream;
typedef void          *CUevent;

#define CUDA_SUCCESS   0
#define CU_CTX_SCHED_AUTO 0
#define CU_CTX_SCHED_BLOCKING_SYNC 4
#define EXT_SEED_SIZE (8 * 128 + 8 * 4 + 8 * 4 + 4 + 4 + 4)  /* 1100 bytes */

/* ─── cuFFT types (delay-loaded, no cufft.h needed) ─── */
typedef void *cufftHandle;
#define CUFFT_SUCCESS 0
#define CUFFT_Z2Z 0x69   /* complex double-to-complex double */
#define CUFFT_FORWARD -1
#define CUFFT_INVERSE  1

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
DECL_PFN(CUresult, cuMemcpyDtoDAsync, CUdeviceptr, CUdeviceptr, size_t, CUstream);
DECL_PFN(CUresult, cuMemsetD8Async, CUdeviceptr, unsigned char, size_t, CUstream);
DECL_PFN(CUresult, cuEventCreate, CUevent *, unsigned int);
DECL_PFN(CUresult, cuEventDestroy, CUevent);
DECL_PFN(CUresult, cuEventRecord, CUevent, CUstream);
DECL_PFN(CUresult, cuEventSynchronize, CUevent);

/* cuFFT function pointers (delay-loaded from cufft64_*.dll) */
typedef int (*PFN_cufftPlan1d)(cufftHandle *, int, int, int);
typedef int (*PFN_cufftExecZ2Z)(cufftHandle, void *, void *, int);
typedef void (*PFN_cufftDestroy)(cufftHandle);
typedef int (*PFN_cufftSetStream)(cufftHandle, CUstream);

static HMODULE g_cufft_dll = NULL;
static bool    g_cufft_available = false;
static PFN_cufftPlan1d     pfn_cufftPlan1d;
static PFN_cufftExecZ2Z    pfn_cufftExecZ2Z;
static PFN_cufftDestroy    pfn_cufftDestroy;
static PFN_cufftSetStream  pfn_cufftSetStream;

static HMODULE g_nvcuda = NULL;
static bool    g_probed = false;
static bool    g_available = false;
static char    g_device_name[128] = "";
static size_t  g_vram_bytes = 0;
static int     g_sm_count = 0;
static int     g_clock_khz = 0;  /* GPU clock rate in kHz */

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
static PFN_cuMemcpyDtoDAsync   pfn_cuMemcpyDtoDAsync;
static PFN_cuMemsetD8Async     pfn_cuMemsetD8Async;
static PFN_cuEventCreate       pfn_cuEventCreate;
static PFN_cuEventDestroy      pfn_cuEventDestroy;
static PFN_cuEventRecord       pfn_cuEventRecord;
static PFN_cuEventSynchronize  pfn_cuEventSynchronize;

/* Extended seed device buffer (used by DAS and hybrid paths) */
static CUdeviceptr s_d_ext_seed = 0;

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
    CUfunction     fn_fir_lowpass;
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
    /* Hawksford intra-step (from sdm_hawksford.ptx) */
    CUmodule       mod_hawksford;
    CUfunction     fn_hawksford;
    /* Multi-bit SDM experiment (from sdm_multibit.ptx) */
    CUmodule       mod_multibit;
    CUfunction     fn_mb_segments;
    CUfunction     fn_mb_to_1bit;
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
    /* FP64 FIR rate conversion */
    CUfunction     fn_fir_up_f64;
    CUfunction     fn_fir_down_f64;
    CUfunction     fn_gain_f64;
    CUdeviceptr    d_f64_a;          /* device buffer A (ping-pong) */
    CUdeviceptr    d_f64_b;          /* device buffer B */
    size_t         cap_f64;          /* capacity in doubles */
    double        *h_f64_in;         /* host staging input */
    double        *h_f64_out;        /* host staging output */
    size_t         cap_h_f64_in;
    size_t         cap_h_f64_out;
    double        *delay_buf_d;      /* [ntaps-1] doubles, host memory */
    bool           delay_valid_d;
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
    int            trellis_depth;    /* cached depth (for path mask) */
    double         trellis_ntf_a[8]; /* cached NTF coefficients */
    double         trellis_ntf_g[8];
    bool           trellis_state_valid; /* false until first chunk or after reset */
    /* Persistent trellis seed (history/path/next for chunk continuity) */
    CUdeviceptr    d_persistent_seed;  /* device buffer for final_seed output */
    unsigned char  h_persistent_seed[1100 * 8]; /* host copy (EXT_SEED_SIZE × max_channels) */
    bool           persistent_seed_valid;
    int            das_seed_nch;        /* num channels in h_persistent_seed */
    CUdeviceptr    d_das_final_seed;    /* DAS path final_seed device buffer */
    int            das_seed_alloc_nch;  /* allocated channels in d_das_final_seed */
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
    /* ─── DAS (Density-Aligned Stitching) buffers ─── */
    CUmodule       mod_das_stitch;
    CUfunction     fn_das_density_scan;
    CUfunction     fn_das_assemble;
    CUdeviceptr    d_das_seg_out;      /* segment outputs with overlap */
    CUdeviceptr    d_das_final;        /* final stitched output */
    CUdeviceptr    d_das_stitch_pos;   /* stitch positions [num_segs-1] */
    CUdeviceptr    d_das_seg_starts;   /* per-seg input offsets */
    CUdeviceptr    d_das_seg_out_starts; /* per-seg output offsets */
    CUdeviceptr    d_das_seg_total_sizes; /* per-seg input count */
    CUdeviceptr    d_das_seg_out_caps; /* per-seg output capacity */
    CUdeviceptr    d_das_seg_counts;   /* actual output counts [num_ch × num_segs] */
    CUdeviceptr    d_das_init_states;  /* replicated seed [num_segs × nc × 8] */
    CUdeviceptr    d_das_init_costs;   /* replicated seed [num_segs × nc] */
    CUdeviceptr    d_das_mid_states;   /* state snapshot at output[D] [num_segs × nc × 8] */
    CUdeviceptr    d_das_mid_costs;    /* cost at output[D] [num_segs × nc] */
    double        *h_das_mid_states;   /* host mirror for CPU re-encoding */
    size_t         das_alloc_segs;     /* allocated segment capacity */
    size_t         das_alloc_samples;  /* allocated per-ch sample capacity */
    /* ─── Hybrid CPU+GPU async state ─── */
    CUevent        hybrid_event;       /* signaled when GPU segments finish */
    bool           hybrid_event_valid; /* true if event was created */
    float         *hybrid_host_out;    /* pinned host buffer for GPU segment outputs */
    size_t         hybrid_host_cap;    /* capacity in floats */
    int            hybrid_num_gpu_segs;/* number of GPU segments in flight */
    int            hybrid_first_seg;   /* first GPU segment index */
    int            hybrid_num_ch;      /* channels in flight */
    int           *hybrid_out_caps;    /* per-segment output capacity [num_segs] */
    int           *hybrid_out_starts;  /* per-segment output offset [num_segs] */
    int            hybrid_block_size;
    int            hybrid_ch_stride_in;
    int            hybrid_ch_stride_out;
    int            hybrid_overlap;
    int            hybrid_M;
    /* Boxcar history for chunk continuity — per channel */
#define BOXCAR_MAX_CH 8
    CUdeviceptr    d_boxcar_hist;    /* last taps samples (shared device buf) */
    float         *h_boxcar_hist[BOXCAR_MAX_CH]; /* host-side per-channel */
    int            boxcar_hist_taps; /* allocated size (0 = none) */
    bool           boxcar_hist_valid[BOXCAR_MAX_CH];
    int            boxcar_ch;       /* current channel index for calls */
    /* FIR lowpass history for chunk continuity — per channel */
    CUdeviceptr    d_lp_hist;        /* last ntaps-1 samples (device) */
    float         *h_lp_hist[BOXCAR_MAX_CH]; /* host-side per-channel */
    int            lp_hist_len;      /* allocated history length (ntaps-1) */
    bool           lp_hist_valid[BOXCAR_MAX_CH];
    int            lp_ch;            /* current lowpass channel index */
    /* FIR lowpass fp64 device buffers (per-context, not static) */
    CUdeviceptr    d_lp_f64_in;
    CUdeviceptr    d_lp_f64_out;
    size_t         lp_f64_cap;
    double        *h_lp_f64_in;     /* host staging buffer */
    size_t         h_lp_f64_cap;
    /* ─── GPU Convolution (cuFFT + custom kernels) ─── */
    CUmodule       mod_conv;           /* conv_kernels PTX module */
    CUfunction     fn_conv_r2c;        /* real_to_complex kernel */
    CUfunction     fn_conv_cmul;       /* complex_mul kernel */
    CUfunction     fn_conv_cmulacc;    /* complex_mul_acc kernel */
    CUfunction     fn_conv_czero;      /* complex_zero kernel */
    CUfunction     fn_conv_extract;    /* extract_output kernel */
    CUfunction     fn_conv_fused;      /* fused_mul_acc kernel */
    int            gpu_clock_khz;      /* GPU clock rate */
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

/* Launch kernel on a specific stream (for functions that use sdm_stream
 * for transfers — must match stream to avoid reading uninitialized memory) */
static int launch_kernel_on(cuda_context_t *c, CUfunction fn,
                             void **args, int num_elements, CUstream stream) {
    unsigned grid = (unsigned)((num_elements + 255) / 256);
    return pfn_cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0,
                               stream, args, NULL)
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
    RESOLVE_V2(cuMemcpyDtoDAsync);
    RESOLVE_V2(cuMemsetD8Async);
    /* Event API (optional — hybrid mode only) */
    pfn_cuEventCreate = (PFN_cuEventCreate)GetProcAddress(g_nvcuda, "cuEventCreate");
    pfn_cuEventDestroy = (PFN_cuEventDestroy)GetProcAddress(g_nvcuda, "cuEventDestroy_v2");
    if (!pfn_cuEventDestroy)
        pfn_cuEventDestroy = (PFN_cuEventDestroy)GetProcAddress(g_nvcuda, "cuEventDestroy");
    pfn_cuEventRecord = (PFN_cuEventRecord)GetProcAddress(g_nvcuda, "cuEventRecord");
    pfn_cuEventSynchronize = (PFN_cuEventSynchronize)GetProcAddress(g_nvcuda, "cuEventSynchronize");

    if (!ok) { g_available = false; return false; }

    if (pfn_cuInit(0) != CUDA_SUCCESS) { g_available = false; return false; }

    CUdevice dev = 0;
    if (pfn_cuDeviceGet(&dev, 0) != CUDA_SUCCESS) { g_available = false; return false; }

    pfn_cuDeviceGetName(g_device_name, sizeof(g_device_name), dev);
    pfn_cuDeviceTotalMem(&g_vram_bytes, dev);

    /* Query SM count and clock for convolution budget */
    if (pfn_cuDeviceGetAttribute) {
        pfn_cuDeviceGetAttribute(&g_sm_count, 16 /* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */, dev);
        pfn_cuDeviceGetAttribute(&g_clock_khz, 13 /* CU_DEVICE_ATTRIBUTE_CLOCK_RATE */, dev);
    }

    /* Probe cuFFT availability (delay-load cufft64_*.dll) */
    if (!g_cufft_dll) {
        /* Try common cuFFT DLL names */
        const char *names[] = { "cufft64_11.dll", "cufft64_12.dll", "cufft64_10.dll", NULL };
        for (int i = 0; names[i] && !g_cufft_dll; i++)
            g_cufft_dll = LoadLibraryA(names[i]);
        if (g_cufft_dll) {
            pfn_cufftPlan1d    = (PFN_cufftPlan1d)GetProcAddress(g_cufft_dll, "cufftPlan1d");
            pfn_cufftExecZ2Z   = (PFN_cufftExecZ2Z)GetProcAddress(g_cufft_dll, "cufftExecZ2Z");
            pfn_cufftDestroy   = (PFN_cufftDestroy)GetProcAddress(g_cufft_dll, "cufftDestroy");
            pfn_cufftSetStream = (PFN_cufftSetStream)GetProcAddress(g_cufft_dll, "cufftSetStream");
            g_cufft_available = (pfn_cufftPlan1d && pfn_cufftExecZ2Z && pfn_cufftDestroy);
        }
    }

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
    if (pfn_cuCtxCreate(&c->context, CU_CTX_SCHED_BLOCKING_SYNC | 0x08 /* CU_CTX_MAP_HOST */,
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
    if (pfn_cuModuleGetFunction(&c->fn_fir_lowpass, c->module, "fir_lowpass") != CUDA_SUCCESS)
        goto fail;
    /* Batched multi-channel kernels */
    if (pfn_cuModuleGetFunction(&c->fn_fir_up_batch, c->module, "fir_upsample_2x_batch") != CUDA_SUCCESS)
        goto fail;
    if (pfn_cuModuleGetFunction(&c->fn_fir_down_batch, c->module, "fir_downsample_2x_batch") != CUDA_SUCCESS)
        goto fail;
    if (pfn_cuModuleGetFunction(&c->fn_gain_batch, c->module, "gain_apply_batch") != CUDA_SUCCESS)
        goto fail;
    /* FP64 FIR kernels */
    if (pfn_cuModuleGetFunction(&c->fn_fir_up_f64, c->module, "fir_upsample_2x_f64") != CUDA_SUCCESS)
        goto fail;
    if (pfn_cuModuleGetFunction(&c->fn_fir_down_f64, c->module, "fir_downsample_2x_f64") != CUDA_SUCCESS)
        goto fail;
    if (pfn_cuModuleGetFunction(&c->fn_gain_f64, c->module, "gain_apply_f64") != CUDA_SUCCESS)
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

    /* Load Hawksford kernel */
    if (pfn_cuModuleLoadData(&c->mod_hawksford, g_ptx_sdm_hawksford) == CUDA_SUCCESS)
        pfn_cuModuleGetFunction(&c->fn_hawksford, c->mod_hawksford, "trellis_hawksford");

    /* Load multi-bit SDM kernel (experimental) */
    if (pfn_cuModuleLoadData(&c->mod_multibit, g_ptx_sdm_multibit) == CUDA_SUCCESS) {
        pfn_cuModuleGetFunction(&c->fn_mb_segments, c->mod_multibit, "sdm_multibit_segments");
        pfn_cuModuleGetFunction(&c->fn_mb_to_1bit, c->mod_multibit, "sdm_multibit_to_1bit");
    }

    /* Load convolution kernels */
    if (pfn_cuModuleLoadData(&c->mod_conv, g_ptx_conv_kernels) == CUDA_SUCCESS) {
        pfn_cuModuleGetFunction(&c->fn_conv_r2c, c->mod_conv, "conv_real_to_complex");
        pfn_cuModuleGetFunction(&c->fn_conv_cmul, c->mod_conv, "conv_complex_mul");
        pfn_cuModuleGetFunction(&c->fn_conv_cmulacc, c->mod_conv, "conv_complex_mul_acc");
        pfn_cuModuleGetFunction(&c->fn_conv_czero, c->mod_conv, "conv_complex_zero");
        pfn_cuModuleGetFunction(&c->fn_conv_extract, c->mod_conv, "conv_extract_output");
        pfn_cuModuleGetFunction(&c->fn_conv_fused, c->mod_conv, "conv_fused_mul_acc");
    }
    c->gpu_clock_khz = g_clock_khz;

    /* Load DAS stitching kernels */
    if (pfn_cuModuleLoadData(&c->mod_das_stitch, g_ptx_das_stitch) == CUDA_SUCCESS) {
        pfn_cuModuleGetFunction(&c->fn_das_density_scan, c->mod_das_stitch, "das_density_scan");
        pfn_cuModuleGetFunction(&c->fn_das_assemble, c->mod_das_stitch, "das_assemble");
    }

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
    if (c) { c->boxcar_ch = 0; c->lp_ch = 0; }
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
    /* FP64 FIR buffers */
    if (c->d_f64_a) pfn_cuMemFree(c->d_f64_a);
    if (c->d_f64_b) pfn_cuMemFree(c->d_f64_b);
    free(c->h_f64_in);
    free(c->h_f64_out);
    free(c->delay_buf_d);
    /* Persistent SDM buffers */
    if (c->d_sdm_in)  pfn_cuMemFree(c->d_sdm_in);
    if (c->d_sdm_out) pfn_cuMemFree(c->d_sdm_out);
    if (c->d_trellis_states) pfn_cuMemFree(c->d_trellis_states);
    if (c->d_trellis_costs)  pfn_cuMemFree(c->d_trellis_costs);
    if (c->d_persistent_seed) pfn_cuMemFree(c->d_persistent_seed);
    if (c->d_das_final_seed)  pfn_cuMemFree(c->d_das_final_seed);
    if (c->d_precorr_init)   pfn_cuMemFree(c->d_precorr_init);
    if (c->d_precorr_pred)   pfn_cuMemFree(c->d_precorr_pred);
    if (c->d_all_final_states) pfn_cuMemFree(c->d_all_final_states);
    if (c->d_all_final_costs)  pfn_cuMemFree(c->d_all_final_costs);
    free(c->h_all_final_states);
    free(c->h_all_final_costs);
    if (c->d_boxcar_hist) pfn_cuMemFree(c->d_boxcar_hist);
    for (int i = 0; i < BOXCAR_MAX_CH; i++) free(c->h_boxcar_hist[i]);
    if (c->d_lp_hist) pfn_cuMemFree(c->d_lp_hist);
    for (int i = 0; i < BOXCAR_MAX_CH; i++) free(c->h_lp_hist[i]);
    if (c->d_lp_f64_in)  pfn_cuMemFree(c->d_lp_f64_in);
    if (c->d_lp_f64_out) pfn_cuMemFree(c->d_lp_f64_out);
    free(c->h_lp_f64_in);
    if (c->mod_sdm_parallel) pfn_cuModuleUnload(c->mod_sdm_parallel);
    if (c->mod_hawksford)    pfn_cuModuleUnload(c->mod_hawksford);
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

    /* Upload fp64 taps to constant memory (for fp64 FIR chain) */
    CUdeviceptr d_taps_d;
    size_t taps_d_size;
    if (pfn_cuModuleGetGlobal(&d_taps_d, &taps_d_size, c->module, "c_taps_d") == CUDA_SUCCESS) {
        double taps_d[64];
        for (int i = 0; i < ntaps; i++)
            taps_d[i] = (double)taps[i];
        pfn_cuMemcpyHtoD(d_taps_d, taps_d, (size_t)ntaps * sizeof(double));
    }

    /* Allocate delay buffers for continuity across chunks */
    free(c->delay_buf);
    c->delay_buf = (float *)calloc((size_t)(ntaps - 1), sizeof(float));
    c->delay_valid = false;

    free(c->delay_buf_d);
    c->delay_buf_d = (double *)calloc((size_t)(ntaps - 1), sizeof(double));
    c->delay_valid_d = false;

    return 0;
}

/* ─── Persistent SDM setup ─── */

static int ensure_sdm_bufs(cuda_context_t *c, size_t floats) {
    if (c->sdm_buf_cap >= floats) return 0;
    if (c->d_sdm_in)  pfn_cuMemFree(c->d_sdm_in);
    if (c->d_sdm_out) pfn_cuMemFree(c->d_sdm_out);
    /* Allocate as doubles (largest element) — covers both fp64 trellis and fp32 precorr */
    if (pfn_cuMemAlloc(&c->d_sdm_in, floats * sizeof(double)) != CUDA_SUCCESS)
        return -1;
    if (pfn_cuMemAlloc(&c->d_sdm_out, floats * sizeof(float)) != CUDA_SUCCESS)
        return -1;
    c->sdm_buf_cap = floats;
    return 0;
}

int gpu_cuda_trellis_setup(cuda_context_t *c, int num_cands, int order,
                            int trellis_lat, const double *ntf_a,
                            const double *ntf_g, double state_limit,
                            int trellis_depth) {
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

    c->trellis_depth = trellis_depth > 0 ? trellis_depth : 8;
    /* Path mask: both CPU and GPU use 8-bit (0xFF) for consistent dedup. */
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

/* ─── FP64 FIR Chain Process ───
 * Same algorithm as gpu_cuda_fir_chain but with double precision.
 * Uses separate fp64 device/host buffers. Simple synchronous path
 * since GPU FIR load is <5%. Input is float (DSD ±1.0), widened to
 * double on host before upload. Output is double. */

static int ensure_f64_bufs(cuda_context_t *c, size_t doubles) {
    if (c->cap_f64 >= doubles) return 0;
    if (c->d_f64_a) pfn_cuMemFree(c->d_f64_a);
    if (c->d_f64_b) pfn_cuMemFree(c->d_f64_b);
    c->d_f64_a = 0; c->d_f64_b = 0;
    if (pfn_cuMemAlloc(&c->d_f64_a, doubles * sizeof(double)) != CUDA_SUCCESS)
        return -1;
    if (pfn_cuMemAlloc(&c->d_f64_b, doubles * sizeof(double)) != CUDA_SUCCESS) {
        pfn_cuMemFree(c->d_f64_a); c->d_f64_a = 0;
        return -1;
    }
    c->cap_f64 = doubles;
    return 0;
}

static int ensure_h_f64_in(cuda_context_t *c, size_t doubles) {
    if (c->cap_h_f64_in >= doubles) return 0;
    free(c->h_f64_in);
    c->h_f64_in = (double *)malloc(doubles * sizeof(double));
    c->cap_h_f64_in = c->h_f64_in ? doubles : 0;
    return c->h_f64_in ? 0 : -1;
}

static int ensure_h_f64_out(cuda_context_t *c, size_t doubles) {
    if (c->cap_h_f64_out >= doubles) return 0;
    free(c->h_f64_out);
    c->h_f64_out = (double *)malloc(doubles * sizeof(double));
    c->cap_h_f64_out = c->h_f64_out ? doubles : 0;
    return c->h_f64_out ? 0 : -1;
}

int gpu_cuda_fir_chain_f64(cuda_context_t *c, const float *in, double *out,
                            size_t in_count, size_t *out_count) {
    if (!c || in_count < GPU_MIN_SAMPLES) return -1;

    CUresult cr = pfn_cuCtxSetCurrent(c->context);
    if (cr != CUDA_SUCCESS) return -1;

    int stages = c->num_stages;
    if (stages <= 0) return -1;

    int dly_len = c->ntaps - 1;
    size_t ext_in = in_count + (size_t)dly_len;

    /* Calculate final output size (from original in_count) */
    size_t cur = in_count;
    for (int s = 0; s < stages; s++)
        cur = c->upsample ? cur * 2 : cur / 2;
    size_t final_out = cur;

    /* Extended output (includes delay prefix) */
    size_t ext_out = ext_in;
    for (int s = 0; s < stages; s++)
        ext_out = c->upsample ? ext_out * 2 : ext_out / 2;

    /* Max intermediate size for buffer allocation */
    size_t max_size = ext_in;
    cur = ext_in;
    for (int s = 0; s < stages; s++) {
        cur = c->upsample ? cur * 2 : cur / 2;
        if (cur > max_size) max_size = cur;
    }

    if (ensure_f64_bufs(c, max_size) != 0) return -1;
    if (ensure_h_f64_in(c, ext_in) != 0) return -1;
    if (ensure_h_f64_out(c, ext_out) != 0) return -1;

    /* Build extended input: [delay | in], widened float→double */
    if (c->delay_valid_d && c->delay_buf_d)
        memcpy(c->h_f64_in, c->delay_buf_d, (size_t)dly_len * sizeof(double));
    else
        memset(c->h_f64_in, 0, (size_t)dly_len * sizeof(double));
    for (size_t i = 0; i < in_count; i++)
        c->h_f64_in[dly_len + i] = (double)in[i];

    /* Save delay for next chunk */
    if (c->delay_buf_d) {
        if (in_count >= (size_t)dly_len) {
            for (size_t i = 0; i < (size_t)dly_len; i++)
                c->delay_buf_d[i] = (double)in[in_count - dly_len + i];
        }
        c->delay_valid_d = true;
    }

    CUstream stream = c->streams[0]; /* Use stream 0 (synchronous) */

    pfn_cuMemcpyHtoDAsync(c->d_f64_a, c->h_f64_in,
                           ext_in * sizeof(double), stream);

    /* Multi-stage FIR dispatch */
    cur = ext_in;
    for (int s = 0; s < stages; s++) {
        size_t next = c->upsample ? cur * 2 : cur / 2;

        CUdeviceptr src = (s & 1) ? c->d_f64_b : c->d_f64_a;
        CUdeviceptr dst = (s & 1) ? c->d_f64_a : c->d_f64_b;
        if (s == 0) src = c->d_f64_a;

        CUfunction fn = c->upsample ? c->fn_fir_up_f64 : c->fn_fir_down_f64;
        int ic = (int)cur, oc = (int)next;
        void *args[] = { &src, &dst, &ic, &oc };
        launch_kernel(c, fn, args, oc);

        cur = next;
        /* After stage 0, alternate between a and b */
        if (s == 0 && !(stages & 1)) {
            /* Ensure final result ends up in d_f64_b for even stages,
             * d_f64_a for odd stages */
        }
    }

    /* Determine which buffer has the final result */
    CUdeviceptr d_result = (stages & 1) ? c->d_f64_b : c->d_f64_a;
    if (stages == 1) d_result = c->d_f64_b; /* stage 0: a→b */

    pfn_cuMemcpyDtoHAsync(c->h_f64_out, d_result,
                           ext_out * sizeof(double), stream);
    pfn_cuStreamSynchronize(stream);

    /* Strip delay prefix */
    size_t out_skip = ext_out - final_out;
    memcpy(out, c->h_f64_out + out_skip, final_out * sizeof(double));

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

/* ─── FIR Lowpass (same-rate pre-SDM) ─── */

int gpu_cuda_fir_lowpass_setup(cuda_context_t *c, const float *taps, int ntaps) {
    if (!c || !taps || ntaps <= 0 || ntaps > 128) return -1;

    /* Upload lowpass taps to c_lp_taps_d constant memory (fp64) */
    double taps_d[128];
    for (int i = 0; i < ntaps; i++) taps_d[i] = (double)taps[i];

    CUdeviceptr d_lp_taps;
    size_t sz;
    if (pfn_cuModuleGetGlobal(&d_lp_taps, &sz, c->module, "c_lp_taps_d") != CUDA_SUCCESS)
        return -1;
    if (pfn_cuMemcpyHtoD(d_lp_taps, taps_d, ntaps * sizeof(double)) != CUDA_SUCCESS)
        return -1;

    CUdeviceptr d_lp_ntaps;
    if (pfn_cuModuleGetGlobal(&d_lp_ntaps, &sz, c->module, "c_lp_ntaps") != CUDA_SUCCESS)
        return -1;
    if (pfn_cuMemcpyHtoD(d_lp_ntaps, &ntaps, sizeof(int)) != CUDA_SUCCESS)
        return -1;

    /* Allocate per-channel history buffers (ntaps-1 samples, fp64) */
    int hist_len = ntaps - 1;
    if (c->lp_hist_len < hist_len) {
        if (c->d_lp_hist) pfn_cuMemFree(c->d_lp_hist);
        for (int i = 0; i < BOXCAR_MAX_CH; i++) {
            free(c->h_lp_hist[i]);
            c->h_lp_hist[i] = (float *)malloc((size_t)hist_len * sizeof(double));
            if (c->h_lp_hist[i]) {
                double *hist_d = (double *)c->h_lp_hist[i];
                for (int j = 0; j < hist_len; j++)
                    hist_d[j] = (j & 1) ? -1.0 : 1.0;
                c->lp_hist_valid[i] = true;
            }
        }
        pfn_cuMemAlloc(&c->d_lp_hist, (size_t)hist_len * sizeof(double));
        c->lp_hist_len = hist_len;
    }
    return 0;
}

int gpu_cuda_fir_lowpass(cuda_context_t *c, const float *in, float *out,
                          size_t count, float gain) {
    /* Delegate to fp64 variant with TLS narrowing buffer */
    static __declspec(thread) double *tls_f64 = NULL;
    static __declspec(thread) size_t tls_cap = 0;
    if (tls_cap < count) {
        free(tls_f64);
        tls_f64 = (double *)malloc(count * sizeof(double));
        tls_cap = tls_f64 ? count : 0;
    }
    if (!tls_f64) return -1;

    int r = gpu_cuda_fir_lowpass_f64(c, in, tls_f64, count, (double)gain);
    if (r != 0) return r;

    for (size_t i = 0; i < count; i++)
        out[i] = (float)tls_f64[i];
    return 0;
}

/* fp64 output variant — writes directly to double buffer, no float narrowing */
int gpu_cuda_fir_lowpass_f64(cuda_context_t *c, const float *in, double *out,
                               size_t count, double gain) {
    if (!c || count < GPU_MIN_SAMPLES) return -1;

    CUresult rc;
    rc = pfn_cuCtxSetCurrent(c->context);
    if (rc != CUDA_SUCCESS) return -1;

    CUstream stream = c->sdm_stream ? c->sdm_stream : c->streams[0];

    size_t bytes_d = count * sizeof(double);

    /* Per-context device buffers (NOT static — static pointers become
     * invalid when the CUDA context is destroyed and recreated). */
    if (c->lp_f64_cap < count) {
        if (c->d_lp_f64_in)  pfn_cuMemFree(c->d_lp_f64_in);
        if (c->d_lp_f64_out) pfn_cuMemFree(c->d_lp_f64_out);
        c->d_lp_f64_in = 0; c->d_lp_f64_out = 0;
        rc = pfn_cuMemAlloc(&c->d_lp_f64_in, bytes_d);
        if (rc != CUDA_SUCCESS) { c->lp_f64_cap = 0; return -1; }
        rc = pfn_cuMemAlloc(&c->d_lp_f64_out, bytes_d);
        if (rc != CUDA_SUCCESS) { pfn_cuMemFree(c->d_lp_f64_in); c->d_lp_f64_in = 0; c->lp_f64_cap = 0; return -1; }
        c->lp_f64_cap = count;
    }

    /* Host staging buffer */
    if (c->h_lp_f64_cap < count) {
        free(c->h_lp_f64_in);
        c->h_lp_f64_in = (double *)malloc(bytes_d);
        c->h_lp_f64_cap = c->h_lp_f64_in ? count : 0;
    }
    if (!c->h_lp_f64_cap) return -1;

    for (size_t i = 0; i < count; i++)
        c->h_lp_f64_in[i] = (double)in[i];

    rc = pfn_cuMemcpyHtoDAsync(c->d_lp_f64_in, c->h_lp_f64_in, bytes_d, stream);
    if (rc != CUDA_SUCCESS) return -1;

    int ch = c->lp_ch % BOXCAR_MAX_CH;
    c->lp_ch++;

    CUdeviceptr hist_ptr = (CUdeviceptr)0;
    if (c->lp_hist_valid[ch] && c->h_lp_hist[ch] && c->lp_hist_len > 0) {
        rc = pfn_cuMemcpyHtoDAsync(c->d_lp_hist, c->h_lp_hist[ch],
                                    (size_t)c->lp_hist_len * sizeof(double), stream);
        if (rc == CUDA_SUCCESS)
            hist_ptr = c->d_lp_hist;
    }

    int cnt = (int)count;
    void *args[] = { &c->d_lp_f64_in, &c->d_lp_f64_out, &cnt, &gain, &hist_ptr };
    launch_kernel_on(c, c->fn_fir_lowpass, args, (int)count, stream);

    /* Download fp64 output directly to caller's double buffer */
    rc = pfn_cuMemcpyDtoHAsync(out, c->d_lp_f64_out, bytes_d, stream);
    if (rc != CUDA_SUCCESS) return -1;
    rc = pfn_cuStreamSynchronize(stream);
    if (rc != CUDA_SUCCESS) return -1;

    /* Save last hist_len INPUT samples as fp64 FIR delay line */
    if (count >= (size_t)c->lp_hist_len && c->h_lp_hist[ch]) {
        double *hist_d = (double *)c->h_lp_hist[ch];
        for (size_t i = 0; i < (size_t)c->lp_hist_len; i++)
            hist_d[i] = c->h_lp_f64_in[count - (size_t)c->lp_hist_len + i];
        c->lp_hist_valid[ch] = true;
    }

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
int gpu_cuda_trellis(cuda_context_t *c, const double *in, float *out,
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

    /* Upload input (fp64) + segment descriptors */
    CUdeviceptr d_seg_starts, d_seg_out_starts;
    pfn_cuMemAlloc(&d_seg_starts, (size_t)num_segs * sizeof(int));
    pfn_cuMemAlloc(&d_seg_out_starts, (size_t)num_segs * sizeof(int));
    pfn_cuMemcpyHtoDAsync(c->d_sdm_in, in, count * sizeof(double), stream);
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

    /* Build per-segment total-sizes and output-caps arrays for new kernel.
     * Old path: no overlap, all segments have same total and output D. */
    int *h_seg_totals = (int *)malloc((size_t)num_segs * sizeof(int));
    int *h_seg_out_caps = (int *)malloc((size_t)num_segs * sizeof(int));
    CUdeviceptr d_seg_totals, d_seg_out_caps;
    pfn_cuMemAlloc(&d_seg_totals, (size_t)num_segs * sizeof(int));
    pfn_cuMemAlloc(&d_seg_out_caps, (size_t)num_segs * sizeof(int));
    for (int i = 0; i < num_segs; i++) {
        h_seg_totals[i] = seg_total;
        h_seg_out_caps[i] = D;
    }
    pfn_cuMemcpyHtoDAsync(d_seg_totals, h_seg_totals,
                           (size_t)num_segs * sizeof(int), stream);
    pfn_cuMemcpyHtoDAsync(d_seg_out_caps, h_seg_out_caps,
                           (size_t)num_segs * sizeof(int), stream);

    /* Replicate seg0 init state to all segments for new kernel */
    CUdeviceptr d_all_init_s = (CUdeviceptr)0, d_all_init_c = (CUdeviceptr)0;
    if (seg0_init_s) {
        pfn_cuMemAlloc(&d_all_init_s, (size_t)num_segs * (size_t)nc * 8 * sizeof(double));
        pfn_cuMemAlloc(&d_all_init_c, (size_t)num_segs * (size_t)nc * sizeof(double));
        for (int i = 0; i < num_segs; i++) {
            pfn_cuMemcpyDtoDAsync(d_all_init_s + (size_t)i * (size_t)nc * 8 * sizeof(double),
                                   seg0_init_s, (size_t)nc * 8 * sizeof(double), stream);
            pfn_cuMemcpyDtoDAsync(d_all_init_c + (size_t)i * (size_t)nc * sizeof(double),
                                   seg0_init_c, (size_t)nc * sizeof(double), stream);
        }
    }

    /* Launch kernel */
    int block_size = 2 * nc;
    if (block_size < 32) block_size = 32;
    int M_param = M;
    int overlap_param = 0;
    int ch_stride_in = (int)count;
    int ch_stride_out = (int)total_out;
    CUdeviceptr null_counts = (CUdeviceptr)0;
    CUdeviceptr null_mid_s = (CUdeviceptr)0;
    CUdeviceptr null_mid_c = (CUdeviceptr)0;
    int D_param_old = D;
    /* Upload persistent seed as ext_seed for chunk continuity.
     * On first chunk, ext_seed is NULL (seed from init_states only).
     * On subsequent chunks, ext_seed has the full trellis state
     * (history/path/next) from the last segment of the previous chunk. */
    CUdeviceptr ext_seed_ptr = 0;
    if (c->persistent_seed_valid) {
        if (!c->d_persistent_seed)
            pfn_cuMemAlloc(&c->d_persistent_seed, EXT_SEED_SIZE);
        if (c->d_persistent_seed) {
            pfn_cuMemcpyHtoDAsync(c->d_persistent_seed,
                                   c->h_persistent_seed, EXT_SEED_SIZE, stream);
            ext_seed_ptr = c->d_persistent_seed;
        }
    }

    /* Allocate final_seed output buffer (reused across chunks) */
    static CUdeviceptr s_d_final_seed = 0;
    if (!s_d_final_seed)
        pfn_cuMemAlloc(&s_d_final_seed, EXT_SEED_SIZE);

    void *args[] = {
        &c->d_sdm_in, &c->d_sdm_out,
        &d_seg_starts, &d_seg_out_starts,
        &d_seg_totals, &d_seg_out_caps,
        &M_param, &nc, &lat, &overlap_param, &num_segs,
        &ch_stride_in, &ch_stride_out,
        &d_all_init_s, &d_all_init_c,
        &c->d_all_final_states, &c->d_all_final_costs,
        &null_counts,
        &D_param_old, &null_mid_s, &null_mid_c,
        &ext_seed_ptr,
        &s_d_final_seed
    };
    pfn_cuLaunchKernel(c->fn_trellis_parallel,
                        (unsigned)num_segs, 1, 1,
                        (unsigned)block_size, 1, 1,
                        0, stream, args, NULL);
    pfn_cuStreamSynchronize(stream);

    /* Download final_seed for next chunk's ext_seed */
    if (s_d_final_seed) {
        pfn_cuMemcpyDtoH(c->h_persistent_seed, s_d_final_seed, EXT_SEED_SIZE);
        c->persistent_seed_valid = true;
    }

    /* Copy last segment's final NTF state to persistent buffer */
    {
        size_t last_off = (size_t)(num_segs - 1) * (size_t)nc;
        pfn_cuMemcpyDtoD(c->d_trellis_states,
                          c->d_all_final_states + last_off * 8 * sizeof(double),
                          (size_t)nc * 8 * sizeof(double));
        pfn_cuMemcpyDtoD(c->d_trellis_costs,
                          c->d_all_final_costs + last_off * sizeof(double),
                          (size_t)nc * sizeof(double));
    }
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
    pfn_cuMemFree(d_seg_totals);
    pfn_cuMemFree(d_seg_out_caps);
    if (d_all_init_s) pfn_cuMemFree(d_all_init_s);
    if (d_all_init_c) pfn_cuMemFree(d_all_init_c);
    free(h_seg_starts);
    free(h_seg_out_starts);
    free(h_seg_totals);
    free(h_seg_out_caps);

    /* Log timing */
    {
        extern void trellis_log_c(const char *);
        double kernel_ms = (double)(t_kernel.QuadPart - t_start_qpc.QuadPart) * 1000.0 / (double)t_freq.QuadPart;
        double total_ms = (double)(t_end.QuadPart - t_start_qpc.QuadPart) * 1000.0 / (double)t_freq.QuadPart;
        double audio_ms = (double)count / (44100.0 * 2.0 * (double)c->trellis_lat) * 1000.0;
        char msg[256];
        sprintf_s(msg, sizeof(msg),
            "[GPU CUDA SBVD] %zu samples, %d/%d segs/SMs, %d cands, M=%d D=%d L=%d: kernel=%.1fms total=%.1fms (%.2fx RT)",
            count, num_segs, c->num_sms, nc, M, D, L, kernel_ms, total_ms, total_ms / audio_ms);
        trellis_log_c(msg);
    }

    c->trellis_state_valid = true;
    return 0;
}

/* ─── DAS (Density-Aligned Stitching) GPU Pipeline ─── */

int gpu_cuda_trellis_das(cuda_context_t *c, const double *in, float *out,
                          size_t count, int num_channels) {
    if (!c || !c->fn_trellis_parallel || !c->fn_das_density_scan ||
        !c->fn_das_assemble || c->trellis_cands < 2)
        return -1;

    pfn_cuCtxSetCurrent(c->context);

    int nc = c->trellis_cands;
    int lat = c->trellis_lat;
    int L = lat;
    int num_segs = c->num_sms * 3;
    if (num_segs < 1) num_segs = 1;

    int D = (int)(count / (size_t)num_segs);

    /* Convergence warmup: scale with lat but cap to keep overhead low.
     * With many segments, M overhead dominates — use 8×lat minimum. */
    int M = 8 * lat;
    if (M < 256) M = 256;
    if (M > 4096) M = 4096;

    /* Adaptive overlap: min(16×lat, D/4) — keep overhead under 20% */
    int das_overlap = 16 * lat;
    if (das_overlap > D / 4) das_overlap = D / 4;
    if (das_overlap < lat) das_overlap = lat;

    /* Ensure D is large enough for meaningful segments */
    size_t min_seg = (size_t)(M + L + das_overlap + 256);
    while (num_segs > 1 && (size_t)D < min_seg) {
        num_segs--;
        D = (int)(count / (size_t)num_segs);
        das_overlap = 32 * lat;
        if (das_overlap > D / 2) das_overlap = D / 2;
        if (das_overlap < lat) das_overlap = lat;
    }
    if (num_segs < 1) num_segs = 1;

    /* Fall back to non-DAS path for single segment */
    if (num_segs <= 1) {
        /* Single segment — no stitching needed, process each channel */
        for (int ch = 0; ch < num_channels; ch++)
            gpu_cuda_trellis(c, in + ch * count, out + ch * count, count);
        return 0;
    }

    CUstream stream = c->sdm_stream;

    /* Save previous chunk's persistent NTF state for chunk boundary re-encoding.
     * Downloaded before d_trellis_states is overwritten at the end. */
    double h_prev_states[8 * 8];  /* max nc=8, order=8 */
    bool prev_state_valid = c->trellis_state_valid;
    if (prev_state_valid) {
        pfn_cuMemcpyDtoH(h_prev_states, c->d_trellis_states,
                          (size_t)nc * 8 * sizeof(double));
    }

    /* Upload NTF constants */
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

    /* Build per-segment descriptors */
    int *h_seg_starts     = (int *)calloc((size_t)num_segs, sizeof(int));
    int *h_seg_out_starts = (int *)calloc((size_t)num_segs, sizeof(int));
    int *h_seg_totals     = (int *)calloc((size_t)num_segs, sizeof(int));
    int *h_seg_out_caps   = (int *)calloc((size_t)num_segs, sizeof(int));
    if (!h_seg_starts || !h_seg_out_starts || !h_seg_totals || !h_seg_out_caps) {
        free(h_seg_starts); free(h_seg_out_starts);
        free(h_seg_totals); free(h_seg_out_caps);
        return -1;
    }

    /* Segment layout with jittered boundaries.
     * Randomize segment sizes by ±30% to convert periodic stitch artifacts
     * into broadband noise (+10-15 dB perceptual headroom per research).
     * Uses a simple hash-based PRNG seeded from chunk sample count
     * for deterministic but aperiodic boundaries. */
    int *seg_D = (int *)calloc((size_t)num_segs, sizeof(int));
    {
        /* Distribute samples across segments with jitter */
        unsigned rng = (unsigned)count ^ 0xDEADBEEF;
        int remaining = (int)count;
        for (int i = 0; i < num_segs; i++) {
            if (i == num_segs - 1) {
                seg_D[i] = remaining;
            } else {
                /* Jitter: D ± 30% */
                rng = rng * 1103515245 + 12345;
                int jitter_range = D * 30 / 100;
                int jitter = (int)((rng >> 16) % (2 * (unsigned)jitter_range + 1)) - jitter_range;
                int this_D = D + jitter;
                if (this_D < das_overlap * 4) this_D = das_overlap * 4;
                if (this_D > remaining - (num_segs - 1 - i) * (das_overlap * 4))
                    this_D = remaining - (num_segs - 1 - i) * (das_overlap * 4);
                if (this_D < 1) this_D = 1;
                seg_D[i] = this_D;
                remaining -= this_D;
            }
        }
    }

    /* Build segment descriptors using jittered sizes */
    size_t out_offset = 0;
    int seg_pos = 0;  /* cumulative input position */
    for (int i = 0; i < num_segs; i++) {
        int this_D = seg_D[i];
        int in_start = seg_pos - M;
        if (in_start < 0) in_start = 0;

        int out_cap;
        int seg_total;
        if (i < num_segs - 1) {
            out_cap = this_D + das_overlap;
            seg_total = M + this_D + das_overlap + L;
        } else {
            out_cap = this_D;
            seg_total = M + this_D + L;
        }

        /* Clamp to input bounds */
        if (in_start + seg_total > (int)count)
            seg_total = (int)count - in_start;
        if (seg_total < 0) seg_total = 0;

        h_seg_starts[i] = in_start;
        h_seg_out_starts[i] = (int)out_offset;
        h_seg_totals[i] = seg_total;
        h_seg_out_caps[i] = out_cap;
        out_offset += (size_t)out_cap;
        seg_pos += this_D;
    }
    free(seg_D);

    size_t total_seg_out = out_offset;  /* per-channel total with overlap */
    size_t total_final = (size_t)count;  /* per-channel stitched = input count */

    /* Ensure DAS device buffers are large enough */
    if (c->das_alloc_segs < (size_t)num_segs ||
        c->das_alloc_samples < total_seg_out * (size_t)num_channels) {
        /* Free old */
        if (c->d_das_seg_out)       pfn_cuMemFree(c->d_das_seg_out);
        if (c->d_das_final)         pfn_cuMemFree(c->d_das_final);
        if (c->d_das_stitch_pos)    pfn_cuMemFree(c->d_das_stitch_pos);
        if (c->d_das_seg_starts)    pfn_cuMemFree(c->d_das_seg_starts);
        if (c->d_das_seg_out_starts)pfn_cuMemFree(c->d_das_seg_out_starts);
        if (c->d_das_seg_total_sizes)pfn_cuMemFree(c->d_das_seg_total_sizes);
        if (c->d_das_seg_out_caps)  pfn_cuMemFree(c->d_das_seg_out_caps);
        if (c->d_das_seg_counts)    pfn_cuMemFree(c->d_das_seg_counts);
        if (c->d_das_init_states)   pfn_cuMemFree(c->d_das_init_states);
        if (c->d_das_init_costs)    pfn_cuMemFree(c->d_das_init_costs);

        {
            int alloc_rc = 0;
            alloc_rc |= (int)pfn_cuMemAlloc(&c->d_das_seg_out, total_seg_out * (size_t)num_channels * sizeof(float));
            alloc_rc |= (int)pfn_cuMemAlloc(&c->d_das_final, total_final * (size_t)num_channels * sizeof(float));
            if (alloc_rc != 0) {
                extern void trellis_log_c(const char *);
                char msg[256];
                snprintf(msg, sizeof(msg), "GPU DAS alloc FAILED: rc=%d seg_out=%zu final=%zu",
                         alloc_rc, total_seg_out * (size_t)num_channels * sizeof(float),
                         total_final * (size_t)num_channels * sizeof(float));
                trellis_log_c(msg);
            }
        }
        pfn_cuMemAlloc(&c->d_das_stitch_pos, (size_t)(num_segs - 1) * sizeof(int));
        pfn_cuMemAlloc(&c->d_das_seg_starts, (size_t)num_segs * sizeof(int));
        pfn_cuMemAlloc(&c->d_das_seg_out_starts, (size_t)num_segs * sizeof(int));
        pfn_cuMemAlloc(&c->d_das_seg_total_sizes, (size_t)num_segs * sizeof(int));
        pfn_cuMemAlloc(&c->d_das_seg_out_caps, (size_t)num_segs * sizeof(int));
        pfn_cuMemAlloc(&c->d_das_seg_counts, (size_t)num_segs * (size_t)num_channels * sizeof(int));
        pfn_cuMemAlloc(&c->d_das_init_states, (size_t)num_segs * (size_t)nc * 8 * sizeof(double));
        pfn_cuMemAlloc(&c->d_das_init_costs, (size_t)num_segs * (size_t)nc * sizeof(double));
        if (c->d_das_mid_states) pfn_cuMemFree(c->d_das_mid_states);
        if (c->d_das_mid_costs) pfn_cuMemFree(c->d_das_mid_costs);
        free(c->h_das_mid_states);
        pfn_cuMemAlloc(&c->d_das_mid_states, (size_t)num_segs * (size_t)nc * 8 * sizeof(double));
        pfn_cuMemAlloc(&c->d_das_mid_costs, (size_t)num_segs * (size_t)nc * sizeof(double));
        c->h_das_mid_states = (double *)malloc((size_t)num_segs * (size_t)nc * 8 * sizeof(double));

        c->das_alloc_segs = (size_t)num_segs;
        c->das_alloc_samples = total_seg_out * (size_t)num_channels;
    }

    /* Ensure input buffer large enough for all channels (fp64 for trellis) */
    size_t total_in = count * (size_t)num_channels;
    size_t total_in_bytes = total_in * sizeof(double);
    if (c->sdm_buf_cap < total_in * 2) {  /* *2 because doubles are 2× floats */
        if (c->d_sdm_in) pfn_cuMemFree(c->d_sdm_in);
        pfn_cuMemAlloc(&c->d_sdm_in, total_in_bytes);
        c->sdm_buf_cap = total_in * 2;
    }

    /* Upload all channels' input (fp64) */
    pfn_cuMemcpyHtoDAsync(c->d_sdm_in, in, total_in_bytes, stream);

    /* Upload segment descriptors */
    pfn_cuMemcpyHtoDAsync(c->d_das_seg_starts, h_seg_starts,
                           (size_t)num_segs * sizeof(int), stream);
    pfn_cuMemcpyHtoDAsync(c->d_das_seg_out_starts, h_seg_out_starts,
                           (size_t)num_segs * sizeof(int), stream);
    pfn_cuMemcpyHtoDAsync(c->d_das_seg_total_sizes, h_seg_totals,
                           (size_t)num_segs * sizeof(int), stream);
    pfn_cuMemcpyHtoDAsync(c->d_das_seg_out_caps, h_seg_out_caps,
                           (size_t)num_segs * sizeof(int), stream);

    /* Replicate persistent state to all segment init slots.
     * If state is valid, broadcast; otherwise leave zero-inited. */
    if (c->trellis_state_valid) {
        for (int i = 0; i < num_segs; i++) {
            pfn_cuMemcpyDtoDAsync(
                c->d_das_init_states + (size_t)i * (size_t)nc * 8 * sizeof(double),
                c->d_trellis_states, (size_t)nc * 8 * sizeof(double), stream);
            pfn_cuMemcpyDtoDAsync(
                c->d_das_init_costs + (size_t)i * (size_t)nc * sizeof(double),
                c->d_trellis_costs, (size_t)nc * sizeof(double), stream);
        }
    } else {
        pfn_cuMemsetD8Async(c->d_das_init_states, 0,
                            (size_t)num_segs * (size_t)nc * 8 * sizeof(double), stream);
        pfn_cuMemsetD8Async(c->d_das_init_costs, 0,
                            (size_t)num_segs * (size_t)nc * sizeof(double), stream);
    }

    /* Ensure all-final-states buffer */
    int needed = num_segs * nc;
    if (c->boundary_alloc < needed) {
        if (c->d_all_final_states) pfn_cuMemFree(c->d_all_final_states);
        if (c->d_all_final_costs)  pfn_cuMemFree(c->d_all_final_costs);
        pfn_cuMemAlloc(&c->d_all_final_states, (size_t)needed * 8 * sizeof(double));
        pfn_cuMemAlloc(&c->d_all_final_costs, (size_t)needed * sizeof(double));
        c->boundary_alloc = needed;
    }

    LARGE_INTEGER t_start, t_k1, t_k2, t_k3, t_end, t_freq;
    QueryPerformanceFrequency(&t_freq);
    QueryPerformanceCounter(&t_start);

    /* ═══ 2-Pass SDM: state propagation for artifact-free stitching ═══
     *
     * Pass 1: All segments seeded from global persistent state (parallel).
     *         Collects each segment's final state.
     * Pass 2: Re-run all segments with chained initial states:
     *         seg[i] seeded from seg[i-1]'s Pass-1 final state.
     *         Adjacent segments now have continuous state → smooth stitching. */
    {
        int block_size = 2 * nc;
        if (block_size < 32) block_size = 32;
        int M_param = M;
        int overlap_param = das_overlap;
        int ch_stride_in = (int)count;
        int ch_stride_out = (int)total_seg_out;

        int D_nominal = D;

        /* Self-seeding disabled: stale traceback history from previous chunk's
         * signal biases NTF convergence during warmup, degrading quality by ~29% RMS.
         * Zero history (neutral) produces better convergence than stale history.
         * NTF state continuity is provided by init_states (sufficient for quality).
         * Chunk boundary pop (~0.0015) from traceback latency gap is acceptable. */
        CUdeviceptr ext_seed_ptr = 0;
        CUdeviceptr null_final_seed = 0;

        void *args_p1[] = {
            &c->d_sdm_in, &c->d_das_seg_out,
            &c->d_das_seg_starts, &c->d_das_seg_out_starts,
            &c->d_das_seg_total_sizes, &c->d_das_seg_out_caps,
            &M_param, &nc, &lat, &overlap_param, &num_segs,
            &ch_stride_in, &ch_stride_out,
            &c->d_das_init_states, &c->d_das_init_costs,
            &c->d_all_final_states, &c->d_all_final_costs,
            &c->d_das_seg_counts,
            &D_nominal, &c->d_das_mid_states, &c->d_das_mid_costs,
            &ext_seed_ptr,
            &null_final_seed  /* pass 1: don't save final seed */
        };

        /* Pass 1: all segments from global seed (seg0 gets ext_seed if valid) */
        {
            int launch_rc = (int)pfn_cuLaunchKernel(c->fn_trellis_parallel,
                                (unsigned)num_segs, (unsigned)num_channels, 1,
                                (unsigned)block_size, 1, 1,
                                0, stream, args_p1, NULL);
            if (launch_rc != 0) {
                extern void trellis_log_c(const char *);
                char msg[128];
                snprintf(msg, sizeof(msg), "GPU DAS pass1 launch FAILED: rc=%d segs=%d ch=%d blk=%d",
                         launch_rc, num_segs, num_channels, block_size);
                trellis_log_c(msg);
            }
        }
        {
            int sync_rc = (int)pfn_cuStreamSynchronize(stream);
            if (sync_rc != 0) {
                extern void trellis_log_c(const char *);
                char msg[128];
                snprintf(msg, sizeof(msg), "GPU DAS pass1 sync FAILED: rc=%d", sync_rc);
                trellis_log_c(msg);
            }
        }

        /* Build chained init states for Pass 2 */
        size_t state_stride = (size_t)nc * 8 * sizeof(double);
        size_t cost_stride  = (size_t)nc * sizeof(double);
        for (int i = 1; i < num_segs; i++) {
            pfn_cuMemcpyDtoDAsync(
                c->d_das_init_states + (size_t)i * state_stride,
                c->d_all_final_states + (size_t)(i - 1) * state_stride,
                state_stride, stream);
            pfn_cuMemcpyDtoDAsync(
                c->d_das_init_costs + (size_t)i * cost_stride,
                c->d_all_final_costs + (size_t)(i - 1) * cost_stride,
                cost_stride, stream);
        }

        /* Pass 2: re-run with chained states, capture final_seed from last seg */
        void *args_p2[] = {
            &c->d_sdm_in, &c->d_das_seg_out,
            &c->d_das_seg_starts, &c->d_das_seg_out_starts,
            &c->d_das_seg_total_sizes, &c->d_das_seg_out_caps,
            &M_param, &nc, &lat, &overlap_param, &num_segs,
            &ch_stride_in, &ch_stride_out,
            &c->d_das_init_states, &c->d_das_init_costs,
            &c->d_all_final_states, &c->d_all_final_costs,
            &c->d_das_seg_counts,
            &D_nominal, &c->d_das_mid_states, &c->d_das_mid_costs,
            &ext_seed_ptr,
            &null_final_seed  /* self-seeding disabled */
        };

        pfn_cuLaunchKernel(c->fn_trellis_parallel,
                            (unsigned)num_segs, (unsigned)num_channels, 1,
                            (unsigned)block_size, 1, 1,
                            0, stream, args_p2, NULL);
        pfn_cuStreamSynchronize(stream);

        {
            {
                static int fs_log = 0;
                if (fs_log++ < 5) {
                    unsigned char *s = c->h_persistent_seed;
                    unsigned *p = (unsigned *)(s + 1024);
                    unsigned *nx = (unsigned *)(s + 1056);
                    int hp = *(int *)(s + 1088);
                    int pend = *(int *)(s + 1092);
                    int nc2 = *(int *)(s + 1096);
                    extern void trellis_log_c(const char *);
                    char msg[256];
                    snprintf(msg, sizeof(msg),
                        "GPU final_seed: nc=%d hp=%d pend=%d path=[%u,%u] "
                        "next=[%u,%u] hist=[%02x%02x%02x%02x]",
                        nc2, hp, pend, p[0], p[1], nx[0], nx[1],
                        s[0], s[1], s[2], s[3]);
                    trellis_log_c(msg);
                }
            }
        }
    }
    QueryPerformanceCounter(&t_k1);

    /* ═══ CPU-side DAS: download GPU segments, stitch on CPU ═══
     * Uses the proven CPU DAS density scan + assembly algorithm.
     * This bypasses GPU DAS kernels to validate stitching quality. */
    {
        /* Download ALL segment outputs for ch0 (and ch1 if stereo) */
        size_t seg_out_bytes = total_seg_out * (size_t)num_channels * sizeof(float);
        float *h_seg_out = (float *)malloc(seg_out_bytes);
        if (!h_seg_out) { free(h_seg_starts); free(h_seg_out_starts); free(h_seg_totals); free(h_seg_out_caps); return -1; }
        pfn_cuMemcpyDtoH(h_seg_out, c->d_das_seg_out, seg_out_bytes);

        /* Download actual segment output counts */
        int *h_seg_actual = (int *)calloc((size_t)num_segs * (size_t)num_channels, sizeof(int));
        if (h_seg_actual)
            pfn_cuMemcpyDtoH(h_seg_actual, c->d_das_seg_counts,
                              (size_t)num_segs * (size_t)num_channels * sizeof(int));

        /* Download mid-states for Viterbi re-encoding */
        if (c->h_das_mid_states)
            pfn_cuMemcpyDtoH(c->h_das_mid_states, c->d_das_mid_states,
                              (size_t)num_segs * (size_t)nc * 8 * sizeof(double));

        /* Build NTF filter struct for CPU re-encoding */
        ntf_filter_t re_flt;
        memset(&re_flt, 0, sizeof(re_flt));
        re_flt.order = c->trellis_order;
        for (int k = 0; k < c->trellis_order; k++) {
            re_flt.a[k] = c->trellis_ntf_a[k];
            re_flt.g[k] = c->trellis_ntf_g[k];
        }

        /* CPU DAS stitch with Viterbi re-encoding at boundaries */
        int stitch_positions[512];
        memset(stitch_positions, 0, sizeof(stitch_positions));

        /* Pass 1: find stitch positions on channel 0 */
        {
            float *ch0_segs = h_seg_out;
            /* Use actual output count (from kernel), not capacity */
            size_t write_pos = h_seg_actual ? (size_t)h_seg_actual[0] : (size_t)h_seg_out_caps[0];
            if (write_pos > (size_t)h_seg_out_caps[0]) write_pos = (size_t)h_seg_out_caps[0];
            /* Copy seg0 to final output */
            memcpy(out, ch0_segs + h_seg_out_starts[0], write_pos * sizeof(float));

            for (int seg = 1; seg < num_segs; seg++) {
                float *seg_data = ch0_segs + h_seg_out_starts[seg];
                size_t seg_out_n = h_seg_actual ? (size_t)h_seg_actual[seg] : (size_t)h_seg_out_caps[seg];
                if (seg_out_n > (size_t)h_seg_out_caps[seg]) seg_out_n = (size_t)h_seg_out_caps[seg];
                if (seg_out_n == 0) { stitch_positions[seg] = 0; continue; }

                size_t prev_ovl_start = (write_pos >= (size_t)das_overlap) ?
                                         write_pos - (size_t)das_overlap : 0;
                float *prev_ovl = out + prev_ovl_start;
                float *this_ovl = seg_data;
                size_t ovl_len = write_pos - prev_ovl_start;
                if (ovl_len > seg_out_n) ovl_len = seg_out_n;
                if (ovl_len > (size_t)das_overlap) ovl_len = (size_t)das_overlap;

                /* Trellis-guided transition: re-encode the FULL overlap
                 * region using a proper CPU trellis SDM seeded from
                 * seg_prev's exact integrator state at output[D].
                 *
                 * The re-encoded output is:
                 * - Continuous with seg_prev (same state, no transient)
                 * - Properly noise-shaped (trellis optimized)
                 * - Converges toward seg_next (same input data)
                 *
                 * After re-encoding, find the best analog match between
                 * the re-encoded tail and seg_next's output, then stitch
                 * there. The transition is smooth because both the
                 * re-encoded SDM and seg_next processed the same input
                 * from related (converging) states. */
                int prev_seg = seg - 1;
                float *re_out = NULL;
                size_t re_n = 0;
                int stitch_in_re = (int)ovl_len;  /* fallback: end of re-encoded */

                /* DAS hard stitch at density-matched position,
                 * then compensate the analog discontinuity by flipping
                 * the minimum number of bits to match the running average
                 * across the boundary. */

                /* Find density-matched stitch position */
                int half_w = lat;
                if (half_w > (int)ovl_len / 2) half_w = (int)ovl_len / 2;
                if (half_w < 4) half_w = 4;
                int best_density = 0, best_density_pos = 0;
                for (size_t p = 0; p < ovl_len; p++) {
                    int start = (int)p - half_w;
                    int end   = (int)p + half_w;
                    if (start < 0) start = 0;
                    if (end > (int)ovl_len) end = (int)ovl_len;
                    int matches = 0;
                    for (int w = start; w < end; w++)
                        if (prev_ovl[w] == this_ovl[w]) matches++;
                    if (matches > best_density) {
                        best_density = matches;
                        best_density_pos = (int)p;
                    }
                }
                int best_pos = best_density_pos;
                /* Find nearest exact bit match */
                for (int r = 0; r <= half_w; r++) {
                    int lo = best_density_pos - r;
                    int hi = best_density_pos + r;
                    if (lo >= 0 && lo < (int)ovl_len &&
                        prev_ovl[lo] == this_ovl[lo]) { best_pos = lo; break; }
                    if (hi != lo && hi >= 0 && hi < (int)ovl_len &&
                        prev_ovl[hi] == this_ovl[hi]) { best_pos = hi; break; }
                }

                stitch_positions[seg] = best_pos;
                size_t stitch_at = prev_ovl_start + (size_t)best_pos;
                size_t copy_count = seg_out_n - (size_t)best_pos;
                memcpy(out + stitch_at, seg_data + best_pos,
                       copy_count * sizeof(float));
                write_pos = stitch_at + copy_count;

                /* Analog compensation disabled — bit-flipping corrupts
                 * trellis-optimized noise shaping. CPU parallel SDM
                 * does not use this; DAS density-matched stitching alone
                 * is sufficient for clean boundaries. */
            }
            /* write_pos is the final output count for ch0 */
        }

        /* Pass 2: apply same stitch positions to other channels */
        for (int ch = 1; ch < num_channels; ch++) {
            float *ch_segs = h_seg_out + ch * (int)total_seg_out;
            float *ch_out_ptr = out + ch * (int)total_final;
            /* Use actual counts for this channel */
            size_t seg0_actual = h_seg_actual ?
                (size_t)h_seg_actual[ch * num_segs + 0] : (size_t)h_seg_out_caps[0];
            if (seg0_actual > (size_t)h_seg_out_caps[0]) seg0_actual = (size_t)h_seg_out_caps[0];
            size_t write_pos = seg0_actual;
            memcpy(ch_out_ptr, ch_segs + h_seg_out_starts[0],
                   write_pos * sizeof(float));

            for (int seg = 1; seg < num_segs; seg++) {
                float *seg_data = ch_segs + h_seg_out_starts[seg];
                size_t seg_out_n = h_seg_actual ?
                    (size_t)h_seg_actual[ch * num_segs + seg] : (size_t)h_seg_out_caps[seg];
                if (seg_out_n > (size_t)h_seg_out_caps[seg]) seg_out_n = (size_t)h_seg_out_caps[seg];
                if (seg_out_n == 0) continue;

                size_t prev_ovl_start = (write_pos >= (size_t)das_overlap) ?
                                         write_pos - (size_t)das_overlap : 0;
                int bp = stitch_positions[seg];
                size_t stitch_at = prev_ovl_start + (size_t)bp;
                size_t copy_count = seg_out_n - (size_t)bp;
                memcpy(ch_out_ptr + stitch_at, seg_data + bp,
                       copy_count * sizeof(float));
                write_pos = stitch_at + copy_count;
            }
        }

        free(h_seg_out);
        free(h_seg_actual);
    }
    QueryPerformanceCounter(&t_k2);
    QueryPerformanceCounter(&t_k3);

    /* ── CPU chunk boundary re-encoding ──
     * Re-encode the first ~256 output samples using a CPU SDM seeded from
     * the previous chunk's persistent NTF state. This bridges the traceback
     * latency gap: GPU seg0 starts with zeroed history (neutral but
     * discontinuous), while CPU re-encoding provides perfect continuity.
     *
     * GPU output[0] corresponds to input[M] (after M warmup in seg0).
     * CPU must process input[0..M+re_half-1]: first M consumed for
     * pipeline fill + convergence, last re_half produce replacement output. */
    if (prev_state_valid && M + das_overlap * 2 <= (int)count) {
        /* Re-encode region: CPU produces re_total output samples.
         * First re_replace samples directly replace GPU output.
         * Next das_overlap samples are DAS density-matched with GPU output
         * to find a clean stitch point (same algorithm as segment stitching). */
        int re_replace = das_overlap;     /* guaranteed replacement */
        int re_total = re_replace + das_overlap;  /* total CPU output */

        ntf_filter_t re_flt;
        memset(&re_flt, 0, sizeof(re_flt));
        re_flt.order = c->trellis_order;
        for (int k = 0; k < c->trellis_order; k++) {
            re_flt.a[k] = c->trellis_ntf_a[k];
            re_flt.g[k] = c->trellis_ntf_g[k];
        }

        float *discard_buf = (float *)malloc((size_t)M * sizeof(float));
        float *re_buf = (float *)malloc((size_t)re_total * sizeof(float));
        if (discard_buf && re_buf) {
            for (int ch = 0; ch < num_channels; ch++) {
                float *ch_out = out + ch * (int)count;
                const double *ch_in = in + ch * (int)count;

                sdm_context_t tmp;
                sdm_context_init(&tmp, &re_flt, 8, nc, lat);
                {
                    sdm_state_t *s = tmp.trellis[0].act[0];
                    for (int k = 0; k < c->trellis_order; k++)
                        s->state[k] = h_prev_states[k];
                    s->cost = 0.0;
                }

                /* Warmup: M input samples → fills pipeline + NTF convergence */
                sdm_process_block(&tmp, ch_in, discard_buf, (size_t)M);

                /* CPU re-encode: re_total output samples starting at output[0] */
                sdm_process_block(&tmp, ch_in + M, re_buf, (size_t)re_total);

                /* Direct replacement: output[0..re_replace-1] = CPU output */
                memcpy(ch_out, re_buf, (size_t)re_replace * sizeof(float));

                /* DAS density-matched stitch in the overlap region.
                 * CPU re_buf[re_replace..re_total-1] overlaps with
                 * GPU ch_out[re_replace..re_replace+das_overlap-1]. */
                {
                    float *cpu_ovl = re_buf + re_replace;
                    float *gpu_ovl = ch_out + re_replace;
                    int ovl_n = das_overlap;
                    int hw = lat;
                    if (hw > ovl_n / 2) hw = ovl_n / 2;
                    if (hw < 4) hw = 4;

                    /* Find best density match */
                    int best_d = 0, best_p = 0;
                    for (int p = 0; p < ovl_n; p++) {
                        int s = p - hw, e = p + hw;
                        if (s < 0) s = 0;
                        if (e > ovl_n) e = ovl_n;
                        int m = 0;
                        for (int w = s; w < e; w++)
                            if (cpu_ovl[w] == gpu_ovl[w]) m++;
                        if (m > best_d) { best_d = m; best_p = p; }
                    }
                    /* Find nearest exact bit match */
                    int sp = best_p;
                    for (int r = 0; r <= hw; r++) {
                        int lo = best_p - r, hi = best_p + r;
                        if (lo >= 0 && lo < ovl_n &&
                            cpu_ovl[lo] == gpu_ovl[lo]) { sp = lo; break; }
                        if (hi != lo && hi >= 0 && hi < ovl_n &&
                            cpu_ovl[hi] == gpu_ovl[hi]) { sp = hi; break; }
                    }

                    /* Copy CPU output up to stitch point */
                    if (sp > 0)
                        memcpy(gpu_ovl, cpu_ovl, (size_t)sp * sizeof(float));
                    /* GPU output continues from stitch point onward (already in place) */
                }

                sdm_context_free(&tmp);
            }
        }
        free(discard_buf);
        free(re_buf);
    }

    /* Save last segment's final state as persistent for next chunk.
     * With adaptive segment cap (D≥64K), the 2-pass DAS state chaining
     * is accurate enough — no sequential verification pass needed. */
    {
        size_t last_seg_off = (size_t)(num_segs - 1) * (size_t)nc;
        pfn_cuMemcpyDtoD(c->d_trellis_states,
                          c->d_all_final_states + last_seg_off * 8 * sizeof(double),
                          (size_t)nc * 8 * sizeof(double));
        pfn_cuMemcpyDtoD(c->d_trellis_costs,
                          c->d_all_final_costs + last_seg_off * sizeof(double),
                          (size_t)nc * sizeof(double));
    }

    /* CPU-side DAS already wrote to `out` directly — no GPU download needed */

    QueryPerformanceCounter(&t_end);

    /* Cleanup host arrays */
    free(h_seg_starts); free(h_seg_out_starts);
    free(h_seg_totals); free(h_seg_out_caps);

    /* Log timing */
    {
        extern void trellis_log_c(const char *);
        double k1_ms = (double)(t_k1.QuadPart - t_start.QuadPart) * 1000.0 / (double)t_freq.QuadPart;
        double k2_ms = (double)(t_k2.QuadPart - t_k1.QuadPart) * 1000.0 / (double)t_freq.QuadPart;
        double k3_ms = (double)(t_k3.QuadPart - t_k2.QuadPart) * 1000.0 / (double)t_freq.QuadPart;
        double total_ms = (double)(t_end.QuadPart - t_start.QuadPart) * 1000.0 / (double)t_freq.QuadPart;
        /* Compute actual audio duration from sample rate.
         * DSD rates: lat=32→2822400, lat=64→5644800, lat=128→11289600, lat=256→22579200.
         * Rate = 44100 × 64 × (lat / 32) = 44100 × 2 × lat. */
        double dsd_rate = 44100.0 * 2.0 * (double)lat;
        double audio_ms = (double)count / dsd_rate * 1000.0;
        char msg[320];
        sprintf_s(msg, sizeof(msg),
            "[GPU DAS] %zu samples, %dch, %d/%d segs/SMs, nc=%d, M=%d D=%d ovl=%d L=%d: "
            "sdm=%.1fms scan=%.1fms asm=%.1fms total=%.1fms (%.2fx RT)",
            count, num_channels, num_segs, c->num_sms, nc,
            M, D, das_overlap, L,
            k1_ms, k2_ms, k3_ms, total_ms, total_ms / audio_ms);
        trellis_log_c(msg);
    }

    c->trellis_state_valid = true;
    return 0;
}

/* ─── Extended seed upload ─── */

/* Upload extended seed (history/path/traceback) from CPU SDM context.
 * Call before gpu_cuda_trellis_das() for full state continuity.
 * Without this, GPU segments start with zeroed history → 60% stitch density. */

/* sdm_ext_seed struct size (must match CUDA kernel struct layout) — defined at top */

int gpu_cuda_upload_ext_seed(cuda_context_t *c,
                              const unsigned char *hist, int hist_bytes, int nc,
                              const unsigned *path,
                              const unsigned *next_stored,
                              int hist_pos, int pending) {
    if (!c) return -1;
    pfn_cuCtxSetCurrent(c->context);

    if (!s_d_ext_seed)
        pfn_cuMemAlloc(&s_d_ext_seed, EXT_SEED_SIZE);
    if (!s_d_ext_seed) return -1;

    unsigned char h_seed[EXT_SEED_SIZE];
    memset(h_seed, 0, sizeof(h_seed));
    for (int i = 0; i < nc && i < 8; i++) {
        int copy = hist_bytes < 128 ? hist_bytes : 128;
        memcpy(h_seed + i * 128, hist + i * hist_bytes, (size_t)copy);
    }
    unsigned *p_path = (unsigned *)(h_seed + 1024);
    unsigned *p_next = (unsigned *)(h_seed + 1056);
    for (int i = 0; i < nc && i < 8; i++) {
        p_path[i] = path[i];
        p_next[i] = next_stored[i];
    }
    *(int *)(h_seed + 1088) = hist_pos;
    *(int *)(h_seed + 1092) = pending;
    *(int *)(h_seed + 1096) = nc;

    pfn_cuMemcpyHtoD(s_d_ext_seed, h_seed, EXT_SEED_SIZE);
    return 0;
}

/* Upload array of ext_seeds (one per segment) + per-segment NTF states.
 * seeds: packed array [num_segs × EXT_SEED_SIZE] from CPU boundary capture.
 * states/costs: [num_segs × nc × 8] / [num_segs × nc] NTF integrator states. */
int gpu_cuda_upload_boundary_seeds(cuda_context_t *c,
                                     const unsigned char *seeds, int num_segs,
                                     const double *states, const double *costs,
                                     int nc) {
    if (!c) return -1;
    pfn_cuCtxSetCurrent(c->context);
    CUstream stream = c->sdm_stream;

    size_t total_seed_bytes = (size_t)num_segs * EXT_SEED_SIZE;

    /* Grow ext_seed device buffer for array */
    if (!s_d_ext_seed || total_seed_bytes > EXT_SEED_SIZE) {
        if (s_d_ext_seed) pfn_cuMemFree(s_d_ext_seed);
        pfn_cuMemAlloc(&s_d_ext_seed, total_seed_bytes);
    }
    if (!s_d_ext_seed) return -1;

    pfn_cuMemcpyHtoDAsync(s_d_ext_seed, seeds, total_seed_bytes, stream);

    /* Upload per-segment NTF states */
    size_t state_bytes = (size_t)num_segs * (size_t)nc * 8 * sizeof(double);
    size_t cost_bytes = (size_t)num_segs * (size_t)nc * sizeof(double);

    if (c->das_alloc_segs < (size_t)num_segs) {
        if (c->d_das_init_states) pfn_cuMemFree(c->d_das_init_states);
        if (c->d_das_init_costs) pfn_cuMemFree(c->d_das_init_costs);
        pfn_cuMemAlloc(&c->d_das_init_states, state_bytes);
        pfn_cuMemAlloc(&c->d_das_init_costs, cost_bytes);
    }
    pfn_cuMemcpyHtoDAsync(c->d_das_init_states, states, state_bytes, stream);
    pfn_cuMemcpyHtoDAsync(c->d_das_init_costs, costs, cost_bytes, stream);

    return 0;
}

/* ─── Hybrid CPU+GPU async SDM ─── */

/* Launch GPU segments asynchronously. Returns immediately.
 * Call gpu_cuda_hybrid_finish() to wait and download results.
 *
 * in: fp32 input per channel [count * num_channels], channel-strided
 * seg_starts/sizes/out_caps: per-GPU-segment descriptors (host arrays)
 * first_seg: index of first GPU segment (for output buffer offset)
 * num_gpu_segs: how many segments the GPU processes
 */
int gpu_cuda_hybrid_async(cuda_context_t *c,
                           const double *in, size_t count,
                           int num_channels,
                           const int *seg_starts,
                           const int *seg_out_starts,
                           const int *seg_total_sizes,
                           const int *seg_out_caps,
                           int num_gpu_segs,
                           int overlap,
                           size_t total_seg_out) {
    if (!c || !c->fn_trellis_parallel || !pfn_cuEventCreate || num_gpu_segs < 1)
        return -1;

    pfn_cuCtxSetCurrent(c->context);

    int nc = c->trellis_cands;
    int lat = c->trellis_lat;
    int M = 16 * lat;
    if (M > 4096) M = 4096;
    CUstream stream = c->sdm_stream;

    /* Upload NTF constants */
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

    /* Ensure input buffer (fp64) */
    size_t total_in = count * (size_t)num_channels;
    size_t total_in_bytes = total_in * sizeof(double);
    if (c->sdm_buf_cap < total_in * 2) {
        if (c->d_sdm_in) pfn_cuMemFree(c->d_sdm_in);
        pfn_cuMemAlloc(&c->d_sdm_in, total_in_bytes);
        c->sdm_buf_cap = total_in * 2;
    }

    /* Ensure segment output buffer */
    size_t total_out = total_seg_out * (size_t)num_channels;
    if (c->das_alloc_samples < total_out) {
        if (c->d_das_seg_out) pfn_cuMemFree(c->d_das_seg_out);
        pfn_cuMemAlloc(&c->d_das_seg_out, total_out * sizeof(float));
        c->das_alloc_samples = total_out;
    }

    /* Ensure descriptor buffers.
     * Skip init_states/costs realloc if boundary seeds already uploaded
     * (s_d_ext_seed != 0 means gpu_cuda_upload_boundary_seeds was called). */
    if (c->das_alloc_segs < (size_t)num_gpu_segs) {
        if (c->d_das_seg_starts) pfn_cuMemFree(c->d_das_seg_starts);
        if (c->d_das_seg_out_starts) pfn_cuMemFree(c->d_das_seg_out_starts);
        if (c->d_das_seg_total_sizes) pfn_cuMemFree(c->d_das_seg_total_sizes);
        if (c->d_das_seg_out_caps) pfn_cuMemFree(c->d_das_seg_out_caps);
        if (c->d_das_seg_counts) pfn_cuMemFree(c->d_das_seg_counts);
        pfn_cuMemAlloc(&c->d_das_seg_starts, (size_t)num_gpu_segs * sizeof(int));
        pfn_cuMemAlloc(&c->d_das_seg_out_starts, (size_t)num_gpu_segs * sizeof(int));
        pfn_cuMemAlloc(&c->d_das_seg_total_sizes, (size_t)num_gpu_segs * sizeof(int));
        pfn_cuMemAlloc(&c->d_das_seg_out_caps, (size_t)num_gpu_segs * sizeof(int));
        pfn_cuMemAlloc(&c->d_das_seg_counts, (size_t)num_gpu_segs * (size_t)num_channels * sizeof(int));
        if (!s_d_ext_seed) {
            /* Only alloc init_states/costs if no boundary seeds uploaded */
            if (c->d_das_init_states) pfn_cuMemFree(c->d_das_init_states);
            if (c->d_das_init_costs) pfn_cuMemFree(c->d_das_init_costs);
            pfn_cuMemAlloc(&c->d_das_init_states, (size_t)num_gpu_segs * (size_t)nc * 8 * sizeof(double));
            pfn_cuMemAlloc(&c->d_das_init_costs, (size_t)num_gpu_segs * (size_t)nc * sizeof(double));
        }
        c->das_alloc_segs = (size_t)num_gpu_segs;
    }

    /* Ensure final state buffers */
    int needed = num_gpu_segs * nc;
    if (c->boundary_alloc < needed) {
        if (c->d_all_final_states) pfn_cuMemFree(c->d_all_final_states);
        if (c->d_all_final_costs) pfn_cuMemFree(c->d_all_final_costs);
        pfn_cuMemAlloc(&c->d_all_final_states, (size_t)needed * 8 * sizeof(double));
        pfn_cuMemAlloc(&c->d_all_final_costs, (size_t)needed * sizeof(double));
        c->boundary_alloc = needed;
    }

    /* Ensure host output buffer (pinned) */
    if (c->hybrid_host_cap < total_out) {
        if (c->hybrid_host_out) pfn_cuMemFreeHost(c->hybrid_host_out);
        pfn_cuMemAllocHost((void **)&c->hybrid_host_out, total_out * sizeof(float));
        c->hybrid_host_cap = c->hybrid_host_out ? total_out : 0;
    }

    /* Upload input (fp64) */
    pfn_cuMemcpyHtoDAsync(c->d_sdm_in, in, total_in_bytes, stream);

    /* Upload segment descriptors */
    pfn_cuMemcpyHtoDAsync(c->d_das_seg_starts, seg_starts,
                           (size_t)num_gpu_segs * sizeof(int), stream);
    pfn_cuMemcpyHtoDAsync(c->d_das_seg_out_starts, seg_out_starts,
                           (size_t)num_gpu_segs * sizeof(int), stream);
    pfn_cuMemcpyHtoDAsync(c->d_das_seg_total_sizes, seg_total_sizes,
                           (size_t)num_gpu_segs * sizeof(int), stream);
    pfn_cuMemcpyHtoDAsync(c->d_das_seg_out_caps, seg_out_caps,
                           (size_t)num_gpu_segs * sizeof(int), stream);

    /* Seed GPU segments' NTF state.
     * If per-segment boundary seeds were uploaded via gpu_cuda_upload_boundary_seeds,
     * d_das_init_states already has the correct per-segment states — skip. */
    if (!s_d_ext_seed) {
        /* No boundary seeds — use persistent state for all segments */
        if (c->trellis_state_valid) {
            size_t state_stride = (size_t)nc * 8 * sizeof(double);
            size_t cost_stride = (size_t)nc * sizeof(double);
            for (int i = 0; i < num_gpu_segs; i++) {
                pfn_cuMemcpyDtoDAsync(
                    c->d_das_init_states + (size_t)i * state_stride,
                    c->d_trellis_states, state_stride, stream);
                pfn_cuMemcpyDtoDAsync(
                    c->d_das_init_costs + (size_t)i * cost_stride,
                    c->d_trellis_costs, cost_stride, stream);
            }
        } else {
            pfn_cuMemsetD8Async(c->d_das_init_states, 0,
                (size_t)num_gpu_segs * (size_t)nc * 8 * sizeof(double), stream);
            pfn_cuMemsetD8Async(c->d_das_init_costs, 0,
                (size_t)num_gpu_segs * (size_t)nc * sizeof(double), stream);
        }
    }
    /* else: per-segment states already in d_das_init_states from boundary upload */

    /* Pass 1 only: all GPU segments from persistent state (parallel).
     * Pass 2 runs later via gpu_cuda_hybrid_pass2() after CPU seg0
     * provides its final state for proper state chaining. */
    int block_size = 2 * nc;
    if (block_size < 32) block_size = 32;
    c->hybrid_block_size = block_size;
    c->hybrid_ch_stride_in = (int)count;
    c->hybrid_ch_stride_out = (int)total_seg_out;
    c->hybrid_overlap = overlap;
    c->hybrid_M = M;

    int ch_stride_in = (int)count;
    int ch_stride_out = (int)total_seg_out;
    int D_nominal = seg_out_caps[0];
    CUdeviceptr null_ptr = 0;
    CUdeviceptr null_seed = 0;
    CUdeviceptr null_final = 0;

    void *args[] = {
        &c->d_sdm_in, &c->d_das_seg_out,
        &c->d_das_seg_starts, &c->d_das_seg_out_starts,
        &c->d_das_seg_total_sizes, &c->d_das_seg_out_caps,
        &M, &nc, &lat, &overlap, &num_gpu_segs,
        &ch_stride_in, &ch_stride_out,
        &c->d_das_init_states, &c->d_das_init_costs,
        &c->d_all_final_states, &c->d_all_final_costs,
        &c->d_das_seg_counts,
        &D_nominal, &null_ptr, &null_ptr,
        &null_seed,
        &null_final
    };

    pfn_cuLaunchKernel(c->fn_trellis_parallel,
                        (unsigned)num_gpu_segs, (unsigned)num_channels, 1,
                        (unsigned)block_size, 1, 1,
                        0, stream, args, NULL);

    /* Record event — caller can check/wait for completion */
    if (!c->hybrid_event_valid) {
        pfn_cuEventCreate(&c->hybrid_event, 0x02 /* CU_EVENT_DISABLE_TIMING */);
        c->hybrid_event_valid = true;
    }
    pfn_cuEventRecord(c->hybrid_event, stream);

    /* Save metadata for finish() — copy arrays (caller may free originals) */
    c->hybrid_num_gpu_segs = num_gpu_segs;
    c->hybrid_num_ch = num_channels;
    {
        size_t arr_bytes = (size_t)num_gpu_segs * sizeof(int);
        c->hybrid_out_caps = (int *)realloc(c->hybrid_out_caps, arr_bytes);
        c->hybrid_out_starts = (int *)realloc(c->hybrid_out_starts, arr_bytes);
        if (c->hybrid_out_caps) memcpy(c->hybrid_out_caps, seg_out_caps, arr_bytes);
        if (c->hybrid_out_starts) memcpy(c->hybrid_out_starts, seg_out_starts, arr_bytes);
    }

    return 0;
}

/* Run GPU pass 2 with CPU seg0's final state seeding GPU seg0.
 * Call after CPU seg0 finishes and GPU pass 1 completes.
 * cpu_states: [nc * 8] doubles from CPU seg0's final candidates.
 * cpu_costs: [nc] doubles. */
int gpu_cuda_hybrid_pass2(cuda_context_t *c,
                           const double *cpu_states, const double *cpu_costs) {
    if (!c || !c->hybrid_event_valid)
        return -1;

    pfn_cuCtxSetCurrent(c->context);
    CUstream stream = c->sdm_stream;
    int nc = c->trellis_cands;
    int lat = c->trellis_lat;
    int num_gpu_segs = c->hybrid_num_gpu_segs;
    int num_ch = c->hybrid_num_ch;

    /* Wait for pass 1 to complete */
    pfn_cuEventSynchronize(c->hybrid_event);

    /* Seed GPU seg0: from CPU state if provided, else from persistent state */
    if (cpu_states && cpu_costs) {
        pfn_cuMemcpyHtoDAsync(c->d_das_init_states,
            cpu_states, (size_t)nc * 8 * sizeof(double), stream);
        pfn_cuMemcpyHtoDAsync(c->d_das_init_costs,
            cpu_costs, (size_t)nc * sizeof(double), stream);
    } else if (c->trellis_state_valid) {
        pfn_cuMemcpyDtoDAsync(c->d_das_init_states,
            c->d_trellis_states, (size_t)nc * 8 * sizeof(double), stream);
        pfn_cuMemcpyDtoDAsync(c->d_das_init_costs,
            c->d_trellis_costs, (size_t)nc * sizeof(double), stream);
    }

    /* Chain seg1+ from pass-1 finals */
    {
        size_t state_stride = (size_t)nc * 8 * sizeof(double);
        size_t cost_stride = (size_t)nc * sizeof(double);
        for (int i = 1; i < num_gpu_segs; i++) {
            pfn_cuMemcpyDtoDAsync(
                c->d_das_init_states + (size_t)i * state_stride,
                c->d_all_final_states + (size_t)(i - 1) * state_stride,
                state_stride, stream);
            pfn_cuMemcpyDtoDAsync(
                c->d_das_init_costs + (size_t)i * cost_stride,
                c->d_all_final_costs + (size_t)(i - 1) * cost_stride,
                cost_stride, stream);
        }
    }

    /* Launch pass 2 */
    CUdeviceptr null_ptr = 0;
    CUdeviceptr null_seed = 0;
    CUdeviceptr null_final = 0;
    int M = c->hybrid_M;
    int overlap = c->hybrid_overlap;
    int D_nominal = c->hybrid_out_caps ? c->hybrid_out_caps[0] : 0;

    void *args[] = {
        &c->d_sdm_in, &c->d_das_seg_out,
        &c->d_das_seg_starts, &c->d_das_seg_out_starts,
        &c->d_das_seg_total_sizes, &c->d_das_seg_out_caps,
        &M, &nc, &lat, &overlap, &num_gpu_segs,
        &c->hybrid_ch_stride_in, &c->hybrid_ch_stride_out,
        &c->d_das_init_states, &c->d_das_init_costs,
        &c->d_all_final_states, &c->d_all_final_costs,
        &c->d_das_seg_counts,
        &D_nominal, &null_ptr, &null_ptr,
        &null_seed,
        &null_final
    };

    pfn_cuLaunchKernel(c->fn_trellis_parallel,
                        (unsigned)num_gpu_segs, (unsigned)num_ch, 1,
                        (unsigned)c->hybrid_block_size, 1, 1,
                        0, stream, args, NULL);

    pfn_cuEventRecord(c->hybrid_event, stream);
    return 0;
}

/* Wait for GPU segments to complete, download results into seg_bufs.
 * seg_bufs[seg * num_ch + ch]: pre-allocated output buffer per segment per channel.
 * seg_out_counts[seg * num_ch + ch]: filled with actual output sample count. */
int gpu_cuda_hybrid_finish(cuda_context_t *c,
                            float **seg_bufs,
                            size_t *seg_out_counts,
                            int num_gpu_segs, int num_channels) {
    if (!c || !c->hybrid_event_valid)
        return -1;

    pfn_cuCtxSetCurrent(c->context);

    /* Wait for kernel completion */
    pfn_cuEventSynchronize(c->hybrid_event);

    /* Download segment outputs */
    CUstream stream = c->sdm_stream;
    size_t total_out = c->hybrid_host_cap;
    if (c->hybrid_host_out && total_out > 0) {
        pfn_cuMemcpyDtoH(c->hybrid_host_out, c->d_das_seg_out,
                          total_out * sizeof(float));
    }

    /* Download actual output counts */
    int *h_counts = (int *)calloc((size_t)num_gpu_segs * (size_t)num_channels, sizeof(int));
    if (h_counts) {
        pfn_cuMemcpyDtoH(h_counts, c->d_das_seg_counts,
                          (size_t)num_gpu_segs * (size_t)num_channels * sizeof(int));
    }

    /* Copy per-segment outputs to caller's buffers */
    for (int ch = 0; ch < num_channels; ch++) {
        size_t ch_off = (size_t)ch * c->hybrid_host_cap / (size_t)num_channels;
        for (int seg = 0; seg < num_gpu_segs; seg++) {
            int out_start = c->hybrid_out_starts[seg];
            int out_cap = c->hybrid_out_caps[seg];
            int actual = h_counts ? h_counts[ch * num_gpu_segs + seg] : out_cap;
            if (actual > out_cap) actual = out_cap;

            int buf_idx = seg * num_channels + ch;
            if (seg_bufs[buf_idx]) {
                memcpy(seg_bufs[buf_idx],
                       c->hybrid_host_out + ch_off + out_start,
                       (size_t)actual * sizeof(float));
            }
            if (seg_out_counts)
                seg_out_counts[buf_idx] = (size_t)actual;
        }
    }

    free(h_counts);
    return 0;
}

/* ─── Hawksford intra-step parallel SDM ─── */

int gpu_cuda_trellis_hawksford(cuda_context_t *c, const float *in, float *out,
                                size_t count) {
    if (!c || !c->fn_hawksford) return -1;
    pfn_cuCtxSetCurrent(c->context);

    int lat = c->trellis_lat;
    CUstream stream = c->sdm_stream;

    /* Upload NTF constants (double) to Hawksford module */
    {
        CUdeviceptr d_a, d_g, d_order;
        size_t sz;
        pfn_cuModuleGetGlobal(&d_a, &sz, c->mod_hawksford, "c_ntf_a");
        pfn_cuMemcpyHtoD(d_a, c->trellis_ntf_a, (size_t)c->trellis_order * sizeof(double));
        pfn_cuModuleGetGlobal(&d_g, &sz, c->mod_hawksford, "c_ntf_g");
        pfn_cuMemcpyHtoD(d_g, c->trellis_ntf_g, (size_t)c->trellis_order * sizeof(double));
        pfn_cuModuleGetGlobal(&d_order, &sz, c->mod_hawksford, "c_ntf_order");
        pfn_cuMemcpyHtoD(d_order, &c->trellis_order, sizeof(int));
    }

    if (ensure_sdm_bufs(c, count) != 0) return -1;

    static CUdeviceptr d_final_s = 0, d_final_c = 0;
    static int hawk_alloc = 0;
    if (!hawk_alloc) {
        pfn_cuMemAlloc(&d_final_s, 64 * 8 * sizeof(double));
        pfn_cuMemAlloc(&d_final_c, 64 * sizeof(double));
        hawk_alloc = 1;
    }

    pfn_cuMemcpyHtoDAsync(c->d_sdm_in, in, count * sizeof(float), stream);

    LARGE_INTEGER t0, t1, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    int cnt = (int)count;
    int nc = c->trellis_cands;
    if (nc > 64) nc = 64;
    if (nc < 4) nc = 4;
    int block_threads = 2 * nc;
    if (block_threads < 32) block_threads = 32;
    CUdeviceptr null_ptr = (CUdeviceptr)0;
    void *args[] = {
        &c->d_sdm_in, &c->d_sdm_out, &cnt, &nc, &lat,
        &null_ptr, &null_ptr,
        &d_final_s, &d_final_c
    };
    pfn_cuLaunchKernel(c->fn_hawksford, 1, 1, 1, (unsigned)block_threads, 1, 1,
                        0, stream, args, NULL);
    pfn_cuStreamSynchronize(stream);
    QueryPerformanceCounter(&t1);

    pfn_cuMemcpyDtoHAsync(out, c->d_sdm_out, count * sizeof(float), stream);
    pfn_cuStreamSynchronize(stream);

    {
        extern void trellis_log_c(const char *);
        double ms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;
        double audio_ms = (double)count / (44100.0 * 2.0 * (double)c->trellis_lat) * 1000.0;
        char msg[256];
        sprintf_s(msg, sizeof(msg),
            "[GPU Hawksford] %zu samples, nc=%d, lat=%d: %.1fms (%.2fx RT)",
            count, nc, lat, ms, ms / audio_ms);
        trellis_log_c(msg);
    }
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

/* ─── Multi-bit SDM experiment ─── */

int gpu_cuda_multibit_setup(cuda_context_t *c, int order,
                             const double *ntf_a, const double *ntf_g,
                             double state_limit, int num_levels) {
    if (!c || !c->mod_multibit) return -1;
    pfn_cuCtxSetCurrent(c->context);

    /* Upload NTF coefficients as fp32 to multi-bit constant memory */
    float a32[8], g32[8];
    for (int i = 0; i < order && i < 8; i++) {
        a32[i] = (float)ntf_a[i];
        g32[i] = (float)ntf_g[i];
    }
    CUdeviceptr sym; size_t sz;
    if (pfn_cuModuleGetGlobal(&sym, &sz, c->mod_multibit, "c_mb_ntf_a") == CUDA_SUCCESS)
        pfn_cuMemcpyHtoD(sym, a32, 8 * sizeof(float));
    if (pfn_cuModuleGetGlobal(&sym, &sz, c->mod_multibit, "c_mb_ntf_g") == CUDA_SUCCESS)
        pfn_cuMemcpyHtoD(sym, g32, 8 * sizeof(float));
    if (pfn_cuModuleGetGlobal(&sym, &sz, c->mod_multibit, "c_mb_order") == CUDA_SUCCESS)
        pfn_cuMemcpyHtoD(sym, &order, sizeof(int));
    float sl32 = (float)state_limit;
    if (pfn_cuModuleGetGlobal(&sym, &sz, c->mod_multibit, "c_mb_state_limit") == CUDA_SUCCESS)
        pfn_cuMemcpyHtoD(sym, &sl32, sizeof(float));
    if (pfn_cuModuleGetGlobal(&sym, &sz, c->mod_multibit, "c_mb_num_levels") == CUDA_SUCCESS)
        pfn_cuMemcpyHtoD(sym, &num_levels, sizeof(int));
    return 0;
}

int gpu_cuda_multibit_process(cuda_context_t *c, const double *in, float *out,
                               size_t count, int num_channels, float gain) {
    if (!c || !c->fn_mb_segments || !c->fn_mb_to_1bit || count == 0) return -1;
    pfn_cuCtxSetCurrent(c->context);
    CUstream stream = c->sdm_stream ? c->sdm_stream : c->streams[0];

    int order = c->trellis_order;
    int lat = c->trellis_lat;
    /* Start with 1 segment to validate quality without stitching artifacts.
     * TODO: add DAS stitching for multibit segments once quality is verified. */
    int num_segs = 1;
    int M = 0;  /* no warmup needed for single segment */
    int D = (int)count;

    size_t total_samples = count * (size_t)num_channels;
    size_t in_bytes = total_samples * sizeof(double);
    size_t out_bytes = total_samples * sizeof(float);
    size_t seg_arr_bytes = (size_t)num_segs * sizeof(int);
    size_t state_bytes = (size_t)num_segs * (size_t)order * sizeof(float);

    /* Allocate device buffers */
    CUdeviceptr d_in = 0, d_out = 0, d_mb = 0;
    CUdeviceptr d_seg_starts = 0, d_seg_out_starts = 0;
    CUdeviceptr d_init_states = 0, d_final_states = 0;

    if (pfn_cuMemAlloc(&d_in, in_bytes) != CUDA_SUCCESS) return -1;
    if (pfn_cuMemAlloc(&d_out, out_bytes) != CUDA_SUCCESS) goto fail;
    if (pfn_cuMemAlloc(&d_mb, out_bytes) != CUDA_SUCCESS) goto fail;
    if (pfn_cuMemAlloc(&d_seg_starts, seg_arr_bytes) != CUDA_SUCCESS) goto fail;
    if (pfn_cuMemAlloc(&d_seg_out_starts, seg_arr_bytes) != CUDA_SUCCESS) goto fail;
    if (pfn_cuMemAlloc(&d_init_states, state_bytes) != CUDA_SUCCESS) goto fail;

    /* Upload input */
    pfn_cuMemcpyHtoDAsync(d_in, in, in_bytes, stream);

    /* Build segment descriptors */
    {
        int *h_starts = (int *)malloc(seg_arr_bytes);
        int *h_out_starts = (int *)malloc(seg_arr_bytes);
        float *h_init = (float *)calloc((size_t)num_segs * (size_t)order, sizeof(float));
        if (!h_starts || !h_out_starts || !h_init) { free(h_starts); free(h_out_starts); free(h_init); goto fail; }

        for (int s = 0; s < num_segs; s++) {
            h_starts[s] = s * D;
            h_out_starts[s] = s * D;
            /* All segments start from zero state (first chunk) or
             * could be seeded from CPU persistent state */
        }
        pfn_cuMemcpyHtoDAsync(d_seg_starts, h_starts, seg_arr_bytes, stream);
        pfn_cuMemcpyHtoDAsync(d_seg_out_starts, h_out_starts, seg_arr_bytes, stream);
        pfn_cuMemcpyHtoDAsync(d_init_states, h_init, state_bytes, stream);
        free(h_starts); free(h_out_starts); free(h_init);
    }

    LARGE_INTEGER t0, t1, t2, pf;
    QueryPerformanceFrequency(&pf);
    QueryPerformanceCounter(&t0);

    /* Stage 1: Multi-bit SDM — one block per segment, one thread per block */
    {
        int seg_total = M + D;
        int ch_stride_in = (int)count;
        int ch_stride_out = (int)count;
        void *args[] = {
            &d_in, &d_mb,
            &d_seg_starts, &d_seg_out_starts,
            &seg_total, &M, &D,
            &num_segs, &ch_stride_in, &ch_stride_out,
            &d_init_states, &d_final_states /* NULL */
        };
        /* Grid: (num_segs, num_channels, 1), Block: (1,1,1) */
        CUresult rc = pfn_cuLaunchKernel(c->fn_mb_segments,
            (unsigned)num_segs, (unsigned)num_channels, 1,
            1, 1, 1, 0, stream, args, NULL);
        if (rc != CUDA_SUCCESS) goto fail;
        pfn_cuStreamSynchronize(stream);
    }
    QueryPerformanceCounter(&t1);

    /* Download multibit output for CPU stage 2 */
    {
        size_t mb_bytes = (size_t)(D * num_segs) * (size_t)num_channels * sizeof(float);
        float *h_mb = (float *)malloc(mb_bytes);
        if (!h_mb) goto fail;
        pfn_cuMemcpyDtoHAsync(h_mb, d_mb, mb_bytes, stream);
        pfn_cuStreamSynchronize(stream);

        /* Diagnostic: log multibit output stats */
        {
            float mn = 1e30f, mx = -1e30f;
            double sum = 0;
            size_t check_n = count < 1000 ? count : 1000;
            for (size_t i = 0; i < check_n; i++) {
                float v = h_mb[i];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                sum += (double)(v * v);
            }
            extern void trellis_log_c(const char *);
            char msg[256];
            sprintf_s(msg, sizeof(msg),
                "multibit output: min=%.4f max=%.4f rms=%.4f first=[%.3f,%.3f,%.3f,%.3f,%.3f]",
                mn, mx, sqrt(sum / (double)check_n),
                h_mb[0], h_mb[1], h_mb[2], h_mb[3], h_mb[4]);
            trellis_log_c(msg);
        }

        /* Stage 2: Multi-bit → 1-bit on CPU.
         * Simple 1st-order delta-sigma: integrator += input - output.
         * The multibit input is already noise-shaped by stage 1's NTF,
         * so stage 2 only needs to handle the 4-bit→1-bit requantization.
         * A 1st-order SDM is stable and adds minimal extra noise. */
        size_t per_ch = (size_t)(D * num_segs);
        for (int ch = 0; ch < num_channels; ch++) {
            float *mb_ch = h_mb + ch * per_ch;
            float *out_ch = out + ch * count;
            double integrator = 0.0;
            for (size_t i = 0; i < per_ch && i < count; i++) {
                double x = (double)mb_ch[i] * (double)gain;
                integrator += x;
                double y = (integrator >= 0.0) ? 1.0 : -1.0;
                integrator -= y;
                out_ch[i] = (float)y;
            }
        }
        free(h_mb);
    }

    QueryPerformanceCounter(&t2);
    /* Log timing */
    {
        extern void trellis_log_c(const char *);
        double ms1 = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)pf.QuadPart;
        double ms2 = (double)(t2.QuadPart - t1.QuadPart) * 1000.0 / (double)pf.QuadPart;
        char msg[256];
        sprintf_s(msg, sizeof(msg),
            "[GPU multibit] %zu samples, %dch, %d segs, M=%d D=%d: "
            "stage1=%.1fms stage2=%.1fms total=%.1fms",
            count, num_channels, num_segs, M, D, ms1, ms2, ms1 + ms2);
        trellis_log_c(msg);
    }

    pfn_cuMemFree(d_in); pfn_cuMemFree(d_out); pfn_cuMemFree(d_mb);
    pfn_cuMemFree(d_seg_starts); pfn_cuMemFree(d_seg_out_starts);
    pfn_cuMemFree(d_init_states);
    return 0;

fail:
    if (d_in) pfn_cuMemFree(d_in);
    if (d_out) pfn_cuMemFree(d_out);
    if (d_mb) pfn_cuMemFree(d_mb);
    if (d_seg_starts) pfn_cuMemFree(d_seg_starts);
    if (d_seg_out_starts) pfn_cuMemFree(d_seg_out_starts);
    if (d_init_states) pfn_cuMemFree(d_init_states);
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * GPU Convolution (cuFFT + custom kernels)
 * ═══════════════════════════════════════════════════════════════════════ */

extern void trellis_log_c(const char *);

/* Forward declarations */
void gpu_cuda_conv_free(void *ctx, void *state);

struct gpu_conv_state {
    int            num_partitions;
    int            partition_size;
    int            fft_size;
    cufftHandle    fft_plan;
    CUdeviceptr    d_ir_freq;
    CUdeviceptr    d_fdl_flat;     /* contiguous: [num_parts * fft_size] cdoubles */
    CUdeviceptr   *d_fdl;          /* unused, kept for compat */
    CUdeviceptr    d_input;
    CUdeviceptr    d_accum;
    CUdeviceptr    d_temp;
    double        *h_staging;
    size_t         h_staging_cap;
    int            fdl_pos;
    int            fdl_filled;
    double        *fifo_buf;
    int            fifo_avail;
    int            fifo_read;
    double        *overlap_buf;
    int            buf_pos;
    /* Device arrays for fused kernel */
    CUdeviceptr    d_fdl_ptrs;     /* CUdeviceptr[num_partitions] on device */
    CUdeviceptr    d_fdl_indices;  /* int[num_partitions] on device */
};

int gpu_cuda_conv_max_partitions(void *ctx, uint32_t signal_rate, int P,
                                  int budget_level) {
    cuda_context_t *c = (cuda_context_t *)ctx;
    if (!c || !g_cufft_available || !c->fn_conv_cmulacc) return 0;

    int sms = c->num_sms > 0 ? c->num_sms : 16;
    int clock_mhz = c->gpu_clock_khz > 0 ? c->gpu_clock_khz / 1000 : 1500;
    int fft_size = P * 2;
    int log2n = 0;
    { int v = fft_size; while (v > 1) { v >>= 1; log2n++; } }

    /* Compute theoretical max partitions from GPU throughput */
    double throughput = (double)sms * clock_mhz * 32.0;
    double blocks_per_sec = (double)signal_rate / P;
    double fft_cost = fft_size * log2n * 5.0 * 2.0;
    double mul_cost_per_part = fft_size * 6.0;
    double budget_flops = throughput * 1e6 / blocks_per_sec;
    int theoretical_max = (int)((budget_flops - fft_cost) / mul_cost_per_part);

    /* Apply target utilization cap based on real-world calibration.
     *
     * Measured: RTX 5080 SM=84, DSD64, 64 partitions (P=4096) = 80% GPU.
     * Target: ~70% GPU utilization for real-time headroom.
     *
     * Reference point: 64 parts × (2822400/P) blocks/s = the GPU load.
     * For other rates, scale partitions inversely with rate.
     * Per-SM calibration: 64 parts / 84 SMs = 0.76 parts/SM at DSD64. */
    double ref_rate = 2822400.0;
    double parts_per_sm = 0.6;  /* ~70% target (0.76 was 80%) */
    int rate_scaled = (int)(parts_per_sm * sms * ref_rate / (double)signal_rate);

    /* Use the smaller of theoretical and rate-scaled */
    int max_parts = theoretical_max < rate_scaled ? theoretical_max : rate_scaled;

    /* Apply budget level: High=100%, Medium=50%, Low=25% */
    if (budget_level == 1)
        max_parts = max_parts / 2;
    else if (budget_level >= 2)
        max_parts = max_parts / 4;

    if (max_parts < 4) max_parts = 4;
    if (max_parts > 4096) max_parts = 4096;

    {
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "conv: GPU budget: SM=%d clock=%dMHz max=%d parts (P=%d rate=%u theory=%d scaled=%d)",
                 sms, clock_mhz, max_parts, P, signal_rate, theoretical_max, rate_scaled);
        trellis_log_c(msg);
    }
    return max_parts;
}

void *gpu_cuda_conv_init(void *ctx, int num_partitions,
                                      int partition_size, int fft_size,
                                      const void *ir_freq_host) {
    cuda_context_t *c = (cuda_context_t *)ctx;
    if (!c || !g_cufft_available || !c->fn_conv_cmulacc) return NULL;

    struct gpu_conv_state *s = (struct gpu_conv_state *)calloc(1, sizeof(*s));
    if (!s) return NULL;

    s->num_partitions = num_partitions;
    s->partition_size = partition_size;
    s->fft_size = fft_size;

    size_t slot_bytes = (size_t)fft_size * 16;  /* sizeof(cuDoubleComplex) */

    /* Create cuFFT plan */
    if (pfn_cufftPlan1d(&s->fft_plan, fft_size, CUFFT_Z2Z, 1) != CUFFT_SUCCESS)
        goto fail;
    if (pfn_cufftSetStream)
        pfn_cufftSetStream(s->fft_plan, c->streams[0]);

    /* Upload IR partitions */
    if (pfn_cuMemAlloc(&s->d_ir_freq, (size_t)num_partitions * slot_bytes) != CUDA_SUCCESS)
        goto fail;
    pfn_cuMemcpyHtoD(s->d_ir_freq, ir_freq_host, (size_t)num_partitions * slot_bytes);

    /* Allocate contiguous FDL buffer */
    if (pfn_cuMemAlloc(&s->d_fdl_flat, (size_t)num_partitions * slot_bytes) != CUDA_SUCCESS)
        goto fail;
    pfn_cuMemsetD8Async(s->d_fdl_flat, 0, (size_t)num_partitions * slot_bytes, c->streams[0]);

    /* Work buffers */
    if (pfn_cuMemAlloc(&s->d_input, slot_bytes) != CUDA_SUCCESS) goto fail;
    if (pfn_cuMemAlloc(&s->d_accum, slot_bytes) != CUDA_SUCCESS) goto fail;
    if (pfn_cuMemAlloc(&s->d_temp, slot_bytes) != CUDA_SUCCESS) goto fail;

    /* Pinned host staging */
    {
        size_t cap = (size_t)fft_size * 2;
        if (pfn_cuMemAllocHost((void **)&s->h_staging, cap * sizeof(double)) != CUDA_SUCCESS)
            s->h_staging = (double *)malloc(cap * sizeof(double));
        s->h_staging_cap = cap;
    }

    /* Host FIFO + overlap */
    s->fifo_buf = (double *)calloc(partition_size * 2, sizeof(double));
    s->overlap_buf = (double *)calloc(2 * partition_size, sizeof(double));
    if (!s->fifo_buf || !s->overlap_buf) goto fail;

    /* Device index array for fused multiply-accumulate kernel */
    if (pfn_cuMemAlloc(&s->d_fdl_indices, (size_t)num_partitions * sizeof(int)) != CUDA_SUCCESS)
        goto fail;

    pfn_cuStreamSynchronize(c->streams[0]);

    {
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "conv: GPU init: %d parts P=%d VRAM=%.1fMB",
                 num_partitions, partition_size,
                 (double)(num_partitions * 2 + 3) * slot_bytes / (1024.0 * 1024.0));
        trellis_log_c(msg);
    }
    return s;

fail:
    gpu_cuda_conv_free(ctx, s);
    return NULL;
}

static void gpu_conv_do_partition(cuda_context_t *c, struct gpu_conv_state *s) {
    int fft_size = s->fft_size;
    int P = s->partition_size;
    int np = s->num_partitions;
    unsigned grid = (unsigned)((fft_size + 255) / 256);
    size_t slot_bytes = (size_t)fft_size * 16;

    /* Upload overlap buffer as real, pack to complex on GPU */
    pfn_cuMemcpyHtoDAsync(s->d_temp, s->overlap_buf,
                           2 * P * sizeof(double), c->streams[0]);
    {
        int count_arg = 2 * P;
        int fs_arg = fft_size;
        void *args[] = { &s->d_temp, &s->d_input, &count_arg, &fs_arg };
        pfn_cuLaunchKernel(c->fn_conv_r2c, grid, 1, 1, 256, 1, 1, 0,
                            c->streams[0], args, NULL);
    }

    /* Forward FFT → FDL[fdl_pos] (contiguous buffer) */
    CUdeviceptr d_fdl_slot = s->d_fdl_flat + (CUdeviceptr)(s->fdl_pos * slot_bytes);
    pfn_cufftExecZ2Z(s->fft_plan, (void *)s->d_input,
                      (void *)d_fdl_slot, CUFFT_FORWARD);

    if (s->fdl_filled < np)
        s->fdl_filled++;

    /* Fused multiply-accumulate: single kernel for all partitions.
     * Upload FDL index array, then launch one kernel. */
    int active = s->fdl_filled < np ? s->fdl_filled : np;
    {
        int indices[4096];
        for (int k = 0; k < active; k++)
            indices[k] = (s->fdl_pos - k + np) % np;
        pfn_cuMemcpyHtoDAsync(s->d_fdl_indices, indices,
                               active * sizeof(int), c->streams[0]);

        int fs_arg = fft_size;
        int na_arg = active;
        void *args[] = { &s->d_fdl_flat, &s->d_ir_freq,
                          &s->d_accum, &s->d_fdl_indices,
                          &fs_arg, &na_arg };
        pfn_cuLaunchKernel(c->fn_conv_fused, grid, 1, 1, 256, 1, 1, 0,
                            c->streams[0], args, NULL);
    }

    /* Inverse FFT */
    pfn_cufftExecZ2Z(s->fft_plan, (void *)s->d_accum,
                      (void *)s->d_input, CUFFT_INVERSE);

    /* Extract second half + download */
    {
        int P_arg = P;
        int fs_arg = fft_size;
        void *args[] = { &s->d_input, &s->d_temp, &P_arg, &fs_arg };
        pfn_cuLaunchKernel(c->fn_conv_extract, (P + 255) / 256, 1, 1,
                            256, 1, 1, 0, c->streams[0], args, NULL);
    }

    /* Download P doubles — NO sync here, defer to batch end */
    pfn_cuMemcpyDtoHAsync(s->h_staging, s->d_temp,
                           P * sizeof(double), c->streams[0]);
    /* Mark that we need sync before reading h_staging */
    s->fdl_pos = (s->fdl_pos + 1) % np;
}

/* Sync GPU stream and normalize the staging buffer */
static void gpu_conv_sync_and_normalize(cuda_context_t *c, struct gpu_conv_state *s) {
    pfn_cuStreamSynchronize(c->streams[0]);
    double norm = 1.0 / (double)s->fft_size;
    int P = s->partition_size;
    for (int i = 0; i < P; i++)
        s->h_staging[i] *= norm;
}

int gpu_cuda_conv_process(void *ctx, void *state,
                           double *buf, size_t count) {
    cuda_context_t *c = (cuda_context_t *)ctx;
    struct gpu_conv_state *s = (struct gpu_conv_state *)state;
    if (!c || !s) return -1;

    int P = s->partition_size;
    int max_new = (int)((s->buf_pos + count) / (size_t)P) + 1;
    int fifo_existing = s->fifo_avail - s->fifo_read;
    if (fifo_existing < 0) fifo_existing = 0;

    size_t total_need = (size_t)fifo_existing + (size_t)max_new * P;

    static __declspec(thread) double *tls_fifo = NULL;
    static __declspec(thread) size_t tls_fifo_sz = 0;
    if (tls_fifo_sz < total_need) {
        free(tls_fifo);
        tls_fifo = (double *)malloc(total_need * sizeof(double));
        tls_fifo_sz = tls_fifo ? total_need : 0;
    }
    if (!tls_fifo) return -1;

    if (fifo_existing > 0)
        memcpy(tls_fifo, s->fifo_buf + s->fifo_read,
               fifo_existing * sizeof(double));
    int fifo_len = fifo_existing;

    /* Feed ALL input */
    for (size_t in_pos = 0; in_pos < count; ) {
        int need = P - s->buf_pos;
        int avail = (int)(count - in_pos);
        int n = (avail < need) ? avail : need;

        memcpy(&s->overlap_buf[P + s->buf_pos], &buf[in_pos],
               n * sizeof(double));
        s->buf_pos += n;
        in_pos += n;

        if (s->buf_pos >= P) {
            gpu_conv_do_partition(c, s);
            gpu_conv_sync_and_normalize(c, s);
            memcpy(&tls_fifo[fifo_len], s->h_staging, P * sizeof(double));
            fifo_len += P;
            memmove(s->overlap_buf, &s->overlap_buf[P],
                    P * sizeof(double));
            s->buf_pos = 0;
        }
    }

    /* Drain exactly count */
    if ((size_t)fifo_len >= count) {
        memcpy(buf, tls_fifo, count * sizeof(double));
        int remain = fifo_len - (int)count;
        if (remain > 0) {
            if ((size_t)remain > (size_t)(P * 2)) {
                free(s->fifo_buf);
                s->fifo_buf = (double *)malloc(remain * sizeof(double));
            }
            if (s->fifo_buf)
                memcpy(s->fifo_buf, &tls_fifo[count],
                       remain * sizeof(double));
            s->fifo_avail = remain;
        } else {
            s->fifo_avail = 0;
        }
        s->fifo_read = 0;
    } else {
        if (fifo_len > 0)
            memcpy(buf, tls_fifo, fifo_len * sizeof(double));
        s->fifo_avail = 0;
        s->fifo_read = 0;
    }

    return 0;
}

void gpu_cuda_conv_free(void *ctx, void *state) {
    struct gpu_conv_state *s = (struct gpu_conv_state *)state;
    if (!s) return;
    (void)ctx;

    if (s->fft_plan && pfn_cufftDestroy)
        pfn_cufftDestroy(s->fft_plan);
    if (s->d_ir_freq) pfn_cuMemFree(s->d_ir_freq);
    if (s->d_fdl_flat) pfn_cuMemFree(s->d_fdl_flat);
    if (s->d_input) pfn_cuMemFree(s->d_input);
    if (s->d_accum) pfn_cuMemFree(s->d_accum);
    if (s->d_temp) pfn_cuMemFree(s->d_temp);
    if (s->d_fdl_indices) pfn_cuMemFree(s->d_fdl_indices);
    if (s->h_staging) pfn_cuMemFreeHost(s->h_staging);
    free(s->fifo_buf);
    free(s->overlap_buf);
    free(s);
}
