/*
 * foo_dsd_trellis — ONNX Runtime ML filter implementation
 *
 * Fully runtime-loaded: no compile-time dependency on ONNX Runtime SDK.
 * Uses LoadLibraryW + GetProcAddress to resolve OrtGetApiBase at runtime.
 * Plugin builds and runs without onnxruntime.dll — the ML filter is
 * simply unavailable.
 *
 * Two modes:
 *   Post-SDM: Input [1,1,hist+N] → Output [1,1,N] (requantized ±1.0)
 *   Pre-SDM:  Input [1,1,N] → Output [1,1,N] (continuous, no requantize)
 *
 * Supports CUDA, DirectML, and CPU execution providers with fallback.
 */

#include "../include/onnx_filter.h"
#include <stdlib.h>
#include <string.h>
#include <windows.h>

/* ─── Minimal ORT C API type definitions (no SDK header needed) ───
 * We access ORT via a void* vtable indexed by slot number, resolved
 * at runtime through OrtGetApiBase()->GetApi().  Slot numbers are
 * stable across ORT versions per the C API contract.
 *
 * Slot mapping (from onnxruntime_c_api.h, verified with ORT 1.24.3):
 *   0  CreateStatus              49 CreateTensorWithDataAsOrtValue
 *   1  GetErrorCode              51 GetTensorMutableData
 *   2  GetErrorMessage           69 CreateCpuMemoryInfo
 *   3  CreateEnv                 76 AllocatorFree
 *   7  CreateSession             78 GetAllocatorWithDefaultOptions
 *   9  Run                       92 ReleaseEnv
 *  10  CreateSessionOptions      93 ReleaseStatus
 *  23  SetSessionGraphOptLevel   94 ReleaseMemoryInfo
 *  24  SetIntraOpNumThreads      95 ReleaseSession
 *  25  SetInterOpNumThreads      96 ReleaseValue
 *  36  SessionGetInputName      100 ReleaseSessionOptions
 *  37  SessionGetOutputName
 */

/* ORT enum constants */
#define ORT_LOGGING_LEVEL_WARNING  2
#define ONNX_TENSOR_ELEMENT_FLOAT  1
#define ORT_ENABLE_ALL             99
#define ORT_ARENA_ALLOCATOR        0
#define ORT_MEM_TYPE_DEFAULT       0
#define ORT_SEQUENTIAL             0   /* ExecutionMode for DirectML */

/* Opaque ORT handles — all accessed through void* */
typedef void OrtEnv;
typedef void OrtSession;
typedef void OrtSessionOptions;
typedef void OrtStatus;
typedef void OrtMemoryInfo;
typedef void OrtValue;
typedef void OrtAllocator;
typedef void OrtRunOptions;

/* ORT API vtable slot indices */
#define ORT_SLOT_CREATE_STATUS                    0
#define ORT_SLOT_GET_ERROR_CODE                   1
#define ORT_SLOT_GET_ERROR_MESSAGE                2
#define ORT_SLOT_CREATE_ENV                       3
#define ORT_SLOT_CREATE_SESSION                   7
#define ORT_SLOT_RUN                              9
#define ORT_SLOT_CREATE_SESSION_OPTIONS           10
#define ORT_SLOT_SET_SESSION_EXECUTION_MODE       13
#define ORT_SLOT_DISABLE_MEM_PATTERN              17
#define ORT_SLOT_SET_GRAPH_OPT_LEVEL             23
#define ORT_SLOT_SET_INTRA_OP_THREADS            24
#define ORT_SLOT_SET_INTER_OP_THREADS            25
#define ORT_SLOT_SESSION_GET_INPUT_NAME          36
#define ORT_SLOT_SESSION_GET_OUTPUT_NAME         37
#define ORT_SLOT_CREATE_TENSOR_WITH_DATA         49
#define ORT_SLOT_GET_TENSOR_MUTABLE_DATA         51
#define ORT_SLOT_CREATE_CPU_MEMORY_INFO          69
#define ORT_SLOT_ALLOCATOR_FREE                  76
#define ORT_SLOT_GET_ALLOCATOR_DEFAULT           78
#define ORT_SLOT_RELEASE_ENV                     92
#define ORT_SLOT_RELEASE_STATUS                  93
#define ORT_SLOT_RELEASE_MEMORY_INFO             94
#define ORT_SLOT_RELEASE_SESSION                 95
#define ORT_SLOT_RELEASE_VALUE                   96
#define ORT_SLOT_RELEASE_SESSION_OPTIONS        100

/* Function pointer typedefs for the slots we use */
typedef OrtStatus *(*ort_CreateEnv_fn)(int, const char *, OrtEnv **);
typedef OrtStatus *(*ort_CreateSession_fn)(OrtEnv *, const wchar_t *, const OrtSessionOptions *, OrtSession **);
typedef OrtStatus *(*ort_Run_fn)(OrtSession *, const OrtRunOptions *, const char *const *, const OrtValue *const *, size_t, const char *const *, size_t, OrtValue **);
typedef OrtStatus *(*ort_CreateSessionOptions_fn)(OrtSessionOptions **);
typedef OrtStatus *(*ort_SetGraphOptLevel_fn)(OrtSessionOptions *, int);
typedef OrtStatus *(*ort_SetThreads_fn)(OrtSessionOptions *, int);
typedef OrtStatus *(*ort_SetSessionExecutionMode_fn)(OrtSessionOptions *, int);
typedef OrtStatus *(*ort_DisableMemPattern_fn)(OrtSessionOptions *);

/* DirectML EP — exported directly from onnxruntime.dll (not vtable) */
typedef OrtStatus *(*OrtDmlAppendEP_fn)(OrtSessionOptions *, int);
/* Extended DML EP — takes pre-created DML device + D3D12 command queue.
 * More reliable than device-ID variant (avoids DXGI adapter enumeration issues). */
typedef OrtStatus *(*OrtDmlAppendEPEx_fn)(OrtSessionOptions *, void * /*IDMLDevice**/, void * /*ID3D12CommandQueue**/);

/* CUDA EP — vtable slot for ORT 1.16+ */
#define ORT_SLOT_CREATE_CUDA_OPTIONS         148
#define ORT_SLOT_SESSION_OPTIONS_APPEND_EP   149
#define ORT_SLOT_RELEASE_CUDA_OPTIONS        150
typedef OrtStatus *(*ort_CreateCudaOptions_fn)(void **);
typedef OrtStatus *(*ort_SessionOptionsAppendEP_fn)(OrtSessionOptions *, const char *, const void *, size_t);

/* CUDA provider options — direct export from onnxruntime.dll */
typedef OrtStatus *(*OrtCudaAppendEP_fn)(OrtSessionOptions *, const void *);

typedef OrtStatus *(*ort_SessionGetName_fn)(OrtSession *, size_t, OrtAllocator *, char **);
typedef OrtStatus *(*ort_CreateTensorWithData_fn)(OrtMemoryInfo *, void *, size_t, const int64_t *, size_t, int, OrtValue **);
typedef OrtStatus *(*ort_GetTensorMutableData_fn)(OrtValue *, void **);
typedef OrtStatus *(*ort_CreateCpuMemoryInfo_fn)(int, int, OrtMemoryInfo **);
typedef OrtStatus *(*ort_AllocatorFree_fn)(OrtAllocator *, void *);
typedef OrtStatus *(*ort_GetAllocatorDefault_fn)(OrtAllocator **);
typedef const char *(*ort_GetErrorMessage_fn)(const OrtStatus *);
typedef void (*ort_Release_fn)(void *);

/* API wrapper — holds the raw vtable and typed accessors */
typedef struct OrtApi {
    void **vt;  /* raw function pointer array from GetApi() */
} OrtApi;

/* Inline accessors — cast vtable slot to typed function pointer */
#define ORT_FN(api, slot, type) ((type)((api)->vt[slot]))

typedef struct OrtApiBase {
    void *(*GetApi)(uint32_t);
    const char *(*GetVersionString)(void);
} OrtApiBase;

typedef const OrtApiBase *(*OrtGetApiBase_fn)(void);

/* Default receptive field — used when model has no metadata.
 * Compact model: 63, Large model: 2047.
 * Over-estimating wastes compute but is safe; under-estimating loses context. */
#define DEFAULT_RECEPTIVE_FIELD 63

/* ORT metadata API slots (counted from ReleaseSessionOptions=100) */
#define ORT_SLOT_SESSION_GET_MODEL_METADATA    111
#define ORT_SLOT_MODEL_METADATA_LOOKUP_KEY     116
#define ORT_SLOT_RELEASE_MODEL_METADATA        118

typedef OrtStatus *(*ort_SessionGetModelMetadata_fn)(OrtSession *, void **);
typedef OrtStatus *(*ort_ModelMetadataLookup_fn)(void *, OrtAllocator *, const char *, char **);
typedef void (*ort_ReleaseModelMetadata_fn)(void *);

/* ORT API version we target */
#define ORT_API_VERSION_TARGET 18

/* Forward declaration for D3D12+DML resources (defined below onnx_filter) */
typedef struct dml_resources dml_resources_t;

struct onnx_filter {
    OrtApi          api;           /* vtable wrapper */
    OrtEnv         *env;
    OrtSession     *session;
    OrtMemoryInfo  *mem_info;
    float          *history;       /* left context buffer [hist_len] */
    float          *delay_buf;     /* non-causal: pending output samples [look_ahead] */
    float          *infer_buf;     /* model input: [hist + block + future] */
    float          *out_buf;       /* unused (output from ORT tensor directly) */
    size_t          hist_len;      /* left context: RF-1 (causal) or RF//2 (noncausal) */
    size_t          look_ahead;    /* 0 for causal, RF//2 for non-causal */
    size_t          block_alloc;   /* allocated block capacity */
    bool            noncausal;     /* true if model uses symmetric context */
    bool            primed;        /* non-causal: have we output delayed samples yet? */
    bool            preemph;       /* true if model is pre-SDM full pipeline */
    char           *input_name;    /* model input tensor name */
    char           *output_name;   /* model output tensor name */
    const char     *ep_name;       /* "CUDA", "DirectML", or "CPU" */
    HMODULE         hort;          /* onnxruntime.dll handle */
    dml_resources_t *dml_res;      /* D3D12+DML resources (Ex API only) */
};

/* ─── Resolve onnxruntime.dll path from component folder ───
 * Uses the same directory as foo_dsd_trellis.dll to avoid picking up
 * a wrong system-wide onnxruntime.dll (e.g. C:\Windows\System32). */

/* Resolve the component directory (same folder as DLL or worker exe). */
static bool resolve_component_dir(wchar_t *dir, size_t dir_size) {
    HMODULE hmod = GetModuleHandleW(L"foo_dsd_trellis.dll");
    if (!hmod)
        hmod = GetModuleHandleW(NULL);  /* Worker exe fallback */
    if (!hmod)
        return false;
    DWORD len = GetModuleFileNameW(hmod, dir, (DWORD)dir_size);
    if (len == 0 || len >= dir_size)
        return false;
    wchar_t *sep = wcsrchr(dir, L'\\');
    if (!sep) sep = wcsrchr(dir, L'/');
    if (sep)
        sep[1] = L'\0';
    else
        dir[0] = L'\0';
    return true;
}

/* Try loading an ORT DLL by name from the component directory.
 * Uses LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR so the DLL's directory is
 * in the search path for its dependencies (e.g., onnxruntime_providers_cuda.dll). */
static HMODULE load_ort_by_name(const wchar_t *filename) {
    wchar_t path[MAX_PATH];
    if (resolve_component_dir(path, MAX_PATH)) {
        /* Add component dir to DLL search path for provider dependencies */
        wchar_t dir[MAX_PATH];
        wcscpy_s(dir, MAX_PATH, path);
        SetDllDirectoryW(dir);

        wcscat_s(path, MAX_PATH, filename);
        HMODULE h = LoadLibraryW(path);
        if (h) return h;
    }
    return LoadLibraryW(filename);  /* fallback: system search */
}

/* Load ORT DLL. Tries CUDA build first (onnxruntime_cuda.dll),
 * then DirectML/default build (onnxruntime.dll).
 * The CUDA build has CUDA EP, the DML build has DirectML EP.
 * Both have CPU EP as fallback. */
static HMODULE load_ort_dll(void) {
    HMODULE h = load_ort_by_name(L"onnxruntime_cuda.dll");
    if (h) return h;
    return load_ort_by_name(L"onnxruntime.dll");
}

/* ─── DLL availability probe ─── */

static volatile long g_ort_checked = 0;
static volatile long g_ort_available = 0;

bool onnx_runtime_available(void) {
    if (InterlockedCompareExchange(&g_ort_checked, 0, 0))
        return InterlockedCompareExchange(&g_ort_available, 0, 0) != 0;

    HMODULE h = load_ort_dll();
    if (h) {
        /* Verify it's a real ORT build (has OrtGetApiBase) */
        OrtGetApiBase_fn get_base =
            (OrtGetApiBase_fn)GetProcAddress(h, "OrtGetApiBase");
        if (get_base)
            InterlockedExchange(&g_ort_available, 1);
        FreeLibrary(h);
    }
    InterlockedExchange(&g_ort_checked, 1);
    return InterlockedCompareExchange(&g_ort_available, 0, 0) != 0;
}

/* ─── Helper: check ORT status and release ─── */

static bool ort_ok(const OrtApi *api, OrtStatus *status) {
    if (status == NULL)
        return true;
    ORT_FN(api, ORT_SLOT_RELEASE_STATUS, ort_Release_fn)(status);
    return false;
}

/* Like ort_ok but saves the last error message for logging */
static char g_ort_last_error[256] = {0};

static bool ort_ok_log(const OrtApi *api, OrtStatus *status, const char *context) {
    if (status == NULL)
        return true;  /* don't clear previous error on success */
    const char *msg = ORT_FN(api, ORT_SLOT_GET_ERROR_MESSAGE, ort_GetErrorMessage_fn)(status);
    if (msg)
        snprintf(g_ort_last_error, sizeof(g_ort_last_error), "%s: %.200s", context, msg);
    else
        snprintf(g_ort_last_error, sizeof(g_ort_last_error), "%s: unknown error", context);
    OutputDebugStringA(g_ort_last_error);
    OutputDebugStringA("\n");
    ORT_FN(api, ORT_SLOT_RELEASE_STATUS, ort_Release_fn)(status);
    return false;
}

const char *onnx_filter_last_error(void) {
    return g_ort_last_error[0] ? g_ort_last_error : NULL;
}

/* ─── D3D12 + DirectML device creation for Ex DML EP ─── */

#include <initguid.h>

/* D3D12 COM IIDs */
DEFINE_GUID(IID_ID3D12Device_onnx,      0x189819f1,0x1db6,0x4b57,0xbe,0x54,0x18,0x21,0x33,0x9b,0x85,0xf7);
DEFINE_GUID(IID_ID3D12CommandQueue_onnx, 0x0ec870a6,0x5d7e,0x4c22,0x8c,0xfc,0x5b,0xaa,0xe0,0x76,0x16,0xed);
DEFINE_GUID(IID_IDXGIFactory4_onnx,     0x1bc6ea02,0xef36,0x464f,0xbf,0x0c,0x21,0xca,0x39,0xe5,0x16,0x8a);
DEFINE_GUID(IID_IDMLDevice_onnx,         0x6dbd6437,0x96fd,0x423f,0xa8,0xc0,0x45,0x26,0x35,0x70,0x10,0x5d);

/* Minimal COM vtable offsets for ID3D12Device, IDXGIFactory4, IDMLDevice */
#define IUNKNOWN_RELEASE 2
#define IDXGIFACTORY_ENUMADAPTERS1 12  /* IDXGIFactory1::EnumAdapters1 */
#define ID3D12DEVICE_CREATECOMMANDQUEUE 8  /* ID3D12Device::CreateCommandQueue (slot) */

typedef HRESULT (WINAPI *CreateDXGIFactory2_fn)(UINT, REFIID, void **);
typedef HRESULT (WINAPI *D3D12CreateDevice_fn)(void *, int, REFIID, void **);
typedef HRESULT (WINAPI *DMLCreateDevice_fn)(void *, int, REFIID, void **);

/* Opaque handles for the DML device resources we create.
 * Stored so we can release them when the ONNX filter is freed. */
struct dml_resources {
    void *dxgi_factory;   /* IDXGIFactory4* */
    void *adapter;        /* IDXGIAdapter1* */
    void *d3d12_device;   /* ID3D12Device* */
    void *cmd_queue;      /* ID3D12CommandQueue* */
    void *dml_device;     /* IDMLDevice* */
    HMODULE h_d3d12;
    HMODULE h_dxgi;
    HMODULE h_dml;
};

static void dml_resources_free(dml_resources_t *r) {
    if (!r) return;
    /* Release in reverse creation order */
    if (r->dml_device)    ((HRESULT (WINAPI **)(void *))*(void **)r->dml_device)[IUNKNOWN_RELEASE](r->dml_device);
    if (r->cmd_queue)     ((HRESULT (WINAPI **)(void *))*(void **)r->cmd_queue)[IUNKNOWN_RELEASE](r->cmd_queue);
    if (r->d3d12_device)  ((HRESULT (WINAPI **)(void *))*(void **)r->d3d12_device)[IUNKNOWN_RELEASE](r->d3d12_device);
    if (r->adapter)       ((HRESULT (WINAPI **)(void *))*(void **)r->adapter)[IUNKNOWN_RELEASE](r->adapter);
    if (r->dxgi_factory)  ((HRESULT (WINAPI **)(void *))*(void **)r->dxgi_factory)[IUNKNOWN_RELEASE](r->dxgi_factory);
    /* Don't FreeLibrary — DLLs may still be in use by ORT session */
}

/* Create D3D12 device + command queue + DML device for the ORT Ex DML EP.
 * Returns true on success; resources must be freed with dml_resources_free. */
static bool dml_resources_create(dml_resources_t *r) {
    memset(r, 0, sizeof(*r));

    r->h_dxgi  = LoadLibraryW(L"dxgi.dll");
    r->h_d3d12 = LoadLibraryW(L"d3d12.dll");
    r->h_dml   = LoadLibraryW(L"DirectML.dll");
    if (!r->h_dxgi || !r->h_d3d12 || !r->h_dml)
        goto fail;

    CreateDXGIFactory2_fn create_factory = (CreateDXGIFactory2_fn)
        GetProcAddress(r->h_dxgi, "CreateDXGIFactory2");
    D3D12CreateDevice_fn create_device = (D3D12CreateDevice_fn)
        GetProcAddress(r->h_d3d12, "D3D12CreateDevice");
    DMLCreateDevice_fn create_dml = (DMLCreateDevice_fn)
        GetProcAddress(r->h_dml, "DMLCreateDevice");
    if (!create_factory || !create_device || !create_dml)
        goto fail;

    /* 1. DXGI factory */
    if (FAILED(create_factory(0, &IID_IDXGIFactory4_onnx, &r->dxgi_factory)))
        goto fail;

    /* 2. Enumerate adapter 0 */
    {
        typedef HRESULT (WINAPI *EnumAdapters1_fn)(void *, UINT, void **);
        EnumAdapters1_fn enum_fn = ((EnumAdapters1_fn *)*(void **)r->dxgi_factory)[IDXGIFACTORY_ENUMADAPTERS1];
        if (FAILED(enum_fn(r->dxgi_factory, 0, &r->adapter)))
            goto fail;
    }

    /* 3. D3D12 device from adapter */
    if (FAILED(create_device(r->adapter, 0xb000 /*D3D_FEATURE_LEVEL_11_0*/,
                              &IID_ID3D12Device_onnx, &r->d3d12_device)))
        goto fail;

    /* 4. D3D12 command queue (direct, normal priority) */
    {
        /* D3D12_COMMAND_QUEUE_DESC: Type=DIRECT(0), Priority=NORMAL(0), Flags=0, NodeMask=0 */
        struct { int Type; int Priority; int Flags; unsigned NodeMask; } desc = {0, 0, 0, 0};
        typedef HRESULT (WINAPI *CreateCQ_fn)(void *, const void *, REFIID, void **);
        /* CreateCommandQueue is at vtable slot 8 in ID3D12Device */
        CreateCQ_fn cq_fn = ((CreateCQ_fn *)*(void **)r->d3d12_device)[ID3D12DEVICE_CREATECOMMANDQUEUE];
        if (FAILED(cq_fn(r->d3d12_device, &desc, &IID_ID3D12CommandQueue_onnx, &r->cmd_queue)))
            goto fail;
    }

    /* 5. DML device (no flags) */
    if (FAILED(create_dml(r->d3d12_device, 0 /*DML_CREATE_DEVICE_FLAG_NONE*/,
                           &IID_IDMLDevice_onnx, &r->dml_device)))
        goto fail;

    return true;

fail:
    dml_resources_free(r);
    memset(r, 0, sizeof(*r));
    return false;
}

/* ─── Create ─── */

onnx_filter_t *onnx_filter_create(const wchar_t *model_path,
                                   uint32_t dsd_rate, ml_ep_t ep) {
    (void)dsd_rate;

    if (!onnx_runtime_available())
        return NULL;

    /* Load DLL from component folder and resolve OrtGetApiBase */
    HMODULE hort = load_ort_dll();
    if (!hort)
        return NULL;

    OrtGetApiBase_fn get_api_base =
        (OrtGetApiBase_fn)GetProcAddress(hort, "OrtGetApiBase");
    if (!get_api_base) {
        FreeLibrary(hort);
        return NULL;
    }

    const OrtApiBase *base = get_api_base();
    void *api_ptr = base->GetApi(ORT_API_VERSION_TARGET);
    if (!api_ptr) {
        FreeLibrary(hort);
        return NULL;
    }

    /* Resolve GPU EP exports (may not exist in CPU-only ORT builds).
     * ML_EP_AUTO tries CUDA → DirectML → CPU. */
    OrtDmlAppendEP_fn dml_append = NULL;
    OrtCudaAppendEP_fn cuda_append = NULL;

    if (ep == ML_EP_CUDA || ep == ML_EP_AUTO) {
        cuda_append = (OrtCudaAppendEP_fn)GetProcAddress(
            hort, "OrtSessionOptionsAppendExecutionProvider_CUDA");
    }
    OrtDmlAppendEPEx_fn dml_append_ex = NULL;
    if (ep == ML_EP_DIRECTML || ep == ML_EP_AUTO) {
        dml_append_ex = (OrtDmlAppendEPEx_fn)GetProcAddress(
            hort, "OrtSessionOptionsAppendExecutionProviderEx_DML");
        dml_append = (OrtDmlAppendEP_fn)GetProcAddress(
            hort, "OrtSessionOptionsAppendExecutionProvider_DML");
    }

    onnx_filter_t *f = (onnx_filter_t *)calloc(1, sizeof(onnx_filter_t));
    if (!f) {
        FreeLibrary(hort);
        return NULL;
    }
    f->api.vt = (void **)api_ptr;
    f->hort = hort;
    f->ep_name = "CPU";

    const OrtApi *api = &f->api;

    /* Create environment */
    if (!ort_ok(api, ORT_FN(api, ORT_SLOT_CREATE_ENV, ort_CreateEnv_fn)(
            ORT_LOGGING_LEVEL_WARNING, "foo_dsd_trellis", &f->env)))
        goto fail;

    /* Session options */
    OrtSessionOptions *opts = NULL;
    if (!ort_ok(api, ORT_FN(api, ORT_SLOT_CREATE_SESSION_OPTIONS,
            ort_CreateSessionOptions_fn)(&opts)))
        goto fail;

    ORT_FN(api, ORT_SLOT_SET_INTRA_OP_THREADS, ort_SetThreads_fn)(opts, 1);
    ORT_FN(api, ORT_SLOT_SET_INTER_OP_THREADS, ort_SetThreads_fn)(opts, 1);
    ORT_FN(api, ORT_SLOT_SET_GRAPH_OPT_LEVEL, ort_SetGraphOptLevel_fn)(
            opts, ORT_ENABLE_ALL);

    /* GPU execution provider fallback chain: CUDA → DirectML → CPU.
     * CUDA EP: pass NULL for default options (device 0, default streams).
     * DirectML EP: requires sequential mode + disabled mem pattern. */
    {
        bool gpu_ok = false;

        /* Try CUDA first */
        if (cuda_append && !gpu_ok) {
            OrtStatus *cs = cuda_append(opts, NULL);  /* NULL = default CUDA options */
            if (ort_ok_log(api, cs, "CUDA EP")) {
                f->ep_name = "CUDA";
                gpu_ok = true;
            }
        }

        /* Try DirectML — Ex API first (explicit device), then basic (device ID) */
        if ((dml_append_ex || dml_append) && !gpu_ok) {
            ORT_FN(api, ORT_SLOT_DISABLE_MEM_PATTERN, ort_DisableMemPattern_fn)(opts);
            ORT_FN(api, ORT_SLOT_SET_SESSION_EXECUTION_MODE,
                   ort_SetSessionExecutionMode_fn)(opts, ORT_SEQUENTIAL);

            /* Ex API: create our own D3D12+DML device (more reliable) */
            if (dml_append_ex && !gpu_ok) {
                dml_resources_t *dr = (dml_resources_t *)calloc(1, sizeof(dml_resources_t));
                if (dr && dml_resources_create(dr)) {
                    if (ort_ok_log(api, dml_append_ex(opts, dr->dml_device,
                                                       dr->cmd_queue), "DirectML EP (Ex)")) {
                        f->ep_name = "DirectML";
                        f->dml_res = dr;
                        gpu_ok = true;
                    } else {
                        dml_resources_free(dr);
                        free(dr);
                    }
                } else {
                    free(dr);
                }
            }

            /* Basic API fallback: let ORT create the device (device ID 0) */
            if (dml_append && !gpu_ok) {
                if (ort_ok_log(api, dml_append(opts, 0), "DirectML EP")) {
                    f->ep_name = "DirectML";
                    gpu_ok = true;
                }
            }
        }
        /* CPU is always the final fallback — no action needed */
    }

    /* Create session from model file */
    if (!ort_ok_log(api, ORT_FN(api, ORT_SLOT_CREATE_SESSION, ort_CreateSession_fn)(
            f->env, model_path, opts, &f->session), "CreateSession")) {
        ORT_FN(api, ORT_SLOT_RELEASE_SESSION_OPTIONS, ort_Release_fn)(opts);
        goto fail;
    }
    ORT_FN(api, ORT_SLOT_RELEASE_SESSION_OPTIONS, ort_Release_fn)(opts);

    /* Memory info for CPU tensors */
    if (!ort_ok(api, ORT_FN(api, ORT_SLOT_CREATE_CPU_MEMORY_INFO,
            ort_CreateCpuMemoryInfo_fn)(
            ORT_ARENA_ALLOCATOR, ORT_MEM_TYPE_DEFAULT, &f->mem_info)))
        goto fail;

    /* Get input/output names */
    OrtAllocator *allocator = NULL;
    if (!ort_ok(api, ORT_FN(api, ORT_SLOT_GET_ALLOCATOR_DEFAULT,
            ort_GetAllocatorDefault_fn)(&allocator)))
        goto fail;

    char *in_name = NULL, *out_name = NULL;
    if (!ort_ok(api, ORT_FN(api, ORT_SLOT_SESSION_GET_INPUT_NAME,
            ort_SessionGetName_fn)(f->session, 0, allocator, &in_name)))
        goto fail;
    if (!ort_ok(api, ORT_FN(api, ORT_SLOT_SESSION_GET_OUTPUT_NAME,
            ort_SessionGetName_fn)(f->session, 0, allocator, &out_name))) {
        ORT_FN(api, ORT_SLOT_ALLOCATOR_FREE, ort_AllocatorFree_fn)(
                allocator, in_name);
        goto fail;
    }

    /* Copy names to our own allocation */
    {
        size_t in_len = strlen(in_name) + 1;
        size_t out_len = strlen(out_name) + 1;
        f->input_name = (char *)malloc(in_len);
        f->output_name = (char *)malloc(out_len);
        if (!f->input_name || !f->output_name) {
            ORT_FN(api, ORT_SLOT_ALLOCATOR_FREE, ort_AllocatorFree_fn)(
                    allocator, in_name);
            ORT_FN(api, ORT_SLOT_ALLOCATOR_FREE, ort_AllocatorFree_fn)(
                    allocator, out_name);
            goto fail;
        }
        memcpy(f->input_name, in_name, in_len);
        memcpy(f->output_name, out_name, out_len);
        ORT_FN(api, ORT_SLOT_ALLOCATOR_FREE, ort_AllocatorFree_fn)(
                allocator, in_name);
        ORT_FN(api, ORT_SLOT_ALLOCATOR_FREE, ort_AllocatorFree_fn)(
                allocator, out_name);
    }

    /* Read model metadata: model_type, receptive_field, noncausal, look_ahead */
    size_t rf = DEFAULT_RECEPTIVE_FIELD;
    bool model_noncausal = false;
    bool model_preemph = false;
    size_t model_look_ahead = 0;
    {
        void *metadata = NULL;
        OrtStatus *ms = ORT_FN(api, ORT_SLOT_SESSION_GET_MODEL_METADATA,
                ort_SessionGetModelMetadata_fn)(f->session, &metadata);
        if (ort_ok(api, ms) && metadata) {
            /* model_type: "preemph_full_pipeline" = pre-SDM mode */
            char *mt_str = NULL;
            OrtStatus *ls = ORT_FN(api, ORT_SLOT_MODEL_METADATA_LOOKUP_KEY,
                    ort_ModelMetadataLookup_fn)(
                    metadata, allocator, "model_type", &mt_str);
            if (ort_ok(api, ls) && mt_str) {
                if (strcmp(mt_str, "preemph_full_pipeline") == 0 ||
                    strcmp(mt_str, "preemph_taps") == 0)
                    model_preemph = true;
                ORT_FN(api, ORT_SLOT_ALLOCATOR_FREE, ort_AllocatorFree_fn)(
                        allocator, mt_str);
            }
            /* receptive_field */
            char *rf_str = NULL;
            ls = ORT_FN(api, ORT_SLOT_MODEL_METADATA_LOOKUP_KEY,
                    ort_ModelMetadataLookup_fn)(
                    metadata, allocator, "receptive_field", &rf_str);
            if (ort_ok(api, ls) && rf_str) {
                int val = atoi(rf_str);
                if (val > 0 && val <= 16384)
                    rf = (size_t)val;
                ORT_FN(api, ORT_SLOT_ALLOCATOR_FREE, ort_AllocatorFree_fn)(
                        allocator, rf_str);
            }
            /* noncausal flag */
            char *nc_str = NULL;
            ls = ORT_FN(api, ORT_SLOT_MODEL_METADATA_LOOKUP_KEY,
                    ort_ModelMetadataLookup_fn)(
                    metadata, allocator, "noncausal", &nc_str);
            if (ort_ok(api, ls) && nc_str) {
                model_noncausal = (atoi(nc_str) != 0);
                ORT_FN(api, ORT_SLOT_ALLOCATOR_FREE, ort_AllocatorFree_fn)(
                        allocator, nc_str);
            }
            /* look_ahead */
            char *la_str = NULL;
            ls = ORT_FN(api, ORT_SLOT_MODEL_METADATA_LOOKUP_KEY,
                    ort_ModelMetadataLookup_fn)(
                    metadata, allocator, "look_ahead", &la_str);
            if (ort_ok(api, ls) && la_str) {
                int val = atoi(la_str);
                if (val > 0 && val <= 16384)
                    model_look_ahead = (size_t)val;
                ORT_FN(api, ORT_SLOT_ALLOCATOR_FREE, ort_AllocatorFree_fn)(
                        allocator, la_str);
            }
            ORT_FN(api, ORT_SLOT_RELEASE_MODEL_METADATA, ort_ReleaseModelMetadata_fn)(
                    metadata);
        }
    }

    f->preemph = model_preemph;
    f->noncausal = model_noncausal;

    if (model_preemph) {
        /* Pre-SDM full pipeline: no history needed — stateless model.
         * Input is (1,1,N), output is (1,1,N). */
        f->hist_len = 0;
        f->look_ahead = 0;
    } else if (model_noncausal) {
        f->look_ahead = model_look_ahead > 0 ? model_look_ahead : rf / 2;
        f->hist_len = rf - 1 - f->look_ahead;
    } else {
        f->hist_len = rf - 1;
        f->look_ahead = 0;
    }

    /* Context buffers (initialized to zero = silence) */
    if (f->hist_len > 0) {
        f->history = (float *)calloc(f->hist_len, sizeof(float));
        if (!f->history)
            goto fail;
    }
    if (f->look_ahead > 0) {
        f->delay_buf = (float *)calloc(f->look_ahead, sizeof(float));
        if (!f->delay_buf)
            goto fail;
    }
    f->primed = false;

    /* Pre-allocate inference buffers */
    f->block_alloc = model_preemph ? 262144 : 8192;  /* preemph: 256K for DSD512 chunks */
    {
        size_t max_input = f->hist_len + f->block_alloc + f->look_ahead;
        f->infer_buf = (float *)malloc(max_input * sizeof(float));
        f->out_buf = (float *)malloc(max_input * sizeof(float));
    }
    if (!f->infer_buf || !f->out_buf)
        goto fail;

    return f;

fail:
    onnx_filter_free(f);
    return NULL;
}

/* ─── Process ─── */

/* Run inference on the infer_buf and write requantized output to dst.
 * Returns number of output samples written. */
static size_t run_inference(onnx_filter_t *f, size_t input_len,
                             float *dst, size_t dst_offset, size_t dst_count) {
    const OrtApi *api = &f->api;

    int64_t input_shape[3] = { 1, 1, (int64_t)input_len };
    OrtValue *input_tensor = NULL;
    if (!ort_ok(api, ORT_FN(api, ORT_SLOT_CREATE_TENSOR_WITH_DATA,
            ort_CreateTensorWithData_fn)(
            f->mem_info, f->infer_buf,
            input_len * sizeof(float),
            input_shape, 3, ONNX_TENSOR_ELEMENT_FLOAT,
            &input_tensor)))
        return 0;

    const char *input_names[] = { f->input_name };
    const char *output_names[] = { f->output_name };
    OrtValue *output_tensor = NULL;

    OrtStatus *run_status = ORT_FN(api, ORT_SLOT_RUN, ort_Run_fn)(
            f->session, NULL,
            input_names,
            (const OrtValue *const *)&input_tensor, 1,
            output_names, 1, &output_tensor);
    ORT_FN(api, ORT_SLOT_RELEASE_VALUE, ort_Release_fn)(input_tensor);

    if (!ort_ok(api, run_status))
        return 0;

    float *out_data = NULL;
    size_t written = 0;
    if (ort_ok(api, ORT_FN(api, ORT_SLOT_GET_TENSOR_MUTABLE_DATA,
            ort_GetTensorMutableData_fn)(output_tensor, (void **)&out_data))) {
        float *src = out_data + dst_offset;
        for (size_t i = 0; i < dst_count; i++)
            dst[i] = src[i] >= 0.0f ? 1.0f : -1.0f;
        written = dst_count;
    }

    ORT_FN(api, ORT_SLOT_RELEASE_VALUE, ort_Release_fn)(output_tensor);
    return written;
}

void onnx_filter_process(onnx_filter_t *f, float *buf, size_t count) {
    if (!f || !f->session || count == 0)
        return;

    size_t input_len;

    /* Grow inference buffers if needed */
    if (count > f->block_alloc) {
        size_t max_input = f->hist_len + count + f->look_ahead;
        float *new_infer = (float *)realloc(f->infer_buf,
                                             max_input * sizeof(float));
        float *new_out = (float *)realloc(f->out_buf,
                                           max_input * sizeof(float));
        if (!new_infer || !new_out)
            return;
        f->infer_buf = new_infer;
        f->out_buf = new_out;
        f->block_alloc = count;
    }

    if (!f->noncausal) {
        /* ─── Causal path ─── */
        /* Input: [history | current block] */
        memcpy(f->infer_buf, f->history, f->hist_len * sizeof(float));
        memcpy(f->infer_buf + f->hist_len, buf, count * sizeof(float));
        input_len = f->hist_len + count;

        /* Update history */
        if (count >= f->hist_len) {
            memcpy(f->history, buf + count - f->hist_len,
                   f->hist_len * sizeof(float));
        } else {
            size_t keep = f->hist_len - count;
            memmove(f->history, f->history + count, keep * sizeof(float));
            memcpy(f->history + keep, buf, count * sizeof(float));
        }

        /* Inference: output[hist_len .. hist_len+count] is our result */
        run_inference(f, input_len, buf, f->hist_len, count);
    } else {
        /* ─── Non-causal path ───
         * Model needs [left_ctx | block | right_ctx] to produce good
         * output for the block region. We process the block that was
         * received look_ahead samples ago (delay_buf holds pending output).
         *
         * Strategy: the model input is [history | delay_buf | buf].
         * hist_len samples of left context + look_ahead delayed samples
         * + count new samples = total input. We run inference and extract
         * the output corresponding to the delay_buf region (the samples
         * that now have full right context from buf). Then rotate:
         * delay_buf ← buf's tail, history ← overlap from buf. */
        size_t la = f->look_ahead;

        /* Build input: [history(hist_len) | delay_buf(la) | buf(count)] */
        memcpy(f->infer_buf, f->history, f->hist_len * sizeof(float));
        memcpy(f->infer_buf + f->hist_len, f->delay_buf, la * sizeof(float));
        memcpy(f->infer_buf + f->hist_len + la, buf, count * sizeof(float));
        input_len = f->hist_len + la + count;

        if (!f->primed) {
            /* First call: no valid delayed output yet. Process and save
             * delayed result; output passthrough (unmodified buf). */
            f->primed = true;
        } else {
            /* We have valid context: extract the look_ahead region's output.
             * The model output at offset hist_len corresponds to delay_buf,
             * which now has right context from buf. */
            size_t out_count = la < count ? la : count;
            /* We want to output the processed delay_buf samples.
             * But caller expects `count` samples in buf. We can only
             * provide min(la, count) processed samples from the delay. */
            if (count <= la) {
                /* Output count samples from the delayed region */
                run_inference(f, input_len, buf, f->hist_len, count);
            } else {
                /* Output all la delayed samples + (count-la) from current */
                run_inference(f, input_len, buf, f->hist_len, count);
            }
        }

        /* Update history: last hist_len samples before the new tail */
        {
            /* The full stream so far: [...history, delay_buf, buf...]
             * New delay_buf = last la samples of buf
             * New history = the hist_len samples just before new delay_buf */
            float *stream = f->infer_buf;  /* already has the full sequence */
            size_t total = input_len;  /* hist_len + la + count */

            /* New delay_buf = last la samples = buf[count-la..count]
             * (or all of buf + some delay_buf if count < la) */
            if (count >= la) {
                memcpy(f->delay_buf, buf + count - la, la * sizeof(float));
            } else {
                size_t keep = la - count;
                memmove(f->delay_buf, f->delay_buf + count,
                        keep * sizeof(float));
                memcpy(f->delay_buf + keep, buf, count * sizeof(float));
            }

            /* New history = hist_len samples before delay_buf in stream.
             * That's stream[total - la - hist_len .. total - la] */
            size_t hist_start = total - la - f->hist_len;
            memcpy(f->history, stream + hist_start,
                   f->hist_len * sizeof(float));
        }
    }
}

/* ─── Reset ─── */

void onnx_filter_reset(onnx_filter_t *f) {
    if (!f)
        return;
    if (f->history)
        memset(f->history, 0, f->hist_len * sizeof(float));
    if (f->delay_buf)
        memset(f->delay_buf, 0, f->look_ahead * sizeof(float));
    f->primed = false;
}

/* ─── Free ─── */

void onnx_filter_free(onnx_filter_t *f) {
    if (!f)
        return;

    const OrtApi *api = &f->api;
    if (api->vt) {
        if (f->session)
            ORT_FN(api, ORT_SLOT_RELEASE_SESSION, ort_Release_fn)(f->session);
        if (f->mem_info)
            ORT_FN(api, ORT_SLOT_RELEASE_MEMORY_INFO, ort_Release_fn)(f->mem_info);
        if (f->env)
            ORT_FN(api, ORT_SLOT_RELEASE_ENV, ort_Release_fn)(f->env);
    }

    free(f->history);
    free(f->delay_buf);
    free(f->infer_buf);
    free(f->out_buf);
    free(f->input_name);
    free(f->output_name);

    /* Release D3D12+DML resources (Ex API only) after ORT session is freed */
    if (f->dml_res) {
        dml_resources_free(f->dml_res);
        free(f->dml_res);
    }

    if (f->hort)
        FreeLibrary(f->hort);

    free(f);
}

/* ─── Pre-SDM processing (full pipeline on GPU) ─── */

void onnx_filter_process_preemph(onnx_filter_t *f, double *buf, size_t count) {
    if (!f || !f->session || count == 0)
        return;

    /* Grow inference buffer if needed */
    if (count > f->block_alloc) {
        float *new_infer = (float *)realloc(f->infer_buf, count * sizeof(float));
        float *new_out = (float *)realloc(f->out_buf, count * sizeof(float));
        if (!new_infer || !new_out)
            return;
        f->infer_buf = new_infer;
        f->out_buf = new_out;
        f->block_alloc = count;
    }

    /* Convert double → float for ONNX tensor */
    for (size_t i = 0; i < count; i++)
        f->infer_buf[i] = (float)buf[i];

    /* Build input tensor: (1, 1, N) */
    const OrtApi *api = &f->api;
    int64_t input_shape[3] = { 1, 1, (int64_t)count };
    OrtValue *input_tensor = NULL;
    if (!ort_ok(api, ORT_FN(api, ORT_SLOT_CREATE_TENSOR_WITH_DATA,
            ort_CreateTensorWithData_fn)(
            f->mem_info, f->infer_buf,
            count * sizeof(float),
            input_shape, 3, ONNX_TENSOR_ELEMENT_FLOAT,
            &input_tensor)))
        return;

    const char *input_names[] = { f->input_name };
    const char *output_names[] = { f->output_name };
    OrtValue *output_tensor = NULL;

    OrtStatus *run_status = ORT_FN(api, ORT_SLOT_RUN, ort_Run_fn)(
            f->session, NULL,
            input_names,
            (const OrtValue *const *)&input_tensor, 1,
            output_names, 1, &output_tensor);
    ORT_FN(api, ORT_SLOT_RELEASE_VALUE, ort_Release_fn)(input_tensor);

    if (!ort_ok(api, run_status))
        return;

    /* Read output and convert float → double back to buf */
    float *out_data = NULL;
    if (ort_ok(api, ORT_FN(api, ORT_SLOT_GET_TENSOR_MUTABLE_DATA,
            ort_GetTensorMutableData_fn)(output_tensor, (void **)&out_data))) {
        for (size_t i = 0; i < count; i++)
            buf[i] = (double)out_data[i];
    }

    ORT_FN(api, ORT_SLOT_RELEASE_VALUE, ort_Release_fn)(output_tensor);
}

/* ─── Tap prediction (lightweight MLP on GPU) ─── */

void onnx_filter_predict_taps(onnx_filter_t *f, const float features[3],
                               float taps_out[3]) {
    if (!f || !f->session) {
        taps_out[0] = 1.0f; taps_out[1] = 0.0f; taps_out[2] = 0.0f;
        return;
    }

    const OrtApi *api = &f->api;
    float input_data[3] = { features[0], features[1], features[2] };
    int64_t input_shape[2] = { 1, 3 };
    OrtValue *input_tensor = NULL;

    if (!ort_ok(api, ORT_FN(api, ORT_SLOT_CREATE_TENSOR_WITH_DATA,
            ort_CreateTensorWithData_fn)(
            f->mem_info, input_data, sizeof(input_data),
            input_shape, 2, ONNX_TENSOR_ELEMENT_FLOAT,
            &input_tensor))) {
        taps_out[0] = 1.0f; taps_out[1] = 0.0f; taps_out[2] = 0.0f;
        return;
    }

    const char *input_names[] = { f->input_name };
    const char *output_names[] = { f->output_name };
    OrtValue *output_tensor = NULL;

    OrtStatus *run_status = ORT_FN(api, ORT_SLOT_RUN, ort_Run_fn)(
            f->session, NULL,
            input_names,
            (const OrtValue *const *)&input_tensor, 1,
            output_names, 1, &output_tensor);
    ORT_FN(api, ORT_SLOT_RELEASE_VALUE, ort_Release_fn)(input_tensor);

    if (!ort_ok(api, run_status)) {
        taps_out[0] = 1.0f; taps_out[1] = 0.0f; taps_out[2] = 0.0f;
        return;
    }

    float *out_data = NULL;
    if (ort_ok(api, ORT_FN(api, ORT_SLOT_GET_TENSOR_MUTABLE_DATA,
            ort_GetTensorMutableData_fn)(output_tensor, (void **)&out_data))) {
        taps_out[0] = out_data[0];
        taps_out[1] = out_data[1];
        taps_out[2] = out_data[2];
    } else {
        taps_out[0] = 1.0f; taps_out[1] = 0.0f; taps_out[2] = 0.0f;
    }

    ORT_FN(api, ORT_SLOT_RELEASE_VALUE, ort_Release_fn)(output_tensor);
}

/* ─── EP name query ─── */

const char *onnx_filter_ep_name(const onnx_filter_t *f) {
    return f ? f->ep_name : "none";
}
