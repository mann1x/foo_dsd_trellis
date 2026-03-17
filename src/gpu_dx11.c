/*
 * foo_dsd_trellis — DirectCompute (D3D11) GPU backend
 *
 * Delay-loads d3d11.dll and d3dcompiler_47.dll via LoadLibrary.
 * Creates a compute-only D3D11 device (no swap chain, no DXGI).
 * HLSL compute shaders compiled at runtime from embedded strings.
 *
 * Implements: FIR upsample/downsample, gain, boxcar smoothing.
 */

#include "../include/gpu_compute.h"

#define WIN32_LEAN_AND_MEAN
#define COBJMACROS          /* Enable C-style COM macros (ID3D11Device_Method) */
#define INITGUID            /* Define GUIDs in this translation unit */
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ─── Delay-loaded function pointers ─── */

typedef HRESULT (WINAPI *PFN_D3D11CreateDevice)(
    IDXGIAdapter *, D3D_DRIVER_TYPE, HMODULE, UINT,
    const D3D_FEATURE_LEVEL *, UINT, UINT,
    ID3D11Device **, D3D_FEATURE_LEVEL *, ID3D11DeviceContext **);

typedef HRESULT (WINAPI *PFN_D3DCompile)(
    LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO *, ID3DInclude *,
    LPCSTR, LPCSTR, UINT, UINT, ID3DBlob **, ID3DBlob **);

static HMODULE g_d3d11_dll = NULL;
static HMODULE g_d3dc_dll  = NULL;
static PFN_D3D11CreateDevice g_pfn_create_device = NULL;
static PFN_D3DCompile        g_pfn_compile       = NULL;
static bool g_probed = false;
static bool g_available = false;
static char g_device_name[128] = "";
static size_t g_vram_bytes = 0;

/* ─── Embedded HLSL shaders ─── */

static const char g_hlsl_fir_upsample[] =
"StructuredBuffer<float>   g_in  : register(t0);\n"
"RWStructuredBuffer<float> g_out : register(u0);\n"
"cbuffer Params : register(b0) {\n"
"    uint in_count;\n"
"    uint out_count;\n"
"    uint ntaps;\n"
"    uint pad0;\n"
"};\n"
"StructuredBuffer<float> g_taps : register(t1);\n"
"\n"
"[numthreads(256, 1, 1)]\n"
"void main(uint3 dtid : SV_DispatchThreadID) {\n"
"    uint oi = dtid.x;\n"
"    if (oi >= out_count) return;\n"
"    float acc = 0.0;\n"
"    for (int k = 0; k < (int)ntaps; k++) {\n"
"        int zsi = (int)oi - k;\n"
"        if (zsi >= 0 && zsi < (int)(in_count * 2)) {\n"
"            float v = (zsi & 1) ? 0.0 : g_in[zsi >> 1];\n"
"            acc += g_taps[k] * v;\n"
"        }\n"
"    }\n"
"    g_out[oi] = acc * 2.0;\n"
"}\n";

static const char g_hlsl_fir_downsample[] =
"StructuredBuffer<float>   g_in  : register(t0);\n"
"RWStructuredBuffer<float> g_out : register(u0);\n"
"cbuffer Params : register(b0) {\n"
"    uint in_count;\n"
"    uint out_count;\n"
"    uint ntaps;\n"
"    uint pad0;\n"
"};\n"
"StructuredBuffer<float> g_taps : register(t1);\n"
"\n"
"[numthreads(256, 1, 1)]\n"
"void main(uint3 dtid : SV_DispatchThreadID) {\n"
"    uint oi = dtid.x;\n"
"    if (oi >= out_count) return;\n"
"    uint ii = oi * 2;\n"
"    float acc = 0.0;\n"
"    for (int k = 0; k < (int)ntaps; k++) {\n"
"        int si = (int)ii - k;\n"
"        if (si >= 0 && si < (int)in_count)\n"
"            acc += g_taps[k] * g_in[si];\n"
"    }\n"
"    g_out[oi] = acc;\n"
"}\n";

static const char g_hlsl_gain[] =
"RWStructuredBuffer<float> g_buf : register(u0);\n"
"cbuffer Params : register(b0) {\n"
"    uint count;\n"
"    float gain_val;\n"
"    uint pad0;\n"
"    uint pad1;\n"
"};\n"
"\n"
"[numthreads(256, 1, 1)]\n"
"void main(uint3 dtid : SV_DispatchThreadID) {\n"
"    if (dtid.x < count)\n"
"        g_buf[dtid.x] *= gain_val;\n"
"}\n";

static const char g_hlsl_boxcar[] =
"StructuredBuffer<float>   g_in  : register(t0);\n"
"RWStructuredBuffer<float> g_out : register(u0);\n"
"cbuffer Params : register(b0) {\n"
"    uint count;\n"
"    uint box_taps;\n"
"    float gain_val;\n"
"    uint pad0;\n"
"};\n"
"\n"
"[numthreads(256, 1, 1)]\n"
"void main(uint3 dtid : SV_DispatchThreadID) {\n"
"    uint i = dtid.x;\n"
"    if (i >= count) return;\n"
"    float sum = 0.0;\n"
"    int bt = (int)box_taps;\n"
"    for (int k = 0; k < bt; k++) {\n"
"        int si = (int)i - k;\n"
"        float v = (si >= 0) ? g_in[si] : 0.0;\n"
"        sum += (v >= 0.0 ? 1.0 : -1.0);\n"
"    }\n"
"    g_out[i] = sum / (float)bt * gain_val;\n"
"}\n";

/* Trellis SDM chunk shader — sequential per-sample, parallel candidate expansion.
 * Thread 0..2*num_cands-1: NTF evaluation. Thread 0: sort+output.
 * Resources: t0=input, u0=output, u1=cand_states(double as uint2), u2=cand_costs */
static const char g_hlsl_trellis[] =
"StructuredBuffer<float>    g_in    : register(t0);\n"
"RWStructuredBuffer<float>  g_out   : register(u0);\n"
"RWStructuredBuffer<uint2>  g_states : register(u1);\n" /* double as uint2 */
"RWStructuredBuffer<uint2>  g_costs  : register(u2);\n"
"cbuffer TrellisParams : register(b0) {\n"
"    int count;\n"
"    int num_cands;\n"
"    int trellis_lat;\n"
"    int ntf_order;\n"
"    double ntf_a[8];\n"  /* 128 bytes */
"    double ntf_g[8];\n"  /* 128 bytes */
"    double state_limit;\n"
"};\n"
"\n"
"/* Double pack/unpack via uint2 (HLSL has no native double UAV) */\n"
"double load_d(uint2 v) { return asdouble(v.x, v.y); }\n"
"uint2 store_d(double v) { uint lo, hi; asuint(v, lo, hi); return uint2(lo, hi); }\n"
"\n"
"groupshared double s_state[64][8];\n"  /* max 32 cands × 2 children */
"groupshared double s_cost[64];\n"
"groupshared uint   s_path[64];\n"
"groupshared uint   s_output;\n"
"groupshared int    s_active;\n"
"\n"
"[numthreads(64, 1, 1)]\n"
"void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {\n"
"    int t = tid.x;\n"
"    int nc = num_cands;\n"
"    int order = ntf_order;\n"
"\n"
"    /* Load initial state from UAV */\n"
"    if (t < nc) {\n"
"        for (int k = 0; k < order; k++)\n"
"            s_state[t][k] = load_d(g_states[t * 8 + k]);\n"
"        s_cost[t] = load_d(g_costs[t]);\n"
"        s_path[t] = 0;\n"
"    }\n"
"    if (t == 0) s_active = nc;\n"
"    GroupMemoryBarrierWithGroupSync();\n"
"\n"
"    for (int s = 0; s < count; s++) {\n"
"        double x = (double)g_in[s];\n"
"        int ac = s_active;\n"
"\n"
"        /* Parallel NTF eval: tid < 2*ac */\n"
"        if (t < 2 * ac) {\n"
"            int pi = t / 2;\n"
"            double y_b = (t & 1) ? -1.0 : 1.0;\n"
"            double d[8];\n"
"            d[0] = s_state[pi][0] - ntf_g[0] * s_state[pi][1] + x;\n"
"            for (int k = 1; k < order - 1; k++)\n"
"                d[k] = s_state[pi][k] + s_state[pi][k-1] - ntf_g[k] * s_state[pi][k+1];\n"
"            d[order-1] = s_state[pi][order-1] + s_state[pi][order-2];\n"
"            double v = x;\n"
"            for (int k = 0; k < order; k++) v += ntf_a[k] * d[k];\n"
"            d[0] += y_b;\n"
"            if (state_limit > 0.0) {\n"
"                for (int k = 0; k < order; k++) {\n"
"                    if (d[k] > state_limit) d[k] = state_limit;\n"
"                    else if (d[k] < -state_limit) d[k] = -state_limit;\n"
"                }\n"
"            }\n"
"            int ci = nc + t;\n" /* children go after parents in shared mem */
"            for (int k = 0; k < order; k++) s_state[ci][k] = d[k];\n"
"            s_cost[ci] = s_cost[pi] + (v + ntf_a[0]*y_b)*(v + ntf_a[0]*y_b);\n"
"            s_path[ci] = (s_path[pi] << 1 | (uint)(t & 1)) & 0xFF;\n"
"        }\n"
"        GroupMemoryBarrierWithGroupSync();\n"
"\n"
"        /* Thread 0: selection sort top nc by cost */\n"
"        if (t == 0) {\n"
"            int total = 2 * ac;\n"
"            /* Simple sort in children range [nc..nc+total) */\n"
"            for (int i = 0; i < ac; i++) {\n"
"                int best = nc + i;\n"
"                for (int j = nc + i + 1; j < nc + total; j++) {\n"
"                    if (s_cost[j] < s_cost[best]) best = j;\n"
"                }\n"
"                if (best != nc + i) {\n"
"                    /* Swap */\n"
"                    double tc = s_cost[nc+i]; s_cost[nc+i] = s_cost[best]; s_cost[best] = tc;\n"
"                    uint tp = s_path[nc+i]; s_path[nc+i] = s_path[best]; s_path[best] = tp;\n"
"                    for (int k = 0; k < order; k++) {\n"
"                        double ts = s_state[nc+i][k]; s_state[nc+i][k] = s_state[best][k]; s_state[best][k] = ts;\n"
"                    }\n"
"                }\n"
"            }\n"
"            s_output = s_path[nc] & 1;\n"
"            double min_c = s_cost[nc];\n"
"            for (int i = 0; i < ac; i++) {\n"
"                s_cost[i] = s_cost[nc+i] - min_c;\n"
"                s_path[i] = s_path[nc+i];\n"
"                for (int k = 0; k < order; k++) s_state[i][k] = s_state[nc+i][k];\n"
"            }\n"
"        }\n"
"        GroupMemoryBarrierWithGroupSync();\n"
"\n"
"        if (t == 0 && s >= trellis_lat)\n"
"            g_out[s - trellis_lat] = s_output ? 1.0f : -1.0f;\n"
"    }\n"
"\n"
"    /* Save final state */\n"
"    if (t < nc) {\n"
"        for (int k = 0; k < order; k++)\n"
"            g_states[t * 8 + k] = store_d(s_state[t][k]);\n"
"        g_costs[t] = store_d(s_cost[t]);\n"
"    }\n"
"}\n";

/* ─── DX11 context structure ─── */

typedef struct {
    gpu_backend_t          backend;      /* Must be first — GPU_BACKEND_DIRECTX */
    ID3D11Device          *device;
    ID3D11DeviceContext   *ctx;
    /* Compute shaders */
    ID3D11ComputeShader   *cs_fir_up;
    ID3D11ComputeShader   *cs_fir_down;
    ID3D11ComputeShader   *cs_gain;
    ID3D11ComputeShader   *cs_boxcar;
    ID3D11ComputeShader   *cs_trellis;
    /* GPU buffers */
    ID3D11Buffer          *buf_in;
    ID3D11ShaderResourceView *srv_in;
    size_t                 cap_in;       /* floats */
    ID3D11Buffer          *buf_out;
    ID3D11UnorderedAccessView *uav_out;
    size_t                 cap_out;      /* floats */
    ID3D11Buffer          *buf_staging;
    size_t                 cap_staging;  /* floats */
    /* FIR taps buffer */
    ID3D11Buffer          *buf_taps;
    ID3D11ShaderResourceView *srv_taps;
    /* Constant buffer */
    ID3D11Buffer          *buf_params;
    /* FIR chain config */
    int                    num_stages;
    bool                   upsample;
    int                    ntaps;
    /* Intermediate buffer for multi-stage */
    ID3D11Buffer          *buf_inter;
    ID3D11ShaderResourceView *srv_inter;
    ID3D11UnorderedAccessView *uav_inter;
    size_t                 cap_inter;
} dx11_context_t;

/* ─── Helpers ─── */

/* Compile HLSL shader from string, return compute shader. */
static ID3D11ComputeShader *compile_cs(ID3D11Device *dev,
                                        const char *src, const char *name) {
    if (!g_pfn_compile) return NULL;

    ID3DBlob *blob = NULL, *err = NULL;
    HRESULT hr = g_pfn_compile(src, strlen(src), name, NULL, NULL,
                                "main", "cs_5_0", 0, 0, &blob, &err);
    if (FAILED(hr)) {
        if (err) {
            OutputDebugStringA((const char *)ID3D10Blob_GetBufferPointer(err));
            ID3D10Blob_Release(err);
        }
        return NULL;
    }
    if (err) ID3D10Blob_Release(err);

    ID3D11ComputeShader *cs = NULL;
    hr = ID3D11Device_CreateComputeShader(dev,
        ID3D10Blob_GetBufferPointer(blob),
        ID3D10Blob_GetBufferSize(blob), NULL, &cs);
    ID3D10Blob_Release(blob);

    return SUCCEEDED(hr) ? cs : NULL;
}

/* Create a structured buffer (GPU default, no CPU access). */
static HRESULT create_structured_buf(ID3D11Device *dev, UINT count,
                                      UINT bind_flags,
                                      ID3D11Buffer **buf) {
    D3D11_BUFFER_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.ByteWidth           = count * sizeof(float);
    desc.Usage               = D3D11_USAGE_DEFAULT;
    desc.BindFlags           = bind_flags;
    desc.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(float);
    return ID3D11Device_CreateBuffer(dev, &desc, NULL, buf);
}

/* Create a staging buffer for CPU readback. */
static HRESULT create_staging_buf(ID3D11Device *dev, UINT count,
                                   ID3D11Buffer **buf) {
    D3D11_BUFFER_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.ByteWidth      = count * sizeof(float);
    desc.Usage          = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    return ID3D11Device_CreateBuffer(dev, &desc, NULL, buf);
}

/* Create SRV for a structured buffer. */
static HRESULT create_srv(ID3D11Device *dev, ID3D11Buffer *buf, UINT count,
                           ID3D11ShaderResourceView **srv) {
    D3D11_SHADER_RESOURCE_VIEW_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.Format               = DXGI_FORMAT_UNKNOWN;
    desc.ViewDimension        = D3D11_SRV_DIMENSION_BUFFER;
    desc.Buffer.FirstElement  = 0;
    desc.Buffer.NumElements   = count;
    return ID3D11Device_CreateShaderResourceView(dev, (ID3D11Resource *)buf,
                                                  &desc, srv);
}

/* Create UAV for a structured buffer. */
static HRESULT create_uav(ID3D11Device *dev, ID3D11Buffer *buf, UINT count,
                           ID3D11UnorderedAccessView **uav) {
    D3D11_UNORDERED_ACCESS_VIEW_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.Format              = DXGI_FORMAT_UNKNOWN;
    desc.ViewDimension       = D3D11_UAV_DIMENSION_BUFFER;
    desc.Buffer.FirstElement = 0;
    desc.Buffer.NumElements  = count;
    return ID3D11Device_CreateUnorderedAccessView(dev, (ID3D11Resource *)buf,
                                                   &desc, uav);
}

/* Create constant buffer (16-byte aligned). */
static HRESULT create_cb(ID3D11Device *dev, UINT size, ID3D11Buffer **buf) {
    D3D11_BUFFER_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.ByteWidth      = (size + 15) & ~15;  /* 16-byte align */
    desc.Usage          = D3D11_USAGE_DEFAULT;
    desc.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    return ID3D11Device_CreateBuffer(dev, &desc, NULL, buf);
}

/* Ensure input buffer capacity (recreate if too small). */
static int ensure_buf_in(dx11_context_t *c, size_t floats) {
    if (c->cap_in >= floats) return 0;
    if (c->srv_in)  { ID3D11ShaderResourceView_Release(c->srv_in); c->srv_in = NULL; }
    if (c->buf_in)  { ID3D11Buffer_Release(c->buf_in); c->buf_in = NULL; }
    if (FAILED(create_structured_buf(c->device, (UINT)floats,
            D3D11_BIND_SHADER_RESOURCE, &c->buf_in)))
        return -1;
    if (FAILED(create_srv(c->device, c->buf_in, (UINT)floats, &c->srv_in)))
        return -1;
    c->cap_in = floats;
    return 0;
}

static int ensure_buf_out(dx11_context_t *c, size_t floats) {
    if (c->cap_out >= floats) return 0;
    if (c->uav_out) { ID3D11UnorderedAccessView_Release(c->uav_out); c->uav_out = NULL; }
    if (c->buf_out) { ID3D11Buffer_Release(c->buf_out); c->buf_out = NULL; }
    if (FAILED(create_structured_buf(c->device, (UINT)floats,
            D3D11_BIND_UNORDERED_ACCESS, &c->buf_out)))
        return -1;
    if (FAILED(create_uav(c->device, c->buf_out, (UINT)floats, &c->uav_out)))
        return -1;
    c->cap_out = floats;
    return 0;
}

static int ensure_staging(dx11_context_t *c, size_t floats) {
    if (c->cap_staging >= floats) return 0;
    if (c->buf_staging) { ID3D11Buffer_Release(c->buf_staging); c->buf_staging = NULL; }
    if (FAILED(create_staging_buf(c->device, (UINT)floats, &c->buf_staging)))
        return -1;
    c->cap_staging = floats;
    return 0;
}

static int ensure_buf_inter(dx11_context_t *c, size_t floats) {
    if (c->cap_inter >= floats) return 0;
    if (c->uav_inter) { ID3D11UnorderedAccessView_Release(c->uav_inter); c->uav_inter = NULL; }
    if (c->srv_inter) { ID3D11ShaderResourceView_Release(c->srv_inter); c->srv_inter = NULL; }
    if (c->buf_inter) { ID3D11Buffer_Release(c->buf_inter); c->buf_inter = NULL; }
    if (FAILED(create_structured_buf(c->device, (UINT)floats,
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
            &c->buf_inter)))
        return -1;
    if (FAILED(create_srv(c->device, c->buf_inter, (UINT)floats, &c->srv_inter)))
        return -1;
    if (FAILED(create_uav(c->device, c->buf_inter, (UINT)floats, &c->uav_inter)))
        return -1;
    c->cap_inter = floats;
    return 0;
}

/* Upload float data to a GPU buffer. */
static void upload_buf(ID3D11DeviceContext *ctx, ID3D11Buffer *buf,
                        const float *data, size_t floats) {
    D3D11_BOX box;
    box.left = 0;
    box.right = (UINT)(floats * sizeof(float));
    box.top = 0; box.bottom = 1;
    box.front = 0; box.back = 1;
    ID3D11DeviceContext_UpdateSubresource(ctx, (ID3D11Resource *)buf,
                                          0, &box, data, 0, 0);
}

/* Download float data from GPU via staging buffer. */
static int download_buf(ID3D11DeviceContext *ctx, ID3D11Buffer *gpu_buf,
                         ID3D11Buffer *staging, float *out, size_t floats) {
    D3D11_BOX box;
    box.left = 0;
    box.right = (UINT)(floats * sizeof(float));
    box.top = 0; box.bottom = 1;
    box.front = 0; box.back = 1;
    ID3D11DeviceContext_CopySubresourceRegion(ctx,
        (ID3D11Resource *)staging, 0, 0, 0, 0,
        (ID3D11Resource *)gpu_buf, 0, &box);

    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr = ID3D11DeviceContext_Map(ctx, (ID3D11Resource *)staging,
                                          0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) return -1;
    memcpy(out, mapped.pData, floats * sizeof(float));
    ID3D11DeviceContext_Unmap(ctx, (ID3D11Resource *)staging, 0);
    return 0;
}

/* ─── Probe ─── */

bool gpu_dx11_probe(void) {
    if (g_probed) return g_available;
    g_probed = true;

    g_d3d11_dll = LoadLibraryA("d3d11.dll");
    if (!g_d3d11_dll) { g_available = false; return false; }

    g_pfn_create_device = (PFN_D3D11CreateDevice)
        GetProcAddress(g_d3d11_dll, "D3D11CreateDevice");
    if (!g_pfn_create_device) {
        FreeLibrary(g_d3d11_dll); g_d3d11_dll = NULL;
        g_available = false; return false;
    }

    g_d3dc_dll = LoadLibraryA("d3dcompiler_47.dll");
    if (!g_d3dc_dll) {
        FreeLibrary(g_d3d11_dll); g_d3d11_dll = NULL;
        g_available = false; return false;
    }

    g_pfn_compile = (PFN_D3DCompile)
        GetProcAddress(g_d3dc_dll, "D3DCompile");
    if (!g_pfn_compile) {
        FreeLibrary(g_d3dc_dll); g_d3dc_dll = NULL;
        FreeLibrary(g_d3d11_dll); g_d3d11_dll = NULL;
        g_available = false; return false;
    }

    /* Try creating a device to verify GPU hardware */
    D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0 };
    D3D_FEATURE_LEVEL got;
    ID3D11Device *dev = NULL;
    ID3D11DeviceContext *ctx = NULL;
    HRESULT hr = g_pfn_create_device(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL,
        D3D11_CREATE_DEVICE_SINGLETHREADED, levels, 1,
        D3D11_SDK_VERSION, &dev, &got, &ctx);
    if (FAILED(hr) || !dev) {
        g_available = false; return false;
    }

    /* Query adapter info via DXGI */
    {
        IDXGIDevice *dxgi_dev = NULL;
        hr = ID3D11Device_QueryInterface(dev, &IID_IDXGIDevice, (void **)&dxgi_dev);
        if (SUCCEEDED(hr) && dxgi_dev) {
            IDXGIAdapter *adapter = NULL;
            hr = IDXGIDevice_GetAdapter(dxgi_dev, &adapter);
            if (SUCCEEDED(hr) && adapter) {
                DXGI_ADAPTER_DESC desc;
                if (SUCCEEDED(IDXGIAdapter_GetDesc(adapter, &desc))) {
                    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1,
                                        g_device_name, sizeof(g_device_name),
                                        NULL, NULL);
                    g_vram_bytes = desc.DedicatedVideoMemory;
                }
                IDXGIAdapter_Release(adapter);
            }
            IDXGIDevice_Release(dxgi_dev);
        }
    }

    ID3D11DeviceContext_Release(ctx);
    ID3D11Device_Release(dev);

    g_available = true;
    return true;
}

void gpu_dx11_get_info(gpu_info_t *info) {
    memset(info, 0, sizeof(*info));
    info->backend = GPU_BACKEND_DIRECTX;
    info->available = g_available;
    strncpy_s(info->device_name, sizeof(info->device_name), g_device_name, _TRUNCATE);
    info->vram_mb = g_vram_bytes / (1024 * 1024);
}

/* ─── Create / Destroy ─── */

gpu_context_t *gpu_dx11_create(void) {
    if (!g_available || !g_pfn_create_device || !g_pfn_compile)
        return NULL;

    D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0 };
    D3D_FEATURE_LEVEL got;
    ID3D11Device *dev = NULL;
    ID3D11DeviceContext *ctx = NULL;
    HRESULT hr = g_pfn_create_device(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL,
        D3D11_CREATE_DEVICE_SINGLETHREADED, levels, 1,
        D3D11_SDK_VERSION, &dev, &got, &ctx);
    if (FAILED(hr) || !dev)
        return NULL;

    dx11_context_t *c = (dx11_context_t *)calloc(1, sizeof(dx11_context_t));
    if (!c) {
        ID3D11DeviceContext_Release(ctx);
        ID3D11Device_Release(dev);
        return NULL;
    }

    c->backend = GPU_BACKEND_DIRECTX;
    c->device  = dev;
    c->ctx     = ctx;

    /* Compile compute shaders */
    c->cs_fir_up   = compile_cs(dev, g_hlsl_fir_upsample, "fir_up");
    c->cs_fir_down = compile_cs(dev, g_hlsl_fir_downsample, "fir_down");
    c->cs_gain     = compile_cs(dev, g_hlsl_gain, "gain");
    c->cs_boxcar   = compile_cs(dev, g_hlsl_boxcar, "boxcar");
    c->cs_trellis  = compile_cs(dev, g_hlsl_trellis, "trellis");

    if (!c->cs_fir_up || !c->cs_fir_down || !c->cs_gain || !c->cs_boxcar) {
        /* Trellis shader is optional — may fail on some GPUs */
        gpu_dx11_destroy(c);
        return NULL;
    }

    /* Create constant buffer (16 bytes = 4 uints) */
    if (FAILED(create_cb(dev, 16, &c->buf_params))) {
        gpu_dx11_destroy(c);
        return NULL;
    }

    return (gpu_context_t *)c;
}

void gpu_dx11_destroy(void *ptr) {
    dx11_context_t *c = (dx11_context_t *)ptr;
    if (!c) return;

    if (c->uav_inter) ID3D11UnorderedAccessView_Release(c->uav_inter);
    if (c->srv_inter) ID3D11ShaderResourceView_Release(c->srv_inter);
    if (c->buf_inter) ID3D11Buffer_Release(c->buf_inter);
    if (c->srv_taps)  ID3D11ShaderResourceView_Release(c->srv_taps);
    if (c->buf_taps)  ID3D11Buffer_Release(c->buf_taps);
    if (c->buf_params) ID3D11Buffer_Release(c->buf_params);
    if (c->buf_staging) ID3D11Buffer_Release(c->buf_staging);
    if (c->uav_out)   ID3D11UnorderedAccessView_Release(c->uav_out);
    if (c->buf_out)   ID3D11Buffer_Release(c->buf_out);
    if (c->srv_in)    ID3D11ShaderResourceView_Release(c->srv_in);
    if (c->buf_in)    ID3D11Buffer_Release(c->buf_in);
    if (c->cs_trellis) ID3D11ComputeShader_Release(c->cs_trellis);
    if (c->cs_boxcar) ID3D11ComputeShader_Release(c->cs_boxcar);
    if (c->cs_gain)   ID3D11ComputeShader_Release(c->cs_gain);
    if (c->cs_fir_down) ID3D11ComputeShader_Release(c->cs_fir_down);
    if (c->cs_fir_up) ID3D11ComputeShader_Release(c->cs_fir_up);
    if (c->ctx)       ID3D11DeviceContext_Release(c->ctx);
    if (c->device)    ID3D11Device_Release(c->device);
    free(c);
}

/* ─── FIR Setup ─── */

int gpu_dx11_fir_setup(dx11_context_t *c, const float *taps, int ntaps,
                        int num_stages, bool upsample) {
    c->ntaps      = ntaps;
    c->num_stages = num_stages;
    c->upsample   = upsample;

    /* Create taps buffer + SRV */
    if (c->srv_taps) { ID3D11ShaderResourceView_Release(c->srv_taps); c->srv_taps = NULL; }
    if (c->buf_taps) { ID3D11Buffer_Release(c->buf_taps); c->buf_taps = NULL; }

    /* Create and upload taps as structured buffer */
    D3D11_BUFFER_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.ByteWidth           = (UINT)(ntaps * sizeof(float));
    desc.Usage               = D3D11_USAGE_DEFAULT;
    desc.BindFlags           = D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(float);

    D3D11_SUBRESOURCE_DATA init;
    memset(&init, 0, sizeof(init));
    init.pSysMem = taps;

    if (FAILED(ID3D11Device_CreateBuffer(c->device, &desc, &init, &c->buf_taps)))
        return -1;
    if (FAILED(create_srv(c->device, c->buf_taps, (UINT)ntaps, &c->srv_taps)))
        return -1;

    return 0;
}

/* ─── Single-stage FIR dispatch ─── */

/* Run one FIR stage: input SRV → output UAV. */
static int dx11_fir_stage(dx11_context_t *c, bool up,
                           ID3D11ShaderResourceView *in_srv,
                           ID3D11UnorderedAccessView *out_uav,
                           UINT in_count, UINT out_count) {
    /* Update constant buffer */
    struct { UINT in_count, out_count, ntaps, pad; } params;
    params.in_count  = in_count;
    params.out_count = out_count;
    params.ntaps     = (UINT)c->ntaps;
    params.pad       = 0;
    ID3D11DeviceContext_UpdateSubresource(c->ctx, (ID3D11Resource *)c->buf_params,
                                          0, NULL, &params, 0, 0);

    /* Bind resources */
    ID3D11ComputeShader *cs = up ? c->cs_fir_up : c->cs_fir_down;
    ID3D11DeviceContext_CSSetShader(c->ctx, cs, NULL, 0);
    ID3D11ShaderResourceView *srvs[2] = { in_srv, c->srv_taps };
    ID3D11DeviceContext_CSSetShaderResources(c->ctx, 0, 2, srvs);
    ID3D11DeviceContext_CSSetUnorderedAccessViews(c->ctx, 0, 1, &out_uav, NULL);
    ID3D11DeviceContext_CSSetConstantBuffers(c->ctx, 0, 1, &c->buf_params);

    /* Dispatch: ceil(out_count / 256) groups */
    UINT groups = (out_count + 255) / 256;
    ID3D11DeviceContext_Dispatch(c->ctx, groups, 1, 1);

    /* Unbind to allow re-binding as SRV in next stage */
    ID3D11ShaderResourceView *null_srv[2] = { NULL, NULL };
    ID3D11UnorderedAccessView *null_uav = NULL;
    ID3D11DeviceContext_CSSetShaderResources(c->ctx, 0, 2, null_srv);
    ID3D11DeviceContext_CSSetUnorderedAccessViews(c->ctx, 0, 1, &null_uav, NULL);

    return 0;
}

/* ─── Public FIR chain process ─── */

int gpu_dx11_fir_chain(dx11_context_t *c, const float *in, float *out,
                        size_t in_count, size_t *out_count) {
    if (!c || !c->device || in_count < GPU_MIN_SAMPLES)
        return -1;

    int stages = c->num_stages;
    if (stages <= 0) { *out_count = in_count; return -1; }

    /* Calculate final output size */
    size_t cur = in_count;
    for (int s = 0; s < stages; s++)
        cur = c->upsample ? cur * 2 : cur / 2;
    size_t final_out = cur;

    /* Calculate max intermediate size */
    size_t max_inter = in_count;
    cur = in_count;
    for (int s = 0; s < stages; s++) {
        cur = c->upsample ? cur * 2 : cur / 2;
        if (cur > max_inter) max_inter = cur;
    }

    /* Ensure buffers */
    if (ensure_buf_in(c, in_count) != 0) return -1;
    if (ensure_buf_out(c, final_out) != 0) return -1;
    if (ensure_staging(c, final_out) != 0) return -1;
    if (stages > 1 && ensure_buf_inter(c, max_inter) != 0) return -1;

    /* Upload input */
    upload_buf(c->ctx, c->buf_in, in, in_count);

    /* Multi-stage dispatch */
    cur = in_count;
    for (int s = 0; s < stages; s++) {
        size_t next = c->upsample ? cur * 2 : cur / 2;

        /* Determine source and dest for this stage */
        ID3D11ShaderResourceView *src_srv;
        ID3D11UnorderedAccessView *dst_uav;

        if (s == 0) {
            src_srv = c->srv_in;
        } else {
            /* Previous stage wrote to out or inter — need to read from there.
             * Ping-pong: even stages read from inter, odd from out.
             * But first stage reads from input. Use out for final. */
            if (stages == 2) {
                src_srv = (s == 1) ? c->srv_inter : c->srv_in;
            } else {
                /* For 3+ stages, alternate between inter and a second
                 * intermediate. Simplification: copy GPU→GPU between stages
                 * using out as second buffer. */
                src_srv = (s & 1) ? c->srv_inter : c->srv_in;
            }
        }

        if (s == stages - 1) {
            dst_uav = c->uav_out;
        } else {
            /* Write to intermediate */
            if (ensure_buf_inter(c, next) != 0) return -1;
            dst_uav = c->uav_inter;
        }

        /* For multi-stage: after stage 0, source is previous output.
         * We need to copy the output to input or use ping-pong.
         * Simplest: for stages > 1, use inter as ping-pong. */
        if (s > 0 && s < stages - 1) {
            /* Copy previous output (inter) to input buf for next stage */
            ID3D11DeviceContext_CopyResource(c->ctx,
                (ID3D11Resource *)c->buf_in, (ID3D11Resource *)c->buf_inter);
            /* Grow input buf if needed */
            if (cur > c->cap_in) {
                ensure_buf_in(c, cur);
            }
            src_srv = c->srv_in;
            dst_uav = (s == stages - 1) ? c->uav_out : c->uav_inter;
        } else if (s == stages - 1 && s > 0) {
            /* Last stage: read from inter, write to out */
            src_srv = c->srv_inter;
            dst_uav = c->uav_out;
        }

        dx11_fir_stage(c, c->upsample, src_srv, dst_uav,
                        (UINT)cur, (UINT)next);
        cur = next;
    }

    /* Download result */
    if (download_buf(c->ctx, c->buf_out, c->buf_staging, out, final_out) != 0)
        return -1;

    *out_count = final_out;
    return 0;
}

/* ─── Gain ─── */

int gpu_dx11_gain(dx11_context_t *c, float *buf, size_t count, float gain) {
    if (!c || count < GPU_MIN_SAMPLES) return -1;

    if (ensure_buf_out(c, count) != 0) return -1;
    if (ensure_staging(c, count) != 0) return -1;

    /* Upload to output UAV (in-place operation) */
    upload_buf(c->ctx, c->buf_out, buf, count);

    /* Update params */
    struct { UINT count; float gain_val; UINT pad0, pad1; } params;
    params.count    = (UINT)count;
    params.gain_val = gain;
    params.pad0 = params.pad1 = 0;
    ID3D11DeviceContext_UpdateSubresource(c->ctx, (ID3D11Resource *)c->buf_params,
                                          0, NULL, &params, 0, 0);

    /* Dispatch */
    ID3D11DeviceContext_CSSetShader(c->ctx, c->cs_gain, NULL, 0);
    ID3D11DeviceContext_CSSetUnorderedAccessViews(c->ctx, 0, 1, &c->uav_out, NULL);
    ID3D11DeviceContext_CSSetConstantBuffers(c->ctx, 0, 1, &c->buf_params);
    ID3D11DeviceContext_Dispatch(c->ctx, ((UINT)count + 255) / 256, 1, 1);

    /* Unbind */
    ID3D11UnorderedAccessView *null_uav = NULL;
    ID3D11DeviceContext_CSSetUnorderedAccessViews(c->ctx, 0, 1, &null_uav, NULL);

    /* Download */
    return download_buf(c->ctx, c->buf_out, c->buf_staging, buf, count);
}

/* ─── Boxcar ─── */

int gpu_dx11_boxcar(dx11_context_t *c, const float *in, float *out,
                     size_t count, int taps, float gain) {
    if (!c || count < GPU_MIN_SAMPLES) return -1;

    if (ensure_buf_in(c, count) != 0) return -1;
    if (ensure_buf_out(c, count) != 0) return -1;
    if (ensure_staging(c, count) != 0) return -1;

    upload_buf(c->ctx, c->buf_in, in, count);

    struct { UINT count, box_taps; float gain_val; UINT pad; } params;
    params.count    = (UINT)count;
    params.box_taps = (UINT)taps;
    params.gain_val = gain;
    params.pad      = 0;
    ID3D11DeviceContext_UpdateSubresource(c->ctx, (ID3D11Resource *)c->buf_params,
                                          0, NULL, &params, 0, 0);

    ID3D11DeviceContext_CSSetShader(c->ctx, c->cs_boxcar, NULL, 0);
    ID3D11ShaderResourceView *srvs[1] = { c->srv_in };
    ID3D11DeviceContext_CSSetShaderResources(c->ctx, 0, 1, srvs);
    ID3D11DeviceContext_CSSetUnorderedAccessViews(c->ctx, 0, 1, &c->uav_out, NULL);
    ID3D11DeviceContext_CSSetConstantBuffers(c->ctx, 0, 1, &c->buf_params);
    ID3D11DeviceContext_Dispatch(c->ctx, ((UINT)count + 255) / 256, 1, 1);

    ID3D11ShaderResourceView *null_srv = NULL;
    ID3D11UnorderedAccessView *null_uav = NULL;
    ID3D11DeviceContext_CSSetShaderResources(c->ctx, 0, 1, &null_srv);
    ID3D11DeviceContext_CSSetUnorderedAccessViews(c->ctx, 0, 1, &null_uav, NULL);

    return download_buf(c->ctx, c->buf_out, c->buf_staging, out, count);
}

/* ─── DX11 Trellis SDM ─── */

int gpu_dx11_trellis_setup(dx11_context_t *c, int num_cands, int order,
                            int trellis_lat, const double *ntf_a,
                            const double *ntf_g, double state_limit) {
    /* DX11 Trellis uses the cs_trellis shader with constant buffer
     * for NTF params. State stored in UAV structured buffers. */
    if (!c || !c->cs_trellis) return -1;
    (void)num_cands; (void)order; (void)trellis_lat;
    (void)ntf_a; (void)ntf_g; (void)state_limit;
    /* TODO: create UAV buffers for state, upload NTF to constant buffer */
    return -1; /* Not yet fully implemented — falls back to CPU */
}

int gpu_dx11_trellis(dx11_context_t *c, const float *in, float *out,
                      size_t count) {
    if (!c || !c->cs_trellis) return -1;
    (void)in; (void)out; (void)count;
    /* TODO: dispatch cs_trellis with persistent state buffers */
    return -1; /* Not yet fully implemented — falls back to CPU */
}
