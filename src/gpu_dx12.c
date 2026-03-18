/*
 * foo_dsd_trellis — Full DX12 Async Compute GPU backend
 *
 * Complete replacement for DX11. All operations on async compute queue.
 * Uses root UAV descriptors (no descriptor heap needed for binding).
 * Targets AMD GPUs without CUDA support.
 *
 * Operations: FIR upsample/downsample, gain, boxcar, parallel-segment Trellis SDM.
 */

#include "../include/gpu_compute.h"

#define WIN32_LEAN_AND_MEAN
#define COBJMACROS
#define INITGUID
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

extern void trellis_log_c(const char *msg);

typedef HRESULT (WINAPI *PFN_D3DCompile)(
    LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO *, ID3DInclude *,
    LPCSTR, LPCSTR, UINT, UINT, ID3DBlob **, ID3DBlob **);

/* ─── Delay-load ─── */

typedef HRESULT (WINAPI *PFN_D3D12CreateDevice)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
typedef HRESULT (WINAPI *PFN_D3D12SerializeRootSignature)(const D3D12_ROOT_SIGNATURE_DESC*,
    D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);
typedef HRESULT (WINAPI *PFN_CreateDXGIFactory1)(REFIID, void**);

static HMODULE g_d3d12 = NULL, g_dxgi = NULL, g_d3dc = NULL;
static PFN_D3D12CreateDevice g_create12 = NULL;
static PFN_D3D12SerializeRootSignature g_serialize_rs = NULL;
static PFN_CreateDXGIFactory1 g_create_dxgi = NULL;
static PFN_D3DCompile g_compile = NULL;
static bool g_probed = false, g_available = false;
static char g_name[128] = "";
static size_t g_vram = 0;

/* ─── Shader IDs ─── */
enum { PSO_FIR_UP, PSO_FIR_DOWN, PSO_GAIN, PSO_BOXCAR, PSO_TRELLIS, PSO_COUNT };

/* ─── Context ─── */
typedef struct {
    gpu_backend_t         backend;
    ID3D12Device         *dev;
    ID3D12CommandQueue   *queue;
    ID3D12CommandAllocator *alloc;
    ID3D12GraphicsCommandList *cmd;
    ID3D12Fence          *fence;
    HANDLE                fence_ev;
    UINT64                fence_val;
    ID3D12RootSignature  *rs;
    ID3D12PipelineState  *pso[PSO_COUNT];
    /* Buffers */
    ID3D12Resource       *b_in, *b_out, *b_aux;  /* default heap (UAV) */
    ID3D12Resource       *b_up, *b_rb;            /* upload / readback */
    ID3D12Resource       *b_cb;                   /* constant buffer (upload, mapped) */
    void                 *cb_map;
    size_t                cap_in, cap_out, cap_aux, cap_up, cap_rb;
    /* FIR config */
    int ntaps, num_stages;
    bool upsample;
    float fir_taps[64];
    /* Trellis config */
    int tr_cands, tr_order, tr_lat;
    float tr_a[8], tr_g[8], tr_limit;
    bool tr_ready;
} dx12_t;

/* ─── Helpers ─── */

static void wait_gpu(dx12_t *c) {
    c->fence_val++;
    ID3D12CommandQueue_Signal(c->queue, c->fence, c->fence_val);
    if (ID3D12Fence_GetCompletedValue(c->fence) < c->fence_val) {
        ID3D12Fence_SetEventOnCompletion(c->fence, c->fence_val, c->fence_ev);
        WaitForSingleObject(c->fence_ev, INFINITE);
    }
}

static void begin_cmd(dx12_t *c) {
    ID3D12CommandAllocator_Reset(c->alloc);
    ID3D12GraphicsCommandList_Reset(c->cmd, c->alloc, NULL);
}

static void exec_cmd(dx12_t *c) {
    ID3D12GraphicsCommandList_Close(c->cmd);
    ID3D12CommandList *l = (ID3D12CommandList*)c->cmd;
    ID3D12CommandQueue_ExecuteCommandLists(c->queue, 1, &l);
    wait_gpu(c);
}

static ID3D12Resource *mk_buf(ID3D12Device *d, UINT64 sz, D3D12_HEAP_TYPE h,
                                D3D12_RESOURCE_STATES st, D3D12_RESOURCE_FLAGS fl) {
    D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = h;
    D3D12_RESOURCE_DESC rd = {0};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = sz; rd.Height = 1; rd.DepthOrArraySize = 1;
    rd.MipLevels = 1; rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = fl;
    ID3D12Resource *r = NULL;
    ID3D12Device_CreateCommittedResource(d, &hp, D3D12_HEAP_FLAG_NONE,
        &rd, st, NULL, &IID_ID3D12Resource, (void**)&r);
    return r;
}

static int grow_buf(dx12_t *c, ID3D12Resource **b, size_t *cap, size_t need,
                     D3D12_HEAP_TYPE h, D3D12_RESOURCE_STATES st, D3D12_RESOURCE_FLAGS fl) {
    if (*cap >= need) return 0;
    if (*b) { wait_gpu(c); ID3D12Resource_Release(*b); }
    *b = mk_buf(c->dev, need, h, st, fl);
    *cap = *b ? need : 0;
    return *b ? 0 : -1;
}

#define GR_IN(c,n)  grow_buf(c,&(c)->b_in, &(c)->cap_in, n, D3D12_HEAP_TYPE_DEFAULT, \
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
#define GR_OUT(c,n) grow_buf(c,&(c)->b_out,&(c)->cap_out,n, D3D12_HEAP_TYPE_DEFAULT, \
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
#define GR_AUX(c,n) grow_buf(c,&(c)->b_aux,&(c)->cap_aux,n, D3D12_HEAP_TYPE_DEFAULT, \
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
#define GR_UP(c,n)  grow_buf(c,&(c)->b_up, &(c)->cap_up, n, D3D12_HEAP_TYPE_UPLOAD, \
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE)
#define GR_RB(c,n)  grow_buf(c,&(c)->b_rb, &(c)->cap_rb, n, D3D12_HEAP_TYPE_READBACK, \
    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE)

/* Upload host → upload heap → default heap buf */
static void upload(dx12_t *c, ID3D12Resource *dst, const void *data, size_t bytes) {
    void *m = NULL;
    D3D12_RANGE r0 = {0,0};
    ID3D12Resource_Map(c->b_up, 0, &r0, &m);
    memcpy(m, data, bytes);
    D3D12_RANGE w = {0, bytes};
    ID3D12Resource_Unmap(c->b_up, 0, &w);
    ID3D12GraphicsCommandList_CopyBufferRegion(c->cmd, dst, 0, c->b_up, 0, bytes);
    /* UAV barrier after copy */
    D3D12_RESOURCE_BARRIER bar = {0};
    bar.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    bar.UAV.pResource = dst;
    ID3D12GraphicsCommandList_ResourceBarrier(c->cmd, 1, &bar);
}

/* Download default heap → readback → host */
static int download(dx12_t *c, ID3D12Resource *src, void *data, size_t bytes) {
    D3D12_RESOURCE_BARRIER bar = {0};
    bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Transition.pResource = src;
    bar.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    ID3D12GraphicsCommandList_ResourceBarrier(c->cmd, 1, &bar);
    ID3D12GraphicsCommandList_CopyBufferRegion(c->cmd, c->b_rb, 0, src, 0, bytes);
    exec_cmd(c);

    void *m = NULL;
    D3D12_RANGE rr = {0, bytes};
    if (FAILED(ID3D12Resource_Map(c->b_rb, 0, &rr, &m))) return -1;
    memcpy(data, m, bytes);
    D3D12_RANGE none = {0,0};
    ID3D12Resource_Unmap(c->b_rb, 0, &none);

    /* Transition back */
    begin_cmd(c);
    bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    bar.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    ID3D12GraphicsCommandList_ResourceBarrier(c->cmd, 1, &bar);
    exec_cmd(c);
    return 0;
}

/* Set PSO + root sig + CBV, then dispatch */
static void dispatch(dx12_t *c, int pso_id, UINT groups_x, UINT groups_y) {
    ID3D12GraphicsCommandList_SetComputeRootSignature(c->cmd, c->rs);
    ID3D12GraphicsCommandList_SetPipelineState(c->cmd, c->pso[pso_id]);
    /* Root param 0 = CBV */
    ID3D12GraphicsCommandList_SetComputeRootConstantBufferView(c->cmd, 0,
        ID3D12Resource_GetGPUVirtualAddress(c->b_cb));
    /* Root param 1-4 = UAV u0-u3 */
    if (c->b_in)
        ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(c->cmd, 1,
            ID3D12Resource_GetGPUVirtualAddress(c->b_in));
    if (c->b_out)
        ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(c->cmd, 2,
            ID3D12Resource_GetGPUVirtualAddress(c->b_out));
    if (c->b_aux)
        ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(c->cmd, 3,
            ID3D12Resource_GetGPUVirtualAddress(c->b_aux));
    ID3D12GraphicsCommandList_Dispatch(c->cmd, groups_x, groups_y, 1);
    /* UAV barrier */
    D3D12_RESOURCE_BARRIER bar = {0};
    bar.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    bar.UAV.pResource = NULL; /* all UAVs */
    ID3D12GraphicsCommandList_ResourceBarrier(c->cmd, 1, &bar);
}

/* ─── HLSL sources ─── */

static const char *g_hlsl[] = {
/* PSO_FIR_UP */
"RWStructuredBuffer<float> u0:register(u0); RWStructuredBuffer<float> u1:register(u1);\n"
"RWStructuredBuffer<float> u2:register(u2);\n"
"cbuffer P:register(b0){uint ic;uint oc;uint nt;uint p0;};\n"
"[numthreads(256,1,1)]void main(uint3 d:SV_DispatchThreadID){\n"
"uint o=d.x;if(o>=oc)return;float a=0;\n"
"for(uint k=0;k<nt;k++){int z=(int)o-(int)k;if(z>=0&&z<(int)(ic*2)){float v=(z&1)?0:u0[z>>1];a+=u2[k]*v;}}\n"
"u1[o]=a*2.0;}\n",
/* PSO_FIR_DOWN */
"RWStructuredBuffer<float> u0:register(u0); RWStructuredBuffer<float> u1:register(u1);\n"
"RWStructuredBuffer<float> u2:register(u2);\n"
"cbuffer P:register(b0){uint ic;uint oc;uint nt;uint p0;};\n"
"[numthreads(256,1,1)]void main(uint3 d:SV_DispatchThreadID){\n"
"uint o=d.x;if(o>=oc)return;int ii=(int)(o*2);float a=0;\n"
"for(uint k=0;k<nt;k++){int s=ii-(int)k;if(s>=0&&s<(int)ic)a+=u2[k]*u0[s];}u1[o]=a;}\n",
/* PSO_GAIN */
"RWStructuredBuffer<float> u0:register(u0);\n"
"cbuffer P:register(b0){uint cnt;float gv;uint p0;uint p1;};\n"
"[numthreads(256,1,1)]void main(uint3 d:SV_DispatchThreadID){if(d.x<cnt)u0[d.x]*=gv;}\n",
/* PSO_BOXCAR */
"RWStructuredBuffer<float> u0:register(u0); RWStructuredBuffer<float> u1:register(u1);\n"
"cbuffer P:register(b0){uint cnt;uint bt;float gv;uint p0;};\n"
"[numthreads(256,1,1)]void main(uint3 d:SV_DispatchThreadID){\n"
"uint i=d.x;if(i>=cnt)return;float s=0;int b=(int)bt;\n"
"for(int k=0;k<b;k++){int si=(int)i-k;float v=(si>=0)?u0[si]:0;s+=(v>=0?1.0:-1.0);}\n"
"u1[i]=s/(float)bt*gv;}\n",
/* PSO_TRELLIS parallel segments */
"RWStructuredBuffer<float> u0:register(u0);RWStructuredBuffer<float> u1:register(u1);\n"
"RWStructuredBuffer<int> u2:register(u2);RWStructuredBuffer<int> u3:register(u3);\n"
"cbuffer P:register(b0){int st;int wu;int nc;int ord;\n"
"float a0,a1,a2,a3,a4,a5,a6,a7,g0,g1,g2,g3,g4,g5,g6,g7;float sl;float p0,p1,p2;};\n"
"groupshared float sa[8],sg_[8],ss[64][8],sc_[64];groupshared uint sp[64];\n"
"groupshared int sac;groupshared uint so;\n"
"[numthreads(64,1,1)]void main(uint3 gid:SV_GroupID,uint3 tid:SV_GroupThreadID){\n"
"int se=gid.x;int t=tid.x;\n"
"if(t==0){sa[0]=a0;sa[1]=a1;sa[2]=a2;sa[3]=a3;sa[4]=a4;sa[5]=a5;sa[6]=a6;sa[7]=a7;\n"
"sg_[0]=g0;sg_[1]=g1;sg_[2]=g2;sg_[3]=g3;sg_[4]=g4;sg_[5]=g5;sg_[6]=g6;sg_[7]=g7;sac=nc;}\n"
"if(t<nc){for(int i=0;i<ord;i++)ss[t][i]=0;sc_[t]=0;sp[t]=0;}\n"
"GroupMemoryBarrierWithGroupSync();\n"
"int sst=u2[se];int ost=u3[se];int oi=0;\n"
"for(int si=0;si<st;si++){float x=u0[sst+si];int ac=sac;\n"
"if(t<2*ac){int pi=(int)((uint)t>>1);float yb=(t&1)?-1.0f:1.0f;float d[8];\n"
"d[0]=ss[pi][0]-sg_[0]*ss[pi][1]+x;\n"
"for(int ka=1;ka<ord-1;ka++)d[ka]=ss[pi][ka]+ss[pi][ka-1]-sg_[ka]*ss[pi][ka+1];\n"
"d[ord-1]=ss[pi][ord-1]+ss[pi][ord-2];\n"
"float v=x;for(int kb=0;kb<ord;kb++)v+=sa[kb]*d[kb];d[0]+=yb;\n"
"if(sl>0){for(int kc=0;kc<ord;kc++)d[kc]=clamp(d[kc],-sl,sl);}\n"
"int ci=nc+t;for(int kd=0;kd<ord;kd++)ss[ci][kd]=d[kd];\n"
"sc_[ci]=sc_[pi]+(v+sa[0]*yb)*(v+sa[0]*yb);sp[ci]=(sp[pi]<<1|(uint)(t&1))&0xFF;}\n"
"GroupMemoryBarrierWithGroupSync();\n"
"if(t==0){int tc=2*ac;for(int ri=0;ri<ac;ri++){int b=nc+ri;\n"
"for(int rj=nc+ri+1;rj<nc+tc;rj++)if(sc_[rj]<sc_[b])b=rj;\n"
"if(b!=nc+ri){float c2=sc_[nc+ri];sc_[nc+ri]=sc_[b];sc_[b]=c2;\n"
"uint p2=sp[nc+ri];sp[nc+ri]=sp[b];sp[b]=p2;\n"
"for(int re=0;re<ord;re++){float s2=ss[nc+ri][re];ss[nc+ri][re]=ss[b][re];ss[b][re]=s2;}}}\n"
"so=sp[nc]&1;float mc=sc_[nc];\n"
"for(int mi=0;mi<ac;mi++){sc_[mi]=sc_[nc+mi]-mc;sp[mi]=sp[nc+mi];\n"
"for(int mf=0;mf<ord;mf++)ss[mi][mf]=ss[nc+mi][mf];}}\n"
"GroupMemoryBarrierWithGroupSync();\n"
"if(t==0&&si>=wu)u1[ost+oi++]=so?1.0f:-1.0f;}}\n"
};

static const char *g_shader_names[] = {
    "fir_up", "fir_down", "gain", "boxcar", "trellis"
};

/* ─── Probe ─── */

bool gpu_dx12_probe(void) {
    if (g_probed) return g_available;
    g_probed = true;

    g_d3d12 = LoadLibraryA("d3d12.dll");
    g_dxgi  = LoadLibraryA("dxgi.dll");
    g_d3dc  = LoadLibraryA("d3dcompiler_47.dll");
    if (!g_d3d12) return (g_available = false);

    g_create12 = (PFN_D3D12CreateDevice)GetProcAddress(g_d3d12, "D3D12CreateDevice");
    g_serialize_rs = (PFN_D3D12SerializeRootSignature)GetProcAddress(g_d3d12, "D3D12SerializeRootSignature");
    if (g_dxgi) g_create_dxgi = (PFN_CreateDXGIFactory1)GetProcAddress(g_dxgi, "CreateDXGIFactory1");
    if (g_d3dc) g_compile = (PFN_D3DCompile)GetProcAddress(g_d3dc, "D3DCompile");

    if (!g_create12 || !g_serialize_rs || !g_compile)
        return (g_available = false);

    ID3D12Device *d = NULL;
    if (FAILED(g_create12(NULL, D3D_FEATURE_LEVEL_12_0, &IID_ID3D12Device, (void**)&d)))
        return (g_available = false);

    if (g_create_dxgi) {
        IDXGIFactory1 *f = NULL;
        if (SUCCEEDED(g_create_dxgi(&IID_IDXGIFactory1, (void**)&f))) {
            IDXGIAdapter1 *a = NULL;
            if (SUCCEEDED(IDXGIFactory1_EnumAdapters1(f, 0, &a))) {
                DXGI_ADAPTER_DESC1 desc;
                if (SUCCEEDED(IDXGIAdapter1_GetDesc1(a, &desc))) {
                    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, g_name, sizeof(g_name), NULL, NULL);
                    g_vram = desc.DedicatedVideoMemory;
                }
                IDXGIAdapter1_Release(a);
            }
            IDXGIFactory1_Release(f);
        }
    }
    ID3D12Device_Release(d);
    g_available = true;
    trellis_log_c("DX12 async compute available");
    return true;
}

void gpu_dx12_get_info(gpu_info_t *info) {
    memset(info, 0, sizeof(*info));
    info->backend = GPU_BACKEND_DIRECTX;
    info->available = g_available;
    strncpy_s(info->device_name, sizeof(info->device_name), g_name, _TRUNCATE);
    info->vram_mb = g_vram / (1024*1024);
}

/* ─── Create ─── */

gpu_context_t *gpu_dx12_create_full(void) {
    if (!g_available) return NULL;
    dx12_t *c = (dx12_t*)calloc(1, sizeof(dx12_t));
    if (!c) return NULL;
    c->backend = GPU_BACKEND_DIRECTX;

    if (FAILED(g_create12(NULL, D3D_FEATURE_LEVEL_12_0, &IID_ID3D12Device, (void**)&c->dev)))
        goto fail;

    /* Async compute queue */
    D3D12_COMMAND_QUEUE_DESC qd = {0};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (FAILED(ID3D12Device_CreateCommandQueue(c->dev, &qd,
            &IID_ID3D12CommandQueue, (void**)&c->queue))) goto fail;
    if (FAILED(ID3D12Device_CreateCommandAllocator(c->dev,
            D3D12_COMMAND_LIST_TYPE_COMPUTE,
            &IID_ID3D12CommandAllocator, (void**)&c->alloc))) goto fail;
    if (FAILED(ID3D12Device_CreateCommandList(c->dev, 0,
            D3D12_COMMAND_LIST_TYPE_COMPUTE, c->alloc, NULL,
            &IID_ID3D12GraphicsCommandList, (void**)&c->cmd))) goto fail;
    ID3D12GraphicsCommandList_Close(c->cmd);

    if (FAILED(ID3D12Device_CreateFence(c->dev, 0, D3D12_FENCE_FLAG_NONE,
            &IID_ID3D12Fence, (void**)&c->fence))) goto fail;
    c->fence_ev = CreateEventW(NULL, FALSE, FALSE, NULL);

    /* Root signature: b0(CBV) + u0,u1,u2,u3 as root UAVs */
    {
        D3D12_ROOT_PARAMETER p[5] = {0};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        p[0].Descriptor.ShaderRegister = 0;
        p[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        for (int i = 0; i < 4; i++) {
            p[1+i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            p[1+i].Descriptor.ShaderRegister = (UINT)i;
            p[1+i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }
        D3D12_ROOT_SIGNATURE_DESC rsd = {0};
        rsd.NumParameters = 5;
        rsd.pParameters = p;
        ID3DBlob *b = NULL;
        if (FAILED(g_serialize_rs(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &b, NULL))) goto fail;
        HRESULT hr = ID3D12Device_CreateRootSignature(c->dev, 0,
            ID3D10Blob_GetBufferPointer(b), ID3D10Blob_GetBufferSize(b),
            &IID_ID3D12RootSignature, (void**)&c->rs);
        ID3D10Blob_Release(b);
        if (FAILED(hr)) goto fail;
    }

    /* Compile all shaders */
    for (int i = 0; i < PSO_COUNT; i++) {
        ID3DBlob *blob = NULL, *err = NULL;
        HRESULT hr = g_compile(g_hlsl[i], strlen(g_hlsl[i]), g_shader_names[i],
                                NULL, NULL, "main", "cs_5_1", 0, 0, &blob, &err);
        if (err) {
            char msg[512];
            sprintf_s(msg, sizeof(msg), "DX12 HLSL [%s]: %.400s",
                g_shader_names[i], (const char*)ID3D10Blob_GetBufferPointer(err));
            trellis_log_c(msg);
            ID3D10Blob_Release(err);
        }
        if (SUCCEEDED(hr) && blob) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {0};
            pd.pRootSignature = c->rs;
            pd.CS.pShaderBytecode = ID3D10Blob_GetBufferPointer(blob);
            pd.CS.BytecodeLength = ID3D10Blob_GetBufferSize(blob);
            ID3D12Device_CreateComputePipelineState(c->dev, &pd,
                &IID_ID3D12PipelineState, (void**)&c->pso[i]);
            ID3D10Blob_Release(blob);
        }
    }
    {
        char msg[256];
        sprintf_s(msg, sizeof(msg), "DX12 PSOs: fir_up=%p fir_down=%p gain=%p boxcar=%p trellis=%p",
            (void*)c->pso[PSO_FIR_UP], (void*)c->pso[PSO_FIR_DOWN],
            (void*)c->pso[PSO_GAIN], (void*)c->pso[PSO_BOXCAR], (void*)c->pso[PSO_TRELLIS]);
        trellis_log_c(msg);
    }
    if (!c->pso[PSO_FIR_UP] || !c->pso[PSO_FIR_DOWN] ||
        !c->pso[PSO_GAIN] || !c->pso[PSO_BOXCAR]) goto fail;

    /* Constant buffer (256 bytes, persistently mapped) */
    c->b_cb = mk_buf(c->dev, 256, D3D12_HEAP_TYPE_UPLOAD,
                       D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
    if (c->b_cb) {
        D3D12_RANGE r = {0,0};
        ID3D12Resource_Map(c->b_cb, 0, &r, &c->cb_map);
    }

    trellis_log_c("DX12 async compute: fully initialized");
    return (gpu_context_t*)c;

fail:
    trellis_log_c("DX12 create failed");
    free(c);
    return NULL;
}

void gpu_dx12_destroy_full(void *p) {
    dx12_t *c = (dx12_t*)p;
    if (!c) return;
    if (c->queue) wait_gpu(c);
    if (c->b_cb) ID3D12Resource_Release(c->b_cb);
    if (c->b_rb) ID3D12Resource_Release(c->b_rb);
    if (c->b_up) ID3D12Resource_Release(c->b_up);
    if (c->b_aux) ID3D12Resource_Release(c->b_aux);
    if (c->b_out) ID3D12Resource_Release(c->b_out);
    if (c->b_in) ID3D12Resource_Release(c->b_in);
    for (int i = 0; i < PSO_COUNT; i++)
        if (c->pso[i]) ID3D12PipelineState_Release(c->pso[i]);
    if (c->rs) ID3D12RootSignature_Release(c->rs);
    if (c->fence_ev) CloseHandle(c->fence_ev);
    if (c->fence) ID3D12Fence_Release(c->fence);
    if (c->cmd) ID3D12GraphicsCommandList_Release(c->cmd);
    if (c->alloc) ID3D12CommandAllocator_Release(c->alloc);
    if (c->queue) ID3D12CommandQueue_Release(c->queue);
    if (c->dev) ID3D12Device_Release(c->dev);
    free(c);
}

/* ─── FIR Setup ─── */

int gpu_dx12_fir_setup(dx12_t *c, const float *taps, int ntaps,
                        int num_stages, bool up) {
    if (!c) return -1;
    c->ntaps = ntaps;
    c->num_stages = num_stages;
    c->upsample = up;
    memcpy(c->fir_taps, taps, (size_t)ntaps * sizeof(float));
    return 0;
}

/* ─── FIR Chain ─── */

int gpu_dx12_fir_chain(dx12_t *c, const float *in, float *out,
                        size_t in_count, size_t *out_count) {
    if (!c || in_count < GPU_MIN_SAMPLES || c->num_stages <= 0) return -1;

    size_t cur = in_count;
    for (int s = 0; s < c->num_stages; s++)
        cur = c->upsample ? cur*2 : cur/2;
    size_t final_out = cur;
    size_t max_sz = (in_count > final_out ? in_count : final_out) * sizeof(float);
    size_t taps_sz = (size_t)c->ntaps * sizeof(float);

    if (GR_IN(c, max_sz) || GR_OUT(c, max_sz) || GR_AUX(c, taps_sz) ||
        GR_UP(c, max_sz > taps_sz ? max_sz : taps_sz) || GR_RB(c, max_sz))
        return -1;

    /* Upload input data */
    begin_cmd(c);
    upload(c, c->b_in, in, in_count * sizeof(float));
    exec_cmd(c);

    /* Upload taps separately (b_up is shared, can't do two uploads in one cmd) */
    begin_cmd(c);
    upload(c, c->b_aux, c->fir_taps, taps_sz);
    exec_cmd(c);

    /* Multi-stage FIR */
    cur = in_count;
    for (int s = 0; s < c->num_stages; s++) {
        size_t next = c->upsample ? cur*2 : cur/2;

        /* Update CB: in_count, out_count, ntaps */
        if (c->cb_map) {
            UINT params[4] = {(UINT)cur, (UINT)next, (UINT)c->ntaps, 0};
            memcpy(c->cb_map, params, 16);
        }

        /* Dispatch: u0=b_in(input), u1=b_out(output), u2=b_aux(taps) */
        begin_cmd(c);
        dispatch(c, c->upsample ? PSO_FIR_UP : PSO_FIR_DOWN, (UINT)((next+255)/256), 1);
        exec_cmd(c);

        /* For multi-stage: copy output→input for next stage */
        if (s < c->num_stages - 1) {
            begin_cmd(c);
            /* Barrier: b_out UAV→COPY_SOURCE */
            D3D12_RESOURCE_BARRIER bar = {0};
            bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            bar.Transition.pResource = c->b_out;
            bar.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            ID3D12GraphicsCommandList_ResourceBarrier(c->cmd, 1, &bar);

            ID3D12GraphicsCommandList_CopyBufferRegion(c->cmd,
                c->b_in, 0, c->b_out, 0, next * sizeof(float));

            /* Barrier: b_out back to UAV, b_in back to UAV */
            D3D12_RESOURCE_BARRIER bars[2] = {0};
            bars[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            bars[0].Transition.pResource = c->b_out;
            bars[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            bars[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            bars[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            bars[1].UAV.pResource = c->b_in;
            ID3D12GraphicsCommandList_ResourceBarrier(c->cmd, 2, bars);

            exec_cmd(c);
        }
        cur = next;
    }

    /* Download from b_out */
    begin_cmd(c);
    if (download(c, c->b_out, out, final_out * sizeof(float)) != 0) return -1;

    *out_count = final_out;
    return 0;
}

/* ─── Gain ─── */

int gpu_dx12_gain(dx12_t *c, float *buf, size_t count, float gain) {
    if (!c || count < GPU_MIN_SAMPLES) return -1;
    size_t bytes = count * sizeof(float);
    if (GR_IN(c, bytes) || GR_UP(c, bytes) || GR_RB(c, bytes)) return -1;

    begin_cmd(c);
    upload(c, c->b_in, buf, bytes);
    if (c->cb_map) {
        struct { UINT cnt; float gv; UINT p0, p1; } p = {(UINT)count, gain, 0, 0};
        memcpy(c->cb_map, &p, sizeof(p));
    }
    /* u0=buf (in-place via b_in) */
    dispatch(c, PSO_GAIN, (UINT)((count+255)/256), 1);
    exec_cmd(c);

    begin_cmd(c);
    if (download(c, c->b_in, buf, bytes) != 0) return -1;
    return 0;
}

/* ─── Boxcar ─── */

int gpu_dx12_boxcar(dx12_t *c, const float *in, float *out,
                     size_t count, int taps, float gain) {
    if (!c || count < GPU_MIN_SAMPLES) return -1;
    size_t bytes = count * sizeof(float);
    if (GR_IN(c,bytes) || GR_OUT(c,bytes) || GR_UP(c,bytes) || GR_RB(c,bytes)) return -1;

    begin_cmd(c);
    upload(c, c->b_in, in, bytes);
    if (c->cb_map) {
        struct { UINT cnt, bt; float gv; UINT p; } p = {(UINT)count,(UINT)taps,gain,0};
        memcpy(c->cb_map, &p, sizeof(p));
    }
    dispatch(c, PSO_BOXCAR, (UINT)((count+255)/256), 1);
    exec_cmd(c);

    begin_cmd(c);
    return download(c, c->b_out, out, bytes);
}

/* ─── Trellis Setup ─── */

int gpu_dx12_trellis_setup_full(dx12_t *c, int nc, int ord, int lat,
                                  const double *a, const double *g, double sl) {
    if (!c || !c->pso[PSO_TRELLIS]) return -1;
    c->tr_cands = nc; c->tr_order = ord; c->tr_lat = lat;
    c->tr_limit = (float)sl;
    for (int k = 0; k < ord && k < 8; k++) {
        c->tr_a[k] = (float)a[k];
        c->tr_g[k] = (float)g[k];
    }
    c->tr_ready = true;
    trellis_log_c("DX12 trellis setup OK");
    return 0;
}

/* ─── Trellis Parallel Dispatch ─── */

int gpu_dx12_trellis_full(dx12_t *c, const float *in, float *out,
                           size_t count, int num_segs) {
    if (!c || !c->tr_ready || !c->pso[PSO_TRELLIS]) return -1;

    int nc = c->tr_cands;
    int warmup = 4 * c->tr_lat;  /* 4× overlap for SDM convergence (matches CPU) */
    size_t base_seg = count / (size_t)num_segs;
    size_t seg_total = base_seg + (size_t)warmup;
    size_t total_out = base_seg * (size_t)num_segs;

    size_t in_bytes = count * sizeof(float);
    size_t out_bytes = total_out * sizeof(float);
    size_t seg_bytes = (size_t)num_segs * sizeof(int);
    size_t max_bytes = in_bytes > out_bytes ? in_bytes : out_bytes;

    /* Need 4 UAVs: u0=in, u1=out, u2=seg_starts, u3=seg_out_starts.
     * b_in=u0, b_out=u1, b_aux=u2 (seg_starts).
     * For u3 we need a 4th buffer. Reuse part of b_aux. */
    if (GR_IN(c, max_bytes) || GR_OUT(c, max_bytes) ||
        GR_AUX(c, seg_bytes * 2) || /* both seg arrays packed */
        GR_UP(c, max_bytes > seg_bytes*2 ? max_bytes : seg_bytes*2) ||
        GR_RB(c, out_bytes)) return -1;

    /* Build segment descriptors */
    int *segs = (int*)malloc(seg_bytes * 2);
    if (!segs) return -1;
    int *seg_starts = segs;
    int *seg_out_starts = segs + num_segs;
    for (int i = 0; i < num_segs; i++) {
        int s = (int)((size_t)i * base_seg) - warmup;
        if (s < 0) s = 0;
        seg_starts[i] = s;
        seg_out_starts[i] = (int)((size_t)i * base_seg);
    }

    /* Update constant buffer */
    if (c->cb_map) {
        struct {
            int st, wu, nc, ord;
            float a[8], g[8];
            float sl, p0, p1, p2;
        } p = {0};
        p.st = (int)seg_total; p.wu = warmup; p.nc = nc; p.ord = c->tr_order;
        memcpy(p.a, c->tr_a, sizeof(p.a));
        memcpy(p.g, c->tr_g, sizeof(p.g));
        p.sl = c->tr_limit;
        memcpy(c->cb_map, &p, sizeof(p));
    }

    LARGE_INTEGER t0, t1, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    begin_cmd(c);
    upload(c, c->b_in, in, in_bytes);
    /* Upload both seg arrays to b_aux */
    upload(c, c->b_aux, segs, seg_bytes * 2);
    exec_cmd(c);
    free(segs);

    /* Need separate UAV for u3 (seg_out_starts).
     * b_aux contains both packed. Create u2 at offset 0, u3 at offset seg_bytes.
     * But root UAVs don't support offsets — need a separate buffer.
     * Workaround: create a temporary buffer for seg_out_starts. */
    /* Actually, we need to split b_aux or use a 4th buffer.
     * For simplicity, create b_seg_out as a 4th default-heap buffer. */
    ID3D12Resource *b_seg_out = mk_buf(c->dev, seg_bytes, D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (!b_seg_out) return -1;

    /* Upload seg_out_starts separately */
    begin_cmd(c);
    {
        void *m = NULL;
        D3D12_RANGE r = {0,0};
        ID3D12Resource_Map(c->b_up, 0, &r, &m);
        memcpy(m, seg_out_starts, seg_bytes);
        D3D12_RANGE w = {0, seg_bytes};
        ID3D12Resource_Unmap(c->b_up, 0, &w);
        /* Wait, seg_out_starts was freed with segs. Need to rebuild. */
    }
    /* Oops — segs was freed. Let me recalculate */
    {
        int *sout = (int*)malloc(seg_bytes);
        for (int i = 0; i < num_segs; i++)
            sout[i] = (int)((size_t)i * base_seg);
        void *m = NULL;
        D3D12_RANGE r = {0,0};
        ID3D12Resource_Map(c->b_up, 0, &r, &m);
        memcpy(m, sout, seg_bytes);
        D3D12_RANGE w = {0, seg_bytes};
        ID3D12Resource_Unmap(c->b_up, 0, &w);
        free(sout);
    }
    ID3D12GraphicsCommandList_CopyBufferRegion(c->cmd, b_seg_out, 0, c->b_up, 0, seg_bytes);
    D3D12_RESOURCE_BARRIER ubar = {0};
    ubar.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    ubar.UAV.pResource = NULL;
    ID3D12GraphicsCommandList_ResourceBarrier(c->cmd, 1, &ubar);

    /* Dispatch trellis: u0=in, u1=out, u2=seg_starts(b_aux), u3=seg_out_starts(b_seg_out) */
    ID3D12GraphicsCommandList_SetComputeRootSignature(c->cmd, c->rs);
    ID3D12GraphicsCommandList_SetPipelineState(c->cmd, c->pso[PSO_TRELLIS]);
    ID3D12GraphicsCommandList_SetComputeRootConstantBufferView(c->cmd, 0,
        ID3D12Resource_GetGPUVirtualAddress(c->b_cb));
    ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(c->cmd, 1,
        ID3D12Resource_GetGPUVirtualAddress(c->b_in));
    ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(c->cmd, 2,
        ID3D12Resource_GetGPUVirtualAddress(c->b_out));
    ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(c->cmd, 3,
        ID3D12Resource_GetGPUVirtualAddress(c->b_aux));
    ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(c->cmd, 4,
        ID3D12Resource_GetGPUVirtualAddress(b_seg_out));
    ID3D12GraphicsCommandList_Dispatch(c->cmd, (UINT)num_segs, 1, 1);
    ID3D12GraphicsCommandList_ResourceBarrier(c->cmd, 1, &ubar);
    exec_cmd(c);

    QueryPerformanceCounter(&t1);

    /* Download output */
    begin_cmd(c);
    int rc = download(c, c->b_out, out, out_bytes);

    ID3D12Resource_Release(b_seg_out);

    {
        double ms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;
        char msg[256];
        sprintf_s(msg, sizeof(msg), "[DX12 SDM parallel] %zu samples, %d segs, %d cands: %.1fms",
                  count, num_segs, nc, ms);
        trellis_log_c(msg);
    }

    return rc;
}
