/*
 * foo_dsd_trellis — Polyphase resampler for cross-family PCM rate conversion
 *
 * IPP backend: ippsResamplePolyphaseFixed_32f
 * soxr backend: runtime-loaded libsoxr.dll
 */

#include "../include/resample.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ipps.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

/* ═══════════════════════════════════════════════════════════════════════
 * soxr runtime loading
 * ═══════════════════════════════════════════════════════════════════════ */

/* Minimal soxr type definitions (from soxr.h) */
typedef struct soxr *soxr_t;
typedef char const *soxr_error_t;

typedef struct {
    double       rate;
    unsigned     bits;
    unsigned     channels;
} soxr_io_spec_t;

typedef struct {
    unsigned long long recipe;
    unsigned long      flags;
    double             passband_end;
    double             stopband_begin;
    double             phase_response;
} soxr_quality_spec_t;

/* soxr quality recipes */
#define SOXR_QQ        0   /* Quick */
#define SOXR_LQ        1   /* Low */
#define SOXR_MQ        2   /* Medium */
#define SOXR_HQ        4   /* High */
#define SOXR_VHQ       6   /* Very High */

/* Function pointers */
typedef soxr_t (*fn_soxr_create)(double input_rate, double output_rate,
                                  unsigned num_channels,
                                  soxr_error_t *error,
                                  const soxr_io_spec_t *io_spec,
                                  const soxr_quality_spec_t *quality_spec,
                                  void *runtime_spec);
typedef soxr_error_t (*fn_soxr_process)(soxr_t resampler,
                                         const void *in, size_t ilen,
                                         size_t *idone,
                                         void *out, size_t olen,
                                         size_t *odone);
typedef void (*fn_soxr_delete)(soxr_t);
typedef soxr_quality_spec_t (*fn_soxr_quality_spec)(unsigned long recipe,
                                                     unsigned long flags);

static HMODULE       g_soxr_dll = NULL;
static fn_soxr_create      g_soxr_create = NULL;
static fn_soxr_process     g_soxr_process = NULL;
static fn_soxr_delete      g_soxr_delete = NULL;
static fn_soxr_quality_spec g_soxr_quality_spec = NULL;
static int           g_soxr_probed = 0;  /* 0=not probed, 1=available, -1=unavailable */

static void soxr_probe(void) {
    if (g_soxr_probed != 0) return;

    /* Try to load from component directory (next to our DLL) */
    HMODULE self = NULL;
    char path[MAX_PATH] = {0};
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)soxr_probe, &self);
    if (self) {
        GetModuleFileNameA(self, path, MAX_PATH);
        char *slash = strrchr(path, '\\');
        if (slash) {
            strcpy_s(slash + 1, (size_t)(path + MAX_PATH - slash - 1), "libsoxr.dll");
            g_soxr_dll = LoadLibraryA(path);
        }
    }
    /* Fallback: system PATH */
    if (!g_soxr_dll)
        g_soxr_dll = LoadLibraryA("libsoxr.dll");

    if (g_soxr_dll) {
        g_soxr_create = (fn_soxr_create)GetProcAddress(g_soxr_dll, "soxr_create");
        g_soxr_process = (fn_soxr_process)GetProcAddress(g_soxr_dll, "soxr_process");
        g_soxr_delete = (fn_soxr_delete)GetProcAddress(g_soxr_dll, "soxr_delete");
        g_soxr_quality_spec = (fn_soxr_quality_spec)GetProcAddress(g_soxr_dll, "soxr_quality_spec");
        if (g_soxr_create && g_soxr_process && g_soxr_delete && g_soxr_quality_spec)
            g_soxr_probed = 1;
        else {
            FreeLibrary(g_soxr_dll);
            g_soxr_dll = NULL;
            g_soxr_probed = -1;
        }
    } else {
        g_soxr_probed = -1;
    }
}

bool resample_soxr_available(void) {
    soxr_probe();
    return g_soxr_probed == 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Resampler context
 * ═══════════════════════════════════════════════════════════════════════ */

struct resample_ctx {
    uint32_t fs_in;
    uint32_t fs_out;
    int      engine;  /* 0=IPP, 1=soxr */

    /* IPP state (arbitrary-ratio polyphase) */
    IppsResamplingPolyphase_32f *ipp_spec;
    Ipp64f   ipp_time;
    int      ipp_history;  /* filter history length */

    /* soxr state */
    soxr_t   soxr;
};

/* ═══════════════════════════════════════════════════════════════════════
 * IPP backend
 * ═══════════════════════════════════════════════════════════════════════ */

/* Find GCD for rate simplification */
static uint32_t gcd(uint32_t a, uint32_t b) {
    while (b) { uint32_t t = b; b = a % b; a = t; }
    return a;
}

static resample_ctx_t *ipp_create(uint32_t fs_in, uint32_t fs_out) {
    /* Use arbitrary-ratio polyphase resampler.
     * window = filter window size in input samples.
     * nStep = number of filter interpolation steps (higher = better quality). */
    Ipp32f window = 64.0f;   /* 64 input samples window (longer = higher quality) */
    int nStep = 256;         /* 256 interpolation steps */

    int spec_size = 0;
    IppStatus st = ippsResamplePolyphaseGetSize_32f(
        window, nStep, &spec_size, ippAlgAuto);
    if (st != ippStsNoErr)
        return NULL;

    resample_ctx_t *ctx = (resample_ctx_t *)calloc(1, sizeof(resample_ctx_t));
    if (!ctx) return NULL;

    ctx->ipp_spec = (IppsResamplingPolyphase_32f *)ippsMalloc_8u(spec_size);
    if (!ctx->ipp_spec) { free(ctx); return NULL; }

    /* rollf=0.95 (95% passband), alpha=16.0 (Kaiser beta for ~160 dB stopband) */
    st = ippsResamplePolyphaseInit_32f(
        window, nStep, 0.95f, 16.0f, ctx->ipp_spec, ippAlgAuto);
    if (st != ippStsNoErr) {
        ippsFree(ctx->ipp_spec);
        free(ctx);
        return NULL;
    }

    ctx->fs_in = fs_in;
    ctx->fs_out = fs_out;
    ctx->engine = 0;
    ctx->ipp_time = 0.0;
    ctx->ipp_history = (int)window;

    return ctx;
}

static size_t ipp_process(resample_ctx_t *ctx,
                           const float *in, float *out, size_t in_count) {
    /* Arbitrary-ratio polyphase resampler.
     * pSrc points to input buffer, pTime tracks fractional position.
     * Need padding at both ends for the filter window. */
    int hist = ctx->ipp_history;
    size_t total = (size_t)hist + in_count + (size_t)hist;
    Ipp32f *padded = ippsMalloc_32f((int)total);
    if (!padded) return 0;

    ippsZero_32f(padded, (int)total);
    memcpy(padded + hist, in, in_count * sizeof(float));

    Ipp64f factor = (Ipp64f)ctx->fs_out / (Ipp64f)ctx->fs_in;
    Ipp64f time = (Ipp64f)hist;
    int out_len = 0;

    /* len = number of input samples to process (not padded total).
     * pSrc = padded buffer; pTime starts at hist so it reads actual data.
     * IPP accesses pSrc[floor(pTime)-window .. floor(pTime)+len+window]. */
    IppStatus st = ippsResamplePolyphase_32f(
        padded, (int)in_count, out,
        factor, 1.0f,
        &time, &out_len,
        ctx->ipp_spec);

    ippsFree(padded);
    if (st != ippStsNoErr)
        return 0;
    return (size_t)out_len;
}

/* ═══════════════════════════════════════════════════════════════════════
 * soxr backend
 * ═══════════════════════════════════════════════════════════════════════ */

static resample_ctx_t *soxr_backend_create(uint32_t fs_in, uint32_t fs_out,
                                            int quality) {
    soxr_probe();
    if (g_soxr_probed != 1)
        return NULL;

    /* Map quality enum to soxr recipe */
    unsigned long recipe;
    switch (quality) {
    case SOXR_QUALITY_MQ:  recipe = SOXR_MQ;  break;
    case SOXR_QUALITY_VHQ: recipe = SOXR_VHQ; break;
    default:               recipe = SOXR_HQ;  break;
    }

    soxr_quality_spec_t qspec = g_soxr_quality_spec(recipe, 0);
    soxr_error_t err = NULL;

    soxr_t soxr = g_soxr_create((double)fs_in, (double)fs_out, 1,
                                 &err, NULL, &qspec, NULL);
    if (!soxr || err)
        return NULL;

    resample_ctx_t *ctx = (resample_ctx_t *)calloc(1, sizeof(resample_ctx_t));
    if (!ctx) {
        g_soxr_delete(soxr);
        return NULL;
    }

    ctx->fs_in = fs_in;
    ctx->fs_out = fs_out;
    ctx->engine = 1;
    ctx->soxr = soxr;

    return ctx;
}

static size_t soxr_backend_process(resample_ctx_t *ctx,
                                    const float *in, float *out, size_t in_count) {
    size_t idone = 0, odone = 0;
    /* Estimate max output */
    size_t max_out = (size_t)((double)in_count * (double)ctx->fs_out / (double)ctx->fs_in) + 256;
    soxr_error_t err = g_soxr_process(ctx->soxr,
                                       in, in_count, &idone,
                                       out, max_out, &odone);
    if (err) return 0;
    return odone;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════ */

resample_ctx_t *resample_create(uint32_t fs_in, uint32_t fs_out,
                                 int engine, int soxr_quality) {
    if (fs_in == fs_out || fs_in == 0 || fs_out == 0)
        return NULL;

    /* Engine selection */
    if (engine == RESAMPLE_SOXR || (engine == RESAMPLE_AUTO && resample_soxr_available())) {
        resample_ctx_t *ctx = soxr_backend_create(fs_in, fs_out, soxr_quality);
        if (ctx) return ctx;
        /* Fall through to IPP if soxr fails */
    }

    return ipp_create(fs_in, fs_out);
}

size_t resample_process(resample_ctx_t *ctx,
                         const float *in, float *out, size_t in_count) {
    if (!ctx || !in || !out || in_count == 0)
        return 0;

    if (ctx->engine == 1)
        return soxr_backend_process(ctx, in, out, in_count);
    else
        return ipp_process(ctx, in, out, in_count);
}

void resample_free(resample_ctx_t *ctx) {
    if (!ctx) return;

    if (ctx->engine == 1 && ctx->soxr) {
        if (g_soxr_delete)
            g_soxr_delete(ctx->soxr);
    }
    if (ctx->ipp_spec)
        ippsFree((void *)ctx->ipp_spec);
    free(ctx);
}

const char *resample_engine_name(const resample_ctx_t *ctx) {
    if (!ctx) return "none";
    return ctx->engine == 1 ? "soxr" : "IPP";
}
