/*
 * foo_dsd_trellis — FIR half-band filter for DSD rate conversion
 *
 * Uses Intel IPP ippsFIRSR_32f with a 63-tap Kaiser half-band filter.
 * Multi-stage 2x up/down-sampling:
 *   Upsample:   zero-stuff → FIR → scale by 2
 *   Downsample: FIR → decimate (keep every other sample)
 */

#include "../include/fir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ipps.h>
#include <ippcore.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define IPP_HB_KAISER_BETA 12.0   /* ~120 dB stopband */

/* Global half-band taps for GPU sharing */
float g_hb_taps[IPP_HB_NTAPS] = {0};
int   g_hb_ntaps = IPP_HB_NTAPS;
static bool g_hb_taps_initialized = false;

/* ═══════════════════════════════════════════════════════════════════════
 * Half-band filter design
 * ═══════════════════════════════════════════════════════════════════════ */

static double bessel_I0(double x) {
    double sum = 1.0;
    double term = 1.0;
    double x_half_sq = (x / 2.0) * (x / 2.0);

    for (int k = 1; k <= 25; k++) {
        term *= x_half_sq / ((double)k * (double)k);
        sum += term;
        if (term < 1e-20 * sum)
            break;
    }
    return sum;
}

static void design_halfband_kaiser(double *h, int ntaps, double beta) {
    int center = (ntaps - 1) / 2;
    double I0b = bessel_I0(beta);
    double sum = 0.0;

    for (int n = 0; n < ntaps; n++) {
        double x = (double)(n - center) / 2.0;
        double sinc_val = (fabs(x) < 1e-15) ? 1.0 : sin(M_PI * x) / (M_PI * x);

        double t = (double)(n - center) / center;
        double arg = 1.0 - t * t;
        double w = bessel_I0(beta * sqrt(arg > 0.0 ? arg : 0.0)) / I0b;

        h[n] = sinc_val * w;
        sum += h[n];
    }

    for (int n = 0; n < ntaps; n++)
        h[n] /= sum;
}

/* ═══════════════════════════════════════════════════════════════════════
 * IPP FIRSR per-stage init/free
 * ═══════════════════════════════════════════════════════════════════════ */

static int ipp_firsr_stage_init(fir_chain_t *chain, int stage_idx) {
    double hd[IPP_HB_NTAPS];
    design_halfband_kaiser(hd, IPP_HB_NTAPS, IPP_HB_KAISER_BETA);

    Ipp32f taps[IPP_HB_NTAPS];
    for (int i = 0; i < IPP_HB_NTAPS; i++)
        taps[i] = (Ipp32f)hd[i];

    /* Cache taps globally for GPU backend */
    if (!g_hb_taps_initialized) {
        memcpy(g_hb_taps, taps, sizeof(g_hb_taps));
        g_hb_taps_initialized = true;
    }

    int specSize = 0, bufSize = 0;
    IppStatus st = ippsFIRSRGetSize(IPP_HB_NTAPS, ipp32f, &specSize, &bufSize);
    if (st != ippStsNoErr)
        return -1;

    IppsFIRSpec_32f *spec = (IppsFIRSpec_32f *)ippsMalloc_8u(specSize);
    if (!spec)
        return -1;

    st = ippsFIRSRInit_32f(taps, IPP_HB_NTAPS, ippAlgDirect, spec);
    if (st != ippStsNoErr) {
        ippsFree(spec);
        return -1;
    }

    Ipp8u *buf = ippsMalloc_8u(bufSize);
    if (!buf) {
        ippsFree(spec);
        return -1;
    }

    int dlyLen = IPP_HB_NTAPS - 1;
    Ipp32f *dly = ippsMalloc_32f(dlyLen);
    if (!dly) {
        ippsFree(buf);
        ippsFree(spec);
        return -1;
    }
    ippsZero_32f(dly, dlyLen);

    chain->ipp_spec[stage_idx] = spec;
    chain->ipp_buf[stage_idx] = buf;
    chain->ipp_dly[stage_idx] = dly;
    chain->ipp_taps_len = IPP_HB_NTAPS;

    return 0;
}

static void ipp_firsr_stage_free(fir_chain_t *chain, int stage_idx) {
    if (chain->ipp_spec[stage_idx]) {
        ippsFree(chain->ipp_spec[stage_idx]);
        chain->ipp_spec[stage_idx] = NULL;
    }
    if (chain->ipp_buf[stage_idx]) {
        ippsFree(chain->ipp_buf[stage_idx]);
        chain->ipp_buf[stage_idx] = NULL;
    }
    if (chain->ipp_dly[stage_idx]) {
        ippsFree(chain->ipp_dly[stage_idx]);
        chain->ipp_dly[stage_idx] = NULL;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * DSD-Wide demodulation stage init/free
 *
 * Applies the same half-band LP filter at the input DSD rate to convert
 * 1-bit ±1.0 into multi-bit "DSD-Wide" before rate conversion.
 * This suppresses the massive out-of-band noise that otherwise creates
 * un-filterable spectral images during zero-stuff upsampling.
 * ═══════════════════════════════════════════════════════════════════════ */

static int ipp_firsr_demod_init(fir_chain_t *chain) {
    double hd[IPP_HB_NTAPS];
    design_halfband_kaiser(hd, IPP_HB_NTAPS, IPP_HB_KAISER_BETA);

    Ipp32f taps[IPP_HB_NTAPS];
    for (int i = 0; i < IPP_HB_NTAPS; i++)
        taps[i] = (Ipp32f)hd[i];

    int specSize = 0, bufSize = 0;
    IppStatus st = ippsFIRSRGetSize(IPP_HB_NTAPS, ipp32f, &specSize, &bufSize);
    if (st != ippStsNoErr)
        return -1;

    IppsFIRSpec_32f *spec = (IppsFIRSpec_32f *)ippsMalloc_8u(specSize);
    if (!spec)
        return -1;

    st = ippsFIRSRInit_32f(taps, IPP_HB_NTAPS, ippAlgDirect, spec);
    if (st != ippStsNoErr) {
        ippsFree(spec);
        return -1;
    }

    Ipp8u *buf = ippsMalloc_8u(bufSize);
    if (!buf) {
        ippsFree(spec);
        return -1;
    }

    int dlyLen = IPP_HB_NTAPS - 1;
    Ipp32f *dly = ippsMalloc_32f(dlyLen);
    if (!dly) {
        ippsFree(buf);
        ippsFree(spec);
        return -1;
    }
    ippsZero_32f(dly, dlyLen);

    chain->demod_spec = spec;
    chain->demod_buf = buf;
    chain->demod_dly = dly;
    return 0;
}

static void ipp_firsr_demod_free(fir_chain_t *chain) {
    if (chain->demod_spec) { ippsFree(chain->demod_spec); chain->demod_spec = NULL; }
    if (chain->demod_buf)  { ippsFree(chain->demod_buf);  chain->demod_buf = NULL; }
    if (chain->demod_dly)  { ippsFree(chain->demod_dly);  chain->demod_dly = NULL; }
    if (chain->demod_tmp)  { ippsFree(chain->demod_tmp);  chain->demod_tmp = NULL; }
    chain->demod_tmp_sz = 0;
    chain->has_demod = false;
}

static int ensure_demod_tmp(fir_chain_t *chain, size_t need) {
    if (chain->demod_tmp_sz >= need)
        return 0;
    if (chain->demod_tmp)
        ippsFree(chain->demod_tmp);
    chain->demod_tmp = ippsMalloc_32f((int)need);
    chain->demod_tmp_sz = chain->demod_tmp ? need : 0;
    return chain->demod_tmp ? 0 : -1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Zero-stuff temp buffer
 * ═══════════════════════════════════════════════════════════════════════ */

static int ipp_ensure_zerostuff(fir_chain_t *chain, size_t need) {
    if (chain->ipp_zerostuff_sz >= need)
        return 0;
    if (chain->ipp_zerostuff)
        ippsFree(chain->ipp_zerostuff);
    chain->ipp_zerostuff = ippsMalloc_32f((int)need);
    chain->ipp_zerostuff_sz = chain->ipp_zerostuff ? need : 0;
    return chain->ipp_zerostuff ? 0 : -1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Upsample / Downsample 2x via IPP FIRSR
 * ═══════════════════════════════════════════════════════════════════════ */

static size_t ipp_upsample2(fir_chain_t *chain, int stage_idx,
                             const float *in, float *out, size_t count) {
    size_t out_count = count * 2;

    if (ipp_ensure_zerostuff(chain, out_count) != 0)
        return 0;

    /* Zero-stuff: in[i] → tmp[2i], 0 → tmp[2i+1] */
    ippsZero_32f(chain->ipp_zerostuff, (int)out_count);
    for (size_t i = 0; i < count; i++)
        chain->ipp_zerostuff[2 * i] = in[i];

    IppsFIRSpec_32f *spec = (IppsFIRSpec_32f *)chain->ipp_spec[stage_idx];
    Ipp32f *dly = chain->ipp_dly[stage_idx];
    Ipp8u *buf = (Ipp8u *)chain->ipp_buf[stage_idx];

    ippsFIRSR_32f(chain->ipp_zerostuff, out, (int)out_count,
                  spec, dly, dly, buf);

    /* Scale by 2 to compensate for zero insertion */
    ippsMulC_32f_I(2.0f, out, (int)out_count);

    return out_count;
}

static size_t ipp_downsample2(fir_chain_t *chain, int stage_idx,
                               const float *in, float *out, size_t in_count) {
    size_t out_count = in_count / 2;

    if (ipp_ensure_zerostuff(chain, in_count) != 0)
        return 0;

    IppsFIRSpec_32f *spec = (IppsFIRSpec_32f *)chain->ipp_spec[stage_idx];
    Ipp32f *dly = chain->ipp_dly[stage_idx];
    Ipp8u *buf = (Ipp8u *)chain->ipp_buf[stage_idx];

    ippsFIRSR_32f(in, chain->ipp_zerostuff, (int)in_count,
                  spec, dly, dly, buf);

    /* Decimate: keep every other sample */
    for (size_t i = 0; i < out_count; i++)
        out[i] = chain->ipp_zerostuff[2 * i];

    return out_count;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Scratch buffer management
 * ═══════════════════════════════════════════════════════════════════════ */

static int ensure_scratch(fir_chain_t *chain, size_t need) {
    if (chain->scratch_size >= need)
        return 0;
    float *p = (float *)realloc(chain->scratch, need * sizeof(float));
    if (!p)
        return -1;
    chain->scratch = p;
    chain->scratch_size = need;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════ */

int fir_chain_init(fir_chain_t *chain, uint32_t fs_in, uint32_t fs_out) {
    memset(chain, 0, sizeof(*chain));

    if (fs_in == fs_out) {
        chain->num_stages = 0;
        return 0;
    }

    /* Determine number of ×2 or ÷2 stages needed */
    uint32_t ratio;

    if (fs_out > fs_in) {
        ratio = fs_out / fs_in;
        chain->upsample = true;
    } else {
        ratio = fs_in / fs_out;
        chain->upsample = false;
    }

    /* Must be power of 2 */
    if (ratio == 0 || (ratio & (ratio - 1)) != 0)
        return -1;

    int stages = 0;
    uint32_t r = ratio;
    while (r > 1) {
        stages++;
        r >>= 1;
    }

    if (stages > FIR_MAX_STAGES)
        return -1;

    chain->num_stages = stages;

    /* Init per-stage IPP FIRSR specs */
    for (int i = 0; i < stages; i++) {
        if (ipp_firsr_stage_init(chain, i) != 0) {
            for (int j = 0; j < i; j++)
                ipp_firsr_stage_free(chain, j);
            return -1;
        }
    }

    /* DSD-Wide demod disabled: the unfiltered DSD noise acts as natural
     * dithering for the SDM re-encoder, producing better end-to-end SINAD
     * than pre-filtered (demod) input despite worse FIR-only SINAD. */

    return 0;
}

size_t fir_chain_process(fir_chain_t *chain,
                         const float *in, float *out,
                         size_t in_count) {
    if (chain->num_stages == 0) {
        memcpy(out, in, in_count * sizeof(float));
        return in_count;
    }

    /* DSD-Wide demod: LP filter at input rate to convert 1-bit → multi-bit */
    const float *rate_in = in;
    if (chain->has_demod) {
        if (ensure_demod_tmp(chain, in_count) != 0)
            return 0;
        ippsFIRSR_32f(in, chain->demod_tmp, (int)in_count,
                       (IppsFIRSpec_32f *)chain->demod_spec,
                       chain->demod_dly, chain->demod_dly,
                       (Ipp8u *)chain->demod_buf);
        rate_in = chain->demod_tmp;
    }

    /* Single stage: direct rate_in → out */
    if (chain->num_stages == 1) {
        if (chain->upsample)
            return ipp_upsample2(chain, 0, rate_in, out, in_count);
        else
            return ipp_downsample2(chain, 0, rate_in, out, in_count);
    }

    /* Multi-stage: ping-pong between scratch and out */
    size_t max_intermediate;
    if (chain->upsample)
        max_intermediate = in_count << (chain->num_stages - 1);
    else
        max_intermediate = in_count / 2;

    if (ensure_scratch(chain, max_intermediate) != 0)
        return 0;

    float *bufs[2] = { out, chain->scratch };
    const float *src = rate_in;
    size_t count = in_count;

    for (int i = 0; i < chain->num_stages; i++) {
        int remaining = chain->num_stages - 1 - i;
        float *dst = bufs[remaining & 1];

        if (chain->upsample)
            count = ipp_upsample2(chain, i, src, dst, count);
        else
            count = ipp_downsample2(chain, i, src, dst, count);

        src = dst;
    }

    return count;
}

void fir_chain_reset(fir_chain_t *chain) {
    if (chain->has_demod && chain->demod_dly)
        ippsZero_32f(chain->demod_dly, chain->ipp_taps_len - 1);

    int dlyLen = chain->ipp_taps_len - 1;
    for (int i = 0; i < chain->num_stages; i++) {
        if (chain->ipp_dly[i])
            ippsZero_32f(chain->ipp_dly[i], dlyLen);
    }
}

void fir_chain_free(fir_chain_t *chain) {
    ipp_firsr_demod_free(chain);

    for (int i = 0; i < FIR_MAX_STAGES; i++)
        ipp_firsr_stage_free(chain, i);

    if (chain->ipp_zerostuff) {
        ippsFree(chain->ipp_zerostuff);
        chain->ipp_zerostuff = NULL;
        chain->ipp_zerostuff_sz = 0;
    }

    free(chain->scratch);
    chain->scratch = NULL;
    chain->scratch_size = 0;
    chain->num_stages = 0;
}

const char *fir_ipp_version(void) {
    static char ver[128] = {0};
    if (ver[0] == 0) {
        const IppLibraryVersion *v = ippGetLibVersion();
        if (v)
            snprintf(ver, sizeof(ver), "%s %s", v->Name, v->Version);
        else
            snprintf(ver, sizeof(ver), "IPP (unknown version)");
    }
    return ver;
}

const char *fir_ipp_kernel_name(void) {
    return "IPP FIRSR (63-tap Kaiser)";
}

/* ═══════════════════════════════════════════════════════════════════════
 * Same-rate lowpass FIR for DSD-Wide re-encoding
 * ═══════════════════════════════════════════════════════════════════════
 * Replaces the boxcar with a proper windowed sinc lowpass.
 * Produces smooth multi-bit output that Trellis SDM can track.
 * Cutoff at ~50 kHz (audio band + margin) regardless of DSD rate.
 */

#define LP_NTAPS 127         /* more taps = sharper transition, less noise leakage */
#define LP_KAISER_BETA 10.0  /* ~100 dB stopband */

int fir_lowpass_init(fir_lowpass_t *lp, uint32_t dsd_rate) {
    memset(lp, 0, sizeof(*lp));

    /* Design lowpass kernel: cutoff at 50 kHz.
     * With 127 taps the transition band is ~22 kHz (50-72 kHz),
     * rejecting most ultrasonic noise while preserving SDM headroom.
     * 22 kHz cutoff caused Gibbs ringing artifacts (pops). */
    double fc = 50000.0 / ((double)dsd_rate / 2.0);  /* normalized cutoff */
    if (fc > 0.5) fc = 0.5;  /* can't exceed Nyquist/2 */

    float taps[LP_NTAPS];
    int M = LP_NTAPS - 1;
    double I0_beta = bessel_I0(LP_KAISER_BETA);

    for (int n = 0; n <= M; n++) {
        double x = (double)n - (double)M / 2.0;
        /* Sinc */
        double sinc = (fabs(x) < 1e-10) ? 2.0 * fc : sin(2.0 * M_PI * fc * x) / (M_PI * x);
        /* Kaiser window */
        double arg = 1.0 - (2.0 * n / M - 1.0) * (2.0 * n / M - 1.0);
        double w = bessel_I0(LP_KAISER_BETA * sqrt(fabs(arg))) / I0_beta;
        taps[n] = (float)(sinc * w);
    }

    /* Normalize to unity gain at DC */
    float sum = 0.0f;
    for (int n = 0; n < LP_NTAPS; n++) sum += taps[n];
    if (sum > 0.0f)
        for (int n = 0; n < LP_NTAPS; n++) taps[n] /= sum;

    /* Create IPP FIRSR spec */
    int specSize = 0, bufSize = 0;
    if (ippsFIRSRGetSize(LP_NTAPS, ipp32f, &specSize, &bufSize) != ippStsNoErr)
        return -1;

    lp->spec = ippsMalloc_8u(specSize);
    lp->buf = ippsMalloc_8u(bufSize);
    lp->dly = ippsMalloc_32f(LP_NTAPS - 1);
    if (!lp->spec || !lp->buf || !lp->dly) {
        fir_lowpass_free(lp);
        return -1;
    }

    ippsZero_32f(lp->dly, LP_NTAPS - 1);

    if (ippsFIRSRInit_32f(taps, LP_NTAPS, ippAlgAuto,
                           (IppsFIRSpec_32f *)lp->spec) != ippStsNoErr) {
        fir_lowpass_free(lp);
        return -1;
    }

    lp->coeffs = (float *)malloc(LP_NTAPS * sizeof(float));
    if (lp->coeffs)
        memcpy(lp->coeffs, taps, LP_NTAPS * sizeof(float));

    lp->taps = LP_NTAPS;
    lp->initialized = true;
    return 0;
}

size_t fir_lowpass_process(fir_lowpass_t *lp, const float *in, float *out, size_t count) {
    if (!lp->initialized || count == 0)
        return 0;

    ippsFIRSR_32f(in, out, (int)count,
                   (IppsFIRSpec_32f *)lp->spec,
                   lp->dly, lp->dly,
                   (Ipp8u *)lp->buf);
    return count;
}

void fir_lowpass_reset(fir_lowpass_t *lp) {
    if (lp->initialized && lp->dly)
        ippsZero_32f(lp->dly, lp->taps - 1);
}

void fir_lowpass_free(fir_lowpass_t *lp) {
    if (lp->spec) { ippsFree(lp->spec); lp->spec = NULL; }
    if (lp->buf)  { ippsFree(lp->buf);  lp->buf = NULL; }
    if (lp->dly)  { ippsFree(lp->dly);  lp->dly = NULL; }
    if (lp->coeffs) { free(lp->coeffs); lp->coeffs = NULL; }
    lp->initialized = false;
}
