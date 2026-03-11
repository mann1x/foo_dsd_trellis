/*
 * foo_dsd_trellis — Polyphase FIR half-band filter for DSD rate conversion
 *
 * Half-band filter design: Kaiser-windowed sinc, computed at init time.
 * Polyphase decomposition for efficient 2x up/down-sampling:
 *   Phase 0 (non-zero taps): FIR convolution
 *   Phase 1 (only center tap = 0.5): trivial delayed copy
 *
 * For DSD rates, the transition band is extremely wide (~0.47 Fs),
 * so a short 23-tap filter provides >90 dB stopband attenuation.
 */

#include "../include/fir.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── Half-band filter parameters ─── */

#define HB_NTAPS         23     /* 4*5 + 3: valid half-band length */
#define HB_CENTER        11     /* (HB_NTAPS - 1) / 2 */
#define HB_PHASE0_TAPS   12     /* h[0], h[2], ..., h[22] */
#define HB_PHASE1_DELAY_UP   5  /* Phase 1 center at index 5 for upsample */
#define HB_PHASE1_DELAY_DN   6  /* Phase 1 delay for downsample */
#define HB_KAISER_BETA   9.0    /* ~90 dB sidelobe suppression */

/* Phase 0 coefficients (even-indexed taps of the half-band filter) */
static double hb_phase0[HB_PHASE0_TAPS];
static int hb_initialized = 0;

/* ─── Bessel I0 (modified, order 0) via series expansion ─── */

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

/* ─── Compute half-band filter coefficients ─── */

static void init_halfband(void) {
    if (hb_initialized)
        return;

    double h[HB_NTAPS];
    double I0b = bessel_I0(HB_KAISER_BETA);
    double sum = 0.0;

    for (int n = 0; n < HB_NTAPS; n++) {
        /* Windowed sinc: sinc((n - center) / 2) * kaiser(n) */
        double x = (double)(n - HB_CENTER) / 2.0;
        double sinc_val = (fabs(x) < 1e-15) ? 1.0 : sin(M_PI * x) / (M_PI * x);

        double t = (double)(n - HB_CENTER) / HB_CENTER;
        double arg = 1.0 - t * t;
        double w = bessel_I0(HB_KAISER_BETA * sqrt(arg > 0.0 ? arg : 0.0)) / I0b;

        h[n] = sinc_val * w;
        sum += h[n];
    }

    /* Normalize for unity DC gain */
    for (int n = 0; n < HB_NTAPS; n++)
        h[n] /= sum;

    /* Extract phase 0 coefficients (even indices) */
    for (int j = 0; j < HB_PHASE0_TAPS; j++)
        hb_phase0[j] = h[2 * j];

    /* Verify half-band property: h[center] should be ~0.5 */
    /* h[11] / sum ≈ 0.5 — the center tap is in phase 1 */

    hb_initialized = 1;
}

/* ─── Polyphase 2x upsample ─── */

static size_t upsample2(fir_stage_t *stage,
                         const float *in, float *out, size_t count) {
    fir_phase_t *ph = &stage->phase[0];
    int ntaps = ph->num_taps;

    for (size_t i = 0; i < count; i++) {
        /* Push new input sample */
        ph->delay[ph->delay_pos] = in[i];

        /* Phase 0 output: FIR with even-indexed coefficients, scaled ×2 */
        double sum = 0.0;
        for (int j = 0; j < ntaps; j++) {
            int idx = (ph->delay_pos - j) & (FIR_MAX_PHASE_TAPS - 1);
            sum += ph->coeffs[j] * (double)ph->delay[idx];
        }
        out[2 * i] = (float)(2.0 * sum);

        /* Phase 1 output: 1.0 × delayed input (0.5 center tap × 2 gain) */
        int d = (ph->delay_pos - HB_PHASE1_DELAY_UP) & (FIR_MAX_PHASE_TAPS - 1);
        out[2 * i + 1] = ph->delay[d];

        ph->delay_pos = (ph->delay_pos + 1) & (FIR_MAX_PHASE_TAPS - 1);
    }

    return count * 2;
}

/* ─── Polyphase 2x downsample ─── */

static size_t downsample2(fir_stage_t *stage,
                           const float *in, float *out, size_t in_count) {
    fir_phase_t *even = &stage->phase[0];  /* Even-indexed input samples */
    fir_phase_t *odd  = &stage->phase[1];  /* Odd-indexed input samples */
    int ntaps = even->num_taps;
    size_t out_count = in_count / 2;

    for (size_t i = 0; i < out_count; i++) {
        /* Push even sample x[2i] */
        even->delay[even->delay_pos] = in[2 * i];
        /* Push odd sample x[2i+1] */
        odd->delay[odd->delay_pos] = in[2 * i + 1];

        /* Phase 0: FIR convolution on even samples */
        double sum = 0.0;
        for (int j = 0; j < ntaps; j++) {
            int idx = (even->delay_pos - j) & (FIR_MAX_PHASE_TAPS - 1);
            sum += even->coeffs[j] * (double)even->delay[idx];
        }

        /* Phase 1: 0.5 × delayed odd sample */
        int d = (odd->delay_pos - HB_PHASE1_DELAY_DN) & (FIR_MAX_PHASE_TAPS - 1);
        sum += 0.5 * (double)odd->delay[d];

        out[i] = (float)sum;

        even->delay_pos = (even->delay_pos + 1) & (FIR_MAX_PHASE_TAPS - 1);
        odd->delay_pos = (odd->delay_pos + 1) & (FIR_MAX_PHASE_TAPS - 1);
    }

    return out_count;
}

/* ─── Scratch buffer management ─── */

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

/* ─── Public API ─── */

int fir_chain_init(fir_chain_t *chain, uint32_t fs_in, uint32_t fs_out) {
    memset(chain, 0, sizeof(*chain));

    init_halfband();

    if (fs_in == fs_out) {
        chain->num_stages = 0;
        return 0;
    }

    /* Determine number of ×2 or ÷2 stages needed */
    uint32_t ratio;
    bool upsample;

    if (fs_out > fs_in) {
        ratio = fs_out / fs_in;
        upsample = true;
    } else {
        ratio = fs_in / fs_out;
        upsample = false;
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

    for (int i = 0; i < stages; i++) {
        fir_stage_t *st = &chain->stages[i];
        st->upsample = upsample;

        /* Load phase 0 coefficients into both phases */
        memcpy(st->phase[0].coeffs, hb_phase0, HB_PHASE0_TAPS * sizeof(double));
        st->phase[0].num_taps = HB_PHASE0_TAPS;
        st->phase[0].delay_pos = 0;
        memset(st->phase[0].delay, 0, sizeof(st->phase[0].delay));

        /* Phase 1 used only for downsampling (odd delay line) */
        memcpy(st->phase[1].coeffs, hb_phase0, HB_PHASE0_TAPS * sizeof(double));
        st->phase[1].num_taps = HB_PHASE0_TAPS;
        st->phase[1].delay_pos = 0;
        memset(st->phase[1].delay, 0, sizeof(st->phase[1].delay));
    }

    return 0;
}

size_t fir_chain_process(fir_chain_t *chain,
                         const float *in, float *out,
                         size_t in_count) {
    if (chain->num_stages == 0) {
        /* Passthrough */
        memcpy(out, in, in_count * sizeof(float));
        return in_count;
    }

    if (chain->num_stages == 1) {
        /* Single stage: process directly into output */
        if (chain->stages[0].upsample)
            return upsample2(&chain->stages[0], in, out, in_count);
        else
            return downsample2(&chain->stages[0], in, out, in_count);
    }

    /* Multi-stage: use ping-pong between scratch and out buffers.
     * For upsampling: N → 2N → 4N → 8N (out is largest, always fits).
     * For downsampling: N → N/2 → N/4 → N/8.
     * Scratch sized for the largest intermediate result. */
    size_t max_intermediate;
    if (chain->stages[0].upsample) {
        max_intermediate = in_count << (chain->num_stages - 1);
    } else {
        max_intermediate = in_count / 2;
    }

    if (ensure_scratch(chain, max_intermediate) != 0)
        return 0;

    /* Ping-pong: stages alternate between scratch and out.
     * Last stage always writes to out. Work backwards to assign:
     *   2 stages: in → scratch → out
     *   3 stages: in → out → scratch → out */
    float *bufs[2] = { out, chain->scratch };
    const float *src = in;
    size_t count = in_count;

    for (int i = 0; i < chain->num_stages; i++) {
        int remaining = chain->num_stages - 1 - i;
        float *dst = bufs[remaining & 1]; /* last stage → bufs[0] = out */

        if (chain->stages[i].upsample)
            count = upsample2(&chain->stages[i], src, dst, count);
        else
            count = downsample2(&chain->stages[i], src, dst, count);

        src = dst;
    }

    return count;
}

void fir_chain_reset(fir_chain_t *chain) {
    for (int i = 0; i < chain->num_stages; i++) {
        memset(chain->stages[i].phase[0].delay, 0,
               sizeof(chain->stages[i].phase[0].delay));
        memset(chain->stages[i].phase[1].delay, 0,
               sizeof(chain->stages[i].phase[1].delay));
        chain->stages[i].phase[0].delay_pos = 0;
        chain->stages[i].phase[1].delay_pos = 0;
    }
}

void fir_chain_free(fir_chain_t *chain) {
    free(chain->scratch);
    chain->scratch = NULL;
    chain->scratch_size = 0;
    chain->num_stages = 0;
}
