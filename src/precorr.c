/*
 * foo_dsd_trellis — PreCorr (Predictive Correction) SDM implementation
 *
 * Greedy SDM + prediction table correction. Same NTF filter topology
 * as trellis.c but in single-precision float.
 *
 * See include/precorr.h for algorithm description.
 */

#include "../include/precorr.h"
#include <string.h>
#include <math.h>

/* ─── NTF filter calc (float, fully unrolled per order) ─── */

static __forceinline float precorr_filter_o4(const float *s, float *d,
                                              const float *a, const float *g,
                                              float x, float y)
{
    d[0] = s[0] - g[0] * s[1] + x - y;
    d[1] = s[1] + s[0] - g[1] * s[2];
    d[2] = s[2] + s[1] - g[2] * s[3];
    d[3] = s[3] + s[2];
    return x + a[0] * d[0] + a[1] * d[1] + a[2] * d[2] + a[3] * d[3];
}

static __forceinline float precorr_filter_o5(const float *s, float *d,
                                              const float *a, const float *g,
                                              float x, float y)
{
    d[0] = s[0] - g[0] * s[1] + x - y;
    d[1] = s[1] + s[0] - g[1] * s[2];
    d[2] = s[2] + s[1] - g[2] * s[3];
    d[3] = s[3] + s[2] - g[3] * s[4];
    d[4] = s[4] + s[3];
    return x + a[0] * d[0] + a[1] * d[1] + a[2] * d[2]
             + a[3] * d[3] + a[4] * d[4];
}

static __forceinline float precorr_filter_o6(const float *s, float *d,
                                              const float *a, const float *g,
                                              float x, float y)
{
    d[0] = s[0] - g[0] * s[1] + x - y;
    d[1] = s[1] + s[0] - g[1] * s[2];
    d[2] = s[2] + s[1] - g[2] * s[3];
    d[3] = s[3] + s[2] - g[3] * s[4];
    d[4] = s[4] + s[3] - g[4] * s[5];
    d[5] = s[5] + s[4];
    return x + a[0] * d[0] + a[1] * d[1] + a[2] * d[2]
             + a[3] * d[3] + a[4] * d[4] + a[5] * d[5];
}

static __forceinline float precorr_filter_o7(const float *s, float *d,
                                              const float *a, const float *g,
                                              float x, float y)
{
    d[0] = s[0] - g[0] * s[1] + x - y;
    d[1] = s[1] + s[0] - g[1] * s[2];
    d[2] = s[2] + s[1] - g[2] * s[3];
    d[3] = s[3] + s[2] - g[3] * s[4];
    d[4] = s[4] + s[3] - g[4] * s[5];
    d[5] = s[5] + s[4] - g[5] * s[6];
    d[6] = s[6] + s[5];
    return x + a[0] * d[0] + a[1] * d[1] + a[2] * d[2]
             + a[3] * d[3] + a[4] * d[4] + a[5] * d[5] + a[6] * d[6];
}

static __forceinline float precorr_filter_o8(const float *s, float *d,
                                              const float *a, const float *g,
                                              float x, float y)
{
    d[0] = s[0] - g[0] * s[1] + x - y;
    d[1] = s[1] + s[0] - g[1] * s[2];
    d[2] = s[2] + s[1] - g[2] * s[3];
    d[3] = s[3] + s[2] - g[3] * s[4];
    d[4] = s[4] + s[3] - g[4] * s[5];
    d[5] = s[5] + s[4] - g[5] * s[6];
    d[6] = s[6] + s[5] - g[6] * s[7];
    d[7] = s[7] + s[6];
    return x + a[0] * d[0] + a[1] * d[1] + a[2] * d[2] + a[3] * d[3]
             + a[4] * d[4] + a[5] * d[5] + a[6] * d[6] + a[7] * d[7];
}

/* Generic fallback for non-standard orders */
static __forceinline float precorr_filter_generic(const float *s, float *d,
                                                   const float *a, const float *g,
                                                   int order, float x, float y)
{
    float v;
    int i;

    d[0] = s[0] - g[0] * s[1] + x - y;
    v = x + a[0] * d[0];

    for (i = 1; i < order - 1; i++) {
        d[i] = s[i] + s[i - 1] - g[i] * s[i + 1];
        v += a[i] * d[i];
    }

    d[i] = s[i] + s[i - 1];
    v += a[i] * d[i];

    return v;
}

/* Dispatch to order-specific filter calc */
static __forceinline float precorr_filter_calc(const float *s, float *d,
                                                const float *a, const float *g,
                                                int order, float x, float y)
{
    switch (order) {
    case 4: return precorr_filter_o4(s, d, a, g, x, y);
    case 5: return precorr_filter_o5(s, d, a, g, x, y);
    case 6: return precorr_filter_o6(s, d, a, g, x, y);
    case 7: return precorr_filter_o7(s, d, a, g, x, y);
    case 8: return precorr_filter_o8(s, d, a, g, x, y);
    default: return precorr_filter_generic(s, d, a, g, order, x, y);
    }
}

/* ─── Simple LCG PRNG for training ─── */

static __forceinline uint32_t lcg_next(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static __forceinline float lcg_float(uint32_t *state, float amplitude) {
    uint32_t r = lcg_next(state);
    /* Map [0, 2^32) to [-amplitude, +amplitude) */
    return amplitude * ((float)(int32_t)r / 2147483648.0f);
}

/* ─── Prediction table training ─── */

static void precorr_train_table(precorr_context_t *ctx) {
    /* Accumulators for mean correction per (history, phase) */
    float  accum[PRECORR_HIST_SIZE][PRECORR_PHASES];
    int    hits[PRECORR_HIST_SIZE][PRECORR_PHASES];
    memset(accum, 0, sizeof(accum));
    memset(hits, 0, sizeof(hits));

    /* Temporary greedy SDM state (no prediction) */
    float state[MAX_NTF_ORDER];
    memset(state, 0, sizeof(state));

    uint8_t history = 0x69;  /* Start with DSD silence pattern */
    int phase = 0;
    float prev_y = 0.0f;

    uint32_t rng = 0xDEADBEEF;

    for (int i = 0; i < PRECORR_TRAIN_SAMPLES; i++) {
        float x = lcg_float(&rng, PRECORR_TRAIN_AMP);

        /* Greedy quantize through NTF filter */
        float d[MAX_NTF_ORDER];
        float v = precorr_filter_calc(state, d, ctx->a, ctx->g, ctx->order, x, prev_y);
        float y = (v >= 0.0f) ? 1.0f : -1.0f;

        /* Update NTF state */
        memcpy(state, d, (size_t)ctx->order * sizeof(float));

        /* Record error: how much the greedy output differs from ideal */
        /* The "ideal" in this context is the NTF-filtered value before quantization.
         * We accumulate (y - sign(v)) which captures the quantization error pattern
         * associated with each (history, phase) pair. Since y = sign(v), the error
         * is always 0 for the greedy decision itself. Instead, we accumulate the
         * residual: how far v is from the quantized output y. */
        float error = y - v;  /* Quantization residual */
        accum[history][phase] += error;
        hits[history][phase]++;

        /* Update history and phase */
        history = (uint8_t)((history << 1) | (y > 0.0f ? 1 : 0));
        phase = (phase + 1) & (PRECORR_PHASES - 1);
        prev_y = y;
    }

    /* Compute mean corrections */
    for (int h = 0; h < PRECORR_HIST_SIZE; h++) {
        for (int p = 0; p < PRECORR_PHASES; p++) {
            if (hits[h][p] >= PRECORR_MIN_HITS)
                ctx->pred_table[h][p] = accum[h][p] / (float)hits[h][p];
            else
                ctx->pred_table[h][p] = 0.0f;
        }
    }
}

/* ─── Public API ─── */

int precorr_context_init(precorr_context_t *ctx, const ntf_filter_t *filter) {
    if (!ctx || !filter)
        return -1;

    memset(ctx, 0, sizeof(*ctx));
    ctx->filter = filter;
    ctx->order = filter->order;

    /* Copy NTF coefficients as float */
    for (int i = 0; i < filter->order; i++) {
        ctx->a[i] = (float)filter->a[i];
        ctx->g[i] = (float)filter->g[i];
    }

    ctx->history = 0x69;  /* DSD silence pattern A */
    ctx->phase = 0;
    ctx->prev_y = 0.0f;

    /* Train prediction table */
    precorr_train_table(ctx);

    return 0;
}

size_t precorr_process_block(precorr_context_t *ctx,
                             const float *in, float *out, size_t count) {
    const float *a = ctx->a;
    const float *g = ctx->g;
    const int order = ctx->order;
    uint8_t history = ctx->history;
    int phase = ctx->phase;
    float prev_y = ctx->prev_y;

    for (size_t i = 0; i < count; i++) {
        float x = in[i];

        /* NTF filter calc */
        float d[MAX_NTF_ORDER];
        float v = precorr_filter_calc(ctx->state, d, a, g, order, x, prev_y);

        /* Greedy quantize */
        float y = (v >= 0.0f) ? 1.0f : -1.0f;

        /* Update NTF state */
        memcpy(ctx->state, d, (size_t)order * sizeof(float));

        /* Apply prediction correction and re-quantize */
        float corr = ctx->pred_table[history][phase];
        float y_corr = y - corr;
        float out_y = (y_corr >= 0.0f) ? 1.0f : -1.0f;

        out[i] = out_y;

        /* Update history with corrected output */
        history = (uint8_t)((history << 1) | (out_y > 0.0f ? 1 : 0));
        phase = (phase + 1) & (PRECORR_PHASES - 1);
        prev_y = out_y;
    }

    /* Save state */
    ctx->history = history;
    ctx->phase = phase;
    ctx->prev_y = prev_y;

    return count;  /* No latency — output count == input count */
}

size_t precorr_drain(precorr_context_t *ctx, float *out, size_t max_out) {
    (void)ctx;
    (void)out;
    (void)max_out;
    return 0;  /* PreCorr has no latency, nothing to drain */
}

void precorr_context_reset(precorr_context_t *ctx) {
    if (!ctx) return;

    /* Reset SDM state but keep trained prediction table */
    memset(ctx->state, 0, sizeof(ctx->state));
    ctx->history = 0x69;
    ctx->phase = 0;
    ctx->prev_y = 0.0f;
}

void precorr_context_free(precorr_context_t *ctx) {
    /* No dynamic allocations — just zero the struct */
    if (ctx) {
        const ntf_filter_t *filter = ctx->filter;
        memset(ctx, 0, sizeof(*ctx));
        ctx->filter = filter;  /* Keep filter reference for potential re-init */
    }
}
