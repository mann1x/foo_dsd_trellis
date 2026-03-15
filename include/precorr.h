/*
 * foo_dsd_trellis — PreCorr (Predictive Correction) SDM
 *
 * Greedy single-path SDM quantizer with prediction table correction.
 * O(1) per sample — table lookup + subtraction.
 * Uses single-precision float for 2x SIMD throughput vs trellis (double).
 *
 * Algorithm (per sample):
 *   1. NTF filter calc (same topology as trellis, float precision)
 *   2. Greedy quantize: y = (v >= 0) ? +1 : -1
 *   3. Lookup correction: c = pred_table[history][phase]
 *   4. Apply correction and re-quantize: out = (y - c >= 0) ? +1 : -1
 *   5. Shift history register, advance phase
 *
 * The prediction table is trained at init by running the greedy SDM
 * on pseudo-random noise and accumulating mean correction per
 * (8-bit history, phase) pair.
 */

#ifndef PRECORR_H
#define PRECORR_H

#include "dsd_types.h"
#include "ntf.h"

#define PRECORR_HIST_SIZE  256   /* 2^8 history patterns */
#define PRECORR_PHASES     8     /* corrections per pattern */

/* Training parameters */
#define PRECORR_TRAIN_SAMPLES  65536
#define PRECORR_TRAIN_AMP      0.1f    /* Small amplitude to stay in linear region */
#define PRECORR_MIN_HITS       4       /* Min hits to trust a correction entry */

typedef struct {
    float   state[MAX_NTF_ORDER];      /* NTF integrator state (float) */
    float   a[MAX_NTF_ORDER];          /* NTF feedback coeffs (float copy) */
    float   g[MAX_NTF_ORDER];          /* NTF resonator gains (float copy) */
    int     order;
    float   pred_table[PRECORR_HIST_SIZE][PRECORR_PHASES]; /* Prediction correction table */
    uint8_t history;                   /* 8-bit output history register */
    int     phase;                     /* Phase counter 0-7 */
    float   prev_y;                    /* Previous output for NTF feedback */
    float   state_limit;               /* Integrator state clamp (0 = disabled) */
    const ntf_filter_t *filter;        /* Reference for reset */
} precorr_context_t;

/* Initialize PreCorr context and train prediction table.
 * Returns 0 on success, -1 on error (null filter). */
int precorr_context_init(precorr_context_t *ctx, const ntf_filter_t *filter);

/* Process a block of float32 samples through PreCorr SDM.
 * Input: FIR-upsampled float samples. Output: ±1.0f DSD bits.
 * Returns the number of output samples written (== count, no latency). */
size_t precorr_process_block(precorr_context_t *ctx,
                             const float *in, float *out, size_t count);

/* Drain remaining samples (PreCorr has no latency, always returns 0). */
size_t precorr_drain(precorr_context_t *ctx, float *out, size_t max_out);

/* Reset state (on seek / discontinuity). Keeps trained prediction table. */
void precorr_context_reset(precorr_context_t *ctx);

/* Free resources (no-op for stack-allocated context, but matches trellis API). */
void precorr_context_free(precorr_context_t *ctx);

#endif /* PRECORR_H */
