/*
 * foo_dsd_trellis — Polyphase FIR half-band filter for DSD rate conversion
 */

#ifndef FIR_H
#define FIR_H

#include "dsd_types.h"

/* Maximum FIR filter length (taps per polyphase phase) */
#define FIR_MAX_PHASE_TAPS 64

/* FIR filter state (per-channel, per-stage) */
typedef struct {
    double   coeffs[FIR_MAX_PHASE_TAPS];   /* Polyphase phase coefficients */
    float    delay[FIR_MAX_PHASE_TAPS];    /* Circular delay buffer */
    int      num_taps;                      /* Active taps in this phase */
    int      delay_pos;                     /* Current position in delay buffer */
} fir_phase_t;

/* Single rate conversion stage (×2 or ÷2) */
typedef struct {
    fir_phase_t phase[2];      /* Two polyphase phases for half-band */
    bool        upsample;      /* true = ×2 interpolation, false = ÷2 decimation */
} fir_stage_t;

/* Rate conversion chain (up to 3 stages for DSD64↔DSD512) */
#define FIR_MAX_STAGES 3

typedef struct {
    fir_stage_t stages[FIR_MAX_STAGES];
    int         num_stages;     /* 0 = passthrough */
    float      *scratch;        /* Intermediate buffer between stages */
    size_t      scratch_size;
} fir_chain_t;

/* Initialise rate conversion chain for given input/output rates.
 * Returns 0 on success, -1 if rates are not a valid power-of-2 ratio. */
int fir_chain_init(fir_chain_t *chain, uint32_t fs_in, uint32_t fs_out);

/* Process a block of float32 samples through the FIR chain.
 * Returns the number of output samples written. */
size_t fir_chain_process(fir_chain_t *chain,
                         const float *in, float *out,
                         size_t in_count);

/* Reset filter state (on seek / discontinuity). */
void fir_chain_reset(fir_chain_t *chain);

/* Free scratch buffer and reset. */
void fir_chain_free(fir_chain_t *chain);

#endif /* FIR_H */
