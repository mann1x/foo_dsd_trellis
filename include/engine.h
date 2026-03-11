/*
 * foo_dsd_trellis — Top-level per-channel processing orchestrator
 */

#ifndef ENGINE_H
#define ENGINE_H

#include "dsd_types.h"
#include "trellis.h"
#include "fir.h"

/* Per-channel processing state */
typedef struct {
    sdm_context_t  sdm;        /* Trellis SDM context */
    fir_chain_t    fir;        /* FIR rate conversion chain */
    float         *fir_buf;    /* Post-FIR intermediate buffer */
    size_t         fir_buf_sz; /* Allocated size of fir_buf */
    int            channel;    /* Channel index (0=L, 1=R, ...) */
    bool           passthrough;/* true if no processing needed */
} engine_channel_t;

/* Work item dispatched to thread pool */
typedef struct {
    float              *in;      /* Float32 DSD samples at Fs_in (unpacked) */
    float              *out;     /* Float32 DSD samples at Fs_out (before repack) */
    size_t              count;   /* Sample count at Fs_in */
    size_t              out_count; /* Output sample count (may differ if rate-converting) */
    int                 channel; /* Channel index */
    engine_channel_t   *eng;     /* Engine channel state */
    const dsd_config_t *cfg;     /* Immutable config snapshot */
} channel_block_t;

/* Initialise engine for a channel with given config.
 * Returns 0 on success. */
int engine_channel_init(engine_channel_t *eng, int channel,
                        const dsd_config_t *cfg);

/* Process one block for a channel. Performs FIR + gain + SDM (or passthrough).
 * Returns number of output samples. */
size_t engine_process_block(engine_channel_t *eng,
                            const float *in, float *out,
                            size_t count, const dsd_config_t *cfg);

/* Reset engine state (seek / discontinuity). */
void engine_channel_reset(engine_channel_t *eng);

/* Free engine resources. */
void engine_channel_free(engine_channel_t *eng);

/* Reconfigure engine for new settings (rate change, filter change, etc.).
 * Returns 0 on success. */
int engine_channel_reconfigure(engine_channel_t *eng,
                               const dsd_config_t *cfg);

#endif /* ENGINE_H */
