/*
 * foo_dsd_trellis — Polyphase resampler for cross-family PCM rate conversion
 *
 * Default: IPP ippsResamplePolyphaseFixed (already linked).
 * Optional: libsoxr (runtime-loaded from component folder).
 *
 * Same-family PCM conversion uses fir_chain (power-of-2 half-band).
 * This module handles non-power-of-2 ratios (e.g., 44100↔48000).
 */

#ifndef RESAMPLE_H
#define RESAMPLE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "dsd_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Resampler context (opaque internals) */
typedef struct resample_ctx resample_ctx_t;

/* Initialize a resampler for the given rate pair.
 * engine: RESAMPLE_AUTO, RESAMPLE_IPP, or RESAMPLE_SOXR.
 * soxr_quality: SOXR_QUALITY_MQ/HQ/VHQ (only used if engine is soxr).
 * Returns NULL on failure. */
resample_ctx_t *resample_create(uint32_t fs_in, uint32_t fs_out,
                                 int engine, int soxr_quality);

/* Process a block of float samples.
 * in:        input buffer (in_count samples, single channel)
 * out:       output buffer (must be large enough for ceil(in_count * fs_out/fs_in) + margin)
 * in_count:  number of input samples
 * Returns:   number of output samples produced. */
size_t resample_process(resample_ctx_t *ctx,
                         const float *in, float *out, size_t in_count);

/* Free the resampler context. */
void resample_free(resample_ctx_t *ctx);

/* Check if libsoxr.dll is available for loading.
 * Probes once, caches result. Thread-safe. */
bool resample_soxr_available(void);

/* Get the name of the active engine ("IPP" or "soxr"). */
const char *resample_engine_name(const resample_ctx_t *ctx);

/* Check if two rates require polyphase resampling (not power-of-2 ratio).
 * Returns true if fir_chain can't handle this pair. */
static inline bool resample_needed(uint32_t fs_in, uint32_t fs_out) {
    if (fs_in == fs_out) return false;
    uint32_t hi = (fs_out > fs_in) ? fs_out : fs_in;
    uint32_t lo = (fs_out > fs_in) ? fs_in  : fs_out;
    if (hi % lo != 0) return true;  /* not integer ratio */
    uint32_t ratio = hi / lo;
    return (ratio & (ratio - 1)) != 0;  /* not power of 2 */
}

#ifdef __cplusplus
}
#endif

#endif /* RESAMPLE_H */
