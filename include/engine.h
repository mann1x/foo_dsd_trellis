/*
 * foo_dsd_trellis — Top-level per-channel processing orchestrator
 */

#ifndef ENGINE_H
#define ENGINE_H

#include "dsd_types.h"
#include "trellis.h"
#include "precorr.h"
#include "fir.h"
#include "onnx_filter.h"

/* DSD-Wide boxcar filter: smooths 1-bit DSD to multi-bit at same rate.
 * Enables gain control without decimating to PCM domain. */
#define BOXCAR_TAPS 8
typedef struct {
    float ring[BOXCAR_TAPS];
    float sum;
    int   pos;
} boxcar_t;

/* Per-channel processing state */
typedef struct {
    sdm_context_t    sdm;        /* Trellis SDM context */
    precorr_context_t precorr;   /* PreCorr SDM context */
    fir_chain_t      fir;        /* FIR rate conversion chain */
    boxcar_t         boxcar;     /* DSD-Wide smoothing for volume control */
    float           *fir_buf;    /* Post-FIR intermediate buffer */
    size_t           fir_buf_sz; /* Allocated size of fir_buf */
    int              channel;    /* Channel index (0=L, 1=R, ...) */
    float            fir_gain;   /* FIR output attenuation (path-adaptive) */
    int              sdm_mode;   /* Cached sdm_mode_t for dispatch */
    bool             passthrough;/* true if no processing needed */
    bool             fir_only;   /* true for DSD→PCM decimation (no SDM) */
    onnx_filter_t   *ml_filter;  /* ONNX ML post-filter (NULL if disabled) */
} engine_channel_t;

/* Block processing mode */
#define BLOCK_MODE_FULL  0   /* FIR + gain + SDM (original path) */
#define BLOCK_MODE_SDM   1   /* SDM segment only (parallel path) */
#define BLOCK_MODE_FIR   2   /* FIR + gain only (parallel path) */

/* Work item dispatched to thread pool */
typedef struct {
    float              *in;      /* Float32 DSD samples at Fs_in (unpacked) */
    float              *out;     /* Float32 DSD samples at Fs_out (before repack) */
    size_t              count;   /* Sample count at Fs_in */
    size_t              out_count; /* Output sample count (may differ if rate-converting) */
    int                 channel; /* Channel index */
    engine_channel_t   *eng;     /* Engine channel state */
    const dsd_config_t *cfg;     /* Immutable config snapshot */
    /* Parallel processing mode fields */
    int                 mode;      /* BLOCK_MODE_FULL, _SDM, or _FIR */
    sdm_context_t      *sdm_ctx;   /* SDM context for segment processing */
    size_t              discard;    /* Output samples to discard (warmup) */
    float              *fir_out;   /* [FIR mode] pointer to FIR output buffer */
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

/* Process FIR + gain only (no SDM). Returns output sample count.
 * *fir_out receives pointer to engine's internal buffer (valid until
 * next call on same engine). */
size_t engine_process_fir_gain(engine_channel_t *eng,
                                const float *in, size_t count,
                                const dsd_config_t *cfg,
                                float **fir_out);

/* Process SDM segment with optional warmup discard.
 * For fresh SDM contexts, the first trellis_lat samples fill the latency
 * buffer (no output). Then 'discard' output samples are thrown away
 * (convergence warmup). Remaining input produces kept output.
 * Returns number of output samples kept. */
size_t sdm_segment_process(sdm_context_t *sdm,
                            const float *in, float *out,
                            size_t count, size_t discard);

/* Path info for UI display — resolved parameters for a given conversion path */
typedef struct {
    int  ntf_filter;    /* Resolved NTF filter ID (from path_table or auto-select) */
    int  cands;         /* Trellis candidates */
    int  lat;           /* Trellis latency */
    int  depth;         /* Trellis depth */
    double state_limit; /* State limiter (0 = disabled) */
    float fir_gain;     /* FIR output gain (1.0 = no attenuation) */
    int  fir_stages;    /* Number of FIR 2x stages */
    bool fir_only;      /* True if FIR decimation only (DSD→PCM) */
} engine_path_info_t;

/* Query resolved path parameters for UI info display.
 * Returns 0 on success, -1 if path is invalid. */
int engine_get_path_info(uint32_t fs_in, uint32_t fs_out,
                          int ntf_override, int sdm_mode,
                          const dsd_config_t *cfg,
                          engine_path_info_t *info);

#endif /* ENGINE_H */
