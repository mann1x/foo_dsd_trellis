/*
 * foo_dsd_trellis — Trellis (Viterbi look-ahead) SDM core
 */

#ifndef TRELLIS_H
#define TRELLIS_H

#include "dsd_types.h"
#include "ntf.h"

#define SDM_TRELLIS_MAX_ORDER 32
#define SDM_TRELLIS_MAX_NUM   32
#define SDM_TRELLIS_MAX_LAT   2048
#define PATH_HASH_SIZE        128

/* Per-candidate path state */
typedef struct sdm_state {
    double  state[MAX_NTF_ORDER];   /* NTF integrator state vector */
    double  cost;                   /* Accumulated squared-error metric */
    uint32_t path;                  /* Bit history (trellis_mask width) */
    uint8_t  next;                  /* Output bit at traceback position */
    uint8_t  hist;                  /* History buffer index */
    uint8_t  hist_used;             /* History buffer ownership flag */
    struct sdm_state *parent;       /* Parent in trellis */
    struct sdm_state *path_list;    /* Hash chain for path dedup */
} sdm_state_t;

/* Trellis generation (double-buffered) */
typedef struct {
    sdm_state_t   sdm[2 * SDM_TRELLIS_MAX_NUM];
    sdm_state_t  *act[SDM_TRELLIS_MAX_NUM];
} sdm_trellis_t;

/* Per-channel SDM context */
typedef struct {
    sdm_trellis_t  trellis[2];
    sdm_state_t   *path_hash[PATH_HASH_SIZE];
    uint8_t        hist_free[2 * SDM_TRELLIS_MAX_NUM];
    unsigned       hist_fnum;
    uint32_t       trellis_mask;
    uint32_t       trellis_num;
    uint32_t       trellis_lat;
    unsigned       num_cands;
    unsigned       pos;
    unsigned       pending;
    unsigned       draining;
    unsigned       idx;
    const ntf_filter_t *filter;
    double         prev_y;
    uint64_t       conv_fail;
    uint8_t        hist[2 * SDM_TRELLIS_MAX_NUM][SDM_TRELLIS_MAX_LAT / 8];
} sdm_context_t;

/* Initialise SDM context for one channel. Returns 0 on success. */
int sdm_context_init(sdm_context_t *ctx, const ntf_filter_t *filter,
                     int trellis_depth, int trellis_cands, int trellis_lat);

/* Process a block of float32 samples through the trellis SDM.
 * Input:  float32 samples at Fs_out (post-FIR, post-gain)
 * Output: float32 ±1.0 (1-bit decisions)
 * Returns number of output samples produced. */
size_t sdm_process_block(sdm_context_t *ctx,
                         const float *in, float *out, size_t count);

/* Drain pending samples (flush at end of stream). */
size_t sdm_drain(sdm_context_t *ctx, float *out, size_t max_out);

/* Reset SDM state (on seek / discontinuity). */
void sdm_context_reset(sdm_context_t *ctx);

/* Free any internal allocations (context itself is caller-owned). */
void sdm_context_free(sdm_context_t *ctx);

#endif /* TRELLIS_H */
