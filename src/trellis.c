/*
 * foo_dsd_trellis — Trellis (Viterbi look-ahead) SDM core
 *
 * Phase 0: Scaffold — stub implementations.
 * Phase 3 will port the full algorithm from mansr/sox sdm.c.
 */

#include "../include/trellis.h"
#include <stdlib.h>
#include <string.h>

int sdm_context_init(sdm_context_t *ctx, const ntf_filter_t *filter,
                     int trellis_depth, int trellis_cands, int trellis_lat) {
    memset(ctx, 0, sizeof(*ctx));

    if (!filter)
        return -1;

    if (trellis_depth > SDM_TRELLIS_MAX_ORDER ||
        trellis_cands > SDM_TRELLIS_MAX_NUM ||
        trellis_lat > SDM_TRELLIS_MAX_LAT)
        return -1;

    ctx->filter = filter;
    ctx->trellis_num = (uint32_t)trellis_cands;
    ctx->trellis_lat = (uint32_t)trellis_lat;
    ctx->trellis_mask = ((uint64_t)1 << trellis_depth) - 1;
    ctx->num_cands = 1;

    /* Init history buffer free list */
    for (unsigned i = 0; i < 2u * (unsigned)trellis_cands; i++)
        ctx->hist_free[ctx->hist_fnum++] = (uint8_t)i;

    /* Init first candidate */
    sdm_trellis_t *st = &ctx->trellis[0];
    st->sdm[0].hist = ctx->hist_free[--ctx->hist_fnum];
    st->sdm[0].path = 0;
    st->act[0] = &st->sdm[0];

    return 0;
}

size_t sdm_process_block(sdm_context_t *ctx,
                         const float *in, float *out, size_t count) {
    /* TODO (Phase 3): Full trellis Viterbi algorithm */
    /* For now: simple sign quantiser (no noise shaping) */
    (void)ctx;
    for (size_t i = 0; i < count; i++)
        out[i] = in[i] >= 0.0f ? 1.0f : -1.0f;
    return count;
}

size_t sdm_drain(sdm_context_t *ctx, float *out, size_t max_out) {
    /* TODO (Phase 3): Flush pending trellis latency samples */
    (void)ctx;
    (void)out;
    (void)max_out;
    return 0;
}

void sdm_context_reset(sdm_context_t *ctx) {
    if (!ctx->filter)
        return;

    const ntf_filter_t *f = ctx->filter;
    uint32_t mask = ctx->trellis_mask;
    uint32_t num = ctx->trellis_num;
    uint32_t lat = ctx->trellis_lat;

    memset(ctx->trellis, 0, sizeof(ctx->trellis));
    memset(ctx->path_hash, 0, sizeof(ctx->path_hash));
    memset(ctx->hist, 0, sizeof(ctx->hist));

    ctx->hist_fnum = 0;
    ctx->num_cands = 1;
    ctx->pos = 0;
    ctx->pending = 0;
    ctx->draining = 0;
    ctx->idx = 0;
    ctx->prev_y = 0.0;
    ctx->conv_fail = 0;

    ctx->filter = f;
    ctx->trellis_mask = mask;
    ctx->trellis_num = num;
    ctx->trellis_lat = lat;

    for (unsigned i = 0; i < 2u * num; i++)
        ctx->hist_free[ctx->hist_fnum++] = (uint8_t)i;

    sdm_trellis_t *st = &ctx->trellis[0];
    st->sdm[0].hist = ctx->hist_free[--ctx->hist_fnum];
    st->sdm[0].path = 0;
    st->act[0] = &st->sdm[0];
}

void sdm_context_free(sdm_context_t *ctx) {
    /* Context is caller-owned, just zero it out */
    memset(ctx, 0, sizeof(*ctx));
}
