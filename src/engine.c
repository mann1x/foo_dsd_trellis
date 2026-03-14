/*
 * foo_dsd_trellis — Top-level per-channel processing orchestrator
 *
 * Phase 0: Scaffold — stub implementations.
 * Phase 4-6 will wire up FIR → gain → SDM → repack pipeline.
 */

#include "../include/engine.h"
#include <stdlib.h>
#include <string.h>

int engine_channel_init(engine_channel_t *eng, int channel,
                        const dsd_config_t *cfg) {
    memset(eng, 0, sizeof(*eng));
    eng->channel = channel;

    uint32_t fs_out = cfg->fs_out ? cfg->fs_out : cfg->fs_in;
    eng->passthrough = (cfg->fs_in == fs_out) &&
                       (cfg->gain == 1.0f) &&
                       (!cfg->mute);

    eng->sdm_mode = cfg->sdm_mode;

    if (!eng->passthrough && !cfg->mute) {
        /* Init FIR chain */
        if (fir_chain_init(&eng->fir, cfg->fs_in, fs_out) != 0)
            return -1;

        /* Init SDM */
        const ntf_filter_t *filter = NULL;
        if (cfg->ntf_filter == NTF_AUTO) {
            if (cfg->sdm_mode == SDM_MODE_PRECORR)
                filter = ntf_auto_select_precorr(fs_out);
            else
                filter = ntf_auto_select(fs_out);
        } else {
            filter = ntf_get_filter((ntf_filter_id_t)cfg->ntf_filter, fs_out);
        }

        if (!filter)
            return -1;

        if (cfg->sdm_mode == SDM_MODE_PRECORR) {
            if (precorr_context_init(&eng->precorr, filter) != 0)
                return -1;
        } else {
            if (sdm_context_init(&eng->sdm, filter,
                                 cfg->trellis_depth,
                                 cfg->trellis_cands,
                                 cfg->trellis_lat) != 0)
                return -1;
        }
    }

    return 0;
}

size_t engine_process_block(engine_channel_t *eng,
                            const float *in, float *out,
                            size_t count, const dsd_config_t *cfg) {
    if (cfg->mute) {
        /* Fill with silence pattern equivalent (±1.0 alternating) */
        for (size_t i = 0; i < count; i++)
            out[i] = (i & 1) ? 1.0f : -1.0f;
        return count;
    }

    if (eng->passthrough) {
        /* Sign-only requantise: passthrough */
        for (size_t i = 0; i < count; i++)
            out[i] = in[i] >= 0.0f ? 1.0f : -1.0f;
        return count;
    }

    /* FIR rate conversion */
    uint32_t fs_out = cfg->fs_out ? cfg->fs_out : cfg->fs_in;
    size_t fir_out_count;
    if (fs_out >= cfg->fs_in)
        fir_out_count = count * (fs_out / cfg->fs_in);
    else
        fir_out_count = count / (cfg->fs_in / fs_out);

    /* Ensure intermediate buffer */
    if (eng->fir_buf_sz < fir_out_count * sizeof(float)) {
        free(eng->fir_buf);
        eng->fir_buf = (float *)malloc(fir_out_count * sizeof(float));
        eng->fir_buf_sz = fir_out_count * sizeof(float);
    }

    size_t fir_out = fir_chain_process(&eng->fir, in, eng->fir_buf, count);

    /* Apply gain */
    if (cfg->gain != 1.0f) {
        for (size_t i = 0; i < fir_out; i++)
            eng->fir_buf[i] *= cfg->gain;
    }

    /* SDM (PreCorr or Trellis) */
    size_t sdm_out;
    if (eng->sdm_mode == SDM_MODE_PRECORR)
        sdm_out = precorr_process_block(&eng->precorr, eng->fir_buf, out, fir_out);
    else
        sdm_out = sdm_process_block(&eng->sdm, eng->fir_buf, out, fir_out);
    return sdm_out;
}

void engine_channel_reset(engine_channel_t *eng) {
    fir_chain_reset(&eng->fir);
    if (eng->sdm_mode == SDM_MODE_PRECORR)
        precorr_context_reset(&eng->precorr);
    else
        sdm_context_reset(&eng->sdm);
}

void engine_channel_free(engine_channel_t *eng) {
    fir_chain_free(&eng->fir);
    if (eng->sdm_mode == SDM_MODE_PRECORR)
        precorr_context_free(&eng->precorr);
    else
        sdm_context_free(&eng->sdm);
    free(eng->fir_buf);
    eng->fir_buf = NULL;
    eng->fir_buf_sz = 0;
}

int engine_channel_reconfigure(engine_channel_t *eng,
                               const dsd_config_t *cfg) {
    engine_channel_free(eng);
    return engine_channel_init(eng, eng->channel, cfg);
}

size_t engine_process_fir_gain(engine_channel_t *eng,
                                const float *in, size_t count,
                                const dsd_config_t *cfg,
                                float **fir_out_ptr) {
    uint32_t fs_out = cfg->fs_out ? cfg->fs_out : cfg->fs_in;
    size_t fir_out_count;
    if (fs_out >= cfg->fs_in)
        fir_out_count = count * (fs_out / cfg->fs_in);
    else
        fir_out_count = count / (cfg->fs_in / fs_out);

    if (eng->fir_buf_sz < fir_out_count * sizeof(float)) {
        free(eng->fir_buf);
        eng->fir_buf = (float *)malloc(fir_out_count * sizeof(float));
        eng->fir_buf_sz = fir_out_count * sizeof(float);
    }

    size_t fir_count = fir_chain_process(&eng->fir, in, eng->fir_buf, count);

    if (cfg->gain != 1.0f) {
        for (size_t i = 0; i < fir_count; i++)
            eng->fir_buf[i] *= cfg->gain;
    }

    *fir_out_ptr = eng->fir_buf;
    return fir_count;
}

/* Thread-local scratch buffer for SDM warmup discard.
 * Avoids malloc/free per segment call on the hot path. */
static __declspec(thread) float *tls_trash_buf = NULL;
static __declspec(thread) size_t tls_trash_sz = 0;

static float *get_trash_buf(size_t need) {
    if (tls_trash_sz < need) {
        free(tls_trash_buf);
        tls_trash_buf = (float *)malloc(need * sizeof(float));
        tls_trash_sz = tls_trash_buf ? need : 0;
    }
    return tls_trash_buf;
}

size_t sdm_segment_process(sdm_context_t *sdm,
                            const float *in, float *out,
                            size_t count, size_t discard) {
    if (discard == 0)
        return sdm_process_block(sdm, in, out, count);

    /* Warmup phase: feed trellis_lat + discard input samples.
     * First trellis_lat fill the latency buffer (no output).
     * Next 'discard' samples produce output we throw away. */
    size_t warmup_in = (size_t)sdm->trellis_lat + discard;
    if (warmup_in > count) {
        float *trash = get_trash_buf(count);
        if (!trash) return 0;
        sdm_process_block(sdm, in, trash, count);
        return 0;
    }

    float *trash = get_trash_buf(discard);
    if (!trash) return 0;
    sdm_process_block(sdm, in, trash, warmup_in);

    /* Process remaining input directly into output */
    return sdm_process_block(sdm, in + warmup_in, out, count - warmup_in);
}
