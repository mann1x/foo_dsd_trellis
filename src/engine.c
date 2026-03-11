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

    if (!eng->passthrough && !cfg->mute) {
        /* Init FIR chain */
        if (fir_chain_init(&eng->fir, cfg->fs_in, fs_out) != 0)
            return -1;

        /* Init SDM */
        const ntf_filter_t *filter = NULL;
        if (cfg->ntf_filter == NTF_AUTO)
            filter = ntf_auto_select(fs_out);
        else
            filter = ntf_get_filter((ntf_filter_id_t)cfg->ntf_filter, fs_out);

        if (!filter)
            return -1;

        if (sdm_context_init(&eng->sdm, filter,
                             cfg->trellis_depth,
                             cfg->trellis_cands,
                             cfg->trellis_lat) != 0)
            return -1;
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

    /* Trellis SDM */
    size_t sdm_out = sdm_process_block(&eng->sdm, eng->fir_buf, out, fir_out);
    return sdm_out;
}

void engine_channel_reset(engine_channel_t *eng) {
    fir_chain_reset(&eng->fir);
    sdm_context_reset(&eng->sdm);
}

void engine_channel_free(engine_channel_t *eng) {
    fir_chain_free(&eng->fir);
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
