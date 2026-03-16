/*
 * foo_dsd_trellis — Top-level per-channel processing orchestrator
 *
 * Phase 0: Scaffold — stub implementations.
 * Phase 4-6 will wire up FIR → gain → SDM → repack pipeline.
 */

#include "../include/engine.h"
#include "../include/ntf.h"
#include <stdlib.h>
#include <string.h>

/* ─── Path-adaptive SDM configuration table ───
 * Optimal NTF filter, limiter, candidates, and latency per rate conversion path.
 * Derived from comprehensive sweep (1,152 measurements, 2026-03-14).
 * Used when ntf_filter == NTF_AUTO and rate conversion is active. */

typedef struct {
    uint32_t        fs_in;
    uint32_t        fs_out;
    ntf_filter_id_t filter;
    double          state_limit;  /* 0.0 = disabled */
    int             cands;
    int             lat;
    int             depth;        /* trellis depth (0 = use global default) */
    float           fir_gain;     /* FIR output gain (attenuate to prevent SDM overload) */
} path_config_t;

/* Optimal per-path configuration derived from comprehensive NTF × limiter × gain sweep.
 * FIR upsampling of ±1.0 DSD produces peaks at ±2.24 due to broadband noise energy.
 * Multi-stage upsampling concentrates this noise in the low-frequency region,
 * overloading the SDM quantizer (output ±1.0). fir_gain attenuates the FIR output
 * to keep the signal within the SDM's linear operating range. */
static const path_config_t path_table[] = {
    /*                                            lim  cands lat  depth gain  */
    /* All paths use 0.708 (-3 dB) gain for uniform volume across rates */
    /* Same-rate re-encode (boxcar → SDM, sequential single-thread).
     * Lower cands/depth for real-time at high DSD rates.
     * CLANS_5: only stable NTF with crude boxcar input. */
    { DSD_RATE_64,  DSD_RATE_64,  NTF_CLANS_5, 0.0,  2,  256, 4, 0.708f },
    { DSD_RATE_128, DSD_RATE_128, NTF_CLANS_5, 0.0,  2,  256, 4, 0.708f },
    { DSD_RATE_256, DSD_RATE_256, NTF_CLANS_5, 0.0,  2,  512, 4, 0.708f },
    { DSD_RATE_512, DSD_RATE_512, NTF_CLANS_5, 0.0,  2,  512, 4, 0.708f },
    /* Upsample paths */
    { DSD_RATE_64,  DSD_RATE_128, NTF_SDM_4,   0.0,  2,  512, 4, 0.708f },
    { DSD_RATE_64,  DSD_RATE_256, NTF_CLANS_8, 0.0,  2,  512, 4, 0.708f },
    { DSD_RATE_64,  DSD_RATE_512, NTF_CLANS_6, 10.0, 2,  512, 4, 0.708f },
    { DSD_RATE_128, DSD_RATE_256, NTF_CLANS_8, 0.0,  2,  512, 4, 0.708f },
    { DSD_RATE_128, DSD_RATE_512, NTF_CLANS_8, 12.0, 2,  512, 4, 0.708f },
    { DSD_RATE_256, DSD_RATE_512, NTF_CLANS_8, 6.0,  2,  512, 4, 0.708f },
    /* Downsample paths */
    { DSD_RATE_128, DSD_RATE_64,  NTF_CLANS_4, 0.0,  32, 512, 0, 0.708f },
    { DSD_RATE_256, DSD_RATE_64,  NTF_CLANS_8, 0.0,  8,  512, 0, 0.708f },
    { DSD_RATE_256, DSD_RATE_128, NTF_CLANS_4, 0.0,  8,  512, 0, 0.708f },
    { DSD_RATE_512, DSD_RATE_64,  NTF_SDM_6,   0.0,  8,  512, 0, 0.708f },
    { DSD_RATE_512, DSD_RATE_128, NTF_SDM_4,  16.0, 16,  512, 0, 0.708f },
    { DSD_RATE_512, DSD_RATE_256, NTF_SDM_6,  16.0,  8,  512, 0, 0.708f },
};

#define PATH_TABLE_COUNT (sizeof(path_table) / sizeof(path_table[0]))

static const path_config_t *path_config_lookup(uint32_t fs_in, uint32_t fs_out) {
    for (int i = 0; i < (int)PATH_TABLE_COUNT; i++) {
        if (path_table[i].fs_in == fs_in && path_table[i].fs_out == fs_out)
            return &path_table[i];
    }
    return NULL;
}

int engine_channel_init(engine_channel_t *eng, int channel,
                        const dsd_config_t *cfg) {
    memset(eng, 0, sizeof(*eng));
    eng->channel = channel;

    uint32_t fs_out = cfg->fs_out ? cfg->fs_out : cfg->fs_in;
    /* passthrough is evaluated dynamically in engine_process_block,
     * not at init time, so FIR+SDM are always available for gain control */
    eng->passthrough = false;

    eng->sdm_mode = cfg->sdm_mode;
    eng->fir_gain = 1.0f;  /* default, may be overridden by path_table */

    /* DSD→PCM decimation: FIR only, no SDM */
    eng->fir_only = (fs_out < DSD_RATE_64 && cfg->fs_in >= DSD_RATE_64);

    if (eng->fir_only && !cfg->mute) {
        if (fir_chain_init(&eng->fir, cfg->fs_in, fs_out) != 0)
            return -1;
        return 0;
    }

    if (!cfg->mute) {
        /* Init FIR chain */
        if (fir_chain_init(&eng->fir, cfg->fs_in, fs_out) != 0)
            return -1;

        /* Path-adaptive lookup for rate conversion with NTF_AUTO */
        bool is_rate_conv = (cfg->fs_in != fs_out);
        const path_config_t *pc = NULL;
        if (cfg->ntf_filter == NTF_AUTO) {
            pc = path_config_lookup(cfg->fs_in, fs_out);
            if (pc)
                eng->fir_gain = pc->fir_gain;
        }

        /* Apply user FIR gain override.
         * Auto: use path_config gain (0.708 = -3 dB for all paths).
         * Explicit: replace path_config gain with user's choice. */
        if (cfg->fir_gain_db != FIR_GAIN_AUTO)
            eng->fir_gain = fir_gain_db_to_linear(cfg->fir_gain_db);

        /* Select NTF filter — PreCorr uses its own auto-select even when
         * path_config exists (path_config NTF is Trellis-optimized) */
        const ntf_filter_t *filter = NULL;
        if (cfg->ntf_filter == NTF_AUTO) {
            if (cfg->sdm_mode == SDM_MODE_PRECORR) {
                filter = ntf_auto_select_precorr(fs_out);
            } else if (pc) {
                filter = ntf_get_filter(pc->filter, fs_out);
            } else {
                filter = ntf_auto_select(fs_out);
            }
        } else {
            filter = ntf_get_filter((ntf_filter_id_t)cfg->ntf_filter, fs_out);
        }

        if (!filter)
            return -1;

        /* Init SDM with path-adaptive or user-configured parameters */
        /* Resolve state limiter: user override > path_config > default */
        double resolved_limit = 0.0;
        if (cfg->state_limit >= 0.0f)
            resolved_limit = (double)cfg->state_limit;  /* user set explicit value */
        else if (pc && pc->state_limit > 0.0)
            resolved_limit = pc->state_limit;  /* path_config default */
        else if (is_rate_conv && cfg->sdm_mode == SDM_MODE_PRECORR)
            resolved_limit = 12.0;  /* PreCorr default for rate conversion */

        if (cfg->sdm_mode == SDM_MODE_PRECORR) {
            if (precorr_context_init(&eng->precorr, filter) != 0)
                return -1;
            if (resolved_limit > 0.0)
                eng->precorr.state_limit = (float)resolved_limit;
        } else {
            int cands = cfg->trellis_cands;
            int lat   = cfg->trellis_lat;
            if (sdm_context_init(&eng->sdm, filter,
                                 cfg->trellis_depth,
                                 cands, lat) != 0)
                return -1;
            if (resolved_limit > 0.0)
                eng->sdm.state_limit = resolved_limit;
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

    /* Same-rate DSD: always re-encode through SDM (boxcar → gain → SDM).
     * Bypass is handled at the rate_map level ("-" = RATE_OUT_BYPASS).
     * When user explicitly selects same-rate output, they want SDM processing. */
    {
        uint32_t fs_out = cfg->fs_out ? cfg->fs_out : cfg->fs_in;
        if (cfg->fs_in == fs_out) {
            /* DSD-Wide: boxcar smooth ±1.0 → multi-bit, apply gain, re-encode via SDM */
            if (eng->fir_buf_sz < count * sizeof(float)) {
                free(eng->fir_buf);
                eng->fir_buf = (float *)malloc(count * sizeof(float));
                eng->fir_buf_sz = eng->fir_buf ? count * sizeof(float) : 0;
            }
            if (!eng->fir_buf)
                return 0;
            boxcar_t *bc = &eng->boxcar;
            const float inv_n = 1.0f / BOXCAR_TAPS;
            for (size_t i = 0; i < count; i++) {
                float s = in[i] >= 0.0f ? 1.0f : -1.0f;
                bc->sum -= bc->ring[bc->pos];
                bc->ring[bc->pos] = s;
                bc->sum += s;
                bc->pos = (bc->pos + 1) % BOXCAR_TAPS;
                eng->fir_buf[i] = bc->sum * inv_n * cfg->gain;
            }
            /* Re-encode via SDM */
            size_t sdm_out;
            if (eng->sdm_mode == SDM_MODE_PRECORR)
                sdm_out = precorr_process_block(&eng->precorr, eng->fir_buf, out, count);
            else
                sdm_out = sdm_process_block(&eng->sdm, eng->fir_buf, out, count);
            if (eng->ml_filter)
                onnx_filter_process(eng->ml_filter, out, sdm_out);
            return sdm_out;
        }
    }

    if (eng->fir_only) {
        /* DSD→PCM decimation: FIR + gain only, no SDM */
        float *fir_out;
        size_t fir_count = engine_process_fir_gain(eng, in, count, cfg, &fir_out);
        memcpy(out, fir_out, fir_count * sizeof(float));
        return fir_count;
    }

    /* FIR rate conversion */
    uint32_t fs_out = cfg->fs_out ? cfg->fs_out : cfg->fs_in;
    size_t fir_out_count;
    if (fs_out >= cfg->fs_in)
        fir_out_count = count * (fs_out / cfg->fs_in);
    else
        fir_out_count = count / (cfg->fs_in / fs_out);

    /* For multi-stage downsample, fir_chain_process uses fir_buf as a
     * ping-pong buffer — intermediate results can be up to count/2. */
    size_t buf_need = fir_out_count;
    if (fs_out < cfg->fs_in && count / 2 > buf_need)
        buf_need = count / 2;

    /* Ensure intermediate buffer */
    if (eng->fir_buf_sz < buf_need * sizeof(float)) {
        free(eng->fir_buf);
        eng->fir_buf = (float *)malloc(buf_need * sizeof(float));
        eng->fir_buf_sz = eng->fir_buf ? buf_need * sizeof(float) : 0;
    }
    if (!eng->fir_buf)
        return 0;

    size_t fir_out = fir_chain_process(&eng->fir, in, eng->fir_buf, count);

    /* Apply FIR output attenuation (path-adaptive SDM overload prevention)
     * and user gain in a single pass */
    float combined_gain = eng->fir_gain * cfg->gain;
    if (combined_gain != 1.0f) {
        for (size_t i = 0; i < fir_out; i++)
            eng->fir_buf[i] *= combined_gain;
    }

    /* SDM (PreCorr or Trellis) */
    size_t sdm_out;
    if (eng->sdm_mode == SDM_MODE_PRECORR)
        sdm_out = precorr_process_block(&eng->precorr, eng->fir_buf, out, fir_out);
    else
        sdm_out = sdm_process_block(&eng->sdm, eng->fir_buf, out, fir_out);
    if (eng->ml_filter)
        onnx_filter_process(eng->ml_filter, out, sdm_out);
    return sdm_out;
}

void engine_channel_reset(engine_channel_t *eng) {
    fir_chain_reset(&eng->fir);
    memset(&eng->boxcar, 0, sizeof(eng->boxcar));
    /* PreCorr must be reset (crashes with stale state).
     * Trellis SDM state is preserved to prevent startup pop —
     * integrators keep their values so playback resumes smoothly. */
    if (!eng->fir_only && eng->sdm_mode == SDM_MODE_PRECORR)
        precorr_context_reset(&eng->precorr);
    if (eng->ml_filter)
        onnx_filter_reset(eng->ml_filter);
}

void engine_channel_free(engine_channel_t *eng) {
    fir_chain_free(&eng->fir);
    if (!eng->fir_only) {
        if (eng->sdm_mode == SDM_MODE_PRECORR)
            precorr_context_free(&eng->precorr);
        else
            sdm_context_free(&eng->sdm);
    }
    free(eng->fir_buf);
    eng->fir_buf = NULL;
    eng->fir_buf_sz = 0;
    if (eng->ml_filter) {
        onnx_filter_free(eng->ml_filter);
        eng->ml_filter = NULL;
    }
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

    /* For multi-stage downsample, fir_chain_process uses out as a ping-pong
     * buffer with scratch. The first intermediate result written to out can
     * be up to count/2 samples, so we must allocate for that, not just the
     * final output size. */
    size_t buf_need = fir_out_count;
    if (fs_out < cfg->fs_in && count / 2 > buf_need)
        buf_need = count / 2;

    if (eng->fir_buf_sz < buf_need * sizeof(float)) {
        free(eng->fir_buf);
        eng->fir_buf = (float *)malloc(buf_need * sizeof(float));
        eng->fir_buf_sz = buf_need * sizeof(float);
    }

    size_t fir_count;
    uint32_t fs_out_actual = cfg->fs_out ? cfg->fs_out : cfg->fs_in;
    if (cfg->fs_in == fs_out_actual) {
        /* Same-rate: boxcar smooth ±1.0 → multi-bit (same as engine_process_block) */
        boxcar_t *bc = &eng->boxcar;
        const float inv_n = 1.0f / BOXCAR_TAPS;
        for (size_t i = 0; i < count; i++) {
            float s = in[i] >= 0.0f ? 1.0f : -1.0f;
            bc->sum -= bc->ring[bc->pos];
            bc->ring[bc->pos] = s;
            bc->sum += s;
            bc->pos = (bc->pos + 1) % BOXCAR_TAPS;
            eng->fir_buf[i] = bc->sum * inv_n;
        }
        fir_count = count;
    } else {
        fir_count = fir_chain_process(&eng->fir, in, eng->fir_buf, count);
    }

    float combined_gain = eng->fir_gain * cfg->gain;
    if (combined_gain != 1.0f) {
        for (size_t i = 0; i < fir_count; i++)
            eng->fir_buf[i] *= combined_gain;
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

    float *trash = get_trash_buf(warmup_in);  /* needs trellis_lat + discard */
    if (!trash) return 0;
    sdm_process_block(sdm, in, trash, warmup_in);

    /* Process remaining input directly into output */
    return sdm_process_block(sdm, in + warmup_in, out, count - warmup_in);
}

/* ─── Path info query for UI display ─── */

/* Count FIR 2x stages needed for a given rate ratio */
static int count_fir_stages(uint32_t fs_in, uint32_t fs_out) {
    if (fs_in == fs_out || fs_in == 0 || fs_out == 0)
        return 0;
    uint32_t ratio = (fs_out >= fs_in) ? (fs_out / fs_in) : (fs_in / fs_out);
    if (ratio == 0) return 0;
    int stages = 0;
    while (ratio > 1) {
        ratio >>= 1;
        stages++;
    }
    return stages;
}

int engine_get_path_info(uint32_t fs_in, uint32_t fs_out,
                          int ntf_override, int sdm_mode,
                          const dsd_config_t *cfg,
                          engine_path_info_t *info) {
    memset(info, 0, sizeof(*info));

    if (fs_in == 0 || fs_out == 0)
        return -1;

    info->fir_stages = count_fir_stages(fs_in, fs_out);
    info->depth = cfg->trellis_depth;

    /* DSD→PCM: FIR decimation only */
    if (fs_out < DSD_RATE_64 && fs_in >= DSD_RATE_64) {
        info->fir_only = true;
        info->ntf_filter = NTF_AUTO;
        info->cands = 0;
        info->lat = 0;
        return 0;
    }

    info->fir_only = false;

    /* Path-adaptive lookup (includes same-rate re-encode entries) */
    const path_config_t *pc = NULL;
    if (ntf_override == NTF_AUTO && sdm_mode == SDM_MODE_TRELLIS) {
        pc = path_config_lookup(fs_in, fs_out);
    }

    if (pc) {
        info->ntf_filter = (int)pc->filter;
        info->cands = pc->cands;
        info->lat = pc->lat;
        info->depth = pc->depth ? pc->depth : cfg->trellis_depth;
        info->state_limit = pc->state_limit;
        info->fir_gain = pc->fir_gain;
    } else {
        /* Auto-select or user override */
        info->ntf_filter = ntf_override; /* keep NTF_AUTO or user value */
        info->cands = cfg->trellis_cands;
        info->lat = cfg->trellis_lat;
        info->fir_gain = 1.0f;
    }

    return 0;
}
