/*
 * foo_dsd_trellis — Configuration serialization and validation
 *
 * Versioned binary format: [uint32 version] [fields in order]
 * Version 0: legacy raw memcpy of dsd_config_t (backward compat)
 * Version 1: field-by-field serialization with output_format field
 */

#include <string.h>
#include "../include/dsd_types.h"

/* Forward declaration */
void config_validate(dsd_config_t *cfg);

/* ─── Helpers for reading/writing fields ─── */

static size_t write_u32(uint8_t *buf, size_t pos, uint32_t val) {
    memcpy(buf + pos, &val, 4);
    return pos + 4;
}

static size_t write_i32(uint8_t *buf, size_t pos, int32_t val) {
    memcpy(buf + pos, &val, 4);
    return pos + 4;
}

static size_t write_f32(uint8_t *buf, size_t pos, float val) {
    memcpy(buf + pos, &val, 4);
    return pos + 4;
}

static size_t write_u8(uint8_t *buf, size_t pos, uint8_t val) {
    buf[pos] = val;
    return pos + 1;
}

static size_t read_u32(const uint8_t *buf, size_t pos, uint32_t *val) {
    memcpy(val, buf + pos, 4);
    return pos + 4;
}

static size_t read_i32(const uint8_t *buf, size_t pos, int32_t *val) {
    memcpy(val, buf + pos, 4);
    return pos + 4;
}

static size_t read_f32(const uint8_t *buf, size_t pos, float *val) {
    memcpy(val, buf + pos, 4);
    return pos + 4;
}

static size_t write_u16(uint8_t *buf, size_t pos, uint16_t val) {
    memcpy(buf + pos, &val, 2);
    return pos + 2;
}

static size_t read_u8(const uint8_t *buf, size_t pos, uint8_t *val) {
    *val = buf[pos];
    return pos + 1;
}

static size_t read_u16(const uint8_t *buf, size_t pos, uint16_t *val) {
    memcpy(val, buf + pos, 2);
    return pos + 2;
}

/* ─── Version field layout ─── */
/* All size constants for v8-v15 use RATE_MAP_COUNT_V15 (12) — the array size
 * that was used when those versions were written. v16+ uses RATE_MAP_COUNT (20). */
#define CONFIG_V1_SIZE 49
#define CONFIG_V2_SIZE 50
#define CONFIG_V3_SIZE 53
#define CONFIG_V4_SIZE 55
#define CONFIG_V5_SIZE 56
#define CONFIG_V6_SIZE 57
#define CONFIG_V7_SIZE 58
#define CONFIG_V8_SIZE  (CONFIG_V7_SIZE + RATE_MAP_COUNT_V15)            /* 70 */
#define CONFIG_V9_SIZE  (CONFIG_V8_SIZE + RATE_MAP_COUNT_V15)            /* 82 */
#define CONFIG_V10_SIZE (CONFIG_V9_SIZE + 2)                             /* 84 */
/* v11: antipop(1) + 4 × 12 for rate_sdm/cands/depth/ml */
#define CONFIG_V11_SIZE (CONFIG_V10_SIZE + 1 + 4 * RATE_MAP_COUNT_V15)   /* 133 */
/* v12: rate_limiter(12) + fir_gain_db(1) */
#define CONFIG_V12_SIZE (CONFIG_V11_SIZE + RATE_MAP_COUNT_V15 + 1)       /* 146 */
/* v13: gpu_enabled(1) + gpu_backend(1) + rate_gpu(12) */
#define CONFIG_V13_SIZE (CONFIG_V12_SIZE + 1 + 1 + RATE_MAP_COUNT_V15)   /* 160 */
/* v14: rate_gpu_sdm(12) — legacy, read and discard */
#define CONFIG_V14_SIZE (CONFIG_V13_SIZE + RATE_MAP_COUNT_V15)           /* 172 */
/* v15: rate_fir_mode(12) + rate_lat(24 = 12 × int16_t) */
#define CONFIG_V15_SIZE (CONFIG_V14_SIZE + RATE_MAP_COUNT_V15 + RATE_MAP_COUNT_V15 * 2) /* 208 */

/* v16: All per-rate arrays expanded to RATE_MAP_COUNT (20).
 * Written as: v15 core fields (using 20-element arrays) + PCM encoding fields.
 * v16 core = v7(58) + rate_map(20) + rate_ntf(20) + ml(2) + antipop(1)
 *   + rate_sdm(20) + rate_cands(20) + rate_depth(20) + rate_ml(20)
 *   + rate_limiter(20) + fir_gain_db(1)
 *   + gpu_enabled(1) + gpu_backend(1) + rate_gpu(20)
 *   + rate_gpu_sdm_legacy(20)
 *   + rate_fir_mode(20) + rate_lat(40)
 *   + pcm_bit_depth(1) + pcm_dither(1) + resample_engine(1) + soxr_quality(1)
 *   + rate_pcm_bits(20) + rate_pcm_dither(20) */
#define CONFIG_V16_SIZE (CONFIG_V7_SIZE \
    + RATE_MAP_COUNT             /* rate_map */ \
    + RATE_MAP_COUNT             /* rate_ntf */ \
    + 2                          /* ml_enabled + ml_ep */ \
    + 1                          /* antipop */ \
    + 4 * RATE_MAP_COUNT         /* rate_sdm/cands/depth/ml */ \
    + RATE_MAP_COUNT             /* rate_limiter */ \
    + 1                          /* fir_gain_db */ \
    + 1 + 1                      /* gpu_enabled + gpu_backend */ \
    + RATE_MAP_COUNT             /* rate_gpu */ \
    + RATE_MAP_COUNT             /* rate_gpu_sdm legacy */ \
    + RATE_MAP_COUNT             /* rate_fir_mode */ \
    + RATE_MAP_COUNT * 2         /* rate_lat (int16_t) */ \
    + 4                          /* pcm_bit_depth + pcm_dither + resample_engine + soxr_quality */ \
    + 2 * RATE_MAP_COUNT         /* rate_pcm_bits + rate_pcm_dither */ \
    )

/* v17: v16 + rate_parallel(20) + gpu_sdm_enabled(1) + rate_fir_prec(20) */
#define CONFIG_V17_SIZE (CONFIG_V16_SIZE \
    + RATE_MAP_COUNT             /* rate_parallel */ \
    + 1                          /* gpu_sdm_enabled */ \
    + RATE_MAP_COUNT             /* rate_fir_prec */ \
    )

/* Serialise config to a byte buffer. Returns bytes written. */
size_t config_serialize(const dsd_config_t *cfg, uint8_t *buf, size_t buf_size) {
    if (buf_size < CONFIG_V17_SIZE)
        return 0;

    size_t pos = 0;
    pos = write_u32(buf, pos, DSD_CONFIG_VERSION);
    pos = write_u32(buf, pos, cfg->fs_in);
    pos = write_u32(buf, pos, 0);  /* fs_out: superseded by rate_map, write 0 for compat */
    pos = write_f32(buf, pos, cfg->gain);
    pos = write_u8(buf, pos, cfg->mute ? 1 : 0);
    pos = write_i32(buf, pos, (int32_t)cfg->trellis_depth);
    pos = write_i32(buf, pos, (int32_t)cfg->trellis_cands);
    pos = write_i32(buf, pos, (int32_t)cfg->trellis_lat);
    pos = write_i32(buf, pos, (int32_t)cfg->ntf_filter);
    pos = write_i32(buf, pos, (int32_t)cfg->thread_count);
    pos = write_u32(buf, pos, (uint32_t)cfg->affinity_mask);
    pos = write_i32(buf, pos, (int32_t)cfg->format);
    pos = write_i32(buf, pos, (int32_t)cfg->output_format);
    pos = write_u8(buf, pos, cfg->debug_log ? 1 : 0);
    pos = write_u8(buf, pos, (uint8_t)cfg->smt_mode);
    pos = write_u8(buf, pos, (uint8_t)cfg->ccd_mode);
    pos = write_u8(buf, pos, (uint8_t)cfg->ecore_mode);
    pos = write_u16(buf, pos, cfg->api_port);
    pos = write_u8(buf, pos, (uint8_t)cfg->sdm_mode);
    pos = write_u8(buf, pos, 0);  /* fir_mode placeholder (v6 compat) */
    pos = write_u8(buf, pos, 0);  /* proc_mode placeholder (v7 compat) */
    /* v8: rate_map (20 elements in v16) */
    memcpy(buf + pos, cfg->rate_map, RATE_MAP_COUNT);
    pos += RATE_MAP_COUNT;
    /* v9: rate_ntf */
    memcpy(buf + pos, cfg->rate_ntf, RATE_MAP_COUNT);
    pos += RATE_MAP_COUNT;
    /* v10: ML filter settings + antipop */
    pos = write_u8(buf, pos, cfg->ml_enabled ? 1 : 0);
    pos = write_u8(buf, pos, (uint8_t)cfg->ml_ep);
    pos = write_u8(buf, pos, cfg->antipop ? 1 : 0);
    /* v11: per-rate SDM mode, cands, depth, ML */
    memcpy(buf + pos, cfg->rate_sdm, RATE_MAP_COUNT);
    pos += RATE_MAP_COUNT;
    memcpy(buf + pos, cfg->rate_cands, RATE_MAP_COUNT);
    pos += RATE_MAP_COUNT;
    memcpy(buf + pos, cfg->rate_depth, RATE_MAP_COUNT);
    pos += RATE_MAP_COUNT;
    memcpy(buf + pos, cfg->rate_ml, RATE_MAP_COUNT);
    pos += RATE_MAP_COUNT;
    /* v12: per-rate state limiter + global FIR gain */
    memcpy(buf + pos, cfg->rate_limiter, RATE_MAP_COUNT);
    pos += RATE_MAP_COUNT;
    pos = write_u8(buf, pos, (uint8_t)(int8_t)cfg->fir_gain_db);
    /* v13: GPU compute settings */
    pos = write_u8(buf, pos, cfg->gpu_enabled ? 1 : 0);
    pos = write_u8(buf, pos, (uint8_t)cfg->gpu_backend);
    memcpy(buf + pos, cfg->rate_gpu, RATE_MAP_COUNT);
    pos += RATE_MAP_COUNT;

    /* v14: legacy rate_gpu_sdm — write defaults for forward compat */
    memset(buf + pos, 0xFF, RATE_MAP_COUNT);
    pos += RATE_MAP_COUNT;

    /* v15: per-rate pre-SDM filter mode + trellis latency */
    memcpy(buf + pos, cfg->rate_fir_mode, RATE_MAP_COUNT);
    pos += RATE_MAP_COUNT;
    memcpy(buf + pos, cfg->rate_lat, RATE_MAP_COUNT * sizeof(int16_t));
    pos += RATE_MAP_COUNT * sizeof(int16_t);

    /* v16: PCM encoding + resampler */
    pos = write_u8(buf, pos, (uint8_t)(int8_t)cfg->pcm_bit_depth);
    pos = write_u8(buf, pos, (uint8_t)(int8_t)cfg->pcm_dither);
    pos = write_u8(buf, pos, (uint8_t)(int8_t)cfg->resample_engine);
    pos = write_u8(buf, pos, (uint8_t)(int8_t)cfg->soxr_quality);
    memcpy(buf + pos, cfg->rate_pcm_bits, RATE_MAP_COUNT);
    pos += RATE_MAP_COUNT;
    memcpy(buf + pos, cfg->rate_pcm_dither, RATE_MAP_COUNT);
    pos += RATE_MAP_COUNT;
    memcpy(buf + pos, cfg->rate_parallel, RATE_MAP_COUNT);
    pos += RATE_MAP_COUNT;
    pos = write_u8(buf, pos, cfg->gpu_sdm_enabled ? 1 : 0);

    /* v17: per-rate FIR precision */
    memcpy(buf + pos, cfg->rate_fir_prec, RATE_MAP_COUNT);
    pos += RATE_MAP_COUNT;

    return pos;
}

/* Deserialise config from a byte buffer. Returns 0 on success. */
int config_deserialize(dsd_config_t *cfg, const uint8_t *buf, size_t buf_size) {
    dsd_config_defaults(cfg);

    if (buf_size < 4)
        return -1;

    /* Read version */
    uint32_t version;
    read_u32(buf, 0, &version);

    /* Determine the per-rate array size for this version.
     * v8-v15 stored 12-element arrays; v16+ stores 20-element arrays. */
    int rate_count = (version >= 16) ? RATE_MAP_COUNT : RATE_MAP_COUNT_V15;

    if ((version >= 1 && version <= DSD_CONFIG_VERSION) && buf_size >= CONFIG_V1_SIZE) {
        /* Version 1+: field-by-field */
        size_t pos = 4;
        uint32_t fs_out_raw = 0;
        pos = read_u32(buf, pos, &cfg->fs_in);
        pos = read_u32(buf, pos, &fs_out_raw);
        pos = read_f32(buf, pos, &cfg->gain);
        uint8_t mute_byte;
        pos = read_u8(buf, pos, &mute_byte);
        cfg->mute = mute_byte != 0;
        int32_t i32;
        pos = read_i32(buf, pos, &i32); cfg->trellis_depth = (int)i32;
        pos = read_i32(buf, pos, &i32); cfg->trellis_cands = (int)i32;
        pos = read_i32(buf, pos, &i32); cfg->trellis_lat = (int)i32;
        pos = read_i32(buf, pos, &i32); cfg->ntf_filter = (int)i32;
        pos = read_i32(buf, pos, &i32); cfg->thread_count = (int)i32;
        uint32_t u32;
        pos = read_u32(buf, pos, &u32); cfg->affinity_mask = (DWORD)u32;
        pos = read_i32(buf, pos, &i32); cfg->format = (int)i32;
        pos = read_i32(buf, pos, &i32); cfg->output_format = (int)i32;
        /* Version 2 adds debug_log */
        if (version >= 2 && buf_size >= CONFIG_V2_SIZE) {
            uint8_t log_byte;
            pos = read_u8(buf, pos, &log_byte);
            cfg->debug_log = log_byte != 0;
        }
        /* Version 3 adds smt_mode, ccd_mode, ecore_mode */
        if (version >= 3 && buf_size >= CONFIG_V3_SIZE) {
            uint8_t u8;
            pos = read_u8(buf, pos, &u8); cfg->smt_mode = (int)u8;
            pos = read_u8(buf, pos, &u8); cfg->ccd_mode = (int)u8;
            pos = read_u8(buf, pos, &u8); cfg->ecore_mode = (int)u8;
        }
        /* Version 4 adds api_port */
        if (version >= 4 && buf_size >= CONFIG_V4_SIZE) {
            pos = read_u16(buf, pos, &cfg->api_port);
        }
        /* Version 5 adds sdm_mode */
        uint8_t sdm_mode_raw = 0;
        if (version >= 5 && buf_size >= CONFIG_V5_SIZE) {
            uint8_t u8;
            pos = read_u8(buf, pos, &u8); cfg->sdm_mode = (int)u8;
            sdm_mode_raw = u8;
        } else {
            /* Pre-v5 configs default to Trellis to preserve existing behavior */
            cfg->sdm_mode = SDM_MODE_TRELLIS;
        }
        /* Version 6 had fir_mode (removed — read and discard) */
        if (version >= 6 && buf_size >= CONFIG_V6_SIZE) {
            uint8_t u8;
            pos = read_u8(buf, pos, &u8); /* discard fir_mode */
        }
        /* Version 7 had proc_mode (superseded by rate_map) */
        uint8_t proc_mode_raw = 2; /* default: DSD recode */
        if (version >= 7 && buf_size >= CONFIG_V7_SIZE) {
            uint8_t u8;
            pos = read_u8(buf, pos, &u8);
            proc_mode_raw = u8;
        }
        /* Version 8+ adds per-rate arrays */
        if (version >= 8 && buf_size >= CONFIG_V8_SIZE) {
            /* Read rate_map: rate_count elements into first rate_count positions */
            memcpy(cfg->rate_map, buf + pos, rate_count);
            pos += rate_count;
            /* Version 9 adds rate_ntf */
            if (version >= 9) {
                memcpy(cfg->rate_ntf, buf + pos, rate_count);
                pos += rate_count;
            }
            /* Version 10 adds ml_enabled, ml_ep, antipop */
            if (version >= 10) {
                uint8_t u8;
                pos = read_u8(buf, pos, &u8); cfg->ml_enabled = (u8 != 0);
                pos = read_u8(buf, pos, &u8); cfg->ml_ep = (int)u8;
                /* antipop: read if available */
                if (pos < buf_size) {
                    pos = read_u8(buf, pos, &u8); cfg->antipop = (u8 != 0);
                }
            }
            /* Version 11 adds per-rate SDM, cands, depth, ML */
            if (version >= 11) {
                memcpy(cfg->rate_sdm, buf + pos, rate_count);
                pos += rate_count;
                memcpy(cfg->rate_cands, buf + pos, rate_count);
                pos += rate_count;
                memcpy(cfg->rate_depth, buf + pos, rate_count);
                pos += rate_count;
                memcpy(cfg->rate_ml, buf + pos, rate_count);
                pos += rate_count;
            }
            /* Version 12 adds per-rate state limiter + global FIR gain */
            if (version >= 12) {
                memcpy(cfg->rate_limiter, buf + pos, rate_count);
                pos += rate_count;
                uint8_t u8;
                pos = read_u8(buf, pos, &u8);
                cfg->fir_gain_db = (int8_t)u8;
            }
            /* Version 13 adds GPU compute settings */
            if (version >= 13) {
                uint8_t u8;
                pos = read_u8(buf, pos, &u8); cfg->gpu_enabled = (u8 != 0);
                pos = read_u8(buf, pos, &u8); cfg->gpu_backend = (int)u8;
                memcpy(cfg->rate_gpu, buf + pos, rate_count);
                pos += rate_count;
            }
            /* Version 14 had per-rate GPU SDM toggle — read and discard */
            if (version >= 14) {
                pos += rate_count; /* skip legacy rate_gpu_sdm */
            }
            /* Version 15 adds per-rate FIR mode + trellis latency */
            if (version >= 15) {
                memcpy(cfg->rate_fir_mode, buf + pos, rate_count);
                pos += rate_count;
                memcpy(cfg->rate_lat, buf + pos, rate_count * sizeof(int16_t));
                pos += rate_count * sizeof(int16_t);
            }
            /* Version 16+ adds PCM encoding + resampler */
            if (version >= 16) {
                uint8_t u8;
                pos = read_u8(buf, pos, &u8); cfg->pcm_bit_depth = (int8_t)u8;
                pos = read_u8(buf, pos, &u8); cfg->pcm_dither = (int8_t)u8;
                pos = read_u8(buf, pos, &u8); cfg->resample_engine = (int8_t)u8;
                pos = read_u8(buf, pos, &u8); cfg->soxr_quality = (int8_t)u8;
                memcpy(cfg->rate_pcm_bits, buf + pos, RATE_MAP_COUNT);
                pos += RATE_MAP_COUNT;
                memcpy(cfg->rate_pcm_dither, buf + pos, RATE_MAP_COUNT);
                pos += RATE_MAP_COUNT;
                /* rate_parallel added later in v16, may not be present */
                if (pos + RATE_MAP_COUNT <= buf_size) {
                    memcpy(cfg->rate_parallel, buf + pos, RATE_MAP_COUNT);
                    pos += RATE_MAP_COUNT;
                }
                if (pos + 1 <= buf_size) {
                    uint8_t u8;
                    pos = read_u8(buf, pos, &u8);
                    cfg->gpu_sdm_enabled = (u8 != 0);
                }
            }
            /* Version 17 adds per-rate FIR precision */
            if (version >= 17) {
                if (pos + RATE_MAP_COUNT <= buf_size) {
                    memcpy(cfg->rate_fir_prec, buf + pos, RATE_MAP_COUNT);
                    pos += RATE_MAP_COUNT;
                }
            }
        } else {
            /* Migrate from v1-v7: use fs_out + proc_mode to populate rate_map */
            uint8_t out_idx = dsd_to_rate_out(fs_out_raw);
            if (proc_mode_raw == 2 || version < 7) {
                /* DSD recode mode or pre-v7: apply to DSD inputs */
                if (out_idx != RATE_OUT_BYPASS) {
                    /* All DSD/44 inputs → specified output rate */
                    for (int i = RATEIDX_DSD64; i <= RATEIDX_DSD512; i++)
                        cfg->rate_map[i] = out_idx;
                } else {
                    /* fs_out=0 means "same as input" = re-encode at same rate */
                    cfg->rate_map[RATEIDX_DSD64]  = RATE_OUT_DSD64;
                    cfg->rate_map[RATEIDX_DSD128] = RATE_OUT_DSD128;
                    cfg->rate_map[RATEIDX_DSD256] = RATE_OUT_DSD256;
                    cfg->rate_map[RATEIDX_DSD512] = RATE_OUT_DSD512;
                }
            }
            if (proc_mode_raw == 1) {
                /* PCM→DSD mode: apply to 44100-family PCM inputs */
                if (out_idx != RATE_OUT_BYPASS) {
                    cfg->rate_map[RATEIDX_44100]  = out_idx;
                    cfg->rate_map[RATEIDX_88200]  = out_idx;
                    cfg->rate_map[RATEIDX_176400] = out_idx;
                    cfg->rate_map[RATEIDX_352800] = out_idx;
                }
            }
            /* proc_mode_raw == 0 (bypass): rate_map stays all zeros */
        }
        (void)pos;
        (void)sdm_mode_raw;
        cfg->fs_out = 0; /* runtime field, set from rate_map */
        config_validate(cfg);
        return 0;
    }

    /* Unknown format — keep defaults */
    return -1;
}

/* Validate config ranges and clamp to valid values. */
void config_validate(dsd_config_t *cfg) {
    if (cfg->gain < 0.0f) cfg->gain = 0.0f;
    if (cfg->gain > 1.0f) cfg->gain = 1.0f;

    /* Trellis depth must be power of 2 in [8, 32].
     * 8-bit minimum for better path dedup granularity (both CPU and GPU). */
    if (cfg->trellis_depth < 8)  cfg->trellis_depth = 8;
    if (cfg->trellis_depth > 32) cfg->trellis_depth = 32;
    /* Snap to nearest power of 2 */
    if (cfg->trellis_depth <= 8)       cfg->trellis_depth = 8;
    else if (cfg->trellis_depth <= 16) cfg->trellis_depth = 16;
    else                               cfg->trellis_depth = 32;

    if (cfg->trellis_cands < 4)  cfg->trellis_cands = 4;
    if (cfg->trellis_cands > 32) cfg->trellis_cands = 32;

    if (cfg->trellis_lat < 16)   cfg->trellis_lat = 16;
    if (cfg->trellis_lat > 2048) cfg->trellis_lat = 2048;

    if (cfg->thread_count < 0) cfg->thread_count = 0;

    /* Validate input format */
    if (cfg->format < FORMAT_AUTO || cfg->format > FORMAT_NATIVE)
        cfg->format = FORMAT_AUTO;

    /* Validate output format — always DoP (PCM output handled via rate_map) */
    cfg->output_format = OUTPUT_DOP;

    /* Validate SDM mode */
    if (cfg->sdm_mode != SDM_MODE_PRECORR && cfg->sdm_mode != SDM_MODE_TRELLIS)
        cfg->sdm_mode = SDM_MODE_PRECORR;

    /* Validate rate map entries */
    for (int i = 0; i < RATE_MAP_COUNT; i++) {
        if (cfg->rate_map[i] >= RATE_OUT_COUNT)
            cfg->rate_map[i] = RATE_OUT_BYPASS;
        if (!rate_map_valid_output(i, cfg->rate_map[i]))
            cfg->rate_map[i] = RATE_OUT_BYPASS;
    }

    /* Validate ML settings */
    if (cfg->ml_ep < 0 || cfg->ml_ep > 2)
        cfg->ml_ep = 2;  /* ML_EP_AUTO */

    /* Validate per-rate NTF overrides */
    for (int i = 0; i < RATE_MAP_COUNT; i++) {
        if (cfg->rate_ntf[i] != NTF_AUTO &&
            (cfg->rate_ntf[i] < 0 || cfg->rate_ntf[i] >= NTF_COUNT))
            cfg->rate_ntf[i] = (int8_t)NTF_AUTO;
    }

    /* Validate FIR gain: Auto or 0 to -12 dB */
    if (cfg->fir_gain_db != FIR_GAIN_AUTO) {
        if (cfg->fir_gain_db > 0) cfg->fir_gain_db = 0;
        if (cfg->fir_gain_db < -12) cfg->fir_gain_db = -12;
    }

    /* Validate GPU settings */
    if (cfg->gpu_backend < 0 || cfg->gpu_backend > 3)
        cfg->gpu_backend = 3;  /* GPU_BACKEND_AUTO */

    /* Validate PCM encoding */
    if (cfg->pcm_bit_depth < PCM_BIT_AUTO || cfg->pcm_bit_depth > PCM_BIT_FLOAT)
        cfg->pcm_bit_depth = PCM_BIT_AUTO;
    if (cfg->pcm_dither < PCM_DITHER_AUTO || cfg->pcm_dither > PCM_DITHER_SHAPED)
        cfg->pcm_dither = PCM_DITHER_AUTO;
    if (cfg->resample_engine < RESAMPLE_AUTO || cfg->resample_engine > RESAMPLE_SOXR)
        cfg->resample_engine = RESAMPLE_AUTO;
    if (cfg->soxr_quality < SOXR_QUALITY_MQ || cfg->soxr_quality > SOXR_QUALITY_VHQ)
        cfg->soxr_quality = SOXR_QUALITY_HQ;

    /* Validate per-rate FIR precision */
    for (int i = 0; i < RATE_MAP_COUNT; i++) {
        if (cfg->rate_fir_prec[i] < FIR_PREC_AUTO || cfg->rate_fir_prec[i] > FIR_PREC_FP64)
            cfg->rate_fir_prec[i] = (int8_t)FIR_PREC_AUTO;
    }
}
