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
/* 4  version
 * 4  fs_in
 * 4  fs_out
 * 4  gain (float)
 * 1  mute (bool)
 * 4  trellis_depth
 * 4  trellis_cands
 * 4  trellis_lat
 * 4  ntf_filter
 * 4  thread_count
 * 4  affinity_mask
 * 4  format
 * 4  output_format
 * Total v1: 49 bytes
 *
 * Version 2 adds:
 * 1  debug_log (bool)
 * Total: 50 bytes
 *
 * Version 3 adds:
 * 1  smt_mode
 * 1  ccd_mode
 * 1  ecore_mode
 * Total: 53 bytes
 *
 * Version 4 adds:
 * 2  api_port (uint16_t)
 * Total: 55 bytes
 *
 * Version 5 adds:
 * 1  sdm_mode
 * Total: 56 bytes
 *
 * Version 6 added fir_mode (1 byte, now removed — read and discard)
 * Total: 57 bytes
 *
 * Version 7 added proc_mode (1 byte, now superseded by rate_map)
 * Total: 58 bytes
 *
 * Version 8 adds:
 * 12  rate_map[12] (per-input-rate output config)
 * Total: 70 bytes
 *
 * Version 9 adds:
 * 12  rate_ntf[12] (per-input-rate NTF override, int8_t, -1=auto)
 * Total: 82 bytes */
/* Correct byte counts: 4(ver)+4(fs_in)+4(fs_out)+4(gain)+1(mute)
 * +4(depth)+4(cands)+4(lat)+4(ntf)+4(threads)+4(affinity)
 * +4(format)+4(output_format) = 49 for v1 */
#define CONFIG_V1_SIZE 49
#define CONFIG_V2_SIZE 50
#define CONFIG_V3_SIZE 53
#define CONFIG_V4_SIZE 55
#define CONFIG_V5_SIZE 56
#define CONFIG_V6_SIZE 57
#define CONFIG_V7_SIZE 58
#define CONFIG_V8_SIZE 70
#define CONFIG_V9_SIZE 82
#define CONFIG_V10_SIZE 84

/* Serialise config to a byte buffer. Returns bytes written. */
size_t config_serialize(const dsd_config_t *cfg, uint8_t *buf, size_t buf_size) {
    if (buf_size < CONFIG_V10_SIZE)
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
    /* v8: rate_map */
    memcpy(buf + pos, cfg->rate_map, RATE_MAP_COUNT);
    pos += RATE_MAP_COUNT;
    /* v9: rate_ntf */
    memcpy(buf + pos, cfg->rate_ntf, RATE_MAP_COUNT);
    pos += RATE_MAP_COUNT;
    /* v10: ML filter settings + antipop */
    pos = write_u8(buf, pos, cfg->ml_enabled ? 1 : 0);
    pos = write_u8(buf, pos, (uint8_t)cfg->ml_ep);
    pos = write_u8(buf, pos, cfg->antipop ? 1 : 0);

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
        /* Version 8 adds rate_map */
        if (version >= 8 && buf_size >= CONFIG_V8_SIZE) {
            memcpy(cfg->rate_map, buf + pos, RATE_MAP_COUNT);
            pos += RATE_MAP_COUNT;
            /* Version 9 adds rate_ntf */
            if (version >= 9 && buf_size >= CONFIG_V9_SIZE) {
                memcpy(cfg->rate_ntf, buf + pos, RATE_MAP_COUNT);
                pos += RATE_MAP_COUNT;
            }
            /* Version 10 adds ml_enabled, ml_ep, antipop */
            if (version >= 10 && buf_size >= CONFIG_V10_SIZE) {
                uint8_t u8;
                pos = read_u8(buf, pos, &u8); cfg->ml_enabled = (u8 != 0);
                pos = read_u8(buf, pos, &u8); cfg->ml_ep = (int)u8;
                /* antipop: read if available (85+ bytes), else keep default */
                if (pos < buf_size) {
                    pos = read_u8(buf, pos, &u8); cfg->antipop = (u8 != 0);
                }
            }
            /* else: ml_enabled/ml_ep/antipop stay at defaults */
        } else {
            /* Migrate from v1-v7: use fs_out + proc_mode to populate rate_map */
            uint8_t out_idx = dsd_to_rate_out(fs_out_raw);
            if (proc_mode_raw == 2 || version < 7) {
                /* DSD recode mode or pre-v7: apply to DSD inputs */
                if (out_idx != RATE_OUT_BYPASS) {
                    /* All DSD inputs → specified output rate */
                    for (int i = RATE_MAP_PCM_COUNT; i < RATE_MAP_COUNT; i++)
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

    /* Version 0 (legacy): raw memcpy of old dsd_config_t. */
    size_t old_size = sizeof(dsd_config_t) - sizeof(int) - RATE_MAP_COUNT;
    if (buf_size == old_size || buf_size == sizeof(dsd_config_t)) {
        memcpy(cfg, buf, buf_size < sizeof(dsd_config_t) ? buf_size : sizeof(dsd_config_t));
        if (buf_size == old_size)
            cfg->output_format = OUTPUT_DOP;
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

    /* Trellis depth must be power of 2 in [4, 32] */
    if (cfg->trellis_depth < 4)  cfg->trellis_depth = 4;
    if (cfg->trellis_depth > 32) cfg->trellis_depth = 32;
    /* Snap to nearest power of 2 */
    if (cfg->trellis_depth <= 4)       cfg->trellis_depth = 4;
    else if (cfg->trellis_depth <= 8)  cfg->trellis_depth = 8;
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
}
