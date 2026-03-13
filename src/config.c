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

static size_t read_u8(const uint8_t *buf, size_t pos, uint8_t *val) {
    *val = buf[pos];
    return pos + 1;
}

/* ─── Version 1 field layout ─── */
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
 * Total: 45 bytes
 *
 * Version 2 adds:
 * 1  debug_log (bool)
 * Total: 46 bytes
 *
 * Version 3 adds:
 * 1  smt_mode
 * 1  ccd_mode
 * 1  ecore_mode
 * Total: 49 bytes */
#define CONFIG_V1_SIZE 45
#define CONFIG_V2_SIZE 46
#define CONFIG_V3_SIZE 49

/* Serialise config to a byte buffer. Returns bytes written. */
size_t config_serialize(const dsd_config_t *cfg, uint8_t *buf, size_t buf_size) {
    if (buf_size < CONFIG_V3_SIZE)
        return 0;

    size_t pos = 0;
    pos = write_u32(buf, pos, DSD_CONFIG_VERSION);
    pos = write_u32(buf, pos, cfg->fs_in);
    pos = write_u32(buf, pos, cfg->fs_out);
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

    if ((version >= 1 && version <= 3) && buf_size >= CONFIG_V1_SIZE) {
        /* Version 1/2: field-by-field */
        size_t pos = 4;
        pos = read_u32(buf, pos, &cfg->fs_in);
        pos = read_u32(buf, pos, &cfg->fs_out);
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
        (void)pos;
        config_validate(cfg);
        return 0;
    }

    /* Version 0 (legacy): raw memcpy of old dsd_config_t (without output_format).
     * The old struct was sizeof(dsd_config_t) minus the new field.
     * Detect by checking if buf_size matches the old struct size. */
    size_t old_size = sizeof(dsd_config_t) - sizeof(int); /* minus output_format */
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

    /* Validate DSD rates */
    switch (cfg->fs_out) {
    case 0: /* as input */
    case DSD_RATE_64:
    case DSD_RATE_128:
    case DSD_RATE_256:
    case DSD_RATE_512:
        break;
    default:
        cfg->fs_out = 0;
        break;
    }

    /* Validate output format */
    if (cfg->output_format != OUTPUT_DOP && cfg->output_format != OUTPUT_PCM)
        cfg->output_format = OUTPUT_DOP;
}
