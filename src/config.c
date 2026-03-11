/*
 * foo_dsd_trellis — Configuration / property page
 *
 * Runtime parameter storage and serialisation for the fb2k config store.
 * Phase 0: Scaffold — default config only.
 */

#include <string.h>
#include "../include/dsd_types.h"

/* Serialise config to a byte buffer. Returns bytes written. */
size_t config_serialize(const dsd_config_t *cfg, uint8_t *buf, size_t buf_size) {
    if (buf_size < sizeof(dsd_config_t))
        return 0;
    memcpy(buf, cfg, sizeof(dsd_config_t));
    return sizeof(dsd_config_t);
}

/* Deserialise config from a byte buffer. Returns 0 on success. */
int config_deserialize(dsd_config_t *cfg, const uint8_t *buf, size_t buf_size) {
    if (buf_size < sizeof(dsd_config_t)) {
        dsd_config_defaults(cfg);
        return -1;
    }
    memcpy(cfg, buf, sizeof(dsd_config_t));
    return 0;
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
}
