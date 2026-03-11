/*
 * foo_dsd_trellis — DSD Trellis SDM DSP Plugin for foobar2000
 * Common type definitions
 */

#ifndef DSD_TYPES_H
#define DSD_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef _WIN32
#include <windows.h>
#else
typedef unsigned long DWORD;
#endif

/* Standard DSD sample rates */
#define DSD_RATE_64   2822400u   /* DSD64:  64 * 44100 */
#define DSD_RATE_128  5644800u   /* DSD128: 128 * 44100 */
#define DSD_RATE_256  11289600u  /* DSD256: 256 * 44100 */
#define DSD_RATE_512  22579200u  /* DSD512: 512 * 44100 */

/* DSD silence patterns */
#define DSD_SILENCE_A 0x69u
#define DSD_SILENCE_B 0x96u

/* Stream format */
typedef enum {
    FORMAT_AUTO   = 0,
    FORMAT_DOP    = 1,
    FORMAT_NATIVE = 2,
} dsd_format_t;

/* NTF filter selection */
typedef enum {
    NTF_AUTO    = -1,
    NTF_CLANS_4 = 0,
    NTF_SDM_4,
    NTF_CLANS_5,
    NTF_SDM_5,
    NTF_CLANS_6,
    NTF_SDM_6,
    NTF_CLANS_7,
    NTF_SDM_7,
    NTF_CLANS_8,
    NTF_SDM_8,
    NTF_COUNT
} ntf_filter_id_t;

/* Runtime configuration (populated from property page) */
typedef struct {
    uint32_t  fs_in;          /* Input DSD rate */
    uint32_t  fs_out;         /* Output DSD rate */
    float     gain;           /* Linear volume gain [0.0f - 1.0f] */
    bool      mute;           /* Substitute silence pattern, skip SDM */
    int       trellis_depth;  /* Look-ahead N: 4, 8, 16, or 32 */
    int       trellis_cands;  /* Max survivors M: 4 - 32 */
    int       trellis_lat;    /* Traceback latency L: 16 - 2048 samples */
    int       ntf_filter;     /* NTF filter enum (ntf_filter_id_t or NTF_AUTO) */
    int       thread_count;   /* Worker threads: 1 - logical_cpu_count */
    DWORD     affinity_mask;  /* SetThreadAffinityMask value, 0 = OS default */
    int       format;         /* dsd_format_t (auto-detected or forced) */
} dsd_config_t;

/* Default configuration values */
#define DSD_DEFAULT_GAIN         1.0f
#define DSD_DEFAULT_TRELLIS_N    8
#define DSD_DEFAULT_TRELLIS_M    16
#define DSD_DEFAULT_TRELLIS_LAT  64
#define DSD_DEFAULT_THREADS      0   /* 0 = auto (logical cores / 2) */

static inline void dsd_config_defaults(dsd_config_t *cfg) {
    cfg->fs_in          = 0;  /* determined at runtime */
    cfg->fs_out         = 0;  /* 0 = same as input */
    cfg->gain           = DSD_DEFAULT_GAIN;
    cfg->mute           = false;
    cfg->trellis_depth  = DSD_DEFAULT_TRELLIS_N;
    cfg->trellis_cands  = DSD_DEFAULT_TRELLIS_M;
    cfg->trellis_lat    = DSD_DEFAULT_TRELLIS_LAT;
    cfg->ntf_filter     = NTF_AUTO;
    cfg->thread_count   = DSD_DEFAULT_THREADS;
    cfg->affinity_mask  = 0;
    cfg->format         = FORMAT_AUTO;
}

#endif /* DSD_TYPES_H */
