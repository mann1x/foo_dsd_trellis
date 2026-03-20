/*
 * foo_dsd_trellis — DSD Trellis SDM DSP Plugin for foobar2000
 * Common type definitions
 */

#ifndef DSD_TYPES_H
#define DSD_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
typedef unsigned long DWORD;
#endif

/* Standard DSD sample rates (44.1kHz family) */
#define DSD_RATE_64   2822400u   /* DSD64:  64 * 44100 */
#define DSD_RATE_128  5644800u   /* DSD128: 128 * 44100 */
#define DSD_RATE_256  11289600u  /* DSD256: 256 * 44100 */
#define DSD_RATE_512  22579200u  /* DSD512: 512 * 44100 */

/* DSD sample rates (48kHz family) */
#define DSD48_RATE_64   3072000u   /* DSD64/48:  64 * 48000 */
#define DSD48_RATE_128  6144000u   /* DSD128/48: 128 * 48000 */
#define DSD48_RATE_256  12288000u  /* DSD256/48: 256 * 48000 */
#define DSD48_RATE_512  24576000u  /* DSD512/48: 512 * 48000 */

/* DSD silence patterns */
#define DSD_SILENCE_A 0x69u
#define DSD_SILENCE_B 0x96u

/* Stream format (input detection) */
typedef enum {
    FORMAT_AUTO   = 0,
    FORMAT_DOP    = 1,
    FORMAT_NATIVE = 2,
} dsd_format_t;

/* Output format */
typedef enum {
    OUTPUT_DOP    = 0,    /* Re-encode to DoP (default) */
    OUTPUT_PCM    = 1,    /* Decimate to PCM (for visualization / non-DSD DACs) */
} dsd_output_format_t;

/* SDM mode selection */
typedef enum {
    SDM_MODE_PRECORR = 0,   /* Greedy + prediction correction (default, low CPU) */
    SDM_MODE_TRELLIS = 1,   /* Viterbi trellis search (high quality, high CPU) */
} sdm_mode_t;

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

/* PCM output bit depth */
typedef enum {
    PCM_BIT_AUTO  = -1,   /* Auto: float output */
    PCM_BIT_16    = 0,
    PCM_BIT_24    = 1,
    PCM_BIT_32    = 2,
    PCM_BIT_FLOAT = 3,
} pcm_bit_depth_t;

/* PCM output dither */
typedef enum {
    PCM_DITHER_AUTO   = -1,   /* Auto: none for float, TPDF for integer */
    PCM_DITHER_NONE   = 0,
    PCM_DITHER_TPDF   = 1,    /* Triangular probability density function */
    PCM_DITHER_SHAPED = 2,    /* First-order noise-shaped */
} pcm_dither_t;

/* Polyphase resampler engine (for cross-family PCM→PCM) */
typedef enum {
    RESAMPLE_AUTO = -1,   /* IPP default, libsoxr if available */
    RESAMPLE_IPP  = 0,    /* Force IPP polyphase */
    RESAMPLE_SOXR = 1,    /* Force libsoxr (fallback to IPP if DLL missing) */
} resample_engine_t;

/* libsoxr quality preset */
typedef enum {
    SOXR_QUALITY_MQ  = 0,  /* Medium: ~100 dB rejection */
    SOXR_QUALITY_HQ  = 1,  /* High: ~125 dB rejection (default) */
    SOXR_QUALITY_VHQ = 2,  /* Very High: ~175 dB rejection */
} soxr_quality_t;

/* ─── Per-rate configuration table ─── */

/* Rate map: per-input-rate output configuration.
 * Each entry maps an input rate to a DSD/PCM output rate (or bypass).
 * Indices 0-11 are v15 legacy (unchanged), 12-19 are v16 additions. */
#define RATE_MAP_COUNT_V15  12   /* v15 array size for config migration */
#define RATE_MAP_COUNT      20
#define RATE_MAP_PCM_COUNT  12   /* 8 standard + 4 high PCM */
#define RATE_MAP_DSD44_COUNT 4
#define RATE_MAP_DSD48_COUNT 4

/* Rate map input indices — first 12 unchanged from v15 */
#define RATEIDX_44100       0
#define RATEIDX_48000       1
#define RATEIDX_88200       2
#define RATEIDX_96000       3
#define RATEIDX_176400      4
#define RATEIDX_192000      5
#define RATEIDX_352800      6
#define RATEIDX_384000      7
#define RATEIDX_DSD64       8
#define RATEIDX_DSD128      9
#define RATEIDX_DSD256     10
#define RATEIDX_DSD512     11
/* v16 additions */
#define RATEIDX_705600     12
#define RATEIDX_768000     13
#define RATEIDX_1411200    14
#define RATEIDX_1536000    15
#define RATEIDX_DSD64_48   16
#define RATEIDX_DSD128_48  17
#define RATEIDX_DSD256_48  18
#define RATEIDX_DSD512_48  19

/* Rate map output encoding: stored in rate_map[i]
 * First 9 unchanged from v15, 10-24 are v16 additions. */
#define RATE_OUT_BYPASS      0
#define RATE_OUT_DSD64       1
#define RATE_OUT_DSD128      2
#define RATE_OUT_DSD256      3
#define RATE_OUT_DSD512      4
#define RATE_OUT_PCM44       5    /* Decimate to 44100 Hz PCM */
#define RATE_OUT_PCM88       6    /* Decimate to 88200 Hz PCM */
#define RATE_OUT_PCM176      7    /* Decimate to 176400 Hz PCM */
#define RATE_OUT_PCM352      8    /* Decimate to 352800 Hz PCM */
/* v16 additions */
#define RATE_OUT_DSD64_48    9
#define RATE_OUT_DSD128_48  10
#define RATE_OUT_DSD256_48  11
#define RATE_OUT_DSD512_48  12
#define RATE_OUT_PCM48      13
#define RATE_OUT_PCM96      14
#define RATE_OUT_PCM192     15
#define RATE_OUT_PCM384     16
#define RATE_OUT_PCM706     17   /* 705600 Hz */
#define RATE_OUT_PCM768     18   /* 768000 Hz */
#define RATE_OUT_PCM1411    19   /* 1411200 Hz */
#define RATE_OUT_PCM1536    20   /* 1536000 Hz */
#define RATE_OUT_COUNT      21

/* Lookup rate map index for a given input rate. Returns -1 if unknown. */
static inline int rate_map_index(uint32_t rate) {
    switch (rate) {
    case 44100:          return RATEIDX_44100;
    case 48000:          return RATEIDX_48000;
    case 88200:          return RATEIDX_88200;
    case 96000:          return RATEIDX_96000;
    case 176400:         return RATEIDX_176400;
    case 192000:         return RATEIDX_192000;
    case 352800:         return RATEIDX_352800;
    case 384000:         return RATEIDX_384000;
    case DSD_RATE_64:    return RATEIDX_DSD64;
    case DSD_RATE_128:   return RATEIDX_DSD128;
    case DSD_RATE_256:   return RATEIDX_DSD256;
    case DSD_RATE_512:   return RATEIDX_DSD512;
    case 705600:         return RATEIDX_705600;
    case 768000:         return RATEIDX_768000;
    case 1411200:        return RATEIDX_1411200;
    case 1536000:        return RATEIDX_1536000;
    case DSD48_RATE_64:  return RATEIDX_DSD64_48;
    case DSD48_RATE_128: return RATEIDX_DSD128_48;
    case DSD48_RATE_256: return RATEIDX_DSD256_48;
    case DSD48_RATE_512: return RATEIDX_DSD512_48;
    default:             return -1;
    }
}

/* Convert rate_map output index to DSD rate. Returns 0 for non-DSD outputs. */
static inline uint32_t rate_out_to_dsd(uint8_t out_idx) {
    switch (out_idx) {
    case RATE_OUT_DSD64:     return DSD_RATE_64;
    case RATE_OUT_DSD128:    return DSD_RATE_128;
    case RATE_OUT_DSD256:    return DSD_RATE_256;
    case RATE_OUT_DSD512:    return DSD_RATE_512;
    case RATE_OUT_DSD64_48:  return DSD48_RATE_64;
    case RATE_OUT_DSD128_48: return DSD48_RATE_128;
    case RATE_OUT_DSD256_48: return DSD48_RATE_256;
    case RATE_OUT_DSD512_48: return DSD48_RATE_512;
    default:                 return 0;
    }
}

/* Convert rate_map output index to PCM rate. Returns 0 for non-PCM outputs. */
static inline uint32_t rate_out_to_pcm(uint8_t out_idx) {
    switch (out_idx) {
    case RATE_OUT_PCM44:   return 44100;
    case RATE_OUT_PCM88:   return 88200;
    case RATE_OUT_PCM176:  return 176400;
    case RATE_OUT_PCM352:  return 352800;
    case RATE_OUT_PCM48:   return 48000;
    case RATE_OUT_PCM96:   return 96000;
    case RATE_OUT_PCM192:  return 192000;
    case RATE_OUT_PCM384:  return 384000;
    case RATE_OUT_PCM706:  return 705600;
    case RATE_OUT_PCM768:  return 768000;
    case RATE_OUT_PCM1411: return 1411200;
    case RATE_OUT_PCM1536: return 1536000;
    default:               return 0;
    }
}

/* Convert rate_map output index to sample rate (DSD or PCM). Returns 0 for bypass. */
static inline uint32_t rate_out_to_hz(uint8_t out_idx) {
    uint32_t r = rate_out_to_dsd(out_idx);
    return r ? r : rate_out_to_pcm(out_idx);
}

static inline bool rate_out_is_dsd(uint8_t out_idx) {
    return (out_idx >= RATE_OUT_DSD64 && out_idx <= RATE_OUT_DSD512) ||
           (out_idx >= RATE_OUT_DSD64_48 && out_idx <= RATE_OUT_DSD512_48);
}

static inline bool rate_out_is_pcm(uint8_t out_idx) {
    return rate_out_to_pcm(out_idx) != 0;
}

/* Convert DSD rate to rate_map output index. Returns RATE_OUT_BYPASS if unknown. */
static inline uint8_t dsd_to_rate_out(uint32_t dsd_rate) {
    switch (dsd_rate) {
    case DSD_RATE_64:    return RATE_OUT_DSD64;
    case DSD_RATE_128:   return RATE_OUT_DSD128;
    case DSD_RATE_256:   return RATE_OUT_DSD256;
    case DSD_RATE_512:   return RATE_OUT_DSD512;
    case DSD48_RATE_64:  return RATE_OUT_DSD64_48;
    case DSD48_RATE_128: return RATE_OUT_DSD128_48;
    case DSD48_RATE_256: return RATE_OUT_DSD256_48;
    case DSD48_RATE_512: return RATE_OUT_DSD512_48;
    default:             return RATE_OUT_BYPASS;
    }
}

/* ─── Rate family helpers ─── */

static inline bool rate_is_44k_family(uint32_t rate) {
    return (rate % 44100 == 0);
}

static inline bool rate_is_48k_family(uint32_t rate) {
    return (rate % 48000 == 0) && (rate % 44100 != 0);
}

static inline bool rate_idx_is_pcm(int idx) {
    return (idx >= RATEIDX_44100 && idx <= RATEIDX_384000) ||
           (idx >= RATEIDX_705600 && idx <= RATEIDX_1536000);
}

static inline bool rate_idx_is_dsd(int idx) {
    return (idx >= RATEIDX_DSD64 && idx <= RATEIDX_DSD512) ||
           (idx >= RATEIDX_DSD64_48 && idx <= RATEIDX_DSD512_48);
}

static inline bool rate_idx_is_dsd48(int idx) {
    return (idx >= RATEIDX_DSD64_48 && idx <= RATEIDX_DSD512_48);
}

static inline bool rate_out_is_dsd44(uint8_t out_idx) {
    return (out_idx >= RATE_OUT_DSD64 && out_idx <= RATE_OUT_DSD512);
}

static inline bool rate_out_is_dsd48(uint8_t out_idx) {
    return (out_idx >= RATE_OUT_DSD64_48 && out_idx <= RATE_OUT_DSD512_48);
}

/* Get rate Hz for an input index */
static inline uint32_t rate_idx_to_hz(int idx) {
    static const uint32_t table[RATE_MAP_COUNT] = {
        44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000,
        DSD_RATE_64, DSD_RATE_128, DSD_RATE_256, DSD_RATE_512,
        705600, 768000, 1411200, 1536000,
        DSD48_RATE_64, DSD48_RATE_128, DSD48_RATE_256, DSD48_RATE_512
    };
    if (idx < 0 || idx >= RATE_MAP_COUNT) return 0;
    return table[idx];
}

/* Check if an input rate index is 48kHz family */
static inline bool rate_idx_is_48k_family(int idx) {
    return idx == RATEIDX_48000  || idx == RATEIDX_96000 ||
           idx == RATEIDX_192000 || idx == RATEIDX_384000 ||
           idx == RATEIDX_768000 || idx == RATEIDX_1536000 ||
           rate_idx_is_dsd48(idx);
}

/* Check if an output is valid for the given input rate index.
 * Rules:
 * - PCM→PCM: any combination (same-family FIR, cross-family polyphase)
 * - PCM→DSD: same family only (44.1k PCM→DSD/44, 48k PCM→DSD/48)
 * - DSD→PCM: any combination (same-family FIR decimate, cross-family polyphase)
 * - DSD→DSD: same family only (DSD/44→DSD/44, DSD/48→DSD/48) */
static inline bool rate_map_valid_output(int input_idx, uint8_t output_idx) {
    if (output_idx == RATE_OUT_BYPASS) return true;
    if (output_idx >= RATE_OUT_COUNT) return false;
    if (input_idx < 0 || input_idx >= RATE_MAP_COUNT) return false;

    bool in_is_pcm = rate_idx_is_pcm(input_idx);
    bool in_is_48k = rate_idx_is_48k_family(input_idx);
    bool out_is_dsd = rate_out_is_dsd(output_idx);
    bool out_is_pcm_val = rate_out_is_pcm(output_idx);

    if (in_is_pcm && out_is_pcm_val) {
        /* PCM→PCM: any combination allowed */
        return true;
    }
    if (in_is_pcm && out_is_dsd) {
        /* PCM→DSD: same family only */
        bool out_is_48k_dsd = rate_out_is_dsd48(output_idx);
        return in_is_48k == out_is_48k_dsd;
    }
    if (rate_idx_is_dsd(input_idx) && out_is_pcm_val) {
        /* DSD→PCM: any combination allowed */
        return true;
    }
    if (rate_idx_is_dsd(input_idx) && out_is_dsd) {
        /* DSD→DSD: same family only */
        bool in_is_dsd48 = rate_idx_is_dsd48(input_idx);
        bool out_is_48k_dsd = rate_out_is_dsd48(output_idx);
        return in_is_dsd48 == out_is_48k_dsd;
    }
    return false;
}

/* Runtime configuration (populated from property page) */
typedef struct {
    uint32_t  fs_in;          /* Input rate (set at runtime) */
    uint32_t  fs_out;         /* Output DSD rate (set at runtime from rate_map) */
    float     gain;           /* Linear volume gain [0.0f - 1.0f] */
    bool      mute;           /* Substitute silence pattern, skip SDM */
    int       trellis_depth;  /* Look-ahead N: 4, 8, 16, or 32 */
    int       trellis_cands;  /* Max survivors M: 4 - 32 */
    int       trellis_lat;    /* Traceback latency L: 16 - 2048 samples */
    int       ntf_filter;     /* NTF filter enum (ntf_filter_id_t or NTF_AUTO) */
    int       thread_count;   /* Worker threads: 1 - logical_cpu_count */
    DWORD     affinity_mask;  /* SetThreadAffinityMask value, 0 = OS default */
    int       format;         /* dsd_format_t (auto-detected or forced) */
    int       output_format;  /* dsd_output_format_t: DoP or PCM output */
    bool      debug_log;      /* Write diagnostic log to file */
    int       smt_mode;       /* 0=Auto (prefer T0), 1=T0 only */
    int       ccd_mode;       /* 0=Auto (prefer first CCD), 1=All CCDs */
    int       ecore_mode;     /* 0=Auto, 1=Exclude E-cores, 2=E-cores only */
    uint16_t  api_port;       /* REST API port (0=disabled, default 8881) */
    int       sdm_mode;       /* sdm_mode_t: PreCorr or Trellis */
    uint8_t   rate_map[RATE_MAP_COUNT]; /* Per-input-rate output config */
    int8_t    rate_ntf[RATE_MAP_COUNT]; /* Per-input-rate NTF override, NTF_AUTO = auto */
    int8_t    rate_sdm[RATE_MAP_COUNT]; /* Per-input-rate SDM mode: -1=Auto, 0=PreCorr, 1=Trellis */
    int8_t    rate_cands[RATE_MAP_COUNT]; /* Per-input-rate candidates: -1=Auto, 4/8/16/32 */
    int8_t    rate_depth[RATE_MAP_COUNT]; /* Per-input-rate depth: -1=Auto, 4/5/6/7/8 */
    int8_t    rate_ml[RATE_MAP_COUNT];    /* Per-input-rate ML filter: -1=Auto, 0=Off, 1=On */
    int8_t    rate_limiter[RATE_MAP_COUNT]; /* Per-input-rate state limiter: -1=Auto, 0=Off, 1-20=limit */
    bool      antipop;        /* Enable anti-pop lead-in silence */
    bool      ml_enabled;     /* Enable ONNX ML post-filter */
    int       ml_ep;          /* ml_ep_t: CPU (0) or DirectML (1) */
    int8_t    fir_gain_db;    /* Global FIR gain limit in dB (0 to -12). FIR_GAIN_AUTO = use path_config */
    bool      gpu_enabled;   /* Enable GPU compute offload */
    int       gpu_backend;   /* gpu_backend_t: None(0)/DirectX(1)/CUDA(2)/Auto(3) */
    int8_t    rate_gpu[RATE_MAP_COUNT];     /* Per-input-rate GPU FIR: -1=Auto, 0=Off, 1=On */
    int8_t    rate_fir_mode[RATE_MAP_COUNT]; /* Per-input-rate pre-SDM filter: -1=Auto, 0=Boxcar, 1=FIR */
    int16_t   rate_lat[RATE_MAP_COUNT];    /* Per-input-rate trellis latency: 0=Auto, >0=explicit */
    int       fir_mode;      /* Runtime: resolved pre-SDM filter mode (-1=Auto, 0=Boxcar, 1=FIR). Not serialized. */
    float     state_limit;   /* Runtime: resolved state limiter (0=off, >0=limit). Set per-chunk. */
    /* v16: PCM encoding */
    int8_t    pcm_bit_depth;     /* Global PCM bit depth: PCM_BIT_AUTO(-1), 16/24/32/float */
    int8_t    pcm_dither;        /* Global PCM dither: PCM_DITHER_AUTO(-1), none/TPDF/shaped */
    int8_t    resample_engine;   /* resample_engine_t: -1=Auto, 0=IPP, 1=soxr */
    int8_t    soxr_quality;      /* soxr_quality_t: 0=MQ, 1=HQ(default), 2=VHQ */
    int8_t    rate_pcm_bits[RATE_MAP_COUNT];   /* Per-rate PCM bit depth override */
    int8_t    rate_pcm_dither[RATE_MAP_COUNT]; /* Per-rate PCM dither override */
} dsd_config_t;

/* Pre-SDM filter mode */
#define FIR_MODE_AUTO    (-1)
#define FIR_MODE_BOXCAR   0
#define FIR_MODE_FIR      1

/* Config serialization version */
#define DSD_CONFIG_VERSION 16

/* FIR gain Auto sentinel and default */
#define FIR_GAIN_AUTO    (-128)
#define FIR_GAIN_DEFAULT (-3)   /* -3 dB = 0.71 linear */

/* Convert FIR gain dB to linear. Handles Auto sentinel. */
static inline float fir_gain_db_to_linear(int8_t db) {
    int d = (db == FIR_GAIN_AUTO) ? FIR_GAIN_DEFAULT : (int)db;
    if (d >= 0) return 1.0f;
    /* Quick lookup for common values, avoids powf */
    switch (d) {
    case -1:  return 0.891f;
    case -2:  return 0.794f;
    case -3:  return 0.708f;
    case -4:  return 0.631f;
    case -5:  return 0.562f;
    case -6:  return 0.501f;
    case -7:  return 0.447f;
    case -8:  return 0.398f;
    case -9:  return 0.355f;
    case -10: return 0.316f;
    case -11: return 0.282f;
    case -12: return 0.251f;
    default:  return 0.251f;  /* clamp to -12 dB */
    }
}

/* Default REST API port */
#define DSD_DEFAULT_API_PORT 8881

/* Default configuration values */
#define DSD_DEFAULT_GAIN         1.0f
#define DSD_DEFAULT_TRELLIS_N    8
#define DSD_DEFAULT_TRELLIS_M    4
#define DSD_DEFAULT_TRELLIS_LAT  32
#define DSD_DEFAULT_THREADS      0   /* 0 = auto (all logical cores) */

static inline void dsd_config_defaults(dsd_config_t *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->fs_in          = 0;  /* determined at runtime */
    cfg->fs_out         = 0;  /* set at runtime from rate_map */
    cfg->gain           = DSD_DEFAULT_GAIN;
    cfg->mute           = false;
    cfg->trellis_depth  = DSD_DEFAULT_TRELLIS_N;
    cfg->trellis_cands  = DSD_DEFAULT_TRELLIS_M;
    cfg->trellis_lat    = DSD_DEFAULT_TRELLIS_LAT;
    cfg->ntf_filter     = NTF_AUTO;
    cfg->thread_count   = DSD_DEFAULT_THREADS;
    cfg->affinity_mask  = 0;
    cfg->format         = FORMAT_AUTO;
    cfg->output_format  = OUTPUT_DOP;
    cfg->debug_log      = false;
    cfg->smt_mode       = 0;  /* SMT_AUTO */
    cfg->ccd_mode       = 0;  /* CCD_AUTO */
    cfg->ecore_mode     = 0;  /* ECORE_AUTO */
    cfg->api_port       = DSD_DEFAULT_API_PORT;
    cfg->sdm_mode       = SDM_MODE_PRECORR;
    memset(cfg->rate_map, RATE_OUT_BYPASS, sizeof(cfg->rate_map));
    memset(cfg->rate_ntf, 0xFF, sizeof(cfg->rate_ntf)); /* NTF_AUTO = -1 = 0xFF */
    memset(cfg->rate_sdm, 0xFF, sizeof(cfg->rate_sdm)); /* -1 = Auto (use global) */
    memset(cfg->rate_cands, 0xFF, sizeof(cfg->rate_cands)); /* -1 = Auto */
    memset(cfg->rate_depth, 0xFF, sizeof(cfg->rate_depth)); /* -1 = Auto */
    memset(cfg->rate_ml, 0xFF, sizeof(cfg->rate_ml)); /* -1 = Auto (use global) */
    memset(cfg->rate_limiter, 0xFF, sizeof(cfg->rate_limiter)); /* -1 = Auto (use path_config) */
    cfg->antipop    = true;   /* enabled by default */
    cfg->ml_enabled = false;
    cfg->ml_ep      = 2;  /* ML_EP_AUTO */
    cfg->fir_gain_db = FIR_GAIN_AUTO;  /* Auto = -3 dB */
    cfg->gpu_enabled = false;
    cfg->gpu_backend = 3;  /* GPU_BACKEND_AUTO */
    memset(cfg->rate_gpu, 0xFF, sizeof(cfg->rate_gpu));     /* -1 = Auto */
    memset(cfg->rate_fir_mode, 0xFF, sizeof(cfg->rate_fir_mode)); /* -1 = Auto */
    memset(cfg->rate_lat, 0, sizeof(cfg->rate_lat));       /* 0 = Auto */
    cfg->fir_mode = FIR_MODE_AUTO;
    cfg->state_limit = -1.0f;         /* -1 = Auto (use path_config) */
    /* v16: PCM encoding */
    cfg->pcm_bit_depth = PCM_BIT_AUTO;
    cfg->pcm_dither = PCM_DITHER_AUTO;
    cfg->resample_engine = RESAMPLE_AUTO;
    cfg->soxr_quality = SOXR_QUALITY_HQ;
    memset(cfg->rate_pcm_bits, 0xFF, sizeof(cfg->rate_pcm_bits));     /* -1 = Auto */
    memset(cfg->rate_pcm_dither, 0xFF, sizeof(cfg->rate_pcm_dither)); /* -1 = Auto */
}

#endif /* DSD_TYPES_H */
