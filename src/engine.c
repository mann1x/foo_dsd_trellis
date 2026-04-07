/*
 * foo_dsd_trellis — Top-level per-channel processing orchestrator
 *
 * Phase 0: Scaffold — stub implementations.
 * Phase 4-6 will wire up FIR → gain → SDM → repack pipeline.
 */

#include "../include/engine.h"
#include "../include/ntf.h"
#include <math.h>
#include "../include/preemph.h"
#include <stdlib.h>
#include <string.h>
#include <ipps.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>

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
    /* All paths use gain=1.0 (0 dB) for maximum signal-to-quantization ratio.
     * SDM limiter (state_limit) handles FIR overload on rate-conversion paths.
     * Previous -3 dB attenuation was unnecessary and worsened noise modulation
     * by reducing the signal to ~7% of the quantizer range on same-rate paths.
     * lat = 0 means "auto" (computed as cands * 8 at runtime).
     * Optimal lat from nc×lat sweep: nc=2→16, nc=4→32, nc=8→64. */
    /* Same-rate re-encode: boxcar DSD-Wide → trellis SDM (2026-03-27).
     * Boxcar taps: DSD64=32, DSD128/256=64, DSD512=16.
     * End-to-end SINAD (boxcar + SDM, multi-freq median):
     *   DSD64:  CLANS6/d=16/lat=32 → 84.5 dB
     *   DSD128: CLANS6/d=4/lat=128 → 99.5 dB
     *   DSD256: CLANS6/d=4/lat=128 → 108.5 dB
     *   DSD512: SDM6/d=4/lat=32    → 108.6 dB
     * nc=2 avoids candidate collapse and is 2x cheaper.
     * DSD64 needs depth=16: 4-bit dedup mask kills path diversity at low OSR. */
    { DSD_RATE_64,  DSD_RATE_64,  NTF_CLANS_6, 0.0,  2,  0, 16, 0.708f },
    { DSD_RATE_128, DSD_RATE_128, NTF_CLANS_6, 0.0,  2,  0, 4, 0.708f },
    { DSD_RATE_256, DSD_RATE_256, NTF_CLANS_6, 0.0,  2,  0, 4, 0.708f },
    { DSD_RATE_512, DSD_RATE_512, NTF_SDM_6,   0.0,  2,  0, 4, 0.708f },
    /* Rate conversion with boxcar DSD-Wide pre-smooth (2026-04-05 sweep).
     * nc=2 for all paths — matches same-rate throughput for parallel DAS.
     * Boxcar eliminates FIR Gibbs peaks, enabling lower cands without
     * quality loss. Sweep: 6 paths × 10 NTF × 8 lim × 3 cands × 3 depth. */
    /* Upsample (swept paths marked ✓, pending marked ~) */
    { DSD_RATE_64,  DSD_RATE_128, NTF_CLANS_6, 0.0,  2,  0, 4, 0.708f },  /* PreCorr forced */
    { DSD_RATE_64,  DSD_RATE_256, NTF_CLANS_6, 0.0,  2,  0, 4, 0.708f },  /* ~ pending sweep */
    { DSD_RATE_64,  DSD_RATE_512, NTF_CLANS_6, 10.0, 2,  0, 16, 0.708f }, /* ✓ 66.3 dB */
    { DSD_RATE_128, DSD_RATE_256, NTF_CLANS_6, 0.0,  2,  0, 4, 0.708f },  /* ~ pending sweep */
    { DSD_RATE_128, DSD_RATE_512, NTF_CLANS_6, 0.0,  2,  0, 4, 0.708f },  /* ~ pending sweep */
    { DSD_RATE_256, DSD_RATE_512, NTF_CLANS_6, 0.0,  2,  0, 4, 0.708f },  /* ~ pending sweep */
    /* Downsample (all swept ✓) */
    { DSD_RATE_128, DSD_RATE_64,  NTF_CLANS_4, 0.0,  2,  0, 4, 0.708f },  /* ✓ 89.4 dB */
    { DSD_RATE_256, DSD_RATE_64,  NTF_SDM_6,  12.0,  2,  0, 4, 0.708f },  /* ✓ 74.2 dB */
    { DSD_RATE_256, DSD_RATE_128, NTF_SDM_6,   0.0,  2,  0, 8, 0.708f },  /* ✓ 101.4 dB */
    { DSD_RATE_512, DSD_RATE_64,  NTF_SDM_4,  12.0,  2,  0, 4, 0.708f },  /* ✓ 77.0 dB */
    { DSD_RATE_512, DSD_RATE_128, NTF_SDM_6,   0.0,  2,  0, 4, 0.708f },  /* ~ pending sweep */
    { DSD_RATE_512, DSD_RATE_256, NTF_SDM_6,   0.0,  2,  0, 4, 0.708f },  /* ~ pending sweep */
    /* ─── DSD/48 paths (independently swept, NOT mirrored from /44) ─── */
    /* Same-rate re-encode /48: boxcar DSD-Wide → trellis SDM (2026-03-27):
     *   DSD64/48:  SDM-6/d=16/lat=64  → 84.6 dB
     *   DSD128/48: CLANS-6/d=4/lat=32 → 103.4 dB
     *   DSD256/48: SDM-4/d=16/lat=128 → 103.8 dB
     *   DSD512/48: SDM-4/d=16/lat=128 → 109.8 dB */
    { DSD48_RATE_64,  DSD48_RATE_64,  NTF_SDM_6,   0.0,  2,  64, 16, 0.708f },
    { DSD48_RATE_128, DSD48_RATE_128, NTF_CLANS_6, 0.0,  2,  32,  4, 0.708f },
    { DSD48_RATE_256, DSD48_RATE_256, NTF_SDM_4,   0.0,  2, 128, 16, 0.708f },
    { DSD48_RATE_512, DSD48_RATE_512, NTF_SDM_4,   0.0,  2, 128, 16, 0.708f },
    /* DSD/48 rate conversion — mirrored from /44 sweep results, nc=2 */
    { DSD48_RATE_64,  DSD48_RATE_128, NTF_CLANS_6, 0.0,  2,  0, 4, 0.708f },  /* PreCorr forced */
    { DSD48_RATE_64,  DSD48_RATE_256, NTF_CLANS_6, 0.0,  2,  0, 4, 0.708f },
    { DSD48_RATE_64,  DSD48_RATE_512, NTF_CLANS_6, 10.0, 2,  0, 16, 0.708f },
    { DSD48_RATE_128, DSD48_RATE_256, NTF_CLANS_6, 0.0,  2,  0, 4, 0.708f },
    { DSD48_RATE_128, DSD48_RATE_512, NTF_CLANS_6, 0.0,  2,  0, 4, 0.708f },
    { DSD48_RATE_256, DSD48_RATE_512, NTF_CLANS_6, 0.0,  2,  0, 4, 0.708f },
    { DSD48_RATE_128, DSD48_RATE_64,  NTF_CLANS_4, 0.0,  2,  0, 4, 0.708f },
    { DSD48_RATE_256, DSD48_RATE_64,  NTF_SDM_6,  12.0,  2,  0, 4, 0.708f },
    { DSD48_RATE_256, DSD48_RATE_128, NTF_SDM_6,   0.0,  2,  0, 8, 0.708f },
    { DSD48_RATE_512, DSD48_RATE_64,  NTF_SDM_4,  12.0,  2,  0, 4, 0.708f },
    { DSD48_RATE_512, DSD48_RATE_128, NTF_SDM_6,   0.0,  2,  0, 4, 0.708f },
    { DSD48_RATE_512, DSD48_RATE_256, NTF_SDM_6,   0.0,  2,  0, 4, 0.708f },
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
    void *saved_gpu = eng->gpu;  /* preserve GPU ptr set by caller */
    memset(eng, 0, sizeof(*eng));
    eng->gpu = saved_gpu;
    eng->channel = channel;

    uint32_t fs_out = cfg->fs_out ? cfg->fs_out : cfg->fs_in;
    /* passthrough is evaluated dynamically in engine_process_block,
     * not at init time, so FIR+SDM are always available for gain control */
    eng->passthrough = false;

    /* Rate-adaptive boxcar taps for DSD-Wide smoothing.
     * Same-rate: DSD64=32, DSD128/256=64, DSD512=16 (tuned for boxcar+SDM).
     * Rate conversion: fewer taps (4-8) preserve more HF content for FIR
     * upsample, giving 30-40 dB better SINAD than same-rate taps.
     * Sweep: DSD128→DSD256 boxcar=8 → 105.5 dB vs boxcar=64 → 66.2 dB. */
    {
        bool is_rate_conv = (cfg->fs_in != fs_out);
        if (is_rate_conv) {
            eng->boxcar.taps = 8;  /* optimal for rate conversion (sweep result) */
        } else {
            if (cfg->fs_in >= DSD_RATE_512)
                eng->boxcar.taps = 16;
            else if (cfg->fs_in >= DSD_RATE_128)
                eng->boxcar.taps = 64;
            else
                eng->boxcar.taps = 32;
        }
    }

    eng->sdm_mode = cfg->sdm_mode;

    /* Force PreCorr for DSD64→DSD128 (both /44 and /48 families).
     * Trellis SDM overloads on this path: FIR 2× upsample produces peaks
     * at ±2.24, and no NTF order (4-8) has sufficient MSA at gain=0.708.
     * PreCorr's greedy quantizer handles overload gracefully (0 dB transfer).
     * Only for DSD rates (≥ DSD_RATE_64), not PCM. */
    if (cfg->fs_in != fs_out && cfg->sdm_mode == SDM_MODE_TRELLIS
        && cfg->fs_in >= DSD_RATE_64) {
        uint32_t ratio = fs_out / cfg->fs_in;
        if (ratio == 2 && cfg->fs_in == DSD_RATE_64)
            eng->sdm_mode = SDM_MODE_PRECORR;
        if (ratio == 2 && cfg->fs_in == DSD48_RATE_64)
            eng->sdm_mode = SDM_MODE_PRECORR;
    }

    eng->fir_gain = 1.0f;  /* default, may be overridden by path_table */

    /* Same-rate pre-SDM filtering: Boxcar (DSD-Wide) for all rates.
     * Boxcar preserves DSD noise as natural dither for the SDM re-encoder,
     * giving +30 dB SINAD over FIR lowpass (DSD64: 94 vs 62 dB).
     * FIR lowpass available via explicit override for backward compat.
     * Boxcar taps are rate-adaptive (set above). */
    if (cfg->fs_in == fs_out && !cfg->mute) {
        bool use_fir = false;  /* default: boxcar for all rates */
        if (cfg->fir_mode == FIR_MODE_FIR)
            use_fir = true;
        else if (cfg->fir_mode == FIR_MODE_BOXCAR)
            use_fir = false;
        if (use_fir)
            fir_lowpass_init(&eng->lowpass, cfg->fs_in);
    }

    /* PCM output: FIR only, no SDM.
     * SDM produces ±1.0 DSD bitstream — meaningless for PCM output.
     * Covers both DSD→PCM decimation and PCM→PCM rate conversion. */
    eng->fir_only = (fs_out < DSD_RATE_64);

    /* Resolve FIR precision: Auto defaults to fp64 */
    bool use_fp64 = true;
    if (cfg->fir_prec == FIR_PREC_FP32)
        use_fp64 = false;

    if (eng->fir_only && !cfg->mute) {
        if (fir_chain_init_ex(&eng->fir, cfg->fs_in, fs_out, use_fp64) != 0)
            return -1;
        return 0;
    }

    if (!cfg->mute) {
        if (fir_chain_init_ex(&eng->fir, cfg->fs_in, fs_out, use_fp64) != 0)
            return -1;

        /* Path-adaptive lookup — always look up for gain/limiter.
         * NTF from path_config only used when ntf_filter == NTF_AUTO. */
        bool is_rate_conv = (cfg->fs_in != fs_out);
        const path_config_t *pc = path_config_lookup(cfg->fs_in, fs_out);
        if (pc)
            eng->fir_gain = pc->fir_gain;
        else if (is_rate_conv)
            eng->fir_gain = fir_gain_db_to_linear(FIR_GAIN_DEFAULT) * 0.5f; /* -9 dB for PCM rate conv (no path table) */

        /* No fir_gain reduction needed for PreCorr upsample.
         * Boxcar DSD-Wide pre-smooth before FIR eliminates Gibbs peaks
         * (±2.24 from raw DSD → ±1.0 from boxcar output). */



        /* Apply user FIR gain override.
         * Auto: use path_config gain (0.708 = -3 dB for all paths).
         * Explicit: replace path_config gain with user's choice. */
        if (cfg->fir_gain_db != FIR_GAIN_AUTO)
            eng->fir_gain = fir_gain_db_to_linear(cfg->fir_gain_db);

        /* Select NTF filter — PreCorr uses its own auto-select even when
         * path_config exists (path_config NTF is Trellis-optimized) */
        const ntf_filter_t *filter = NULL;
        if (cfg->ntf_filter == NTF_AUTO) {
            if (eng->sdm_mode == SDM_MODE_PRECORR) {
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
        else if (is_rate_conv && eng->sdm_mode == SDM_MODE_PRECORR)
            resolved_limit = 12.0;  /* PreCorr default for rate conversion */

        if (eng->sdm_mode == SDM_MODE_PRECORR) {
            if (precorr_context_init(&eng->precorr, filter) != 0)
                return -1;
            if (resolved_limit > 0.0)
                eng->precorr.state_limit = (float)resolved_limit;
        } else {
            int cands = cfg->trellis_cands;
            int lat   = cfg->trellis_lat;
            eng->sdm_fast = (cfg->fir_prec == FIR_PREC_FP32);
            int rc = eng->sdm_fast
                ? sdm_context_init_fast(&eng->sdm, filter,
                                         cfg->trellis_depth, cands, lat)
                : sdm_context_init(&eng->sdm, filter,
                                    cfg->trellis_depth, cands, lat);
            if (rc != 0)
                return -1;
            if (resolved_limit > 0.0)
                eng->sdm.state_limit = resolved_limit;
        }
    }

    /* Convolution filter (room correction).
     * GPU path: convolve at full DSD rate (max quality, no decimation).
     * CPU path: decimate to 176.4/192kHz, convolve, interpolate back.
     * PCM: always direct (no decimation needed). */
    eng->conv = NULL;
    if (cfg->conv_enabled && channel < CONV_MAX_CHANNELS
        && cfg->conv_paths[channel][0] != '\0') {
        bool try_gpu = (eng->gpu && cfg->gpu_enabled && cfg->conv_gpu && !eng->fir_only);
        uint32_t conv_rate;
        if (try_gpu) {
            conv_rate = fs_out;  /* GPU: full DSD rate */
        } else if (eng->fir_only) {
            conv_rate = fs_out;  /* PCM: direct */
        } else {
            uint32_t base = (fs_out % 48000 == 0) ? 48000 : 44100;
            conv_rate = base * 4;  /* CPU DSD: decimated */
        }
        conv_state_t *cs = (conv_state_t *)calloc(1, sizeof(conv_state_t));
        if (cs) {
            if (conv_init(cs, fs_out, conv_rate) == 0) {
                /* GPU conv: use larger partition sizes and cap IR taps
                 * for real-time performance. Must set AFTER conv_init
                 * (which memsets) and BEFORE conv_load_ir. */
                if (try_gpu) {
                    cs->gpu_partitions = true;
                    /* Budget-based IR tap cap for GPU real-time conv.
                     * Caps are applied AFTER the IR is resampled to the
                     * DSD playback rate, so they limit the post-resample
                     * tap count (= the actual size of the partitioned
                     * frequency-domain IR the GPU operates on).
                     *
                     * Calibrated empirically on RTX 5080 with the conv
                     * dispatch optimizations applied (P=65536 single-
                     * partition mode, consolidated DtoH, fdl_pos in
                     * kernel, CUDA Graph capture). Trellis SDM cost
                     * dominates the per-batch budget; conv cap controls
                     * how much additional GPU/dispatch headroom is left.
                     *
                     * High budget (proc ≤ ~85%, GPU ≤ ~30%):
                     *   DSD512+Trellis 2M (np=32): 80-83% proc, 28% GPU
                     *   DSD256+Trellis 1M (np=16): 79-84% proc, 25% GPU
                     *   DSD256+Trellis 2M (np=32): 79-91% proc — TOO HIGH (peaks)
                     *   DSD128+Trellis 2M cap → 512K post-resample: 39-42% (room)
                     *   DSD64 +Trellis 2M cap → 256K post-resample: 20-29% (room)
                     *
                     * Medium / Low: cap is halved / quartered. After the
                     * conv overhead optimizations the GPU is no longer
                     * the bottleneck — these mostly save VRAM and give
                     * the user a way to dial down conv work for
                     * VRAM-constrained or shared-GPU scenarios. */
                    bool trellis = (eng->sdm_mode == SDM_MODE_TRELLIS);
                    int cap;
                    if (fs_out >= 22000000) {        /* DSD512 */
                        cap = trellis ? (1 << 21)    /* 2M — measured 80-83% */
                                      : (1 << 21);   /* 2M — PreCorr (cheaper SDM, more headroom) */
                    } else if (fs_out >= 11000000) { /* DSD256 */
                        cap = trellis ? (1 << 20)    /* 1M — measured 79-84% (2M peaked 91%) */
                                      : (1 << 21);   /* 2M — PreCorr */
                    } else if (fs_out >=  5000000) { /* DSD128 */
                        cap = trellis ? (1 << 21)    /* 2M — DSD128 has plenty of headroom */
                                      : (1 << 21);   /* 2M — PreCorr */
                    } else {                          /* DSD64 */
                        cap = trellis ? (1 << 21)    /* 2M — DSD64 has lots of headroom */
                                      : (1 << 21);   /* 2M — PreCorr */
                    }
                    if (cfg->conv_budget == 1)      cap >>= 1;
                    else if (cfg->conv_budget >= 2)  cap >>= 2;
                    cs->max_ir_taps = cap;
                }
                if (conv_load_ir(cs, cfg->conv_paths[channel]) == 0) {
                    /* Try GPU convolution if available */
                    if (try_gpu && cs->ir.freq_partitions) {
                        int parts = cs->ir.num_partitions;
                        cs->gpu_conv = gpu_conv_init(
                            eng->gpu, parts, cs->ir.partition_size,
                            cs->ir.fft_size, cs->ir.freq_partitions, channel);
                        if (cs->gpu_conv) {
                            cs->use_gpu = true;
                            cs->gpu_ctx = eng->gpu;
                            cs->dec_ratio = 1;
                            /* FIFO pre-fill is done inside gpu_cuda_conv_init */
                        }
                    }
                    eng->conv = cs;
                } else {
                    conv_free(cs);
                    free(cs);
                }
            } else {
                free(cs);
            }
        }
    }

    return 0;
}

/* Warm up boxcar+SDM with the first N samples of real audio input.
 * Called once after reset, before the first engine_process_block.
 * Feeds input through boxcar→SDM, discards output. This primes both
 * the boxcar ring buffer and SDM integrators with real audio state,
 * preventing the startup pop from zero-state mismatch. */
void engine_channel_warmup(engine_channel_t *eng, const float *in,
                            size_t count, const dsd_config_t *cfg) {
    if (!in || count == 0 || cfg->mute)
        return;

    uint32_t fs_out = cfg->fs_out ? cfg->fs_out : cfg->fs_in;
    if (cfg->fs_in != fs_out)
        return;  /* warmup only for same-rate (boxcar path) */

    /* Feed through boxcar — use all available samples to prime SDM
     * integrators. SDM needs hundreds of samples to settle, not just
     * 2x boxcar taps. */
    size_t warmup = count;

    double *tmp = (double *)malloc(warmup * sizeof(double));
    if (!tmp) return;

    boxcar_t *bc = &eng->boxcar;
    const double inv_n = 1.0 / (double)bc->taps;
    for (size_t i = 0; i < warmup; i++) {
        double s = in[i] >= 0.0f ? 1.0 : -1.0;
        bc->sum -= bc->ring[bc->pos];
        bc->ring[bc->pos] = s;
        bc->sum += s;
        bc->pos = (bc->pos + 1) % bc->taps;
        tmp[i] = bc->sum * inv_n * (double)eng->fir_gain * (double)cfg->gain;
    }

    /* Feed through SDM (discard output) */
    float *trash = (float *)malloc(warmup * sizeof(float));
    if (trash) {
        if (eng->sdm_mode == SDM_MODE_PRECORR)
            precorr_process_block(&eng->precorr, tmp, trash, warmup);
        else
            (eng->sdm_fast ? sdm_process_block_fast : sdm_process_block)(&eng->sdm, tmp, trash, warmup);
        free(trash);
    }
    free(tmp);
}

/* Pre-SDM pre-emphasis: extract features, predict adaptive FIR taps, apply.
 * Uses ONNX GPU inference if available, falls back to embedded CPU MLP. */
static void engine_apply_preemph(engine_channel_t *eng,
                                  const dsd_config_t *cfg, size_t count) {
    size_t feat_n = count < 4096 ? count : 4096;
    float centroid = preemph_spectral_centroid(eng->fir_buf, feat_n, (double)cfg->fs_in);
    float rms = preemph_rms(eng->fir_buf, feat_n);
    float crest = preemph_crest_factor(eng->fir_buf, feat_n);
    float taps[3];
    if (eng->ml_filter) {
        float features[3] = { centroid, rms, crest };
        onnx_filter_predict_taps(eng->ml_filter, features, taps);
    } else {
        preemph_predict_taps(centroid, rms, crest, taps);
    }
    preemph_apply(eng->fir_buf, count, taps);
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
            /* DSD-Wide: smooth ±1.0 → multi-bit, apply gain, re-encode via SDM.
             * Trellis: use FIR lowpass (smooth output for parallel stitching).
             * PreCorr: use boxcar (fast, sufficient for greedy quantizer). */
            if (eng->fir_buf_sz < count * sizeof(double)) {
                free(eng->fir_buf);
                eng->fir_buf = (double *)malloc(count * sizeof(double));
                eng->fir_buf_sz = eng->fir_buf ? count * sizeof(double) : 0;
            }
            if (!eng->fir_buf)
                return 0;

            if (eng->lowpass.initialized) {
                /* FIR lowpass — fp64 throughout to match SDM pipeline */
                static __declspec(thread) double *tls_lp_in = NULL;
                static __declspec(thread) size_t tls_lp_sz = 0;
                if (tls_lp_sz < count) {
                    free(tls_lp_in);
                    tls_lp_in = (double *)malloc(count * sizeof(double));
                    tls_lp_sz = tls_lp_in ? count : 0;
                }
                if (!tls_lp_in) return 0;
                for (size_t i = 0; i < count; i++)
                    tls_lp_in[i] = in[i] >= 0.0f ? 1.0 : -1.0;
                fir_lowpass_process(&eng->lowpass, tls_lp_in, eng->fir_buf, count);
                /* Smooth gain ramp from previous to current */
                double g_start = (double)eng->fir_gain * eng->prev_gain;
                double g_end   = (double)eng->fir_gain * (double)cfg->gain;
                if (g_start == 0.0) g_start = g_end; /* first chunk */
                double g_step = (count > 1) ? (g_end - g_start) / (double)(count - 1) : 0.0;
                for (size_t i = 0; i < count; i++)
                    eng->fir_buf[i] *= g_start + g_step * (double)i;
                eng->prev_gain = (double)cfg->gain;
            } else {
                /* Boxcar: CPU fp64 */
                boxcar_t *bc = &eng->boxcar;
                const double inv_n = 1.0 / (double)bc->taps;
                /* Smooth gain ramp from previous to current */
                double g_start = (double)eng->fir_gain * eng->prev_gain;
                double g_end   = (double)eng->fir_gain * (double)cfg->gain;
                if (g_start == 0.0) g_start = g_end;
                double g_step = (count > 1) ? (g_end - g_start) / (double)(count - 1) : 0.0;
                for (size_t i = 0; i < count; i++) {
                    double s = in[i] >= 0.0f ? 1.0 : -1.0;
                    bc->sum -= bc->ring[bc->pos];
                    bc->ring[bc->pos] = s;
                    bc->sum += s;
                    bc->pos = (bc->pos + 1) % bc->taps;
                    eng->fir_buf[i] = bc->sum * inv_n * (g_start + g_step * (double)i);
                }
                eng->prev_gain = (double)cfg->gain;
            }
            /* Pre-SDM pre-emphasis (ML model, all DSD rates).
             * Features on CPU (subsampled), MLP via ONNX on GPU, FIR on CPU. */
            if (cfg->ml_enabled)
                engine_apply_preemph(eng, cfg, count);
            /* Convolution filter (room correction) — same-rate DSD path */
            if (eng->conv)
                conv_process(eng->conv, eng->fir_buf, count);
            /* Re-encode via SDM (CPU only).
             * Trellis SDM internally scales input by 0.5 to prevent quantizer
             * overload from FIR rate-conversion peaks (±2.24). For same-rate
             * boxcar path (output ≤ ±1.0), compensate by 2× so the 0.5
             * cancels out and the gain is unity. */
            size_t sdm_out;
            if (eng->sdm_mode == SDM_MODE_PRECORR)
                sdm_out = precorr_process_block(&eng->precorr, eng->fir_buf, out, count);
            else
                sdm_out = (eng->sdm_fast ? sdm_process_block_fast : sdm_process_block)(&eng->sdm, eng->fir_buf, out, count);
            return sdm_out;
        }
    }

    if (eng->fir_only) {
        /* DSD→PCM decimation: FIR + gain only, no SDM.
         * engine_process_fir_gain outputs double; narrow to float for output.
         * Uses IPP for SIMD-accelerated conversion when available. */
        double *fir_out_d;
        size_t fir_count = engine_process_fir_gain(eng, in, count, cfg, &fir_out_d);
        /* Convolution filter (room correction) — PCM output path */
        if (eng->conv)
            conv_process(eng->conv, fir_out_d, fir_count);
        ippsConvert_64f32f(fir_out_d, out, (int)fir_count);
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

    /* Ensure intermediate buffer (double for SDM) */
    if (eng->fir_buf_sz < buf_need * sizeof(double)) {
        free(eng->fir_buf);
        eng->fir_buf = (double *)malloc(buf_need * sizeof(double));
        eng->fir_buf_sz = eng->fir_buf ? buf_need * sizeof(double) : 0;
    }
    if (!eng->fir_buf)
        return 0;

    size_t fir_out;
    /* Gain ramp: interpolate from previous to current for smooth volume */
    double g_start = (double)eng->fir_gain * eng->prev_gain;
    double g_end   = (double)eng->fir_gain * (double)cfg->gain;
    if (g_start == 0.0) g_start = g_end;

    if (eng->fir.use_fp64) {
        /* FP64 path: widen float DSD input to double, FIR in fp64, output double directly */
        static __declspec(thread) double *tls_fir_d = NULL;
        static __declspec(thread) size_t tls_fir_d_sz = 0;
        if (tls_fir_d_sz < count) {
            free(tls_fir_d);
            tls_fir_d = (double *)malloc(count * sizeof(double));
            tls_fir_d_sz = tls_fir_d ? count : 0;
        }
        if (!tls_fir_d) return 0;

        /* Widen input: float -> double */
        for (size_t i = 0; i < count; i++)
            tls_fir_d[i] = (double)in[i];

        /* Boxcar DSD-Wide pre-smooth before FIR rate conversion.
         * Raw ±1.0 DSD through FIR produces ±2.24 Gibbs peaks.
         * Boxcar first converts DSD to smooth multi-bit (within ±1.0),
         * then FIR upsamples the smooth signal — peaks stay within
         * PreCorr/Trellis linear range. No gain reduction needed.
         * Uses the same rate-adaptive taps as same-rate boxcar. */
        {
            boxcar_t *bc = &eng->boxcar;
            if (bc->taps > 0) {
                const double inv_n = 1.0 / (double)bc->taps;
                for (size_t i = 0; i < count; i++) {
                    double s = tls_fir_d[i];
                    bc->sum -= bc->ring[bc->pos];
                    bc->ring[bc->pos] = s;
                    bc->sum += s;
                    bc->pos = (bc->pos + 1) % bc->taps;
                    tls_fir_d[i] = bc->sum * inv_n;
                }
            }
        }

        fir_out = fir_chain_process_d(&eng->fir, tls_fir_d, eng->fir_buf, count);

        /* Apply gain with smooth ramp */
        if (g_start != 1.0 || g_end != 1.0) {
            double step = (fir_out > 1) ? (g_end - g_start) / (double)(fir_out - 1) : 0.0;
            for (size_t i = 0; i < fir_out; i++)
                eng->fir_buf[i] *= g_start + step * (double)i;
        }
        eng->prev_gain = (double)cfg->gain;
    } else {
        /* FP32 path: FIR in fp32, then widen to double */
        static __declspec(thread) float *tls_fir_f = NULL;
        static __declspec(thread) size_t tls_fir_sz = 0;
        if (tls_fir_sz < buf_need) {
            free(tls_fir_f);
            tls_fir_f = (float *)malloc(buf_need * sizeof(float));
            tls_fir_sz = tls_fir_f ? buf_need : 0;
        }
        if (!tls_fir_f) return 0;

        /* Boxcar DSD-Wide pre-smooth (fp32 path) — same as fp64 path above */
        {
            boxcar_t *bc = &eng->boxcar;
            if (bc->taps > 0) {
                const double inv_n = 1.0 / (double)bc->taps;
                /* Pre-smooth into TLS buffer, then FIR processes the smooth signal */
                static __declspec(thread) float *tls_smooth = NULL;
                static __declspec(thread) size_t tls_smooth_sz = 0;
                if (tls_smooth_sz < count) {
                    free(tls_smooth);
                    tls_smooth = (float *)malloc(count * sizeof(float));
                    tls_smooth_sz = tls_smooth ? count : 0;
                }
                if (tls_smooth) {
                    for (size_t i = 0; i < count; i++) {
                        double s = (double)in[i];
                        bc->sum -= bc->ring[bc->pos];
                        bc->ring[bc->pos] = s;
                        bc->sum += s;
                        bc->pos = (bc->pos + 1) % bc->taps;
                        tls_smooth[i] = (float)(bc->sum * inv_n);
                    }
                    fir_out = fir_chain_process(&eng->fir, tls_smooth, tls_fir_f, count);
                } else {
                    fir_out = fir_chain_process(&eng->fir, in, tls_fir_f, count);
                }
            } else {
                fir_out = fir_chain_process(&eng->fir, in, tls_fir_f, count);
            }
        }

        /* Widen to double and apply gain ramp in one pass */
        {
            double step = (fir_out > 1) ? (g_end - g_start) / (double)(fir_out - 1) : 0.0;
            for (size_t i = 0; i < fir_out; i++)
                eng->fir_buf[i] = (double)tls_fir_f[i] * (g_start + step * (double)i);
        }
        eng->prev_gain = (double)cfg->gain;
    }

    /* Soft-clip FIR output to prevent Trellis SDM overload on rate conversion.
     * FIR peaks at ±2.24 (Gibbs) after gain still exceed Trellis MSA.
     * PreCorr handles overload natively — skip clip for PreCorr. */
    if (eng->sdm_mode != SDM_MODE_PRECORR) {
        const double clip_thresh = 0.95;
        const double inv_knee = 1.0 / (1.0 - clip_thresh);
        for (size_t i = 0; i < fir_out; i++) {
            double x = eng->fir_buf[i];
            if (x > clip_thresh)
                eng->fir_buf[i] = clip_thresh + (1.0 - clip_thresh) * tanh((x - clip_thresh) * inv_knee);
            else if (x < -clip_thresh)
                eng->fir_buf[i] = -clip_thresh - (1.0 - clip_thresh) * tanh((-x - clip_thresh) * inv_knee);
        }
    }

    /* Convolution filter (room correction) — rate conversion DSD path */
    if (eng->conv)
        conv_process(eng->conv, eng->fir_buf, fir_out);

    /* SDM (CPU only) */
    size_t sdm_out;
    if (eng->sdm_mode == SDM_MODE_PRECORR)
        sdm_out = precorr_process_block(&eng->precorr, eng->fir_buf, out, fir_out);
    else
        sdm_out = (eng->sdm_fast ? sdm_process_block_fast : sdm_process_block)(&eng->sdm, eng->fir_buf, out, fir_out);
    return sdm_out;
}

void engine_channel_reset(engine_channel_t *eng, bool preserve_sdm) {
    fir_chain_reset(&eng->fir);
    /* Reset boxcar (preserve taps). */
    {
        int saved_taps = eng->boxcar.taps;
        memset(&eng->boxcar, 0, sizeof(eng->boxcar));
        eng->boxcar.taps = saved_taps;
    }
    if (!eng->fir_only) {
        if (preserve_sdm && eng->sdm_mode == SDM_MODE_TRELLIS) {
            /* Anti-pop: keep Trellis SDM integrators warm across flush.
             * The candidate state and history buffers survive so the first
             * chunk after stop→play resumes without a DC step pop. */
        } else if (eng->sdm_mode == SDM_MODE_PRECORR) {
            precorr_context_reset(&eng->precorr);
        } else {
            sdm_context_reset(&eng->sdm);
        }
    }
    if (eng->lowpass.initialized)
        fir_lowpass_reset(&eng->lowpass);
    if (eng->ml_filter)
        onnx_filter_reset(eng->ml_filter);
    if (eng->conv)
        conv_reset(eng->conv);
}

void engine_channel_free(engine_channel_t *eng) {
    fir_lowpass_free(&eng->lowpass);
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
    if (eng->conv) {
        conv_free(eng->conv);
        free(eng->conv);
        eng->conv = NULL;
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
                                double **fir_out_ptr) {
    uint32_t fs_out = cfg->fs_out ? cfg->fs_out : cfg->fs_in;
    size_t fir_out_count;
    if (fs_out >= cfg->fs_in)
        fir_out_count = count * (fs_out / cfg->fs_in);
    else
        fir_out_count = count / (cfg->fs_in / fs_out);

    size_t buf_need = fir_out_count;
    if (fs_out < cfg->fs_in && count / 2 > buf_need)
        buf_need = count / 2;

    if (eng->fir_buf_sz < buf_need * sizeof(double)) {
        free(eng->fir_buf);
        eng->fir_buf = (double *)malloc(buf_need * sizeof(double));
        eng->fir_buf_sz = buf_need * sizeof(double);
    }

    size_t fir_count;
    uint32_t fs_out_actual = cfg->fs_out ? cfg->fs_out : cfg->fs_in;
    if (cfg->fs_in == fs_out_actual) {
        /* Same-rate: smooth ±1.0 → multi-bit (fp64 pipeline) */
        double gs = (double)eng->fir_gain * eng->prev_gain;
        double ge = (double)eng->fir_gain * (double)cfg->gain;
        if (gs == 0.0) gs = ge;
        double gstep = (count > 1) ? (ge - gs) / (double)(count - 1) : 0.0;
        if (eng->lowpass.initialized) {
            /* FIR lowpass fp64 */
            static __declspec(thread) double *tls_lp_in2 = NULL;
            static __declspec(thread) size_t tls_lp_sz2 = 0;
            if (tls_lp_sz2 < count) {
                free(tls_lp_in2);
                tls_lp_in2 = (double *)malloc(count * sizeof(double));
                tls_lp_sz2 = tls_lp_in2 ? count : 0;
            }
            if (tls_lp_in2) {
                for (size_t i = 0; i < count; i++)
                    tls_lp_in2[i] = in[i] >= 0.0f ? 1.0 : -1.0;
                fir_lowpass_process(&eng->lowpass, tls_lp_in2, eng->fir_buf, count);
                for (size_t i = 0; i < count; i++)
                    eng->fir_buf[i] *= gs + gstep * (double)i;
            }
        } else {
            /* Boxcar fp64 */
            boxcar_t *bc = &eng->boxcar;
            const double inv_n = 1.0 / (double)bc->taps;
            for (size_t i = 0; i < count; i++) {
                double s = in[i] >= 0.0f ? 1.0 : -1.0;
                bc->sum -= bc->ring[bc->pos];
                bc->ring[bc->pos] = s;
                bc->sum += s;
                bc->pos = (bc->pos + 1) % bc->taps;
                eng->fir_buf[i] = bc->sum * inv_n * (gs + gstep * (double)i);
            }
        }
        eng->prev_gain = (double)cfg->gain;
        /* Pre-SDM pre-emphasis (parallel/DAS path) */
        if (cfg->ml_enabled)
            engine_apply_preemph(eng, cfg, count);
        fir_count = count;
    } else if (eng->fir.use_fp64) {
        /* Rate conversion: FP64 path — widen input, FIR in fp64, output double directly */
        static __declspec(thread) double *tls_fir_d2 = NULL;
        static __declspec(thread) size_t tls_fir_d_sz2 = 0;
        if (tls_fir_d_sz2 < count) {
            free(tls_fir_d2);
            tls_fir_d2 = (double *)malloc(count * sizeof(double));
            tls_fir_d_sz2 = tls_fir_d2 ? count : 0;
        }
        if (!tls_fir_d2) { *fir_out_ptr = NULL; return 0; }
        for (size_t i = 0; i < count; i++)
            tls_fir_d2[i] = (double)in[i];
        fir_count = fir_chain_process_d(&eng->fir, tls_fir_d2, eng->fir_buf, count);
    } else {
        /* Rate conversion: FP32 path — FIR in fp32, then widen to double */
        static __declspec(thread) float *tls_fir_f2 = NULL;
        static __declspec(thread) size_t tls_fir_sz2 = 0;
        if (tls_fir_sz2 < buf_need) {
            free(tls_fir_f2);
            tls_fir_f2 = (float *)malloc(buf_need * sizeof(float));
            tls_fir_sz2 = tls_fir_f2 ? buf_need : 0;
        }
        if (!tls_fir_f2) { *fir_out_ptr = NULL; return 0; }
        fir_count = fir_chain_process(&eng->fir, in, tls_fir_f2, count);
        for (size_t i = 0; i < fir_count; i++)
            eng->fir_buf[i] = (double)tls_fir_f2[i];
    }

    /* Apply gain ramp — rate-conversion path only.
     * Same-rate path already applied gain inside the lowpass/boxcar block. */
    if (cfg->fs_in != fs_out_actual) {
        double gs2 = (double)eng->fir_gain * eng->prev_gain;
        double ge2 = (double)eng->fir_gain * (double)cfg->gain;
        if (gs2 == 0.0) gs2 = ge2;
        if (gs2 != 1.0 || ge2 != 1.0) {
            double step2 = (fir_count > 1) ? (ge2 - gs2) / (double)(fir_count - 1) : 0.0;
            for (size_t i = 0; i < fir_count; i++)
                eng->fir_buf[i] *= gs2 + step2 * (double)i;
        }
        eng->prev_gain = (double)cfg->gain;
    }

    /* Convolution is applied in dsp_plugin.c after the FIR phase completes
     * for all channels — NOT here, to avoid double-application when
     * dsp_plugin.c also hooks convolution after GPU/CPU FIR. */

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
                            const double *in, float *out,
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

    /* DSD→PCM: FIR decimation only.
     * All PCM rates < DSD_RATE_64 (2.8MHz), all DSD rates >= DSD_RATE_64. */
    if (fs_out < DSD_RATE_64 && fs_in >= DSD_RATE_64) {
        info->fir_only = true;
        info->ntf_filter = NTF_AUTO;
        info->cands = 0;
        info->lat = 0;
        return 0;
    }

    info->fir_only = false;

    /* Path-adaptive lookup — always look up for cands/depth/lat/gain.
     * NTF from path_config is only used when ntf_override == NTF_AUTO. */
    /* Look up path config for ALL SDM modes — fir_gain applies to both
     * Trellis and PreCorr (prevents SDM overload from FIR output peaks).
     * NTF/cands/depth/lat from path_config are Trellis-specific. */
    const path_config_t *pc = path_config_lookup(fs_in, fs_out);

    if (pc && sdm_mode == SDM_MODE_TRELLIS) {
        /* Trellis: use path table for NTF, cands, depth, lat, gain */
        info->ntf_filter = (int)pc->filter;
        info->cands = pc->cands;
        info->lat = pc->lat;
        info->depth = pc->depth ? pc->depth : cfg->trellis_depth;
        info->state_limit = pc->state_limit;
        info->fir_gain = pc->fir_gain;
    } else {
        /* PreCorr or no path config: use config values for SDM params */
        info->ntf_filter = ntf_override; /* keep NTF_AUTO or user value */
        info->cands = cfg->trellis_cands;
        info->lat = cfg->trellis_lat;
        /* fir_gain from path table if available, else 1.0 */
        info->fir_gain = pc ? pc->fir_gain : 1.0f;
    }

    /* Auto-compute optimal lat when lat=0 (auto).
     * From comprehensive nc×lat sweep with stability analysis (2026-03-19):
     *   DSD64:     lat=32  (110.7 dB with d=16) — d=16 needs shorter lat for best quality
     *   DSD128:    lat=128 (127.4 dB, 0 collapse) — lat=32 was only 107 dB
     *   DSD256:    lat=128 (143.5 dB, 0 collapse)
     *   DSD512:    lat=32  (137.6 dB, 0 collapse) — lat=16 was 124.8 dB */
    if (info->lat <= 0) {
        uint32_t rate = (fs_out > fs_in) ? fs_out : fs_in;
        /* DSD512 or DSD512/48 */
        if (rate >= DSD_RATE_512)
            info->lat = 32;
        /* DSD128+ or DSD128/48+ (but below DSD512) */
        else if (rate >= DSD_RATE_128)
            info->lat = 128;
        else
            info->lat = 32;   /* DSD64 or DSD64/48 (d=16 needs short lat) */
    }

    return 0;
}
