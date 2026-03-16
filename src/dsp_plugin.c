/*
 * foo_dsd_trellis — Plugin state management and chunk processing
 *
 * Provides per-instance state (no globals) so multiple fb2k DSP
 * instances can coexist. The C++ wrapper (dsp_fb2k.cpp) owns one
 * plugin_state_t per DSP instance.
 *
 * Processing flow for DoP chunks:
 *   1. Detect DoP markers in interleaved float32 PCM
 *   2. De-interleave + unpack DoP to per-channel DSD float arrays
 *   3. Dispatch channels to thread pool (FIR + gain + SDM)
 *   4. Repack per-channel DSD floats to interleaved DoP PCM
 */

#include <windows.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

#include "../include/dsd_types.h"
#include "../include/engine.h"
#include "../include/ntf.h"
#include "../include/dop.h"
#include "../include/threadpool.h"
#include "../include/cpuset.h"
#include "../include/onnx_filter.h"

/*
 * Plugin identity
 */
#define PLUGIN_NAME        "DSD Trellis SDM"

/* DoP PCM sample rates → DSD rates.
 * DoP encodes 16 DSD bits per 24-bit PCM sample.
 * PCM rate 176400 = DSD64 (176400 * 16 = 2822400).
 * PCM rate 352800 = DSD128, etc. */
#define DOP_BITS_PER_FRAME 16

/*
 * Per-instance plugin state
 */
typedef struct plugin_state {
    dsd_config_t       config;
    engine_channel_t  *channels;
    int                num_channels;
    threadpool_t      *pool;
    bool               initialized;
    uint32_t           detected_dsd_rate;
    uint32_t           active_fs_out;     /* Output rate engine was initialized with */
    float              active_gain;       /* Gain engine was initialized with */
    bool               active_mute;       /* Mute state engine was initialized with */
    int                active_sdm_mode;   /* SDM mode engine was initialized with */

    /* Per-channel buffers for unpack/repack */
    float            **ch_in;        /* [ch][dsd_samples] unpacked DSD input */
    float            **ch_out;       /* [ch][dsd_samples] processed DSD output */
    size_t             ch_buf_size;  /* Current allocation per channel (floats) */

    /* CPU topology (detected once, reused) */
    cpu_topology_t     topology;
    bool               topology_detected;

    /* Workload tracking for debug logging */
    int                last_num_threads;
    int                last_segments_per_ch;
    size_t             last_dsd_in_count;
    bool               workload_changed;
    bool               cpuset_changed;
    uint64_t           last_cpuset_mask;

    /* Selected core IDs for the current threadpool */
    uint32_t           selected_core_ids[CPUSET_MAX_CPUS];
    int                selected_core_count;

    /* CPUSET change hysteresis — avoid rebuilding threadpool on transient OS parking */
    uint64_t           pending_cpuset_mask;  /* mask we're considering switching to */
    int                cpuset_stable_count;  /* how many checks the pending mask has been stable */
    int                cpuset_check_counter; /* throttle: only check every N chunks */

    /* Per-phase timing (milliseconds) */
    double             time_unpack_ms;
    double             time_fir_ms;
    double             time_sdm_ms;
    double             time_pack_ms;

    /* Cached allocations to avoid per-chunk malloc/free */
    sdm_context_t     *cached_temp_sdms;
    int                cached_temp_sdm_count;
    channel_block_t   *cached_blocks;
    int                cached_block_count;
    float             *cached_pcm_temp;
    size_t             cached_pcm_temp_sz;

    /* Previous chunk FIR tail for parallel SDM segment 0 warmup.
     * Fixes chunk-boundary state discontinuity: after first chunk,
     * segment 0 also uses a temp SDM with warmup from this tail. */
    float            **fir_tail;        /* [ch][overlap] last FIR samples */
    float            **seg0_buf;        /* [ch][overlap + seg0_size] contiguous buffer */
    size_t             seg0_buf_sz;     /* allocated size per channel */
    size_t             fir_tail_len;    /* = overlap */
    bool               fir_tail_valid;  /* false until first chunk completes */
} plugin_state_t;

/* ─── Timing helper ─── */

static double perf_ms(LARGE_INTEGER start, LARGE_INTEGER end) {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return (double)(end.QuadPart - start.QuadPart) * 1000.0 / (double)freq.QuadPart;
}

/* ─── Helpers ─── */

/* Convert DoP PCM sample rate to DSD rate */
static uint32_t dop_pcm_to_dsd_rate(uint32_t pcm_rate) {
    uint32_t dsd_rate = pcm_rate * DOP_BITS_PER_FRAME;
    switch (dsd_rate) {
    case DSD_RATE_64:
    case DSD_RATE_128:
    case DSD_RATE_256:
    case DSD_RATE_512:
        return dsd_rate;
    default:
        return 0;  /* Not a valid DoP rate */
    }
}

/* Convert DSD rate to DoP PCM sample rate */
static uint32_t dsd_to_dop_pcm_rate(uint32_t dsd_rate) {
    return dsd_rate / DOP_BITS_PER_FRAME;
}

/* Determine target DSD rate for PCM→DSD conversion.
 * Returns 0 if the PCM rate is not supported or the ratio is not a power of 2. */
static uint32_t pcm_to_target_dsd_rate(uint32_t pcm_rate, uint32_t cfg_fs_out) {
    /* If user configured a specific output rate, check compatibility */
    if (cfg_fs_out != 0) {
        if (cfg_fs_out <= pcm_rate)
            return 0;  /* DSD rate must be higher than PCM rate */
        uint32_t ratio = cfg_fs_out / pcm_rate;
        if (cfg_fs_out != ratio * pcm_rate)
            return 0;  /* Not evenly divisible */
        /* Check power of 2 */
        if (ratio == 0 || (ratio & (ratio - 1)) != 0)
            return 0;
        return cfg_fs_out;
    }

    /* Auto-select: DSD64 base rate for the PCM rate's family */
    /* 44100 family: 44100, 88200, 176400, 352800 → DSD base = 2822400 */
    /* 48000 family: 48000, 96000, 192000, 384000 → DSD base = 3072000 */
    uint32_t dsd_base;
    if (pcm_rate % 44100 == 0)
        dsd_base = DSD_RATE_64;    /* 2822400 = 64 * 44100 */
    else if (pcm_rate % 48000 == 0)
        dsd_base = 64 * 48000;     /* 3072000 = 64 * 48000 */
    else
        return 0;  /* Unsupported PCM rate family */

    /* dsd_base must be > pcm_rate and ratio must be power of 2 */
    if (dsd_base <= pcm_rate) {
        /* PCM rate is already at or above DSD64 equivalent — use DSD128 etc. */
        /* Find smallest DSD rate > pcm_rate with power-of-2 ratio */
        uint32_t base_unit = (pcm_rate % 44100 == 0) ? 44100 : 48000;
        uint32_t mult = 64;  /* Start at DSD64 */
        while (mult <= 512) {
            uint32_t dsd_rate = mult * base_unit;
            if (dsd_rate > pcm_rate) {
                uint32_t ratio = dsd_rate / pcm_rate;
                if (dsd_rate == ratio * pcm_rate && (ratio & (ratio - 1)) == 0)
                    return dsd_rate;
            }
            mult *= 2;
        }
        return 0;
    }

    uint32_t ratio = dsd_base / pcm_rate;
    if (dsd_base != ratio * pcm_rate || (ratio & (ratio - 1)) != 0)
        return 0;
    return dsd_base;
}

/* Ensure per-channel buffers are large enough */
static int ensure_ch_bufs(plugin_state_t *s, int num_ch, size_t dsd_samples) {
    size_t needed = dsd_samples * 8;  /* worst case: 8x rate conversion */
    if (s->ch_in && s->num_channels >= num_ch && s->ch_buf_size >= needed)
        return 0;

    /* Free old */
    if (s->ch_in) {
        for (int i = 0; i < s->num_channels; i++) {
            free(s->ch_in[i]);
            free(s->ch_out[i]);
        }
        free(s->ch_in);
        free(s->ch_out);
    }

    s->ch_in  = (float **)calloc((size_t)num_ch, sizeof(float *));
    s->ch_out = (float **)calloc((size_t)num_ch, sizeof(float *));
    if (!s->ch_in || !s->ch_out)
        return -1;

    /* Allocate with headroom for rate conversion (max 8x upsample) */
    size_t alloc = dsd_samples * 8;
    for (int i = 0; i < num_ch; i++) {
        s->ch_in[i]  = (float *)malloc(alloc * sizeof(float));
        s->ch_out[i] = (float *)malloc(alloc * sizeof(float));
        if (!s->ch_in[i] || !s->ch_out[i])
            return -1;
    }

    s->ch_buf_size = alloc;
    return 0;
}

/* ─── Public API (called from C++ wrapper) ─── */

plugin_state_t *plugin_create(void) {
    plugin_state_t *s = (plugin_state_t *)calloc(1, sizeof(plugin_state_t));
    if (s)
        dsd_config_defaults(&s->config);
    return s;
}

void plugin_destroy(plugin_state_t *s) {
    if (!s)
        return;

    if (s->pool) {
        threadpool_destroy(s->pool);
        s->pool = NULL;
    }

    if (s->channels) {
        for (int i = 0; i < s->num_channels; i++)
            engine_channel_free(&s->channels[i]);
        free(s->channels);
    }

    if (s->ch_in) {
        for (int i = 0; i < s->num_channels; i++) {
            free(s->ch_in[i]);
            free(s->ch_out[i]);
        }
        free(s->ch_in);
        free(s->ch_out);
    }

    /* Free cached allocations */
    if (s->cached_temp_sdms) {
        for (int i = 0; i < s->cached_temp_sdm_count; i++)
            sdm_context_free(&s->cached_temp_sdms[i]);
        free(s->cached_temp_sdms);
    }
    free(s->cached_blocks);
    free(s->cached_pcm_temp);

    /* Free FIR tail and seg0 buffers */
    if (s->fir_tail) {
        for (int i = 0; i < s->num_channels; i++)
            free(s->fir_tail[i]);
        free(s->fir_tail);
    }
    if (s->seg0_buf) {
        for (int i = 0; i < s->num_channels; i++)
            free(s->seg0_buf[i]);
        free(s->seg0_buf);
    }

    free(s);
}

/* Resolve ML model path from DLL directory.
 * Returns true if path was built; does not check if file exists. */
static bool resolve_ml_model_path(wchar_t *path, size_t path_size) {
    HMODULE hmod = GetModuleHandleW(L"foo_dsd_trellis.dll");
    if (!hmod)
        return false;
    DWORD len = GetModuleFileNameW(hmod, path, (DWORD)path_size);
    if (len == 0 || len >= path_size)
        return false;
    /* Strip filename, keep directory */
    wchar_t *sep = wcsrchr(path, L'\\');
    if (!sep) sep = wcsrchr(path, L'/');
    if (sep)
        sep[1] = L'\0';
    else
        path[0] = L'\0';
    wcscat_s(path, path_size, L"foo_dsd_trellis_ml.onnx");
    return true;
}

/* Warm up full pipeline (FIR + SDM) with DSD silence to settle
 * both FIR ring buffers and SDM integrator states.
 * Without this, the first output samples have a startup transient (pop). */
static void plugin_warmup(plugin_state_t *s) {
    if (!s || !s->initialized || !s->channels)
        return;
    size_t warmup = 8192;
    float *sil_in  = (float *)malloc(warmup * sizeof(float));
    uint32_t fs_out = s->config.fs_out ? s->config.fs_out : s->config.fs_in;
    size_t ratio = (fs_out > s->config.fs_in) ? (fs_out / s->config.fs_in) : 1;
    size_t out_sz = warmup * ratio + 4096;
    float *sil_out = (float *)malloc(out_sz * sizeof(float));
    if (sil_in && sil_out) {
        for (size_t j = 0; j < warmup; j++) {
            uint8_t pat = (j / 8) & 1 ? 0x96u : 0x69u;
            sil_in[j] = (pat >> (7 - (j & 7))) & 1 ? 1.0f : -1.0f;
        }
        for (int i = 0; i < s->num_channels; i++)
            engine_process_block(&s->channels[i], sil_in, sil_out,
                                 warmup, &s->config);
    }
    free(sil_in);
    free(sil_out);
}

/* Initialize engine for given channel count and config.
 * Called when we first detect DSD rate in on_chunk. */
static int plugin_init_engine(plugin_state_t *s, int num_channels,
                               uint32_t dsd_rate) {
    if (s->initialized)
        return 0;

    /* Don't overwrite fs_in — caller sets it appropriately:
     * DSD path: fs_in = detected DSD rate
     * PCM path: fs_in = PCM sample rate (for FIR upsample ratio) */
    s->num_channels = num_channels;
    s->detected_dsd_rate = dsd_rate;
    s->active_fs_out = s->config.fs_out;
    s->active_gain = s->config.gain;
    s->active_mute = s->config.mute;
    s->active_sdm_mode = s->config.sdm_mode;

    s->channels = (engine_channel_t *)calloc(
        (size_t)num_channels, sizeof(engine_channel_t));
    if (!s->channels)
        return -1;

    for (int i = 0; i < num_channels; i++) {
        if (engine_channel_init(&s->channels[i], i, &s->config) != 0) {
            for (int j = 0; j < i; j++)
                engine_channel_free(&s->channels[j]);
            free(s->channels);
            s->channels = NULL;
            return -1;
        }
    }

    /* Warm up full pipeline */
    plugin_warmup(s);

    /* Detect CPU topology if not already done */
    if (!s->topology_detected) {
        cpuset_detect(&s->topology);
        cpuset_benchmark(&s->topology);
        s->topology_detected = true;
    }

    /* Select threads based on topology and config */
    uint32_t selected_ids[CPUSET_MAX_CPUS];
    int max_t = s->config.thread_count > 0 ? s->config.thread_count : 0;
    int selected = cpuset_select(&s->topology,
                                  (smt_mode_t)s->config.smt_mode,
                                  (ccd_mode_t)s->config.ccd_mode,
                                  (ecore_mode_t)s->config.ecore_mode,
                                  max_t, selected_ids, CPUSET_MAX_CPUS);

    /* Store selected core IDs for logging */
    s->selected_core_count = selected > 0 ? selected : 0;
    if (selected > 0) {
        memcpy(s->selected_core_ids, selected_ids,
               (size_t)selected * sizeof(uint32_t));
    }

    if (selected > 0 && s->topology.initialized) {
        s->pool = threadpool_create_cpuset(selected_ids, selected);
    } else {
        int tc = s->config.thread_count > 0 ? s->config.thread_count : 0;
        s->pool = threadpool_create(tc, s->config.affinity_mask);
    }
    if (!s->pool) {
        for (int i = 0; i < num_channels; i++)
            engine_channel_free(&s->channels[i]);
        free(s->channels);
        s->channels = NULL;
        return -1;
    }

    /* Create ML post-filters if enabled and ONNX Runtime is available */
    if (s->config.ml_enabled && onnx_runtime_available()) {
        wchar_t model_path[MAX_PATH];
        if (resolve_ml_model_path(model_path, MAX_PATH)) {
            uint32_t fs_out = s->config.fs_out ? s->config.fs_out : dsd_rate;
            for (int i = 0; i < num_channels; i++) {
                s->channels[i].ml_filter = onnx_filter_create(
                    model_path, fs_out, (ml_ep_t)s->config.ml_ep);
                /* NULL is fine — filter is just unavailable */
            }
        }
    }

    s->initialized = true;
    s->fir_tail_valid = false;  /* new engine — no previous FIR tail */
    return 0;
}

void plugin_set_config(plugin_state_t *s, const dsd_config_t *cfg) {
    if (!s) return;
    s->config = *cfg;
}

const dsd_config_t *plugin_get_config(const plugin_state_t *s) {
    return s ? &s->config : NULL;
}

const cpu_topology_t *plugin_get_topology(const plugin_state_t *s) {
    return (s && s->topology_detected) ? &s->topology : NULL;
}

/* Query per-phase timing for debug logging */
void plugin_get_phase_timing(const plugin_state_t *s,
                              double *unpack_ms, double *fir_ms,
                              double *sdm_ms, double *pack_ms) {
    if (!s) {
        *unpack_ms = *fir_ms = *sdm_ms = *pack_ms = 0.0;
        return;
    }
    *unpack_ms = s->time_unpack_ms;
    *fir_ms = s->time_fir_ms;
    *sdm_ms = s->time_sdm_ms;
    *pack_ms = s->time_pack_ms;
}

/* Query current workload parameters for debug logging */
void plugin_get_workload(const plugin_state_t *s,
                          int *num_threads, int *segments_per_ch,
                          bool *changed) {
    if (!s) {
        *num_threads = 0;
        *segments_per_ch = 0;
        *changed = false;
        return;
    }
    *num_threads = s->last_num_threads;
    *segments_per_ch = s->last_segments_per_ch;
    *changed = s->workload_changed;
}

/* Query selected core IDs for debug logging */
int plugin_get_selected_cores(const plugin_state_t *s,
                               uint32_t *ids, int max_ids) {
    if (!s || s->selected_core_count == 0)
        return 0;
    int n = s->selected_core_count < max_ids ? s->selected_core_count : max_ids;
    memcpy(ids, s->selected_core_ids, (size_t)n * sizeof(uint32_t));
    return n;
}

/* Query RT stress from the threadpool */
int plugin_get_stressed_worker(const plugin_state_t *s, double *ratio) {
    if (!s || !s->pool)
        return -1;
    return threadpool_get_stressed_thread(s->pool, ratio);
}

/* Query CPUSET change info for debug logging */
bool plugin_get_cpuset_change(const plugin_state_t *s, uint64_t *mask) {
    if (!s) {
        if (mask) *mask = 0;
        return false;
    }
    if (mask) *mask = s->last_cpuset_mask;
    return s->cpuset_changed;
}

/*
 * Process an interleaved DoP PCM chunk.
 *
 * in_pcm:    interleaved float32 PCM (DoP-encoded), num_channels * pcm_frames
 * out_pcm:   output buffer (caller-allocated, large enough for rate change)
 * pcm_frames: number of PCM frames (per channel) in input
 * num_channels: channel count
 * pcm_rate:  PCM sample rate (176400, 352800, etc.)
 *
 * Returns: number of output PCM frames per channel, or 0 on error/passthrough.
 *          If return is 0, caller should pass chunk unmodified.
 */
size_t plugin_process(plugin_state_t *s,
                      const float *in_pcm, float *out_pcm,
                      size_t pcm_frames, int num_channels,
                      uint32_t pcm_rate) {
    if (!s)
        return 0;

    /* Determine DSD rate from PCM rate */
    uint32_t dsd_rate = dop_pcm_to_dsd_rate(pcm_rate);
    if (dsd_rate == 0)
        return 0;  /* Not DoP, pass through */

    /* Detect DoP markers in channel 0 of interleaved data */
    if (pcm_frames >= 8 && !dop_detect_interleaved(in_pcm, pcm_frames, num_channels))
        return 0;  /* No DoP markers, pass through */

    /* Always keep fs_in in sync (plugin_set_config may have overwritten it) */
    s->config.fs_in = dsd_rate;

    /* Initialize engine on first use, channel/rate change, or output rate change */
    if (!s->initialized || s->num_channels != num_channels ||
        s->detected_dsd_rate != dsd_rate ||
        s->active_fs_out != s->config.fs_out ||
        s->active_sdm_mode != s->config.sdm_mode) {
        /* Tear down old state */
        if (s->initialized) {
            if (s->pool) {
                threadpool_destroy(s->pool);
                s->pool = NULL;
            }
            if (s->channels) {
                for (int i = 0; i < s->num_channels; i++)
                    engine_channel_free(&s->channels[i]);
                free(s->channels);
                s->channels = NULL;
            }
            s->initialized = false;
        }
        if (plugin_init_engine(s, num_channels, dsd_rate) != 0)
            return 0;
    }

    /* Check for system CPUSET changes (CPUDoc dynamic core management).
     * Only check every CPUSET_CHECK_INTERVAL chunks to avoid
     * kernel syscall overhead on every audio chunk. */
    #define CPUSET_CHECK_INTERVAL   100  /* check every ~100 chunks */
    #define CPUSET_STABLE_THRESHOLD 5    /* 5 consecutive checks stable = rebuild */
    s->cpuset_changed = false;
    s->cpuset_check_counter++;
    if (s->topology_detected && (s->cpuset_check_counter % CPUSET_CHECK_INTERVAL) == 0) {
        bool mask_changed = false;
        uint64_t new_mask = cpuset_refresh(&s->topology, &mask_changed);
        if (mask_changed && s->pool) {
            if (new_mask == s->pending_cpuset_mask) {
                s->cpuset_stable_count++;
            } else {
                s->pending_cpuset_mask = new_mask;
                s->cpuset_stable_count = 1;
            }

            if (s->cpuset_stable_count >= CPUSET_STABLE_THRESHOLD) {
                s->cpuset_changed = true;
                s->last_cpuset_mask = new_mask;
                s->cpuset_stable_count = 0;

                /* Rebuild thread pool with new set of enabled cores */
                threadpool_destroy(s->pool);
                s->pool = NULL;

                uint32_t selected_ids[CPUSET_MAX_CPUS];
                int max_t = s->config.thread_count > 0 ? s->config.thread_count : 0;
                int selected = cpuset_select(&s->topology,
                                              (smt_mode_t)s->config.smt_mode,
                                              (ccd_mode_t)s->config.ccd_mode,
                                              (ecore_mode_t)s->config.ecore_mode,
                                              max_t, selected_ids, CPUSET_MAX_CPUS);
                /* Update selected core IDs for logging */
                s->selected_core_count = selected > 0 ? selected : 0;
                if (selected > 0) {
                    memcpy(s->selected_core_ids, selected_ids,
                           (size_t)selected * sizeof(uint32_t));
                    s->pool = threadpool_create_cpuset(selected_ids, selected);
                }
                if (!s->pool)
                    s->pool = threadpool_create(
                        s->config.thread_count > 0 ? s->config.thread_count : 0,
                        s->config.affinity_mask);
            }
        } else {
            /* Mask didn't change — reset hysteresis counter */
            s->cpuset_stable_count = 0;
        }
    }

    /* DSD samples per channel = PCM frames * 16 bits/frame */
    size_t dsd_in_count = pcm_frames * DOP_BITS_PER_FRAME;

    /* Ensure per-channel buffers */
    if (ensure_ch_bufs(s, num_channels, dsd_in_count) != 0)
        return 0;

    /* De-interleave and unpack DoP for each channel.
     * Input is interleaved: [ch0_f0, ch1_f0, ch0_f1, ch1_f1, ...]
     * Each channel's PCM frames are at stride = num_channels. */
    for (int ch = 0; ch < num_channels; ch++) {
        /* Extract this channel's PCM frames (strided) into a temp buffer */
        float *ch_pcm = s->ch_in[ch];  /* Reuse as temp for PCM frames */
        for (size_t f = 0; f < pcm_frames; f++)
            ch_pcm[f] = in_pcm[f * (size_t)num_channels + (size_t)ch];

        /* Unpack DoP: each PCM frame → 16 DSD bits as ±1.0 floats.
         * Output goes into ch_in[ch], starting at offset pcm_frames
         * (since we used the beginning for temp PCM storage). */
        /* Actually, we need separate space. Use ch_out as temp for PCM,
         * unpack into ch_in. */
    }

    /* Better approach: allocate a small temp buffer for per-channel PCM,
     * then unpack into ch_in. */
    LARGE_INTEGER t_start, t_end;
    QueryPerformanceCounter(&t_start);

    /* Ensure cached pcm_temp buffer is large enough */
    if (s->cached_pcm_temp_sz < pcm_frames) {
        free(s->cached_pcm_temp);
        s->cached_pcm_temp = (float *)malloc(pcm_frames * sizeof(float));
        s->cached_pcm_temp_sz = s->cached_pcm_temp ? pcm_frames : 0;
    }
    float *pcm_temp = s->cached_pcm_temp;
    if (!pcm_temp)
        return 0;

    for (int ch = 0; ch < num_channels; ch++) {
        /* Extract strided PCM for this channel */
        for (size_t f = 0; f < pcm_frames; f++)
            pcm_temp[f] = in_pcm[f * (size_t)num_channels + (size_t)ch];

        /* Unpack DoP to DSD floats */
        dop_unpack(pcm_temp, s->ch_in[ch], pcm_frames);
    }

    QueryPerformanceCounter(&t_end);
    s->time_unpack_ms = perf_ms(t_start, t_end);

    /* Determine if parallel SDM segmentation is beneficial.
     * Worth it only when rate-converting (SDM is the bottleneck)
     * and we have more threads than channels. */
    bool need_rate_conv = !s->channels[0].passthrough && !s->config.mute;
    int num_threads = threadpool_get_thread_count(s->pool);
    int segments_per_ch = 1;
    size_t overlap = 0;

    if (need_rate_conv && num_threads > num_channels &&
        s->config.sdm_mode == SDM_MODE_TRELLIS) {
        uint32_t fs_out = s->config.fs_out ? s->config.fs_out : s->config.fs_in;
        size_t fir_out_est = dsd_in_count;
        if (fs_out > s->config.fs_in)
            fir_out_est = dsd_in_count * (fs_out / s->config.fs_in);

        overlap = 2 * (size_t)s->config.trellis_lat;
        segments_per_ch = num_threads / num_channels;
        if (segments_per_ch < 1) segments_per_ch = 1;
        if (segments_per_ch > 4) segments_per_ch = 4;

        /* Ensure minimum segment size (at least 4x overlap) */
        size_t min_seg = overlap * 4;
        if (min_seg < 256) min_seg = 256;
        while (segments_per_ch > 1 && fir_out_est / (size_t)segments_per_ch < min_seg)
            segments_per_ch--;
    }

    /* Track workload changes — only log on thread count change,
     * NOT on segment count fluctuations (those are normal). */
    s->workload_changed = (num_threads != s->last_num_threads);
    s->last_num_threads = num_threads;
    s->last_segments_per_ch = segments_per_ch;
    s->last_dsd_in_count = dsd_in_count;

    size_t dsd_out_count;

    if (segments_per_ch > 1) {
        /* === Parallel SDM path: FIR+gain sequential, SDM parallel === */

        /* Phase 1: FIR + gain per channel (parallel via threadpool) */
        LARGE_INTEGER t_fir_start, t_fir_end;
        QueryPerformanceCounter(&t_fir_start);

        float *fir_data[32];  /* max 32 channels */
        size_t fir_counts[32];

        /* Submit FIR blocks to threadpool for parallel processing */
        channel_block_t fir_blocks[32];
        memset(fir_blocks, 0, (size_t)num_channels * sizeof(channel_block_t));
        for (int ch = 0; ch < num_channels; ch++) {
            fir_blocks[ch].mode    = BLOCK_MODE_FIR;
            fir_blocks[ch].eng     = &s->channels[ch];
            fir_blocks[ch].in      = s->ch_in[ch];
            fir_blocks[ch].count   = dsd_in_count;
            fir_blocks[ch].cfg     = &s->config;
            fir_blocks[ch].channel = ch;
            threadpool_submit(s->pool, &fir_blocks[ch]);
        }
        threadpool_wait(s->pool);

        for (int ch = 0; ch < num_channels; ch++) {
            fir_counts[ch] = fir_blocks[ch].out_count;
            fir_data[ch] = fir_blocks[ch].fir_out;
        }

        QueryPerformanceCounter(&t_fir_end);
        s->time_fir_ms = perf_ms(t_fir_start, t_fir_end);

        /* Phase 2: Get temp SDM contexts.
         * When fir_tail is available (chunk 2+), ALL segments use temp SDMs.
         * This avoids persistent SDM state discontinuity at chunk boundaries. */
        bool use_fir_tail = s->fir_tail_valid && s->fir_tail_len == overlap;
        int temps_per_ch = use_fir_tail ? segments_per_ch : (segments_per_ch - 1);
        int temp_sdm_count = num_channels * temps_per_ch;

        uint32_t fs_out = s->config.fs_out ? s->config.fs_out : s->config.fs_in;
        const ntf_filter_t *filter = NULL;
        if (s->config.ntf_filter == NTF_AUTO)
            filter = ntf_auto_select(fs_out);
        else
            filter = ntf_get_filter((ntf_filter_id_t)s->config.ntf_filter, fs_out);

        /* Grow cached temp SDMs if needed */
        if (s->cached_temp_sdm_count < temp_sdm_count) {
            if (s->cached_temp_sdms) {
                for (int i = 0; i < s->cached_temp_sdm_count; i++)
                    sdm_context_free(&s->cached_temp_sdms[i]);
                free(s->cached_temp_sdms);
            }
            s->cached_temp_sdms = (sdm_context_t *)calloc(
                (size_t)temp_sdm_count, sizeof(sdm_context_t));
            s->cached_temp_sdm_count = s->cached_temp_sdms ? temp_sdm_count : 0;
        }
        sdm_context_t *temp_sdms = s->cached_temp_sdms;
        if (!temp_sdms)
            return 0;

        /* Reset temp SDM contexts (init once, reset each chunk). */
        bool init_ok = true;
        for (int i = 0; i < temp_sdm_count; i++) {
            if (temp_sdms[i].filter == NULL) {
                if (sdm_context_init(&temp_sdms[i], filter,
                                      s->config.trellis_depth,
                                      s->config.trellis_cands,
                                      s->config.trellis_lat) != 0) {
                    init_ok = false;
                    break;
                }
            } else {
                sdm_context_reset(&temp_sdms[i]);
            }
        }

        if (!init_ok)
            return 0;

        /* Phase 3: Setup and submit SDM segment blocks */
        int total_blocks = num_channels * segments_per_ch;
        if (s->cached_block_count < total_blocks) {
            free(s->cached_blocks);
            s->cached_blocks = (channel_block_t *)calloc(
                (size_t)total_blocks, sizeof(channel_block_t));
            s->cached_block_count = s->cached_blocks ? total_blocks : 0;
        }
        channel_block_t *blocks = s->cached_blocks;
        if (!blocks)
            return 0;
        memset(blocks, 0, (size_t)total_blocks * sizeof(channel_block_t));

        size_t discard = (size_t)s->config.trellis_lat;

        /* Allocate FIR tail and seg0 buffers if needed */
        if (!s->fir_tail && num_channels > 0) {
            s->fir_tail = (float **)calloc((size_t)num_channels, sizeof(float *));
            s->seg0_buf = (float **)calloc((size_t)num_channels, sizeof(float *));
        }

        for (int ch = 0; ch < num_channels; ch++) {
            size_t fir_count = fir_counts[ch];
            size_t base_seg = fir_count / (size_t)segments_per_ch;
            size_t remainder = fir_count % (size_t)segments_per_ch;
            size_t seg0_size = base_seg + (0 < remainder ? 1 : 0);

            int bi = ch * segments_per_ch;

            if (use_fir_tail) {
                /* All-temp mode: segment 0 uses temp SDM with warmup from
                 * previous chunk's FIR tail. No persistent SDM state needed. */
                size_t seg0_total = seg0_size + overlap;

                /* Grow seg0_buf if needed */
                if (s->seg0_buf_sz < seg0_total) {
                    for (int c = 0; c < num_channels; c++) {
                        free(s->seg0_buf[c]);
                        s->seg0_buf[c] = (float *)malloc(seg0_total * sizeof(float));
                    }
                    s->seg0_buf_sz = seg0_total;
                }

                /* Build contiguous buffer: prev_fir_tail + seg0_data */
                memcpy(s->seg0_buf[ch], s->fir_tail[ch], overlap * sizeof(float));
                memcpy(s->seg0_buf[ch] + overlap, fir_data[ch], seg0_size * sizeof(float));

                int temp_idx = ch * temps_per_ch + 0;
                blocks[bi].mode     = BLOCK_MODE_SDM;
                blocks[bi].sdm_ctx  = &temp_sdms[temp_idx];
                blocks[bi].in       = s->seg0_buf[ch];
                blocks[bi].out      = s->ch_out[ch];
                blocks[bi].count    = seg0_total;
                blocks[bi].discard  = discard;
                blocks[bi].channel  = ch;
            } else {
                /* First chunk: use persistent SDM (handles initial latency fill) */
                blocks[bi].mode     = BLOCK_MODE_SDM;
                blocks[bi].sdm_ctx  = &s->channels[ch].sdm;
                blocks[bi].in       = fir_data[ch];
                blocks[bi].out      = s->ch_out[ch];
                blocks[bi].count    = seg0_size;
                blocks[bi].discard  = 0;
                blocks[bi].channel  = ch;
            }

            /* Compute segment 0 output size for offset calculation */
            size_t seg0_out;
            if (use_fir_tail) {
                /* Temp SDM: overlap warmup, so output = seg0_size */
                seg0_out = seg0_size;
            } else {
                /* Persistent SDM: subtract remaining latency fill */
                size_t actual_lat = s->channels[ch].sdm.trellis_lat;
                if (actual_lat == 0) actual_lat = (size_t)s->config.trellis_lat;
                size_t pending = s->channels[ch].sdm.pending;
                size_t lat_rem = (pending < actual_lat) ? (actual_lat - pending) : 0;
                seg0_out = (seg0_size > lat_rem) ? (seg0_size - lat_rem) : 0;
            }

            /* Segments 1..N-1: temp SDM contexts with overlap warmup */
            size_t seg_start = seg0_size;
            size_t out_offset = seg0_out;

            for (int seg = 1; seg < segments_per_ch; seg++) {
                size_t this_seg;
                if (seg == segments_per_ch - 1)
                    this_seg = fir_count - seg_start;
                else
                    this_seg = base_seg + ((size_t)seg < remainder ? 1 : 0);

                int temp_idx = ch * temps_per_ch + (use_fir_tail ? seg : (seg - 1));
                int block_idx = bi + seg;

                blocks[block_idx].mode     = BLOCK_MODE_SDM;
                blocks[block_idx].sdm_ctx  = &temp_sdms[temp_idx];
                blocks[block_idx].in       = fir_data[ch] + seg_start - overlap;
                blocks[block_idx].out      = s->ch_out[ch] + out_offset;
                blocks[block_idx].count    = this_seg + overlap;
                blocks[block_idx].discard  = discard;
                blocks[block_idx].channel  = ch;

                seg_start += this_seg;
                out_offset += this_seg;
            }
        }

        LARGE_INTEGER t_sdm_start, t_sdm_end;
        QueryPerformanceCounter(&t_sdm_start);

        for (int i = 0; i < total_blocks; i++)
            threadpool_submit(s->pool, &blocks[i]);
        threadpool_wait(s->pool);

        QueryPerformanceCounter(&t_sdm_end);
        s->time_sdm_ms = perf_ms(t_sdm_start, t_sdm_end);

        /* Sum output counts per channel */
        dsd_out_count = 0;
        for (int seg = 0; seg < segments_per_ch; seg++)
            dsd_out_count += blocks[seg].out_count;  /* channel 0 */

        /* Save FIR tail for next chunk's segment 0 warmup.
         * This enables all-temp SDM mode from chunk 2 onward,
         * eliminating the persistent SDM state gap at chunk boundaries. */
        if (s->fir_tail) {
            for (int ch = 0; ch < num_channels; ch++) {
                size_t fir_count = fir_counts[ch];
                if (fir_count >= overlap) {
                    if (!s->fir_tail[ch])
                        s->fir_tail[ch] = (float *)malloc(overlap * sizeof(float));
                    if (s->fir_tail[ch])
                        memcpy(s->fir_tail[ch],
                               fir_data[ch] + fir_count - overlap,
                               overlap * sizeof(float));
                }
            }
            s->fir_tail_len = overlap;
            s->fir_tail_valid = true;
        }
    } else {
        /* === Sequential path: dispatch full blocks per channel === */
        channel_block_t *blocks = (channel_block_t *)calloc(
            (size_t)num_channels, sizeof(channel_block_t));
        if (!blocks)
            return 0;

        for (int ch = 0; ch < num_channels; ch++) {
            blocks[ch].in       = s->ch_in[ch];
            blocks[ch].out      = s->ch_out[ch];
            blocks[ch].count    = dsd_in_count;
            blocks[ch].out_count = 0;
            blocks[ch].channel  = ch;
            blocks[ch].eng      = &s->channels[ch];
            blocks[ch].cfg      = &s->config;
            blocks[ch].mode     = BLOCK_MODE_FULL;
            threadpool_submit(s->pool, &blocks[ch]);
        }
        threadpool_wait(s->pool);

        dsd_out_count = blocks[0].out_count;
        free(blocks);
    }

    if (dsd_out_count == 0)
        return 0;

    LARGE_INTEGER t_pack_start, t_pack_end;
    QueryPerformanceCounter(&t_pack_start);

    /* Check if FIR-only mode (DSD→PCM decimation) */
    bool fir_only = (s->channels && s->channels[0].fir_only);

    if (fir_only) {
        /* DSD→PCM: output is already multi-bit PCM, just interleave */
        for (int ch = 0; ch < num_channels; ch++) {
            for (size_t f = 0; f < dsd_out_count; f++)
                out_pcm[f * (size_t)num_channels + (size_t)ch] = s->ch_out[ch][f];
        }
        QueryPerformanceCounter(&t_pack_end);
        s->time_pack_ms = perf_ms(t_pack_start, t_pack_end);
        return dsd_out_count;
    }

    /* Pack DSD to DoP and interleave. ASIO+DSD output plugin detects
     * DoP markers and sends native DSD to the driver. */
    size_t out_pcm_frames = dsd_out_count / DOP_BITS_PER_FRAME;
    if (out_pcm_frames == 0)
        return 0;

    if (s->cached_pcm_temp_sz < out_pcm_frames) {
        free(s->cached_pcm_temp);
        s->cached_pcm_temp = (float *)malloc(out_pcm_frames * sizeof(float));
        s->cached_pcm_temp_sz = s->cached_pcm_temp ? out_pcm_frames : 0;
    }
    float *pcm_temp2 = s->cached_pcm_temp;
    if (!pcm_temp2)
        return 0;

    for (int ch = 0; ch < num_channels; ch++) {
        dop_pack(s->ch_out[ch], pcm_temp2, dsd_out_count);
        for (size_t f = 0; f < out_pcm_frames; f++)
            out_pcm[f * (size_t)num_channels + (size_t)ch] = pcm_temp2[f];
    }

    QueryPerformanceCounter(&t_pack_end);
    s->time_pack_ms = perf_ms(t_pack_start, t_pack_end);

    return out_pcm_frames;
}

/*
 * Process an interleaved PCM chunk and convert to DoP DSD output.
 *
 * in_pcm:      interleaved float32 PCM, num_channels * pcm_frames
 * out_pcm:     output buffer (caller-allocated, large enough for DSD output)
 * pcm_frames:  number of PCM frames (per channel) in input
 * num_channels: channel count
 * pcm_rate:    PCM sample rate (44100, 48000, 88200, 96000, etc.)
 *
 * Returns: number of output PCM frames per channel (DoP), or 0 on error.
 */
size_t plugin_process_pcm(plugin_state_t *s,
                           const float *in_pcm, float *out_pcm,
                           size_t pcm_frames, int num_channels,
                           uint32_t pcm_rate) {
    if (!s)
        return 0;

    /* Determine target DSD rate */
    uint32_t dsd_rate = pcm_to_target_dsd_rate(pcm_rate, s->config.fs_out);
    if (dsd_rate == 0)
        return 0;  /* Unsupported rate combination */

    /* Always keep fs_in/fs_out in sync (plugin_set_config may have overwritten them) */
    s->config.fs_in = pcm_rate;
    s->config.fs_out = dsd_rate;

    /* Initialize engine on first use or parameter change */
    if (!s->initialized || s->num_channels != num_channels ||
        s->detected_dsd_rate != dsd_rate ||
        s->active_fs_out != s->config.fs_out ||
        s->active_sdm_mode != s->config.sdm_mode) {
        /* Tear down old state */
        if (s->initialized) {
            if (s->pool) {
                threadpool_destroy(s->pool);
                s->pool = NULL;
            }
            if (s->channels) {
                for (int i = 0; i < s->num_channels; i++)
                    engine_channel_free(&s->channels[i]);
                free(s->channels);
                s->channels = NULL;
            }
            s->initialized = false;
        }
        if (plugin_init_engine(s, num_channels, dsd_rate) != 0)
            return 0;
    }

    /* FIR upsample ratio */
    uint32_t ratio = dsd_rate / pcm_rate;
    size_t dsd_out_count = pcm_frames * ratio;

    /* Ensure per-channel buffers */
    if (ensure_ch_bufs(s, num_channels, dsd_out_count) != 0)
        return 0;

    LARGE_INTEGER t_start, t_end;
    QueryPerformanceCounter(&t_start);

    /* De-interleave PCM input to per-channel arrays */
    for (int ch = 0; ch < num_channels; ch++) {
        for (size_t f = 0; f < pcm_frames; f++)
            s->ch_in[ch][f] = in_pcm[f * (size_t)num_channels + (size_t)ch];
    }

    QueryPerformanceCounter(&t_end);
    s->time_unpack_ms = perf_ms(t_start, t_end);

    /* Process each channel: FIR upsample + gain + SDM */
    channel_block_t *blocks = (channel_block_t *)calloc(
        (size_t)num_channels, sizeof(channel_block_t));
    if (!blocks)
        return 0;

    for (int ch = 0; ch < num_channels; ch++) {
        blocks[ch].in       = s->ch_in[ch];
        blocks[ch].out      = s->ch_out[ch];
        blocks[ch].count    = pcm_frames;
        blocks[ch].out_count = 0;
        blocks[ch].channel  = ch;
        blocks[ch].eng      = &s->channels[ch];
        blocks[ch].cfg      = &s->config;
        blocks[ch].mode     = BLOCK_MODE_FULL;
        threadpool_submit(s->pool, &blocks[ch]);
    }
    threadpool_wait(s->pool);

    size_t sdm_out_count = blocks[0].out_count;
    free(blocks);

    if (sdm_out_count == 0)
        return 0;

    /* Pack DSD to DoP and interleave. */
    size_t out_pcm_frames = sdm_out_count / DOP_BITS_PER_FRAME;
    if (out_pcm_frames == 0)
        return 0;

    LARGE_INTEGER t_pack_start, t_pack_end;
    QueryPerformanceCounter(&t_pack_start);

    if (s->cached_pcm_temp_sz < out_pcm_frames) {
        free(s->cached_pcm_temp);
        s->cached_pcm_temp = (float *)malloc(out_pcm_frames * sizeof(float));
        s->cached_pcm_temp_sz = s->cached_pcm_temp ? out_pcm_frames : 0;
    }
    float *pcm_temp = s->cached_pcm_temp;
    if (!pcm_temp)
        return 0;

    for (int ch = 0; ch < num_channels; ch++) {
        dop_pack(s->ch_out[ch], pcm_temp, sdm_out_count);
        for (size_t f = 0; f < out_pcm_frames; f++)
            out_pcm[f * (size_t)num_channels + (size_t)ch] = pcm_temp[f];
    }

    QueryPerformanceCounter(&t_pack_end);
    s->time_pack_ms = perf_ms(t_pack_start, t_pack_end);

    return out_pcm_frames;
}

/*
 * Drain remaining SDM latency at end of playback.
 * Returns number of output PCM frames per channel.
 */
size_t plugin_drain(plugin_state_t *s, float *out_pcm, int num_channels) {
    if (!s || !s->initialized || !s->channels)
        return 0;

    /* PreCorr has no latency — nothing to drain */
    if (s->config.sdm_mode == SDM_MODE_PRECORR)
        return 0;

    /* Drain each channel's SDM */
    size_t max_drain = (size_t)s->config.trellis_lat;
    float *drain_buf = (float *)malloc(max_drain * sizeof(float));
    if (!drain_buf)
        return 0;

    size_t min_drained = max_drain;

    float *pcm_temp = (float *)malloc((max_drain / DOP_BITS_PER_FRAME + 1) * sizeof(float));
    if (!pcm_temp) {
        free(drain_buf);
        return 0;
    }

    for (int ch = 0; ch < num_channels; ch++) {
        size_t drained = sdm_drain(&s->channels[ch].sdm, drain_buf, max_drain);
        if (drained < min_drained)
            min_drained = drained;

        /* Pack to DoP and interleave */
        size_t pcm_frames = drained / DOP_BITS_PER_FRAME;
        dop_pack(drain_buf, pcm_temp, drained);

        for (size_t f = 0; f < pcm_frames; f++)
            out_pcm[f * (size_t)num_channels + (size_t)ch] = pcm_temp[f];
    }

    size_t out_frames = min_drained / DOP_BITS_PER_FRAME;

    free(drain_buf);
    free(pcm_temp);

    return out_frames;
}

/* Get processing latency in seconds */
double plugin_get_latency(const plugin_state_t *s) {
    if (!s || !s->initialized || s->detected_dsd_rate == 0)
        return 0.0;

    uint32_t fs_out = s->config.fs_out ? s->config.fs_out : s->config.fs_in;

    /* SDM latency (trellis only) */
    double sdm_lat = 0.0;
    if (s->config.sdm_mode != SDM_MODE_PRECORR)
        sdm_lat = (double)s->config.trellis_lat / (double)fs_out;

    /* Processing buffer: report extra latency so fb2k prefetches
     * more audio, preventing output underruns during heavy SDM work.
     * Scale with output rate — DSD512 needs more buffer than DSD64. */
    return sdm_lat;
}

/* Reset all channel states (on seek / discontinuity) */
void plugin_flush(plugin_state_t *s) {
    if (!s || !s->initialized)
        return;
    for (int i = 0; i < s->num_channels; i++)
        engine_channel_reset(&s->channels[i]);
    /* SDM state is preserved (not reset) to avoid startup transient pop.
     * Only FIR/boxcar are reset. FIR tail is invalidated. */
    s->fir_tail_valid = false;
}

/* Reconfigure with new settings */
int plugin_reconfigure(plugin_state_t *s, const dsd_config_t *cfg) {
    if (!s)
        return -1;

    s->config = *cfg;

    if (!s->initialized)
        return 0;

    for (int i = 0; i < s->num_channels; i++) {
        if (engine_channel_reconfigure(&s->channels[i], cfg) != 0)
            return -1;
    }

    return 0;
}
