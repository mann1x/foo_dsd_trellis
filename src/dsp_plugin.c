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
#include "../include/httpapi.h"
#include "../include/fir.h"
#include "../include/precorr.h"
#include "../include/resample.h"

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
    threadpool_t      *pool;         /* SDM/FIR worker pool (pinned to selected cores) */
    threadpool_t      *io_pool;      /* Dedicated unpack/pack pool (2 threads) */
    bool               initialized;
    uint32_t           detected_dsd_rate;
    uint32_t           active_fs_out;     /* Output rate engine was initialized with */
    float              active_gain;       /* Gain engine was initialized with */
    bool               active_mute;       /* Mute state engine was initialized with */
    int                active_sdm_mode;   /* SDM mode engine was initialized with */
    int                active_cands;      /* Trellis cands engine was initialized with */
    int                active_depth;      /* Trellis depth engine was initialized with */
    bool               active_gpu;        /* GPU state engine was initialized with */
    bool               needs_warmup;      /* true after flush — prime SDM with real audio */
    int                chunk_counter;     /* chunks since init — for startup pacing */

    /* Per-channel buffers for unpack/repack */
    float            **ch_in;        /* [ch][dsd_samples] unpacked DSD input */
    float            **ch_out;       /* [ch][dsd_samples] processed DSD output */
    size_t             ch_buf_size;  /* Current allocation per channel (floats) */
    int                dop_marker_phase; /* DoP marker A/B phase for chunk continuity */

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

    /* Background CPU monitor (owns load updates + CPUSET refresh) */
    cpuset_monitor_t  *cpu_monitor;

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
    uint8_t           *cached_pcm_temp;     /* i24 per-channel pack buffer */
    size_t             cached_pcm_temp_sz;  /* frames capacity */

    /* Cached per-channel pack temp buffers (avoid per-chunk malloc/free) */
    uint8_t          **cached_pack_temps;
    size_t             cached_pack_temps_sz;  /* frames per channel */
    int                cached_pack_temps_n;   /* number of channels */

    /* Cached segment output buffers for overlap stitching (avoid per-chunk malloc) */
    float            **cached_seg_bufs;
    size_t             cached_seg_buf_sz;    /* per-buffer capacity (samples) */
    int                cached_seg_buf_count; /* number of buffers */

    /* GPU DAS cached buffers (avoid per-chunk malloc/free) */
    float             *gpu_das_in;       /* fp32 input [num_ch × count] */
    float             *gpu_das_out;      /* fp32 output [num_ch × count] */
    size_t             gpu_das_buf_cap;  /* allocated capacity per channel */
    int                gpu_das_buf_nch;  /* allocated channel count */

    /* GPU compute context (shared across all channels, NULL if disabled) */
    gpu_context_t     *gpu;

    /* Previous chunk FIR tail for parallel SDM segment 0 warmup.
     * Fixes chunk-boundary state discontinuity: after first chunk,
     * segment 0 also uses a temp SDM with warmup from this tail. */
    double           **fir_tail;        /* [ch][overlap] last FIR samples (fp64) */
    double           **seg0_buf;        /* [ch][overlap + seg0_size] contiguous buffer (fp64) */
    size_t             seg0_buf_sz;     /* allocated size per channel */
    size_t             fir_tail_len;    /* = overlap */
    bool               fir_tail_valid;  /* false until first chunk completes */

    /* ── Worker migration (dry-run probe) ── */
    struct {
        bool     active;              /* probe in flight */
        int      stressed_worker;     /* worker being evaluated */
        uint32_t original_cpuset;     /* so we can revert on abandon */
        uint32_t candidate_cpuset;    /* core being tested */
        double   pre_avg_rt;          /* avg RT% before migration */
        int      probe_chunks;        /* chunks remaining in probe */
        int      cooldown_chunks;     /* chunks before next attempt */
    } migration;
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
    case DSD48_RATE_64:
    case DSD48_RATE_128:
    case DSD48_RATE_256:
    case DSD48_RATE_512:
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
    if (!s) return NULL;
    dsd_config_defaults(&s->config);

    /* Eagerly detect topology + create threadpool at plugin load time.
     * This avoids 150-580ms cold start on first audio chunk. */
    cpuset_detect(&s->topology);
    cpuset_benchmark(&s->topology);
    for (int i = 0; i < 5; i++) {
        cpuset_update_load(&s->topology);
        Sleep(100);
    }
    s->topology_detected = true;

    s->cpu_monitor = cpuset_monitor_create(&s->topology, 750, 30);
    if (s->cpu_monitor)
        cpuset_monitor_read(s->cpu_monitor, &s->topology);

    uint32_t selected_ids[CPUSET_MAX_CPUS];
    uint8_t  selected_lps[CPUSET_MAX_CPUS];
    uint16_t selected_groups[CPUSET_MAX_CPUS];
    int max_t = s->config.thread_count > 0 ? s->config.thread_count : 0;
    int selected = cpuset_select(&s->topology,
                                  (smt_mode_t)s->config.smt_mode,
                                  (ccd_mode_t)s->config.ccd_mode,
                                  (ecore_mode_t)s->config.ecore_mode,
                                  max_t, selected_ids, selected_lps,
                                  selected_groups, CPUSET_MAX_CPUS);
    s->selected_core_count = selected > 0 ? selected : 0;
    if (selected > 0) {
        memcpy(s->selected_core_ids, selected_ids,
               (size_t)selected * sizeof(uint32_t));
        s->pool = threadpool_create_cpuset(selected_ids, selected_lps,
                                            selected_groups, selected);
    }
    if (!s->pool)
        s->pool = threadpool_create(
            s->config.thread_count > 0 ? s->config.thread_count : 0,
            s->config.affinity_mask);

    /* Eagerly create GPU context (device + shader compilation = 400ms+).
     * Create unconditionally — config.gpu_enabled isn't loaded yet
     * (fb2k applies config after plugin_create via plugin_set_config).
     * If GPU is later disabled, the context sits idle (no overhead). */
    {
        gpu_backend_t be = GPU_BACKEND_AUTO;
        if (gpu_available(be))
            s->gpu = gpu_create(be);
    }

    return s;
}

void plugin_destroy(plugin_state_t *s) {
    if (!s)
        return;

    /* Allow system to idle/sleep again */
    SetThreadExecutionState(ES_CONTINUOUS);

    /* Stop background CPU monitor first (before threadpool) */
    if (s->cpu_monitor) {
        cpuset_monitor_destroy(s->cpu_monitor);
        s->cpu_monitor = NULL;
    }

    if (s->pool) {
        threadpool_destroy(s->pool);
        s->pool = NULL;
    }
    if (s->io_pool) {
        threadpool_destroy(s->io_pool);
        s->io_pool = NULL;
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

    /* Free cached pack temp buffers */
    if (s->cached_pack_temps) {
        for (int i = 0; i < s->cached_pack_temps_n; i++)
            free(s->cached_pack_temps[i]);
        free(s->cached_pack_temps);
    }

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

    /* Destroy GPU compute context */
    if (s->gpu) {
        gpu_destroy(s->gpu);
        s->gpu = NULL;
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
    s->active_cands = s->config.trellis_cands;
    s->active_depth = s->config.trellis_depth;
    s->active_gpu = s->config.gpu_enabled;

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

    /* No silence warmup — real audio warmup happens on first chunk
     * via engine_channel_warmup() (needs_warmup flag). */

    /* Detect CPU topology if not already done.
     * Use static cache so topology survives DSP instance destroy/recreate
     * (fb2k destroys DSP on stop, recreates on play). Without caching,
     * the 500ms load seeding blocks the audio thread on every play → pop. */
    {
        static cpu_topology_t s_cached_topology = {0};
        static bool s_topology_cached = false;

        if (!s->topology_detected) {
            if (s_topology_cached) {
                /* Reuse cached topology — no blocking */
                memcpy(&s->topology, &s_cached_topology, sizeof(cpu_topology_t));
                /* Re-seed monitor pointer (not transferable) */
                s->topology.initialized = s_cached_topology.initialized;
            } else {
                /* First-ever detection: full init + load seeding */
                cpuset_detect(&s->topology);
                cpuset_benchmark(&s->topology);
                for (int i = 0; i < 5; i++) {
                    cpuset_update_load(&s->topology);
                    Sleep(100);
                }
                memcpy(&s_cached_topology, &s->topology, sizeof(cpu_topology_t));
                s_topology_cached = true;
            }
            s->topology_detected = true;
        }
    }

    /* Start background CPU monitor (load + CPUSET refresh on dedicated thread).
     * First call: creates monitor. Subsequent calls: read latest snapshot. */
    if (!s->cpu_monitor && s->topology.initialized)
        s->cpu_monitor = cpuset_monitor_create(&s->topology, 750, 30);

    /* Read latest load snapshot from background monitor (never blocks) */
    if (s->cpu_monitor)
        cpuset_monitor_read(s->cpu_monitor, &s->topology);

    extern void trellis_log_c(const char *);

    /* Select threads based on topology, load, and config */
    uint32_t selected_ids[CPUSET_MAX_CPUS];
    uint8_t  selected_lps[CPUSET_MAX_CPUS];
    uint16_t selected_groups[CPUSET_MAX_CPUS];
    int max_t = s->config.thread_count > 0 ? s->config.thread_count : 0;
    int selected = cpuset_select(&s->topology,
                                  (smt_mode_t)s->config.smt_mode,
                                  (ccd_mode_t)s->config.ccd_mode,
                                  (ecore_mode_t)s->config.ecore_mode,
                                  max_t, selected_ids, selected_lps,
                                  selected_groups, CPUSET_MAX_CPUS);

    /* Store selected core IDs for logging */
    s->selected_core_count = selected > 0 ? selected : 0;
    if (selected > 0) {
        memcpy(s->selected_core_ids, selected_ids,
               (size_t)selected * sizeof(uint32_t));
    }

    /* Reuse existing threadpool if available (avoids 500ms+ cold start
     * from 16x CreateThread + SetThreadAffinityMask + core unpark). */
    if (!s->pool) {
        if (selected > 0 && s->topology.initialized) {
            s->pool = threadpool_create_cpuset(selected_ids, selected_lps,
                                                selected_groups, selected);
        } else {
            int tc = s->config.thread_count > 0 ? s->config.thread_count : 0;
            s->pool = threadpool_create(tc, s->config.affinity_mask);
        }
    } else if (selected > 0 && s->topology.initialized) {
        /* Pool exists — re-pin ALL workers to freshly selected cores.
         * Core loads change over time; workers may be stuck on cores
         * that were idle at pool creation but are now loaded.
         * Workers wrap around selected cores (e.g., 16 workers / 7 cores). */
        int pool_count = threadpool_get_thread_count(s->pool);
        int repinned = 0;
        for (int i = 0; i < pool_count; i++) {
            int core_idx = i % selected;
            uint32_t cur = threadpool_get_worker_cpuset(s->pool, i);
            if (cur != selected_ids[core_idx]) {
                threadpool_migrate_thread(s->pool, i, selected_ids[core_idx]);
                repinned++;
            }
        }
        if (repinned > 0) {
            char msg[128];
            snprintf(msg, sizeof(msg),
                "re-pinned %d/%d workers to %d selected cores",
                repinned, pool_count, selected);
            trellis_log_c(msg);
        }
    }
    if (!s->pool) {
        for (int i = 0; i < num_channels; i++)
            engine_channel_free(&s->channels[i]);
        free(s->channels);
        s->channels = NULL;
        return -1;
    }

    /* Reset worker log counter so tasks get logged on each engine init */
    threadpool_reset_log(s->pool);

    /* Create GPU compute context if enabled */
    {
        extern void trellis_log_c(const char *);
        char msg[256];
        sprintf_s(msg, sizeof(msg),
            "GPU init check: enabled=%d gpu_ptr=%p backend=%d",
            s->config.gpu_enabled, (void*)s->gpu, s->config.gpu_backend);
        trellis_log_c(msg);
    }
    /* Step 1: Create GPU context if not yet created */
    if (s->config.gpu_enabled && !s->gpu) {
        bool avail = gpu_available((gpu_backend_t)s->config.gpu_backend);
        {
            extern void log_ring_write(const char *);
            char msg[128];
            sprintf_s(msg, sizeof(msg), "GPU available(%d) = %d", s->config.gpu_backend, avail);
            trellis_log_c(msg);
        }
        if (avail) {
            s->gpu = gpu_create((gpu_backend_t)s->config.gpu_backend);
            if (s->gpu) {
                gpu_info_t ginfo;
                gpu_get_info(&ginfo);
                const char *be_name =
                    s->config.gpu_backend == 2 ? "CUDA" :
                    gpu_dx12_probe() ? "DX12 Async Compute" : "DX11";
                char gmsg[256];
                sprintf_s(gmsg, sizeof(gmsg),
                    "GPU created: %s, %s (%zu MB). "
                    "Offloading: FIR, gain, boxcar, Trellis SDM parallel",
                    be_name, ginfo.device_name, ginfo.vram_mb);
                trellis_log_c(gmsg);
            }
        }
    }
    /* Step 2: (Re-)configure GPU for current rate/SDM params.
     * Runs every rate change even if GPU already existed. */
    if (s->gpu) {
        /* Upload FIR coefficients for current rate */
        int num_stages = s->channels[0].fir.num_stages;
        bool is_upsample = s->channels[0].fir.upsample;
        if (num_stages > 0) {
            extern float g_hb_taps[];
            extern int   g_hb_ntaps;
            gpu_fir_setup(s->gpu, g_hb_taps, g_hb_ntaps,
                          num_stages, is_upsample);
        }
        /* Upload FIR lowpass coefficients for same-rate path */
        if (s->channels[0].lowpass.initialized && s->channels[0].lowpass.coeffs) {
            int lp_rc = gpu_fir_lowpass_setup(s->gpu, s->channels[0].lowpass.coeffs,
                                               s->channels[0].lowpass.taps);
            trellis_log_c(lp_rc == 0 ? "GPU FIR lowpass setup OK" : "GPU FIR lowpass setup FAILED");
        } else {
            trellis_log_c("GPU FIR lowpass: not initialized (lowpass not active for this rate)");
        }
        /* Setup persistent SDM buffers on GPU */
        if (s->config.sdm_mode == SDM_MODE_TRELLIS &&
            s->channels[0].sdm.filter) {
            const ntf_filter_t *f = s->channels[0].sdm.filter;
            int actual_cands = (int)s->channels[0].sdm.trellis_num;
            int actual_lat = (int)s->channels[0].sdm.trellis_lat;
            int rc;
            if (s->config.gpu_backend == 2 || s->config.gpu_backend == 3)
                rc = gpu_cuda_trellis_setup(s->gpu,
                        actual_cands, f->order,
                        actual_lat, f->a, f->g,
                        s->channels[0].sdm.state_limit,
                        s->config.trellis_depth);
            else {
                rc = gpu_dx12_trellis_setup_full(s->gpu,
                        actual_cands, f->order,
                        actual_lat, f->a, f->g,
                        s->channels[0].sdm.state_limit);
                if (rc != 0)
                    rc = gpu_dx11_trellis_setup(s->gpu,
                            actual_cands, f->order,
                            actual_lat, f->a, f->g,
                            s->channels[0].sdm.state_limit);
            }
            {
                char msg[256];
                sprintf_s(msg, sizeof(msg),
                    "GPU SDM setup: backend=%d trellis cands=%d order=%d lat=%d limit=%.1f rc=%d",
                    s->config.gpu_backend, actual_cands, f->order,
                    actual_lat, s->channels[0].sdm.state_limit, rc);
                trellis_log_c(msg);
            }
        } else if (s->config.sdm_mode == SDM_MODE_PRECORR) {
            const ntf_filter_t *f = s->channels[0].precorr.filter;
            if (f) {
                float a_f[8], g_f[8];
                for (int k = 0; k < f->order; k++) {
                    a_f[k] = s->channels[0].precorr.a[k];
                    g_f[k] = s->channels[0].precorr.g[k];
                }
                gpu_cuda_precorr_setup(s->gpu, f->order, a_f, g_f,
                    (const float *)s->channels[0].precorr.pred_table,
                    s->channels[0].precorr.state_limit);
            }
        }
        /* Assign GPU context to all engine channels */
        for (int ch = 0; ch < s->num_channels; ch++) {
            s->channels[ch].gpu = s->gpu;
        }
    }

    /* Create dedicated IO pool for DoP unpack/pack (2 threads).
     * Separate from SDM pool so unpack/pack don't compete with
     * audio processing for cores. Uses OS scheduling (no pinning). */
    /* io_pool disabled — use main pool for unpack/pack too.
     * Separate unpinned io_pool causes those tasks to land on LP0/LP1. */

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
    s->needs_warmup = true;    /* prime SDM with real audio on first chunk */
    s->chunk_counter = 0;      /* reset for startup pacing */

    /* Prevent system idle/sleep during playback — discourages core parking.
     * Windows parks cores when the system appears idle. MMCSS "Pro Audio"
     * helps but doesn't fully prevent parking. ES_SYSTEM_REQUIRED keeps
     * the system in an active power state. */
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);

    return 0;
}

void plugin_set_config(plugin_state_t *s, const dsd_config_t *cfg) {
    if (!s) return;
    s->config = *cfg;
}

const dsd_config_t *plugin_get_config(const plugin_state_t *s) {
    return s ? &s->config : NULL;
}

void plugin_set_gpu_sdm(plugin_state_t *s, bool enabled) {
    if (s) s->config.gpu_sdm_enabled = enabled;
}

const engine_channel_t *plugin_get_channels(const plugin_state_t *s) {
    return (s && s->initialized) ? s->channels : NULL;
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

/* Query actual cpuset ID for a specific worker */
uint32_t plugin_get_worker_cpuset(const plugin_state_t *s, int worker_index) {
    if (!s || !s->pool)
        return 0;
    return threadpool_get_worker_cpuset(s->pool, worker_index);
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

/* ─── Worker migration (dry-run probe) ─── */

#define MIGRATION_PROBE_CHUNKS  5     /* chunks to measure on new core */
#define MIGRATION_COOLDOWN      50    /* chunks between migration attempts */
#define MIGRATION_THRESHOLD     0.15  /* 15% improvement required to keep */

static void migration_tick(plugin_state_t *s) {
    if (!s->pool || !s->topology_detected)
        return;

    extern void trellis_log_c(const char *);

    /* Cooldown between re-selections */
    if (s->migration.cooldown_chunks > 0) {
        s->migration.cooldown_chunks--;
        return;
    }

    /* ── Check if any worker's core became loaded ──
     * Read fresh load data from background monitor.
     * If any assigned core has >60% load, re-select all workers.
     * 60% avoids false triggers from normal OS background (30-40%). */
    if (!s->cpu_monitor)
        return;

    cpu_topology_t snap;
    cpuset_monitor_read(s->cpu_monitor, &snap);

    /* Check load on cores assigned to workers */
    int num_workers = threadpool_get_thread_count(s->pool);
    bool any_loaded = false;
    uint32_t loaded_cpuset = 0;
    double loaded_pct = 0.0;
    for (int i = 0; i < num_workers; i++) {
        uint32_t wid = threadpool_get_worker_cpuset(s->pool, i);
        if (wid == 0) continue;
        for (int j = 0; j < snap.count; j++) {
            if (snap.entries[j].id == wid && snap.entries[j].load > 0.60) {
                if (snap.entries[j].load > loaded_pct) {
                    loaded_pct = snap.entries[j].load;
                    loaded_cpuset = wid;
                }
                any_loaded = true;
            }
        }
    }

    if (!any_loaded)
        return;

    /* ── Core contention detected: re-select ALL workers ── */
    uint32_t new_ids[CPUSET_MAX_CPUS];
    uint8_t  new_lps[CPUSET_MAX_CPUS];
    int max_t = s->config.thread_count > 0 ? s->config.thread_count : 0;
    int new_count = cpuset_select(&snap,
        (smt_mode_t)s->config.smt_mode,
        (ccd_mode_t)s->config.ccd_mode,
        (ecore_mode_t)s->config.ecore_mode,
        max_t, new_ids, new_lps, NULL, CPUSET_MAX_CPUS);

    if (new_count <= 0)
        return;

    int pool_count = threadpool_get_thread_count(s->pool);
    int repinned = 0;
    for (int i = 0; i < pool_count; i++) {
        int core_idx = i % new_count;
        uint32_t cur = threadpool_get_worker_cpuset(s->pool, i);
        if (cur != new_ids[core_idx]) {
            threadpool_migrate_thread(s->pool, i, new_ids[core_idx]);
            repinned++;
        }
    }

    if (repinned > 0) {
        s->selected_core_count = new_count;
        memcpy(s->selected_core_ids, new_ids,
               (size_t)new_count * sizeof(uint32_t));

        /* Find LP index of loaded core for logging */
        uint8_t loaded_lp = 0;
        for (int j = 0; j < snap.count; j++)
            if (snap.entries[j].id == loaded_cpuset)
                { loaded_lp = snap.entries[j].logical_index; break; }

        char msg[256];
        int pos = snprintf(msg, sizeof(msg),
            "migration: LP%d loaded (%.0f%%), re-pinned %d workers to (",
            loaded_lp, loaded_pct * 100.0, repinned);
        for (int i = 0; i < new_count && pos < 240; i++)
            pos += snprintf(msg + pos, sizeof(msg) - pos,
                "LP%d%s", new_lps[i], i < new_count - 1 ? "," : "");
        snprintf(msg + pos, sizeof(msg) - pos, ")");
        trellis_log_c(msg);
    }

    s->migration.cooldown_chunks = MIGRATION_COOLDOWN;
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
                      const float *in_pcm, uint8_t *out_i24,
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

    /* Check lightweight GPU SDM toggle from REST API */
    {
        extern volatile LONG g_gpu_sdm_override;
        extern volatile LONG g_gpu_sdm_override_valid;
        if (InterlockedCompareExchange(&g_gpu_sdm_override_valid, 0, 0)) {
            /* Read override without clearing — stays active until API changes it */
            bool new_val = InterlockedCompareExchange(&g_gpu_sdm_override, 0, 0) ? true : false;
            if (new_val != s->config.gpu_sdm_enabled) {
                s->config.gpu_sdm_enabled = new_val;
                trellis_log_c(new_val ? "GPU SDM: enabled via API" : "GPU SDM: disabled via API");
            }
        }
    }

    /* Always keep fs_in in sync (plugin_set_config may have overwritten it) */
    s->config.fs_in = dsd_rate;

    /* Initialize engine on first use, channel/rate change, output rate,
     * SDM mode, cands/depth, or GPU toggle */
    if (!s->initialized || s->num_channels != num_channels ||
        s->detected_dsd_rate != dsd_rate ||
        s->active_fs_out != s->config.fs_out ||
        s->active_sdm_mode != s->config.sdm_mode ||
        s->active_cands != s->config.trellis_cands ||
        s->active_depth != s->config.trellis_depth ||
        s->active_gpu != s->config.gpu_enabled) {
        /* Tear down old state — keep threadpool alive (expensive to recreate:
         * 16 CreateThread + SetThreadAffinityMask + core unpark = 500ms+).
         * Pool is reused across track/rate changes. */
        if (s->initialized) {
            /* pool kept alive — NOT destroyed here */
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

    /* CPUSET changes detected by background monitor — just log, don't rebuild.
     * Rebuilding mid-playback causes heap corruption (worker TLS in use). */
    s->cpuset_changed = false;
    if (s->cpu_monitor) {
        uint64_t new_mask = 0;
        if (cpuset_monitor_changed(s->cpu_monitor, &new_mask)) {
            s->cpuset_changed = true;
            s->last_cpuset_mask = new_mask;
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

    /* Parallel DoP unpack on dedicated IO pool (separate from SDM workers) */
    threadpool_t *io = s->io_pool ? s->io_pool : s->pool;
    if (io && num_channels > 0) {
        channel_block_t unpack_blocks[32];
        memset(unpack_blocks, 0, (size_t)num_channels * sizeof(channel_block_t));
        for (int ch = 0; ch < num_channels; ch++) {
            unpack_blocks[ch].mode = BLOCK_MODE_UNPACK;
            unpack_blocks[ch].channel = ch;
            unpack_blocks[ch].cfg = &s->config;
            unpack_blocks[ch].pcm_interleaved = (float *)in_pcm;
            unpack_blocks[ch].dsd_channel = s->ch_in[ch];
            unpack_blocks[ch].pcm_frames = pcm_frames;
            unpack_blocks[ch].num_channels = num_channels;
            threadpool_submit(io, &unpack_blocks[ch]);
        }
        threadpool_wait(io);
    } else {
        for (int ch = 0; ch < num_channels; ch++) {
            for (size_t f = 0; f < pcm_frames; f++)
                pcm_temp[f] = in_pcm[f * (size_t)num_channels + (size_t)ch];
            dop_unpack(pcm_temp, s->ch_in[ch], pcm_frames);
        }
    }

    QueryPerformanceCounter(&t_end);
    s->time_unpack_ms = perf_ms(t_start, t_end);

    /* Diagnostic removed — input verified OK */

    /* Determine if parallel SDM segmentation is beneficial.
     * Worth it only when rate-converting (SDM is the bottleneck)
     * and we have more threads than channels. */
    bool need_rate_conv = !s->channels[0].passthrough && !s->config.mute;

    /* Warmup: prime SDM integrators with real audio on first chunk after
     * flush/init. SDM is fully reset on flush (preserve_sdm=false), so
     * without warmup the integrators start from zero → DC transient pop.
     * Feed real audio through boxcar→SDM (output discarded), then reset
     * Pop at playback start is from the ASIO+DSD output driver, not our SDM.
     * Confirmed by removing DSD Trellis from DSP chain — same pop occurs. */
    int num_threads = threadpool_get_thread_count(s->pool);
    int segments_per_ch = 1;
    size_t overlap = 0;

    /* Parallel path: state-seeded segments + channel parallelism.
     * DSD64 same-rate is trivially fast — skip parallel overhead. */
    uint32_t fs_out = s->config.fs_out ? s->config.fs_out : s->config.fs_in;
    bool is_same_rate = (s->config.fs_in == fs_out);
    bool skip_parallel = (is_same_rate && fs_out <= DSD_RATE_64 &&
                          !(s->config.gpu_sdm_enabled && s->gpu &&
                            s->config.sdm_mode == SDM_MODE_TRELLIS));

    /* Resolve per-rate parallel mode: Auto/Sequential/Parallel */
    int par_mode = TRELLIS_PAR_AUTO;
    {
        int map_idx = rate_map_index(s->config.fs_in);
        if (map_idx >= 0 && map_idx < RATE_MAP_COUNT)
            par_mode = (int)s->config.rate_parallel[map_idx];
    }

    /* Auto: Sequential for DSD64 same-rate, Parallel for higher rates */
    bool use_parallel;
    if (par_mode == TRELLIS_PAR_SEQUENTIAL)
        use_parallel = false;
    else if (par_mode == TRELLIS_PAR_PARALLEL)
        use_parallel = true;
    else /* Auto */
        use_parallel = (fs_out > DSD_RATE_64) ||
                       (s->config.gpu_sdm_enabled && s->gpu &&
                        s->config.sdm_mode == SDM_MODE_TRELLIS);

    if (need_rate_conv && !skip_parallel && use_parallel &&
        num_threads > num_channels &&
        (s->config.sdm_mode == SDM_MODE_TRELLIS || s->config.fs_in != fs_out)) {
        /* Trellis: always (same-rate uses boxcar internally via GPU lowpass).
         * PreCorr rate-conv: GPU FIR + GPU PreCorr intercept.
         * PreCorr same-rate: engine_process_block (CPU boxcar + CPU PreCorr).
         * GPU PreCorr is single-threaded → too slow for 22M samples. */
        size_t fir_out_est = dsd_in_count;
        if (fs_out > s->config.fs_in)
            fir_out_est = dsd_in_count * (fs_out / s->config.fs_in);

        /* Overlap for SDM convergence: 32x lat, capped at 1024.
         * Matches validated config (commit 4e00224) that had zero artifacts.
         * DSD128 lat=128 → overlap=1024 (8×lat). Uncapping to 4096 didn't
         * improve audible quality (pops are ultrasonic, not overlap-related). */
        overlap = 32 * (size_t)s->config.trellis_lat;
        if (overlap > 1024) overlap = 1024;
        segments_per_ch = num_threads / num_channels;
        if (segments_per_ch < 1) segments_per_ch = 1;
        int max_seg;
        /* DSD512: allow more segments — stitch pops less audible at high
         * rates where ultrasonic noise floor is already elevated.
         * DSD64-256: 2 segments (1 stitch/sec) minimizes audible pops. */
        max_seg = (fs_out >= DSD_RATE_512) ? 4 : 2;
        if (segments_per_ch > max_seg) segments_per_ch = max_seg;

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
        /* === Parallel SDM path: FIR+gain, then state-seeded SDM === */

        /* Reset GPU per-chunk state (channel counters for boxcar/lowpass history) */
        if (s->gpu)
            gpu_reset_chunk(s->gpu);

        /* Phase 1: FIR + gain per channel (parallel via threadpool) */
        LARGE_INTEGER t_fir_start, t_fir_end;
        QueryPerformanceCounter(&t_fir_start);

        double *fir_data[32];  /* max 32 channels (fp64 pipeline) */
        size_t fir_counts[32];

        /* GPU FIR: single batched dispatch for all channels when available.
         * Falls back to threadpool FIR on GPU failure or small buffers.
         * TEMP: force CPU FIR to isolate GPU SDM noise source. */
        bool gpu_fir_ok = false;
        bool force_cpu_fir = false;

        /* GPU FIR lowpass for same-rate path (no rate conversion stages).
         * Runs on main thread (GPU not thread-safe).
         * GPU kernel processes in fp64; output goes directly to fir_buf (double). */
        if (!force_cpu_fir && s->gpu && s->config.gpu_enabled &&
            dsd_in_count >= GPU_MIN_SAMPLES &&
            s->channels[0].fir.num_stages == 0 &&
            s->channels[0].lowpass.initialized) {
            float combined = s->channels[0].fir_gain * s->config.gain;
            gpu_fir_ok = true;
            for (int ch = 0; ch < num_channels && gpu_fir_ok; ch++) {
                /* Ensure double fir_buf */
                if (s->channels[ch].fir_buf_sz < dsd_in_count * sizeof(double)) {
                    free(s->channels[ch].fir_buf);
                    s->channels[ch].fir_buf = (double *)malloc(dsd_in_count * sizeof(double));
                    s->channels[ch].fir_buf_sz = s->channels[ch].fir_buf ? dsd_in_count * sizeof(double) : 0;
                }
                if (!s->channels[ch].fir_buf) { gpu_fir_ok = false; break; }
                if (gpu_cuda_fir_lowpass_f64(s->gpu, s->ch_in[ch],
                                              s->channels[ch].fir_buf,
                                              dsd_in_count, (double)combined) != 0) {
                    gpu_fir_ok = false; break;
                }
                fir_counts[ch] = dsd_in_count;
                fir_data[ch] = s->channels[ch].fir_buf;
            }
            if (gpu_fir_ok) {
                static int lp_log_count = 0;
                if (lp_log_count++ < 3)
                    trellis_log_c("GPU FIR lowpass: dispatched on main thread");
            }
        }

        /* GPU rate conversion FIR: GPU outputs float, fir_buf is double.
         * Use static TLS float buffer → GPU → widen to double + apply gain.
         * Same pattern as GPU FIR lowpass path above. */
        if (!gpu_fir_ok && s->gpu && s->config.gpu_enabled &&
            dsd_in_count >= GPU_MIN_SAMPLES &&
            s->channels[0].fir.num_stages > 0) {
            /* Estimate output count for buffer sizing */
            size_t est_out = dsd_in_count;
            for (int st = 0; st < s->channels[0].fir.num_stages; st++)
                est_out = s->channels[0].fir.upsample ? est_out * 2 : est_out / 2;

            /* TLS float buffer for GPU output (before double widening) */
            static float *s_gpu_fir_tmp = NULL;
            static size_t s_gpu_fir_sz = 0;
            if (s_gpu_fir_sz < est_out) {
                free(s_gpu_fir_tmp);
                s_gpu_fir_tmp = (float *)malloc(est_out * sizeof(float));
                s_gpu_fir_sz = s_gpu_fir_tmp ? est_out : 0;
            }

            if (s_gpu_fir_tmp) {
                double combined_gain = (double)s->channels[0].fir_gain
                                     * (double)s->config.gain;
                for (int ch = 0; ch < num_channels; ch++) {
                    size_t gpu_out = 0;
                    if (gpu_fir_chain_process(s->gpu, s->ch_in[ch],
                            s_gpu_fir_tmp, dsd_in_count,
                            &gpu_out, NULL, NULL) == 0) {
                        /* Ensure fir_buf is large enough */
                        size_t need = gpu_out * sizeof(double);
                        if (s->channels[ch].fir_buf_sz < need) {
                            free(s->channels[ch].fir_buf);
                            s->channels[ch].fir_buf = (double *)malloc(need);
                            s->channels[ch].fir_buf_sz =
                                s->channels[ch].fir_buf ? need : 0;
                        }
                        if (!s->channels[ch].fir_buf) {
                            gpu_fir_ok = false; break;
                        }
                        /* Widen float→double + apply gain in one pass */
                        for (size_t i = 0; i < gpu_out; i++)
                            s->channels[ch].fir_buf[i] =
                                (double)s_gpu_fir_tmp[i] * combined_gain;
                        fir_counts[ch] = gpu_out;
                        fir_data[ch] = s->channels[ch].fir_buf;
                        gpu_fir_ok = true;
                    } else {
                        gpu_fir_ok = false;
                        break;
                    }
                }
            }
        }

        if (!gpu_fir_ok) {
            /* Threadpool FIR fallback (CPU) */
            {
                static int cpu_fir_log = 0;
                if (cpu_fir_log++ < 3)
                    trellis_log_c("FIR: CPU threadpool fallback");
            }
            channel_block_t fir_blocks[32];
            memset(fir_blocks, 0, (size_t)num_channels * sizeof(channel_block_t));
            for (int ch = 0; ch < num_channels; ch++) {
                fir_blocks[ch].mode    = BLOCK_MODE_FIR;
                fir_blocks[ch].eng     = &s->channels[ch];
                fir_blocks[ch].in_f32  = s->ch_in[ch];
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
        }

        QueryPerformanceCounter(&t_fir_end);
        s->time_fir_ms = perf_ms(t_fir_start, t_fir_end);

        /* PreCorr: GPU FIR done above, now run GPU PreCorr SDM.
         * Skip the parallel Trellis Phase 2 entirely. */
        if (s->config.sdm_mode == SDM_MODE_PRECORR) {
            LARGE_INTEGER t_pc_start, t_pc_end;
            QueryPerformanceCounter(&t_pc_start);
            size_t fir_n = fir_counts[0];
            /* Convert fp64 FIR output → fp32 for GPU PreCorr */
            size_t total_f32 = fir_n * (size_t)num_channels;
            if (s->gpu_das_buf_cap < fir_n || s->gpu_das_buf_nch < num_channels) {
                free(s->gpu_das_in); free(s->gpu_das_out);
                s->gpu_das_in  = (float *)malloc(total_f32 * sizeof(float));
                s->gpu_das_out = (float *)malloc(total_f32 * sizeof(float));
                s->gpu_das_buf_cap = s->gpu_das_in ? fir_n : 0;
                s->gpu_das_buf_nch = s->gpu_das_in ? num_channels : 0;
            }
            if (s->gpu_das_in && s->gpu_das_out) {
                for (int ch = 0; ch < num_channels; ch++)
                    for (size_t i = 0; i < fir_n; i++)
                        s->gpu_das_in[ch * fir_n + i] = (float)fir_data[ch][i];
                /* Ensure GPU PreCorr kernel is set up for current NTF */
                {
                    const ntf_filter_t *pf = s->channels[0].precorr.filter;
                    if (pf) {
                        float a_f[8], g_f[8];
                        for (int k = 0; k < pf->order; k++) {
                            a_f[k] = s->channels[0].precorr.a[k];
                            g_f[k] = s->channels[0].precorr.g[k];
                        }
                        gpu_cuda_precorr_setup(s->gpu, pf->order, a_f, g_f,
                            (const float *)s->channels[0].precorr.pred_table,
                            s->channels[0].precorr.state_limit);
                    }
                }
                /* GPU PreCorr: single kernel for all samples */
                int rc = -1;
                for (int ch = 0; ch < num_channels; ch++) {
                    rc = gpu_cuda_precorr(s->gpu,
                        s->gpu_das_in + ch * fir_n,
                        s->gpu_das_out + ch * fir_n,
                        fir_n, NULL, NULL);
                    if (rc != 0) break;
                }
                if (rc == 0) {
                    {
                        static int pc_ok_log = 0;
                        if (pc_ok_log++ < 3) {
                            char msg[128];
                            snprintf(msg, sizeof(msg),
                                "GPU PreCorr OK: %zu samples, %d ch",
                                fir_n, num_channels);
                            trellis_log_c(msg);
                        }
                    }
                    for (int ch = 0; ch < num_channels; ch++)
                        memcpy(s->ch_out[ch], s->gpu_das_out + ch * fir_n,
                               fir_n * sizeof(float));
                    dsd_out_count = fir_n;
                } else {
                    /* GPU PreCorr failed, fall back to CPU */
                    {
                        static int pc_fail_log = 0;
                        if (pc_fail_log++ < 5) {
                            const ntf_filter_t *pf = s->channels[0].precorr.filter;
                            char msg[256];
                            snprintf(msg, sizeof(msg),
                                "GPU PreCorr FAILED rc=%d, filter=%p, "
                                "fir_n=%zu, CPU fallback",
                                rc, (void *)pf, fir_n);
                            trellis_log_c(msg);
                        }
                    }
                    for (int ch = 0; ch < num_channels; ch++)
                        dsd_out_count = precorr_process_block(
                            &s->channels[ch].precorr, fir_data[ch],
                            s->ch_out[ch], fir_n);
                }
            } else {
                /* Buffer alloc failed, CPU fallback */
                for (int ch = 0; ch < num_channels; ch++)
                    dsd_out_count = precorr_process_block(
                        &s->channels[ch].precorr, fir_data[ch],
                        s->ch_out[ch], fir_n);
            }
            QueryPerformanceCounter(&t_pc_end);
            s->time_sdm_ms = perf_ms(t_pc_start, t_pc_end);
            goto sdm_done;
        }

        /* Phase 2: Get temp SDM contexts.
         * When fir_tail is available (chunk 2+), ALL segments use temp SDMs.
         * This keeps segment boundary artifacts symmetric (both sides warmup). */
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

        /* discard = overlap - trellis_lat, so output_count = this_seg exactly */
        size_t discard = overlap - (size_t)s->config.trellis_lat;

        /* Allocate FIR tail and seg0 buffers if needed */
        if (!s->fir_tail && num_channels > 0) {
            s->fir_tail = (double **)calloc((size_t)num_channels, sizeof(double *));
            s->seg0_buf = (double **)calloc((size_t)num_channels, sizeof(double *));
        }

        LARGE_INTEGER t_sdm_start, t_sdm_end;
        QueryPerformanceCounter(&t_sdm_start);

        /* Upload extended seed (history/path/traceback) from CPU SDM context
         * for full state continuity in GPU segments. Without this, GPU segments
         * start with zeroed history → low stitch density with complex signals. */
        if (0 && s->gpu && s->config.gpu_sdm_enabled &&
            s->config.sdm_mode == SDM_MODE_TRELLIS &&
            s->channels[0].sdm.pending >= s->channels[0].sdm.trellis_lat) {
            /* DISABLED: CPU ext_seed diverges from GPU state after first chunk,
             * causing transients. GPU relies on NTF init_states + convergence
             * warmup (M samples) instead. */
            sdm_context_t *ctx0 = &s->channels[0].sdm;
            int bank = ctx0->idx & 1;
            sdm_trellis_t *st = &ctx0->trellis[bank];
            int nc_act = (int)ctx0->num_cands;
            int lat_act = (int)ctx0->trellis_lat;
            int hist_bytes_act = (lat_act + 7) / 8;

            /* Extract per-candidate history, path, next from CPU context */
            unsigned char h_hist[8][128];
            unsigned h_path[8], h_next[8];
            memset(h_hist, 0, sizeof(h_hist));
            memset(h_path, 0, sizeof(h_path));
            memset(h_next, 0, sizeof(h_next));
            for (int i = 0; i < nc_act && i < 8; i++) {
                sdm_state_t *a = st->act[i];
                if (a) {
                    /* Copy history ring buffer */
                    int hist_idx = a->hist;
                    for (int b = 0; b < hist_bytes_act && b < 128; b++)
                        h_hist[i][b] = ctx0->hist[hist_idx][b];
                    h_path[i] = a->path;
                    h_next[i] = a->next;
                }
            }
            gpu_cuda_upload_ext_seed(s->gpu,
                (const unsigned char *)h_hist, hist_bytes_act, nc_act,
                h_path, h_next, (int)ctx0->pos, (int)ctx0->pending);
            {
                static int seed_log = 0;
                if (seed_log++ < 5) {
                    char msg[256];
                    snprintf(msg, sizeof(msg),
                        "ext_seed: nc=%d pos=%d pending=%d path=[%u,%u] "
                        "next=[%u,%u] hist[0]=[%02x%02x%02x%02x]",
                        nc_act, (int)ctx0->pos, (int)ctx0->pending,
                        h_path[0], h_path[1], h_next[0], h_next[1],
                        h_hist[0][0], h_hist[0][1], h_hist[0][2], h_hist[0][3]);
                    trellis_log_c(msg);
                }
            }
        }

        /* GPU offload: CPU state pass + GPU parallel output.
         * CPU runs the full chunk sequentially to capture boundary states,
         * then GPU re-processes segments in parallel with perfect seeding.
         * No stitching needed — each GPU segment starts from exact CPU state. */
        if (0 && s->config.gpu_sdm_enabled && s->gpu &&
            s->config.sdm_mode == SDM_MODE_TRELLIS &&
            (s->config.gpu_backend == 2 || s->config.gpu_backend == 3)) {
            size_t fir_n = fir_counts[0];

            /* Ensure GPU SDM is set up for current rate */
            {
                const ntf_filter_t *f = s->channels[0].sdm.filter;
                int cur_cands = (int)s->channels[0].sdm.trellis_num;
                int cur_lat   = (int)s->channels[0].sdm.trellis_lat;
                if (f)
                    gpu_cuda_trellis_setup(s->gpu, cur_cands, f->order,
                        cur_lat, f->a, f->g, s->channels[0].sdm.state_limit,
                        s->config.trellis_depth);
            }

            /* GPU Trellis SDM: 2-pass DAS with internal stitching.
             * gpu_cuda_trellis_das handles segment layout, 2-pass state
             * chaining, and DAS density-matched stitching internally. */
            {
                /* Pack fp64 fir_data into contiguous buffer for GPU */
                size_t total_f64 = fir_n * (size_t)num_channels;
                if (s->gpu_das_buf_cap < fir_n || s->gpu_das_buf_nch < num_channels) {
                    free(s->gpu_das_in); free(s->gpu_das_out);
                    s->gpu_das_in  = (float *)malloc(total_f64 * sizeof(double));
                    s->gpu_das_out = (float *)malloc(total_f64 * sizeof(float));
                    s->gpu_das_buf_cap = s->gpu_das_in ? fir_n : 0;
                    s->gpu_das_buf_nch = s->gpu_das_in ? num_channels : 0;
                }

                if (s->gpu_das_in && s->gpu_das_out) {
                    double *das_in_f64 = (double *)s->gpu_das_in;
                    for (int ch = 0; ch < num_channels; ch++)
                        memcpy(das_in_f64 + ch * fir_n, fir_data[ch],
                               fir_n * sizeof(double));

                    int rc = gpu_cuda_trellis_das(s->gpu, das_in_f64,
                                                   s->gpu_das_out, fir_n,
                                                   num_channels);
                    if (rc == 0) {
                        for (int ch = 0; ch < num_channels; ch++)
                            memcpy(s->ch_out[ch], s->gpu_das_out + ch * fir_n,
                                   fir_n * sizeof(float));
                        dsd_out_count = fir_n;
                    }
                }

                {
                    static int offload_log = 0;
                    if (offload_log++ < 3) {
                        char msg[128];
                        snprintf(msg, sizeof(msg),
                            "GPU SDM: trellis_das %zu samples, %d ch, rc=%d",
                            fir_n, num_channels, dsd_out_count > 0 ? 0 : -1);
                        trellis_log_c(msg);
                    }
                }
            }

            if (dsd_out_count > 0) {
                /* Run CPU SDM on the same input to keep persistent state
                 * current for next chunk's ext_seed. Output discarded —
                 * only the state evolution matters. */
                for (int ch = 0; ch < num_channels; ch++) {
                    static __declspec(thread) float *tls_trash = NULL;
                    static __declspec(thread) size_t tls_trash_sz = 0;
                    if (tls_trash_sz < fir_n) {
                        free(tls_trash);
                        tls_trash = (float *)malloc(fir_n * sizeof(float));
                        tls_trash_sz = tls_trash ? fir_n : 0;
                    }
                    if (tls_trash)
                        sdm_process_block(&s->channels[ch].sdm,
                                           fir_data[ch], tls_trash, fir_n);
                }
                QueryPerformanceCounter(&t_sdm_end);
                s->time_sdm_ms = perf_ms(t_sdm_start, t_sdm_end);
                goto sdm_done;
            }
            /* Fall through to CPU parallel path on failure */
        }

        /* Phase 2: Launch ALL segments in parallel — no sequential dependency.
         * All segments (including seg0) are seeded from the persistent SDM's
         * current state and run simultaneously. Overlap stitching handles
         * boundary alignment after all segments complete. */

        /* Compute nominal segment boundaries */
        size_t seg_nominal_start[32][64];  /* [ch][seg] — up to 64 segs/ch for hybrid */
        size_t seg_nominal_size[32][64];
        size_t seg0_nominal[32];
        size_t seg0_outs[32];
        for (int ch = 0; ch < num_channels; ch++) {
            size_t fir_count = fir_counts[ch];
            size_t base_seg = fir_count / (size_t)segments_per_ch;
            size_t remainder = fir_count % (size_t)segments_per_ch;
            size_t pos = 0;
            for (int seg = 0; seg < segments_per_ch; seg++) {
                seg_nominal_start[ch][seg] = pos;
                size_t sz;
                if (seg == segments_per_ch - 1)
                    sz = fir_count - pos;
                else
                    sz = base_seg + ((size_t)seg < remainder ? 1 : 0);
                seg_nominal_size[ch][seg] = sz;
                pos += sz;
            }
            seg0_nominal[ch] = seg_nominal_size[ch][0];
        }

        /* Ensure cached segment output buffers are large enough.
         * Reused across chunks to avoid malloc/free contention. */
        int total_all_segs = num_channels * segments_per_ch;
        size_t max_seg_samples = fir_counts[0] / (size_t)segments_per_ch + overlap + 4096;
        if (s->cached_seg_buf_count < total_all_segs ||
            s->cached_seg_buf_sz < max_seg_samples) {
            /* Grow cached buffers */
            if (s->cached_seg_bufs) {
                for (int i = 0; i < s->cached_seg_buf_count; i++)
                    free(s->cached_seg_bufs[i]);
                free(s->cached_seg_bufs);
            }
            s->cached_seg_bufs = (float **)calloc((size_t)total_all_segs, sizeof(float *));
            for (int i = 0; i < total_all_segs; i++)
                s->cached_seg_bufs[i] = (float *)malloc(max_seg_samples * sizeof(float));
            s->cached_seg_buf_count = total_all_segs;
            s->cached_seg_buf_sz = max_seg_samples;
        }
        float **seg_bufs = s->cached_seg_bufs;
        size_t seg_out_counts_arr[128];  /* up to 64 segs × 2 ch */
        size_t *seg_out_counts = seg_out_counts_arr;
        memset(seg_out_counts, 0, (size_t)total_all_segs * sizeof(size_t));

        /* ── Hybrid CPU+GPU segment split ──
         * CPU processes segments 0..(cpu_segs-1) on threadpool.
         * GPU processes segments cpu_segs..(segments_per_ch-1) concurrently.
         * Both seed from the same persistent state. DAS stitching merges all. */
        int cpu_segs = segments_per_ch;
        int gpu_segs = 0;
        bool gpu_hybrid_ok = false;

        if (0 && s->config.gpu_sdm_enabled && s->gpu &&
            s->config.sdm_mode == SDM_MODE_TRELLIS &&
            (s->config.gpu_backend == 2 || s->config.gpu_backend == 3) &&
            segments_per_ch >= 2) {
            /* GPU handles ALL output segments (self-consistent fp32).
             * CPU runs seg0 concurrently for state persistence only
             * (fp64 accuracy for next chunk). No CPU-GPU stitch needed. */
            cpu_segs = 1;  /* CPU seg0: for state persistence, output discarded */
            gpu_segs = segments_per_ch;  /* GPU: ALL segments produce output */
        }

        /* ── Phase 2a: Launch GPU segments async (non-blocking) ── */
        if (gpu_segs > 0) {
            /* Build GPU segment descriptors (GPU segments = cpu_segs..segments_per_ch-1) */
            int *gpu_seg_starts = (int *)calloc((size_t)gpu_segs, sizeof(int));
            int *gpu_seg_out_starts = (int *)calloc((size_t)gpu_segs, sizeof(int));
            int *gpu_seg_totals = (int *)calloc((size_t)gpu_segs, sizeof(int));
            int *gpu_seg_out_caps_arr = (int *)calloc((size_t)gpu_segs, sizeof(int));

            if (gpu_seg_starts && gpu_seg_out_starts && gpu_seg_totals && gpu_seg_out_caps_arr) {
                size_t fir_count = fir_counts[0];
                int M_gpu = 16 * s->config.trellis_lat;
                if (M_gpu > 4096) M_gpu = 4096;
                int gpu_overlap = (int)overlap;

                size_t gpu_out_offset = 0;
                for (int g = 0; g < gpu_segs; g++) {
                    int seg = g;  /* GPU handles ALL segments (0..gpu_segs-1) */
                    size_t nom_start = seg_nominal_start[0][seg];
                    size_t nom_size = seg_nominal_size[0][seg];

                    int in_start = (int)nom_start - M_gpu;
                    if (in_start < 0) in_start = 0;

                    int out_cap;
                    int seg_total;
                    bool extend = (seg < segments_per_ch - 1);
                    if (extend) {
                        out_cap = (int)nom_size + gpu_overlap;
                        seg_total = M_gpu + (int)nom_size + gpu_overlap + s->config.trellis_lat;
                    } else {
                        out_cap = (int)nom_size;
                        seg_total = M_gpu + (int)nom_size + s->config.trellis_lat;
                    }
                    if (in_start + seg_total > (int)fir_count)
                        seg_total = (int)fir_count - in_start;

                    gpu_seg_starts[g] = in_start;
                    gpu_seg_out_starts[g] = (int)gpu_out_offset;
                    gpu_seg_totals[g] = seg_total;
                    gpu_seg_out_caps_arr[g] = out_cap;
                    gpu_out_offset += (size_t)out_cap;
                }

                /* Pack fp64 fir_data into contiguous buffer for GPU */
                size_t total_f64 = fir_count * (size_t)num_channels;
                if (s->gpu_das_buf_cap < fir_count || s->gpu_das_buf_nch < num_channels) {
                    free(s->gpu_das_in);
                    s->gpu_das_in = (float *)malloc(total_f64 * sizeof(double));
                    s->gpu_das_buf_cap = s->gpu_das_in ? fir_count : 0;
                    s->gpu_das_buf_nch = s->gpu_das_in ? num_channels : 0;
                }
                if (s->gpu_das_in) {
                    double *das_in_f64 = (double *)s->gpu_das_in;
                    for (int ch = 0; ch < num_channels; ch++)
                        memcpy(das_in_f64 + ch * fir_count, fir_data[ch],
                               fir_count * sizeof(double));
                }

                /* Launch GPU segments (returns immediately) */
                int rc = gpu_cuda_hybrid_async(s->gpu,
                    (const double *)s->gpu_das_in, fir_count, num_channels,
                    gpu_seg_starts, gpu_seg_out_starts,
                    gpu_seg_totals, gpu_seg_out_caps_arr,
                    gpu_segs, gpu_overlap, gpu_out_offset);

                if (rc == 0) {
                    gpu_hybrid_ok = true;
                    {
                        static int hybrid_log = 0;
                        if (hybrid_log++ < 3) {
                            char msg[128];
                            snprintf(msg, sizeof(msg),
                                "hybrid: %d CPU + %d GPU segments, %zu samples",
                                cpu_segs, gpu_segs, fir_count);
                            trellis_log_c(msg);
                        }
                    }
                }
            }
            free(gpu_seg_starts); free(gpu_seg_out_starts);
            free(gpu_seg_totals); free(gpu_seg_out_caps_arr);

            if (!gpu_hybrid_ok) {
                /* GPU launch failed — fall back to all-CPU */
                cpu_segs = segments_per_ch;
                gpu_segs = 0;
            }
        }

        /* ── Phase 2b: Configure CPU segment blocks ── */
        channel_block_t all_blocks[32];
        int all_block_count = 0;
        memset(all_blocks, 0, sizeof(all_blocks));

        for (int ch = 0; ch < num_channels; ch++) {
            size_t fir_count = fir_counts[ch];
            for (int seg = 0; seg < cpu_segs; seg++) {
                int buf_idx = ch * segments_per_ch + seg;
                size_t nom_start = seg_nominal_start[ch][seg];
                size_t nom_size = seg_nominal_size[ch][seg];

                size_t input_start, input_count, warmup_discard;
                bool extend_fwd = (seg < segments_per_ch - 1);
                if (seg == 0) {
                    input_start = 0;
                    input_count = nom_size;
                    if (extend_fwd && nom_size + overlap <= fir_count)
                        input_count += overlap;
                    warmup_discard = 0;
                } else {
                    input_start = (nom_start >= overlap) ? nom_start - overlap : 0;
                    input_count = nom_start + nom_size - input_start;
                    if (extend_fwd && nom_start + nom_size + overlap <= fir_count)
                        input_count += overlap;
                    warmup_discard = (nom_start >= overlap) ?
                        overlap - (size_t)s->config.trellis_lat : 0;
                }

                sdm_context_t *ctx;
                if (seg == 0) {
                    ctx = &s->channels[ch].sdm;
                } else {
                    int temp_idx = ch * temps_per_ch + (seg - 1);
                    sdm_context_copy_state(&temp_sdms[temp_idx], &s->channels[ch].sdm);
                    ctx = &temp_sdms[temp_idx];
                }

                all_blocks[all_block_count].mode     = BLOCK_MODE_SDM;
                all_blocks[all_block_count].sdm_ctx  = ctx;
                all_blocks[all_block_count].in       = fir_data[ch] + input_start;
                all_blocks[all_block_count].out      = seg_bufs[buf_idx];
                all_blocks[all_block_count].count    = input_count;
                all_blocks[all_block_count].discard  = warmup_discard;
                all_blocks[all_block_count].channel  = ch;
                all_block_count++;
            }
        }

        /* Launch CPU segments (runs concurrently with GPU) */
        for (int i = 0; i < all_block_count; i++)
            threadpool_submit_to(s->pool, i % num_threads, &all_blocks[i]);
        threadpool_wait(s->pool);

        /* Record CPU output counts */
        for (int i = 0; i < all_block_count; i++) {
            int ch = all_blocks[i].channel;
            int seg = i - ch * cpu_segs;
            int idx = ch * segments_per_ch + seg;
            seg_out_counts[idx] = all_blocks[i].out_count;
        }

        /* ── Phase 2c: GPU pass 2 + download ── */
        if (gpu_hybrid_ok && gpu_segs > 0) {
            /* GPU pass 2: chained from pass-1 finals (no CPU state needed —
             * GPU handles ALL segments self-consistently in fp32). */
            {
                /* Wait for pass 1, chain, launch pass 2 */
                int nc_cfg = s->config.trellis_cands;
                /* Use persistent state as seg0 seed (same as pass 1) */
                double dummy_states[8 * 8] = {0};
                double dummy_costs[8] = {0};
                /* Actually: pass2 should chain from pass-1 finals.
                 * seg0 keeps persistent seed (already in d_das_init_states[0]).
                 * Use pass2 with seg0 re-seeded from persistent state. */
                gpu_cuda_hybrid_pass2(s->gpu, NULL, NULL);
            }

            /* Download GPU results into seg_bufs */
            float *gpu_seg_ptrs[128];
            size_t gpu_seg_counts[128];
            memset(gpu_seg_counts, 0, sizeof(gpu_seg_counts));
            for (int ch = 0; ch < num_channels; ch++)
                for (int g = 0; g < gpu_segs; g++)
                    gpu_seg_ptrs[g * num_channels + ch] =
                        seg_bufs[ch * segments_per_ch + g];

            gpu_cuda_hybrid_finish(s->gpu, gpu_seg_ptrs, gpu_seg_counts,
                                    gpu_segs, num_channels);

            /* Record GPU output counts — GPU covers ALL segments */
            for (int ch = 0; ch < num_channels; ch++)
                for (int g = 0; g < gpu_segs; g++)
                    seg_out_counts[ch * segments_per_ch + g] =
                        gpu_seg_counts[g * num_channels + ch];

            /* CPU seg0 output is discarded (only used for state persistence).
             * Clear its count so DAS stitching skips it. */
            /* Actually: GPU covers seg0..N, DAS uses seg_bufs[0..N].
             * CPU seg0's output in seg_bufs[ch*segs+0] gets OVERWRITTEN
             * by GPU seg0's output above. That's correct — GPU output
             * is self-consistent. */
        }

        /* Phase 2c: Assemble output with hybrid overlap stitching.
         * Uses windowed match density to find the region of best SDM
         * convergence, then picks the nearest exact bit-match for a
         * clean transition.  Computed on ch0, applied to all channels. */
        int stitch_positions[64];
        memset(stitch_positions, 0, sizeof(stitch_positions));

        /* Pass 1: find stitch positions on channel 0 */
        {
            int ch = 0;
            int buf0 = ch * segments_per_ch;
            size_t seg0_out = seg_out_counts[buf0];
            memcpy(s->ch_out[ch], seg_bufs[buf0], seg0_out * sizeof(float));
            size_t write_pos = seg0_out;

            for (int seg = 1; seg < segments_per_ch; seg++) {
                int buf_idx = ch * segments_per_ch + seg;
                size_t seg_out = seg_out_counts[buf_idx];
                if (seg_out == 0) { stitch_positions[seg] = 0; continue; }

                size_t prev_ovl_start = (write_pos >= overlap) ? write_pos - overlap : 0;
                float *prev_ovl = s->ch_out[ch] + prev_ovl_start;
                float *this_ovl = seg_bufs[buf_idx];
                size_t ovl_len = write_pos - prev_ovl_start;
                if (ovl_len > seg_out) ovl_len = seg_out;
                if (ovl_len > overlap) ovl_len = overlap;

                /* Step 1: Windowed match density — find region of best
                 * convergence.  Window size = 2 × trellis_lat. */
                int half_w = s->config.trellis_lat;
                if (half_w > (int)ovl_len / 2) half_w = (int)ovl_len / 2;
                if (half_w < 4) half_w = 4;

                /* O(n) sliding window: initialize window [0, min(2*half_w, ovl_len)),
                 * then slide right, adding one sample and removing one. */
                int best_density = 0, best_density_pos = 0;
                int win = half_w * 2;
                if (win > (int)ovl_len) win = (int)ovl_len;
                /* Seed: count matches in initial window [0, win) */
                int cur_matches = 0;
                for (int w = 0; w < win && w < (int)ovl_len; w++) {
                    if (prev_ovl[w] == this_ovl[w])
                        cur_matches++;
                }
                /* Position 0: window center at 0, covers [0, win) */
                best_density = cur_matches;
                best_density_pos = 0;
                /* Slide window across all positions */
                for (int p = 1; p < (int)ovl_len; p++) {
                    /* Window for position p: [p - half_w, p + half_w)
                     * clamped to [0, ovl_len). As p advances by 1:
                     * - add sample at right edge (p + half_w - 1) if in range
                     * - remove sample at left edge (p - half_w - 1) if was in range */
                    int add_idx = p + half_w - 1;
                    int rem_idx = p - half_w - 1;
                    if (add_idx >= 0 && add_idx < (int)ovl_len) {
                        if (prev_ovl[add_idx] == this_ovl[add_idx])
                            cur_matches++;
                    }
                    if (rem_idx >= 0 && rem_idx < (int)ovl_len) {
                        if (prev_ovl[rem_idx] == this_ovl[rem_idx])
                            cur_matches--;
                    }
                    if (cur_matches > best_density) {
                        best_density = cur_matches;
                        best_density_pos = p;
                    }
                }

                /* Step 2: Within the best-density region, find nearest
                 * run of consecutive matching samples. Longer runs produce
                 * smoother transitions between different noise-shaping patterns.
                 * Try min_run=8, then 4, then 2, then 1 (single match). */
                int best_pos = best_density_pos;  /* fallback: density peak */
                int search_r = half_w;
                int found_match = 0;
                for (int min_run = 8; min_run >= 1 && !found_match; min_run >>= 1) {
                    for (int r = 0; r <= search_r && !found_match; r++) {
                        int candidates[2] = { best_density_pos - r, best_density_pos + r };
                        for (int c = 0; c < 2 && !found_match; c++) {
                            int p = candidates[c];
                            if (c == 1 && p == candidates[0]) continue;
                            if (p < 0 || p + min_run > (int)ovl_len) continue;
                            int run = 0;
                            for (int k = 0; k < min_run; k++) {
                                if (prev_ovl[p + k] == this_ovl[p + k]) run++;
                                else break;
                            }
                            if (run >= min_run) {
                                best_pos = p;
                                found_match = min_run;
                            }
                        }
                    }
                }

                stitch_positions[seg] = best_pos;

                size_t stitch_at = prev_ovl_start + (size_t)best_pos;
                size_t copy_count = seg_out - (size_t)best_pos;
                memcpy(s->ch_out[ch] + stitch_at,
                       seg_bufs[buf_idx] + best_pos,
                       copy_count * sizeof(float));
                write_pos = stitch_at + copy_count;

                {
                    char msg[256];
                    snprintf(msg, sizeof(msg),
                             "stitch seg%d: ovl=%zu density=%d/%d pos=%d run=%d "
                             "wp=%zu stitch_at=%zu",
                             seg, ovl_len, best_density, half_w * 2,
                             best_pos, found_match,
                             write_pos, stitch_at);
                    extern void trellis_log_c(const char *);
                    trellis_log_c(msg);
                }
            }
            seg0_outs[0] = write_pos;
        }

        /* Pass 2: apply same stitch positions to all other channels */
        for (int ch = 1; ch < num_channels; ch++) {
            int buf0 = ch * segments_per_ch;
            size_t seg0_out = seg_out_counts[buf0];
            memcpy(s->ch_out[ch], seg_bufs[buf0], seg0_out * sizeof(float));
            size_t write_pos = seg0_out;

            for (int seg = 1; seg < segments_per_ch; seg++) {
                int buf_idx = ch * segments_per_ch + seg;
                size_t seg_out = seg_out_counts[buf_idx];
                if (seg_out == 0) continue;

                size_t prev_ovl_start = (write_pos >= overlap) ? write_pos - overlap : 0;
                int bp = stitch_positions[seg];
                size_t stitch_at = prev_ovl_start + (size_t)bp;
                size_t copy_count = seg_out - (size_t)bp;
                memcpy(s->ch_out[ch] + stitch_at,
                       seg_bufs[buf_idx] + bp,
                       copy_count * sizeof(float));
                write_pos = stitch_at + copy_count;
            }
        }

        /* Copy last segment's final state back into persistent SDM.
         * seg0 uses persistent directly, seg1+ use temps at index (seg-1).
         * Last segment = segments_per_ch-1, its temp index = segments_per_ch-2.
         * This is correct regardless of use_fir_tail (extra temps are unused). */
        if (segments_per_ch > 1) {
            for (int ch = 0; ch < num_channels; ch++) {
                int last_temp = ch * temps_per_ch + (segments_per_ch - 2);
                if (last_temp >= 0 && last_temp < temp_sdm_count)
                    sdm_context_copy_state(&s->channels[ch].sdm, &temp_sdms[last_temp]);
            }
        }

        QueryPerformanceCounter(&t_sdm_end);
        s->time_sdm_ms = perf_ms(t_sdm_start, t_sdm_end);

        dsd_out_count = seg0_outs[0];

        /* Always-on DAS output verification:
         * - Normal chunks: dsd_out_count == fir_counts[0]
         * - First chunk after flush: dsd_out_count == fir_counts[0] - trellis_lat
         *   (seg0 loses trellis_lat samples to latency buffer fill)
         * - dsd_out_count must be multiple of 16 (DoP frame alignment)
         * - Marker phase: (dsd_out_count/16) parity must be predictable */
        {
            long long diff = (long long)dsd_out_count - (long long)fir_counts[0];
            int lat = s->config.trellis_lat;
            bool count_ok = (diff == 0 || diff == -(long long)lat);
            bool align_ok = (dsd_out_count % 16 == 0);
            int marker_frames = (int)(dsd_out_count / 16);
            int marker_phase_delta = marker_frames & 1;

            if (!count_ok || !align_ok) {
                char msg[256];
                snprintf(msg, sizeof(msg),
                    "WARNING DAS output: %zu samples (fir_in=%zu, diff=%lld, "
                    "lat=%d, align16=%s, marker_phase=%d+%d)",
                    dsd_out_count, fir_counts[0], diff, lat,
                    align_ok ? "ok" : "FAIL",
                    s->dop_marker_phase, marker_phase_delta);
                trellis_log_c(msg);
            } else if (s->config.debug_log) {
                char msg[128];
                snprintf(msg, sizeof(msg),
                    "DAS output: %zu samples (diff=%lld, phase=%d)",
                    dsd_out_count, diff, s->dop_marker_phase);
                trellis_log_c(msg);
            }
        }

        /* seg_bufs are cached — no per-chunk cleanup needed */

        /* Accumulate temp SDM diagnostics into persistent SDM for logging */
        if (temp_sdms && temp_sdm_count > 0) {
            for (int i = 0; i < temp_sdm_count; i++) {
                s->channels[0].sdm.conv_fail += temp_sdms[i].conv_fail;
                s->channels[0].sdm.cands_collapse += temp_sdms[i].cands_collapse;
                s->channels[0].sdm.next_filter_drops += temp_sdms[i].next_filter_drops;
                s->channels[0].sdm.total_children += temp_sdms[i].total_children;
            }
        }

        /* FIR tail no longer needed for segment warmup (state seeding replaces it).
         * Keep saving for potential future use. */
        if (s->fir_tail) {
            for (int ch = 0; ch < num_channels; ch++) {
                size_t fir_count = fir_counts[ch];
                if (fir_count >= overlap) {
                    if (!s->fir_tail[ch])
                        s->fir_tail[ch] = (double *)malloc(overlap * sizeof(double));
                    if (s->fir_tail[ch])
                        memcpy(s->fir_tail[ch],
                               fir_data[ch] + fir_count - overlap,
                               overlap * sizeof(double));
                }
            }
            s->fir_tail_len = overlap;
            s->fir_tail_valid = true;
        }
sdm_done:
        ;  /* GPU DAS path jumps here after successful GPU processing */
    } else {
        /* === Channel-parallel path: dispatch full blocks to threadpool.
         * Each channel runs boxcar+SDM independently on separate workers.
         * For stereo, 2 channels run in parallel → halves wall-clock time. */
        if (s->gpu && s->config.gpu_enabled)
            gpu_reset_chunk(s->gpu);  /* reset per-channel boxcar history index */

        channel_block_t blocks[32];
        memset(blocks, 0, (size_t)num_channels * sizeof(channel_block_t));

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
    }

    /* Worker migration tick — evaluate/start probes after SDM completes */
    migration_tick(s);

    if (dsd_out_count == 0)
        return 0;

    /* Warmup mute: on first chunk after flush, replace output with DSD silence.
     * The SDM has processed real audio (integrators are warm), but the output
     * contains a DC startup transient. Muting hides the pop; subsequent chunks
     * start with warm SDM state → clean audio. */
    if (s->needs_warmup) {
        s->needs_warmup = false;
        static const float dsd_0x69[8] = {
            -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f
        };
        for (int ch = 0; ch < num_channels; ch++) {
            for (size_t i = 0; i < dsd_out_count; i++)
                s->ch_out[ch][i] = dsd_0x69[i & 7];
        }
        trellis_log_c("warmup mute: replaced first chunk with DSD silence");
    }


    /* On-demand raw DSD capture (mode=0): grab ±1.0 before DoP packing */
    if (g_audio_capture.mode == 0 &&
        (g_audio_capture.state == CAPTURE_RECORDING || capture_check_armed())) {
        capture_write_dsd(s->ch_out, dsd_out_count, num_channels,
                           s->config.fs_out ? s->config.fs_out : s->config.fs_in);
    }

    LARGE_INTEGER t_pack_start, t_pack_end;
    QueryPerformanceCounter(&t_pack_start);

    /* Check if FIR-only mode (DSD→PCM decimation) */
    bool fir_only = (s->channels && s->channels[0].fir_only);

    if (fir_only) {
        /* DSD→PCM: output is already multi-bit PCM, interleave as float.
         * Reinterpret out_i24 as float* — caller provides float-sized buffer. */
        float *out_f = (float *)out_i24;
        for (int ch = 0; ch < num_channels; ch++) {
            for (size_t f = 0; f < dsd_out_count; f++)
                out_f[f * (size_t)num_channels + (size_t)ch] = s->ch_out[ch][f];
        }
        QueryPerformanceCounter(&t_pack_end);
        s->time_pack_ms = perf_ms(t_pack_start, t_pack_end);
        return dsd_out_count;
    }

    /* Pack DSD to DoP and interleave. ASIO+DSD output plugin detects
     * DoP markers and sends native DSD to the driver. */
    if (dsd_out_count % DOP_BITS_PER_FRAME != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg),
            "WARNING DoP align: dsd_out_count=%zu not multiple of %d, "
            "truncating %zu samples",
            dsd_out_count, DOP_BITS_PER_FRAME,
            dsd_out_count % DOP_BITS_PER_FRAME);
        trellis_log_c(msg);
        dsd_out_count -= dsd_out_count % DOP_BITS_PER_FRAME;
    }
    size_t out_pcm_frames = dsd_out_count / DOP_BITS_PER_FRAME;
    if (out_pcm_frames == 0)
        return 0;

    /* Parallel DoP pack on dedicated IO pool (separate from SDM workers).
     * Uses cached per-channel temp buffers to avoid per-chunk malloc/free. */
    threadpool_t *io_pack = s->io_pool ? s->io_pool : s->pool;
    if (io_pack && num_channels > 0 && !fir_only) {
        /* Grow cached pack temp buffers if needed */
        if (s->cached_pack_temps_sz < out_pcm_frames ||
            s->cached_pack_temps_n < num_channels) {
            /* Free old */
            if (s->cached_pack_temps) {
                for (int j = 0; j < s->cached_pack_temps_n; j++)
                    free(s->cached_pack_temps[j]);
                free(s->cached_pack_temps);
            }
            s->cached_pack_temps = (uint8_t **)calloc((size_t)num_channels, sizeof(uint8_t *));
            if (s->cached_pack_temps) {
                for (int ch = 0; ch < num_channels; ch++)
                    s->cached_pack_temps[ch] = (uint8_t *)malloc(out_pcm_frames * 3);
                s->cached_pack_temps_sz = out_pcm_frames;
                s->cached_pack_temps_n = num_channels;
            }
        }
        if (!s->cached_pack_temps) return 0;

        channel_block_t pack_blocks[32];
        memset(pack_blocks, 0, (size_t)num_channels * sizeof(channel_block_t));
        for (int ch = 0; ch < num_channels; ch++) {
            pack_blocks[ch].mode = BLOCK_MODE_PACK;
            pack_blocks[ch].channel = ch;
            pack_blocks[ch].cfg = &s->config;
            pack_blocks[ch].dsd_channel = s->ch_out[ch];
            pack_blocks[ch].pcm_interleaved = (float *)out_i24; /* reinterpret for i24 */
            pack_blocks[ch].pcm_temp = (float *)s->cached_pack_temps[ch];
            pack_blocks[ch].count = dsd_out_count;
            pack_blocks[ch].num_channels = num_channels;
            pack_blocks[ch].discard = (size_t)s->dop_marker_phase; /* pass marker phase */
            threadpool_submit(io_pack, &pack_blocks[ch]);
        }
        threadpool_wait(io_pack);
        /* Update marker phase from channel 0's result */
        s->dop_marker_phase = (int)((out_pcm_frames + s->dop_marker_phase) & 1);
    } else {
        /* Fallback: sequential pack to i24 */
        if (s->cached_pcm_temp_sz < out_pcm_frames) {
            free(s->cached_pcm_temp);
            s->cached_pcm_temp = (uint8_t *)malloc(out_pcm_frames * 3);
            s->cached_pcm_temp_sz = s->cached_pcm_temp ? out_pcm_frames : 0;
        }
        uint8_t *i24_temp = s->cached_pcm_temp;
        if (!i24_temp) return 0;
        for (int ch = 0; ch < num_channels; ch++) {
            int next_phase = dop_pack_i24(s->ch_out[ch], i24_temp, dsd_out_count, s->dop_marker_phase);
            if (ch == 0) s->dop_marker_phase = next_phase;
            for (size_t f = 0; f < out_pcm_frames; f++) {
                size_t dst = (f * (size_t)num_channels + (size_t)ch) * 3;
                size_t src = f * 3;
                out_i24[dst]     = i24_temp[src];
                out_i24[dst + 1] = i24_temp[src + 1];
                out_i24[dst + 2] = i24_temp[src + 2];
            }
        }
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
                           const float *in_pcm, uint8_t *out_i24,
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
        s->active_sdm_mode != s->config.sdm_mode ||
        s->active_cands != s->config.trellis_cands ||
        s->active_depth != s->config.trellis_depth ||
        s->active_gpu != s->config.gpu_enabled) {
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
        s->cached_pcm_temp = (uint8_t *)malloc(out_pcm_frames * 3);
        s->cached_pcm_temp_sz = s->cached_pcm_temp ? out_pcm_frames : 0;
    }
    uint8_t *i24_temp = s->cached_pcm_temp;
    if (!i24_temp)
        return 0;

    for (int ch = 0; ch < num_channels; ch++) {
        int next_phase = dop_pack_i24(s->ch_out[ch], i24_temp, sdm_out_count, s->dop_marker_phase);
        if (ch == 0) s->dop_marker_phase = next_phase;
        for (size_t f = 0; f < out_pcm_frames; f++) {
            size_t dst = (f * (size_t)num_channels + (size_t)ch) * 3;
            size_t src = f * 3;
            out_i24[dst]     = i24_temp[src];
            out_i24[dst + 1] = i24_temp[src + 1];
            out_i24[dst + 2] = i24_temp[src + 2];
        }
    }

    QueryPerformanceCounter(&t_pack_end);
    s->time_pack_ms = perf_ms(t_pack_start, t_pack_end);

    return out_pcm_frames;
}

/*
 * Process an interleaved PCM chunk and convert to a different PCM sample rate.
 * Same-family: uses FIR chain (power-of-2 ratio).
 * Cross-family: uses IPP polyphase or libsoxr.
 *
 * in_pcm:      interleaved float32 PCM input
 * out_pcm:     output float32 buffer (caller-allocated)
 * pcm_frames:  number of PCM frames (per channel) in input
 * num_channels: channel count
 * pcm_rate_in: input PCM sample rate
 * pcm_rate_out: output PCM sample rate
 *
 * Returns: number of output PCM frames per channel, or 0 on error.
 */
size_t plugin_process_pcm_to_pcm(plugin_state_t *s,
                                  const float *in_pcm, float *out_pcm,
                                  size_t pcm_frames, int num_channels,
                                  uint32_t pcm_rate_in, uint32_t pcm_rate_out) {
    if (!s || pcm_rate_in == 0 || pcm_rate_out == 0 || num_channels == 0)
        return 0;

    if (pcm_rate_in == pcm_rate_out) {
        /* Passthrough — shouldn't reach here, but handle gracefully */
        memcpy(out_pcm, in_pcm, pcm_frames * (size_t)num_channels * sizeof(float));
        return pcm_frames;
    }

    bool needs_polyphase = resample_needed(pcm_rate_in, pcm_rate_out);

    /* Estimate max output frames */
    size_t max_out = (size_t)((double)pcm_frames * (double)pcm_rate_out / (double)pcm_rate_in) + 256;

    /* Per-channel processing */
    float *ch_in  = (float *)malloc(pcm_frames * sizeof(float));
    float *ch_out = (float *)malloc(max_out * sizeof(float));
    if (!ch_in || !ch_out) { free(ch_in); free(ch_out); return 0; }

    size_t out_frames = 0;

    for (int ch = 0; ch < num_channels; ch++) {
        /* De-interleave */
        for (size_t f = 0; f < pcm_frames; f++)
            ch_in[f] = in_pcm[f * (size_t)num_channels + (size_t)ch];

        size_t produced = 0;

        if (needs_polyphase) {
            /* Cross-family: polyphase resampler */
            resample_ctx_t *rs = resample_create(pcm_rate_in, pcm_rate_out,
                                                  s->config.resample_engine,
                                                  s->config.soxr_quality);
            if (!rs) { free(ch_in); free(ch_out); return 0; }
            produced = resample_process(rs, ch_in, ch_out, pcm_frames);
            resample_free(rs);
        } else {
            /* Same-family: FIR chain (power-of-2 ratio) */
            fir_chain_t fir;
            if (fir_chain_init(&fir, pcm_rate_in, pcm_rate_out) != 0) {
                free(ch_in); free(ch_out);
                return 0;
            }
            produced = fir_chain_process(&fir, ch_in, ch_out, pcm_frames);
            fir_chain_free(&fir);
        }

        if (ch == 0) out_frames = produced;

        /* Re-interleave */
        for (size_t f = 0; f < produced; f++)
            out_pcm[f * (size_t)num_channels + (size_t)ch] = ch_out[f];
    }

    free(ch_in);
    free(ch_out);
    return out_frames;
}

/*
 * Drain remaining SDM latency at end of playback.
 * Returns number of output PCM frames per channel.
 */
size_t plugin_drain(plugin_state_t *s, uint8_t *out_i24, int num_channels) {
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

    uint8_t *i24_temp = (uint8_t *)malloc((max_drain / DOP_BITS_PER_FRAME + 1) * 3);
    if (!i24_temp) {
        free(drain_buf);
        return 0;
    }

    for (int ch = 0; ch < num_channels; ch++) {
        size_t drained = sdm_drain(&s->channels[ch].sdm, drain_buf, max_drain);
        if (drained < min_drained)
            min_drained = drained;

        /* Pack to DoP i24 and interleave */
        size_t pcm_frames = drained / DOP_BITS_PER_FRAME;
        dop_pack_i24(drain_buf, i24_temp, drained, s->dop_marker_phase);

        for (size_t f = 0; f < pcm_frames; f++) {
            size_t dst = (f * (size_t)num_channels + (size_t)ch) * 3;
            size_t src = f * 3;
            out_i24[dst]     = i24_temp[src];
            out_i24[dst + 1] = i24_temp[src + 1];
            out_i24[dst + 2] = i24_temp[src + 2];
        }
    }

    size_t out_frames = min_drained / DOP_BITS_PER_FRAME;

    free(drain_buf);
    free(i24_temp);

    return out_frames;
}

/*
 * Generate trailing DSD silence by feeding zeros through the SDM engine.
 * The SDM winds down naturally from the last music state → silence,
 * avoiding the hard discontinuity of raw idle pattern insertion.
 * Returns number of output DoP PCM frames per channel.
 */
size_t plugin_generate_tail(plugin_state_t *s, uint8_t *out_i24,
                            int num_channels, int ms) {
    if (!s || !s->initialized || !s->channels || num_channels <= 0 || ms <= 0)
        return 0;

    uint32_t fs_out = s->config.fs_out ? s->config.fs_out : s->config.fs_in;
    if (fs_out == 0)
        return 0;

    /* DSD samples for the requested duration */
    size_t dsd_count = (size_t)((uint64_t)fs_out * (uint64_t)ms / 1000);
    if (dsd_count == 0)
        return 0;

    /* Zero input = silence fed through the SDM */
    float *sil_in  = (float *)calloc(dsd_count, sizeof(float));
    float *dsd_out = (float *)malloc(dsd_count * sizeof(float));
    size_t pcm_frames = dsd_count / DOP_BITS_PER_FRAME;
    uint8_t *i24_temp = (uint8_t *)malloc((pcm_frames + 1) * 3);
    if (!sil_in || !dsd_out || !i24_temp) {
        free(sil_in); free(dsd_out); free(i24_temp);
        return 0;
    }

    int ch_count = num_channels < s->num_channels ? num_channels : s->num_channels;
    for (int ch = 0; ch < ch_count; ch++) {
        size_t n = engine_process_block(&s->channels[ch], sil_in, dsd_out,
                                         dsd_count, &s->config);
        if (n == 0) continue;

        size_t frames = n / DOP_BITS_PER_FRAME;
        if (frames > pcm_frames) frames = pcm_frames;
        dop_pack_i24(dsd_out, i24_temp, frames * DOP_BITS_PER_FRAME, 0);

        for (size_t f = 0; f < frames; f++) {
            size_t dst = (f * (size_t)num_channels + (size_t)ch) * 3;
            size_t src = f * 3;
            out_i24[dst]     = i24_temp[src];
            out_i24[dst + 1] = i24_temp[src + 1];
            out_i24[dst + 2] = i24_temp[src + 2];
        }
    }

    free(sil_in);
    free(dsd_out);
    free(i24_temp);

    return pcm_frames;
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
    /* Always fully reset SDM state on flush. With state-seeded parallelism,
     * the persistent SDM gets overwritten by temp SDM state each chunk.
     * Preserving this stale state across stop→play causes massive pops.
     * Anti-pop lead-in silence handles DAC mode-switch pops instead. */
    for (int i = 0; i < s->num_channels; i++)
        engine_channel_reset(&s->channels[i], false);
    s->fir_tail_valid = false;
    s->needs_warmup = true;  /* prime SDM with real audio on next chunk */
    s->dop_marker_phase = 0; /* reset DoP marker phase */
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
