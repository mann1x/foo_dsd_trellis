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

    free(s);
}

/* Initialize engine for given channel count and config.
 * Called when we first detect DSD rate in on_chunk. */
static int plugin_init_engine(plugin_state_t *s, int num_channels,
                               uint32_t dsd_rate) {
    if (s->initialized)
        return 0;

    s->config.fs_in = dsd_rate;
    s->num_channels = num_channels;
    s->detected_dsd_rate = dsd_rate;

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

    s->initialized = true;
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

    /* Initialize engine on first use or channel count change */
    if (!s->initialized || s->num_channels != num_channels ||
        s->detected_dsd_rate != dsd_rate) {
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
        s->config.fs_in = dsd_rate;
        if (plugin_init_engine(s, num_channels, dsd_rate) != 0)
            return 0;
    }

    /* Check for system CPUSET changes (CPUDoc dynamic core management) */
    s->cpuset_changed = false;
    if (s->topology_detected) {
        bool mask_changed = false;
        uint64_t new_mask = cpuset_refresh(&s->topology, &mask_changed);
        if (mask_changed && s->pool) {
            s->cpuset_changed = true;
            s->last_cpuset_mask = new_mask;

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
            if (selected > 0)
                s->pool = threadpool_create_cpuset(selected_ids, selected);
            if (!s->pool)
                s->pool = threadpool_create(
                    s->config.thread_count > 0 ? s->config.thread_count : 0,
                    s->config.affinity_mask);
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

        /* Ensure minimum segment size (at least 4x overlap) */
        size_t min_seg = overlap * 4;
        if (min_seg < 256) min_seg = 256;
        while (segments_per_ch > 1 && fir_out_est / (size_t)segments_per_ch < min_seg)
            segments_per_ch--;
    }

    /* Track workload changes */
    s->workload_changed = (num_threads != s->last_num_threads ||
                            segments_per_ch != s->last_segments_per_ch);
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

        /* Phase 2: Get temp SDM contexts for segments 1..N-1 (cached) */
        int temp_sdm_count = num_channels * (segments_per_ch - 1);
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

        /* Re-init temp SDM contexts (must reset state each chunk) */
        bool init_ok = true;
        for (int i = 0; i < temp_sdm_count; i++) {
            sdm_context_free(&temp_sdms[i]);
            if (sdm_context_init(&temp_sdms[i], filter,
                                  s->config.trellis_depth,
                                  s->config.trellis_cands,
                                  s->config.trellis_lat) != 0) {
                init_ok = false;
                break;
            }
        }

        if (!init_ok)
            return 0;

        /* Phase 3: Setup and submit SDM segment blocks (cached) */
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

        for (int ch = 0; ch < num_channels; ch++) {
            size_t fir_count = fir_counts[ch];
            size_t base_seg = fir_count / (size_t)segments_per_ch;
            size_t remainder = fir_count % (size_t)segments_per_ch;

            /* Compute segment 0 expected output for offset calculation */
            size_t pending = s->channels[ch].sdm.pending;
            size_t lat_rem = (pending < (size_t)s->config.trellis_lat) ?
                ((size_t)s->config.trellis_lat - pending) : 0;
            size_t seg0_size = base_seg + (0 < remainder ? 1 : 0);
            size_t seg0_out = (seg0_size > lat_rem) ? (seg0_size - lat_rem) : 0;

            /* Segment 0: uses channel's persistent SDM context */
            int bi = ch * segments_per_ch;
            blocks[bi].mode     = BLOCK_MODE_SDM;
            blocks[bi].sdm_ctx  = &s->channels[ch].sdm;
            blocks[bi].in       = fir_data[ch];
            blocks[bi].out      = s->ch_out[ch];
            blocks[bi].count    = seg0_size;
            blocks[bi].discard  = 0;
            blocks[bi].channel  = ch;

            /* Segments 1..N-1: temp SDM contexts with overlap warmup */
            size_t seg_start = seg0_size;
            size_t out_offset = seg0_out;

            for (int seg = 1; seg < segments_per_ch; seg++) {
                size_t this_seg;
                if (seg == segments_per_ch - 1)
                    this_seg = fir_count - seg_start;
                else
                    this_seg = base_seg + ((size_t)seg < remainder ? 1 : 0);

                int temp_idx = ch * (segments_per_ch - 1) + (seg - 1);
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

        /* Note: temp_sdms and blocks are cached in plugin_state,
         * freed on destroy or when reallocated. */
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

    /* Repack DSD floats to DoP PCM, then interleave into output.
     * Output PCM frames = dsd_out_count / 16 */
    size_t out_pcm_frames = dsd_out_count / DOP_BITS_PER_FRAME;
    if (out_pcm_frames == 0)
        return 0;

    LARGE_INTEGER t_pack_start, t_pack_end;
    QueryPerformanceCounter(&t_pack_start);

    /* Reuse cached pcm_temp, ensure it's large enough for output */
    if (s->cached_pcm_temp_sz < out_pcm_frames) {
        free(s->cached_pcm_temp);
        s->cached_pcm_temp = (float *)malloc(out_pcm_frames * sizeof(float));
        s->cached_pcm_temp_sz = s->cached_pcm_temp ? out_pcm_frames : 0;
    }
    pcm_temp = s->cached_pcm_temp;
    if (!pcm_temp)
        return 0;

    for (int ch = 0; ch < num_channels; ch++) {
        /* Pack DSD bits to DoP PCM */
        dop_pack(s->ch_out[ch], pcm_temp, dsd_out_count);

        /* Interleave into output */
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

    /* PreCorr has no latency */
    if (s->config.sdm_mode == SDM_MODE_PRECORR)
        return 0.0;

    uint32_t fs_out = s->config.fs_out ? s->config.fs_out : s->config.fs_in;
    /* Latency is trellis_lat DSD samples at the output DSD rate,
     * expressed as DoP PCM frames / DoP PCM rate */
    double dsd_lat_sec = (double)s->config.trellis_lat / (double)fs_out;
    return dsd_lat_sec;
}

/* Reset all channel states (on seek / discontinuity) */
void plugin_flush(plugin_state_t *s) {
    if (!s || !s->initialized)
        return;
    for (int i = 0; i < s->num_channels; i++)
        engine_channel_reset(&s->channels[i]);
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
