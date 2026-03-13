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
#include "../include/dop.h"
#include "../include/threadpool.h"

/*
 * Plugin identity
 */
#define PLUGIN_NAME        "DSD Trellis SDM"
#define PLUGIN_VERSION     "0.1.0"

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
} plugin_state_t;

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

    int tc = s->config.thread_count > 0 ? s->config.thread_count : 0;
    s->pool = threadpool_create(tc, s->config.affinity_mask);
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
    float *pcm_temp = (float *)malloc(pcm_frames * sizeof(float));
    if (!pcm_temp)
        return 0;

    for (int ch = 0; ch < num_channels; ch++) {
        /* Extract strided PCM for this channel */
        for (size_t f = 0; f < pcm_frames; f++)
            pcm_temp[f] = in_pcm[f * (size_t)num_channels + (size_t)ch];

        /* Unpack DoP to DSD floats */
        dop_unpack(pcm_temp, s->ch_in[ch], pcm_frames);
    }

    free(pcm_temp);

    /* Dispatch all channels to thread pool */
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
        threadpool_submit(s->pool, &blocks[ch]);
    }
    threadpool_wait(s->pool);

    /* All channels should produce the same output count */
    size_t dsd_out_count = blocks[0].out_count;
    free(blocks);

    if (dsd_out_count == 0)
        return 0;

    /* Repack DSD floats to DoP PCM, then interleave into output.
     * Output PCM frames = dsd_out_count / 16 */
    size_t out_pcm_frames = dsd_out_count / DOP_BITS_PER_FRAME;
    if (out_pcm_frames == 0)
        return 0;

    pcm_temp = (float *)malloc(out_pcm_frames * sizeof(float));
    if (!pcm_temp)
        return 0;

    for (int ch = 0; ch < num_channels; ch++) {
        /* Pack DSD bits to DoP PCM */
        dop_pack(s->ch_out[ch], pcm_temp, dsd_out_count);

        /* Interleave into output */
        for (size_t f = 0; f < out_pcm_frames; f++)
            out_pcm[f * (size_t)num_channels + (size_t)ch] = pcm_temp[f];
    }

    free(pcm_temp);

    return out_pcm_frames;
}

/*
 * Drain remaining SDM latency at end of playback.
 * Returns number of output PCM frames per channel.
 */
size_t plugin_drain(plugin_state_t *s, float *out_pcm, int num_channels) {
    if (!s || !s->initialized || !s->channels)
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
