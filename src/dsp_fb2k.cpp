/*
 * foo_dsd_trellis — foobar2000 DSP v2 C++ wrapper
 *
 * Thin C++ layer implementing the fb2k service interface.
 * Each DSP instance owns a plugin_state_t for independent processing.
 * Detects DoP in incoming chunks, processes through the DSD engine,
 * and returns DoP-encoded output.
 */

#include "stdafx.h"

extern "C" {
#include "../include/dsd_types.h"

/* Per-instance plugin state (opaque, defined in dsp_plugin.c) */
typedef struct plugin_state plugin_state_t;

plugin_state_t *plugin_create(void);
void            plugin_destroy(plugin_state_t *s);
void            plugin_set_config(plugin_state_t *s, const dsd_config_t *cfg);
const dsd_config_t *plugin_get_config(const plugin_state_t *s);
size_t          plugin_process(plugin_state_t *s,
                               const float *in_pcm, float *out_pcm,
                               size_t pcm_frames, int num_channels,
                               uint32_t pcm_rate);
size_t          plugin_drain(plugin_state_t *s, float *out_pcm,
                             int num_channels);
double          plugin_get_latency(const plugin_state_t *s);
void            plugin_flush(plugin_state_t *s);
int             plugin_reconfigure(plugin_state_t *s, const dsd_config_t *cfg);
}

/* {7A3F2D1E-B4C5-4E6F-8A9B-0C1D2E3F4A5B} */
static const GUID g_dsp_guid =
    { 0x7a3f2d1e, 0xb4c5, 0x4e6f,
      { 0x8a, 0x9b, 0x0c, 0x1d, 0x2e, 0x3f, 0x4a, 0x5b } };

static void make_preset(const dsd_config_t &cfg, dsp_preset &out) {
    out.set_owner(g_dsp_guid);
    out.set_data(&cfg, sizeof(cfg));
}

static dsd_config_t parse_preset(const dsp_preset &in) {
    dsd_config_t cfg;
    if (in.get_owner() == g_dsp_guid && in.get_data_size() == sizeof(cfg)) {
        memcpy(&cfg, in.get_data(), sizeof(cfg));
    } else {
        dsd_config_defaults(&cfg);
    }
    return cfg;
}

class dsp_dsd_trellis : public dsp_impl_base {
public:
    dsp_dsd_trellis(dsp_preset const &preset)
        : m_config(parse_preset(preset))
        , m_state(plugin_create())
        , m_channels(0)
        , m_pcm_rate(0)
    {
        if (m_state)
            plugin_set_config(m_state, &m_config);
    }

    ~dsp_dsd_trellis() {
        plugin_destroy(m_state);
    }

    static GUID g_get_guid() { return g_dsp_guid; }

    static void g_get_name(pfc::string_base &out) {
        out = "DSD Trellis SDM";
    }

    static bool g_get_default_preset(dsp_preset &out) {
        dsd_config_t cfg;
        dsd_config_defaults(&cfg);
        make_preset(cfg, out);
        return true;
    }

    static bool g_have_config_popup() {
        return true;
    }

#ifdef _WIN32
    static void g_show_config_popup(const dsp_preset &p_data,
                                    HWND p_parent,
                                    dsp_preset_edit_callback & /*p_callback*/) {
        dsd_config_t cfg = parse_preset(p_data);

        /* Simple message box showing current settings.
         * Full property page dialog deferred to Phase 7. */
        pfc::string8 msg;
        msg << "DSD Trellis SDM Settings\n\n"
            << "Trellis Depth: " << cfg.trellis_depth << "\n"
            << "Candidates: " << cfg.trellis_cands << "\n"
            << "Latency: " << cfg.trellis_lat << "\n"
            << "Gain: " << pfc::format_float(cfg.gain, 0, 2) << "\n"
            << "NTF Filter: " << cfg.ntf_filter << " (Auto=-1)\n"
            << "\nEdit settings via the preset system.";

        MessageBoxA(p_parent, msg.c_str(), "DSD Trellis SDM", MB_OK);
    }
#endif

    bool on_chunk(audio_chunk *chunk, abort_callback & /*abort*/) override {
        if (!m_state)
            return true;  /* Pass through on allocation failure */

        const unsigned channels = chunk->get_channel_count();
        const unsigned pcm_rate = chunk->get_sample_rate();
        const t_size pcm_frames = chunk->get_sample_count();

        if (channels == 0 || pcm_frames == 0)
            return true;

        /* Track channel/rate changes for latency reporting */
        m_channels = (int)channels;
        m_pcm_rate = pcm_rate;

        /* Convert audio_sample (double on x64) to float for C engine */
        const audio_sample *src = chunk->get_data();
        size_t total_in = pcm_frames * channels;
        pfc::array_staticsize_t<float> in_f32;
        in_f32.set_size_discard(total_in);
        for (size_t i = 0; i < total_in; i++)
            in_f32[i] = (float)src[i];

        /* Allocate output buffer: worst case is 8x rate conversion.
         * Output is interleaved: out_frames * channels. */
        size_t max_out_frames = pcm_frames * 8;
        pfc::array_staticsize_t<float> out_buf;
        out_buf.set_size_discard(max_out_frames * channels);

        size_t out_frames = plugin_process(
            m_state, in_f32.get_ptr(), out_buf.get_ptr(),
            pcm_frames, (int)channels, pcm_rate);

        if (out_frames == 0) {
            /* Not DoP or processing failed — pass through unmodified */
            return true;
        }

        /* Determine output PCM sample rate */
        uint32_t dsd_rate_in = pcm_rate * 16;
        uint32_t dsd_rate_out = m_config.fs_out ? m_config.fs_out : dsd_rate_in;
        uint32_t out_pcm_rate = dsd_rate_out / 16;

        /* Convert float output back to audio_sample and update chunk */
        size_t total_out = out_frames * channels;
        pfc::array_staticsize_t<audio_sample> out_as;
        out_as.set_size_discard(total_out);
        for (size_t i = 0; i < total_out; i++)
            out_as[i] = (audio_sample)out_buf[i];
        chunk->set_data(out_as.get_ptr(), out_frames, channels, out_pcm_rate);

        return true;
    }

    void on_endofplayback(abort_callback & /*abort*/) override {
        if (!m_state || m_channels == 0)
            return;

        /* Drain remaining SDM latency and output as a final chunk */
        size_t max_drain_frames = 2048 / 16;  /* max latency / bits per frame */
        pfc::array_staticsize_t<float> drain_buf;
        drain_buf.set_size_discard(max_drain_frames * (unsigned)m_channels);

        size_t drain_frames = plugin_drain(m_state, drain_buf.get_ptr(),
                                           m_channels);

        if (drain_frames > 0 && m_pcm_rate > 0) {
            uint32_t dsd_rate_in = m_pcm_rate * 16;
            uint32_t dsd_rate_out = m_config.fs_out ? m_config.fs_out
                                                     : dsd_rate_in;
            uint32_t out_pcm_rate = dsd_rate_out / 16;

            /* Convert float to audio_sample */
            size_t total = drain_frames * (unsigned)m_channels;
            pfc::array_staticsize_t<audio_sample> drain_as;
            drain_as.set_size_discard(total);
            for (size_t i = 0; i < total; i++)
                drain_as[i] = (audio_sample)drain_buf[i];

            audio_chunk_impl chunk_out;
            chunk_out.set_data(drain_as.get_ptr(), drain_frames,
                               (unsigned)m_channels, out_pcm_rate);
            insert_chunk(chunk_out);
        }
    }

    void on_endoftrack(abort_callback & /*abort*/) override {
        /* No action needed */
    }

    void flush() override {
        plugin_flush(m_state);
    }

    double get_latency() override {
        return plugin_get_latency(m_state);
    }

    bool need_track_change_mark() override {
        return false;
    }

private:
    dsd_config_t     m_config;
    plugin_state_t  *m_state;
    int              m_channels;
    unsigned         m_pcm_rate;
};

static dsp_factory_t<dsp_dsd_trellis> g_dsp_factory;

DECLARE_COMPONENT_VERSION(
    "DSD Trellis SDM",
    "0.1.0",
    "DSD Trellis (Viterbi) Sigma-Delta Modulator\n"
    "Rate conversion, volume control, and noise shaping for DSD streams.\n\n"
    "NTF coefficients ported from mansr/sox sdm.c (LGPL v2.1+)"
);

VALIDATE_COMPONENT_FILENAME("foo_dsd_trellis.dll");
