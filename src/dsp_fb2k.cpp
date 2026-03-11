/*
 * foo_dsd_trellis — foobar2000 DSP v2 C++ wrapper
 *
 * Thin C++ layer implementing the fb2k service interface.
 * Delegates all processing to the C engine (dsp_plugin.c).
 *
 * Phase 0: Scaffold — passthrough only.
 */

#include "stdafx.h"

extern "C" {
#include "../include/dsd_types.h"

/* C-side plugin management from dsp_plugin.c */
int              plugin_init(int num_channels, const dsd_config_t *cfg);
void             plugin_shutdown(void);
int              plugin_reconfigure(const dsd_config_t *cfg);
const dsd_config_t *plugin_get_config(void);
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
    {
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
        /* TODO (Phase 6): Return true when config dialog is implemented */
        return false;
    }

#ifdef _WIN32
    static void g_show_config_popup(const dsp_preset & /*p_data*/,
                                    HWND /*p_parent*/,
                                    dsp_preset_edit_callback & /*p_callback*/) {
        /* TODO (Phase 6): Property page dialog */
    }
#endif

    bool on_chunk(audio_chunk *chunk, abort_callback & /*abort*/) override {
        /* Phase 0: passthrough — return chunk unmodified */
        (void)chunk;
        return true;
    }

    void on_endofplayback(abort_callback & /*abort*/) override {
        /* TODO (Phase 6): Flush SDM latency buffer */
    }

    void on_endoftrack(abort_callback & /*abort*/) override {
        /* No action needed unless need_track_change_mark() returns true */
    }

    void flush() override {
        /* Reset SDM state on seek */
    }

    double get_latency() override {
        return 0.0;
    }

    bool need_track_change_mark() override {
        return false;
    }

private:
    dsd_config_t m_config;
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
