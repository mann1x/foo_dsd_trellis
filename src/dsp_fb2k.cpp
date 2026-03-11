/*
 * foo_dsd_trellis — foobar2000 DSP v2 C++ wrapper
 *
 * Thin C++ layer implementing the fb2k service interface.
 * Each DSP instance owns a plugin_state_t for independent processing.
 * Detects DoP in incoming chunks, processes through the DSD engine,
 * and returns DoP-encoded or PCM output.
 */

#include "stdafx.h"
#include "resource.h"
#include <helpers/DarkMode.h>
#include <math.h>

extern "C" {
#include "../include/dsd_types.h"
#include "../include/simd_detect.h"
#include "../include/fir.h"

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

/* Config serialization (config.c) */
size_t config_serialize(const dsd_config_t *cfg, uint8_t *buf, size_t buf_size);
int    config_deserialize(dsd_config_t *cfg, const uint8_t *buf, size_t buf_size);
void   config_validate(dsd_config_t *cfg);
}

/* ─── Gain helpers ─── */

static inline float gain_to_db(float linear) {
    if (linear <= 0.0f) return -120.0f;
    return 20.0f * log10f(linear);
}

static inline float db_to_gain(float db) {
    if (db <= -120.0f) return 0.0f;
    return powf(10.0f, db / 20.0f);
}

/* {7A3F2D1E-B4C5-4E6F-8A9B-0C1D2E3F4A5B} */
static const GUID g_dsp_guid =
    { 0x7a3f2d1e, 0xb4c5, 0x4e6f,
      { 0x8a, 0x9b, 0x0c, 0x1d, 0x2e, 0x3f, 0x4a, 0x5b } };

/* ─── Preset serialization using dsp_preset_builder/parser ─── */

static void make_preset(const dsd_config_t &cfg, dsp_preset &out) {
    uint8_t buf[128];
    size_t len = config_serialize(&cfg, buf, sizeof(buf));
    out.set_owner(g_dsp_guid);
    out.set_data(buf, len);
}

static dsd_config_t parse_preset(const dsp_preset &in) {
    dsd_config_t cfg;
    dsd_config_defaults(&cfg);

    if (in.get_owner() != g_dsp_guid) return cfg;

    const void *data = in.get_data();
    t_size size = in.get_data_size();

    if (data && size > 0) {
        config_deserialize(&cfg, (const uint8_t *)data, (size_t)size);
    }

    return cfg;
}

/* ─── Property page dialog ─── */

class CDSPTrellisPopup : public CDialogImpl<CDSPTrellisPopup> {
public:
    CDSPTrellisPopup(const dsp_preset &initData, dsp_preset_edit_callback &callback)
        : m_initData(initData), m_callback(callback) {}

    enum { IDD = IDD_DSP_TRELLIS };

    BEGIN_MSG_MAP_EX(CDSPTrellisPopup)
        MSG_WM_INITDIALOG(OnInitDialog)
        COMMAND_HANDLER_EX(IDOK, BN_CLICKED, OnButton)
        COMMAND_HANDLER_EX(IDCANCEL, BN_CLICKED, OnButton)
        COMMAND_HANDLER_EX(IDC_COMBO_OUTPUT_RATE, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_COMBO_TRELLIS_DEPTH, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_COMBO_NTF, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_COMBO_FORMAT, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_COMBO_OUTPUT_FORMAT, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_CHECK_MUTE, BN_CLICKED, OnChange)
        COMMAND_HANDLER_EX(IDC_EDIT_GAIN, EN_CHANGE, OnEditChange)
        COMMAND_HANDLER_EX(IDC_EDIT_TRELLIS_CANDS, EN_CHANGE, OnEditChange)
        COMMAND_HANDLER_EX(IDC_EDIT_TRELLIS_LAT, EN_CHANGE, OnEditChange)
        COMMAND_HANDLER_EX(IDC_EDIT_THREADS, EN_CHANGE, OnEditChange)
    END_MSG_MAP()

private:
    BOOL OnInitDialog(CWindow, LPARAM) {
        m_dark.AddDialogWithControls(m_hWnd);

        dsd_config_t cfg = parse_preset(m_initData);

        /* Output Rate combo */
        CComboBox rate(GetDlgItem(IDC_COMBO_OUTPUT_RATE));
        rate.AddString(L"As Input");
        rate.AddString(L"DSD64");
        rate.AddString(L"DSD128");
        rate.AddString(L"DSD256");
        rate.AddString(L"DSD512");
        int rate_idx = 0;
        switch (cfg.fs_out) {
        case DSD_RATE_64:  rate_idx = 1; break;
        case DSD_RATE_128: rate_idx = 2; break;
        case DSD_RATE_256: rate_idx = 3; break;
        case DSD_RATE_512: rate_idx = 4; break;
        }
        rate.SetCurSel(rate_idx);

        /* Gain in dB */
        float db = gain_to_db(cfg.gain);
        pfc::string_formatter gain_str;
        gain_str << pfc::format_float(db, 0, 1);
        ::uSetDlgItemText(*this, IDC_EDIT_GAIN, gain_str);
        RefreshGainLabel(db);

        /* Trellis depth combo */
        CComboBox depth(GetDlgItem(IDC_COMBO_TRELLIS_DEPTH));
        depth.AddString(L"4");
        depth.AddString(L"8");
        depth.AddString(L"16");
        depth.AddString(L"32");
        int depth_idx = 1;
        if (cfg.trellis_depth <= 4) depth_idx = 0;
        else if (cfg.trellis_depth <= 8) depth_idx = 1;
        else if (cfg.trellis_depth <= 16) depth_idx = 2;
        else depth_idx = 3;
        depth.SetCurSel(depth_idx);

        /* Candidates */
        SetDlgItemInt(IDC_EDIT_TRELLIS_CANDS, cfg.trellis_cands, FALSE);
        CUpDownCtrl(GetDlgItem(IDC_SPIN_TRELLIS_CANDS)).SetRange(4, 32);

        /* Latency */
        SetDlgItemInt(IDC_EDIT_TRELLIS_LAT, cfg.trellis_lat, FALSE);
        CUpDownCtrl(GetDlgItem(IDC_SPIN_TRELLIS_LAT)).SetRange(16, 2048);

        /* NTF filter combo */
        CComboBox ntf(GetDlgItem(IDC_COMBO_NTF));
        ntf.AddString(L"Auto");
        ntf.AddString(L"CLANS-4");
        ntf.AddString(L"SDM-4");
        ntf.AddString(L"CLANS-5");
        ntf.AddString(L"SDM-5");
        ntf.AddString(L"CLANS-6");
        ntf.AddString(L"SDM-6");
        ntf.AddString(L"CLANS-7");
        ntf.AddString(L"SDM-7");
        ntf.AddString(L"CLANS-8");
        ntf.AddString(L"SDM-8");
        ntf.SetCurSel(cfg.ntf_filter + 1); /* NTF_AUTO = -1 → index 0 */

        /* Threads */
        SetDlgItemInt(IDC_EDIT_THREADS, cfg.thread_count, FALSE);
        CUpDownCtrl(GetDlgItem(IDC_SPIN_THREADS)).SetRange(0, 32);

        /* Input format combo */
        CComboBox fmt(GetDlgItem(IDC_COMBO_FORMAT));
        fmt.AddString(L"Auto");
        fmt.AddString(L"DoP");
        fmt.AddString(L"Native");
        fmt.SetCurSel(cfg.format);

        /* Output format combo */
        CComboBox ofmt(GetDlgItem(IDC_COMBO_OUTPUT_FORMAT));
        ofmt.AddString(L"DoP (DSD output)");
        ofmt.AddString(L"PCM (for VU / non-DSD DAC)");
        ofmt.SetCurSel(cfg.output_format);

        /* Mute checkbox */
        CheckDlgButton(IDC_CHECK_MUTE, cfg.mute ? BST_CHECKED : BST_UNCHECKED);

        /* Show SIMD engine info */
        const cpu_features_t *cpu = cpu_detect();
        const char *simd = fir_simd_name();
        pfc::string_formatter info;
        info << "Engine: " << simd
             << " | CPU: " << (cpu->vendor == CPU_VENDOR_INTEL ? "Intel" :
                               cpu->vendor == CPU_VENDOR_AMD   ? "AMD" : "Unknown");
        ::uSetDlgItemText(*this, IDC_STATIC_VU_L, info);

        return TRUE;
    }

    void OnButton(UINT, int id, CWindow) {
        EndDialog(id);
    }

    void OnChange(UINT, int, CWindow) {
        if (!m_updating) UpdatePreset();
    }

    void OnEditChange(UINT, int, CWindow) {
        if (!m_updating) UpdatePreset();
    }

    void UpdatePreset() {
        m_updating = true;

        dsd_config_t cfg;
        dsd_config_defaults(&cfg);

        /* Read output rate */
        int rate_idx = CComboBox(GetDlgItem(IDC_COMBO_OUTPUT_RATE)).GetCurSel();
        static const uint32_t rates[] = { 0, DSD_RATE_64, DSD_RATE_128, DSD_RATE_256, DSD_RATE_512 };
        cfg.fs_out = (rate_idx >= 0 && rate_idx < 5) ? rates[rate_idx] : 0;

        /* Read gain */
        pfc::string8 gain_text;
        uGetDlgItemText(*this, IDC_EDIT_GAIN, gain_text);
        float db = (float)atof(gain_text.c_str());
        if (db > 0.0f) db = 0.0f;
        if (db < -120.0f) db = -120.0f;
        cfg.gain = db_to_gain(db);
        RefreshGainLabel(db);

        /* Read trellis depth */
        static const int depths[] = { 4, 8, 16, 32 };
        int depth_idx = CComboBox(GetDlgItem(IDC_COMBO_TRELLIS_DEPTH)).GetCurSel();
        cfg.trellis_depth = (depth_idx >= 0 && depth_idx < 4) ? depths[depth_idx] : 8;

        /* Read candidates and latency */
        cfg.trellis_cands = (int)GetDlgItemInt(IDC_EDIT_TRELLIS_CANDS, NULL, FALSE);
        cfg.trellis_lat = (int)GetDlgItemInt(IDC_EDIT_TRELLIS_LAT, NULL, FALSE);

        /* NTF filter */
        int ntf_idx = CComboBox(GetDlgItem(IDC_COMBO_NTF)).GetCurSel();
        cfg.ntf_filter = ntf_idx - 1; /* index 0 = Auto = -1 */

        /* Threads */
        cfg.thread_count = (int)GetDlgItemInt(IDC_EDIT_THREADS, NULL, FALSE);

        /* Input format */
        cfg.format = CComboBox(GetDlgItem(IDC_COMBO_FORMAT)).GetCurSel();

        /* Output format */
        cfg.output_format = CComboBox(GetDlgItem(IDC_COMBO_OUTPUT_FORMAT)).GetCurSel();

        /* Mute */
        cfg.mute = IsDlgButtonChecked(IDC_CHECK_MUTE) == BST_CHECKED;

        config_validate(&cfg);

        dsp_preset_impl preset;
        make_preset(cfg, preset);
        m_callback.on_preset_changed(preset);

        m_updating = false;
    }

    void RefreshGainLabel(float db) {
        pfc::string_formatter msg;
        msg << pfc::format_float(db, 0, 2) << " dB";
        ::uSetDlgItemText(*this, IDC_STATIC_GAIN_DB, msg);
    }

    const dsp_preset & m_initData;
    dsp_preset_edit_callback & m_callback;
    fb2k::CDarkModeHooks m_dark;
    bool m_updating = false;
};

static void RunDSPConfigPopup(const dsp_preset &p_data, HWND p_parent,
                               dsp_preset_edit_callback &p_callback) {
    CDSPTrellisPopup popup(p_data, p_callback);
    if (popup.DoModal(p_parent) != IDOK) {
        p_callback.on_preset_changed(p_data);
    }
}

/* ─── DSP implementation ─── */

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
                                    dsp_preset_edit_callback &p_callback) {
        ::RunDSPConfigPopup(p_data, p_parent, p_callback);
    }
#endif

    bool on_chunk(audio_chunk *chunk, abort_callback & /*abort*/) override {
        if (!m_state)
            return true;

        const unsigned channels = chunk->get_channel_count();
        const unsigned pcm_rate = chunk->get_sample_rate();
        const t_size pcm_frames = chunk->get_sample_count();

        if (channels == 0 || pcm_frames == 0)
            return true;

        m_channels = (int)channels;
        m_pcm_rate = pcm_rate;

        /* Convert audio_sample (double on x64) to float for C engine */
        const audio_sample *src = chunk->get_data();
        size_t total_in = pcm_frames * channels;
        pfc::array_staticsize_t<float> in_f32;
        in_f32.set_size_discard(total_in);
        for (size_t i = 0; i < total_in; i++)
            in_f32[i] = (float)src[i];

        /* Allocate output buffer: worst case is 8x rate conversion */
        size_t max_out_frames = pcm_frames * 8;
        pfc::array_staticsize_t<float> out_buf;
        out_buf.set_size_discard(max_out_frames * channels);

        size_t out_frames = plugin_process(
            m_state, in_f32.get_ptr(), out_buf.get_ptr(),
            pcm_frames, (int)channels, pcm_rate);

        if (out_frames == 0)
            return true;  /* Not DoP or processing failed — pass through */

        /* Determine output PCM sample rate */
        uint32_t dsd_rate_in = pcm_rate * 16;
        uint32_t dsd_rate_out = m_config.fs_out ? m_config.fs_out : dsd_rate_in;
        uint32_t out_pcm_rate = dsd_rate_out / 16;

        /* Convert float output back to audio_sample */
        size_t total_out = out_frames * channels;
        pfc::array_staticsize_t<audio_sample> out_as;
        out_as.set_size_discard(total_out);
        for (size_t i = 0; i < total_out; i++)
            out_as[i] = (audio_sample)out_buf[i];

        if (m_config.output_format == OUTPUT_PCM) {
            /* PCM output mode: decimate DoP to PCM for visualization.
             * Simple average of 16 DSD bits → 1 PCM sample.
             * Output rate = out_pcm_rate (same as DoP rate). */
            chunk->set_data(out_as.get_ptr(), out_frames, channels, out_pcm_rate);
        } else {
            /* DoP output (default) */
            chunk->set_data(out_as.get_ptr(), out_frames, channels, out_pcm_rate);
        }

        return true;
    }

    void on_endofplayback(abort_callback & /*abort*/) override {
        if (!m_state || m_channels == 0)
            return;

        size_t max_drain_frames = 2048 / 16;
        pfc::array_staticsize_t<float> drain_buf;
        drain_buf.set_size_discard(max_drain_frames * (unsigned)m_channels);

        size_t drain_frames = plugin_drain(m_state, drain_buf.get_ptr(),
                                           m_channels);

        if (drain_frames > 0 && m_pcm_rate > 0) {
            uint32_t dsd_rate_in = m_pcm_rate * 16;
            uint32_t dsd_rate_out = m_config.fs_out ? m_config.fs_out
                                                     : dsd_rate_in;
            uint32_t out_pcm_rate = dsd_rate_out / 16;

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

    void on_endoftrack(abort_callback & /*abort*/) override {}

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
    "0.2.0",
    "DSD Trellis (Viterbi) Sigma-Delta Modulator\n"
    "Rate conversion, volume control, and noise shaping for DSD streams.\n"
    "Supports DoP and native DSD input from foo_input_sacd / foo_input_udsd.\n\n"
    "NTF coefficients ported from mansr/sox sdm.c (LGPL v2.1+)"
);

VALIDATE_COMPONENT_FILENAME("foo_dsd_trellis.dll");
