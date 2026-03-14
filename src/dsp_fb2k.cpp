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
#include <stdio.h>
#include <stdarg.h>

extern "C" {
#include "../include/dsd_types.h"
#include "../include/simd_detect.h"
#include "../include/fir.h"
#include "../include/cpuset.h"
#include "../include/httpapi.h"
#include "build_version.h"

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
const cpu_topology_t *plugin_get_topology(const plugin_state_t *s);
void            plugin_get_workload(const plugin_state_t *s,
                                     int *num_threads, int *segments_per_ch,
                                     bool *changed);
void            plugin_get_phase_timing(const plugin_state_t *s,
                                         double *unpack_ms, double *fir_ms,
                                         double *sdm_ms, double *pack_ms);
bool            plugin_get_cpuset_change(const plugin_state_t *s, uint64_t *mask);

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

/* ─── File logger ─── */

static CRITICAL_SECTION g_log_cs;
static bool g_log_cs_init = false;
static FILE *g_log_file = NULL;
static bool g_log_enabled = false;

static void log_init_cs() {
    if (!g_log_cs_init) {
        InitializeCriticalSection(&g_log_cs);
        g_log_cs_init = true;
    }
}

/* Build log path next to the DLL */
static pfc::string8 get_log_path() {
    pfc::string8 path;
    path = core_api::get_my_full_path();
    /* Strip filename, keep directory */
    t_size slash = path.find_last('/');
    t_size bslash = path.find_last('\\');
    t_size sep = 0;
    if (slash != pfc::infinite_size && slash > sep) sep = slash;
    if (bslash != pfc::infinite_size && bslash > sep) sep = bslash;
    if (sep > 0)
        path.truncate(sep + 1);
    path += "foo_dsd_trellis.log";
    return path;
}

static void log_open() {
    if (g_log_file) return;
    pfc::string8 path = get_log_path();
    g_log_file = fopen(path.c_str(), "a");
    if (g_log_file) {
        SYSTEMTIME st;
        GetLocalTime(&st);
        fprintf(g_log_file, "\n--- DSD Trellis log opened %04d-%02d-%02d %02d:%02d:%02d ---\n",
                st.wYear, st.wMonth, st.wDay,
                st.wHour, st.wMinute, st.wSecond);
        fflush(g_log_file);
    }
}

static void log_close() {
    if (g_log_file) {
        fprintf(g_log_file, "--- log closed ---\n");
        fclose(g_log_file);
        g_log_file = NULL;
    }
}

static void trellis_log(const char *fmt, ...) {
    log_init_cs();
    EnterCriticalSection(&g_log_cs);

    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    /* Always write to fb2k console */
    pfc::string_formatter msg;
    msg << "DSD Trellis: " << buf;
    console::print(msg);

    /* Write to file if enabled */
    if (g_log_enabled && g_log_file) {
        SYSTEMTIME st;
        GetLocalTime(&st);
        fprintf(g_log_file, "[%02d:%02d:%02d.%03d] %s\n",
                st.wHour, st.wMinute, st.wSecond, st.wMilliseconds, buf);
        fflush(g_log_file);
    }

    LeaveCriticalSection(&g_log_cs);
}

static void log_set_enabled(bool enabled) {
    log_init_cs();
    EnterCriticalSection(&g_log_cs);
    g_log_enabled = enabled;
    if (enabled)
        log_open();
    else
        log_close();
    LeaveCriticalSection(&g_log_cs);
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
        COMMAND_HANDLER_EX(IDC_COMBO_SDM_MODE, CBN_SELCHANGE, OnSdmModeChange)
        COMMAND_HANDLER_EX(IDC_COMBO_OUTPUT_RATE, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_COMBO_TRELLIS_DEPTH, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_COMBO_NTF, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_COMBO_FORMAT, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_COMBO_OUTPUT_FORMAT, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_CHECK_MUTE, BN_CLICKED, OnChange)
        COMMAND_HANDLER_EX(IDC_CHECK_DEBUG_LOG, BN_CLICKED, OnChange)
        COMMAND_HANDLER_EX(IDC_EDIT_GAIN, EN_CHANGE, OnEditChange)
        COMMAND_HANDLER_EX(IDC_EDIT_TRELLIS_CANDS, EN_CHANGE, OnEditChange)
        COMMAND_HANDLER_EX(IDC_EDIT_TRELLIS_LAT, EN_CHANGE, OnEditChange)
        COMMAND_HANDLER_EX(IDC_EDIT_THREADS, EN_CHANGE, OnEditChange)
    END_MSG_MAP()

private:
    BOOL OnInitDialog(CWindow, LPARAM) {
        m_dark.AddDialogWithControls(m_hWnd);

        dsd_config_t cfg = parse_preset(m_initData);

        /* SDM Mode combo */
        CComboBox sdm(GetDlgItem(IDC_COMBO_SDM_MODE));
        sdm.AddString(L"PreCorr (low CPU)");
        sdm.AddString(L"Trellis (high quality)");
        sdm.SetCurSel(cfg.sdm_mode);

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

        /* Debug log checkbox */
        CheckDlgButton(IDC_CHECK_DEBUG_LOG, cfg.debug_log ? BST_CHECKED : BST_UNCHECKED);

        /* Show engine info */
        const cpu_features_t *cpu = cpu_detect();
        pfc::string_formatter info;
        info << "FIR: " << fir_ipp_kernel_name()
             << " | CPU: " << (cpu->vendor == CPU_VENDOR_INTEL ? "Intel" :
                               cpu->vendor == CPU_VENDOR_AMD   ? "AMD" : "Unknown");
        ::uSetDlgItemText(*this, IDC_STATIC_VU_L, info);

        /* Enable/disable trellis controls based on SDM mode */
        EnableTrellisControls(cfg.sdm_mode == SDM_MODE_TRELLIS);

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

    void OnSdmModeChange(UINT, int, CWindow) {
        int mode = CComboBox(GetDlgItem(IDC_COMBO_SDM_MODE)).GetCurSel();
        EnableTrellisControls(mode == SDM_MODE_TRELLIS);
        if (!m_updating) UpdatePreset();
    }

    void EnableTrellisControls(bool enable) {
        BOOL en = enable ? TRUE : FALSE;
        ::EnableWindow(GetDlgItem(IDC_COMBO_TRELLIS_DEPTH), en);
        ::EnableWindow(GetDlgItem(IDC_EDIT_TRELLIS_CANDS), en);
        ::EnableWindow(GetDlgItem(IDC_SPIN_TRELLIS_CANDS), en);
        ::EnableWindow(GetDlgItem(IDC_EDIT_TRELLIS_LAT), en);
        ::EnableWindow(GetDlgItem(IDC_SPIN_TRELLIS_LAT), en);
    }

    void UpdatePreset() {
        m_updating = true;

        dsd_config_t cfg;
        dsd_config_defaults(&cfg);

        /* Read SDM mode */
        cfg.sdm_mode = CComboBox(GetDlgItem(IDC_COMBO_SDM_MODE)).GetCurSel();

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

        /* Debug log */
        cfg.debug_log = IsDlgButtonChecked(IDC_CHECK_DEBUG_LOG) == BST_CHECKED;

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
        , m_httpapi(nullptr)
    {
        if (m_state)
            plugin_set_config(m_state, &m_config);

        log_set_enabled(m_config.debug_log);

        /* Log version, build hash, and build time */
        trellis_log("DSD Trellis v%s (build %s, %s %s)",
                    BUILD_VERSION, BUILD_GIT_HASH, BUILD_DATE, BUILD_TIME);

        uint32_t fs = m_config.fs_out;
        const char *sdm_mode_str = m_config.sdm_mode == SDM_MODE_TRELLIS
                                   ? "Trellis" : "PreCorr";
        trellis_log("initialized (sdm=%s, fir=%s, output=%s, depth=%d, cands=%d)",
                    sdm_mode_str, fir_ipp_kernel_name(),
                    fs == 0 ? "as-input" :
                    fs == DSD_RATE_64  ? "DSD64" :
                    fs == DSD_RATE_128 ? "DSD128" :
                    fs == DSD_RATE_256 ? "DSD256" :
                    fs == DSD_RATE_512 ? "DSD512" : "?",
                    m_config.trellis_depth, m_config.trellis_cands);

        /* Start REST API server */
        if (m_config.api_port > 0) {
            m_httpapi = httpapi_create(m_config.api_port, &m_config);
            if (m_httpapi)
                trellis_log("REST API listening on 127.0.0.1:%u", (unsigned)m_config.api_port);
            else
                trellis_log("REST API failed to start on port %u", (unsigned)m_config.api_port);
        }
    }

    ~dsp_dsd_trellis() {
        trellis_log("shutting down");
        httpapi_destroy(m_httpapi);
        log_set_enabled(false);
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

        /* Check for pending config from REST API */
        {
            dsd_config_t pending;
            if (httpapi_get_pending_config(m_httpapi, &pending)) {
                /* Preserve runtime-detected fs_in — it's not user-settable */
                const dsd_config_t *current = plugin_get_config(m_state);
                if (current)
                    pending.fs_in = current->fs_in;
                m_config = pending;
                plugin_set_config(m_state, &m_config);
                plugin_reconfigure(m_state, &m_config);
                httpapi_update_config(m_httpapi, &m_config);
                log_set_enabled(m_config.debug_log);
                trellis_log("config updated via REST API (cands=%d, depth=%d, gain=%.3f)",
                            m_config.trellis_cands, m_config.trellis_depth, (double)m_config.gain);
            }
        }

        const unsigned channels = chunk->get_channel_count();
        const unsigned pcm_rate = chunk->get_sample_rate();
        const t_size pcm_frames = chunk->get_sample_count();

        if (channels == 0 || pcm_frames == 0)
            return true;

        /* Log first chunk detection */
        if (m_pcm_rate != pcm_rate || m_channels != (int)channels) {
            uint32_t dsd_rate = pcm_rate * 16;
            unsigned dsd_mult = dsd_rate / 44100;
            trellis_log("detected DSD%u (%u Hz) via DoP @ %u Hz, %uch",
                        dsd_mult, dsd_rate, pcm_rate, channels);
        }

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

        LARGE_INTEGER t0, t1, freq;
        QueryPerformanceCounter(&t0);

        size_t out_frames = plugin_process(
            m_state, in_f32.get_ptr(), out_buf.get_ptr(),
            pcm_frames, (int)channels, pcm_rate);

        QueryPerformanceCounter(&t1);
        QueryPerformanceFrequency(&freq);
        double process_ms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;

        if (out_frames == 0) {
            m_chunk_count++;
            if (!m_logged_passthrough) {
                trellis_log("no DoP detected, passing through (chunk #%u, %.1fms)",
                            m_chunk_count, process_ms);
                m_logged_passthrough = true;
            }
            /* Log first 5 passthrough chunks for diagnostics */
            if (m_chunk_count <= 5) {
                trellis_log("chunk #%u: out_frames=0, process_ms=%.1f, pcm_frames=%zu",
                            m_chunk_count, process_ms, pcm_frames);
            }
            return true;  /* Not DoP or processing failed — pass through */
        }
        m_logged_passthrough = false;

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

        {
            unsigned out_mult = dsd_rate_out / 44100;
            unsigned in_mult  = dsd_rate_in  / 44100;
            double chunk_ms = (double)pcm_frames / (double)pcm_rate * 1000.0;
            double ratio = chunk_ms > 0 ? process_ms / chunk_ms : 0;

            if (!m_logged_processing) {
                if (dsd_rate_in != dsd_rate_out)
                    trellis_log("processing DSD%u -> DSD%u (%zu -> %zu frames, %.1fms/%.1fms = %.1fx RT)",
                                in_mult, out_mult, pcm_frames, out_frames,
                                process_ms, chunk_ms, ratio);
                else
                    trellis_log("processing DSD%u (%zu -> %zu frames, %.1fms/%.1fms = %.1fx RT)",
                                in_mult, pcm_frames, out_frames,
                                process_ms, chunk_ms, ratio);
                m_logged_processing = true;

                /* Log topology after first successful processing */
                const cpu_topology_t *topo = plugin_get_topology(m_state);
                if (topo) {
                    char topo_buf[512];
                    cpuset_summary(topo, topo_buf, sizeof(topo_buf));
                    trellis_log("CPU: %s", topo_buf);

                    /* Detailed per-core info when debug log is enabled */
                    if (g_log_enabled) {
                        cpuset_log_detail(topo, [](const char *line, void *) {
                            trellis_log("  topo: %s", line);
                        }, nullptr);
                    }
                }

                /* Log initial workload and phase timing */
                int wl_threads = 0, wl_segments = 0;
                bool wl_changed = false;
                plugin_get_workload(m_state, &wl_threads, &wl_segments, &wl_changed);
                trellis_log("workload: %d threads, %d segments/ch",
                            wl_threads, wl_segments);

                double t_unpack, t_fir, t_sdm, t_pack;
                plugin_get_phase_timing(m_state, &t_unpack, &t_fir, &t_sdm, &t_pack);
                trellis_log("phase timing: unpack=%.1fms fir=%.1fms sdm=%.1fms pack=%.1fms",
                            t_unpack, t_fir, t_sdm, t_pack);
            }

            /* Log CPUSET changes (CPUDoc dynamic core management) */
            {
                uint64_t cpuset_mask = 0;
                if (plugin_get_cpuset_change(m_state, &cpuset_mask)) {
                    int enabled = 0;
                    for (int b = 0; b < 64; b++)
                        if (cpuset_mask & ((uint64_t)1 << b)) enabled++;
                    trellis_log("cpuset changed: 0x%016llX (%d cores enabled)",
                                (unsigned long long)cpuset_mask, enabled);
                }
            }

            /* Log workload changes (thread count / segment count) */
            {
                int wl_threads = 0, wl_segments = 0;
                bool wl_changed = false;
                plugin_get_workload(m_state, &wl_threads, &wl_segments, &wl_changed);
                if (wl_changed && m_logged_processing) {
                    trellis_log("workload changed: %d threads, %d segments/ch",
                                wl_threads, wl_segments);
                }
            }

            /* Log periodic timing every 100 chunks when debug log is enabled */
            m_chunk_count++;
            if (g_log_enabled && (m_chunk_count <= 5 || (m_chunk_count % 100) == 0)) {
                double t_unpack, t_fir, t_sdm, t_pack;
                plugin_get_phase_timing(m_state, &t_unpack, &t_fir, &t_sdm, &t_pack);
                trellis_log("chunk #%u: %.1fms total (unpack=%.1f fir=%.1f sdm=%.1f pack=%.1f) / %.1fms audio = %.2fx RT",
                            m_chunk_count, process_ms,
                            t_unpack, t_fir, t_sdm, t_pack,
                            chunk_ms, ratio);
            }

            /* Push status to REST API */
            if (m_httpapi) {
                httpapi_status_t st;
                memset(&st, 0, sizeof(st));
                st.playing = true;
                st.dsd_rate_in = dsd_rate_in;
                st.dsd_rate_out = dsd_rate_out;
                st.channels = (int)channels;
                int wl_t = 0, wl_s = 0;
                bool wl_c = false;
                plugin_get_workload(m_state, &wl_t, &wl_s, &wl_c);
                st.threads = wl_t;
                st.segments_per_ch = wl_s;
                uint64_t cm = 0;
                plugin_get_cpuset_change(m_state, &cm);
                st.cpuset_mask = cm;
                for (int b = 0; b < 64; b++)
                    if (cm & ((uint64_t)1 << b)) st.cpuset_enabled++;
                st.chunk_count = m_chunk_count;
                st.last_chunk_ms = process_ms;
                st.last_audio_ms = chunk_ms;
                st.rt_ratio = ratio;
                double tu, tf, ts, tp;
                plugin_get_phase_timing(m_state, &tu, &tf, &ts, &tp);
                st.unpack_ms = tu;
                st.fir_ms = tf;
                st.sdm_ms = ts;
                st.pack_ms = tp;
                httpapi_update_status(m_httpapi, &st);
            }
        }

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
        m_logged_passthrough = false;
        m_logged_processing = false;
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
    httpapi_t       *m_httpapi;
    int              m_channels;
    unsigned         m_pcm_rate;
    bool             m_logged_passthrough = false;
    bool             m_logged_processing = false;
    unsigned         m_chunk_count = 0;
};

static dsp_factory_t<dsp_dsd_trellis> g_dsp_factory;

DECLARE_COMPONENT_VERSION(
    "DSD Trellis SDM",
    BUILD_VERSION,
    "DSD Trellis (Viterbi) Sigma-Delta Modulator\n"
    "Rate conversion, volume control, and noise shaping for DSD streams.\n"
    "Supports DoP and native DSD input from foo_input_sacd / foo_input_udsd.\n\n"
    "NTF coefficients ported from mansr/sox sdm.c (LGPL v2.1+)"
);

VALIDATE_COMPONENT_FILENAME("foo_dsd_trellis.dll");
