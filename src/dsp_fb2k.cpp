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
#include <commctrl.h>

extern "C" {
#include "../include/dsd_types.h"
#include "../include/simd_detect.h"
#include "../include/fir.h"
#include "../include/cpuset.h"
#include "../include/httpapi.h"
#include "../include/dop.h"
#include "../include/engine.h"
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
size_t          plugin_process_pcm(plugin_state_t *s,
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

/* ─── Rate map display helpers ─── */

static const wchar_t *g_rate_names[RATE_MAP_COUNT] = {
    L"44100", L"48000", L"88200", L"96000",
    L"176400", L"192000", L"352800", L"384000",
    L"DSD64", L"DSD128", L"DSD256", L"DSD512"
};

static const wchar_t *g_output_names[RATE_OUT_COUNT] = {
    L"-", L"DSD64", L"DSD128", L"DSD256", L"DSD512",
    L"PCM 44.1k", L"PCM 88.2k", L"PCM 176.4k", L"PCM 352.8k"
};

static const wchar_t *g_ntf_names[] = {
    L"Auto", L"CLANS-4", L"SDM-4", L"CLANS-5", L"SDM-5",
    L"CLANS-6", L"SDM-6", L"CLANS-7", L"SDM-7", L"CLANS-8", L"SDM-8"
};
#define NTF_NAME_COUNT (sizeof(g_ntf_names) / sizeof(g_ntf_names[0]))

/* ─── Gain helpers ─── */

static inline float gain_to_db(float linear) {
    if (linear <= 0.0f) return -120.0f;
    return 20.0f * log10f(linear);
}

static inline float db_to_gain(float db) {
    if (db <= -120.0f) return 0.0f;
    return powf(10.0f, db / 20.0f);
}

/* ─── Volume tracking (main thread → DSP thread) ─── */
/* playback_control is main-thread-only; we cache volume via play_callback */

static volatile LONG g_volume_gain_x1000 = 1000;  /* linear gain * 1000 */
static volatile LONG g_volume_muted = 0;

static float get_cached_gain() {
    return (float)InterlockedCompareExchange(&g_volume_gain_x1000, 0, 0) / 1000.0f;
}

static bool get_cached_muted() {
    return InterlockedCompareExchange(&g_volume_muted, 0, 0) != 0;
}

class volume_callback_impl : public play_callback_static {
public:
    unsigned get_flags() override {
        return flag_on_volume_change;
    }

    void on_playback_starting(play_control::t_track_command, bool) override {}
    void on_playback_new_track(metadb_handle_ptr) override {}
    void on_playback_stop(play_control::t_stop_reason) override {}
    void on_playback_seek(double) override {}
    void on_playback_pause(bool) override {}
    void on_playback_edited(metadb_handle_ptr) override {}
    void on_playback_dynamic_info(const file_info &) override {}
    void on_playback_dynamic_info_track(const file_info &) override {}
    void on_playback_time(double) override {}

    void on_volume_change(float p_new_val) override {
        bool muted = (p_new_val <= playback_control::volume_mute);
        float gain = muted ? 0.0f : db_to_gain(p_new_val);
        InterlockedExchange(&g_volume_gain_x1000, (LONG)(gain * 1000.0f));
        InterlockedExchange(&g_volume_muted, muted ? 1 : 0);
    }
};

static play_callback_static_factory_t<volume_callback_impl> g_volume_callback;

/* Initialize cached volume from current setting (called from main thread at startup) */
class volume_init : public initquit {
public:
    void on_init() override {
        auto pc = playback_control::get();
        float vol_db = pc->get_volume();
        bool muted = (vol_db <= playback_control::volume_mute);
        float gain = muted ? 0.0f : db_to_gain(vol_db);
        InterlockedExchange(&g_volume_gain_x1000, (LONG)(gain * 1000.0f));
        InterlockedExchange(&g_volume_muted, muted ? 1 : 0);
    }
    void on_quit() override {}
};

static initquit_factory_t<volume_init> g_volume_init;

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

/* Use foo_dsd_processor's GUID so foo_input_sacd's autoproxy_dsp
 * whitelist allows us to modify DoP data.
 * {C5F65473-79C6-466E-9F6B-AFA46F87249E} */
static const GUID g_dsp_guid =
    { 0xc5f65473, 0x79c6, 0x466e,
      { 0x9f, 0x6b, 0xaf, 0xa4, 0x6f, 0x87, 0x24, 0x9e } };

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
        : m_initData(initData), m_callback(callback), m_editRow(-1), m_editCol(-1) {}

    enum { IDD = IDD_DSP_TRELLIS };

    BEGIN_MSG_MAP_EX(CDSPTrellisPopup)
        MSG_WM_INITDIALOG(OnInitDialog)
        COMMAND_HANDLER_EX(IDOK, BN_CLICKED, OnButton)
        COMMAND_HANDLER_EX(IDCANCEL, BN_CLICKED, OnButton)
        COMMAND_HANDLER_EX(IDC_COMBO_SDM_MODE, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_COMBO_FORMAT, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_CHECK_DEBUG_LOG, BN_CLICKED, OnChange)
        COMMAND_HANDLER_EX(IDC_EDIT_THREADS, EN_CHANGE, OnEditChange)
        COMMAND_HANDLER_EX(IDC_COMBO_RATEMAP_EDIT, CBN_SELCHANGE, OnRateMapEditChange)
        COMMAND_HANDLER_EX(IDC_COMBO_RATEMAP_EDIT, CBN_KILLFOCUS, OnComboKillFocus)
        COMMAND_HANDLER_EX(IDC_COMBO_RATEMAP_EDIT, CBN_CLOSEUP, OnComboCloseUp)
        COMMAND_HANDLER_EX(IDC_COMBO_RATEMAP_NTF_EDIT, CBN_SELCHANGE, OnNtfEditChange)
        COMMAND_HANDLER_EX(IDC_COMBO_RATEMAP_NTF_EDIT, CBN_KILLFOCUS, OnComboKillFocus)
        COMMAND_HANDLER_EX(IDC_COMBO_RATEMAP_NTF_EDIT, CBN_CLOSEUP, OnComboCloseUp)
        NOTIFY_HANDLER_EX(IDC_LIST_RATEMAP, NM_CLICK, OnRateMapClick)
    END_MSG_MAP()

private:
    BOOL OnInitDialog(CWindow, LPARAM) {
        m_dark.AddDialogWithControls(m_hWnd);

        m_cfg = parse_preset(m_initData);

        /* Rate map ListView */
        m_listRate = GetDlgItem(IDC_LIST_RATEMAP);
        m_listRate.SetExtendedListViewStyle(
            LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES);
        m_listRate.InsertColumn(0, L"Input", LVCFMT_LEFT, 65);
        m_listRate.InsertColumn(1, L"Output", LVCFMT_LEFT, 85);
        m_listRate.InsertColumn(2, L"NTF", LVCFMT_LEFT, 65);

        for (int i = 0; i < RATE_MAP_COUNT; i++) {
            m_listRate.InsertItem(i, g_rate_names[i]);
            m_listRate.SetItemText(i, 1, g_output_names[m_cfg.rate_map[i]]);
            int ntf_idx = m_cfg.rate_ntf[i] + 1; /* NTF_AUTO=-1 → 0 */
            if (ntf_idx < 0 || ntf_idx >= (int)NTF_NAME_COUNT) ntf_idx = 0;
            m_listRate.SetItemText(i, 2, g_ntf_names[ntf_idx]);
        }

        /* Create in-place editor combos (children of dialog, initially hidden) */
        CRect rc(0, 0, 85, 200);
        m_editCombo.Create(m_hWnd, rc, NULL,
            WS_CHILD | CBS_DROPDOWNLIST | WS_VSCROLL,
            0, IDC_COMBO_RATEMAP_EDIT);
        m_editCombo.SetFont(GetFont());

        m_ntfCombo.Create(m_hWnd, rc, NULL,
            WS_CHILD | CBS_DROPDOWNLIST | WS_VSCROLL,
            0, IDC_COMBO_RATEMAP_NTF_EDIT);
        m_ntfCombo.SetFont(GetFont());

        /* SDM Mode combo */
        CComboBox sdm(GetDlgItem(IDC_COMBO_SDM_MODE));
        sdm.AddString(L"PreCorr (low CPU)");
        sdm.AddString(L"Trellis (high quality)");
        sdm.SetCurSel(m_cfg.sdm_mode);

        /* Threads */
        SetDlgItemInt(IDC_EDIT_THREADS, m_cfg.thread_count, FALSE);
        CUpDownCtrl(GetDlgItem(IDC_SPIN_THREADS)).SetRange(0, 32);

        /* Input format combo */
        CComboBox fmt(GetDlgItem(IDC_COMBO_FORMAT));
        fmt.AddString(L"Auto");
        fmt.AddString(L"DoP");
        fmt.AddString(L"Native");
        fmt.SetCurSel(m_cfg.format);

        /* Debug log checkbox */
        CheckDlgButton(IDC_CHECK_DEBUG_LOG, m_cfg.debug_log ? BST_CHECKED : BST_UNCHECKED);

        /* Show initial engine info */
        const cpu_features_t *cpu = cpu_detect();
        pfc::string_formatter info;
        info << "FIR: " << fir_ipp_kernel_name()
             << " | CPU: " << (cpu->vendor == CPU_VENDOR_INTEL ? "Intel" :
                               cpu->vendor == CPU_VENDOR_AMD   ? "AMD" : "Unknown")
             << "\nSelect a rate mapping to see path details.";
        ::uSetDlgItemText(*this, IDC_STATIC_PATH_INFO, info);

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

    /* ─── Rate map in-place editing ─── */

    LRESULT OnRateMapClick(LPNMHDR pnmh) {
        NMITEMACTIVATE *nm = (NMITEMACTIVATE *)pnmh;

        /* Hide both combos first */
        m_editCombo.ShowWindow(SW_HIDE);
        m_ntfCombo.ShowWindow(SW_HIDE);

        if (nm->iItem < 0 || nm->iItem >= RATE_MAP_COUNT)
            return 0;

        /* Always update path info on row click */
        UpdatePathInfo(nm->iItem);

        if (nm->iSubItem == 1) {
            /* Output column — show output combo */
            ShowOutputCombo(nm->iItem);
        } else if (nm->iSubItem == 2) {
            /* NTF column — show NTF combo */
            ShowNtfCombo(nm->iItem);
        }

        return 0;
    }

    void ShowOutputCombo(int row) {
        CRect cellRC;
        m_listRate.GetSubItemRect(row, 1, LVIR_BOUNDS, &cellRC);
        m_listRate.MapWindowPoints(m_hWnd, &cellRC);

        m_editCombo.ResetContent();
        for (int j = 0; j < RATE_OUT_COUNT; j++) {
            if (rate_map_valid_output(row, (uint8_t)j)) {
                int idx = m_editCombo.AddString(g_output_names[j]);
                m_editCombo.SetItemData(idx, (DWORD_PTR)j);
            }
        }

        uint8_t cur = m_cfg.rate_map[row];
        for (int j = 0; j < m_editCombo.GetCount(); j++) {
            if ((int)m_editCombo.GetItemData(j) == cur) {
                m_editCombo.SetCurSel(j);
                break;
            }
        }

        m_editRow = row;
        m_editCol = 1;
        m_editCombo.SetWindowPos(HWND_TOP,
                                 cellRC.left, cellRC.top,
                                 cellRC.Width(), 200,
                                 SWP_SHOWWINDOW);
        m_editCombo.SetFocus();
        m_editCombo.ShowDropDown(TRUE);
    }

    void ShowNtfCombo(int row) {
        CRect cellRC;
        m_listRate.GetSubItemRect(row, 2, LVIR_BOUNDS, &cellRC);
        m_listRate.MapWindowPoints(m_hWnd, &cellRC);

        m_ntfCombo.ResetContent();
        for (int j = 0; j < (int)NTF_NAME_COUNT; j++) {
            int idx = m_ntfCombo.AddString(g_ntf_names[j]);
            m_ntfCombo.SetItemData(idx, (DWORD_PTR)(j - 1)); /* 0→-1(Auto), 1→0(CLANS-4), etc */
        }

        int8_t cur = m_cfg.rate_ntf[row];
        m_ntfCombo.SetCurSel(cur + 1); /* NTF_AUTO=-1 → index 0 */

        m_editRow = row;
        m_editCol = 2;
        m_ntfCombo.SetWindowPos(HWND_TOP,
                                cellRC.left, cellRC.top,
                                cellRC.Width(), 200,
                                SWP_SHOWWINDOW);
        m_ntfCombo.SetFocus();
        m_ntfCombo.ShowDropDown(TRUE);
    }

    void OnRateMapEditChange(UINT, int, CWindow) {
        int sel = m_editCombo.GetCurSel();
        if (sel >= 0 && m_editRow >= 0 && m_editRow < RATE_MAP_COUNT) {
            uint8_t out_idx = (uint8_t)m_editCombo.GetItemData(sel);
            m_cfg.rate_map[m_editRow] = out_idx;
            m_listRate.SetItemText(m_editRow, 1, g_output_names[out_idx]);
            UpdatePathInfo(m_editRow);
            if (!m_updating) UpdatePreset();
        }
    }

    void OnNtfEditChange(UINT, int, CWindow) {
        int sel = m_ntfCombo.GetCurSel();
        if (sel >= 0 && m_editRow >= 0 && m_editRow < RATE_MAP_COUNT) {
            int8_t ntf_val = (int8_t)(sel - 1); /* index 0 → NTF_AUTO (-1) */
            m_cfg.rate_ntf[m_editRow] = ntf_val;
            m_listRate.SetItemText(m_editRow, 2, g_ntf_names[sel]);
            UpdatePathInfo(m_editRow);
            if (!m_updating) UpdatePreset();
        }
    }

    void OnComboKillFocus(UINT, int, CWindow) {
        m_editCombo.ShowWindow(SW_HIDE);
        m_ntfCombo.ShowWindow(SW_HIDE);
    }

    void OnComboCloseUp(UINT, int, CWindow) {
        /* Hide combo after dropdown closes */
        m_editCombo.ShowWindow(SW_HIDE);
        m_ntfCombo.ShowWindow(SW_HIDE);
    }

    /* ─── Contextual path info panel ─── */

    void UpdatePathInfo(int row) {
        if (row < 0 || row >= RATE_MAP_COUNT) return;

        uint8_t out_idx = m_cfg.rate_map[row];
        pfc::string_formatter info;

        if (out_idx == RATE_OUT_BYPASS) {
            info << g_rate_names[row] << ": Passthrough (no processing)";
            ::uSetDlgItemText(*this, IDC_STATIC_PATH_INFO, info);
            return;
        }

        /* Determine input/output rates */
        uint32_t fs_in = 0;
        bool is_dsd_input = (row >= RATE_MAP_PCM_COUNT);
        switch (row) {
        case RATEIDX_44100:  fs_in = 44100;       break;
        case RATEIDX_48000:  fs_in = 48000;       break;
        case RATEIDX_88200:  fs_in = 88200;       break;
        case RATEIDX_96000:  fs_in = 96000;       break;
        case RATEIDX_176400: fs_in = 176400;      break;
        case RATEIDX_192000: fs_in = 192000;      break;
        case RATEIDX_352800: fs_in = 352800;      break;
        case RATEIDX_384000: fs_in = 384000;      break;
        case RATEIDX_DSD64:  fs_in = DSD_RATE_64; break;
        case RATEIDX_DSD128: fs_in = DSD_RATE_128; break;
        case RATEIDX_DSD256: fs_in = DSD_RATE_256; break;
        case RATEIDX_DSD512: fs_in = DSD_RATE_512; break;
        }

        uint32_t fs_out = rate_out_to_hz(out_idx);
        if (fs_out == 0) {
            info << "Invalid output configuration";
            ::uSetDlgItemText(*this, IDC_STATIC_PATH_INFO, info);
            return;
        }

        /* Query path info from engine */
        int8_t ntf_val = m_cfg.rate_ntf[row];
        int sdm_mode = m_cfg.sdm_mode;
        engine_path_info_t pi;
        engine_get_path_info(fs_in, fs_out, (int)ntf_val, sdm_mode, &m_cfg, &pi);

        /* Format info text */
        info << g_rate_names[row];
        if (is_dsd_input && rate_out_is_dsd(out_idx)) {
            /* DSD → DSD */
            if (fs_in == fs_out)
                info << " -> " << g_output_names[out_idx] << " (re-encode)";
            else if (fs_out > fs_in)
                info << " -> " << g_output_names[out_idx] << " (" << (fs_out/fs_in) << "x upsample)";
            else
                info << " -> " << g_output_names[out_idx] << " (1/" << (fs_in/fs_out) << " downsample)";
        } else if (is_dsd_input && rate_out_is_pcm(out_idx)) {
            /* DSD → PCM */
            info << " -> " << g_output_names[out_idx] << " (1/" << (fs_in/fs_out) << " decimation)";
        } else {
            /* PCM → DSD */
            info << " -> " << g_output_names[out_idx] << " (" << (fs_out/fs_in) << "x upsample)";
        }

        info << "\nFIR: " << pi.fir_stages << " stages, " << fir_ipp_kernel_name();

        if (pi.fir_only) {
            info << "\nFIR decimation only (no SDM)";
        } else {
            const char *sdm_name = (sdm_mode == SDM_MODE_TRELLIS) ? "Trellis" : "PreCorr";
            info << "\nSDM: " << sdm_name;
            if (sdm_mode == SDM_MODE_TRELLIS)
                info << ", depth=" << pi.depth << ", cands=" << pi.cands << ", lat=" << pi.lat;

            /* NTF display */
            int ntf_display = pi.ntf_filter;
            info << "\nNTF: ";
            if (ntf_val == NTF_AUTO) {
                info << "Auto";
                if (ntf_display >= 0 && ntf_display < (int)(NTF_NAME_COUNT - 1))
                    info << " (" << g_ntf_names[ntf_display + 1] << ")";
            } else {
                int idx = ntf_val + 1;
                if (idx >= 0 && idx < (int)NTF_NAME_COUNT)
                    info << g_ntf_names[idx];
            }

            if (pi.state_limit > 0.0)
                info << "\nState limiter: " << pfc::format_float(pi.state_limit, 0, 1);
        }

        ::uSetDlgItemText(*this, IDC_STATIC_PATH_INFO, info);
    }

    void UpdatePreset() {
        m_updating = true;

        /* Read SDM mode */
        m_cfg.sdm_mode = CComboBox(GetDlgItem(IDC_COMBO_SDM_MODE)).GetCurSel();

        /* Threads */
        m_cfg.thread_count = (int)GetDlgItemInt(IDC_EDIT_THREADS, NULL, FALSE);

        /* Input format */
        m_cfg.format = CComboBox(GetDlgItem(IDC_COMBO_FORMAT)).GetCurSel();

        /* Debug log */
        m_cfg.debug_log = IsDlgButtonChecked(IDC_CHECK_DEBUG_LOG) == BST_CHECKED;

        /* rate_map and rate_ntf are maintained via OnRateMapEditChange/OnNtfEditChange */

        config_validate(&m_cfg);

        dsp_preset_impl preset;
        make_preset(m_cfg, preset);
        m_callback.on_preset_changed(preset);

        m_updating = false;
    }

    const dsp_preset & m_initData;
    dsp_preset_edit_callback & m_callback;
    fb2k::CDarkModeHooks m_dark;
    bool m_updating = false;
    dsd_config_t m_cfg;
    CListViewCtrl m_listRate;
    CComboBox m_editCombo;
    CComboBox m_ntfCombo;
    int m_editRow;
    int m_editCol;
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

        const char *sdm_mode_str = m_config.sdm_mode == SDM_MODE_TRELLIS
                                   ? "Trellis" : "PreCorr";
        trellis_log("initialized (sdm=%s, fir=%s)",
                    sdm_mode_str, fir_ipp_kernel_name());

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
                const dsd_config_t *current = plugin_get_config(m_state);
                if (current)
                    pending.fs_in = current->fs_in;
                m_config = pending;
                plugin_set_config(m_state, &m_config);
                plugin_reconfigure(m_state, &m_config);
                httpapi_update_config(m_httpapi, &m_config);
                log_set_enabled(m_config.debug_log);
                trellis_log("config updated via REST API");
            }
        }

        const unsigned channels = chunk->get_channel_count();
        const unsigned pcm_rate = chunk->get_sample_rate();
        const t_size pcm_frames = chunk->get_sample_count();

        if (channels == 0 || pcm_frames == 0)
            return true;

        /* ─── Volume / mute ─── */
        m_config.mute = get_cached_muted();
        m_config.gain = get_cached_gain();

        /* ─── Rate map dispatch ─── */

        /* Check if input could be DoP */
        bool is_dop = false;
        uint32_t dsd_rate = 0;
        {
            uint32_t candidate_dsd = pcm_rate * 16u;
            switch (candidate_dsd) {
            case DSD_RATE_64: case DSD_RATE_128: case DSD_RATE_256: case DSD_RATE_512:
                dsd_rate = candidate_dsd;
                break;
            }
        }

        /* Convert audio_sample to float for detection */
        const audio_sample *src = chunk->get_data();
        size_t total_in = pcm_frames * channels;
        pfc::array_staticsize_t<float> in_f32;
        in_f32.set_size_discard(total_in);
        for (size_t i = 0; i < total_in; i++)
            in_f32[i] = (float)src[i];

        if (dsd_rate != 0 && pcm_frames >= 8 &&
            dop_detect_interleaved(in_f32.get_ptr(), pcm_frames, (int)channels)) {
            is_dop = true;
        }

        /* Look up rate in rate map */
        uint32_t lookup_rate = is_dop ? dsd_rate : pcm_rate;
        int map_idx = rate_map_index(lookup_rate);
        if (map_idx < 0 || m_config.rate_map[map_idx] == RATE_OUT_BYPASS)
            return true;  /* Bypass: pass through unchanged */

        uint8_t out_idx = m_config.rate_map[map_idx];
        bool out_is_pcm = rate_out_is_pcm(out_idx);
        uint32_t out_rate = rate_out_to_hz(out_idx);

        if (out_rate == 0)
            return true;

        /* Apply per-rate NTF override */
        m_config.ntf_filter = (int)m_config.rate_ntf[map_idx];

        /* Set output rate for this chunk */
        m_config.fs_out = out_rate;
        if (!m_logged_processing) {
            trellis_log("rate_map: lookup_rate=%u map_idx=%d out_idx=%u out_rate=%u is_dop=%d gain=%.3f fs_in=%u fs_out=%u",
                        lookup_rate, map_idx, (unsigned)out_idx, out_rate, is_dop, m_config.gain, m_config.fs_in, m_config.fs_out);
        }
        plugin_set_config(m_state, &m_config);

        m_channels = (int)channels;
        m_pcm_rate = pcm_rate;

        /* Allocate output buffer */
        size_t max_ratio = is_dop ? 8 : (out_rate > pcm_rate ? out_rate / pcm_rate : 1);
        if (max_ratio < 8) max_ratio = 8;
        size_t max_out_frames = pcm_frames * max_ratio;
        pfc::array_staticsize_t<float> out_buf;
        out_buf.set_size_discard(max_out_frames * channels);

        LARGE_INTEGER t0, t1, freq;
        QueryPerformanceCounter(&t0);

        size_t out_frames = 0;

        if (is_dop) {
            /* DSD input via DoP */
            if (!m_logged_processing) {
                unsigned dsd_mult = dsd_rate / 44100;
                if (out_is_pcm)
                    trellis_log("DSD%u -> PCM %uHz, %uch", dsd_mult, out_rate, channels);
                else if (dsd_rate == out_rate)
                    trellis_log("DSD%u re-encode, %uch", dsd_mult, channels);
                else
                    trellis_log("DSD%u -> DSD%u, %uch", dsd_mult, out_rate / 44100, channels);
            }
            out_frames = plugin_process(m_state, in_f32.get_ptr(), out_buf.get_ptr(),
                                        pcm_frames, (int)channels, pcm_rate);
        } else {
            /* PCM input — convert to DSD */
            if (!m_logged_processing) {
                unsigned out_mult = out_rate / 44100;
                trellis_log("PCM %uHz -> DSD%u, %uch", pcm_rate, out_mult, channels);
            }
            out_frames = plugin_process_pcm(m_state, in_f32.get_ptr(), out_buf.get_ptr(),
                                             pcm_frames, (int)channels, pcm_rate);
        }

        QueryPerformanceCounter(&t1);
        QueryPerformanceFrequency(&freq);
        double process_ms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;

        if (out_frames == 0) {
            m_chunk_count++;
            if (!m_logged_passthrough) {
                trellis_log("no output produced, passing through (chunk #%u, %.1fms)",
                            m_chunk_count, process_ms);
                m_logged_passthrough = true;
            }
            return true;
        }
        m_logged_passthrough = false;

        /* Determine output sample rate */
        uint32_t out_pcm_rate;
        if (out_is_pcm) {
            out_pcm_rate = out_rate;  /* Direct PCM output rate */
        } else {
            out_pcm_rate = out_rate / 16;  /* DoP PCM rate */
        }

        /* Convert float output back to audio_sample */
        size_t total_out = out_frames * channels;
        pfc::array_staticsize_t<audio_sample> out_as;
        out_as.set_size_discard(total_out);
        for (size_t i = 0; i < total_out; i++)
            out_as[i] = (audio_sample)out_buf[i];

        {
            double chunk_ms = (double)pcm_frames / (double)pcm_rate * 1000.0;
            double ratio = chunk_ms > 0 ? process_ms / chunk_ms : 0;

            if (!m_logged_processing) {
                m_logged_processing = true;

                /* Log topology after first successful processing */
                const cpu_topology_t *topo = plugin_get_topology(m_state);
                if (topo) {
                    char topo_buf[512];
                    cpuset_summary(topo, topo_buf, sizeof(topo_buf));
                    trellis_log("CPU: %s", topo_buf);

                    if (g_log_enabled) {
                        cpuset_log_detail(topo, [](const char *line, void *) {
                            trellis_log("  topo: %s", line);
                        }, nullptr);
                    }
                }

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

            /* Log CPUSET changes */
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

            /* Log workload changes */
            {
                int wl_threads = 0, wl_segments = 0;
                bool wl_changed = false;
                plugin_get_workload(m_state, &wl_threads, &wl_segments, &wl_changed);
                if (wl_changed && m_logged_processing) {
                    trellis_log("workload changed: %d threads, %d segments/ch",
                                wl_threads, wl_segments);
                }
            }

            /* Log periodic timing */
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
                st.dsd_rate_in = is_dop ? dsd_rate : 0;
                st.dsd_rate_out = out_is_pcm ? 0 : out_rate;
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

        chunk->set_data(out_as.get_ptr(), out_frames, channels, out_pcm_rate);

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
            uint32_t out_pcm_rate = m_config.fs_out / 16;
            if (out_pcm_rate == 0)
                out_pcm_rate = m_pcm_rate;

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
        m_chunk_count = 0;
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
    "Rate conversion and noise shaping for DSD streams.\n"
    "Supports DoP and native DSD input from foo_input_sacd / foo_input_udsd.\n\n"
    "NTF coefficients ported from mansr/sox sdm.c (LGPL v2.1+)"
);

VALIDATE_COMPONENT_FILENAME("foo_dsd_trellis.dll");
