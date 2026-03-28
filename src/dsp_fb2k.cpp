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
#include "../include/gpu_compute.h"
#include <commctrl.h>

/* DoP: 16 DSD bits packed per PCM frame */
#define DOP_BITS_PER_FRAME 16

extern "C" {
#include "../include/dsd_types.h"
#include "../include/simd_detect.h"
#include "../include/fir.h"
#include "../include/cpuset.h"
#include "../include/httpapi.h"
#include "../include/dop.h"
#include "../include/engine.h"
#include "../include/ntf.h"
#include "../include/tusbaudio.h"
#include "../include/sinad_measure.h"
#include "build_version.h"

/* Per-instance plugin state (opaque, defined in dsp_plugin.c) */
typedef struct plugin_state plugin_state_t;

#include "../include/resample.h"
plugin_state_t *plugin_create(void);
void            plugin_destroy(plugin_state_t *s);
void            plugin_set_config(plugin_state_t *s, const dsd_config_t *cfg);
const dsd_config_t *plugin_get_config(const plugin_state_t *s);
const engine_channel_t *plugin_get_channels(const plugin_state_t *s);
size_t          plugin_process(plugin_state_t *s,
                               const float *in_pcm, uint8_t *out_i24,
                               size_t pcm_frames, int num_channels,
                               uint32_t pcm_rate);
size_t          plugin_process_pcm(plugin_state_t *s,
                                    const float *in_pcm, uint8_t *out_i24,
                                    size_t pcm_frames, int num_channels,
                                    uint32_t pcm_rate);
size_t          plugin_process_pcm_to_pcm(plugin_state_t *s,
                                          const float *in_pcm, float *out_pcm,
                                          size_t pcm_frames, int num_channels,
                                          uint32_t pcm_rate_in, uint32_t pcm_rate_out);
size_t          plugin_drain(plugin_state_t *s, uint8_t *out_i24,
                             int num_channels);
size_t          plugin_generate_tail(plugin_state_t *s, uint8_t *out_i24,
                                      int num_channels, int ms);
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
int             plugin_get_selected_cores(const plugin_state_t *s,
                                           uint32_t *ids, int max_ids);
int             plugin_get_stressed_worker(const plugin_state_t *s, double *ratio);
uint32_t        plugin_get_worker_cpuset(const plugin_state_t *s, int worker_index);

/* Config serialization (config.c) */
size_t config_serialize(const dsd_config_t *cfg, uint8_t *buf, size_t buf_size);
int    config_deserialize(dsd_config_t *cfg, const uint8_t *buf, size_t buf_size);
void   config_validate(dsd_config_t *cfg);
}

/* ─── Rate map display helpers ─── */

static const wchar_t *g_rate_names[RATE_MAP_COUNT] = {
    L"44100", L"48000", L"88200", L"96000",
    L"176400", L"192000", L"352800", L"384000",
    L"DSD64", L"DSD128", L"DSD256", L"DSD512",
    L"705600", L"768000", L"1411200", L"1536000",
    L"DSD64/48", L"DSD128/48", L"DSD256/48", L"DSD512/48"
};

static const wchar_t *g_output_names[RATE_OUT_COUNT] = {
    L"-", L"DSD64", L"DSD128", L"DSD256", L"DSD512",
    L"PCM 44.1k", L"PCM 88.2k", L"PCM 176.4k", L"PCM 352.8k",
    L"DSD64/48", L"DSD128/48", L"DSD256/48", L"DSD512/48",
    L"PCM 48k", L"PCM 96k", L"PCM 192k", L"PCM 384k",
    L"PCM 705.6k", L"PCM 768k", L"PCM 1411.2k", L"PCM 1536k"
};

/* Display order for ListView rows: DSD rates first (paired 44/48),
 * then PCM rates ascending. Maps display row → RATEIDX. */
static const int g_display_order[RATE_MAP_COUNT] = {
    RATEIDX_DSD64,     RATEIDX_DSD64_48,
    RATEIDX_DSD128,    RATEIDX_DSD128_48,
    RATEIDX_DSD256,    RATEIDX_DSD256_48,
    RATEIDX_DSD512,    RATEIDX_DSD512_48,
    RATEIDX_44100,     RATEIDX_48000,
    RATEIDX_88200,     RATEIDX_96000,
    RATEIDX_176400,    RATEIDX_192000,
    RATEIDX_352800,    RATEIDX_384000,
    RATEIDX_705600,    RATEIDX_768000,
    RATEIDX_1411200,   RATEIDX_1536000,
};

/* Reverse mapping: RATEIDX → display row */
static int g_rateidx_to_row[RATE_MAP_COUNT];
static bool g_reverse_map_init = false;
static void ensure_reverse_map(void) {
    if (g_reverse_map_init) return;
    for (int r = 0; r < RATE_MAP_COUNT; r++)
        g_rateidx_to_row[g_display_order[r]] = r;
    g_reverse_map_init = true;
}

static const wchar_t *g_ntf_names[] = {
    L"Auto", L"CLANS-4", L"SDM-4", L"CLANS-5", L"SDM-5",
    L"CLANS-6", L"SDM-6", L"CLANS-7", L"SDM-7", L"CLANS-8", L"SDM-8"
};
#define NTF_NAME_COUNT (sizeof(g_ntf_names) / sizeof(g_ntf_names[0]))

/* Input rates by rate_map index (for path_info lookup in UI) */
static const uint32_t g_input_rates[RATE_MAP_COUNT] = {
    44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000,
    DSD_RATE_64, DSD_RATE_128, DSD_RATE_256, DSD_RATE_512,
    705600, 768000, 1411200, 1536000,
    DSD48_RATE_64, DSD48_RATE_128, DSD48_RATE_256, DSD48_RATE_512
};

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

/* Forward declaration for logger (defined below) */
static void trellis_log(const char *fmt, ...);

/* C-callable log function for engine/GPU code */
extern "C" void trellis_log_c(const char *msg) {
    trellis_log("%s", msg);
}

/* ─── Output device detection ─── */
/* Caches the current output module type (ASIO, WASAPI, DS, etc.)
 * and device name. Updated on init and via output_config_change_callback. */

static CRITICAL_SECTION g_output_cs;
static bool g_output_cs_init = false;
static char g_output_type[64] = "";     /* "ASIO+DSD", "DS", "WASAPI", etc. */
static char g_output_device[256] = "";  /* device name within the module */
static bool g_output_is_asio = false;

static void update_output_info() {
    try {
        auto cfg = output_manager::get()->getCoreConfig();
        output_entry::ptr entry;
        const char *type_name = "Unknown";
        if (output_entry::g_find(cfg.m_output, entry))
            type_name = entry->get_name();

        pfc::string8 device_name;
        if (output_entry::g_find(cfg.m_output, entry))
            device_name = entry->get_device_name(cfg.m_device);

        if (g_output_cs_init) EnterCriticalSection(&g_output_cs);
        strncpy_s(g_output_type, sizeof(g_output_type), type_name, _TRUNCATE);
        strncpy_s(g_output_device, sizeof(g_output_device),
                  device_name.get_ptr(), _TRUNCATE);
        g_output_is_asio = (strstr(type_name, "ASIO") != nullptr);
        if (g_output_cs_init) LeaveCriticalSection(&g_output_cs);
    } catch (...) {}
}

static bool is_output_asio() {
    if (g_output_cs_init) {
        EnterCriticalSection(&g_output_cs);
        bool result = g_output_is_asio;
        LeaveCriticalSection(&g_output_cs);
        return result;
    }
    return g_output_is_asio;
}

static void get_output_info(char *type_buf, size_t type_size,
                             char *device_buf, size_t device_size) {
    if (g_output_cs_init) EnterCriticalSection(&g_output_cs);
    if (type_buf)
        strncpy_s(type_buf, type_size, g_output_type, _TRUNCATE);
    if (device_buf)
        strncpy_s(device_buf, device_size, g_output_device, _TRUNCATE);
    if (g_output_cs_init) LeaveCriticalSection(&g_output_cs);
}

class output_change_callback_impl : public output_config_change_callback {
public:
    void outputConfigChanged() override {
        update_output_info();
        trellis_log("output changed: %s [%s]%s",
                    g_output_type, g_output_device,
                    g_output_is_asio ? " (ASIO)" : "");
    }
};

static output_change_callback_impl g_output_change_cb;

class output_init : public initquit {
public:
    void on_init() override {
        InitializeCriticalSection(&g_output_cs);
        g_output_cs_init = true;
        update_output_info();

        /* Register for output config change notifications */
        try {
            auto mgr = output_manager_v2::get();
            mgr->addCallback(&g_output_change_cb);
        } catch (...) {}
    }
    void on_quit() override {
        try {
            auto mgr = output_manager_v2::get();
            mgr->removeCallback(&g_output_change_cb);
        } catch (...) {}
        g_output_cs_init = false;
        DeleteCriticalSection(&g_output_cs);
    }
};

static initquit_factory_t<output_init> g_output_init;

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
    /* Open with shared read access so the REST API can read the log
     * while we're writing to it. _fsopen with _SH_DENYNO allows concurrent reads. */
    g_log_file = _fsopen(path.c_str(), "a", _SH_DENYNO);
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
        char line[512];
        snprintf(line, sizeof(line), "[%02d:%02d:%02d.%03d] %s",
                 st.wHour, st.wMinute, st.wSecond, st.wMilliseconds, buf);
        fprintf(g_log_file, "%s\n", line);
        fflush(g_log_file);
        /* Also write to ring buffer for API access */
        log_ring_write(line);
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

/* Our own GUID — added to foo_input_sacd's autoproxy_dsp whitelist
 * in v2.0.19-1. No longer need to impersonate foo_dsd_processor.
 * {7B3A4E5C-2D1F-4A8B-9C6E-3F5D8A2B1E7C} */
static const GUID g_dsp_guid =
    { 0x7b3a4e5c, 0x2d1f, 0x4a8b,
      { 0x9c, 0x6e, 0x3f, 0x5d, 0x8a, 0x2b, 0x1e, 0x7c } };

/* ─── Preset serialization using dsp_preset_builder/parser ─── */

static void make_preset(const dsd_config_t &cfg, dsp_preset &out) {
    uint8_t buf[512];
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

/* ─── ML EP combo ↔ enum mapping ─── */
/* Combo order: Auto(0), CPU(1), DirectML(2), CUDA(3) */
/* Enum order:  CPU(0), DirectML(1), Auto(2), CUDA(3) */
static int ml_ep_to_combo(int ep) {
    switch (ep) {
    case 2:  return 0;  /* Auto → first */
    case 0:  return 1;  /* CPU → second */
    case 1:  return 2;  /* DirectML → third */
    case 3:  return 3;  /* CUDA → fourth */
    default: return 0;
    }
}
static int combo_to_ml_ep(int idx) {
    switch (idx) {
    case 0:  return 2;  /* first → Auto */
    case 1:  return 0;  /* second → CPU */
    case 2:  return 1;  /* third → DirectML */
    case 3:  return 3;  /* fourth → CUDA */
    default: return 2;
    }
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
        /* IDC_COMBO_SDM_MODE removed — SDM is per-rate now */
        COMMAND_HANDLER_EX(IDC_COMBO_FORMAT, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_CHECK_DEBUG_LOG, BN_CLICKED, OnChange)
        COMMAND_HANDLER_EX(IDC_CHECK_ANTIPOP, BN_CLICKED, OnChange)
        COMMAND_HANDLER_EX(IDC_COMBO_FIR_GAIN, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_CHECK_ML_ENABLED, BN_CLICKED, OnMlChange)
        COMMAND_HANDLER_EX(IDC_COMBO_ML_EP, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_CHECK_GPU_ENABLED, BN_CLICKED, OnGpuChange)
        COMMAND_HANDLER_EX(IDC_CHECK_GPU_SDM, BN_CLICKED, OnChange)
        COMMAND_HANDLER_EX(IDC_COMBO_GPU_BACKEND, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_COMBO_PCM_BITS, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_COMBO_PCM_DITHER, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_COMBO_RESAMPLE, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_COMBO_SOXR_QUALITY, CBN_SELCHANGE, OnChange)
        COMMAND_HANDLER_EX(IDC_BTN_TEST_SINAD, BN_CLICKED, OnTestSinad)
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
        m_updating = true;  /* Prevent OnChange during init */

        /* Debug: log incoming preset data */
        {
            const void *pdata = m_initData.get_data();
            t_size psize = m_initData.get_data_size();
            pfc::string_formatter hex;
            hex << "preset init: " << (unsigned)psize << " bytes [";
            const uint8_t *pb = (const uint8_t *)pdata;
            for (t_size i = 0; i < psize && i < 84; i++) {
                if (i > 0) hex << " ";
                hex << pfc::format_hex(pb[i], 2);
            }
            hex << "]";
            console::print(hex);
        }

        m_cfg = parse_preset(m_initData);

        /* Debug: log parsed config */
        {
            pfc::string_formatter dbg;
            dbg << "parsed config: debug_log=" << (int)m_cfg.debug_log
                << " ml_enabled=" << (int)m_cfg.ml_enabled
                << " ml_ep=" << m_cfg.ml_ep
                << " rate_map[9]=" << (int)m_cfg.rate_map[9];
            console::print(dbg);
        }

        /* Rate map ListView */
        m_listRate = GetDlgItem(IDC_LIST_RATEMAP);
        m_listRate.SetExtendedListViewStyle(
            LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES);
        ensure_reverse_map();
        int cw[] = { 66, 66, 50, 46, 42, 42, 46, 34, 32, 38, 32, 32 };
        m_listRate.InsertColumn(0, L"Input", LVCFMT_LEFT, cw[0]);
        m_listRate.InsertColumn(1, L"Output", LVCFMT_LEFT, cw[1]);
        m_listRate.InsertColumn(2, L"NTF", LVCFMT_LEFT, cw[2]);
        m_listRate.InsertColumn(3, L"SDM", LVCFMT_LEFT, cw[3]);
        m_listRate.InsertColumn(4, L"Cands", LVCFMT_LEFT, cw[4]);
        m_listRate.InsertColumn(5, L"Depth", LVCFMT_LEFT, cw[5]);
        m_listRate.InsertColumn(6, L"Limiter", LVCFMT_LEFT, cw[6]);
        m_listRate.InsertColumn(7, L"ML", LVCFMT_LEFT, cw[7]);
        m_listRate.InsertColumn(8, L"GPU", LVCFMT_LEFT, cw[8]);
        m_listRate.InsertColumn(9, L"FIR", LVCFMT_LEFT, cw[9]);
        m_listRate.InsertColumn(10, L"Par", LVCFMT_LEFT, cw[10]);
        m_listRate.InsertColumn(11, L"Lat", LVCFMT_LEFT, cw[11]);
        /* Last column: fill remaining width to avoid horizontal scrollbar */
        {
            CRect rc;
            m_listRate.GetClientRect(&rc);
            int used = 0;
            for (int c = 0; c < 12; c++) used += cw[c];
            int vscroll = GetSystemMetrics(SM_CXVSCROLL);
            int remain = rc.Width() - used - vscroll - 4;
            if (remain < 36) remain = 36;
            m_listRate.InsertColumn(12, L"Prec", LVCFMT_LEFT, remain);
        }

        for (int row = 0; row < RATE_MAP_COUNT; row++) {
            int i = g_display_order[row];  /* RATEIDX for this display row */
            m_listRate.InsertItem(row, g_rate_names[i]);
            m_listRate.SetItemText(row, 1, g_output_names[m_cfg.rate_map[i]]);
            int ntf_idx = m_cfg.rate_ntf[i] + 1; /* NTF_AUTO=-1 → 0 */
            if (ntf_idx < 0 || ntf_idx >= (int)NTF_NAME_COUNT) ntf_idx = 0;
            m_listRate.SetItemText(row, 2, g_ntf_names[ntf_idx]);
            /* Set explicit (non-Auto) values first */
            {
                wchar_t buf[32];
                if (m_cfg.rate_sdm[i] >= 0) {
                    wcscpy_s(buf, m_cfg.rate_sdm[i] == SDM_MODE_PRECORR ? L"PreCorr" : L"Trellis");
                    m_listRate.SetItemText(row, 3, buf);
                }
                if (m_cfg.rate_cands[i] >= 0) {
                    _snwprintf_s(buf, _countof(buf), _TRUNCATE, L"%d", m_cfg.rate_cands[i]);
                    m_listRate.SetItemText(row, 4, buf);
                }
                if (m_cfg.rate_depth[i] >= 0) {
                    _snwprintf_s(buf, _countof(buf), _TRUNCATE, L"%d", m_cfg.rate_depth[i]);
                    m_listRate.SetItemText(row, 5, buf);
                }
                if (m_cfg.rate_limiter[i] >= 0) {
                    if (m_cfg.rate_limiter[i] == 0) wcscpy_s(buf, L"Off");
                    else _snwprintf_s(buf, _countof(buf), _TRUNCATE, L"%.1f", (float)m_cfg.rate_limiter[i]);
                    m_listRate.SetItemText(row, 6, buf);
                }
                if (m_cfg.rate_ml[i] >= 0) {
                    wcscpy_s(buf, m_cfg.rate_ml[i] == 0 ? L"Off" : L"On");
                    m_listRate.SetItemText(row, 7, buf);
                }
                if (m_cfg.rate_gpu[i] >= 0) {
                    wcscpy_s(buf, m_cfg.rate_gpu[i] == 0 ? L"Off" : L"On");
                    m_listRate.SetItemText(row, 8, buf);
                }
                if (m_cfg.rate_fir_mode[i] >= 0) {
                    wcscpy_s(buf, m_cfg.rate_fir_mode[i] == FIR_MODE_BOXCAR ? L"Boxcar" : L"FIR");
                    m_listRate.SetItemText(row, 9, buf);
                }
                if (m_cfg.rate_parallel[i] >= 0) {
                    if (m_cfg.rate_parallel[i] == TRELLIS_PAR_SEQUENTIAL)
                        wcscpy_s(buf, L"Seq");
                    else if (m_cfg.rate_parallel[i] >= 2 && m_cfg.rate_parallel[i] <= 8) {
                        swprintf_s(buf, sizeof(buf)/sizeof(buf[0]), L"Par%d", m_cfg.rate_parallel[i]);
                    } else
                        wcscpy_s(buf, L"Par2");
                    m_listRate.SetItemText(row, 10, buf);
                }
                if (m_cfg.rate_lat[i] > 0) {
                    _snwprintf_s(buf, _countof(buf), _TRUNCATE, L"%d", m_cfg.rate_lat[i]);
                    m_listRate.SetItemText(row, 11, buf);
                }
                if (m_cfg.rate_fir_prec[i] >= 0) {
                    wcscpy_s(buf, m_cfg.rate_fir_prec[i] == FIR_PREC_FP32 ? L"FP32" : L"FP64");
                    m_listRate.SetItemText(row, 12, buf);
                }
            }
            /* Resolve Auto values using path-optimal defaults */
            RefreshAutoText(row);
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

        /* SDM Mode is now per-rate (in the ListView), no global combo */

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

        /* Anti-pop checkbox */
        CheckDlgButton(IDC_CHECK_ANTIPOP, m_cfg.antipop ? BST_CHECKED : BST_UNCHECKED);

        /* FIR Gain combo: Auto(-3dB), 0dB, -1dB, ... -12dB */
        {
            CComboBox firg(GetDlgItem(IDC_COMBO_FIR_GAIN));
            firg.AddString(L"Auto (-3 dB)");
            firg.AddString(L"0 dB");
            for (int i = -1; i >= -12; i--) {
                wchar_t buf[16];
                _snwprintf_s(buf, _countof(buf), _TRUNCATE, L"%d dB", i);
                firg.AddString(buf);
            }
            /* Map fir_gain_db to combo index: Auto=0, 0dB=1, -1dB=2, ... -12dB=13 */
            int sel = 0;
            if (m_cfg.fir_gain_db != FIR_GAIN_AUTO) {
                int db = (int)m_cfg.fir_gain_db;
                if (db >= 0) sel = 1;
                else if (db >= -12) sel = 1 - db;  /* -1→2, -2→3, ..., -12→13 */
                else sel = 13;
            }
            firg.SetCurSel(sel);
        }

        /* ML Noise Filter */
        CheckDlgButton(IDC_CHECK_ML_ENABLED, m_cfg.ml_enabled ? BST_CHECKED : BST_UNCHECKED);
        {
            CComboBox mlep(GetDlgItem(IDC_COMBO_ML_EP));
            /* Display order: Auto(0), CPU(1), DirectML(2), CUDA(3) */
            mlep.AddString(L"Auto");
            mlep.AddString(L"CPU");
            mlep.AddString(L"DirectML (GPU)");
            mlep.AddString(L"CUDA (GPU)");
            mlep.SetCurSel(ml_ep_to_combo(m_cfg.ml_ep));
            mlep.EnableWindow(m_cfg.ml_enabled);
        }
        UpdateMlStatus();

        /* GPU Compute */
        CheckDlgButton(IDC_CHECK_GPU_ENABLED, m_cfg.gpu_enabled ? BST_CHECKED : BST_UNCHECKED);
        {
            CComboBox gpub(GetDlgItem(IDC_COMBO_GPU_BACKEND));
            gpub.AddString(L"Auto");
            gpub.AddString(L"DirectCompute");
            gpub.AddString(L"CUDA");
            /* Map gpu_backend: None=0→Auto, DX=1, CUDA=2, Auto=3→0 */
            int sel = 0;
            switch (m_cfg.gpu_backend) {
            case 1: sel = 1; break;  /* DirectX */
            case 2: sel = 2; break;  /* CUDA */
            default: sel = 0; break; /* Auto or None */
            }
            gpub.SetCurSel(sel);
            gpub.EnableWindow(m_cfg.gpu_enabled);
        }
        /* GPU SDM hidden — 2-stage multibit approach parked */
        CheckDlgButton(IDC_CHECK_GPU_SDM, BST_UNCHECKED);
        ::ShowWindow(GetDlgItem(IDC_CHECK_GPU_SDM), SW_HIDE);
        UpdateGpuStatus();

        /* PCM Encoding */
        {
            CComboBox bits(GetDlgItem(IDC_COMBO_PCM_BITS));
            bits.AddString(L"Auto (float)");
            bits.AddString(L"16-bit");
            bits.AddString(L"24-bit");
            bits.AddString(L"32-bit");
            bits.AddString(L"Float");
            /* Map: Auto(-1)→0, 16→1, 24→2, 32→3, float→4 */
            int sel = 0;
            switch (m_cfg.pcm_bit_depth) {
            case PCM_BIT_16: sel = 1; break;
            case PCM_BIT_24: sel = 2; break;
            case PCM_BIT_32: sel = 3; break;
            case PCM_BIT_FLOAT: sel = 4; break;
            default: sel = 0; break;
            }
            bits.SetCurSel(sel);
        }
        {
            CComboBox dith(GetDlgItem(IDC_COMBO_PCM_DITHER));
            dith.AddString(L"Auto");
            dith.AddString(L"None");
            dith.AddString(L"TPDF");
            dith.AddString(L"Shaped");
            int sel = 0;
            switch (m_cfg.pcm_dither) {
            case PCM_DITHER_NONE: sel = 1; break;
            case PCM_DITHER_TPDF: sel = 2; break;
            case PCM_DITHER_SHAPED: sel = 3; break;
            default: sel = 0; break;
            }
            dith.SetCurSel(sel);
        }
        {
            CComboBox rs(GetDlgItem(IDC_COMBO_RESAMPLE));
            rs.AddString(L"Auto");
            rs.AddString(L"IPP");
            rs.AddString(L"soxr");
            int sel = 0;
            switch (m_cfg.resample_engine) {
            case RESAMPLE_IPP: sel = 1; break;
            case RESAMPLE_SOXR: sel = 2; break;
            default: sel = 0; break;
            }
            rs.SetCurSel(sel);
        }
        {
            CComboBox sq(GetDlgItem(IDC_COMBO_SOXR_QUALITY));
            sq.AddString(L"Medium");
            sq.AddString(L"High");
            sq.AddString(L"Very High");
            int sel = 1;  /* default HQ */
            switch (m_cfg.soxr_quality) {
            case SOXR_QUALITY_MQ: sel = 0; break;
            case SOXR_QUALITY_VHQ: sel = 2; break;
            default: sel = 1; break;
            }
            sq.SetCurSel(sel);
        }

        /* Show initial engine info */
        const cpu_features_t *cpu = cpu_detect();
        pfc::string_formatter info;
        info << "FIR: " << fir_ipp_kernel_name()
             << " | CPU: " << (cpu->vendor == CPU_VENDOR_INTEL ? "Intel" :
                               cpu->vendor == CPU_VENDOR_AMD   ? "AMD" : "Unknown")
             << "\nSelect a rate mapping to see path details.";
        ::uSetDlgItemText(*this, IDC_STATIC_PATH_INFO, info);

        m_updating = false;  /* Init done — allow OnChange to fire */
        return TRUE;
    }

    void OnButton(UINT, int id, CWindow) {
        if (id == IDOK) UpdatePreset();  /* Ensure final state is saved */
        EndDialog(id);
    }

    void OnChange(UINT, int, CWindow) {
        /* Settings only apply on OK — no live reconfigure during playback */
    }

    void OnEditChange(UINT, int, CWindow) {
        /* Settings only apply on OK */
    }

    void OnMlChange(UINT, int, CWindow) {
        if (m_updating) return;
        bool enabled = IsDlgButtonChecked(IDC_CHECK_ML_ENABLED) == BST_CHECKED;
        CComboBox(GetDlgItem(IDC_COMBO_ML_EP)).EnableWindow(enabled);
        UpdateMlStatus();
    }

    void UpdateMlStatus() {
        bool enabled = IsDlgButtonChecked(IDC_CHECK_ML_ENABLED) == BST_CHECKED;
        const char *status = "";
        if (enabled) {
            if (!onnx_runtime_available()) {
                status = "onnxruntime.dll not found";
            } else {
                int ep = combo_to_ml_ep(
                    CComboBox(GetDlgItem(IDC_COMBO_ML_EP)).GetCurSel());
                bool has_dml = false, has_cuda = false;
                if (ep != 0) {  /* Any GPU option: probe DLL from component folder */
                    extern bool onnx_runtime_available(void);
                    /* Use the same DLL that onnx_filter_create will use.
                     * Probe the already-cached availability + check EP exports. */
                    if (onnx_runtime_available()) {
                        /* DLL is in our component folder — re-load to check EPs */
                        wchar_t ort_path[MAX_PATH];
                        HMODULE hmod = GetModuleHandleW(L"foo_dsd_trellis.dll");
                        HMODULE hort = NULL;
                        if (hmod) {
                            DWORD len = GetModuleFileNameW(hmod, ort_path, MAX_PATH);
                            if (len > 0 && len < MAX_PATH) {
                                wchar_t *sep = wcsrchr(ort_path, L'\\');
                                if (sep) sep[1] = L'\0';
                                wcscat_s(ort_path, MAX_PATH, L"onnxruntime.dll");
                                hort = LoadLibraryW(ort_path);
                            }
                        }
                        if (!hort) hort = LoadLibraryW(L"onnxruntime.dll");
                        if (hort) {
                            has_dml = (GetProcAddress(hort,
                                "OrtSessionOptionsAppendExecutionProvider_DML") != NULL);
                            has_cuda = (GetProcAddress(hort,
                                "OrtSessionOptionsAppendExecutionProvider_CUDA") != NULL);
                            FreeLibrary(hort);
                        }
                    }
                }
                if (ep == 0)       status = "Ready (CPU)";
                else if (ep == 3)  status = has_cuda ? "Ready (CUDA)" : "Ready (CPU)";
                else if (ep == 1)  status = has_dml ? "Ready (DirectML)" : "Ready (CPU)";
                else               status = (has_cuda || has_dml) ? "Ready (GPU)" : "Ready (CPU)";
            }
        }
        ::uSetDlgItemText(*this, IDC_STATIC_ML_STATUS, status);
    }

    void OnGpuChange(UINT, int, CWindow) {
        if (m_updating) return;
        bool enabled = IsDlgButtonChecked(IDC_CHECK_GPU_ENABLED) == BST_CHECKED;
        CComboBox(GetDlgItem(IDC_COMBO_GPU_BACKEND)).EnableWindow(enabled);
        UpdateGpuStatus();
    }

    void UpdateGpuStatus() {
        bool enabled = IsDlgButtonChecked(IDC_CHECK_GPU_ENABLED) == BST_CHECKED;
        if (!enabled) {
            ::uSetDlgItemText(*this, IDC_STATIC_GPU_INFO, "");
            return;
        }

        gpu_info_t info;
        gpu_get_info(&info);
        if (info.available) {
            char buf[256];
            _snprintf_s(buf, sizeof(buf), _TRUNCATE, "%s (%zu MB)",
                         info.device_name, info.vram_mb);
            ::uSetDlgItemText(*this, IDC_STATIC_GPU_INFO, buf);
        } else {
            ::uSetDlgItemText(*this, IDC_STATIC_GPU_INFO, "No GPU detected");
        }
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
            ShowOutputCombo(nm->iItem);
        } else if (nm->iSubItem == 2) {
            ShowNtfCombo(nm->iItem);
        } else if (nm->iSubItem >= 3 && nm->iSubItem <= 12) {
            ShowPerRateCombo(nm->iItem, nm->iSubItem);
        }

        return 0;
    }

    void ShowOutputCombo(int row) {
        int ridx = rowToIdx(row);
        if (ridx < 0) return;
        CRect cellRC;
        m_listRate.GetSubItemRect(row, 1, LVIR_BOUNDS, &cellRC);
        m_listRate.MapWindowPoints(m_hWnd, &cellRC);

        m_editCombo.ResetContent();
        for (int j = 0; j < RATE_OUT_COUNT; j++) {
            if (rate_map_valid_output(ridx, (uint8_t)j)) {
                int ci = m_editCombo.AddString(g_output_names[j]);
                m_editCombo.SetItemData(ci, (DWORD_PTR)j);
            }
        }

        uint8_t cur = m_cfg.rate_map[ridx];
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

        int8_t cur = m_cfg.rate_ntf[rowToIdx(row)];
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

    void ShowPerRateCombo(int row, int col) {
        CRect cellRC;
        m_listRate.GetSubItemRect(row, col, LVIR_BOUNDS, &cellRC);
        m_listRate.MapWindowPoints(m_hWnd, &cellRC);

        m_ntfCombo.ResetContent();

        if (col == 3) {
            /* SDM mode */
            m_ntfCombo.AddString(L"Auto");
            m_ntfCombo.AddString(L"PreCorr");
            m_ntfCombo.AddString(L"Trellis");
            int cur = m_cfg.rate_sdm[rowToIdx(row)];
            m_ntfCombo.SetCurSel(cur < 0 ? 0 : cur + 1);
        } else if (col == 4) {
            /* Candidates */
            m_ntfCombo.AddString(L"Auto");
            m_ntfCombo.AddString(L"2"); m_ntfCombo.AddString(L"4");
            m_ntfCombo.AddString(L"8"); m_ntfCombo.AddString(L"16");
            m_ntfCombo.AddString(L"32");
            int cur = m_cfg.rate_cands[rowToIdx(row)];
            int sel = 0;
            if (cur == 2) sel = 1; else if (cur == 4) sel = 2;
            else if (cur == 8) sel = 3; else if (cur == 16) sel = 4;
            else if (cur == 32) sel = 5;
            m_ntfCombo.SetCurSel(sel);
        } else if (col == 5) {
            /* Depth */
            m_ntfCombo.AddString(L"Auto");
            m_ntfCombo.AddString(L"4"); m_ntfCombo.AddString(L"5");
            m_ntfCombo.AddString(L"6"); m_ntfCombo.AddString(L"7");
            m_ntfCombo.AddString(L"8");
            int cur = m_cfg.rate_depth[rowToIdx(row)];
            int sel = 0;
            if (cur >= 4 && cur <= 8) sel = cur - 3;
            m_ntfCombo.SetCurSel(sel);
        } else if (col == 6) {
            /* Limiter: Auto, Off, 3, 6, 8, 10, 12, 16, 20 */
            m_ntfCombo.AddString(L"Auto");
            m_ntfCombo.AddString(L"Off");
            m_ntfCombo.AddString(L"3"); m_ntfCombo.AddString(L"6");
            m_ntfCombo.AddString(L"8"); m_ntfCombo.AddString(L"10");
            m_ntfCombo.AddString(L"12"); m_ntfCombo.AddString(L"16");
            m_ntfCombo.AddString(L"20");
            int cur = m_cfg.rate_limiter[rowToIdx(row)];
            int sel = 0;
            if (cur == 0) sel = 1;
            else if (cur == 3) sel = 2; else if (cur == 6) sel = 3;
            else if (cur == 8) sel = 4; else if (cur == 10) sel = 5;
            else if (cur == 12) sel = 6; else if (cur == 16) sel = 7;
            else if (cur == 20) sel = 8;
            m_ntfCombo.SetCurSel(sel);
        } else if (col == 7) {
            /* ML */
            m_ntfCombo.AddString(L"Auto");
            m_ntfCombo.AddString(L"Off");
            m_ntfCombo.AddString(L"On");
            int cur = m_cfg.rate_ml[rowToIdx(row)];
            m_ntfCombo.SetCurSel(cur < 0 ? 0 : cur + 1);
        } else if (col == 8) {
            /* GPU FIR */
            m_ntfCombo.AddString(L"Auto");
            m_ntfCombo.AddString(L"Off");
            m_ntfCombo.AddString(L"On");
            int cur = m_cfg.rate_gpu[rowToIdx(row)];
            m_ntfCombo.SetCurSel(cur < 0 ? 0 : cur + 1);
        } else if (col == 9) {
            /* FIR mode: pre-SDM filter */
            m_ntfCombo.AddString(L"Auto");
            m_ntfCombo.AddString(L"Boxcar");
            m_ntfCombo.AddString(L"FIR");
            int cur = m_cfg.rate_fir_mode[rowToIdx(row)];
            m_ntfCombo.SetCurSel(cur < 0 ? 0 : cur + 1);
        } else if (col == 10) {
            /* Parallel mode: Auto/Seq/Par2-Par8 (DAS segment count) */
            m_ntfCombo.AddString(L"Auto");
            m_ntfCombo.AddString(L"Seq");
            m_ntfCombo.AddString(L"Par2");
            m_ntfCombo.AddString(L"Par3");
            m_ntfCombo.AddString(L"Par4");
            m_ntfCombo.AddString(L"Par5");
            m_ntfCombo.AddString(L"Par6");
            m_ntfCombo.AddString(L"Par7");
            m_ntfCombo.AddString(L"Par8");
            int cur = m_cfg.rate_parallel[rowToIdx(row)];
            /* Map: -1=Auto→0, 0=Seq→1, 2=Par2→2, 3=Par3→3, ... 8=Par8→8 */
            int sel_idx = 0;
            if (cur == 0) sel_idx = 1;
            else if (cur >= 2 && cur <= 8) sel_idx = cur;
            m_ntfCombo.SetCurSel(sel_idx);
        } else if (col == 11) {
            /* Trellis latency */
            m_ntfCombo.AddString(L"Auto");
            m_ntfCombo.AddString(L"16"); m_ntfCombo.AddString(L"32");
            m_ntfCombo.AddString(L"64"); m_ntfCombo.AddString(L"128");
            m_ntfCombo.AddString(L"256"); m_ntfCombo.AddString(L"512");
            int16_t cur = m_cfg.rate_lat[rowToIdx(row)];
            int sel = 0;
            if (cur == 16) sel = 1; else if (cur == 32) sel = 2;
            else if (cur == 64) sel = 3; else if (cur == 128) sel = 4;
            else if (cur == 256) sel = 5; else if (cur == 512) sel = 6;
            m_ntfCombo.SetCurSel(sel);
        } else if (col == 12) {
            /* FIR precision: Auto(fp64), FP32, FP64 */
            m_ntfCombo.AddString(L"Auto");
            m_ntfCombo.AddString(L"FP32");
            m_ntfCombo.AddString(L"FP64");
            int cur = m_cfg.rate_fir_prec[rowToIdx(row)];
            m_ntfCombo.SetCurSel(cur < 0 ? 0 : cur + 1);
        }

        m_editRow = row;
        m_editCol = col;
        m_ntfCombo.SetWindowPos(HWND_TOP,
                                cellRC.left, cellRC.top,
                                cellRC.Width(), 200,
                                SWP_SHOWWINDOW);
        m_ntfCombo.SetFocus();
        m_ntfCombo.ShowDropDown(TRUE);
    }

    /* Convert display row → config array index (RATEIDX) */
    int rowToIdx(int row) const {
        if (row < 0 || row >= RATE_MAP_COUNT) return -1;
        return g_display_order[row];
    }

    void OnRateMapEditChange(UINT, int, CWindow) {
        int sel = m_editCombo.GetCurSel();
        int idx = rowToIdx(m_editRow);
        if (sel >= 0 && idx >= 0) {
            uint8_t out_idx = (uint8_t)m_editCombo.GetItemData(sel);
            m_cfg.rate_map[idx] = out_idx;
            m_listRate.SetItemText(m_editRow, 1, g_output_names[out_idx]);
            RefreshAutoText(m_editRow);
            UpdatePathInfo(m_editRow);
        }
    }

    void OnNtfEditChange(UINT, int, CWindow) {
        int sel = m_ntfCombo.GetCurSel();
        int idx = rowToIdx(m_editRow);
        if (sel < 0 || idx < 0)
            return;

        if (m_editCol == 2) {
            int8_t ntf_val = (int8_t)(sel - 1);
            m_cfg.rate_ntf[idx] = ntf_val;
            m_listRate.SetItemText(m_editRow, 2, g_ntf_names[sel]);
        } else if (m_editCol == 3) {
            int8_t val = (int8_t)(sel == 0 ? -1 : sel - 1);
            m_cfg.rate_sdm[idx] = val;
            const wchar_t *names[] = { L"Auto", L"PreCorr", L"Trellis" };
            m_listRate.SetItemText(m_editRow, 3, names[sel < 3 ? sel : 0]);
        } else if (m_editCol == 4) {
            static const int8_t cvals[] = { -1, 2, 4, 8, 16, 32 };
            int8_t val = (sel >= 0 && sel < 6) ? cvals[sel] : -1;
            m_cfg.rate_cands[idx] = val;
            if (val >= 0) {
                wchar_t buf[16];
                _snwprintf_s(buf, _countof(buf), _TRUNCATE, L"%d", val);
                m_listRate.SetItemText(m_editRow, 4, buf);
            }
        } else if (m_editCol == 5) {
            int8_t val = (sel == 0) ? -1 : (int8_t)(sel + 3);
            m_cfg.rate_depth[idx] = val;
            if (val >= 0) {
                wchar_t buf[16];
                _snwprintf_s(buf, _countof(buf), _TRUNCATE, L"%d", val);
                m_listRate.SetItemText(m_editRow, 5, buf);
            }
        } else if (m_editCol == 6) {
            static const int8_t lvals[] = { -1, 0, 3, 6, 8, 10, 12, 16, 20 };
            int8_t val = (sel >= 0 && sel < 9) ? lvals[sel] : -1;
            m_cfg.rate_limiter[idx] = val;
            if (val > 0) {
                wchar_t buf[16];
                _snwprintf_s(buf, _countof(buf), _TRUNCATE, L"%.1f", (float)val);
                m_listRate.SetItemText(m_editRow, 6, buf);
            } else if (val == 0) {
                m_listRate.SetItemText(m_editRow, 6, L"Off");
            }
        } else if (m_editCol == 7) {
            int8_t val = (int8_t)(sel == 0 ? -1 : sel - 1);
            m_cfg.rate_ml[idx] = val;
            const wchar_t *names[] = { L"Auto", L"Off", L"On" };
            m_listRate.SetItemText(m_editRow, 7, names[sel < 3 ? sel : 0]);
        } else if (m_editCol == 8) {
            int8_t val = (int8_t)(sel == 0 ? -1 : sel - 1);
            m_cfg.rate_gpu[idx] = val;
            const wchar_t *names[] = { L"Auto", L"Off", L"On" };
            m_listRate.SetItemText(m_editRow, 8, names[sel < 3 ? sel : 0]);
        } else if (m_editCol == 9) {
            int8_t val = (int8_t)(sel == 0 ? -1 : sel - 1);
            m_cfg.rate_fir_mode[idx] = val;
            const wchar_t *names[] = { L"Auto", L"Boxcar", L"FIR" };
            m_listRate.SetItemText(m_editRow, 9, names[sel < 3 ? sel : 0]);
        } else if (m_editCol == 10) {
            /* Par: 0=Auto, 1=Seq, 2=Par2, 3=Par3, ... 8=Par8 */
            static const int8_t par_vals[] = { -1, 0, 2, 3, 4, 5, 6, 7, 8 };
            int8_t val = (sel >= 0 && sel < 9) ? par_vals[sel] : -1;
            m_cfg.rate_parallel[idx] = val;
            static const wchar_t *names[] = {
                L"Auto", L"Seq", L"Par2", L"Par3", L"Par4",
                L"Par5", L"Par6", L"Par7", L"Par8"
            };
            m_listRate.SetItemText(m_editRow, 10, names[sel < 9 ? sel : 0]);
        } else if (m_editCol == 11) {
            static const int16_t lvals[] = { 0, 16, 32, 64, 128, 256, 512 };
            int16_t val = (sel >= 0 && sel < 7) ? lvals[sel] : 0;
            m_cfg.rate_lat[idx] = val;
            if (val > 0) {
                wchar_t buf[16];
                _snwprintf_s(buf, _countof(buf), _TRUNCATE, L"%d", val);
                m_listRate.SetItemText(m_editRow, 11, buf);
            }
        } else if (m_editCol == 12) {
            /* FIR precision: 0=Auto(-1), 1=FP32(0), 2=FP64(1) */
            int8_t val = (int8_t)(sel == 0 ? -1 : sel - 1);
            m_cfg.rate_fir_prec[idx] = val;
            const wchar_t *names[] = { L"Auto", L"FP32", L"FP64" };
            m_listRate.SetItemText(m_editRow, 12, names[sel < 3 ? sel : 0]);
        }

        RefreshAutoText(m_editRow);
        UpdatePathInfo(m_editRow);
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
        int idx = rowToIdx(row);
        if (idx < 0) return;

        uint8_t out_idx = m_cfg.rate_map[idx];
        pfc::string_formatter info;

        if (out_idx == RATE_OUT_BYPASS) {
            info << g_rate_names[idx] << ": Passthrough (no processing)";
            ::uSetDlgItemText(*this, IDC_STATIC_PATH_INFO, info);
            return;
        }

        /* Determine input/output rates */
        uint32_t fs_in = rate_idx_to_hz(idx);
        bool is_dsd_input = rate_idx_is_dsd(idx);

        uint32_t fs_out = rate_out_to_hz(out_idx);
        if (fs_out == 0) {
            info << "Invalid output configuration";
            ::uSetDlgItemText(*this, IDC_STATIC_PATH_INFO, info);
            return;
        }

        /* Query path info — always query Trellis path for resolved defaults */
        int8_t ntf_val = m_cfg.rate_ntf[rowToIdx(row)];
        int sdm_mode = (m_cfg.rate_sdm[rowToIdx(row)] >= 0) ? (int)m_cfg.rate_sdm[rowToIdx(row)] : m_cfg.sdm_mode;

        /* Always get Trellis path info for resolved defaults */
        engine_path_info_t pi_trellis;
        engine_get_path_info(fs_in, fs_out, (int)ntf_val, SDM_MODE_TRELLIS, &m_cfg, &pi_trellis);

        /* Also get current mode's path info */
        engine_path_info_t pi;
        if (sdm_mode == SDM_MODE_TRELLIS)
            pi = pi_trellis;
        else
            engine_get_path_info(fs_in, fs_out, (int)ntf_val, sdm_mode, &m_cfg, &pi);

        /* Format info text */
        info << g_rate_names[idx];
        if (is_dsd_input && rate_out_is_dsd(out_idx)) {
            if (fs_in == fs_out)
                info << " -> " << g_output_names[out_idx] << " (re-encode)";
            else if (fs_out > fs_in)
                info << " -> " << g_output_names[out_idx] << " (" << (fs_out/fs_in) << "x upsample)";
            else
                info << " -> " << g_output_names[out_idx] << " (1/" << (fs_in/fs_out) << " downsample)";
        } else if (is_dsd_input && rate_out_is_pcm(out_idx)) {
            uint32_t ratio = fs_in / fs_out;
            info << " -> " << g_output_names[out_idx] << " (1/" << ratio << " decimation)";
        } else if (!is_dsd_input && rate_out_is_dsd(out_idx)) {
            uint32_t ratio = fs_out / fs_in;
            info << " -> " << g_output_names[out_idx] << " (" << ratio << "x upsample to DSD)";
        } else if (!is_dsd_input && rate_out_is_pcm(out_idx)) {
            if (fs_out > fs_in)
                info << " -> " << g_output_names[out_idx] << " (" << (fs_out/fs_in) << "x upsample)";
            else if (fs_out < fs_in)
                info << " -> " << g_output_names[out_idx] << " (1/" << (fs_in/fs_out) << " downsample)";
            else
                info << " -> " << g_output_names[out_idx] << " (passthrough)";
        } else {
            info << " -> " << g_output_names[out_idx];
        }

        info << "\nFIR: " << pi.fir_stages << " stages, " << fir_ipp_kernel_name();

        if (pi.fir_only) {
            info << "\nFIR decimation only (no SDM)";
            /* Show resampler for cross-family DSD→PCM */
            bool dsd_in = is_dsd_input;
            bool pcm_out_v = rate_out_is_pcm(out_idx);
            if (dsd_in && pcm_out_v) {
                bool in_48k = rate_idx_is_48k_family(idx);
                bool out_48k = rate_is_48k_family(fs_out);
                if (in_48k != out_48k) {
                    uint32_t inter = in_48k ? 48000u : 44100u;
                    while (inter * 2 <= fs_out && inter * 2 <= (in_48k ? 384000u : 352800u))
                        inter *= 2;
                    const char *rs = resample_soxr_available() ? "soxr" : "IPP";
                    if (m_cfg.resample_engine == RESAMPLE_IPP) rs = "IPP";
                    else if (m_cfg.resample_engine == RESAMPLE_SOXR) rs = "soxr";
                    info << "\nCross-family: FIR→" << inter << "Hz → " << rs << " polyphase → " << fs_out << "Hz";
                }
            }
        } else if (!is_dsd_input && rate_out_is_pcm(out_idx)) {
            /* PCM→PCM path */
            bool needs_poly = resample_needed(fs_in, fs_out);
            if (needs_poly) {
                const char *rs = resample_soxr_available() ? "soxr" : "IPP";
                if (m_cfg.resample_engine == RESAMPLE_IPP) rs = "IPP";
                else if (m_cfg.resample_engine == RESAMPLE_SOXR) rs = "soxr";
                info << "\nPolyphase resampler: " << rs;
                if (m_cfg.resample_engine != RESAMPLE_IPP && resample_soxr_available()) {
                    static const char *sq[] = { "Medium", "High", "Very High" };
                    int qi = m_cfg.soxr_quality;
                    if (qi < 0 || qi > 2) qi = 1;
                    info << " (" << sq[qi] << ")";
                }
            } else {
                info << "\nFIR chain (power-of-2 ratio, no SDM)";
            }
        } else {
            /* Check for cross-family PCM→DSD */
            if (!is_dsd_input && rate_out_is_dsd(out_idx)) {
                bool in_48k = rate_idx_is_48k_family(idx);
                bool out_48k = rate_out_is_dsd48(out_idx);
                if (in_48k != out_48k) {
                    uint32_t target_base = out_48k ? 48000u : 44100u;
                    uint32_t inter = target_base;
                    while (inter * 2 <= fs_in)
                        inter *= 2;
                    const char *rs = resample_soxr_available() ? "soxr" : "IPP";
                    if (m_cfg.resample_engine == RESAMPLE_IPP) rs = "IPP";
                    else if (m_cfg.resample_engine == RESAMPLE_SOXR) rs = "soxr";
                    info << "\nCross-family: " << rs << " " << fs_in << "Hz -> " << inter << "Hz -> FIR upsample";
                }
            }
            const char *sdm_name = (sdm_mode == SDM_MODE_TRELLIS) ? "Trellis" : "PreCorr";
            info << "\nSDM: " << sdm_name;
            if (sdm_mode == SDM_MODE_TRELLIS)
                info << ", cands=" << pi.cands << ", depth=" << pi.depth << ", lat=" << pi.lat;

            /* NTF: resolve and show actual filter name */
            {
                int ntf_id = (sdm_mode == SDM_MODE_TRELLIS) ? pi_trellis.ntf_filter : pi.ntf_filter;
                const char *ntf_name = NULL;
                if (ntf_id >= 0 && ntf_id < (int)(NTF_NAME_COUNT - 1)) {
                    /* Resolved from path_config */
                    const ntf_filter_t *f = ntf_get_filter((ntf_filter_id_t)ntf_id, fs_out);
                    if (f) ntf_name = f->name;
                }
                if (!ntf_name) {
                    /* Auto — resolve via auto-select */
                    const ntf_filter_t *f = (sdm_mode == SDM_MODE_PRECORR)
                        ? ntf_auto_select_precorr(fs_out)
                        : ntf_auto_select(fs_out);
                    if (f) ntf_name = f->name;
                }
                info << "\nNTF: " << (ntf_name ? ntf_name : "Auto");
            }

            /* Show effective FIR gain (global setting) */
            int eff_db = (m_cfg.fir_gain_db == FIR_GAIN_AUTO) ? FIR_GAIN_DEFAULT : (int)m_cfg.fir_gain_db;
            float eff_gain = fir_gain_db_to_linear(m_cfg.fir_gain_db);
            info << "\nFIR gain: " << eff_db << " dB (" << pfc::format_float(eff_gain, 0, 3) << ")";
            if (pi_trellis.state_limit > 0.0)
                info << ", state limiter: " << pfc::format_float(pi_trellis.state_limit, 0, 1);

            /* ML filter status */
            bool ml_active = (m_cfg.rate_ml[rowToIdx(row)] >= 0) ? (m_cfg.rate_ml[rowToIdx(row)] != 0) : m_cfg.ml_enabled;
            info << "\nML filter: " << (ml_active ? "On" : "Off");

            /* Pre-SDM filter mode */
            {
                int fir = m_cfg.rate_fir_mode[rowToIdx(row)];
                const char *fir_name;
                if (fir == FIR_MODE_BOXCAR)
                    fir_name = "Boxcar";
                else if (fir == FIR_MODE_FIR)
                    fir_name = "FIR Lowpass";
                else
                    fir_name = (sdm_mode == SDM_MODE_TRELLIS) ? "FIR Lowpass (auto)" : "Boxcar (auto)";
                info << "\nPre-SDM: " << fir_name;
            }

            /* FIR precision */
            {
                int prec = m_cfg.rate_fir_prec[rowToIdx(row)];
                const char *prec_name = (prec == FIR_PREC_FP32) ? "FP32" :
                                        (prec == FIR_PREC_FP64) ? "FP64" : "FP64 (auto)";
                info << "\nFIR precision: " << prec_name;
            }

            /* GPU FIR status */
            {
                bool gpu_fir_active;
                if (m_cfg.rate_gpu[rowToIdx(row)] == 0)
                    gpu_fir_active = false;
                else if (m_cfg.rate_gpu[rowToIdx(row)] == 1)
                    gpu_fir_active = true;
                else
                    gpu_fir_active = m_cfg.gpu_enabled;

                info << "\nGPU FIR: " << (gpu_fir_active ? "On" : "Off");

                /* Parallel SDM mode (resolved) */
                {
                    int par = m_cfg.rate_parallel[idx];
                    if (sdm_mode == SDM_MODE_TRELLIS) {
                        if (par == TRELLIS_PAR_SEQUENTIAL) {
                            info << "\nTrellis SDM: Sequential";
                        } else if (par >= TRELLIS_PAR_PAR2) {
                            char seg_buf[32];
                            sprintf_s(seg_buf, sizeof(seg_buf), "Parallel (%d segments)", par);
                            info << "\nTrellis SDM: " << seg_buf;
                        } else {
                            /* Auto */
                            int auto_segs = (fs_out >= DSD_RATE_512) ? 4 :
                                            (fs_out > DSD_RATE_64) ? 2 : 1;
                            char seg_buf[32];
                            if (auto_segs > 1)
                                sprintf_s(seg_buf, sizeof(seg_buf), "Parallel (auto %d seg)", auto_segs);
                            else
                                sprintf_s(seg_buf, sizeof(seg_buf), "Sequential (auto)");
                            info << "\nTrellis SDM: " << seg_buf;
                        }
                    }
                }
            }
        }

        /* Append cached SINAD result if available */
        {
            char cached[128];
            if (LoadCachedSinad(row, pi.ntf_filter, pi.cands, pi.depth, pi.lat,
                                cached, sizeof(cached)))
                info << "\n" << cached;
        }

        ::uSetDlgItemText(*this, IDC_STATIC_PATH_INFO, info);
    }

    /* ─── SINAD JSON cache ─── */

    void GetSinadJsonPath(char *path, size_t sz) {
        /* Same directory as the DLL */
        char dll_path[MAX_PATH];
        GetModuleFileNameA(core_api::get_my_instance(), dll_path, MAX_PATH);
        char *last_sep = strrchr(dll_path, '\\');
        if (last_sep) *(last_sep + 1) = '\0';
        snprintf(path, sz, "%sfoo_dsd_trellis_sinad.json", dll_path);
    }

#pragma warning(push)
#pragma warning(disable: 4996) /* fopen/sscanf deprecation */
    void SaveSinadResult(int row, int ntf_id, int cands, int depth, int lat,
                          int fir_mode, const sinad_result_t *r) {
        SYSTEMTIME st;
        GetLocalTime(&st);
        char ts[32];
        snprintf(ts, sizeof(ts), "%04d-%02d-%02d %02d:%02d:%02d",
                 st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);

        char path[MAX_PATH];
        GetSinadJsonPath(path, sizeof(path));

        /* Read existing file */
        char *existing = NULL;
        size_t existing_len = 0;
        {
            FILE *f = fopen(path, "rb");
            if (f) {
                fseek(f, 0, SEEK_END);
                existing_len = (size_t)ftell(f);
                fseek(f, 0, SEEK_SET);
                existing = (char *)malloc(existing_len + 1);
                if (existing) {
                    fread(existing, 1, existing_len, f);
                    existing[existing_len] = '\0';
                }
                fclose(f);
            }
        }

        /* Build new entry */
        char entry[512];
        snprintf(entry, sizeof(entry),
            "{\"row\":%d,\"ntf\":%d,\"nc\":%d,\"depth\":%d,\"lat\":%d,"
            "\"fir\":%d,\"theo\":%.1f,\"awtd\":%.1f,"
            "\"mt\":%.1f,\"nmod\":%.1f,\"nmr\":%.1f,\"ts\":\"%s\"}",
            row, ntf_id, cands, depth, lat, fir_mode,
            r->sinad_theoretical, r->sinad_awtd_theo,
            r->multitone_sinad_db, r->noise_mod_db, r->nmr_db, ts);

        /* Write file: simple array of entries */
        FILE *f = fopen(path, "w");
        if (f) {
            fprintf(f, "[\n");
            /* Copy existing entries (skip matching row+params) */
            if (existing) {
                char *p = existing;
                while ((p = strstr(p, "{\"row\":")) != NULL) {
                    char *end = strchr(p, '}');
                    if (!end) break;
                    end++;
                    /* Check if this entry matches our row */
                    int erow = -1;
                    sscanf(p, "{\"row\":%d", &erow);
                    if (erow != row) {
                        fprintf(f, "  %.*s,\n", (int)(end - p), p);
                    }
                    p = end;
                }
            }
            fprintf(f, "  %s\n", entry);
            fprintf(f, "]\n");
            fclose(f);
        }
        free(existing);
    }

    bool LoadCachedSinad(int row, int ntf_id, int cands, int depth, int lat,
                          char *out, size_t out_sz) {
        (void)ntf_id; (void)cands; (void)depth; (void)lat;
        char path[MAX_PATH];
        GetSinadJsonPath(path, sizeof(path));

        FILE *f = fopen(path, "rb");
        if (!f) return false;

        fseek(f, 0, SEEK_END);
        size_t len = (size_t)ftell(f);
        fseek(f, 0, SEEK_SET);
        char *buf = (char *)malloc(len + 1);
        if (!buf) { fclose(f); return false; }
        fread(buf, 1, len, f);
        buf[len] = '\0';
        fclose(f);

        /* Scan for matching entry */
        bool found = false;
        char *p = buf;
        while ((p = strstr(p, "{\"row\":")) != NULL) {
            int erow, entf, enc, edepth, elat, efir;
            char ets[32] = "";
            if (sscanf(p, "{\"row\":%d,\"ntf\":%d,\"nc\":%d,\"depth\":%d,"
                       "\"lat\":%d,\"fir\":%d",
                       &erow, &entf, &enc, &edepth, &elat, &efir) == 6) {
                if (erow == row) {
                    /* Extract all metric fields */
                    double etheo = -999.0, eawtd = -999.0, emt = -999.0;
                    double enmod = -999.0, enmr = -999.0;
                    char *fp;
                    fp = strstr(p, "\"theo\":"); if (fp) sscanf(fp, "\"theo\":%lf", &etheo);
                    fp = strstr(p, "\"awtd\":"); if (fp) sscanf(fp, "\"awtd\":%lf", &eawtd);
                    fp = strstr(p, "\"mt\":");   if (fp) sscanf(fp, "\"mt\":%lf", &emt);
                    fp = strstr(p, "\"nmod\":"); if (fp) sscanf(fp, "\"nmod\":%lf", &enmod);
                    fp = strstr(p, "\"nmr\":");  if (fp) sscanf(fp, "\"nmr\":%lf", &enmr);
                    /* Extract timestamp */
                    char *ts_start = strstr(p, "\"ts\":\"");
                    if (ts_start) {
                        ts_start += 6;
                        char *ts_end = strchr(ts_start, '"');
                        if (ts_end) {
                            size_t ts_len = (size_t)(ts_end - ts_start);
                            if (ts_len < sizeof(ets))
                                memcpy(ets, ts_start, ts_len);
                        }
                    }
                    if (etheo > -900.0) {
                        const char *nmr_label =
                            (enmr <= -30.0) ? "Transparent" :
                            (enmr <= -20.0) ? "Excellent" :
                            (enmr <= -10.0) ? "Good" :
                            (enmr <=   0.0) ? "Fair" : "Poor";
                        snprintf(out, out_sz,
                                 "SINAD: %.1f dB (A-wtd: %.1f)\n"
                                 "Multitone: %.1f dB | NMod: %.1f dB\n"
                                 "NMR: %.1f dB (%s)\n"
                                 "(%s)",
                                 etheo, eawtd, emt, enmod,
                                 enmr, nmr_label, ets);
                    }
                    else
                        snprintf(out, out_sz, "Quality test cached (%s)", ets);
                    found = true;
                }
            }
            p++;
        }
        free(buf);
        return found;
    }
#pragma warning(pop)

    /* ─── Test Quality button ─── */

    void OnTestSinad(UINT, int, CWindow) {
        int row = m_listRate.GetNextItem(-1, LVNI_SELECTED);
        if (row < 0 || row >= RATE_MAP_COUNT) {
            ::uSetDlgItemText(*this, IDC_STATIC_PATH_INFO,
                "Select a rate mapping first.");
            return;
        }
        int idx = rowToIdx(row);
        uint8_t out_idx = m_cfg.rate_map[idx];
        if (out_idx == RATE_OUT_BYPASS) return;

        uint32_t fs_in = rate_idx_to_hz(idx);
        uint32_t fs_out = rate_out_to_hz(out_idx);
        if (fs_out == 0) return;

        bool out_is_dsd_val = rate_out_is_dsd(out_idx);
        bool in_is_dsd = rate_idx_is_dsd(idx);

        /* Pause playback */
        {
            static_api_ptr_t<playback_control> pc;
            if (pc->is_playing() && !pc->is_paused())
                pc->pause(true);
        }

        /* Show measuring... */
        ::uSetDlgItemText(*this, IDC_STATIC_PATH_INFO, "Measuring quality...");
        ::EnableWindow(GetDlgItem(IDC_BTN_TEST_SINAD), FALSE);
        UpdateWindow();

        if (out_is_dsd_val) {
            /* DSD output: full quality measurement (SINAD, A-wtd, multitone, NMod, NMR) */
            int8_t ntf_val = m_cfg.rate_ntf[idx];
            int sdm_mode = (m_cfg.rate_sdm[idx] >= 0) ? (int)m_cfg.rate_sdm[idx] : m_cfg.sdm_mode;
            engine_path_info_t pi;
            engine_get_path_info(fs_in, fs_out, (int)ntf_val, sdm_mode, &m_cfg, &pi);

            sinad_result_t result;

            if (sdm_mode == SDM_MODE_PRECORR) {
                /* PreCorr: use dedicated measurement (precorr_process_block) */
                sinad_measure_precorr(fs_out, &result);
            } else {
                /* Trellis: standard measurement with path_table params */
                int fir_mode_raw = m_cfg.rate_fir_mode[idx];
                int use_fir = (fir_mode_raw == FIR_MODE_FIR) ? 1 :
                              (fir_mode_raw == FIR_MODE_BOXCAR) ? 0 : 1;
                float fir_gain = fir_gain_db_to_linear(m_cfg.fir_gain_db);

                sinad_measure(fs_out, pi.ntf_filter, pi.cands, pi.depth, pi.lat,
                              use_fir, fir_gain, &result);
            }

            if (result.ok) {
                SaveSinadResult(row, pi.ntf_filter, pi.cands, pi.depth, pi.lat,
                                0, &result);
            }
        } else if (in_is_dsd) {
            /* DSD → PCM: measure FIR decimation SINAD */
            sinad_result_t result;
            sinad_measure_dsd_to_pcm(fs_in, fs_out, &result);
            if (result.ok) {
                SaveSinadResult(row, 0, 0, 0, 0, 0, &result);
            }
        } else {
            /* PCM → PCM: measure resample SINAD */
            sinad_result_t result;
            sinad_measure_pcm_to_pcm(fs_in, fs_out,
                                      m_cfg.resample_engine, m_cfg.soxr_quality,
                                      &result);
            if (result.ok) {
                SaveSinadResult(row, 0, 0, 0, 0, 0, &result);
            }
        }

        ::EnableWindow(GetDlgItem(IDC_BTN_TEST_SINAD), TRUE);

        /* Refresh Path Info (will include cached result for DSD paths) */
        UpdatePathInfo(row);
    }

    /* Set Auto text (plain "Auto") for any Auto columns in a row. */
    void RefreshAutoText(int row) {
        int idx = rowToIdx(row);
        if (idx < 0) return;
        if (m_cfg.rate_sdm[idx] < 0)
            m_listRate.SetItemText(row, 3, L"Auto");
        if (m_cfg.rate_cands[idx] < 0)
            m_listRate.SetItemText(row, 4, L"Auto");
        if (m_cfg.rate_depth[idx] < 0)
            m_listRate.SetItemText(row, 5, L"Auto");
        if (m_cfg.rate_limiter[idx] < 0)
            m_listRate.SetItemText(row, 6, L"Auto");
        if (m_cfg.rate_ml[idx] < 0)
            m_listRate.SetItemText(row, 7, L"Auto");
        if (m_cfg.rate_gpu[idx] < 0)
            m_listRate.SetItemText(row, 8, L"Auto");
        if (m_cfg.rate_fir_mode[idx] < 0)
            m_listRate.SetItemText(row, 9, L"Auto");
        if (m_cfg.rate_parallel[idx] < 0)
            m_listRate.SetItemText(row, 10, L"Auto");
        if (m_cfg.rate_lat[idx] <= 0)
            m_listRate.SetItemText(row, 11, L"Auto");
        if (m_cfg.rate_fir_prec[idx] < 0)
            m_listRate.SetItemText(row, 12, L"Auto");
    }

    void UpdatePreset() {
        m_updating = true;

        /* Read SDM mode */
        /* SDM mode is per-rate now, global kept at default */

        /* Threads */
        m_cfg.thread_count = (int)GetDlgItemInt(IDC_EDIT_THREADS, NULL, FALSE);

        /* Input format */
        m_cfg.format = CComboBox(GetDlgItem(IDC_COMBO_FORMAT)).GetCurSel();

        /* Debug log */
        m_cfg.debug_log = IsDlgButtonChecked(IDC_CHECK_DEBUG_LOG) == BST_CHECKED;

        /* Anti-pop */
        m_cfg.antipop = IsDlgButtonChecked(IDC_CHECK_ANTIPOP) == BST_CHECKED;

        /* FIR Gain: combo index 0=Auto, 1=0dB, 2=-1dB, ..., 13=-12dB */
        {
            int sel = CComboBox(GetDlgItem(IDC_COMBO_FIR_GAIN)).GetCurSel();
            if (sel <= 0)
                m_cfg.fir_gain_db = FIR_GAIN_AUTO;
            else if (sel == 1)
                m_cfg.fir_gain_db = 0;
            else
                m_cfg.fir_gain_db = (int8_t)(1 - sel);  /* 2→-1, 3→-2, ..., 13→-12 */
        }

        /* ML filter */
        m_cfg.ml_enabled = IsDlgButtonChecked(IDC_CHECK_ML_ENABLED) == BST_CHECKED;
        m_cfg.ml_ep = combo_to_ml_ep(CComboBox(GetDlgItem(IDC_COMBO_ML_EP)).GetCurSel());

        /* GPU compute */
        m_cfg.gpu_enabled = IsDlgButtonChecked(IDC_CHECK_GPU_ENABLED) == BST_CHECKED;
        {
            int sel = CComboBox(GetDlgItem(IDC_COMBO_GPU_BACKEND)).GetCurSel();
            /* Combo: 0=Auto, 1=DirectCompute, 2=CUDA → backend enum: 3,1,2 */
            switch (sel) {
            case 1: m_cfg.gpu_backend = 1; break;  /* GPU_BACKEND_DIRECTX */
            case 2: m_cfg.gpu_backend = 2; break;  /* GPU_BACKEND_CUDA */
            default: m_cfg.gpu_backend = 3; break;  /* GPU_BACKEND_AUTO */
            }
        }

        m_cfg.gpu_sdm_enabled = IsDlgButtonChecked(IDC_CHECK_GPU_SDM) == BST_CHECKED;

        /* PCM encoding */
        {
            int sel = CComboBox(GetDlgItem(IDC_COMBO_PCM_BITS)).GetCurSel();
            static const int8_t bits_map[] = { PCM_BIT_AUTO, PCM_BIT_16, PCM_BIT_24, PCM_BIT_32, PCM_BIT_FLOAT };
            m_cfg.pcm_bit_depth = (sel >= 0 && sel < 5) ? bits_map[sel] : PCM_BIT_AUTO;
        }
        {
            int sel = CComboBox(GetDlgItem(IDC_COMBO_PCM_DITHER)).GetCurSel();
            static const int8_t dith_map[] = { PCM_DITHER_AUTO, PCM_DITHER_NONE, PCM_DITHER_TPDF, PCM_DITHER_SHAPED };
            m_cfg.pcm_dither = (sel >= 0 && sel < 4) ? dith_map[sel] : PCM_DITHER_AUTO;
        }
        {
            int sel = CComboBox(GetDlgItem(IDC_COMBO_RESAMPLE)).GetCurSel();
            static const int8_t rs_map[] = { (int8_t)RESAMPLE_AUTO, (int8_t)RESAMPLE_IPP, (int8_t)RESAMPLE_SOXR };
            m_cfg.resample_engine = (sel >= 0 && sel < 3) ? rs_map[sel] : (int8_t)RESAMPLE_AUTO;
        }
        {
            int sel = CComboBox(GetDlgItem(IDC_COMBO_SOXR_QUALITY)).GetCurSel();
            static const int8_t sq_map[] = { (int8_t)SOXR_QUALITY_MQ, (int8_t)SOXR_QUALITY_HQ, (int8_t)SOXR_QUALITY_VHQ };
            m_cfg.soxr_quality = (sel >= 0 && sel < 3) ? sq_map[sel] : (int8_t)SOXR_QUALITY_HQ;
        }

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
        /* Do NOT call plugin_set_config here with the raw global config.
         * The global sdm_mode may differ from the per-rate override (e.g.,
         * global=PreCorr but rate_sdm[DSD128]=Trellis). Setting the wrong
         * mode here would cause a wasteful engine reinit on the first chunk,
         * losing audio. The first on_chunk() resolves per-rate overrides
         * and calls plugin_set_config with the correct config. */

        log_set_enabled(m_config.debug_log);

        /* Log version, build hash, and build time */
        trellis_log("DSD Trellis v%s (build %s, %s %s)",
                    BUILD_VERSION, BUILD_GIT_HASH, BUILD_DATE, BUILD_TIME);

        trellis_log("initialized (fir=%s)", fir_ipp_kernel_name());

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
                /* Check if only gpu_sdm_enabled changed (safe toggle,
                 * no engine reinit needed — flag checked each chunk). */
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
            case DSD48_RATE_64: case DSD48_RATE_128: case DSD48_RATE_256: case DSD48_RATE_512:
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

        /* Detect cross-family DSD→PCM: need 2-stage path
         * (FIR decimate to same-family PCM, then polyphase to target) */
        bool dsd_pcm_cross = false;
        uint32_t dsd_pcm_intermediate = 0;
        if (is_dop && out_is_pcm) {
            bool dsd_is_48k = rate_is_48k_family(dsd_rate);
            bool out_is_48k = rate_is_48k_family(out_rate);
            if (dsd_is_48k != out_is_48k) {
                dsd_pcm_cross = true;
                uint32_t base = dsd_is_48k ? 48000u : 44100u;
                dsd_pcm_intermediate = base;
                while (dsd_pcm_intermediate * 2 <= out_rate && dsd_pcm_intermediate * 2 <= base * 8)
                    dsd_pcm_intermediate *= 2;
            }
        }

        /* Apply per-rate overrides to a chunk-local copy of the config.
         * This prevents stale values (cands, lat, depth) from leaking
         * between chunks at different rates. */
        dsd_config_t chunk_cfg = m_config;
        chunk_cfg.fs_in = is_dop ? dsd_rate : pcm_rate;  /* must set before path lookup */

        chunk_cfg.ntf_filter = (int)m_config.rate_ntf[map_idx];
        if (m_config.rate_sdm[map_idx] >= 0)
            chunk_cfg.sdm_mode = (int)m_config.rate_sdm[map_idx];
        if (m_config.rate_ml[map_idx] >= 0)
            chunk_cfg.ml_enabled = (m_config.rate_ml[map_idx] != 0);

        /* Per-rate pre-SDM filter mode: -1=Auto, 0=Boxcar, 1=FIR */
        chunk_cfg.fir_mode = (int)m_config.rate_fir_mode[map_idx];
        chunk_cfg.fir_prec = (int)m_config.rate_fir_prec[map_idx];

        /* Per-rate state limiter: -1=Auto (engine uses path_config), 0=Off, >0=value */
        if (m_config.rate_limiter[map_idx] >= 0)
            chunk_cfg.state_limit = (float)m_config.rate_limiter[map_idx];
        else
            chunk_cfg.state_limit = -1.0f;

        /* Resolve cands/lat/depth from path_config when Auto. */
        {
            uint32_t pi_in = is_dop ? dsd_rate : pcm_rate;
            engine_path_info_t pi;
            int pi_rc = engine_get_path_info(pi_in, out_rate,
                    (int)m_config.rate_ntf[map_idx],
                    chunk_cfg.sdm_mode, &chunk_cfg, &pi);
            if (m_chunk_count < 3)
                trellis_log("path_info: rc=%d fir_only=%d pi.cands=%d pi.depth=%d pi.lat=%d pi_in=%u out=%u ntf=%d sdm=%d",
                            pi_rc, (int)pi.fir_only, pi.cands, pi.depth, pi.lat,
                            pi_in, out_rate, (int)m_config.rate_ntf[map_idx], chunk_cfg.sdm_mode);
            if (pi_rc == 0 && !pi.fir_only) {
                if (m_config.rate_cands[map_idx] > 0)
                    chunk_cfg.trellis_cands = (int)m_config.rate_cands[map_idx];
                else if (pi.cands > 0)
                    chunk_cfg.trellis_cands = pi.cands;

                if (m_config.rate_depth[map_idx] > 0)
                    chunk_cfg.trellis_depth = (int)m_config.rate_depth[map_idx];
                else if (pi.depth > 0)
                    chunk_cfg.trellis_depth = pi.depth;

                if (pi.lat > 0)
                    chunk_cfg.trellis_lat = pi.lat;
            } else {
                if (m_config.rate_cands[map_idx] > 0)
                    chunk_cfg.trellis_cands = (int)m_config.rate_cands[map_idx];
                if (m_config.rate_depth[map_idx] > 0)
                    chunk_cfg.trellis_depth = (int)m_config.rate_depth[map_idx];
            }

            /* Trellis latency: per-rate override or auto-compute from DSD rate.
             * From NTF sweep (2026-03-24, depth=4, nc=2, path_table NTFs):
             *   DSD64:  CLANS6/d=16/lat=32 → 110.7 dB, 0 collapse
             *   DSD128: CLANS6/lat=128 → 121.5 dB, 0 collapse
             *   DSD256: CLANS6/lat=128 → 128.9 dB, 0 collapse
             *   DSD512: lat=32  → 137.6 dB, 0 collapse */
            if (m_config.rate_lat[map_idx] > 0) {
                chunk_cfg.trellis_lat = (int)m_config.rate_lat[map_idx];
            } else {
                uint32_t out_dsd = chunk_cfg.fs_out ? chunk_cfg.fs_out : chunk_cfg.fs_in;
                if (out_dsd >= DSD_RATE_512)
                    chunk_cfg.trellis_lat = 32;
                else if (out_dsd >= DSD_RATE_128)
                    chunk_cfg.trellis_lat = 128;
                else
                    chunk_cfg.trellis_lat = 32;   /* DSD64 (d=16 needs short lat) */
            }
        }

        /* Set output rate for this chunk.
         * For cross-family DSD→PCM, use the intermediate same-family rate
         * so the engine can do power-of-2 FIR decimation. The final polyphase
         * resample to the target rate happens after engine processing. */
        chunk_cfg.fs_out = dsd_pcm_cross ? dsd_pcm_intermediate : out_rate;
        if (!m_logged_processing) {
            trellis_log("rate_map: lookup_rate=%u map_idx=%d out_idx=%u out_rate=%u is_dop=%d gain=%.3f fs_in=%u fs_out=%u",
                        lookup_rate, map_idx, (unsigned)out_idx, out_rate, is_dop, chunk_cfg.gain, chunk_cfg.fs_in, chunk_cfg.fs_out);
            trellis_log("resolved: sdm=%s cands=%d depth=%d lat=%d fir_gain_db=%d fs_in=%u fs_out=%u",
                        chunk_cfg.sdm_mode == SDM_MODE_TRELLIS ? "Trellis" : "PreCorr",
                        chunk_cfg.trellis_cands, chunk_cfg.trellis_depth, chunk_cfg.trellis_lat,
                        (int)chunk_cfg.fir_gain_db, chunk_cfg.fs_in, chunk_cfg.fs_out);
        }
        plugin_set_config(m_state, &chunk_cfg);

        /* Log actual engine config on first few chunks */
        if (m_chunk_count < 3) {
            const dsd_config_t *ecfg = plugin_get_config(m_state);
            if (ecfg)
                trellis_log("engine cfg: sdm=%s cands=%d depth=%d lat=%d fir_gain=%s%.0fdB fs=%u→%u",
                            ecfg->sdm_mode == SDM_MODE_TRELLIS ? "Trellis" :
                            ecfg->sdm_mode == SDM_MODE_PRECORR ? "PreCorr" : "???",
                            ecfg->trellis_cands, ecfg->trellis_depth,
                            ecfg->trellis_lat,
                            ecfg->fir_gain_db == FIR_GAIN_AUTO ? "Auto(" : "",
                            ecfg->fir_gain_db == FIR_GAIN_AUTO ? -3.0 : (double)ecfg->fir_gain_db,
                            ecfg->fs_in, ecfg->fs_out);
        }

        m_channels = (int)channels;
        m_pcm_rate = pcm_rate;

        /* Allocate output buffer */
        /* Output is DoP-encoded. DSD→DSD upsample ratio determines buffer. */
        size_t max_ratio = is_dop ? 8 : (out_rate > pcm_rate ? out_rate / pcm_rate : 1);
        if (max_ratio < 8) max_ratio = 8;
        size_t max_out_frames = pcm_frames * max_ratio;
        /* Output buffer: i24 for DoP (3 bytes/sample), float for PCM (4 bytes/sample).
         * Allocate max of both sizes to share one buffer. */
        size_t out_buf_bytes = max_out_frames * channels * 4; /* 4 >= 3 */
        pfc::array_staticsize_t<uint8_t> out_buf;
        out_buf.set_size_discard(out_buf_bytes);

        LARGE_INTEGER t0, t1, freq;
        QueryPerformanceCounter(&t0);

        size_t out_frames = 0;

        if (is_dop && !dsd_pcm_cross) {
            /* DSD input via DoP — same-family DSD→DSD or DSD→PCM */
            if (!m_logged_processing) {
                unsigned dsd_base = rate_is_48k_family(dsd_rate) ? 48000 : 44100;
                unsigned dsd_mult = dsd_rate / dsd_base;
                if (out_is_pcm)
                    trellis_log("DSD%u%s -> PCM %uHz, %uch",
                                dsd_mult, dsd_base == 48000 ? "/48" : "", out_rate, channels);
                else if (dsd_rate == out_rate)
                    trellis_log("DSD%u%s re-encode, %uch",
                                dsd_mult, dsd_base == 48000 ? "/48" : "", channels);
                else {
                    unsigned out_base = rate_is_48k_family(out_rate) ? 48000 : 44100;
                    unsigned out_mult_v = out_rate / out_base;
                    trellis_log("DSD%u%s -> DSD%u%s, %uch",
                                dsd_mult, dsd_base == 48000 ? "/48" : "",
                                out_mult_v, out_base == 48000 ? "/48" : "", channels);
                }
            }

            out_frames = plugin_process(m_state, in_f32.get_ptr(), out_buf.get_ptr(),
                                        pcm_frames, (int)channels, pcm_rate);

        } else if (is_dop && dsd_pcm_cross) {
            /* Cross-family DSD→PCM: 2-stage pipeline
             * Stage 1: DSD → same-family PCM via FIR decimation (engine)
             *           (chunk_cfg.fs_out = dsd_pcm_intermediate, set above)
             * Stage 2: same-family PCM → target PCM via polyphase resample */
            if (!m_logged_processing) {
                unsigned dsd_base = rate_is_48k_family(dsd_rate) ? 48000 : 44100;
                unsigned dsd_mult = dsd_rate / dsd_base;
                trellis_log("DSD%u%s -> PCM %uHz (via %uHz), %uch",
                            dsd_mult, dsd_base == 48000 ? "/48" : "",
                            out_rate, dsd_pcm_intermediate, channels);
            }

            /* Stage 1: DSD→PCM at intermediate rate (engine configured above) */
            size_t stage1_frames = plugin_process(m_state, in_f32.get_ptr(), out_buf.get_ptr(),
                                                   pcm_frames, (int)channels, pcm_rate);

            if (stage1_frames > 0) {
                /* Stage 2: polyphase resample intermediate → target */
                size_t stage2_max = (size_t)((double)stage1_frames *
                    (double)out_rate / (double)dsd_pcm_intermediate) + 256;
                pfc::array_staticsize_t<float> stage2_buf;
                stage2_buf.set_size_discard(stage2_max * channels);

                out_frames = plugin_process_pcm_to_pcm(m_state,
                    (const float *)out_buf.get_ptr(), stage2_buf.get_ptr(),
                    stage1_frames, (int)channels,
                    dsd_pcm_intermediate, out_rate);

                /* Copy stage2 output back to out_buf for the PCM output path */
                if (out_frames > 0) {
                    size_t total = out_frames * channels;
                    if (total * sizeof(float) <= out_buf_bytes)
                        memcpy(out_buf.get_ptr(), stage2_buf.get_ptr(), total * sizeof(float));
                    else
                        out_frames = 0;
                }
            } else {
                out_frames = 0;
            }

        } else if (out_is_pcm && !is_dop) {
            /* PCM→PCM rate conversion (FIR-only or polyphase) */
            if (!m_logged_processing)
                trellis_log("PCM %uHz -> PCM %uHz, %uch", pcm_rate, out_rate, channels);

            /* PCM→PCM output is float, not DoP i24. Reuse out_buf as float. */
            float *pcm_out = (float *)(void *)out_buf.get_ptr();
            out_frames = plugin_process_pcm_to_pcm(m_state, in_f32.get_ptr(), pcm_out,
                                                    pcm_frames, (int)channels,
                                                    pcm_rate, out_rate);
            if (!m_logged_processing && out_frames > 0)
                trellis_log("PCM->PCM: in=%u @ %uHz, out=%u @ %uHz",
                            (unsigned)pcm_frames, pcm_rate, (unsigned)out_frames, out_rate);

        } else {
            /* PCM input — convert to DSD */
            if (!m_logged_processing) {
                unsigned out_base = rate_is_48k_family(out_rate) ? 48000 : 44100;
                unsigned out_mult_v = out_rate / out_base;
                trellis_log("PCM %uHz -> DSD%u%s, %uch",
                            pcm_rate, out_mult_v, out_base == 48000 ? "/48" : "", channels);
            }
            out_frames = plugin_process_pcm(m_state, in_f32.get_ptr(), out_buf.get_ptr(),
                                             pcm_frames, (int)channels, pcm_rate);
            if (!m_logged_processing)
                trellis_log("PCM->DSD: in=%u frames @ %uHz, out=%u frames @ %uHz (ratio=%u)",
                            (unsigned)pcm_frames, pcm_rate, (unsigned)out_frames,
                            out_rate / 16, out_rate / pcm_rate);
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

        /* Determine output sample rate.
         * DSD output: DoP encoding at out_rate/16. The ASIO+DSD output
         * plugin detects DoP markers and sends native DSD to the driver. */
        uint32_t out_pcm_rate;
        if (out_is_pcm) {
            out_pcm_rate = out_rate;  /* Direct PCM output rate */
        } else {
            out_pcm_rate = out_rate / 16;  /* DoP PCM rate */
        }

        /* Track output format for anti-pop trailing silence */
        m_last_out_pcm_rate = out_pcm_rate;
        m_last_out_is_dop = !out_is_pcm;
        m_last_channels = (int)channels;
        m_last_pcm_rate = pcm_rate;
        m_last_is_dop_input = is_dop;

        size_t total_out = out_frames * channels;

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
                int active_workers = (int)channels * wl_segments;
                if (active_workers > wl_threads) active_workers = wl_threads;
                trellis_log("workload: %d threads (%d active: %uch x %d seg), %d segments/ch",
                            wl_threads, active_workers, channels, wl_segments, wl_segments);
                log_selected_cores();

                double t_unpack, t_fir, t_sdm, t_pack;
                plugin_get_phase_timing(m_state, &t_unpack, &t_fir, &t_sdm, &t_pack);
                trellis_log("phase timing: unpack=%.1fms fir=%.1fms sdm=%.1fms pack=%.1fms",
                            t_unpack, t_fir, t_sdm, t_pack);

                /* Log TUSBAudio DAC status if available */
                if (tusb_available()) {
                    tusb_log_status([](const char *line, void *) {
                        trellis_log("  dac: %s", line);
                    }, nullptr);
                }
            }

            /* Log workload changes (thread count only, not segments) */
            {
                int wl_threads = 0, wl_segments = 0;
                bool wl_changed = false;
                plugin_get_workload(m_state, &wl_threads, &wl_segments, &wl_changed);
                if (wl_changed && m_logged_processing) {
                    trellis_log("workload changed: %d threads, %d segments/ch",
                                wl_threads, wl_segments);
                }
            }

            /* Log CPUSET rebuild (threadpool actually rebuilt with new cores) */
            {
                uint64_t cpuset_mask = 0;
                if (plugin_get_cpuset_change(m_state, &cpuset_mask)) {
                    trellis_log("cpuset rebuild: new mask 0x%016llX",
                                (unsigned long long)cpuset_mask);
                    log_selected_cores();
                }
            }

            /* Log periodic timing + RT stress */
            m_chunk_count++;
            if (g_log_enabled && (m_chunk_count <= 5 || (m_chunk_count % 100) == 0)) {
                double t_unpack, t_fir, t_sdm, t_pack;
                plugin_get_phase_timing(m_state, &t_unpack, &t_fir, &t_sdm, &t_pack);
                trellis_log("chunk #%u: %.1fms total (unpack=%.1f fir=%.1f sdm=%.1f pack=%.1f) / %.1fms audio = %.2fx RT",
                            m_chunk_count, process_ms,
                            t_unpack, t_fir, t_sdm, t_pack,
                            chunk_ms, ratio);

                /* Log worker RT stress with actual core assignment */
                double stressed_ratio = 0.0;
                int stressed_idx = plugin_get_stressed_worker(m_state, &stressed_ratio);
                if (stressed_idx >= 0) {
                    uint32_t worker_cpuset = plugin_get_worker_cpuset(m_state, stressed_idx);
                    trellis_log("  WARNING: worker %d stressed (%.0f%% RT budget) on cpuset %u",
                                stressed_idx, stressed_ratio * 100.0, worker_cpuset);
                }

                /* Log which cores actually processed work (first chunk only) */
                if (m_chunk_count <= 3) {
                    log_active_workers();
                }
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

        /* ─── DSD stream entry + anti-pop lead-in ───
         * Always insert DSD priming silence on first chunk for clean XMOS
         * marker detection (32 consecutive markers needed). Anti-pop adds
         * the rate-switch trick to trigger DAC hardware mute for pop suppression. */
        if (m_antipop_pending && out_frames > 0 && is_dop && out_pcm_rate > 0) {
            m_antipop_pending = false;

            if (m_config.antipop) {
                /* Full anti-pop: rate-switch trick */
                if (m_trail_inserted) {
                    int post_trail_ms = ANTIPOP_LEADIN_MS + 5;
                    trellis_log("anti-pop lead-in: %dms at %u Hz (trail active, skip rate switch)",
                                post_trail_ms, out_pcm_rate);
                    insert_silence_chunk(channels, out_pcm_rate, true, post_trail_ms);
                } else {
                    int half_ms = ANTIPOP_LEADIN_MS / 2;
                    uint32_t alt_rate = (out_pcm_rate == 176400) ? 352800 : 176400;
                    trellis_log("anti-pop lead-in: %dms at %u Hz + %dms at %u Hz",
                                half_ms, alt_rate, half_ms, out_pcm_rate);
                    insert_silence_chunk(channels, alt_rate, true, half_ms);
                    insert_silence_chunk(channels, out_pcm_rate, true, half_ms);
                }
            }
            /* Without anti-pop: no lead-in. Pop at start is an ASIO+DSD
             * output driver limitation (also affects foo_dsd_processor). */
            m_trail_inserted = false;
        }

        if (m_chunk_count < 10) {
            trellis_log("chunk #%u output: %u frames, %u ch, %u Hz (out_rate=%u, is_pcm=%d, is_dop_in=%d)",
                        m_chunk_count, (unsigned)out_frames, channels, out_pcm_rate,
                        out_rate, (int)out_is_pcm, (int)is_dop);

            /* Log SDM candidate diagnostics for Trellis mode */
            if (chunk_cfg.sdm_mode == SDM_MODE_TRELLIS && m_state) {
                const engine_channel_t *eng = plugin_get_channels(m_state);
                if (eng) {
                    const sdm_context_t *sdm = &eng[0].sdm;
                    trellis_log("  SDM diag: conv_fail=%llu collapse=%llu next_drops=%llu/%llu (%.1f%%) num_cands=%d",
                                (unsigned long long)sdm->conv_fail,
                                (unsigned long long)sdm->cands_collapse,
                                (unsigned long long)sdm->next_filter_drops,
                                (unsigned long long)sdm->total_children,
                                sdm->total_children > 0 ? 100.0 * sdm->next_filter_drops / sdm->total_children : 0.0,
                                sdm->num_cands);
                }
            }
        }

        if (out_is_pcm) {
            /* PCM output: apply bit depth conversion and optional dither.
             * Resolve per-rate override or global setting. */
            int8_t bits_cfg = chunk_cfg.rate_pcm_bits[map_idx];
            if (bits_cfg == PCM_BIT_AUTO) bits_cfg = chunk_cfg.pcm_bit_depth;
            int8_t dith_cfg = chunk_cfg.rate_pcm_dither[map_idx];
            if (dith_cfg == PCM_DITHER_AUTO) dith_cfg = chunk_cfg.pcm_dither;

            const float *pcm_f = (const float *)out_buf.get_ptr();

            if (bits_cfg == PCM_BIT_16 || bits_cfg == PCM_BIT_24 || bits_cfg == PCM_BIT_32) {
                int bit_depth = (bits_cfg == PCM_BIT_16) ? 16 :
                                (bits_cfg == PCM_BIT_32) ? 32 : 24;
                /* Quantize float to fixed-point with optional dither */
                double scale = (double)((1 << (bit_depth - 1)) - 1);
                double inv_scale = 1.0 / scale;
                pfc::array_staticsize_t<audio_sample> pcm_as;
                pcm_as.set_size_discard(total_out);

                static unsigned dither_seed = 12345;
                double dither_err = 0.0;  /* for noise-shaped dither */

                for (size_t i = 0; i < total_out; i++) {
                    double v = (double)pcm_f[i];
                    double dither_val = 0.0;

                    if (dith_cfg == PCM_DITHER_TPDF || dith_cfg == PCM_DITHER_AUTO) {
                        /* TPDF dither: ±1 LSB triangular noise */
                        dither_seed = dither_seed * 1664525u + 1013904223u;
                        double r1 = ((double)(dither_seed >> 16) / 32768.0) - 1.0;
                        dither_seed = dither_seed * 1664525u + 1013904223u;
                        double r2 = ((double)(dither_seed >> 16) / 32768.0) - 1.0;
                        dither_val = (r1 + r2) * inv_scale;
                    } else if (dith_cfg == PCM_DITHER_SHAPED) {
                        /* Noise-shaped: TPDF + first-order error feedback */
                        dither_seed = dither_seed * 1664525u + 1013904223u;
                        double r1 = ((double)(dither_seed >> 16) / 32768.0) - 1.0;
                        dither_seed = dither_seed * 1664525u + 1013904223u;
                        double r2 = ((double)(dither_seed >> 16) / 32768.0) - 1.0;
                        dither_val = (r1 + r2) * inv_scale - dither_err;
                    }

                    double quantized = floor((v + dither_val) * scale + 0.5) * inv_scale;
                    if (dith_cfg == PCM_DITHER_SHAPED)
                        dither_err = quantized - v;

                    /* Clamp to ±1.0 */
                    if (quantized > 1.0) quantized = 1.0;
                    if (quantized < -1.0) quantized = -1.0;
                    pcm_as[i] = (audio_sample)quantized;
                }
                chunk->set_data(pcm_as.get_ptr(), out_frames, channels, out_pcm_rate);
            } else {
                /* Float output (default / Auto) — no quantization */
                pfc::array_staticsize_t<audio_sample> pcm_as;
                pcm_as.set_size_discard(total_out);
                for (size_t i = 0; i < total_out; i++)
                    pcm_as[i] = (audio_sample)pcm_f[i];
                chunk->set_data(pcm_as.get_ptr(), out_frames, channels, out_pcm_rate);
            }
        } else {
            /* DoP output: 24-bit signed LE, pass directly to output */
            chunk->set_data_fixedpoint_ex(
                out_buf.get_ptr(), total_out * 3,
                out_pcm_rate, channels, 24,
                audio_chunk::FLAG_LITTLE_ENDIAN | audio_chunk::FLAG_SIGNED,
                audio_chunk::g_guess_channel_config(channels));
        }

        /* On-demand audio capture: DoP mode (mode=1) captures packed output.
         * Capture still uses float for file format compatibility. */
        if (g_audio_capture.mode == 1 &&
            (g_audio_capture.state == CAPTURE_RECORDING || capture_check_armed())) {
            /* Convert i24 back to float for capture (capture path only, not hot) */
            pfc::array_staticsize_t<float> cap_f;
            cap_f.set_size_discard(total_out);
            const uint8_t *cap_src = out_buf.get_ptr();
            for (size_t i = 0; i < total_out; i++) {
                int32_t v = (int32_t)cap_src[i*3] | ((int32_t)cap_src[i*3+1] << 8) | ((int32_t)cap_src[i*3+2] << 16);
                if (v & 0x800000) v |= (int32_t)0xFF000000;
                cap_f[i] = (float)((double)v / 8388608.0);
            }
            capture_write(cap_f.get_ptr(), out_frames, channels, out_pcm_rate);
        }

        return true;
    }

    void on_endofplayback(abort_callback & /*abort*/) override {
        if (!m_state || m_channels == 0)
            return;

        /* Drain SDM latency — but NOT when antipop preserves SDM state,
         * because sdm_drain feeds zeros and sets draining=1, corrupting
         * the preserved state for the next play (crash on rapid stop→play). */
        if (!m_config.antipop) {
            size_t max_drain_frames = 2048 / 16;
            pfc::array_staticsize_t<uint8_t> drain_buf;
            drain_buf.set_size_discard(max_drain_frames * (unsigned)m_channels * 3);

            size_t drain_frames = plugin_drain(m_state, drain_buf.get_ptr(),
                                               m_channels);

            if (drain_frames > 0 && m_pcm_rate > 0 && m_last_is_dop_input) {
                uint32_t out_pcm_rate = m_config.fs_out / 16;
                if (out_pcm_rate == 0)
                    out_pcm_rate = m_pcm_rate;

                size_t total = drain_frames * (unsigned)m_channels;
                audio_chunk_impl chunk_out;
                chunk_out.set_data_fixedpoint_ex(
                    drain_buf.get_ptr(), total * 3,
                    out_pcm_rate, (unsigned)m_channels, 24,
                    audio_chunk::FLAG_LITTLE_ENDIAN | audio_chunk::FLAG_SIGNED,
                    audio_chunk::g_guess_channel_config((unsigned)m_channels));
                insert_chunk(chunk_out);
            }
        }

        /* DSD stream exit: always insert trailing DSD silence for clean
         * XMOS/Thesycon USB driver transition out of DSD mode.
         * Without this, the abrupt stream stop can leave the driver in a
         * bad state where DSD128+ produces no audio on next play.
         * Anti-pop adds the rate-switch trick on top for pop suppression. */
        if (m_last_is_dop_input && m_last_out_pcm_rate > 0) {
            if (m_config.antipop) {
                insert_antipop_trail();
            } else {
                /* Minimal trail: DSD silence at current rate for clean exit */
                trellis_log("DSD stream exit: %dms silence at %u Hz",
                            DSD_STREAM_MS, m_last_out_pcm_rate);
                insert_silence_chunk(m_last_channels, m_last_out_pcm_rate,
                                      true, DSD_STREAM_MS);
            }
            m_trail_inserted = true;
        }
    }

    void on_endoftrack(abort_callback & /*abort*/) override {}

    void flush() override {
        plugin_flush(m_state);
        m_logged_passthrough = false;
        m_logged_processing = false;
        m_chunk_count = 0;
        m_antipop_pending = true;
    }

    double get_latency() override {
        return plugin_get_latency(m_state);
    }

    bool need_track_change_mark() override {
        return false;
    }

    void log_selected_cores() {
        uint32_t ids[CPUSET_MAX_CPUS];
        int n = plugin_get_selected_cores(m_state, ids, CPUSET_MAX_CPUS);
        if (n == 0) return;

        const cpu_topology_t *topo = plugin_get_topology(m_state);

        /* Build a compact description: "core LP/T0(perf) ..." */
        char buf[1024];
        int pos = 0;
        pos += snprintf(buf + pos, sizeof(buf) - pos, "selected cores (%d): ", n);

        for (int i = 0; i < n && pos < (int)sizeof(buf) - 40; i++) {
            /* Find this ID in the topology */
            const char *smt_tag = "?";
            int lp = -1;
            int cluster = -1;
            double perf = 0.0;
            double load = 0.0;
            bool parked = false;
            if (topo) {
                for (int j = 0; j < topo->count; j++) {
                    if (topo->entries[j].id == ids[i]) {
                        lp = topo->entries[j].logical_index;
                        smt_tag = topo->entries[j].smt_thread == 0 ? "T0" : "T1";
                        cluster = topo->entries[j].cluster;
                        perf = topo->entries[j].perf_score;
                        load = topo->entries[j].load;
                        parked = topo->entries[j].parked;
                        break;
                    }
                }
            }
            if (i > 0) pos += snprintf(buf + pos, sizeof(buf) - pos, ", ");
            pos += snprintf(buf + pos, sizeof(buf) - pos,
                           "LP%d/%s/C%d(perf=%.0f%% load=%.0f%%%s)",
                           lp, smt_tag, cluster, perf * 100, load * 100,
                           parked ? " parked" : "");
        }
        trellis_log("%s", buf);
    }

    void log_active_workers() {
        if (!m_state) return;
        uint32_t ids[CPUSET_MAX_CPUS];
        int n = plugin_get_selected_cores(m_state, ids, CPUSET_MAX_CPUS);
        if (n == 0) return;

        const cpu_topology_t *topo = plugin_get_topology(m_state);

        /* Report thread count vs core count */
        char buf[512];
        int pos = 0;
        int wl_threads = 0, wl_segments = 0;
        bool wl_changed = false;
        plugin_get_workload(m_state, &wl_threads, &wl_segments, &wl_changed);
        int active_count = (int)m_channels * wl_segments;
        if (active_count > wl_threads) active_count = wl_threads;

        pos = 0;
        pos += snprintf(buf + pos, sizeof(buf) - pos,
                        "worker assignment: %d active of %d threads on cores: ", active_count, n);
        /* The first active_count cores in selection order get the work
         * (semaphore wakes threads 0..N-1 in order) */
        for (int i = 0; i < active_count && i < n && pos < (int)sizeof(buf) - 20; i++) {
            int lp = -1;
            if (topo) {
                for (int j = 0; j < topo->count; j++) {
                    if (topo->entries[j].id == ids[i]) {
                        lp = topo->entries[j].logical_index;
                        break;
                    }
                }
            }
            if (i > 0) pos += snprintf(buf + pos, sizeof(buf) - pos, ", ");
            pos += snprintf(buf + pos, sizeof(buf) - pos, "LP%d", lp);
        }
        trellis_log("%s", buf);
    }

private:
    static const int ANTIPOP_MS = 150;       /* ms of DSD silence for trail */
    static const int ANTIPOP_LEADIN_MS = 70; /* ms of DSD silence for lead-in */
    static const int DSD_STREAM_MS = 50;     /* ms of DSD silence for clean stream entry/exit */

    /* Insert a silence chunk directly (no engine needed).
     * For DoP: generates DSD idle pattern via dop_pack.
     * For PCM: zeros. */
    void insert_silence_chunk(int num_channels, uint32_t out_pcm_rate,
                               bool is_dop_output, int ms) {
        if (out_pcm_rate == 0 || num_channels == 0)
            return;

        size_t sil_frames = (size_t)(out_pcm_rate * ms / 1000);
        if (sil_frames == 0)
            return;

        size_t sil_total = sil_frames * (size_t)num_channels;

        if (is_dop_output) {
            /* DoP silence: DSD idle pattern 0x69 (01101001) → dop_pack_i24.
             * 0x69 is the standard DSD silence byte — a toggling pattern
             * with zero DC content. */
            static const float dsd_0x69[8] = {
                -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f
            };
            size_t dsd_per_frame = sil_frames * DOP_BITS_PER_FRAME;
            pfc::array_staticsize_t<float> dsd_idle;
            dsd_idle.set_size_discard(dsd_per_frame);
            for (size_t i = 0; i < dsd_per_frame; i++)
                dsd_idle[i] = dsd_0x69[i & 7];

            pfc::array_staticsize_t<uint8_t> dop_i24;
            dop_i24.set_size_discard(sil_frames * 3);
            dop_pack_i24(dsd_idle.get_ptr(), dop_i24.get_ptr(), dsd_per_frame, 0);

            /* Interleave: replicate mono i24 to all channels */
            pfc::array_staticsize_t<uint8_t> sil_i24;
            sil_i24.set_size_discard(sil_total * 3);
            for (size_t f = 0; f < sil_frames; f++)
                for (int ch = 0; ch < num_channels; ch++) {
                    size_t dst = (f * num_channels + ch) * 3;
                    size_t src = f * 3;
                    sil_i24[dst]     = dop_i24[src];
                    sil_i24[dst + 1] = dop_i24[src + 1];
                    sil_i24[dst + 2] = dop_i24[src + 2];
                }

            audio_chunk_impl chunk_out;
            chunk_out.set_data_fixedpoint_ex(
                sil_i24.get_ptr(), sil_total * 3,
                out_pcm_rate, (unsigned)num_channels, 24,
                audio_chunk::FLAG_LITTLE_ENDIAN | audio_chunk::FLAG_SIGNED,
                audio_chunk::g_guess_channel_config((unsigned)num_channels));
            insert_chunk(chunk_out);
        } else {
            pfc::array_staticsize_t<audio_sample> sil_as;
            sil_as.set_size_discard(sil_total);
            memset(sil_as.get_ptr(), 0, sil_total * sizeof(audio_sample));

            audio_chunk_impl chunk_out;
            chunk_out.set_data(sil_as.get_ptr(), sil_frames,
                               (unsigned)num_channels, out_pcm_rate);
            insert_chunk(chunk_out);
        }
    }

    /* Insert trailing silence to flush ASIO buffer before stop.
     * Two phases: (1) DSD silence at current rate to settle analog output,
     * (2) DSD silence at a different rate to trigger DAC hardware mute
     * via rate-change detection — DAC mutes before stream actually stops. */
    void insert_antipop_trail() {
        if (m_last_out_pcm_rate == 0 || m_last_channels == 0)
            return;

        int half_ms = ANTIPOP_MS / 2;

        /* Phase 1: silence at current DSD rate */
        trellis_log("anti-pop trail: %dms at %u Hz + rate switch", half_ms, m_last_out_pcm_rate);
        insert_silence_chunk(m_last_channels, m_last_out_pcm_rate,
                              m_last_out_is_dop, half_ms);

        /* Phase 2: silence at a different DoP rate to trigger DAC mute. */
        if (m_last_out_is_dop) {
            uint32_t alt_rate = (m_last_out_pcm_rate == 176400) ? 352800 : 176400;
            insert_silence_chunk(m_last_channels, alt_rate, true, half_ms);
        }
    }

    dsd_config_t     m_config;
    plugin_state_t  *m_state;
    httpapi_t       *m_httpapi;
    int              m_channels;
    unsigned         m_pcm_rate;
    bool             m_logged_passthrough = false;
    bool             m_thread_pinned = false;
    bool             m_logged_processing = false;
    unsigned         m_chunk_count = 0;

    /* Anti-pop / deferred output state */
    bool             m_antipop_pending = true;         /* insert lead-in on first chunk */
    bool             m_trail_inserted = false;         /* trailing silence was just inserted */
    uint32_t         m_last_out_pcm_rate = 0;          /* last output PCM rate */
    uint32_t         m_last_pcm_rate = 0;              /* last input PCM rate */
    int              m_last_channels = 0;              /* last channel count */
    bool             m_last_out_is_dop = false;        /* last output was DoP */
    bool             m_last_is_dop_input = false;      /* last input was DoP */
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
