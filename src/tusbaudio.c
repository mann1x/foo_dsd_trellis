/*
 * foo_dsd_trellis — TUSBAudio API runtime integration
 *
 * Discovers and queries Thesycon TUSBAudio-based USB DACs at runtime.
 * No compile-time dependency on the proprietary Thesycon SDK.
 */

#include "../include/tusbaudio.h"
#include <windows.h>
#include <stdio.h>
#include <string.h>

/* ─── TUSBAudio API types ─── */

typedef unsigned int TUsbAudioStatus;
typedef unsigned int TUsbAudioHandle;

#define TUSB_MAX_STRDESC 200
#define TUSB_STATUS_SUCCESS 0

typedef struct {
    unsigned int usbVendorId;
    unsigned int usbProductId;
    unsigned int usbRevisionId;
    wchar_t serialNumberString[TUSB_MAX_STRDESC];
    wchar_t manufacturerString[TUSB_MAX_STRDESC];
    wchar_t productString[TUSB_MAX_STRDESC];
    unsigned int flags;
} TUsbAudioDeviceProperties;

typedef struct {
    unsigned int formatId;
    unsigned int bitsPerSample;
    unsigned int numberOfChannels;
    wchar_t formatNameString[TUSB_MAX_STRDESC];
} TUsbAudioStreamFormat;

typedef struct {
    unsigned int apiVersionMajor;
    unsigned int apiVersionMinor;
    unsigned int driverVersionMajor;
    unsigned int driverVersionMinor;
    unsigned int driverVersionSub;
    unsigned int flags;
} TUsbAudioDriverInfo;

/* ─── Function pointer typedefs ─── */

typedef unsigned int (*fn_GetApiVersion)(void);
typedef TUsbAudioStatus (*fn_EnumerateDevices)(void);
typedef unsigned int (*fn_GetDeviceCount)(void);
typedef TUsbAudioStatus (*fn_OpenDeviceByIndex)(unsigned int, TUsbAudioHandle *);
typedef TUsbAudioStatus (*fn_CloseDevice)(TUsbAudioHandle);
typedef TUsbAudioStatus (*fn_GetDeviceProperties)(TUsbAudioHandle, TUsbAudioDeviceProperties *);
typedef TUsbAudioStatus (*fn_GetDriverInfo)(TUsbAudioDriverInfo *);
typedef TUsbAudioStatus (*fn_GetCurrentSampleRate)(TUsbAudioHandle, unsigned int *);
typedef TUsbAudioStatus (*fn_GetSupportedSampleRates)(TUsbAudioHandle, unsigned int, unsigned int *, unsigned int *);
typedef TUsbAudioStatus (*fn_GetCurrentStreamFormat)(TUsbAudioHandle, unsigned int, unsigned int *);
typedef TUsbAudioStatus (*fn_GetSupportedStreamFormats)(TUsbAudioHandle, unsigned int, unsigned int, TUsbAudioStreamFormat *, unsigned int *);
typedef TUsbAudioStatus (*fn_GetDeviceStreamingMode)(TUsbAudioHandle, unsigned int *);
typedef const char *(*fn_StatusCodeStringA)(TUsbAudioStatus);

/* ─── Runtime state ─── */

static struct {
    HMODULE dll;
    TUsbAudioHandle device;
    bool opened;
    char dll_path[260];

    /* Resolved function pointers */
    fn_GetApiVersion         GetApiVersion;
    fn_EnumerateDevices      EnumerateDevices;
    fn_GetDeviceCount        GetDeviceCount;
    fn_OpenDeviceByIndex     OpenDeviceByIndex;
    fn_CloseDevice           CloseDevice;
    fn_GetDeviceProperties   GetDeviceProperties;
    fn_GetDriverInfo         GetDriverInfo;
    fn_GetCurrentSampleRate  GetCurrentSampleRate;
    fn_GetSupportedSampleRates GetSupportedSampleRates;
    fn_GetCurrentStreamFormat GetCurrentStreamFormat;
    fn_GetSupportedStreamFormats GetSupportedStreamFormats;
    fn_GetDeviceStreamingMode GetDeviceStreamingMode;
    fn_StatusCodeStringA     StatusCodeStringA;
} g_tusb;

/* ─── DLL discovery ─── */

/* Known TUSBAudio driver install locations.
 * Each manufacturer brands the DLL differently but the API is identical. */
static const wchar_t *g_search_dirs[] = {
    L"C:\\Program Files\\Fosi Audio\\USB Audio Device Driver\\x64",
    L"C:\\Program Files\\Topping\\USB Audio Device Driver\\x64",
    L"C:\\Program Files\\SMSL\\USB Audio Device Driver\\x64",
    L"C:\\Program Files\\Gustard\\USB Audio Device Driver\\x64",
    L"C:\\Program Files\\XMOS\\USB Audio Device Driver\\x64",
    L"C:\\Program Files\\Singxer\\USB Audio Device Driver\\x64",
    L"C:\\Program Files\\Matrix Audio\\USB Audio Device Driver\\x64",
    L"C:\\Program Files\\S.M.S.L\\USB Audio Device Driver\\x64",
    L"C:\\Program Files\\Aune\\USB Audio Device Driver\\x64",
    L"C:\\Program Files\\Denafrips\\USB Audio Device Driver\\x64",
    NULL
};

/* Try to find a tusbaudioapi DLL in the given directory */
static HMODULE try_load_dir(const wchar_t *dir, char *out_path, size_t path_size) {
    WIN32_FIND_DATAW fd;
    wchar_t pattern[MAX_PATH];
    _snwprintf(pattern, MAX_PATH, L"%s\\*api_x64.dll", dir);

    HANDLE h = FindFirstFileW(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) {
        /* Also try *api.dll for 32-bit naming */
        _snwprintf(pattern, MAX_PATH, L"%s\\*api.dll", dir);
        h = FindFirstFileW(pattern, &fd);
    }
    if (h == INVALID_HANDLE_VALUE)
        return NULL;

    wchar_t full_path[MAX_PATH];
    _snwprintf(full_path, MAX_PATH, L"%s\\%s", dir, fd.cFileName);
    FindClose(h);

    HMODULE dll = LoadLibraryW(full_path);
    if (dll && out_path) {
        WideCharToMultiByte(CP_UTF8, 0, full_path, -1,
                           out_path, (int)path_size, NULL, NULL);
    }
    return dll;
}

/* Scan registry for TUSBAudio installations */
static HMODULE try_load_registry(char *out_path, size_t path_size) {
    /* Look for Thesycon driver entries in the device installer registry */
    const wchar_t *key_path = L"SOFTWARE\\Thesycon\\TUSBAudio";
    HKEY hkey;
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, key_path, 0, KEY_READ, &hkey) == ERROR_SUCCESS) {
        wchar_t install_dir[MAX_PATH];
        DWORD size = sizeof(install_dir);
        if (RegQueryValueExW(hkey, L"InstallDir", NULL, NULL,
                            (LPBYTE)install_dir, &size) == ERROR_SUCCESS) {
            RegCloseKey(hkey);
            return try_load_dir(install_dir, out_path, path_size);
        }
        RegCloseKey(hkey);
    }
    return NULL;
}

static bool resolve_functions(void) {
    HMODULE d = g_tusb.dll;
    if (!d) return false;

#define RESOLVE(name) do { \
    g_tusb.name = (fn_##name)GetProcAddress(d, "TUSBAUDIO_" #name); \
    if (!g_tusb.name) return false; \
} while (0)

    RESOLVE(GetApiVersion);
    RESOLVE(EnumerateDevices);
    RESOLVE(GetDeviceCount);
    RESOLVE(OpenDeviceByIndex);
    RESOLVE(CloseDevice);
    RESOLVE(GetDeviceProperties);
    RESOLVE(GetCurrentSampleRate);
    RESOLVE(GetSupportedSampleRates);
    RESOLVE(GetCurrentStreamFormat);
    RESOLVE(GetSupportedStreamFormats);
    RESOLVE(GetDeviceStreamingMode);

#undef RESOLVE

    /* Optional functions (may not exist in older API versions) */
    g_tusb.GetDriverInfo = (fn_GetDriverInfo)GetProcAddress(d, "TUSBAUDIO_GetDriverInfo");
    g_tusb.StatusCodeStringA = (fn_StatusCodeStringA)GetProcAddress(d, "TUSBAUDIO_StatusCodeStringA");

    return true;
}

/* ─── Public API ─── */

bool tusb_available(void) {
    if (g_tusb.dll)
        return true;

    /* Try known directories */
    for (int i = 0; g_search_dirs[i]; i++) {
        HMODULE dll = try_load_dir(g_search_dirs[i],
                                    g_tusb.dll_path, sizeof(g_tusb.dll_path));
        if (dll) {
            g_tusb.dll = dll;
            if (resolve_functions())
                return true;
            FreeLibrary(dll);
            g_tusb.dll = NULL;
        }
    }

    /* Try registry */
    HMODULE dll = try_load_registry(g_tusb.dll_path, sizeof(g_tusb.dll_path));
    if (dll) {
        g_tusb.dll = dll;
        if (resolve_functions())
            return true;
        FreeLibrary(dll);
        g_tusb.dll = NULL;
    }

    return false;
}

bool tusb_open(void) {
    if (g_tusb.opened)
        return true;
    if (!tusb_available())
        return false;

    TUsbAudioStatus st = g_tusb.EnumerateDevices();
    if (st != TUSB_STATUS_SUCCESS)
        return false;

    unsigned int count = g_tusb.GetDeviceCount();
    if (count == 0)
        return false;

    /* Open first device */
    st = g_tusb.OpenDeviceByIndex(0, &g_tusb.device);
    if (st != TUSB_STATUS_SUCCESS)
        return false;

    g_tusb.opened = true;
    return true;
}

void tusb_close(void) {
    if (g_tusb.opened) {
        g_tusb.CloseDevice(g_tusb.device);
        g_tusb.opened = false;
    }
    if (g_tusb.dll) {
        FreeLibrary(g_tusb.dll);
        g_tusb.dll = NULL;
    }
    memset(&g_tusb, 0, sizeof(g_tusb));
}

static bool is_dsd_rate(uint32_t rate) {
    return rate == 2822400 || rate == 5644800 || rate == 11289600 ||
           rate == 22579200 || rate == 45158400;
}

bool tusb_query_status(tusb_status_t *status) {
    memset(status, 0, sizeof(*status));

    if (!g_tusb.opened) {
        if (!tusb_open())
            return false;
    }
    status->available = true;
    strncpy(status->dll_path, g_tusb.dll_path, sizeof(status->dll_path) - 1);

    /* Device properties */
    TUsbAudioDeviceProperties props;
    memset(&props, 0, sizeof(props));
    if (g_tusb.GetDeviceProperties(g_tusb.device, &props) == TUSB_STATUS_SUCCESS) {
        wcsncpy(status->product, props.productString, TUSB_MAX_STR - 1);
        wcsncpy(status->manufacturer, props.manufacturerString, TUSB_MAX_STR - 1);
        status->usb_vid = props.usbVendorId;
        status->usb_pid = props.usbProductId;
    }

    /* Current sample rate */
    g_tusb.GetCurrentSampleRate(g_tusb.device, &status->current_sample_rate);

    /* Streaming mode */
    g_tusb.GetDeviceStreamingMode(g_tusb.device, &status->streaming_mode);

    /* Current format */
    g_tusb.GetCurrentStreamFormat(g_tusb.device, 0, &status->current_format_id);

    /* Supported sample rates */
    unsigned int rate_count = 0;
    g_tusb.GetSupportedSampleRates(g_tusb.device, 64,
                                    status->supported_rates, &rate_count);
    status->supported_rate_count = (int)rate_count;

    /* Check DSD support from supported rates */
    for (int i = 0; i < status->supported_rate_count; i++) {
        uint32_t r = status->supported_rates[i];
        if (r == 2822400)  status->supports_dsd64 = true;
        if (r == 5644800)  status->supports_dsd128 = true;
        if (r == 11289600) status->supports_dsd256 = true;
        if (r == 22579200) status->supports_dsd512 = true;
        if (r == 45158400) status->supports_dsd1024 = true;
    }

    /* Supported stream formats */
    unsigned int fmt_count = 0;
    TUsbAudioStreamFormat fmts[32];
    memset(fmts, 0, sizeof(fmts));
    if (g_tusb.GetSupportedStreamFormats(g_tusb.device, 0, 32,
                                          fmts, &fmt_count) == TUSB_STATUS_SUCCESS) {
        int n = fmt_count > 32 ? 32 : (int)fmt_count;
        for (int i = 0; i < n; i++) {
            status->formats[i].format_id = fmts[i].formatId;
            status->formats[i].bits_per_sample = fmts[i].bitsPerSample;
            status->formats[i].num_channels = fmts[i].numberOfChannels;
            wcsncpy(status->formats[i].name, fmts[i].formatNameString, TUSB_MAX_STR - 1);

            /* Check for DSD format names or 1-bit samples */
            if (fmts[i].bitsPerSample == 1 ||
                wcsstr(fmts[i].formatNameString, L"DSD") ||
                wcsstr(fmts[i].formatNameString, L"dsd"))
                status->supports_native_dsd = true;
        }
        status->format_count = n;
    }

    /* Also flag native DSD if any DSD rate is supported */
    if (status->supports_dsd64 || status->supports_dsd128 ||
        status->supports_dsd256 || status->supports_dsd512)
        status->supports_native_dsd = true;

    return true;
}

bool tusb_get_sample_rate(uint32_t *rate) {
    if (!g_tusb.opened)
        return false;
    return g_tusb.GetCurrentSampleRate(g_tusb.device, rate) == TUSB_STATUS_SUCCESS;
}

bool tusb_get_streaming_mode(uint32_t *mode) {
    if (!g_tusb.opened)
        return false;
    return g_tusb.GetDeviceStreamingMode(g_tusb.device, mode) == TUSB_STATUS_SUCCESS;
}

void tusb_log_status(tusb_log_fn log_fn, void *ctx) {
    tusb_status_t st;
    char line[512];

    if (!tusb_query_status(&st)) {
        log_fn("TUSBAudio: not available", ctx);
        return;
    }

    snprintf(line, sizeof(line), "TUSBAudio: %S [%S] VID=%04X PID=%04X",
             st.product, st.manufacturer, st.usb_vid, st.usb_pid);
    log_fn(line, ctx);

    snprintf(line, sizeof(line), "  DLL: %s", st.dll_path);
    log_fn(line, ctx);

    snprintf(line, sizeof(line), "  rate: %u Hz, mode: %s, format_id: %u",
             st.current_sample_rate,
             st.streaming_mode == 0 ? "PCM" :
             st.streaming_mode == 1 ? "DSD" : "unknown",
             st.current_format_id);
    log_fn(line, ctx);

    /* DSD capabilities */
    snprintf(line, sizeof(line), "  DSD: %s%s%s%s%s%s",
             st.supports_native_dsd ? "native " : "",
             st.supports_dsd64 ? "DSD64 " : "",
             st.supports_dsd128 ? "DSD128 " : "",
             st.supports_dsd256 ? "DSD256 " : "",
             st.supports_dsd512 ? "DSD512 " : "",
             st.supports_dsd1024 ? "DSD1024 " : "");
    log_fn(line, ctx);

    /* Supported rates */
    char rates_buf[512];
    int pos = 0;
    pos += snprintf(rates_buf + pos, sizeof(rates_buf) - pos, "  rates (%d):", st.supported_rate_count);
    for (int i = 0; i < st.supported_rate_count && pos < (int)sizeof(rates_buf) - 20; i++) {
        uint32_t r = st.supported_rates[i];
        if (is_dsd_rate(r))
            pos += snprintf(rates_buf + pos, sizeof(rates_buf) - pos, " DSD%u", r / 44100);
        else
            pos += snprintf(rates_buf + pos, sizeof(rates_buf) - pos, " %u", r);
    }
    log_fn(rates_buf, ctx);

    /* Stream formats */
    for (int i = 0; i < st.format_count; i++) {
        snprintf(line, sizeof(line), "  fmt[%d]: id=%u %ubit %uch \"%S\"",
                 i, st.formats[i].format_id,
                 st.formats[i].bits_per_sample,
                 st.formats[i].num_channels,
                 st.formats[i].name);
        log_fn(line, ctx);
    }
}
