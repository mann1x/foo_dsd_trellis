/*
 * foo_dsd_trellis — TUSBAudio API integration
 *
 * Runtime-loaded interface to Thesycon TUSBAudio driver API.
 * Works with any XMOS-based USB audio DAC (Fosi, Topping, SMSL,
 * Gustard, etc.) that uses the TUSBAudio driver framework.
 *
 * No compile-time dependency — all functions resolved at runtime
 * via LoadLibraryW + GetProcAddress.
 */

#ifndef TUSBAUDIO_H
#define TUSBAUDIO_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Max string length in TUSBAudio structs */
#define TUSB_MAX_STR 200

/* Device properties */
typedef struct {
    uint32_t usb_vendor_id;
    uint32_t usb_product_id;
    uint32_t usb_revision_id;
    wchar_t  serial[TUSB_MAX_STR];
    wchar_t  manufacturer[TUSB_MAX_STR];
    wchar_t  product[TUSB_MAX_STR];
    uint32_t flags;
} tusb_device_props_t;

/* Stream format descriptor */
typedef struct {
    uint32_t format_id;
    uint32_t bits_per_sample;
    uint32_t num_channels;
    wchar_t  name[TUSB_MAX_STR];
} tusb_stream_format_t;

/* DAC status snapshot */
typedef struct {
    bool     available;           /* TUSBAudio DLL found and device opened */
    char     dll_path[260];       /* Path to the loaded DLL */
    wchar_t  product[TUSB_MAX_STR];
    wchar_t  manufacturer[TUSB_MAX_STR];
    uint32_t usb_vid;
    uint32_t usb_pid;
    uint32_t current_sample_rate; /* Hz */
    uint32_t streaming_mode;      /* 0=PCM, 1=DSD (tentative) */
    uint32_t current_format_id;   /* Active stream format ID */

    /* Supported rates */
    uint32_t supported_rates[64];
    int      supported_rate_count;

    /* Supported formats */
    tusb_stream_format_t formats[32];
    int      format_count;

    /* DSD capability (derived from supported rates/formats) */
    bool     supports_dsd64;
    bool     supports_dsd128;
    bool     supports_dsd256;
    bool     supports_dsd512;
    bool     supports_dsd1024;
    bool     supports_native_dsd;
} tusb_status_t;

/* Check if any TUSBAudio-based DAC is available.
 * Scans common driver install paths for tusbaudioapi DLLs. */
bool tusb_available(void);

/* Open the first available TUSBAudio device.
 * Returns true if a device was found and opened. */
bool tusb_open(void);

/* Close the device and unload the DLL. */
void tusb_close(void);

/* Query current DAC status (sample rate, format, capabilities).
 * Returns false if no device is open. */
bool tusb_query_status(tusb_status_t *status);

/* Query just the current sample rate (lightweight, for real-time use). */
bool tusb_get_sample_rate(uint32_t *rate);

/* Query current streaming mode (0=PCM, 1=DSD). */
bool tusb_get_streaming_mode(uint32_t *mode);

/* Log full device status to the provided callback. */
typedef void (*tusb_log_fn)(const char *line, void *ctx);
void tusb_log_status(tusb_log_fn log_fn, void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* TUSBAUDIO_H */
