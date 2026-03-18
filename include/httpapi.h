/*
 * foo_dsd_trellis — Lightweight HTTP REST API for real-time config control
 *
 * Embeds a single-threaded Winsock2 HTTP server on 127.0.0.1:8881.
 * Audio thread pushes status; HTTP thread serves GET/PUT requests.
 *
 * Endpoints:
 *   GET  /api/config   — current config as JSON
 *   PUT  /api/config   — partial JSON update, triggers reconfigure
 *   GET  /api/status   — runtime status (timing, threads, CPUSET, RT ratio)
 *   POST /api/render   — generate tone, process through SDM, save WAV, return SINAD
 *   POST /api/measure  — read WAV file (raw or DoP), measure SINAD
 */

#ifndef HTTPAPI_H
#define HTTPAPI_H

#include <stdint.h>
#include <stdbool.h>
#include "dsd_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct httpapi httpapi_t;

/* Runtime status snapshot (audio thread → HTTP thread) */
typedef struct {
    bool        playing;
    uint32_t    dsd_rate_in;
    uint32_t    dsd_rate_out;
    int         channels;
    int         threads;
    int         segments_per_ch;
    uint64_t    cpuset_mask;
    int         cpuset_enabled;
    uint32_t    chunk_count;
    double      last_chunk_ms;
    double      last_audio_ms;
    double      rt_ratio;
    double      unpack_ms;
    double      fir_ms;
    double      sdm_ms;
    double      pack_ms;
} httpapi_status_t;

/* Create and start the HTTP server on 127.0.0.1:port.
 * Takes a copy of the initial config. Returns NULL on failure. */
httpapi_t *httpapi_create(uint16_t port, const dsd_config_t *initial_cfg);

/* Stop and destroy the HTTP server. Safe to call with NULL. */
void httpapi_destroy(httpapi_t *api);

/* Push a status snapshot from the audio thread (lock-free write). */
void httpapi_update_status(httpapi_t *api, const httpapi_status_t *status);

/* Push current config from the audio thread (after reconfigure). */
void httpapi_update_config(httpapi_t *api, const dsd_config_t *cfg);

/* Log ring buffer: stores last N log lines in memory for fast API access.
 * Thread-safe: writer (DSP thread) and reader (HTTP thread) use SRWLOCK. */
#define LOG_RING_MAX_LINES  500
#define LOG_RING_MAX_LINE   256

typedef struct {
    char    lines[LOG_RING_MAX_LINES][LOG_RING_MAX_LINE];
    int     head;       /* next write position */
    int     count;      /* total lines stored (max LOG_RING_MAX_LINES) */
    SRWLOCK lock;
} log_ring_t;

/* Global log ring (shared between DSP writer and HTTP reader) */
extern log_ring_t g_log_ring;

/* Write a line to the ring buffer (called from trellis_log) */
void log_ring_write(const char *line);

/* Read last N lines from ring buffer into a buffer. Returns bytes written. */
int log_ring_read(char *buf, int buf_size, int max_lines);

/* Check if the HTTP thread received a config update via PUT.
 * Returns true and fills *cfg if a pending config exists (clears pending flag).
 * Audio thread calls this at the start of each on_chunk. */
bool httpapi_get_pending_config(httpapi_t *api, dsd_config_t *cfg);

/* ─── On-demand audio capture ─── */

/* Capture state: HTTP thread sets ARMED, audio thread writes samples
 * and sets DONE when buffer is full. HTTP thread reads and serves. */
#define CAPTURE_IDLE     0
#define CAPTURE_ARMED    1
#define CAPTURE_RECORDING 2
#define CAPTURE_DONE     3

#define CAPTURE_MAX_SAMPLES (2822400 * 2 * 5)  /* 5 seconds stereo DSD64 */

typedef struct {
    volatile LONG state;        /* CAPTURE_IDLE/ARMED/RECORDING/DONE */
    float        *buf;          /* sample buffer (allocated on arm) */
    size_t        buf_size;     /* allocated samples */
    size_t        written;      /* samples written so far */
    size_t        target;       /* target sample count */
    uint32_t      rate;         /* DSD or PCM sample rate */
    int           channels;     /* channel count */
    int           mode;         /* 0=raw DSD (±1.0), 1=DoP packed */
} audio_capture_t;

extern audio_capture_t g_audio_capture;

/* Called by audio thread: write interleaved samples (DoP mode) */
void capture_write(const float *samples, size_t count, int channels, uint32_t rate);

/* Called by audio thread: write raw DSD per-channel (mode=0).
 * ch_data[ch] = array of ±1.0 floats, count = samples per channel. */
void capture_write_dsd(float **ch_data, size_t count, int channels, uint32_t dsd_rate);

/* Called by audio thread: check if capture is armed and start recording */
static inline bool capture_check_armed(void) {
    return InterlockedCompareExchange(&g_audio_capture.state,
                                      CAPTURE_RECORDING, CAPTURE_ARMED)
           == CAPTURE_ARMED;
}

#ifdef __cplusplus
}
#endif

#endif /* HTTPAPI_H */
