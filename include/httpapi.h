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

/* Check if the HTTP thread received a config update via PUT.
 * Returns true and fills *cfg if a pending config exists (clears pending flag).
 * Audio thread calls this at the start of each on_chunk. */
bool httpapi_get_pending_config(httpapi_t *api, dsd_config_t *cfg);

#ifdef __cplusplus
}
#endif

#endif /* HTTPAPI_H */
