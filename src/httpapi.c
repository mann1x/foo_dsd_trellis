/*
 * foo_dsd_trellis — Lightweight HTTP REST API
 *
 * Raw Winsock2 server, single listener thread, 127.0.0.1 only.
 * SRWLOCK protects config/status snapshots shared between audio and HTTP threads.
 */

#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#pragma comment(lib, "ws2_32.lib")

#include "../include/httpapi.h"
#include "../include/trellis.h"
#include "../include/precorr.h"
#include "../include/ntf.h"
#include "../include/wav_io.h"
#include "../include/dop.h"
#include "../include/fir.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Forward declaration from config.c */
void config_validate(dsd_config_t *cfg);

/* ─── Log ring buffer ─── */
log_ring_t g_log_ring = { .head = 0, .count = 0, .lock = SRWLOCK_INIT };

void log_ring_write(const char *line) {
    AcquireSRWLockExclusive(&g_log_ring.lock);
    strncpy_s(g_log_ring.lines[g_log_ring.head], LOG_RING_MAX_LINE, line, _TRUNCATE);
    g_log_ring.head = (g_log_ring.head + 1) % LOG_RING_MAX_LINES;
    if (g_log_ring.count < LOG_RING_MAX_LINES)
        g_log_ring.count++;
    ReleaseSRWLockExclusive(&g_log_ring.lock);
}

int log_ring_read(char *buf, int buf_size, int max_lines) {
    AcquireSRWLockShared(&g_log_ring.lock);
    int n = g_log_ring.count;
    if (max_lines < n) n = max_lines;
    int pos = 0;
    /* Start from oldest of the N lines we want */
    int start = (g_log_ring.head - n + LOG_RING_MAX_LINES) % LOG_RING_MAX_LINES;
    for (int i = 0; i < n && pos < buf_size - 2; i++) {
        int idx = (start + i) % LOG_RING_MAX_LINES;
        int len = (int)strlen(g_log_ring.lines[idx]);
        if (pos + len + 1 >= buf_size) break;
        memcpy(buf + pos, g_log_ring.lines[idx], len);
        pos += len;
        buf[pos++] = '\n';
    }
    buf[pos] = '\0';
    ReleaseSRWLockShared(&g_log_ring.lock);
    return pos;
}

#define HTTP_MAX_REQUEST  8192
#define HTTP_MAX_RESPONSE 16384

struct httpapi {
    HANDLE          thread;
    volatile LONG   stop_flag;
    SOCKET          listen_sock;
    uint16_t        port;

    /* Shared state protected by SRWLOCK */
    SRWLOCK         lock;
    dsd_config_t    config;
    httpapi_status_t status;

    /* Pending config from PUT (audio thread reads + clears) */
    SRWLOCK         pending_lock;
    dsd_config_t    pending_config;
    volatile LONG   has_pending;
};

/* ─── Simple JSON helpers (no external deps) ─── */

static int json_append(char *buf, int pos, int max, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + pos, (size_t)(max - pos), fmt, ap);
    va_end(ap);
    if (n < 0 || pos + n >= max) return max - 1;
    return pos + n;
}

static int config_to_json(const dsd_config_t *cfg, char *buf, int max) {
    int p = 0;
    p = json_append(buf, p, max, "{\n");
    p = json_append(buf, p, max, "  \"fs_out\": %u,\n", cfg->fs_out);
    p = json_append(buf, p, max, "  \"gain\": %.6f,\n", (double)cfg->gain);
    p = json_append(buf, p, max, "  \"mute\": %s,\n", cfg->mute ? "true" : "false");
    p = json_append(buf, p, max, "  \"trellis_depth\": %d,\n", cfg->trellis_depth);
    p = json_append(buf, p, max, "  \"trellis_cands\": %d,\n", cfg->trellis_cands);
    p = json_append(buf, p, max, "  \"trellis_lat\": %d,\n", cfg->trellis_lat);
    p = json_append(buf, p, max, "  \"ntf_filter\": %d,\n", cfg->ntf_filter);
    p = json_append(buf, p, max, "  \"thread_count\": %d,\n", cfg->thread_count);
    p = json_append(buf, p, max, "  \"format\": %d,\n", cfg->format);
    p = json_append(buf, p, max, "  \"output_format\": %d,\n", cfg->output_format);
    p = json_append(buf, p, max, "  \"debug_log\": %s,\n", cfg->debug_log ? "true" : "false");
    p = json_append(buf, p, max, "  \"smt_mode\": %d,\n", cfg->smt_mode);
    p = json_append(buf, p, max, "  \"ccd_mode\": %d,\n", cfg->ccd_mode);
    p = json_append(buf, p, max, "  \"ecore_mode\": %d,\n", cfg->ecore_mode);
    p = json_append(buf, p, max, "  \"api_port\": %u,\n", (unsigned)cfg->api_port);
    p = json_append(buf, p, max, "  \"sdm_mode\": %d,\n", cfg->sdm_mode);
    p = json_append(buf, p, max, "  \"sdm_mode_name\": \"%s\",\n",
                    cfg->sdm_mode == 1 ? "trellis" : "precorr");
    p = json_append(buf, p, max, "  \"antipop\": %s,\n", cfg->antipop ? "true" : "false");
    p = json_append(buf, p, max, "  \"ml_enabled\": %s,\n", cfg->ml_enabled ? "true" : "false");
    p = json_append(buf, p, max, "  \"fir_gain_db\": %d,\n", (int)cfg->fir_gain_db);
    p = json_append(buf, p, max, "  \"gpu_enabled\": %s,\n", cfg->gpu_enabled ? "true" : "false");
    p = json_append(buf, p, max, "  \"gpu_backend\": %d,\n", cfg->gpu_backend);
    p = json_append(buf, p, max, "  \"gpu_backend_name\": \"%s\",\n",
                    cfg->gpu_backend == 1 ? "DirectCompute" :
                    cfg->gpu_backend == 2 ? "CUDA" :
                    cfg->gpu_backend == 3 ? "Auto" : "None");
    p = json_append(buf, p, max, "  \"fir_engine\": \"%s\"\n", fir_ipp_kernel_name());
    p = json_append(buf, p, max, "}");
    return p;
}

static int status_to_json(const httpapi_status_t *st, char *buf, int max) {
    int p = 0;
    p = json_append(buf, p, max, "{\n");
    p = json_append(buf, p, max, "  \"playing\": %s,\n", st->playing ? "true" : "false");
    p = json_append(buf, p, max, "  \"dsd_rate_in\": %u,\n", st->dsd_rate_in);
    p = json_append(buf, p, max, "  \"dsd_rate_out\": %u,\n", st->dsd_rate_out);
    p = json_append(buf, p, max, "  \"channels\": %d,\n", st->channels);
    p = json_append(buf, p, max, "  \"threads\": %d,\n", st->threads);
    p = json_append(buf, p, max, "  \"segments_per_ch\": %d,\n", st->segments_per_ch);
    p = json_append(buf, p, max, "  \"cpuset_mask\": \"0x%016llX\",\n", (unsigned long long)st->cpuset_mask);
    p = json_append(buf, p, max, "  \"cpuset_enabled\": %d,\n", st->cpuset_enabled);
    p = json_append(buf, p, max, "  \"chunk_count\": %u,\n", st->chunk_count);
    p = json_append(buf, p, max, "  \"last_chunk_ms\": %.1f,\n", st->last_chunk_ms);
    p = json_append(buf, p, max, "  \"last_audio_ms\": %.1f,\n", st->last_audio_ms);
    p = json_append(buf, p, max, "  \"rt_ratio\": %.3f,\n", st->rt_ratio);
    p = json_append(buf, p, max, "  \"phase\": {\n");
    p = json_append(buf, p, max, "    \"unpack_ms\": %.1f,\n", st->unpack_ms);
    p = json_append(buf, p, max, "    \"fir_ms\": %.1f,\n", st->fir_ms);
    p = json_append(buf, p, max, "    \"sdm_ms\": %.1f,\n", st->sdm_ms);
    p = json_append(buf, p, max, "    \"pack_ms\": %.1f\n", st->pack_ms);
    p = json_append(buf, p, max, "  }\n");
    p = json_append(buf, p, max, "}");
    return p;
}

/* ─── Minimal JSON key:value parser for PUT ─── */

/* Find a key in JSON and return pointer to value start, or NULL */
static const char *json_find_key(const char *json, const char *key) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t' || *p == ':') p++;
    return p;
}

static bool json_read_int(const char *json, const char *key, int *out) {
    const char *v = json_find_key(json, key);
    if (!v) return false;
    *out = atoi(v);
    return true;
}

static bool json_read_uint32(const char *json, const char *key, uint32_t *out) {
    const char *v = json_find_key(json, key);
    if (!v) return false;
    *out = (uint32_t)strtoul(v, NULL, 10);
    return true;
}

static bool json_read_float(const char *json, const char *key, float *out) {
    const char *v = json_find_key(json, key);
    if (!v) return false;
    *out = (float)atof(v);
    return true;
}

static bool json_read_bool(const char *json, const char *key, bool *out) {
    const char *v = json_find_key(json, key);
    if (!v) return false;
    if (strncmp(v, "true", 4) == 0) { *out = true; return true; }
    if (strncmp(v, "false", 5) == 0) { *out = false; return true; }
    return false;
}

static bool json_read_string(const char *json, const char *key, char *out, size_t max) {
    const char *v = json_find_key(json, key);
    if (!v || *v != '"') return false;
    v++;  /* skip opening quote */
    size_t i = 0;
    while (*v && *v != '"' && i < max - 1) {
        if (*v == '\\' && v[1]) { v++; }  /* skip escape */
        out[i++] = *v++;
    }
    out[i] = '\0';
    return i > 0;
}

static bool json_read_double(const char *json, const char *key, double *out) {
    const char *v = json_find_key(json, key);
    if (!v) return false;
    *out = atof(v);
    return true;
}

/* Parse partial JSON and merge into config */
static void json_merge_config(const char *json, dsd_config_t *cfg) {
    json_read_uint32(json, "fs_out", &cfg->fs_out);
    json_read_float(json, "gain", &cfg->gain);
    json_read_bool(json, "mute", &cfg->mute);
    json_read_int(json, "trellis_depth", &cfg->trellis_depth);
    json_read_int(json, "trellis_cands", &cfg->trellis_cands);
    json_read_int(json, "trellis_lat", &cfg->trellis_lat);
    json_read_int(json, "ntf_filter", &cfg->ntf_filter);
    json_read_int(json, "thread_count", &cfg->thread_count);
    json_read_int(json, "format", &cfg->format);
    json_read_int(json, "output_format", &cfg->output_format);
    json_read_bool(json, "debug_log", &cfg->debug_log);
    json_read_int(json, "smt_mode", &cfg->smt_mode);
    json_read_int(json, "ccd_mode", &cfg->ccd_mode);
    json_read_int(json, "ecore_mode", &cfg->ecore_mode);
    json_read_int(json, "sdm_mode", &cfg->sdm_mode);
    config_validate(cfg);
}

/* ─── HTTP response helpers ─── */

static void send_response(SOCKET s, int code, const char *status,
                           const char *content_type, const char *body, int body_len) {
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, PUT, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Connection: close\r\n"
        "\r\n",
        code, status, content_type, body_len);
    send(s, header, hlen, 0);
    if (body_len > 0)
        send(s, body, body_len, 0);
}

static void send_json(SOCKET s, int code, const char *status,
                       const char *json, int json_len) {
    send_response(s, code, status, "application/json", json, json_len);
}

static void send_not_found(SOCKET s) {
    const char *body = "{\"error\":\"not found\"}";
    send_json(s, 404, "Not Found", body, (int)strlen(body));
}

static void send_method_not_allowed(SOCKET s) {
    const char *body = "{\"error\":\"method not allowed\"}";
    send_json(s, 405, "Method Not Allowed", body, (int)strlen(body));
}

/* ─── Goertzel SINAD measurement ─── */

static double goertzel_pwr(const float *x, size_t n, double freq_hz, double fs) {
    double k = freq_hz * (double)n / fs;
    double w = 2.0 * M_PI * k / (double)n;
    double coeff = 2.0 * cos(w);
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (size_t i = 0; i < n; i++) {
        s0 = (double)x[i] + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    double re = s1 - s2 * cos(w);
    double im = s2 * sin(w);
    return (re * re + im * im) / ((double)n * (double)n);
}

static double measure_sinad(const float *x, size_t n, double freq_hz, double fs) {
    double sig = goertzel_pwr(x, n, freq_hz, fs);
    double bw = fs / (double)n;
    unsigned max_bin = (unsigned)(22050.0 / bw);
    unsigned sig_bin = (unsigned)(freq_hz / bw + 0.5);
    double noise = 0.0;
    for (unsigned b = 1; b <= max_bin; b++) {
        if (b >= sig_bin - 1 && b <= sig_bin + 1) continue;
        noise += goertzel_pwr(x, n, b * bw, fs);
    }
    if (noise <= 0.0) noise = 1e-30;
    return 10.0 * log10(sig / noise);
}

/* ─── POST /api/render ─── */

static void handle_render(SOCKET client, const char *body) {
    char resp[HTTP_MAX_RESPONSE];

    /* Parse parameters */
    double freq = 1000.0, amplitude = 0.5, duration_s = -1.0;
    uint32_t output_rate = DSD_RATE_128;
    int sdm_mode = 0, ntf_filter = -1;
    int depth = 8, cands = 8, lat = DSD_DEFAULT_TRELLIS_LAT;
    char output_path[512] = {0};
    char input_path[512] = {0};
    int channel = 0;

    json_read_string(body, "input", input_path, sizeof(input_path));
    json_read_double(body, "freq", &freq);
    json_read_double(body, "amplitude", &amplitude);
    json_read_double(body, "duration_s", &duration_s);
    json_read_uint32(body, "output_rate", &output_rate);
    json_read_int(body, "sdm_mode", &sdm_mode);
    json_read_int(body, "ntf_filter", &ntf_filter);
    json_read_int(body, "trellis_depth", &depth);
    json_read_int(body, "trellis_cands", &cands);
    json_read_int(body, "trellis_lat", &lat);
    json_read_string(body, "output", output_path, sizeof(output_path));
    json_read_int(body, "channel", &channel);

    /* Validate output rate */
    if (output_rate != DSD_RATE_64 && output_rate != DSD_RATE_128 &&
        output_rate != DSD_RATE_256 && output_rate != DSD_RATE_512) {
        const char *e = "{\"error\":\"invalid output_rate\"}";
        send_json(client, 400, "Bad Request", e, (int)strlen(e));
        return;
    }

    /* Select NTF filter */
    const ntf_filter_t *filter = NULL;
    if (ntf_filter < 0) {
        filter = (sdm_mode == SDM_MODE_PRECORR)
            ? ntf_auto_select_precorr(output_rate)
            : ntf_auto_select(output_rate);
    } else {
        filter = ntf_get_filter((ntf_filter_id_t)ntf_filter, output_rate);
    }
    if (!filter) {
        const char *e = "{\"error\":\"no NTF filter for rate\"}";
        send_json(client, 400, "Bad Request", e, (int)strlen(e));
        return;
    }

    /* --- Prepare SDM input: either from PCM file or generated tone --- */
    float *in = NULL;
    float *out = NULL;
    size_t n_samples = 0;
    double aligned_freq = 0.0;
    uint32_t input_rate = 0;
    uint32_t upsample_ratio = 0;

    if (input_path[0]) {
        /* ── PCM file input: read WAV → FIR upsample → SDM ── */
        wav_data_t wav;
        if (wav_read(input_path, &wav) != 0) {
            const char *e = "{\"error\":\"failed to read input WAV\"}";
            send_json(client, 400, "Bad Request", e, (int)strlen(e));
            return;
        }

        input_rate = wav.sample_rate;

        /* Validate upsample ratio is power of 2 */
        if (output_rate <= input_rate || (output_rate % input_rate) != 0) {
            wav_free(&wav);
            const char *e = "{\"error\":\"output_rate must be a multiple of input sample rate\"}";
            send_json(client, 400, "Bad Request", e, (int)strlen(e));
            return;
        }
        upsample_ratio = output_rate / input_rate;
        if ((upsample_ratio & (upsample_ratio - 1)) != 0) {
            wav_free(&wav);
            const char *e = "{\"error\":\"upsample ratio must be power of 2\"}";
            send_json(client, 400, "Bad Request", e, (int)strlen(e));
            return;
        }

        /* Cap DSD output to ~2.8M samples to keep HTTP thread responsive.
         * Scale input frames inversely with upsample ratio. */
        uint32_t max_dsd_samples = 2822400;  /* ~1s at DSD64 */
        uint32_t max_frames = max_dsd_samples / upsample_ratio;
        if (max_frames < 1000) max_frames = 1000;  /* minimum for SINAD accuracy */
        if (duration_s > 0 && duration_s < (double)max_frames / input_rate)
            max_frames = (uint32_t)(duration_s * input_rate);
        uint32_t pcm_frames = wav.num_frames < max_frames ? wav.num_frames : max_frames;

        /* Extract mono channel */
        float *pcm_mono = (float *)malloc(pcm_frames * sizeof(float));
        if (!pcm_mono) { wav_free(&wav); goto oom_render; }

        if (wav.channels == 1) {
            memcpy(pcm_mono, wav.samples, pcm_frames * sizeof(float));
        } else {
            if (channel >= wav.channels) channel = 0;
            for (uint32_t i = 0; i < pcm_frames; i++)
                pcm_mono[i] = wav.samples[i * wav.channels + channel];
        }
        wav_free(&wav);

        /* FIR upsample: PCM rate → DSD rate */
        n_samples = (size_t)pcm_frames * upsample_ratio;
        in = (float *)malloc(n_samples * sizeof(float));
        out = (float *)malloc(n_samples * sizeof(float));
        if (!in || !out) { free(pcm_mono); free(in); free(out); goto oom_render; }

        fir_chain_t fir;
        if (fir_chain_init(&fir, input_rate, output_rate) != 0) {
            free(pcm_mono); free(in); free(out);
            const char *e = "{\"error\":\"FIR init failed for upsample ratio\"}";
            send_json(client, 400, "Bad Request", e, (int)strlen(e));
            return;
        }

        size_t fir_out = fir_chain_process(&fir, pcm_mono, in, pcm_frames);
        fir_chain_free(&fir);
        free(pcm_mono);
        n_samples = fir_out;

        /* Bin-align freq for SINAD measurement at output rate */
        double bw = (double)output_rate / (double)n_samples;
        unsigned sig_bin = (unsigned)(freq / bw + 0.5);
        aligned_freq = sig_bin * bw;

    } else {
        /* ── Tone generation: synthesize at DSD rate ── */
        if (freq <= 0 || freq >= 22050.0) {
            const char *e = "{\"error\":\"freq must be in (0, 22050)\"}";
            send_json(client, 400, "Bad Request", e, (int)strlen(e));
            return;
        }
        if (duration_s < 0) duration_s = 0.1;  /* default for tone mode */
        if (duration_s <= 0 || duration_s > 30.0) {
            const char *e = "{\"error\":\"duration_s must be in (0, 30]\"}";
            send_json(client, 400, "Bad Request", e, (int)strlen(e));
            return;
        }

        n_samples = (size_t)(output_rate * duration_s);
        double bw = (double)output_rate / (double)n_samples;
        unsigned sig_bin = (unsigned)(freq / bw + 0.5);
        aligned_freq = sig_bin * bw;

        in = (float *)malloc(n_samples * sizeof(float));
        out = (float *)malloc(n_samples * sizeof(float));
        if (!in || !out) { free(in); free(out); goto oom_render; }

        for (size_t i = 0; i < n_samples; i++)
            in[i] = (float)(amplitude * sin(2.0 * M_PI * aligned_freq * i / output_rate));
    }

    /* --- Process through SDM --- */
    LARGE_INTEGER t0, t1, tfreq;
    QueryPerformanceFrequency(&tfreq);
    QueryPerformanceCounter(&t0);

    size_t produced;
    if (sdm_mode == SDM_MODE_PRECORR) {
        precorr_context_t ctx;
        if (precorr_context_init(&ctx, filter) != 0) {
            free(in); free(out);
            const char *e = "{\"error\":\"precorr init failed\"}";
            send_json(client, 500, "Internal Server Error", e, (int)strlen(e));
            return;
        }
        produced = precorr_process_block(&ctx, in, out, n_samples);
        precorr_context_free(&ctx);
    } else {
        sdm_context_t ctx;
        if (sdm_context_init(&ctx, filter, depth, cands, lat) != 0) {
            free(in); free(out);
            const char *e = "{\"error\":\"trellis init failed\"}";
            send_json(client, 500, "Internal Server Error", e, (int)strlen(e));
            return;
        }
        produced = sdm_process_block(&ctx, in, out, n_samples);
        /* Drain latency buffer */
        float *drain_buf = (float *)malloc((size_t)lat * sizeof(float));
        if (drain_buf) {
            size_t drained = sdm_drain(&ctx, drain_buf, (size_t)lat);
            if (produced + drained <= n_samples) {
                memcpy(out + produced, drain_buf, drained * sizeof(float));
                produced += drained;
            }
            free(drain_buf);
        }
        sdm_context_free(&ctx);
    }

    QueryPerformanceCounter(&t1);
    double proc_ms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / tfreq.QuadPart;

    /* Measure SINAD — cap to 0.1s window to keep Goertzel sweep fast */
    double sinad_db = -999.0;
    if (produced > 1024) {
        size_t sinad_n = produced;
        size_t sinad_max = (size_t)(output_rate / 10);  /* 0.1s */
        if (sinad_n > sinad_max) {
            /* Use the last sinad_max samples (skip transient at start) */
            sinad_db = measure_sinad(out + (produced - sinad_max), sinad_max,
                                     aligned_freq, (double)output_rate);
        } else {
            sinad_db = measure_sinad(out, sinad_n, aligned_freq, (double)output_rate);
        }
    }

    /* Save output WAV */
    int saved = 0;
    if (output_path[0] && produced > 0) {
        saved = (wav_write(output_path, out, (uint32_t)produced, 1, output_rate) == 0);
    }

    free(in);
    free(out);

    /* Build response */
    int p = 0;
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "{\n");
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"status\": \"ok\",\n");
    if (input_path[0]) {
        p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"input_file\": \"%s\",\n", input_path);
        p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"input_rate\": %u,\n", input_rate);
        p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"upsample_ratio\": %u,\n", upsample_ratio);
    }
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"output_rate\": %u,\n", output_rate);
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"output_samples\": %zu,\n", produced);
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"sdm_mode\": \"%s\",\n",
                    sdm_mode == SDM_MODE_PRECORR ? "precorr" : "trellis");
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"ntf_filter\": \"%s\",\n", filter->name);
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"ntf_order\": %d,\n", filter->order);
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"signal_freq_hz\": %.6f,\n", aligned_freq);
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"sinad_db\": %.1f,\n", sinad_db);
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"processing_time_ms\": %.1f,\n", proc_ms);
    if (output_path[0])
        p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"output_file\": \"%s\",\n", output_path);
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"output_saved\": %s\n",
                    saved ? "true" : "false");
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "}");
    send_json(client, 200, "OK", resp, p);
    return;

oom_render:
    {
        const char *e = "{\"error\":\"out of memory\"}";
        send_json(client, 500, "Internal Server Error", e, (int)strlen(e));
    }
}

/* ─── POST /api/measure ─── */

static void handle_measure(SOCKET client, const char *body) {
    char resp[HTTP_MAX_RESPONSE];

    char input_path[512] = {0};
    double freq = 1000.0;
    int is_dop = 0;
    int channel = 0;

    json_read_string(body, "input", input_path, sizeof(input_path));
    json_read_double(body, "freq", &freq);
    json_read_int(body, "dop", &is_dop);
    json_read_int(body, "channel", &channel);

    if (!input_path[0]) {
        const char *e = "{\"error\":\"input path required\"}";
        send_json(client, 400, "Bad Request", e, (int)strlen(e));
        return;
    }

    /* Read WAV */
    wav_data_t wav;
    if (wav_read(input_path, &wav) != 0) {
        const char *e = "{\"error\":\"failed to read WAV file\"}";
        send_json(client, 400, "Bad Request", e, (int)strlen(e));
        return;
    }

    /* Extract single channel if multi-channel */
    float *mono = NULL;
    size_t mono_n = wav.num_frames;
    uint32_t dsd_rate = wav.sample_rate;

    if (is_dop) {
        /* DoP: unpack to DSD ±1 float.
         * Each PCM frame contains 16 DSD bits in DoP encoding.
         * DSD rate = PCM rate * 16. */
        dsd_rate = wav.sample_rate * 16;
        size_t dsd_samples = (size_t)wav.num_frames * 16;
        mono = (float *)malloc(dsd_samples * sizeof(float));
        if (!mono) { wav_free(&wav); goto oom; }

        if (wav.channels == 1) {
            dop_unpack(wav.samples, mono, wav.num_frames);
        } else {
            /* Extract one channel, then unpack */
            float *ch_data = (float *)malloc(wav.num_frames * sizeof(float));
            if (!ch_data) { free(mono); wav_free(&wav); goto oom; }
            if (channel >= wav.channels) channel = 0;
            for (uint32_t i = 0; i < wav.num_frames; i++)
                ch_data[i] = wav.samples[i * wav.channels + channel];
            dop_unpack(ch_data, mono, wav.num_frames);
            free(ch_data);
        }
        mono_n = dsd_samples;
    } else {
        /* Raw float samples — extract single channel */
        mono = (float *)malloc(wav.num_frames * sizeof(float));
        if (!mono) { wav_free(&wav); goto oom; }

        if (wav.channels == 1) {
            memcpy(mono, wav.samples, wav.num_frames * sizeof(float));
        } else {
            if (channel >= wav.channels) channel = 0;
            for (uint32_t i = 0; i < wav.num_frames; i++)
                mono[i] = wav.samples[i * wav.channels + channel];
        }
    }

    wav_free(&wav);

    /* Bin-align measurement frequency */
    double bw = (double)dsd_rate / (double)mono_n;
    unsigned sig_bin = (unsigned)(freq / bw + 0.5);
    double aligned_freq = sig_bin * bw;

    /* Measure SINAD — cap to 0.1s window to keep Goertzel sweep fast */
    double sinad_db = -999.0;
    if (mono_n > 1024) {
        size_t sinad_n = mono_n;
        size_t sinad_max = (size_t)(dsd_rate / 10);  /* 0.1s */
        if (sinad_n > sinad_max) {
            sinad_db = measure_sinad(mono + (mono_n - sinad_max), sinad_max,
                                     aligned_freq, (double)dsd_rate);
        } else {
            sinad_db = measure_sinad(mono, sinad_n, aligned_freq, (double)dsd_rate);
        }
    }

    free(mono);

    /* Build response */
    int p = 0;
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "{\n");
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"status\": \"ok\",\n");
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"input_file\": \"%s\",\n", input_path);
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"dsd_rate\": %u,\n", dsd_rate);
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"samples\": %zu,\n", mono_n);
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"signal_freq_hz\": %.6f,\n", aligned_freq);
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"sinad_db\": %.1f,\n", sinad_db);
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "  \"dop\": %s\n", is_dop ? "true" : "false");
    p = json_append(resp, p, HTTP_MAX_RESPONSE, "}");
    send_json(client, 200, "OK", resp, p);
    return;

oom:
    {
        const char *e = "{\"error\":\"out of memory\"}";
        send_json(client, 500, "Internal Server Error", e, (int)strlen(e));
    }
}

/* ─── Request handling ─── */

static void handle_request(httpapi_t *api, SOCKET client) {
    char req[HTTP_MAX_REQUEST];
    int n = recv(client, req, sizeof(req) - 1, 0);
    if (n <= 0) return;
    req[n] = '\0';

    /* Parse method and path */
    char method[8] = {0};
    char path[256] = {0};
    if (sscanf_s(req, "%7s %255s", method, (unsigned)sizeof(method), path, (unsigned)sizeof(path)) != 2)
        return;

    /* CORS preflight */
    if (_stricmp(method, "OPTIONS") == 0) {
        send_response(client, 204, "No Content", "text/plain", "", 0);
        return;
    }

    if (strcmp(path, "/api/config") == 0) {
        if (_stricmp(method, "GET") == 0) {
            /* Read config under shared lock */
            char json[HTTP_MAX_RESPONSE];
            AcquireSRWLockShared(&api->lock);
            int len = config_to_json(&api->config, json, sizeof(json));
            ReleaseSRWLockShared(&api->lock);
            send_json(client, 200, "OK", json, len);

        } else if (_stricmp(method, "PUT") == 0) {
            /* Find body after \r\n\r\n */
            const char *body = strstr(req, "\r\n\r\n");
            if (!body) {
                send_json(client, 400, "Bad Request",
                          "{\"error\":\"no body\"}", 18);
                return;
            }
            body += 4;

            /* Read current config, merge, validate, set as pending */
            AcquireSRWLockExclusive(&api->pending_lock);
            dsd_config_t cfg;
            /* Start from current config for partial merge */
            AcquireSRWLockShared(&api->lock);
            cfg = api->config;
            ReleaseSRWLockShared(&api->lock);
            json_merge_config(body, &cfg);
            api->pending_config = cfg;
            InterlockedExchange(&api->has_pending, 1);
            ReleaseSRWLockExclusive(&api->pending_lock);

            const char *ok = "{\"status\":\"accepted\"}";
            send_json(client, 200, "OK", ok, (int)strlen(ok));

        } else {
            send_method_not_allowed(client);
        }

    } else if (strcmp(path, "/api/status") == 0) {
        if (_stricmp(method, "GET") == 0) {
            char json[HTTP_MAX_RESPONSE];
            AcquireSRWLockShared(&api->lock);
            int len = status_to_json(&api->status, json, sizeof(json));
            ReleaseSRWLockShared(&api->lock);
            send_json(client, 200, "OK", json, len);
        } else {
            send_method_not_allowed(client);
        }

    } else if (strcmp(path, "/api/render") == 0) {
        if (_stricmp(method, "POST") == 0) {
            const char *body = strstr(req, "\r\n\r\n");
            if (!body) {
                send_json(client, 400, "Bad Request",
                          "{\"error\":\"no body\"}", 18);
                return;
            }
            body += 4;
            handle_render(client, body);
        } else {
            send_method_not_allowed(client);
        }

    } else if (strcmp(path, "/api/measure") == 0) {
        if (_stricmp(method, "POST") == 0) {
            const char *body = strstr(req, "\r\n\r\n");
            if (!body) {
                send_json(client, 400, "Bad Request",
                          "{\"error\":\"no body\"}", 18);
                return;
            }
            body += 4;
            handle_measure(client, body);
        } else {
            send_method_not_allowed(client);
        }

    } else if (strncmp(path, "/api/log", 8) == 0) {
        if (_stricmp(method, "GET") == 0) {
            /* GET /api/log?lines=N          — read from ring buffer (fast, default)
             * GET /api/log?lines=N&logfile=true — read from log file (full history) */
            int max_lines = 50;
            bool from_file = false;
            const char *q = strchr(path, '?');
            if (q) {
                const char *lp = strstr(q, "lines=");
                if (lp) {
                    max_lines = atoi(lp + 6);
                    if (max_lines < 1) max_lines = 1;
                    if (max_lines > 500) max_lines = 500;
                }
                if (strstr(q, "logfile=true") || strstr(q, "logfile=1"))
                    from_file = true;
            }

            if (!from_file) {
                /* Ring buffer: instant, no file I/O */
                char *buf = (char *)malloc(max_lines * LOG_RING_MAX_LINE);
                if (!buf) {
                    send_json(client, 500, "Internal Error",
                              "{\"error\":\"malloc\"}", 18);
                    return;
                }
                int len = log_ring_read(buf, max_lines * LOG_RING_MAX_LINE, max_lines);
                send_response(client, 200, "OK", "text/plain", buf, len);
                free(buf);
            } else {
                /* File: read last N lines from log file on disk */
                wchar_t dll_path[MAX_PATH];
                HMODULE hmod = GetModuleHandleW(L"foo_dsd_trellis.dll");
                if (!hmod) {
                    MEMORY_BASIC_INFORMATION mbi;
                    if (VirtualQuery((void *)handle_request, &mbi, sizeof(mbi)))
                        hmod = (HMODULE)mbi.AllocationBase;
                }
                if (!hmod) {
                    send_json(client, 500, "Internal Error",
                              "{\"error\":\"dll not found\"}", 24);
                    return;
                }
                GetModuleFileNameW(hmod, dll_path, MAX_PATH);
                wchar_t *sep = wcsrchr(dll_path, L'\\');
                if (!sep) sep = wcsrchr(dll_path, L'/');
                if (sep) *(sep + 1) = L'\0';
                wcscat_s(dll_path, MAX_PATH, L"foo_dsd_trellis.log");

                HANDLE hFile = CreateFileW(dll_path, GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
                if (hFile == INVALID_HANDLE_VALUE) {
                    char err[512];
                    char np[MAX_PATH];
                    WideCharToMultiByte(CP_UTF8, 0, dll_path, -1, np, MAX_PATH, NULL, NULL);
                    int el = snprintf(err, sizeof(err),
                        "{\"error\":\"log not found\",\"path\":\"%s\",\"err\":%lu}", np, GetLastError());
                    send_json(client, 404, "Not Found", err, el);
                    return;
                }
                LARGE_INTEGER fsz;
                GetFileSizeEx(hFile, &fsz);
                DWORD rlen = (DWORD)fsz.QuadPart;
                LONGLONG rstart = 0;
                if (rlen > 65536) { rstart = fsz.QuadPart - 65536; rlen = 65536; }
                char *buf = (char *)malloc(rlen + 1);
                if (!buf) { CloseHandle(hFile); return; }
                LARGE_INTEGER li; li.QuadPart = rstart;
                SetFilePointerEx(hFile, li, NULL, FILE_BEGIN);
                DWORD br = 0;
                ReadFile(hFile, buf, rlen, &br, NULL);
                CloseHandle(hFile);
                buf[br] = '\0';
                int lc = 0;
                char *ls = buf + br;
                for (char *p = buf + br - 1; p >= buf; p--) {
                    if (*p == '\n' && ++lc >= max_lines) { ls = p + 1; break; }
                }
                if (ls == buf + br && br > 0) ls = buf;
                send_response(client, 200, "OK", "text/plain", ls, (int)((buf + br) - ls));
                free(buf);
            }
        } else {
            send_method_not_allowed(client);
        }

    } else {
        send_not_found(client);
    }
}

/* ─── Listener thread ─── */

static DWORD WINAPI httpapi_thread(LPVOID param) {
    httpapi_t *api = (httpapi_t *)param;

    while (!InterlockedCompareExchange(&api->stop_flag, 0, 0)) {
        /* Use select() with timeout so we can check stop_flag */
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(api->listen_sock, &fds);
        struct timeval tv = { 0, 250000 }; /* 250ms */

        int sel = select(0, &fds, NULL, NULL, &tv);
        if (sel <= 0) continue;

        SOCKET client = accept(api->listen_sock, NULL, NULL);
        if (client == INVALID_SOCKET) continue;

        /* Set short recv timeout to avoid blocking */
        DWORD timeout_ms = 1000;
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO,
                   (const char *)&timeout_ms, sizeof(timeout_ms));

        handle_request(api, client);
        closesocket(client);
    }

    return 0;
}

/* ─── Public API ─── */

httpapi_t *httpapi_create(uint16_t port, const dsd_config_t *initial_cfg) {
    /* Init Winsock */
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
        return NULL;

    httpapi_t *api = (httpapi_t *)calloc(1, sizeof(httpapi_t));
    if (!api) {
        WSACleanup();
        return NULL;
    }

    api->port = port;
    InitializeSRWLock(&api->lock);
    InitializeSRWLock(&api->pending_lock);
    api->config = *initial_cfg;

    /* Create listening socket */
    api->listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (api->listen_sock == INVALID_SOCKET) {
        free(api);
        WSACleanup();
        return NULL;
    }

    /* Allow port reuse (in case of quick restart) */
    int opt = 1;
    setsockopt(api->listen_sock, SOL_SOCKET, SO_REUSEADDR,
               (const char *)&opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); /* 127.0.0.1 only */
    addr.sin_port = htons(port);

    if (bind(api->listen_sock, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        closesocket(api->listen_sock);
        free(api);
        WSACleanup();
        return NULL;
    }

    if (listen(api->listen_sock, 4) != 0) {
        closesocket(api->listen_sock);
        free(api);
        WSACleanup();
        return NULL;
    }

    /* Start listener thread */
    api->thread = CreateThread(NULL, 0, httpapi_thread, api, 0, NULL);
    if (!api->thread) {
        closesocket(api->listen_sock);
        free(api);
        WSACleanup();
        return NULL;
    }

    return api;
}

void httpapi_destroy(httpapi_t *api) {
    if (!api) return;

    /* Signal stop and close listening socket to unblock select() */
    InterlockedExchange(&api->stop_flag, 1);
    closesocket(api->listen_sock);

    /* Wait for thread to finish */
    if (api->thread) {
        WaitForSingleObject(api->thread, 3000);
        CloseHandle(api->thread);
    }

    free(api);
    WSACleanup();
}

void httpapi_update_status(httpapi_t *api, const httpapi_status_t *status) {
    if (!api) return;
    AcquireSRWLockExclusive(&api->lock);
    api->status = *status;
    ReleaseSRWLockExclusive(&api->lock);
}

void httpapi_update_config(httpapi_t *api, const dsd_config_t *cfg) {
    if (!api) return;
    AcquireSRWLockExclusive(&api->lock);
    api->config = *cfg;
    ReleaseSRWLockExclusive(&api->lock);
}

bool httpapi_get_pending_config(httpapi_t *api, dsd_config_t *cfg) {
    if (!api) return false;
    if (!InterlockedCompareExchange(&api->has_pending, 0, 0))
        return false;

    AcquireSRWLockExclusive(&api->pending_lock);
    *cfg = api->pending_config;
    InterlockedExchange(&api->has_pending, 0);
    ReleaseSRWLockExclusive(&api->pending_lock);
    return true;
}
