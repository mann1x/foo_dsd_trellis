/*
 * foo_dsd_trellis — Out-of-process worker
 *
 * Standalone exe that runs DSP processing (FIR + SDM) in its own process
 * with unrestricted CPU affinity. Communicates with the fb2k plugin
 * via shared memory ring buffers.
 *
 * Usage: foo_dsd_trellis_worker.exe <name_suffix>
 */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <share.h>
#include "../include/shm_ipc.h"
#include "../include/dsd_types.h"

/* Forward declarations — same C API used by the DLL */
typedef struct plugin_state plugin_state_t;
plugin_state_t *plugin_create(void);
void            plugin_destroy(plugin_state_t *s);
void            plugin_set_config(plugin_state_t *s, const dsd_config_t *cfg);
size_t          plugin_process(plugin_state_t *s, const float *in_pcm,
                               uint8_t *out_i24, size_t pcm_frames,
                               int num_channels, uint32_t pcm_rate);
void            plugin_set_gain(plugin_state_t *s, float gain, bool mute);
void            plugin_flush(plugin_state_t *s);
int             plugin_get_selected_cores(const plugin_state_t *s,
                                           uint32_t *ids, int max_ids);

/* ─── Stubs for symbols referenced by dsp_plugin.c but not needed in worker ─── */

/* Audio capture (HTTP API feature — not used in worker) */
#include "../include/httpapi.h"
audio_capture_t g_audio_capture = { 0 };
void capture_write_dsd(float **ch_out, size_t count, int channels,
                       uint32_t rate) { (void)ch_out; (void)count; (void)channels; (void)rate; }

/* GPU SDM override (REST API toggle — not used in worker) */
volatile LONG g_gpu_sdm_override = 0;
volatile LONG g_gpu_sdm_override_valid = 0;

/* trellis_log_c — required by linked C modules */
static FILE *g_log_file = NULL;

void trellis_log_c(const char *msg) {
    SYSTEMTIME st;
    GetLocalTime(&st);
    if (g_log_file) {
        fprintf(g_log_file, "[%02d:%02d:%02d.%03d] %s\n",
                st.wHour, st.wMinute, st.wSecond, st.wMilliseconds, msg);
        fflush(g_log_file);
    }
}

/* g_log_enabled — referenced by dsp_plugin.c */
bool g_log_enabled = true;

static void log_open(const char *suffix) {
    /* Write log to TEMP dir (always writable) */
    char temp[MAX_PATH];
    GetTempPathA(MAX_PATH, temp);
    char path[MAX_PATH];
    sprintf_s(path, sizeof(path), "%sfoo_dsd_trellis_worker_%s.log", temp, suffix);
    /* Share read access so logs can be read while worker is running */
    g_log_file = _fsopen(path, "a", _SH_DENYWR);
    if (g_log_file)
        fprintf(g_log_file, "--- worker started ---\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: foo_dsd_trellis_worker.exe <name_suffix>\n");
        return 1;
    }

    const char *suffix = argv[1];
    log_open(suffix);
    trellis_log_c("worker: opening shared memory");

    /* Open shared memory created by parent */
    shm_ipc_t ipc;
    int rc = shm_ipc_open(&ipc, suffix);
    if (rc != 0) {
        char msg[128];
        sprintf_s(msg, sizeof(msg), "worker: shm_ipc_open failed (%d)", rc);
        trellis_log_c(msg);
        return 2;
    }

    shm_control_t *ctrl = ipc.ctrl;
    int channels = ctrl->channels;
    unsigned pcm_rate = ctrl->pcm_rate;
    int batch_target = ctrl->batch_target;

    {
        char msg[256];
        sprintf_s(msg, sizeof(msg),
            "worker: ch=%d rate=%u batch=%d in_cap=%d out_cap=%d",
            channels, pcm_rate, batch_target,
            ctrl->in_capacity, ctrl->out_capacity);
        trellis_log_c(msg);
    }

    /* Set parent's affinity as host_thread hint BEFORE plugin_create.
     * cpuset_detect in plugin_create will detect our unpinned threads and
     * default to LP0-3. We override via a global hint that detect_host_threads
     * checks. */
    if (ctrl->parent_affinity != ctrl->system_affinity) {
        extern uint64_t g_host_affinity_hint;
        extern uint64_t g_system_affinity_hint;
        g_host_affinity_hint = ctrl->parent_affinity;
        g_system_affinity_hint = ctrl->system_affinity;
        char msg[128];
        sprintf_s(msg, sizeof(msg), "worker: parent affinity=0x%llx sys=0x%llx",
                  (unsigned long long)ctrl->parent_affinity,
                  (unsigned long long)ctrl->system_affinity);
        trellis_log_c(msg);
    }

    /* Create plugin state */
    plugin_state_t *state = plugin_create();
    if (!state) {
        trellis_log_c("worker: plugin_create failed");
        InterlockedExchange(&ctrl->worker_status, SHM_STATUS_ERROR);
        shm_ipc_close(&ipc);
        return 3;
    }

    plugin_set_config(state, &ctrl->config);

    /* No eager warmup here — the first real batch from the input ring
     * has valid DoP markers which are needed for proper DSD rate detection
     * and engine initialization. The engine's warmup mute fires on the
     * first batch (replacing output with DSD silence), which is fine
     * because the parent outputs silence during startup anyway. */

    /* Accumulation buffer (local to worker) */
    int accum_alloc = batch_target * 2;
    float *accum_buf = (float *)malloc((size_t)accum_alloc * channels * sizeof(float));
    int accum_frames = 0;

    /* Output buffer for plugin_process */
    int out_alloc = batch_target * channels * 3 * 8;  /* 8x for rate conversion */
    uint8_t *out_buf = (uint8_t *)malloc((size_t)out_alloc);

    if (!accum_buf || !out_buf) {
        trellis_log_c("worker: allocation failed");
        InterlockedExchange(&ctrl->worker_status, SHM_STATUS_ERROR);
        plugin_destroy(state);
        shm_ipc_close(&ipc);
        return 4;
    }

    /* Signal ready */
    InterlockedExchange(&ctrl->worker_status, SHM_STATUS_READY);
    trellis_log_c("worker: ready");

    /* Main processing loop */
    HANDLE wait_events[2] = { ipc.input_event, ipc.stop_event };
    int batch_count = 0;

    while (1) {
        DWORD wait = WaitForMultipleObjects(2, wait_events, FALSE, 50);

        if (wait == WAIT_OBJECT_0 + 1)  /* stop event */
            break;
        if (InterlockedCompareExchange(&ctrl->worker_status, 0, 0) == SHM_STATUS_ERROR)
            break;

        /* Read available input from shared memory ring */
        int in_avail = shm_ring_available(&ctrl->in_write_pos, &ctrl->in_read_pos);
        if (in_avail > 0) {
            /* How many frames fit in accum buffer? */
            int frame_bytes = channels * (int)sizeof(float);
            int avail_frames = in_avail / frame_bytes;
            int need = accum_frames + avail_frames;

            /* Grow accum buffer if needed */
            if (need > accum_alloc) {
                accum_alloc = need * 2;
                accum_buf = (float *)realloc(accum_buf,
                    (size_t)accum_alloc * channels * sizeof(float));
            }

            /* Read from input ring into accum buffer */
            int read_bytes = avail_frames * frame_bytes;
            shm_ring_read(ipc.in_ring, ctrl->in_capacity,
                          &ctrl->in_write_pos, &ctrl->in_read_pos,
                          accum_buf + (size_t)accum_frames * channels,
                          read_bytes);
            accum_frames += avail_frames;
        }

        /* Process in fixed batch_target chunks (200ms).
         * Consistent output size prevents ring residual accumulation.
         * The ring (4 seconds) absorbs the batch/drain size mismatch. */
        while (accum_frames >= batch_target) {
            int proc_frames = batch_target;

            /* Grow output buffer if needed */
            int out_need = proc_frames * channels * 3 * 8;
            if (out_need > out_alloc) {
                out_alloc = out_need;
                out_buf = (uint8_t *)realloc(out_buf, (size_t)out_alloc);
            }

            /* Apply gain/mute from control block */
            plugin_set_gain(state,
                            ctrl->gain,
                            InterlockedCompareExchange(&ctrl->mute, 0, 0) != 0);

            /* Process */
            size_t result = plugin_process(state, accum_buf, out_buf,
                                            (size_t)proc_frames, channels, pcm_rate);

            /* Write i24 output to shared memory ring */
            if (result > 0) {
                int out_bytes = (int)(result * (size_t)channels * 3);
                /* Wait for ring to have space before writing.
                 * The parent drains at real-time rate. If the ring is full,
                 * the worker is ahead of real-time — wait for the parent
                 * to catch up. Never drop output data. */
                {
                    int wait_ms = 0;
                    while (shm_ring_free(&ctrl->out_write_pos,
                                          &ctrl->out_read_pos,
                                          ctrl->out_capacity) < out_bytes) {
                        if (WaitForSingleObject(ipc.stop_event, 10) == WAIT_OBJECT_0)
                            goto done;
                        wait_ms += 10;
                        if (wait_ms > 2000) {
                            trellis_log_c("worker: ring full timeout, parent not draining");
                            break;
                        }
                    }
                }
                shm_ring_write(ipc.out_ring, ctrl->out_capacity,
                               &ctrl->out_write_pos, &ctrl->out_read_pos,
                               out_buf, out_bytes);
            }

            /* Shift leftover input */
            int leftover = accum_frames - proc_frames;
            if (leftover > 0) {
                memmove(accum_buf,
                        accum_buf + (size_t)proc_frames * channels,
                        (size_t)leftover * channels * sizeof(float));
            }
            accum_frames = leftover;
            batch_count++;

            /* Check if ring has enough for primed */
            int out_avail = shm_ring_available(&ctrl->out_write_pos,
                                                &ctrl->out_read_pos);
            int primed_threshold = batch_target * channels * 3 * 3;
            if (out_avail >= primed_threshold && !ctrl->primed) {
                InterlockedExchange(&ctrl->primed, 1);
                char msg[128];
                sprintf_s(msg, sizeof(msg),
                    "worker: primed (ring=%d bytes, batch #%d)", out_avail, batch_count);
                trellis_log_c(msg);
            }

            if (batch_count <= 5 || (batch_count % 50) == 0) {
                char msg[128];
                sprintf_s(msg, sizeof(msg),
                    "worker: batch #%d, %zu frames out, ring=%d bytes",
                    batch_count, result, out_avail);
                trellis_log_c(msg);
            }

            /* Check stop between batches */
            if (WaitForSingleObject(ipc.stop_event, 0) == WAIT_OBJECT_0)
                goto done;
        }
    }

done:
    InterlockedExchange(&ctrl->worker_status, SHM_STATUS_EXITING);
    trellis_log_c("worker: shutting down");

    free(accum_buf);
    free(out_buf);
    plugin_destroy(state);
    shm_ipc_close(&ipc);

    if (g_log_file) {
        fprintf(g_log_file, "--- worker stopped ---\n");
        fclose(g_log_file);
    }

    return 0;
}
