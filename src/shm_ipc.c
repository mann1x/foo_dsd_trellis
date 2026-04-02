/*
 * foo_dsd_trellis — Shared memory IPC implementation
 */

#include "../include/shm_ipc.h"
#include <stdio.h>
#include <string.h>

/* Named object prefixes */
#define SHM_PREFIX  "Local\\foo_dsd_trellis_shm_"
#define EVT_INPUT   "Local\\foo_dsd_trellis_input_"
#define EVT_STOP    "Local\\foo_dsd_trellis_stop_"

static void make_name(char *dst, size_t dst_sz,
                      const char *prefix, const char *suffix) {
    sprintf_s(dst, dst_sz, "%s%s", prefix, suffix);
}

int shm_ipc_create(shm_ipc_t *ipc, const char *name_suffix,
                    int channels, unsigned pcm_rate, uint32_t out_pcm_rate,
                    const dsd_config_t *config) {
    memset(ipc, 0, sizeof(*ipc));

    int batch_target = (int)(pcm_rate / 5);  /* ~200ms */

    /* Output batch size: for rate conversion, output has more frames.
     * config->fs_out/fs_in gives the rate ratio. */
    uint32_t fs_in = config->fs_in;
    uint32_t fs_out_actual = config->fs_out ? config->fs_out : fs_in;
    int out_batch = batch_target;
    if (fs_out_actual > fs_in && fs_in > 0)
        out_batch = (int)((double)batch_target * (double)fs_out_actual / (double)fs_in);

    /* Ring sizes: 8× batch, power of 2.
     * Must hold enough batches so the worker never drops output
     * during startup (before drain starts). */
    int in_bytes_raw = batch_target * channels * (int)sizeof(float) * 8;
    int out_bytes_raw = out_batch * channels * 3 * 8;
    int in_capacity = shm_next_pow2(in_bytes_raw);
    int out_capacity = shm_next_pow2(out_bytes_raw);

    uint32_t in_offset = SHM_CTRL_SIZE;
    uint32_t out_offset = in_offset + (uint32_t)in_capacity;
    uint32_t total = out_offset + (uint32_t)out_capacity;

    /* Create file mapping */
    char shm_name[256];
    make_name(shm_name, sizeof(shm_name), SHM_PREFIX, name_suffix);

    ipc->mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL,
                                       PAGE_READWRITE, 0, total, shm_name);
    if (!ipc->mapping)
        return -1;

    ipc->base = MapViewOfFile(ipc->mapping, FILE_MAP_ALL_ACCESS, 0, 0, total);
    if (!ipc->base) {
        CloseHandle(ipc->mapping);
        ipc->mapping = NULL;
        return -2;
    }

    /* Zero everything */
    memset(ipc->base, 0, total);

    /* Set up control block */
    ipc->ctrl = (shm_control_t *)ipc->base;
    ipc->ctrl->magic = SHM_MAGIC;
    ipc->ctrl->version = SHM_VERSION;
    ipc->ctrl->config = *config;
    ipc->ctrl->channels = channels;
    ipc->ctrl->pcm_rate = pcm_rate;
    ipc->ctrl->out_pcm_rate = out_pcm_rate;
    ipc->ctrl->batch_target = batch_target;
    ipc->ctrl->gain = config->gain;
    ipc->ctrl->mute = config->mute ? 1 : 0;
    ipc->ctrl->config_version = 1;
    ipc->ctrl->in_capacity = in_capacity;
    ipc->ctrl->out_capacity = out_capacity;
    ipc->ctrl->in_ring_offset = in_offset;
    ipc->ctrl->out_ring_offset = out_offset;
    ipc->ctrl->total_size = total;
    ipc->ctrl->worker_status = SHM_STATUS_STARTING;
    ipc->ctrl->primed = 0;

    /* Pass parent's process affinity so worker can avoid host cores */
    {
        DWORD_PTR proc_mask = 0, sys_mask = 0;
        GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask);
        ipc->ctrl->parent_affinity = (uint64_t)proc_mask;
        ipc->ctrl->system_affinity = (uint64_t)sys_mask;
    }

    ipc->in_ring = (unsigned char *)ipc->base + in_offset;
    ipc->out_ring = (unsigned char *)ipc->base + out_offset;

    /* Create events */
    char evt_name[256];
    make_name(evt_name, sizeof(evt_name), EVT_INPUT, name_suffix);
    ipc->input_event = CreateEventA(NULL, FALSE, FALSE, evt_name);

    make_name(evt_name, sizeof(evt_name), EVT_STOP, name_suffix);
    ipc->stop_event = CreateEventA(NULL, TRUE, FALSE, evt_name);

    if (!ipc->input_event || !ipc->stop_event) {
        shm_ipc_destroy(ipc);
        return -3;
    }

    return 0;
}

void shm_ipc_destroy(shm_ipc_t *ipc) {
    if (!ipc) return;
    if (ipc->base) { UnmapViewOfFile(ipc->base); ipc->base = NULL; }
    if (ipc->mapping) { CloseHandle(ipc->mapping); ipc->mapping = NULL; }
    if (ipc->input_event) { CloseHandle(ipc->input_event); ipc->input_event = NULL; }
    if (ipc->stop_event) { CloseHandle(ipc->stop_event); ipc->stop_event = NULL; }
    ipc->ctrl = NULL;
    ipc->in_ring = NULL;
    ipc->out_ring = NULL;
}

int shm_ipc_open(shm_ipc_t *ipc, const char *name_suffix) {
    memset(ipc, 0, sizeof(*ipc));

    char shm_name[256];
    make_name(shm_name, sizeof(shm_name), SHM_PREFIX, name_suffix);

    ipc->mapping = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, shm_name);
    if (!ipc->mapping)
        return -1;

    /* Map just the control block first to read total_size */
    void *peek = MapViewOfFile(ipc->mapping, FILE_MAP_ALL_ACCESS, 0, 0, SHM_CTRL_SIZE);
    if (!peek) {
        CloseHandle(ipc->mapping);
        ipc->mapping = NULL;
        return -2;
    }

    shm_control_t *ctrl = (shm_control_t *)peek;
    if (ctrl->magic != SHM_MAGIC || ctrl->version != SHM_VERSION) {
        UnmapViewOfFile(peek);
        CloseHandle(ipc->mapping);
        ipc->mapping = NULL;
        return -3;
    }

    uint32_t total = ctrl->total_size;
    UnmapViewOfFile(peek);

    /* Remap full size */
    ipc->base = MapViewOfFile(ipc->mapping, FILE_MAP_ALL_ACCESS, 0, 0, total);
    if (!ipc->base) {
        CloseHandle(ipc->mapping);
        ipc->mapping = NULL;
        return -4;
    }

    ipc->ctrl = (shm_control_t *)ipc->base;
    ipc->in_ring = (unsigned char *)ipc->base + ipc->ctrl->in_ring_offset;
    ipc->out_ring = (unsigned char *)ipc->base + ipc->ctrl->out_ring_offset;

    /* Open events */
    char evt_name[256];
    make_name(evt_name, sizeof(evt_name), EVT_INPUT, name_suffix);
    ipc->input_event = OpenEventA(EVENT_ALL_ACCESS, FALSE, evt_name);

    make_name(evt_name, sizeof(evt_name), EVT_STOP, name_suffix);
    ipc->stop_event = OpenEventA(EVENT_ALL_ACCESS, FALSE, evt_name);

    if (!ipc->input_event || !ipc->stop_event) {
        shm_ipc_close(ipc);
        return -5;
    }

    return 0;
}

void shm_ipc_close(shm_ipc_t *ipc) {
    if (!ipc) return;
    if (ipc->base) { UnmapViewOfFile(ipc->base); ipc->base = NULL; }
    if (ipc->mapping) { CloseHandle(ipc->mapping); ipc->mapping = NULL; }
    if (ipc->input_event) { CloseHandle(ipc->input_event); ipc->input_event = NULL; }
    if (ipc->stop_event) { CloseHandle(ipc->stop_event); ipc->stop_event = NULL; }
    ipc->ctrl = NULL;
    ipc->in_ring = NULL;
    ipc->out_ring = NULL;
}
