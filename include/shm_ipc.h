/*
 * foo_dsd_trellis — Shared memory IPC for out-of-process worker
 *
 * Lock-free SPSC ring buffers over named shared memory.
 * Parent (fb2k plugin): writes input PCM, reads output i24.
 * Worker (standalone exe): reads input, processes, writes output.
 */

#ifndef SHM_IPC_H
#define SHM_IPC_H

#include "dsd_types.h"

#define SHM_MAGIC       0x54524C53  /* "TRLS" */
#define SHM_VERSION     1
#define SHM_CTRL_SIZE   4096        /* control block aligned to page */

/* Worker status */
#define SHM_STATUS_STARTING  0
#define SHM_STATUS_READY     1
#define SHM_STATUS_ERROR     2
#define SHM_STATUS_EXITING   3

/* Control block — lives at offset 0 of the shared memory mapping.
 * All fields are plain types (no pointers). Volatile fields are
 * accessed atomically via InterlockedExchange / InterlockedCompareExchange. */
typedef struct {
    /* Protocol handshake */
    uint32_t        magic;
    uint32_t        version;

    /* Audio config (parent writes at init, worker reads once) */
    dsd_config_t    config;
    int             channels;
    unsigned        pcm_rate;
    uint32_t        out_pcm_rate;
    int             batch_target;       /* pcm_rate/5 = ~200ms */

    /* Live controls (parent writes, worker reads per-batch) */
    volatile float  gain;
    volatile LONG   mute;               /* 0 or 1 */
    volatile LONG   config_version;     /* incremented on config change */

    /* Input ring (parent produces, worker consumes) */
    volatile LONG   in_write_pos;       /* byte offset, monotonic */
    volatile LONG   in_read_pos;
    int             in_capacity;        /* power of 2 */

    /* Output ring (worker produces, parent consumes) */
    volatile LONG   out_write_pos;
    volatile LONG   out_read_pos;
    int             out_capacity;

    /* Worker status */
    volatile LONG   worker_status;
    volatile LONG   primed;             /* 1 = ring has >= 2x batch */

    /* Parent process affinity mask — worker marks these as host_thread */
    uint64_t        parent_affinity;
    uint64_t        system_affinity;

    /* Ring data offsets from start of mapping */
    uint32_t        in_ring_offset;
    uint32_t        out_ring_offset;

    /* Total mapping size (for validation) */
    uint32_t        total_size;

    /* Log ring (worker → DLL, SPSC) */
    volatile LONG   log_write_pos;
    volatile LONG   log_read_pos;
    uint32_t        log_ring_offset;
    int             log_ring_capacity;     /* power of 2, 0 = disabled */
} shm_control_t;

#define SHM_LOG_RING_SIZE  32768   /* 32KB log ring */

/* Handle for an active IPC connection (one side) */
typedef struct {
    HANDLE          mapping;            /* CreateFileMapping handle */
    HANDLE          input_event;        /* auto-reset: input available */
    HANDLE          stop_event;         /* manual-reset: shutdown */
    void           *base;              /* MapViewOfFile base pointer */
    shm_control_t  *ctrl;              /* -> base + 0 */
    unsigned char  *in_ring;           /* -> base + in_ring_offset */
    unsigned char  *out_ring;          /* -> base + out_ring_offset */
    unsigned char  *log_ring;          /* -> base + log_ring_offset */
} shm_ipc_t;

/* ─── Ring buffer ops (inline, work on shared memory indices) ─── */

static __forceinline int shm_ring_available(volatile LONG *write_pos,
                                             volatile LONG *read_pos) {
    LONG w = InterlockedCompareExchange(write_pos, 0, 0);
    LONG r = InterlockedCompareExchange(read_pos, 0, 0);
    return (int)(w - r);
}

static __forceinline int shm_ring_free(volatile LONG *write_pos,
                                        volatile LONG *read_pos,
                                        int capacity) {
    return capacity - shm_ring_available(write_pos, read_pos);
}

static __forceinline int shm_ring_write(unsigned char *data, int capacity,
                                         volatile LONG *write_pos,
                                         volatile LONG *read_pos,
                                         const void *src, int len) {
    int avail = shm_ring_free(write_pos, read_pos, capacity);
    if (len > avail) len = avail;
    if (len <= 0) return 0;

    LONG w = InterlockedCompareExchange(write_pos, 0, 0);
    int mask = capacity - 1;
    int offset = (int)(w & mask);
    int first = capacity - offset;
    if (first > len) first = len;
    memcpy(data + offset, src, (size_t)first);
    if (first < len)
        memcpy(data, (const unsigned char *)src + first, (size_t)(len - first));
    InterlockedExchange(write_pos, w + len);
    return len;
}

static __forceinline int shm_ring_read(unsigned char *data, int capacity,
                                        volatile LONG *write_pos,
                                        volatile LONG *read_pos,
                                        void *dst, int len) {
    int avail = shm_ring_available(write_pos, read_pos);
    if (len > avail) len = avail;
    if (len <= 0) return 0;

    LONG r = InterlockedCompareExchange(read_pos, 0, 0);
    int mask = capacity - 1;
    int offset = (int)(r & mask);
    int first = capacity - offset;
    if (first > len) first = len;
    memcpy(dst, data + offset, (size_t)first);
    if (first < len)
        memcpy((unsigned char *)dst + first, data, (size_t)(len - first));
    InterlockedExchange(read_pos, r + len);
    return len;
}

/* ─── Parent-side API ─── */

/* Create shared memory + events. Returns 0 on success.
 * name_suffix: unique suffix (e.g., PID string).
 * channels, pcm_rate, out_pcm_rate: audio params.
 * config: initial DSP config. */
int shm_ipc_create(shm_ipc_t *ipc, const char *name_suffix,
                    int channels, unsigned pcm_rate, uint32_t out_pcm_rate,
                    const dsd_config_t *config);

/* Destroy shared memory + events. */
void shm_ipc_destroy(shm_ipc_t *ipc);

/* ─── Worker-side API ─── */

/* Open existing shared memory + events. Returns 0 on success. */
int shm_ipc_open(shm_ipc_t *ipc, const char *name_suffix);

/* Close mapping (worker side). Does not destroy — parent owns. */
void shm_ipc_close(shm_ipc_t *ipc);

/* ─── Helpers ─── */

/* Round up to next power of 2 */
static __forceinline int shm_next_pow2(int v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16;
    return v + 1;
}

#endif /* SHM_IPC_H */
