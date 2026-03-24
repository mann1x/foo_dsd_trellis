/*
 * foo_dsd_trellis — Thread pool for per-channel parallel processing
 *
 * Windows thread pool with:
 *   - Worker threads via CreateThread
 *   - CRITICAL_SECTION-protected work queue
 *   - Semaphore to wake workers
 *   - Atomic completion counter + event for batch synchronization
 *   - Optional CPU affinity via SetThreadAffinityMask
 *   - MMCSS "Pro Audio" scheduling for real-time priority
 */

#include "../include/threadpool.h"
#include "../include/dop.h"

extern void trellis_log_c(const char *msg);
#include <stdio.h>

#ifndef DOP_BITS_PER_FRAME
#define DOP_BITS_PER_FRAME 16
#endif
#include <stdlib.h>
#include <string.h>

#define MAX_QUEUE_SIZE 256

/* ─── MMCSS (Multimedia Class Scheduler Service) ─── */

typedef HANDLE (WINAPI *PFN_AvSetMmThreadCharacteristicsW)(
    LPCWSTR TaskName, LPDWORD TaskIndex);
typedef BOOL (WINAPI *PFN_AvRevertMmThreadCharacteristics)(HANDLE AvrtHandle);
typedef BOOL (WINAPI *PFN_AvSetMmThreadPriority)(
    HANDLE AvrtHandle, int Priority);

/* AVRT_PRIORITY_HIGH = 2 (from avrt.h) */
#define MMCSS_PRIORITY_HIGH 2

static HMODULE              g_avrt_dll = NULL;
static PFN_AvSetMmThreadCharacteristicsW g_pfn_set_mmthread = NULL;
static PFN_AvRevertMmThreadCharacteristics g_pfn_revert_mmthread = NULL;
static PFN_AvSetMmThreadPriority g_pfn_set_mm_priority = NULL;

static void mmcss_init(void) {
    if (g_avrt_dll)
        return;
    g_avrt_dll = LoadLibraryW(L"avrt.dll");
    if (!g_avrt_dll)
        return;
    g_pfn_set_mmthread = (PFN_AvSetMmThreadCharacteristicsW)
        GetProcAddress(g_avrt_dll, "AvSetMmThreadCharacteristicsW");
    g_pfn_revert_mmthread = (PFN_AvRevertMmThreadCharacteristics)
        GetProcAddress(g_avrt_dll, "AvRevertMmThreadCharacteristics");
    g_pfn_set_mm_priority = (PFN_AvSetMmThreadPriority)
        GetProcAddress(g_avrt_dll, "AvSetMmThreadPriority");
}

/* Register calling thread with MMCSS "Pro Audio" task.
 * Returns MMCSS handle (or NULL if unavailable). */
static HANDLE mmcss_register(void) {
    if (!g_pfn_set_mmthread)
        return NULL;
    DWORD task_index = 0;
    HANDLE h = g_pfn_set_mmthread(L"Pro Audio", &task_index);
    if (h && g_pfn_set_mm_priority)
        g_pfn_set_mm_priority(h, MMCSS_PRIORITY_HIGH);
    return h;
}

static void mmcss_unregister(HANDLE h) {
    if (h && g_pfn_revert_mmthread)
        g_pfn_revert_mmthread(h);
}

#define PER_WORKER_QUEUE_SIZE 8
#define RT_WINDOW_SIZE 5

/* Per-worker direct queue for targeted task assignment.
 * Padded to 128 bytes (2 cache lines) to prevent false sharing
 * between adjacent workers' volatile count fields. */
typedef struct __declspec(align(128)) {
    channel_block_t *queue[PER_WORKER_QUEUE_SIZE];
    volatile LONG    count;
    HANDLE           wake_event;  /* Auto-reset event for targeted wakeup */
} worker_queue_t;

/* Per-worker rolling RT% window for stress detection */
typedef struct {
    double   samples[RT_WINDOW_SIZE];  /* circular buffer */
    int      pos;                       /* next write position */
    int      count;                     /* filled entries (0..RT_WINDOW_SIZE) */
} rt_window_t;

struct threadpool {
    int          thread_count;
    DWORD        affinity_mask;

    /* Worker threads */
    HANDLE      *threads;

    /* Shared work queue (ring buffer, protected by critical section) */
    CRITICAL_SECTION queue_cs;
    channel_block_t *queue[MAX_QUEUE_SIZE];
    int              queue_head;    /* Next item to dequeue */
    int              queue_tail;    /* Next slot to enqueue */
    int              queue_count;   /* Items in queue */

    /* Per-worker direct queues (for submit_to) */
    worker_queue_t  *worker_queues; /* [thread_count] */

    /* Synchronization */
    HANDLE       work_sem;          /* Signaled when work available (shared queue) */
    HANDLE       done_event;        /* Signaled when batch complete */
    volatile LONG pending;          /* Items remaining in current batch */
    volatile LONG shutdown;         /* Non-zero = workers should exit */

    /* Per-thread RT stress tracking */
    double      *last_rt_ratio;     /* [thread_count] last RT ratio per worker */
    rt_window_t *rt_windows;        /* [thread_count] rolling RT% windows */
    uint32_t    *cpuset_ids;        /* [thread_count] current CPU set IDs */
    uint8_t     *lp_indices;        /* [thread_count] logical processor indices */
    uint16_t    *groups;            /* [thread_count] processor groups */
    int          cpuset_id_count;
    volatile LONG reset_log_count;  /* set to 1 to reset worker log counter */

};

/* SetThreadSelectedCpuSets — loaded dynamically (Windows 10+) */
typedef BOOL (WINAPI *PFN_SetThreadSelectedCpuSets)(
    HANDLE Thread, const ULONG *CpuSetIds, ULONG CpuSetIdCount);

/* SetThreadGroupAffinity — for hard affinity on >64 CPU systems */
typedef BOOL (WINAPI *PFN_SetThreadGroupAffinity)(
    HANDLE hThread, const GROUP_AFFINITY *GroupAffinity,
    GROUP_AFFINITY *PreviousGroupAffinity);

/* Per-worker context passed to the thread function */
typedef struct {
    threadpool_t *pool;
    int           thread_index;
} worker_context_t;

/* ─── Worker thread function ─── */

static DWORD WINAPI worker_func(LPVOID param) {
    worker_context_t *ctx = (worker_context_t *)param;
    threadpool_t *pool = ctx->pool;
    int my_index = ctx->thread_index;
    free(ctx);  /* context was heap-allocated */

    /* MMCSS disabled — its scheduling constraints halve performance
     * (1.57x RT with MMCSS vs 0.77x without, both with hard pin).
     * THREAD_PRIORITY_HIGHEST provides sufficient priority elevation. */
    HANDLE mmcss_handle = NULL;

    /* Hard-pin this thread to its designated LP.
     * SetThreadAffinityMask is a hard constraint the OS cannot override. */
    if (pool->lp_indices && my_index < pool->cpuset_id_count) {
        DWORD_PTR mask = (DWORD_PTR)1 << pool->lp_indices[my_index];
        SetThreadAffinityMask(GetCurrentThread(), mask);
    }

    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

    /* Build wait handles: [0] = per-worker event, [1] = shared semaphore */
    worker_queue_t *my_queue = (pool->worker_queues) ? &pool->worker_queues[my_index] : NULL;
    HANDLE wait_handles[2];
    int n_handles = 0;
    if (my_queue) wait_handles[n_handles++] = my_queue->wake_event;
    wait_handles[n_handles++] = pool->work_sem;

    for (;;) {
        /* Wait for per-worker task, shared task, or shutdown */
        WaitForMultipleObjects((DWORD)n_handles, wait_handles, FALSE, INFINITE);

        if (InterlockedCompareExchange(&pool->shutdown, 0, 0))
            break;

        /* Check per-worker queue first (targeted tasks) */
        channel_block_t *block = NULL;
        if (my_queue && InterlockedCompareExchange(&my_queue->count, 0, 0) > 0) {
            LONG idx = InterlockedDecrement(&my_queue->count);
            if (idx >= 0)
                block = my_queue->queue[idx];
        }

        /* Fall back to shared queue */
        if (!block) {
            EnterCriticalSection(&pool->queue_cs);
            if (pool->queue_count > 0) {
                block = pool->queue[pool->queue_head];
                pool->queue_head = (pool->queue_head + 1) % MAX_QUEUE_SIZE;
                pool->queue_count--;
            }
            LeaveCriticalSection(&pool->queue_cs);
        }

        if (!block)
            continue;

        /* Log task: what the worker is doing and which core it runs on */
        {
            static volatile LONG s_log_count = 0;
            if (pool->reset_log_count) {
                InterlockedExchange(&s_log_count, 0);
                InterlockedExchange(&pool->reset_log_count, 0);
            }
            if (InterlockedIncrement(&s_log_count) <= 40) {
                DWORD cpu = GetCurrentProcessorNumber();
                const char *job = "Process";
                switch (block->mode) {
                case BLOCK_MODE_SDM:    job = "SDM";     break;
                case BLOCK_MODE_FIR:    job = "FIR";     break;
                case BLOCK_MODE_UNPACK: job = "Unpack";  break;
                case BLOCK_MODE_PACK:   job = "Pack";    break;
                case BLOCK_MODE_FULL:     job = "Full";     break;
                case BLOCK_MODE_ESTIMATE: job = "Estimate"; break;
                }
                char msg[128];
                sprintf_s(msg, sizeof(msg),
                    "%s (ch %d) on LP%u [w%d, %zu samples]",
                    job, block->channel, cpu, my_index, block->count);
                trellis_log_c(msg);
            }
        }

        /* Process the block based on mode, with RT headroom measurement */
        LARGE_INTEGER t_start, t_end, freq;
        QueryPerformanceCounter(&t_start);

        if (block->mode == BLOCK_MODE_SDM) {
            block->out_count = sdm_segment_process(block->sdm_ctx,
                                                     block->in, block->out,
                                                     block->count, block->discard);
        } else if (block->mode == BLOCK_MODE_FIR) {
            block->out_count = engine_process_fir_gain(block->eng,
                                                        block->in_f32, block->count,
                                                        block->cfg, &block->fir_out);
        } else if (block->mode == BLOCK_MODE_UNPACK) {
            /* DoP unpack: extract this channel from interleaved PCM.
             * Uses TLS buffer to avoid per-call malloc/free. */
            int ch = block->channel;
            int nch = block->num_channels;
            size_t frames = block->pcm_frames;
            const float *pcm = block->pcm_interleaved;
            float *dsd = block->dsd_channel;
            static __declspec(thread) float *tls_unpack = NULL;
            static __declspec(thread) size_t tls_unpack_sz = 0;
            if (tls_unpack_sz < frames) {
                free(tls_unpack);
                tls_unpack = (float *)malloc(frames * sizeof(float));
                tls_unpack_sz = tls_unpack ? frames : 0;
            }
            if (tls_unpack) {
                for (size_t f = 0; f < frames; f++)
                    tls_unpack[f] = pcm[f * (size_t)nch + (size_t)ch];
                dop_unpack(tls_unpack, dsd, frames);
            }
            block->out_count = frames * DOP_BITS_PER_FRAME;
        } else if (block->mode == BLOCK_MODE_PACK) {
            /* DoP pack: pack this channel's DSD into interleaved i24 */
            int ch = block->channel;
            int nch = block->num_channels;
            size_t dsd_count = block->count;
            size_t pcm_frames = dsd_count / DOP_BITS_PER_FRAME;
            uint8_t *i24_temp = (uint8_t *)block->pcm_temp;
            dop_pack_i24(block->dsd_channel, i24_temp, dsd_count, (int)block->discard);
            uint8_t *out_i24 = (uint8_t *)block->pcm_interleaved;
            for (size_t f = 0; f < pcm_frames; f++) {
                size_t dst = (f * (size_t)nch + (size_t)ch) * 3;
                size_t src = f * 3;
                out_i24[dst]     = i24_temp[src];
                out_i24[dst + 1] = i24_temp[src + 1];
                out_i24[dst + 2] = i24_temp[src + 2];
            }
            block->out_count = pcm_frames;
        } else if (block->mode == BLOCK_MODE_ESTIMATE) {
            /* State estimation: nc=1 greedy SDM pre-pass */
            const ntf_filter_t *f = block->est_filter;
            if (f) {
                double state[MAX_NTF_ORDER];
                for (int i = 0; i < f->order; i++)
                    state[i] = block->est_init[i];
                const double *data = block->in;
                size_t pos = 0;
                for (int seg = 1; seg < block->est_num_segs; seg++) {
                    size_t boundary = block->est_boundaries[seg];
                    if (boundary > pos && boundary <= block->count) {
                        sdm_estimate_state(f, state, data + pos,
                                           boundary - pos,
                                           block->est_state_limit, state);
                        pos = boundary;
                    }
                    for (int i = 0; i < f->order; i++)
                        block->est_result[seg][i] = state[i];
                }
            }
            block->out_count = 0;
        } else {
            block->out_count = engine_process_block(block->eng, block->in,
                                                     block->out, block->count,
                                                     block->cfg);
        }

        QueryPerformanceCounter(&t_end);
        QueryPerformanceFrequency(&freq);

        /* Compute RT ratio: how much of the real-time budget was consumed. */
        {
            uint32_t fs_out = (block->cfg)
                ? (block->cfg->fs_out ? block->cfg->fs_out : block->cfg->fs_in)
                : 0;
            /* For PACK mode, out_count is DoP PCM frames (rate = fs_out/16).
             * For UNPACK mode, out_count is DSD samples (rate = fs_out). */
            double rate = (double)fs_out;
            if (block->mode == BLOCK_MODE_PACK && fs_out > 0)
                rate = (double)(fs_out / DOP_BITS_PER_FRAME);
            double audio_sec = (rate > 0.0 && block->out_count > 0)
                ? (double)block->out_count / rate
                : 0.0;
            double proc_sec = (double)(t_end.QuadPart - t_start.QuadPart) / (double)freq.QuadPart;
            block->rt_ratio = (audio_sec > 0.0) ? proc_sec / audio_sec : 0.0;
            block->stressed = (block->rt_ratio > 0.7);

            /* Record per-thread stress and which worker processed this block */
            block->worker_index = my_index;
            if (my_index >= 0 && my_index < pool->thread_count) {
                if (pool->last_rt_ratio)
                    pool->last_rt_ratio[my_index] = block->rt_ratio;
                /* Push to rolling window (only for SDM blocks — the heavy work) */
                if (pool->rt_windows && block->mode == BLOCK_MODE_SDM) {
                    rt_window_t *w = &pool->rt_windows[my_index];
                    w->samples[w->pos] = block->rt_ratio;
                    w->pos = (w->pos + 1) % RT_WINDOW_SIZE;
                    if (w->count < RT_WINDOW_SIZE)
                        w->count++;
                }
            }
        }

        /* Decrement pending counter; if zero, signal done */
        if (InterlockedDecrement(&pool->pending) == 0)
            SetEvent(pool->done_event);
    }

    mmcss_unregister(mmcss_handle);
    return 0;
}

/* ─── Public API ─── */

threadpool_t *threadpool_create(int thread_count, DWORD affinity_mask) {
    mmcss_init();

    threadpool_t *pool = (threadpool_t *)calloc(1, sizeof(threadpool_t));
    if (!pool)
        return NULL;

    if (thread_count <= 0) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        thread_count = (int)si.dwNumberOfProcessors;
        if (thread_count < 2) thread_count = 2;
    }

    pool->thread_count = thread_count;
    pool->affinity_mask = affinity_mask;

    InitializeCriticalSection(&pool->queue_cs);

    /* Semaphore: initial count 0, max = queue size */
    pool->work_sem = CreateSemaphoreW(NULL, 0, MAX_QUEUE_SIZE, NULL);
    if (!pool->work_sem) {
        DeleteCriticalSection(&pool->queue_cs);
        free(pool);
        return NULL;
    }

    /* Manual-reset event for batch completion */
    pool->done_event = CreateEventW(NULL, TRUE, TRUE, NULL);
    if (!pool->done_event) {
        CloseHandle(pool->work_sem);
        DeleteCriticalSection(&pool->queue_cs);
        free(pool);
        return NULL;
    }

    /* Create worker threads + tracking arrays */
    pool->threads = (HANDLE *)calloc((size_t)thread_count, sizeof(HANDLE));
    pool->last_rt_ratio = (double *)calloc((size_t)thread_count, sizeof(double));
    pool->rt_windows = (rt_window_t *)calloc((size_t)thread_count, sizeof(rt_window_t));
    pool->worker_queues = (worker_queue_t *)calloc((size_t)thread_count, sizeof(worker_queue_t));
    if (!pool->threads || !pool->last_rt_ratio || !pool->rt_windows || !pool->worker_queues) {
        free(pool->threads);
        free(pool->last_rt_ratio);
        free(pool->rt_windows);
        free(pool->worker_queues);
        CloseHandle(pool->done_event);
        CloseHandle(pool->work_sem);
        DeleteCriticalSection(&pool->queue_cs);
        free(pool);
        return NULL;
    }

    for (int i = 0; i < thread_count; i++) {
        pool->worker_queues[i].wake_event = CreateEventW(NULL, FALSE, FALSE, NULL);
        worker_context_t *wctx = (worker_context_t *)malloc(sizeof(worker_context_t));
        if (!wctx) break;
        wctx->pool = pool;
        wctx->thread_index = i;
        pool->threads[i] = CreateThread(NULL, 0, worker_func, wctx, 0, NULL);
        if (!pool->threads[i]) {
            InterlockedExchange(&pool->shutdown, 1);
            ReleaseSemaphore(pool->work_sem, i, NULL);
            for (int j = 0; j < i; j++) {
                WaitForSingleObject(pool->threads[j], INFINITE);
                CloseHandle(pool->threads[j]);
            }
            free(pool->threads);
            CloseHandle(pool->done_event);
            CloseHandle(pool->work_sem);
            DeleteCriticalSection(&pool->queue_cs);
            free(pool);
            return NULL;
        }

        if (affinity_mask != 0)
            SetThreadAffinityMask(pool->threads[i], (DWORD_PTR)affinity_mask);
    }

    return pool;
}

int threadpool_submit(threadpool_t *pool, channel_block_t *block) {
    EnterCriticalSection(&pool->queue_cs);

    if (pool->queue_count >= MAX_QUEUE_SIZE) {
        LeaveCriticalSection(&pool->queue_cs);
        return -1;
    }

    pool->queue[pool->queue_tail] = block;
    pool->queue_tail = (pool->queue_tail + 1) % MAX_QUEUE_SIZE;
    pool->queue_count++;

    LeaveCriticalSection(&pool->queue_cs);

    /* Increment pending count and wake a worker */
    InterlockedIncrement(&pool->pending);
    ReleaseSemaphore(pool->work_sem, 1, NULL);

    return 0;
}

int threadpool_submit_to(threadpool_t *pool, int worker_index, channel_block_t *block) {
    if (!pool || worker_index < 0 || worker_index >= pool->thread_count)
        return -1;
    if (!pool->worker_queues)
        return threadpool_submit(pool, block);  /* fallback to shared queue */

    worker_queue_t *wq = &pool->worker_queues[worker_index];
    LONG idx = InterlockedIncrement(&wq->count) - 1;
    if (idx >= PER_WORKER_QUEUE_SIZE)
        return -1;  /* per-worker queue full */
    wq->queue[idx] = block;

    InterlockedIncrement(&pool->pending);
    SetEvent(wq->wake_event);  /* wake this specific worker */
    return 0;
}

/* Submit multiple blocks at once and wake all workers simultaneously.
 * Much faster than N individual submit+wake cycles. */
int threadpool_submit_batch(threadpool_t *pool, channel_block_t **blocks, int count) {
    EnterCriticalSection(&pool->queue_cs);

    for (int i = 0; i < count; i++) {
        if (pool->queue_count >= MAX_QUEUE_SIZE) {
            LeaveCriticalSection(&pool->queue_cs);
            return i;  /* partial submit */
        }
        pool->queue[pool->queue_tail] = blocks[i];
        pool->queue_tail = (pool->queue_tail + 1) % MAX_QUEUE_SIZE;
        pool->queue_count++;
    }

    LeaveCriticalSection(&pool->queue_cs);

    /* Wake all workers at once */
    InterlockedAdd(&pool->pending, count);
    ReleaseSemaphore(pool->work_sem, count, NULL);

    return count;
}

void threadpool_wait(threadpool_t *pool) {
    if (InterlockedCompareExchange(&pool->pending, 0, 0) == 0)
        return;

    ResetEvent(pool->done_event);

    if (InterlockedCompareExchange(&pool->pending, 0, 0) == 0) {
        SetEvent(pool->done_event);
        return;
    }

    WaitForSingleObject(pool->done_event, INFINITE);
}

void threadpool_destroy(threadpool_t *pool) {
    if (!pool)
        return;

    /* Signal shutdown */
    InterlockedExchange(&pool->shutdown, 1);

    /* Wake all workers so they can see the shutdown flag */
    ReleaseSemaphore(pool->work_sem, pool->thread_count, NULL);
    if (pool->worker_queues) {
        for (int i = 0; i < pool->thread_count; i++)
            SetEvent(pool->worker_queues[i].wake_event);
    }

    /* Wait for all workers to exit */
    WaitForMultipleObjects((DWORD)pool->thread_count, pool->threads,
                           TRUE, INFINITE);

    for (int i = 0; i < pool->thread_count; i++)
        CloseHandle(pool->threads[i]);

    free(pool->threads);
    free(pool->last_rt_ratio);
    free(pool->rt_windows);
    free(pool->cpuset_ids);
    free(pool->lp_indices);
    free(pool->groups);
    if (pool->worker_queues) {
        for (int i = 0; i < pool->thread_count; i++)
            CloseHandle(pool->worker_queues[i].wake_event);
        free(pool->worker_queues);
    }
    CloseHandle(pool->done_event);
    CloseHandle(pool->work_sem);
    DeleteCriticalSection(&pool->queue_cs);
    free(pool);
}

int threadpool_get_thread_count(threadpool_t *pool) {
    return pool ? pool->thread_count : 0;
}

void threadpool_reset_log(threadpool_t *pool) {
    if (pool) InterlockedExchange(&pool->reset_log_count, 1);
}

int threadpool_get_stressed_thread(threadpool_t *pool, double *stressed_ratio) {
    if (!pool || !pool->last_rt_ratio)
        return -1;

    int worst = -1;
    double worst_ratio = 0.0;
    for (int i = 0; i < pool->thread_count; i++) {
        if (pool->last_rt_ratio[i] > worst_ratio) {
            worst_ratio = pool->last_rt_ratio[i];
            worst = i;
        }
    }

    if (stressed_ratio)
        *stressed_ratio = worst_ratio;

    return (worst_ratio > 0.7) ? worst : -1;
}

int threadpool_migrate_thread(threadpool_t *pool, int thread_index, uint32_t new_cpuset_id) {
    if (!pool || thread_index < 0 || thread_index >= pool->thread_count)
        return -1;
    if (!pool->threads[thread_index])
        return -1;

    /* We need the LP index for SetThreadAffinityMask.
     * The cpuset_id encodes the LP (id = base + lp_index on most systems).
     * Compute LP from the offset to the first known cpuset_id. */
    uint8_t new_lp = 0;
    if (pool->cpuset_ids && pool->lp_indices && pool->cpuset_id_count > 0) {
        /* Find LP index for this cpuset_id by offset from known base */
        uint32_t base_id = pool->cpuset_ids[0];
        uint8_t base_lp = pool->lp_indices[0];
        new_lp = (uint8_t)((int)base_lp + ((int)new_cpuset_id - (int)base_id));
    }

    /* Hard affinity — check if multi-group */
    bool multi_group = false;
    if (pool->groups) {
        for (int i = 0; i < pool->cpuset_id_count; i++) {
            if (pool->groups[i] > 0) { multi_group = true; break; }
        }
    }

    if (multi_group) {
        PFN_SetThreadGroupAffinity pfn_ga = (PFN_SetThreadGroupAffinity)
            GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadGroupAffinity");
        if (pfn_ga) {
            /* Find group for this cpuset_id (search known entries, fallback to 0) */
            uint16_t grp = 0;
            for (int i = 0; i < pool->cpuset_id_count; i++) {
                if (pool->cpuset_ids[i] == new_cpuset_id) {
                    grp = pool->groups[i];
                    break;
                }
            }
            GROUP_AFFINITY ga;
            memset(&ga, 0, sizeof(ga));
            ga.Group = grp;
            ga.Mask = (KAFFINITY)1 << new_lp;
            pfn_ga(pool->threads[thread_index], &ga, NULL);
        }
    } else {
        DWORD_PTR mask = (DWORD_PTR)1 << new_lp;
        SetThreadAffinityMask(pool->threads[thread_index], mask);
    }

    /* Update tracked cpuset ID and LP index */
    if (pool->cpuset_ids && thread_index < pool->cpuset_id_count)
        pool->cpuset_ids[thread_index] = new_cpuset_id;
    if (pool->lp_indices && thread_index < pool->cpuset_id_count)
        pool->lp_indices[thread_index] = new_lp;

    return 0;
}

/* ─── Rolling RT% stress detection ─── */

worker_stress_level_t threadpool_get_worker_stress(threadpool_t *pool, int worker_index) {
    if (!pool || !pool->rt_windows || worker_index < 0 || worker_index >= pool->thread_count)
        return WORKER_HEALTHY;

    rt_window_t *w = &pool->rt_windows[worker_index];
    if (w->count == 0)
        return WORKER_HEALTHY;

    /* CRITICAL: any of the last 1 entry > 1.0 */
    int newest = (w->pos + RT_WINDOW_SIZE - 1) % RT_WINDOW_SIZE;
    if (w->samples[newest] > 1.0)
        return WORKER_CRITICAL;

    /* WARN: last 3+ consecutive entries all > 0.9 */
    if (w->count >= 3) {
        int warn_count = 0;
        for (int i = 0; i < w->count && i < RT_WINDOW_SIZE; i++) {
            int idx = (w->pos + RT_WINDOW_SIZE - 1 - i) % RT_WINDOW_SIZE;
            if (w->samples[idx] > 0.9)
                warn_count++;
            else
                break;  /* must be consecutive */
        }
        if (warn_count >= 3)
            return WORKER_WARN;
    }

    return WORKER_HEALTHY;
}

double threadpool_get_worker_avg_rt(threadpool_t *pool, int worker_index) {
    if (!pool || !pool->rt_windows || worker_index < 0 || worker_index >= pool->thread_count)
        return 0.0;

    rt_window_t *w = &pool->rt_windows[worker_index];
    if (w->count == 0)
        return 0.0;

    double sum = 0.0;
    for (int i = 0; i < w->count; i++)
        sum += w->samples[i];
    return sum / (double)w->count;
}

uint32_t threadpool_get_worker_cpuset(threadpool_t *pool, int worker_index) {
    if (!pool || !pool->cpuset_ids || worker_index < 0 || worker_index >= pool->cpuset_id_count)
        return 0;
    return pool->cpuset_ids[worker_index];
}

/* ─── CPUSET-aware thread pool creation ─── */

threadpool_t *threadpool_create_cpuset(const uint32_t *cpuset_ids,
                                       const uint8_t *lp_indices,
                                       const uint16_t *groups,
                                       int count) {
    if (!cpuset_ids || !lp_indices || count < 1)
        return NULL;

    mmcss_init();

    threadpool_t *pool = (threadpool_t *)calloc(1, sizeof(threadpool_t));
    if (!pool)
        return NULL;

    pool->thread_count = count;
    pool->affinity_mask = 0;

    InitializeCriticalSection(&pool->queue_cs);

    pool->work_sem = CreateSemaphoreW(NULL, 0, MAX_QUEUE_SIZE, NULL);
    if (!pool->work_sem) {
        DeleteCriticalSection(&pool->queue_cs);
        free(pool);
        return NULL;
    }

    pool->done_event = CreateEventW(NULL, TRUE, TRUE, NULL);
    if (!pool->done_event) {
        CloseHandle(pool->work_sem);
        DeleteCriticalSection(&pool->queue_cs);
        free(pool);
        return NULL;
    }

    pool->threads = (HANDLE *)calloc((size_t)count, sizeof(HANDLE));
    pool->last_rt_ratio = (double *)calloc((size_t)count, sizeof(double));
    pool->rt_windows = (rt_window_t *)calloc((size_t)count, sizeof(rt_window_t));
    pool->cpuset_ids = (uint32_t *)malloc((size_t)count * sizeof(uint32_t));
    pool->lp_indices = (uint8_t *)malloc((size_t)count * sizeof(uint8_t));
    pool->groups = (uint16_t *)calloc((size_t)count, sizeof(uint16_t));
    pool->worker_queues = (worker_queue_t *)calloc((size_t)count, sizeof(worker_queue_t));
    if (!pool->threads || !pool->last_rt_ratio || !pool->rt_windows ||
        !pool->cpuset_ids || !pool->lp_indices || !pool->groups || !pool->worker_queues) {
        free(pool->threads);
        free(pool->last_rt_ratio);
        free(pool->rt_windows);
        free(pool->cpuset_ids);
        free(pool->lp_indices);
        free(pool->groups);
        free(pool->worker_queues);
        CloseHandle(pool->done_event);
        CloseHandle(pool->work_sem);
        DeleteCriticalSection(&pool->queue_cs);
        free(pool);
        return NULL;
    }
    memcpy(pool->cpuset_ids, cpuset_ids, (size_t)count * sizeof(uint32_t));
    memcpy(pool->lp_indices, lp_indices, (size_t)count * sizeof(uint8_t));
    if (groups)
        memcpy(pool->groups, groups, (size_t)count * sizeof(uint16_t));
    pool->cpuset_id_count = count;

    /* Initialize per-worker wake events */
    for (int i = 0; i < count; i++)
        pool->worker_queues[i].wake_event = CreateEventW(NULL, FALSE, FALSE, NULL);

    for (int i = 0; i < count; i++) {
        worker_context_t *wctx = (worker_context_t *)malloc(sizeof(worker_context_t));
        if (!wctx) break;
        wctx->pool = pool;
        wctx->thread_index = i;

        pool->threads[i] = CreateThread(NULL, 0, worker_func, wctx, 0, NULL);
        if (!pool->threads[i]) {
            free(wctx);
            InterlockedExchange(&pool->shutdown, 1);
            ReleaseSemaphore(pool->work_sem, i, NULL);
            for (int j = 0; j < i; j++) {
                WaitForSingleObject(pool->threads[j], INFINITE);
                CloseHandle(pool->threads[j]);
            }
            free(pool->threads);
            free(pool->last_rt_ratio);
            free(pool->rt_windows);
            free(pool->cpuset_ids);
            free(pool->lp_indices);
            free(pool->groups);
            free(pool->worker_queues);
            CloseHandle(pool->done_event);
            CloseHandle(pool->work_sem);
            DeleteCriticalSection(&pool->queue_cs);
            free(pool);
            return NULL;
        }
        /* Hard affinity is set inside worker_func via SetThreadAffinityMask */
    }

    return pool;
}
