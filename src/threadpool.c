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

struct threadpool {
    int          thread_count;
    DWORD        affinity_mask;

    /* Worker threads */
    HANDLE      *threads;

    /* Work queue (ring buffer, protected by critical section) */
    CRITICAL_SECTION queue_cs;
    channel_block_t *queue[MAX_QUEUE_SIZE];
    int              queue_head;    /* Next item to dequeue */
    int              queue_tail;    /* Next slot to enqueue */
    int              queue_count;   /* Items in queue */

    /* Synchronization */
    HANDLE       work_sem;          /* Signaled when work available */
    HANDLE       done_event;        /* Signaled when batch complete */
    volatile LONG pending;          /* Items remaining in current batch */
    volatile LONG shutdown;         /* Non-zero = workers should exit */

    /* Per-thread RT stress tracking */
    double      *last_rt_ratio;     /* [thread_count] last RT ratio per worker */
    uint32_t    *cpuset_ids;        /* [thread_count] current CPU set IDs */
    int          cpuset_id_count;
};

/* SetThreadSelectedCpuSets — loaded dynamically (Windows 10+) */
typedef BOOL (WINAPI *PFN_SetThreadSelectedCpuSets)(
    HANDLE Thread, const ULONG *CpuSetIds, ULONG CpuSetIdCount);

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

    /* Register with MMCSS for real-time audio scheduling */
    HANDLE mmcss_handle = mmcss_register();

    for (;;) {
        /* Wait for work or shutdown */
        WaitForSingleObject(pool->work_sem, INFINITE);

        if (InterlockedCompareExchange(&pool->shutdown, 0, 0))
            break;

        /* Dequeue a work item */
        channel_block_t *block = NULL;

        EnterCriticalSection(&pool->queue_cs);
        if (pool->queue_count > 0) {
            block = pool->queue[pool->queue_head];
            pool->queue_head = (pool->queue_head + 1) % MAX_QUEUE_SIZE;
            pool->queue_count--;
        }
        LeaveCriticalSection(&pool->queue_cs);

        if (!block)
            continue;

        /* Process the block based on mode, with RT headroom measurement */
        LARGE_INTEGER t_start, t_end, freq;
        QueryPerformanceCounter(&t_start);

        if (block->mode == BLOCK_MODE_SDM) {
            block->out_count = sdm_segment_process(block->sdm_ctx,
                                                     block->in, block->out,
                                                     block->count, block->discard);
        } else if (block->mode == BLOCK_MODE_FIR) {
            block->out_count = engine_process_fir_gain(block->eng,
                                                        block->in, block->count,
                                                        block->cfg, &block->fir_out);
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
            double audio_sec = (fs_out > 0 && block->out_count > 0)
                ? (double)block->out_count / (double)fs_out
                : 0.0;
            double proc_sec = (double)(t_end.QuadPart - t_start.QuadPart) / (double)freq.QuadPart;
            block->rt_ratio = (audio_sec > 0.0) ? proc_sec / audio_sec : 0.0;
            block->stressed = (block->rt_ratio > 0.7);

            /* Record per-thread stress and which worker processed this block */
            block->worker_index = my_index;
            if (my_index >= 0 && my_index < pool->thread_count && pool->last_rt_ratio)
                pool->last_rt_ratio[my_index] = block->rt_ratio;
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
    if (!pool->threads || !pool->last_rt_ratio) {
        free(pool->threads);
        free(pool->last_rt_ratio);
        CloseHandle(pool->done_event);
        CloseHandle(pool->work_sem);
        DeleteCriticalSection(&pool->queue_cs);
        free(pool);
        return NULL;
    }

    for (int i = 0; i < thread_count; i++) {
        worker_context_t *wctx = (worker_context_t *)malloc(sizeof(worker_context_t));
        if (!wctx) break;
        wctx->pool = pool;
        wctx->thread_index = i;
        pool->threads[i] = CreateThread(NULL, 0, worker_func, wctx, 0, NULL);
        if (!pool->threads[i]) {
            /* Cleanup already-created threads */
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

        /* Set CPU affinity if requested */
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

void threadpool_wait(threadpool_t *pool) {
    /* If nothing was submitted, return immediately */
    if (InterlockedCompareExchange(&pool->pending, 0, 0) == 0)
        return;

    /* Reset done event before waiting (it's manual-reset) */
    ResetEvent(pool->done_event);

    /* Check again after reset to avoid race */
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

    /* Wait for all workers to exit */
    WaitForMultipleObjects((DWORD)pool->thread_count, pool->threads,
                           TRUE, INFINITE);

    for (int i = 0; i < pool->thread_count; i++)
        CloseHandle(pool->threads[i]);

    free(pool->threads);
    free(pool->last_rt_ratio);
    free(pool->cpuset_ids);
    CloseHandle(pool->done_event);
    CloseHandle(pool->work_sem);
    DeleteCriticalSection(&pool->queue_cs);
    free(pool);
}

int threadpool_get_thread_count(threadpool_t *pool) {
    return pool ? pool->thread_count : 0;
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

    PFN_SetThreadSelectedCpuSets pfn_set_cpusets =
        (PFN_SetThreadSelectedCpuSets)GetProcAddress(
            GetModuleHandleW(L"kernel32.dll"), "SetThreadSelectedCpuSets");
    if (!pfn_set_cpusets)
        return -1;

    ULONG id = new_cpuset_id;
    if (!pfn_set_cpusets(pool->threads[thread_index], &id, 1))
        return -1;

    /* Update tracked cpuset ID */
    if (pool->cpuset_ids && thread_index < pool->cpuset_id_count)
        pool->cpuset_ids[thread_index] = new_cpuset_id;

    return 0;
}

/* ─── CPUSET-aware thread pool creation ─── */

threadpool_t *threadpool_create_cpuset(const uint32_t *cpuset_ids, int cpuset_count) {
    if (!cpuset_ids || cpuset_count < 1)
        return NULL;

    mmcss_init();

    threadpool_t *pool = (threadpool_t *)calloc(1, sizeof(threadpool_t));
    if (!pool)
        return NULL;

    pool->thread_count = cpuset_count;
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

    pool->threads = (HANDLE *)calloc((size_t)cpuset_count, sizeof(HANDLE));
    pool->last_rt_ratio = (double *)calloc((size_t)cpuset_count, sizeof(double));
    pool->cpuset_ids = (uint32_t *)malloc((size_t)cpuset_count * sizeof(uint32_t));
    if (!pool->threads || !pool->last_rt_ratio || !pool->cpuset_ids) {
        free(pool->threads);
        free(pool->last_rt_ratio);
        free(pool->cpuset_ids);
        CloseHandle(pool->done_event);
        CloseHandle(pool->work_sem);
        DeleteCriticalSection(&pool->queue_cs);
        free(pool);
        return NULL;
    }
    memcpy(pool->cpuset_ids, cpuset_ids, (size_t)cpuset_count * sizeof(uint32_t));
    pool->cpuset_id_count = cpuset_count;

    /* Load SetThreadSelectedCpuSets dynamically */
    PFN_SetThreadSelectedCpuSets pfn_set_cpusets =
        (PFN_SetThreadSelectedCpuSets)GetProcAddress(
            GetModuleHandleW(L"kernel32.dll"), "SetThreadSelectedCpuSets");

    for (int i = 0; i < cpuset_count; i++) {
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

        /* Pin worker to its designated CPU set */
        if (pfn_set_cpusets) {
            ULONG id = cpuset_ids[i];
            pfn_set_cpusets(pool->threads[i], &id, 1);
        }
    }

    return pool;
}
