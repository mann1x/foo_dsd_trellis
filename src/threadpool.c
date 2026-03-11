/*
 * foo_dsd_trellis — Thread pool for per-channel parallel processing
 *
 * Windows thread pool with:
 *   - Worker threads via CreateThread
 *   - CRITICAL_SECTION-protected work queue
 *   - Semaphore to wake workers
 *   - Atomic completion counter + event for batch synchronization
 *   - Optional CPU affinity via SetThreadAffinityMask
 */

#include "../include/threadpool.h"
#include <stdlib.h>
#include <string.h>

#define MAX_QUEUE_SIZE 256

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
};

/* ─── Worker thread function ─── */

static DWORD WINAPI worker_func(LPVOID param) {
    threadpool_t *pool = (threadpool_t *)param;

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

        /* Process the block */
        block->out_count = engine_process_block(block->eng, block->in,
                                                 block->out, block->count,
                                                 block->cfg);

        /* Decrement pending counter; if zero, signal done */
        if (InterlockedDecrement(&pool->pending) == 0)
            SetEvent(pool->done_event);
    }

    return 0;
}

/* ─── Public API ─── */

threadpool_t *threadpool_create(int thread_count, DWORD affinity_mask) {
    threadpool_t *pool = (threadpool_t *)calloc(1, sizeof(threadpool_t));
    if (!pool)
        return NULL;

    if (thread_count <= 0) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        thread_count = (int)si.dwNumberOfProcessors / 2;
        if (thread_count < 1) thread_count = 1;
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

    /* Create worker threads */
    pool->threads = (HANDLE *)calloc((size_t)thread_count, sizeof(HANDLE));
    if (!pool->threads) {
        CloseHandle(pool->done_event);
        CloseHandle(pool->work_sem);
        DeleteCriticalSection(&pool->queue_cs);
        free(pool);
        return NULL;
    }

    for (int i = 0; i < thread_count; i++) {
        pool->threads[i] = CreateThread(NULL, 0, worker_func, pool, 0, NULL);
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
    CloseHandle(pool->done_event);
    CloseHandle(pool->work_sem);
    DeleteCriticalSection(&pool->queue_cs);
    free(pool);
}

int threadpool_get_thread_count(threadpool_t *pool) {
    return pool ? pool->thread_count : 0;
}
