/*
 * foo_dsd_trellis — Lock-free MPMC thread pool
 *
 * Phase 0: Scaffold — single-threaded stub.
 * Phase 5 will implement lock-free queue and worker threads.
 */

#include "../include/threadpool.h"
#include <stdlib.h>
#include <string.h>

#define MAX_QUEUE_SIZE 256

struct threadpool {
    int          thread_count;
    DWORD        affinity_mask;
    /* TODO (Phase 5): HANDLE threads[], MPMC ring buffer, semaphores */

    /* Temporary: synchronous queue for Phase 0 */
    channel_block_t *pending[MAX_QUEUE_SIZE];
    int              pending_count;
};

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

    /* TODO (Phase 5): Create worker threads */

    return pool;
}

int threadpool_submit(threadpool_t *pool, channel_block_t *block) {
    if (pool->pending_count >= MAX_QUEUE_SIZE)
        return -1;
    pool->pending[pool->pending_count++] = block;
    return 0;
}

void threadpool_wait(threadpool_t *pool) {
    /* Phase 0: Process synchronously on calling thread */
    for (int i = 0; i < pool->pending_count; i++) {
        channel_block_t *b = pool->pending[i];
        b->out_count = engine_process_block(b->eng, b->in, b->out,
                                            b->count, b->cfg);
    }
    pool->pending_count = 0;
}

void threadpool_destroy(threadpool_t *pool) {
    if (!pool)
        return;
    /* TODO (Phase 5): Signal workers to exit, join threads */
    free(pool);
}

int threadpool_get_thread_count(threadpool_t *pool) {
    return pool ? pool->thread_count : 0;
}
