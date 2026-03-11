/*
 * foo_dsd_trellis — Lock-free MPMC thread pool
 */

#ifndef THREADPOOL_H
#define THREADPOOL_H

#include "dsd_types.h"
#include "engine.h"

/* Opaque thread pool handle */
typedef struct threadpool threadpool_t;

/* Create thread pool with given worker count.
 * If thread_count == 0, uses logical_cpu_count / 2.
 * affinity_mask: per-worker CPU affinity (0 = OS default).
 * Returns NULL on failure. */
threadpool_t *threadpool_create(int thread_count, DWORD affinity_mask);

/* Submit a channel block for processing.
 * Non-blocking; the block is queued for a worker thread. */
int threadpool_submit(threadpool_t *pool, channel_block_t *block);

/* Wait for all submitted blocks in the current batch to complete. */
void threadpool_wait(threadpool_t *pool);

/* Destroy thread pool, joining all worker threads. */
void threadpool_destroy(threadpool_t *pool);

/* Get number of active worker threads. */
int threadpool_get_thread_count(threadpool_t *pool);

#endif /* THREADPOOL_H */
