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
 * If thread_count == 0, uses logical_cpu_count.
 * affinity_mask: per-worker CPU affinity (0 = OS default).
 * Returns NULL on failure. */
threadpool_t *threadpool_create(int thread_count, DWORD affinity_mask);

/* Create thread pool with CPUSET-based thread placement.
 * Each worker is pinned to the corresponding cpuset_id.
 * cpuset_ids: array of CPU set IDs (one per worker thread).
 * cpuset_count: number of IDs (= number of worker threads).
 * Returns NULL on failure. */
threadpool_t *threadpool_create_cpuset(const uint32_t *cpuset_ids, int cpuset_count);

/* Submit a channel block for processing.
 * Non-blocking; the block is queued for a worker thread. */
int threadpool_submit(threadpool_t *pool, channel_block_t *block);

/* Submit multiple blocks and wake all workers simultaneously.
 * Returns number of blocks submitted. */
int threadpool_submit_batch(threadpool_t *pool, channel_block_t **blocks, int count);

/* Wait for all submitted blocks in the current batch to complete. */
void threadpool_wait(threadpool_t *pool);

/* Destroy thread pool, joining all worker threads. */
void threadpool_destroy(threadpool_t *pool);

/* Get number of active worker threads. */
int threadpool_get_thread_count(threadpool_t *pool);

/* Check if any worker reported RT stress after the last batch.
 * Returns the index of the most stressed thread, or -1 if none.
 * stressed_ratio receives the worst RT ratio. */
int threadpool_get_stressed_thread(threadpool_t *pool, double *stressed_ratio);

/* Migrate a specific worker thread to a new CPU set ID.
 * thread_index: 0-based index of the worker to migrate.
 * new_cpuset_id: the CPU set ID to assign. */
int threadpool_migrate_thread(threadpool_t *pool, int thread_index, uint32_t new_cpuset_id);

#endif /* THREADPOOL_H */
