/*
 * foo_dsd_trellis — Lock-free MPMC thread pool
 */

#ifndef THREADPOOL_H
#define THREADPOOL_H

#include "dsd_types.h"
#include "engine.h"

/* Opaque thread pool handle */
typedef struct threadpool threadpool_t;

/* Worker stress level (based on rolling RT% window) */
typedef enum {
    WORKER_HEALTHY  = 0,   /* <70% for 5+ consecutive — no action */
    WORKER_WARN     = 1,   /* >90% for 3+ consecutive — start probe */
    WORKER_CRITICAL = 2,   /* >100% for 1 chunk — immediate probe */
} worker_stress_level_t;

/* Create thread pool with given worker count.
 * If thread_count == 0, uses logical_cpu_count.
 * affinity_mask: per-worker CPU affinity (0 = OS default).
 * Returns NULL on failure. */
threadpool_t *threadpool_create(int thread_count, DWORD affinity_mask);

/* Create thread pool with hard per-core thread pinning.
 * Each worker is pinned to the corresponding logical processor.
 * cpuset_ids: array of CPU set IDs (for tracking/logging).
 * lp_indices: array of logical processor indices (for SetThreadAffinityMask).
 * groups: array of processor groups (for >64 CPU support, NULL if single group).
 * count: number of entries (= number of worker threads).
 * Returns NULL on failure. */
threadpool_t *threadpool_create_cpuset(const uint32_t *cpuset_ids,
                                       const uint8_t *lp_indices,
                                       const uint16_t *groups,
                                       int count);

/* Set number of reserved workers.
 * Workers 0..reserved_count-1 ONLY process their per-worker queue
 * (submit_to targets). They never touch the shared queue.
 * Workers reserved_count..thread_count-1 ONLY process the shared queue.
 * This prevents shared tasks from starving pinned SDM segments.
 * Must be called after create, before any submit. */
void threadpool_set_reserved(threadpool_t *pool, int reserved_count);

/* Submit a channel block for processing.
 * Non-blocking; the block is queued for a worker thread. */
int threadpool_submit(threadpool_t *pool, channel_block_t *block);

/* Submit a task to a specific worker by index.
 * Guarantees the task runs on that worker's pinned core.
 * Use for heavy tasks (SDM) that need dedicated cores. */
int threadpool_submit_to(threadpool_t *pool, int worker_index, channel_block_t *block);

/* Submit multiple blocks and wake all workers simultaneously.
 * Returns number of blocks submitted. */
int threadpool_submit_batch(threadpool_t *pool, channel_block_t **blocks, int count);

/* Wait for all submitted blocks in the current batch to complete. */
void threadpool_wait(threadpool_t *pool);

/* Destroy thread pool, joining all worker threads. */
void threadpool_destroy(threadpool_t *pool);

/* Get number of active worker threads. */
int threadpool_get_thread_count(threadpool_t *pool);

/* Reset worker task log counter (re-enables first-N task logging). */
void threadpool_reset_log(threadpool_t *pool);

/* Check if any worker reported RT stress after the last batch.
 * Returns the index of the most stressed thread, or -1 if none.
 * stressed_ratio receives the worst RT ratio. */
int threadpool_get_stressed_thread(threadpool_t *pool, double *stressed_ratio);

/* Migrate a specific worker thread to a new CPU set ID.
 * thread_index: 0-based index of the worker to migrate.
 * new_cpuset_id: the CPU set ID to assign. */
int threadpool_migrate_thread(threadpool_t *pool, int thread_index, uint32_t new_cpuset_id);

/* Get stress level for a specific worker based on rolling RT% window.
 * CRITICAL: >100% for 1 chunk (immediate probe)
 * WARN:     >90% for 3+ consecutive (start probe)
 * HEALTHY:  default (no action needed) */
worker_stress_level_t threadpool_get_worker_stress(threadpool_t *pool, int worker_index);

/* Get average RT% for a worker over the rolling window.
 * Returns 0.0 if no data. */
double threadpool_get_worker_avg_rt(threadpool_t *pool, int worker_index);

/* Get the CPU set ID currently assigned to a worker.
 * Returns 0 if not available. */
uint32_t threadpool_get_worker_cpuset(threadpool_t *pool, int worker_index);

/* Measure the actual CPU consumption fraction of a worker thread since the
 * last call. Uses GetThreadTimes to read per-thread kernel+user time and
 * divides by the wall-clock interval since the previous sample. Returns a
 * fraction in [0..1] where 1.0 means the thread was on-CPU 100% of the
 * window. The first call after pool creation returns 0.55 (a neutral
 * default — calibrated for stereo at moderate DSD rates) because there
 * is no baseline yet to diff against.
 *
 * Window length is determined by how often the function is called: every
 * call updates the baseline. Intended to be called from migration_tick
 * with its natural ~5 batch cooldown rhythm (~1 second window).
 *
 * Used by the migration logic to distinguish between "the core is busy
 * because of MY workers" and "the core is busy because of external
 * contention". Without this, the migration logic credits 0.55 per worker
 * which is too low for heavy multichannel DSD256+ workloads, causing
 * the workers to perpetually flee their own legitimate self-load. */
double threadpool_get_worker_self_load(threadpool_t *pool, int worker_index);

#endif /* THREADPOOL_H */
