/*
 * foo_dsd_trellis — CPU topology detection and thread placement
 *
 * Detects:
 *   - AMD CCDs via LastLevelCacheIndex (L3 sharing)
 *   - Intel hybrid P-core/E-core via EfficiencyClass
 *   - SMT T0/T1 via GetLogicalProcessorInformationEx
 *   - Processor groups for >64 logical CPUs
 *   - Per-core performance via micro-benchmark
 *
 * Uses Windows 10+ GetSystemCpuSetInformation / SetThreadSelectedCpuSets
 * for group-aware, >64-core-safe thread placement.
 */

#ifndef CPUSET_H
#define CPUSET_H

#include "dsd_types.h"

/* Maximum supported logical processors */
#define CPUSET_MAX_CPUS     256
#define CPUSET_MAX_CLUSTERS  16

/* SMT thread selection mode */
typedef enum {
    SMT_AUTO    = 0,   /* Prefer T0, use T1 under load */
    SMT_T0_ONLY = 1,   /* Only T0 (one thread per physical core) */
} smt_mode_t;

/* CCD/cluster selection mode */
typedef enum {
    CCD_AUTO = 0,       /* Prefer first CCD/cluster, overflow to next */
    CCD_ALL  = 1,       /* Use all CCDs equally */
} ccd_mode_t;

/* E-core handling (Intel hybrid only, ignored on AMD) */
typedef enum {
    ECORE_AUTO    = 0,  /* Include E-cores when P-cores exhausted */
    ECORE_EXCLUDE = 1,  /* Never use E-cores */
    ECORE_ONLY   = 2,   /* Only use E-cores */
} ecore_mode_t;

/* Per-logical-processor info */
typedef struct {
    uint32_t id;                /* CPU set ID (for SetThreadSelectedCpuSets) */
    uint16_t group;             /* Processor group (for >64 CPUs) */
    uint8_t  logical_index;     /* Logical processor index within group */
    uint8_t  core_index;        /* Physical core index */
    uint8_t  llc_index;         /* Last Level Cache index (CCD on AMD) */
    uint8_t  numa_node;         /* NUMA node */
    uint8_t  efficiency_class;  /* 0=E-core (or AMD), 1=P-core */
    uint8_t  scheduling_class;  /* OS scheduling priority */
    uint8_t  smt_thread;        /* 0=T0, 1=T1, ... */
    uint8_t  cluster;           /* Computed cluster index */
    double   perf_score;        /* Benchmark: relative performance (higher=faster) */
    bool     enabled;           /* Available for use (not parked, in process CPU set) */
    bool     parked;            /* Core parked by OS */
    bool     allocated;         /* Allocated to a specific process */
    bool     allocated_to_target; /* Allocated to this process */
    bool     realtime;          /* Marked as real-time capable */
    uint8_t  raw_flags;         /* Raw AllFlags byte from GetSystemCpuSetInformation */
    double   load;              /* CPU load 0.0-1.0 (updated by cpuset_update_load) */
} cpuset_entry_t;

/* Cluster (CCD/Intel cluster) summary */
typedef struct {
    int  first_entry;           /* Index of first entry in this cluster */
    int  entry_count;           /* Logical processors in this cluster */
    int  physical_cores;        /* Physical cores in this cluster */
    bool has_ecores;            /* Contains E-cores (Intel) */
    bool has_pcores;            /* Contains P-cores (Intel) */
} cpuset_cluster_t;

/* Full CPU topology */
typedef struct {
    cpuset_entry_t    entries[CPUSET_MAX_CPUS];
    int               count;               /* Total logical processors */
    int               num_physical_cores;
    cpuset_cluster_t  clusters[CPUSET_MAX_CLUSTERS];
    int               num_clusters;
    int               num_groups;          /* Processor groups */
    bool              is_hybrid;           /* Intel hybrid (P+E) */
    bool              is_amd;
    bool              is_intel;
    bool              has_smt;             /* SMT/HT active */
    bool              initialized;
    uint64_t          last_cpuset_mask;   /* Last observed system CPUSET bitmask */
} cpu_topology_t;

/* Detect CPU topology. Returns 0 on success. */
int cpuset_detect(cpu_topology_t *topo);

/* Run micro-benchmark on each core to estimate performance.
 * Updates perf_score in each entry. */
void cpuset_benchmark(cpu_topology_t *topo);

/* Update per-core CPU load (0.0-1.0) by querying OS processor times.
 * Call periodically (e.g., every few chunks) — caches internally. */
/* Note: mutates topo->entries[].load — not const despite only updating load. */
void cpuset_update_load(cpu_topology_t *topo);

/* Select which logical processors to use for thread pool workers.
 * Returns number of selected CPUs. selected_ids[] is filled with
 * CPU set IDs suitable for SetThreadSelectedCpuSets.
 * max_threads: 0 = auto (select based on modes). */
int cpuset_select(const cpu_topology_t *topo,
                  smt_mode_t smt_mode,
                  ccd_mode_t ccd_mode,
                  ecore_mode_t ecore_mode,
                  int max_threads,
                  uint32_t *selected_ids,
                  int max_ids);

/* Pin a thread to a specific CPU set ID. Returns 0 on success. */
int cpuset_pin_thread(HANDLE thread, const uint32_t *ids, int count);

/* Free any resources (currently nothing to free, but for future use). */
void cpuset_free(cpu_topology_t *topo);

/* Get a human-readable summary for logging. */
void cpuset_summary(const cpu_topology_t *topo, char *buf, size_t buf_size);

/* Re-query system CPUSET flags and update enabled status.
 * Returns the current system CPUSET bitmask (bit per logical processor).
 * If the bitmask changed since the topology was last queried,
 * *changed is set to true and enabled flags are updated. */
uint64_t cpuset_refresh(cpu_topology_t *topo, bool *changed);

/* Get detailed per-core info for logging. Calls log_fn for each line.
 * log_fn signature: void(*)(const char *line, void *ctx) */
typedef void (*cpuset_log_fn)(const char *line, void *ctx);
void cpuset_log_detail(const cpu_topology_t *topo, cpuset_log_fn log_fn, void *ctx);

#endif /* CPUSET_H */
