/*
 * foo_dsd_trellis — CPU topology detection and thread placement
 *
 * Uses GetSystemCpuSetInformation (Win10+) for topology detection
 * and SetThreadSelectedCpuSets for thread placement.
 * Handles >64 logical CPUs via processor groups.
 */

#include "../include/cpuset.h"
#include "../include/trellis.h"
#include "../include/ntf.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ─── Win32 API types for GetSystemCpuSetInformation ─── */

/* SYSTEM_CPU_SET_INFORMATION — defined here because older SDKs lack it */
#pragma pack(push, 1)
typedef struct {
    DWORD Size;
    DWORD Type;      /* 0 = CpuSetInformation */
    /* CpuSet union member: */
    DWORD Id;
    WORD  Group;
    BYTE  LogicalProcessorIndex;
    BYTE  CoreIndex;
    BYTE  LastLevelCacheIndex;
    BYTE  NumaNodeIndex;
    BYTE  EfficiencyClass;
    union {
        BYTE AllFlags;
        struct {
            BYTE Parked               : 1;
            BYTE Allocated            : 1;
            BYTE AllocatedToTarget    : 1;
            BYTE RealTime             : 1;
            BYTE ReservedFlags        : 4;
        };
    };
    union {
        DWORD Reserved;
        BYTE  SchedulingClass;
    };
    DWORD64 AllocationTag;
} CPUSET_INFO;
#pragma pack(pop)

/* Function pointer types for dynamic loading */
typedef BOOL (WINAPI *PFN_GetSystemCpuSetInformation)(
    PVOID Buffer, ULONG BufferLength, PULONG ReturnedLength,
    HANDLE Process, ULONG Flags);

typedef BOOL (WINAPI *PFN_SetThreadSelectedCpuSets)(
    HANDLE Thread, const ULONG *CpuSetIds, ULONG CpuSetIdCount);

typedef BOOL (WINAPI *PFN_GetLogicalProcessorInformationEx)(
    DWORD RelationshipType, PVOID Buffer, PDWORD ReturnedLength);

typedef WORD (WINAPI *PFN_GetActiveProcessorGroupCount)(void);

/* Cached function pointers */
static PFN_GetSystemCpuSetInformation   pfn_GetCpuSetInfo;
static PFN_SetThreadSelectedCpuSets     pfn_SetThreadCpuSets;
static PFN_GetLogicalProcessorInformationEx pfn_GetLogicalProcInfoEx;
static PFN_GetActiveProcessorGroupCount pfn_GetGroupCount;
static bool apis_loaded = false;

static void load_apis(void) {
    if (apis_loaded) return;
    HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
    if (k32) {
        pfn_GetCpuSetInfo = (PFN_GetSystemCpuSetInformation)
            GetProcAddress(k32, "GetSystemCpuSetInformation");
        pfn_SetThreadCpuSets = (PFN_SetThreadSelectedCpuSets)
            GetProcAddress(k32, "SetThreadSelectedCpuSets");
        pfn_GetLogicalProcInfoEx = (PFN_GetLogicalProcessorInformationEx)
            GetProcAddress(k32, "GetLogicalProcessorInformationEx");
        pfn_GetGroupCount = (PFN_GetActiveProcessorGroupCount)
            GetProcAddress(k32, "GetActiveProcessorGroupCount");
    }
    apis_loaded = true;
}

/* ─── SMT detection via GetLogicalProcessorInformationEx ─── */

/* RelationProcessorCore = 0 */
#define MY_RelationProcessorCore 0

static void detect_smt(cpu_topology_t *topo) {
    if (!pfn_GetLogicalProcInfoEx)
        return;

    DWORD buf_len = 0;
    pfn_GetLogicalProcInfoEx(MY_RelationProcessorCore, NULL, &buf_len);
    if (buf_len == 0)
        return;

    BYTE *buf = (BYTE *)malloc(buf_len);
    if (!buf) return;

    if (!pfn_GetLogicalProcInfoEx(MY_RelationProcessorCore, buf, &buf_len)) {
        free(buf);
        return;
    }

    /* Each PROCESSOR_RELATIONSHIP entry describes one physical core.
     * GroupMask[0].Mask has bits set for each logical processor in that core.
     * The bit position within the mask gives the logical index within the group.
     * First set bit = T0, second = T1, etc. */
    BYTE *ptr = buf;
    BYTE *end = buf + buf_len;

    while (ptr < end) {
        /* SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX layout:
         * DWORD Relationship (offset 0)
         * DWORD Size (offset 4)
         * then relationship-specific data */
        DWORD relationship = *(DWORD *)ptr;
        DWORD size = *(DWORD *)(ptr + 4);

        if (relationship == MY_RelationProcessorCore && size >= 32) {
            /* PROCESSOR_RELATIONSHIP layout:
             * BYTE  Flags (offset 8) — 1 = SMT enabled on this core
             * BYTE  EfficiencyClass (offset 9)
             * BYTE  Reserved[20] (offset 10)
             * WORD  GroupCount (offset 30)
             * GROUP_AFFINITY[GroupCount] (offset 32)
             *   GROUP_AFFINITY: KAFFINITY Mask(8), WORD Group(2), WORD Reserved[3](6) = 16 bytes */
            BYTE flags = *(ptr + 8);
            WORD group_count = *(WORD *)(ptr + 30);

            if (flags & 1)
                topo->has_smt = true;

            for (WORD gi = 0; gi < group_count; gi++) {
                BYTE *ga_ptr = ptr + 32 + gi * 16;
                DWORD64 mask = *(DWORD64 *)ga_ptr;
                WORD group = *(WORD *)(ga_ptr + 8);

                int thread_num = 0;
                for (int bit = 0; bit < 64; bit++) {
                    if (mask & ((DWORD64)1 << bit)) {
                        /* Find the matching entry in topology */
                        for (int e = 0; e < topo->count; e++) {
                            if (topo->entries[e].group == group &&
                                topo->entries[e].logical_index == bit) {
                                topo->entries[e].smt_thread = (uint8_t)thread_num;
                                break;
                            }
                        }
                        thread_num++;
                    }
                }
            }
        }

        ptr += size;
    }

    free(buf);
}

/* ─── Cluster assignment (AMD CCD / Intel hybrid) ─── */

static void assign_clusters(cpu_topology_t *topo) {
    if (topo->count == 0) return;

    /* Phase 1: LLC-based clustering (works for AMD CCDs) */
    int cluster = 0;
    uint8_t prev_llc = topo->entries[0].llc_index;
    topo->entries[0].cluster = 0;

    for (int i = 1; i < topo->count; i++) {
        if (topo->entries[i].llc_index != prev_llc) {
            cluster++;
            prev_llc = topo->entries[i].llc_index;
        }
        topo->entries[i].cluster = (uint8_t)cluster;
    }

    int num_clusters = cluster + 1;

    /* Phase 2: Intel hybrid pattern detection.
     * If only 1 LLC cluster but mixed efficiency classes, try 6-LP pattern. */
    if (num_clusters == 1 && topo->count > 6 && topo->is_intel) {
        /* Check for Intel repeating 6-LP pattern */
        bool pattern_a = true;  /* [P,P,E,E,E,E] */
        bool pattern_b = true;  /* [E,E,E,E,P,P] */
        int chunks = topo->count / 6;

        for (int c = 0; c < chunks && (pattern_a || pattern_b); c++) {
            int base = c * 6;
            if (base + 5 >= topo->count) break;

            /* Pattern A: first 2 are P-core (eff=1), next 4 are E-core (eff=0) */
            if (!(topo->entries[base].efficiency_class == 1 &&
                  topo->entries[base + 1].efficiency_class == 1 &&
                  topo->entries[base + 2].efficiency_class == 0 &&
                  topo->entries[base + 3].efficiency_class == 0 &&
                  topo->entries[base + 4].efficiency_class == 0 &&
                  topo->entries[base + 5].efficiency_class == 0))
                pattern_a = false;

            /* Pattern B: first 4 are E-core, next 2 are P-core */
            if (!(topo->entries[base].efficiency_class == 0 &&
                  topo->entries[base + 1].efficiency_class == 0 &&
                  topo->entries[base + 2].efficiency_class == 0 &&
                  topo->entries[base + 3].efficiency_class == 0 &&
                  topo->entries[base + 4].efficiency_class == 1 &&
                  topo->entries[base + 5].efficiency_class == 1))
                pattern_b = false;
        }

        if (pattern_a || pattern_b) {
            /* Assign cluster per 6-LP chunk */
            for (int i = 0; i < topo->count; i++)
                topo->entries[i].cluster = (uint8_t)(i / 6);
            num_clusters = (topo->count + 5) / 6;
        }
    }

    /* Build cluster summaries */
    topo->num_clusters = num_clusters;
    memset(topo->clusters, 0, sizeof(topo->clusters));

    for (int i = 0; i < topo->count; i++) {
        int cl = topo->entries[i].cluster;
        if (cl >= CPUSET_MAX_CLUSTERS) continue;

        if (topo->clusters[cl].entry_count == 0)
            topo->clusters[cl].first_entry = i;
        topo->clusters[cl].entry_count++;

        if (topo->entries[i].efficiency_class > 0)
            topo->clusters[cl].has_pcores = true;
        else
            topo->clusters[cl].has_ecores = true;
    }

    /* Count physical cores per cluster (T0 threads only) */
    for (int i = 0; i < topo->count; i++) {
        if (topo->entries[i].smt_thread == 0) {
            int cl = topo->entries[i].cluster;
            if (cl < CPUSET_MAX_CLUSTERS)
                topo->clusters[cl].physical_cores++;
        }
    }
}

/* ─── CPU vendor detection ─── */

static void detect_vendor(cpu_topology_t *topo) {
    int regs[4] = {0};
    __cpuid(regs, 0);

    char vendor[13];
    *(int *)(vendor + 0) = regs[1]; /* EBX */
    *(int *)(vendor + 4) = regs[3]; /* EDX */
    *(int *)(vendor + 8) = regs[2]; /* ECX */
    vendor[12] = '\0';

    topo->is_amd = (strcmp(vendor, "AuthenticAMD") == 0);
    topo->is_intel = (strcmp(vendor, "GenuineIntel") == 0);
}

/* ─── Public API ─── */

int cpuset_detect(cpu_topology_t *topo) {
    memset(topo, 0, sizeof(*topo));
    load_apis();

    detect_vendor(topo);

    /* Processor group count */
    topo->num_groups = 1;
    if (pfn_GetGroupCount)
        topo->num_groups = (int)pfn_GetGroupCount();

    if (!pfn_GetCpuSetInfo) {
        /* Fallback: no CpuSet API (pre-Win10). Use GetSystemInfo. */
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        int ncpu = (int)si.dwNumberOfProcessors;
        if (ncpu > CPUSET_MAX_CPUS) ncpu = CPUSET_MAX_CPUS;

        for (int i = 0; i < ncpu; i++) {
            topo->entries[i].id = (uint32_t)(0x100 + i); /* synthetic IDs */
            topo->entries[i].logical_index = (uint8_t)i;
            topo->entries[i].core_index = (uint8_t)(i / 2);
            topo->entries[i].smt_thread = (uint8_t)(i % 2);
            topo->entries[i].enabled = true;
            topo->entries[i].perf_score = 1.0;
        }
        topo->count = ncpu;
        topo->num_physical_cores = ncpu / 2;
        topo->num_clusters = 1;
        topo->clusters[0].entry_count = ncpu;
        topo->clusters[0].physical_cores = ncpu / 2;
        topo->initialized = true;
        return 0;
    }

    /* Query GetSystemCpuSetInformation with Process=NULL for system-level view */
    ULONG buf_len = 0;
    pfn_GetCpuSetInfo(NULL, 0, &buf_len, NULL, 0);
    if (buf_len == 0)
        return -1;

    BYTE *buf = (BYTE *)malloc(buf_len);
    if (!buf) return -1;

    if (!pfn_GetCpuSetInfo(buf, buf_len, &buf_len, NULL, 0)) {
        free(buf);
        return -1;
    }

    /* Parse entries */
    BYTE *ptr = buf;
    BYTE *end = buf + buf_len;
    int n = 0;
    uint8_t max_eff = 0;

    while (ptr < end && n < CPUSET_MAX_CPUS) {
        CPUSET_INFO *info = (CPUSET_INFO *)ptr;
        if (info->Size == 0) break;

        cpuset_entry_t *e = &topo->entries[n];
        e->id = info->Id;
        e->group = info->Group;
        e->logical_index = info->LogicalProcessorIndex;
        e->core_index = info->CoreIndex;
        e->llc_index = info->LastLevelCacheIndex;
        e->numa_node = info->NumaNodeIndex;
        e->efficiency_class = info->EfficiencyClass;
        e->scheduling_class = info->SchedulingClass;
        e->parked = info->Parked ? true : false;
        e->allocated = info->Allocated ? true : false;
        e->allocated_to_target = info->AllocatedToTarget ? true : false;
        e->realtime = info->RealTime ? true : false;
        e->raw_flags = info->AllFlags;
        /* A core is enabled if it's not allocated to a specific process.
         * "Allocated" means reserved/removed from general pool
         * (e.g., by CPUDoc's system CPUSET management).
         * Parked cores are still enabled — they're just idle and will
         * unpark when assigned work via SetThreadSelectedCpuSets. */
        e->enabled = !e->allocated;
        e->perf_score = 1.0;

        if (info->EfficiencyClass > max_eff)
            max_eff = info->EfficiencyClass;

        n++;
        ptr += info->Size;
    }

    free(buf);
    topo->count = n;
    topo->is_hybrid = (max_eff > 0);

    /* Detect SMT thread numbers */
    detect_smt(topo);

    /* Count physical cores */
    topo->num_physical_cores = 0;
    for (int i = 0; i < n; i++) {
        if (topo->entries[i].smt_thread == 0)
            topo->num_physical_cores++;
    }

    /* Assign clusters */
    assign_clusters(topo);

    /* Check system affinity mask — which cores are available at system level */
    HANDLE proc = GetCurrentProcess();
    DWORD_PTR proc_mask = 0, sys_mask = 0;
    if (GetProcessAffinityMask(proc, &proc_mask, &sys_mask)) {
        /* Apply system affinity mask: disable cores not in the system mask */
        for (int i = 0; i < n; i++) {
            uint8_t lp = topo->entries[i].logical_index;
            if (lp < sizeof(DWORD_PTR) * 8) {
                if (!(sys_mask & ((DWORD_PTR)1 << lp)))
                    topo->entries[i].enabled = false;
            }
        }
        /* Also apply process affinity mask if it's a subset of system mask */
        if (proc_mask != sys_mask) {
            for (int i = 0; i < n; i++) {
                uint8_t lp = topo->entries[i].logical_index;
                if (lp < sizeof(DWORD_PTR) * 8) {
                    if (!(proc_mask & ((DWORD_PTR)1 << lp)))
                        topo->entries[i].enabled = false;
                }
            }
        }
    }

    /* Also check process-level CPU set restriction (Windows 10+) */
    typedef BOOL (WINAPI *PFN_GetProcessDefaultCpuSets)(
        HANDLE Process, PULONG CpuSetIds, ULONG CpuSetIdCount, PULONG RequiredIdCount);
    PFN_GetProcessDefaultCpuSets pfn_getProcCpuSets =
        (PFN_GetProcessDefaultCpuSets)GetProcAddress(
            GetModuleHandleW(L"kernel32.dll"), "GetProcessDefaultCpuSets");

    if (pfn_getProcCpuSets) {
        ULONG req = 0;
        pfn_getProcCpuSets(proc, NULL, 0, &req);
        if (req > 0 && req <= CPUSET_MAX_CPUS) {
            ULONG *ids = (ULONG *)malloc(req * sizeof(ULONG));
            if (ids) {
                ULONG got = 0;
                if (pfn_getProcCpuSets(proc, ids, req, &got)) {
                    for (int i = 0; i < n; i++)
                        topo->entries[i].enabled = false;
                    for (ULONG j = 0; j < got; j++) {
                        for (int i = 0; i < n; i++) {
                            if (topo->entries[i].id == ids[j])
                                topo->entries[i].enabled = true;
                        }
                    }
                }
                free(ids);
            }
        }
    }

    /* Compute initial system CPUSET bitmask */
    topo->last_cpuset_mask = 0;
    for (int i = 0; i < n; i++) {
        if (topo->entries[i].enabled) {
            uint8_t lp = topo->entries[i].logical_index;
            if (lp < 64)
                topo->last_cpuset_mask |= ((uint64_t)1 << lp);
        }
    }

    topo->initialized = true;
    return 0;
}

/* ─── System CPUSET refresh ─── */

/* Cached buffer for cpuset_refresh to avoid per-chunk malloc/free */
static BYTE  *g_refresh_buf = NULL;
static ULONG  g_refresh_buf_sz = 0;

uint64_t cpuset_refresh(cpu_topology_t *topo, bool *changed) {
    if (changed) *changed = false;
    if (!topo->initialized || !pfn_GetCpuSetInfo)
        return topo->last_cpuset_mask;

    /* Re-query system CPU set info (Process=NULL for system-level view) */
    ULONG buf_len = 0;
    pfn_GetCpuSetInfo(NULL, 0, &buf_len, NULL, 0);
    if (buf_len == 0)
        return topo->last_cpuset_mask;

    /* Use cached buffer to avoid malloc/free on every chunk */
    if (buf_len > g_refresh_buf_sz) {
        free(g_refresh_buf);
        g_refresh_buf = (BYTE *)malloc(buf_len);
        g_refresh_buf_sz = g_refresh_buf ? buf_len : 0;
    }
    BYTE *buf = g_refresh_buf;
    if (!buf)
        return topo->last_cpuset_mask;

    if (!pfn_GetCpuSetInfo(buf, buf_len, &buf_len, NULL, 0)) {
        return topo->last_cpuset_mask;
    }

    /* Parse and update flags */
    BYTE *ptr = buf;
    BYTE *end = buf + buf_len;
    int idx = 0;
    uint64_t new_mask = 0;

    while (ptr < end && idx < topo->count) {
        CPUSET_INFO *info = (CPUSET_INFO *)ptr;
        if (info->Size == 0) break;

        /* Match by ID to handle potential reordering */
        for (int i = 0; i < topo->count; i++) {
            if (topo->entries[i].id == info->Id) {
                topo->entries[i].parked = info->Parked ? true : false;
                topo->entries[i].allocated = info->Allocated ? true : false;
                topo->entries[i].realtime = info->RealTime ? true : false;
                topo->entries[i].raw_flags = info->AllFlags;
                topo->entries[i].enabled = !info->Allocated;

                if (topo->entries[i].enabled) {
                    uint8_t lp = topo->entries[i].logical_index;
                    if (lp < 64)
                        new_mask |= ((uint64_t)1 << lp);
                }
                break;
            }
        }

        idx++;
        ptr += info->Size;
    }

    free(buf);

    if (new_mask != topo->last_cpuset_mask) {
        topo->last_cpuset_mask = new_mask;
        if (changed) *changed = true;
    }

    return new_mask;
}

/* ─── Micro-benchmark ─── */

/* Short SDM computation to estimate per-core performance */
static double benchmark_one_core(void) {
    /* Run a small SDM block and measure time */
    const ntf_filter_t *filter = ntf_auto_select(DSD_RATE_64);
    if (!filter) return 1.0;

    sdm_context_t ctx;
    if (sdm_context_init(&ctx, filter, 8, 8, 64) != 0)
        return 1.0;

    /* Generate 2048 test samples (alternating ±0.5) */
    float in_buf[2048];
    float out_buf[2048];
    for (int i = 0; i < 2048; i++)
        in_buf[i] = (i & 1) ? 0.5f : -0.5f;

    LARGE_INTEGER t0, t1, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    sdm_process_block(&ctx, in_buf, out_buf, 2048);

    QueryPerformanceCounter(&t1);
    sdm_context_free(&ctx);

    double elapsed = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
    if (elapsed <= 0.0) elapsed = 1e-9;

    return 2048.0 / elapsed;  /* samples/sec — higher is faster */
}

typedef struct {
    uint32_t cpuset_id;
    double   result;
} bench_work_t;

static DWORD WINAPI bench_thread(LPVOID param) {
    bench_work_t *work = (bench_work_t *)param;

    /* Pin to this CPU set */
    if (pfn_SetThreadCpuSets) {
        ULONG id = work->cpuset_id;
        pfn_SetThreadCpuSets(GetCurrentThread(), &id, 1);
    }

    /* Warm up */
    benchmark_one_core();

    /* Measure */
    work->result = benchmark_one_core();

    return 0;
}

void cpuset_benchmark(cpu_topology_t *topo) {
    if (!topo->initialized || topo->count == 0)
        return;

    /* Benchmark each enabled T0 core (skip T1 — same physical core) */
    bench_work_t *works = (bench_work_t *)calloc(
        (size_t)topo->count, sizeof(bench_work_t));
    if (!works) return;

    int bench_count = 0;
    for (int i = 0; i < topo->count; i++) {
        if (!topo->entries[i].enabled)
            continue;
        if (topo->entries[i].smt_thread != 0)
            continue;  /* Only benchmark T0 */

        works[bench_count].cpuset_id = topo->entries[i].id;
        bench_count++;
    }

    /* Run benchmarks sequentially to avoid interference */
    for (int i = 0; i < bench_count; i++) {
        HANDLE h = CreateThread(NULL, 0, bench_thread, &works[i], 0, NULL);
        if (h) {
            WaitForSingleObject(h, 5000);  /* 5s timeout */
            CloseHandle(h);
        }
    }

    /* Find max score for normalization */
    double max_score = 0.0;
    for (int i = 0; i < bench_count; i++) {
        if (works[i].result > max_score)
            max_score = works[i].result;
    }
    if (max_score <= 0.0) max_score = 1.0;

    /* Assign normalized scores. T1 threads get same score as their T0. */
    int bi = 0;
    for (int i = 0; i < topo->count; i++) {
        if (!topo->entries[i].enabled) {
            topo->entries[i].perf_score = 0.0;
            continue;
        }
        if (topo->entries[i].smt_thread == 0) {
            if (bi < bench_count) {
                topo->entries[i].perf_score = works[bi].result / max_score;
                bi++;
            }
        }
    }

    /* Copy T0 scores to T1 siblings */
    for (int i = 0; i < topo->count; i++) {
        if (topo->entries[i].smt_thread > 0) {
            /* Find T0 sibling (same core_index and group) */
            for (int j = 0; j < topo->count; j++) {
                if (topo->entries[j].smt_thread == 0 &&
                    topo->entries[j].core_index == topo->entries[i].core_index &&
                    topo->entries[j].group == topo->entries[i].group) {
                    topo->entries[i].perf_score = topo->entries[j].perf_score;
                    break;
                }
            }
        }
    }

    free(works);
}

/* ─── Thread selection ─── */

/* Comparison function for sorting entries by preference */
typedef struct {
    int     entry_index;
    int     priority;     /* Lower = preferred */
    double  perf;         /* Higher = preferred */
} select_candidate_t;

static int cmp_candidate(const void *a, const void *b) {
    const select_candidate_t *ca = (const select_candidate_t *)a;
    const select_candidate_t *cb = (const select_candidate_t *)b;
    if (ca->priority != cb->priority)
        return ca->priority - cb->priority;
    /* Higher perf score first */
    if (ca->perf > cb->perf) return -1;
    if (ca->perf < cb->perf) return 1;
    return 0;
}

int cpuset_select(const cpu_topology_t *topo,
                  smt_mode_t smt_mode,
                  ccd_mode_t ccd_mode,
                  ecore_mode_t ecore_mode,
                  int max_threads,
                  uint32_t *selected_ids,
                  int max_ids) {
    if (!topo->initialized || topo->count == 0)
        return 0;

    select_candidate_t cands[CPUSET_MAX_CPUS];
    int ncands = 0;

    for (int i = 0; i < topo->count; i++) {
        const cpuset_entry_t *e = &topo->entries[i];
        if (!e->enabled)
            continue;
        /* Don't exclude parked cores — SetThreadSelectedCpuSets will
         * cause the OS to unpark them when needed. Excluding parked
         * cores means we'd only use the busy/hot cores, which is the
         * opposite of what we want. */

        /* SMT filtering */
        if (smt_mode == SMT_T0_ONLY && e->smt_thread != 0)
            continue;

        /* E-core filtering (Intel hybrid) */
        if (topo->is_hybrid) {
            if (ecore_mode == ECORE_EXCLUDE && e->efficiency_class == 0)
                continue;
            if (ecore_mode == ECORE_ONLY && e->efficiency_class > 0)
                continue;
        }

        /* Compute priority: lower = more preferred.
         *
         * Strategy: avoid cores that Windows uses heavily (high
         * scheduling_class). Prefer parked/idle cores that will be
         * unparked on demand — they're less likely to compete with
         * the OS and other apps for the same cores.
         *
         * Priority bands:
         *   0-99:    T0 threads, sorted by scheduling_class (low = idle)
         *   100-199: T0 threads on parked cores (OS will unpark)
         *   1000+:   T1 (SMT sibling) threads — last resort
         *
         * Within each band, lower scheduling_class = preferred.
         * Ties broken by perf_score (higher = better). */
        int prio = 0;

        /* Don't penalize parked cores — the Parked flag is transient
         * (Windows parks idle cores and unparks on demand). At query
         * time most cores appear parked even when they'll be active
         * under load. Treat all enabled cores equally. */

        /* Avoid only the busiest cores (sched_class >= 13).
         * Mid-range cores (sched 8-12) are fine — they have some OS
         * activity but not enough to impact our workload.
         * This keeps workers on a mix of CCDs instead of exiling
         * everything to the cold CCD. */
        if (e->scheduling_class >= 13)
            prio += 50;  /* push LP0/LP2/LP10 to end */

        /* SMT: T0 preferred over T1 */
        if (e->smt_thread > 0)
            prio += 1000;

        /* CCD/cluster preference — minimal penalty.
         * Idle cores on any cluster beat loaded cores on preferred cluster. */
        if (ccd_mode == CCD_AUTO) {
            prio += e->cluster * 1;
        }
        /* CCD_ALL: all clusters equally preferred (no cluster penalty) */

        /* E-core: P-cores preferred in AUTO mode */
        if (topo->is_hybrid && ecore_mode == ECORE_AUTO) {
            if (e->efficiency_class == 0)
                prio += 500;  /* E-cores last */
        }

        cands[ncands].entry_index = i;
        cands[ncands].priority = prio;
        cands[ncands].perf = e->perf_score;
        ncands++;
    }

    /* Sort by priority then by performance */
    qsort(cands, (size_t)ncands, sizeof(select_candidate_t), cmp_candidate);

    /* Determine how many to select.
     * max_threads=0 (auto): cap to physical core count.
     * T1 (SMT) threads share physical cores and cause contention. */
    int select_count = ncands;
    if (max_threads > 0) {
        if (max_threads < select_count)
            select_count = max_threads;
    } else {
        int phys = topo->num_physical_cores > 0 ? topo->num_physical_cores : ncands;
        if (phys < select_count)
            select_count = phys;
    }
    if (select_count > max_ids)
        select_count = max_ids;

    for (int i = 0; i < select_count; i++)
        selected_ids[i] = topo->entries[cands[i].entry_index].id;

    return select_count;
}

/* ─── Thread pinning ─── */

int cpuset_pin_thread(HANDLE thread, const uint32_t *ids, int count) {
    if (!pfn_SetThreadCpuSets || count <= 0)
        return -1;

    ULONG *uids = (ULONG *)ids;
    if (pfn_SetThreadCpuSets(thread, uids, (ULONG)count))
        return 0;
    return -1;
}

/* ─── Summary ─── */

void cpuset_summary(const cpu_topology_t *topo, char *buf, size_t buf_size) {
    if (!topo->initialized) {
        snprintf(buf, buf_size, "CPU topology not detected");
        return;
    }

    const char *vendor = topo->is_amd ? "AMD" : (topo->is_intel ? "Intel" : "Unknown");

    int enabled = 0;
    for (int i = 0; i < topo->count; i++)
        if (topo->entries[i].enabled) enabled++;

    int pos = snprintf(buf, buf_size,
        "%s: %d logical (%d enabled), %d physical, %d cluster%s",
        vendor, topo->count, enabled,
        topo->num_physical_cores,
        topo->num_clusters, topo->num_clusters > 1 ? "s" : "");

    if (topo->has_smt && pos < (int)buf_size)
        pos += snprintf(buf + pos, buf_size - (size_t)pos, ", SMT");

    if (topo->is_hybrid && pos < (int)buf_size)
        pos += snprintf(buf + pos, buf_size - (size_t)pos, ", hybrid P+E");

    if (topo->num_groups > 1 && pos < (int)buf_size)
        pos += snprintf(buf + pos, buf_size - (size_t)pos, ", %d groups", topo->num_groups);

    /* Cluster details */
    for (int c = 0; c < topo->num_clusters && c < CPUSET_MAX_CLUSTERS; c++) {
        if (pos >= (int)buf_size - 1) break;
        pos += snprintf(buf + pos, buf_size - (size_t)pos,
            "\n  cluster %d: %d cores, %d threads",
            c, topo->clusters[c].physical_cores,
            topo->clusters[c].entry_count);
    }
}

/* ─── Detailed logging ─── */

void cpuset_log_detail(const cpu_topology_t *topo, cpuset_log_fn log_fn, void *ctx) {
    if (!topo->initialized || !log_fn) return;

    char line[256];

    /* Overall summary */
    const char *vendor = topo->is_amd ? "AMD" : (topo->is_intel ? "Intel" : "Unknown");
    int enabled = 0;
    for (int i = 0; i < topo->count; i++)
        if (topo->entries[i].enabled) enabled++;

    snprintf(line, sizeof(line),
        "vendor=%s logical=%d enabled=%d physical=%d clusters=%d smt=%s hybrid=%s groups=%d",
        vendor, topo->count, enabled, topo->num_physical_cores,
        topo->num_clusters, topo->has_smt ? "yes" : "no",
        topo->is_hybrid ? "yes" : "no", topo->num_groups);
    log_fn(line, ctx);

    /* Per-cluster summary */
    for (int c = 0; c < topo->num_clusters && c < CPUSET_MAX_CLUSTERS; c++) {
        const cpuset_cluster_t *cl = &topo->clusters[c];

        /* Calculate min/max perf within cluster */
        double min_perf = 999.0, max_perf = 0.0;
        for (int i = 0; i < topo->count; i++) {
            if (topo->entries[i].cluster == c && topo->entries[i].smt_thread == 0 &&
                topo->entries[i].enabled) {
                if (topo->entries[i].perf_score < min_perf)
                    min_perf = topo->entries[i].perf_score;
                if (topo->entries[i].perf_score > max_perf)
                    max_perf = topo->entries[i].perf_score;
            }
        }
        if (min_perf > 900.0) min_perf = 0.0;

        const char *core_tag = "";
        if (topo->is_hybrid) {
            if (cl->has_pcores && cl->has_ecores) core_tag = " [P+E]";
            else if (cl->has_pcores) core_tag = " [P-core]";
            else if (cl->has_ecores) core_tag = " [E-core]";
        }

        snprintf(line, sizeof(line),
            "  cluster %d: %d cores, %d threads, perf=%.2f-%.2f%s",
            c, cl->physical_cores, cl->entry_count,
            min_perf, max_perf, core_tag);
        log_fn(line, ctx);
    }

    /* Per-core detail (all cores including T1 for visibility) */
    for (int i = 0; i < topo->count; i++) {
        const cpuset_entry_t *e = &topo->entries[i];

        /* Build flags string */
        char flags[64] = "";
        int fpos = 0;
        if (e->parked)
            fpos += snprintf(flags + fpos, sizeof(flags) - fpos, " parked");
        if (e->allocated)
            fpos += snprintf(flags + fpos, sizeof(flags) - fpos, " alloc");
        if (e->allocated_to_target)
            fpos += snprintf(flags + fpos, sizeof(flags) - fpos, " target");
        if (e->realtime)
            fpos += snprintf(flags + fpos, sizeof(flags) - fpos, " rt");
        if (!e->enabled)
            fpos += snprintf(flags + fpos, sizeof(flags) - fpos, " DISABLED");

        snprintf(line, sizeof(line),
            "  core %2d/%d: id=%u grp=%d lp=%d llc=%d numa=%d eff=%d sched=%d "
            "flags=0x%02x perf=%.3f%s",
            (int)e->core_index, (int)e->smt_thread, e->id, (int)e->group,
            (int)e->logical_index, (int)e->llc_index,
            (int)e->numa_node, (int)e->efficiency_class,
            (int)e->scheduling_class, (int)e->raw_flags,
            e->perf_score, flags);
        log_fn(line, ctx);
    }
}

void cpuset_free(cpu_topology_t *topo) {
    memset(topo, 0, sizeof(*topo));
}
