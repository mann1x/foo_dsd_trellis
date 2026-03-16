# foo_dsd_trellis — Task List

## Pending

### 1. Deeper SDM Vectorization (HIGH)
SDM is still the bottleneck (~1050-1300ms for 1s DSD512 audio). Target: under 1.0x RT.
- File: `src/trellis.c`
- **Strategy A**: Batch 4 candidates' `sdm_filter_calc` using `__m256d`
  - d[i] values are independent across candidates (same formula, different state vectors)
  - Interleave 4 candidates' state[0..order-1] into AVX2 lanes
  - Compute 4× filter outputs in parallel, then scatter back
  - Requires SoA (struct-of-arrays) layout for candidate states, or gather/scatter
- **Strategy B**: Reduce `sdm_sort_cands` overhead
  - Currently insertion sort with hash-based path dedup over 2×num_cands items
  - Consider partial sort (only need top num_cands), or radix sort on cost
- **Measure**: Profile to confirm filter_calc vs sort split before optimizing

### 2. Benchmark-Driven Workload Planning (MEDIUM)
- Use `perf_score` from `cpuset_benchmark()` to size segments proportionally
- Faster cores get larger segments; slower cores get smaller ones
- Requires: map segments to specific threads (currently any-thread-dequeues-any)
- Alternative: just determine optimal thread count (don't use slow cores at all)
- Consider: skip cores with perf_score < 0.5× max (e.g., core 0 loaded by OS)

### 3. Reduce FIR Phase Time (LOW)
FIR is ~275ms (19% of total). Currently parallelized via BLOCK_MODE_FIR.
- Check if FIR SIMD (`fir_simd.c`) is being used in the hot path
- Consider: merge FIR+SDM into single per-segment dispatch (avoid FIR→barrier→SDM roundtrip)

## Completed
- [x] AVX2 batched 4-candidate filter calc (order 8) — ~7.5% SDM improvement
- [x] REST API control interface (port 8881, GET/PUT config, GET status)
- [x] CPUSET dynamic refresh verified with CPUDoc (T1 disabled, mask changes logged)
- [x] Config version 4 (adds api_port field)
- [x] Force-inline + unroll trellis filter calc orders 4-8
- [x] AVX2 state copy in sdm_filter_calc2
- [x] MMCSS "Pro Audio" scheduling on worker threads
- [x] Default trellis_cands 16→8
- [x] System CPUSET detection with Allocated/Parked/RealTime flags
- [x] Dynamic CPUSET refresh + thread pool rebuild
- [x] Build version system (git hash, date/time)
- [x] Parallel FIR dispatch (BLOCK_MODE_FIR)
- [x] TLS scratch buffer for SDM warmup
- [x] Cached per-chunk allocations
- [x] Per-phase timing instrumentation
- [x] Suppress [E-core] label on AMD CPUs
- [x] Detailed topology logging with per-core flags

## Performance Reference (DSD64→DSD512, 5950X, 1s chunks, 16 T0 threads)
| cands | SDM ms | FIR ms | Total ms | RT ratio |
|-------|--------|--------|----------|----------|
| 8     | 990    | 275    | 1345     | 1.34x    |
| 6     | 608    | 275    | 964      | 0.96x    |
| 4     | 339    | 275    | 696      | 0.70x    |
- **cands=6 achieves real-time DSD512** (0.96x RT)
- **Target: <1.0x RT** for gapless DSD512 playback
