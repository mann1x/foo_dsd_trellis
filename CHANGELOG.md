# Changelog

All notable changes to foo_dsd_trellis will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-04-07

### Added

- **Convolution filter (room correction)** — full GPU pipeline running at the playback DSD rate. Per-channel WAV impulse responses, up to 6 channels (5.1 surround). cuFFT for forward/inverse transforms, custom CUDA kernels for the partition multiply-accumulate, CUDA Graph capture for the per-batch chunk loop. Real-time at every DSD rate from DSD64 to DSD512 on an RTX 5080.
- **Per-channel dedicated CUDA streams**: each conv channel creates and owns its own `CUstream` so up to 7.1 channels can overlap on the GPU without sharing.
- **Per-rate / per-SDM-mode / per-channel-count budget caps**: the IR tap cap auto-scales by playback rate, by SDM mode (Trellis is heavier than PreCorr), and inversely by channel count (5.1 caps are 2/5 of stereo). Calibrated empirically via sweeps.
- **Smart IR pre-truncation**: very long input IRs (e.g. 32K-tap @ 44.1k room IRs) are pre-truncated centered on the impulse peak before resample, bounding resample memory regardless of input size. Mathematically equivalent to post-truncation but doesn't allocate huge intermediate buffers.
- **Latency compensation**: IR group delay (impulse peak position) is reported through fb2k's DSP latency API so video players A/V-sync compensate automatically. For a 2M-tap linear-phase IR at DSD512 the reported latency is 44 ms.
- **Min-phase IR converter** (cepstral Hilbert transform): optional checkbox in conv settings. Converts the loaded IR to its min-phase equivalent before partitioning. Same magnitude response, near-zero group delay (latency drops from L/2 to ~0 ms). The phase distortion introduced is mostly inaudible at audio frequencies for typical room correction filters.
- **Custom IR cap override** (expert mode): "Cap (taps, 0=auto)" edit field in the conv settings. Setting this non-zero replaces the auto-calibrated cap entirely — lets you push above the safe value at your own risk.
- **Tail energy logging**: when budget truncation discards part of the IR, the log now reports head/tail energy in dB relative to total. E.g. "discarded head=-72 dB tail=-58 dB rel total" tells you the truncation cost ~58 dB worth of late reverb (inaudible).
- **Worker batch phase timing** + per-batch summary log line for diagnosing time-budget bottlenecks (unpack / FIR / SDM / pack / conv breakdown per chunk).

### Changed

- **Worker self-load measurement via `GetThreadTimes`**: the worker thread migration logic was estimating self-load as a hardcoded `our_count * 0.55` per worker, calibrated for stereo at moderate DSD rates. On heavy multichannel workloads (5ch DSD256+Trellis+conv) each worker is much busier than that, and the algorithm was treating its own legitimate load as external contention — causing a "self-flee" loop with ~6 migrations/minute. Now uses real `GetThreadTimes`-based measurement plus stricter trigger criteria (core load ≥ 90% AND external ≥ 40%) and longer cooldown (5 → 25 batches).
- **CUDA Graph cache 4 → 64 slots**: the conv graph cache key is `(batch_count, init_fdl_pos)` and `init_fdl_pos` cycles through 0..np-1 each batch. With np=7 (DSD256 5ch) the 4-slot cache thrashed: ~6 captures/sec, conv finalize variable 20-69 ms with periodic spikes. With 64 slots the cache fully covers all current rate × cap combinations, captures only happen at startup, and finalize is stable at 20-32 ms.
- **Convolution settings dialog**: aligned all rows (FIR Gain, ML Filter, GPU Compute, PCM, Convolution) so labels, checkboxes, edits, and combos share a common visual baseline. Cap label width fixed (was truncating "0=auto):").
- **README**: added a comprehensive Convolution section covering CPU/GPU implementation, time budget per rate, REW workflow, tips, common pitfalls, and limitations.

### Fixed

- **DSD512 + Trellis + conv silent crash**: `cached_seg_bufs` was sized for the *average* segment size, but reposition can shift segments by ±200K samples. The undersized buffer caused heap corruption — silent without conv (different heap layout), access violation with conv. Now sized for the largest *actual* segment size after reposition.
- **DSD512 + Trellis startup struggle**: `gpu_create()` ran synchronously inside the first `plugin_process` call, costing ~456 ms — first batch was 290% over budget. GPU pre-init now runs in `worker_main` *before* the worker signals ready, hiding the cost in the parent's worker-start wait window.
- **Conv overhead bottleneck**: 7-stage optimization arc dropped DSD512+Trellis steady proc from 210 ms (108% over budget, ring drained → crash) to 134 ms (67% of budget):
  - Single-IR-partition mode for IRs ≤ 64K (P bumped to 65536) → 4× fewer batches, 75% fewer driver calls
  - DtoH consolidation: one big copy per batch instead of one per chunk
  - `fdl_pos` computed in-kernel: removed indices array upload
  - CUDA Graph capture: entire upload+chunk-loop+download collapsed into a single replayable launch
- **Debug build broken since v1.0.5**: `dsp_fb2k.cpp:976` used `strncpy` which trips warning-as-error in Debug. Replaced with `strncpy_s` using `_TRUNCATE`. All 4 configs (x64/Win32 × Release/Debug) now build clean.

### Configuration

- **Config version 18 → 20**: added `conv_enabled`, `conv_gpu`, `conv_budget`, `conv_paths[6]` (v18), `conv_max_taps_override` (v19), `conv_min_phase` (v20). Forward-compatible deserialization preserves earlier versions.

## [1.0.5] - 2026-04-05

### Fixed
- **DSD rate conversion noise**: Boxcar DSD-Wide pre-smooth (8 taps) before FIR upsample eliminates Gibbs overshoot (±2.24 peaks → ±0.7). Enables full 0 dB gain for PreCorr — no more noise on DSD64→DSD256 or DSD128→DSD256 upsample.
- **DSD rate conversion stutter**: All rate conversion paths now use nc=2 candidates (sweep-optimized NTF/limiter/depth per path). Matches same-rate throughput for parallel DAS processing.
- **PreCorr rate conversion corruption**: PreCorr rate conversion uses sequential engine_process_block path. The parallel FIR+PreCorr split was corrupting the output.
- **Worker priming delay on downsample**: Priming threshold now based on actual output batch size, not input rate. DSD512→DSD64 primes in ~0.8s instead of ~6.5s.
- **Playback time desync**: Latency reporting uses output rate for output ring (was using input rate — 8x underreport for downsample paths).
- **Worker always-on**: All DSD processing uses out-of-process worker (removed 100ms chunk threshold). Bypasses Process Lasso CPU affinity restrictions for both small and large chunk installs.
- **Threadpool semaphore starvation**: Shared workers respond to both wake_event and work_sem. Reserved workers that consume work_sem re-release immediately. Fixes Phase 2 parallel SDM deadlock.
- **GPU PreCorr crash**: Skip GPU PreCorr path when `s->gpu` is NULL (GPU disabled).
- **Worker ring buffer sizing**: Rings sized for 4 seconds of audio (was 8× 200ms batch). Prevents ring-full blocking on rate conversion.
- **Worker ring drain**: Drain all available complete frames instead of demanding exact match. Prevents DSD512 silence from 128-frame warmup mismatch.

### Changed
- **Boxcar taps for rate conversion**: 8 taps (was same as same-rate 32-64). Fewer taps preserve HF content for FIR, improving DSD128→DSD256 SINAD from 66 to 105 dB (+39 dB).
- **Per-path nc=2 configs**: All rate conversion paths swept and optimized for nc=2 candidates with boxcar pre-smooth. Quality maintained while halving Trellis CPU cost.
- **Targeted test suites**: Added `--suite samerate`, `upsample`, `downsample`, `dsdpcm`, `rate48` for faster iteration on specific path families.

## [1.0.4] - 2026-04-03

### Fixed
- **DirectML ML filter now works**: ORT uses the extended DML API (`OrtSessionOptionsAppendExecutionProviderEx_DML`) with an explicitly created D3D12 device, bypassing ORT's internal DXGI adapter enumeration that failed with `DXGI_ERROR_NOT_FOUND` on some systems.
- **CUDA ML filter now works**: Dual ORT runtime support — `onnxruntime_cuda.dll` (CUDA EP, uses system CUDA toolkit) and `onnxruntime.dll` (DirectML EP). Auto mode tries CUDA first, falls back to DirectML, then CPU.
- **ML filter in worker process**: `onnxruntime.dll` and ML model path resolution now falls back to the exe directory when `foo_dsd_trellis.dll` isn't loaded (out-of-process worker).
- **Dialog shows actual ML EP**: Config dialog now shows "Ready (CUDA)" or "Ready (DirectML)" instead of generic "Ready (GPU)" for Auto mode.

### Changed
- **Split release packages**: Base package includes DirectML ML support (~48 MB). Separate CUDA package for NVIDIA users with system CUDA toolkit (~976 MB additional).
- **Updated ONNX Runtime**: v1.24.4 (DirectML + CUDA builds).
- **Updated DirectML**: v1.15.4 redistributable bundled (was missing, caused EP init failure).
- **cuDNN 9**: Required for CUDA ML EP, loaded from system or component folder.

## [1.0.3] - 2026-04-03

### Fixed
- **PCM rate conversion crash**: Heap buffer overflow on multi-stage FIR downsample (e.g., PCM 352.8k→44.1k). The FIR chain's ping-pong intermediate buffer exceeded the output allocation for ratios > 2x.
- **PCM rate conversion crackling**: FIR chains and polyphase resamplers were created/destroyed per chunk, losing filter delay line state at every chunk boundary. Now persistent across chunks.
- **PCM→DSD cross-family routing**: Cross-family paths (e.g., PCM 96k→DSD256/44) failed because engine was initialized with the pre-resample rate, giving a non-power-of-2 FIR ratio. Now sets engine fs_in to the post-resample rate.
- **PCM→DSD cross-family crackling**: Same stateless resampler issue as PCM→PCM, now persistent.
- **PCM output through SDM**: `fir_only` condition only covered DSD→PCM, not PCM→PCM. SDM initialization failed silently for PCM output rates. Now all PCM output paths are FIR-only.
- **PCM rate conversion pop at start**: 128-sample linear fade-in on first chunk suppresses FIR delay line startup transient.
- **Trellis SDM -6 dB gain loss**: Hardcoded `× 0.5` in `sdm_process_block` was never compensated. Removed the internal scaling; gain now matches PreCorr on all paths.
- **Consistent gain across all processing paths**: Same-rate, rate conversion, PCM — all produce matched output volume. Soft-clip at 0.95 (tanh) prevents Trellis overload on rate conversion. DSD64→DSD128 forced to PreCorr (Trellis overloads at any gain).

### Changed
- **Worker reservation re-enabled**: `threadpool_set_reserved` moved from `plugin_init_engine` (caused deadlock) to after first successful chunk. Reserved workers now exclusively process pinned SDM segments.
- **Out-of-process worker**: Separate `foo_dsd_trellis_worker.exe` process with own CPU affinity for async DSD processing via shared memory IPC. Bypasses Process Lasso affinity restrictions.

## [1.0.2] - 2026-03-31

### Fixed
- **Smooth volume control**: Gain ramping across batch boundaries eliminates clicks/pops on volume changes. Linear interpolation from previous to current gain applied per-sample in all processing paths (same-rate boxcar/FIR lowpass, rate-conversion FP64/FP32).
- **Async batch size tuned to 200ms**: 100ms batches caused garbled audio on volume changes due to insufficient processing margin. 200ms provides 60ms slack while keeping volume response acceptable.

## [1.0.1] - 2026-03-31

### Fixed
- **DSD256+ stutter on small-chunk ASIO outputs**: Some foobar2000 installs deliver 13.3ms chunks instead of 1-second chunks. The parallel SDM processing blocked `on_chunk()` for ~350ms, starving the ASIO output. Now uses a dedicated processing thread with lock-free SPSC ring buffers — `on_chunk()` returns in <1ms, batches process asynchronously on a background thread with ~500ms latency.

### Added
- **Lock-free SPSC ring buffer** (`include/ringbuf.h`): PortAudio-style byte ring buffer with monotonic atomic counters, power-of-two capacity, no locks in hot path.
- **Async batch processor**: Dedicated thread with THREAD_PRIORITY_HIGHEST processes ~500ms batches while the audio thread handles I/O. Double-buffered: processing fills output ring while `on_chunk()` drains it.
- **Pre-SDM ML pre-emphasis for all DSD rates**: Previously DSD512-only, now enabled at all rates. Training data and model already supported all rates.
- **Advanced Preferences**: HTTP REST API enable/disable and port configuration under Tools → DSD Trellis (default: disabled).
- **MIT License**

### Changed
- **HTTP REST API**: No longer starts automatically. Must be explicitly enabled in Advanced Preferences.
- **DSD512 Auto mode**: Par4 (was Par8) — faster due to less L3 cache contention (0.52x vs 0.60x RT).
- **DSD256 Auto mode**: Par2 with quality-search DAS (0 >20x spikes measured).
- **Engine code cleanup**: Pre-emphasis logic deduplicated into `engine_apply_preemph()` helper.

## [1.0.0] - 2026-03-30

Initial public release.

### SDM Engine
- **Trellis SDM**: Viterbi look-ahead sigma-delta modulator with configurable depth (4-32), candidates (2-8), and latency (16-128)
- **PreCorr SDM**: Greedy + prediction correction mode, near-zero CPU (~0.01x RT)
- **Rate-specific NTF optimization**: CLANS-6 for DSD64-256, SDM-6 for DSD512, with per-rate depth/latency tuning
- Optimal SINAD: DSD64=110.7 dB, DSD128=121.5 dB, DSD256=128.9 dB, DSD512=140.5 dB

### Parallel SDM with Density-Aligned Stitching (DAS) v2
- **Quality-search DAS**: Multi-candidate stitch point selection using FIR-decode pop detection (255-tap Kaiser at 20kHz)
- **Rate-adaptive dispatch**: Sequential for DSD128 (artifact-free), Par2 for DSD256, Par4 full-parallel for DSD512
- **Boundary repositioning**: RMS-based boundary shift (+-40%) to avoid quiet passages
- **Full-parallel mode** (DSD512+): All segments run simultaneously, no sequential chain, no estimation phase — 0.52x RT on Ryzen 9 5950X
- Production quality at DSD256+ (zero >20x derivative spikes in 20-capture analysis)

### FIR Rate Conversion
- Intel IPP polyphase FIR with automatic SSE2/AVX2/AVX-512 dispatch
- fp64 FIR chain for maximum precision
- Per-rate FIR precision control (Auto/FP32/FP64)
- Cross-family DSD-to-PCM: 2-stage (FIR decimate + polyphase resample)
- IPP + libsoxr runtime-loaded (soxr preferred for 114 dB VHQ)

### Audio Output
- DSD64-512 output (44.1kHz + 48kHz families)
- Native i24 DoP output via `set_data_fixedpoint_ex`
- DoP marker A/B phase tracking across chunks
- PCM encoding: 16/24/32/float with TPDF/noise-shaped dither
- Anti-pop lead-in: rate-switch trick for clean DSD entry

### Thread Pool
- MMCSS "Pro Audio" scheduling for low-latency processing
- Load-binned core selection with cpuset_select
- Per-worker queues, cache-line aligned (128B)
- Background CPU monitor with rolling RT% window
- LP0 excluded from workers (reserved for OS/fb2k)

### Configuration
- Per-rate SDM/NTF/depth/latency/candidates/FIR precision/parallel settings
- 20 input rates, 25 output rates
- Config version 17 with field-by-field serialization
- Debug log with millisecond precision

### GPU Support (experimental)
- CUDA PTX kernel for trellis SDM (--fmad=false for quality)
- DX11/DX12 FIR lowpass compute shaders
- CUDA fp64 upsample/downsample kernels

### ML Noise Filter (experimental)
- ONNX model runtime-loaded via DirectML
- Causal dilated CNN for DSD quantization noise reduction

[1.0.5]: https://github.com/mann1x/foo_dsd_trellis/releases/tag/v1.0.5
[1.0.4]: https://github.com/mann1x/foo_dsd_trellis/releases/tag/v1.0.4
[1.0.3]: https://github.com/mann1x/foo_dsd_trellis/releases/tag/v1.0.3
[1.0.2]: https://github.com/mann1x/foo_dsd_trellis/releases/tag/v1.0.2
[1.0.1]: https://github.com/mann1x/foo_dsd_trellis/releases/tag/v1.0.1
[1.0.0]: https://github.com/mann1x/foo_dsd_trellis/releases/tag/v1.0.0
