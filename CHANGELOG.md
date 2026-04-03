# Changelog

All notable changes to foo_dsd_trellis will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.0.4]: https://github.com/mann1x/foo_dsd_trellis/releases/tag/v1.0.4
[1.0.3]: https://github.com/mann1x/foo_dsd_trellis/releases/tag/v1.0.3
[1.0.2]: https://github.com/mann1x/foo_dsd_trellis/releases/tag/v1.0.2
[1.0.1]: https://github.com/mann1x/foo_dsd_trellis/releases/tag/v1.0.1
[1.0.0]: https://github.com/mann1x/foo_dsd_trellis/releases/tag/v1.0.0
