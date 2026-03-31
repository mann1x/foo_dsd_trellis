# Changelog

All notable changes to foo_dsd_trellis will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.0.1]: https://github.com/mann1x/foo_dsd_trellis/releases/tag/v1.0.1
[1.0.0]: https://github.com/mann1x/foo_dsd_trellis/releases/tag/v1.0.0
