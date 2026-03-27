# foo_dsd_trellis

A native foobar2000 DSP plugin that converts PCM and DSD audio to DSD using sigma-delta modulation. Supports two SDM modes: **Trellis** (Viterbi look-ahead, highest quality) and **PreCorr** (greedy + prediction correction, near-zero CPU). FIR rate conversion uses Intel IPP with automatic SSE2/AVX2/AVX-512 dispatch.

## Features

| Feature | Description |
|---------|-------------|
| SDM Modes | PreCorr (default, ~0.01x RT) and Trellis (high quality, configurable depth/candidates) |
| Rate Conversion | DSD64-512 (44.1kHz + 48kHz families) via Intel IPP FIRMR polyphase (63-tap Kaiser half-band) |
| PCM to DSD | Float32 PCM input upsampled via FIR then quantised to 1-bit DSD |
| PCM to PCM | Same-family (FIR chain) and cross-family (polyphase resampler: IPP or libsoxr) |
| PCM Encoding | Output bit depth (16/24/32/float) with TPDF or noise-shaped dither, global + per-rate |
| DSD/48 Rates | Full support for 48kHz-family DSD: DSD64/48 (3.072 MHz) through DSD512/48 (24.576 MHz) |
| High PCM | Input/output up to 1536 kHz (44.1k and 48k families) |
| Per-Rate Config | SDM mode, candidates, depth, NTF, state limiter, ML, PCM bits/dither configurable per input rate |
| FIR Gain | Global FIR gain (Auto = -3 dB / 0.708) for uniform volume across all rate conversion paths |
| Volume Control | DSD-Wide: boxcar smoothing + gain + SDM re-encode (no PCM decimation) |
| Anti-Pop | Three-layer anti-pop: SDM state preservation + DoP 0x69 silence trail + rate-switch DAC mute trick |
| Parallel SDM | Density-Aligned Stitching (DAS) for artifact-free parallel Trellis processing |
| GPU Compute | CUDA/DX12 acceleration for FIR, boxcar, lowpass. GPU SDM (experimental) |
| Mute | Silence pattern substitution (0x69/0x96) |
| DoP Detection | Auto-detect DoP markers (0x05/0xFA) in 24-bit PCM frames |
| Output Modes | DoP (native DSD output) or PCM (for VU meter / non-DSD DACs) |
| Intel IPP | Statically linked, automatic CPU dispatch (SSE2 -> AVX2 -> AVX-512) |
| ONNX ML Filter | Optional causal CNN post-filter for DSD noise reduction (delay-loaded onnxruntime.dll) |
| Property Page | Full configuration dialog with per-rate settings, path info display, dark mode support |
| Config Versioning | Forward-compatible binary preset serialization (v17) with legacy fallback |
| REST API | HTTP control/monitoring API on port 8881 |
| CPU Topology | Dynamic CPUSET core selection with scheduling_class priority, SMT/CCD/E-core awareness |
| TUSBAudio | Runtime XMOS DAC detection and status logging |

## Architecture

Four-layer design with clean separation of concerns:

```
1-bit in -> unpack -> float32 @ Fs_in -> [IPP FIRMR] -> x gain -> SDM (Trellis|PreCorr) -> 1-bit out @ Fs_out
```

| Layer | Purpose | Files |
|-------|---------|-------|
| fb2k Interface | foobar2000 DSP v2 glue, config dialog | `dsp_fb2k.cpp`, `dsp_plugin.c`, `config.c` |
| Format Bridge | DoP/native detection, 1-bit <-> float32 | `dop.c`, `bitpack.c` |
| Processing Engine | IPP FIR rate conversion + gain + parallel SDM | `engine.c`, `fir.c` |
| SDM | Trellis (Viterbi) or PreCorr (greedy + prediction) | `trellis.c`, `precorr.c`, `ntf.c` |
| Polyphase Resampler | Cross-family PCM rate conversion (IPP + libsoxr) | `resample.c` |
| Infrastructure | Thread pool, CPU topology, REST API, ONNX ML, TUSBAudio | `threadpool.c`, `cpuset.c`, `httpapi.c`, `onnx_filter.c`, `tusbaudio.c` |

Thread pool (`threadpool.c`) provides per-channel and per-segment parallelism with MMCSS "Pro Audio" scheduling.

### Parallel SDM with Density-Aligned Stitching (DAS)

For Trellis mode at high DSD rates (DSD512), single-core SDM processing exceeds real-time. The engine splits FIR output into N segments (up to 4 on CPU, 252 on GPU) processed in parallel:

1. **State estimation**: a 16-level (4-bit) greedy SDM pre-pass estimates the NTF integrator state at each segment boundary. Channels run in parallel on the threadpool. Segments 1+ are seeded with these estimated states instead of replicating the persistent state.
2. **Overlap extension**: each non-last segment extends by `overlap = 32 × trellis_lat` into the next segment's territory
3. **DAS density scan**: O(n) sliding window match density (window = 2 × trellis_lat) finds the region of best SDM convergence in the overlap
4. **Hybrid stitch**: density peak + nearest exact bit-match for clean transition

**Performance** (AMD Ryzen 9 9950X, stereo, CUDA GPU for FIR):

| Rate | Segments | RT Ratio | Overlap | Density |
|------|----------|----------|---------|---------|
| DSD512 | 4 | 0.85x | 1024 | 53-77% |
| DSD256 | 2 | 0.52x | 4096 | 43-82% |
| DSD128 | 2 | 0.27x | 4096 | 50-73% |
| DSD64 | 2 | 0.19x | 1024 | 55-88% |

**Validated artifact-free**: DSF A/B comparison proves CPU parallel DAS produces **bit-identical** output to sequential for simple signals, and **perceptually identical** output for real music (48.7% bit mismatch but identical noise characteristics).

**GPU DAS pipeline** (experimental): 3-kernel CUDA pipeline — parallel-segment SBVD + density scan + gather-assemble. 252 segments on RTX 5080 at 0.03x RT for DSD512. GPU SDM kernel quality under investigation.

See `Density-Aligned-Stitching/DAS-Algorithm.md` for the full algorithm description.

### Anti-Pop System

DSD playback over DoP is prone to pops at start and stop because the DAC switches between PCM and DSD modes when the DoP stream starts or stops. The anti-pop system uses three complementary layers:

#### Layer 1: SDM State Preservation (Play Start)

When `engine_channel_reset` is called (stop, seek, track change) with `antipop` enabled, the Trellis SDM internal state (integrator values, candidate paths, history buffers) is **preserved** — only the FIR chain and boxcar filter are reset. This eliminates the DC step that occurs when SDM integrators restart from zero.

- **Stop → Play**: SDM continues from its previous integrator state, producing smooth output from the first sample
- **PreCorr**: Always fully reset (greedy quantizer doesn't benefit from preserved state)
- **sdm_drain skipped**: When antipop is enabled, `on_endofplayback` does not call `sdm_drain` — draining feeds zeros and sets `draining=1`, which would corrupt preserved state on rapid stop→play

#### Layer 2: DoP 0x69 Silence Trail (Stop)

At `on_endofplayback`, inserts 150ms of properly DoP-framed DSD silence before the stream stops:

1. **75ms at current DSD rate**: DSD idle pattern `0x69` (`01101001` — a toggling pattern with zero DC content) packed with DoP markers (0x05/0xFA). The DAC stays in DSD mode while its analog output settles to zero.
2. **75ms at an alternate DSD rate** (DSD64↔DSD128): Forces a sample rate change, triggering the DAC's hardware mute circuit. The DAC mutes its output before the stream actually stops, preventing the PCM mode-revert pop.

**Why 0x69?** The standard DSD silence byte. PCM silence (0x00) represents negative full-scale in DSD — a massive DC transient. The alternating bit pattern 0xAA was also wrong (different DC characteristics). 0x69 is recognized by DAC firmware as proper DSD idle.

#### Layer 3: DoP Silence Lead-In (Play Start)

On the first chunk after stop→play, inserts DoP 0x69 silence before real audio:

- **Cold start** (no prior trailing silence): 35ms at alternate rate + 35ms at target rate — the rate switch triggers the DAC's hardware mute, then the target-rate silence lets the DAC settle at the correct rate before audio begins.
- **After trailing silence** (recent stop→play): 75ms at target rate only — the trail already muted the DAC, so no rate switch is needed. Avoids rapid rate-change conflicts.

#### Design Constraints

- **No `sdm_drain` with antipop**: Draining corrupts preserved SDM state (`draining=1`, zero-fed integrators) — causes crashes on rapid stop→play.
- **No rate switch in trailing silence AND lead-in**: Rapid stop→play can queue 4+ rate changes in the output pipeline, crashing fb2k's ASIO output. The `m_trail_inserted` flag prevents lead-in rate switching when the trail already handled it.
- **Settings apply on OK only**: The property page does not push `on_preset_changed` on every control toggle — fb2k would destroy and recreate the DSP instance mid-playback, causing volume spikes (gain reset to 1.0) and stuttering.

## DSD Rates

### 44.1 kHz Family

| Rate | Sample Rate | Multiplier |
|------|-------------|------------|
| DSD64 | 2,822,400 Hz | 64 × 44.1 kHz |
| DSD128 | 5,644,800 Hz | 128 × 44.1 kHz |
| DSD256 | 11,289,600 Hz | 256 × 44.1 kHz |
| DSD512 | 22,579,200 Hz | 512 × 44.1 kHz |

### 48 kHz Family

| Rate | Sample Rate | Multiplier |
|------|-------------|------------|
| DSD64/48 | 3,072,000 Hz | 64 × 48 kHz |
| DSD128/48 | 6,144,000 Hz | 128 × 48 kHz |
| DSD256/48 | 12,288,000 Hz | 256 × 48 kHz |
| DSD512/48 | 24,576,000 Hz | 512 × 48 kHz |

### Supported PCM Rates

44.1k family: 44100, 88200, 176400, 352800, 705600, 1411200 Hz
48k family: 48000, 96000, 192000, 384000, 768000, 1536000 Hz

## Rate Conversion Paths

| Path | Method | Notes |
|------|--------|-------|
| PCM → DSD (same family) | FIR upsample + SDM | 44.1k→DSD/44, 48k→DSD/48 |
| PCM → DSD (cross family) | Polyphase + FIR + SDM | 48k→DSD/44, 44.1k→DSD/48 via soxr/IPP |
| PCM → PCM (same family) | FIR chain | Power-of-2 ratio, no SDM |
| PCM → PCM (cross family) | Polyphase resampler | IPP (72 dB) or libsoxr (114 dB) |
| DSD → DSD (same family) | Boxcar/FIR + SDM | DSD/44↔DSD/44, DSD/48↔DSD/48 |
| DSD → PCM (same family) | FIR decimation | No SDM, multi-bit float output |
| DSD → PCM (cross family) | FIR decimate + polyphase | 2-stage: DSD→same-family PCM→target PCM |

### Polyphase Resampler

For cross-family PCM conversion (e.g., 44.1k↔48k), the plugin uses a polyphase resampler:

- **Default**: Intel IPP `ippsResamplePolyphase_32f` — ~72 dB SINAD at 44.1k↔48k, ~113 dB at 96k↔88.2k
- **Optional**: libsoxr VHQ — ~114 dB SINAD at all rate pairs. Runtime-loaded from `soxr.dll` in the component folder

When `Resampler = Auto`, libsoxr is used if `soxr.dll` is present, otherwise falls back to IPP.

### PCM Output Encoding

| Setting | Options | Notes |
|---------|---------|-------|
| Bit Depth | Auto (float), 16-bit, 24-bit, 32-bit, Float | Per-rate override available |
| Dither | Auto (TPDF for integer), None, TPDF, Noise-Shaped | Per-rate override available |

- **TPDF**: Triangular probability density dither, ±1 LSB. Eliminates quantization distortion. ~86 dB SNR at 16-bit.
- **Noise-Shaped**: First-order error feedback. Pushes dither noise to high frequencies where hearing is less sensitive. ~47 dB wideband SNR but perceptually superior.
- **Auto**: Float for float output, TPDF for integer output.

## SINAD Measurement Methodology

**Signal**: Multi-frequency median of three bin-aligned sine waves (900/1000/1100 Hz) at amplitude 0.5 (50% full scale), for robustness against SDM limit-cycle sensitivity at any single frequency. Each test frequency is adjusted slightly so that it falls exactly on a DFT bin boundary at the output sample rate — this prevents spectral leakage from corrupting the signal power measurement. The median of the three SINAD values is reported.

**Generation**: The test signal is encoded to DSD at the input rate using a Trellis SDM (depth=8, candidates=16, latency=512) with the auto-selected NTF filter for that rate. This produces a 1-bit DSD representation of the sine wave.

**Processing pipeline** (for rate conversion tests): DSD encode at `fs_in` → FIR rate conversion (127-tap Kaiser half-band, beta=10) → SDM re-encode at `fs_out`. Uses production path_config values (per-path NTF, FIR gain, state limiter, candidates, depth). For DSD-to-PCM tests: DSD encode → FIR decimation only (no SDM re-encode, output is multi-bit float32 PCM). Same-rate re-encode uses FIR lowpass (127-tap, 50 kHz cutoff) instead of the rate conversion FIR chain.

**Measurement**: Goertzel algorithm on the output stream. Signal power is measured at the test frequency bin. Noise power is the sum of all DFT bins from DC to 22,050 Hz (audio band), excluding the signal bin ±1 neighbour. SINAD = 10 × log10(signal_power / noise_power). For DSD-to-PCM tests, the FIR startup transient is skipped before measurement.

**Sample counts**: 262,144 samples (DSD64), 524,288 (DSD128), 1,048,576 (DSD256), 2,097,152 (DSD512) — approximately 93 ms of audio at each rate.

## Audio Quality Measurement

The **Test Quality** button in the settings dialog runs a comprehensive quality assessment for any configured rate conversion path — DSD→DSD, DSD→PCM, PCM→PCM, and PCM→DSD. The `/api/test_sinad` REST endpoint provides the same measurement for DSD paths.

### How It Works

1. **Test signal generation**: A bin-aligned ~1 kHz sine wave at 50% amplitude. For DSD paths, encoded through a Trellis SDM; for PCM paths, generated directly as float32.
2. **Processing**: Signal passes through the full conversion pipeline (FIR + SDM for DSD, FIR decimation for DSD→PCM, polyphase resampler for cross-family PCM)
3. **Analysis**: Goertzel algorithm measures signal and noise power across 0–20 kHz. FIR startup transient skipped for accurate measurement.
4. **Multiple signals**: Multitone uses 32 bin-aligned tones; noise modulation runs 4 passes at different amplitudes

### Metrics

#### SINAD (Signal-to-Noise-and-Distortion Ratio)

Measures the SDM's theoretical noise-shaping quality. A clean analog sine is fed directly to the SDM — no DSD quantization, no pipeline pre-filter. This isolates the SDM's performance from input limitations.

| Rating | SINAD (dB) | Interpretation |
|--------|-----------|----------------|
| Excellent | > 120 | Exceeds 20-bit PCM quality |
| Very Good | 100–120 | 16–20 bit equivalent dynamic range |
| Good | 80–100 | Solid DSD quality, typical for DSD64 |
| Fair | 60–80 | Audible noise floor, acceptable for background |
| Poor | < 60 | Significant noise, check NTF/settings |

#### A-weighted SINAD

Same as SINAD but with IEC 61672 A-weighting applied to the noise measurement. Human hearing is less sensitive to low frequencies (< 200 Hz) and very high frequencies (> 10 kHz). A-weighted SINAD is typically 3–6 dB higher than flat SINAD because noise at frequency extremes is downweighted.

**What to expect**: A-weighted should always be ≥ flat SINAD. A large difference (> 8 dB) indicates the noise is concentrated at frequencies where hearing is less sensitive — good news perceptually.

#### Multitone SINAD (32-tone)

Measures SDM quality with a complex signal resembling music. 32 tones at 1/10-decade spacing from 17 Hz to 21.2 kHz (matching Archimago's measurement methodology) are fed simultaneously through the SDM. Each tone is bin-aligned to prevent spectral leakage.

| Rating | Multitone (dB) | Interpretation |
|--------|---------------|----------------|
| Excellent | > 120 | Handles complex music with negligible distortion |
| Very Good | 90–120 | High-quality DSD encoding |
| Good | 70–90 | Normal DSD quality |
| Fair | 50–70 | Some intermodulation products present |
| Poor | < 50 | Significant intermodulation, check settings |

**What to expect**: Multitone SINAD is typically close to single-tone SINAD for well-designed NTFs. A large gap (> 20 dB) may indicate the SDM overloads with complex signals — try increasing the state limiter or reducing FIR gain.

#### Noise Modulation Index (NMod)

Measures how much the noise floor varies with signal level. An ideal SDM has a constant noise floor regardless of signal amplitude. Signal-dependent noise (noise modulation) is perceptually more noticeable than constant noise at the same level.

The test runs 4 SDM passes at amplitudes 0.05, 0.15, 0.30, and 0.50, measuring the noise floor at each level. NMod = max(noise floor) − min(noise floor) in dB.

| Rating | NMod (dB) | Interpretation |
|--------|----------|----------------|
| Excellent | < 3 | Constant noise floor, inaudible modulation |
| Good | 3–6 | Minor variation, unlikely to be audible |
| Fair | 6–10 | Moderate variation, may be noticeable in quiet passages |
| Poor | > 10 | Significant noise modulation, check NTF order/settings |

#### NMR (Noise-to-Mask Ratio)

Simplified perceptual metric inspired by PEAQ (ITU-R BS.1387). Measures whether the SDM's noise is below the psychoacoustic masking threshold — i.e., whether the noise is actually *audible* given the signal content.

The output is analyzed per Bark critical band (25 bands from 0–20 kHz). For each band, the noise energy is compared against a simplified masking threshold derived from the signal energy in that band.

| Rating | NMR (dB) | Label | Interpretation |
|--------|---------|-------|----------------|
| Excellent | ≤ −30 | Transparent | Noise completely masked — indistinguishable from perfect |
| Very Good | −30 to −20 | Excellent | Noise well below masking, inaudible |
| Good | −20 to −10 | Good | Noise mostly masked, minimal audibility |
| Fair | −10 to 0 | Fair | Noise approaching masking threshold |
| Poor | > 0 | Poor | Noise above masking — audible artifacts |

### REST API

```bash
# Measure DSD128 quality with default settings
curl "http://localhost:8881/api/test_sinad?rate=5644800&nc=2&depth=4&lat=128&fir=1&gain=0.708"

# Response:
{
  "sinad_theoretical": 129.4,
  "sinad_awtd": 134.6,
  "multitone_sinad": 101.2,
  "noise_mod_db": 9.8,
  "nmr_db": -117.2,
  "conv_fail": 0,
  "collapse": 0,
  "drop_pct": 0.0
}
```

Parameters: `rate` (DSD rate in Hz), `nc` (candidates), `depth`, `lat` (latency), `fir` (0=boxcar, 1=FIR lowpass), `gain` (linear, e.g., 0.708 for −3 dB), `ntf` (NTF filter ID, −1=auto).

### Reference Results

#### DSD → PCM Decimation (FIR only, no SDM)

| Path | SINAD | A-wtd | Multitone | NMod |
|------|-------|-------|-----------|------|
| DSD64 → PCM 44.1k | 104.7 | 106.9 | 93.6 | 3.7 |
| DSD128 → PCM 88.2k | 132.6 | 134.2 | 114.0 | 4.6 |
| DSD256 → PCM 176.4k | 133.9 | 135.7 | 117.4 | 3.5 |
| DSD64/48 → PCM 48k | 105.4 | 107.6 | 93.9 | 2.8 |

#### PCM → PCM Rate Conversion

| Path | Method | SINAD | A-wtd | Multitone |
|------|--------|-------|-------|-----------|
| 44.1k → 88.2k | FIR (2×) | 146.4 | 148.5 | 138.9 |
| 96k → 48k | FIR (÷2) | 146.1 | 148.2 | 141.1 |
| 44.1k → 48k | soxr polyphase | 95.3 | 102.2 | 62.7 |
| 96k → 88.2k | soxr polyphase | 104.4 | 106.5 | 78.7 |

## SINAD Results

### Trellis SDM (nc=2, production path_table NTFs)

Full end-to-end pipeline: generate DSD → boxcar DSD-Wide → SDM re-encode → Goertzel (multi-frequency median). Same-rate uses **boxcar (DSD-Wide) 4-bit intermediate** instead of FIR lowpass — the boxcar preserves DSD shaped noise as natural dither for the trellis re-encoder, giving +22 dB over FIR lowpass at DSD64.

**44.1 kHz family:**

| Rate | NTF | Depth | Cands | SINAD | A-wtd |
|------|-----|------:|------:|------:|------:|
| DSD64 | CLANS-6 | 16 | 2 | 84.5 | 90.0 |
| DSD128 | CLANS-6 | 4 | 2 | 99.5 | 105.6 |
| DSD256 | CLANS-6 | 4 | 2 | 108.5 | 114.4 |
| DSD512 | SDM-6 | 4 | 2 | 108.6 | 114.1 |

**48 kHz family:**

| Rate | NTF | Depth | Cands | SINAD | A-wtd |
|------|-----|------:|------:|------:|------:|
| DSD64/48 | SDM-6 | 16 | 2 | 84.6 | 91.0 |
| DSD128/48 | CLANS-6 | 4 | 2 | 103.4 | 110.4 |
| DSD256/48 | SDM-4 | 16 | 2 | 103.8 | 109.7 |
| DSD512/48 | SDM-4 | 16 | 2 | 109.8 | 116.0 |

Quality scales ~15 dB per octave of OSR (DSD64: 85 dB, DSD512: 109 dB). Published references: SACD spec 120 dB (encoding only), Archimago ~110-116 dB, HQPlayer ASDM7 ~110 dB.

`/fp:precise` required — `/fp:fast` causes up to 13 dB quality variation from FMA reordering (same root cause as CUDA `--fmad=false`). PGO provides 0% improvement (trellis is dependency-chain-bound, not branch/layout-bound).

### PreCorr SDM (greedy + prediction correction)

Greedy quantiser with trained prediction correction table. Near-zero CPU (~0.01x realtime).

| Rate | NTF Filter | SINAD | A-wtd |
|------|------------|------:|------:|
| DSD64 | CLANS-6 (order 6) | 117.2 | — |
| DSD128 | CLANS-7 (order 7) | 115.3 | — |
| DSD256 | CLANS-7 (order 7) | 135.7 | — |
| DSD512 | CLANS-7 (order 7) | 137.5 | — |

PreCorr outperforms Trellis at DSD64 (+6.5 dB) because its trained prediction table avoids the candidate collapse that degrades Trellis at low OSR. Trellis catches up at DSD128+ where higher OSR gives the look-ahead more room to work.

### DSD Rate Conversion — Path-Adaptive Tuning

Rate conversion uses production path_config values: per-path optimal NTF filter, FIR gain (-3 dB uniform), state limiter, candidates, depth.

**Pre-SDM processing pipeline by path type:**

| Path | Pre-SDM Processing | Why |
|------|-------------------|-----|
| **Same-rate** | Boxcar DSD-Wide (4-bit) | Preserves DSD noise as natural dither (+22 dB vs FIR LP) |
| **Upsample** | Raw DSD ±1.0 → fp64 FIR polyphase | DSD noise acts as dither; FIR handles anti-imaging |
| **→DSD512 upsample** | Raw DSD → 127-tap first stage FIR | Narrower transition band reduces noise accumulation |
| **Downsample** | Raw DSD ±1.0 → fp64 FIR polyphase | DSD noise as dither is optimal |
| **DSD256→128 DN** | 32-tap boxcar → fp64 FIR | Pre-smooth removes noise spikes (+11 dB /48) |
| **DSD→PCM** | FIR decimation only (no SDM) | Multi-bit output, no re-encoding needed |

Boxcar DSD-Wide: N-tap running average of ±1.0 DSD samples → multi-bit (log2(N)-bit) intermediate at DSD rate. Rate-adaptive taps: DSD64=32, DSD128/256=64, DSD512=128. The multi-bit intermediate preserves shaped noise as natural dither that the trellis SDM re-encoder tracks efficiently.

**44.1 kHz family — upsample** (end-to-end: DSD→fp64 FIR→SDM→Goertzel at output rate):

| Conversion | NTF | Gain | Lim | Cands | SINAD |
|------------|-----|------|-----|-------|------:|
| DSD64→DSD128 | SDM-7 | 0.71 | off | 2 | 99.7 |
| DSD64→DSD256 | CLANS-8 | 0.71 | off | 4 | 97.6 |
| DSD64→DSD512 | SDM-8 | 0.71 | on | 4 | 60.2 |
| DSD128→DSD256 | CLANS-6 | 0.71 | off | 4 | 105.9 |
| DSD128→DSD512 | CLANS-6 | 0.71 | off | 2 | 89.4 |
| DSD256→DSD512 | CLANS-8 | 0.71 | on | 2 | 114.2 |

**44.1 kHz family — downsample** (end-to-end):

| Conversion | NTF | Gain | Lim | Cands | SINAD |
|------------|-----|------|-----|-------|------:|
| DSD128→DSD64 | SDM-5 | 0.71 | off | 4 | 71.8 |
| DSD256→DSD64 | CLANS-8 | 0.71 | off | 8 | 73.0 |
| DSD512→DSD64 | SDM-6 | 0.71 | off | 8 | 67.7 |
| DSD256→DSD128 | CLANS-6 | 0.71 | off | 2 | 87.5 |
| DSD512→DSD128 | SDM-6 | 0.71 | off | 8 | 101.0 |
| DSD512→DSD256 | SDM-6 | 0.71 | on | 8 | 94.1 |

**48 kHz family — rate conversion** (independently swept):

| Conversion | NTF | Gain | Cands | SINAD |
|------------|-----|------|-------|------:|
| DSD64/48→DSD128/48 (UP) | SDM-4 | 0.71 | 2 | 94.7 |
| DSD64/48→DSD256/48 (UP) | SDM-7 | 0.71 | 8 | 82.9 |
| DSD128/48→DSD256/48 (UP) | CLANS-6 | 0.71 | 4 | 107.2 |
| DSD128/48→DSD64/48 (DN) | SDM-5 | 0.71 | 4 | 71.9 |
| DSD256/48→DSD64/48 (DN) | CLANS-8 | 0.71 | 8 | 73.6 |
| DSD256/48→DSD128/48 (DN) | SDM-6 | 0.71 | 8 | 92.3 |

**Key observations:**
- Same-rate: 85–109 dB via boxcar DSD-Wide (+22 dB vs FIR lowpass at DSD64)
- Upsample 2x paths: 95–106 dB — excellent quality
- DSD128→DSD512: 89 dB — 127-tap first FIR stage broke the 60 dB floor (+29 dB)
- DSD256→DSD512: 114 dB — single-step upsample preserves quality
- DSD64→DSD512: 60 dB — 4x+2x hybrid, fundamentally limited by DSD64 noise density
- Downsample paths: 68–101 dB. fp64 FIR critical (fp32 loses 3–40 dB on multi-stage)
- DSD256/48→DSD128/48: 92 dB — boxcar pre-smooth +11 dB over raw DSD input

**Measurement methodology**: End-to-end pipeline (generate DSD → boxcar or FIR → SDM re-encode → Goertzel, audio band 0–22 kHz). fp64 FIR matches production (Auto=fp64). Multi-frequency median (900/1000/1100 Hz) for robustness against SDM limit-cycle sensitivity.

### DSD to PCM Decimation

FIR-only decimation (no SDM re-encoding) for DSD-to-PCM conversion. Output is multi-bit float32 PCM. FIR startup transient skipped before measurement.

| Input | Output | Ratio | SINAD (dB) |
|-------|--------|------:|-----------:|
| DSD64 | PCM 44.1k | 64x | 103.3 |
| DSD64 | PCM 88.2k | 32x | 103.0 |
| DSD64 | PCM 176.4k | 16x | 85.7 |
| DSD128 | PCM 44.1k | 128x | 131.5 |
| DSD128 | PCM 88.2k | 64x | 132.1 |
| DSD128 | PCM 176.4k | 32x | 121.1 |
| DSD256 | PCM 44.1k | 256x | 133.6 |
| DSD256 | PCM 88.2k | 128x | 133.5 |
| DSD256 | PCM 176.4k | 64x | 133.3 |
| DSD512 | PCM 44.1k | 512x | 134.9 |
| DSD512 | PCM 88.2k | 256x | 136.2 |
| DSD512 | PCM 176.4k | 128x | 136.0 |
| DSD512 | PCM 352.8k | 64x | 136.5 |
| DSD64/48 | PCM 48k | 64x | 105.0 |
| DSD128/48 | PCM 96k | 64x | 132.2 |

**Key observations:**
- Higher input DSD rates yield better SINAD (more aggressive noise shaping, lower in-band noise)
- DSD64→PCM176.4k shows lower SINAD (85.7 dB) because 176.4k's Nyquist (88.2 kHz) reaches into the noise shaping transition region
- DSD128+ to PCM44.1k/88.2k all exceed 120 dB — well beyond CD quality
- A-weighted SINAD is 1.5–2 dB higher than flat (DSD noise concentrated at high frequencies)

### PCM to PCM Rate Conversion

Same-family conversions use the FIR half-band chain (power-of-2 ratio). Cross-family conversions use polyphase resampling (soxr preferred, IPP fallback).

| Input | Output | Method | SINAD (dB) | A-wtd (dB) | Multitone (dB) |
|-------|--------|--------|------------|------------|----------------|
| 44.1k | 88.2k | FIR 2× | 146.4 | 148.5 | 138.9 |
| 96k | 48k | FIR ÷2 | 146.1 | 148.2 | 141.1 |
| 44.1k | 176.4k | FIR 4× | 130.1 | 132.0 | — |
| 44.1k | 48k | soxr VHQ | 95.3 | 102.2 | 62.7 |
| 48k | 44.1k | soxr VHQ | 102.4 | 108.8 | — |
| 96k | 88.2k | soxr VHQ | 104.4 | 106.5 | 78.7 |
| 44.1k | 48k | IPP polyphase | 72.1 | 74.0 | — |

**Key observations:**
- FIR same-family: ~146 dB SINAD — near float32 precision limit (24-bit mantissa)
- soxr cross-family: 95–104 dB — excellent for arbitrary-ratio resampling
- IPP cross-family: ~72 dB — adequate fallback when soxr.dll not available
- With `Resampler = Auto` (default), soxr is used whenever soxr.dll is present

## Module Details

### DoP Format Bridge (`dop.c`)

Handles DSD-over-PCM encoding as defined by dCS:

- **`dop_detect`** -- Scans first 8 frames for alternating 0x05/0xFA markers in MSB of 24-bit PCM words
- **`dop_unpack`** -- Extracts 16 DSD bits per PCM frame into +/-1.0 float array
- **`dop_pack`** -- Repacks 1-bit float stream into DoP frames with markers
- **`bits_unpack` / `bits_pack`** -- Raw bitstream <-> float for native ASIO path

Float convention: `+1.0f` = logic 1, `-1.0f` = logic 0. Scaled via `2^23` for 24-bit PCM conversion.

### NTF Coefficient Tables (`ntf.c`)

40 noise transfer function filters across 4 DSD rates, ported from [mansr/sox](https://github.com/mansr/sox):

| Rate | Filters | Source |
|------|---------|--------|
| DSD64 (64x44100) | CLANS-4..8, SDM-4..8 | Reference sdm.c |
| DSD128 (128x44100) | CLANS-4..8, SDM-4..8 | Reference sdm.c |
| DSD256 (256x44100) | CLANS-4..8, SDM-4..8 | Reference sdm.c |
| DSD512 (512x44100) | CLANS-4..8, SDM-4..8 | Generated analytically |

Each filter contains:
- `a[order]` -- Feedback coefficients (CRFB topology)
- `g[order]` -- Resonator gain coefficients
- `order` -- Filter order (4-8)

**Trellis auto-selection** (via path_table in engine.c):
- DSD64/44: CLANS-6/d=16/lat=32 (85 dB). DSD64/48: SDM-6/d=16/lat=64 (85 dB).
- DSD128: CLANS-6/d=4/lat=128 (100 dB). DSD256: CLANS-6/d=4/lat=128 (109 dB).
- DSD512: SDM-6/d=4/lat=32 (109 dB). Depth=16 critical for DSD64 (4-bit dedup mask kills path diversity at low OSR).
- Values are end-to-end (DSD→boxcar DSD-Wide→SDM→Goertzel), not encoding-only.
- 48k family: independently swept — SDM-4 dominates at DSD256/48 (143.4) and DSD512/48 (139.2).

**PreCorr auto-selection**: CLANS-6 for DSD64, CLANS-7 for DSD128/DSD256/DSD512. (CLANS-8 is unstable with PreCorr's greedy quantizer at DSD256 boxcar input.)

**DSD512 generation** (`tools/gen_ntf_512.py`): Analytically derives DSD512 coefficients from DSD256 filters by scaling resonator gains and solving a linear system for feedback coefficients using polynomial NTF representation with tridiagonal determinant recurrence. Cross-validated against known filters (<1% Hinf error).

### Trellis SDM (`trellis.c`)

Viterbi look-ahead sigma-delta modulator, ported from mansr/sox sdm.c (LGPL v2.1+):

**Algorithm overview:**
1. For each input sample, expand all M candidates into 2M branches (output +1 or -1)
2. Compute NTF state advance for each branch via CRFB filter topology
3. Score each branch by accumulated squared error (cost)
4. Deduplicate paths via hash table, sort by cost, keep M best survivors
5. Output the traceback bit from the lowest-cost candidate's history buffer

**Latency**: configurable via `trellis_lat` (16-2048 DSD samples). First `trellis_lat` input samples fill the latency buffer with no output; `sdm_drain` flushes remaining samples at end of stream.

**State preservation**: Trellis integrator state is preserved across flush/stop when `antipop` is enabled (see Anti-Pop System).

### PreCorr SDM (`precorr.c`)

Greedy sigma-delta modulator with prediction correction table:

1. Greedy quantise: `y = (v >= 0) ? +1 : -1`
2. Apply learned correction from `pred_table[history][phase]`
3. Re-quantise corrected output
4. Update 8-bit output history register

**Table training**: At init, runs 65536 pseudo-random noise samples through a greedy SDM, accumulating mean corrections per (8-bit history, phase) pair.

**Zero latency**: Output count equals input count from first sample. No drain needed.

### FIR Rate Conversion (`fir.c`)

Intel IPP FIRMR polyphase multi-rate FIR for power-of-2 DSD rate conversion:

**Half-band filter (rate conversion):**
- 63-tap Kaiser-windowed sinc (beta=12.0, ~120 dB stopband)
- Coefficients computed at init time via Bessel I0 series expansion (25 terms)

**Same-rate lowpass (DSD re-encode input conditioning):**
- 127-tap Kaiser-windowed sinc (beta=10.0, ~100 dB stopband, 50 kHz cutoff)
- Sharper transition band (22 kHz vs 45 kHz with 63 taps) reduces ultrasonic noise leakage from original DSD noise shaping into the SDM re-encoder input

**Upsample 2x:** `ippsFIRMR_32f` polyphase (upFactor=2, downFactor=1) — no explicit zero-stuffing or scaling

**Downsample 2x:** `ippsFIRMR_32f` polyphase (upFactor=1, downFactor=2) — no explicit decimation

**Multi-stage chaining:** up to 9 stages (512x ratio, e.g., PCM 44.1k → DSD512) with ping-pong scratch buffers.

### DSD-Wide Volume Control (`engine.c`)

Digital volume control for DSD without leaving the DSD domain. Traditional approaches decimate DSD to multi-bit PCM, apply gain, then re-modulate back to DSD -- this introduces unnecessary latency and quality loss from the round-trip rate conversion. DSD-Wide stays at the original DSD sample rate throughout.

**The problem:** DSD is a 1-bit stream (values are strictly +1.0 or -1.0). Multiplying by a gain factor (e.g., 0.5) produces values like +0.5/-0.5, but the signal is still binary -- it carries the full ultrasonic quantisation noise of the original DSD encoding. Feeding this directly into an SDM causes it to saturate, producing constant -1.0 output (silence).

**The solution -- boxcar smoothing:**

```
DSD ±1.0 @ Fs  -->  boxcar(N)  -->  multi-bit @ Fs  -->  × gain  -->  SDM  -->  DSD ±1.0 @ Fs
```

1. **Boxcar filter** (N=8 taps): A running average over N consecutive DSD samples. Converts the 1-bit stream to ~4-bit resolution ({-1.0, -0.75, -0.5, ..., +0.75, +1.0}) at the **same** sample rate.

2. **Gain multiply**: Applied to the multi-bit smoothed signal. The SDM now receives a properly-shaped waveform it can encode.

3. **SDM re-encode**: Trellis or PreCorr converts back to 1-bit DSD. The NTF reshapes quantisation noise optimally.

### Processing Engine (`engine.c`)

Per-channel orchestrator:
- Configures FIR chain based on input/output rate ratio
- **Path-adaptive SDM tuning**: When `NTF filter = Auto` and rate converting, selects optimal NTF filter, FIR gain, integrator limiter, candidates, depth, and latency per conversion path from a lookup table derived from comprehensive sweep measurements
- **Global FIR gain**: Uniform -3 dB (0.708) attenuation across all paths for consistent volume. User-configurable from 0 dB to -12 dB.
- **DSD-Wide volume control**: Boxcar smoothing + gain + SDM for same-rate DSD
- Applies gain multiply after FIR, before SDM (for rate conversion paths)
- Dispatches to Trellis or PreCorr based on `sdm_mode` config
- Handles mute (silence pattern substitution)
- **Engine reinit on SDM mode change**: automatically reinitializes when per-rate override changes between Trellis and PreCorr

### Configuration (`config.c`)

Runtime parameters serialized to foobar2000 config store (version 16):

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| SDM mode | enum | PreCorr / Trellis | PreCorr |
| Output DSD rate | per-rate | Bypass / DSD64-512 / PCM | Bypass |
| Volume | float | 0.0 - 1.0 | 1.0 |
| Mute | bool | on/off | off |
| NTF filter | per-rate | Auto / CLANS-4..8 / SDM-4..8 | Auto |
| Trellis depth (N) | per-rate | Auto / 4-8 | Auto |
| Trellis candidates (M) | per-rate | Auto / 2-32 | Auto |
| State limiter | per-rate | Auto / Off / 3-20 | Auto |
| ML noise filter | per-rate | Auto / Off / On | Auto |
| FIR gain | global | Auto (-3 dB) / 0 to -12 dB | Auto |
| FIR mode | per-rate | Auto / Boxcar / FIR | Auto |
| FIR precision | per-rate | Auto (FP64) / FP32 / FP64 | Auto |
| Trellis latency | per-rate | Auto / 16-512 | Auto |
| GPU FIR | per-rate | Auto / Off / On | Auto |
| PCM bit depth | global + per-rate | Auto (float) / 16 / 24 / 32 / Float | Auto |
| PCM dither | global + per-rate | Auto / None / TPDF / Shaped | Auto |
| Resampler | global | Auto / IPP / soxr | Auto |
| soxr quality | global | Medium / High / Very High | High |
| Anti-pop lead-in | bool | on/off | on |
| Thread count | int | 0 (auto) - cores | 0 |
| Input format | enum | Auto / DoP / Native | Auto |
| Debug log | bool | on/off | off |
| REST API port | int | 0 (disabled) - 65535 | 8881 |

### ONNX ML Post-Filter (`onnx_filter.c`)

Optional non-causal CNN post-filter for DSD noise reduction. Delay-loads `onnxruntime.dll` — plugin works without it. Supports CPU and DirectML (GPU) execution providers. Model file: `foo_dsd_trellis_ml.onnx` next to the DLL.

### TUSBAudio Integration (`tusbaudio.c`)

Runtime detection and status logging for XMOS-based USB DACs (Topping, Fosi, SMSL, Gustard, etc.) via Thesycon TUSBAudio driver API. Scans common driver install paths, enumerates devices, reads capabilities (sample rates, DSD support, streaming mode).

## Building

### Requirements

- Visual Studio 2022 with MSVC v142 toolset
- Windows 10 SDK
- foobar2000 SDK (place in `foobar2000-sdk/`)
- Intel IPP (NuGet packages: `intelipp.devel.win-x64`, `intelipp.static.win-x64`)

### Build with MSBuild

```
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" ^
    foo_dsd_trellis.sln /p:Configuration=Release /p:Platform=x64
```

### Run tests

```bash
bin\Release\x64\foo_dsd_trellis_test.exe
```

## Project Structure

```
foo_dsd_trellis/
|-- src/
|   |-- dsp_fb2k.cpp          fb2k DSP v2 C++ wrapper
|   |-- dsp_plugin.c          Plugin state management
|   |-- config.c              Configuration serialization (v17)
|   |-- dop.c                 DoP detection, pack/unpack
|   |-- bitpack.c             Native ASIO bitstream pack/unpack
|   |-- engine.c              Per-channel processing orchestrator
|   |-- fir.c                 IPP FIRMR polyphase FIR (rate conversion)
|   |-- trellis.c             Viterbi look-ahead trellis SDM
|   |-- precorr.c             Greedy + prediction correction SDM
|   |-- ntf.c                 NTF coefficient tables (40 filters)
|   |-- threadpool.c          Worker thread pool (MMCSS)
|   |-- simd_detect.c         CPU feature detection
|   |-- cpuset.c              CPU topology and dynamic CPUSET
|   |-- httpapi.c             REST API server
|   |-- resample.c            Polyphase resampler (IPP + soxr runtime)
|   |-- sinad_measure.c       Audio quality measurement engine
|   |-- onnx_filter.c         ONNX ML post-filter (delay-loaded)
|   |-- tusbaudio.c           TUSBAudio XMOS DAC integration
|   +-- wav_io.c              WAV file read/write
|-- include/
|   |-- dsd_types.h           Core types, constants, enums
|   |-- engine.h              Engine API
|   |-- trellis.h             Trellis SDM API
|   |-- precorr.h             PreCorr SDM API
|   |-- ntf.h                 NTF filter API
|   |-- fir.h                 FIR chain API
|   |-- dop.h                 DoP API
|   |-- threadpool.h          Thread pool API
|   |-- simd_detect.h         CPU detection API
|   |-- cpuset.h              CPU topology API
|   |-- resample.h            Polyphase resampler API
|   |-- sinad_measure.h       Quality measurement API
|   |-- httpapi.h             REST API
|   |-- onnx_filter.h         ONNX ML API
|   |-- tusbaudio.h           TUSBAudio API
|   +-- wav_io.h              WAV I/O API
|-- test/
|   |-- test.h                Minimal test framework (no dependencies)
|   |-- test_main.c           Test runner with suite selection
|   |-- test_dop.c            DoP tests
|   |-- test_ntf.c            NTF tests
|   |-- test_fir.c            FIR tests
|   |-- test_trellis.c        Trellis SDM tests (SINAD measurement)
|   |-- test_precorr.c        PreCorr SDM tests (SINAD measurement)
|   |-- test_rate_sinad.c     Rate conversion SINAD + gain sweep tests
|   |-- test_config.c         Config serialization tests
|   |-- test_simd.c           CPU detection and IPP tests
|   |-- test_hardening.c      Edge cases and robustness
|   |-- test_threadpool.c     Thread pool tests
|   |-- test_onnx_filter.c    ONNX ML filter tests
|   |-- test_resample.c       Polyphase resampler tests (IPP + soxr)
|   |-- test_validation.c     Rate map validation tests
|   |-- test_dither.c         PCM dither and bit depth tests
|   |-- test_quality.c        Quality metrics tests (all path types)
|   +-- test_sinad_diag.c     Extended SINAD diagnostics
|-- training_pipeline/         ML model training scripts (Python/PyTorch)
|-- tools/
|   |-- gen_ntf_512.py        DSD512 NTF coefficient generator
|   +-- dsd_encode.c          Standalone DSD encoding tool
|-- reference/
|   +-- sdm.c                 Source reference from mansr/sox (LGPL v2.1+)
|-- foobar2000-sdk/            Vendored fb2k SDK
+-- foo_dsd_trellis.sln        Visual Studio solution (7 projects)
```

## Test Results

1016 tests across 18 suites, all passing:

| Suite | Tag | Tests | Coverage |
|-------|-----|-------|----------|
| DoP | `dop` | 24 | Detection, pack/unpack, round-trip, edge cases |
| NTF | `ntf` | 18 | All 40 filters, coefficient verification, auto-select |
| FIR | `fir` | 17 | Passband/stopband, round-trip, chain tests |
| Trellis SDM | `trellis` | 13 | Init, reset, latency, drain, SINAD (4 rates), DC stability |
| PreCorr SDM | `precorr` | 8 | Init, binary output, no latency, SINAD (4 rates) |
| Rate Conversion | `rate` | 37 | SINAD for all DSD/44 + DSD/48 upsample/downsample + DSD-to-PCM decimation |
| Config | `config` | 99 | Serialization, versioning (v1-v17), validation, rate/NTF/limiter maps |
| CPU & IPP | `simd` | 5 | CPU detection, IPP kernel, FIR correctness |
| Hardening | `hardening` | 24 | Edge cases, robustness |
| Thread Pool | `threadpool` | 8 | Create/destroy, concurrent SDM, stress |
| ONNX ML | `onnx` | 7 | Runtime probe, null safety, session create, live inference, SINAD |
| GPU Compute | `gpu` | — | DX11/DX12/CUDA FIR, boxcar, SDM kernels |
| Resample | `resample` | 14 | IPP polyphase + soxr SINAD at all cross-family rate pairs |
| Validation | `validation` | 150 | Rate map indices, family helpers, output rules for all 20×25 combinations |
| Dither | `dither` | 6 | Truncation (94 dB), TPDF (86 dB), noise-shaped, float passthrough |
| Quality Metrics | `quality` | 62 | A-weight curve, DSD/DSD→PCM/PCM→PCM quality matrix (10 paths) |
| Rate Conv Sweep | `sweep` | 6 | FIR-only SINAD, limiter sweep, NTF × limiter sweep (extended) |
| SINAD Diagnostics | `diag` | 7 | NTF sweeps, warmup analysis (extended) |

## References

- Reefman, D. & Janssen, E. (2002). "Signal processing for Direct Stream Digital." Philips Research.
- Harpe, P. et al. (2003). "Trellis-type sigma delta modulators for DSD." AES Convention.
- Schreier, R. & Temes, G. "Understanding Delta-Sigma Data Converters." Wiley.
- Hawksford, M. (2008). "Parallel Look-Ahead Digital SDM with Energy-Balance." JAES.
- [mansr/sox](https://github.com/mansr/sox) -- Reference SDM implementation (LGPL v2.1+)

## License

NTF coefficient tables and SDM algorithm ported from mansr/sox under LGPL v2.1+. See `reference/sdm.c` for original source.
