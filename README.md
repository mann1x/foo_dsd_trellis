# foo_dsd_trellis

A native foobar2000 DSP plugin that converts PCM and DSD audio to DSD using sigma-delta modulation. Supports two SDM modes: **Trellis** (Viterbi look-ahead, highest quality) and **PreCorr** (greedy + prediction correction, near-zero CPU). FIR rate conversion uses Intel IPP with automatic SSE2/AVX2/AVX-512 dispatch.

## Features

| Feature | Description |
|---------|-------------|
| SDM Modes | PreCorr (default, ~0.01x RT) and Trellis (high quality, configurable depth/candidates) |
| Rate Conversion | DSD64-512 (44.1kHz + 48kHz families) via Intel IPP FIRSR (63-tap Kaiser half-band) |
| PCM to DSD | Float32 PCM input upsampled via FIR then quantised to 1-bit DSD |
| PCM to PCM | Same-family (FIR chain) and cross-family (polyphase resampler: IPP or libsoxr) |
| PCM Encoding | Output bit depth (16/24/32/float) with TPDF or noise-shaped dither, global + per-rate |
| DSD/48 Rates | Full support for 48kHz-family DSD: DSD64/48 (3.072 MHz) through DSD512/48 (24.576 MHz) |
| High PCM | Input/output up to 1536 kHz (44.1k and 48k families) |
| Per-Rate Config | SDM mode, candidates, depth, NTF, state limiter, ML, PCM bits/dither configurable per input rate |
| FIR Gain | Global FIR gain limiter (Auto = -3 dB) for uniform volume across all rate conversion paths |
| Volume Control | DSD-Wide: boxcar smoothing + gain + SDM re-encode (no PCM decimation) |
| Anti-Pop | Three-layer anti-pop: SDM state preservation + DoP 0x69 silence trail + rate-switch DAC mute trick |
| Parallel SDM | Chunk segmentation with FIR tail warmup for real-time DSD256/DSD512 Trellis processing |
| Mute | Silence pattern substitution (0x69/0x96) |
| DoP Detection | Auto-detect DoP markers (0x05/0xFA) in 24-bit PCM frames |
| Output Modes | DoP (native DSD output) or PCM (for VU meter / non-DSD DACs) |
| Intel IPP | Statically linked, automatic CPU dispatch (SSE2 -> AVX2 -> AVX-512) |
| ONNX ML Filter | Optional causal CNN post-filter for DSD noise reduction (delay-loaded onnxruntime.dll) |
| Property Page | Full configuration dialog with per-rate settings, path info display, dark mode support |
| Config Versioning | Forward-compatible binary preset serialization (v16) with legacy fallback |
| REST API | HTTP control/monitoring API on port 8881 |
| CPU Topology | Dynamic CPUSET core selection with scheduling_class priority, SMT/CCD/E-core awareness |
| TUSBAudio | Runtime XMOS DAC detection and status logging |

## Architecture

Four-layer design with clean separation of concerns:

```
1-bit in -> unpack -> float32 @ Fs_in -> [IPP FIRSR] -> x gain -> SDM (Trellis|PreCorr) -> 1-bit out @ Fs_out
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

### Parallel SDM Segmentation

For Trellis mode at high DSD rates (DSD256/DSD512), single-core SDM processing exceeds real-time. The engine splits FIR output into N segments (up to 4) processed in parallel by independent SDM contexts:

1. **Segment 0** (first chunk): uses persistent SDM context for initial latency fill
2. **Segments 0-3** (subsequent chunks): all use temp SDMs with warmup from previous chunk's FIR tail
3. **Overlap warmup**: each temp SDM processes `2 × trellis_lat` warmup samples before its segment boundary, then discards `trellis_lat` output samples for convergence
4. **FIR tail preservation**: last `overlap` samples of FIR output saved per-channel, prepended to next chunk's segment 0 input

This eliminates the persistent SDM state dependency between chunks that previously caused periodic audio glitches.

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

**Signal**: Bin-aligned 1 kHz sine wave at amplitude 0.5 (50% full scale). The test frequency is adjusted slightly from 1000 Hz so that it falls exactly on a DFT bin boundary at the output sample rate — this prevents spectral leakage from corrupting the signal power measurement.

**Generation**: The test signal is encoded to DSD at the input rate using a Trellis SDM (depth=8, candidates=16, latency=512) with the auto-selected NTF filter for that rate. This produces a 1-bit DSD representation of the sine wave.

**Processing pipeline** (for rate conversion tests): DSD encode at `fs_in` → FIR rate conversion (63-tap Kaiser half-band, beta=12) → SDM re-encode at `fs_out`. Uses production path_config values (per-path NTF, FIR gain, state limiter, candidates, depth). For DSD-to-PCM tests: DSD encode → FIR decimation only (no SDM re-encode, output is multi-bit float32 PCM).

**Measurement**: Goertzel algorithm on the output stream. Signal power is measured at the test frequency bin. Noise power is the sum of all DFT bins from DC to 22,050 Hz (audio band), excluding the signal bin ±1 neighbour. SINAD = 10 × log10(signal_power / noise_power). For DSD-to-PCM tests, the FIR startup transient is skipped before measurement.

**Sample counts**: 262,144 samples (DSD64), 524,288 (DSD128), 1,048,576 (DSD256), 2,097,152 (DSD512) — approximately 93 ms of audio at each rate.

## Audio Quality Measurement

The **Test Quality** button in the settings dialog and the `/api/test_sinad` REST endpoint run a comprehensive quality assessment of the SDM configuration for the selected DSD rate. Four complementary metrics evaluate different aspects of audio quality.

### How It Works

1. **Test signal generation**: A bin-aligned ~1 kHz sine wave at 50% amplitude, encoded as clean analog (for SINAD/NMR) or as a 32-tone composite (for multitone)
2. **SDM processing**: Signal is fed through the Trellis SDM with the selected NTF, candidates, depth, and latency settings
3. **Analysis**: Goertzel algorithm measures signal and noise power across 0–20 kHz (the audible band)
4. **Multiple amplitudes**: Noise modulation runs 4 passes at different signal levels to detect signal-dependent noise

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

Measured with default auto settings (nc=2, depth=4):

| Rate | SINAD | A-wtd | Multitone | NMod | NMR |
|------|-------|-------|-----------|------|-----|
| DSD64 | 96.8 | 101.1 | 99.7 | 8.4 | −85.1 |
| DSD128 | 129.4 | 134.6 | 101.2 | 9.8 | −117.2 |
| DSD256 | 115.6 | 120.7 | 132.3 | 6.8 | −106.6 |

## SINAD Results

### Trellis SDM (depth=8, candidates=8, latency=512)

Viterbi look-ahead search. Highest quality, higher CPU.

| Rate | NTF Filter | SINAD (dB) |
|------|------------|------------|
| DSD64 | CLANS-5 (order 5) | 86.9 |
| DSD128 | CLANS-6 (order 6) | 112.2 |
| DSD256 | CLANS-7 (order 7) | 136.7 |
| DSD512 | CLANS-8 (order 8) | 139.8 |

### PreCorr SDM (greedy + prediction correction)

Greedy quantiser with trained prediction table. Near-zero CPU (~0.01x realtime).

| Rate | NTF Filter | SINAD (dB) |
|------|------------|------------|
| DSD64 | CLANS-6 (order 6) | 117.2 |
| DSD128 | CLANS-7 (order 7) | 115.3 |
| DSD256 | CLANS-7 (order 7) | 135.7 |
| DSD512 | CLANS-7 (order 7) | 137.5 |

### DSD Rate Conversion — Path-Adaptive Tuning

Rate conversion uses production path_config values: per-path optimal NTF filter, FIR gain (-3 dB uniform), state limiter, candidates=2, depth=4 for DSD256/512 output. All paths use 0.708 (-3 dB) FIR gain for consistent volume across rate transitions.

**Upsample paths:**

| Conversion | NTF Filter | FIR Gain | Limiter | Cands | Depth | SINAD (dB) |
|------------|------------|----------|---------|-------|-------|------------|
| DSD64 -> DSD128 | SDM-4 | 0.71 | off | 2 | 4 | 91.1 |
| DSD64 -> DSD256 | CLANS-8 | 0.71 | off | 2 | 4 | 89.8 |
| DSD64 -> DSD512 | CLANS-6 | 0.71 | 10.0 | 2 | 4 | 60.3 |
| DSD128 -> DSD256 | CLANS-8 | 0.71 | off | 2 | 4 | 107.3 |
| DSD128 -> DSD512 | CLANS-8 | 0.71 | 12.0 | 2 | 4 | 60.2 |
| DSD256 -> DSD512 | CLANS-8 | 0.71 | 6.0 | 2 | 4 | 118.8 |

**Downsample paths:**

| Conversion | NTF Filter | FIR Gain | Limiter | Cands | Depth | SINAD (dB) |
|------------|------------|----------|---------|-------|-------|------------|
| DSD128 -> DSD64 | CLANS-4 | 0.71 | off | 32 | 8 | 75.3 |
| DSD256 -> DSD64 | CLANS-8 | 0.71 | off | 8 | 8 | 74.3 |
| DSD512 -> DSD64 | SDM-6 | 0.71 | off | 8 | 8 | 74.0 |
| DSD256 -> DSD128 | CLANS-4 | 0.71 | off | 8 | 8 | 91.3 |
| DSD512 -> DSD128 | SDM-4 | 0.71 | 16.0 | 16 | 8 | 98.0 |
| DSD512 -> DSD256 | SDM-6 | 0.71 | 16.0 | 8 | 8 | 97.1 |

**Key observations:**
- All paths use uniform FIR gain of 0.708 (-3 dB) to prevent volume changes across rate transitions
- DSD64->DSD512 and DSD128->DSD512 have lower SINAD (~60 dB) due to SDM overload at -3 dB gain with 8x/4x FIR upsample peaks — users can increase FIR gain attenuation via the global setting for these paths
- Path-adaptive settings (NTF, limiter, cands, depth) applied automatically when NTF = Auto
- Per-rate overrides available for SDM mode, candidates, depth, and state limiter

### DSD to PCM Decimation

FIR-only decimation (no SDM re-encoding) for DSD-to-PCM conversion. Output is multi-bit float32 PCM. Measured with Goertzel SINAD on the steady-state portion (startup transient skipped).

| Input | Output | Ratio | SINAD (dB) |
|-------|--------|-------|------------|
| DSD64 | PCM 44.1k | 64x | 103.7 |
| DSD64 | PCM 88.2k | 32x | 103.1 |
| DSD64 | PCM 176.4k | 16x | 93.8 |
| DSD128 | PCM 44.1k | 128x | 131.8 |
| DSD128 | PCM 88.2k | 64x | 128.1 |
| DSD128 | PCM 176.4k | 32x | 103.5 |
| DSD256 | PCM 44.1k | 256x | 133.9 |
| DSD256 | PCM 88.2k | 128x | 133.5 |
| DSD256 | PCM 176.4k | 64x | 133.4 |
| DSD512 | PCM 44.1k | 512x | 135.6 |
| DSD512 | PCM 88.2k | 256x | 136.9 |
| DSD512 | PCM 176.4k | 128x | 136.4 |
| DSD512 | PCM 352.8k | 64x | 137.1 |
| DSD64/48 | PCM 48k | 64x | 105.0 |
| DSD128/48 | PCM 96k | 64x | 132.4 |

**Key observations:**
- Higher input DSD rates yield better SINAD (more aggressive noise shaping, lower in-band noise)
- DSD64->PCM176.4k shows lower SINAD (93.8 dB) because 176.4k's Nyquist (88.2 kHz) reaches into the noise shaping transition region
- DSD128+ to PCM44.1k/88.2k all exceed 100 dB — well beyond CD quality

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

**Trellis auto-selection**: CLANS-5 for DSD64, CLANS-6 for DSD128, CLANS-8 for DSD256/DSD512.

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

Intel IPP FIRSR-based half-band FIR for power-of-2 DSD rate conversion:

**Filter design:**
- 63-tap Kaiser-windowed sinc (beta=12.0, ~120 dB stopband)
- Coefficients computed at init time via Bessel I0 series expansion (25 terms)

**Upsample 2x:** zero-stuff -> `ippsFIRSR_32f` -> scale by 2

**Downsample 2x:** `ippsFIRSR_32f` -> decimate (keep every other sample)

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

Runtime parameters serialized to foobar2000 config store (version 12):

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
|   |-- config.c              Configuration serialization (v12)
|   |-- dop.c                 DoP detection, pack/unpack
|   |-- bitpack.c             Native ASIO bitstream pack/unpack
|   |-- engine.c              Per-channel processing orchestrator
|   |-- fir.c                 IPP FIRSR half-band FIR (rate conversion)
|   |-- trellis.c             Viterbi look-ahead trellis SDM
|   |-- precorr.c             Greedy + prediction correction SDM
|   |-- ntf.c                 NTF coefficient tables (40 filters)
|   |-- threadpool.c          Worker thread pool (MMCSS)
|   |-- simd_detect.c         CPU feature detection
|   |-- cpuset.c              CPU topology and dynamic CPUSET
|   |-- httpapi.c             REST API server
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

954 tests across 17 suites, all passing:

| Suite | Tag | Tests | Coverage |
|-------|-----|-------|----------|
| DoP | `dop` | 24 | Detection, pack/unpack, round-trip, edge cases |
| NTF | `ntf` | 18 | All 40 filters, coefficient verification, auto-select |
| FIR | `fir` | 17 | Passband/stopband, round-trip, chain tests |
| Trellis SDM | `trellis` | 13 | Init, reset, latency, drain, SINAD (4 rates), DC stability |
| PreCorr SDM | `precorr` | 8 | Init, binary output, no latency, SINAD (4 rates) |
| Rate Conversion | `rate` | 37 | SINAD for all DSD/44 + DSD/48 upsample/downsample + DSD-to-PCM decimation |
| Config | `config` | 99 | Serialization, versioning (v1-v16), validation, rate/NTF/limiter maps |
| CPU & IPP | `simd` | 5 | CPU detection, IPP kernel, FIR correctness |
| Hardening | `hardening` | 24 | Edge cases, robustness |
| Thread Pool | `threadpool` | 8 | Create/destroy, concurrent SDM, stress |
| ONNX ML | `onnx` | 7 | Runtime probe, null safety, session create, live inference, SINAD |
| GPU Compute | `gpu` | — | DX11/DX12/CUDA FIR, boxcar, SDM kernels |
| Resample | `resample` | 14 | IPP polyphase + soxr SINAD at all cross-family rate pairs |
| Validation | `validation` | 150 | Rate map indices, family helpers, output rules for all 20×25 combinations |
| Dither | `dither` | 6 | Truncation, TPDF, noise-shaped, 24-bit, float passthrough |
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
