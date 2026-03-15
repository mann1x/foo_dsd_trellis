# foo_dsd_trellis

A native foobar2000 DSP plugin that converts PCM and DSD audio to DSD using sigma-delta modulation. Supports two SDM modes: **Trellis** (Viterbi look-ahead, highest quality) and **PreCorr** (greedy + prediction correction, near-zero CPU). FIR rate conversion uses Intel IPP with automatic SSE2/AVX2/AVX-512 dispatch.

## Features

| Feature | Description |
|---------|-------------|
| SDM Modes | PreCorr (default, ~0.01x RT) and Trellis (high quality, configurable depth/candidates) |
| Rate Conversion | DSD64 / DSD128 / DSD256 / DSD512 via Intel IPP FIRSR (63-tap Kaiser half-band) |
| PCM to DSD | Float32 PCM input upsampled via FIR then quantised to 1-bit DSD |
| Volume Control | DSD-Wide: boxcar smoothing + gain + SDM re-encode (no PCM decimation) |
| Passthrough | Repack only -- bypass FIR and SDM when input/output rate and gain are identical |
| Mute | Silence pattern substitution (0x69/0x96) |
| DoP Detection | Auto-detect DoP markers (0x05/0xFA) in 24-bit PCM frames; falls back to native ASIO |
| Native DSD | Raw DSD bitstream support (FORMAT_NATIVE) for ASIO and native input components |
| Output Modes | DoP (native DSD output) or PCM (for VU meter / non-DSD DACs) |
| Intel IPP | Statically linked, automatic CPU dispatch (SSE2 -> AVX2 -> AVX-512) |
| Property Page | Full configuration dialog with dark mode support |
| Config Versioning | Forward-compatible binary preset serialization with legacy fallback |
| REST API | HTTP control/monitoring API with render and SINAD measurement endpoints |
| foo_input_udsd | Compatible with foo_input_udsd / foo_input_sacd DSD input components |

## Architecture

Four-layer design with clean separation of concerns:

```
1-bit in -> unpack -> float32 @ Fs_in -> [IPP FIRSR] -> x gain -> SDM (Trellis|PreCorr) -> 1-bit out @ Fs_out
```

| Layer | Purpose | Files |
|-------|---------|-------|
| fb2k Interface | foobar2000 DSP v2 glue, config dialog | `dsp_fb2k.cpp`, `dsp_plugin.c`, `config.c` |
| Format Bridge | DoP/native detection, 1-bit <-> float32 | `dop.c`, `bitpack.c` |
| Processing Engine | IPP FIR rate conversion + gain | `engine.c`, `fir.c` |
| SDM | Trellis (Viterbi) or PreCorr (greedy + prediction) | `trellis.c`, `precorr.c`, `ntf.c` |
| Infrastructure | Thread pool, CPU detection, REST API | `threadpool.c`, `simd_detect.c`, `cpuset.c`, `httpapi.c` |

Thread pool (`threadpool.c`) provides per-channel parallelism with MMCSS "Pro Audio" scheduling.

## DSD Rates

| Rate | Sample Rate | Multiplier |
|------|-------------|------------|
| DSD64 | 2,822,400 Hz | 64 x 44.1 kHz |
| DSD128 | 5,644,800 Hz | 128 x 44.1 kHz |
| DSD256 | 11,289,600 Hz | 256 x 44.1 kHz |
| DSD512 | 22,579,200 Hz | 512 x 44.1 kHz |

## SINAD Measurement Methodology

**Signal**: Bin-aligned 1 kHz sine wave at amplitude 0.5 (50% full scale). The test frequency is adjusted slightly from 1000 Hz so that it falls exactly on a DFT bin boundary at the output sample rate — this prevents spectral leakage from corrupting the signal power measurement.

**Generation**: The test signal is encoded to DSD at the input rate using a Trellis SDM (depth=8, candidates=16, latency=512) with the auto-selected NTF filter for that rate. This produces a 1-bit DSD representation of the sine wave.

**Processing pipeline** (for rate conversion tests): DSD encode at `fs_in` → FIR rate conversion (63-tap Kaiser half-band, beta=12) → SDM re-encode at `fs_out`. The FIR performs zero-stuff + lowpass (upsample) or lowpass + decimate (downsample), chained for multi-stage conversions (e.g., 8x = three 2x stages).

**Measurement**: Goertzel algorithm on the output DSD stream. Signal power is measured at the test frequency bin. Noise power is the sum of all DFT bins from DC to 22,050 Hz (audio band), excluding the signal bin ±1 neighbour. SINAD = 10 × log10(signal_power / noise_power).

**Sample counts**: 262,144 samples (DSD64), 524,288 (DSD128), 1,048,576 (DSD256), 2,097,152 (DSD512) — approximately 93 ms of audio at each rate.

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
| DSD256 | CLANS-8 (order 8) | 125.6 |
| DSD512 | CLANS-7 (order 7) | 137.5 |

### DSD Rate Conversion — Path-Adaptive Tuning

Rate conversion SINAD depends heavily on the combination of NTF filter, integrator limiter, trellis candidates, and latency. A comprehensive sweep (1,152 measurements across 12 paths × 10 filters × 8 limiter values × 4 candidate counts × 4 latencies) identified the optimal configuration per path. These are applied automatically when `NTF filter = Auto`.

**Optimal configuration per rate conversion path:**

| Conversion | NTF Filter | Limiter | Cands | Latency | SINAD (dB) |
|------------|------------|---------|-------|---------|------------|
| DSD64 -> DSD128 | CLANS-6 | off | 8 | 256 | 97.6 |
| DSD64 -> DSD256 | SDM-7 | off | 8 | 256 | 80.6 |
| DSD64 -> DSD512 | SDM-8 | 10.0 | 8 | 256 | 54.4 |
| DSD128 -> DSD256 | SDM-4 | 12.0 | 8 | 256 | 113.3 |
| DSD128 -> DSD512 | CLANS-8 | 12.0 | 8 | 512 | 120.3 |
| DSD256 -> DSD512 | CLANS-8 | 6.0 | 16 | 512 | 121.0 |
| DSD128 -> DSD64 | CLANS-4 | off | 32 | 128 | 83.3 |
| DSD256 -> DSD64 | CLANS-8 | off | 8 | 64 | 92.7 |
| DSD256 -> DSD128 | CLANS-4 | off | 8 | 256 | 112.0 |
| DSD512 -> DSD64 | SDM-6 | off | 8 | 256 | 85.5 |
| DSD512 -> DSD128 | SDM-4 | 16.0 | 16 | 128 | 99.4 |
| DSD512 -> DSD256 | SDM-6 | 16.0 | 8 | 256 | 116.1 |

**Key observations:**
- Downsample paths generally prefer no limiter and lower-order filters (CLANS-4/CLANS-8)
- Upsample paths benefit from moderate limiters (6-16) to prevent integrator overload from FIR image noise
- DSD64 -> DSD512 remains limited at ~54 dB due to 3-stage FIR image accumulation
- Higher-order SDM filters (SDM-8) are unstable without limiters (-37 to -49 dB)
- Some paths benefit from non-default candidates/latency (e.g., DSD256->DSD64 improves +10 dB with lat=64)

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

**Auto-selection**: CLANS-5 for DSD64, CLANS-6 for DSD128, CLANS-7 for DSD256, CLANS-8 for DSD512.

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

**Multi-stage chaining:** up to 3 stages (8x ratio) with ping-pong scratch buffers.

### DSD-Wide Volume Control (`engine.c`)

Digital volume control for DSD without leaving the DSD domain. Traditional approaches decimate DSD to multi-bit PCM, apply gain, then re-modulate back to DSD -- this introduces unnecessary latency and quality loss from the round-trip rate conversion. DSD-Wide stays at the original DSD sample rate throughout.

**The problem:** DSD is a 1-bit stream (values are strictly +1.0 or -1.0). Multiplying by a gain factor (e.g., 0.5) produces values like +0.5/-0.5, but the signal is still binary -- it carries the full ultrasonic quantisation noise of the original DSD encoding. Feeding this directly into an SDM causes it to saturate, producing constant -1.0 output (silence).

**The solution -- boxcar smoothing:**

```
DSD ±1.0 @ Fs  -->  boxcar(N)  -->  multi-bit @ Fs  -->  × gain  -->  SDM  -->  DSD ±1.0 @ Fs
```

1. **Boxcar filter** (N=8 taps): A running average over N consecutive DSD samples. Converts the 1-bit stream to ~4-bit resolution ({-1.0, -0.75, -0.5, ..., +0.75, +1.0}) at the **same** sample rate. This smooths the ultrasonic quantisation noise just enough for the SDM to track the signal.

2. **Gain multiply**: Applied to the multi-bit smoothed signal. The SDM now receives a properly-shaped waveform it can encode, not raw binary noise.

3. **SDM re-encode**: Trellis or PreCorr converts back to 1-bit DSD. The NTF reshapes quantisation noise optimally regardless of the boxcar's crude frequency response.

**Why a boxcar and not a proper FIR?** The boxcar is O(1) per sample (one add, one subtract, one multiply) regardless of tap count. Its frequency response has deep nulls and poor stopband -- but this doesn't matter. We're not producing a final PCM output; the SDM's noise-shaping will dominate the output spectrum. The boxcar only needs to provide enough multi-bit resolution for the SDM to converge, which 4 bits achieves comfortably.

**Performance:** The boxcar adds negligible CPU overhead. At DSD64 (2.8 MHz), the 8-tap boxcar is ~3 integer ops per sample vs ~50+ ops for the SDM. The dominant cost remains the Trellis/PreCorr SDM itself.

**Dynamic passthrough:** When gain = 1.0 (volume 100%), the engine bypasses both the boxcar and SDM entirely -- a simple sign-only requantise copies input to output with zero processing cost. Volume changes take effect immediately on the next audio block with no reinitialisation.

### Processing Engine (`engine.c`)

Per-channel orchestrator:
- Configures FIR chain based on input/output rate ratio
- **Path-adaptive SDM tuning**: When `NTF filter = Auto` and rate converting, selects optimal NTF filter, integrator limiter, candidates, and latency per conversion path from a lookup table derived from comprehensive sweep measurements
- **DSD-Wide volume control**: Boxcar smoothing + gain + SDM for same-rate DSD with gain != 1.0
- Applies gain multiply after FIR, before SDM (for rate conversion paths)
- Dispatches to Trellis or PreCorr based on `sdm_mode` config
- Detects passthrough (same rate, unity gain) to bypass FIR+SDM entirely
- Handles mute (silence pattern substitution)

### Configuration (`config.c`)

Runtime parameters serialized to foobar2000 config store:

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| SDM mode | enum | PreCorr / Trellis | PreCorr |
| Output DSD rate | enum | As input / DSD64-512 | As input |
| Volume | float | 0.0 - 1.0 | 1.0 |
| Mute | bool | on/off | off |
| NTF filter | enum | Auto / CLANS-4..8 / SDM-4..8 | Auto |
| Trellis depth (N) | int | 4, 8, 16, 32 | 8 |
| Trellis candidates (M) | int | 4 - 32 | 4 |
| Trellis latency | int | 16 - 2048 | 128 |
| Output format | enum | DoP / PCM | DoP |
| Thread count | int | 0 (auto) - cores | 0 |

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
|   |-- config.c              Configuration serialization
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
|   |-- httpapi.h             REST API
|   +-- wav_io.h              WAV I/O API
|-- test/
|   |-- test.h                Minimal test framework (no dependencies)
|   |-- test_main.c           Test runner with suite selection
|   |-- test_dop.c            DoP tests
|   |-- test_ntf.c            NTF tests
|   |-- test_fir.c            FIR tests
|   |-- test_trellis.c        Trellis SDM tests (SINAD measurement)
|   |-- test_precorr.c        PreCorr SDM tests (SINAD measurement)
|   |-- test_rate_sinad.c     Rate conversion SINAD tests
|   |-- test_config.c         Config serialization tests
|   |-- test_simd.c           CPU detection and IPP tests
|   |-- test_hardening.c      Edge cases and robustness
|   |-- test_threadpool.c     Thread pool tests
|   +-- test_sinad_diag.c     Extended SINAD diagnostics
|-- tools/
|   |-- gen_ntf_512.py        DSD512 NTF coefficient generator
|   +-- sinad_check.c         Standalone SINAD verification tool
|-- reference/
|   +-- sdm.c                 Source reference from mansr/sox (LGPL v2.1+)
|-- foobar2000-sdk/            Vendored fb2k SDK
+-- foo_dsd_trellis.sln        Visual Studio solution (7 projects)
```

## Test Results

694 tests across 12 suites, all passing:

| Suite | Tag | Tests | Coverage |
|-------|-----|-------|----------|
| DoP | `dop` | 24 | Detection, pack/unpack, round-trip, edge cases |
| NTF | `ntf` | 18 | All 40 filters, coefficient verification, auto-select |
| FIR | `fir` | 17 | Passband/stopband, round-trip, chain tests |
| Trellis SDM | `trellis` | 13 | Init, reset, latency, drain, SINAD (4 rates), DC stability |
| PreCorr SDM | `precorr` | 8 | Init, binary output, no latency, SINAD (4 rates) |
| Rate Conversion | `rate` | 12 | SINAD for all upsample/downsample pairs |
| Config | `config` | 8 | Serialization, versioning, validation |
| CPU & IPP | `simd` | 5 | CPU detection, IPP kernel, FIR correctness |
| Hardening | `hardening` | 22 | Edge cases, robustness |
| Thread Pool | `threadpool` | 8 | Create/destroy, concurrent SDM, stress |
| Rate Conv Sweep | `sweep` | 4 | FIR-only SINAD, limiter sweep, NTF×limiter sweep, cands×lat sweep (extended) |
| SINAD Diagnostics | `diag` | 7 | NTF sweeps, warmup analysis (extended) |

## References

- Reefman, D. & Janssen, E. (2002). "Signal processing for Direct Stream Digital." Philips Research.
- Harpe, P. et al. (2003). "Trellis-type sigma delta modulators for DSD." AES Convention.
- Schreier, R. & Temes, G. "Understanding Delta-Sigma Data Converters." Wiley.
- [mansr/sox](https://github.com/mansr/sox) -- Reference SDM implementation (LGPL v2.1+)

## License

NTF coefficient tables and SDM algorithm ported from mansr/sox under LGPL v2.1+. See `reference/sdm.c` for original source.
