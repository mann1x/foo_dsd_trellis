# foo_dsd_trellis

A native foobar2000 DSP plugin that converts PCM and DSD audio to DSD using sigma-delta modulation. Supports two SDM modes: **Trellis** (Viterbi look-ahead, highest quality) and **PreCorr** (greedy + prediction correction, near-zero CPU). FIR rate conversion uses Intel IPP with automatic SSE2/AVX2/AVX-512 dispatch.

## Features

| Feature | Description |
|---------|-------------|
| SDM Modes | PreCorr (default, ~0.01x RT) and Trellis (high quality, configurable depth/candidates) |
| Rate Conversion | DSD64 / DSD128 / DSD256 / DSD512 via Intel IPP FIRSR (63-tap Kaiser half-band) |
| PCM to DSD | Float32 PCM input upsampled via FIR then quantised to 1-bit DSD |
| Volume Control | Linear gain multiply before SDM requantiser |
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

## SINAD Measurements

All measurements use bin-aligned 1 kHz sine at amplitude 0.5, measured via Goertzel algorithm on the DSD output stream (0-22.05 kHz noise floor). Auto-selected NTF filters per rate.

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

### DSD Rate Conversion (IPP FIRSR, 63-tap Kaiser half-band)

End-to-end SINAD through FIR + Trellis SDM pipeline. Measures combined degradation from rate conversion and requantisation.

**Upsample:**

| Conversion | SINAD (dB) |
|------------|------------|
| DSD64 -> DSD128 | 51.2 |
| DSD64 -> DSD256 | 90.2 |
| DSD64 -> DSD512 | 57.3 |
| DSD128 -> DSD256 | 107.5 |
| DSD128 -> DSD512 | 63.2 |
| DSD256 -> DSD512 | 106.1 |

**Downsample:**

| Conversion | SINAD (dB) |
|------------|------------|
| DSD128 -> DSD64 | 75.4 |
| DSD256 -> DSD64 | 79.4 |
| DSD512 -> DSD64 | 91.2 |
| DSD256 -> DSD128 | 106.9 |
| DSD512 -> DSD128 | 118.3 |
| DSD512 -> DSD256 | 103.0 |

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

### Processing Engine (`engine.c`)

Per-channel orchestrator:
- Configures FIR chain based on input/output rate ratio
- Applies gain multiply after FIR, before SDM
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
| Trellis candidates (M) | int | 4 - 32 | 8 |
| Trellis latency | int | 16 - 2048 | 64 |
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

635 tests across 11 suites, all passing:

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
| Thread Pool | `threadpool` | 3 | Concurrent SDM processing |
| SINAD Diagnostics | `diag` | 7 | NTF sweeps, warmup analysis (extended) |

## References

- Reefman, D. & Janssen, E. (2002). "Signal processing for Direct Stream Digital." Philips Research.
- Harpe, P. et al. (2003). "Trellis-type sigma delta modulators for DSD." AES Convention.
- Schreier, R. & Temes, G. "Understanding Delta-Sigma Data Converters." Wiley.
- [mansr/sox](https://github.com/mansr/sox) -- Reference SDM implementation (LGPL v2.1+)

## License

NTF coefficient tables and SDM algorithm ported from mansr/sox under LGPL v2.1+. See `reference/sdm.c` for original source.
