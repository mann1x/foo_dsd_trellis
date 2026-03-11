# foo_dsd_trellis

A native foobar2000 DSP plugin that processes DSD (Direct Stream Digital) audio using a Trellis (Viterbi look-ahead) Sigma-Delta Modulator. All processing stays in the 1-bit DSD domain -- no decimation to audio-rate PCM ever occurs.

## Features

| Feature | Description |
|---------|-------------|
| Rate Conversion | DSD64 / DSD128 / DSD256 / DSD512 (power-of-2 ratios via polyphase half-band FIR) |
| Volume Control | Linear gain multiply before SDM requantiser |
| Passthrough | Repack only -- bypass FIR and SDM when input/output rate and gain are identical |
| Mute | Silence pattern substitution (0x69/0x96) |
| DoP Detection | Auto-detect DoP markers (0x05/0xFA) in 24-bit PCM frames; falls back to native ASIO |
| Native DSD | Raw DSD bitstream support (FORMAT_NATIVE) for ASIO and native input components |
| Trellis Depth | Configurable look-ahead N (4, 8, 16, 32) and candidates M (4-32) |
| SIMD Acceleration | Runtime CPU dispatch: AVX2+FMA (Intel/Zen 3+), AVX128+FMA (Zen 1/2), SSE2 fallback |
| Property Page | Full configuration dialog with dark mode support |
| Output Modes | DoP (native DSD output) or PCM (for VU meter / non-DSD DACs) |
| Config Versioning | Forward-compatible binary preset serialization with legacy fallback |
| foo_input_udsd | Compatible with foo_input_udsd / foo_input_sacd DSD input components |

## Architecture

Four-layer design with clean separation of concerns:

```
1-bit in -> unpack -> float32 @ Fs_in -> [polyphase FIR] -> x gain -> Trellis SDM -> 1-bit out @ Fs_out
```

| Layer | Purpose | Files |
|-------|---------|-------|
| fb2k Interface | foobar2000 DSP v2 glue, config dialog | `dsp_fb2k.cpp`, `dsp_plugin.c`, `config.c` |
| Format Bridge | DoP/native detection, 1-bit <-> float32 | `dop.c`, `bitpack.c` |
| Processing Engine | FIR rate conversion + gain + SIMD | `engine.c`, `fir.c`, `fir_simd.c` |
| Trellis SDM | Viterbi look-ahead requantiser | `trellis.c`, `ntf.c` |
| CPU Detection | Runtime SSE2/AVX2/FMA dispatch, AMD vs Intel tuning | `simd_detect.c` |

Thread pool (`threadpool.c`) provides per-channel parallelism.

## DSD Rates

| Rate | Sample Rate | Multiplier |
|------|-------------|------------|
| DSD64 | 2,822,400 Hz | 64 x 44.1 kHz |
| DSD128 | 5,644,800 Hz | 128 x 44.1 kHz |
| DSD256 | 11,289,600 Hz | 256 x 44.1 kHz |
| DSD512 | 22,579,200 Hz | 512 x 44.1 kHz |

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

**Key functions:**
- `sdm_filter_calc` -- CRFB state vector advance: `d[i] = s[i] + s[i-1] - g[i]*s[i+1]`
- `sdm_filter_calc2` -- Branch computation for y=+1 and y=-1 simultaneously
- `sdm_sort_cands` -- Insertion sort with path deduplication via PATH_HASH_SIZE=128 hash table
- `sdm_sample_trellis` -- Main per-sample algorithm with double-buffered trellis generations

**Structures:**
- `sdm_state_t` -- Per-candidate: NTF state vector, cost, path bits, history buffer index
- `sdm_trellis_t` -- Double-buffered generation: 2*MAX_NUM candidates, MAX_NUM active pointers
- `sdm_context_t` -- Per-channel: two trellis generations, path hash, history buffers, config

**Cost comparison** uses an IEEE 754 trick: cast double to int64 for integer comparison, avoiding floating-point comparison overhead while preserving ordering for non-negative values.

**Input scaling**: float +/-1.0 scaled by 0.5 before SDM (matching SoX convention). Output: +/-1.0 float.

**Latency**: configurable via `trellis_lat` (16-2048 DSD samples). First `trellis_lat` input samples fill the latency buffer with no output; `sdm_drain` flushes remaining samples at end of stream.

### FIR Rate Conversion (`fir.c`)

Polyphase half-band FIR for power-of-2 DSD rate conversion:

**Filter design:**
- 23-tap Kaiser-windowed sinc (beta=9.0, ~90 dB sidelobe suppression)
- Coefficients computed at init time via Bessel I0 series expansion (25 terms)
- Half-band property: every other tap is zero, center tap = 0.5

**Polyphase decomposition** (2x upsample):
- Phase 0 (even outputs): 12-tap FIR convolution with non-zero coefficients, gain x2
- Phase 1 (odd outputs): trivial delayed copy (0.5 x input, scaled by 2 = passthrough)

**Polyphase decomposition** (2x downsample):
- Separate even/odd delay lines
- Phase 0: FIR on even-indexed inputs
- Phase 1: 0.5 x delayed odd-indexed input

**Multi-stage chaining:** up to 3 stages (8x ratio) with ping-pong scratch buffers. Output buffer doubles as intermediate storage.

**Measured performance:**
- Passband: flat to +/-0.000 dB from 1 kHz to 20 kHz
- Stopband: -105 dB image attenuation
- Round-trip (up then down): 0.000 dB at 1 kHz

### Processing Engine (`engine.c`)

Per-channel orchestrator:
- Configures FIR chain based on input/output rate ratio
- Applies gain multiply after FIR, before SDM
- Detects passthrough (same rate, unity gain) to bypass FIR+SDM entirely
- Handles mute (silence pattern substitution)

### Configuration (`config.c`)

Runtime parameters serialized to foobar2000 config store:

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| Output DSD rate | enum | As input / DSD64-512 | As input |
| Volume | float | 0.0 - 1.0 | 1.0 |
| Mute | bool | on/off | off |
| NTF filter | enum | Auto / CLANS-4..8 / SDM-4..8 | Auto |
| Trellis depth (N) | int | 4, 8, 16, 32 | 8 |
| Trellis candidates (M) | int | 4 - 32 | 16 |
| Trellis latency | int | 16 - 2048 | 64 |
| Thread count | int | 0 (auto) - cores | 0 |

## SINAD Measurements

Measured via Goertzel algorithm directly on DSD output stream (bin-aligned 1 kHz sine, amplitude 0.5, trellis depth=8, candidates=16, latency=512):

| Rate | NTF Filter | SINAD (dB) | Convergence Failures |
|------|------------|------------|----------------------|
| DSD64 | CLANS-5 | 85.3 | 0 |
| DSD128 | CLANS-6 | 117.4 | 0 |
| DSD256 | CLANS-7 | 115.8 | 0 |
| DSD512 | CLANS-8 | 121.0 | 0 |

## Building

### Requirements

- Visual Studio 2022 with MSVC v142 toolset
- Windows 10 SDK
- foobar2000 SDK (place in `foobar2000-sdk/`)

### Build with MSBuild

```
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" ^
    foo_dsd_trellis.sln /p:Configuration=Release /p:Platform=x64
```

### Build tests with GCC (MSYS2)

```bash
gcc -std=c17 -Wall -Wextra -O2 -Iinclude -o test_runner.exe \
    test/test_main.c test/test_dop.c test/test_ntf.c test/test_fir.c \
    test/test_trellis.c test/test_threadpool.c \
    src/config.c src/dop.c src/bitpack.c src/engine.c src/fir.c \
    src/trellis.c src/ntf.c src/threadpool.c -lm
```

### Run tests

```bash
./test_runner.exe
# or
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
|   |-- fir.c                 Polyphase half-band FIR (rate conversion)
|   |-- trellis.c             Viterbi look-ahead trellis SDM
|   |-- ntf.c                 NTF coefficient tables (40 filters)
|   +-- threadpool.c          Worker thread pool
|-- include/
|   |-- dsd_types.h           Core types, constants, enums
|   |-- engine.h              Engine API
|   |-- trellis.h             SDM API and structures
|   |-- ntf.h                 NTF filter API
|   |-- fir.h                 FIR chain API
|   |-- dop.h                 DoP API
|   +-- threadpool.h          Thread pool API
|-- test/
|   |-- test.h                Minimal test framework (no dependencies)
|   |-- test_main.c           Test runner
|   |-- test_dop.c            DoP tests (24 functions)
|   |-- test_ntf.c            NTF tests (18 functions)
|   |-- test_fir.c            FIR tests (18 functions)
|   |-- test_trellis.c        Trellis SDM tests (13 functions, SINAD measurement)
|   +-- test_threadpool.c     Thread pool tests (3 functions)
|-- tools/
|   |-- gen_ntf_512.py        DSD512 NTF coefficient generator
|   +-- sinad_check.c         Standalone SINAD verification tool
|-- reference/
|   +-- sdm.c                 Source reference from mansr/sox (LGPL v2.1+)
|-- foobar2000-sdk/            Vendored fb2k SDK
|-- PLAN.md                    Full implementation plan and architecture
+-- foo_dsd_trellis.sln        Visual Studio solution (7 projects)
```

## Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Scaffold, MSVC project, plugin loads | Complete |
| 1 | Format bridge (DoP pack/unpack) | Complete |
| 2 | NTF coefficient tables (40 filters, 4 DSD rates) | Complete |
| 3 | Trellis SDM (full Viterbi algorithm) | Complete |
| 4 | FIR rate conversion (polyphase half-band) | Complete |
| 5 | Thread pool (lock-free MPMC queue) | Stub |
| 6 | fb2k integration (full DSP chain) | Stub |
| 7 | Hardening (edge cases, mid-stream config) | Pending |

## Test Results

472 assertions across 5 test suites, all passing:

| Suite | Tests | Assertions | Coverage |
|-------|-------|------------|----------|
| DoP | 24 | 134 | Detection, pack/unpack, round-trip, edge cases |
| NTF | 18 | ~80 | All 40 filters, coefficient verification, auto-select |
| FIR | 18 | ~30 | Passband/stopband response, round-trip, chain tests |
| Trellis | 13 | ~20 | Init, reset, latency, drain, SINAD (4 DSD rates), DC stability |
| Thread Pool | 3 | ~6 | Create/destroy, auto core count |

## References

- Reefman, D. & Janssen, E. (2002). "Signal processing for Direct Stream Digital." Philips Research.
- Harpe, P. et al. (2003). "Trellis-type sigma delta modulators for DSD." AES Convention.
- Schreier, R. & Temes, G. "Understanding Delta-Sigma Data Converters." Wiley.
- [mansr/sox](https://github.com/mansr/sox) -- Reference SDM implementation (LGPL v2.1+)

## License

NTF coefficient tables and SDM algorithm ported from mansr/sox under LGPL v2.1+. See `reference/sdm.c` for original source.
