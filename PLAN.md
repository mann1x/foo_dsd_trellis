# foo_dsd_trellis

**DSD Trellis SDM DSP Plugin for foobar2000**

Project Architecture & Implementation Plan | C / MSVC / Windows x64

---

## 1. Project Goals

Build a native foobar2000 DSP plugin (foo_dsp interface, Windows x64, MSVC) that processes DSD audio using a Trellis (look-ahead Viterbi) Sigma-Delta Modulator without ever decimating to audio-rate PCM. The plugin must support all four standard DSD rates, handle both DoP and native ASIO bitstream formats, and perform rate conversion, volume control, passthrough, and mute — all in the 1-bit DSD domain.

## 2. Scope & Features

| Feature | Description |
|---------|-------------|
| Rate Conversion | DSD64 ↔ DSD128 ↔ DSD256 ↔ DSD512 (power-of-2 ratios only via polyphase FIR at DSD rate) |
| Volume Control | Linear gain multiply [0.0–1.0] inserted before the SDM requantiser; no separate pipeline |
| Passthrough | Repack only — bypass polyphase FIR and SDM when input/output rate and gain are identical |
| Mute | Silence pattern substitution (0x69/0x96); no SDM invocation |
| DoP detection | Auto-detect DoP markers (0x05/0xFA in MSB of 24-bit PCM frames); fall back to native ASIO bitstream |
| Trellis depth | Configurable at runtime: look-ahead N ∈ {4,8,16,32}, candidates M ∈ {4–32} |
| Thread pool | Worker threads pick channel blocks from a lock-free queue; CPU affinity configurable |
| NTF filters | CLANS-5 through CLANS-8 and SDM-5 through SDM-8 coefficient tables (ported from sdm.c) |

## 3. Architecture Overview

The plugin is structured as four loosely coupled layers:

| Layer | Responsibility | Key Files |
|-------|---------------|-----------|
| fb2k interface | foobar2000 DSP v2 glue — chunk list I/O, config dialog, property page | `dsp_plugin.c`, `config.c` |
| Format bridge | DoP/native detection, pack/unpack 1-bit ↔ float32 at DSD rate | `dop.c`, `bitpack.c` |
| Processing engine | Polyphase FIR rate conversion + gain multiply — operates at full DSD rate | `engine.c`, `fir.c` |
| Trellis SDM | Look-ahead Viterbi requantiser — float32 at Fs_out → 1-bit output | `trellis.c`, `ntf.c` |

**Data flow** (simplified, one channel):

```
1-bit in → unpack → float32 @ Fs_in → [polyphase FIR @ Fs_in/out] → × gain → Trellis SDM → 1-bit out @ Fs_out
```

## 4. Module Design

### 4.1 Format Bridge (`dop.c`, `bitpack.c`)

Responsible for detecting the incoming stream format and unpacking 1-bit data to float32 before processing, and repacking after.

| Function | Signature | Notes |
|----------|-----------|-------|
| `dop_detect` | `bool dop_detect(const float* pcm24, size_t frames)` | Scan first 8 frames for alternating 0x05/0xFA in byte 2 of each 24-bit word |
| `dop_unpack` | `void dop_unpack(const float* pcm24, float* bits, size_t frames)` | Extract 16 DSD bits per PCM frame; output ±1.0f |
| `dop_pack` | `void dop_pack(const float* bits, float* pcm24, size_t frames)` | Repack 1-bit float stream into DoP PCM frames with markers |
| `bits_unpack` | `void bits_unpack(const uint8_t* src, float* dst, size_t n)` | Raw bitstream → ±1.0f float array (native ASIO path) |
| `bits_pack` | `void bits_pack(const float* src, uint8_t* dst, size_t n)` | float array → packed bitstream (native ASIO path) |

### 4.2 Polyphase FIR Rate Engine (`fir.c`, `engine.c`)

Operates entirely at DSD rate. No decimation to audio-rate PCM occurs at any point. For power-of-2 upsampling (×2), a half-band lowpass FIR is applied as an interpolation filter. For downsampling (÷2), the same filter is applied as an anti-alias filter before decimation. Chains of ×2/÷2 stages handle larger ratios (e.g. DSD64→DSD256 = two ×2 stages).

| Parameter | Value / Rationale |
|-----------|-------------------|
| Filter type | Linear-phase FIR half-band, symmetric coefficients |
| Cutoff | ~80 kHz (well below DSD noise floor onset at ~100 kHz; preserves full audio band) |
| Filter length | 63–127 taps (polyphase decomposed into 2 phases of 32–64 taps each) |
| Precision | double for coefficient storage, float32 for signal path (sufficient at DSD OSR) |
| State | Per-channel circular buffer; length = filter_length / 2 (one polyphase phase) |
| Gain insert | Multiply by gain scalar immediately after FIR output, before SDM input |

**Passthrough path:** when `Fs_in == Fs_out` AND `gain == 1.0f`, the FIR and SDM are both bypassed; the float32 buffer is requantised trivially (sign → bit) and repacked.

### 4.3 Trellis SDM (`trellis.c`, `ntf.c`)

Direct port of the Viterbi look-ahead algorithm from mansr/sox `sdm.c` (GPL), adapted to the project's channel-block processing model. The algorithm evaluates 2^N candidate bit sequences of depth N, propagates integrator state vectors forward for each candidate, accumulates a squared-error cost metric, prunes to M survivors, and emits the output bit by tracing back L samples from the surviving path.

#### State structure (per candidate path):

```c
typedef struct {
    double   state[MAX_NTF_ORDER]; /* NTF integrator state vector      */
    double   cost;                 /* accumulated squared-error metric  */
    uint32_t path;                 /* bit history (trellis_mask width)  */
    uint8_t  next;                 /* output bit at traceback position  */
    uint8_t  hist_idx;             /* circular history buffer index     */
} sdm_path_t;
```

#### Per-sample trellis step:

1. For each of M current candidate paths, branch into two children (output bit = 0 or 1).
2. Advance the NTF integrator state vector one sample for each child (`sdm_step`).
3. Compute cost increment: squared difference between input sample and predicted output.
4. Accumulate cost and store child states in the next-generation pool.
5. Sort all 2M children by cost; keep the M lowest (`sdm_sort_cands` — partial sort).
6. Emit output bit: trace back L steps from any surviving path (all converge at depth L).

#### NTF filter tables (`ntf.c`):

Coefficient tables ported verbatim from `sdm.c` (mansr/sox). Each entry defines the feedback coefficients `a[]` and the resonator gain `g[]` for a specific order and optimisation target:

| Filter name | Order | OSR target | Optimisation | Recommended for |
|-------------|-------|------------|--------------|-----------------|
| CLANS-5 | 5 | 64 (DSD64) | CLANS — closed-loop stability | DSD64 default (best stability) |
| CLANS-6 | 6 | 128 (DSD128) | CLANS | DSD128 |
| CLANS-7 | 7 | 256 (DSD256) | CLANS | DSD256 |
| CLANS-8 | 8 | 512 (DSD512) | CLANS | DSD512 |
| SDM-5 | 5 | 64 | Schreier Delta-Sigma Toolbox | DSD64 lower noise floor |
| SDM-6 | 6 | 128 | Schreier | DSD128 lower noise floor |
| SDM-7 | 7 | 256 | Schreier | DSD256 |
| SDM-8 | 8 | 512 | Schreier | DSD512 |

NTF order is selected automatically based on output DSD rate (CLANS-5 for DSD64, CLANS-6 for DSD128, etc.) unless the user overrides via the config dialog. SDM-x variants offer a ~4–6 dB lower in-band noise floor at the cost of slightly reduced stability margin at near-full-scale signals.

### 4.4 Thread Pool (`threadpool.c`)

One thread per logical CPU core (or a user-configured subset). Channel blocks are queued as work items; workers dequeue and process independently. Left and right channels are independent — no cross-channel state.

| Design decision | Choice & Rationale |
|-----------------|-------------------|
| Queue type | Lock-free MPMC ring buffer (single producer — the DSP callback — multiple consumers). Avoids mutex contention at DSD processing rates. |
| Work item | Pointer to `channel_block_t`: input float* buffer, output float* buffer, length, channel index, sdm context pointer. |
| CPU affinity | Optional: `SetThreadAffinityMask` per worker. Controlled by plugin config — user maps worker threads to specific cores away from fb2k GUI/audio thread. |
| Synchronisation | Completion barrier: DSP callback waits on a semaphore posted by the last worker to finish for a given chunk. Latency = trellis_latency + one chunk. |
| SDM context | One `sdm_context_t` per channel, permanently allocated, never shared between threads. Thread picks up the channel's context with the work item. |

## 5. Repository Structure

```
foo_dsd_trellis/
├── src/
│   ├── dsp_plugin.c       foobar2000 DSP v2 interface, chunk list I/O
│   ├── config.c           Property page, runtime parameter storage
│   ├── dop.c              DoP auto-detect, pack/unpack
│   ├── bitpack.c          Native ASIO bitstream pack/unpack
│   ├── engine.c           Top-level per-channel processing orchestrator
│   ├── fir.c              Polyphase FIR half-band filter (rate conversion)
│   ├── trellis.c          Viterbi look-ahead trellis SDM core
│   ├── ntf.c              NTF coefficient tables (CLANS-5..8, SDM-5..8)
│   └── threadpool.c       Lock-free MPMC queue + worker thread pool
├── include/
│   ├── engine.h
│   ├── trellis.h
│   ├── ntf.h
│   ├── fir.h
│   ├── dop.h
│   └── threadpool.h
├── test/
│   ├── test_dop.c         Unit tests: DoP detect/pack/unpack round-trip
│   ├── test_fir.c         FIR passband/stopband response check
│   ├── test_trellis.c     SDM SINAD measurement vs reference (SoX output)
│   └── test_threadpool.c  Concurrency smoke test
├── tools/
│   └── sinad_check.c      Standalone CLI: encode WAV→DSD, measure SINAD
├── foobar2000-sdk/        foobar2000 SDK (submodule or local copy)
├── foo_dsd_trellis.vcxproj
└── README.md
```

## 6. Key Data Types & Interfaces

```c
/* ---- Runtime configuration (populated from property page) ---- */
typedef struct {
    uint32_t  fs_in;          /* Input DSD rate: 2822400, 5644800, 11289600, 22579200 */
    uint32_t  fs_out;         /* Output DSD rate (same options)                        */
    float     gain;           /* Linear volume gain [0.0f – 1.0f]; 1.0f = unity        */
    bool      mute;           /* If true: substitute silence pattern, skip SDM         */
    int       trellis_depth;  /* Look-ahead N: 4, 8, 16, or 32                         */
    int       trellis_cands;  /* Max survivors M: 4 – 32                               */
    int       trellis_lat;    /* Traceback latency L: 16 – 2048 samples                */
    int       ntf_filter;     /* NTF_CLANS_5..8 or NTF_SDM_5..8 (enum)                */
    int       thread_count;   /* Worker threads: 1 – logical_cpu_count                 */
    DWORD     affinity_mask;  /* SetThreadAffinityMask value, 0 = OS default           */
    int       format;         /* FORMAT_DOP or FORMAT_NATIVE (auto-detected)           */
} dsd_config_t;

/* ---- Per-channel SDM context (one per channel, lives for plugin lifetime) ---- */
typedef struct {
    sdm_path_t  *paths;       /* Allocated pool: 2 * trellis_cands path states         */
    sdm_path_t **act;         /* Active candidate pointer array [trellis_cands]        */
    double      *hist;        /* Circular input history buffer [trellis_lat]           */
    int          pos;         /* Current position in history buffer                    */
    int          num_cands;   /* Current live candidate count                          */
    const ntf_filter_t *ntf;  /* Pointer into ntf.c coefficient table                 */
} sdm_context_t;

/* ---- Work item dispatched to thread pool ---- */
typedef struct {
    float         *in;        /* Float32 DSD samples at Fs_in  (unpacked 1-bit)        */
    float         *out;       /* Float32 DSD samples at Fs_out (before repack)         */
    size_t         count;     /* Sample count (at Fs_in; output count may differ)      */
    int            channel;   /* Channel index (0=L, 1=R, …)                          */
    sdm_context_t *ctx;       /* SDM context for this channel                          */
    const dsd_config_t *cfg;  /* Immutable config snapshot for this chunk              */
} channel_block_t;
```

## 7. Implementation Phases

| Phase | Deliverable | Key acceptance criterion |
|-------|-------------|------------------------|
| **0 — Scaffold** | MSVC project, fb2k SDK linked, plugin loads and appears in DSP chain | Plugin visible in foobar2000 → Preferences → DSP Manager |
| **1 — Format bridge** | `dop.c` + `bitpack.c` complete with unit tests | Round-trip test: DoP unpack → repack produces bit-identical output for silence and sine-tone DSF |
| **2 — NTF tables** | `ntf.c` with all 8 filter coefficient tables from `sdm.c` | Coefficients match `sdm.c` source exactly; unit test compares tables byte-for-byte |
| **3 — Trellis SDM** | `trellis.c`: `sdm_context_init`, `sdm_process_block`, `sdm_context_free` | SINAD test: 1 kHz sine encoded to DSD64 matches or exceeds SoX `-f clans-5 -t 8 -n 16` within ±1 dB |
| **4 — FIR engine** | `fir.c`: polyphase half-band FIR, `engine.c`: gain insert, rate mux | DSD64→DSD128 conversion: no audible artefacts; passband flat ±0.1 dB to 20 kHz |
| **5 — Thread pool** | `threadpool.c`: lock-free queue, worker threads, affinity config | Stereo DSD64 passthrough processes in real time with CPU usage ≤ one core at trellis depth 8/16 |
| **6 — fb2k integration** | Full DSP chain: DoP/native auto-detect, chunklist processing, property page UI | Plays SACD ISO via foo_input_sacd with rate conversion and volume control audibly correct |
| **7 — Hardening** | Edge cases: mute, passthrough, config changes mid-stream, stream discontinuities | No glitches on pause/resume/seek; mute produces clean silence pattern |

## 8. Runtime Configuration Parameters

All parameters are exposed via the foobar2000 property page (`config.c`) and persist in the fb2k configuration store.

| Parameter | Type | Range / Options | Default | Notes |
|-----------|------|-----------------|---------|-------|
| Output DSD rate | enum | As input / DSD64 / DSD128 / DSD256 / DSD512 | As input | Drives polyphase FIR chain selection |
| Volume | float | 0.0 – 1.0 (or dB display) | 1.0 | Applied as gain multiply before SDM; 1.0 = no SDM invocation on passthrough |
| Mute | bool | On / Off | Off | Substitutes 0x69/0x96 silence pattern; SDM and FIR bypassed |
| NTF filter | enum | CLANS-5..8, SDM-5..8, Auto | Auto | Auto selects CLANS-N where N matches output OSR |
| Trellis depth (N) | int | 4, 8, 16, 32 | 8 | 2^N paths tracked; doubles CPU cost per step |
| Trellis candidates (M) | int | 4 – 32 | 16 | Survivors after pruning; quality/CPU tradeoff |
| Traceback latency (L) | int | 16 – 2048 | 64 | Output latency in DSD samples; larger = better but more delay |
| Worker threads | int | 1 – logical cores | logical cores / 2 | Set lower to leave cores for fb2k UI/audio output |
| CPU affinity mask | hex DWORD | 0x00000000 – 0xFFFFFFFF | 0 (OS default) | 0 = let OS schedule; non-zero pins workers to specific cores |

## 9. Testing Strategy

### 9.1 Unit Tests (`test/` directory, standalone executables)

- **test_dop.c** — DoP marker detection on valid, invalid, and borderline streams; pack/unpack round-trip identity
- **test_fir.c** — Polyphase FIR frequency response: passband ripple < 0.1 dB to 80 kHz, stopband attenuation > 80 dB above 100 kHz
- **test_trellis.c** — SINAD measurement of 1 kHz, 10 kHz, and 19 kHz sine tones encoded to DSD64/DSD128/DSD256 at each NTF filter setting; compare against SoX reference output
- **test_threadpool.c** — Concurrency: 1000 blocks dispatched across 8 threads with randomised delays; verify ordering and no data races (build with `/fsanitize=address` on MSVC 17+)

### 9.2 Integration / Reference Test (`tools/sinad_check.c`)

Standalone CLI tool: reads a 24-bit/192 kHz WAV sine tone, upsamples to DSD rate via the engine, measures in-band SINAD, and prints a summary. Used to verify output quality against SoX-DSD benchmarks before any fb2k integration.

### 9.3 Listening Test Procedure

1. Select a known SACD ISO with wide dynamic range material (classical, full orchestra).
2. Play via foo_input_sacd with plugin in DSP chain at each rate conversion path (DSD64→DSD128, DSD128→DSD64, passthrough).
3. Verify no artefacts on loud transients (instability), no clicks on pause/resume (state handling), no tonal distortion on sustained tones (idle tones / limit cycles).
4. Compare volume control at -6 dB and -20 dB against a linear PCM reference for the same material.

## 10. Performance Targets & Notes

At DSD64 (2.8224 MHz), stereo processing requires evaluating the trellis inner loop at approximately 5.6 million samples per second. The dominant cost is `sdm_step()` — the NTF state vector advance — called M×2 times per output sample (M candidates, 2 branches each).

| Configuration | Inner loop calls/sec (stereo DSD64) | Estimated CPU (single core, modern x64) |
|---------------|--------------------------------------|----------------------------------------|
| N=4, M=8 | 5.6M × 8 × 2 = ~90M/s | ~15–20% one core |
| N=8, M=16 (default) | 5.6M × 16 × 2 = ~180M/s | ~30–40% one core |
| N=16, M=32 | 5.6M × 32 × 2 = ~360M/s | ~60–80% one core — may need 2 cores |
| N=32, M=32 | 5.6M × 32 × 2 = ~360M/s + sort overhead | Not real-time at DSD64 stereo on single core |

- The NTF state update (`sdm_step`) is 5th–8th order IIR — approximately 10–16 multiply-accumulate operations per call. Auto-vectorisation with `/O2 /arch:AVX2` should give a useful speedup.
- The candidate sort (`sdm_sort_cands`) is a partial sort of 2M doubles. For M ≤ 16 an insertion sort is faster than `std::partial_sort` due to cache locality.
- At DSD512, the sample rate is ×8 relative to DSD64, but the required NTF order drops to 5 (CLANS-8 is 8th order for quality; can use 5th for speed). CPU cost scales linearly with sample rate.
- Thread pool assignment: for stereo DSD64 at N=8/M=16, one worker thread per channel is sufficient. For DSD512 or N=32, assign 2 workers per channel and split the block in half.

## 11. Dependencies & References

### Code dependencies

| Dependency | Version / Source | Usage |
|------------|-----------------|-------|
| foobar2000 SDK | Latest from hydrogenaud.io / fb2k developer portal | DSP v2 plugin interface, configuration store, chunk list API |
| mansr/sox `sdm.c` | GPL-2.0 — github.com/mansr/sox | Reference for trellis algorithm and NTF coefficient tables (porting, not linking) |
| Pieter Harpe thesis | pieterharpe.nl/docs/report_trunc.pdf | Theoretical basis for efficient trellis SDM |
| MSVC 2022 | v17+ | C17, `/O2`, `/arch:AVX2`, optional `/fsanitize=address` for test builds |

### Key academic references

- Reefman & Janssen (Philips, 2002): "Signal processing for Direct Stream Digital" — canonical DSD SDM reference
- Janssen & Reefman (AES 114th, 2003): "Advances in Trellis Based SDM Structures" — pruned trellis algorithm
- Angus (AES 115th, 2003): "Efficient Algorithms for Look-Ahead Sigma-Delta Modulators" — tree-based alternatives
- Harpe, Reefman, Janssen (AES 114th, 2003): "Efficient Trellis-type Sigma-Delta Modulator" — basis of `sdm.c` implementation
- Reiss & Sandler (DAFx 2004): "Digital Audio Effects Applied Directly on a DSD Bitstream" — volume/filter theory
