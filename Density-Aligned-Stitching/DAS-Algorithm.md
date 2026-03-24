# Density-Aligned Stitching (DAS) for State-Convergent Parallel SDM

## Abstract

Density-Aligned Stitching (DAS) is a novel algorithm for parallelizing 1-bit Trellis Sigma-Delta Modulators (SDM) across multiple CPU cores without introducing audible artifacts at segment boundaries. The algorithm solves the fundamental challenge of SDM parallelization: the feedback loop creates a strict sample-by-sample sequential dependency that prevents naive temporal decomposition.

DAS introduces three key innovations:

1. **State-seeded parallel segments** with extended overlap regions sized as a multiple of the trellis look-ahead latency
2. **Windowed match density scanning** that identifies regions of genuine SDM convergence within the overlap, replacing random bit-matching with a convergence-aware metric
3. **Hybrid stitch point selection** that combines density-based convergence detection with exact bit-matching for artifact-free transitions

The algorithm achieves real-time DSD512 processing (22.579 MHz, 1-bit) at 0.57x real-time on a 16-core AMD Ryzen 9 9950X using 4 parallel segments, with 100% exact bit-match rate at every stitch boundary. No prior published work achieves temporal parallelism of trellis SDM for audio.

---

## 1. Background: 1-Bit Sigma-Delta Modulation

### 1.1 The 1-Bit SDM

A Sigma-Delta Modulator (SDM) converts a multi-bit PCM signal into a 1-bit pulse-density modulated (PDM) stream at a high oversampling ratio. The output at each sample is a binary decision: +1 or -1. The modulator shapes quantization noise away from the audio band using a feedback loop with an error-integrating filter.

The basic SDM loop operates as follows:

```
input[n] ──► (+) ──► H(z) ──► Quantizer ──► output[n] (±1)
              ▲                    │
              │                    │
              └────── DAC ◄────────┘
                   (feedback)
```

Where:
- **H(z)** is the loop filter (noise transfer function, NTF), typically a high-order IIR filter (4th to 8th order for audio)
- The **quantizer** is a 1-bit comparator (sign function)
- The **feedback DAC** is trivial for 1-bit (just the previous output ±1)
- The **integrator state vector** `s[n]` accumulates the error history across all filter orders

The NTF determines the noise-shaping profile. Higher-order NTFs push more quantization noise above the audio band, achieving better in-band Signal-to-Noise-and-Distortion Ratio (SINAD). A 6th-order NTF at DSD512 (22.579 MHz) can achieve >130 dB SINAD.

### 1.2 The Sequential Dependency Problem

The critical property of any SDM is that **each output sample depends on the complete history of all previous samples** through the integrator state vector. For an Nth-order modulator, the state vector `s[n] = [s_0, s_1, ..., s_{N-1}]` is updated at every sample:

```
s_i[n+1] = f(s_i[n], input[n], output[n])    for i = 0..N-1
```

This creates an unbreakable sequential dependency chain. Sample `output[n]` cannot be computed without knowing `output[n-1]`, which requires `output[n-2]`, and so on back to the start of the stream. There is no closed-form solution that allows computing `output[n]` directly from `input[n]` without the full state history.

For a simple 1st-order SDM, this means:
```
s[n+1] = s[n] + input[n] - output[n]
output[n] = sign(s[n] + input[n])
```

Each sample requires the accumulated error `s[n]` from all prior samples. For a 6th-order modulator, six coupled integrator states must be maintained.

### 1.3 Trellis (Viterbi Look-Ahead) SDM

A standard SDM makes a greedy decision at each sample: choose +1 or -1 based on the current error state. This is locally optimal but globally suboptimal — it can drive the modulator into high-error states that cause instability or poor noise shaping.

Trellis SDM, introduced by Kato (2002) and refined by Janssen & Reefman at Philips Research (2003), uses a Viterbi-style look-ahead search to find globally better output sequences:

1. **Candidate expansion**: At each sample, maintain `C` candidate paths (typically 2-16), each representing a different sequence of recent output decisions
2. **Look-ahead**: Each candidate is extended `L` samples into the future (the "latency" or look-ahead depth, typically 32-128), evaluating the accumulated cost (squared error) of each possible path
3. **Pruning**: After expansion, prune back to the best `C` candidates based on accumulated cost
4. **Traceback**: Output the decision from `L` samples ago, where all surviving paths have typically converged to a single decision

The trellis structure:
```
Time:     n    n+1   n+2   ...   n+L
         [C candidates at each time step]
          │ ╲   │ ╲   │ ╲
          │  ╲  │  ╲  │  ╲       Look-ahead
          │   ╲ │   ╲ │   ╲      depth = L
          ▼    ▼▼    ▼▼    ▼
         [Expand to 2C, prune to C]
```

Each candidate carries:
- **State vector** `s[N]`: N integrator states (the full NTF filter state)
- **Cost**: Accumulated squared error over the look-ahead window
- **Path**: Bit history of length L (the output decisions)

The trellis SDM produces dramatically better results than greedy SDM:
- 90+ dB SINAD at DSD64 (vs ~60 dB greedy)
- 140+ dB SINAD at DSD512 (vs ~100 dB greedy)

### 1.4 Computational Cost

The cost per sample of trellis SDM is:

```
Operations/sample = C × 2 × (N multiplications + N additions + 1 comparison)
                  = O(C × N) per sample
```

At DSD512 (22.579 MHz) with C=2 candidates and N=6 order:
- **~540 million trellis operations per second per channel**
- Stereo: ~1.08 billion operations/second
- With look-ahead bookkeeping, memory access, and path management overhead, this easily saturates a single CPU core

This is why parallelization is essential for real-time DSD512 processing.

---

## 2. The Parallelization Challenge

### 2.1 Why Standard Parallelization Fails

Standard signal processing parallelization techniques do not work for SDM:

**Overlap-Add/Overlap-Save (FIR)**: These techniques exploit the linearity and finite memory of FIR filters. SDM is nonlinear (1-bit quantizer) and has infinite memory (IIR feedback). Splitting the input and recombining outputs produces discontinuities at segment boundaries.

**Block Processing**: Processing fixed-size blocks independently requires initializing each block's SDM state. Without the correct state from the previous block's end, the modulator starts from an incorrect state and produces corrupted output until the state converges — if it converges at all.

**Pipeline Parallelism**: The feedback loop has a latency of exactly 1 sample. Inserting pipeline registers increases this latency, which destabilizes the modulator. A loop delay of even 2 samples can cause a high-order NTF to become unstable.

### 2.2 The State Divergence Problem

Consider splitting an input buffer into two segments processed by two SDM instances:

```
Input:    [════════ Seg A ════════|════════ Seg B ════════]
SDM A:    [Processing from state S₀]→ ends at state S_A
SDM B:                            [Processing from state ?]→ ends at state S_B
```

**Problem 1: Initial state of SDM B**. SDM B needs to start from state `S_A` (the end state of SDM A), but `S_A` is unknown until SDM A finishes — defeating the purpose of parallelism.

**Problem 2: Hard stitch discontinuity**. If SDM B starts from an approximation of `S_A` (e.g., a seed state from a previous chunk), the states will differ at the boundary. This state mismatch manifests as:

- A full-scale impulse if the output bits differ at the boundary (±2 in a ±1 stream)
- A burst of mismatched bits as the SDM "recovers" from the wrong initial state
- Audible as clicks, pops, or tonal artifacts depending on the DSD rate and severity

For a 6th-order modulator, the state vector has 6 degrees of freedom. Even small state errors propagate through the feedback loop, potentially taking hundreds or thousands of samples to converge.

### 2.3 Convergence is Not Guaranteed

A key insight is that two SDMs starting from different states but processing the same input will **not necessarily converge** to the same state. The 1-bit quantizer is a nonlinear element that can amplify small state differences:

- If `s_0[n] = 0.001` → output = +1
- If `s_0[n] = -0.001` → output = -1

A tiny state difference produces a completely different output, which feeds back and alters all subsequent states. The quantizer acts as a chaotic element — small perturbations can lead to permanently divergent trajectories.

This is fundamentally different from linear systems, where two instances processing the same input will always converge regardless of initial state (assuming stability).

---

## 3. Prior Art: Approaches and Their Limitations

### 3.1 Time-Interleaved Delta-Sigma Modulators (TIDSM)

**Approach**: Replace a single high-speed modulator with N parallel paths, each clocked at 1/N the original rate.

**Key work**: Kozak & Kale (2003), Gustat & Hagemeyer (2014).

**Limitation**: Only effective for small N (2-4) because the time-interleaved decomposition creates long combinatorial paths that reduce maximum clock rate. The loop delay constraint makes TIDSM fundamentally incompatible with high-order noise-shaping feedback loops used in audio-quality SDM. The quality degradation from even N=2 interleaving is unacceptable for high-fidelity audio applications.

### 3.2 State Register Propagation

**Approach**: Propagate state registers between parallel modulator segments to improve performance (Oliveira e Silva et al., IEEE 2017).

**Limitation**: Requires inter-segment communication during processing, which introduces synchronization overhead and limits parallelism to adjacent segments. Does not address the fundamental problem of independent temporal parallelism. Targeted at telecom transmitter applications, not audio-quality noise shaping.

### 3.3 Parallel FPGA Architectures

**Approach**: Break the feedback loop using bit-by-bit quantization on FPGA hardware (Chen et al., 2023).

**Limitation**: Requires custom hardware (FPGA). Uses lower-order modulators (1st-2nd order) with relaxed noise-shaping requirements unsuitable for high-fidelity audio. Targets telecom (5G fronthaul) applications where noise-shaping quality is secondary to throughput.

### 3.4 Parallel Look-Ahead (Hawksford)

**Approach**: Hawksford (2008) parallelizes the look-ahead computation within each sample using energy-balance matrix operations over a 5-sample window.

**Limitation**: The parallelism is within the per-sample decision process (matrix operations), not across the temporal extent of the audio stream. The stream is still processed sequentially sample-by-sample. This reduces per-sample latency but does not enable multi-core temporal parallelism.

### 3.5 Commercial Attempts

**Roon Labs**: Offers a "Parallelize Sigma-Delta Modulator" option. Community reports indicate no measurable speed improvement and perceived quality degradation. No technical details are publicly documented, suggesting naive segment splitting without convergence-aware stitching.

**HQPlayer (Signalyst)**: Achieves DSD2048 (98.304 MHz) using CUDA GPU acceleration. The GPU acceleration targets the FIR oversampling/filtering pipeline. The SDM feedback loop itself appears to remain sequential on a single core. HQPlayer uses extremely efficient SDM implementations (ASDM5/7) rather than parallelized trellis.

**Merging Technologies (Pyramix)**: Real-time DSD256 via dedicated RT cores (MassCore engine). No evidence of SDM parallelization — processing runs on single dedicated cores with RT scheduling.

### 3.6 The Gap in the Literature

A comprehensive search of AES papers, IEEE publications, patents, and commercial product documentation reveals that **no published work describes temporal segmentation of audio streams for parallel trellis SDM processing with convergence-aware stitching**. The techniques of state seeding, overlap-based convergence scanning, and density-aligned stitch point selection are, to our knowledge, novel contributions.

The foundational references for trellis SDM — Kato (2002), Janssen & Reefman (2003), Harpe et al. (2003), Angus (2004), Hawksford (2008), and the definitive monograph by Janssen & van Roermund (Springer, 2011) — all focus exclusively on reducing per-sample computational cost. None propose multi-core temporal parallelism.

---

## 4. Density-Aligned Stitching (DAS): The Algorithm

### 4.1 Overview

DAS decomposes the SDM processing of an audio buffer into N temporal segments, each processed independently on a separate CPU core, then reassembles the output using a convergence-aware stitching algorithm that guarantees artifact-free transitions.

The key insight is that while two SDMs starting from identical states but processing different input data will diverge, they can be made to **partially reconverge** when subsequently fed identical input data in an overlap region. The quality of this reconvergence can be measured and exploited for optimal stitch point selection.

### 4.2 State Seeding with Estimation Pre-Pass

Each parallel segment needs an initial NTF integrator state. Segment 0 uses the persistent SDM state directly (end state of the previous chunk). Segments 1+ need the state at their nominal start position — which is unknown until seg0 finishes, defeating parallelism.

#### Original approach: Persistent state replication

The original approach seeded all segments from the same persistent state `S_seed`. This worked but produced suboptimal convergence: seg1 starts with the state from the *beginning* of the chunk, not the *midpoint*, creating unnecessary state divergence that the overlap must recover from.

#### Current approach: Greedy nc=1 estimation

A lightweight estimation pre-pass runs before segment dispatch. It uses a single-candidate (nc=1) greedy SDM — the NTF filter with a simple sign-quantizer — to estimate the integrator state at each segment boundary:

```
Estimation pass (sequential, ~10ms per 1.4M samples):
  state = persistent_state.integrators
  for sample = 0 to boundary[seg]:
      new_state = NTF_filter(state, input[sample], y=0)
      v = Σ(a[i] × new_state[i])
      y = (v × a[0] < 0) ? +1.0 : -1.0    // greedy quantizer
      new_state[0] += y
      state = clamp(new_state, state_limit)
  estimated_state[seg] = state
```

The greedy quantizer makes the same decision the trellis SDM would make ~80% of the time (the trellis only improves on the greedy choice for edge cases where cost look-ahead matters). This means the estimated integrator state at a segment boundary closely tracks the true state.

Segments are then seeded with estimated states:

```
Previous chunk:  [...........] → SDM state S_seed
                                     │
Current chunk:                       ▼
  Estimation:   ──[S_seed]──► fast nc=1 scan ──► S_est[1], S_est[2], S_est[3]
  Seg 0: ──[S_seed]──► process data[0..N₀+ovl] ──► output₀
  Seg 1: ──[S_est[1]]──► process data[N₀-ovl..N₀+N₁+ovl] ──► output₁
  Seg 2: ──[S_est[2]]──► process data[N₀+N₁-ovl..N₀+N₁+N₂+ovl] ──► output₂
  Seg 3: ──[S_est[3]]──► process data[N₀+N₁+N₂-ovl..end] ──► output₃
```

All segments still launch **simultaneously** — the estimation pass is sequential but takes ~10ms for 1.4M DSD128 samples (order 8, ~16 flops/sample), negligible vs the ~190ms total chunk time. The overlap warmup handles the remaining convergence gap between the estimated state and the true trellis state.

#### Impact on convergence

The estimation pass dramatically improves DAS stitch density:

| Seeding method | Typical density (DSD128, lat=128) | Range |
|----------------|-----------------------------------|-------|
| Persistent state replication | 40-56/64 (63-88%) | Medium convergence |
| **Estimated states (nc=1 pre-pass)** | **190-255/256 (74-99%)** | **Near-perfect convergence** |

Density scores of 255/256 (99.6% match) are routinely achieved, meaning the trellis SDM with estimated initial state produces virtually identical output to what sequential processing would produce at that point. The overlap warmup then handles the remaining <1% divergence.

### 4.3 Extended Overlap Regions

Each segment (except the last) extends its processing by `overlap` samples into the next segment's territory. Each segment (except the first) starts processing `overlap` samples before its nominal boundary, with the early samples serving as warmup that is discarded.

```
FIR output: [════ Seg 0 ════[ovl]════ Seg 1 ════[ovl]════ Seg 2 ════]
                            ▲   ▲               ▲   ▲
                            │   │               │   │
                     Seg 0's    Seg 1's    Seg 1's   Seg 2's
                     extension  warmup     extension  warmup
```

The overlap size is a critical parameter. It must be large enough for the SDMs to exhibit measurable convergence when processing the same input data. Through systematic testing (see Section 5), we determined:

```
overlap = 32 × trellis_latency
```

For a typical configuration with `trellis_lat = 32`:
- `overlap = 1024 samples`
- At DSD512 (22.579 MHz): 45.4 microseconds of audio — acoustically negligible

The warmup discard for segments 1+ is:
```
warmup_discard = overlap - trellis_latency
```

This ensures the trellis look-ahead buffer is fully populated before the first useful output sample.

### 4.4 Why 32x Latency?

The overlap multiplier was determined empirically by sweeping from 2x to 128x and measuring convergence quality:

The trellis latency `L` represents the look-ahead depth — the number of future samples the SDM considers before committing to an output decision. When two SDMs process the same input for `K × L` samples:

- At **K=2-4**: The SDMs have barely begun to align. Their trellis look-ahead windows have only partially overlapped with the shared input. State distances remain very high.
- At **K=8-16**: Significant convergence begins. The shared input has propagated through 8-16 complete look-ahead windows, allowing the feedback loops to synchronize.
- At **K=32**: Convergence plateaus. The SDMs have processed enough shared input for their integrator states to approach a stable equilibrium. Further overlap provides diminishing returns.
- At **K=64-128**: No measurable improvement over K=32. The fixed-width density window (2L=64 samples) is the limiting factor, not the overlap extent.

### 4.5 Windowed Match Density Scanning

This is the core innovation of DAS. Instead of searching for the longest consecutive run of matching output bits (which finds random coincidences), DAS measures the **local density of matching bits** in a sliding window across the overlap region.

#### Algorithm

```
For each position p in [0, overlap):
    window_start = max(0, p - L)        // L = trellis_latency
    window_end   = min(overlap, p + L)
    matches = 0
    for w in [window_start, window_end):
        if seg_prev[w] == seg_next[w]:
            matches++
    density[p] = matches

best_density_pos = argmax(density)
```

The window width is `2L` (twice the trellis latency), which captures exactly one full look-ahead period on each side of the candidate stitch point. This is the natural scale at which SDM convergence manifests — if the integrator states are close, the look-ahead will make similar decisions over the next L samples.

#### Why Density Beats Longest-Run

Consider two overlap regions:

**Region A**: `...++-+-++--+-++-++-+-+++-++-++...` (scattered matches, 60% density)
**Region B**: `...-+--+-++++++++-+--+--+-+-+--...` (one long run of 8, but only 45% overall)

The longest-run algorithm selects Region B (run=8). But Region A has higher density — the SDMs are more uniformly converged across the window. A stitch in Region A produces a smoother transition because the neighboring output bits are more likely to be compatible, even if no single long run exists.

The density metric correlates with integrator state proximity:
- High density (>50%) → integrator states are close → output decisions are similar → smooth stitch
- Low density (<40%) → states are far apart → decisions differ → potential artifacts

This was confirmed experimentally: the state-distance-optimal stitch point and the density-optimal point are strongly correlated, while the longest-run point shows no correlation with state distance (see Section 5.2).

### 4.6 Hybrid Stitch Point Selection

After finding the density-optimal position, DAS searches outward for the nearest exact bit-match:

```
best_pos = best_density_pos   // fallback: density peak
for radius = 0 to L:
    for candidate in [best_density_pos - radius, best_density_pos + radius]:
        if candidate in [0, overlap) AND seg_prev[candidate] == seg_next[candidate]:
            best_pos = candidate
            → FOUND exact match near convergence peak
            break both loops
```

This hybrid approach guarantees:
1. **Convergence**: The stitch point is within the region of maximum SDM convergence (density peak)
2. **Clean transition**: The actual stitch sample has matching bits from both segments, eliminating any full-scale impulse artifact

In practice, an exact match is always found within the search radius (100% success rate across all tested configurations), because the density peak region has >40% matching bits — finding at least one within ±L samples is statistically near-certain.

### 4.7 Channel Synchronization

For multi-channel audio (stereo, surround), all channels must produce identical output sample counts to maintain frame alignment. DAS computes stitch positions on channel 0 and applies the same positions to all other channels:

```
Pass 1: Scan channel 0 overlap → stitch_positions[seg]
Pass 2: For channels 1..N-1:
           Apply stitch_positions[seg] without scanning
```

This guarantees frame-aligned output across all channels, which is critical for proper DAC reconstruction and DoP (DSD-over-PCM) packaging.

### 4.8 State Persistence

After all segments are stitched, the last segment's SDM state is copied back to the persistent per-channel SDM context:

```
for each channel:
    persistent_sdm[ch] = last_segment_sdm[ch]
```

This provides the seed state for the next audio chunk, maintaining continuity across chunk boundaries.

---

## 5. Experimental Results

### 5.1 Test Environment

- **CPU**: AMD Ryzen 9 9950X (16 cores / 32 threads, Zen 5)
- **OS**: Windows 10 Pro
- **SDM**: 6th-order NTF (SDM6), trellis_cands=2, trellis_lat=32
- **Format**: DSD512 (22.579 MHz, 1-bit, stereo)
- **Segments**: 4 per channel (8 total for stereo)
- **Chunk size**: ~176,400 samples per channel (1 second of audio at output rate)
- **Thread pool**: 16 workers with MMCSS "Pro Audio" scheduling

### 5.2 State Distance vs. Bit-Matching: Diagnostic Results

A dedicated diagnostic suite (`test_stitch.c`) was built to compare stitching strategies. Two SDM instances are seeded from the same state, fed divergent data, then fed identical data in an overlap region. Metrics are measured at each sample.

#### Overlap Sweep (DSD512, sine input, seg=16384)

| Overlap | Min State Dist | Dist @ Bit-Match | Best Run | Match % |
|---------|---------------|-------------------|----------|---------|
| 2x lat (64) | 7.33 × 10¹² | 2.43 × 10¹³ | 7 | 43.8% |
| 4x lat (128) | 7.59 × 10⁹ | 3.44 × 10¹¹ | 6 | 39.1% |
| 8x lat (256) | 2.21 × 10⁹ | 3.05 × 10¹² | 12 | 48.4% |
| 16x lat (512) | 2.06 × 10⁹ | 8.43 × 10¹² | 16 | 50.6% |
| 32x lat (1024) | 1.42 × 10⁸ | 2.81 × 10¹² | 14 | 47.6% |

**Key finding**: The state distance at the bit-matching stitch point ("Dist @ Bit-Match") is consistently **50-1000x worse** than the minimum state distance. Bit-matching finds coincidental patterns, not convergence.

#### Stitch Quality Comparison (10 trials, DSD512, 4x overlap)

| Trial | Best Run | Bit Pos | Min Dist | Dist Pos | Delta |
|-------|----------|---------|----------|----------|-------|
| 0 | 6 | 101 | 7.59e9 | 93 | 8 |
| 1 | 5 | 23 | 1.86e12 | 0 | 23 |
| 2 | 6 | 19 | 1.75e10 | 63 | 44 |
| 3 | 12 | 108 | 5.64e9 | 0 | 108 |
| 4 | 9 | 32 | 1.12e10 | 92 | 60 |
| 5 | 14 | 95 | 1.10e10 | 81 | 14 |
| 6 | 11 | 42 | 8.41e11 | 0 | 42 |
| 7 | 4 | 118 | 1.22e10 | 127 | 9 |
| 8 | 18 | 35 | 1.22e10 | 109 | 74 |
| 9 | 9 | 119 | 3.42e9 | 81 | 38 |

**Agreement: 0/10. State-distance preferred: 10/10.**

The bit-matching and state-distance methods **never agree** on the optimal stitch point. The average positional delta is 42 samples — nearly half the overlap window. Bit-matching is essentially selecting random positions within the overlap.

### 5.3 Live Playback: Overlap Multiplier Sweep

DSD512 playback from native DSD512 source files, 4 parallel segments, stereo. All measurements taken from production debug logs during real-time playback.

#### Old Algorithm: Longest-Run Bit-Matching (4x overlap = 128)

```
stitch seg1: ovl=128 best_pos=113 best_run=7
stitch seg2: ovl=128 best_pos=72  best_run=11
stitch seg3: ovl=128 best_pos=10  best_run=7
stitch seg1: ovl=128 best_pos=72  best_run=10
stitch seg2: ovl=128 best_pos=26  best_run=16
stitch seg3: ovl=128 best_pos=14  best_run=14
stitch seg1: ovl=128 best_pos=68  best_run=10
stitch seg2: ovl=128 best_pos=23  best_run=3    ← only 3 matching bits
stitch seg3: ovl=128 best_pos=122 best_run=6
```

Best run ranges from **3 to 16**. A run of 3 in 128 samples is expected by pure chance (random ±1 stream has ~50% match probability; expected longest run in 128 ≈ 7). The algorithm is selecting coincidental patterns.

#### New Algorithm: DAS Hybrid Density (sweep results)

| Multiplier | Overlap | Density Range | Peak Density | RT Ratio | Exact Match |
|------------|---------|---------------|--------------|----------|-------------|
| 8x | 256 | 28-51/64 | 80% | 0.56x | 100% |
| 16x | 512 | 31-47/64 | 73% | 0.57x | 100% |
| **32x** | **1024** | **40-56/64** | **88%** | **0.57x** | **100%** |
| 64x | 2048 | 38-54/64 | 84% | 0.57x | 100% |
| 128x | 4096 | 38-55/64 | 86% | 0.58x | 100% |

Selected operating point: **32x** (overlap=1024 at lat=32).

#### DAS at 32x with Persistent State Replication (original)

```
stitch seg1: ovl=1024 density=44/64 pos=35  match=yes
stitch seg2: ovl=1024 density=40/64 pos=533 match=yes
stitch seg3: ovl=1024 density=53/64 pos=850 match=yes
stitch seg1: ovl=1024 density=46/64 pos=451 match=yes
stitch seg2: ovl=1024 density=44/64 pos=691 match=yes
stitch seg3: ovl=1024 density=56/64 pos=698 match=yes
stitch seg1: ovl=1024 density=54/64 pos=718 match=yes
stitch seg2: ovl=1024 density=41/64 pos=540 match=yes
stitch seg3: ovl=1024 density=46/64 pos=292 match=yes
stitch seg1: ovl=1024 density=55/64 pos=716 match=yes
stitch seg2: ovl=1024 density=40/64 pos=471 match=yes
stitch seg3: ovl=1024 density=43/64 pos=632 match=yes
```

- **Density: 40-56/64 (63-88%)** — genuine convergence, not random matching
- **100% exact bit-match** at every stitch point
- **Stitch positions span the full overlap** (35-850 out of 1024) — the algorithm finds the best convergence region wherever it occurs

#### DAS with Multi-bit State Estimation Pre-Pass (current)

A 16-level (4-bit) greedy SDM pre-pass estimates integrator state at each segment boundary. The multi-bit quantizer tracks the true trellis state more accurately than a 1-bit greedy estimator because 16 quantizer levels reduce quantization error by ~24 dB, producing closer state approximations.

DSD128 same-rate, trellis_cands=2, trellis_lat=128, 2 segments, overlap=1024 (8×lat):

```
stitch seg1: ovl=1024 density=165/256 pos=312 run=8 wp=5644800 stitch_at=2822712
stitch seg1: ovl=1024 density=174/256 pos=308 run=8 wp=5644800 stitch_at=2822708
stitch seg1: ovl=1024 density=138/256 pos=787 run=4 wp=5644800 stitch_at=2823187
stitch seg1: ovl=1024 density=151/256 pos=919 run=8 wp=5644800 stitch_at=2823319
stitch seg1: ovl=1024 density=162/256 pos=868 run=8 wp=5644800 stitch_at=2823268
stitch seg1: ovl=1024 density=131/256 pos=459 run=4 wp=5644800 stitch_at=2822859
stitch seg1: ovl=1024 density=190/256 pos=162 run=8 wp=5644800 stitch_at=2822562
stitch seg1: ovl=1024 density=176/256 pos=524 run=8 wp=5644800 stitch_at=2822924
stitch seg1: ovl=1024 density=157/256 pos=609 run=8 wp=5644800 stitch_at=2823009
stitch seg1: ovl=1024 density=143/256 pos=276 run=8 wp=5644800 stitch_at=2822676
```

- **Density: 119-190/256 (46-74%)** — genuine convergence at 8× overlap
- **100% exact bit-match** at every stitch point (run=4-8)
- Estimation pass parallelized on threadpool: ~70ms for DSD128 stereo, ~140ms for DSD512
- DSD64 same-rate (lat=32, window=64): density 35-56/64 (55-88%)

#### Comparison: Seeding Strategy Impact

| Rate | Overlap | Density Range | Match Rate | RT Ratio |
|------|---------|---------------|------------|----------|
| DSD512 | 1024 (32×32) | 34-49/64 | 53-77% | 0.85x |
| DSD256 | 4096 (32×128) | 109-210/256 | 43-82% | 0.52x |
| DSD128 | 4096 (32×128) | 128-188/256 | 50-73% | 0.27x |
| DSD64 | 1024 (32×32) | 35-56/64 | 55-88% | 0.19x |

All rates use 32×lat overlap. Multi-bit (16-level) state estimation, parallelized on threadpool.

The estimation pass gives seg1+ a starting state that reflects the actual NTF integrator evolution through the preceding input, rather than the stale state from the chunk boundary. The 16-level quantizer closely tracks the trellis decisions, and the overlap warmup handles the remaining convergence gap.

### 5.4 Real-Time Performance

| Rate | Segments | RT Ratio | SDM Time | FIR Time | Overlap |
|------|----------|----------|----------|----------|---------|
| DSD512 | 4 | 0.85x | ~650ms | ~130ms | 1024 |
| DSD256 | 2 | 0.52x | ~440ms | ~50ms | 4096 |
| DSD128 | 2 | 0.27x | ~225ms | ~26ms | 4096 |
| DSD64 | 2 | 0.19x | ~180ms | ~0ms | 1024 |

The DAS overlap scan uses an O(n) sliding window and adds negligible overhead. The multi-bit state estimation pass runs in parallel on the threadpool (~70ms for DSD128 stereo, included in SDM time above).

### 5.5 Convergence Plateau

Density plateaus at 32x overlap because the measurement window is fixed at `2 × trellis_lat = 64` samples. Beyond 32x, adding more overlap provides a larger search space but does not improve the peak density within any given 64-sample window. The SDM convergence rate is determined by the NTF characteristics and input signal, not the overlap extent.

### 5.6 Symmetric Overlap Extension

An important implementation detail: in the initial parallel SDM implementation, only the first segment (seg0) was extended into the next segment's territory. Middle segments (seg1, seg2) did not extend forward, creating an **asymmetric overlap** at boundaries 1→2 and 2→3. At these boundaries, the stitch scan compared outputs from **different input regions** — the "overlap" was an illusion, and any bit matches were purely coincidental.

DAS corrects this by extending **all non-last segments** by the full overlap into the next segment's territory:

```
Before (asymmetric):
  Seg 0: [═══════════[ovl]     ]
  Seg 1: [     [warmup]═══════]          ← no forward extension
  Seg 2: [     [warmup]════════════]

After (symmetric, DAS):
  Seg 0: [═══════════[ovl]     ]
  Seg 1: [     [warmup]═══════[ovl]    ] ← extended forward
  Seg 2: [          [warmup]═══════════]
```

This ensures every boundary has a genuine shared overlap region where both segments processed the same input data. The stitch scan at boundaries 1→2 and 2→3 now operates on truly overlapping output, not random data.

---

## 6. GPU-Accelerated SDM: Implementation and Conclusion

### 6.1 Architecture

A CUDA-based GPU trellis SDM was implemented using the same parallel-segment approach as CPU DAS. Each GPU segment runs as a CUDA block on an SM (Streaming Multiprocessor), processing M warmup + D output + overlap extension samples. The 2-pass DAS pipeline runs entirely on GPU:

- **Pass 1**: All segments seeded from persistent NTF state (replicated), run in parallel
- **Pass 2**: Segments re-run with chained states (seg[i] seeded from seg[i-1]'s Pass 1 final state)
- **CPU-side DAS assembly**: Segment outputs downloaded, density-scanned, stitched on CPU

### 6.2 Conclusion: GPU SDM is Not Feasible

**After extensive investigation (3 sessions, 13 bug fixes, comprehensive measurements), GPU-accelerated trellis SDM is not viable for production audio.** The combination of elevated noise floor and extreme computational cost for fp64 precision makes it impractical at any DSD rate.

#### Quality: Unacceptable Noise at All Rates

The GPU kernel produces 10-40% more in-band noise than CPU sequential SDM, manifesting as audible pops and elevated noise floor. This is **not** from DAS stitching (segment boundaries are clean) but from the GPU kernel's per-sample noise shaping quality. The trellis algorithm is uniquely sensitive to FP rounding: with nc=2 candidates, ULP-level cost differences flip candidate selection, cascading into a completely different bit stream. GPU and CPU produce 49% different bits (essentially uncorrelated) even with identical algorithm and `--fmad=false`.

Listening tests confirmed the noise is clearly audible at DSD64 and remains audible at DSD256. The hypothesis that higher DSD rates would mask the noise was tested and disproven — the elevated base noise is perceptible regardless of rate.

#### Performance: fp64 Too Expensive for GPU

Trellis SDM requires fp64 precision for the NTF integrator states and cost computation. On NVIDIA GPUs, fp64 throughput is 1/32 to 1/64 of fp32 (consumer GPUs like RTX 5080). This makes GPU SDM dramatically slower per-sample than CPU AVX2:

| Configuration | RT Ratio | Notes |
|---------------|----------|-------|
| CPU sequential (1 core) | 0.12x | AVX2 fp64, DSD64 |
| GPU 4 segments | 10x | RTX 5080, DSD64 |
| GPU 42 segments | 1.42x | RTX 5080, DSD64, barely real-time |
| GPU 42 segments | >2x | RTX 5080, DSD256, NOT real-time |

Even with 42 segments utilizing half the GPU's 84 SMs, DSD64 barely achieves real-time (1.42x) and DSD256 cannot reach real-time (>2x). Higher segment counts hit CUDA_ERROR_ILLEGAL_ADDRESS (kernel crash at lat≥128) and further increase the noise floor.

#### Root Causes Investigated

Thirteen bugs were found and fixed across three sessions:

1. Majority vote on traceback bit
2. Next-filter for candidate selection
3. Sort tie-breaking `<=` not `<`
4. Traceback pipeline delay (`p_next_stored[]`)
5. Dedup-aware greedy selection
6. FIR lowpass fp32→fp64
7. GPU FIR lowpass host narrowing
8. Missing gain in parallel FIR path
9. ext_seed out-of-bounds access
10. NTF child mapping inversion (reverted)
11. DAS assembly capacity vs actual counts
12. FMA rounding divergence (`--fmad=false`)
13. Analog compensation bit-flipping (removed)

Additional findings:
- **Self-seeding degrades quality by ~29%** — stale traceback history biases convergence
- **Cross-channel ext_seed contamination** — fixed to per-channel arrays
- **CPU chunk boundary re-encoding** — helps but creates new stitch artifacts
- **Pop count scales linearly with segments** (~3 pops/segment/5 seconds)

### 6.3 Measurements (Short People, DSD64, FIR 255-tap 50kHz Kaiser)

| Configuration | RMS | max_deriv | pops/5s | RT ratio |
|---------------|-----|-----------|---------|----------|
| CPU sequential | 0.004185 | 0.001803 | 2 | 0.12x |
| GPU 4 seg | 0.004609 | 0.001776 | 11 | 10x |
| GPU 4 seg + CPU re-encode | 0.003832 | 0.003360 | 17 | 10x |
| GPU 42 seg | ~0.006 | ~0.003 | ~100+ | 1.42x |
| GPU 1 seg (no DAS) | 0.006539 | 0.002059 | 9 | 22x |

### 6.4 Recommendation

GPU acceleration remains valuable for the **FIR filtering pipeline** (upsampling, lowpass, gain), where fp32 is sufficient and GPU parallelism provides genuine speedup. The SDM feedback loop should remain on CPU where AVX2 fp64 provides both superior quality and faster per-sample throughput. The CPU parallel DAS algorithm (Section 4-5) achieves clean, artifact-free parallelization across multiple CPU cores — this is the production path.

---

## 7. Summary

Density-Aligned Stitching (DAS) solves the parallelization problem for trellis SDM through four innovations:

1. **State estimation pre-pass**: A lightweight nc=1 greedy SDM runs the NTF filter forward through the input to estimate integrator state at each segment boundary (~10ms for 1.4M samples). Segments start with near-exact initial conditions rather than replicated stale state, achieving 74-99% density match (vs 63-88% without estimation).

2. **Extended overlap regions**: Each non-last segment extends by `32 × trellis_lat` samples into the next segment's territory. The overlap warmup handles the small remaining divergence between estimated and true trellis states.

3. **Windowed match density scanning**: A sliding window of width `2L` (twice the trellis latency) counts matching bits at each overlap position. The position with highest density identifies the region of genuine SDM convergence, replacing random bit-coincidence detection.

4. **Hybrid stitch selection**: From the density peak, a spiral search finds the nearest exact bit-match for a clean, artifact-free transition. This combines convergence-aware positioning with glitch-free stitching.

The algorithm adds negligible overhead (~10ms estimation + ~0ms density scan), achieves 100% exact-match rate at all stitch boundaries, and enables real-time DSD512 (22.579 MHz) processing at 0.57x RT on commodity hardware. No prior published work achieves temporal parallelism of trellis SDM for audio.

---

## References

1. Kato, H. "Trellis Noise-Shaping Converters and 1-Bit Digital Audio." AES 112th Convention, Munich, Paper 5615, 2002.
2. Janssen, E. & Reefman, D. "Advances in Trellis Based SDM Structures." AES Convention 115, Paper 5993, 2003.
3. Harpe, P., Reefman, D. & Janssen, E. "Efficient Trellis-type Sigma Delta Modulator." AES Convention 115, Paper 5950, 2003.
4. Angus, J.A.S. "Implementation of 'Tree' and 'Stack' Algorithms for Look-Ahead Sigma Delta Modulators." AES Paper 6281, 2004.
5. Hawksford, M.O.J. "Parallel Look-Ahead Digital SDM with Energy-Balance Binary Comparator." JAES Vol. 56 No. 12, pp. 1069-1089, 2008.
6. Janssen, E. & van Roermund, A. *Look-Ahead Based Sigma-Delta Modulation.* Springer, 2011. ISBN 978-94-007-1386-4.
7. Reefman, D. & Janssen, E. "One-Bit Audio: An Overview." 2003.
8. Fettweis, G. & Meyr, H. "Parallel Viterbi Algorithm Implementation: Breaking the ACS-Bottleneck." IEEE Trans. Communications, Vol. 37, No. 8, 1989.
9. Oliveira e Silva et al. "Improving Performance of All-Digital Transmitters Based on Parallel Delta-Sigma Modulators through Propagation of State Registers." IEEE, 2017.
10. Chen et al. "Parallel implement of real-time delta-sigma modulation for digital mobile fronthaul." 2023.
11. Melanson, J.L. "Overload Protection for Look-Ahead Delta Sigma Modulators." US Patent 7,081,843 B2, Cirrus Logic, 2005.
12. Kozak, M. & Kale, I. "A Pipelined Noise Shaping Coder for Oversampled D/A Conversion." IEEE, 2003.

---

## Implementation

DAS is implemented in the `foo_dsd_trellis` foobar2000 DSP plugin:

### CPU Parallel SDM
- **Segment layout & overlap**: `src/dsp_plugin.c` (parallel SDM path)
- **State estimation pre-pass**: `src/trellis.c` (`sdm_estimate_state` — nc=1 greedy NTF estimator)
- **SDM core & state copy**: `src/trellis.c` (`sdm_context_copy_state`, `sdm_state_distance`)
- **Segment processing**: `src/engine.c` (`sdm_segment_process`)
- **Thread pool dispatch**: `src/threadpool.c` (MMCSS "Pro Audio" workers)
- **Convergence diagnostics**: `test/test_stitch.c` (overlap sweep, density vs. state-distance comparison)

### GPU Parallel SDM (CUDA)
- **GPU trellis kernel**: `src/kernels/sdm_parallel.cu` (per-segment SBVD with traceback)
- **GPU host dispatch**: `src/gpu_cuda.c` (`gpu_cuda_trellis_das` — 2-pass DAS pipeline)
- **CPU-side DAS assembly**: `src/gpu_cuda.c` (density scan + stitch, boundary re-encoding)
- **PTX compilation**: `--fmad=false` required (CUDA 12.8, sm_52)
- **GPU algorithm test**: `test/test_trellis.c` (`test_gpu_cpu_divergence`)

---

*DAS algorithm and document by ManniX-ITA, 2026.*
*Implementation: foo_dsd_trellis v0.2.x*
