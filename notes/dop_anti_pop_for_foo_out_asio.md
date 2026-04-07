# DoP anti-pop for `foo_out_asio+dsd`

A reference / suggestion for Maxim covering: where the silence has to be
injected, what the silence bytes have to be, how rate changes need to be
handled, and the XMOS-specific quirks that make this trickier than it looks.

## TL;DR

1. The silence **must** be injected at the **output plugin**, not the
   input/decoder side. Anything earlier in the chain gets reshaped by the
   downstream DSPs and the timing slips so the silence no longer covers the
   noisy DAC mode-switch window.
2. Use the proper **DoP idle pattern** (`0x05 96 96` / `0xFA 96 96`
   alternating per frame), **not** raw zeros — XMOS will not detect raw zeros
   as DSD.
3. Inject silence **on every format change**, not just at playback start.
   Each new sample rate causes the DAC to re-lock; the noisy window happens
   every time.
4. **35 ms at the *previous* rate, then 35 ms at the *target* rate** is the
   standard "mute by rate change" trick that works on every XMOS DAC we've
   measured. 500 ms simple silence works on most but not all.
5. Track the marker A/B phase carefully — a single ±1 frame discrepancy
   inverts the phase and XMOS drops to PCM mode.

## Why injecting silence in `foo_input_sacd` doesn't work

Three reasons, in order of severity:

1. **Timing**: input-side silence has to traverse the entire DSP chain
   (resampler, ML, SDM, convolution, antipop, ...). By the time it reaches
   the ASIO buffer the rate change has already happened — the DAC has
   finished its mode switch and is *waiting* for samples. The silence
   arrives **after** the noisy window, not during it.

2. **Transformation**: the silence bytes get reshaped by every intermediate
   plugin. SDMs can produce non-zero output from zero input (especially
   during their own warmup). Resamplers add filter ringing. Convolution
   spreads the silence across the IR length. Even if the data ends at the
   right time, it's no longer DoP-silent at the byte level.

3. **Marker phase**: if any plugin in the chain duplicates, drops, or
   reorders frames by even ±1, the 0x05/0xFA marker alternation flips and
   XMOS sees the very *first* frame of "silence" as an invalid marker —
   then the whole stream drops back to PCM.

The output plugin doesn't suffer from any of this. It owns the ASIO buffer
directly and writes the very last bytes the DAC sees.

## Where to inject silence in the ASIO output plugin

Two hook points, both required:

### 1. Initial playback start

Right after `ASIOStart()` returns and before the first decoded audio sample
is fed into the ASIO buffer. The first thing the DAC processes after
`ASIOStart()` should be the warmup silence sequence.

### 2. Format change between tracks

Whenever the incoming audio format differs from the currently active one
(rate change OR channel count change), the output plugin should:

1. Drain the current buffer to the DAC (so the DAC's currently-playing
   audio isn't truncated mid-sample)
2. Send a short trail of DoP silence at the **current** rate (≥ 35 ms)
3. Switch the ASIO sample rate (`ASIOSetSampleRate(new_rate)` →
   `ASIOStop()` → `ASIOStart()` if needed)
4. Send a fresh warmup of DoP silence at the **new** rate (≥ 35 ms)
5. Then start feeding the actual audio

The "trail at current rate" before the switch is what triggers the DAC's
internal mute on the rate change — by the time the rate changes, the DAC
is already silenced. The "head at new rate" gives it 32+ valid DoP markers
to lock the new rate before audio arrives.

## The DoP silence pattern (XMOS-correct)

```
Standard DoP v1.1 frame, 24-bit big-endian PCM, per channel per sample:

  byte 0 (MSB) = marker byte: 0x05 or 0xFA, alternating per frame
  byte 1       = upper 8 DSD bits
  byte 2 (LSB) = lower 8 DSD bits

For DSD silence (analog zero with zero DC), use 0x96 0x96:

  Frame N (even): 0x05 0x96 0x96
  Frame N+1     : 0xFA 0x96 0x96
  Frame N+2     : 0x05 0x96 0x96
  ...
```

`0x96 0x96` = `1001 0110 1001 0110` = 16 alternating-pair DSD bits, which
is the canonical DSD "true silence" pattern (zero DC content). The XMOS
underflow word is documented as `0xFA969600` (marker A + 0x96 0x96 + 0x00
sync), confirming this is what XMOS firmware itself emits when its FIFO
runs dry.

**Do not use raw zeros (`0x00 0x00 0x00`)**. Without the marker bytes XMOS
sees PCM, not DSD, and either drops out or produces a different artifact.

**Do not use `0x69` instead of `0x96`** unless you've confirmed your DAC
treats them equivalently — `0x69` is the bit-inverse and produces the same
analog zero, but firmware is finicky.

## XMOS DAC behavior to know about

This is the firmware behavior we've observed across XMOS reference designs
and most XMOS-based commercial DACs (Singxer, Matrix, Holo, Topping,
SMSL, Gustard, ...):

| Behavior | Detail |
|---|---|
| **Enter DSD mode** | Requires **32 consecutive valid DoP markers on ALL channels** before the firmware switches to DSD output. Latency: ~180 µs after the 32nd marker. |
| **Exit DSD mode** | A **single missing/invalid marker on ANY channel** causes immediate PCM fallback. No hysteresis. |
| **Mute on rate change** | Internally: clocks low → `AudioHwConfig_Mute()` → reconfigure clocks → `AudioHwConfig_UnMute()`. The mute envelope is what causes the audible pop if real audio is present during the unmute. |
| **Underflow** | XMOS firmware emits `0xFA 96 96 00` per channel when the input FIFO runs dry. This is *also* what we should send during anti-pop windows — same byte pattern. |
| **Thesycon driver lockup** | After heavy DSD512 use, Thesycon USB drivers can enter a bad state where the next stream open fails or produces noise. Recovery: open the device with **250 ms of 44.1 kHz PCM silence** before re-opening for DSD. This forces the driver out of its DSD state and resets the USB pipe. |

The implications for the silence injection:

- **At least 32 frames** of valid DoP must be sent before any real audio,
  or the DAC won't be in DSD mode when the audio arrives. 32 frames at
  DSD64 PCM rate (176.4 kHz) = ~180 µs, but you want a substantial safety
  margin to cover the analog mute envelope, so use 35 ms minimum.
- **Marker phase MUST be unbroken** across the silence-to-audio transition.
  If the silence ends on a 0x05 frame, the first audio frame must be a
  0xFA frame.
- **Per-channel matters**: silence has to be written into ALL channels of
  the ASIO buffer simultaneously. A single channel with a stale frame can
  trigger the "invalid marker on any channel" exit.

## Reference C implementation

A drop-in helper for the ASIO buffer-switch callback:

```c
/* DoP anti-pop helper for foo_out_asio+dsd
 *
 * Call once after ASIOStart() (initial warmup) and again on every
 * format change (trail at old rate + warmup at new rate).
 */

#include <stdint.h>
#include <string.h>

/* DoP marker phase — persists across calls so the alternation never
 * breaks. Reset to 0 only on full ASIO close/reopen. */
static int g_dop_marker_phase = 0;  /* 0 → next frame uses 0x05, 1 → 0xFA */

/* Build a single 24-bit big-endian DoP silence frame for one channel. */
static inline void build_dop_silence_frame(uint8_t *dst3, int phase) {
    dst3[0] = phase ? 0xFA : 0x05;  /* marker byte */
    dst3[1] = 0x96;                  /* DSD silence pattern (zero DC) */
    dst3[2] = 0x96;
}

/* Write `frames` of DoP silence into the ASIO output buffer.
 * num_channels: total channel count (must write all channels per frame).
 * asio_write_24be(ch, frame_idx, bytes3): your existing helper that puts
 * one 24-bit big-endian sample into the ASIO buffer for the given channel
 * at the given frame offset within the current ASIO block.
 *
 * Returns the next ASIO frame offset to write to.
 */
static int write_dop_silence(int num_channels, int frame_offset, int frames,
                             void (*asio_write_24be)(int ch, int frame, const uint8_t *bytes3))
{
    uint8_t silent[3];
    for (int f = 0; f < frames; f++) {
        build_dop_silence_frame(silent, g_dop_marker_phase);
        for (int ch = 0; ch < num_channels; ch++)
            asio_write_24be(ch, frame_offset + f, silent);
        g_dop_marker_phase ^= 1;
    }
    return frame_offset + frames;
}

/* Compute frames-per-millisecond for the current ASIO sample rate (which
 * for DoP is the PCM-encoded rate: 176400 for DSD64, 352800 for DSD128,
 * 705600 for DSD256, 1411200 for DSD512). */
static int frames_for_ms(int sample_rate, int ms) {
    return (int)(((int64_t)sample_rate * ms) / 1000);
}

/* === High-level entry points === */

/* Call once after ASIOStart() returns, before feeding any real audio.
 * 35 ms is the minimum we've measured that suppresses the pop on every
 * XMOS DAC tested. Some DACs need 100 ms; make this configurable. */
void dop_warmup_at_start(int num_channels, int pcm_rate,
                         int (*asio_buffer_advance)(int frames),
                         void (*asio_write_24be)(int ch, int frame, const uint8_t *bytes3))
{
    const int ms = 35;
    int total = frames_for_ms(pcm_rate, ms);
    int written = 0;
    while (written < total) {
        /* Use whatever the current ASIO buffer's free-frame count is.
         * Wait for the next bufferSwitch callback if we run out of room. */
        int chunk = asio_buffer_advance(total - written);
        if (chunk <= 0) break;
        /* asio_buffer_advance is assumed to give you a frame_offset to
         * write to and to advance internal write cursor by `chunk`. The
         * actual integration depends on your ASIO buffer abstraction. */
        write_dop_silence(num_channels, /*frame_offset=*/0, chunk, asio_write_24be);
        written += chunk;
    }
}

/* Call when the incoming audio format changes. This emits a 35 ms trail
 * at the OLD rate (which triggers the DAC's internal mute on the rate
 * change) followed by 35 ms of silence at the NEW rate (which lets the
 * DAC re-lock the DoP stream before real audio arrives). */
void dop_handle_format_change(int num_channels, int old_pcm_rate, int new_pcm_rate,
                              int (*asio_set_sample_rate)(int rate),
                              int (*asio_buffer_advance)(int frames),
                              void (*asio_write_24be)(int ch, int frame, const uint8_t *bytes3))
{
    /* 1) Trail at OLD rate */
    {
        int total = frames_for_ms(old_pcm_rate, 35);
        int written = 0;
        while (written < total) {
            int chunk = asio_buffer_advance(total - written);
            if (chunk <= 0) break;
            write_dop_silence(num_channels, 0, chunk, asio_write_24be);
            written += chunk;
        }
    }

    /* 2) Tell ASIO to switch sample rate. Some drivers need stop+start. */
    asio_set_sample_rate(new_pcm_rate);

    /* 3) Reset marker phase on the new stream */
    g_dop_marker_phase = 0;

    /* 4) Warmup at NEW rate */
    {
        int total = frames_for_ms(new_pcm_rate, 35);
        int written = 0;
        while (written < total) {
            int chunk = asio_buffer_advance(total - written);
            if (chunk <= 0) break;
            write_dop_silence(num_channels, 0, chunk, asio_write_24be);
            written += chunk;
        }
    }
}

/* Optional: Thesycon recovery before opening DSD on a freshly-plugged or
 * just-stopped device. Sends 250 ms of PCM silence at 44.1 kHz to force
 * the driver out of any latched DSD state. Skip on non-Thesycon DACs. */
void thesycon_pcm_reset(int num_channels,
                        int (*asio_set_sample_rate)(int rate),
                        int (*asio_buffer_advance)(int frames),
                        void (*asio_write_24be)(int ch, int frame, const uint8_t *bytes3))
{
    asio_set_sample_rate(44100);
    int total = frames_for_ms(44100, 250);
    int written = 0;
    uint8_t pcm_zero[3] = { 0x00, 0x00, 0x00 };
    while (written < total) {
        int chunk = asio_buffer_advance(total - written);
        if (chunk <= 0) break;
        for (int f = 0; f < chunk; f++)
            for (int ch = 0; ch < num_channels; ch++)
                asio_write_24be(ch, f, pcm_zero);
        written += chunk;
    }
}
```

Adapt the `asio_write_24be` / `asio_buffer_advance` shims to your ASIO
buffer abstraction — the logic is the same regardless.

## Verification checklist

After implementing, you can verify each piece:

1. **Capture the ASIO buffer bytes** during a play-start and confirm:
   - First N frames are `0x05 96 96` / `0xFA 96 96` alternating
   - Marker phase is unbroken across the silence-to-audio boundary
   - No raw zeros in the silence window
2. **Test on at least 3 XMOS DACs** with different firmware ages — older
   XMOS reference firmware (XU208) is stricter than newer (XU316).
3. **Test rate-change paths**: DSD64 → DSD128 → DSD256 → DSD512 → DSD64.
   Each transition should be silent.
4. **Test stop → wait 5 s → play** (USB driver may have entered idle
   state). Anti-pop should still cover.
5. **Test with the foo_dsd_trellis DSP active** at every DSD rate
   conversion path. Our DSP also tracks marker phase across chunks; if
   the output plugin's anti-pop is correct, the entire chain stays clean.

## What doesn't work (don't waste time on these)

- **Sending silence in `bufferSwitch` for the first N callbacks but
  letting the underlying decoder also feed audio**: races. The decoder
  may have already pushed audio into the ASIO buffer by the time you
  start writing silence.
- **`ASIOOutputReady()` based gating**: this is about whether the host
  has consumed buffers, not about whether the DAC has locked.
- **Per-format `Sleep(50)` after `ASIOStart()`**: the DAC starts
  consuming the buffer immediately when `ASIOStart()` returns; sleeping
  doesn't change what's *in* the buffer. You have to write actual silence
  bytes.
- **Increasing the ASIO buffer size**: makes the pop softer but doesn't
  eliminate it, and adds latency for everyone.

## Contact

This note was written by the foo_dsd_trellis project as part of debugging
play-start pops and rate-change pops on DSD output. If anything in here
contradicts what you've measured, the measurement wins — please let us
know what you find.
