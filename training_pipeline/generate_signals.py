"""
foo_dsd_trellis — Synthetic test signal generator.

Generates Audio Precision-style test signals for training the DSD noise
reduction model. Each signal is saved as a WAV file at the specified
sample rate and duration.

Signal catalog:
  - Single sines (sub-bass to near-Nyquist)
  - Multitone (IMD stress testing, increasing spectral density)
  - Waveforms (square, triangle, sawtooth with harmonics)
  - JTest (jitter sensitivity patterns)
  - IMD standards (SMPTE, DIN, CCIF)
  - TIM/DIM (transient intermodulation)
  - Sweeps (log chirp, linear chirp, frequency sweep, level sweep)
  - Noise (white, pink, band-limited)
  - Silence (digital zero)
  - FM signals (frequency modulation)
  - Burst signals (CEA, EAIJ transient tests)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal as sig


# ─── Helpers ───

def normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Normalize to peak amplitude."""
    mx = np.max(np.abs(x))
    if mx > 0:
        x = x * (peak / mx)
    return x


def fade_inout(x: np.ndarray, fs: int, fade_ms: float = 10.0) -> np.ndarray:
    """Apply cosine fade-in/out to avoid clicks."""
    n = int(fs * fade_ms / 1000.0)
    if n > len(x) // 4:
        n = len(x) // 4
    if n < 1:
        return x
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(n) / n))
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def sine(fs: int, freq: float, duration: float, amplitude: float = 0.9,
         phase: float = 0.0) -> np.ndarray:
    """Generate a sine wave."""
    t = np.arange(int(fs * duration)) / fs
    return amplitude * np.sin(2.0 * np.pi * freq * t + phase)


def square_wave(fs: int, freq: float, duration: float,
                amplitude: float = 0.9) -> np.ndarray:
    """Band-limited square wave via additive synthesis."""
    t = np.arange(int(fs * duration)) / fs
    nyquist = fs / 2.0
    x = np.zeros_like(t)
    for k in range(1, 200, 2):  # odd harmonics
        f = freq * k
        if f >= nyquist:
            break
        x += np.sin(2.0 * np.pi * f * t) / k
    return normalize(x, amplitude)


def triangle_wave(fs: int, freq: float, duration: float,
                  amplitude: float = 0.9) -> np.ndarray:
    """Band-limited triangle wave via additive synthesis."""
    t = np.arange(int(fs * duration)) / fs
    nyquist = fs / 2.0
    x = np.zeros_like(t)
    for k in range(0, 100):
        n = 2 * k + 1
        f = freq * n
        if f >= nyquist:
            break
        x += ((-1.0) ** k) * np.sin(2.0 * np.pi * f * t) / (n * n)
    return normalize(x, amplitude)


def sawtooth_wave(fs: int, freq: float, duration: float,
                  amplitude: float = 0.9) -> np.ndarray:
    """Band-limited sawtooth wave via additive synthesis."""
    t = np.arange(int(fs * duration)) / fs
    nyquist = fs / 2.0
    x = np.zeros_like(t)
    for k in range(1, 200):
        f = freq * k
        if f >= nyquist:
            break
        x += ((-1.0) ** (k + 1)) * np.sin(2.0 * np.pi * f * t) / k
    return normalize(x, amplitude)


def white_noise(fs: int, duration: float, amplitude: float = 0.9,
                seed: int = 42) -> np.ndarray:
    """White noise."""
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    x = rng.standard_normal(n)
    return normalize(x, amplitude)


def pink_noise(fs: int, duration: float, amplitude: float = 0.9,
               seed: int = 42) -> np.ndarray:
    """Pink noise (1/f) via Voss-McCartney algorithm."""
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    # Use spectral shaping: generate white, FFT, multiply by 1/sqrt(f), IFFT
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    freqs[0] = 1.0  # avoid div by zero
    X *= 1.0 / np.sqrt(freqs)
    x = np.fft.irfft(X, n)
    return normalize(x, amplitude)


def bandlimited_noise(fs: int, duration: float, f_low: float, f_high: float,
                      amplitude: float = 0.9, seed: int = 42) -> np.ndarray:
    """Band-limited noise."""
    x = white_noise(fs, duration, 1.0, seed)
    sos = sig.butter(6, [f_low, f_high], btype='band', fs=fs, output='sos')
    x = sig.sosfilt(sos, x)
    return normalize(x, amplitude)


def log_chirp(fs: int, duration: float, f0: float, f1: float,
              amplitude: float = 0.9) -> np.ndarray:
    """Logarithmic frequency sweep."""
    t = np.arange(int(fs * duration)) / fs
    x = sig.chirp(t, f0, t[-1], f1, method='logarithmic')
    return fade_inout(amplitude * x, fs)


def linear_chirp(fs: int, duration: float, f0: float, f1: float,
                 amplitude: float = 0.9) -> np.ndarray:
    """Linear frequency sweep."""
    t = np.arange(int(fs * duration)) / fs
    x = sig.chirp(t, f0, t[-1], f1, method='linear')
    return fade_inout(amplitude * x, fs)


def multitone(fs: int, duration: float, n_tones: int,
              amplitude: float = 0.9, seed: int = 0) -> np.ndarray:
    """Equal-amplitude multitone with random phases.

    Frequencies are log-spaced from 20 Hz to 20 kHz (capped at Nyquist).
    """
    rng = np.random.default_rng(seed)
    max_freq = min(20000.0, fs / 2.0 - 100.0)
    freqs = np.geomspace(20.0, max_freq, n_tones)
    phases = rng.uniform(0, 2 * np.pi, n_tones)
    n = int(fs * duration)
    t = np.arange(n) / fs
    x = np.zeros(n)
    for f, p in zip(freqs, phases):
        x += np.sin(2.0 * np.pi * f * t + p)
    return normalize(x, amplitude)


def multitone_ap(fs: int, duration: float, n_tones: int,
                 amplitude: float = 0.9) -> np.ndarray:
    """Audio Precision-style multitone: Schroeder phases for low crest factor."""
    max_freq = min(20000.0, fs / 2.0 - 100.0)
    freqs = np.geomspace(20.0, max_freq, n_tones)
    n = int(fs * duration)
    t = np.arange(n) / fs
    x = np.zeros(n)
    for k, f in enumerate(freqs):
        # Schroeder phase: phi_k = pi * k * (k+1) / N
        phase = np.pi * k * (k + 1) / n_tones
        x += np.sin(2.0 * np.pi * f * t + phase)
    return normalize(x, amplitude)


# ─── IMD standard signals ───

def smpte_imd(fs: int, duration: float, amplitude: float = 0.9) -> np.ndarray:
    """SMPTE IMD: 60 Hz + 7 kHz at 4:1 ratio."""
    t = np.arange(int(fs * duration)) / fs
    x = 0.8 * np.sin(2 * np.pi * 60 * t) + 0.2 * np.sin(2 * np.pi * 7000 * t)
    return normalize(x, amplitude)


def din_imd(fs: int, duration: float, amplitude: float = 0.9) -> np.ndarray:
    """DIN IMD: 250 Hz + 8 kHz at 4:1 ratio."""
    t = np.arange(int(fs * duration)) / fs
    x = 0.8 * np.sin(2 * np.pi * 250 * t) + 0.2 * np.sin(2 * np.pi * 8000 * t)
    return normalize(x, amplitude)


def ccif_imd(fs: int, duration: float, f1: float, f2: float,
             amplitude: float = 0.9) -> np.ndarray:
    """CCIF twin-tone IMD: two equal-amplitude tones."""
    t = np.arange(int(fs * duration)) / fs
    f1 = min(f1, fs / 2.0 - 100)
    f2 = min(f2, fs / 2.0 - 100)
    x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    return normalize(x, amplitude)


# ─── TIM/DIM signals ───

def tim_signal(fs: int, duration: float, sq_freq: float, sine_freq: float,
               ratio: float, amplitude: float = 0.9) -> np.ndarray:
    """Transient intermodulation: band-limited square + sine at given ratio."""
    sq = square_wave(fs, sq_freq, duration, 1.0)
    sn = sine(fs, sine_freq, duration, 1.0)
    x = ratio / (ratio + 1) * sq + 1 / (ratio + 1) * sn
    return normalize(x, amplitude)


def dim_signal(fs: int, duration: float, sq_freq: float, sine_freq: float,
               ratio: float, amplitude: float = 0.9) -> np.ndarray:
    """DIM (dynamic intermodulation): same as TIM with specific standard freqs."""
    return tim_signal(fs, duration, sq_freq, sine_freq, ratio, amplitude)


def mim_signal(fs: int, duration: float, f1: float, f2: float, f3: float,
               amplitude: float = 0.9) -> np.ndarray:
    """MIM triple-tone: three equal-amplitude sines."""
    t = np.arange(int(fs * duration)) / fs
    x = (np.sin(2 * np.pi * f1 * t) +
         np.sin(2 * np.pi * f2 * t) +
         np.sin(2 * np.pi * f3 * t))
    return normalize(x, amplitude)


# ─── JTest ───

def jtest(fs: int, duration: float, bits: int = 24,
          amplitude: float = 0.9) -> np.ndarray:
    """JTest signal: square wave at fs/192 with LSB toggling.

    Standard jitter sensitivity pattern.
    """
    n = int(fs * duration)
    t = np.arange(n) / fs
    freq = fs / 192.0
    # High-level square wave (fills upper bits)
    x = np.sign(np.sin(2 * np.pi * freq * t))
    # Add LSB toggling pattern
    lsb_val = 2.0 ** (-(bits - 1))
    lsb = lsb_val * np.sign(np.sin(2 * np.pi * freq * 4 * t))
    x = x * 0.5 + lsb
    return normalize(x, amplitude)


# ─── FM signals ───

def fm_signal(fs: int, duration: float, carrier: float, mod_freq: float,
              deviation: float, amplitude: float = 0.9) -> np.ndarray:
    """Frequency-modulated signal."""
    t = np.arange(int(fs * duration)) / fs
    modulator = deviation * np.sin(2 * np.pi * mod_freq * t)
    phase = 2 * np.pi * carrier * t + 2 * np.pi * np.cumsum(modulator) / fs
    x = np.sin(phase)
    return normalize(x, amplitude)


# ─── Burst signals ───

def burst_signal(fs: int, duration: float, tone_freq: float,
                 burst_dur_ms: float, gap_dur_ms: float,
                 amplitude: float = 0.9) -> np.ndarray:
    """Repeated tone burst with gaps."""
    n = int(fs * duration)
    burst_n = int(fs * burst_dur_ms / 1000.0)
    gap_n = int(fs * gap_dur_ms / 1000.0)
    cycle = burst_n + gap_n
    x = np.zeros(n)
    t_burst = np.arange(burst_n) / fs
    tone = np.sin(2 * np.pi * tone_freq * t_burst)
    # Apply Hann window to burst edges
    win_n = min(burst_n // 10, int(fs * 0.5 / 1000))
    if win_n > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(win_n) / win_n))
        tone[:win_n] *= ramp
        tone[-win_n:] *= ramp[::-1]

    pos = 0
    while pos + burst_n <= n:
        end = min(pos + burst_n, n)
        x[pos:end] = tone[:end - pos]
        pos += cycle
    return normalize(x, amplitude)


def level_sweep(fs: int, duration: float, freq: float = 1000.0,
                db_start: float = -90.0, db_end: float = 0.0,
                amplitude: float = 0.9) -> np.ndarray:
    """Sine at fixed frequency with linearly increasing level (dB)."""
    n = int(fs * duration)
    t = np.arange(n) / fs
    db = np.linspace(db_start, db_end, n)
    envelope = 10.0 ** (db / 20.0)
    x = envelope * np.sin(2 * np.pi * freq * t)
    return normalize(x, amplitude)


# ─── Signal catalog ───

def get_signal_catalog():
    """Return list of (name, generator_function) tuples.

    Each generator takes (fs, duration) and returns np.ndarray.
    """
    catalog = []

    # Single sines
    for freq in [20, 50, 60, 1000, 3000, 5000, 10000, 11000, 12000, 15000]:
        name = f"sine_{freq}Hz"
        catalog.append((name, lambda fs, dur, f=freq: sine(fs, f, dur)))

    # Multitone (various tone counts)
    for n in [3, 4, 5, 6, 7, 10, 20, 32, 64, 100, 200, 300, 400, 500, 750, 1000]:
        name = f"multitone_{n}"
        catalog.append((name, lambda fs, dur, n=n: multitone(fs, dur, n)))

    # AP-style multitone (Schroeder phases)
    catalog.append(("multitone_32_AP", lambda fs, dur: multitone_ap(fs, dur, 32)))

    # Waveforms
    catalog.append(("square_100Hz", lambda fs, dur: square_wave(fs, 100, dur)))
    catalog.append(("square_1kHz", lambda fs, dur: square_wave(fs, 1000, dur)))
    catalog.append(("triangle_1kHz", lambda fs, dur: triangle_wave(fs, 1000, dur)))
    catalog.append(("sawtooth_1kHz", lambda fs, dur: sawtooth_wave(fs, 1000, dur)))
    catalog.append(("sawtooth_1kHz_rev",
                     lambda fs, dur: -sawtooth_wave(fs, 1000, dur)))

    # JTest
    catalog.append(("jtest_24bit", lambda fs, dur: jtest(fs, dur, 24)))
    catalog.append(("jtest_16bit", lambda fs, dur: jtest(fs, dur, 16)))

    # IMD standards
    catalog.append(("SMPTE_60Hz_7k", lambda fs, dur: smpte_imd(fs, dur)))
    catalog.append(("DIN_250Hz_8k", lambda fs, dur: din_imd(fs, dur)))
    catalog.append(("CCIF_19k_20k",
                     lambda fs, dur: ccif_imd(fs, dur, 19000, 20000)))
    catalog.append(("CCIF_18.5k_19.5k",
                     lambda fs, dur: ccif_imd(fs, dur, 18500, 19500)))

    # TIM / DIM
    catalog.append(("MIM_9k_10.05k_20k",
                     lambda fs, dur: mim_signal(fs, dur, 9000, 10050, 20000)))
    catalog.append(("TIM_Curl_sq3.18k_15k",
                     lambda fs, dur: tim_signal(fs, dur, 3180, 15000, 4.0)))
    catalog.append(("TIM_moar_sq1k_12k",
                     lambda fs, dur: tim_signal(fs, dur, 1000, 12000, 5.0)))
    catalog.append(("DIM30_sq3.15k_15k",
                     lambda fs, dur: dim_signal(fs, dur, 3150, 15000, 4.0)))
    catalog.append(("DIM100_sq3.15k_15k",
                     lambda fs, dur: dim_signal(fs, dur, 3150, 15000, 4.0)))

    # Sweeps
    catalog.append(("log_chirp_20_20k",
                     lambda fs, dur: log_chirp(fs, dur, 20, 20000)))
    catalog.append(("linear_chirp_20_20k",
                     lambda fs, dur: linear_chirp(fs, dur, 20, 20000)))
    catalog.append(("freq_sweep_20_40k",
                     lambda fs, dur: log_chirp(fs, dur, 20, min(40000, fs / 2 - 100))))
    catalog.append(("level_sweep_1kHz",
                     lambda fs, dur: level_sweep(fs, dur, 1000)))

    # Noise
    catalog.append(("white_noise", lambda fs, dur: white_noise(fs, dur, seed=42)))
    catalog.append(("white_noise_bl",
                     lambda fs, dur: bandlimited_noise(fs, dur, 20, 20000, seed=43)))
    catalog.append(("pink_noise", lambda fs, dur: pink_noise(fs, dur, seed=44)))
    catalog.append(("pink_noise_bl",
                     lambda fs, dur: bandlimited_noise(fs, dur, 20, 20000, seed=45)))

    # Silence
    catalog.append(("silence", lambda fs, dur: np.zeros(int(fs * dur))))

    # FM signals
    catalog.append(("FM_1k_200Hz_900Hz",
                     lambda fs, dur: fm_signal(fs, dur, 1000, 200, 900)))
    catalog.append(("FM_3k_50Hz_10k",
                     lambda fs, dur: fm_signal(fs, dur, 3000, 50, 10000)))

    # Burst signals
    catalog.append(("CEA2006_burst_1kHz",
                     lambda fs, dur: burst_signal(fs, dur, 1000, 6.5, 993.5)))
    catalog.append(("CEA2010_burst_100Hz",
                     lambda fs, dur: burst_signal(fs, dur, 100, 200, 800)))
    catalog.append(("EAIJ_burst_1kHz",
                     lambda fs, dur: burst_signal(fs, dur, 1000, 5.0, 995.0)))

    return catalog


def generate_all(output_dir: str, sample_rates: list[int],
                 duration: float = 5.0, bit_depth: int = 32):
    """Generate all synthetic signals at all sample rates."""
    catalog = get_signal_catalog()
    output_path = Path(output_dir)

    total = len(catalog) * len(sample_rates)
    count = 0

    for fs in sample_rates:
        rate_dir = output_path / f"{fs}Hz"
        rate_dir.mkdir(parents=True, exist_ok=True)

        for name, gen_fn in catalog:
            count += 1
            filepath = rate_dir / f"{name}.wav"

            if filepath.exists():
                print(f"  [{count}/{total}] SKIP {filepath.name} (exists)")
                continue

            print(f"  [{count}/{total}] {fs}Hz/{name} ...", end="", flush=True)
            try:
                x = gen_fn(fs, duration)
                x = np.asarray(x, dtype=np.float32)

                # Clip to safe range
                x = np.clip(x, -1.0, 1.0)

                subtype = 'FLOAT' if bit_depth == 32 else f'PCM_{bit_depth}'
                sf.write(str(filepath), x, fs, subtype=subtype)
                print(f" OK ({len(x)} samples, {len(x)/fs:.1f}s)")
            except Exception as e:
                print(f" FAILED: {e}")

    print(f"\nGenerated {count} signal files in {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic test signals for DSD ML training")
    parser.add_argument("--output", "-o", default="data/synthetic",
                        help="Output directory (default: data/synthetic)")
    parser.add_argument("--rates", nargs="+", type=int,
                        default=[44100, 88200, 176400],
                        help="PCM sample rates (default: 44100 88200 176400)")
    parser.add_argument("--duration", "-d", type=float, default=5.0,
                        help="Signal duration in seconds (default: 5.0)")
    parser.add_argument("--list", action="store_true",
                        help="List available signals and exit")
    args = parser.parse_args()

    if args.list:
        catalog = get_signal_catalog()
        print(f"Available signals ({len(catalog)}):")
        for name, _ in catalog:
            print(f"  {name}")
        return

    print(f"Generating signals: {len(get_signal_catalog())} signals x "
          f"{len(args.rates)} rates x {args.duration}s")
    print(f"Output: {args.output}")
    print(f"Sample rates: {args.rates}")
    print()
    generate_all(args.output, args.rates, args.duration)


if __name__ == "__main__":
    main()
