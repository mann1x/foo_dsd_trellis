"""Analyze captured WAV for convolution filter verification."""
import numpy as np
import struct
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/capture_conv.wav"

# Read WAV
with open(path, 'rb') as f:
    f.read(12)  # RIFF header
    rate = 44100
    nch = 2
    while True:
        chunk_id = f.read(4)
        if len(chunk_id) < 4:
            break
        chunk_size = struct.unpack('<I', f.read(4))[0]
        if chunk_id == b'fmt ':
            fmt = f.read(chunk_size)
            nch = struct.unpack('<H', fmt[2:4])[0]
            rate = struct.unpack('<I', fmt[4:8])[0]
            print(f"Format: ch={nch} rate={rate}")
        elif chunk_id == b'data':
            data = np.frombuffer(f.read(chunk_size), dtype=np.float32)
            break
        else:
            f.seek(chunk_size, 1)

# Take left channel
left = data[::nch]
print(f"Samples: {len(left)}, duration: {len(left)/rate:.2f}s")

# Decimate to ~44.1kHz for spectral analysis
dec_ratio = max(1, rate // 44100)
n = (len(left) // dec_ratio) * dec_ratio
left_dec = left[:n].reshape(-1, dec_ratio).mean(axis=1)
eff_rate = rate / dec_ratio
print(f"Decimated {dec_ratio}x to {len(left_dec)} samples at {eff_rate:.0f} Hz")

# FFT
N = min(len(left_dec), 65536)
window = np.hanning(N)
spectrum = np.abs(np.fft.rfft(left_dec[:N] * window))
freqs = np.fft.rfftfreq(N, d=1.0/eff_rate)

# Convert to dB
spectrum_db = 20 * np.log10(spectrum + 1e-30)
peak_db = float(np.max(spectrum_db))
spectrum_db -= peak_db

# Print frequency bins
bins = [
    ("Sub-bass (20-80 Hz)", 20, 80),
    ("Bass (80-200 Hz)", 80, 200),
    ("Low-mid (200-500 Hz)", 200, 500),
    ("Mid (500-2000 Hz)", 500, 2000),
    ("Upper-mid (2000-4000 Hz)", 2000, 4000),
    ("Presence (4000-8000 Hz)", 4000, 8000),
    ("Brilliance (8000-16000 Hz)", 8000, 16000),
]

print(f"\nSpectral Analysis (peak={peak_db:.1f} dB)")
print(f"{'Band':<30s} {'Avg dB':>8s}  {'Note':>10s}")
print("-" * 55)
for name, f_lo, f_hi in bins:
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if np.any(mask):
        avg = float(np.mean(spectrum_db[mask]))
        note = ""
        if f_lo >= 500 and f_hi <= 4000:
            note = "<-- CUT?"
        print(f"  {name:<28s} {avg:>8.1f}  {note}")

print("\nIf convolution works, 'Mid' and 'Upper-mid' should be")
print("significantly lower (>10 dB) than adjacent bands.")
