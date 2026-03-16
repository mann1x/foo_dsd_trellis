"""
foo_dsd_trellis — Training pair generator using the real C engine.

Uses foo_dsd_trellis_test.exe --encode to run the actual Trellis/PreCorr
SDM encoder with IPP FIR upsampling. Each SDM configuration produces
a different noise signature.

Pipeline per (signal, config, rate):
  1. Ensure input is WAV (decode FLAC/MP3/etc via soundfile if needed)
  2. Call dsd_encode with specific SDM settings → raw FIR ref + SDM output
  3. Save as .npy pairs for training

Requires: bin/Release/x64/foo_dsd_trellis_test.exe (built from solution)
"""

import argparse
import os
import subprocess
import struct
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Path to encoder exe (relative to training_pipeline/)
ENCODER_EXE = str(Path(__file__).parent.parent / "bin" / "Release" / "x64" /
                   "foo_dsd_trellis_test.exe")

# DSD rates
DSD_RATES = {
    "DSD64":  2822400,
    "DSD128": 5644800,
    "DSD256": 11289600,
    "DSD512": 22579200,
}

# SDM configuration matrix — each produces a distinct noise signature
SDM_CONFIGS = [
    # (name, sdm_mode, ntf, cands, lat, depth, state_limit, gain)
    # --- Trellis with different NTF filters ---
    ("trellis_clans4_c8_l256",  "trellis", "clans-4", 8,  256, 4, 0.0, 1.0),
    ("trellis_clans5_c8_l256",  "trellis", "clans-5", 8,  256, 4, 0.0, 1.0),
    ("trellis_clans6_c8_l256",  "trellis", "clans-6", 8,  256, 4, 0.0, 1.0),
    ("trellis_clans7_c8_l256",  "trellis", "clans-7", 8,  256, 4, 0.0, 1.0),
    ("trellis_clans8_c8_l256",  "trellis", "clans-8", 8,  256, 4, 0.0, 1.0),
    ("trellis_sdm5_c8_l256",    "trellis", "sdm-5",   8,  256, 4, 0.0, 1.0),
    ("trellis_sdm7_c8_l256",    "trellis", "sdm-7",   8,  256, 4, 0.0, 1.0),
    ("trellis_sdm8_c8_l256",    "trellis", "sdm-8",   8,  256, 4, 0.0, 1.0),
    # --- Trellis with different candidates ---
    ("trellis_clans8_c4_l256",  "trellis", "clans-8", 4,  256, 4, 0.0, 1.0),
    ("trellis_clans8_c16_l256", "trellis", "clans-8", 16, 256, 4, 0.0, 1.0),
    # --- Trellis with different latency ---
    ("trellis_clans8_c8_l64",   "trellis", "clans-8", 8,  64,  4, 0.0, 1.0),
    ("trellis_clans8_c8_l128",  "trellis", "clans-8", 8,  128, 4, 0.0, 1.0),
    ("trellis_clans8_c8_l512",  "trellis", "clans-8", 8,  512, 4, 0.0, 1.0),
    # --- Trellis with state limiter ---
    ("trellis_clans8_c8_l256_lim6",  "trellis", "clans-8", 8, 256, 4, 6.0, 1.0),
    ("trellis_clans8_c8_l256_lim10", "trellis", "clans-8", 8, 256, 4, 10.0, 1.0),
    ("trellis_clans8_c8_l256_lim12", "trellis", "clans-8", 8, 256, 4, 12.0, 1.0),
    # --- Trellis with FIR gain (overload prevention) ---
    ("trellis_clans8_c8_l256_g50",  "trellis", "clans-8", 8, 256, 4, 0.0, 0.5),
    ("trellis_clans8_c8_l256_g25",  "trellis", "clans-8", 8, 256, 4, 0.0, 0.25),
    # --- PreCorr ---
    ("precorr_auto",            "precorr", "auto",    0,  0,   0, 0.0, 1.0),
    ("precorr_clans5",          "precorr", "clans-5", 0,  0,   0, 0.0, 1.0),
    ("precorr_sdm5",            "precorr", "sdm-5",   0,  0,   0, 0.0, 1.0),
    ("precorr_clans8",          "precorr", "clans-8", 0,  0,   0, 0.0, 1.0),
    ("precorr_lim12",           "precorr", "auto",    0,  0,   0, 12.0, 1.0),
]


def ensure_wav(path: str) -> tuple[str, bool]:
    """Ensure input is WAV. If not, decode via soundfile to temp WAV.

    Returns (wav_path, is_temp). Caller deletes temp files.
    """
    ext = Path(path).suffix.lower()
    if ext == '.wav':
        return path, False

    # Decode any format soundfile/librosa can handle
    try:
        x, fs = sf.read(path, dtype='float32')
    except Exception:
        try:
            import librosa
            x, fs = librosa.load(path, sr=None, mono=True)
            x = x.astype(np.float32)
        except Exception as e:
            print(f"  Cannot decode {path}: {e}")
            return path, False

    # Mix to mono
    if x.ndim > 1:
        x = x.mean(axis=1)

    # Write temp WAV
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp_path = tmp.name
    tmp.close()
    sf.write(tmp_path, x, fs, subtype='FLOAT')
    return tmp_path, True


def read_raw_f32(path: str) -> np.ndarray:
    """Read raw float32 file as numpy array."""
    return np.fromfile(path, dtype=np.float32)


def run_encoder(input_wav: str, dsd_rate: int, config: tuple,
                fir_out: str, sdm_out: str,
                max_samples: int = 0) -> bool:
    """Run dsd_encode via subprocess."""
    name, sdm_mode, ntf, cands, lat, depth, state_limit, gain = config

    cmd = [
        ENCODER_EXE, "--encode",
        "--input", input_wav,
        "--dsd-rate", str(dsd_rate),
        "--sdm", sdm_mode,
        "--ntf", ntf,
        "--fir-out", fir_out,
        "--sdm-out", sdm_out,
        "--quiet",
    ]

    if sdm_mode == "trellis":
        cmd += ["--cands", str(cands), "--lat", str(lat), "--depth", str(depth)]

    if state_limit > 0:
        cmd += ["--state-limit", str(state_limit)]

    if gain != 1.0:
        cmd += ["--gain", str(gain)]

    if max_samples > 0:
        cmd += ["--samples", str(max_samples)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  ENCODER FAILED: {result.stderr.strip()}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  ENCODER TIMEOUT")
        return False
    except FileNotFoundError:
        print(f"  ENCODER NOT FOUND: {ENCODER_EXE}")
        print("  Build with: MSBuild foo_dsd_trellis.sln -p:Configuration=Release -p:Platform=x64")
        return False


def process_signal(wav_path: str, output_dir: Path,
                   dsd_rates: list[str], configs: list[tuple],
                   max_duration: float = 10.0) -> int:
    """Generate training pairs for one signal across all configs and rates."""
    stem = Path(wav_path).stem
    pairs = 0

    # Convert to WAV if needed
    input_wav, is_temp = ensure_wav(wav_path)

    # Compute max_samples from WAV sample rate
    max_samples = 0
    if max_duration > 0:
        try:
            info = sf.info(input_wav)
            max_samples = int(info.samplerate * max_duration)
        except Exception:
            pass

    try:
        for rate_name in dsd_rates:
            dsd_rate = DSD_RATES[rate_name]

            for cfg in configs:
                cfg_name = cfg[0]
                pair_dir = output_dir / rate_name / cfg_name / stem
                pair_dir.mkdir(parents=True, exist_ok=True)

                dsd_path = pair_dir / "dsd_input.npy"
                ref_path = pair_dir / "fir_reference.npy"

                # Skip if already generated
                if dsd_path.exists() and ref_path.exists():
                    pairs += 1
                    continue

                # Temp files for encoder output
                fir_raw = str(pair_dir / "_fir.raw")
                sdm_raw = str(pair_dir / "_sdm.raw")

                ok = run_encoder(input_wav, dsd_rate, cfg,
                                 fir_raw, sdm_raw, max_samples)
                if not ok:
                    # Clean up temp files
                    for p in [fir_raw, sdm_raw]:
                        if os.path.exists(p):
                            os.remove(p)
                    continue

                # Convert raw float32 to numpy
                try:
                    fir_data = read_raw_f32(fir_raw)
                    sdm_data = read_raw_f32(sdm_raw)

                    # Ensure same length
                    min_len = min(len(fir_data), len(sdm_data))
                    np.save(str(ref_path), fir_data[:min_len])
                    np.save(str(dsd_path), sdm_data[:min_len])
                    pairs += 1
                except Exception as e:
                    print(f"  Failed to save pair: {e}")

                # Clean up temp raw files
                for p in [fir_raw, sdm_raw]:
                    if os.path.exists(p):
                        os.remove(p)

    finally:
        if is_temp and os.path.exists(input_wav):
            os.remove(input_wav)

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate (DSD, FIR_reference) training pairs using real C engine")
    parser.add_argument("--signals", "-s", default="data/synthetic",
                        help="Directory containing synthetic WAV signals")
    parser.add_argument("--music", "-m", nargs="*", default=[],
                        help="Music file paths (WAV, FLAC, MP3, etc.)")
    parser.add_argument("--output", "-o", default="data/pairs",
                        help="Output directory for training pairs")
    parser.add_argument("--rates", nargs="+",
                        default=["DSD64"],
                        choices=list(DSD_RATES.keys()),
                        help="DSD rates to generate (default: DSD64)")
    parser.add_argument("--max-duration", type=float, default=10.0,
                        help="Max signal duration in seconds (default: 10)")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Config name filter (substring match, default: all)")
    parser.add_argument("--list-configs", action="store_true",
                        help="List available SDM configs and exit")
    parser.add_argument("--encoder", default=None,
                        help="Path to test exe (default: auto-detect)")
    args = parser.parse_args()

    if args.encoder:
        global ENCODER_EXE
        ENCODER_EXE = args.encoder

    if args.list_configs:
        print(f"Available SDM configs ({len(SDM_CONFIGS)}):")
        for cfg in SDM_CONFIGS:
            name, sdm, ntf, c, l, d, sl, g = cfg
            extra = f"cands={c} lat={l}" if sdm == "trellis" else ""
            if sl > 0: extra += f" lim={sl}"
            if g != 1.0: extra += f" gain={g}"
            print(f"  {name:40s} {sdm:8s} {ntf:8s} {extra}")
        return

    # Filter configs if requested
    configs = SDM_CONFIGS
    if args.configs:
        configs = [c for c in SDM_CONFIGS
                   if any(f.lower() == c[0].lower() or f.lower() in c[0].lower()
                          for f in args.configs)]
        # If any filter is an exact match, keep only exact matches
        exact = [c for c in SDM_CONFIGS
                 if any(f.lower() == c[0].lower() for f in args.configs)]
        if exact:
            configs = exact
        if not configs:
            print("No configs matched filter. Use --list-configs to see available.")
            return

    # Verify encoder exists
    if not os.path.exists(ENCODER_EXE):
        print(f"Encoder not found: {ENCODER_EXE}")
        print("Build with: MSBuild foo_dsd_trellis.sln -p:Configuration=Release -p:Platform=x64")
        return

    # Collect signal files
    signal_files = []

    synth_dir = Path(args.signals)
    if synth_dir.exists():
        for rate_dir in sorted(synth_dir.iterdir()):
            if rate_dir.is_dir():
                for wav in sorted(rate_dir.glob("*.wav")):
                    signal_files.append(str(wav))
        print(f"Found {len(signal_files)} synthetic signal files")

    for mf in args.music:
        if os.path.exists(mf):
            signal_files.append(mf)
            print(f"Added music: {mf}")
        else:
            print(f"WARNING: Not found: {mf}")

    if not signal_files:
        print("No signal files found. Run generate_signals.py first.")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_combinations = len(signal_files) * len(configs) * len(args.rates)
    print(f"\nGenerating pairs: {len(signal_files)} signals x "
          f"{len(configs)} configs x {len(args.rates)} rates = "
          f"{total_combinations} pairs")
    print(f"DSD rates: {args.rates}")
    print(f"SDM configs: {len(configs)}")
    print(f"Output: {output_dir}\n")

    total_pairs = 0
    for wav_path in tqdm(signal_files, desc="Signals"):
        n = process_signal(wav_path, output_dir, args.rates, configs,
                           args.max_duration)
        total_pairs += n

    print(f"\nGenerated {total_pairs} training pairs in {output_dir}")


if __name__ == "__main__":
    main()
