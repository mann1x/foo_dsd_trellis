"""
Complete adaptive pre-emphasis training pipeline.

Generates per-signal-type CMA-ES training data for all DSD rates,
combines into a single dataset, trains an MLP, and exports ONNX.

The pipeline runs end-to-end without manual intervention:
  1. For each of 8 DSD rates (/44 and /48):
     - Run CMA-ES black-box optimization on 20 signal types
     - Each signal type gets its own optimal 3-tap FIR taps
     - Fitness metric: A-weighted SINAD (end-to-end boxcar pipeline)
  2. Combine all (rate, signal_type) pairs into one training set
     - Features: [spectral_centroid, rms, crest_factor, rate_mhz]
     - Targets: [tap0, tap1, tap2]
  3. Train MLP: Linear(4->16)->ReLU->Linear(16->16)->ReLU->Linear(16->3)
  4. Export ONNX with baked-in feature normalization

Requirements:
  pip install cmaes numpy torch onnx onnxruntime

Usage:
  python -u train_all_rates.py
  python -u train_all_rates.py --rates dsd64 dsd128
  python -u train_all_rates.py --skip-generate   # use existing data, retrain only
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

PYTHON = sys.executable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATE_SCRIPT = os.path.join(SCRIPT_DIR, 'generate_preemph_data.py')
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, 'train_adaptive_fir.py')

ALL_RATES = [
    'dsd64', 'dsd128', 'dsd256', 'dsd512',
    'dsd64_48', 'dsd128_48', 'dsd256_48', 'dsd512_48',
]

DEFAULT_OUTPUT_DIR = 'I:/foo_dsd_trellis/checkpoints/cmaes'


def main():
    parser = argparse.ArgumentParser(
        description='Complete pre-emphasis training pipeline')
    parser.add_argument('--rates', nargs='+', default=ALL_RATES,
                        help='DSD rates to train (default: all 8)')
    parser.add_argument('--generations', type=int, default=15,
                        help='CMA-ES generations per signal type (default: 15)')
    parser.add_argument('--population', type=int, default=10,
                        help='CMA-ES population size (default: 10)')
    parser.add_argument('--workers', type=int, default=24,
                        help='Parallel evaluation workers (default: 24)')
    parser.add_argument('--taps', type=int, default=3,
                        help='FIR tap count (default: 3)')
    parser.add_argument('--hidden', type=int, default=16,
                        help='MLP hidden layer size (default: 16)')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='MLP training epochs (default: 2000)')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                        help='Directory for training data')
    parser.add_argument('--skip-generate', action='store_true',
                        help='Skip CMA-ES, use existing training data')
    args = parser.parse_args()

    t_total = time.time()

    # ── Stage 1: Generate per-signal training data ──
    if not args.skip_generate:
        print("=" * 60)
        print("STAGE 1: Generate per-signal training data (CMA-ES)")
        print(f"  Rates: {len(args.rates)}")
        print(f"  Signal types: 20 per rate")
        print(f"  CMA-ES: {args.generations} gen x {args.population} pop")
        print(f"  Workers: {args.workers}")
        print("=" * 60)

        for i, rate in enumerate(args.rates):
            out_file = os.path.join(args.output_dir,
                                    f'training_data_{rate}.npz')
            print(f"\n[{i+1}/{len(args.rates)}] {rate.upper()}")
            sys.stdout.flush()

            cmd = [
                PYTHON, '-u', GENERATE_SCRIPT,
                '--rate', rate,
                '--taps', str(args.taps),
                '--generations', str(args.generations),
                '--population', str(args.population),
                '--workers', str(args.workers),
                '--output', out_file,
            ]
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"  WARNING: {rate} failed (rc={rc}), skipping")

    # ── Stage 2: Combine all rate data ──
    print("\n" + "=" * 60)
    print("STAGE 2: Combine training data")
    print("=" * 60)

    all_features = []
    all_taps = []
    all_names = []
    all_rates_hz = []

    for rate in args.rates:
        path = os.path.join(args.output_dir, f'training_data_{rate}.npz')
        if not os.path.exists(path):
            print(f"  SKIP {rate} (no data at {path})")
            continue
        d = np.load(path)
        n = len(d['features'])
        all_features.append(d['features'])
        all_taps.append(d['taps'])
        all_names.extend([f"{rate}_{name}" for name in d['names']])
        all_rates_hz.extend([int(d['rate'])] * n)
        print(f"  {rate}: {n} examples")

    if not all_features:
        print("ERROR: No training data found")
        sys.exit(1)

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    taps = np.concatenate(all_taps, axis=0).astype(np.float32)
    names = np.array(all_names)
    rates_hz = np.array(all_rates_hz)

    # Add rate as 4th feature (MHz)
    rate_feature = (rates_hz / 1e6).astype(np.float32).reshape(-1, 1)
    features_with_rate = np.column_stack([features, rate_feature])

    combined_path = os.path.join(args.output_dir,
                                  'training_data_all_rates.npz')
    np.savez(combined_path,
             features=features_with_rate,
             taps=taps,
             names=names,
             num_taps=args.taps,
             rate=0)

    print(f"\n  Combined: {len(features)} examples, "
          f"{features_with_rate.shape[1]} features")
    print(f"  Saved to {combined_path}")

    # ── Stage 3: Train MLP + export ONNX ──
    print("\n" + "=" * 60)
    print("STAGE 3: Train MLP + export ONNX")
    print("=" * 60)

    onnx_path = os.path.join(SCRIPT_DIR,
                              'foo_dsd_trellis_preemph_taps.onnx')
    cmd = [
        PYTHON, '-u', TRAIN_SCRIPT,
        '--data', combined_path,
        '--output', onnx_path,
        '--hidden', str(args.hidden),
        '--epochs', str(args.epochs),
    ]
    subprocess.call(cmd)

    dt = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"DONE in {dt/60:.1f} minutes")
    print(f"Training data: {combined_path}")
    print(f"ONNX model:    {onnx_path}")
    print(f"{'=' * 60}")
    print(f"\nDeploy: copy {os.path.basename(onnx_path)} to the "
          f"foo_dsd_trellis component folder")


if __name__ == '__main__':
    main()
