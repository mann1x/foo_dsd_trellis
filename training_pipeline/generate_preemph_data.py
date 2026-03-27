"""
Generate training data for adaptive pre-SDM model.

Runs CMA-ES across diverse signal types to build a dataset of
(signal_features, optimal_taps) pairs. Each signal type gets its
own optimized FIR taps via black-box optimization with the real SDM.

Output: training_pairs.npz with:
  - features: (N, num_features) signal characteristics per example
  - taps: (N, num_taps) optimal FIR taps per example
  - sinads: (N, num_freqs) per-frequency SINAD for each example

Usage:
  python -u generate_preemph_data.py --rate dsd512 --taps 3
"""

import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np

try:
    from cmaes import CMA
except ImportError:
    print("Install: pip install cmaes")
    sys.exit(1)

from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes

TEST_EXE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..',
                             'bin', 'Release', 'x64', 'foo_dsd_trellis_test.exe'))

AFFINITY_MASK = 0xFFFF0000  # cores 8-15

DSD_RATES = {
    'dsd64': 2822400,
    'dsd128': 5644800,
    'dsd256': 11289600,
    'dsd512': 22579200,
}

BOX_TAPS = {'dsd64': 32, 'dsd128': 64, 'dsd256': 64, 'dsd512': 16}


def evaluate_taps(rate_hz, freq_hz, taps):
    """Evaluate FIR taps using real C SDM. Returns SINAD in dB."""
    cmd = [TEST_EXE, '--preemph', str(rate_hz), str(freq_hz)]
    cmd.extend(str(t) for t in taps)
    try:
        kernel32 = ctypes.windll.kernel32
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        handle = kernel32.OpenProcess(0x0200, False, proc.pid)
        if handle:
            kernel32.SetProcessAffinityMask(handle, AFFINITY_MASK)
            kernel32.CloseHandle(handle)
        stdout, _ = proc.communicate(timeout=120)
        if proc.returncode != 0:
            return -999.0
        return float(stdout.decode().strip())
    except:
        try: proc.kill()
        except: pass
        return -999.0


def evaluate_population_parallel(rate_hz, population, freqs, max_workers=24):
    """Evaluate entire population in parallel."""
    jobs = []
    for ci, taps in enumerate(population):
        for fi, freq in enumerate(freqs):
            jobs.append((ci, fi, freq, taps))

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for ci, fi, freq, taps in jobs:
            future = executor.submit(evaluate_taps, rate_hz, freq, taps)
            futures[future] = (ci, fi)
        for future in as_completed(futures):
            ci, fi = futures[future]
            results[(ci, fi)] = future.result()

    out = []
    for ci in range(len(population)):
        per_freq = [results.get((ci, fi), -999.0) for fi in range(len(freqs))]
        valid = [r for r in per_freq if r > -900]
        if valid:
            out.append((float(np.median(valid)), min(valid), per_freq))
        else:
            out.append((-999.0, -999.0, per_freq))
    return out


def cmaes_for_signal(rate_hz, test_freqs, num_taps=3, generations=20, population=10):
    """Run CMA-ES for a specific set of test frequencies."""
    x0 = np.zeros(num_taps)
    x0[0] = 1.0

    optimizer = CMA(mean=x0, sigma=0.01, population_size=population)
    best_fitness = -999.0
    best_taps = x0.copy()

    for gen in range(generations):
        pop = [optimizer.ask() for _ in range(optimizer.population_size)]
        eval_results = evaluate_population_parallel(rate_hz, pop, test_freqs)

        solutions = []
        for i, (median, minimum, per_freq) in enumerate(eval_results):
            fitness = minimum  # worst-case optimization
            solutions.append((pop[i], -fitness))
            if fitness > best_fitness:
                best_fitness = fitness
                best_taps = pop[i].copy()

        optimizer.tell(solutions)

    return best_taps, best_fitness


# Signal configurations: each defines a set of test frequencies and features
SIGNAL_CONFIGS = [
    # Single frequencies at different points in the spectrum
    {'name': 'bass_50hz', 'freqs': [50], 'features': [50, 0.5, 1.0]},
    {'name': 'bass_100hz', 'freqs': [100], 'features': [100, 0.5, 1.0]},
    {'name': 'bass_200hz', 'freqs': [200], 'features': [200, 0.5, 1.0]},
    {'name': 'mid_500hz', 'freqs': [500], 'features': [500, 0.5, 1.0]},
    {'name': 'mid_1khz', 'freqs': [1000], 'features': [1000, 0.5, 1.0]},
    {'name': 'mid_2khz', 'freqs': [2000], 'features': [2000, 0.5, 1.0]},
    {'name': 'mid_3khz', 'freqs': [3000], 'features': [3000, 0.5, 1.0]},
    {'name': 'mid_5khz', 'freqs': [5000], 'features': [5000, 0.5, 1.0]},
    {'name': 'hi_7khz', 'freqs': [7000], 'features': [7000, 0.5, 1.0]},
    {'name': 'hi_10khz', 'freqs': [10000], 'features': [10000, 0.5, 1.0]},
    {'name': 'hi_12khz', 'freqs': [12000], 'features': [12000, 0.5, 1.0]},
    {'name': 'hi_14khz', 'freqs': [14000], 'features': [14000, 0.5, 1.0]},
    # Multi-frequency (simulating music content at different spectral centroids)
    {'name': 'bass_heavy', 'freqs': [50, 100, 200, 500], 'features': [200, 0.5, 0.5]},
    {'name': 'mid_heavy', 'freqs': [500, 1000, 2000, 3000], 'features': [1500, 0.5, 0.5]},
    {'name': 'hi_heavy', 'freqs': [5000, 7000, 10000, 14000], 'features': [9000, 0.5, 0.5]},
    {'name': 'full_range', 'freqs': [100, 300, 1000, 3000, 7000, 10000, 14000],
     'features': [5000, 0.5, 0.3]},
    # Different amplitudes (low level = more noise relative to signal)
    {'name': '1khz_quiet', 'freqs': [1000], 'features': [1000, 0.1, 1.0]},
    {'name': '1khz_loud', 'freqs': [1000], 'features': [1000, 0.9, 1.0]},
    {'name': '10khz_quiet', 'freqs': [10000], 'features': [10000, 0.1, 1.0]},
    {'name': '10khz_loud', 'freqs': [10000], 'features': [10000, 0.9, 1.0]},
]


def main():
    parser = argparse.ArgumentParser(description='Generate pre-SDM training data')
    parser.add_argument('--rate', default='dsd512',
                        choices=['dsd64', 'dsd128', 'dsd256', 'dsd512'])
    parser.add_argument('--taps', type=int, default=3)
    parser.add_argument('--generations', type=int, default=15,
                        help='CMA-ES generations per signal type')
    parser.add_argument('--population', type=int, default=10)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    rate_hz = DSD_RATES[args.rate]
    if args.output is None:
        args.output = f'I:/foo_dsd_trellis/checkpoints/cmaes/training_data_{args.rate}.npz'

    print(f"=== Generating Pre-SDM Training Data ===")
    print(f"Rate: {args.rate.upper()} ({rate_hz} Hz)")
    print(f"Taps: {args.taps}")
    print(f"Signal configs: {len(SIGNAL_CONFIGS)}")
    print(f"CMA-ES: {args.generations} gen x {args.population} pop per config")
    print()

    all_features = []
    all_taps = []
    all_names = []

    for i, cfg in enumerate(SIGNAL_CONFIGS):
        t0 = time.time()
        print(f"  [{i+1}/{len(SIGNAL_CONFIGS)}] {cfg['name']} "
              f"(freqs={cfg['freqs']})...", end=' ', flush=True)

        taps, fitness = cmaes_for_signal(
            rate_hz, cfg['freqs'], args.taps, args.generations, args.population)

        dt = time.time() - t0
        print(f"fitness={fitness:.1f} dB, taps=[{', '.join(f'{t:.4f}' for t in taps)}] "
              f"({dt:.0f}s)")

        all_features.append(cfg['features'])
        all_taps.append(taps)
        all_names.append(cfg['name'])

    # Save training data
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output,
             features=np.array(all_features),
             taps=np.array(all_taps),
             names=np.array(all_names),
             rate=rate_hz,
             num_taps=args.taps)

    print(f"\nSaved {len(all_features)} examples to {args.output}")

    # Print summary
    taps_arr = np.array(all_taps)
    print(f"\nTap statistics:")
    for t in range(args.taps):
        print(f"  tap[{t}]: mean={taps_arr[:,t].mean():.4f} "
              f"std={taps_arr[:,t].std():.4f} "
              f"range=[{taps_arr[:,t].min():.4f}, {taps_arr[:,t].max():.4f}]")


if __name__ == '__main__':
    main()
