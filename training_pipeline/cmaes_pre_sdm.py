"""
CMA-ES optimizer for pre-SDM pre-emphasis using the real C trellis SDM.

Uses foo_dsd_trellis_test.exe --preemph as a black-box fitness evaluator.
Optimizes FIR coefficients to maximize worst-case SINAD across diverse
test frequencies (robust optimization, not single-frequency lucky).

Usage:
  python -u cmaes_pre_sdm.py --rate dsd512 --taps 3 --generations 30
  python -u cmaes_pre_sdm.py --rate dsd512 --taps 7 --generations 50

Requires: pip install cmaes
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
    print("Install cmaes: pip install cmaes")
    sys.exit(1)


TEST_EXE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..',
                             'bin', 'Release', 'x64', 'foo_dsd_trellis_test.exe'))

DSD_RATES = {
    'dsd64': 2822400,
    'dsd128': 5644800,
    'dsd256': 11289600,
    'dsd512': 22579200,
}

# Test frequencies for robust evaluation (diverse content simulation)
TEST_FREQS = [100, 300, 1000, 3000, 7000, 10000, 14000]


# Affinity mask: skip first 8 cores (LPs 0-15), use cores 8-15 (LPs 16-31)
# On 16-core/32-thread: 0xFFFF0000 = LPs 16-31
AFFINITY_MASK = 0xFFFF0000


def evaluate_taps(rate_hz, freq_hz, taps):
    """Evaluate pre-emphasis FIR taps using the real C trellis SDM.

    Returns SINAD in dB, or -999 on error.
    """
    cmd = [TEST_EXE, '--preemph', str(rate_hz), str(freq_hz)]
    cmd.extend(str(t) for t in taps)

    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        # Set affinity to cores 8-15 (LPs 16-31)
        handle = kernel32.OpenProcess(0x0200, False, proc.pid)  # PROCESS_SET_INFORMATION
        if handle:
            kernel32.SetProcessAffinityMask(handle, AFFINITY_MASK)
            kernel32.CloseHandle(handle)

        stdout, _ = proc.communicate(timeout=120)
        if proc.returncode != 0:
            return -999.0
        return float(stdout.decode().strip())
    except (subprocess.TimeoutExpired, ValueError, Exception):
        try:
            proc.kill()
        except:
            pass
        return -999.0


def evaluate_robust(rate_hz, taps, freqs=TEST_FREQS):
    """Evaluate taps across multiple frequencies (sequential).

    Returns: (median_sinad, min_sinad, per_freq_results)
    """
    results = []
    for f in freqs:
        s = evaluate_taps(rate_hz, f, taps)
        results.append(s)

    valid = [r for r in results if r > -900]
    if not valid:
        return -999.0, -999.0, results

    median = float(np.median(valid))
    minimum = min(valid)
    return median, minimum, results


def evaluate_population_parallel(rate_hz, population, freqs=TEST_FREQS, max_workers=16):
    """Evaluate entire CMA-ES population in parallel.

    Launches all (candidate × frequency) evaluations concurrently.
    Returns: list of (median, min, per_freq) per candidate.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Build all jobs: (candidate_idx, freq, taps)
    jobs = []
    for ci, taps in enumerate(population):
        for fi, freq in enumerate(freqs):
            jobs.append((ci, fi, freq, taps))

    results = {}  # (ci, fi) → sinad

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for ci, fi, freq, taps in jobs:
            future = executor.submit(evaluate_taps, rate_hz, freq, taps)
            futures[future] = (ci, fi)

        for future in as_completed(futures):
            ci, fi = futures[future]
            results[(ci, fi)] = future.result()

    # Assemble per-candidate results
    out = []
    for ci in range(len(population)):
        per_freq = [results.get((ci, fi), -999.0) for fi in range(len(freqs))]
        valid = [r for r in per_freq if r > -900]
        if valid:
            out.append((float(np.median(valid)), min(valid), per_freq))
        else:
            out.append((-999.0, -999.0, per_freq))

    return out


def cmaes_optimize(args):
    rate_hz = DSD_RATES[args.rate]
    num_taps = args.taps

    print(f"=== CMA-ES Pre-SDM Optimization ===")
    print(f"Rate: {args.rate.upper()} ({rate_hz} Hz)")
    print(f"FIR taps: {num_taps}")
    print(f"Test frequencies: {TEST_FREQS}")
    print(f"Fitness: {'worst-case' if args.robust else 'median'} SINAD")
    print(f"Generations: {args.generations}")
    print(f"Population: {args.population}")
    print()

    # Baseline: identity (no pre-emphasis)
    identity = np.zeros(num_taps)
    identity[0] = 1.0
    base_med, base_min, base_results = evaluate_robust(rate_hz, identity)
    print(f"Baseline (identity): median={base_med:.1f} dB, min={base_min:.1f} dB")
    print(f"  Per-freq: {['%.1f' % r for r in base_results]}")
    print()

    # CMA-ES setup
    x0 = identity.copy()
    sigma0 = args.sigma

    optimizer = CMA(
        mean=x0,
        sigma=sigma0,
        population_size=args.population,
    )

    best_fitness = -999.0
    best_taps = x0.copy()
    best_med = base_med
    best_min = base_min

    for gen in range(args.generations):
        t0 = time.time()

        # Ask for all candidates
        population = [optimizer.ask() for _ in range(optimizer.population_size)]

        # Evaluate entire population in parallel
        eval_results = evaluate_population_parallel(
            rate_hz, population, TEST_FREQS, max_workers=args.workers)

        # Build solutions for CMA-ES
        solutions = []
        for i, (median, minimum, per_freq) in enumerate(eval_results):
            fitness = minimum if args.robust else median
            solutions.append((population[i], -fitness))

            if fitness > best_fitness:
                best_fitness = fitness
                best_taps = population[i].copy()
                best_med = median
                best_min = minimum

        optimizer.tell(solutions)
        dt = time.time() - t0

        if gen % args.log_interval == 0 or gen == args.generations - 1:
            # Evaluate current best
            med, mn, pf = evaluate_robust(rate_hz, best_taps)
            taps_str = ', '.join(f'{t:.6f}' for t in best_taps)
            print(f"  Gen {gen:3d}: fitness={best_fitness:.1f} dB "
                  f"(med={med:.1f} min={mn:.1f}) "
                  f"({dt:.1f}s/gen)")
            print(f"           taps=[{taps_str}]")
            sys.stdout.flush()

    # Final evaluation
    print(f"\n=== Final Result ===")
    final_med, final_min, final_pf = evaluate_robust(rate_hz, best_taps)
    print(f"Best taps: {best_taps.tolist()}")
    print(f"Median SINAD: {final_med:.1f} dB (baseline: {base_med:.1f}, delta: {final_med-base_med:+.1f})")
    print(f"Min SINAD:    {final_min:.1f} dB (baseline: {base_min:.1f}, delta: {final_min-base_min:+.1f})")
    print(f"Per-frequency:")
    for i, f in enumerate(TEST_FREQS):
        delta = final_pf[i] - base_results[i]
        print(f"  {f:6d} Hz: {final_pf[i]:7.1f} dB (baseline: {base_results[i]:.1f}, delta: {delta:+.1f})")

    # Save
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'best_taps_{args.rate}_{num_taps}tap.npz')
    np.savez(out_file,
             taps=best_taps,
             rate=rate_hz,
             baseline_med=base_med,
             baseline_min=base_min,
             final_med=final_med,
             final_min=final_min,
             test_freqs=TEST_FREQS,
             baseline_per_freq=base_results,
             final_per_freq=final_pf)
    print(f"\nSaved to {out_file}")

    return best_taps


def main():
    parser = argparse.ArgumentParser(description='CMA-ES pre-SDM FIR optimization')
    parser.add_argument('--rate', default='dsd512',
                        choices=['dsd64', 'dsd128', 'dsd256', 'dsd512'])
    parser.add_argument('--taps', type=int, default=3,
                        help='Number of FIR taps (3=minimal, 7=standard)')
    parser.add_argument('--generations', type=int, default=30)
    parser.add_argument('--population', type=int, default=8)
    parser.add_argument('--sigma', type=float, default=0.005,
                        help='Initial step size for CMA-ES')
    parser.add_argument('--robust', action='store_true', default=True,
                        help='Optimize worst-case (min) instead of median')
    parser.add_argument('--log-interval', type=int, default=3)
    parser.add_argument('--workers', type=int, default=16,
                        help='Parallel subprocess workers for evaluation')
    parser.add_argument('--output-dir', default='I:/foo_dsd_trellis/checkpoints/cmaes')
    args = parser.parse_args()
    cmaes_optimize(args)


if __name__ == '__main__':
    main()
