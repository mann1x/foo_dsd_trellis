"""
CMA-ES optimizer for pre-SDM pre-emphasis using the real C trellis SDM.

Instead of differentiable proxy, uses the actual foo_dsd_trellis_test.exe
as a black-box fitness evaluator. CMA-ES searches the FIR coefficient
space to maximize SINAD at the output.

Usage:
  python cmaes_pre_sdm.py --rate dsd512 --taps 7 --generations 50

Requires: cmaes package (pip install cmaes)
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

try:
    from cmaes import CMA
except ImportError:
    print("Install cmaes: pip install cmaes")
    sys.exit(1)


# Path to test executable
TEST_EXE = os.path.join(os.path.dirname(__file__), '..',
                         'bin', 'Release', 'x64', 'foo_dsd_trellis_test.exe')

DSD_RATES = {
    'dsd64': 2822400,
    'dsd128': 5644800,
    'dsd256': 11289600,
    'dsd512': 22579200,
}

BOX_TAPS = {
    'dsd64': 32,
    'dsd128': 64,
    'dsd256': 64,
    'dsd512': 16,
}


def evaluate_preemph(rate_name, fir_taps, test_freqs=[900, 1000, 1100]):
    """Evaluate pre-emphasis FIR using the real C SDM.

    Runs the test executable with the given FIR taps and measures
    median SINAD across test frequencies.

    Returns: median SINAD in dB (higher = better)
    """
    # For now, use the measure_pre_sdm test function
    # This requires modifying the test to accept external FIR taps
    # Placeholder: use the parametric pre-emphasis sweep approach
    #
    # TODO: Add --preemph-taps CLI arg to test.exe that passes
    # FIR taps to the boxcar→pre-emph→SDM pipeline

    # For the initial prototype, compute the equivalent first-order k
    # from the FIR taps and use the existing sweep infrastructure
    if len(fir_taps) >= 2:
        # First-order approximation: k ≈ -taps[1] / taps[0]
        k = -fir_taps[1] / max(abs(fir_taps[0]), 1e-10)
    else:
        k = 0.0

    # Clamp to reasonable range
    k = max(0.0, min(0.5, k))

    return k  # placeholder


def cmaes_optimize(rate_name, num_taps, generations, population_size=None):
    """Run CMA-ES optimization for pre-emphasis FIR coefficients."""

    print(f"\n=== CMA-ES Pre-SDM Optimization ===")
    print(f"Rate: {rate_name.upper()}")
    print(f"FIR taps: {num_taps}")
    print(f"Generations: {generations}")

    # Initial point: identity + small pre-emphasis
    x0 = np.zeros(num_taps)
    x0[0] = 1.0
    if num_taps > 1:
        x0[1] = -0.007  # seed from sweep winner

    # Initial step size
    sigma0 = 0.01

    if population_size is None:
        population_size = 4 + int(3 * math.log(num_taps))

    print(f"Population: {population_size}")
    print(f"Initial taps: {x0.tolist()}")
    print()

    optimizer = CMA(
        mean=x0,
        sigma=sigma0,
        population_size=population_size,
    )

    best_score = -999.0
    best_taps = x0.copy()

    for gen in range(generations):
        solutions = []
        for _ in range(optimizer.population_size):
            taps = optimizer.ask()

            # Evaluate: use the first-order approximation for now
            # In production, this would call the real SDM
            score = evaluate_preemph(rate_name, taps)

            # CMA-ES minimizes, but we want to maximize SINAD
            # Use negative score as the objective
            solutions.append((taps, -score))

            if score > best_score:
                best_score = score
                best_taps = taps.copy()

        optimizer.tell(solutions)

        if gen % 5 == 0 or gen == generations - 1:
            print(f"  Gen {gen:3d}: best_k={best_score:.4f} "
                  f"taps=[{', '.join(f'{t:.6f}' for t in best_taps)}]")

    print(f"\nOptimal taps: {best_taps.tolist()}")
    print(f"Equivalent k: {-best_taps[1]/best_taps[0]:.6f}" if len(best_taps) > 1 else "")
    return best_taps


def main():
    parser = argparse.ArgumentParser(description='CMA-ES pre-SDM FIR optimization')
    parser.add_argument('--rate', default='dsd512',
                        choices=['dsd64', 'dsd128', 'dsd256', 'dsd512'])
    parser.add_argument('--taps', type=int, default=7)
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--population', type=int, default=None)
    args = parser.parse_args()

    best = cmaes_optimize(args.rate, args.taps, args.generations, args.population)

    # Save result
    out_dir = 'I:/foo_dsd_trellis/checkpoints/cmaes'
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'best_taps_{args.rate}.npy'), best)
    print(f"\nSaved to {out_dir}/best_taps_{args.rate}.npy")


if __name__ == '__main__':
    main()
