"""
Launch CMA-ES training for all rate × SDM combinations.

Runs sequentially (each run uses parallel workers internally).
8 rates × 2 SDM types = 16 runs.

Usage:
  python -u run_all_cmaes.py
  python -u run_all_cmaes.py --generations 50
  python -u run_all_cmaes.py --rates dsd256 dsd512 --sdm trellis
"""

import argparse
import os
import subprocess
import sys
import time

PYTHON = sys.executable
SCRIPT = os.path.join(os.path.dirname(__file__), 'cmaes_pre_sdm.py')

ALL_RATES = [
    'dsd64', 'dsd128', 'dsd256', 'dsd512',
    'dsd64_48', 'dsd128_48', 'dsd256_48', 'dsd512_48',
]
ALL_SDMS = ['trellis']  # PreCorr shows 0 improvement from pre-emphasis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rates', nargs='+', default=ALL_RATES,
                        help='Rates to train (default: all)')
    parser.add_argument('--sdm', nargs='+', default=ALL_SDMS,
                        choices=ALL_SDMS, help='SDM types (default: both)')
    parser.add_argument('--generations', type=int, default=30)
    parser.add_argument('--population', type=int, default=8)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--taps', type=int, default=3)
    parser.add_argument('--metric', default='awtd', choices=['sinad', 'awtd', 'mt'],
                        help='Fitness metric (default: awtd)')
    args = parser.parse_args()

    runs = [(rate, sdm) for rate in args.rates for sdm in args.sdm]
    total = len(runs)

    print(f"=== CMA-ES Batch Training ===")
    print(f"Runs: {total} ({len(args.rates)} rates x {len(args.sdm)} SDMs)")
    print(f"Generations: {args.generations}, Population: {args.population}")
    print(f"Workers: {args.workers}, Taps: {args.taps}")
    print()

    results = {}
    t_total = time.time()

    for i, (rate, sdm) in enumerate(runs):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {rate.upper()} / {sdm.capitalize()}")
        print(f"{'='*60}")
        sys.stdout.flush()

        t0 = time.time()
        cmd = [
            PYTHON, '-u', SCRIPT,
            '--rate', rate,
            '--sdm', sdm,
            '--taps', str(args.taps),
            '--generations', str(args.generations),
            '--population', str(args.population),
            '--workers', str(args.workers),
            '--metric', args.metric,
        ]

        rc = subprocess.call(cmd)
        dt = time.time() - t0

        results[(rate, sdm)] = {
            'time': dt,
            'success': rc == 0,
        }
        print(f"\n[{rate}/{sdm}] {'OK' if rc == 0 else 'FAILED'} in {dt:.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"=== Summary ({time.time() - t_total:.0f}s total) ===")
    print(f"{'='*60}")
    for (rate, sdm), info in results.items():
        status = 'OK' if info['success'] else 'FAILED'
        print(f"  {rate:12s} {sdm:8s}: {status} ({info['time']:.0f}s)")


if __name__ == '__main__':
    main()
