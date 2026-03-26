#!/usr/bin/env python3
"""
NTF coefficient generator for foo_dsd_trellis CRFB format.
Ported from python-deltasigma realizeNTF algorithm.

Usage:
  python gen_ntf_pydelsig.py              # Generate and validate
  python gen_ntf_pydelsig.py --roundtrip  # Round-trip validation with existing filters
"""
import numpy as np
from scipy import signal
import sys

def cplxpair(x, tol=1e-6):
    """Sort: real first, then conjugate pairs (ascending |imag|)."""
    x = np.asarray(x, dtype=complex)
    reals = x[np.abs(np.imag(x)) <= tol * np.abs(x)]
    cx = x[np.abs(np.imag(x)) > tol * np.abs(x)]
    reals = np.sort(reals.real).astype(complex)
    cx_sorted = []
    remaining = list(cx)
    while remaining:
        z = remaining.pop(0)
        dists = [abs(w - z.conjugate()) for w in remaining]
        pi = int(np.argmin(dists))
        pair = [z, remaining.pop(pi)]
        pair.sort(key=lambda w: np.imag(w))
        cx_sorted.extend(pair)
    return np.concatenate([reals, np.array(cx_sorted)])

def evalRPoly(roots, z):
    y = complex(1.0)
    for r in roots:
        if not np.isinf(r):
            y *= (z - r)
    return y

def realizeNTF_CRFB(ntf_zeros, ntf_poles):
    """
    Convert NTF zeros/poles to CRFB a[] and g[] in our convention.

    Returns:
      a[order]  - feedback coefficients (low-freq pair first)
      g[order]  - resonator gains, zero-interleaved (g[2k] = pair_k, g[2k+1] = 0)
    """
    ntf_z = cplxpair(np.asarray(ntf_zeros, dtype=complex))[::-1]
    ntf_p = np.asarray(ntf_poles, dtype=complex)
    order = len(ntf_p)
    odd = order % 2

    # g[] from zero pair angles (using cos(angle), not Re(z) — zeros may be off unit circle)
    g_pairs = []
    for i in range(order // 2):
        z = ntf_z[2*i + odd]
        theta = np.abs(np.angle(z))
        g_pairs.append(2.0 * (1.0 - np.cos(theta)))

    # Build evaluation points on r=1.1 circle, avoiding NTF zeros
    N = 200
    zSet = []
    for i in range(1, N + 1):
        z = 1.1 * np.exp(2j * np.pi * i / N)
        if all(abs(ntf_z - z) > 0.09):
            zSet.append(z)
    zSet = zSet[:2 * order]
    assert len(zSet) >= 2 * order, f"Need {2*order} eval points, got {len(zSet)}"

    # Target loop filter and basis matrix
    L1 = np.zeros(2 * order, dtype=complex)
    T = np.zeros((order, 2 * order), dtype=complex)
    for i, z in enumerate(zSet):
        L1[i] = 1.0 - evalRPoly(ntf_p, z) / evalRPoly(ntf_z, z)
        Dfactor = (z - 1.0) / z
        product = 1.0
        j = order
        while j > odd:
            product = z / evalRPoly(ntf_z[j-2:j], z) * product
            T[j-1, i] = product * Dfactor
            T[j-2, i] = product
            j -= 2
        if odd:
            T[0, i] = product / (z - 1.0)

    a_raw, _, _, _ = np.linalg.lstsq(T.T, L1, rcond=None)
    a_raw = -np.real(a_raw)

    # REVERSE a[] and g_pairs[] to match our convention (low-freq pair first)
    a = a_raw[::-1]
    g_pairs_rev = g_pairs[::-1]
    g_full = np.zeros(order)
    for k in range(len(g_pairs_rev)):
        g_full[2*k] = g_pairs_rev[k]

    return a, g_full

def extract_ntf_poles(a, g, order):
    """
    Extract NTF poles from existing CRFB coefficients via polynomial fit.
    Evaluates the loop filter H(z) numerically, computes NTF = 1/(1+H),
    then fits zeros/poles to the NTF.
    """
    A = np.eye(order)
    for i in range(order - 1):
        A[i + 1, i] = 1.0
        A[i, i + 1] = -g[i]

    # Sample NTF at multiple points
    n_pts = 3 * order
    w_pts = np.linspace(0.01, np.pi - 0.01, n_pts)
    z_pts = np.exp(1j * w_pts)
    ntf_vals = np.zeros(n_pts, dtype=complex)
    for k in range(n_pts):
        G = np.linalg.solve(z_pts[k] * np.eye(order) - A, np.ones(order))
        H = np.dot(a, G)
        ntf_vals[k] = 1.0 / (1.0 + H)

    # Fit NTF = num(z)/den(z), both degree order
    V = np.vander(z_pts, order + 1, increasing=True)
    M = np.zeros((n_pts, 2*order + 1), dtype=complex)
    M[:, :order+1] = V
    for k in range(n_pts):
        M[k, order+1:] = -ntf_vals[k] * V[k, :order]
    rhs = ntf_vals * z_pts**order

    coeff, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
    den_c = np.append(coeff[order+1:], 1.0)
    poles = np.roots(den_c[::-1])
    return poles

def extract_ntf_zeros(g, order):
    """Extract NTF zeros from g[] values."""
    zeros = []
    for i in range(0, order, 2):
        if g[i] > 0:
            theta = np.arccos(1.0 - g[i] / 2.0)
            zeros.append(np.exp(1j * theta))
            zeros.append(np.exp(-1j * theta))
        else:
            zeros.append(1.0 + 0j)
            zeros.append(1.0 + 0j)
    return np.array(zeros[:order])

def format_c_struct(a, g, order, rate_hz, name):
    lines = [f'  /* {name} @ {rate_hz // 44100}x44100 */']
    lines.append('  {')
    lines.append('    {')
    for v in a: lines.append(f'      {v:.17e},')
    lines.append('    },')
    lines.append('    {')
    for v in g: lines.append(f'      {v:.17e},')
    lines.append('    },')
    lines.append(f'    {order}, {rate_hz}, "{name}"')
    lines.append('  },')
    return '\n'.join(lines)

# ─── Existing filters for validation ───
EXISTING = {
    'clans-4 DSD64': {
        'a': [1.01153638671875, 0.37469482421875, 0.066162109375, 0.005279541015625],
        'g': [0.003073931718223, 0, 0.016561398338484, 0],
        'order': 4, 'rate': 64*44100
    },
    'clans-6 DSD64': {
        'a': [9.98107910156250e-01, 3.51013183593750e-01, 6.01196289062500e-02,
              5.47790527343750e-03, 2.59399414062500e-04, 5.00679016113281e-06],
        'g': [3.07369532217957e-03, 0, 1.65613983384842e-02, 0, 2.40414488117409e-02, 0],
        'order': 6, 'rate': 64*44100
    },
    'clans-6 DSD128': {
        'a': [1.00118112564087e+00, 3.49838972091675e-01, 5.87449073791504e-02,
              5.14984130859375e-03, 2.30312347412109e-04, 4.13060188293457e-06],
        'g': [7.65808490118989e-04, 0, 4.16694756269693e-03, 0, 6.13696953145345e-03, 0],
        'order': 6, 'rate': 128*44100
    },
}

if __name__ == '__main__':
    for name, ex in EXISTING.items():
        print(f'\n{"="*60}')
        print(f'Round-trip: {name}')
        print(f'{"="*60}')

        a_ex = np.array(ex['a'])
        g_ex = np.array(ex['g'])
        order = ex['order']

        # Extract zeros and poles
        zeros = extract_ntf_zeros(g_ex, order)
        poles = extract_ntf_poles(a_ex, g_ex, order)
        print(f'Extracted poles: {poles}')

        # Round-trip through realizeNTF_CRFB
        a_rt, g_rt = realizeNTF_CRFB(zeros, poles)

        print(f'\n  {"":>6} {"round-trip":>22} {"existing":>22} {"ratio":>10}')
        for i in range(order):
            r = a_rt[i] / a_ex[i] if a_ex[i] != 0 else 0
            print(f'  a[{i}] {a_rt[i]:>22.15e} {a_ex[i]:>22.15e} {r:>10.6f}')
        for i in range(order):
            eg = g_ex[i]
            r = g_rt[i] / eg if eg != 0 else 0
            print(f'  g[{i}] {g_rt[i]:>22.15e} {eg:>22.15e} {r:>10.6f}')
