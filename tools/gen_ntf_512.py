#!/usr/bin/env python3
"""
Generate DSD512 (512x44100) NTF filter coefficients for foo_dsd_trellis.

Uses analytical linear-algebra approach:
  1. Compute NTF as ratio of polynomials in w = z^{-1}
  2. Extract NTF poles from existing 256x44100 filter denominator
  3. For DSD512: compute new numerator from scaled g[], keep same poles
  4. Solve LINEAR system for a[] coefficients (no iterative optimization)
"""

import numpy as np
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MAX_ORDER = 8


# --- Polynomial helpers (ascending power of w = z^{-1}) ---------------

def pmul(a, b):
    return np.convolve(a, b)

def padd(a, b):
    n = max(len(a), len(b))
    r = np.zeros(n)
    r[:len(a)] += a
    r[:len(b)] += b
    return r

def peval(poly, w):
    """Evaluate polynomial at complex array w using Horner's method."""
    n = len(poly)
    if n == 0:
        return np.zeros_like(w, dtype=complex)
    result = np.full_like(w, poly[-1], dtype=complex)
    for i in range(n - 2, -1, -1):
        result = result * w + poly[i]
    return result


# --- CRFB NTF polynomial computation ----------------------------------

def crfb_polys(g, order):
    """Compute det(M) and backward determinants b[j] for CRFB topology.

    Returns (num_poly, b_list) where:
      num_poly = det(M) as polynomial in w
      b_list[j] = det of trailing submatrix M[j:,j:], j=0..order
    """
    d = np.array([1.0, -1.0])   # 1 - w
    w2 = np.array([0.0, 0.0, 1.0])  # w^2

    # Forward recurrence for det(M)
    f_prev2 = np.array([1.0])
    f_prev1 = d.copy()
    for i in range(1, order):
        gi = g[i - 1] if i - 1 < len(g) else 0.0
        f_i = pmul(d, f_prev1)
        if abs(gi) > 1e-30:
            f_i = padd(f_i, gi * pmul(w2, f_prev2))
        f_prev2 = f_prev1
        f_prev1 = f_i
    num_poly = f_prev1

    # Backward recurrence for b[j] = det(M[j:,j:])
    b = [None] * (order + 1)
    b[order] = np.array([1.0])
    if order >= 1:
        b[order - 1] = d.copy()
    for i in range(order - 2, -1, -1):
        gi = g[i] if i < len(g) else 0.0
        b[i] = pmul(d, b[i + 1])
        if abs(gi) > 1e-30:
            b[i] = padd(b[i], gi * pmul(w2, b[i + 2]))

    return num_poly, b


def crfb_den(a, g, order):
    """Compute NTF denominator polynomial: det(M) + sum_j a[j]*w^j*b[j+1]."""
    num, b = crfb_polys(g, order)
    den = num.copy()
    for j in range(order):
        if abs(a[j]) < 1e-30:
            continue
        wj = np.zeros(j + 1)
        wj[j] = 1.0
        den = padd(den, a[j] * pmul(wj, b[j + 1]))
    return den


def ntf_eval(a, g, order, omega):
    """Evaluate |NTF(e^{jw})|^2 at frequency array."""
    num, b = crfb_polys(g, order)
    den = num.copy()
    for j in range(order):
        if abs(a[j]) < 1e-30:
            continue
        wj = np.zeros(j + 1)
        wj[j] = 1.0
        den = padd(den, a[j] * pmul(wj, b[j + 1]))

    w = np.exp(-1j * omega)
    n_vals = peval(num.astype(complex), w)
    d_vals = peval(den.astype(complex), w)
    d_vals = np.where(np.abs(d_vals) < 1e-30, 1e-30, d_vals)
    return np.abs(n_vals / d_vals) ** 2


# --- Metrics -----------------------------------------------------------

N_FREQ = 4096
OMEGA = np.linspace(1e-6, np.pi, N_FREQ)

def hinf(a, g, order):
    return np.sqrt(np.max(ntf_eval(a, g, order, OMEGA)))

def ibn(a, g, order, osr):
    resp = ntf_eval(a, g, order, OMEGA)
    idx = max(2, np.searchsorted(OMEGA, np.pi / osr))
    return np.trapezoid(resp[:idx], OMEGA[:idx])


# --- Solve for a[] given desired poles ---------------------------------

def solve_a_from_poles(g_new, order, poles_w):
    """Given CRFB numerator (from g[]) and desired NTF poles in w-domain,
    solve the linear system for a[] coefficients.

    den(w) = num(w) + sum_j a[j] * w^j * b[j+1](w)

    We want den(w) to have the given roots (poles_w).
    """
    num, b = crfb_polys(g_new, order)

    # Build desired denominator from poles
    # Using numpy polynomial: coefficients in ascending power order
    den_desired = np.polynomial.polynomial.polyfromroots(poles_w).real

    # Normalize so that leading coefficient matches num
    if len(num) > order:
        lead_num = num[order]
    else:
        lead_num = 0.0
        for i in range(len(num) - 1, -1, -1):
            if abs(num[i]) > 1e-30:
                lead_num = num[i]
                break

    lead_den = den_desired[-1] if len(den_desired) > 0 else 1.0
    if abs(lead_den) > 1e-30:
        den_desired = den_desired * (lead_num / lead_den)

    # Pad to same length
    max_len = max(len(num), len(den_desired))
    num_padded = np.zeros(max_len)
    num_padded[:len(num)] = num
    den_padded = np.zeros(max_len)
    den_padded[:len(den_desired)] = den_desired

    # RHS = den_desired - num (should be degree order-1 at most)
    rhs = den_padded - num_padded

    # Build matrix: Phi[k, j] = coeff of w^k in (w^j * b[j+1])
    # w^j * b[j+1] has degree j + deg(b[j+1]) = j + (order-j-1) = order-1
    Phi = np.zeros((order, order))
    for j in range(order):
        bj = b[j + 1]
        for m in range(len(bj)):
            k = j + m  # power of w
            if k < order:
                Phi[k, j] = bj[m]

    # Solve the linear system
    a_new = np.linalg.solve(Phi, rhs[:order])
    return list(a_new)


# --- Existing filters --------------------------------------------------

FILTERS_256 = [
    ("clans-4", 4,
     [1.00323940832478e+00, 3.54975562370606e-01, 5.64754047673194e-02, 3.99067228430322e-03],
     [1.74071110561285e-05, 0, 1.11672812199443e-04, 0]),
    ("sdm-4", 4,
     [8.69746397840960e-01, 3.58080546314756e-01, 8.02654082306273e-02, 8.06528716282692e-03],
     [1.74071110561285e-05, 0, 1.11672812199443e-04, 0]),
    ("clans-5", 5,
     [1.10212073518628e+00, 4.33447134954244e-01, 7.17865111532609e-02, 4.48367825425951e-03, 8.60861641068938e-05],
     [0, 4.36651951230006e-05, 0, 1.23660417994961e-04, 0]),
    ("sdm-5", 5,
     [8.07768375734983e-01, 3.16440095967511e-01, 7.38231738259889e-02, 1.01432044963374e-02, 6.46658652275506e-04],
     [0, 4.36651951230006e-05, 0, 1.23660417994961e-04, 0]),
    ("clans-6", 6,
     [9.97000121097967e-01, 3.46002867430604e-01, 5.74352078895161e-02, 4.96197900435677e-03, 2.16319301330580e-04, 3.45938007947910e-06],
     [8.57500543083848e-06, 0, 6.58398680532347e-05, 0, 1.30939362595793e-04, 0]),
    ("sdm-6", 6,
     [8.08851952379691e-01, 3.20414766828429e-01, 7.85858596284593e-02, 1.24781319607895e-02, 1.21202847406105e-03, 5.51622876557856e-05],
     [8.57500543083848e-06, 0, 6.58398680532347e-05, 0, 1.30939362595793e-04, 0]),
    ("clans-7", 7,
     [1.10629931445134e+00, 4.22135693734657e-01, 7.54595882135669e-02, 7.07164815703843e-03, 3.53092575577382e-04, 8.89662856104825e-06, 5.79674109824069e-08],
     [0, 2.48046933669715e-05, 0, 8.28068362972358e-05, 0, 1.35653594733585e-04, 0]),
    ("sdm-7", 7,
     [7.82785077952658e-01, 3.01888671316811e-01, 7.36594376027782e-02, 1.22068270909817e-02, 1.36572694403914e-03, 9.57806082134936e-05, 3.13239368043838e-06],
     [0, 2.48046933669715e-05, 0, 8.28068362972358e-05, 0, 1.35653594733585e-04, 0]),
    ("clans-8", 8,
     [1.15188624720851e+00, 5.45054196257555e-01, 1.38703640845632e-01, 2.07076444822072e-02, 1.85506614417771e-03, 9.63403135615390e-05, 2.69174565706992e-06, 2.22594461751768e-08],
     [5.06749566262594e-06, 0, 4.15924517416912e-05, 0, 9.55783346944871e-05, 0, 1.38868728742641e-04, 0]),
    ("sdm-8", 8,
     [7.42329617949054e-01, 2.72509195471757e-01, 6.41424039739473e-02, 1.05299412132258e-02, 1.23178223428228e-03, 9.94985029720342e-05, 5.13169547054423e-06, 1.20466411041020e-07],
     [5.06749566262594e-06, 0, 4.15924517416912e-05, 0, 9.55783346944871e-05, 0, 1.38868728742641e-04, 0]),
]


# --- Cross-validation --------------------------------------------------

def cross_validate():
    """Verify approach: use 128x44100 clans-5 to reconstruct 256x44100 clans-5."""
    print("\n--- Cross-validation: extract poles from 128, reconstruct 256 ---")

    # Known 128x44100 clans-5
    a_128 = [1.12849522129362e+00, 5.02128177800632e-01, 1.10084368682902e-01, 1.18635667860902e-02, 4.71059243536326e-04]
    g_128 = [0, 1.74653153894942e-04, 0, 4.94580504383930e-04, 0]

    # Known 256x44100 clans-5 (target)
    a_256_known = [1.10212073518628e+00, 4.33447134954244e-01, 7.17865111532609e-02, 4.48367825425951e-03, 8.60861641068938e-05]
    g_256_known = [0, 4.36651951230006e-05, 0, 1.23660417994961e-04, 0]

    order = 5

    # Extract NTF poles from 128 filter
    den_128 = crfb_den(a_128, g_128, order)
    poles_128_w = np.roots(den_128[::-1])  # np.roots expects descending order
    print(f"  128 poles (w-domain): {poles_128_w}")
    print(f"  128 pole magnitudes:  {np.abs(poles_128_w)}")
    print(f"  128 Hinf: {hinf(a_128, g_128, order):.6f}")

    # Scale g for 256
    g_256_scaled = [gi / 4.0 for gi in g_128[:order]]
    print(f"  g_256 scaled: {g_256_scaled}")
    print(f"  g_256 known:  {g_256_known}")

    # Reconstruct a_256 using same poles
    a_256_recon = solve_a_from_poles(g_256_scaled, order, poles_128_w)

    h_recon = hinf(a_256_recon, g_256_scaled, order)
    h_known = hinf(a_256_known, g_256_known, order)
    print(f"  Hinf reconstructed: {h_recon:.6f}")
    print(f"  Hinf known 256:     {h_known:.6f}")
    print(f"  a reconstructed: {[f'{x:.8e}' for x in a_256_recon]}")
    print(f"  a known 256:     {[f'{x:.8e}' for x in a_256_known]}")


# --- Main --------------------------------------------------------------

def main():
    print("DSD512 NTF Filter Generation (linear algebra method)")
    print("=" * 60)

    # Verify existing filters
    print("\nVerifying 256x44100 filters:")
    for name, order, a, g in FILTERS_256:
        h = hinf(a, g[:order], order)
        print(f"  {name:8s} order={order}  Hinf={h:.6f}")

    # Cross-validate
    cross_validate()

    # Generate DSD512
    print("\n\nGenerating 512x44100 filters:")
    print("-" * 60)
    results = []

    for name, order, a_256, g_256 in FILTERS_256:
        h_256 = hinf(a_256, g_256[:order], order)

        # Extract poles from 256 filter's denominator
        den_256 = crfb_den(a_256, g_256[:order], order)
        poles_w = np.roots(den_256[::-1])

        # Scale g for 512
        g_512 = [g_256[i] / 4.0 if i < len(g_256) else 0.0 for i in range(order)]

        # Solve for a_512 using same poles
        a_512 = solve_a_from_poles(g_512, order, poles_w)

        h_512 = hinf(a_512, g_512, order)
        ibn_512 = ibn(a_512, g_512, order, 512)

        # Check pole stability
        all_stable = all(abs(p) < 1.0 for p in poles_w)

        print(f"  {name:8s} order={order}  Hinf_256={h_256:.6f}  Hinf_512={h_512:.6f}  "
              f"IBN={ibn_512:.4e}  stable={all_stable}")

        results.append((name, order, a_512, g_512))

    # Output C code
    print("\n\n" + "=" * 60)
    print("C code for ntf.c")
    print("=" * 60)
    print()
    print("  /* ===============================================================")
    print("   * 512 x 44100  (DSD512 = 22579200 Hz)")
    print("   * Generated by tools/gen_ntf_512.py")
    print("   * Resonator g[] scaled /4 from 256x44100;")
    print("   * a[] derived analytically preserving NTF pole locations.")
    print("   * =============================================================== */")
    print()

    for idx, (name, order, a, g) in enumerate(results):
        print(f"  /* [{idx}] {name} @ 512x44100 */")
        print("  {")
        print("    {")
        for j in range(order):
            comma = "," if j < order - 1 else ","
            print(f"      {a[j]:.17e}{comma}")
        print("    },")
        print("    {")
        for j in range(0, order, 2):
            g0 = g[j] if j < len(g) else 0.0
            g1 = g[j + 1] if j + 1 < len(g) else 0.0
            if j + 1 < order:
                print(f"      {g0:.17e}, {g1:.17e},")
            else:
                print(f"      {g0:.17e},")
        print("    },")
        print(f"    {order}, 512 * 44100, \"{name}\"")
        print("  },")
        print()


if __name__ == "__main__":
    main()
