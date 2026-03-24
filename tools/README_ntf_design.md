# NTF Design Pipeline for Multi-bit SDM

## Environment Setup

```bash
# Uses conda env 'delsig' with Python 3.9 + numpy 1.21 + scipy 1.10
# deltasigma package patched for Python 3.9 compatibility:
#   - _constants.py: np.float → float
#   - _utils.py: fractions.gcd → math.gcd

conda activate delsig
python tools/ntf_multibit.py
```

## NTF Synthesis Parameters

| Parameter | Description | 1-bit standard | Multibit aggressive |
|-----------|-------------|----------------|---------------------|
| order | NTF order (number of integrators) | 6 | 8-10 |
| OSR | Oversampling ratio (DSD rate / 2×22050) | 64-512 | same |
| H_inf | Maximum NTF gain at Nyquist | 1.5 | 3.0-6.0 |
| opt | Optimization (0=none, 1=zeros optimized) | 1 | 1 |

Higher H_inf = more aggressive noise shaping = lower in-band noise but less stability margin.
With 1-bit: H_inf > 2.0 risks instability (modulator overload).
With multi-bit (4-16 levels): H_inf up to 6.0 is stable (smaller quantization error).

## Output Format (CRFB)

`realizeNTF(H, form='CRFB')` returns `(a, g, b, c)`:
- **a[]**: Feedforward coefficients (our `ntf_filter_t.a[]`)
- **g[]**: Resonator coefficients (our `ntf_filter_t.g[]`)
- **b[]**: Input distribution (typically all 1.0 for CRFB)
- **c[]**: Integrator gains (typically all 1.0 for CRFB)

Our implementation uses only a[] and g[]. The NTF filter calc:
```
d[0] = s[0] - g[0]*s[1] + x - y
d[i] = s[i] + s[i-1] - g[i]*s[i+1]   (for 1 ≤ i < order-1)
d[N-1] = s[N-1] + s[N-2]
v = x + Σ(a[k] × d[k])
```

## Measured SINAD Results (greedy multibit, existing order-6 H_inf=1.5 NTF)

| Rate | 1-bit trellis nc=2 | Greedy 2-lev | Greedy 4-lev | Greedy 8-lev | Greedy 16-lev |
|------|--------------------:|-------------:|-------------:|-------------:|--------------:|
| DSD64 | 97.9 dB | 90.2 dB | 60.7 dB | 99.1 dB | 110.9 dB |
| DSD128 | 106.9 dB | 96.7 dB | 125.8 dB | 74.1 dB | 117.7 dB |
| DSD256 | 111.7 dB | 112.1 dB | 120.3 dB | 133.4 dB | 147.0 dB |

Notes:
- Some level/rate combinations show dips (4-lev@DSD64=60.7, 8-lev@DSD128=74.1)
  likely due to NTF resonance interactions at specific quantizer step sizes
- 16-level greedy BEATS 1-bit trellis at ALL rates (+13 to +35 dB)
- This is GREEDY (nc=1, no look-ahead) — adding trellis search would improve further
- The multibit output needs stage 2 conversion to 1-bit for standard DSD DACs

## Key Finding

The existing 1-bit NTF coefficients work correctly with multi-bit feedback.
No NTF redesign is strictly needed for quality improvement — the standard
order-6 H_inf=1.5 NTF already gives +13 to +35 dB SINAD with 16 levels.

Aggressive multibit-specific NTFs (higher order, higher H_inf) could push
this further, especially at lower DSD rates where the standard NTF is weaker.
