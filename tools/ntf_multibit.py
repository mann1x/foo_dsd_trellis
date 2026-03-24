"""
NTF synthesis for multi-bit SDM using pydelsig.
Generates CRFB coefficients (a[], g[]) for various quantizer levels.
"""
import deltasigma as ds
import numpy as np
import warnings
warnings.filterwarnings('ignore')

rates = [('DSD64', 64), ('DSD128', 128), ('DSD256', 256), ('DSD512', 512)]

print('=== Standard NTF (order=6, H_inf=1.5) - CRFB coefficients ===\n')

for name, osr in rates:
    H = ds.synthesizeNTF(order=6, osr=osr, opt=1, H_inf=1.5)
    form = ds.realizeNTF(H, form='CRFB')
    a, g, b, c = form[0], form[1], form[2], form[3]
    print('%s (OSR=%d):' % (name, osr))
    print('  a = [%s]' % ', '.join('%.6f' % x for x in a))
    print('  g = [%s]' % ', '.join('%.6f' % x for x in g))
    print()

print('\n=== Aggressive NTFs for multibit ===\n')

for name, osr in rates:
    for H_inf in [2.0, 3.0, 4.0, 6.0]:
        for order in [6, 8, 10]:
            try:
                H = ds.synthesizeNTF(order=order, osr=osr, opt=1, H_inf=H_inf)
                form = ds.realizeNTF(H, form='CRFB')
                a, g = form[0], form[1]
                print('%s o=%d H_inf=%.1f:' % (name, order, H_inf))
                print('  a = [%s]' % ', '.join('%.6f' % x for x in a))
                print('  g = [%s]' % ', '.join('%.6f' % x for x in g))
            except Exception as e:
                print('%s o=%d H_inf=%.1f: FAILED' % (name, order, H_inf))
    print()
