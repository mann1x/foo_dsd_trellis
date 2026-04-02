/*
 * foo_dsd_trellis — Trellis (Viterbi look-ahead) SDM core
 *
 * Ported from mansr/sox sdm.c (LGPL v2.1+)
 * Adapted for float32 I/O and foobar2000 DSP context.
 */

/* Force /fp:precise for quality-critical trellis computation.
 * The cost computation is extremely sensitive to FP operation ordering —
 * FMA causes up to 13 dB quality variation (same root cause as CUDA --fmad=false).
 *
 * When SDM_FAST_MODE is defined (by trellis_fast.c), the pragma is skipped
 * and public functions get _fast suffix for runtime selection. */
#if defined(_MSC_VER) && !defined(SDM_FAST_MODE)
#pragma float_control(precise, on, push)
#endif

#include "../include/trellis.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _MSC_VER
#include <immintrin.h>
#endif

#define PATH_HASH_MASK (PATH_HASH_SIZE - 1)

#define sqr(x) ((x) * (x))

/* ─── NTF state advance (hot path — force-inlined) ─── */

/*
 * Generic filter calc for arbitrary order.
 * Used as fallback when order doesn't match specialized versions.
 */
static __forceinline double sdm_filter_calc_generic(const double *s, double *d,
                                                      const ntf_filter_t *f,
                                                      double x, double y)
{
    const double *a = f->a;
    const double *g = f->g;
    double v;
    int i;

    d[0] = s[0] - g[0] * s[1] + x - y;
    v = x + a[0] * d[0];

    for (i = 1; i < f->order - 1; i++) {
        d[i] = s[i] + s[i - 1] - g[i] * s[i + 1];
        v += a[i] * d[i];
    }

    d[i] = s[i] + s[i - 1];
    v += a[i] * d[i];

    return v;
}

/* Fully unrolled order-4 filter calc */
static __forceinline double sdm_filter_calc_o4(
    const double * __restrict s, double * __restrict d,
    const double * __restrict a, const double * __restrict g,
    double x, double y)
{
    d[0] = s[0] - g[0] * s[1] + x - y;
    d[1] = s[1] + s[0] - g[1] * s[2];
    d[2] = s[2] + s[1] - g[2] * s[3];
    d[3] = s[3] + s[2];
    return x + a[0] * d[0] + a[1] * d[1] + a[2] * d[2] + a[3] * d[3];
}

/* Fully unrolled order-5 filter calc */
static __forceinline double sdm_filter_calc_o5(
    const double * __restrict s, double * __restrict d,
    const double * __restrict a, const double * __restrict g,
    double x, double y)
{
    d[0] = s[0] - g[0] * s[1] + x - y;
    d[1] = s[1] + s[0] - g[1] * s[2];
    d[2] = s[2] + s[1] - g[2] * s[3];
    d[3] = s[3] + s[2] - g[3] * s[4];
    d[4] = s[4] + s[3];
    return x + a[0] * d[0] + a[1] * d[1] + a[2] * d[2] + a[3] * d[3] + a[4] * d[4];
}

/* Fully unrolled order-6 filter calc */
static __forceinline double sdm_filter_calc_o6(
    const double * __restrict s, double * __restrict d,
    const double * __restrict a, const double * __restrict g,
    double x, double y)
{
    d[0] = s[0] - g[0] * s[1] + x - y;
    d[1] = s[1] + s[0] - g[1] * s[2];
    d[2] = s[2] + s[1] - g[2] * s[3];
    d[3] = s[3] + s[2] - g[3] * s[4];
    d[4] = s[4] + s[3] - g[4] * s[5];
    d[5] = s[5] + s[4];
    return x + a[0] * d[0] + a[1] * d[1] + a[2] * d[2]
             + a[3] * d[3] + a[4] * d[4] + a[5] * d[5];
}

/* Fully unrolled order-7 filter calc */
static __forceinline double sdm_filter_calc_o7(
    const double * __restrict s, double * __restrict d,
    const double * __restrict a, const double * __restrict g,
    double x, double y)
{
    d[0] = s[0] - g[0] * s[1] + x - y;
    d[1] = s[1] + s[0] - g[1] * s[2];
    d[2] = s[2] + s[1] - g[2] * s[3];
    d[3] = s[3] + s[2] - g[3] * s[4];
    d[4] = s[4] + s[3] - g[4] * s[5];
    d[5] = s[5] + s[4] - g[5] * s[6];
    d[6] = s[6] + s[5];
    return x + a[0] * d[0] + a[1] * d[1] + a[2] * d[2]
             + a[3] * d[3] + a[4] * d[4] + a[5] * d[5] + a[6] * d[6];
}

/* Fully unrolled order-8 filter calc */
static __forceinline double sdm_filter_calc_o8(
    const double * __restrict s, double * __restrict d,
    const double * __restrict a, const double * __restrict g,
    double x, double y)
{
    d[0] = s[0] - g[0] * s[1] + x - y;
    d[1] = s[1] + s[0] - g[1] * s[2];
    d[2] = s[2] + s[1] - g[2] * s[3];
    d[3] = s[3] + s[2] - g[3] * s[4];
    d[4] = s[4] + s[3] - g[4] * s[5];
    d[5] = s[5] + s[4] - g[5] * s[6];
    d[6] = s[6] + s[5] - g[6] * s[7];
    d[7] = s[7] + s[6];
    return x + a[0] * d[0] + a[1] * d[1] + a[2] * d[2] + a[3] * d[3]
             + a[4] * d[4] + a[5] * d[5] + a[6] * d[6] + a[7] * d[7];
}

/* Dispatch to order-specialized filter calc */
static __forceinline double sdm_filter_calc(const double *s, double *d,
                                              const ntf_filter_t *f,
                                              double x, double y)
{
    switch (f->order) {
    case 4: return sdm_filter_calc_o4(s, d, f->a, f->g, x, y);
    case 5: return sdm_filter_calc_o5(s, d, f->a, f->g, x, y);
    case 6: return sdm_filter_calc_o6(s, d, f->a, f->g, x, y);
    case 7: return sdm_filter_calc_o7(s, d, f->a, f->g, x, y);
    case 8: return sdm_filter_calc_o8(s, d, f->a, f->g, x, y);
    default: return sdm_filter_calc_generic(s, d, f, x, y);
    }
}

static __forceinline void sdm_filter_calc2(sdm_state_t *src, sdm_state_t *dst,
                                             const ntf_filter_t *f, double x,
                                             double state_limit)
{
    const double *a = f->a;
    double v;

    v = sdm_filter_calc(src->state, dst[0].state, f, x, 0.0);

    /* Copy state vector: dst[1] = dst[0] using AVX2 where possible */
#if defined(_MSC_VER) && defined(__AVX2__)
    {
        /* Copy first 4 doubles (32 bytes) with AVX2, rest scalar */
        __m256d st4 = _mm256_loadu_pd(dst[0].state);
        _mm256_storeu_pd(dst[1].state, st4);
        for (int i = 4; i < f->order; i++)
            dst[1].state[i] = dst[0].state[i];
    }
#else
    for (int i = 0; i < f->order; i++)
        dst[1].state[i] = dst[0].state[i];
#endif

    dst[0].state[0] += 1.0;
    dst[1].state[0] -= 1.0;

    /* SDM limiter: clamp integrator states to prevent overload instability */
    if (state_limit > 0.0) {
        for (int i = 0; i < f->order; i++) {
            if (dst[0].state[i] > state_limit)       dst[0].state[i] = state_limit;
            else if (dst[0].state[i] < -state_limit)  dst[0].state[i] = -state_limit;
            if (dst[1].state[i] > state_limit)       dst[1].state[i] = state_limit;
            else if (dst[1].state[i] < -state_limit)  dst[1].state[i] = -state_limit;
        }
    }

    dst[0].cost = src->cost + sqr(v + a[0]);
    dst[1].cost = src->cost + sqr(v - a[0]);
}

/* ─── Batched 2-candidate SSE2/FMA filter calc (all orders) ─── */

#if defined(_MSC_VER) && defined(__AVX2__)

/*
 * Process 2 candidates simultaneously through filter calc + step.
 * Uses __m128d (2 doubles) with explicit FMA intrinsics.
 * This is the hot path for nc=2 (production config) at ALL orders.
 *
 * Under #pragma float_control(precise), the compiler won't auto-FMA.
 * Explicit _mm_fmadd_pd / _mm_fnmadd_pd intrinsics ARE honored,
 * giving us controlled FMA with deterministic results.
 */
static __forceinline void sdm_step_2x(
    sdm_context_t *p,
    sdm_state_t *src0, sdm_state_t *src1,
    sdm_state_t *dst0, sdm_state_t *dst1,
    double x)
{
    const ntf_filter_t *f = p->filter;
    const double *a = f->a;
    const double *g = f->g;
    const int order = f->order;

    /* Pack 2 candidates' state[i] into __m128d: {src0->state[i], src1->state[i]} */
    __m128d s[MAX_NTF_ORDER], d[MAX_NTF_ORDER];
    for (int i = 0; i < order; i++)
        s[i] = _mm_set_pd(src1->state[i], src0->state[i]);

    __m128d x2 = _mm_set1_pd(x);

    /* d[0] = s[0] - g[0]*s[1] + x  (y=0)
     * MUST match scalar operation order EXACTLY — IEEE 754 is not
     * associative. Scalar: (s[0] - g[0]*s[1]) + x.
     * NO FMA — FMA changes rounding and causes up to 20 dB quality loss. */
    d[0] = _mm_sub_pd(s[0], _mm_mul_pd(_mm_set1_pd(g[0]), s[1]));
    d[0] = _mm_add_pd(d[0], x2);

    /* d[i] = s[i] + s[i-1] - g[i]*s[i+1]  for i=1..order-2
     * Scalar order: (s[i] + s[i-1]) - g[i]*s[i+1] */
    for (int i = 1; i < order - 1; i++) {
        __m128d sum = _mm_add_pd(s[i], s[i - 1]);
        d[i] = _mm_sub_pd(sum, _mm_mul_pd(_mm_set1_pd(g[i]), s[i + 1]));
    }

    /* d[order-1] = s[order-1] + s[order-2] */
    d[order - 1] = _mm_add_pd(s[order - 1], s[order - 2]);

    /* v = x + a[0]*d[0] + a[1]*d[1] + ...
     * Scalar order: v = (x + a[0]*d[0]); v += a[1]*d[1]; ... */
    __m128d v2 = _mm_add_pd(x2, _mm_mul_pd(_mm_set1_pd(a[0]), d[0]));
    for (int i = 1; i < order; i++)
        v2 = _mm_add_pd(v2, _mm_mul_pd(_mm_set1_pd(a[i]), d[i]));

    /* Store states for both children of each candidate.
     * dst[0] = y=+1: state = d, state[0] += 1
     * dst[1] = y=-1: state = d, state[0] -= 1 */
    for (int i = 0; i < order; i++) {
        double dd[2];
        _mm_storeu_pd(dd, d[i]);
        dst0[0].state[i] = dd[0];
        dst0[1].state[i] = dd[0];
        dst1[0].state[i] = dd[1];
        dst1[1].state[i] = dd[1];
    }
    dst0[0].state[0] += 1.0;
    dst0[1].state[0] -= 1.0;
    dst1[0].state[0] += 1.0;
    dst1[1].state[0] -= 1.0;

    /* Clamp states if limiter is active */
    if (p->state_limit > 0.0) {
        double lim = p->state_limit;
        __m128d lim_p = _mm_set1_pd(lim);
        __m128d lim_n = _mm_set1_pd(-lim);
        for (int b = 0; b < 2; b++) {
            for (int i = 0; i < order; i += 2) {
                int n = (i + 2 <= order) ? 2 : 1;
                if (n == 2) {
                    __m128d v0 = _mm_loadu_pd(&dst0[b].state[i]);
                    __m128d v1 = _mm_loadu_pd(&dst1[b].state[i]);
                    v0 = _mm_min_pd(_mm_max_pd(v0, lim_n), lim_p);
                    v1 = _mm_min_pd(_mm_max_pd(v1, lim_n), lim_p);
                    _mm_storeu_pd(&dst0[b].state[i], v0);
                    _mm_storeu_pd(&dst1[b].state[i], v1);
                } else {
                    if (dst0[b].state[i] > lim) dst0[b].state[i] = lim;
                    else if (dst0[b].state[i] < -lim) dst0[b].state[i] = -lim;
                    if (dst1[b].state[i] > lim) dst1[b].state[i] = lim;
                    else if (dst1[b].state[i] < -lim) dst1[b].state[i] = -lim;
                }
            }
        }
    }

    /* Costs: sqr(v ± a[0]) + src->cost */
    __m128d a0_2 = _mm_set1_pd(a[0]);
    __m128d vpa = _mm_add_pd(v2, a0_2);
    __m128d vma = _mm_sub_pd(v2, a0_2);
    __m128d cost_p = _mm_mul_pd(vpa, vpa);
    __m128d cost_m = _mm_mul_pd(vma, vma);
    __m128d src_costs = _mm_set_pd(src1->cost, src0->cost);
    cost_p = _mm_add_pd(src_costs, cost_p);
    cost_m = _mm_add_pd(src_costs, cost_m);

    double cp[2], cm[2];
    _mm_storeu_pd(cp, cost_p);
    _mm_storeu_pd(cm, cost_m);
    dst0[0].cost = cp[0]; dst0[1].cost = cm[0];
    dst1[0].cost = cp[1]; dst1[1].cost = cm[1];

    /* Path, hist, parent assignments */
    uint32_t mask = (uint32_t)p->trellis_mask;
    dst0[0].path = (src0->path << 1 | 0u) & mask;
    dst0[1].path = (src0->path << 1 | 1u) & mask;
    dst0[0].hist = dst0[1].hist = src0->hist;
    dst0[0].next = dst0[1].next = src0->next;
    dst0[0].parent = dst0[1].parent = src0;

    dst1[0].path = (src1->path << 1 | 0u) & mask;
    dst1[1].path = (src1->path << 1 | 1u) & mask;
    dst1[0].hist = dst1[1].hist = src1->hist;
    dst1[0].next = dst1[1].next = src1->next;
    dst1[0].parent = dst1[1].parent = src1;
}

#endif /* _MSC_VER && __AVX2__ */

/* ─── Batched 4-candidate AVX2 filter calc ─── */

#if defined(_MSC_VER) && defined(__AVX2__)

/*
 * Transpose 4 __m256d row vectors into 4 column vectors.
 * Input:  a = {a0,a1,a2,a3}, b = {b0,b1,b2,b3}, c, d
 * Output: r0 = {a0,b0,c0,d0}, r1 = {a1,b1,c1,d1}, r2, r3
 */
static __forceinline void transpose_4x4_pd(
    __m256d a, __m256d b, __m256d c, __m256d d,
    __m256d *r0, __m256d *r1, __m256d *r2, __m256d *r3)
{
    __m256d t0 = _mm256_unpacklo_pd(a, b);
    __m256d t1 = _mm256_unpackhi_pd(a, b);
    __m256d t2 = _mm256_unpacklo_pd(c, d);
    __m256d t3 = _mm256_unpackhi_pd(c, d);
    *r0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    *r1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    *r2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    *r3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}

/*
 * Process 4 candidates simultaneously through sdm_step for order 8.
 * Computes filter_calc2 + path/hist assignments for all 4 in parallel.
 */
static __forceinline void sdm_step_4x_o8(
    sdm_context_t *p,
    sdm_state_t *src0, sdm_state_t *src1,
    sdm_state_t *src2, sdm_state_t *src3,
    sdm_state_t *dst0, sdm_state_t *dst1,
    sdm_state_t *dst2, sdm_state_t *dst3,
    double x)
{
    const ntf_filter_t *f = p->filter;
    const double *a = f->a;
    const double *g = f->g;

    /* ── Gather: transpose 4 AoS state vectors → 8 SoA __m256d ── */
    __m256d s0, s1, s2, s3, s4, s5, s6, s7;
    {
        /* Load state[0..3] from each candidate, then transpose */
        __m256d r0 = _mm256_loadu_pd(src0->state);
        __m256d r1 = _mm256_loadu_pd(src1->state);
        __m256d r2 = _mm256_loadu_pd(src2->state);
        __m256d r3 = _mm256_loadu_pd(src3->state);
        transpose_4x4_pd(r0, r1, r2, r3, &s0, &s1, &s2, &s3);

        /* Load state[4..7] from each candidate, then transpose */
        r0 = _mm256_loadu_pd(src0->state + 4);
        r1 = _mm256_loadu_pd(src1->state + 4);
        r2 = _mm256_loadu_pd(src2->state + 4);
        r3 = _mm256_loadu_pd(src3->state + 4);
        transpose_4x4_pd(r0, r1, r2, r3, &s4, &s5, &s6, &s7);
    }

    /* ── Compute: order-8 filter calc for all 4 candidates ── */
    __m256d x4 = _mm256_set1_pd(x);
    __m256d d0, d1, d2, d3, d4, d5, d6, d7;

    /* d[0] = s[0] - g[0]*s[1] + x  (y=0) */
    d0 = _mm256_add_pd(s0, x4);
    d0 = _mm256_fnmadd_pd(_mm256_set1_pd(g[0]), s1, d0);

    /* d[1] = s[1] + s[0] - g[1]*s[2] */
    d1 = _mm256_add_pd(s1, s0);
    d1 = _mm256_fnmadd_pd(_mm256_set1_pd(g[1]), s2, d1);

    /* d[2] = s[2] + s[1] - g[2]*s[3] */
    d2 = _mm256_add_pd(s2, s1);
    d2 = _mm256_fnmadd_pd(_mm256_set1_pd(g[2]), s3, d2);

    /* d[3] = s[3] + s[2] - g[3]*s[4] */
    d3 = _mm256_add_pd(s3, s2);
    d3 = _mm256_fnmadd_pd(_mm256_set1_pd(g[3]), s4, d3);

    /* d[4] = s[4] + s[3] - g[4]*s[5] */
    d4 = _mm256_add_pd(s4, s3);
    d4 = _mm256_fnmadd_pd(_mm256_set1_pd(g[4]), s5, d4);

    /* d[5] = s[5] + s[4] - g[5]*s[6] */
    d5 = _mm256_add_pd(s5, s4);
    d5 = _mm256_fnmadd_pd(_mm256_set1_pd(g[5]), s6, d5);

    /* d[6] = s[6] + s[5] - g[6]*s[7] */
    d6 = _mm256_add_pd(s6, s5);
    d6 = _mm256_fnmadd_pd(_mm256_set1_pd(g[6]), s7, d6);

    /* d[7] = s[7] + s[6] */
    d7 = _mm256_add_pd(s7, s6);

    /* v = x + a[0]*d[0] + a[1]*d[1] + ... + a[7]*d[7] */
    __m256d v4 = _mm256_fmadd_pd(_mm256_set1_pd(a[0]), d0, x4);
    v4 = _mm256_fmadd_pd(_mm256_set1_pd(a[1]), d1, v4);
    v4 = _mm256_fmadd_pd(_mm256_set1_pd(a[2]), d2, v4);
    v4 = _mm256_fmadd_pd(_mm256_set1_pd(a[3]), d3, v4);
    v4 = _mm256_fmadd_pd(_mm256_set1_pd(a[4]), d4, v4);
    v4 = _mm256_fmadd_pd(_mm256_set1_pd(a[5]), d5, v4);
    v4 = _mm256_fmadd_pd(_mm256_set1_pd(a[6]), d6, v4);
    v4 = _mm256_fmadd_pd(_mm256_set1_pd(a[7]), d7, v4);

    /* ── Scatter: transpose 8 SoA __m256d → 4 AoS state vectors ── */
    __m256d out0_lo, out0_hi, out1_lo, out1_hi, out2_lo, out2_hi, out3_lo, out3_hi;
    transpose_4x4_pd(d0, d1, d2, d3, &out0_lo, &out1_lo, &out2_lo, &out3_lo);
    transpose_4x4_pd(d4, d5, d6, d7, &out0_hi, &out1_hi, &out2_hi, &out3_hi);

    /* Store to dst[0].state and copy to dst[1].state for each candidate */

    /* Candidate 0 */
    _mm256_storeu_pd(dst0[0].state, out0_lo);
    _mm256_storeu_pd(dst0[0].state + 4, out0_hi);
    _mm256_storeu_pd(dst0[1].state, out0_lo);
    _mm256_storeu_pd(dst0[1].state + 4, out0_hi);
    dst0[0].state[0] += 1.0;
    dst0[1].state[0] -= 1.0;

    /* Candidate 1 */
    _mm256_storeu_pd(dst1[0].state, out1_lo);
    _mm256_storeu_pd(dst1[0].state + 4, out1_hi);
    _mm256_storeu_pd(dst1[1].state, out1_lo);
    _mm256_storeu_pd(dst1[1].state + 4, out1_hi);
    dst1[0].state[0] += 1.0;
    dst1[1].state[0] -= 1.0;

    /* Candidate 2 */
    _mm256_storeu_pd(dst2[0].state, out2_lo);
    _mm256_storeu_pd(dst2[0].state + 4, out2_hi);
    _mm256_storeu_pd(dst2[1].state, out2_lo);
    _mm256_storeu_pd(dst2[1].state + 4, out2_hi);
    dst2[0].state[0] += 1.0;
    dst2[1].state[0] -= 1.0;

    /* Candidate 3 */
    _mm256_storeu_pd(dst3[0].state, out3_lo);
    _mm256_storeu_pd(dst3[0].state + 4, out3_hi);
    _mm256_storeu_pd(dst3[1].state, out3_lo);
    _mm256_storeu_pd(dst3[1].state + 4, out3_hi);
    dst3[0].state[0] += 1.0;
    dst3[1].state[0] -= 1.0;

    /* SDM limiter: clamp integrator states (AVX2 vectorized) */
    if (p->state_limit > 0.0) {
        __m256d lim_p = _mm256_set1_pd(p->state_limit);
        __m256d lim_n = _mm256_set1_pd(-p->state_limit);
        #define CLAMP_STATE_AVX2(dst_pair) do { \
            for (int _b = 0; _b < 2; _b++) { \
                __m256d lo = _mm256_loadu_pd((dst_pair)[_b].state); \
                __m256d hi = _mm256_loadu_pd((dst_pair)[_b].state + 4); \
                lo = _mm256_min_pd(_mm256_max_pd(lo, lim_n), lim_p); \
                hi = _mm256_min_pd(_mm256_max_pd(hi, lim_n), lim_p); \
                _mm256_storeu_pd((dst_pair)[_b].state, lo); \
                _mm256_storeu_pd((dst_pair)[_b].state + 4, hi); \
            } \
        } while(0)
        CLAMP_STATE_AVX2(dst0);
        CLAMP_STATE_AVX2(dst1);
        CLAMP_STATE_AVX2(dst2);
        CLAMP_STATE_AVX2(dst3);
        #undef CLAMP_STATE_AVX2
    }

    /* ── Costs: extract v values and compute per-candidate ── */
    {
        __m256d a0_4 = _mm256_set1_pd(a[0]);
        __m256d vpa = _mm256_add_pd(v4, a0_4);  /* v + a[0] for all 4 */
        __m256d vma = _mm256_sub_pd(v4, a0_4);  /* v - a[0] for all 4 */
        __m256d cost_p = _mm256_mul_pd(vpa, vpa); /* sqr(v + a[0]) */
        __m256d cost_m = _mm256_mul_pd(vma, vma); /* sqr(v - a[0]) */

        /* Load source costs and add */
        __m256d src_costs = _mm256_set_pd(
            src3->cost, src2->cost, src1->cost, src0->cost);
        cost_p = _mm256_add_pd(src_costs, cost_p);
        cost_m = _mm256_add_pd(src_costs, cost_m);

        /* Extract and store costs */
        double cp[4], cm[4];
        _mm256_storeu_pd(cp, cost_p);
        _mm256_storeu_pd(cm, cost_m);
        dst0[0].cost = cp[0]; dst0[1].cost = cm[0];
        dst1[0].cost = cp[1]; dst1[1].cost = cm[1];
        dst2[0].cost = cp[2]; dst2[1].cost = cm[2];
        dst3[0].cost = cp[3]; dst3[1].cost = cm[3];
    }

    /* ── Path, hist, parent assignments ── */
    uint32_t mask = (uint32_t)p->trellis_mask;

    dst0[0].path = (src0->path << 1 | 0u) & mask;
    dst0[1].path = (src0->path << 1 | 1u) & mask;
    dst0[0].hist = dst0[1].hist = src0->hist;
    dst0[0].next = dst0[1].next = src0->next;
    dst0[0].parent = dst0[1].parent = src0;

    dst1[0].path = (src1->path << 1 | 0u) & mask;
    dst1[1].path = (src1->path << 1 | 1u) & mask;
    dst1[0].hist = dst1[1].hist = src1->hist;
    dst1[0].next = dst1[1].next = src1->next;
    dst1[0].parent = dst1[1].parent = src1;

    dst2[0].path = (src2->path << 1 | 0u) & mask;
    dst2[1].path = (src2->path << 1 | 1u) & mask;
    dst2[0].hist = dst2[1].hist = src2->hist;
    dst2[0].next = dst2[1].next = src2->next;
    dst2[0].parent = dst2[1].parent = src2;

    dst3[0].path = (src3->path << 1 | 0u) & mask;
    dst3[1].path = (src3->path << 1 | 1u) & mask;
    dst3[0].hist = dst3[1].hist = src3->hist;
    dst3[0].next = dst3[1].next = src3->next;
    dst3[0].parent = dst3[1].parent = src3;
}

#endif /* _MSC_VER && __AVX2__ */

/* ─── History buffer management ─── */

static inline unsigned sdm_histbuf_get(sdm_context_t *p)
{
    return p->hist_free[--p->hist_fnum];
}

static inline void sdm_histbuf_put(sdm_context_t *p, unsigned h)
{
    p->hist_free[p->hist_fnum++] = (uint8_t)h;
}

/* ─── Bit-packed history access ─── */

static inline unsigned get_bit(const uint8_t *p, unsigned i)
{
    return (p[i >> 3] >> (i & 7)) & 1;
}

static inline void put_bit(uint8_t *p, unsigned i, unsigned v)
{
    int b = p[i >> 3];
    int s = i & 7;
    b &= ~(1 << s);
    b |= (int)v << s;
    p[i >> 3] = (uint8_t)b;
}

static inline unsigned sdm_hist_get(const sdm_context_t *p, unsigned h, unsigned i)
{
    return get_bit(p->hist[h], i);
}

static inline void sdm_hist_put(sdm_context_t *p, unsigned h, unsigned i, unsigned v)
{
    put_bit(p->hist[h], i, v);
}

static inline void sdm_hist_copy(sdm_context_t *p, unsigned d, unsigned s)
{
    memcpy(p->hist[d], p->hist[s], (size_t)(p->trellis_lat + 7) / 8);
}

/* ─── Cost comparison (IEEE 754 integer comparison trick) ─── */

static inline int64_t dbl2int64(double a)
{
    union { double d; int64_t i; } v;
    v.d = a;
    return v.i;
}

static __forceinline int sdm_cmplt(const sdm_state_t *a, const sdm_state_t *b)
{
    return dbl2int64(a->cost) < dbl2int64(b->cost);
}

static __forceinline int sdm_cmple(const sdm_state_t *a, const sdm_state_t *b)
{
    return dbl2int64(a->cost) <= dbl2int64(b->cost);
}

/* ─── Path deduplication via hash table ─── */

static sdm_state_t *sdm_check_path(sdm_context_t *p, sdm_state_t *s)
{
    unsigned index = s->path & PATH_HASH_MASK;
    sdm_state_t **hash = p->path_hash;
    sdm_state_t *t = hash[index];

    while (t) {
        if (t->path == s->path)
            return t;
        t = t->path_list;
    }

    s->path_list = hash[index];
    hash[index] = s;

    return NULL;
}

/* ─── Candidate sorting with path dedup ─── */

static unsigned sdm_sort_cands(sdm_context_t *p, sdm_trellis_t *st)
{
    sdm_state_t *r = NULL, *s, *t;
    sdm_state_t *min = NULL;
    unsigned i, j, n;
    unsigned total_children = p->mb_levels * p->num_cands;

    for (i = 0; i < total_children; i++) {
        s = &st->sdm[i];
        p->path_hash[s->path & PATH_HASH_MASK] = NULL;
        if (!min || sdm_cmplt(s, min))
            min = s;
    }

    p->total_children += total_children;

    /* Determine the output bit from the majority of candidates, not just min.
     * This is more robust than using only min->next, especially with many candidates. */
    unsigned next_votes[2] = {0, 0};
    for (i = 0; i < total_children; i++)
        next_votes[st->sdm[i].next & 1]++;
    unsigned majority_next = (next_votes[1] > next_votes[0]) ? 1 : 0;
    if (min->next != majority_next) {
        sdm_state_t *best_maj = NULL;
        for (i = 0; i < total_children; i++) {
            s = &st->sdm[i];
            if (s->next == majority_next && (!best_maj || sdm_cmplt(s, best_maj)))
                best_maj = s;
        }
        if (best_maj && best_maj->cost < min->cost * 1.1)
            min = best_maj;
    }

    for (i = 0, n = 0; i < total_children; i++) {
        s = &st->sdm[i];

        if (s->next != min->next) {
            p->next_filter_drops++;
            continue;
        }

        if (n == p->trellis_num && sdm_cmple(st->act[n - 1], s))
            continue;

        t = sdm_check_path(p, s);

        if (!t) {
            for (j = n; j > 0; j--) {
                t = st->act[j - 1];
                if (sdm_cmple(t, s))
                    break;
                st->act[j] = t;
            }
            if (j < p->trellis_num)
                st->act[j] = s;
            if (n < p->trellis_num)
                n++;
            continue;
        }

        if (sdm_cmple(t, s))
            continue;

        /* Find where t is in act[] before replacing */
        unsigned t_pos;
        bool t_found = false;
        for (t_pos = 0; t_pos < n; t_pos++) {
            if (st->act[t_pos] == t) {
                t_found = true;
                break;
            }
        }

        if (!t_found) {
            /* t was evicted from act[] by a better entry earlier.
             * Just insert s like a new non-duplicate entry. */
            for (j = n; j > 0; j--) {
                r = st->act[j - 1];
                if (sdm_cmple(r, s))
                    break;
                if (j < p->trellis_num)
                    st->act[j] = r;
            }
            if (j < p->trellis_num)
                st->act[j] = s;
            if (n < p->trellis_num)
                n++;
            continue;
        }

        /* Remove t from act[] and insert s at correct position */
        /* First remove t by shifting elements after t_pos left */
        for (j = t_pos; j + 1 < n; j++)
            st->act[j] = st->act[j + 1];
        n--;

        /* Now insert s in sorted position */
        for (j = n; j > 0; j--) {
            r = st->act[j - 1];
            if (sdm_cmple(r, s))
                break;
            if (j < p->trellis_num)
                st->act[j] = r;
        }
        if (j < p->trellis_num)
            st->act[j] = s;
        n++;
    }

    return n;
}

/* ─── Per-candidate step ─── */

static __forceinline void sdm_step(sdm_context_t *p, sdm_state_t *cur,
                                    sdm_state_t *next, double x)
{
    sdm_filter_calc2(cur, next, p->filter, x, p->state_limit);

    for (int i = 0; i < 2; i++) {
        next[i].path = (cur->path << 1 | (unsigned)i) & p->trellis_mask;
        next[i].hist = cur->hist;
        next[i].next = cur->next;
        next[i].parent = cur;
    }
}

/* ─── Multi-bit step: expand one candidate to N children ─── */

static __forceinline void sdm_step_mb(sdm_context_t *p, sdm_state_t *cur,
                                       sdm_state_t *children, double x, int nlevels)
{
    const ntf_filter_t *f = p->filter;
    const double *a = f->a;
    double d_base[MAX_NTF_ORDER];
    double v;

    /* NTF filter calc with y=0 */
    v = sdm_filter_calc(cur->state, d_base, f, x, 0.0);

    /* Generate N children at equally-spaced levels in [-1, +1] */
    double step = 2.0 / (double)(nlevels - 1);
    for (int lv = 0; lv < nlevels; lv++) {
        double y = -1.0 + (double)lv * step;
        sdm_state_t *ch = &children[lv];

        /* State = d_base - y (NTF formula: d[0] = s[0] - g[0]*s[1] + x - y) */
        for (int k = 0; k < f->order; k++)
            ch->state[k] = d_base[k];
        ch->state[0] -= y;

        /* Clamp */
        if (p->state_limit > 0.0) {
            for (int k = 0; k < f->order; k++) {
                if (ch->state[k] > p->state_limit) ch->state[k] = p->state_limit;
                else if (ch->state[k] < -p->state_limit) ch->state[k] = -p->state_limit;
            }
        }

        /* Cost: sqr(v - y * a[0]) */
        double veff = v - y * a[0];
        ch->cost = cur->cost + veff * veff;

        /* Path: store sign bit (1 if positive, 0 if negative) for dedup
         * compatibility with existing 1-bit path hash. */
        unsigned sign_bit = (lv >= nlevels / 2) ? 1 : 0;
        ch->path = (cur->path << 1 | sign_bit) & p->trellis_mask;
        ch->hist = cur->hist;
        ch->next = cur->next;
        ch->parent = cur;
    }
}

/* ─── Multi-bit per-sample trellis: N levels, multibit output ─── */

static double sdm_sample_trellis_mb(sdm_context_t *p, double x)
{
    sdm_trellis_t *st_cur  = &p->trellis[p->idx];
    sdm_trellis_t *st_next = &p->trellis[p->idx ^ 1];
    int nlevels = (int)p->mb_levels;
    double step = 2.0 / (double)(nlevels - 1);
    unsigned next_pos = p->pos + 1;
    if (next_pos == p->trellis_lat)
        next_pos = 0;

    /* Expand each candidate to N children */
    for (unsigned i = 0; i < p->num_cands; i++) {
        sdm_state_t *cur = st_cur->act[i];
        sdm_step_mb(p, cur, &st_next->sdm[nlevels * i], x, nlevels);
        cur->next = (uint8_t)sdm_hist_get(p, cur->hist, next_pos);
        cur->hist_used = 0;
    }

    /* Sort/prune */
    unsigned new_cands = sdm_sort_cands(p, st_next);
    double min_cost = st_next->act[0]->cost;

    /* Traceback output: the 'next' field carries the sign bit from lat samples ago.
     * For multibit stage 1 output, we output the sign as ±1 for now.
     * TODO: store full level index in history for true multibit traceback. */
    unsigned output_bit = st_next->act[0]->next;

    /* Also capture the IMMEDIATE best candidate's level for multibit output.
     * The best candidate's path[0:1] holds the level index it chose. */
    unsigned best_level_idx = st_next->act[0]->path & (unsigned)(nlevels - 1);
    double best_level = -1.0 + (double)best_level_idx * step;

    for (unsigned i = 0; i < new_cands; i++) {
        sdm_state_t *s = st_next->act[i];
        if (s->parent->hist_used) {
            unsigned h = sdm_histbuf_get(p);
            sdm_hist_copy(p, h, s->hist);
            s->hist = (uint8_t)h;
        } else {
            s->parent->hist_used = 1;
        }
        s->cost -= min_cost;
        s->next = s->parent->next;
        /* Store sign bit in history for traceback */
        unsigned lv_idx = s->path & (unsigned)(nlevels - 1);
        unsigned sign_bit = (lv_idx >= (unsigned)(nlevels / 2)) ? 1 : 0;
        sdm_hist_put(p, s->hist, p->pos, sign_bit);
    }

    for (unsigned i = 0; i < p->num_cands; i++) {
        sdm_state_t *s = st_cur->act[i];
        if (!s->hist_used)
            sdm_histbuf_put(p, s->hist);
    }

    if (new_cands < p->num_cands) {
        p->conv_fail++;
        p->cands_collapse++;
    }

    p->num_cands = new_cands;
    p->pos = next_pos;
    p->idx ^= 1;

    /* For multibit: output the immediate best level (not traceback).
     * This loses look-ahead latency compensation but gives true multibit output.
     * The pending/latency mechanism still buffers — output starts after lat samples. */
    return best_level;
}

/* ─── Main per-sample trellis algorithm ─── */

static double sdm_sample_trellis(sdm_context_t *p, double x)
{
    sdm_trellis_t *st_cur  = &p->trellis[p->idx];
    sdm_trellis_t *st_next = &p->trellis[p->idx ^ 1];
    double min_cost;
    unsigned new_cands;
    unsigned next_pos;
    unsigned output;
    unsigned i;

    next_pos = p->pos + 1;
    if (next_pos == p->trellis_lat)
        next_pos = 0;

#if defined(_MSC_VER) && defined(__AVX2__)
    /* NOTE: SIMD filter calc was tested and produces different bit decisions
     * from scalar (up to 24 dB SINAD variation). The trellis is chaotically
     * sensitive to ULP-level numerical differences in the filter output.
     * Even with identical operation order and no FMA, SIMD mul/add on packed
     * data produces different rounding than scalar on separate data.
     * The filter calc MUST remain scalar. Same root cause as CUDA --fmad=false.
     *
     * Batched 2-candidate SIMD path DISABLED — quality degradation unacceptable.
     */
    if (0 && p->num_cands == 2 && p->mb_levels == 2) {
        sdm_step_2x(p,
            st_cur->act[0], st_cur->act[1],
            &st_next->sdm[0], &st_next->sdm[2],
            x);
        st_cur->act[0]->next = (uint8_t)sdm_hist_get(p, st_cur->act[0]->hist, next_pos);
        st_cur->act[0]->hist_used = 0;
        st_cur->act[1]->next = (uint8_t)sdm_hist_get(p, st_cur->act[1]->hist, next_pos);
        st_cur->act[1]->hist_used = 0;
        i = 2;
    } else
    /* Batched 4-candidate AVX2 path for order 8 (1-bit only) */
    if (p->filter->order == 8 && p->mb_levels == 2) {
        for (i = 0; i + 3 < p->num_cands; i += 4) {
            sdm_step_4x_o8(p,
                st_cur->act[i], st_cur->act[i+1],
                st_cur->act[i+2], st_cur->act[i+3],
                &st_next->sdm[2*i], &st_next->sdm[2*(i+1)],
                &st_next->sdm[2*(i+2)], &st_next->sdm[2*(i+3)],
                x);
            st_cur->act[i]->next   = (uint8_t)sdm_hist_get(p, st_cur->act[i]->hist, next_pos);
            st_cur->act[i]->hist_used = 0;
            st_cur->act[i+1]->next = (uint8_t)sdm_hist_get(p, st_cur->act[i+1]->hist, next_pos);
            st_cur->act[i+1]->hist_used = 0;
            st_cur->act[i+2]->next = (uint8_t)sdm_hist_get(p, st_cur->act[i+2]->hist, next_pos);
            st_cur->act[i+2]->hist_used = 0;
            st_cur->act[i+3]->next = (uint8_t)sdm_hist_get(p, st_cur->act[i+3]->hist, next_pos);
            st_cur->act[i+3]->hist_used = 0;
        }
        for (; i < p->num_cands; i++) {
            sdm_state_t *cur  = st_cur->act[i];
            sdm_state_t *next = &st_next->sdm[2 * i];
            sdm_step(p, cur, next, x);
            cur->next = (uint8_t)sdm_hist_get(p, cur->hist, next_pos);
            cur->hist_used = 0;
        }
    } else
#endif
    for (i = 0; i < p->num_cands; i++) {
        sdm_state_t *cur  = st_cur->act[i];
        sdm_state_t *next = &st_next->sdm[p->mb_levels * i];
        sdm_step(p, cur, next, x);
        cur->next = (uint8_t)sdm_hist_get(p, cur->hist, next_pos);
        cur->hist_used = 0;
    }

    new_cands = sdm_sort_cands(p, st_next);
    min_cost = st_next->act[0]->cost;
    output = st_next->act[0]->next;

    for (i = 0; i < new_cands; i++) {
        sdm_state_t *s = st_next->act[i];
        if (s->parent->hist_used) {
            unsigned h = sdm_histbuf_get(p);
            sdm_hist_copy(p, h, s->hist);
            s->hist = (uint8_t)h;
        } else {
            s->parent->hist_used = 1;
        }

        s->cost -= min_cost;
        s->next = s->parent->next;
        sdm_hist_put(p, s->hist, p->pos, s->path & 1);
    }

    for (i = 0; i < p->num_cands; i++) {
        sdm_state_t *s = st_cur->act[i];
        if (!s->hist_used)
            sdm_histbuf_put(p, s->hist);
    }

    if (new_cands < p->num_cands) {
        p->conv_fail++;
        p->cands_collapse++;
    }

    p->num_cands = new_cands;
    p->pos = next_pos;
    p->idx ^= 1;

    return output ? 1.0 : -1.0;
}

/* ─── Public API ─── */

int sdm_context_init(sdm_context_t *ctx, const ntf_filter_t *filter,
                     int trellis_depth, int trellis_cands, int trellis_lat) {
    memset(ctx, 0, sizeof(*ctx));

    if (!filter)
        return -1;

    if (trellis_depth > SDM_TRELLIS_MAX_ORDER ||
        trellis_cands > SDM_TRELLIS_MAX_NUM ||
        trellis_lat > SDM_TRELLIS_MAX_LAT)
        return -1;

    if (trellis_depth < 1 || trellis_cands < 1 || trellis_lat < 1)
        return -1;

    ctx->filter = filter;
    ctx->trellis_num = (uint32_t)trellis_cands;
    ctx->trellis_lat = (uint32_t)trellis_lat;
    ctx->trellis_mask = ((uint64_t)1 << trellis_depth) - 1;
    ctx->num_cands = 1;
    ctx->mb_levels = 2;  /* default: 1-bit (2 children). Set to SDM_MB_LEVELS for multibit. */
    ctx->state_limit = 0.0;  /* 0 = disabled; set by caller if needed */

    /* Init history buffer free list */
    for (unsigned i = 0; i < 2u * (unsigned)trellis_cands; i++)
        ctx->hist_free[ctx->hist_fnum++] = (uint8_t)i;

    /* Init first candidate */
    sdm_trellis_t *st = &ctx->trellis[0];
    st->sdm[0].hist = (uint8_t)sdm_histbuf_get(ctx);
    st->sdm[0].path = 0;
    st->act[0] = &st->sdm[0];

    return 0;
}

size_t sdm_process_block(sdm_context_t *ctx,
                         const double *in, float *out, size_t count) {
    float *outp = out;
    size_t len = count;

    /* Select per-sample function based on branching mode */
    double (*sample_fn)(sdm_context_t *, double) =
        (ctx->mb_levels > 2) ? sdm_sample_trellis_mb : sdm_sample_trellis;

    /* Fill latency buffer first (no output produced) */
    if (ctx->pending < ctx->trellis_lat) {
        size_t pre = ctx->trellis_lat - ctx->pending;
        if (pre > len)
            pre = len;
        ctx->pending += (unsigned)pre;
        len -= pre;
        while (pre--) {
            sample_fn(ctx, *in++);
        }
    }

    /* Produce output samples */
    while (len--) {
        *outp++ = (float)sample_fn(ctx, *in++);
    }

    return (size_t)(outp - out);
}

size_t sdm_drain(sdm_context_t *ctx, float *out, size_t max_out) {
    size_t len = ctx->pending;
    if (len > max_out)
        len = max_out;

    /* If we haven't started draining yet and didn't fully fill latency,
       flush the remaining latency slots by feeding zeros */
    if (!ctx->draining && ctx->pending < ctx->trellis_lat) {
        unsigned flush = ctx->trellis_lat - ctx->pending;
        while (flush--)
            sdm_sample_trellis(ctx, 0.0);
    }

    ctx->draining = 1;
    ctx->pending -= (unsigned)len;

    for (size_t i = 0; i < len; i++)
        out[i] = (float)sdm_sample_trellis(ctx, 0.0);

    return len;
}

void sdm_context_copy_state(sdm_context_t *dst, const sdm_context_t *src) {
    /* sdm_context_t has no external pointers — full memcpy is a valid deep copy.
     * But internal pointers (act[], path_hash[]) point into the source's arrays
     * and need fixing up or clearing. */
    memcpy(dst, src, sizeof(*dst));

    /* Fix up act[] pointers in both trellis banks.
     * The SDM ping-pongs between trellis[0] and trellis[1]. */
    for (int bank = 0; bank < 2; bank++) {
        unsigned nc = dst->num_cands;
        if (nc > SDM_TRELLIS_MAX_NUM) nc = SDM_TRELLIS_MAX_NUM;
        for (unsigned i = 0; i < nc; i++) {
            if (src->trellis[bank].act[i]) {
                ptrdiff_t offset = src->trellis[bank].act[i] - &src->trellis[bank].sdm[0];
                dst->trellis[bank].act[i] = &dst->trellis[bank].sdm[0] + offset;
            }
        }
    }

    /* Clear path_hash — contains stale pointers into src's trellis.
     * Will be rebuilt on the first sample processing step. */
    memset(dst->path_hash, 0, sizeof(dst->path_hash));

    /* Reset diagnostic counters for this segment */
    dst->conv_fail = 0;
    dst->cands_collapse = 0;
    dst->next_filter_drops = 0;
    dst->total_children = 0;
}

double sdm_state_distance(const sdm_context_t *a, const sdm_context_t *b) {
    /* Compare integrator states of the best (lowest-cost) candidate in each
     * context. The active bank alternates each sample via idx. */
    int bank_a = a->idx & 1;
    int bank_b = b->idx & 1;
    const sdm_state_t *sa = a->trellis[bank_a].act[0];
    const sdm_state_t *sb = b->trellis[bank_b].act[0];
    if (!sa || !sb) return 1e30;
    int order = a->filter->order;
    double dist = 0.0;
    for (int i = 0; i < order; i++) {
        double d = sa->state[i] - sb->state[i];
        dist += d * d;
    }
    return dist;
}

/* Multi-bit greedy SDM estimator.
 * Runs the NTF filter forward from init_state with a multi-level quantizer.
 * 16 quantizer levels (4-bit) tracks the true trellis state much more
 * accurately than 2-level (1-bit), giving better segment boundary estimates
 * for DAS parallel seeding. Same computational cost — one NTF evaluation
 * per sample, ~16 flops for order 8.
 * num_levels: 2 = 1-bit (legacy), 16 = 4-bit (default). */
void sdm_estimate_state(const ntf_filter_t *filter, const double *init_state,
                        const double *in, size_t count, double state_limit,
                        double *out_state) {
    sdm_estimate_state_multibit(filter, init_state, in, count, state_limit,
                                 16, out_state);
}

void sdm_estimate_state_multibit(const ntf_filter_t *filter,
                                  const double *init_state,
                                  const double *in, size_t count,
                                  double state_limit, int num_levels,
                                  double *out_state) {
    const int order = filter->order;
    const double *a = filter->a;
    const double *g = filter->g;
    double state[MAX_NTF_ORDER];
    double new_state[MAX_NTF_ORDER];
    const double step = 2.0 / (double)(num_levels - 1);

    for (int i = 0; i < order; i++)
        state[i] = init_state[i];

    for (size_t n = 0; n < count; n++) {
        double x = in[n];

        /* NTF filter calc with y=0 (generic order) */
        new_state[0] = state[0] - g[0] * state[1] + x;
        double v = x + a[0] * new_state[0];

        for (int i = 1; i < order - 1; i++) {
            new_state[i] = state[i] + state[i - 1] - g[i] * state[i + 1];
            v += a[i] * new_state[i];
        }
        new_state[order - 1] = state[order - 1] + state[order - 2];
        v += a[order - 1] * new_state[order - 1];

        /* Multi-bit greedy quantizer: minimize (v - y*a[0])².
         * Optimal y = v / a[0], rounded to nearest quantizer level. */
        double y;
        if (num_levels <= 2) {
            /* 1-bit: y ∈ {-1, +1}. Pick y=+1 when v/a[0] > 0. */
            y = (v * a[0] > 0.0) ? 1.0 : -1.0;
        } else {
            /* Multi-bit: round v/a[0] to nearest level in [-1, +1] */
            double y_ideal = v / a[0];
            int idx = (int)((y_ideal + 1.0) / step + 0.5);
            if (idx < 0) idx = 0;
            if (idx > num_levels - 1) idx = num_levels - 1;
            y = -1.0 + (double)idx * step;
        }

        /* Update state: NTF formula d[0] = s[0] - g[0]*s[1] + x - y
         * new_state was computed with y=0, so subtract y now. */
        new_state[0] -= y;

        /* State limit clamping */
        if (state_limit > 0.0) {
            for (int i = 0; i < order; i++) {
                if (new_state[i] > state_limit) new_state[i] = state_limit;
                else if (new_state[i] < -state_limit) new_state[i] = -state_limit;
            }
        }

        for (int i = 0; i < order; i++)
            state[i] = new_state[i];
    }

    for (int i = 0; i < order; i++)
        out_state[i] = state[i];
}

void sdm_context_reset(sdm_context_t *ctx) {
    if (!ctx->filter)
        return;

    const ntf_filter_t *f = ctx->filter;
    uint32_t mask = ctx->trellis_mask;
    uint32_t num = ctx->trellis_num;
    uint32_t lat = ctx->trellis_lat;

    memset(ctx->trellis, 0, sizeof(ctx->trellis));
    memset(ctx->path_hash, 0, sizeof(ctx->path_hash));
    memset(ctx->hist, 0, sizeof(ctx->hist));

    ctx->hist_fnum = 0;
    ctx->num_cands = 1;
    /* Preserve mb_levels across reset (set by caller) */
    ctx->pos = 0;
    ctx->pending = 0;
    ctx->draining = 0;
    ctx->idx = 0;
    ctx->prev_y = 0.0;
    ctx->conv_fail = 0;

    ctx->filter = f;
    ctx->trellis_mask = mask;
    ctx->trellis_num = num;
    ctx->trellis_lat = lat;

    for (unsigned i = 0; i < 2u * num; i++)
        ctx->hist_free[ctx->hist_fnum++] = (uint8_t)i;

    sdm_trellis_t *st = &ctx->trellis[0];
    st->sdm[0].hist = (uint8_t)sdm_histbuf_get(ctx);
    st->sdm[0].path = 0;
    st->act[0] = &st->sdm[0];
}

void sdm_context_free(sdm_context_t *ctx) {
    memset(ctx, 0, sizeof(*ctx));
}

#if defined(_MSC_VER) && !defined(SDM_FAST_MODE)
#pragma float_control(pop)
#endif
