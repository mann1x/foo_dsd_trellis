/*
 * foo_dsd_trellis — SIMD-optimized FIR convolution kernels
 *
 * When USE_IPP is defined, delegates to Intel IPP for auto-dispatched
 * SIMD (SSE2→AVX2→AVX-512). Otherwise falls back to hand-rolled kernels.
 *
 * The FIR inner loop computes:
 *   sum = sum(coeffs[j] * delay[(pos-j) & mask], j=0..ntaps-1)
 * where coeffs are double, delay are float, ntaps=12.
 */

#include "../include/simd_detect.h"
#include "../include/fir.h"
#include <string.h>

#ifdef USE_IPP
#include <ipps.h>
#include <ippcore.h>
#endif

#ifdef _MSC_VER
#include <immintrin.h>
#endif

/* Number of taps in the polyphase phase 0 filter */
#define NTAPS 12

/*
 * Linearize circular delay buffer for SIMD-friendly access.
 * Copies ntaps floats from the circular buffer starting at pos,
 * going backwards (pos, pos-1, ..., pos-ntaps+1).
 */
static inline void linearize_delay(const float *delay, int pos, int ntaps,
                                   float *linear) {
    for (int j = 0; j < ntaps; j++)
        linear[j] = delay[(pos - j) & (FIR_MAX_PHASE_TAPS - 1)];
}

/* ─── Scalar reference (always available) ─── */

double fir_convolve_scalar(const double *coeffs, const float *delay,
                           int pos, int ntaps) {
    double sum = 0.0;
    for (int j = 0; j < ntaps; j++) {
        int idx = (pos - j) & (FIR_MAX_PHASE_TAPS - 1);
        sum += coeffs[j] * (double)delay[idx];
    }
    return sum;
}

/* ─── SSE2 kernel (2 doubles at a time) ─── */

#if defined(_MSC_VER) || defined(__SSE2__)
double fir_convolve_sse2(const double *coeffs, const float *delay,
                         int pos, int ntaps) {
    float linear[FIR_MAX_PHASE_TAPS];
    linearize_delay(delay, pos, ntaps, linear);

    __m128d sum = _mm_setzero_pd();

    int j = 0;
    for (; j + 2 <= ntaps; j += 2) {
        __m128d c = _mm_loadu_pd(&coeffs[j]);
        /* Convert 2 floats to 2 doubles */
        __m128 f = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *)&linear[j]));
        __m128d d = _mm_cvtps_pd(f);
        sum = _mm_add_pd(sum, _mm_mul_pd(c, d));
    }

    /* Horizontal add */
    double result[2];
    _mm_storeu_pd(result, sum);
    double total = result[0] + result[1];

    /* Handle remaining sample (odd ntaps) */
    for (; j < ntaps; j++)
        total += coeffs[j] * (double)linear[j];

    return total;
}
#endif

/* ─── AVX2 + FMA kernel (4 doubles at a time) ─── */

#if defined(_MSC_VER) || defined(__AVX2__)
double fir_convolve_avx2(const double *coeffs, const float *delay,
                         int pos, int ntaps) {
    float linear[FIR_MAX_PHASE_TAPS];
    linearize_delay(delay, pos, ntaps, linear);

    __m256d sum = _mm256_setzero_pd();

    int j = 0;
    for (; j + 4 <= ntaps; j += 4) {
        __m256d c = _mm256_loadu_pd(&coeffs[j]);
        __m128 f = _mm_loadu_ps(&linear[j]);
        __m256d d = _mm256_cvtps_pd(f);
        /* FMA: single rounding, 2× throughput vs separate mul+add.
         * This function is only called after runtime FMA3 detection.
         * Note: __FMA__ is GCC/Clang only; MSVC defines __AVX2__ but not __FMA__. */
        sum = _mm256_fmadd_pd(c, d, sum);
    }

    /* Reduce 256-bit to scalar */
    __m128d lo = _mm256_castpd256_pd128(sum);
    __m128d hi = _mm256_extractf128_pd(sum, 1);
    __m128d s = _mm_add_pd(lo, hi);
    double result[2];
    _mm_storeu_pd(result, s);
    double total = result[0] + result[1];

    /* Remainder */
    for (; j < ntaps; j++)
        total += coeffs[j] * (double)linear[j];

    return total;
}
#endif

/* ─── AMD Zen 1/2 optimized: use 128-bit AVX (no 256-bit cracking penalty) ─── */

#if defined(_MSC_VER) || defined(__AVX2__)
double fir_convolve_avx128(const double *coeffs, const float *delay,
                           int pos, int ntaps) {
    float linear[FIR_MAX_PHASE_TAPS];
    linearize_delay(delay, pos, ntaps, linear);

    __m128d sum0 = _mm_setzero_pd();
    __m128d sum1 = _mm_setzero_pd();

    int j = 0;
    for (; j + 4 <= ntaps; j += 4) {
        __m128d c0 = _mm_loadu_pd(&coeffs[j]);
        __m128d c1 = _mm_loadu_pd(&coeffs[j + 2]);

        __m128 f = _mm_loadu_ps(&linear[j]);
        __m128d d0 = _mm_cvtps_pd(f);
        __m128d d1 = _mm_cvtps_pd(_mm_movehl_ps(f, f));

        /* Use VEX-encoded FMA (128-bit, no cracking on Zen 1/2) */
        sum0 = _mm_fmadd_pd(c0, d0, sum0);
        sum1 = _mm_fmadd_pd(c1, d1, sum1);
    }

    __m128d s = _mm_add_pd(sum0, sum1);
    double result[2];
    _mm_storeu_pd(result, s);
    double total = result[0] + result[1];

    for (; j < ntaps; j++)
        total += coeffs[j] * (double)linear[j];

    return total;
}
#endif

/* ─── Intel IPP kernel (auto-dispatched SIMD) ─── */

#ifdef USE_IPP
static int g_ipp_initialized = 0;

double fir_convolve_ipp(const double *coeffs, const float *delay,
                        int pos, int ntaps) {
    float linear[FIR_MAX_PHASE_TAPS];
    linearize_delay(delay, pos, ntaps, linear);

    /* Convert float delay to double for IPP dot product */
    Ipp64f delay_d[FIR_MAX_PHASE_TAPS];
    ippsConvert_32f64f(linear, delay_d, ntaps);

    Ipp64f result = 0.0;
    ippsDotProd_64f(coeffs, delay_d, ntaps, &result);
    return result;
}
#endif

/* ─── Function pointer dispatch ─── */

typedef double (*fir_convolve_fn)(const double *, const float *, int, int);

static fir_convolve_fn g_convolve_fn = NULL;

void fir_simd_init(void) {
#ifdef USE_IPP
    if (!g_ipp_initialized) {
        ippInit();
        g_ipp_initialized = 1;
    }
    g_convolve_fn = fir_convolve_ipp;
#else
    const cpu_features_t *cpu = cpu_detect();

    if (cpu->avx2 && cpu->fma3) {
        if (cpu_prefer_128bit_avx()) {
            /* AMD Zen 1/2: use 128-bit FMA path */
            g_convolve_fn = fir_convolve_avx128;
        } else {
            /* Intel or AMD Zen 3+: full 256-bit AVX2 */
            g_convolve_fn = fir_convolve_avx2;
        }
    } else if (cpu->sse2) {
        g_convolve_fn = fir_convolve_sse2;
    } else {
        g_convolve_fn = fir_convolve_scalar;
    }
#endif
}

double fir_convolve_dispatch(const double *coeffs, const float *delay,
                             int pos, int ntaps) {
    if (!g_convolve_fn)
        fir_simd_init();
    return g_convolve_fn(coeffs, delay, pos, ntaps);
}

const char *fir_simd_name(void) {
    if (!g_convolve_fn)
        fir_simd_init();

#ifdef USE_IPP
    return "IPP (auto-dispatch)";
#else
    if (g_convolve_fn == fir_convolve_avx2)    return "AVX2+FMA";
    if (g_convolve_fn == fir_convolve_avx128)  return "AVX128+FMA (Zen)";
    if (g_convolve_fn == fir_convolve_sse2)    return "SSE2";
    return "Scalar";
#endif
}

#ifdef USE_IPP
const char *fir_ipp_version(void) {
    if (!g_ipp_initialized) {
        ippInit();
        g_ipp_initialized = 1;
    }
    const IppLibraryVersion *ver = ippsGetLibVersion();
    return ver ? ver->Version : "unknown";
}
#endif
