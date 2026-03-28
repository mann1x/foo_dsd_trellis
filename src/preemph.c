/*
 * Adaptive pre-SDM pre-emphasis: tiny embedded MLP (387 params).
 *
 * Predicts optimal 3-tap FIR from signal features:
 *   Input:  [spectral_centroid_hz, rms, crest_factor]
 *   Output: [tap0, tap1, tap2]
 *
 * Architecture: Linear(3→16) → ReLU → Linear(16→16) → ReLU → Linear(16→3)
 * with baked-in feature normalization.
 *
 * Trained via CMA-ES black-box optimization with real trellis SDM across
 * 20 signal types (50-14000 Hz, varying amplitude and spectral content).
 */

#include <math.h>
#include <string.h>
#include "preemph_model.h"

#define PREEMPH_HIDDEN 16
#define PREEMPH_FEATURES 3
#define PREEMPH_TAPS 3


void preemph_predict_taps(float spectral_centroid, float rms, float crest_factor,
                           float taps_out[3]) {
    float features[PREEMPH_FEATURES];
    float h1[PREEMPH_HIDDEN], h2[PREEMPH_HIDDEN];

    /* Normalize features */
    features[0] = (spectral_centroid - preemph_feat_mean[0]) / preemph_feat_std[0];
    features[1] = (rms - preemph_feat_mean[1]) / preemph_feat_std[1];
    features[2] = (crest_factor - preemph_feat_mean[2]) / preemph_feat_std[2];

    /* Layer 1: Linear(3→16) + ReLU */
    for (int i = 0; i < PREEMPH_HIDDEN; i++) {
        float sum = preemph_b0[i];
        for (int j = 0; j < PREEMPH_FEATURES; j++)
            sum += preemph_w0[i * PREEMPH_FEATURES + j] * features[j];
        h1[i] = sum > 0.0f ? sum : 0.0f;  /* ReLU */
    }

    /* Layer 2: Linear(16→16) + ReLU */
    for (int i = 0; i < PREEMPH_HIDDEN; i++) {
        float sum = preemph_b1[i];
        for (int j = 0; j < PREEMPH_HIDDEN; j++)
            sum += preemph_w1[i * PREEMPH_HIDDEN + j] * h1[j];
        h2[i] = sum > 0.0f ? sum : 0.0f;  /* ReLU */
    }

    /* Layer 3: Linear(16→3) — output taps */
    for (int i = 0; i < PREEMPH_TAPS; i++) {
        float sum = preemph_b2[i];
        for (int j = 0; j < PREEMPH_HIDDEN; j++)
            sum += preemph_w2[i * PREEMPH_HIDDEN + j] * h2[j];
        taps_out[i] = sum;
    }

    /* Normalize: force tap[0] = 1.0 to prevent gain changes.
     * Scale all taps so the DC gain (sum of taps) = 1.0.
     * This preserves the pre-emphasis shape without volume pumping. */
    float dc_gain = 0.0f;
    for (int i = 0; i < PREEMPH_TAPS; i++)
        dc_gain += taps_out[i];
    if (dc_gain > 0.01f) {
        float scale = 1.0f / dc_gain;
        for (int i = 0; i < PREEMPH_TAPS; i++)
            taps_out[i] *= scale;
    }
}


void preemph_apply(double *buf, size_t count, const float taps[3]) {
    /* Apply 3-tap causal FIR: y[n] = t0*x[n] + t1*x[n-1] + t2*x[n-2] */
    if (count < 3) return;

    double t0 = (double)taps[0];
    double t1 = (double)taps[1];
    double t2 = (double)taps[2];

    /* Check if taps are near-identity — skip if so */
    if (fabs(t0 - 1.0) < 0.001 && fabs(t1) < 0.001 && fabs(t2) < 0.001)
        return;

    /* In-place: process backwards to avoid overwriting unread values */
    /* Actually, for a causal FIR we need previous values, so use a temp */
    double prev1 = buf[0], prev2 = 0.0;
    double cur = buf[0];
    buf[0] = t0 * cur;  /* no history for first sample */

    cur = buf[1];
    buf[1] = t0 * cur + t1 * prev1;
    prev2 = prev1;
    prev1 = cur;

    for (size_t i = 2; i < count; i++) {
        cur = buf[i];
        buf[i] = t0 * cur + t1 * prev1 + t2 * prev2;
        prev2 = prev1;
        prev1 = cur;
    }
}


float preemph_spectral_centroid(const double *buf, size_t count, double sample_rate) {
    /* Estimate spectral centroid from zero-crossing rate.
     * ZCR ≈ 2 * centroid / sample_rate for narrowband signals.
     * Fast O(n) approximation — no FFT needed. */
    if (count < 2) return 1000.0f;

    size_t crossings = 0;
    for (size_t i = 1; i < count; i++) {
        if ((buf[i] >= 0.0) != (buf[i-1] >= 0.0))
            crossings++;
    }

    float zcr = (float)crossings / (float)(count - 1);
    float centroid = zcr * (float)sample_rate / 2.0f;

    /* Clamp to reasonable range */
    if (centroid < 20.0f) centroid = 20.0f;
    if (centroid > 20000.0f) centroid = 20000.0f;
    return centroid;
}


float preemph_rms(const double *buf, size_t count) {
    if (count == 0) return 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < count; i++)
        sum += buf[i] * buf[i];
    return (float)sqrt(sum / (double)count);
}


float preemph_crest_factor(const double *buf, size_t count) {
    if (count == 0) return 1.0f;
    double sum = 0.0, peak = 0.0;
    for (size_t i = 0; i < count; i++) {
        double a = fabs(buf[i]);
        sum += buf[i] * buf[i];
        if (a > peak) peak = a;
    }
    float rms = (float)sqrt(sum / (double)count);
    if (rms < 1e-10f) return 1.0f;
    return (float)peak / rms;
}
