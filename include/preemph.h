#ifndef PREEMPH_H
#define PREEMPH_H

#include <stddef.h>

/* Adaptive pre-SDM pre-emphasis: embedded MLP (387 params, ~420 FLOPs).
 * Predicts optimal 3-tap FIR from signal features, then applies it.
 * Trained via CMA-ES with real trellis SDM across 20 signal types. */

/* Predict optimal FIR taps from signal features */
void preemph_predict_taps(float spectral_centroid, float rms, float crest_factor,
                           float taps_out[3]);

/* Apply 3-tap causal FIR pre-emphasis in-place */
void preemph_apply(double *buf, size_t count, const float taps[3]);

/* Feature extractors (fast, O(n)) */
float preemph_spectral_centroid(const double *buf, size_t count, double sample_rate);
float preemph_rms(const double *buf, size_t count);
float preemph_crest_factor(const double *buf, size_t count);

#endif /* PREEMPH_H */
