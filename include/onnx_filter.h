/*
 * foo_dsd_trellis — ONNX Runtime ML filter
 *
 * Two modes:
 *   1. Post-SDM: causal 1D CNN on ±1.0 DSD bitstream (legacy)
 *   2. Pre-SDM:  full-pipeline pre-emphasis (features → MLP → FIR) on boxcar output
 *
 * Delay-loads onnxruntime.dll — plugin works without it.
 * Supports CUDA, DirectML, and CPU execution providers.
 */

#ifndef ONNX_FILTER_H
#define ONNX_FILTER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* Execution provider */
typedef enum {
    ML_EP_CPU      = 0,
    ML_EP_DIRECTML = 1,
    ML_EP_AUTO     = 2,  /* Try CUDA → DirectML → CPU */
    ML_EP_CUDA     = 3,
} ml_ep_t;

/* Opaque filter handle */
typedef struct onnx_filter onnx_filter_t;

/* Check if ONNX Runtime DLL is available (delay-load probe).
 * Thread-safe, caches result after first call. */
bool onnx_runtime_available(void);

/* Create filter from .onnx model file.
 * dsd_rate: target DSD sample rate (for rate-specific models).
 * ep: execution provider preference.
 * Returns NULL on failure (DLL missing, model invalid, etc.). */
onnx_filter_t *onnx_filter_create(const wchar_t *model_path,
                                   uint32_t dsd_rate, ml_ep_t ep);

/* Process DSD samples in-place (post-SDM mode).
 * buf contains ±1.0 DSD samples. count = number of samples.
 * Reads buf[], runs inference, writes refined ±1.0 back to buf[]. */
void onnx_filter_process(onnx_filter_t *f, float *buf, size_t count);

/* Process boxcar output in-place (pre-SDM full-pipeline mode).
 * buf contains fp64 boxcar samples. count = number of samples.
 * The full-pipeline model runs features → MLP → FIR entirely on GPU.
 * Converts double↔float32 for ONNX tensors internally. */
void onnx_filter_process_preemph(onnx_filter_t *f, double *buf, size_t count);

/* Predict pre-emphasis taps from signal features (tap-prediction mode).
 * features: [spectral_centroid, rms, crest_factor] (3 floats).
 * taps_out: [tap0, tap1, tap2] (3 floats, DC-gain-normalized).
 * Runs the tiny MLP on GPU — sub-millisecond inference. */
void onnx_filter_predict_taps(onnx_filter_t *f, const float features[3],
                               float taps_out[3]);

/* Query which execution provider the session is actually using.
 * Returns descriptive string like "CUDA", "DirectML", "CPU". */
const char *onnx_filter_ep_name(const onnx_filter_t *f);

/* Reset causal state (seek / track change). */
void onnx_filter_reset(onnx_filter_t *f);

/* Free all resources. Safe to call with NULL. */
void onnx_filter_free(onnx_filter_t *f);

#endif /* ONNX_FILTER_H */
