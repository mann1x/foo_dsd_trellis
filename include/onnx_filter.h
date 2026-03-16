/*
 * foo_dsd_trellis — ONNX Runtime ML post-filter for DSD noise reduction
 *
 * Causal 1D CNN operating on ±1.0 DSD bitstream.
 * Delay-loads onnxruntime.dll — plugin works without it.
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
    ML_EP_AUTO     = 2,  /* Try DirectML first, fall back to CPU */
} ml_ep_t;

/* Opaque filter handle */
typedef struct onnx_filter onnx_filter_t;

/* Check if ONNX Runtime DLL is available (delay-load probe).
 * Thread-safe, caches result after first call. */
bool onnx_runtime_available(void);

/* Create filter from .onnx model file.
 * dsd_rate: target DSD sample rate (for rate-specific models).
 * ep: execution provider (CPU or DirectML).
 * Returns NULL on failure (DLL missing, model invalid, etc.). */
onnx_filter_t *onnx_filter_create(const wchar_t *model_path,
                                   uint32_t dsd_rate, ml_ep_t ep);

/* Process DSD samples in-place. Maintains causal state across calls.
 * buf contains ±1.0 DSD samples. count = number of samples.
 * Reads buf[], runs inference, writes refined ±1.0 back to buf[]. */
void onnx_filter_process(onnx_filter_t *f, float *buf, size_t count);

/* Reset causal state (seek / track change). */
void onnx_filter_reset(onnx_filter_t *f);

/* Free all resources. Safe to call with NULL. */
void onnx_filter_free(onnx_filter_t *f);

#endif /* ONNX_FILTER_H */
