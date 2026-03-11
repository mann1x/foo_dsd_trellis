/*
 * foo_dsd_trellis — Runtime CPU feature detection
 */

#ifndef SIMD_DETECT_H
#define SIMD_DETECT_H

#include <stdint.h>
#include <stdbool.h>

/* CPU vendor */
typedef enum {
    CPU_VENDOR_UNKNOWN = 0,
    CPU_VENDOR_INTEL,
    CPU_VENDOR_AMD
} cpu_vendor_t;

/* Detected feature flags */
typedef struct {
    cpu_vendor_t vendor;
    bool sse2;
    bool avx2;
    bool fma3;
    bool avx512f;     /* AVX-512 Foundation */
    int  family;      /* CPU family (for Zen detection) */
    int  model;       /* CPU model */
} cpu_features_t;

/* Detect CPU features (call once at startup, result is cached) */
const cpu_features_t *cpu_detect(void);

/* Query whether to prefer 128-bit AVX over 256-bit (AMD Zen 1/2) */
bool cpu_prefer_128bit_avx(void);

#endif /* SIMD_DETECT_H */
