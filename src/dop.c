/*
 * foo_dsd_trellis — DoP (DSD over PCM) detection, pack/unpack
 *
 * DoP encodes DSD data into 24-bit PCM frames:
 *   - Upper 8 bits: alternating marker 0x05 / 0xFA per frame
 *   - Lower 16 bits: 16 DSD samples (MSB first)
 *
 * foobar2000 delivers PCM24 as float32 normalised to [-1.0, 1.0].
 * Conversion: int24 = round(float * 2^23)
 */

#include "../include/dop.h"
#include <string.h>
#include <math.h>

#define SCALE_23 8388608.0f  /* 2^23 */

/* Convert float PCM sample to signed 24-bit integer */
static inline int32_t float_to_int24(float f) {
    double d = (double)f * (double)SCALE_23;
    if (d >= 0.0)
        return (int32_t)(d + 0.5);
    else
        return (int32_t)(d - 0.5);
}

/* Convert signed 24-bit integer to float PCM sample */
static inline float int24_to_float(int32_t i) {
    return (float)((double)i / (double)SCALE_23);
}

bool dop_detect(const float *pcm24, size_t frames) {
    if (!pcm24 || frames < 2)
        return false;

    /* Scan up to 8 frames for alternating DoP markers */
    size_t check = frames < 8 ? frames : 8;

    for (size_t i = 0; i < check; i++) {
        int32_t val = float_to_int24(pcm24[i]);
        /* Extract marker byte: upper 8 bits of 24-bit word.
         * For signed 24-bit, we need the unsigned representation. */
        uint32_t uval = (uint32_t)val & 0x00FFFFFFu;
        uint8_t marker = (uint8_t)(uval >> 16);

        uint8_t expected = (i & 1) ? DOP_MARKER_B : DOP_MARKER_A;
        if (marker != expected)
            return false;
    }

    return true;
}

void dop_unpack(const float *pcm24, float *bits, size_t frames) {
    if (!pcm24 || !bits || frames == 0)
        return;

    for (size_t i = 0; i < frames; i++) {
        int32_t val = float_to_int24(pcm24[i]);
        uint16_t dsd_bits = (uint16_t)(val & 0xFFFF);

        /* Extract 16 DSD bits, MSB first → ±1.0f */
        for (int b = 15; b >= 0; b--) {
            *bits++ = (dsd_bits >> b) & 1 ? 1.0f : -1.0f;
        }
    }
}

void dop_pack(const float *bits, float *pcm24, size_t bit_count) {
    if (!bits || !pcm24 || bit_count == 0)
        return;

    size_t frames = bit_count / 16;

    for (size_t i = 0; i < frames; i++) {
        uint16_t dsd_bits = 0;

        /* Pack 16 float ±1.0 → 16 bits, MSB first */
        for (int b = 15; b >= 0; b--) {
            if (*bits++ >= 0.0f)
                dsd_bits |= (uint16_t)(1u << b);
        }

        /* Add DoP marker in upper 8 bits */
        uint8_t marker = (i & 1) ? DOP_MARKER_B : DOP_MARKER_A;
        int32_t val = ((int32_t)marker << 16) | (int32_t)dsd_bits;

        pcm24[i] = int24_to_float(val);
    }
}

void bits_unpack(const uint8_t *src, float *dst, size_t n_bits) {
    if (!src || !dst || n_bits == 0)
        return;

    size_t full_bytes = n_bits / 8;
    size_t remaining = n_bits % 8;

    for (size_t i = 0; i < full_bytes; i++) {
        uint8_t byte = src[i];
        for (int b = 7; b >= 0; b--) {
            *dst++ = (byte >> b) & 1 ? 1.0f : -1.0f;
        }
    }

    if (remaining > 0) {
        uint8_t byte = src[full_bytes];
        for (size_t b = 0; b < remaining; b++) {
            *dst++ = (byte >> (7 - b)) & 1 ? 1.0f : -1.0f;
        }
    }
}

void bits_pack(const float *src, uint8_t *dst, size_t n_bits) {
    if (!src || !dst || n_bits == 0)
        return;

    size_t full_bytes = n_bits / 8;
    size_t remaining = n_bits % 8;

    for (size_t i = 0; i < full_bytes; i++) {
        uint8_t byte = 0;
        for (int b = 7; b >= 0; b--) {
            if (*src++ >= 0.0f)
                byte |= (uint8_t)(1u << b);
        }
        dst[i] = byte;
    }

    if (remaining > 0) {
        uint8_t byte = 0;
        for (size_t b = 0; b < remaining; b++) {
            if (*src++ >= 0.0f)
                byte |= (uint8_t)(1u << (7 - b));
        }
        dst[full_bytes] = byte;
    }
}
