/*
 * foo_dsd_trellis — DoP (DSD over PCM) detection and pack/unpack
 */

#ifndef DOP_H
#define DOP_H

#include "dsd_types.h"

/* DoP marker bytes (alternating per frame) */
#define DOP_MARKER_A 0x05u
#define DOP_MARKER_B 0xFAu

/* Detect DoP markers in a PCM24 stream (single channel, contiguous).
 * Scans first 8 frames for alternating 0x05/0xFA in MSB of 24-bit words.
 * Returns true if DoP is detected. */
bool dop_detect(const float *pcm24, size_t frames);

/* Detect DoP markers in interleaved multi-channel PCM24.
 * Checks channel 0 at the given stride (= num_channels). */
bool dop_detect_interleaved(const float *pcm24, size_t frames, int channels);

/* Unpack DoP PCM24 frames to float32 ±1.0 DSD bits.
 * Extracts 16 DSD bits per PCM frame.
 * Output buffer must hold frames * 16 floats. */
void dop_unpack(const float *pcm24, float *bits, size_t frames);

/* Pack float32 ±1.0 DSD bits into DoP PCM24 frames.
 * Consumes 16 DSD bits per output PCM frame.
 * pcm24 buffer must hold bit_count / 16 frames. */
void dop_pack(const float *bits, float *pcm24, size_t bit_count);

/* Unpack raw bitstream bytes to float32 ±1.0 (native ASIO path).
 * Each bit becomes one float: 1 → +1.0f, 0 → -1.0f */
void bits_unpack(const uint8_t *src, float *dst, size_t n_bits);

/* Pack float32 ±1.0 to raw bitstream bytes (native ASIO path).
 * Each float becomes one bit: ≥0 → 1, <0 → 0 */
void bits_pack(const float *src, uint8_t *dst, size_t n_bits);

#endif /* DOP_H */
