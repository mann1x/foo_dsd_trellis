/*
 * foo_dsd_trellis — Minimal WAV file read/write
 * Supports PCM 16/24/32-bit and 32-bit float.
 */

#ifndef WAV_IO_H
#define WAV_IO_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float      *samples;       /* Interleaved float32 samples, [-1.0, 1.0] */
    uint32_t    sample_rate;
    uint16_t    channels;
    uint32_t    num_frames;    /* Frames per channel */
} wav_data_t;

/* Read a WAV file into wav_data_t. Caller must call wav_free().
 * Returns 0 on success, -1 on error. */
int wav_read(const char *path, wav_data_t *wav);

/* Write float32 samples as 32-bit float WAV.
 * Returns 0 on success, -1 on error. */
int wav_write(const char *path, const float *samples, uint32_t num_frames,
              uint16_t channels, uint32_t sample_rate);

/* Free wav_data_t samples buffer. */
void wav_free(wav_data_t *wav);

#ifdef __cplusplus
}
#endif

#endif /* WAV_IO_H */
