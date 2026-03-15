/*
 * foo_dsd_trellis — Minimal WAV file read/write
 * Handles PCM 16/24/32 and IEEE float 32.
 */

#define _CRT_SECURE_NO_WARNINGS
#include "../include/wav_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* WAV chunk IDs */
#define RIFF_ID  0x46464952u  /* "RIFF" */
#define WAVE_ID  0x45564157u  /* "WAVE" */
#define FMT_ID   0x20746D66u  /* "fmt " */
#define DATA_ID  0x61746164u  /* "data" */

#define WAV_PCM        1
#define WAV_IEEE_FLOAT 3

#pragma pack(push, 1)
typedef struct {
    uint32_t riff_id;
    uint32_t file_size;
    uint32_t wave_id;
} riff_header_t;

typedef struct {
    uint32_t chunk_id;
    uint32_t chunk_size;
} chunk_header_t;

typedef struct {
    uint16_t format;
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
} fmt_chunk_t;
#pragma pack(pop)

int wav_read(const char *path, wav_data_t *wav) {
    if (!path || !wav) return -1;
    memset(wav, 0, sizeof(*wav));

    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    riff_header_t riff;
    if (fread(&riff, sizeof(riff), 1, f) != 1 ||
        riff.riff_id != RIFF_ID || riff.wave_id != WAVE_ID) {
        fclose(f);
        return -1;
    }

    fmt_chunk_t fmt;
    memset(&fmt, 0, sizeof(fmt));
    int got_fmt = 0;
    uint32_t data_size = 0;
    long data_offset = 0;

    /* Scan chunks */
    while (1) {
        chunk_header_t ch;
        if (fread(&ch, sizeof(ch), 1, f) != 1) break;

        if (ch.chunk_id == FMT_ID) {
            size_t to_read = ch.chunk_size < sizeof(fmt) ? ch.chunk_size : sizeof(fmt);
            if (fread(&fmt, to_read, 1, f) != 1) break;
            if (ch.chunk_size > (uint32_t)to_read)
                fseek(f, (long)(ch.chunk_size - to_read), SEEK_CUR);
            got_fmt = 1;
        } else if (ch.chunk_id == DATA_ID) {
            data_size = ch.chunk_size;
            data_offset = ftell(f);
            break;
        } else {
            /* Skip unknown chunk */
            fseek(f, (long)ch.chunk_size, SEEK_CUR);
        }
    }

    if (!got_fmt || data_offset == 0 || data_size == 0) {
        fclose(f);
        return -1;
    }

    /* Validate format */
    if (fmt.format != WAV_PCM && fmt.format != WAV_IEEE_FLOAT) {
        fclose(f);
        return -1;
    }
    if (fmt.channels == 0 || fmt.bits_per_sample == 0) {
        fclose(f);
        return -1;
    }

    uint32_t bytes_per_sample = fmt.bits_per_sample / 8;
    uint32_t frame_size = bytes_per_sample * fmt.channels;
    if (frame_size == 0) { fclose(f); return -1; }
    uint32_t num_frames = data_size / frame_size;

    /* Read raw data */
    fseek(f, data_offset, SEEK_SET);
    uint8_t *raw = (uint8_t *)malloc(data_size);
    if (!raw) { fclose(f); return -1; }
    if (fread(raw, 1, data_size, f) != data_size) {
        free(raw);
        fclose(f);
        return -1;
    }
    fclose(f);

    /* Convert to float32 */
    size_t total_samples = (size_t)num_frames * fmt.channels;
    float *samples = (float *)malloc(total_samples * sizeof(float));
    if (!samples) { free(raw); return -1; }

    for (size_t i = 0; i < total_samples; i++) {
        const uint8_t *p = raw + i * bytes_per_sample;

        if (fmt.format == WAV_IEEE_FLOAT && bytes_per_sample == 4) {
            float v;
            memcpy(&v, p, 4);
            samples[i] = v;
        } else if (fmt.format == WAV_PCM && bytes_per_sample == 2) {
            int16_t v;
            memcpy(&v, p, 2);
            samples[i] = (float)v / 32768.0f;
        } else if (fmt.format == WAV_PCM && bytes_per_sample == 3) {
            int32_t v = (int32_t)(p[0] | (p[1] << 8) | (p[2] << 16));
            if (v & 0x800000) v |= (int32_t)0xFF000000;  /* sign extend */
            samples[i] = (float)v / 8388608.0f;
        } else if (fmt.format == WAV_PCM && bytes_per_sample == 4) {
            int32_t v;
            memcpy(&v, p, 4);
            samples[i] = (float)((double)v / 2147483648.0);
        } else {
            free(raw);
            free(samples);
            return -1;
        }
    }

    free(raw);
    wav->samples = samples;
    wav->sample_rate = fmt.sample_rate;
    wav->channels = fmt.channels;
    wav->num_frames = num_frames;
    return 0;
}

int wav_write(const char *path, const float *samples, uint32_t num_frames,
              uint16_t channels, uint32_t sample_rate) {
    if (!path || !samples) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    uint32_t data_size = num_frames * channels * 4;  /* 32-bit float */
    uint32_t file_size = 36 + data_size;

    riff_header_t riff = { RIFF_ID, file_size, WAVE_ID };
    fwrite(&riff, sizeof(riff), 1, f);

    chunk_header_t fmt_hdr = { FMT_ID, 16 };
    fwrite(&fmt_hdr, sizeof(fmt_hdr), 1, f);

    fmt_chunk_t fmt = {
        WAV_IEEE_FLOAT,
        channels,
        sample_rate,
        sample_rate * channels * 4,
        (uint16_t)(channels * 4),
        32
    };
    fwrite(&fmt, sizeof(fmt), 1, f);

    chunk_header_t data_hdr = { DATA_ID, data_size };
    fwrite(&data_hdr, sizeof(data_hdr), 1, f);
    fwrite(samples, 4, (size_t)num_frames * channels, f);

    fclose(f);
    return 0;
}

void wav_free(wav_data_t *wav) {
    if (!wav) return;
    free(wav->samples);
    wav->samples = NULL;
    wav->num_frames = 0;
}
