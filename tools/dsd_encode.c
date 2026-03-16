/*
 * dsd_encode — DSD encoder using the real foo_dsd_trellis engine.
 *
 * Reads PCM input (WAV or raw float32), runs FIR upsample + SDM encode,
 * writes output in multiple formats:
 *   - .raw  — raw float32 ±1.0 (for ML training)
 *   - .dff  — DSDIFF (DSD Interchange File Format)
 *   - .dsf  — DSF (DSD Stream File)
 *   - .wav  — float32 WAV (FIR reference only)
 *
 * Invoked via: foo_dsd_trellis_test.exe --encode [args...]
 *
 * Usage:
 *   test.exe --encode --input signal.wav --dsd-rate 2822400
 *            --sdm trellis --ntf clans-8 --cands 8 --lat 256
 *            --fir-out ref.raw --sdm-out dsd.dff
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "../include/dsd_types.h"
#include "../include/fir.h"
#include "../include/trellis.h"
#include "../include/precorr.h"
#include "../include/ntf.h"
#include "../include/wav_io.h"

/* ─── NTF name → ID lookup ─── */

typedef struct {
    const char      *name;
    ntf_filter_id_t  id;
} ntf_name_entry_t;

static const ntf_name_entry_t g_ntf_names[] = {
    { "auto",    NTF_AUTO },
    { "clans-4", NTF_CLANS_4 }, { "clans-5", NTF_CLANS_5 },
    { "clans-6", NTF_CLANS_6 }, { "clans-7", NTF_CLANS_7 },
    { "clans-8", NTF_CLANS_8 },
    { "sdm-4",   NTF_SDM_4 },   { "sdm-5",   NTF_SDM_5 },
    { "sdm-6",   NTF_SDM_6 },   { "sdm-7",   NTF_SDM_7 },
    { "sdm-8",   NTF_SDM_8 },
};
#define G_NTF_NAME_COUNT (sizeof(g_ntf_names) / sizeof(g_ntf_names[0]))

static ntf_filter_id_t parse_ntf_name(const char *name) {
    for (int i = 0; i < (int)G_NTF_NAME_COUNT; i++) {
        if (_stricmp(g_ntf_names[i].name, name) == 0)
            return g_ntf_names[i].id;
    }
    return NTF_AUTO;
}

/* ─── Output format detection ─── */

typedef enum {
    OUT_FMT_RAW,    /* raw float32 ±1.0 */
    OUT_FMT_DFF,    /* DSDIFF */
    OUT_FMT_DSF,    /* DSF */
    OUT_FMT_WAV,    /* float32 WAV (for FIR reference) */
} out_format_t;

static out_format_t detect_format(const char *path) {
    const char *ext = strrchr(path, '.');
    if (!ext) return OUT_FMT_RAW;
    if (_stricmp(ext, ".dff") == 0) return OUT_FMT_DFF;
    if (_stricmp(ext, ".dsf") == 0) return OUT_FMT_DSF;
    if (_stricmp(ext, ".wav") == 0) return OUT_FMT_WAV;
    return OUT_FMT_RAW;
}

/* ─── Float ±1.0 → packed DSD bits ───
 * Packs float ±1.0 samples into DSD bytes.
 * DFF: MSB first within each byte.
 * DSF: LSB first within each byte. */

static size_t pack_dsd_msb(const float *in, uint8_t *out, size_t count) {
    size_t bytes = (count + 7) / 8;
    memset(out, 0, bytes);
    for (size_t i = 0; i < count; i++) {
        if (in[i] >= 0.0f)
            out[i / 8] |= (uint8_t)(0x80 >> (i % 8));
    }
    return bytes;
}

static size_t pack_dsd_lsb(const float *in, uint8_t *out, size_t count) {
    size_t bytes = (count + 7) / 8;
    memset(out, 0, bytes);
    for (size_t i = 0; i < count; i++) {
        if (in[i] >= 0.0f)
            out[i / 8] |= (uint8_t)(1 << (i % 8));
    }
    return bytes;
}

/* ─── File I/O helpers ─── */

static void write_be32(FILE *f, uint32_t v) {
    uint8_t b[4] = { (uint8_t)(v >> 24), (uint8_t)(v >> 16),
                      (uint8_t)(v >> 8),  (uint8_t)v };
    fwrite(b, 1, 4, f);
}

static void write_be16(FILE *f, uint16_t v) {
    uint8_t b[2] = { (uint8_t)(v >> 8), (uint8_t)v };
    fwrite(b, 1, 2, f);
}

static void write_le32(FILE *f, uint32_t v) {
    fwrite(&v, 4, 1, f);
}

static void write_le64(FILE *f, uint64_t v) {
    fwrite(&v, 8, 1, f);
}

static float *read_input(const char *path, uint32_t *pcm_rate_out,
                          size_t *count_out) {
    /* Try WAV first */
    wav_data_t wav;
    if (wav_read(path, &wav) == 0) {
        /* Mix to mono if needed */
        size_t n = (size_t)wav.num_frames;
        float *mono;
        if (wav.channels == 1) {
            *pcm_rate_out = wav.sample_rate;
            *count_out = n;
            return wav.samples;  /* caller frees */
        }
        mono = (float *)malloc(n * sizeof(float));
        if (!mono) { wav_free(&wav); return NULL; }
        for (size_t i = 0; i < n; i++) {
            float sum = 0.0f;
            for (int ch = 0; ch < wav.channels; ch++)
                sum += wav.samples[i * wav.channels + ch];
            mono[i] = sum / wav.channels;
        }
        *pcm_rate_out = wav.sample_rate;
        *count_out = n;
        wav_free(&wav);
        return mono;
    }

    /* Fall back to raw float32 */
    FILE *f = NULL;
    fopen_s(&f, path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    *count_out = (size_t)sz / sizeof(float);
    float *buf = (float *)malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, sizeof(float), *count_out, f);
    fclose(f);
    /* pcm_rate_out must be set by caller for raw input */
    return buf;
}

/* ─── DFF (DSDIFF) writer ───
 * Mono only. Standard DSDIFF 1.5 structure:
 *   FRM8 { DSD | FVER | PROP { SND | FS | CHNL } | DSD-data } */

static int write_dff(const char *path, const float *dsd, size_t count,
                     uint32_t dsd_rate) {
    FILE *f = NULL;
    fopen_s(&f, path, "wb");
    if (!f) { fprintf(stderr, "Cannot write: %s\n", path); return -1; }

    size_t dsd_bytes = (count + 7) / 8;
    uint8_t *packed = (uint8_t *)malloc(dsd_bytes);
    if (!packed) { fclose(f); return -1; }
    pack_dsd_msb(dsd, packed, count);

    /* Sizes */
    uint64_t prop_payload = 4 + (12 + 4) + (12 + 2 + 4);  /* SND + FS + CHNL */
    uint64_t prop_chunk = 12 + prop_payload;
    uint64_t dsd_data_chunk = 12 + dsd_bytes;
    uint64_t fver_chunk = 12 + 4;
    uint64_t frm8_payload = 4 + fver_chunk + prop_chunk + dsd_data_chunk;

    /* FRM8 */
    fwrite("FRM8", 1, 4, f);
    { uint8_t b[8]; for (int i = 7; i >= 0; i--) { b[7-i] = (uint8_t)(frm8_payload >> (i*8)); } fwrite(b, 1, 8, f); }
    fwrite("DSD ", 1, 4, f);

    /* FVER */
    fwrite("FVER", 1, 4, f);
    { uint64_t s = 4; uint8_t b[8]; for (int i = 7; i >= 0; i--) b[7-i] = (uint8_t)(s >> (i*8)); fwrite(b, 1, 8, f); }
    write_be32(f, 0x01050000);  /* Version 1.5.0.0 */

    /* PROP */
    fwrite("PROP", 1, 4, f);
    { uint8_t b[8]; for (int i = 7; i >= 0; i--) b[7-i] = (uint8_t)(prop_payload >> (i*8)); fwrite(b, 1, 8, f); }
    fwrite("SND ", 1, 4, f);

    /* FS (sample rate) */
    fwrite("FS  ", 1, 4, f);
    { uint64_t s = 4; uint8_t b[8]; for (int i = 7; i >= 0; i--) b[7-i] = (uint8_t)(s >> (i*8)); fwrite(b, 1, 8, f); }
    write_be32(f, dsd_rate);

    /* CHNL (channels) */
    fwrite("CHNL", 1, 4, f);
    { uint64_t s = 2 + 4; uint8_t b[8]; for (int i = 7; i >= 0; i--) b[7-i] = (uint8_t)(s >> (i*8)); fwrite(b, 1, 8, f); }
    write_be16(f, 1);  /* mono */
    fwrite("SLFT", 1, 4, f);  /* single channel = left */

    /* DSD sound data */
    fwrite("DSD ", 1, 4, f);
    { uint8_t b[8]; for (int i = 7; i >= 0; i--) b[7-i] = (uint8_t)((uint64_t)dsd_bytes >> (i*8)); fwrite(b, 1, 8, f); }
    fwrite(packed, 1, dsd_bytes, f);

    free(packed);
    fclose(f);
    return 0;
}

/* ─── DSF writer ───
 * Mono only. Sony DSF format:
 *   DSD chunk | fmt chunk | data chunk | metadata chunk (empty) */

static int write_dsf(const char *path, const float *dsd, size_t count,
                     uint32_t dsd_rate) {
    FILE *f = NULL;
    fopen_s(&f, path, "wb");
    if (!f) { fprintf(stderr, "Cannot write: %s\n", path); return -1; }

    /* DSF uses 4096-byte blocks per channel, LSB bit order */
    uint16_t channels = 1;
    uint32_t block_size = 4096;
    size_t dsd_bytes = (count + 7) / 8;
    /* Round up to block_size */
    size_t blocks = (dsd_bytes + block_size - 1) / block_size;
    size_t padded_bytes = blocks * block_size;

    uint8_t *packed = (uint8_t *)calloc(padded_bytes, 1);
    if (!packed) { fclose(f); return -1; }
    pack_dsd_lsb(dsd, packed, count);

    uint64_t data_chunk_size = 12 + padded_bytes * channels;
    uint64_t fmt_chunk_size = 52;
    uint64_t dsd_chunk_size = 28;
    uint64_t total_size = dsd_chunk_size + fmt_chunk_size + data_chunk_size;

    /* DSD chunk */
    fwrite("DSD ", 1, 4, f);
    write_le64(f, dsd_chunk_size);
    write_le64(f, total_size);
    write_le64(f, 0);  /* metadata offset (0 = none) */

    /* fmt chunk */
    fwrite("fmt ", 1, 4, f);
    write_le64(f, fmt_chunk_size);
    write_le32(f, 1);                    /* format version */
    write_le32(f, 0);                    /* format ID (DSD raw) */
    write_le32(f, (uint32_t)channels);   /* channel type: mono */
    write_le32(f, (uint32_t)channels);   /* channel count */
    write_le32(f, dsd_rate);             /* sample rate */
    write_le32(f, 1);                    /* bits per sample */
    write_le64(f, (uint64_t)count);      /* sample count per channel */
    write_le32(f, block_size);           /* block size per channel */
    write_le32(f, 0);                    /* reserved */

    /* data chunk */
    fwrite("data", 1, 4, f);
    write_le64(f, data_chunk_size);
    fwrite(packed, 1, padded_bytes, f);

    free(packed);
    fclose(f);
    return 0;
}

/* ─── Raw float32 writer ─── */

static int write_raw_f32(const char *path, const float *data, size_t count) {
    FILE *f = NULL;
    fopen_s(&f, path, "wb");
    if (!f) { fprintf(stderr, "Cannot write: %s\n", path); return -1; }
    fwrite(data, sizeof(float), count, f);
    fclose(f);
    return 0;
}

/* ─── Write output in detected format ─── */

static int write_output(const char *path, const float *data, size_t count,
                        uint32_t dsd_rate, bool is_fir_ref) {
    out_format_t fmt = detect_format(path);

    if (is_fir_ref && (fmt == OUT_FMT_DFF || fmt == OUT_FMT_DSF)) {
        fprintf(stderr, "WARNING: FIR reference is multi-bit, "
                "DFF/DSF only store 1-bit. Using raw.\n");
        fmt = OUT_FMT_RAW;
    }

    switch (fmt) {
    case OUT_FMT_DFF:
        return write_dff(path, data, count, dsd_rate);
    case OUT_FMT_DSF:
        return write_dsf(path, data, count, dsd_rate);
    case OUT_FMT_WAV:
        return wav_write(path, data, (uint32_t)count, 1, dsd_rate);
    case OUT_FMT_RAW:
    default:
        return write_raw_f32(path, data, count);
    }
}

/* ─── Main ─── */

static void print_usage(void) {
    fprintf(stderr,
        "dsd_encode — DSD encoder using foo_dsd_trellis engine\n\n"
        "Usage:\n"
        "  test.exe --encode --input FILE --dsd-rate RATE\n"
        "           --sdm MODE --ntf FILTER [options]\n\n"
        "Required:\n"
        "  --input FILE       Input PCM (WAV or raw float32 mono)\n"
        "  --dsd-rate RATE    Target DSD rate (2822400, 5644800, ...)\n"
        "  --sdm MODE         SDM mode: trellis or precorr\n"
        "  --ntf FILTER       NTF filter: auto, clans-4..8, sdm-4..8\n\n"
        "Output (at least one required):\n"
        "  --fir-out FILE     FIR reference (.raw, .wav)\n"
        "  --sdm-out FILE     SDM output (.raw, .dff, .dsf, .wav)\n\n"
        "Options:\n"
        "  --pcm-rate RATE    PCM rate (auto-detected from WAV, required for raw)\n"
        "  --cands N          Trellis candidates (default: 8)\n"
        "  --lat N            Trellis latency (default: 256)\n"
        "  --depth N          Trellis depth (default: 4)\n"
        "  --state-limit F    Integrator state limiter (default: 0 = off)\n"
        "  --gain F           FIR output gain (default: 1.0)\n"
        "  --samples N        Max input samples (default: all)\n"
        "  --quiet            Suppress progress output\n\n"
        "Output format is auto-detected from file extension:\n"
        "  .raw  — raw float32 (+/-1.0 for SDM, multi-bit for FIR)\n"
        "  .dff  — DSDIFF (standard DSD interchange format)\n"
        "  .dsf  — DSF (Sony DSD stream format)\n"
        "  .wav  — 32-bit float WAV\n\n"
        "Examples:\n"
        "  test.exe --encode --input music.wav --dsd-rate 2822400 \\\n"
        "           --sdm trellis --ntf clans-8 --cands 8 --lat 256 \\\n"
        "           --sdm-out output.dff --fir-out ref.raw\n"
        "  test.exe --encode --input music.wav --dsd-rate 5644800 \\\n"
        "           --sdm precorr --ntf auto --sdm-out output.dsf\n"
    );
}

int dsd_encode_main(int argc, char **argv) {
    /* Defaults */
    const char *input_path = NULL;
    const char *fir_out_path = NULL;
    const char *sdm_out_path = NULL;
    uint32_t pcm_rate = 0;
    uint32_t dsd_rate = 0;
    int sdm_mode = SDM_MODE_TRELLIS;
    const char *ntf_name = "auto";
    int cands = 8;
    int lat = 256;
    int depth = 4;
    double state_limit = 0.0;
    float gain = 1.0f;
    size_t max_samples = 0;
    bool quiet = false;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            input_path = argv[++i];
        else if (strcmp(argv[i], "--pcm-rate") == 0 && i + 1 < argc)
            pcm_rate = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--dsd-rate") == 0 && i + 1 < argc)
            dsd_rate = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--sdm") == 0 && i + 1 < argc) {
            i++;
            if (_stricmp(argv[i], "precorr") == 0)
                sdm_mode = SDM_MODE_PRECORR;
            else
                sdm_mode = SDM_MODE_TRELLIS;
        }
        else if (strcmp(argv[i], "--ntf") == 0 && i + 1 < argc)
            ntf_name = argv[++i];
        else if (strcmp(argv[i], "--fir-out") == 0 && i + 1 < argc)
            fir_out_path = argv[++i];
        else if (strcmp(argv[i], "--sdm-out") == 0 && i + 1 < argc)
            sdm_out_path = argv[++i];
        else if (strcmp(argv[i], "--cands") == 0 && i + 1 < argc)
            cands = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lat") == 0 && i + 1 < argc)
            lat = atoi(argv[++i]);
        else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc)
            depth = atoi(argv[++i]);
        else if (strcmp(argv[i], "--state-limit") == 0 && i + 1 < argc)
            state_limit = atof(argv[++i]);
        else if (strcmp(argv[i], "--gain") == 0 && i + 1 < argc)
            gain = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc)
            max_samples = (size_t)_atoi64(argv[++i]);
        else if (strcmp(argv[i], "--quiet") == 0)
            quiet = true;
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        }
        else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            print_usage();
            return 1;
        }
    }

    if (!input_path || dsd_rate == 0 || (!fir_out_path && !sdm_out_path)) {
        fprintf(stderr, "Missing required arguments.\n\n");
        print_usage();
        return 1;
    }

    /* Read input */
    size_t in_count = 0;
    uint32_t detected_rate = 0;
    float *pcm_in = read_input(input_path, &detected_rate, &in_count);
    if (!pcm_in) return 1;

    if (detected_rate > 0 && pcm_rate == 0)
        pcm_rate = detected_rate;
    if (pcm_rate == 0) {
        fprintf(stderr, "PCM rate unknown. Use --pcm-rate for raw input.\n");
        free(pcm_in);
        return 1;
    }

    if (max_samples > 0 && in_count > max_samples)
        in_count = max_samples;

    if (!quiet)
        fprintf(stderr, "Input: %zu samples @ %u Hz (%.3f s)\n",
                in_count, pcm_rate, (double)in_count / pcm_rate);

    /* FIR upsample */
    uint32_t ratio = dsd_rate / pcm_rate;
    if (ratio < 1 || dsd_rate != ratio * pcm_rate) {
        fprintf(stderr, "Invalid rate ratio: %u / %u\n", dsd_rate, pcm_rate);
        free(pcm_in);
        return 1;
    }

    fir_chain_t fir;
    memset(&fir, 0, sizeof(fir));
    if (fir_chain_init(&fir, pcm_rate, dsd_rate) != 0) {
        fprintf(stderr, "FIR chain init failed (%u -> %u)\n", pcm_rate, dsd_rate);
        free(pcm_in);
        return 1;
    }

    size_t fir_alloc = in_count * ratio + 1024;
    float *fir_out = (float *)malloc(fir_alloc * sizeof(float));
    if (!fir_out) { free(pcm_in); fir_chain_free(&fir); return 1; }

    size_t fir_count = fir_chain_process(&fir, pcm_in, fir_out, in_count);
    fir_chain_free(&fir);
    free(pcm_in);

    if (!quiet)
        fprintf(stderr, "FIR: %zu -> %zu samples (%ux)\n",
                in_count, fir_count, ratio);

    /* Apply gain */
    if (gain != 1.0f) {
        for (size_t i = 0; i < fir_count; i++)
            fir_out[i] *= gain;
        if (!quiet)
            fprintf(stderr, "Gain: %.4f\n", gain);
    }

    /* Write FIR reference */
    if (fir_out_path) {
        if (write_output(fir_out_path, fir_out, fir_count, dsd_rate, true) != 0) {
            free(fir_out);
            return 1;
        }
        if (!quiet)
            fprintf(stderr, "FIR ref: %s (%zu samples)\n", fir_out_path, fir_count);
    }

    /* SDM encode */
    if (!sdm_out_path) {
        /* Only FIR reference requested */
        free(fir_out);
        return 0;
    }

    ntf_filter_id_t ntf_id = parse_ntf_name(ntf_name);
    const ntf_filter_t *filter;
    if (ntf_id == NTF_AUTO) {
        if (sdm_mode == SDM_MODE_PRECORR)
            filter = ntf_auto_select_precorr(dsd_rate);
        else
            filter = ntf_auto_select(dsd_rate);
    } else {
        filter = ntf_get_filter(ntf_id, dsd_rate);
    }

    if (!filter) {
        fprintf(stderr, "NTF filter not found: %s @ %u Hz\n", ntf_name, dsd_rate);
        free(fir_out);
        return 1;
    }

    float *sdm_out = (float *)malloc(fir_count * sizeof(float));
    if (!sdm_out) { free(fir_out); return 1; }

    size_t sdm_count;
    if (sdm_mode == SDM_MODE_PRECORR) {
        precorr_context_t precorr;
        if (precorr_context_init(&precorr, filter) != 0) {
            fprintf(stderr, "PreCorr init failed\n");
            free(fir_out); free(sdm_out);
            return 1;
        }
        if (state_limit > 0.0)
            precorr.state_limit = (float)state_limit;

        sdm_count = precorr_process_block(&precorr, fir_out, sdm_out, fir_count);
        precorr_context_free(&precorr);

        if (!quiet)
            fprintf(stderr, "SDM: PreCorr, NTF=%s, order=%d, %zu samples\n",
                    ntf_name, filter->order, sdm_count);
    } else {
        sdm_context_t sdm;
        if (sdm_context_init(&sdm, filter, depth, cands, lat) != 0) {
            fprintf(stderr, "Trellis SDM init failed\n");
            free(fir_out); free(sdm_out);
            return 1;
        }
        if (state_limit > 0.0)
            sdm.state_limit = state_limit;

        sdm_count = sdm_process_block(&sdm, fir_out, sdm_out, fir_count);

        /* Drain latency */
        if (sdm.pending > 0) {
            size_t drained = sdm_drain(&sdm, sdm_out + sdm_count,
                                        fir_count - sdm_count);
            sdm_count += drained;
        }

        sdm_context_free(&sdm);

        if (!quiet)
            fprintf(stderr, "SDM: Trellis, NTF=%s, depth=%d, cands=%d, lat=%d, %zu samples\n",
                    ntf_name, depth, cands, lat, sdm_count);
    }

    free(fir_out);

    /* Write SDM output */
    if (write_output(sdm_out_path, sdm_out, sdm_count, dsd_rate, false) != 0) {
        free(sdm_out);
        return 1;
    }

    free(sdm_out);

    if (!quiet)
        fprintf(stderr, "Done: %s\n", sdm_out_path);

    return 0;
}
