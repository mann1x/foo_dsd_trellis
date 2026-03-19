/*
 * dsd_reencode — Compare sequential vs parallel same-rate re-encoding.
 *
 * Generates a 1kHz DSD sine, re-encodes through boxcar→SDM:
 *   - Sequential: single sdm_process_block call (no segment stitching)
 *   - Parallel: N segments with overlap/warmup (simulates threadpool path)
 *
 * Outputs two DSF files for A/B listening comparison.
 *
 * Usage: test.exe --reencode [rate_mult] [seconds]
 *   rate_mult: 64/128/256/512 (default: 128)
 *   seconds:   duration (default: 10)
 */

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "../include/dsd_types.h"
#include "../include/trellis.h"
#include "../include/ntf.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── DSF writer (from dsd_encode.c) ─── */

static void pack_dsd_lsb(const float *dsd, uint8_t *out, size_t n_bits) {
    memset(out, 0, (n_bits + 7) / 8);
    for (size_t i = 0; i < n_bits; i++) {
        if (dsd[i] >= 0.0f)
            out[i / 8] |= (uint8_t)(1u << (i % 8));  /* LSB first */
    }
}

static void write_le32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void write_le64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }

static int write_dsf(const char *path, const float *dsd, size_t count,
                     uint32_t dsd_rate) {
    FILE *f = NULL;
    fopen_s(&f, path, "wb");
    if (!f) { fprintf(stderr, "Cannot write: %s\n", path); return -1; }

    uint16_t channels = 1;
    uint32_t block_size = 4096;
    size_t dsd_bytes = (count + 7) / 8;
    size_t blocks = (dsd_bytes + block_size - 1) / block_size;
    size_t padded_bytes = blocks * block_size;

    uint8_t *packed = (uint8_t *)calloc(padded_bytes, 1);
    if (!packed) { fclose(f); return -1; }
    pack_dsd_lsb(dsd, packed, count);

    uint64_t data_chunk_size = 12 + padded_bytes * channels;
    uint64_t fmt_chunk_size = 52;
    uint64_t dsd_chunk_size = 28;
    uint64_t total_size = dsd_chunk_size + fmt_chunk_size + data_chunk_size;

    fwrite("DSD ", 1, 4, f);
    write_le64(f, dsd_chunk_size);
    write_le64(f, total_size);
    write_le64(f, 0);

    fwrite("fmt ", 1, 4, f);
    write_le64(f, fmt_chunk_size);
    write_le32(f, 1);
    write_le32(f, 0);
    write_le32(f, (uint32_t)channels);
    write_le32(f, (uint32_t)channels);
    write_le32(f, dsd_rate);
    write_le32(f, 1);
    write_le64(f, (uint64_t)count);
    write_le32(f, block_size);
    write_le32(f, 0);

    fwrite("data", 1, 4, f);
    write_le64(f, data_chunk_size);
    fwrite(packed, 1, padded_bytes, f);

    free(packed);
    fclose(f);
    return 0;
}

/* ─── Re-encode engine ─── */

int dsd_reencode_main(int argc, char **argv) {
    int rate_mult = 128;
    int seconds = 10;
    if (argc >= 2) rate_mult = atoi(argv[1]);
    if (argc >= 3) seconds = atoi(argv[2]);
    if (rate_mult < 64) rate_mult = 64;

    uint32_t dsd_rate = (uint32_t)rate_mult * 44100;
    size_t n = (size_t)dsd_rate * (size_t)seconds;

    printf("DSD%d re-encode test: %zu samples (%.1fs)\n", rate_mult, n, (double)seconds);

    /* Look up optimal config from path table */
    int nc = 2;
    int lat = (rate_mult >= 512) ? 32 : (rate_mult >= 128) ? 128 : 32;
    ntf_filter_id_t ntf_id = (rate_mult == 64) ? NTF_CLANS_6 :
                              (rate_mult == 128) ? NTF_SDM_6 :
                              (rate_mult == 256) ? NTF_CLANS_6 : NTF_SDM_6;

    const ntf_filter_t *f = ntf_get_filter(ntf_id, dsd_rate);
    if (!f) { printf("NTF not found\n"); return 1; }

    /* Generate DSD input: 1kHz sine, hard-quantized to ±1.0 */
    float *dsd_in = (float *)malloc(n * sizeof(float));
    if (!dsd_in) { printf("OOM\n"); return 1; }
    for (size_t i = 0; i < n; i++) {
        double s = 0.1 * sin(2.0 * M_PI * 100.0 * i / dsd_rate);
        dsd_in[i] = (s >= 0.0) ? 1.0f : -1.0f;
    }

    /* Boxcar smooth (fp64) */
    int taps = (dsd_rate >= DSD_RATE_512) ? 128 :
               (dsd_rate >= DSD_RATE_128) ? 64 : 32;
    double *smooth = (double *)malloc(n * sizeof(double));
    {
        double ring[128] = {0};
        double sum = 0; int pos = 0;
        double inv_n = 1.0 / (double)taps;
        double gain = 0.708;
        for (size_t i = 0; i < n; i++) {
            double s = dsd_in[i] >= 0.0f ? 1.0 : -1.0;
            sum -= ring[pos]; ring[pos] = s; sum += s;
            pos = (pos + 1) % taps;
            smooth[i] = sum * inv_n * gain;
        }
    }

    /* ─── Sequential: single SDM call ─── */
    printf("Sequential (1 segment)...\n");
    float *out_seq = (float *)malloc(n * sizeof(float));
    {
        sdm_context_t ctx;
        sdm_context_init(&ctx, f, 8, nc, lat);
        size_t produced = sdm_process_block(&ctx, smooth, out_seq, n);
        printf("  produced=%zu conv_fail=%llu collapse=%llu\n",
               produced, (unsigned long long)ctx.conv_fail,
               (unsigned long long)ctx.cands_collapse);
        sdm_context_free(&ctx);
    }

    /* ─── Parallel: N segments with overlap/warmup ─── */
    int num_segs = (rate_mult >= 512) ? 4 : 2;
    size_t seg_size = (n - lat) / (size_t)num_segs;
    size_t overlap = (size_t)lat * 4;  /* warmup = 4x latency */

    printf("Parallel (%d segments, overlap=%zu)...\n", num_segs, overlap);
    float *out_par = (float *)calloc(n, sizeof(float));
    {
        size_t total_produced = 0;
        uint64_t total_conv = 0, total_collapse = 0;
        for (int seg = 0; seg < num_segs; seg++) {
            size_t seg_start = seg * seg_size;
            size_t seg_end = (seg == num_segs - 1) ? (n - lat) : (seg + 1) * seg_size;
            size_t seg_len = seg_end - seg_start;

            sdm_context_t ctx;
            sdm_context_init(&ctx, f, 8, nc, lat);

            if (seg == 0) {
                /* Segment 0: no warmup needed, process from start */
                size_t produced = sdm_process_block(&ctx, smooth, out_par, seg_len + lat);
                total_produced += produced;
            } else {
                /* Segments 1+: warmup with overlap, discard warmup output */
                size_t warmup_start = seg_start > overlap ? seg_start - overlap : 0;
                size_t warmup_len = seg_start - warmup_start;

                /* Feed warmup (output discarded) */
                float *trash = (float *)malloc((warmup_len + lat) * sizeof(float));
                sdm_process_block(&ctx, smooth + warmup_start, trash, warmup_len + lat);
                free(trash);

                /* Process actual segment */
                size_t produced = sdm_process_block(&ctx, smooth + seg_start,
                                                     out_par + seg_start - lat, seg_len);
                total_produced += produced;
            }
            total_conv += ctx.conv_fail;
            total_collapse += ctx.cands_collapse;
            sdm_context_free(&ctx);
        }
        printf("  produced=%zu conv_fail=%llu collapse=%llu\n",
               total_produced, (unsigned long long)total_conv,
               (unsigned long long)total_collapse);
    }

    /* ─── Chunked: simulate live playback (1s chunks, persistent SDM) ─── */
    size_t chunk_samples = (size_t)dsd_rate;  /* 1 second per chunk */
    int n_chunks = (int)(n / chunk_samples);

    printf("Chunked (%d x 1s chunks, %d segs/chunk)...\n", n_chunks, num_segs);
    float *out_chunked = (float *)calloc(n, sizeof(float));
    {
        /* Persistent SDM for segment 0 */
        sdm_context_t seg0_sdm;
        sdm_context_init(&seg0_sdm, f, 8, nc, lat);

        size_t total_out = 0;
        uint64_t total_conv = 0, total_collapse = 0;

        /* Previous chunk's FIR tail for warmup */
        double *fir_tail = NULL;
        size_t fir_tail_len = 0;

        for (int c = 0; c < n_chunks; c++) {
            size_t chunk_off = (size_t)c * chunk_samples;
            double *chunk_in = smooth + chunk_off;
            float *chunk_out = out_chunked + total_out;

            if (num_segs <= 1 || !fir_tail) {
                /* Single segment or first chunk: use persistent SDM */
                size_t produced = sdm_process_block(&seg0_sdm, chunk_in,
                                                     chunk_out, chunk_samples);
                total_out += produced;
                total_conv += seg0_sdm.conv_fail;
                total_collapse += seg0_sdm.cands_collapse;
            } else {
                /* Multi-segment: seg0 uses persistent SDM, seg1+ use temp */
                size_t seg_size_c = chunk_samples / (size_t)num_segs;

                /* Segment 0: persistent SDM, warmup from prev fir_tail */
                size_t seg0_total = seg_size_c + overlap;
                double *seg0_buf = (double *)malloc(seg0_total * sizeof(double));
                memcpy(seg0_buf, fir_tail, overlap * sizeof(double));
                memcpy(seg0_buf + overlap, chunk_in, seg_size_c * sizeof(double));

                sdm_context_t seg0_temp;
                sdm_context_init(&seg0_temp, f, 8, nc, lat);
                /* Warmup with fir_tail */
                float *trash = (float *)malloc((overlap + lat) * sizeof(float));
                sdm_process_block(&seg0_temp, seg0_buf, trash, overlap + lat);
                free(trash);
                /* Process seg0 data */
                size_t p0 = sdm_process_block(&seg0_temp, seg0_buf + overlap,
                                               chunk_out, seg_size_c);
                total_out += p0;
                sdm_context_free(&seg0_temp);
                free(seg0_buf);

                /* Also advance persistent SDM (keep it in sync) */
                float *dummy = (float *)malloc(seg_size_c * sizeof(float));
                sdm_process_block(&seg0_sdm, chunk_in, dummy, seg_size_c);
                free(dummy);

                /* Segments 1+: temp SDMs with overlap warmup */
                for (int seg = 1; seg < num_segs; seg++) {
                    size_t seg_start = seg * seg_size_c;
                    size_t seg_len = (seg == num_segs - 1) ?
                        chunk_samples - seg_start : seg_size_c;

                    sdm_context_t temp;
                    sdm_context_init(&temp, f, 8, nc, lat);

                    size_t warmup_start = seg_start > overlap ? seg_start - overlap : 0;
                    size_t warmup_len = seg_start - warmup_start;
                    float *tw = (float *)malloc((warmup_len + lat) * sizeof(float));
                    sdm_process_block(&temp, chunk_in + warmup_start, tw, warmup_len + lat);
                    free(tw);

                    size_t p = sdm_process_block(&temp, chunk_in + seg_start,
                                                  chunk_out + p0 + (seg - 1) * seg_size_c,
                                                  seg_len);
                    total_out += p;
                    total_conv += temp.conv_fail;
                    total_collapse += temp.cands_collapse;
                    sdm_context_free(&temp);
                }

                /* Advance persistent SDM through remaining segments */
                float *dummy2 = (float *)malloc((chunk_samples - seg_size_c) * sizeof(float));
                sdm_process_block(&seg0_sdm, chunk_in + seg_size_c, dummy2,
                                   chunk_samples - seg_size_c);
                free(dummy2);
            }

            /* Save fir_tail for next chunk */
            if (!fir_tail) fir_tail = (double *)malloc(overlap * sizeof(double));
            if (chunk_samples >= overlap)
                memcpy(fir_tail, chunk_in + chunk_samples - overlap, overlap * sizeof(double));
            fir_tail_len = overlap;
        }
        printf("  total_out=%zu conv_fail=%llu collapse=%llu\n",
               total_out, (unsigned long long)total_conv,
               (unsigned long long)total_collapse);
        sdm_context_free(&seg0_sdm);
        free(fir_tail);
    }

    /* ─── Convergence test: measure SDM state error vs overlap size ─── */
    printf("\n=== SDM state convergence vs overlap ===\n");
    printf("  Processing seg0 (%zu samples) to get reference state...\n", n/2);
    {
        sdm_context_t ref;
        sdm_context_init(&ref, f, 8, nc, lat);

        /* Process full first half to get reference final state */
        float *tmp_out = (float *)malloc(n * sizeof(float));
        sdm_process_block(&ref, smooth, tmp_out, n/2);

        /* Extract reference state from best candidate */
        int order = ref.filter->order;
        double ref_state[8];
        double ref_prev_y = ref.prev_y;
        for (int k = 0; k < order; k++)
            ref_state[k] = ref.trellis[ref.idx].act[0]->state[k];

        printf("  Reference state: [");
        for (int k = 0; k < order; k++)
            printf("%.6f%s", ref_state[k], k < order-1 ? ", " : "");
        printf("] prev_y=%.1f\n", ref_prev_y);

        /* Test convergence at various overlap sizes */
        size_t overlaps[] = { 128, 256, 512, 1024, 2048, 4096, 8192,
                              16384, 32768, 65536, n/4, n/2 };
        int n_overlaps = sizeof(overlaps) / sizeof(overlaps[0]);

        printf("\n  %8s  %12s  %12s  %s\n", "overlap", "max_err", "rms_err", "converged");
        for (int oi = 0; oi < n_overlaps; oi++) {
            size_t ov = overlaps[oi];
            if (ov > n/2) continue;

            /* Start from beginning, process 'ov' samples as warmup */
            size_t warmup_start = (n/2 > ov) ? (n/2 - ov) : 0;
            size_t warmup_len = n/2 - warmup_start;

            sdm_context_t test;
            sdm_context_init(&test, f, 8, nc, lat);

            /* Feed warmup */
            float *tw = (float *)malloc((warmup_len + lat) * sizeof(float));
            sdm_process_block(&test, smooth + warmup_start, tw, warmup_len + lat);
            free(tw);

            /* Compare states */
            double max_err = 0.0;
            double rms_err = 0.0;
            for (int k = 0; k < order; k++) {
                double err = fabs(test.trellis[test.idx].act[0]->state[k] - ref_state[k]);
                if (err > max_err) max_err = err;
                rms_err += err * err;
            }
            /* Also check prev_y */
            double py_err = fabs(test.prev_y - ref_prev_y);
            rms_err = sqrt(rms_err / order);

            printf("  %8zu  %12.2e  %12.2e  %s (prev_y err=%.2e)\n",
                   ov, max_err, rms_err,
                   max_err < 1e-10 ? "YES" : (max_err < 1e-6 ? "almost" : "no"),
                   py_err);

            sdm_context_free(&test);
        }

        free(tmp_out);
        sdm_context_free(&ref);
    }

    /* Write DSF files */
    char seq_path[256], par_path[256], chunked_path[256];
    sprintf_s(seq_path, sizeof(seq_path), "reencode_dsd%d_seq.dsf", rate_mult);
    sprintf_s(par_path, sizeof(par_path), "reencode_dsd%d_par%d.dsf", rate_mult, num_segs);
    sprintf_s(chunked_path, sizeof(chunked_path), "reencode_dsd%d_chunked%d.dsf", rate_mult, num_segs);

    printf("Writing %s...\n", seq_path);
    write_dsf(seq_path, out_seq, n - lat, dsd_rate);
    printf("Writing %s...\n", par_path);
    write_dsf(par_path, out_par, n - lat, dsd_rate);
    printf("Writing %s...\n", chunked_path);
    write_dsf(chunked_path, out_chunked, n - lat, dsd_rate);

    free(dsd_in); free(smooth); free(out_seq); free(out_par); free(out_chunked);
    printf("Done. Compare:\n  %s (sequential)\n  %s (parallel single-pass)\n  %s (chunked like live)\n",
           seq_path, par_path, chunked_path);
    return 0;
}
