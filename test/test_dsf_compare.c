/*
 * Generate DSF files for A/B comparison: CPU vs GPU parallel SDM.
 * Processes same input through both paths and writes output as DSF.
 */

#include "test.h"
#include "../include/trellis.h"
#include "../include/ntf.h"
#include "../include/gpu_compute.h"
#include "../include/fir.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _WIN32
#include <windows.h>
#endif

static void write_dsf(const char *filename, const float *dsd, size_t n_samples,
                       unsigned sample_rate) {
    int channels = 2;  /* stereo (both channels identical for mono source) */
    size_t block_size = 4096;
    size_t bpb = block_size * 8;
    size_t n_blocks = (n_samples + bpb - 1) / bpb;
    size_t data_size = n_blocks * block_size * (size_t)channels;
    unsigned char *packed = (unsigned char *)calloc(data_size, 1);
    if (!packed) return;

    for (size_t blk = 0; blk < n_blocks; blk++) {
        for (int ch = 0; ch < channels; ch++) {
            for (size_t bi = 0; bi < block_size; bi++) {
                size_t so = blk * bpb + bi * 8;
                unsigned char bv = 0;
                for (int bit = 0; bit < 8; bit++) {
                    size_t si = so + bit;
                    if (si < n_samples && dsd[si] > 0.0f)
                        bv |= (1 << bit);
                }
                packed[(blk * channels + ch) * block_size + bi] = bv;
            }
        }
    }

    /* DSF file: DSD chunk (28) + fmt chunk (52) + data chunk (12+data) */
    unsigned long long fmt_sz = 52;
    unsigned long long data_chunk_sz = 12 + data_size;
    unsigned long long total = 28 + fmt_sz + data_chunk_sz;

#ifdef _WIN32
    HANDLE hf = CreateFileA(filename, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, 0, NULL);
    if (hf == INVALID_HANDLE_VALUE) { free(packed); return; }
    DWORD bw;
    /* DSD chunk (28 bytes) */
    WriteFile(hf, "DSD ", 4, &bw, NULL);
    { unsigned long long v = 28; WriteFile(hf, &v, 8, &bw, NULL); }
    WriteFile(hf, &total, 8, &bw, NULL);
    { unsigned long long v = 0; WriteFile(hf, &v, 8, &bw, NULL); }
    /* fmt chunk (52 bytes) */
    WriteFile(hf, "fmt ", 4, &bw, NULL);
    WriteFile(hf, &fmt_sz, 8, &bw, NULL);
    { unsigned int v;
      v = 1; WriteFile(hf, &v, 4, &bw, NULL);         /* format version */
      v = 0; WriteFile(hf, &v, 4, &bw, NULL);         /* format ID: DSD raw */
      v = (channels == 1) ? 1 : 2;
      WriteFile(hf, &v, 4, &bw, NULL);                /* channel type: 1=mono 2=stereo */
      v = channels; WriteFile(hf, &v, 4, &bw, NULL);  /* channel count */
      v = sample_rate; WriteFile(hf, &v, 4, &bw, NULL); /* sample rate */
      v = 1; WriteFile(hf, &v, 4, &bw, NULL);         /* bits per sample */
    }
    { unsigned long long v = (unsigned long long)n_samples * channels;
      WriteFile(hf, &v, 8, &bw, NULL); }              /* total sample count */
    { unsigned int v;
      v = (unsigned int)block_size; WriteFile(hf, &v, 4, &bw, NULL);
      v = 0; WriteFile(hf, &v, 4, &bw, NULL);
    }
    /* data chunk */
    WriteFile(hf, "data", 4, &bw, NULL);
    WriteFile(hf, &data_chunk_sz, 8, &bw, NULL);
    WriteFile(hf, packed, (DWORD)data_size, &bw, NULL);
    CloseHandle(hf);
#endif
    free(packed);
    printf("    Written: %s (%zu samples, %.1fs)\n",
           filename, n_samples, (double)n_samples / sample_rate);
}

void test_dsf_compare(void) {
    if (!test_should_run_suite("dsfcompare")) return;
    printf("\n=== DSF A/B Compare: CPU vs GPU SDM ===\n");

    if (!gpu_available(GPU_BACKEND_CUDA)) {
        printf("  (skipped: CUDA not available)\n");
        return;
    }

    uint32_t dsd_rate = 11289600;  /* DSD128 */
    int nc = 2, lat = 128;
    double duration = 10.0;  /* 10 seconds */
    size_t N = (size_t)(dsd_rate * duration);
    double freq = 50.0;  /* base freq for multi-tone */

    const ntf_filter_t *f = ntf_auto_select(dsd_rate);
    if (!f) { printf("  No NTF\n"); return; }

    printf("  Generating %.0fs DSD128 test signal (%zu samples)...\n", duration, N);

    /* Generate sine → encode to DSD → boxcar smooth */
    double *sine = (double *)malloc(N * sizeof(double));
    float *dsd_enc = (float *)calloc(N, sizeof(float));
    if (!sine || !dsd_enc) { free(sine); free(dsd_enc); return; }

    /* Complex test signal: 50Hz tone + random transients.
     * Simulates real music with dynamic content that stresses DAS. */
    {
        unsigned rng = 12345;
        for (size_t i = 0; i < N; i++) {
            double t = (double)i / (double)dsd_rate;
            /* Base: 50Hz + 440Hz + 3kHz at moderate level */
            double v = 0.15 * sin(2.0 * M_PI * 50.0 * t)
                     + 0.10 * sin(2.0 * M_PI * 440.0 * t)
                     + 0.05 * sin(2.0 * M_PI * 3000.0 * t);
            /* Random transients: short bursts every ~100ms */
            rng = rng * 1103515245 + 12345;
            if ((rng >> 16) % 1000 < 2) {  /* ~0.2% of samples = burst start */
                /* Add a click/transient burst of ~50 samples */
                for (size_t j = 0; j < 50 && i + j < N; j++) {
                    rng = rng * 1103515245 + 12345;
                    double noise = ((double)(int)(rng >> 8) / 8388608.0) - 1.0;
                    sine[i + j] += 0.3 * noise * (1.0 - (double)j / 50.0);
                }
            }
            sine[i] = v;
            /* Fade in/out */
            if (t < 0.05) sine[i] *= t / 0.05;
            if (t > duration - 0.05) sine[i] *= (duration - t) / 0.05;
        }
    }

    sdm_context_t enc;
    sdm_context_init(&enc, f, 4, 16, 512);
    size_t enc_n = sdm_process_block(&enc, sine, dsd_enc, N);
    sdm_context_free(&enc);
    free(sine);
    printf("  Encoded %zu samples\n", enc_n);

    /* Proper FIR lowpass (matching live engine, not crude boxcar).
     * Quantize to ±1 first, then apply FIR lowpass. */
    float *quantized = (float *)malloc(enc_n * sizeof(float));
    for (size_t i = 0; i < enc_n; i++)
        quantized[i] = dsd_enc[i] >= 0.0f ? 1.0f : -1.0f;
    free(dsd_enc);

    fir_lowpass_t lp;
    memset(&lp, 0, sizeof(lp));
    float *smoothed = NULL;
    if (fir_lowpass_init(&lp, dsd_rate) == 0 && lp.initialized) {
        smoothed = (float *)malloc(enc_n * sizeof(float));
        if (smoothed)
            fir_lowpass_process(&lp, quantized, smoothed, enc_n);
        fir_lowpass_free(&lp);
        printf("  FIR lowpass: %d taps\n", lp.taps);
    }
    if (!smoothed) {
        /* Fallback to boxcar if IPP not available */
        printf("  WARNING: FIR lowpass not available, using boxcar fallback\n");
        smoothed = (float *)malloc(enc_n * sizeof(float));
        if (!smoothed) { free(quantized); return; }
        float ring[128] = {0}; float sum = 0; int p = 0;
        int taps = 64;
        for (size_t i = 0; i < enc_n; i++) {
            sum -= ring[p]; ring[p] = quantized[i]; sum += quantized[i];
            p = (p + 1) % taps;
            smoothed[i] = sum / (float)taps;
        }
    }
    free(quantized);

    /* === CPU parallel SDM (8 segments with DAS stitching) === */
    printf("  CPU parallel SDM (8 segments + DAS)...\n");
    float *cpu_out = (float *)calloc(enc_n, sizeof(float));
    {
        double *din = (double *)malloc(enc_n * sizeof(double));
        for (size_t i = 0; i < enc_n; i++) din[i] = (double)smoothed[i];

        /* Process in chunks, each with 8 parallel segments */
        size_t chunk_size = (size_t)(dsd_rate * 0.5);
        size_t pos = 0;
        sdm_context_t persistent;
        sdm_context_init(&persistent, f, f->order, nc, lat);
        int cpu_chunks = 0;

        while (pos < enc_n) {
            size_t this_chunk = enc_n - pos;
            if (this_chunk > chunk_size) this_chunk = chunk_size;

            int num_seg = 2;  /* minimal parallel: 1 stitch boundary */
            size_t overlap = 32 * (size_t)lat;
            if (overlap > 1024) overlap = 1024;
            size_t D = this_chunk / (size_t)num_seg;
            if (D < overlap * 4) { num_seg = 1; D = this_chunk; }

            if (num_seg <= 1) {
                /* Sequential fallback */
                sdm_process_block(&persistent, din + pos, cpu_out + pos, this_chunk);
            } else {
                /* Parallel: expand, process, stitch */
                float **seg_bufs = (float **)calloc((size_t)num_seg, sizeof(float *));
                size_t *seg_counts = (size_t *)calloc((size_t)num_seg, sizeof(size_t));
                sdm_context_t *temps = (sdm_context_t *)calloc((size_t)(num_seg - 1), sizeof(sdm_context_t));

                for (int s = 0; s < num_seg; s++) {
                    size_t nom_start = (size_t)s * D;
                    size_t nom_size = (s == num_seg - 1) ? (this_chunk - nom_start) : D;
                    size_t in_start, in_count, discard;
                    size_t out_cap = nom_size + ((s < num_seg - 1) ? overlap : 0);
                    seg_bufs[s] = (float *)calloc(out_cap, sizeof(float));

                    sdm_context_t *ctx;
                    if (s == 0) {
                        in_start = 0;
                        in_count = nom_size + ((num_seg > 1) ? overlap : 0);
                        discard = 0;
                        ctx = &persistent;
                    } else {
                        in_start = (nom_start >= overlap) ? nom_start - overlap : 0;
                        in_count = nom_start + nom_size - in_start;
                        if (s < num_seg - 1) in_count += overlap;
                        discard = (nom_start >= overlap) ? overlap - (size_t)lat : 0;
                        sdm_context_copy_state(&temps[s-1], &persistent);
                        ctx = &temps[s-1];
                    }
                    if (pos + in_start + in_count > enc_n)
                        in_count = enc_n - pos - in_start;

                    /* Process with warmup discard */
                    if (discard > 0) {
                        size_t warmup_in = (size_t)lat + discard;
                        if (warmup_in < in_count) {
                            float *trash = (float *)calloc(warmup_in, sizeof(float));
                            sdm_process_block(ctx, din + pos + in_start, trash, warmup_in);
                            free(trash);
                            seg_counts[s] = sdm_process_block(ctx, din + pos + in_start + warmup_in,
                                seg_bufs[s], in_count - warmup_in);
                        }
                    } else {
                        seg_counts[s] = sdm_process_block(ctx, din + pos + in_start,
                            seg_bufs[s], in_count);
                    }
                }

                /* DAS stitch on channel 0 */
                size_t write_pos = seg_counts[0];
                memcpy(cpu_out + pos, seg_bufs[0], write_pos * sizeof(float));

                for (int s = 1; s < num_seg; s++) {
                    float *seg_data = seg_bufs[s];
                    size_t seg_n = seg_counts[s];
                    if (seg_n == 0) continue;
                    size_t prev_start = (write_pos >= overlap) ? write_pos - overlap : 0;
                    float *prev_ovl = cpu_out + pos + prev_start;
                    size_t ovl_len = write_pos - prev_start;
                    if (ovl_len > seg_n) ovl_len = seg_n;
                    if (ovl_len > overlap) ovl_len = overlap;

                    /* Density scan */
                    int hw = lat; if (hw > (int)ovl_len/2) hw = (int)ovl_len/2;
                    int best_d = 0, best_p = 0;
                    for (size_t p = 0; p < ovl_len; p++) {
                        int st = (int)p - hw, en = (int)p + hw;
                        if (st < 0) st = 0; if (en > (int)ovl_len) en = (int)ovl_len;
                        int m = 0;
                        for (int w = st; w < en; w++)
                            if (prev_ovl[w] == seg_data[w]) m++;
                        if (m > best_d) { best_d = m; best_p = (int)p; }
                    }
                    int bp = best_p;
                    for (int r = 0; r <= hw; r++) {
                        int lo = best_p-r, hi = best_p+r;
                        if (lo >= 0 && lo < (int)ovl_len && prev_ovl[lo] == seg_data[lo]) { bp=lo; break; }
                        if (hi!=lo && hi >= 0 && hi < (int)ovl_len && prev_ovl[hi] == seg_data[hi]) { bp=hi; break; }
                    }
                    size_t stitch_at = prev_start + (size_t)bp;
                    memcpy(cpu_out + pos + stitch_at, seg_data + bp, (seg_n - (size_t)bp) * sizeof(float));
                    write_pos = stitch_at + seg_n - (size_t)bp;
                }

                /* Copy last segment's state to persistent */
                if (num_seg > 1)
                    sdm_context_copy_state(&persistent, &temps[num_seg - 2]);

                for (int s = 0; s < num_seg; s++) free(seg_bufs[s]);
                free(seg_bufs); free(seg_counts);
                for (int s = 0; s < num_seg - 1; s++) sdm_context_free(&temps[s]);
                free(temps);
            }
            pos += this_chunk;
            cpu_chunks++;
        }
        sdm_context_free(&persistent);
        free(din);
        printf("  CPU: %d chunks processed\n", cpu_chunks);
    }

    /* === GPU 8-segment DAS SDM === */
    printf("  GPU 8-segment DAS SDM...\n");
    float *gpu_out = (float *)calloc(enc_n, sizeof(float));
    {
        gpu_context_t *ctx = gpu_create(GPU_BACKEND_CUDA);
        if (!ctx) { printf("  GPU create failed\n"); goto cleanup; }
        gpu_cuda_trellis_setup(ctx, nc, f->order, lat, f->a, f->g, 0.0);

        /* Process in chunks (~0.5s each) to simulate live playback */
        size_t chunk_size = (size_t)(dsd_rate * 0.5);
        size_t pos = 0;
        int chunk_num = 0;
        while (pos < enc_n) {
            size_t this_chunk = enc_n - pos;
            if (this_chunk > chunk_size) this_chunk = chunk_size;

            float *chunk_out = (float *)calloc(this_chunk, sizeof(float));
            if (!chunk_out) break;

            int rc = gpu_cuda_trellis_das(ctx, smoothed + pos, chunk_out,
                                           this_chunk, 1);
            if (rc != 0) {
                printf("  GPU DAS failed at chunk %d\n", chunk_num);
                free(chunk_out);
                break;
            }

            memcpy(gpu_out + pos, chunk_out, this_chunk * sizeof(float));
            free(chunk_out);
            pos += this_chunk;
            chunk_num++;
        }
        printf("  Processed %d chunks (%zu samples)\n", chunk_num, pos);
        gpu_destroy(ctx);
    }

    /* Write DSF files */
    printf("  Writing DSF files...\n");
    write_dsf("tmp/cpu_2seg_das_10s.dsf", cpu_out, enc_n, dsd_rate);
    write_dsf("tmp/gpu_2seg_das_10s.dsf", gpu_out, enc_n, dsd_rate);

    /* Also generate sequential (no parallel, no stitch) as clean baseline */
    printf("  CPU sequential SDM (baseline)...\n");
    {
        double *din = (double *)malloc(enc_n * sizeof(double));
        for (size_t i = 0; i < enc_n; i++) din[i] = (double)smoothed[i];
        sdm_context_t seq;
        sdm_context_init(&seq, f, f->order, nc, lat);
        float *seq_out = (float *)calloc(enc_n, sizeof(float));
        size_t seq_n = sdm_process_block(&seq, din, seq_out, enc_n);
        /* Drain remaining lat samples to avoid trailing zeros */
        size_t drain_n = sdm_drain(&seq, seq_out + seq_n, enc_n - seq_n);
        seq_n += drain_n;
        sdm_context_free(&seq);
        free(din);
        printf("  Sequential: %zu output + %zu drained = %zu total\n", seq_n - drain_n, drain_n, seq_n);
        write_dsf("tmp/sequential_10s.dsf", seq_out, seq_n, dsd_rate);
        free(seq_out);
    }

    /* === Real music DSF: read DSD128 test asset, re-encode === */
    printf("  Real music re-encode...\n");
    {
        const char *src_path = "\\\\solidpc\\media\\Music\\Test files\\NativeDSD\\"
            "1. DSD Bit Rate Comparison (2ch 5ch) [Orig. 256 Recording]\\"
            "2ch Stereo\\DSD 128 Stereo.dsf";
#ifdef _WIN32
        HANDLE hf = CreateFileA(src_path, GENERIC_READ, FILE_SHARE_READ,
                                 NULL, OPEN_EXISTING, 0, NULL);
        if (hf != INVALID_HANDLE_VALUE) {
            /* Read DSF header */
            unsigned char hdr[92];
            DWORD br;
            ReadFile(hf, hdr, 92, &br, NULL);
            unsigned int src_rate = *(unsigned int *)(hdr + 56);
            unsigned int src_ch = *(unsigned int *)(hdr + 52);
            unsigned long long src_total = *(unsigned long long *)(hdr + 64);
            unsigned int src_bs = *(unsigned int *)(hdr + 72);
            size_t src_per_ch = (size_t)(src_total / src_ch);
            /* Limit to 10 seconds */
            size_t max_samples = (size_t)src_rate * 10;
            if (src_per_ch > max_samples) src_per_ch = max_samples;

            printf("    Source: %u Hz, %u ch, %zu samples/ch\n", src_rate, src_ch, src_per_ch);

            /* Read and unpack channel 0 DSD bits */
            size_t bpb = (size_t)src_bs * 8;
            size_t n_blocks = (src_per_ch + bpb - 1) / bpb;
            size_t data_size = n_blocks * (size_t)src_bs * (size_t)src_ch;
            unsigned char *raw = (unsigned char *)malloc(data_size);
            float *src_dsd = (float *)malloc(src_per_ch * sizeof(float));
            if (raw && src_dsd) {
                ReadFile(hf, raw, (DWORD)data_size, &br, NULL);
                /* Unpack ch0 */
                for (size_t blk = 0; blk < n_blocks; blk++) {
                    for (size_t bi = 0; bi < (size_t)src_bs; bi++) {
                        unsigned char bv = raw[(blk * src_ch) * src_bs + bi];
                        for (int bit = 0; bit < 8; bit++) {
                            size_t si = blk * bpb + bi * 8 + bit;
                            if (si < src_per_ch)
                                src_dsd[si] = ((bv >> bit) & 1) ? 1.0f : -1.0f;
                        }
                    }
                }

                /* FIR lowpass */
                fir_lowpass_t lp2;
                memset(&lp2, 0, sizeof(lp2));
                float *music_smooth = NULL;
                if (fir_lowpass_init(&lp2, src_rate) == 0 && lp2.initialized) {
                    music_smooth = (float *)malloc(src_per_ch * sizeof(float));
                    if (music_smooth)
                        fir_lowpass_process(&lp2, src_dsd, music_smooth, src_per_ch);
                    fir_lowpass_free(&lp2);
                }

                if (music_smooth) {
                    const ntf_filter_t *mf = ntf_auto_select(src_rate);
                    if (mf) {
                        /* Sequential re-encode */
                        double *mdin = (double *)malloc(src_per_ch * sizeof(double));
                        float *mseq = (float *)calloc(src_per_ch, sizeof(float));
                        for (size_t i = 0; i < src_per_ch; i++)
                            mdin[i] = (double)music_smooth[i];
                        sdm_context_t msdm;
                        sdm_context_init(&msdm, mf, mf->order, nc, lat);
                        size_t mseq_n = sdm_process_block(&msdm, mdin, mseq, src_per_ch);
                        mseq_n += sdm_drain(&msdm, mseq + mseq_n, src_per_ch - mseq_n);
                        sdm_context_free(&msdm);
                        write_dsf("tmp/music_sequential_10s.dsf", mseq, mseq_n, src_rate);

                        /* 2-seg parallel re-encode */
                        float *mpar = (float *)calloc(src_per_ch, sizeof(float));
                        sdm_context_t mpar_sdm;
                        sdm_context_init(&mpar_sdm, mf, mf->order, nc, lat);
                        {
                            size_t chunk_sz = (size_t)(src_rate / 2);  /* 0.5s chunks */
                            size_t mpos = 0;
                            while (mpos < src_per_ch) {
                                size_t this_c = src_per_ch - mpos;
                                if (this_c > chunk_sz) this_c = chunk_sz;
                                int nseg = 2;
                                size_t ovl = 32 * (size_t)lat;
                                if (ovl > 1024) ovl = 1024;
                                size_t segD = this_c / (size_t)nseg;
                                if (segD < ovl * 4) nseg = 1;

                                if (nseg <= 1) {
                                    size_t nn = sdm_process_block(&mpar_sdm, mdin + mpos, mpar + mpos, this_c);
                                    (void)nn;
                                } else {
                                    /* seg0 */
                                    float *s0 = (float *)calloc(segD + ovl, sizeof(float));
                                    size_t s0n = sdm_process_block(&mpar_sdm, mdin + mpos, s0, segD + ovl);
                                    /* seg1 */
                                    sdm_context_t tmp1;
                                    sdm_context_copy_state(&tmp1, &mpar_sdm);
                                    size_t s1_start = (segD >= ovl) ? segD - ovl : 0;
                                    size_t s1_in = this_c - s1_start;
                                    size_t warmup = ovl;
                                    float *trash = (float *)calloc(warmup, sizeof(float));
                                    sdm_process_block(&tmp1, mdin + mpos + s1_start, trash, warmup);
                                    free(trash);
                                    float *s1 = (float *)calloc(s1_in, sizeof(float));
                                    size_t s1n = sdm_process_block(&tmp1, mdin + mpos + s1_start + warmup, s1, s1_in - warmup);

                                    /* DAS stitch */
                                    memcpy(mpar + mpos, s0, s0n * sizeof(float));
                                    size_t wp = s0n;
                                    size_t ps = (wp >= ovl) ? wp - ovl : 0;
                                    float *po = mpar + mpos + ps;
                                    size_t ol = wp - ps;
                                    if (ol > s1n) ol = s1n;
                                    if (ol > ovl) ol = ovl;
                                    int hw = lat; if (hw > (int)ol/2) hw = (int)ol/2;
                                    int bd=0, bp2=0;
                                    for (size_t p=0; p<ol; p++) {
                                        int st=(int)p-hw, en=(int)p+hw;
                                        if(st<0)st=0; if(en>(int)ol)en=(int)ol;
                                        int m=0; for(int w=st;w<en;w++) if(po[w]==s1[w])m++;
                                        if(m>bd){bd=m;bp2=(int)p;}
                                    }
                                    for(int r=0;r<=hw;r++){
                                        int lo2=bp2-r,hi2=bp2+r;
                                        if(lo2>=0&&lo2<(int)ol&&po[lo2]==s1[lo2]){bp2=lo2;break;}
                                        if(hi2!=lo2&&hi2>=0&&hi2<(int)ol&&po[hi2]==s1[hi2]){bp2=hi2;break;}
                                    }
                                    memcpy(mpar+mpos+ps+(size_t)bp2, s1+bp2, (s1n-(size_t)bp2)*sizeof(float));

                                    sdm_context_copy_state(&mpar_sdm, &tmp1);
                                    sdm_context_free(&tmp1);
                                    free(s0); free(s1);
                                }
                                mpos += this_c;
                            }
                        }
                        sdm_context_free(&mpar_sdm);
                        write_dsf("tmp/music_2seg_das_10s.dsf", mpar, src_per_ch, src_rate);

                        /* Mismatch analysis */
                        {
                            size_t mm = 0;
                            size_t mn = src_per_ch < mseq_n ? src_per_ch : mseq_n;
                            for (size_t i = 0; i < mn; i++)
                                if (mpar[i] != mseq[i]) mm++;
                            printf("    Music: CPU 2seg vs sequential: %zu/%zu (%.1f%%) mismatch\n",
                                   mm, mn, 100.0*(double)mm/(double)mn);
                        }

                        free(mdin); free(mseq); free(mpar);
                    }
                    free(music_smooth);
                }
                free(raw); free(src_dsd);
            }
            CloseHandle(hf);
        } else {
            printf("    Could not open: %s\n", src_path);
        }
#endif
    }

    /* Pure DSD silence reference — alternating ±1 pattern, no SDM.
     * Tests if the DSF format itself is clean. */
    printf("  DSD silence (no SDM)...\n");
    {
        size_t sil_n = (size_t)(dsd_rate * 5);  /* 5 seconds */
        float *sil = (float *)malloc(sil_n * sizeof(float));
        if (sil) {
            for (size_t i = 0; i < sil_n; i++)
                sil[i] = (i & 1) ? 1.0f : -1.0f;
            write_dsf("tmp/silence_5s.dsf", sil, sil_n, dsd_rate);
            free(sil);
        }
    }

    /* Original DSD passthrough — the source DSD before re-encoding.
     * Tests if the original encode has artifacts. */
    printf("  Original DSD (before re-encode)...\n");
    {
        float *orig = (float *)malloc(enc_n * sizeof(float));
        if (orig) {
            /* Re-generate the original encoded DSD */
            double *sine2 = (double *)malloc(enc_n * sizeof(double));
            for (size_t i = 0; i < enc_n; i++)
                sine2[i] = 0.5 * sin(2.0 * M_PI * freq * (double)i / (double)dsd_rate);
            sdm_context_t enc2;
            sdm_context_init(&enc2, f, 4, 16, 512);
            float *enc_buf = (float *)calloc(enc_n, sizeof(float));
            size_t enc2_n = sdm_process_block(&enc2, sine2, enc_buf, enc_n);
            size_t drain2 = sdm_drain(&enc2, enc_buf + enc2_n, enc_n - enc2_n);
            enc2_n += drain2;
            /* Quantize to ±1 */
            for (size_t i = 0; i < enc2_n; i++)
                orig[i] = enc_buf[i] >= 0.0f ? 1.0f : -1.0f;
            write_dsf("tmp/original_dsd_10s.dsf", orig, enc2_n, dsd_rate);
            sdm_context_free(&enc2);
            free(sine2); free(enc_buf); free(orig);
        }
    }

cleanup:
    free(smoothed);
    free(cpu_out);
    free(gpu_out);
    printf("  Done! Play both in foobar2000 with bypass.\n");
    g_tests_run++; g_tests_passed++;
}
