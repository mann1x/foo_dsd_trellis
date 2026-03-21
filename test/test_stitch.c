/*
 * foo_dsd_trellis — Overlap stitch convergence diagnostics
 *
 * Measures how well parallel SDM segments converge in the overlap region,
 * comparing bit-matching (current) vs state-distance stitching approaches,
 * across different overlap sizes.
 */

#include "test.h"
#include "../include/trellis.h"
#include "../include/ntf.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Generate a 1kHz sine tone at DSD rate, normalized to SDM input range */
static void gen_sine(double *buf, size_t count, unsigned dsd_rate) {
    double freq = 1000.0;
    double phase_inc = 2.0 * M_PI * freq / (double)dsd_rate;
    for (size_t i = 0; i < count; i++)
        buf[i] = 0.5 * sin(phase_inc * (double)i);
}

/* Generate pink-ish noise (filtered random) for more realistic SDM input */
static void gen_noise(double *buf, size_t count, unsigned seed) {
    double state = 0.0;
    unsigned s = seed;
    for (size_t i = 0; i < count; i++) {
        /* Simple LCG */
        s = s * 1103515245 + 12345;
        double r = ((double)(int)(s >> 8) / 16777216.0);  /* [-0.5, 0.5) */
        state = 0.99 * state + 0.01 * r;  /* 1-pole lowpass */
        buf[i] = state * 0.3;  /* Scale to reasonable SDM input level */
    }
}

/* Simulate parallel SDM stitching with configurable overlap.
 *
 * Process a buffer with a reference (sequential) SDM, then simulate
 * two parallel segments with overlap and measure convergence. */
static void measure_convergence(
    unsigned dsd_rate, int trellis_lat, int trellis_cands,
    size_t overlap, size_t seg_size,
    const double *input, size_t total_count,
    /* output stats */
    int *out_best_run, int *out_best_run_pos,
    double *out_min_dist, int *out_min_dist_pos,
    double *out_dist_at_bit_stitch,
    int *out_match_count)
{
    const ntf_filter_t *f = ntf_auto_select(dsd_rate);
    if (!f) { *out_best_run = -1; return; }

    /* Seg0: processes data[0 .. seg_size + overlap) */
    sdm_context_t seg0;
    sdm_context_init(&seg0, f, f->order, trellis_cands, trellis_lat);

    /* Seg1: processes data[seg_size - overlap .. seg_size + seg_size)
     * Seeded from the INITIAL state (same as seg0 start) — simulates
     * what happens when both segments start from persistent SDM state. */
    sdm_context_t seg1;
    sdm_context_init(&seg1, f, f->order, trellis_cands, trellis_lat);

    /* To properly simulate: first warm up both SDMs to a steady state
     * by processing some data, then copy that state as the "seed point" */
    size_t warmup = 4096;
    float *warmup_out = (float *)malloc(warmup * sizeof(float));
    sdm_process_block(&seg0, input, warmup_out, warmup);
    sdm_process_block(&seg1, input, warmup_out, warmup);  /* Same warmup = same state */
    free(warmup_out);

    /* Now seg0 and seg1 have identical state — this is our seed point.
     * Save this state, then process divergent data. */
    sdm_context_t seed_state;
    sdm_context_copy_state(&seed_state, &seg0);

    /* Ensure enough data */
    size_t data_offset = warmup;
    size_t seg0_input_count = seg_size + overlap;
    size_t seg1_warmup_start = seg_size - overlap;  /* relative to data_offset */
    size_t seg1_input_count = seg_size + overlap;    /* warmup + nominal */

    if (data_offset + seg0_input_count > total_count ||
        data_offset + seg1_warmup_start + seg1_input_count > total_count) {
        *out_best_run = -1;
        sdm_context_free(&seg0);
        sdm_context_free(&seg1);
        sdm_context_free(&seed_state);
        return;
    }

    /* Re-seed both from the same state */
    sdm_context_copy_state(&seg0, &seed_state);
    sdm_context_copy_state(&seg1, &seed_state);

    /* Process seg0: data[offset .. offset + seg_size + overlap) */
    float *out0 = (float *)malloc(seg0_input_count * sizeof(float));
    size_t n0 = sdm_process_block(&seg0, input + data_offset, out0, seg0_input_count);

    /* Process seg1: data[offset + seg_size - overlap .. offset + 2*seg_size)
     * First 'overlap' samples are warmup (seg1 sees different data than seg0 saw
     * for this region). Then 'seg_size' nominal samples. */
    float *out1 = (float *)malloc(seg1_input_count * sizeof(float));

    /* Seg1 sees completely different data for its warmup region
     * (in reality: the next chunk's data, but starting from same seed). */
    size_t n1 = sdm_process_block(&seg1, input + data_offset + seg1_warmup_start,
                                  out1, seg1_input_count);

    /* The overlap region:
     * seg0 output[seg_size .. seg_size + overlap) = seg0's extra output
     * seg1 output[0 .. overlap) = seg1's warmup region output
     *
     * Both processed the SAME input data[offset + seg_size .. offset + seg_size + overlap)
     * but arrived from different internal states. */

    /* But wait — seg1 had warmup discard. In the real code, discard = overlap - lat.
     * Seg1 processes overlap samples of warmup, discards first (overlap-lat), keeps last lat.
     * Then produces seg_size nominal samples.
     * So seg1's first 'lat' output samples correspond to the last 'lat' of the overlap input.
     *
     * Actually, for this diagnostic we process WITHOUT discard to see the full picture.
     * seg1 output[0..overlap) covers the overlap input region.
     * seg0 output[seg_size..seg_size+overlap) also covers the same input region. */

    size_t ovl_len = overlap;
    if (ovl_len > n0 - seg_size) ovl_len = n0 - seg_size;
    if (ovl_len > n1) ovl_len = n1;

    float *ovl0 = out0 + seg_size;  /* seg0's overlap output */
    float *ovl1 = out1;             /* seg1's overlap output */

    /* --- Metric 1: Bit matching (current algorithm) --- */
    int best_run = 0, best_pos = 0, match_count = 0;
    for (size_t p = 0; p < ovl_len; p++) {
        if (ovl0[p] == ovl1[p]) {
            match_count++;
            int run = 1;
            while (p + (size_t)run < ovl_len && ovl0[p + run] == ovl1[p + run])
                run++;
            if (run > best_run) {
                best_run = run;
                best_pos = (int)p;
            }
        }
    }

    /* --- Metric 2: State distance at each sample in overlap --- */
    /* Re-process overlap sample-by-sample to get per-sample state distance.
     * Re-seed both SDMs and re-process to get to the overlap start. */
    sdm_context_t probe0, probe1;
    sdm_context_copy_state(&probe0, &seed_state);
    sdm_context_copy_state(&probe1, &seed_state);

    /* Advance probe0 to overlap start (process seg0's nominal data) */
    float *tmp_out = (float *)malloc(seg_size * sizeof(float));
    sdm_process_block(&probe0, input + data_offset, tmp_out, seg_size);
    free(tmp_out);

    /* Advance probe1 through its warmup (data before the overlap region) */
    tmp_out = (float *)malloc(seg1_warmup_start * sizeof(float));
    sdm_process_block(&probe1, input + data_offset + seg1_warmup_start,
                      tmp_out, 0);  /* Actually need to process the warmup */
    free(tmp_out);

    /* Hmm, probe1 needs to process its warmup input which is the overlap region itself.
     * Let me re-think: both probes need to arrive at the start of the shared input region. */

    /* Actually simpler approach: just use sdm_state_distance on the full contexts
     * that we already have after processing. They represent the state at the END of
     * their respective processing runs. For per-sample distance we'd need to
     * re-process sample by sample. Let's do that. */

    /* Re-seed and process sample-by-sample through the overlap */
    sdm_context_copy_state(&probe0, &seed_state);
    sdm_context_copy_state(&probe1, &seed_state);

    /* Process probe0 through its pre-overlap data (seg0's nominal region) */
    {
        float *bulk = (float *)malloc(seg_size * sizeof(float));
        sdm_process_block(&probe0, input + data_offset, bulk, seg_size);
        free(bulk);
    }

    /* Process probe1 through its warmup (seg1_warmup_start samples before overlap) */
    if (seg1_warmup_start > 0) {
        /* seg1 starts at data_offset + seg_size - overlap, needs to process
         * (seg_size - overlap) samples = seg1_warmup_start ... wait, that's 0
         * if seg1_warmup_start = seg_size - overlap.
         * Actually seg1 warmup = overlap samples before the nominal boundary.
         * seg1 processes data[seg_size - overlap .. seg_size + seg_size].
         * The shared input region is data[seg_size .. seg_size + overlap].
         * seg1's warmup is data[seg_size - overlap .. seg_size] = overlap samples. */

        /* But probe1 starts from seed state (same as seg0 start).
         * Seg1 in real code also starts from seed state and processes
         * data[seg_size - overlap .. ...]. So probe1 needs to process
         * data[seg_size - overlap .. seg_size] = overlap samples of warmup. */
        float *bulk = (float *)malloc(overlap * sizeof(float));
        sdm_process_block(&probe1, input + data_offset + seg_size - overlap,
                          bulk, overlap);
        free(bulk);
    }

    /* Now both probes are at the start of the shared overlap region.
     * Process sample-by-sample and measure state distance. */
    double min_dist = 1e30;
    int min_dist_pos = 0;
    float s0, s1;

    for (size_t p = 0; p < ovl_len; p++) {
        double sample = input[data_offset + seg_size + p];

        /* Process one sample in each probe */
        sdm_process_block(&probe0, &sample, &s0, 1);
        sdm_process_block(&probe1, &sample, &s1, 1);

        double dist = sdm_state_distance(&probe0, &probe1);
        if (dist < min_dist) {
            min_dist = dist;
            min_dist_pos = (int)p;
        }
    }

    /* Distance at the bit-matching stitch point */
    double dist_at_bit = 1e30;
    if (best_run > 0 && best_pos < (int)ovl_len) {
        /* Re-process to get distance at best_pos */
        sdm_context_copy_state(&probe0, &seed_state);
        sdm_context_copy_state(&probe1, &seed_state);
        {
            float *bulk = (float *)malloc(seg_size * sizeof(float));
            sdm_process_block(&probe0, input + data_offset, bulk, seg_size);
            free(bulk);
        }
        if (overlap > 0) {
            float *bulk = (float *)malloc(overlap * sizeof(float));
            sdm_process_block(&probe1, input + data_offset + seg_size - overlap,
                              bulk, overlap);
            free(bulk);
        }
        /* Advance to best_pos */
        for (int p = 0; p <= best_pos; p++) {
            double sample = input[data_offset + seg_size + p];
            sdm_process_block(&probe0, &sample, &s0, 1);
            sdm_process_block(&probe1, &sample, &s1, 1);
        }
        dist_at_bit = sdm_state_distance(&probe0, &probe1);
    }

    *out_best_run = best_run;
    *out_best_run_pos = best_pos;
    *out_min_dist = min_dist;
    *out_min_dist_pos = min_dist_pos;
    *out_dist_at_bit_stitch = dist_at_bit;
    *out_match_count = match_count;

    sdm_context_free(&seg0);
    sdm_context_free(&seg1);
    sdm_context_free(&seed_state);
    sdm_context_free(&probe0);
    sdm_context_free(&probe1);
    free(out0);
    free(out1);
}

/* ─── Overlap sweep diagnostic ─── */

static void test_stitch_convergence_sweep(void) {
    /* Test at DSD512 (the primary parallel target) with lat=32 */
    unsigned dsd_rate = DSD_RATE_512;
    int lat = 32;
    int cands = 2;  /* nc=2 for same-rate DSD512 */
    size_t seg_size = 16384;  /* Typical segment size */

    /* Generate enough input data */
    size_t max_overlap = (size_t)lat * 32;
    size_t total = 4096 + seg_size + max_overlap + seg_size;
    double *input = (double *)malloc(total * sizeof(double));
    gen_sine(input, total, dsd_rate);

    printf("\n  Overlap convergence sweep (DSD512, lat=%d, cands=%d, seg=%zu):\n",
           lat, cands, seg_size);
    printf("  %8s  %8s %8s  %10s %8s  %10s  %6s\n",
           "overlap", "best_run", "run_pos", "min_dist", "dist_pos",
           "dist@bit", "match%");

    int multipliers[] = { 2, 4, 8, 16, 32 };
    for (int m = 0; m < 5; m++) {
        size_t overlap = (size_t)lat * multipliers[m];
        int best_run, best_pos, min_dist_pos, match_count;
        double min_dist, dist_at_bit;

        measure_convergence(dsd_rate, lat, cands, overlap, seg_size,
                            input, total,
                            &best_run, &best_pos, &min_dist, &min_dist_pos,
                            &dist_at_bit, &match_count);

        double match_pct = 100.0 * (double)match_count / (double)overlap;
        printf("  %4zux%-3d  %8d %8d  %10.4f %8d  %10.4f  %5.1f%%\n",
               (size_t)multipliers[m], lat,
               best_run, best_pos, min_dist, min_dist_pos,
               dist_at_bit, match_pct);

        TEST_ASSERT(best_run >= 0, "convergence measurement should succeed");
    }

    free(input);
}

/* Same test with noise input (more realistic than pure sine) */
static void test_stitch_convergence_noise(void) {
    unsigned dsd_rate = DSD_RATE_512;
    int lat = 32;
    int cands = 2;
    size_t seg_size = 16384;

    size_t max_overlap = (size_t)lat * 32;
    size_t total = 4096 + seg_size + max_overlap + seg_size;
    double *input = (double *)malloc(total * sizeof(double));
    gen_noise(input, total, 42);

    printf("\n  Noise input convergence (DSD512, lat=%d, cands=%d):\n",
           lat, cands);
    printf("  %8s  %8s %8s  %10s %8s  %10s  %6s\n",
           "overlap", "best_run", "run_pos", "min_dist", "dist_pos",
           "dist@bit", "match%");

    int multipliers[] = { 2, 4, 8, 16, 32 };
    for (int m = 0; m < 5; m++) {
        size_t overlap = (size_t)lat * multipliers[m];
        int best_run, best_pos, min_dist_pos, match_count;
        double min_dist, dist_at_bit;

        measure_convergence(dsd_rate, lat, cands, overlap, seg_size,
                            input, total,
                            &best_run, &best_pos, &min_dist, &min_dist_pos,
                            &dist_at_bit, &match_count);

        double match_pct = 100.0 * (double)match_count / (double)overlap;
        printf("  %4zux%-3d  %8d %8d  %10.4f %8d  %10.4f  %5.1f%%\n",
               (size_t)multipliers[m], lat,
               best_run, best_pos, min_dist, min_dist_pos,
               dist_at_bit, match_pct);
    }

    free(input);
}

/* Test at DSD256 (2 segments) */
static void test_stitch_convergence_dsd256(void) {
    unsigned dsd_rate = DSD_RATE_256;
    int lat = 32;
    int cands = 2;
    size_t seg_size = 16384;

    size_t max_overlap = (size_t)lat * 32;
    size_t total = 4096 + seg_size + max_overlap + seg_size;
    double *input = (double *)malloc(total * sizeof(double));
    gen_sine(input, total, dsd_rate);

    printf("\n  Overlap convergence sweep (DSD256, lat=%d, cands=%d):\n",
           lat, cands);
    printf("  %8s  %8s %8s  %10s %8s  %10s  %6s\n",
           "overlap", "best_run", "run_pos", "min_dist", "dist_pos",
           "dist@bit", "match%");

    int multipliers[] = { 2, 4, 8, 16, 32 };
    for (int m = 0; m < 5; m++) {
        size_t overlap = (size_t)lat * multipliers[m];
        int best_run, best_pos, min_dist_pos, match_count;
        double min_dist, dist_at_bit;

        measure_convergence(dsd_rate, lat, cands, overlap, seg_size,
                            input, total,
                            &best_run, &best_pos, &min_dist, &min_dist_pos,
                            &dist_at_bit, &match_count);

        double match_pct = 100.0 * (double)match_count / (double)overlap;
        printf("  %4zux%-3d  %8d %8d  %10.4f %8d  %10.4f  %5.1f%%\n",
               (size_t)multipliers[m], lat,
               best_run, best_pos, min_dist, min_dist_pos,
               dist_at_bit, match_pct);
    }

    free(input);
}

/* Compare stitch quality: bit-matching vs state-distance */
static void test_stitch_quality_comparison(void) {
    unsigned dsd_rate = DSD_RATE_512;
    int lat = 32;
    int cands = 2;
    size_t seg_size = 16384;
    size_t overlap = (size_t)lat * 4;  /* Current default: 4x */

    size_t total = 4096 + seg_size + overlap + seg_size + 20000;  /* extra for phase shifts */
    double *input = (double *)malloc(total * sizeof(double));
    gen_sine(input, total, dsd_rate);

    printf("\n  Stitch quality comparison (DSD512, overlap=%zux%d=%zu):\n",
           (size_t)4, lat, overlap);

    /* Run multiple trials with different input phases */
    printf("  %6s  %8s %8s  %10s %8s  %8s\n",
           "trial", "best_run", "bit_pos", "min_dist", "dist_pos", "delta");

    int bit_better = 0, dist_better = 0, same = 0;
    int num_trials = 10;

    for (int trial = 0; trial < num_trials; trial++) {
        /* Shift input phase for each trial */
        size_t phase_shift = (size_t)trial * 1500;
        double *shifted = input + phase_shift;
        size_t shifted_total = total - phase_shift;

        int best_run, best_pos, min_dist_pos, match_count;
        double min_dist, dist_at_bit;

        measure_convergence(dsd_rate, lat, cands, overlap, seg_size,
                            shifted, shifted_total,
                            &best_run, &best_pos, &min_dist, &min_dist_pos,
                            &dist_at_bit, &match_count);

        if (best_run < 0) continue;

        int delta = abs(best_pos - min_dist_pos);
        printf("  %6d  %8d %8d  %10.4f %8d  %8d\n",
               trial, best_run, best_pos, min_dist, min_dist_pos, delta);

        /* Count: when do the methods disagree? */
        if (best_pos == min_dist_pos)
            same++;
        else if (dist_at_bit <= min_dist * 1.1)  /* bit stitch has similar distance */
            bit_better++;
        else
            dist_better++;
    }

    printf("\n  Agreement: %d/10, bit-match preferred: %d, state-dist preferred: %d\n",
           same, bit_better, dist_better);

    free(input);
}

/* ─── Suite registration ─── */

void test_stitch_suite(void) {
    if (!test_should_run_suite("stitch")) return;
    TEST_SUITE("Stitch Convergence");

    TEST_RUN(test_stitch_convergence_sweep);
    TEST_RUN(test_stitch_convergence_noise);
    TEST_RUN(test_stitch_convergence_dsd256);
    TEST_RUN(test_stitch_quality_comparison);
}
