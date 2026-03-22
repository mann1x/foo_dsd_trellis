/*
 * 1.5-bit SDM prototype: 3-level quantizer (-1, 0, +1).
 * Compares quality metrics with standard 1-bit trellis SDM.
 *
 * The 1.5-bit SDM has a smaller quantization step (1.0 vs 2.0),
 * reducing noise modulation. The output is converted to 1-bit
 * via a simple 1st-order SDM for DSD compatibility.
 */

#include "test.h"
#include "../include/trellis.h"
#include "../include/ntf.h"
#include "../include/fir.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── 1.5-bit (3-level) Trellis SDM ─── */

typedef struct {
    double state[8];     /* NTF integrator state */
    double cost;         /* accumulated error */
    unsigned path;       /* recent output decisions */
} cand_1_5_t;

typedef struct {
    cand_1_5_t cands[32]; /* max candidates */
    int nc;               /* active candidates */
    int order;
    const double *a;
    const double *g;
    double state_limit;
} sdm_1_5bit_t;

static void sdm_1_5bit_init(sdm_1_5bit_t *s, const ntf_filter_t *f,
                              int nc, double limit) {
    memset(s, 0, sizeof(*s));
    s->nc = nc;
    s->order = f->order;
    s->a = f->a;
    s->g = f->g;
    s->state_limit = limit;
    /* Start with 1 candidate at zero state */
    s->cands[0].cost = 0;
}

/* Process one sample through the 1.5-bit trellis.
 * Returns the quantized output: -1.0, 0.0, or +1.0 */
static double sdm_1_5bit_sample(sdm_1_5bit_t *s, double x) {
    int order = s->order;
    int nc = s->nc;

    /* Expand: each candidate produces 3 children */
    cand_1_5_t children[96]; /* max 32 * 3 */
    int n_children = 0;
    int ac = 1;
    for (int i = 0; i < 32 && i < nc; i++) {
        if (s->cands[i].cost < 1e20) ac = i + 1;
    }

    for (int i = 0; i < ac; i++) {
        for (int q = -1; q <= 1; q++) {  /* 3 levels: -1, 0, +1 */
            double y = (double)q;
            cand_1_5_t *c = &children[n_children];

            /* NTF filter calc */
            double d[8];
            d[0] = s->cands[i].state[0] - s->g[0] * s->cands[i].state[1] + x;
            for (int k = 1; k < order - 1; k++)
                d[k] = s->cands[i].state[k] + s->cands[i].state[k-1]
                     - s->g[k] * s->cands[i].state[k+1];
            if (order > 1)
                d[order-1] = s->cands[i].state[order-1] + s->cands[i].state[order-2];

            double v = x;
            for (int k = 0; k < order; k++) v += s->a[k] * d[k];

            /* Apply feedback: d[0] -= y (since d[0] = ... + x - y) */
            d[0] -= y;

            /* State limiter */
            if (s->state_limit > 0) {
                for (int k = 0; k < order; k++) {
                    if (d[k] > s->state_limit) d[k] = s->state_limit;
                    else if (d[k] < -s->state_limit) d[k] = -s->state_limit;
                }
            }

            for (int k = 0; k < order; k++) c->state[k] = d[k];
            c->cost = s->cands[i].cost + (v - y) * (v - y);
            c->path = ((s->cands[i].path << 2) | ((unsigned)(q + 1) & 3)) & 0xFFFF;
            n_children++;
        }
    }

    /* Sort by cost, keep best nc */
    for (int i = 0; i < nc && i < n_children; i++) {
        int best = i;
        for (int j = i + 1; j < n_children; j++)
            if (children[j].cost < children[best].cost) best = j;
        if (best != i) {
            cand_1_5_t tmp = children[i];
            children[i] = children[best];
            children[best] = tmp;
        }
    }

    /* Path dedup */
    int kept = 1;
    for (int i = 1; i < nc && i < n_children; i++) {
        int dup = 0;
        for (int j = 0; j < kept; j++)
            if (children[i].path == children[j].path) { dup = 1; break; }
        if (!dup) {
            if (kept != i) children[kept] = children[i];
            kept++;
        }
    }

    /* Output = best candidate's decision */
    int best_q = ((int)(children[0].path & 3)) - 1;  /* decode: 0→-1, 1→0, 2→+1 */
    double output = (double)best_q;

    /* Normalize costs and copy to state */
    double min_cost = children[0].cost;
    for (int i = 0; i < kept; i++) {
        children[i].cost -= min_cost;
        s->cands[i] = children[i];
    }
    /* Clear unused */
    for (int i = kept; i < nc; i++)
        s->cands[i].cost = 1e30;

    return output;
}

/* Convert 1.5-bit stream to 1-bit via 1st-order error diffusion */
static void convert_1_5_to_1bit(const double *in_1_5, float *out_1bit,
                                  size_t count) {
    double state = 0.0;
    for (size_t i = 0; i < count; i++) {
        state += in_1_5[i];
        float bit = (state > 0.0) ? 1.0f : -1.0f;
        state -= (double)bit;
        out_1bit[i] = bit;
    }
}

/* ─── Quality metrics ─── */

static double measure_sinad_dsd(const float *dsd, size_t n, uint32_t rate,
                                 double test_freq) {
    /* Decimate to ~44.1 kHz via FIR chain */
    fir_chain_t fir;
    memset(&fir, 0, sizeof(fir));
    fir_chain_init(&fir, rate, 44100);
    float *pcm = (float *)calloc(n, sizeof(float));
    size_t pcm_n = fir_chain_process(&fir, dsd, pcm, n);
    fir_chain_free(&fir);

    size_t skip = 256;
    if (pcm_n <= skip + 1024) { free(pcm); return -999; }
    size_t mn = pcm_n - skip;

    /* Goertzel for signal power */
    double k = (double)mn * test_freq / 44100.0;
    double w = 2.0 * M_PI * k / (double)mn;
    double c = 2.0 * cos(w);
    double s1 = 0, s2 = 0;
    for (size_t i = 0; i < mn; i++) {
        double s0 = (double)pcm[skip + i] + c * s1 - s2;
        s2 = s1; s1 = s0;
    }
    double sig = 2.0 * (s1*s1 + s2*s2 - c*s1*s2) / ((double)mn * mn);

    double total = 0;
    for (size_t i = skip; i < pcm_n; i++)
        total += (double)pcm[i] * pcm[i];
    total /= (double)mn;
    double noise = total - sig;
    if (noise < 1e-30) noise = 1e-30;

    free(pcm);
    return 10.0 * log10(sig / noise);
}

/* Measure noise modulation: RMS of noise in low-signal vs high-signal regions */
static void measure_noise_mod(const float *dsd, size_t n, uint32_t rate,
                               double *nm_low, double *nm_high) {
    /* Decimate to audio rate */
    size_t dec = rate / 44100;
    size_t pcm_n = n / dec;
    float *pcm = (float *)malloc(pcm_n * sizeof(float));
    for (size_t i = 0; i < pcm_n; i++) {
        double sum = 0;
        for (size_t j = 0; j < dec; j++) sum += dsd[i * dec + j];
        pcm[i] = (float)(sum / dec);
    }

    /* Split into windows, classify by signal level */
    size_t win = 256;
    double sum_low = 0, sum_high = 0;
    int n_low = 0, n_high = 0;
    for (size_t i = 0; i + win < pcm_n; i += win) {
        /* Signal level = abs of mean */
        double mean = 0;
        for (size_t j = 0; j < win; j++) mean += pcm[i + j];
        mean /= win;
        /* Noise = variance around mean */
        double var = 0;
        for (size_t j = 0; j < win; j++) {
            double d = pcm[i + j] - mean;
            var += d * d;
        }
        var /= win;
        if (fabs(mean) < 0.01) {
            sum_low += var; n_low++;
        } else {
            sum_high += var; n_high++;
        }
    }

    *nm_low = (n_low > 0) ? sqrt(sum_low / n_low) : 0;
    *nm_high = (n_high > 0) ? sqrt(sum_high / n_high) : 0;
    free(pcm);
}

/* ─── Main comparison test ─── */

static void test_1_5bit_comparison(void) {
    uint32_t rates[] = {2822400, 5644800, 11289600, 22579200};
    const char *names[] = {"DSD64", "DSD128", "DSD256", "DSD512"};
    int nc = 2, lat = 0;  /* lat=0 = auto */
    double test_freq = 997.0;

    printf("\n  %-8s  %10s %10s %10s  %10s %10s %10s  %8s\n",
           "Rate", "1bit SINAD", "1.5b SINAD", "delta",
           "1b NM_lo", "1b NM_hi", "1.5 NM_lo", "1.5 NM_hi");

    for (int r = 0; r < 4; r++) {
        uint32_t rate = rates[r];
        size_t N = rate * 2;  /* 2 seconds */
        const ntf_filter_t *f = ntf_auto_select(rate);
        if (!f) continue;

        /* Bin-align frequency */
        int decimation = (int)(rate / 44100);
        size_t pcm_est = N / (size_t)decimation;
        double bin = round(test_freq * (double)pcm_est / 44100.0);
        double freq = bin * 44100.0 / (double)pcm_est;

        /* Generate test signal */
        double *input = (double *)malloc(N * sizeof(double));
        for (size_t i = 0; i < N; i++)
            input[i] = 0.5 * sin(2.0 * M_PI * freq * (double)i / (double)rate);

        /* 1-bit trellis SDM */
        sdm_context_t sdm1;
        int actual_lat = lat ? lat : (nc * 8);
        sdm_context_init(&sdm1, f, f->order, nc, actual_lat);
        float *out_1bit = (float *)calloc(N, sizeof(float));
        size_t n1 = sdm_process_block(&sdm1, input, out_1bit, N);
        sdm_context_free(&sdm1);

        /* 1.5-bit trellis SDM → 1-bit conversion */
        sdm_1_5bit_t sdm15;
        sdm_1_5bit_init(&sdm15, f, nc, 0.0);
        double *out_1_5raw = (double *)malloc(N * sizeof(double));
        for (size_t i = 0; i < N; i++)
            out_1_5raw[i] = sdm_1_5bit_sample(&sdm15, input[i] * 0.5);
        float *out_1_5bit = (float *)calloc(N, sizeof(float));
        convert_1_5_to_1bit(out_1_5raw, out_1_5bit, N);

        /* Measure SINAD */
        double sinad_1 = measure_sinad_dsd(out_1bit, n1, rate, freq);
        double sinad_15 = measure_sinad_dsd(out_1_5bit, N, rate, freq);

        /* Measure noise modulation */
        double nm1_lo, nm1_hi, nm15_lo, nm15_hi;
        measure_noise_mod(out_1bit, n1, rate, &nm1_lo, &nm1_hi);
        measure_noise_mod(out_1_5bit, N, rate, &nm15_lo, &nm15_hi);

        printf("  %-8s  %8.1f dB %8.1f dB %+7.1f dB  %8.6f %8.6f %8.6f  %8.6f\n",
               names[r], sinad_1, sinad_15, sinad_15 - sinad_1,
               nm1_lo, nm1_hi, nm15_lo, nm15_hi);

        free(input); free(out_1bit); free(out_1_5raw); free(out_1_5bit);
    }
}

void test_1_5bit_suite(void) {
    if (!test_should_run_suite("1.5bit")) return;
    printf("\n=== 1.5-bit vs 1-bit SDM Comparison ===\n");
    TEST_RUN(test_1_5bit_comparison);
    g_tests_run++; g_tests_passed++;
}
