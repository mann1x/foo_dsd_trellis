/*
 * GPU vs CPU trellis algorithm sample-by-sample comparison.
 * Processes a small number of samples through both and finds divergence.
 */

#include "test.h"
#include "../include/trellis.h"
#include "../include/ntf.h"
#include "../include/gpu_compute.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Emulate the GPU kernel's per-sample algorithm on CPU.
 * This replicates the EXACT logic from sdm_parallel.cu. */
typedef struct {
    double state[8][8];  /* [candidate][order] */
    double cost[8];
    unsigned path[8];
    unsigned char hist[8][128]; /* [candidate][hist_bytes] */
    unsigned next_stored[8]; /* traceback from previous iteration (CPU pipeline) */
    int active;
    int hist_pos;
    int pending;
    int nc;
    int order;
    int lat;
    const double *a;
    const double *g;
} gpu_emu_t;

static void gpu_emu_init(gpu_emu_t *e, const ntf_filter_t *f, int nc, int lat) {
    memset(e, 0, sizeof(*e));
    e->nc = nc;
    e->order = f->order;
    e->lat = lat;
    e->a = f->a;
    e->g = f->g;
    e->active = 1;  /* match CPU: starts with 1 candidate, grows after first sort */
}

/* Returns the output bit (0 or 1) */
static int gpu_emu_sample(gpu_emu_t *e, double x_raw,
                           double *out_v, double *out_cost0, double *out_cost1,
                           double *out_state0_0, double *out_state1_0) {
    double x = x_raw * 0.5;
    int ac = e->active;
    int order = e->order;
    int lat = e->lat;

    /* Phase 0: Read CURRENT traceback from history.
     * Children will inherit STORED traceback (from previous post-sort),
     * matching the CPU's 1-step pipeline delay. */
    unsigned cur_tb[8];  /* current traceback (for post-sort saving) */
    for (int t = 0; t < ac; t++) {
        if (e->pending >= lat) {
            int next_pos = (e->hist_pos + 1) % lat;
            int nb = next_pos / 8;
            int ni = next_pos % 8;
            cur_tb[t] = (e->hist[t][nb] >> ni) & 1;
        } else {
            cur_tb[t] = 0;
        }
    }

    /* Phase 1: expand candidates */
    double c_state[16][8];
    double c_cost[16];
    unsigned c_bit[16];
    unsigned c_path[16];
    unsigned c_next[16];
    unsigned c_pi[16]; /* parent index for post-sort traceback */
    unsigned char c_hist[16][128];

    for (int t = 0; t < 2 * ac; t++) {
        int pi = t / 2;
        double y_b = (t & 1) ? -1.0 : 1.0;
        double d[8];

        /* NTF calc — exact copy from GPU kernel */
        d[0] = e->state[pi][0] - e->g[0] * e->state[pi][1] + x;
        for (int k = 1; k < order - 1; k++)
            d[k] = e->state[pi][k] + e->state[pi][k-1] - e->g[k] * e->state[pi][k+1];
        d[order-1] = e->state[pi][order-1] + e->state[pi][order-2];
        double v = x;
        for (int k = 0; k < order; k++)
            v += e->a[k] * d[k];

        d[0] += y_b;
        /* No state limit (limit=0) */

        for (int k = 0; k < order; k++)
            c_state[t][k] = d[k];
        c_cost[t] = e->cost[pi] + (v + e->a[0]*y_b)*(v + e->a[0]*y_b);
        c_bit[t] = t & 1;
        c_path[t] = (e->path[pi] << 1 | c_bit[t]) & 0xFF;
        c_next[t] = e->next_stored[pi]; /* inherit STORED, not current */
        c_pi[t] = pi; /* track parent */
        memcpy(c_hist[t], e->hist[pi], sizeof(c_hist[0]));

        if (t == 0) {
            *out_v = v;
            *out_cost0 = c_cost[0];
            *out_state0_0 = c_state[0][0];
        }
        if (t == 1) {
            *out_cost1 = c_cost[1];
            *out_state1_0 = c_state[1][0];
        }
    }

    /* Phase 2: sort + dedup + next-filter (matching CPU algorithm) */

    /* Majority vote for traceback: count which next bit has more votes */
    int tc = 2 * ac;
    {
        int votes[2] = {0, 0};
        int best_idx = 0;
        for (int i = 0; i < tc; i++) {
            votes[c_next[i] & 1]++;
            if (c_cost[i] < c_cost[best_idx]) best_idx = i;
        }
        unsigned majority = (votes[1] > votes[0]) ? 1 : 0;
        unsigned min_next = c_next[best_idx];

        /* If best disagrees with majority, find best with majority */
        if (min_next != majority) {
            int best_maj = -1;
            for (int i = 0; i < tc; i++) {
                if (c_next[i] == majority &&
                    (best_maj < 0 || c_cost[i] < c_cost[best_maj]))
                    best_maj = i;
            }
            if (best_maj >= 0 && c_cost[best_maj] < c_cost[best_idx] * 1.1)
                min_next = majority;
        }

        /* Next-filter: drop candidates that disagree with chosen next */
        int filtered = 0;
        for (int i = 0; i < tc; i++) {
            if (c_next[i] == min_next) {
                if (filtered != i) {
                    c_cost[filtered] = c_cost[i];
                    c_bit[filtered] = c_bit[i];
                    c_path[filtered] = c_path[i];
                    c_next[filtered] = c_next[i];
                    c_pi[filtered] = c_pi[i];
                    for (int k = 0; k < e->order; k++)
                        c_state[filtered][k] = c_state[i][k];
                    memcpy(c_hist[filtered], c_hist[i], sizeof(c_hist[0]));
                }
                filtered++;
            }
        }
        tc = filtered;
    }

    /* Dedup-aware greedy selection (matches GPU kernel + CPU sdm_sort_cands) */
    {
        int selected = 0;
        unsigned used_paths[8];
        for (int slot = 0; slot < e->nc && slot < tc; slot++) {
            int best = -1;
            for (int j = 0; j < tc; j++) {
                if (c_cost[j] < 0.0) continue;
                int dup = 0;
                for (int k = 0; k < selected; k++)
                    if (c_path[j] == used_paths[k]) { dup = 1; break; }
                if (dup) continue;
                if (best < 0 || c_cost[j] < c_cost[best])
                    best = j;
            }
            if (best < 0) break;
            used_paths[selected] = c_path[best];
            if (best != selected) {
                double t_c = c_cost[selected]; c_cost[selected] = c_cost[best]; c_cost[best] = t_c;
                unsigned t_b = c_bit[selected]; c_bit[selected] = c_bit[best]; c_bit[best] = t_b;
                unsigned t_p = c_path[selected]; c_path[selected] = c_path[best]; c_path[best] = t_p;
                unsigned t_n = c_next[selected]; c_next[selected] = c_next[best]; c_next[best] = t_n;
                unsigned t_pi2 = c_pi[selected]; c_pi[selected] = c_pi[best]; c_pi[best] = t_pi2;
                for (int k = 0; k < order; k++) {
                    double t_s = c_state[selected][k]; c_state[selected][k] = c_state[best][k]; c_state[best][k] = t_s;
                }
                unsigned char t_h[128];
                memcpy(t_h, c_hist[selected], sizeof(t_h));
                memcpy(c_hist[selected], c_hist[best], sizeof(t_h));
                memcpy(c_hist[best], t_h, sizeof(t_h));
            }
            selected++;
        }
        ac = selected;
    }

    /* Output = traceback of best candidate (STORED from previous iteration) */
    unsigned output_bit = c_next[0];

    /* Record bits in history */
    int byte_pos = e->hist_pos / 8;
    int bit_pos = e->hist_pos % 8;
    for (int i = 0; i < ac; i++) {
        if (c_bit[i])
            c_hist[i][byte_pos] |= (1u << bit_pos);
        else
            c_hist[i][byte_pos] &= ~(1u << bit_pos);
    }

    e->hist_pos = (e->hist_pos + 1) % lat;
    if (e->pending < lat) e->pending++;

    /* Promote children to parents.
     * CPU pipeline: s->next = s->parent->next (parent's CURRENT traceback).
     * Save parent's current traceback as next_stored for next iteration. */
    e->active = ac;
    double min_c = c_cost[0];
    for (int i = 0; i < ac; i++) {
        e->cost[i] = c_cost[i] - min_c;
        e->path[i] = c_path[i];
        e->next_stored[i] = cur_tb[c_pi[i]]; /* parent's current traceback */
        for (int k = 0; k < order; k++)
            e->state[i][k] = c_state[i][k];
        memcpy(e->hist[i], c_hist[i], sizeof(e->hist[0]));
    }

    return (int)output_bit;
}

static void test_gpu_cpu_divergence(void) {
    uint32_t dsd_rate = 11289600;  /* DSD128 */
    int nc = 2, lat = 128;
    const ntf_filter_t *f = ntf_auto_select(dsd_rate);
    if (!f) { printf("    no NTF for DSD128\n"); return; }

    printf("    DSD128: order=%d nc=%d lat=%d\n", f->order, nc, lat);

    /* CPU reference */
    sdm_context_t cpu;
    sdm_context_init(&cpu, f, f->order, nc, lat);

    /* GPU emulator */
    gpu_emu_t gpu;
    gpu_emu_init(&gpu, f, nc, lat);

    /* Test input: small constant (mimics boxcar near-silence) */
    double input_val = 0.01;

    /* Trace early samples: what bit is written to history at each sample? */
    printf("    Early sample trace (bit written to history):\n");

    int first_diff = -1;
    for (int s = 0; s < 300; s++) {
/* CPU */
        float cpu_out_f;
        double cpu_in = input_val;
        size_t cpu_n = sdm_process_block(&cpu, &cpu_in, &cpu_out_f, 1);
        int cpu_bit = (cpu_out_f > 0) ? 1 : 0;

        /* GPU emulator */
        double gpu_v, gpu_c0, gpu_c1, gpu_s0, gpu_s1;
        int gpu_bit = gpu_emu_sample(&gpu, input_val,
                                      &gpu_v, &gpu_c0, &gpu_c1, &gpu_s0, &gpu_s1);

        /* During latency fill, CPU produces no output.
         * But both CPU and GPU ARE recording history bits. */
        if (s < 5 || s == lat - 1 || s == lat || s == lat + 1) {
            /* Get CPU's recorded bit (path & 1 of best candidate after sort) */
            int cpu_bank_pre = cpu.idx & 1;
            sdm_state_t *cpu_best_pre = cpu.trellis[cpu_bank_pre].act[0];
            sdm_state_t *cpu_sec_pre = (cpu.num_cands > 1) ? cpu.trellis[cpu_bank_pre].act[1] : NULL;
            unsigned cpu_recorded = cpu_best_pre ? (cpu_best_pre->path & 1) : 99;
            unsigned gpu_recorded = gpu.path[0] & 1;
            printf("    s=%3d: cpu_bit_recorded=%u gpu_bit_recorded=%u %s",
                   s, cpu_recorded, gpu_recorded,
                   cpu_recorded == gpu_recorded ? "SAME" : "DIFF");
            if (cpu_best_pre) {
                printf("  cpu_cost=%.6f cpu_path=%u gpu_path=%u",
                       cpu_best_pre->cost, cpu_best_pre->path, gpu.path[0]);
            }
            printf("\n");
        }

        if (cpu_n == 0) {
            continue;
        }

        int match = (gpu_bit == cpu_bit);
        float gpu_out = gpu_bit ? 1.0f : -1.0f;

        if (!match && first_diff < 0) {
            first_diff = s;
            printf("    *** FIRST DIVERGENCE at sample %d ***\n", s);
            printf("    CPU output: %d (%.1f)  GPU output: %d (%.1f)\n",
                   cpu_bit, cpu_out_f, gpu_bit, gpu_out);

            /* Trace traceback: what history bit was read? */
            int cpu_bank = cpu.idx & 1;
            unsigned cpu_next = cpu.trellis[cpu_bank].act[0] ?
                cpu.trellis[cpu_bank].act[0]->next : 99;
            /* GPU traceback: hist[0] at position (hist_pos+1)%lat was read
             * BEFORE the expansion at this sample */
            int tb_pos = (gpu.hist_pos) % lat;  /* hist_pos was already incremented */
            int tb_byte = tb_pos / 8;
            int tb_bit = tb_pos % 8;
            unsigned gpu_hist_bit = (gpu.hist[0][tb_byte] >> tb_bit) & 1;
            printf("    CPU next=%u, GPU hist_bit=%u at pos=%d (byte=%d bit=%d)\n",
                   cpu_next, gpu_hist_bit, tb_pos, tb_byte, tb_bit);

            /* Dump history bytes for both */
            printf("    GPU hist[0] bytes 0-3: %02x %02x %02x %02x\n",
                   gpu.hist[0][0], gpu.hist[0][1], gpu.hist[0][2], gpu.hist[0][3]);
        }

        if (s < lat + 5 || (first_diff >= 0 && s <= first_diff + 5)) {
            /* Also get CPU internal state for comparison */
            int cpu_bank = cpu.idx & 1;
            sdm_state_t *cpu_best = cpu.trellis[cpu_bank].act[0];
            double cpu_s0 = cpu_best ? cpu_best->state[0] : -999;
            double cpu_cost = cpu_best ? cpu_best->cost : -999;
            unsigned cpu_path = cpu_best ? cpu_best->path : 0;

            printf("    s=%3d: CPU=%d GPU=%d %s  v=%.6f c0=%.4f c1=%.4f"
                   "  cpu_s0=%.6f gpu_s0=%.6f cpu_cost=%.4f cpu_path=%u gpu_path=%u"
                   "  gpu_hp=%d cpu_pos=%d\n",
                   s, cpu_bit, gpu_bit, match ? "OK" : "DIFF",
                   gpu_v, gpu_c0, gpu_c1,
                   cpu_s0, gpu.state[0][0], cpu_cost, cpu_path, gpu.path[0],
                   gpu.hist_pos, cpu.pos);
        }
    }

    if (first_diff < 0)
        printf("    All 300 samples match!\n");
    else
        printf("    First divergence at sample %d\n", first_diff);

    sdm_context_free(&cpu);
}

/* Measure SINAD of GPU emulator with different sort orders */
static void test_gpu_sort_sinad(void) {
    uint32_t dsd_rate = 11289600;
    int nc = 2, lat_val = 128;
    size_t N = 500000;
    const ntf_filter_t *f = ntf_auto_select(dsd_rate);
    if (!f) return;

    /* Generate sine test signal */
    double freq = 1000.0;
    double *input = (double *)malloc(N * sizeof(double));
    for (size_t i = 0; i < N; i++)
        input[i] = 0.01 * sin(2.0 * M_PI * freq * (double)i / (double)dsd_rate);

    /* CPU reference */
    sdm_context_t cpu;
    sdm_context_init(&cpu, f, f->order, nc, lat_val);
    float *cpu_out = (float *)calloc(N, sizeof(float));
    sdm_process_block(&cpu, input, cpu_out, N);
    sdm_context_free(&cpu);

    /* GPU emulator with < sort (current GPU) */
    gpu_emu_t gpu_lt;
    gpu_emu_init(&gpu_lt, f, nc, lat_val);
    float *gpu_lt_out = (float *)calloc(N, sizeof(float));
    for (size_t s = 0; s < N; s++) {
        double v, c0, c1, s0, s1;
        /* Temporarily switch to < sort */
        int bit = gpu_emu_sample(&gpu_lt, input[s], &v, &c0, &c1, &s0, &s1);
        gpu_lt_out[s] = bit ? 1.0f : -1.0f;
    }

    /* Compare noise via decimation */
    size_t dec = 128;
    size_t pcm_n = N / dec;
    float *cpu_pcm = (float *)malloc(pcm_n * sizeof(float));
    float *gpu_pcm = (float *)malloc(pcm_n * sizeof(float));
    for (size_t i = 0; i < pcm_n; i++) {
        double sc = 0, sg = 0;
        for (size_t j = 0; j < dec; j++) {
            sc += cpu_out[i * dec + j];
            sg += gpu_lt_out[i * dec + j];
        }
        cpu_pcm[i] = (float)(sc / dec);
        gpu_pcm[i] = (float)(sg / dec);
    }

    /* RMS comparison */
    double cpu_rms = 0, gpu_rms = 0;
    for (size_t i = 0; i < pcm_n; i++) {
        cpu_rms += cpu_pcm[i] * cpu_pcm[i];
        gpu_rms += gpu_pcm[i] * gpu_pcm[i];
    }
    cpu_rms = sqrt(cpu_rms / pcm_n);
    gpu_rms = sqrt(gpu_rms / pcm_n);

    printf("    Sort SINAD test (500K samples, DSD128):\n");
    printf("    CPU RMS: %.8f\n", cpu_rms);
    printf("    GPU emu RMS (<=sort): %.8f\n", gpu_rms);
    printf("    Ratio: %.2f dB\n", 20.0 * log10(gpu_rms / (cpu_rms + 1e-30)));

    free(input); free(cpu_out); free(gpu_lt_out);
    free(cpu_pcm); free(gpu_pcm);
}

void test_gpu_kernel_debug_suite(void) {
    if (!test_should_run_suite("gpudebug")) return;
    printf("\n=== GPU Kernel Debug ===\n");
    TEST_RUN(test_gpu_cpu_divergence);
    TEST_RUN(test_gpu_sort_sinad);
    g_tests_run++; g_tests_passed++;
}
