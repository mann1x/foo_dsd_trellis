/*
 * foo_dsd_trellis — Hawksford intra-step parallel GPU Trellis SDM
 *
 * Single-segment kernel with dynamic nc (4-64), fully parallel NTF
 * evaluation within each step. Proper Viterbi traceback.
 * nc > 4 = better quality than CPU (which maxes at nc=4 real-time).
 */

extern "C" {

__constant__ double c_ntf_a[8];
__constant__ double c_ntf_g[8];
__constant__ int    c_ntf_order;

#define MAX_NC 64
#define MAX_NC2 128
#define TB_WORDS 8  /* 256 bits max traceback */

__device__ double ntf_calc_d(const double *s, double *d, int order, double x) {
    d[0] = s[0] - c_ntf_g[0] * s[1] + x;
    for (int k = 1; k < order - 1; k++)
        d[k] = s[k] + s[k - 1] - c_ntf_g[k] * s[k + 1];
    d[order - 1] = s[order - 1] + s[order - 2];
    double v = x;
    for (int k = 0; k < order; k++)
        v += c_ntf_a[k] * d[k];
    return v;
}

__global__ void trellis_hawksford(
    const float *in, float *out, int count,
    int num_cands,      /* runtime nc: 4, 8, 16, 32, or 64 */
    int trellis_lat,
    const double *init_states,
    const double *init_costs,
    double *final_states,
    double *final_costs)
{
    int tid = threadIdx.x;
    int order = c_ntf_order;
    int nc = num_cands;
    int lat = trellis_lat;
    int tb_words = (lat + 31) / 32;
    if (tb_words > TB_WORDS) tb_words = TB_WORDS;

    __shared__ double p_state[MAX_NC][8];
    __shared__ double p_cost[MAX_NC];
    __shared__ unsigned p_path[MAX_NC];
    __shared__ unsigned p_tb[MAX_NC][TB_WORDS];

    __shared__ double c_state[MAX_NC2][8];
    __shared__ double c_cost[MAX_NC2];
    __shared__ unsigned c_path[MAX_NC2];
    __shared__ unsigned c_bit[MAX_NC2];
    __shared__ unsigned c_tb[MAX_NC2][TB_WORDS];

    __shared__ int s_active;
    __shared__ unsigned s_output_bit;
    __shared__ int s_hist_pos;
    __shared__ int s_pending;

    /* Init parents */
    if (tid < nc) {
        if (init_states) {
            for (int k = 0; k < order; k++)
                p_state[tid][k] = init_states[tid * 8 + k];
            p_cost[tid] = init_costs ? init_costs[tid] : 0.0;
        } else {
            for (int k = 0; k < order; k++)
                p_state[tid][k] = 0.0;
            p_cost[tid] = 0.0;
        }
        p_path[tid] = 0;
        for (int w = 0; w < tb_words; w++)
            p_tb[tid][w] = 0;
    }
    if (tid == 0) {
        s_active = nc;
        s_hist_pos = 0;
        s_pending = 0;
    }
    __syncthreads();

    int out_idx = 0;

    for (int s = 0; s < count; s++) {
        double x = (double)in[s] * 0.5;
        int ac = s_active;

        /* ══════ Phase 1: Parallel NTF expansion (mixed precision) ══════
         * NTF state in double for noise shaping accuracy.
         * Intermediate calc uses double. Consumer GPU FP64 is 1/64 of FP32
         * but correctness requires it for the integrator accumulation. */
        if (tid < 2 * ac) {
            int pi = tid / 2;
            double y_b = (tid & 1) ? 1.0 : -1.0;
            double d[8];
            double v = ntf_calc_d(p_state[pi], d, order, x);
            d[0] += y_b;
            for (int k = 0; k < order; k++)
                c_state[tid][k] = d[k];
            double err = v + c_ntf_a[0] * y_b;
            c_cost[tid] = p_cost[pi] + err * err;
            c_bit[tid] = (tid & 1) ? 0u : 1u;
            c_path[tid] = (p_path[pi] << 1 | c_bit[tid]) & 0xFF;
            for (int w = 0; w < tb_words; w++)
                c_tb[tid][w] = p_tb[pi][w];
        }
        __syncthreads();

        /* ══════ Phase 2+3: Thread 0 — sort + dedup + traceback ══════ */
        if (tid == 0) {
            int tc = 2 * ac;
            /* Selection sort by cost (top ac only) */
            for (int i = 0; i < ac && i < tc; i++) {
                int best = i;
                for (int j = i + 1; j < tc; j++)
                    if (c_cost[j] < c_cost[best]) best = j;
                if (best != i) {
                    double t_c = c_cost[i]; c_cost[i] = c_cost[best]; c_cost[best] = t_c;
                    unsigned t_b = c_bit[i]; c_bit[i] = c_bit[best]; c_bit[best] = t_b;
                    unsigned t_p = c_path[i]; c_path[i] = c_path[best]; c_path[best] = t_p;
                    for (int k = 0; k < order; k++) {
                        double t_s = c_state[i][k]; c_state[i][k] = c_state[best][k]; c_state[best][k] = t_s;
                    }
                    for (int w = 0; w < tb_words; w++) {
                        unsigned t_h = c_tb[i][w]; c_tb[i][w] = c_tb[best][w]; c_tb[best][w] = t_h;
                    }
                }
            }
            /* Path dedup */
            int deduped = 1;
            for (int i = 1; i < ac; i++) {
                int dup = 0;
                for (int j = 0; j < deduped; j++) {
                    if (c_path[i] == c_path[j]) { dup = 1; break; }
                }
                if (!dup) {
                    if (deduped != i) {
                        c_cost[deduped] = c_cost[i];
                        c_bit[deduped] = c_bit[i];
                        c_path[deduped] = c_path[i];
                        for (int k = 0; k < order; k++)
                            c_state[deduped][k] = c_state[i][k];
                        for (int w = 0; w < tb_words; w++)
                            c_tb[deduped][w] = c_tb[i][w];
                    }
                    deduped++;
                }
            }
            ac = deduped;

            /* Record bit in traceback */
            int bp = s_hist_pos / 32;
            int bi = s_hist_pos % 32;
            for (int i = 0; i < ac; i++) {
                if (c_bit[i])
                    c_tb[i][bp] |= (1u << bi);
                else
                    c_tb[i][bp] &= ~(1u << bi);
            }

            /* Output: traceback */
            if (s_pending >= lat) {
                int np = (s_hist_pos + 1) % lat;
                s_output_bit = (c_tb[0][np / 32] >> (np % 32)) & 1;
            } else {
                s_output_bit = c_bit[0];
            }

            s_hist_pos = (s_hist_pos + 1) % lat;
            if (s_pending < lat) s_pending++;

            /* Promote */
            double min_c = c_cost[0];
            for (int i = 0; i < ac; i++) {
                p_cost[i] = c_cost[i] - min_c;
                p_path[i] = c_path[i];
                for (int k = 0; k < order; k++)
                    p_state[i][k] = c_state[i][k];
                for (int w = 0; w < tb_words; w++)
                    p_tb[i][w] = c_tb[i][w];
            }
            s_active = ac;
        }
        __syncthreads();

        if (tid == 0 && s_pending >= lat && out_idx < count) {
            out[out_idx++] = s_output_bit ? -1.0f : 1.0f;
        }
    }

    if (final_states && tid < nc) {
        for (int k = 0; k < order; k++)
            final_states[tid * 8 + k] = p_state[tid][k];
        final_costs[tid] = p_cost[tid];
    }
}

} /* extern "C" */
