/*
 * foo_dsd_trellis — Hawksford intra-step parallel GPU Trellis SDM
 *
 * Single-segment kernel with nc=64 candidates, fully parallel within
 * each time step. No temporal segmentation = no stitching artifacts.
 * Uses double for NTF state (float32 insufficient for noise shaping).
 *
 * Architecture:
 *   128 threads (2*nc) on 1 SM per channel
 *   Phase 1: Parallel NTF expansion (128 threads)
 *   Phase 2: Parallel bitonic sort by cost (128 threads)
 *   Phase 3: Serial dedup + promote (thread 0)
 *   Output:  Best candidate's current bit
 */

extern "C" {

__constant__ double c_ntf_a[8];
__constant__ double c_ntf_g[8];
__constant__ int    c_ntf_order;

#define NC 64
#define NC2 128  /* 2 * NC */

/* NTF filter calc — double precision */
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
    int trellis_lat,
    const double *init_states,   /* [NC*8] or NULL */
    const double *init_costs,    /* [NC] or NULL */
    double *final_states,        /* [NC*8] */
    double *final_costs)         /* [NC] */
{
    int tid = threadIdx.x;
    int order = c_ntf_order;

    /* Shared memory — double precision for NTF */
    __shared__ double p_state[NC][8];
    __shared__ double p_cost[NC];
    __shared__ unsigned p_path[NC];

    __shared__ double c_state[NC2][8];
    __shared__ double c_cost[NC2];
    __shared__ unsigned c_path[NC2];
    __shared__ unsigned c_bit[NC2];

    /* Control */
    __shared__ int s_active;
    __shared__ unsigned s_output_bit;

    /* Init parents */
    if (tid < NC) {
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
    }
    if (tid == 0) s_active = NC;
    __syncthreads();

    int out_idx = 0;
    int lat = trellis_lat;

    for (int s = 0; s < count; s++) {
        double x = (double)in[s] * 0.5;
        int ac = s_active;

        /* ══════ Phase 1: Parallel NTF expansion ══════ */
        if (tid < 2 * ac) {
            int pi = tid / 2;
            double y_b = (tid & 1) ? 1.0 : -1.0;
            double d[8];
            double v = ntf_calc_d(p_state[pi], d, order, x);
            d[0] += y_b;
            for (int k = 0; k < order; k++)
                c_state[tid][k] = d[k];
            c_cost[tid] = p_cost[pi] + (v + c_ntf_a[0] * y_b) * (v + c_ntf_a[0] * y_b);
            c_bit[tid] = (tid & 1) ? 0u : 1u;
            c_path[tid] = (p_path[pi] << 1 | c_bit[tid]) & 0xFF;
        }
        __syncthreads();

        /* ══════ Phase 2+3: Thread 0 — selection sort + dedup + output ══════
         * Same algorithm as proven sdm_parallel.cu kernel. */
        if (tid == 0) {
            int tc = 2 * ac;
            /* Selection sort children by cost */
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
                    }
                    deduped++;
                }
            }
            ac = deduped;
            s_output_bit = c_bit[0];
            /* Promote to parents */
            double min_c = c_cost[0];
            for (int i = 0; i < ac; i++) {
                p_cost[i] = c_cost[i] - min_c;
                p_path[i] = c_path[i];
                for (int k = 0; k < order; k++)
                    p_state[i][k] = c_state[i][k];
            }
            s_active = ac;
        }
        __syncthreads();

        if (tid == 0 && s >= lat && out_idx < count) {
            out[out_idx++] = s_output_bit ? -1.0f : 1.0f;
        }
    }

    if (final_states && tid < NC) {
        for (int k = 0; k < order; k++)
            final_states[tid * 8 + k] = p_state[tid][k];
        final_costs[tid] = p_cost[tid];
    }
}

} /* extern "C" */
