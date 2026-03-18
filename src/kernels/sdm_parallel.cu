/*
 * foo_dsd_trellis — SBVD parallel-segment GPU Trellis SDM
 *
 * Proper Sliding Block Viterbi Decoder with full traceback history.
 * Each segment: M (convergence) + D (output) + L (lookahead).
 * Optimized: sort uses index array (no history data movement).
 */

extern "C" {

__constant__ double c_ntf_a[8];
__constant__ double c_ntf_g[8];
__constant__ int    c_ntf_order;
__constant__ double c_state_limit;

__device__ double ntf_calc(const double *s, double *d,
                            int order, double x) {
    d[0] = s[0] - c_ntf_g[0] * s[1] + x;
    for (int k = 1; k < order - 1; k++)
        d[k] = s[k] + s[k-1] - c_ntf_g[k] * s[k+1];
    d[order-1] = s[order-1] + s[order-2];
    double v = x;
    for (int k = 0; k < order; k++)
        v += c_ntf_a[k] * d[k];
    return v;
}

#define MAX_CANDS 32
#define MAX_CHILDREN (2 * MAX_CANDS)
#define MAX_HIST_BYTES 256

__global__ void trellis_parallel_segments(
    const float *in, float *out,
    const int *seg_starts,
    const int *seg_out_starts,
    int seg_total_size,
    int M_convergence,
    int D_output,
    int num_cands,
    int trellis_lat,
    const double *seg0_init_states,
    const double *seg0_init_costs,
    double *seg0_final_states,
    double *seg0_final_costs)
{
    int seg = blockIdx.x;
    int tid = threadIdx.x;
    int order = c_ntf_order;
    double limit = c_state_limit;
    int nc = num_cands;
    int lat = trellis_lat;
    int hist_bytes = (lat + 7) / 8;

    const float *seg_in = in + seg_starts[seg];
    float *seg_out = out + seg_out_starts[seg];
    int total = seg_total_size;

    /* All candidate data in flat arrays. Slots 0..nc-1 = parents,
     * nc..2*nc-1 = children. Use index array for sort (no data movement). */
    __shared__ double s_state[MAX_CHILDREN][8];
    __shared__ double s_cost[MAX_CHILDREN];
    __shared__ unsigned char s_hist[MAX_CHILDREN][MAX_HIST_BYTES];
    __shared__ int s_sorted[MAX_CANDS]; /* indices into children (nc..2*nc-1) */
    __shared__ int s_active;
    __shared__ unsigned s_output_bit;
    __shared__ int s_hist_pos;
    __shared__ int s_pending;

    /* Initialize parents */
    if (tid < nc) {
        if (seg == 0 && seg0_init_states) {
            for (int k = 0; k < order; k++)
                s_state[tid][k] = seg0_init_states[tid * 8 + k];
            s_cost[tid] = seg0_init_costs[tid];
        } else {
            for (int k = 0; k < order; k++)
                s_state[tid][k] = 0.0;
            s_cost[tid] = 0.0;
        }
        for (int b = 0; b < hist_bytes; b++)
            s_hist[tid][b] = 0;
    }
    if (tid == 0) {
        s_active = nc;
        s_hist_pos = 0;
        s_pending = 0;
    }
    __syncthreads();

    int out_idx = 0;
    for (int s = 0; s < total; s++) {
        double x = (double)seg_in[s];
        int ac = s_active;

        /* Phase 1: Parallel candidate expansion (threads 0..2*ac-1) */
        if (tid < 2 * ac) {
            int pi = tid / 2;  /* parent index */
            double y_b = (tid & 1) ? -1.0 : 1.0;
            double d[8];
            double v = ntf_calc(s_state[pi], d, order, x);
            d[0] += y_b;
            if (limit > 0.0) {
                for (int k = 0; k < order; k++) {
                    if (d[k] > limit) d[k] = limit;
                    else if (d[k] < -limit) d[k] = -limit;
                }
            }
            int ci = nc + tid;  /* child slot */
            for (int k = 0; k < order; k++)
                s_state[ci][k] = d[k];
            s_cost[ci] = s_cost[pi] + (v + c_ntf_a[0]*y_b)*(v + c_ntf_a[0]*y_b);
            /* Copy parent history to child */
            for (int b = 0; b < hist_bytes; b++)
                s_hist[ci][b] = s_hist[pi][b];
        }
        __syncthreads();

        /* Phase 2: Thread 0 — index-based sort, record history, output */
        if (tid == 0) {
            int tc = 2 * ac;

            /* Init sorted indices pointing to children */
            for (int i = 0; i < tc; i++)
                s_sorted[i] = nc + i;

            /* Selection sort by cost — swap indices only, no data movement */
            for (int i = 0; i < ac && i < tc; i++) {
                int best_idx = i;
                for (int j = i + 1; j < tc; j++)
                    if (s_cost[s_sorted[j]] < s_cost[s_sorted[best_idx]])
                        best_idx = j;
                if (best_idx != i) {
                    int tmp = s_sorted[i];
                    s_sorted[i] = s_sorted[best_idx];
                    s_sorted[best_idx] = tmp;
                }
            }

            /* Record bit in history for top-ac candidates */
            int byte_pos = s_hist_pos / 8;
            int bit_pos = s_hist_pos % 8;
            for (int i = 0; i < ac; i++) {
                int ci = s_sorted[i];
                unsigned bit = (s_cost[ci] == s_cost[ci]) ? /* NaN check */
                    ((int)(s_state[ci][0] + 1000.0) & 1) : 0; /* fallback */
                /* Actually, the bit decision is y=+1 or y=-1 from expansion.
                 * Child index ci = nc + tid, where tid&1 = bit choice.
                 * ci - nc = original tid. (ci - nc) & 1 = bit. */
                bit = ((unsigned)(ci - nc)) & 1;
                if (bit)
                    s_hist[ci][byte_pos] |= (1u << bit_pos);
                else
                    s_hist[ci][byte_pos] &= ~(1u << bit_pos);
            }

            /* Output: read traceback bit from trellis_lat ago */
            if (s_pending >= lat) {
                int read_pos = (s_hist_pos + 1) % lat;
                int r_byte = read_pos / 8;
                int r_bit = read_pos % 8;
                int best = s_sorted[0];
                s_output_bit = (s_hist[best][r_byte] >> r_bit) & 1;
            }

            s_hist_pos = (s_hist_pos + 1) % lat;
            if (s_pending < lat) s_pending++;

            /* Move top-ac children to parent slots (compact) */
            double min_c = s_cost[s_sorted[0]];
            for (int i = 0; i < ac; i++) {
                int ci = s_sorted[i];
                s_cost[i] = s_cost[ci] - min_c;
                for (int k = 0; k < order; k++)
                    s_state[i][k] = s_state[ci][k];
                for (int b = 0; b < hist_bytes; b++)
                    s_hist[i][b] = s_hist[ci][b];
            }
        }
        __syncthreads();

        /* SBVD output */
        int eff_M = (seg == 0 && seg0_init_states) ? trellis_lat : M_convergence;
        if (tid == 0 && s_pending >= lat && s >= eff_M && out_idx < D_output) {
            seg_out[out_idx++] = s_output_bit ? 1.0f : -1.0f;
        }
    }

    /* Segment 0: save final state */
    if (seg == 0 && seg0_final_states && tid < nc) {
        for (int k = 0; k < order; k++)
            seg0_final_states[tid * 8 + k] = s_state[tid][k];
        seg0_final_costs[tid] = s_cost[tid];
    }
}

} /* extern "C" */
