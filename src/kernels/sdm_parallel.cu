/*
 * foo_dsd_trellis — SBVD parallel-segment GPU Trellis SDM
 *
 * Proper Sliding Block Viterbi Decoder with full traceback history.
 * Each segment: M (convergence) + D (output) + L (lookahead).
 * Children store their bit choice explicitly (no index tricks).
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

    /* Parent state: slots 0..nc-1 */
    __shared__ double p_state[MAX_CANDS][8];
    __shared__ double p_cost[MAX_CANDS];
    __shared__ unsigned char p_hist[MAX_CANDS][MAX_HIST_BYTES];

    /* Child state: slots 0..2*nc-1 */
    __shared__ double c_state[MAX_CHILDREN][8];
    __shared__ double c_cost[MAX_CHILDREN];
    __shared__ unsigned char c_hist[MAX_CHILDREN][MAX_HIST_BYTES];
    __shared__ unsigned c_bit[MAX_CHILDREN]; /* which bit this child chose */

    __shared__ int s_active;
    __shared__ unsigned s_output_bit;
    __shared__ int s_hist_pos;
    __shared__ int s_pending;

    /* Initialize parents */
    if (tid < nc) {
        if (seg == 0 && seg0_init_states) {
            for (int k = 0; k < order; k++)
                p_state[tid][k] = seg0_init_states[tid * 8 + k];
            p_cost[tid] = seg0_init_costs[tid];
        } else {
            for (int k = 0; k < order; k++)
                p_state[tid][k] = 0.0;
            p_cost[tid] = 0.0;
        }
        for (int b = 0; b < hist_bytes; b++)
            p_hist[tid][b] = 0;
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

        /* Phase 1: Parallel candidate expansion */
        if (tid < 2 * ac) {
            int pi = tid / 2;
            double y_b = (tid & 1) ? -1.0 : 1.0;
            double d[8];
            double v = ntf_calc(p_state[pi], d, order, x);
            d[0] += y_b;
            if (limit > 0.0) {
                for (int k = 0; k < order; k++) {
                    if (d[k] > limit) d[k] = limit;
                    else if (d[k] < -limit) d[k] = -limit;
                }
            }
            for (int k = 0; k < order; k++)
                c_state[tid][k] = d[k];
            c_cost[tid] = p_cost[pi] + (v + c_ntf_a[0]*y_b)*(v + c_ntf_a[0]*y_b);
            c_bit[tid] = (tid & 1);  /* 0 = +1.0, 1 = -1.0 */
            /* Copy parent history to child */
            for (int b = 0; b < hist_bytes; b++)
                c_hist[tid][b] = p_hist[pi][b];
        }
        __syncthreads();

        /* Phase 2: Thread 0 — sort, record bit, output */
        if (tid == 0) {
            int tc = 2 * ac;

            /* Selection sort children by cost — swap all data */
            for (int i = 0; i < ac && i < tc; i++) {
                int best = i;
                for (int j = i + 1; j < tc; j++)
                    if (c_cost[j] < c_cost[best]) best = j;
                if (best != i) {
                    double t_c = c_cost[i]; c_cost[i] = c_cost[best]; c_cost[best] = t_c;
                    unsigned t_b = c_bit[i]; c_bit[i] = c_bit[best]; c_bit[best] = t_b;
                    for (int k = 0; k < order; k++) {
                        double t_s = c_state[i][k]; c_state[i][k] = c_state[best][k]; c_state[best][k] = t_s;
                    }
                    for (int b = 0; b < hist_bytes; b++) {
                        unsigned char t_h = c_hist[i][b]; c_hist[i][b] = c_hist[best][b]; c_hist[best][b] = t_h;
                    }
                }
            }

            /* Record each selected child's bit in its history */
            int byte_pos = s_hist_pos / 8;
            int bit_pos = s_hist_pos % 8;
            for (int i = 0; i < ac; i++) {
                if (c_bit[i])
                    c_hist[i][byte_pos] |= (1u << bit_pos);
                else
                    c_hist[i][byte_pos] &= ~(1u << bit_pos);
            }

            /* Output: read traceback from best candidate's history */
            if (s_pending >= lat) {
                int read_pos = (s_hist_pos + 1) % lat;
                int r_byte = read_pos / 8;
                int r_bit = read_pos % 8;
                s_output_bit = (c_hist[0][r_byte] >> r_bit) & 1;
            }

            s_hist_pos = (s_hist_pos + 1) % lat;
            if (s_pending < lat) s_pending++;

            /* Move top-ac children to parents */
            double min_c = c_cost[0];
            for (int i = 0; i < ac; i++) {
                p_cost[i] = c_cost[i] - min_c;
                for (int k = 0; k < order; k++)
                    p_state[i][k] = c_state[i][k];
                for (int b = 0; b < hist_bytes; b++)
                    p_hist[i][b] = c_hist[i][b];
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
            seg0_final_states[tid * 8 + k] = p_state[tid][k];
        seg0_final_costs[tid] = p_cost[tid];
    }
}

} /* extern "C" */
