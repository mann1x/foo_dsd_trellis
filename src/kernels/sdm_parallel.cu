/*
 * foo_dsd_trellis — SBVD parallel-segment GPU Trellis SDM
 *
 * Proper Sliding Block Viterbi Decoder with full traceback history.
 * Each segment: M (convergence) + D (output) + L (lookahead).
 * Full trellis_lat-depth history buffer for correct bit decisions.
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
/* History: packed bits, trellis_lat / 8 bytes per candidate.
 * Max trellis_lat = 2048 → 256 bytes per candidate.
 * Max 32 candidates → 8KB total. Fits in shared memory. */
#define MAX_HIST_BYTES 256  /* supports trellis_lat up to 2048 */

__global__ void trellis_parallel_segments(
    const float *in, float *out,
    const int *seg_starts,
    const int *seg_out_starts,
    int seg_total_size,       /* M + D + L */
    int M_convergence,        /* convergence depth (discard) */
    int D_output,             /* valid output count */
    int num_cands,
    int trellis_lat,          /* traceback depth */
    /* Persistent state for segment 0 */
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
    int hist_bytes = (lat + 7) / 8;  /* packed bit history bytes per candidate */

    const float *seg_in = in + seg_starts[seg];
    float *seg_out = out + seg_out_starts[seg];
    int total = seg_total_size;

    /* Shared memory */
    __shared__ double s_state[MAX_CANDS * 2][8];
    __shared__ double s_cost[MAX_CANDS * 2];
    __shared__ unsigned s_path[MAX_CANDS * 2];
    __shared__ int s_active;
    __shared__ unsigned s_output_bit;

    /* History ring buffer: packed bits for traceback.
     * hist[cand][byte] — each candidate has hist_bytes of bit history. */
    __shared__ unsigned char s_hist[MAX_CANDS * 2][MAX_HIST_BYTES];
    __shared__ int s_hist_pos;  /* current write position in ring buffer (0..lat-1) */
    __shared__ int s_pending;   /* samples processed but not yet output (0..lat) */

    /* Initialize */
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
        s_path[tid] = 0;
        /* Zero history */
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

        /* Phase 1: Parallel candidate expansion */
        if (tid < 2 * ac) {
            int pi = tid / 2;
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
            int ci = nc + tid;
            for (int k = 0; k < order; k++)
                s_state[ci][k] = d[k];
            s_cost[ci] = s_cost[pi] + (v + c_ntf_a[0]*y_b)*(v + c_ntf_a[0]*y_b);
            s_path[ci] = (s_path[pi] << 1 | (unsigned)(tid & 1)) & 0xFF;

            /* Copy parent's history to child */
            for (int b = 0; b < hist_bytes; b++)
                s_hist[ci][b] = s_hist[pi][b];
        }
        __syncthreads();

        /* Phase 2: Thread 0 — sort, select, record history, output */
        if (tid == 0) {
            int tc = 2 * ac;

            /* Selection sort: pick best nc candidates by cost */
            for (int i = 0; i < ac && i < tc; i++) {
                int best = nc + i;
                for (int j = nc + i + 1; j < nc + tc; j++)
                    if (s_cost[j] < s_cost[best]) best = j;
                if (best != nc + i) {
                    /* Swap cost, path, state, history */
                    double c2 = s_cost[nc+i]; s_cost[nc+i] = s_cost[best]; s_cost[best] = c2;
                    unsigned p2 = s_path[nc+i]; s_path[nc+i] = s_path[best]; s_path[best] = p2;
                    for (int k = 0; k < order; k++) {
                        double t2 = s_state[nc+i][k]; s_state[nc+i][k] = s_state[best][k]; s_state[best][k] = t2;
                    }
                    for (int b = 0; b < hist_bytes; b++) {
                        unsigned char h2 = s_hist[nc+i][b]; s_hist[nc+i][b] = s_hist[best][b]; s_hist[best][b] = h2;
                    }
                }
            }

            /* Record current bit decision in history ring buffer.
             * Each selected candidate records its bit at s_hist_pos. */
            int byte_pos = s_hist_pos / 8;
            int bit_pos = s_hist_pos % 8;
            for (int i = 0; i < ac; i++) {
                unsigned bit = s_path[nc + i] & 1;
                if (bit)
                    s_hist[nc + i][byte_pos] |= (1u << bit_pos);
                else
                    s_hist[nc + i][byte_pos] &= ~(1u << bit_pos);
            }

            /* Output: read the bit from trellis_lat positions ago
             * in the best candidate's history ring buffer. */
            if (s_pending >= lat) {
                int read_pos = (s_hist_pos + 1) % lat;  /* oldest entry */
                int r_byte = read_pos / 8;
                int r_bit = read_pos % 8;
                s_output_bit = (s_hist[nc][r_byte] >> r_bit) & 1;
            }

            /* Advance ring buffer position */
            s_hist_pos = (s_hist_pos + 1) % lat;
            if (s_pending < lat)
                s_pending++;

            /* Move selected children to parents */
            double min_c = s_cost[nc];
            for (int i = 0; i < ac; i++) {
                s_cost[i] = s_cost[nc+i] - min_c;
                s_path[i] = s_path[nc+i];
                for (int k = 0; k < order; k++)
                    s_state[i][k] = s_state[nc+i][k];
                for (int b = 0; b < hist_bytes; b++)
                    s_hist[i][b] = s_hist[nc+i][b];
            }
        }
        __syncthreads();

        /* SBVD output: only samples in [M, M+D) are output.
         * Must also have accumulated enough history (pending >= lat).
         * Segment 0 with persistent state: integrators are warm but
         * history buffer is cold — need lat samples to fill it. */
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
