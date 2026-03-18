/*
 * foo_dsd_trellis — Parallel-segment GPU Trellis SDM
 *
 * Each thread block processes one segment independently.
 * Segments have overlap warmup for SDM convergence at boundaries.
 * Hundreds of segments run simultaneously — massive GPU parallelism.
 *
 * Grid: (num_segments, 1, 1)
 * Block: (2 * num_cands, 1, 1) — parallel candidate expansion
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

/* Per-segment trellis SDM.
 * Each block = one segment. tid = candidate thread.
 *
 * in:          full FIR output (all segments contiguous)
 * out:         output buffer (only actual segment data, no warmup)
 * seg_starts:  [num_segments] — start index of each segment in `in`
 * seg_sizes:   [num_segments] — total size including warmup
 * warmup:      number of warmup samples to discard per segment
 * num_cands:   candidates for trellis
 */
__global__ void trellis_parallel_segments(
    const float *in, float *out,
    const int *seg_starts,    /* [num_segments] start offset in input */
    const int *seg_out_starts,/* [num_segments] start offset in output */
    int seg_total_size,       /* samples per segment including warmup */
    int warmup,               /* warmup samples to discard */
    int num_cands,
    /* Persistent state for segment 0 (continuity across chunks) */
    const double *seg0_init_states,  /* [num_cands * 8] or NULL */
    const double *seg0_init_costs,   /* [num_cands] or NULL */
    double *seg0_final_states,       /* [num_cands * 8] */
    double *seg0_final_costs)        /* [num_cands] */
{
    int seg = blockIdx.x;
    int tid = threadIdx.x;
    int order = c_ntf_order;
    double limit = c_state_limit;
    int nc = num_cands;

    /* Each segment's input starts at seg_starts[seg] */
    const float *seg_in = in + seg_starts[seg];
    float *seg_out = out + seg_out_starts[seg];
    int total = seg_total_size;

    /* Shared memory for candidates */
    __shared__ double s_state[MAX_CANDS * 2][8];
    __shared__ double s_cost[MAX_CANDS * 2];
    __shared__ unsigned s_path[MAX_CANDS * 2];
    __shared__ unsigned s_next[MAX_CANDS * 2];
    __shared__ int s_active;
    __shared__ unsigned s_output;

    /* Segment 0: load persistent state for continuity across chunks.
     * Segments 1+: start from zero with warmup overlap for convergence. */
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
        s_next[tid] = 0;
    }
    if (tid == 0) s_active = nc;
    __syncthreads();

    /* Process all samples (warmup + actual) */
    int out_idx = 0;
    for (int s = 0; s < total; s++) {
        double x = (double)seg_in[s];
        int ac = s_active;

        /* Parallel NTF evaluation */
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
            s_next[ci] = s_next[pi];
        }
        __syncthreads();

        /* Thread 0: sort + select */
        if (tid == 0) {
            int total_children = 2 * ac;
            for (int i = 0; i < ac && i < total_children; i++) {
                int best = nc + i;
                for (int j = nc + i + 1; j < nc + total_children; j++)
                    if (s_cost[j] < s_cost[best]) best = j;
                if (best != nc + i) {
                    double tc = s_cost[nc+i]; s_cost[nc+i] = s_cost[best]; s_cost[best] = tc;
                    unsigned tp = s_path[nc+i]; s_path[nc+i] = s_path[best]; s_path[best] = tp;
                    unsigned tn = s_next[nc+i]; s_next[nc+i] = s_next[best]; s_next[best] = tn;
                    for (int k = 0; k < order; k++) {
                        double ts = s_state[nc+i][k]; s_state[nc+i][k] = s_state[best][k]; s_state[best][k] = ts;
                    }
                }
            }
            s_output = s_path[nc] & 1;
            double min_c = s_cost[nc];
            for (int i = 0; i < ac; i++) {
                s_cost[i] = s_cost[nc+i] - min_c;
                s_path[i] = s_path[nc+i];
                s_next[i] = s_path[nc+i] & 1;
                for (int k = 0; k < order; k++)
                    s_state[i][k] = s_state[nc+i][k];
            }
        }
        __syncthreads();

        /* Write output only after warmup.
         * Segment 0 has no warmup (persistent state = already converged). */
        int eff_warmup = (seg == 0 && seg0_init_states) ? 0 : warmup;
        if (tid == 0 && s >= eff_warmup) {
            seg_out[out_idx++] = s_output ? 1.0f : -1.0f;
        }
    }

    /* Segment 0: save final state for next chunk */
    if (seg == 0 && seg0_final_states && tid < nc) {
        for (int k = 0; k < order; k++)
            seg0_final_states[tid * 8 + k] = s_state[tid][k];
        seg0_final_costs[tid] = s_cost[tid];
    }
}

} /* extern "C" */
