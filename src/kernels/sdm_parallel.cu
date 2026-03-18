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
    __shared__ unsigned p_path[MAX_CANDS]; /* path bits for dedup */

    /* Child state: slots 0..2*nc-1 */
    __shared__ double c_state[MAX_CHILDREN][8];
    __shared__ double c_cost[MAX_CHILDREN];
    __shared__ unsigned char c_hist[MAX_CHILDREN][MAX_HIST_BYTES];
    __shared__ unsigned c_bit[MAX_CHILDREN]; /* which bit this child chose */
    __shared__ unsigned c_path[MAX_CHILDREN]; /* path bits for dedup */
    __shared__ unsigned c_next[MAX_CHILDREN]; /* traceback output bit (inherited from parent) */
    __shared__ unsigned p_next[MAX_CANDS]; /* parent's traceback bit */

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
        p_path[tid] = 0;
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

        /* Phase 0: Compute parent traceback bits (before expansion).
         * Each parent reads its history at next_pos to get the output
         * bit from trellis_lat ago. Children inherit this. */
        if (tid < ac) {
            if (s_pending >= lat) {
                int next_pos = (s_hist_pos + 1) % lat;
                int nb = next_pos / 8;
                int ni = next_pos % 8;
                p_next[tid] = (p_hist[tid][nb] >> ni) & 1;
            } else {
                p_next[tid] = 0;
            }
        }
        __syncthreads();

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
            /* Bit convention: 1 = y=+1.0, 0 = y=-1.0 (matches CPU).
             * tid&1=0 → y_b=+1.0 → bit=1. tid&1=1 → y_b=-1.0 → bit=0. */
            c_bit[tid] = (tid & 1) ? 0u : 1u;
            /* Path register for dedup + inherit parent's traceback bit */
            c_path[tid] = (p_path[pi] << 1 | c_bit[tid]) & 0xFF;
            c_next[tid] = p_next[pi];
            /* Copy parent history to child */
            for (int b = 0; b < hist_bytes; b++)
                c_hist[tid][b] = p_hist[pi][b];
        }
        __syncthreads();

        /* Phase 2: Thread 0 — sort, record bit, output */
        if (tid == 0) {
            int tc = 2 * ac;

            /* Selection sort children by cost with path deduplication.
             * After sorting by cost, skip children with duplicate path
             * bits — they represent identical future trajectories and
             * waste candidate slots. Matches CPU hash-based dedup. */
            for (int i = 0; i < ac && i < tc; i++) {
                int best = i;
                for (int j = i + 1; j < tc; j++)
                    if (c_cost[j] < c_cost[best]) best = j;
                if (best != i) {
                    double t_c = c_cost[i]; c_cost[i] = c_cost[best]; c_cost[best] = t_c;
                    unsigned t_b = c_bit[i]; c_bit[i] = c_bit[best]; c_bit[best] = t_b;
                    unsigned t_p = c_path[i]; c_path[i] = c_path[best]; c_path[best] = t_p;
                    unsigned t_n = c_next[i]; c_next[i] = c_next[best]; c_next[best] = t_n;
                    for (int k = 0; k < order; k++) {
                        double t_s = c_state[i][k]; c_state[i][k] = c_state[best][k]; c_state[best][k] = t_s;
                    }
                    for (int b = 0; b < hist_bytes; b++) {
                        unsigned char t_h = c_hist[i][b]; c_hist[i][b] = c_hist[best][b]; c_hist[best][b] = t_h;
                    }
                }
            }
            /* Path dedup: collapse children with identical paths.
             * Keep lowest-cost (already sorted). Remove duplicates by
             * shifting remaining entries down. */
            {
                int deduped = 1; /* first is always unique */
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
                            for (int b = 0; b < hist_bytes; b++)
                                c_hist[deduped][b] = c_hist[i][b];
                        }
                        deduped++;
                    }
                }
                ac = deduped;  /* may be less than nc */
            }

            /* Majority vote on traceback output bit (matches CPU).
             * Count how many candidates agree on next=0 vs next=1.
             * Filter out candidates that disagree with majority. */
            if (s_pending >= lat) {
                unsigned votes[2] = {0, 0};
                for (int i = 0; i < ac; i++)
                    votes[c_next[i] & 1]++;
                unsigned majority = (votes[1] > votes[0]) ? 1 : 0;

                /* If best candidate disagrees with majority and majority's
                 * best is within 10% cost, use majority (matches CPU). */
                if (c_next[0] != majority) {
                    for (int i = 1; i < ac; i++) {
                        if (c_next[i] == majority && c_cost[i] < c_cost[0] * 1.1) {
                            /* Swap this candidate to position 0 */
                            double t_c = c_cost[0]; c_cost[0] = c_cost[i]; c_cost[i] = t_c;
                            unsigned t_b = c_bit[0]; c_bit[0] = c_bit[i]; c_bit[i] = t_b;
                            unsigned t_p = c_path[0]; c_path[0] = c_path[i]; c_path[i] = t_p;
                            unsigned t_n = c_next[0]; c_next[0] = c_next[i]; c_next[i] = t_n;
                            for (int k = 0; k < order; k++) {
                                double t_s = c_state[0][k]; c_state[0][k] = c_state[i][k]; c_state[i][k] = t_s;
                            }
                            for (int b = 0; b < hist_bytes; b++) {
                                unsigned char t_h = c_hist[0][b]; c_hist[0][b] = c_hist[i][b]; c_hist[i][b] = t_h;
                            }
                            break;
                        }
                    }
                }

                /* Filter: keep only candidates matching best's next */
                unsigned best_next = c_next[0];
                int filtered = 0;
                for (int i = 0; i < ac; i++) {
                    if (c_next[i] == best_next) {
                        if (filtered != i) {
                            c_cost[filtered] = c_cost[i];
                            c_bit[filtered] = c_bit[i];
                            c_path[filtered] = c_path[i];
                            c_next[filtered] = c_next[i];
                            for (int k = 0; k < order; k++)
                                c_state[filtered][k] = c_state[i][k];
                            for (int b = 0; b < hist_bytes; b++)
                                c_hist[filtered][b] = c_hist[i][b];
                        }
                        filtered++;
                    }
                }
                ac = filtered;

                s_output_bit = best_next;
            } else {
                s_output_bit = c_bit[0];
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

            s_hist_pos = (s_hist_pos + 1) % lat;
            if (s_pending < lat) s_pending++;

            /* Move deduped children to parents */
            s_active = ac;
            double min_c = c_cost[0];
            for (int i = 0; i < ac; i++) {
                p_cost[i] = c_cost[i] - min_c;
                p_path[i] = c_path[i];
                for (int k = 0; k < order; k++)
                    p_state[i][k] = c_state[i][k];
                for (int b = 0; b < hist_bytes; b++)
                    p_hist[i][b] = c_hist[i][b];
            }
        }
        __syncthreads();

        /* SBVD output.
         * Segment 0 with persistent state: output from sample 0 using
         * immediate best-candidate bit (no traceback wait). Avoids gap.
         * Segments 1+: wait for both M convergence and lat history fill. */
        int eff_M = (seg == 0 && seg0_init_states) ? 0 : M_convergence;
        bool hist_ready = (s_pending >= lat) || (seg == 0 && seg0_init_states);
        if (tid == 0 && hist_ready && s >= eff_M && out_idx < D_output) {
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
