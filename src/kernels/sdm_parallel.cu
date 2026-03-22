/*
 * foo_dsd_trellis — SBVD parallel-segment GPU Trellis SDM with DAS
 *
 * Proper Sliding Block Viterbi Decoder with full traceback history.
 * Each segment: M (convergence) + D (output) + overlap (non-last) + L.
 * All segments state-seeded from persistent SDM for DAS stitching.
 * Multi-channel via grid.y — all channels processed simultaneously.
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

/* Reduced from 32 to 8 to fit more blocks per SM.
 * With nc=2 (typical), shared mem drops to ~1.5KB per block.
 * This allows ~30 blocks/SM × 84 SMs = 2520 concurrent segments.
 * Supports up to nc=8 candidates. */
#define MAX_CANDS 8
#define MAX_CHILDREN (2 * MAX_CANDS)
#define MAX_HIST_BYTES 128  /* supports lat up to 1024 */

/* DAS-enabled parallel-segment SBVD.
 *
 * Grid:  (num_segments, num_channels, 1)
 * Block: (2 * num_cands, 1, 1)
 *
 * All segments are state-seeded from the same persistent SDM state
 * (uploaded to all_init_states). Non-last segments output D + overlap
 * samples for DAS overlap stitching. Last segment outputs D samples.
 *
 * seg_total_sizes[seg]: per-segment input count (M + D + [overlap] + L)
 * seg_out_caps[seg]:    per-segment output capacity (D + overlap or D)
 */
__global__ void trellis_parallel_segments(
    const float *in, float *out,
    const int *seg_starts,
    const int *seg_out_starts,
    const int *seg_total_sizes,    /* per-segment input sample count */
    const int *seg_out_caps,       /* per-segment output capacity */
    int M_convergence,
    int num_cands,
    int trellis_lat,
    int overlap,
    int num_segs,                  /* total segment count (= gridDim.x) */
    int ch_stride_in,              /* per-channel input stride */
    int ch_stride_out,             /* per-channel output stride */
    const double *all_init_states, /* [num_segs * nc * 8] replicated seed */
    const double *all_init_costs,  /* [num_segs * nc] replicated seed */
    double *all_final_states,      /* [num_segs * nc * 8] */
    double *all_final_costs,       /* [num_segs * nc] */
    int *seg_actual_counts,        /* [num_ch * num_segs] actual output */
    int D_nominal,                 /* snapshot trigger: save state at this output count */
    double *all_mid_states,        /* [num_segs * nc * 8] state at output[D] (NULL=skip) */
    double *all_mid_costs)         /* [num_segs * nc] cost at output[D] (NULL=skip) */
{
    int seg = blockIdx.x;
    int ch  = blockIdx.y;
    int tid = threadIdx.x;
    int order = c_ntf_order;
    double limit = c_state_limit;
    int nc = num_cands;
    int lat = trellis_lat;
    int hist_bytes = (lat + 7) / 8;

    const float *seg_in = in + ch * ch_stride_in + seg_starts[seg];
    float *seg_out = out + ch * ch_stride_out + seg_out_starts[seg];
    int total = seg_total_sizes[seg];
    int out_cap = seg_out_caps[seg];

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
    __shared__ unsigned c_pi[MAX_CHILDREN]; /* parent index (for post-sort traceback) */
    __shared__ unsigned p_next[MAX_CANDS]; /* parent's current traceback bit */
    __shared__ unsigned p_next_stored[MAX_CANDS]; /* traceback from PREVIOUS iteration (pipeline delay) */

    __shared__ int s_active;
    __shared__ unsigned s_output_bit;
    __shared__ int s_hist_pos;
    __shared__ int s_pending;

    /* Initialize ALL segments from persistent state (DAS state seeding).
     * Every segment starts from the same seed — the end state of the
     * previous chunk. Convergence warmup (M samples) lets each segment's
     * SDM diverge into its local signal before producing output. */
    if (tid < nc) {
        if (all_init_states) {
            for (int k = 0; k < order; k++)
                p_state[tid][k] = all_init_states[seg * nc * 8 + tid * 8 + k];
            p_cost[tid] = all_init_costs[seg * nc + tid];
        } else {
            for (int k = 0; k < order; k++)
                p_state[tid][k] = 0.0;
            p_cost[tid] = 0.0;
        }
        for (int b = 0; b < hist_bytes; b++)
            p_hist[tid][b] = 0;
        p_path[tid] = 0;
        p_next_stored[tid] = 0;
    }
    if (tid == 0) {
        s_active = nc;
        s_hist_pos = 0;
        /* For seg0 with valid persisted state, mark history as ready
         * so output starts from sample 0 (no warmup gap between chunks).
         * History is zeroed but traceback errors for the first `lat` samples
         * are negligible (~32 bits in millions). */
        s_pending = (seg == 0 && all_init_states) ? lat : 0;
    }
    __syncthreads();

    int out_idx = 0;
    for (int s = 0; s < total; s++) {
        double x = (double)seg_in[s] * 0.5;
        int ac = s_active;

        /* Phase 0: Compute parent traceback bits (before expansion). */
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
            /* y_b is the NEGATED quantizer output: y_b = -y.
             * CPU: d[0] = s[0] - g[0]*s[1] + x - y
             * GPU: d[0] = s[0] - g[0]*s[1] + x  (from ntf_calc) + y_b
             * So y_b must equal -y for the NTF state to match CPU.
             * tid&1=0 → y_b=-1.0 (CPU y=+1), tid&1=1 → y_b=+1.0 (CPU y=-1) */
            double y_b = (tid & 1) ? 1.0 : -1.0;
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
             * tid&1=0 → y_b=-1.0 → CPU y=+1 → bit=1.
             * tid&1=1 → y_b=+1.0 → CPU y=-1 → bit=0. */
            c_bit[tid] = (tid & 1) ? 0u : 1u;
            /* Path register for dedup + inherit parent's STORED traceback
             * (1-step pipeline delay matching CPU: children inherit the value
             * from BEFORE the current history read, not after). */
            c_path[tid] = (p_path[pi] << 1 | c_bit[tid]) & 0xFF;
            c_next[tid] = p_next_stored[pi];
            c_pi[tid] = pi;
            /* Copy parent history to child */
            for (int b = 0; b < hist_bytes; b++)
                c_hist[tid][b] = p_hist[pi][b];
        }
        __syncthreads();

        /* Phase 2: Thread 0 — sort, record bit, output */
        if (tid == 0) {
            int tc = 2 * ac;

            /* CPU-matched sort: majority vote + next-filter + insertion
             * sort with path dedup. Replicates sdm_sort_cands from trellis.c
             * for identical noise-shaping quality. */

            /* Step 1: Find best candidate and majority traceback vote */
            int best_idx = 0;
            int next_votes_0 = 0, next_votes_1 = 0;
            for (int i = 0; i < tc; i++) {
                if (c_cost[i] < c_cost[best_idx]) best_idx = i;
                if (c_next[i] & 1) next_votes_1++; else next_votes_0++;
            }
            unsigned majority_next = (next_votes_1 > next_votes_0) ? 1 : 0;
            unsigned min_next = c_next[best_idx];

            /* Override min_next with majority if best-with-majority is
             * within 10% cost of absolute best */
            if (min_next != majority_next) {
                int best_maj = -1;
                for (int i = 0; i < tc; i++) {
                    if (c_next[i] == majority_next &&
                        (best_maj < 0 || c_cost[i] < c_cost[best_maj]))
                        best_maj = i;
                }
                if (best_maj >= 0 && c_cost[best_maj] < c_cost[best_idx] * 1.1)
                    min_next = majority_next;
            }

            /* Step 2: Next-filter — drop candidates with wrong traceback.
             * This ensures output coherence across the trellis. */
            int filtered = 0;
            for (int i = 0; i < tc; i++) {
                if (c_next[i] == min_next) {
                    if (filtered != i) {
                        c_cost[filtered] = c_cost[i];
                        c_bit[filtered] = c_bit[i];
                        c_path[filtered] = c_path[i];
                        c_next[filtered] = c_next[i];
                        c_pi[filtered] = c_pi[i];
                        for (int k = 0; k < order; k++)
                            c_state[filtered][k] = c_state[i][k];
                        for (int b = 0; b < hist_bytes; b++)
                            c_hist[filtered][b] = c_hist[i][b];
                    }
                    filtered++;
                }
            }
            tc = filtered;

            /* Step 3: Dedup-aware greedy selection.
             * For each slot, pick the lowest-cost candidate with a unique path.
             * Uses <= tie-breaking to match CPU's sdm_cmple.
             * Simultaneously sorts and deduplicates (matches CPU sdm_sort_cands). */
            {
                int selected = 0;
                unsigned used_paths[MAX_CANDS];
                for (int slot = 0; slot < nc && slot < tc; slot++) {
                    int best = -1;
                    for (int j = 0; j < tc; j++) {
                        /* Skip duplicate paths */
                        int dup = 0;
                        for (int k = 0; k < selected; k++)
                            if (c_path[j] == used_paths[k]) { dup = 1; break; }
                        if (dup) continue;
                        if (best < 0 || c_cost[j] <= c_cost[best])
                            best = j;
                    }
                    if (best < 0) break;
                    used_paths[selected] = c_path[best];
                    if (best != selected) {
                        double t_c = c_cost[selected]; c_cost[selected] = c_cost[best]; c_cost[best] = t_c;
                        unsigned t_b = c_bit[selected]; c_bit[selected] = c_bit[best]; c_bit[best] = t_b;
                        unsigned t_p = c_path[selected]; c_path[selected] = c_path[best]; c_path[best] = t_p;
                        unsigned t_n = c_next[selected]; c_next[selected] = c_next[best]; c_next[best] = t_n;
                        unsigned t_pi = c_pi[selected]; c_pi[selected] = c_pi[best]; c_pi[best] = t_pi;
                        for (int k = 0; k < order; k++) {
                            double t_s = c_state[selected][k]; c_state[selected][k] = c_state[best][k]; c_state[best][k] = t_s;
                        }
                        for (int b = 0; b < hist_bytes; b++) {
                            unsigned char t_h = c_hist[selected][b]; c_hist[selected][b] = c_hist[best][b]; c_hist[best][b] = t_h;
                        }
                    }
                    selected++;
                }
                ac = selected;
            }

            /* Output: best candidate's traceback bit */
            s_output_bit = c_next[0];

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

            /* Move selected children to parents.
             * Pipeline delay: save parent's CURRENT traceback (p_next) as
             * p_next_stored for the next iteration. This replicates the CPU's
             * s->next = s->parent->next (line 676 in trellis.c). */
            s_active = ac;
            double min_c = c_cost[0];
            for (int i = 0; i < ac; i++) {
                p_cost[i] = c_cost[i] - min_c;
                p_path[i] = c_path[i];
                p_next_stored[i] = p_next[c_pi[i]];
                for (int k = 0; k < order; k++)
                    p_state[i][k] = c_state[i][k];
                for (int b = 0; b < hist_bytes; b++)
                    p_hist[i][b] = c_hist[i][b];
            }
        }
        __syncthreads();

        /* SBVD output with DAS overlap extension.
         * All segments: wait M samples for NTF convergence + history fill.
         * Seeded segments (all_init_states != NULL): skip latency fill
         * for seg0 (persistent state already has valid history).
         * Non-last segments output D + overlap samples for DAS scanning.
         * Last segment outputs D samples (no forward extension). */
        int eff_M = (seg == 0 && all_init_states) ? 0 : M_convergence;
        if (eff_M < lat) eff_M = lat;  /* ensure history valid */
        bool hist_ready = (s_pending >= lat);
        if (tid == 0 && hist_ready && s >= eff_M && out_idx < out_cap) {
            /* bit=1 → y=+1.0, bit=0 → y=-1.0 (matches CPU convention) */
            seg_out[out_idx++] = s_output_bit ? 1.0f : -1.0f;
        }
        /* Save state snapshot when output reaches D_nominal (overlap start).
         * Best candidate (idx 0) state used for Viterbi re-encoding. */
        if (tid == 0 && out_idx == D_nominal && all_mid_states && ch == 0) {
            for (int k = 0; k < order; k++)
                all_mid_states[seg * nc * 8 + k] = p_state[0][k];
            if (all_mid_costs)
                all_mid_costs[seg * nc] = p_cost[0];
        }
    }

    /* Save all segments' final states (channel 0 only — used for
     * persistent state update from seg0, and potential diagnostics). */
    if (ch == 0 && all_final_states && tid < nc) {
        for (int k = 0; k < order; k++)
            all_final_states[seg * nc * 8 + tid * 8 + k] = p_state[tid][k];
        all_final_costs[seg * nc + tid] = p_cost[tid];
    }

    /* Report actual output count per segment per channel */
    if (tid == 0 && seg_actual_counts)
        seg_actual_counts[ch * num_segs + seg] = out_idx;
}

} /* extern "C" */
