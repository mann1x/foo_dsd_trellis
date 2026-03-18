/*
 * foo_dsd_trellis — CUDA Trellis SDM + PreCorr batch kernels
 *
 * Compiled to PTX with: nvcc -ptx -arch=sm_75 sdm_kernels.cu
 * PTX embedded as string constant in gpu_cuda.c
 *
 * Trellis: one block processes entire chunk sequentially.
 *   Thread 0..2*num_cands-1: parallel NTF evaluation per sample.
 *   Thread 0: serial sort/dedup after expansion.
 * Only viable with num_cands >= 16 (32+ children per sample).
 *
 * PreCorr: one thread per channel, sequential per sample.
 */

extern "C" {

/* NTF coefficients in constant memory */
__constant__ double c_ntf_a[8];     /* Feedback coefficients */
__constant__ double c_ntf_g[8];     /* Resonator gains */
__constant__ int    c_ntf_order;
__constant__ double c_state_limit;

/* ─── Device NTF filter calc (generic order) ───
 * Computes: d[] = NTF state advance, returns v = output before quantize */
__device__ double ntf_filter_calc(const double *s, double *d,
                                   const double *a, const double *g,
                                   int order, double x, double y) {
    /* CRFB topology: d[0] = s[0] - g[0]*s[1] + x - y
     * d[k] = s[k] + s[k-1] - g[k]*s[k+1]  for k=1..order-2
     * d[order-1] = s[order-1] + s[order-2] */
    d[0] = s[0] - g[0] * s[1] + x - y;
    for (int k = 1; k < order - 1; k++)
        d[k] = s[k] + s[k-1] - g[k] * s[k+1];
    d[order-1] = s[order-1] + s[order-2];

    double v = x;
    for (int k = 0; k < order; k++)
        v += a[k] * d[k];
    return v;
}

/* ─── Trellis chunk kernel ───
 * Processes entire chunk sequentially with parallel candidate expansion.
 * Block size = max(64, 2*num_cands) threads.
 * Shared memory holds candidate states and costs. */

/* Flat candidate state: 8 doubles + cost + path bits */
struct cand_t {
    double state[8];
    double cost;
    unsigned path;
    int    parent_idx;  /* index of parent candidate */
    unsigned next_bit;  /* traceback output bit */
};

/* Max 32 candidates → 64 children */
#define MAX_CANDS_GPU 32
#define MAX_CHILDREN  (2 * MAX_CANDS_GPU)

__global__ void trellis_chunk(
    const float *in, float *out, int count,
    int num_cands, int trellis_lat,
    /* Initial state: num_cands candidates, each with state[order] + cost + path + next_bit */
    const double *init_states,  /* [num_cands][8] */
    const double *init_costs,   /* [num_cands] */
    /* Output final state */
    double *final_states,       /* [num_cands][8] */
    double *final_costs,        /* [num_cands] */
    /* Path + next_bit persistence (packed as int: low 8 bits = path, bit 8 = next_bit) */
    int *init_paths,            /* [num_cands] — NULL on first invocation */
    int *final_paths)           /* [num_cands] */
{
    int tid = threadIdx.x;
    int order = c_ntf_order;
    double limit = c_state_limit;

    /* Shared memory for candidates */
    __shared__ struct cand_t parents[MAX_CANDS_GPU];
    __shared__ struct cand_t children[MAX_CHILDREN];
    __shared__ double child_costs[MAX_CHILDREN];
    __shared__ int sorted_idx[MAX_CHILDREN];
    __shared__ int active_cands;
    __shared__ unsigned output_bit;

    /* Initialize parents from input state */
    if (tid < num_cands) {
        for (int k = 0; k < order; k++)
            parents[tid].state[k] = init_states[tid * 8 + k];
        parents[tid].cost = init_costs[tid];
        if (init_paths) {
            parents[tid].path = (unsigned)(init_paths[tid] & 0xFF);
            parents[tid].next_bit = (unsigned)((init_paths[tid] >> 8) & 1);
        } else {
            parents[tid].path = 0;
            parents[tid].next_bit = 0;
        }
    }
    if (tid == 0)
        active_cands = num_cands;
    __syncthreads();

    /* Sequential loop over samples */
    for (int s = 0; s < count; s++) {
        double x = (double)in[s];
        int nc = active_cands;

        /* Phase 1: Parallel candidate expansion.
         * tid < 2*nc: each thread evaluates one child.
         * tid/2 = parent index, tid%2 = y=+1 (0) or y=-1 (1) */
        if (tid < 2 * nc) {
            int pi = tid / 2;
            double y_branch = (tid & 1) ? -1.0 : 1.0;

            /* NTF filter calc */
            double d[8];
            double v = ntf_filter_calc(parents[pi].state, d,
                                        c_ntf_a, c_ntf_g, order,
                                        x, 0.0);

            /* Apply y offset to state[0] */
            d[0] += y_branch;

            /* State limiter */
            if (limit > 0.0) {
                for (int k = 0; k < order; k++) {
                    if (d[k] > limit) d[k] = limit;
                    else if (d[k] < -limit) d[k] = -limit;
                }
            }

            /* Store child */
            for (int k = 0; k < order; k++)
                children[tid].state[k] = d[k];

            double y_val = (tid & 1) ? -1.0 : 1.0;
            children[tid].cost = parents[pi].cost +
                (v + c_ntf_a[0] * y_val) * (v + c_ntf_a[0] * y_val);
            children[tid].path = (parents[pi].path << 1 |
                                  (unsigned)(tid & 1)) & ((1u << 8) - 1);
            children[tid].parent_idx = pi;
            children[tid].next_bit = parents[pi].next_bit;
            child_costs[tid] = children[tid].cost;
        }
        __syncthreads();

        /* Phase 2: Sort and select top num_cands (thread 0 only).
         * Simple insertion sort — num_cands ≤ 32 so 64 children max. */
        if (tid == 0) {
            int total = 2 * nc;

            /* Initialize index array */
            for (int i = 0; i < total; i++)
                sorted_idx[i] = i;

            /* Selection sort: find top num_cands by cost */
            for (int i = 0; i < nc && i < total; i++) {
                int best = i;
                for (int j = i + 1; j < total; j++) {
                    if (child_costs[sorted_idx[j]] < child_costs[sorted_idx[best]])
                        best = j;
                }
                if (best != i) {
                    int tmp = sorted_idx[i];
                    sorted_idx[i] = sorted_idx[best];
                    sorted_idx[best] = tmp;
                }
            }

            /* Output: traceback from best candidate */
            output_bit = children[sorted_idx[0]].next_bit;

            /* Move selected children to parents for next iteration */
            double min_cost = children[sorted_idx[0]].cost;
            for (int i = 0; i < nc; i++) {
                int ci = sorted_idx[i];
                for (int k = 0; k < order; k++)
                    parents[i].state[k] = children[ci].state[k];
                parents[i].cost = children[ci].cost - min_cost;
                parents[i].path = children[ci].path;
                /* Advance next_bit: after trellis_lat samples, start outputting */
                parents[i].next_bit = children[ci].path & 1;
            }

            active_cands = nc;
        }
        __syncthreads();

        /* Write output sample */
        if (tid == 0 && s >= trellis_lat)
            out[s - trellis_lat] = output_bit ? 1.0f : -1.0f;
    }

    /* Save final state including path + next_bit */
    if (tid < num_cands) {
        for (int k = 0; k < order; k++)
            final_states[tid * 8 + k] = parents[tid].state[k];
        final_costs[tid] = parents[tid].cost;
        if (final_paths)
            final_paths[tid] = (int)(parents[tid].path & 0xFF) |
                               (int)((parents[tid].next_bit & 1) << 8);
    }
}

/* ─── PreCorr batch kernel ───
 * One thread per channel, sequential per sample.
 * Prediction table in shared memory (2KB). */

__constant__ float c_precorr_a[8];
__constant__ float c_precorr_g[8];
__constant__ int   c_precorr_order;
__constant__ float c_precorr_limit;

/* Per-channel initial conditions (uploaded alongside state) */
struct precorr_init_t {
    float state[8];
    float prev_y;
    int   history;
    int   phase;
    float pad;  /* alignment */
};

__global__ void precorr_chunk(
    const float *in, float *out, int count,
    const float *pred_table,           /* [256][8] = 2048 floats */
    const struct precorr_init_t *init, /* [num_channels] */
    struct precorr_init_t *final_out,  /* [num_channels] */
    int num_channels)
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= num_channels) return;

    int order = c_precorr_order;
    float limit = c_precorr_limit;
    float state[8];
    float d[8];

    /* Load initial conditions from CPU-trained context */
    for (int k = 0; k < order; k++)
        state[k] = init[ch].state[k];

    unsigned history = (unsigned)init[ch].history;
    int phase = init[ch].phase;
    float prev_y = init[ch].prev_y;

    const float *ch_in  = in  + ch * count;
    float       *ch_out = out + ch * count;

    for (int s = 0; s < count; s++) {
        float x = ch_in[s];

        /* NTF filter calc (float) */
        d[0] = state[0] - c_precorr_g[0] * state[1] + x - prev_y;
        for (int k = 1; k < order - 1; k++)
            d[k] = state[k] + state[k-1] - c_precorr_g[k] * state[k+1];
        d[order-1] = state[order-1] + state[order-2];

        float v = x;
        for (int k = 0; k < order; k++)
            v += c_precorr_a[k] * d[k];

        /* Greedy quantize */
        float y = (v >= 0.0f) ? 1.0f : -1.0f;

        /* State limiter */
        if (limit > 0.0f) {
            for (int k = 0; k < order; k++) {
                if (d[k] > limit) d[k] = limit;
                else if (d[k] < -limit) d[k] = -limit;
            }
        }
        for (int k = 0; k < order; k++)
            state[k] = d[k];

        /* Prediction correction */
        float corr = pred_table[history * 8 + phase];
        float y_corr = y - corr;
        float out_y = (y_corr >= 0.0f) ? 1.0f : -1.0f;

        prev_y = out_y;
        history = ((history << 1) | (out_y > 0.0f ? 1u : 0u)) & 0xFF;
        phase = (phase + 1) & 7;

        ch_out[s] = out_y;
    }

    /* Save final state + conditions for next chunk */
    for (int k = 0; k < order; k++)
        final_out[ch].state[k] = state[k];
    final_out[ch].prev_y = prev_y;
    final_out[ch].history = (int)history;
    final_out[ch].phase = phase;
}

} /* extern "C" */
