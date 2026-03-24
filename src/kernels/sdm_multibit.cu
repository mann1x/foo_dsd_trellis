/*
 * foo_dsd_trellis — Multi-bit greedy SDM kernel (GPU experiment)
 *
 * 4-bit SDM (16 levels): quantization noise is ~24 dB lower than 1-bit.
 * Greedy quantizer (no trellis search) because multi-bit error is small
 * enough that the greedy decision is near-optimal.
 *
 * Two-stage pipeline:
 *   Stage 1: fp64 input → multi-bit SDM → float output (this kernel)
 *   Stage 2: multi-bit → 1-bit SDM (separate, CPU or GPU)
 *
 * fp32 should be sufficient for the NTF filter because the quantizer
 * has 16 levels — a ULP rounding difference doesn't flip the decision
 * (unlike 1-bit where ULP differences cascade into completely different
 * bit streams).
 */

extern "C" {

__constant__ float c_mb_ntf_a[8];
__constant__ float c_mb_ntf_g[8];
__constant__ int   c_mb_order;
__constant__ float c_mb_state_limit;
__constant__ int   c_mb_num_levels;  /* 16 for 4-bit */

/* Multi-bit greedy SDM: one thread per segment, sequential per sample.
 *
 * Grid:  (num_segments, num_channels, 1)
 * Block: (1, 1, 1)
 *
 * Each segment processes M warmup + D output samples.
 * Output: float values in [-1, +1] quantized to num_levels steps.
 */
__global__ void sdm_multibit_segments(
    const double *in,         /* fp64 input (post-FIR) */
    float *out,               /* multi-bit output */
    const int *seg_starts,    /* per-segment input offset */
    const int *seg_out_starts,/* per-segment output offset */
    int seg_total,            /* M + D per segment */
    int M_warmup,             /* warmup samples (discarded) */
    int D_output,             /* output samples per segment */
    int num_segs,
    int ch_stride_in,
    int ch_stride_out,
    const float *init_states, /* [num_segs * order] replicated seed */
    float *final_states)      /* [num_segs * order] final states (NULL=skip) */
{
    int seg = blockIdx.x;
    int ch  = blockIdx.y;
    if (seg >= num_segs) return;

    int order = c_mb_order;
    float limit = c_mb_state_limit;
    int nlevels = c_mb_num_levels;
    float step = 2.0f / (float)(nlevels - 1);  /* quantization step size */

    const double *seg_in = in + ch * ch_stride_in + seg_starts[seg];
    float *seg_out = out + ch * ch_stride_out + seg_out_starts[seg];

    /* Initialize integrator state from seed */
    float state[8];
    if (init_states) {
        for (int k = 0; k < order; k++)
            state[k] = init_states[seg * order + k];
    } else {
        for (int k = 0; k < order; k++)
            state[k] = 0.0f;
    }

    int total = M_warmup + D_output;
    int out_count = 0;

    for (int n = 0; n < total; n++) {
        float x = (float)(seg_in[n] * 0.5);  /* same 0.5× scale as CPU SDM */

        /* NTF filter calc (same structure as CPU, fp32) */
        float d[8];
        d[0] = state[0] - c_mb_ntf_g[0] * state[1] + x;
        for (int k = 1; k < order - 1; k++)
            d[k] = state[k] + state[k-1] - c_mb_ntf_g[k] * state[k+1];
        d[order-1] = state[order-1] + state[order-2];

        float v = x;
        for (int k = 0; k < order; k++)
            v += c_mb_ntf_a[k] * d[k];

        /* Multi-bit quantizer: minimize (v - y * a[0])².
         * Ideal output: y = v / a[0]. Round to nearest level. */
        float y_ideal = v / c_mb_ntf_a[0];
        int level = (int)roundf(y_ideal / step);
        if (level > nlevels / 2 - 1) level = nlevels / 2 - 1;
        if (level < -(nlevels / 2 - 1)) level = -(nlevels / 2 - 1);
        float y = (float)level * step;

        /* Update integrator state: d[0] was computed with y=0,
         * actual state = d[0] - y (NTF formula: d[0] = s[0] - g[0]*s[1] + x - y) */
        state[0] = d[0] - y;
        for (int k = 1; k < order; k++)
            state[k] = d[k];

        /* State limiter */
        if (limit > 0.0f) {
            for (int k = 0; k < order; k++) {
                if (state[k] > limit) state[k] = limit;
                else if (state[k] < -limit) state[k] = -limit;
            }
        }

        /* Output after warmup */
        if (n >= M_warmup && out_count < D_output) {
            seg_out[out_count++] = y;
        }
    }

    /* Save final state for potential chaining */
    if (final_states) {
        for (int k = 0; k < order; k++)
            final_states[seg * order + k] = state[k];
    }
}

/* Stage 2: Convert multi-bit stream to 1-bit.
 * Simple 2nd-order greedy SDM — the input is already noise-shaped,
 * so a low-order modulator with no trellis is sufficient.
 *
 * Grid:  (ceil(count/256), num_channels, 1)
 * Block: (256, 1, 1)  — BUT sequential within each channel.
 *
 * Actually, this must be sequential per channel (feedback dependency).
 * Grid:  (1, num_channels, 1)
 * Block: (1, 1, 1)
 */
__global__ void sdm_multibit_to_1bit(
    const float *in,    /* multi-bit input [-1, +1] */
    float *out,         /* 1-bit output ±1.0 */
    int count,
    int ch_stride,
    float gain)         /* volume gain to apply */
{
    int ch = blockIdx.y;
    const float *ch_in = in + ch * ch_stride;
    float *ch_out = out + ch * ch_stride;

    /* 3rd-order error-feedback SDM (no trellis, no NTF — just error diffusion).
     * The input is already well-shaped, so simple error feedback suffices. */
    float e1 = 0.0f, e2 = 0.0f, e3 = 0.0f;

    for (int i = 0; i < count; i++) {
        float x = ch_in[i] * gain;

        /* Error-feedback: subtract accumulated error from input */
        float corrected = x - (1.5f * e1 - 0.75f * e2 + 0.25f * e3);

        /* 1-bit quantizer */
        float y = (corrected >= 0.0f) ? 1.0f : -1.0f;

        /* Update error history */
        e3 = e2;
        e2 = e1;
        e1 = y - corrected;  /* quantization error */

        ch_out[i] = y;
    }
}

} /* extern "C" */
