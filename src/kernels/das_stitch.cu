/*
 * foo_dsd_trellis — Density-Aligned Stitching (DAS) GPU kernels
 *
 * Two kernels that run after the parallel-segment SBVD:
 * 1. das_density_scan  — find optimal stitch points via windowed match density
 * 2. das_assemble      — gather stitched output from all segments
 *
 * Operates entirely on device memory. Host downloads only the final output.
 */

extern "C" {

/* ─── Kernel 1: Windowed Match Density Scan ─── */

/* Grid:  (num_boundaries, 1, 1)   — one block per segment boundary
 * Block: (256, 1, 1)
 *
 * For each boundary between segment b and b+1:
 * - prev_ovl = seg b's output at [D_output .. D_output + overlap)
 * - next_ovl = seg (b+1)'s output at [0 .. overlap)
 * - Scan overlap positions with a sliding window of width 2×trellis_lat
 * - Find position with highest match density
 * - Spiral outward from peak to find nearest exact bit-match
 * - Write stitch position for this boundary
 *
 * Only operates on channel 0 segment outputs.
 */
__global__ void das_density_scan(
    const float *seg_out,          /* ch0 segment outputs (contiguous) */
    const int *seg_out_starts,     /* [num_segs] output offset per segment */
    const int *seg_out_caps,       /* [num_segs] output capacity per segment */
    int D_output,                  /* nominal output per segment */
    int overlap_samples,           /* overlap region size */
    int trellis_lat,               /* half-window = trellis_lat */
    int *stitch_positions)         /* OUTPUT: [num_boundaries] */
{
    int b = blockIdx.x;  /* boundary index: between seg b and seg b+1 */
    int tid = threadIdx.x;

    /* Overlap region pointers within ch0 output.
     * Segment b extended by overlap: output[D..D+overlap) is the extension.
     * Segment b+1 starts with overlap warmup-produced output at [0..overlap). */
    const float *prev_ovl = seg_out + seg_out_starts[b] + D_output;
    const float *next_ovl = seg_out + seg_out_starts[b + 1];

    int ovl_len = overlap_samples;
    /* Clamp to actual output capacity */
    int prev_cap = seg_out_caps[b];
    if (D_output + ovl_len > prev_cap)
        ovl_len = prev_cap - D_output;
    int next_cap = seg_out_caps[b + 1];
    if (ovl_len > next_cap)
        ovl_len = next_cap;
    if (ovl_len <= 0) {
        if (tid == 0) stitch_positions[b] = 0;
        return;
    }

    int half_w = trellis_lat;
    if (half_w > ovl_len / 2) half_w = ovl_len / 2;
    if (half_w < 4) half_w = 4;

    /* Load overlap regions into shared memory for fast access */
    extern __shared__ float s_ovl[];  /* 2 × ovl_len floats */
    float *s_prev = s_ovl;
    float *s_next = s_ovl + ovl_len;

    for (int i = tid; i < ovl_len; i += blockDim.x) {
        s_prev[i] = prev_ovl[i];
        s_next[i] = next_ovl[i];
    }
    __syncthreads();

    /* Each thread computes density for its assigned positions */
    __shared__ int s_best_density[256];
    __shared__ int s_best_pos[256];

    int my_best_density = 0;
    int my_best_pos = 0;

    for (int p = tid; p < ovl_len; p += blockDim.x) {
        int start = p - half_w;
        int end   = p + half_w;
        if (start < 0) start = 0;
        if (end > ovl_len) end = ovl_len;
        int matches = 0;
        for (int w = start; w < end; w++) {
            if (s_prev[w] == s_next[w])
                matches++;
        }
        if (matches > my_best_density) {
            my_best_density = matches;
            my_best_pos = p;
        }
    }

    s_best_density[tid] = my_best_density;
    s_best_pos[tid] = my_best_pos;
    __syncthreads();

    /* Block-wide parallel reduction for argmax */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_best_density[tid + stride] > s_best_density[tid]) {
                s_best_density[tid] = s_best_density[tid + stride];
                s_best_pos[tid] = s_best_pos[tid + stride];
            }
        }
        __syncthreads();
    }

    /* Thread 0: spiral search from density peak for exact bit-match */
    if (tid == 0) {
        int peak = s_best_pos[0];
        int best = peak;  /* fallback: density peak */

        for (int r = 0; r <= half_w; r++) {
            int lo = peak - r;
            int hi = peak + r;
            if (lo >= 0 && lo < ovl_len &&
                s_prev[lo] == s_next[lo]) {
                best = lo;
                break;
            }
            if (hi != lo && hi >= 0 && hi < ovl_len &&
                s_prev[hi] == s_next[hi]) {
                best = hi;
                break;
            }
        }

        stitch_positions[b] = best;
    }
}


/* ─── Kernel 2: Gather-Assemble with Dithered Crossfade ─── */

/* Grid:  (ceil(final_count / 256), num_channels, 1)
 * Block: (256, 1, 1)
 *
 * Assembles output from segments using stitch positions from ch0.
 * At each stitch boundary, applies a dithered crossfade over a window
 * of 'fade_width' samples to smooth the noise texture transition.
 * Crossfade uses probability-weighted random bit selection:
 *   p = position_in_fade / fade_width  (0→1)
 *   output = (hash < p) ? seg_next : seg_prev
 * This keeps output in ±1.0 DSD domain while smoothing transitions.
 */

/* Fast deterministic hash for dithered crossfade.
 * Produces uniform [0,1) from sample position + channel. */
__device__ float das_hash(int pos, int ch, int boundary) {
    unsigned h = (unsigned)pos * 2654435761u ^ (unsigned)ch * 2246822519u
               ^ (unsigned)boundary * 3266489917u;
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return (float)(h & 0x7FFFFF) / 8388608.0f;  /* 23-bit mantissa → [0,1) */
}

__global__ void das_assemble(
    const float *seg_out,          /* all segment outputs [num_ch × total_cap] */
    float *final_out,              /* assembled output [num_ch × final_count] */
    const int *seg_out_starts,     /* [num_segs] per-channel output offsets */
    const int *stitch_positions,   /* [num_segs - 1] from density scan (ch0) */
    int num_segs,
    int D_output,                  /* nominal D per segment */
    int overlap_samples,
    int final_count,               /* output samples per channel */
    int ch_stride_seg,             /* per-channel stride in seg_out */
    int ch_stride_final,           /* per-channel stride in final_out */
    int fade_half_width)           /* half-width of crossfade window */
{
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int ch   = blockIdx.y;

    if (gidx >= final_count) return;

    __shared__ int s_cum_end[512];
    __shared__ int s_stitch[512];

    /* Cooperatively load stitch positions */
    for (int i = threadIdx.x; i < num_segs - 1; i += blockDim.x)
        s_stitch[i] = stitch_positions[i];
    __syncthreads();

    /* Build cumulative boundary table */
    if (threadIdx.x == 0) {
        int cum = 0;
        for (int s = 0; s < num_segs; s++) {
            int seg_take;
            if (s == 0)
                seg_take = D_output + (num_segs > 1 ? s_stitch[0] : 0);
            else if (s < num_segs - 1)
                seg_take = D_output + s_stitch[s] - s_stitch[s - 1];
            else
                seg_take = final_count - cum;
            cum += seg_take;
            s_cum_end[s] = cum;
        }
    }
    __syncthreads();

    /* Binary search for source segment */
    int src_seg = 0;
    {
        int lo = 0, hi = num_segs - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (s_cum_end[mid] <= gidx) lo = mid + 1;
            else hi = mid;
        }
        src_seg = lo;
    }

    int seg_start_in_final = (src_seg > 0) ? s_cum_end[src_seg - 1] : 0;
    int local_idx = gidx - seg_start_in_final;
    int seg_take = s_cum_end[src_seg] - seg_start_in_final;

    /* Read offset within source segment */
    int seg_read_offset;
    if (src_seg == 0)
        seg_read_offset = local_idx;
    else
        seg_read_offset = s_stitch[src_seg - 1] + local_idx;

    int seg_base = ch * ch_stride_seg;
    float val = seg_out[seg_base + seg_out_starts[src_seg] + seg_read_offset];

    /* Dithered crossfade at stitch boundaries.
     * For samples within fade_half_width of a boundary, blend between
     * the previous and next segment's output using probabilistic selection. */
    if (fade_half_width > 0) {
        /* Check if we're near the END of this segment (transition to next) */
        int dist_to_end = seg_take - 1 - local_idx;
        if (src_seg < num_segs - 1 && dist_to_end < fade_half_width) {
            /* Crossfade zone: transition from this segment to next.
             * p = 0 at fade start (fully this seg), 1 at boundary (fully next). */
            float p = 1.0f - (float)dist_to_end / (float)fade_half_width;
            float rnd = das_hash(gidx, ch, src_seg);
            if (rnd < p) {
                /* Use next segment's sample instead */
                int next_seg = src_seg + 1;
                /* At the boundary, next seg starts at stitch[src_seg].
                 * The sample we want from next seg is at the corresponding position. */
                int boundary_pos = s_cum_end[src_seg];  /* absolute pos of boundary */
                int abs_pos = seg_start_in_final + local_idx;
                int into_next = abs_pos - (boundary_pos - fade_half_width);
                int next_read = s_stitch[src_seg] - fade_half_width + into_next;
                if (next_read >= 0 && next_read < D_output + overlap_samples)
                    val = seg_out[seg_base + seg_out_starts[next_seg] + next_read];
            }
        }

        /* Check if we're near the START of this segment (transition from prev) */
        if (src_seg > 0 && local_idx < fade_half_width) {
            /* Crossfade zone: transition from previous segment to this.
             * p = 0 at boundary (fully prev seg), 1 at fade end (fully this seg). */
            float p = (float)local_idx / (float)fade_half_width;
            float rnd = das_hash(gidx, ch, src_seg - 1);
            if (rnd > p) {
                /* Use previous segment's sample instead */
                int prev_seg = src_seg - 1;
                /* Previous seg's output at this position */
                int prev_read_base;
                if (prev_seg == 0)
                    prev_read_base = 0;
                else
                    prev_read_base = s_stitch[prev_seg - 1];
                int prev_local = (seg_start_in_final + local_idx) -
                                 (prev_seg > 0 ? s_cum_end[prev_seg - 1] : 0);
                int prev_read = prev_read_base + prev_local;
                if (prev_read >= 0 && prev_read < D_output + overlap_samples)
                    val = seg_out[seg_base + seg_out_starts[prev_seg] + prev_read];
            }
        }
    }

    final_out[ch * ch_stride_final + gidx] = val;
}

} /* extern "C" */
