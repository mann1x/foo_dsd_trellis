/*
 * foo_dsd_trellis — CUDA FIR, gain, and boxcar kernels
 *
 * Compiled to PTX with: nvcc -ptx -arch=sm_52 fir_kernels.cu
 * PTX embedded as string constant in gpu_cuda.c
 */

extern "C" {

/* FIR coefficients in constant memory (uploaded once) */
__constant__ float c_taps[64];
__constant__ int   c_ntaps;

/* ─── FIR 2x Upsample ───
 * Zero-stuff input, convolve with FIR, scale by 2.
 * in_count: number of input samples
 * out_count = in_count * 2 */
__global__ void fir_upsample_2x(const float *in, float *out,
                                 int in_count, int out_count) {
    int oi = blockIdx.x * blockDim.x + threadIdx.x;
    if (oi >= out_count) return;

    float acc = 0.0f;
    int half = c_ntaps / 2;
    int zs_len = in_count * 2;

    for (int k = 0; k < c_ntaps; k++) {
        int zsi = oi - k + half;
        if (zsi >= 0 && zsi < zs_len) {
            /* Zero-stuffed: even indices = input, odd = 0 */
            if ((zsi & 1) == 0)
                acc += c_taps[k] * __ldg(&in[zsi >> 1]);
        }
    }
    out[oi] = acc * 2.0f;
}

/* ─── FIR 2x Downsample ───
 * Convolve with FIR, then decimate by 2.
 * in_count: number of input samples
 * out_count = in_count / 2 */
__global__ void fir_downsample_2x(const float *in, float *out,
                                   int in_count, int out_count) {
    int oi = blockIdx.x * blockDim.x + threadIdx.x;
    if (oi >= out_count) return;

    int ii = oi * 2;  /* decimated position */
    float acc = 0.0f;
    int half = c_ntaps / 2;

    for (int k = 0; k < c_ntaps; k++) {
        int si = ii - k + half;
        if (si >= 0 && si < in_count)
            acc += c_taps[k] * __ldg(&in[si]);
    }
    out[oi] = acc;
}

/* ─── Gain multiply (in-place) ─── */
__global__ void gain_apply(float *buf, int count, float gain) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        buf[i] *= gain;
}

/* ─── Boxcar smoothing + gain ───
 * Converts ±1.0 DSD → multi-bit via running average, applies gain.
 * Causal boxcar: output[i] = mean(sign(in[i-taps+1..i])) * gain */
__global__ void boxcar_smooth(const float *in, float *out,
                               int count, int taps, float gain) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float sum = 0.0f;
    for (int k = 0; k < taps; k++) {
        int si = i - k;
        float v = (si >= 0) ? in[si] : 0.0f;
        sum += (v >= 0.0f) ? 1.0f : -1.0f;
    }
    out[i] = sum / (float)taps * gain;
}

} /* extern "C" */
