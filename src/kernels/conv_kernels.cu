/*
 * foo_dsd_trellis — GPU convolution kernels
 *
 * Used with cuFFT for UPOLS convolution at full DSD rate.
 * Kernels handle the non-FFT parts: real→complex packing,
 * complex multiply-accumulate, and output extraction.
 */

/* Complex double type (matches cuDoubleComplex / Ipp64fc) */
struct cdouble { double re, im; };

/* Pack real doubles into complex (imaginary = 0) */
extern "C" __global__
void conv_real_to_complex(const double *in, cdouble *out,
                           int count, int fft_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < fft_size) {
        out[i].re = (i < count) ? in[i] : 0.0;
        out[i].im = 0.0;
    }
}

/* Element-wise complex multiply: out = a * b */
extern "C" __global__
void conv_complex_mul(const cdouble *a, const cdouble *b,
                       cdouble *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i].re = a[i].re * b[i].re - a[i].im * b[i].im;
        out[i].im = a[i].re * b[i].im + a[i].im * b[i].re;
    }
}

/* Complex multiply-accumulate: accum += a * b */
extern "C" __global__
void conv_complex_mul_acc(const cdouble *a, const cdouble *b,
                           cdouble *accum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double re = a[i].re * b[i].re - a[i].im * b[i].im;
        double im = a[i].re * b[i].im + a[i].im * b[i].re;
        accum[i].re += re;
        accum[i].im += im;
    }
}

/* Zero a complex buffer */
extern "C" __global__
void conv_complex_zero(cdouble *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        buf[i].re = 0.0;
        buf[i].im = 0.0;
    }
}

/* Extract second half of IFFT output (overlap-save valid region).
 * Reads complex IFFT output, writes real doubles. */
extern "C" __global__
void conv_extract_output(const cdouble *ifft_out, double *out,
                          int P, int fft_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < P) {
        out[i] = ifft_out[P + i].re;
    }
}

/* Copy FDL slot: dst = src (complex buffer copy) */
extern "C" __global__
void conv_copy_complex(const cdouble *src, cdouble *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}
