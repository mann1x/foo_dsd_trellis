/*
 * foo_dsd_trellis — Convolution filter (room correction)
 *
 * Uniform-Partitioned Overlap-Save (UPOLS) convolution using IPP FFT in fp64.
 * Loads WAV impulse responses per channel, resamples to processing rate,
 * and applies real-time convolution. For DSD paths, decimates to an
 * intermediate PCM rate (176.4/192 kHz), convolves, then interpolates back.
 */

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CONV_MAX_CHANNELS    6
#define CONV_PATH_MAX        260   /* MAX_PATH on Windows */
#define CONV_MAX_IR_TAPS     (1 << 22)  /* 4M taps max (44.1k IR at DSD512 = ~2M) */

/* Forward declarations for IPP types (avoid including ipps.h in header) */
struct Ipp64fc_tag;

/* Pre-FFT'd impulse response */
typedef struct {
    void    *freq_partitions;   /* Ipp64fc* array [num_partitions * fft_size] */
    int      num_partitions;
    int      partition_size;    /* P (real samples per partition) */
    int      fft_size;          /* 2*P (complex FFT length) */
    int      fft_order;         /* log2(fft_size) */
    int      ir_length;         /* Original tap count (at target rate) */
    uint32_t target_rate;       /* Rate IR was resampled to */
} conv_ir_t;

/* Per-channel convolution state */
typedef struct {
    conv_ir_t  *ir;             /* Points to loaded IR (NULL = passthrough) */
    double     *input_buf;      /* 2*P real samples (overlap-save ring) */
    void       *fft_buf;        /* Ipp64fc[fft_size] scratch for forward FFT */
    void       *accum;          /* Ipp64fc[fft_size] complex accumulator */
    void      **fdl;            /* Frequency Domain Delay Line: Ipp64fc*[num_partitions] */
    int         fdl_pos;        /* Circular write index into FDL */
    int         fdl_filled;     /* Number of FDL slots filled (0..num_partitions) */
    int         buf_pos;        /* Samples accumulated in current partition (0..P-1) */
    double     *out_buf;        /* Buffered output (P doubles, for partial reads) */
    int         out_avail;      /* Samples available in out_buf */
    int         out_read;       /* Read position in out_buf */
    /* IPP FFT state */
    void       *fft_spec;       /* IppsFFTSpec_C_64fc* */
    void       *fft_work;       /* Ipp8u* work buffer */
} conv_channel_t;

/* Top-level convolution engine (one per engine_channel_t) */
typedef struct {
    conv_channel_t  ch;         /* Single channel state */
    conv_ir_t       ir;         /* Owned IR data */
    bool            active;     /* IR loaded and ready */
    uint32_t        conv_rate;  /* Rate at which convolution operates */
    uint32_t        signal_rate;/* Actual signal rate (DSD or PCM) */
    /* DSD paths: decimate/interpolate */
    double         *dec_buf;    /* Decimated signal buffer */
    size_t          dec_buf_sz; /* Allocated size in samples */
    int             dec_ratio;  /* Decimation factor (1 = no decimation) */
    /* GPU convolution (NULL if CPU-only) */
    void           *gpu_conv;   /* gpu_conv_state_t* from gpu_compute.h */
    void           *gpu_ctx;    /* gpu_context_t* for GPU calls */
    bool            use_gpu;    /* true if GPU convolution is active */
    bool            gpu_partitions; /* true: use larger P for GPU efficiency */
    int             max_ir_taps;    /* 0=no limit, >0=truncate IR to this many taps */
    /* IR group delay in samples at conv_rate. For linear-phase IRs this
     * is the position of the impulse peak in the (post-truncation) IR.
     * Used by plugin_get_latency to report A/V sync compensation to fb2k. */
    int             latency_samples;
} conv_state_t;

/* Initialise convolution state for a given processing rate.
 * signal_rate: actual sample rate of the signal (DSD rate or PCM rate)
 * conv_rate:   rate at which to perform convolution (176.4k/192k for DSD, same as signal for PCM)
 * Returns 0 on success. */
int conv_init(conv_state_t *state, uint32_t signal_rate, uint32_t conv_rate);

/* Load an impulse response WAV file for this channel.
 * The IR is resampled to the convolution rate and partitioned.
 * Returns 0 on success, -1 on error (file not found, invalid format, etc.) */
int conv_load_ir(conv_state_t *state, const char *wav_path);

/* Process a block of fp64 samples in-place.
 * Handles arbitrary block sizes with internal buffering.
 * If decimation is configured (DSD paths), decimates first, convolves, then interpolates.
 * buf:   fp64 samples (modified in-place)
 * count: number of samples */
void conv_process(conv_state_t *state, double *buf, size_t count);

/* Pipelined process: launch + finalize.
 * For GPU conv: launch queues GPU work asynchronously (no sync), finalize
 * syncs and drains output to buf. Multiple channels can launch in parallel
 * (each on its own CUDA stream) before any sync, allowing overlap.
 * For CPU conv: launch is no-op, finalize calls conv_process. */
void conv_launch(conv_state_t *state, double *buf, size_t count);
void conv_finalize(conv_state_t *state, double *buf, size_t count);

/* Reset convolution state (seek / track change).
 * Clears delay lines and internal buffers but keeps the loaded IR. */
void conv_reset(conv_state_t *state);

/* Free all convolution resources. */
void conv_free(conv_state_t *state);

#ifdef __cplusplus
}
#endif

#endif /* CONVOLUTION_H */
