/*
 * foo_dsd_trellis — FIR half-band filter for DSD rate conversion
 *
 * Uses Intel IPP ippsFIRMR (multi-rate polyphase) with a 63-tap Kaiser
 * half-band filter. Multi-stage 2x up/down-sampling without zero-stuffing.
 */

#ifndef FIR_H
#define FIR_H

#include "dsd_types.h"

/* Rate conversion chain (up to 9 stages for PCM 44.1k → DSD512) */
#define FIR_MAX_STAGES 9

/* IPP half-band filter length (must be odd).
 * 63 = ~120 dB stopband at beta=12. 127 = sharper transition. 255 = even sharper. */
#define IPP_HB_NTAPS_MAX 255
#define IPP_HB_NTAPS 63

/* Global half-band taps (computed once, shared with GPU backend) */
extern float  g_hb_taps[IPP_HB_NTAPS_MAX];
extern double g_hb_taps_d[IPP_HB_NTAPS_MAX];
extern int    g_hb_ntaps;

typedef struct {
    int         num_stages;     /* 0 = passthrough */
    float      *scratch;        /* Intermediate buffer between stages (fp32 path) */
    size_t      scratch_size;
    bool        upsample;       /* Direction for all stages */
    bool        use_fp64;       /* true: use fp64 FIR path */

    /* DSD-Wide demodulation (pre-rate-conversion LP filter) */
    bool        has_demod;      /* true when demod stage is active */
    void       *demod_spec;     /* IppsFIRSpec_32f* */
    void       *demod_buf;      /* Ipp8u* work buffer */
    float      *demod_dly;      /* Delay line (tapsLen-1 floats) */
    float      *demod_tmp;      /* Temp output buffer (same size as input) */
    size_t      demod_tmp_sz;

    /* IPP FIRMR state — fp32 (per stage, polyphase multi-rate) */
    void       *ipp_spec[FIR_MAX_STAGES];   /* IppsFIRSpec_32f* */
    void       *ipp_buf[FIR_MAX_STAGES];    /* Ipp8u* work buffer */
    float      *ipp_dly[FIR_MAX_STAGES];    /* Delay lines (tapsLen-1 floats) */
    int         ipp_taps[FIR_MAX_STAGES];   /* Per-stage filter length */

    /* IPP FIRMR state — fp64 (per stage, polyphase multi-rate) */
    void       *ipp_spec_d[FIR_MAX_STAGES]; /* IppsFIRSpec_64f* */
    void       *ipp_buf_d[FIR_MAX_STAGES];  /* Ipp8u* work buffer */
    double     *ipp_dly_d[FIR_MAX_STAGES];  /* Delay lines (tapsLen-1 doubles) */
    double     *scratch_d;                  /* Intermediate buffer (fp64 path) */
    size_t      scratch_d_size;
} fir_chain_t;

/* Initialise rate conversion chain for given input/output rates.
 * Returns 0 on success, -1 if rates are not a valid power-of-2 ratio. */
int fir_chain_init(fir_chain_t *chain, uint32_t fs_in, uint32_t fs_out);

/* Initialise rate conversion chain with explicit precision selection.
 * use_fp64=true selects the fp64 path (ippsFIRSR_64f), false uses fp32.
 * Returns 0 on success, -1 if rates are not a valid power-of-2 ratio. */
int fir_chain_init_ex(fir_chain_t *chain, uint32_t fs_in, uint32_t fs_out,
                      bool use_fp64);

/* Process a block of float32 samples through the FIR chain (fp32 path).
 * Returns the number of output samples written. */
size_t fir_chain_process(fir_chain_t *chain,
                         const float *in, float *out,
                         size_t in_count);

/* Process a block of float64 samples through the FIR chain (fp64 path).
 * Requires chain initialised with use_fp64=true.
 * Returns the number of output samples written. */
size_t fir_chain_process_d(fir_chain_t *chain,
                           const double *in, double *out,
                           size_t in_count);

/* Reset filter state (on seek / discontinuity). */
void fir_chain_reset(fir_chain_t *chain);

/* Free scratch buffer and reset. */
void fir_chain_free(fir_chain_t *chain);

/* Same-rate lowpass FIR for DSD-Wide re-encoding.
 * Replaces the boxcar with a proper IPP FIRSR lowpass.
 * Produces smooth multi-bit output for Trellis SDM.
 * Uses fp64 throughout to match the SDM pipeline precision. */
typedef struct {
    void   *spec;       /* IppsFIRSpec_64f* */
    void   *buf;        /* Ipp8u* work buffer */
    double *dly;        /* Delay line (fp64) */
    float  *coeffs;     /* Tap coefficients fp32 (kept for GPU upload) */
    double *coeffs_d;   /* Tap coefficients fp64 */
    int     taps;       /* Filter length */
    bool    initialized;
} fir_lowpass_t;

int fir_lowpass_init(fir_lowpass_t *lp, uint32_t dsd_rate);
size_t fir_lowpass_process(fir_lowpass_t *lp, const double *in, double *out, size_t count);
void fir_lowpass_reset(fir_lowpass_t *lp);
void fir_lowpass_free(fir_lowpass_t *lp);

/* IPP info */
const char *fir_ipp_version(void);
const char *fir_ipp_kernel_name(void);

#endif /* FIR_H */
