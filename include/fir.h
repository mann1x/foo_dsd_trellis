/*
 * foo_dsd_trellis — FIR half-band filter for DSD rate conversion
 *
 * Uses Intel IPP ippsFIRSR_32f with a 63-tap Kaiser half-band filter.
 * Multi-stage 2x up/down-sampling via zero-stuffing + FIR + scaling.
 */

#ifndef FIR_H
#define FIR_H

#include "dsd_types.h"

/* Rate conversion chain (up to 9 stages for PCM 44.1k → DSD512) */
#define FIR_MAX_STAGES 9

/* IPP FIRSR half-band filter length */
#define IPP_HB_NTAPS 63

/* Global half-band taps (computed once, shared with GPU backend) */
extern float g_hb_taps[IPP_HB_NTAPS];
extern int   g_hb_ntaps;

typedef struct {
    int         num_stages;     /* 0 = passthrough */
    float      *scratch;        /* Intermediate buffer between stages */
    size_t      scratch_size;
    bool        upsample;       /* Direction for all stages */

    /* DSD-Wide demodulation (pre-rate-conversion LP filter) */
    bool        has_demod;      /* true when demod stage is active */
    void       *demod_spec;     /* IppsFIRSpec_32f* */
    void       *demod_buf;      /* Ipp8u* work buffer */
    float      *demod_dly;      /* Delay line (tapsLen-1 floats) */
    float      *demod_tmp;      /* Temp output buffer (same size as input) */
    size_t      demod_tmp_sz;

    /* IPP FIRSR state (per stage) */
    void       *ipp_spec[FIR_MAX_STAGES];   /* IppsFIRSpec_32f* */
    void       *ipp_buf[FIR_MAX_STAGES];    /* Ipp8u* work buffer */
    float      *ipp_dly[FIR_MAX_STAGES];    /* Delay lines (tapsLen-1 floats) */
    float      *ipp_zerostuff;              /* Temp buffer for zero-stuffing */
    size_t      ipp_zerostuff_sz;
    int         ipp_taps_len;               /* Actual filter length */
} fir_chain_t;

/* Initialise rate conversion chain for given input/output rates.
 * Returns 0 on success, -1 if rates are not a valid power-of-2 ratio. */
int fir_chain_init(fir_chain_t *chain, uint32_t fs_in, uint32_t fs_out);

/* Process a block of float32 samples through the FIR chain.
 * Returns the number of output samples written. */
size_t fir_chain_process(fir_chain_t *chain,
                         const float *in, float *out,
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
