/*
 * foo_dsd_trellis — Audio quality measurement for "Test Quality" button
 */

#ifndef SINAD_MEASURE_H
#define SINAD_MEASURE_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    /* Theoretical SINAD: clean analog sine → SDM → Goertzel (0-20kHz) */
    double   sinad_theoretical;
    double   sinad_awtd_theo;     /* A-weighted theoretical SINAD */

    /* Multitone: 32-tone pipeline SINAD (complex signal quality) */
    double   multitone_sinad_db;

    /* Noise modulation: noise floor variation across signal levels (dB) */
    double   noise_mod_db;

    /* NMR: noise-to-mask ratio, simplified PEAQ (lower = better) */
    double   nmr_db;

    /* SDM diagnostics */
    uint64_t conv_fail;
    uint64_t cands_collapse;
    double   drop_pct;
    int      ok;              /* 1 = success, 0 = error */
} sinad_result_t;

/* Measure audio quality for a given same-rate DSD re-encode path.
 * Runs 4 metrics: A-weighted SINAD, multitone, noise modulation, NMR.
 *
 * dsd_rate:        full DSD rate (e.g., 2822400 for DSD64)
 * ntf_id:          resolved NTF filter ID (not -1/Auto)
 * cands:           trellis candidates
 * depth:           trellis depth
 * lat:             trellis latency
 * use_fir_lowpass: 0=boxcar, 1=FIR lowpass
 * fir_gain:        linear gain (e.g., 0.708 for -3 dB)
 *
 * Blocking call, ~2-8 seconds depending on DSD rate. */
void sinad_measure(uint32_t dsd_rate, int ntf_id,
                   int cands, int depth, int lat,
                   int use_fir_lowpass, float fir_gain,
                   sinad_result_t *result);

/* Measure DSD→DSD rate conversion quality.
 * Generates clean PCM sine, encodes to DSD at fs_in, FIR rate-converts,
 * re-encodes via SDM at fs_out, decimates to PCM, measures SINAD.
 * Tests the full conversion pipeline. */
void sinad_measure_dsd_to_dsd(uint32_t fs_in, uint32_t fs_out,
                               int ntf_id, int cands, int depth, int lat,
                               float fir_gain, double state_limit,
                               sinad_result_t *result);

/* Measure DSD→PCM decimation quality.
 * Generates DSD sine at dsd_rate, decimates via FIR to pcm_rate.
 * Returns SINAD in result->sinad_theoretical. */
void sinad_measure_dsd_to_pcm(uint32_t dsd_rate, uint32_t pcm_rate,
                               sinad_result_t *result);

/* Measure PCM→PCM resampling quality.
 * Generates PCM sine at fs_in, resamples to fs_out.
 * resample_engine: RESAMPLE_AUTO/IPP/SOXR.
 * soxr_quality: SOXR_QUALITY_MQ/HQ/VHQ. */
void sinad_measure_pcm_to_pcm(uint32_t fs_in, uint32_t fs_out,
                               int resample_engine, int soxr_quality,
                               sinad_result_t *result);

/* A-weighting factor (linear power) for a given frequency.
 * IEC 61672 standard. Returns multiplier for noise power. */
double a_weight_factor(double freq_hz);

#endif /* SINAD_MEASURE_H */
