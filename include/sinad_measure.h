/*
 * foo_dsd_trellis — SINAD measurement for UI "Test SINAD" button
 */

#ifndef SINAD_MEASURE_H
#define SINAD_MEASURE_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    double   sinad_db;        /* Pipeline SINAD: DSD→boxcar/FIR→SDM→decode (actual quality) */
    double   sinad_theoretical; /* Theoretical SINAD: analog sine→SDM (best-case) */
    uint64_t conv_fail;       /* SDM convergence failures */
    uint64_t cands_collapse;  /* Candidate collapses */
    double   drop_pct;        /* Next-filter drop percentage */
    int      ok;              /* 1 = success, 0 = error */
} sinad_result_t;

/* Measure SINAD for a given same-rate DSD re-encode path.
 * Generates 1kHz DSD test signal, processes through boxcar/FIR + SDM,
 * measures output SINAD via Goertzel analysis.
 *
 * dsd_rate:        full DSD rate (e.g., 2822400 for DSD64)
 * ntf_id:          resolved NTF filter ID (not -1/Auto)
 * cands:           trellis candidates
 * depth:           trellis depth
 * lat:             trellis latency
 * use_fir_lowpass: 0=boxcar, 1=FIR lowpass
 * fir_gain:        linear gain (e.g., 0.708 for -3 dB)
 *
 * Blocking call, ~0.5-2 seconds. */
void sinad_measure(uint32_t dsd_rate, int ntf_id,
                   int cands, int depth, int lat,
                   int use_fir_lowpass, float fir_gain,
                   sinad_result_t *result);

#endif /* SINAD_MEASURE_H */
