/*
 * foo_dsd_trellis — Fast-mode trellis (compiled WITHOUT /fp:precise)
 *
 * ~2x faster than precise mode but up to 13 dB SINAD variation.
 * Used when FP32 precision is explicitly selected per-rate.
 * Compiled from the same source as trellis.c with renamed symbols.
 */

#define SDM_FAST_MODE

/* Rename public symbols to avoid linker conflicts with trellis.obj */
#define sdm_context_init       sdm_context_init_fast
#define sdm_process_block      sdm_process_block_fast
#define sdm_drain              sdm_drain_fast
#define sdm_context_copy_state sdm_context_copy_state_fast
#define sdm_state_distance     sdm_state_distance_fast
#define sdm_context_reset      sdm_context_reset_fast
#define sdm_context_free       sdm_context_free_fast
#define sdm_estimate_state     sdm_estimate_state_fast
#define sdm_estimate_state_multibit sdm_estimate_state_multibit_fast

#include "trellis.c"
