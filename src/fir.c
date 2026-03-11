/*
 * foo_dsd_trellis — Polyphase FIR half-band filter for DSD rate conversion
 *
 * Phase 0: Scaffold — stub implementations.
 * Phase 4 will implement the polyphase half-band FIR.
 */

#include "../include/fir.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int fir_chain_init(fir_chain_t *chain, uint32_t fs_in, uint32_t fs_out) {
    memset(chain, 0, sizeof(*chain));

    if (fs_in == fs_out) {
        chain->num_stages = 0;
        return 0;
    }

    /* Determine number of ×2 or ÷2 stages needed */
    uint32_t ratio;
    bool upsample;

    if (fs_out > fs_in) {
        ratio = fs_out / fs_in;
        upsample = true;
    } else {
        ratio = fs_in / fs_out;
        upsample = false;
    }

    /* Must be power of 2 */
    if (ratio == 0 || (ratio & (ratio - 1)) != 0)
        return -1;

    int stages = 0;
    uint32_t r = ratio;
    while (r > 1) {
        stages++;
        r >>= 1;
    }

    if (stages > FIR_MAX_STAGES)
        return -1;

    chain->num_stages = stages;

    for (int i = 0; i < stages; i++) {
        chain->stages[i].upsample = upsample;
        /* TODO (Phase 4): Load half-band FIR coefficients into polyphase phases */
    }

    return 0;
}

size_t fir_chain_process(fir_chain_t *chain,
                         const float *in, float *out,
                         size_t in_count) {
    if (chain->num_stages == 0) {
        /* Passthrough */
        memcpy(out, in, in_count * sizeof(float));
        return in_count;
    }

    /* TODO (Phase 4): Polyphase FIR chain processing */
    memcpy(out, in, in_count * sizeof(float));
    return in_count;
}

void fir_chain_reset(fir_chain_t *chain) {
    for (int i = 0; i < chain->num_stages; i++) {
        memset(chain->stages[i].phase[0].delay, 0,
               sizeof(chain->stages[i].phase[0].delay));
        memset(chain->stages[i].phase[1].delay, 0,
               sizeof(chain->stages[i].phase[1].delay));
        chain->stages[i].phase[0].delay_pos = 0;
        chain->stages[i].phase[1].delay_pos = 0;
    }
}

void fir_chain_free(fir_chain_t *chain) {
    free(chain->scratch);
    chain->scratch = NULL;
    chain->scratch_size = 0;
    chain->num_stages = 0;
}
