/*
 * foo_dsd_trellis — SINAD measurement CLI tool
 *
 * Standalone: reads a WAV sine tone, encodes to DSD via the engine,
 * measures in-band SINAD, and prints a summary.
 *
 * Phase 0: Scaffold — placeholder.
 */

#include <stdio.h>
#include <stdlib.h>
#include "../include/dsd_types.h"
#include "../include/trellis.h"
#include "../include/ntf.h"

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    printf("foo_dsd_trellis SINAD check tool\n");
    printf("Not yet implemented (Phase 3+)\n");

    /* Quick sanity: verify NTF lookup works */
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_64);
    if (f) {
        printf("NTF filter: %s, order %d, freq %u\n",
               f->name, f->order, f->freq);
    } else {
        printf("ERROR: No NTF filter found for DSD64\n");
        return 1;
    }

    return 0;
}
