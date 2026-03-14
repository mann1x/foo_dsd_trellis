/*
 * foo_dsd_trellis — NTF coefficient tables
 * Ported from mansr/sox sdm.c (LGPL v2.1+)
 */

#ifndef NTF_H
#define NTF_H

#include "dsd_types.h"

#define MAX_NTF_ORDER 8

/* NTF filter coefficient structure */
typedef struct {
    double      a[MAX_NTF_ORDER];   /* Feedback coefficients */
    double      g[MAX_NTF_ORDER];   /* Resonator gain coefficients */
    int         order;              /* Filter order (4-8) */
    unsigned    freq;               /* Target frequency (DSD rate) */
    const char *name;               /* Filter name (e.g. "clans-5") */
} ntf_filter_t;

/* Get filter by enum ID and DSD rate. Returns NULL if not found. */
const ntf_filter_t *ntf_get_filter(ntf_filter_id_t id, unsigned dsd_rate);

/* Auto-select best filter for a given DSD rate (CLANS-N, order matched to rate). */
const ntf_filter_t *ntf_auto_select(unsigned dsd_rate);

/* Auto-select best filter for PreCorr SDM at a given DSD rate.
 * PreCorr's greedy quantizer benefits from +1 NTF order vs Trellis. */
const ntf_filter_t *ntf_auto_select_precorr(unsigned dsd_rate);

/* Get filter count */
int ntf_get_filter_count(void);

/* Get filter by index (for enumeration) */
const ntf_filter_t *ntf_get_by_index(int index);

#endif /* NTF_H */
