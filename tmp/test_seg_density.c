/* Quick test: DAS SINAD with varying segment counts */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/gpu.h"
#include "../include/ntf.h"
#include "../include/trellis.h"

extern double measure_sinad(const float *dsd, size_t n, uint32_t rate, double freq);
extern float *make_test_signal(const ntf_filter_t *f, uint32_t rate, double freq, size_t n, size_t *out_n);

int main(void) {
    printf("Segment density vs SINAD test\n");
    /* This would need custom gpu_cuda_trellis_das with configurable num_segs
       For now, just run the existing test with different chunk sizes */
    return 0;
}
