/*
 * foo_dsd_trellis — Test runner entry point
 */

#include "test.h"

/* Shared test counters */
int g_tests_run = 0;
int g_tests_passed = 0;
int g_tests_failed = 0;

int main(void) {
    printf("foo_dsd_trellis test suite\n");

    test_dop_suite();
    test_ntf_suite();
    test_fir_suite();
    test_trellis_suite();
    test_threadpool_suite();

    TEST_SUMMARY();
}
