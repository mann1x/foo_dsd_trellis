/*
 * foo_dsd_trellis — Minimal test framework
 * No external dependencies — just assert macros and a test runner.
 */

#ifndef TEST_H
#define TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Counters defined in test_main.c, shared across all TUs */
extern int g_tests_run;
extern int g_tests_passed;
extern int g_tests_failed;

#define TEST_ASSERT(cond, msg) do {                                    \
    g_tests_run++;                                                     \
    if (!(cond)) {                                                     \
        printf("  FAIL: %s:%d: %s\n", __FILE__, __LINE__, (msg));      \
        g_tests_failed++;                                              \
    } else {                                                           \
        g_tests_passed++;                                              \
    }                                                                  \
} while (0)

#define TEST_ASSERT_EQ(a, b, msg)    TEST_ASSERT((a) == (b), msg)
#define TEST_ASSERT_NEQ(a, b, msg)   TEST_ASSERT((a) != (b), msg)
#define TEST_ASSERT_TRUE(a, msg)     TEST_ASSERT((a), msg)
#define TEST_ASSERT_FALSE(a, msg)    TEST_ASSERT(!(a), msg)
#define TEST_ASSERT_NULL(a, msg)     TEST_ASSERT((a) == NULL, msg)
#define TEST_ASSERT_NOT_NULL(a, msg) TEST_ASSERT((a) != NULL, msg)

#define TEST_ASSERT_FLOAT_EQ(a, b, eps, msg) \
    TEST_ASSERT(fabs((double)(a) - (double)(b)) < (eps), msg)

#define TEST_RUN(fn) do {                          \
    printf("  %s...\n", #fn);                      \
    fn();                                          \
} while (0)

#define TEST_SUITE(name) \
    printf("\n=== %s ===\n", name)

#define TEST_SUMMARY() do {                                                 \
    printf("\n--- Results: %d/%d passed", g_tests_passed, g_tests_run);     \
    if (g_tests_failed) printf(" (%d FAILED)", g_tests_failed);             \
    printf(" ---\n");                                                       \
    return g_tests_failed ? 1 : 0;                                          \
} while (0)

/* Declarations for test suites (implemented in separate files) */
void test_dop_suite(void);
void test_ntf_suite(void);
void test_fir_suite(void);
void test_trellis_suite(void);
void test_threadpool_suite(void);
void test_hardening_suite(void);
void test_config_suite(void);
void test_simd_suite(void);

#endif /* TEST_H */
