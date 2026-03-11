/*
 * foo_dsd_trellis — Thread pool tests
 * Phase 5 will add concurrency stress tests.
 */

#include "test.h"
#include "../include/threadpool.h"

static void test_threadpool_create_destroy(void) {
    threadpool_t *pool = threadpool_create(2, 0);
    TEST_ASSERT_NOT_NULL(pool, "pool creation should succeed");
    TEST_ASSERT_EQ(threadpool_get_thread_count(pool), 2,
                   "thread count should be 2");
    threadpool_destroy(pool);
}

static void test_threadpool_auto_count(void) {
    threadpool_t *pool = threadpool_create(0, 0);
    TEST_ASSERT_NOT_NULL(pool, "auto-count pool should succeed");
    TEST_ASSERT_TRUE(threadpool_get_thread_count(pool) >= 1,
                     "auto-count should give at least 1 thread");
    threadpool_destroy(pool);
}

void test_threadpool_suite(void) {
    TEST_SUITE("Thread Pool");
    TEST_RUN(test_threadpool_create_destroy);
    TEST_RUN(test_threadpool_auto_count);
}
