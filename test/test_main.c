/*
 * foo_dsd_trellis — Test runner with suite filtering
 *
 * Usage:
 *   test.exe                 — run quick suites only (default)
 *   test.exe --all           — run all suites including extended
 *   test.exe --suite NAME    — run only suites whose tag contains NAME (case-insensitive)
 *   test.exe --exclude NAME  — skip suites whose tag contains NAME
 *   test.exe --list          — list available suites and exit
 *
 * Examples:
 *   test.exe --suite diag          — run only SINAD diagnostics
 *   test.exe --suite precorr       — run only PreCorr tests
 *   test.exe --all --exclude diag  — run everything except diagnostics
 */

#include "test.h"

/* Shared test counters */
int g_tests_run = 0;
int g_tests_passed = 0;
int g_tests_failed = 0;

/* ─── Suite registry ─── */

typedef struct {
    const char *tag;        /* Short identifier for filtering */
    const char *name;       /* Display name */
    void (*fn)(void);       /* Suite function */
    int extended;           /* 1 = only runs with --all or explicit --suite */
} suite_entry_t;

static const suite_entry_t suites[] = {
    { "dop",       "DoP",               test_dop_suite,        0 },
    { "ntf",       "NTF",               test_ntf_suite,        0 },
    { "fir",       "FIR",               test_fir_suite,        0 },
    { "trellis",   "Trellis SDM",       test_trellis_suite,    0 },
    { "config",    "Config",            test_config_suite,     0 },
    { "simd",      "SIMD",              test_simd_suite,       0 },
    { "rate",      "Rate Conversion",   test_rate_sinad_suite, 0 },
    { "precorr",   "PreCorr SDM",       test_precorr_suite,    0 },
    { "hardening", "Hardening",         test_hardening_suite,  0 },
    { "threadpool","Thread Pool",       test_threadpool_suite,  0 },
    { "sweep",     "Rate Conv Sweep",   test_rate_sweep_suite, 1 },
    { "onnx",      "ONNX Filter",       test_onnx_filter_suite, 0 },
    { "gpu",       "GPU Compute",       test_gpu_suite,         0 },
    { "gpusinad",  "GPU SINAD Compare", test_gpu_sinad_comparison, 1 },
    { "diag",      "SINAD Diagnostics", test_sinad_diag_suite, 1 },
};

#define SUITE_COUNT (sizeof(suites) / sizeof(suites[0]))

/* ─── Filter state ─── */

#define MAX_FILTERS 16

static int g_run_all = 0;
static int g_has_include = 0;
static int g_include_count = 0;
static const char *g_includes[MAX_FILTERS];
static int g_exclude_count = 0;
static const char *g_excludes[MAX_FILTERS];

/* Case-insensitive substring match */
static int stricasestr(const char *haystack, const char *needle) {
    size_t nlen = strlen(needle);
    size_t hlen = strlen(haystack);
    if (nlen > hlen) return 0;
    for (size_t i = 0; i <= hlen - nlen; i++) {
        int match = 1;
        for (size_t j = 0; j < nlen; j++) {
            char h = haystack[i + j];
            char n = needle[j];
            if (h >= 'A' && h <= 'Z') h += 32;
            if (n >= 'A' && n <= 'Z') n += 32;
            if (h != n) { match = 0; break; }
        }
        if (match) return 1;
    }
    return 0;
}

int test_should_run_suite(const char *suite_tag) {
    /* Check excludes first */
    for (int i = 0; i < g_exclude_count; i++) {
        if (stricasestr(suite_tag, g_excludes[i]))
            return 0;
    }

    /* If explicit includes, must match at least one */
    if (g_has_include) {
        for (int i = 0; i < g_include_count; i++) {
            if (stricasestr(suite_tag, g_includes[i]))
                return 1;
        }
        return 0;
    }

    return 1;
}

static void print_usage(void) {
    printf("Usage: test.exe [options]\n");
    printf("  (no args)          Run quick suites only\n");
    printf("  --all              Run all suites including extended\n");
    printf("  --suite NAME       Run only suites matching NAME (repeatable)\n");
    printf("  --exclude NAME     Skip suites matching NAME (repeatable)\n");
    printf("  --list             List available suites and exit\n");
    printf("  --help             Show this help\n");
}

static void list_suites(void) {
    printf("Available test suites:\n");
    for (int i = 0; i < (int)SUITE_COUNT; i++) {
        printf("  %-12s  %s%s\n", suites[i].tag, suites[i].name,
               suites[i].extended ? "  [extended]" : "");
    }
}

int main(int argc, char **argv) {
    /* Dispatch to dsd_encode tool if --encode is first arg */
    if (argc >= 2 && strcmp(argv[1], "--encode") == 0)
        return dsd_encode_main(argc - 1, argv + 1);

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--all") == 0) {
            g_run_all = 1;
        } else if (strcmp(argv[i], "--suite") == 0 && i + 1 < argc) {
            g_has_include = 1;
            if (g_include_count < MAX_FILTERS)
                g_includes[g_include_count++] = argv[++i];
        } else if (strcmp(argv[i], "--exclude") == 0 && i + 1 < argc) {
            if (g_exclude_count < MAX_FILTERS)
                g_excludes[g_exclude_count++] = argv[++i];
        } else if (strcmp(argv[i], "--list") == 0) {
            list_suites();
            return 0;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        } else {
            printf("Unknown argument: %s\n", argv[i]);
            print_usage();
            return 1;
        }
    }

    printf("foo_dsd_trellis test suite\n");

    int suites_run = 0;
    for (int i = 0; i < (int)SUITE_COUNT; i++) {
        /* Skip extended suites unless --all or explicit --suite match */
        if (suites[i].extended && !g_run_all && !g_has_include)
            continue;

        if (!test_should_run_suite(suites[i].tag))
            continue;

        suites[i].fn();
        suites_run++;
    }

    if (suites_run == 0) {
        printf("\nNo suites matched the filter.\n");
        list_suites();
        return 1;
    }

    TEST_SUMMARY();
}
