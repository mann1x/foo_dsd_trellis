/*
 * foo_dsd_trellis — Runtime CPU feature detection
 *
 * Uses CPUID to detect SSE2, AVX2, FMA3, and CPU vendor (Intel/AMD).
 * AMD Zen 1/2 prefer 128-bit AVX operations; Zen 3+ and all Intel
 * handle 256-bit AVX2 natively.
 */

#include "../include/simd_detect.h"

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

static cpu_features_t g_features;
static int g_detected = 0;

static void do_cpuid(int leaf, int subleaf, int regs[4]) {
#ifdef _MSC_VER
    __cpuidex(regs, leaf, subleaf);
#else
    __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
#endif
}

const cpu_features_t *cpu_detect(void) {
    if (g_detected)
        return &g_features;

    int regs[4];

    /* Leaf 0: max leaf + vendor string */
    do_cpuid(0, 0, regs);
    int max_leaf = regs[0];

    /* Vendor: EBX-EDX-ECX forms the 12-byte ASCII string */
    char vendor[13];
    *(int *)(vendor + 0) = regs[1]; /* EBX */
    *(int *)(vendor + 4) = regs[3]; /* EDX */
    *(int *)(vendor + 8) = regs[2]; /* ECX */
    vendor[12] = '\0';

    if (vendor[0] == 'G' && vendor[1] == 'e') /* "GenuineIntel" */
        g_features.vendor = CPU_VENDOR_INTEL;
    else if (vendor[0] == 'A' && vendor[1] == 'u') /* "AuthenticAMD" */
        g_features.vendor = CPU_VENDOR_AMD;
    else
        g_features.vendor = CPU_VENDOR_UNKNOWN;

    /* Leaf 1: feature bits + family/model */
    if (max_leaf >= 1) {
        do_cpuid(1, 0, regs);

        /* SSE2: EDX bit 26 */
        g_features.sse2 = (regs[3] & (1 << 26)) != 0;

        /* FMA3: ECX bit 12 */
        g_features.fma3 = (regs[2] & (1 << 12)) != 0;

        /* Family/model for Zen detection */
        int family = (regs[0] >> 8) & 0xF;
        int model  = (regs[0] >> 4) & 0xF;
        int ext_family = (regs[0] >> 20) & 0xFF;
        int ext_model  = (regs[0] >> 16) & 0xF;

        if (family == 0xF)
            family += ext_family;
        model += (ext_model << 4);

        g_features.family = family;
        g_features.model  = model;
    }

    /* Leaf 7, subleaf 0: extended features */
    if (max_leaf >= 7) {
        do_cpuid(7, 0, regs);

        /* AVX2: EBX bit 5 */
        g_features.avx2 = (regs[1] & (1 << 5)) != 0;

        /* AVX-512F: EBX bit 16 */
        g_features.avx512f = (regs[1] & (1 << 16)) != 0;
    }

    /* Verify OS supports AVX state saving (XGETBV) before trusting AVX flags */
    if (g_features.avx2 || g_features.fma3) {
        do_cpuid(1, 0, regs);
        bool osxsave = (regs[2] & (1 << 27)) != 0;

        if (osxsave) {
#ifdef _MSC_VER
            unsigned long long xcr0 = _xgetbv(0);
#else
            unsigned int lo, hi;
            __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
            unsigned long long xcr0 = ((unsigned long long)hi << 32) | lo;
#endif
            bool avx_os = (xcr0 & 0x6) == 0x6; /* XMM + YMM state */
            if (!avx_os) {
                g_features.avx2 = false;
                g_features.fma3 = false;
                g_features.avx512f = false;
            }
        } else {
            g_features.avx2 = false;
            g_features.fma3 = false;
            g_features.avx512f = false;
        }
    }

    g_detected = 1;
    return &g_features;
}

bool cpu_prefer_128bit_avx(void) {
    const cpu_features_t *f = cpu_detect();

    if (f->vendor != CPU_VENDOR_AMD)
        return false;

    /* AMD Family 17h = Zen 1 / Zen+ / Zen 2
     * These crack 256-bit AVX ops into two 128-bit micro-ops.
     * Family 19h (Zen 3/4) and 1Ah (Zen 5) have native 256-bit. */
    return (f->family == 0x17);
}
