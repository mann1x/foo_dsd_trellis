/*
 * foo_dsd_trellis — Trellis SDM tests
 * Phase 3: Full trellis algorithm with SINAD measurement.
 */

#include "test.h"
#include "../include/trellis.h"
#include "../include/ntf.h"
#include "../include/fir.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── Init / lifecycle tests ─── */

static void test_sdm_context_init(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_64);
    TEST_ASSERT_NOT_NULL(f, "auto-select should return a filter");

    sdm_context_t ctx;
    int ret = sdm_context_init(&ctx, f, 8, 16, 64);
    TEST_ASSERT_EQ(ret, 0, "sdm_context_init should succeed");
    TEST_ASSERT_EQ(ctx.num_cands, 1u, "initial candidate count should be 1");
    TEST_ASSERT_EQ(ctx.trellis_num, 16u, "trellis_num should be 16");
    TEST_ASSERT_EQ(ctx.trellis_lat, 64u, "trellis_lat should be 64");

    sdm_context_free(&ctx);
}

static void test_sdm_null_filter(void) {
    sdm_context_t ctx;
    int ret = sdm_context_init(&ctx, NULL, 8, 16, 64);
    TEST_ASSERT_NEQ(ret, 0, "null filter should fail init");
}

static void test_sdm_invalid_params(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_64);
    sdm_context_t ctx;

    TEST_ASSERT_NEQ(sdm_context_init(&ctx, f, SDM_TRELLIS_MAX_ORDER + 1, 16, 64), 0,
                    "excessive depth should fail");
    TEST_ASSERT_NEQ(sdm_context_init(&ctx, f, 8, SDM_TRELLIS_MAX_NUM + 1, 64), 0,
                    "excessive cands should fail");
    TEST_ASSERT_NEQ(sdm_context_init(&ctx, f, 8, 16, SDM_TRELLIS_MAX_LAT + 1), 0,
                    "excessive latency should fail");
    TEST_ASSERT_NEQ(sdm_context_init(&ctx, f, 0, 16, 64), 0,
                    "zero depth should fail");
}

static void test_sdm_reset(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_64);
    sdm_context_t ctx;
    sdm_context_init(&ctx, f, 8, 16, 64);

    /* Process some samples to dirty the state */
    double in[32]; float out[32];
    for (int i = 0; i < 32; i++)
        in[i] = sin(2.0 * M_PI * i / 32.0) * 0.5;
    sdm_process_block(&ctx, in, out, 32);

    /* Reset and verify clean state */
    sdm_context_reset(&ctx);
    TEST_ASSERT_EQ(ctx.num_cands, 1u, "after reset, num_cands should be 1");
    TEST_ASSERT_EQ(ctx.pos, 0u, "after reset, pos should be 0");
    TEST_ASSERT_EQ(ctx.pending, 0u, "after reset, pending should be 0");
    TEST_ASSERT_EQ(ctx.draining, 0u, "after reset, draining should be 0");
    TEST_ASSERT_EQ(ctx.conv_fail, 0ull, "after reset, conv_fail should be 0");

    sdm_context_free(&ctx);
}

/* ─── Output validity tests ─── */

static void test_sdm_output_binary(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_64);
    sdm_context_t ctx;
    sdm_context_init(&ctx, f, 8, 16, 64);

    const unsigned dsd_rate = 64 * 44100;
    const unsigned n = 256;
    double in[256]; float out[256];
    for (unsigned i = 0; i < n; i++)
        in[i] = 0.5 * sin(2.0 * M_PI * 1000.0 * i / dsd_rate);

    size_t produced = sdm_process_block(&ctx, in, out, n);

    int all_binary = 1;
    for (size_t i = 0; i < produced; i++) {
        if (out[i] != 1.0f && out[i] != -1.0f) {
            all_binary = 0;
            break;
        }
    }
    TEST_ASSERT_TRUE(all_binary, "all output samples should be +/-1.0");

    sdm_context_free(&ctx);
}

static void test_sdm_latency(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_64);
    sdm_context_t ctx;
    unsigned lat = 64;
    sdm_context_init(&ctx, f, 8, 16, lat);

    double in[128]; float out[128];
    for (int i = 0; i < 128; i++)
        in[i] = 0.5;

    size_t produced = sdm_process_block(&ctx, in, out, 128);
    TEST_ASSERT_EQ(produced, (size_t)(128 - lat),
                   "output should be input count minus latency");

    sdm_context_free(&ctx);
}

static void test_sdm_drain_basic(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_64);
    sdm_context_t ctx;
    unsigned lat = 64;
    sdm_context_init(&ctx, f, 8, 16, lat);

    double in[64]; float out[256];
    for (int i = 0; i < 64; i++)
        in[i] = 0.0;
    size_t produced = sdm_process_block(&ctx, in, out, 64);
    TEST_ASSERT_EQ(produced, 0u, "no output during latency fill");

    size_t drained = sdm_drain(&ctx, out, 256);
    TEST_ASSERT_EQ(drained, (size_t)lat, "drain should produce latency samples");

    int all_binary = 1;
    for (size_t i = 0; i < drained; i++) {
        if (out[i] != 1.0f && out[i] != -1.0f) {
            all_binary = 0;
            break;
        }
    }
    TEST_ASSERT_TRUE(all_binary, "drained output should be +/-1.0");

    sdm_context_free(&ctx);
}

/* ─── SINAD measurement via Goertzel on DSD stream ─── */

/*
 * Compute |X(f)|^2 for a specific frequency using Goertzel algorithm.
 * Operates directly on the +/-1.0 DSD stream (no decimation needed).
 * Returns power spectral density at the given frequency.
 */
static double goertzel_power(const float *x, size_t n, double freq_hz,
                             double sample_rate) {
    double k = freq_hz * n / sample_rate;
    double w = 2.0 * M_PI * k / n;
    double coeff = 2.0 * cos(w);
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;

    for (size_t i = 0; i < n; i++) {
        s0 = (double)x[i] + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }

    double real = s1 - s2 * cos(w);
    double imag = s2 * sin(w);
    return (real * real + imag * imag) / ((double)n * (double)n);
}

/*
 * Measure SINAD of a 1-bit DSD stream:
 *  1. Generate 1kHz sine, encode through trellis SDM
 *  2. Goertzel at 1kHz on DSD output -> signal power
 *  3. Goertzel at many freqs in 0-20kHz -> total in-band power
 *  4. SINAD = signal / (total_in_band - signal)
 */
static const ntf_filter_t *tls_ntf_override = NULL;
static double measure_sinad_1khz(unsigned dsd_rate_mult, int trellis_depth,
                                 int trellis_cands, int trellis_lat) {
    const unsigned base_rate = 44100;
    const unsigned dsd_rate = dsd_rate_mult * base_rate;

    /* Choose N so the analysis window is large enough, then pick the
     * test frequency to land exactly on a DFT bin to avoid spectral
     * leakage. produced = n_dsd - trellis_lat. */
    const unsigned n_dsd = (dsd_rate_mult <= 64)  ? 262144u :
                           (dsd_rate_mult <= 128) ? 524288u :
                           (dsd_rate_mult <= 256) ? 1048576u : 2097152u;
    const unsigned produced_est = n_dsd - (unsigned)trellis_lat;
    double bin_width = (double)dsd_rate / (double)produced_est;
    unsigned sig_bin = (unsigned)(1000.0 / bin_width + 0.5);
    double freq = sig_bin * bin_width;  /* exact DFT bin frequency (~1kHz) */

    const ntf_filter_t *f = ntf_auto_select(dsd_rate);
    if (!f) return -999.0;
    /* Allow override via thread-local for sweep */
    if (tls_ntf_override) f = tls_ntf_override;

    sdm_context_t ctx;
    if (sdm_context_init(&ctx, f, trellis_depth, trellis_cands,
                         trellis_lat) != 0)
        return -999.0;

    double *in = (double *)malloc(n_dsd * sizeof(double));
    float *out  = (float *)malloc(n_dsd * sizeof(float));
    if (!in || !out) {
        free(in); free(out);
        sdm_context_free(&ctx);
        return -999.0;
    }

    /* Generate sine at bin-aligned frequency, amplitude 0.5 */
    for (unsigned i = 0; i < n_dsd; i++)
        in[i] = 0.5 * sin(2.0 * M_PI * freq * i / dsd_rate);

    /* Process through SDM */
    size_t produced = sdm_process_block(&ctx, in, out, n_dsd);

    /* Drain remaining */
    float *drain_buf = (float *)malloc((size_t)trellis_lat * sizeof(float));
    size_t drained = 0;
    if (drain_buf)
        drained = sdm_drain(&ctx, drain_buf, (size_t)trellis_lat);

    if (produced < 1024) {
        free(in); free(out); free(drain_buf);
        sdm_context_free(&ctx);
        return -999.0;
    }

    /* Measure signal power at exactly 1kHz */
    double signal_power = goertzel_power(out, produced, freq, (double)dsd_rate);

    /* Measure total in-band power (0 to 22050 Hz) by summing Goertzel
     * at every DFT bin. Bin spacing = dsd_rate / produced.
     * For DSD64 with 262k samples: ~10.8 Hz/bin, ~2048 bins to 22050 Hz. */
    double actual_bw = (double)dsd_rate / (double)produced;
    unsigned max_bin = (unsigned)(22050.0 / actual_bw);
    double total_inband = 0.0;

    /* Recompute signal bin for actual produced count */
    unsigned actual_sig_bin = (unsigned)(freq / actual_bw + 0.5);

    for (unsigned b = 1; b <= max_bin; b++) {
        /* Skip the signal bin and its immediate neighbors (leakage) */
        if (b >= actual_sig_bin - 1 && b <= actual_sig_bin + 1)
            continue;
        double fq = b * actual_bw;
        total_inband += goertzel_power(out, produced, fq, (double)dsd_rate);
    }

    double noise_power = total_inband;
    if (noise_power <= 0.0) noise_power = 1e-30;

    double sinad_db = 10.0 * log10(signal_power / noise_power);

    printf("    [SINAD] DSD%u: %zu samples, "
           "signal=%.2e noise=%.2e SINAD=%.1f dB "
           "(conv_fail=%llu collapse=%llu drops=%.1f%%, %u bins)\n",
           dsd_rate_mult, produced, signal_power, noise_power, sinad_db,
           (unsigned long long)ctx.conv_fail,
           (unsigned long long)ctx.cands_collapse,
           ctx.total_children > 0 ? 100.0 * ctx.next_filter_drops / ctx.total_children : 0.0,
           max_bin);

    (void)drained;
    free(in);
    free(out);
    free(drain_buf);
    sdm_context_free(&ctx);

    return sinad_db;
}

static void test_sdm_sinad_dsd64(void) {
    /* Use default latency (64) matching runtime config.
     * Goertzel-based measurement directly on DSD avoids decimation
     * artifacts. Expect >80 dB for a well-functioning trellis. */
    double sinad = measure_sinad_1khz(64, 8, 16, DSD_DEFAULT_TRELLIS_LAT);
    TEST_ASSERT_TRUE(sinad > 60.0,
                     "DSD64 trellis SINAD should exceed 60 dB for 1kHz sine");
}

static void test_sdm_sinad_dsd128(void) {
    double sinad = measure_sinad_1khz(128, 8, 16, DSD_DEFAULT_TRELLIS_LAT);
    TEST_ASSERT_TRUE(sinad > 60.0,
                     "DSD128 trellis SINAD should exceed 60 dB for 1kHz sine");
}

static void test_sdm_sinad_dsd256(void) {
    double sinad = measure_sinad_1khz(256, 8, 16, DSD_DEFAULT_TRELLIS_LAT);
    TEST_ASSERT_TRUE(sinad > 60.0,
                     "DSD256 trellis SINAD should exceed 60 dB for 1kHz sine");
}

static void test_sdm_sinad_dsd512(void) {
    double sinad = measure_sinad_1khz(512, 8, 16, DSD_DEFAULT_TRELLIS_LAT);
    TEST_ASSERT_TRUE(sinad > 60.0,
                     "DSD512 trellis SINAD should exceed 60 dB for 1kHz sine");
}

static void test_sdm_dc_stability(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_64);
    sdm_context_t ctx;
    sdm_context_init(&ctx, f, 8, 16, 128);

    double in[1024]; float out[1024];
    memset(in, 0, sizeof(in));

    size_t produced = sdm_process_block(&ctx, in, out, 1024);

    int valid = 1;
    for (size_t i = 0; i < produced; i++) {
        if (out[i] != 1.0f && out[i] != -1.0f) {
            valid = 0;
            break;
        }
    }
    TEST_ASSERT_TRUE(valid, "DC=0 input should produce valid +/-1.0 output");

    if (produced > 0) {
        double avg = 0.0;
        for (size_t i = 0; i < produced; i++)
            avg += out[i];
        avg /= (double)produced;
        TEST_ASSERT_TRUE(fabs(avg) < 0.1,
                         "DC=0 average output should be near zero");
    }

    sdm_context_free(&ctx);
}

static void test_sdm_conv_fail_low(void) {
    const ntf_filter_t *f = ntf_auto_select(DSD_RATE_64);
    sdm_context_t ctx;
    sdm_context_init(&ctx, f, 8, 16, 256);

    unsigned n = 4096;
    double *in = (double *)malloc(n * sizeof(double));
    float *out = (float *)malloc(n * sizeof(float));

    for (unsigned i = 0; i < n; i++)
        in[i] = 0.3 * sin(2.0 * M_PI * 1000.0 * i / (64 * 44100));

    sdm_process_block(&ctx, in, out, n);

    double fail_rate = (double)ctx.conv_fail / n;
    printf("    [conv_fail] rate = %.4f%% (%llu / %u)\n",
           fail_rate * 100.0, (unsigned long long)ctx.conv_fail, n);
    TEST_ASSERT_TRUE(fail_rate < 0.01,
                     "convergence failure rate should be under 1%%");

    free(in);
    free(out);
    sdm_context_free(&ctx);
}

void test_trellis_suite(void) {
    TEST_SUITE("Trellis SDM");
    TEST_RUN(test_sdm_context_init);
    TEST_RUN(test_sdm_null_filter);
    TEST_RUN(test_sdm_invalid_params);
    TEST_RUN(test_sdm_reset);
    TEST_RUN(test_sdm_output_binary);
    TEST_RUN(test_sdm_latency);
    TEST_RUN(test_sdm_drain_basic);
    TEST_RUN(test_sdm_dc_stability);
    TEST_RUN(test_sdm_conv_fail_low);
    TEST_RUN(test_sdm_sinad_dsd64);
    TEST_RUN(test_sdm_sinad_dsd128);
    TEST_RUN(test_sdm_sinad_dsd256);
    TEST_RUN(test_sdm_sinad_dsd512);
}

/* ─── Pipeline SINAD: simulate actual engine path ─── */

static double measure_pipeline_sinad(unsigned rate_mult, int nc, int lat,
                                      int use_fir_lowpass) {
    const unsigned dsd_rate = rate_mult * 44100;
    /* Use same sample counts as the SINAD tests for fast Goertzel */
    const unsigned n_dsd = (rate_mult <= 64)  ? 262144u :
                           (rate_mult <= 128) ? 524288u :
                           (rate_mult <= 256) ? 1048576u : 2097152u;

    const ntf_filter_t *f = ntf_auto_select(dsd_rate);
    if (tls_ntf_override) f = tls_ntf_override;
    if (!f) return -999.0;

    sdm_context_t ctx;
    if (sdm_context_init(&ctx, f, 8, nc, lat) != 0)
        return -999.0;

    /* Generate 1kHz DSD signal: encode sine through a reference SDM first */
    double freq = 1000.0;
    float *dsd_in = (float *)malloc(n_dsd * sizeof(float));
    double *smooth = (double *)malloc(n_dsd * sizeof(double));
    float *out = (float *)malloc(n_dsd * sizeof(float));
    if (!dsd_in || !smooth || !out) {
        free(dsd_in); free(smooth); free(out);
        sdm_context_free(&ctx); return -999.0;
    }

    /* Create DSD input: ±1.0 from sine (hard quantize) */
    for (unsigned i = 0; i < n_dsd; i++) {
        double s = 0.5 * sin(2.0 * M_PI * freq * i / dsd_rate);
        dsd_in[i] = (s >= 0.0) ? 1.0f : -1.0f;
    }

    /* Simulate pipeline: boxcar or FIR lowpass → gain → SDM */
    if (use_fir_lowpass) {
        /* FIR lowpass path (IPP float, widen to double) */
        fir_lowpass_t lp;
        fir_lowpass_init(&lp, dsd_rate);
        float *lp_in = (float *)malloc(n_dsd * sizeof(float));
        float *lp_out_f = (float *)malloc(n_dsd * sizeof(float));
        for (unsigned i = 0; i < n_dsd; i++)
            lp_in[i] = dsd_in[i];
        fir_lowpass_process(&lp, lp_in, lp_out_f, n_dsd);
        double gain = 0.708;
        for (unsigned i = 0; i < n_dsd; i++)
            smooth[i] = (double)lp_out_f[i] * gain;
        free(lp_in); free(lp_out_f);
        fir_lowpass_free(&lp);
    } else {
        /* Boxcar path */
        int taps = (dsd_rate >= DSD_RATE_512) ? 128 :
                   (dsd_rate >= DSD_RATE_128) ? 64 : 32;
        double sum = 0.0;
        double ring[128] = {0};
        int pos = 0;
        double inv_n = 1.0 / (double)taps;
        double gain = 0.708;
        for (unsigned i = 0; i < n_dsd; i++) {
            double s = dsd_in[i] >= 0.0f ? 1.0 : -1.0;
            sum -= ring[pos];
            ring[pos] = s;
            sum += s;
            pos = (pos + 1) % taps;
            smooth[i] = sum * inv_n * gain;
        }
    }

    /* Process through SDM */
    size_t produced = sdm_process_block(&ctx, smooth, out, n_dsd);

    /* Measure SINAD */
    if (produced < 1024) {
        free(dsd_in); free(smooth); free(out);
        sdm_context_free(&ctx); return -999.0;
    }
    double actual_bw = (double)dsd_rate / (double)produced;
    unsigned sig_bin = (unsigned)(freq / actual_bw + 0.5);
    double signal_power = goertzel_power(out, produced, freq, (double)dsd_rate);
    unsigned max_bin = (unsigned)(22050.0 / actual_bw);
    double noise = 0.0;
    for (unsigned b = 1; b <= max_bin; b++) {
        if (b >= sig_bin - 1 && b <= sig_bin + 1) continue;
        noise += goertzel_power(out, produced, b * actual_bw, (double)dsd_rate);
    }
    if (noise <= 0.0) noise = 1e-30;
    double sinad = 10.0 * log10(signal_power / noise);

    printf("  DSD%u %s nc=%d lat=%d: SINAD=%.1f dB "
           "(conv_fail=%llu collapse=%llu drops=%.1f%%)\n",
           rate_mult, use_fir_lowpass ? "FIR" : "BOX", nc, lat, sinad,
           (unsigned long long)ctx.conv_fail,
           (unsigned long long)ctx.cands_collapse,
           ctx.total_children > 0 ? 100.0 * ctx.next_filter_drops / ctx.total_children : 0.0);

    free(dsd_in); free(smooth); free(out);
    sdm_context_free(&ctx);
    return sinad;
}

void test_pipeline_sinad(void) {
    printf("\n=== Pipeline SINAD (boxcar/FIR → gain → SDM, 2s signal) ===\n");

    /* DSD64 — baseline */
    measure_pipeline_sinad(64, 2, 32, 0);   /* boxcar */
    measure_pipeline_sinad(64, 2, 32, 1);   /* FIR */
    /* DSD128 — compare */
    measure_pipeline_sinad(128, 2, 128, 0);  /* boxcar */
    measure_pipeline_sinad(128, 2, 128, 1);  /* FIR */
    /* DSD128 with DSD64's NTF for isolation */
    tls_ntf_override = ntf_get_filter(NTF_CLANS_6, DSD_RATE_128);
    measure_pipeline_sinad(128, 2, 128, 0);
    measure_pipeline_sinad(128, 2, 128, 1);
    tls_ntf_override = NULL;
    /* DSD128 at lat=32 (DSD64's latency) */
    measure_pipeline_sinad(128, 2, 32, 0);
    measure_pipeline_sinad(128, 2, 32, 1);

    g_tests_run++; g_tests_passed++;
}

/* NTF sweep at optimal nc/lat for specific rates */
void test_lat_sweep(void) {
    /* NTF IDs: 0=CLANS4, 1=SDM4, 2=CLANS5, 3=SDM5, 4=CLANS6, 5=SDM6,
     *          6=CLANS7, 7=SDM7, 8=CLANS8, 9=SDM8 */
    const char *ntf_names[] = {
        "CLANS4", "SDM4", "CLANS5", "SDM5", "CLANS6",
        "SDM6", "CLANS7", "SDM7", "CLANS8", "SDM8"
    };

    /* Sweep NTFs across rates, nc, and lat.
     * Includes stability-relevant configs (nc=2 vs nc=4, various lat). */
    struct { int rate; int nc; int lat; } configs[] = {
        /* DSD64 */
        {64, 2, 32},  {64, 4, 32},  {64, 2, 128}, {64, 4, 128},
        /* DSD128 */
        {128, 2, 32}, {128, 4, 32}, {128, 2, 64}, {128, 4, 64},
        {128, 2, 128},{128, 4, 128},
        /* DSD256 */
        {256, 2, 32}, {256, 2, 128},
        /* DSD512 */
        {512, 2, 16}, {512, 2, 32}, {512, 2, 128},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    for (int ci = 0; ci < n_configs; ci++) {
        int rate = configs[ci].rate;
        int nc = configs[ci].nc;
        int lat = configs[ci].lat;
        unsigned dsd_rate = (unsigned)rate * 44100;

        printf("\n=== DSD%d NTF sweep (nc=%d, lat=%d, depth=8) ===\n", rate, nc, lat);
        for (int ntf_id = 0; ntf_id < 10; ntf_id++) {
            const ntf_filter_t *f = ntf_get_filter((ntf_filter_id_t)ntf_id, dsd_rate);
            if (!f) { printf("  %-6s: N/A\n", ntf_names[ntf_id]); continue; }
            tls_ntf_override = f;
            double sinad = measure_sinad_1khz((unsigned)rate, 8, nc, lat);
            printf("  => %5.1f dB\n", sinad);
            tls_ntf_override = NULL;
        }
    }
    g_tests_run++; g_tests_passed++;
}
