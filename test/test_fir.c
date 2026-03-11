/*
 * foo_dsd_trellis — FIR rate conversion tests
 * Phase 4: Polyphase half-band FIR frequency response and round-trip.
 */

#include "test.h"
#include "../include/fir.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── Init tests ─── */

static void test_fir_passthrough_init(void) {
    fir_chain_t chain;
    int ret = fir_chain_init(&chain, DSD_RATE_64, DSD_RATE_64);
    TEST_ASSERT_EQ(ret, 0, "passthrough init should succeed");
    TEST_ASSERT_EQ(chain.num_stages, 0, "passthrough should have 0 stages");
    fir_chain_free(&chain);
}

static void test_fir_2x_upsample_init(void) {
    fir_chain_t chain;
    int ret = fir_chain_init(&chain, DSD_RATE_64, DSD_RATE_128);
    TEST_ASSERT_EQ(ret, 0, "2x upsample init should succeed");
    TEST_ASSERT_EQ(chain.num_stages, 1, "2x should need 1 stage");
    fir_chain_free(&chain);
}

static void test_fir_4x_upsample_init(void) {
    fir_chain_t chain;
    int ret = fir_chain_init(&chain, DSD_RATE_64, DSD_RATE_256);
    TEST_ASSERT_EQ(ret, 0, "4x upsample init should succeed");
    TEST_ASSERT_EQ(chain.num_stages, 2, "4x should need 2 stages");
    fir_chain_free(&chain);
}

static void test_fir_8x_upsample_init(void) {
    fir_chain_t chain;
    int ret = fir_chain_init(&chain, DSD_RATE_64, DSD_RATE_512);
    TEST_ASSERT_EQ(ret, 0, "8x upsample init should succeed");
    TEST_ASSERT_EQ(chain.num_stages, 3, "8x should need 3 stages");
    fir_chain_free(&chain);
}

static void test_fir_2x_downsample_init(void) {
    fir_chain_t chain;
    int ret = fir_chain_init(&chain, DSD_RATE_128, DSD_RATE_64);
    TEST_ASSERT_EQ(ret, 0, "2x downsample init should succeed");
    TEST_ASSERT_EQ(chain.num_stages, 1, "2x down should need 1 stage");
    TEST_ASSERT_FALSE(chain.stages[0].upsample, "should be downsample");
    fir_chain_free(&chain);
}

static void test_fir_invalid_ratio(void) {
    fir_chain_t chain;
    int ret = fir_chain_init(&chain, DSD_RATE_64, DSD_RATE_64 * 3);
    TEST_ASSERT_NEQ(ret, 0, "non-power-of-2 ratio should fail");
}

/* ─── Passthrough test ─── */

static void test_fir_passthrough_process(void) {
    fir_chain_t chain;
    fir_chain_init(&chain, DSD_RATE_64, DSD_RATE_64);

    float in[64], out[64];
    for (int i = 0; i < 64; i++)
        in[i] = (float)(0.5 * sin(2.0 * M_PI * i / 64.0));

    size_t n = fir_chain_process(&chain, in, out, 64);
    TEST_ASSERT_EQ(n, 64u, "passthrough should produce same count");

    int match = 1;
    for (int i = 0; i < 64; i++) {
        if (out[i] != in[i]) { match = 0; break; }
    }
    TEST_ASSERT_TRUE(match, "passthrough output should equal input");

    fir_chain_free(&chain);
}

/* ─── Output count tests ─── */

static void test_fir_2x_upsample_count(void) {
    fir_chain_t chain;
    fir_chain_init(&chain, DSD_RATE_64, DSD_RATE_128);

    float in[128];
    float *out = (float *)malloc(256 * sizeof(float));
    for (int i = 0; i < 128; i++)
        in[i] = 0.0f;

    size_t n = fir_chain_process(&chain, in, out, 128);
    TEST_ASSERT_EQ(n, 256u, "2x upsample should double count");

    free(out);
    fir_chain_free(&chain);
}

static void test_fir_2x_downsample_count(void) {
    fir_chain_t chain;
    fir_chain_init(&chain, DSD_RATE_128, DSD_RATE_64);

    float in[256], out[128];
    for (int i = 0; i < 256; i++)
        in[i] = 0.0f;

    size_t n = fir_chain_process(&chain, in, out, 256);
    TEST_ASSERT_EQ(n, 128u, "2x downsample should halve count");

    fir_chain_free(&chain);
}

static void test_fir_4x_upsample_count(void) {
    fir_chain_t chain;
    fir_chain_init(&chain, DSD_RATE_64, DSD_RATE_256);

    float in[64];
    float *out = (float *)malloc(256 * sizeof(float));
    for (int i = 0; i < 64; i++)
        in[i] = 0.0f;

    size_t n = fir_chain_process(&chain, in, out, 64);
    TEST_ASSERT_EQ(n, 256u, "4x upsample should quadruple count");

    free(out);
    fir_chain_free(&chain);
}

static void test_fir_8x_upsample_count(void) {
    fir_chain_t chain;
    fir_chain_init(&chain, DSD_RATE_64, DSD_RATE_512);

    float in[32];
    float *out = (float *)malloc(256 * sizeof(float));
    for (int i = 0; i < 32; i++)
        in[i] = 0.0f;

    size_t n = fir_chain_process(&chain, in, out, 32);
    TEST_ASSERT_EQ(n, 256u, "8x upsample should 8x count");

    free(out);
    fir_chain_free(&chain);
}

/* ─── Goertzel for frequency response measurement ─── */

static double goertzel_power(const float *x, size_t n, double freq_hz,
                             double sample_rate) {
    double k = freq_hz * (double)n / sample_rate;
    double w = 2.0 * M_PI * k / (double)n;
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

/* ─── Frequency response: 2x upsample ─── */

/*
 * Measure gain at a specific frequency by:
 * 1. Generate sine at freq_hz at input rate
 * 2. Upsample by 2
 * 3. Goertzel at freq_hz at output rate
 * 4. Compare input and output power
 */
static double measure_upsample_gain_db(uint32_t fs_in, uint32_t fs_out,
                                        double freq_hz) {
    fir_chain_t chain;
    fir_chain_init(&chain, fs_in, fs_out);

    unsigned ratio = fs_out / fs_in;
    unsigned n_in = 16384;
    unsigned n_out = n_in * ratio;

    float *in  = (float *)malloc(n_in * sizeof(float));
    float *out = (float *)malloc(n_out * sizeof(float));

    /* Align frequency to DFT bin for both input and output */
    double bin_in  = (double)fs_in / (double)n_in;
    unsigned sig_bin = (unsigned)(freq_hz / bin_in + 0.5);
    double exact_freq = sig_bin * bin_in;

    /* Generate sine at input rate */
    for (unsigned i = 0; i < n_in; i++)
        in[i] = (float)(0.5 * sin(2.0 * M_PI * exact_freq * i / fs_in));

    /* Process */
    size_t produced = fir_chain_process(&chain, in, out, n_in);

    /* Measure power at output */
    double p_in  = goertzel_power(in, n_in, exact_freq, (double)fs_in);
    double p_out = goertzel_power(out, produced, exact_freq, (double)fs_out);

    double gain_db = 10.0 * log10(p_out / (p_in > 0.0 ? p_in : 1e-30));

    free(in);
    free(out);
    fir_chain_free(&chain);

    return gain_db;
}

static void test_fir_passband_flat(void) {
    /* Passband should be flat (±0.1 dB) from 1 kHz to 20 kHz
     * for 2x upsample DSD64→DSD128 */
    double freqs[] = { 1000.0, 5000.0, 10000.0, 15000.0, 20000.0 };
    int n_freqs = 5;
    int all_flat = 1;

    for (int i = 0; i < n_freqs; i++) {
        double gain = measure_upsample_gain_db(DSD_RATE_64, DSD_RATE_128, freqs[i]);
        printf("    [FIR passband] %.0f Hz: %+.3f dB\n", freqs[i], gain);
        if (fabs(gain) > 0.1) {
            all_flat = 0;
        }
    }
    TEST_ASSERT_TRUE(all_flat, "passband should be flat +/-0.1 dB to 20 kHz");
}

static void test_fir_stopband_atten(void) {
    /* Stopband attenuation: >80 dB above Fs_in/2 for DSD64→DSD128.
     * At DSD128 rate, Fs_in/2 = DSD64/2 = 1411200 Hz.
     * Test at several frequencies above the nyquist image region. */
    fir_chain_t chain;
    uint32_t fs_in = DSD_RATE_64;
    uint32_t fs_out = DSD_RATE_128;
    fir_chain_init(&chain, fs_in, fs_out);

    /* Generate a test: input sine at some freq, check alias attenuation.
     * For a 2x upsample, the image of freq_in appears at fs_in - freq_in
     * in the output. So a 20 kHz input creates an image at fs_in - 20000
     * = 2822400 - 20000 = 2802400 Hz at the DSD128 rate.
     * The filter should suppress this image. */
    unsigned n_in = 16384;
    unsigned n_out = n_in * 2;

    float *in  = (float *)malloc(n_in * sizeof(float));
    float *out = (float *)malloc(n_out * sizeof(float));

    double test_freq = 10000.0;  /* Hz at input rate */
    /* Bin-align at input rate */
    double bin_in = (double)fs_in / (double)n_in;
    unsigned sig_bin = (unsigned)(test_freq / bin_in + 0.5);
    double exact_freq = sig_bin * bin_in;
    double exact_image = (double)fs_in - exact_freq;

    for (unsigned i = 0; i < n_in; i++)
        in[i] = (float)(0.5 * sin(2.0 * M_PI * exact_freq * i / fs_in));

    fir_chain_process(&chain, in, out, n_in);

    double p_signal = goertzel_power(out, n_out, exact_freq, (double)fs_out);
    double p_image  = goertzel_power(out, n_out, exact_image, (double)fs_out);

    double atten_db = 10.0 * log10(p_image / (p_signal > 0 ? p_signal : 1e-30));
    printf("    [FIR stopband] image atten at %.0f Hz: %.1f dB\n",
           exact_image, atten_db);

    TEST_ASSERT_TRUE(atten_db < -80.0,
                     "image should be attenuated >80 dB");

    free(in);
    free(out);
    fir_chain_free(&chain);
}

/* ─── Upsample then downsample round-trip ─── */

static void test_fir_roundtrip_2x(void) {
    /* DSD64 → DSD128 → DSD64 should preserve audio band */
    fir_chain_t up, down;
    fir_chain_init(&up, DSD_RATE_64, DSD_RATE_128);
    fir_chain_init(&down, DSD_RATE_128, DSD_RATE_64);

    unsigned n = 4096;
    float *in   = (float *)malloc(n * sizeof(float));
    float *mid  = (float *)malloc(n * 2 * sizeof(float));
    float *out  = (float *)malloc(n * sizeof(float));

    /* 1 kHz sine at DSD64 rate, bin-aligned */
    double bin_w = (double)DSD_RATE_64 / (double)n;
    unsigned sig_bin = (unsigned)(1000.0 / bin_w + 0.5);
    double freq = sig_bin * bin_w;

    for (unsigned i = 0; i < n; i++)
        in[i] = (float)(0.5 * sin(2.0 * M_PI * freq * i / DSD_RATE_64));

    size_t n_up = fir_chain_process(&up, in, mid, n);
    TEST_ASSERT_EQ(n_up, n * 2, "upsample should double count");

    size_t n_down = fir_chain_process(&down, mid, out, n_up);
    TEST_ASSERT_EQ(n_down, (size_t)n, "downsample should restore count");

    /* Measure signal power in roundtrip output vs input.
     * Expect near-unity gain at 1kHz (within ±0.1 dB). */
    double p_in  = goertzel_power(in, n, freq, (double)DSD_RATE_64);
    double p_out = goertzel_power(out, n_down, freq, (double)DSD_RATE_64);

    /* Account for group delay: roundtrip has filter delay, so skip
     * initial transient. Use a generous signal window. */
    double gain_db = 10.0 * log10(p_out / (p_in > 0 ? p_in : 1e-30));
    printf("    [FIR roundtrip] 2x gain at %.0f Hz: %+.3f dB\n", freq, gain_db);

    TEST_ASSERT_TRUE(fabs(gain_db) < 0.2,
                     "round-trip gain should be near unity (+/-0.2 dB)");

    free(in);
    free(mid);
    free(out);
    fir_chain_free(&up);
    fir_chain_free(&down);
}

/* ─── 4x and 8x chain tests ─── */

static void test_fir_4x_passband(void) {
    double gain = measure_upsample_gain_db(DSD_RATE_64, DSD_RATE_256, 10000.0);
    printf("    [FIR 4x] 10 kHz gain: %+.3f dB\n", gain);
    TEST_ASSERT_TRUE(fabs(gain) < 0.2,
                     "4x upsample passband should be flat at 10 kHz");
}

static void test_fir_8x_passband(void) {
    double gain = measure_upsample_gain_db(DSD_RATE_64, DSD_RATE_512, 10000.0);
    printf("    [FIR 8x] 10 kHz gain: %+.3f dB\n", gain);
    TEST_ASSERT_TRUE(fabs(gain) < 0.3,
                     "8x upsample passband should be flat at 10 kHz");
}

/* ─── Reset test ─── */

static void test_fir_reset(void) {
    fir_chain_t chain;
    fir_chain_init(&chain, DSD_RATE_64, DSD_RATE_128);

    /* Process some data to dirty the state */
    float in[128];
    float *out = (float *)malloc(256 * sizeof(float));
    for (int i = 0; i < 128; i++)
        in[i] = 1.0f;
    fir_chain_process(&chain, in, out, 128);

    /* Reset */
    fir_chain_reset(&chain);

    /* After reset, processing silence should produce near-silence output */
    for (int i = 0; i < 128; i++)
        in[i] = 0.0f;
    fir_chain_process(&chain, in, out, 128);

    /* The output should be very small after transient settles */
    double max_val = 0.0;
    for (int i = 100; i < 256; i++) {  /* skip initial transient */
        double v = fabs((double)out[i]);
        if (v > max_val) max_val = v;
    }
    TEST_ASSERT_TRUE(max_val < 0.001,
                     "after reset, silence input should produce near-silence");

    free(out);
    fir_chain_free(&chain);
}

void test_fir_suite(void) {
    TEST_SUITE("FIR");
    TEST_RUN(test_fir_passthrough_init);
    TEST_RUN(test_fir_2x_upsample_init);
    TEST_RUN(test_fir_4x_upsample_init);
    TEST_RUN(test_fir_8x_upsample_init);
    TEST_RUN(test_fir_2x_downsample_init);
    TEST_RUN(test_fir_invalid_ratio);
    TEST_RUN(test_fir_passthrough_process);
    TEST_RUN(test_fir_2x_upsample_count);
    TEST_RUN(test_fir_2x_downsample_count);
    TEST_RUN(test_fir_4x_upsample_count);
    TEST_RUN(test_fir_8x_upsample_count);
    TEST_RUN(test_fir_passband_flat);
    TEST_RUN(test_fir_stopband_atten);
    TEST_RUN(test_fir_roundtrip_2x);
    TEST_RUN(test_fir_4x_passband);
    TEST_RUN(test_fir_8x_passband);
    TEST_RUN(test_fir_reset);
}
