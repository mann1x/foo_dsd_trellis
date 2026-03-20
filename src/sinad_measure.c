/*
 * foo_dsd_trellis — SINAD measurement (shared between test and UI)
 *
 * Full pipeline SINAD: measures the actual roundtrip quality.
 *   1. Generate ~1kHz sine, quantize to DSD ±1.0 (simulates real DSD input)
 *   2. Boxcar or FIR lowpass → multi-bit smoothed signal
 *   3. Apply gain
 *   4. Trellis SDM → DSD ±1.0 output
 *   5. Decode output: boxcar lowpass → analog (simulates DAC)
 *   6. Goertzel on decoded analog → SINAD
 *
 * This measures what the user actually hears: DSD → re-encode → DSD → decode.
 */

#include "../include/sinad_measure.h"
#include "../include/trellis.h"
#include "../include/ntf.h"
#include "../include/fir.h"
#include "../include/dsd_types.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Goertzel single-bin power measurement */
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

void sinad_measure(uint32_t dsd_rate, int ntf_id,
                   int cands, int depth, int lat,
                   int use_fir_lowpass, float fir_gain,
                   sinad_result_t *result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    result->sinad_db = -999.0;

    unsigned rate_mult = dsd_rate / 44100;
    const unsigned n_dsd = (rate_mult <= 64)  ? 262144u :
                           (rate_mult <= 128) ? 524288u :
                           (rate_mult <= 256) ? 1048576u : 2097152u;

    /* Bin-align frequency to avoid spectral leakage */
    const unsigned produced_est = n_dsd - (unsigned)lat;
    double bin_width = (double)dsd_rate / (double)produced_est;
    unsigned sig_bin = (unsigned)(1000.0 / bin_width + 0.5);
    double freq = sig_bin * bin_width;

    /* Look up NTF */
    const ntf_filter_t *f = ntf_get_filter((ntf_filter_id_t)ntf_id, dsd_rate);
    if (!f) f = ntf_auto_select(dsd_rate);
    if (!f) return;

    /* Init SDM */
    sdm_context_t ctx;
    if (sdm_context_init(&ctx, f, (unsigned)depth, (unsigned)cands, (unsigned)lat) != 0)
        return;

    /* Allocate buffers */
    float  *dsd_in = (float  *)malloc(n_dsd * sizeof(float));
    double *smooth = (double *)malloc(n_dsd * sizeof(double));
    float  *out    = (float  *)malloc(n_dsd * sizeof(float));
    if (!dsd_in || !smooth || !out) {
        free(dsd_in); free(smooth); free(out);
        sdm_context_free(&ctx);
        return;
    }

    /* Step 1: Generate DSD input — quantize sine to ±1.0 (simulates real DSD) */
    for (unsigned i = 0; i < n_dsd; i++) {
        double s = 0.5 * sin(2.0 * M_PI * freq * (double)i / (double)dsd_rate);
        dsd_in[i] = (s >= 0.0) ? 1.0f : -1.0f;
    }

    /* Step 2+3: Pre-SDM filter + gain (same as engine pipeline) */
    double gain = (double)fir_gain;
    if (gain <= 0.0) gain = 0.708;

    if (use_fir_lowpass) {
        fir_lowpass_t lp;
        fir_lowpass_init(&lp, dsd_rate);
        float *lp_in  = (float *)malloc(n_dsd * sizeof(float));
        float *lp_out = (float *)malloc(n_dsd * sizeof(float));
        if (lp_in && lp_out) {
            memcpy(lp_in, dsd_in, n_dsd * sizeof(float));
            fir_lowpass_process(&lp, lp_in, lp_out, n_dsd);
            for (unsigned i = 0; i < n_dsd; i++)
                smooth[i] = (double)lp_out[i] * gain;
        }
        free(lp_in); free(lp_out);
        fir_lowpass_free(&lp);
    } else {
        int taps = (dsd_rate >= DSD_RATE_512) ? 128 :
                   (dsd_rate >= DSD_RATE_128) ? 64 : 32;
        double sum = 0.0;
        double ring[128] = {0};
        int pos = 0;
        double inv_n = 1.0 / (double)taps;
        for (unsigned i = 0; i < n_dsd; i++) {
            double s = dsd_in[i] >= 0.0f ? 1.0 : -1.0;
            sum -= ring[pos];
            ring[pos] = s;
            sum += s;
            pos = (pos + 1) % taps;
            smooth[i] = sum * inv_n * gain;
        }
    }

    /* Step 4: Process through SDM */
    size_t produced = sdm_process_block(&ctx, smooth, out, n_dsd);

    if (produced < 1024) {
        free(dsd_in); free(smooth); free(out);
        sdm_context_free(&ctx);
        return;
    }

    /* Step 5: Pipeline SINAD — same Goertzel method as theoretical, but
     * on the re-encoded output. Measures in-band audio quality of the
     * DSD→boxcar/FIR→SDM→DSD roundtrip. The difference between this
     * and the theoretical SINAD shows the re-encode degradation. */
    {
        double actual_bw = (double)dsd_rate / (double)produced;
        unsigned actual_sig_bin = (unsigned)(freq / actual_bw + 0.5);
        double signal_power = goertzel_power(out, produced, freq, (double)dsd_rate);

        unsigned max_bin = (unsigned)(20000.0 / actual_bw);
        double noise = 0.0;
        for (unsigned b = 1; b <= max_bin; b++) {
            if (b >= actual_sig_bin - 1 && b <= actual_sig_bin + 1) continue;
            noise += goertzel_power(out, produced, b * actual_bw, (double)dsd_rate);
        }
        if (noise <= 0.0) noise = 1e-30;

        result->sinad_db = 10.0 * log10(signal_power / noise);
    }
    result->conv_fail = ctx.conv_fail;
    result->cands_collapse = ctx.cands_collapse;
    result->drop_pct = ctx.total_children > 0
        ? 100.0 * (double)ctx.next_filter_drops / (double)ctx.total_children
        : 0.0;
    result->ok = 1;

    sdm_context_free(&ctx);

    /* ── Theoretical SINAD: clean analog sine → SDM (best-case) ── */
    {
        sdm_context_t ctx2;
        if (sdm_context_init(&ctx2, f, (unsigned)depth, (unsigned)cands, (unsigned)lat) == 0) {
            /* Feed clean analog sine directly (no DSD quantization, no boxcar) */
            for (unsigned i = 0; i < n_dsd; i++)
                smooth[i] = 0.5 * sin(2.0 * M_PI * freq * (double)i / (double)dsd_rate);

            size_t produced2 = sdm_process_block(&ctx2, smooth, out, n_dsd);

            if (produced2 >= 1024) {
                double actual_bw2 = (double)dsd_rate / (double)produced2;
                unsigned sig_bin2 = (unsigned)(freq / actual_bw2 + 0.5);
                double sig2 = goertzel_power(out, produced2, freq, (double)dsd_rate);
                unsigned max_bin2 = (unsigned)(20000.0 / actual_bw2);
                double noise2 = 0.0;
                for (unsigned b = 1; b <= max_bin2; b++) {
                    if (b >= sig_bin2 - 1 && b <= sig_bin2 + 1) continue;
                    noise2 += goertzel_power(out, produced2, b * actual_bw2, (double)dsd_rate);
                }
                if (noise2 <= 0.0) noise2 = 1e-30;
                result->sinad_theoretical = 10.0 * log10(sig2 / noise2);
            }
            sdm_context_free(&ctx2);
        }
    }

    free(dsd_in); free(smooth); free(out);
}
