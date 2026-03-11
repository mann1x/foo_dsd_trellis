/*
 * foo_dsd_trellis — foobar2000 DSP v2 interface
 *
 * This file implements the fb2k DSP plugin entry point, chunk list
 * processing, and service registration. It bridges foobar2000's audio
 * pipeline to the DSD processing engine.
 *
 * Phase 0: Scaffold — plugin loads and appears in DSP chain.
 * Processing is a no-op passthrough until Phase 6.
 */

#include <windows.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "../include/dsd_types.h"
#include "../include/engine.h"
#include "../include/dop.h"
#include "../include/threadpool.h"

/*
 * Plugin identity
 */
#define PLUGIN_NAME        "DSD Trellis SDM"
#define PLUGIN_VERSION     "0.1.0"
#define PLUGIN_DESCRIPTION "DSD Trellis (Viterbi) Sigma-Delta Modulator — " \
                           "rate conversion, volume control, and noise shaping " \
                           "for DSD audio streams."

/* Plugin GUID: {7A3F2D1E-B4C5-4E6F-8A9B-0C1D2E3F4A5B} */
static const uint8_t PLUGIN_GUID[16] = {
    0x7A, 0x3F, 0x2D, 0x1E, 0xB4, 0xC5, 0x4E, 0x6F,
    0x8A, 0x9B, 0x0C, 0x1D, 0x2E, 0x3F, 0x4A, 0x5B
};

/*
 * Plugin state (will be populated in later phases)
 */
typedef struct {
    dsd_config_t       config;
    engine_channel_t  *channels;
    int                num_channels;
    threadpool_t      *pool;
    bool               initialized;
} plugin_state_t;

static plugin_state_t g_state = {0};

/*
 * DLL entry point
 */
BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID lpReserved) {
    (void)hModule;
    (void)lpReserved;

    switch (reason) {
    case DLL_PROCESS_ATTACH:
        dsd_config_defaults(&g_state.config);
        break;
    case DLL_PROCESS_DETACH:
        if (g_state.pool) {
            threadpool_destroy(g_state.pool);
            g_state.pool = NULL;
        }
        if (g_state.channels) {
            for (int i = 0; i < g_state.num_channels; i++)
                engine_channel_free(&g_state.channels[i]);
            free(g_state.channels);
            g_state.channels = NULL;
        }
        break;
    }
    return TRUE;
}

/*
 * TODO (Phase 6): Implement fb2k DSP v2 service registration.
 *
 * The foobar2000 SDK is C++ and uses COM-like service interfaces.
 * The actual fb2k integration will require a thin C++ wrapper file
 * (dsp_fb2k.cpp) that:
 *
 * 1. Subclasses dsp_impl_base_t<dsp_dsd_trellis>
 * 2. Implements on_chunk(), flush(), get_latency(), etc.
 * 3. Calls into the C engine via the functions in this file
 * 4. Registers via static service_factory_single_t
 *
 * For now, this file provides the C-side plugin state management
 * that the C++ wrapper will call into.
 */

/* Initialise plugin state for a given channel count and config. */
int plugin_init(int num_channels, const dsd_config_t *cfg) {
    if (g_state.initialized)
        return -1;

    g_state.config = *cfg;
    g_state.num_channels = num_channels;

    g_state.channels = (engine_channel_t *)calloc(
        (size_t)num_channels, sizeof(engine_channel_t));
    if (!g_state.channels)
        return -1;

    for (int i = 0; i < num_channels; i++) {
        if (engine_channel_init(&g_state.channels[i], i, cfg) != 0) {
            /* Cleanup on failure */
            for (int j = 0; j < i; j++)
                engine_channel_free(&g_state.channels[j]);
            free(g_state.channels);
            g_state.channels = NULL;
            return -1;
        }
    }

    int tc = cfg->thread_count > 0 ? cfg->thread_count : 0;
    g_state.pool = threadpool_create(tc, cfg->affinity_mask);
    if (!g_state.pool) {
        for (int i = 0; i < num_channels; i++)
            engine_channel_free(&g_state.channels[i]);
        free(g_state.channels);
        g_state.channels = NULL;
        return -1;
    }

    g_state.initialized = true;
    return 0;
}

/* Shut down plugin state. */
void plugin_shutdown(void) {
    if (!g_state.initialized)
        return;

    if (g_state.pool) {
        threadpool_destroy(g_state.pool);
        g_state.pool = NULL;
    }

    if (g_state.channels) {
        for (int i = 0; i < g_state.num_channels; i++)
            engine_channel_free(&g_state.channels[i]);
        free(g_state.channels);
        g_state.channels = NULL;
    }

    g_state.initialized = false;
}

/* Reconfigure with new settings (called from config dialog). */
int plugin_reconfigure(const dsd_config_t *cfg) {
    if (!g_state.initialized)
        return -1;

    g_state.config = *cfg;

    for (int i = 0; i < g_state.num_channels; i++) {
        if (engine_channel_reconfigure(&g_state.channels[i], cfg) != 0)
            return -1;
    }

    return 0;
}

/* Get current config (for property page read-back). */
const dsd_config_t *plugin_get_config(void) {
    return &g_state.config;
}
