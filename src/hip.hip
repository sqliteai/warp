/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * hip.hip — AMD ROCm/HIP backend for WARP.
 *
 * Implements runtime GPU detection, HIP context initialization,
 * and dispatch table registration for AMD Radeon / Instinct hardware.
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "waste_backend.h"
#include "waste_hip.h"

static char g_hip_device_name[128] = "HIP (ROCm)";
static int g_hip_initialized = 0;
static int g_hip_device_id = 0;

const char *waste_register_hip(waste_kernels *t)
{
    if (g_hip_initialized) {
        return g_hip_device_name;
    }

    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count <= 0) {
        return NULL; /* No usable HIP device; decline acceleration gracefully */
    }

    /* Select device 0 or honour HIP_VISIBLE_DEVICES */
    hipDeviceProp_t prop;
    err = hipGetDeviceProperties(&prop, g_hip_device_id);
    if (err != hipSuccess) {
        return NULL;
    }

    /* Format descriptive name: e.g. "HIP (AMD Radeon RX 9060 XT - gfx1200)" */
    snprintf(g_hip_device_name, sizeof(g_hip_device_name), "HIP (%s - %s)",
             prop.name, prop.gcnArchName);

    /* For Day 1: Device is verified and probed.
     * Kernels remain on CPU fallback until Day 4 implements mvq8. */
    (void)t;

    g_hip_initialized = 1;
    return g_hip_device_name;
}
