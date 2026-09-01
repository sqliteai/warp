/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#ifndef WASTE_HIP_H
#define WASTE_HIP_H

#include "waste_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Registers the HIP backend into the waste_kernels dispatch table.
 * Returns the backend name if an AMD GPU is detected and initialized,
 * or NULL to decline (falling back to CPU/SIMD). */
const char *waste_register_hip(waste_kernels *t);

#ifdef __cplusplus
}
#endif

#endif /* WASTE_HIP_H */
