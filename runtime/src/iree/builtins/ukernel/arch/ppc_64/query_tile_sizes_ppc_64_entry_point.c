// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/ppc_64/common_ppc_64.h"
#include "iree/builtins/ukernel/query_tile_sizes_internal.h"

static iree_uk_matmul_tile_sizes_t
iree_uk_query_matmul_tile_sizes_ppc_64_f32f32f32(
    const iree_uk_query_tile_sizes_2d_params_t* params) {
#if defined(IREE_UK_BUILD_PPC_64_MMA)
  if (iree_uk_cpu_ppc_64_mma(params->cpu_data)) {
    // 8x8 macro-tile built from 4 MMA accumulators (xvf32gerpp, K0=1).
    return (iree_uk_matmul_tile_sizes_t){.M = 8, .K = 1, .N = 8};
  }
#endif
  // generic fallback
  return (iree_uk_matmul_tile_sizes_t){.M = 8, .K = 1, .N = 8};
}

static iree_uk_matmul_tile_sizes_t
iree_uk_query_matmul_tile_sizes_ppc_64_bf16bf16f32(
    const iree_uk_query_tile_sizes_2d_params_t* params) {
#if defined(IREE_UK_BUILD_PPC_64_MMA)
  if (iree_uk_cpu_ppc_64_mma(params->cpu_data)) {
    // 8x8 macro-tile built from 4 MMA accumulators (xvbf16ger2pp, K0=2).
    return (iree_uk_matmul_tile_sizes_t){.M = 8, .K = 2, .N = 8};
  }
#endif
  // generic fallback
  return (iree_uk_matmul_tile_sizes_t){.M = 8, .K = 2, .N = 8};
}

bool iree_uk_query_matmul_tile_sizes_arch(
    const iree_uk_query_tile_sizes_2d_params_t* params,
    iree_uk_matmul_tile_sizes_t* out_matmul_tile_sizes) {
  iree_uk_uint32_t op = iree_uk_query_tile_sizes_operation(params->flags);
  if (op == IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F32F32F32) {
    *out_matmul_tile_sizes =
        iree_uk_query_matmul_tile_sizes_ppc_64_f32f32f32(params);
    return true;
  } else if (op == IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_BF16BF16F32) {
    *out_matmul_tile_sizes =
        iree_uk_query_matmul_tile_sizes_ppc_64_bf16bf16f32(params);
    return true;
  }
  return false;
}
