// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/ppc_64/common_ppc_64.h"
#include "iree/builtins/ukernel/query_tile_sizes_internal.h"

// Selects M0 for the 8-accumulator (M0=16) vs 4-accumulator (M0=8) MMA tile.
//
// M0=16 needs a full 16-row LHS tile to invoke the MMA kernel; if M < 16 the
// kernel is never reached and the compiler falls back to generic for all rows,
// which is strictly worse than using M0=8 (1 MMA tile + small generic tail).
//
// size0 represents M for LHS (M×K) and RESULT (M×N) roles only.  For RHS
// (K×N) M is not in the params, but M0 from the RHS return value is discarded
// by the caller (it uses N0 and K0 only), so returning 16 is safe.
// Negative size0 is the "dynamic M" sentinel (IREE_UK_INT64_MIN) — use M0=16
// optimistically since dynamic shapes are typically large at inference time.
static int iree_uk_ppc_64_mma_m0(
    const iree_uk_query_tile_sizes_2d_params_t* params) {
  iree_uk_uint32_t role = iree_uk_query_tile_sizes_operand_role(params->flags);
  if (role == IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_LHS ||
      role == IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RESULT) {
    // Static M known and too small to fill a 16-row tile: use M0=8 instead.
    if (params->size0 >= 0 && params->size0 < 16) return 8;
  }
  return 16;
}

static iree_uk_matmul_tile_sizes_t
iree_uk_query_matmul_tile_sizes_ppc_64_f32f32f32(
    const iree_uk_query_tile_sizes_2d_params_t* params) {
#if defined(IREE_UK_BUILD_PPC_64_MMA)
  if (iree_uk_cpu_ppc_64_mma(params->cpu_data)) {
    int m0 = iree_uk_ppc_64_mma_m0(params);
    return (iree_uk_matmul_tile_sizes_t){.M = m0, .K = 1, .N = 8};
  }
#endif
  return (iree_uk_matmul_tile_sizes_t){.M = 8, .K = 1, .N = 8};
}

static iree_uk_matmul_tile_sizes_t
iree_uk_query_matmul_tile_sizes_ppc_64_bf16bf16f32(
    const iree_uk_query_tile_sizes_2d_params_t* params) {
#if defined(IREE_UK_BUILD_PPC_64_MMA)
  if (iree_uk_cpu_ppc_64_mma(params->cpu_data)) {
    int m0 = iree_uk_ppc_64_mma_m0(params);
    return (iree_uk_matmul_tile_sizes_t){.M = m0, .K = 2, .N = 8};
  }
#endif
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
