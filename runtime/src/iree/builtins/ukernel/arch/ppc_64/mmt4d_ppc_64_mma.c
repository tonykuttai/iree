// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <altivec.h>

#include "iree/builtins/ukernel/arch/ppc_64/common_ppc_64.h"
#include "iree/builtins/ukernel/arch/ppc_64/mmt4d_ppc_64_internal.h"

// f32*f32->f32 matmul tile using the Power10 MMA (Matrix-Multiply Assist)
// outer-product accumulator. `xvf32gerpp` computes a full 4x4 outer product
// of two 4-element f32 vectors per instruction, accumulating into a 512-bit
// accumulator (`__vector_quad`) representing the 4x4 output tile. This is
// why the tile shape is fixed at M0=4, N0=4, K0=1: one GER call per k
// consumes one 4-element column of the LHS panel and one 4-element row of
// the RHS panel (both contiguous in the K0=1 panel layout), and the
// accumulator IS the 4x4 output tile, disassembled into row-major order
// matching out_tile[i0*N0+j0] directly -- no transpose needed.
//
// The accumulator's hardware register class (ACC0-ACC7, each overlapping 4
// VSRs) is managed entirely by the compiler's register allocator via these
// builtins; this code never hand-allocates VSRs or emits inline assembly.
void iree_uk_mmt4d_tile_f32f32f32_4x4x1_ppc_64_mma(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float* IREE_UK_RESTRICT out_ptr = out_tile;

  __vector_quad acc;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    vector float row0 = vec_xl(0, out_ptr + 0);
    vector float row1 = vec_xl(0, out_ptr + 4);
    vector float row2 = vec_xl(0, out_ptr + 8);
    vector float row3 = vec_xl(0, out_ptr + 12);
    // NOTE: __builtin_mma_assemble_acc's operand order is reversed relative
    // to __builtin_mma_disassemble_acc's output order (verified against the
    // generic reference implementation on real Power10 hardware) -- pass the
    // rows back-to-front to correctly round-trip with disassemble_acc below.
    __builtin_mma_assemble_acc(&acc, (vector unsigned char)row3,
                               (vector unsigned char)row2,
                               (vector unsigned char)row1,
                               (vector unsigned char)row0);
  } else {
    __builtin_mma_xxsetaccz(&acc);
  }

  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    vector float lhs = vec_xl(0, lhs_ptr + 4 * k);
    vector float rhs = vec_xl(0, rhs_ptr + 4 * k);
    __builtin_mma_xvf32gerpp(&acc, (vector unsigned char)lhs,
                             (vector unsigned char)rhs);
  }

  vector unsigned char out_rows[4];
  __builtin_mma_disassemble_acc((void*)out_rows, &acc);
  vec_xst((vector float)out_rows[0], 0, out_ptr + 0);
  vec_xst((vector float)out_rows[1], 0, out_ptr + 4);
  vec_xst((vector float)out_rows[2], 0, out_ptr + 8);
  vec_xst((vector float)out_rows[3], 0, out_ptr + 12);
}
