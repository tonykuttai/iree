// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <altivec.h>

#include "iree/builtins/ukernel/arch/ppc_64/common_ppc_64.h"
#include "iree/builtins/ukernel/arch/ppc_64/mmt4d_ppc_64_internal.h"

// Power10 MMA (Matrix-Multiply Assist) mmt4d tiles.
//
// MMA exposes 8 architected 512-bit accumulators (ACC0-ACC7, each overlapping
// 4 VSRs). Each "GER" (outer-product) instruction computes a full 4x4 output
// tile per call:
//   * xvf32gerpp  : f32*f32->f32, reducing K0=1 per call.
//   * xvbf16ger2pp: bf16*bf16->f32, reducing K0=2 per call.
// The accumulator's hardware register class is managed entirely by the
// compiler's register allocator via these builtins; this code never
// hand-allocates VSRs or emits inline assembly.
//
// Because one GER == one 4x4 sub-tile, larger M0xN0 macro-tiles are built from
// a grid of 4x4 accumulators. Using multiple accumulators keeps more
// independent GER chains in flight, hiding the instruction latency. An 8x8
// tile uses 4 accumulators (the lower/upper 4 rows crossed with the left/right
// 4 columns), and a 4x8 tile uses 2.

// Loads a 4x4 f32 sub-tile of the output into an accumulator, for the
// accumulate path. `base` points at the top-left element of the 4x4 sub-tile
// within the (row-major) output tile, `row_stride` is the output tile's N0.
//
// NOTE: __builtin_mma_assemble_acc's operand order is reversed relative to
// __builtin_mma_disassemble_acc's output order (verified against the generic
// reference implementation on real Power10 hardware) -- pass the rows
// back-to-front to correctly round-trip with iree_uk_ppc_64_mma_store_f32.
static inline void iree_uk_ppc_64_mma_load_f32(__vector_quad* acc,
                                               const float* base,
                                               iree_uk_index_t row_stride) {
  vector float row0 = vec_xl(0, base + 0 * row_stride);
  vector float row1 = vec_xl(0, base + 1 * row_stride);
  vector float row2 = vec_xl(0, base + 2 * row_stride);
  vector float row3 = vec_xl(0, base + 3 * row_stride);
  __builtin_mma_assemble_acc(acc, (vector unsigned char)row3,
                             (vector unsigned char)row2,
                             (vector unsigned char)row1,
                             (vector unsigned char)row0);
}

// Stores a 4x4 f32 accumulator into the output tile. Mirrors
// iree_uk_ppc_64_mma_load_f32.
static inline void iree_uk_ppc_64_mma_store_f32(__vector_quad* acc, float* base,
                                                iree_uk_index_t row_stride) {
  vector unsigned char out_rows[4];
  __builtin_mma_disassemble_acc((void*)out_rows, acc);
  vec_xst((vector float)out_rows[0], 0, base + 0 * row_stride);
  vec_xst((vector float)out_rows[1], 0, base + 1 * row_stride);
  vec_xst((vector float)out_rows[2], 0, base + 2 * row_stride);
  vec_xst((vector float)out_rows[3], 0, base + 3 * row_stride);
}

// f32*f32->f32, single 4x4 accumulator (one xvf32gerpp per k). The accumulator
// IS the 4x4 output tile, disassembled into row-major order matching
// out_tile[i0*N0+j0] directly -- no transpose needed.
//
// When not accumulating into existing output, skip the explicit
// __builtin_mma_xxsetaccz + redundant first accumulate-into-zero: use the
// non-accumulating __builtin_mma_xvf32ger directly for k=0 instead, then
// __builtin_mma_xvf32gerpp for k=1..K-1. Mirrors OpenBLAS's
// sgemm_kernel_power10.c technique (verified via direct read of that source);
// confirmed via disassembly that LLVM does not perform this transformation
// on its own (it requires recognizing two separate C calls -- zero, then
// accumulate -- as equivalent to one direct-set call).
void iree_uk_mmt4d_tile_f32f32f32_4x4x1_ppc_64_mma(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float* IREE_UK_RESTRICT out_ptr = out_tile;

  __vector_quad acc;
  iree_uk_index_t k = 0;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    iree_uk_ppc_64_mma_load_f32(&acc, out_ptr, /*row_stride=*/4);
  } else if (params->K > 0) {
    vector float lhs0 = vec_xl(0, lhs_ptr);
    vector float rhs0 = vec_xl(0, rhs_ptr);
    __builtin_mma_xvf32ger(&acc, (vector unsigned char)lhs0,
                           (vector unsigned char)rhs0);
    k = 1;
  } else {
    __builtin_mma_xxsetaccz(&acc);
  }

  for (; k < params->K; ++k) {
    vector float lhs = vec_xl(0, lhs_ptr + 4 * k);
    vector float rhs = vec_xl(0, rhs_ptr + 4 * k);
    __builtin_mma_xvf32gerpp(&acc, (vector unsigned char)lhs,
                             (vector unsigned char)rhs);
  }

  iree_uk_ppc_64_mma_store_f32(&acc, out_ptr, /*row_stride=*/4);
}

// f32*f32->f32, 4x8 tile using 2 accumulators (columns [0,4) and [4,8)).
void iree_uk_mmt4d_tile_f32f32f32_4x8x1_ppc_64_mma(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float* IREE_UK_RESTRICT out_ptr = out_tile;

  __vector_quad acc0, acc1;
  iree_uk_index_t k = 0;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    iree_uk_ppc_64_mma_load_f32(&acc0, out_ptr + 0, /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc1, out_ptr + 4, /*row_stride=*/8);
  } else if (params->K > 0) {
    vector float lhs0 = vec_xl(0, lhs_ptr);
    vector float rhs_lo0 = vec_xl(0, rhs_ptr + 0);
    vector float rhs_hi0 = vec_xl(0, rhs_ptr + 4);
    __builtin_mma_xvf32ger(&acc0, (vector unsigned char)lhs0,
                           (vector unsigned char)rhs_lo0);
    __builtin_mma_xvf32ger(&acc1, (vector unsigned char)lhs0,
                           (vector unsigned char)rhs_hi0);
    k = 1;
  } else {
    __builtin_mma_xxsetaccz(&acc0);
    __builtin_mma_xxsetaccz(&acc1);
  }

  for (; k < params->K; ++k) {
    vector float lhs = vec_xl(0, lhs_ptr + 4 * k);
    vector float rhs_lo = vec_xl(0, rhs_ptr + 8 * k + 0);
    vector float rhs_hi = vec_xl(0, rhs_ptr + 8 * k + 4);
    __builtin_mma_xvf32gerpp(&acc0, (vector unsigned char)lhs,
                             (vector unsigned char)rhs_lo);
    __builtin_mma_xvf32gerpp(&acc1, (vector unsigned char)lhs,
                             (vector unsigned char)rhs_hi);
  }

  iree_uk_ppc_64_mma_store_f32(&acc0, out_ptr + 0, /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc1, out_ptr + 4, /*row_stride=*/8);
}

// f32*f32->f32, 8x8 tile using 4 accumulators covering the four 4x4 quadrants:
//   acc00: rows [0,4) cols [0,4)    acc01: rows [0,4) cols [4,8)
//   acc10: rows [4,8) cols [0,4)    acc11: rows [4,8) cols [4,8)
void iree_uk_mmt4d_tile_f32f32f32_8x8x1_ppc_64_mma(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float* IREE_UK_RESTRICT out_ptr = out_tile;

  __vector_quad acc00, acc01, acc10, acc11;
  iree_uk_index_t k = 0;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    iree_uk_ppc_64_mma_load_f32(&acc00, out_ptr + 0, /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc01, out_ptr + 4, /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc10, out_ptr + 32, /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc11, out_ptr + 36, /*row_stride=*/8);
  } else if (params->K > 0) {
    vector float lhs_lo0 = vec_xl(0, lhs_ptr + 0);
    vector float lhs_hi0 = vec_xl(0, lhs_ptr + 4);
    vector float rhs_lo0 = vec_xl(0, rhs_ptr + 0);
    vector float rhs_hi0 = vec_xl(0, rhs_ptr + 4);
    __builtin_mma_xvf32ger(&acc00, (vector unsigned char)lhs_lo0,
                           (vector unsigned char)rhs_lo0);
    __builtin_mma_xvf32ger(&acc01, (vector unsigned char)lhs_lo0,
                           (vector unsigned char)rhs_hi0);
    __builtin_mma_xvf32ger(&acc10, (vector unsigned char)lhs_hi0,
                           (vector unsigned char)rhs_lo0);
    __builtin_mma_xvf32ger(&acc11, (vector unsigned char)lhs_hi0,
                           (vector unsigned char)rhs_hi0);
    k = 1;
  } else {
    __builtin_mma_xxsetaccz(&acc00);
    __builtin_mma_xxsetaccz(&acc01);
    __builtin_mma_xxsetaccz(&acc10);
    __builtin_mma_xxsetaccz(&acc11);
  }

  for (; k < params->K; ++k) {
    vector float lhs_lo = vec_xl(0, lhs_ptr + 8 * k + 0);
    vector float lhs_hi = vec_xl(0, lhs_ptr + 8 * k + 4);
    vector float rhs_lo = vec_xl(0, rhs_ptr + 8 * k + 0);
    vector float rhs_hi = vec_xl(0, rhs_ptr + 8 * k + 4);
    __builtin_mma_xvf32gerpp(&acc00, (vector unsigned char)lhs_lo,
                             (vector unsigned char)rhs_lo);
    __builtin_mma_xvf32gerpp(&acc01, (vector unsigned char)lhs_lo,
                             (vector unsigned char)rhs_hi);
    __builtin_mma_xvf32gerpp(&acc10, (vector unsigned char)lhs_hi,
                             (vector unsigned char)rhs_lo);
    __builtin_mma_xvf32gerpp(&acc11, (vector unsigned char)lhs_hi,
                             (vector unsigned char)rhs_hi);
  }

  iree_uk_ppc_64_mma_store_f32(&acc00, out_ptr + 0, /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc01, out_ptr + 4, /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc10, out_ptr + 32, /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc11, out_ptr + 36, /*row_stride=*/8);
}

// f32*f32->f32, 16x8 tile using all 8 accumulators — matching OpenBLAS's
// sgemm_kernel_power10.c primary tile (SGEMM_DEFAULT_UNROLL_M=16,
// SGEMM_DEFAULT_UNROLL_N=8). Accumulator layout (each block is 4x4):
//   acc00: rows[0:3]   x cols[0:3]    acc01: rows[0:3]   x cols[4:7]
//   acc10: rows[4:7]   x cols[0:3]    acc11: rows[4:7]   x cols[4:7]
//   acc20: rows[8:11]  x cols[0:3]    acc21: rows[8:11]  x cols[4:7]
//   acc30: rows[12:15] x cols[0:3]    acc31: rows[12:15] x cols[4:7]
void iree_uk_mmt4d_tile_f32f32f32_16x8x1_ppc_64_mma(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float* IREE_UK_RESTRICT out_ptr = out_tile;

  __vector_quad acc00, acc01, acc10, acc11, acc20, acc21, acc30, acc31;
  iree_uk_index_t k = 0;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    iree_uk_ppc_64_mma_load_f32(&acc00, out_ptr + 0,   /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc01, out_ptr + 4,   /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc10, out_ptr + 32,  /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc11, out_ptr + 36,  /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc20, out_ptr + 64,  /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc21, out_ptr + 68,  /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc30, out_ptr + 96,  /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc31, out_ptr + 100, /*row_stride=*/8);
  } else if (params->K > 0) {
    vector float lhs0 = vec_xl(0, lhs_ptr + 0);
    vector float lhs1 = vec_xl(0, lhs_ptr + 4);
    vector float lhs2 = vec_xl(0, lhs_ptr + 8);
    vector float lhs3 = vec_xl(0, lhs_ptr + 12);
    vector float rhs0 = vec_xl(0, rhs_ptr + 0);
    vector float rhs1 = vec_xl(0, rhs_ptr + 4);
    __builtin_mma_xvf32ger(&acc00, (vector unsigned char)lhs0,
                           (vector unsigned char)rhs0);
    __builtin_mma_xvf32ger(&acc01, (vector unsigned char)lhs0,
                           (vector unsigned char)rhs1);
    __builtin_mma_xvf32ger(&acc10, (vector unsigned char)lhs1,
                           (vector unsigned char)rhs0);
    __builtin_mma_xvf32ger(&acc11, (vector unsigned char)lhs1,
                           (vector unsigned char)rhs1);
    __builtin_mma_xvf32ger(&acc20, (vector unsigned char)lhs2,
                           (vector unsigned char)rhs0);
    __builtin_mma_xvf32ger(&acc21, (vector unsigned char)lhs2,
                           (vector unsigned char)rhs1);
    __builtin_mma_xvf32ger(&acc30, (vector unsigned char)lhs3,
                           (vector unsigned char)rhs0);
    __builtin_mma_xvf32ger(&acc31, (vector unsigned char)lhs3,
                           (vector unsigned char)rhs1);
    k = 1;
  } else {
    __builtin_mma_xxsetaccz(&acc00);
    __builtin_mma_xxsetaccz(&acc01);
    __builtin_mma_xxsetaccz(&acc10);
    __builtin_mma_xxsetaccz(&acc11);
    __builtin_mma_xxsetaccz(&acc20);
    __builtin_mma_xxsetaccz(&acc21);
    __builtin_mma_xxsetaccz(&acc30);
    __builtin_mma_xxsetaccz(&acc31);
  }

  for (; k < params->K; ++k) {
    vector float lhs0 = vec_xl(0, lhs_ptr + 16 * k + 0);
    vector float lhs1 = vec_xl(0, lhs_ptr + 16 * k + 4);
    vector float lhs2 = vec_xl(0, lhs_ptr + 16 * k + 8);
    vector float lhs3 = vec_xl(0, lhs_ptr + 16 * k + 12);
    vector float rhs0 = vec_xl(0, rhs_ptr + 8 * k + 0);
    vector float rhs1 = vec_xl(0, rhs_ptr + 8 * k + 4);
    __builtin_mma_xvf32gerpp(&acc00, (vector unsigned char)lhs0,
                             (vector unsigned char)rhs0);
    __builtin_mma_xvf32gerpp(&acc01, (vector unsigned char)lhs0,
                             (vector unsigned char)rhs1);
    __builtin_mma_xvf32gerpp(&acc10, (vector unsigned char)lhs1,
                             (vector unsigned char)rhs0);
    __builtin_mma_xvf32gerpp(&acc11, (vector unsigned char)lhs1,
                             (vector unsigned char)rhs1);
    __builtin_mma_xvf32gerpp(&acc20, (vector unsigned char)lhs2,
                             (vector unsigned char)rhs0);
    __builtin_mma_xvf32gerpp(&acc21, (vector unsigned char)lhs2,
                             (vector unsigned char)rhs1);
    __builtin_mma_xvf32gerpp(&acc30, (vector unsigned char)lhs3,
                             (vector unsigned char)rhs0);
    __builtin_mma_xvf32gerpp(&acc31, (vector unsigned char)lhs3,
                             (vector unsigned char)rhs1);
  }

  iree_uk_ppc_64_mma_store_f32(&acc00, out_ptr + 0,   /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc01, out_ptr + 4,   /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc10, out_ptr + 32,  /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc11, out_ptr + 36,  /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc20, out_ptr + 64,  /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc21, out_ptr + 68,  /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc30, out_ptr + 96,  /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc31, out_ptr + 100, /*row_stride=*/8);
}

// bf16*bf16->f32. The accumulator holds f32 values, so the output load/store
// path is identical to f32; only the GER instruction and input panel stride
// differ. xvbf16ger2pp consumes, per call, one VSR of 8 bf16 from each operand
// laid out as 4 rows (or columns) x K0=2, accumulating the 2 partial products
// into the f32 4x4 accumulator. The mmt4d panel layout (M0/N0 major, K0 minor)
// matches that operand layout directly, so a 16-byte load yields the operand
// for one GER. Byte stride per K0-block is M0*K0*sizeof(bf16) for the lhs and
// N0*K0*sizeof(bf16) for the rhs.

// bf16*bf16->f32, 4x8 tile using 2 accumulators.
void iree_uk_mmt4d_tile_bf16bf16f32_4x8x2_ppc_64_mma(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const unsigned char* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const unsigned char* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float* IREE_UK_RESTRICT out_ptr = out_tile;

  __vector_quad acc0, acc1;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    iree_uk_ppc_64_mma_load_f32(&acc0, out_ptr + 0, /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc1, out_ptr + 4, /*row_stride=*/8);
  } else {
    __builtin_mma_xxsetaccz(&acc0);
    __builtin_mma_xxsetaccz(&acc1);
  }

  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    vector unsigned char lhs = vec_xl(0, lhs_ptr + k * (4 * 2 * 2));
    vector unsigned char rhs_lo = vec_xl(0, rhs_ptr + k * (8 * 2 * 2) + 0);
    vector unsigned char rhs_hi = vec_xl(0, rhs_ptr + k * (8 * 2 * 2) + 16);
    __builtin_mma_xvbf16ger2pp(&acc0, lhs, rhs_lo);
    __builtin_mma_xvbf16ger2pp(&acc1, lhs, rhs_hi);
  }

  iree_uk_ppc_64_mma_store_f32(&acc0, out_ptr + 0, /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc1, out_ptr + 4, /*row_stride=*/8);
}

// bf16*bf16->f32, 8x8 tile using 4 accumulators.
void iree_uk_mmt4d_tile_bf16bf16f32_8x8x2_ppc_64_mma(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const unsigned char* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const unsigned char* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float* IREE_UK_RESTRICT out_ptr = out_tile;

  __vector_quad acc00, acc01, acc10, acc11;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    iree_uk_ppc_64_mma_load_f32(&acc00, out_ptr + 0, /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc01, out_ptr + 4, /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc10, out_ptr + 32, /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc11, out_ptr + 36, /*row_stride=*/8);
  } else {
    __builtin_mma_xxsetaccz(&acc00);
    __builtin_mma_xxsetaccz(&acc01);
    __builtin_mma_xxsetaccz(&acc10);
    __builtin_mma_xxsetaccz(&acc11);
  }

  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    vector unsigned char lhs_lo = vec_xl(0, lhs_ptr + k * (8 * 2 * 2) + 0);
    vector unsigned char lhs_hi = vec_xl(0, lhs_ptr + k * (8 * 2 * 2) + 16);
    vector unsigned char rhs_lo = vec_xl(0, rhs_ptr + k * (8 * 2 * 2) + 0);
    vector unsigned char rhs_hi = vec_xl(0, rhs_ptr + k * (8 * 2 * 2) + 16);
    __builtin_mma_xvbf16ger2pp(&acc00, lhs_lo, rhs_lo);
    __builtin_mma_xvbf16ger2pp(&acc01, lhs_lo, rhs_hi);
    __builtin_mma_xvbf16ger2pp(&acc10, lhs_hi, rhs_lo);
    __builtin_mma_xvbf16ger2pp(&acc11, lhs_hi, rhs_hi);
  }

  iree_uk_ppc_64_mma_store_f32(&acc00, out_ptr + 0, /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc01, out_ptr + 4, /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc10, out_ptr + 32, /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc11, out_ptr + 36, /*row_stride=*/8);
}

// bf16*bf16->f32, 16x8 tile using all 8 accumulators — matching OpenBLAS's
// sbgemm_kernel_power10.c primary tile (SBGEMM_DEFAULT_UNROLL_M=16,
// SBGEMM_DEFAULT_UNROLL_N=8). Same accumulator layout as f32 16x8x1.
// LHS panel stride per k-step: M0*K0*sizeof(bf16) = 16*2*2 = 64 bytes.
// RHS panel stride per k-step: N0*K0*sizeof(bf16) = 8*2*2 = 32 bytes.
void iree_uk_mmt4d_tile_bf16bf16f32_16x8x2_ppc_64_mma(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const unsigned char* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const unsigned char* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float* IREE_UK_RESTRICT out_ptr = out_tile;

  __vector_quad acc00, acc01, acc10, acc11, acc20, acc21, acc30, acc31;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    iree_uk_ppc_64_mma_load_f32(&acc00, out_ptr + 0,   /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc01, out_ptr + 4,   /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc10, out_ptr + 32,  /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc11, out_ptr + 36,  /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc20, out_ptr + 64,  /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc21, out_ptr + 68,  /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc30, out_ptr + 96,  /*row_stride=*/8);
    iree_uk_ppc_64_mma_load_f32(&acc31, out_ptr + 100, /*row_stride=*/8);
  } else {
    __builtin_mma_xxsetaccz(&acc00);
    __builtin_mma_xxsetaccz(&acc01);
    __builtin_mma_xxsetaccz(&acc10);
    __builtin_mma_xxsetaccz(&acc11);
    __builtin_mma_xxsetaccz(&acc20);
    __builtin_mma_xxsetaccz(&acc21);
    __builtin_mma_xxsetaccz(&acc30);
    __builtin_mma_xxsetaccz(&acc31);
  }

  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    vector unsigned char lhs0 = vec_xl(0, lhs_ptr + k * (16 * 2 * 2) + 0);
    vector unsigned char lhs1 = vec_xl(0, lhs_ptr + k * (16 * 2 * 2) + 16);
    vector unsigned char lhs2 = vec_xl(0, lhs_ptr + k * (16 * 2 * 2) + 32);
    vector unsigned char lhs3 = vec_xl(0, lhs_ptr + k * (16 * 2 * 2) + 48);
    vector unsigned char rhs0 = vec_xl(0, rhs_ptr + k * (8 * 2 * 2) + 0);
    vector unsigned char rhs1 = vec_xl(0, rhs_ptr + k * (8 * 2 * 2) + 16);
    __builtin_mma_xvbf16ger2pp(&acc00, lhs0, rhs0);
    __builtin_mma_xvbf16ger2pp(&acc01, lhs0, rhs1);
    __builtin_mma_xvbf16ger2pp(&acc10, lhs1, rhs0);
    __builtin_mma_xvbf16ger2pp(&acc11, lhs1, rhs1);
    __builtin_mma_xvbf16ger2pp(&acc20, lhs2, rhs0);
    __builtin_mma_xvbf16ger2pp(&acc21, lhs2, rhs1);
    __builtin_mma_xvbf16ger2pp(&acc30, lhs3, rhs0);
    __builtin_mma_xvbf16ger2pp(&acc31, lhs3, rhs1);
  }

  iree_uk_ppc_64_mma_store_f32(&acc00, out_ptr + 0,   /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc01, out_ptr + 4,   /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc10, out_ptr + 32,  /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc11, out_ptr + 36,  /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc20, out_ptr + 64,  /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc21, out_ptr + 68,  /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc30, out_ptr + 96,  /*row_stride=*/8);
  iree_uk_ppc_64_mma_store_f32(&acc31, out_ptr + 100, /*row_stride=*/8);
}
