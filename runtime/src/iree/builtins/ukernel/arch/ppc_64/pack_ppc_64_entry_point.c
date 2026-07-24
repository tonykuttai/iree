// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/ppc_64/common_ppc_64.h"
#include "iree/builtins/ukernel/arch/ppc_64/pack_ppc_64_internal.h"

// Tile functions for the layouts produced by CPU data-tiling for mmt4d on
// ppc_64 (M0=16/N0=8 MMA tiles and the 8x8 fallback, all K0=1 f32).
// Plain C with compile-time-constant tile sizes: VSX is baseline on ppc64le,
// so the bitcode/native compiler auto-vectorizes these loops. Baseline VSX
// only — keep these out of any -mcpu-gated build target.
//
// Signature contract (see pack_tile.c generic funcs): strides are in
// elements; for TRANSPOSE_INNER layouts the caller has already swapped
// tile_size0/tile_size1 before calling.

// LHS f32 [16,1]: each tile is a 16-element column gathered at stride
// in_stride0 into a contiguous 64-byte run.
static void iree_uk_pack_tile_16x1_x32_ppc_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 16);
  IREE_UK_ASSERT(tile_size1 == 1);
  const iree_uk_uint32_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  iree_uk_uint32_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  for (iree_uk_index_t outer_i1 = 0; outer_i1 < outer_size1; ++outer_i1) {
    for (int i = 0; i < 16; ++i) out_ptr[i] = in_ptr[i * in_stride0];
    out_ptr += out_stride1;
    in_ptr += 1;
  }
}

// LHS f32 [8,1] (8x8 fallback tile): as above with an 8-element column.
static void iree_uk_pack_tile_8x1_x32_ppc_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 1);
  const iree_uk_uint32_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  iree_uk_uint32_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  for (iree_uk_index_t outer_i1 = 0; outer_i1 < outer_size1; ++outer_i1) {
    for (int i = 0; i < 8; ++i) out_ptr[i] = in_ptr[i * in_stride0];
    out_ptr += out_stride1;
    in_ptr += 1;
  }
}

// RHS f32 [8,1] TRANSPOSE_INNER (tile sizes arrive swapped: tile_size0=1,
// tile_size1=8): each tile is 8 contiguous source elements copied to 8
// contiguous destination elements — a straight 32-byte copy.
static void iree_uk_pack_tile_8x1_x32_ppc_64_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 1);
  IREE_UK_ASSERT(tile_size1 == 8);
  const iree_uk_uint32_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  iree_uk_uint32_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  for (iree_uk_index_t outer_i1 = 0; outer_i1 < outer_size1; ++outer_i1) {
    for (int i = 0; i < 8; ++i) out_ptr[i] = in_ptr[i];
    out_ptr += out_stride1;
    in_ptr += 8;
  }
}

iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func_arch(
    const iree_uk_pack_params_t* params) {
  // Pack does no arithmetic, so only the element size matters, not the type.
  iree_uk_pack_type_t pack_type = iree_uk_pack_type(params->flags);
  int esize = iree_uk_type_size(iree_uk_pack_out_type(pack_type));
  bool transpose = params->flags & IREE_UK_FLAG_PACK_TRANSPOSE_INNER;
  if (esize == 4 && params->out_size3 == 1) {
    if (params->out_size2 == 16) {
      return transpose ? 0 : iree_uk_pack_tile_16x1_x32_ppc_64_direct;
    }
    if (params->out_size2 == 8) {
      return transpose ? iree_uk_pack_tile_8x1_x32_ppc_64_transpose
                       : iree_uk_pack_tile_8x1_x32_ppc_64_direct;
    }
  }
  return 0;
}
