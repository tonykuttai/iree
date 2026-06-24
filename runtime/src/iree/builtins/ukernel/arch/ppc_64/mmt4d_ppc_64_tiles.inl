// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Ordering matters when multiple lines have the same types and tile shape and
// are supported by the CPU. In that case, the last-enumerated line overrides
// preceding lines. Always go from oldest to shiniest code path.

// Power10 (ISA 3.1) MMA: each GER instruction computes a full 4x4 outer
// product per instruction. Larger M0xN0 macro-tiles are built from a grid of
// 4x4 accumulators (8 hardware accumulators are available), keeping more
// independent GER chains in flight to hide latency. Narrow-M cases that do not
// match a registered tile fall back to the generic implementation.
//
// f32*f32->f32 uses xvf32gerpp (K0=1).
IREE_UK_MMT4D_TILE(ppc_64, f32, f32, f32, 4, 4, 1, _mma)
IREE_UK_MMT4D_TILE(ppc_64, f32, f32, f32, 4, 8, 1, _mma)
IREE_UK_MMT4D_TILE(ppc_64, f32, f32, f32, 8, 8, 1, _mma)
IREE_UK_MMT4D_TILE(ppc_64, f32, f32, f32, 16, 8, 1, _mma)
// bf16*bf16->f32 uses xvbf16ger2pp (K0=2).
IREE_UK_MMT4D_TILE(ppc_64, bf16, bf16, f32, 4, 8, 2, _mma)
IREE_UK_MMT4D_TILE(ppc_64, bf16, bf16, f32, 8, 8, 2, _mma)
IREE_UK_MMT4D_TILE(ppc_64, bf16, bf16, f32, 16, 8, 2, _mma)
