// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Ordering matters when multiple lines have the same types and tile shape and
// are supported by the CPU. In that case, the last-enumerated line overrides
// preceding lines. Always go from oldest to shiniest code path.

// Power10 (ISA 3.1) MMA: xvf32gerpp computes a full 4x4 f32 outer product per
// instruction, so the natural (and only, for now) tile shape is 4x4x1: M0=4,
// N0=4 match the hardware's fixed 4-wide accumulator dimensions, and K0=1
// because each GER call consumes one column of LHS and one row of RHS.
IREE_UK_MMT4D_TILE(ppc_64, f32, f32, f32, 4, 4, 1, _mma)
