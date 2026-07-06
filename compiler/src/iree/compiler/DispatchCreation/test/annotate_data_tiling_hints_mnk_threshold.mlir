// Gate enabled: exclude contraction ops whose static M*N*K volume < 4194304.
// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-annotate-data-tiling-hints{data-tiling-mnk-threshold=4194304}))" --split-input-file %s | FileCheck %s --check-prefix=THRESH
// Gate disabled (default): every supported matmul keeps the hint.
// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-annotate-data-tiling-hints))" --split-input-file %s | FileCheck %s --check-prefix=NOGATE

// Static volume 32*32*32 = 32768 < threshold -> gated out when enabled.
util.func public @small_matmul_static(%arg0 : tensor<32x32xf32>, %arg1 : tensor<32x32xf32>, %arg2 : tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = linalg.matmul
         ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>)
         outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
  util.return %0 : tensor<32x32xf32>
}
// THRESH-LABEL: @small_matmul_static(
// THRESH:         linalg.matmul
// THRESH-NOT:       iree.opt.data_tiling
// NOGATE-LABEL: @small_matmul_static(
// NOGATE:         linalg.matmul
// NOGATE-SAME:      iree.opt.data_tiling

// -----

// Static volume 256*256*256 = 16777216 > threshold -> keeps the hint.
util.func public @large_matmul_static(%arg0 : tensor<256x256xf32>, %arg1 : tensor<256x256xf32>, %arg2 : tensor<256x256xf32>) -> tensor<256x256xf32> {
  %0 = linalg.matmul
         ins(%arg0, %arg1 : tensor<256x256xf32>, tensor<256x256xf32>)
         outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  util.return %0 : tensor<256x256xf32>
}
// THRESH-LABEL: @large_matmul_static(
// THRESH:         linalg.matmul
// THRESH-SAME:      iree.opt.data_tiling
// NOGATE-LABEL: @large_matmul_static(
// NOGATE:         linalg.matmul
// NOGATE-SAME:      iree.opt.data_tiling

// -----

// Static volume 64*256*256 = 4194304 == threshold. The gate excludes only
// volumes strictly below the threshold, so this op keeps the hint.
util.func public @matmul_at_threshold(%arg0 : tensor<64x256xf32>, %arg1 : tensor<256x256xf32>, %arg2 : tensor<64x256xf32>) -> tensor<64x256xf32> {
  %0 = linalg.matmul
         ins(%arg0, %arg1 : tensor<64x256xf32>, tensor<256x256xf32>)
         outs(%arg2 : tensor<64x256xf32>) -> tensor<64x256xf32>
  util.return %0 : tensor<64x256xf32>
}
// THRESH-LABEL: @matmul_at_threshold(
// THRESH:         linalg.matmul
// THRESH-SAME:      iree.opt.data_tiling

// -----

// Dynamic M/N/K: volume cannot be evaluated, so the op is never gated.
util.func public @dynamic_matmul(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
         ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
         outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}
// THRESH-LABEL: @dynamic_matmul(
// THRESH:         linalg.matmul
// THRESH-SAME:      iree.opt.data_tiling

// -----

// Batch is excluded from the volume: per-contraction M*N*K = 8*8*8 = 512 is
// below the threshold even though the batch dimension is large, so it is gated.
util.func public @batch_matmul_small_per_contraction(%arg0 : tensor<128x8x8xf32>, %arg1 : tensor<128x8x8xf32>, %arg2 : tensor<128x8x8xf32>) -> tensor<128x8x8xf32> {
  %0 = linalg.batch_matmul
         ins(%arg0, %arg1 : tensor<128x8x8xf32>, tensor<128x8x8xf32>)
         outs(%arg2 : tensor<128x8x8xf32>) -> tensor<128x8x8xf32>
  util.return %0 : tensor<128x8x8xf32>
}
// THRESH-LABEL: @batch_matmul_small_per_contraction(
// THRESH:         linalg.batch_matmul
// THRESH-NOT:       iree.opt.data_tiling
// NOGATE-LABEL: @batch_matmul_small_per_contraction(
// NOGATE:         linalg.batch_matmul
// NOGATE-SAME:      iree.opt.data_tiling
