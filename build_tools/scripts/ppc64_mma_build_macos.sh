#!/bin/bash
# Build + verify the Power10 MMA ppc64 work on macOS (cross-compile host).
#
# macOS can't *execute* ppc64le code, so "verify" here means:
#   1. Build the full compiler (iree-compile/iree-opt) with the ppc64
#      bitcode-embedded ukernel.
#   2. Cross-compile a real matmul targeting --iree-llvmcpu-target-cpu=pwr10
#      with data-tiling + ukernels enabled.
#   3. Disassemble the resulting native object and grep for the actual
#      Power10 MMA instructions (xvf32gerpp / xvbf16ger2pp) to prove the
#      whole pipeline (CPU feature resolution -> tile enumeration ->
#      data tiling -> ukernel embedding) really selected and used MMA.
#
# Usage:
#   ./build_tools/scripts/ppc64_mma_build_macos.sh
#
# Env overrides:
#   IREE_SRC_DIR    (default: repo root, i.e. this script's grandparent dir)
#   IREE_BUILD_DIR  (default: ../iree-build relative to IREE_SRC_DIR)
#   CC / CXX        (default: a local LLVM build's clang/clang++; falls back
#                    to system clang if unset and not found)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IREE_SRC_DIR="${IREE_SRC_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
IREE_BUILD_DIR="${IREE_BUILD_DIR:-$(cd "${IREE_SRC_DIR}/.." && pwd)/iree-build}"

# Prefer a real clang (not Apple clang) so -mcpu=pwr10/PowerPC backend & MMA
# builtins are available. Adjust this path to wherever your LLVM build is.
DEFAULT_CLANG="${HOME}/work/llvm/community/build/bin/clang"
DEFAULT_CLANGXX="${HOME}/work/llvm/community/build/bin/clang++"
CC="${CC:-$( [[ -x "${DEFAULT_CLANG}" ]] && echo "${DEFAULT_CLANG}" || echo clang )}"
CXX="${CXX:-$( [[ -x "${DEFAULT_CLANGXX}" ]] && echo "${DEFAULT_CLANGXX}" || echo clang++ )}"

echo "==> Source dir: ${IREE_SRC_DIR}"
echo "==> Build dir:  ${IREE_BUILD_DIR}"
echo "==> CC=${CC}  CXX=${CXX}"

mkdir -p "${IREE_BUILD_DIR}"

echo "==> Configuring..."
cmake -G Ninja -B "${IREE_BUILD_DIR}" -S "${IREE_SRC_DIR}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DIREE_ENABLE_LLD=ON \
  -DIREE_BUILD_COMPILER=ON \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=ON \
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON

echo "==> Building iree-compile, iree-opt, and the ppc_64 ukernel bitcode..."
ninja -C "${IREE_BUILD_DIR}" \
  tools/iree-compile tools/iree-opt \
  runtime/src/iree/builtins/ukernel/ukernel_bitcode_ppc_64.bc

IREE_COMPILE="${IREE_BUILD_DIR}/tools/iree-compile"
OBJDUMP="$(dirname "${CC}")/llvm-objdump"
[[ -x "${OBJDUMP}" ]] || OBJDUMP="llvm-objdump"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

cat > "${WORKDIR}/matmul_f32.mlir" <<'EOF'
func.func @matmul(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>) -> tensor<512x512xf32> {
  %cst = arith.constant dense<0.0> : tensor<512x512xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%cst : tensor<512x512xf32>) -> tensor<512x512xf32>
  return %0 : tensor<512x512xf32>
}
EOF

cat > "${WORKDIR}/matmul_bf16.mlir" <<'EOF'
func.func @matmul_bf16(%arg0: tensor<512x512xbf16>, %arg1: tensor<512x512xbf16>) -> tensor<512x512xf32> {
  %cst = arith.constant dense<0.0> : tensor<512x512xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xbf16>, tensor<512x512xbf16>) outs(%cst : tensor<512x512xf32>) -> tensor<512x512xf32>
  return %0 : tensor<512x512xf32>
}
EOF

verify_mma() {
  local mlir_file="$1" obj_name="$2" instr_grep="$3" label="$4"
  echo "==> Compiling ${label} for powerpc64le-unknown-linux-gnu (pwr10, ukernels+data-tiling)..."
  "${IREE_COMPILE}" \
    --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=powerpc64le-unknown-linux-gnu \
    --iree-llvmcpu-target-cpu=pwr10 \
    --iree-llvmcpu-enable-ukernels=all \
    --iree-opt-data-tiling \
    --iree-llvmcpu-debug-symbols=true \
    --iree-llvmcpu-link-embedded=false \
    --iree-llvmcpu-link-static \
    --iree-llvmcpu-static-library-output-path="${WORKDIR}/${obj_name}" \
    "${mlir_file}" -o "${WORKDIR}/${obj_name}.vmfb"

  file "${WORKDIR}/${obj_name}"
  local count
  count="$("${OBJDUMP}" -d --mcpu=pwr10 "${WORKDIR}/${obj_name}" 2>/dev/null | grep -ci "${instr_grep}" || true)"
  echo "    ${instr_grep}* instruction count: ${count}"
  if [[ "${count}" -gt 0 ]]; then
    echo "    PASS: ${label} compiled down to real Power10 MMA instructions."
  else
    echo "    FAIL: no ${instr_grep} instructions found -- MMA path was not used!" >&2
    return 1
  fi
}

verify_mma "${WORKDIR}/matmul_f32.mlir"  matmul_f32.o  gerpp     "f32 matmul (xvf32gerpp)"
verify_mma "${WORKDIR}/matmul_bf16.mlir" matmul_bf16.o bf16ger2pp "bf16 matmul (xvbf16ger2pp)"

echo "==> All checks passed."
