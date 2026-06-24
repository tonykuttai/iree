#!/bin/bash
# Build + run the real Power10 MMA tests natively, on a ppc64le Power10 Linux
# box. Run this script ON the remote machine itself (see the companion
# ppc64_mma_build_macos.sh for the cross-compile side, and the README
# section in the commit/PR description for how to get this script there).
#
# Usage (on the remote machine):
#   ./build_tools/scripts/ppc64_mma_build_remote.sh
#
# Env overrides:
#   IREE_SRC_DIR   (default: repo root, i.e. this script's grandparent dir)
#   IREE_BUILD_DIR (default: <IREE_SRC_DIR>/build)
#   CMAKE_BIN      (default: cmake on PATH, falling back to ~/.local/bin/cmake
#                   if the system cmake is too old -- IREE needs >= 3.26)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IREE_SRC_DIR="${IREE_SRC_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
IREE_BUILD_DIR="${IREE_BUILD_DIR:-${IREE_SRC_DIR}/build}"

if [[ -z "${CMAKE_BIN:-}" ]]; then
  if cmake --version 2>/dev/null | head -1 | grep -qE "version (3\.2[6-9]|3\.[3-9][0-9]|[4-9])"; then
    CMAKE_BIN="cmake"
  elif [[ -x "${HOME}/.local/bin/cmake" ]]; then
    CMAKE_BIN="${HOME}/.local/bin/cmake"
  else
    echo "No CMake >= 3.26 found. Set CMAKE_BIN explicitly." >&2
    exit 1
  fi
fi

echo "==> Source dir: ${IREE_SRC_DIR}"
echo "==> Build dir:  ${IREE_BUILD_DIR}"
echo "==> CMake:      ${CMAKE_BIN} ($(${CMAKE_BIN} --version | head -1))"
echo "==> nproc:      $(nproc)"

echo "==> Configuring (full compiler + runtime, native ppc64le)..."
"${CMAKE_BIN}" -G Ninja -B "${IREE_BUILD_DIR}" -S "${IREE_SRC_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREE_BUILD_COMPILER=ON \
  -DIREE_BUILD_TESTS=ON \
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON

# Confirm the MMA compiler-support probe actually passed -- if this is
# #undef, the host compiler couldn't compile __builtin_mma_* and the ukernel
# silently falls back to generic code with no error.
CONFIG_HEADER="${IREE_BUILD_DIR}/runtime/src/iree/builtins/ukernel/arch/ppc_64/config_ppc_64.h"
if [[ -f "${CONFIG_HEADER}" ]] && grep -q "^#define IREE_UK_BUILD_PPC_64_MMA" "${CONFIG_HEADER}"; then
  echo "==> IREE_UK_BUILD_PPC_64_MMA is enabled."
else
  echo "==> WARNING: IREE_UK_BUILD_PPC_64_MMA is NOT enabled (see ${CONFIG_HEADER})." >&2
fi

echo "==> Building iree-compile, iree-opt, and mmt4d_test..."
ninja -C "${IREE_BUILD_DIR}" -j "$(nproc)" \
  tools/iree-compile tools/iree-opt mmt4d_test

echo "==> Running mmt4d_test (real Power10 hardware execution)..."
"${IREE_BUILD_DIR}/runtime/src/iree/builtins/ukernel/tools/mmt4d_test"

echo "==> Running the full unit test suite (ctest)..."
(cd "${IREE_BUILD_DIR}" && ctest --output-on-failure -j "$(nproc)")

echo "==> All done."
