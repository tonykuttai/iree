#!/bin/bash

# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Test the cross-compiled ppc64le (PowerPC64 little-endian) Linux targets.
#
# The desired build directory can be passed as the first argument. Otherwise, it
# uses the environment variable IREE_TARGET_BUILD_DIR, defaulting to
# "build-ppc64le". Designed for CI, but can be run manually. Expects to be run
# from the root of the IREE repository.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_TARGET_BUILD_DIR:-build-ppc64le}}"
PPC64LE_PLATFORM="${IREE_TARGET_PLATFORM:-linux}"
PPC64LE_ARCH="${IREE_TARGET_ARCH:-ppc_64}"

export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-$(nproc)}

ctest_args=(
  "--timeout 900"
  "--output-on-failure"
  "--no-tests=error"
)

declare -a label_exclude_args=(
  "^nodocker$"
  "^driver=vulkan$"
  "^driver=metal$"
  "^driver=cuda$"
  "^driver=hip$"
  "^vulkan_uses_vk_khr_shader_float16_int8$"
  "^requires-filesystem$"
  "^requires-dtz$"
  "^noppc64le$"
)

declare -a test_exclude_args=()

# Test runtime unit tests
runtime_label_exclude_regex="($(IFS="|" ; echo "${label_exclude_args[*]}"))"
runtime_ctest_args=(
  "--test-dir ${BUILD_DIR}/runtime/"
  ${ctest_args[@]}
  "--label-exclude ${runtime_label_exclude_regex}"
)
echo "******** Running runtime CTest ********"
ctest ${runtime_ctest_args[@]}

tests_label_exclude_regex="($(IFS="|" ; echo "${label_exclude_args[*]}"))"
test_ctest_args=(
  "--test-dir ${BUILD_DIR}/tests/e2e"
  ${ctest_args[@]}
  "--label-exclude ${tests_label_exclude_regex}"
)
if [[ "${#test_exclude_args[@]}" -gt 0 ]]; then
  tests_exclude_regex="($(IFS="|" ; echo "${test_exclude_args[*]}"))"
  test_ctest_args+=("--exclude-regex ${tests_exclude_regex}")
fi
echo "******** Running e2e CTest ********"
ctest  ${test_ctest_args[@]}
