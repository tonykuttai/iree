#!/bin/bash

# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Wrapper script to run the artifact on ppc64le (PowerPC64 little-endian) Linux.
# This script checks if a QEMU emulator is set, and uses either the emulator or
# the actual device to run the cross-compiled ppc64le linux artifacts.
#
# CMake always passes a leading "-L <sysroot>" (the QEMU sysroot flag) ahead of
# the actual test binary and its args, regardless of whether QEMU is in use.
# Consume it here so the on-device/native (no QEMU_BIN) path execs the real
# binary instead of treating "-L" itself as the command to run.

set -x
set -e

if [[ "$1" == "-L" ]]; then
  QEMU_SYSROOT="$2"
  shift 2
fi

# A QEMU Linux emulator must be available within the system that matches the
# processor architecture.
if [[ ! -z "${QEMU_BIN}" ]] && [[ ! -z "${QEMU_CPU_FLAGS}" ]]; then
  "${QEMU_BIN}" "-cpu" "${QEMU_CPU_FLAGS}" ${QEMU_SYSROOT:+-L "${QEMU_SYSROOT}"} "$@"
else
  # Running directly on ppc64le hardware: the sysroot flag is QEMU-only and is
  # already stripped above.
  "$@"
fi
