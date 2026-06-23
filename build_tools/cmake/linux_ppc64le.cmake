# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26)

# CMake invokes the toolchain file twice during the first build, but only once
# during subsequent rebuilds. This was causing the various flags to be added
# twice on the first build, and on a rebuild ninja would see only one set of the
# flags and rebuild the world.
# https://github.com/android-ndk/ndk/issues/323
if(PPC64LE_TOOLCHAIN_INCLUDED)
  return()
endif()
set(PPC64LE_TOOLCHAIN_INCLUDED true)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR ppc64le)

if(NOT "${PPC64LE_TOOLCHAIN_ROOT}" STREQUAL "")
  set(CMAKE_AR           "${PPC64LE_TOOLCHAIN_ROOT}/bin/${PPC64LE_TOOLCHAIN_PREFIX}llvm-ar")
  set(CMAKE_C_COMPILER   "${PPC64LE_TOOLCHAIN_ROOT}/bin/${PPC64LE_TOOLCHAIN_PREFIX}clang")
  set(CMAKE_CXX_COMPILER "${PPC64LE_TOOLCHAIN_ROOT}/bin/${PPC64LE_TOOLCHAIN_PREFIX}clang++")
  set(CMAKE_RANLIB       "${PPC64LE_TOOLCHAIN_ROOT}/bin/${PPC64LE_TOOLCHAIN_PREFIX}llvm-ranlib")
  set(CMAKE_STRIP        "${PPC64LE_TOOLCHAIN_ROOT}/bin/${PPC64LE_TOOLCHAIN_PREFIX}llvm-strip")
  set(CMAKE_SYSROOT "${PPC64LE_TOOLCHAIN_ROOT}/sysroot")
endif()

# Power8 (-mcpu=power8) is the baseline ISA targeted by mainstream ppc64le
# Linux distributions (the "ppc64le" port floor). Override with
# -DPPC64LE_COMPILER_FLAGS=... at configure time to target a newer core (e.g.
# -mcpu=power10 to enable Power10 MMA codegen).
set(PPC64LE_COMPILER_FLAGS "-mcpu=power8 -mabi=elfv2"
    CACHE STRING "Compiler flags for the ppc64le toolchain.")
# NOTE: QEMU's PowerPC CPU model names are a third, independent naming scheme
# from both LLVM's ("pwr8") and Clang's ("power8") -mcpu spellings. Verify the
# exact spelling accepted by your qemu-ppc64le build with `qemu-ppc64le -cpu help`
# before relying on this default.
set(PPC64LE_QEMU_CPU_FLAGS "POWER10"
    CACHE STRING "QEMU -cpu flag for running ppc64le binaries.")
# --iree-llvmcpu-target-cpu is resolved directly against LLVM's MCSubtargetInfo
# (bypassing Clang's driver-level aliasing), so it must use LLVM's canonical
# PowerPC processor name ("pwr8"/"pwr9"/"pwr10"), not Clang's "power8" alias
# used in PPC64LE_COMPILER_FLAGS below.
set(PPC64LE_TEST_DEFAULT_LLVM_FLAGS
  "--iree-llvmcpu-target-triple=powerpc64le-unknown-linux-gnu"
  "--iree-llvmcpu-target-cpu=pwr8"
  CACHE INTERNAL "Default llvm codegen flags for testing purposes")

set(CMAKE_C_FLAGS_INIT   "${PPC64LE_COMPILER_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${PPC64LE_COMPILER_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${PPC64LE_COMPILER_FLAGS}")
