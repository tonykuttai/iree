---
hide:
  - tags
tags:
  - CPU
icon: octicons/cpu-16
---

# PowerPC64 little-endian (ppc64le) cross-compilation

Running on a platform like ppc64le (PowerPC64 little-endian Linux, e.g. IBM
POWER8/POWER9/POWER10 systems) involves cross-compiling from a _host_ platform
(e.g. x86_64 Linux) to the ppc64le _target_ platform:

* IREE's _compiler_ is built on the host and is used there to generate modules
  for the target
* IREE's _runtime_ is built on the host for the target. The runtime is then
  pushed to the target to run natively.

## :octicons-download-16: Prerequisites

### Host environment setup

You should already be able to build IREE from source on your host platform.
Please make sure you have followed the [getting started](./getting-started.md)
steps.

### Install a ppc64le cross-compile toolchain and emulator

You'll need a `clang`/LLVM toolchain capable of targeting
`powerpc64le-unknown-linux-gnu` (clang's multi-target support means a single
`clang`/`clang++`/`lld` install is sufficient — no separate cross-`gcc` is
required, as long as `LLVM_TARGETS_TO_BUILD`/`IREE_DEFAULT_CPU_LLVM_TARGETS`
includes `PowerPC`), plus a ppc64le `sysroot` (e.g. extracted from a ppc64le
Linux distribution package set, or built with
[crosstool-NG](https://github.com/crosstool-ng/crosstool-ng)), and a ppc64le
[QEMU](https://www.qemu.org/) user-mode or system emulator if you don't have
physical ppc64le hardware.

There is currently no IREE-provided bootstrap script for a ppc64le toolchain
(unlike `build_tools/riscv/riscv_bootstrap.sh` for RISC-V) — point
`PPC64LE_TOOLCHAIN_ROOT` (see below) at whatever clang+sysroot you have
available.

## :octicons-sliders-16: Configure and build

### Host configuration

Build and install on your host machine:

``` shell
cmake -GNinja -B ../iree-build/ \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_INSTALL_PREFIX=../iree-build/install \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  .
cmake --build ../iree-build/ --target install
```

### Target configuration

The following instructions show how to build for ppc64le Linux. See
[linux_ppc64le.cmake](https://github.com/iree-org/iree/blob/main/build_tools/cmake/linux_ppc64le.cmake)
for the toolchain file, and
[build_ppc64le.sh](https://github.com/iree-org/iree/blob/main/build_tools/cmake/build_ppc64le.sh)
for a CI-style wrapper script that drives the same configuration.

#### ppc64le Linux target

```shell
cmake -GNinja -B ../iree-build-ppc64le/ \
  -DCMAKE_TOOLCHAIN_FILE="./build_tools/cmake/linux_ppc64le.cmake" \
  -DIREE_HOST_BIN_DIR=$(realpath ../iree-build/install/bin) \
  -DIREE_BUILD_COMPILER=OFF \
  -DPPC64LE_TOOLCHAIN_ROOT=/path/to/ppc64le/toolchain \
  .
cmake --build ../iree-build-ppc64le/
```

`PPC64LE_TOOLCHAIN_ROOT` is expected to contain `bin/clang`, `bin/clang++`,
`bin/llvm-ar`, `bin/llvm-ranlib`, `bin/llvm-strip`, and a `sysroot/` directory
with ppc64le glibc headers/libraries — the same layout used by the RISC-V
toolchain (`RISCV_TOOLCHAIN_ROOT`).

The default `-mcpu` for the toolchain is `power8` (the architectural floor for
mainstream ppc64le Linux distributions). Override `PPC64LE_COMPILER_FLAGS` at
configure time (e.g. `-DPPC64LE_COMPILER_FLAGS="-mcpu=power10 -mabi=elfv2"`) to
target newer cores, including Power10's MMA (Matrix-Multiply Assist) unit.

## :octicons-code-16: Running IREE bytecode modules on the ppc64le system

Set the path to a ppc64le Linux QEMU emulator binary in the `QEMU_BIN`
environment variable:

```shell
export QEMU_BIN=<path to qemu-ppc64le binary>
```

Invoke the host compiler tools to produce a bytecode module FlatBuffer,
targeting ppc64le explicitly. Note that `--iree-llvmcpu-target-cpu` is resolved
directly against LLVM's `MCSubtargetInfo` rather than going through Clang's
driver-level CPU name aliasing, so it must use LLVM's canonical PowerPC
processor name (`pwr8`/`pwr9`/`pwr10`), not Clang's `-mcpu=power8`-style alias:

``` shell
../iree-build/install/bin/iree-compile \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=powerpc64le-unknown-linux-gnu \
  --iree-llvmcpu-target-cpu=pwr8 \
  samples/models/simple_abs.mlir \
  -o /tmp/simple_abs_cpu.vmfb
```

Run under ppc64le emulation. QEMU's `-cpu` flag uses a third, independent
naming scheme from both LLVM's and Clang's — confirm the exact spelling with
`qemu-ppc64le -cpu help` before relying on the example below:

```shell
${QEMU_BIN} \
  -cpu POWER10 \
  -L ${PPC64LE_TOOLCHAIN_ROOT}/sysroot/ \
  ../iree-build-ppc64le/tools/iree-run-module \
  --device=local-task \
  --module=/tmp/simple_abs_cpu.vmfb \
  --function=abs \
  --input=f32=-5
```

## Running tests

[build_tools/cmake/test_ppc64le.sh](https://github.com/iree-org/iree/blob/main/build_tools/cmake/test_ppc64le.sh)
runs the runtime and `tests/e2e` CTest suites against a `build-ppc64le`
directory produced by `build_ppc64le.sh`, using
[run_ppc64le_test.sh](https://github.com/iree-org/iree/blob/main/build_tools/cmake/run_ppc64le_test.sh)
(set `QEMU_BIN`/`QEMU_CPU_FLAGS`, mirroring the RISC-V `run_riscv_test.sh`
convention) to wrap each test binary in QEMU. Tests that are not yet validated
on ppc64le can be tagged with the `noppc64le` label to exclude them.

## Status and known gaps

* **CPU codegen and the runtime build/link** for `powerpc64le-unknown-linux-gnu`
  are supported through the LLVMCPU backend, the `IREE_ARCH_PPC_64` runtime
  platform macro (`runtime/src/iree/base/target_platform.h`, little-endian
  only), and a little-endian ELF loader/relocator
  (`runtime/src/iree/hal/local/elf/arch/ppc_64.c`).
* **A hand-written `mmt4d` microkernel exists for Power10 MMA**
  (`runtime/src/iree/builtins/ukernel/arch/ppc_64/`), using
  `__builtin_mma_xvf32gerpp` et al. (`<altivec.h>`) for `f32f32f32`, with a
  fixed M0=4, N0=4, K0=1 tile matching the instruction's fixed 4x4
  outer-product width. Validated against the generic reference
  implementation on real POWER10 hardware (`mmt4d_test`), including the
  accumulate path. Note `__builtin_mma_assemble_acc`'s operand order is
  reversed relative to `__builtin_mma_disassemble_acc`'s output order — easy
  to get backwards, and the cause of a real bug caught during validation.
  Power8/Power9 (no MMA) and all other dtypes (int8, bf16, f16) still fall
  back to generic vectorized codegen, not a tuned microkernel — extending
  tile coverage is the natural next step. There is still no `ppc_mma` MLIR
  dialect or vector-to-MMA lowering at the compiler IR level; this ukernel is
  the only path to MMA acceleration today.
* **Validated on real POWER10 Linux hardware** (not just QEMU): a full-pipeline
  `iree-compile`/`iree-run-module` round trip for `linalg.matmul` produces
  numerically correct results, both with `--iree-llvmcpu-target-cpu=host` and
  with an explicit `--iree-llvmcpu-target-triple=powerpc64le-unknown-linux-gnu
  --iree-llvmcpu-target-cpu=pwr10`. The native-object path
  (`--iree-llvmcpu-link-embedded=false` +
  `--iree-llvmcpu-static-library-output-path`) emits a correct
  `ELF 64-bit LSB relocatable, 64-bit PowerPC, OpenPOWER ELF V2 ABI` object on
  both that hardware and when cross-compiled from a non-ppc64 host.
* **`runtime/src/iree/hal/local/elf/elf_module_test` cannot pass today**: there
  is no checked-in `elementwise_mul_ppc_64.so` fixture in
  `runtime/src/iree/hal/local/elf/testdata/` (every other supported arch has
  one), so the test correctly builds but skips with "no architecture-specific
  ELF binary embedded." **This is not a ppc64-specific gap** — it is blocked by
  a general regression in `testdata/generate.sh`'s fixture-regeneration path
  that reproduces identically on `x86_64`:
  `iree-compile --compile-mode=hal-executable` fails to lower
  `testdata/elementwise_mul.mlir` with `'iree_codegen.workgroup_count_hint' op
  failed to resolve workgroup count hint in terms of workload ordinals`.
  Root cause: `ResolveWorkgroupCountHintsPass`
  (`compiler/src/iree/compiler/Codegen/Common/ResolveWorkgroupCountHints.cpp`)
  resolves the hint by walking the callgraph from a host-side dispatch
  call-site, but `--compile-mode=hal-executable` compiles a standalone,
  hand-authored `hal.executable.source` with no such call-site (the file's own
  header comment notes this: "linking and multi-executable embedding support
  requires our host-side IR"). The tests that exercise the same op and still
  pass (`compiler/plugins/target/LLVMCPU/test/smoketest_{embedded,system}.mlir`)
  go through the full `stream.executable` → HAL pipeline instead, which does
  have a call-site. The existing checked-in `.so` fixtures for every other arch
  only still work because they are pre-baked artifacts from an older compiler
  — `generate.sh` cannot currently regenerate *any* of them against this
  repository's `HEAD`. Fixing this properly means rewriting
  `elementwise_mul.mlir` to the `stream.executable` idiom and regenerating
  fixtures for every supported architecture, which is out of scope here since
  it touches shared, cross-platform test infrastructure this investigation
  did not set out to change.
* **Two runtime unit tests fail on the POWER10 Linux box tested against**,
  root cause not yet determined (could be architecture-specific or particular
  to that machine's kernel/NUMA configuration — not confirmed either way):
  `iree/base/internal/shm_test` (`SealWrite*` cases get `PERMISSION_DENIED` on
  `mmap(PROT_READ)` after a memfd write-seal) and
  `iree/base/threading/numa_test` (`BindMemoryBasic` gets `INVALID_ARGUMENT`
  from `mbind()` targeting NUMA node 0). `iree/base/internal/fpu_state_test`
  also fails if run unfiltered, but that one is expected: it is labeled
  `requires-dtz` (IEEE denormals-are-zero flush control), which PowerPC does
  not implement the same way x86/ARM do, and `test_ppc64le.sh` already
  excludes that label.
* **AIX (`powerpc64-ibm-aix`) is explicitly out of scope** of this
  configuration. AIX differs in object format (XCOFF, not ELF), default
  endianness (big-endian, not little-endian), and OS/runtime assumptions
  throughout the codebase; none of that is addressed here.
