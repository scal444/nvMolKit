---
name: nvmolkit-cpp-build
description: Build and test the nvMolKit C++/CUDA tree with CMake and ctest. Use when configuring, building, running ctest, debugging C++/CUDA test failures, or running cmake-format, clang-format, and include-check helpers in this repository. Covers CMAKE_BUILD_TYPE, NVMOLKIT_CUDA_TARGET_MODE, explicit CUDA architectures, sanitizer builds, GPU test execution, and known-flaky tests.
---

# nvMolKit C++/CUDA Build And Test

Imported from `~/omg/repos/nvmolkit/.cursor/skills/nvmolkit-cpp-build/SKILL.md`, with project rules folded in.

## Environment

- Use the `rdcu_dev` conda env. It has `cmake>=3.30`, `ninja`, `cuda-toolkit`, the system C++ toolchain, `rdkit`, `boost`, OpenMP, and GTest via FetchContent.
- Activate before CMake, build, or ctest calls:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate rdcu_dev
```

## CUDA Is Required

Do not add CUDA-optional guards, CPU shims, or skip markers just to pass without CUDA. nvMolKit assumes CUDA and at least one GPU. If CUDA is missing, fail loudly.

Allowed runtime gates are only for real GPU properties beyond CUDA existence, such as multi-GPU count, peer access, or compute capability routing.

## Configure

Use a build dir under `${TMPDIR:-/tmp}` keyed off the current git toplevel basename so each worktree gets its own build and the source tree stays clean:

```bash
SRC="$(git rev-parse --show-toplevel)"
BUILD_DIR="${TMPDIR:-/tmp}/nvmolkit-build-$(basename "$SRC")"
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
cmake -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DNVMOLKIT_CUDA_TARGET_MODE=native \
      -DNVMOLKIT_EXTRA_DEV_FLAGS=OFF \
      "$SRC"
```

- Always pass `-DNVMOLKIT_EXTRA_DEV_FLAGS=OFF` for normal development builds.
- `NVMOLKIT_CUDA_TARGET_MODE=native` keeps build time short on a dev box. Use `full` only for release artifacts.
- If the user asks for an explicit architecture, prefer `-DCMAKE_CUDA_ARCHITECTURES=<arch>` and do not rely on `native`.
- This fMCS import task uses SM 89 explicitly.
- `CMAKE_BUILD_TYPE` accepts `Debug`, `RelWithDebInfo`, `Release`, and legacy sanitizer build types `asan`, `tsan`, `ubsan`.
- Prefer `-DNVMOLKIT_SANITIZER=asan` or equivalent over legacy sanitizer build types.

## Build

From the build dir:

```bash
make -j 8
```

Full first builds are roughly minutes; incremental edits are often tens of seconds. Reuse the same build dir between edits.

## Test

From the same build dir:

```bash
ctest -j 8 --output-on-failure
```

- GPU test invocations must run where the GPU is visible. In Codex sandboxed sessions, request escalation for `ctest`, individual GPU tests, and `nvidia-smi`.
- Test data is set by ctest as `NVMOLKIT_TESTDATA=<repo>/tests/test_data`.
- Known flaky under load: `*/CrossCpuSimilarityParamTestFixture.{Default,Constrained}MemoryAgrees/*` and `*/CrossCpuCosineSimilarityParamTestFixture.{Default,Constrained}MemoryAgrees/*`; rerun with:

```bash
ctest -R "CrossCpuSimilarity|CrossCpuCosine" --output-on-failure
```

## Common Failure Modes

- `fatal error: cuda_error_check.h: No such file or directory` or similar project headers usually means an include uses bare `"foo.h"` instead of project-rooted `"src/utils/foo.h"`. Run `bash admin/run_include_check.sh` to fix.
- Runtime search path warnings about conda `libgomp.so.1` versus system OpenMP are benign.

## Do Not

- Do not disable a failing test to make ctest green, aside from known flaky tests.
- Do not add CUDA-less fallbacks or `pytest.mark.skipif(not torch.cuda.is_available(), ...)` patterns.
