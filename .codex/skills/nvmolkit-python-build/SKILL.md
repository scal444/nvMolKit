---
name: nvmolkit-python-build
description: Build and test nvMolKit Python bindings via skbuild and setup.py, and run pytest in this repository. Use when building or installing the Python package, running pytest against installed nvmolkit modules or pure-Python benchmark/autotune tests, resolving the right conda env, debugging compiled extension import errors, or reproducing production-style Python builds.
---

# nvMolKit Python Build And Test

Imported from `~/omg/repos/nvmolkit/.cursor/skills/nvmolkit-python-build/SKILL.md`, with project rules folded in.

## Environment

Conda env naming rule:

- If `git rev-parse --show-toplevel` is under `~/.cursor/worktrees/`, use `nvmolkit-py-<WORKTREE_ID>`, where `<WORKTREE_ID>` is the directory two levels under `~/.cursor/worktrees/<WORKTREE_ID>/<repo-key>`.
- Otherwise use `rdcu_dev`. Do not create a per-checkout env for the main working tree.

Resolve the env name at the start of Python build/test sessions:

```bash
SRC="$(git rev-parse --show-toplevel)"
if [[ "$SRC" == "$HOME/.cursor/worktrees/"* ]]; then
  WORKTREE_ID="$(echo "$SRC" | sed -E "s|^$HOME/.cursor/worktrees/([^/]+)/.*|\1|")"
  ENV_NAME="nvmolkit-py-$WORKTREE_ID"
else
  ENV_NAME="rdcu_dev"
fi
echo "ENV_NAME=$ENV_NAME"
```

If the env does not exist, create it from scratch:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" -y -c conda-forge \
    python=3.13 \
    cmake \
    rdkit librdkit-dev \
    libboost-devel libboost-headers libboost-python-devel \
    eigen \
    pytest pandas psutil \
    setuptools wheel pip
  conda activate "$ENV_NAME"
  pip install "scikit-build>=0.18" "ruff==0.15.8" numpy torch triton
else
  conda activate "$ENV_NAME"
fi
```

## Python Import Placement

Keep Python imports at module top with the rest of the imports. Only import inside functions, branches, or `try` blocks when runtime probing, delayed heavy dependency loading, or an unavoidable import cycle requires it.

Do not add CUDA-availability probes or CPU fallbacks just to run without CUDA. nvMolKit requires CUDA.

## Build And Install

The canonical path is `pip install .` from the worktree root. Editable installs do not work because skbuild editable mode skips CMake.

Always tee the full build output:

```bash
cd <worktree-root>
BUILD_LOG="$(mktemp -t nvmolkit-pybuild-XXXXXX.log)"
echo "BUILD_LOG=$BUILD_LOG"
CMAKE_BUILD_PARALLEL_LEVEL="$(( $(nproc) / 2 ))" \
NVMOLKIT_CUDA_TARGET_MODE=native \
  pip install . --no-deps --no-build-isolation -v 2>&1 | tee "$BUILD_LOG"
```

- Use `CMAKE_BUILD_PARALLEL_LEVEL` so scikit-build uses parallel CMake builds.
- Use `NVMOLKIT_CUDA_TARGET_MODE=native` for normal Python builds unless the user requests an explicit arch or release matrix.
- Use `--no-build-isolation` so pip uses the active env's `scikit-build`.
- Use `--no-deps`; runtime deps are already in the env.
- `setup.py` passes `-DNVMOLKIT_BUILD_PYTHON_BINDINGS=ON`, `-DNVMOLKIT_BUILD_TESTS=OFF`, `-DNVMOLKIT_BUILD_BENCHMARKS=OFF`, `-DCMAKE_BUILD_TYPE=Release`, and `-DCMAKE_PREFIX_PATH=$CONDA_PREFIX`.

If a build fails, search the log instead of rebuilding:

```bash
rg -n "error:|fatal error:|undefined reference|ld: " "$BUILD_LOG" | head
rg -n "CMake Error|CMake Warning" "$BUILD_LOG"
```

## Run Pytest Against The Installed Package

Run pytest through the resolved conda env. Do not run pytest from the worktree root: the source `nvmolkit/` package lacks compiled `.so` files and can shadow the installed package. Run from `/tmp` or another neutral cwd:

```bash
cd /tmp
python -m pytest <worktree-root>/nvmolkit/tests --tb=short
```

- Expected on a single-GPU host is roughly hundreds of passed tests and some multi-GPU skips.
- Known flaky: `test_types.py::test_async_gpu_result_release_frees_memory`.

## Smoke Test Imports

Run from outside the worktree root:

```bash
python -c "
import importlib
mods = ['_DataStructs', '_Fingerprints', '_arrayHelpers', '_embedMolecules',
        '_mmffOptimization', '_uffOptimization',
        '_batchedForcefield', '_clustering',
        '_conformerRmsd', '_substructure', '_mcs', '_TFD']
for m in mods:
    importlib.import_module('nvmolkit.' + m)
    print(m, 'OK')
"
```

Add newly introduced native modules to this smoke list while working on Python bindings.

## Common Failure Modes

- `ImportError: cannot import name '_batchedForcefield' from 'nvmolkit'` usually means Python is running from the worktree root and seeing the source package without compiled extensions. Change to `/tmp`.
- `ModuleNotFoundError: No module named 'skbuild'` means wrong env or missing `scikit-build`.
- `pip install -e .` succeeding quickly and producing no `.so` files is skbuild editable mode doing nothing. Use `pip install .`.

## Do Not

- Do not add `pip install -e .` instructions.
- Do not add CUDA-less fallback paths or tests skipped solely because CUDA is unavailable.
