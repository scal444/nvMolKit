#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Drive cibuildwheel across the full (rdkit, python) matrix from
# admin/distribute/rdkit_build_matrix.yaml in N parallel jobs, each pinned to
# THREADS_PER_JOB compile threads.
#
# Each job runs in its own copy of the source tree (so cibuildwheel's per-build
# _skbuild/ and CMakeCache.txt don't collide across parallel runs) and writes
# its wheel under wheelhouse/rdkit<RDKIT>/py<PY>/.
#
# Usage:
#   bash admin/deploy/build_full_matrix.sh [JOBS] [THREADS_PER_JOB]
#     JOBS              default 8
#     THREADS_PER_JOB   default 2
#
# Environment overrides:
#   CIBW_MANYLINUX_X86_64_IMAGE   default ghcr.io/.../nvmolkit-manylinux-cuda12:local
#   WORKTREE_ROOT                 default /home/kevin/scratch/nvmolkit_wheels
#   WHEELHOUSE                    default <repo>/wheelhouse

set -euo pipefail

JOBS=${1:-8}
THREADS_PER_JOB=${2:-2}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

WHEELHOUSE=${WHEELHOUSE:-$REPO/wheelhouse}
WORKTREE_ROOT=${WORKTREE_ROOT:-$HOME/scratch/nvmolkit_wheels}
LOG_DIR=$WHEELHOUSE/logs
JOB_DIR=$WHEELHOUSE/jobs

mkdir -p "$WHEELHOUSE" "$LOG_DIR" "$JOB_DIR" "$WORKTREE_ROOT"

export CIBW_MANYLINUX_X86_64_IMAGE=${CIBW_MANYLINUX_X86_64_IMAGE:-ghcr.io/nvidia-digital-bio/nvmolkit-manylinux-cuda12:local}
export CMAKE_BUILD_PARALLEL_LEVEL=$THREADS_PER_JOB
export MAKEFLAGS=-j$THREADS_PER_JOB
export CONAN_CPU_COUNT=$THREADS_PER_JOB

# cibuildwheel itself only needs to launch docker; the actual builds run
# inside the manylinux image. Expect the caller to have already activated
# a conda env that provides cibuildwheel (or to have it on PATH).
if ! command -v cibuildwheel >/dev/null 2>&1; then
    echo "Error: cibuildwheel not found on PATH." >&2
    echo "       Activate the conda env that supplies it first, e.g." >&2
    echo "       'conda activate nvmolkit_pip_build'." >&2
    exit 1
fi

# Snapshot the working tree's pyproject.toml so each worktree picks up local
# (uncommitted) changes like the extended environment-pass list.
PYPROJECT_SRC=$REPO/pyproject.toml

# Enumerate (rdkit_version, python) pairs from the matrix YAML.
# Avoid a pyyaml dependency by walking the file structurally: each
# top-level "<rdkit>": entry has exactly one python_versions: list.
#
# rdkit-pypi tags 2025.3.1 through 2025.3.5 use conan-1 invocation syntax
# in their setup.py (e.g. `conan export ... 1.85.0@chris/mod_boost`), which
# the manylinux+CUDA image's conan>=2 cannot parse. Skip those tags here;
# the cutover to conan-2 syntax happened at the 2025.03.6 tag.
PAIRS_FILE=$JOB_DIR/pairs.txt
awk '
    /^"[0-9]+\.[0-9]+\.[0-9]+":/ {
        gsub(/[":]/, "", $1)
        rdkit = $1
        skip = (rdkit == "2025.3.1" || rdkit == "2025.3.2" || \
                rdkit == "2025.3.3" || rdkit == "2025.3.4" || \
                rdkit == "2025.3.5")
    }
    /^[[:space:]]+python_versions:/ {
        if (skip) next
        sub(/.*\[/, "")
        sub(/\].*/, "")
        gsub(/[",]/, "")
        for (i = 1; i <= NF; i++) print rdkit, $i
    }
' admin/distribute/rdkit_build_matrix.yaml > "$PAIRS_FILE"

NUM_PAIRS=$(wc -l < "$PAIRS_FILE")
# Use the awk-comparable form (no %Z) for filtering rows in this run; keep a
# separate human-readable form including the timezone for the run header.
RUN_STARTED_AT=$(date '+%Y-%m-%d %H:%M:%S')
RUN_STARTED_DISPLAY=$(date '+%Y-%m-%d %H:%M:%S %Z')
echo "Run started: $RUN_STARTED_DISPLAY"
echo "Matrix has $NUM_PAIRS (rdkit, python) pairs."
echo "Running $JOBS jobs in parallel, $THREADS_PER_JOB threads each."
echo "Wheelhouse: $WHEELHOUSE"
echo "Worktrees:  $WORKTREE_ROOT"
echo

# Per-pair timing log. One row per (rdkit, py) attempt. Concurrent appends
# from multiple xargs workers are safe: each printf is well under PIPE_BUF
# bytes and the file is opened O_APPEND, so writes are atomic on Linux.
TIMINGS_TSV=$JOB_DIR/timings.tsv
if [ ! -f "$TIMINGS_TSV" ]; then
    printf 'rdkit\tpy\tstatus\tstarted_at\tended_at\telapsed_sec\n' > "$TIMINGS_TSV"
fi
export TIMINGS_TSV

# Per-pair worker lives in its own script (admin/deploy/build_one_wheel.sh)
# so xargs can fork+exec it cleanly. We deliberately don't use bash's
# exported-function pattern: serializing a function body that contains
# nested $(...) and $((...)) breaks under some bash versions when xargs
# re-execs `bash -c '<func> "$@"'`.
WORKER=$REPO/admin/deploy/build_one_wheel.sh
if [ ! -x "$WORKER" ]; then
    echo "Error: worker script not executable at $WORKER" >&2
    exit 1
fi

export REPO WHEELHOUSE LOG_DIR WORKTREE_ROOT PYPROJECT_SRC TIMINGS_TSV

# Hand each pair to xargs as two args ($1=rdkit, $2=py). Don't let a single
# failed build kill the whole matrix - we tally results at the end.
START=$(date +%s)
set +e
xargs -a "$PAIRS_FILE" -P "$JOBS" -L 1 "$WORKER"
RC=$?
set -e
END=$(date +%s)
WALL=$((END - START))

echo
echo "All jobs finished. Wall time: ${WALL}s ($(printf '%dh%02dm%02ds' $((WALL/3600)) $(((WALL%3600)/60)) $((WALL%60))))"
echo

# Summary: tally success/failure by inspecting wheelhouse.
total=0
ok=0
missing=()
while read -r rdkit py; do
    total=$((total + 1))
    if compgen -G "$WHEELHOUSE/rdkit${rdkit}/py${py}/*.whl" > /dev/null; then
        ok=$((ok + 1))
    else
        missing+=("rdkit=$rdkit py=$py")
    fi
done < "$PAIRS_FILE"

echo "Wheels built: $ok / $total"
if [ ${#missing[@]} -gt 0 ]; then
    echo "Missing:"
    for m in "${missing[@]}"; do
        echo "  $m"
    done
fi

# Per-build timing summary from the TSV. Reads only this run's rows
# (started_at >= RUN_STARTED_AT). Output: count, sum, avg, min, p50, max
# of elapsed_sec across rows in {ok, fail}, and a sorted-slowest table.
echo
echo "Timings (this run):"
awk -F'\t' -v run_start="$RUN_STARTED_AT" -v wall="$WALL" '
    NR == 1 { next }
    $4 >= run_start && ($3 == "ok" || $3 == "fail") {
        rows[++n] = $0
        elapsed[n] = $6 + 0
        rdkit[n] = $1; py[n] = $2; status[n] = $3
    }
    END {
        if (n == 0) { print "  (no completed builds)"; exit }
        sum = 0; mn = elapsed[1]; mx = elapsed[1]
        for (i = 1; i <= n; i++) {
            sum += elapsed[i]
            if (elapsed[i] < mn) mn = elapsed[i]
            if (elapsed[i] > mx) mx = elapsed[i]
        }
        # sort copy for median
        for (i = 1; i <= n; i++) sorted[i] = elapsed[i]
        for (i = 2; i <= n; i++) {
            v = sorted[i]; j = i
            while (j > 1 && sorted[j-1] > v) { sorted[j] = sorted[j-1]; j-- }
            sorted[j] = v
        }
        med = (n % 2) ? sorted[(n+1)/2] : (sorted[n/2] + sorted[n/2+1]) / 2
        printf "  builds=%d  sum=%ds  avg=%ds  min=%ds  p50=%ds  max=%ds\n", \
               n, sum, sum/n, mn, med, mx
        printf "  speedup vs serial: %.2fx (sum/wall, wall=%ds)\n", sum / wall, wall
        printf "\n  Slowest builds:\n"
        # Pair (elapsed, idx) and sort descending. Bubble-print top 10.
        for (i = 1; i <= n; i++) ord[i] = i
        for (i = 2; i <= n; i++) {
            v = ord[i]; j = i
            while (j > 1 && elapsed[ord[j-1]] < elapsed[v]) { ord[j] = ord[j-1]; j-- }
            ord[j] = v
        }
        m = (n < 10) ? n : 10
        for (i = 1; i <= m; i++) {
            k = ord[i]
            printf "    %-9s py%s  %5ds  %s\n", rdkit[k], py[k], elapsed[k], status[k]
        }
    }
' "$TIMINGS_TSV"

echo
echo "Per-pair TSV: $TIMINGS_TSV"

exit $RC
