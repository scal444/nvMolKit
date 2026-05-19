#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Per-pair worker for build_full_matrix.sh. Builds one (rdkit, python) wheel
# in a fresh copy of the source tree, with its own conan2 cache so concurrent
# invocations for the same python don't race on shared boost recipe refs.
#
# Usage: build_one_wheel.sh <rdkit_version> <python_version>
#
# Inputs read from environment (set by build_full_matrix.sh):
#   REPO              : repo root to copy from
#   WHEELHOUSE        : output dir; wheel ends up at <wheelhouse>/rdkit<X>/py<Y>/
#   LOG_DIR           : per-pair log dir
#   WORKTREE_ROOT     : where to create the throwaway source-tree copy
#   PYPROJECT_SRC     : path to pyproject.toml to copy into the worktree
#                       (so local uncommitted changes apply)
#   TIMINGS_TSV       : append-only timings log (rdkit\tpy\tstatus\tstart\tend\telapsed_sec)
#
# Exits 0 on success or skip, non-zero on build failure. All build output is
# captured to the per-pair log; only one-line status goes to stdout.

set -uo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <rdkit_version> <python_version>" >&2
    exit 2
fi

rdkit=$1
py=$2

: "${REPO:?REPO must be set}"
: "${WHEELHOUSE:?WHEELHOUSE must be set}"
: "${LOG_DIR:?LOG_DIR must be set}"
: "${WORKTREE_ROOT:?WORKTREE_ROOT must be set}"
: "${PYPROJECT_SRC:?PYPROJECT_SRC must be set}"
: "${TIMINGS_TSV:?TIMINGS_TSV must be set}"

pyTag=${py//./}
outDir=$WHEELHOUSE/rdkit${rdkit}/py${py}
logFile=$LOG_DIR/rdkit${rdkit}_py${py}.log
wt=$WORKTREE_ROOT/wt_rdkit${rdkit}_py${py}

mkdir -p "$outDir"

startEpoch=$(date +%s)
startedAt=$(date '+%Y-%m-%d %H:%M:%S')

format_hms() {
    local secs=$1
    printf '%dh%02dm%02ds' $((secs / 3600)) $(((secs % 3600) / 60)) $((secs % 60))
}

if compgen -G "$outDir/*.whl" > /dev/null; then
    echo "[skip] rdkit=$rdkit py=$py (wheel already at $outDir)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$rdkit" "$py" "skip" "$startedAt" "$startedAt" 0 >> "$TIMINGS_TSV"
    exit 0
fi

echo "[start $startedAt] rdkit=$rdkit py=$py worktree=$wt"

rm -rf "$wt"
mkdir -p "$wt"
rsync -a \
    --exclude '.git' \
    --exclude 'wheelhouse' \
    --exclude 'wheelhouse.backup' \
    --exclude '_skbuild' \
    --exclude 'build' \
    "$REPO"/ "$wt"/ > "$logFile" 2>&1
cp "$PYPROJECT_SRC" "$wt/pyproject.toml"

{
    echo "=== nvmolkit wheel build ==="
    echo "    rdkit=$rdkit py=$py"
    echo "    started_at=$startedAt"
    echo "    worktree=$wt"
    echo "============================"
} >> "$logFile"

set +e
(
    set -e
    cd "$wt"
    export CIBW_BUILD=cp${pyTag}-manylinux_x86_64
    # rdkit_recipe and pip caches are multi-process safe and shared per
    # python; conan2 is not concurrency-safe so each (rdkit, py) pair gets
    # its own conan cache to avoid races on shared recipe refs.
    export NVMOLKIT_CACHE_ROOT=$HOME/.cache/nvmolkit/py${py}
    export NVMOLKIT_CONAN_CACHE_ROOT=$HOME/.cache/nvmolkit/conan2/rdkit${rdkit}_py${py}
    bash admin/deploy/build_pip_wheels.sh "$rdkit" "$outDir"
) >> "$logFile" 2>&1
rc=$?
set -e

endEpoch=$(date +%s)
endedAt=$(date '+%Y-%m-%d %H:%M:%S')
elapsed=$((endEpoch - startEpoch))
elapsedHms=$(format_hms "$elapsed")

{
    echo
    echo "=== build finished ==="
    echo "    rdkit=$rdkit py=$py rc=$rc"
    echo "    ended_at=$endedAt"
    echo "    elapsed=${elapsed}s ($elapsedHms)"
    echo "======================"
} >> "$logFile"

rm -rf "$wt"

if [ "$rc" -eq 0 ]; then
    status=ok
    printf '[ok %s] rdkit=%s py=%s elapsed=%s\n' \
        "$endedAt" "$rdkit" "$py" "$elapsedHms"
else
    status=fail
    printf '[FAIL %s] rdkit=%s py=%s rc=%d elapsed=%s (see %s)\n' \
        "$endedAt" "$rdkit" "$py" "$rc" "$elapsedHms" "$logFile"
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$rdkit" "$py" "$status" "$startedAt" "$endedAt" "$elapsed" >> "$TIMINGS_TSV"

exit "$rc"
