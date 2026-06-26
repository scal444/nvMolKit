#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run installed-wheel tests across nvMolKit wheel matrix entries.
#
# Usage:
#   bash admin/test/test_all_wheels.sh <wheelhouse_dir>
#   bash admin/test/test_all_wheels.sh <wheelhouse_dir> <pairs_file>
#   bash admin/test/test_all_wheels.sh <wheelhouse_dir> <rdkit_version> <python_version>
#
# With only <wheelhouse_dir>, every wheel found under rdkit*/py*/ is tested.
# A pairs file contains one "<rdkit_version> <python_version>" pair per line;
# blank lines and '#' comments are ignored.

set -uo pipefail

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <wheelhouse_dir> [pairs_file]" >&2
    echo "       $0 <wheelhouse_dir> <rdkit_version> <python_version>" >&2
    exit 2
fi

WHEELHOUSE=$(cd "$1" 2>/dev/null && pwd) || {
    echo "Error: wheelhouse_dir '$1' is not a readable directory" >&2
    exit 2
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
WORKER=$SCRIPT_DIR/test_one_wheel.sh

TEST_LOG_DIR=$WHEELHOUSE/test_logs
VENV_ROOT=${VENV_ROOT:-${TMPDIR:-/tmp}/nvmolkit_test_venvs}
IFACE_ENV_PREFIX=${IFACE_ENV_PREFIX:-nvmolkit_iface_}
TIMINGS_TSV=${TIMINGS_TSV:-$TEST_LOG_DIR/timings.tsv}

mkdir -p "$TEST_LOG_DIR" "$VENV_ROOT"
if [ ! -f "$TIMINGS_TSV" ]; then
    printf 'rdkit\tpy\tstatus\tstarted_at\tended_at\telapsed_sec\n' > "$TIMINGS_TSV"
fi

if [ ! -x "$WORKER" ]; then
    echo "Error: per-pair worker not executable at $WORKER" >&2
    exit 2
fi
if [ ! -d "$REPO/nvmolkit/tests" ]; then
    echo "Error: nvmolkit/tests not found under $REPO" >&2
    exit 2
fi

if [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ]; then
    CONDA_BIN=$CONDA_EXE
elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN=$(command -v conda)
else
    echo "Error: conda not found on PATH and CONDA_EXE is not set." >&2
    exit 2
fi

CONDA_BASE=$("$CONDA_BIN" info --base 2>/dev/null) || {
    echo "Error: '$CONDA_BIN info --base' failed" >&2
    exit 2
}
ENVS_ROOT=$CONDA_BASE/envs

tmp_files=()
cleanup() {
    rm -f "${tmp_files[@]}"
}
trap cleanup EXIT

make_tmp() {
    local f
    f=$(mktemp)
    tmp_files+=("$f")
    echo "$f"
}

discover_pairs() {
    local out=$1
    : > "$out"
    shopt -s nullglob
    local wheel_dir rdkit py matches
    for wheel_dir in "$WHEELHOUSE"/rdkit*/py*; do
        rdkit=$(basename "$(dirname "$wheel_dir")")
        rdkit=${rdkit#rdkit}
        py=$(basename "$wheel_dir")
        py=${py#py}
        matches=("$wheel_dir"/nvmolkit-*.whl)
        if [ ${#matches[@]} -gt 0 ]; then
            echo "$rdkit $py" >> "$out"
        fi
    done
    shopt -u nullglob
}

ensure_iface_env() {
    local pyver=$1
    local env_name=${IFACE_ENV_PREFIX}py${pyver}
    local env_python=$ENVS_ROOT/$env_name/bin/python
    if [ -x "$env_python" ]; then
        return 0
    fi
    echo "Creating interpreter env $env_name (python=$pyver)..."
    "$CONDA_BIN" create -y -n "$env_name" -c conda-forge "python=$pyver" >&2
}

raw_pairs=$(make_tmp)
case $# in
    1)
        discover_pairs "$raw_pairs"
        ;;
    2)
        [ -f "$2" ] || { echo "Error: pairs file not found: $2" >&2; exit 2; }
        cp "$2" "$raw_pairs"
        ;;
    3)
        printf '%s %s\n' "$2" "$3" > "$raw_pairs"
        ;;
esac

pairs=$(make_tmp)
grep -vE '^[[:space:]]*(#|$)' "$raw_pairs" > "$pairs" || true
if [ ! -s "$pairs" ]; then
    echo "Error: no wheel test pairs found" >&2
    exit 2
fi

py_versions=$(awk '{print $2}' "$pairs" | sort -u)
for py in $py_versions; do
    ensure_iface_env "$py" || exit 1
done

export REPO WHEELHOUSE TEST_LOG_DIR VENV_ROOT IFACE_ENV_PREFIX TIMINGS_TSV
export NVMOLKIT_CONDA_ENVS_ROOT=$ENVS_ROOT

started=$(date '+%Y-%m-%d %H:%M:%S %Z')
start_epoch=$(date +%s)
total=$(wc -l < "$pairs")
ok=0
fail=0
skip=0
failed_pairs=()

echo "Wheelhouse: $WHEELHOUSE"
echo "Pairs: $pairs ($total)"
echo "Started: $started"
echo "Logs: $TEST_LOG_DIR"

while read -r rdkit py _; do
    [ -n "$rdkit" ] || continue
    set +e
    "$WORKER" "$rdkit" "$py"
    rc=$?
    set -e

    if [ "$rc" -eq 0 ]; then
        if compgen -G "$WHEELHOUSE/rdkit${rdkit}/py${py}/nvmolkit-*.whl" > /dev/null; then
            ok=$((ok + 1))
        else
            skip=$((skip + 1))
        fi
    else
        fail=$((fail + 1))
        failed_pairs+=("rdkit=$rdkit py=$py")
    fi
done < "$pairs"

elapsed=$(( $(date +%s) - start_epoch ))
printf 'Wheel tests finished in %dh%02dm%02ds\n' \
    $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
echo "Results: ok=$ok fail=$fail skip=$skip total=$total"

if [ ${#failed_pairs[@]} -gt 0 ]; then
    echo "Failed:"
    for pair in "${failed_pairs[@]}"; do
        echo "  $pair"
    done
fi

if [ "$fail" -eq 0 ]; then
    echo "All requested wheel tests passed."
    exit 0
fi
echo "$fail pair(s) failed. See per-pair logs in $TEST_LOG_DIR."
exit 1
