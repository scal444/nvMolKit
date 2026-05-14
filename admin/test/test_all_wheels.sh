#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Drive admin/test/test_one_wheel.sh across a list of (rdkit, python) pairs
# serially. Discovers conda automatically and auto-creates per-python
# interpreter envs as needed.
#
# Usage:
#   bash admin/test/test_all_wheels.sh <wheelhouse_dir> <smoke|full|both> [pairs_file]
#
#   wheelhouse_dir : directory containing rdkit<X>/py<Y>/nvmolkit-*.whl
#   mode = smoke   -> default pairs: every wheel discovered under wheelhouse_dir
#          full    -> default pairs: admin/test/full_test_subset.txt
#          both    -> smoke then full
#   pairs_file     : optional override; one "<rdkit> <py>" per line, '#' comments ok.
#                    Not allowed with mode=both.
#
# Per-pair output: <wheelhouse>/test_logs/rdkit<X>_py<Y>_<mode>.log
# Per-pair timing rows appended to: <wheelhouse>/test_logs/timings.tsv
#
# Environment overrides (none required):
#   VENV_ROOT          throwaway pip venv dir (default ${TMPDIR:-/tmp}/nvmolkit_test_venvs)
#   IFACE_ENV_PREFIX   prefix for per-python conda envs that supply the cpython
#                      interpreter (default nvmolkit_iface_).
#                      Final env name is "<prefix>py<X.Y>" e.g. nvmolkit_iface_py3.12.
#                      Auto-created via `conda create -c conda-forge python=X.Y`
#                      if missing.
#   TIMINGS_TSV        append-only timings tsv (default <wheelhouse>/test_logs/timings.tsv)

set -uo pipefail

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <wheelhouse_dir> <smoke|full|both> [pairs_file]" >&2
    exit 2
fi

WHEELHOUSE=$(cd "$1" 2>/dev/null && pwd) || {
    echo "Error: wheelhouse_dir '$1' is not a readable directory" >&2
    exit 2
}
MODE=$2
PAIRS_FILE_ARG=${3:-}

case "$MODE" in
    smoke|full|both) ;;
    *) echo "Error: mode must be 'smoke', 'full', or 'both', got '$MODE'" >&2; exit 2 ;;
esac

if [ "$MODE" = "both" ] && [ -n "$PAIRS_FILE_ARG" ]; then
    echo "Error: explicit pairs_file is not supported with mode=both" >&2
    exit 2
fi

# Repo root = parent of admin/test/. Works regardless of where the tree is
# mounted.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
WORKER=$SCRIPT_DIR/test_one_wheel.sh
SUBSET_FILE=$SCRIPT_DIR/full_test_subset.txt

if [ ! -x "$WORKER" ]; then
    echo "Error: per-pair worker not executable at $WORKER" >&2
    exit 2
fi
if [ ! -d "$REPO/nvmolkit/tests" ]; then
    echo "Error: nvmolkit/tests not found under $REPO" >&2
    exit 2
fi

# Conda discovery: prefer $CONDA_EXE (set by `conda activate` or by the
# conda installer's shell hook), else `conda` on PATH. We need both
# `conda info --base` (to locate the envs dir) and `conda create`.
if [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ]; then
    CONDA_BIN=$CONDA_EXE
elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN=$(command -v conda)
else
    echo "Error: 'conda' not found on PATH and CONDA_EXE not set." >&2
    echo "       This script needs conda to provide cpython interpreters" >&2
    echo "       for testing wheels (3.11, 3.12, 3.13, 3.14)." >&2
    exit 2
fi

CONDA_BASE=$("$CONDA_BIN" info --base 2>/dev/null) || {
    echo "Error: '$CONDA_BIN info --base' failed" >&2
    exit 2
}
ENVS_ROOT=$CONDA_BASE/envs

IFACE_ENV_PREFIX=${IFACE_ENV_PREFIX:-nvmolkit_iface_}
VENV_ROOT=${VENV_ROOT:-${TMPDIR:-/tmp}/nvmolkit_test_venvs}
TEST_LOG_DIR=$WHEELHOUSE/test_logs
TIMINGS_TSV=${TIMINGS_TSV:-$TEST_LOG_DIR/timings.tsv}

mkdir -p "$TEST_LOG_DIR" "$VENV_ROOT"
if [ ! -f "$TIMINGS_TSV" ]; then
    printf 'rdkit\tpy\tmode\tstatus\tstarted_at\tended_at\telapsed_sec\n' > "$TIMINGS_TSV"
fi

export REPO WHEELHOUSE TEST_LOG_DIR VENV_ROOT IFACE_ENV_PREFIX TIMINGS_TSV
export NVMOLKIT_CONDA_ENVS_ROOT=$ENVS_ROOT

ensure_iface_env() {
    local pyver=$1
    local envName=${IFACE_ENV_PREFIX}py${pyver}
    local envPython=$ENVS_ROOT/$envName/bin/python
    if [ -x "$envPython" ]; then
        return 0
    fi
    echo "Creating interpreter env $envName (python=$pyver)..."
    "$CONDA_BIN" create -y -n "$envName" -c conda-forge "python=$pyver" >&2 || {
        echo "Error: failed to create conda env $envName" >&2
        return 1
    }
    if [ ! -x "$envPython" ]; then
        echo "Error: conda env $envName created but python not at $envPython" >&2
        return 1
    fi
}

# For smoke without a pairs file: discover every wheel under wheelhouse.
discover_smoke_pairs() {
    local out=$1
    : > "$out"
    shopt -s nullglob
    local wheelDir
    for wheelDir in "$WHEELHOUSE"/rdkit*/py*; do
        local rdkit py
        rdkit=$(basename "$(dirname "$wheelDir")")
        rdkit=${rdkit#rdkit}
        py=$(basename "$wheelDir")
        py=${py#py}
        local matches=("$wheelDir"/nvmolkit-*.whl)
        if [ ${#matches[@]} -gt 0 ]; then
            echo "$rdkit $py" >> "$out"
        fi
    done
    shopt -u nullglob
}

drive_one_mode() {
    local mode=$1
    local pairsFile=$2

    local effectivePairs
    effectivePairs=$(mktemp)
    grep -vE '^[[:space:]]*(#|$)' "$pairsFile" > "$effectivePairs" || true

    local numPairs runStartedAt runStartedDisplay
    numPairs=$(wc -l < "$effectivePairs")
    # Use the awk-comparable form (no %Z) for filtering; keep a separate
    # human-readable form including the timezone for the run header.
    runStartedAt=$(date '+%Y-%m-%d %H:%M:%S')
    runStartedDisplay=$(date '+%Y-%m-%d %H:%M:%S %Z')

    echo
    echo "=========================================="
    echo "Mode: $mode"
    echo "Pairs file: $pairsFile ($numPairs pairs)"
    echo "Run started: $runStartedDisplay"
    echo "Wheelhouse: $WHEELHOUSE"
    echo "Test logs:  $TEST_LOG_DIR"
    echo "Timings:    $TIMINGS_TSV"
    echo "=========================================="

    local pyversNeeded
    pyversNeeded=$(awk '{print $2}' "$effectivePairs" | sort -u)
    local pyver
    for pyver in $pyversNeeded; do
        ensure_iface_env "$pyver" || {
            rm -f "$effectivePairs"
            return 1
        }
    done

    local start end wall
    start=$(date +%s)
    local fail_count=0 ok_count=0 skip_count=0
    declare -a failed_pairs=()

    while IFS= read -r line; do
        [ -z "$line" ] && continue
        local rdkit py
        rdkit=$(echo "$line" | awk '{print $1}')
        py=$(echo "$line" | awk '{print $2}')
        if [ -z "$rdkit" ] || [ -z "$py" ]; then
            echo "Warning: malformed line: '$line', skipping" >&2
            continue
        fi

        set +e
        "$WORKER" "$rdkit" "$py" "$mode"
        local rc=$?
        set -e

        case $rc in
            0)
                local wheelDir=$WHEELHOUSE/rdkit${rdkit}/py${py}
                if compgen -G "$wheelDir/nvmolkit-*.whl" > /dev/null; then
                    ok_count=$((ok_count + 1))
                else
                    skip_count=$((skip_count + 1))
                fi
                ;;
            *)
                fail_count=$((fail_count + 1))
                failed_pairs+=("rdkit=$rdkit py=$py")
                ;;
        esac
    done < "$effectivePairs"

    end=$(date +%s)
    wall=$((end - start))
    rm -f "$effectivePairs"

    echo
    echo "Mode '$mode' finished. Wall time: ${wall}s ($(printf '%dh%02dm%02ds' $((wall/3600)) $(((wall%3600)/60)) $((wall%60))))"
    echo "Results: ok=$ok_count fail=$fail_count skip=$skip_count total=$numPairs"
    if [ ${#failed_pairs[@]} -gt 0 ]; then
        echo "Failed:"
        local f
        for f in "${failed_pairs[@]}"; do
            echo "  $f"
        done
    fi

    awk -F'\t' -v run_start="$runStartedAt" -v wall="$wall" -v sel_mode="$mode" '
        NR == 1 { next }
        $5 >= run_start && $3 == sel_mode && ($4 == "ok" || $4 == "fail") {
            elapsed[++n] = $7 + 0
            rdkit[n] = $1; py[n] = $2; status[n] = $4
        }
        END {
            if (n == 0) { print "  (no completed tests)"; exit }
            sum = 0; mn = elapsed[1]; mx = elapsed[1]
            for (i = 1; i <= n; i++) {
                sum += elapsed[i]
                if (elapsed[i] < mn) mn = elapsed[i]
                if (elapsed[i] > mx) mx = elapsed[i]
            }
            for (i = 1; i <= n; i++) sorted[i] = elapsed[i]
            for (i = 2; i <= n; i++) {
                v = sorted[i]; j = i
                while (j > 1 && sorted[j-1] > v) { sorted[j] = sorted[j-1]; j-- }
                sorted[j] = v
            }
            med = (n % 2) ? sorted[(n+1)/2] : (sorted[n/2] + sorted[n/2+1]) / 2
            printf "\nTimings: tests=%d sum=%ds avg=%ds min=%ds p50=%ds max=%ds\n", \
                   n, sum, sum/n, mn, med, mx
            print "  Slowest:"
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

    return "$fail_count"
}

# Resolve pairs files per requested mode.
SMOKE_PAIRS_FILE=
FULL_PAIRS_FILE=
case "$MODE" in
    smoke)
        if [ -n "$PAIRS_FILE_ARG" ]; then
            [ -f "$PAIRS_FILE_ARG" ] || { echo "Error: pairs file not found: $PAIRS_FILE_ARG" >&2; exit 2; }
            SMOKE_PAIRS_FILE=$PAIRS_FILE_ARG
        fi
        ;;
    full)
        if [ -n "$PAIRS_FILE_ARG" ]; then
            [ -f "$PAIRS_FILE_ARG" ] || { echo "Error: pairs file not found: $PAIRS_FILE_ARG" >&2; exit 2; }
            FULL_PAIRS_FILE=$PAIRS_FILE_ARG
        else
            FULL_PAIRS_FILE=$SUBSET_FILE
        fi
        ;;
    both)
        FULL_PAIRS_FILE=$SUBSET_FILE
        ;;
esac

# For smoke (with no explicit pairs file) generate one by walking the wheelhouse.
if { [ "$MODE" = "smoke" ] || [ "$MODE" = "both" ]; } && [ -z "$SMOKE_PAIRS_FILE" ]; then
    SMOKE_PAIRS_FILE=$(mktemp)
    # shellcheck disable=SC2064  # interpolate path now so the trap fires correctly
    trap "rm -f '$SMOKE_PAIRS_FILE'" EXIT
    discover_smoke_pairs "$SMOKE_PAIRS_FILE"
    if [ ! -s "$SMOKE_PAIRS_FILE" ]; then
        echo "Error: no wheels discovered under $WHEELHOUSE/rdkit*/py*/" >&2
        exit 2
    fi
fi

if { [ "$MODE" = "full" ] || [ "$MODE" = "both" ]; } && [ ! -f "$FULL_PAIRS_FILE" ]; then
    echo "Error: full pairs file not found: $FULL_PAIRS_FILE" >&2
    exit 2
fi

overall_fail=0
case "$MODE" in
    smoke)
        drive_one_mode smoke "$SMOKE_PAIRS_FILE"
        overall_fail=$?
        ;;
    full)
        drive_one_mode full "$FULL_PAIRS_FILE"
        overall_fail=$?
        ;;
    both)
        drive_one_mode smoke "$SMOKE_PAIRS_FILE"
        smoke_fail=$?
        drive_one_mode full "$FULL_PAIRS_FILE"
        full_fail=$?
        overall_fail=$((smoke_fail + full_fail))
        ;;
esac

echo
if [ "$overall_fail" -eq 0 ]; then
    echo "All requested tests passed."
    exit 0
else
    echo "$overall_fail pair(s) failed across all modes. See per-pair logs in $TEST_LOG_DIR."
    exit 1
fi
