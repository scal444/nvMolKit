#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Drive admin/test/test_one_wheel.sh across a list of (rdkit, python) pairs
# serially.
#
# Usage:
#   bash admin/test/test_all_wheels.sh <smoke|full> [pairs_file]
#
#   mode = smoke   -> default pairs file: wheelhouse/jobs/pairs.txt (built matrix)
#          full    -> default pairs file: admin/test/full_test_subset.txt
#
# Per-pair output: wheelhouse/test_logs/rdkit<X>_py<Y>_<mode>.log
# Per-pair timing rows appended to: wheelhouse/jobs/test_timings.tsv
#
# Environment overrides:
#   WHEELHOUSE        default <repo>/wheelhouse
#   VENV_ROOT         default /home/kevin/scratch/nvmolkit_test_venvs
#   IFACE_ENV_PREFIX  default nvmolkit_iface_
#   TIMINGS_TSV       default <wheelhouse>/jobs/test_timings.tsv

set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <smoke|full> [pairs_file]" >&2
    exit 2
fi

mode=$1
case "$mode" in
    smoke|full) ;;
    *) echo "Error: mode must be 'smoke' or 'full', got '$mode'" >&2; exit 2 ;;
esac

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

WHEELHOUSE=${WHEELHOUSE:-$REPO/wheelhouse}
VENV_ROOT=${VENV_ROOT:-/home/kevin/scratch/nvmolkit_test_venvs}
IFACE_ENV_PREFIX=${IFACE_ENV_PREFIX:-nvmolkit_iface_}
TIMINGS_TSV=${TIMINGS_TSV:-$WHEELHOUSE/jobs/test_timings.tsv}
TEST_LOG_DIR=$WHEELHOUSE/test_logs

if [ -n "${2:-}" ]; then
    PAIRS_FILE=$2
elif [ "$mode" = "smoke" ]; then
    PAIRS_FILE=$WHEELHOUSE/jobs/pairs.txt
else
    PAIRS_FILE=$REPO/admin/test/full_test_subset.txt
fi

if [ ! -f "$PAIRS_FILE" ]; then
    echo "Error: pairs file not found: $PAIRS_FILE" >&2
    exit 2
fi

mkdir -p "$TEST_LOG_DIR" "$VENV_ROOT" "$(dirname "$TIMINGS_TSV")"
if [ ! -f "$TIMINGS_TSV" ]; then
    printf 'rdkit\tpy\tmode\tstatus\tstarted_at\tended_at\telapsed_sec\n' > "$TIMINGS_TSV"
fi

# Strip comments and blank lines.
EFFECTIVE_PAIRS=$(mktemp)
trap 'rm -f "$EFFECTIVE_PAIRS"' EXIT
grep -vE '^[[:space:]]*(#|$)' "$PAIRS_FILE" > "$EFFECTIVE_PAIRS" || true

NUM_PAIRS=$(wc -l < "$EFFECTIVE_PAIRS")
RUN_STARTED_AT=$(date '+%Y-%m-%d %H:%M:%S %Z')

echo "Run started: $RUN_STARTED_AT"
echo "Mode: $mode"
echo "Pairs file: $PAIRS_FILE ($NUM_PAIRS pairs)"
echo "Wheelhouse: $WHEELHOUSE"
echo "Test logs:  $TEST_LOG_DIR"
echo "Venv root:  $VENV_ROOT"
echo "Timings:    $TIMINGS_TSV"
echo

WORKER=$REPO/admin/test/test_one_wheel.sh
if [ ! -x "$WORKER" ]; then
    echo "Error: worker script not executable at $WORKER" >&2
    exit 1
fi

export REPO WHEELHOUSE TEST_LOG_DIR VENV_ROOT IFACE_ENV_PREFIX TIMINGS_TSV

START=$(date +%s)
fail_count=0
ok_count=0
skip_count=0
declare -a failed_pairs=()

while IFS= read -r line; do
    [ -z "$line" ] && continue
    rdkit=$(echo "$line" | awk '{print $1}')
    py=$(echo "$line" | awk '{print $2}')
    if [ -z "$rdkit" ] || [ -z "$py" ]; then
        echo "Warning: malformed line: '$line', skipping" >&2
        continue
    fi

    set +e
    "$WORKER" "$rdkit" "$py" "$mode"
    rc=$?
    set -e

    case $rc in
        0)  # could be ok or skip - distinguish via the printed message? Or by
            # checking if the pair has a wheel. If no wheel, it was a skip.
            wheelDir=$WHEELHOUSE/rdkit${rdkit}/py${py}
            if compgen -G "$wheelDir/*.whl" > /dev/null; then
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
done < "$EFFECTIVE_PAIRS"

END=$(date +%s)
WALL=$((END - START))

echo
echo "All tests finished. Wall time: ${WALL}s ($(printf '%dh%02dm%02ds' $((WALL/3600)) $(((WALL%3600)/60)) $((WALL%60))))"
echo
echo "Results: ok=$ok_count fail=$fail_count skip=$skip_count total=$NUM_PAIRS"
if [ ${#failed_pairs[@]} -gt 0 ]; then
    echo "Failed:"
    for f in "${failed_pairs[@]}"; do
        echo "  $f"
    done
fi

# Per-build timing summary from the TSV. Reads only this run's rows
# (started_at >= RUN_STARTED_AT) for the requested mode.
echo
echo "Timings (this run, mode=$mode):"
awk -F'\t' -v run_start="$RUN_STARTED_AT" -v wall="$WALL" -v sel_mode="$mode" '
    NR == 1 { next }
    $5 >= run_start && $3 == sel_mode && ($4 == "ok" || $4 == "fail") {
        rows[++n] = $0
        elapsed[n] = $7 + 0
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
        printf "  tests=%d  sum=%ds  avg=%ds  min=%ds  p50=%ds  max=%ds\n", \
               n, sum, sum/n, mn, med, mx
        printf "\n  Slowest tests:\n"
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
echo "Per-pair logs: $TEST_LOG_DIR"

[ $fail_count -eq 0 ] && exit 0 || exit 1
