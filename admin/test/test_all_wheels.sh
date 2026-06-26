#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run installed-wheel tests across nvMolKit wheel matrix entries.
#
# Usage:
#   bash admin/test/test_all_wheels.sh <wheelhouse_dir>
#   bash admin/test/test_all_wheels.sh <wheelhouse_dir> <pairs_file>
#
# With only <wheelhouse_dir>, every wheel found under rdkit*/py*/ is tested.
# A pairs file contains one "<rdkit_version> <python_version>" pair per line;
# blank lines and '#' comments are ignored.

set -uo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <wheelhouse_dir> [pairs_file]" >&2
    exit 2
fi

WHEELHOUSE=$(cd "$1" 2>/dev/null && pwd) || {
    echo "Error: wheelhouse_dir '$1' is not a readable directory" >&2
    exit 2
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
WORKER=$SCRIPT_DIR/test_one_wheel.sh
TEST_LOG_DIR=${TEST_LOG_DIR:-$WHEELHOUSE/test_logs}

if [ ! -x "$WORKER" ]; then
    echo "Error: per-pair worker not executable at $WORKER" >&2
    exit 2
fi
if [ ! -d "$REPO/nvmolkit/tests" ]; then
    echo "Error: nvmolkit/tests not found under $REPO" >&2
    exit 2
fi

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

raw_pairs=$(make_tmp)
case $# in
    1)
        discover_pairs "$raw_pairs"
        ;;
    2)
        [ -f "$2" ] || { echo "Error: pairs file not found: $2" >&2; exit 2; }
        cp "$2" "$raw_pairs"
        ;;
esac

pairs=$(make_tmp)
awk '$1 !~ /^#/ && NF >= 2 { print $1, $2 }' "$raw_pairs" > "$pairs"
if [ ! -s "$pairs" ]; then
    echo "Error: no wheel test pairs found" >&2
    exit 2
fi

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
    "$WORKER" "$WHEELHOUSE" "$rdkit" "$py"
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
