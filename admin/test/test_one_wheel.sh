#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test one nvmolkit wheel against its target RDKit version.
#
# Usage: test_one_wheel.sh <wheelhouse_dir> <rdkit_version> <python_version>
#
# Environment overrides:
#   VENV_ROOT          throwaway pip venv dir (default ${TMPDIR:-/tmp}/nvmolkit_test_venvs)
#   IFACE_ENV_PREFIX   prefix for per-python conda envs that supply cpython
#                      interpreters (default nvmolkit_iface_)
#   TIMINGS_TSV        append-only timings tsv (default <wheelhouse>/test_logs/timings.tsv)

set -uo pipefail

if [ $# -ne 3 ]; then
    echo "Usage: $0 <wheelhouse_dir> <rdkit_version> <python_version>" >&2
    exit 2
fi

WHEELHOUSE=$(cd "$1" 2>/dev/null && pwd) || {
    echo "Error: wheelhouse_dir '$1' is not a readable directory" >&2
    exit 2
}
rdkit=$2
py=$3

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)

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

if ! CONDA_BASE=$("$CONDA_BIN" info --base 2>/dev/null); then
    echo "Error: '$CONDA_BIN info --base' failed" >&2
    exit 2
fi
ENVS_ROOT=$CONDA_BASE/envs

TEST_LOG_DIR=${TEST_LOG_DIR:-$WHEELHOUSE/test_logs}
VENV_ROOT=${VENV_ROOT:-${TMPDIR:-/tmp}/nvmolkit_test_venvs}
IFACE_ENV_PREFIX=${IFACE_ENV_PREFIX:-nvmolkit_iface_}
TIMINGS_TSV=${TIMINGS_TSV:-$TEST_LOG_DIR/timings.tsv}

mkdir -p "$TEST_LOG_DIR" "$VENV_ROOT"
if [ ! -f "$TIMINGS_TSV" ]; then
    printf 'rdkit\tpy\tstatus\tstarted_at\tended_at\telapsed_sec\n' > "$TIMINGS_TSV"
fi

ensure_iface_env() {
    local pyver=$1
    local env_name=${IFACE_ENV_PREFIX}py${pyver}
    local env_python=$ENVS_ROOT/$env_name/bin/python
    if [ -x "$env_python" ]; then
        return 0
    fi
    echo "Creating interpreter env $env_name (python=$pyver)..."
    "$CONDA_BIN" create -y -n "$env_name" -c conda-forge "python=$pyver" >&2 || {
        echo "Error: failed to create conda env $env_name" >&2
        return 1
    }
    if [ ! -x "$env_python" ]; then
        echo "Error: conda env $env_name created but python not at $env_python" >&2
        return 1
    fi
}

ensure_iface_env "$py" || exit 1
iface_python=$ENVS_ROOT/${IFACE_ENV_PREFIX}py${py}/bin/python

wheel_dir=$WHEELHOUSE/rdkit${rdkit}/py${py}
shopt -s nullglob
wheel_matches=("$wheel_dir"/nvmolkit-*-cp${py//./}-cp${py//./}-*.whl)
shopt -u nullglob
if [ ${#wheel_matches[@]} -eq 0 ]; then
    echo "[skip] rdkit=$rdkit py=$py (no wheel at $wheel_dir)"
    exit 0
fi
if [ ${#wheel_matches[@]} -gt 1 ]; then
    echo "Error: multiple wheels in $wheel_dir (${wheel_matches[*]})" >&2
    exit 2
fi

wheel=${wheel_matches[0]}
venv=$VENV_ROOT/rdkit${rdkit}_py${py}
test_root=$VENV_ROOT/rdkit${rdkit}_py${py}_tests
log_file=$TEST_LOG_DIR/rdkit${rdkit}_py${py}.log
started_at=$(date '+%Y-%m-%d %H:%M:%S')
start_epoch=$(date +%s)
rc=0

cleanup() {
    rm -rf "$venv" "$test_root"
}
rm -rf "$venv" "$test_root"
trap cleanup EXIT

{
    echo "=== nvmolkit wheel test ==="
    echo "rdkit=$rdkit"
    echo "python=$py"
    echo "wheel=$wheel"
    echo "interpreter=$iface_python ($($iface_python --version 2>&1))"
    echo "venv=$venv"
    echo "test_root=$test_root"
} > "$log_file"

format_hms() {
    local secs=$1
    printf '%dh%02dm%02ds' $((secs / 3600)) $(((secs % 3600) / 60)) $((secs % 60))
}

run_step() {
    local label=$1
    shift
    {
        echo
        echo "--- $label ---"
        echo "+ $*"
    } >> "$log_file"
    "$@" >> "$log_file" 2>&1
    local step_rc=$?
    echo "--- $label exit=$step_rc ---" >> "$log_file"
    return "$step_rc"
}

run_step_in_dir() {
    local label=$1
    local dir=$2
    shift 2
    {
        echo
        echo "--- $label ---"
        echo "cwd: $dir"
        echo "+ $*"
    } >> "$log_file"
    (cd "$dir" && "$@") >> "$log_file" 2>&1
    local step_rc=$?
    echo "--- $label exit=$step_rc ---" >> "$log_file"
    return "$step_rc"
}

if ! run_step "venv-create" "$iface_python" -m venv "$venv"; then
    rc=1
elif ! run_step "pip-upgrade" "$venv/bin/pip" install --upgrade pip; then
    rc=1
elif ! run_step "install-wheel" "$venv/bin/pip" install "$wheel" "rdkit==${rdkit}"; then
    rc=1
elif ! run_step "installed-wheel-check" "$venv/bin/python" "$REPO/admin/test/smoke_check.py"; then
    rc=1
elif ! run_step "install-test-deps" "$venv/bin/pip" install pandas pytest psutil optuna; then
    rc=1
else
    mkdir -p "$test_root/nvmolkit" "$test_root/tests" "$test_root/run"
    cp -a "$REPO/nvmolkit/tests" "$test_root/nvmolkit/"
    cp -a "$REPO/tests/test_data" "$test_root/tests/"
    if [ -d "$REPO/agent-skills" ]; then
        cp -a "$REPO/agent-skills" "$test_root/"
    fi
    find "$test_root" -type d -name __pycache__ -prune -exec rm -rf {} +
    run_step_in_dir "pytest" "$test_root/run" "$venv/bin/pytest" \
        "$test_root/nvmolkit/tests" -k "not long" -v || rc=1
fi

ended_at=$(date '+%Y-%m-%d %H:%M:%S')
elapsed=$(( $(date +%s) - start_epoch ))
elapsed_hms=$(format_hms "$elapsed")

{
    echo
    echo "=== test finished ==="
    echo "rc=$rc"
    echo "ended_at=$ended_at"
    echo "elapsed=${elapsed}s ($elapsed_hms)"
} >> "$log_file"

if [ "$rc" -eq 0 ]; then
    status=ok
    printf '[ok %s] rdkit=%s py=%s elapsed=%s\n' "$ended_at" "$rdkit" "$py" "$elapsed_hms"
else
    status=fail
    printf '[FAIL %s] rdkit=%s py=%s elapsed=%s (see %s)\n' \
        "$ended_at" "$rdkit" "$py" "$elapsed_hms" "$log_file"
fi

printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$rdkit" "$py" "$status" "$started_at" "$ended_at" "$elapsed" >> "$TIMINGS_TSV"
exit "$rc"
