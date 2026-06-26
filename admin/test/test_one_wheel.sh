#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test one nvmolkit wheel against its target RDKit version.
#
# Usage: test_one_wheel.sh <rdkit_version> <python_version>
#
# Required environment is set by admin/test/test_all_wheels.sh:
#   REPO WHEELHOUSE TEST_LOG_DIR VENV_ROOT IFACE_ENV_PREFIX
#   NVMOLKIT_CONDA_ENVS_ROOT TIMINGS_TSV

set -uo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <rdkit_version> <python_version>" >&2
    exit 2
fi

rdkit=$1
py=$2

: "${REPO:?REPO must be set}"
: "${WHEELHOUSE:?WHEELHOUSE must be set}"
: "${TEST_LOG_DIR:?TEST_LOG_DIR must be set}"
: "${VENV_ROOT:?VENV_ROOT must be set}"
: "${IFACE_ENV_PREFIX:?IFACE_ENV_PREFIX must be set}"
: "${NVMOLKIT_CONDA_ENVS_ROOT:?NVMOLKIT_CONDA_ENVS_ROOT must be set}"
: "${TIMINGS_TSV:?TIMINGS_TSV must be set}"

iface_python=$NVMOLKIT_CONDA_ENVS_ROOT/${IFACE_ENV_PREFIX}py${py}/bin/python
if [ ! -x "$iface_python" ]; then
    echo "Error: interpreter env not found at $iface_python" >&2
    exit 2
fi

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

mkdir -p "$TEST_LOG_DIR" "$VENV_ROOT"
rm -rf "$venv" "$test_root"

{
    echo "=== nvmolkit wheel test ==="
    echo "rdkit=$rdkit"
    echo "python=$py"
    echo "wheel=$wheel"
    echo "interpreter=$iface_python ($($iface_python --version 2>&1))"
    echo "venv=$venv"
} > "$log_file"

run_step() {
    local label=$1
    shift
    {
        echo
        echo "--- $label ---"
        echo "+ $*"
    } >> "$log_file"
    "$@" >> "$log_file" 2>&1
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
    find "$test_root" -type d -name __pycache__ -prune -exec rm -rf {} +
    run_step_in_dir "pytest" "$test_root/run" "$venv/bin/pytest" \
        "$test_root/nvmolkit/tests" -k "not long" -v || rc=1
fi

ended_at=$(date '+%Y-%m-%d %H:%M:%S')
elapsed=$(( $(date +%s) - start_epoch ))

{
    echo
    echo "=== test finished ==="
    echo "rc=$rc"
    echo "ended_at=$ended_at"
    echo "elapsed_sec=$elapsed"
} >> "$log_file"

rm -rf "$venv" "$test_root"

if [ "$rc" -eq 0 ]; then
    status=ok
    printf '[ok %s] rdkit=%s py=%s elapsed=%ds\n' "$ended_at" "$rdkit" "$py" "$elapsed"
else
    status=fail
    printf '[FAIL %s] rdkit=%s py=%s elapsed=%ds (see %s)\n' \
        "$ended_at" "$rdkit" "$py" "$elapsed" "$log_file"
fi

printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$rdkit" "$py" "$status" "$started_at" "$ended_at" "$elapsed" >> "$TIMINGS_TSV"
exit "$rc"
