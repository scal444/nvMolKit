#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test one nvmolkit wheel against its target rdkit version.
#
# Usage: test_one_wheel.sh <rdkit_version> <python_version> <mode>
#   mode = smoke    -> import + tiny GPU op (admin/test/smoke_check.py)
#          full     -> smoke + full pytest from repo's nvmolkit/tests
#
# Required environment:
#   REPO              : repo root (provides smoke_check.py and nvmolkit/tests/)
#   WHEELHOUSE        : where wheels live; reads <wheelhouse>/rdkit<X>/py<Y>/*.whl
#   TEST_LOG_DIR      : per-pair log dir
#   VENV_ROOT         : where to create throwaway venvs
#   IFACE_ENV_PREFIX  : prefix for interpreter conda envs;
#                       env name is "<prefix>py<version>" (e.g. nvmolkit_iface_py3.12)
#   TIMINGS_TSV       : append-only timings log
#                       (rdkit\tpy\tmode\tstatus\tstart\tend\telapsed_sec)
#
# Exits 0 on pass, 1 on failure of any test step, 2 on usage error.

set -uo pipefail

if [ $# -ne 3 ]; then
    echo "Usage: $0 <rdkit_version> <python_version> <smoke|full>" >&2
    exit 2
fi

rdkit=$1
py=$2
mode=$3

case "$mode" in
    smoke|full) ;;
    *) echo "Error: mode must be 'smoke' or 'full', got '$mode'" >&2; exit 2 ;;
esac

: "${REPO:?REPO must be set}"
: "${WHEELHOUSE:?WHEELHOUSE must be set}"
: "${TEST_LOG_DIR:?TEST_LOG_DIR must be set}"
: "${VENV_ROOT:?VENV_ROOT must be set}"
: "${IFACE_ENV_PREFIX:?IFACE_ENV_PREFIX must be set}"
: "${TIMINGS_TSV:?TIMINGS_TSV must be set}"

ifaceEnv=${IFACE_ENV_PREFIX}py${py}
ifacePython=/home/kevin/programs/miniforge3/envs/${ifaceEnv}/bin/python
if [ ! -x "$ifacePython" ]; then
    echo "Error: interpreter env not found at $ifacePython" >&2
    echo "       Create it with: conda create -y -n ${ifaceEnv} -c conda-forge python=${py}" >&2
    exit 2
fi

wheelDir=$WHEELHOUSE/rdkit${rdkit}/py${py}
shopt -s nullglob
wheelMatches=("$wheelDir"/nvmolkit-*-cp${py//./}-cp${py//./}-*.whl)
shopt -u nullglob
if [ ${#wheelMatches[@]} -eq 0 ]; then
    echo "[skip] rdkit=$rdkit py=$py mode=$mode (no wheel at $wheelDir)"
    exit 0
fi
if [ ${#wheelMatches[@]} -gt 1 ]; then
    echo "Error: multiple wheels in $wheelDir (${wheelMatches[*]})" >&2
    exit 2
fi
wheel=${wheelMatches[0]}

mkdir -p "$TEST_LOG_DIR" "$VENV_ROOT"
logFile=$TEST_LOG_DIR/rdkit${rdkit}_py${py}_${mode}.log
venv=$VENV_ROOT/rdkit${rdkit}_py${py}

startEpoch=$(date +%s)
startedAt=$(date '+%Y-%m-%d %H:%M:%S')

format_hms() {
    local secs=$1
    printf '%dh%02dm%02ds' $((secs / 3600)) $(((secs % 3600) / 60)) $((secs % 60))
}

echo "[start $startedAt] rdkit=$rdkit py=$py mode=$mode wheel=$(basename "$wheel")"

{
    echo "=== nvmolkit wheel test ==="
    echo "    rdkit=$rdkit py=$py mode=$mode"
    echo "    started_at=$startedAt"
    echo "    wheel=$wheel"
    echo "    interpreter=$ifacePython ($($ifacePython --version 2>&1))"
    echo "    venv=$venv"
    echo "==========================="
} > "$logFile"

run_step() {
    local label=$1
    shift
    {
        echo
        echo "--- $label ---"
        echo "+ $*"
    } >> "$logFile"
    "$@" >> "$logFile" 2>&1
    local rc=$?
    echo "--- $label exit=$rc ---" >> "$logFile"
    return $rc
}

# Recreate venv from scratch so we don't carry state from prior test runs.
rm -rf "$venv"
if ! run_step "venv-create" "$ifacePython" -m venv "$venv"; then
    echo "[FAIL] rdkit=$rdkit py=$py mode=$mode step=venv-create (see $logFile)"
    rc=1
elif ! run_step "pip-upgrade" "$venv/bin/pip" install --upgrade pip; then
    echo "[FAIL] rdkit=$rdkit py=$py mode=$mode step=pip-upgrade (see $logFile)"
    rc=1
elif ! run_step "install-wheel" "$venv/bin/pip" install \
        "$wheel" "rdkit==${rdkit}"; then
    echo "[FAIL] rdkit=$rdkit py=$py mode=$mode step=install-wheel (see $logFile)"
    rc=1
elif ! run_step "smoke-check" "$venv/bin/python" "$REPO/admin/test/smoke_check.py"; then
    echo "[FAIL] rdkit=$rdkit py=$py mode=$mode step=smoke-check (see $logFile)"
    rc=1
elif [ "$mode" = "full" ]; then
    if ! run_step "install-test-deps" "$venv/bin/pip" install \
            pandas pytest psutil optuna; then
        echo "[FAIL] rdkit=$rdkit py=$py mode=$mode step=install-test-deps (see $logFile)"
        rc=1
    elif ! run_step "pytest" "$venv/bin/pytest" \
            "$REPO/nvmolkit/tests" -k "not long" -v; then
        echo "[FAIL] rdkit=$rdkit py=$py mode=$mode step=pytest (see $logFile)"
        rc=1
    else
        rc=0
    fi
else
    rc=0
fi

endEpoch=$(date +%s)
endedAt=$(date '+%Y-%m-%d %H:%M:%S')
elapsed=$((endEpoch - startEpoch))
elapsedHms=$(format_hms "$elapsed")

{
    echo
    echo "=== test finished ==="
    echo "    rdkit=$rdkit py=$py mode=$mode rc=$rc"
    echo "    ended_at=$endedAt"
    echo "    elapsed=${elapsed}s ($elapsedHms)"
    echo "====================="
} >> "$logFile"

# Tear down venv. Keep the log.
rm -rf "$venv"

if [ "$rc" -eq 0 ]; then
    status=ok
    printf '[ok %s] rdkit=%s py=%s mode=%s elapsed=%s\n' \
        "$endedAt" "$rdkit" "$py" "$mode" "$elapsedHms"
else
    status=fail
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$rdkit" "$py" "$mode" "$status" "$startedAt" "$endedAt" "$elapsed" \
    >> "$TIMINGS_TSV"

exit "$rc"
