#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -u

aap_result_dir=/rdcu_profiles/aap_java_reference_2015/gpu_dise
aap_log="$aap_result_dir/build_20260910.log"
aap_meta="$aap_result_dir/build_20260910.meta"
aap_command="CMAKE_BUILD_TYPE=RelWithDebInfo NVMOLKIT_CUDA_TARGET_MODE=native CMAKE_BUILD_PARALLEL_LEVEL=\$(nproc) pip -v install '.[test]'"

if ! test -w /nvmolkit || ! test -w "$aap_result_dir"; then
    printf 'FINISHED exit_code=1 reason=path_not_writable\n' >"$aap_log"
    exit 1
fi

{
    printf 'container_pid=%d\n' "$$"
    printf 'start_time=%s\n' "$(date --iso-8601=seconds)"
    printf 'exact_command=%s\n' "$aap_command"
    printf 'source=/nvmolkit\n'
    printf 'log=%s\n' "$aap_log"
} >"$aap_meta"

printf 'START time=%s command=%s\n' "$(date --iso-8601=seconds)" "$aap_command" >"$aap_log"
printf 'PROGRESS phase=preflight source_writable=true results_writable=true\n' >>"$aap_log"
cd /nvmolkit || exit 1
CMAKE_BUILD_TYPE=RelWithDebInfo \
NVMOLKIT_CUDA_TARGET_MODE=native \
CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
pip -v install '.[test]' >>"$aap_log" 2>&1
aap_rc=$?
if test "$aap_rc" -eq 0; then
    printf 'PROGRESS phase=installed_import_check\n' >>"$aap_log"
    cd / || exit 1
    python -c 'import nvmolkit, nvmolkit._clustering as c; print(nvmolkit.__file__); print(c.__file__); print(hasattr(c, "aap_dise_clustering"))' >>"$aap_log" 2>&1
    aap_rc=$?
fi
printf 'FINISHED exit_code=%d time=%s\n' "$aap_rc" "$(date --iso-8601=seconds)" >>"$aap_log"
exit "$aap_rc"
