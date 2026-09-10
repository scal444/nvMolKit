#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -u

aap_result_dir=/rdcu_profiles/aap_java_reference_2015/validation
aap_graphs="$aap_result_dir/novartis_sorted_graphs.tsv"
aap_assignments="$aap_result_dir/novartis_default8_assignments.tsv"
aap_log="$aap_result_dir/full_validation_20260910.log"
aap_meta="$aap_result_dir/full_validation_20260910.meta"
aap_classpath=/rdcu_profiles/aap_java_reference_2015/java_build
aap_command="java -Xmx30g -cp $aap_classpath GCheminfoAAPDISE cluster $aap_graphs 0.3 7 2 $aap_assignments"

if ! test -w /nvmolkit || ! test -w "$aap_result_dir"; then
    printf 'FAILED preflight source_or_results_not_writable\n' >>"$aap_log"
    exit 73
fi

{
    printf 'container_pid=%s\n' "$$"
    printf 'start_time=%s\n' "$(date --iso-8601=seconds)"
    printf 'command=%s\n' "$aap_command"
    printf 'graph_path=%s\n' "$aap_graphs"
    printf 'assignment_path=%s\n' "$aap_assignments"
    printf 'log_path=%s\n' "$aap_log"
} >"$aap_meta"

printf 'START time=%s command=%s\n' "$(date --iso-8601=seconds)" "$aap_command" >"$aap_log"
printf 'PROGRESS phase=preflight source_writable=true results_writable=true\n' >>"$aap_log"

java -Xmx30g -cp "$aap_classpath" GCheminfoAAPDISE cluster \
    "$aap_graphs" 0.3 7 2 "$aap_assignments" >>"$aap_log" 2>&1
aap_rc=$?
printf 'FINISHED exit_code=%d time=%s\n' "$aap_rc" "$(date --iso-8601=seconds)" >>"$aap_log"
exit "$aap_rc"
