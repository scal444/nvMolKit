#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

output_root=$1
stage=$2
size=${3:-}
mkdir -p "$output_root"
log_path="$output_root/$stage${size:+_$size}.log"
metadata_path="$output_root/$stage${size:+_$size}.meta"
exec >>"$log_path" 2>&1

common_args=(
  --csv /data/assay_data/pubchem_aid_686979/molecules.csv
  --operation dise
  --threshold 0.30
  --sort-tag sort_value
  --sort-descending
  --sample-pool-size 70000
  --no_ligand_clustering_cpu
)

echo "START time=$(date --iso-8601=seconds) pid=$$ stage=$stage size=${size:-multiple}"
printf 'pid=%s\nstart_time=%s\nstage=%s\nsize=%s\ninput=%s\nsample_pool_size=70000\noutput_root=%s\nlog=%s\n' \
  "$$" "$(date --iso-8601=seconds)" "$stage" "${size:-multiple}" \
  /data/assay_data/pubchem_aid_686979/molecules.csv "$output_root" "$log_path" >"$metadata_path"

exit_code=0
if [[ "$stage" == cpu1024 ]]; then
  echo "PROGRESS stage=$stage state=benchmark_start runs=2 warmup=1"
  PYTHONDONTWRITEBYTECODE=1 python -u /nvmolkit/benchmarks/aap_clustering_bench.py \
    "${common_args[@]}" --sizes 1024 --runs 2 --warmup \
    --rdkit-workflow-max-size 1024 --no_nvmolkit \
    --output "$output_root/cpu_1024_runs2.csv" || exit_code=$?
elif [[ "$stage" == gpu_scan ]]; then
  for gpu_size in 1024 2048 4096 8192 16384 32768 65536; do
    echo "PROGRESS stage=$stage size=$gpu_size state=benchmark_start runs=3 warmup=1"
    utilization_log="$output_root/gpu_${gpu_size}_utilization.csv"
    nvidia-smi \
      --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
      --format=csv -l 1 >"$utilization_log" 2>&1 &
    monitor_pid=$!
    PYTHONDONTWRITEBYTECODE=1 python -u /nvmolkit/benchmarks/aap_clustering_bench.py \
      "${common_args[@]}" --sizes "$gpu_size" --runs 3 --warmup --no_rdkit \
      --output "$output_root/gpu_$gpu_size.csv" || exit_code=$?
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    echo "PROGRESS stage=$stage size=$gpu_size state=benchmark_finished exit_code=$exit_code"
    if [[ "$exit_code" -ne 0 ]]; then
      break
    fi
  done
elif [[ "$stage" == cpu_single && -n "$size" ]]; then
  echo "PROGRESS stage=$stage size=$size state=benchmark_start runs=1 warmup=0"
  PYTHONDONTWRITEBYTECODE=1 python -u /nvmolkit/benchmarks/aap_clustering_bench.py \
    "${common_args[@]}" --sizes "$size" --runs 1 --no_warmup \
    --rdkit-workflow-max-size "$size" --no_nvmolkit \
    --output "$output_root/cpu_${size}_run1.csv" || exit_code=$?
else
  echo "ERROR unknown stage or missing size: $stage ${size:-}" >&2
  exit_code=2
fi

echo "FINISHED exit_code=$exit_code time=$(date --iso-8601=seconds) stage=$stage size=${size:-multiple}"
exit "$exit_code"
