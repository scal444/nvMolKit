#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

output_root=${1:-/rdcu_profiles/aap_assay_benchmark_20260910}
selection=${2:-all}
log_path="$output_root/campaign.log"
metadata_path="$output_root/campaign.meta"
mkdir -p "$output_root"
exec >>"$log_path" 2>&1

command_line="selection=$selection python /nvmolkit/benchmarks/aap_clustering_bench.py --operation dise --sizes 32 64 128 256 512 1024 --runs 3 --warmup --threshold 0.30 --rdkit-workflow-max-size 512 --sort-tag <dataset score> --no_ligand_clustering_cpu"
echo "START time=$(date --iso-8601=seconds) pid=$$"
printf 'pid=%s\nstart_time=%s\ncommand=%s\noutput_root=%s\nlog=%s\n' \
  "$$" "$(date --iso-8601=seconds)" "$command_line" "$output_root" "$log_path" >"$metadata_path"

run_dataset() {
  local name=$1
  local sort_tag=$2
  shift 2
  echo "PROGRESS dataset=$name state=start time=$(date --iso-8601=seconds)"
  PYTHONDONTWRITEBYTECODE=1 python -u /nvmolkit/benchmarks/aap_clustering_bench.py \
    --operation dise \
    --sizes 32 64 128 256 512 1024 \
    --runs 3 \
    --warmup \
    --threshold 0.30 \
    --rdkit-workflow-max-size 512 \
    --sort-tag "$sort_tag" \
    --no_ligand_clustering_cpu \
    --output "$output_root/$name.csv" \
    "$@"
  local exit_code=$?
  echo "PROGRESS dataset=$name state=finished exit_code=$exit_code time=$(date --iso-8601=seconds)"
  return "$exit_code"
}

exit_code=0
if [[ "$selection" == all || "$selection" == novartis ]]; then
  run_dataset novartis_malaria "PF proliferation inhibition 3D7 EC50 uM" \
    --sdf /data/assay_data/novartis_malaria/raw/Novartis_GNF_NoModifier.sdf || exit_code=$?
fi

if [[ "$selection" == all ]]; then
  for aid in 485297 485313 588342 686979; do
    run_dataset "pubchem_aid_$aid" sort_value --sort-descending \
      --csv "/data/assay_data/pubchem_aid_$aid/molecules.csv" || exit_code=$?
  done
fi

echo "FINISHED exit_code=$exit_code time=$(date --iso-8601=seconds)"
exit "$exit_code"
