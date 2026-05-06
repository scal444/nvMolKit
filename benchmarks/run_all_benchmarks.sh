#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Run the full set of nvMolKit Python benchmarks and collect results plus
# system metadata into a single output directory.
#
# The driver runs:
#   - butina_clustering_bench.py
#   - conformer_rmsd_bench.py
#   - cross_similarity_bench.py
#   - substruct_bench.py
#   - tfd_bench.py
#
# Usage:
#   ./run_all_benchmarks.sh [output_dir] [-- extra args forwarded to every bench]
#
# Defaults to ./benchmark_results_<timestamp>/.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
SMARTS_DIR="$REPO_ROOT/tests/test_data/SMARTS"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${1:-$SCRIPT_DIR/benchmark_results_$TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
RESULT_DIR="$OUTPUT_DIR/results"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

SMILES_CSV="$DATA_DIR/benchmark_smiles.csv"
CHEMBL_SMI="$DATA_DIR/chembl_10k.smi"
ENAMINE_CXSMILES="$DATA_DIR/enamine_10M.cxsmiles"

if [ ! -f "$SMILES_CSV" ]; then
  echo "Missing $SMILES_CSV (required for cross_similarity / tfd benchmarks)" >&2
  exit 1
fi
if [ ! -f "$CHEMBL_SMI" ]; then
  echo "Missing $CHEMBL_SMI (required for butina / substruct benchmarks)" >&2
  exit 1
fi

# Capture system metadata up front so it's preserved even if a bench aborts.
SYSINFO_DIR="$OUTPUT_DIR/sysinfo"
mkdir -p "$SYSINFO_DIR"
{
  echo "date_utc: $(date -u -Iseconds)"
  echo "date_local: $(date -Iseconds)"
  echo "hostname: $(hostname)"
  echo "uname: $(uname -a)"
  echo "git_commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_status:"
  git -C "$REPO_ROOT" status --short 2>/dev/null || true
} > "$SYSINFO_DIR/run_info.txt"

lscpu > "$SYSINFO_DIR/lscpu.txt" 2>&1 || echo "lscpu unavailable" > "$SYSINFO_DIR/lscpu.txt"
nvidia-smi > "$SYSINFO_DIR/nvidia-smi.txt" 2>&1 || echo "nvidia-smi unavailable" > "$SYSINFO_DIR/nvidia-smi.txt"
nvidia-smi -q > "$SYSINFO_DIR/nvidia-smi-q.txt" 2>&1 || true
free -h > "$SYSINFO_DIR/meminfo.txt" 2>&1 || true
python -c "import sys, platform; print('python:', sys.version); print('platform:', platform.platform())" \
  > "$SYSINFO_DIR/python.txt" 2>&1 || true
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')" \
  >> "$SYSINFO_DIR/python.txt" 2>&1 || true
pip freeze > "$SYSINFO_DIR/pip_freeze.txt" 2>&1 || true

# Substruct bench config: rewrite the shipped CSV (which uses container paths)
# so it points at the SMARTS files in this checkout.
SUBSTRUCT_CONFIG="$OUTPUT_DIR/substruct_config_resolved.csv"
{
  echo "smarts,batch_size,workers,prep_threads,mode,num_gpus"
  echo "$SMARTS_DIR/rdkit_fragment_descriptors_supported.txt,8192,4,10,countSubstructMatches,1"
  echo "$SMARTS_DIR/wehi_pains_supported.txt,8192,6,8,hasSubstructMatch,1"
  echo "$SMARTS_DIR/BMS_2006_filter_supported.txt,8192,6,8,hasSubstructMatch,1"
  echo "$SMARTS_DIR/rdkit_tautomer_transforms_supported.txt,8192,4,10,getSubstructMatches,1"
  echo "$SMARTS_DIR/rdkit_torsionPreferences_v2_supported.txt,8192,4,10,getSubstructMatches,1"
} > "$SUBSTRUCT_CONFIG"

SUMMARY="$OUTPUT_DIR/summary.tsv"
printf "benchmark\tstatus\texit_code\tduration_s\tlog\tresult\n" > "$SUMMARY"

run_bench() {
  local name="$1"
  local result_path="$2"
  shift 2
  local log_path="$LOG_DIR/${name}.log"

  echo "=========================================="
  echo "[$(date -Iseconds)] Running $name"
  echo "  cmd: $*"
  echo "  log: $log_path"
  echo "=========================================="

  local start_s end_s duration status code
  start_s=$(date +%s)
  set +e
  ( cd "$SCRIPT_DIR" && "$@" ) > "$log_path" 2>&1
  code=$?
  set -e
  end_s=$(date +%s)
  duration=$((end_s - start_s))
  if [ "$code" -eq 0 ]; then
    status="ok"
  else
    status="fail"
  fi
  printf "%s\t%s\t%d\t%d\t%s\t%s\n" \
    "$name" "$status" "$code" "$duration" "$log_path" "$result_path" >> "$SUMMARY"
  echo "[$name] $status (exit=$code, ${duration}s)"
}

run_bench "butina_clustering" \
  "$RESULT_DIR/butina_clustering.csv" \
  python "$SCRIPT_DIR/butina_clustering_bench.py" \
  "$CHEMBL_SMI" \
  --output "$RESULT_DIR/butina_clustering.csv"

run_bench "conformer_rmsd" \
  "$LOG_DIR/conformer_rmsd.log" \
  python "$SCRIPT_DIR/conformer_rmsd_bench.py"

run_bench "cross_similarity" \
  "$RESULT_DIR/cross_similarity.json" \
  python "$SCRIPT_DIR/cross_similarity_bench.py" \
  --input "$SMILES_CSV" \
  --output "$RESULT_DIR/cross_similarity.json"

run_bench "substruct" \
  "$LOG_DIR/substruct.log" \
  python "$SCRIPT_DIR/substruct_bench.py" \
  --smiles "$CHEMBL_SMI" \
  --config "$SUBSTRUCT_CONFIG" \
  --no_validate

run_bench "tfd" \
  "$RESULT_DIR/tfd.csv" \
  python "$SCRIPT_DIR/tfd_bench.py" \
  --smiles-file "$SMILES_CSV" \
  --output "$RESULT_DIR/tfd.csv"

echo
echo "All benchmarks complete. Output: $OUTPUT_DIR"
echo "Summary:"
column -t -s $'\t' "$SUMMARY" || cat "$SUMMARY"
