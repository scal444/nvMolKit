#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the publication benchmark suite on one dataset, one GPU, and at most
# fourteen physical CPU cores. Unlike run_all_benchmarks.sh, this runner has no
# multi-GPU mode and no molecule-size scans.

set -uo pipefail

usage() {
  cat >&2 <<EOF
Usage: $0 --dataset FILE --output-dir DIR [options]

Required:
  --dataset FILE       SMILES/CXSMILES dataset used by every benchmark
  --output-dir DIR     Directory for results, logs, tuning configs, and metadata

Options:
  --gpu-id ID          Physical GPU to expose (default: 0)
  --cpus N             CPU-core budget, 1-14 (default: min(14, physical cores))
  --no-rdkit           Run nvMolKit only; disables cross-implementation validation
  --no-nvmolkit        Run RDKit only; disables autotuning
  --no-autotune        Use benchmark defaults instead of autotuning nvMolKit
  --autotune-trials N  Optuna trials per tuned benchmark (default: 40)
  --autotune-seconds N Target seconds per autotune trial (default: 60)
  -h, --help           Show this help

The suite runs the non-size-scan benchmarks from run_all_benchmarks.sh:
Butina clustering, conformer RMSD, cross similarity, ETKDG, all MMFF/UFF x
BFGS/FIRE combinations, all supported substructure sets, and TFD.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SMARTS_DIR="$REPO_ROOT/tests/test_data/SMARTS"

DATASET=""
OUTPUT_DIR=""
GPU_ID=0
REQUESTED_CPUS=""
SKIP_RDKIT=0
SKIP_NVMOLKIT=0
AUTOTUNE_ENABLED=1
AUTOTUNE_TRIALS=40
AUTOTUNE_TIME_BUDGET=60

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset)
      [ $# -ge 2 ] || { echo "Error: --dataset requires a value" >&2; exit 2; }
      DATASET="$2"
      shift 2
      ;;
    --output-dir)
      [ $# -ge 2 ] || { echo "Error: --output-dir requires a value" >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --gpu-id)
      [ $# -ge 2 ] || { echo "Error: --gpu-id requires a value" >&2; exit 2; }
      GPU_ID="$2"
      shift 2
      ;;
    --cpus)
      [ $# -ge 2 ] || { echo "Error: --cpus requires a value" >&2; exit 2; }
      REQUESTED_CPUS="$2"
      shift 2
      ;;
    --no-rdkit)
      SKIP_RDKIT=1
      shift
      ;;
    --no-nvmolkit)
      SKIP_NVMOLKIT=1
      shift
      ;;
    --no-autotune)
      AUTOTUNE_ENABLED=0
      shift
      ;;
    --autotune-trials)
      [ $# -ge 2 ] || { echo "Error: --autotune-trials requires a value" >&2; exit 2; }
      AUTOTUNE_TRIALS="$2"
      shift 2
      ;;
    --autotune-seconds)
      [ $# -ge 2 ] || { echo "Error: --autotune-seconds requires a value" >&2; exit 2; }
      AUTOTUNE_TIME_BUDGET="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [ -z "$DATASET" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Error: --dataset and --output-dir are required" >&2
  usage
  exit 2
fi
if [ ! -f "$DATASET" ]; then
  echo "Error: dataset does not exist: $DATASET" >&2
  exit 1
fi
# Benchmark commands execute from SCRIPT_DIR, so make caller-relative paths
# absolute before constructing any command lines.
if ! DATASET_DIR="$(cd "$(dirname "$DATASET")" && pwd)"; then
  echo "Error: could not resolve dataset directory: $(dirname "$DATASET")" >&2
  exit 1
fi
DATASET="$DATASET_DIR/$(basename "$DATASET")"
if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "Error: --gpu-id must be a non-negative integer" >&2
  exit 2
fi
if [ "$SKIP_RDKIT" = "1" ] && [ "$SKIP_NVMOLKIT" = "1" ]; then
  echo "Error: --no-rdkit and --no-nvmolkit are mutually exclusive" >&2
  exit 2
fi
if ! [[ "$AUTOTUNE_TRIALS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --autotune-trials must be a positive integer" >&2
  exit 2
fi
if ! [[ "$AUTOTUNE_TIME_BUDGET" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --autotune-seconds must be a positive integer" >&2
  exit 2
fi
if [ "$SKIP_NVMOLKIT" = "1" ]; then
  AUTOTUNE_ENABLED=0
fi

PHYSICAL_CORES="$(lscpu -p=Core,Socket 2>/dev/null | awk -F, '!/^#/ {print $1 "," $2}' | sort -u | wc -l)"
if [ -z "$PHYSICAL_CORES" ] || [ "$PHYSICAL_CORES" -lt 1 ]; then
  PHYSICAL_CORES=1
fi
AVAILABLE_CPUS="$(nproc 2>/dev/null || echo 1)"
CPU_LIMIT="$PHYSICAL_CORES"
if [ "$AVAILABLE_CPUS" -lt "$CPU_LIMIT" ]; then
  CPU_LIMIT="$AVAILABLE_CPUS"
fi
if [ "$CPU_LIMIT" -gt 14 ]; then
  CPU_LIMIT=14
fi
if [ -z "$REQUESTED_CPUS" ]; then
  CPUS="$CPU_LIMIT"
else
  if ! [[ "$REQUESTED_CPUS" =~ ^[1-9][0-9]*$ ]] || [ "$REQUESTED_CPUS" -gt 14 ]; then
    echo "Error: --cpus must be an integer from 1 through 14" >&2
    exit 2
  fi
  if [ "$REQUESTED_CPUS" -gt "$CPU_LIMIT" ]; then
    echo "Error: requested $REQUESTED_CPUS CPUs but the usable publication limit is $CPU_LIMIT" >&2
    echo "       (physical cores=$PHYSICAL_CORES, CPUs available to this process=$AVAILABLE_CPUS, cap=14)" >&2
    exit 2
  fi
  CPUS="$REQUESTED_CPUS"
fi

# Expose exactly one GPU. All benchmark CLIs see it as logical GPU 0.
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS="$CPUS"
export MKL_NUM_THREADS="$CPUS"
export OPENBLAS_NUM_THREADS="$CPUS"
export NUMEXPR_NUM_THREADS="$CPUS"

if [ "$SKIP_NVMOLKIT" = "0" ]; then
  if ! GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null)"; then
    echo "Error: could not query CUDA device count through torch" >&2
    exit 1
  fi
  if [ "$GPU_COUNT" -ne 1 ]; then
    echo "Error: expected exactly one visible GPU, but torch reports $GPU_COUNT" >&2
    echo "       CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 1
  fi
else
  GPU_COUNT="not_checked"
fi

for module in nvtx numpy pandas pyperf rdkit torch tqdm; do
  if ! python -c "import $module" 2>/dev/null; then
    echo "Error: required Python package '$module' is not importable" >&2
    exit 1
  fi
done
if [ "$AUTOTUNE_ENABLED" = "1" ] && ! python -c 'import optuna' 2>/dev/null; then
  echo "Error: autotuning requires optuna; install it or pass --no-autotune" >&2
  exit 1
fi

if ! mkdir -p "$OUTPUT_DIR"; then
  echo "Error: could not create output directory: $OUTPUT_DIR" >&2
  exit 1
fi
if ! OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"; then
  echo "Error: could not resolve output directory" >&2
  exit 1
fi
LOG_DIR="$OUTPUT_DIR/logs"
RESULT_DIR="$OUTPUT_DIR/results"
AUTOTUNE_DIR="$OUTPUT_DIR/autotune"
SYSINFO_DIR="$OUTPUT_DIR/sysinfo"
if ! mkdir -p "$LOG_DIR" "$RESULT_DIR" "$AUTOTUNE_DIR" "$SYSINFO_DIR"; then
  echo "Error: could not create benchmark output subdirectories" >&2
  exit 1
fi

# The largest ETKDG/FF candidate has 1024 conformers/batch * 8 batches/GPU.
# Eight complete pipeline fills keep even that candidate out of the short-tail
# regime. Ceiling division is deliberate: the old two-fill formula rounded
# down and could provide fewer conformers than its stated minimum.
CONFS_PER_MOL=200
AUTOTUNE_BATCH_SIZE_MAX=1024
AUTOTUNE_BATCHES_PER_GPU_MAX=8
AUTOTUNE_PIPELINE_FILLS=8
MAX_INFLIGHT_CONFS=$(( AUTOTUNE_BATCH_SIZE_MAX * AUTOTUNE_BATCHES_PER_GPU_MAX ))
AUTOTUNE_CONFS=$(( AUTOTUNE_PIPELINE_FILLS * MAX_INFLIGHT_CONFS ))
ETKDG_CAL_SIZE=$(( (AUTOTUNE_CONFS + CONFS_PER_MOL - 1) / CONFS_PER_MOL ))
FF_CAL_SIZE="$ETKDG_CAL_SIZE"

# The substructure workers may each hold one maximum-sized batch. Give every
# candidate eight full waves at the largest worker count.
SUBSTRUCT_CAL_SIZE=$(( AUTOTUNE_PIPELINE_FILLS * AUTOTUNE_BATCH_SIZE_MAX * AUTOTUNE_BATCHES_PER_GPU_MAX ))

# Timed ETKDG/FF inputs are three times the calibration set. Substructure uses
# a larger steady-state workload while retaining the calibration set as a
# proper subset.
ETKDG_NUM_MOLS=$(( 3 * ETKDG_CAL_SIZE ))
FF_NUM_MOLS=$(( 3 * FF_CAL_SIZE ))
SUBSTRUCT_NUM_MOLS=1250000
RDKIT_MAX_SECONDS=300
FF_MAX_ITERS=200

DATASET_LINES="$(wc -l < "$DATASET")"
if [ "$AUTOTUNE_ENABLED" = "1" ] && [ "$DATASET_LINES" -lt "$SUBSTRUCT_CAL_SIZE" ]; then
  echo "Error: dataset has $DATASET_LINES lines; publication autotuning requires at least" >&2
  echo "       $SUBSTRUCT_CAL_SIZE so the largest substructure configuration gets" >&2
  echo "       $AUTOTUNE_PIPELINE_FILLS complete pipeline fills." >&2
  exit 1
fi

{
  echo "date_utc: $(date -u -Iseconds)"
  echo "date_local: $(date -Iseconds)"
  echo "hostname: $(hostname)"
  echo "uname: $(uname -a)"
  echo "dataset: $DATASET"
  echo "dataset_lines: $DATASET_LINES"
  echo "physical_cores: $PHYSICAL_CORES"
  echo "cpus_available_to_process: $AVAILABLE_CPUS"
  echo "cpu_budget: $CPUS"
  echo "gpu_id_physical: $GPU_ID"
  echo "num_gpus_visible: $GPU_COUNT"
  echo "autotune_enabled: $AUTOTUNE_ENABLED"
  echo "autotune_trials: $AUTOTUNE_TRIALS"
  echo "autotune_seconds_per_trial: $AUTOTUNE_TIME_BUDGET"
  echo "autotune_pipeline_fills: $AUTOTUNE_PIPELINE_FILLS"
  echo "etkdg_calibration_mols: $ETKDG_CAL_SIZE"
  echo "ff_calibration_mols: $FF_CAL_SIZE"
  echo "substruct_calibration_mols: $SUBSTRUCT_CAL_SIZE"
  echo "etkdg_num_mols: $ETKDG_NUM_MOLS"
  echo "ff_num_mols: $FF_NUM_MOLS"
  echo "substruct_num_mols: $SUBSTRUCT_NUM_MOLS"
  echo "skip_rdkit: $SKIP_RDKIT"
  echo "skip_nvmolkit: $SKIP_NVMOLKIT"
  echo "git_commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_status:"
  git -C "$REPO_ROOT" status --short 2>/dev/null || true
} > "$SYSINFO_DIR/run_info.txt"
lscpu > "$SYSINFO_DIR/lscpu.txt" 2>&1 || true
nvidia-smi > "$SYSINFO_DIR/nvidia-smi.txt" 2>&1 || true
nvidia-smi -q > "$SYSINFO_DIR/nvidia-smi-q.txt" 2>&1 || true
python -c "import sys, torch; print('python:', sys.version); print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')" \
  > "$SYSINFO_DIR/python.txt" 2>&1 || true
pip freeze > "$SYSINFO_DIR/pip_freeze.txt" 2>&1 || true

SUMMARY="$OUTPUT_DIR/summary.tsv"
printf "benchmark\tstatus\texit_code\tduration_s\tlog\tresult\n" > "$SUMMARY"
FAILURES=0

run_bench() {
  local name="$1"
  local result_path="$2"
  shift 2
  local log_path="$LOG_DIR/${name}.log"
  local start_s end_s duration code status

  echo "=========================================="
  echo "[$(date -Iseconds)] Running $name"
  printf '  cmd:'
  printf ' %q' "$@"
  echo
  echo "  log: $log_path"
  echo "=========================================="

  start_s="$(date +%s)"
  (cd "$SCRIPT_DIR" && PYTHONUNBUFFERED=1 "$@") > "$log_path" 2>&1
  code=$?
  end_s="$(date +%s)"
  duration=$(( end_s - start_s ))
  if [ "$code" -eq 0 ]; then
    status=ok
  else
    status=fail
    FAILURES=$(( FAILURES + 1 ))
  fi
  printf "%s\t%s\t%d\t%d\t%s\t%s\n" \
    "$name" "$status" "$code" "$duration" "$log_path" "$result_path" >> "$SUMMARY"
  echo "[$name] $status (exit=$code, ${duration}s)"
}

BUTINA_MODE_FLAGS=()
CONFORMER_RMSD_MODE_FLAGS=()
CROSS_SIMILARITY_MODE_FLAGS=()
ETKDG_MODE_FLAGS=()
FF_MODE_FLAGS=()
SUBSTRUCT_MODE_FLAGS=()
TFD_MODE_FLAGS=(--verify)
if [ "$SKIP_RDKIT" = "1" ]; then
  BUTINA_MODE_FLAGS=(--no-rdkit)
  CONFORMER_RMSD_MODE_FLAGS=(--no_rdkit --no_validate)
  CROSS_SIMILARITY_MODE_FLAGS=(--no-rdkit)
  ETKDG_MODE_FLAGS=(--no_rdkit --no_validate)
  FF_MODE_FLAGS=(--no_rdkit --no_validate)
  SUBSTRUCT_MODE_FLAGS=(--no_rdkit --no_validate)
  TFD_MODE_FLAGS=(--skip-rdkit)
elif [ "$SKIP_NVMOLKIT" = "1" ]; then
  BUTINA_MODE_FLAGS=(--no-nvmolkit --no-fused)
  CONFORMER_RMSD_MODE_FLAGS=(--no_nvmolkit --no_validate)
  CROSS_SIMILARITY_MODE_FLAGS=(--no-nvmolkit)
  ETKDG_MODE_FLAGS=(--no_nvmolkit --no_validate)
  FF_MODE_FLAGS=(--no_nvmolkit --no_validate)
  SUBSTRUCT_MODE_FLAGS=(--no_nvmolkit --no_validate)
  TFD_MODE_FLAGS=(--skip-nvmolkit)
fi

ETKDG_AUTOTUNE_FLAGS=()
FF_AUTOTUNE_FLAGS=()
SUBSTRUCT_AUTOTUNE_FLAGS=()
if [ "$AUTOTUNE_ENABLED" = "1" ]; then
  ETKDG_AUTOTUNE_FLAGS=(
    --autotune
    --autotune_trials "$AUTOTUNE_TRIALS"
    --autotune_time_budget "$AUTOTUNE_TIME_BUDGET"
    --autotune_cpu_budget "$CPUS"
    --autotune_calibration_size "$ETKDG_CAL_SIZE"
  )
  FF_AUTOTUNE_FLAGS=(
    --autotune
    --autotune_trials "$AUTOTUNE_TRIALS"
    --autotune_time_budget "$AUTOTUNE_TIME_BUDGET"
    --autotune_cpu_budget "$CPUS"
    --autotune_calibration_size "$FF_CAL_SIZE"
  )
  SUBSTRUCT_AUTOTUNE_FLAGS=(
    --autotune
    --autotune_trials "$AUTOTUNE_TRIALS"
    --autotune_time_budget "$AUTOTUNE_TIME_BUDGET"
    --autotune_cpu_budget "$CPUS"
    --autotune_calibration_size "$SUBSTRUCT_CAL_SIZE"
  )
fi

autotune_save_arg() {
  local path="$1"
  if [ "$AUTOTUNE_ENABLED" = "1" ]; then
    AUTOTUNE_SAVE_FLAGS=(--autotune_save "$path")
  else
    AUTOTUNE_SAVE_FLAGS=()
  fi
}

run_bench butina_clustering "$RESULT_DIR/butina_clustering.csv" \
  python "$SCRIPT_DIR/butina_clustering_bench.py" "$DATASET" \
  --nvmolkit-reordering both \
  --output "$RESULT_DIR/butina_clustering.csv" \
  "${BUTINA_MODE_FLAGS[@]}"

run_bench conformer_rmsd "$RESULT_DIR/conformer_rmsd.csv" \
  python "$SCRIPT_DIR/conformer_rmsd_bench.py" \
  --smiles "$DATASET" \
  --num_mols 2000 \
  --confs_per_mol 10 25 50 100 200 \
  --prep_workers "$CPUS" \
  --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
  --output "$RESULT_DIR/conformer_rmsd.csv" \
  "${CONFORMER_RMSD_MODE_FLAGS[@]}"

run_bench cross_similarity "$RESULT_DIR/cross_similarity.json" \
  python "$SCRIPT_DIR/cross_similarity_bench.py" \
  --input "$DATASET" \
  --cosine \
  --output "$RESULT_DIR/cross_similarity.json" \
  "${CROSS_SIMILARITY_MODE_FLAGS[@]}"

autotune_save_arg "$AUTOTUNE_DIR/etkdg_hardware.json"
run_bench etkdg "$RESULT_DIR/etkdg.csv" \
  python "$SCRIPT_DIR/etkdg_bench.py" \
  --smiles "$DATASET" \
  --num_mols "$ETKDG_NUM_MOLS" \
  --confs_per_mol "$CONFS_PER_MOL" \
  --num_gpus 1 \
  --rdkit_threads "$CPUS" \
  --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
  "${ETKDG_AUTOTUNE_FLAGS[@]}" \
  "${AUTOTUNE_SAVE_FLAGS[@]}" \
  --output "$RESULT_DIR/etkdg.csv" \
  "${ETKDG_MODE_FLAGS[@]}"

FF_ROWS=("mmff:BFGS" "mmff:FIRE" "uff:BFGS" "uff:FIRE")
for row in "${FF_ROWS[@]}"; do
  ff="${row%%:*}"
  minimizer_kind="${row##*:}"
  minimizer_stem="${minimizer_kind,,}"
  bench_name="ff_optimize_${ff}_${minimizer_stem}"
  autotune_save_arg "$AUTOTUNE_DIR/${bench_name}_hardware.json"
  run_bench "$bench_name" "$RESULT_DIR/${bench_name}.csv" \
    python "$SCRIPT_DIR/ff_optimize_bench.py" \
    --smiles "$DATASET" \
    --num_mols "$FF_NUM_MOLS" \
    --confs_per_mol "$CONFS_PER_MOL" \
    --ff "$ff" \
    --minimizer_kind "$minimizer_kind" \
    --max_iters "$FF_MAX_ITERS" \
    --num_gpus 1 \
    --rdkit_threads "$CPUS" \
    --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
    "${FF_AUTOTUNE_FLAGS[@]}" \
    "${AUTOTUNE_SAVE_FLAGS[@]}" \
    --output "$RESULT_DIR/${bench_name}.csv" \
    "${FF_MODE_FLAGS[@]}"
done

SUBSTRUCT_ROWS=(
  "rdkit_fragment_descriptors_supported.txt:countSubstructMatches"
  "wehi_pains_supported.txt:hasSubstructMatch"
  "BMS_2006_filter_supported.txt:hasSubstructMatch"
  "rdkit_tautomer_transforms_supported.txt:getSubstructMatches"
  "rdkit_torsionPreferences_v2_supported.txt:getSubstructMatches"
)
for row in "${SUBSTRUCT_ROWS[@]}"; do
  smarts_file="${row%%:*}"
  mode="${row##*:}"
  smarts_stem="${smarts_file%.txt}"
  bench_name="substruct_${smarts_stem}"
  autotune_save_arg "$AUTOTUNE_DIR/${bench_name}_config.json"
  run_bench "$bench_name" "$LOG_DIR/${bench_name}.log" \
    python "$SCRIPT_DIR/substruct_bench.py" \
    --smiles "$DATASET" \
    --num_mols "$SUBSTRUCT_NUM_MOLS" \
    --sanitize \
    --smarts "$SMARTS_DIR/$smarts_file" \
    --mode "$mode" \
    --num_gpus 1 \
    --rdkit_threads "$CPUS" \
    --rdkit_match_mode raw substructlib \
    --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
    "${SUBSTRUCT_AUTOTUNE_FLAGS[@]}" \
    "${AUTOTUNE_SAVE_FLAGS[@]}" \
    "${SUBSTRUCT_MODE_FLAGS[@]}"
done

run_bench tfd "$RESULT_DIR/tfd.csv" \
  python "$SCRIPT_DIR/tfd_bench.py" \
  --smiles-file "$DATASET" \
  --prep-workers "$CPUS" \
  --output "$RESULT_DIR/tfd.csv" \
  "${TFD_MODE_FLAGS[@]}"

echo
echo "Publication benchmarks complete. Output: $OUTPUT_DIR"
echo "Summary:"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || sed -n '1,200p' "$SUMMARY"
if [ "$FAILURES" -ne 0 ]; then
  echo "Error: $FAILURES benchmark(s) failed" >&2
  exit 1
fi
