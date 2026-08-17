#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run ETKDG and force-field optimization benchmarks across heavy-atom-count
# bins. Each bin is autotuned independently and writes its own result/config.

set -uo pipefail

usage() {
  cat >&2 <<EOF
Usage: $0 --output-dir DIR [--num-gpus N] [--gpu-ids LIST] [--data-dir DIR]
          [--include NAME [NAME ...]] [--no-rdkit | --no-nvmolkit] [--continue]
       $0 <N> --output-dir DIR ...  (legacy positional GPU count)
       $0 --list

  --num-gpus N      GPUs used by each benchmark (default: 1)
  --gpu-ids LIST    Comma-separated physical GPU IDs to expose
  --output-dir DIR  Results directory (required for runs)
  --data-dir DIR    Directory containing chembl_size_splits (default: /data)
  --include NAME... Whitelist of size scans to run (run --list to see them)
  --no-rdkit        Skip RDKit timing and validation
  --no-nvmolkit     Run RDKit only and disable autotuning
  --continue        Skip bin rows already marked status=ok in summary.tsv
  --list            Print size-scan names and exit
EOF
}

NUM_GPUS=1
GPU_IDS=""
OUTPUT_DIR=""
DATA_DIR="/data"
INCLUDE_LIST=()
SKIP_RDKIT=0
SKIP_NVMOLKIT=0
CONTINUE=0
LIST_ONLY=0

if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
  NUM_GPUS="$1"
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_WORKDIR="${TMPDIR:-/tmp}"

SIZE_SCAN_BINS=("0-20" "20-40" "40-60" "60-80" "80-100")
ALL_BENCH_NAMES=(
  "etkdg_size_scan"
  "ff_optimize_mmff_bfgs_size_scan"
  "ff_optimize_mmff_fire_size_scan"
  "ff_optimize_uff_bfgs_size_scan"
  "ff_optimize_uff_fire_size_scan"
)

while [ $# -gt 0 ]; do
  case "$1" in
    --num-gpus)
      [ $# -ge 2 ] || { echo "Error: --num-gpus requires a value" >&2; exit 2; }
      NUM_GPUS="$2"
      shift 2
      ;;
    --gpu-ids)
      [ $# -ge 2 ] || { echo "Error: --gpu-ids requires a value" >&2; exit 2; }
      GPU_IDS="$2"
      shift 2
      ;;
    --output-dir)
      [ $# -ge 2 ] || { echo "Error: --output-dir requires a value" >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --data-dir)
      [ $# -ge 2 ] || { echo "Error: --data-dir requires a value" >&2; exit 2; }
      DATA_DIR="$2"
      shift 2
      ;;
    --include)
      shift
      if [ $# -eq 0 ] || [[ "$1" == --* ]]; then
        echo "Error: --include requires at least one benchmark name" >&2
        exit 2
      fi
      while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
        INCLUDE_LIST+=("$1")
        shift
      done
      ;;
    --no-rdkit)
      SKIP_RDKIT=1
      shift
      ;;
    --no-nvmolkit)
      SKIP_NVMOLKIT=1
      shift
      ;;
    --continue)
      CONTINUE=1
      shift
      ;;
    --list)
      LIST_ONLY=1
      shift
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

if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --num-gpus must be a positive integer" >&2
  exit 2
fi
if [ -n "$GPU_IDS" ]; then
  if ! [[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Error: --gpu-ids must be a comma-separated list of non-negative integers" >&2
    exit 2
  fi
  IFS=',' read -r -a REQUESTED_GPU_IDS <<< "$GPU_IDS"
  if [ "${#REQUESTED_GPU_IDS[@]}" -ne "$NUM_GPUS" ]; then
    echo "Error: --gpu-ids contains ${#REQUESTED_GPU_IDS[@]} IDs but --num-gpus is $NUM_GPUS" >&2
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi
if [ "$SKIP_RDKIT" = "1" ] && [ "$SKIP_NVMOLKIT" = "1" ]; then
  echo "Error: --no-rdkit and --no-nvmolkit are mutually exclusive" >&2
  exit 2
fi
if [ "$LIST_ONLY" = "1" ]; then
  printf '%s\n' "${ALL_BENCH_NAMES[@]}"
  exit 0
fi
if [ -z "$OUTPUT_DIR" ]; then
  echo "Error: --output-dir is required" >&2
  usage
  exit 2
fi

should_run() {
  local name="$1"
  if [ "${#INCLUDE_LIST[@]}" -eq 0 ]; then
    return 0
  fi
  local entry
  for entry in "${INCLUDE_LIST[@]}"; do
    [ "$entry" = "$name" ] && return 0
  done
  return 1
}

for requested in "${INCLUDE_LIST[@]}"; do
  known=0
  for candidate in "${ALL_BENCH_NAMES[@]}"; do
    [ "$requested" = "$candidate" ] && known=1
  done
  if [ "$known" = "0" ]; then
    echo "Error: --include name '$requested' is not a known size scan" >&2
    printf '  %s\n' "${ALL_BENCH_NAMES[@]}" >&2
    exit 2
  fi
done

if [ ! -d "$DATA_DIR/chembl_size_splits" ]; then
  echo "Error: missing $DATA_DIR/chembl_size_splits" >&2
  exit 1
fi
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
SIZE_SCAN_DIR="$DATA_DIR/chembl_size_splits"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
LOG_DIR="$OUTPUT_DIR/logs"
RESULT_DIR="$OUTPUT_DIR/results"
AUTOTUNE_DIR="$OUTPUT_DIR/autotune"
SYSINFO_DIR="$OUTPUT_DIR/sysinfo"
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$AUTOTUNE_DIR" "$SYSINFO_DIR"

GPU_COUNT="not_checked"
if [ "$SKIP_NVMOLKIT" = "0" ]; then
  if ! GPU_COUNT="$(cd "$PYTHON_WORKDIR" && python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null)"; then
    echo "Error: could not query CUDA device count through torch" >&2
    exit 1
  fi
  if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
    echo "Error: requested $NUM_GPUS GPUs but torch reports $GPU_COUNT visible" >&2
    echo "       CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
    exit 1
  fi
fi

ensure_pip_pkg() {
  local import_name="$1"
  local pip_name="${2:-$1}"
  if (cd "$PYTHON_WORKDIR" && python -c "import $import_name" 2>/dev/null); then
    return 0
  fi
  echo "[deps] $import_name not importable; installing $pip_name"
  (cd "$PYTHON_WORKDIR" && pip install --quiet "$pip_name")
}
ensure_pip_pkg nvtx || exit 1
ensure_pip_pkg pandas || exit 1
if [ "$SKIP_NVMOLKIT" = "0" ]; then
  ensure_pip_pkg optuna || exit 1
fi

CONFS_PER_MOL=200
FF_MAX_ITERS=200
ETKDG_AUTOTUNE_BS_MAX=1024
FF_AUTOTUNE_BS_MAX=4096
AUTOTUNE_BPG_MAX=8
AUTOTUNE_TRIALS=20
AUTOTUNE_TIME_BUDGET=10
BENCHMARK_SEED=42
TIMING_RUNS=3
RDKIT_MAX_SECONDS=300

etkdg_num_mols() { case "$1" in 0-20) echo 80000;; 20-40) echo 47700;; 40-60) echo 15600;; 60-80) echo 5940;; 80-100) echo 2590;; esac; }
etkdg_cal_mols() { case "$1" in 0-20) echo 16000;; 20-40) echo 3980;; 40-60) echo 1300;; 60-80) echo 500;; 80-100) echo 220;; esac; }
ff_num_mols() { case "$1" in 0-20) echo 80000;; 20-40) echo 73500;; 40-60) echo 42300;; 60-80) echo 26400;; 80-100) echo 18700;; esac; }
ff_cal_mols() { case "$1" in 0-20) echo 14000;; 20-40) echo 6130;; 40-60) echo 3530;; 60-80) echo 2200;; 80-100) echo 1560;; esac; }

PHYSICAL_CORES="$(lscpu -p=Core,Socket 2>/dev/null | awk -F, '!/^#/ {print $1 "," $2}' | sort -u | wc -l)"
if [ -z "$PHYSICAL_CORES" ] || [ "$PHYSICAL_CORES" -lt 1 ]; then
  PHYSICAL_CORES=1
fi
if [ "$NUM_GPUS" -eq 1 ] && [ "$PHYSICAL_CORES" -gt 16 ]; then
  RDKIT_THREADS=16
else
  RDKIT_THREADS="$PHYSICAL_CORES"
fi

{
  echo "date_utc: $(date -u -Iseconds)"
  echo "git_commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "num_gpus_requested: $NUM_GPUS"
  echo "num_gpus_visible: $GPU_COUNT"
  echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "physical_cores: $PHYSICAL_CORES"
  echo "rdkit_threads: $RDKIT_THREADS"
  echo "size_scan_dir: $SIZE_SCAN_DIR"
  echo "size_scan_bins: ${SIZE_SCAN_BINS[*]}"
  for bin in "${SIZE_SCAN_BINS[@]}"; do
    echo "etkdg_${bin}_num_mols: $(etkdg_num_mols "$bin")"
    echo "etkdg_${bin}_calibration_mols: $(etkdg_cal_mols "$bin")"
    echo "ff_${bin}_num_mols: $(ff_num_mols "$bin")"
    echo "ff_${bin}_calibration_mols: $(ff_cal_mols "$bin")"
  done
  echo "autotune_trials: $AUTOTUNE_TRIALS"
  echo "autotune_seconds_per_trial: $AUTOTUNE_TIME_BUDGET"
  echo "benchmark_seed: $BENCHMARK_SEED"
  echo "timing_runs: $TIMING_RUNS"
} > "$SYSINFO_DIR/run_info.txt"
lscpu > "$SYSINFO_DIR/lscpu.txt" 2>&1 || true
nvidia-smi > "$SYSINFO_DIR/nvidia-smi.txt" 2>&1 || true

SUMMARY="$OUTPUT_DIR/summary.tsv"
COMPLETED_BENCHES=""
if [ "$CONTINUE" = "1" ] && [ -f "$SUMMARY" ]; then
  COMPLETED_BENCHES="$(awk -F'\t' 'NR>1 && $2=="ok" {print $1}' "$SUMMARY")"
else
  printf "benchmark\tstatus\texit_code\tduration_s\tlog\tresult\n" > "$SUMMARY"
fi
FAILURES=0

is_completed() {
  local name="$1"
  local entry
  for entry in $COMPLETED_BENCHES; do
    [ "$entry" = "$name" ] && return 0
  done
  return 1
}

run_bench() {
  local name="$1"
  local result_path="$2"
  shift 2
  local log_path="$LOG_DIR/${name}.log"
  local start_s end_s duration code status
  if is_completed "$name"; then
    echo "[$name] skipped (--continue: already ok)"
    return 0
  fi
  echo "[$(date -Iseconds)] Running $name"
  printf '  cmd:'
  printf ' %q' "$@"
  echo
  start_s="$(date +%s)"
  (cd "$PYTHON_WORKDIR" && PYTHONUNBUFFERED=1 "$@") > "$log_path" 2>&1
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

ETKDG_MODE_FLAGS=()
FF_MODE_FLAGS=()
AUTOTUNE_ENABLED=1
if [ "$SKIP_RDKIT" = "1" ]; then
  ETKDG_MODE_FLAGS=(--no_rdkit --no_validate)
  FF_MODE_FLAGS=(--no_rdkit --no_validate)
elif [ "$SKIP_NVMOLKIT" = "1" ]; then
  ETKDG_MODE_FLAGS=(--no_nvmolkit --no_validate)
  FF_MODE_FLAGS=(--no_nvmolkit --no_validate)
  AUTOTUNE_ENABLED=0
fi

autotune_save_arg() {
  local path="$1"
  if [ "$AUTOTUNE_ENABLED" = "1" ]; then
    AUTOTUNE_SAVE_FLAGS=(--autotune_save "$path")
  else
    AUTOTUNE_SAVE_FLAGS=()
  fi
}

if should_run etkdg_size_scan; then
  for bin in "${SIZE_SCAN_BINS[@]}"; do
    bin_smi="$SIZE_SCAN_DIR/chembl_${bin}.smi"
    if [ ! -f "$bin_smi" ]; then
      echo "[etkdg_size_scan_${bin}] skipped (missing $bin_smi)"
      printf "%s\tskipped\t0\t0\t\t\n" "etkdg_size_scan_${bin}" >> "$SUMMARY"
      continue
    fi
    name="etkdg_size_scan_${bin}"
    num_mols="$(etkdg_num_mols "$bin")"
    cal_mols="$(etkdg_cal_mols "$bin")"
    ETKDG_AUTOTUNE_FLAGS=()
    if [ "$AUTOTUNE_ENABLED" = "1" ]; then
      ETKDG_AUTOTUNE_FLAGS=(--autotune --autotune_trials "$AUTOTUNE_TRIALS" --autotune_time_budget "$AUTOTUNE_TIME_BUDGET" --autotune_calibration_size "$cal_mols")
    fi
    autotune_save_arg "$AUTOTUNE_DIR/${name}_hardware.json"
    run_bench "$name" "$RESULT_DIR/${name}.csv" \
      python "$SCRIPT_DIR/etkdg_bench.py" \
      --smiles "$bin_smi" --num_mols "$num_mols" --seed "$BENCHMARK_SEED" \
      --runs "$TIMING_RUNS" --confs_per_mol "$CONFS_PER_MOL" \
      --num_gpus "$NUM_GPUS" --rdkit_threads "$RDKIT_THREADS" \
      --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
      "${ETKDG_AUTOTUNE_FLAGS[@]}" "${AUTOTUNE_SAVE_FLAGS[@]}" \
      --output "$RESULT_DIR/${name}.csv" "${ETKDG_MODE_FLAGS[@]}"
  done
fi

FF_ROWS=("mmff:BFGS" "mmff:FIRE" "uff:BFGS" "uff:FIRE")
for row in "${FF_ROWS[@]}"; do
  ff="${row%%:*}"
  minimizer_kind="${row##*:}"
  scan_name="ff_optimize_${ff}_${minimizer_kind,,}_size_scan"
  should_run "$scan_name" || continue
  for bin in "${SIZE_SCAN_BINS[@]}"; do
    bin_smi="$SIZE_SCAN_DIR/chembl_${bin}.smi"
    name="${scan_name}_${bin}"
    if [ ! -f "$bin_smi" ]; then
      echo "[$name] skipped (missing $bin_smi)"
      printf "%s\tskipped\t0\t0\t\t\n" "$name" >> "$SUMMARY"
      continue
    fi
    num_mols="$(ff_num_mols "$bin")"
    cal_mols="$(ff_cal_mols "$bin")"
    FF_AUTOTUNE_FLAGS=()
    if [ "$AUTOTUNE_ENABLED" = "1" ]; then
      FF_AUTOTUNE_FLAGS=(--autotune --autotune_trials "$AUTOTUNE_TRIALS" --autotune_time_budget "$AUTOTUNE_TIME_BUDGET" --autotune_calibration_size "$cal_mols")
    fi
    autotune_save_arg "$AUTOTUNE_DIR/${name}_hardware.json"
    run_bench "$name" "$RESULT_DIR/${name}.csv" \
      python "$SCRIPT_DIR/ff_optimize_bench.py" \
      --smiles "$bin_smi" --num_mols "$num_mols" --seed "$BENCHMARK_SEED" \
      --runs "$TIMING_RUNS" --confs_per_mol "$CONFS_PER_MOL" \
      --ff "$ff" --minimizer_kind "$minimizer_kind" --max_iters "$FF_MAX_ITERS" \
      --num_gpus "$NUM_GPUS" --rdkit_threads "$RDKIT_THREADS" \
      --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
      "${FF_AUTOTUNE_FLAGS[@]}" "${AUTOTUNE_SAVE_FLAGS[@]}" \
      --output "$RESULT_DIR/${name}.csv" "${FF_MODE_FLAGS[@]}"
  done
done

echo
echo "Size-scan benchmarks complete. Output: $OUTPUT_DIR"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || sed -n '1,200p' "$SUMMARY"
if [ "$FAILURES" -ne 0 ]; then
  echo "Error: $FAILURES benchmark bin(s) failed" >&2
  exit 1
fi
