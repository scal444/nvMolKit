#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the multi-GPU production benchmarks with literal, previously measured
# configurations. This runner never autotunes. Workloads target roughly
# 20-60 seconds per timed sample on 8x H200; 8x B200 should be similar or
# faster. Each benchmark records three timed samples.

set -uo pipefail

usage() {
  cat >&2 <<EOF
Usage: $0 --output-dir DIR [--data-dir DIR] [--gpu-ids LIST]
          [--include NAME [NAME ...]] [--continue] [--dry-run]
       $0 --list

Fixed 8-GPU, nvMolKit-only production runner. It does not invoke autotuning.

  --output-dir DIR  Fresh results directory (required)
  --data-dir DIR    Directory containing enamine_real_10M.cxsmiles (default: /data)
  --gpu-ids LIST    Eight comma-separated physical GPU IDs (default: 0,1,2,3,4,5,6,7)
  --include NAME... Run only the named benchmarks
  --continue        Skip benchmarks already marked ok in summary.tsv
  --dry-run         Print commands without executing them
  --list            Print benchmark names and exit
EOF
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SMARTS_DIR="$REPO_ROOT/tests/test_data/SMARTS"

DATA_DIR=/data
OUTPUT_DIR=""
GPU_IDS=0,1,2,3,4,5,6,7
CONTINUE=0
DRY_RUN=0
LIST_ONLY=0
INCLUDE_LIST=()

BENCHMARKS=(
  etkdg
  ff_optimize_mmff_bfgs
  ff_optimize_mmff_fire
  ff_optimize_uff_bfgs
  ff_optimize_uff_fire
  mcs_lax_lax
  mcs_lax_strict
  mcs_strict_lax
  mcs_strict_strict
  mcs_strict_strict_ring
  substruct
)

while [ $# -gt 0 ]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="${2:?--output-dir requires a value}"; shift 2 ;;
    --data-dir) DATA_DIR="${2:?--data-dir requires a value}"; shift 2 ;;
    --gpu-ids) GPU_IDS="${2:?--gpu-ids requires a value}"; shift 2 ;;
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
    --continue) CONTINUE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --list) LIST_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Error: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [ "$LIST_ONLY" = 1 ]; then
  printf '%s\n' "${BENCHMARKS[@]}"
  exit 0
fi
if [ -z "$OUTPUT_DIR" ]; then
  echo "Error: --output-dir is required" >&2
  exit 2
fi
if ! [[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+){7}$ ]]; then
  echo "Error: --gpu-ids must contain exactly eight comma-separated GPU IDs" >&2
  exit 2
fi

for requested in "${INCLUDE_LIST[@]}"; do
  known=0
  for name in "${BENCHMARKS[@]}"; do
    if [ "$requested" = "$name" ]; then known=1; break; fi
  done
  if [ "$known" = 0 ]; then
    echo "Error: unknown benchmark: $requested" >&2
    exit 2
  fi
done

should_run() {
  local requested="$1"
  if [ "${#INCLUDE_LIST[@]}" -eq 0 ]; then return 0; fi
  local name
  for name in "${INCLUDE_LIST[@]}"; do
    if [ "$requested" = "$name" ]; then return 0; fi
  done
  return 1
}

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
NUM_GPUS=8
SEED=42
RUNS=3
ENAMINE="$DATA_DIR/enamine_real_10M.cxsmiles"

# Literal production workloads, sized from the slower genuine-H200 results.
ETKDG_NUM_MOLS=3500
MMFF_NUM_MOLS=12000
UFF_NUM_MOLS=6000
MCS_NUM_MOLS=10000
MCS_STRICT_NUM_PAIRS=8000000
SUBSTRUCT_NUM_MOLS=10000000

# MCS parameter rows: (benchmark name, atom compare, bond compare, ring-only, num-pairs).
# Strict is limited to element/order matching.
MCS_ROWS=(
  "mcs_lax_lax:any:any:0:800000"
  "mcs_lax_strict:any:order:0:1600000"
  "mcs_strict_lax:elements:any:0:1600000"
  "mcs_strict_strict:elements:order:0:$MCS_STRICT_NUM_PAIRS"
  "mcs_strict_strict_ring:elements:order:1:$MCS_STRICT_NUM_PAIRS"
)

if [ "$DRY_RUN" = 0 ]; then
  if [ ! -f "$ENAMINE" ]; then
    echo "Error: missing $ENAMINE" >&2
    exit 1
  fi
  visible_gpus="$(python -c 'import torch; print(torch.cuda.device_count())')" || exit 1
  if [ "$visible_gpus" -ne 8 ]; then
    echo "Error: fixed runner requires 8 visible GPUs; torch reports $visible_gpus" >&2
    exit 1
  fi
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
LOG_DIR="$OUTPUT_DIR/logs"
RESULT_DIR="$OUTPUT_DIR/results"
SYSINFO_DIR="$OUTPUT_DIR/sysinfo"
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$SYSINFO_DIR"
SUMMARY="$OUTPUT_DIR/summary.tsv"
SUBSTRUCT_CONFIG="$OUTPUT_DIR/substruct_config.csv"

# Scale the measured one-GPU/14-core GSI profiles to the fixed 8-GPU CPU
# budget. workerThreads is per GPU, while preprocessingThreads is a total pool,
# so preserving the measured split requires:
#   preprocessingThreads = 14 * numGpus - workerThreads * numGpus
# Apply the same per-dataset hardware profile to DFS until it has an equivalent
# fixed-budget sweep of its own.
SUBSTRUCT_CPU_BUDGET=$(( 14 * NUM_GPUS ))
SUBSTRUCT_PREP_W4=$(( SUBSTRUCT_CPU_BUDGET - 4 * NUM_GPUS ))
SUBSTRUCT_PREP_W8=$(( SUBSTRUCT_CPU_BUDGET - 8 * NUM_GPUS ))

{
  echo "smarts,batch_size,workers,prep_threads,mode,num_gpus,algorithm"
  for algorithm in gsi dfs; do
    echo "$SMARTS_DIR/rdkit_fragment_descriptors_supported.txt,8192,4,$SUBSTRUCT_PREP_W4,countSubstructMatches,$NUM_GPUS,$algorithm"
    echo "$SMARTS_DIR/wehi_pains_supported.txt,8192,4,$SUBSTRUCT_PREP_W4,hasSubstructMatch,$NUM_GPUS,$algorithm"
    echo "$SMARTS_DIR/BMS_2006_filter_supported.txt,8192,4,$SUBSTRUCT_PREP_W4,hasSubstructMatch,$NUM_GPUS,$algorithm"
    echo "$SMARTS_DIR/rdkit_tautomer_transforms_supported.txt,2048,4,$SUBSTRUCT_PREP_W4,getSubstructMatches,$NUM_GPUS,$algorithm"
    echo "$SMARTS_DIR/rdkit_torsionPreferences_v2_supported.txt,4096,8,$SUBSTRUCT_PREP_W8,getSubstructMatches,$NUM_GPUS,$algorithm"
  done
} > "$SUBSTRUCT_CONFIG"

if [ "$CONTINUE" = 0 ] || [ ! -f "$SUMMARY" ]; then
  printf 'benchmark\tstatus\texit_code\tduration_s\tlog\tresult\n' > "$SUMMARY"
fi

is_complete() {
  [ "$CONTINUE" = 1 ] && [ -f "$SUMMARY" ] &&
    awk -F '\t' -v name="$1" 'NR > 1 && $1 == name && $2 == "ok" { found=1 } END { exit !found }' "$SUMMARY"
}

run_bench() {
  local name="$1"
  local result="$2"
  shift 2
  local log="$LOG_DIR/$name.log"

  if ! should_run "$name"; then return 0; fi
  if is_complete "$name"; then
    echo "[$name] skipped (--continue: already ok)"
    return 0
  fi

  echo "=========================================="
  echo "[$(date -Iseconds)] Running $name"
  printf '  cmd:'
  printf ' %q' "$@"
  echo
  echo "  log: $log"
  echo "=========================================="
  if [ "$DRY_RUN" = 1 ]; then return 0; fi

  local start end code status
  start="$(date +%s)"
  (cd "$OUTPUT_DIR" && PYTHONUNBUFFERED=1 "$@") > "$log" 2>&1
  code=$?
  end="$(date +%s)"
  if [ "$code" -eq 0 ]; then status=ok; else status=fail; fi
  printf '%s\t%s\t%d\t%d\t%s\t%s\n' \
    "$name" "$status" "$code" "$((end - start))" "$log" "$result" >> "$SUMMARY"
  echo "[$name] $status (exit=$code, $((end - start))s)"
}

if [ "$DRY_RUN" = 0 ]; then
  {
    echo "date_utc: $(date -u -Iseconds)"
    echo "hostname: $(hostname)"
    echo "cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
    echo "git_commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "timing_runs: $RUNS"
    echo "config_source: fixed_known_best"
    echo "etkdg: num_mols=$ETKDG_NUM_MOLS batch_size=1024 batches_per_gpu=5 prep_threads=92"
    echo "mmff_bfgs: num_mols=$MMFF_NUM_MOLS batch_size=1280 batches_per_gpu=5"
    echo "mmff_fire: num_mols=$MMFF_NUM_MOLS batch_size=1600 batches_per_gpu=6"
    echo "uff_bfgs: num_mols=$UFF_NUM_MOLS batch_size=448 batches_per_gpu=4"
    echo "uff_fire: num_mols=$UFF_NUM_MOLS batch_size=512 batches_per_gpu=3"
    echo "mcs: num_mols=$MCS_NUM_MOLS batch_size=4096 workers=4 prep_threads=72 executors_per_runner=4"
    echo "mcs_parameter_sets: ${MCS_ROWS[*]}"
    echo "substruct: num_mols=$SUBSTRUCT_NUM_MOLS cpu_budget=$SUBSTRUCT_CPU_BUDGET config=$SUBSTRUCT_CONFIG"
    echo "substruct_profile_source: scaled from one-GPU 14-core GSI sweep on 2026-08-16; also applied to DFS"
  } > "$SYSINFO_DIR/run_info.txt"
  nvidia-smi > "$SYSINFO_DIR/nvidia-smi.txt" 2>&1 || true
  lscpu > "$SYSINFO_DIR/lscpu.txt" 2>&1 || true
fi

run_bench etkdg "$RESULT_DIR/etkdg.csv" \
  python "$SCRIPT_DIR/etkdg_bench.py" \
  --smiles "$ENAMINE" --num_mols "$ETKDG_NUM_MOLS" --seed "$SEED" \
  --runs "$RUNS" --confs_per_mol 200 --num_gpus "$NUM_GPUS" \
  --batch_size 1024 --batches_per_gpu 5 --prep_threads 92 \
  --output "$RESULT_DIR/etkdg.csv" --no_rdkit --no_validate

run_bench ff_optimize_mmff_bfgs "$RESULT_DIR/ff_optimize_mmff_bfgs.csv" \
  python "$SCRIPT_DIR/ff_optimize_bench.py" \
  --smiles "$ENAMINE" --num_mols "$MMFF_NUM_MOLS" --seed "$SEED" \
  --runs "$RUNS" --confs_per_mol 200 --ff mmff --minimizer_kind BFGS --max_iters 200 \
  --num_gpus "$NUM_GPUS" --batch_size 1280 --batches_per_gpu 5 \
  --output "$RESULT_DIR/ff_optimize_mmff_bfgs.csv" --no_rdkit --no_validate

run_bench ff_optimize_mmff_fire "$RESULT_DIR/ff_optimize_mmff_fire.csv" \
  python "$SCRIPT_DIR/ff_optimize_bench.py" \
  --smiles "$ENAMINE" --num_mols "$MMFF_NUM_MOLS" --seed "$SEED" \
  --runs "$RUNS" --confs_per_mol 200 --ff mmff --minimizer_kind FIRE --max_iters 200 \
  --num_gpus "$NUM_GPUS" --batch_size 1600 --batches_per_gpu 6 \
  --output "$RESULT_DIR/ff_optimize_mmff_fire.csv" --no_rdkit --no_validate

run_bench ff_optimize_uff_bfgs "$RESULT_DIR/ff_optimize_uff_bfgs.csv" \
  python "$SCRIPT_DIR/ff_optimize_bench.py" \
  --smiles "$ENAMINE" --num_mols "$UFF_NUM_MOLS" --seed "$SEED" \
  --runs "$RUNS" --confs_per_mol 200 --ff uff --minimizer_kind BFGS --max_iters 200 \
  --num_gpus "$NUM_GPUS" --batch_size 448 --batches_per_gpu 4 \
  --output "$RESULT_DIR/ff_optimize_uff_bfgs.csv" --no_rdkit --no_validate

run_bench ff_optimize_uff_fire "$RESULT_DIR/ff_optimize_uff_fire.csv" \
  python "$SCRIPT_DIR/ff_optimize_bench.py" \
  --smiles "$ENAMINE" --num_mols "$UFF_NUM_MOLS" --seed "$SEED" \
  --runs "$RUNS" --confs_per_mol 200 --ff uff --minimizer_kind FIRE --max_iters 200 \
  --num_gpus "$NUM_GPUS" --batch_size 512 --batches_per_gpu 3 \
  --output "$RESULT_DIR/ff_optimize_uff_fire.csv" --no_rdkit --no_validate

for row in "${MCS_ROWS[@]}"; do
  IFS=: read -r mcs_name atom_compare bond_compare ring_only num_pairs <<< "$row"
  MCS_PARAMETER_FLAGS=(
    --atom_compare "$atom_compare"
    --bond_compare "$bond_compare"
  )
  if [ "$ring_only" = "1" ]; then
    MCS_PARAMETER_FLAGS+=(--ring_matches_ring_only)
  fi
  run_bench "$mcs_name" "$RESULT_DIR/${mcs_name}.csv" \
    python "$SCRIPT_DIR/mcs_bench.py" \
    --smiles "$ENAMINE" --num_mols "$MCS_NUM_MOLS" --num_pairs "$num_pairs" \
    --seed "$SEED" --runs "$RUNS" --num_gpus "$NUM_GPUS" \
    --batch_size 4096 --workers 4 --prep_threads 72 --executors_per_runner 2 \
    "${MCS_PARAMETER_FLAGS[@]}" \
    --output "$RESULT_DIR/${mcs_name}.csv" --no_rdkit --no_validate
done

run_bench substruct "$RESULT_DIR/substruct.csv" \
  python "$SCRIPT_DIR/substruct_bench.py" \
  --smiles "$ENAMINE" --config "$SUBSTRUCT_CONFIG" \
  --num_mols "$SUBSTRUCT_NUM_MOLS" --seed "$SEED" --runs "$RUNS" \
  --output "$RESULT_DIR/substruct.csv" --no_rdkit --no_validate

if [ "$DRY_RUN" = 0 ]; then
  failures="$(awk -F '\t' 'NR > 1 && $2 == "fail" { n++ } END { print n+0 }' "$SUMMARY")"
  echo "Results: $OUTPUT_DIR"
  exit "$failures"
fi
