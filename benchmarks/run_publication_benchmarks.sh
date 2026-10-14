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
  --autotune-trials N  Optuna trials per tuned benchmark (default: 20)
  --autotune-seconds N Target seconds per autotune trial (default: 10)
  -h, --help           Show this help

The suite runs Butina clustering, conformer RMSD, ETKDG and MMFF BFGS/FIRE (full
and single precision, with scanned HardwareOptions), the MCS atom/bond
comparison matrix plus ring support, GSI and DFS substructure search for every
supported SMARTS set, Morgan fingerprints, 3D descriptors (both precisions),
diversity pickers, AAP clustering, and TFD. Autotuning applies to MCS and
substructure search only.
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
AUTOTUNE_TRIALS=20
AUTOTUNE_TIME_BUDGET=10

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

# CPU policy. Two separate controls, deliberately not conflated:
#   * Timed work is held to $CPUS cores through each benchmark's own flags (--rdkit_threads,
#     --prep_threads, --autotune_cpu_budget). conformer_rmsd and tfd call nvMolKit code whose
#     OpenMP loops have no thread argument, so only those two get OMP_NUM_THREADS (see below).
#   * Untimed setup (conformer generation, validation, fingerprinting) is not capped: the
#     --prep_workers / --conf_gen_workers flags default to every available CPU.
# No global OMP_NUM_THREADS is exported, because that would also throttle untimed setup.

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

for module in nvtx numpy pandas rdkit torch tqdm; do
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

# Fixed single-GPU production workloads. Every workload uses three samples. BFGS and FIRE deliberately share
# the exact same input sample so their throughput and wall times are directly comparable.
#
# ETKDG and MMFF are not autotuned. Their HardwareOptions come from a grid scan on this machine
# (scan_hardware_options.py; GB300, 14-core budget, 2026-10-14): batch 256-16384 x batches per GPU 2-14,
# each point timed on the workloads below, which are sized to several complete fills of the largest
# configuration scanned. Optuna autotuning was replaced because its small fixed calibration sets could not
# fill the pipeline for large configurations and so were biased toward small batches.
# Format: batchSize:batchesPerGpu. Re-scan before running on different hardware.
CONFS_PER_MOL=200
BENCHMARK_SEED=42
TIMING_RUNS=3
ETKDG_NUM_MOLS=1311
FF_MMFF_NUM_MOLS=2622
declare -A HW_CONFIG=(
  [etkdg:full]="512:4"
  [etkdg:single]="1024:8"
  [mmff_bfgs:full]="512:6"
  [mmff_bfgs:single]="16384:6"
  [mmff_fire:full]="256:10"
  [mmff_fire:single]="512:12"
)
SUBSTRUCT_NUM_MOLS=1250000
MORGAN_NUM_MOLS=1000000
MCS_NUM_MOLS=10000
MCS_NUM_PAIRS=2800000
MCS_AUTOTUNE_PAIRS=230000
RDKIT_MAX_SECONDS=300
FF_MAX_ITERS=200

# MCS parameter rows: (benchmark name, atom compare, bond compare, ring-only).
# Strict means element/order matching, without the additional comparison
# variants exposed by mcs_bench.py.
MCS_ROWS=(
  "mcs_lax_lax:any:any:0"
  "mcs_lax_strict:any:order:0"
  "mcs_strict_lax:elements:any:0"
  "mcs_strict_strict:elements:order:0"
  "mcs_strict_strict_ring:elements:order:1"
)

DATASET_LINES="$(wc -l < "$DATASET")"
if [ "$AUTOTUNE_ENABLED" = "1" ] && [ "$DATASET_LINES" -lt "$SUBSTRUCT_NUM_MOLS" ]; then
  echo "Error: dataset has $DATASET_LINES lines; publication runs require at least" >&2
  echo "       $SUBSTRUCT_NUM_MOLS molecules for the fixed substructure workload." >&2
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
  echo "benchmark_seed: $BENCHMARK_SEED"
  echo "timing_runs: $TIMING_RUNS"
  echo "ff_mmff_num_mols: $FF_MMFF_NUM_MOLS"
  for key in "${!HW_CONFIG[@]}"; do echo "hw_config_${key/:/_}: ${HW_CONFIG[$key]}"; done
  echo "mcs_calibration_pairs: $MCS_AUTOTUNE_PAIRS"
  echo "etkdg_num_mols: $ETKDG_NUM_MOLS"
  echo "substruct_num_mols: $SUBSTRUCT_NUM_MOLS"
  echo "morgan_num_mols: $MORGAN_NUM_MOLS"
  echo "cpu_budget_not_enforced_for: aap_clustering (nvMolKit#368, hardware_concurrency threads)"
  echo "mcs_num_mols: $MCS_NUM_MOLS"
  echo "mcs_num_pairs: $MCS_NUM_PAIRS"
  echo "mcs_parameter_sets: ${MCS_ROWS[*]}"
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
ETKDG_MODE_FLAGS=()
FF_MODE_FLAGS=()
MCS_MODE_FLAGS=()
SUBSTRUCT_MODE_FLAGS=()
TFD_MODE_FLAGS=(--verify)
MORGAN_MODE_FLAGS=()
DESCRIPTORS3D_MODE_FLAGS=()
DIVERSITY_MODE_FLAGS=()
AAP_MODE_FLAGS=()
if [ "$SKIP_RDKIT" = "1" ]; then
  BUTINA_MODE_FLAGS=(--no-rdkit)
  CONFORMER_RMSD_MODE_FLAGS=(--no_rdkit --no_validate)
  ETKDG_MODE_FLAGS=(--no_rdkit --no_validate)
  FF_MODE_FLAGS=(--no_rdkit --no_validate)
  MCS_MODE_FLAGS=(--no_rdkit --no_validate)
  SUBSTRUCT_MODE_FLAGS=(--no_rdkit --no_validate)
  TFD_MODE_FLAGS=(--skip-rdkit)
  MORGAN_MODE_FLAGS=(--no-rdkit --no_validate)
  DESCRIPTORS3D_MODE_FLAGS=(--no-rdkit --no_validate)
  DIVERSITY_MODE_FLAGS=(--no-rdkit --no_validate)
  AAP_MODE_FLAGS=(--no-rdkit)
elif [ "$SKIP_NVMOLKIT" = "1" ]; then
  BUTINA_MODE_FLAGS=(--no-nvmolkit --no-fused)
  CONFORMER_RMSD_MODE_FLAGS=(--no_nvmolkit --no_validate)
  ETKDG_MODE_FLAGS=(--no_nvmolkit --no_validate)
  FF_MODE_FLAGS=(--no_nvmolkit --no_validate)
  MCS_MODE_FLAGS=(--no_nvmolkit --no_validate)
  SUBSTRUCT_MODE_FLAGS=(--no_nvmolkit --no_validate)
  TFD_MODE_FLAGS=(--skip-nvmolkit)
  MORGAN_MODE_FLAGS=(--no-nvmolkit --no_validate)
  DESCRIPTORS3D_MODE_FLAGS=(--no-nvmolkit --no_validate)
  DIVERSITY_MODE_FLAGS=(--no-nvmolkit --no_validate)
  AAP_MODE_FLAGS=(--no-nvmolkit --no-ligand-clustering-cpu)
fi

MCS_AUTOTUNE_FLAGS=()
if [ "$AUTOTUNE_ENABLED" = "1" ]; then
  MCS_AUTOTUNE_FLAGS=(
    --autotune
    --autotune_trials "$AUTOTUNE_TRIALS"
    --autotune_time_budget "$AUTOTUNE_TIME_BUDGET"
    --autotune_cpu_budget "$CPUS"
    --autotune_calibration_size "$MCS_AUTOTUNE_PAIRS"
  )
fi

substruct_calibration_size() {
  case "$1:$2" in
    gsi:rdkit_fragment_descriptors_supported) echo 1250000 ;;
    gsi:wehi_pains_supported) echo 475000 ;;
    gsi:BMS_2006_filter_supported) echo 1050000 ;;
    gsi:rdkit_tautomer_transforms_supported) echo 1250000 ;;
    gsi:rdkit_torsionPreferences_v2_supported) echo 258000 ;;
    dfs:rdkit_fragment_descriptors_supported) echo 1250000 ;;
    dfs:wehi_pains_supported) echo 1000000 ;;
    dfs:BMS_2006_filter_supported) echo 1250000 ;;
    dfs:rdkit_tautomer_transforms_supported) echo 1250000 ;;
    dfs:rdkit_torsionPreferences_v2_supported) echo 270000 ;;
  esac
}

substruct_timing_runs() {
  case "$1:$2" in
    gsi:rdkit_fragment_descriptors_supported) echo 3 ;;
    gsi:wehi_pains_supported) echo 3 ;;
    gsi:BMS_2006_filter_supported) echo 3 ;;
    gsi:rdkit_tautomer_transforms_supported) echo 3 ;;
    gsi:rdkit_torsionPreferences_v2_supported) echo 3 ;;
    dfs:rdkit_fragment_descriptors_supported) echo 3 ;;
    dfs:wehi_pains_supported) echo 3 ;;
    dfs:BMS_2006_filter_supported) echo 3 ;;
    dfs:rdkit_tautomer_transforms_supported) echo 3 ;;
    dfs:rdkit_torsionPreferences_v2_supported) echo 3 ;;
  esac
}

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
  --seed "$BENCHMARK_SEED" \
  --nvmolkit-reordering both \
  --output "$RESULT_DIR/butina_clustering.csv" \
  "${BUTINA_MODE_FLAGS[@]}"

run_bench conformer_rmsd "$RESULT_DIR/conformer_rmsd.csv" \
  env OMP_NUM_THREADS="$CPUS" python "$SCRIPT_DIR/conformer_rmsd_bench.py" \
  --smiles "$DATASET" \
  --num_mols 8000 \
  --seed "$BENCHMARK_SEED" \
  --confs_per_mol 10 25 50 100 200 \
  --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
  --output "$RESULT_DIR/conformer_rmsd.csv" \
  "${CONFORMER_RMSD_MODE_FLAGS[@]}"

# nvMolKit 0.7 adds single precision for ETKDG and the force-field minimizers. Both precisions run on
# identical inputs so their results are directly comparable, each with its own scanned HardwareOptions.
# Full precision keeps the historical benchmark names; single precision adds a _single suffix.
PRECISIONS=(full single)
precision_suffix() {
  if [ "$1" = "full" ]; then echo ""; else echo "_$1"; fi
}

for precision in "${PRECISIONS[@]}"; do
  etkdg_name="etkdg$(precision_suffix "$precision")"
  IFS=: read -r hw_batch hw_bpg <<< "${HW_CONFIG[etkdg:$precision]}"
  run_bench "$etkdg_name" "$RESULT_DIR/${etkdg_name}.csv" \
    python "$SCRIPT_DIR/etkdg_bench.py" \
    --smiles "$DATASET" \
    --num_mols "$ETKDG_NUM_MOLS" \
    --seed "$BENCHMARK_SEED" \
    --runs "$TIMING_RUNS" \
    --confs_per_mol "$CONFS_PER_MOL" \
    --num_gpus 1 \
    --batch_size "$hw_batch" \
    --batches_per_gpu "$hw_bpg" \
    --prep_threads "$CPUS" \
    --precision "$precision" \
    --rdkit_threads "$CPUS" \
    --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
    --output "$RESULT_DIR/${etkdg_name}.csv" \
    "${ETKDG_MODE_FLAGS[@]}"
done

# MMFF minimizers use one CPU thread per batch in flight (batchesPerGpu x GPUs) for both per-batch setup
# and GPU dispatch; HardwareOptions.preprocessingThreads is not used. The CPU budget is therefore the
# scanned batches_per_gpu, which is at most $CPUS. UFF is out of scope for this release's figures.
FF_ROWS=("mmff:BFGS" "mmff:FIRE")
for precision in "${PRECISIONS[@]}"; do
for row in "${FF_ROWS[@]}"; do
  ff="${row%%:*}"
  minimizer_kind="${row##*:}"
  minimizer_stem="${minimizer_kind,,}"
  bench_name="ff_optimize_${ff}_${minimizer_stem}$(precision_suffix "$precision")"
  IFS=: read -r hw_batch hw_bpg <<< "${HW_CONFIG[${ff}_${minimizer_stem}:$precision]}"
  if [ "$hw_bpg" -gt "$CPUS" ]; then
    echo "Error: ${bench_name} batches_per_gpu=$hw_bpg exceeds the CPU budget of $CPUS" >&2
    exit 2
  fi
  run_bench "$bench_name" "$RESULT_DIR/${bench_name}.csv" \
    python "$SCRIPT_DIR/ff_optimize_bench.py" \
    --smiles "$DATASET" \
    --num_mols "$FF_MMFF_NUM_MOLS" \
    --seed "$BENCHMARK_SEED" \
    --runs "$TIMING_RUNS" \
    --confs_per_mol "$CONFS_PER_MOL" \
    --ff "$ff" \
    --minimizer_kind "$minimizer_kind" \
    --max_iters "$FF_MAX_ITERS" \
    --no_validate \
    --num_gpus 1 \
    --batch_size "$hw_batch" \
    --batches_per_gpu "$hw_bpg" \
    --precision "$precision" \
    --rdkit_threads "$CPUS" \
    --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
    --output "$RESULT_DIR/${bench_name}.csv" \
    "${FF_MODE_FLAGS[@]}"
done
done

for row in "${MCS_ROWS[@]}"; do
  IFS=: read -r mcs_name atom_compare bond_compare ring_only <<< "$row"
  MCS_PARAMETER_FLAGS=(
    --atom_compare "$atom_compare"
    --bond_compare "$bond_compare"
  )
  if [ "$ring_only" = "1" ]; then
    MCS_PARAMETER_FLAGS+=(--ring_matches_ring_only)
  fi
  autotune_save_arg "$AUTOTUNE_DIR/${mcs_name}_config.json"
  run_bench "$mcs_name" "$RESULT_DIR/${mcs_name}.csv" \
    python "$SCRIPT_DIR/mcs_bench.py" \
    --smiles "$DATASET" \
    --num_mols "$MCS_NUM_MOLS" \
    --num_pairs "$MCS_NUM_PAIRS" \
    --seed "$BENCHMARK_SEED" \
    --runs "$TIMING_RUNS" \
    --num_gpus 1 \
    --prep_threads "$CPUS" \
    --rdkit_threads "$CPUS" \
    --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
    "${MCS_PARAMETER_FLAGS[@]}" \
    "${MCS_AUTOTUNE_FLAGS[@]}" \
    "${AUTOTUNE_SAVE_FLAGS[@]}" \
    --output "$RESULT_DIR/${mcs_name}.csv" \
    "${MCS_MODE_FLAGS[@]}"
done

SUBSTRUCT_ROWS=(
  "rdkit_fragment_descriptors_supported.txt:countSubstructMatches"
  "wehi_pains_supported.txt:hasSubstructMatch"
  "BMS_2006_filter_supported.txt:hasSubstructMatch"
  "rdkit_tautomer_transforms_supported.txt:getSubstructMatches"
  "rdkit_torsionPreferences_v2_supported.txt:getSubstructMatches"
)
for algorithm in gsi dfs; do
  for row in "${SUBSTRUCT_ROWS[@]}"; do
    smarts_file="${row%%:*}"
    mode="${row##*:}"
    smarts_stem="${smarts_file%.txt}"
    bench_name="substruct_${smarts_stem}_${algorithm}"
    cal_size="$(substruct_calibration_size "$algorithm" "$smarts_stem")"
    runs="$(substruct_timing_runs "$algorithm" "$smarts_stem")"
    SUBSTRUCT_AUTOTUNE_FLAGS=()
    if [ "$AUTOTUNE_ENABLED" = "1" ]; then
      SUBSTRUCT_AUTOTUNE_FLAGS=(
        --autotune
        --autotune_trials "$AUTOTUNE_TRIALS"
        --autotune_time_budget "$AUTOTUNE_TIME_BUDGET"
        --autotune_cpu_budget "$CPUS"
        --autotune_calibration_size "$cal_size"
      )
    fi
    autotune_save_arg "$AUTOTUNE_DIR/${bench_name}_config.json"
    run_bench "$bench_name" "$LOG_DIR/${bench_name}.log" \
      python "$SCRIPT_DIR/substruct_bench.py" \
      --smiles "$DATASET" \
      --num_mols "$SUBSTRUCT_NUM_MOLS" \
      --seed "$BENCHMARK_SEED" \
      --runs "$runs" \
      --sanitize \
      --smarts "$SMARTS_DIR/$smarts_file" \
      --mode "$mode" \
      --algorithm "$algorithm" \
      --num_gpus 1 \
      --prep_threads "$CPUS" \
      --rdkit_threads "$CPUS" \
      --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
      "${SUBSTRUCT_AUTOTUNE_FLAGS[@]}" \
      "${AUTOTUNE_SAVE_FLAGS[@]}" \
      "${SUBSTRUCT_MODE_FLAGS[@]}"
  done
done

run_bench morgan_fp "$RESULT_DIR/morgan_fp.csv" \
  python "$SCRIPT_DIR/morgan_fp_bench.py" \
  --smiles "$DATASET" \
  --num_mols "$MORGAN_NUM_MOLS" \
  --seed "$BENCHMARK_SEED" \
  --runs "$TIMING_RUNS" \
  --rdkit_threads "$CPUS" \
  --prep_threads "$CPUS" \
  --gpu_id 0 \
  --output "$RESULT_DIR/morgan_fp.csv" \
  "${MORGAN_MODE_FLAGS[@]}"

# 3D descriptors: --prep_threads is the CPU budget inside the timed call; embedding prep uses all cores.
for precision in "${PRECISIONS[@]}"; do
  descriptors_name="descriptors3d$(precision_suffix "$precision")"
  run_bench "$descriptors_name" "$RESULT_DIR/${descriptors_name}.csv" \
    python "$SCRIPT_DIR/descriptors3d_bench.py" \
    --smiles "$DATASET" \
    --seed "$BENCHMARK_SEED" \
    --precision "$precision" \
    --prep_threads "$CPUS" \
    --device_input \
    --output "$RESULT_DIR/${descriptors_name}.csv" \
    "${DESCRIPTORS3D_MODE_FLAGS[@]}"
done

run_bench diversity_pickers "$RESULT_DIR/diversity_pickers.csv" \
  python "$SCRIPT_DIR/diversity_pickers_bench.py" \
  --smiles "$DATASET" \
  --seed "$BENCHMARK_SEED" \
  --output "$RESULT_DIR/diversity_pickers.csv" \
  "${DIVERSITY_MODE_FLAGS[@]}"

# KNOWN CAVEAT: nvMolKit 0.7.0 builds AAP host descriptors with std::thread::hardware_concurrency() threads and
# exposes no way to limit them (NVIDIA-BioNeMo/nvMolKit#368), so nvMolKit's side of this benchmark is NOT held to
# the $CPUS budget. It is deliberately left uncapped rather than pinned; do not compare it as a $CPUS-core result.
run_bench aap_clustering "$RESULT_DIR/aap_clustering.csv" \
  python "$SCRIPT_DIR/aap_clustering_bench.py" \
  --smiles "$DATASET" \
  --seed "$BENCHMARK_SEED" \
  --operation all \
  --output "$RESULT_DIR/aap_clustering.csv" \
  "${AAP_MODE_FLAGS[@]}"

run_bench tfd "$RESULT_DIR/tfd.csv" \
  env OMP_NUM_THREADS="$CPUS" python "$SCRIPT_DIR/tfd_bench.py" \
  --smiles-file "$DATASET" \
  --seed "$BENCHMARK_SEED" \
  --num-mols 100 1000 5000 \
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
