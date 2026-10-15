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
  --no-nvmolkit        Run RDKit only
  --aap-csv FILE       Scored CSV (columns smiles, QED) for AAP clustering (default:
                       ~/data/enamine_aap_100k_qed.csv; AAP is skipped if missing)
  --hw-configs FILE    Scanned nvMolKit execution settings (default:
                       hardware_configs/<gpu>_cpu<budget>.json; see select_scan_configs.py)
  --only GLOBS         Comma-separated benchmark-name globs to run (e.g. 'mcs_*,substruct_*');
                       other benchmarks are skipped and left out of summary.tsv
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
ONLY_GLOBS=""
HW_CONFIG_FILE=""
AAP_CSV="$HOME/data/enamine_aap_100k_qed.csv"

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
    --aap-csv)
      [ $# -ge 2 ] || { echo "Error: --aap-csv requires a value" >&2; exit 2; }
      AAP_CSV="$2"
      shift 2
      ;;
    --hw-configs)
      [ $# -ge 2 ] || { echo "Error: --hw-configs requires a value" >&2; exit 2; }
      HW_CONFIG_FILE="$2"
      shift 2
      ;;
    --only)
      [ $# -ge 2 ] || { echo "Error: --only requires a value" >&2; exit 2; }
      ONLY_GLOBS="$2"
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
#     --prep_threads, --workers, scanned HardwareOptions). conformer_rmsd and tfd call nvMolKit code whose
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
SYSINFO_DIR="$OUTPUT_DIR/sysinfo"
if ! mkdir -p "$LOG_DIR" "$RESULT_DIR" "$SYSINFO_DIR"; then
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
# Settings are loaded from a JSON file written by select_scan_configs.py, keyed by GPU and CPU budget so
# that configurations scanned on one machine are never applied to different hardware.
# Formats: HW_CONFIG batchSize:batchesPerGpu; MCS_CONFIG batchSize:workers:preprocessingThreads:executors;
# SUBSTRUCT_CONFIG batchSize:workers:preprocessingThreads.
declare -A HW_CONFIG=()
validate_concurrent_budget() {
  local name="$1" workers="$2" prep="$3"
  if [ $(( workers + prep )) -gt "$CPUS" ]; then
    echo "Error: $name uses workers=$workers + preprocessing=$prep > CPU budget $CPUS" >&2
    exit 2
  fi
}
# MCS and substructure search run GPU-worker and preprocessing threads concurrently, so the scans
# (scan_search_options.py) hold workers + preprocessing threads to the CPU budget.
# MCS format: batchSize:workers:preprocessingThreads:executorsPerRunner.
declare -A MCS_CONFIG=()
# Substructure format: batchSize:workers:preprocessingThreads, keyed by SMARTS stem and algorithm.
declare -A SUBSTRUCT_CONFIG=()

if [ -z "$HW_CONFIG_FILE" ] && [ "$SKIP_NVMOLKIT" = "0" ]; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_ID" 2>/dev/null | head -1)"
  GPU_SLUG="$(python -c 'import re, sys; print(re.sub(r"[^a-z0-9]+", "_", sys.argv[1].lower().replace("nvidia", "")).strip("_"))' "$GPU_NAME")"
  HW_CONFIG_FILE="$SCRIPT_DIR/hardware_configs/${GPU_SLUG}_cpu${CPUS}.json"
fi
if [ -n "$HW_CONFIG_FILE" ]; then
  if [ ! -f "$HW_CONFIG_FILE" ]; then
    echo "Error: no scanned settings at $HW_CONFIG_FILE" >&2
    echo "       Run scan_hardware_options.py and scan_search_options.py on this machine with this CPU" >&2
    echo "       budget, then select_scan_configs.py, or pass --hw-configs." >&2
    exit 1
  fi
  if ! HW_CONFIG_ASSIGNMENTS="$(python - "$HW_CONFIG_FILE" <<'PY'
import json
import shlex
import sys

document = json.load(open(sys.argv[1]))
for key, entry in document["configs"].items():
    kind, _, name = key.partition(":")
    c = entry["config"]
    if kind == "mcs":
        value = f"{c['batchSize']}:{c['workers']}:{c['preprocessingThreads']}:{c['executors']}"
        print(f"MCS_CONFIG[{shlex.quote(name)}]={value}")
    elif kind == "substruct":
        value = f"{c['batchSize']}:{c['workers']}:{c['preprocessingThreads']}"
        print(f"SUBSTRUCT_CONFIG[{shlex.quote(name)}]={value}")
    else:
        print(f"HW_CONFIG[{shlex.quote(key)}]={c['batchSize']}:{c['batchesPerGpu']}")
print(f"HW_CONFIG_CPU_BUDGET={int(document['cpu_budget'])}")
print(f"HW_CONFIG_GPU={shlex.quote(document['gpu'])}")
PY
)"; then
    echo "Error: could not read $HW_CONFIG_FILE" >&2
    exit 1
  fi
  eval "$HW_CONFIG_ASSIGNMENTS"
  if [ "$HW_CONFIG_CPU_BUDGET" != "$CPUS" ]; then
    echo "Error: $HW_CONFIG_FILE was scanned with a $HW_CONFIG_CPU_BUDGET-core budget, not $CPUS" >&2
    exit 2
  fi
  if [ "$SKIP_NVMOLKIT" = "0" ] && [ -n "${GPU_NAME:-}" ] && [ "$HW_CONFIG_GPU" != "$GPU_NAME" ]; then
    echo "Error: $HW_CONFIG_FILE was scanned on '$HW_CONFIG_GPU', but GPU $GPU_ID is '$GPU_NAME'" >&2
    exit 2
  fi
fi

# Look up a scanned setting into HW_VALUE. RDKit-only runs ignore nvMolKit settings, so they fall back
# to library defaults; nvMolKit runs fail rather than benchmark an unscanned configuration.
lookup_config() {
  local -n table="$1"
  local key="$2" default="$3"
  if [ -n "${table[$key]:-}" ]; then
    HW_VALUE="${table[$key]}"
  elif [ "$SKIP_NVMOLKIT" = "1" ]; then
    HW_VALUE="$default"
  else
    echo "Error: no scanned $1 entry for '$key' in $HW_CONFIG_FILE" >&2
    exit 2
  fi
}
SUBSTRUCT_NUM_MOLS=5000000
MORGAN_NUM_MOLS=1000000
# Clustering baselines have no time cap; their RDKit sizes are capped instead. Butina's RDKit backend is
# single-threaded O(n^2); diversity's LeaderPicker uses the CPU budget (MaxMinPicker is serial).
BUTINA_RDKIT_MAX_SIZE=20000
DIVERSITY_RDKIT_MAX_SIZE=100000
# AAP similarity pairs: large for nvMolKit, small when the (pure-Python) RDKit reference runs.
# (The RDKit Contrib reference runs ~35 pairs/s single-threaded, so 2k pairs keeps it near a minute per run.)
# nvMolKit pair similarity is host-bound per pair, so 10k pairs (not millions) keeps it to seconds.
if [ "$SKIP_RDKIT" = "1" ]; then AAP_NUM_PAIRS=10000; else AAP_NUM_PAIRS=2000; fi
MCS_NUM_MOLS=10000
MCS_NUM_PAIRS=2800000
RDKIT_MAX_SECONDS=300
FF_MAX_ITERS=200

# MCS parameter rows: (benchmark name, atom compare, bond compare, ring-only).
# Strict means element/order matching, without the additional comparison
# variants exposed by mcs_bench.py.
# The fifth field is the timed pair count. Lax comparisons are ~10x slower per pair on one GPU (about
# 4-6k pairs/s on GB300 versus 45-100k for strict rows), so they time fewer pairs to keep each run near a
# minute. Lax rows use the same 200k-pair sample their settings were scanned on: their throughput depends
# strongly on which pairs are drawn (a 300k sample measured 33% slower than the 200k scan).
MCS_ROWS=(
  "mcs_lax_lax:any:any:0:200000"
  "mcs_lax_strict:any:order:0:200000"
  "mcs_strict_lax:elements:any:0:$MCS_NUM_PAIRS"
  "mcs_strict_strict:elements:order:0:$MCS_NUM_PAIRS"
  "mcs_strict_strict_ring:elements:order:1:$MCS_NUM_PAIRS"
)

DATASET_LINES="$(wc -l < "$DATASET")"
if [ "$DATASET_LINES" -lt "$SUBSTRUCT_NUM_MOLS" ]; then
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
  echo "benchmark_seed: $BENCHMARK_SEED"
  echo "timing_runs: $TIMING_RUNS"
  echo "ff_mmff_num_mols: $FF_MMFF_NUM_MOLS"
  for key in "${!HW_CONFIG[@]}"; do echo "hw_config_${key/:/_}: ${HW_CONFIG[$key]}"; done
  for key in "${!MCS_CONFIG[@]}"; do echo "mcs_config_${key}: ${MCS_CONFIG[$key]}"; done
  for key in "${!SUBSTRUCT_CONFIG[@]}"; do echo "substruct_config_${key/:/_}: ${SUBSTRUCT_CONFIG[$key]}"; done
  echo "etkdg_num_mols: $ETKDG_NUM_MOLS"
  echo "substruct_num_mols: $SUBSTRUCT_NUM_MOLS"
  echo "morgan_num_mols: $MORGAN_NUM_MOLS"
  echo "cpu_budget_not_enforced_for: aap_clustering (nvMolKit#368, hardware_concurrency threads)"
  echo "mcs_num_mols: $MCS_NUM_MOLS"
  echo "mcs_num_pairs: $MCS_NUM_PAIRS"
  echo "mcs_parameter_sets: ${MCS_ROWS[*]}"
  echo "only: ${ONLY_GLOBS:-all}"
  echo "hw_config_file: ${HW_CONFIG_FILE:-none}"
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

selected() {
  local name="$1" glob
  local -a globs
  [ -z "$ONLY_GLOBS" ] && return 0
  # Split without pathname expansion: an unquoted 'mcs_*' would otherwise match files such as mcs_bench.py.
  IFS=, read -ra globs <<< "$ONLY_GLOBS"
  for glob in "${globs[@]}"; do
    # shellcheck disable=SC2053
    [[ "$name" == $glob ]] && return 0
  done
  return 1
}

run_bench() {
  local name="$1"
  local result_path="$2"
  shift 2
  if ! selected "$name"; then
    echo "[$name] skipped (--only $ONLY_GLOBS)"
    return 0
  fi
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
  AAP_MODE_FLAGS=(--no-nvmolkit)
fi

run_bench butina_clustering "$RESULT_DIR/butina_clustering.csv" \
  python "$SCRIPT_DIR/butina_clustering_bench.py" "$DATASET" \
  --seed "$BENCHMARK_SEED" \
  --nvmolkit-reordering both \
  --rdkit-max-size "$BUTINA_RDKIT_MAX_SIZE" \
  --output "$RESULT_DIR/butina_clustering.csv" \
  "${BUTINA_MODE_FLAGS[@]}"

run_bench conformer_rmsd "$RESULT_DIR/conformer_rmsd.csv" \
  env OMP_NUM_THREADS="$CPUS" python "$SCRIPT_DIR/conformer_rmsd_bench.py" \
  --smiles "$DATASET" \
  --num_mols 8000 \
  --seed "$BENCHMARK_SEED" \
  --confs_per_mol 10 25 50 100 200 \
  --gpu_runs 20 \
  --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
  --output "$RESULT_DIR/conformer_rmsd.csv" \
  "${CONFORMER_RMSD_MODE_FLAGS[@]}"

# nvMolKit 0.7 adds single precision for ETKDG and the force-field minimizers. Both precisions run on
# identical inputs so their results are directly comparable, each with its own scanned HardwareOptions.
# Full precision keeps the historical benchmark names; single precision adds a _single suffix.
PRECISIONS=(full single)
# RDKit has no precision mode, so an RDKit-only run measures each precision-dependent baseline once.
if [ "$SKIP_NVMOLKIT" = "1" ]; then PRECISIONS=(full); fi
precision_suffix() {
  if [ "$1" = "full" ]; then echo ""; else echo "_$1"; fi
}

for precision in "${PRECISIONS[@]}"; do
  etkdg_name="etkdg$(precision_suffix "$precision")"
  selected "$etkdg_name" || { echo "[$etkdg_name] skipped (--only $ONLY_GLOBS)"; continue; }
  lookup_config HW_CONFIG "etkdg:$precision" "-1:-1"
  IFS=: read -r hw_batch hw_bpg <<< "$HW_VALUE"
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
# RDKit only has BFGS, so an RDKit-only run measures the MMFF baseline once.
if [ "$SKIP_NVMOLKIT" = "1" ]; then FF_ROWS=("mmff:BFGS"); fi
for precision in "${PRECISIONS[@]}"; do
for row in "${FF_ROWS[@]}"; do
  ff="${row%%:*}"
  minimizer_kind="${row##*:}"
  minimizer_stem="${minimizer_kind,,}"
  bench_name="ff_optimize_${ff}_${minimizer_stem}$(precision_suffix "$precision")"
  selected "$bench_name" || { echo "[$bench_name] skipped (--only $ONLY_GLOBS)"; continue; }
  lookup_config HW_CONFIG "${ff}_${minimizer_stem}:$precision" "-1:-1"
  IFS=: read -r hw_batch hw_bpg <<< "$HW_VALUE"
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
  IFS=: read -r mcs_name atom_compare bond_compare ring_only mcs_pairs <<< "$row"
  MCS_PARAMETER_FLAGS=(
    --atom_compare "$atom_compare"
    --bond_compare "$bond_compare"
  )
  if [ "$ring_only" = "1" ]; then
    MCS_PARAMETER_FLAGS+=(--ring_matches_ring_only)
  fi
  selected "$mcs_name" || { echo "[$mcs_name] skipped (--only $ONLY_GLOBS)"; continue; }
  lookup_config MCS_CONFIG "$mcs_name" "0:-1:-1:-1"
  IFS=: read -r hw_batch hw_workers hw_prep hw_exec <<< "$HW_VALUE"
  validate_concurrent_budget "$mcs_name" "$hw_workers" "$hw_prep"
  run_bench "$mcs_name" "$RESULT_DIR/${mcs_name}.csv" \
    python "$SCRIPT_DIR/mcs_bench.py" \
    --smiles "$DATASET" \
    --num_mols "$MCS_NUM_MOLS" \
    --num_pairs "$mcs_pairs" \
    --seed "$BENCHMARK_SEED" \
    --runs "$TIMING_RUNS" \
    --num_gpus 1 \
    --batch_size "$hw_batch" \
    --workers "$hw_workers" \
    --prep_threads "$hw_prep" \
    --executors_per_runner "$hw_exec" \
    --rdkit_threads "$CPUS" \
    --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
    "${MCS_PARAMETER_FLAGS[@]}" \
    --output "$RESULT_DIR/${mcs_name}.csv" \
    "${MCS_MODE_FLAGS[@]}"
done

# SMARTS file : search mode : timed molecules. Workloads are sized for >=~10 s per timed run at the scanned
# GB300 throughput; the torsion set (~365 patterns, all matches) uses fewer molecules to stay near 30 s.
SUBSTRUCT_ROWS=(
  "rdkit_fragment_descriptors_supported.txt:countSubstructMatches:$SUBSTRUCT_NUM_MOLS"
  "wehi_pains_supported.txt:hasSubstructMatch:$SUBSTRUCT_NUM_MOLS"
  "BMS_2006_filter_supported.txt:hasSubstructMatch:$SUBSTRUCT_NUM_MOLS"
  "rdkit_tautomer_transforms_supported.txt:getSubstructMatches:$SUBSTRUCT_NUM_MOLS"
  "rdkit_torsionPreferences_v2_supported.txt:getSubstructMatches:1250000"
)
for algorithm in gsi dfs; do
  for row in "${SUBSTRUCT_ROWS[@]}"; do
    IFS=: read -r smarts_file mode substruct_mols <<< "$row"
    smarts_stem="${smarts_file%.txt}"
    bench_name="substruct_${smarts_stem}_${algorithm}"
    selected "$bench_name" || { echo "[$bench_name] skipped (--only $ONLY_GLOBS)"; continue; }
    lookup_config SUBSTRUCT_CONFIG "$smarts_stem:$algorithm" "1024:-1:-1"
    IFS=: read -r hw_batch hw_workers hw_prep <<< "$HW_VALUE"
    validate_concurrent_budget "$bench_name" "$hw_workers" "$hw_prep"
    run_bench "$bench_name" "$RESULT_DIR/${bench_name}.csv" \
      python "$SCRIPT_DIR/substruct_bench.py" \
      --smiles "$DATASET" \
      --num_mols "$substruct_mols" \
      --seed "$BENCHMARK_SEED" \
      --runs "$TIMING_RUNS" \
      --sanitize \
      --smarts "$SMARTS_DIR/$smarts_file" \
      --mode "$mode" \
      --algorithm "$algorithm" \
      --num_gpus 1 \
      --batch_size "$hw_batch" \
      --workers "$hw_workers" \
      --prep_threads "$hw_prep" \
      --rdkit_threads "$CPUS" \
      --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
      --output "$RESULT_DIR/${bench_name}.csv" \
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
  --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
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
    --rdkit_threads "$CPUS" \
    --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
    --num_mols 1000 10000 100000 \
    --confs_per_mol 10 50 \
    --device_input \
    --output "$RESULT_DIR/${descriptors_name}.csv" \
    "${DESCRIPTORS3D_MODE_FLAGS[@]}"
done

run_bench diversity_pickers "$RESULT_DIR/diversity_pickers.csv" \
  python "$SCRIPT_DIR/diversity_pickers_bench.py" \
  --smiles "$DATASET" \
  --seed "$BENCHMARK_SEED" \
  --sizes 10000 100000 1000000 3000000 \
  --pick_sizes 100 1000 10000 \
  --rdkit_threads "$CPUS" \
  --rdkit_max_size "$DIVERSITY_RDKIT_MAX_SIZE" \
  --output "$RESULT_DIR/diversity_pickers.csv" \
  "${DIVERSITY_MODE_FLAGS[@]}"

# KNOWN CAVEAT: nvMolKit 0.7.0 builds AAP host descriptors with std::thread::hardware_concurrency() threads and
# exposes no way to limit them (NVIDIA-BioNeMo/nvMolKit#368), so nvMolKit's side of this benchmark is NOT held to
# the $CPUS budget. It is deliberately left uncapped rather than pinned; do not compare it as a $CPUS-core result.
# AAP clustering orders molecules by a numeric priority (DISE requires one). Enamine has no assay values,
# so a 100k random Enamine sample ordered by descending QED stands in; it is not comparable with the
# assay-based (PubChem AID 686979) AAP figures from the v0.7 pre-release analysis.
if [ -f "$AAP_CSV" ]; then
  run_bench aap_clustering "$RESULT_DIR/aap_clustering.csv" \
    python "$SCRIPT_DIR/aap_clustering_bench.py" \
    --csv "$AAP_CSV" \
    --smiles-column smiles \
    --sort-tag QED \
    --sort-descending \
    --sizes 1024 4096 16384 65536 \
    --num_pairs "$AAP_NUM_PAIRS" \
    --seed "$BENCHMARK_SEED" \
    --operation all \
    --no-ligand-clustering-cpu \
    --output "$RESULT_DIR/aap_clustering.csv" \
    "${AAP_MODE_FLAGS[@]}"
else
  echo "[aap_clustering] skipped: no AAP CSV at $AAP_CSV (pass --aap-csv)"
fi

run_bench tfd "$RESULT_DIR/tfd.csv" \
  env OMP_NUM_THREADS="$CPUS" python "$SCRIPT_DIR/tfd_bench.py" \
  --smiles-file "$DATASET" \
  --seed "$BENCHMARK_SEED" \
  --num-mols 100 1000 5000 \
  --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
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
