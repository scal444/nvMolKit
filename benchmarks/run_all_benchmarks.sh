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
# system metadata into a single output directory. Runs use one GPU by default;
# multi-GPU benchmarking is opt-in with --num-gpus.
#
# Both modes run the full non-size-scan bench list. Molecule-size sweeps live
# in run_size_scan_benchmarks.sh. The benches that don't expose a
# --num_gpus flag (butina_clustering, conformer_rmsd, cross_similarity, tfd)
# always run on a single GPU. --num-gpus only affects the multi-GPU-aware
# benches:
#
#   - butina_clustering_bench.py    (single-GPU library; --num_gpus N/A)
#   - conformer_rmsd_bench.py       (single-GPU library; --num_gpus N/A)
#   - cross_similarity_bench.py     (single-GPU library; --num_gpus N/A)
#   - etkdg_bench.py                (autotuned, --num_gpus = mode arg)
#   - ff_optimize_bench.py          (autotuned, MMFF and UFF, --num_gpus = mode arg)
#   - mcs_bench.py                  (autotuned, --num_gpus = mode arg)
#   - substruct_bench.py            (autotuned, one row per SMARTS, --num_gpus = mode arg)
#   - tfd_bench.py                  (single-GPU library; --num_gpus N/A)
#
# Autotune defaults to 20 trials × 60 s/trial per autotuned invocation.
# Tuned HardwareOptions / MCSConfig / SubstructSearchConfig JSON files are saved under
# $OUTPUT_DIR/autotune for reproducibility.
#
# Data inputs come from --data-dir (default /data, the in-container path; on
# the host it is typically ~/data). Required layout:
#
#   $DATA_DIR/enamine_real_10M.cxsmiles
#
# Usage:
#   ./run_all_benchmarks.sh --output-dir DIR [--num-gpus N] [--gpu-ids LIST]
#                           [--data-dir DIR] [--include NAME [NAME ...]]
#   ./run_all_benchmarks.sh <N> --output-dir DIR ...  # legacy positional form
#   ./run_all_benchmarks.sh --list
#
# --include restricts the run to the named benches (whitelist). Unknown names
# are an error. With no --include all benches run. --list prints the bench
# names and exits.

set -uo pipefail

usage() {
  cat >&2 <<EOF
Usage: $0 --output-dir DIR [--num-gpus N] [--gpu-ids LIST] [--data-dir DIR]
          [--include NAME [NAME ...]] [--no-rdkit | --no-nvmolkit]
          [--autotune-trials N] [--autotune-seconds N]
       $0 <N> --output-dir DIR ...  (legacy positional GPU count)
       $0 --list

  --num-gpus N      GPUs for multi-GPU-aware benchmarks (default: 1)
  --gpu-ids LIST    Comma-separated physical GPU IDs to expose. The number of
                    IDs must match --num-gpus. By default, preserve the
                    caller's CUDA_VISIBLE_DEVICES (if any).
  --output-dir DIR  Results directory (required for runs)
  --data-dir DIR    Input data directory (default: /data)
  --include NAME... Whitelist of bench names to run (run --list to see them)
  --autotune-trials N  Optuna trials per tuned benchmark (default: 20)
  --autotune-seconds N Target seconds per autotune trial (default: 10)
  --no-rdkit        Skip RDKit head-to-head on every bench. When NOT set,
                    validation/verification is enabled where the bench supports
                    it (etkdg, ff_optimize_*, substruct, tfd, conformer_rmsd).
  --no-nvmolkit     RDKit-only mode: skip the nvMolKit side on every bench.
                    Autotune is dropped (it requires nvMolKit) and the
                    GPU-only benches still walk through their RDKit reference
                    implementations. Mutually exclusive with --no-rdkit.
  --continue        Resume from a previous run in --output-dir: skip every
                    bench whose row in summary.tsv has status=ok (rows with
                    status=fail or status=skipped are re-run).
  --list            Print the bench names and exit
EOF
}

NUM_GPUS=1
GPU_IDS=""
LIST_ONLY=0

# Preserve the original positional GPU-count interface, but no longer require
# it. This makes the common single-GPU invocation the shortest one.
if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
  NUM_GPUS="$1"
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SMARTS_DIR="$REPO_ROOT/tests/test_data/SMARTS"

# Substruct bench config rows: (smarts_file, mode). Substruct autotune is
# incompatible with --config (single config per run only), so we drive each
# pattern set individually below.
SUBSTRUCT_ROWS=(
  "rdkit_fragment_descriptors_supported.txt:countSubstructMatches"
  "wehi_pains_supported.txt:hasSubstructMatch"
  "BMS_2006_filter_supported.txt:hasSubstructMatch"
  "rdkit_tautomer_transforms_supported.txt:getSubstructMatches"
  "rdkit_torsionPreferences_v2_supported.txt:getSubstructMatches"
)

# MCS parameter rows: (benchmark name, atom compare, bond compare, ring-only).
# "strict" deliberately means element/order matching; the isotope, exact-order,
# valence, and formal-charge variants are outside this benchmark matrix.
MCS_ROWS=(
  "mcs_lax_lax:any:any:0"
  "mcs_lax_strict:any:order:0"
  "mcs_strict_lax:elements:any:0"
  "mcs_strict_strict:elements:order:0"
  "mcs_strict_strict_ring:elements:order:1"
)

# Enumerate every bench name this script can run, in the order they execute.
ALL_BENCH_NAMES=(
  "butina_clustering"
  "conformer_rmsd"
  "cross_similarity"
  "etkdg"
  "ff_optimize_mmff_bfgs"
  "ff_optimize_mmff_fire"
  "ff_optimize_uff_bfgs"
  "ff_optimize_uff_fire"
)
for row in "${MCS_ROWS[@]}"; do
  ALL_BENCH_NAMES+=("${row%%:*}")
done
for row in "${SUBSTRUCT_ROWS[@]}"; do
  smarts_file="${row%%:*}"
  for algorithm in gsi dfs; do
    ALL_BENCH_NAMES+=("substruct_${smarts_file%.txt}_${algorithm}")
  done
done
ALL_BENCH_NAMES+=("tfd")

OUTPUT_DIR=""
DATA_DIR="/data"
INCLUDE_LIST=()
SKIP_RDKIT=0
SKIP_NVMOLKIT=0
CONTINUE=0
AUTOTUNE_TRIALS=20
AUTOTUNE_TIME_BUDGET=10

while [ $# -gt 0 ]; do
  case "$1" in
    --num-gpus)
      if [ $# -lt 2 ]; then
        echo "Error: --num-gpus requires a value" >&2
        exit 2
      fi
      NUM_GPUS="$2"
      shift 2
      ;;
    --gpu-ids)
      if [ $# -lt 2 ]; then
        echo "Error: --gpu-ids requires a value" >&2
        exit 2
      fi
      GPU_IDS="$2"
      shift 2
      ;;
    --output-dir)
      if [ $# -lt 2 ]; then
        echo "Error: --output-dir requires a value" >&2
        exit 2
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --data-dir)
      if [ $# -lt 2 ]; then
        echo "Error: --data-dir requires a value" >&2
        exit 2
      fi
      DATA_DIR="$2"
      shift 2
      ;;
    --include)
      shift
      if [ $# -eq 0 ] || [[ "$1" == --* ]]; then
        echo "Error: --include requires at least one bench name" >&2
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
    --autotune-trials)
      if [ $# -lt 2 ]; then
        echo "Error: --autotune-trials requires a value" >&2
        exit 2
      fi
      AUTOTUNE_TRIALS="$2"
      shift 2
      ;;
    --autotune-seconds)
      if [ $# -lt 2 ]; then
        echo "Error: --autotune-seconds requires a value" >&2
        exit 2
      fi
      AUTOTUNE_TIME_BUDGET="$2"
      shift 2
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
  echo "Error: --num-gpus must be a positive integer (got: $NUM_GPUS)" >&2
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

# Validate --include against the known bench names.
if [ "${#INCLUDE_LIST[@]}" -gt 0 ]; then
  for requested in "${INCLUDE_LIST[@]}"; do
    found=0
    for known in "${ALL_BENCH_NAMES[@]}"; do
      if [ "$requested" = "$known" ]; then
        found=1
        break
      fi
    done
    if [ "$found" = "0" ]; then
      echo "Error: --include name '$requested' is not a known bench" >&2
      echo "Known names:" >&2
      printf '  %s\n' "${ALL_BENCH_NAMES[@]}" >&2
      exit 2
    fi
  done
fi

# Returns 0 (run) if a bench should execute under the current --include filter.
should_run() {
  local name="$1"
  if [ "${#INCLUDE_LIST[@]}" -eq 0 ]; then
    return 0
  fi
  local entry
  for entry in "${INCLUDE_LIST[@]}"; do
    if [ "$entry" = "$name" ]; then
      return 0
    fi
  done
  return 1
}

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
RESULT_DIR="$OUTPUT_DIR/results"
AUTOTUNE_DIR="$OUTPUT_DIR/autotune"
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$AUTOTUNE_DIR"

if [ ! -d "$DATA_DIR" ]; then
  echo "Error: --data-dir '$DATA_DIR' does not exist" >&2
  exit 1
fi

ENAMINE_CXSMILES="$DATA_DIR/enamine_real_10M.cxsmiles"
if [ ! -f "$ENAMINE_CXSMILES" ]; then
  echo "Missing $ENAMINE_CXSMILES" >&2
  exit 1
fi

# Verify enough CUDA devices are visible to the benchmark process before
# spending time tuning. Unlike nvidia-smi, torch honors CUDA_VISIBLE_DEVICES.
# RDKit-only runs intentionally do not require a GPU.
GPU_COUNT="not_checked"
if [ "$SKIP_NVMOLKIT" = "0" ]; then
  if ! GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null)"; then
    echo "Error: could not query CUDA device count through torch" >&2
    exit 1
  fi
  if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
    echo "Error: requested $NUM_GPUS GPUs but torch reports $GPU_COUNT visible" >&2
    echo "       CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
    exit 1
  fi
fi

# Ensure runtime dependencies the bench scripts assume but the conda image may
# not ship. Dependency packaging is intentionally outside this script's scope.
ensure_pip_pkg() {
  local import_name="$1"
  local pip_name="${2:-$1}"
  if python -c "import $import_name" 2>/dev/null; then
    return 0
  fi
  echo "[deps] $import_name not importable; installing $pip_name"
  pip install --quiet "$pip_name" || {
    echo "[deps] failed to install $pip_name" >&2
    return 1
  }
}
ensure_pip_pkg pyperf
ensure_pip_pkg optuna
ensure_pip_pkg nvtx
ensure_pip_pkg pandas

# RDKit's ETKDG/FF parallelize across conformers within a single mol, so
# numThreads is capped in practice by confs_per_mol. confs_per_mol >= 128 keeps
# the RDKit comparison fully parallel on the >=128-core nodes we target.
# nvmolkit parallelizes across all conformers in the batch independent of
# this. The calibration formula below divides by confs_per_mol, so mol counts
# scale inversely as this knob changes.
ETKDG_CONFS_PER_MOL=200
FF_CONFS_PER_MOL=200
FF_MAX_ITERS=200

# Calibration size for autotune trials. The biggest searchable config is
# (batchSize_max * batchesPerGpu_max * num_gpus) conformers in flight; we want
# at least 2 full pipeline fills so even the largest config measures steady
# state instead of tail/transient effects. Solve for mols:
#   calibration_mols = 2 * batchSize_max * batchesPerGpu_max * num_gpus / confs_per_mol
# Mirrors the separate ETKDG and FF autotune search spaces.
ETKDG_AUTOTUNE_BS_MAX=1024
FF_AUTOTUNE_BS_MAX=4096
AUTOTUNE_BPG_MAX=8
ETKDG_CAL_SIZE=$(( 2 * ETKDG_AUTOTUNE_BS_MAX * AUTOTUNE_BPG_MAX * NUM_GPUS / ETKDG_CONFS_PER_MOL ))

# Fixed production workloads, measured on the 8-GPU H200/B200 comprehensive
# runs. Every timed API uses three samples for comparable median/std data.
# Counts are deliberately literal so changes to tuning bounds cannot silently
# resize the production measurement.
BENCHMARK_SEED=42
TIMING_RUNS=3
ETKDG_NUM_MOLS=8000
ETKDG_CAL_SIZE=1000
FF_MMFF_BFGS_NUM_MOLS=5000
FF_MMFF_BFGS_CAL_SIZE=2750
FF_MMFF_FIRE_NUM_MOLS=5000
FF_MMFF_FIRE_CAL_SIZE=5000
FF_UFF_BFGS_NUM_MOLS=5000
FF_UFF_BFGS_CAL_SIZE=950
FF_UFF_FIRE_NUM_MOLS=5000
FF_UFF_FIRE_CAL_SIZE=1800
MCS_AUTOTUNE_PAIRS=1200000
MCS_NUM_PAIRS=14000000
MCS_NUM_MOLS=10000

# Butina needs >=40k molecules for its rdkit_lowmem variant. Cap at 60k so
# fingerprint construction stays bounded.
BUTINA_NUM_MOLS=60000

# Substructure uses the complete 10M target set. Per-SMARTS calibration sizes
# and repetition counts are literal values in the functions below.
SUBSTRUCT_NUM_MOLS=3000000

# RDKit thread count for the head-to-head comparison on the multi-GPU benches.
# A single-GPU run caps RDKit at 16 physical cores; multi-GPU runs use all
# physical cores.
#
# Single-GPU benches (butina, conformer_rmsd, cross_similarity, tfd) compare
# against single-threaded RDKit and ignore this variable.
PHYSICAL_CORES="$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
if [ -z "$PHYSICAL_CORES" ] || [ "$PHYSICAL_CORES" -lt 1 ]; then
  PHYSICAL_CORES=1
fi
if [ "$NUM_GPUS" -eq 1 ] && [ "$PHYSICAL_CORES" -gt 16 ]; then
  RDKIT_THREADS=16
else
  RDKIT_THREADS="$PHYSICAL_CORES"
fi

# Cap on the RDKit timed comparison for benches that operate on independent
# items (etkdg, ff_optimize, conformer_rmsd). Once exceeded, the bench breaks
# out of the per-mol loop and reports throughput on the molecules actually
# processed. Butina and cross_similarity run whole-batch operations that
# aren't cleanly cuttable, so they ignore this cap.
RDKIT_MAX_SECONDS=300

SYSINFO_DIR="$OUTPUT_DIR/sysinfo"
mkdir -p "$SYSINFO_DIR"
{
  echo "date_utc: $(date -u -Iseconds)"
  echo "date_local: $(date -Iseconds)"
  echo "hostname: $(hostname)"
  echo "uname: $(uname -a)"
  echo "num_gpus_requested: $NUM_GPUS"
  echo "num_gpus_visible: $GPU_COUNT"
  echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "physical_cores: $PHYSICAL_CORES"
  echo "rdkit_threads: $RDKIT_THREADS"
  echo "data_dir: $DATA_DIR"
  echo "enamine_path: $ENAMINE_CXSMILES"
  echo "skip_rdkit: $SKIP_RDKIT"
  echo "skip_nvmolkit: $SKIP_NVMOLKIT"
  echo "etkdg_num_mols: $ETKDG_NUM_MOLS"
  echo "etkdg_calibration_mols: $ETKDG_CAL_SIZE"
  echo "ff_mmff_bfgs_num_mols: $FF_MMFF_BFGS_NUM_MOLS"
  echo "ff_mmff_bfgs_calibration_mols: $FF_MMFF_BFGS_CAL_SIZE"
  echo "ff_mmff_fire_num_mols: $FF_MMFF_FIRE_NUM_MOLS"
  echo "ff_mmff_fire_calibration_mols: $FF_MMFF_FIRE_CAL_SIZE"
  echo "ff_uff_bfgs_num_mols: $FF_UFF_BFGS_NUM_MOLS"
  echo "ff_uff_bfgs_calibration_mols: $FF_UFF_BFGS_CAL_SIZE"
  echo "ff_uff_fire_num_mols: $FF_UFF_FIRE_NUM_MOLS"
  echo "ff_uff_fire_calibration_mols: $FF_UFF_FIRE_CAL_SIZE"
  echo "mcs_num_mols: $MCS_NUM_MOLS"
  echo "mcs_num_pairs: $MCS_NUM_PAIRS"
  echo "mcs_calibration_pairs: $MCS_AUTOTUNE_PAIRS"
  echo "mcs_parameter_sets: ${MCS_ROWS[*]}"
  echo "autotune_trials: $AUTOTUNE_TRIALS"
  echo "autotune_seconds_per_trial: $AUTOTUNE_TIME_BUDGET"
  echo "benchmark_seed: $BENCHMARK_SEED"
  echo "timing_runs: $TIMING_RUNS"
  echo "substruct_num_mols: $SUBSTRUCT_NUM_MOLS"
  echo "substruct_calibration_mols: per_algorithm_and_smarts"
  echo "etkdg_autotune_bs_max: $ETKDG_AUTOTUNE_BS_MAX"
  echo "ff_autotune_bs_max: $FF_AUTOTUNE_BS_MAX"
  echo "autotune_bpg_max: $AUTOTUNE_BPG_MAX"
  echo "rdkit_max_seconds: $RDKIT_MAX_SECONDS"
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

SUMMARY="$OUTPUT_DIR/summary.tsv"
FAILURES=0
COMPLETED_BENCHES=""
if [ "$CONTINUE" = "1" ] && [ -f "$SUMMARY" ]; then
  COMPLETED_BENCHES="$(awk -F'\t' 'NR>1 && $2=="ok" {print $1}' "$SUMMARY")"
  if [ -n "$COMPLETED_BENCHES" ]; then
    echo "[--continue] resuming from $SUMMARY; skipping previously-ok benches:"
    printf '  %s\n' $COMPLETED_BENCHES
  else
    echo "[--continue] $SUMMARY has no ok rows; running everything"
  fi
else
  if [ "$CONTINUE" = "1" ]; then
    echo "[--continue] no prior $SUMMARY found; running everything"
  fi
  printf "benchmark\tstatus\texit_code\tduration_s\tlog\tresult\n" > "$SUMMARY"
fi

is_completed() {
  local name="$1"
  local entry
  for entry in $COMPLETED_BENCHES; do
    if [ "$entry" = "$name" ]; then
      return 0
    fi
  done
  return 1
}

# Run one bench unconditionally and append its result to $SUMMARY.
run_bench_inner() {
  local name="$1"
  local result_path="$2"
  shift 2
  local log_path="$LOG_DIR/${name}.log"

  if is_completed "$name"; then
    echo "[$name] skipped (--continue: already ok in $SUMMARY)"
    return 0
  fi

  echo "==========================================" 
  echo "[$(date -Iseconds)] Running $name"
  echo "  cmd: $*"
  echo "  log: $log_path"
  echo "=========================================="

  local start_s end_s duration status code
  start_s=$(date +%s)
  # PYTHONUNBUFFERED=1 forces line-buffered stdout/stderr so the tail of the
  # log reflects what the bench is actually doing, not a stale 8 KB block.
  ( cd "$SCRIPT_DIR" && PYTHONUNBUFFERED=1 "$@" ) > "$log_path" 2>&1
  code=$?
  end_s=$(date +%s)
  duration=$((end_s - start_s))
  if [ "$code" -eq 0 ]; then
    status="ok"
  else
    status="fail"
    FAILURES=$(( FAILURES + 1 ))
  fi
  printf "%s\t%s\t%d\t%d\t%s\t%s\n" \
    "$name" "$status" "$code" "$duration" "$log_path" "$result_path" >> "$SUMMARY"
  echo "[$name] $status (exit=$code, ${duration}s)"
}

run_bench() {
  local name="$1"
  if ! should_run "$name"; then
    echo "[$name] skipped (not in --include)"
    printf "%s\t%s\t%d\t%d\t%s\t%s\n" \
      "$name" "skipped" 0 0 "" "" >> "$SUMMARY"
    return 0
  fi
  run_bench_inner "$@"
}

# Per-bench mode flags.
#
# Default (head-to-head): both implementations run with validation enabled.
# --no-rdkit: skip every RDKit timing (and validation, which diffs vs RDKit).
# --no-nvmolkit: RDKit-only mode. Skip every nvMolKit timing AND drop
#   autotune (which requires nvMolKit).
BUTINA_MODE_FLAGS=()
CONFORMER_RMSD_MODE_FLAGS=()
CROSS_SIMILARITY_MODE_FLAGS=()
ETKDG_MODE_FLAGS=()
FF_MODE_FLAGS=()
MCS_MODE_FLAGS=()
SUBSTRUCT_MODE_FLAGS=()
TFD_MODE_FLAGS=(--verify)
# Autotune flag set is added to the etkdg / ff / substruct bench invocations
# verbatim and zeroed out in --no-nvmolkit mode (the bench scripts reject
# --autotune when nvmolkit is disabled). When non-empty the per-bench
# invocations also pass --autotune_save / --autotune_calibration_size /
# --autotune_load to the matching argparse args; those calls live next to the
# bench invocations below and are guarded by AUTOTUNE_ENABLED.
AUTOTUNE_ENABLED=1
if [ "$SKIP_RDKIT" = "1" ]; then
  BUTINA_MODE_FLAGS=(--no-rdkit)
  CONFORMER_RMSD_MODE_FLAGS=(--no_rdkit --no_validate)
  CROSS_SIMILARITY_MODE_FLAGS=(--no-rdkit)
  ETKDG_MODE_FLAGS=(--no_rdkit --no_validate)
  FF_MODE_FLAGS=(--no_rdkit --no_validate)
  MCS_MODE_FLAGS=(--no_rdkit --no_validate)
  SUBSTRUCT_MODE_FLAGS=(--no_rdkit --no_validate)
  TFD_MODE_FLAGS=(--skip-rdkit)
fi
if [ "$SKIP_NVMOLKIT" = "1" ]; then
  BUTINA_MODE_FLAGS=(--no-nvmolkit --no-fused)
  CONFORMER_RMSD_MODE_FLAGS=(--no_nvmolkit --no_validate)
  CROSS_SIMILARITY_MODE_FLAGS=(--no-nvmolkit)
  ETKDG_MODE_FLAGS=(--no_nvmolkit --no_validate)
  FF_MODE_FLAGS=(--no_nvmolkit --no_validate)
  MCS_MODE_FLAGS=(--no_nvmolkit --no_validate)
  SUBSTRUCT_MODE_FLAGS=(--no_nvmolkit --no_validate)
  TFD_MODE_FLAGS=(--skip-nvmolkit)
  AUTOTUNE_ENABLED=0
fi

# Build autotune flag arrays per bench. Empty in --no-nvmolkit mode, since
# autotune requires nvmolkit. The --autotune_save path is appended per
# invocation because each call (top-level + every size-scan bin + every
# substruct row) writes a different config file.
ETKDG_AUTOTUNE_FLAGS=()
MCS_AUTOTUNE_FLAGS=()
if [ "$AUTOTUNE_ENABLED" = "1" ]; then
  ETKDG_AUTOTUNE_FLAGS=(
    --autotune
    --autotune_trials "$AUTOTUNE_TRIALS"
    --autotune_time_budget "$AUTOTUNE_TIME_BUDGET"
    --autotune_calibration_size "$ETKDG_CAL_SIZE"
  )
  MCS_AUTOTUNE_FLAGS=(
    --autotune
    --autotune_trials "$AUTOTUNE_TRIALS"
    --autotune_time_budget "$AUTOTUNE_TIME_BUDGET"
    --autotune_calibration_size "$MCS_AUTOTUNE_PAIRS"
  )
fi

substruct_calibration_size() {
  case "$1:$2" in
    gsi:rdkit_fragment_descriptors_supported) echo 3000000 ;;
    gsi:wehi_pains_supported) echo 3000000 ;;
    gsi:BMS_2006_filter_supported) echo 3000000 ;;
    gsi:rdkit_tautomer_transforms_supported) echo 3000000 ;;
    gsi:rdkit_torsionPreferences_v2_supported) echo 670000 ;;
    dfs:rdkit_fragment_descriptors_supported) echo 3000000 ;;
    dfs:wehi_pains_supported) echo 3000000 ;;
    dfs:BMS_2006_filter_supported) echo 3000000 ;;
    dfs:rdkit_tautomer_transforms_supported) echo 3000000 ;;
    dfs:rdkit_torsionPreferences_v2_supported) echo 700000 ;;
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

# Build the per-invocation "--autotune_save PATH" pair into the named array.
# Empty when autotune is disabled. Pass the variable name (not the value) so
# the caller's array gets populated; e.g. autotune_save_arg my_arr /path.json
autotune_save_arg() {
  local out_var="$1"
  local path="$2"
  if [ "$AUTOTUNE_ENABLED" = "1" ]; then
    eval "$out_var=(--autotune_save \"\$path\")"
  else
    eval "$out_var=()"
  fi
}

run_bench "butina_clustering" \
  "$RESULT_DIR/butina_clustering.csv" \
  python "$SCRIPT_DIR/butina_clustering_bench.py" \
  "$ENAMINE_CXSMILES" \
  --seed "$BENCHMARK_SEED" \
  --nvmolkit-reordering both \
  --output "$RESULT_DIR/butina_clustering.csv" \
  "${BUTINA_MODE_FLAGS[@]}"

run_bench "conformer_rmsd" \
  "$RESULT_DIR/conformer_rmsd.csv" \
  python "$SCRIPT_DIR/conformer_rmsd_bench.py" \
  --smiles "$ENAMINE_CXSMILES" \
  --num_mols 8000 \
  --seed "$BENCHMARK_SEED" \
  --confs_per_mol 10 25 50 100 200 \
  --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
  --output "$RESULT_DIR/conformer_rmsd.csv" \
  "${CONFORMER_RMSD_MODE_FLAGS[@]}"

run_bench "cross_similarity" \
  "$RESULT_DIR/cross_similarity.json" \
  python "$SCRIPT_DIR/cross_similarity_bench.py" \
  --input "$ENAMINE_CXSMILES" \
  --seed "$BENCHMARK_SEED" \
  --cosine \
  --output "$RESULT_DIR/cross_similarity.json" \
  "${CROSS_SIMILARITY_MODE_FLAGS[@]}"

autotune_save_arg etkdg_save_flags "$AUTOTUNE_DIR/etkdg_hardware.json"
run_bench "etkdg" \
  "$RESULT_DIR/etkdg.csv" \
  python "$SCRIPT_DIR/etkdg_bench.py" \
  --smiles "$ENAMINE_CXSMILES" \
  --num_mols "$ETKDG_NUM_MOLS" \
  --seed "$BENCHMARK_SEED" \
  --runs "$TIMING_RUNS" \
  --confs_per_mol "$ETKDG_CONFS_PER_MOL" \
  --num_gpus "$NUM_GPUS" \
  --rdkit_threads "$RDKIT_THREADS" \
  --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
  "${ETKDG_AUTOTUNE_FLAGS[@]}" \
  "${etkdg_save_flags[@]}" \
  --output "$RESULT_DIR/etkdg.csv" \
  "${ETKDG_MODE_FLAGS[@]}"

# Exercise every supported force-field/minimizer pairing. The minimizer kind
# is part of the benchmark name because BFGS and FIRE have distinct tuning
# optima and performance characteristics.
FF_ROWS=(
  "mmff:BFGS"
  "mmff:FIRE"
  "uff:BFGS"
  "uff:FIRE"
)
for row in "${FF_ROWS[@]}"; do
  ff="${row%%:*}"
  minimizer_kind="${row##*:}"
  minimizer_stem="${minimizer_kind,,}"
  ff_name="ff_optimize_${ff}_${minimizer_stem}"
  raw_prefix="FF_${ff^^}_${minimizer_kind}"
  num_mols_var="${raw_prefix}_NUM_MOLS"
  cal_size_var="${raw_prefix}_CAL_SIZE"
  ff_num_mols="${!num_mols_var}"
  ff_cal_size="${!cal_size_var}"
  ff_timing_runs="$TIMING_RUNS"
  FF_AUTOTUNE_FLAGS=()
  if [ "$AUTOTUNE_ENABLED" = "1" ]; then
    FF_AUTOTUNE_FLAGS=(
      --autotune
      --autotune_trials "$AUTOTUNE_TRIALS"
      --autotune_time_budget "$AUTOTUNE_TIME_BUDGET"
      --autotune_calibration_size "$ff_cal_size"
    )
  fi
  autotune_save_arg ff_save_flags "$AUTOTUNE_DIR/${ff_name}_hardware.json"
  run_bench "$ff_name" \
    "$RESULT_DIR/${ff_name}.csv" \
    python "$SCRIPT_DIR/ff_optimize_bench.py" \
    --smiles "$ENAMINE_CXSMILES" \
    --num_mols "$ff_num_mols" \
    --seed "$BENCHMARK_SEED" \
    --runs "$ff_timing_runs" \
    --confs_per_mol "$FF_CONFS_PER_MOL" \
    --ff "$ff" \
    --minimizer_kind "$minimizer_kind" \
    --max_iters "$FF_MAX_ITERS" \
    --num_gpus "$NUM_GPUS" \
    --rdkit_threads "$RDKIT_THREADS" \
    --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
    "${FF_AUTOTUNE_FLAGS[@]}" \
    "${ff_save_flags[@]}" \
    --output "$RESULT_DIR/${ff_name}.csv" \
    "${FF_MODE_FLAGS[@]}"

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
  autotune_save_arg mcs_save_flags "$AUTOTUNE_DIR/${mcs_name}_config.json"
  run_bench "$mcs_name" \
    "$RESULT_DIR/${mcs_name}.csv" \
    python "$SCRIPT_DIR/mcs_bench.py" \
    --smiles "$ENAMINE_CXSMILES" \
    --num_mols "$MCS_NUM_MOLS" \
    --num_pairs "$MCS_NUM_PAIRS" \
    --seed "$BENCHMARK_SEED" \
    --runs "$TIMING_RUNS" \
    --num_gpus "$NUM_GPUS" \
    --rdkit_threads "$RDKIT_THREADS" \
    --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
    "${MCS_PARAMETER_FLAGS[@]}" \
    "${MCS_AUTOTUNE_FLAGS[@]}" \
    "${mcs_save_flags[@]}" \
    --output "$RESULT_DIR/${mcs_name}.csv" \
    "${MCS_MODE_FLAGS[@]}"
done

for algorithm in gsi dfs; do
  for row in "${SUBSTRUCT_ROWS[@]}"; do
    smarts_file="${row%%:*}"
    mode="${row##*:}"
    smarts_path="$SMARTS_DIR/$smarts_file"
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
        --autotune_calibration_size "$cal_size"
      )
    fi
    autotune_save_arg substruct_save_flags "$AUTOTUNE_DIR/${bench_name}_config.json"
    run_bench "$bench_name" \
      "$LOG_DIR/${bench_name}.log" \
      python "$SCRIPT_DIR/substruct_bench.py" \
      --smiles "$ENAMINE_CXSMILES" \
      --num_mols "$SUBSTRUCT_NUM_MOLS" \
      --seed "$BENCHMARK_SEED" \
      --runs "$runs" \
      --sanitize \
      --smarts "$smarts_path" \
      --mode "$mode" \
      --algorithm "$algorithm" \
      --num_gpus "$NUM_GPUS" \
      --rdkit_threads "$RDKIT_THREADS" \
      --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
      "${SUBSTRUCT_AUTOTUNE_FLAGS[@]}" \
      "${substruct_save_flags[@]}" \
      "${SUBSTRUCT_MODE_FLAGS[@]}"
  done
done

run_bench "tfd" \
  "$RESULT_DIR/tfd.csv" \
  python "$SCRIPT_DIR/tfd_bench.py" \
  --smiles-file "$ENAMINE_CXSMILES" \
  --seed "$BENCHMARK_SEED" \
  --num-mols 100 1000 5000 \
  --output "$RESULT_DIR/tfd.csv" \
  "${TFD_MODE_FLAGS[@]}"

echo
echo "All benchmarks complete. Output: $OUTPUT_DIR"
echo "Summary:"
column -t -s $'\t' "$SUMMARY" || cat "$SUMMARY"
if [ "$FAILURES" -ne 0 ]; then
  echo "Error: $FAILURES benchmark(s) failed" >&2
  exit 1
fi
