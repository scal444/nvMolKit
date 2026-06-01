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
# Run the full set of nvMolKit Python benchmarks for either a 1-GPU or 8-GPU
# hardware configuration and collect results plus system metadata into a
# single output directory.
#
# Both modes run the full bench list. The benches that don't expose a
# --num_gpus flag (butina_clustering, conformer_rmsd, cross_similarity, tfd)
# always run on a single GPU regardless of the mode argument; the mode arg
# only affects --num_gpus on the multi-GPU-aware benches:
#
#   - butina_clustering_bench.py    (single-GPU library; --num_gpus N/A)
#   - conformer_rmsd_bench.py       (single-GPU library; --num_gpus N/A)
#   - cross_similarity_bench.py     (single-GPU library; --num_gpus N/A)
#   - etkdg_bench.py                (autotuned, --num_gpus = mode arg)
#   - etkdg_size_scan               (etkdg_bench.py looped over chembl_size_splits bins)
#   - ff_optimize_bench.py          (autotuned, MMFF and UFF, --num_gpus = mode arg)
#   - ff_optimize_{mmff,uff}_size_scan (looped over chembl_size_splits bins)
#   - substruct_bench.py            (autotuned, one row per SMARTS, --num_gpus = mode arg)
#   - tfd_bench.py                  (single-GPU library; --num_gpus N/A)
#
# Autotune budget is fixed at 20 trials × 60 s/trial per autotuned invocation.
# Size-scan benches autotune once per bin, so the total budget multiplies by
# the number of bins.
#
# Tuned HardwareOptions / SubstructSearchConfig JSON files are saved under
# $OUTPUT_DIR/autotune for reproducibility.
#
# Data inputs come from --data-dir (default /data, the in-container path; on
# the host it is typically ~/data). Required layout:
#
#   $DATA_DIR/enamine_real_10M.cxsmiles
#   $DATA_DIR/chembl_size_splits/chembl_<lo>-<hi>.smi   (for size scans)
#
# Usage:
#   ./run_all_benchmarks.sh <1|8> --output-dir DIR [--data-dir DIR] [--include NAME [NAME ...]]
#   ./run_all_benchmarks.sh --list
#
# --include restricts the run to the named benches (whitelist). Unknown names
# are an error. With no --include all benches run. --list prints the bench
# names and exits.

set -uo pipefail

usage() {
  cat >&2 <<EOF
Usage: $0 <1|8> --output-dir DIR [--data-dir DIR] [--include NAME [NAME ...]] [--no-rdkit | --no-nvmolkit]
       $0 --list

  <1|8>             GPU mode (required)
  --output-dir DIR  Results directory (required for runs)
  --data-dir DIR    Input data directory (default: /data)
  --include NAME... Whitelist of bench names to run (run --list to see them)
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

if [ $# -lt 1 ]; then
  usage
  exit 2
fi

case "$1" in
  -h|--help)
    usage
    exit 0
    ;;
  --list)
    LIST_ONLY=1
    NUM_GPUS=""
    shift
    ;;
  *)
    NUM_GPUS="$1"
    shift
    case "$NUM_GPUS" in
      1|8) ;;
      *)
        echo "Error: first argument must be 1 or 8 (got: $NUM_GPUS)" >&2
        usage
        exit 2
        ;;
    esac
    LIST_ONLY=0
    ;;
esac

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

# Heavy-atom-count bins for ETKDG / FF size scans, matching the file names
# in $DATA_DIR/chembl_size_splits/chembl_<bin>.smi. Stops at 80-100 because
# bins >=100 atoms are biological outliers (peptides etc.), have too few
# molecules to saturate the 8-GPU calibration target, and aren't
# representative of typical drug-discovery workloads.
SIZE_SCAN_BINS=(
  "0-20"
  "20-40"
  "40-60"
  "60-80"
  "80-100"
)

# Enumerate every bench name this script can run, in the order they execute.
ALL_BENCH_NAMES=(
  "butina_clustering"
  "conformer_rmsd"
  "cross_similarity"
  "etkdg"
  "etkdg_size_scan"
  "ff_optimize_mmff"
  "ff_optimize_mmff_size_scan"
  # "ff_optimize_uff"            # disabled; MMFF is the reference
  # "ff_optimize_uff_size_scan"  # disabled; MMFF is the reference
)
for row in "${SUBSTRUCT_ROWS[@]}"; do
  smarts_file="${row%%:*}"
  ALL_BENCH_NAMES+=("substruct_${smarts_file%.txt}")
done
ALL_BENCH_NAMES+=("tfd")

OUTPUT_DIR=""
DATA_DIR="/data"
INCLUDE_LIST=()
SKIP_RDKIT=0
SKIP_NVMOLKIT=0
CONTINUE=0

while [ $# -gt 0 ]; do
  case "$1" in
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
    --continue)
      CONTINUE=1
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
SIZE_SCAN_DIR="$DATA_DIR/chembl_size_splits"

if [ ! -f "$ENAMINE_CXSMILES" ]; then
  echo "Missing $ENAMINE_CXSMILES (used by butina/cross_similarity/etkdg/ff/substruct/tfd)" >&2
  exit 1
fi
if [ ! -d "$SIZE_SCAN_DIR" ]; then
  echo "Missing $SIZE_SCAN_DIR (required for *_size_scan benches)" >&2
  exit 1
fi

# Verify enough GPUs are visible before we burn time tuning configs that
# request hardware we don't have.
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
  echo "Error: requested $NUM_GPUS GPUs but nvidia-smi reports $GPU_COUNT visible" >&2
  exit 1
fi

# Ensure runtime dependencies the bench scripts assume but the conda image may
# not ship. pyperf is used by cross_similarity_bench.py; optuna is required for
# autotune in etkdg/ff/substruct benches.
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
# Mirrors the autotune defaults in nvmolkit/autotune/_ff_common.py and
# tune_embed_molecules.py: batchSize_max=1024, batchesPerGpu_max=8.
AUTOTUNE_BS_MAX=1024
AUTOTUNE_BPG_MAX=8
ETKDG_CAL_SIZE=$(( 2 * AUTOTUNE_BS_MAX * AUTOTUNE_BPG_MAX * NUM_GPUS / ETKDG_CONFS_PER_MOL ))
FF_CAL_SIZE=$(( 2 * AUTOTUNE_BS_MAX * AUTOTUNE_BPG_MAX * NUM_GPUS / FF_CONFS_PER_MOL ))
# Substruct has no confsPerMol multiplier; the calibration is in mols directly.
SUBSTRUCT_CAL_SIZE=$(( 2 * AUTOTUNE_BS_MAX * AUTOTUNE_BPG_MAX * NUM_GPUS ))

# Runtime workload is RUNTIME_MULTIPLIER * calibration size, sized so the
# head-to-head measurement runs for ~10-100s of nvmolkit work even on the
# smallest-molecule bin. RUNTIME_MULTIPLIER must be >= 1 so the autotune
# calibration is a proper subset of the runtime workload. RDKit-side wall is
# bounded by RDKIT_MAX_SECONDS regardless.
RUNTIME_MULTIPLIER=10
ETKDG_NUM_MOLS=$(( ETKDG_CAL_SIZE * RUNTIME_MULTIPLIER ))
FF_NUM_MOLS=$(( FF_CAL_SIZE * RUNTIME_MULTIPLIER ))
# Per-bin workload for the size scan: same formula.
SIZE_SCAN_NUM_MOLS="$ETKDG_NUM_MOLS"

# Butina needs >=40k molecules for its rdkit_lowmem variant. Cap at 60k so
# fingerprint construction stays bounded.
BUTINA_NUM_MOLS=60000

# Substruct full run scans the entire enamine_real_10M file. load_smiles with
# max_count=10M effectively streams the whole file (10.35M lines). RAM cost is
# substantial (tens of GB of parsed Mols); only safe on the fat 8-GPU node.
SUBSTRUCT_NUM_MOLS=10000000

# RDKit thread count for the head-to-head comparison on the multi-GPU benches
# (etkdg, ff_optimize, substruct). 1-GPU mode caps at 16 (or physical-core
# count if smaller); 8-GPU mode uses every physical core on the assumption
# it's a fat multi-socket node.
#
# Single-GPU benches (butina, conformer_rmsd, cross_similarity, tfd) compare
# against single-threaded RDKit and ignore this variable.
PHYSICAL_CORES="$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
if [ -z "$PHYSICAL_CORES" ] || [ "$PHYSICAL_CORES" -lt 1 ]; then
  PHYSICAL_CORES=1
fi
case "$NUM_GPUS" in
  1)
    if [ "$PHYSICAL_CORES" -lt 16 ]; then
      RDKIT_THREADS="$PHYSICAL_CORES"
    else
      RDKIT_THREADS=16
    fi
    ;;
  8)
    RDKIT_THREADS="$PHYSICAL_CORES"
    ;;
esac

# Autotune budget (per autotuned invocation).
AUTOTUNE_TRIALS=20
AUTOTUNE_TIME_BUDGET=60

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
  echo "physical_cores: $PHYSICAL_CORES"
  echo "rdkit_threads: $RDKIT_THREADS"
  echo "data_dir: $DATA_DIR"
  echo "enamine_path: $ENAMINE_CXSMILES"
  echo "size_scan_dir: $SIZE_SCAN_DIR"
  echo "skip_rdkit: $SKIP_RDKIT"
  echo "skip_nvmolkit: $SKIP_NVMOLKIT"
  echo "etkdg_num_mols: $ETKDG_NUM_MOLS"
  echo "etkdg_calibration_mols: $ETKDG_CAL_SIZE"
  echo "ff_num_mols: $FF_NUM_MOLS"
  echo "ff_calibration_mols: $FF_CAL_SIZE"
  echo "runtime_multiplier: $RUNTIME_MULTIPLIER"
  echo "substruct_num_mols: $SUBSTRUCT_NUM_MOLS"
  echo "substruct_calibration_mols: $SUBSTRUCT_CAL_SIZE"
  echo "autotune_bs_max: $AUTOTUNE_BS_MAX"
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

# Run one bench unconditionally and append its result to $SUMMARY. Used directly
# for size-scan bins (parent scan does its own --include check) and via
# run_bench for top-level benches.
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
  set +e
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
# Default (head-to-head): both implementations run; validation is opt-in.
# --no-rdkit: skip every RDKit timing (and validation, which diffs vs RDKit).
# --no-nvmolkit: RDKit-only mode. Skip every nvMolKit timing AND drop
#   autotune (which requires nvMolKit).
#
# Validation is currently disabled even when RDKit is on: the per-conformer
# MMFF energy reconstruction in etkdg/ff/substruct is single-threaded and
# blows up wall time on large workloads. Re-enable by dropping --no_validate
# once those validators are parallelized.
BUTINA_MODE_FLAGS=()
CONFORMER_RMSD_MODE_FLAGS=()
CROSS_SIMILARITY_MODE_FLAGS=()
ETKDG_MODE_FLAGS=(--no_validate)
FF_MODE_FLAGS=(--no_validate)
SUBSTRUCT_MODE_FLAGS=(--no_validate)
TFD_MODE_FLAGS=()
# Autotune flag set is added to the etkdg / ff / substruct bench invocations
# verbatim and zeroed out in --no-nvmolkit mode (the bench scripts reject
# --autotune when nvmolkit is disabled). When non-empty the per-bench
# invocations also pass --autotune_save / --autotune_calibration_size /
# --autotune_load to the matching argparse args; those calls live next to the
# bench invocations below and are guarded by AUTOTUNE_ENABLED.
AUTOTUNE_ENABLED=1
if [ "$SKIP_RDKIT" = "1" ]; then
  BUTINA_MODE_FLAGS=(--no-rdkit)
  CONFORMER_RMSD_MODE_FLAGS=(--no-rdkit)
  CROSS_SIMILARITY_MODE_FLAGS=(--no-rdkit)
  ETKDG_MODE_FLAGS=(--no_rdkit --no_validate)
  FF_MODE_FLAGS=(--no_rdkit --no_validate)
  SUBSTRUCT_MODE_FLAGS=(--no_rdkit --no_validate)
  TFD_MODE_FLAGS=(--skip-rdkit)
fi
if [ "$SKIP_NVMOLKIT" = "1" ]; then
  BUTINA_MODE_FLAGS=(--no-nvmolkit --no-fused)
  CONFORMER_RMSD_MODE_FLAGS=(--no-nvmolkit)
  CROSS_SIMILARITY_MODE_FLAGS=(--no-nvmolkit)
  ETKDG_MODE_FLAGS=(--no_nvmolkit --no_validate)
  FF_MODE_FLAGS=(--no_nvmolkit --no_validate)
  SUBSTRUCT_MODE_FLAGS=(--no_nvmolkit --no_validate)
  TFD_MODE_FLAGS=(--skip-nvmolkit)
  AUTOTUNE_ENABLED=0
fi

# Build autotune flag arrays per bench. Empty in --no-nvmolkit mode, since
# autotune requires nvmolkit. The --autotune_save path is appended per
# invocation because each call (top-level + every size-scan bin + every
# substruct row) writes a different config file.
ETKDG_AUTOTUNE_FLAGS=()
FF_AUTOTUNE_FLAGS=()
SUBSTRUCT_AUTOTUNE_FLAGS=()
if [ "$AUTOTUNE_ENABLED" = "1" ]; then
  ETKDG_AUTOTUNE_FLAGS=(
    --autotune
    --autotune_trials "$AUTOTUNE_TRIALS"
    --autotune_time_budget "$AUTOTUNE_TIME_BUDGET"
    --autotune_calibration_size "$ETKDG_CAL_SIZE"
  )
  FF_AUTOTUNE_FLAGS=(
    --autotune
    --autotune_trials "$AUTOTUNE_TRIALS"
    --autotune_time_budget "$AUTOTUNE_TIME_BUDGET"
    --autotune_calibration_size "$FF_CAL_SIZE"
  )
  SUBSTRUCT_AUTOTUNE_FLAGS=(
    --autotune
    --autotune_trials "$AUTOTUNE_TRIALS"
    --autotune_time_budget "$AUTOTUNE_TIME_BUDGET"
    --autotune_calibration_size "$SUBSTRUCT_CAL_SIZE"
  )
fi

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
  --output "$RESULT_DIR/butina_clustering.csv" \
  "${BUTINA_MODE_FLAGS[@]}"

run_bench "conformer_rmsd" \
  "$RESULT_DIR/conformer_rmsd.csv" \
  python "$SCRIPT_DIR/conformer_rmsd_bench.py" \
  --smiles "$ENAMINE_CXSMILES" \
  --num_mols 2000 \
  --confs_per_mol 10 25 50 100 200 \
  --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
  --output "$RESULT_DIR/conformer_rmsd.csv" \
  "${CONFORMER_RMSD_MODE_FLAGS[@]}"

run_bench "cross_similarity" \
  "$RESULT_DIR/cross_similarity.json" \
  python "$SCRIPT_DIR/cross_similarity_bench.py" \
  --input "$ENAMINE_CXSMILES" \
  --output "$RESULT_DIR/cross_similarity.json" \
  "${CROSS_SIMILARITY_MODE_FLAGS[@]}"

autotune_save_arg etkdg_save_flags "$AUTOTUNE_DIR/etkdg_hardware.json"
run_bench "etkdg" \
  "$RESULT_DIR/etkdg.csv" \
  python "$SCRIPT_DIR/etkdg_bench.py" \
  --smiles "$ENAMINE_CXSMILES" \
  --num_mols "$ETKDG_NUM_MOLS" \
  --confs_per_mol "$ETKDG_CONFS_PER_MOL" \
  --num_gpus "$NUM_GPUS" \
  --rdkit_threads "$RDKIT_THREADS" \
  --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
  "${ETKDG_AUTOTUNE_FLAGS[@]}" \
  "${etkdg_save_flags[@]}" \
  --output "$RESULT_DIR/etkdg.csv" \
  "${ETKDG_MODE_FLAGS[@]}"

if should_run "etkdg_size_scan"; then
  echo "[etkdg_size_scan] sweeping ${#SIZE_SCAN_BINS[@]} bins"
  for bin in "${SIZE_SCAN_BINS[@]}"; do
    bin_smi="$SIZE_SCAN_DIR/chembl_${bin}.smi"
    if [ ! -f "$bin_smi" ]; then
      echo "  skipping bin $bin (missing $bin_smi)"
      continue
    fi
    bin_name="etkdg_size_scan_${bin}"
    autotune_save_arg etkdg_bin_save_flags "$AUTOTUNE_DIR/etkdg_size_scan_${bin}_hardware.json"
    run_bench_inner "$bin_name" \
      "$RESULT_DIR/etkdg_size_scan_${bin}.csv" \
      python "$SCRIPT_DIR/etkdg_bench.py" \
      --smiles "$bin_smi" \
      --num_mols "$SIZE_SCAN_NUM_MOLS" \
      --confs_per_mol "$ETKDG_CONFS_PER_MOL" \
      --num_gpus "$NUM_GPUS" \
      --rdkit_threads "$RDKIT_THREADS" \
      --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
      "${ETKDG_AUTOTUNE_FLAGS[@]}" \
      "${etkdg_bin_save_flags[@]}" \
      --output "$RESULT_DIR/etkdg_size_scan_${bin}.csv" \
      "${ETKDG_MODE_FLAGS[@]}"
  done
else
  echo "[etkdg_size_scan] skipped (not in --include)"
  printf "%s\t%s\t%d\t%d\t%s\t%s\n" "etkdg_size_scan" "skipped" 0 0 "" "" >> "$SUMMARY"
fi

# UFF disabled; MMFF is the reference. Re-add "uff" to the list to run it.
for ff in mmff; do
  autotune_save_arg ff_save_flags "$AUTOTUNE_DIR/ff_optimize_${ff}_hardware.json"
  run_bench "ff_optimize_${ff}" \
    "$RESULT_DIR/ff_optimize_${ff}.csv" \
    python "$SCRIPT_DIR/ff_optimize_bench.py" \
    --smiles "$ENAMINE_CXSMILES" \
    --num_mols "$FF_NUM_MOLS" \
    --confs_per_mol "$FF_CONFS_PER_MOL" \
    --ff "$ff" \
    --max_iters "$FF_MAX_ITERS" \
    --num_gpus "$NUM_GPUS" \
    --rdkit_threads "$RDKIT_THREADS" \
    --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
    "${FF_AUTOTUNE_FLAGS[@]}" \
    "${ff_save_flags[@]}" \
    --output "$RESULT_DIR/ff_optimize_${ff}.csv" \
    "${FF_MODE_FLAGS[@]}"

  scan_name="ff_optimize_${ff}_size_scan"
  if should_run "$scan_name"; then
    echo "[$scan_name] sweeping ${#SIZE_SCAN_BINS[@]} bins"
    for bin in "${SIZE_SCAN_BINS[@]}"; do
      bin_smi="$SIZE_SCAN_DIR/chembl_${bin}.smi"
      if [ ! -f "$bin_smi" ]; then
        echo "  skipping bin $bin (missing $bin_smi)"
        continue
      fi
      bin_name="${scan_name}_${bin}"
      autotune_save_arg ff_bin_save_flags "$AUTOTUNE_DIR/${bin_name}_hardware.json"
      run_bench_inner "$bin_name" \
        "$RESULT_DIR/${bin_name}.csv" \
        python "$SCRIPT_DIR/ff_optimize_bench.py" \
        --smiles "$bin_smi" \
        --num_mols "$SIZE_SCAN_NUM_MOLS" \
        --confs_per_mol "$FF_CONFS_PER_MOL" \
        --ff "$ff" \
        --max_iters "$FF_MAX_ITERS" \
        --num_gpus "$NUM_GPUS" \
        --rdkit_threads "$RDKIT_THREADS" \
        --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
        "${FF_AUTOTUNE_FLAGS[@]}" \
        "${ff_bin_save_flags[@]}" \
        --output "$RESULT_DIR/${bin_name}.csv" \
        "${FF_MODE_FLAGS[@]}"
    done
  else
    echo "[$scan_name] skipped (not in --include)"
    printf "%s\t%s\t%d\t%d\t%s\t%s\n" "$scan_name" "skipped" 0 0 "" "" >> "$SUMMARY"
  fi
done

for row in "${SUBSTRUCT_ROWS[@]}"; do
  smarts_file="${row%%:*}"
  mode="${row##*:}"
  smarts_path="$SMARTS_DIR/$smarts_file"
  smarts_stem="${smarts_file%.txt}"
  bench_name="substruct_${smarts_stem}"
  autotune_save_arg substruct_save_flags "$AUTOTUNE_DIR/${bench_name}_config.json"
  run_bench "$bench_name" \
    "$LOG_DIR/${bench_name}.log" \
    python "$SCRIPT_DIR/substruct_bench.py" \
    --smiles "$ENAMINE_CXSMILES" \
    --num_mols "$SUBSTRUCT_NUM_MOLS" \
    --sanitize \
    --smarts "$smarts_path" \
    --mode "$mode" \
    --num_gpus "$NUM_GPUS" \
    --rdkit_threads "$RDKIT_THREADS" \
    --rdkit_match_mode raw substructlib \
    --rdkit_max_seconds "$RDKIT_MAX_SECONDS" \
    "${SUBSTRUCT_AUTOTUNE_FLAGS[@]}" \
    "${substruct_save_flags[@]}" \
    "${SUBSTRUCT_MODE_FLAGS[@]}"
done

run_bench "tfd" \
  "$RESULT_DIR/tfd.csv" \
  python "$SCRIPT_DIR/tfd_bench.py" \
  --smiles-file "$ENAMINE_CXSMILES" \
  --output "$RESULT_DIR/tfd.csv" \
  "${TFD_MODE_FLAGS[@]}"

echo
echo "All benchmarks complete. Output: $OUTPUT_DIR"
echo "Summary:"
column -t -s $'\t' "$SUMMARY" || cat "$SUMMARY"
