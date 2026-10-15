#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the publication suite with fixed nvMolKit execution settings on any machine.
#
# Every batch size, batches-per-GPU, worker, preprocessing-thread and executor count comes from one
# settings file (default: hardware_configs/gb300_cpu14.json, the GB300 14-core scan), whatever GPU is
# present, so results from different machines use identical knobs. run_publication_benchmarks.sh does the
# work; this script pins its settings and CPU budget.
#
# Repeats are a benchmark parameter: --runs N times each workload N times in one process, after parsing
# and warmup, and each result row records the mean and standard deviation.
#
# nvMolKit and RDKit run separately so each side measures the same workload on every machine (several
# workload sizes differ when RDKit runs alongside):
#   OUTPUT_DIR/nvmolkit   nvMolKit results
#   OUTPUT_DIR/rdkit      RDKit baselines (several hours; skip with --no-rdkit)
# and OUTPUT_DIR/summary.tsv lists every benchmark of both.

set -uo pipefail

usage() {
  cat >&2 <<EOF
Usage: $0 --dataset FILE --output-dir DIR [options]

Required:
  --dataset FILE        SMILES/CXSMILES dataset used by every benchmark
  --output-dir DIR      Parent directory for the nvMolKit and RDKit results

Options:
  --runs N              Timed runs per benchmark (default: 3)
  --no-rdkit            Skip the RDKit baselines
  --no-nvmolkit         Skip nvMolKit
  --gpu-id ID           Physical GPU to expose (default: 0)
  --hw-configs FILE     Fixed settings (default: hardware_configs/gb300_cpu14.json)
  --aap-csv FILE        Scored CSV for AAP clustering (passed through)
  --only GLOBS          Comma-separated benchmark-name globs (passed through)
  -h, --help            Show this help

The settings were scanned with a 14-core CPU budget (workers + preprocessing threads = 14), so the
machine needs at least 14 physical cores available to this process.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER="$SCRIPT_DIR/run_publication_benchmarks.sh"
CPUS=14

DATASET=""
OUTPUT_DIR=""
RUN_RDKIT=1
RUN_NVMOLKIT=1
HW_CONFIG_FILE="$SCRIPT_DIR/hardware_configs/gb300_cpu14.json"
PASSTHROUGH=()

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset|--output-dir|--runs|--gpu-id|--hw-configs|--aap-csv|--only)
      [ $# -ge 2 ] || { echo "Error: $1 requires a value" >&2; exit 2; }
      case "$1" in
        --dataset) DATASET="$2" ;;
        --output-dir) OUTPUT_DIR="$2" ;;
        --hw-configs) HW_CONFIG_FILE="$2" ;;
        *) PASSTHROUGH+=("$1" "$2") ;;
      esac
      shift 2
      ;;
    --no-rdkit)
      RUN_RDKIT=0
      shift
      ;;
    --no-nvmolkit)
      RUN_NVMOLKIT=0
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

if [ -z "$DATASET" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Error: --dataset and --output-dir are required" >&2
  usage
  exit 2
fi
if [ "$RUN_RDKIT" = "0" ] && [ "$RUN_NVMOLKIT" = "0" ]; then
  echo "Error: --no-rdkit and --no-nvmolkit are mutually exclusive" >&2
  exit 2
fi
if [ ! -f "$HW_CONFIG_FILE" ]; then
  echo "Error: settings file does not exist: $HW_CONFIG_FILE" >&2
  exit 1
fi
HW_CONFIG_FILE="$(cd "$(dirname "$HW_CONFIG_FILE")" && pwd)/$(basename "$HW_CONFIG_FILE")"
mkdir -p "$OUTPUT_DIR" || { echo "Error: could not create $OUTPUT_DIR" >&2; exit 1; }
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

SUMMARY="$OUTPUT_DIR/summary.tsv"
printf "side\tbenchmark\tstatus\texit_code\tduration_s\tlog\tresult\n" > "$SUMMARY"
FAILED=0

# run_side SIDE MODE_FLAG: one complete suite run into OUTPUT_DIR/SIDE.
run_side() {
  local side="$1" mode_flag="$2"
  echo "##########################################"
  echo "[$(date -Iseconds)] $side -> $OUTPUT_DIR/$side"
  echo "##########################################"
  # The runner always receives an explicit settings file, so it applies it regardless of the GPU model.
  bash "$RUNNER" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR/$side" \
    --cpus "$CPUS" \
    --hw-configs "$HW_CONFIG_FILE" \
    "$mode_flag" \
    "${PASSTHROUGH[@]}"
  local code=$?
  [ "$code" -ne 0 ] && FAILED=1
  # Exit code 2 is an argument or budget error; the other side would hit it too.
  if [ "$code" -eq 2 ]; then
    echo "Error: the $side run was rejected by the runner; stopping" >&2
    exit 2
  fi
  if [ -f "$OUTPUT_DIR/$side/summary.tsv" ]; then
    awk -F'\t' -v OFS='\t' -v side="$side" 'NR > 1 { print side, $0 }' "$OUTPUT_DIR/$side/summary.tsv" >> "$SUMMARY"
  fi
}

[ "$RUN_NVMOLKIT" = "1" ] && run_side nvmolkit --no-rdkit
[ "$RUN_RDKIT" = "1" ] && run_side rdkit --no-nvmolkit

echo
echo "Benchmarks complete. Output: $OUTPUT_DIR"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
if [ "$FAILED" -ne 0 ]; then
  echo "Error: some benchmarks failed; see $SUMMARY" >&2
  exit 1
fi
