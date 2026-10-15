#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the publication suite with fixed nvMolKit execution settings on any machine, repeated for statistics.
#
# Every batch size, batches-per-GPU, worker, preprocessing-thread and executor count comes from one
# settings file (default: hardware_configs/gb300_cpu14.json, the GB300 14-core scan), whatever GPU is
# present, so results from different machines use identical knobs. run_publication_benchmarks.sh does the
# work; this script pins its settings and CPU budget and repeats it.
#
# nvMolKit and RDKit run as separate passes so each repeat of one side measures the same workload
# (several workload sizes differ when RDKit runs alongside). Each pass writes a complete run directory:
#   OUTPUT_DIR/nvmolkit/rep_01 ... rep_NN
#   OUTPUT_DIR/rdkit/rep_01 ...    rep_MM
# and OUTPUT_DIR/summary.tsv lists every benchmark of every pass.

set -uo pipefail

usage() {
  cat >&2 <<EOF
Usage: $0 --dataset FILE --output-dir DIR [options]

Required:
  --dataset FILE        SMILES/CXSMILES dataset used by every benchmark
  --output-dir DIR      Parent directory for all passes

Options:
  --repeats N           nvMolKit passes (default: 1; 0 skips nvMolKit)
  --rdkit-repeats M     RDKit passes (default: 1; 0 skips RDKit). One RDKit pass takes several hours.
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
REPEATS=1
RDKIT_REPEATS=1
HW_CONFIG_FILE="$SCRIPT_DIR/hardware_configs/gb300_cpu14.json"
PASSTHROUGH=()

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset|--output-dir|--repeats|--rdkit-repeats|--gpu-id|--hw-configs|--aap-csv|--only)
      [ $# -ge 2 ] || { echo "Error: $1 requires a value" >&2; exit 2; }
      case "$1" in
        --dataset) DATASET="$2" ;;
        --output-dir) OUTPUT_DIR="$2" ;;
        --repeats) REPEATS="$2" ;;
        --rdkit-repeats) RDKIT_REPEATS="$2" ;;
        --hw-configs) HW_CONFIG_FILE="$2" ;;
        *) PASSTHROUGH+=("$1" "$2") ;;
      esac
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
for value in "$REPEATS" "$RDKIT_REPEATS"; do
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "Error: --repeats and --rdkit-repeats must be non-negative integers" >&2
    exit 2
  fi
done
if [ "$REPEATS" -eq 0 ] && [ "$RDKIT_REPEATS" -eq 0 ]; then
  echo "Error: nothing to run (--repeats 0 and --rdkit-repeats 0)" >&2
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
printf "pass\trep\tbenchmark\tstatus\texit_code\tduration_s\tlog\tresult\n" > "$SUMMARY"
FAILED_PASSES=0

# run_pass SIDE REP MODE_FLAG: one complete suite run into OUTPUT_DIR/SIDE/rep_NN.
run_pass() {
  local side="$1" rep="$2" mode_flag="$3"
  local rep_dir
  rep_dir="$OUTPUT_DIR/$side/$(printf 'rep_%02d' "$rep")"
  echo "##########################################"
  echo "[$(date -Iseconds)] $side pass $rep -> $rep_dir"
  echo "##########################################"
  # The runner always receives an explicit settings file, so it applies it regardless of the GPU model.
  bash "$RUNNER" \
    --dataset "$DATASET" \
    --output-dir "$rep_dir" \
    --cpus "$CPUS" \
    --hw-configs "$HW_CONFIG_FILE" \
    "$mode_flag" \
    "${PASSTHROUGH[@]}"
  local code=$?
  if [ "$code" -ne 0 ]; then
    FAILED_PASSES=$(( FAILED_PASSES + 1 ))
  fi
  # Exit code 2 is an argument or budget error raised before any benchmark ran.
  if [ "$code" -eq 2 ]; then
    echo "Error: $side pass $rep was rejected by the runner; stopping" >&2
    exit 2
  fi
  if [ -f "$rep_dir/summary.tsv" ]; then
    awk -F'\t' -v OFS='\t' -v side="$side" -v rep="$rep" 'NR > 1 { print side, rep, $0 }' \
      "$rep_dir/summary.tsv" >> "$SUMMARY"
  fi
}

for (( rep = 1; rep <= REPEATS; rep++ )); do
  run_pass nvmolkit "$rep" --no-rdkit
done
for (( rep = 1; rep <= RDKIT_REPEATS; rep++ )); do
  run_pass rdkit "$rep" --no-nvmolkit
done

echo
echo "All passes complete. Output: $OUTPUT_DIR"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
if [ "$FAILED_PASSES" -ne 0 ]; then
  echo "Error: $FAILED_PASSES pass(es) had failures; see $SUMMARY" >&2
  exit 1
fi
