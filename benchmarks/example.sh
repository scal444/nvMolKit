#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# End-to-end FIRE benchmarking demo:
#   1. Minimize an SDF with nvmolkit FIRE 2.0 (mass-weighted) and again without.
#   2. Minimize the same SDF with RDKit MMFF94 as a reference.
#   3. Compare the three runs' final energies side-by-side.
#
# Usage: ./example.sh path/to/conformers.sdf [output_dir]

set -euo pipefail

INPUT_SDF="${1:-}"
OUTPUT_DIR="${2:-./fire_example_output}"

if [ -z "$INPUT_SDF" ]; then
  echo "Usage: $0 <input.sdf> [output_dir]" >&2
  exit 2
fi
if [ ! -f "$INPUT_SDF" ]; then
  echo "Input SDF not found: $INPUT_SDF" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python "$SCRIPT_DIR/minimize_nvmolkit_conformers.py" \
  "$INPUT_SDF" \
  --output-prefix "$OUTPUT_DIR/nvmolkit_mass" \
  --mass-weighting

python "$SCRIPT_DIR/minimize_nvmolkit_conformers.py" \
  "$INPUT_SDF" \
  --output-prefix "$OUTPUT_DIR/nvmolkit_nomass"

python "$SCRIPT_DIR/minimize_rdkit_conformers.py" \
  "$INPUT_SDF" \
  --output-prefix "$OUTPUT_DIR/rdkit"

python "$SCRIPT_DIR/mmff_energy_comparison.py" \
  --output-dir "$OUTPUT_DIR/comparison" \
  --run rdkit="$OUTPUT_DIR/rdkit" \
  --run nvmolkit_mass="$OUTPUT_DIR/nvmolkit_mass" \
  --run nvmolkit_nomass="$OUTPUT_DIR/nvmolkit_nomass"

echo "Done. Outputs under $OUTPUT_DIR/"
