#!/usr/bin/env bash
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

# Staged GPU runs for nvmolkit.clustering.kmeans. Usage: run_gpu.sh {tests|landrum|chembl|enamine}
# Env: NVMOLKIT_ROOT (repo checkout, default: this script's repo), KMEANS_DATA (data dir),
#      KMEANS_OUT (results dir), CHEMBL34_FP (ChEMBL34 ecfp4_2048.npy),
#      ENAMINE_FP (Enamine 10M morgan2_1024.npy from prep_enamine.py).
# Cluster counts follow the literature: k=7 is the NVIDIA RAPIDS ChEMBL demo; ~42 molecules per
# cluster is Walters' FAISS run (k=50,000 on 2.1M). All ChEMBL/Enamine fits use every row.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${NVMOLKIT_ROOT:-$(cd "$HERE/../.." && pwd)}
DATA=${KMEANS_DATA:-$HOME/experiments/kmeans/data}
OUT=${KMEANS_OUT:-$HOME/experiments/kmeans/results}
CHEMBL=${CHEMBL34_FP:-$HOME/data/clustering/chembl34/ecfp4_2048.npy}
ENAMINE=${ENAMINE_FP:-$DATA/enamine10m/morgan2_1024.npy}
export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT"

case "${1:-}" in
  tests)
    cd "$ROOT" && timeout 120 python -m pytest -q nvmolkit/tests/test_clustering.py -k kmeans
    ;;
  landrum)
    # Labelled. Published ARI on the full 90,498 set: CLASSIX-T 0.130, DBSCAN 0.089, Butina 0.017.
    D=$DATA/landrum
    timeout 600 python "$HERE/bench_kmeans.py" "$D/morgan2_1024.npy" --labels "$D/labels.npy" \
      --k 35 100 184 500 --n-init 3 --seed 0 --out "$OUT/landrum_full.jsonl"
    timeout 600 python "$HERE/bench_kmeans.py" "$D/single_morgan2_1024.npy" --labels "$D/single_labels.npy" \
      --k 35 100 184 500 --n-init 3 --seed 0 --out "$OUT/landrum_single.jsonl"
    ;;
  chembl)
    # ChEMBL34 ECFP4 2048, 2,409,133 rows: k=7 (NVIDIA demo) and k=57,000 (~42 per cluster).
    timeout 1800 python "$HERE/bench_kmeans.py" "$CHEMBL" --k 7 --batch-size 16384 \
      --seed 0 --out "$OUT/chembl34.jsonl"
    timeout 5400 python "$HERE/bench_kmeans.py" "$CHEMBL" --k 57000 --batch-size 4096 \
      --seed 0 --out "$OUT/chembl34.jsonl"
    ;;
  enamine)
    # Enamine REAL 10M, Morgan r2 1024. k=7, then k=250,000 (~40 per cluster). At that k, k-means|| and
    # k-means++ init are impractical (candidate memory / 250k sequential steps), so use random init and
    # FAISS-style 20 Lloyd iterations (as in Walters' 2.1M run); cosine only, since each pass is ~5 PFLOP.
    timeout 1800 python "$HERE/bench_kmeans.py" "$ENAMINE" --k 7 --batch-size 16384 \
      --seed 0 --out "$OUT/enamine10m.jsonl"
    timeout 10800 python "$HERE/bench_kmeans.py" "$ENAMINE" --k 250000 --metrics cosine --init random \
      --max-iter 20 --batch-size 2048 --seed 0 --out "$OUT/enamine10m.jsonl"
    ;;
  *)
    echo "usage: $0 {tests|landrum|chembl|enamine}" >&2
    exit 2
    ;;
esac
