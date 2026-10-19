# k-means via cuML: handoff

Branch `feat/kmeans-cuml`, based on upstream `main` (8907b74b). Context: issue
[#286](https://github.com/NVIDIA-BioNeMo/nvMolKit/issues/286) (clustering methods used on molecular
fingerprints). Requirement: use cuML for k-means, or show where it can't be used.

## What is on the branch

- `nvmolkit/clustering.py`: `kmeans(x, n_clusters, *, metric, init, n_init, max_iter, tol, fit_size,
  batch_size, random_state, stream) -> KMeansResult` (labels, centers, cluster_sizes, inertia, n_iter).
  - Packed fingerprints are unpacked to float32 bits (LSB-first, as `unpack_fingerprint`) and passed
    to `cuml.cluster.KMeans`.
  - `metric="cosine"` L2-normalizes rows first.
  - `fit_size` fits on a random subsample, then labels all rows with chunked `predict`.
  - `batch_size` bounds unpacking and is passed to cuML as `max_samples_per_batch`.
  - cuML is imported lazily (`is_cuml_available()`, `CUML_INSTALL_HINT`), like optuna in autotune.
- `pyproject.toml`: optional extra `kmeans = ["cuml-cu12"]`.
- `docs/clustering.rst`: k-means section.
- `nvmolkit/tests/test_clustering.py`: 17 k-means tests. They skip without cuML, except the
  install-hint test.
- `benchmarks/kmeans/`: `prep_landrum.py`, `bench_kmeans.py`, `run_gpu.sh`, this file.

Status: all 17 k-means tests pass on an RTX 1000 Ada (sm_89, 6 GB) with cuML 26.10 and torch
2.8.0+cu128. The full clustering suite and the C++ tests were not run.

## Environment

Install cuML into a copy of the dev env. A venv layered over a conda env (`--system-site-packages`)
does not work: the RAPIDS wheels find NCCL, nvrtc and the other CUDA libraries by RUNPATH inside their
own site-packages, so `import cuml` fails with `libcuml.so: cannot open shared object file`.

```bash
conda create -n rdcu_kmeans --clone rdcu_dev
conda activate rdcu_kmeans
pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12   # 26.10.0 at time of writing
```

`cuml-cu12` upgrades `nvidia-cublas-cu12` (12.9), `nvidia-nvjitlink-cu12` (12.9) and
`nvidia-cusparse-cu12` past torch 2.8 cu128's pins. pip warns, but torch and the tests ran fine.

Build the extension as usual. Set `CMAKE_CUDA_ARCHITECTURES` to the target GPU: 90 for H100/H200,
100 for B200. If the `.so` files land in `build/nvmolkit/`, symlink them into `nvmolkit/`.

## Data

- **Landrum BL sets** (labelled; 90,498 rows after pKi > 4; 35 targets).
  1. Download
     `https://raw.githubusercontent.com/greglandrum/rdkit_blog/master/data/BLSets_actives.txt` into
     `$KMEANS_DATA/landrum/`.
  2. Run `python benchmarks/kmeans/prep_landrum.py $KMEANS_DATA/landrum`.
  3. This writes `morgan2_1024.npy` and `labels.npy`, plus a single-target deduplicated subset
     (`single_*`, 40,980 rows). The single-target subset matters: 14,533 SMILES appear under more than
     one target, so ARI on the full set cannot reach 1.
- **ChEMBL34 ECFP4 2048** (2,409,133 rows, 617 MB).
  - Copy `~/data/clustering/chembl34/ecfp4_2048.npy` from the dev box, or regenerate with
    `~/data/clustering/scripts/prepare_fingerprints.py` (see `~/data/clustering/chembl34/README.md`).
  - Set `CHEMBL34_FP` to its path.

## Running

```bash
export KMEANS_DATA=... KMEANS_OUT=... CHEMBL34_FP=...
benchmarks/kmeans/run_gpu.sh tests       # seconds
benchmarks/kmeans/run_gpu.sh landrum     # ~1 min
benchmarks/kmeans/run_gpu.sh chembl      # k=7, 1000 on 2.4M; ~2 min on sm_89
benchmarks/kmeans/run_gpu.sh chembl10k   # k=10,000 with k-means||, random, k-means++
```

`bench_kmeans.py` writes one JSON line per (k, metric). Fields:

- `seconds`
- `n_iter`, `inertia`
- `empty_clusters`, `singletons`, `largest_cluster`
- `tanimoto_agreement`: fraction of rows whose cuML label is also their highest continuous-Tanimoto
  center.
- `mean_tanimoto_to_center`, and the same after Tanimoto reassignment (`_reassign`).
- `silhouette_tanimoto` on 10k sampled rows, with and without reassignment.
- `ari` / `ari_reassign` when labels are given.

## Results so far (RTX 1000 Ada, 6 GB)

Landrum full set (n_init=3):

| k | metric | s | ARI | ARI reassign | agreement | T to center | silhouette |
|---|---|---|---|---|---|---|---|
| 35 | euclidean | 2.4 | 0.164 | 0.180 | 0.742 | 0.389 | 0.053 |
| 35 | cosine | 1.2 | 0.157 | 0.164 | 0.827 | 0.397 | 0.069 |
| 184 | euclidean | 2.6 | 0.177 | 0.164 | 0.855 | 0.514 | 0.115 |
| 184 | cosine | 2.6 | 0.113 | 0.120 | 0.893 | 0.523 | 0.135 |
| 500 | euclidean | 6.5 | 0.075 | 0.072 | 0.894 | 0.600 | 0.156 |
| 500 | cosine | 6.8 | 0.061 | 0.062 | 0.938 | 0.610 | 0.180 |

Published ARI on this set: CLASSIX-T 0.130 (184 clusters), DBSCAN 0.089, Butina 0.017. On the
single-target subset, cosine at k=35 reaches ARI 0.243 (0.272 after reassignment).

ChEMBL34, 2.4M rows, `fit_size=256000`:

| k | init | metric | s | agreement | T to center (reassigned) |
|---|---|---|---|---|---|
| 7 | k-means\|\| | euclidean / cosine | 5.4 / 3.9 | 0.705 / 0.793 | 0.240 (0.247) / 0.244 (0.247) |
| 1000 | k-means\|\| | euclidean / cosine | 31.5 / 29.5 | 0.734 / 0.801 | 0.394 (0.402) / 0.406 (0.411) |
| 10000 | k-means\|\| | both | OOM | | |
| 10000 | k-means++ | euclidean / cosine | 586 / 591 | 0.800 / 0.831 | 0.517 (0.523) / 0.530 (0.535) |
| 10000 | random | cosine | 69 | 0.832 | 0.521 (0.526) |

Raw JSON lines are in `~/experiments/kmeans/results/` on the dev box.

## Findings: cuML as-is

1. **No Tanimoto.** cuML KMeans hard-codes L2: `cuml/cluster/kmeans.pyx:52`, and cuVS
   `kmeans_fit`/`kmeans_predict` reject other metrics. There is no custom-distance hook. Cosine
   (normalized rows) is the closest available metric.
   - Measured cost is small. 70–94% of rows already sit at their best-Tanimoto center, and Tanimoto
     reassignment raises mean Tanimoto-to-center by only 0.003–0.008.
   - ARI changes either way after reassignment.
   - Cosine beats euclidean on Tanimoto cohesion and silhouette at every k.
2. **32x memory from unpacking.** cuML needs dense float32/64 and rejects sparse. Mitigated by
   `fit_size` plus chunked predict. Only cuVS C++ balanced k-means takes uint8 (4x smaller); that
   needs a direct libcuvs link.
3. **The k-means|| init has no memory bound in cuML's API.**
   - It scores all points against all candidates (about 2k per round, up to 8 rounds). cuVS's
     `batch_centroids` would tile this, but cuML does not expose it.
   - At k=10,000 the tile was 1.31 GB on top of 2.1 GB of fit features, and that OOMed on 6 GB.
   - `k-means++` avoids the OOM but is sequential: about 520 of its 590 s.
   - `random` is 8.5x faster with slightly worse cohesion.
   - On a large GPU the default may just fit. Measure with `chembl10k`.
4. **No MiniBatchKMeans** in cuML or cuVS. `fit_size` covers the use case.
5. **Integration.**
   - cuML runs on its own `cudaStreamPerThread`. The wrapper synchronizes the caller's stream first.
   - cuML has RMM and CuPy memory pools separate from torch's caching allocator.
   - Empty clusters keep their stale center, so `cluster_sizes` can contain zeros.
   - cuML is a heavy dependency (pulls cudf, cupy; about 1 GB; Linux only).

## Open decisions / next steps

- Default `metric`: the data favors `cosine`. The code currently defaults to `"euclidean"`.
- Pre-check the k-means|| tile size against free memory, and raise with advice to pass `init`
  instead of surfacing an RMM OOM.
- `init="maxmin"`: seed cuML with Tanimoto MaxMin picks from `fused_maxmin`. This bounds memory and
  avoids sequential k-means++. Quality vs k-means++ is unmeasured.
- Optional Tanimoto reassignment pass after cuML. Low gain so far; leave out unless
  Tanimoto-consistent labels are required.
- Not yet measured: Enamine 10M, multi-GPU (`cuml.dask`), and an FAISS baseline (`niter=20`, random
  init).
