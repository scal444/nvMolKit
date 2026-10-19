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

"""Benchmark nvmolkit.clustering.kmeans (cuML) on packed fingerprints and score it in Tanimoto terms.

For each metric, reports fit/label time, and:
  - tanimoto_agreement: fraction of molecules whose cuML label equals the center with the highest
    continuous Tanimoto x.c / (|x| + |c|^2 - x.c). Measures how far cuML's Euclidean/cosine assignment
    is from a Tanimoto assignment with the same centers.
  - mean_tanimoto_to_center: mean continuous Tanimoto of each molecule to its assigned center,
    for the cuML labels and for the Tanimoto-reassigned labels ("reassign").
  - silhouette (Tanimoto distance, sampled rows) and ARI vs labels when --labels is given.

Input: .npy of uint8 rows packed with numpy.packbits(bitorder="little") (as ~/data/clustering/*),
or uint32/int32 rows already in nvMolKit's layout.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from nvmolkit.clustering import kmeans
from nvmolkit.fingerprints import unpack_fingerprint


def load_packed(path, limit, seed):
    array = np.load(path, mmap_mode="r")
    rows = np.arange(array.shape[0])
    if limit and limit < array.shape[0]:
        rows = np.sort(np.random.default_rng(seed).choice(array.shape[0], limit, replace=False))
    packed = np.ascontiguousarray(array[rows])
    if packed.dtype == np.uint8:
        packed = packed.view("<u4")
    return torch.from_numpy(packed.view(np.int32)).cuda(), rows


def tanimoto_to_centers(packed, centers, batch_size):
    """Return the highest-Tanimoto center and its Tanimoto per row, in batches."""
    center_sq = (centers * centers).sum(1)
    best_ids, best_vals = [], []
    for start in range(0, packed.shape[0], batch_size):
        bits = unpack_fingerprint(packed[start : start + batch_size]).float()
        dot = bits @ centers.T
        sim = dot / (bits.sum(1, keepdim=True) + center_sq - dot).clamp_min(1e-12)
        val, idx = sim.max(1)
        best_ids.append(idx)
        best_vals.append(val)
    return torch.cat(best_ids), torch.cat(best_vals)


def bit_means(packed, labels, sizes, k, batch_size):
    sums = torch.zeros((k, packed.shape[1] * 32), device=packed.device)
    for start in range(0, packed.shape[0], batch_size):
        bits = unpack_fingerprint(packed[start : start + batch_size]).float()
        sums.index_add_(0, labels[start : start + batch_size].long(), bits)
    return sums / sizes.clamp_min(1).unsqueeze(1).float()


def tanimoto_to_assigned(packed, centers, labels, batch_size):
    out = []
    for start in range(0, packed.shape[0], batch_size):
        bits = unpack_fingerprint(packed[start : start + batch_size]).float()
        c = centers[labels[start : start + batch_size].long()]
        dot = (bits * c).sum(1)
        out.append(dot / (bits.sum(1) + (c * c).sum(1) - dot).clamp_min(1e-12))
    return torch.cat(out)


def tanimoto_silhouette(packed, labels, sample, seed):
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    rows = torch.from_numpy(rng.choice(packed.shape[0], min(sample, packed.shape[0]), replace=False)).cuda()
    bits = unpack_fingerprint(packed[rows]).float()
    inter = bits @ bits.T
    counts = bits.sum(1)
    dist = 1 - inter / (counts[:, None] + counts[None, :] - inter).clamp_min(1e-12)
    dist.fill_diagonal_(0)
    lab = labels[rows].cpu().numpy()
    if len(np.unique(lab)) < 2:
        return float("nan")
    return float(silhouette_score(dist.cpu().numpy(), lab, metric="precomputed"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fingerprints", type=Path)
    parser.add_argument("--k", type=int, nargs="+", required=True)
    parser.add_argument("--metrics", nargs="+", default=["euclidean", "cosine"])
    parser.add_argument("--limit", type=int, default=0, help="random subset of rows (0 = all)")
    parser.add_argument("--fit-size", type=int, default=0, help="0 = fit on all rows")
    parser.add_argument("--n-init", type=int, default=1)
    parser.add_argument("--init", default="k-means||", choices=["k-means||", "k-means++", "random"])
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--labels", type=Path, help=".npy of int labels row-aligned with the full input")
    parser.add_argument("--silhouette-sample", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True, help="JSON lines output")
    args = parser.parse_args()

    packed, rows = load_packed(args.fingerprints, args.limit, args.seed)
    truth = np.load(args.labels)[rows] if args.labels else None
    print(f"loaded {packed.shape[0]} x {packed.shape[1] * 32} bits", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for k in args.k:
        for metric in args.metrics:
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = kmeans(
                packed,
                k,
                metric=metric,
                init=args.init,
                n_init=args.n_init,
                max_iter=args.max_iter,
                fit_size=args.fit_size or None,
                batch_size=args.batch_size,
                random_state=args.seed,
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start

            labels = result.labels.torch()
            centers = result.centers.torch()
            if metric == "cosine":
                # Centers live in normalized space; Tanimoto needs bit-space means.
                centers = bit_means(packed, labels, result.cluster_sizes.torch(), k, args.batch_size)
            best, best_sim = tanimoto_to_centers(packed, centers, args.batch_size)
            assigned_sim = tanimoto_to_assigned(packed, centers, labels, args.batch_size)
            sizes = result.cluster_sizes.numpy()
            record = {
                "dataset": str(args.fingerprints),
                "n": int(packed.shape[0]),
                "k": k,
                "metric": metric,
                "fit_size": args.fit_size or int(packed.shape[0]),
                "init": args.init,
                "n_init": args.n_init,
                "seconds": seconds,
                "n_iter": result.n_iter,
                "inertia": result.inertia,
                "empty_clusters": int((sizes == 0).sum()),
                "singletons": int((sizes == 1).sum()),
                "largest_cluster": int(sizes.max()),
                "tanimoto_agreement": float((best == labels.long()).float().mean()),
                "mean_tanimoto_to_center": float(assigned_sim.mean()),
                "mean_tanimoto_to_center_reassign": float(best_sim.mean()),
                "silhouette_tanimoto": tanimoto_silhouette(packed, labels, args.silhouette_sample, args.seed),
                "silhouette_tanimoto_reassign": tanimoto_silhouette(packed, best, args.silhouette_sample, args.seed),
            }
            if truth is not None:
                from sklearn.metrics import adjusted_rand_score

                record["ari"] = adjusted_rand_score(truth, labels.cpu().numpy())
                record["ari_reassign"] = adjusted_rand_score(truth, best.cpu().numpy())
            print(json.dumps(record), flush=True)
            with args.out.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            del result, labels, centers, best, best_sim, assigned_sim
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
