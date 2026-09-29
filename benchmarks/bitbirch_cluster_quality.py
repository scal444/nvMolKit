# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intrinsic quality metrics for BitBIRCH partitions of binary fingerprints.

Reads the label archives written by ``bitbirch_order_sensitivity.py --save-labels``
(one ``labels_threshold_<t>.npz`` per threshold, one array per run) and reports,
per run, metrics that do not depend on a reference partition:

* Shape: cluster count, singleton fraction, cluster-size quantiles, largest cluster.
* Compactness: per-cluster iSIM Tanimoto (mean pairwise similarity) quantiles and
  the number of multi-member clusters below the merge threshold, which the
  diameter criterion should make zero.
* Member fit: Tanimoto of every member to its cluster's majority centroid
  (ties set the bit), and the fraction of members below the threshold.
* Misassignment: for a random sample of members of multi-member clusters, the
  fraction whose most similar centroid belongs to a different cluster.
* Chemistry: Murcko-scaffold purity (share of members carrying the cluster's
  most common scaffold) and unique scaffolds per cluster, when scaffold IDs
  are given.
* Representatives: iSIM of the set of cluster medoids (lower is more diverse).

Fingerprints are the ``uint8`` packed arrays from ``~/data/clustering``
(``numpy.packbits(bitorder="little")``). Computation runs on the GPU in chunks
of whole clusters.
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch


def isim_from_counts(counts, sizes):
    c = counts.double()
    n = sizes.double().unsqueeze(1)
    common = (c * (c - 1) / 2).sum(1)
    total = common + (c * (n - c)).sum(1)
    return torch.where(total > 0, common / total.clamp_min(1e-300), torch.ones_like(total))


def unpack(packed, shifts):
    return ((packed.unsqueeze(-1) >> shifts) & 1).reshape(packed.shape[0], -1)


def pack(bits, weights):
    return (bits.reshape(bits.shape[0], -1, 8).to(torch.int32) * weights).sum(-1).to(torch.uint8)


def chunk_bounds(sizes_sorted, target):
    """Split clusters (already in label order) into chunks of about ``target`` molecules."""
    bounds, begin, total = [], 0, 0
    for index, size in enumerate(sizes_sorted):
        total += size
        if total >= target:
            bounds.append((begin, index + 1))
            begin, total = index + 1, 0
    if begin < len(sizes_sorted):
        bounds.append((begin, len(sizes_sorted)))
    return bounds


def quality(fingerprints, labels, threshold, scaffolds, sample, device, chunk_molecules):
    n_molecules, n_bytes = fingerprints.shape
    n_bits = n_bytes * 8
    shifts = torch.arange(8, device=device, dtype=torch.uint8)
    weights = (1 << torch.arange(8, device=device, dtype=torch.int32)).view(1, 1, 8)

    sizes = np.bincount(labels)
    n_clusters = len(sizes)
    order = np.argsort(labels, kind="stable")
    offsets = np.concatenate([[0], np.cumsum(sizes)])

    member_similarity = np.ones(n_molecules, dtype=np.float32)
    cluster_isim = np.ones(n_clusters, dtype=np.float64)
    medoids = np.empty(n_clusters, dtype=np.int64)
    centroids = np.empty((n_clusters, n_bytes), dtype=np.uint8)

    for first, last in chunk_bounds(sizes, chunk_molecules):
        rows = order[offsets[first] : offsets[last]]
        chunk_sizes = torch.from_numpy(sizes[first:last]).to(device)
        local = torch.repeat_interleave(torch.arange(last - first, device=device), chunk_sizes)
        bits = unpack(torch.from_numpy(fingerprints[rows]).to(device), shifts)
        counts = torch.zeros((last - first, n_bits), dtype=torch.int32, device=device)
        counts.index_add_(0, local, bits.to(torch.int32))
        cluster_isim[first:last] = isim_from_counts(counts, chunk_sizes).cpu().numpy()
        majority = counts >= (chunk_sizes // 2 + chunk_sizes % 2).unsqueeze(1)
        centroids[first:last] = pack(majority.to(torch.uint8), weights).cpu().numpy()
        centroid_bits = majority.to(torch.uint8)[local]
        inter = (bits & centroid_bits).sum(1, dtype=torch.int32)
        union = (bits | centroid_bits).sum(1, dtype=torch.int32)
        similarity = torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(inter, dtype=torch.float32))
        member_similarity[rows] = similarity.cpu().numpy()
        best = torch.full((last - first,), -1.0, device=device).scatter_reduce(0, local, similarity, "amax")
        position = torch.arange(len(rows), device=device)
        candidates = torch.where(similarity == best[local], position, torch.full_like(position, len(rows)))
        first_best = torch.full((last - first,), len(rows), device=device).scatter_reduce(0, local, candidates, "amin")
        medoids[first:last] = rows[first_best.cpu().numpy()]

    multi = sizes >= 2
    in_multi = multi[labels]

    # Misassignment against every centroid, on a sample of multi-member cluster members.
    # Blocks of 512 queries x 32,768 centroids keep temporaries near 64 MiB each.
    del bits, counts, majority, centroid_bits
    torch.cuda.empty_cache()
    rng = np.random.default_rng(0)
    pool = np.flatnonzero(in_multi)
    queries = np.sort(rng.choice(pool, size=min(sample, len(pool)), replace=False)) if len(pool) else pool
    best_other = torch.full((len(queries),), -1.0, device=device)
    own = torch.from_numpy(labels[queries].astype(np.int64)).to(device)
    query_bits = unpack(torch.from_numpy(fingerprints[queries]).to(device), shifts).half()
    query_pop = query_bits.float().sum(1)
    for begin in range(0, n_clusters, 32_768):
        end = min(begin + 32_768, n_clusters)
        centroid_bits = unpack(torch.from_numpy(centroids[begin:end]).to(device), shifts).half()
        centroid_pop = centroid_bits.float().sum(1)
        for q in range(0, len(queries), 512):
            inter = (query_bits[q : q + 512] @ centroid_bits.T).float()
            union = query_pop[q : q + 512, None] + centroid_pop[None, :] - inter
            tanimoto = torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(inter))
            mine = own[q : q + 512, None] == torch.arange(begin, end, device=device)[None, :]
            tanimoto[mine] = -1.0
            best_other[q : q + 512] = torch.maximum(best_other[q : q + 512], tanimoto.max(1).values)
    own_similarity = torch.from_numpy(member_similarity[queries]).to(device)
    misassigned = (best_other > own_similarity).cpu().numpy()
    gap = (best_other - own_similarity).cpu().numpy()

    # Representatives: iSIM over one medoid per cluster.
    medoid_counts = torch.zeros(n_bits, dtype=torch.int64, device=device)
    for begin in range(0, n_clusters, 32_768):
        medoid_bits = unpack(torch.from_numpy(fingerprints[medoids[begin : begin + 32_768]]).to(device), shifts)
        medoid_counts += medoid_bits.sum(0, dtype=torch.int64)
    medoid_isim = float(isim_from_counts(medoid_counts[None, :], torch.tensor([n_clusters], device=device))[0])

    multi_isim = cluster_isim[multi]
    multi_sizes = sizes[multi]
    members = member_similarity[in_multi]
    result = {
        "clusters": n_clusters,
        "multi_member_clusters": int(multi.sum()),
        "singleton_fraction": float((sizes == 1).sum() / n_molecules),
        "size_p50_multi": float(np.percentile(multi_sizes, 50)) if multi.any() else 0.0,
        "size_p90_multi": float(np.percentile(multi_sizes, 90)) if multi.any() else 0.0,
        "size_p99_multi": float(np.percentile(multi_sizes, 99)) if multi.any() else 0.0,
        "largest_cluster": int(sizes.max()),
        "fraction_in_clusters_ge10": float(sizes[sizes >= 10].sum() / n_molecules),
        "isim_p10": float(np.percentile(multi_isim, 10)),
        "isim_p50": float(np.percentile(multi_isim, 50)),
        "isim_p90": float(np.percentile(multi_isim, 90)),
        "isim_size_weighted": float((multi_isim * multi_sizes).sum() / multi_sizes.sum()),
        "isim_below_threshold": int((multi_isim < threshold - 1e-12).sum()),
        "member_similarity_mean": float(members.mean()),
        "member_similarity_p10": float(np.percentile(members, 10)),
        "members_below_threshold": float((members < threshold).mean()),
        "misassigned_fraction": float(misassigned.mean()) if len(queries) else 0.0,
        "misassigned_mean_gap": float(gap[misassigned].mean()) if misassigned.any() else 0.0,
        "medoid_set_isim": medoid_isim,
    }
    if scaffolds is not None:
        valid = in_multi & (scaffolds >= 0)
        key = labels[valid].astype(np.int64) * (int(scaffolds.max()) + 1) + scaffolds[valid]
        pairs, pair_counts = np.unique(key, return_counts=True)
        pair_labels = pairs // (int(scaffolds.max()) + 1)
        starts = np.flatnonzero(np.r_[True, pair_labels[1:] != pair_labels[:-1]])
        modal = np.maximum.reduceat(pair_counts, starts)
        per_cluster_members = np.add.reduceat(pair_counts, starts)
        result["scaffold_purity_weighted"] = float(modal.sum() / per_cluster_members.sum())
        result["scaffold_purity_median"] = float(np.median(modal / per_cluster_members))
        result["unique_scaffolds_per_cluster"] = float(np.diff(np.r_[starts, len(pairs)]).mean())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fingerprints", help="packed uint8 .npy fingerprints, shape (N, bits/8)")
    parser.add_argument("labels", nargs="+", type=Path, help="labels_threshold_<t>.npz archives")
    parser.add_argument("--scaffolds", help="int32 .npy of per-molecule scaffold IDs (-1 = unknown)")
    parser.add_argument("--misassignment-sample", type=int, default=20_000)
    parser.add_argument("--chunk-molecules", type=int, default=50_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    fingerprints = np.load(args.fingerprints)
    scaffolds = np.load(args.scaffolds) if args.scaffolds else None
    rows = []
    for archive_path in args.labels:
        threshold = float(re.search(r"labels_threshold_([0-9.]+)\.npz", archive_path.name).group(1))
        archive = np.load(archive_path)
        for name in sorted(archive.files):
            method, batch, seed = re.fullmatch(r"(\w+?)_b(\d+)_s(\d+)", name).groups()
            labels = archive[name]
            metrics = quality(
                fingerprints[: len(labels)],
                labels,
                threshold,
                None if scaffolds is None else scaffolds[: len(labels)],
                args.misassignment_sample,
                device,
                args.chunk_molecules,
            )
            rows.append(
                {"threshold": threshold, "method": method, "batch_size": int(batch), "seed": int(seed), **metrics}
            )
            print(f"threshold {threshold} {name}: done", flush=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
