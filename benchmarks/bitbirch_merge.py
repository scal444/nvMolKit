# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guarded cluster merge after BitBIRCH refinement, to restore separation between adjacent tight clusters.

One merge round:
1. Per-cluster linear sums (host uint16) and majority centroids (ties set the bit).
2. Candidate partners: for every cluster, the ``k`` most similar other centroids from the flat IVF search
   used for refinement (cells = sqrt(K), 16 probes; for k > 1, the best hit of each of the k best probed
   cells). Cost ~ K * sqrt(K) <= N * sqrt(K); no all-pairs work.
3. For each unordered candidate pair, iSIM of the union from the summed linear sums (BitBIRCH's own
   diameter rule). Rule ``d``: eligible if iSIM(union) >= threshold + delta. Rule ``e`` (tolerance, like
   bblean's tolerance-diameter merge): eligible if iSIM(union) >= max(threshold, size-weighted iSIM of the
   pair's multi-member parts - delta); singletons need only the threshold.
4. Selection: ``mutual`` = only mutual nearest neighbours (k = 1; disjoint by construction); ``greedy`` =
   pairs sorted by union iSIM (descending, ties by ids), accepted when neither cluster is already used in
   this round. Every cluster merges at most once per round, so each accepted union's iSIM is exactly the
   checked value and every multi-member cluster keeps iSIM >= threshold.
Rounds repeat until nothing merges (``c``) or once (``1``).

Variant tag: m{mut|gr}{k}{d|e}{delta*100:02d}{1|c}, e.g. mgr4e02c. Bases: ``none`` (unrefined), ``r1`` (one
guarded IVF sqrt(K) x 16 refinement pass), ``r3d`` (three passes + identical-centroid dedup).

usage: bitbirch_merge.py FPS ARCHIVE RUN --variants TAG... --bases none r1 r3d --out-dir DIR
Writes DIR/<run>_<base>/labels_threshold_<t>.npz (keys <method>_<base>[_<tag>]_b<batch>_s<seed>) and
merge_<t>.json; existing outputs are skipped.
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitbirch_beam import cluster_sums, load_run
from bitbirch_cluster_quality import isim_from_counts
from bitbirch_refinement import (
    N_BITS,
    best_other_centroid_ivf,
    compact,
    dedup_step,
    popcounts,
    refine_step,
)

TAG = re.compile(r"m(mut|gr)(\d+)([de])(\d\d)([1c])")


def parse_tag(tag):
    mode, k, rule, delta, rounds = TAG.fullmatch(tag).groups()
    return mode, int(k), rule, int(delta) / 100, rounds


def centroids_from_sums(sums, sizes, device, chunk=65_536):
    """int8 (K, 2048) majority centroids on the GPU and packed uint8 (K, 256) on the host."""
    k = len(sizes)
    bits = torch.empty((k, N_BITS), dtype=torch.int8, device=device)
    packed = np.empty((k, N_BITS // 8), dtype=np.uint8)
    weights = (1 << torch.arange(8, device=device, dtype=torch.int32)).view(1, 1, 8)
    for c in range(0, k, chunk):
        s = torch.from_numpy(sums[c : c + chunk].astype(np.int32)).to(device)
        n = torch.from_numpy(sizes[c : c + chunk]).to(device)
        block = (s >= (n // 2 + n % 2).unsqueeze(1)).to(torch.int8)
        bits[c : c + chunk] = block
        packed[c : c + chunk] = (
            (block.reshape(-1, N_BITS // 8, 8).to(torch.int32) * weights).sum(-1).to(torch.uint8).cpu().numpy()
        )
    return bits, packed


def union_isim(sums, sizes, a, b, device, chunk=16_384):
    out = np.empty(len(a), dtype=np.float64)
    for c in range(0, len(a), chunk):
        aa, bb = a[c : c + chunk], b[c : c + chunk]
        counts = torch.from_numpy(sums[aa].astype(np.int32) + sums[bb].astype(np.int32)).to(device)
        n = torch.from_numpy(sizes[aa] + sizes[bb]).to(device)
        out[c : c + chunk] = isim_from_counts(counts, n).cpu().numpy()
    return out


def merge_round(fps, labels, threshold, mode, k, rule, delta, device):
    t0 = time.perf_counter()
    sums, sizes = cluster_sums(fps, labels, device)
    n_clusters = len(sizes)
    bits, packed = centroids_from_sums(sums, sizes, device)
    pop = popcounts(bits)
    cells = max(16, round(math.sqrt(n_clusters)))
    s, i = best_other_centroid_ivf(
        packed, np.arange(n_clusters), bits, pop, device, n_cells=cells, probes=16, query_chunk=32_768, topk=k
    )
    del bits, pop
    torch.cuda.empty_cache()
    t_search = time.perf_counter() - t0
    s, i = s.reshape(n_clusters, -1), i.reshape(n_clusters, -1)
    src = np.repeat(np.arange(n_clusters), i.shape[1])
    dst = i.ravel()
    ok = (dst >= 0) & (dst != src)
    if mode == "mut":
        nearest = i[:, 0]
        ok &= nearest[np.clip(dst, 0, None)] == src
    a, b = np.minimum(src[ok], dst[ok]), np.maximum(src[ok], dst[ok])
    pairs = np.unique(np.stack([a, b], 1), axis=0) if len(a) else np.empty((0, 2), dtype=np.int64)
    isim = union_isim(sums, sizes, pairs[:, 0], pairs[:, 1], device) if len(pairs) else np.empty(0)
    if rule == "d":
        floor = np.full(len(pairs), threshold + delta)
    else:
        # Tolerance rule (bblean tolerance-diameter style): the union may lose at most ``delta`` iSIM against
        # the size-weighted iSIM of its multi-member parts; singletons only need the threshold.
        own = np.empty(n_clusters)
        for c in range(0, n_clusters, 65_536):
            counts = torch.from_numpy(sums[c : c + 65_536].astype(np.int32)).to(device)
            own[c : c + 65_536] = (
                isim_from_counts(counts, torch.from_numpy(sizes[c : c + 65_536]).to(device)).cpu().numpy()
            )
        weight = np.where(sizes >= 2, sizes, 0).astype(np.float64)
        wa, wb = weight[pairs[:, 0]], weight[pairs[:, 1]]
        parent = np.where(wa + wb > 0, (wa * own[pairs[:, 0]] + wb * own[pairs[:, 1]]) / np.maximum(wa + wb, 1), 0.0)
        floor = np.maximum(threshold, parent - delta)
    eligible = isim >= floor - 1e-12
    pairs, isim = pairs[eligible], isim[eligible]
    order = np.lexsort((pairs[:, 1], pairs[:, 0], -isim))
    used = np.zeros(n_clusters, dtype=bool)
    target = np.arange(n_clusters)
    accepted = 0
    for a, b in pairs[order]:  # greedy matching; mutual pairs are already disjoint
        if used[a] or used[b]:
            continue
        used[a] = used[b] = True
        target[b] = a
        accepted += 1
    stats = {
        "clusters": n_clusters,
        "candidate_pairs": int(eligible.size),
        "eligible_pairs": int(eligible.sum()),
        "merged_pairs": accepted,
        "seconds_search": t_search,
        "seconds_total": time.perf_counter() - t0,
    }
    return compact(target[labels]), stats


def merge(fps, labels, threshold, tag, device, max_rounds=10):
    mode, k, rule, delta, rounds = parse_tag(tag)
    log = []
    for _ in range(1 if rounds == "1" else max_rounds):
        labels, stats = merge_round(fps, labels, threshold, mode, k, rule, delta, device)
        log.append(stats)
        print(f"  {tag} round {len(log)}: {stats}", flush=True)
        if stats["merged_pairs"] == 0:
            break
    return labels, log


def make_bases(fps, base, threshold, device, wanted):
    out, log = {}, {}
    if "none" in wanted:
        out["none"] = base
    if "r1" in wanted or "r3d" in wanted:
        cells = max(16, round(math.sqrt(int(base.max()) + 1)))
        labels, steps = base, []
        for iteration in range(1, 4 if "r3d" in wanted else 2):
            labels, step = refine_step(fps, labels, threshold, device, "ivf", (cells, 16))
            steps.append(step)
            if iteration == 1:
                out["r1"] = compact(labels)
        log["refine"] = steps
        if "r3d" in wanted:
            out["r3d"], log["dedup"] = dedup_step(fps, compact(labels), threshold, device)
    return {b: out[b] for b in wanted}, log


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fingerprints")
    parser.add_argument("labels")
    parser.add_argument("run")
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--bases", nargs="+", default=["none", "r1", "r3d"])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    threshold, base, fps = load_run(args)
    method, batch, seed = re.fullmatch(r"(\w+?)_b(\d+)_s(\d+)", args.run).groups()
    todo = [
        b for b in args.bases if not (args.out_dir / f"{args.run}_{b}" / f"labels_threshold_{threshold}.npz").exists()
    ]
    if not todo:
        return
    t0 = time.perf_counter()
    bases, refine_log = make_bases(fps, base, threshold, device, todo)
    refine_seconds = time.perf_counter() - t0
    for name, labels in bases.items():
        target = args.out_dir / f"{args.run}_{name}" / f"labels_threshold_{threshold}.npz"
        out = {f"{method}_{name}_b{batch}_s{seed}": labels}
        logs = {"refine_seconds_all_bases": refine_seconds, "refine": refine_log}
        for tag in args.variants:
            print(f"threshold {threshold} {args.run} base {name} {tag}", flush=True)
            merged, log = merge(fps, labels, threshold, tag, device)
            out[f"{method}_{name}_{tag}_b{batch}_s{seed}"] = merged
            logs[tag] = log
        target.parent.mkdir(parents=True, exist_ok=True)
        (target.parent / f"merge_{threshold}.json").write_text(json.dumps(logs, indent=1))
        partial = target.with_suffix(".partial.npz")
        np.savez_compressed(partial, **out)
        partial.rename(target)


if __name__ == "__main__":
    main()
