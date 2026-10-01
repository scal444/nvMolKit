# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical k-means index over cluster centroids as the BitBIRCH refinement candidate search.

Index: ``levels`` levels of cells over the cluster majority centroids. Level 0 runs k-means over
all centroids; each deeper level runs k-means inside every parent cell (centroids only compare
against their parent's sub-centers, via the gather/popcount kernel). Assignment is Tanimoto
argmax; centers are majority bits (ties set the bit); empty sub-cells are dropped. Fan-out per
level is ``ceil((K / leaf_size) ** (1 / levels))`` unless fixed with ``--fanout`` (then the
number of levels follows from K). Leaf cells larger than ``max_leaf`` are cut into consecutive
pieces (siblings under the same parent) to bound the padded candidate width.

Search: the index is a fixed-topology tree whose leaf entries are the clusters, so it reuses
``BeamSearch`` from ``bitbirch_beam.py`` with ``beam = p`` probes per level and cell summaries
recomputed each pass from the current cluster centroids (unweighted majority, like the flat IVF
coarse centers). Per-molecule cost is about levels x p x fan-out + p x leaf size.

Commands (outputs skipped when they exist):
  recall FPS ARCHIVE RUN --levels L... --probes P... --cache EXH.npz --output CSV
  refine FPS ARCHIVE RUN --levels L --probes P... --out-dir DIR
  cost   FPS ARCHIVE RUN --sizes N... --fanout F --probes P... --output CSV
"""

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitbirch_beam import BeamSearch, Tree, gather_tanimoto, load_run, pack_words, pass_inputs
from bitbirch_refinement import N_BITS, compact, dedup_step, popcounts, refine_step


def majority_centers(centroids, assign, n_cells, old):
    counts = torch.zeros((n_cells, N_BITS), dtype=torch.int32, device=centroids.device)
    for c in range(0, len(assign), 65_536):
        counts.index_add_(0, assign[c : c + 65_536], centroids[c : c + 65_536].to(torch.int32))
    n = torch.bincount(assign, minlength=n_cells)
    new = (counts >= (n // 2 + n % 2).unsqueeze(1)).to(torch.int8)
    return torch.where((n > 0)[:, None], new, old)


def build_hier(centroids, levels=None, fanout=None, leaf_size=100, iterations=5, max_leaf=None, seed=0):
    """Hierarchical k-means over int8 centroids (K, 2048). Returns ``Tree``-compatible arrays + seconds."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    device = centroids.device
    k = centroids.shape[0]
    target_cells = max(1.0, k / leaf_size)
    if fanout is None:
        fanout = max(2, math.ceil(target_cells ** (1 / levels)))
    else:
        levels = max(1, math.ceil(math.log(target_cells) / math.log(fanout) - 1e-9))
    max_leaf = max_leaf or 4 * leaf_size
    rng = np.random.default_rng(seed)
    words, pop = pack_words(centroids), popcounts(centroids)
    parent = np.zeros(k, dtype=np.int64)
    n_parent = 1
    node_sizes = []
    for level in range(levels):
        members = np.bincount(parent, minlength=n_parent)
        remaining = levels - level - 1
        n_sub = np.clip(np.ceil(members / (leaf_size * fanout**remaining)), 1, fanout).astype(np.int64)
        sub_start = np.concatenate([[0], np.cumsum(n_sub)[:-1]])
        order = np.lexsort((rng.random(k), parent))
        group_start = np.concatenate([[0], np.cumsum(members)[:-1]])
        rank = np.empty(k, dtype=np.int64)
        rank[order] = np.arange(k) - group_start[parent[order]]
        seeds = rank < n_sub[parent]
        n_cells = int(n_sub.sum())
        centers = torch.empty((n_cells, N_BITS), dtype=torch.int8, device=device)
        seed_ids = np.flatnonzero(seeds)
        centers[torch.from_numpy(sub_start[parent[seed_ids]] + rank[seed_ids]).to(device)] = centroids[
            torch.from_numpy(seed_ids).to(device)
        ]
        width = int(n_sub.max())
        span = torch.arange(width, device=device)
        start_t = torch.from_numpy(sub_start[parent]).to(device)
        count_t = torch.from_numpy(n_sub[parent]).to(device)
        cand = torch.where(span < count_t[:, None], start_t[:, None] + span, -1).to(torch.int32)
        assign = torch.empty(k, dtype=torch.int64, device=device)
        for step in range(iterations + 1):
            cw, cp = pack_words(centers), popcounts(centers)
            for c in range(0, k, 65_536):
                sim = gather_tanimoto(
                    words[c : c + 65_536], pop[c : c + 65_536], cw, cp, cand[c : c + 65_536].contiguous()
                )
                assign[c : c + 65_536] = cand[c : c + 65_536].gather(1, sim.argmax(1, keepdim=True)).squeeze(1).long()
            if step < iterations:
                centers = majority_centers(centroids, assign, n_cells, centers)
        assign_np = assign.cpu().numpy()
        sizes = np.bincount(assign_np, minlength=n_cells)
        if level == levels - 1:
            # Cut oversized leaf cells into consecutive pieces of <= max_leaf clusters.
            pieces = np.maximum(1, np.ceil(sizes / max_leaf)).astype(np.int64)
            order = np.argsort(assign_np, kind="stable")
            cell_start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
            within = np.empty(k, dtype=np.int64)
            within[order] = np.arange(k) - cell_start[assign_np[order]]
            piece_size = np.ceil(sizes / pieces).clip(min=1).astype(np.int64)
            piece_start = np.concatenate([[0], np.cumsum(pieces)[:-1]])
            assign_np = piece_start[assign_np] + within // piece_size[assign_np]
            cell_parent = np.repeat(np.repeat(np.arange(n_parent), n_sub), pieces)
            sizes = np.bincount(assign_np, minlength=int(pieces.sum()))
        else:
            cell_parent = np.repeat(np.arange(n_parent), n_sub)
        keep = sizes > 0
        renumber = np.cumsum(keep) - 1
        cell_parent = cell_parent[keep]
        node_sizes.append(np.bincount(cell_parent, minlength=n_parent))
        parent = renumber[assign_np]
        n_parent = int(keep.sum())
    node_sizes.append(np.bincount(parent, minlength=n_parent))
    torch.cuda.synchronize()
    out = {
        "leaf_cluster": np.argsort(parent, kind="stable"),
        "depth": levels + 1,
        "fanout": fanout,
        "levels": levels,
        "seconds_build": time.perf_counter() - t0,
        "max_leaf_cell": int(node_sizes[-1].max()),
        "mean_leaf_cell": float(node_sizes[-1].mean()),
    }
    for level, sizes in enumerate(node_sizes):
        out[f"node_sizes_{level}"] = sizes
    return out


def index_name(levels, fanout):
    return f"hier{levels}" if fanout is None else f"hierf{fanout}"


def cmd_recall(args, device):
    _, labels, fps = load_run(args)
    sizes, centroids, pop, own_s = pass_inputs(fps, labels, device)
    data = np.load(args.cache)  # exhaustive ground truth from bitbirch_beam.py recall
    exh_s, exh_i = data["s"], data["i"]
    movers = exh_s > own_s
    total_gain = (exh_s[movers] - own_s[movers]).sum()
    rows = []
    for levels in args.levels:
        arrays = build_hier(centroids, levels=levels)
        torch.cuda.empty_cache()
        tree = Tree(arrays, device)
        for probes in args.probes:
            search = BeamSearch(tree, probes, device, summary="centroids")
            torch.cuda.synchronize()
            start = time.perf_counter()
            s, i = search(fps, labels, centroids, pop, sizes)
            torch.cuda.synchronize()
            found = s > own_s
            rows.append(
                {
                    "run": args.run,
                    "search": f"hier{levels}",
                    "levels": levels,
                    "fanout": arrays["fanout"],
                    "mean_leaf_cell": arrays["mean_leaf_cell"],
                    "max_leaf_cell": arrays["max_leaf_cell"],
                    "build_seconds": arrays["seconds_build"],
                    "probes": probes,
                    "seconds": time.perf_counter() - start,
                    "exhaustive_movers": float(movers.mean()),
                    "recall_exact_best": float((i == exh_i)[movers].mean()),
                    "found_move": float(found[movers].mean()),
                    "gain_captured": float((s[movers & found] - own_s[movers & found]).sum() / total_gain),
                }
            )
            print(rows[-1], flush=True)
            torch.cuda.empty_cache()
    write(args.output, rows)


def cmd_refine(args, device):
    threshold, base, fps = load_run(args)
    method, batch, seed = re.fullmatch(r"(\w+?)_b(\d+)_s(\d+)", args.run).groups()
    levels = args.levels[0]
    tree = None
    for probes in args.probes:
        name = f"hier{levels}p{probes}"
        target = args.out_dir / f"{args.run}_{name}" / f"labels_threshold_{threshold}.npz"
        if target.exists():
            continue
        if tree is None:
            # Index built once on the base partition; cell summaries are refreshed every pass.
            _, centroids, _, _ = pass_inputs(fps, base, device)
            arrays = build_hier(centroids, levels=levels)
            del centroids
            torch.cuda.empty_cache()
            tree = Tree(arrays, device)
        search = BeamSearch(tree, probes, device, summary="centroids")
        labels, steps, out = base, [], {}
        for iteration in range(1, args.iterations + 1):
            labels, step = refine_step(fps, labels, threshold, device, search)
            steps.append(step)
            print(f"{name} iteration {iteration}: {step}", flush=True)
            if iteration == 1:
                out["r1"] = compact(labels)
        out["r3"] = compact(labels)
        out["r3dedup"], dedup = dedup_step(fps, out["r3"], threshold, device)
        target.parent.mkdir(parents=True, exist_ok=True)
        log = {
            "index_build_seconds": float(arrays["seconds_build"]),
            "fanout": int(arrays["fanout"]),
            "probes": probes,
        }
        (target.parent / f"steps_{threshold}.json").write_text(
            json.dumps({**log, "r": steps, "r3dedup": dedup}, indent=1)
        )
        partial = target.with_suffix(".partial.npz")
        np.savez_compressed(partial, **{f"{method}_r{v[1:]}{name}_b{batch}_s{seed}": lab for v, lab in out.items()})
        partial.rename(target)


def cmd_cost(args, device):
    rows = []
    for n in args.sizes:
        args.prefix = n
        _, labels, fps = load_run(args)
        sizes, centroids, pop, _ = pass_inputs(fps, labels, device)
        for fanout in args.fanouts:
            arrays = build_hier(centroids, fanout=fanout)
            torch.cuda.empty_cache()
            tree = Tree(arrays, device)
            for probes in args.probes:
                search = BeamSearch(tree, probes, device, summary="centroids")
                search(fps[:20_000], labels[:20_000], centroids, pop, sizes)
                torch.cuda.synchronize()
                start = time.perf_counter()
                search(fps, labels, centroids, pop, sizes)
                torch.cuda.synchronize()
                rows.append(
                    {
                        "molecules": n,
                        "clusters": len(sizes),
                        "fanout": fanout,
                        "levels": int(arrays["levels"]),
                        "max_leaf_cell": int(arrays["max_leaf_cell"]),
                        "probes": probes,
                        "search_seconds": time.perf_counter() - start,
                        "index_build_seconds": float(arrays["seconds_build"]),
                    }
                )
                print(rows[-1], flush=True)
                torch.cuda.empty_cache()
        del centroids, pop
        torch.cuda.empty_cache()
    write(args.output, rows)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["recall", "refine", "cost"])
    parser.add_argument("fingerprints")
    parser.add_argument("labels")
    parser.add_argument("run")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--probes", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--fanouts", type=int, nargs="+", default=[32])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--cache")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prefix", type=int)
    parser.add_argument("--sizes", type=int, nargs="+", default=[250_000, 500_000, 1_000_000, 2_409_133])
    args = parser.parse_args()
    device = torch.device("cuda")
    if args.output is not None and args.output.exists():
        print(f"{args.output}: already done")
        return
    {"recall": cmd_recall, "refine": cmd_refine, "cost": cmd_cost}[args.command](args, device)


if __name__ == "__main__":
    main()
