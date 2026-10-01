# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flat IVF refinement search with the coarse cell count scaled to the number of clusters K.

Per molecule the IVF search costs about ``cells`` (coarse scan) + ``probes * K / cells`` (candidate
scan), minimized at ``cells ~ sqrt(probes * K)``. Configurations are ``cells = round(c * sqrt(K))``
for a multiplier ``c`` and ``probes`` per molecule; ``fixed:<cells>`` keeps the cell count constant.
The coarse quantizer is ``build_coarse`` from ``bitbirch_refinement.py`` (majority-centroid k-means
over the cluster centroids, 5 iterations).

Commands (outputs skipped when they exist):
  recall FPS ARCHIVE RUN --configs C:P... --cache EXH.npz --output CSV
  refine FPS ARCHIVE RUN --configs C:P... --out-dir DIR      (3 guarded iterations + dedup)
  cost   FPS ARCHIVE RUN --configs C:P... --sizes N... --output CSV [--cluster]
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
from bitbirch_beam import load_run, pass_inputs
from bitbirch_refinement import best_other_centroid_ivf, build_coarse, compact, dedup_step, refine_step


def parse(config):
    scale, probes = config.split(":")
    return scale, int(probes)


def cells_for(scale, k):
    if scale.startswith("fixed"):
        return int(scale.split("fixed")[1])
    return max(16, round(float(scale) * math.sqrt(k)))


def name_of(scale, probes):
    return f"ivf{scale}x{probes}"


def synced():
    torch.cuda.synchronize()
    return time.perf_counter()


def cmd_recall(args, device):
    _, labels, fps = load_run(args)
    _, centroids, pop, own_s = pass_inputs(fps, labels, device)
    data = np.load(args.cache)
    exh_s, exh_i = data["s"], data["i"]
    movers = exh_s > own_s
    total_gain = (exh_s[movers] - own_s[movers]).sum()
    k = centroids.shape[0]
    rows, built = [], {}
    best_other_centroid_ivf(fps[:20_000], labels[:20_000], centroids, pop, device, n_cells=256, probes=4)
    for config in args.configs:
        scale, probes = parse(config)
        cells = cells_for(scale, k)
        if cells not in built:
            start = synced()
            built.clear()
            torch.cuda.empty_cache()
            built[cells] = (build_coarse(centroids, pop, cells, device), synced() - start)
        index, t_build = built[cells]
        start = synced()
        s, i = best_other_centroid_ivf(fps, labels, centroids, pop, device, probes=probes, index=index)
        t_search = synced() - start
        found = s > own_s
        rows.append(
            {
                "run": args.run,
                "scale": scale,
                "cells": cells,
                "probes": probes,
                "clusters": k,
                "build_seconds": t_build,
                "search_seconds": t_search,
                "recall_exact_best": float((i == exh_i)[movers].mean()),
                "found_move": float(found[movers].mean()),
                "gain_captured": float((s[movers & found] - own_s[movers & found]).sum() / total_gain),
            }
        )
        print(rows[-1], flush=True)
    write(args.output, rows)


def cmd_refine(args, device):
    threshold, base, fps = load_run(args)
    method, batch, seed = re.fullmatch(r"(\w+?)_b(\d+)_s(\d+)", args.run).groups()
    k = int(base.max()) + 1
    for config in args.configs:
        scale, probes = parse(config)
        cells = cells_for(scale, k)
        name = name_of(scale, probes).replace(".", "p")
        target = args.out_dir / f"{args.run}_{name}" / f"labels_threshold_{threshold}.npz"
        if target.exists():
            continue
        labels, steps = base, []
        for iteration in range(1, args.iterations + 1):
            labels, step = refine_step(fps, labels, threshold, device, "ivf", (cells, probes))
            steps.append(step)
            print(f"{name} iteration {iteration}: {step}", flush=True)
        final, dedup = dedup_step(fps, compact(labels), threshold, device)
        target.parent.mkdir(parents=True, exist_ok=True)
        (target.parent / f"steps_{threshold}.json").write_text(
            json.dumps({"cells": cells, "probes": probes, "r": steps, "r3dedup": dedup}, indent=1)
        )
        partial = target.with_suffix(".partial.npz")
        np.savez_compressed(partial, **{f"{method}_r3dedup{name}_b{batch}_s{seed}": final})
        partial.rename(target)


def cmd_cost(args, device):
    rows = []
    for n in args.sizes:
        args.prefix = n
        _, labels, fps = load_run(args)
        t_cluster = float("nan")
        if args.cluster:
            from nvmolkit.clustering import bitbirch

            words = np.ascontiguousarray(fps).view(np.int32)
            bitbirch(torch.from_numpy(words[:50_000]).cuda(), 0.296, branching_factor=254, batch_size=1024).numpy()
            start = synced()
            bitbirch(torch.from_numpy(words).cuda(), 0.296, branching_factor=254, batch_size=1024).numpy()
            t_cluster = synced() - start
            torch.cuda.empty_cache()
        start = synced()
        _, centroids, pop, _ = pass_inputs(fps, labels, device)
        t_centroids = synced() - start
        k = centroids.shape[0]
        best_other_centroid_ivf(fps[:20_000], labels[:20_000], centroids, pop, device, n_cells=256, probes=4)
        for config in args.configs:
            scale, probes = parse(config)
            cells = cells_for(scale, k)
            torch.cuda.empty_cache()
            start = synced()
            index = build_coarse(centroids, pop, cells, device)
            t_build = synced() - start
            start = synced()
            best_other_centroid_ivf(fps, labels, centroids, pop, device, probes=probes, index=index)
            t_search = synced() - start
            rows.append(
                {
                    "molecules": n,
                    "clusters": k,
                    "scale": scale,
                    "cells": cells,
                    "probes": probes,
                    "build_seconds": t_build,
                    "search_seconds": t_search,
                    "centroid_seconds": t_centroids,
                    "nvmolkit_cluster_seconds": t_cluster,
                }
            )
            print(rows[-1], flush=True)
            del index
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
    parser.add_argument("--configs", nargs="+", required=True, help="SCALE:PROBES, e.g. 2:16 or fixed4096:16")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--cache")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prefix", type=int)
    parser.add_argument("--sizes", type=int, nargs="+", default=[250_000, 500_000, 1_000_000, 2_409_133])
    parser.add_argument("--cluster", action="store_true", help="also time one nvMolKit BitBIRCH pass per size")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        print(f"{args.output}: already done")
        return
    {"recall": cmd_recall, "refine": cmd_refine, "cost": cmd_cost}[args.command](args, torch.device("cuda"))


if __name__ == "__main__":
    main()
