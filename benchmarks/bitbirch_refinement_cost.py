# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time one refinement iteration's parts vs dataset size: centroids, exhaustive search, IVF search, guard.

Uses prefixes of one base partition (compacted). Exhaustive search is timed on up to
``--max-queries`` molecules and scaled linearly (its cost is linear in queries at fixed K).
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitbirch_refinement import (
    best_other_centroid,
    best_other_centroid_ivf,
    cluster_stats,
    compact,
    popcounts,
    violating,
)


class Timer:
    """Wall-clock seconds of a block, synchronized with the GPU on both ends."""

    def __enter__(self):  # noqa: D105
        torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):  # noqa: D105
        torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fingerprints")
    parser.add_argument("labels", type=Path)
    parser.add_argument("run")
    parser.add_argument("--sizes", type=int, nargs="+", default=[250_000, 500_000, 1_000_000, 2_409_133])
    parser.add_argument("--max-queries", type=int, default=262_144)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    threshold = float(re.search(r"labels_threshold_([0-9.]+)\.npz", args.labels.name).group(1))
    fps_all = np.load(args.fingerprints, mmap_mode="r")
    full = np.load(args.labels)[args.run]
    rows = []
    for n in args.sizes:
        labels = compact(full[:n])
        fps = np.ascontiguousarray(fps_all[:n])
        with Timer() as t_centroids:
            _, _, centroids, member_sim = cluster_stats(fps, labels, device)
        torch.cuda.empty_cache()
        pop = popcounts(centroids)
        queries = np.arange(min(n, args.max_queries))
        best_other_centroid(fps, labels, centroids, pop, device, rows=queries[:4096])
        with Timer() as t_exh:
            best_other_centroid(fps, labels, centroids, pop, device, rows=queries)
        best_other_centroid_ivf(fps[:20_000], labels[:20_000], centroids, pop, device)
        with Timer() as t_ivf:
            best_s, best_i = best_other_centroid_ivf(fps, labels, centroids, pop, device)
        k = centroids.shape[0]
        del centroids, pop
        torch.cuda.empty_cache()
        movers = best_s > member_sim
        proposal = labels.copy()
        proposal[movers] = best_i[movers]
        with Timer() as t_guard:
            violating(fps, proposal, threshold, device, np.ones(k, dtype=bool))
        row = {
            "molecules": n,
            "clusters": k,
            "centroid_seconds": t_centroids.seconds,
            "exhaustive_seconds": t_exh.seconds * n / len(queries),
            "ivf_seconds": t_ivf.seconds,
            "guard_round_seconds": t_guard.seconds,
        }
        rows.append(row)
        print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
