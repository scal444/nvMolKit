# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare nvMolKit batching effects on BitBIRCH with serial BitBIRCH's own input-order sensitivity.

Serial BitBIRCH depends on input order, so agreement with one serial run is not
a meaningful target by itself. For each threshold and input-order seed, this
script clusters the same fingerprints with serial bblean and with nvMolKit at
several batch sizes, then reports agreement for:

* ``noise_floor``: bblean vs bblean on two different input orders.
* ``same_order``: nvMolKit vs bblean on the same input order (pure batching effect).
* ``cross_order``: nvMolKit vs bblean on different orders (comparable to the noise floor).
* ``nvmolkit_order``: nvMolKit vs nvMolKit on different orders at the same batch size.
* ``vs_nvmolkit_serial``: nvMolKit at a batch size vs nvMolKit batch size one on
  the same order, isolating batching from any implementation difference.

With batch size one on the same order, ``same_order`` measures the gap between
the two serial implementations.
"""

import argparse
import csv
import itertools
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

from nvmolkit.clustering import bitbirch


def canonical_labels(labels: np.ndarray) -> np.ndarray:
    """Relabel clusters by their first member so equal partitions compare equal."""
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))
    return order[inverse].astype(np.int32)


def bblean_labels(bblean, fingerprints: np.ndarray, threshold: float, branching_factor: int) -> np.ndarray:
    tree = bblean.BitBirch(threshold=threshold, branching_factor=branching_factor, merge_criterion="diameter")
    tree.fit(fingerprints.view(np.uint8).reshape(len(fingerprints), -1))
    labels = np.empty(len(fingerprints), dtype=np.int32)
    for cluster, members in enumerate(tree.get_cluster_mol_ids()):
        labels[np.asarray(members)] = cluster
    return labels


def nvmolkit_labels(fingerprints: np.ndarray, threshold: float, branching_factor: int, batch_size: int) -> np.ndarray:
    device = torch.from_numpy(fingerprints).cuda()
    return bitbirch(device, threshold, branching_factor=branching_factor, batch_size=batch_size).numpy()


def pair_counts(values: np.ndarray) -> float:
    values = values.astype(np.float64)
    return float((values * (values - 1) / 2).sum())


def agreement(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    table = contingency_matrix(reference, candidate, sparse=True)
    together = pair_counts(table.data)
    reference_pairs = pair_counts(np.asarray(table.sum(axis=1)).ravel())
    candidate_pairs = pair_counts(np.asarray(table.sum(axis=0)).ravel())
    precision = together / candidate_pairs if candidate_pairs else 1.0
    recall = together / reference_pairs if reference_pairs else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "identical": int(np.array_equal(canonical_labels(reference), canonical_labels(candidate))),
        "ari": adjusted_rand_score(reference, candidate),
        "nmi": normalized_mutual_info_score(reference, candidate),
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "cluster_ratio": (candidate.max() + 1) / (reference.max() + 1),
    }


def summary(labels: np.ndarray) -> dict[str, float]:
    sizes = np.bincount(labels)
    return {"clusters": len(sizes), "singleton_fraction": float((sizes == 1).sum() / len(labels))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fingerprints", help="NumPy .npy file of packed int32/uint32 fingerprints, shape (N, W)")
    parser.add_argument("--size", type=int, default=100_000)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.45, 0.65, 0.8])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64, 256, 1024, 4096])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--branching-factor", type=int, default=254)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--save-labels", action="store_true", help="write every run's labels to labels_threshold_<t>.npz"
    )
    args = parser.parse_args()

    import bblean  # optional reference dependency

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = np.ascontiguousarray(np.load(args.fingerprints, mmap_mode="r")[: args.size]).view(np.int32)
    permutations = [np.random.default_rng(seed).permutation(len(fingerprints)) for seed in range(args.seeds)]

    runs, rows = [], []
    for threshold in args.thresholds:
        serial, batched = {}, {}
        for seed, permutation in enumerate(permutations):
            ordered = np.ascontiguousarray(fingerprints[permutation])
            start = time.perf_counter()
            labels = np.empty(len(ordered), dtype=np.int32)
            labels[permutation] = bblean_labels(bblean, ordered, threshold, args.branching_factor)
            serial[seed] = labels
            runs.append(("bblean", threshold, 1, seed, time.perf_counter() - start, *summary(labels).values()))
            for batch_size in args.batch_sizes:
                start = time.perf_counter()
                labels = np.empty(len(ordered), dtype=np.int32)
                labels[permutation] = nvmolkit_labels(ordered, threshold, args.branching_factor, batch_size)
                batched[batch_size, seed] = labels
                runs.append(
                    ("nvmolkit", threshold, batch_size, seed, time.perf_counter() - start, *summary(labels).values())
                )
            print(f"threshold {threshold}: seed {seed} done ({time.strftime('%H:%M:%S')})", flush=True)
        if args.save_labels:
            np.savez_compressed(
                args.output_dir / f"labels_threshold_{threshold}.npz",
                **{f"bblean_b1_s{seed}": labels for seed, labels in serial.items()},
                **{f"nvmolkit_b{size}_s{seed}": labels for (size, seed), labels in batched.items()},
            )

        def record(kind, batch_size, reference_seed, candidate_seed, reference, candidate):
            rows.append(
                {
                    "threshold": threshold,
                    "comparison": kind,
                    "batch_size": batch_size,
                    "reference_seed": reference_seed,
                    "candidate_seed": candidate_seed,
                    **agreement(reference, candidate),
                }
            )

        for left, right in itertools.combinations(range(args.seeds), 2):
            record("noise_floor", 1, left, right, serial[left], serial[right])
        for (batch_size, seed), labels in batched.items():
            for reference_seed in range(args.seeds):
                kind = "same_order" if reference_seed == seed else "cross_order"
                record(kind, batch_size, reference_seed, seed, serial[reference_seed], labels)
        for (batch_size, seed), labels in batched.items():
            if batch_size != 1 and (1, seed) in batched:
                record("vs_nvmolkit_serial", batch_size, seed, seed, batched[1, seed], labels)
        for batch_size in args.batch_sizes:
            seeds = [seed for (size, seed) in batched if size == batch_size]
            for left, right in itertools.combinations(seeds, 2):
                record(
                    "nvmolkit_order", batch_size, left, right, batched[batch_size, left], batched[batch_size, right]
                )

    with open(args.output_dir / "order_sensitivity_agreement.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.output_dir / "order_sensitivity_runs.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "threshold", "batch_size", "seed", "seconds", "clusters", "singleton_fraction"])
        writer.writerows(runs)
    print(f"wrote {len(rows)} comparisons and {len(runs)} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
