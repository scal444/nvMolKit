# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure BitBIRCH partition quality and stability on molecular fingerprints."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from rdkit import Chem
from sklearn.metrics import adjusted_mutual_info_score


def _cluster_isim(bits: np.ndarray) -> float:
    count = bits.shape[0]
    if count <= 1:
        return 1.0
    linear_sum = bits.sum(axis=0, dtype=np.float64)
    common_pairs = np.sum(linear_sum * (linear_sum - 1.0) * 0.5, dtype=np.float64)
    mismatches = np.sum(linear_sum * (count - linear_sum), dtype=np.float64)
    denominator = common_pairs + mismatches
    return float(common_pairs / denominator) if denominator else 1.0


def _cluster_quality(bits: np.ndarray, labels: np.ndarray) -> dict:
    cluster_ids, sizes = np.unique(labels, return_counts=True)
    isims = np.asarray([_cluster_isim(bits[labels == cluster_id]) for cluster_id in cluster_ids])
    size_quantiles = np.quantile(sizes, [0.0, 0.25, 0.5, 0.75, 1.0])
    isim_quantiles = np.quantile(isims, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "num_clusters": int(cluster_ids.size),
        "cluster_size_quantiles": size_quantiles.tolist(),
        "within_cluster_isim_quantiles": isim_quantiles.tolist(),
        "singleton_cluster_fraction": float(np.mean(sizes == 1)),
        "items_in_singletons_fraction": float(np.sum(sizes[sizes == 1]) / labels.size),
        "largest_cluster_fraction": float(sizes.max() / labels.size),
    }


def _load_molecules(path: Path, target_count: int) -> list:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        molecules = [Chem.MolFromSmiles(row["smiles"]) for row in reader]
    molecules = [molecule for molecule in molecules if molecule is not None]
    if not molecules:
        raise ValueError(f"no valid molecules in {path}")
    repeats = (target_count + len(molecules) - 1) // len(molecules)
    return (molecules * repeats)[:target_count]


def main() -> None:
    import torch

    from nvmolkit.clustering import bitbirch
    from nvmolkit.fingerprints import MorganFingerprintGenerator, unpack_fingerprint

    default_smiles = Path(__file__).parents[1] / "nvmolkit" / "tests" / "testdata" / "smiles.csv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles-file", type=Path, default=default_smiles)
    parser.add_argument("--num-fingerprints", type=int, default=1000)
    parser.add_argument("--fp-size", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--branching-factor", type=int, default=254)
    parser.add_argument("--partitions", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()

    molecules = _load_molecules(args.smiles_file, args.num_fingerprints)
    start = time.perf_counter()
    fingerprints = MorganFingerprintGenerator(args.radius, args.fp_size).GetFingerprints(
        molecules,
        num_threads=args.num_threads,
    )
    bits = unpack_fingerprint(fingerprints.torch()).cpu().numpy().astype(np.uint8)
    fingerprint_seconds = time.perf_counter() - start

    runs = []
    serial_labels = None
    for num_partitions in args.partitions:
        start = time.perf_counter()
        labels = bitbirch(
            fingerprints,
            args.threshold,
            branching_factor=args.branching_factor,
            num_partitions=num_partitions,
        ).numpy()
        elapsed = time.perf_counter() - start
        if serial_labels is None:
            if num_partitions != 1:
                raise ValueError("--partitions must begin with 1 so serial quality is the baseline")
            serial_labels = labels
        runs.append(
            {
                "num_partitions": num_partitions,
                "seconds": elapsed,
                "adjusted_mutual_information_vs_serial": float(adjusted_mutual_info_score(serial_labels, labels)),
                **_cluster_quality(bits, labels),
            }
        )

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "compute_capability": ".".join(str(value) for value in torch.cuda.get_device_capability()),
                "configuration": {
                    "smiles_file": str(args.smiles_file),
                    "num_fingerprints": args.num_fingerprints,
                    "fp_size": args.fp_size,
                    "radius": args.radius,
                    "threshold": args.threshold,
                    "branching_factor": args.branching_factor,
                    "partitions": args.partitions,
                },
                "fingerprint_seconds": fingerprint_seconds,
                "runs": runs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
