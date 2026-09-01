# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark nvMolKit approximate AAP similarity and clustering against RDKit."""

import argparse
import importlib
import math
import random
import sys
import warnings
from pathlib import Path

import numpy as np
from bench_utils import (
    add_backend_selection_args,
    load_smiles,
    print_csv_rows,
    throughput_per_s,
    time_it,
    write_csv_rows,
)
from rdkit import Chem

from nvmolkit.clustering import aap_similarity, aap_similarity_clustering


SUPPORTED_BOND_TYPES = {
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
}
DEFAULT_INPUT = Path(__file__).resolve().parent / "data" / "chembl_10k.smi"


def _load_rdkit_reference():
    try:
        return importlib.import_module("rdkit.Contrib.AtomAtomSimilarity.AtomAtomPathSimilarity")
    except ImportError:
        return None


def _filter_supported_molecules(molecules, max_atoms):
    return [
        molecule
        for molecule in molecules
        if 0 < molecule.GetNumAtoms() <= max_atoms
        and all(bond.GetBondType() in SUPPORTED_BOND_TYPES for bond in molecule.GetBonds())
    ]


def _sample_pairs(num_molecules, num_pairs, seed):
    rng = random.Random(seed)
    pairs = []
    for _ in range(num_pairs):
        left = rng.randrange(num_molecules)
        right = rng.randrange(num_molecules - 1)
        if right >= left:
            right += 1
        pairs.append((left, right))
    return pairs


def _remap_clusters_by_size(labels):
    sizes = np.bincount(labels)
    order = sorted(range(len(sizes)), key=lambda cluster: -sizes[cluster])
    remap = {old: new for new, old in enumerate(order)}
    return [remap[label] for label in labels]


def _rdkit_pair_scores(molecules, pairs, max_path_length, reference):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return [
            reference.AtomAtomPathSimilarity(
                molecules[left],
                molecules[right],
                reference.getpathintegers(molecules[left], max_path_length),
                reference.getpathintegers(molecules[right], max_path_length),
            )
            for left, right in pairs
        ]


def _rdkit_aap_clustering(molecules, threshold, max_path_length, reference):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        descriptors = [reference.getpathintegers(molecule, max_path_length) for molecule in molecules]
        labels = [-1] * len(molecules)
        cluster = 0
        for centroid in range(len(molecules)):
            if labels[centroid] >= 0:
                continue
            labels[centroid] = cluster
            for candidate in range(len(molecules)):
                if labels[candidate] >= 0:
                    continue
                similarity = reference.AtomAtomPathSimilarity(
                    molecules[centroid],
                    molecules[candidate],
                    descriptors[centroid],
                    descriptors[candidate],
                )
                if similarity >= threshold:
                    labels[candidate] = cluster
            cluster += 1
    return _remap_clusters_by_size(labels)


def _cluster_agreement(left, right):
    agreements = 0
    comparisons = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            agreements += (left[first] == left[second]) == (right[first] == right[second])
            comparisons += 1
    return agreements / comparisons if comparisons else 1.0


def _time_callable(function, runs, warmup, gpu_sync=False):
    result = None

    def invoke():
        nonlocal result
        result = function()

    timing = time_it(invoke, runs=runs, warmups=1 if warmup else 0, gpu_sync=gpu_sync)
    return timing, result


def _aap_kwargs(args):
    return {
        "max_path_length": args.max_path_length,
        "histogram_bins": args.histogram_bins,
        "sinkhorn_iterations": args.sinkhorn_iterations,
        "sinkhorn_temperature": args.sinkhorn_temperature,
    }


def _base_row(args, operation, method, count, timing, molecules):
    atom_counts = [molecule.GetNumAtoms() for molecule in molecules]
    is_pair = operation == "pair_similarity"
    is_nvmolkit = method == "nvmolkit"
    return {
        "method": method,
        "operation": operation,
        "input_file": str(args.smiles),
        "num_mols": len(molecules),
        "num_pairs": count if is_pair else "N/A",
        "mols_processed": count if not is_pair else "N/A",
        "time_ms": round(timing.mean_ms, 4),
        "std_ms": round(timing.std_ms, 4),
        "pairs_per_second": round(throughput_per_s(count, timing.mean_ms), 2) if is_pair else "N/A",
        "molecules_per_second": round(throughput_per_s(count, timing.mean_ms), 2) if not is_pair else "N/A",
        "vs_rdkit_throughput_ratio": "N/A",
        "threshold": args.threshold,
        "max_path_length": args.max_path_length,
        "histogram_bins": args.histogram_bins if is_nvmolkit else "N/A",
        "sinkhorn_iterations": args.sinkhorn_iterations if is_nvmolkit else "N/A",
        "sinkhorn_temperature": args.sinkhorn_temperature if is_nvmolkit else "N/A",
        "mean_atoms": round(float(np.mean(atom_counts)), 2),
        "min_atoms": min(atom_counts),
        "max_atoms": max(atom_counts),
    }


def _benchmark_pairs(args, molecules, reference):
    pairs = _sample_pairs(len(molecules), args.num_pairs, args.seed)
    kwargs = _aap_kwargs(args)
    rows = []
    outputs = {}

    if not args.no_nvmolkit:
        timing, scores = _time_callable(
            lambda: [aap_similarity(molecules[left], molecules[right], **kwargs) for left, right in pairs],
            args.runs,
            args.warmup,
            gpu_sync=True,
        )
        outputs["nvmolkit"] = scores
        rows.append(_base_row(args, "pair_similarity", "nvmolkit", len(pairs), timing, molecules))

    if reference is not None and not args.no_rdkit:
        timing, scores = _time_callable(
            lambda: _rdkit_pair_scores(molecules, pairs, args.max_path_length, reference),
            args.runs,
            args.warmup,
        )
        outputs["rdkit"] = scores
        rows.append(_base_row(args, "pair_similarity", "rdkit", len(pairs), timing, molecules))

    if "nvmolkit" in outputs and "rdkit" in outputs:
        expected = np.asarray(outputs["rdkit"])
        actual = np.asarray(outputs["nvmolkit"])
        correlation = float(np.corrcoef(expected, actual)[0, 1]) if np.std(expected) and np.std(actual) else math.nan
        mae = float(np.mean(np.abs(expected - actual)))
        for row in rows:
            row["reference_correlation"] = correlation
            row["reference_mae"] = mae
        rows[0]["vs_rdkit_throughput_ratio"] = round(rows[0]["pairs_per_second"] / rows[1]["pairs_per_second"], 4)
    return rows


def _benchmark_clustering(args, molecules, reference):
    kwargs = _aap_kwargs(args)
    rows = []
    outputs = {}

    if not args.no_nvmolkit:
        timing, labels = _time_callable(
            lambda: aap_similarity_clustering(molecules, threshold=args.threshold, **kwargs),
            args.runs,
            args.warmup,
            gpu_sync=True,
        )
        outputs["nvmolkit"] = labels
        row = _base_row(args, "clustering", "nvmolkit", len(molecules), timing, molecules)
        row["num_clusters"] = len(set(labels))
        rows.append(row)

    if reference is not None and not args.no_rdkit and len(molecules) <= args.rdkit_max_size:
        timing, labels = _time_callable(
            lambda: _rdkit_aap_clustering(molecules, args.threshold, args.max_path_length, reference),
            args.runs,
            args.warmup,
        )
        outputs["rdkit"] = labels
        row = _base_row(args, "clustering", "rdkit", len(molecules), timing, molecules)
        row["num_clusters"] = len(set(labels))
        rows.append(row)

    if "nvmolkit" in outputs and "rdkit" in outputs:
        agreement = _cluster_agreement(outputs["nvmolkit"], outputs["rdkit"])
        for row in rows:
            row["reference_cluster_agreement"] = agreement
        rows[0]["vs_rdkit_throughput_ratio"] = round(
            rows[0]["molecules_per_second"] / rows[1]["molecules_per_second"], 4
        )
    return rows


def _build_parser():
    parser = argparse.ArgumentParser(description="AAP similarity and directed sphere-exclusion clustering benchmark")
    parser.add_argument("--smiles", "-s", default=str(DEFAULT_INPUT), help="Input SMILES file")
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 128, 512], help="Clustering sizes")
    parser.add_argument("--num_pairs", type=int, default=128, help="Random molecule pairs for similarity timing")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Number of timing runs")
    parser.add_argument("--seed", type=int, default=42, help="Molecule and pair sampling seed")
    parser.add_argument("--threshold", type=float, default=0.217, help="Inclusive clustering similarity threshold")
    parser.add_argument("--max_atoms", type=int, default=64, help="Maximum atoms retained, at most 64")
    parser.add_argument("--max_path_length", type=int, default=7)
    parser.add_argument("--histogram_bins", type=int, default=2048)
    parser.add_argument("--sinkhorn_iterations", type=int, default=8)
    parser.add_argument("--sinkhorn_temperature", type=float, default=0.104)
    parser.add_argument("--rdkit_max_size", type=int, default=128, help="Largest clustering size run with RDKit")
    parser.add_argument("--operation", choices=["pair", "clustering", "both"], default="both")
    parser.add_argument("--warmup", action="store_true", dest="warmup", help="Perform one warmup run (default)")
    parser.add_argument("--no_warmup", action="store_false", dest="warmup", help="Skip warmup")
    parser.set_defaults(warmup=True)
    add_backend_selection_args(parser)
    parser.add_argument("--output", "-o", help="Optional path to write CSV results")
    return parser


def main():
    args = _build_parser().parse_args()
    if args.runs <= 0 or args.num_pairs <= 0:
        print("Error: --runs and --num_pairs must be positive", file=sys.stderr)
        sys.exit(1)
    if not args.sizes or any(size <= 0 for size in args.sizes):
        print("Error: --sizes must contain positive values", file=sys.stderr)
        sys.exit(1)
    if not 1 <= args.max_atoms <= 64:
        print("Error: --max_atoms must be between 1 and 64", file=sys.stderr)
        sys.exit(1)
    if not 0.0 <= args.threshold <= 1.0:
        print("Error: --threshold must be between 0 and 1", file=sys.stderr)
        sys.exit(1)
    if args.rdkit_max_size < 0:
        print("Error: --rdkit_max_size must be non-negative", file=sys.stderr)
        sys.exit(1)
    if args.no_nvmolkit and args.no_rdkit:
        print("Error: cannot disable both nvMolKit and RDKit", file=sys.stderr)
        sys.exit(1)

    reference = None if args.no_rdkit else _load_rdkit_reference()
    if reference is None and not args.no_rdkit:
        print("RDKit AAP reference unavailable; skipping RDKit rows")
    if args.no_nvmolkit and reference is None:
        print("Error: no benchmark backend is available", file=sys.stderr)
        sys.exit(1)

    largest_size = max(args.sizes)
    requested = max(largest_size * 3, args.num_pairs * 2)
    loaded = load_smiles(args.smiles, max_count=requested, sanitize=True, seed=args.seed)
    molecules = _filter_supported_molecules(loaded, args.max_atoms)
    if len(molecules) < max(largest_size, 2):
        print(
            f"Error: need {max(largest_size, 2)} supported molecules, retained {len(molecules)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nConfiguration:")
    print(f"  Input: {args.smiles}")
    print(f"  Retained molecules: {len(molecules)}")
    print(f"  Clustering sizes: {args.sizes}")
    print(f"  Pair count: {args.num_pairs}")
    print(f"  Runs: {args.runs}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Run nvMolKit: {not args.no_nvmolkit}")
    print(f"  Run RDKit: {reference is not None and not args.no_rdkit}")

    rows = []
    if args.operation in ("pair", "both"):
        print(f"\nBenchmarking {args.num_pairs} pair similarities...")
        rows.extend(_benchmark_pairs(args, molecules[:largest_size], reference))
    if args.operation in ("clustering", "both"):
        for size in args.sizes:
            print(f"\nBenchmarking clustering for {size} molecules...")
            rows.extend(_benchmark_clustering(args, molecules[:size], reference))

    if not rows:
        print("Error: no benchmark rows were produced", file=sys.stderr)
        sys.exit(1)

    print("\nResults:")
    print_csv_rows(rows)
    write_csv_rows(rows, args.output)
    if args.output:
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
