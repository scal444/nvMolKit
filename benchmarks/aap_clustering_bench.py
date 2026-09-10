# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark AAP similarity and AAP+DISE clustering implementations."""

import argparse
import importlib
import math
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from bench_utils import (
    TimingResult,
    add_backend_selection_args,
    load_sdf,
    load_smiles,
    print_csv_rows,
    throughput_per_s,
    time_it,
    write_csv_rows,
)
from rdkit import Chem

from nvmolkit.clustering import aap_dise_clustering, aap_similarity, aap_similarity_clustering
from reference.gcheminfo_aap.gcheminfo_aap import (
    compile_runner as compile_gcheminfo_runner,
    read_assignments as read_gcheminfo_assignments,
    run_benchmark as run_gcheminfo_benchmark,
    write_graphs as write_gcheminfo_graphs,
)


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


def _load_ligand_clustering_cpu_reference():
    for module_name in ("ligand_clustering_cpu", "ligand_clustering.ligand_clustering_cpu"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            pass
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


def _ligand_clustering_cpu_pair_scores(molecules, pairs, max_path_length, reference):
    return [
        reference.atom_atom_path_similarity(
            molecules[left],
            molecules[right],
            reference.get_path_integers(molecules[left], max_path_length),
            reference.get_path_integers(molecules[right], max_path_length),
        )
        for left, right in pairs
    ]


def _remap_clusters_by_size(labels, num_clusters):
    sizes = [0] * num_clusters
    for label in labels:
        sizes[label] += 1
    order = sorted(range(num_clusters), key=lambda cluster: -sizes[cluster])
    remap = {old: new + 1 for new, old in enumerate(order)}
    return [remap[label] for label in labels]


def _ligand_clustering_cpu_cluster(molecules, threshold, max_path_length, reference):
    """Run the ligand_clustering exact CPU AAP primitives with DISE.

    The public ligand_clustering CPU wrapper accepts SMILES and parses them
    internally. Keeping the DISE driver here lets every timed backend start
    from the same pre-parsed RDKit molecules while retaining that project's
    descriptor and exact Hungarian AAP implementations.
    """
    descriptors = reference.precompute_path_integers_batch(molecules, max_path_length)
    labels = [-1] * len(molecules)
    cluster_id = 0
    for centroid in range(len(molecules)):
        if labels[centroid] >= 0:
            continue
        labels[centroid] = cluster_id
        for candidate in range(len(molecules)):
            if labels[candidate] >= 0:
                continue
            similarity = reference.atom_atom_path_similarity(
                molecules[centroid],
                molecules[candidate],
                descriptors[centroid],
                descriptors[candidate],
            )
            if similarity >= threshold:
                labels[candidate] = cluster_id
        cluster_id += 1
    return _remap_clusters_by_size(labels, cluster_id)


def _cluster_agreement(left, right):
    if len(left) != len(right):
        raise ValueError("cluster label vectors must have equal length")
    total_pairs = len(left) * (len(left) - 1) // 2
    if total_pairs == 0:
        return 1.0
    left_counts = Counter(left)
    right_counts = Counter(right)
    joint_counts = Counter(zip(left, right, strict=True))
    choose_two = lambda count: count * (count - 1) // 2
    same_left = sum(choose_two(count) for count in left_counts.values())
    same_right = sum(choose_two(count) for count in right_counts.values())
    same_both = sum(choose_two(count) for count in joint_counts.values())
    different_both = total_pairs - same_left - same_right + same_both
    return (same_both + different_both) / total_pairs


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


def _base_row(args, operation, method, count, timing, molecules, status="ok"):
    atom_counts = [molecule.GetNumAtoms() for molecule in molecules]
    is_pair = operation == "similarity"
    is_nvmolkit = method.startswith("nvmolkit_gpu")
    return {
        "method": method,
        "operation": operation,
        "status": status,
        "timing_scope": "from_preparsed_rdkit_molecules",
        "input_file": str(args.sdf or args.smiles),
        "num_mols": len(molecules),
        "num_pairs": count if is_pair else "N/A",
        "mols_processed": count if not is_pair else "N/A",
        "time_ms": round(timing.mean_ms, 4) if timing is not None else "",
        "std_ms": round(timing.std_ms, 4) if timing is not None else "",
        "pairs_per_second": (
            round(throughput_per_s(count, timing.mean_ms), 2) if is_pair and timing is not None else ""
        ),
        "molecules_per_second": (
            round(throughput_per_s(count, timing.mean_ms), 2) if not is_pair and timing is not None else ""
        ),
        "vs_ligand_clustering_cpu_throughput_ratio": "",
        "threshold": args.threshold,
        "max_path_length": args.max_path_length,
        "histogram_bins": args.histogram_bins if is_nvmolkit else "",
        "sinkhorn_iterations": args.sinkhorn_iterations if is_nvmolkit else "",
        "sinkhorn_temperature": args.sinkhorn_temperature if is_nvmolkit else "",
        "mean_atoms": round(float(np.mean(atom_counts)), 2),
        "min_atoms": min(atom_counts),
        "max_atoms": max(atom_counts),
    }


def _set_pair_comparison(rows, outputs, reference_method, field_prefix):
    if reference_method not in outputs:
        return
    expected = np.asarray(outputs[reference_method])
    for row in rows:
        method = row["method"]
        if method not in outputs:
            continue
        actual = np.asarray(outputs[method])
        correlation = float(np.corrcoef(expected, actual)[0, 1]) if np.std(expected) and np.std(actual) else math.nan
        row[f"{field_prefix}_correlation"] = correlation
        row[f"{field_prefix}_mae"] = float(np.mean(np.abs(expected - actual)))


def _benchmark_pairs(args, molecules, rdkit_reference, ligand_clustering_cpu_reference):
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
        outputs["nvmolkit_gpu"] = scores
        rows.append(_base_row(args, "similarity", "nvmolkit_gpu", len(pairs), timing, molecules))

    if not args.no_rdkit:
        if rdkit_reference is None:
            rows.append(_base_row(args, "similarity", "rdkit_contrib", len(pairs), None, molecules, "unavailable"))
        else:
            timing, scores = _time_callable(
                lambda: _rdkit_pair_scores(molecules, pairs, args.max_path_length, rdkit_reference),
                args.runs,
                args.warmup,
            )
            outputs["rdkit_contrib"] = scores
            rows.append(_base_row(args, "similarity", "rdkit_contrib", len(pairs), timing, molecules))

    if not args.no_ligand_clustering_cpu:
        if ligand_clustering_cpu_reference is None:
            rows.append(
                _base_row(
                    args,
                    "similarity",
                    "ligand_clustering_cpu",
                    len(pairs),
                    None,
                    molecules,
                    "unavailable",
                )
            )
        else:
            timing, scores = _time_callable(
                lambda: _ligand_clustering_cpu_pair_scores(
                    molecules, pairs, args.max_path_length, ligand_clustering_cpu_reference
                ),
                args.runs,
                args.warmup,
            )
            outputs["ligand_clustering_cpu"] = scores
            rows.append(_base_row(args, "similarity", "ligand_clustering_cpu", len(pairs), timing, molecules))

    _set_pair_comparison(rows, outputs, "rdkit_contrib", "vs_rdkit_contrib")
    _set_pair_comparison(rows, outputs, "ligand_clustering_cpu", "vs_ligand_clustering_cpu")
    by_method = {row["method"]: row for row in rows}
    if "nvmolkit_gpu" in outputs and "ligand_clustering_cpu" in outputs:
        by_method["nvmolkit_gpu"]["vs_ligand_clustering_cpu_throughput_ratio"] = round(
            by_method["nvmolkit_gpu"]["pairs_per_second"] / by_method["ligand_clustering_cpu"]["pairs_per_second"],
            4,
        )
    return rows


def _benchmark_clustering(args, molecules, ligand_clustering_cpu_reference):
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
        outputs["nvmolkit_gpu"] = labels
        row = _base_row(args, "clustering", "nvmolkit_gpu", len(molecules), timing, molecules)
        row["num_clusters"] = len(set(labels))
        rows.append(row)

    if not args.no_rdkit:
        rows.append(
            _base_row(
                args,
                "clustering",
                "rdkit_contrib",
                len(molecules),
                None,
                molecules,
                "unsupported",
            )
        )

    if not args.no_ligand_clustering_cpu:
        if ligand_clustering_cpu_reference is None:
            rows.append(
                _base_row(
                    args,
                    "clustering",
                    "ligand_clustering_cpu",
                    len(molecules),
                    None,
                    molecules,
                    "unavailable",
                )
            )
        elif len(molecules) > args.cpu_max_size:
            rows.append(
                _base_row(
                    args,
                    "clustering",
                    "ligand_clustering_cpu",
                    len(molecules),
                    None,
                    molecules,
                    "size_limit",
                )
            )
        else:
            timing, labels = _time_callable(
                lambda: _ligand_clustering_cpu_cluster(
                    molecules,
                    args.threshold,
                    args.max_path_length,
                    ligand_clustering_cpu_reference,
                ),
                args.runs,
                args.warmup,
            )
            outputs["ligand_clustering_cpu"] = labels
            row = _base_row(args, "clustering", "ligand_clustering_cpu", len(molecules), timing, molecules)
            row["num_clusters"] = len(set(labels))
            rows.append(row)

    by_method = {row["method"]: row for row in rows}
    if "nvmolkit_gpu" in outputs and "ligand_clustering_cpu" in outputs:
        agreement = _cluster_agreement(outputs["nvmolkit_gpu"], outputs["ligand_clustering_cpu"])
        for method in ("nvmolkit_gpu", "ligand_clustering_cpu"):
            by_method[method]["vs_ligand_clustering_cpu_cluster_agreement"] = agreement
        by_method["nvmolkit_gpu"]["vs_ligand_clustering_cpu_throughput_ratio"] = round(
            by_method["nvmolkit_gpu"]["molecules_per_second"]
            / by_method["ligand_clustering_cpu"]["molecules_per_second"],
            4,
        )
    return rows


def _parse_gcheminfo_benchmark(stdout):
    samples = []
    phase_samples = {
        "ordering_ms": [],
        "descriptors_ms": [],
        "selection_ms": [],
        "assignment_ms": [],
    }
    centroids = None
    for line in stdout.splitlines():
        if not line.startswith("BENCHMARK "):
            continue
        values = dict(field.split("=", 1) for field in line.split()[1:])
        samples.append(float(values["workflow_ms"]))
        for field in phase_samples:
            phase_samples[field].append(float(values[field]))
        centroids = int(values["centroids"])
    if len(samples) == 0:
        raise RuntimeError(f"Java AAP benchmark produced no measured samples:\n{stdout}")
    return TimingResult(times_ms=samples), phase_samples, centroids


def _benchmark_dise(args, molecules, java_classpath):
    kwargs = _aap_kwargs(args)
    rows = []
    outputs = {}

    def ordered_molecules_with_indices():
        indexed = list(enumerate(molecules))
        if args.sort_tag:
            def priority(item):
                value = (
                    item[1].GetProp(args.sort_tag).replace("<", "")
                    if item[1].HasProp(args.sort_tag)
                    else ""
                )
                return (not bool(value), float(value) if value else 0.0)

            indexed.sort(key=priority)
        return indexed

    if not args.no_nvmolkit:
        gpu_labels = None

        def run_gpu_workflow():
            nonlocal gpu_labels
            indexed = ordered_molecules_with_indices()
            ordered_labels = aap_dise_clustering(
                [molecule for _, molecule in indexed], threshold=args.threshold, **kwargs
            )
            gpu_labels = [None] * len(molecules)
            for (input_index, _), label in zip(indexed, ordered_labels, strict=True):
                gpu_labels[input_index] = label

        timing, labels = _time_callable(
            run_gpu_workflow,
            args.runs,
            args.warmup,
            gpu_sync=True,
        )
        labels = gpu_labels
        outputs["nvmolkit_gpu_dise"] = gpu_labels
        row = _base_row(args, "dise", "nvmolkit_gpu_dise", len(molecules), timing, molecules)
        row["num_clusters"] = len(set(labels))
        rows.append(row)

    if java_classpath is not None:
        work_dir = Path(args.gcheminfo_java_work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        graph_path = work_dir / f"gcheminfo_{len(molecules)}_graphs.tsv"
        assignment_path = work_dir / f"gcheminfo_{len(molecules)}_assignments.tsv"
        # Molecular conversion is shared preparation and intentionally untimed.
        priorities = (
            [
                molecule.GetProp(args.sort_tag) if molecule.HasProp(args.sort_tag) else ""
                for molecule in molecules
            ]
            if args.sort_tag
            else None
        )
        write_gcheminfo_graphs(molecules, graph_path, priorities=priorities)
        completed = run_gcheminfo_benchmark(
            graph_path,
            assignment_path,
            classpath=java_classpath,
            threshold=args.threshold,
            max_path_length=args.max_path_length,
            threads=args.gcheminfo_java_threads,
            warmups=1 if args.warmup else 0,
            runs=args.runs,
            order_by_priority=bool(args.sort_tag),
        )
        timing, phases, centroids = _parse_gcheminfo_benchmark(completed.stdout)
        assignment_rows = read_gcheminfo_assignments(assignment_path)
        labels = [None] * len(molecules)
        for assignment in assignment_rows:
            labels[int(assignment["input_index"])] = int(assignment["cluster_index"])
        outputs["gcheminfo_java_default8_dise"] = labels
        row = _base_row(
            args,
            "dise",
            "gcheminfo_java_default8_dise",
            len(molecules),
            timing,
            molecules,
        )
        row["timing_scope"] = "from_preparsed_molecular_graphs"
        row["ordering_time_ms"] = round(float(np.mean(phases["ordering_ms"])), 4)
        row["num_clusters"] = centroids
        row["descriptor_time_ms"] = round(float(np.mean(phases["descriptors_ms"])), 4)
        row["centroid_selection_time_ms"] = round(float(np.mean(phases["selection_ms"])), 4)
        row["nearest_assignment_time_ms"] = round(float(np.mean(phases["assignment_ms"])), 4)
        rows.append(row)

    if "nvmolkit_gpu_dise" in outputs and "gcheminfo_java_default8_dise" in outputs:
        agreement = _cluster_agreement(
            outputs["nvmolkit_gpu_dise"], outputs["gcheminfo_java_default8_dise"]
        )
        by_method = {row["method"]: row for row in rows}
        for method in outputs:
            by_method[method]["vs_gcheminfo_java_cluster_agreement"] = agreement
        by_method["nvmolkit_gpu_dise"]["vs_gcheminfo_java_throughput_ratio"] = round(
            by_method["nvmolkit_gpu_dise"]["molecules_per_second"]
            / by_method["gcheminfo_java_default8_dise"]["molecules_per_second"],
            4,
        )
    return rows


def _build_parser():
    parser = argparse.ArgumentParser(description="AAP similarity and directed sphere-exclusion clustering benchmark")
    parser.add_argument("--smiles", "-s", default=str(DEFAULT_INPUT), help="Input SMILES file")
    parser.add_argument("--sdf", help="Input SDF file; overrides --smiles")
    parser.add_argument(
        "--sort-tag",
        help="Numeric priority property sorted ascending inside each timed DISE workflow",
    )
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
    parser.add_argument(
        "--cpu-max-size",
        "--cpu_max_size",
        type=int,
        default=128,
        help="Largest clustering size run with ligand_clustering CPU",
    )
    parser.add_argument(
        "--operation",
        choices=["similarity", "clustering", "dise", "both", "all"],
        default="both",
        help="'clustering' is selection-only; 'dise' adds nearest-centroid reassignment",
    )
    parser.add_argument("--warmup", action="store_true", dest="warmup", help="Perform one warmup run (default)")
    parser.add_argument("--no_warmup", action="store_false", dest="warmup", help="Skip warmup")
    parser.set_defaults(warmup=True)
    add_backend_selection_args(parser)
    parser.add_argument(
        "--no-ligand-clustering-cpu",
        "--no_ligand_clustering_cpu",
        action="store_true",
        help="Skip the ligand_clustering CPU reference",
    )
    parser.add_argument(
        "--gcheminfo-java",
        action="store_true",
        help="Run the validated gCheminfoCommands DEFAULT8 full-DISE Java reference",
    )
    parser.add_argument(
        "--gcheminfo-java-threads",
        type=int,
        default=2,
        help="Threads for the Java nearest-centroid stage (selection stays sequential)",
    )
    parser.add_argument(
        "--gcheminfo-java-work-dir",
        default="aap_gcheminfo_java_bench",
        help="Directory for untimed Java graph and assignment interchange files",
    )
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
    if args.cpu_max_size < 0:
        print("Error: --cpu-max-size must be non-negative", file=sys.stderr)
        sys.exit(1)
    if args.gcheminfo_java_threads <= 0:
        print("Error: --gcheminfo-java-threads must be positive", file=sys.stderr)
        sys.exit(1)
    if args.no_nvmolkit and args.no_rdkit and args.no_ligand_clustering_cpu:
        print("Error: cannot disable every benchmark backend", file=sys.stderr)
        sys.exit(1)

    rdkit_reference = None if args.no_rdkit else _load_rdkit_reference()
    ligand_clustering_cpu_reference = (
        None if args.no_ligand_clustering_cpu else _load_ligand_clustering_cpu_reference()
    )
    java_classpath = None
    if args.gcheminfo_java:
        java_classpath = compile_gcheminfo_runner(
            Path(args.gcheminfo_java_work_dir) / "classes"
        )
    if rdkit_reference is None and not args.no_rdkit:
        print("RDKit Contrib AAP unavailable; its timing fields will be empty")
    if ligand_clustering_cpu_reference is None and not args.no_ligand_clustering_cpu:
        print("ligand_clustering CPU unavailable; its timing fields will be empty")

    largest_size = max(args.sizes)
    requested = max(largest_size * 3, args.num_pairs * 2)
    if args.sdf:
        loaded = load_sdf(args.sdf, max_count=requested, sanitize=True, seed=args.seed)
    else:
        loaded = load_smiles(args.smiles, max_count=requested, sanitize=True, seed=args.seed)
    molecules = _filter_supported_molecules(loaded, args.max_atoms)
    if len(molecules) < max(largest_size, 2):
        print(
            f"Error: need {max(largest_size, 2)} supported molecules, retained {len(molecules)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nConfiguration:")
    print(f"  Input: {args.sdf or args.smiles}")
    print(f"  Timed priority sort: {args.sort_tag or 'none'}")
    print(f"  Retained molecules: {len(molecules)}")
    print(f"  Clustering sizes: {args.sizes}")
    print(f"  Pair count: {args.num_pairs}")
    print(f"  Runs: {args.runs}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Run nvMolKit GPU: {not args.no_nvmolkit}")
    print(f"  Run RDKit Contrib similarity: {rdkit_reference is not None and not args.no_rdkit}")
    print(
        "  Run ligand_clustering CPU: "
        f"{ligand_clustering_cpu_reference is not None and not args.no_ligand_clustering_cpu}"
    )
    print(f"  Run gCheminfoCommands Java DISE: {args.gcheminfo_java}")

    rows = []
    if args.operation in ("similarity", "both", "all"):
        print(f"\nBenchmarking {args.num_pairs} pair similarities...")
        rows.extend(
            _benchmark_pairs(
                args,
                molecules[:largest_size],
                rdkit_reference,
                ligand_clustering_cpu_reference,
            )
        )
    if args.operation in ("clustering", "both", "all"):
        for size in args.sizes:
            print(f"\nBenchmarking clustering for {size} molecules...")
            rows.extend(_benchmark_clustering(args, molecules[:size], ligand_clustering_cpu_reference))
    if args.operation in ("dise", "all"):
        for size in args.sizes:
            print(f"\nBenchmarking full DISE workflow for {size} molecules...")
            rows.extend(_benchmark_dise(args, molecules[:size], java_classpath))

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
