# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark nvMolKit BitBIRCH on real molecular fingerprints.

The optional ``bblean`` backend receives the exact same packed fingerprints as
nvMolKit. Fingerprint generation and host/device conversion are intentionally
outside the measured clustering region.
"""

import argparse
import importlib
import importlib.metadata
import math
import sys
import time
from types import ModuleType

import numpy as np
import torch
from bench_utils import load_smiles, print_csv_rows, time_it, write_csv_rows

from nvmolkit.clustering import bitbirch
from nvmolkit.fingerprints import MorganFingerprintGenerator

DEFAULT_SIZES = [1_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]


def _load_bblean(enabled: bool) -> tuple[ModuleType | None, str, str]:
    if not enabled:
        return None, "disabled", ""
    try:
        module = importlib.import_module("bblean")
    except ImportError as error:
        return None, "not_installed", str(error)
    try:
        version = importlib.metadata.version("bblean")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return module, "available", version


def _cluster_count(labels: torch.Tensor) -> int:
    return int(labels.max().item()) + 1 if labels.numel() else 0


def _bench_nvmolkit(
    fingerprints: torch.Tensor,
    *,
    threshold: float,
    branching_factor: int,
    num_partitions: int | None,
    runs: int,
    warmups: int,
) -> tuple[object, int]:
    holder: dict[str, torch.Tensor] = {}

    def run() -> None:
        holder["labels"] = bitbirch(
            fingerprints,
            threshold,
            branching_factor=branching_factor,
            num_partitions=num_partitions,
        ).torch()

    timing = time_it(run, runs=runs, warmups=warmups, gpu_sync=True)
    return timing, _cluster_count(holder["labels"])


def _bench_bblean(
    bblean: ModuleType,
    fingerprints: np.ndarray,
    *,
    threshold: float,
    branching_factor: int,
    runs: int,
    warmups: int,
) -> tuple[object, dict[str, float | int]]:
    holder: dict[str, object] = {}

    def run() -> None:
        tree = bblean.BitBirch(
            threshold=threshold,
            branching_factor=branching_factor,
            merge_criterion="diameter",
        )
        tree.fit(fingerprints)
        holder["tree"] = tree

    timing = time_it(run, runs=runs, warmups=warmups)
    tree = holder["tree"]
    return timing, _cluster_distribution(tree.get_cluster_mol_ids(), len(fingerprints))


def _cluster_distribution(clusters, num_fingerprints: int) -> dict[str, float | int]:
    sizes = np.asarray([len(cluster) for cluster in clusters], dtype=np.int64)
    if not sizes.size:
        return {
            "num_clusters": 0,
            "mean_cluster_size": math.nan,
            "cluster_size_p50": math.nan,
            "cluster_size_p90": math.nan,
            "cluster_size_p99": math.nan,
            "max_cluster_size": 0,
            "singleton_cluster_fraction": math.nan,
            "items_in_singletons_fraction": math.nan,
            "largest_cluster_fraction": math.nan,
        }
    quantiles = np.quantile(sizes, [0.5, 0.9, 0.99])
    singleton_count = int(np.count_nonzero(sizes == 1))
    return {
        "num_clusters": int(sizes.size),
        "mean_cluster_size": float(sizes.mean()),
        "cluster_size_p50": float(quantiles[0]),
        "cluster_size_p90": float(quantiles[1]),
        "cluster_size_p99": float(quantiles[2]),
        "max_cluster_size": int(sizes.max()),
        "singleton_cluster_fraction": singleton_count / int(sizes.size),
        "items_in_singletons_fraction": singleton_count / num_fingerprints,
        "largest_cluster_fraction": int(sizes.max()) / num_fingerprints,
    }


def _empty_timing_fields(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean_ms": math.nan,
        f"{prefix}_median_ms": math.nan,
        f"{prefix}_std_ms": math.nan,
    }


def _timing_fields(prefix: str, timing) -> dict[str, float]:
    return {
        f"{prefix}_mean_ms": timing.mean_ms,
        f"{prefix}_median_ms": timing.median_ms,
        f"{prefix}_std_ms": timing.std_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_smiles_file", help="Path to a real SMILES/CXSMILES input file")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Sweep multiple thresholds; overrides --threshold",
    )
    parser.add_argument("--branching-factor", type=int, default=254)
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=0,
        help="nvMolKit partitions; 0 uses automatic selection (default: 0)",
    )
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp-size", type=int, default=1024)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bblean", action="store_true", help="Benchmark the optional bblean CPU backend")
    parser.add_argument(
        "--nvmolkit-max-size",
        type=int,
        default=0,
        help="Skip nvMolKit above this size; 0 disables the limit",
    )
    parser.add_argument(
        "--bblean-max-size",
        type=int,
        default=0,
        help="Skip bblean above this size; 0 disables the limit",
    )
    parser.add_argument("--no-nvmolkit", action="store_true")
    parser.add_argument("-o", "--output", default="bitbirch_results.csv")
    args = parser.parse_args()
    thresholds = args.thresholds if args.thresholds is not None else [args.threshold]

    if not args.sizes or any(size <= 0 for size in args.sizes):
        parser.error("--sizes must contain positive integers")
    if args.runs <= 0 or args.warmups < 0:
        parser.error("--runs must be positive and --warmups must be nonnegative")
    if args.num_partitions < 0:
        parser.error("--num-partitions must be nonnegative")
    if args.nvmolkit_max_size < 0 or args.bblean_max_size < 0:
        parser.error("backend maximum sizes must be nonnegative")
    if any(not math.isfinite(threshold) or not 0 <= threshold <= 1 for threshold in thresholds):
        parser.error("thresholds must be finite and in [0, 1]")
    if args.no_nvmolkit and not args.bblean:
        parser.error("no benchmark backend selected")

    sizes = sorted(set(args.sizes))
    max_size = sizes[-1]
    bblean, bblean_status, bblean_version = _load_bblean(args.bblean)
    if args.bblean and bblean is None:
        print(f"bblean unavailable ({bblean_status}): {bblean_version}", file=sys.stderr)

    molecules = load_smiles(args.input_smiles_file, max_count=max_size + 100, sanitize=True, seed=args.seed)
    if len(molecules) < max_size:
        parser.error(f"requested {max_size} molecules, but only {len(molecules)} valid molecules were loaded")

    print(f"Generating {max_size} Morgan fingerprints with all available CPU threads")
    fingerprint_start = time.perf_counter()
    fingerprints = (
        MorganFingerprintGenerator(args.radius, args.fp_size)
        .GetFingerprints(molecules[:max_size], num_threads=0)
        .torch()
    )
    torch.cuda.synchronize()
    fingerprint_seconds = time.perf_counter() - fingerprint_start
    print(f"Fingerprint generation finished in {fingerprint_seconds:.3f} s")

    results = []
    num_partitions = args.num_partitions or None
    try:
        for size in sizes:
            device_fps = fingerprints[:size].contiguous()
            host_fps = (
                device_fps.cpu().numpy().view(np.uint8).reshape(size, -1).copy() if bblean is not None else None
            )
            for threshold in thresholds:
                print(f"Benchmarking {size} real fingerprints at threshold {threshold}")
                row = {
                    "size": size,
                    "threshold": threshold,
                    "branching_factor": args.branching_factor,
                    "num_partitions": "auto" if num_partitions is None else num_partitions,
                    "radius": args.radius,
                    "fp_size": args.fp_size,
                    "runs": args.runs,
                    "warmups": args.warmups,
                    "seed": args.seed,
                    "fingerprint_total_seconds": fingerprint_seconds,
                    "device": torch.cuda.get_device_name(device_fps.device),
                    "bblean_version": bblean_version if bblean is not None else "",
                }

                if args.no_nvmolkit:
                    row.update(_empty_timing_fields("nvmolkit"))
                    row.update({"nvmolkit_status": "disabled", "nvmolkit_num_clusters": math.nan})
                elif args.nvmolkit_max_size and size > args.nvmolkit_max_size:
                    row.update(_empty_timing_fields("nvmolkit"))
                    row.update({"nvmolkit_status": "size_limit", "nvmolkit_num_clusters": math.nan})
                else:
                    timing, num_clusters = _bench_nvmolkit(
                        device_fps,
                        threshold=threshold,
                        branching_factor=args.branching_factor,
                        num_partitions=num_partitions,
                        runs=args.runs,
                        warmups=args.warmups,
                    )
                    row.update(_timing_fields("nvmolkit", timing))
                    row.update({"nvmolkit_status": "ok", "nvmolkit_num_clusters": num_clusters})

                if bblean is None:
                    row.update(_empty_timing_fields("bblean"))
                    row.update({"bblean_status": bblean_status})
                elif args.bblean_max_size and size > args.bblean_max_size:
                    row.update(_empty_timing_fields("bblean"))
                    row.update({"bblean_status": "size_limit"})
                else:
                    timing, distribution = _bench_bblean(
                        bblean,
                        host_fps,
                        threshold=threshold,
                        branching_factor=args.branching_factor,
                        runs=args.runs,
                        warmups=args.warmups,
                    )
                    row.update(_timing_fields("bblean", timing))
                    row.update({f"bblean_{key}": value for key, value in distribution.items()})
                    row["bblean_status"] = "ok"

                nvmolkit_ms = row["nvmolkit_median_ms"]
                bblean_ms = row["bblean_median_ms"]
                row["nvmolkit_fingerprints_per_second"] = (
                    size * 1000.0 / nvmolkit_ms if math.isfinite(nvmolkit_ms) else math.nan
                )
                row["bblean_fingerprints_per_second"] = (
                    size * 1000.0 / bblean_ms if math.isfinite(bblean_ms) else math.nan
                )
                row["bblean_speedup_vs_nvmolkit"] = (
                    nvmolkit_ms / bblean_ms
                    if math.isfinite(nvmolkit_ms) and math.isfinite(bblean_ms)
                    else math.nan
                )

                results.append(row)
                write_csv_rows(results, args.output)
    except Exception:
        write_csv_rows(results, args.output)
        raise

    print("\nCSV Results:")
    print_csv_rows(results)


if __name__ == "__main__":
    main()
