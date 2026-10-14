# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grid-scan nvMolKit substructure-search and MCS execution settings on one GPU.

Companion to ``scan_hardware_options.py`` for the two benchmarks whose GPU-worker and preprocessing
thread pools run concurrently. The CPU budget is shared, so every point uses
``preprocessingThreads = cpu_budget - workerThreads`` and the scan covers batch size and worker threads
(and, for MCS, executors per runner). Inputs are built with the benchmarks' own loaders, so a scanned
configuration applies to exactly the workload the publication runner times.

Pruning, as in ``scan_hardware_options.py``: batch sizes are scanned in ascending order and a row stops
once a point falls ``--prune_drop`` below that row's best; worker threads are raised only while each step
gains at least ``--worker_min_gain``. Results are appended per point and an interrupted scan resumes.

Examples:
    python scan_search_options.py substruct --smiles data.smi --output scan.csv
    python scan_search_options.py mcs --smiles data.smi --output scan.csv
"""

import argparse
import csv
import itertools
import time
from pathlib import Path

import mcs_bench
import substruct_bench
import torch
from bench_utils import load_smarts, load_smiles

from nvmolkit.mcs import MCSConfig
from nvmolkit.substructure import SubstructSearchConfig

SMARTS_DIR = Path(__file__).resolve().parent.parent / "tests" / "test_data" / "SMARTS"
SUBSTRUCT_SETS = (
    ("rdkit_fragment_descriptors_supported", "countSubstructMatches"),
    ("wehi_pains_supported", "hasSubstructMatch"),
    ("BMS_2006_filter_supported", "hasSubstructMatch"),
    ("rdkit_tautomer_transforms_supported", "getSubstructMatches"),
    ("rdkit_torsionPreferences_v2_supported", "getSubstructMatches"),
)
# (name, atom compare, bond compare, ring matches ring only), matching run_publication_benchmarks.sh.
MCS_ROWS = (
    ("mcs_lax_lax", "any", "any", False),
    ("mcs_lax_strict", "any", "order", False),
    ("mcs_strict_lax", "elements", "any", False),
    ("mcs_strict_strict", "elements", "order", False),
    ("mcs_strict_strict_ring", "elements", "order", True),
)
FIELDS = [
    "benchmark",
    "variant",
    "batch_size",
    "workers",
    "prep_threads",
    "executors",
    "num_items",
    "time_ms",
    "items_per_second",
    "status",
]


def _done(path: Path) -> dict[tuple, float | None]:
    if not path.exists():
        return {}
    with open(path) as handle:
        return {
            (r["benchmark"], r["variant"], int(r["batch_size"]), int(r["workers"]), int(r["executors"])): (
                float(r["items_per_second"]) if r["items_per_second"] else None
            )
            for r in csv.DictReader(handle)
        }


def _append(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


class Pruner:
    """Row pruning along batch size and plateau pruning along worker threads."""

    def __init__(self, drop: float, min_gain: float):
        """Set the batch-size drop and worker-thread gain thresholds."""
        self.drop, self.min_gain = drop, min_gain
        self.row_best: dict[tuple, float] = {}
        self.pruned_rows: set[tuple] = set()
        self.prev_worker_rate: dict[tuple, float] = {}
        self.plateaued: set[tuple] = set()
        # Rows whose worker count lost to the previous worker count, counted per row, and the rows cut for it.
        self.row_losses: dict[tuple, int] = {}
        self.worker_lost_rows: set[tuple] = set()

    def skip(self, row_key: tuple, col_key: tuple) -> bool:
        """Return True when the point's row or worker column has already been pruned."""
        if row_key in self.worker_lost_rows:
            # This worker count lost to fewer workers; do not try more workers at the batch sizes it skipped.
            self.plateaued.add(col_key)
        return row_key in self.pruned_rows or col_key in self.plateaued

    def record(self, row_key: tuple, col_key: tuple, rate: float | None) -> list[str]:
        """Record a point's throughput and return the pruning events it triggers."""
        events = []
        if rate is None:
            if self.drop > 0:
                self.pruned_rows.add(row_key)
                events.append("PRUNED")
        else:
            self.row_best[row_key] = max(rate, self.row_best.get(row_key, 0.0))
            if self.drop > 0 and rate < (1.0 - self.drop) * self.row_best[row_key]:
                self.pruned_rows.add(row_key)
                events.append("PRUNED")
        previous = self.prev_worker_rate.get(col_key)
        if rate is not None:
            self.prev_worker_rate[col_key] = rate
        if self.min_gain > 0 and previous is not None and (rate is None or rate < (1.0 + self.min_gain) * previous):
            self.plateaued.add(col_key)
            events.append("PLATEAU")
            if rate is None or rate < previous:
                # More workers lost outright at two batch sizes: the rest of the row will not win either.
                self.row_losses[row_key] = self.row_losses.get(row_key, 0) + 1
                if self.row_losses[row_key] >= 2 and row_key not in self.pruned_rows:
                    self.pruned_rows.add(row_key)
                    self.worker_lost_rows.add(row_key)
                    events.append("PRUNED")
        return events


def _scan(args, benchmark: str, variant: str, num_items: int, run_point, output: Path, done: dict) -> None:
    """Scan one (benchmark, variant). ``run_point(batch, workers, prep, executors)`` returns elapsed ms."""
    pruner = Pruner(args.prune_drop, args.worker_min_gain)
    for executors, workers, batch in itertools.product(args.executors, args.workers, args.batch_sizes):
        row_key, col_key = (executors, workers), (executors, batch)
        if pruner.skip(row_key, col_key):
            continue
        key = (benchmark, variant, batch, workers, executors)
        if key in done:
            pruner.record(row_key, col_key, done[key])
            continue
        prep = args.cpu_budget - workers
        row = {
            "benchmark": benchmark,
            "variant": variant,
            "batch_size": batch,
            "workers": workers,
            "prep_threads": prep,
            "executors": executors,
            "num_items": num_items,
            "time_ms": "",
            "items_per_second": "",
            "status": "ok",
        }
        try:
            elapsed_ms = run_point(batch, workers, prep, executors)
            row["time_ms"] = round(elapsed_ms, 1)
            row["items_per_second"] = round(num_items / (elapsed_ms / 1000), 1)
        except _Skipped as reason:
            row["status"] = f"skipped: {reason}"
            _append(output, row)
            print(
                "SKIP",
                benchmark,
                variant,
                f"batch={batch}",
                f"workers={workers}",
                f"exec={executors}",
                reason,
                flush=True,
            )
            continue
        except torch.cuda.OutOfMemoryError:
            row["status"] = "oom"
        except RuntimeError as error:
            row["status"] = "oom" if "out of memory" in str(error).lower() else f"error: {error}"[:200]
        torch.cuda.empty_cache()
        _append(output, row)
        rate = float(row["items_per_second"]) if row["items_per_second"] != "" else None
        print(
            "RESULT",
            benchmark,
            variant,
            f"batch={batch}",
            f"workers={workers}",
            f"prep={prep}",
            f"exec={executors}",
            f"items/s={row['items_per_second']}",
            f"t={row['time_ms']}ms",
            row["status"],
            flush=True,
        )
        for event in pruner.record(row_key, col_key, rate):
            print(event, benchmark, variant, f"batch={batch}", f"workers={workers}", f"exec={executors}", flush=True)


def _adaptive_mean_ms(time_once, max_runs: int, min_seconds: float) -> float:
    """Time once; repeat only while the measurement is short, up to ``max_runs``. Return the mean in ms."""
    samples = [time_once()]
    while len(samples) < max_runs and sum(samples) / len(samples) < min_seconds * 1000:
        samples.append(time_once())
    return sum(samples) / len(samples)


def scan_substruct(args, output: Path, done: dict) -> None:
    mols = load_smiles(args.smiles, args.num_mols, True, seed=args.seed)
    print(f"substruct: {len(mols)} molecules", flush=True)
    for (smarts_stem, mode), algorithm in itertools.product(SUBSTRUCT_SETS, args.algorithms):
        if args.sets and smarts_stem not in args.sets:
            continue
        queries, _ = load_smarts(str(SMARTS_DIR / f"{smarts_stem}.txt"))
        pairs = len(mols) * len(queries)
        warmup = mols[: min(2000, len(mols))]

        def run_point(
            batch, workers, prep, _executors, queries=queries, mode=mode, algorithm=algorithm, warmup=warmup
        ):
            config = SubstructSearchConfig()
            config.batchSize, config.workerThreads, config.preprocessingThreads = batch, workers, prep
            config.gpuIds = [0]
            config.algorithm = algorithm
            substruct_bench.bench_nvmolkit(warmup, queries, 1, mode, config)
            return _adaptive_mean_ms(
                lambda: substruct_bench.bench_nvmolkit(mols, queries, 1, mode, config)[0], args.runs, args.min_seconds
            )

        _scan(args, "substruct", f"{smarts_stem}:{algorithm}", pairs, run_point, output, done)


class _Skipped(Exception):
    """A grid point excluded before running (recorded with status ``skipped``)."""


def scan_mcs(args, output: Path, done: dict) -> None:
    row_pairs = {k: int(v) for k, v in (item.split("=", 1) for item in args.pairs_per_row)}
    bench_args = mcs_bench._build_parser().parse_args(
        [
            "--smiles",
            args.smiles,
            "--num_mols",
            str(args.num_mols),
            "--num_pairs",
            str(args.num_pairs),
            "--seed",
            str(args.seed),
        ]
    )
    mols = load_smiles(args.smiles, bench_args.num_mols, bench_args.sanitize, seed=args.seed)
    mols = mcs_bench._filter_molecules(mols, bench_args.max_atoms, bench_args.max_bonds)
    print(f"mcs: {len(mols)} molecules", flush=True)
    for name, atom_compare, bond_compare, ring_only in MCS_ROWS:
        if args.sets and name not in args.sets:
            continue
        bench_args.atom_compare, bench_args.bond_compare = atom_compare, bond_compare
        bench_args.ring_matches_ring_only = ring_only
        num_pairs = row_pairs.get(name, args.num_pairs)
        bench_args.num_pairs = num_pairs
        config_row = mcs_bench._normalize_config_row({}, bench_args)
        pairs = mcs_bench.sample_pairs(len(mols), num_pairs, args.seed)
        max_in_flight = num_pairs // args.min_fills if args.min_fills > 0 else None
        warmup = pairs[: min(2000, len(pairs))]

        def run_point(
            batch, workers, prep, executors, config_row=config_row, pairs=pairs, warmup=warmup, cap=max_in_flight
        ):
            if cap is not None and batch * workers * executors > cap:
                raise _Skipped(f"in-flight {batch * workers * executors} > {cap}")
            config = MCSConfig(
                batchSize=batch,
                workerThreads=workers,
                preprocessingThreads=prep,
                executorsPerRunner=executors,
                gpuIds=[0],
            )
            mcs_bench.bench_nvmolkit_mcs(mols, warmup, 1, config_row, config)
            elapsed_ms, _, _ = mcs_bench.bench_nvmolkit_mcs(mols, pairs, args.runs, config_row, config)
            return elapsed_ms

        _scan(args, "mcs", name, len(pairs), run_point, output, done)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", choices=("substruct", "mcs"))
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sets", nargs="+", default=[], help="Restrict to these SMARTS stems or MCS row names")
    parser.add_argument("--algorithms", nargs="+", choices=("gsi", "dfs"), default=["gsi", "dfs"])
    parser.add_argument("--num_mols", type=int, default=0, help="Molecules (default: 1250000 substruct, 10000 MCS)")
    parser.add_argument("--num_pairs", type=int, default=230000, help="MCS pairs per scan point")
    parser.add_argument(
        "--pairs_per_row",
        nargs="+",
        default=[],
        metavar="NAME=PAIRS",
        help="Per-MCS-row pair counts overriding --num_pairs (lax rows are ~10x slower than strict ones)",
    )
    parser.add_argument(
        "--min_fills",
        type=int,
        default=0,
        help="MCS: skip configurations whose in-flight pairs (batch x workers x executors) exceed "
        "num_pairs / min_fills, so every point runs several complete pipeline fills (0 disables)",
    )
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--executors", type=int, nargs="+", default=[1], help="MCS executors per runner")
    parser.add_argument("--cpu_budget", type=int, default=14, help="workers + preprocessing threads")
    parser.add_argument("--runs", type=int, default=1, help="Maximum timed runs per point")
    parser.add_argument(
        "--min_seconds",
        type=float,
        default=10.0,
        help="Repeat a point (up to --runs) only while its mean run is shorter than this",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prune_drop", type=float, default=0.10)
    parser.add_argument("--worker_min_gain", type=float, default=0.03)
    args = parser.parse_args()
    args.batch_sizes, args.workers, args.executors = (
        sorted(args.batch_sizes),
        sorted(args.workers),
        sorted(args.executors),
    )
    if max(args.workers) >= args.cpu_budget:
        parser.error("every --workers value must leave at least one preprocessing thread within --cpu_budget")
    if args.num_mols <= 0:
        args.num_mols = 1250000 if args.target == "substruct" else 10000
    output = Path(args.output)
    done = _done(output)
    (scan_substruct if args.target == "substruct" else scan_mcs)(args, output, done)
    print("SCAN DONE", time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
