# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grid-scan nvMolKit HardwareOptions for ETKDG and MMFF/UFF minimization on one GPU.

Replaces Optuna autotuning for publication runs: every (batchSize, batchesPerGpu) point is timed on
the same fixed workload, so the selected configuration and the full throughput curve are both
recorded. The workload is sized to several complete fills of the largest configuration so that
large batches are not scored on pipeline ramp-up and tail. Results are appended per point, and an
interrupted scan resumes from the points already in the output CSV.

Example:
    python scan_hardware_options.py --smiles data.smi --output scan.csv --prep_threads 14
"""

import argparse
import csv
import itertools
import time
from pathlib import Path

import etkdg_bench
import ff_optimize_bench
import torch
from bench_utils import available_cpu_count, embed_and_jitter, load_smiles, prep_mols
from rdkit.Chem import AllChem

from nvmolkit.types import HardwareOptions

BENCHMARKS = ("etkdg", "mmff_bfgs", "mmff_fire", "uff_bfgs", "uff_fire")
FIELDS = [
    "benchmark",
    "precision",
    "batch_size",
    "batches_per_gpu",
    "prep_threads",
    "num_mols",
    "confs_per_mol",
    "time_ms",
    "confs_per_second",
    "status",
]


def _done_points(path: Path) -> dict[tuple, float | None]:
    """Map each recorded (benchmark, precision, batch, batches_per_gpu) point to its throughput."""
    if not path.exists():
        return {}
    with open(path) as handle:
        return {
            (r["benchmark"], r["precision"], int(r["batch_size"]), int(r["batches_per_gpu"])): (
                float(r["confs_per_second"]) if r["confs_per_second"] else None
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--precisions", nargs="+", choices=("single", "full"), default=["single", "full"])
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--batches_per_gpu", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--prep_threads", type=int, default=14, help="nvMolKit CPU threads in the timed call")
    parser.add_argument("--prep_workers", type=int, default=0, help="Untimed conformer-generation workers (0 = all)")
    parser.add_argument("--confs_per_mol", type=int, default=200)
    parser.add_argument("--fills", type=int, default=4, help="Complete fills of the largest configuration")
    parser.add_argument(
        "--num_mols",
        type=int,
        default=0,
        help="Explicit workload size in molecules, overriding --fills (to extend an earlier scan on the same inputs)",
    )
    parser.add_argument("--max_iters", type=int, default=200, help="Force-field iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1, help="Timed runs per point; the mean is recorded")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Time points even if already recorded (appends new rows; disables pruning from earlier results)",
    )
    parser.add_argument(
        "--prune_drop",
        type=float,
        default=0.10,
        help="Stop a (benchmark, precision, batches_per_gpu) row once a point falls this fraction below the row's "
        "best; batch sizes are scanned in ascending order, so the row is past its peak (0 disables)",
    )
    parser.add_argument(
        "--bpg_min_gain",
        type=float,
        default=0.03,
        help="For each batch size, stop raising batches_per_gpu once a step gains less than this fraction over the "
        "previous batches_per_gpu at that batch size (0 disables)",
    )
    args = parser.parse_args()
    args.batch_sizes = sorted(args.batch_sizes)
    args.batches_per_gpu = sorted(args.batches_per_gpu)

    largest = max(args.batch_sizes) * max(args.batches_per_gpu)
    num_mols = args.num_mols if args.num_mols > 0 else -(-args.fills * largest // args.confs_per_mol)
    output = Path(args.output)
    done = {} if args.rerun else _done_points(output)
    print(
        f"Workload: {num_mols} molecules x {args.confs_per_mol} conformers "
        f"({args.fills} fills of {largest} in-flight conformers)",
        flush=True,
    )

    raw = load_smiles(args.smiles, int(num_mols * 1.5), True, seed=args.seed, keep_buffer=True)
    prepared = prep_mols(raw)
    workers = args.prep_workers if args.prep_workers > 0 else available_cpu_count()
    inputs: dict[str, list] = {}
    if "etkdg" in args.benchmarks:
        inputs["etkdg"] = prepared[:num_mols]
    for ff, has_params in (("mmff", AllChem.MMFFHasAllMoleculeParams), ("uff", AllChem.UFFHasAllMoleculeParams)):
        if any(b.startswith(ff) for b in args.benchmarks):
            mols = [m for m in prepared if has_params(m)][:num_mols]
            inputs[ff] = embed_and_jitter(mols, args.confs_per_mol, seed=args.seed, num_workers=workers)
    for name, mols in inputs.items():
        print(f"  {name}: {len(mols)} molecules", flush=True)
    params = etkdg_bench._build_etkdg_params(-1, 1, args.seed)

    row_best: dict[tuple, float] = {}
    pruned: set[tuple] = set()
    # Along batches_per_gpu: last throughput seen at each (benchmark, precision, batch) and the batch sizes whose
    # batches_per_gpu axis has plateaued.
    prev_bpg_rate: dict[tuple, float] = {}
    bpg_plateaued: set[tuple] = set()

    def record_bpg(col_key: tuple, rate: float | None) -> bool:
        """Return True when raising batches_per_gpu further at this batch size is not worth scanning."""
        previous = prev_bpg_rate.get(col_key)
        if rate is not None:
            prev_bpg_rate[col_key] = rate
        if args.bpg_min_gain <= 0 or previous is None:
            return False
        return rate is None or rate < (1.0 + args.bpg_min_gain) * previous

    def record(row_key: tuple, rate: float | None) -> bool:
        """Track the row's best throughput; return True when the row is past its peak."""
        if rate is None:
            return args.prune_drop > 0  # OOM or error: larger batches will not do better
        row_best[row_key] = max(rate, row_best.get(row_key, 0.0))
        return args.prune_drop > 0 and rate < (1.0 - args.prune_drop) * row_best[row_key]

    for bench, precision in itertools.product(args.benchmarks, args.precisions):
        for bpg, batch in itertools.product(args.batches_per_gpu, args.batch_sizes):
            row_key = (bench, precision, bpg)
            col_key = (bench, precision, batch)
            if row_key in pruned or col_key in bpg_plateaued:
                continue
            if (bench, precision, batch, bpg) in done:
                rate = done[(bench, precision, batch, bpg)]
                if record(row_key, rate):
                    pruned.add(row_key)
                if record_bpg(col_key, rate):
                    bpg_plateaued.add(col_key)
                continue
            hw = HardwareOptions(
                batchSize=batch, batchesPerGpu=bpg, preprocessingThreads=args.prep_threads, gpuIds=[0]
            )
            mols = inputs["etkdg" if bench == "etkdg" else bench.split("_")[0]]
            row = {
                "benchmark": bench,
                "precision": precision,
                "batch_size": batch,
                "batches_per_gpu": bpg,
                "prep_threads": args.prep_threads,
                "num_mols": len(mols),
                "confs_per_mol": args.confs_per_mol,
                "time_ms": "",
                "confs_per_second": "",
                "status": "ok",
            }
            try:
                if bench == "etkdg":
                    timing, _ = etkdg_bench.bench_nvmolkit(
                        mols, params, args.confs_per_mol, -1, hw, args.runs, True, precision
                    )
                    elapsed_ms = timing.mean_ms
                else:
                    ff, kind = bench.split("_")
                    elapsed_ms, _, _ = ff_optimize_bench.bench_nvmolkit(
                        mols, ff, kind.upper(), args.max_iters, hw, args.runs, True, precision
                    )
                row["time_ms"] = round(elapsed_ms, 1)
                row["confs_per_second"] = round(len(mols) * args.confs_per_mol / (elapsed_ms / 1000), 1)
            except torch.cuda.OutOfMemoryError:
                row["status"] = "oom"
            except RuntimeError as error:
                row["status"] = "oom" if "out of memory" in str(error).lower() else f"error: {error}"[:200]
            torch.cuda.empty_cache()
            _append(output, row)
            print(
                "RESULT",
                bench,
                precision,
                f"batch={batch}",
                f"bpg={bpg}",
                f"confs/s={row['confs_per_second']}",
                f"t={row['time_ms']}ms",
                row["status"],
                flush=True,
            )
            rate = float(row["confs_per_second"]) if row["confs_per_second"] != "" else None
            if record(row_key, rate):
                pruned.add(row_key)
                print("PRUNED", bench, precision, f"bpg={bpg}", f"after batch={batch}", flush=True)
            if record_bpg(col_key, rate):
                bpg_plateaued.add(col_key)
                print("PLATEAU", bench, precision, f"batch={batch}", f"at bpg={bpg}", flush=True)
    print("SCAN DONE", time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
