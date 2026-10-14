# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Select the best scanned nvMolKit execution settings and save them for reuse.

Reads scan CSVs written by ``scan_hardware_options.py`` (ETKDG and MMFF/UFF) and
``scan_search_options.py`` (substructure search and MCS), picks the highest-throughput successful
point for every API workload, and writes a JSON file that ``run_publication_benchmarks.sh`` loads.
Files are keyed by GPU and CPU budget, e.g. ``hardware_configs/gb300_cpu14.json``, so configurations
scanned on one machine are reused there and never silently applied to different hardware.

Keys:
    ``etkdg:<precision>``, ``mmff_bfgs:<precision>``, ``mmff_fire:<precision>`` -> batchSize, batchesPerGpu
    ``mcs:<row>``                                     -> batchSize, workers, preprocessingThreads, executors
    ``substruct:<smarts stem>:<algorithm>``           -> batchSize, workers, preprocessingThreads

Rows from several CSVs can describe the same point (for example a single-run scan point and a later
multi-run re-timing). When ``--prefer`` CSVs are given, their points replace scan points for the same
key and configuration.

Example:
    python select_scan_configs.py --gpu "NVIDIA GB300" --cpu_budget 14 --nvmolkit_version 0.7.0 \\
        --hardware_scans ff_scan.csv --search_scans substruct_scan.csv mcs_scan.csv \\
        --prefer retime.csv --output hardware_configs/gb300_cpu14.json
"""

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def gpu_slug(gpu_name: str) -> str:
    """``'NVIDIA GB300'`` -> ``'gb300'``."""
    return re.sub(r"[^a-z0-9]+", "_", gpu_name.lower().replace("nvidia", "")).strip("_")


def _rows(path: str) -> list[dict]:
    with open(path) as handle:
        return [dict(row, _source=Path(path).name) for row in csv.DictReader(handle)]


def _hardware_points(paths: list[str]) -> dict[tuple, dict]:
    points = {}
    for path in paths:
        for r in _rows(path):
            if r.get("status") != "ok" or not r.get("confs_per_second"):
                continue
            key = f"{r['benchmark']}:{r['precision']}"
            config = {"batchSize": int(r["batch_size"]), "batchesPerGpu": int(r["batches_per_gpu"])}
            points[(key, tuple(config.values()))] = {
                "config": config,
                "throughput": float(r["confs_per_second"]),
                "unit": "conformers/s",
                "workload": {"num_mols": int(r["num_mols"]), "confs_per_mol": int(r["confs_per_mol"])},
                "source": r["_source"],
            }
    return points


def _search_points(paths: list[str]) -> dict[tuple, dict]:
    points = {}
    for path in paths:
        for r in _rows(path):
            if r.get("status") != "ok" or not r.get("items_per_second"):
                continue
            config = {
                "batchSize": int(r["batch_size"]),
                "workers": int(r["workers"]),
                "preprocessingThreads": int(r["prep_threads"]),
            }
            if r["benchmark"] == "mcs":
                config["executors"] = int(r["executors"])
            key = f"{r['benchmark']}:{r['variant']}"
            points[(key, tuple(config.values()))] = {
                "config": config,
                "throughput": float(r["items_per_second"]),
                "unit": "pairs/s",
                "workload": {"items": int(r["num_items"])},
                "source": r["_source"],
            }
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", required=True, help="GPU name as nvidia-smi reports it, e.g. 'NVIDIA GB300'")
    parser.add_argument("--cpu_budget", type=int, required=True)
    parser.add_argument("--nvmolkit_version", required=True)
    parser.add_argument("--hardware_scans", nargs="*", default=[], help="scan_hardware_options.py CSVs")
    parser.add_argument("--search_scans", nargs="*", default=[], help="scan_search_options.py CSVs")
    parser.add_argument("--prefer", nargs="*", default=[], help="Multi-run re-timing CSVs that override scan points")
    parser.add_argument("--output", default=None, help="Default: hardware_configs/<gpu>_cpu<budget>.json")
    args = parser.parse_args()

    points = {}
    points.update(_hardware_points(args.hardware_scans))
    points.update(_search_points(args.search_scans))
    for path in args.prefer:
        with open(path) as handle:
            is_hardware = "confs_per_second" in handle.readline()
        preferred = _hardware_points([path]) if is_hardware else _search_points([path])
        points.update(preferred)

    best: dict[str, dict] = {}
    scanned: dict[str, int] = {}
    for (key, _), point in points.items():
        scanned[key] = scanned.get(key, 0) + 1
        if key not in best or point["throughput"] > best[key]["throughput"]:
            best[key] = point
    for key, point in best.items():
        point["points_scanned"] = scanned[key]

    output = (
        Path(args.output)
        if args.output
        else (Path(__file__).resolve().parent / "hardware_configs" / f"{gpu_slug(args.gpu)}_cpu{args.cpu_budget}.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "gpu": args.gpu,
        "cpu_budget": args.cpu_budget,
        "nvmolkit_version": args.nvmolkit_version,
        "generated": datetime.now(timezone.utc).date().isoformat(),
        "selection": "highest-throughput successful scan point per key; re-timed points override scan points",
        "configs": dict(sorted(best.items())),
    }
    output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"Wrote {len(best)} configurations to {output}")
    for key, point in sorted(best.items()):
        print(f"  {key:52s} {point['config']}  {point['throughput']:,.0f} {point['unit']}")


if __name__ == "__main__":
    main()
