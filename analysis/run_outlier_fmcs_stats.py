#!/usr/bin/env python
"""Collect fMCS diagnostic counters for the known slow Enamine pair."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rdkit import Chem

from nvmolkit.mcs import findMCS


SMILES_A = "C[C@@]12CCC[C@H]1N(C(=O)C1=CC3=C(C=N1)NC=C3)CCN(C(=O)C1=CN=C3CCCCN13)C2"
SMILES_B = "CON1CCC(C(=O)N2CC(N3CCN(C(=O)C45CCCCC4CCC5)CC3)C2)CC1"
BLOCK_SIZES = (64, 128, 256, 512)


def main() -> None:
    out_dir = Path("/home/kboyd/omg/repos/nvmolkit/analysis/mcs_1k_timing_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "outlier_pair_851_nvmolkit_stats_by_block.json"
    csv_path = out_dir / "outlier_pair_851_nvmolkit_stats_by_block.csv"

    mols = [Chem.MolFromSmiles(SMILES_A), Chem.MolFromSmiles(SMILES_B)]
    rows = []
    for block_size in BLOCK_SIZES:
        result = findMCS(
            mols,
            mode="pairs",
            pairs=[(0, 1)],
            allow_rdkit_fallback=False,
            timeout_seconds=0,
            block_size=block_size,
            worker_threads=1,
            preprocessing_threads=1,
            executors_per_runner=1,
            gpu_ids=[0],
            collect_timings=True,
            collect_stats=True,
        )
        item = result[0]
        stats = item.fmcs_stats or {}
        num_groups = block_size // 32
        phase2_iters = int(stats.get("phase2_iters", 0))
        popped = int(stats.get("popped", 0))
        row = {
            "block_size": block_size,
            "num_groups": num_groups,
            "elapsed_ms": item.elapsed_ms,
            "num_atoms": item.num_atoms,
            "num_bonds": item.num_bonds,
            "canceled": item.canceled,
            "overflowed": item.overflowed,
            "avg_popped_per_phase2_iter": popped / phase2_iters if phase2_iters else 0.0,
            "group_slot_utilization": popped / (phase2_iters * num_groups) if phase2_iters else 0.0,
        }
        row.update(stats)
        rows.append(row)

    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
