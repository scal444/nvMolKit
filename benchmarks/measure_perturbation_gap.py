# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure starting MMFF energy gap for various Cartesian perturbation sigmas.

Loads molecules from a prepared dataset (output of fire_optuna_prep.py), takes the
RDKit-minimized reference geometries (by re-minimizing the perturbed conformers with
RDKit), then for each test sigma applies fresh per-atom Gaussian noise and computes
``(E_perturbed - E_ref) / n_atoms`` for every conformer. Reports distribution stats.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Output of fire_optuna_prep.py.")
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05, 0.1, 0.2],
    )
    parser.add_argument("--num-mols", type=int, default=200, help="Subsample for speed.")
    parser.add_argument("--rdkit-mmff-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_mols(data_dir: Path, num_mols: int) -> list[Chem.Mol]:
    sdf_path = data_dir / "perturbed.sdf"
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    by_idx: dict[int, Chem.Mol] = {}
    for mol in supplier:
        if mol is None:
            continue
        idx = int(mol.GetProp("_MolIndex"))
        if idx in by_idx:
            by_idx[idx].AddConformer(mol.GetConformer(), assignId=True)
        else:
            by_idx[idx] = mol
    mols = [by_idx[i] for i in sorted(by_idx)]
    return mols[:num_mols]


def reminimize_to_reference(mols: list[Chem.Mol], max_iters: int) -> dict[tuple[int, int], tuple[np.ndarray, float]]:
    """Returns map from (mol_idx, conf_id) -> (positions, ref_energy)."""
    reference: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
    for mol_idx, mol in enumerate(mols):
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iters, mmffVariant="MMFF94")
        for conf, (status, energy) in zip(mol.GetConformers(), results):
            if status != 0:
                continue
            positions = np.asarray(conf.GetPositions(), dtype=float)
            reference[(mol_idx, conf.GetId())] = (positions.copy(), float(energy))
    return reference


def perturb_and_score(
    mols: list[Chem.Mol],
    reference: dict[tuple[int, int], tuple[np.ndarray, float]],
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Returns array of (E_perturbed - E_ref) / n_atoms per (mol, conf)."""
    gaps: list[float] = []
    for mol_idx, mol in enumerate(mols):
        n_atoms = mol.GetNumAtoms()
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        for conf in mol.GetConformers():
            key = (mol_idx, conf.GetId())
            if key not in reference:
                continue
            ref_positions, ref_energy = reference[key]
            noise = rng.normal(loc=0.0, scale=sigma, size=ref_positions.shape)
            new_positions = ref_positions + noise
            for atom_idx in range(n_atoms):
                conf.SetAtomPosition(atom_idx, new_positions[atom_idx].tolist())
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf.GetId())
            if ff is None:
                continue
            energy = ff.CalcEnergy()
            if not np.isfinite(energy):
                continue
            gaps.append((energy - ref_energy) / n_atoms)
    return np.asarray(gaps)


def summarize(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.num_mols} mols from {args.data_dir}...")
    mols = load_mols(args.data_dir, args.num_mols)

    print("Re-minimizing with RDKit MMFF to establish reference geometry...")
    reference = reminimize_to_reference(mols, args.rdkit_mmff_iters)
    n_systems = sum(mol.GetNumConformers() for mol in mols)
    print(f"Reference established for {len(reference)} / {n_systems} systems.")

    results: dict[float, dict[str, float]] = {}
    for sigma in args.sigmas:
        gaps = perturb_and_score(mols, reference, sigma, rng)
        stats = summarize(gaps)
        results[sigma] = stats
        print(f"sigma={sigma:.3f}  -> {json.dumps(stats)}")

    print("\n--- Summary table ---")
    print(f"{'sigma':>8} {'mean':>12} {'median':>12} {'p90':>12} {'p99':>12} {'max':>14}")
    for sigma, stats in results.items():
        print(f"{sigma:>8.3f} {stats['mean']:>12.3f} {stats['median']:>12.3f} {stats['p90']:>12.3f} "
              f"{stats['p99']:>12.3f} {stats['max']:>14.3f}")


if __name__ == "__main__":
    main()
