# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimize conformers from an SDF using RDKit MMFF and save energies and minimized structures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RDKit MMFF minimization on conformers in an SDF file.",
    )
    parser.add_argument("input_sdf", type=Path, help="Path to the input SDF containing conformers.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path prefix (default: derive from input SDF path).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=1000,
        help="Maximum MMFF iterations per conformer (default: 1000).",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="Number of threads for MMFF optimization (default: 0 for RDKit default).",
    )
    return parser.parse_args()


def load_conformers(sdf_path: Path) -> list[Chem.Mol]:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [mol for mol in supplier if mol is not None]
    if not mols:
        raise ValueError(f"No molecules found in {sdf_path}.")
    return mols


def compute_initial_energies(mol: Chem.Mol) -> np.ndarray:
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    energies = np.full(mol.GetNumConformers(), np.nan, dtype=float)
    if props is None:
        return energies
    for idx, conf in enumerate(mol.GetConformers()):
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf.GetId())
        if ff is None:
            continue
        energies[idx] = ff.CalcEnergy()
    return energies


def minimize_molecule(
    mol: Chem.Mol,
    max_iters: int,
    num_threads: int,
) -> np.ndarray:
    results = AllChem.MMFFOptimizeMoleculeConfs(
        mol,
        maxIters=max_iters,
        numThreads=num_threads,
        mmffVariant="MMFF94",
    )
    energies = np.full(mol.GetNumConformers(), np.nan, dtype=float)
    for idx, (_, energy) in enumerate(results):
        if energy is None:
            continue
        energies[idx] = energy
    return energies


def process_conformers(
    mols: list[Chem.Mol],
    max_iters: int,
    num_threads: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[Chem.Mol]]:
    initial: list[np.ndarray] = []
    minimized: list[np.ndarray] = []
    minimized_mols: list[Chem.Mol] = []

    for mol in mols:
        init = compute_initial_energies(mol)
        final = minimize_molecule(mol, max_iters, num_threads)
        initial.append(init)
        minimized.append(final)
        minimized_mols.append(mol)

    return initial, minimized, minimized_mols


def write_minimized_sdf(mols: list[Chem.Mol], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_path))
    if writer is None:
        raise RuntimeError(f"Unable to create SDWriter for {output_path}.")
    for mol in mols:
        for conf in mol.GetConformers():
            writer.write(mol, confId=conf.GetId())
    writer.close()


def main() -> None:
    args = parse_args()
    mols = load_conformers(args.input_sdf)

    output_prefix = (
        args.output_prefix
        if args.output_prefix is not None
        else args.input_sdf.parent / args.input_sdf.stem
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    initial, minimized, minimized_mols = process_conformers(
        mols,
        args.max_iters,
        args.num_threads,
    )

    if not initial:
        raise RuntimeError("No valid conformers found for minimization.")

    initial_array = np.concatenate(initial)
    minimized_array = np.concatenate(minimized)

    initial_path = output_prefix.parent / f"{output_prefix.name}_initial_energies.npy"
    final_path = output_prefix.parent / f"{output_prefix.name}_final_energies.npy"
    minimized_sdf = output_prefix.parent / f"{output_prefix.name}_minimized.sdf"

    np.save(initial_path, initial_array)
    np.save(final_path, minimized_array)
    write_minimized_sdf(minimized_mols, minimized_sdf)

    print(
        f"Processed {len(initial_array)} conformers. Saved minimized structures to {minimized_sdf}."
    )


if __name__ == "__main__":
    main()


