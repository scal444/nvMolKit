# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimize conformers from an SDF using nvmolkit MMFF and save energies and minimized structures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run nvmolkit MMFF minimization on conformers in an SDF file.",
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
        "--gradtol",
        type=float,
        default=1e-4,
        help="Gradient convergence tolerance for the FIRE optimizer (default: 1e-4).",
    )
    parser.add_argument(
        "--mass-weighting",
        action="store_true",
        help="Enable mass weighting during minimization.",
    )
    parser.add_argument(
        "--save-initial",
        action="store_true",
        help="Also compute and save initial MMFF energies before minimization.",
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


def minimize_molecules(
    mols: list[Chem.Mol],
    max_iters: int,
    grad_tol: float,
    mass_weighting: bool,
) -> list[np.ndarray]:
    options: dict[str, object] = {
        "use_masses": mass_weighting,
        "grad_tol": grad_tol,
    }
    energies_nested = MMFFOptimizeMoleculesConfs(
        mols,
        maxIters=max_iters,
        optimizer_backend="FIRE",
        optimizer_options=options,
    )
    per_mol: list[np.ndarray] = []
    for mol, energies in zip(mols, energies_nested):
        num_confs = mol.GetNumConformers()
        arr = np.full(num_confs, np.nan, dtype=float)
        for idx, energy in enumerate(energies):
            if idx >= num_confs:
                break
            if energy is not None:
                arr[idx] = energy
        per_mol.append(arr)
    return per_mol


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

    initial_per_mol: list[np.ndarray] = []
    if args.save_initial:
        for mol in mols:
            initial_per_mol.append(compute_initial_energies(mol))

    minimized_per_mol = minimize_molecules(
        mols,
        args.max_iters,
        args.gradtol,
        args.mass_weighting,
    )

    minimized_array = np.concatenate(minimized_per_mol)
    final_path = output_prefix.parent / f"{output_prefix.name}_final_energies_nvm.npy"
    np.save(final_path, minimized_array)
    if initial_per_mol:
        initial_array = np.concatenate(initial_per_mol)
        initial_path = output_prefix.parent / f"{output_prefix.name}_initial_energies_nvm.npy"
        np.save(initial_path, initial_array)

    minimized_sdf = output_prefix.parent / f"{output_prefix.name}_minimized_nvm.sdf"
    write_minimized_sdf(mols, minimized_sdf)

    print(
        f"Processed {len(minimized_array)} conformers. Saved minimized structures to {minimized_sdf}."
    )


if __name__ == "__main__":
    main()


