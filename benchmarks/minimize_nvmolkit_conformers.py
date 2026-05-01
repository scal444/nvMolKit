# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimize conformers from an SDF using nvmolkit MMFF and save energies and minimized structures.

Currently uses the FIRE 2.0 backend (`MMFFOptimizeMoleculesConfsFire`). For BFGS,
use `minimize_nvmolkit_conformers_bfgs.py` (or call `MMFFOptimizeMoleculesConfs`
directly) - this script is FIRE-specific so the FIRE-only knobs (alpha/dt/half-step
back/ABC) can be exposed without overloading the CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from nvmolkit.mmffOptimization import FireOptions, MMFFOptimizeMoleculesConfsFire


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run nvmolkit MMFF/FIRE 2.0 minimization on conformers in an SDF file.",
    )
    parser.add_argument("input_sdf", type=Path, help="Path to the input SDF containing conformers.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path prefix (default: derive from input SDF path).",
    )
    parser.add_argument("--max-iters", type=int, default=1000, help="Maximum FIRE iterations (default: 1000).")
    parser.add_argument(
        "--gradtol",
        type=float,
        default=1e-4,
        help="Gradient convergence tolerance for FIRE (default: 1e-4).",
    )
    parser.add_argument("--mass-weighting", action="store_true", help="Enable mass weighting in the FIRE kick.")
    parser.add_argument("--use-abc", action="store_true", help="Enable the ABC-FIRE mixer correction.")
    parser.add_argument(
        "--no-half-step-back",
        action="store_true",
        help="Disable the half-step-back behavior on negative-power steps. ASE FIRE2 always half-steps; this is for ablation.",
    )
    parser.add_argument("--dt-init", type=float, default=0.001, help="Initial dt in picoseconds (default: 0.001).")
    parser.add_argument("--dt-max-factor", type=float, default=10.0, help="dtmax / dtinit (default: 10.0).")
    parser.add_argument("--dt-min-factor", type=float, default=0.002, help="dtmin / dtinit (default: 0.002).")
    parser.add_argument(
        "--n-min-for-increase",
        type=int,
        default=20,
        help="Number of consecutive positive-power steps before dt grows (default: 20, ASE FIRE2 default).",
    )
    parser.add_argument("--alpha-init", type=float, default=0.25, help="Initial mixer alpha (default: 0.25).")
    parser.add_argument(
        "--max-step",
        type=float,
        default=0.2,
        help="Maximum displacement per step in Angstroms (default: 0.2). 0 disables clipping.",
    )
    parser.add_argument(
        "--save-initial",
        action="store_true",
        help="Also compute and save initial MMFF energies before minimization.",
    )
    parser.add_argument(
        "--fire-debug-output",
        type=Path,
        default=None,
        help=(
            "Optional path to write FIRE per-iteration debug data (alphas, dt, powers, energies) as JSON. "
            "Slated for removal before PR; for diagnostic use only."
        ),
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


def fire_options_from_args(args: argparse.Namespace) -> FireOptions:
    opts = FireOptions()
    opts.gradTol = args.gradtol
    opts.useMass = args.mass_weighting
    opts.abcCorrection = args.use_abc
    opts.takeHalfStepBack = not args.no_half_step_back
    opts.dtInit = args.dt_init
    opts.dtMaxFactor = args.dt_max_factor
    opts.dtMinFactor = args.dt_min_factor
    opts.nMinForIncrease = args.n_min_for_increase
    opts.alphaInit = args.alpha_init
    opts.dMax = args.max_step
    return opts


def minimize_molecules(
    mols: list[Chem.Mol],
    max_iters: int,
    fire_opts: FireOptions,
    collect_fire_debug: bool,
) -> tuple[list[np.ndarray], list[list[dict[str, list[float]]]] | None]:
    fire_debug: list[list[dict[str, list[float]]]] | None = [] if collect_fire_debug else None
    energies_nested = MMFFOptimizeMoleculesConfsFire(
        mols,
        maxIters=max_iters,
        fireOptions=fire_opts,
        fireDebugOutput=fire_debug,
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
    return per_mol, fire_debug


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
        args.output_prefix if args.output_prefix is not None else args.input_sdf.parent / args.input_sdf.stem
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    initial_per_mol: list[np.ndarray] = []
    if args.save_initial:
        for mol in mols:
            initial_per_mol.append(compute_initial_energies(mol))

    fire_opts = fire_options_from_args(args)
    collect_fire_debug = args.fire_debug_output is not None

    minimized_per_mol, fire_debug = minimize_molecules(mols, args.max_iters, fire_opts, collect_fire_debug)

    if args.fire_debug_output is not None:
        if fire_debug is None:
            raise RuntimeError("FIRE debug output was not collected despite request.")
        args.fire_debug_output.parent.mkdir(parents=True, exist_ok=True)
        with args.fire_debug_output.open("w", encoding="utf-8") as handle:
            json.dump(fire_debug, handle)

    minimized_array = np.concatenate(minimized_per_mol)
    final_path = output_prefix.parent / f"{output_prefix.name}_final_energies_nvm.npy"
    np.save(final_path, minimized_array)
    if initial_per_mol:
        initial_array = np.concatenate(initial_per_mol)
        initial_path = output_prefix.parent / f"{output_prefix.name}_initial_energies_nvm.npy"
        np.save(initial_path, initial_array)

    minimized_sdf = output_prefix.parent / f"{output_prefix.name}_minimized_nvm.sdf"
    write_minimized_sdf(mols, minimized_sdf)

    print(f"Processed {len(minimized_array)} conformers. Saved minimized structures to {minimized_sdf}.")


if __name__ == "__main__":
    main()
