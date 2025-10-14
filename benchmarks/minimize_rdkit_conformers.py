# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimize conformers from an SDF using RDKit MMFF and save energies and minimized structures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


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
    parser.add_argument(
        "--fire-debug-output",
        type=Path,
        default=None,
        help=(
            "Optional path to write per-step RDKit MMFF debug information as JSON. "
            "Collecting debug information incurs significant overhead."
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


def _clone_molecule_with_positions(mol: Chem.Mol, positions: list[Chem.rdGeometry.Point3D]) -> tuple[Chem.Mol, Chem.Conformer]:
    clone = Chem.Mol(mol)
    clone.RemoveAllConformers()
    conf = Chem.Conformer(len(positions))
    for atom_idx, pos in enumerate(positions):
        conf.SetAtomPosition(atom_idx, pos)
    clone.AddConformer(conf, assignId=True)
    return clone, conf


def _positions_from_conformer(conf: Chem.Conformer) -> list[list[float]]:
    coords: list[list[float]] = []
    for atom_idx in range(conf.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        coords.append([float(pos.x), float(pos.y), float(pos.z)])
    return coords


def _apply_positions_to_mol(mol: Chem.Mol, positions: list[list[list[float]]]) -> None:
    if mol.GetNumConformers() != len(positions):
        raise ValueError("Mismatch between conformer count and provided positions")
    for conf, conf_positions in zip(mol.GetConformers(), positions):
        if conf.GetNumAtoms() != len(conf_positions):
            raise ValueError("Mismatch between atom count and provided coordinates")
        for atom_idx, (x, y, z) in enumerate(conf_positions):
            conf.SetAtomPosition(atom_idx, Chem.rdGeometry.Point3D(x, y, z))


def minimize_molecule_with_debug(
    mol: Chem.Mol,
    max_iters: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, list[float]]], list[list[list[float]]]]:
    num_confs = mol.GetNumConformers()
    initial = np.full(num_confs, np.nan, dtype=float)
    final = np.full(num_confs, np.nan, dtype=float)
    debug_entries: list[dict[str, list[float]]] = []
    final_positions: list[list[list[float]]] = []

    for conf_idx, conf in enumerate(mol.GetConformers()):
        conf_debug: dict[str, list[float]] = {"energies": []}

        original_positions = [conf.GetAtomPosition(i) for i in range(conf.GetNumAtoms())]

        props_initial = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        ff_initial = AllChem.MMFFGetMoleculeForceField(mol, props_initial, confId=conf.GetId())
        if ff_initial is None:
            debug_entries.append(conf_debug)
            continue

        energy0 = float(ff_initial.CalcEnergy())
        conf_debug["energies"].append(energy0)
        initial[conf_idx] = energy0

        last_energy = energy0
        final_conf: Chem.Conformer | None = None

        for step in range(1, max_iters + 1):
            working_mol, working_conf = _clone_molecule_with_positions(mol, original_positions)
            working_conf_id = working_conf.GetId()

            props = AllChem.MMFFGetMoleculeProperties(working_mol, mmffVariant="MMFF94")
            if props is None:
                conf_debug["energies"].append(float("nan"))
                break

            status = AllChem.MMFFOptimizeMolecule(
                working_mol,
                maxIters=step,
                confId=working_conf_id,
                mmffVariant="MMFF94",
            )

            ff_step = AllChem.MMFFGetMoleculeForceField(working_mol, props, confId=working_conf_id)
            if ff_step is None:
                last_energy = float("nan")
                conf_debug["energies"].append(last_energy)
                break

            last_energy = float(ff_step.CalcEnergy())
            conf_debug["energies"].append(last_energy)

            final_conf = working_conf

            if status == 0 or status == -1:
                break

        if final_conf is not None:
            final_positions.append(_positions_from_conformer(final_conf))
        else:
            final_positions.append([[float(pos.x), float(pos.y), float(pos.z)] for pos in original_positions])
        final[conf_idx] = last_energy
        debug_entries.append(conf_debug)

    return initial, final, debug_entries, final_positions


def _process_debug_task(args: tuple[int, str, int]) -> tuple[int, list[float], list[float], list[dict[str, list[float]]], list[list[list[float]]]]:
    index, mol_block, max_iters = args
    mol = Chem.MolFromMolBlock(mol_block, sanitize=True, removeHs=False, strictParsing=False)
    if mol is None:
        raise ValueError("Failed to deserialize molecule for debug processing")
    init, final, debug_entries, positions = minimize_molecule_with_debug(mol, max_iters)
    return index, init.tolist(), final.tolist(), debug_entries, positions


def process_conformers(
    mols: list[Chem.Mol],
    max_iters: int,
    num_threads: int,
    collect_debug: bool,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[Chem.Mol],
    list[list[dict[str, list[float]]]] | None,
]:
    initial: list[np.ndarray] = []
    minimized: list[np.ndarray] = []
    minimized_mols: list[Chem.Mol] = []
    debug_data: list[list[dict[str, list[float]]]] | None = [None] * len(mols) if collect_debug else None

    if collect_debug:
        tasks = [
            (
                idx,
                Chem.MolToMolBlock(mol, includeStereo=True, forceV3000=False),
                max_iters,
            )
            for idx, mol in enumerate(mols)
        ]
        worker_count = num_threads if num_threads > 0 else None
        results = process_map(
            _process_debug_task,
            tasks,
            desc="Molecules",
            chunksize=1,
            max_workers=worker_count,
        )
        sorted_initial: list[np.ndarray | None] = [None] * len(mols)
        sorted_final: list[np.ndarray | None] = [None] * len(mols)
        for index, init_list, final_list, per_conf_debug, positions in results:
            if debug_data is None:
                raise RuntimeError("collect_debug=True but debug_data list was not initialized")
            sorted_initial[index] = np.array(init_list, dtype=float)
            sorted_final[index] = np.array(final_list, dtype=float)
            debug_data[index] = per_conf_debug
            _apply_positions_to_mol(mols[index], positions)
        initial = [arr if arr is not None else np.array([], dtype=float) for arr in sorted_initial]
        minimized = [arr if arr is not None else np.array([], dtype=float) for arr in sorted_final]
        minimized_mols = mols
    else:
        for mol in tqdm(mols, desc="Molecules"):
            init = compute_initial_energies(mol)
            final = minimize_molecule(mol, max_iters, num_threads)
            initial.append(init)
            minimized.append(final)
            minimized_mols.append(mol)

    return initial, minimized, minimized_mols, debug_data


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

    collect_debug = args.fire_debug_output is not None
    initial, minimized, minimized_mols, debug_data = process_conformers(
        mols,
        args.max_iters,
        args.num_threads,
        collect_debug,
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

    if args.fire_debug_output is not None:
        if debug_data is None:
            raise RuntimeError("Debug data was not collected despite fire_debug_output being specified.")
        args.fire_debug_output.parent.mkdir(parents=True, exist_ok=True)
        with args.fire_debug_output.open("w", encoding="utf-8") as handle:
            json.dump(debug_data, handle)

    print(
        f"Processed {len(initial_array)} conformers. Saved minimized structures to {minimized_sdf}."
    )


if __name__ == "__main__":
    main()


