# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run nvMolKit FIRE vs ASE FIRE2 on the full perturbed dataset.

Two checks per system:
    1. Initial MMFF energy (nvMolKit MMFF energy kernel vs RDKit MMFF94).
    2. After 1 FIRE step with identical params: per-atom max position drift and
       energy delta against ASE.

Reports distribution stats so we can see whether divergence appears immediately
(gradient kernel mismatch) or accumulates with iteration count (integrator).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE2 as ASEFire2
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.contrib.concurrent import process_map

from nvmolkit.mmffOptimization import FireOptions, MMFFOptimizeMoleculesConfsFire


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--num-mols", type=int, default=None, help="Subsample for speed.")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON dump of per-system stats.")
    parser.add_argument("--workers", type=int, default=None, help="Worker process count (default: cpu_count()).")
    return parser.parse_args()


def load_dataset(data_dir: Path, num_mols: int | None) -> list[Chem.Mol]:
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
    if num_mols is not None:
        mols = mols[:num_mols]
    return mols


def deep_copy(mols: list[Chem.Mol]) -> list[Chem.Mol]:
    return [Chem.Mol(mol) for mol in mols]


def make_fire_options() -> FireOptions:
    options = FireOptions()
    options.useMass = False
    options.abcCorrection = False
    options.takeHalfStepBack = True
    options.dtInit = 0.1
    options.dtMaxFactor = 10.0
    options.dtMinFactor = 0.02
    options.dMax = 0.2
    options.alphaInit = 0.25
    options.alphaDecrement = 0.99
    options.timeStepIncrement = 1.1
    options.timeStepDecrement = 0.5
    options.nMinForIncrease = 20
    options.gradTol = 1e-4
    options.stuckDetectionEnabled = False
    return options


def rdkit_energy(mol: Chem.Mol, conf_id: int) -> float:
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    return float(ff.CalcEnergy())


def positions_of(mol: Chem.Mol, conf_id: int) -> np.ndarray:
    return np.asarray(mol.GetConformer(conf_id).GetPositions(), dtype=float)


class _RdkitMmffCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, mol: Chem.Mol, conf_id: int) -> None:
        super().__init__()
        self._mol = mol
        self._conf_id = conf_id
        self._props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes) -> None:  # noqa: D401
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        conf = self._mol.GetConformer(self._conf_id)
        for idx in range(self._mol.GetNumAtoms()):
            conf.SetAtomPosition(idx, positions[idx].tolist())
        ff = AllChem.MMFFGetMoleculeForceField(self._mol, self._props, confId=self._conf_id)
        energy = ff.CalcEnergy()
        grad = np.asarray(ff.CalcGrad()).reshape(-1, 3)
        self.results = {"energy": float(energy), "forces": -grad}


def _ase_worker(payload: dict) -> dict:
    """Worker for tqdm.process_map. Picklable inputs/outputs only.

    Receives a dict containing: mol_idx, conf_id, n_atoms, molblock (RDKit Mol bytes),
    fire kwargs (already extracted from FireOptions), steps. Returns the per-system
    metrics dict the main process aggregates.
    """
    mol = Chem.MolFromMolBlock(payload["molblock"], removeHs=False)
    # The serialized molblock contains exactly one conformer (the one we want), so the
    # in-worker conf id is always 0 regardless of the original conf id.
    conf_id = 0
    steps = payload["steps"]
    fire_kwargs = payload["fire_kwargs"]
    grad_tol = payload["grad_tol"]
    nvm_init_e = payload["nvm_init_e"]
    nvm_step_e = payload["nvm_step_e"]
    nvm_pos = np.asarray(payload["nvm_step_pos"], dtype=float)

    rdkit_init = rdkit_energy(mol, conf_id)

    calc = _RdkitMmffCalculator(mol, conf_id)
    atoms = Atoms(
        symbols=[atom.GetSymbol() for atom in mol.GetAtoms()],
        positions=positions_of(mol, conf_id),
    )
    atoms.calc = calc
    optimizer = ASEFire2(atoms, logfile=None, **fire_kwargs)
    optimizer.run(fmax=grad_tol, steps=steps)
    final_positions = atoms.get_positions().copy()
    calc.calculate(atoms, ["energy"], all_changes)
    ase_e = float(calc.results["energy"])

    de_init = abs(nvm_init_e - rdkit_init)
    de_step = abs(nvm_step_e - ase_e)
    pos_drift = float(np.linalg.norm(final_positions - nvm_pos, axis=1).max())
    return {
        "mol_idx": payload["mol_idx"],
        "conf_id": conf_id,
        "n_atoms": payload["n_atoms"],
        "rdkit_init_e": rdkit_init,
        "nvm_init_e": nvm_init_e,
        "ase_step_e": ase_e,
        "nvm_step_e": nvm_step_e,
        "init_de": de_init,
        "step_de": de_step,
        "step_pos_max_drift": pos_drift,
    }


def _fire_kwargs_from_options(options: FireOptions) -> dict:
    return {
        "dt": options.dtInit,
        "maxstep": options.dMax,
        "dtmax": options.dtInit * options.dtMaxFactor,
        "dtmin": options.dtInit * options.dtMinFactor,
        "Nmin": options.nMinForIncrease,
        "astart": options.alphaInit,
        "fa": options.alphaDecrement,
        "finc": options.timeStepIncrement,
        "fdec": options.timeStepDecrement,
    }


def main() -> None:
    args = parse_args()
    mols = load_dataset(args.data_dir, args.num_mols)
    print(f"Loaded {len(mols)} molecules; total systems {sum(m.GetNumConformers() for m in mols)}.")

    options = make_fire_options()

    # ---------- nvMolKit batched run ----------
    nvm_mols_initial = deep_copy(mols)
    nvm_initial_energies = MMFFOptimizeMoleculesConfsFire(
        nvm_mols_initial,
        maxIters=0,
        fireOptions=options,
    )

    nvm_mols_step = deep_copy(mols)
    nvm_step_energies = MMFFOptimizeMoleculesConfsFire(
        nvm_mols_step,
        maxIters=args.steps,
        fireOptions=options,
    )

    # ---------- build payloads for ASE workers ----------
    fire_kwargs = _fire_kwargs_from_options(options)
    payloads: list[dict] = []
    for mol_idx, (mol, nvm_init_per, nvm_step_per, nvm_step_mol) in enumerate(
        zip(mols, nvm_initial_energies, nvm_step_energies, nvm_mols_step)
    ):
        for conf_id, (nvm_init_e, nvm_step_e) in enumerate(zip(nvm_init_per, nvm_step_per)):
            molblock = Chem.MolToMolBlock(mol, confId=conf_id, kekulize=False)
            payloads.append(
                {
                    "mol_idx": mol_idx,
                    "conf_id": conf_id,
                    "n_atoms": mol.GetNumAtoms(),
                    "molblock": molblock,
                    "fire_kwargs": fire_kwargs,
                    "grad_tol": options.gradTol,
                    "steps": args.steps,
                    "nvm_init_e": float(nvm_init_e),
                    "nvm_step_e": float(nvm_step_e),
                    "nvm_step_pos": positions_of(nvm_step_mol, conf_id).tolist(),
                }
            )

    per_system = process_map(
        _ase_worker,
        payloads,
        max_workers=args.workers,
        chunksize=8,
        desc="ase",
    )

    init_energy_diffs = [r["init_de"] for r in per_system]
    step_energy_diffs = [r["step_de"] for r in per_system]
    step_pos_max_drifts = [r["step_pos_max_drift"] for r in per_system]

    def stats(arr: list[float]) -> dict[str, float]:
        a = np.asarray(arr)
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
        }

    print("\n--- Initial energy diff (|nvm - rdkit|, kcal/mol) ---")
    print(json.dumps(stats(init_energy_diffs), indent=2))
    print(f"\n--- After {args.steps} step(s): energy diff (|nvm - ase|, kcal/mol) ---")
    print(json.dumps(stats(step_energy_diffs), indent=2))
    print(f"\n--- After {args.steps} step(s): max per-atom position drift (Å) ---")
    print(json.dumps(stats(step_pos_max_drifts), indent=2))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"systems": per_system}))
        print(f"\nWrote per-system data to {args.out}")


if __name__ == "__main__":
    main()
