# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optuna search for FIRE MMFF-minimization parameters on perturbed conformers.

Two studies, one per backend, both at fixed ``maxIters=200``:

    gpu  -- nvMolKit FIRE MMFF
    ase  -- ASE FIRE2 (CPU serial), 50-mol subset

In both cases the Optuna objective is minimized: ``mean((E_final - E_ref) / n_atoms)``
over all systems with finite final energies, where ``E_ref`` is the RDKit-MMFF
reference energy from the prep stage. Systems with non-finite final energies are
dropped from the mean.

Studies are stored in an SQLite RDB (``--storage``) so SIGINT or process restart
preserves intermediate state. Re-running the script appends additional trials up
to ``--n-trials``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import optuna
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE2 as ASEFire2
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.contrib.concurrent import process_map

from nvmolkit.mmffOptimization import FireOptions, MMFFOptimizeMoleculesConfsFire
from nvmolkit.types import HardwareOptions

MAX_ITERS = 200
INVALID_TRIAL_SCORE = 1e6  # used when no system produced a finite energy
OBJECTIVE_CHOICES = ("gap", "abs_gap", "grad_norm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Output of fire_optuna_prep.py")
    parser.add_argument(
        "--storage",
        type=Path,
        default=Path("benchmarks/fire_optuna.db"),
        help="SQLite RDB path for Optuna storage.",
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--ase-subset", type=int, default=50, help="Mol count for ASE arm.")
    parser.add_argument(
        "--ase-workers",
        type=int,
        default=None,
        help="Worker process count for the ASE arm (default: cpu_count()).",
    )
    parser.add_argument(
        "--objective",
        choices=OBJECTIVE_CHOICES,
        default="gap",
        help=(
            "What to minimize. 'gap': mean signed (E_final - E_ref) / n_atoms. "
            "'abs_gap': same but absolute value (penalizes both above and below the reference). "
            "'grad_norm': mean per-atom MMFF gradient norm at the final positions, computed via RDKit "
            "(reference-independent and uses the same MMFF implementation for both backends)."
        ),
    )
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=["gpu", "ase"],
        default=["gpu", "ase"],
    )
    parser.add_argument(
        "--no-ase",
        action="store_true",
        help="Skip the ASE study (overrides --studies).",
    )
    return parser.parse_args()


def load_dataset(data_dir: Path) -> tuple[list[Chem.Mol], list[list[float]]]:
    sdf_path = data_dir / "perturbed.sdf"
    metadata = json.loads((data_dir / "metadata.json").read_text())
    ref_energies: list[list[float]] = metadata["ref_energies"]

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols_by_index: dict[int, Chem.Mol] = {}
    for mol in supplier:
        if mol is None:
            continue
        mol_idx = int(mol.GetProp("_MolIndex"))
        if mol_idx in mols_by_index:
            existing = mols_by_index[mol_idx]
            new_conf_id = existing.AddConformer(mol.GetConformer(), assignId=True)
            del new_conf_id
        else:
            mols_by_index[mol_idx] = mol
    mols = [mols_by_index[i] for i in sorted(mols_by_index)]
    return mols, ref_energies


def deep_copy(mols: list[Chem.Mol]) -> list[Chem.Mol]:
    return [Chem.Mol(mol) for mol in mols]


def suggest_fire_options(trial: optuna.Trial) -> FireOptions:
    options = FireOptions()
    options.useMass = False
    options.abcCorrection = False
    options.takeHalfStepBack = True
    options.stuckDetectionEnabled = False

    options.dtInit = trial.suggest_float("dtInit", 1e-4, 1e-2, log=True)
    options.dtMaxFactor = trial.suggest_float("dtMaxFactor", 2.0, 30.0, log=True)
    options.dtMinFactor = trial.suggest_float("dtMinFactor", 1e-4, 1e-1, log=True)
    options.dMax = trial.suggest_float("dMax", 0.05, 1.0, log=True)
    options.nMinForIncrease = trial.suggest_int("nMinForIncrease", 3, 30)
    options.alphaInit = trial.suggest_float("alphaInit", 0.05, 0.5)
    # Sample (1 - alphaDecrement) on a log scale: linear sampling on [0.9, 0.999] biases
    # heavily toward the "barely decays" tail because the meaningful quantity is the
    # per-step shrinkage 1 - fa.
    one_minus_alpha_decrement = trial.suggest_float("one_minus_alphaDecrement", 1e-3, 1e-1, log=True)
    options.alphaDecrement = 1.0 - one_minus_alpha_decrement
    # Same reasoning: the meaningful quantity is the relative dt growth per step (finc - 1).
    inc_minus_one = trial.suggest_float("timeStepIncrement_minus_one", 0.05, 0.5, log=True)
    options.timeStepIncrement = 1.0 + inc_minus_one
    options.timeStepDecrement = trial.suggest_float("timeStepDecrement", 0.2, 0.7)
    return options


def total_atoms_and_systems(mols: list[Chem.Mol]) -> tuple[int, int]:
    total_atoms = 0
    total_systems = 0
    for mol in mols:
        total_atoms += mol.GetNumAtoms() * mol.GetNumConformers()
        total_systems += mol.GetNumConformers()
    return total_atoms, total_systems


def rdkit_per_system_metrics(mols: list[Chem.Mol]) -> tuple[list[list[float]], list[list[float]]]:
    """For each (mol, conformer) compute RDKit MMFF94 energy and gradient L2 norm.

    Returned shapes mirror the input layout. NaN for systems where the FF fails to build.
    """
    energies: list[list[float]] = []
    grad_norms: list[list[float]] = []
    for mol in mols:
        per_mol_e: list[float] = []
        per_mol_g: list[float] = []
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        for conf in mol.GetConformers():
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf.GetId()) if props is not None else None
            if ff is None:
                per_mol_e.append(float("nan"))
                per_mol_g.append(float("nan"))
                continue
            energy = ff.CalcEnergy()
            grad = np.asarray(ff.CalcGrad(), dtype=float)
            per_mol_e.append(float(energy))
            per_mol_g.append(float(np.linalg.norm(grad)))
        energies.append(per_mol_e)
        grad_norms.append(per_mol_g)
    return energies, grad_norms


def aggregate_objective(
    objective: str,
    rdkit_final_energies: list[list[float]],
    rdkit_final_grad_norms: list[list[float]],
    ref_energies: list[list[float]],
    mols: list[Chem.Mol],
) -> tuple[float, int, int]:
    """Returns (objective_value, valid_systems, total_systems).

    Per-atom mean over systems with finite metrics. Systems with non-finite values
    are silently dropped.
    """
    samples: list[float] = []
    total = 0
    for mol, ref_per, e_per, g_per in zip(mols, ref_energies, rdkit_final_energies, rdkit_final_grad_norms):
        n_atoms = mol.GetNumAtoms()
        for ref, energy, grad_norm in zip(ref_per, e_per, g_per):
            total += 1
            if objective == "grad_norm":
                if not math.isfinite(grad_norm):
                    continue
                samples.append(grad_norm / n_atoms)
            else:
                if not math.isfinite(energy) or not math.isfinite(ref):
                    continue
                delta = (energy - ref) / n_atoms
                if objective == "abs_gap":
                    samples.append(abs(delta))
                else:  # "gap"
                    samples.append(delta)
    if not samples:
        return float("inf"), 0, total
    return float(np.asarray(samples).mean()), len(samples), total


class _NvmMmffCalculator(Calculator):
    """ASE Calculator wrapping RDKit's MMFF94 force field for a single conformer."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, mol: Chem.Mol, conf_id: int) -> None:
        super().__init__()
        self._mol = mol
        self._conf_id = conf_id
        self._props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        self._ff = AllChem.MMFFGetMoleculeForceField(mol, self._props, confId=conf_id)
        if self._ff is None:
            raise RuntimeError("Unable to construct MMFF force field for ASE.")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes) -> None:  # noqa: D401
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        conf = self._mol.GetConformer(self._conf_id)
        for idx in range(self._mol.GetNumAtoms()):
            conf.SetAtomPosition(idx, positions[idx].tolist())
        self._ff = AllChem.MMFFGetMoleculeForceField(self._mol, self._props, confId=self._conf_id)
        energy = self._ff.CalcEnergy()
        grad = np.asarray(self._ff.CalcGrad()).reshape(-1, 3)
        self.results = {"energy": float(energy), "forces": -grad}


def _ase_atoms_for_conformer(mol: Chem.Mol, conf_id: int) -> Atoms:
    conf = mol.GetConformer(conf_id)
    positions = np.asarray(conf.GetPositions(), dtype=float)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=positions)


def _ase_fire_kwargs(options: FireOptions) -> dict:
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


def _ase_trial_worker(payload: dict) -> tuple[int, int, float, list[list[float]] | None]:
    """Worker for tqdm.process_map.

    Returns (mol_idx, conf_id, final_energy_or_nan, final_positions_or_none). Positions
    are returned as a nested Python list so the main process can write them back into
    the mol-with-conformers it owns; subsequent ``rdkit_per_system_metrics`` then sees
    the optimized geometry rather than the original perturbed geometry.
    """
    mol_idx = payload["mol_idx"]
    conf_id = payload["conf_id"]
    mol = Chem.MolFromMolBlock(payload["molblock"], removeHs=False)
    if mol is None:
        return mol_idx, conf_id, float("nan"), None
    fire_kwargs = payload["fire_kwargs"]
    grad_tol = payload["grad_tol"]
    max_iters = payload["max_iters"]

    calc = _NvmMmffCalculator(mol, conf_id=0)
    atoms = _ase_atoms_for_conformer(mol, conf_id=0)
    atoms.calc = calc
    optimizer = ASEFire2(atoms, logfile=None, **fire_kwargs)
    try:
        optimizer.run(fmax=grad_tol, steps=max_iters)
    except Exception:
        return mol_idx, conf_id, float("nan"), None

    final_positions = atoms.get_positions().tolist()
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=0)
    if ff is None:
        return mol_idx, conf_id, float("nan"), final_positions
    return mol_idx, conf_id, float(ff.CalcEnergy()), final_positions


def run_ase_trial(
    mols: list[Chem.Mol],
    ref_energies: list[list[float]],  # noqa: ARG001 -- kept to match GPU signature
    options: FireOptions,
    max_iters: int,
    workers: int | None,
) -> tuple[list[list[float]], float]:
    """Parallel ASE-FIRE2 trial via tqdm.process_map. Returns per-mol per-conf energies."""
    start = time.perf_counter()

    fire_kwargs = _ase_fire_kwargs(options)
    payloads: list[dict] = []
    for mol_idx, mol in enumerate(mols):
        for conf in mol.GetConformers():
            payloads.append(
                {
                    "mol_idx": mol_idx,
                    "conf_id": conf.GetId(),
                    "molblock": Chem.MolToMolBlock(mol, confId=conf.GetId(), kekulize=False),
                    "fire_kwargs": fire_kwargs,
                    "grad_tol": options.gradTol,
                    "max_iters": max_iters,
                }
            )

    results = process_map(
        _ase_trial_worker,
        payloads,
        max_workers=workers,
        chunksize=8,
        desc="ase trial",
        leave=False,
    )

    final_energies: list[list[float]] = [[float("nan")] * mol.GetNumConformers() for mol in mols]
    for mol_idx, conf_id, energy, positions in results:
        final_energies[mol_idx][conf_id] = energy
        if positions is not None:
            conf = mols[mol_idx].GetConformer(conf_id)
            for atom_idx, xyz in enumerate(positions):
                conf.SetAtomPosition(atom_idx, [float(c) for c in xyz])

    elapsed = time.perf_counter() - start
    return final_energies, elapsed


def run_gpu_trial(
    mols: list[Chem.Mol],
    options: FireOptions,
    max_iters: int,
) -> tuple[list[list[float]], float]:
    """Returns (final_energies_per_mol, elapsed_seconds)."""
    start = time.perf_counter()
    energies = MMFFOptimizeMoleculesConfsFire(
        mols,
        maxIters=max_iters,
        fireOptions=options,
        hardwareOptions=HardwareOptions(),
    )
    elapsed = time.perf_counter() - start
    return energies, elapsed


def make_objective(
    backend: str,
    base_mols: list[Chem.Mol],
    ref_energies: list[list[float]],
    ase_subset: int,
    ase_workers: int | None,
    objective_kind: str,
):
    def objective(trial: optuna.Trial) -> float:
        options = suggest_fire_options(trial)

        if backend == "gpu":
            mols = deep_copy(base_mols)
            refs = ref_energies
            _native_energies, elapsed = run_gpu_trial(mols, options, MAX_ITERS)
        else:
            mols = deep_copy(base_mols[:ase_subset])
            refs = ref_energies[:ase_subset]
            _native_energies, elapsed = run_ase_trial(mols, refs, options, MAX_ITERS, ase_workers)

        rdkit_energies, rdkit_grad_norms = rdkit_per_system_metrics(mols)
        score, valid, total = aggregate_objective(
            objective_kind,
            rdkit_energies,
            rdkit_grad_norms,
            refs,
            mols,
        )
        trial.set_user_attr("elapsed", elapsed)
        trial.set_user_attr("valid_systems", valid)
        trial.set_user_attr("total_systems", total)
        trial.set_user_attr("objective_kind", objective_kind)
        trial.set_user_attr(f"mean_{objective_kind}_per_atom", score)

        if not math.isfinite(score):
            return INVALID_TRIAL_SCORE
        return score

    return objective


STUDY_CONFIG: dict[str, str] = {
    "gpu": "gpu",
    "ase": "ase",
}


def report_best(study: optuna.Study, name: str) -> None:
    print(f"\n=== Study: {name} ===")
    if not study.trials:
        print("  no trials run.")
        return
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"  trials run: {len(study.trials)}, completed: {len(completed)}")
    if not completed:
        return
    try:
        best = study.best_trial
    except ValueError:
        return
    print(f"  best score: {best.value:.4f}")
    print(f"  best params: {json.dumps(best.params, indent=2)}")
    print(f"  best user_attrs: {json.dumps(best.user_attrs, indent=2)}")


def main() -> None:
    args = parse_args()
    args.storage.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{args.storage.resolve()}"

    base_mols, ref_energies = load_dataset(args.data_dir)
    print(f"Loaded {len(base_mols)} molecules from {args.data_dir}.")

    studies: dict[str, optuna.Study] = {}

    def shutdown_handler(signum, frame):  # noqa: ARG001
        print("\nReceived signal; printing partial results.")
        for name, study in studies.items():
            report_best(study, name)
        os._exit(130)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    studies_to_run = [name for name in args.studies if not (args.no_ase and name == "ase")]
    for name in studies_to_run:
        backend = STUDY_CONFIG[name]
        study = optuna.create_study(
            study_name=name,
            storage=storage_url,
            direction="minimize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        studies[name] = study
        objective = make_objective(backend, base_mols, ref_energies, args.ase_subset, args.ase_workers, args.objective)
        already = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining = max(0, args.n_trials - already)
        print(
            f"\n>>> Running study {name} (maxIters={MAX_ITERS}); "
            f"already complete={already}, will run {remaining} more."
        )
        if remaining > 0:
            study.optimize(objective, n_trials=remaining, gc_after_trial=True, show_progress_bar=True)
        report_best(study, name)

    print("\nAll studies complete.")


if __name__ == "__main__":
    main()
