# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare three minimizers on the perturbed Enamine conformer dataset.

For each conformer (perturbed geometry from ``fire_optuna_prep.py`` output), run:

    1. RDKit MMFF94 (serial, ``MMFFOptimizeMoleculeConfs``).
    2. nvMolKit MMFF BFGS (``MMFFOptimizeMoleculesConfs``).
    3. nvMolKit MMFF FIRE (``MMFFOptimizeMoleculesConfsFire``) using the best
       parameters from a previously-run Optuna study.

After minimization every conformer is scored with **RDKit MMFF94** — same scoring
function across arms — to produce a fair comparison of (a) final per-atom MMFF
energy and (b) per-conformer max per-atom force norm at the final position.

Outputs histograms and per-arm summary stats.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.contrib.concurrent import process_map

from nvmolkit.mmffOptimization import (
    FireOptions,
    MMFFOptimizeMoleculesConfs,
    MMFFOptimizeMoleculesConfsFire,
)
from nvmolkit.types import HardwareOptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Output of fire_optuna_prep.py")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--optuna-storage",
        type=Path,
        default=Path("benchmarks/fire_optuna_gradnorm.db"),
        help="Sqlite path for the optuna run whose 'gpu' study supplies FIRE params.",
    )
    parser.add_argument("--optuna-study", default="gpu")
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument(
        "--fire-extended-iters",
        type=int,
        default=400,
        help="Iteration count for the additional 'nvmolkit_fire_long' arm. Set to 0 to disable.",
    )
    parser.add_argument("--rdkit-workers", type=int, default=None, help="Default cpu_count().")
    parser.add_argument("--num-mols", type=int, default=None)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip all minimization; reload arrays.npz from output_dir and re-render plots.",
    )
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


def fire_options_from_optuna(storage: Path, study_name: str) -> FireOptions:
    storage_url = f"sqlite:///{storage.resolve()}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    params = study.best_trial.params
    print(
        f"FIRE params from study {study_name!r} best trial #{study.best_trial.number} "
        f"(value={study.best_trial.value:.6g}):"
    )
    for key, value in params.items():
        print(f"  {key}={value}")

    options = FireOptions()
    options.useMass = False
    options.abcCorrection = False
    options.takeHalfStepBack = True
    options.stuckDetectionEnabled = False
    options.dtInit = params["dtInit"]
    options.dtMaxFactor = params["dtMaxFactor"]
    options.dtMinFactor = params["dtMinFactor"]
    options.dMax = params["dMax"]
    options.nMinForIncrease = params["nMinForIncrease"]
    options.alphaInit = params["alphaInit"]
    if "one_minus_alphaDecrement" in params:
        options.alphaDecrement = 1.0 - params["one_minus_alphaDecrement"]
    else:
        options.alphaDecrement = params["alphaDecrement"]
    if "timeStepIncrement_minus_one" in params:
        options.timeStepIncrement = 1.0 + params["timeStepIncrement_minus_one"]
    else:
        options.timeStepIncrement = params["timeStepIncrement"]
    options.timeStepDecrement = params["timeStepDecrement"]
    return options


def _rdkit_minimize_worker(payload: dict) -> tuple[int, int, list[list[float]] | None]:
    mol_idx = payload["mol_idx"]
    conf_id = payload["conf_id"]
    mol = Chem.MolFromMolBlock(payload["molblock"], removeHs=False)
    if mol is None:
        return mol_idx, conf_id, None
    results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=payload["max_iters"], mmffVariant="MMFF94")
    if not results:
        return mol_idx, conf_id, None
    final_positions = mol.GetConformer(0).GetPositions().tolist()
    return mol_idx, conf_id, final_positions


def rdkit_minimize_in_place(mols: list[Chem.Mol], max_iters: int, workers: int | None) -> None:
    payloads: list[dict] = []
    for mol_idx, mol in enumerate(mols):
        for conf in mol.GetConformers():
            payloads.append(
                {
                    "mol_idx": mol_idx,
                    "conf_id": conf.GetId(),
                    "molblock": Chem.MolToMolBlock(mol, confId=conf.GetId(), kekulize=False),
                    "max_iters": max_iters,
                }
            )
    results = process_map(
        _rdkit_minimize_worker,
        payloads,
        max_workers=workers,
        chunksize=8,
        desc="rdkit",
    )
    for mol_idx, conf_id, positions in results:
        if positions is None:
            continue
        conf = mols[mol_idx].GetConformer(conf_id)
        for atom_idx, xyz in enumerate(positions):
            conf.SetAtomPosition(atom_idx, [float(c) for c in xyz])


def score_with_rdkit(mol: Chem.Mol) -> tuple[list[float], list[float]]:
    """For each conformer of ``mol`` return (final_energy, rdkit_grad_test).

    ``rdkit_grad_test`` is the scaled max-component gradient metric RDKit's BFGS
    optimizer uses to decide convergence: ``max_i(|g_i| * max(|x_i|, 1)) / max(|E|, 1)``.
    Mirrors ``BFGSOpt::minimize`` lines 380-390. The internal ``gradScale`` factor
    (returned by MMFF's gradient functor in C++) is not exposed in Python, so this
    omits it; for comparing arms on the same molecule with the same final energy that
    omission is irrelevant.
    """
    energies: list[float] = []
    grad_tests: list[float] = []
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    for conf in mol.GetConformers():
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf.GetId()) if props is not None else None
        if ff is None:
            energies.append(float("nan"))
            grad_tests.append(float("nan"))
            continue
        energy = float(ff.CalcEnergy())
        positions = np.asarray(conf.GetPositions(), dtype=float).reshape(-1)
        grad = np.asarray(ff.CalcGrad(), dtype=float)
        if positions.size != grad.size:
            energies.append(energy)
            grad_tests.append(float("nan"))
            continue
        scale = np.maximum(np.abs(positions), 1.0)
        per_coord = np.abs(grad) * scale
        denom = max(abs(energy), 1.0)
        grad_tests.append(float(per_coord.max()) / denom)
        energies.append(energy)
    return energies, grad_tests


def collect_arm_metrics(mols: list[Chem.Mol]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (energies, rdkit_grad_test_per_conformer, n_atoms_per_conformer)."""
    energies: list[float] = []
    grad_tests: list[float] = []
    n_atoms: list[int] = []
    for mol in mols:
        e_per, gt_per = score_with_rdkit(mol)
        energies.extend(e_per)
        grad_tests.extend(gt_per)
        n_atoms.extend([mol.GetNumAtoms()] * mol.GetNumConformers())
    return np.asarray(energies), np.asarray(grad_tests), np.asarray(n_atoms)


def summarize(name: str, label: str, values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {}
    summary = {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite.max()),
        "min": float(finite.min()),
    }
    print(f"[{name}] {label}: " + ", ".join(f"{k}={v:.4g}" for k, v in summary.items()))
    return summary


def plot_step_overlay(
    arrays: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    xlabel: str,
    bins: int = 40,
    use_log_x: bool = False,
    x_range: tuple[float, float] | None = None,
    clip_quantile: float | None = 0.99,
) -> None:
    """Histogram-counts as step-curves, one per array, overlaid.

    Counts are computed with shared bin edges so the curves are directly comparable.
    Use ``x_range`` for a hard range; otherwise auto-clip to the joint
    ``clip_quantile`` of the finite values across all arrays. Step-curves render the
    distribution shape without the visual noise of overlapping bar plots.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    finite_arrays = {label: arr[np.isfinite(arr)] for label, arr in arrays.items()}
    finite_arrays = {label: arr for label, arr in finite_arrays.items() if arr.size > 0}
    if not finite_arrays:
        return

    if x_range is not None:
        lower, upper = x_range
    else:
        joint = np.concatenate(list(finite_arrays.values()))
        if clip_quantile is not None:
            upper = float(np.quantile(joint, clip_quantile))
            lower = float(np.quantile(joint, 1.0 - clip_quantile))
        else:
            lower = float(joint.min())
            upper = float(joint.max())

    if use_log_x:
        lower_log = max(lower, 1e-8)
        edges = np.logspace(np.log10(lower_log), np.log10(upper), bins + 1)
    else:
        edges = np.linspace(lower, upper, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    for label, arr in finite_arrays.items():
        counts, _ = np.histogram(arr, bins=edges)
        ax.plot(centers, counts, drawstyle="steps-mid", label=f"{label} (n={arr.size})", linewidth=1.6)

    if use_log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xlim(lower, upper)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_delta_scatter_vs_rdkit(
    energies: dict[str, np.ndarray],
    n_atoms: np.ndarray,
    output_path: Path,
    title: str,
    y_range: tuple[float, float] | None = None,
) -> None:
    """One panel per non-RDKit arm; x = RDKit energy, y = (E_arm - E_rdkit) / n_atoms.

    A horizontal y=0 reference line marks parity with RDKit. If ``y_range`` is None the
    axis auto-scales to the joint 99th percentile of |delta|.
    """
    if "rdkit" not in energies:
        return
    other_arms = [name for name in energies if name != "rdkit"]
    if not other_arms:
        return

    rdkit_e = energies["rdkit"]
    n_arms = len(other_arms)
    fig, axes = plt.subplots(1, n_arms, figsize=(5.0 * n_arms, 5.0), squeeze=False)

    if y_range is None:
        joint_abs: list[float] = []
        for name in other_arms:
            arm_e = energies[name]
            mask = np.isfinite(rdkit_e) & np.isfinite(arm_e)
            if mask.any():
                deltas = (arm_e[mask] - rdkit_e[mask]) / n_atoms[mask]
                joint_abs.append(float(np.quantile(np.abs(deltas), 0.99)))
        bound = max(joint_abs) if joint_abs else 1.0
        y_range = (-bound, bound)

    for idx, name in enumerate(other_arms):
        ax = axes[0][idx]
        arm_e = energies[name]
        mask = np.isfinite(rdkit_e) & np.isfinite(arm_e)
        x = rdkit_e[mask]
        y = (arm_e[mask] - rdkit_e[mask]) / n_atoms[mask]
        if x.size == 0:
            ax.set_title(f"{name}: no data")
            continue
        ax.scatter(x, y, s=4, alpha=0.4, edgecolors="none")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="ΔE = 0")
        bias = float(np.mean(y))
        rmse = float(np.sqrt(np.mean(y**2)))
        ax.set_xlabel("rdkit energy (kcal/mol)")
        ax.set_ylabel(f"({name} - rdkit) / atom (kcal/mol)")
        ax.set_ylim(*y_range)
        ax.set_title(f"{name}\nbias={bias:+.4g}, rmse={rmse:.4g}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_energy_scatter_vs_rdkit(
    energies: dict[str, np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    """One panel per non-RDKit arm; x = RDKit energy, y = arm energy. Includes y=x line."""
    if "rdkit" not in energies:
        return
    other_arms = [name for name in energies if name != "rdkit"]
    if not other_arms:
        return

    n_arms = len(other_arms)
    fig, axes = plt.subplots(1, n_arms, figsize=(5.0 * n_arms, 5.0), squeeze=False)
    rdkit_e = energies["rdkit"]

    for idx, name in enumerate(other_arms):
        ax = axes[0][idx]
        arm_e = energies[name]
        mask = np.isfinite(rdkit_e) & np.isfinite(arm_e)
        x = rdkit_e[mask]
        y = arm_e[mask]
        if x.size == 0:
            ax.set_title(f"{name}: no data")
            continue
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.scatter(x, y, s=4, alpha=0.4, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0, label="y = x")
        rmse = float(np.sqrt(np.mean((y - x) ** 2)))
        bias = float(np.mean(y - x))
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("rdkit energy (kcal/mol)")
        ax.set_ylabel(f"{name} energy (kcal/mol)")
        ax.set_title(f"{name}\nbias={bias:+.4g}, rmse={rmse:.4g}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_minimization_arms(
    base_mols: list[Chem.Mol],
    fire_options: FireOptions,
    max_iters: int,
    fire_extended_iters: int,
    rdkit_workers: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, list[str]]:
    arm_mols: dict[str, list[Chem.Mol]] = {
        "rdkit": deep_copy(base_mols),
        "nvmolkit_bfgs": deep_copy(base_mols),
        "nvmolkit_fire": deep_copy(base_mols),
    }
    if fire_extended_iters > 0:
        arm_mols[f"nvmolkit_fire_{fire_extended_iters}"] = deep_copy(base_mols)

    print("\nRDKit MMFF (parallel)...")
    rdkit_minimize_in_place(arm_mols["rdkit"], max_iters, rdkit_workers)

    print("nvMolKit MMFF BFGS...")
    MMFFOptimizeMoleculesConfs(arm_mols["nvmolkit_bfgs"], maxIters=max_iters, hardwareOptions=HardwareOptions())

    print(f"nvMolKit MMFF FIRE ({max_iters} iters)...")
    MMFFOptimizeMoleculesConfsFire(
        arm_mols["nvmolkit_fire"],
        maxIters=max_iters,
        fireOptions=fire_options,
        hardwareOptions=HardwareOptions(),
    )

    if fire_extended_iters > 0:
        long_key = f"nvmolkit_fire_{fire_extended_iters}"
        print(f"nvMolKit MMFF FIRE ({fire_extended_iters} iters)...")
        MMFFOptimizeMoleculesConfsFire(
            arm_mols[long_key],
            maxIters=fire_extended_iters,
            fireOptions=fire_options,
            hardwareOptions=HardwareOptions(),
        )

    print("\nScoring with RDKit MMFF...")
    energies: dict[str, np.ndarray] = {}
    grad_tests: dict[str, np.ndarray] = {}
    n_atoms = np.empty(0, dtype=int)
    for name, mols in arm_mols.items():
        e_arr, gt_arr, n_atoms_arr = collect_arm_metrics(mols)
        energies[name] = e_arr
        grad_tests[name] = gt_arr
        if n_atoms.size == 0:
            n_atoms = n_atoms_arr
    return energies, grad_tests, n_atoms, list(arm_mols.keys())


def save_arrays(
    output_path: Path,
    energies: dict[str, np.ndarray],
    grad_tests: dict[str, np.ndarray],
    n_atoms: np.ndarray,
    arm_order: list[str],
) -> None:
    payload: dict[str, np.ndarray] = {"_n_atoms": n_atoms, "_arm_order": np.asarray(arm_order)}
    for name in arm_order:
        payload[f"energy::{name}"] = energies[name]
        payload[f"grad_test::{name}"] = grad_tests[name]
    np.savez(output_path, **payload)


def load_arrays(input_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, list[str]]:
    with np.load(input_path, allow_pickle=False) as data:
        n_atoms = data["_n_atoms"]
        arm_order = list(data["_arm_order"])
        energies = {name: data[f"energy::{name}"] for name in arm_order}
        grad_tests = {name: data[f"grad_test::{name}"] for name in arm_order}
    return energies, grad_tests, n_atoms, arm_order


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = args.output_dir / "arrays.npz"

    if args.plots_only:
        if not arrays_path.exists():
            raise SystemExit(f"--plots-only requested but {arrays_path} does not exist.")
        energies, grad_tests, n_atoms, _arm_order = load_arrays(arrays_path)
        print(f"Loaded cached arrays from {arrays_path} (no minimization).")
    else:
        base_mols = load_dataset(args.data_dir, args.num_mols)
        print(f"Loaded {len(base_mols)} mols / {sum(m.GetNumConformers() for m in base_mols)} conformers.")
        fire_options = fire_options_from_optuna(args.optuna_storage, args.optuna_study)
        energies, grad_tests, n_atoms, arm_order = run_minimization_arms(
            base_mols,
            fire_options,
            args.max_iters,
            args.fire_extended_iters,
            args.rdkit_workers,
        )
        save_arrays(arrays_path, energies, grad_tests, n_atoms, arm_order)
        print(f"Saved arrays to {arrays_path}")

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for name, energy_arr in energies.items():
        summary[name] = {
            "energy_kcal_per_mol": summarize(name, "E", energy_arr),
            "energy_per_atom": summarize(name, "E/atom", energy_arr / n_atoms),
            "rdkit_grad_test": summarize(name, "rdkit_grad_test", grad_tests[name]),
        }

    e_rdkit = energies["rdkit"]
    delta_vs_rdkit: dict[str, np.ndarray] = {}
    for name, e_arr in energies.items():
        if name == "rdkit":
            continue
        delta_vs_rdkit[f"{name} - rdkit"] = (e_arr - e_rdkit) / n_atoms
    for name, arr in delta_vs_rdkit.items():
        summarize(name, "ΔE/atom", arr)

    plot_step_overlay(
        {name: arr for name, arr in energies.items()},
        args.output_dir / "final_energies.png",
        title=f"Final RDKit-MMFF energy after {args.max_iters} iters",
        xlabel="Energy (kcal/mol)",
    )
    plot_step_overlay(
        {name: arr / n_atoms for name, arr in energies.items()},
        args.output_dir / "final_energies_per_atom.png",
        title=f"Final RDKit-MMFF energy per atom after {args.max_iters} iters",
        xlabel="Energy / atom (kcal/mol)",
    )
    plot_step_overlay(
        delta_vs_rdkit,
        args.output_dir / "delta_vs_rdkit_per_atom.png",
        title=f"(E_nvmolkit_arm - E_rdkit) / atom after {args.max_iters} iters",
        xlabel="ΔE/atom (kcal/mol)",
        x_range=(-0.0002, 0.0002),
    )
    plot_step_overlay(
        grad_tests,
        args.output_dir / "rdkit_grad_test.png",
        title=f"RDKit BFGS scaled max-grad test after {args.max_iters} iters",
        xlabel="max_i(|g_i| · max(|x_i|, 1)) / max(|E|, 1)",
        use_log_x=True,
    )

    plot_energy_scatter_vs_rdkit(
        energies,
        args.output_dir / "energy_scatter_vs_rdkit.png",
        title=f"Per-conformer final energy vs RDKit ({args.max_iters} iters)",
    )
    plot_delta_scatter_vs_rdkit(
        energies,
        n_atoms,
        args.output_dir / "delta_scatter_vs_rdkit.png",
        title=f"Per-conformer ΔE/atom vs RDKit ({args.max_iters} iters)",
        y_range=(-0.005, 0.005),
    )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary + plots to {args.output_dir}")


if __name__ == "__main__":
    main()
