# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy benchmark for the nvMolKit ETKDG integration with FIRE / BFGS minimizers.

For a sample of Enamine REAL molecules, generate ``confsPerMolecule`` conformers via
three pipelines under identical ETKDG parameters (``maxAttempts=10``):

    1. RDKit ``EmbedMultipleConfs`` (reference, RDKit's own BFGS-flavored minimizer).
    2. nvMolKit ``EmbedMolecules`` with ``MinimizerKind.BFGS``.
    3. nvMolKit ``EmbedMolecules`` with ``MinimizerKind.FIRE``.

Each conformer is single-point scored with RDKit MMFF94 (no further minimization, so
the comparison reflects the geometry produced by each embed+minimize pipeline). The
script reports:

    * per-conformer MMFF energies, saved to a ``.npz``,
    * failure tallies (SMILES parse, embedding produced zero confs, partial conformer
      generation, MMFF parameterization failure, NaN energy),
    * ΔE distributions: ``E_min(nvm) - E_min(rdkit)`` per molecule, and intra-molecule
      ``E - E_min`` to summarize ensemble spread per pipeline,
    * histogram / box plots written next to the energy file.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import ETKDGv3
from tqdm import tqdm

from nvmolkit.embedMolecules import EmbedMolecules, MinimizerKind
from nvmolkit.types import HardwareOptions

PIPELINES = ("rdkit", "nvmolkit_bfgs", "nvmolkit_fire")


@dataclass
class PipelineFailures:
    """Per-pipeline failure tallies. ``smiles_parse`` is shared but kept here for symmetry."""

    no_conformers: int = 0
    partial_conformers: int = 0
    mmff_param_failure: int = 0
    nan_energies: int = 0


@dataclass
class BenchResult:
    """Collected per-molecule results for one pipeline."""

    energies: list[np.ndarray] = field(default_factory=list)
    failures: PipelineFailures = field(default_factory=PipelineFailures)
    stage_names: list[str] = field(default_factory=list)
    stage_failure_counts: list[int] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_smiles",
        type=Path,
        help="Enamine CXSMILES file (tab-separated; first column SMILES, header expected).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for energy arrays, failure JSON, and plots.",
    )
    parser.add_argument("--num-molecules", type=int, default=200, help="Molecules to sample (default: 200).")
    parser.add_argument("--num-conformers", type=int, default=10, help="Conformers per molecule (default: 10).")
    parser.add_argument("--max-attempts", type=int, default=10, help="ETKDG maxAttempts (default: 10).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for SMILES sampling and ETKDG.")
    parser.add_argument(
        "--rdkit-threads",
        type=int,
        default=10,
        help="Threads for RDKit EmbedMultipleConfs (default: 10).",
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        default=True,
        help="Input file has a header row (default: True for Enamine CXSMILES).",
    )
    parser.add_argument(
        "--no-header",
        dest="has_header",
        action="store_false",
        help="Disable header skipping if the input is a plain SMILES list.",
    )
    return parser.parse_args()


def load_smiles(path: Path, num_molecules: int, seed: int, has_header: bool) -> list[str]:
    """Reservoir-sample ``num_molecules`` SMILES from ``path`` in a single pass.

    Avoids materializing the full 10M-line file in memory.
    """
    rng = random.Random(seed)
    reservoir: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        if has_header:
            handle.readline()
        seen = 0
        for line in handle:
            line = line.strip()
            if not line:
                continue
            smiles = line.split()[0]
            if seen < num_molecules:
                reservoir.append(smiles)
            else:
                slot = rng.randint(0, seen)
                if slot < num_molecules:
                    reservoir[slot] = smiles
            seen += 1
    if not reservoir:
        raise ValueError(f"No SMILES entries found in {path}.")
    rng.shuffle(reservoir)
    return reservoir


def make_etkdg_params(seed: int, max_attempts: int) -> ETKDGv3:
    params = ETKDGv3()
    params.randomSeed = seed
    params.maxAttempts = max_attempts
    params.useRandomCoords = True
    params.pruneRmsThresh = -1.0
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.useBasicKnowledge = True
    params.enforceChirality = True
    return params


def prepare_mols(smiles_list: list[str]) -> tuple[list[Chem.Mol], int]:
    """Parse and protonate SMILES, returning (mols, smiles_parse_failures)."""
    mols: list[Chem.Mol] = []
    parse_failures = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            parse_failures += 1
            continue
        mol = Chem.AddHs(mol)
        mol.SetProp("OriginalSMILES", smi)
        mols.append(mol)
    return mols, parse_failures


def deep_copy(mols: list[Chem.Mol]) -> list[Chem.Mol]:
    return [Chem.Mol(mol) for mol in mols]


def score_mmff(mol: Chem.Mol, num_conformers: int, failures: PipelineFailures) -> np.ndarray:
    """Single-point MMFF94 score for every conformer; pad to ``num_conformers`` with NaN."""
    energies = np.full(num_conformers, np.nan, dtype=float)
    if mol.GetNumConformers() == 0:
        return energies
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    if props is None:
        failures.mmff_param_failure += 1
        return energies
    for slot, conf in enumerate(mol.GetConformers()):
        if slot >= num_conformers:
            break
        force_field = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf.GetId())
        if force_field is None:
            continue
        energy = force_field.CalcEnergy()
        if np.isfinite(energy):
            energies[slot] = energy
        else:
            failures.nan_energies += 1
    return energies


def run_rdkit(mols: list[Chem.Mol], params: ETKDGv3, num_conformers: int, threads: int) -> BenchResult:
    result = BenchResult()
    saved_threads = params.numThreads
    params.numThreads = threads
    for mol in tqdm(mols, desc="rdkit embed"):
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
        if not conf_ids:
            result.failures.no_conformers += 1
        elif len(conf_ids) < num_conformers:
            result.failures.partial_conformers += 1
        result.energies.append(score_mmff(mol, num_conformers, result.failures))
    params.numThreads = saved_threads
    return result


def run_nvmolkit(
    mols: list[Chem.Mol],
    params: ETKDGv3,
    num_conformers: int,
    minimizer_kind: MinimizerKind,
    label: str,
) -> BenchResult:
    result = BenchResult()
    failures_out: dict = {}
    EmbedMolecules(
        mols,
        params,
        confsPerMolecule=num_conformers,
        maxIterations=-1,
        hardwareOptions=HardwareOptions(),
        minimizerKind=minimizer_kind,
        failuresOut=failures_out,
    )
    stage_names = list(failures_out.get("stage_names", []))
    counts = failures_out.get("counts", [])
    stage_totals = [int(sum(stage)) for stage in counts]
    result.stage_names = stage_names
    result.stage_failure_counts = stage_totals
    for mol in tqdm(mols, desc=f"{label} score"):
        num_confs = mol.GetNumConformers()
        if num_confs == 0:
            result.failures.no_conformers += 1
        elif num_confs < num_conformers:
            result.failures.partial_conformers += 1
        result.energies.append(score_mmff(mol, num_conformers, result.failures))
    return result


def per_molecule_min(energies_per_mol: list[np.ndarray]) -> np.ndarray:
    mins = np.full(len(energies_per_mol), np.nan, dtype=float)
    for idx, arr in enumerate(energies_per_mol):
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            mins[idx] = float(finite.min())
    return mins


def intra_mol_delta(energies_per_mol: list[np.ndarray]) -> np.ndarray:
    deltas: list[float] = []
    for arr in energies_per_mol:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        deltas.extend((finite - finite.min()).tolist())
    return np.asarray(deltas, dtype=float)


def plot_delta_histogram(
    deltas: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in deltas.items():
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        ax.hist(finite, bins=60, alpha=0.5, label=f"{label} (n={finite.size})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_stage_failures(
    stage_failures: dict[str, dict[str, int]],
    output_path: Path,
    num_molecules: int,
    confs_per_mol: int,
) -> None:
    """Grouped bar chart of per-stage failure tallies for each pipeline.

    Tallies are summed across all conformers of all molecules; the total possible
    failure count per pipeline is at most ``num_molecules * confs_per_mol`` per stage.
    """
    pipelines = list(stage_failures.keys())
    stage_order: list[str] = []
    seen: set[str] = set()
    for failures in stage_failures.values():
        for stage_name in failures:
            if stage_name not in seen:
                seen.add(stage_name)
                stage_order.append(stage_name)

    indices = np.arange(len(stage_order))
    width = 0.8 / max(len(pipelines), 1)
    max_attempts = num_molecules * confs_per_mol

    fig, ax = plt.subplots(figsize=(max(10, len(stage_order) * 1.2), 6))
    for slot, pipeline in enumerate(pipelines):
        counts = [stage_failures[pipeline].get(stage, 0) for stage in stage_order]
        offset = (slot - (len(pipelines) - 1) / 2) * width
        bars = ax.bar(indices + offset, counts, width=width, label=pipeline)
        for rect, count in zip(bars, counts):
            if count > 0:
                ax.annotate(
                    str(count),
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_xticks(indices)
    ax.set_xticklabels(stage_order, rotation=30, ha="right")
    ax.set_ylabel(f"Failure count (max possible {max_attempts})")
    ax.set_title("ETKDG per-stage failure tallies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_delta_box(deltas: dict[str, np.ndarray], output_path: Path, title: str, ylabel: str) -> None:
    labels = list(deltas.keys())
    data = [deltas[label][np.isfinite(deltas[label])] for label in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def summarize(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    smiles_list = load_smiles(args.input_smiles, args.num_molecules, args.seed, args.has_header)
    print(f"Loaded {len(smiles_list)} SMILES from {args.input_smiles}.")

    base_mols, smiles_parse_failures = prepare_mols(smiles_list)
    print(f"Prepared {len(base_mols)} molecules ({smiles_parse_failures} SMILES parse failures).")

    params = make_etkdg_params(args.seed, args.max_attempts)

    pipeline_mols = {name: deep_copy(base_mols) for name in PIPELINES}

    results: dict[str, BenchResult] = {}
    results["rdkit"] = run_rdkit(pipeline_mols["rdkit"], params, args.num_conformers, args.rdkit_threads)
    results["nvmolkit_bfgs"] = run_nvmolkit(
        pipeline_mols["nvmolkit_bfgs"], params, args.num_conformers, MinimizerKind.BFGS, "nvm_bfgs"
    )
    results["nvmolkit_fire"] = run_nvmolkit(
        pipeline_mols["nvmolkit_fire"], params, args.num_conformers, MinimizerKind.FIRE, "nvm_fire"
    )

    energy_arrays = {
        name: np.stack(result.energies, axis=0) if result.energies else np.empty((0, args.num_conformers))
        for name, result in results.items()
    }
    energies_path = args.output_dir / "energies.npz"
    np.savez(energies_path, **energy_arrays)
    print(f"Saved energies to {energies_path}.")

    rdkit_min = per_molecule_min(results["rdkit"].energies)
    bfgs_min = per_molecule_min(results["nvmolkit_bfgs"].energies)
    fire_min = per_molecule_min(results["nvmolkit_fire"].energies)

    inter_deltas = {
        "nvm_bfgs - rdkit": bfgs_min - rdkit_min,
        "nvm_fire - rdkit": fire_min - rdkit_min,
    }
    intra_deltas = {
        "rdkit": intra_mol_delta(results["rdkit"].energies),
        "nvm_bfgs": intra_mol_delta(results["nvmolkit_bfgs"].energies),
        "nvm_fire": intra_mol_delta(results["nvmolkit_fire"].energies),
    }

    plot_delta_histogram(
        inter_deltas,
        args.output_dir / "delta_min_energy_hist.png",
        title="Per-molecule min(E_nvm) - min(E_rdkit) (MMFF94, kcal/mol)",
        xlabel="ΔE_min (kcal/mol)",
    )
    plot_delta_box(
        inter_deltas,
        args.output_dir / "delta_min_energy_box.png",
        title="Per-molecule min(E_nvm) - min(E_rdkit)",
        ylabel="ΔE_min (kcal/mol)",
    )
    plot_delta_histogram(
        intra_deltas,
        args.output_dir / "intra_mol_delta_hist.png",
        title="Per-conformer E - min_E within molecule (MMFF94, kcal/mol)",
        xlabel="E - E_min (kcal/mol)",
    )
    plot_delta_box(
        intra_deltas,
        args.output_dir / "intra_mol_delta_box.png",
        title="Per-conformer E - min_E within molecule",
        ylabel="E - E_min (kcal/mol)",
    )

    stage_failures = {
        name: dict(zip(result.stage_names, result.stage_failure_counts))
        for name, result in results.items()
        if result.stage_names
    }
    if stage_failures:
        plot_stage_failures(
            stage_failures, args.output_dir / "stage_failures.png", len(base_mols), args.num_conformers
        )

    summary = {
        "config": {
            "input_smiles": str(args.input_smiles),
            "num_molecules_requested": args.num_molecules,
            "num_molecules_used": len(base_mols),
            "num_conformers": args.num_conformers,
            "max_attempts": args.max_attempts,
            "seed": args.seed,
        },
        "smiles_parse_failures": smiles_parse_failures,
        "failures": {
            name: {
                "no_conformers": result.failures.no_conformers,
                "partial_conformers": result.failures.partial_conformers,
                "mmff_param_failure": result.failures.mmff_param_failure,
                "nan_energies": result.failures.nan_energies,
            }
            for name, result in results.items()
        },
        "delta_min_energy_summary": {label: summarize(values) for label, values in inter_deltas.items()},
        "intra_mol_delta_summary": {label: summarize(values) for label, values in intra_deltas.items()},
        "stage_failures": stage_failures,
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary to {summary_path}.")
    print(json.dumps(summary["failures"], indent=2))
    print(json.dumps(summary["delta_min_energy_summary"], indent=2))


if __name__ == "__main__":
    main()
