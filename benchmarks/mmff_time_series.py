# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plot per-step FIRE 2.0 minimization trajectories for a single molecule.

Diagnostic tool for the nvmolkit FIRE 2.0 minimizer. Loads the first molecule
from an SDF, runs FIRE under several variant configurations (semi-implicit Euler
with/without half-step-back, with/without ABC, sweeps over a few key knobs),
and writes per-step alpha/dt/power/energy plots to disk.

Depends on `MMFFOptimizeMoleculesConfsFire(..., fireDebugOutput=...)` which is
slated for removal before PR. This script will need to be deleted or rewritten
to drive FIRE one-step-at-a-time (much slower) once that API is removed.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write figures to disk
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from rdkit import Chem

from nvmolkit.mmffOptimization import FireOptions, MMFFOptimizeMoleculesConfsFire


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_sdf", type=Path, help="Input SDF; the first molecule is used.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fire_time_series"),
        help="Directory for figures and CSV output (default: ./fire_time_series).",
    )
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum FIRE steps per run (default: 500).")
    parser.add_argument(
        "--rdkit-reference",
        type=float,
        default=None,
        help="Optional RDKit reference final energy (kcal/mol) plotted as a horizontal line.",
    )
    parser.add_argument(
        "--no-scan",
        action="store_true",
        help="Skip the parameter-sweep stage; produce only the four-variant plot.",
    )
    return parser.parse_args()


def load_first_mol(sdf_path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for mol in supplier:
        if mol is not None:
            return mol
    raise ValueError(f"No molecules in {sdf_path}.")


@dataclass
class FireRun:
    label: str
    energies: list[float]
    alphas: list[float]
    dts: list[float]
    powers: list[float]


def run_single_minimization(
    mol: Chem.Mol,
    *,
    steps: int,
    half_step: bool,
    use_abc: bool,
    use_masses: bool,
    n_min_for_increase: int = 5,
    dt_init: float = 0.001,
    time_step_increment: float = 1.1,
    alpha_init: float = 0.25,
    max_step: float = 0.0,
    dt_max_factor: float = 10.0,
    label: str = "",
) -> FireRun:
    opts = FireOptions()
    opts.dtInit = dt_init
    opts.dtMaxFactor = dt_max_factor
    opts.nMinForIncrease = n_min_for_increase
    opts.alphaInit = alpha_init
    opts.timeStepIncrement = time_step_increment
    opts.dMax = max_step
    opts.takeHalfStepBack = half_step
    opts.abcCorrection = use_abc
    opts.useMass = use_masses

    fire_debug: list = []
    MMFFOptimizeMoleculesConfsFire(
        [Chem.Mol(mol)],
        maxIters=steps,
        fireOptions=opts,
        fireDebugOutput=fire_debug,
    )
    debug = fire_debug[0][0]
    return FireRun(
        label=label or "fire",
        energies=list(debug.get("energies", [])),
        alphas=list(debug.get("alphas", [])),
        dts=list(debug.get("dt", [])),
        powers=list(debug.get("powers", [])),
    )


def plot_runs_energy(runs: list[FireRun], output_path: Path, *, rdkit_reference: float | None, title: str) -> None:
    plt.figure(figsize=(9, 5))
    for run in runs:
        plt.plot(run.energies, label=run.label)
    if rdkit_reference is not None:
        plt.axhline(rdkit_reference, color="k", linestyle="--", label=f"RDKit ref ({rdkit_reference:.4f})")
    plt.xlabel("Step")
    plt.ylabel("Energy (kcal/mol)")
    plt.title(title)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_run_diagnostic_panel(run: FireRun, output_path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    x_range = min(200, len(run.energies))
    axes[0].plot(run.alphas[:x_range])
    axes[0].set_title("Alpha per step")
    axes[0].set_ylabel("alpha")
    axes[1].plot(run.dts[:x_range])
    axes[1].set_title("dt per step")
    axes[1].set_ylabel("dt (ps)")
    axes[2].plot(run.powers[:x_range])
    axes[2].set_title("Power per step")
    axes[2].set_ylabel("v . F")
    axes[3].plot(run.energies[:x_range])
    axes[3].set_title("Energy per step")
    axes[3].set_ylabel("Energy (kcal/mol)")
    axes[3].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def four_variant_comparison(mol: Chem.Mol, max_steps: int, output_dir: Path, rdkit_reference: float | None) -> FireRun:
    base_kwargs = dict(steps=max_steps, dt_init=0.0001, alpha_init=0.25, time_step_increment=1.1, use_masses=True)
    variants = [
        run_single_minimization(mol, half_step=True, use_abc=False, label="half-step ON, ABC OFF", **base_kwargs),
        run_single_minimization(mol, half_step=False, use_abc=False, label="half-step OFF, ABC OFF", **base_kwargs),
        run_single_minimization(mol, half_step=True, use_abc=True, label="half-step ON, ABC ON", **base_kwargs),
        run_single_minimization(mol, half_step=False, use_abc=True, label="half-step OFF, ABC ON", **base_kwargs),
    ]
    plot_runs_energy(
        variants,
        output_dir / "four_variants.png",
        rdkit_reference=rdkit_reference,
        title="FIRE 2.0 - half-step-back x ABC variants",
    )
    plot_run_diagnostic_panel(variants[0], output_dir / "diagnostic_half_step_no_abc.png")
    return variants[0]


def parameter_sweep(mol: Chem.Mol, max_steps: int, output_dir: Path, rdkit_reference: float | None) -> None:
    scan = {
        "dt_init": [0.0001, 0.001],
        "time_step_increment": [1.1, 1.3, 1.5],
        "n_min_for_increase": [0, 5, 20],
        "max_step": [0.0],
        "dt_max_factor": [5.0, 10.0],
        "use_abc": [False, True],
    }
    keys = list(scan.keys())
    combos = list(itertools.product(*[scan[k] for k in keys]))

    plt.figure(figsize=(11, 6))
    interesting: list[tuple[str, list[float]]] = []
    for tup in tqdm.tqdm(combos, desc="param scan"):
        kwargs = dict(zip(keys, tup))
        run = run_single_minimization(
            mol,
            steps=max_steps,
            half_step=True,
            use_masses=True,
            label="dt={dt_init}, finc={time_step_increment}, Nmin={n_min_for_increase}, abc={use_abc}".format(
                **kwargs
            ),
            **kwargs,
        )
        if run.energies and run.energies[-1] < 1e4:
            interesting.append((run.label, run.energies))

    for label, energies in interesting:
        plt.plot(energies, alpha=0.5, label=label)
    if rdkit_reference is not None:
        plt.axhline(rdkit_reference, color="k", linestyle="--", label=f"RDKit ref ({rdkit_reference:.4f})")
    plt.xlabel("Step")
    plt.ylabel("Energy (kcal/mol)")
    plt.title(f"FIRE 2.0 parameter sweep ({len(interesting)} of {len(combos)} runs converged below 1e4)")
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_sweep.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mol = load_first_mol(args.input_sdf)

    print(f"Running four-variant comparison on first molecule of {args.input_sdf} ({mol.GetNumAtoms()} atoms)...")
    four_variant_comparison(mol, args.max_steps, args.output_dir, args.rdkit_reference)

    if not args.no_scan:
        print("Running parameter sweep...")
        parameter_sweep(mol, args.max_steps, args.output_dir, args.rdkit_reference)

    print(f"Wrote figures to {args.output_dir}/")


if __name__ == "__main__":
    main()
