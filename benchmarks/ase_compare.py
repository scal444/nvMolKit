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

"""Compare per-step FIRE 2.0 energies for a single molecule across:

  - nvmolkit's FIRE 2.0 minimizer (driven via the MMFF batched forcefield)
  - ASE's FIRE 2.0 minimizer driving an RDKit-MMFF calculator (reference impl)
  - RDKit's own MMFF minimizer (reference final energy only)

Loads the first molecule from an SDF and writes per-step energy plots and a
summary CSV to disk. Used to confirm nvmolkit's FIRE matches the ASE reference
on a real molecule (the C++ test_fire_minimizer suite already covers synthetic
harmonic systems against an in-tree reference).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE2
from ase.units import eV, kcal
from ase.units import mol as avogadro_number
from rdkit import Chem
from rdkit.Chem import AllChem

from nvmolkit.mmffOptimization import FireOptions, MMFFOptimizeMoleculesConfsFire


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_sdf", type=Path, help="Input SDF; the first molecule is used.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ase_compare"),
        help="Directory for output figures and CSV (default: ./ase_compare).",
    )
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum FIRE steps for both backends (default: 300).")
    parser.add_argument(
        "--mass-weighting",
        action="store_true",
        help="Enable nvmolkit FIRE mass weighting. ASE FIRE always treats mass=1; this is for ablation.",
    )
    parser.add_argument("--use-abc", action="store_true", help="Enable ABC-FIRE in both backends.")
    return parser.parse_args()


def load_first_mol(sdf_path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for mol in supplier:
        if mol is not None:
            return mol
    raise ValueError(f"No molecules in {sdf_path}.")


class RDKitMMFFCalculator(Calculator):
    """ASE Calculator that delegates to RDKit's MMFF94. Energies/forces converted to ASE units."""

    implemented_properties = ("energy", "forces")

    def __init__(self, mol: Chem.Mol):
        super().__init__()
        # Operate on a private copy so we don't disturb the caller's mol.
        self._mol = Chem.Mol(mol)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        conf = self._mol.GetConformer()
        for atom_idx in range(self._mol.GetNumAtoms()):
            conf.SetAtomPosition(atom_idx, tuple(positions[atom_idx]))
        ff = AllChem.MMFFGetMoleculeForceField(
            self._mol,
            AllChem.MMFFGetMoleculeProperties(self._mol),
            confId=conf.GetId(),
        )
        energy_kcal = ff.CalcEnergy()
        gradient_kcal = np.array(ff.CalcGrad()).reshape(-1, 3)
        forces_kcal = -gradient_kcal
        self.results = {
            "energy": energy_kcal * (kcal / avogadro_number) / eV,  # eV
            "forces": forces_kcal * (kcal / avogadro_number) / eV,  # eV/Å
        }


def run_ase_fire2(mol: Chem.Mol, max_steps: int, use_abc: bool) -> list[float]:
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    conf = mol.GetConformer()
    positions = [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.calc = RDKitMMFFCalculator(mol)
    dyn = FIRE2(atoms, logfile=None, use_abc=use_abc)
    energies_kcal: list[float] = []
    energies_kcal.append(float(atoms.get_potential_energy() * avogadro_number / kcal))
    for _ in range(max_steps):
        dyn.step()
        energies_kcal.append(float(atoms.get_potential_energy() * avogadro_number / kcal))
    return energies_kcal


def run_nvmolkit_fire(mol: Chem.Mol, max_steps: int, mass_weighting: bool, use_abc: bool) -> list[float]:
    opts = FireOptions()
    opts.useMass = mass_weighting
    opts.abcCorrection = use_abc
    opts.takeHalfStepBack = True
    opts.dtInit = 0.001
    opts.gradTol = 1e-8  # don't let it terminate early; we want all max_steps points
    fire_debug: list = []
    MMFFOptimizeMoleculesConfsFire(
        [Chem.Mol(mol)],
        maxIters=max_steps,
        fireOptions=opts,
        fireDebugOutput=fire_debug,
    )
    return list(fire_debug[0][0].get("energies", []))


def rdkit_final_energy(mol: Chem.Mol, max_iters: int) -> float:
    results = AllChem.MMFFOptimizeMoleculeConfs(Chem.Mol(mol), maxIters=max_iters, mmffVariant="MMFF94")
    return float(results[0][1]) if results else float("nan")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mol = load_first_mol(args.input_sdf)
    print(f"Comparing FIRE 2.0 backends on first molecule of {args.input_sdf} ({mol.GetNumAtoms()} atoms).")

    print("Running ASE FIRE2 (RDKit MMFF calculator)...")
    ase_energies = run_ase_fire2(mol, args.max_steps, args.use_abc)
    print("Running nvmolkit FIRE 2.0...")
    nvm_energies = run_nvmolkit_fire(mol, args.max_steps, args.mass_weighting, args.use_abc)
    print("Computing RDKit MMFF94 reference final energy...")
    rdkit_ref = rdkit_final_energy(mol, args.max_steps * 4)

    plt.figure(figsize=(9, 5))
    plt.plot(ase_energies, label="ASE FIRE2 (RDKit-MMFF calculator)", linewidth=1)
    plt.plot(nvm_energies, label="nvmolkit FIRE 2.0", linewidth=1, linestyle="--")
    plt.axhline(rdkit_ref, color="k", linestyle=":", linewidth=1, label=f"RDKit MMFF94 ref ({rdkit_ref:.4f})")
    plt.xlabel("Step")
    plt.ylabel("Energy (kcal/mol)")
    plt.title(
        f"FIRE 2.0 trajectory comparison "
        f"(mass={args.mass_weighting}, abc={args.use_abc})",
    )
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(args.output_dir / "trajectory.png", dpi=150)
    plt.close()

    csv_path = args.output_dir / "trajectory.csv"
    n = max(len(ase_energies), len(nvm_energies))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "ase_fire2_kcalmol", "nvmolkit_fire_kcalmol"])
        for step in range(n):
            ase_val = ase_energies[step] if step < len(ase_energies) else ""
            nvm_val = nvm_energies[step] if step < len(nvm_energies) else ""
            writer.writerow([step, ase_val, nvm_val])
    print(f"Wrote {args.output_dir / 'trajectory.png'} and {csv_path}.")
    print(f"Summary: ASE final={ase_energies[-1]:.4f}, nvmolkit final={nvm_energies[-1]:.4f}, RDKit ref={rdkit_ref:.4f} kcal/mol")


if __name__ == "__main__":
    main()
