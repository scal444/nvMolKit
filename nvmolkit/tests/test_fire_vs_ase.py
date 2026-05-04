# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step-by-step parity tests between nvMolKit FIRE and ASE FIRE2 on MMFF.

Both optimizers see the same starting geometry, the same RDKit MMFF94 force field, and
the same FIRE parameters. The tests check that they agree on:

    1. Initial energy and gradient (parity of the calculator side).
    2. Per-step energy / position behavior at 1, 10, 50, 200 steps.

Failures in (1) point to a force-field bug. Failures in (2) point to an integrator bug.
ASE FIRE2 is the reference; nvMolKit is expected to track it within 1e-3 relative error
on energies and reasonable per-component agreement on positions for a clean unit
match.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE2 as ASEFire2
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import ETKDGv3

from nvmolkit.mmffOptimization import FireOptions, MMFFOptimizeMoleculesConfsFire


SMILES_LIST = ["CCO", "CCN", "CC(=O)O", "c1ccccc1", "OCC(O)CO"]


def _embed(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True
    AllChem.EmbedMolecule(mol, params)
    return mol


def _perturb(mol: Chem.Mol, sigma: float, seed: int) -> Chem.Mol:
    rng = np.random.default_rng(seed)
    perturbed = Chem.Mol(mol)
    conf = perturbed.GetConformer(0)
    positions = np.asarray(conf.GetPositions(), dtype=float)
    noise = rng.normal(0.0, sigma, positions.shape)
    new_positions = positions + noise
    for atom_idx in range(perturbed.GetNumAtoms()):
        conf.SetAtomPosition(atom_idx, new_positions[atom_idx].tolist())
    return perturbed


def _rdkit_energy_and_grad(mol: Chem.Mol) -> tuple[float, np.ndarray]:
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=mol.GetConformer(0).GetId())
    energy = ff.CalcEnergy()
    grad = np.asarray(ff.CalcGrad()).reshape(-1, 3)
    return float(energy), grad


class _RdkitMmffCalculator(Calculator):
    """ASE Calculator backed by RDKit MMFF94. Forces = -gradient."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, mol: Chem.Mol) -> None:
        super().__init__()
        self._mol = mol
        self._props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes) -> None:  # noqa: D401
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        conf = self._mol.GetConformer(0)
        for idx in range(self._mol.GetNumAtoms()):
            conf.SetAtomPosition(idx, positions[idx].tolist())
        ff = AllChem.MMFFGetMoleculeForceField(self._mol, self._props)
        energy = ff.CalcEnergy()
        grad = np.asarray(ff.CalcGrad()).reshape(-1, 3)
        self.results = {"energy": float(energy), "forces": -grad}


def _ase_atoms(mol: Chem.Mol) -> Atoms:
    conf = mol.GetConformer(0)
    positions = np.asarray(conf.GetPositions(), dtype=float)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.calc = _RdkitMmffCalculator(Chem.Mol(mol))
    return atoms


def _make_fire_options() -> FireOptions:
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


@pytest.mark.parametrize("smiles", SMILES_LIST)
def test_initial_energy_parity(smiles: str) -> None:
    """nvMolKit and RDKit should agree on the initial MMFF energy."""
    mol = _perturb(_embed(smiles), sigma=0.05, seed=1)
    rdkit_energy, _ = _rdkit_energy_and_grad(mol)

    nvm_mol = Chem.Mol(mol)
    energies = MMFFOptimizeMoleculesConfsFire(
        [nvm_mol],
        maxIters=0,
        fireOptions=_make_fire_options(),
    )
    nvm_energy = energies[0][0]

    rel_err = abs(nvm_energy - rdkit_energy) / max(abs(rdkit_energy), 1e-6)
    assert rel_err < 1e-3, (
        f"Initial MMFF energy mismatch for {smiles}: rdkit={rdkit_energy:.6f}, "
        f"nvm={nvm_energy:.6f}, rel_err={rel_err:.3e}"
    )


def _atoms_positions(atoms: Atoms) -> np.ndarray:
    return atoms.get_positions().copy()


def _mol_positions(mol: Chem.Mol) -> np.ndarray:
    return np.asarray(mol.GetConformer(0).GetPositions(), dtype=float)


@pytest.mark.parametrize("smiles", SMILES_LIST)
@pytest.mark.parametrize("steps", [1, 10, 50, 200])
def test_fire_vs_ase_step_parity(smiles: str, steps: int) -> None:
    """After ``steps`` FIRE iterations starting from the same perturbed geometry,
    nvMolKit and ASE FIRE2 should agree on energy and positions within tight tolerance.
    """
    base = _perturb(_embed(smiles), sigma=0.05, seed=1)
    options = _make_fire_options()

    ase_atoms = _ase_atoms(base)
    optimizer = ASEFire2(
        ase_atoms,
        logfile=None,
        dt=options.dtInit,
        maxstep=options.dMax,
        dtmax=options.dtInit * options.dtMaxFactor,
        dtmin=options.dtInit * options.dtMinFactor,
        Nmin=options.nMinForIncrease,
        astart=options.alphaInit,
        fa=options.alphaDecrement,
        finc=options.timeStepIncrement,
        fdec=options.timeStepDecrement,
    )
    optimizer.run(fmax=options.gradTol, steps=steps)
    ase_positions = _atoms_positions(ase_atoms)
    ase_calc = _RdkitMmffCalculator(Chem.Mol(base))
    ase_calc.calculate(ase_atoms, ["energy"], all_changes)
    ase_energy = ase_calc.results["energy"]

    nvm_mol = Chem.Mol(base)
    nvm_energies = MMFFOptimizeMoleculesConfsFire(
        [nvm_mol],
        maxIters=steps,
        fireOptions=options,
    )
    nvm_positions = _mol_positions(nvm_mol)
    nvm_energy = nvm_energies[0][0]

    pos_diff = np.linalg.norm(ase_positions - nvm_positions, axis=1)
    max_pos_diff = float(pos_diff.max())
    energy_diff = abs(ase_energy - nvm_energy)
    energy_rel = energy_diff / max(abs(ase_energy), 1e-6)

    assert energy_rel < 1e-3, (
        f"steps={steps} smiles={smiles}: energy mismatch "
        f"ase={ase_energy:.6f} nvm={nvm_energy:.6f} rel_err={energy_rel:.3e} "
        f"max_pos_diff={max_pos_diff:.4f} A"
    )
    assert max_pos_diff < 1e-3, (
        f"steps={steps} smiles={smiles}: position drift {max_pos_diff:.4f} A "
        f"(ase_E={ase_energy:.6f} nvm_E={nvm_energy:.6f})"
    )
