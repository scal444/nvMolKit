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

"""Compare RDKit and nvmolkit MMFF energy minimization over sampled molecules."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs


SMILES_PATH = Path("/home/kboyd/data/chembl_size_splits/chembl_40-60.smi")
NUM_MOLECULES = 100
NUM_CONFORMERS = 10
MAX_ITERS = 10_000
RNG_SEED = 42


def load_smiles(smiles_path: Path) -> list[str]:
    with smiles_path.open("r", encoding="utf-8") as handle:
        smiles = [line.strip() for line in handle if line.strip()]
    if len(smiles) < NUM_MOLECULES:
        raise ValueError(
            f"Requested {NUM_MOLECULES} molecules but only found {len(smiles)} entries in {smiles_path}."
        )
    random.seed(RNG_SEED)
    random.shuffle(smiles)
    return smiles


def duplicate_conformers(mol: Chem.Mol) -> Chem.Mol:
    copy = Chem.Mol(mol)
    for conf_idx in range(mol.GetNumConformers()):
        source_conf = mol.GetConformer(conf_idx)
        target_conf = copy.GetConformer(conf_idx)
        for atom_idx in range(source_conf.GetNumAtoms()):
            target_conf.SetAtomPosition(atom_idx, source_conf.GetAtomPosition(atom_idx))
    return copy


def prepare_molecules(smiles: list[str]) -> tuple[list[Chem.Mol], list[Chem.Mol]]:
    rdkit_mols: list[Chem.Mol] = []
    nvmolkit_mols: list[Chem.Mol] = []

    params = AllChem.ETKDGv3()
    params.randomSeed = RNG_SEED
    params.numThreads = 0
    params.maxAttempts = 1000
    params.pruneRmsThresh = 0.1
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.useBasicKnowledge = True
    params.enforceChirality = True

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)

        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=NUM_CONFORMERS, params=params)
        if len(conf_ids) < NUM_CONFORMERS:
            continue

        rdkit_mols.append(mol)
        nvmolkit_mols.append(duplicate_conformers(mol))

        if len(rdkit_mols) == NUM_MOLECULES:
            break

    if len(rdkit_mols) < NUM_MOLECULES:
        raise RuntimeError(
            f"Unable to embed {NUM_MOLECULES} molecules with {NUM_CONFORMERS} conformers each; "
            f"only {len(rdkit_mols)} succeeded."
        )

    return rdkit_mols, nvmolkit_mols


def minimize_rdkit(mols: list[Chem.Mol]) -> list[float]:
    energies: list[float] = []
    failures = 0
    for mol in mols:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=MAX_ITERS)
        for status, energy in results:
            if energy is None:
                failures += 1
                continue
            energies.append(energy)
            if status != 0:
                failures += 1
    if failures:
        print(f"RDKit MMFF encountered {failures} non-converged conformers.")
    return energies


def minimize_nvmolkit(mols: list[Chem.Mol]) -> list[float]:
    energies_nested = MMFFOptimizeMoleculesConfs(mols, maxIters=MAX_ITERS)
    energies = [energy for mol_energies in energies_nested for energy in mol_energies if energy is not None]
    missing = sum(1 for mol_energies in energies_nested for energy in mol_energies if energy is None)
    if missing:
        print(f"nvmolkit MMFF reported {missing} conformers with missing energies.")
    return energies


def plot_histogram(rdkit_energies: list[float], nvmolkit_energies: list[float]) -> None:
    plt.figure(figsize=(12, 6))
    plt.hist(rdkit_energies, bins=50, alpha=0.6, label="RDKit MMFF")
    plt.hist(nvmolkit_energies, bins=50, alpha=0.6, label="nvmolkit MMFF")
    plt.xlabel("Energy (kcal/mol)")
    plt.ylabel("Count")
    plt.title("MMFF Minimized Energy Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    smiles = load_smiles(SMILES_PATH)
    rdkit_mols, nvmolkit_mols = prepare_molecules(smiles)

    rdkit_energies = minimize_rdkit(rdkit_mols)
    nvmolkit_energies = minimize_nvmolkit(nvmolkit_mols)

    print(f"Collected {len(rdkit_energies)} RDKit energies and {len(nvmolkit_energies)} nvmolkit energies.")

    plot_histogram(rdkit_energies, nvmolkit_energies)


if __name__ == "__main__":
    main()

