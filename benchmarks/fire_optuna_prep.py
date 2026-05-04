# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a fixed dataset for the FIRE Optuna study.

Steps:
    1. Sample SMILES from an Enamine CXSMILES file.
    2. RDKit ETKDG embed (10 confs/mol, maxAttempts=10).
    3. RDKit MMFF94 minimization to put each conformer at a basin.
    4. Apply Gaussian per-atom Cartesian noise (sigma=0.1 Å) to produce a perturbed
       starting state.
    5. Save the molecules with perturbed conformers as an SDF, plus per-molecule MMFF
       reference energies as a JSON file.

The perturbed SDF is what the Optuna trials read to measure how quickly each
configuration drives the conformers back to a basin.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import ETKDGv3
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_smiles", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-molecules", type=int, default=1000)
    parser.add_argument("--num-conformers", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=0.1, help="Per-atom Cartesian noise sigma in Angstrom.")
    parser.add_argument("--rdkit-mmff-iters", type=int, default=2000)
    return parser.parse_args()


def load_smiles(path: Path, num_molecules: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
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
    rng.shuffle(reservoir)
    return reservoir


def make_etkdg_params(seed: int, max_attempts: int, num_conformers: int) -> ETKDGv3:
    params = ETKDGv3()
    params.randomSeed = seed
    params.maxAttempts = max_attempts
    params.useRandomCoords = True
    params.pruneRmsThresh = -1.0
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.useBasicKnowledge = True
    params.enforceChirality = True
    params.numThreads = 0
    return params


def embed_and_minimize(smiles_list: list[str], num_conformers: int, params: ETKDGv3, mmff_iters: int) -> list[Chem.Mol]:
    mols: list[Chem.Mol] = []
    for smiles in tqdm(smiles_list, desc="prep"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
        if not conf_ids:
            continue
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=mmff_iters, mmffVariant="MMFF94")
        if any(status != 0 for status, _ in results):
            continue
        mol.SetProp("OriginalSMILES", smiles)
        for idx, (_, energy) in enumerate(results):
            mol.SetDoubleProp(f"_RefEnergy_{idx}", float(energy))
        mols.append(mol)
    return mols


def perturb_conformers(mols: list[Chem.Mol], sigma: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for mol in mols:
        for conf in mol.GetConformers():
            positions = conf.GetPositions()
            noise = rng.normal(loc=0.0, scale=sigma, size=positions.shape)
            new_positions = positions + noise
            for atom_idx in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(atom_idx, new_positions[atom_idx])


def save(mols: list[Chem.Mol], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = output_dir / "perturbed.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    ref_energies: list[list[float]] = []
    for mol_idx, mol in enumerate(mols):
        mol.SetIntProp("_MolIndex", mol_idx)
        per_mol: list[float] = []
        for conf in mol.GetConformers():
            energy_key = f"_RefEnergy_{conf.GetId()}"
            if mol.HasProp(energy_key):
                per_mol.append(mol.GetDoubleProp(energy_key))
            else:
                per_mol.append(float("nan"))
            writer.write(mol, confId=conf.GetId())
        ref_energies.append(per_mol)
    writer.close()
    metadata = {
        "num_molecules": len(mols),
        "ref_energies": ref_energies,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata))
    print(f"Saved {len(mols)} mols to {sdf_path}")


def main() -> None:
    args = parse_args()
    smiles = load_smiles(args.input_smiles, args.num_molecules, args.seed)
    print(f"Loaded {len(smiles)} SMILES.")
    params = make_etkdg_params(args.seed, args.max_attempts, args.num_conformers)
    mols = embed_and_minimize(smiles, args.num_conformers, params, args.rdkit_mmff_iters)
    print(f"Successfully prepared {len(mols)} mols with reference energies.")
    perturb_conformers(mols, args.sigma, args.seed + 1)
    save(mols, args.output_dir)


if __name__ == "__main__":
    main()
