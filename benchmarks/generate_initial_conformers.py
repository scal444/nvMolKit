# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate initial conformers for molecules and write them to an SDF file."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed conformers for molecules listed in a SMILES file and write them to an SDF file.",
    )
    parser.add_argument("smiles_path", type=Path, help="Path to the input SMILES file.")
    parser.add_argument("output_sdf", type=Path, help="Path to the output SDF file.")
    parser.add_argument(
        "--num-molecules",
        type=int,
        default=100,
        help="Number of molecules to sample from the SMILES file (default: 100).",
    )
    parser.add_argument(
        "--num-conformers",
        type=int,
        default=10,
        help="Target number of conformers to generate per molecule (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for SMILES shuffling and conformer embedding (default: 42).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1000,
        help="Maximum number of embedding attempts per conformer (default: 1000).",
    )
    return parser.parse_args()


def load_smiles(smiles_path: Path, num_molecules: int, seed: int) -> list[str]:
    with smiles_path.open("r", encoding="utf-8") as handle:
        smiles = [line.strip() for line in handle if line.strip()]
    if not smiles:
        raise ValueError(f"No SMILES entries found in {smiles_path}.")
    random.seed(seed)
    random.shuffle(smiles)
    return smiles[:num_molecules]


def embed_conformers(
    smiles_list: list[str],
    num_conformers: int,
    seed: int,
    max_attempts: int,
) -> list[Chem.Mol]:
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0
    params.maxAttempts = max_attempts
    params.pruneRmsThresh = 0.1
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.useBasicKnowledge = True
    params.enforceChirality = True

    embedded: list[Chem.Mol] = []
    partial = 0
    failures = 0

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failures += 1
            continue
        mol = Chem.AddHs(mol)
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
        if not conf_ids:
            failures += 1
            continue
        if len(conf_ids) < num_conformers:
            partial += 1
        mol.SetProp("OriginalSMILES", smi)
        mol.SetProp("MoleculeIndex", str(idx))
        embedded.append(mol)

    print(
        f"Embedded {len(embedded)} molecules. "
        f"{partial} had fewer than {num_conformers} conformers. {failures} failed entirely."
    )
    return embedded


def write_conformers_to_sdf(mols: list[Chem.Mol], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_path))
    if writer is None:
        raise RuntimeError(f"Unable to create SDWriter for {output_path}.")

    total_confs = 0
    for mol in mols:
        name = mol.GetProp("MoleculeIndex") if mol.HasProp("MoleculeIndex") else "mol"
        mol.SetProp("_Name", name)
        for conf in mol.GetConformers():
            mol.SetIntProp("ConformerID", conf.GetId())
            writer.write(mol, confId=conf.GetId())
            total_confs += 1

    writer.close()
    print(f"Wrote {total_confs} conformers to {output_path}.")


def main() -> None:
    args = parse_args()
    smiles = load_smiles(args.smiles_path, args.num_molecules, args.seed)
    mols = embed_conformers(smiles, args.num_conformers, args.seed, args.max_attempts)
    if not mols:
        raise RuntimeError("No molecules were embedded; nothing to write.")
    write_conformers_to_sdf(mols, args.output_sdf)


if __name__ == "__main__":
    main()


