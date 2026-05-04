# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tight diagnostic harness for nvMolKit ETKDG: BFGS vs FIRE wall time and yield.

Skips the RDKit reference and downstream MMFF scoring; the goal is fast
iteration on FIRE-side performance fixes. Reports wall time and per-pipeline
conformer yield (full / partial / zero), nothing else.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.rdDistGeom import ETKDGv3

from nvmolkit.embedMolecules import EmbedMolecules, MinimizerKind
from nvmolkit.types import HardwareOptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_smiles", type=Path)
    parser.add_argument("--num-molecules", type=int, default=50)
    parser.add_argument("--num-conformers", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
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


def make_params(seed: int, max_attempts: int) -> ETKDGv3:
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


def prepare(smiles_list: list[str]) -> list[Chem.Mol]:
    mols: list[Chem.Mol] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mols.append(Chem.AddHs(mol))
    return mols


def yield_counts(mols: list[Chem.Mol], target: int) -> tuple[int, int, int]:
    full = 0
    partial = 0
    zero = 0
    for mol in mols:
        num_confs = mol.GetNumConformers()
        if num_confs == 0:
            zero += 1
        elif num_confs < target:
            partial += 1
        else:
            full += 1
    return full, partial, zero


def run_pipeline(label: str, mols: list[Chem.Mol], params: ETKDGv3, num_conformers: int, kind: MinimizerKind) -> None:
    start = time.perf_counter()
    EmbedMolecules(
        mols,
        params,
        confsPerMolecule=num_conformers,
        maxIterations=-1,
        hardwareOptions=HardwareOptions(),
        minimizerKind=kind,
    )
    elapsed = time.perf_counter() - start
    full, partial, zero = yield_counts(mols, num_conformers)
    print(f"[{label}] elapsed={elapsed:.2f}s  full={full}  partial={partial}  zero={zero}")


def main() -> None:
    args = parse_args()
    smiles_list = load_smiles(args.input_smiles, args.num_molecules, args.seed)
    print(f"Loaded {len(smiles_list)} SMILES.")
    params = make_params(args.seed, args.max_attempts)

    mols_bfgs = prepare(smiles_list)
    mols_fire = prepare(smiles_list)

    run_pipeline("nvm_bfgs", mols_bfgs, params, args.num_conformers, MinimizerKind.BFGS)
    run_pipeline("nvm_fire", mols_fire, params, args.num_conformers, MinimizerKind.FIRE)


if __name__ == "__main__":
    main()
