# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-time data prep: 1M random Enamine REAL targets + query pool from the first (smaller) half."""

import pickle
import random
from multiprocessing import Pool

from rdkit import Chem, RDLogger

SOURCE = "/home/kevin/data/enamine_real_10M.csxmiles"
NUM_TARGETS = 1_000_000
NUM_QUERY_POOL = 20_000
SEED = 42


def to_binary(smiles):
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else mol.ToBinary()


def main():
    with open(SOURCE) as fh:
        next(fh)
        smiles = [line.split("\t", 1)[0] for line in fh]
    rng = random.Random(SEED)
    target_idx = rng.sample(range(len(smiles)), NUM_TARGETS)
    half = len(smiles) // 2
    query_idx = rng.sample(range(half), NUM_QUERY_POOL)
    with Pool(16) as pool:
        binaries = pool.map(to_binary, [smiles[i] for i in target_idx], chunksize=5000)
    binaries = [b for b in binaries if b is not None]
    with open("targets_1M.pkl", "wb") as fh:
        pickle.dump(binaries, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open("queries_pool.smi", "w") as fh:
        fh.writelines(smiles[i] + "\n" for i in query_idx)
    print(f"targets {len(binaries)}, query pool {len(query_idx)} (source rows {len(smiles)}, half {half})")


if __name__ == "__main__":
    main()
