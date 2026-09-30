# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the RDKit SubstructLibrary baseline across holders and thread counts (200 queries, 1M targets)."""

import pickle
import time

from rdkit import Chem
from rdkit.Chem import rdSubstructLibrary

MATCH_OPTIONS = {"recursionPossible": True, "useChirality": False, "useQueryQueryMatches": False}


def run(library, queries, label, threads):
    """Time HasMatch over the queries, one query at a time."""
    start = time.time()
    hits = [library.HasMatch(query, numThreads=threads, **MATCH_OPTIONS) for query in queries]
    per_query_ms = 1e3 * (time.time() - start) / len(queries)
    print(f"{label} threads={threads}: {per_query_ms:.3f} ms/query, hits={sum(hits)}", flush=True)


def main():
    with open("targets_1M.pkl", "rb") as fh:
        binaries = pickle.load(fh)
    with open("queries_pool.smi") as fh:
        queries = [Chem.MolFromSmiles(line.strip()) for line in fh][:200]
    for query in queries:
        Chem.RemoveStereochemistry(query)

    start = time.time()
    for query in queries:
        Chem.PatternFingerprint(query)
    print(f"query PatternFingerprint alone: {1e6 * (time.time() - start) / len(queries):.0f} us/query")

    cached = rdSubstructLibrary.CachedMolHolder()
    for binary in binaries:
        cached.AddBinary(binary)
    library = rdSubstructLibrary.SubstructLibrary(cached)
    rdSubstructLibrary.AddPatterns(library, numThreads=16)
    for threads in (16, 1):
        run(library, queries, "CachedMolHolder+PatternHolder", threads)

    molecules = rdSubstructLibrary.MolHolder()
    for binary in binaries:
        molecules.AddMol(Chem.Mol(binary))
    run(
        rdSubstructLibrary.SubstructLibrary(molecules, library.GetFpHolder()),
        queries,
        "MolHolder+PatternHolder(shared fps)",
        16,
    )


if __name__ == "__main__":
    main()
