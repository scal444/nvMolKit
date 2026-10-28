# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Morgan fingerprints for large fingerprint-search runs, computed once with RDKit and cached as packed words.

Molecules are taken at an even stride through the input file so that files ordered by size or series are sampled
across their whole range. The cache holds the packed words that nvMolKit searches; RDKit bit vectors and SMILES
are rebuilt from it on demand.
"""

import os
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ExplicitBitVect


def _read_strided_smiles(path, count, seed):
    with open(path) as handle:
        lines = handle.readlines()
    header = lines[0].split("\t")[0].lower() in ("smiles", "smi")
    lines = lines[1:] if header else lines
    rng = np.random.default_rng(seed)
    # Oversample so that unparseable lines can be dropped while keeping the requested count.
    picks = np.sort(rng.choice(len(lines), size=min(len(lines), int(count * 1.05) + 100), replace=False))
    return [lines[i].split()[0] for i in picks]


def _fingerprint_chunk(job):
    smiles, radius, fp_size = job
    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    rows = []
    kept = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bits = np.zeros(fp_size, dtype=np.uint8)
        bits[list(generator.GetFingerprint(mol).GetOnBits())] = 1
        rows.append(np.packbits(bits, bitorder="little").view("<u4"))
        kept.append(smi)
    return np.stack(rows) if rows else np.zeros((0, fp_size // 32), "<u4"), kept


def load_packed_fingerprints(path, count, radius, fp_size, workers, seed, cache_dir):
    """Return (packed uint32 array of shape (count, fp_size // 32), SMILES list), using a cache when present."""
    stem = f"{os.path.basename(path)}_n{count}_r{radius}_b{fp_size}_s{seed}"
    words_path = os.path.join(cache_dir, stem + ".npy")
    smiles_path = os.path.join(cache_dir, stem + ".smi")
    if os.path.exists(words_path) and os.path.exists(smiles_path):
        with open(smiles_path) as handle:
            return np.load(words_path), handle.read().split("\n")

    smiles = _read_strided_smiles(path, count, seed)
    chunk = (len(smiles) + workers * 8 - 1) // (workers * 8)
    jobs = [(smiles[i : i + chunk], radius, fp_size) for i in range(0, len(smiles), chunk)]
    with Pool(workers) as pool:
        parts = pool.map(_fingerprint_chunk, jobs)
    words = np.concatenate([p[0] for p in parts])[:count]
    kept = [s for p in parts for s in p[1]][:count]
    if len(kept) < count:
        raise ValueError(f"Only {len(kept)} parseable molecules in {path}, wanted {count}")

    os.makedirs(cache_dir, exist_ok=True)
    np.save(words_path, words)
    with open(smiles_path, "w") as handle:
        handle.write("\n".join(kept))
    return words, kept


def to_rdkit_bitvects(words):
    """Rebuild RDKit ExplicitBitVects from packed words (bit j of word w is fingerprint bit 32 * w + j)."""
    fp_size = words.shape[1] * 32
    bits = np.unpackbits(np.ascontiguousarray(words).view(np.uint8), axis=1, bitorder="little")
    vects = []
    for row in bits:
        vect = ExplicitBitVect(fp_size)
        vect.SetBitsFromList(np.flatnonzero(row).tolist())
        vects.append(vect)
    return vects
