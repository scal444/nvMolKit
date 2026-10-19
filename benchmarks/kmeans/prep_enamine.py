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

"""Enamine REAL .cxsmiles (tab-separated, `smiles` and `id` columns) -> Morgan r2 1024-bit packed fingerprints.

Writes `<out_dir>/morgan2_1024.npy` (uint8 rows, numpy.packbits bitorder="little", same layout as the
ChEMBL34 set: `.view("<u4")` is nvMolKit's packed input), row-aligned `ids.txt`, and `failures.tsv`.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

FP_SIZE = 1024


def _fingerprint_chunk(lines):
    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)
    packed, ids, failures = [], [], []
    for line in lines:
        fields = line.rstrip("\n").split("\t")
        mol = Chem.MolFromSmiles(fields[0])
        if mol is None:
            failures.append((fields[1], fields[0]))
            continue
        packed.append(np.packbits(generator.GetFingerprintAsNumPy(mol).astype(bool), bitorder="little"))
        ids.append(fields[1])
    rows = np.stack(packed) if packed else np.zeros((0, FP_SIZE // 8), dtype=np.uint8)
    return rows, ids, failures


def _chunks(handle, size):
    chunk = []
    for line in handle:
        chunk.append(line)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cxsmiles", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=20000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    blocks, all_ids, all_failures = [], [], []
    with args.cxsmiles.open() as handle, Pool(args.workers) as pool:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:2] != ["smiles", "id"]:
            raise ValueError(f"expected 'smiles\\tid' header, got {header[:2]}")
        for rows, ids, failures in pool.imap(_fingerprint_chunk, _chunks(handle, args.chunk_size)):
            blocks.append(rows)
            all_ids.extend(ids)
            all_failures.extend(failures)
            if len(blocks) % 50 == 0:
                print(f"{len(all_ids)} fingerprints", flush=True)

    np.save(args.out_dir / "morgan2_1024.npy", np.concatenate(blocks))
    (args.out_dir / "ids.txt").write_text("\n".join(all_ids) + "\n")
    (args.out_dir / "failures.tsv").write_text("".join(f"{i}\t{s}\n" for i, s in all_failures))
    summary = {"input": str(args.cxsmiles), "kept": len(all_ids), "failed_parse": len(all_failures)}
    (args.out_dir / "prep_summary.json").write_text(json.dumps(summary, indent=1))
    print(summary)


if __name__ == "__main__":
    main()
