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

"""
Serialize a SMILES file into a pickled list of RDKit binary molecules.

The output can be used with benchmarks/substruct_bench.py via --pickle.
"""

from __future__ import annotations

import pickle
import sys

from rdkit import Chem, RDLogger
from tqdm.contrib.concurrent import process_map


def _parse_smiles(smi: str) -> Chem.Mol | None:
    """Parse a single SMILES string."""
    return Chem.MolFromSmiles(smi, sanitize=True)


def _load_smiles_binaries(
    filepath: str,
) -> list[bytes]:
    """Load and parse molecules from a SMILES file into binary RDKit mols."""
    mol_binaries: list[bytes] = []
    smiles_list: list[str] = []

    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            smi = line.split()[0]
            if i == 0:
                RDLogger.DisableLog("rdApp.*")
                mol = Chem.MolFromSmiles(smi, sanitize=True)
                RDLogger.EnableLog("rdApp.*")
                if mol:
                    mol_binaries.append(mol.ToBinary())
            else:
                smiles_list.append(smi)

    if smiles_list:
        parsed = process_map(
            _parse_smiles,
            smiles_list,
            desc="Parsing molecules",
            chunksize=1000,
        )

        parse_failures = 0
        for mol in parsed:
            if mol is None:
                parse_failures += 1
            else:
                mol_binaries.append(mol.ToBinary())

        if parse_failures > 0:
            print(f"    ({parse_failures} parse failures)")

    print(f"  Loaded {len(mol_binaries)} molecules from {filepath}")
    return mol_binaries


def _print_usage() -> None:
    print("Usage: python serialize_smiles.py <input_smiles> <output_pickle>")


def main() -> None:
    if len(sys.argv) != 3:
        _print_usage()
        sys.exit(1)
    input_smiles = sys.argv[1]
    output_pickle = sys.argv[2]

    print("Loading SMILES...")
    mol_binaries = _load_smiles_binaries(input_smiles)
    if not mol_binaries:
        print("Error: No valid molecules loaded")
        sys.exit(1)

    print(f"Writing pickle: {output_pickle}")
    with open(output_pickle, "wb") as f:
        pickle.dump(mol_binaries, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Wrote {len(mol_binaries)} molecules")


if __name__ == "__main__":
    main()
