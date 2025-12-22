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
Benchmark script to collect RDKit substructure matching performance data.

Usage:
    python substruct_data_collection_rdkit.py <targets_file> <queries_file> <output_csv>

targets_file: File containing target molecules (SMILES), one per line
queries_file: File containing query patterns (SMILES or SMARTS), one per line
output_csv: Path to output CSV file
"""

import argparse
import re
import time
from typing import Tuple

import pandas as pd
from rdkit import Chem


def count_smarts_recursion(smarts: str) -> Tuple[int, int]:
    """
    Count the number and maximum depth of recursive SMARTS patterns.
    Recursive SMARTS use the $(...) syntax.

    Returns:
        (recursion_count, max_depth)
    """
    recursion_count = 0
    max_depth = 0
    current_depth = 0
    i = 0

    while i < len(smarts):
        if i < len(smarts) - 1 and smarts[i] == "$" and smarts[i + 1] == "(":
            recursion_count += 1
            current_depth += 1
            max_depth = max(max_depth, current_depth)
            i += 2
        elif smarts[i] == "(" and current_depth > 0:
            i += 1
        elif smarts[i] == ")" and current_depth > 0:
            current_depth -= 1
            i += 1
        else:
            i += 1

    return recursion_count, max_depth


def get_num_atoms(mol: Chem.Mol) -> int:
    """Get number of atoms in a molecule (works for both Mol and query patterns)."""
    if mol is None:
        return 0
    return mol.GetNumAtoms()


def time_substruct_match(target: Chem.Mol, query: Chem.Mol, max_matches: int = 0) -> Tuple[float, int]:
    """
    Time a single substructure match operation.

    Args:
        target: Target molecule
        query: Query pattern
        max_matches: Maximum number of matches to find (0 = unlimited)

    Returns:
        (time_seconds, num_matches)
    """
    start = time.perf_counter()
    matches = target.GetSubstructMatches(query, maxMatches=max_matches)
    end = time.perf_counter()

    return end - start, len(matches)


def load_molecules(filepath: str, as_smarts: bool = False) -> list:
    """
    Load molecules from a file (one SMILES/SMARTS per line).

    Args:
        filepath: Path to input file
        as_smarts: If True, parse as SMARTS patterns

    Returns:
        List of (original_string, Mol) tuples
    """
    molecules = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            mol_str = parts[0]

            if as_smarts:
                mol = Chem.MolFromSmarts(mol_str)
            else:
                mol = Chem.MolFromSmiles(mol_str)

            molecules.append((mol_str, mol))

    return molecules


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RDKit substructure matching and collect timing data."
    )
    parser.add_argument("targets_file", help="File containing target molecules (SMILES)")
    parser.add_argument("queries_file", help="File containing query patterns (SMILES or SMARTS)")
    parser.add_argument("output_csv", help="Output CSV file path")
    parser.add_argument(
        "--queries-are-smiles",
        action="store_true",
        help="Parse queries as SMILES instead of SMARTS",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=0,
        help="Maximum matches per query (0 = unlimited)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of timing runs per pair (results averaged)",
    )
    args = parser.parse_args()

    targets = load_molecules(args.targets_file, as_smarts=False)
    queries = load_molecules(args.queries_file, as_smarts=not args.queries_are_smiles)

    print(f"Loaded {len(targets)} targets and {len(queries)} queries")

    valid_targets = [(s, m) for s, m in targets if m is not None]
    valid_queries = [(s, m) for s, m in queries if m is not None]

    if len(valid_targets) < len(targets):
        print(f"Warning: {len(targets) - len(valid_targets)} targets failed to parse")
    if len(valid_queries) < len(queries):
        print(f"Warning: {len(queries) - len(valid_queries)} queries failed to parse")

    results = []
    total_pairs = len(valid_targets) * len(valid_queries)
    pair_count = 0

    for target_str, target_mol in valid_targets:
        target_atoms = get_num_atoms(target_mol)

        for query_str, query_mol in valid_queries:
            pair_count += 1
            if pair_count % 1000 == 0:
                print(f"Progress: {pair_count}/{total_pairs} pairs")

            query_atoms = get_num_atoms(query_mol)
            recursion_count, recursion_depth = count_smarts_recursion(query_str)

            times = []
            num_matches = 0

            for _ in range(args.runs):
                elapsed, matches = time_substruct_match(
                    target_mol, query_mol, args.max_matches
                )
                times.append(elapsed)
                num_matches = matches

            avg_time = sum(times) / len(times)

            results.append(
                {
                    "target_smiles": target_str,
                    "query_pattern": query_str,
                    "time_seconds": avg_time,
                    "num_matches": num_matches,
                    "target_num_atoms": target_atoms,
                    "query_num_atoms": query_atoms,
                    "query_recursion_count": recursion_count,
                    "query_recursion_depth": recursion_depth,
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    print(f"Total pairs benchmarked: {len(results)}")


if __name__ == "__main__":
    main()

