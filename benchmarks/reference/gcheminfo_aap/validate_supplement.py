# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and validate the compatibility runner against the authors' SDF."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from rdkit import Chem

from gcheminfo_aap import read_assignments, write_graphs


ACTIVITY_TAG = "PF proliferation inhibition 3D7 EC50 uM"
IDENTIFIER_TAG = "GNF-Pf identifier"


def load_input(path: Path):
    molecules = [molecule for molecule in Chem.SDMolSupplier(str(path), removeHs=False)]
    if any(molecule is None for molecule in molecules):
        raise ValueError(f"RDKit failed to parse at least one molecule from {path}")
    return molecules


def prepare(input_sdf: Path, graph_output: Path, smiles_output: Path | None = None) -> None:
    molecules = load_input(input_sdf)
    indexed = list(enumerate(molecules))
    # autocorrelator.apps.NumericComparator sorts missing/empty values last.
    def activity_key(item):
        value = item[1].GetProp(ACTIVITY_TAG).lstrip("<")
        return (not bool(value), float(value) if value else 0.0)

    indexed.sort(key=activity_key)
    write_graphs(
        [molecule for _, molecule in indexed],
        graph_output,
        [original_index for original_index, _ in indexed],
    )
    if smiles_output is not None:
        with smiles_output.open("w", encoding="utf-8") as stream:
            for _, molecule in indexed:
                stream.write(Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=True) + "\n")
    print(json.dumps({"molecules": len(indexed), "graph_output": str(graph_output)}))


def _load_expected(reference_sdf_gz: Path):
    with gzip.open(reference_sdf_gz, "rb") as stream:
        supplier = Chem.ForwardSDMolSupplier(stream, removeHs=False)
        expected = {}
        for molecule in supplier:
            if molecule is None:
                raise ValueError(f"RDKit failed to parse a molecule from {reference_sdf_gz}")
            expected[molecule.GetProp(IDENTIFIER_TAG)] = {
                "cluster": int(molecule.GetProp("clusterIdx")),
                "centroid": molecule.HasProp("centroidIdx"),
                "similarity": float(molecule.GetProp("NNSim")),
            }
    return expected


def compare(input_sdf: Path, reference_sdf_gz: Path, assignments_tsv: Path) -> None:
    molecules = load_input(input_sdf)
    identifiers = [molecule.GetProp(IDENTIFIER_TAG) for molecule in molecules]
    expected = _load_expected(reference_sdf_gz)
    actual_rows = read_assignments(assignments_tsv)
    actual = {int(row["input_index"]): row for row in actual_rows}

    cluster_matches = 0
    centroid_matches = 0
    similarity_matches_at_published_precision = 0
    absolute_errors = []
    for input_index, identifier in enumerate(identifiers):
        expected_row = expected[identifier]
        actual_row = actual[input_index]
        cluster_matches += int(int(actual_row["cluster_index"]) == expected_row["cluster"])
        centroid_matches += int(bool(int(actual_row["is_centroid"])) == expected_row["centroid"])
        error = abs(float(actual_row["similarity"]) - expected_row["similarity"])
        absolute_errors.append(error)
        similarity_matches_at_published_precision += int(error <= 0.0005000001)

    count = len(molecules)
    summary = {
        "molecules": count,
        "expected_clusters": len({row["cluster"] for row in expected.values()}),
        "actual_clusters": len({int(row["cluster_index"]) for row in actual_rows}),
        "cluster_assignment_matches": cluster_matches,
        "cluster_assignment_fraction": cluster_matches / count,
        "centroid_flag_matches": centroid_matches,
        "centroid_flag_fraction": centroid_matches / count,
        "similarity_matches_at_3dp": similarity_matches_at_published_precision,
        "similarity_match_fraction": similarity_matches_at_published_precision / count,
        "similarity_mae_vs_3dp_reference": sum(absolute_errors) / count,
        "similarity_max_error_vs_3dp_reference": max(absolute_errors),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("input_sdf", type=Path)
    prepare_parser.add_argument("graph_output", type=Path)
    prepare_parser.add_argument("--smiles-output", type=Path)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("input_sdf", type=Path)
    compare_parser.add_argument("reference_sdf_gz", type=Path)
    compare_parser.add_argument("assignments_tsv", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.input_sdf, args.graph_output, args.smiles_output)
    else:
        compare(args.input_sdf, args.reference_sdf_gz, args.assignments_tsv)


if __name__ == "__main__":
    main()
