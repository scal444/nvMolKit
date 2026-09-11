# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create benchmark-ready SMILES/score CSVs from the downloaded assay sources."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

PUBCHEM_AIDS = (485297, 485313, 588342, 686979)
NOVARTIS_SCORE = "PF proliferation inhibition 3D7 EC50 uM"
OUTPUT_FIELDS = (
    "smiles",
    "sort_value",
    "compound_id",
    "substance_id",
    "activity_score",
    "activity_outcome",
    "potency_um",
    "fit_log_ac50",
    "max_response",
)


def _write_outputs(directory, rows, metadata):
    output_path = directory / "molecules.csv"
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (directory / "manifest.json").open("w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"Wrote {len(rows)} molecules to {output_path}")


def _score_stats(rows):
    counts = Counter(row["sort_value"] for row in rows)
    return {
        "molecule_rows": len(rows),
        "distinct_sort_values": len(counts),
        "priority_tie_fraction": 1.0 - len(counts) / len(rows),
        "largest_priority_tie": max(counts.values()),
    }


def prepare_pubchem(source_root, output_root, aid):
    source_directory = source_root / f"pubchem_aid_{aid}"
    output_directory = output_root / f"pubchem_aid_{aid}"
    output_directory.mkdir(parents=True, exist_ok=True)
    source = source_directory / "raw" / f"{aid}.csv"
    best_by_compound = {}
    source_rows = missing_smiles = missing_score = duplicate_rows = 0
    with source.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            if raw["PUBCHEM_RESULT_TAG"].startswith("RESULT_"):
                continue
            source_rows += 1
            smiles = raw["PUBCHEM_EXT_DATASOURCE_SMILES"].strip()
            score_text = raw.get("Max_Response", "").strip()
            if not smiles:
                missing_smiles += 1
                continue
            try:
                score = float(score_text)
            except ValueError:
                missing_score += 1
                continue
            cid = raw["PUBCHEM_CID"].strip()
            key = cid or f"SMILES:{smiles}"
            row = {
                "smiles": smiles,
                "sort_value": score_text,
                "compound_id": cid,
                "substance_id": raw["PUBCHEM_SID"].strip(),
                "activity_score": raw["PUBCHEM_ACTIVITY_SCORE"].strip(),
                "activity_outcome": raw["PUBCHEM_ACTIVITY_OUTCOME"].strip(),
                "potency_um": raw.get("Potency", "").strip(),
                "fit_log_ac50": raw.get("Fit_LogAC50", "").strip(),
                "max_response": raw.get("Max_Response", "").strip(),
            }
            previous = best_by_compound.get(key)
            if previous is not None:
                duplicate_rows += 1
            if previous is None or score > float(previous["sort_value"]):
                best_by_compound[key] = row

    rows = list(best_by_compound.values())
    metadata = {
        "dataset": f"PubChem BioAssay AID {aid}",
        "source": f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}",
        "raw_file": str(source.relative_to(source_root)),
        "sort_column": "sort_value",
        "sort_semantics": "Max_Response; higher observed assay response values are prioritized",
        "sort_direction": "descending",
        "deduplication": "one row per CID (highest Max_Response retained); SMILES fallback when CID is absent",
        "source_rows": source_rows,
        "missing_smiles_rows": missing_smiles,
        "missing_score_rows": missing_score,
        "duplicate_compound_rows": duplicate_rows,
        **_score_stats(rows),
    }
    _write_outputs(output_directory, rows, metadata)


def prepare_novartis(source_root, output_root):
    from rdkit import Chem

    source_directory = source_root / "novartis_malaria"
    output_directory = output_root / "novartis_malaria"
    output_directory.mkdir(parents=True, exist_ok=True)
    source = source_directory / "raw" / "Novartis_GNF_NoModifier.sdf"
    rows = []
    missing_score = parse_failures = 0
    for index, molecule in enumerate(Chem.SDMolSupplier(str(source), removeHs=False, sanitize=True)):
        if molecule is None:
            parse_failures += 1
            continue
        if not molecule.HasProp(NOVARTIS_SCORE):
            missing_score += 1
            continue
        score = molecule.GetProp(NOVARTIS_SCORE).strip()
        try:
            float(score.lstrip("<>").strip())
        except ValueError:
            missing_score += 1
            continue
        rows.append(
            {
                "smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
                "sort_value": score,
                "compound_id": molecule.GetProp("ID") if molecule.HasProp("ID") else str(index),
                "substance_id": "",
                "activity_score": "",
                "activity_outcome": "",
                "potency_um": score,
                "fit_log_ac50": "",
                "max_response": "",
            }
        )
    metadata = {
        "dataset": "Novartis-GNF Malaria Box",
        "source": "Novartis_GNF_NoModifier.sdf from the published DISE workflow supplement",
        "raw_file": str(source.relative_to(source_root)),
        "sort_column": "sort_value",
        "sort_semantics": f"{NOVARTIS_SCORE}; lower EC50 values are more active",
        "sort_direction": "ascending",
        "missing_score_rows": missing_score,
        "parse_failures": parse_failures,
        **_score_stats(rows),
    }
    _write_outputs(output_directory, rows, metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="/data/assay_data")
    parser.add_argument("--output-root", help="Writable output root; defaults to the input root")
    parser.add_argument("--skip-novartis", action="store_true")
    args = parser.parse_args()
    source_root = Path(args.root)
    output_root = Path(args.output_root) if args.output_root else source_root
    if not args.skip_novartis:
        prepare_novartis(source_root, output_root)
    for aid in PUBCHEM_AIDS:
        prepare_pubchem(source_root, output_root, aid)


if __name__ == "__main__":
    main()
