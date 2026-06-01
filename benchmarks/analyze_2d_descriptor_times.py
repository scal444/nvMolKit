#!/usr/bin/env python
"""Analyze per-descriptor timings produced by time_2d_descriptors.py.

Rolls the per-descriptor CSV up into the RDKit subsections that the descriptors
come from (e.g. rdkit.Chem.Fragments, rdkit.Chem.EState.EState_VSA,
rdkit.Chem.MolSurf) and reports where the runtime is spent.
"""

import argparse
import csv
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default="descriptor_times.csv",
        help="Per-descriptor timing CSV produced by time_2d_descriptors.py.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of individual descriptors to show in the top-N table.",
    )
    parser.add_argument(
        "--section-output",
        default=None,
        help="Optional path to write the per-section rollup as CSV.",
    )
    return parser.parse_args()


def section_label(module):
    """Human-friendly subsection name derived from a descriptor's module."""
    if not module:
        return "(unknown)"
    prefix = "rdkit.Chem."
    return module[len(prefix):] if module.startswith(prefix) else module


def load_rows(path):
    rows = []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                {
                    "descriptor": raw["descriptor"],
                    "module": raw.get("module", ""),
                    "section": section_label(raw.get("module", "")),
                    "total_seconds": float(raw["total_seconds"]),
                    "mean_us_per_mol": float(raw["mean_us_per_mol"]),
                    "num_molecules": int(raw["num_molecules"]),
                    "errors": int(raw.get("errors", 0) or 0),
                }
            )
    return rows


def summarize_sections(rows):
    by_section = defaultdict(lambda: {"total_seconds": 0.0, "count": 0, "errors": 0})
    for row in rows:
        bucket = by_section[row["section"]]
        bucket["total_seconds"] += row["total_seconds"]
        bucket["count"] += 1
        bucket["errors"] += row["errors"]
    return by_section


def print_table(headers, widths, alignments, data_rows):
    header_cells = []
    for header, width, align in zip(headers, widths, alignments):
        header_cells.append(header.rjust(width) if align == ">" else header.ljust(width))
    line = "  ".join(header_cells)
    print(line)
    print("-" * len(line))
    for cells in data_rows:
        rendered = []
        for cell, width, align in zip(cells, widths, alignments):
            rendered.append(cell.rjust(width) if align == ">" else cell.ljust(width))
        print("  ".join(rendered))


def main():
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        print("No rows found in input CSV.")
        return

    processed = max(row["num_molecules"] for row in rows)
    grand_total = sum(row["total_seconds"] for row in rows)

    print(f"Input: {args.input}")
    print(f"Molecules processed: {processed}")
    print(f"Total measured time (parse + all descriptors): {grand_total:.3f}s")
    print()

    by_section = summarize_sections(rows)
    section_rows = sorted(
        by_section.items(), key=lambda item: item[1]["total_seconds"], reverse=True
    )

    print("=== Runtime by subsection ===")
    table = []
    for section, stats in section_rows:
        share = (stats["total_seconds"] / grand_total * 100) if grand_total else 0.0
        per_mol_ms = (stats["total_seconds"] / processed * 1e3) if processed else 0.0
        table.append(
            [
                section,
                str(stats["count"]),
                f"{stats['total_seconds']:.3f}",
                f"{share:.1f}%",
                f"{per_mol_ms:.4f}",
                str(stats["errors"]),
            ]
        )
    print_table(
        ["section", "#desc", "total_s", "share", "ms/mol", "errs"],
        [
            max(len("section"), max(len(r[0]) for r in table)),
            5,
            9,
            7,
            9,
            5,
        ],
        ["<", ">", ">", ">", ">", ">"],
        table,
    )
    print()

    print(f"=== Top {args.top} individual descriptors ===")
    top_rows = sorted(rows, key=lambda row: row["total_seconds"], reverse=True)[: args.top]
    table = []
    for row in top_rows:
        share = (row["total_seconds"] / grand_total * 100) if grand_total else 0.0
        table.append(
            [
                row["descriptor"],
                row["section"],
                f"{row['total_seconds']:.3f}",
                f"{share:.1f}%",
                f"{row['mean_us_per_mol']:.2f}",
            ]
        )
    print_table(
        ["descriptor", "section", "total_s", "share", "us/mol"],
        [
            max(len("descriptor"), max(len(r[0]) for r in table)),
            max(len("section"), max(len(r[1]) for r in table)),
            9,
            7,
            10,
        ],
        ["<", "<", ">", ">", ">"],
        table,
    )

    if args.section_output:
        with open(args.section_output, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["section", "num_descriptors", "total_seconds", "share_percent", "ms_per_mol", "errors"]
            )
            for section, stats in section_rows:
                share = (stats["total_seconds"] / grand_total * 100) if grand_total else 0.0
                per_mol_ms = (stats["total_seconds"] / processed * 1e3) if processed else 0.0
                writer.writerow(
                    [
                        section,
                        stats["count"],
                        f"{stats['total_seconds']:.6f}",
                        f"{share:.4f}",
                        f"{per_mol_ms:.6f}",
                        stats["errors"],
                    ]
                )
        print()
        print(f"Wrote section rollup to {args.section_output}")


if __name__ == "__main__":
    main()
