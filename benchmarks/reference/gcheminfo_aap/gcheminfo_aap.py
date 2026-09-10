# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the recovered gCheminfoCommands AAP DEFAULT8/DISE Java code."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from rdkit import Chem


HERE = Path(__file__).resolve().parent
JAVA_SOURCE = HERE / "GCheminfoAAPDISE.java"
JAVA_CLASS = "GCheminfoAAPDISE"


def compile_runner(build_dir: Path) -> Path:
    """Compile the dependency-free compatibility runner and return its classpath."""
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["javac", "-encoding", "UTF-8", "-d", str(build_dir), str(JAVA_SOURCE)],
        check=True,
    )
    return build_dir


def _bond_type(bond: Chem.Bond) -> int:
    if bond.GetIsAromatic():
        return 4
    value = int(round(bond.GetBondTypeAsDouble()))
    if value not in (1, 2, 3):
        raise ValueError(f"unsupported bond type: {bond.GetBondType()}")
    return value


def write_graphs(molecules, output: Path, input_indices=None, priorities=None) -> None:
    """Write heavy-atom RDKit graphs in the runner's intentionally simple TSV format."""
    if input_indices is None:
        input_indices = range(len(molecules))
    if priorities is None:
        priorities = [None] * len(molecules)
    with output.open("w", encoding="utf-8") as stream:
        stream.write("# input_index\tatom_types\tbonds\tpriority\n")
        for input_index, original, priority in zip(
            input_indices, molecules, priorities, strict=True
        ):
            molecule = Chem.RemoveHs(original)
            atom_types = [
                atom.GetAtomicNum() + (108 if atom.GetIsAromatic() else 0)
                for atom in molecule.GetAtoms()
            ]
            bonds = [
                f"{bond.GetBeginAtomIdx()},{bond.GetEndAtomIdx()},{_bond_type(bond)}"
                for bond in molecule.GetBonds()
            ]
            row = f"{input_index}\t{','.join(map(str, atom_types))}\t{';'.join(bonds)}"
            if priority is not None:
                row += f"\t{priority}"
            stream.write(row + "\n")


def run_cluster(
    graph_path: Path,
    output_path: Path,
    *,
    classpath: Path,
    threshold: float = 0.3,
    max_path_length: int = 7,
    threads: int = 1,
    java_heap: str = "30g",
):
    """Run published DEFAULT8 seed selection plus nearest-centroid reassignment."""
    completed = subprocess.run(
        [
            "java",
            f"-Xmx{java_heap}",
            "-cp",
            str(classpath),
            JAVA_CLASS,
            "cluster",
            str(graph_path),
            str(threshold),
            str(max_path_length),
            str(threads),
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed


def run_benchmark(
    graph_path: Path,
    output_path: Path,
    *,
    classpath: Path,
    threshold: float = 0.3,
    max_path_length: int = 7,
    threads: int = 1,
    warmups: int = 1,
    runs: int = 3,
    java_heap: str = "30g",
    order_by_priority: bool = False,
):
    """Benchmark compute passes in one JVM, excluding graph parsing and output."""
    return subprocess.run(
        [
            "java",
            f"-Xmx{java_heap}",
            "-cp",
            str(classpath),
            JAVA_CLASS,
            "benchmark-workflow" if order_by_priority else "benchmark",
            str(graph_path),
            str(threshold),
            str(max_path_length),
            str(threads),
            str(warmups),
            str(runs),
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def read_assignments(output_path: Path):
    with output_path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))
