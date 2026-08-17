# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for benchmark molecule-loader candidate reserves."""

import io
import os

from bench_utils import loaders, molprep
from rdkit import Chem

def test_load_smiles_can_return_candidate_buffer(tmp_path, monkeypatch):
    smiles_path = tmp_path / "mols.smi"
    smiles_path.write_text("\n".join(["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"]))
    monkeypatch.setattr(loaders, "process_map", lambda fn, values, **_kwargs: [fn(value) for value in values])

    trimmed = loaders.load_smiles(str(smiles_path), max_count=3, seed=42)
    buffered = loaders.load_smiles(str(smiles_path), max_count=3, seed=42, keep_buffer=True)

    assert len(trimmed) == 3
    assert len(buffered) == 4


def test_load_smiles_uniformly_samples_the_whole_file(tmp_path, monkeypatch):
    smiles_path = tmp_path / "ordered_mols.smi"
    smiles_path.write_text("\n".join("C" * length for length in range(1, 101)))
    monkeypatch.setattr(loaders, "process_map", lambda fn, values, **_kwargs: [fn(value) for value in values])

    sample = loaders.load_smiles(str(smiles_path), max_count=10, seed=42)
    repeated = loaders.load_smiles(str(smiles_path), max_count=10, seed=42)
    different_seed = loaders.load_smiles(str(smiles_path), max_count=10, seed=43)

    atom_counts = [mol.GetNumAtoms() for mol in sample]
    assert atom_counts == [mol.GetNumAtoms() for mol in repeated]
    assert atom_counts != [mol.GetNumAtoms() for mol in different_seed]
    assert set(atom_counts) != set(range(1, 11))
    assert max(atom_counts) > 50


def test_buffered_count_reserves_at_least_one_candidate():
    assert loaders._buffered_count(0) == 0
    assert loaders._buffered_count(1) == 2
    assert loaders._buffered_count(1000) == 1100


def test_available_cpu_count_supports_python_312(monkeypatch):
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {2, 4, 6}, raising=False)

    assert molprep.available_cpu_count() == 3


def test_physical_cpu_count_deduplicates_smt_siblings_within_affinity(monkeypatch):
    cpuinfo = """
processor : 0
physical id : 0
core id : 0

processor : 1
physical id : 0
core id : 0

processor : 2
physical id : 0
core id : 1

processor : 3
physical id : 0
core id : 1
"""
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    monkeypatch.setattr("builtins.open", lambda _path: io.StringIO(cpuinfo))

    assert molprep.available_physical_cpu_count() == 2


def test_slice_conformers_copies_and_limits_each_molecule():
    mol = Chem.MolFromSmiles("CC")
    for _ in range(3):
        mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()), assignId=True)

    sliced = molprep.slice_conformers([mol], 2)

    assert sliced[0] is not mol
    assert sliced[0].GetNumConformers() == 2
    assert mol.GetNumConformers() == 3
