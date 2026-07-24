# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared benchmark molecule preparation."""

from bench_utils import molprep
from rdkit import Chem

def _serial_process_map(func, items, **kwargs):
    return [func(item) for item in items]


def test_prep_and_embed_preserve_explicit_hydrogens(monkeypatch):
    """FF preparation adds Hs once and keeps them through base embedding."""
    monkeypatch.setattr(molprep, "process_map", _serial_process_map)
    raw = Chem.MolFromSmiles("CCO")

    prepped = molprep.prep_mols([raw])
    assert len(prepped) == 1
    assert prepped[0].GetNumAtoms() > raw.GetNumAtoms()
    assert any(atom.GetAtomicNum() == 1 for atom in prepped[0].GetAtoms())

    embedded = molprep.embed_and_jitter(prepped, confs_per_mol=1, seed=42, num_workers=1)
    assert len(embedded) == 1
    assert embedded[0].GetNumAtoms() == prepped[0].GetNumAtoms()
    assert any(atom.GetAtomicNum() == 1 for atom in embedded[0].GetAtoms())
    assert embedded[0].GetConformer().GetNumAtoms() == embedded[0].GetNumAtoms()


def test_filter_force_field_mols_drops_incomplete_parameters():
    mmff_bad = Chem.AddHs(Chem.MolFromSmiles("CC1(C)OB(CC2=CC=CC=C2)OC1(C)C"))
    unsupported = Chem.AddHs(Chem.MolFromSmiles("[He]"))
    good = Chem.AddHs(Chem.MolFromSmiles("CCO"))

    assert molprep.filter_force_field_mols([mmff_bad, good], "mmff") == [good]
    assert molprep.filter_force_field_mols([unsupported, good], "uff") == [good]


def test_embed_once_then_jitter_uses_auto_cpu_workers(monkeypatch):
    """One base embed per mol produces the requested ensemble via jitter."""
    process_map_calls = []

    def recording_process_map(func, items, **kwargs):
        items = list(items)
        process_map_calls.append((len(items), kwargs))
        return [func(item) for item in items]

    monkeypatch.setattr(molprep, "process_map", recording_process_map)
    monkeypatch.setattr(molprep, "_available_cpu_count", lambda: 6)
    prepped = molprep.prep_mols([Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCCC")])

    embedded = molprep.embed_and_jitter(prepped, confs_per_mol=10, seed=42, num_workers=0)

    assert len(process_map_calls) == 1
    num_embed_jobs, kwargs = process_map_calls[0]
    assert num_embed_jobs == len(prepped)
    assert kwargs["max_workers"] == len(prepped)
    assert kwargs["chunksize"] == 1
    assert all(mol.GetNumConformers() == 10 for mol in embedded)
