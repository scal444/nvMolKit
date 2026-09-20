# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from rdkit import Chem
from rdkit.Chem import FilterCatalog as RDFilterCatalog

from nvmolkit.filter_catalog import FilterCatalog, FilterCatalogPreset
from nvmolkit.substructure import SubstructSearchConfig


def _molecules(smiles):
    return [Chem.MolFromSmiles(value) for value in smiles]


def _reference_catalog(entries):
    catalog = RDFilterCatalog.FilterCatalog()
    for smarts, description, trigger_count in entries:
        matcher = RDFilterCatalog.SmartsMatcher(description, Chem.MolFromSmarts(smarts), trigger_count)
        catalog.AddEntry(RDFilterCatalog.FilterCatalogEntry(description, matcher))
    return catalog


@pytest.mark.parametrize("algorithm", ["gsi", "dfs"])
def test_custom_catalog_matches_rdkit_and_preserves_order(algorithm):
    entries = [
        ("C=O", "carbonyl", 1),
        ("[#6]", "three carbons", 3),
        ("[N+](=O)[O-]", "nitro", 1),
    ]
    catalog = FilterCatalog(config=SubstructSearchConfig(algorithm=algorithm, workerThreads=1, preprocessingThreads=1))
    for index, (smarts, description, trigger_count) in enumerate(entries):
        assert (
            catalog.addSmarts(
                smarts,
                description,
                trigger_count,
                {"source": "test", "ordinal": str(index)},
            )
            == index
        )

    assert len(catalog) == 0
    assert catalog.pendingSize == 3
    catalog.finalize()

    targets = _molecules(["CC=O", "CCC", "C[N+](=O)[O-]", "O", "CC(=O)CCC"])
    reference = _reference_catalog(entries)
    expected = [
        [index for index in range(reference.GetNumEntries()) if reference.GetEntry(index).HasFilterMatch(mol)]
        for mol in targets
    ]
    assert catalog.getMatches(targets) == expected
    assert catalog.hasMatch(iter(targets)) == [bool(ids) for ids in expected]
    assert catalog.getFirstMatch(targets) == [ids[0] if ids else None for ids in expected]


def test_entry_metadata_and_query_molecule_input():
    catalog = FilterCatalog()
    entry_id = catalog.addQuery(
        Chem.MolFromSmarts("[O;H1]"),
        description="hydroxy",
        metadata={"reference": "unit-test", "severity": "low"},
    )
    catalog.finalize()

    entry = catalog.getEntry(entry_id)
    assert entry.id == entry_id
    assert entry.description == "hydroxy"
    assert entry.triggerCount == 1
    assert entry.metadata == {"reference": "unit-test", "severity": "low"}
    assert catalog.hasMatch(_molecules(["CO", "COC"])) == [True, False]


def test_empty_batch_and_finalize_generation():
    catalog = FilterCatalog()
    catalog.addSmarts("N", "nitrogen")
    catalog.finalize()
    assert catalog.hasMatch([]) == []
    assert catalog.getFirstMatch([]) == []
    assert catalog.getMatches([]) == []

    catalog.addSmarts("O", "oxygen")
    assert catalog.hasMatch(_molecules(["CO"])) == [False]
    catalog.finalize()
    assert catalog.getMatches(_molecules(["CO"])) == [[1]]


def test_query_requires_first_finalize():
    catalog = FilterCatalog()
    catalog.addSmarts("N", "nitrogen")
    with pytest.raises(RuntimeError, match="finalized"):
        catalog.hasMatch(_molecules(["CN"]))


def test_empty_finalized_catalog():
    catalog = FilterCatalog()
    catalog.finalize()
    targets = _molecules(["CC", "N"])
    assert catalog.hasMatch(targets) == [False, False]
    assert catalog.getFirstMatch(targets) == [None, None]
    assert catalog.getMatches(targets) == [[], []]


def test_lowest_id_wins_when_multiple_queries_match():
    catalog = FilterCatalog()
    assert catalog.addSmarts("O", "oxygen") == 0
    assert catalog.addSmarts("[#6]", "two carbons", triggerCount=2) == 1
    assert catalog.addSmarts("CO", "carbon oxygen") == 2
    catalog.finalize()

    targets = _molecules(["CCO", "CO", "O"])
    assert catalog.getMatches(targets) == [[0, 1, 2], [0, 2], [0]]
    assert catalog.getFirstMatch(targets) == [0, 0, 0]


@pytest.mark.parametrize("trigger_count", [0, -1])
def test_invalid_trigger_count(trigger_count):
    catalog = FilterCatalog()
    with pytest.raises((OverflowError, ValueError), match=r"trigger|positive|greater"):
        catalog.addSmarts("C", triggerCount=trigger_count)


def test_invalid_smarts_and_entry_id():
    catalog = FilterCatalog()
    with pytest.raises(ValueError, match=r"SMARTS|parse|valid"):
        catalog.addSmarts("[")
    with pytest.raises((IndexError, ValueError), match=r"entry|range|ID|id"):
        catalog.getEntry(0)
    catalog.addSmarts("C")
    catalog.finalize()
    with pytest.raises(ValueError, match="null"):
        catalog.hasMatch([None])


def test_brenk_preset_matches_rdkit_reference():
    catalog = FilterCatalog(FilterCatalogPreset.BRENK)
    assert catalog.pendingSize > 0
    catalog.finalize()

    params = RDFilterCatalog.FilterCatalogParams()
    params.AddCatalog(RDFilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    reference = RDFilterCatalog.FilterCatalog(params)
    targets = _molecules(["CCO", "C1CO1", "CCN=NCC", "c1ccccc1", "C[Si](C)(C)C"])

    assert len(catalog) == reference.GetNumEntries()
    assert catalog.hasMatch(targets) == [reference.HasMatch(molecule) for molecule in targets]


@pytest.mark.parametrize(
    ("preset", "rdkit_preset"),
    [
        (FilterCatalogPreset.PAINS, RDFilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS),
        (FilterCatalogPreset.CHEMBL, RDFilterCatalog.FilterCatalogParams.FilterCatalogs.CHEMBL),
        (FilterCatalogPreset.ALL, RDFilterCatalog.FilterCatalogParams.FilterCatalogs.ALL),
    ],
)
def test_composite_presets_expose_all_rdkit_entries(preset, rdkit_preset):
    catalog = FilterCatalog(preset)
    reference = RDFilterCatalog.FilterCatalog(rdkit_preset)
    assert catalog.pendingSize == reference.GetNumEntries()


def test_add_preset_to_custom_catalog():
    catalog = FilterCatalog()
    assert catalog.addPreset(FilterCatalogPreset.PAINS_A) > 0
    assert catalog.pendingSize > 0
