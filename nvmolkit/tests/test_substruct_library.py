# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import Future

import pytest
from rdkit import Chem

from nvmolkit.substruct_library import SubstructLibrary
from nvmolkit.substructure import SubstructSearchConfig

PLAIN_QUERY_TARGETS = [
    "Cn1ccc2ccccc21",
    "c1ccc2[nH]ccc2c1",
    "CN(C)C",
    "CNC",
    "CC[O-]",
    "CCO",
    "[13CH3]O",
    "CO",
    "C[CH2]",
    "CC",
    "*C",
    "[1*]C",
    "[2*]C",
]
# Plain (SMILES) query atoms follow RDKit's Atom::Match: hydrogen counts are ignored, set charges, isotopes, and
# radicals must match, and dummy isotopes only conflict when both atoms carry one.
PLAIN_QUERIES = ["c1ccc2[nH]ccc2c1", "[NH2]C", "C[O-]", "[13CH3]O", "C[CH2]", "*C", "[1*]C", "[300CH4]"]


def test_finalize_generation_and_result_operations():
    library = SubstructLibrary(chunkSize=2)
    assert library.addMols(Chem.MolFromSmiles(smiles) for smiles in ["CCO", "c1ccccc1", "CC(=O)O"]) == [0, 1, 2]
    assert len(library) == 0
    assert library.pendingSize == 3

    query = Chem.MolFromSmarts("[#6]")
    with pytest.raises(RuntimeError, match="finalized"):
        library.hasMatch(query).result()

    library.finalize()
    assert len(library) == 3
    assert library.pendingSize == 0
    assert library.queryConcurrency >= 1
    assert library.getMatches(query).result() == [0, 1, 2]
    assert library.getMatches(query, maxResults=2).result() == [0, 1]
    assert library.countMatches(query).result() == 3
    assert library.hasMatch(Chem.MolFromSmarts("c")).result()

    futures = [library.hasMatch(Chem.MolFromSmarts(pattern)) for pattern in ["C", "N", "O"]]
    assert all(isinstance(future, Future) for future in futures)
    assert [future.result() for future in futures] == [True, False, True]
    assert library.batchesInFlightPerGpu >= library.queryConcurrency
    assert library.workspaceBytesPerQueryPerGpu > 0

    assert library.addMol(Chem.MolFromSmiles("N")) == 3
    assert library.getMatches(query).result() == [0, 1, 2]
    library.finalize()
    assert len(library) == 4


class _FailingFinalizeNative:
    """Delegate to a native library but fail finalize, as an OOM or admission failure would."""

    def __init__(self, native):
        self._native = native

    def __getattr__(self, name):
        return getattr(self._native, name)

    def finalize(self):
        raise RuntimeError("simulated finalize failure")


def test_failed_finalize_keeps_library_queryable():
    library = SubstructLibrary()
    library.addMols([Chem.MolFromSmiles("c1ccccc1")])
    library.finalize()
    library.addMol(Chem.MolFromSmiles("Oc1ccccc1"))

    native = library._native
    library._native = _FailingFinalizeNative(native)
    with pytest.raises(RuntimeError, match="simulated"):
        library.finalize()
    library._native = native

    query = Chem.MolFromSmarts("c")
    assert library.getMatches(query).result() == [0]
    library.finalize()
    assert library.getMatches(query).result() == [0, 1]


@pytest.mark.parametrize("algorithm", ["gsi", "dfs"])
def test_configured_production_backends(algorithm):
    config = SubstructSearchConfig(algorithm=algorithm, workerThreads=1, preprocessingThreads=1)
    library = SubstructLibrary(config=config)
    library.addMol(Chem.MolFromSmiles("CCOC(=O)C"))
    library.finalize()
    assert library.getMatches(Chem.MolFromSmarts("C=O")).result() == [0]


def test_invalid_library_configuration():
    with pytest.raises(ValueError, match="greater than zero"):
        SubstructLibrary(chunkSize=0)

    config = SubstructSearchConfig(gpuIds=[0, 0])
    with pytest.raises(ValueError, match="unique"):
        SubstructLibrary(config=config)


def test_multi_gpu_library_merges_shards_in_insertion_order():
    import torch

    if torch.cuda.device_count() < 2:
        pytest.skip("multi-GPU library test requires at least two CUDA devices")

    config = SubstructSearchConfig(gpuIds=[0, 1], workerThreads=1, preprocessingThreads=4)
    library = SubstructLibrary(chunkSize=2, config=config)
    targets = [Chem.MolFromSmiles(smiles) for smiles in ["CC", "O", "CCC", "N", "c1ccccc1", "CO", "C=O"]]
    assert library.addMols(targets) == list(range(len(targets)))
    library.finalize()

    query = Chem.MolFromSmarts("[#6]")
    assert library.getMatches(query).result() == [0, 2, 4, 5, 6]
    assert library.getMatches(query, maxResults=3).result() == [0, 2, 4]
    assert library.countMatches(query).result() == 5
    assert library.hasMatch(query).result()


@pytest.mark.parametrize("algorithm", ["gsi", "dfs"])
def test_plain_molecule_query_preserves_atom_constraints(algorithm):
    config = SubstructSearchConfig(algorithm=algorithm, workerThreads=1, preprocessingThreads=2)
    library = SubstructLibrary(chunkSize=2, config=config)
    targets = [
        Chem.MolFromSmiles("CCC(C)CNc1cccc2c1COCC2"),
        Chem.MolFromSmiles("CC[C@H](CO)NCc1cccc2c1OCCO2"),
        Chem.MolFromSmiles("C[C@@H]1CCN(c2ncnc3c2OCCO3)C1"),
    ]
    library.addMols(targets)
    library.finalize()

    query = Chem.MolFromSmiles("CCC(C)CNc1cccc2c1COCC2")
    assert library.getMatches(query).result() == [0]
    assert library.countMatches(query).result() == 1
    assert library.hasMatch(query).result()


@pytest.mark.parametrize("use_pattern_fingerprints", [False, True])
@pytest.mark.parametrize("query_smiles", PLAIN_QUERIES)
def test_plain_molecule_query_atoms_match_like_rdkit(query_smiles, use_pattern_fingerprints):
    library = SubstructLibrary(chunkSize=4, usePatternFingerprints=use_pattern_fingerprints)
    targets = [Chem.MolFromSmiles(smiles) for smiles in PLAIN_QUERY_TARGETS]
    library.addMols(targets)
    library.finalize()

    query = Chem.MolFromSmiles(query_smiles)
    expected = [index for index, target in enumerate(targets) if target.HasSubstructMatch(query)]
    assert library.getMatches(query).result() == expected


def test_finalize_routes_unpackable_targets_to_rdkit():
    library = SubstructLibrary(chunkSize=2)
    targets = [Chem.MolFromSmiles(smiles) for smiles in ["CCN", "[NH3]->[Cu]", "[300C]CN", "c1ccccc1N"]]
    library.addMols(targets)
    library.finalize()

    for smarts in ["[#7]", "[Cu]", "[#6]-[#7]"]:
        query = Chem.MolFromSmarts(smarts)
        expected = [index for index, target in enumerate(targets) if target.HasSubstructMatch(query)]
        assert library.getMatches(query).result() == expected


@pytest.mark.parametrize("use_pattern_fingerprints", [False, True])
def test_pattern_fingerprint_screening_preserves_exact_results(use_pattern_fingerprints):
    config = SubstructSearchConfig(workerThreads=1, preprocessingThreads=2)
    library = SubstructLibrary(
        chunkSize=3,
        config=config,
        usePatternFingerprints=use_pattern_fingerprints,
    )
    targets = [
        Chem.MolFromSmiles(smiles)
        for smiles in [
            "CCO",
            "CC(=O)C",
            "c1ccccc1",
            "C1CCCCC1",
            "C[N+](C)(C)C",
            "CC(=O)[O-]",
            "[Na+].[Cl-]",
            "CCOC(=O)c1ccccc1O",
        ]
    ]
    library.addMols(targets)
    library.finalize()

    for smarts in ["[#6]", "C=O", "c1ccccc1", "[N+]", "[$([CX3]=[OX1])]", "[Si]"]:
        query = Chem.MolFromSmarts(smarts)
        expected = [index for index, target in enumerate(targets) if target.HasSubstructMatch(query)]
        assert library.getMatches(query).result() == expected
        assert library.countMatches(query).result() == len(expected)
        assert library.hasMatch(query).result() == bool(expected)
