# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from rdkit import Chem

from nvmolkit.substruct_library import SubstructLibrary
from nvmolkit.substructure import SubstructSearchConfig


def test_finalize_generation_and_result_operations():
    library = SubstructLibrary(chunkSize=2)
    assert library.addMols(Chem.MolFromSmiles(smiles) for smiles in ["CCO", "c1ccccc1", "CC(=O)O"]) == [0, 1, 2]
    assert len(library) == 0
    assert library.pendingSize == 3

    query = Chem.MolFromSmarts("[#6]")
    with pytest.raises(RuntimeError, match="finalized"):
        library.hasMatch(query)

    library.finalize()
    assert len(library) == 3
    assert library.pendingSize == 0
    assert library.getMatches(query) == [0, 1, 2]
    assert library.getMatches(query, maxResults=2) == [0, 1]
    assert library.countMatches(query) == 3
    assert library.hasMatch(Chem.MolFromSmarts("c"))

    assert library.addMol(Chem.MolFromSmiles("N")) == 3
    assert library.getMatches(query) == [0, 1, 2]
    library.finalize()
    assert len(library) == 4


@pytest.mark.parametrize("algorithm", ["gsi", "dfs"])
def test_configured_production_backends(algorithm):
    config = SubstructSearchConfig(algorithm=algorithm, workerThreads=1, preprocessingThreads=1)
    library = SubstructLibrary(config=config)
    library.addMol(Chem.MolFromSmiles("CCOC(=O)C"))
    library.finalize()
    assert library.getMatches(Chem.MolFromSmarts("C=O")) == [0]


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
    assert library.getMatches(query) == [0, 2, 4, 5, 6]
    assert library.getMatches(query, maxResults=3) == [0, 2, 4]
    assert library.countMatches(query) == 5
    assert library.hasMatch(query)


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
    assert library.getMatches(query) == [0]
    assert library.countMatches(query) == 1
    assert library.hasMatch(query)
