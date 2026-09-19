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

    config = SubstructSearchConfig(gpuIds=[0, 1])
    with pytest.raises(ValueError, match="one GPU"):
        SubstructLibrary(config=config)
