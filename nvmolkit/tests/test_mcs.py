# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from rdkit import Chem
from rdkit.Chem import rdFMCS

from nvmolkit.mcs import MCSConfig, MCS_STATS_ENABLED, MCS_TIMINGS_ENABLED, findMCS


def _mols(smiles: list[str]):
    return [Chem.MolFromSmiles(smi) for smi in smiles]


def _rdkit_params(
    *,
    atom_compare: str = "elements",
    bond_compare: str = "order",
    ring_matches_ring_only: bool = False,
    complete_rings_only: bool = False,
):
    params = rdFMCS.MCSParameters()
    params.MaximizeBonds = True
    params.AtomCompareParameters.RingMatchesRingOnly = ring_matches_ring_only
    params.BondCompareParameters.RingMatchesRingOnly = ring_matches_ring_only
    params.AtomCompareParameters.CompleteRingsOnly = complete_rings_only
    params.BondCompareParameters.CompleteRingsOnly = complete_rings_only

    atom_types = {
        "any": rdFMCS.AtomCompare.CompareAny,
        "elements": rdFMCS.AtomCompare.CompareElements,
        "isotopes": rdFMCS.AtomCompare.CompareIsotopes,
    }
    bond_types = {
        "any": rdFMCS.BondCompare.CompareAny,
        "order": rdFMCS.BondCompare.CompareOrder,
        "order_exact": rdFMCS.BondCompare.CompareOrderExact,
    }
    params.AtomTyper = atom_types[atom_compare]
    params.BondTyper = bond_types[bond_compare]
    return params


def _assert_matches_rdkit(result, mol_table, *, atom_compare="elements", bond_compare="order", **kwargs):
    params = _rdkit_params(atom_compare=atom_compare, bond_compare=bond_compare, **kwargs)
    assert len(result.num_atoms) == len(result.pairs)
    assert len(result.num_bonds) == len(result.pairs)
    for pair_idx, (idx_a, idx_b) in enumerate(result.pairs):
        rd_result = rdFMCS.FindMCS([mol_table[idx_a], mol_table[idx_b]], params)
        item = result[pair_idx]
        assert item.num_atoms == rd_result.numAtoms
        assert item.num_bonds == rd_result.numBonds
        assert item.atom_mapping.shape[1] == 2
        assert item.bond_mapping.shape[1] == 2
        query = Chem.MolFromSmarts(item.smarts_string)
        rd_query = Chem.MolFromSmarts(rd_result.smartsString)
        assert query is not None
        assert rd_query is not None
        assert query.GetNumAtoms() == rd_query.GetNumAtoms()
        assert query.GetNumBonds() == rd_query.GetNumBonds()
        if item.num_atoms:
            assert mol_table[idx_a].HasSubstructMatch(query)
            assert mol_table[idx_b].HasSubstructMatch(query)


def test_pairs_mode_matches_rdkit_and_preserves_order():
    mols = _mols(["CCO", "CCN", "c1ccccc1", "c1ccc(O)cc1"])
    pairs = [(0, 1), (2, 3), (0, 2)]

    result = findMCS(mols, mode="pairs", pairs=pairs)

    assert result.mode == "pairs"
    assert result.pairs == tuple(pairs)
    assert len(result) == len(pairs)
    assert result.elapsed_ms is None
    assert result.used_gpu.any()
    _assert_matches_rdkit(result, mols)


def test_collect_timings_follows_build_flag():
    mols = _mols(["CCO", "CCN", "c1ccccc1", "c1ccc(O)cc1"])
    pairs = [(0, 1), (2, 3), (0, 2)]

    if MCS_TIMINGS_ENABLED:
        result = findMCS(mols, mode="pairs", pairs=pairs, collect_timings=True)
        assert result.elapsed_ms is not None
        assert result.elapsed_ms.shape == (len(pairs),)
        assert (result.elapsed_ms >= 0.0).all()
        assert result.fmcs_timings is not None
        assert set(result.fmcs_timings) == {"total_clocks", "phase1_clocks", "phase2_clocks"}
        assert all(values.shape == (len(pairs),) for values in result.fmcs_timings.values())
        assert result[0].elapsed_ms is not None
        assert result[0].fmcs_timings is not None
        assert result.fmcs_stats is None
        return

    with pytest.raises(RuntimeError, match="timing instrumentation is not instantiated"):
        findMCS(mols, mode="pairs", pairs=pairs, collect_timings=True)


def test_collect_stats_follows_build_flag():
    mols = _mols(["CCO", "CCN", "c1ccccc1", "c1ccc(O)cc1"])
    pairs = [(0, 1), (2, 3), (0, 2)]

    if MCS_STATS_ENABLED:
        result = findMCS(mols, mode="pairs", pairs=pairs, collect_stats=True)
        assert result.elapsed_ms is None
        assert result.fmcs_stats is not None
        assert set(result.fmcs_stats) >= {"phase2_iters", "total_clocks"}
        assert all(values.shape == (len(pairs),) for values in result.fmcs_stats.values())
        assert result[0].fmcs_stats is not None
        return

    with pytest.raises(RuntimeError, match="stat instrumentation is not instantiated"):
        findMCS(mols, mode="pairs", pairs=pairs, collect_stats=True)


def test_pairs_mode_chunked_multi_executor_matches_rdkit():
    mols = _mols(["CCO", "CCN", "c1ccccc1", "c1ccc(O)cc1", "CC(C)O"])
    pairs = [(0, 1), (2, 3), (4, 0), (1, 4), (3, 2)]

    result = findMCS(
        mols,
        mode="pairs",
        pairs=pairs,
        batch_size=1,
        block_size=352,
        executors_per_runner=2,
    )

    assert result.pairs == tuple(pairs)
    assert result.used_gpu.any()
    _assert_matches_rdkit(result, mols)


def test_pairs_mode_threaded_gpu_options_match_rdkit():
    mols = _mols(["CCO", "CCN", "c1ccccc1", "c1ccc(O)cc1", "CC(C)O"])
    pairs = [(0, 1), (2, 3), (4, 0), (1, 4), (3, 2)]

    result = findMCS(
        mols,
        mode="pairs",
        pairs=pairs,
        batch_size=1,
        block_size=352,
        worker_threads=2,
        preprocessing_threads=2,
        executors_per_runner=1,
        gpu_ids=[],
    )

    assert result.pairs == tuple(pairs)
    assert result.used_gpu.any()
    _assert_matches_rdkit(result, mols)


def test_config_path_matches_rdkit_and_rejects_duplicate_execution_options():
    mols = _mols(["CCO", "CCN", "c1ccccc1", "c1ccc(O)cc1"])
    pairs = [(0, 1), (2, 3)]
    config = MCSConfig(
        batchSize=1,
        blockSize=352,
        workerThreads=1,
        preprocessingThreads=1,
        executorsPerRunner=1,
    )

    result = findMCS(mols, mode="pairs", pairs=pairs, config=config)

    assert result.pairs == tuple(pairs)
    assert result.used_gpu.any()
    _assert_matches_rdkit(result, mols)

    with pytest.raises(ValueError, match="config cannot be combined"):
        findMCS(mols, mode="pairs", pairs=pairs, config=config, block_size=640)


def test_all_pairs_default_is_upper_triangle_with_diagonal():
    mols = _mols(["CCO", "CCN", "c1ccccc1"])

    result = findMCS(mols)

    assert result.mode == "all_pairs"
    assert result.pairs == ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    _assert_matches_rdkit(result, mols)


def test_all_pairs_full_matrix_without_diagonal():
    mols = _mols(["CCO", "CCN", "c1ccccc1"])

    result = findMCS(mols, upper_triangle=False, include_diagonal=False)

    assert result.pairs == ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
    _assert_matches_rdkit(result, mols)


def test_paired_lists_mode_matches_rdkit():
    mols_a = _mols(["CCO", "c1ccccc1"])
    mols_b = _mols(["CCN", "c1ccc(O)cc1"])

    result = findMCS(mols_a, mode="paired_lists", mols_b=mols_b)

    assert result.mode == "paired_lists"
    assert result.pairs == ((0, 2), (1, 3))
    _assert_matches_rdkit(result, mols_a + mols_b)


def test_compare_and_ring_options_match_rdkit():
    mols = _mols(["C1CCCCC1", "CCCCCC", "c1ccccc1"])
    pairs = [(0, 1), (0, 2)]

    result = findMCS(
        mols,
        mode="pairs",
        pairs=pairs,
        atom_compare="any",
        bond_compare="any",
        ring_matches_ring_only=True,
    )

    _assert_matches_rdkit(result, mols, atom_compare="any", bond_compare="any", ring_matches_ring_only=True)


def test_complete_rings_only_gpu_path_is_reported():
    mols = _mols(["c1ccccc1", "c1ccc(O)cc1"])

    result = findMCS(mols, mode="pairs", pairs=[(0, 1)], complete_rings_only=True)

    assert result.used_fallback.tolist() == [0]
    assert result.used_gpu.tolist() == [1]
    _assert_matches_rdkit(result, mols, complete_rings_only=True)


def test_complete_rings_only_runs_when_fallback_is_disallowed():
    mols = _mols(["c1ccccc1", "c1ccc(O)cc1"])

    result = findMCS(
        mols,
        mode="pairs",
        pairs=[(0, 1)],
        complete_rings_only=True,
        allow_rdkit_fallback=False,
    )

    assert result.used_fallback.tolist() == [0]
    assert result.used_gpu.tolist() == [1]
    _assert_matches_rdkit(result, mols, complete_rings_only=True)


def test_allow_rdkit_fallback_controls_unsupported_options():
    mols = _mols(["CCO", "CCN"])

    result = findMCS(mols, mode="pairs", pairs=[(0, 1)], connected_only=False)
    assert result.used_fallback.tolist() == [1]
    assert result.used_gpu.tolist() == [0]

    with pytest.raises(RuntimeError, match="RDKit fallback is disabled: fMCS supports connected MCS only"):
        findMCS(
            mols,
            mode="pairs",
            pairs=[(0, 1)],
            connected_only=False,
            allow_rdkit_fallback=False,
        )


def test_invalid_mode_and_optional_arguments():
    mols = _mols(["CCO", "CCN"])

    with pytest.raises(ValueError, match="mode must be"):
        findMCS(mols, mode="wat")
    with pytest.raises(ValueError, match="pairs is required"):
        findMCS(mols, mode="pairs")
    with pytest.raises(ValueError, match="mols_b is only valid"):
        findMCS(mols, mode="pairs", pairs=[(0, 1)], mols_b=mols)
    with pytest.raises(ValueError, match="pairs is only valid"):
        findMCS(mols, pairs=[(0, 1)])
    with pytest.raises(ValueError, match="mols_b is required"):
        findMCS(mols, mode="paired_lists")
    with pytest.raises(ValueError, match="pairs is only valid"):
        findMCS(mols, mode="paired_lists", pairs=[(0, 1)], mols_b=mols)
    with pytest.raises(ValueError, match="same length"):
        findMCS(mols, mode="paired_lists", mols_b=mols[:1])
    with pytest.raises(ValueError, match="exactly two"):
        findMCS(mols, mode="pairs", pairs=[(0, 1, 2)])
    with pytest.raises(ValueError, match="Unsupported atom_compare"):
        findMCS(mols, mode="pairs", pairs=[(0, 1)], atom_compare="mass")
    with pytest.raises(ValueError, match="Unsupported bond_compare"):
        findMCS(mols, mode="pairs", pairs=[(0, 1)], bond_compare="shape")
    with pytest.raises(ValueError, match="Unsupported scratch_location"):
        findMCS(mols, mode="pairs", pairs=[(0, 1)], scratch_location="l2")
    for block_size in (32, 64, 384):
        with pytest.raises(ValueError, match="blockSize"):
            findMCS(mols, mode="pairs", pairs=[(0, 1)], block_size=block_size)


def test_scratch_location_invalid_value_rejected_in_config():
    with pytest.raises(ValueError, match="Unsupported scratch_location"):
        MCSConfig(scratchLocation="l2")


def test_mcsconfig_roundtrip_with_and_without_scratch_location():
    config = MCSConfig(blockSize=640, scratchLocation="global")
    data = config.to_dict()
    assert data["scratchLocation"] == "global"
    restored = MCSConfig.from_dict(data)
    assert restored.scratchLocation == "global"
    assert restored.blockSize == 640

    # Old-format dict (before scratchLocation existed) must still load,
    # defaulting the missing key to "auto".
    legacy = {
        "batchSize": 0,
        "blockSize": 128,
        "workerThreads": -1,
        "preprocessingThreads": -1,
        "executorsPerRunner": -1,
        "gpuIds": [],
    }
    restored_legacy = MCSConfig.from_dict(legacy)
    assert restored_legacy.scratchLocation == "auto"


def test_block_size_640_tier128_succeeds_on_gpu():
    # A 128-carbon chain is a tier-128 molecule. blockSize 640 handles it
    # via global substructure scratch (auto), so fallback can be disallowed.
    chain = Chem.MolFromSmiles("C" * 128)
    assert chain is not None
    mols = [chain, chain]
    result = findMCS(
        mols,
        mode="pairs",
        pairs=[(0, 1)],
        allow_rdkit_fallback=False,
        block_size=640,
        scratch_location="auto",
    )
    assert result.used_gpu.tolist() == [1]
    assert not bool(result.overflowed[0])
    assert int(result.num_atoms[0]) == 128


def test_out_of_range_pair_raises_from_native_layer():
    mols = _mols(["CCO", "CCN"])

    with pytest.raises(RuntimeError, match="pair index out of range"):
        findMCS(mols, mode="pairs", pairs=[(0, 3)])
