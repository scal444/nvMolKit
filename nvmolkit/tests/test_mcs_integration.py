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

"""Dataset-backed integration tests for the Python MCS API."""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdFMCS

from nvmolkit.mcs import findMCS

_DEFAULT_MOLECULES = 64
_DEFAULT_PAIRS = 64
_DEFAULT_SEED = 1337
_MAX_GRAPH_SIZE = 24
_CHEMBL_PATH = Path(__file__).parents[2] / "tests" / "test_data" / "chembl_1k.smi"


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _graph_size(mol: Chem.Mol) -> int:
    return max(mol.GetNumAtoms(), mol.GetNumBonds())


@pytest.fixture(scope="module")
def chembl_mols() -> list[Chem.Mol]:
    """Load a deterministic, size-stratified ChEMBL subset."""
    molecule_limit = _env_int("NVMOLKIT_MCS_PY_TEST_MOLECULES", _DEFAULT_MOLECULES)
    seed = _env_int("NVMOLKIT_MCS_PY_TEST_SEED", _DEFAULT_SEED)
    tiers: dict[int, list[Chem.Mol]] = {16: [], 24: []}

    for line in _CHEMBL_PATH.read_text(encoding="utf-8").splitlines():
        smiles = line.strip()
        if not smiles or smiles.startswith("#"):
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        size = _graph_size(mol)
        if size <= 16:
            tiers[16].append(mol)
        elif size <= _MAX_GRAPH_SIZE:
            tiers[24].append(mol)

    rng = random.Random(seed)
    for molecules in tiers.values():
        rng.shuffle(molecules)

    selected: list[Chem.Mol] = []
    tier_limits = [molecule_limit // 2] * 2
    for i in range(molecule_limit % 2):
        tier_limits[i] += 1
    used_per_tier: dict[int, int] = {}
    for tier, limit in zip(tiers, tier_limits, strict=True):
        used_per_tier[tier] = min(limit, len(tiers[tier]))
        selected.extend(tiers[tier][: used_per_tier[tier]])

    unused = [mol for tier, molecules in tiers.items() for mol in molecules[used_per_tier[tier] :]]
    rng.shuffle(unused)
    selected.extend(unused[: molecule_limit - len(selected)])

    if len(selected) != molecule_limit:
        raise RuntimeError(
            f"requested {molecule_limit} integration molecules, but only "
            f"{len(selected)} satisfy the {_MAX_GRAPH_SIZE}-atom/bond limit"
        )
    return selected


@pytest.fixture(scope="module")
def chembl_pairs(chembl_mols: list[Chem.Mol]) -> list[tuple[int, int]]:
    """Generate deterministic pairs across the 16 and 24 tiers."""
    pair_count = _env_int("NVMOLKIT_MCS_PY_TEST_PAIRS", _DEFAULT_PAIRS)
    seed = _env_int("NVMOLKIT_MCS_PY_TEST_SEED", _DEFAULT_SEED)
    rng = random.Random(seed)
    tier_indices = {
        16: [i for i, mol in enumerate(chembl_mols) if _graph_size(mol) <= 16],
        24: [i for i, mol in enumerate(chembl_mols) if 16 < _graph_size(mol) <= 24],
    }

    pairs: list[tuple[int, int]] = []
    tier_pair_counts = [pair_count // 2] * 2
    for i in range(pair_count % 2):
        tier_pair_counts[i] += 1
    for tier, count in zip(tier_indices, tier_pair_counts, strict=True):
        indices = tier_indices[tier]
        assert len(indices) >= 2
        for _ in range(count):
            pairs.append(tuple(rng.sample(indices, 2)))
    return pairs


def _rdkit_params(
    atom_compare: str,
    bond_compare: str,
    *,
    atom_ring_matches_ring_only: bool = False,
    bond_ring_matches_ring_only: bool = False,
) -> rdFMCS.MCSParameters:
    params = rdFMCS.MCSParameters()
    params.MaximizeBonds = True
    params.AtomTyper = {
        "any": rdFMCS.AtomCompare.CompareAny,
        "elements": rdFMCS.AtomCompare.CompareElements,
        "isotopes": rdFMCS.AtomCompare.CompareIsotopes,
    }[atom_compare]
    params.BondTyper = {
        "any": rdFMCS.BondCompare.CompareAny,
        "order": rdFMCS.BondCompare.CompareOrder,
        "order_exact": rdFMCS.BondCompare.CompareOrderExact,
    }[bond_compare]
    params.AtomCompareParameters.RingMatchesRingOnly = atom_ring_matches_ring_only
    params.BondCompareParameters.RingMatchesRingOnly = bond_ring_matches_ring_only
    return params


def _assert_mapping_is_common_subgraph(item, mol_a: Chem.Mol, mol_b: Chem.Mol) -> None:
    atom_mapping = [tuple(map(int, pair)) for pair in item.atom_mapping]
    bond_mapping = [tuple(map(int, pair)) for pair in item.bond_mapping]
    assert len(atom_mapping) == item.num_atoms
    assert len(bond_mapping) == item.num_bonds
    assert len({a for a, _ in atom_mapping}) == item.num_atoms
    assert len({b for _, b in atom_mapping}) == item.num_atoms
    assert len({a for a, _ in bond_mapping}) == item.num_bonds
    assert len({b for _, b in bond_mapping}) == item.num_bonds

    atom_a_to_b = dict(atom_mapping)
    for atom_a, atom_b in atom_mapping:
        assert 0 <= atom_a < mol_a.GetNumAtoms()
        assert 0 <= atom_b < mol_b.GetNumAtoms()

    for bond_a_idx, bond_b_idx in bond_mapping:
        bond_a = mol_a.GetBondWithIdx(bond_a_idx)
        bond_b = mol_b.GetBondWithIdx(bond_b_idx)
        mapped_ends = {
            atom_a_to_b[bond_a.GetBeginAtomIdx()],
            atom_a_to_b[bond_a.GetEndAtomIdx()],
        }
        assert mapped_ends == {bond_b.GetBeginAtomIdx(), bond_b.GetEndAtomIdx()}


def _assert_batch_matches_rdkit(
    result,
    mols: list[Chem.Mol],
    *,
    atom_compare: str = "elements",
    bond_compare: str = "order",
    atom_ring_matches_ring_only: bool = False,
    bond_ring_matches_ring_only: bool = False,
) -> None:
    params = _rdkit_params(
        atom_compare,
        bond_compare,
        atom_ring_matches_ring_only=atom_ring_matches_ring_only,
        bond_ring_matches_ring_only=bond_ring_matches_ring_only,
    )
    assert result.used_gpu.all()
    assert not result.used_fallback.any()
    assert not result.overflowed.any()
    assert not result.canceled.any()

    for pair_idx, (idx_a, idx_b) in enumerate(result.pairs):
        expected = rdFMCS.FindMCS([mols[idx_a], mols[idx_b]], params)
        item = result[pair_idx]
        assert (item.num_atoms, item.num_bonds) == (
            expected.numAtoms,
            expected.numBonds,
        ), f"pair {pair_idx}: molecule indices {(idx_a, idx_b)}"
        query = Chem.MolFromSmarts(item.smarts_string)
        expected_query = Chem.MolFromSmarts(expected.smartsString)
        assert query is not None, f"pair {pair_idx}: invalid SMARTS {item.smarts_string!r}"
        assert expected_query is not None
        assert (query.GetNumAtoms(), query.GetNumBonds()) == (
            expected_query.GetNumAtoms(),
            expected_query.GetNumBonds(),
        ), f"pair {pair_idx}: molecule indices {(idx_a, idx_b)}"
        if item.num_atoms:
            assert mols[idx_a].HasSubstructMatch(query), (
                f"pair {pair_idx}: SMARTS does not match first molecule: {item.smarts_string}"
            )
            assert mols[idx_b].HasSubstructMatch(query), (
                f"pair {pair_idx}: SMARTS does not match second molecule: {item.smarts_string}"
            )
        _assert_mapping_is_common_subgraph(item, mols[idx_a], mols[idx_b])


_COMPARE_CONFIGS = [
    pytest.param(atom, bond, id=f"atom-{atom}_bond-{bond}")
    for atom in ("any", "elements", "isotopes")
    for bond in ("any", "order", "order_exact")
]


@pytest.mark.parametrize(("atom_compare", "bond_compare"), _COMPARE_CONFIGS)
def test_chembl_pairs_match_rdkit_across_compare_modes(
    chembl_mols: list[Chem.Mol],
    chembl_pairs: list[tuple[int, int]],
    atom_compare: str,
    bond_compare: str,
) -> None:
    result = findMCS(
        chembl_mols,
        mode="pairs",
        pairs=chembl_pairs,
        atom_compare=atom_compare,
        bond_compare=bond_compare,
        require_gpu=True,
    )
    _assert_batch_matches_rdkit(
        result,
        chembl_mols,
        atom_compare=atom_compare,
        bond_compare=bond_compare,
    )


@pytest.mark.parametrize(
    ("atom_ring_matches_ring_only", "bond_ring_matches_ring_only"),
    [
        pytest.param(True, False, id="atom-ring"),
        pytest.param(False, True, id="bond-ring"),
        pytest.param(True, True, id="atom-and-bond-ring"),
    ],
)
def test_chembl_pairs_match_rdkit_across_ring_modes(
    chembl_mols: list[Chem.Mol],
    chembl_pairs: list[tuple[int, int]],
    atom_ring_matches_ring_only: bool,
    bond_ring_matches_ring_only: bool,
) -> None:
    result = findMCS(
        chembl_mols,
        mode="pairs",
        pairs=chembl_pairs,
        atom_ring_matches_ring_only=atom_ring_matches_ring_only,
        bond_ring_matches_ring_only=bond_ring_matches_ring_only,
        require_gpu=True,
    )
    _assert_batch_matches_rdkit(
        result,
        chembl_mols,
        atom_ring_matches_ring_only=atom_ring_matches_ring_only,
        bond_ring_matches_ring_only=bond_ring_matches_ring_only,
    )


@pytest.mark.parametrize(
    "execution_options",
    [
        pytest.param({"batch_size": 7}, id="chunked"),
        pytest.param(
            {"batch_size": 5, "executors_per_runner": 2},
            id="multi-executor",
        ),
        pytest.param(
            {
                "batch_size": 5,
                "worker_threads": 2,
                "preprocessing_threads": 2,
            },
            id="threaded",
        ),
        pytest.param({"batch_size": 7, "block_size": 512}, id="block-512"),
    ],
)
def test_chembl_pairs_match_rdkit_across_dispatch_options(
    chembl_mols: list[Chem.Mol],
    chembl_pairs: list[tuple[int, int]],
    execution_options: dict[str, int],
) -> None:
    result = findMCS(
        chembl_mols,
        mode="pairs",
        pairs=chembl_pairs,
        require_gpu=True,
        **execution_options,
    )
    _assert_batch_matches_rdkit(result, chembl_mols)


def test_chembl_all_pairs_mode_matches_rdkit(chembl_mols: list[Chem.Mol]) -> None:
    mols = chembl_mols[:12]
    result = findMCS(mols, require_gpu=True, batch_size=11)
    assert len(result) == 78
    _assert_batch_matches_rdkit(result, mols)


def test_chembl_paired_lists_mode_matches_rdkit(
    chembl_mols: list[Chem.Mol],
) -> None:
    midpoint = len(chembl_mols) // 2
    mols_a = chembl_mols[:midpoint]
    mols_b = chembl_mols[midpoint : 2 * midpoint]
    combined = mols_a + mols_b
    result = findMCS(
        mols_a,
        mode="paired_lists",
        mols_b=mols_b,
        require_gpu=True,
        batch_size=7,
    )
    _assert_batch_matches_rdkit(result, combined)


@pytest.mark.parametrize("num_atoms", [16, 17, 32, 33, 64, 65, 128])
def test_python_binding_dispatches_every_gpu_tier_boundary(num_atoms: int) -> None:
    mol = Chem.MolFromSmiles("C" * num_atoms)
    assert mol is not None

    result = findMCS(
        [mol],
        mode="pairs",
        pairs=[(0, 0)],
        require_gpu=True,
        block_size=128,
    )

    assert result.used_gpu.tolist() == [1]
    assert result.used_fallback.tolist() == [0]
    assert result.num_atoms.tolist() == [num_atoms]
    assert result.num_bonds.tolist() == [num_atoms - 1]
    _assert_mapping_is_common_subgraph(result[0], mol, mol)
