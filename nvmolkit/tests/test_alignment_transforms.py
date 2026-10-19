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

import gc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolAlign

from nvmolkit.conformerRmsd import GetAlignmentTransformBatch


def _molecule(positions, conformer_id=0):
    molecule = Chem.RWMol()
    for _ in positions:
        molecule.AddAtom(Chem.Atom(6))
    conformer = Chem.Conformer(len(positions))
    conformer.SetId(conformer_id)
    for atom, position in enumerate(positions):
        conformer.SetAtomPosition(atom, position)
    molecule.AddConformer(conformer, assignId=False)
    return molecule


def _apply(positions, transform):
    return np.asarray(positions) @ transform[:3, :3].T + transform[:3, 3]


def _svd_transform(moving, fixed):
    moving_center = moving.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    left, _, right = np.linalg.svd((moving - moving_center).T @ (fixed - fixed_center))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right)
    transform = np.eye(4)
    transform[:3, :3] = (left @ correction @ right).T
    transform[:3, 3] = fixed_center - transform[:3, :3] @ moving_center
    return transform


def _svd_rmsd(moving, fixed):
    aligned = _apply(moving, _svd_transform(moving, fixed))
    return np.sqrt(np.mean(np.sum((aligned - fixed) ** 2, axis=1)))


@pytest.mark.parametrize("count", [1, 2, 3, 4, 31, 128, 129, 1024, 2049])
@pytest.mark.parametrize("noise", [0.0, 0.07])
def test_alignment_matches_rdkit_and_independent_svd(count, noise):
    random = np.random.default_rng(108 + count)
    moving = random.normal(size=(count, 3))
    rotation, _ = np.linalg.qr(random.normal(size=(3, 3)))
    rotation[:, -1] *= np.linalg.det(rotation)
    fixed = moving @ rotation + [13, -5, 1.2] + noise * random.normal(size=moving.shape)
    probe, reference = _molecule(moving), _molecule(fixed)
    atom_map = [(atom, atom) for atom in range(count)]
    rmsds, transforms = GetAlignmentTransformBatch([probe], reference, [atom_map])
    rmsd, transform = rmsds.numpy()[0], transforms.numpy()[0]
    aligned = _apply(moving, transform)
    np.testing.assert_allclose(rmsd, _svd_rmsd(moving, fixed), atol=2e-9)
    np.testing.assert_allclose(rmsd, np.sqrt(np.mean(np.sum((aligned - fixed) ** 2, axis=1))), atol=2e-12)
    np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=2e-12)
    np.testing.assert_allclose(np.linalg.det(transform[:3, :3]), 1.0, atol=2e-12)
    np.testing.assert_array_equal(transform[3], [0, 0, 0, 1])
    rdkit_rmsd, rdkit_transform = rdMolAlign.GetAlignmentTransform(probe, reference, atomMap=atom_map)
    np.testing.assert_allclose(rmsd, rdkit_rmsd, atol=2e-6)
    if count >= 4:
        np.testing.assert_allclose(transform, rdkit_transform, atol=2e-5)
    np.testing.assert_array_equal(probe.GetConformer().GetPositions(), moving)
    np.testing.assert_array_equal(reference.GetConformer().GetPositions(), fixed)


def test_partial_reordered_maps_mixed_sizes_and_conformer_ids():
    random = np.random.default_rng(91)
    fixed = random.normal(size=(11, 3))
    reference = _molecule(fixed, 37)
    probes, maps, ids, expected = [], [], [], []
    for count, selected in [(4, [8, 1, 4]), (7, [4, 2, 9, 5, 3]), (11, list(range(11)))]:
        moving = random.normal(size=(count, 3))
        probe = _molecule(moving, 23)
        other = Chem.Conformer(probe.GetConformer(23))
        other.SetId(51)
        for atom in range(count):
            other.SetAtomPosition(atom, moving[atom] + [4, -8, 3])
        probe.AddConformer(other, assignId=False)
        atom_map = list(zip(range(len(selected)), selected))
        for conformer_id in [51, 23]:
            probes.append(probe)
            maps.append(atom_map)
            ids.append(conformer_id)
            expected.append(
                rdMolAlign.GetAlignmentTransform(probe, reference, prbCid=conformer_id, refCid=37, atomMap=atom_map)
            )
    rmsds, transforms = GetAlignmentTransformBatch(probes, reference, maps, ids, 37)
    np.testing.assert_allclose(rmsds.numpy(), [item[0] for item in expected], atol=2e-6)
    np.testing.assert_allclose(transforms.numpy(), [item[1] for item in expected], atol=2e-5)
    first_rmsds, first_transforms = GetAlignmentTransformBatch(probes, reference, maps)
    explicit_rmsds, explicit_transforms = GetAlignmentTransformBatch(probes, reference, maps, [23] * len(probes), 37)
    np.testing.assert_array_equal(first_rmsds.numpy(), explicit_rmsds.numpy())
    np.testing.assert_array_equal(first_transforms.numpy(), explicit_transforms.numpy())


@pytest.mark.parametrize("geometry", ["coincident", "collinear", "planar", "mirror", "large_translation"])
def test_degenerate_and_reflected_geometries(geometry):
    moving = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    if geometry == "coincident":
        moving[:] = 1.7
    elif geometry == "collinear":
        moving[:, 1:] = 0
    elif geometry == "planar":
        moving[:, 2] = 0
    fixed = moving[:, [1, 2, 0]] + [3, -7, 8]
    if geometry == "mirror":
        fixed[:, 0] *= -1
    elif geometry == "large_translation":
        moving += 1e6
        fixed += [-1e6, 2e6, -3e6]
    atom_map = [(atom, atom) for atom in range(4)]
    probe, reference = _molecule(moving), _molecule(fixed)
    first = GetAlignmentTransformBatch([probe], reference, [atom_map])
    second = GetAlignmentTransformBatch([probe], reference, [atom_map])
    rmsd, transform = first[0].numpy()[0], first[1].numpy()[0]
    np.testing.assert_allclose(rmsd, _svd_rmsd(moving, fixed), atol=2e-9)
    aligned_rmsd = np.sqrt(np.mean(np.sum((_apply(moving, transform) - fixed) ** 2, axis=1)))
    np.testing.assert_allclose(aligned_rmsd, rmsd, atol=2e-9)
    np.testing.assert_allclose(np.linalg.det(transform[:3, :3]), 1.0, atol=2e-12)
    if geometry == "mirror":
        assert rmsd > 0.5
    np.testing.assert_array_equal(first[0].numpy(), second[0].numpy())
    np.testing.assert_array_equal(first[1].numpy(), second[1].numpy())


@pytest.mark.parametrize(
    "atom_map",
    [
        [],
        [(-1, 0)],
        [(0, -1)],
        [(4, 0)],
        [(0, 4)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0,)],
        [(0, 1, 2)],
        [(0.5, 0)],
        [(0, "1")],
    ],
)
def test_invalid_maps(atom_map):
    molecule = _molecule(np.zeros((4, 3)))
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        GetAlignmentTransformBatch([molecule], molecule, [atom_map])


@pytest.mark.parametrize(
    "case",
    [
        "null_probe",
        "null_reference",
        "missing_map",
        "extra_map",
        "missing_id",
        "invalid_probe_id",
        "invalid_reference_id",
        "negative_probe_id",
        "negative_reference_id",
        "no_probe_conformer",
        "no_reference_conformer",
        "no_atoms",
        "bad_stream",
    ],
)
def test_invalid_inputs(case):
    molecule = _molecule(np.zeros((3, 3)))
    arguments = dict(probes=[molecule], reference=molecule, atom_maps=[[(0, 0)]])
    if case == "null_probe":
        arguments["probes"] = [None]
    elif case == "null_reference":
        arguments["reference"] = None
    elif case in ["missing_map", "extra_map"]:
        arguments["atom_maps"] = [] if case == "missing_map" else [[(0, 0)], [(1, 1)]]
    elif case == "missing_id":
        arguments["probe_conf_ids"] = []
    elif case in ["invalid_probe_id", "negative_probe_id"]:
        arguments["probe_conf_ids"] = [42 if case == "invalid_probe_id" else -2]
    elif case in ["invalid_reference_id", "negative_reference_id"]:
        arguments["ref_conf_id"] = 42 if case == "invalid_reference_id" else -2
    elif case in ["no_probe_conformer", "no_reference_conformer"]:
        empty = Chem.Mol(molecule)
        empty.RemoveAllConformers()
        arguments["probes" if case == "no_probe_conformer" else "reference"] = (
            [empty] if case == "no_probe_conformer" else empty
        )
    elif case == "no_atoms":
        arguments["probes"] = [_molecule(np.zeros((0, 3)))]
    elif case == "bad_stream":
        arguments["stream"] = 7
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        GetAlignmentTransformBatch(**arguments)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("side", ["probe", "reference"])
def test_nonfinite_mapped_coordinates_rejected(value, side):
    clean = _molecule(np.zeros((3, 3)))
    positions = np.zeros((3, 3))
    positions[1, 2] = value
    invalid = _molecule(positions)
    probe, reference = (invalid, clean) if side == "probe" else (clean, invalid)
    with pytest.raises(ValueError, match="finite"):
        GetAlignmentTransformBatch([probe], reference, [[(1, 1)]])
    rmsds, _ = GetAlignmentTransformBatch([probe], reference, [[(0, 0)]])
    np.testing.assert_array_equal(rmsds.numpy(), [0])


def test_empty_batch():
    reference = _molecule(np.zeros((1, 3)))
    rmsds, transforms = GetAlignmentTransformBatch([], reference, [])
    assert rmsds.torch().shape == (0,)
    assert transforms.torch().shape == (0, 4, 4)
    assert rmsds.torch().dtype == transforms.torch().dtype == torch.float64


def test_stream_concurrency_and_result_ownership():
    def run(seed):
        random = np.random.default_rng(seed)
        positions = random.normal(size=(135, 3))
        probe = _molecule(positions)
        reference = _molecule(positions + [9, -3, 7])
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            rmsds, transforms = GetAlignmentTransformBatch(
                [probe] * 17,
                reference,
                [[(atom, atom) for atom in range(135)]] * 17,
                stream=stream if seed % 2 else None,
            )
            aligned = torch.tensor(positions, device="cuda", dtype=torch.float64) @ transforms.torch()[0, :3, :3].T
            aligned += transforms.torch()[0, :3, 3]
        del probe, reference
        gc.collect()
        stream.synchronize()
        np.testing.assert_allclose(rmsds.numpy(), 0, atol=1e-12)
        np.testing.assert_allclose(aligned.cpu().numpy(), positions + [9, -3, 7], atol=1e-12)
        np.testing.assert_allclose(transforms.numpy()[:, :3, 3], np.tile([9, -3, 7], (17, 1)), atol=1e-12)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, [10, 11, 12, 13]))


def test_molecular_conformer_batch_matches_rdkit():
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCOC(=O)c1ccccc1"))
    ids = list(rdDistGeom.EmbedMultipleConfs(molecule, numConfs=9, randomSeed=42))
    assert len(ids) == 9
    heavy_atoms = [(atom.GetIdx(), atom.GetIdx()) for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    positions_before = [conformer.GetPositions().copy() for conformer in molecule.GetConformers()]
    rmsds, transforms = GetAlignmentTransformBatch(
        [molecule] * len(ids), molecule, [heavy_atoms] * len(ids), ids, ref_conf_id=ids[3]
    )
    expected = [
        rdMolAlign.GetAlignmentTransform(molecule, molecule, prbCid=conformer_id, refCid=ids[3], atomMap=heavy_atoms)
        for conformer_id in ids
    ]
    np.testing.assert_allclose(rmsds.numpy(), [item[0] for item in expected], atol=2e-6)
    indices = [probe_atom for probe_atom, _ in heavy_atoms]
    reference_positions = molecule.GetConformer(ids[3]).GetPositions()[indices]
    svd_transforms = [
        _svd_transform(molecule.GetConformer(conformer_id).GetPositions()[indices], reference_positions)
        for conformer_id in ids
    ]
    np.testing.assert_allclose(transforms.numpy(), svd_transforms, atol=2e-12)
    for before, conformer in zip(positions_before, molecule.GetConformers()):
        np.testing.assert_array_equal(before, conformer.GetPositions())
