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
from rdkit import Chem, Geometry
from rdkit.Chem import rdDistGeom, rdMIF, rdPartialCharges

from nvmolkit.interactionFields import CalcCoulombFields
from nvmolkit.types import AsyncGpuResult


def _molecule(positions, charges, conformer_id=0, charge_property="_GasteigerCharge"):
    molecule = Chem.RWMol()
    for charge in charges:
        atom = Chem.Atom(6)
        atom.SetDoubleProp(charge_property, float(charge))
        molecule.AddAtom(atom)
    conformer = Chem.Conformer(len(charges))
    conformer.SetId(conformer_id)
    for atom, position in enumerate(positions):
        conformer.SetAtomPosition(atom, position)
    molecule.AddConformer(conformer, assignId=False)
    return molecule


def _reference(molecules, points, model="vacuum", conf_ids=None, **options):
    constructor = rdMIF.Coulomb if model == "vacuum" else rdMIF.CoulombDielectric
    arguments = dict(
        probeCharge=options.get("probe_charge", 1.0),
        absVal=options.get("abs_val", False),
        chargeKey=options.get("charge_property", "_GasteigerCharge"),
        softcoreParam=options.get("alpha", 0.0),
        cutoff=options.get("cutoff", 1.0),
    )
    if model == "dielectric":
        arguments.update(epsilon=options.get("epsilon", 80.0), xi=options.get("xi", 4.0))
    ids = [-1] * len(molecules) if conf_ids is None else conf_ids
    threshold = options.get("threshold")
    squared_threshold = np.inf if threshold is None else threshold**2
    rows = []
    for molecule, conformer_id in zip(molecules, ids):
        field = constructor(molecule, confId=conformer_id, **arguments)
        rows.append(
            [field(*map(float, point), float(squared_threshold)) for point in np.asarray(points).reshape(-1, 3)]
        )
    return np.asarray(rows).reshape((len(molecules), *points.shape[:-1]))


@pytest.mark.parametrize("model", ["vacuum", "dielectric"])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"alpha": 0.37},
        {"probe_charge": -0.7, "abs_val": True, "cutoff": 0.4},
        {"cutoff": 0.0},
        {"threshold": 2.3},
        {"alpha": 1e-6, "epsilon": 5.0, "xi": 2.0},
    ],
)
def test_mixed_batches_match_rdkit(model, options):
    random = np.random.default_rng(19)
    molecules = []
    for count in [1, 6, 7, 8, 9, 10, 11, 12, 13, 65, 129]:
        positions = random.normal(size=(count, 3)) * 0.5
        positions[0] = 0
        molecules.append(_molecule(positions, random.normal(size=count)))
    points = random.normal(size=(19, 3)) * 3
    points[0] = 0
    result = CalcCoulombFields(molecules, points, model=model, **options)
    expected = _reference(molecules, points, model=model, **options)
    np.testing.assert_allclose(result.numpy(), expected, rtol=2e-11, atol=2e-9)
    np.testing.assert_array_equal(result.numpy(), CalcCoulombFields(molecules, points, model=model, **options).numpy())


@pytest.mark.parametrize("model", ["vacuum", "dielectric"])
@pytest.mark.parametrize(
    "alpha,cutoff", [(0, 0), (0, 1e-7), (0, 1e-6), (0, 0.001), (0, 1), (0.999e-6, 0.2), (1e-6, 0.2), (1.001e-6, 0.2)]
)
def test_cutoff_and_softcore_boundaries(model, alpha, cutoff):
    molecule = _molecule([[0, 0, 0]], [1.2])
    points = np.array([[distance, 0, 0] for distance in [0, 1e-7, 0.001, 0.199999, 0.2, 0.200001, 1, 4]])
    result = CalcCoulombFields([molecule], points, model=model, alpha=alpha, cutoff=cutoff)
    np.testing.assert_allclose(
        result.numpy(), _reference([molecule], points, model=model, alpha=alpha, cutoff=cutoff), rtol=2e-12, atol=2e-9
    )


@pytest.mark.parametrize("model", ["vacuum", "dielectric"])
def test_threshold_is_strict_and_unsquared_in_public_api(model):
    molecule = _molecule([[0, 0, 0]], [1])
    points = np.array([[2 - 1e-8, 0, 0], [2, 0, 0], [2 + 1e-8, 0, 0], [0, 0, 0]])
    output = CalcCoulombFields([molecule], points, model=model, threshold=2).numpy()
    assert output[0, 0] > 0
    assert output[0, 3] > 0
    np.testing.assert_array_equal(output[0, 1:3], 0)
    np.testing.assert_array_equal(CalcCoulombFields([molecule], points, model=model, threshold=0).numpy(), 0)
    np.testing.assert_allclose(output, _reference([molecule], points, model=model, threshold=2), rtol=2e-12)


def test_dielectric_density_counts_use_all_atoms_and_strict_four_angstrom_radius():
    positions = np.zeros((14, 3))
    positions[:, 0] = np.linspace(-1, 1, 14)
    positions[-1, 0] = 4
    charges = np.linspace(-0.2, 0.3, 14)
    molecule = _molecule(positions, charges)
    points = np.array([[0, 0, 0], [1e-8, 0, 0], [-1e-8, 0, 0], [4, 0, 0], [8, 0, 0]])
    output = CalcCoulombFields([molecule], points, model="dielectric", threshold=0.75, cutoff=0.2).numpy()
    expected = _reference([molecule], points, model="dielectric", threshold=0.75, cutoff=0.2)
    np.testing.assert_allclose(output, expected, rtol=2e-12, atol=2e-10)
    isolated = _molecule([[0, 0, 0]], [1])
    crowded = _molecule(np.vstack([np.zeros(3), np.tile([2, 0, 0], (7, 1))]), [1] + [0] * 7)
    probe = np.array([[0.0, 0.0, 0.0]])
    fields = CalcCoulombFields([isolated, crowded], probe, model="dielectric", threshold=0.5).numpy()
    assert abs(fields[0, 0] - fields[1, 0]) > 1


def test_physical_scaling_equal_permittivities_and_absolute_value_after_sum():
    molecule = _molecule([[0, 0, 0], [3, 0, 0]], [1, -0.5])
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    prefactor = 1 / (4 * 3.141592 * 8.854188) * 1.602**2 * 6.02214129 * 10000
    vacuum = CalcCoulombFields([molecule], points).numpy()
    np.testing.assert_allclose(vacuum, [[0.75 * prefactor, 0]], atol=2e-12)
    negative = CalcCoulombFields([molecule], points, probe_charge=-2).numpy()
    np.testing.assert_allclose(negative, -2 * vacuum, atol=2e-12)
    np.testing.assert_allclose(CalcCoulombFields([molecule], points, abs_val=True).numpy(), -abs(vacuum), atol=2e-12)
    np.testing.assert_array_equal(CalcCoulombFields([molecule], points, probe_charge=0).numpy(), 0)
    dielectric = CalcCoulombFields([molecule], points, model="dielectric", epsilon=7, xi=7).numpy()
    np.testing.assert_allclose(dielectric, vacuum / 7, atol=2e-12)


@pytest.mark.parametrize("kind", ["numpy32", "numpy64", "cpu_tensor", "cuda32", "cuda64", "cuda_strided", "async"])
def test_grid_shape_input_types_and_layout(kind):
    random = np.random.default_rng(7)
    points = random.normal(size=(2, 3, 5, 3))
    if kind == "numpy32":
        value = points.astype(np.float32)
    elif kind == "numpy64":
        value = points[:, ::-1]
    elif kind == "cpu_tensor":
        value = torch.tensor(points)
    else:
        tensor = torch.tensor(points, device="cuda", dtype=torch.float32 if kind == "cuda32" else torch.float64)
        value = tensor.transpose(0, 1) if kind == "cuda_strided" else tensor
        if kind == "async":
            value = AsyncGpuResult(value)
    actual_points = value.numpy() if isinstance(value, AsyncGpuResult) else value
    if isinstance(actual_points, torch.Tensor):
        actual_points = actual_points.cpu().numpy()
    molecule = _molecule([[1, 2, 3]], [-0.3])
    result = CalcCoulombFields([molecule], value)
    assert result.torch().dtype == torch.float64
    assert result.torch().shape == (1, *actual_points.shape[:-1])
    np.testing.assert_allclose(result.numpy(), _reference([molecule], actual_points), rtol=2e-12)


def test_rdkit_regular_grid_ordering_and_molecular_conformers():
    molecule = Chem.AddHs(Chem.MolFromSmiles("CC(=O)N"))
    conformer_ids = list(rdDistGeom.EmbedMultipleConfs(molecule, numConfs=3, randomSeed=42))
    assert len(conformer_ids) == 3
    rdPartialCharges.ComputeGasteigerCharges(molecule)
    original_positions = [conformer.GetPositions().copy() for conformer in molecule.GetConformers()]
    for atom in molecule.GetAtoms():
        atom.SetDoubleProp("custom_charge", atom.GetDoubleProp("_GasteigerCharge"))
    for index, conformer in enumerate(molecule.GetConformers()):
        conformer.SetId(10 + index * 3)
    conformer_ids = [16, 10, 13]
    for model in ["vacuum", "dielectric"]:
        grid = Geometry.UniformRealValueGrid3D(4, 5, 6, 0.5)
        flat_points = np.array([list(grid.GetGridPointLoc(index)) for index in range(grid.GetSize())])
        points = flat_points.reshape(grid.GetNumZ(), grid.GetNumY(), grid.GetNumX(), 3)
        result = CalcCoulombFields(
            [molecule] * 3, points, model=model, conf_ids=conformer_ids, charge_property="custom_charge"
        )
        expected = _reference(
            [molecule] * 3, points, model=model, conf_ids=conformer_ids, charge_property="custom_charge"
        )
        np.testing.assert_allclose(result.numpy(), expected, rtol=2e-11, atol=2e-10)
        constructor = rdMIF.Coulomb if model == "vacuum" else rdMIF.CoulombDielectric
        field = constructor(molecule, confId=16, chargeKey="custom_charge")
        rdMIF.CalculateDescriptors(grid, field)
        np.testing.assert_allclose(
            result.numpy()[0].reshape(-1), np.array(grid.GetOccupancyVect()), rtol=2e-11, atol=2e-10
        )
    for before, conformer in zip(original_positions, molecule.GetConformers()):
        np.testing.assert_array_equal(before, conformer.GetPositions())
    default = CalcCoulombFields([molecule], points).numpy()
    explicit = CalcCoulombFields([molecule], points, conf_ids=[10]).numpy()
    np.testing.assert_array_equal(default, explicit)


@pytest.mark.parametrize("model", ["vacuum", "dielectric"])
def test_empty_batches_queries_and_molecules(model):
    molecule = _molecule([[0, 0, 0]], [1])
    points = np.zeros((2, 3))
    assert CalcCoulombFields([], points, model=model).torch().shape == (0, 2)
    assert CalcCoulombFields([molecule], np.zeros((3, 0, 3)), model=model).torch().shape == (1, 3, 0)
    empty = _molecule(np.zeros((0, 3)), [])
    result = CalcCoulombFields([empty, molecule, empty], points, model=model).numpy()
    np.testing.assert_array_equal(result[[0, 2]], 0)
    assert (result[1] > 0).all()


@pytest.mark.parametrize(
    "option,value",
    [
        ("alpha", -1),
        ("alpha", np.nan),
        ("alpha", np.inf),
        ("cutoff", -1),
        ("cutoff", np.nan),
        ("cutoff", np.inf),
        ("cutoff", 1e200),
        ("threshold", -1),
        ("threshold", np.nan),
        ("threshold", np.inf),
        ("threshold", 1e200),
        ("epsilon", 0),
        ("epsilon", -1),
        ("epsilon", np.nan),
        ("epsilon", np.inf),
        ("xi", 0),
        ("xi", -1),
        ("xi", np.nan),
        ("xi", np.inf),
        ("probe_charge", np.nan),
        ("probe_charge", np.inf),
        ("model", "other"),
        ("charge_property", ""),
    ],
)
def test_invalid_options(option, value):
    molecule = _molecule([[0, 0, 0]], [1])
    with pytest.raises((ValueError, TypeError)):
        CalcCoulombFields([molecule], np.zeros((1, 3)), **{option: value})


@pytest.mark.parametrize(
    "case",
    [
        "null",
        "missing_id",
        "invalid_id",
        "negative_id",
        "no_conformer",
        "missing_charge",
        "nan_charge",
        "inf_charge",
        "nan_position",
        "inf_position",
        "bad_stream",
    ],
)
def test_invalid_molecule_inputs(case):
    molecule = _molecule([[0, 0, 0]], [1])
    arguments = dict(mols=[molecule], points=np.zeros((1, 3)))
    if case == "null":
        arguments["mols"] = [None]
    elif case in ["missing_id", "invalid_id", "negative_id"]:
        arguments["conf_ids"] = [] if case == "missing_id" else [42 if case == "invalid_id" else -2]
    elif case == "no_conformer":
        molecule.RemoveAllConformers()
    elif case == "missing_charge":
        molecule.GetAtomWithIdx(0).ClearProp("_GasteigerCharge")
    elif case in ["nan_charge", "inf_charge"]:
        molecule.GetAtomWithIdx(0).SetDoubleProp("_GasteigerCharge", np.nan if case == "nan_charge" else np.inf)
    elif case in ["nan_position", "inf_position"]:
        molecule.GetConformer().SetAtomPosition(0, [np.nan if case == "nan_position" else np.inf, 0, 0])
    elif case == "bad_stream":
        arguments["stream"] = 5
    with pytest.raises((ValueError, RuntimeError, KeyError, TypeError)):
        CalcCoulombFields(**arguments)


@pytest.mark.parametrize(
    "points",
    [
        np.zeros(3),
        np.zeros((2, 2)),
        np.zeros((1, 3), dtype=int),
        np.array([[np.nan, 0, 0]]),
        np.array([[np.inf, 0, 0]]),
        [[0, 0, 0]],
    ],
)
def test_invalid_query_inputs(points):
    molecule = _molecule([[0, 0, 0]], [1])
    with pytest.raises((ValueError, TypeError)):
        CalcCoulombFields([molecule], points)


def test_streams_concurrency_and_buffer_lifetimes():
    def run(seed):
        random = np.random.default_rng(seed)
        molecule = _molecule(random.normal(size=(33, 3)), random.normal(size=33))
        host_points = random.normal(size=(1025, 3))
        model = "vacuum" if seed % 2 else "dielectric"
        expected = _reference([molecule], host_points, model=model)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            points = torch.tensor(host_points, device="cuda", dtype=torch.float32)
            result = CalcCoulombFields([molecule], points, model=model, stream=stream if seed % 2 else None)
            downstream = result.torch() * 2
        del points, molecule, result
        gc.collect()
        stream.synchronize()
        np.testing.assert_allclose(downstream.cpu().numpy(), 2 * expected, rtol=2e-5, atol=2e-4)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, [10, 11, 12, 13]))


def test_wrong_stream_device_rejected():
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two GPUs")
    molecule = _molecule([[0, 0, 0]], [1])
    points = torch.zeros((1, 3), device="cuda:0")
    stream = torch.cuda.Stream(device=1)
    with pytest.raises(ValueError, match="stream"):
        CalcCoulombFields([molecule], points, stream=stream)


def test_large_grid_spans_more_than_one_launch_iteration():
    molecule = _molecule([[0, 0, 0]], [1])
    points = torch.zeros((65535 * 128 + 17, 3), device="cuda", dtype=torch.float64)
    prefactor = 1 / (4 * 3.141592 * 8.854188) * 1.602**2 * 6.02214129 * 10000
    for model, factor in [("vacuum", 1.0), ("dielectric", 1 / 42)]:
        output = CalcCoulombFields([molecule], points, model=model).torch()
        torch.testing.assert_close(output, torch.full_like(output, prefactor * factor), rtol=2e-12, atol=2e-12)
