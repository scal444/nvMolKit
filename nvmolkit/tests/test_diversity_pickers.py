# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import cache

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.SimDivFilters import rdSimDivPickers

from nvmolkit.clustering import (
    ClusterDeviceResult,
    OutputMode,
    SelectionDeviceResult,
    dise,
    fused_butina,
    fused_dise,
    fused_leader,
    leader,
)
from nvmolkit.similarity import CosineMetric, TanimotoMetric, aap_similarity
from nvmolkit.types import AsyncGpuResult

RDKIT = OutputMode.RDKIT


@pytest.fixture(scope="module")
def aap_molecules(chembl_molecules):
    """110 AAP-compatible molecules plus atom-renumbered copies of two of them."""
    molecules = [molecule for molecule in chembl_molecules if molecule.GetNumAtoms() <= 64][:110]
    duplicates = [
        Chem.RenumberAtoms(molecules[index], list(reversed(range(molecules[index].GetNumAtoms()))))
        for index in (5, 17)
    ]
    return molecules + duplicates


def _reference_dise(distances, cutoff, assignment):
    """Direct NumPy DISE in RDKit cluster format."""
    cutoff = distances.dtype.type(cutoff)
    num_items = len(distances)
    active = np.ones(num_items, dtype=bool)
    labels = np.full(num_items, -1)
    centroids = []
    for item in range(num_items):
        if not active[item]:
            continue
        excluded = active & (distances[item] <= cutoff)
        excluded[item] = True
        labels[excluded] = len(centroids)
        active &= ~excluded
        centroids.append(item)
    if assignment == "nearest":
        non_centroids = np.ones(num_items, dtype=bool)
        non_centroids[centroids] = False
        labels[non_centroids] = np.argmin(distances[centroids], axis=0)[non_centroids]
    sizes = np.bincount(labels, minlength=len(centroids))
    order = sorted(range(len(centroids)), key=lambda cluster: -sizes[cluster])
    return tuple(
        (centroids[cluster], *(int(item) for item in np.flatnonzero(labels == cluster) if item != centroids[cluster]))
        for cluster in order
    )


def _fingerprint_distance_matrix(fingerprints, metric):
    counts = np.asarray([sum(int(word).bit_count() for word in row) for row in fingerprints])
    result = np.zeros((len(fingerprints), len(fingerprints)), dtype=np.float64)
    for left in range(len(fingerprints)):
        for right in range(len(fingerprints)):
            intersection = sum(
                (int(left_word) & int(right_word)).bit_count()
                for left_word, right_word in zip(fingerprints[left], fingerprints[right], strict=True)
            )
            if metric == "tanimoto":
                denominator = counts[left] + counts[right] - intersection
                similarity = intersection / denominator if denominator else 1.0
            else:
                denominator = np.sqrt(counts[left] * counts[right])
                similarity = intersection / denominator if denominator else 0.0
            result[left, right] = 1.0 - similarity
    return result


# ---------------------------------------------------------------------------
# Integration: ChEMBL fingerprints against RDKit and the matrix APIs
# ---------------------------------------------------------------------------


# Distinct Tanimoto values keep their order in float32, so results equal RDKit's double-precision pickers except
# where a distance equals a cutoff that float32 cannot represent exactly. RDKit parity tests use dyadic cutoffs.
@pytest.mark.parametrize("cutoff", [0.0, 0.25, 0.5, 0.625, 1.0])
def test_fused_leader_matches_rdkit_on_chembl(chembl_fingerprints, cutoff):
    bit_vectors, packed = chembl_fingerprints
    expected = tuple(rdSimDivPickers.LeaderPicker().LazyBitVectorPick(bit_vectors, len(bit_vectors), cutoff))

    assert fused_leader(packed, cutoff, output=RDKIT) == expected


@pytest.mark.parametrize("num_first_picks", [3, 40])
def test_fused_leader_first_picks_and_pick_size_match_rdkit_on_chembl(chembl_fingerprints, num_first_picks):
    bit_vectors, packed = chembl_fingerprints
    first_picks = [int(index) for index in np.random.default_rng(5).permutation(len(bit_vectors))[:num_first_picks]]
    expected = tuple(
        rdSimDivPickers.LeaderPicker().LazyBitVectorPick(
            bit_vectors, len(bit_vectors), 0.5, pickSize=50, firstPicks=first_picks
        )
    )

    assert fused_leader(packed, 0.5, pick_size=50, first_picks=first_picks, output=RDKIT) == expected


@pytest.mark.parametrize("metric", ["tanimoto", "cosine"])
@pytest.mark.parametrize("cutoff", [0.3, 0.55])
def test_fused_algorithms_match_matrix_forms_on_chembl(chembl_fingerprints, chembl_distances, metric, cutoff):
    _, packed = chembl_fingerprints
    distances = chembl_distances[metric]

    assert fused_leader(packed, cutoff, metric=metric, output=RDKIT) == leader(distances, cutoff, output=RDKIT)
    for assignment in ("first", "nearest"):
        assert fused_dise(packed, cutoff, metric=metric, assignment=assignment, output=RDKIT) == dise(
            distances, cutoff, assignment=assignment, output=RDKIT
        )


def test_float32_and_float64_matrices_agree_on_chembl(chembl_distances):
    distances = chembl_distances["tanimoto"]
    widened = distances.astype(np.float64)

    assert leader(widened, 0.3, output=RDKIT) == leader(distances, 0.3, output=RDKIT)
    assert dise(widened, 0.3, output=RDKIT) == dise(distances, 0.3, output=RDKIT)


@pytest.mark.parametrize("assignment", ["first", "nearest"])
@pytest.mark.parametrize("cutoff", [0.3, 0.6])
def test_dise_matches_reference_on_chembl(chembl_distances, assignment, cutoff):
    distances = chembl_distances["tanimoto"]

    assert dise(distances, cutoff, assignment=assignment, output=RDKIT) == _reference_dise(
        distances, cutoff, assignment
    )


def test_dise_device_result_is_consistent_on_chembl(chembl_fingerprints):
    _, packed = chembl_fingerprints

    result = fused_dise(packed, 0.4)
    cluster_ids = result.cluster_ids.numpy()
    centroids = result.centroids.numpy()
    sizes = result.cluster_sizes.numpy()

    assert cluster_ids.shape == (len(packed),)
    assert np.array_equal(np.bincount(cluster_ids, minlength=len(centroids)), sizes)
    assert np.array_equal(cluster_ids[centroids], np.arange(len(centroids)))
    assert np.all(np.diff(sizes) <= 0)


@pytest.mark.parametrize("num_words", [3, 260])
@pytest.mark.parametrize("metric", ["tanimoto", "cosine"])
def test_fused_forms_match_matrix_forms_for_unusual_widths(float32_distances, num_words, metric):
    # 3 words cannot use vector loads; 260 words exceed the shared-memory source tile.
    rng = np.random.default_rng(num_words)
    bits = rng.random((300, num_words * 32)) < 0.03
    bits[:40] = bits[40:80] | (rng.random((40, num_words * 32)) < 0.01)
    packed = np.packbits(bits, axis=1, bitorder="little").view(np.uint32)
    distances = float32_distances(packed, metric)

    assert fused_leader(packed, 0.9, metric=metric, output=RDKIT) == leader(distances, 0.9, output=RDKIT)
    assert fused_dise(packed, 0.9, metric=metric, output=RDKIT) == dise(distances, 0.9, output=RDKIT)


def test_fused_forms_handle_buffers_not_aligned_for_vector_loads(chembl_fingerprints):
    _, packed = chembl_fingerprints
    storage = torch.zeros(packed.size + 1, dtype=torch.int32, device="cuda")
    shifted = storage[1:].view(packed.shape)
    shifted.copy_(torch.from_numpy(packed.view(np.int32)))
    assert shifted.data_ptr() % 16 != 0

    assert fused_leader(shifted, 0.4, output=RDKIT) == fused_leader(packed, 0.4, output=RDKIT)
    assert fused_dise(shifted, 0.4, output=RDKIT) == fused_dise(packed, 0.4, output=RDKIT)


@pytest.mark.parametrize("form", ["int32", "torch_cpu", "torch_cuda", "async", "non_contiguous"])
def test_fused_inputs_accept_array_forms(chembl_fingerprints, form):
    _, packed = chembl_fingerprints
    expected = fused_leader(packed, 0.4, output=RDKIT)
    if form == "int32":
        value = packed.view(np.int32)
    elif form == "torch_cpu":
        value = torch.from_numpy(packed.view(np.int32))
    elif form == "torch_cuda":
        value = torch.from_numpy(packed.view(np.int32)).cuda()
    elif form == "async":
        value = AsyncGpuResult(torch.from_numpy(packed.view(np.int32)).cuda())
    else:
        value = torch.from_numpy(np.asfortranarray(packed.view(np.int32))).cuda()
        value = value.t().contiguous().t()
        assert not value.is_contiguous()

    assert fused_leader(value, 0.4, output=RDKIT) == expected


def test_explicit_stream_matches_default_stream_on_chembl(chembl_fingerprints):
    _, packed = chembl_fingerprints
    stream = torch.cuda.Stream()

    selection = fused_leader(packed, 0.4, stream=stream)
    clusters = fused_dise(packed, 0.4, stream=stream)
    stream.synchronize()

    assert selection.indices.numpy().tolist() == list(fused_leader(packed, 0.4, output=RDKIT))
    assert clusters.cluster_ids.numpy().tolist() == fused_dise(packed, 0.4).cluster_ids.numpy().tolist()


# ---------------------------------------------------------------------------
# Integration: AAP as a fused metric, against RDKit's lazy pickers
# ---------------------------------------------------------------------------


def _aap_distance(molecules):
    @cache
    def distance(selected, candidate):
        return 1.0 - aap_similarity(molecules[selected], molecules[candidate])

    return distance


def test_aap_leader_and_first_assignment_match_rdkit_lazy_picker(aap_molecules):
    # AAP similarities across diverse ChEMBL compounds are low; this cutoff still yields ~30 multi-member clusters.
    cutoff = 0.95
    distance = _aap_distance(aap_molecules)
    # RDKit's LeaderPicker evaluates func(leader, candidate).
    expected_leaders = tuple(rdSimDivPickers.LeaderPicker().LazyPick(distance, len(aap_molecules), cutoff))

    assert fused_leader(aap_molecules, cutoff, metric="aap", output=RDKIT) == expected_leaders

    labels = [
        next(k for k, leader_index in enumerate(expected_leaders) if distance(leader_index, item) <= cutoff)
        if item not in expected_leaders
        else expected_leaders.index(item)
        for item in range(len(aap_molecules))
    ]
    result = fused_dise(aap_molecules, cutoff, metric="aap", assignment="first")
    centroids = result.centroids.numpy()
    assert [int(centroids[label]) for label in result.cluster_ids.numpy()] == [expected_leaders[k] for k in labels]


def test_aap_nearest_assignment_picks_the_nearest_centroid(aap_molecules):
    molecules = aap_molecules[:40]
    cutoff = 0.7
    distance = _aap_distance(molecules)

    result = fused_dise(molecules, cutoff, metric="aap", assignment="nearest")
    centroids = result.centroids.numpy().tolist()
    for item, label in enumerate(result.cluster_ids.numpy().tolist()):
        if item in centroids:
            assert centroids[label] == item
            continue
        nearest = min(distance(centroid, item) for centroid in centroids)
        assert distance(centroids[label], item) == nearest


def test_aap_is_not_yet_supported_by_fused_butina():
    with pytest.raises(NotImplementedError, match="AAPMetric"):
        fused_butina(np.zeros((1, 1), dtype=np.uint32), 0.5, metric="aap")


# ---------------------------------------------------------------------------
# Semantics and pathological inputs
# ---------------------------------------------------------------------------


def test_matrix_rows_are_distances_from_the_selected_item():
    distances = np.asarray(
        [
            [0.0, 0.1, 0.8],
            [0.9, 0.0, 0.1],
            [0.2, 0.9, 0.0],
        ],
        dtype=np.float64,
    )

    assert leader(distances, 0.2, output=RDKIT) == (0, 2)
    assert dise(distances, 0.2, assignment="first", output=RDKIT) == ((0, 1), (2,))


def test_forced_picks_are_kept_even_when_they_exclude_each_other():
    points = np.asarray([0.0, 0.05, 0.25, 0.6, 0.65, 1.0])
    distances = np.abs(points[:, None] - points[None, :])
    distance = lambda left, right: distances[left, right]  # noqa: E731

    expected = tuple(rdSimDivPickers.LeaderPicker().LazyPick(distance, 6, 0.2, pickSize=1, firstPicks=[4, 3, 1]))
    assert leader(distances, 0.2, pick_size=1, first_picks=(4, 3, 1), output=RDKIT) == expected == (4, 3, 1)


def test_identical_inputs_collapse_to_one_leader():
    fingerprints = np.tile(np.asarray([[0b1011, 0b0110]], dtype=np.uint32), (300, 1))

    assert fused_leader(fingerprints, 0.0, output=RDKIT) == (0,)
    assert fused_dise(fingerprints, 0.0, output=RDKIT) == (tuple(range(300)),)


@pytest.mark.parametrize("metric", [TanimotoMetric(), CosineMetric()])
@pytest.mark.parametrize("cutoff", [0.0, 0.5, 1.0])
def test_empty_fingerprints_follow_each_metric_convention(metric, cutoff):
    # Tanimoto defines two empty fingerprints as identical; cosine defines them as unrelated.
    metric_name = "tanimoto" if isinstance(metric, TanimotoMetric) else "cosine"
    fingerprints = np.asarray([[0], [0b0011], [0], [0b0010], [0b1100]], dtype=np.uint32)
    distances = _fingerprint_distance_matrix(fingerprints, metric_name)

    assert fused_leader(fingerprints, cutoff, metric=metric, output=RDKIT) == leader(distances, cutoff, output=RDKIT)
    for assignment in ("first", "nearest"):
        assert fused_dise(fingerprints, cutoff, metric=metric, assignment=assignment, output=RDKIT) == dise(
            distances, cutoff, assignment=assignment, output=RDKIT
        )


def test_leader_removes_itself_without_a_zero_diagonal():
    distances = np.full((3, 3), 0.8)

    assert leader(distances, 0.1, output=RDKIT) == (0, 1, 2)


def test_empty_and_singleton_inputs():
    empty_matrix = np.empty((0, 0))
    empty_fingerprints = np.empty((0, 4), dtype=np.uint32)
    singleton = np.zeros((1, 1))

    assert leader(empty_matrix, 0.0, output=RDKIT) == ()
    assert dise(empty_matrix, 0.0, output=RDKIT) == ()
    assert fused_leader(empty_fingerprints, 0.5, output=RDKIT) == ()
    assert fused_dise(empty_fingerprints, 0.5, output=RDKIT) == ()
    assert fused_leader([], 0.5, metric="aap", output=RDKIT) == ()
    assert fused_dise([], 0.5, metric="aap", output=RDKIT) == ()
    assert leader(singleton, 0.0, output=RDKIT) == (0,)
    assert dise(singleton, 0.0, output=RDKIT) == ((0,),)


def test_numpy_and_torch_integer_arguments_are_accepted():
    points = np.asarray([0.0, 0.05, 0.25, 0.6, 0.65, 1.0])
    distances = np.abs(points[:, None] - points[None, :])

    assert leader(distances, 0.2, pick_size=np.int64(2), first_picks=np.asarray([3]), output=RDKIT) == (3, 0)


@pytest.mark.parametrize(
    "function, args",
    [
        (fused_leader, (0.5,)),
        (fused_dise, (0.5,)),
        (fused_butina, (0.5,)),
    ],
)
def test_metric_names_and_instances_are_equivalent(function, args):
    fingerprints = np.asarray([[0b0011], [0b0010], [0b1100], [0b0111]], dtype=np.uint32)

    for name, instance in (("tanimoto", TanimotoMetric()), ("cosine", CosineMetric())):
        by_name = function(fingerprints, *args, metric=name, output=RDKIT)
        assert function(fingerprints, *args, metric=instance, output=RDKIT) == by_name


def test_device_outputs_have_documented_types():
    points = np.asarray([0.0, 0.05, 0.25, 0.6, 0.65, 1.0])
    distances = np.abs(points[:, None] - points[None, :])

    selection = leader(distances, 0.2)
    clusters = dise(distances, 0.2)

    assert isinstance(selection, SelectionDeviceResult)
    assert selection.indices.torch().dtype == torch.int32
    assert selection.indices.torch().shape == (4,)
    assert isinstance(clusters, ClusterDeviceResult)
    assert clusters.cluster_ids.torch().dtype == torch.int32
    assert clusters.centroids.torch().dtype == torch.int32
    assert clusters.cluster_sizes.torch().dtype == torch.int64


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _small_matrix():
    return np.zeros((6, 6))


@pytest.mark.parametrize("function", [leader, dise])
@pytest.mark.parametrize("cutoff", [-0.1, np.nan, np.inf])
def test_matrix_sphere_exclusion_rejects_invalid_cutoffs(function, cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        function(_small_matrix(), cutoff)


@pytest.mark.parametrize("function", [fused_leader, fused_dise])
@pytest.mark.parametrize("metric", ["tanimoto", "aap"])
@pytest.mark.parametrize("cutoff", [-0.1, 1.1, np.nan])
def test_fused_sphere_exclusion_rejects_invalid_cutoffs(function, metric, cutoff):
    x = [Chem.MolFromSmiles("CC")] if metric == "aap" else np.asarray([[1]], dtype=np.uint32)
    with pytest.raises(ValueError, match="cutoff"):
        function(x, cutoff, metric=metric)


@pytest.mark.parametrize(
    "matrix, message",
    [
        (np.zeros(3), "square 2D"),
        (np.zeros((2, 3)), "square 2D"),
        (np.zeros((2, 2), dtype=np.float16), "float32 or float64"),
        (np.zeros((2, 2), dtype=np.int32), "float32 or float64"),
    ],
)
def test_matrix_shape_and_dtype_are_validated(matrix, message):
    with pytest.raises(ValueError, match=message):
        leader(matrix, 0.2)


@pytest.mark.parametrize("first_picks", [(0, 0), (-1,), (6,)])
def test_first_picks_are_validated(first_picks):
    with pytest.raises(ValueError, match="first_picks"):
        leader(_small_matrix(), 0.2, first_picks=first_picks)


def test_first_picks_must_be_integers():
    with pytest.raises(TypeError, match="first_picks"):
        leader(_small_matrix(), 0.2, first_picks=(1.5,))


@pytest.mark.parametrize("pick_size", [-1, 7])
def test_pick_size_is_validated(pick_size):
    with pytest.raises(ValueError, match="pick_size"):
        leader(_small_matrix(), 0.2, pick_size=pick_size)


def test_fingerprint_input_is_validated():
    with pytest.raises(ValueError, match="2D"):
        fused_leader(np.asarray([1, 2], dtype=np.uint32), 0.2)
    with pytest.raises(ValueError, match="dtype"):
        fused_leader(np.asarray([[1.0]]), 0.2)
    with pytest.raises(ValueError, match="at least one fingerprint word"):
        fused_leader(np.empty((2, 0), dtype=np.uint32), 0.2)


def test_metric_output_and_assignment_are_validated():
    with pytest.raises(ValueError, match="metric must be"):
        fused_leader(np.asarray([[1]], dtype=np.uint32), 0.2, metric="euclidean")
    with pytest.raises(TypeError, match="OutputMode"):
        leader(_small_matrix(), 0.2, output="rdkit")
    with pytest.raises(ValueError, match="assignment"):
        dise(_small_matrix(), 0.2, assignment="unknown")
