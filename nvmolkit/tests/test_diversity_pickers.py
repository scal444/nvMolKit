# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    fused_maxmin,
    leader,
    maxmin,
)
from nvmolkit.similarity import AAPSimilarity, CosineSimilarity, TanimotoSimilarity


def _distance_matrix():
    points = np.asarray([0.0, 0.05, 0.25, 0.6, 0.65, 1.0])
    return np.abs(points[:, None] - points[None, :]).astype(np.float64)


def _lower_triangle(matrix):
    return np.asarray(
        [matrix[row, column] for row in range(1, len(matrix)) for column in range(row)],
        dtype=np.float64,
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


def test_leader_distance_matrix_matches_rdkit():
    distances = _distance_matrix()
    expected = tuple(
        rdSimDivPickers.LeaderPicker().LazyPick(lambda left, right: distances[left, right], len(distances), 0.2)
    )

    actual = leader(distances, 0.2, output=OutputMode.RDKIT)

    assert actual == expected


def test_leader_supports_directed_distance_rows():
    distances = np.asarray(
        [
            [0.0, 0.1, 0.8],
            [0.9, 0.0, 0.1],
            [0.2, 0.9, 0.0],
        ],
        dtype=np.float64,
    )

    assert leader(distances, 0.2, output=OutputMode.RDKIT) == (0, 2)


def test_maxmin_distance_matrix_matches_rdkit_seed_and_first_picks():
    distances = _distance_matrix()
    picker = rdSimDivPickers.MaxMinPicker()
    expected_seeded = tuple(picker.Pick(_lower_triangle(distances), len(distances), 4, seed=19))
    expected_first = tuple(picker.Pick(_lower_triangle(distances), len(distances), 4, firstPicks=(2,), seed=19))

    actual_seeded, _ = maxmin(distances, 4, seed=19, output=OutputMode.RDKIT)
    actual_first, _ = maxmin(distances, 4, first_picks=(2,), seed=19, output=OutputMode.RDKIT)

    assert actual_seeded == expected_seeded
    assert actual_first == expected_first


def test_maxmin_threshold_matches_rdkit():
    distances = _distance_matrix()
    expected, expected_last = rdSimDivPickers.MaxMinPicker().LazyPickWithThreshold(
        lambda left, right: distances[left, right], len(distances), len(distances), 0.2, firstPicks=(0,)
    )

    actual, actual_last = maxmin(
        distances,
        len(distances),
        first_picks=(0,),
        threshold=0.2,
        output=OutputMode.RDKIT,
    )

    assert actual == tuple(expected)
    assert actual_last == pytest.approx(expected_last)


@pytest.mark.parametrize(
    "metric_name, metric",
    [("tanimoto", TanimotoSimilarity()), ("cosine", CosineSimilarity())],
)
def test_fused_pickers_match_distance_matrix(metric_name, metric):
    fingerprints = np.asarray(
        [
            [0b0011, 0b0101],
            [0b0011, 0b0100],
            [0b1100, 0b1010],
            [0b1111, 0b1111],
            [0b0001, 0b0000],
        ],
        dtype=np.uint32,
    )
    distances = _fingerprint_distance_matrix(fingerprints, metric_name)

    assert fused_leader(fingerprints, 0.4, metric=metric, output=OutputMode.RDKIT) == leader(
        distances, 0.4, output=OutputMode.RDKIT
    )
    assert (
        fused_maxmin(fingerprints, 4, metric=metric, first_picks=(0,), output=OutputMode.RDKIT)[0]
        == maxmin(distances, 4, first_picks=(0,), output=OutputMode.RDKIT)[0]
    )


@pytest.mark.parametrize("assignment", ["first", "nearest"])
def test_fused_dise_matches_distance_matrix(assignment):
    fingerprints = np.asarray([[0b0011], [0b0010], [0b1100], [0b1110]], dtype=np.uint32)
    distances = _fingerprint_distance_matrix(fingerprints, "tanimoto")

    expected = dise(distances, 0.5, assignment=assignment, output=OutputMode.RDKIT)
    actual = fused_dise(
        fingerprints,
        0.5,
        metric=TanimotoSimilarity(),
        assignment=assignment,
        output=OutputMode.RDKIT,
    )

    assert actual == expected


def test_aap_provider_uses_standard_fused_apis():
    molecules = [Chem.MolFromSmiles(smiles) for smiles in ("CCCC", "CCCO", "CCOC")]
    metric = AAPSimilarity()

    picks = fused_leader(molecules, 0.8, metric=metric)
    clusters = fused_dise(molecules, 0.8, metric=metric)

    assert isinstance(picks, SelectionDeviceResult)
    assert picks.indices.numpy().tolist() == [0, 2]
    assert isinstance(clusters, ClusterDeviceResult)
    assert clusters.cluster_ids.numpy().tolist() == [1, 0, 0]

    seeded = fused_leader(
        molecules,
        0.8,
        metric=metric,
        pick_size=2,
        first_picks=(2,),
        output=OutputMode.RDKIT,
    )
    assert seeded == (2, 0)


def test_aap_provider_is_rejected_for_symmetric_maxmin():
    with pytest.raises(ValueError, match="directed"):
        fused_maxmin([], 1, metric=AAPSimilarity())

    with pytest.raises(ValueError, match="directed"):
        fused_butina(np.empty((0, 1), dtype=np.uint32), 0.5, metric=AAPSimilarity())


def test_butina_accepts_packed_provider_configuration():
    fingerprints = np.asarray([[0b0011], [0b0010], [0b1100]], dtype=np.uint32)

    configured = fused_butina(
        fingerprints,
        0.5,
        metric=TanimotoSimilarity(),
        output=OutputMode.RDKIT,
    )
    named = fused_butina(fingerprints, 0.5, metric="tanimoto", output=OutputMode.RDKIT)

    assert configured == named


def test_fused_picker_empty_inputs():
    fingerprints = np.empty((0, 4), dtype=np.uint32)

    assert fused_leader(fingerprints, 0.5, output=OutputMode.RDKIT) == ()
    assert fused_dise(fingerprints, 0.5, output=OutputMode.RDKIT) == ()
    with pytest.raises(ValueError, match="no larger than the input size"):
        fused_maxmin(fingerprints, 1)


@pytest.mark.parametrize("size", [1, 2, 7, 33])
@pytest.mark.parametrize("cutoff", [0.0, 0.25, 1.0])
def test_leader_random_symmetric_matrices_match_rdkit(size, cutoff):
    rng = np.random.default_rng(192 + size)
    distances = rng.random((size, size))
    distances = ((distances + distances.T) / 2).astype(np.float64)
    np.fill_diagonal(distances, 0.0)
    expected = tuple(rdSimDivPickers.LeaderPicker().LazyPick(lambda left, right: distances[left, right], size, cutoff))

    assert leader(distances, cutoff, output=OutputMode.RDKIT) == expected


@pytest.mark.parametrize("size", [2, 7, 33])
@pytest.mark.parametrize("seed", [0, 19])
def test_maxmin_random_matrices_match_rdkit(size, seed):
    rng = np.random.default_rng(731 + size)
    distances = rng.random((size, size))
    distances = ((distances + distances.T) / 2).astype(np.float64)
    np.fill_diagonal(distances, 0.0)
    # RDKit's eager Pick() requires pick_size < pool_size. Full-pool and
    # singleton behavior are covered separately below.
    pick_size = min(size - 1, 6)
    expected = tuple(rdSimDivPickers.MaxMinPicker().Pick(_lower_triangle(distances), size, pick_size, seed=seed))

    actual, _ = maxmin(distances, pick_size, seed=seed, output=OutputMode.RDKIT)

    assert actual == expected


def test_maxmin_ties_choose_lowest_index_and_threshold_is_inclusive():
    distances = np.ones((3, 3), dtype=np.float64)
    np.fill_diagonal(distances, 0.0)

    picks, last_distance = maxmin(distances, 3, first_picks=(0,), threshold=1.0, output=OutputMode.RDKIT)
    assert picks == (0,)
    assert last_distance == -1.0

    picks, _ = maxmin(distances, 2, first_picks=(0,), output=OutputMode.RDKIT)
    assert picks == (0, 1)


def test_leader_pick_limit_and_forced_pick_order():
    distances = _distance_matrix()

    assert leader(distances, 2.0, first_picks=(4, 1), pick_size=1, output=OutputMode.RDKIT) == (4, 1)
    assert leader(distances, 0.0, pick_size=2, output=OutputMode.RDKIT) == (0, 1)


def test_matrix_pickers_handle_empty_and_singleton_inputs():
    empty = np.empty((0, 0), dtype=np.float64)
    singleton = np.zeros((1, 1), dtype=np.float64)

    assert leader(empty, 0.0, output=OutputMode.RDKIT) == ()
    assert dise(empty, 0.0, output=OutputMode.RDKIT) == ()
    assert leader(singleton, 0.0, output=OutputMode.RDKIT) == (0,)
    assert maxmin(singleton, 1, seed=7, output=OutputMode.RDKIT) == ((0,), -1.0)
    assert dise(singleton, 0.0, output=OutputMode.RDKIT) == ((0,),)


def test_dise_assignment_modes_and_cluster_ordering():
    distances = np.asarray(
        [
            [0.0, 0.2, 0.8],
            [0.2, 0.0, 0.1],
            [0.8, 0.1, 0.0],
        ],
        dtype=np.float64,
    )

    assert dise(distances, 0.3, assignment="first", output=OutputMode.RDKIT) == ((0, 1), (2,))
    assert dise(distances, 0.3, assignment="nearest", output=OutputMode.RDKIT) == ((2, 1), (0,))


@pytest.mark.parametrize("metric", [TanimotoSimilarity(), CosineSimilarity()])
@pytest.mark.parametrize("cutoff", [0.0, 0.5, 1.0])
def test_fused_algorithms_match_materialized_distances_with_zero_fingerprints(metric, cutoff):
    metric_name = "tanimoto" if isinstance(metric, TanimotoSimilarity) else "cosine"
    fingerprints = np.asarray([[0], [0b0011], [0b0010], [0b1100]], dtype=np.uint32)
    distances = _fingerprint_distance_matrix(fingerprints, metric_name)

    assert fused_leader(fingerprints, cutoff, metric=metric, output=OutputMode.RDKIT) == leader(
        distances, cutoff, output=OutputMode.RDKIT
    )
    assert fused_maxmin(fingerprints, 4, metric=metric, first_picks=(0,), output=OutputMode.RDKIT) == maxmin(
        distances, 4, first_picks=(0,), output=OutputMode.RDKIT
    )
    for assignment in ("first", "nearest"):
        assert fused_dise(fingerprints, cutoff, metric=metric, assignment=assignment, output=OutputMode.RDKIT) == dise(
            distances, cutoff, assignment=assignment, output=OutputMode.RDKIT
        )


def test_cosine_zero_fingerprint_regression_has_unique_leaders():
    fingerprints = np.asarray([[0], [0b0011], [0b0010], [0b1100]], dtype=np.uint32)

    picks = fused_leader(fingerprints, 0.5, metric="cosine", output=OutputMode.RDKIT)

    assert picks == (0, 1, 3)
    assert len(picks) == len(set(picks))


def test_device_results_have_documented_shapes_dtypes_and_stream_support():
    distances = _distance_matrix()
    fingerprints = np.asarray([[0b0011], [0b0010], [0b1100]], dtype=np.uint32)
    stream = torch.cuda.Stream()

    selection = maxmin(distances, 3, first_picks=(0,), stream=stream)
    clustering = fused_dise(fingerprints, 0.5, stream=stream)
    stream.synchronize()

    assert isinstance(selection, SelectionDeviceResult)
    assert selection.indices.torch().dtype == torch.int32
    assert selection.indices.torch().shape == (3,)
    assert selection.last_distance is not None
    assert isinstance(clustering, ClusterDeviceResult)
    assert clustering.cluster_ids.torch().dtype == torch.int32
    assert clustering.cluster_ids.torch().shape == (3,)
    assert clustering.centroids.torch().dtype == torch.int32
    assert clustering.cluster_sizes.torch().dtype == torch.int64
    assert clustering.cluster_sizes.numpy().sum() == 3


@pytest.mark.parametrize("function", [leader, dise])
@pytest.mark.parametrize("cutoff", [-0.1, np.nan, np.inf])
def test_matrix_sphere_exclusion_rejects_invalid_cutoffs(function, cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        function(_distance_matrix(), cutoff)


@pytest.mark.parametrize("function", [fused_leader, fused_dise])
@pytest.mark.parametrize("cutoff", [-0.1, 1.1, np.nan, np.inf])
def test_fused_sphere_exclusion_rejects_invalid_cutoffs(function, cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        function(np.asarray([[1]], dtype=np.uint32), cutoff)


@pytest.mark.parametrize(
    "matrix, message",
    [
        (np.zeros(3, dtype=np.float64), "square 2D"),
        (np.zeros((2, 3), dtype=np.float64), "square 2D"),
        (np.zeros((2, 2), dtype=np.float32), "float64"),
    ],
)
def test_matrix_picker_input_validation(matrix, message):
    with pytest.raises(ValueError, match=message):
        leader(matrix, 0.2)


@pytest.mark.parametrize("first_picks", [(0, 0), (-1,), (6,)])
def test_forced_pick_validation_is_shared(first_picks):
    distances = _distance_matrix()

    with pytest.raises(ValueError, match="first_picks"):
        leader(distances, 0.2, first_picks=first_picks)
    with pytest.raises(ValueError, match="first_picks"):
        maxmin(distances, 2, first_picks=first_picks)


@pytest.mark.parametrize("pick_size", [-1, 7])
def test_leader_rejects_invalid_pick_size(pick_size):
    with pytest.raises(ValueError, match="pick_size"):
        leader(_distance_matrix(), 0.2, pick_size=pick_size)


@pytest.mark.parametrize("pick_size", [0, -1, 7])
def test_maxmin_rejects_invalid_pick_size(pick_size):
    with pytest.raises(ValueError, match="pick_size"):
        maxmin(_distance_matrix(), pick_size)


@pytest.mark.parametrize("threshold", [-0.1, np.nan, np.inf])
def test_matrix_maxmin_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        maxmin(_distance_matrix(), 2, threshold=threshold)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf])
def test_fused_maxmin_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        fused_maxmin(np.asarray([[1], [2]], dtype=np.uint32), 2, threshold=threshold)


def test_fused_picker_input_validation():
    with pytest.raises(ValueError, match="2D"):
        fused_leader(np.asarray([1, 2], dtype=np.uint32), 0.2)
    with pytest.raises(ValueError, match="dtype"):
        fused_leader(np.asarray([[1.0]], dtype=np.float64), 0.2)
    with pytest.raises(ValueError, match="at least one fingerprint word"):
        fused_leader(np.empty((2, 0), dtype=np.uint32), 0.2)


def test_output_and_assignment_validation():
    with pytest.raises(TypeError, match="OutputMode"):
        leader(_distance_matrix(), 0.2, output="rdkit")
    with pytest.raises(ValueError, match="assignment"):
        dise(_distance_matrix(), 0.2, assignment="unknown")
