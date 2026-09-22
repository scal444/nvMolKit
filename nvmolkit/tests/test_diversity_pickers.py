# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
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
