# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools

import numpy as np
import pytest
from _bitbirch_reference import (
    isim_tanimoto,
    majority_centroid,
    partitioned_tree_reference,
    serial_leaf_reference,
    serial_tree_reference,
)


def _direct_isim(bits: np.ndarray) -> float:
    common_pairs = 0
    mismatches = 0
    for lhs, rhs in itertools.combinations(bits, 2):
        common_pairs += int(np.count_nonzero(lhs & rhs))
        mismatches += int(np.count_nonzero(lhs ^ rhs))
    denominator = common_pairs + mismatches
    return common_pairs / denominator if denominator else 1.0


@pytest.mark.parametrize("count", range(2, 7))
def test_isim_matches_exhaustive_pair_counts(count):
    rng = np.random.default_rng(count)
    for _ in range(50):
        bits = rng.integers(0, 2, size=(count, 17), dtype=np.uint8)
        assert isim_tanimoto(bits.sum(axis=0), count) == pytest.approx(_direct_isim(bits))


def test_isim_empty_singleton_and_all_zero_are_one():
    assert isim_tanimoto(np.zeros(8, dtype=np.uint64), 0) == 1.0
    assert isim_tanimoto(np.array([1, 0, 1], dtype=np.uint64), 1) == 1.0
    assert isim_tanimoto(np.zeros(8, dtype=np.uint64), 5) == 1.0


def test_majority_centroid_sets_exact_ties():
    np.testing.assert_array_equal(
        majority_centroid(np.array([0, 1, 2, 3, 4]), 4),
        np.array([0, 0, 1, 1, 1], dtype=np.uint8),
    )


def test_serial_leaf_reference_threshold_boundary_and_labels():
    bits = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0]], dtype=np.uint8)
    labels, features = serial_leaf_reference(bits, threshold=0.5)
    np.testing.assert_array_equal(labels, np.array([0, 0, 1, 1], dtype=np.int32))
    assert [feature.members for feature in features] == [[0, 1], [2, 3]]


def test_serial_leaf_reference_is_deterministic_on_centroid_tie():
    bits = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    labels, _ = serial_leaf_reference(bits, threshold=0.9)
    np.testing.assert_array_equal(labels, np.array([0, 1, 2], dtype=np.int32))


def _assert_tree_capacity(node, branching_factor, *, is_root=True):
    assert 1 <= len(node.entries) <= branching_factor
    if branching_factor >= 3 and not is_root:
        assert len(node.entries) >= 2
    if not node.leaf:
        for entry in node.entries:
            assert entry.child is not None
            assert entry.child.parent is node
            _assert_tree_capacity(entry.child, branching_factor, is_root=False)


@pytest.mark.parametrize("branching_factor", [2, 3, 4, 7])
def test_serial_tree_forces_and_propagates_splits(branching_factor):
    bits = np.eye(24, dtype=np.uint8)
    labels, features, root = serial_tree_reference(bits, threshold=0.9, branching_factor=branching_factor)
    np.testing.assert_array_equal(labels, np.arange(24, dtype=np.int32))
    assert len(features) == 24
    assert root is not None
    assert not root.leaf
    _assert_tree_capacity(root, branching_factor)


def test_serial_tree_matches_leaf_reference_without_splits():
    rng = np.random.default_rng(91)
    bits = rng.integers(0, 2, size=(30, 37), dtype=np.uint8)
    leaf_labels, leaf_features = serial_leaf_reference(bits, threshold=0.35, tolerance=0.05)
    tree_labels, tree_features, root = serial_tree_reference(
        bits,
        threshold=0.35,
        branching_factor=30,
        tolerance=0.05,
    )
    np.testing.assert_array_equal(tree_labels, leaf_labels)
    assert [feature.members for feature in tree_features] == [feature.members for feature in leaf_features]
    assert root is not None and root.leaf


def test_split_capacity_balancing_prevents_single_entry_child():
    bits = np.array(
        [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1]],
        dtype=np.uint8,
    )
    _, _, root = serial_tree_reference(bits, threshold=1.0, branching_factor=3)
    assert root is not None and not root.leaf
    assert sorted(len(entry.child.entries) for entry in root.entries) == [2, 2]
    _assert_tree_capacity(root, 3)


def test_partitioned_reference_merges_leaf_summaries_in_first_member_order():
    bits = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.uint8)
    labels, features, _ = partitioned_tree_reference(bits, 0.5, branching_factor=3, num_partitions=2)
    np.testing.assert_array_equal(labels, np.array([0, 1, 0, 1], dtype=np.int32))
    assert [feature.members for feature in features] == [[0, 2], [1, 3]]


def test_serial_tree_assigns_every_input_once_after_merge_and_splits():
    rng = np.random.default_rng(123)
    bases = rng.integers(0, 2, size=(8, 65), dtype=np.uint8)
    bits = np.repeat(bases, 6, axis=0)
    bits[1::6, 0] ^= 1
    labels, features, _ = serial_tree_reference(bits, threshold=0.7, branching_factor=3)
    assert np.all(labels >= 0)
    assert sorted(member for feature in features for member in feature.members) == list(range(len(bits)))
    assert len(np.unique(labels)) == len(features)


def test_serial_tree_empty_input():
    labels, features, root = serial_tree_reference(np.empty((0, 32), dtype=np.uint8), 0.5, branching_factor=2)
    assert labels.shape == (0,)
    assert features == []
    assert root is None


@pytest.mark.parametrize(
    "bits,threshold,tolerance,error",
    [
        (np.zeros(4), 0.5, None, "2D"),
        (np.array([[0, 2]]), 0.5, None, "zero and one"),
        (np.array([[0, 1]]), -0.1, None, r"\[0, 1\]"),
        (np.array([[0, 1]]), 0.5, -0.1, "nonnegative"),
    ],
)
def test_serial_leaf_reference_validation(bits, threshold, tolerance, error):
    with pytest.raises(ValueError, match=error):
        serial_leaf_reference(bits, threshold, tolerance=tolerance)
