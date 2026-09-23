# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness checks for the experimental insertion schedule."""

import numpy as np
import pytest

from _bitbirch_batched_reference import BatchedBitBirch, Entry, centroid, isim


def test_joint_validation_and_nonmonotone_prefix():
    tree = BatchedBitBirch(threshold=0.5)
    tree.bits = np.asarray([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    sums = np.ones(4, dtype=np.uint64)
    entry = Entry(10, 1, sums, centroid(sums, 1), [10])
    assert tree._try_group(entry, [0, 1]) == [1]
    assert entry.members == [10, 0]
    assert isim(entry.sums, entry.count) == 0.5
    assert tree.stats["failed_groups"] == 1

    tree = BatchedBitBirch(threshold=0.6)
    tree.bits = np.asarray([[0, 1], [1, 1]], dtype=np.uint8)
    sums = np.ones(2, dtype=np.uint64)
    entry = Entry(10, 1, sums, centroid(sums, 1), [10])
    assert isim(sums + tree.bits[0], 2) < tree.threshold
    assert tree._try_group(entry, [0, 1]) == []
    assert tree.stats["bulk_committed"] == 2


@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.7, 1.0])
@pytest.mark.parametrize("branching_factor", [3, 7])
@pytest.mark.parametrize("policy", ["grouped", "filtered", "ordered", "leaf_ordered"])
def test_audits_splits_and_service_order(threshold, branching_factor, policy):
    rng = np.random.default_rng(18)
    bases = rng.integers(0, 256, (35, 4), dtype=np.uint8)
    data = np.concatenate([np.zeros((4, 4), dtype=np.uint8), bases, bases, bases[::-1]])
    for batch in (1, 16, 128):
        forward = BatchedBitBirch(threshold, branching_factor, batch, route_tile=7, commit_policy=policy).fit(data)
        reverse = BatchedBitBirch(
            threshold, branching_factor, batch, route_tile=31, reverse_service=True, commit_policy=policy
        ).fit(data)
        assert forward.audit()["audited_molecules"] == len(data)
        reverse.audit()
        np.testing.assert_array_equal(forward.labels(), reverse.labels())


def test_batch_one_matches_independent_serial_reference():
    from _bitbirch_reference import serial_tree_reference

    for seed in range(3):
        rng = np.random.default_rng(seed)
        data = rng.integers(0, 256, (60, 2), dtype=np.uint8)
        for threshold in (0.25, 0.6, 1.0):
            expected, _, _ = serial_tree_reference(np.unpackbits(data, axis=1), threshold, branching_factor=3)
            tree = BatchedBitBirch(threshold, branching_factor=3, batch_size=1).fit(data)
            tree.audit()
            np.testing.assert_array_equal(tree.labels(), expected)


def test_empty_zero_and_hot_cluster():
    empty = BatchedBitBirch().fit(np.empty((0, 4), dtype=np.uint8))
    assert empty.labels().size == 0
    empty.audit()
    for byte in (0, 255):
        tree = BatchedBitBirch(threshold=1.0, batch_size=32).fit(np.full((300, 4), byte, dtype=np.uint8))
        tree.audit()
        assert len(tree.clusters()) == 1
        assert tree.clusters()[0].count == 300
        assert tree.stats["bulk_committed"] == 268


def test_snapshot_filter_does_not_replace_joint_validation():
    tree = BatchedBitBirch(threshold=0.5, commit_policy="filtered")
    tree.bits = np.asarray([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    sums = np.ones(4, dtype=np.uint64)
    entry = Entry(10, 1, sums, centroid(sums, 1), [10])
    assert tree._try_group(entry, [0, 1]) == [1]
    assert tree.stats["snapshot_rejected"] == 0
    assert tree.stats["failed_groups"] == 1
