# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental shared GPU tree must implement the stated leaf-owner schedule."""

import numpy as np
import pytest
import torch

from _bitbirch_batched_reference import BatchedBitBirch
from _bitbirch_beam_reference import BeamBitBirch
from nvmolkit.clustering import bitbirch, bitbirch_shared


def reference(packed, threshold, branching, batch, policy="ordered-leaf", ordered_prefix_size=0):
    bytes_ = packed.view(np.uint8).reshape(len(packed), packed.shape[1] * 4)
    commit_policy = "filtered" if policy == "filtered-group" else "leaf_ordered"
    tree = BatchedBitBirch(
        threshold, branching, batch, commit_policy=commit_policy, ordered_prefix_size=ordered_prefix_size
    ).fit(bytes_)
    tree.audit()
    centroids = np.stack([entry.packed for entry in tree.clusters()]).view(np.uint32).reshape(-1, packed.shape[1])
    return tree.labels(), centroids


@pytest.mark.parametrize("branching,batch,words", [(3, 7, 1), (7, 32, 3), (254, 32, 32)])
@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.7, 1.0])
@pytest.mark.parametrize("policy", ["ordered-leaf", "filtered-group"])
def test_matches_independent_leaf_owner_reference(branching, batch, words, threshold, policy):
    rng = np.random.default_rng(740 + words)
    packed = rng.integers(0, 2**32, size=(97, words), dtype=np.uint32)
    packed[30:40] = packed[:10]
    expected, expected_centroids = reference(packed, threshold, branching, batch, policy)
    actual, centroids = bitbirch_shared(
        packed,
        threshold,
        branching_factor=branching,
        insertion_batch_size=batch,
        insertion_policy=policy,
        return_centroids=True,
    )
    np.testing.assert_array_equal(actual.numpy(), expected)
    np.testing.assert_array_equal(centroids.numpy(), expected_centroids)


def test_batch_one_matches_native_serial_with_splits():
    packed = np.random.default_rng(51).integers(0, 2**32, size=(111, 2), dtype=np.uint32)
    expected = bitbirch(packed, 0.4, branching_factor=3, num_partitions=1).numpy()
    actual = bitbirch_shared(packed, 0.4, branching_factor=3, insertion_batch_size=1).numpy()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("words", [128, 129, 256])
def test_routing_precision_fast_path_boundary_and_wide_fallback(words):
    packed = np.random.default_rng(202 + words).integers(0, 2**32, size=(43, words), dtype=np.uint32)
    packed[30:40] = packed[:10]
    expected, expected_centroids = reference(packed, 0.4, 3, 16, "filtered-group")
    labels, centroids = bitbirch_shared(
        packed,
        0.4,
        branching_factor=3,
        insertion_batch_size=16,
        insertion_policy="filtered-group",
        return_centroids=True,
    )
    np.testing.assert_array_equal(labels.numpy(), expected)
    np.testing.assert_array_equal(centroids.numpy(), expected_centroids)


def test_split_fallback_above_cooperative_block_width():
    packed = np.random.default_rng(82).integers(0, 2**32, size=(529, 1), dtype=np.uint32)
    expected, expected_centroids = reference(packed, 1.0, 257, 512)
    labels, centroids = bitbirch_shared(
        packed,
        1.0,
        branching_factor=257,
        insertion_batch_size=512,
        return_centroids=True,
        insertion_policy="filtered-group",
    )
    np.testing.assert_array_equal(labels.numpy(), expected)
    np.testing.assert_array_equal(centroids.numpy(), expected_centroids)


@pytest.mark.parametrize("branching", [254, 255])
def test_full_1024bit_split_cache_matches_uncached_reference(branching):
    packed = np.random.default_rng(991).integers(0, 2**32, size=(529, 32), dtype=np.uint32)
    packed[510:520] = packed[:10]
    expected, expected_centroids = reference(packed, 0.55, branching, 512, "filtered-group")
    labels, centroids = bitbirch_shared(
        packed,
        0.55,
        branching_factor=branching,
        insertion_batch_size=512,
        insertion_policy="filtered-group",
        return_centroids=True,
    )
    np.testing.assert_array_equal(labels.numpy(), expected)
    np.testing.assert_array_equal(centroids.numpy(), expected_centroids)


@pytest.mark.parametrize("policy", ["ordered-leaf", "filtered-group"])
def test_concurrent_leaf_allocation_is_schedule_independent(policy):
    packed = np.random.default_rng(73).integers(0, 2**32, size=(503, 3), dtype=np.uint32)
    expected, _ = reference(packed, 0.55, 3, 128, policy)
    for _ in range(5):
        labels = bitbirch_shared(
            packed, 0.55, branching_factor=3, insertion_batch_size=128, insertion_policy=policy
        ).numpy()
        np.testing.assert_array_equal(labels, expected)


def test_empty_zeros_duplicates_and_uint32_count_dispatch():
    labels, centroids = bitbirch_shared(np.zeros((0, 2), dtype=np.uint32), 0.5, return_centroids=True)
    assert labels.numpy().shape == (0,)
    assert centroids.numpy().shape == (0, 2)
    packed = np.zeros((65537, 1), dtype=np.uint32)
    labels, centroids = bitbirch_shared(packed, 1.0, return_centroids=True)
    np.testing.assert_array_equal(labels.numpy(), np.zeros(len(packed), dtype=np.int32))
    np.testing.assert_array_equal(centroids.numpy(), np.zeros((1, 1), dtype=np.uint32))


def test_explicit_stream_orders_input_clustering_and_consumption():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        packed = torch.tensor([[1], [2], [4], [8], [1], [2], [4], [8]], dtype=torch.int32, device="cuda")
        labels = bitbirch_shared(packed, 1.0, branching_factor=3, insertion_batch_size=4, stream=stream)
        copied = labels.torch().clone()
    stream.synchronize()
    expected, _ = reference(packed.cpu().numpy().view(np.uint32), 1.0, 3, 4)
    np.testing.assert_array_equal(copied.cpu().numpy(), expected)


def test_sanitizer_smoke_multiple_owners_and_cascading_splits():
    packed = np.random.default_rng(331).integers(0, 2**32, size=(43, 2), dtype=np.uint32)
    packed[29:36] = packed[:7]
    for policy in ("ordered-leaf", "filtered-group"):
        expected, expected_centroids = reference(packed, 0.4, 3, 16, policy)
        labels, centroids = bitbirch_shared(
            packed, 0.4, branching_factor=3, insertion_batch_size=16, insertion_policy=policy, return_centroids=True
        )
        np.testing.assert_array_equal(labels.numpy(), expected)
        np.testing.assert_array_equal(centroids.numpy(), expected_centroids)


@pytest.mark.parametrize("prefix", [0, 1, 33, 97, 200])
def test_ordered_prefix_matches_reference_including_partial_batches(prefix):
    packed = np.random.default_rng(117).integers(0, 2**32, (97, 3), dtype=np.uint32)
    expected, expected_centroids = reference(packed, 0.4, 3, 32, "filtered-group", prefix)
    labels, centroids = bitbirch_shared(
        packed,
        0.4,
        branching_factor=3,
        insertion_batch_size=32,
        insertion_policy="filtered-group",
        ordered_prefix_size=prefix,
        return_centroids=True,
    )
    np.testing.assert_array_equal(labels.numpy(), expected)
    np.testing.assert_array_equal(centroids.numpy(), expected_centroids)


@pytest.mark.parametrize("branching,words", [(3, 2), (7, 3), (254, 32), (257, 1)])
@pytest.mark.parametrize("threshold", [0.3, 0.55, 1.0])
def test_two_path_routing_matches_independent_reference(branching, words, threshold):
    packed = np.random.default_rng(991).integers(0, 2**32, (529, words), dtype=np.uint32)
    packed[510:520] = packed[:10]
    tree = BeamBitBirch(threshold, branching, 128, ordered_prefix_size=33).fit(
        packed.view(np.uint8).reshape(len(packed), -1)
    )
    tree.audit()
    expected_centroids = np.stack([entry.packed for entry in tree.clusters()]).view(np.uint32).reshape(-1, words)
    for _ in range(2):
        labels, centroids = bitbirch_shared(
            packed,
            threshold,
            branching_factor=branching,
            insertion_batch_size=128,
            insertion_policy="filtered-group",
            ordered_prefix_size=33,
            routing_width=2,
            return_centroids=True,
        )
        np.testing.assert_array_equal(labels.numpy(), tree.labels())
        np.testing.assert_array_equal(centroids.numpy(), expected_centroids)


def test_two_path_sanitizer_smoke():
    packed = np.random.default_rng(743).integers(0, 2**32, (43, 2), dtype=np.uint32)
    packed[30:36] = packed[:6]
    tree = BeamBitBirch(0.4, 3, 16, ordered_prefix_size=5).fit(packed.view(np.uint8).reshape(len(packed), -1))
    tree.audit()
    labels = bitbirch_shared(
        packed,
        0.4,
        branching_factor=3,
        insertion_batch_size=16,
        insertion_policy="filtered-group",
        ordered_prefix_size=5,
        routing_width=2,
    )
    np.testing.assert_array_equal(labels.numpy(), tree.labels())


def test_host_output_sanitizer_smoke():
    packed = np.random.default_rng(857).integers(0, 2**32, (43, 2), dtype=np.uint32)
    packed[30:36] = packed[:6]
    tree = BeamBitBirch(0.4, 3, 16, ordered_prefix_size=5).fit(packed.view(np.uint8).reshape(len(packed), -1))
    tree.audit()
    labels = bitbirch_shared(
        packed,
        0.4,
        branching_factor=3,
        insertion_batch_size=16,
        insertion_policy="filtered-group",
        ordered_prefix_size=5,
        routing_width=2,
        host_input=True,
        host_output=True,
    )
    assert isinstance(labels, np.ndarray)
    np.testing.assert_array_equal(labels, tree.labels())


@pytest.mark.parametrize("words,routes", [(1, 1), (1, 2), (32, 2)])
def test_cpu_backed_summary_rotation_preserves_partition(words, routes):
    rng = np.random.default_rng(1591)
    unique = rng.integers(0, 2**32, (5000, words), dtype=np.uint32)
    packed = np.concatenate([unique, unique[rng.permutation(len(unique))]])
    options = dict(
        branching_factor=7,
        insertion_batch_size=512,
        insertion_policy="filtered-group",
        routing_width=routes,
        return_centroids=True,
    )
    expected, expected_centroids = bitbirch_shared(packed, 1.0, **options)
    # 4096 uint16 summaries per page: duplicates force more than one page,
    # in addition to internal-node summaries. One and two slots force eviction.
    page_bytes = 4096 * words * 32 * 2
    for pages in (1, 2):
        labels, centroids = bitbirch_shared(packed, 1.0, summary_cache_bytes=pages * page_bytes, **options)
        np.testing.assert_array_equal(labels.numpy(), expected.numpy())
        np.testing.assert_array_equal(centroids.numpy(), expected_centroids.numpy())


@pytest.mark.parametrize("words", [1, 32])
def test_cpu_backed_singleton_rotation_preserves_partition(words):
    rng = np.random.default_rng(2903)
    packed = rng.integers(0, 2**32, (10000, words), dtype=np.uint32)
    options = dict(
        branching_factor=7,
        insertion_batch_size=512,
        insertion_policy="filtered-group",
        routing_width=2,
        return_centroids=True,
    )
    expected, expected_centroids = bitbirch_shared(packed, 1.0, **options)
    page_bytes = 4096 * packed.shape[1] * np.dtype(np.uint32).itemsize
    for pages in (1, 2):
        labels, centroids = bitbirch_shared(
            packed,
            1.0,
            host_input=True,
            fingerprint_cache_bytes=pages * page_bytes,
            **options,
        )
        np.testing.assert_array_equal(labels.numpy(), expected.numpy())
        np.testing.assert_array_equal(centroids.numpy(), expected_centroids.numpy())


@pytest.mark.parametrize("policy,routes", [("ordered-leaf", 1), ("filtered-group", 1), ("filtered-group", 2)])
@pytest.mark.parametrize("prefix", [0, 33])
def test_host_input_tiles_preserve_labels_and_centroids(policy, routes, prefix, tmp_path):
    packed = np.random.default_rng(871).integers(0, 2**32, (529, 3), dtype=np.uint32)
    packed[250:300] = packed[:50]
    packed[480:520] = packed[300:340]
    path = tmp_path / "fingerprints.npy"
    np.save(path, packed)
    mapped = np.load(path, mmap_mode="r")
    options = dict(
        branching_factor=3,
        insertion_batch_size=32,
        insertion_policy=policy,
        ordered_prefix_size=prefix,
        routing_width=routes,
        return_centroids=True,
    )
    expected, expected_centroids = bitbirch_shared(packed, 0.4, **options)
    for cache in (0, 4096 * 3 * 32 * 2):
        labels, centroids = bitbirch_shared(mapped, 0.4, host_input=True, summary_cache_bytes=cache, **options)
        np.testing.assert_array_equal(labels.numpy(), expected.numpy())
        np.testing.assert_array_equal(centroids.numpy(), expected_centroids.numpy())


def test_host_input_empty_and_uint32_summary_dispatch():
    labels, centroids = bitbirch_shared(np.zeros((0, 1), dtype=np.uint32), 0.5, host_input=True, return_centroids=True)
    assert labels.numpy().shape == (0,)
    assert centroids.numpy().shape == (0, 1)
    packed = np.zeros((65537, 1), dtype=np.uint32)
    labels = bitbirch_shared(packed, 1.0, host_input=True, insertion_policy="filtered-group", routing_width=2)
    np.testing.assert_array_equal(labels.numpy(), np.zeros(len(packed), dtype=np.int32))


def test_host_output_preserves_labels_with_host_input_cache_and_centroids():
    rng = np.random.default_rng(421)
    packed = rng.integers(0, 2**32, (529, 3), dtype=np.uint32)
    packed[430:500] = packed[:70]
    options = dict(
        branching_factor=7,
        insertion_batch_size=32,
        insertion_policy="filtered-group",
        ordered_prefix_size=33,
        routing_width=2,
        return_centroids=True,
    )
    expected, expected_centroids = bitbirch_shared(packed, 0.4, **options)
    device_input_labels, device_input_centroids = bitbirch_shared(packed, 0.4, host_output=True, **options)
    assert isinstance(device_input_labels, np.ndarray)
    np.testing.assert_array_equal(device_input_labels, expected.numpy())
    np.testing.assert_array_equal(device_input_centroids.numpy(), expected_centroids.numpy())

    page_bytes = 4096 * packed.shape[1] * 32 * 2
    labels, centroids = bitbirch_shared(
        packed,
        0.4,
        host_input=True,
        host_output=True,
        summary_cache_bytes=page_bytes,
        **options,
    )
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int32
    np.testing.assert_array_equal(labels, expected.numpy())
    np.testing.assert_array_equal(centroids.numpy(), expected_centroids.numpy())

    empty = bitbirch_shared(np.zeros((0, 1), dtype=np.uint32), 0.5, host_input=True, host_output=True)
    assert isinstance(empty, np.ndarray)
    assert empty.shape == (0,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": float("nan")},
        {"threshold": -1},
        {"threshold": 0.5, "branching_factor": 2},
        {"threshold": 0.5, "ordered_prefix_size": -1},
        {"threshold": 0.5, "routing_width": 0},
        {"threshold": 0.5, "routing_width": 3, "insertion_policy": "filtered-group"},
        {"threshold": 0.5, "routing_width": 2},
        {"threshold": 0.5, "summary_cache_bytes": -1},
        {"threshold": 0.5, "summary_cache_bytes": 1},
        {"threshold": 0.5, "fingerprint_cache_bytes": -1},
        {"threshold": 0.5, "fingerprint_cache_bytes": 4096},
        {"threshold": 0.5, "fingerprint_cache_bytes": 1, "host_input": True},
        {"threshold": 0.5, "insertion_batch_size": 0},
    ],
)
def test_invalid_options_rejected(kwargs):
    with pytest.raises(ValueError):
        bitbirch_shared(np.zeros((1, 1), dtype=np.uint32), **kwargs)
