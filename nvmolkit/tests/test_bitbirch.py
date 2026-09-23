# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and scaling tests for concurrent BitBIRCH insertion."""

import numpy as np
import pytest
import torch
from _bitbirch_reference import BatchedBitBirch

from nvmolkit.clustering import bitbirch


def reference(packed, threshold, branching_factor, batch_size):
    bytes_ = packed.view(np.uint8).reshape(len(packed), packed.shape[1] * 4)
    tree = BatchedBitBirch(threshold, branching_factor, batch_size).fit(bytes_)
    tree.audit()
    centroids = np.stack([entry.packed for entry in tree.clusters()]).view(np.uint32).reshape(-1, packed.shape[1])
    return tree.labels(), centroids


@pytest.mark.parametrize("branching_factor,batch_size,words", [(3, 7, 1), (7, 32, 3), (254, 32, 32)])
@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.7, 1.0])
def test_matches_independent_reference(branching_factor, batch_size, words, threshold):
    rng = np.random.default_rng(740 + words)
    packed = rng.integers(0, 2**32, size=(97, words), dtype=np.uint32)
    packed[30:40] = packed[:10]
    expected, expected_centroids = reference(packed, threshold, branching_factor, batch_size)
    actual, centroids = bitbirch(
        packed,
        threshold,
        branching_factor=branching_factor,
        batch_size=batch_size,
        return_centroids=True,
    )
    np.testing.assert_array_equal(actual.numpy(), expected)
    np.testing.assert_array_equal(centroids.numpy(), expected_centroids)


@pytest.mark.parametrize("words", [128, 129, 256])
def test_routing_precision_fast_path_boundary_and_wide_fallback(words):
    packed = np.random.default_rng(202 + words).integers(0, 2**32, size=(43, words), dtype=np.uint32)
    packed[30:40] = packed[:10]
    expected, expected_centroids = reference(packed, 0.4, 3, 16)
    labels, centroids = bitbirch(packed, 0.4, branching_factor=3, batch_size=16, return_centroids=True)
    np.testing.assert_array_equal(labels.numpy(), expected)
    np.testing.assert_array_equal(centroids.numpy(), expected_centroids)


def test_split_fallback_above_cooperative_block_width():
    packed = np.random.default_rng(82).integers(0, 2**32, size=(529, 1), dtype=np.uint32)
    expected, expected_centroids = reference(packed, 1.0, 257, 512)
    labels, centroids = bitbirch(packed, 1.0, branching_factor=257, batch_size=512, return_centroids=True)
    np.testing.assert_array_equal(labels.numpy(), expected)
    np.testing.assert_array_equal(centroids.numpy(), expected_centroids)


@pytest.mark.parametrize("branching_factor", [254, 255])
def test_full_1024bit_split_cache_matches_reference(branching_factor):
    packed = np.random.default_rng(991).integers(0, 2**32, size=(529, 32), dtype=np.uint32)
    packed[510:520] = packed[:10]
    expected, expected_centroids = reference(packed, 0.55, branching_factor, 512)
    labels, centroids = bitbirch(
        packed, 0.55, branching_factor=branching_factor, batch_size=512, return_centroids=True
    )
    np.testing.assert_array_equal(labels.numpy(), expected)
    np.testing.assert_array_equal(centroids.numpy(), expected_centroids)


def test_concurrent_leaf_allocation_is_schedule_independent():
    packed = np.random.default_rng(73).integers(0, 2**32, size=(503, 3), dtype=np.uint32)
    expected, _ = reference(packed, 0.55, 3, 128)
    for _ in range(5):
        labels = bitbirch(packed, 0.55, branching_factor=3, batch_size=128).numpy()
        np.testing.assert_array_equal(labels, expected)


def test_empty_zeros_duplicates_and_uint32_count_dispatch():
    labels, centroids = bitbirch(np.zeros((0, 2), dtype=np.uint32), 0.5, return_centroids=True)
    assert labels.numpy().shape == (0,)
    assert centroids.numpy().shape == (0, 2)
    packed = np.zeros((65537, 1), dtype=np.uint32)
    labels, centroids = bitbirch(packed, 1.0, return_centroids=True)
    np.testing.assert_array_equal(labels.numpy(), np.zeros(len(packed), dtype=np.int32))
    np.testing.assert_array_equal(centroids.numpy(), np.zeros((1, 1), dtype=np.uint32))


def test_explicit_stream_orders_input_clustering_and_consumption():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        packed = torch.tensor([[1], [2], [4], [8], [1], [2], [4], [8]], dtype=torch.int32, device="cuda")
        labels = bitbirch(packed, 1.0, branching_factor=3, batch_size=4, stream=stream)
        copied = labels.torch().clone()
    stream.synchronize()
    expected, _ = reference(packed.cpu().numpy().view(np.uint32), 1.0, 3, 4)
    np.testing.assert_array_equal(copied.cpu().numpy(), expected)


def test_multiple_owners_and_cascading_splits():
    packed = np.random.default_rng(331).integers(0, 2**32, size=(43, 2), dtype=np.uint32)
    packed[29:36] = packed[:7]
    expected, expected_centroids = reference(packed, 0.4, 3, 16)
    labels, centroids = bitbirch(packed, 0.4, branching_factor=3, batch_size=16, return_centroids=True)
    np.testing.assert_array_equal(labels.numpy(), expected)
    np.testing.assert_array_equal(centroids.numpy(), expected_centroids)


@pytest.mark.parametrize("words", [1, 32])
def test_cpu_backed_summary_rotation_preserves_result(words):
    rng = np.random.default_rng(1591)
    unique = rng.integers(0, 2**32, (5000, words), dtype=np.uint32)
    packed = np.concatenate([unique, unique[rng.permutation(len(unique))]])
    options = {"branching_factor": 7, "batch_size": 512, "return_centroids": True}
    expected, expected_centroids = bitbirch(packed, 1.0, **options)
    page_bytes = 4096 * words * 32 * 2
    for pages in (1, 2):
        labels, centroids = bitbirch(packed, 1.0, summary_cache_bytes=pages * page_bytes, **options)
        np.testing.assert_array_equal(labels.numpy(), expected.numpy())
        np.testing.assert_array_equal(centroids.numpy(), expected_centroids.numpy())


@pytest.mark.parametrize("words", [1, 32])
def test_cpu_backed_singleton_rotation_preserves_result(words):
    packed = np.random.default_rng(2903).integers(0, 2**32, (10000, words), dtype=np.uint32)
    options = {"branching_factor": 7, "batch_size": 512, "return_centroids": True}
    expected, expected_centroids = bitbirch(packed, 1.0, **options)
    page_bytes = 4096 * words * np.dtype(np.uint32).itemsize
    for pages in (1, 2):
        labels, centroids = bitbirch(packed, 1.0, fingerprint_cache_bytes=pages * page_bytes, **options)
        np.testing.assert_array_equal(labels.numpy(), expected.numpy())
        np.testing.assert_array_equal(centroids.numpy(), expected_centroids.numpy())


def test_memory_mapped_input_preserves_labels_and_centroids(tmp_path):
    packed = np.random.default_rng(871).integers(0, 2**32, (529, 3), dtype=np.uint32)
    packed[250:300] = packed[:50]
    path = tmp_path / "fingerprints.npy"
    np.save(path, packed)
    mapped = np.load(path, mmap_mode="r")
    options = {"branching_factor": 3, "batch_size": 32, "return_centroids": True}
    expected, expected_centroids = bitbirch(packed, 0.4, **options)
    labels, centroids = bitbirch(mapped, 0.4, summary_cache_bytes=4096 * 3 * 32 * 2, **options)
    np.testing.assert_array_equal(labels.numpy(), expected.numpy())
    np.testing.assert_array_equal(centroids.numpy(), expected_centroids.numpy())


@pytest.mark.parametrize("device_input", [False, True])
def test_host_output_preserves_labels_and_centroids(device_input):
    packed = np.random.default_rng(421).integers(0, 2**32, (529, 3), dtype=np.uint32)
    packed[430:500] = packed[:70]
    expected, expected_centroids = bitbirch(packed, 0.4, branching_factor=7, batch_size=32, return_centroids=True)
    source = torch.from_numpy(packed.view(np.int32)).cuda() if device_input else packed
    labels, centroids = bitbirch(
        source, 0.4, branching_factor=7, batch_size=32, host_output=True, return_centroids=True
    )
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int32
    np.testing.assert_array_equal(labels, expected.numpy())
    np.testing.assert_array_equal(centroids.numpy(), expected_centroids.numpy())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": float("nan")},
        {"threshold": -1},
        {"threshold": 0.5, "branching_factor": 2},
        {"threshold": 0.5, "batch_size": 0},
        {"threshold": 0.5, "summary_cache_bytes": -1},
        {"threshold": 0.5, "summary_cache_bytes": 1},
        {"threshold": 0.5, "fingerprint_cache_bytes": -1},
        {"threshold": 0.5, "fingerprint_cache_bytes": 1},
    ],
)
def test_invalid_options_rejected(kwargs):
    with pytest.raises(ValueError):
        bitbirch(np.zeros((1, 1), dtype=np.uint32), **kwargs)
