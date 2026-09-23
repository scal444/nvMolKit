# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from _bitbirch_batched_reference import BatchedBitBirch
from _bitbirch_reference import partitioned_tree_reference, serial_tree_reference
from rdkit.Chem import rdFingerprintGenerator

from nvmolkit.clustering import bitbirch, bitbirch_shared
from nvmolkit.fingerprints import MorganFingerprintGenerator, unpack_fingerprint


def test_morgan_fingerprints_to_shared_tree_matches_independent_schedule(size_limited_mols):
    mols = size_limited_mols[:80]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    bits = np.asarray([generator.GetFingerprint(mol).ToList() for mol in mols], dtype=np.uint8)
    packed = np.packbits(bits, axis=1, bitorder="little")
    reference = BatchedBitBirch(0.25, 7, 32, commit_policy="filtered", ordered_prefix_size=17).fit(packed)
    reference.audit()
    fingerprints = MorganFingerprintGenerator(radius=2, fpSize=1024).GetFingerprints(mols, num_threads=1)
    labels, centroids = bitbirch_shared(
        fingerprints,
        0.25,
        branching_factor=7,
        insertion_batch_size=32,
        insertion_policy="filtered-group",
        ordered_prefix_size=17,
        return_centroids=True,
    )
    np.testing.assert_array_equal(labels.numpy(), reference.labels())
    np.testing.assert_array_equal(
        centroids.numpy(), np.stack([entry.packed for entry in reference.clusters()]).view(np.uint32)
    )


def test_morgan_fingerprint_to_partitioned_bitbirch_matches_reference(size_limited_mols):
    mols = size_limited_mols[:80]
    fingerprints = MorganFingerprintGenerator(radius=3, fpSize=1024).GetFingerprints(mols, num_threads=1)
    bits = unpack_fingerprint(fingerprints.torch()).cpu().numpy().astype(np.uint8)
    expected_labels, expected_features, _ = partitioned_tree_reference(
        bits,
        0.55,
        branching_factor=7,
        num_partitions=5,
    )

    labels, centroids = bitbirch(
        fingerprints,
        threshold=0.55,
        branching_factor=7,
        num_partitions=5,
        return_centroids=True,
    )
    expected_centroids = np.packbits(
        np.stack([feature.centroid for feature in expected_features]), axis=1, bitorder="little"
    ).view(np.uint32)
    np.testing.assert_array_equal(labels.numpy(), expected_labels)
    np.testing.assert_array_equal(centroids.numpy(), expected_centroids)
    assert min(feature.isim for feature in expected_features) >= 0.55


def test_fingerprint_generation_and_clustering_chain_on_explicit_stream(size_limited_mols):
    base = size_limited_mols[:32]
    mols = base + base
    stream = torch.cuda.Stream()
    fingerprints = MorganFingerprintGenerator(radius=3, fpSize=512).GetFingerprints(
        mols,
        num_threads=1,
        stream=stream,
    )
    labels = bitbirch(
        fingerprints,
        threshold=1.0,
        branching_factor=4,
        num_partitions=4,
        stream=stream,
    ).torch()
    stream.synchronize()

    bits = unpack_fingerprint(fingerprints.torch()).cpu().numpy().astype(np.uint8)
    expected, _, _ = partitioned_tree_reference(bits, 1.0, branching_factor=4, num_partitions=4)
    np.testing.assert_array_equal(labels.cpu().numpy(), expected)


def test_host_rdkit_and_device_nvmolkit_fingerprints_cluster_identically(size_limited_mols):
    mols = size_limited_mols[:40]
    rdkit_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
    rdkit_bits = np.asarray([rdkit_generator.GetFingerprint(mol).ToList() for mol in mols], dtype=np.uint8)
    rdkit_packed = np.packbits(rdkit_bits, axis=1, bitorder="little").view(np.uint32)
    expected, _, _ = serial_tree_reference(rdkit_bits, 0.5, branching_factor=7)

    host_labels = bitbirch(rdkit_packed, threshold=0.5, branching_factor=7, num_partitions=1).numpy()
    device_fingerprints = MorganFingerprintGenerator(radius=2, fpSize=256).GetFingerprints(mols, num_threads=1)
    device_labels = bitbirch(device_fingerprints, threshold=0.5, branching_factor=7, num_partitions=1).numpy()

    np.testing.assert_array_equal(host_labels, expected)
    np.testing.assert_array_equal(device_labels, expected)
