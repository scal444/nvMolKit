# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent BitBIRCH reference used by differential tests.

This follows the publication equations and is not derived from either GPL
BitBIRCH software implementation. It includes both an unbounded-leaf reference
and a complete ordered tree with deterministic split propagation.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _validate_bits(bits: np.ndarray) -> np.ndarray:
    result = np.asarray(bits)
    if result.ndim != 2:
        raise ValueError("bits must be a 2D matrix")
    if not np.all((result == 0) | (result == 1)):
        raise ValueError("bits must contain only zero and one")
    return result.astype(np.uint8, copy=False)


def isim_tanimoto(linear_sum: np.ndarray, count: int) -> float:
    """Evaluate the publication's iSIM Jaccard--Tanimoto equation."""
    if count <= 1:
        return 1.0
    ls = np.asarray(linear_sum, dtype=np.float64)
    common_pairs = np.sum(ls * (ls - 1.0) * 0.5, dtype=np.float64)
    mismatches = np.sum(ls * (float(count) - ls), dtype=np.float64)
    denominator = common_pairs + mismatches
    return float(common_pairs / denominator) if denominator > 0.0 else 1.0


def majority_centroid(linear_sum: np.ndarray, count: int) -> np.ndarray:
    """Return floor(LS / N + 1/2), including one-valued exact ties."""
    if count <= 0:
        raise ValueError("count must be positive")
    ls = np.asarray(linear_sum, dtype=np.uint64)
    return (ls >= count // 2 + count % 2).astype(np.uint8)


def tanimoto(lhs: np.ndarray, rhs: np.ndarray) -> float:
    intersection = int(np.count_nonzero(lhs & rhs))
    union = int(np.count_nonzero(lhs | rhs))
    return intersection / union if union else 1.0


@dataclass
class BitFeature:
    count: int
    linear_sum: np.ndarray
    members: list[int]

    @classmethod
    def from_fingerprint(cls, fingerprint: np.ndarray, index: int) -> "BitFeature":
        return cls(1, fingerprint.astype(np.uint64, copy=True), [index])

    @property
    def centroid(self) -> np.ndarray:
        return majority_centroid(self.linear_sum, self.count)

    @property
    def isim(self) -> float:
        return isim_tanimoto(self.linear_sum, self.count)

    def combined_isim(self, fingerprint: np.ndarray) -> float:
        return isim_tanimoto(self.linear_sum + fingerprint, self.count + 1)

    def add(self, fingerprint: np.ndarray, index: int) -> None:
        self.count += 1
        self.linear_sum += fingerprint
        self.members.append(index)

    def merged(self, other: "BitFeature") -> "BitFeature":
        return BitFeature(
            self.count + other.count,
            self.linear_sum + other.linear_sum,
            self.members + other.members,
        )


@dataclass
class TreeEntry:
    feature: BitFeature
    child: Optional["TreeNode"] = None


@dataclass
class TreeNode:
    leaf: bool
    entries: list[TreeEntry]
    parent: Optional["TreeNode"] = None


def _summarize_node(node: TreeNode) -> BitFeature:
    if not node.entries:
        raise ValueError("cannot summarize an empty node")
    summary = node.entries[0].feature
    for entry in node.entries[1:]:
        summary = summary.merged(entry.feature)
    return BitFeature(summary.count, summary.linear_sum.copy(), summary.members.copy())


def _closest_entry(entries: list[TreeEntry], centroid: np.ndarray) -> int:
    similarities = [tanimoto(centroid, entry.feature.centroid) for entry in entries]
    return int(np.argmax(similarities))


def _split_seeds(entries: list[TreeEntry]) -> tuple[int, int]:
    if len(entries) < 2:
        raise ValueError("a split requires at least two entries")
    best_pair = (0, 1)
    best_similarity = tanimoto(entries[0].feature.centroid, entries[1].feature.centroid)
    for lhs in range(len(entries)):
        for rhs in range(lhs + 1, len(entries)):
            similarity = tanimoto(entries[lhs].feature.centroid, entries[rhs].feature.centroid)
            if similarity < best_similarity:
                best_similarity = similarity
                best_pair = (lhs, rhs)
    return best_pair


def _replace_parent_summary(node: TreeNode) -> None:
    if node.parent is None:
        return
    for entry in node.parent.entries:
        if entry.child is node:
            entry.feature = _summarize_node(node)
            return
    raise RuntimeError("parent does not reference child")


def _split_node(node: TreeNode, branching_factor: int) -> TreeNode:
    seed_lhs, seed_rhs = _split_seeds(node.entries)
    lhs_seed = node.entries[seed_lhs]
    rhs_seed = node.entries[seed_rhs]
    lhs_entries = [lhs_seed]
    rhs_entries = [rhs_seed]
    max_group = (len(node.entries) + 1) // 2
    for index, entry in enumerate(node.entries):
        if index in (seed_lhs, seed_rhs):
            continue
        lhs_similarity = tanimoto(entry.feature.centroid, lhs_seed.feature.centroid)
        rhs_similarity = tanimoto(entry.feature.centroid, rhs_seed.feature.centroid)
        assign_left = lhs_similarity > rhs_similarity or (
            lhs_similarity == rhs_similarity and len(lhs_entries) <= len(rhs_entries)
        )
        if len(lhs_entries) >= max_group:
            assign_left = False
        elif len(rhs_entries) >= max_group:
            assign_left = True
        if assign_left:
            lhs_entries.append(entry)
        else:
            rhs_entries.append(entry)

    # An overflow contains branching_factor + 1 entries and each side owns a
    # seed, so neither side can remain over capacity.
    assert len(lhs_entries) <= branching_factor
    assert len(rhs_entries) <= branching_factor
    node.entries = lhs_entries
    sibling = TreeNode(node.leaf, rhs_entries, node.parent)
    for entry in node.entries:
        if entry.child is not None:
            entry.child.parent = node
    for entry in sibling.entries:
        if entry.child is not None:
            entry.child.parent = sibling

    if node.parent is None:
        root = TreeNode(
            False,
            [TreeEntry(_summarize_node(node), node), TreeEntry(_summarize_node(sibling), sibling)],
        )
        node.parent = root
        sibling.parent = root
        return root

    parent = node.parent
    _replace_parent_summary(node)
    parent.entries.append(TreeEntry(_summarize_node(sibling), sibling))
    if len(parent.entries) > branching_factor:
        return _split_node(parent, branching_factor)
    _replace_parent_summary(parent)
    while parent.parent is not None:
        parent = parent.parent
        _replace_parent_summary(parent)
    return parent


def _leaf_features(root: TreeNode) -> list[BitFeature]:
    pending = [root]
    result: list[BitFeature] = []
    while pending:
        node = pending.pop()
        if node.leaf:
            result.extend(entry.feature for entry in node.entries)
        else:
            pending.extend(reversed([entry.child for entry in node.entries if entry.child is not None]))
    return result


def _insert_feature(
    root: TreeNode,
    incoming: BitFeature,
    threshold: float,
    branching_factor: int,
    tolerance: float | None,
) -> TreeNode:
    node = root
    while not node.leaf:
        child = node.entries[_closest_entry(node.entries, incoming.centroid)].child
        if child is None:
            raise RuntimeError("internal entry has no child")
        node = child

    if not node.entries:
        node.entries.append(TreeEntry(incoming))
    else:
        entry_index = _closest_entry(node.entries, incoming.centroid)
        entry = node.entries[entry_index]
        combined = entry.feature.merged(incoming)
        merge = combined.isim >= threshold
        if merge and tolerance is not None and entry.feature.count > 1:
            if incoming.count != 1:
                raise ValueError("tolerance-diameter summary merging is not defined")
            n = float(entry.feature.count)
            affinity = ((n + 1.0) * combined.isim - (n - 1.0) * entry.feature.isim) * 0.5
            merge = affinity >= entry.feature.isim - tolerance
        if merge:
            entry.feature = combined
        else:
            node.entries.append(TreeEntry(incoming))

    current = node
    while current.parent is not None:
        _replace_parent_summary(current)
        current = current.parent
    if len(node.entries) > branching_factor:
        return _split_node(node, branching_factor)
    return root


def serial_tree_reference(
    bits: np.ndarray,
    threshold: float,
    *,
    branching_factor: int = 254,
    tolerance: float | None = None,
) -> tuple[np.ndarray, list[BitFeature], TreeNode | None]:
    """Complete ordered serial-tree reference with split propagation."""
    fingerprints = _validate_bits(bits)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if branching_factor < 2:
        raise ValueError("branching_factor must be at least 2")
    if tolerance is not None and tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    if fingerprints.shape[0] == 0:
        return np.empty(0, dtype=np.int32), [], None
    root = TreeNode(True, [])
    for index, fingerprint in enumerate(fingerprints):
        root = _insert_feature(
            root,
            BitFeature.from_fingerprint(fingerprint, index),
            threshold,
            branching_factor,
            tolerance,
        )

    features = sorted(_leaf_features(root), key=lambda feature: min(feature.members))
    labels = np.empty(fingerprints.shape[0], dtype=np.int32)
    for cluster_id, feature in enumerate(features):
        labels[feature.members] = cluster_id
    return labels, features, root


def partitioned_tree_reference(
    bits: np.ndarray,
    threshold: float,
    *,
    branching_factor: int = 254,
    tolerance: float | None = None,
    num_partitions: int,
) -> tuple[np.ndarray, list[BitFeature], TreeNode | None]:
    """Reference contiguous partial trees followed by one Bit Feature merge round."""
    fingerprints = _validate_bits(bits)
    if num_partitions < 1 or (fingerprints.shape[0] and num_partitions > fingerprints.shape[0]):
        raise ValueError("invalid partition count")
    if fingerprints.shape[0] == 0:
        return np.empty(0, dtype=np.int32), [], None
    if tolerance is not None and num_partitions > 1:
        raise ValueError("tolerance-diameter summary merging is not defined")

    partition_size = (fingerprints.shape[0] + num_partitions - 1) // num_partitions
    partial_features: list[BitFeature] = []
    for begin in range(0, fingerprints.shape[0], partition_size):
        end = min(begin + partition_size, fingerprints.shape[0])
        _, features, _ = serial_tree_reference(
            fingerprints[begin:end],
            threshold,
            branching_factor=branching_factor,
            tolerance=tolerance,
        )
        for feature in features:
            feature.members = [member + begin for member in feature.members]
        partial_features.extend(features)

    root = TreeNode(True, [])
    for feature in partial_features:
        root = _insert_feature(root, feature, threshold, branching_factor, tolerance)
    features = sorted(_leaf_features(root), key=lambda feature: min(feature.members))
    labels = np.empty(fingerprints.shape[0], dtype=np.int32)
    for cluster_id, feature in enumerate(features):
        labels[feature.members] = cluster_id
    return labels, features, root


def serial_leaf_reference(
    bits: np.ndarray,
    threshold: float,
    *,
    tolerance: float | None = None,
) -> tuple[np.ndarray, list[BitFeature]]:
    """Reference ordered insertion into one unbounded BitBIRCH leaf."""
    fingerprints = _validate_bits(bits)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if tolerance is not None and tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")

    labels = np.empty(fingerprints.shape[0], dtype=np.int32)
    features: list[BitFeature] = []
    for index, fingerprint in enumerate(fingerprints):
        if not features:
            features.append(BitFeature.from_fingerprint(fingerprint, index))
            labels[index] = 0
            continue

        similarities = [tanimoto(fingerprint, feature.centroid) for feature in features]
        feature_index = int(np.argmax(similarities))
        feature = features[feature_index]
        combined_isim = feature.combined_isim(fingerprint)
        merge = combined_isim >= threshold
        if merge and tolerance is not None and feature.count > 1:
            n = float(feature.count)
            affinity = ((n + 1.0) * combined_isim - (n - 1.0) * feature.isim) * 0.5
            merge = affinity >= feature.isim - tolerance

        if merge:
            feature.add(fingerprint, index)
        else:
            feature_index = len(features)
            features.append(BitFeature.from_fingerprint(fingerprint, index))
        labels[index] = feature_index

    return labels, features
