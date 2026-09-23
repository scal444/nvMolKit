# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental multi-route try-inserts on the same shared BF tree.

This is a quality hypothesis, not a GPU performance estimate. At each directory
level keep the highest-scoring child BFs across the current beam, then choose
the closest leaf entry across the surviving leaves. Payload admission and split
rules remain those of the independent batched reference.
"""

from collections import defaultdict

import numpy as np

from _bitbirch_batched_reference import BatchedBitBirch, similarities


def decreasing_score(candidate):
    return -candidate[0]


class BeamBitBirch(BatchedBitBirch):
    def __init__(
        self,
        threshold=0.25,
        branching_factor=254,
        batch_size=1024,
        beam_width=2,
        ordered_prefix_size=0,
        reverse_service=False,
    ):
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        super().__init__(
            threshold,
            branching_factor,
            batch_size,
            commit_policy="filtered",
            ordered_prefix_size=ordered_prefix_size,
            reverse_service=reverse_service,
        )
        self.beam_width = beam_width

    def _candidates(self, node, molecule, limit):
        scores = similarities(self.packed[molecule : molecule + 1], node.matrix())[0]
        choices = np.argsort(-scores, kind="stable")[:limit]
        self.stats["beam_centroid_comparisons"] += len(node.entries)
        return [(float(scores[index]), node.entries[index], node) for index in choices]

    def _route(self, molecule_ids):
        if self.root.leaf or self.beam_width == 1 or molecule_ids[0] < self.ordered_prefix_size:
            return super()._route(molecule_ids)
        groups = defaultdict(list)
        nodes, entries = {}, {}
        for molecule in molecule_ids:
            beam = [self.root]
            while not beam[0].leaf:
                candidates = []
                for node in beam:
                    candidates.extend(self._candidates(node, molecule, self.beam_width))
                candidates.sort(key=decreasing_score)
                beam = [entry.child for _, entry, _ in candidates[: self.beam_width]]
            candidates = []
            for node in beam:
                candidates.extend(self._candidates(node, molecule, 1))
            candidates.sort(key=decreasing_score)
            _, entry, node = candidates[0]
            groups[(node.uid, entry.uid)].append(int(molecule))
            nodes[node.uid] = node
            entries[entry.uid] = entry
        self.stats["route_attempts"] += len(molecule_ids)
        self.stats["beam_routed_queries"] += len(molecule_ids)
        self.stats["max_cluster_queue"] = max(self.stats["max_cluster_queue"], max(map(len, groups.values())))
        return groups, nodes, entries
