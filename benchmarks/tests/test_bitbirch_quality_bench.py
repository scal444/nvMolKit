# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from bitbirch_quality_bench import _cluster_isim, _cluster_quality


def test_cluster_isim_edge_cases_and_pair_formula():
    assert _cluster_isim(np.zeros((0, 4), dtype=np.uint8)) == 1.0
    assert _cluster_isim(np.array([[1, 0, 1]], dtype=np.uint8)) == 1.0
    assert _cluster_isim(np.zeros((3, 4), dtype=np.uint8)) == 1.0
    assert _cluster_isim(np.array([[1, 1], [1, 0]], dtype=np.uint8)) == pytest.approx(0.5)


def test_cluster_quality_reports_distribution_and_singletons():
    bits = np.array([[1, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    quality = _cluster_quality(bits, np.array([0, 0, 1, 2], dtype=np.int32))
    assert quality["num_clusters"] == 3
    assert quality["cluster_size_quantiles"] == [1.0, 1.0, 1.0, 1.5, 2.0]
    assert quality["singleton_cluster_fraction"] == pytest.approx(2 / 3)
    assert quality["items_in_singletons_fraction"] == 0.5
    assert quality["largest_cluster_fraction"] == 0.5
