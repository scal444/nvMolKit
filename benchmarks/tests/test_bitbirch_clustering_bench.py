# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import bitbirch_clustering_bench as bench
import pytest

def test_bblean_is_not_imported_when_disabled(monkeypatch):
    def unexpected_import(_name):
        raise AssertionError("bblean import should not be attempted")

    monkeypatch.setattr(bench.importlib, "import_module", unexpected_import)

    module, status, detail = bench._load_bblean(False)

    assert module is None
    assert status == "disabled"
    assert detail == ""


def test_missing_bblean_is_reported_as_optional(monkeypatch):
    def missing_import(_name):
        raise ImportError("no bblean")

    monkeypatch.setattr(bench.importlib, "import_module", missing_import)

    module, status, detail = bench._load_bblean(True)

    assert module is None
    assert status == "not_installed"
    assert detail == "no bblean"


def test_cluster_distribution_reports_sizes_and_singletons():
    distribution = bench._cluster_distribution([[0, 1, 2], [3], [4]], 5)

    assert distribution["num_clusters"] == 3
    assert distribution["mean_cluster_size"] == pytest.approx(5 / 3)
    assert distribution["cluster_size_p50"] == 1
    assert distribution["cluster_size_p90"] == pytest.approx(2.6)
    assert distribution["cluster_size_p99"] == pytest.approx(2.96)
    assert distribution["max_cluster_size"] == 3
    assert distribution["singleton_cluster_fraction"] == pytest.approx(2 / 3)
    assert distribution["items_in_singletons_fraction"] == pytest.approx(2 / 5)
    assert distribution["largest_cluster_fraction"] == pytest.approx(3 / 5)
