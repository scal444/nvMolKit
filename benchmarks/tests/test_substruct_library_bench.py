# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import ANY

import pytest
import substruct_library_bench as benchmark
from bench_utils import TimingResult


class FakeNvLibrary:
    def __init__(self):
        self.calls = []

    def hasMatch(self, query):
        self.calls.append(("has", query))
        return query == "hit"

    def countMatches(self, query):
        self.calls.append(("count", query))
        return len(query)

    def getMatches(self, query, maxResults):
        self.calls.append(("get", query, maxResults))
        return range(min(len(query), maxResults if maxResults >= 0 else len(query)))


class FakeRdkitLibrary:
    def __init__(self):
        self.calls = []

    def HasMatch(self, query, **kwargs):
        self.calls.append(("has", query, kwargs))
        return query == "hit"

    def CountMatches(self, query, **kwargs):
        self.calls.append(("count", query, kwargs))
        return len(query)

    def GetMatches(self, query, **kwargs):
        self.calls.append(("get", query, kwargs))
        max_results = kwargs["maxResults"]
        return range(min(len(query), max_results if max_results >= 0 else len(query)))


@pytest.mark.parametrize(
    ("operation", "expected"),
    [("has", [True, False]), ("count", [3, 2]), ("get", [[0, 1], [0, 1]])],
)
def test_nvmolkit_operations_dispatch_and_preserve_query_order(operation, expected):
    library = FakeNvLibrary()

    result = benchmark._run_nvmolkit_queries(library, ["hit", "zz"], operation, max_results=2)

    assert result == expected
    if operation == "get":
        assert library.calls == [("get", "hit", 2), ("get", "zz", 2)]


@pytest.mark.parametrize(
    ("operation", "expected"),
    [("has", [True, False]), ("count", [3, 2]), ("get", [[0], [0]])],
)
def test_rdkit_operations_forward_threads_and_max_results(operation, expected):
    library = FakeRdkitLibrary()

    result = benchmark._run_rdkit_queries(library, ["hit", "zz"], operation, max_results=1, num_threads=7)

    assert result == expected
    for _, _, kwargs in library.calls:
        assert kwargs["recursionPossible"] is True
        assert kwargs["useChirality"] is False
        assert kwargs["useQueryQueryMatches"] is False
        assert kwargs["numThreads"] == 7
        assert kwargs.get("maxResults", 1) == 1


def test_lifecycle_separates_staging_finalize_and_repeated_search(monkeypatch):
    events = []
    timing_values = iter([[10.0, 14.0], [20.0, 24.0], [30.0, 34.0]])

    def fake_time_it(function, *, runs, warmups, gpu_sync=False, setup=None):
        events.append((runs, warmups, gpu_sync))
        if setup is not None:
            setup()
        function()
        return TimingResult(times_ms=next(timing_values))

    monkeypatch.setattr(benchmark, "time_it", fake_time_it)
    serial = iter(range(10))

    def make_library():
        value = {"id": next(serial), "staged": False, "finalized": False}
        events.append(("make", value["id"]))
        return value

    def stage_library(library):
        library["staged"] = True
        events.append(("stage", library["id"]))

    def finalize_library(library):
        assert library["staged"]
        library["finalized"] = True
        events.append(("finalize", library["id"]))

    search_calls = []

    def search_library(library):
        assert library["finalized"]
        search_calls.append(library["id"])
        return [library["id"]]

    result = benchmark._benchmark_lifecycle(
        make_library=make_library,
        stage_library=stage_library,
        finalize_library=finalize_library,
        search_library=search_library,
        runs=2,
        warmups=3,
        repetitions=4,
        gpu_finalize=True,
        gpu_search=True,
    )

    assert result.staging_ms == 12.0
    assert result.finalize_ms == 22.0
    assert result.steady_ms == 32.0
    assert result.amortized_ms == 66.0
    assert result.results == [1]
    assert search_calls == [1] * 4
    assert events[:1] == [(2, 0, False)]
    assert (2, 0, True) in events
    assert (2, 3, True) in events


def test_result_row_reports_steady_and_amortized_work_rates():
    measurement = benchmark.LifecycleMeasurement(10, 0, 20, 0, 50, 0, [5, 0, 20, 1, 0])

    row = benchmark._result_row(
        backend="gpu",
        operation="count",
        measurement=measurement,
        num_mols=100,
        num_queries=5,
        repetitions=2,
        algorithm="dfs",
    )

    assert row["steady_queries_per_s"] == 200
    assert row["steady_offered_pairs_per_s"] == 20_000
    assert row["amortized_queries_per_s"] == 125
    assert row["amortized_offered_pairs_per_s"] == 12_500
    assert row["positive_queries"] == 3
    assert row["query_hit_rate"] == 0.6
    assert row["result_cardinality"] == 26
    assert row["result_pair_density"] == 0.052
    assert row["algorithm"] == "dfs"


def test_validation_checks_all_queries_and_get_order():
    benchmark._validate_results([[0, 2], []], [[0, 2], []], "get")

    with pytest.raises(AssertionError, match="query 1"):
        benchmark._validate_results([[0, 2], [3]], [[0, 2], [4]], "get")
    with pytest.raises(AssertionError, match="result length"):
        benchmark._validate_results([True], [True, False], "has")


def test_reference_uses_single_threaded_mol_holder(monkeypatch):
    library = FakeRdkitLibrary()
    added = []
    library.AddMol = added.append
    monkeypatch.setattr(benchmark, "_make_rdkit_library", lambda holder: library)

    results = benchmark._rdkit_reference_results(["mol-a", "mol-b"], ["hit"], "get", 4)

    assert added == ["mol-a", "mol-b"]
    assert results == [[0, 1, 2]]
    _, _, kwargs = library.calls[0]
    assert kwargs["numThreads"] == 1
    assert kwargs["maxResults"] == 4


def _args(**overrides):
    values = {
        "num_mols": 0,
        "chunk_sizes": [65_536],
        "batch_size": 1024,
        "workers": -1,
        "prep_threads": -1,
        "gpu_id": 0,
        "rdkit_threads": [-1],
        "max_results": -1,
        "runs": 3,
        "warmups": 1,
        "repetitions": 1,
        "no_rdkit": False,
        "no_nvmolkit": False,
        "validate": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"chunk_sizes": [0]}, "chunk_sizes"),
        ({"batch_size": 0}, "batch_size"),
        ({"workers": -2}, "workers"),
        ({"prep_threads": -2}, "prep_threads"),
        ({"gpu_id": -1}, "gpu_id"),
        ({"rdkit_threads": [0]}, "rdkit_threads"),
        ({"max_results": -2}, "max_results"),
        ({"repetitions": 0}, "repetitions"),
        ({"no_rdkit": True, "no_nvmolkit": True}, "disable both"),
        ({"no_rdkit": True}, "validation requires both"),
    ],
)
def test_argument_validation_rejects_invalid_or_incomparable_runs(overrides, message):
    with pytest.raises(ValueError, match=message):
        benchmark._validate_args(_args(**overrides))


def test_parser_exposes_backend_sweeps_and_lifecycle_controls():
    args = benchmark._build_parser().parse_args(
        [
            "--smiles",
            "mols.smi",
            "--smarts",
            "queries.smarts",
            "--operations",
            "has",
            "count",
            "get",
            "--algorithms",
            "gsi",
            "dfs",
            "--chunk_sizes",
            "8192",
            "65536",
            "--rdkit_holders",
            "mol",
            "cached-pattern",
            "--rdkit_threads",
            "1",
            "8",
            "--maxResults",
            "20",
            "--repetitions",
            "5",
        ]
    )

    assert args.operations == ["has", "count", "get"]
    assert args.algorithms == ["gsi", "dfs"]
    assert args.chunk_sizes == [8192, 65536]
    assert args.rdkit_holders == ["mol", "cached-pattern"]
    assert args.rdkit_threads == [1, 8]
    assert args.max_results == 20
    assert args.repetitions == 5


def test_nvmolkit_benchmark_constructs_requested_config_and_library(monkeypatch):
    configured = []
    constructed = []
    selected_devices = []
    expected = benchmark.LifecycleMeasurement(1, 0, 2, 0, 3, 0, [])

    class FakeConfig:
        def __init__(self, **kwargs):
            configured.append(kwargs)

    class FakeLibrary:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    substructure = ModuleType("nvmolkit.substructure")
    substructure.SubstructSearchConfig = FakeConfig
    substruct_library = ModuleType("nvmolkit.substruct_library")
    substruct_library.SubstructLibrary = FakeLibrary
    monkeypatch.setitem(sys.modules, "nvmolkit.substructure", substructure)
    monkeypatch.setitem(sys.modules, "nvmolkit.substruct_library", substruct_library)
    monkeypatch.setattr("torch.cuda.set_device", selected_devices.append)

    def fake_lifecycle(**kwargs):
        kwargs["make_library"]()
        assert kwargs["gpu_finalize"]
        assert kwargs["gpu_search"]
        return expected

    monkeypatch.setattr(benchmark, "_benchmark_lifecycle", fake_lifecycle)

    result = benchmark.benchmark_nvmolkit(
        [object()],
        [object()],
        operation="get",
        algorithm="dfs",
        chunk_size=8192,
        batch_size=512,
        worker_threads=3,
        preprocessing_threads=4,
        gpu_id=2,
        max_results=10,
        runs=2,
        warmups=1,
        repetitions=5,
    )

    assert result is expected
    assert selected_devices == [2]
    assert configured == [
        {
            "batchSize": 512,
            "workerThreads": 3,
            "preprocessingThreads": 4,
            "gpuIds": [2],
            "algorithm": "dfs",
        }
    ]
    assert constructed == [{"chunkSize": 8192, "config": ANY}]


def test_main_runs_requested_cross_product_and_validates_each_gpu_result(monkeypatch):
    measurement = benchmark.LifecycleMeasurement(1, 0, 2, 0, 3, 0, [True])
    rdkit_calls = []
    nvmolkit_calls = []
    validation_calls = []
    reference_calls = []
    emitted_rows = []

    monkeypatch.setattr(
        "sys.argv",
        [
            "substruct_library_bench.py",
            "--smiles",
            "mols.smi",
            "--smarts",
            "queries.smarts",
            "--rdkit_holders",
            "mol",
            "cached-pattern",
            "--rdkit_threads",
            "1",
            "2",
            "--algorithms",
            "gsi",
            "dfs",
            "--chunk_sizes",
            "8",
            "16",
            "--max_results",
            "0",
        ],
    )
    monkeypatch.setattr(benchmark, "_load_molecules", lambda args: [object(), object()])
    monkeypatch.setattr(benchmark, "load_smarts", lambda path: ([object()], ["C"]))
    monkeypatch.setattr(
        benchmark,
        "benchmark_rdkit",
        lambda *args, **kwargs: rdkit_calls.append(kwargs) or measurement,
    )
    monkeypatch.setattr(
        benchmark,
        "benchmark_nvmolkit",
        lambda *args, **kwargs: nvmolkit_calls.append(kwargs) or measurement,
    )
    monkeypatch.setattr(
        benchmark,
        "_validate_results",
        lambda *args: validation_calls.append(args),
    )
    monkeypatch.setattr(
        benchmark,
        "_rdkit_reference_results",
        lambda *args: reference_calls.append(args) or [True],
    )
    monkeypatch.setattr(benchmark, "print_csv_rows", emitted_rows.extend)
    monkeypatch.setattr(benchmark, "write_csv_rows", lambda rows, output: None)

    benchmark.main()

    assert {(call["holder"], call["num_threads"]) for call in rdkit_calls} == {
        ("mol", 1),
        ("mol", 2),
        ("cached-pattern", 1),
        ("cached-pattern", 2),
    }
    assert {(call["algorithm"], call["chunk_size"]) for call in nvmolkit_calls} == {
        ("gsi", 8),
        ("gsi", 16),
        ("dfs", 8),
        ("dfs", 16),
    }
    assert len(validation_calls) == len(nvmolkit_calls) == 4
    assert len(reference_calls) == 1
    assert all(call["max_results"] == -1 for call in rdkit_calls + nvmolkit_calls)
    assert len(emitted_rows) == 8
    assert {row["backend"] for row in emitted_rows} == {
        "rdkit-substruct-library",
        "nvmolkit-substruct-library",
    }
