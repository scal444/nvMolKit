# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import ANY

import pytest
import substruct_library_bench as benchmark
from bench_utils import TimingResult
from rdkit import Chem


class _FakeNvLibrary:
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


class _FakeRdkitLibrary:
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
    library = _FakeNvLibrary()

    result = benchmark._run_nvmolkit_queries(library, ["hit", "zz"], operation, max_results=2)

    assert result == expected
    if operation == "get":
        assert library.calls == [("get", "hit", 2), ("get", "zz", 2)]


@pytest.mark.parametrize(
    ("operation", "expected"),
    [("has", [True, False]), ("count", [3, 2]), ("get", [[0], [0]])],
)
def test_rdkit_operations_forward_threads_and_max_results(operation, expected):
    library = _FakeRdkitLibrary()

    result = benchmark._run_rdkit_queries(library, ["hit", "zz"], operation, max_results=1, num_threads=7)

    assert result == expected
    for _, _, kwargs in library.calls:
        assert kwargs["recursionPossible"] is True
        assert kwargs["useChirality"] is False
        assert kwargs["useQueryQueryMatches"] is False
        assert kwargs["numThreads"] == 7
        assert kwargs.get("maxResults", 1) == 1


def test_rdkit_operations_stop_between_queries_when_deadline_expires():
    library = _FakeRdkitLibrary()
    expiry_checks = iter([False, True])
    deadline = SimpleNamespace(expired=lambda: next(expiry_checks))

    result = benchmark._run_rdkit_queries(
        library,
        ["hit", "zz"],
        "has",
        max_results=-1,
        num_threads=112,
        deadline=deadline,
    )

    assert result == [True]
    assert len(library.calls) == 1


def test_library_is_built_once_and_reused_by_each_measured_search(monkeypatch):
    events = []
    timing_values = iter([[12.0], [22.0], [30.0, 34.0]])

    def fake_time_it(function, *, runs, warmups, gpu_sync=False):
        events.append((runs, warmups, gpu_sync))
        function()
        return TimingResult(times_ms=next(timing_values))

    monkeypatch.setattr(benchmark, "time_it", fake_time_it)
    made = []

    def make_library():
        library = {"id": len(made), "staged": False, "finalized": False}
        made.append(library)
        return library

    def stage_library(library):
        library["staged"] = True

    def finalize_library(library):
        assert library["staged"]
        library["finalized"] = True

    built = benchmark._build_library(
        make_library=make_library,
        stage_library=stage_library,
        finalize_library=finalize_library,
        gpu_finalize=True,
    )
    search_calls = []

    def search_library(library):
        assert library["finalized"]
        search_calls.append(library["id"])
        return [library["id"]]

    result = benchmark._measure_search(built, search_library, runs=2, warmups=3, repetitions=4, gpu_search=True)

    assert len(made) == 1
    assert result.staging_ms == 12.0
    assert result.finalize_ms == 22.0
    assert result.steady_ms == 32.0
    assert result.amortized_ms == 66.0
    assert result.results == [0]
    assert search_calls == [0] * 4
    assert events == [(1, 0, False), (1, 0, True), (2, 3, True)]


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


def test_validation_compares_queries_completed_before_rdkit_deadline():
    benchmark._validate_results([True, False, True], [True, False], "has")
    benchmark._validate_results([True, False], [], "has")

    with pytest.raises(AssertionError, match="query 1"):
        benchmark._validate_results([True, True, True], [True, False], "has")


def test_reference_collects_every_match_with_the_requested_threads(monkeypatch):
    library = _FakeRdkitLibrary()
    staged = []
    monkeypatch.setattr(benchmark, "_make_rdkit_library", lambda holder: library)
    monkeypatch.setattr(
        benchmark,
        "_stage_rdkit_library",
        lambda lib, mols, holder, num_threads: staged.append((list(mols), holder, num_threads)),
    )

    results = benchmark._rdkit_reference_results(["mol-a", "mol-b"], ["hit"], 16)

    assert staged == [(["mol-a", "mol-b"], "cached-pattern", 16)]
    assert results == [[0, 1, 2]]
    _, _, kwargs = library.calls[0]
    assert kwargs["numThreads"] == 16
    assert kwargs["maxResults"] == -1


def test_reference_derives_every_operation_from_complete_matches():
    all_matches = [[3, 5, 9], [], [1]]

    assert benchmark._reference_for_operation(all_matches, "has", -1) == [True, False, True]
    assert benchmark._reference_for_operation(all_matches, "count", -1) == [3, 0, 1]
    assert benchmark._reference_for_operation(all_matches, "get", -1) == all_matches
    assert benchmark._reference_for_operation(all_matches, "get", 2) == [[3, 5], [], [1]]


def test_reference_cache_is_reused_only_for_identical_inputs(monkeypatch, tmp_path):
    computed = []

    def fake_reference(mols, queries, num_threads):
        computed.append(len(queries))
        return [[index] for index in range(len(queries))]

    monkeypatch.setattr(benchmark, "_rdkit_reference_results", fake_reference)
    cache = str(tmp_path / "reference.pkl")
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")]
    queries = [Chem.MolFromSmarts("CO")]

    assert benchmark.load_or_compute_reference(cache, mols, queries, 4) == [[0]]
    assert benchmark.load_or_compute_reference(cache, mols, queries, 4) == [[0]]
    assert computed == [1]

    more_queries = queries + [Chem.MolFromSmarts("c")]
    assert benchmark.load_or_compute_reference(cache, mols, more_queries, 4) == [[0], [1]]
    assert computed == [1, 2]


def _args(**overrides):
    values = {
        "num_mols": 0,
        "chunk_sizes": [65_536],
        "batch_size": 1024,
        "workers": -1,
        "prep_threads": -1,
        "gpu_ids": None,
        "gpu_id": 0,
        "rdkit_threads": [-1],
        "rdkit_max_seconds": 0.0,
        "max_results": -1,
        "runs": 3,
        "warmups": 1,
        "repetitions": 1,
        "no_rdkit": False,
        "no_nvmolkit": False,
        "validate": True,
        "num_queries": 0,
        "reference_cache": None,
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
        ({"gpu_id": -1}, "gpu_ids"),
        ({"gpu_ids": [0, 0], "gpu_id": None}, "unique"),
        ({"rdkit_threads": [0]}, "rdkit_threads"),
        ({"rdkit_max_seconds": -1}, "rdkit_max_seconds"),
        ({"max_results": -2}, "max_results"),
        ({"repetitions": 0}, "repetitions"),
        ({"no_rdkit": True, "no_nvmolkit": True}, "disable both"),
        ({"num_queries": -1}, "num_queries"),
        ({"no_rdkit": True}, "requires --reference_cache"),
        ({"no_nvmolkit": True}, "validating RDKit alone"),
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
            "--rdkit_max_seconds",
            "300",
            "--gpu_ids",
            "0",
            "1",
            "--maxResults",
            "20",
            "--repetitions",
            "5",
            "--no_nvmolkit_pattern_fingerprints",
            "--query_modes",
            "serial",
            "concurrent",
            "--num_queries",
            "7",
            "--reference_cache",
            "reference.pkl",
        ]
    )

    assert args.operations == ["has", "count", "get"]
    assert args.algorithms == ["gsi", "dfs"]
    assert args.chunk_sizes == [8192, 65536]
    assert args.rdkit_holders == ["mol", "cached-pattern"]
    assert args.rdkit_threads == [1, 8]
    assert args.rdkit_max_seconds == 300
    assert args.gpu_ids == [0, 1]
    assert args.max_results == 20
    assert args.repetitions == 5
    assert args.nvmolkit_pattern_fingerprints is False
    assert args.query_modes == ["serial", "concurrent"]
    assert args.num_queries == 7
    assert args.reference_cache == "reference.pkl"


def test_query_smiles_are_molecules_with_stereochemistry_removed(tmp_path):
    query_path = tmp_path / "queries.smi"
    query_path.write_text("N[C@@H](C)C(=O)O\n")
    args = benchmark._build_parser().parse_args(["--smiles", "mols.smi", "--query_smiles", str(query_path)])

    queries = benchmark._load_queries(args)

    assert len(queries) == 1
    assert all(atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED for atom in queries[0].GetAtoms())


def test_nvmolkit_benchmark_builds_one_library_for_every_operation_and_mode(monkeypatch):
    configured = []
    constructed = []
    selected_devices = []
    searched = []

    class FakeConfig:
        def __init__(self, **kwargs):
            configured.append(kwargs)

    class FakeLibrary:
        queryConcurrency = 4
        batchesInFlightPerGpu = 12
        workspaceBytesPerQueryPerGpu = 99

        def __init__(self, **kwargs):
            constructed.append(kwargs)

    substructure = ModuleType("nvmolkit.substructure")
    substructure.SubstructSearchConfig = FakeConfig
    substruct_library = ModuleType("nvmolkit.substruct_library")
    substruct_library.SubstructLibrary = FakeLibrary
    monkeypatch.setitem(sys.modules, "nvmolkit.substructure", substructure)
    monkeypatch.setitem(sys.modules, "nvmolkit.substruct_library", substruct_library)
    monkeypatch.setattr("torch.cuda.set_device", selected_devices.append)

    def fake_build(**kwargs):
        assert kwargs["gpu_finalize"]
        return benchmark.BuiltLibrary(kwargs["make_library"](), 1.0, 2.0)

    def fake_measure(built, search_library, **kwargs):
        assert kwargs["gpu_search"]
        searched.append(built.library)
        return benchmark.LifecycleMeasurement(1, 0, 2, 0, 3, 0, [])

    monkeypatch.setattr(benchmark, "_build_library", fake_build)
    monkeypatch.setattr(benchmark, "_measure_search", fake_measure)

    result = benchmark.benchmark_nvmolkit(
        [object()],
        [object()],
        operations=["has", "get"],
        query_modes=["serial", "concurrent"],
        algorithm="dfs",
        chunk_size=8192,
        batch_size=512,
        worker_threads=3,
        preprocessing_threads=4,
        gpu_ids=[2, 3],
        max_results=10,
        runs=2,
        warmups=1,
        repetitions=5,
        use_pattern_fingerprints=False,
    )

    assert set(result) == {("has", "serial"), ("has", "concurrent"), ("get", "serial"), ("get", "concurrent")}
    assert all(measurement.query_concurrency == 4 for measurement in result.values())
    assert len(constructed) == 1
    assert len(searched) == 4 and len({id(library) for library in searched}) == 1
    assert selected_devices == [2]
    assert configured == [
        {
            "batchSize": 512,
            "workerThreads": 3,
            "preprocessingThreads": 4,
            "gpuIds": [2, 3],
            "algorithm": "dfs",
        }
    ]
    assert constructed == [{"chunkSize": 8192, "config": ANY, "usePatternFingerprints": False}]


@pytest.mark.parametrize(
    ("nvmolkit_results", "mismatch"),
    [
        pytest.param([True, False, True], None, id="agrees"),
        pytest.param([True, True, True], "query 1", id="differs-after-shorter-deadline"),
    ],
)
def test_main_validates_against_longest_deadline_bounded_rdkit_run(monkeypatch, capsys, nvmolkit_results, mismatch):
    rdkit_measurements = iter(
        [
            benchmark.LifecycleMeasurement(1, 0, 0, 0, 3, 0, [True], completed_queries=1),
            benchmark.LifecycleMeasurement(1, 0, 0, 0, 3, 0, [True, False], completed_queries=2),
        ]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "substruct_library_bench.py",
            "--smiles",
            "mols.smi",
            "--smarts",
            "queries.smarts",
            "--rdkit_threads",
            "1",
            "2",
            "--rdkit_max_seconds",
            "5",
        ],
    )
    monkeypatch.setattr(benchmark, "_load_molecules", lambda args: [object(), object()])
    monkeypatch.setattr(benchmark, "_load_queries", lambda args: [object(), object(), object()])
    monkeypatch.setattr(benchmark, "benchmark_rdkit", lambda *args, **kwargs: {"has": next(rdkit_measurements)})
    monkeypatch.setattr(
        benchmark,
        "benchmark_nvmolkit",
        lambda *args, **kwargs: {
            ("has", "serial"): benchmark.LifecycleMeasurement(1, 0, 2, 0, 3, 0, nvmolkit_results)
        },
    )
    monkeypatch.setattr(benchmark, "print_csv_rows", lambda rows: None)
    monkeypatch.setattr(benchmark, "write_csv_rows", lambda rows, output: None)

    if mismatch is not None:
        with pytest.raises(AssertionError, match=mismatch):
            benchmark.main()
        return
    benchmark.main()
    assert "compared 2/3 queries" in capsys.readouterr().out


def test_main_runs_requested_cross_product_and_validates_each_gpu_result(monkeypatch):
    measurement = benchmark.LifecycleMeasurement(1, 0, 2, 0, 3, 0, [True])
    rdkit_calls = []
    nvmolkit_calls = []
    validation_calls = []
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
            "--gpu_ids",
            "0",
            "1",
            "--max_results",
            "0",
        ],
    )
    monkeypatch.setattr(benchmark, "_load_molecules", lambda args: [object(), object()])
    monkeypatch.setattr(benchmark, "_load_queries", lambda args: [object()])
    monkeypatch.setattr(
        benchmark,
        "benchmark_rdkit",
        lambda *args, **kwargs: rdkit_calls.append(kwargs) or {"has": measurement},
    )
    monkeypatch.setattr(
        benchmark,
        "benchmark_nvmolkit",
        lambda *args, **kwargs: nvmolkit_calls.append(kwargs) or {("has", "serial"): measurement},
    )
    monkeypatch.setattr(
        benchmark,
        "_validate_results",
        lambda *args: validation_calls.append(args),
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
    assert len(validation_calls) == 7
    assert len(nvmolkit_calls) == 4
    assert all(call["max_results"] == -1 for call in rdkit_calls + nvmolkit_calls)
    assert all(call["gpu_ids"] == [0, 1] for call in nvmolkit_calls)
    assert len(emitted_rows) == 8
    assert {row["backend"] for row in emitted_rows} == {
        "rdkit-substruct-library",
        "nvmolkit-substruct-library",
    }
