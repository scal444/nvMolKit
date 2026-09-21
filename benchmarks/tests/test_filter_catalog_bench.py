# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from filter_catalog_bench import (
    CatalogRun,
    FilterDefinition,
    _build_parser,
    _make_rdkit_catalog,
    _search,
    _validate_args,
    benchmark_catalog,
    load_filter_definitions,
    rdkit_reference_ids,
    result_rows,
    validate_results,
)
from rdkit import Chem


class _NativeEntry:
    def __init__(self, description):
        self.description = description


class _NativeCatalog:
    def __init__(self, counters, results):
        self._counters = counters
        self._results = results
        self._descriptions = ["first", "second", "counted"]

    def finalize(self):
        self._counters["finalize"] += 1

    def __len__(self):
        return len(self._descriptions)

    def getEntry(self, index):
        return _NativeEntry(self._descriptions[index])

    def hasMatch(self, molecules):
        self._counters["search"] += 1
        assert len(molecules) == 3
        return self._results["any"]

    def getFirstMatch(self, molecules):
        self._counters["search"] += 1
        return self._results["first"]

    def getMatches(self, molecules):
        self._counters["search"] += 1
        return self._results["all"]


class _RDEntry:
    def __init__(self, description):
        self._description = description

    def GetDescription(self):
        return self._description


class _RDCatalog:
    def __init__(self, counters, results):
        self._counters = counters
        self._results = results
        self._entries = [_RDEntry("first"), _RDEntry("second"), _RDEntry("counted")]

    def GetNumEntries(self):
        return len(self._entries)

    def GetEntryWithIdx(self, index):
        return self._entries[index]

    def HasMatch(self, molecule):
        self._counters["search"] += 1
        return self._results["any"][molecule]

    def GetFirstMatch(self, molecule):
        self._counters["search"] += 1
        index = self._results["first"][molecule]
        return None if index is None else self._entries[index]

    def GetMatches(self, molecule):
        self._counters["search"] += 1
        return [self._entries[index] for index in self._results["all"][molecule]]


def _empty_run(**overrides):
    values = {
        "stage_ms": 2.0,
        "stage_std_ms": 0.2,
        "finalize_ms": 3.0,
        "finalize_std_ms": 0.3,
        "search_ms": 4.0,
        "search_std_ms": 0.4,
        "amortized_ms": 9.0,
        "amortized_std_ms": 0.9,
        "results": [],
        "entry_descriptions": ["first", "second", "counted"],
    }
    values.update(overrides)
    return CatalogRun(**values)


def test_custom_definition_loader_supports_defaults_counts_and_limits(tmp_path):
    path = tmp_path / "filters.tsv"
    path.write_text(
        "# comment\nC=O\n[#6]\t3\tat least three carbons\n[N+](=O)[O-]\t1\tnitro\n",
        encoding="utf-8",
    )

    definitions = load_filter_definitions(str(path), max_entries=2)

    assert [(item.smarts, item.trigger_count, item.description) for item in definitions] == [
        ("C=O", 1, "custom_0"),
        ("[#6]", 3, "at least three carbons"),
    ]


@pytest.mark.parametrize(
    ("contents", "message"),
    [("C\t0\n", "positive"), ("[\t1\n", "invalid SMARTS"), ("# empty\n", "no valid")],
)
def test_custom_definition_loader_rejects_invalid_catalogs(tmp_path, contents, message):
    path = tmp_path / "filters.tsv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_filter_definitions(str(path))


def test_rdkit_custom_catalog_preserves_order_and_trigger_counts():
    catalog = _make_rdkit_catalog(
        None,
        [FilterDefinition("O", 1, "oxygen"), FilterDefinition("[#6]", 3, "three carbons")],
    )
    molecules = [Chem.MolFromSmiles(smiles) for smiles in ["O", "CC", "CCC"]]

    matches = _search(catalog, molecules, "first", "rdkit")
    assert [None if entry is None else entry.GetDescription() for entry in matches] == [
        "oxygen",
        None,
        "three carbons",
    ]


@pytest.mark.parametrize("operation", ["any", "first", "all"])
def test_nvmolkit_lifecycle_measurements_isolate_each_phase(operation):
    counters = {"make": 0, "finalize": 0, "search": 0}
    expected = {
        "any": [True, False, True],
        "first": [0, None, 2],
        "all": [[0, 1], [], [2]],
    }

    def make_catalog():
        counters["make"] += 1
        return _NativeCatalog(counters, expected)

    run = benchmark_catalog(
        make_catalog,
        [object(), object(), object()],
        operation,
        "nvmolkit",
        runs=1,
        warmups=0,
        gpu_sync=False,
        reuse_count=3,
    )

    assert counters == {"make": 4, "finalize": 3, "search": 4}
    assert run.results == expected[operation]
    assert run.entry_descriptions == ["first", "second", "counted"]
    assert run.finalize_ms is not None


def test_rdkit_lifecycle_has_no_artificial_finalize_phase():
    counters = {"make": 0, "search": 0}
    expected = {"any": [True, False, True], "first": [0, None, 2], "all": [[0], [], [2]]}

    def make_catalog():
        counters["make"] += 1
        return _RDCatalog(counters, expected)

    run = benchmark_catalog(
        make_catalog,
        [0, 1, 2],
        "any",
        "rdkit",
        runs=1,
        warmups=0,
        gpu_sync=False,
        reuse_count=2,
    )

    assert counters == {"make": 3, "search": 9}
    assert run.results == [True, False, True]
    assert run.finalize_ms is None


@pytest.mark.parametrize(
    ("operation", "native_results"),
    [
        ("any", [True, False, True]),
        ("first", [0, None, 2]),
        ("all", [[0, 1], [], [2]]),
    ],
)
def test_validation_compares_exact_stable_ids(operation, native_results):
    native = _empty_run(results=native_results)
    expected = [[0, 1], [], [2]]

    validate_results(native, expected, operation)
    expected[-1] = []
    with pytest.raises(AssertionError, match="target 2"):
        validate_results(native, expected, operation)


def test_rdkit_reference_enumerates_duplicate_descriptions_as_distinct_ids():
    counters = {"search": 0}
    catalog = _RDCatalog(
        counters,
        {"any": [], "first": [], "all": [[0, 1], [], [2]]},
    )
    catalog._entries[1] = _RDEntry("first")
    for index, entry in enumerate(catalog._entries):
        entry.HasFilterMatch = lambda molecule, index=index: index in molecule

    assert rdkit_reference_ids(catalog, [[0, 1], [], [2]]) == [[0, 1], [], [2]]


def test_validation_reports_result_length_mismatch():
    native = _empty_run(results=[True])

    with pytest.raises(AssertionError, match="length mismatch"):
        validate_results(native, [[0], []], "any")


def test_result_rows_use_target_throughput_and_report_known_fallbacks():
    rows = result_rows(
        _empty_run(),
        backend="nvmolkit",
        operation="first",
        catalog_name="custom.tsv",
        num_targets=20,
        known_fallback_entries=1,
        config={"seed": 7},
    )

    assert [row["phase"] for row in rows] == ["stage", "finalize", "steady_search", "amortized"]
    assert rows[2]["throughput_unit"] == "targets/s"
    assert rows[2]["throughput"] == pytest.approx(5000.0)
    assert rows[2]["nominal_target_entry_pairs"] == 60
    assert rows[2]["known_trigger_count_fallback_entries"] == 1
    assert rows[2]["trigger_count_one_entries"] == 2
    assert rows[2]["seed"] == 7


def test_rdkit_rows_identify_the_python_loop_and_omit_gpu_configuration():
    rows = result_rows(
        _empty_run(finalize_ms=None, finalize_std_ms=None),
        backend="rdkit",
        operation="any",
        catalog_name="BRENK",
        num_targets=20,
        known_fallback_entries="N/A",
        config={"runs": 3},
    )

    assert rows[0]["method"] == "rdkit-python-loop"
    assert rows[0]["runs"] == 3
    assert "algorithm" not in rows[0]
    assert "gpu" not in rows[0]


def test_parser_exposes_catalog_lifecycle_workload_controls():
    args = _build_parser().parse_args(
        [
            "--smiles",
            "molecules.smi",
            "--smarts",
            "filters.tsv",
            "--operation",
            "any",
            "first",
            "all",
            "--max_entries",
            "100",
            "--num_mols",
            "500",
            "--no-warmup",
        ]
    )

    assert args.operation == ["any", "first", "all"]
    assert args.max_entries == 100
    assert args.num_mols == 500
    assert not args.warmup


def test_argument_validation_rejects_incoherent_backend_selection():
    base = dict(
        no_rdkit=False,
        no_nvmolkit=False,
        runs=1,
        reuse_count=10,
        num_mols=0,
        max_entries=0,
        batch_size=1024,
        workers=-1,
        prep_threads=-1,
        gpu=0,
        validate=True,
    )
    with pytest.raises(ValueError, match="disable both"):
        _validate_args(SimpleNamespace(**(base | {"no_rdkit": True, "no_nvmolkit": True})))
    with pytest.raises(ValueError, match="requires both"):
        _validate_args(SimpleNamespace(**(base | {"no_rdkit": True})))
    with pytest.raises(ValueError, match="runs"):
        _validate_args(SimpleNamespace(**(base | {"runs": 0})))
    with pytest.raises(ValueError, match="reuse_count"):
        _validate_args(SimpleNamespace(**(base | {"reuse_count": 0})))


@pytest.mark.parametrize(
    ("field", "value"),
    [("batch_size", 0), ("workers", -2), ("prep_threads", -2), ("gpu", -1)],
)
def test_argument_validation_rejects_invalid_execution_config(field, value):
    args = SimpleNamespace(
        no_rdkit=False,
        no_nvmolkit=False,
        runs=1,
        reuse_count=10,
        num_mols=0,
        max_entries=0,
        batch_size=1024,
        workers=-1,
        prep_threads=-1,
        gpu=0,
        validate=True,
    )
    setattr(args, field, value)

    with pytest.raises(ValueError, match=field):
        _validate_args(args)
