# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark persistent nvMolKit and RDKit filter catalogs.

The benchmark reports catalog staging, nvMolKit finalization, steady-state
search, and end-to-end amortized timings separately. Search throughput is
reported in targets per second because ``any`` and ``first`` intentionally
avoid evaluating every target-entry pair.

Custom catalogs use a text file containing one SMARTS per line. A tab-separated
second column may specify the trigger count and a third column the description::

    [N+](=O)[O-]\t1\tnitro
    [#6]\t3\tat least three carbons

Usage::

    python filter_catalog_bench.py --smiles molecules.smi --preset BRENK
    python filter_catalog_bench.py --smiles molecules.smi --smarts filters.tsv \
        --operation any first all
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from bench_utils import (
    add_backend_selection_args,
    load_pickle,
    load_smiles,
    print_csv_rows,
    throughput_per_s,
    time_it,
    write_csv_rows,
)
from rdkit import Chem
from rdkit.Chem import FilterCatalog as RDFilterCatalog


@dataclass(frozen=True)
class FilterDefinition:
    """One custom filter definition."""

    smarts: str
    trigger_count: int
    description: str


@dataclass
class CatalogRun:
    """Lifecycle measurements and the result of the final search run."""

    stage_ms: float
    stage_std_ms: float
    finalize_ms: float | None
    finalize_std_ms: float | None
    search_ms: float
    search_std_ms: float
    amortized_ms: float
    amortized_std_ms: float
    results: list[Any]
    entry_descriptions: list[str]


_PRESET_NAMES = (
    "PAINS_A",
    "PAINS_B",
    "PAINS_C",
    "PAINS",
    "BRENK",
    "NIH",
    "ZINC",
    "CHEMBL_GLAXO",
    "CHEMBL_DUNDEE",
    "CHEMBL_BMS",
    "CHEMBL_SURECHEMBL",
    "CHEMBL_MLSMR",
    "CHEMBL_INPHARMATICA",
    "CHEMBL_LINT",
    "CHEMBL",
    "ALL",
)

_RDKIT_PRESET_ALIASES = {
    "CHEMBL_GLAXO": "CHEMBL_Glaxo",
    "CHEMBL_DUNDEE": "CHEMBL_Dundee",
    "CHEMBL_SURECHEMBL": "CHEMBL_SureChEMBL",
    "CHEMBL_INPHARMATICA": "CHEMBL_Inpharmatica",
}


def load_filter_definitions(path: str, max_entries: int = 0) -> list[FilterDefinition]:
    """Load SMARTS, trigger count, and description fields from a text file."""
    definitions: list[FilterDefinition] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split("\t")
            smarts = fields[0].strip()
            trigger_count = int(fields[1]) if len(fields) > 1 and fields[1].strip() else 1
            description = fields[2].strip() if len(fields) > 2 else f"custom_{len(definitions)}"
            if trigger_count <= 0:
                raise ValueError(f"line {line_number}: trigger count must be positive")
            if Chem.MolFromSmarts(smarts) is None:
                raise ValueError(f"line {line_number}: invalid SMARTS {smarts!r}")
            definitions.append(FilterDefinition(smarts, trigger_count, description))
            if max_entries > 0 and len(definitions) == max_entries:
                break
    if not definitions:
        raise ValueError("custom filter file contains no valid definitions")
    return definitions


def _rdkit_preset(name: str):
    member_name = _RDKIT_PRESET_ALIASES.get(name, name)
    return getattr(RDFilterCatalog.FilterCatalogParams.FilterCatalogs, member_name)


def _make_rdkit_catalog(preset: str | None, definitions: list[FilterDefinition]):
    if preset is not None:
        return RDFilterCatalog.FilterCatalog(_rdkit_preset(preset))
    catalog = RDFilterCatalog.FilterCatalog()
    for definition in definitions:
        matcher = RDFilterCatalog.SmartsMatcher(
            definition.description,
            Chem.MolFromSmarts(definition.smarts),
            definition.trigger_count,
        )
        catalog.AddEntry(RDFilterCatalog.FilterCatalogEntry(definition.description, matcher))
    return catalog


def _make_nvmolkit_catalog(preset: str | None, definitions: list[FilterDefinition], config):
    from nvmolkit.filter_catalog import FilterCatalog, FilterCatalogPreset

    catalog = FilterCatalog(config=config)
    if preset is not None:
        catalog.addPreset(FilterCatalogPreset[preset])
    else:
        for definition in definitions:
            catalog.addSmarts(
                definition.smarts,
                definition.description,
                definition.trigger_count,
            )
    return catalog


def _catalog_descriptions(catalog, backend: str) -> list[str]:
    if backend == "nvmolkit":
        return [catalog.getEntry(index).description for index in range(len(catalog))]
    return [catalog.GetEntryWithIdx(index).GetDescription() for index in range(catalog.GetNumEntries())]


def _search(catalog, molecules: list[Chem.Mol], operation: str, backend: str) -> list[Any]:
    if backend == "nvmolkit":
        function = {
            "any": catalog.hasMatch,
            "first": catalog.getFirstMatch,
            "all": catalog.getMatches,
        }[operation]
        return list(function(molecules))

    function = {
        "any": catalog.HasMatch,
        "first": catalog.GetFirstMatch,
        "all": catalog.GetMatches,
    }[operation]
    return [function(molecule) for molecule in molecules]


def benchmark_catalog(
    make_catalog: Callable[[], Any],
    molecules: list[Chem.Mol],
    operation: str,
    backend: str,
    runs: int,
    warmups: int,
    *,
    gpu_sync: bool,
    reuse_count: int = 10,
) -> CatalogRun:
    """Measure each catalog lifecycle phase without charging setup to search."""
    staged_catalog = None

    def stage() -> None:
        nonlocal staged_catalog
        staged_catalog = make_catalog()

    stage_timing = time_it(stage, runs=runs, warmups=warmups, gpu_sync=False)

    finalize_timing = None
    if backend == "nvmolkit":

        def finalize_setup() -> None:
            stage()

        def finalize() -> None:
            staged_catalog.finalize()

        finalize_timing = time_it(
            finalize,
            runs=runs,
            warmups=warmups,
            gpu_sync=gpu_sync,
            setup=finalize_setup,
        )

    catalog = make_catalog()
    if backend == "nvmolkit":
        catalog.finalize()
    results: list[Any] = []

    def search() -> None:
        nonlocal results
        results = _search(catalog, molecules, operation, backend)

    search_timing = time_it(search, runs=runs, warmups=warmups, gpu_sync=gpu_sync)

    def end_to_end() -> None:
        nonlocal results
        fresh_catalog = make_catalog()
        if backend == "nvmolkit":
            fresh_catalog.finalize()
        for _ in range(reuse_count):
            results = _search(fresh_catalog, molecules, operation, backend)

    amortized_timing = time_it(end_to_end, runs=runs, warmups=warmups, gpu_sync=gpu_sync)
    return CatalogRun(
        stage_ms=stage_timing.mean_ms,
        stage_std_ms=stage_timing.std_ms,
        finalize_ms=finalize_timing.mean_ms if finalize_timing is not None else None,
        finalize_std_ms=finalize_timing.std_ms if finalize_timing is not None else None,
        search_ms=search_timing.mean_ms,
        search_std_ms=search_timing.std_ms,
        amortized_ms=amortized_timing.mean_ms / reuse_count,
        amortized_std_ms=amortized_timing.std_ms / reuse_count,
        results=results,
        entry_descriptions=_catalog_descriptions(catalog, backend),
    )


def _canonical_results(run: CatalogRun, operation: str, backend: str) -> list[Any]:
    if operation == "any":
        return [bool(value) for value in run.results]
    if backend == "nvmolkit":
        if operation == "first":
            return [None if entry_id is None else run.entry_descriptions[entry_id] for entry_id in run.results]
        return [[run.entry_descriptions[entry_id] for entry_id in entry_ids] for entry_ids in run.results]
    if operation == "first":
        return [None if entry is None else entry.GetDescription() for entry in run.results]
    return [[entry.GetDescription() for entry in entries] for entries in run.results]


def rdkit_reference_ids(catalog, molecules: list[Chem.Mol]) -> list[list[int]]:
    """Evaluate every RDKit entry to produce exact stable IDs outside timing."""
    entries = [catalog.GetEntryWithIdx(index) for index in range(catalog.GetNumEntries())]
    return [[index for index, entry in enumerate(entries) if entry.HasFilterMatch(molecule)] for molecule in molecules]


def validate_results(nvmolkit_run: CatalogRun, expected_ids: list[list[int]], operation: str) -> None:
    """Require exact target-level IDs and ordering from nvMolKit."""
    if operation == "any":
        actual = [bool(value) for value in nvmolkit_run.results]
        expected: list[Any] = [bool(entry_ids) for entry_ids in expected_ids]
    elif operation == "first":
        actual = list(nvmolkit_run.results)
        expected = [entry_ids[0] if entry_ids else None for entry_ids in expected_ids]
    else:
        actual = [list(entry_ids) for entry_ids in nvmolkit_run.results]
        expected = expected_ids
    if actual != expected:
        if len(actual) != len(expected):
            raise AssertionError(f"result length mismatch: nvMolKit={len(actual)}, RDKit={len(expected)}")
        mismatch = next(index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1])
        raise AssertionError(
            f"result mismatch for target {mismatch}: nvMolKit={actual[mismatch]!r}, RDKit={expected[mismatch]!r}"
        )


def result_rows(
    run: CatalogRun,
    *,
    backend: str,
    operation: str,
    catalog_name: str,
    num_targets: int,
    known_fallback_entries: int | str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert lifecycle measurements into stable CSV rows."""
    entry_count = len(run.entry_descriptions)
    canonical = _canonical_results(run, operation, backend)
    if operation == "any":
        matched_targets = sum(canonical)
        returned_matches: int | str = matched_targets
    elif operation == "first":
        matched_targets = sum(value is not None for value in canonical)
        returned_matches = matched_targets
    else:
        matched_targets = sum(bool(value) for value in canonical)
        returned_matches = sum(len(value) for value in canonical)
    trigger_count_one_entries: int | str = (
        entry_count - known_fallback_entries if isinstance(known_fallback_entries, int) else "N/A"
    )
    common = {
        "method": "rdkit-python-loop" if backend == "rdkit" else backend,
        "catalog": catalog_name,
        "operation": operation,
        "num_targets": num_targets,
        "num_entries": entry_count,
        "nominal_target_entry_pairs": num_targets * entry_count,
        "matched_targets": matched_targets,
        "match_fraction": matched_targets / num_targets if num_targets else 0.0,
        "returned_matches": returned_matches,
        "known_trigger_count_fallback_entries": known_fallback_entries if backend == "nvmolkit" else "N/A",
        "trigger_count_one_entries": trigger_count_one_entries if backend == "nvmolkit" else "N/A",
        **config,
    }
    phases = [
        ("stage", run.stage_ms, run.stage_std_ms, entry_count, "entries"),
        ("steady_search", run.search_ms, run.search_std_ms, num_targets, "targets"),
        ("amortized", run.amortized_ms, run.amortized_std_ms, num_targets, "targets"),
    ]
    if run.finalize_ms is not None and run.finalize_std_ms is not None:
        phases.insert(1, ("finalize", run.finalize_ms, run.finalize_std_ms, entry_count, "entries"))
    rows = []
    for phase, elapsed_ms, std_ms, items, unit in phases:
        rows.append(
            {
                **common,
                "phase": phase,
                "time_ms": elapsed_ms,
                "std_ms": std_ms,
                "throughput_unit": f"{unit}/s",
                "throughput": throughput_per_s(items, elapsed_ms),
            }
        )
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent filter catalog benchmark: nvMolKit vs RDKit")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--smiles", "-s", help="Path to a SMILES file")
    inputs.add_argument("--pickle", help="Path to pickled RDKit molecule binaries")
    catalogs = parser.add_mutually_exclusive_group(required=True)
    catalogs.add_argument("--preset", choices=_PRESET_NAMES, help="RDKit built-in filter collection")
    catalogs.add_argument("--smarts", help="Custom filter definition file")
    parser.add_argument("--max_entries", type=int, default=0, help="Maximum custom entries; 0 loads all")
    parser.add_argument("--num_mols", "-n", type=int, default=0, help="Maximum targets; 0 loads all")
    parser.add_argument("--seed", type=int, default=42, help="Input sampling seed")
    parser.add_argument("--sanitize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--operation", choices=["any", "first", "all"], nargs="+", default=["any"])
    parser.add_argument("--runs", "-r", type=int, default=3)
    parser.add_argument(
        "--reuse_count",
        type=int,
        default=10,
        help="Search batches used to amortize catalog construction (default: 10)",
    )
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    add_backend_selection_args(parser)
    parser.add_argument("--batch_size", "-b", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--prep_threads", type=int, default=-1)
    parser.add_argument("--algorithm", choices=["gsi", "dfs"], default="gsi")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", "-o", help="Optional CSV output path")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.no_rdkit and args.no_nvmolkit:
        raise ValueError("cannot disable both nvMolKit and RDKit")
    if args.runs <= 0:
        raise ValueError("runs must be positive")
    if args.reuse_count <= 0:
        raise ValueError("reuse_count must be positive")
    if args.num_mols < 0 or args.max_entries < 0:
        raise ValueError("num_mols and max_entries must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.workers < -1:
        raise ValueError("workers must be -1 or non-negative")
    if args.prep_threads < -1:
        raise ValueError("prep_threads must be -1 or non-negative")
    if args.gpu < 0:
        raise ValueError("gpu must be non-negative")
    if args.validate and (args.no_rdkit or args.no_nvmolkit):
        raise ValueError("validation requires both nvMolKit and RDKit")


def main() -> None:
    args = _build_parser().parse_args()
    try:
        _validate_args(args)
        definitions = load_filter_definitions(args.smarts, args.max_entries) if args.smarts else []
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    input_path = args.smiles or args.pickle
    molecules = (
        load_smiles(input_path, args.num_mols, args.sanitize, seed=args.seed)
        if args.smiles
        else load_pickle(input_path, args.num_mols, seed=args.seed)
    )
    if not molecules:
        raise SystemExit("Error: no valid target molecules were loaded")

    catalog_name = args.preset or Path(args.smarts).name
    known_fallback_entries: int | str = (
        sum(definition.trigger_count > 1 for definition in definitions) if definitions else "N/A"
    )
    workload_fields = {
        "runs": args.runs,
        "reuse_count": args.reuse_count,
        "warmup": args.warmup,
        "input_file": input_path,
        "seed": args.seed,
        "min_target_atoms": min(molecule.GetNumAtoms() for molecule in molecules),
        "max_target_atoms": max(molecule.GetNumAtoms() for molecule in molecules),
        "mean_target_atoms": sum(molecule.GetNumAtoms() for molecule in molecules) / len(molecules),
    }
    native_config_fields = {
        "batch_size": args.batch_size,
        "workers": args.workers,
        "prep_threads": args.prep_threads,
        "algorithm": args.algorithm,
        "gpu": args.gpu,
    }
    native_config = None
    if not args.no_nvmolkit:
        import torch

        from nvmolkit.substructure import SubstructSearchConfig

        torch.cuda.set_device(args.gpu)
        native_config = SubstructSearchConfig(
            batchSize=args.batch_size,
            workerThreads=args.workers,
            preprocessingThreads=args.prep_threads,
            algorithm=args.algorithm,
            gpuIds=[args.gpu],
        )

    rows: list[dict[str, Any]] = []
    expected_ids = None
    if args.validate:
        reference_catalog = _make_rdkit_catalog(args.preset, definitions)
        expected_ids = rdkit_reference_ids(reference_catalog, molecules)
    for operation in dict.fromkeys(args.operation):
        runs: dict[str, CatalogRun] = {}
        if not args.no_nvmolkit:
            runs["nvmolkit"] = benchmark_catalog(
                lambda: _make_nvmolkit_catalog(args.preset, definitions, native_config),
                molecules,
                operation,
                "nvmolkit",
                args.runs,
                int(args.warmup),
                gpu_sync=True,
                reuse_count=args.reuse_count,
            )
        if not args.no_rdkit:
            runs["rdkit"] = benchmark_catalog(
                lambda: _make_rdkit_catalog(args.preset, definitions),
                molecules,
                operation,
                "rdkit",
                args.runs,
                int(args.warmup),
                gpu_sync=False,
                reuse_count=args.reuse_count,
            )
        if args.validate:
            validate_results(runs["nvmolkit"], expected_ids, operation)
        for backend, run in runs.items():
            rows.extend(
                result_rows(
                    run,
                    backend=backend,
                    operation=operation,
                    catalog_name=catalog_name,
                    num_targets=len(molecules),
                    known_fallback_entries=known_fallback_entries,
                    config=workload_fields | (native_config_fields if backend == "nvmolkit" else {}),
                )
            )
        gc.collect()

    print_csv_rows(rows)
    if args.output:
        write_csv_rows(rows, args.output)


if __name__ == "__main__":
    main()
