# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark persistent nvMolKit and RDKit substructure libraries.

Each backend configuration stages and finalizes one library, timed once, and
then measures every requested operation and query mode against it. Steady-state
throughput is expressed as queries and offered target-query pairs per second.

Results are validated against RDKit. ``--reference_cache`` stores complete
RDKit matches for a fixed target and query set, so repeated nvMolKit-only runs
(``--no-rdkit``) stay validated without re-running RDKit.

Examples:
    python substruct_library_bench.py --smiles molecules.smi --smarts queries.smarts
    python substruct_library_bench.py --smiles molecules.smi --smarts queries.smarts \
        --operations has get --algorithms gsi dfs --chunk_sizes 8192 65536
    python substruct_library_bench.py --smiles molecules.smi --smarts queries.smarts \
        --rdkit_holders mol cached-pattern --rdkit_threads 1 8
    python substruct_library_bench.py --pickle targets.pkl --query_smiles queries.smi --num_queries 1000 \
        --operations has count get --query_modes serial concurrent --no-rdkit --reference_cache ref.pkl
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import pickle
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from bench_utils import (
    Deadline,
    add_backend_selection_args,
    add_rdkit_max_seconds_arg,
    load_pickle,
    load_smarts,
    load_smiles,
    print_csv_rows,
    throughput_per_s,
    time_it,
    write_csv_rows,
)
from rdkit import Chem
from rdkit.Chem import rdSubstructLibrary


@dataclass(frozen=True)
class LifecycleMeasurement:
    """Timings and final query results for one persistent-library backend."""

    staging_ms: float
    staging_std_ms: float
    finalize_ms: float
    finalize_std_ms: float
    steady_ms: float
    steady_std_ms: float
    results: list[Any]
    completed_queries: int | None = None
    query_concurrency: int | None = None
    batches_in_flight_per_gpu: int | None = None
    workspace_bytes_per_query_per_gpu: int | None = None

    @property
    def amortized_ms(self) -> float:
        """Total staging, finalization, and repeated-query time."""
        return self.staging_ms + self.finalize_ms + self.steady_ms

    @property
    def amortized_std_ms(self) -> float:
        """Combined standard deviation for the amortized measurement."""
        return math.sqrt(self.staging_std_ms**2 + self.finalize_std_ms**2 + self.steady_std_ms**2)


def _run_nvmolkit_queries(
    library: Any,
    queries: Sequence[Any],
    operation: str,
    max_results: int,
    query_mode: str = "serial",
) -> list[Any]:
    def resolve(value: Any) -> Any:
        return value.result() if hasattr(value, "result") else value

    if operation not in {"has", "count", "get"}:
        raise ValueError(f"unsupported operation {operation!r}")

    if query_mode == "serial":
        # Preserve the ordinary call-and-wait usage pattern.
        results = []
        for query in queries:
            if operation == "has":
                value = library.hasMatch(query)
            elif operation == "count":
                value = library.countMatches(query)
            else:
                value = library.getMatches(query, maxResults=max_results)
            results.append(resolve(value))
    elif query_mode == "concurrent":
        if operation == "has":
            pending = [library.hasMatch(query) for query in queries]
        elif operation == "count":
            pending = [library.countMatches(query) for query in queries]
        else:
            pending = [library.getMatches(query, maxResults=max_results) for query in queries]
        results = [resolve(value) for value in pending]
    else:
        raise ValueError(f"unsupported query mode {query_mode!r}")
    return [list(value) for value in results] if operation == "get" else results


def _run_rdkit_queries(
    library: Any,
    queries: Sequence[Any],
    operation: str,
    max_results: int,
    num_threads: int,
    deadline: Deadline | None = None,
) -> list[Any]:
    match_options = {
        "recursionPossible": True,
        "useChirality": False,
        "useQueryQueryMatches": False,
        "numThreads": num_threads,
    }
    if operation not in {"has", "count", "get"}:
        raise ValueError(f"unsupported operation {operation!r}")

    results = []
    for query in queries:
        if deadline is not None and deadline.expired():
            break
        if operation == "has":
            results.append(library.HasMatch(query, **match_options))
        elif operation == "count":
            results.append(library.CountMatches(query, **match_options))
        else:
            results.append(list(library.GetMatches(query, maxResults=max_results, **match_options)))
    return results


@dataclass(frozen=True)
class BuiltLibrary:
    """A library staged and finalized once, reused by every measured query sweep."""

    library: Any
    staging_ms: float
    finalize_ms: float


def _build_library(
    *,
    make_library: Callable[[], Any],
    stage_library: Callable[[Any], None],
    finalize_library: Callable[[Any], None] | None,
    gpu_finalize: bool,
) -> BuiltLibrary:
    library = make_library()
    staging = time_it(lambda: stage_library(library), runs=1, warmups=0)
    finalization = None
    if finalize_library is not None:
        finalization = time_it(lambda: finalize_library(library), runs=1, warmups=0, gpu_sync=gpu_finalize)
    return BuiltLibrary(
        library=library,
        staging_ms=staging.mean_ms,
        finalize_ms=0.0 if finalization is None else finalization.mean_ms,
    )


def _measure_search(
    built: BuiltLibrary,
    search_library: Callable[[Any], list[Any]],
    *,
    runs: int,
    warmups: int,
    repetitions: int,
    gpu_search: bool,
) -> LifecycleMeasurement:
    results: list[Any] = []

    def search() -> None:
        nonlocal results
        for _ in range(repetitions):
            results = search_library(built.library)

    steady = time_it(search, runs=runs, warmups=warmups, gpu_sync=gpu_search)
    return LifecycleMeasurement(
        staging_ms=built.staging_ms,
        staging_std_ms=0.0,
        finalize_ms=built.finalize_ms,
        finalize_std_ms=0.0,
        steady_ms=steady.mean_ms,
        steady_std_ms=steady.std_ms,
        results=results,
    )


def _make_rdkit_library(holder: str) -> Any:
    if holder == "mol":
        return rdSubstructLibrary.SubstructLibrary(rdSubstructLibrary.MolHolder())
    if holder == "cached-pattern":
        # Pattern fingerprints are added in bulk by _stage_rdkit_library.
        return rdSubstructLibrary.SubstructLibrary(rdSubstructLibrary.CachedMolHolder())
    raise ValueError(f"unsupported RDKit holder {holder!r}")


def _stage_rdkit_library(library: Any, mols: Sequence[Any], holder: str, num_threads: int) -> None:
    mol_holder = library.GetMolHolder()
    for mol in mols:
        mol_holder.AddMol(mol)
    if holder == "cached-pattern":
        rdSubstructLibrary.AddPatterns(library, numThreads=num_threads)


def _measure_rdkit_with_deadline(
    built: BuiltLibrary,
    queries: Sequence[Any],
    *,
    operation: str,
    max_results: int,
    num_threads: int,
    runs: int,
    warmups: int,
    max_seconds: float,
) -> LifecycleMeasurement:
    results: list[Any] = []

    def search(deadline: Deadline) -> None:
        nonlocal results
        results = _run_rdkit_queries(built.library, queries, operation, max_results, num_threads, deadline)

    steady = time_it(
        search,
        runs=runs,
        warmups=warmups,
        max_seconds=max_seconds,
        progress_getter=lambda: len(results),
        progress_target=len(queries),
    )
    return LifecycleMeasurement(
        staging_ms=built.staging_ms,
        staging_std_ms=0.0,
        finalize_ms=0.0,
        finalize_std_ms=0.0,
        steady_ms=steady.mean_ms,
        steady_std_ms=steady.std_ms,
        results=results,
        completed_queries=steady.progress,
    )


def benchmark_rdkit(
    mols: Sequence[Any],
    queries: Sequence[Any],
    *,
    operations: Sequence[str],
    holder: str,
    num_threads: int,
    max_results: int,
    runs: int,
    warmups: int,
    repetitions: int,
    max_seconds: float = 0.0,
) -> dict[str, LifecycleMeasurement]:
    """Stage one RDKit library and measure every operation against it."""
    built = _build_library(
        make_library=lambda: _make_rdkit_library(holder),
        stage_library=lambda library: _stage_rdkit_library(library, mols, holder, num_threads),
        finalize_library=None,
        gpu_finalize=False,
    )
    measurements: dict[str, LifecycleMeasurement] = {}
    for operation in operations:
        if max_seconds > 0:
            measurements[operation] = _measure_rdkit_with_deadline(
                built,
                queries,
                operation=operation,
                max_results=max_results,
                num_threads=num_threads,
                runs=runs,
                warmups=warmups,
                max_seconds=max_seconds,
            )
        else:
            measurements[operation] = _measure_search(
                built,
                lambda library, operation=operation: _run_rdkit_queries(
                    library, queries, operation, max_results, num_threads
                ),
                runs=runs,
                warmups=warmups,
                repetitions=repetitions,
                gpu_search=False,
            )
    return measurements


def _rdkit_reference_results(mols: Sequence[Any], queries: Sequence[Any], num_threads: int) -> list[list[int]]:
    """Every matching target index per query, from a pattern-screened RDKit library."""
    library = _make_rdkit_library("cached-pattern")
    _stage_rdkit_library(library, mols, "cached-pattern", num_threads)
    return _run_rdkit_queries(library, queries, "get", -1, num_threads)


def _reference_for_operation(all_matches: Sequence[Sequence[int]], operation: str, max_results: int) -> list[Any]:
    """Derive has/count/get expectations from complete per-query match lists."""
    if operation == "has":
        return [bool(matches) for matches in all_matches]
    if operation == "count":
        return [len(matches) for matches in all_matches]
    if max_results > 0:
        return [list(matches[:max_results]) for matches in all_matches]
    return [list(matches) for matches in all_matches]


def _reference_key(mols: Sequence[Any], queries: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for mol in mols:
        digest.update(mol.ToBinary())
    digest.update(b"\0queries\0")
    for query in queries:
        digest.update(Chem.MolToSmarts(query).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def load_or_compute_reference(
    path: str, mols: Sequence[Any], queries: Sequence[Any], num_threads: int
) -> list[list[int]]:
    """Return cached complete RDKit matches for these targets and queries, computing them on a miss."""
    key = _reference_key(mols, queries)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            cached = pickle.load(fh)
        if cached.get("key") == key:
            print(f"PROGRESS reference loaded from {path}", flush=True)
            return cached["matches"]
        print(f"PROGRESS reference cache {path} does not match these inputs; recomputing", flush=True)
    matches = _rdkit_reference_results(mols, queries, num_threads)
    with open(path, "wb") as fh:
        pickle.dump({"key": key, "matches": matches}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"PROGRESS reference written to {path}", flush=True)
    return matches


def benchmark_nvmolkit(
    mols: Sequence[Any],
    queries: Sequence[Any],
    *,
    operations: Sequence[str],
    query_modes: Sequence[str],
    algorithm: str,
    chunk_size: int,
    batch_size: int,
    worker_threads: int,
    preprocessing_threads: int,
    gpu_ids: Sequence[int],
    max_results: int,
    runs: int,
    warmups: int,
    repetitions: int,
    use_pattern_fingerprints: bool = True,
) -> dict[tuple[str, str], LifecycleMeasurement]:
    """Build one nvMolKit library and measure every operation and query mode against it."""
    import torch

    from nvmolkit.substruct_library import SubstructLibrary
    from nvmolkit.substructure import SubstructSearchConfig

    torch.cuda.set_device(gpu_ids[0])
    config = SubstructSearchConfig(
        batchSize=batch_size,
        workerThreads=worker_threads,
        preprocessingThreads=preprocessing_threads,
        gpuIds=list(gpu_ids),
        algorithm=algorithm,
    )
    built = _build_library(
        make_library=lambda: SubstructLibrary(
            chunkSize=chunk_size,
            config=config,
            usePatternFingerprints=use_pattern_fingerprints,
        ),
        stage_library=lambda library: library.addMols(mols),
        finalize_library=lambda library: library.finalize(),
        gpu_finalize=True,
    )
    admission = {
        "query_concurrency": getattr(built.library, "queryConcurrency", 1),
        "batches_in_flight_per_gpu": getattr(built.library, "batchesInFlightPerGpu", 1),
        "workspace_bytes_per_query_per_gpu": getattr(built.library, "workspaceBytesPerQueryPerGpu", 0),
    }
    print(
        "PROGRESS admission " + " ".join(f"{name}={value}" for name, value in admission.items()),
        flush=True,
    )

    measurements: dict[tuple[str, str], LifecycleMeasurement] = {}
    for operation in operations:
        for query_mode in query_modes:
            measurement = _measure_search(
                built,
                lambda library, operation=operation, query_mode=query_mode: _run_nvmolkit_queries(
                    library, queries, operation, max_results, query_mode=query_mode
                ),
                runs=runs,
                warmups=warmups,
                repetitions=repetitions,
                gpu_search=True,
            )
            measurements[(operation, query_mode)] = replace(measurement, **admission)
    return measurements


def _result_row(
    *,
    backend: str,
    operation: str,
    measurement: LifecycleMeasurement,
    num_mols: int,
    num_queries: int,
    repetitions: int,
    **configuration: Any,
) -> dict[str, Any]:
    completed_queries = measurement.completed_queries if measurement.completed_queries is not None else num_queries
    query_work = completed_queries * repetitions
    pair_work = num_mols * query_work
    if operation == "has":
        positive_queries = sum(bool(result) for result in measurement.results)
        result_cardinality = None
    else:
        result_counts = [result if operation == "count" else len(result) for result in measurement.results]
        positive_queries = sum(count > 0 for count in result_counts)
        result_cardinality = sum(result_counts)
    return {
        "backend": backend,
        "operation": operation,
        "num_mols": num_mols,
        "num_queries": num_queries,
        "repetitions": repetitions,
        "positive_queries": positive_queries,
        "completed_queries": completed_queries,
        "completed_pairs": pair_work,
        "query_hit_rate": positive_queries / completed_queries if completed_queries else None,
        "result_cardinality": result_cardinality,
        "result_pair_density": (
            None
            if result_cardinality is None or completed_queries == 0
            else result_cardinality / (num_mols * completed_queries)
        ),
        **configuration,
        "staging_ms": measurement.staging_ms,
        "staging_std_ms": measurement.staging_std_ms,
        "finalize_ms": measurement.finalize_ms,
        "finalize_std_ms": measurement.finalize_std_ms,
        "steady_ms": measurement.steady_ms,
        "steady_std_ms": measurement.steady_std_ms,
        "amortized_ms": measurement.amortized_ms,
        "amortized_std_ms": measurement.amortized_std_ms,
        "steady_queries_per_s": throughput_per_s(query_work, measurement.steady_ms),
        "steady_offered_pairs_per_s": throughput_per_s(pair_work, measurement.steady_ms),
        "amortized_queries_per_s": throughput_per_s(query_work, measurement.amortized_ms),
        "amortized_offered_pairs_per_s": throughput_per_s(pair_work, measurement.amortized_ms),
    }


def _validate_results(nvmolkit_results: Sequence[Any], rdkit_results: Sequence[Any], operation: str) -> None:
    """Compare per-query results over the queries both runs completed.

    Deadline-bounded RDKit runs stop between queries, so either side may hold only a prefix of the query list.
    """
    for query_index, (nvmolkit_result, rdkit_result) in enumerate(zip(nvmolkit_results, rdkit_results)):
        actual = list(nvmolkit_result) if operation == "get" else nvmolkit_result
        expected = list(rdkit_result) if operation == "get" else rdkit_result
        if actual != expected:
            raise AssertionError(
                f"{operation} result differs for query {query_index}: nvMolKit={actual!r}, RDKit={expected!r}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent SubstructLibrary benchmark: nvMolKit vs RDKit")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--smiles", "-s", help="SMILES input file")
    inputs.add_argument("--pickle", help="Pickled RDKit molecule binaries")
    queries = parser.add_mutually_exclusive_group(required=True)
    queries.add_argument("--smarts", "-q", help="SMARTS query file")
    queries.add_argument("--query_smiles", help="SMILES molecule-query file; stereochemistry is ignored")
    parser.add_argument("--num_mols", "-n", type=int, default=0, help="Maximum target molecules; 0 means all")
    parser.add_argument(
        "--num_queries", type=int, default=0, help="Randomly sample this many queries (seeded); 0 means all"
    )
    parser.add_argument("--seed", type=int, default=42, help="Molecule sampling seed")
    parser.add_argument("--no_sanitize", dest="sanitize", action="store_false", default=True)
    parser.add_argument("--operations", "--operation", nargs="+", choices=["has", "count", "get"], default=["has"])
    parser.add_argument("--algorithms", "--algorithm", nargs="+", choices=["gsi", "dfs"], default=["gsi", "dfs"])
    parser.add_argument("--chunk_sizes", "--chunk_size", nargs="+", type=int, default=[65_536])
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument(
        "--prep_threads", type=int, default=-1, help="Preprocessing threads and input parsing processes (-1 or 0 = auto)"
    )
    gpu_selection = parser.add_mutually_exclusive_group()
    gpu_selection.add_argument("--gpu_ids", nargs="+", type=int, help="GPU IDs used by one internally sharded library")
    gpu_selection.add_argument("--gpu_id", type=int, help="Deprecated single-GPU spelling")
    parser.add_argument("--rdkit_holders", nargs="+", choices=["mol", "cached-pattern"], default=["mol"])
    parser.add_argument("--rdkit_threads", nargs="+", type=int, default=[-1])
    add_rdkit_max_seconds_arg(
        parser,
        extra_help="The persistent-library comparison checks the deadline between completed queries.",
    )
    parser.add_argument(
        "--max_results",
        "--maxResults",
        type=int,
        default=-1,
        help="Maximum results for get; 0 or -1 means all (default: -1)",
    )
    parser.add_argument("--runs", "-r", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1, help="Query sweeps per timed iteration")
    parser.add_argument(
        "--query_modes",
        "--query_mode",
        nargs="+",
        choices=["serial", "concurrent"],
        default=["serial"],
        help="Wait after each nvMolKit query, or resolve an asynchronously submitted sweep",
    )
    parser.add_argument(
        "--no_nvmolkit_pattern_fingerprints",
        dest="nvmolkit_pattern_fingerprints",
        action="store_false",
        default=True,
        help="Disable safe pattern-fingerprint screening for nvMolKit diagnostics",
    )
    parser.add_argument("--no_validate", dest="validate", action="store_false", default=True)
    parser.add_argument(
        "--reference_cache",
        help="Validate against complete RDKit matches cached at this path, computing them on a miss; "
        "allows validation without running the RDKit benchmark",
    )
    parser.add_argument("--output", "-o", help="Optional CSV output path")
    add_backend_selection_args(parser)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_mols < 0:
        raise ValueError("num_mols must be non-negative")
    if any(chunk_size <= 0 for chunk_size in args.chunk_sizes):
        raise ValueError("chunk_sizes must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.workers < -1:
        raise ValueError("workers must be -1 or non-negative")
    if args.prep_threads < -1:
        raise ValueError("prep_threads must be -1 or non-negative")
    gpu_ids = args.gpu_ids if args.gpu_ids is not None else [0 if args.gpu_id is None else args.gpu_id]
    if not gpu_ids or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("gpu_ids must be non-empty and non-negative")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("gpu_ids must be unique")
    if any(num_threads == 0 or num_threads < -1 for num_threads in args.rdkit_threads):
        raise ValueError("rdkit_threads entries must be -1 or positive")
    if args.rdkit_max_seconds < 0:
        raise ValueError("rdkit_max_seconds must be non-negative")
    if args.max_results < -1:
        raise ValueError("max_results must be -1, 0, or positive")
    if args.runs <= 0:
        raise ValueError("runs must be positive")
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if args.no_rdkit and args.no_nvmolkit:
        raise ValueError("cannot disable both backends")
    if args.num_queries < 0:
        raise ValueError("num_queries must be non-negative")
    if args.validate and args.no_nvmolkit and not args.reference_cache:
        raise ValueError("validating RDKit alone requires --reference_cache; pass --no_validate otherwise")
    if args.validate and args.no_rdkit and not args.reference_cache:
        raise ValueError("validation without RDKit requires --reference_cache; pass --no_validate otherwise")


def _load_molecules(args: argparse.Namespace) -> list[Any]:
    max_workers = args.prep_threads if args.prep_threads > 0 else None
    if args.pickle:
        return load_pickle(args.pickle, args.num_mols, seed=args.seed, max_workers=max_workers)
    return load_smiles(args.smiles, args.num_mols, args.sanitize, seed=args.seed, max_workers=max_workers)


def _load_queries(args: argparse.Namespace) -> list[Any]:
    if args.smarts:
        queries, _ = load_smarts(args.smarts)
        if 0 < args.num_queries < len(queries):
            queries = random.Random(args.seed).sample(queries, args.num_queries)
        return queries
    queries = load_smiles(
        args.query_smiles,
        args.num_queries,
        args.sanitize,
        seed=args.seed,
        max_workers=args.prep_threads if args.prep_threads > 0 else None,
    )
    for query in queries:
        Chem.RemoveStereochemistry(query)
    return queries


def main() -> None:
    args = _build_parser().parse_args()
    _validate_args(args)
    mols = _load_molecules(args)
    queries = _load_queries(args)
    if not mols:
        raise ValueError("no valid target molecules loaded")
    if not queries:
        raise ValueError("no valid queries loaded")
    max_results = -1 if args.max_results == 0 else args.max_results
    gpu_ids = args.gpu_ids if args.gpu_ids is not None else [0 if args.gpu_id is None else args.gpu_id]

    references: dict[str, list[Any]] = {}
    if args.validate and args.reference_cache:
        all_matches = load_or_compute_reference(args.reference_cache, mols, queries, args.rdkit_threads[0])
        references = {
            operation: _reference_for_operation(all_matches, operation, max_results) for operation in args.operations
        }

    rows: list[dict[str, Any]] = []
    if not args.no_rdkit:
        for holder in args.rdkit_holders:
            for num_threads in args.rdkit_threads:
                measurements = benchmark_rdkit(
                    mols,
                    queries,
                    operations=args.operations,
                    holder=holder,
                    num_threads=num_threads,
                    max_results=max_results,
                    runs=args.runs,
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                    max_seconds=args.rdkit_max_seconds,
                )
                for operation, measurement in measurements.items():
                    if args.validate:
                        reference = references.get(operation)
                        if reference is not None:
                            _validate_results(measurement.results, reference, operation)
                        if not args.reference_cache and (
                            reference is None or len(measurement.results) > len(reference)
                        ):
                            references[operation] = measurement.results
                    rows.append(
                        _result_row(
                            backend="rdkit-substruct-library",
                            operation=operation,
                            measurement=measurement,
                            num_mols=len(mols),
                            num_queries=len(queries),
                            repetitions=args.repetitions,
                            holder=holder,
                            rdkit_threads=num_threads,
                            max_results=max_results,
                        )
                    )
                print(f"PROGRESS completed backend=rdkit holder={holder} threads={num_threads}", flush=True)

    if not args.no_nvmolkit:
        for algorithm in args.algorithms:
            for chunk_size in args.chunk_sizes:
                measurements = benchmark_nvmolkit(
                    mols,
                    queries,
                    operations=args.operations,
                    query_modes=args.query_modes,
                    algorithm=algorithm,
                    chunk_size=chunk_size,
                    batch_size=args.batch_size,
                    worker_threads=args.workers,
                    preprocessing_threads=args.prep_threads,
                    gpu_ids=gpu_ids,
                    max_results=max_results,
                    runs=args.runs,
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                    use_pattern_fingerprints=args.nvmolkit_pattern_fingerprints,
                )
                for (operation, query_mode), measurement in measurements.items():
                    if args.validate:
                        reference = references.get(operation)
                        if reference is None:
                            raise RuntimeError("validation requires an RDKit reference result")
                        _validate_results(measurement.results, reference, operation)
                        if len(reference) < len(queries):
                            print(
                                f"VALIDATION partial operation={operation} algorithm={algorithm} "
                                f"chunk_size={chunk_size}: compared {len(reference)}/{len(queries)} "
                                "queries completed by RDKit before its deadline",
                                flush=True,
                            )
                    rows.append(
                        _result_row(
                            backend="nvmolkit-substruct-library",
                            operation=operation,
                            measurement=measurement,
                            num_mols=len(mols),
                            num_queries=len(queries),
                            repetitions=args.repetitions,
                            algorithm=algorithm,
                            chunk_size=chunk_size,
                            batch_size=args.batch_size,
                            workers=args.workers,
                            prep_threads=args.prep_threads,
                            gpu_ids=",".join(str(gpu_id) for gpu_id in gpu_ids),
                            num_gpus=len(gpu_ids),
                            query_mode=query_mode,
                            pattern_fingerprints=args.nvmolkit_pattern_fingerprints,
                            query_concurrency=measurement.query_concurrency,
                            batches_in_flight_per_gpu=measurement.batches_in_flight_per_gpu,
                            workspace_bytes_per_query_per_gpu=measurement.workspace_bytes_per_query_per_gpu,
                            max_results=max_results,
                        )
                    )
                print(
                    f"PROGRESS completed backend=nvmolkit algorithm={algorithm} chunk_size={chunk_size} "
                    f"gpu_ids={gpu_ids}",
                    flush=True,
                )

    print_csv_rows(rows)
    write_csv_rows(rows, args.output)


if __name__ == "__main__":
    main()
