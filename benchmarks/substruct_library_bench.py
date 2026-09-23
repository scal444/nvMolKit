# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark persistent nvMolKit and RDKit substructure libraries.

The benchmark reports library staging, nvMolKit's explicit ``finalize()``
upload, repeated query time, and the amortized total. Steady-state throughput
is expressed as queries and offered target-query pairs per second.

Examples:
    python substruct_library_bench.py --smiles molecules.smi --smarts queries.smarts
    python substruct_library_bench.py --smiles molecules.smi --smarts queries.smarts \
        --operations has get --algorithms gsi dfs --chunk_sizes 8192 65536
    python substruct_library_bench.py --smiles molecules.smi --smarts queries.smarts \
        --rdkit_holders mol cached-pattern --rdkit_threads 1 8
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from bench_utils import (
    add_backend_selection_args,
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

    @property
    def amortized_ms(self) -> float:
        """Total staging, finalization, and repeated-query time."""
        return self.staging_ms + self.finalize_ms + self.steady_ms

    @property
    def amortized_std_ms(self) -> float:
        """Combined standard deviation for the amortized measurement."""
        return math.sqrt(self.staging_std_ms**2 + self.finalize_std_ms**2 + self.steady_std_ms**2)


def _run_nvmolkit_queries(library: Any, queries: Sequence[Any], operation: str, max_results: int) -> list[Any]:
    if operation == "has":
        return [library.hasMatch(query) for query in queries]
    if operation == "count":
        return [library.countMatches(query) for query in queries]
    if operation == "get":
        return [list(library.getMatches(query, maxResults=max_results)) for query in queries]
    raise ValueError(f"unsupported operation {operation!r}")


def _run_rdkit_queries(
    library: Any,
    queries: Sequence[Any],
    operation: str,
    max_results: int,
    num_threads: int,
) -> list[Any]:
    match_options = {
        "recursionPossible": True,
        "useChirality": False,
        "useQueryQueryMatches": False,
        "numThreads": num_threads,
    }
    if operation == "has":
        return [library.HasMatch(query, **match_options) for query in queries]
    if operation == "count":
        return [library.CountMatches(query, **match_options) for query in queries]
    if operation == "get":
        return [list(library.GetMatches(query, maxResults=max_results, **match_options)) for query in queries]
    raise ValueError(f"unsupported operation {operation!r}")


def _benchmark_lifecycle(
    *,
    make_library: Callable[[], Any],
    stage_library: Callable[[Any], None],
    finalize_library: Callable[[Any], None] | None,
    search_library: Callable[[Any], list[Any]],
    runs: int,
    warmups: int,
    repetitions: int,
    gpu_finalize: bool,
    gpu_search: bool,
) -> LifecycleMeasurement:
    current_library: Any = None
    results: list[Any] = []

    def reset_library() -> None:
        nonlocal current_library
        current_library = make_library()

    def stage() -> None:
        stage_library(current_library)

    staging = time_it(stage, runs=runs, warmups=0, setup=reset_library)

    if finalize_library is not None:

        def reset_staged_library() -> None:
            reset_library()
            stage()

        def finalize() -> None:
            finalize_library(current_library)

        finalization = time_it(
            finalize,
            runs=runs,
            warmups=0,
            setup=reset_staged_library,
            gpu_sync=gpu_finalize,
        )
    else:
        finalization = None

    def search() -> None:
        nonlocal results
        for _ in range(repetitions):
            results = search_library(current_library)

    steady = time_it(search, runs=runs, warmups=warmups, gpu_sync=gpu_search)
    return LifecycleMeasurement(
        staging_ms=staging.mean_ms,
        staging_std_ms=staging.std_ms,
        finalize_ms=0.0 if finalization is None else finalization.mean_ms,
        finalize_std_ms=0.0 if finalization is None else finalization.std_ms,
        steady_ms=steady.mean_ms,
        steady_std_ms=steady.std_ms,
        results=results,
    )


def _make_rdkit_library(holder: str) -> Any:
    if holder == "mol":
        return rdSubstructLibrary.SubstructLibrary(rdSubstructLibrary.MolHolder())
    if holder == "cached-pattern":
        return rdSubstructLibrary.SubstructLibrary(
            rdSubstructLibrary.CachedMolHolder(),
            rdSubstructLibrary.PatternHolder(),
        )
    raise ValueError(f"unsupported RDKit holder {holder!r}")


def benchmark_rdkit(
    mols: Sequence[Any],
    queries: Sequence[Any],
    *,
    operation: str,
    holder: str,
    num_threads: int,
    max_results: int,
    runs: int,
    warmups: int,
    repetitions: int,
) -> LifecycleMeasurement:
    return _benchmark_lifecycle(
        make_library=lambda: _make_rdkit_library(holder),
        stage_library=lambda library: [library.AddMol(mol) for mol in mols],
        finalize_library=None,
        search_library=lambda library: _run_rdkit_queries(
            library,
            queries,
            operation,
            max_results,
            num_threads,
        ),
        runs=runs,
        warmups=warmups,
        repetitions=repetitions,
        gpu_finalize=False,
        gpu_search=False,
    )


def _rdkit_reference_results(
    mols: Sequence[Any],
    queries: Sequence[Any],
    operation: str,
    max_results: int,
) -> list[Any]:
    library = _make_rdkit_library("mol")
    for mol in mols:
        library.AddMol(mol)
    return _run_rdkit_queries(library, queries, operation, max_results, num_threads=-1)


def benchmark_nvmolkit(
    mols: Sequence[Any],
    queries: Sequence[Any],
    *,
    operation: str,
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
) -> LifecycleMeasurement:
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
    return _benchmark_lifecycle(
        make_library=lambda: SubstructLibrary(chunkSize=chunk_size, config=config),
        stage_library=lambda library: library.addMols(mols),
        finalize_library=lambda library: library.finalize(),
        search_library=lambda library: _run_nvmolkit_queries(library, queries, operation, max_results),
        runs=runs,
        warmups=warmups,
        repetitions=repetitions,
        gpu_finalize=True,
        gpu_search=True,
    )


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
    query_work = num_queries * repetitions
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
        "query_hit_rate": positive_queries / num_queries,
        "result_cardinality": result_cardinality,
        "result_pair_density": None if result_cardinality is None else result_cardinality / (num_mols * num_queries),
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
    if len(nvmolkit_results) != len(rdkit_results):
        raise AssertionError(f"result length differs: nvMolKit={len(nvmolkit_results)}, RDKit={len(rdkit_results)}")
    for query_index, (actual, expected) in enumerate(zip(nvmolkit_results, rdkit_results, strict=True)):
        if operation == "get":
            actual = list(actual)
            expected = list(expected)
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
    parser.add_argument("--seed", type=int, default=42, help="Molecule sampling seed")
    parser.add_argument("--no_sanitize", dest="sanitize", action="store_false", default=True)
    parser.add_argument("--operations", "--operation", nargs="+", choices=["has", "count", "get"], default=["has"])
    parser.add_argument("--algorithms", "--algorithm", nargs="+", choices=["gsi", "dfs"], default=["gsi", "dfs"])
    parser.add_argument("--chunk_sizes", "--chunk_size", nargs="+", type=int, default=[65_536])
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--prep_threads", type=int, default=-1)
    gpu_selection = parser.add_mutually_exclusive_group()
    gpu_selection.add_argument("--gpu_ids", nargs="+", type=int, help="GPU IDs used by one internally sharded library")
    gpu_selection.add_argument("--gpu_id", type=int, help="Deprecated single-GPU spelling")
    parser.add_argument("--rdkit_holders", nargs="+", choices=["mol", "cached-pattern"], default=["mol"])
    parser.add_argument("--rdkit_threads", nargs="+", type=int, default=[-1])
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
    parser.add_argument("--no_validate", dest="validate", action="store_false", default=True)
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
    if args.validate and (args.no_rdkit or args.no_nvmolkit):
        raise ValueError("validation requires both backends; pass --no_validate for a single backend")


def _load_molecules(args: argparse.Namespace) -> list[Any]:
    if args.pickle:
        return load_pickle(args.pickle, args.num_mols, seed=args.seed)
    return load_smiles(args.smiles, args.num_mols, args.sanitize, seed=args.seed)


def _load_queries(args: argparse.Namespace) -> list[Any]:
    if args.smarts:
        queries, _ = load_smarts(args.smarts)
        return queries
    queries = load_smiles(args.query_smiles, 0, args.sanitize, seed=args.seed)
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

    rows: list[dict[str, Any]] = []
    for operation in args.operations:
        reference_results = None
        if not args.no_rdkit:
            for holder in args.rdkit_holders:
                for num_threads in args.rdkit_threads:
                    measurement = benchmark_rdkit(
                        mols,
                        queries,
                        operation=operation,
                        holder=holder,
                        num_threads=num_threads,
                        max_results=max_results,
                        runs=args.runs,
                        warmups=args.warmups,
                        repetitions=args.repetitions,
                    )
                    if args.validate:
                        if reference_results is None:
                            reference_results = measurement.results
                        else:
                            _validate_results(measurement.results, reference_results, operation)
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
                    print(
                        f"PROGRESS completed backend=rdkit operation={operation} "
                        f"holder={holder} threads={num_threads}",
                        flush=True,
                    )

        if not args.no_nvmolkit:
            for algorithm in args.algorithms:
                for chunk_size in args.chunk_sizes:
                    measurement = benchmark_nvmolkit(
                        mols,
                        queries,
                        operation=operation,
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
                    )
                    if args.validate:
                        if reference_results is None:
                            raise RuntimeError("validation requires an RDKit reference result")
                        _validate_results(measurement.results, reference_results, operation)
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
                            max_results=max_results,
                        )
                    )
                    print(
                        f"PROGRESS completed backend=nvmolkit operation={operation} "
                        f"algorithm={algorithm} chunk_size={chunk_size} gpu_ids={gpu_ids}",
                        flush=True,
                    )

    print_csv_rows(rows)
    write_csv_rows(rows, args.output)


if __name__ == "__main__":
    main()
