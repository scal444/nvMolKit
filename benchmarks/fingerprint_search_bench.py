# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark FingerprintSearchIndex top-k search against FPSim2 and against itself without pruning.

Methods:
    nvmolkit           FingerprintSearchIndex.search with popcount pruning (the default).
    nvmolkit_noprune   The same kernel scanning every library row; isolates the effect of pruning.
    fpsim2_cpu         FPSim2Engine.top_k, one query at a time, with --fpsim2-workers threads per query.
    fpsim2_cuda        FPSim2CudaEngine.similarity. It only supports threshold searches, so it runs for
                       thresholds > 0 and returns every hit above the threshold rather than the top k.

Every cell is a (library size, method, number of queries, k, threshold) combination. Queries are held-out molecules
that are not in the library. A cell whose first run exceeds --cell-timeout is recorded as a timeout, and larger query
counts for that method and library are skipped. CPU cells are skipped when the first query predicts a total above the
cap. Index or database build time is recorded separately from search time.

Run timing cells under the shared GPU lock so that concurrent GPU jobs do not distort them:
    python fingerprint_search_bench.py --input enamine.cxsmiles --gpu-lock ~/experiments/kmeans/gpu_perf.lock

FPSim2 is optional; its methods are skipped if it cannot be imported. fpsim2_cuda also needs cupy.
"""

import argparse
import contextlib
import fcntl
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
from fingerprint_search_data import load_packed_fingerprints, to_rdkit_bitvects

from nvmolkit.similarity import FingerprintSearchIndex

ALL_METHODS = ["nvmolkit", "nvmolkit_noprune", "fpsim2_cpu", "fpsim2_cuda"]


@contextlib.contextmanager
def gpu_lock(path):
    """Hold an exclusive flock on path for the duration of one timed cell (no-op without a path)."""
    if not path:
        yield
        return
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def write_fpsim2_db(words, radius, path):
    """Write an FPSim2 database for the packed library, reusing FPSim2's own row encoding and sorting."""
    import tables as tb
    from FPSim2.io.backends.pytables import BATCH_WRITE_SIZE, create_schema, sort_db_file
    from FPSim2.io.chem import process_fp

    fp_size = words.shape[1] * 32
    fp_params = {"radius": radius, "fpSize": fp_size}
    filters = tb.Filters(complib="blosc2", complevel=9, fletcher32=False)
    with tb.open_file(path, mode="w") as handle:
        table = handle.create_table(handle.root, "fps", create_schema(fp_size), "Table storing fps", filters=filters)
        config = handle.create_vlarray(handle.root, "config", atom=tb.ObjectAtom())
        for value in ("Morgan", fp_params, __import__("rdkit").__version__, __import__("FPSim2").__version__):
            config.append(value)
        for start in range(0, len(words), BATCH_WRITE_SIZE):
            batch = to_rdkit_bitvects(words[start : start + BATCH_WRITE_SIZE])
            table.append([process_fp(fp, start + i) for i, fp in enumerate(batch)])
        table.cols.popcnt.create_index(kind="full")
    sort_db_file(path)


class Runner:
    """Times cells and appends one JSON line per cell to the output file."""

    def __init__(self, args, out):
        """Store the parsed arguments and the open results file."""
        self.args = args
        self.out = out

    def record(self, **row):
        """Append a result row and print a one-line summary."""
        row.update(fp_size=self.args.fp_size, radius=self.args.radius, metric="tanimoto", time=time.time())
        self.out.write(json.dumps(row) + "\n")
        self.out.flush()
        status = row["status"]
        detail = f"median {row['median_s']:.4g} s" if status == "ok" else row.get("note", "")
        print(
            f"N={row['library_size']:<9d} {row['method']:17s} q={row['num_queries']:<6d} k={row.get('k')!s:<5s} "
            f"t={row['threshold']:<4} {status} {detail}",
            flush=True,
        )

    def time_cell(self, run, repeats, cap):
        """Warm up once, then time repeats. Returns (times, status); stops after a run exceeding cap."""
        with gpu_lock(self.args.gpu_lock):
            start = time.perf_counter()
            run()
            warmup = time.perf_counter() - start
            if warmup > cap:
                return [warmup], "timeout"
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                run()
                times.append(time.perf_counter() - start)
        return times, "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="SMILES file (first column SMILES, optional header)")
    parser.add_argument("--library-sizes", type=int, nargs="+", default=[100_000, 1_000_000, 10_000_000])
    parser.add_argument("--query-counts", type=int, nargs="+", default=[1, 100, 10_000])
    parser.add_argument("--k", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.7])
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp-size", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cell-timeout", type=float, default=120.0, help="Seconds per run before a cell times out")
    parser.add_argument("--fpsim2-workers", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32, help="Processes for fingerprint generation")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--cache-dir", default="fingerprint_search_cache")
    parser.add_argument("--gpu-lock", default=None, help="flock file held while each cell is timed")
    parser.add_argument("--out", default="fingerprint_search_results.jsonl")
    args = parser.parse_args()

    methods = list(args.methods)
    try:
        import FPSim2  # noqa: F401
    except ImportError:
        methods = [m for m in methods if not m.startswith("fpsim2")]
        print("FPSim2 not importable; skipping fpsim2 methods", flush=True)

    max_queries = max(args.query_counts)
    largest = max(args.library_sizes)
    words, _ = load_packed_fingerprints(
        args.input, largest + max_queries, args.radius, args.fp_size, args.workers, args.seed, args.cache_dir
    )
    query_words = words[largest:]
    query_gpu_all = torch.from_numpy(query_words.view(np.int32)).cuda()
    query_fps_all = to_rdkit_bitvects(query_words) if any(m.startswith("fpsim2") for m in methods) else None

    with open(args.out, "a") as out:
        runner = Runner(args, out)
        for library_size in sorted(args.library_sizes):
            library_words = words[:library_size]
            for method in methods:
                base = {"library_size": library_size, "method": method}
                engine = None
                start = time.perf_counter()
                if method.startswith("nvmolkit"):
                    library_gpu = torch.from_numpy(library_words.view(np.int32)).cuda()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    engine = FingerprintSearchIndex(library_gpu)
                    del library_gpu
                else:
                    db_path = os.path.join(
                        args.cache_dir, f"fpsim2_n{library_size}_r{args.radius}_b{args.fp_size}_s{args.seed}.h5"
                    )
                    if not os.path.exists(db_path):
                        write_fpsim2_db(library_words, args.radius, db_path)
                    start = time.perf_counter()
                    if method == "fpsim2_cpu":
                        from FPSim2 import FPSim2Engine

                        engine = FPSim2Engine(db_path)
                    else:
                        from FPSim2 import FPSim2CudaEngine

                        engine = FPSim2CudaEngine(db_path)
                build_s = time.perf_counter() - start
                timeout_queries = None
                for num_queries in sorted(args.query_counts):
                    queries_gpu = query_gpu_all[:num_queries].contiguous()
                    query_fps = query_fps_all[:num_queries] if query_fps_all is not None else None
                    for threshold in args.thresholds:
                        k_values = [None] if method == "fpsim2_cuda" else args.k
                        if method == "fpsim2_cuda" and threshold <= 0.0:
                            continue
                        for k in k_values:
                            cell = dict(base, num_queries=num_queries, k=k, threshold=threshold, build_s=build_s)
                            if timeout_queries is not None and num_queries > timeout_queries:
                                runner.record(**cell, status="skipped", note=f"{timeout_queries} queries timed out")
                                continue
                            if method.startswith("nvmolkit"):
                                prune = method == "nvmolkit"

                                def run(q=queries_gpu, kk=k, t=threshold, p=prune, index=engine):
                                    index.search(q, k=kk, threshold=t, prune=p)
                                    torch.cuda.synchronize()

                            else:

                                def run(fps=query_fps, kk=k, t=threshold, eng=engine, m=method):
                                    for fp in fps:
                                        if m == "fpsim2_cpu":
                                            eng.top_k(fp, k=kk, threshold=t, n_workers=args.fpsim2_workers)
                                        else:
                                            eng.similarity(fp, threshold=t)

                                # Predict the cell from one query before committing to it.
                                start = time.perf_counter()
                                run(fps=query_fps[:1])
                                predicted = (time.perf_counter() - start) * num_queries
                                if predicted > args.cell_timeout:
                                    runner.record(**cell, status="skipped", note=f"predicted {predicted:.0f} s")
                                    continue
                            times, status = runner.time_cell(run, args.repeats, args.cell_timeout)
                            if status == "timeout":
                                timeout_queries = num_queries
                                runner.record(
                                    **cell, status="timeout", note=f"first run {times[0]:.0f} s", times=times
                                )
                                continue
                            runner.record(**cell, status="ok", times=times, median_s=statistics.median(times))
                del engine
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
