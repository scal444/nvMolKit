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

"""Check FingerprintSearchIndex against RDKit on a large library with sampled queries.

The unit tests cover small libraries exhaustively. This script searches a library of real molecules too large for an
exhaustive RDKit comparison, and checks a random sample of queries against RDKit's Bulk*Similarity over the whole
library. Queries are drawn both from the library (each finds itself) and from held-out molecules.

Example:
    python fingerprint_search_validate.py --input enamine.cxsmiles --library-size 1000000
"""

import argparse
import sys
import time

import numpy as np
import torch
from fingerprint_search_data import load_packed_fingerprints, to_rdkit_bitvects
from rdkit.DataStructs import BulkCosineSimilarity, BulkTanimotoSimilarity

from nvmolkit.similarity import FingerprintSearchIndex

RDKIT_BULK = {"tanimoto": BulkTanimotoSimilarity, "cosine": BulkCosineSimilarity}


def rdkit_top_k(similarities, k, threshold):
    """Top k of one query's RDKit similarities, ordered by descending score then ascending index."""
    order = np.lexsort((np.arange(len(similarities)), -similarities))
    order = order[similarities[order] >= threshold][:k]
    scores = np.full(k, np.nan)
    indices = np.full(k, -1, dtype=np.int64)
    scores[: len(order)] = similarities[order]
    indices[: len(order)] = order
    return scores, indices


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="SMILES file (first column SMILES, optional header)")
    parser.add_argument("--library-size", type=int, default=1_000_000)
    parser.add_argument("--queries", type=int, default=40, help="Sampled queries of each kind (library, held-out)")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp-size", type=int, default=2048)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--metrics", nargs="+", default=["tanimoto", "cosine"])
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cache-dir", default="fingerprint_search_cache")
    args = parser.parse_args()

    start = time.perf_counter()
    words, _ = load_packed_fingerprints(
        args.input,
        args.library_size + args.queries,
        args.radius,
        args.fp_size,
        args.workers,
        args.seed,
        args.cache_dir,
    )
    library_words, held_out_words = words[: args.library_size], words[args.library_size :]
    rng = np.random.default_rng(args.seed)
    library_queries = np.sort(rng.choice(args.library_size, size=args.queries, replace=False))
    query_words = np.concatenate([library_words[library_queries], held_out_words])
    print(f"fingerprints ready: {words.shape} in {time.perf_counter() - start:.0f} s", flush=True)

    library_fps = to_rdkit_bitvects(library_words)
    query_fps = to_rdkit_bitvects(query_words)
    library_gpu = torch.from_numpy(library_words.view(np.int32)).cuda()
    queries_gpu = torch.from_numpy(query_words.view(np.int32)).cuda()

    failures = 0
    checked = 0
    for metric in args.metrics:
        reference = [np.asarray(RDKIT_BULK[metric](fp, library_fps)) for fp in query_fps]
        index = FingerprintSearchIndex(library_gpu, metric=metric)
        for k in args.k:
            for threshold in args.thresholds:
                for prune in (True, False):
                    scores, indices = index.search(queries_gpu, k=k, threshold=threshold, prune=prune)
                    got_scores, got_indices = scores.numpy(), indices.numpy()
                    bad = []
                    for row, sims in enumerate(reference):
                        want_scores, want_indices = rdkit_top_k(sims, k, threshold)
                        same_scores = np.array_equal(got_scores[row], want_scores, equal_nan=True)
                        if not (same_scores and np.array_equal(got_indices[row], want_indices)):
                            bad.append(row)
                    checked += len(reference)
                    failures += len(bad)
                    status = "ok" if not bad else f"MISMATCH rows {bad[:10]}"
                    print(f"{metric:8s} k={k:<5d} threshold={threshold:<4} prune={prune!s:5s} {status}", flush=True)

    print(f"{checked - failures}/{checked} query results match RDKit")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
