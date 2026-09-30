# SubstructLibrary query optimization, 2026-09-30

Results behind the commits from `b8c1990` to `cab0e2e` on this branch, for
critical review. The base for review is `31d28aa`, the branch
`codex/substruct-pattern-fingerprints` plus two finalize fixes.

## Hardware and software
- GPU: NVIDIA GeForce RTX 5080, 16 GB (16303 MiB), compute capability 12.0, driver 615.71.09
- CPU: AMD Ryzen 9 9950X, 16 cores / 32 threads; runs capped at 16 threads (`OMP_NUM_THREADS=16`)
- RAM: 59 GB
- OS: Ubuntu 26.04.1 LTS, kernel 7.0.0-31-generic
- CUDA 13.3 (nvcc V13.3.73), host compiler g++ 14.3.0, `NVMOLKIT_CUDA_TARGET_MODE=native` (sm_120), Release
- RDKit 2025.09.1 (conda-forge), Python 3.13.8

## Data (`prep_data.py`)
- Targets: 1,000,000 molecules sampled at random (seed 42) from a 10.35M-row
  Enamine REAL CXSMILES file, parsed with RDKit and sanitized. Average 26 heavy atoms.
- Queries: 20,000 random molecules from the first half of the same file,
  which is sorted by size, so queries are smaller (average 23 heavy atoms).
  The benchmark samples 1000 of them (`--num_queries 1000`, seed 42) and
  removes stereochemistry. They are whole-molecule SMILES queries, not SMARTS.
- Over the 1000 queries, 113 have at least one match, with 231 matching
  target-query pairs in total.

## Method
`benchmarks/substruct_library_bench.py` as of this branch:
- Each library is built once, then every operation (`has`, `count`, `get`)
  and query mode is timed against it.
- Settings: `--runs 3 --warmups 1` (5 runs for the `ab_*` and `07_*` files).
- `steady_ms` is the time to run all 1000 queries once, so ms per query =
  `steady_ms / num_queries`.
- `serial` issues one query and waits for it before the next.
- `concurrent` submits all 1000 at once; up to `query_concurrency` run at a time.
- Every nvMolKit result is checked query by query against complete RDKit
  matches cached by `--reference_cache`, deriving has/count/get from them.
  All runs listed here passed that check.

Iteration command (about 60 s):
```
python substruct_library_bench.py --pickle targets_1M.pkl --query_smiles queries_pool.smi \
  --num_queries 1000 --operations has count get --query_modes serial concurrent \
  --algorithms gsi dfs --no-rdkit --runs 3 --warmups 1 --reference_cache reference_1M_q1000.pkl
```

RDKit baseline (`00_...csv`):
- RDKit `SubstructLibrary` built from `CachedMolHolder` + `PatternHolder`,
  with fingerprints added by `AddPatterns(numThreads=16)`.
- Queries are issued one at a time with `numThreads=16`,
  `useChirality=False`, `recursionPossible=True`.

## Results, ms per query (1M targets, 1000 queries, dfs / gsi)

| state | CSV | dfs serial | dfs concurrent | gsi serial | gsi concurrent |
|---|---|---|---|---|---|
| RDKit, 16 threads | 00 | 4.3 | — | — | — |
| 31d28aa, CPU fingerprint screen, per 64K chunk | probe¹ | ~50 | ~12 | ~50 | ~12 |
| uncommitted: GPU screen per 64K chunk | 01 | 0.75 | 0.42 | 0.79 | 0.42 |
| b8c1990 one resident set per device | 02 | 0.236 | 0.083 | 0.241 | 0.088 |
| 619a397 | 03 | 0.227 | 0.083 | 0.235 | 0.086 |
| 3e21196 bit-sliced screen | 04 | 0.170 | 0.050 | 0.179 | 0.054 |
| 931f310 | 05 | 0.169 | 0.045 | 0.178 | 0.048 |
| e0cdfd9 **unsafe**, see below | 06 | 0.170 | 0.022 (16) | 0.178 | 0.028 (9) |
| a9d30b7 | 07 | 0.164 | 0.017 (16) | 0.172 | 0.026 (9) |
| 9e70e70 memory fix | 08 | 0.166 | 0.031 (6) | 0.169 | 0.047 (4) |
| cab0e2e (HEAD) | 09 | 0.162 | 0.016 (16) | 0.171 | 0.029 (8) |

Numbers in parentheses are the admitted `query_concurrency`.

¹ The 31d28aa numbers come from a small Python probe (100 queries, submitted
the same way), not from this benchmark, because the old benchmark rebuilt the
1M-target library for every mode.

`01b_...csv` is a chunk-size sweep on the uncommitted per-chunk state:

| chunk size | dfs serial | dfs concurrent |
|---|---|---|
| 64K | 0.75 | 0.42 |
| 256K | 0.34 | 0.14 |
| 1M | 0.30 | 0.097 |

Per-chunk overhead motivated b8c1990.

## RDKit baseline checks (`rdkit_variants.py`, 200 queries)

| RDKit setup | per query |
|---|---|
| Query `PatternFingerprint` only | 81 µs |
| `CachedMolHolder` + `PatternHolder`, 16 threads | 4.48 ms |
| `CachedMolHolder` + `PatternHolder`, 1 thread | 39.3 ms |
| `MolHolder` + `PatternHolder`, 16 threads | 4.29 ms |

## Where the time goes at HEAD (nsys, serial, dfs)

| stage | time |
|---|---|
| RDKit `PatternFingerprintMol(query)` on the CPU | ~89 µs per query |
| GPU screen (`intersectPatternSlicesKernel`) | ~14 µs kernel, ~35-45 µs with launches and syncs |
| Candidate match, on the 38% of queries with survivors | ~150-280 µs, of which ~30 µs is GPU kernels |

Screen survivors per query: 62% of queries have zero, median 0, p90 23,
p99 2526, max 38366. That is 153k candidates for 231 true matches (~0.15%).

## Caveats worth checking
- **Admission memory (e0cdfd9 → 9e70e70 → cab0e2e).**
  - e0cdfd9 admitted 16 queries. Concurrent recursive-SMARTS queries, which
    search every target, then ran out of GPU memory, because the recursive
    painting scratch (about 400 MB per executor) was never in the workspace
    estimate.
  - 9e70e70 counts it.
  - cab0e2e releases it after each recursive query and uses a semaphore to
    admit recursive queries separately. CUDA's stream-ordered pool keeps freed
    scratch reserved, so peak use is bounded, but `cudaMemGetInfo` does not
    drop after a recursive query.
- **Serial ordering artifact.** Serial runs measured right after a concurrent
  sweep in the same process read about 5-15% higher (for example `count`/`get`
  serial in 06 and 09). Compare serial numbers from like-for-like runs, such
  as the `ab_*` files: 0.164 vs 0.167 ms for 16 vs 4 workspaces, serial only.
- **Unequal CPU use.** nvMolKit concurrent uses up to 16 host threads, one per
  admitted query. RDKit uses 16 threads within each query.
- **Untested paths.** Multi-GPU paths are not covered on this 1-GPU machine;
  that test is skipped.
- **Tuned constant.** The one tuned constant is 16 independent slice loads per
  step in the screen kernel (8 → 18.4 µs, 16 → 14.3 µs, 32 → 14.5 µs).
- **Measured and reverted:**
  - a 4096-index prefix copy to save a sync;
  - a shorter first step in the screen kernel;
  - processing candidate batches in waves on the caller thread.

  None made a measurable difference.
