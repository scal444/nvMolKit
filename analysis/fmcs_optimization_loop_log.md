# fMCS Optimization Loop Log

Benchmark protocol for this loop:

- Dataset: `/home/kevin/data/enamine_real_10M.csxmiles`
  - Note: `/home/kevin/data/enamine_real_10M.cxsmiles` was requested but is
    not present on this machine.
- Benchmark script: `/home/kevin/repos/nvmolkit/benchmarks/mcs_bench.py`
- Fixed run controls:
  - `--max-mols 1000`
  - `--pairs 1000`
  - `--seed 42`
  - `--warmups 1`
  - `--runs 1`
  - `--batch-size 1000`
  - `--block-size 128`
  - `--no-rdkit`
- Resource gates:
  - Track ptxas registers and shared memory.
  - Hard fail on nonzero stack frame or spills unless explicitly diagnostic.

## Baseline: fixed synchronization state

- Branch: `codex/fmcs-timing-instrumentation`
- Commit: `982b2aa Fix fMCS phase-2 synchronization races`
- Status: accepted baseline
- Benchmark CSV: `/tmp/fmcs_baseline_982b2aa_block128_seed42_1k.csv`
- Python build log: `/tmp/nvmolkit-pybuild-fmcs-baseline.log`

Benchmark result:

- Wall time: `8755.909 ms`
- Throughput: `114.21 pairs/s`
- Per-pair mean: `57.458 ms`
- Per-pair median: `23.466 ms`
- Per-pair p90: `106.314 ms`
- Per-pair p95: `174.051 ms`
- Per-pair p99: `418.859 ms`
- Slowest pair: pair index `851`, `8269.949 ms`
- GPU/fallback/overflow/canceled: `1000/0/0/0`

Resource tracking:

- SM120 block 128 tier128 non-stats:
  - `80` registers/thread
  - `13032 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM120 block 128 tier128 stats:
  - `80` registers/thread
  - `13128 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM89 block 128 tier128 non-stats:
  - `91` registers/thread
  - `13032 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM89 block 128 tier128 stats:
  - `89` registers/thread
  - `13128 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads

## Experiment 001: adjacency scan in fast incremental match

- Branch: `codex/fmcs-opt-001-adj-fast-match`
- Base commit: `63e0a68 Record fMCS optimization loop baseline`
- Status: accepted
- Benchmark CSV: `/tmp/fmcs_opt001_adj_fast_match_block128_seed42_1k.csv`
- Python build log: `/tmp/nvmolkit-pybuild-fmcs-opt001.log`
- SM120 build log: `/tmp/nvmolkit_fmcs_opt001_sm120_build.log`
- SM89 build log: `/tmp/nvmolkit_fmcs_opt001_sm89_build.log`

Optimization tried:

- Added a host-built CSR adjacency `bondIndices` array parallel to
  `colIndices`.
- Threaded `bondIndices` through `DevicePerPairInput` and `DeviceCsrView`.
- Changed the atom-adding branch of `matchIncrementalFastCooperative` to scan
  only bonds adjacent to the already mapped target atom when adjacency bond ids
  are available.
- Left the previous full-target-bond scan as the fallback path for test views
  or future topologies without `bondIndices`.

Validation:

- `test_fmcs_unit --gtest_brief=1`: `58` tests passed in `233 ms`.
- `test_fmcs --gtest_brief=1`: `62` tests passed in `1534 ms`.
- `git diff --check`: passed.

Benchmark result:

- Wall time: `8113.362 ms`
- Throughput: `123.25 pairs/s`
- Delta vs baseline wall time: `-642.547 ms` (`-7.34%`)
- Delta vs baseline throughput: `+7.91%`
- Per-pair mean: `57.201 ms`
- Per-pair median: `23.515 ms`
- Per-pair p90: `105.185 ms`
- Per-pair p95: `167.341 ms`
- Per-pair p99: `416.264 ms`
- Slowest pair: pair index `851`, `8164.044 ms`
- GPU/fallback/overflow/canceled: `1000/0/0/0`

Resource tracking:

- SM120 block 128 tier128 non-stats:
  - `80` registers/thread
  - `13048 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM120 block 128 tier128 stats:
  - `80` registers/thread
  - `13144 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM89 block 128 tier128 non-stats:
  - `91` registers/thread
  - `13048 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM89 block 128 tier128 stats:
  - `89` registers/thread
  - `13144 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads

Decision:

- Accepted. Carry future experiment branches forward from this branch.

## Experiment 004: fallback query adjacency scans

- Branch: `codex/fmcs-opt-004-fallback-query-adj`
- Base commit: `aad968b Optimize fMCS target bond lookups with adjacency`
- Status: accepted
- Benchmark CSV: `/tmp/fmcs_opt004_fallback_query_adj_block128_seed42_1k.csv`
- Python build log: `/tmp/nvmolkit-pybuild-fmcs-opt004.log`
- SM120 build log: `/tmp/nvmolkit_fmcs_opt004_sm120_build.log`
- SM89 build log: `/tmp/nvmolkit_fmcs_opt004_sm89_build.log`

Optimization tried:

- Added a scalar `seedContainsBondWithinThread` bitset helper.
- Used target CSR row lengths to compute fallback target degrees instead of
  decoding all target bond endpoints.
- Used query CSR adjacency bond ids for fallback mapped-neighbor counting,
  mapped-neighbor selection, and partial edge-consistency checks.
- Kept all previous seed-bond scans as fallbacks for topology views without
  adjacency bond ids.

Validation:

- `test_fmcs_unit --gtest_brief=1`: `58` tests passed in `214 ms`.
- `test_fmcs --gtest_brief=1`: `62` tests passed in `1446 ms`.
- `git diff --check`: passed.

Benchmark result:

- Wall time: `3461.522 ms`
- Throughput: `288.89 pairs/s`
- Delta vs accepted experiment 003 wall time: `-4061.823 ms` (`-53.99%`)
- Delta vs accepted experiment 003 throughput: `+117.34%`
- Delta vs original baseline wall time: `-5294.387 ms` (`-60.47%`)
- Delta vs original baseline throughput: `+152.94%`
- Per-pair mean: `32.120 ms`
- Per-pair median: `16.979 ms`
- Per-pair p90: `59.595 ms`
- Per-pair p95: `93.767 ms`
- Per-pair p99: `205.574 ms`
- Slowest pair: pair index `851`, `3459.965 ms`
- GPU/fallback/overflow/canceled: `1000/0/0/0`

Resource tracking:

- SM120 block 128 tier128 non-stats:
  - `80` registers/thread
  - `13048 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM120 block 128 tier128 stats:
  - `80` registers/thread
  - `13144 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM89 block 128 tier128 non-stats:
  - `98` registers/thread
  - `13048 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM89 block 128 tier128 stats:
  - `98` registers/thread
  - `13144 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads

Decision:

- Accepted despite the SM89 register regression because the fixed benchmark
  wall time improved by `53.99%` from the previous accepted branch with no
  stack or spills. Next resource-focused work should try to recover SM89
  register pressure without giving back the adjacency fallback win.

## Experiment 005: compile-time adjacency specialization

- Branch: `codex/fmcs-opt-005-adj-constexpr-specialize`
- Base commit: `641c71f Optimize fMCS fallback adjacency scans`
- Status: accepted
- Benchmark CSV: `/tmp/fmcs_opt005_adj_constexpr_specialize_block128_seed42_1k.csv`
- Python build log: `/tmp/nvmolkit-pybuild-fmcs-opt005.log`
- SM120 build log: `/tmp/nvmolkit_fmcs_opt005_sm120_build.log`
- SM89 build log: `/tmp/nvmolkit_fmcs_opt005_sm89_build.log`

Optimization tried:

- Marked production `DeviceCsrView` as compile-time adjacency-bond capable.
- Marked the unit-test duck `TestCsrView` as fallback-only.
- Added a `__host__ __device__ constexpr` topology trait and used
  `if constexpr` in fallback-heavy helper paths so production kernels compile
  adjacency-only code while fallback views keep runtime full-scan fallback
  behavior.

Validation:

- `test_fmcs_unit --gtest_brief=1`: `58` tests passed in `209 ms`.
- `test_fmcs --gtest_brief=1`: `62` tests passed in `1430 ms`.
- `git diff --check`: passed.

Benchmark result:

- Wall time: `3339.929 ms`
- Throughput: `299.41 pairs/s`
- Delta vs accepted experiment 004 wall time: `-121.593 ms` (`-3.51%`)
- Delta vs accepted experiment 004 throughput: `+3.64%`
- Delta vs original baseline wall time: `-5415.980 ms` (`-61.85%`)
- Delta vs original baseline throughput: `+162.16%`
- Per-pair mean: `30.398 ms`
- Per-pair median: `15.417 ms`
- Per-pair p90: `57.882 ms`
- Per-pair p95: `89.924 ms`
- Per-pair p99: `201.458 ms`
- Slowest pair: pair index `851`, `3338.519 ms`
- GPU/fallback/overflow/canceled: `1000/0/0/0`

Resource tracking:

- SM120 block 128 tier128 non-stats:
  - `80` registers/thread
  - `13048 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM120 block 128 tier128 stats:
  - `80` registers/thread
  - `13144 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM89 block 128 tier128 non-stats:
  - `91` registers/thread
  - `13048 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM89 block 128 tier128 stats:
  - `92` registers/thread
  - `13144 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads

Decision:

- Accepted. It improves wall time and recovers most of experiment 004's SM89
  register regression while preserving zero stack/spills.

## Experiment 002: adjacency walk for remaining-size bound

- Branch: `codex/fmcs-opt-002-remaining-adj-bound`
- Base commit: `de5c3fd Optimize fMCS fast match adjacency scans`
- Status: rejected
- Benchmark CSV: `/tmp/fmcs_opt002_remaining_adj_bound_block128_seed42_1k.csv`
- Python build log: `/tmp/nvmolkit-pybuild-fmcs-opt002.log`
- SM120 build log: `/tmp/nvmolkit_fmcs_opt002_sm120_build.log`
- SM89 build log: `/tmp/nvmolkit_fmcs_opt002_sm89_build.log`

Optimization tried:

- Kept the exact connectivity-aware remaining-size traversal semantics.
- Added force-inlined helpers that visit reachable query bonds through CSR
  `rowOffsets`, `colIndices`, and `bondIndices` when available.
- Preserved the old all-query-bond scan as a fallback for topology views without
  adjacency bond ids.

Validation:

- `test_fmcs_unit --gtest_brief=1`: `58` tests passed in `230 ms`.
- `test_fmcs --gtest_brief=1`: `62` tests passed in `513 ms`.
- `git diff --check`: passed.

Benchmark result:

- Wall time: `8832.086 ms`
- Throughput: `113.22 pairs/s`
- Delta vs accepted experiment 001 wall time: `+718.724 ms` (`+8.86%`)
- Delta vs accepted experiment 001 throughput: `-8.14%`
- Per-pair mean: `55.415 ms`
- Per-pair median: `22.056 ms`
- Per-pair p90: `105.296 ms`
- Per-pair p95: `165.670 ms`
- Per-pair p99: `398.791 ms`
- Slowest pair: pair index `851`, `8457.654 ms`
- GPU/fallback/overflow/canceled: `1000/0/0/0`

Resource tracking:

- SM120 block 128 tier128 non-stats:
  - `80` registers/thread
  - `13048 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM120 block 128 tier128 stats:
  - `80` registers/thread
  - `13144 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM89 block 128 tier128 non-stats:
  - `91` registers/thread
  - `13048 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM89 block 128 tier128 stats:
  - `89` registers/thread
  - `13144 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads

Decision:

- Rejected. Do not carry forward; return to
  `codex/fmcs-opt-001-adj-fast-match`.

## Experiment 003: adjacency target-bond lookup

- Branch: `codex/fmcs-opt-003-adj-bond-lookup`
- Base commit: `01ce7ea Record rejected fMCS remaining-bound experiment`
- Status: accepted
- Benchmark CSV: `/tmp/fmcs_opt003_adj_bond_lookup_block128_seed42_1k.csv`
- Python build log: `/tmp/nvmolkit-pybuild-fmcs-opt003.log`
- SM120 build log: `/tmp/nvmolkit_fmcs_opt003_sm120_build.log`
- SM89 build log: `/tmp/nvmolkit_fmcs_opt003_sm89_build.log`

Optimization tried:

- Used target CSR adjacency bond ids for the fast incremental match
  ring-closing case.
- Used target CSR adjacency bond ids in
  `findTargetBondBetweenAtomsWithinThread`, which feeds fallback rebuild and
  substructure edge-consistency checks.
- Preserved all-bond scans as fallbacks for topology views without adjacency
  bond ids.

Validation:

- `test_fmcs_unit --gtest_brief=1`: `58` tests passed in `218 ms`.
- `test_fmcs --gtest_brief=1`: `62` tests passed in `1502 ms`.
- `git diff --check`: passed.

Benchmark result:

- Wall time: `7523.345 ms`
- Throughput: `132.92 pairs/s`
- Delta vs accepted experiment 001 wall time: `-590.017 ms` (`-7.27%`)
- Delta vs accepted experiment 001 throughput: `+7.85%`
- Delta vs original baseline wall time: `-1232.564 ms` (`-14.08%`)
- Delta vs original baseline throughput: `+16.38%`
- Per-pair mean: `53.157 ms`
- Per-pair median: `22.112 ms`
- Per-pair p90: `98.753 ms`
- Per-pair p95: `161.680 ms`
- Per-pair p99: `386.781 ms`
- Slowest pair: pair index `851`, `7649.792 ms`
- GPU/fallback/overflow/canceled: `1000/0/0/0`

Resource tracking:

- SM120 block 128 tier128 non-stats:
  - `80` registers/thread
  - `13048 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM120 block 128 tier128 stats:
  - `80` registers/thread
  - `13144 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM89 block 128 tier128 non-stats:
  - `89` registers/thread
  - `13048 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads
- SM89 block 128 tier128 stats:
  - `89` registers/thread
  - `13144 B` shared memory
  - `0 B` stack
  - `0 B` spill stores
  - `0 B` spill loads

Decision:

- Accepted. Carry future experiment branches forward from this branch.
