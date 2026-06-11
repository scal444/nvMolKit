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
