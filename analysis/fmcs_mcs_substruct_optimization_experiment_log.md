# fMCS MCS Substructure Optimization Log

Scope for this log:

- Target path: fMCS MCS substructure fallback in
  `src/mcs/fmcs_cuda/fmcs_match.cuh`.
- Benchmark: `/home/kevin/repos/nvmolkit/benchmarks/mcs_bench.py`.
- Dataset: `/home/kevin/data/enamine_real_10M.csxmiles`.
- Fixed benchmark controls unless noted:
  - `--max-mols 1000`
  - `--pairs 1000`
  - `--seed 42`
  - `--warmups 1`
  - `--runs 3`
  - `--batch-size 1000`
  - `--block-size 512`
  - `--no-rdkit`
- Standalone `src/substruct` benchmarks are intentionally out of scope unless
  fMCS is changed to call shared kernel code.

## Baseline: queue-lock fMCS

- Branch base: `codex/fmcs-phase2-queue-lock`
- Commit: `b1f5ea5 Decouple fMCS phase-2 queue progress with a lock`
- Benchmark CSV: `/tmp/fmcs_mcs_substruct_baseline_nostats_bs512_seed42.csv`

Benchmark result:

- Median: `492.654 ms`
- Mean: `481.582 ms`
- Stddev: `26.121 ms`
- Throughput: `2029.82 pairs/s`
- GPU/fallback/overflow: `1000/0/0`

SM89 resource baseline:

- fMCS kernel reports: `60`
- Stack/spills: `0` stack, `0` spill stores, `0` spill loads in all reports
- Register distribution:
  - `90`: `16`
  - `92`: `14`
  - `96`: `8`
  - `98`: `6`
  - `100`: `6`
  - `102`: `4`
  - `103`: `4`
  - `106`: `2`

## Experiment 001: hoist fallback mapped-neighbor lookup

- Branch: `codex/fmcs-mcs-substruct-optimization`
- Status: accepted
- Benchmark CSV: `/tmp/fmcs_mcs_substruct_opt001e_nostats_bs512_seed42.csv`
- SM89 resource build log:
  `/tmp/fmcs_mcs_substruct_opt001f_sm89_resource_build.log`
- Python install log: `/tmp/fmcs_mcs_substruct_opt001e_pip_install.log`

Optimization tried:

- Added a template switch on `matchSeedSubstructureCooperative`.
- In the production/no-stats instantiation, lane 0 computes the mapped query
  neighbor order position once per fallback depth instead of once per partial.
- Reused `scratch.orderedQueryAtom[queryAtomIdx]` after query ordering is
  complete to store that per-depth order position or `kUnmappedTargetIdx`.
- Left stats/measure instantiations on the original inner-loop lookup path so
  diagnostic resource usage stays comparable to baseline.

Benchmark result:

- Median: `453.501 ms`
- Mean: `451.151 ms`
- Stddev: `6.454 ms`
- Throughput: `2205.07 pairs/s`
- GPU/fallback/overflow: `1000/0/0`
- Delta vs queue-lock baseline median: `-39.153 ms` (`-7.95%`)
- Delta vs queue-lock baseline throughput: `+8.63%`

SM89 resource result:

- fMCS kernel reports: `60`
- Stack/spills: `0` stack, `0` spill stores, `0` spill loads in all reports
- Shared memory: unchanged from baseline
- Register distribution:
  - `86`: `16`
  - `87`: `2`
  - `88`: `12`
  - `96`: `8`
  - `98`: `6`
  - `100`: `6`
  - `102`: `4`
  - `103`: `4`
  - `106`: `2`
- Product/no-stats variants changed from `90/92` registers to `86/87/88`
  registers. Stats variants remained at baseline.

Validation:

- Python MCS tests:
  `python -m pytest /home/kevin/repos/nvmolkit/nvmolkit/tests/test_mcs.py --tb=short`
  - Result: `14 passed`
- C++ focused fMCS tests:
  `ctest --test-dir /tmp/nvmolkit-build-nvmolkit-fmcs-mcs-substruct-tests -R "FMCS" -j 8 --output-on-failure`
  - Result: `124/124 passed`
- Racecheck:
  `compute-sanitizer --tool racecheck --error-exitcode 1` on a single-pair
  no-stats fMCS Python benchmark
  - Result: `0 errors`, `0 warnings`
- Synccheck:
  `compute-sanitizer --tool synccheck --error-exitcode 1` on the same
  single-pair no-stats fMCS Python benchmark
  - Result: `0 errors`

Decision:

- Accepted. It improves product/no-stats runtime and register pressure without
  stack, spills, or shared-memory growth, while keeping stats-resource behavior
  at baseline.

## Experiment 002: skip rechecking primary adjacency edge

- Branch: `codex/fmcs-mcs-substruct-optimization`
- Status: rejected
- Race-prone benchmark CSV:
  `/tmp/fmcs_mcs_substruct_opt002b_nostats_bs512_seed42.csv`
- Race-safe benchmark CSV:
  `/tmp/fmcs_mcs_substruct_opt002c_nostats_bs512_seed42.csv`
- Final SM89 resource build log:
  `/tmp/fmcs_mcs_substruct_opt002c_sm89_resource_build.log`

Optimization tried:

- When fallback candidate generation scans target adjacency from a mapped
  query neighbor, use the current target adjacency entry's bond id to test the
  primary query bond directly.
- Pass the primary query neighbor/order pair into
  `substructurePartialEdgeConsistentWithinThread` so that edge is not found a
  second time by scanning target adjacency from the candidate atom.
- Initial version reused `scratch.targetAtomForQuery[queryAtomIdx]` for the
  primary query bond id. Racecheck found warnings because successful lanes can
  write `targetAtomForQuery` while other lanes are still reading it as scratch.
- Final race-safe version used a dedicated per-group `primaryQueryBondIdx`
  byte in `FmcsSubstructureScratch`.

Race-prone benchmark result:

- Median: `442.555 ms`
- Mean: `441.805 ms`
- Stddev: `4.293 ms`
- GPU/fallback/overflow: `1000/0/0`
- Rejected before acceptance because racecheck reported `4` warnings.

Race-safe benchmark result:

- Median: `453.086 ms`
- Mean: `458.518 ms`
- Stddev: `15.780 ms`
- Throughput: `2207.08 pairs/s`
- GPU/fallback/overflow: `1000/0/0`
- Delta vs accepted experiment 001 median: `-0.415 ms` (`-0.09%`)

SM89 resource result for race-safe version:

- fMCS kernel reports: `60`
- Stack/spills: `0` stack, `0` spill stores, `0` spill loads in all reports
- Product/no-stats registers increased by `+2` in `28` variants and `+3` in
  `2` variants.
- Shared memory increased by small padding amounts:
  - `+16 B` in `8` no-stats and `8` stats variants
  - `+32 B` in `8` no-stats and `8` stats variants
  - `+64 B` in `6` no-stats and `6` stats variants

Validation:

- Race-prone version:
  - Python MCS tests: `14 passed`
  - Focused C++ fMCS ctests: `124/124 passed`
  - Racecheck: `4` warnings, rejected
- Race-safe version:
  - Rejected after resource and benchmark checks; full validation not run.

Decision:

- Rejected. The race-safe form loses the speedup and increases registers/shared
  memory, so the accepted branch should continue from experiment 001.

## Experiment 003: cache fallback ordering candidate counts

- Branch: `codex/fmcs-mcs-substruct-optimization`
- Status: accepted
- Benchmark CSV: `/tmp/fmcs_mcs_substruct_opt003a_nostats_bs512_seed42.csv`
- SM89 resource build log:
  `/tmp/fmcs_mcs_substruct_opt003a_sm89_resource_build.log`
- Python install log: `/tmp/fmcs_mcs_substruct_opt003a_pip_install.log`

Optimization tried:

- During fallback query-atom ordering, cache
  `countCandidateTargetAtomsCooperative` per query atom.
- Reused `scratch.targetAtomForQuery[queryAtomIdx]` as a lane-0-only
  prepare-time cache. This is safe because actual target mappings are written
  only after a full substructure mapping is found.
- The candidate count is invariant across ordering rounds, so this avoids
  repeated full target-atom scans for atoms that remain unordered for multiple
  rounds.

Benchmark result:

- Median: `434.914 ms`
- Mean: `434.826 ms`
- Stddev: `9.773 ms`
- Throughput: `2299.31 pairs/s`
- GPU/fallback/overflow: `1000/0/0`
- Delta vs accepted experiment 001 median: `-18.587 ms` (`-4.10%`)
- Delta vs queue-lock baseline median: `-57.740 ms` (`-11.72%`)

SM89 resource result:

- fMCS kernel reports: `60`
- Stack/spills: `0` stack, `0` spill stores, `0` spill loads in all reports
- Shared memory: unchanged from accepted experiment 001
- Product/no-stats registers increased by `+2` in `12` variants and `+3` in
  `2` variants.
- Stats registers changed by `-2` in `4` variants, `+1` in `4` variants, and
  `+2` in `8` variants.

Validation:

- Python MCS tests:
  `python -m pytest /home/kevin/repos/nvmolkit/nvmolkit/tests/test_mcs.py --tb=short`
  - Result: `14 passed`
- C++ focused fMCS tests:
  `ctest --test-dir /tmp/nvmolkit-build-nvmolkit-fmcs-mcs-substruct-tests -R "FMCS" -j 8 --output-on-failure`
  - Result: `124/124 passed`
- Racecheck:
  `compute-sanitizer --tool racecheck --error-exitcode 1` on a single-pair
  no-stats fMCS Python benchmark
  - Result: `0 errors`, `0 warnings`
- Synccheck:
  `compute-sanitizer --tool synccheck --error-exitcode 1` on the same
  single-pair no-stats fMCS Python benchmark
  - Result: `0 errors`

Decision:

- Accepted. The runtime improvement was large enough to justify the small
  register increase, with no stack, spills, or shared-memory growth.
