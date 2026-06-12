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
