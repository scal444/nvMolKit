# fMCS Stats Repro Bisection Log

Date: 2026-06-12

## Repro

Command shape:

```bash
timeout 60s python <worktree>/benchmarks/mcs_bench.py \
  --smiles /home/kevin/data/enamine_real_10M.csxmiles \
  --max-mols 1000 \
  --pairs 1 \
  --seed 42 \
  --warmups 0 \
  --runs 1 \
  --batch-size 1 \
  --block-size 128 \
  --no-rdkit \
  --collect-stats \
  --timings-csv /tmp/fmcs_collect_stats_pair0_<commit>.csv
```

Selected pair: `(654, 114)`, input lines `(5240309, 5937853)`, both
`27 atoms / 28 bonds`.

## Commit-Level Results

| Commit | Result |
| --- | --- |
| `50665e0` | Hangs at `starting findMCS`; host timeout 60s |
| `7ea4cd7` | Hangs at `starting findMCS`; host timeout 60s |
| `f4e10a8` | Hangs at `starting findMCS`; host timeout 60s |
| `956abea` | Hangs at `starting findMCS`; host timeout 60s |
| `641c71f` | Fails quickly with illegal memory access at result copyback |
| `aad968b` | Hangs at `starting findMCS`; host timeout 60s |
| `de5c3fd` | Hangs at `starting findMCS`; host timeout 60s |
| `982b2aa` | Hangs at `starting findMCS`; host timeout 60s |
| `6322f5b` | Hangs at `starting findMCS`; host timeout 60s |

`6322f5b Add fMCS timing instrumentation` is the commit that introduced
the Python `--collect-stats` path. Its parent does not provide the same
user-facing stats repro. The accepted optimization chain therefore has no
racecheck-clean stats baseline before the failure.

Control check: `982b2aa` without `--collect-stats` completed quickly with
`12 atoms / 11 bonds`.

Control racecheck on the currently installed patched `6322f5b` build without
`--collect-stats` also completed with `12 atoms / 11 bonds`, but it was not
racecheck clean:

```text
RACECHECK SUMMARY: 12 hazards displayed (0 errors, 12 warnings)
```

The run used the main checkout benchmark driver because the prior `/tmp`
candidate worktree path had been pruned. Output:

- Log: `/tmp/fmcs_racecheck_main_driver_syncwarp_active3_nostats.log`
- CSV: `/tmp/fmcs_racecheck_main_driver_syncwarp_active3_nostats.csv`

## Sanitizer And Block-Size Probes

- `6322f5b` native block 128 stats hangs.
- `6322f5b` native block 64 stats completes with `12 atoms / 11 bonds`.
- `6322f5b` native block 256 stats hangs.
- `6322f5b` racecheck block 128 completes but reports warnings in
  `matchIncrementalFastCooperative`, `matchResultClearWithinThread`, and
  `warpCopy` handoffs.
- Rebuilding `6322f5b` as `RelWithDebInfo` provided source-mapped racecheck
  warnings.
- Adding a block barrier after iteration-start pops did not fix native
  block 128.
- Capping the stats-specialized kernel to three active warp groups made
  native block 128 and block 256 stats complete quickly.
- Adding explicit `group.sync()` / `__syncwarp()` handoffs reduced racecheck
  warnings but did not make racecheck clean.

## Resolution Patch

The accepted race fix is in the main checkout, not the pruned `/tmp`
candidate worktree. It changes:

- `src/mcs/fmcs_cuda/fmcs_kernel.cuh`
- `src/mcs/fmcs_cuda/fmcs_match.cuh`

Mechanism:

- Convert shared loop-control flags from racy `bool` reads/writes to `int`
  flags read or written through atomics where groups can observe them.
- Keep stats updates per warp group and reduce them at block exit, avoiding
  cross-group shared stats contention.
- Make `candidate.match.empty` a lane-0 read broadcast through the warp group,
  then synchronize before the match object can be reused.
- Add warp-group barriers after `seedCanGrowBiggerThanWithinThread` reads and
  before lane 0 mutates `MatchResult`, so all lanes finish reading old
  `Seed`/`MatchResult` state before `warpCopy`, clear, or commit writes can
  overwrite it.
- Keep the queue pop/push phase separated by a block barrier, and add a narrow
  bottom-of-loop block barrier so `phase2Done` is not overwritten while slower
  threads are still reading the previous value.

Validation on 2026-06-12 with `CMAKE_BUILD_TYPE=RelWithDebInfo`,
`NVMOLKIT_CUDA_TARGET_MODE=native`:

- Python package rebuild/install: pass.
- ptxas resource gate for fMCS kernels: `0 bytes stack frame`, `0 bytes spill
  stores`, `0 bytes spill loads`; max observed registers 90/thread, max shared
  memory 27816 bytes.
- Isolated non-stats racecheck, block 128: pass, `12 atoms / 11 bonds`,
  `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`.
- Isolated stats native repro, block 128: pass, `12 atoms / 11 bonds`, GPU
  path, no fallback, no overflow.
- Isolated stats racecheck, block 128: pass, `12 atoms / 11 bonds`,
  `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`.
- Targeted Python MCS tests: `22 passed, 411 deselected`.

Output logs:

- Build: `/tmp/nvmolkit-pybuild-fmcs-racefix.log`
- Non-stats racecheck: `/tmp/fmcs_racecheck_racefix_nostats.log`
- Stats native repro: `/tmp/fmcs_racefix_stats.log`
- Stats racecheck: `/tmp/fmcs_racecheck_racefix_stats.log`
- Python MCS tests: `/tmp/nvmolkit-pytest-mcs-racefix.log`

Post-fix benchmark check on 2026-06-12:

- Same isolated pair `(654, 114)`, block 128, `--warmups 1 --runs 5`:
  - non-stats: median `10.572 ms`, mean `10.743 ms`, throughput
    `94.59 pairs/s`, `gpu=1/1 fallback=0/1 overflow=0/1`
  - stats: median `11.350 ms`, mean `11.532 ms`, throughput
    `88.10 pairs/s`, `gpu=1/1 fallback=0/1 overflow=0/1`
- Main 1000-pair no-RDKit benchmark, block 128, batch size 1000:
  - non-stats: wall `3154.536 ms`, throughput `317.00 pairs/s`,
    per-pair mean `27.420 ms`, median `13.783 ms`, p95 `80.533 ms`,
    p99 `191.768 ms`, slowest pair `851` at `3171.771 ms`
  - stats: wall `3109.726 ms`, throughput `321.57 pairs/s`,
    per-pair mean `27.517 ms`, median `14.078 ms`, p95 `80.324 ms`,
    p99 `186.691 ms`, slowest pair `851` at `3131.524 ms`,
    forced exits `0`
  - both 1000-pair runs completed with `gpu=1000/1000`, `fallback=0/1000`,
    `overflow=0/1000`, `canceled=0/1000`

Benchmark logs:

- Isolated no-stats: `/tmp/fmcs_racefix_current_nostats_runs5.log`
- Isolated stats: `/tmp/fmcs_racefix_current_stats_runs5.log`
- 1000-pair no-stats: `/tmp/fmcs_racefix_current_1k_nostats.log`
- 1000-pair stats: `/tmp/fmcs_racefix_current_1k_stats.log`

## Superseded Candidate Patch State

Experimental patch lives in:

```text
/tmp/nvmolkit-fmcs-line-6322f5b
```

Touched files:

- `src/mcs/fmcs_cuda/fmcs_kernel.cuh`
- `src/mcs/fmcs_cuda/fmcs_match.cuh`
- `src/mcs/mcs_common/mcs_cooperative_copy.cuh`

Validation on candidate patch:

- Native isolated stats repro, block 128: pass
- Native isolated stats repro, block 256: pass
- Racecheck isolated stats repro, block 128: still reports 12 warning groups
- Racecheck isolated non-stats repro, block 128: still reports 12 warning
  groups

Do not propagate this candidate as accepted without resolving the racecheck
warnings or explicitly accepting them as sanitizer false positives.
