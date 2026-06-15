# FMCS BSSY/BSYNC Source-Control Review

Report analyzed:
`/home/kboyd/omg/workflows/etkdg/mcs_2026_06_15/ncu_bs_512.ncu-rep`

Generated artifacts:

- `/tmp/ncu-branch-sync.json`
- `/tmp/ncu-branch-points.csv`
- `/tmp/ncu-bssy.csv`

## Method

The analysis starts from source control points in the checked-out nvMolKit
repo, then attaches NCU-correlated `BSSY` instructions to those source spans.
Direct counts mean the `BSSY` mapped to the control statement condition/header.
Span counts mean the `BSSY` mapped somewhere inside that control span; these
are recorded, but are not treated as direct evidence that the enclosing
condition itself needs a uniformity fix.

The FMCS cooperative group in this report is a full warp:

- `kFmcsGroupSize = 32`
- `group = cg::tiled_partition<kFmcsGroupSize>(block)`

So a predicate is full-warp uniform when it is derived only from:

- a `group.shfl`/cooperative helper return,
- per-group shared state indexed by a warp-uniform `groupId`,
- scalar seed/queue/best-score state that all lanes in the warp read identically.

Predicates involving `groupRank`, `laneRank`, `block.thread_rank()`,
sub-warp rank/lane, per-lane target indices, per-lane partials, or atomic
winner lanes are not `mark_warp_uniform` candidates from this report.

## Applied decisions after reassessment

The first broad patch marked too many values. The follow-up report
`ncu_2_bsyncfix__bs_512.ncu-rep` showed nearly identical runtime, static
`BSSY` increasing from 1220 to 1238, and dynamic `BSSY`/`BSYNC` and `SHFL`
execution increasing. That established the placement rule used by the current
reduced patch:

- A `mark_warp_uniform` call is only a candidate when it dominates the branch
  or loop whose predicate needs uniformity.
- Do not add it inside a loop/body that the compiler already treats as
  potentially divergent.
- Do not replace existing `group.shfl(..., 0)` broadcasts with it; that only
  moves source correlation into the shuffle intrinsic.

The current code keeps only the outside-in candidates:

- `fmcsKernel` marks `groupId` before the top-level `if (groupId == 0)`.
- Phase-2 growth-bound checks mark the call-site result of
  `seedCanGrowBiggerThanWithinThread`, leaving the helper itself unchanged.
- Phase-2 grow-state and new-bond count are marked once before the stage
  branches and loops.
- Phase-2 match results from `checkSeedMatchAndAppendCooperative` are marked at
  the call sites before `if (ok)`.

The following broad-patch changes were reverted:

- Redundant `group.shfl` to `mark_warp_uniform` substitutions.
- Phase-1 markings, because phase 1 is not a runtime-significant path here.
- Fast-match, prepare-ordering, fallback-depth, sub-warp, `bondAlive`, and
  final `found` markings that sat inside potentially divergent control.

## Rejected and non-actionable mappings

The other mapped `BSSY` sites from the report were rejected or left as inventory
for these reasons; the exhaustive table below keeps every mapped control point.

- Owner-lane control: examples include `fmcs_kernel.cuh:667`,
  `fmcs_kernel.cuh:702`, `fmcs_match.cuh:873`, and `fmcs_match.cuh:885`.
  These intentionally guard single-lane mutation or construction work.
- Lane- or sub-warp-distributed work: examples include `fmcs_grow.cuh:92`,
  `fmcs_match.cuh:266`, `fmcs_match.cuh:360`, `fmcs_match.cuh:1481`, and
  `fmcs_match.cuh:1504`. These predicates are expected to differ by lane or
  sub-warp lane.
- Atomic winner/contention control: examples include `fmcs_kernel.cuh:352` and
  `fmcs_match.cuh:1321`. Marking these uniform would change the intended
  arbitration behavior.
- Span-only mappings: examples include `fmcs_kernel.cuh:1383`,
  `fmcs_kernel.cuh:1395`, `fmcs_match.cuh:641`, and `fmcs_match.cuh:1331`.
  In these cases the source-correlated `BSSY` lands inside a source span rather
  than on the control header, so it is not direct evidence that the enclosing
  branch predicate should be marked.
- Transfer/helper-line mappings: examples include `fmcs_match.cuh:676` and
  `fmcs_seed_queue.cuh:163`. These are kept in the inventory, but do not name a
  branch predicate to mark.

## Verification

Code validation performed after applying the reduced outside-in patch:

- Built `/tmp/nvmolkit-build-nvmolkit` target `test_fmcs_unit`.
- The `fmcsKernel` ptxas output reported 0 stack bytes, 0 spill stores, and
  0 spill loads for all emitted variants.
- Register counts remain high at 94 or 96 registers depending on
  specialization in this SM89 build; this patch is not an occupancy fix by
  itself.
- `ctest --test-dir /tmp/nvmolkit-build-nvmolkit -R "FMCSUnit|FMCSDispatch|FMCSBasics|FMCSRegression|FMCSBlockSize|FMCSIntegration|FMCSMolecule|FMCSTiers|FMCSBatch|FMCSConnected|FMCSLabels|FMCSObjective|FMCSMappingConsistency|FMCSTimeout|FMCSOverflow|FMCSDegenerate" --output-on-failure`
  passed 124/124 tests.

This reduced patch has not yet been re-profiled in NCU, so this note records
source disposition and validation, not post-change `BSSY` removal.

## Exhaustive mapped-control review

Verdicts:

- `INVESTIGATE`: direct predicate appears full-warp uniform in this kernel.
- `INVESTIGATE-SUBWARP`: direct predicate is full-warp uniform, but the selected body intentionally uses sub-warp work.
- `NO-LANE`: predicate can vary by lane/sub-warp/per-lane target or partial state.
- `NO-OWNER`: predicate is inside or directly selects lane-owner/thread-owner code.
- `NO-ATOMIC`: atomic contention/winner control; not a uniform branch fix.
- `SPAN`: `BSSY` mapped inside the source span, not to the control header.
- `NO-TRANSFER`: source correlation landed on a return/helper access, not a branch predicate to mark.

| Source | Direct | Span | Verdict | Reason |
| --- | ---: | ---: | --- | --- |
| `fmcs_grow.cuh:33` `if (bond.endAtomSeedIdx == NewBond::kNotInSeed)` | 10 | 0 | NO-OWNER | Active call sites patch seeds from owner-lane code. |
| `fmcs_grow.cuh:92` `for (int q = laneRank; ...)` | 2 | 0 | NO-LANE | Lane-distributed query-bond scan. |
| `fmcs_grow.cuh:96` excluded-bond `continue` | 2 | 0 | NO-LANE | `q` differs by lane. |
| `fmcs_kernel.cuh:319` word-copy loop | 0 | 1 | SPAN | Span-only inside lane-distributed copy helper. |
| `fmcs_kernel.cuh:342` incumbent CAS loop | 0 | 10 | NO-ATOMIC | Span-only inside lane-0 atomic score update. |
| `fmcs_kernel.cuh:352` copy-lock CAS loop | 10 | 0 | NO-ATOMIC | Lane-0 lock acquisition. |
| `fmcs_kernel.cuh:359` `if (shouldCopy)` | 2 | 0 | INVESTIGATE | `shouldCopy` is broadcast with `group.shfl`. |
| `fmcs_kernel.cuh:447` `if (!ok)` | 8 | 6 | INVESTIGATE | Cooperative match result controls fallback. |
| `fmcs_kernel.cuh:609` `if (slot < 0)` | 2 | 0 | INVESTIGATE | Queue reservation result is broadcast. |
| `fmcs_kernel.cuh:667` `if (groupRank == 0)` | 8 | 0 | NO-OWNER | Single-lane owner block. |
| `fmcs_kernel.cuh:677` `if (ok)` | 4 | 8 | INVESTIGATE | Queue status is broadcast. |
| `fmcs_kernel.cuh:702` `if (groupRank == 0)` | 2 | 0 | NO-OWNER | Single-lane owner block. |
| `fmcs_kernel.cuh:722` `if (popped)` | 1 | 2 | INVESTIGATE | Pop status is broadcast. |
| `fmcs_kernel.cuh:782` DFS `while (remaining != 0)` | 8 | 4 | NO-OWNER | Inside `group.thread_rank() == 0` remaining-size computation. |
| `fmcs_kernel.cuh:793` bond scan loop | 4 | 8 | NO-OWNER | Inside `group.thread_rank() == 0` remaining-size computation. |
| `fmcs_kernel.cuh:797` visited-bond `continue` | 8 | 0 | NO-OWNER | Inside `group.thread_rank() == 0`. |
| `fmcs_kernel.cuh:830` stack bond scan loop | 8 | 8 | NO-OWNER | Inside `group.thread_rank() == 0` remaining-size computation. |
| `fmcs_kernel.cuh:834` visited-bond `continue` | 8 | 0 | NO-OWNER | Inside `group.thread_rank() == 0`. |
| `fmcs_kernel.cuh:1113` `if (groupId == 0)` | 3 | 0 | INVESTIGATE | `groupId` is uniform for a 32-lane FMCS tile. |
| `fmcs_kernel.cuh:1114` phase-1 `qBond` loop | 1 | 4 | INVESTIGATE | Loop counter and shared abort flags are uniform in group 0. |
| `fmcs_kernel.cuh:1117` `if (groupRank == 0)` | 1 | 1 | NO-OWNER | Single-lane owner block. |
| `fmcs_kernel.cuh:1152` `if (matched)` | 2 | 2 | INVESTIGATE | Cooperative match result. |
| `fmcs_kernel.cuh:1155` push overflow branch | 2 | 0 | INVESTIGATE | `pushBackCooperative` returns a broadcast status. |
| `fmcs_kernel.cuh:1160` queued-seed loop | 2 | 0 | NO-OWNER | Inside the `groupRank == 0` mismatch path. |
| `fmcs_kernel.cuh:1188` block-timeout lane | 2 | 0 | NO-LANE | `block.thread_rank() == 0` is divergent in warp 0. |
| `fmcs_kernel.cuh:1212` `while (true)` | 4 | 2 | SPAN | Loop has no predicate to mark; direct syncs belong to inner branches. |
| `fmcs_kernel.cuh:1213` phase-2 abort flags | 8 | 0 | INVESTIGATE | Flags are read through `readFlagCooperative`. |
| `fmcs_kernel.cuh:1258` `do` wrapper | 0 | 2 | SPAN | Structural wrapper; no predicate to mark. |
| `fmcs_kernel.cuh:1281` `if (!canGrowCurrent)` | 2 | 0 | INVESTIGATE | Shared seed plus cooperative best-score snapshot. |
| `fmcs_kernel.cuh:1329` grow-stage branch | 2 | 4 | INVESTIGATE | `myCurrent` is per-group shared state. |
| `fmcs_kernel.cuh:1343` stage-0 new-bond loop | 2 | 2 | INVESTIGATE | `newBondCount[groupId]` is per-group shared. |
| `fmcs_kernel.cuh:1383` `if (stage0Ok[groupId])` | 0 | 2 | SPAN | Span-only in this report; do not attribute direct `BSSY` to the header. |
| `fmcs_kernel.cuh:1395` `if (newBondCount[groupId] > 1)` | 0 | 2 | SPAN | Span-only in this report. |
| `fmcs_kernel.cuh:1418` stage-1 new-bond loop | 4 | 4 | INVESTIGATE | `newBondCount[groupId]` is per-group shared. |
| `fmcs_kernel.cuh:1419` alive check | 2 | 0 | INVESTIGATE | `myNewBonds[i]` and `i` are shared/uniform across the group. |
| `fmcs_kernel.cuh:1463` `if (ok)` | 2 | 2 | INVESTIGATE | Cooperative match result. |
| `fmcs_kernel.cuh:1487` `if (groupRank == 0)` | 2 | 0 | NO-OWNER | Single-lane owner block. |
| `fmcs_kernel.cuh:1488` alive-count loop | 10 | 0 | NO-OWNER | Inside `groupRank == 0`. |
| `fmcs_kernel.cuh:1506` subset-composition loop | 0 | 4 | SPAN | Span-only in this report. |
| `fmcs_kernel.cuh:1535` alive check | 6 | 0 | NO-OWNER | Inside `groupRank == 0` subset construction. |
| `fmcs_kernel.cuh:1536` composition bit check | 4 | 0 | NO-OWNER | Inside `groupRank == 0` subset construction. |
| `fmcs_kernel.cuh:1572` `if (ok)` | 2 | 2 | INVESTIGATE | Cooperative match result. |
| `fmcs_kernel.cuh:1585` overflow flag break | 2 | 0 | INVESTIGATE | Flag is read through `readFlagCooperative`. |
| `fmcs_kernel.cuh:1622` phase-2 abort flags | 4 | 0 | INVESTIGATE | Flags are read through `readFlagCooperative`. |
| `fmcs_match.cuh:196` fast-match bond-bit loop | 8 | 16 | INVESTIGATE | Seed bond word is common to all lanes. |
| `fmcs_match.cuh:266` target atom compare | 8 | 0 | NO-LANE | Per-lane target adjacency scan. |
| `fmcs_match.cuh:360` adjacency scan loop | 8 | 8 | NO-LANE | Starts from `laneRank`; per-lane scan. |
| `fmcs_match.cuh:641` seed-bit loop | 0 | 16 | SPAN | Span-only inside helper code. |
| `fmcs_match.cuh:676` `return false` | 4 | 0 | NO-TRANSFER | Transfer from helper path, not a predicate to mark. |
| `fmcs_match.cuh:774` `if (compatible)` | 8 | 0 | NO-LANE | `targetAtomIdx` differs by lane. |
| `fmcs_match.cuh:814` seed-atom bit loop | 8 | 0 | NO-OWNER | Inside `laneRank == 0` prepare path. |
| `fmcs_match.cuh:862` ordering loop | 15 | 0 | INVESTIGATE | `numSeedAtoms` is broadcast before the loop. |
| `fmcs_match.cuh:873` `if (laneRank == 0)` | 8 | 0 | NO-OWNER | Single-lane owner block. |
| `fmcs_match.cuh:875` ordered-query guard | 3 | 0 | NO-OWNER | Inside `laneRank == 0`. |
| `fmcs_match.cuh:876` order-position guard | 5 | 0 | NO-OWNER | Inside `laneRank == 0`. |
| `fmcs_match.cuh:882` adjacency loop | 1 | 7 | NO-OWNER | Inside `laneRank == 0` ordering work. |
| `fmcs_match.cuh:885` seed-bond guard | 74 | 0 | NO-OWNER | Inside `laneRank == 0` ordering work. |
| `fmcs_match.cuh:892` ordered-neighbor guard | 2 | 0 | NO-OWNER | Inside `laneRank == 0` ordering work. |
| `fmcs_match.cuh:975` candidate-count cache miss | 3 | 8 | INVESTIGATE | Cache miss predicate is broadcast from lane 0. |
| `fmcs_match.cuh:1011` `if (!prepareOk)` | 3 | 0 | INVESTIGATE | `prepareOk` is broadcast before the branch. |
| `fmcs_match.cuh:1013` `laneRank == 0 && bestAtom < 0` | 5 | 0 | NO-OWNER | Single-lane owner fallback. |
| `fmcs_match.cuh:1022` `if (laneRank == 0)` | 8 | 0 | NO-OWNER | Single-lane owner block. |
| `fmcs_match.cuh:1064` mapped-neighbor adjacency loop | 8 | 0 | NO-OWNER | Active hoisted call is made from lane 0; non-hoisted path is not a current fix target. |
| `fmcs_match.cuh:1067` mapped-neighbor seed-bond guard | 12 | 0 | NO-OWNER | Same helper context as line 1064. |
| `fmcs_match.cuh:1179` edge-consistency seed-bond guard | 24 | 0 | NO-LANE | Called from per-lane partial/target extension context. |
| `fmcs_match.cuh:1307` degree check | 8 | 0 | NO-LANE | `targetAtomIdx` differs by lane. |
| `fmcs_match.cuh:1310` atom table check | 16 | 0 | NO-LANE | `targetAtomIdx` differs by lane. |
| `fmcs_match.cuh:1321` `atomicCAS(&scratch.found, 0, 1)` | 16 | 0 | NO-ATOMIC | Atomic winner lane is intentionally divergent. |
| `fmcs_match.cuh:1322` mapping copy loop | 62 | 0 | NO-LANE | Runs only for the atomic winner path. |
| `fmcs_match.cuh:1331` slot-capacity branch | 0 | 16 | SPAN | Span-only in per-lane atomic-slot path. |
| `fmcs_match.cuh:1333` partial-copy loop | 16 | 0 | NO-LANE | Per-lane slot path. |
| `fmcs_match.cuh:1367` `if (seed.numAtoms != 0)` | 10 | 16 | INVESTIGATE | Per-group seed value is uniform. |
| `fmcs_match.cuh:1373` `prepareOk ? 1 : 0` | 8 | 0 | NO-OWNER | Ternary is inside `laneRank == 0` prepared-state write. |
| `fmcs_match.cuh:1391` `if (prepared < 0)` | 8 | 0 | INVESTIGATE | `prepared` is broadcast before the branch. |
| `fmcs_match.cuh:1405` first-target loop | 6 | 0 | NO-LANE | Starts from `laneRank`; per-lane target scan. |
| `fmcs_match.cuh:1408` first-target degree check | 8 | 0 | NO-LANE | `targetAtomIdx` differs by lane. |
| `fmcs_match.cuh:1423` `if (scratch.currentCount > 0)` | 8 | 0 | INVESTIGATE | Shared counter after group sync. |
| `fmcs_match.cuh:1436` fallback depth loop | 2 | 16 | INVESTIGATE | `numSeedAtoms` is broadcast before the loop. |
| `fmcs_match.cuh:1472` `if (hoistedScanAdjacency)` | 8 | 0 | INVESTIGATE-SUBWARP | Predicate is uniform; selected body intentionally uses 4-lane sub-warps. |
| `fmcs_match.cuh:1481` `if (partialIdx >= numPartials)` | 8 | 0 | NO-LANE | `partialIdx` depends on sub-warp rank. |
| `fmcs_match.cuh:1504` target scan loop | 8 | 8 | NO-LANE | Starts from `laneRank`; per-lane target scan. |
| `fmcs_match.cuh:1562` `if (found != 0)` | 8 | 0 | INVESTIGATE | Shared `scratch.found` after sync. |
| `fmcs_match.cuh:1569` `return found != 0` | 8 | 0 | INVESTIGATE | `found` is broadcast before return. |
| `fmcs_seed.cuh:180` seed-add atom guard | 1 | 0 | NO-OWNER | Active call sites use owner-lane seed patching. |
| `fmcs_seed.cuh:261` seed growth bound | 10 | 0 | INVESTIGATE | Active call sites pass per-group seed and cooperative best-score snapshot. |
| `fmcs_seed_queue.cuh:163` queue slot access return | 4 | 0 | NO-TRANSFER | Access helper line, not a branch predicate to mark. |

## Unmapped repo-source BSSY

There are 41 repo-source `BSSY` instructions that did not attach to a source
control point. They mapped to declarations, `group.sync()`, shuffle lines, or
helper access lines such as:

- `fmcs_kernel.cuh:1016` `groupId` definition
- `fmcs_kernel.cuh:358` `shouldCopy = group.shfl(shouldCopy, 0)`
- `fmcs_kernel.cuh:362`, `367`, `646` `group.sync()`
- `fmcs_kernel.cuh:951`, `1030`, `1092` declarations/indexing
- `fmcs_match.cuh:1400` `effectiveCapacity` definition

These should be kept in the raw sync inventory. They do not identify a source
branch predicate by themselves.
