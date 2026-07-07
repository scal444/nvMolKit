# fMCS performance options

**Status:** catalogue of performance opportunities found in review of branch
`fmcs-20260630-cleanup` (merge-base `331be7d`). None implemented. Line numbers
as of that branch — re-grep before editing.

These are ordered by expected impact. Each is independent; none changes results.
"Confidence" is how sure the review is that the item is real (vs. that the fix
will pay off — measure before/after in all cases). Prior NCU tooling lives under
`analysis/fmcs_ncu/`; the build already emits `--ptxas-options=-v`
(`src/mcs/CMakeLists.txt:25`) for register/smem readouts.

---

## 1. Redundant global atomics on the queue push/pop hot path

- **Where:** `src/mcs/fmcs_cuda/fmcs_queue_cooperative.cuh:14-33` (`warpAtomicStoreWords`
  / `warpAtomicLoadWords`), used at `:91` (push) and `:135` (pop); incumbent copy
  at `fmcs_kernel.cuh:75`.
- **What:** Push/pop move the entire `QueuedSeed` payload via per-32-bit-word
  `atomicExch` / `atomicAdd(...,0)` **while already holding** `queueLock`
  (and the incumbent copy under `bestCopyLock`). The lock provides mutual
  exclusion and the acquire/release already carry `__threadfence_block`, so the
  payload is never touched concurrently with these copies — the atomics are
  redundant.
- **Fix:** Replace with plain `warpCopy` (int4), exactly what the lockless
  Phase-1 `pushBackCooperative` (`:42`) already uses.
- **Impact:** At tier 128 a `QueuedSeed` is 384 B → ~96 word atomics per push and
  per pop, on the innermost search loop. Removing them cuts L2 atomic traffic on
  the hottest path. Likely the single biggest win here.
- **Risk:** Low, but verify the lock discipline holds at every call site (no
  lockless caller reuses these helpers). **Confidence: high (confirmed
  redundant).**

## 2. Idle-group lock contention at large block sizes

- **Where:** `fmcs_kernel.cuh:770-814` and `:1196-1210`, with
  `fmcs_queue_cooperative.cuh:48-67` (`popBackLockedOrFinishCooperative`).
- **What:** When the queue is transiently empty but `activeGroups > 0`, idle
  groups busy-wait by re-entering the locked pop: each spin does a `queueLock`
  CAS-acquire + 4 `__threadfence_block` + several `group.sync` + release, plus 3
  flag-read atomics per outer iteration.
- **Fix:** Add a lightweight **unlocked** pre-check (queue-empty / `activeGroups`)
  or a short backoff before acquiring `queueLock`; only take the lock when there
  is plausibly work.
- **Impact:** At `blockThreads == 512` (16 groups) up to 15 idle groups hammer
  the single `queueLock` while one group works, throttling that group's child
  pushes. Grows with block size — matters more once the 512 path is enabled
  (see `analysis/fmcs_scratch_placement_plan.md`).
- **Risk:** Medium — must preserve the termination protocol
  (`activeGroups`/`phase2Done` accounting) exactly; the unlocked pre-check must
  not race the decrement that ends Phase 2. **Confidence: high on the contention,
  medium on the magnitude.**

## 3. No `__launch_bounds__` on the main kernel

- **Where:** `fmcs_kernel.cuh:480` (`fmcsKernel`).
- **What:** The ~1400-line kernel has no `__launch_bounds__`, so the compiler
  assumes up to 1024 threads/block for register allocation across the 128- and
  512-thread launches, which can force spills.
- **Fix:** Add `__launch_bounds__(blockThreads)` (the block size is already a
  template param).
- **Impact:** Occupancy is shared-memory-bound (~1 block/SM), so this won't raise
  occupancy, but reducing register spills still cuts local-memory traffic. Read
  the ptxas `registers`/`spill` lines before and after.
- **Risk:** Low. **Confidence: medium (worth measuring).**

## 4. O(N²) per-pair label matrix rebuilt for every pair

- **Where:** `src/mcs/mcs_rdkit_adapter.cpp:147` (dense `numAtoms*numAtoms`
  edge-label matrix) and `:260-269` (`buildLabeledGraphPair`), driven from
  `mcs_search.cpp:135`.
- **What:** Label maps are pair-local, so `buildLabeledGraph` recomputes the CSR
  and the dense N×N edge-label matrix for **both** molecules of **every** pair.
  In `findMCSAllPairs` each molecule is reprocessed O(n) times.
- **Fix:** Cache the molecule-intrinsic parts (per-atom `AtomLabelKey`, per-bond
  `BondLabelKey`, CSR) once per molecule; only re-intern labels per pair. This is
  host-side CPU preprocessing, not kernel time, but it dominates for large
  all-pairs runs.
- **Impact:** Large for `findMCSAllPairs` / big batches; negligible for a single
  pair.
- **Risk:** Low-medium — interning must stay pair-consistent (labels for A and B
  of a pair must come from the same table). **Confidence: high.**

## 5. Per-seed recomputation of target degrees in the fallback

- **Where:** `fmcs_match.cuh:724-786` (`initializeSeedSubstructureScratchCooperative`).
- **What:** `targetDegree[]` is a function of the **target graph only** (invariant
  across every seed of a pair) but is recomputed on every fallback invocation.
- **Fix:** Compute target degrees once per pair (at kernel init, into shared or
  the global scratch slab) and reuse.
- **Impact:** Bounded — only on the substructure fallback path (already the slow
  path), but it is pure repeated work when fallback fires often. Cross-check
  against the `substructureMatchCycles1024` stat.
- **Risk:** Low. **Confidence: high (confirmed recompute).**

## 6. Serial per-attempt match-result clear

- **Where:** `fmcs_seed.cuh:277-291` (`matchResultClearWithinThread`), called at
  `fmcs_kernel.cuh:274` and throughout `fmcs_match.cuh` (654, 690, 698, 704, 717,
  1208, 1491).
- **What:** A serial ~`2*maxAtoms + 2*maxBonds` byte fill on lane 0, on
  essentially every match attempt (~512 lane-0 stores per attempt at tier 128).
  The code comment already flags this as deferred.
- **Fix:** Make the clear cooperative across the group, or clear only the touched
  entries.
- **Impact:** Moderate; scales with attempt count and tier size.
- **Risk:** Low-medium — must keep the "caller owns exclusive write" contract; a
  cooperative clear needs a `group.sync()` before the cleared state is read.
  **Confidence: high.**

## 7. Synchronous clock-rate driver query per chunk

- **Where:** `fmcs.cpp:486-497` (`launchTierAsync`, when `timeoutMs > 0`) and again
  at `:736-745` (`runTierChunks`, when collecting timings).
- **What:** `cudaGetDevice` + `cudaDeviceGetAttribute(cudaDevAttrClockRate)` are
  issued per chunk; these are synchronous driver queries.
- **Fix:** Hoist to once per `runTierChunks` and pass the clock rate down.
- **Impact:** Small (one sync query per chunk), but it is a needless
  host/device sync point on the dispatch path. **Confidence: high.**

## 8. Benign non-atomic incumbent read (correctness-adjacent, list for completeness)

- **Where:** `fmcs_kernel.cuh:55, 69` (`updateIncumbentCooperative`): lane 0 reads
  `*bestScore` with plain loads concurrently with other groups' `atomicCAS`.
- **What:** A formal data race, benign on real hardware (naturally-atomic aligned
  32-bit; value is only a heuristic). Not a perf item per se; `atomicAdd(bestScore,0)`
  would make it clean at negligible cost. **Confidence: high (race confirmed,
  impact benign).**

---

## Note: instrumentation compile flags default ON

`NVMOLKIT_ENABLE_MCS_TIMINGS` and `NVMOLKIT_ENABLE_MCS_STATS` default to `1`
(`src/mcs/mcs_compile_flags.h:7-13`, `cmake/nvmolkit_cmake_options.cmake`). This
is **not** a runtime hot-path cost — the launch dispatches to the lean
`<CollectTimings=false, CollectStats=false>` kernel instantiation when the
runtime params leave collection off (`fmcs_launch.cu:90-124`). It **is** a
compile-time / binary-size cost (all four instantiations built) and it keeps a
few always-zero stat fields wired through to Python. Consider defaulting them OFF
for release builds. See also the dead-stats-fields note in
`analysis/fmcs_experimental_headers_audit.md` / the review.

## Suggested measurement protocol

For each item: capture ptxas `smem`/`registers`/`spill`, and an NCU section
(`SpeedOfLight`, `MemoryWorkloadAnalysis`, `WarpStateStats`) before/after on a
representative ChEMBL batch that actually exercises the fallback (items 5–6) and
a large all-pairs run (item 4). Items 1–2 want a tier-128, block-512 workload
once that path exists.
