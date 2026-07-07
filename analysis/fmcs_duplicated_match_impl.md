# fMCS duplicated substructure-match implementation

**Status:** cleanup/refactor note from review of branch `fmcs-20260630-cleanup`
(merge-base `331be7d`). Not implemented. Line numbers as of that branch —
re-grep before editing.

> ⚠️ **Read the "Known hazard" section first.** A previous attempt to unify these
> two implementations produced **hangs**. This is not a routine dedup. Treat any
> unification as a concurrency change, not a cosmetic one, and test for hangs
> explicitly (see "Testing for hangs").

## Summary

`matchSeedSubstructureCooperative<bool HoistNeighborOrder>`
(`fmcs_match.cuh:1186-1467`) carries **two full parallel implementations** of the
substructure walk, selected by the `HoistNeighborOrder` template parameter. They
are ~130 lines of near-identical logic that differ only in *how and when* the
per-partial neighbor-order / adjacency-scan decision is computed.

- `HoistNeighborOrder == true`: lane 0 computes `neighborOrderPos` once per depth,
  stores it in `scratch.orderedQueryAtom[queryAtomIdx]`, then **`group.sync()`**
  (`:1199-1209`); all lanes read the hoisted value (`:1214-1216`) and the
  partial loops use a subwarp-partitioned adjacency scan (`:1220-1244`,
  `:1290-1317`).
- `HoistNeighborOrder == false`: every lane recomputes
  `findMappedQueryNeighborWithinThread` and `scanAdjacency` **per partial, inline,
  with no `group.sync()`** in that stretch (`:1260-1287`, and the non-final
  counterpart below it).

## What is compiled where (and the test blind spot)

- **Production** compiles only `HoistNeighborOrder == true` — the kernel selects
  it via `<!CollectStats && !kFmcsMeasure>` (`fmcs_kernel.cuh:172`).
- The **standalone unit test** at `test_fmcs_unit.cu:1276` also uses the default
  `= true`.
- The `HoistNeighborOrder == false` path is compiled **only** in stats/measure
  builds and has **no direct unit coverage**. So a divergence between the two
  implementations would not be caught by the normal test build — the two are
  supposed to be behavior-identical, but nothing asserts it.

That test gap is itself worth closing (see "Testing for hangs") even if you never
unify: add a unit test that runs both instantiations on the same inputs and
asserts identical results.

## Known hazard: why unification has hung before

The two variants differ precisely in **cooperative-group synchronization
structure**, which is the classic source of cooperative-group deadlocks:

- The hoisted path has a **collective `group.sync()`** (`:1208`) that *every* lane
  in the group must reach. The non-hoisted path has **no sync** in the
  corresponding region — each lane is independent.
- Both paths contain loops whose trip counts are **data-dependent and differ per
  lane / per subwarp**: the `partialBase`/`targetScan` loops are bounded by
  `numPartials`, `targetScanEnd`, and the shared `scratch.found == 0` early-exit
  (e.g. `:1225-1244`, `:1246-1258`), and the adjacency path further splits the
  group into size-`kFallbackAdjacencySubwarpSize` (4) subwarps with a `continue`
  when `partialIdx >= numPartials` (`:1227-1228`).

A naive merge that places a `group.sync()`, `group.shfl`, or `group.ballot` on a
path that **not all lanes reach** — or that lets lanes exit a loop on the
data-dependent `scratch.found` / `numPartials` / `targetScanEnd` bound at
different iterations while a sync sits in the loop body — produces a
**partial-group barrier deadlock**: some lanes wait at the barrier forever
because their peers already left. This is almost certainly what caused the prior
hangs. The hoisted variant is safe today because its single `group.sync()` sits
*outside* the divergent loops (all lanes reach it before the per-partial work
begins); the non-hoisted variant is safe because it has no collective op in the
divergent region at all.

### Unification safety checklist

If you attempt to unify, hold these invariants:

1. **No collective op (`group.sync`/`shfl`/`ballot`/`any`/`all`) inside a loop
   whose iteration bound differs across lanes.** The `scratch.found == 0`
   early-exit and per-partial/per-subwarp bounds all vary per lane.
2. **Every collective op must be reached by the full group** on every control
   path. Hoist decisions to lane 0 *before* the divergent loops (as the current
   hoisted path does), never inside them.
3. **Do not let the subwarp `continue` (`:1227`) skip a collective op.** Lanes
   whose `partialIdx >= numPartials` still must participate in any group-wide
   sync.
4. Prefer keeping the sync topology of the **hoisted** variant and making the
   non-hoisted variant's per-lane recompute a special case *inside* the hoisted
   skeleton (compute-once vs compute-per-partial), rather than the reverse.

## Recommendation

Given the hazard, the lower-risk options first:

- **Option A (recommended): factor only the shared *leaf* work**, not the loop/
  sync skeleton. The `tryCommitFinalSubstructurePartialWithinThread` /
  `tryAppendSubstructurePartialWithinThread` calls are already shared; extract the
  common inner scan body (the `for targetScanIdx ...` block) into a helper that
  both skeletons call, leaving each variant's sync structure intact. Cuts most of
  the duplication with minimal concurrency risk.
- **Option B: drop the non-hoisted variant entirely.** Production never uses it;
  it exists only for stats/measure builds. If the measurement value is low, delete
  `HoistNeighborOrder == false` and always hoist. This removes the duplication
  outright and eliminates the divergence-risk surface — at the cost of the
  measure-build comparison it was presumably added for. Confirm with whoever added
  the measure path whether the non-hoisted numbers are still needed.
- **Option C (highest risk): full unification** into one templated body. Only do
  this with the checklist above and the hang-detection harness below, and expect
  iteration.

## Testing for hangs

**A hang here presents as a test that never returns, not a failing assertion.**
The default test invocation can mask it (a stuck kernel looks like a slow test).
Before touching this code, stand up hang detection:

1. **Tight per-test timeouts.** Run the fMCS tests under a wall-clock kill so a
   deadlock fails fast and visibly rather than hanging CI:
   - `ctest --timeout 60 -R fmcs` (fail any fMCS test that exceeds 60 s), or
   - wrap the gtest binary: `timeout 60 ./test_fmcs_unit --gtest_filter='*Substructure*'`.
   Pick a timeout well above the real runtime but far below "CI hung" (tens of
   seconds for the unit tests).
2. **Run in a loop.** Cooperative hangs can be intermittent (they depend on the
   lane arrival pattern). Loop the relevant tests many times under the timeout:
   `for i in $(seq 1 200); do timeout 60 ./test_fmcs_unit --gtest_filter='*Substruct*' || { echo "HANG/FAIL on iter $i"; break; }; done`.
3. **Exercise both instantiations.** Add/enable a test that runs
   `matchSeedSubstructureCooperative<true>` **and** `<false>` on the same inputs
   and asserts identical mappings — this both closes the coverage gap and gives
   the loop something that hits the merged code from both sides.
4. **Sanitizers.** Run under `compute-sanitizer --tool synccheck` (flags
   divergent/`__syncthreads`-style barrier misuse) and `--tool racecheck` on the
   shared scratch. `synccheck` is the most likely to catch a partial-group
   barrier directly.
5. **Diverse shapes.** Deadlocks show up on inputs where `numPartials`,
   `targetScanEnd`, and `scratch.found` timing vary across lanes — use several
   tier sizes and both dense and sparse target graphs, not just one small case.

Only widen from the small unit driver to the full integration suite once the
looped, timeout-guarded unit runs are clean.
