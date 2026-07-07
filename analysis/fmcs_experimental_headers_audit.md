# fMCS experimental headers audit

**Status:** per-header assessment from review of branch `fmcs-20260630-cleanup`
(merge-base `331be7d`). No files deleted. The question this answers: for each
header under `src/mcs/**/experimental/`, is it a **genuine future optimization**
worth keeping, or **truly dead** and safe to drop?

Method: read each header in full and grep the whole tree for `#include` of it
(both `experimental/<name>` and bare `<name>`). "Includers" below is the exact,
verified set.

## Verdict table

| Header | Includers | Verdict |
| --- | --- | --- |
| `fmcs_cuda/experimental/fmcs_match_cache.cuh` | `tests/test_fmcs_unit.cu` | **KEEP — future opt, in progress + tested** |
| `fmcs_cuda/experimental/fmcs_candidate_count_cache.cuh` | none | **KEEP — future opt, blocked on a known race** |
| `fmcs_cuda/experimental/fmcs_sorted_seed_queue.cuh` | none | **KEEP — future opt, real algorithmic alternative** |
| `fmcs_cuda/experimental/fmcs_match_with_fallback.cuh` | `tests/test_fmcs_unit.cu` | **DROP-ish — superseded reference** |
| `fmcs_cuda/experimental/fmcs_seed_mark.cuh` | `tests/test_fmcs_unit.cu` | **DROP-ish — abandoned approach** |
| `fmcs_cuda/experimental/fmcs_queue_scopes.cuh` | none | **DROP — empty placeholder** |
| `mcs_common/experimental/mcs_bitops.cuh` | none | **DROP — unused generic utility** |
| `mcs_common/experimental/mcs_cooperative_copy.cuh` | none | **DROP — low-value / speculative** |

No shipping (non-test) source includes **any** experimental header. Three are
compiled in CI via `test_fmcs_unit` (so they won't bit-rot); five are compiled by
nothing and will silently rot.

---

## Keep — genuine future optimizations

### `fmcs_match_cache.cuh` — cross-lineage success cache
This is the "caching that was in progress but unfinished." It is the most
developed of the experimental set: `DeviceMatchCache` (a per-block,
global-backed, open-addressed hash table of successful `(seed shape, embedding)`
keys) plus `mappingHashWithinThread` (a canonical-ordering SplitMix64 hash over
seed atoms/bonds/frontier/exclusions). It is well-documented, thought-through
(failures deliberately not cached so they can't poison lineages; 0-sentinel
handling; power-of-two capacity), and **unit-tested** (8 `Cache*` tests in
`test_fmcs_unit.cu`, which the file's own comment says are its only consumer).
This mirrors RDKit FMCS's duplicate-seed pruning and is a legitimate search-speed
optimization. **Keep**, and when picking it back up, wire it into the kernel
behind a flag and add an RDKit-parity test that the cached path yields identical
MCS sizes to the uncached path.

### `fmcs_candidate_count_cache.cuh` — cached fallback candidate counts
`countCandidateTargetAtomsCachedCooperative` caches per-`(queryAtom, degree)`
candidate counts to avoid recomputing them per warp group. **It has a known
blocker, stated in its own header comment:** the active kernel recomputes counts
per group precisely because a block-shared cache **races** when groups enter the
fallback concurrently; a future use "must provide group-private storage or
explicit cross-group synchronization." So this is a real optimization (it targets
the fallback recompute called out as perf item #5 in
`analysis/fmcs_performance_options.md`) that was parked on a correctness issue.
**Keep**, but only revive alongside the group-private-storage fix; do not enable
as-is.

### `fmcs_sorted_seed_queue.cuh` — bond-count-sorted seed scheduling
Sorted-insert / pop-front operations (within-thread and cooperative) that keep the
seed worklist ordered by descending bond count, an RDKit-parity `SEED_GROW`
scheduling discipline. The active kernel uses a LIFO stack instead (simpler and
concurrency-friendly). Sorted/best-first scheduling can improve incumbent-bound
pruning, so this is a genuine algorithmic alternative worth retaining as a
starting point. **Keep**, with two caveats:
- It is the **only** consumer of the unused within-thread `SeedQueue` API
  (`setSizeWithinThread`, `slot`, `size`, `capacity`, `clear`, `full`, `empty`)
  that the kernel review flagged as dead. That API's fate is tied to this header:
  keep both or drop both.
- It `#include`s the shipping `mcs_common/mcs_cooperative_copy.cuh` and
  `fmcs_seed_queue.cuh` (not the experimental copy) — so it is self-consistent
  with production headers.

---

## Drop-ish — superseded or abandoned (test-compiled, so not urgent)

### `fmcs_match_with_fallback.cuh` — composite matcher
`matchSeedWithSubstructureFallbackCooperative` bundles "try fast incremental
match, else take the locked substructure fallback." Production deliberately does
this same sequencing **inline and explicitly** (it selects its match path by
hand), so this wrapper is a **superseded convenience reference**. It is unit-
tested (`MatchSeedFallbackRebuilds...`), so it is compiled and correct, but it
adds no capability production lacks and duplicates the fast→fallback logic.
**Lean delete**; if kept, label it clearly as a reference implementation, not a
production path. (Related to the duplicated-match concern in
`analysis/fmcs_duplicated_match_impl.md`.)

### `fmcs_seed_mark.cuh` — alternative last-added-atom marking
`seedMarkLastAddedAtomWithinThread` marks an existing seed atom as boundary
without changing `seed.atoms`/`numAtoms`. The header comment says the RDKit-shaped
kernel derives the frontier solely from `seedAddAtomWithinThread`, so this is an
**abandoned alternative frontier approach** kept "for targeted experiments." Tiny
and unit-tested, but no path forward is articulated. **Lean delete** unless a
specific frontier experiment is planned.

---

## Drop — truly dead, low value

### `fmcs_queue_scopes.cuh` — empty scope tags
Two empty structs (`ClusterScope`, `GridScope`) marking "potential distributed
shared-memory and device-global queue implementations. Neither scope is
implemented today." This is an **aspirational placeholder**, not code — it
carries no logic. The idea (cross-block / cluster-wide queue for very large
problems) is legitimate and connects to the block-size scaling discussion in
`analysis/fmcs_scratch_placement_plan.md`, but empty tag structs are not useful
scaffolding. **Delete the header; capture the idea as a line in a roadmap doc**
instead.

### `mcs_common/experimental/mcs_bitops.cuh` — generic bit-mask helpers
`popcount`/`anySet`/`ctz`/`lowestBit`/`nthSetBit` over word arrays, host+device.
Correct and generic, but **used by nothing**, and the shipping code uses its own
inlined `__ffs`/`__ffsll`/`__popc` (e.g. in `mappingHashWithinThread`). It is a
utility looking for a caller. **Delete**; if a future algorithm needs these, they
are trivial to reintroduce (or promote to a real shared `mcs_common` utility
header at that point, not an `experimental/` one).

### `mcs_common/experimental/mcs_cooperative_copy.cuh` — async/block copies
Two functions: `warpCopyAsync` (a `cg::memcpy_async` wrapper that **waits
immediately**, so it provides no overlap benefit over a plain copy — the comment
admits callers needing overlap should use cg ops directly) and `blockCopy` (a
block-wide int4 coalesced copy). Neither is used. Note this is **not** a duplicate
of the shipping `mcs_common/mcs_cooperative_copy.cuh`, which provides the
*warp*-scoped `warpCopy` used by 4 production headers — the functions here are
different and unused.
- `warpCopyAsync`: **delete** — a no-overlap async wrapper is a dead end.
- `blockCopy`: could matter *if* a future path does block-wide (not per-group)
  copies; today the kernel is group-oriented so it is dead. If you want to retain
  it, move it into the shipping `mcs_cooperative_copy.cuh` next to `warpCopy`
  rather than leaving it stranded in `experimental/`.

---

## Bottom line

- **Keep 3** as staged future work, each with a documented next step:
  `fmcs_match_cache.cuh` (finish + wire in behind a flag), `fmcs_candidate_count_cache.cuh`
  (revive with group-private storage), `fmcs_sorted_seed_queue.cuh` (best-first
  scheduling; decide jointly with the within-thread `SeedQueue` API).
- **Drop 5**: `fmcs_queue_scopes.cuh`, `mcs_bitops.cuh`,
  `mcs_common/experimental/mcs_cooperative_copy.cuh`, and (lower urgency, since
  they're test-compiled) `fmcs_match_with_fallback.cuh` and `fmcs_seed_mark.cuh`.
- Whatever is kept: add a short `experimental/README.md` stating, per file, the
  intended future use and its current blocker, so "experimental" means "staged
  work with a plan" rather than "unowned dead code." The 5 dead headers compile
  in no TU and will bit-rot otherwise.
