# fMCS triplicated adjacency logic

**Status:** cleanup/refactor note from review of branch `fmcs-20260630-cleanup`
(merge-base `331be7d`). Not implemented. Line numbers as of that branch —
re-grep before editing.

## Summary

The device matcher branches three ways on how it walks target adjacency, and one
of the three branches is **dead in both production and tests**. The result is
that `matchIncrementalFastCooperative` (and its helper
`findTargetBondBetweenAtomsWithinThread`) carry roughly 3× the body they need.
This is a redundancy/organization issue, not a correctness bug — all three
branches compute the same predicate.

## The three branches

Every adjacency site is gated on `topologyHasAdjacencyBondIndices<TargetTopology>()`,
which is a compile-time property of the topology type:

1. **Compile-time-adjacency branch** — `if constexpr (topologyHasAdjacencyBondIndices<T>())`
   true. Uses the CSR `rowOffsets`/`colIndices`/`bondIndices` directly.
2. **Runtime-adjacency (middle) branch** — the `else` block's
   `scanAdjacency = (rowOffsets != nullptr && colIndices != nullptr && ...)`
   path: a *non-adjacency* topology type that nonetheless carries non-null CSR
   pointers at runtime.
3. **Full-scan branch** — the `else` block's `scanAdjacency == false` path:
   iterate all target atoms/bonds.

## Why the middle branch is dead

- **Production** only ever instantiates the matcher with `DeviceCsrView`, whose
  `kHasAdjacencyBondIndices == true` (`fmcs_kernel_types.cuh:16`,
  wired at `fmcs_kernel.cuh:607-619`). So the compile-time-true branch (1) is
  always taken.
- **Tests** use `TestCsrView` (`test_fmcs_unit.cu:703-712`), which is *always*
  constructed with `rowOffsets = colIndices = bondIndices = nullptr`
  (`test_fmcs_unit.cu:791-792, 968-1084, 1273-1328`). For that type
  `topologyHasAdjacencyBondIndices<>()` is false and `scanAdjacency` is always
  false → the full-scan branch (3) is taken.
- Therefore branch (2) — a non-adjacency topology with non-null CSR pointers — is
  **never executed by any code path**, production or test.

## Locations

Confirmed at (re-grep, the file has since had comments added near the top):

- `fmcs_match.cuh:279-338` — `matchIncrementalFastCooperative` main adjacency walk.
- `fmcs_match.cuh:389-472` — second adjacency site in the same function.
- `fmcs_match.cuh:587-631` — third site.
- `fmcs_match.cuh:1341-1466` — related duplication in the substructure path.
- `fmcs_match.cuh:495-533` — `findTargetBondBetweenAtomsWithinThread` mirrors the
  same three-way split (see the `if constexpr (...) { ... } else { scanAdjacency ... }`
  structure).

## Recommendation

Two options, in preference order:

1. **Delete the dead middle branch.** Since no topology is both
   non-adjacency-typed and CSR-pointer-bearing, the `else` block only needs its
   full-scan path. This removes ~1/3 of each site with zero behavior change.
2. **Collapse (1) and (3) into one helper** templated on the compile-time
   adjacency flag, so the CSR-walk vs full-scan choice lives in a single place
   used by both `matchIncrementalFastCooperative` and
   `findTargetBondBetweenAtomsWithinThread`, instead of being copy-pasted at
   four sites.

Either way, add a `static_assert` or a comment documenting the invariant "a
topology either has compile-time adjacency (production) or no CSR pointers
(tests)" so the dead combination cannot silently reappear.

## Testing

Pure refactor; the safety net is the existing unit tests plus the RDKit-parity
integration tests. Because the change touches the matcher inner loop, run the
full `test_fmcs_unit` + `test_fmcs` + integration suites and confirm identical
results. If you collapse into a shared helper (option 2), add a test that
instantiates the matcher against **both** a `DeviceCsrView`-like (adjacency) and
a `TestCsrView`-like (full-scan) topology on the same small graph and asserts
identical mappings, to lock the two live branches together.
