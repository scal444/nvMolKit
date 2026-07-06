# fMCS experimental headers

Staged future optimizations, each with a concrete next step. See
`analysis/fmcs_experimental_headers_audit.md` for the full assessment.

- **`fmcs_match_cache.cuh`** — cross-lineage success cache mirroring RDKit
  FMCS's duplicate-seed pruning. Unit-tested (`Cache*` tests in
  `test_fmcs_unit.cu`). Next step: wire into the kernel behind a flag and add
  an RDKit-parity test that the cached path yields identical MCS sizes.
- **`fmcs_candidate_count_cache.cuh`** — cached fallback candidate counts.
  Blocked: a block-shared cache races when groups enter the fallback
  concurrently (see header comment). Revive only with group-private storage
  or explicit cross-group synchronization.
- **`fmcs_sorted_seed_queue.cuh`** — bond-count-sorted (best-first) seed
  scheduling, an alternative to the kernel's LIFO stack that can improve
  incumbent-bound pruning. Sole consumer of the within-thread `SeedQueue`
  API; keep or drop the two together.

Ideas captured from deleted headers: a cluster/grid-scoped work queue for
very large problems (was `fmcs_queue_scopes.cuh`) relates to the block-size
scaling discussion in `analysis/fmcs_scratch_placement_plan.md`.
