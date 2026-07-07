# fMCS scratch placement + shared-memory carveout readiness

**Status:** design/implementation plan. Not yet implemented. Intended to be
handed to a fresh implementer (human or model) on a new branch.

**Author context:** written against branch `fmcs-20260630-cleanup`
(merge-base `331be7d`). All line numbers below are as of that branch; re-grep
before editing.

---

## 1. Goal

Make the fMCS kernel run at `blockSize == 512` for the tier-128 size class
(molecules with 65–128 atoms/bonds), which today throws for the whole batch,
**and** lay the groundwork to scale to larger block sizes in the future — without
starving the L1 cache.

We do this two ways, both landing in this change:

1. **Primary mechanism — template the *location* of the large per-group
   substructure scratch** (shared vs. a per-block global slab). Moving the
   biggest, coldest per-group array out of static shared drops `512 @ tier-128`
   from ~70 KB to ~36 KB of static shared, i.e. back under the 48 KB static cap,
   with **no** dynamic-shared opt-in required, and it *reduces* L1 pressure at
   every configuration.

2. **Readiness mechanism — wire up (but do not yet depend on) the extended
   shared-memory carveout** (`cudaFuncAttributeMaxDynamicSharedMemorySize`) so a
   future shared-placement config can exceed 48 KB on architectures that allow
   it. See §7 for exactly what lands now vs. what is a documented follow-on.

The scratch-location choice must be **piped through to the Python API, the
autotuner, the benchmark harness, and the test suites** so it is measurable and
tunable, not just an internal constant.

## 2. Non-goals

- Do **not** change the matching algorithm, the queue protocol, or the result
  semantics. This is a memory-placement refactor only; outputs must be identical.
- Do **not** convert the remaining static `__shared__` arrays to dynamic
  `extern __shared__` in this change. That conversion is what would let
  *shared*-placement exceed 48 KB; it is the documented follow-on in §7. This
  change only adds the carveout plumbing and the shared-byte accounting it needs.
- Do **not** attempt `blockSize > 512` here. The design must not *block* it
  (see §6), but new block sizes are out of scope.

## 3. Root cause recap (why 512 @ tier-128 throws today)

- The kernel's per-group scratch is **static** `__shared__`, sized
  `[kNumGroups]` where `kNumGroups = blockThreads / 32`
  (`src/mcs/fmcs_cuda/fmcs_config.cuh:26`). Block 128 → 4 groups; block 512 → 16
  groups.
- Several of those arrays also scale with the tier (`maxAtoms`/`maxBonds`), so
  total static shared ≈ `numGroups × per-group-scratch(tier)`
  (`src/mcs/fmcs_cuda/fmcs_kernel.cuh:507-556`).
- `cudaFuncAttributePreferredSharedMemoryCarveout` (set in
  `configureSharedMemCarveout`, `fmcs_launch.cu:18`) only shifts the L1/shared
  split; it does **not** lift the **48 KB static-shared-per-block cap**. Static
  shared above 48 KB requires `extern __shared__` + the max-dynamic-shared
  opt-in.
- Therefore `fmcsKernel<128,128,512,...>` cannot compile and is never
  instantiated (`fmcs_launch.cu:230-241` stop at `<64,64>` for the 512 variant).
  The host guards at `fmcs.cpp:854-858`, `:778-781`, `:797-800` throw
  `std::invalid_argument` rather than hit an undefined symbol.

### Measured static-shared footprint (hand-computed from struct layouts)

Tier 128 → `BitWord` is `uint64`, 2 words per bitset. Key `sizeof`s:

| Type | Bytes (tier 128) |
| --- | --- |
| `QueuedSeed<128,128,128,128>` | 384 (`Seed` 80 + `MatchResult` 296, `alignas(16)`) |
| `FmcsSubstructureScratch<128,128,128>` | ~2068 |
| `NewBond` | 10 |
| `ExecutionStats` | 160 |

Per-group shared (production build, no stats), ~4.29 KB:

| Per-group item | Bytes | Access |
| --- | --- | --- |
| `FmcsSubstructureScratch substructureScratch` | ~2068 | **cold** (fallback path only) |
| `currentStorage` + `biggestStorage` (2× `QueuedSeed`) | 768 | hot |
| `newBondsArr[128]` | 1280 | hot (grow step) |
| `remainingAtomStack[128]` + bitsets + counters | ~170 | warm |

Plus ~0.9 KB fixed block-wide state (`bestStorage` 384, two `DeviceCsrView`,
`measureStats` + `groupStats[1]` 160 each, locks, clocks, `initialExcludedBonds`).

Totals:

| Config | Groups | Static shared (all in shared) | With `substructureScratch` moved to global |
| --- | --- | --- | --- |
| 128 @ tier-128 | 4 | **~18 KB** | ~10 KB |
| 512 @ tier-128 | 16 | **~70 KB** (over 48 KB cap → won't compile) | **~36 KB** (fits) |

**Confirm before implementing:** the build already passes `--ptxas-options=-v`
(`src/mcs/CMakeLists.txt:25`). Build the tier-128 TU and read the real
`smem=` for `fmcsKernel<128,128,128,...>` to validate the ~18 KB figure and to
re-check ~36 KB for the new 512 kernel after the move.

## 4. Design

### 4.1 Scratch-location policy

Add a compile-time enum and thread it through the kernel/launch as a template
parameter; add a runtime knob that selects it.

```cpp
// new: src/mcs/fmcs_cuda/fmcs_policy.cuh (or fmcs_config.cuh)
enum class FmcsScratchLocation { Shared, Global };
```

- **Kernel template gains one param:**
  `fmcsKernel<maxAtoms, maxBonds, blockThreads, CollectTimings, CollectStats,
  FmcsScratchLocation Scratch>` (default `Shared` to preserve current codegen).
- Inside the kernel, `substructureScratch` becomes an *accessor*, not a fixed
  `__shared__` array:
  - `Scratch == Shared`: keep `__shared__ SubstructureScratchT
    substructureScratch[kNumGroups];` exactly as today.
  - `Scratch == Global`: take a new global pointer param
    `SubstructureScratchT* scratchStorageAll` and index
    `scratchStorageAll[pairIdx * kNumGroups + groupId]`. **Do not** allocate the
    `__shared__` array in this specialization (use `if constexpr` so the shared
    array is not declared and does not count against the budget).
  - `SubstructureScratchT& mySubstructureScratch = ...;` (line
    `fmcs_kernel.cuh:561`) is the single site the matcher reads; keep that
    reference so `fmcs_match.cuh` is untouched.
- Struct is unchanged. `FmcsSubstructureScratch` is a POD; it already carries
  `alignas` via its members. When placed in the global slab, ensure the slab base
  and stride are 16-byte aligned (it contains no >8-byte members, so 8 is
  sufficient, but match the `QueuedSeed` convention of 16 to be safe).

**Runtime policy (`Auto`) — recommended default:**
- Add `FmcsScratchLocation` with a third *host-only* sentinel `Auto` at the
  public-parameter layer (see §4.4). `Auto` resolves per (blockThreads, tier):
  - `Global` when `blockThreads == 512 && tier == 3` (the 128 tier),
  - `Shared` otherwise.
- Rationale: keep small/hot configs on fast shared; only pay global-memory
  latency for the fallback scratch where static shared cannot fit. The scratch
  is fallback-only, so the latency lands on the already-slow path.
- Explicit `Shared` / `Global` overrides are honored for testing and
  benchmarking (e.g. force `Global` at tier-64 to measure the latency delta, or
  request `Shared` to confirm the clear-error path below).
- **Illegal combo must error clearly, not silently:** explicit `Shared` +
  `512 @ tier-128` cannot fit static shared. Reject it at dispatch with a
  message like `"fMCS scratchLocation=shared cannot satisfy blockSize 512 at
  tier-128 (needs ~70 KB static shared > 48 KB); use scratchLocation=global or
  auto"`. Do **not** fall through to a missing instantiation.

### 4.2 Launch layer (`fmcs_launch.cu` / `fmcs_launch.cuh`)

- `launchFmcsKernelSpecialization` and `launchFmcsKernelSelected` gain the
  `FmcsScratchLocation Scratch` template param and a
  `SubstructureScratchT* scratchStorage` (or `void* scratchStorage`) runtime
  param (nullptr when `Shared`).
- New sizing helper mirroring the existing ones:
  ```cpp
  template <int blockThreads, int maxAtoms, int maxBonds>
  std::size_t fmcsScratchStorageBytes(std::size_t numPairs);
  // = numPairs * FmcsBlockConfig<blockThreads>::numGroups
  //            * sizeof(FmcsSubstructureScratch<maxAtoms,maxBonds,maxAtoms>)
  ```
- `launchFmcsKernel128` / `launchFmcsKernel512` (declared in
  `fmcs_launch.cuh:28-54`) gain the `scratchStorage` param and forward the
  scratch-location choice. Simplest: pass the resolved `FmcsScratchLocation` as a
  runtime arg and branch to the right template specialization inside the `.cu`
  (bounded set: Shared/Global), keeping the exported signature non-templated on
  the enum.
- **Instantiation matrix (this is the crux):** add the 512 @ tier-128 kernel
  under **Global** scratch, plus its storage helpers:
  ```
  launchFmcsKernel512<128,128>            // NEW (Global scratch only)
  fmcsSubstructureStorageBytes<512,128>   // NEW
  fmcsScratchStorageBytes<512,128,128>    // NEW
  fmcsScratchStorageBytes<128,128,128>    // NEW (for optional Global at 128)
  ... (add Global specializations for the tiers you expose Global on)
  ```
  Keep the existing 128-block `<128,128>` and 512-block `<16..64>` kernels as
  `Shared`. Decide (see §11) whether to also instantiate `Global` variants for
  the smaller tiers to allow forced-Global benchmarking; at minimum instantiate
  Global for every (blockThreads, tier) the public API can request.
- **Carveout readiness hooks (see §7):**
  - Add `template<...> std::size_t fmcsKernelStaticSharedBytes()` OR query
    `cudaFuncGetAttributes(&attr, kernel)` and read `attr.sharedSizeBytes` after
    selecting the specialization. Prefer the runtime query — it is exact and
    tracks compiler temporaries.
  - Add `configureFmcsKernelMaxDynamicSharedMem(kernel, bytes)` that calls
    `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
    bytes)`; call it (guarded) only when a dynamic-shared layout is in use.
    Today it is a no-op path because all shared is static; land it wired but
    dormant, behind the `Parameters` flag in §4.4.

### 4.3 Host dispatch (`src/mcs/fmcs_cuda/fmcs.cpp`)

- `runBatchWithBlockSize` (`:814`): replace the blanket tier-3 throw (`:854-858`)
  with placement resolution. After building `tierIndices`, for each non-empty
  tier compute the `FmcsScratchLocation` from `params` (`Auto` → §4.1 rule).
  Only throw on the genuinely-unsupported explicit combo (§4.1).
- The two `std::visit` guards in `launchChunk`/`drainChunk` (`:778-781`,
  `:797-800`) must be removed/replaced: route the tier-128 chunk to
  `launchTierChunk<512>` with `Global` scratch instead of throwing.
- **Executor buffers:** `FmcsExecutor` currently owns `queueStorage` and
  `substructureStorage` `AsyncDeviceVector`s and grows them via
  `ensureScratchCapacity` (`fmcs.cpp:423-455`). Add a third slab
  `scratchStorage`, sized by `fmcsScratchStorageBytes<blockThreads,maxAtoms,
  maxBonds>(numPairs)`, grown with the same free-before-alloc pattern, and pass
  its `.data()` as `dScratch` into the launch. When placement is `Shared`, keep
  it at size 0 / pass nullptr.
  - Peak-memory note: `512 @ tier-128` scratch slab per chunk ≈
    `chunkSize(512) × 16 groups × 2068 B ≈ 16.9 MB` per executor — small next to
    the existing partial-storage slab. Record it but it is not a concern.
- `Parameters` (`fmcs.cuh`) gains the placement field (see §4.4). Map it in
  `launchTierAsync` when selecting the kernel.

### 4.4 Public C++ parameter surface

- **`src/mcs/fmcs_cuda/fmcs.cuh` `struct Parameters`** (`:54-73`): add
  ```cpp
  FmcsScratchLocation scratchLocation = FmcsScratchLocation::Auto; // host-only Auto
  bool enableExtendedSharedCarveout = false; // §7 readiness; default off
  ```
  Update the doc comment; note "512 supports tiers up to 64" line (`:56`) becomes
  "512 supports tier-128 when scratch is placed in global memory".
- **`src/mcs/mcs_types.h` `struct MCSParameters`** (`:52-69`): add a
  matching `MCSScratchLocation scratchLocation = MCSScratchLocation::Auto;` enum
  (define alongside `MCSAtomCompare` etc.), plus optionally the carveout flag.
- **`src/mcs/mcs_search.cpp` `runGpuPairs`** (`:240-246`): map
  `params.scratchLocation` → `fmcsParams.scratchLocation`.
- **`src/mcs/mcs_rdkit_adapter.cpp` `shouldFallbackToRDKit`** (`:271-297`):
  today it does not consider `blockSize` and lets 512+tier-128 reach the GPU and
  throw. After this change the GPU path handles 512+tier-128, so **no new
  fallback gate is needed** — but add a comment stating that 512+tier-128 is now
  GPU-supported via global scratch, so nobody re-adds a gate. (This resolves the
  separate "512 throws the whole batch" correctness finding.)

### 4.5 Python bindings + API

- **`nvmolkit/mcs.cpp`**: read a new option key (e.g. `"scratchLocation"` as a
  string `"auto"|"shared"|"global"`) into `MCSParameters::scratchLocation`.
  Follow the existing key-by-key pattern (`mcs.cpp:360-388`). Add the carveout
  flag key if exposing it.
- **`nvmolkit/mcs.py`**:
  - `MCSConfig.__init__` (`:161`) gains `scratchLocation: str = "auto"`; store,
    validate against `{"auto","shared","global"}` (raise `ValueError` on bad
    input, matching the friendly-error style), and include it in `to_dict` /
    `to_kwargs` / `from_dict`.
  - `findMCS` (`:261`) gains a `scratch_location: str = "auto"` flat kwarg
    guarded by the same config-vs-kwargs mutual-exclusion check already present.
  - Document it in the `MCSConfig` and `findMCS` docstrings, including the
    "auto picks global for 512@tier-128" behavior and the shared-vs-global
    latency tradeoff.
- **Autotune persistence:** `nvmolkit/autotune/_persistence.py` is additive-tag
  based; adding a field to the serialized `MCSConfig` dict is backward-compatible
  as long as `from_dict` defaults missing keys to `"auto"`. Add a test
  (see §4.8) that an old-format dict without `scratchLocation` still loads.

### 4.6 Autotuner (`nvmolkit/autotune/tune_mcs.py`)

- Add `scratchLocation` to the search space as a categorical:
  `["auto", "shared", "global"]`. Given the shared cap, the meaningful sweep is
  at `blockSize == 512`: `global` unlocks tier-128 and may change occupancy at
  smaller tiers too. Ensure the generated configs never emit
  `blockSize=512, scratch=shared` for datasets that contain tier-128 molecules
  (either prune that combo in the search-space generator or rely on the
  clear-error and skip). Prefer pruning so tuning does not waste trials on a
  guaranteed error.
- Reuse the shared spec helpers in `_core.py`; do not copy-paste.

### 4.7 Benchmarks (`benchmarks/mcs_bench.py`)

- Add a `--scratch-location {auto,shared,global}` CLI flag (default `auto`) via
  the existing `_add_option`/`legacy_flags` helper so a `--scratch_location`
  alias also works. Pass it into the `MCSConfig` used by `_bench_nvmolkit`.
- Add `scratch_location` (and, if exposed, the carveout flag) as a **column in
  the output CSV** so sweeps are self-describing. Verify against
  `FMCS_STATS_COLUMNS` / the writer so no `KeyError`.
- Add a documented sweep example in the file header / `--help` epilog:
  block-size 512 across a tier-128-containing dataset with
  `--scratch-location global`, comparing throughput vs. block-128.
- No new dependency on `analysis/` paths; keep the committed
  `benchmarks/data/chembl_10k.smi` default.

### 4.8 Tests

Split by the existing three-tier C++ layout + Python.

- **`tests/test_fmcs_unit.cu`** (device primitives): add a driver-kernel test
  that runs the matcher with `Scratch == Global` on the same inputs as an
  existing `Shared` test and asserts **bit-identical** match results
  (same atom/bond mapping, same size). This is the core parity guarantee that
  the placement change is behavior-preserving. Use a small tier so the driver
  kernel is cheap.
- **`tests/test_fmcs.cpp`** (graph-level batch API):
  - Replace the existing "block 512 rejects tier-128" test (it currently asserts
    the throw — grep for the tier-128/512 rejection case) with:
    1. `blockSize=512` + a tier-128 pair + `scratchLocation=Auto` (or `Global`)
       **succeeds** and matches the block-128 result for the same pair.
    2. `blockSize=512` + tier-128 + explicit `scratchLocation=Shared` **throws**
       with the clear message from §4.1.
  - Add a mixed-tier batch at `blockSize=512` (tiers 16/32/64 on shared, 128 on
    global) verifying per-tier placement and that no pair is dropped
    (mirrors the existing `OverflowPairDoesNotBlockNeighbors` intent).
- **`tests/test_fmcs_integration.cu`** + **`nvmolkit/tests/test_mcs_integration.py`**:
  parametrize the RDKit-parity suites over `scratch_location ∈ {auto, global}`
  (and `shared` for tiers ≤64) at `block_size=512`, and add a tier-128 fixture
  (e.g. a 128-carbon chain, as `test_python_binding_dispatches_every_gpu_tier_boundary`
  already builds at `:313`) so the 512@128 path is actually exercised end-to-end
  against `rdFMCS.FindMCS`. The Python dispatch-options test
  (`test_mcs_integration.py:255-286`) already has a `block-512` case — extend it
  with a `scratch_location` axis and ensure it includes a tier-128 molecule.
- **`nvmolkit/tests/test_mcs.py`**: add option-validation tests — invalid
  `scratch_location` string raises `ValueError`; `allow_rdkit_fallback=False` +
  `block_size=512` + tier-128 now **succeeds** (previously the error path);
  round-trip `MCSConfig.to_dict`/`from_dict` with and without the new key
  (backward-compat, §4.5).

## 5. Instantiation matrix (before → after)

Current explicit instantiations in `fmcs_launch.cu:213-254`:

```
launchFmcsKernel128<16,16> <32,32> <64,64> <128,128>       (Shared)
launchFmcsKernel512<16,16> <32,32> <64,64>                 (Shared)
fmcsSubstructureStorageBytes<128, 16|32|64|128>, <512, 16|32|64>
```

After:

```
launchFmcsKernel128<16,16> <32,32> <64,64> <128,128>       (Shared)   [unchanged]
launchFmcsKernel512<16,16> <32,32> <64,64>                 (Shared)   [unchanged]
launchFmcsKernel512<128,128>                               (Global)   [NEW]
+ Global specializations for any (blockThreads,tier) the API may force to Global (§11)
fmcsSubstructureStorageBytes<512,128>                                 [NEW]
fmcsScratchStorageBytes<512,128,128>  (+ others exposed to Global)    [NEW]
```

Each added instantiation multiplies by the enabled `(CollectTimings,
CollectStats)` combinations, same as today. Watch compile time / binary size —
the tier-128 kernels are the largest TUs. Only instantiate Global for configs
the API can actually request.

## 6. Forward-compatibility (blockSize > 512)

The design must not wall off 1024. With `substructureScratch` in global,
`1024 @ tier-128` (32 groups) is `0.9 + 32 × 2.22 KB ≈ 72 KB` static shared —
over the cap again. So for >512 the plan is to relocate the *next* biggest hot
arrays behind the same template mechanism:

- `newBondsArr` (1.28 KB/group) and the `current`/`biggest` `QueuedSeed` copies
  (0.75 KB/group) are the next candidates.

Do **not** implement these now, but structure the `FmcsScratchLocation` accessor
so extending it to cover additional arrays is mechanical (e.g. a small
`FmcsScratchPlan` struct of per-array locations, or reuse the single enum for a
grouped "big scratch" set). Leave a TODO pointing here.

## 7. Carveout readiness — what lands now vs. follow-on

**Lands now (dormant but wired):**
- `Parameters::enableExtendedSharedCarveout` (default `false`) and the
  `MCSParameters` mirror.
- A launch-side path that, when a dynamic-shared layout is selected AND the flag
  is set, computes required bytes and calls
  `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, N)`
  before launch, and passes `N` as the third `<<<grid, block, N, stream>>>`
  argument. Keep `configureSharedMemCarveout` (preferred carveout) as-is.
- Exact shared accounting via `cudaFuncGetAttributes(...).sharedSizeBytes`,
  used both to validate the static budget and to size the dynamic request.

**Follow-on (NOT in this change):**
- Converting the static `__shared__` arrays that remain in `Shared` placement to
  a single `extern __shared__` dynamic block. This is the only thing that lets
  *shared*-placement actually exceed 48 KB; until then the flag has no effect for
  the default static layout. Document this explicitly next to the flag so nobody
  assumes the flag alone enables >48 KB shared.

Rationale for shipping it dormant: the user wants the carveout ability staged so
the branch that adds dynamic shared later does not also have to re-plumb the API,
executor, and benchmark/test flags.

## 8. Acceptance criteria

1. `blockSize=512` on a batch containing tier-128 molecules **returns results**
   (no throw) with `scratchLocation` `auto` or `global`, and those results are
   identical (atom/bond mappings and sizes) to the `blockSize=128` results for
   the same pairs.
2. Explicit `scratchLocation=shared` + `512 @ tier-128` raises the clear,
   documented error — not an undefined-symbol/link failure, not a silent wrong
   answer.
3. ptxas `smem=` for `fmcsKernel<128,128,512,...,Global>` is under 48 KB
   (target ~36 KB; confirm with `--ptxas-options=-v`).
4. The `scratch_location` option is settable and observable from Python
   (`findMCS` kwarg + `MCSConfig`), appears in the benchmark CSV, and is in the
   autotune search space.
5. RDKit-parity integration tests pass across `{auto, global}` at `block_size=512`
   including at least one tier-128 molecule.
6. Old-format autotune configs (no `scratchLocation`) still load.
7. No change to results, timing-stat semantics, or the default path when
   `blockSize=128` (regression guard: existing tests unchanged and green).

## 9. Risks & mitigations

- **Behavioral drift Shared vs Global.** Mitigate with the direct unit-test
  parity check (§4.8) and the batch parity check; the struct and matcher code are
  untouched, only the address of the scratch changes.
- **Global-scratch latency on the fallback path.** Acceptable by design (cold
  path), but measure with the benchmark's `--scratch-location` sweep at a tier
  where both placements are legal (e.g. tier-64) to quantify the delta.
- **Occupancy shift.** Dropping shared per block can *raise* occupancy, which is
  usually good but can change register/L1 behavior; capture `smem`/occupancy from
  ptxas and an NCU snapshot before/after (there is prior NCU tooling under
  `analysis/fmcs_ncu/`).
- **Compile-time/binary-size growth** from new instantiations. Only instantiate
  Global for API-reachable configs; consider gating rarely-used Global tiers
  behind a build option if TU time regresses badly.
- **Global slab alignment.** Ensure 16-byte base+stride for the scratch slab to
  avoid `cudaErrorMisalignedAddress` (same class of bug the `QueuedSeed`
  `alignas(16)` comment at `fmcs_seed.cuh:124-130` guards against).

## 10. Suggested commit sequencing

1. Add `FmcsScratchLocation` enum + kernel template param + `if constexpr`
   accessor; default `Shared`; **no behavior change**, verify identical codegen
   path for existing instantiations.
2. Add `fmcsScratchStorageBytes`, executor `scratchStorage` slab, launch param
   plumbing; still `Shared` everywhere.
3. Add the `Global` specialization + `launchFmcsKernel512<128,128>`; unit-test
   parity (Shared vs Global) at a small tier.
4. Host dispatch: `Auto` resolution, remove the tier-3 throw, add clear-error for
   illegal explicit `Shared`. Graph-level tests.
5. Public C++ `Parameters`/`MCSParameters` + `mcs_search` mapping; adapter
   comment.
6. Python bindings + `mcs.py` + validation + docstrings; Python tests incl.
   backward-compat.
7. Autotune search space.
8. Benchmark flag + CSV column + doc example.
9. Carveout readiness plumbing (dormant) + `Parameters::enableExtendedSharedCarveout`.
10. ptxas/NCU confirmation notes; update `fmcs.cuh` doc comment.

Steps 1–4 are the functional core; 5–8 are the "pipe it through"; 9 is readiness.

## 11. Open decisions for the implementer

- **Which tiers get a `Global` instantiation?** Minimum: `512 @ 128`. To allow
  forced-`Global` benchmarking at smaller tiers you must also instantiate
  `Global` for those (128- and 512-block, tiers 16/32/64). Recommendation:
  instantiate `Global` for all 512-block tiers (cheap, enables clean sweeps) and
  for 128-block tier-128 (to A/B the ~18 KB→~10 KB shared reduction); skip
  Global for the small 128-block tiers unless a benchmark needs them.
- **Enum vs bool for placement.** Enum (`Shared`/`Global`, host-only `Auto`) is
  clearer and extends to the multi-array plan in §6; prefer it over a bool.
- **Where does `Auto` live?** Resolve `Auto` on the host in `fmcs.cpp` so the
  device only ever sees `Shared`/`Global`; do not template the kernel on `Auto`.
- **Expose `enableExtendedSharedCarveout` in Python now, or keep it C++-only
  until the dynamic-shared follow-on lands?** Recommendation: keep it C++-only /
  undocumented for now to avoid advertising a no-op knob; add the Python surface
  with the follow-on that makes it functional.
