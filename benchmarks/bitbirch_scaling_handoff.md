# BitBIRCH hierarchical scaling handoff

## Status

Branch: `bitbirch-9-21`

The last known-good pushed baseline is commit `254ba86b`. The handoff commit on
top of it contains an unfinished replacement of the density- and size-dependent
partition merge paths with one bounded hierarchical algorithm. It compiles, has
zero CUDA stack and spills, and passes most correctness tests, but it is **not
correct and must not be merged**. Four Python correctness cases still fail.

The objective is generic linear-memory scaling. Input count and observed
cluster density must not select different algorithms. The intended algorithm
is:

1. Build fixed-size leaf trees in parallel.
2. Merge adjacent groups of at most four trees in parallel.
3. Repeat bounded merge rounds while more than one tree remains and the exact
   number of summaries decreases.
4. If a round makes no progress, finalize the remaining forest in parallel.
5. Never allocate an `N * fingerprint_bits` summary slab.

The fan-in and construction batch are execution tiling parameters, not
dataset-size or similarity thresholds. There are no 10k/100k/1M branches and
no percentage-density early exits in the new path.

## Changes in the handoff commit

- `src/bitbirch.cu`
  - Adds `PartitionedForest`, which owns one merge level's tree metadata and
    paged summary arena.
  - Adds kernels to collect live forest entries, merge indexed groups, remap
    labels, index output clusters, and finalize a forest.
  - Replaces the old early-sparse, single-intermediate, sparse-forest, and
    final single-block merge paths with repeated bounded merge rounds.
  - Releases each source level after its successor has been built, keeping
    metadata linear in the input size.
  - Uses 32-input construction batches to avoid the first-launch over-reserve
    that previously caused the 1M out-of-memory failure.
  - Removes the merge-component dispatch that selected fan-in from partition
    count. Integer component widths now only represent required count ranges.
- `nvmolkit/tests/_bitbirch_reference.py`
  - Changes the partitioned reference to the same fixed-fan-in repeated merge
    structure and exact no-progress stopping rule.
- `benchmarks/bitbirch_perf_log.md`
  - Records the corrected memory diagnosis and corrects the interpretation of
    the rejected fan-in experiment.
- `.codex/skills/nvmolkit-nsight-profiling/SKILL.md`
  - Captures the corrected profiler handoff rules: only use capture modes the
    benchmark supports, persist external-terminal logs/status, diagnose the
    application before the report, and use nonblocking five-minute checks for
    unattended user-run profiles.

## Build and CUDA resource evidence

Host GPU: NVIDIA RTX 1000 Ada Generation Laptop GPU, compute capability 8.9.
The build directory is `/tmp/nvmolkit-bitbirch-perf-build`, configured for
`CMAKE_CUDA_ARCHITECTURES=89` with ptxas verbose reporting.

The following targets build successfully:

- `bitbirch`
- `_clustering`
- `test_bitbirch_primitives`
- `integration_test_bitbirch`

All emitted kernels have zero-byte stack frames and zero spill loads/stores.
Relevant sm89 figures:

- hierarchical merge kernel: 80 registers for most specializations, 96 for
  the `uint8 -> uint8` specialization, 9,280 bytes shared memory;
- partial-tree kernel: 80 registers normally and 92 for `uint8`, 9,280 bytes
  shared memory;
- collect/index/finalize/remap kernels: 15-24 registers.

The high register counts are unresolved performance concerns, but no stack or
spill hard-gate failure is present.

## Correctness results

The latest complete log is `/tmp/bitbirch_hierarchical_correctness.log`; its
status file is `/tmp/bitbirch_hierarchical_correctness.status` and contains
`1`.

Passing:

- C++ primitive/partitioned suite: 10/10.
- C++ molecule integration suite: 2/2.
- Selected Python BitBIRCH suite: 54/58.

Failing Python tests:

1. `test_bitbirch_partitioned_component_width_dispatch_uint8_to_uint16`
2. `test_bitbirch_partitioned_matches_bit_feature_merge_reference[7-9]`
3. `test_morgan_fingerprint_to_partitioned_bitbirch_matches_reference`
4. `test_fingerprint_generation_and_clustering_chain_on_explicit_stream`

The failures are substantive label/cluster mismatches. In the explicit-stream
case, two copies of 32 molecules should merge across partitions, but the GPU
result returns 64 singleton labels. Do not update expected results or weaken
these assertions.

A smaller exact-duplicate reproducer does work:

```python
x = np.array([[1], [2], [4], [8], [1], [2], [4], [8]], dtype=np.uint32)
bitbirch(x, threshold=1.0, branching_factor=254, num_partitions=4)
# [0, 1, 2, 3, 0, 1, 2, 3]
```

Therefore cross-partition merging is not universally broken. The remaining
problem depends on the tree/summary shape, branching/split behavior, input
width, or ordering.

## Debugging already performed

- Verified the tests import
  `/tmp/nvmolkit-bitbirch-python/nvmolkit/_clustering.so`.
- Rebuilt and recopied the extension after removing the obsolete
  all-singleton shortcut, then reran the suite. The same four failures remain;
  this is not a stale-extension artifact.
- Confirmed source entries are collected from `entryClusterIds`, labels are
  remapped through `sourceToOutput`, and each output level is indexed before
  the next round.
- Confirmed all kernels and host/device copies use the supplied CUDA stream.

High-value next checks:

1. Add a focused regression containing the 32 duplicated fingerprints across
   four partitions without the Morgan generator, then reduce the word count
   and branching factor independently.
2. Compare the exact source-summary insertion order, counts, centroids, and
   linear sums between the Python reference and the first GPU merge round.
3. Instrument the first mismatching candidate in
   `bitBirchMergeIndexedTreeGroupsKernel`: selected output entry, candidate
   count, combined iSIM terms, and merge decision.
4. Check whether a split-created leaf/internal summary is being collected or
   materialized differently from the reference. The tiny no-split duplicate
   reproducer passing makes split-state handling a leading suspect.
5. Once correctness is restored, add an explicit multi-round adversarial test
   that fails if compatible summaries remain separated after more than one
   fan-in level.

## Performance and memory evidence before this rewrite

The prior batch-32, fixed-four-tree implementation completed 1M real Enamine
fingerprints on the laptop GPU with these single-run times:

| Threshold | Time (ms) | Clusters |
| --- | ---: | ---: |
| 0.10 | 679.786 | 1 |
| 0.15 | 9,681.396 | 7,772 |
| 0.25 | 5,495.668 | 577,093 |
| 0.99 | 5,174.305 | 1,000,000 |

The earlier profiler run at threshold 0.25 peaked at a 2.85 GB CUDA pool and
spent 4.871 seconds of 5.54 seconds in the intermediate merge. The committed
256-input reservation scheme had previously OOMed at roughly a 5.91 GB pool.

Those measurements are context only. The hierarchical implementation in this
handoff has not passed correctness and has not been benchmarked. Do not quote
it as a performance result.

## Reproduction setup

The Python overlay used for local testing is
`/tmp/nvmolkit-bitbirch-python`. After rebuilding, copy:

```bash
cp /tmp/nvmolkit-bitbirch-perf-build/nvmolkit/_clustering.so \
  /tmp/nvmolkit-bitbirch-python/nvmolkit/_clustering.so
```

Run Python from `/tmp`, not from the repository checkout, with:

```bash
PYTHONPATH=/tmp/nvmolkit-bitbirch-python \
/home/kboyd/miniforge3/envs/rdcu_dev/bin/python -m pytest -q \
  /home/kboyd/omg/repos/nvmolkit/nvmolkit/tests/test_clustering.py \
  /home/kboyd/omg/repos/nvmolkit/nvmolkit/tests/test_bitbirch_integration.py \
  -k bitbirch
```

GPU runtime commands must run outside the Codex filesystem sandbox. Nsight
commands requiring privileged performance counters should be handed to the
user with `sudo`, absolute executable paths, persisted stdout/stderr, and an
exit-status file.
