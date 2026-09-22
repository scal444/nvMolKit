# BitBIRCH GPU optimization log

Target system: NVIDIA RTX 1000 Ada Generation Laptop GPU (sm89). Unless noted,
the workload is a seed-42 sample of Enamine, Morgan radius 2, 1024 bits,
branching factor 254, and automatic partitioning. Timings exclude fingerprint
generation.

The standing performance matrix is:

- `threshold=0.10`: minimum-cluster edge case (one cluster).
- `threshold=0.15`: real dense case.
- `threshold=0.25`: real sparse case.
- `threshold=0.99`: maximum-cluster edge case (all singletons).

Experiments are accepted only when they preserve correctness tests and do not
materially regress either real case (`0.15` or `0.25`).

## Accepted experiments

### Unique-summary final merge

The final merge previously performed block-wide synchronization once per input
fingerprint. It now collects distinct source summaries in first-occurrence
order, merges each once, and remaps labels cooperatively. This preserves the
summary insertion order while removing redundant synchronization.

At 10k, `threshold=0.15` improved from 824 ms to 74.6 ms. The
`threshold=0.25` case changed from taking longer than 90 seconds for the
repeated benchmark to 1.99 seconds for one run.

### Merge fan-in 8 -> 2

Smaller groups expose more independent merge blocks. At 10k, one warmed run
changed as follows:

| Threshold | Fan-in 8 | Fan-in 2 |
|---:|---:|---:|
| 0.10 | 33.4 ms | 32.1 ms |
| 0.15 | 74.6 ms | 59.6 ms |
| 0.25 | 1,992 ms | 1,813 ms |
| 0.99 | 578 ms | 129 ms |

### 50% post-merge forest cutoff

When at least half of the inputs remain as summaries after the parallel merge
round, the implementation returns the deterministic forest rather than running
the serial global merge. At 10k and `threshold=0.25`, this changed runtime from
1.81 seconds to 106.5 ms. The expected approximation tradeoff is visible in
cluster count: 6,236 GPU clusters versus 5,161 for serial bblean.

### 90% pre-merge forest cutoff

When at least 90% of inputs survive partial-tree construction, the
implementation finalizes the partial forest directly and skips the pair-merge
round. This targets the maximum-cluster edge without changing either real-case
result.

### Cached source centroids during summary merging

Merge kernels previously reconstructed a source summary's centroid bit-by-bit
for every tree comparison even though the centroid was already cached. Passing
the cached source centroid through the merge kernels improved the 100k real
dense case from 332.0 ms to 325.8 ms and the real sparse case from 1,364.8 ms
to 1,330.4 ms, with unchanged cluster assignments.

### Warp-reduced iSIM accumulation

The cooperative iSIM path previously had thread 0 serially sum 256 terms for
each 256-bit tile. Warp shuffles now reduce within each warp, leaving only
eight warp totals for thread 0. At 100k this improved the real dense case from
325.8 ms to 131.9 ms (2.47x) and the real sparse case from 1,330.4 ms to
1,079.9 ms (1.23x), with identical cluster counts. The 10k matrix also
improved across all four regimes: 9.55/19.18/44.61/7.47 ms for thresholds
0.10/0.15/0.25/0.99.

### Deterministic warp max-reduction

Closest-entry selection previously had thread 0 scan up to 254 similarity
results. A warp reduction now selects the maximum while preserving the first
entry on ties. An initial version was rejected by integration testing because
warp leaders overwrote shared input arrays still being read by other warps; the
corrected version uses separate warp-result arrays. It reduces the cooperative
kernels from 96 to 80 registers with zero stack/spills. At 100k the real sparse
case improved from 1,079.9 ms to 1,057.1 ms. A seven-run dense measurement was
132.57 +/- 0.30 ms versus 131.90 ms before the change, treated as neutral.

## Current accepted 100k baseline

Three timed runs after one warmup:

| Threshold | nvMolKit | bblean | Throughput ratio | GPU clusters | bblean clusters |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 257.27 ms | 1,280.87 ms | 4.98x | 1 | 1 |
| 0.15 | 132.57 ms | 1,355.43 ms | 10.22x | 229 | 54 |
| 0.25 | 1,057.06 ms | 3,137.31 ms | 2.97x | 50,460 | 36,085 |
| 0.99 | 863.39 ms | 4,290.89 ms | 4.97x | 100,000 | 100,000 |

## Resource gate

The sm89 build reports zero stack frames and zero spills for all affected
kernels. The cooperative partial and group-merge kernels use 80 registers and
9,272 bytes shared memory. The final merge uses 56 registers and 9,276 bytes
shared memory. Forest indexing/finalization use 20/26 registers and no shared
memory.

## Rejected experiments

### 128-thread cooperative blocks

Although 128 threads permits more resident blocks with the 96-register merge
kernels, a full 254-entry node then requires two cooperative chunks. At 10k it
regressed the real sparse case from 76.9 ms to 87.2 ms (+13.4%) and changed the
real dense case from 53.6 ms to 54.6 ms. Rejected without spending time on a
100k run; the accepted block size remains 256.

### One fixed partition count for every density

Auto uses 80 partitions on the 20-SM target GPU. At 100k, 160 partitions
improved the real sparse case from 1,365 ms to 1,059 ms but regressed the real
dense case from 332 ms to 493 ms. Forty partitions improved the dense case to
297 ms, but caused the sparse case to fall below the forest cutoff and enter a
global merge that exceeded 60 seconds. Neither fixed count is acceptable.
These results motivate density-adaptive partitioning or another merge level;
they do not justify changing the forest cutoff to conceal a bad partition
choice.

### Threshold-selected partition count

Selecting 40 partitions below threshold 0.20 and 160 above it improved the
primary 100k sample to 127.60 ms in the real dense case and 845.38 ms in the
real sparse case. It was rejected because threshold does not determine dataset
compression. On a second Enamine population, the 10k threshold-0.15 workload
produced 624 GPU clusters instead of 167 and took 133.07 ms instead of 19.07
ms. Future adaptive selection must use compression observed from the actual
input rather than threshold as a proxy.

### Unique-summary collection in the pair-merge round

Applying the final merge's unique-summary prepass to each pair-merge block was
neutral at 100k: 325.9/1,331.2 ms versus the cached-centroid baseline of
325.8/1,330.4 ms for the real dense/sparse cases. Rejected because it adds an
`N`-entry workspace and control-flow complexity without measurable speedup.

## Validation to date

- `test_bitbirch_primitives`: 10/10 passed.
- `integration_test_bitbirch`: 2/2 passed.
- Python BitBIRCH selection: 55/55 passed.
- Full four-regime 10k and 100k performance matrices completed.

## Cold scaling run

One timed run with no warmup used a single seed-42 sample from the 10M Enamine
source, with the 10k and 100k cases taken as prefixes of the 1M sample.

| Inputs | Threshold | nvMolKit | bblean | bblean / nvMolKit |
|---:|---:|---:|---:|---:|
| 10k | 0.10 | 10.630 ms | 167.950 ms | 15.80x |
| 10k | 0.15 | 18.853 ms | 123.341 ms | 6.54x |
| 10k | 0.25 | 45.745 ms | 1,091.532 ms | 23.86x |
| 10k | 0.99 | 9.138 ms | 335.153 ms | 36.68x |
| 100k | 0.10 | 144.546 ms | 1,241.211 ms | 8.59x |
| 100k | 0.15 | 148.901 ms | 1,420.847 ms | 9.54x |
| 100k | 0.25 | 1,026.752 ms | 2,902.697 ms | 2.83x |
| 100k | 0.99 | 718.020 ms | 5,266.815 ms | 7.33x |
| 1M | 0.10 | OOM | 12,900.359 ms | N/A |
| 1M | 0.15 | OOM | 12,553.027 ms | N/A |
| 1M | 0.25 | OOM | 38,054.605 ms | N/A |
| 1M | 0.99 | OOM | 53,063.440 ms | N/A |

The GPU has 6,141 MiB. At 1M inputs and 80 automatic partitions, the partial
linear-sum slab alone occupies approximately 5.72 GiB because each partition
contains 12,500 inputs and therefore uses 16-bit components. The additional
tree, centroid, fingerprint, label, and result allocations exceed device
capacity before the first threshold is timed.

## Output-sensitive paged storage

The dense slab baseline above is no longer considered a valid performance
baseline for scalable execution: it reserves a linear sum and packed centroid
for every possible entry, whether that entry ever represents a merged Bit
Feature or remains a singleton.

The accepted replacement stores a singleton as an index into the original
packed fingerprints. It materializes a counter vector and cached centroid only
on the first merge. Summary storage grows in 4,096-entry pages between bounded
serial, partial-tree, and intermediate-merge launches. Partial and
intermediate stages select counter widths independently. Automatic partitioning
keeps partial trees at no more than 255 inputs, making their counters 8-bit and
keeping per-tree insertion work bounded as `N` grows. The final cooperative
merge accepts at most 65,535 source summaries; larger forests are finalized in
parallel instead of entering a non-scalable single-block merge.

The first paged version completed the 1M dense cases but still OOMed at
threshold 0.25 because the intermediate arena pessimistically reserved twice
the observed partial-cluster count using final-stage 32-bit counters. Separate
stage widths halved that request but were insufficient. Incremental merge-page
growth removed the OOM. A subsequent 80-partition run exceeded four minutes at
1M/0.25 and was rejected; width-bounded automatic partitioning plus the bounded
final merge reduced it to 31.18 seconds.

One timed run with no warmup, using the same seed-42 1M sample and prefixes:

| Inputs | Threshold | Paged nvMolKit | Prior bblean | bblean / nvMolKit | GPU clusters |
|---:|---:|---:|---:|---:|---:|
| 10k | 0.10 | 34.105 ms | 167.950 ms | 4.92x | 1 |
| 10k | 0.15 | 26.420 ms | 123.341 ms | 4.67x | 174 |
| 10k | 0.25 | 106.274 ms | 1,091.532 ms | 10.27x | 6,171 |
| 10k | 0.99 | 8.655 ms | 335.153 ms | 38.72x | 10,000 |
| 100k | 0.10 | 72.746 ms | 1,241.211 ms | 17.06x | 1 |
| 100k | 0.15 | 98.617 ms | 1,420.847 ms | 14.41x | 238 |
| 100k | 0.25 | 1,960.290 ms | 2,902.697 ms | 1.48x | 50,638 |
| 100k | 0.99 | 1,025.796 ms | 5,266.815 ms | 5.13x | 100,000 |
| 1M | 0.10 | 619.220 ms | 12,900.359 ms | 20.83x | 1 |
| 1M | 0.15 | 745.982 ms | 12,553.027 ms | 16.83x | 138 |
| 1M | 0.25 | 31,175.589 ms | 38,054.605 ms | 1.22x | 367,129 |
| 1M | 0.99 | 15,359.401 ms | 53,063.440 ms | 3.45x | 1,000,000 |

All four 1M regimes now fit on the 6 GiB GPU. The dense cases scale close to
linearly from 100k to 1M. The sparse and singleton-heavy cases scale by 15.9x
and 15.0x respectively for 10x more inputs; further work should focus on page
allocation overhead and the parallel forest finalization path without restoring
speculative `N * bits` summary slabs.
