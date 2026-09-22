# BitBIRCH design record

This records the publication-derived contract used by nvMolKit's native
implementation. No source from the GPL-3.0 `bitbirch` or `bblean` packages is
used in this implementation.

## Sources

- K. L. Pérez et al., *BitBIRCH: efficient clustering of large molecular
  libraries*, Digital Discovery 4 (2025), 1042--1051,
  <https://doi.org/10.1039/D5DD00030K>. The Bit Feature representation,
  centroid, iSIM Jaccard--Tanimoto equations, tree insertion, node splitting,
  and initial parallel formulation are derived from this paper and its Section
  S1 pseudocode.
- K. L. Pérez et al., *BitBIRCH Clustering Refinement Strategies*, J. Chem.
  Inf. Model. 65 (2025), 5280--5288,
  <https://doi.org/10.1021/acs.jcim.5c00627>. Diameter and tolerance merge
  behavior follows Equations 1, 3, 4, and 5.
- I. Pickering et al., *BitBIRCH-Lean: chemical space in the palm of your
  workstation*, bioRxiv (2025),
  <https://doi.org/10.1101/2025.10.22.684015>. Dynamic component widths and the
  multi-round algorithm inform storage and scheduling; they do not fix the CUDA
  layout.

## Mathematical contract

For a Bit Feature with count `N` and component-wise bit sums `LS`, define:

```text
A = sum_q LS[q] * (LS[q] - 1) / 2
B = sum_q LS[q] * (N - LS[q])
iSIM_JT = A / (A + B)
centroid[q] = floor(LS[q] / N + 1/2)
```

The centroid sets an exact half-count tie to one. Diameter merge accepts a
combined Bit Feature when `iSIM_JT >= threshold`; equality is accepted.
Tolerance insertion additionally applies refinement Equation 5:

```text
((N + 1) * combined_iSIM - (N - 1) * old_iSIM) / 2
    >= old_iSIM - tolerance
```

An old singleton bypasses this secondary tolerance check because it has no old
pairwise diversity to preserve. It must still pass the combined diameter
threshold.

Empty and singleton summaries have no distinct pairs and are assigned iSIM 1.
When `A + B == 0`, an all-zero summary is assigned iSIM 1, matching nvMolKit's
pairwise convention. Accumulation converts each component to double before
multiplication, avoiding fixed-width integer products and sums. Threshold
comparisons use `A >= threshold * (A + B)` to avoid a final division. This
evaluation order is deterministic on a supported GPU architecture; a future
exact-rational path may be added if workloads exceed the range where double
represents summary counts adequately.

## Native serial and partitioned implementation

`src/bitbirch_common.cuh` contains host/device mathematical primitives with no
scalable local state. The independent Python reference implements the complete
ordered tree, including cascading and root splits. Randomized tests compare it
directly with the native serial GPU path.

Production tree state uses index-based structure-of-arrays storage in global
memory owned by `AsyncDeviceVector`. Component sums, centroids, membership, and
node bodies scale with the input or options and therefore remain in global
memory.

Node splitting chooses the lowest-index pair among equally separated seed
pairs. Entries prefer the closer seed, with similarity ties assigned to the
smaller group and then left. Assignment is forced to the other group once one
side reaches `ceil((branching_factor + 1) / 2)`. Consequently a split with
`branching_factor >= 3` leaves at least two entries on each side, preventing
pathological unary internal chains while remaining deterministic.

The native implementation requires `branching_factor >= 3`. A factor of two
necessarily splits three entries into 2+1 and permits unary internal-node
chains. A factor of at least three gives two nontrivial split groups and
supports conservative pools of `2N + 8` nodes and `3N + 8` allocated entries.
The independent test reference retains factor two solely to exercise cascading
split semantics.

The public implementation retains a one-thread ordered tree as the exact
reference and small-workload path. For larger inputs, or when selected with
``num_partitions``, contiguous ordered partitions build independent trees in
separate CUDA blocks. One or two merge kernels scan each partial leaf Bit
Feature once, in first-member order, and insert its count and linear sum into a
larger tree. Labels retain the stable order of the earliest original member.
Groups of two partial trees are merged concurrently for at most 128 partitions;
groups of four are used above that point to limit the number of summaries sent
to the cooperative final merge. The Python default selects one partition below
512 inputs and enough partitions above that to keep each partial tree at no
more than 255 inputs. This keeps partial sums in 8-bit components and prevents
per-tree work from growing superlinearly with the full input.
Tolerance-diameter mode automatically selects one partition; callers can also
explicitly select one for serial semantics. Partial and final summary arenas
independently dispatch to 8-, 16-, or 32-bit components according to their
maximum represented counts. A cooperative final merge is limited to 65,535
source summaries; larger merge forests are finalized in parallel.

Partial-tree construction and merge-tree insertion use 256-thread cooperative
blocks. Entry searches, Bit Feature updates, ancestor summaries, diameter
evaluation, and split-seed selection distribute their word, component, or
entry work across the block. Tolerance-diameter mode
currently requires one partition because
refinement Equation 5 is defined for singleton insertion; applying it to a
weighted incoming Bit Feature requires an independently specified criterion.
Refinement and out-of-core execution also remain future work.

## Public API and supported range

Call ``nvmolkit.clustering.bitbirch`` with a packed ``int32`` or ``uint32``
array of shape ``(N, W)``. Device-resident fingerprints can be passed directly
from nvMolKit's fingerprint generators:

```python
from nvmolkit.clustering import bitbirch
from nvmolkit.fingerprints import MorganFingerprintGenerator

fingerprints = MorganFingerprintGenerator(radius=3, fpSize=1024).GetFingerprints(molecules)
labels, centroids = bitbirch(
    fingerprints,
    threshold=0.55,
    branching_factor=64,
    num_partitions=None,
    return_centroids=True,
)
```

Each row represents exactly ``32W`` logical bits; non-word-multiple logical
lengths are not represented separately. ``threshold`` must be finite and in
``[0, 1]``, ``tolerance`` must be finite and nonnegative, and
``branching_factor`` must be at least three. ``num_partitions`` is either
automatic or an integer in ``[1, N]`` for nonempty input. Empty input is
accepted. Native indexing limits ``N`` to ``(INT_MAX - 8) / 3`` and ``W`` to
``INT_MAX / 32``; practical device-memory capacity is much lower.

Diameter mode supports serial and partitioned execution. Tolerance-diameter
mode currently requires ``num_partitions=1``. Labels are deterministic for a
fixed input, options, GPU architecture, and execution mode, but changing the
input order or partition count can change the clustering. The call performs
host synchronization to inspect tree status and obtain the compact cluster
count between stages. Fingerprints, tree state, final labels, and centroids
otherwise remain on the GPU; no pairwise matrix or host fingerprint round trip
is introduced.

## Prototype resource and timing record

The current build was compiled explicitly for ``sm_89`` on an NVIDIA RTX 1000
Ada Generation Laptop GPU. Ptxas reported 66 registers for the serial kernels,
69 for final merge, and 77--92 for partial and intermediate merge variants.
Every BitBIRCH kernel reported a zero-byte stack frame and zero spills. The
cooperative kernels use one barrier and approximately 9.3 KiB of static shared
memory.

On the committed 1,000-molecule Morgan/ECFP comparison workload (1,024 bits,
radius 2, threshold 0.55, branching factor 254), the pre-redesign GPU path took
8,947.58 ms. The redesigned path took 258.35 +/- 0.04 ms after one warm-up over
three runs, a 34.6x speedup. The recorded BitBIRCH-Lean result is 36.30 ms, so
this GPU path remains about 7.1x slower on this nearly all-singleton workload.

An end-to-end probe used 4,096 three-word fingerprints drawn from
32 repeated random bases at threshold 0.95. After warm-up, 16 partitions took
0.0666 seconds versus 0.2836 seconds for one serial tree (4.26x), and both
reported 32 clusters. This demonstrates a useful prototype region only; it is
not a universal crossover claim. A dense adversarial probe with 2,048 random
three-word fingerprints, threshold 0.6, and branching factor 64 showed the
opposite regime: eight partitions took a 0.5920-second median versus 0.4618
seconds serial because 2,047 summaries still reached the serial final merge.
Automatic partitioning therefore exposes parallel construction but does not
guarantee a speedup when partial trees provide almost no compression.

A 1,000-molecule Morgan/ECFP quality run (1,024 bits, radius 3, threshold 0.55,
branching factor 64) produced adjusted mutual information of 0.7884 for four
partitions and 0.8234 for 16 partitions relative to the serial tree. Cluster
counts were 179 and 186 versus 224 serial; minimum within-cluster iSIM values
were 0.5543 and 0.5654, and largest-cluster fractions were 0.034 and 0.031
versus 0.032. These results meet the provisional local gates of AMI >= 0.75,
minimum within-cluster iSIM >= threshold, cluster-count drift <= 25%, and
largest-cluster-fraction drift <= 0.01. Additional GPU architectures and
larger non-replicated chemical collections remain release-validation work.
``benchmarks/bitbirch_clustering_bench.py`` makes the synthetic comparison
reproducible across input sizes, word counts, partition counts, host/device
inputs, and duplicate-heavy or random workloads.
`benchmarks/bitbirch_quality_bench.py` reports AMI, cluster-size and iSIM
quantiles, singleton fractions, and largest-cluster fraction on SMILES-derived
Morgan fingerprints.

## Memory scaling

For `N` inputs and `W` packed words, topology reserves conservative linear
pools of `2N + O(P)` nodes and `3N + O(P)` entries. A singleton Bit Feature
stores only its original fingerprint index; it neither copies a linear sum nor
allocates a centroid. On the first merge, the entry receives a slot from a
paged summary arena. Serial, partial, and intermediate construction grow these
arenas between bounded kernel launches from the observed summary cursor rather
than reserving one `32W`-component vector for every possible entry.

Consequently summary storage is `O(E * 32W)` for the `E` materialized Bit
Features actually created, plus at most one bounded growth batch and page
rounding, while topology and mappings remain `O(N)`. Partial, intermediate,
and final component widths are selected independently. Cached packed
centroids exist only for materialized summaries. Labels, mapping arrays, and
optional output centroids are linear in `N`.
No pairwise matrix, recursive device state, device allocation, or storage
proportional to `N^2` is used.

## Release boundaries

The local implementation and tests are independently derived from the cited
publications, but normal NVIDIA legal/IP and open-source provenance review is
still required before release. This validation record covers one workstation
GPU (``sm_89``); the supported architecture matrix still needs CI or hardware
coverage on at least one data-center GPU. Refinement, out-of-core operation,
multi-GPU scheduling, weighted summaries, non-word-aligned logical bit counts,
and weighted merge criteria are not part of this initial API. The included
benchmarks expose the current useful and adverse regimes rather than asserting
a universal speedup.
