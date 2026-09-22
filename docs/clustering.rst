.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Clustering and diversity selection
==================================

nvMolKit exposes clustering and diversity-selection algorithms through paired
APIs:

* the base function consumes a precomputed square distance matrix; and
* the ``fused_`` function computes distances from its inputs only when the
  algorithm needs them, avoiding an ``N x N`` matrix.

All cutoffs use **distance** units. For fused similarity providers nvMolKit
uses ``distance = 1 - similarity``. Consequently, a similarity threshold of
``0.7`` is written as ``cutoff=0.3``.

Choosing an algorithm
---------------------

.. list-table::
   :header-rows: 1
   :widths: 18 23 26 33

   * - Algorithm
     - Matrix API
     - Fused API
     - Use it when
   * - Butina
     - :func:`nvmolkit.clustering.butina`
     - :func:`nvmolkit.clustering.fused_butina`
     - You want neighborhood-density clustering with RDKit-compatible cluster
       ordering.
   * - Leader
     - :func:`nvmolkit.clustering.leader`
     - :func:`nvmolkit.clustering.fused_leader`
     - Input order defines priority and you want one representative from each
       exclusion sphere.
   * - MaxMin
     - :func:`nvmolkit.clustering.maxmin`
     - :func:`nvmolkit.clustering.fused_maxmin`
     - You want a fixed-size, diverse subset rather than clusters.
   * - DISE
     - :func:`nvmolkit.clustering.dise`
     - :func:`nvmolkit.clustering.fused_dise`
     - You want Leader-style centroids plus a cluster assignment for every
       input.

The fused APIs use provider configuration objects rather than callbacks. The
provider determines the accepted input and how pairwise distance is evaluated:

.. list-table::
   :header-rows: 1
   :widths: 25 30 15 15 15

   * - Provider
     - Input
     - Butina
     - Leader / DISE
     - MaxMin
   * - :class:`~nvmolkit.similarity.TanimotoSimilarity`
     - Packed ``int32`` or ``uint32`` fingerprints
     - Yes
     - Yes
     - Yes
   * - :class:`~nvmolkit.similarity.CosineSimilarity`
     - Packed ``int32`` or ``uint32`` fingerprints
     - Yes
     - Yes
     - Yes
   * - :class:`~nvmolkit.similarity.AAPSimilarity`
     - RDKit molecules
     - No
     - Yes
     - No

The string values ``"tanimoto"`` and ``"cosine"`` remain convenient aliases
for the corresponding packed-fingerprint providers. AAP is directed, so it is
only available to algorithms whose semantics define a direction from a
centroid to a candidate. It is intentionally unavailable to symmetric Butina
and MaxMin workflows.

Matrix and fused forms
----------------------

The matrix functions accept a square ``float64`` NumPy array, PyTorch tensor,
or :class:`~nvmolkit.types.AsyncGpuResult`. CPU inputs are copied to the active
CUDA device. The matrix functions require the full square matrix, not RDKit's
packed lower-triangle representation.

For Leader and DISE, element ``[i, j]`` is the distance **from selected leader
``i`` to candidate ``j``**. The matrix may therefore be directed. Butina and
MaxMin normally require a symmetric distance matrix.

The fused packed-fingerprint functions accept shape ``(N, num_words)`` with
``int32`` or ``uint32`` elements. Fingerprints returned by
:class:`~nvmolkit.fingerprints.MorganFingerprintGenerator` can be passed
directly without forming a similarity matrix.

.. code-block:: python

    from rdkit import Chem

    from nvmolkit.clustering import OutputMode, fused_maxmin
    from nvmolkit.fingerprints import MorganFingerprintGenerator
    from nvmolkit.similarity import TanimotoSimilarity

    smiles = ["CCO", "CCN", "CCCC", "c1ccccc1", "c1ccncc1", "CC(=O)O"]
    molecules = [Chem.MolFromSmiles(value) for value in smiles]
    fingerprints = MorganFingerprintGenerator(radius=2, fpSize=2048).GetFingerprints(molecules)

    indices, last_distance = fused_maxmin(
        fingerprints,
        pick_size=3,
        metric=TanimotoSimilarity(),
        seed=23,
        output=OutputMode.RDKIT,
    )
    diverse_molecules = [molecules[index] for index in indices]

Leader selection
----------------

Leader examines candidates in input order. Each selected leader excludes every
remaining candidate at distance ``<= cutoff``. ``first_picks`` forces initial
leaders in the supplied order, and ``pick_size=0`` means continue until no
candidate remains. Forced indices must be unique and in range. A leader is
always removed from further consideration, so uniqueness does not depend on
the distance-matrix diagonal or on a metric's empty-fingerprint convention.

.. code-block:: python

    import numpy as np

    from nvmolkit.clustering import OutputMode, leader

    distance_matrix = np.asarray(
        [
            [0.0, 0.1, 0.8],
            [0.9, 0.0, 0.1],
            [0.2, 0.9, 0.0],
        ],
        dtype=np.float64,
    )

    # Rows are leader -> candidate, so directed matrices are supported.
    selected = leader(distance_matrix, cutoff=0.2, output=OutputMode.RDKIT)
    assert selected == (0, 2)

MaxMin selection
----------------

MaxMin greedily selects the candidate whose distance to its nearest existing
pick is largest. Its seed handling and deterministic tie-breaking match
RDKit's ``MaxMinPicker`` for the same distance values.

When ``first_picks`` is empty, ``seed`` controls the random first pick. Supplying
``first_picks`` makes those indices the initial pick sequence. If ``threshold``
is set, selection stops before adding a candidate whose nearest-pick distance
is at most that threshold. The returned ``last_distance`` is the separation of
the last candidate that was actually added, or ``-1`` when no additional
candidate was selected. Forced indices must be unique and in range. Matrix
thresholds must be finite and non-negative; packed-fingerprint thresholds must
also be at most ``1``.

DISE clustering
---------------

Directed sphere exclusion (DISE) uses the same ordered centroid selection as
Leader, then assigns every input to a selected centroid. ``assignment="first"``
keeps the first qualifying centroid. ``assignment="nearest"`` compares all
centroids and uses the nearest one; this is the default.

Clusters are ordered by descending size, with centroid-selection order breaking
ties. RDKit-format clusters place the centroid first in each cluster tuple.

AAP example
~~~~~~~~~~~

:class:`~nvmolkit.similarity.AAPSimilarity` operates on RDKit molecules and is
directed: the selected centroid is the left-hand molecule and the candidate is
the right-hand molecule. Its historical similarity threshold ``0.217`` becomes
the distance cutoff ``1.0 - 0.217`` in the common clustering API.

.. code-block:: python

    from rdkit import Chem

    from nvmolkit.clustering import OutputMode, fused_dise, fused_leader
    from nvmolkit.similarity import AAPSimilarity

    smiles = ["CCCC", "CCCO", "CCOC", "c1ccccc1"]
    molecules = [Chem.MolFromSmiles(value) for value in smiles]
    aap = AAPSimilarity(
        max_path_length=7,
        histogram_bins=2048,
        sinkhorn_iterations=8,
        sinkhorn_temperature=0.104,
    )
    cutoff = 1.0 - 0.217

    leaders = fused_leader(
        molecules,
        cutoff,
        metric=aap,
        first_picks=(2,),
        output=OutputMode.RDKIT,
    )
    clusters = fused_dise(
        molecules,
        cutoff,
        metric=aap,
        assignment="nearest",
        output=OutputMode.RDKIT,
    )

AAP currently supports nonempty RDKit molecules with at most 64 atoms,
including explicit hydrogens. Rooted-path descriptor construction occurs on
the CPU; pair scoring occurs on the GPU.

Output modes
------------

Every API accepts ``output=OutputMode.DEVICE`` (the default) or
``output=OutputMode.RDKIT``.

Clustering functions return a :class:`~nvmolkit.clustering.ClusterDeviceResult`
in device mode. Its fields are:

* ``cluster_ids``: one contiguous zero-based ``int32`` cluster ID per input;
* ``centroids``: the input index for each cluster ID; and
* ``cluster_sizes``: the ``int64`` member count for each cluster ID.

Leader and MaxMin return a
:class:`~nvmolkit.clustering.SelectionDeviceResult`. ``indices`` contains the
ordered ``int32`` selections. ``last_distance`` is populated for MaxMin and is
``None`` for Leader.

In RDKit mode, clustering functions return centroid-first tuples of input
indices and Leader returns a tuple of indices. MaxMin returns
``(indices, last_distance)``.

Execution and memory behavior
-----------------------------

The matrix APIs require ``O(N^2)`` input storage. Fused Leader, MaxMin, and
DISE store ``O(N)`` algorithm state and compute requested similarities on
demand. Fused Butina likewise avoids materializing the complete distance
matrix, though its internal neighborhood data has algorithm-specific memory
requirements.

All functions accept an optional :class:`torch.cuda.Stream`. The selection and
DISE implementations currently make host-side control decisions between GPU
passes, so the call synchronizes its stream during the control loop. Device
output avoids a final host representation and is directly consumable by
PyTorch, but these algorithms should not yet be treated as fully asynchronous.

Compatibility names
-------------------

``ButinaOutputMode`` and ``DISEOutputMode`` are aliases of ``OutputMode``.
``ButinaDeviceResult`` and ``DISEDeviceResult`` are aliases of
``ClusterDeviceResult``. New code should prefer the common names.
