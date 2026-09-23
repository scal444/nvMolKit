.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Clustering and diversity selection
==================================

:mod:`nvmolkit.clustering` provides Butina, Leader, MaxMin, and directed
sphere exclusion (DISE). Each algorithm has two forms:

.. list-table::
   :header-rows: 1

   * - Algorithm
     - Matrix form
     - Fused form
   * - Butina
     - :func:`~nvmolkit.clustering.butina`
     - :func:`~nvmolkit.clustering.fused_butina`
   * - Leader
     - :func:`~nvmolkit.clustering.leader`
     - :func:`~nvmolkit.clustering.fused_leader`
   * - MaxMin
     - :func:`~nvmolkit.clustering.maxmin`
     - :func:`~nvmolkit.clustering.fused_maxmin`
   * - DISE
     - :func:`~nvmolkit.clustering.dise`
     - :func:`~nvmolkit.clustering.fused_dise`

Matrix and fused forms
----------------------

The matrix form takes a precomputed square ``float64`` distance matrix, where
element ``[i, j]`` is the distance from item ``i`` to item ``j``. Memory scales
as ``O(N^2)``.

The fused form takes fingerprints or molecules and a similarity metric, and
computes distances as the algorithm needs them. Memory scales as ``O(N)``.
Given the same distances, both forms return the same result.

All cutoffs and thresholds are distances. Fused forms use
``distance = 1 - similarity``, so a similarity threshold of ``0.7`` is
``cutoff=0.3``.

Metrics
-------

Fused forms select a similarity metric with ``metric=``, given as a name or as a
metric object:

.. list-table::
   :header-rows: 1

   * - Metric
     - Name
     - Input
   * - :class:`~nvmolkit.similarity.TanimotoMetric`
     - ``"tanimoto"``
     - Packed ``int32`` or ``uint32`` fingerprints, shape ``(N, num_words)``
   * - :class:`~nvmolkit.similarity.CosineMetric`
     - ``"cosine"``
     - Packed ``int32`` or ``uint32`` fingerprints, shape ``(N, num_words)``
   * - :class:`~nvmolkit.similarity.AAPMetric`
     - ``"aap"``
     - Sequence of RDKit molecules

Every fused form accepts every metric, except that
:func:`~nvmolkit.clustering.fused_butina` does not yet support
:class:`~nvmolkit.similarity.AAPMetric`. Metric objects carry parameters; for
example, ``AAPMetric(max_path_length=8)``.

.. code-block:: python

    from rdkit import Chem

    from nvmolkit.clustering import OutputMode, fused_leader, fused_maxmin
    from nvmolkit.fingerprints import MorganFingerprintGenerator
    from nvmolkit.similarity import AAPMetric

    smiles = ["CCO", "CCN", "CCCC", "c1ccccc1", "c1ccncc1", "CC(=O)O"]
    molecules = [Chem.MolFromSmiles(value) for value in smiles]
    fingerprints = MorganFingerprintGenerator(radius=2, fpSize=2048).GetFingerprints(molecules)

    picks, last_distance = fused_maxmin(fingerprints, 3, metric="tanimoto", seed=23, output=OutputMode.RDKIT)
    leaders = fused_leader(molecules, 0.8, metric=AAPMetric(max_path_length=8), output=OutputMode.RDKIT)

Output modes
------------

``output=OutputMode.DEVICE`` (the default) returns
:class:`~nvmolkit.clustering.ClusterDeviceResult` or
:class:`~nvmolkit.clustering.SelectionDeviceResult` with GPU buffers.
``output=OutputMode.RDKIT`` returns host tuples of input indices in RDKit's
format.
