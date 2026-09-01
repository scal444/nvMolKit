.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Molecular Clustering
====================

Approximate Atom-Atom-Path clustering
-------------------------------------

``aap_similarity`` compares rooted atom-path environments and uses a
fixed-iteration Sinkhorn assignment on the GPU. The score is directed: swapping
the centroid-side and candidate-side molecules can change the result.

``aap_similarity_clustering`` applies directed sphere exclusion in input order.
The first unassigned molecule becomes a centroid and claims every remaining
molecule whose directed similarity is at least the threshold. Labels start at
one and are renumbered by descending cluster size, with centroid order breaking
ties.

.. code-block:: python

   from rdkit import Chem
   from nvmolkit.clustering import aap_similarity, aap_similarity_clustering

   molecules = [Chem.MolFromSmiles(smiles) for smiles in ("CC", "CCC", "c1ccccc1")]

   score = aap_similarity(molecules[0], molecules[1])
   labels = aap_similarity_clustering(molecules, threshold=0.217)

The default descriptor path length is 7, with 2048 hashed histogram bins. The
Sinkhorn approximation uses 8 iterations and temperature 0.104. These defaults
match the original GPU AAP implementation; changing any of them can change both
similarities and cluster assignments. The default clustering threshold of 0.217
is a compatibility default and should be validated for the intended chemical
series.

The current fused implementation accepts molecules with at most 64 RDKit atoms.
This is an implementation limit, including any explicit hydrogens, rather than
a limitation of AAP. Empty molecules and bonds other than single, double,
triple, or aromatic bonds are rejected.

The APIs accept an optional ``torch.cuda.Stream``. Because they return Python
scalars or lists, they synchronize that stream before returning.

Butina clustering
-----------------

Use ``butina`` when a reusable square distance matrix is already available.
Use ``fused_butina`` with packed fingerprints to avoid materializing that
matrix; it recomputes similarities during clustering and uses O(N) working
memory.
