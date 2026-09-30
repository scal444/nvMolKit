# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-resident molecule library for repeated substructure queries."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor

from rdkit.Chem import Mol

from nvmolkit._substructLibrary import SubstructLibrary as _NativeSubstructLibrary
from nvmolkit.substructure import SubstructSearchConfig

__all__ = ["SubstructLibrary"]


class SubstructLibrary:
    """A collection of molecules kept on the GPU for repeated substructure searches.

    This is the GPU counterpart of RDKit's ``rdSubstructLibrary.SubstructLibrary``:
    target molecules are prepared and uploaded once, then any number of queries
    run against them. Use it when the same targets are searched many times; for a
    one-off batch of targets and queries, :func:`nvmolkit.substructure.hasSubstructMatch`
    avoids the upload step.

    Molecules added with :meth:`addMol` or :meth:`addMols` receive stable indices
    in insertion order and become searchable after :meth:`finalize`. Queries keep
    seeing the last finalized collection while newly added molecules are pending,
    so the library can grow between query rounds.

    Query methods return :class:`concurrent.futures.Future` objects so several
    queries can be in flight at once; the ``*Sync`` variants block for the result.

    Matching follows RDKit's ``SubstructLibrary`` with ``useChirality=False``:
    recursive SMARTS are supported and chirality is ignored. Targets outside the
    GPU representation limits (more than 128 atoms, high-degree atoms, isotopes
    above 255, or dative and other unsupported bond types) are matched with
    RDKit on the CPU, so results cover every added molecule.

    Example::

        from rdkit import Chem
        from nvmolkit.substruct_library import SubstructLibrary

        library = SubstructLibrary()
        library.addMols(Chem.MolFromSmiles(smiles) for smiles in ["CCO", "c1ccccc1O", "CCN"])
        library.finalize()
        library.getMatchesSync(Chem.MolFromSmarts("[OX2H]"))  # [0, 1]

    Args:
        chunkSize: Maximum number of molecules per GPU upload chunk. Default 65536.
        config: Search configuration. ``algorithm`` selects the ``"dfs"`` (default)
            or ``"gsi"`` backend, and ``gpuIds`` spreads the molecules across
            several GPUs. ``batchSize``, ``workerThreads``, and
            ``preprocessingThreads`` tune throughput; ``maxMatches`` and
            ``uniquify`` do not apply because queries return molecule indices.
        usePatternFingerprints: Screen targets with RDKit pattern fingerprints
            before exact GPU matching, as RDKit's ``PatternHolder`` does. The
            screen never changes results. Default True; disable it only for
            measurement or diagnostics.
    """

    def __init__(
        self,
        chunkSize: int = 65_536,
        config: SubstructSearchConfig | None = None,
        usePatternFingerprints: bool = True,
    ) -> None:
        """Create an empty library."""
        if config is None:
            config = SubstructSearchConfig()
        self._native = _NativeSubstructLibrary(int(chunkSize), config._as_native(), bool(usePatternFingerprints))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nvmolkit-substruct")

    def __len__(self) -> int:
        """Return the number of molecules in the finalized collection."""
        return len(self._native)

    @property
    def pendingSize(self) -> int:
        """Number of added molecules awaiting finalization."""
        return self._native.pendingSize

    @property
    def queryConcurrency(self) -> int:
        """Number of queries admitted concurrently under the GPU-memory budget."""
        return int(self._native.queryConcurrency)

    @property
    def batchesInFlightPerGpu(self) -> int:
        """Maximum number of mini-batches executing concurrently per GPU."""
        return int(self._native.batchesInFlightPerGpu)

    @property
    def workspaceBytesPerQueryPerGpu(self) -> int:
        """Conservative device-memory reservation for one admitted query."""
        return int(self._native.workspaceBytesPerQueryPerGpu)

    def addMol(self, molecule: Mol) -> int:
        """Stage a copy of ``molecule`` and return its stable library index.

        The molecule becomes searchable after the next :meth:`finalize`.
        """
        return self._native.addMol(molecule)

    def addMols(self, molecules: Iterable[Mol]) -> list[int]:
        """Stage copies of ``molecules`` in parallel and return their stable library indices."""
        return self._native.addMols(molecules)

    def finalize(self) -> None:
        """Upload pending molecules to the GPU and make them visible to subsequent queries.

        Must be called at least once before querying. Queries raise ``RuntimeError``
        until the library has been finalized.
        """
        self._executor.shutdown(wait=True)
        self._native.finalize()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, self.queryConcurrency),
            thread_name_prefix="nvmolkit-substruct",
        )

    def getMatches(self, query: Mol, maxResults: int = -1) -> Future[list[int]]:
        """Queue a query and return a future with the indices of matching molecules.

        Args:
            query: Query molecule, typically from ``Chem.MolFromSmarts`` or ``Chem.MolFromSmiles``.
            maxResults: Return at most this many indices, keeping the lowest. -1 (default) returns all.

        Returns:
            Future resolving to matching indices in ascending order.
        """
        return self._executor.submit(self._native.getMatches, query, int(maxResults))

    def countMatches(self, query: Mol) -> Future[int]:
        """Queue a query and return a future containing its match count."""
        return self._executor.submit(self._native.countMatches, query)

    def hasMatch(self, query: Mol) -> Future[bool]:
        """Queue a query and return a future containing whether a match exists."""
        return self._executor.submit(self._native.hasMatch, query)

    def getMatchesSync(self, query: Mol, maxResults: int = -1) -> list[int]:
        """Run one query synchronously."""
        return self.getMatches(query, maxResults).result()

    def countMatchesSync(self, query: Mol) -> int:
        """Run one count query synchronously."""
        return self.countMatches(query).result()

    def hasMatchSync(self, query: Mol) -> bool:
        """Run one existence query synchronously."""
        return self.hasMatch(query).result()
