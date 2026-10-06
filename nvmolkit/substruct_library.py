# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Molecules kept on the GPU for repeated substructure queries."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

from rdkit.Chem import Mol

from nvmolkit._substructLibrary import SubstructLibrary as _NativeSubstructLibrary
from nvmolkit.substructure import SubstructSearchConfig

__all__ = ["SubstructLibrary"]

_T = TypeVar("_T")


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
        config: Search configuration. ``algorithm`` selects the ``"dfs"`` (default)
            or ``"gsi"`` backend, and ``gpuIds`` spreads the molecules across
            several GPUs. ``batchSize``, ``workerThreads``, and
            ``preprocessingThreads`` tune throughput; ``maxMatches`` and
            ``uniquify`` do not apply because queries return molecule indices.
    """

    def __init__(self, config: SubstructSearchConfig | None = None) -> None:
        """Create an empty library."""
        if config is None:
            config = SubstructSearchConfig()
        self._native = _NativeSubstructLibrary(config._as_native())
        self._executorLock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nvmolkit-substruct")

    def __len__(self) -> int:
        """Return the number of searchable molecules."""
        return len(self._native)

    @property
    def pendingSize(self) -> int:
        """Number of molecules added since the last :meth:`finalize`."""
        return self._native.pendingSize

    @property
    def maxConcurrentQueries(self) -> int:
        """How many queries can run at once, set by :meth:`finalize` from the GPU memory left."""
        return int(self._native.maxConcurrentQueries)

    def addMol(self, molecule: Mol) -> int:
        """Copy ``molecule`` into the library and return its index.

        The molecule becomes searchable after the next :meth:`finalize`.
        """
        return self._native.addMol(molecule)

    def addMols(self, molecules: Iterable[Mol]) -> list[int]:
        """Copy ``molecules`` into the library in parallel and return their indices."""
        return self._native.addMols(molecules)

    def finalize(self) -> None:
        """Upload molecules added since the last call and make them searchable.

        Must be called at least once before querying; queries raise ``RuntimeError``
        until then. Queries already queued search whatever is finalized when they run.
        """
        try:
            self._native.finalize()
        finally:
            # A failed finalize keeps the previous molecules searchable, so always size the
            # executor to what the native library now allows. Queued queries keep running.
            with self._executorLock:
                previous = self._executor
                self._executor = ThreadPoolExecutor(
                    max_workers=max(1, self.maxConcurrentQueries),
                    thread_name_prefix="nvmolkit-substruct",
                )
            previous.shutdown(wait=False)

    def _submit(self, function: Callable[..., _T], *args: Any) -> Future[_T]:
        with self._executorLock:
            return self._executor.submit(function, *args)

    def getMatches(self, query: Mol, maxResults: int = -1) -> Future[list[int]]:
        """Queue a query and return a future with the indices of matching molecules.

        Args:
            query: Query molecule, typically from ``Chem.MolFromSmarts`` or ``Chem.MolFromSmiles``.
            maxResults: Return at most this many indices, keeping the lowest. -1 (default) returns all.

        Returns:
            Future resolving to matching indices in ascending order.
        """
        return self._submit(self._native.getMatches, query, int(maxResults))

    def countMatches(self, query: Mol) -> Future[int]:
        """Queue a query and return a future containing its match count."""
        return self._submit(self._native.countMatches, query)

    def hasMatch(self, query: Mol) -> Future[bool]:
        """Queue a query and return a future containing whether a match exists."""
        return self._submit(self._native.hasMatch, query)

    def getMatchesSync(self, query: Mol, maxResults: int = -1) -> list[int]:
        """Run one query synchronously."""
        return self.getMatches(query, maxResults).result()

    def countMatchesSync(self, query: Mol) -> int:
        """Run one count query synchronously."""
        return self.countMatches(query).result()

    def hasMatchSync(self, query: Mol) -> bool:
        """Run one existence query synchronously."""
        return self.hasMatch(query).result()
