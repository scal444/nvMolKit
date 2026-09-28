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
    """A collection of molecules prepared for repeated GPU substructure searches.

    Molecules added after construction become searchable after :meth:`finalize`.
    Queries continue to see the last successfully finalized collection while
    additional molecules are pending.
    """

    def __init__(
        self,
        chunkSize: int = 65_536,
        config: SubstructSearchConfig | None = None,
    ) -> None:
        """Create an empty library with the requested chunk size and search configuration."""
        if config is None:
            config = SubstructSearchConfig()
        self._native = _NativeSubstructLibrary(int(chunkSize), config._as_native())
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
        """Add a molecule and return its stable library index."""
        return self._native.addMol(molecule)

    def addMols(self, molecules: Iterable[Mol]) -> list[int]:
        """Add molecules and return their stable library indices."""
        return self._native.addMols(molecules)

    def finalize(self) -> None:
        """Upload pending molecules and publish them for subsequent queries."""
        self._executor.shutdown(wait=True)
        self._native.finalize()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, self.queryConcurrency),
            thread_name_prefix="nvmolkit-substruct",
        )

    def getMatches(self, query: Mol, maxResults: int = -1) -> Future[list[int]]:
        """Queue a query and return a future containing matching molecule indices."""
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
