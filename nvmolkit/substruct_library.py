# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-resident molecule library for repeated substructure queries."""

from __future__ import annotations

from collections.abc import Iterable

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

    def __len__(self) -> int:
        """Return the number of molecules in the finalized collection."""
        return len(self._native)

    @property
    def pendingSize(self) -> int:
        """Number of added molecules awaiting finalization."""
        return self._native.pendingSize

    def addMol(self, molecule: Mol) -> int:
        """Add a molecule and return its stable library index."""
        return self._native.addMol(molecule)

    def addMols(self, molecules: Iterable[Mol]) -> list[int]:
        """Add molecules and return their stable library indices."""
        return self._native.addMols(molecules)

    def finalize(self) -> None:
        """Upload pending molecules and publish them for subsequent queries."""
        self._native.finalize()

    def getMatches(self, query: Mol, maxResults: int = -1) -> list[int]:
        """Return matching molecule indices in insertion order."""
        return self._native.getMatches(query, int(maxResults))

    def countMatches(self, query: Mol) -> int:
        """Return the number of molecules that match the query."""
        return self._native.countMatches(query)

    def hasMatch(self, query: Mol) -> bool:
        """Return whether any finalized molecule matches the query."""
        return self._native.hasMatch(query)
