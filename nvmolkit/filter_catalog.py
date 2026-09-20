# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU filter catalogs for screening batches of molecules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import IntFlag

from rdkit.Chem import Mol

from nvmolkit._filterCatalog import FilterCatalog as _NativeFilterCatalog
from nvmolkit.substructure import SubstructSearchConfig

__all__ = ["FilterCatalog", "FilterCatalogEntry", "FilterCatalogPreset"]


class FilterCatalogPreset(IntFlag):
    """Filter collections bundled with RDKit."""

    PAINS_A = 1 << 1
    PAINS_B = 1 << 2
    PAINS_C = 1 << 3
    PAINS = PAINS_A | PAINS_B | PAINS_C
    BRENK = 1 << 4
    NIH = 1 << 5
    ZINC = 1 << 6
    CHEMBL_GLAXO = 1 << 7
    CHEMBL_DUNDEE = 1 << 8
    CHEMBL_BMS = 1 << 9
    CHEMBL_SURECHEMBL = 1 << 10
    CHEMBL_MLSMR = 1 << 11
    CHEMBL_INPHARMATICA = 1 << 12
    CHEMBL_LINT = 1 << 13
    CHEMBL_Glaxo = CHEMBL_GLAXO
    CHEMBL_Dundee = CHEMBL_DUNDEE
    CHEMBL_SureChEMBL = CHEMBL_SURECHEMBL
    CHEMBL_Inpharmatica = CHEMBL_INPHARMATICA
    CHEMBL = (
        CHEMBL_GLAXO
        | CHEMBL_DUNDEE
        | CHEMBL_BMS
        | CHEMBL_SURECHEMBL
        | CHEMBL_MLSMR
        | CHEMBL_INPHARMATICA
        | CHEMBL_LINT
    )
    ALL = PAINS | BRENK | NIH | ZINC | CHEMBL


def _normalize_preset(preset: FilterCatalogPreset | str | int) -> FilterCatalogPreset:
    if isinstance(preset, str):
        return FilterCatalogPreset[preset]
    return FilterCatalogPreset(preset)


@dataclass(frozen=True)
class FilterCatalogEntry:
    """Description and application metadata for one filter query."""

    id: int
    description: str
    smarts: str
    triggerCount: int
    metadata: dict[str, str]


class FilterCatalog:
    """A persistent query catalog applied to batches of target molecules.

    Query definitions remain on the CPU until :meth:`finalize` publishes them
    to the GPU. Query methods preserve target order. Match IDs refer to entries
    in insertion order and can be resolved with :meth:`getEntry`.
    """

    def __init__(
        self,
        preset: FilterCatalogPreset | str | int | None = None,
        config: SubstructSearchConfig | None = None,
    ) -> None:
        """Create an empty catalog or load one RDKit built-in filter collection."""
        if config is None:
            config = SubstructSearchConfig()
        self._native = _NativeFilterCatalog(config._as_native())
        if preset is not None:
            self._native.addPreset(int(_normalize_preset(preset)))

    def __len__(self) -> int:
        """Return the number of finalized catalog entries."""
        return len(self._native)

    @property
    def pendingSize(self) -> int:
        """Number of entries awaiting finalization."""
        return self._native.pendingSize

    def addQuery(
        self,
        query: Mol,
        description: str = "",
        triggerCount: int = 1,
        metadata: Mapping[str, str] | None = None,
    ) -> int:
        """Add an RDKit query molecule and return its stable entry ID."""
        if triggerCount <= 0:
            raise ValueError("triggerCount must be greater than zero")
        return self._native.addQuery(query, description, int(triggerCount), dict(metadata or {}))

    def addPreset(self, preset: FilterCatalogPreset | str | int) -> int:
        """Stage every entry in an RDKit built-in filter collection."""
        return self._native.addPreset(int(_normalize_preset(preset)))

    def addSmarts(
        self,
        smarts: str,
        description: str = "",
        triggerCount: int = 1,
        metadata: Mapping[str, str] | None = None,
    ) -> int:
        """Parse and add a SMARTS query, returning its stable entry ID."""
        if triggerCount <= 0:
            raise ValueError("triggerCount must be greater than zero")
        return self._native.addSmarts(smarts, description, int(triggerCount), dict(metadata or {}))

    def finalize(self) -> None:
        """Upload pending queries and publish the resulting catalog."""
        self._native.finalize()

    def getEntry(self, entryId: int) -> FilterCatalogEntry:
        """Return the definition and metadata for an entry ID."""
        entry = self._native.getEntry(int(entryId))
        return FilterCatalogEntry(
            id=entry["id"],
            description=entry["description"],
            smarts=entry["smarts"],
            triggerCount=entry["triggerCount"],
            metadata=dict(entry["metadata"]),
        )

    def hasMatch(self, molecules: Iterable[Mol]) -> list[bool]:
        """Return whether each target matches any catalog entry."""
        return self._native.hasMatch(molecules)

    def getFirstMatch(self, molecules: Iterable[Mol]) -> list[int | None]:
        """Return the first matching entry ID for each target, or ``None``."""
        return self._native.getFirstMatch(molecules)

    def getMatches(self, molecules: Iterable[Mol]) -> list[list[int]]:
        """Return all matching entry IDs for each target in catalog order."""
        return self._native.getMatches(molecules)
