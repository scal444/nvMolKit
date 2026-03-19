"""Python API for batched forcefield energy and gradient evaluation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdkit.Chem import Mol
    from rdkit.Chem.rdDistGeom import EmbedParameters

from nvmolkit import _batchedForcefield  # type: ignore


class MMFFBatchedForcefield:
    def __init__(self, molecules: list["Mol"], nonBondedThreshold: float = 100.0):
        self._molecules = molecules
        self._non_bonded_threshold = nonBondedThreshold
        self.num_molecules = len(molecules)
        self.data_dim = 3

    def compute_energy(self) -> list[float]:
        return _batchedForcefield.MMFFComputeEnergies(self._molecules, self._non_bonded_threshold)

    def compute_gradients(self) -> list[list[float]]:
        return _batchedForcefield.MMFFComputeGradients(self._molecules, self._non_bonded_threshold)


class DGBatchedForcefield:
    def __init__(self, molecules: list["Mol"], params: "EmbedParameters"):
        self._molecules = molecules
        self._params = params
        self.num_molecules = len(molecules)
        self.data_dim = 4

    def compute_energy(self) -> list[float]:
        return _batchedForcefield.DGComputeEnergies(self._molecules, self._params)

    def compute_gradients(self) -> list[list[float]]:
        return _batchedForcefield.DGComputeGradients(self._molecules, self._params)


class ETKBatchedForcefield:
    def __init__(self, molecules: list["Mol"], params: "EmbedParameters"):
        self._molecules = molecules
        self._params = params
        self.num_molecules = len(molecules)
        self.data_dim = 3

    def compute_energy(self) -> list[float]:
        return _batchedForcefield.ETKComputeEnergies(self._molecules, self._params)

    def compute_gradients(self) -> list[list[float]]:
        return _batchedForcefield.ETKComputeGradients(self._molecules, self._params)
