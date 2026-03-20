"""Python API for batched forcefield energy and gradient evaluation."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdkit.Chem import Mol
    from rdkit.Chem.rdDistGeom import EmbedParameters

from nvmolkit import _batchedForcefield  # type: ignore
from nvmolkit.types import MMFFProperties


@dataclass
class _DistanceConstraint:
    idx1: int
    idx2: int
    relative: bool
    min_len: float
    max_len: float
    force_constant: float


@dataclass
class _PositionConstraint:
    idx: int
    max_displ: float
    force_constant: float


@dataclass
class _AngleConstraint:
    idx1: int
    idx2: int
    idx3: int
    relative: bool
    min_angle_deg: float
    max_angle_deg: float
    force_constant: float


@dataclass
class _TorsionConstraint:
    idx1: int
    idx2: int
    idx3: int
    idx4: int
    relative: bool
    min_dihedral_deg: float
    max_dihedral_deg: float
    force_constant: float


class MMFFBatchElement:
    def __init__(self, parent: "MMFFBatchedForcefield", idx: int):
        self._parent = parent
        self._idx = idx

    @property
    def num_atoms(self) -> int:
        return self._parent._molecules[self._idx].GetNumAtoms()

    def add_distance_constraint(
        self,
        idx1: int,
        idx2: int,
        relative: bool,
        min_len: float,
        max_len: float,
        force_constant: float,
    ) -> None:
        self._parent._validate_atom_indices(self._idx, idx1, idx2)
        self._parent._distance_constraints[self._idx].append(
            _DistanceConstraint(idx1, idx2, relative, min_len, max_len, force_constant)
        )
        self._parent._dirty = True

    def add_position_constraint(
        self,
        idx: int,
        max_displ: float,
        force_constant: float,
    ) -> None:
        self._parent._validate_atom_indices(self._idx, idx)
        self._parent._position_constraints[self._idx].append(
            _PositionConstraint(idx, max_displ, force_constant)
        )
        self._parent._dirty = True

    def add_angle_constraint(
        self,
        idx1: int,
        idx2: int,
        idx3: int,
        relative: bool,
        min_angle_deg: float,
        max_angle_deg: float,
        force_constant: float,
    ) -> None:
        self._parent._validate_atom_indices(self._idx, idx1, idx2, idx3)
        self._parent._angle_constraints[self._idx].append(
            _AngleConstraint(idx1, idx2, idx3, relative, min_angle_deg, max_angle_deg, force_constant)
        )
        self._parent._dirty = True

    def add_torsion_constraint(
        self,
        idx1: int,
        idx2: int,
        idx3: int,
        idx4: int,
        relative: bool,
        min_dihedral_deg: float,
        max_dihedral_deg: float,
        force_constant: float,
    ) -> None:
        self._parent._validate_atom_indices(self._idx, idx1, idx2, idx3, idx4)
        self._parent._torsion_constraints[self._idx].append(
            _TorsionConstraint(
                idx1, idx2, idx3, idx4, relative, min_dihedral_deg, max_dihedral_deg, force_constant
            )
        )
        self._parent._dirty = True


class MMFFBatchedForcefield:
    def __init__(
        self,
        molecules: list["Mol"],
        properties: MMFFProperties | list[MMFFProperties] | None = None,
        conf_id: int | list[int] = -1,
    ):
        self._molecules = molecules
        self._properties = self._normalize_properties(properties)
        self._conf_ids = self._normalize_conf_ids(conf_id)
        self._distance_constraints: list[list[_DistanceConstraint]] = [[] for _ in molecules]
        self._position_constraints: list[list[_PositionConstraint]] = [[] for _ in molecules]
        self._angle_constraints: list[list[_AngleConstraint]] = [[] for _ in molecules]
        self._torsion_constraints: list[list[_TorsionConstraint]] = [[] for _ in molecules]
        self._native_ff = None
        self._dirty = True
        self.num_molecules = len(molecules)
        self.data_dim = 3

    def __len__(self) -> int:
        return len(self._molecules)

    def __getitem__(self, idx: int) -> MMFFBatchElement:
        if idx < 0 or idx >= len(self._molecules):
            raise IndexError(f"Batch element index {idx} out of range")
        return MMFFBatchElement(self, idx)

    def _normalize_properties(
        self, properties: MMFFProperties | list[MMFFProperties] | None
    ) -> list[MMFFProperties]:
        if properties is None:
            return [MMFFProperties() for _ in self._molecules]
        if isinstance(properties, MMFFProperties):
            return [replace(properties) for _ in self._molecules]
        if len(properties) != len(self._molecules):
            raise ValueError(
                f"Expected {len(self._molecules)} MMFFProperties objects, got {len(properties)}"
            )
        return [replace(prop) for prop in properties]

    def _normalize_conf_ids(self, conf_id: int | list[int]) -> list[int]:
        if isinstance(conf_id, int):
            return [conf_id for _ in self._molecules]
        if len(conf_id) != len(self._molecules):
            raise ValueError(f"Expected {len(self._molecules)} conf_id values, got {len(conf_id)}")
        return list(conf_id)

    def _validate_atom_indices(self, batch_idx: int, *indices: int) -> None:
        num_atoms = self._molecules[batch_idx].GetNumAtoms()
        for idx in indices:
            if idx < 0 or idx >= num_atoms:
                raise IndexError(
                    f"Atom index {idx} out of range for molecule {batch_idx} with {num_atoms} atoms"
                )

    def _build(self) -> None:
        if not self._molecules:
            self._native_ff = None
            self._dirty = False
            return
        native_properties = [props._as_native() for props in self._properties]
        distance_constraints = [
            [
                (
                    c.idx1,
                    c.idx2,
                    c.relative,
                    c.min_len,
                    c.max_len,
                    c.force_constant,
                )
                for c in constraints
            ]
            for constraints in self._distance_constraints
        ]
        position_constraints = [
            [(c.idx, c.max_displ, c.force_constant) for c in constraints]
            for constraints in self._position_constraints
        ]
        angle_constraints = [
            [
                (
                    c.idx1,
                    c.idx2,
                    c.idx3,
                    c.relative,
                    c.min_angle_deg,
                    c.max_angle_deg,
                    c.force_constant,
                )
                for c in constraints
            ]
            for constraints in self._angle_constraints
        ]
        torsion_constraints = [
            [
                (
                    c.idx1,
                    c.idx2,
                    c.idx3,
                    c.idx4,
                    c.relative,
                    c.min_dihedral_deg,
                    c.max_dihedral_deg,
                    c.force_constant,
                )
                for c in constraints
            ]
            for constraints in self._torsion_constraints
        ]
        self._native_ff = _batchedForcefield.NativeMMFFBatchedForcefield(
            self._molecules,
            native_properties,
            self._conf_ids,
            distance_constraints,
            position_constraints,
            angle_constraints,
            torsion_constraints,
        )
        self._dirty = False

    def _ensure_built(self) -> None:
        if self._dirty or self._native_ff is None:
            self._build()

    def rebuild(self) -> None:
        self._build()

    def compute_energy(self) -> list[float]:
        if not self._molecules:
            return []
        self._ensure_built()
        return self._native_ff.computeEnergy()

    def compute_gradients(self) -> list[list[float]]:
        if not self._molecules:
            return []
        self._ensure_built()
        return self._native_ff.computeGradients()


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
