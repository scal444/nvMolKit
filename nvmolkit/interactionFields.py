# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Batched electrostatic molecular interaction fields."""

from typing import TYPE_CHECKING

import torch

from nvmolkit import _interactionFields
from nvmolkit.types import ArrayInput, AsyncGpuResult, _as_cuda_tensor, _resolve_cuda_stream

if TYPE_CHECKING:
    from rdkit.Chem import Mol

__all__ = ["CalcCoulombFields"]


def CalcCoulombFields(
    mols: list["Mol"],
    points: ArrayInput,
    *,
    model: str = "vacuum",
    conf_ids: list[int] | None = None,
    charge_property: str = "_GasteigerCharge",
    probe_charge: float = 1.0,
    abs_val: bool = False,
    alpha: float = 0.0,
    cutoff: float = 1.0,
    threshold: float | None = None,
    epsilon: float = 80.0,
    xi: float = 4.0,
    stream: torch.cuda.Stream | None = None,
) -> AsyncGpuResult:
    """Evaluate shared query points against one conformer per molecule.

    ``points`` is a NumPy array, PyTorch tensor, or AsyncGpuResult of shape
    ``(..., 3)`` and dtype float32 or float64. The float64 device result has
    shape ``(len(mols), ...)``, preserving all query dimensions and input
    molecule order. Regular grids can be supplied as ``(nz, ny, nx, 3)``.
    An empty molecule with a conformer produces zeros; empty molecule or
    query batches produce empty outputs.

    ``model="vacuum"`` evaluates RDKit's Coulomb field. ``"dielectric"``
    evaluates its distance-dependent dielectric field, including the 4-Angstrom
    neighborhood counts for atoms and query points. ``epsilon`` is solvent
    permittivity and ``xi`` is solute permittivity; both must be positive.
    Energies are in kJ/mol, coordinates in Angstrom, charges in electron units.

    Each atom must already have a finite partial charge in ``charge_property``;
    this function does not calculate charges or modify molecules. ``conf_ids``
    defaults to the first conformer of each molecule. ``-1`` selects the first
    conformer; nonnegative values are RDKit conformer IDs. Repeat a molecule
    with different IDs to evaluate several conformers.

    ``alpha`` is the nonnegative softcore parameter in squared Angstrom.
    Values below 1e-6 use the minimum-distance ``cutoff`` instead, including
    RDKit's model-specific clamp for very small cutoffs. ``threshold=None``
    includes all atoms; otherwise only distances strictly below this finite,
    nonnegative threshold contribute. Neighborhood counts always include all
    atoms. ``abs_val=True`` returns the negative absolute value after summation.

    CUDA inputs must be on the stream's device. Producers on other streams
    must be ordered before this call. Query validation and coordinate upload
    synchronize the selected stream; field evaluation remains asynchronous.
    Consumers on another stream must wait for the selected stream.
    """
    mols = list(mols)
    if any(mol is None for mol in mols):
        raise ValueError("Molecules must not be None")
    if model not in ("vacuum", "dielectric"):
        raise ValueError("model must be 'vacuum' or 'dielectric'")
    if threshold is not None and threshold < 0:
        raise ValueError("threshold must be nonnegative or None")
    ids = [-1] * len(mols) if conf_ids is None else list(conf_ids)
    if len(ids) != len(mols):
        raise ValueError("Expected one conformer ID per molecule")
    active_stream = _resolve_cuda_stream(stream, points)
    with torch.cuda.device(active_stream.device), torch.cuda.stream(active_stream):
        queries = _as_cuda_tensor("points", points, stream=active_stream)
        if queries.ndim < 2 or queries.shape[-1] != 3:
            raise ValueError("points must have shape (..., 3) with at least two dimensions")
        if queries.dtype not in (torch.float32, torch.float64):
            raise ValueError("points must have dtype float32 or float64")
        queries = queries.to(torch.float64).contiguous()
        if not bool(torch.isfinite(queries).all()):
            raise ValueError("Query coordinates must be finite")
        queries.record_stream(active_stream)
        result = AsyncGpuResult(
            _interactionFields.CalcCoulombFields(
                mols,
                queries.data_ptr(),
                queries.numel() // 3,
                ids,
                charge_property,
                model == "dielectric",
                probe_charge,
                abs_val,
                alpha,
                cutoff,
                -1.0 if threshold is None else threshold,
                epsilon,
                xi,
                active_stream.cuda_stream,
            )
        )
        result.arr = result.torch().reshape((len(mols), *queries.shape[:-1]))
        return result
