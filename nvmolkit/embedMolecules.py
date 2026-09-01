# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""GPU-accelerated ETKDG conformer generation for multiple molecules.

This module provides GPU-accelerated implementations of ETKDG (Experimental-Torsion-Knowledge Distance-Geometry) conformer generation for multiple molecules using CUDA and OpenMP.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Optional, overload

import numpy as np

if TYPE_CHECKING:
    from rdkit.Chem import Mol
    from rdkit.Chem.rdDistGeom import EmbedParameters

__all__ = ["AnalyzeETKDGStage", "EmbedMolecules"]

from nvmolkit.types import CoordinateOutput, Device3DResult, HardwareOptions, PrecisionOptions  # noqa: I001
from nvmolkit import _embedMolecules  # type: ignore

def AnalyzeETKDGStage(
    molecules: list["Mol"],
    coordinates: Sequence[Any],
    params: "EmbedParameters",
    stage: str,
    backend: str = "BATCHED",
    precisionOptions: Optional[PrecisionOptions] = None,
    includeCpuReference: bool = True,
    fixedSteps: Optional[int] = None,
    maxSteps: Optional[int] = None,
) -> dict[str, Any]:
    """Run one ETKDGv3 minimization stage from exact input coordinates.

    This analysis API bypasses coordinate generation while reusing nvMolKit's
    production stage classes unchanged. ``FIRST`` and ``FOURTH`` accept one
    ``(num_atoms, 4)`` array per molecule; ``ETK`` accepts ``(num_atoms, 3)``.
    Each molecule represents one conformer, so callers comparing multiple
    conformers should pass independent molecule copies.

    Returned ``gpu_energies`` and optional ``cpu_energies`` are both rescored
    with the same RDKit stage force field. Native BFGS status is zero on
    convergence. ``stage_failed`` is the production stage's post-minimization
    check (currently meaningful for ``FIRST`` and ETK planarity checks).

    ``fixedSteps=N`` disables convergence stopping on both reference and GPU
    BFGS loops and executes exactly N outer updates. ``maxSteps=N`` retains
    normal convergence stopping with a shared cap. The returned
    ``cpu_iterations`` and ``gpu_iterations`` are per-system outer-update
    counts. These two options are mutually exclusive.
    """
    normalized_stage = str(stage).upper()
    aliases = {"DG_FIRST": "FIRST", "DG_FOURTH": "FOURTH", "ETK_3D": "ETK"}
    normalized_stage = aliases.get(normalized_stage, normalized_stage)
    if normalized_stage not in {"FIRST", "FOURTH", "ETK"}:
        raise ValueError("stage must be 'FIRST', 'FOURTH', or 'ETK'")
    if len(molecules) != len(coordinates):
        raise ValueError("coordinates must contain one array per molecule")
    dim = 3 if normalized_stage == "ETK" else 4
    flattened = []
    for idx, (mol, values) in enumerate(zip(molecules, coordinates)):
        if mol is None:
            raise ValueError(f"Molecule at index {idx} is None")
        array = np.asarray(values, dtype=np.float64)
        expected_shape = (mol.GetNumAtoms(), dim)
        if array.shape != expected_shape:
            raise ValueError(f"coordinates[{idx}] has shape {array.shape}; expected {expected_shape}")
        flattened.append(np.ascontiguousarray(array).ravel().tolist())
    if precisionOptions is None:
        precisionOptions = PrecisionOptions()
    if fixedSteps is not None and (not isinstance(fixedSteps, int) or fixedSteps <= 0):
        raise ValueError("fixedSteps must be a positive integer")
    if maxSteps is not None and (not isinstance(maxSteps, int) or maxSteps <= 0):
        raise ValueError("maxSteps must be a positive integer")
    if fixedSteps is not None and maxSteps is not None:
        raise ValueError("fixedSteps and maxSteps are mutually exclusive")
    return _embedMolecules.AnalyzeETKDGStage(
        molecules,
        flattened,
        params,
        normalized_stage,
        str(backend).upper(),
        precisionOptions._as_native(),
        bool(includeCpuReference),
        -1 if fixedSteps is None else fixedSteps,
        -1 if maxSteps is None else maxSteps,
    )


@overload
def EmbedMolecules(
    molecules: list["Mol"],
    params: "EmbedParameters",
    confsPerMolecule: int = 1,
    maxIterations: int = -1,
    hardwareOptions: Optional[HardwareOptions] = None,
    precisionOptions: Optional[PrecisionOptions] = None,
    output: Literal[CoordinateOutput.RDKIT_CONFORMERS] = CoordinateOutput.RDKIT_CONFORMERS,
    targetGpu: int = -1,
) -> None: ...
@overload
def EmbedMolecules(
    molecules: list["Mol"],
    params: "EmbedParameters",
    confsPerMolecule: int = 1,
    maxIterations: int = -1,
    hardwareOptions: Optional[HardwareOptions] = None,
    precisionOptions: Optional[PrecisionOptions] = None,
    *,
    output: Literal[CoordinateOutput.DEVICE],
    targetGpu: int = -1,
) -> Device3DResult: ...
def EmbedMolecules(
    molecules: list["Mol"],
    params: "EmbedParameters",
    confsPerMolecule: int = 1,
    maxIterations: int = -1,
    hardwareOptions: Optional[HardwareOptions] = None,
    precisionOptions: Optional[PrecisionOptions] = None,
    output: CoordinateOutput = CoordinateOutput.RDKIT_CONFORMERS,
    targetGpu: int = -1,
):
    """Embed multiple molecules with multiple conformers on GPUs.

    This function performs GPU-accelerated ETKDG conformer generation on multiple molecules.
    It uses CUDA for GPU acceleration and OpenMP for CPU parallelization to achieve high
    performance embedding of large molecule sets.

    nvMolKit implements a subset of features specified in the EmbedParameters class. The following features are restricted:

        - useRandomCoords must be True
        - Bounds matrices are not supported (setBoundsMat)
        - Custom Coulomb potentials are not supported (SetCPCI)
        - Coordinate constraints are not supported (SetCoordMap)
        - embedFragmentsSeparately is not supported. All fragments will be embedded together.

    Args:
        molecules: List of RDKit molecules to embed. Molecules should be prepared
                  (sanitized, explicit hydrogens added if needed).
        params: RDKit EmbedParameters object with embedding settings. Must have
               useRandomCoords=True for ETKDG.
        confsPerMolecule: Number of conformers to generate per molecule (default: 1)
        maxIterations: Maximum ETKDG iterations, -1 for automatic calculation (default: -1)
        hardwareOptions: HardwareOptions with hardware settings. If None, uses defaults.
        precisionOptions: Precision preset and per-axis overrides. Defaults to
            legacy float64 storage when omitted.
        output: ``RDKIT_CONFORMERS`` (default) writes generated conformers back into each input
            molecule in-place and returns ``None``. ``DEVICE`` retains conformer coordinates on
            GPU and returns a :class:`Device3DResult`; RDKit conformers are NOT modified.
            DEVICE mode is incompatible with ``params.pruneRmsThresh > 0``.
        targetGpu: In DEVICE mode, the GPU to consolidate the result onto. ``-1`` selects the
            first configured execution GPU.

    Returns:
        For ``RDKIT_CONFORMERS``: ``None``; input molecules are modified in-place with
        generated conformers.
        For ``DEVICE``: a :class:`Device3DResult` whose ``values`` field carries
        ``(total_atoms, 3)`` coordinates plus CSR indices.

    Raises:
        ValueError: If any molecule in the input list is invalid, or if hardware
                   configuration parameters are invalid
        RuntimeError: If CUDA operations fail or embedding encounters errors

    Example:
        >>> from rdkit import Chem
        >>> from rdkit.Chem.rdDistGeom import ETKDGv3
        >>> from nvmolkit.types import HardwareOptions
        >>> import nvmolkit.embedMolecules as embed
        >>>
        >>> # Load molecules
        >>> mol1 = Chem.AddHs(Chem.MolFromSmiles('CCO'))
        >>> mol2 = Chem.AddHs(Chem.MolFromSmiles('CCC'))
        >>>
        >>> # Set up embedding parameters
        >>> params = ETKDGv3()
        >>> params.useRandomCoords = True  # Required for nvMolKit ETKDG
        >>>
        >>> # Configure hardware options
        >>> hardware_opts = HardwareOptions(
        ...     preprocessingThreads=8,
        ...     batchSize=500,
        ...     batchesPerGpu=4,
        ...     gpuIds=[0, 1],
        ... )
        >>> embed.EmbedMolecules([mol1, mol2], params, confsPerMolecule=5, hardwareOptions=hardware_opts)
        >>>
        >>> # Check conformers were generated
        >>> mol1.GetNumConformers()  # Should be 5
        >>> mol2.GetNumConformers()  # Should be 5

    Note:
        - In ``RDKIT_CONFORMERS`` mode (default), input molecules are modified in-place with
          generated conformers. In ``DEVICE`` mode, RDKit conformers are not touched.
        - params.useRandomCoords must be True for ETKDG algorithm
        - If gpuIds is empty, all available GPUs (0 to N-1) will be used automatically
    """
    if not molecules:
        if output == CoordinateOutput.DEVICE:
            raise ValueError("EmbedMolecules(output=DEVICE) requires at least one molecule")
        return None

    for i, mol in enumerate(molecules):
        if mol is None:
            raise ValueError(f"Molecule at index {i} is None")

    if not params.useRandomCoords:
        raise ValueError("ETKDG requires useRandomCoords=True in EmbedParameters")

    if hardwareOptions is None:
        hardwareOptions = HardwareOptions()
    native_options = hardwareOptions._as_native()
    if precisionOptions is None:
        precisionOptions = PrecisionOptions()
    native_precision = precisionOptions._as_native()

    if output == CoordinateOutput.DEVICE:
        return _embedMolecules.EmbedMoleculesDevice(
            molecules, params, confsPerMolecule, maxIterations, native_options, int(targetGpu), native_precision
        )
    _embedMolecules.EmbedMolecules(
        molecules, params, confsPerMolecule, maxIterations, native_options, native_precision
    )
    return None
