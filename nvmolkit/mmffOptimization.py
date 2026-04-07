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

"""GPU-accelerated MMFF optimization for molecular conformers.

This module provides GPU-accelerated implementations of MMFF (Molecular Mechanics Force Field)
optimization for multiple molecules and conformers using CUDA and OpenMP.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from rdkit.Chem import AllChem

if TYPE_CHECKING:
    from rdkit.Chem import Mol
    from rdkit.ForceField.rdForceField import MMFFMolProperties as RDKitMMFFMolProperties

from nvmolkit import _mmffOptimization
from nvmolkit._mmff_bridge import default_rdkit_mmff_properties, make_internal_mmff_properties
from nvmolkit.types import HardwareOptions


def MMFFOptimizeMoleculesConfs(
    molecules: list["Mol"],
    maxIters: int = 200,
    nonBondedThreshold: float = 100.0,
    properties: "RDKitMMFFMolProperties | Sequence[RDKitMMFFMolProperties | None] | None" = None,
    ignoreInterfragInteractions: bool = True,
    hardwareOptions: HardwareOptions | None = None,
) -> list[list[float]]:
    """Optimize conformers for multiple molecules using MMFF force field with BFGS minimization.

    This function performs GPU-accelerated MMFF optimization on multiple molecules with
    multiple conformers each. It uses CUDA for GPU acceleration and OpenMP for CPU
    parallelization to achieve high performance.

    Args:
        molecules: List of RDKit molecules to optimize. Each molecule should have
                  conformers already generated.
        maxIters: Maximum number of BFGS optimization iterations (default: 200)
        nonBondedThreshold: Radius threshold for non-bonded interactions in Angstroms
            (default: 100.0).  Ignored when ``properties`` supplies per-molecule
            settings that already include a threshold.
        properties: RDKit ``MMFFMolProperties`` object, a per-molecule list of
            those objects, or ``None`` to use default MMFF94 settings.  A single
            object is broadcast to all molecules.  Allows selecting MMFF94 vs
            MMFF94s, dielectric settings, and per-term toggles.
        ignoreInterfragInteractions: Whether to omit interfragment non-bonded
            interactions (default: True).  Ignored when ``properties`` is given.
        hardwareOptions: Configures CPU and GPU batching, threading, and device selection. Will attempt to use reasonable defaults if not set.

    Returns:
        List of lists of energies, where each inner list contains the optimized energies
        for all conformers of the corresponding molecule. The order matches the input
        molecule order and conformer iteration order.

    Raises:
        ValueError: If any molecules in the input list are None or lack MMFF atom types.
            ``e.args[0]`` is a summary message, ``e.args[1]`` is a dict
            with keys ``"none"`` (indices of None molecules) and ``"no_params"``
            (indices of molecules lacking MMFF atom types). Example::

                try:
                    MMFFOptimizeMoleculesConfs(mols, ...)
                except ValueError as e:
                    failed = e.args[1]
                    none_idx = failed["none"]
                    no_params_idx = failed["no_params"]
        RuntimeError: If CUDA operations fail or optimization encounters errors

    Example:
        >>> from rdkit import Chem
        >>> from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
        >>> from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
        >>>
        >>> mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
        >>> rdDistGeom.EmbedMultipleConfs(mol, numConfs=5)
        >>>
        >>> # Default MMFF94 optimization
        >>> energies = MMFFOptimizeMoleculesConfs([mol])
        >>>
        >>> # MMFF94s with custom dielectric
        >>> props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
        >>> props.SetMMFFDielectricConstant(80.0)
        >>> energies = MMFFOptimizeMoleculesConfs([mol], properties=props)

    Note:
        - Input molecules are modified in-place with optimized conformer coordinates
    """
    if not molecules:
        return []

    none_indices = []
    no_params_indices = []
    for i, mol in enumerate(molecules):
        if mol is None:
            none_indices.append(i)
        elif not AllChem.MMFFHasAllMoleculeParams(mol):
            no_params_indices.append(i)

    if none_indices or no_params_indices:
        parts = []
        if none_indices:
            parts.append(f"None at indices {none_indices}")
        if no_params_indices:
            parts.append(f"lacking MMFF atom types at indices {no_params_indices}")
        raise ValueError(
            "; ".join(parts),
            {"none": none_indices, "no_params": no_params_indices},
        )

    if hardwareOptions is None:
        hardwareOptions = HardwareOptions()
    native_options = hardwareOptions._as_native()

    if properties is not None:
        native_props = _build_native_properties(molecules, properties, nonBondedThreshold, ignoreInterfragInteractions)
        if len(native_props) == 1:
            return _mmffOptimization.MMFFOptimizeMoleculesConfs(
                molecules, maxIters, native_props[0], native_options
            )
        return _mmffOptimization.MMFFOptimizeMoleculesConfsPerMol(
            molecules, maxIters, native_props, native_options
        )

    props = _mmffOptimization.MMFFProperties()
    props.nonBondedThreshold = nonBondedThreshold
    props.ignoreInterfragInteractions = ignoreInterfragInteractions
    return _mmffOptimization.MMFFOptimizeMoleculesConfs(molecules, maxIters, props, native_options)


def _build_native_properties(
    molecules: list["Mol"],
    properties: "RDKitMMFFMolProperties | Sequence[RDKitMMFFMolProperties | None] | None",
    non_bonded_threshold: float,
    ignore_interfrag_interactions: bool,
) -> list:
    if properties is None:
        return [_mmffOptimization.MMFFProperties() for _ in molecules]

    is_sequence = isinstance(properties, Sequence) and not hasattr(properties, "SetMMFFVariant")
    if is_sequence:
        if len(properties) != len(molecules):
            raise ValueError(f"Expected {len(molecules)} MMFFMolProperties, got {len(properties)}")
        result = []
        for mol, prop in zip(molecules, properties):
            if prop is None:
                source = default_rdkit_mmff_properties(mol)
            else:
                source = prop
            result.append(
                make_internal_mmff_properties(
                    source,
                    non_bonded_threshold=non_bonded_threshold,
                    ignore_interfrag_interactions=ignore_interfrag_interactions,
                )
            )
        return result

    return [
        make_internal_mmff_properties(
            properties,
            non_bonded_threshold=non_bonded_threshold,
            ignore_interfrag_interactions=ignore_interfrag_interactions,
        )
    ]
