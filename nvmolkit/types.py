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

"""Types facilitating GPU-accelerated operations."""
from dataclasses import dataclass
from typing import Iterable, List

import torch

from nvmolkit import _embedMolecules  # type: ignore


class HardwareOptions:
    """Configures GPU hardware settings for batch operations.

    Use this class to control threading and batching behavior for GPU-accelerated
    workflows such as ETKDG embedding and MMFF optimization.

    Parameters:
        preprocessingThreads: Number of CPU threads for preprocessing. Use -1 to auto-detect and use all CPU threads.
        batchSize: Number of conformers processed per batch. -1 for default.
        batchesPerGpu: Number of batches processed concurrently on each GPU. Use -1 for default.
        gpuIds: GPU device IDs to target. Provide an empty list to use all available GPUs.
    """

    def __init__(
        self,
        preprocessingThreads: int = -1,
        batchSize: int = -1,
        batchesPerGpu: int = -1,
        gpuIds: Iterable[int] | None = None,
    ) -> None:
        if _embedMolecules is None:  # propagate real import failure early
            raise ImportError("nvmolkit._embedMolecules is not available; build native extensions")
        native = _embedMolecules.BatchHardwareOptions()
        native.preprocessingThreads = int(preprocessingThreads)
        native.batchSize = int(batchSize)
        native.gpuIds = list(gpuIds) if gpuIds is not None else []
        self._native = native
        self.batchesPerGpu = batchesPerGpu  # reuses setter validation

    @property
    def preprocessingThreads(self) -> int:
        """Number of CPU threads for preprocessing. -1 auto-detects."""
        return self._native.preprocessingThreads

    @preprocessingThreads.setter
    def preprocessingThreads(self, value: int) -> None:
        self._native.preprocessingThreads = int(value)

    @property
    def batchSize(self) -> int:
        """Number of conformers per batch. -1 selects an auto-tuned value."""
        return self._native.batchSize

    @batchSize.setter
    def batchSize(self, value: int) -> None:
        self._native.batchSize = int(value)

    @property
    def batchesPerGpu(self) -> int:
        """Batches processed concurrently on each GPU. -1 auto."""
        return self._native.batchesPerGpu

    @batchesPerGpu.setter
    def batchesPerGpu(self, value: int) -> None:
        value = int(value)
        if value != -1 and value <= 0:
            raise ValueError("batchesPerGpu must be greater than 0 or -1 for automatic")
        self._native.batchesPerGpu = value

    @property
    def gpuIds(self) -> List[int]:
        """GPU device IDs to target. Empty list uses all available GPUs."""
        return list(self._native.gpuIds)

    @gpuIds.setter
    def gpuIds(self, value: Iterable[int]) -> None:
        self._native.gpuIds = list(value)

    def _as_native(self):
        """Internal: return the underlying BatchHardwareOptions object."""
        return self._native


@dataclass
class MMFFProperties:
    """Configuration for MMFF forcefield construction.

    These settings correspond to RDKit's ``MMFFMolProperties`` configuration.
    The resulting parameters are baked into the flattened forcefield during
    build, so changing them requires a rebuild of the batched forcefield.
    """

    variant: str = "MMFF94"
    dielectric_constant: float = 1.0
    dielectric_model: int = 1
    non_bonded_threshold: float = 100.0
    ignore_interfrag_interactions: bool = True
    bond_term: bool = True
    angle_term: bool = True
    stretch_bend_term: bool = True
    oop_term: bool = True
    torsion_term: bool = True
    vdw_term: bool = True
    ele_term: bool = True

    def _as_native(self):
        """Internal: convert to the native MMFF properties struct."""
        from nvmolkit import _batchedForcefield  # type: ignore

        native = _batchedForcefield.MMFFPropertiesNative()
        native.variant = str(self.variant)
        native.dielectricConstant = float(self.dielectric_constant)
        native.dielectricModel = int(self.dielectric_model)
        native.nonBondedThreshold = float(self.non_bonded_threshold)
        native.ignoreInterfragInteractions = bool(self.ignore_interfrag_interactions)
        native.bondTerm = bool(self.bond_term)
        native.angleTerm = bool(self.angle_term)
        native.stretchBendTerm = bool(self.stretch_bend_term)
        native.oopTerm = bool(self.oop_term)
        native.torsionTerm = bool(self.torsion_term)
        native.vdwTerm = bool(self.vdw_term)
        native.eleTerm = bool(self.ele_term)
        return native

class AsyncGpuResult:
    """Handle to a GPU result.

    Populates the ``__cuda_array_interface__`` attribute which can be consumed by other libraries. Note that
    this result is async, and the data cannot be accessed without a sync, such as ``torch.cuda.synchronize()``.
    """
    def __init__(self, obj):
        """Internal construction of the AsyncGpuResult object."""
        if not hasattr(obj, '__cuda_array_interface__'):
            raise TypeError(f"Object {obj} does not have a __cuda_array_interface__ attribute")
        self.arr = torch.as_tensor(obj, device='cuda')

    @property
    def __cuda_array_interface__(self):
        """Return the CUDA array interface for the underlying data."""
        return self.arr.__cuda_array_interface__

    @property
    def device(self):
        """Return the device of the underlying data."""
        return self.arr.device

    def torch(self):
        """Return the underlying data as a torch tensor. This is an asynchronous operation."""
        return self.arr

    def numpy(self):
        """Return the underlying data as a numpy array. This is a blocking operation."""
        torch.cuda.synchronize()
        return self.arr.cpu().numpy()
