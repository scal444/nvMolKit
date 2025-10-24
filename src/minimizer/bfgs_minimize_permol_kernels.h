// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NVMOLKIT_BFGS_MINIMIZE_PERMOL_KERNELS_H
#define NVMOLKIT_BFGS_MINIMIZE_PERMOL_KERNELS_H

#include <cuda_runtime.h>

#include "mmff_kernels.h"

namespace nvMolKit {

/// Launch per-molecule BFGS minimization kernel
/// 
/// This kernel performs complete BFGS minimization for a batch of molecules,
/// with one thread block per molecule. Suitable for small to medium-sized
/// molecules (up to ~2048 atoms).
///
/// Uses pre-allocated buffers from BfgsBatchMinimizer to avoid duplication.
///
/// \param binCounts Array of molecule counts per size bin (size 5: bins for 32,64,128,256,2048 atoms)
/// \param binMolIds Array of device pointers to molecule ID lists (size 5, one per bin)
/// \param atomStarts Array of atom start indices (size numMols+1)
/// \param hessianStarts Array of Hessian start indices (size numMols+1)
/// \param numIters Maximum number of BFGS iterations
/// \param gradTol Gradient tolerance for convergence
/// \param scaleGrads Whether to scale gradients to match RDKit behavior
/// \param terms Force field terms for all molecules
/// \param systemIndices Indices mapping terms to molecules
/// \param positions Atomic positions (updated in-place)
/// \param grad Pre-allocated gradient buffer (size: total atoms * dataDim)
/// \param inverseHessian Pre-allocated Hessian buffer (see hessianStarts for indexing)
/// \param scratchBuffers Pre-allocated scratch buffers (size: total atoms * dataDim each, 5 buffers)
/// \param energyOuts Output energies for each molecule
/// \param dataDim Dimensionality of coordinates (typically 3)
/// \param stream CUDA stream for asynchronous execution
/// \return CUDA error code
cudaError_t launchBfgsMinimizePerMolKernel(const int* binCounts,
                                           const int** binMolIds,
                                           const int* atomStarts,
                                           const int* hessianStarts,
                                           int numIters,
                                           double gradTol,
                                           bool scaleGrads,
                                           const MMFF::EnergyForceContribsDevicePtr& terms,
                                           const MMFF::BatchedIndicesDevicePtr& systemIndices,
                                           double* positions,
                                           double* grad,
                                           double* inverseHessian,
                                           double** scratchBuffers,
                                           double* energyOuts,
                                           int dataDim,
                                           cudaStream_t stream = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_BFGS_MINIMIZE_PERMOL_KERNELS_H

