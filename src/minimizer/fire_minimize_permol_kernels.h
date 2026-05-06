// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMOLKIT_FIRE_MINIMIZE_PERMOL_KERNELS_H
#define NVMOLKIT_FIRE_MINIMIZE_PERMOL_KERNELS_H

#include <cstdint>
#include <cuda_runtime.h>

#include "dist_geom_kernels.h"
#include "fire_minimizer.h"
#include "mmff_kernels.h"

namespace nvMolKit {

//! \brief Per-block launch configuration for the per-molecule FIRE kernels.
//!
//! Mirrors the BFGS per-mol launcher pattern: caller supplies per-system state
//! buffers owned by ::FireBatchMinimizer and the per-FF term descriptors. The
//! kernel iterates the full FIRE 2.0 loop internally and writes a per-system
//! status (0 = converged, 1 = active) into @p statuses.
struct FirePerMolLaunchParams {
  int            numIters       = 0;           //!< Maximum FIRE iterations to run inside the kernel.
  double         gradTol        = 0.0;         //!< sqrt(sum(grad^2)) per-system convergence tolerance.
  bool           takeHalfStepBack = true;      //!< When true and power<0, take a half step back and zero v.
  bool           useAbc         = false;       //!< Apply ABC-FIRE mixer correction.
  bool           useMass        = false;       //!< Mass-weight the force kick (requires non-null masses).
};

//! \brief Launch per-molecule FIRE 2.0 minimization - MMFF specialization.
//! \note ::FireOptions stuck-detection fields are not supported on the per-mol path.
cudaError_t launchFirePerMolKernel(int                                       numMols,
                                   const int*                                molIds,
                                   int                                       maxAtoms,
                                   const int*                                atomStarts,
                                   const FireOptions&                        fireOptions,
                                   int                                       numIters,
                                   double                                    gradTol,
                                   const MMFF::EnergyForceContribsDevicePtr& terms,
                                   const MMFF::BatchedIndicesDevicePtr&      systemIndices,
                                   double*                                   positions,
                                   double*                                   grad,
                                   double*                                   velocities,
                                   double*                                   alphas,
                                   double*                                   dts,
                                   int*                                      nStepsPositive,
                                   const double*                             masses,
                                   double*                                   energyOuts,
                                   uint8_t*                                  statuses,
                                   cudaStream_t                              stream = nullptr);

//! \brief Launch per-molecule FIRE 2.0 minimization - ETK specialization.
cudaError_t launchFirePerMolKernelETK(int                                             numMols,
                                      const int*                                      molIds,
                                      int                                             maxAtoms,
                                      const int*                                      atomStarts,
                                      const FireOptions&                              fireOptions,
                                      int                                             numIters,
                                      double                                          gradTol,
                                      const DistGeom::Energy3DForceContribsDevicePtr& terms,
                                      const DistGeom::BatchedIndices3DDevicePtr&      systemIndices,
                                      double*                                         positions,
                                      double*                                         grad,
                                      double*                                         velocities,
                                      double*                                         alphas,
                                      double*                                         dts,
                                      int*                                            nStepsPositive,
                                      const double*                                   masses,
                                      double*                                         energyOuts,
                                      uint8_t*                                        statuses,
                                      cudaStream_t                                    stream = nullptr);

//! \brief Launch per-molecule FIRE 2.0 minimization - DG specialization.
cudaError_t launchFirePerMolKernelDG(int                                           numMols,
                                     const int*                                    molIds,
                                     int                                           maxAtoms,
                                     const int*                                    atomStarts,
                                     const FireOptions&                            fireOptions,
                                     int                                           numIters,
                                     double                                        gradTol,
                                     const DistGeom::EnergyForceContribsDevicePtr& terms,
                                     const DistGeom::BatchedIndicesDevicePtr&      systemIndices,
                                     double*                                       positions,
                                     double*                                       grad,
                                     double*                                       velocities,
                                     double*                                       alphas,
                                     double*                                       dts,
                                     int*                                          nStepsPositive,
                                     const double*                                 masses,
                                     double*                                       energyOuts,
                                     double                                        chiralWeight,
                                     double                                        fourthDimWeight,
                                     uint8_t*                                      statuses,
                                     cudaStream_t                                  stream = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_FIRE_MINIMIZE_PERMOL_KERNELS_H
