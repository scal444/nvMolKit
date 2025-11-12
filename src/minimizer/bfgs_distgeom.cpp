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

#include "bfgs_distgeom.h"

#include "bfgs_minimize.h"
#include "bfgs_minimize_permol_kernels.h"
#include "dist_geom.h"

namespace nvMolKit::DistGeom {

void DistGeomMinimizeBFGS(BatchedMolecularSystemHost&    molSystemHost,
                          BatchedMolecularDeviceBuffers& molSystemDevice,
                          detail::ETKDGContext&          context,
                          const int                      maxIters,
                          const double                   gradTol,
                          const bool                     repeatUntilConverged,
                          cudaStream_t                   stream) {
  // Setup device buffers
  setupDeviceBuffers(molSystemHost,
                     molSystemDevice,
                     context.systemHost.positions,
                     static_cast<int>(context.systemHost.atomStarts.size() - 1));

  const size_t numAtoms = context.systemHost.atomStarts.back();
  const size_t numPos   = context.systemHost.positions.size();
  const int    dim      = (numPos == numAtoms * 3) ? 3 : 4;

  // Create energy and gradient functions
  auto eFunc = [&](const double* positions) {
    computeEnergy(molSystemDevice,
                  context.systemDevice.atomStarts,
                  context.systemDevice.positions,
                  context.activeThisStage.data(),
                  positions,
                  stream);
  };

  auto gFunc = [&]() {
    computeGradients(molSystemDevice,
                     context.systemDevice.atomStarts,
                     context.systemDevice.positions,
                     context.activeThisStage.data(),
                     stream);
  };

  // Create and configure BFGS minimizer
  nvMolKit::BfgsBatchMinimizer bfgsMinimizer(/*dataDim=*/dim, nvMolKit::DebugLevel::NONE, true, stream);

  // Run minimization
  bool needsMore = bfgsMinimizer.minimize(maxIters,
                                          gradTol,
                                          context.systemHost.atomStarts,
                                          context.systemDevice.atomStarts,
                                          context.systemDevice.positions,
                                          molSystemDevice.grad,
                                          molSystemDevice.energyOuts,
                                          molSystemDevice.energyBuffer,
                                          eFunc,
                                          gFunc,
                                          context.activeThisStage.data());
  while (needsMore && repeatUntilConverged) {
    needsMore = bfgsMinimizer.minimize(maxIters,
                                       gradTol,
                                       context.systemHost.atomStarts,
                                       context.systemDevice.atomStarts,
                                       context.systemDevice.positions,
                                       molSystemDevice.grad,
                                       molSystemDevice.energyOuts,
                                       molSystemDevice.energyBuffer,
                                       eFunc,
                                       gFunc,
                                       context.activeThisStage.data());
  }
}

namespace {
// Helper to bin molecules by size for optimal kernel dispatch
void binMoleculesBySize(const std::vector<int>& atomStartsHost,
                        const std::vector<int>& activeMolIndices,
                        std::vector<std::vector<int>>& bins) {
  bins.resize(5);  // 5 bins to match BFGS kernel bins
  
  for (int molIdx : activeMolIndices) {
    const int numAtoms = atomStartsHost[molIdx + 1] - atomStartsHost[molIdx];
    
    if (numAtoms <= 32) {
      bins[0].push_back(molIdx);
    } else if (numAtoms <= 64) {
      bins[1].push_back(molIdx);
    } else if (numAtoms <= 128) {
      bins[2].push_back(molIdx);
    } else if (numAtoms <= 256) {
      bins[3].push_back(molIdx);
    } else {
      bins[4].push_back(molIdx);
    }
  }
}
}  // namespace

void DistGeomMinimizeBFGSPerMol(BatchedMolecularSystemHost&    molSystemHost,
                                BatchedMolecularDeviceBuffers& molSystemDevice,
                                detail::ETKDGContext&          context,
                                const int                      maxIters,
                                const double                   gradTol,
                                const bool                     repeatUntilConverged,
                                cudaStream_t                   stream) {
  // Setup device buffers
  setupDeviceBuffers(molSystemHost,
                     molSystemDevice,
                     context.systemHost.positions,
                     static_cast<int>(context.systemHost.atomStarts.size() - 1));

  const size_t numAtoms = context.systemHost.atomStarts.back();
  const size_t numPos   = context.systemHost.positions.size();
  const int    dim      = (numPos == numAtoms * 3) ? 3 : 4;
  const int    numMols  = static_cast<int>(context.systemHost.atomStarts.size() - 1);

  // Create BFGS minimizer
  nvMolKit::BfgsBatchMinimizer bfgsMinimizer(dim, nvMolKit::DebugLevel::NONE, true, stream, 
                                               nvMolKit::BfgsBackend::PER_MOLECULE);

  // Filter active molecules
  std::vector<uint8_t> activeHost(numMols);
  cudaMemcpyAsync(activeHost.data(), context.activeThisStage.data(), numMols * sizeof(uint8_t),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  
  std::vector<int> activeMolIndices;
  for (int i = 0; i < numMols; ++i) {
    if (activeHost[i] == 1) {
      activeMolIndices.push_back(i);
    }
  }
  
  // If no active molecules, return early
  if (activeMolIndices.empty()) {
    return;
  }
  
  // Bin active molecules by size
  std::vector<std::vector<int>> bins;
  binMoleculesBySize(context.systemHost.atomStarts, activeMolIndices, bins);
  
  // Copy bins to device
  std::vector<nvMolKit::AsyncDeviceVector<int>> binListsDevice;
  binListsDevice.reserve(5);
  for (int i = 0; i < 5; ++i) {
    binListsDevice.emplace_back(0, stream);  // size 0, will be resized below
    if (!bins[i].empty()) {
      binListsDevice[i].resize(bins[i].size());
      binListsDevice[i].copyFromHost(bins[i]);
    }
  }
  
  // Prepare pointers for launcher
  int binCounts[5];
  const int* binMolIds[5];
  for (int i = 0; i < 5; ++i) {
    binCounts[i] = static_cast<int>(bins[i].size());
    binMolIds[i] = binListsDevice[i].data();
  }
  
  // Allocate convergence status buffer
  nvMolKit::AsyncDeviceVector<uint8_t> convergenceStatus(numMols, stream);
  
  // Initialize minimizer
  bfgsMinimizer.initialize(context.systemHost.atomStarts,
                          context.systemDevice.atomStarts.data(),
                          context.systemDevice.positions.data(),
                          molSystemDevice.grad.data(),
                          molSystemDevice.energyOuts.data(),
                          context.activeThisStage.data());

  // Run minimization loop
  bool needsMore = true;
  while (needsMore) {
    convergenceStatus.zero();
    
    cudaError_t err = launchBfgsMinimizePerMolKernelDG(binCounts,
                                                        binMolIds,
                                                        context.systemDevice.atomStarts.data(),
                                                        bfgsMinimizer.getHessianStarts(),
                                                        maxIters,
                                                        gradTol,
                                                        true,  // scaleGrads
                                                        toEnergyForceContribsDevicePtr(molSystemDevice),
                                                        toBatchedIndicesDevicePtr(molSystemDevice, context.systemDevice.atomStarts.data()),
                                                        context.systemDevice.positions.data(),
                                                        molSystemDevice.grad.data(),
                                                        bfgsMinimizer.getInverseHessian(),
                                                        bfgsMinimizer.getScratchBuffersDevice(),
                                                        molSystemDevice.energyOuts.data(),
                                                        convergenceStatus.data(),
                                                        stream);
    
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("Per-molecule BFGS DG kernel failed: ") + cudaGetErrorString(err));
    }
    
    // Check if all active molecules converged
    if (!repeatUntilConverged) {
      break;
    }
    
    std::vector<uint8_t> convergenceHost(numMols);
    convergenceStatus.copyToHost(convergenceHost);
    cudaStreamSynchronize(stream);
    
    needsMore = false;
    for (int molIdx : activeMolIndices) {
      if (convergenceHost[molIdx] == 0) {
        needsMore = true;
        break;
      }
    }
  }
}

void ETKMinimizeBFGSPerMol(BatchedMolecularSystem3DHost&    molSystemHost,
                           BatchedMolecular3DDeviceBuffers& molSystemDevice,
                           detail::ETKDGContext&            context,
                           const int                        maxIters,
                           const double                     gradTol,
                           const bool                       repeatUntilConverged,
                           cudaStream_t                     stream) {
  // Setup device buffers
  setupDeviceBuffers3D(molSystemHost,
                       molSystemDevice,
                       context.systemHost.positions,
                       static_cast<int>(context.systemHost.atomStarts.size() - 1));

  const int numMols = static_cast<int>(context.systemHost.atomStarts.size() - 1);

  // Create BFGS minimizer (3D for ETK)
  nvMolKit::BfgsBatchMinimizer bfgsMinimizer(3, nvMolKit::DebugLevel::NONE, true, stream,
                                               nvMolKit::BfgsBackend::PER_MOLECULE);

  // Filter active molecules
  std::vector<uint8_t> activeHost(numMols);
  cudaMemcpyAsync(activeHost.data(), context.activeThisStage.data(), numMols * sizeof(uint8_t),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  
  std::vector<int> activeMolIndices;
  for (int i = 0; i < numMols; ++i) {
    if (activeHost[i] == 1) {
      activeMolIndices.push_back(i);
    }
  }
  
  // If no active molecules, return early
  if (activeMolIndices.empty()) {
    return;
  }
  
  // Bin active molecules by size
  std::vector<std::vector<int>> bins;
  binMoleculesBySize(context.systemHost.atomStarts, activeMolIndices, bins);
  
  // Copy bins to device
  std::vector<nvMolKit::AsyncDeviceVector<int>> binListsDevice;
  binListsDevice.reserve(5);
  for (int i = 0; i < 5; ++i) {
    binListsDevice.emplace_back(0, stream);  // size 0, will be resized below
    if (!bins[i].empty()) {
      binListsDevice[i].resize(bins[i].size());
      binListsDevice[i].copyFromHost(bins[i]);
    }
  }
  
  // Prepare pointers for launcher
  int binCounts[5];
  const int* binMolIds[5];
  for (int i = 0; i < 5; ++i) {
    binCounts[i] = static_cast<int>(bins[i].size());
    binMolIds[i] = binListsDevice[i].data();
  }
  
  // Allocate convergence status buffer
  nvMolKit::AsyncDeviceVector<uint8_t> convergenceStatus(numMols, stream);
  
  // Initialize minimizer
  bfgsMinimizer.initialize(context.systemHost.atomStarts,
                          context.systemDevice.atomStarts.data(),
                          context.systemDevice.positions.data(),
                          molSystemDevice.grad.data(),
                          molSystemDevice.energyOuts.data(),
                          context.activeThisStage.data());

  // Run minimization loop
  bool needsMore = true;
  while (needsMore) {
    convergenceStatus.zero();
    
    cudaError_t err = launchBfgsMinimizePerMolKernelETK(binCounts,
                                                         binMolIds,
                                                         context.systemDevice.atomStarts.data(),
                                                         bfgsMinimizer.getHessianStarts(),
                                                         maxIters,
                                                         gradTol,
                                                         true,  // scaleGrads
                                                         toEnergy3DForceContribsDevicePtr(molSystemDevice),
                                                         toBatchedIndices3DDevicePtr(molSystemDevice, context.systemDevice.atomStarts.data()),
                                                         context.systemDevice.positions.data(),
                                                         molSystemDevice.grad.data(),
                                                         bfgsMinimizer.getInverseHessian(),
                                                         bfgsMinimizer.getScratchBuffersDevice(),
                                                         molSystemDevice.energyOuts.data(),
                                                         convergenceStatus.data(),
                                                         stream);
    
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("Per-molecule BFGS ETK kernel failed: ") + cudaGetErrorString(err));
    }
    
    // Check if all active molecules converged
    if (!repeatUntilConverged) {
      break;
    }
    
    std::vector<uint8_t> convergenceHost(numMols);
    convergenceStatus.copyToHost(convergenceHost);
    cudaStreamSynchronize(stream);
    
    needsMore = false;
    for (int molIdx : activeMolIndices) {
      if (convergenceHost[molIdx] == 0) {
        needsMore = true;
        break;
      }
    }
  }
}

}  // namespace nvMolKit::DistGeom
