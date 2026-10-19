// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cub/cub.cuh>
#include <type_traits>

#include "src/forcefields/dist_geom_kernels.h"
#include "src/forcefields/dist_geom_kernels_device_dispatch.cuh"
#include "src/forcefields/kernel_utils.cuh"

using namespace nvMolKit::FFKernelUtils;

namespace nvMolKit {
namespace DistGeom {

template <int dimension>
__global__ void DistViolationEnergyKernel(const int      numDist,
                                          const int*     idx1s,
                                          const int*     idx2s,
                                          const double*  lb2s,
                                          const double*  ub2s,
                                          const double*  weights,
                                          const double*  pos,
                                          double*        energyBuffer,
                                          const int*     energyBufferStarts,
                                          const int*     atomIdxToBatchIdx,
                                          const int*     distTermStarts,
                                          const int*     atomStarts,
                                          const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numDist) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2   = idx2s[idx];
      const double lb2    = lb2s[idx];
      const double ub2    = ub2s[idx];
      const double weight = weights[idx];

      const double energy = distViolationEnergy<dimension>(pos, idx1, idx2, lb2, ub2, weight);
      if (energy > 0.0) {
        const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, distTermStarts);
        energyBuffer[outputIdx] += energy;
      }
    }
  }
}

template <int dimension>
__global__ void DistViolationGradientKernel(const int      numDist,
                                            const int*     idx1s,
                                            const int*     idx2s,
                                            const double*  lb2s,
                                            const double*  ub2s,
                                            const double*  weights,
                                            const double*  pos,
                                            double*        grad,
                                            const int*     atomIdxToBatchIdx,
                                            const int*     atomStarts,
                                            const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numDist) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2   = idx2s[idx];
      const double lb2    = lb2s[idx];
      const double ub2    = ub2s[idx];
      const double weight = weights[idx];

      distViolationGrad<dimension>(pos, idx1, idx2, lb2, ub2, weight, grad);
    }
  }
}

template <int dimension>
__global__ void ChiralViolationEnergyKernel(const int      numChiral,
                                            const int*     idx1s,
                                            const int*     idx2s,
                                            const int*     idx3s,
                                            const int*     idx4s,
                                            const double*  volLower,
                                            const double*  volUpper,
                                            const double   weight,
                                            const double*  pos,
                                            double*        energyBuffer,
                                            const int*     energyBufferStarts,
                                            const int*     atomIdxToBatchIdx,
                                            const int*     chiralTermStarts,
                                            const int*     atomStarts,
                                            const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numChiral) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2      = idx2s[idx];
      const int    idx3      = idx3s[idx];
      const int    idx4      = idx4s[idx];
      const double lb        = volLower[idx];
      const double ub        = volUpper[idx];
      const double energy    = chiralViolationEnergy<dimension>(pos, idx1, idx2, idx3, idx4, lb, ub, weight);
      const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, chiralTermStarts);
      energyBuffer[outputIdx] += energy;
    }
  }
}

template <int dimension>
__global__ void ChiralViolationGradientKernel(const int      numChiral,
                                              const int*     idx1s,
                                              const int*     idx2s,
                                              const int*     idx3s,
                                              const int*     idx4s,
                                              const double*  volLower,
                                              const double*  volUpper,
                                              const double   weight,
                                              const double*  pos,
                                              double*        grad,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     atomStarts,
                                              const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numChiral) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2 = idx2s[idx];
      const int    idx3 = idx3s[idx];
      const int    idx4 = idx4s[idx];
      const double lb   = volLower[idx];
      const double ub   = volUpper[idx];
      chiralViolationGrad<dimension>(pos, idx1, idx2, idx3, idx4, lb, ub, weight, grad);
    }
  }
}

template <int dimension>
__global__ void fourthDimEnergyKernel(const int      numFD,
                                      const int*     idxs,
                                      const double   weight,
                                      const double*  pos,
                                      double*        energyBuffer,
                                      const int*     energyBufferStarts,
                                      const int*     atomIdxToBatchIdx,
                                      const int*     fourthTermStarts,
                                      const int*     atomStarts,
                                      const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numFD) {
    const int idx1     = idxs[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      unsigned  pid       = idx1 * dimension + 3;
      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, fourthTermStarts);
      energyBuffer[outputIdx] += weight * pos[pid] * pos[pid];
    }
  }
}

template <int dimension>
__global__ void fourthDimGradientKernel(const int      numFD,
                                        const int*     idxs,
                                        const double   weight,
                                        const double*  pos,
                                        double*        grad,
                                        const int*     atomIdxToBatchIdx,
                                        const int*     atomStarts,
                                        const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numFD) {
    const int idx1     = idxs[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      int pid = idx1 * dimension + 3;
      grad[pid] += weight * pos[pid];
    }
  }
}

__global__ void TorsionAngleGradientKernel(const int      numTorsion,
                                           const int*     idx1s,
                                           const int*     idx2s,
                                           const int*     idx3s,
                                           const int*     idx4s,
                                           const double*  forceConstants,
                                           const int*     signs,
                                           const double*  pos,
                                           double*        grad,
                                           const int*     atomIdxToBatchIdx,
                                           const int*     atomStarts,
                                           const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTorsion) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      torsionAngleGrad(pos, idx1, idx2s[idx], idx3s[idx], idx4s[idx], &forceConstants[idx * 6], &signs[idx * 6], grad);
    }
  }
}

__global__ void InversionEnergyKernel(const int      numInversion,
                                      const int*     idx1s,
                                      const int*     idx2s,
                                      const int*     idx3s,
                                      const int*     idx4s,
                                      const int*     at2AtomicNum,
                                      const uint8_t* isCBoundToO,
                                      const double*  C0,
                                      const double*  C1,
                                      const double*  C2,
                                      const double*  forceConstants,
                                      const double*  pos,
                                      double*        energyBuffer,
                                      const int*     energyBufferStarts,
                                      const int*     atomIdxToBatchIdx,
                                      const int*     inversionTermStarts,
                                      const int*     atomStarts,
                                      const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numInversion) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int idx2 = idx2s[idx];
      const int idx3 = idx3s[idx];
      const int idx4 = idx4s[idx];

      const double energy =
        inversionEnergy(pos, idx1, idx2, idx3, idx4, C0[idx], C1[idx], C2[idx], forceConstants[idx]);

      // Accumulate energy in the appropriate buffer
      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, inversionTermStarts);
      energyBuffer[outputIdx] += energy;
    }
  }
}

__global__ void InversionGradientKernel(const int      numInversion,
                                        const int*     idx1s,
                                        const int*     idx2s,
                                        const int*     idx3s,
                                        const int*     idx4s,
                                        const int*     at2AtomicNum,
                                        const uint8_t* isCBoundToO,
                                        const double*  C0,
                                        const double*  C1,
                                        const double*  C2,
                                        const double*  forceConstants,
                                        const double*  pos,
                                        double*        grad,
                                        const int*     atomIdxToBatchIdx,
                                        const int*     atomStarts,
                                        const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numInversion) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      inversionGrad(pos,
                    idx1,
                    idx2s[idx],
                    idx3s[idx],
                    idx4s[idx],
                    C0[idx],
                    C1[idx],
                    C2[idx],
                    forceConstants[idx],
                    grad);
    }
  }
}

__global__ void DistanceConstraintGradientKernel(const int      numDist,
                                                 const int*     idx1s,
                                                 const int*     idx2s,
                                                 const double*  minLen,
                                                 const double*  maxLen,
                                                 const double*  forceConstants,
                                                 const double*  pos,
                                                 double*        grad,
                                                 const int*     atomIdxToBatchIdx,
                                                 const int*     atomStarts,
                                                 const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numDist) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      distanceConstraintGrad(pos, idx1, idx2s[idx], minLen[idx], maxLen[idx], forceConstants[idx], grad);
    }
  }
}

__global__ void AngleConstraintEnergyKernel(const int      numAngle,
                                            const int*     idx1s,
                                            const int*     idx2s,
                                            const int*     idx3s,
                                            const double*  minAngle,
                                            const double*  maxAngle,
                                            const double*  pos,
                                            double*        energyBuffer,
                                            const int*     energyBufferStarts,
                                            const int*     atomIdxToBatchIdx,
                                            const int*     angleTermStarts,
                                            const int*     atomStarts,
                                            const uint8_t* activeThisStage,
                                            const double   forceConstant) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numAngle) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2   = idx2s[idx];
      const int    idx3   = idx3s[idx];
      const double minAng = minAngle[idx];
      const double maxAng = maxAngle[idx];

      const double energy = angleConstraintEnergy(pos, idx1, idx2, idx3, minAng, maxAng, forceConstant);

      // Accumulate energy in the appropriate buffer
      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, angleTermStarts);
      energyBuffer[outputIdx] += energy;
    }
  }
}

__global__ void AngleConstraintGradientKernel(const int      numAngle,
                                              const int*     idx1s,
                                              const int*     idx2s,
                                              const int*     idx3s,
                                              const double*  minAngle,
                                              const double*  maxAngle,
                                              const double*  pos,
                                              double*        grad,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     atomStarts,
                                              const uint8_t* activeThisStage,
                                              const double   forceConstant) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numAngle) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      angleConstraintGrad(pos, idx1, idx2s[idx], idx3s[idx], minAngle[idx], maxAngle[idx], forceConstant, grad);
    }
  }
}

cudaError_t launchDistViolationEnergyKernel(const int      numDist,
                                            const int*     idx1,
                                            const int*     idx2,
                                            const double*  lb2,
                                            const double*  ub2,
                                            const double*  weight,
                                            const double*  pos,
                                            double*        energyBuffer,
                                            const int*     energyBufferStarts,
                                            const int*     atomIdxToBatchIdx,
                                            const int*     distTermStarts,
                                            const int*     atomStarts,
                                            const int      dimension,
                                            const uint8_t* activeThisStage,
                                            cudaStream_t   stream) {
  if (numDist == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numDist + blockSize - 1) / blockSize;
  if (dimension == 3) {
    DistViolationEnergyKernel<3><<<numBlocks, blockSize, 0, stream>>>(numDist,
                                                                      idx1,
                                                                      idx2,
                                                                      lb2,
                                                                      ub2,
                                                                      weight,
                                                                      pos,
                                                                      energyBuffer,
                                                                      energyBufferStarts,
                                                                      atomIdxToBatchIdx,
                                                                      distTermStarts,
                                                                      atomStarts,
                                                                      activeThisStage);
  } else if (dimension == 4) {
    DistViolationEnergyKernel<4><<<numBlocks, blockSize, 0, stream>>>(numDist,
                                                                      idx1,
                                                                      idx2,
                                                                      lb2,
                                                                      ub2,
                                                                      weight,
                                                                      pos,
                                                                      energyBuffer,
                                                                      energyBufferStarts,
                                                                      atomIdxToBatchIdx,
                                                                      distTermStarts,
                                                                      atomStarts,
                                                                      activeThisStage);
  }
  return cudaGetLastError();
}

cudaError_t launchDistViolationGradientKernel(const int      numDist,
                                              const int*     idx1,
                                              const int*     idx2,
                                              const double*  lb2,
                                              const double*  ub2,
                                              const double*  weight,
                                              const double*  pos,
                                              double*        grad,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     atomStarts,
                                              const int      dimension,
                                              const uint8_t* activeThisStage,
                                              cudaStream_t   stream) {
  if (numDist == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numDist + blockSize - 1) / blockSize;
  if (dimension == 3) {
    DistViolationGradientKernel<3><<<numBlocks, blockSize, 0, stream>>>(numDist,
                                                                        idx1,
                                                                        idx2,
                                                                        lb2,
                                                                        ub2,
                                                                        weight,
                                                                        pos,
                                                                        grad,
                                                                        atomIdxToBatchIdx,
                                                                        atomStarts,
                                                                        activeThisStage);
  } else if (dimension == 4) {
    DistViolationGradientKernel<4><<<numBlocks, blockSize, 0, stream>>>(numDist,
                                                                        idx1,
                                                                        idx2,
                                                                        lb2,
                                                                        ub2,
                                                                        weight,
                                                                        pos,
                                                                        grad,
                                                                        atomIdxToBatchIdx,
                                                                        atomStarts,
                                                                        activeThisStage);
  }
  return cudaGetLastError();
}

cudaError_t launchChiralViolationEnergyKernel(const int      numChiral,
                                              const int*     idx1,
                                              const int*     idx2,
                                              const int*     idx3,
                                              const int*     idx4,
                                              const double*  volLower,
                                              const double*  volUpper,
                                              double         weight,
                                              const double*  pos,
                                              double*        energyBuffer,
                                              const int*     energyBufferStarts,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     chiralTermStarts,
                                              const int*     atomStarts,
                                              const int      dimension,
                                              const uint8_t* activeThisStage,
                                              cudaStream_t   stream) {
  if (numChiral == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numChiral + blockSize - 1) / blockSize;
  if (dimension == 3) {
    ChiralViolationEnergyKernel<3><<<numBlocks, blockSize, 0, stream>>>(numChiral,
                                                                        idx1,
                                                                        idx2,
                                                                        idx3,
                                                                        idx4,
                                                                        volLower,
                                                                        volUpper,
                                                                        weight,
                                                                        pos,
                                                                        energyBuffer,
                                                                        energyBufferStarts,
                                                                        atomIdxToBatchIdx,
                                                                        chiralTermStarts,
                                                                        atomStarts,
                                                                        activeThisStage);
  } else if (dimension == 4) {
    ChiralViolationEnergyKernel<4><<<numBlocks, blockSize, 0, stream>>>(numChiral,
                                                                        idx1,
                                                                        idx2,
                                                                        idx3,
                                                                        idx4,
                                                                        volLower,
                                                                        volUpper,
                                                                        weight,
                                                                        pos,
                                                                        energyBuffer,
                                                                        energyBufferStarts,
                                                                        atomIdxToBatchIdx,
                                                                        chiralTermStarts,
                                                                        atomStarts,
                                                                        activeThisStage);
  }
  return cudaGetLastError();
}

cudaError_t launchChiralViolationGradientKernel(const int      numChiral,
                                                const int*     idx1,
                                                const int*     idx2,
                                                const int*     idx3,
                                                const int*     idx4,
                                                const double*  volLower,
                                                const double*  volUpper,
                                                double         weight,
                                                const double*  pos,
                                                double*        grad,
                                                const int*     atomIdxToBatchIdx,
                                                const int*     atomStarts,
                                                const int      dimension,
                                                const uint8_t* activeThisStage,
                                                cudaStream_t   stream) {
  if (numChiral == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numChiral + blockSize - 1) / blockSize;
  if (dimension == 3) {
    ChiralViolationGradientKernel<3><<<numBlocks, blockSize, 0, stream>>>(numChiral,
                                                                          idx1,
                                                                          idx2,
                                                                          idx3,
                                                                          idx4,
                                                                          volLower,
                                                                          volUpper,
                                                                          weight,
                                                                          pos,
                                                                          grad,
                                                                          atomIdxToBatchIdx,
                                                                          atomStarts,
                                                                          activeThisStage);
  } else if (dimension == 4) {
    ChiralViolationGradientKernel<4><<<numBlocks, blockSize, 0, stream>>>(numChiral,
                                                                          idx1,
                                                                          idx2,
                                                                          idx3,
                                                                          idx4,
                                                                          volLower,
                                                                          volUpper,
                                                                          weight,
                                                                          pos,
                                                                          grad,
                                                                          atomIdxToBatchIdx,
                                                                          atomStarts,
                                                                          activeThisStage);
  }
  return cudaGetLastError();
}

cudaError_t launchFourthDimEnergyKernel(const int      numFD,
                                        const int*     idx,
                                        double         weight,
                                        const double*  pos,
                                        double*        energyBuffer,
                                        const int*     energyBufferStarts,
                                        const int*     atomIdxToBatchIdx,
                                        const int*     fourthTermStarts,
                                        const int*     atomStarts,
                                        const int      dimension,
                                        const uint8_t* activeThisStage,
                                        cudaStream_t   stream) {
  if (numFD == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numFD + blockSize - 1) / blockSize;
  if (dimension == 3) {
    fourthDimEnergyKernel<3><<<numBlocks, blockSize, 0, stream>>>(numFD,
                                                                  idx,
                                                                  weight,
                                                                  pos,
                                                                  energyBuffer,
                                                                  energyBufferStarts,
                                                                  atomIdxToBatchIdx,
                                                                  fourthTermStarts,
                                                                  atomStarts,
                                                                  activeThisStage);
  } else if (dimension == 4) {
    fourthDimEnergyKernel<4><<<numBlocks, blockSize, 0, stream>>>(numFD,
                                                                  idx,
                                                                  weight,
                                                                  pos,
                                                                  energyBuffer,
                                                                  energyBufferStarts,
                                                                  atomIdxToBatchIdx,
                                                                  fourthTermStarts,
                                                                  atomStarts,
                                                                  activeThisStage);
  }
  return cudaGetLastError();
}

cudaError_t launchFourthDimGradientKernel(const int      numFD,
                                          const int*     idx,
                                          double         weight,
                                          const double*  pos,
                                          double*        grad,
                                          const int*     atomIdxToBatchIdx,
                                          const int*     atomStarts,
                                          const int      dimension,
                                          const uint8_t* activeThisStage,
                                          cudaStream_t   stream) {
  if (numFD == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numFD + blockSize - 1) / blockSize;
  if (dimension == 3) {
    fourthDimGradientKernel<3><<<numBlocks, blockSize, 0, stream>>>(numFD,
                                                                    idx,
                                                                    weight,
                                                                    pos,
                                                                    grad,
                                                                    atomIdxToBatchIdx,
                                                                    atomStarts,
                                                                    activeThisStage);
  } else if (dimension == 4) {
    fourthDimGradientKernel<4><<<numBlocks, blockSize, 0, stream>>>(numFD,
                                                                    idx,
                                                                    weight,
                                                                    pos,
                                                                    grad,
                                                                    atomIdxToBatchIdx,
                                                                    atomStarts,
                                                                    activeThisStage);
  }
  return cudaGetLastError();
}

cudaError_t launchTorsionAngleGradientKernel(const int      numTorsion,
                                             const int*     idx1,
                                             const int*     idx2,
                                             const int*     idx3,
                                             const int*     idx4,
                                             const double*  forceConstant,
                                             const int*     signs,
                                             const double*  pos,
                                             double*        grad,
                                             const int*     atomIdxToBatchIdx,
                                             const int*     atomStarts,
                                             const uint8_t* activeThisStage,
                                             cudaStream_t   stream) {
  if (numTorsion == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numTorsion + blockSize - 1) / blockSize;
  TorsionAngleGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numTorsion,
                                                                  idx1,
                                                                  idx2,
                                                                  idx3,
                                                                  idx4,
                                                                  forceConstant,
                                                                  signs,
                                                                  pos,
                                                                  grad,
                                                                  atomIdxToBatchIdx,
                                                                  atomStarts,
                                                                  activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchInversionEnergyKernel(const int      numInversion,
                                        const int*     idx1,
                                        const int*     idx2,
                                        const int*     idx3,
                                        const int*     idx4,
                                        const int*     at2AtomicNum,
                                        const uint8_t* isCBoundToO,
                                        const double*  C0,
                                        const double*  C1,
                                        const double*  C2,
                                        const double*  forceConstants,
                                        const double*  pos,
                                        double*        energyBuffer,
                                        const int*     energyBufferStarts,
                                        const int*     atomIdxToBatchIdx,
                                        const int*     inversionTermStarts,
                                        const int*     atomStarts,
                                        const uint8_t* activeThisStage,
                                        cudaStream_t   stream) {
  if (numInversion == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numInversion + blockSize - 1) / blockSize;
  InversionEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numInversion,
                                                             idx1,
                                                             idx2,
                                                             idx3,
                                                             idx4,
                                                             at2AtomicNum,
                                                             isCBoundToO,
                                                             C0,
                                                             C1,
                                                             C2,
                                                             forceConstants,
                                                             pos,
                                                             energyBuffer,
                                                             energyBufferStarts,
                                                             atomIdxToBatchIdx,
                                                             inversionTermStarts,
                                                             atomStarts,
                                                             activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchInversionGradientKernel(const int      numInversion,
                                          const int*     idx1,
                                          const int*     idx2,
                                          const int*     idx3,
                                          const int*     idx4,
                                          const int*     at2AtomicNum,
                                          const uint8_t* isCBoundToO,
                                          const double*  C0,
                                          const double*  C1,
                                          const double*  C2,
                                          const double*  forceConstants,
                                          const double*  pos,
                                          double*        grad,
                                          const int*     atomIdxToBatchIdx,
                                          const int*     atomStarts,
                                          const uint8_t* activeThisStage,
                                          cudaStream_t   stream) {
  if (numInversion == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numInversion + blockSize - 1) / blockSize;
  InversionGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numInversion,
                                                               idx1,
                                                               idx2,
                                                               idx3,
                                                               idx4,
                                                               at2AtomicNum,
                                                               isCBoundToO,
                                                               C0,
                                                               C1,
                                                               C2,
                                                               forceConstants,
                                                               pos,
                                                               grad,
                                                               atomIdxToBatchIdx,
                                                               atomStarts,
                                                               activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchDistanceConstraintGradientKernel(const int      numDist,
                                                   const int*     idx1s,
                                                   const int*     idx2s,
                                                   const double*  minLen,
                                                   const double*  maxLen,
                                                   const double*  forceConstants,
                                                   const double*  pos,
                                                   double*        grad,
                                                   const int*     atomIdxToBatchIdx,
                                                   const int*     atomStarts,
                                                   const uint8_t* activeThisStage,
                                                   cudaStream_t   stream) {
  if (numDist == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numDist + blockSize - 1) / blockSize;
  DistanceConstraintGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numDist,
                                                                        idx1s,
                                                                        idx2s,
                                                                        minLen,
                                                                        maxLen,
                                                                        forceConstants,
                                                                        pos,
                                                                        grad,
                                                                        atomIdxToBatchIdx,
                                                                        atomStarts,
                                                                        activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchAngleConstraintEnergyKernel(const int      numAngle,
                                              const int*     idx1,
                                              const int*     idx2,
                                              const int*     idx3,
                                              const double*  minAngle,
                                              const double*  maxAngle,
                                              const double*  pos,
                                              double*        energyBuffer,
                                              const int*     energyBufferStarts,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     angleTermStarts,
                                              const int*     atomStarts,
                                              const uint8_t* activeThisStage,
                                              const double   forceConstant,
                                              cudaStream_t   stream) {
  if (numAngle == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngle + blockSize - 1) / blockSize;
  AngleConstraintEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numAngle,
                                                                   idx1,
                                                                   idx2,
                                                                   idx3,
                                                                   minAngle,
                                                                   maxAngle,
                                                                   pos,
                                                                   energyBuffer,
                                                                   energyBufferStarts,
                                                                   atomIdxToBatchIdx,
                                                                   angleTermStarts,
                                                                   atomStarts,
                                                                   activeThisStage,
                                                                   forceConstant);
  return cudaGetLastError();
}

cudaError_t launchAngleConstraintGradientKernel(const int      numAngle,
                                                const int*     idx1,
                                                const int*     idx2,
                                                const int*     idx3,
                                                const double*  minAngle,
                                                const double*  maxAngle,
                                                const double*  pos,
                                                double*        grad,
                                                const int*     atomIdxToBatchIdx,
                                                const int*     atomStarts,
                                                const uint8_t* activeThisStage,
                                                const double   forceConstant,
                                                cudaStream_t   stream) {
  if (numAngle == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngle + blockSize - 1) / blockSize;
  AngleConstraintGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numAngle,
                                                                     idx1,
                                                                     idx2,
                                                                     idx3,
                                                                     minAngle,
                                                                     maxAngle,
                                                                     pos,
                                                                     grad,
                                                                     atomIdxToBatchIdx,
                                                                     atomStarts,
                                                                     activeThisStage,
                                                                     forceConstant);
  return cudaGetLastError();
}

cudaError_t launchReduceEnergiesKernel(const int      numBlocks,
                                       const double*  energyBuffer,
                                       const int*     energyBufferBlockIdxToBatchIdx,
                                       double*        outs,
                                       const uint8_t* activeThisStage,
                                       cudaStream_t   stream) {
  reduceEnergiesKernel<<<numBlocks, blockSizeEnergyReduction, 0, stream>>>(energyBuffer,
                                                                           energyBufferBlockIdxToBatchIdx,
                                                                           outs,
                                                                           activeThisStage);
  return cudaGetLastError();
}

constexpr int blockSizePerMol = 128;

template <int dimension>
__device__ double molEnergyDGDispatch(const EnergyForceContribsDevicePtr& terms,
                                      const BatchedIndicesDevicePtr&      indices,
                                      const double*                       coords,
                                      int                                 molIdx,
                                      double                              cw,
                                      double                              fw,
                                      int                                 tid) {
  return fp64::molEnergyDG<dimension>(terms, indices, coords, molIdx, cw, fw, tid);
}
template <int dimension>
__device__ float molEnergyDGDispatch(const EnergyForceContribsDevicePtrSingle& terms,
                                     const BatchedIndicesDevicePtr&            indices,
                                     const float*                              coords,
                                     int                                       molIdx,
                                     float                                     cw,
                                     float                                     fw,
                                     int                                       tid) {
  return fp32::molEnergyDG<dimension>(terms, indices, coords, molIdx, cw, fw, tid);
}

template <int dimension>
__device__ void molGradDGDispatch(const EnergyForceContribsDevicePtr& terms,
                                  const BatchedIndicesDevicePtr&      indices,
                                  const double*                       coords,
                                  double*                             grad,
                                  int                                 molIdx,
                                  double                              cw,
                                  double                              fw,
                                  int                                 tid) {
  fp64::molGradDG<dimension>(terms, indices, coords, grad, molIdx, cw, fw, tid);
}
template <int dimension>
__device__ void molGradDGDispatch(const EnergyForceContribsDevicePtrSingle& terms,
                                  const BatchedIndicesDevicePtr&            indices,
                                  const float*                              coords,
                                  float*                                    grad,
                                  int                                       molIdx,
                                  float                                     cw,
                                  float                                     fw,
                                  int                                       tid) {
  fp32::molGradDG<dimension>(terms, indices, coords, grad, molIdx, cw, fw, tid);
}

template <int dimension, typename real, typename reduceT, typename Terms>
__global__ void combinedEnergiesKernel(const Terms*                   terms,
                                       const BatchedIndicesDevicePtr* systemIndices,
                                       const real*                    coords,
                                       real*                          energies,
                                       const real                     chiralWeight,
                                       const real                     fourthDimWeight,
                                       const uint8_t*                 activeThisStage) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;

  if (activeThisStage != nullptr && activeThisStage[molIdx] == 0) {
    if (tid == 0) {
      energies[molIdx] = 0.0;
    }
    return;
  }

  using BlockReduce = cub::BlockReduce<reduceT, blockSizePerMol>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  const int   atomStart = systemIndices->atomStarts[molIdx];
  const real* molCoords = coords + atomStart * dimension;
  const real  threadEnergy =
    molEnergyDGDispatch<dimension>(*terms, *systemIndices, molCoords, molIdx, chiralWeight, fourthDimWeight, tid);
  const reduceT blockEnergy = BlockReduce(tempStorage).Sum(static_cast<reduceT>(threadEnergy));

  if (tid == 0) {
    energies[molIdx] = static_cast<real>(blockEnergy);
  }
}

template <int dimension, typename real, typename Terms>
__global__ void combinedGradKernel(const Terms*                   terms,
                                   const BatchedIndicesDevicePtr* systemIndices,
                                   const real*                    coords,
                                   real*                          grad,
                                   const real                     chiralWeight,
                                   const real                     fourthDimWeight,
                                   const uint8_t*                 activeThisStage) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;

  if (activeThisStage != nullptr && activeThisStage[molIdx] == 0) {
    return;
  }

  const int atomStart = systemIndices->atomStarts[molIdx];
  const int atomEnd   = systemIndices->atomStarts[molIdx + 1];
  const int numAtoms  = atomEnd - atomStart;

  constexpr int   maxAtomSize = 256;
  __shared__ real accumGrad[maxAtomSize * 4];  // Support up to 4D

  const bool useSharedMem = numAtoms * dimension <= maxAtomSize * 4;
  real*      molGradBase  = useSharedMem ? accumGrad : grad + atomStart * dimension;

  for (int i = tid; i < numAtoms * dimension; i += blockSizePerMol) {
    molGradBase[i] = real{0};
  }
  __syncthreads();

  const real* molCoords = coords + atomStart * dimension;
  molGradDGDispatch<dimension>(*terms,
                               *systemIndices,
                               molCoords,
                               molGradBase,
                               molIdx,
                               chiralWeight,
                               fourthDimWeight,
                               tid);
  __syncthreads();

  if (useSharedMem) {
    real* globalGrad = grad + (atomStart * dimension);
    for (int i = tid; i < numAtoms * dimension; i += blockSizePerMol) {
      globalGrad[i] = molGradBase[i];
    }
  }
}

template <typename real, typename reduceT, typename Terms>
cudaError_t launchBlockPerMolEnergyKernelImpl(int                            numMols,
                                              const Terms&                   terms,
                                              const BatchedIndicesDevicePtr& systemIndices,
                                              const real*                    coords,
                                              real*                          energies,
                                              const int                      dimension,
                                              const real                     chiralWeight,
                                              const real                     fourthDimWeight,
                                              const uint8_t*                 activeThisStage,
                                              cudaStream_t                   stream) {
  const AsyncDevicePtr<Terms>                   devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndicesDevicePtr> devSysIdx(systemIndices, stream);
  if (dimension == 3) {
    combinedEnergiesKernel<3, real, reduceT><<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(),
                                                                                      devSysIdx.data(),
                                                                                      coords,
                                                                                      energies,
                                                                                      chiralWeight,
                                                                                      fourthDimWeight,
                                                                                      activeThisStage);
  } else {
    combinedEnergiesKernel<4, real, reduceT><<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(),
                                                                                      devSysIdx.data(),
                                                                                      coords,
                                                                                      energies,
                                                                                      chiralWeight,
                                                                                      fourthDimWeight,
                                                                                      activeThisStage);
  }
  return cudaGetLastError();
}

template <typename real, typename Terms>
cudaError_t launchBlockPerMolGradKernelImpl(int                            numMols,
                                            const Terms&                   terms,
                                            const BatchedIndicesDevicePtr& systemIndices,
                                            const real*                    coords,
                                            real*                          grad,
                                            const int                      dimension,
                                            const real                     chiralWeight,
                                            const real                     fourthDimWeight,
                                            const uint8_t*                 activeThisStage,
                                            cudaStream_t                   stream) {
  const AsyncDevicePtr<Terms>                   devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndicesDevicePtr> devSysIdx(systemIndices, stream);
  if (dimension == 3) {
    combinedGradKernel<3, real><<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(),
                                                                         devSysIdx.data(),
                                                                         coords,
                                                                         grad,
                                                                         chiralWeight,
                                                                         fourthDimWeight,
                                                                         activeThisStage);
  } else {
    combinedGradKernel<4, real><<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(),
                                                                         devSysIdx.data(),
                                                                         coords,
                                                                         grad,
                                                                         chiralWeight,
                                                                         fourthDimWeight,
                                                                         activeThisStage);
  }
  return cudaGetLastError();
}

cudaError_t launchBlockPerMolEnergyKernel(int                                 numMols,
                                          const EnergyForceContribsDevicePtr& terms,
                                          const BatchedIndicesDevicePtr&      systemIndices,
                                          const double*                       coords,
                                          double*                             energies,
                                          const int                           dimension,
                                          const double                        chiralWeight,
                                          const double                        fourthDimWeight,
                                          const uint8_t*                      activeThisStage,
                                          cudaStream_t                        stream) {
  return launchBlockPerMolEnergyKernelImpl<double, double>(numMols,
                                                           terms,
                                                           systemIndices,
                                                           coords,
                                                           energies,
                                                           dimension,
                                                           chiralWeight,
                                                           fourthDimWeight,
                                                           activeThisStage,
                                                           stream);
}

cudaError_t launchBlockPerMolEnergyKernel(int                                       numMols,
                                          const EnergyForceContribsDevicePtrSingle& terms,
                                          const BatchedIndicesDevicePtr&            systemIndices,
                                          const float*                              coords,
                                          float*                                    energies,
                                          const int                                 dimension,
                                          const float                               chiralWeight,
                                          const float                               fourthDimWeight,
                                          const uint8_t*                            activeThisStage,
                                          cudaStream_t                              stream) {
  return launchBlockPerMolEnergyKernelImpl<float, float>(numMols,
                                                         terms,
                                                         systemIndices,
                                                         coords,
                                                         energies,
                                                         dimension,
                                                         chiralWeight,
                                                         fourthDimWeight,
                                                         activeThisStage,
                                                         stream);
}

cudaError_t launchBlockPerMolGradKernel(int                                 numMols,
                                        const EnergyForceContribsDevicePtr& terms,
                                        const BatchedIndicesDevicePtr&      systemIndices,
                                        const double*                       coords,
                                        double*                             grad,
                                        const int                           dimension,
                                        const double                        chiralWeight,
                                        const double                        fourthDimWeight,
                                        const uint8_t*                      activeThisStage,
                                        cudaStream_t                        stream) {
  return launchBlockPerMolGradKernelImpl(numMols,
                                         terms,
                                         systemIndices,
                                         coords,
                                         grad,
                                         dimension,
                                         chiralWeight,
                                         fourthDimWeight,
                                         activeThisStage,
                                         stream);
}

cudaError_t launchBlockPerMolGradKernel(int                                       numMols,
                                        const EnergyForceContribsDevicePtrSingle& terms,
                                        const BatchedIndicesDevicePtr&            systemIndices,
                                        const float*                              coords,
                                        float*                                    grad,
                                        const int                                 dimension,
                                        const float                               chiralWeight,
                                        const float                               fourthDimWeight,
                                        const uint8_t*                            activeThisStage,
                                        cudaStream_t                              stream) {
  return launchBlockPerMolGradKernelImpl(numMols,
                                         terms,
                                         systemIndices,
                                         coords,
                                         grad,
                                         dimension,
                                         chiralWeight,
                                         fourthDimWeight,
                                         activeThisStage,
                                         stream);
}

// ETK (3D) combined kernels
__device__ double molEnergyETKDispatch(const Energy3DForceContribsDevicePtr& terms,
                                       const BatchedIndices3DDevicePtr&      indices,
                                       const double*                         coords,
                                       int                                   molIdx,
                                       int                                   tid) {
  return fp64::molEnergyETK(terms, indices, coords, molIdx, tid);
}
__device__ float molEnergyETKDispatch(const Energy3DForceContribsDevicePtrSingle& terms,
                                      const BatchedIndices3DDevicePtr&            indices,
                                      const float*                                coords,
                                      int                                         molIdx,
                                      int                                         tid) {
  return fp32::molEnergyETK(terms, indices, coords, molIdx, tid);
}

__device__ void molGradETKDispatch(const Energy3DForceContribsDevicePtr& terms,
                                   const BatchedIndices3DDevicePtr&      indices,
                                   const double*                         coords,
                                   double*                               grad,
                                   int                                   molIdx,
                                   int                                   tid) {
  fp64::molGradETK(terms, indices, coords, grad, molIdx, tid);
}
__device__ void molGradETKDispatch(const Energy3DForceContribsDevicePtrSingle& terms,
                                   const BatchedIndices3DDevicePtr&            indices,
                                   const float*                                coords,
                                   float*                                      grad,
                                   int                                         molIdx,
                                   int                                         tid) {
  fp32::molGradETK(terms, indices, coords, grad, molIdx, tid);
}

template <typename real, typename reduceT, typename Terms>
__global__ void combinedEnergiesKernelETK(const Terms*                     terms,
                                          const BatchedIndices3DDevicePtr* systemIndices,
                                          const real*                      coords,
                                          real*                            energies,
                                          const uint8_t*                   activeThisStage) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;

  if (activeThisStage != nullptr && activeThisStage[molIdx] == 0) {
    if (tid == 0) {
      energies[molIdx] = 0.0;
    }
    return;
  }

  using BlockReduce = cub::BlockReduce<reduceT, blockSizePerMol>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  const int     atomStart    = systemIndices->atomStarts[molIdx];
  const real*   molCoords    = coords + atomStart * 4;  // ETK uses 4D
  const real    threadEnergy = molEnergyETKDispatch(*terms, *systemIndices, molCoords, molIdx, tid);
  const reduceT blockEnergy  = BlockReduce(tempStorage).Sum(static_cast<reduceT>(threadEnergy));

  if (tid == 0) {
    energies[molIdx] = static_cast<real>(blockEnergy);
  }
}

template <typename real, typename reduceT, typename Terms>
cudaError_t launchBlockPerMolEnergyKernelETKImpl(int                              numMols,
                                                 const Terms&                     terms,
                                                 const BatchedIndices3DDevicePtr& systemIndices,
                                                 const real*                      coords,
                                                 real*                            energies,
                                                 const uint8_t*                   activeThisStage,
                                                 cudaStream_t                     stream) {
  const AsyncDevicePtr<Terms>                     devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndices3DDevicePtr> devSysIdx(systemIndices, stream);
  combinedEnergiesKernelETK<real, reduceT>
    <<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(), devSysIdx.data(), coords, energies, activeThisStage);
  return cudaGetLastError();
}

// ETK (3D) combined gradient kernel
template <typename real, typename Terms>
__global__ void combinedGradKernelETK(const Terms*                     terms,
                                      const BatchedIndices3DDevicePtr* systemIndices,
                                      const real*                      coords,
                                      real*                            grad,
                                      const uint8_t*                   activeThisStage) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;

  if (activeThisStage != nullptr && activeThisStage[molIdx] == 0) {
    return;
  }

  const int   atomStart = systemIndices->atomStarts[molIdx];
  const real* molCoords = coords + atomStart * 4;  // ETK uses 4D
  real*       molGrad   = grad + atomStart * 4;    // Offset to molecule start for ETK (4D)

  molGradETKDispatch(*terms, *systemIndices, molCoords, molGrad, molIdx, tid);
}

template <typename real, typename Terms>
cudaError_t launchBlockPerMolGradKernelETKImpl(int                              numMols,
                                               const Terms&                     terms,
                                               const BatchedIndices3DDevicePtr& systemIndices,
                                               const real*                      coords,
                                               real*                            grad,
                                               const uint8_t*                   activeThisStage,
                                               cudaStream_t                     stream) {
  const AsyncDevicePtr<Terms>                     devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndices3DDevicePtr> devSysIdx(systemIndices, stream);
  combinedGradKernelETK<real>
    <<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(), devSysIdx.data(), coords, grad, activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchBlockPerMolEnergyKernelETK(int                                   numMols,
                                             const Energy3DForceContribsDevicePtr& terms,
                                             const BatchedIndices3DDevicePtr&      systemIndices,
                                             const double*                         coords,
                                             double*                               energies,
                                             const uint8_t*                        activeThisStage,
                                             cudaStream_t                          stream) {
  return launchBlockPerMolEnergyKernelETKImpl<double, double>(numMols,
                                                              terms,
                                                              systemIndices,
                                                              coords,
                                                              energies,
                                                              activeThisStage,
                                                              stream);
}

cudaError_t launchBlockPerMolEnergyKernelETK(int                                         numMols,
                                             const Energy3DForceContribsDevicePtrSingle& terms,
                                             const BatchedIndices3DDevicePtr&            systemIndices,
                                             const float*                                coords,
                                             float*                                      energies,
                                             const uint8_t*                              activeThisStage,
                                             cudaStream_t                                stream) {
  return launchBlockPerMolEnergyKernelETKImpl<float, float>(numMols,
                                                            terms,
                                                            systemIndices,
                                                            coords,
                                                            energies,
                                                            activeThisStage,
                                                            stream);
}

cudaError_t launchBlockPerMolGradKernelETK(int                                   numMols,
                                           const Energy3DForceContribsDevicePtr& terms,
                                           const BatchedIndices3DDevicePtr&      systemIndices,
                                           const double*                         coords,
                                           double*                               grad,
                                           const uint8_t*                        activeThisStage,
                                           cudaStream_t                          stream) {
  return launchBlockPerMolGradKernelETKImpl(numMols, terms, systemIndices, coords, grad, activeThisStage, stream);
}

cudaError_t launchBlockPerMolGradKernelETK(int                                         numMols,
                                           const Energy3DForceContribsDevicePtrSingle& terms,
                                           const BatchedIndices3DDevicePtr&            systemIndices,
                                           const float*                                coords,
                                           float*                                      grad,
                                           const uint8_t*                              activeThisStage,
                                           cudaStream_t                                stream) {
  return launchBlockPerMolGradKernelETKImpl(numMols, terms, systemIndices, coords, grad, activeThisStage, stream);
}

template <typename reduceT, typename Terms>
__global__ void planarEnergyKernelETK(const Terms*                     terms,
                                      const BatchedIndices3DDevicePtr* indices,
                                      const float*                     coords,
                                      float*                           energies,
                                      const uint8_t*                   activeThisStage) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;
  if (activeThisStage != nullptr && activeThisStage[molIdx] == 0) {
    if (tid == 0)
      energies[molIdx] = 0.0;
    return;
  }
  const int    atomStart = indices->atomStarts[molIdx];
  const float* molCoords = coords + atomStart * 4;
  float        energy    = 0.0f;
  for (int termIdx = indices->improperTorsionTermStarts[molIdx] + tid;
       termIdx < indices->improperTorsionTermStarts[molIdx + 1];
       termIdx += blockSizePerMol) {
    const auto& t = terms->improperTorsionTerms;
    energy += fp32::inversionEnergy(molCoords,
                                    t.idx1[termIdx] - atomStart,
                                    t.idx2[termIdx] - atomStart,
                                    t.idx3[termIdx] - atomStart,
                                    t.idx4[termIdx] - atomStart,
                                    t.C0[termIdx],
                                    t.C1[termIdx],
                                    t.C2[termIdx],
                                    t.forceConstant[termIdx]);
  }
  for (int termIdx = indices->angle13TermStarts[molIdx] + tid; termIdx < indices->angle13TermStarts[molIdx + 1];
       termIdx += blockSizePerMol) {
    const auto& t = terms->angle13Terms;
    energy += fp32::angleConstraintEnergy(molCoords,
                                          t.idx1[termIdx] - atomStart,
                                          t.idx2[termIdx] - atomStart,
                                          t.idx3[termIdx] - atomStart,
                                          t.minAngle[termIdx],
                                          t.maxAngle[termIdx],
                                          10.0f);
  }
  using BlockReduce = cub::BlockReduce<reduceT, blockSizePerMol>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  const reduceT                                sum = BlockReduce(tempStorage).Sum(static_cast<reduceT>(energy));
  if (tid == 0)
    energies[molIdx] = static_cast<float>(sum);
}

cudaError_t launchPlanarEnergyKernelETK(int                                         numMols,
                                        const Energy3DForceContribsDevicePtrSingle& terms,
                                        const BatchedIndices3DDevicePtr&            indices,
                                        const float*                                coords,
                                        float*                                      energies,
                                        const uint8_t*                              activeThisStage,
                                        cudaStream_t                                stream) {
  const AsyncDevicePtr<Energy3DForceContribsDevicePtrSingle> devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndices3DDevicePtr>            devIndices(indices, stream);
  planarEnergyKernelETK<float>
    <<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(), devIndices.data(), coords, energies, activeThisStage);
  return cudaGetLastError();
}

// ----------------------------------------------------------------------------------------------
// Fused single-launch forcefield evaluation.
// One block per molecule computes every selected term family and updates the molecule's output in place, so a call
// is one kernel: no scratch memset, per-family launches or separate reduction, and the term/index pointer structs
// travel as kernel parameters instead of per-call device copies. Molecules too large for the shared coordinate tile
// read coordinates from global memory. The per-molecule driver lives in dist_geom_kernels_device.cuh (fusedMolecule)
// and is instantiated per precision.
// ----------------------------------------------------------------------------------------------
namespace {
// 128 threads (one 4-term bulk chunk each per pass); the per-kernel register cap comes from FusedOccupancy.
constexpr int kFusedBlockSize      = 128;
constexpr int kFusedMinBlocksPerSM = 7;
//! Most 128-thread blocks an SM holds: Turing caps an SM at 1024 threads, later parts at 1536 or 2048.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
constexpr int kFusedMaxBlocksPerSM = 8;
#else
constexpr int kFusedMaxBlocksPerSM = 12;
#endif
constexpr int kFusedSmemAtoms = 128;

template <typename real> struct FusedPrecision;
template <> struct FusedPrecision<float> {
  template <int dimension, bool vecPos = false> using DG = fp32::DGFamily<dimension, vecPos>;
  using ETK                                              = fp32::ETKFamily;
  template <class Family, bool isGrad, bool isSingle, bool overwriteGrad>
  static __device__ __forceinline__ void run(const Family&  fam,
                                             const int*     atomStarts,
                                             const uint8_t* active,
                                             const float*   coords,
                                             float*         out,
                                             float*         sPos,
                                             float*         sBulk,
                                             float*         sRed,
                                             int            tileVals) {
    fp32::fusedMolecule<Family, isGrad, isSingle, overwriteGrad>(fam,
                                                                 atomStarts,
                                                                 active,
                                                                 coords,
                                                                 out,
                                                                 sPos,
                                                                 sBulk,
                                                                 sRed,
                                                                 tileVals);
  }
};
template <> struct FusedPrecision<double> {
  template <int dimension, bool vecPos = false> using DG = fp64::DGFamily<dimension, vecPos>;
  using ETK                                              = fp64::ETKFamily;
  template <class Family, bool isGrad, bool isSingle, bool overwriteGrad>
  static __device__ __forceinline__ void run(const Family&  fam,
                                             const int*     atomStarts,
                                             const uint8_t* active,
                                             const double*  coords,
                                             double*        out,
                                             double*        sPos,
                                             double*        sBulk,
                                             double*        sRed,
                                             int            tileVals) {
    fp64::fusedMolecule<Family, isGrad, isSingle, overwriteGrad>(fam,
                                                                 atomStarts,
                                                                 active,
                                                                 coords,
                                                                 out,
                                                                 sPos,
                                                                 sBulk,
                                                                 sRed,
                                                                 tileVals);
  }
};

//! Residency target per kernel. The fused kernels are latency bound (dependent ranges -> terms -> coordinates round
//! trips per molecule), so more resident molecules win even at the cost of some L1-cached spills:
//!   - single-precision energies: 12 blocks for large DG, 10 for ETK, and a spill-free 7 for small DG3 batches;
//!   - single-precision gradients: 8 blocks, raised to 10 for large DG batches;
//!   - double precision keeps seven: fewer blocks (no spills) and more blocks (heavier spills) both measured slower.
template <typename real, class Family, bool isGrad, bool large> struct FusedOccupancy {
  static constexpr bool kDouble         = std::is_same<real, double>::value;
  static constexpr bool kETK            = Family::kNumRanges == 12;
  static constexpr bool kSmallDG3Energy = !kDouble && !isGrad && !large && !kETK && Family::kDimension == 3;
  static constexpr int  kSingleTarget   = isGrad ? (large && !kETK ? 10 : 8) : (kSmallDG3Energy ? 7 : (kETK ? 10 : 12));
  static constexpr int  kTarget         = kDouble ? kFusedMinBlocksPerSM : kSingleTarget;
  static constexpr int  kMinBlocks      = kTarget < kFusedMaxBlocksPerSM ? kTarget : kFusedMaxBlocksPerSM;
  //! Gradients (and ETK's double-precision energy) stage coordinates in the shared tile; the other energies read them
  //! straight from global memory (L1-cached), skipping the staging barrier, which measured faster.
  static constexpr bool kUseTile        = isGrad || (kDouble && kETK);
};

template <typename real, class Family, bool isGrad, bool isSingle, bool overwriteGrad, bool large>
__global__ void __launch_bounds__(kFusedBlockSize, (FusedOccupancy<real, Family, isGrad, large>::kMinBlocks))
  fusedFFKernel(const Family fam,
                const int* __restrict__ atomStarts,
                const real* __restrict__ coords,
                real* __restrict__ out,
                const uint8_t* __restrict__ active) {
  constexpr bool kUseTile  = FusedOccupancy<real, Family, isGrad, large>::kUseTile;
  constexpr int  kTileVals = kFusedSmemAtoms * Family::kDimension;
  constexpr int  kBulkVals =
    Family::Bulk::kTileStride != Family::kDimension ? kFusedSmemAtoms * Family::Bulk::kTileStride : 1;
  // Kernels that read coordinates from global memory allocate no tile (a larger L1 share).
  __shared__ __align__(16) real sPos[kUseTile ? kTileVals : 1];
  __shared__ real               sBulk[kUseTile ? kBulkVals : 1];
  __shared__ real               sRed[kFusedBlockSize / 32];
  FusedPrecision<real>::template run<Family, isGrad, isSingle, overwriteGrad>(fam,
                                                                              atomStarts,
                                                                              active,
                                                                              coords,
                                                                              out,
                                                                              sPos,
                                                                              sBulk,
                                                                              sRed,
                                                                              kUseTile ? kTileVals : 0);
}

bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

template <typename real, bool isGrad, bool isSingle, bool overwriteGrad, bool large, class Family>
cudaError_t launchFusedImpl(const int      numMols,
                            const int      blockSize,
                            const Family&  fam,
                            const int*     atomStarts,
                            const real*    coords,
                            real*          out,
                            const uint8_t* active,
                            cudaStream_t   stream) {
  fusedFFKernel<real, Family, isGrad, isSingle, overwriteGrad, large>
    <<<numMols, blockSize, 0, stream>>>(fam, atomStarts, coords, out, active);
  return cudaGetLastError();
}

//! Large single-precision batches use the higher-residency kernel variants.
constexpr int kFusedLargeBatch = 4096;

template <typename real, bool isGrad, bool vecPos, int dimension>
cudaError_t launchFusedDGFamily(const int                                  numMols,
                                const int                                  blockSize,
                                const EnergyForceContribsDevicePtrT<real>& terms,
                                const BatchedIndicesDevicePtr&             indices,
                                const real*                                coords,
                                real*                                      out,
                                const real                                 chiralWeight,
                                const real                                 fourthDimWeight,
                                const bool                                 aligned,
                                const uint8_t*                             active,
                                cudaStream_t                               stream) {
  constexpr bool                                                      isSingle      = std::is_same<real, float>::value;
  // SINGLE overwrites the DG gradient (launchBlockPerMolGradKernel contract); FULL accumulates.
  constexpr bool                                                      overwriteGrad = isSingle;
  const typename FusedPrecision<real>::template DG<dimension, vecPos> fam{terms,
                                                                          indices,
                                                                          chiralWeight,
                                                                          fourthDimWeight,
                                                                          aligned};
  if constexpr (isSingle) {
    if (numMols >= kFusedLargeBatch) {
      return launchFusedImpl<real, isGrad, isSingle, overwriteGrad, true>(numMols,
                                                                          blockSize,
                                                                          fam,
                                                                          indices.atomStarts,
                                                                          coords,
                                                                          out,
                                                                          active,
                                                                          stream);
    }
  }
  return launchFusedImpl<real, isGrad, isSingle, overwriteGrad, false>(numMols,
                                                                       blockSize,
                                                                       fam,
                                                                       indices.atomStarts,
                                                                       coords,
                                                                       out,
                                                                       active,
                                                                       stream);
}

template <typename real, bool isGrad>
cudaError_t launchFusedDGImpl(const int                                  numMols,
                              const EnergyForceContribsDevicePtrT<real>& terms,
                              const BatchedIndicesDevicePtr&             indices,
                              const real*                                coords,
                              real*                                      out,
                              const int                                  dimension,
                              const real                                 chiralWeight,
                              const real                                 fourthDimWeight,
                              const uint8_t*                             active,
                              cudaStream_t                               stream) {
  if (numMols <= 0) {
    return cudaSuccess;
  }
  const auto& d = terms.distTerms;
  const bool  aligned =
    aligned16(d.idx1) && aligned16(d.idx2) && aligned16(d.lb2) && aligned16(d.ub2) && aligned16(d.weight);
  // Large FULL energy batches: two-warp CTAs (finer per-molecule scheduling at the same per-thread residency).
  const int threads =
    (!std::is_same<real, float>::value && !isGrad && numMols >= kFusedLargeBatch) ? 64 : kFusedBlockSize;
  if (dimension == 3) {
    return launchFusedDGFamily<real, isGrad, false, 3>(numMols,
                                                       threads,
                                                       terms,
                                                       indices,
                                                       coords,
                                                       out,
                                                       chiralWeight,
                                                       fourthDimWeight,
                                                       aligned,
                                                       active,
                                                       stream);
  }
  if (dimension == 4) {
    if constexpr (!isGrad) {
      if (aligned16(coords)) {
        // DG4 energies read coordinates from global memory: gather each atom with 16-byte loads.
        return launchFusedDGFamily<real, isGrad, true, 4>(numMols,
                                                          threads,
                                                          terms,
                                                          indices,
                                                          coords,
                                                          out,
                                                          chiralWeight,
                                                          fourthDimWeight,
                                                          aligned,
                                                          active,
                                                          stream);
      }
    }
    return launchFusedDGFamily<real, isGrad, false, 4>(numMols,
                                                       threads,
                                                       terms,
                                                       indices,
                                                       coords,
                                                       out,
                                                       chiralWeight,
                                                       fourthDimWeight,
                                                       aligned,
                                                       active,
                                                       stream);
  }
  return cudaErrorInvalidValue;
}

template <typename real, bool isGrad>
cudaError_t launchFusedETKImpl(const int                                    numMols,
                               const Energy3DForceContribsDevicePtrT<real>& terms,
                               const BatchedIndices3DDevicePtr&             indices,
                               const real*                                  coords,
                               real*                                        out,
                               const int                                    termMask,
                               const uint8_t*                               active,
                               cudaStream_t                                 stream) {
  if (numMols <= 0) {
    return cudaSuccess;
  }
  constexpr bool isSingle = std::is_same<real, float>::value;
  const auto&    d        = terms.longRangeDistTerms;
  const bool     aligned =
    aligned16(d.idx1) && aligned16(d.idx2) && aligned16(d.minLen) && aligned16(d.maxLen) && aligned16(d.forceConstant);
  // Large SINGLE gradients: two-warp CTAs (same per-thread residency, finer per-molecule scheduling).
  const int threads = (isSingle && isGrad && numMols >= kFusedLargeBatch) ? 64 : kFusedBlockSize;
  const typename FusedPrecision<real>::ETK fam{terms, indices, termMask, aligned};
  return launchFusedImpl<real, isGrad, isSingle, false, false>(numMols,
                                                               threads,
                                                               fam,
                                                               indices.atomStarts,
                                                               coords,
                                                               out,
                                                               active,
                                                               stream);
}
}  // namespace

cudaError_t launchFusedEnergyKernel(const int                           numMols,
                                    const EnergyForceContribsDevicePtr& terms,
                                    const BatchedIndicesDevicePtr&      systemIndices,
                                    const double*                       coords,
                                    double*                             energies,
                                    const int                           dimension,
                                    const double                        chiralWeight,
                                    const double                        fourthDimWeight,
                                    const uint8_t*                      activeThisStage,
                                    cudaStream_t                        stream) {
  return launchFusedDGImpl<double, false>(numMols,
                                          terms,
                                          systemIndices,
                                          coords,
                                          energies,
                                          dimension,
                                          chiralWeight,
                                          fourthDimWeight,
                                          activeThisStage,
                                          stream);
}

cudaError_t launchFusedGradKernel(const int                           numMols,
                                  const EnergyForceContribsDevicePtr& terms,
                                  const BatchedIndicesDevicePtr&      systemIndices,
                                  const double*                       coords,
                                  double*                             grad,
                                  const int                           dimension,
                                  const double                        chiralWeight,
                                  const double                        fourthDimWeight,
                                  const uint8_t*                      activeThisStage,
                                  cudaStream_t                        stream) {
  return launchFusedDGImpl<double, true>(numMols,
                                         terms,
                                         systemIndices,
                                         coords,
                                         grad,
                                         dimension,
                                         chiralWeight,
                                         fourthDimWeight,
                                         activeThisStage,
                                         stream);
}

cudaError_t launchFusedEnergyKernel(const int                                 numMols,
                                    const EnergyForceContribsDevicePtrSingle& terms,
                                    const BatchedIndicesDevicePtr&            systemIndices,
                                    const float*                              coords,
                                    float*                                    energies,
                                    const int                                 dimension,
                                    const float                               chiralWeight,
                                    const float                               fourthDimWeight,
                                    const uint8_t*                            activeThisStage,
                                    cudaStream_t                              stream) {
  return launchFusedDGImpl<float, false>(numMols,
                                         terms,
                                         systemIndices,
                                         coords,
                                         energies,
                                         dimension,
                                         chiralWeight,
                                         fourthDimWeight,
                                         activeThisStage,
                                         stream);
}

cudaError_t launchFusedGradKernel(const int                                 numMols,
                                  const EnergyForceContribsDevicePtrSingle& terms,
                                  const BatchedIndicesDevicePtr&            systemIndices,
                                  const float*                              coords,
                                  float*                                    grad,
                                  const int                                 dimension,
                                  const float                               chiralWeight,
                                  const float                               fourthDimWeight,
                                  const uint8_t*                            activeThisStage,
                                  cudaStream_t                              stream) {
  return launchFusedDGImpl<float, true>(numMols,
                                        terms,
                                        systemIndices,
                                        coords,
                                        grad,
                                        dimension,
                                        chiralWeight,
                                        fourthDimWeight,
                                        activeThisStage,
                                        stream);
}

cudaError_t launchFusedEnergyKernelETK(const int                             numMols,
                                       const Energy3DForceContribsDevicePtr& terms,
                                       const BatchedIndices3DDevicePtr&      systemIndices,
                                       const double*                         coords,
                                       double*                               energies,
                                       const int                             termMask,
                                       const uint8_t*                        activeThisStage,
                                       cudaStream_t                          stream) {
  return launchFusedETKImpl<double,
                            false>(numMols, terms, systemIndices, coords, energies, termMask, activeThisStage, stream);
}

cudaError_t launchFusedEnergyKernelETK(const int                                   numMols,
                                       const Energy3DForceContribsDevicePtrSingle& terms,
                                       const BatchedIndices3DDevicePtr&            systemIndices,
                                       const float*                                coords,
                                       float*                                      energies,
                                       const int                                   termMask,
                                       const uint8_t*                              activeThisStage,
                                       cudaStream_t                                stream) {
  return launchFusedETKImpl<float,
                            false>(numMols, terms, systemIndices, coords, energies, termMask, activeThisStage, stream);
}

cudaError_t launchFusedGradKernelETK(const int                                   numMols,
                                     const Energy3DForceContribsDevicePtrSingle& terms,
                                     const BatchedIndices3DDevicePtr&            systemIndices,
                                     const float*                                coords,
                                     float*                                      grad,
                                     const uint8_t*                              activeThisStage,
                                     cudaStream_t                                stream) {
  return launchFusedETKImpl<float, true>(numMols,
                                         terms,
                                         systemIndices,
                                         coords,
                                         grad,
                                         kETKAllTermsMask,
                                         activeThisStage,
                                         stream);
}
}  // namespace DistGeom
}  // namespace nvMolKit
