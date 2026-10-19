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

#include <array>
#include <atomic>
#include <cub/cub.cuh>

#include "src/forcefields/kernel_utils.cuh"
#include "src/forcefields/mmff_kernels.h"
#include "src/forcefields/mmff_kernels_device_dispatch.cuh"

using namespace nvMolKit::MMFF::fp64;

using namespace nvMolKit::FFKernelUtils;

__global__ void bondStretchEnergyKernel(const int     numBonds,
                                        const int*    idx1,
                                        const int*    idx2,
                                        const double* r0,
                                        const double* kb,
                                        const double* pos,
                                        double*       energyBuffer,
                                        const int*    energyBufferStarts,
                                        const int*    atomBatchMap,
                                        const int*    termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numBonds) {
    const double energy    = bondStretchEnergy(pos, idx1[idx], idx2[idx], r0[idx], kb[idx]);
    const int    batchIdx  = atomBatchMap[idx1[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void bondStretchGradKernel(const int     numBonds,
                                      const int*    idx1s,
                                      const int*    idx2s,
                                      const double* r0,
                                      const double* kb,
                                      const double* pos,
                                      double*       grad) {
  const int bondIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (bondIdx < numBonds) {
    bondStretchGrad(pos, idx1s[bondIdx], idx2s[bondIdx], r0[bondIdx], kb[bondIdx], grad);
  }
}

__global__ void angleBendEnergyKernel(const int      numAngles,
                                      const int*     idx1s,
                                      const int*     idx2s,
                                      const int*     idx3s,
                                      const double*  theta0,
                                      const double*  ka,
                                      const uint8_t* isLinear,
                                      const double*  pos,
                                      double*        energyBuffer,
                                      const int*     energyBufferStarts,
                                      const int*     atomBatchMap,
                                      const int*     termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numAngles) {
    const double energy = angleBendEnergy(pos, idx1s[idx], idx2s[idx], idx3s[idx], theta0[idx], ka[idx], isLinear[idx]);

    const int batchIdx  = atomBatchMap[idx1s[idx]];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void angleBendGradientKernel(const int      numAngles,
                                        const int*     idx1s,
                                        const int*     idx2s,
                                        const int*     idx3s,
                                        const double*  theta0,
                                        const double*  ka,
                                        const uint8_t* isLinear,
                                        const double*  pos,
                                        double*        grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numAngles) {
    angleBendGrad(idx1s[idx], idx2s[idx], idx3s[idx], theta0[idx], ka[idx], isLinear[idx], pos, grad);
  }
}

__global__ void bendStretchEnergyKernel(const int     numAngles,
                                        const int*    idx1s,
                                        const int*    idx2s,
                                        const int*    idx3s,
                                        const double* theta0,
                                        const double* restLen1,
                                        const double* restLen2,
                                        const double* forceConst1,
                                        const double* forceConst2,
                                        const double* pos,
                                        double*       energyBuffer,
                                        const int*    energyBufferStarts,
                                        const int*    atomBatchMap,
                                        const int*    termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numAngles) {
    const double energy    = bendStretchEnergy(pos,
                                            idx1s[idx],
                                            idx2s[idx],
                                            idx3s[idx],
                                            theta0[idx],
                                            restLen1[idx],
                                            restLen2[idx],
                                            forceConst1[idx],
                                            forceConst2[idx]);
    const int    batchIdx  = atomBatchMap[idx1s[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}
__global__ void bendStretchGradKernel(const int     numAngles,
                                      const int*    idx1s,
                                      const int*    idx2s,
                                      const int*    idx3s,
                                      const double* theta0,
                                      const double* restLen1,
                                      const double* restLen2,
                                      const double* forceConst1,
                                      const double* forceConst2,
                                      const double* pos,
                                      double*       grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numAngles) {
    bendStretchGrad(pos,
                    idx1s[idx],
                    idx2s[idx],
                    idx3s[idx],
                    theta0[idx],
                    restLen1[idx],
                    restLen2[idx],
                    forceConst1[idx],
                    forceConst2[idx],
                    grad);
  }
}

__global__ void oopBendEnergyKernel(const int     numOopBends,
                                    const int*    idx1s,
                                    const int*    idx2s,
                                    const int*    idx3s,
                                    const int*    idx4s,
                                    const double* koop,
                                    const double* pos,
                                    double*       energyBuffer,
                                    const int*    energyBufferStarts,
                                    const int*    atomBatchMap,
                                    const int*    termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numOopBends) {
    // Using I, J, K, L notation

    const double energy    = oopBendEnergy(pos, idx1s[idx], idx2s[idx], idx3s[idx], idx4s[idx], koop[idx]);
    const int    batchIdx  = atomBatchMap[idx1s[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}
__global__ void oopBendGradKernel(const int     numOopBends,
                                  const int*    idx1s,
                                  const int*    idx2s,
                                  const int*    idx3s,
                                  const int*    idx4s,
                                  const double* koop,
                                  const double* pos,
                                  double*       grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numOopBends) {
    rdkit_ports::oopGrad(pos, idx1s[idx], idx2s[idx], idx3s[idx], idx4s[idx], koop[idx], grad);
  }
}

__global__ void torsionEnergyKernel(const int     numTorsions,
                                    const int*    idx1s,
                                    const int*    idx2s,
                                    const int*    idx3s,
                                    const int*    idx4s,
                                    const float*  V1s,
                                    const float*  V2s,
                                    const float*  V3s,
                                    const double* pos,
                                    double*       energyBuffer,
                                    const int*    energyBufferStarts,
                                    const int*    atomBatchMap,
                                    const int*    termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numTorsions) {
    const double energy =
      torsionEnergy(pos, idx1s[idx], idx2s[idx], idx3s[idx], idx4s[idx], V1s[idx], V2s[idx], V3s[idx]);
    const int batchIdx  = atomBatchMap[idx1s[idx]];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}
__global__ void torsionGradKernel(const int     numTorsions,
                                  const int*    idx1s,
                                  const int*    idx2s,
                                  const int*    idx3s,
                                  const int*    idx4s,
                                  const float*  V1s,
                                  const float*  V2s,
                                  const float*  V3s,
                                  const double* pos,
                                  double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTorsions) {
    rdkit_ports::torsionGrad(pos, idx1s[idx], idx2s[idx], idx3s[idx], idx4s[idx], V1s[idx], V2s[idx], V3s[idx], grad);
  }
}

__global__ void vdwEnergyKernel(const int     numVdws,
                                const int*    idxs1,
                                const int*    idx2s,
                                const double* R_ij_stars,
                                const double* wellDepths,
                                const double* pos,
                                double*       energyBuffer,
                                const int*    energyBufferStarts,
                                const int*    atomBatchMap,
                                const int*    termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numVdws) {
    const double energy = vdwEnergy(pos, idxs1[idx], idx2s[idx], R_ij_stars[idx], wellDepths[idx]);

    const int batchIdx  = atomBatchMap[idxs1[idx]];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}
__global__ void vdwGradKernel(const int     numVdws,
                              const int*    idx1,
                              const int*    idx2,
                              const double* R_ij_star,
                              const double* wellDepth,
                              const double* pos,
                              double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numVdws) {
    rdkit_ports::vDWGrad(pos, idx1[idx], idx2[idx], R_ij_star[idx], wellDepth[idx], grad);
  }
}

__global__ void eleEnergyKernel(const int      numEles,
                                const int*     idx1s,
                                const int*     idx2s,
                                const double*  chargeTerms,
                                const uint8_t* dielModels,
                                const uint8_t* is1_4s,
                                const double*  pos,
                                double*        energyBuffer,
                                const int*     energyBufferStarts,
                                const int*     atomBatchMap,
                                const int*     termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numEles) {
    const double energy = eleEnergy(pos, idx1s[idx], idx2s[idx], chargeTerms[idx], dielModels[idx], is1_4s[idx]);

    const int batchIdx  = atomBatchMap[idx1s[idx]];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}
__global__ void eleGradKernel(const int      numEles,
                              const int*     idx1s,
                              const int*     idx2s,
                              const double*  chargeTerms,
                              const uint8_t* dielModels,
                              const uint8_t* is1_4s,
                              const double*  pos,
                              double*        grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numEles) {
    eleGrad(pos, idx1s[idx], idx2s[idx], chargeTerms[idx], dielModels[idx], is1_4s[idx], grad);
  }
}

__global__ void distanceConstraintEnergyKernel(const int     numConstraints,
                                               const int*    idx1s,
                                               const int*    idx2s,
                                               const double* minLens,
                                               const double* maxLens,
                                               const double* forceConstants,
                                               const double* pos,
                                               double*       energyBuffer,
                                               const int*    energyBufferStarts,
                                               const int*    atomBatchMap,
                                               const int*    termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numConstraints) {
    const double energy =
      distanceConstraintEnergy(pos, idx1s[idx], idx2s[idx], minLens[idx], maxLens[idx], forceConstants[idx]);
    const int batchIdx  = atomBatchMap[idx1s[idx]];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void distanceConstraintGradKernel(const int     numConstraints,
                                             const int*    idx1s,
                                             const int*    idx2s,
                                             const double* minLens,
                                             const double* maxLens,
                                             const double* forceConstants,
                                             const double* pos,
                                             double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numConstraints) {
    distanceConstraintGrad(pos, idx1s[idx], idx2s[idx], minLens[idx], maxLens[idx], forceConstants[idx], grad);
  }
}

__global__ void positionConstraintEnergyKernel(const int     numConstraints,
                                               const int*    idxs,
                                               const double* refXs,
                                               const double* refYs,
                                               const double* refZs,
                                               const double* maxDispls,
                                               const double* forceConstants,
                                               const double* pos,
                                               double*       energyBuffer,
                                               const int*    energyBufferStarts,
                                               const int*    atomBatchMap,
                                               const int*    termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numConstraints) {
    const double energy =
      positionConstraintEnergy(pos, idxs[idx], refXs[idx], refYs[idx], refZs[idx], maxDispls[idx], forceConstants[idx]);
    const int batchIdx  = atomBatchMap[idxs[idx]];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void positionConstraintGradKernel(const int     numConstraints,
                                             const int*    idxs,
                                             const double* refXs,
                                             const double* refYs,
                                             const double* refZs,
                                             const double* maxDispls,
                                             const double* forceConstants,
                                             const double* pos,
                                             double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numConstraints) {
    positionConstraintGrad(pos,
                           idxs[idx],
                           refXs[idx],
                           refYs[idx],
                           refZs[idx],
                           maxDispls[idx],
                           forceConstants[idx],
                           grad);
  }
}

__global__ void angleConstraintEnergyKernel(const int     numConstraints,
                                            const int*    idx1s,
                                            const int*    idx2s,
                                            const int*    idx3s,
                                            const double* minAngleDegs,
                                            const double* maxAngleDegs,
                                            const double* forceConstants,
                                            const double* pos,
                                            double*       energyBuffer,
                                            const int*    energyBufferStarts,
                                            const int*    atomBatchMap,
                                            const int*    termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numConstraints) {
    const double energy    = angleConstraintEnergy(pos,
                                                idx1s[idx],
                                                idx2s[idx],
                                                idx3s[idx],
                                                minAngleDegs[idx],
                                                maxAngleDegs[idx],
                                                forceConstants[idx]);
    const int    batchIdx  = atomBatchMap[idx1s[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void angleConstraintGradKernel(const int     numConstraints,
                                          const int*    idx1s,
                                          const int*    idx2s,
                                          const int*    idx3s,
                                          const double* minAngleDegs,
                                          const double* maxAngleDegs,
                                          const double* forceConstants,
                                          const double* pos,
                                          double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numConstraints) {
    angleConstraintGrad(pos,
                        idx1s[idx],
                        idx2s[idx],
                        idx3s[idx],
                        minAngleDegs[idx],
                        maxAngleDegs[idx],
                        forceConstants[idx],
                        grad);
  }
}

__global__ void torsionConstraintEnergyKernel(const int     numConstraints,
                                              const int*    idx1s,
                                              const int*    idx2s,
                                              const int*    idx3s,
                                              const int*    idx4s,
                                              const double* minDihedralDegs,
                                              const double* maxDihedralDegs,
                                              const double* forceConstants,
                                              const double* pos,
                                              double*       energyBuffer,
                                              const int*    energyBufferStarts,
                                              const int*    atomBatchMap,
                                              const int*    termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numConstraints) {
    const double energy    = torsionConstraintEnergy(pos,
                                                  idx1s[idx],
                                                  idx2s[idx],
                                                  idx3s[idx],
                                                  idx4s[idx],
                                                  minDihedralDegs[idx],
                                                  maxDihedralDegs[idx],
                                                  forceConstants[idx]);
    const int    batchIdx  = atomBatchMap[idx1s[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void torsionConstraintGradKernel(const int     numConstraints,
                                            const int*    idx1s,
                                            const int*    idx2s,
                                            const int*    idx3s,
                                            const int*    idx4s,
                                            const double* minDihedralDegs,
                                            const double* maxDihedralDegs,
                                            const double* forceConstants,
                                            const double* pos,
                                            double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numConstraints) {
    torsionConstraintGrad(pos,
                          idx1s[idx],
                          idx2s[idx],
                          idx3s[idx],
                          idx4s[idx],
                          minDihedralDegs[idx],
                          maxDihedralDegs[idx],
                          forceConstants[idx],
                          grad);
  }
}

namespace nvMolKit {
namespace MMFF {

cudaError_t launchBondStretchEnergyKernel(const int     numBonds,
                                          const int*    idx1,
                                          const int*    idx2,
                                          const double* r0,
                                          const double* kb,
                                          const double* pos,
                                          double*       energyBuffer,
                                          const int*    energyBufferStarts,
                                          const int*    atomBatchMap,
                                          const int*    termBatchStarts,
                                          cudaStream_t  stream) {
  assert(numBonds > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numBonds + blockSize - 1) / blockSize;
  bondStretchEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numBonds,
                                                               idx1,
                                                               idx2,
                                                               r0,
                                                               kb,
                                                               pos,
                                                               energyBuffer,
                                                               energyBufferStarts,
                                                               atomBatchMap,
                                                               termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchBondStretchGradientKernel(const int     numBonds,
                                            const int*    idx1,
                                            const int*    idx2,
                                            const double* r0,
                                            const double* kb,
                                            const double* pos,
                                            double*       grad,
                                            cudaStream_t  stream) {
  assert(numBonds > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numBonds + blockSize - 1) / blockSize;
  bondStretchGradKernel<<<numBlocks, blockSize, 0, stream>>>(numBonds, idx1, idx2, r0, kb, pos, grad);

  return cudaGetLastError();
}

cudaError_t launchAngleBendEnergyKernel(const int      numAngles,
                                        const int*     idx1,
                                        const int*     idx2,
                                        const int*     idx3,
                                        const double*  theta0,
                                        const double*  ka,
                                        const uint8_t* isLinear,
                                        const double*  pos,
                                        double*        energyBuffer,
                                        const int*     energyBufferStarts,
                                        const int*     atomBatchMap,
                                        const int*     termBatchStarts,
                                        cudaStream_t   stream) {
  assert(numAngles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngles + blockSize - 1) / blockSize;
  angleBendEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numAngles,
                                                             idx1,
                                                             idx2,
                                                             idx3,
                                                             theta0,
                                                             ka,
                                                             isLinear,
                                                             pos,
                                                             energyBuffer,
                                                             energyBufferStarts,
                                                             atomBatchMap,
                                                             termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchAngleBendGradientKernel(const int      numAngles,
                                          const int*     idx1,
                                          const int*     idx2,
                                          const int*     idx3,
                                          const double*  theta0,
                                          const double*  ka,
                                          const uint8_t* isLinear,
                                          const double*  pos,
                                          double*        grad,
                                          cudaStream_t   stream) {
  assert(numAngles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngles + blockSize - 1) / blockSize;
  angleBendGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numAngles,
                                                               idx1,
                                                               idx2,
                                                               idx3,
                                                               theta0,
                                                               ka,
                                                               isLinear,
                                                               pos,
                                                               grad);

  return cudaGetLastError();
}

cudaError_t launchBendStretchEnergyKernel(const int     numAngles,
                                          const int*    idx1,
                                          const int*    idx2,
                                          const int*    idx3,
                                          const double* theta0,
                                          const double* restLen1,
                                          const double* restLen2,
                                          const double* forceConst1,
                                          const double* forceConst2,
                                          const double* pos,
                                          double*       energyBuffer,
                                          const int*    energyBufferStarts,
                                          const int*    atomBatchMap,
                                          const int*    termBatchStarts,
                                          cudaStream_t  stream) {
  assert(numAngles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngles + blockSize - 1) / blockSize;
  bendStretchEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numAngles,
                                                               idx1,
                                                               idx2,
                                                               idx3,
                                                               theta0,
                                                               restLen1,
                                                               restLen2,
                                                               forceConst1,
                                                               forceConst2,
                                                               pos,
                                                               energyBuffer,
                                                               energyBufferStarts,
                                                               atomBatchMap,
                                                               termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchBendStretchGradientKernel(const int     numAngles,
                                            const int*    idx1,
                                            const int*    idx2,
                                            const int*    idx3,
                                            const double* theta0,
                                            const double* restLen1,
                                            const double* restLen2,
                                            const double* forceConst1,
                                            const double* forceConst2,
                                            const double* pos,
                                            double*       grad,
                                            cudaStream_t  stream) {
  assert(numAngles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngles + blockSize - 1) / blockSize;
  bendStretchGradKernel<<<numBlocks, blockSize, 0, stream>>>(numAngles,
                                                             idx1,
                                                             idx2,
                                                             idx3,
                                                             theta0,
                                                             restLen1,
                                                             restLen2,
                                                             forceConst1,
                                                             forceConst2,
                                                             pos,
                                                             grad);
  return cudaGetLastError();
}

cudaError_t launchOopBendEnergyKernel(const int     numOopBends,
                                      const int*    idx1,
                                      const int*    idx2,
                                      const int*    idx3,
                                      const int*    idx4,
                                      const double* koop,
                                      const double* pos,
                                      double*       energyBuffer,
                                      const int*    energyBufferStarts,
                                      const int*    atomBatchMap,
                                      const int*    termBatchStarts,
                                      cudaStream_t  stream) {
  assert(numOopBends > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numOopBends + blockSize - 1) / blockSize;
  oopBendEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numOopBends,
                                                           idx1,
                                                           idx2,
                                                           idx3,
                                                           idx4,
                                                           koop,
                                                           pos,
                                                           energyBuffer,
                                                           energyBufferStarts,
                                                           atomBatchMap,
                                                           termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchOopBendGradientKernel(const int     numOopBends,
                                        const int*    idx1,
                                        const int*    idx2,
                                        const int*    idx3,
                                        const int*    idx4,
                                        const double* koop,
                                        const double* pos,
                                        double*       grad,
                                        cudaStream_t  stream) {
  assert(numOopBends > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numOopBends + blockSize - 1) / blockSize;
  oopBendGradKernel<<<numBlocks, blockSize, 0, stream>>>(numOopBends, idx1, idx2, idx3, idx4, koop, pos, grad);
  return cudaGetLastError();
}

cudaError_t launchTorsionEnergyKernel(const int     numTorsions,
                                      const int*    idx1,
                                      const int*    idx2,
                                      const int*    idx3,
                                      const int*    idx4,
                                      const float*  V1,
                                      const float*  V2,
                                      const float*  V3,
                                      const double* pos,
                                      double*       energyBuffer,
                                      const int*    energyBufferStarts,
                                      const int*    atomBatchMap,
                                      const int*    termBatchStarts,
                                      cudaStream_t  stream) {
  assert(numTorsions > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numTorsions + blockSize - 1) / blockSize;
  torsionEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numTorsions,
                                                           idx1,
                                                           idx2,
                                                           idx3,
                                                           idx4,
                                                           V1,
                                                           V2,
                                                           V3,
                                                           pos,
                                                           energyBuffer,
                                                           energyBufferStarts,
                                                           atomBatchMap,
                                                           termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchTorsionGradientKernel(const int     numTorsions,
                                        const int*    idx1,
                                        const int*    idx2,
                                        const int*    idx3,
                                        const int*    idx4,
                                        const float*  V1,
                                        const float*  V2,
                                        const float*  V3,
                                        const double* pos,
                                        double*       grad,
                                        cudaStream_t  stream) {
  assert(numTorsions > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numTorsions + blockSize - 1) / blockSize;
  torsionGradKernel<<<numBlocks, blockSize, 0, stream>>>(numTorsions, idx1, idx2, idx3, idx4, V1, V2, V3, pos, grad);
  return cudaGetLastError();
}

cudaError_t launchVdwEnergyKernel(const int     numVdws,
                                  const int*    idx1,
                                  const int*    idx2,
                                  const double* R_ij_star,
                                  const double* wellDepth,
                                  const double* pos,
                                  double*       energyBuffer,
                                  const int*    energyBufferStarts,
                                  const int*    atomBatchMap,
                                  const int*    termBatchStarts,
                                  cudaStream_t  stream) {
  assert(numVdws > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numVdws + blockSize - 1) / blockSize;
  vdwEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numVdws,
                                                       idx1,
                                                       idx2,
                                                       R_ij_star,
                                                       wellDepth,
                                                       pos,
                                                       energyBuffer,
                                                       energyBufferStarts,
                                                       atomBatchMap,
                                                       termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchVdwGradientKernel(const int     numVdws,
                                    const int*    idx1,
                                    const int*    idx2,
                                    const double* R_ij_star,
                                    const double* wellDepth,
                                    const double* pos,
                                    double*       grad,
                                    cudaStream_t  stream) {
  assert(numVdws > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numVdws + blockSize - 1) / blockSize;
  vdwGradKernel<<<numBlocks, blockSize, 0, stream>>>(numVdws, idx1, idx2, R_ij_star, wellDepth, pos, grad);
  return cudaGetLastError();
}

cudaError_t launchEleEnergyKernel(const int      numEles,
                                  const int*     idx1,
                                  const int*     idx2,
                                  const double*  chargeTerm,
                                  const uint8_t* dielModel,
                                  const uint8_t* is1_4,
                                  const double*  pos,
                                  double*        energyBuffer,
                                  const int*     energyBufferStarts,
                                  const int*     atomBatchMap,
                                  const int*     termBatchStarts,
                                  cudaStream_t   stream) {
  assert(numEles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numEles + blockSize - 1) / blockSize;
  eleEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numEles,
                                                       idx1,
                                                       idx2,
                                                       chargeTerm,
                                                       dielModel,
                                                       is1_4,
                                                       pos,
                                                       energyBuffer,
                                                       energyBufferStarts,
                                                       atomBatchMap,
                                                       termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchEleGradientKernel(const int      numEles,
                                    const int*     idx1,
                                    const int*     idx2,
                                    const double*  chargeTerm,
                                    const uint8_t* dielModel,
                                    const uint8_t* is1_4,
                                    const double*  pos,
                                    double*        grad,
                                    cudaStream_t   stream) {
  assert(numEles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numEles + blockSize - 1) / blockSize;
  eleGradKernel<<<numBlocks, blockSize, 0, stream>>>(numEles, idx1, idx2, chargeTerm, dielModel, is1_4, pos, grad);
  return cudaGetLastError();
}

cudaError_t launchDistanceConstraintEnergyKernel(const int     numConstraints,
                                                 const int*    idx1,
                                                 const int*    idx2,
                                                 const double* minLen,
                                                 const double* maxLen,
                                                 const double* forceConstant,
                                                 const double* pos,
                                                 double*       energyBuffer,
                                                 const int*    energyBufferStarts,
                                                 const int*    atomBatchMap,
                                                 const int*    termBatchStarts,
                                                 cudaStream_t  stream) {
  assert(numConstraints > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numConstraints + blockSize - 1) / blockSize;
  distanceConstraintEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numConstraints,
                                                                      idx1,
                                                                      idx2,
                                                                      minLen,
                                                                      maxLen,
                                                                      forceConstant,
                                                                      pos,
                                                                      energyBuffer,
                                                                      energyBufferStarts,
                                                                      atomBatchMap,
                                                                      termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchDistanceConstraintGradientKernel(const int     numConstraints,
                                                   const int*    idx1,
                                                   const int*    idx2,
                                                   const double* minLen,
                                                   const double* maxLen,
                                                   const double* forceConstant,
                                                   const double* pos,
                                                   double*       grad,
                                                   cudaStream_t  stream) {
  assert(numConstraints > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numConstraints + blockSize - 1) / blockSize;
  distanceConstraintGradKernel<<<numBlocks, blockSize, 0, stream>>>(numConstraints,
                                                                    idx1,
                                                                    idx2,
                                                                    minLen,
                                                                    maxLen,
                                                                    forceConstant,
                                                                    pos,
                                                                    grad);
  return cudaGetLastError();
}

cudaError_t launchPositionConstraintEnergyKernel(const int     numConstraints,
                                                 const int*    idx,
                                                 const double* refX,
                                                 const double* refY,
                                                 const double* refZ,
                                                 const double* maxDispl,
                                                 const double* forceConstant,
                                                 const double* pos,
                                                 double*       energyBuffer,
                                                 const int*    energyBufferStarts,
                                                 const int*    atomBatchMap,
                                                 const int*    termBatchStarts,
                                                 cudaStream_t  stream) {
  assert(numConstraints > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numConstraints + blockSize - 1) / blockSize;
  positionConstraintEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numConstraints,
                                                                      idx,
                                                                      refX,
                                                                      refY,
                                                                      refZ,
                                                                      maxDispl,
                                                                      forceConstant,
                                                                      pos,
                                                                      energyBuffer,
                                                                      energyBufferStarts,
                                                                      atomBatchMap,
                                                                      termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchPositionConstraintGradientKernel(const int     numConstraints,
                                                   const int*    idx,
                                                   const double* refX,
                                                   const double* refY,
                                                   const double* refZ,
                                                   const double* maxDispl,
                                                   const double* forceConstant,
                                                   const double* pos,
                                                   double*       grad,
                                                   cudaStream_t  stream) {
  assert(numConstraints > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numConstraints + blockSize - 1) / blockSize;
  positionConstraintGradKernel<<<numBlocks, blockSize, 0, stream>>>(numConstraints,
                                                                    idx,
                                                                    refX,
                                                                    refY,
                                                                    refZ,
                                                                    maxDispl,
                                                                    forceConstant,
                                                                    pos,
                                                                    grad);
  return cudaGetLastError();
}

cudaError_t launchAngleConstraintEnergyKernel(const int     numConstraints,
                                              const int*    idx1,
                                              const int*    idx2,
                                              const int*    idx3,
                                              const double* minAngleDeg,
                                              const double* maxAngleDeg,
                                              const double* forceConstant,
                                              const double* pos,
                                              double*       energyBuffer,
                                              const int*    energyBufferStarts,
                                              const int*    atomBatchMap,
                                              const int*    termBatchStarts,
                                              cudaStream_t  stream) {
  assert(numConstraints > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numConstraints + blockSize - 1) / blockSize;
  angleConstraintEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numConstraints,
                                                                   idx1,
                                                                   idx2,
                                                                   idx3,
                                                                   minAngleDeg,
                                                                   maxAngleDeg,
                                                                   forceConstant,
                                                                   pos,
                                                                   energyBuffer,
                                                                   energyBufferStarts,
                                                                   atomBatchMap,
                                                                   termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchAngleConstraintGradientKernel(const int     numConstraints,
                                                const int*    idx1,
                                                const int*    idx2,
                                                const int*    idx3,
                                                const double* minAngleDeg,
                                                const double* maxAngleDeg,
                                                const double* forceConstant,
                                                const double* pos,
                                                double*       grad,
                                                cudaStream_t  stream) {
  assert(numConstraints > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numConstraints + blockSize - 1) / blockSize;
  angleConstraintGradKernel<<<numBlocks, blockSize, 0, stream>>>(numConstraints,
                                                                 idx1,
                                                                 idx2,
                                                                 idx3,
                                                                 minAngleDeg,
                                                                 maxAngleDeg,
                                                                 forceConstant,
                                                                 pos,
                                                                 grad);
  return cudaGetLastError();
}

cudaError_t launchTorsionConstraintEnergyKernel(const int     numConstraints,
                                                const int*    idx1,
                                                const int*    idx2,
                                                const int*    idx3,
                                                const int*    idx4,
                                                const double* minDihedralDeg,
                                                const double* maxDihedralDeg,
                                                const double* forceConstant,
                                                const double* pos,
                                                double*       energyBuffer,
                                                const int*    energyBufferStarts,
                                                const int*    atomBatchMap,
                                                const int*    termBatchStarts,
                                                cudaStream_t  stream) {
  assert(numConstraints > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numConstraints + blockSize - 1) / blockSize;
  torsionConstraintEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numConstraints,
                                                                     idx1,
                                                                     idx2,
                                                                     idx3,
                                                                     idx4,
                                                                     minDihedralDeg,
                                                                     maxDihedralDeg,
                                                                     forceConstant,
                                                                     pos,
                                                                     energyBuffer,
                                                                     energyBufferStarts,
                                                                     atomBatchMap,
                                                                     termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchTorsionConstraintGradientKernel(const int     numConstraints,
                                                  const int*    idx1,
                                                  const int*    idx2,
                                                  const int*    idx3,
                                                  const int*    idx4,
                                                  const double* minDihedralDeg,
                                                  const double* maxDihedralDeg,
                                                  const double* forceConstant,
                                                  const double* pos,
                                                  double*       grad,
                                                  cudaStream_t  stream) {
  assert(numConstraints > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numConstraints + blockSize - 1) / blockSize;
  torsionConstraintGradKernel<<<numBlocks, blockSize, 0, stream>>>(numConstraints,
                                                                   idx1,
                                                                   idx2,
                                                                   idx3,
                                                                   idx4,
                                                                   minDihedralDeg,
                                                                   maxDihedralDeg,
                                                                   forceConstant,
                                                                   pos,
                                                                   grad);
  return cudaGetLastError();
}

cudaError_t launchReduceEnergiesKernel(const int      numBlocks,
                                       const double*  energyBuffer,
                                       const int*     energyBufferBlockIdxToBatchIdx,
                                       double*        outs,
                                       const uint8_t* activeThisStage,
                                       cudaStream_t   stream) {
  reduceEnergiesKernel<<<numBlocks, nvMolKit::FFKernelUtils::blockSizeEnergyReduction, 0, stream>>>(
    energyBuffer,
    energyBufferBlockIdxToBatchIdx,
    outs,
    activeThisStage);
  return cudaGetLastError();
}

constexpr int blockSizePerMol = 128;

//! Programmatic dependent launch (sm_90+): the block-per-molecule kernels may be launched while the ordering kernel
//! still runs, and wait for its molecule order here. Without a programmatic launch (or before sm_90) this is a no-op.
static __device__ __forceinline__ void waitForMolOrder() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

//! Molecule of this block. Blocks before firstOrdered (the first wave of the launch) take the molecule of their own
//! index and start without waiting for the ordering kernel; the later blocks take the remaining molecules, largest
//! first, from molOrder (or keep the batch order without one).
static __device__ __forceinline__ int blockMolecule(const int* molOrder, const int firstOrdered) {
  const int block = static_cast<int>(blockIdx.x);
  if (molOrder == nullptr || block < firstOrdered) {
    return block;
  }
  waitForMolOrder();
  return molOrder[block - firstOrdered];
}

template <bool HasConstraints, typename real, typename reduceT, typename Terms, typename storageT>
__global__ void combinedEnergiesKernel(const Terms*                   terms,
                                       const BatchedIndicesDevicePtr* systemIndices,
                                       const storageT*                coords,
                                       storageT*                      energies,
                                       const uint8_t*                 activeSystemMask,
                                       const int*                     molOrder,
                                       const int                      firstOrdered) {
  const int molIdx = blockMolecule(molOrder, firstOrdered);
  const int tid    = threadIdx.x;

  if (activeSystemMask != nullptr && activeSystemMask[molIdx] == 0) {
    if (tid == 0)
      energies[molIdx] = 0.0;
    return;
  }

  using BlockReduce = cub::BlockReduce<reduceT, blockSizePerMol>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  const int       atomStart = systemIndices->atomStarts[molIdx];
  const storageT* molCoords = coords + atomStart * 3;
  real            threadEnergy;
  if constexpr (std::is_same_v<real, float>)
    threadEnergy = fp32::molEnergy<blockSizePerMol, HasConstraints>(*terms, *systemIndices, molCoords, molIdx, tid);
  else
    threadEnergy = fp64::molEnergy<blockSizePerMol, HasConstraints>(*terms, *systemIndices, molCoords, molIdx, tid);
  const reduceT blockEnergy = BlockReduce(tempStorage).Sum(static_cast<reduceT>(threadEnergy));

  if (tid == 0) {
    energies[molIdx] = blockEnergy;
  }
}

template <bool HasConstraints, typename real, typename Terms, typename storageT>
__global__ void combinedGradKernel(const Terms*                   terms,
                                   const BatchedIndicesDevicePtr* systemIndices,
                                   const storageT*                coords,
                                   storageT*                      grad,
                                   const uint8_t*                 activeSystemMask,
                                   const int*                     molOrder,
                                   const int                      firstOrdered) {
  const int molIdx = blockMolecule(molOrder, firstOrdered);
  const int tid    = threadIdx.x;

  const int atomStart = systemIndices->atomStarts[molIdx];
  const int atomEnd   = systemIndices->atomStarts[molIdx + 1];
  const int numAtoms  = atomEnd - atomStart;

  if (activeSystemMask != nullptr && activeSystemMask[molIdx] == 0) {
    for (int i = tid; i < numAtoms * 3; i += blockSizePerMol)
      grad[atomStart * 3 + i] = 0.0;
    return;
  }

  constexpr int       maxAtomSize = 256;
  __shared__ storageT accumGrad[maxAtomSize * 3];

  const bool useSharedMem = numAtoms <= maxAtomSize;
  storageT*  molGradBase  = useSharedMem ? accumGrad : grad + atomStart * 3;

  for (int i = tid; i < numAtoms * 3; i += blockSizePerMol) {
    molGradBase[i] = 0.0;
  }
  __syncthreads();

  const storageT* molCoords = coords + atomStart * 3;
  if constexpr (std::is_same_v<real, float>)
    fp32::molGrad<blockSizePerMol, HasConstraints>(*terms, *systemIndices, molCoords, molGradBase, molIdx, tid);
  else
    fp64::molGrad<blockSizePerMol, HasConstraints>(*terms, *systemIndices, molCoords, molGradBase, molIdx, tid);
  __syncthreads();

  if (useSharedMem) {
    storageT* globalGrad = grad + (atomStart * 3);
    for (int i = tid; i < numAtoms * 3; i += blockSizePerMol) {
      globalGrad[i] = molGradBase[i];
    }
  }
}

//! Molecule ordering by decreasing work. A batch's molecules differ in size several times over and every block is bound
//! by the latency of its own molecule, so a large molecule that starts in the last wave sets the kernel's run time. A
//! counting sort over the pair-term count (most of each molecule's work) lets the blocks start the largest molecules
//! first. The sort is a single block, so it is skipped for batches too small to need it or too large for it to stay
//! cheap.
constexpr int orderBlockSize = 1024;
constexpr int orderNumBins   = 1024;
constexpr int orderBinShift  = 4;  //!< log2 of the pair terms per bin.
constexpr int minOrderedMols = 256;
constexpr int maxOrderedMols = 1 << 15;

//! Molecules per thread whose bins and in-bin ranks the sort keeps in registers between its passes (batches of up to
//! orderMolsPerThread * orderBlockSize molecules); larger batches recompute the bins in the scatter pass.
constexpr int orderMolsPerThread = 8;

//! Bin of a molecule, 0 for the most work. Inactive molecules only zero their outputs and go last. All loads are issued
//! together, without a branch on the mask, so the sort waits for a single memory round trip per molecule.
static __device__ __forceinline__ int orderBinOf(const int      molIdx,
                                                 const int*     vdwTermStarts,
                                                 const int*     eleTermStarts,
                                                 const uint8_t* activeSystemMask) {
  const bool inactive = activeSystemMask != nullptr && activeSystemMask[molIdx] == 0;
  const int  pairs =
    (vdwTermStarts[molIdx + 1] - vdwTermStarts[molIdx]) + (eleTermStarts[molIdx + 1] - eleTermStarts[molIdx]);
  const int bin = pairs >> orderBinShift;
  return inactive ? orderNumBins - 1 : orderNumBins - 1 - (bin < orderNumBins - 1 ? bin : orderNumBins - 1);
}

//! Writes molecules [firstMol, numMols) to molOrder[0, numMols - firstMol), largest first (stable within a bin).
__global__ void __launch_bounds__(orderBlockSize) orderMolsByWorkKernel(const int      numMols,
                                                                        const int      firstMol,
                                                                        const int*     vdwTermStarts,
                                                                        const int*     eleTermStarts,
                                                                        const uint8_t* activeSystemMask,
                                                                        int*           molOrder) {
  static_assert(orderNumBins == orderBlockSize);
  constexpr int  kOrderWarp = 32;
  constexpr int  numWarps   = orderBlockSize / kOrderWarp;
  __shared__ int binStarts[orderNumBins];
  __shared__ int warpTotals[numWarps];
  const int      tid    = threadIdx.x;
  const int      lane   = tid % kOrderWarp;
  const int      warp   = tid / kOrderWarp;
  const int      count  = numMols - firstMol;
  const bool     cached = count <= orderMolsPerThread * orderBlockSize;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // Lets the dependent block-per-molecule kernel launch now; its blocks wait for this grid to complete before they read
  // the order.
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
  // The bins of this thread's molecules are loaded before the barrier that publishes the zeroed counts.
  int bins[orderMolsPerThread];
  int ranks[orderMolsPerThread];
  if (cached) {
#pragma unroll
    for (int k = 0; k < orderMolsPerThread; ++k) {
      const int m = firstMol + tid + k * orderBlockSize;
      bins[k]     = m < numMols ? orderBinOf(m, vdwTermStarts, eleTermStarts, activeSystemMask) : -1;
    }
  }
  binStarts[tid] = 0;
  __syncthreads();
  if (cached) {
#pragma unroll
    for (int k = 0; k < orderMolsPerThread; ++k) {
      if (bins[k] >= 0) {
        ranks[k] = atomicAdd(&binStarts[bins[k]], 1);
      }
    }
  } else {
    for (int m = firstMol + tid; m < numMols; m += orderBlockSize) {
      atomicAdd(&binStarts[orderBinOf(m, vdwTermStarts, eleTermStarts, activeSystemMask)], 1);
    }
  }
  __syncthreads();
  // Exclusive scan of the bin counts.
  const int binCount  = binStarts[tid];
  int       inclusive = binCount;
#pragma unroll
  for (int delta = 1; delta < kOrderWarp; delta <<= 1) {
    const int other = __shfl_up_sync(0xffffffffu, inclusive, delta);
    inclusive += lane >= delta ? other : 0;
  }
  if (lane == kOrderWarp - 1) {
    warpTotals[warp] = inclusive;
  }
  __syncthreads();
  if (warp == 0) {
    int total = lane < numWarps ? warpTotals[lane] : 0;
#pragma unroll
    for (int delta = 1; delta < kOrderWarp; delta <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, total, delta);
      total += lane >= delta ? other : 0;
    }
    if (lane < numWarps) {
      warpTotals[lane] = total;
    }
  }
  __syncthreads();
  binStarts[tid] = inclusive - binCount + (warp > 0 ? warpTotals[warp - 1] : 0);
  __syncthreads();
  if (cached) {
#pragma unroll
    for (int k = 0; k < orderMolsPerThread; ++k) {
      if (bins[k] >= 0) {
        molOrder[binStarts[bins[k]] + ranks[k]] = firstMol + tid + k * orderBlockSize;
      }
    }
  } else {
    // Without cached ranks the order within a bin follows the atomics, which only affects the order of molecules of
    // the same size class.
    for (int m = firstMol + tid; m < numMols; m += orderBlockSize) {
      molOrder[atomicAdd(&binStarts[orderBinOf(m, vdwTermStarts, eleTermStarts, activeSystemMask)], 1)] = m;
    }
  }
}

//! Fills molOrder with molecules [firstMol, numMols) largest first and returns it, or returns nullptr when the blocks
//! keep the batch order.
static const int* orderMolsByWork(const int                      numMols,
                                  const int                      firstMol,
                                  const BatchedIndicesDevicePtr& systemIndices,
                                  const uint8_t*                 activeSystemMask,
                                  int*                           molOrder,
                                  cudaStream_t                   stream) {
  if (molOrder == nullptr || firstMol >= numMols || numMols < minOrderedMols || numMols - firstMol > maxOrderedMols) {
    return nullptr;
  }
  orderMolsByWorkKernel<<<1, orderBlockSize, 0, stream>>>(numMols,
                                                          firstMol,
                                                          systemIndices.vdwTermStarts,
                                                          systemIndices.eleTermStarts,
                                                          activeSystemMask,
                                                          molOrder);
  return molOrder;
}

constexpr int maxCachedDevices = 64;

//! Device properties the ordered launches need, queried once per device.
struct OrderLaunchDeviceInfo {
  int computeMajor = 0;
  int numSMs       = 0;
};

static OrderLaunchDeviceInfo orderLaunchDeviceInfo(const int device) {
  static std::array<std::atomic<int>, maxCachedDevices> majors{};
  static std::array<std::atomic<int>, maxCachedDevices> sms{};
  const bool                                            cacheable = device >= 0 && device < maxCachedDevices;
  if (cacheable && sms[device].load(std::memory_order_relaxed) > 0) {
    return {majors[device].load(std::memory_order_relaxed), sms[device].load(std::memory_order_relaxed)};
  }
  OrderLaunchDeviceInfo info;
  if (cudaDeviceGetAttribute(&info.computeMajor, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&info.numSMs, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
    cudaGetLastError();
    return {};
  }
  if (cacheable) {
    majors[device].store(info.computeMajor, std::memory_order_relaxed);
    sms[device].store(info.numSMs, std::memory_order_relaxed);
  }
  return info;
}

//! Resident blocks per SM of kernel at blockSizePerMol threads, cached per device (0 if the query fails).
template <auto kernel> static int residentBlocksPerSM(const int device) {
  static std::array<std::atomic<int>, maxCachedDevices> cache{};
  const bool                                            cacheable = device >= 0 && device < maxCachedDevices;
  if (cacheable) {
    if (const int cached = cache[device].load(std::memory_order_relaxed); cached > 0) {
      return cached;
    }
  }
  int blocksPerSM = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel, blockSizePerMol, 0) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  if (cacheable) {
    cache[device].store(blocksPerSM, std::memory_order_relaxed);
  }
  return blocksPerSM;
}

//! Waves of resident blocks from which a launch lets its first wave skip the molecule order.
constexpr int minWavesUnorderedFirst = 3;

//! Launches a block-per-molecule kernel over numMols molecules with the ordering parameters (molOrder, firstOrdered)
//! appended. In launches of at least minWavesUnorderedFirst waves, the molecules the launch holds resident at once (its
//! first wave) keep the batch order, so those blocks start right away; the ordering kernel sorts the remaining
//! molecules largest first while the first wave runs, and the blocks that start later take them in that order. Smaller
//! launches order every molecule. On compute capability 9.0+ the kernel is a programmatic dependent launch, which may
//! start while the ordering kernel runs (blocks wait for the order in waitForMolOrder) and hides the launch latency
//! between the two. Without an order buffer every block keeps the batch order.
template <auto kernel, typename... Args>
static cudaError_t launchBlockPerMolOrdered(const int                      numMols,
                                            const BatchedIndicesDevicePtr& systemIndices,
                                            const uint8_t*                 activeSystemMask,
                                            int*                           molOrder,
                                            cudaStream_t                   stream,
                                            Args... args) {
  int                   firstOrdered = 0;
  OrderLaunchDeviceInfo info;
  if (molOrder != nullptr && numMols >= minOrderedMols) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
      cudaGetLastError();
    } else {
      info = orderLaunchDeviceInfo(device);
    }
    const int residentBlocks = residentBlocksPerSM<kernel>(device) * info.numSMs;
    firstOrdered             = numMols >= minWavesUnorderedFirst * residentBlocks ? residentBlocks : 0;
  }
  const int* blockMols = orderMolsByWork(numMols, firstOrdered, systemIndices, activeSystemMask, molOrder, stream);
  if (blockMols == nullptr || info.computeMajor < 9) {
    kernel<<<numMols, blockSizePerMol, 0, stream>>>(args..., blockMols, firstOrdered);
    return cudaGetLastError();
  }
  cudaLaunchConfig_t config = {};
  config.gridDim            = dim3(numMols);
  config.blockDim           = dim3(blockSizePerMol);
  config.dynamicSmemBytes   = 0;
  config.stream             = stream;
  cudaLaunchAttribute attribute;
  attribute.id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs                                         = &attribute;
  config.numAttrs                                      = 1;
  return cudaLaunchKernelEx(&config, kernel, args..., blockMols, firstOrdered);
}

template <typename real, typename reduceT, typename Terms, typename storageT>
cudaError_t launchBlockPerMolEnergyKernelImpl(int                            numMols,
                                              const Terms&                   terms,
                                              const BatchedIndicesDevicePtr& sytemIndices,
                                              const storageT*                coords,
                                              storageT*                      energies,
                                              bool                           hasConstraints,
                                              cudaStream_t                   stream,
                                              const uint8_t*                 activeSystemMask,
                                              int*                           molOrder) {
  if (numMols <= 0) {
    return cudaSuccess;
  }
  const AsyncDevicePtr<Terms>                   devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndicesDevicePtr> devSysIdx(sytemIndices, stream);
  if (hasConstraints)
    return launchBlockPerMolOrdered<combinedEnergiesKernel<true, real, reduceT, Terms, storageT>>(numMols,
                                                                                                  sytemIndices,
                                                                                                  activeSystemMask,
                                                                                                  molOrder,
                                                                                                  stream,
                                                                                                  devTerms.data(),
                                                                                                  devSysIdx.data(),
                                                                                                  coords,
                                                                                                  energies,
                                                                                                  activeSystemMask);
  return launchBlockPerMolOrdered<combinedEnergiesKernel<false, real, reduceT, Terms, storageT>>(numMols,
                                                                                                 sytemIndices,
                                                                                                 activeSystemMask,
                                                                                                 molOrder,
                                                                                                 stream,
                                                                                                 devTerms.data(),
                                                                                                 devSysIdx.data(),
                                                                                                 coords,
                                                                                                 energies,
                                                                                                 activeSystemMask);
}

template <typename real, typename Terms, typename storageT>
cudaError_t launchBlockPerMolGradKernelImpl(int                            numMols,
                                            const Terms&                   terms,
                                            const BatchedIndicesDevicePtr& sytemIndices,
                                            const storageT*                coords,
                                            storageT*                      grad,
                                            bool                           hasConstraints,
                                            cudaStream_t                   stream,
                                            const uint8_t*                 activeSystemMask,
                                            int*                           molOrder) {
  if (numMols <= 0) {
    return cudaSuccess;
  }
  const AsyncDevicePtr<Terms>                   devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndicesDevicePtr> devSysIdx(sytemIndices, stream);
  if (hasConstraints)
    return launchBlockPerMolOrdered<combinedGradKernel<true, real, Terms, storageT>>(numMols,
                                                                                     sytemIndices,
                                                                                     activeSystemMask,
                                                                                     molOrder,
                                                                                     stream,
                                                                                     devTerms.data(),
                                                                                     devSysIdx.data(),
                                                                                     coords,
                                                                                     grad,
                                                                                     activeSystemMask);
  return launchBlockPerMolOrdered<combinedGradKernel<false, real, Terms, storageT>>(numMols,
                                                                                    sytemIndices,
                                                                                    activeSystemMask,
                                                                                    molOrder,
                                                                                    stream,
                                                                                    devTerms.data(),
                                                                                    devSysIdx.data(),
                                                                                    coords,
                                                                                    grad,
                                                                                    activeSystemMask);
}

#define NVMOLKIT_DEFINE_MMFF_ENERGY_LAUNCHER(real, reduceT, TermsType, storageT)             \
  cudaError_t launchBlockPerMolEnergyKernel(int                            numMols,          \
                                            const TermsType&               terms,            \
                                            const BatchedIndicesDevicePtr& indices,          \
                                            const storageT*                coords,           \
                                            storageT*                      energies,         \
                                            bool                           hasConstraints,   \
                                            cudaStream_t                   stream,           \
                                            const uint8_t*                 activeSystemMask, \
                                            int*                           molOrder) {                                 \
    return launchBlockPerMolEnergyKernelImpl<real, reduceT>(numMols,                         \
                                                            terms,                           \
                                                            indices,                         \
                                                            coords,                          \
                                                            energies,                        \
                                                            hasConstraints,                  \
                                                            stream,                          \
                                                            activeSystemMask,                \
                                                            molOrder);                       \
  }

#define NVMOLKIT_DEFINE_MMFF_GRAD_LAUNCHER(real, TermsType, storageT)                      \
  cudaError_t launchBlockPerMolGradKernel(int                            numMols,          \
                                          const TermsType&               terms,            \
                                          const BatchedIndicesDevicePtr& indices,          \
                                          const storageT*                coords,           \
                                          storageT*                      grad,             \
                                          bool                           hasConstraints,   \
                                          cudaStream_t                   stream,           \
                                          const uint8_t*                 activeSystemMask, \
                                          int*                           molOrder) {                                 \
    return launchBlockPerMolGradKernelImpl<real>(numMols,                                  \
                                                 terms,                                    \
                                                 indices,                                  \
                                                 coords,                                   \
                                                 grad,                                     \
                                                 hasConstraints,                           \
                                                 stream,                                   \
                                                 activeSystemMask,                         \
                                                 molOrder);                                \
  }

NVMOLKIT_DEFINE_MMFF_ENERGY_LAUNCHER(double, double, EnergyForceContribsDevicePtr, double)
NVMOLKIT_DEFINE_MMFF_GRAD_LAUNCHER(double, EnergyForceContribsDevicePtr, double)
NVMOLKIT_DEFINE_MMFF_ENERGY_LAUNCHER(float, float, EnergyForceContribsDevicePtrSingle, float)
NVMOLKIT_DEFINE_MMFF_GRAD_LAUNCHER(float, EnergyForceContribsDevicePtrSingle, float)
#undef NVMOLKIT_DEFINE_MMFF_GRAD_LAUNCHER
#undef NVMOLKIT_DEFINE_MMFF_ENERGY_LAUNCHER
}  // namespace MMFF
}  // namespace nvMolKit
