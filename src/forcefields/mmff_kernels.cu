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

//! Molecules with at most this many atoms keep their coordinates in shared memory; larger molecules read them
//! directly from global memory. For molecules that fit, SINGLE gradients accumulate in per-warp planar (x, y and z
//! arrays) shared buffers that are summed in warp order, and FULL gradients accumulate with global reductions into the
//! molecule's output range, planar and reordered to interleaved at the end. Larger molecules accumulate interleaved
//! with global reductions in both precisions.
constexpr int maxSharedAtomsPerMol = 256;
constexpr int numMolRangeFields    = 12;

//! Copies a molecule's interleaved xyz coordinates (numValues values) into the shared tile as x, y and z arrays of
//! maxSharedAtomsPerMol values each (the layout of PlanarCoords).
template <int BlockSize, typename T>
static __device__ __forceinline__ void stagePlanarCoords(const T* molCoords, const int numValues, T* sharedCoords) {
  for (int i = threadIdx.x; i < numValues; i += BlockSize) {
    const int atom                                             = i / 3;
    sharedCoords[(i - 3 * atom) * maxSharedAtomsPerMol + atom] = molCoords[i];
  }
}

//! Start offsets of field f of the batched indices: atoms, then bond, angle, stretch-bend, out-of-plane, torsion, vdW
//! and electrostatic terms, then distance, position, angle and torsion constraints.
static __device__ __forceinline__ const int* molRangeField(const BatchedIndicesDevicePtr& indices, const int f) {
  switch (f) {
    case 0:
      return indices.atomStarts;
    case 1:
      return indices.bondTermStarts;
    case 2:
      return indices.angleTermStarts;
    case 3:
      return indices.bendTermStarts;
    case 4:
      return indices.oopTermStarts;
    case 5:
      return indices.torsionTermStarts;
    case 6:
      return indices.vdwTermStarts;
    case 7:
      return indices.eleTermStarts;
    case 8:
      return indices.distanceConstraintTermStarts;
    case 9:
      return indices.positionConstraintTermStarts;
    case 10:
      return indices.angleConstraintTermStarts;
    default:
      return indices.torsionConstraintTermStarts;
  }
}

//! Cooperatively caches every per-molecule [start, end) range (ranges[2 f], ranges[2 f + 1] for field f of
//! molRangeField) so the hot term loops read shared memory instead of issuing the same uniform global loads from each
//! warp. Returns whether the molecule is active.
template <bool HasConstraints>
static __device__ __forceinline__ bool loadMolRanges(const BatchedIndicesDevicePtr& systemIndices,
                                                     const uint8_t*                 activeSystemMask,
                                                     const int                      molIdx,
                                                     int*                           ranges) {
  constexpr int numLoads = 2 * (HasConstraints ? numMolRangeFields : numMolRangeFields - 4);
  const int     tid      = threadIdx.x;
  const bool    active   = activeSystemMask == nullptr || activeSystemMask[molIdx] != 0;
  if (tid < numLoads) {
    ranges[tid] = molRangeField(systemIndices, tid >> 1)[molIdx + (tid & 1)];
  }
  __syncthreads();
  return active;
}

//! Resident warps per SM that ptxas is asked to allow (through the minimum blocks of __launch_bounds__), or 0 to leave
//! the register allocation to ptxas. The constrained FULL gradient otherwise grows to a register count that costs a
//! resident block. The unconstrained FULL energy is held to 48 registers so that its wide blocks fit five per SM; the
//! narrow SINGLE gradient trades a few L1-resident spills for 12 blocks per SM. The SINGLE energy is held at full
//! occupancy (32 registers).
template <typename real, bool Gradient, bool HasConstraints, int BlockSize>
__host__ __device__ constexpr int targetWarpsPerSM() {
  if constexpr (std::is_same_v<real, double>) {
    return Gradient ? 32 : (HasConstraints ? 0 : 40);
  } else {
    return Gradient ? (!HasConstraints && BlockSize <= 128 ? 48 : 0) : 64;
  }
}

//! Resident threads and blocks per SM of the architecture being compiled (CUDA occupancy tables). The warp targets
//! above are sm_100 occupancies; ptxas rejects a minimum block count the SM cannot hold, so architectures with fewer
//! resident threads or blocks per SM get the largest minimum they can hold. The host pass uses sm_100's limits.
__host__ __device__ constexpr int maxThreadsPerSMForArch() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return 1024;
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890 || \
                                 __CUDA_ARCH__ == 1010 || __CUDA_ARCH__ == 1100 || __CUDA_ARCH__ >= 1200)
  return 1536;
#else
  return 2048;
#endif
}

__host__ __device__ constexpr int maxBlocksPerSMForArch() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800 || __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870)
  return 16;
#elif defined(__CUDA_ARCH__) && \
  (__CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 1010 || __CUDA_ARCH__ == 1100 || __CUDA_ARCH__ >= 1200)
  return 24;
#else
  return 32;
#endif
}

template <typename real, bool Gradient, bool HasConstraints, int BlockSize>
__host__ __device__ constexpr int minBlocksPerSM() {
  const int target     = targetWarpsPerSM<real, Gradient, HasConstraints, BlockSize>() * kWarpSize / BlockSize;
  const int threadsCap = maxThreadsPerSMForArch() / BlockSize;
  const int blocksCap  = maxBlocksPerSMForArch();
  const int capped     = target < threadsCap ? target : threadsCap;
  return capped < blocksCap ? capped : blocksCap;
}

template <int BlockSize, bool HasConstraints, typename real, typename reduceT, typename Terms, typename storageT>
__global__ void __launch_bounds__(BlockSize, (minBlocksPerSM<real, false, HasConstraints, BlockSize>()))
  combinedEnergiesKernel(const __grid_constant__ Terms                   terms,
                         const __grid_constant__ BatchedIndicesDevicePtr systemIndices,
                         const storageT*                                 coords,
                         storageT*                                       energies,
                         const uint8_t*                                  activeSystemMask) {
  constexpr int numWarps = BlockSize / kWarpSize;
  const int     molIdx   = blockIdx.x;
  const int     tid      = threadIdx.x;

  __shared__ int ranges[2 * numMolRangeFields];
  if (!loadMolRanges<HasConstraints>(systemIndices, activeSystemMask, molIdx, ranges)) {
    if (tid == 0)
      energies[molIdx] = 0.0;
    return;
  }

  __shared__ storageT sharedCoords[3 * maxSharedAtomsPerMol];
  __shared__ reduceT  warpEnergies[numWarps];

  const int       atomStart = ranges[0];
  const int       numValues = 3 * (ranges[1] - atomStart);
  const storageT* molCoords = coords + atomStart * 3;
  real            threadEnergy;
  if (numValues <= 3 * maxSharedAtomsPerMol) {
    stagePlanarCoords<BlockSize>(molCoords, numValues, sharedCoords);
    __syncthreads();
    if constexpr (std::is_same_v<real, float>)
      threadEnergy = fp32::molEnergyBlock<BlockSize, HasConstraints>(
        terms,
        ranges,
        fp32::PlanarCoords<storageT, maxSharedAtomsPerMol>{sharedCoords});
    else
      threadEnergy = fp64::molEnergyBlock<BlockSize, HasConstraints>(
        terms,
        ranges,
        fp64::PlanarCoords<storageT, maxSharedAtomsPerMol>{sharedCoords});
  } else {
    if constexpr (std::is_same_v<real, float>)
      threadEnergy = fp32::molEnergyBlock<BlockSize, HasConstraints>(terms, ranges, molCoords);
    else
      threadEnergy = fp64::molEnergyBlock<BlockSize, HasConstraints>(terms, ranges, molCoords);
  }

  reduceT energy = static_cast<reduceT>(threadEnergy);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    energy += __shfl_down_sync(0xffffffffu, energy, offset);
  }
  if (tid % kWarpSize == 0) {
    warpEnergies[tid / kWarpSize] = energy;
  }
  __syncthreads();
  if (tid == 0) {
    reduceT blockEnergy = warpEnergies[0];
#pragma unroll
    for (int w = 1; w < numWarps; ++w) {
      blockEnergy += warpEnergies[w];
    }
    energies[molIdx] = blockEnergy;
  }
}

template <int BlockSize, bool HasConstraints, typename real, typename Terms, typename storageT>
__global__ void __launch_bounds__(BlockSize, (minBlocksPerSM<real, true, HasConstraints, BlockSize>()))
  combinedGradKernel(const __grid_constant__ Terms                   terms,
                     const __grid_constant__ BatchedIndicesDevicePtr systemIndices,
                     const storageT*                                 coords,
                     storageT*                                       grad,
                     const uint8_t*                                  activeSystemMask) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;

  __shared__ int ranges[2 * numMolRangeFields];
  const bool     active    = loadMolRanges<HasConstraints>(systemIndices, activeSystemMask, molIdx, ranges);
  const int      atomStart = ranges[0];
  const int      numValues = 3 * (ranges[1] - atomStart);
  storageT*      molGrad   = grad + atomStart * 3;

  if (!active) {
    for (int i = tid; i < numValues; i += BlockSize)
      molGrad[i] = 0.0;
    return;
  }

  const storageT* molCoords = coords + atomStart * 3;
  if (numValues <= 3 * maxSharedAtomsPerMol) {
    __shared__ storageT sharedCoords[3 * maxSharedAtomsPerMol];
    const int           numAtoms = numValues / 3;
    if constexpr (std::is_same_v<real, float>) {
      // Each warp accumulates into its own planar buffer (x, y and z arrays of maxSharedAtomsPerMol values), and the
      // buffers are summed in warp order. Every warp evaluates a fixed sequence of term chunks, so each gradient value
      // is the same float sum in the same order on every launch; atomics shared between warps would add the
      // contributions in whatever order the warps reach them, and SINGLE results would vary between launches.
      constexpr int numWarps     = BlockSize / kWarpSize;
      constexpr int warpGradSize = 3 * maxSharedAtomsPerMol;
      __shared__ __align__(16) storageT warpGrads[numWarps * warpGradSize];
      stagePlanarCoords<BlockSize>(molCoords, numValues, sharedCoords);
      for (int i = tid; i < numWarps * warpGradSize / 4; i += BlockSize) {
        reinterpret_cast<float4*>(warpGrads)[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
      __syncthreads();
      fp32::molGradBlock<BlockSize, HasConstraints>(
        terms,
        ranges,
        fp32::PlanarCoords<storageT, maxSharedAtomsPerMol>{sharedCoords},
        fp32::PlanarAtomicGradSink<storageT>{warpGrads + (tid / kWarpSize) * warpGradSize, maxSharedAtomsPerMol});
      __syncthreads();
      for (int atom = tid; atom < numAtoms; atom += BlockSize) {
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          storageT sum = warpGrads[c * maxSharedAtomsPerMol + atom];
#pragma unroll
          for (int w = 1; w < numWarps; ++w) {
            sum += warpGrads[w * warpGradSize + c * maxSharedAtomsPerMol + atom];
          }
          molGrad[3 * atom + c] = sum;
        }
      }
    } else {
      // The molecule's output range doubles as a planar accumulator (x, y and z arrays of numAtoms values each).
      stagePlanarCoords<BlockSize>(molCoords, numValues, sharedCoords);
      for (int i = tid; i < numValues; i += BlockSize) {
        molGrad[i] = 0.0;
      }
      __syncthreads();
      fp64::molGradBlock<BlockSize, HasConstraints>(terms,
                                                    ranges,
                                                    fp64::PlanarCoords<storageT, maxSharedAtomsPerMol>{sharedCoords},
                                                    fp64::PlanarAtomicGradSink<storageT>{molGrad, numAtoms});
      // Reorder the planar sums into the interleaved output in place: every value is read (from L2, where the
      // reductions were performed) before any is overwritten.
      constexpr int maxAtomsPerThread = (maxSharedAtomsPerMol + BlockSize - 1) / BlockSize;
      storageT      sums[maxAtomsPerThread][3];
      __syncthreads();
#pragma unroll
      for (int k = 0; k < maxAtomsPerThread; ++k) {
        const int atom = tid + k * BlockSize;
        if (atom < numAtoms) {
#pragma unroll
          for (int c = 0; c < 3; ++c) {
            sums[k][c] = __ldcg(&molGrad[c * numAtoms + atom]);
          }
        }
      }
      __syncthreads();
#pragma unroll
      for (int k = 0; k < maxAtomsPerThread; ++k) {
        const int atom = tid + k * BlockSize;
        if (atom < numAtoms) {
#pragma unroll
          for (int c = 0; c < 3; ++c) {
            molGrad[3 * atom + c] = sums[k][c];
          }
        }
      }
    }
  } else {
    for (int i = tid; i < numValues; i += BlockSize) {
      molGrad[i] = 0.0;
    }
    __syncthreads();
    if constexpr (std::is_same_v<real, float>)
      fp32::molGradBlock<BlockSize, HasConstraints>(terms, ranges, molCoords, fp32::AtomicGradSink<storageT>{molGrad});
    else
      fp64::molGradBlock<BlockSize, HasConstraints>(terms, ranges, molCoords, fp64::AtomicGradSink<storageT>{molGrad});
  }
}

//! Threads per molecule. The batched backend's molecules (65 atoms and up) are bound by the latency of each warp's
//! chain of term chunks, not by throughput: a single wave of narrow blocks leaves the SMs idle behind their largest
//! molecules, while wide blocks shorten every molecule and leave the block scheduler several waves to balance. Once a
//! launch has enough waves of wide blocks to balance anyway, narrow blocks do the same work with less per-block
//! overhead (fewer partly filled chunks, fewer warps reducing each molecule's gradient), which pays off in FULL.
constexpr int wideBlockSizePerMol     = 256;
constexpr int narrowBlockSizePerMol   = 128;
//! Waves of resident wide blocks from which FULL launches switch to narrow blocks.
constexpr int minWavesForNarrowBlocks = 4;
constexpr int maxCachedDevices        = 64;

//! Blocks of kernel at blockSize threads that the current device holds at once (resident blocks per SM times SMs), or
//! 0 if the device cannot be queried. Cached per device.
template <auto kernel> static int residentBlocks(const int blockSize) {
  static std::array<std::atomic<int>, maxCachedDevices> cache{};
  int                                                   device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  const bool cacheable = device >= 0 && device < maxCachedDevices;
  if (cacheable) {
    if (const int cached = cache[device].load(std::memory_order_relaxed); cached > 0) {
      return cached;
    }
  }
  int blocksPerSM = 0;
  int numSMs      = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel, blockSize, 0) != cudaSuccess ||
      cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  const int resident = blocksPerSM * numSMs;
  if (cacheable) {
    cache[device].store(resident, std::memory_order_relaxed);
  }
  return resident;
}

//! Whether a FULL launch of numMols molecules takes narrow blocks: when the wide kernel would run at least
//! minWavesForNarrowBlocks waves. SINGLE launches always take wide blocks.
template <typename real, auto wideKernel> static bool useNarrowBlocks(const int numMols) {
  if constexpr (std::is_same_v<real, double>) {
    const int resident = residentBlocks<wideKernel>(wideBlockSizePerMol);
    return resident > 0 && numMols >= minWavesForNarrowBlocks * resident;
  } else {
    return false;
  }
}

template <typename real, typename reduceT, typename Terms, typename storageT>
cudaError_t launchBlockPerMolEnergyKernelImpl(int                            numMols,
                                              const Terms&                   terms,
                                              const BatchedIndicesDevicePtr& sytemIndices,
                                              const storageT*                coords,
                                              storageT*                      energies,
                                              bool                           hasConstraints,
                                              cudaStream_t                   stream,
                                              const uint8_t*                 activeSystemMask) {
  if (numMols <= 0) {
    return cudaSuccess;
  }
#define NVMOLKIT_LAUNCH_MMFF_ENERGY(HasConstraints)                                                                    \
  {                                                                                                                    \
    constexpr auto wide = combinedEnergiesKernel<wideBlockSizePerMol, HasConstraints, real, reduceT, Terms, storageT>; \
    constexpr auto narrow =                                                                                            \
      combinedEnergiesKernel<narrowBlockSizePerMol, HasConstraints, real, reduceT, Terms, storageT>;                   \
    if (useNarrowBlocks<real, wide>(numMols)) {                                                                        \
      narrow<<<numMols, narrowBlockSizePerMol, 0, stream>>>(terms, sytemIndices, coords, energies, activeSystemMask);  \
    } else {                                                                                                           \
      wide<<<numMols, wideBlockSizePerMol, 0, stream>>>(terms, sytemIndices, coords, energies, activeSystemMask);      \
    }                                                                                                                  \
  }
  if (hasConstraints)
    NVMOLKIT_LAUNCH_MMFF_ENERGY(true)
  else
    NVMOLKIT_LAUNCH_MMFF_ENERGY(false)
#undef NVMOLKIT_LAUNCH_MMFF_ENERGY
  return cudaGetLastError();
}

template <typename real, typename Terms, typename storageT>
cudaError_t launchBlockPerMolGradKernelImpl(int                            numMols,
                                            const Terms&                   terms,
                                            const BatchedIndicesDevicePtr& sytemIndices,
                                            const storageT*                coords,
                                            storageT*                      grad,
                                            bool                           hasConstraints,
                                            cudaStream_t                   stream,
                                            const uint8_t*                 activeSystemMask) {
  if (numMols <= 0) {
    return cudaSuccess;
  }
#define NVMOLKIT_LAUNCH_MMFF_GRAD(HasConstraints)                                                                 \
  {                                                                                                               \
    constexpr auto wide   = combinedGradKernel<wideBlockSizePerMol, HasConstraints, real, Terms, storageT>;       \
    constexpr auto narrow = combinedGradKernel<narrowBlockSizePerMol, HasConstraints, real, Terms, storageT>;     \
    if (useNarrowBlocks<real, wide>(numMols)) {                                                                   \
      narrow<<<numMols, narrowBlockSizePerMol, 0, stream>>>(terms, sytemIndices, coords, grad, activeSystemMask); \
    } else {                                                                                                      \
      wide<<<numMols, wideBlockSizePerMol, 0, stream>>>(terms, sytemIndices, coords, grad, activeSystemMask);     \
    }                                                                                                             \
  }
  if (hasConstraints)
    NVMOLKIT_LAUNCH_MMFF_GRAD(true)
  else
    NVMOLKIT_LAUNCH_MMFF_GRAD(false)
#undef NVMOLKIT_LAUNCH_MMFF_GRAD
  return cudaGetLastError();
}

#define NVMOLKIT_DEFINE_MMFF_ENERGY_LAUNCHER(real, reduceT, TermsType, storageT)           \
  cudaError_t launchBlockPerMolEnergyKernel(int                            numMols,        \
                                            const TermsType&               terms,          \
                                            const BatchedIndicesDevicePtr& indices,        \
                                            const storageT*                coords,         \
                                            storageT*                      energies,       \
                                            bool                           hasConstraints, \
                                            cudaStream_t                   stream,         \
                                            const uint8_t*                 activeSystemMask) {             \
    return launchBlockPerMolEnergyKernelImpl<real, reduceT>(numMols,                       \
                                                            terms,                         \
                                                            indices,                       \
                                                            coords,                        \
                                                            energies,                      \
                                                            hasConstraints,                \
                                                            stream,                        \
                                                            activeSystemMask);             \
  }

#define NVMOLKIT_DEFINE_MMFF_GRAD_LAUNCHER(real, TermsType, storageT)                    \
  cudaError_t launchBlockPerMolGradKernel(int                            numMols,        \
                                          const TermsType&               terms,          \
                                          const BatchedIndicesDevicePtr& indices,        \
                                          const storageT*                coords,         \
                                          storageT*                      grad,           \
                                          bool                           hasConstraints, \
                                          cudaStream_t                   stream,         \
                                          const uint8_t*                 activeSystemMask) {             \
    return launchBlockPerMolGradKernelImpl<real>(numMols,                                \
                                                 terms,                                  \
                                                 indices,                                \
                                                 coords,                                 \
                                                 grad,                                   \
                                                 hasConstraints,                         \
                                                 stream,                                 \
                                                 activeSystemMask);                      \
  }

NVMOLKIT_DEFINE_MMFF_ENERGY_LAUNCHER(double, double, EnergyForceContribsDevicePtr, double)
NVMOLKIT_DEFINE_MMFF_GRAD_LAUNCHER(double, EnergyForceContribsDevicePtr, double)
NVMOLKIT_DEFINE_MMFF_ENERGY_LAUNCHER(float, float, EnergyForceContribsDevicePtrSingle, float)
NVMOLKIT_DEFINE_MMFF_GRAD_LAUNCHER(float, EnergyForceContribsDevicePtrSingle, float)
#undef NVMOLKIT_DEFINE_MMFF_GRAD_LAUNCHER
#undef NVMOLKIT_DEFINE_MMFF_ENERGY_LAUNCHER
}  // namespace MMFF
}  // namespace nvMolKit
