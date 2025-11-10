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

#include <cub/cub.cuh>

#include "dist_geom_kernels.h"
#include "dist_geom_kernels_device.cuh"
#include "kernel_utils.cuh"

using namespace nvMolKit::FFKernelUtils;

namespace nvMolKit {
namespace DistGeom {

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
                                          const int      dimension,
                                          const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numDist) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2    = idx2s[idx];
      const double lb2     = lb2s[idx];
      const double ub2     = ub2s[idx];
      const double weight  = weights[idx];
      const int    posIdx1 = idx1 * dimension;
      const int    posIdx2 = idx2 * dimension;

      const double distance2 = distanceSquaredPosIdx(pos, posIdx1, posIdx2, dimension);
      double       val       = 0.0;
      if (distance2 > ub2) {
        val = (distance2 / ub2) - 1.0;
      } else if (distance2 < lb2) {
        val = ((2 * lb2) / (lb2 + distance2)) - 1.0;
      }
      if (val > 0.0) {
        const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, distTermStarts);
        energyBuffer[outputIdx] += weight * val * val;
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
      const int   idx2    = idx2s[idx];
      const float lb2     = lb2s[idx];
      const float ub2     = ub2s[idx];
      const float weight  = weights[idx];
      const int   posIdx1 = idx1 * dimension;
      const int   posIdx2 = idx2 * dimension;

      const float distance2 = distanceSquaredPosIdx<dimension, float>(pos, posIdx1, posIdx2);
      float       preFactor = 0.0;
      if (distance2 > ub2) {
        preFactor = 4.f * ((distance2 / ub2) - 1.0f) / ub2;
      } else if (distance2 < lb2) {
        const float l2d2 = distance2 + lb2;
        preFactor        = 8.f * lb2 * (1.f - 2.0f * lb2 / l2d2) / (l2d2 * l2d2);
      } else {
        return;
      }
      const float dGradx = weight * preFactor * (pos[posIdx1 + 0] - pos[posIdx2 + 0]);
      const float dGrady = weight * preFactor * (pos[posIdx1 + 1] - pos[posIdx2 + 1]);
      const float dGradz = weight * preFactor * (pos[posIdx1 + 2] - pos[posIdx2 + 2]);
      float       dGradw;
      if constexpr (dimension == 4) {
        dGradw = weight * preFactor * (pos[posIdx1 + 3] - pos[posIdx2 + 3]);
      }
      atomicAdd(&grad[posIdx1 + 0], dGradx);
      atomicAdd(&grad[posIdx1 + 1], dGrady);
      atomicAdd(&grad[posIdx1 + 2], dGradz);
      atomicAdd(&grad[posIdx2 + 0], -dGradx);
      atomicAdd(&grad[posIdx2 + 1], -dGrady);
      atomicAdd(&grad[posIdx2 + 2], -dGradz);
      if constexpr (dimension == 4) {
        atomicAdd(&grad[posIdx1 + 3], dGradw);
        atomicAdd(&grad[posIdx2 + 3], -dGradw);
      }
    }
  }
}

__global__ void ChiralViolationEnergyKernel(const int      numChiral,
                                            const int*     idx1s,
                                            const int*     idx2s,
                                            const int*     idx3s,
                                            const int*     idx4s,
                                            const double*  volLower,
                                            const double*  volUpper,
                                            const double*  weights,
                                            const double*  pos,
                                            double*        energyBuffer,
                                            const int*     energyBufferStarts,
                                            const int*     atomIdxToBatchIdx,
                                            const int*     chiralTermStarts,
                                            const int*     atomStarts,
                                            const int      dimension,
                                            const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numChiral) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2    = idx2s[idx];
      const int    idx3    = idx3s[idx];
      const int    idx4    = idx4s[idx];
      const double lb      = volLower[idx];
      const double ub      = volUpper[idx];
      const double weight  = weights[idx];
      const int    posIdx1 = idx1 * dimension;
      const int    posIdx2 = idx2 * dimension;
      const int    posIdx3 = idx3 * dimension;
      const int    posIdx4 = idx4 * dimension;

      double v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z;
      double vol =
        calcChiralVolume(posIdx1, posIdx2, posIdx3, posIdx4, pos, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z);

      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, chiralTermStarts);
      if (vol < lb) {
        energyBuffer[outputIdx] += weight * (vol - lb) * (vol - lb);
      } else if (vol > ub) {
        energyBuffer[outputIdx] += weight * (vol - ub) * (vol - ub);
      }
    }
  }
}

__global__ void ChiralViolationGradientKernel(const int      numChiral,
                                              const int*     idx1s,
                                              const int*     idx2s,
                                              const int*     idx3s,
                                              const int*     idx4s,
                                              const double*  volLower,
                                              const double*  volUpper,
                                              const double*  weights,
                                              const double*  pos,
                                              double*        grad,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     atomStarts,
                                              const int      dimension,
                                              const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numChiral) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2    = idx2s[idx];
      const int    idx3    = idx3s[idx];
      const int    idx4    = idx4s[idx];
      const double lb      = volLower[idx];
      const double ub      = volUpper[idx];
      const double weight  = weights[idx];
      const int    posIdx1 = idx1 * dimension;
      const int    posIdx2 = idx2 * dimension;
      const int    posIdx3 = idx3 * dimension;
      const int    posIdx4 = idx4 * dimension;

      double v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z;
      double vol =
        calcChiralVolume(posIdx1, posIdx2, posIdx3, posIdx4, pos, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z);

      if (vol < lb || vol > ub) {
        double preFactor;
        if (vol < lb) {
          preFactor = weight * (vol - lb);
        } else {  // guaranteed != with outer conditional.
          preFactor = weight * (vol - ub);
        }

        atomicAdd(&grad[posIdx1 + 0], preFactor * (v2y * v3z - v2z * v3y));
        atomicAdd(&grad[posIdx1 + 1], preFactor * (v2z * v3x - v2x * v3z));
        atomicAdd(&grad[posIdx1 + 2], preFactor * (v2x * v3y - v2y * v3x));

        atomicAdd(&grad[posIdx2 + 0], preFactor * (v3y * v1z - v3z * v1y));
        atomicAdd(&grad[posIdx2 + 1], preFactor * (v3z * v1x - v3x * v1z));
        atomicAdd(&grad[posIdx2 + 2], preFactor * (v3x * v1y - v3y * v1x));

        atomicAdd(&grad[posIdx3 + 0], preFactor * (v2z * v1y - v2y * v1z));
        atomicAdd(&grad[posIdx3 + 1], preFactor * (v2x * v1z - v2z * v1x));
        atomicAdd(&grad[posIdx3 + 2], preFactor * (v2y * v1x - v2x * v1y));

        double x1 = pos[posIdx1 + 0];
        double y1 = pos[posIdx1 + 1];
        double z1 = pos[posIdx1 + 2];
        double x2 = pos[posIdx2 + 0];
        double y2 = pos[posIdx2 + 1];
        double z2 = pos[posIdx2 + 2];
        double x3 = pos[posIdx3 + 0];
        double y3 = pos[posIdx3 + 1];
        double z3 = pos[posIdx3 + 2];
        atomicAdd(&grad[posIdx4 + 0], preFactor * (z1 * (y2 - y3) + z2 * (y3 - y1) + z3 * (y1 - y2)));
        atomicAdd(&grad[posIdx4 + 1], preFactor * (x1 * (z2 - z3) + x2 * (z3 - z1) + x3 * (z1 - z2)));
        atomicAdd(&grad[posIdx4 + 2], preFactor * (y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2)));
      }
    }
  }
}

__global__ void fourthDimEnergyKernel(const int      numFD,
                                      const int*     idxs,
                                      const double*  weights,
                                      const double*  pos,
                                      double*        energyBuffer,
                                      const int*     energyBufferStarts,
                                      const int*     atomIdxToBatchIdx,
                                      const int*     fourthTermStarts,
                                      const int*     atomStarts,
                                      const int      dimension,
                                      const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numFD) {
    const int idx1     = idxs[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const double weight    = weights[idx];
      unsigned     pid       = idx1 * dimension + 3;
      const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, fourthTermStarts);
      energyBuffer[outputIdx] += weight * pos[pid] * pos[pid];
    }
  }
}

__global__ void fourthDimGradientKernel(const int      numFD,
                                        const int*     idxs,
                                        const double*  weights,
                                        const double*  pos,
                                        double*        grad,
                                        const int*     atomIdxToBatchIdx,
                                        const int*     atomStarts,
                                        const int      dimension,
                                        const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numFD) {
    const int idx1     = idxs[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const double weight = weights[idx];
      int          pid    = idx1 * dimension + 3;
      grad[pid] += weight * pos[pid];
    }
  }
}

__global__ void TorsionAngleEnergyKernel(const int      numTorsion,
                                         const int*     idx1s,
                                         const int*     idx2s,
                                         const int*     idx3s,
                                         const int*     idx4s,
                                         const double*  forceConstants,
                                         const int*     signs,
                                         const double*  pos,
                                         double*        energyBuffer,
                                         const int*     energyBufferStarts,
                                         const int*     atomIdxToBatchIdx,
                                         const int*     torsionTermStarts,
                                         const int*     atomStarts,
                                         const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTorsion) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int idx2 = idx2s[idx];
      const int idx3 = idx3s[idx];
      const int idx4 = idx4s[idx];

      // Get positions for all four atoms
      const int posIdx1 = idx1 * 4;
      const int posIdx2 = idx2 * 4;
      const int posIdx3 = idx3 * 4;
      const int posIdx4 = idx4 * 4;

      // Calculate cosine of torsion angle
      double cosPhi = calcTorsionCosPhi(pos, posIdx1, posIdx2, posIdx3, posIdx4);

      // Calculate energy using the M6 formula
      const double* fc     = &forceConstants[idx * 6];  // 6 force constants per torsion
      const int*    s      = &signs[idx * 6];           // 6 signs per torsion
      double        energy = calcTorsionEnergyM6(fc, s, cosPhi);

      // Accumulate energy in the appropriate buffer
      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, torsionTermStarts);
      energyBuffer[outputIdx] += energy;
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

      // Get positions for all four atoms
      const int posIdx1 = idx1 * 4;
      const int posIdx2 = idx2 * 4;
      const int posIdx3 = idx3 * 4;
      const int posIdx4 = idx4 * 4;

      // Calculate cosine of inversion angle
      double cosY = calcInversionCosY(pos, posIdx1, posIdx2, posIdx3, posIdx4);

      // Calculate sinY
      const double sinYSq = 1.0 - cosY * cosY;
      const double sinY   = ((sinYSq > 0.0) ? sqrt(sinYSq) : 0.0);

      // Calculate cos(2W)
      const double cos2W = 2.0 * sinY * sinY - 1.0;

      // Calculate energy
      double energy = forceConstants[idx] * (C0[idx] + C1[idx] * sinY + C2[idx] * cos2W);

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
      inversionGrad(pos, idx1, idx2s[idx], idx3s[idx], idx4s[idx], C0[idx], C1[idx], C2[idx], forceConstants[idx], grad);
    }
  }
}

__global__ void DistanceConstraintEnergyKernel(const int      numDist,
                                               const int*     idx1s,
                                               const int*     idx2s,
                                               const double*  minLen,
                                               const double*  maxLen,
                                               const double*  forceConstants,
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
      const int    idx2          = idx2s[idx];
      const double minLen2       = minLen[idx] * minLen[idx];  // Square min length
      const double maxLen2       = maxLen[idx] * maxLen[idx];  // Square max length
      const double forceConstant = forceConstants[idx];
      const int    posIdx1       = idx1 * 4;
      const int    posIdx2       = idx2 * 4;

      // Calculate squared distance - always first 3 dimensions.
      const double distance2 = distanceSquaredPosIdx(pos, posIdx1, posIdx2, 3);

      // Check if distance is outside bounds
      double difference = 0.0;
      if (distance2 < minLen2) {
        difference = minLen[idx] - sqrt(distance2);
      } else if (distance2 > maxLen2) {
        difference = sqrt(distance2) - maxLen[idx];
      } else {
        return;  // Distance within bounds, no energy contribution
      }

      // Calculate energy contribution
      const double energy = 0.5 * forceConstant * difference * difference;

      // Accumulate energy in the appropriate buffer
      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, distTermStarts);
      energyBuffer[outputIdx] += energy;
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

      // Get positions for all three atoms
      const int posIdx1 = idx1 * 4;
      const int posIdx2 = idx2 * 4;
      const int posIdx3 = idx3 * 4;

      // Calculate vectors r1 = p1 - p2 and r2 = p3 - p2
      double r1x = pos[posIdx1 + 0] - pos[posIdx2 + 0];
      double r1y = pos[posIdx1 + 1] - pos[posIdx2 + 1];
      double r1z = pos[posIdx1 + 2] - pos[posIdx2 + 2];

      double r2x = pos[posIdx3 + 0] - pos[posIdx2 + 0];
      double r2y = pos[posIdx3 + 1] - pos[posIdx2 + 1];
      double r2z = pos[posIdx3 + 2] - pos[posIdx2 + 2];

      // Calculate squared lengths and take max with 1.0e-5 as in RDKit
      const double r1LengthSq = fmax(1.0e-5, r1x * r1x + r1y * r1y + r1z * r1z);
      const double r2LengthSq = fmax(1.0e-5, r2x * r2x + r2y * r2y + r2z * r2z);

      // Calculate cosine of angle using dot product
      double cosTheta = dotProduct(r1x, r1y, r1z, r2x, r2y, r2z) / sqrt(r1LengthSq * r2LengthSq);

      // Clamp cosTheta to [-1, 1]
      clipToOne(cosTheta);

      // Convert to degrees using RDKit's RAD2DEG constant
      const double angle = RAD2DEG * acos(cosTheta);

      // Calculate angle term using the separate device function
      const double angleTerm = computeAngleTerm(angle, minAng, maxAng);

      const double energy = forceConstant * angleTerm * angleTerm;

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
  DistViolationEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numDist,
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
                                                                 dimension,
                                                                 activeThisStage);
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
  } else {
    throw std::runtime_error("Unsupported dimension for DistViolationGradientKernel: " + std::to_string(dimension));
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
                                              const double*  weight,
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
  ChiralViolationEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numChiral,
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
                                                                   dimension,
                                                                   activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchChiralViolationGradientKernel(const int      numChiral,
                                                const int*     idx1,
                                                const int*     idx2,
                                                const int*     idx3,
                                                const int*     idx4,
                                                const double*  volLower,
                                                const double*  volUpper,
                                                const double*  weight,
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
  ChiralViolationGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numChiral,
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
                                                                     dimension,
                                                                     activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchFourthDimEnergyKernel(const int      numFD,
                                        const int*     idx,
                                        const double*  weight,
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
  fourthDimEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numFD,
                                                             idx,
                                                             weight,
                                                             pos,
                                                             energyBuffer,
                                                             energyBufferStarts,
                                                             atomIdxToBatchIdx,
                                                             fourthTermStarts,
                                                             atomStarts,
                                                             dimension,
                                                             activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchFourthDimGradientKernel(const int      numFD,
                                          const int*     idx,
                                          const double*  weight,
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
  fourthDimGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numFD,
                                                               idx,
                                                               weight,
                                                               pos,
                                                               grad,
                                                               atomIdxToBatchIdx,
                                                               atomStarts,
                                                               dimension,
                                                               activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchTorsionAngleEnergyKernel(const int      numTorsion,
                                           const int*     idx1,
                                           const int*     idx2,
                                           const int*     idx3,
                                           const int*     idx4,
                                           const double*  forceConstant,
                                           const int*     signs,
                                           const double*  pos,
                                           double*        energyBuffer,
                                           const int*     energyBufferStarts,
                                           const int*     atomIdxToBatchIdx,
                                           const int*     torsionTermStarts,
                                           const int*     atomStarts,
                                           const uint8_t* activeThisStage,
                                           cudaStream_t   stream) {
  if (numTorsion == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numTorsion + blockSize - 1) / blockSize;
  TorsionAngleEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numTorsion,
                                                                idx1,
                                                                idx2,
                                                                idx3,
                                                                idx4,
                                                                forceConstant,
                                                                signs,
                                                                pos,
                                                                energyBuffer,
                                                                energyBufferStarts,
                                                                atomIdxToBatchIdx,
                                                                torsionTermStarts,
                                                                atomStarts,
                                                                activeThisStage);
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

cudaError_t launchDistanceConstraintEnergyKernel(const int      numDist,
                                                 const int*     idx1,
                                                 const int*     idx2,
                                                 const double*  minLen,
                                                 const double*  maxLen,
                                                 const double*  forceConstants,
                                                 const double*  pos,
                                                 double*        energyBuffer,
                                                 const int*     energyBufferStarts,
                                                 const int*     atomIdxToBatchIdx,
                                                 const int*     distTermStarts,
                                                 const int*     atomStarts,
                                                 const uint8_t* activeThisStage,
                                                 cudaStream_t   stream) {
  if (numDist == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numDist + blockSize - 1) / blockSize;
  DistanceConstraintEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numDist,
                                                                      idx1,
                                                                      idx2,
                                                                      minLen,
                                                                      maxLen,
                                                                      forceConstants,
                                                                      pos,
                                                                      energyBuffer,
                                                                      energyBufferStarts,
                                                                      atomIdxToBatchIdx,
                                                                      distTermStarts,
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

__global__ void combinedEnergiesKernel(const EnergyForceContribsDevicePtr* terms,
                                       const BatchedIndicesDevicePtr*      systemIndices,
                                       const double*                       coords,
                                       double*                             energies,
                                       const int                           dimension,
                                       const uint8_t*                      activeThisStage) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;
  const int stride = blockDim.x;

  if (activeThisStage != nullptr && activeThisStage[molIdx] == 0) {
    if (tid == 0) {
      energies[molIdx] = 0.0;
    }
    return;
  }

  using BlockReduce = cub::BlockReduce<double, blockSizePerMol>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  const double threadEnergy = molEnergy(*terms, *systemIndices, coords, molIdx, dimension, tid, stride);
  const double blockEnergy  = BlockReduce(tempStorage).Sum(threadEnergy);

  if (tid == 0) {
    energies[molIdx] = blockEnergy;
  }
}

__global__ void combinedGradKernel(const EnergyForceContribsDevicePtr* terms,
                                   const BatchedIndicesDevicePtr*      systemIndices,
                                   const double*                       coords,
                                   double*                             grad,
                                   const int                           dimension,
                                   const uint8_t*                      activeThisStage) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;
  const int stride = blockDim.x;

  if (activeThisStage != nullptr && activeThisStage[molIdx] == 0) {
    return;
  }

  const int atomStart = systemIndices->atomStarts[molIdx];
  const int atomEnd   = systemIndices->atomStarts[molIdx + 1];
  const int numAtoms  = atomEnd - atomStart;

  constexpr int     maxAtomSize = 256;
  __shared__ double accumGrad[maxAtomSize * 4];  // Support up to 4D

  const bool useSharedMem = numAtoms * dimension <= maxAtomSize * 4;
  double*    molGradBase  = useSharedMem ? accumGrad : grad + atomStart * dimension;

  for (int i = tid; i < numAtoms * dimension; i += stride) {
    molGradBase[i] = 0.0;
  }
  __syncthreads();

  molGrad(*terms, *systemIndices, coords, molGradBase, molIdx, dimension, tid, stride);
  __syncthreads();

  if (useSharedMem) {
    double* globalGrad = grad + (atomStart * dimension);
    for (int i = tid; i < numAtoms * dimension; i += stride) {
      globalGrad[i] = molGradBase[i];
    }
  }
}

cudaError_t launchBlockPerMolEnergyKernel(int                                    numMols,
                                          const EnergyForceContribsDevicePtr&    terms,
                                          const BatchedIndicesDevicePtr&         systemIndices,
                                          const double*                          coords,
                                          double*                                energies,
                                          const int                              dimension,
                                          const uint8_t*                         activeThisStage,
                                          cudaStream_t                           stream) {
  const AsyncDevicePtr<EnergyForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndicesDevicePtr>      devSysIdx(systemIndices, stream);
  combinedEnergiesKernel<<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(),
                                                                   devSysIdx.data(),
                                                                   coords,
                                                                   energies,
                                                                   dimension,
                                                                   activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchBlockPerMolGradKernel(int                                 numMols,
                                        const EnergyForceContribsDevicePtr& terms,
                                        const BatchedIndicesDevicePtr&      systemIndices,
                                        const double*                       coords,
                                        double*                             grad,
                                        const int                           dimension,
                                        const uint8_t*                      activeThisStage,
                                        cudaStream_t                        stream) {
  const AsyncDevicePtr<EnergyForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndicesDevicePtr>      devSysIdx(systemIndices, stream);
  combinedGradKernel<<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(),
                                                               devSysIdx.data(),
                                                               coords,
                                                               grad,
                                                               dimension,
                                                               activeThisStage);
  return cudaGetLastError();
}

// ETK (3D) combined kernels
__global__ void combinedEnergiesKernelETK(const Energy3DForceContribsDevicePtr* terms,
                                          const BatchedIndices3DDevicePtr*      systemIndices,
                                          const double*                         coords,
                                          double*                               energies,
                                          const uint8_t*                        activeThisStage) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;
  const int stride = blockDim.x;

  if (activeThisStage != nullptr && activeThisStage[molIdx] == 0) {
    if (tid == 0) {
      energies[molIdx] = 0.0;
    }
    return;
  }

  using BlockReduce = cub::BlockReduce<double, blockSizePerMol>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  const double threadEnergy = molEnergyETK(*terms, *systemIndices, coords, molIdx, tid, stride);
  const double blockEnergy  = BlockReduce(tempStorage).Sum(threadEnergy);

  if (tid == 0) {
    energies[molIdx] = blockEnergy;
  }
}

cudaError_t launchBlockPerMolEnergyKernelETK(int                                     numMols,
                                             const Energy3DForceContribsDevicePtr&   terms,
                                             const BatchedIndices3DDevicePtr&        systemIndices,
                                             const double*                           coords,
                                             double*                                 energies,
                                             const uint8_t*                          activeThisStage,
                                             cudaStream_t                            stream) {
  const AsyncDevicePtr<Energy3DForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndices3DDevicePtr>      devSysIdx(systemIndices, stream);
  combinedEnergiesKernelETK<<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(),
                                                                      devSysIdx.data(),
                                                                      coords,
                                                                      energies,
                                                                      activeThisStage);
  return cudaGetLastError();
}

// ETK (3D) combined gradient kernel
__global__ void combinedGradKernelETK(const Energy3DForceContribsDevicePtr* terms,
                                      const BatchedIndices3DDevicePtr*      systemIndices,
                                      const double*                         coords,
                                      double*                               grad,
                                      const uint8_t*                        activeThisStage) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;
  const int stride = blockDim.x;

  if (activeThisStage != nullptr && activeThisStage[molIdx] == 0) {
    return;
  }

  molGradETK(*terms, *systemIndices, coords, grad, molIdx, tid, stride);
}

cudaError_t launchBlockPerMolGradKernelETK(int                                     numMols,
                                           const Energy3DForceContribsDevicePtr&   terms,
                                           const BatchedIndices3DDevicePtr&        systemIndices,
                                           const double*                           coords,
                                           double*                                 grad,
                                           const uint8_t*                          activeThisStage,
                                           cudaStream_t                            stream) {
  const AsyncDevicePtr<Energy3DForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndices3DDevicePtr>      devSysIdx(systemIndices, stream);
  combinedGradKernelETK<<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(),
                                                                  devSysIdx.data(),
                                                                  coords,
                                                                  grad,
                                                                  activeThisStage);
  return cudaGetLastError();
}

}  // namespace DistGeom
}  // namespace nvMolKit
