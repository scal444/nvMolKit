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

#ifndef NVMOLKIT_DISTGEOM_KERNELS_DEVICE_CUH
#define NVMOLKIT_DISTGEOM_KERNELS_DEVICE_CUH

#include "dist_geom_kernels.h"
#include "kernel_utils.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
constexpr double RAD2DEG = 180.0 / M_PI;

using namespace nvMolKit::FFKernelUtils;

namespace nvMolKit {
namespace DistGeom {

// Device pointer structs are now defined in dist_geom_kernels.h

// Device helper functions for energy calculations
static __device__ __forceinline__ double distViolationEnergy(const double* pos,
                                                             const int     idx1,
                                                             const int     idx2,
                                                             const double  lb2,
                                                             const double  ub2,
                                                             const double  weight,
                                                             const int     dimension) {
  const int    posIdx1   = idx1 * dimension;
  const int    posIdx2   = idx2 * dimension;
  const double distance2 = distanceSquaredPosIdx(pos, posIdx1, posIdx2, dimension);
  double       val       = 0.0;
  if (distance2 > ub2) {
    val = (distance2 / ub2) - 1.0;
  } else if (distance2 < lb2) {
    val = ((2 * lb2) / (lb2 + distance2)) - 1.0;
  }
  if (val > 0.0) {
    return weight * val * val;
  }
  return 0.0;
}

static __device__ __forceinline__ void distViolationGrad(const double* pos,
                                                         const int     idx1,
                                                         const int     idx2,
                                                         const double  lb2,
                                                         const double  ub2,
                                                         const double  weight,
                                                         const int     dimension,
                                                         double*       grad) {
  const int   posIdx1   = idx1 * dimension;
  const int   posIdx2   = idx2 * dimension;
  const float distance2 = distanceSquaredPosIdx(pos, posIdx1, posIdx2, dimension);
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
  
  atomicAdd(&grad[posIdx1 + 0], dGradx);
  atomicAdd(&grad[posIdx1 + 1], dGrady);
  atomicAdd(&grad[posIdx1 + 2], dGradz);
  atomicAdd(&grad[posIdx2 + 0], -dGradx);
  atomicAdd(&grad[posIdx2 + 1], -dGrady);
  atomicAdd(&grad[posIdx2 + 2], -dGradz);
  
  if (dimension == 4) {
    const float dGradw = weight * preFactor * (pos[posIdx1 + 3] - pos[posIdx2 + 3]);
    atomicAdd(&grad[posIdx1 + 3], dGradw);
    atomicAdd(&grad[posIdx2 + 3], -dGradw);
  }
}

static __device__ __forceinline__ double calcChiralVolume(const int&    posIdx1,
                                                          const int&    posIdx2,
                                                          const int&    posIdx3,
                                                          const int&    posIdx4,
                                                          const double* pos,
                                                          double&       v1x,
                                                          double&       v1y,
                                                          double&       v1z,
                                                          double&       v2x,
                                                          double&       v2y,
                                                          double&       v2z,
                                                          double&       v3x,
                                                          double&       v3y,
                                                          double&       v3z) {
  v1x = pos[posIdx1 + 0] - pos[posIdx4 + 0];
  v1y = pos[posIdx1 + 1] - pos[posIdx4 + 1];
  v1z = pos[posIdx1 + 2] - pos[posIdx4 + 2];

  v2x = pos[posIdx2 + 0] - pos[posIdx4 + 0];
  v2y = pos[posIdx2 + 1] - pos[posIdx4 + 1];
  v2z = pos[posIdx2 + 2] - pos[posIdx4 + 2];

  v3x = pos[posIdx3 + 0] - pos[posIdx4 + 0];
  v3y = pos[posIdx3 + 1] - pos[posIdx4 + 1];
  v3z = pos[posIdx3 + 2] - pos[posIdx4 + 2];

  double v2v3x, v2v3y, v2v3z;
  crossProduct(v2x, v2y, v2z, v3x, v3y, v3z, v2v3x, v2v3y, v2v3z);
  double vol = dotProduct(v1x, v1y, v1z, v2v3x, v2v3y, v2v3z);
  return vol;
}

static __device__ __forceinline__ double chiralViolationEnergy(const double* pos,
                                                               const int     idx1,
                                                               const int     idx2,
                                                               const int     idx3,
                                                               const int     idx4,
                                                               const double  lb,
                                                               const double  ub,
                                                               const double  weight,
                                                               const int     dimension) {
  const int posIdx1 = idx1 * dimension;
  const int posIdx2 = idx2 * dimension;
  const int posIdx3 = idx3 * dimension;
  const int posIdx4 = idx4 * dimension;

  double v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z;
  double vol = calcChiralVolume(posIdx1, posIdx2, posIdx3, posIdx4, pos, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z);

  if (vol < lb) {
    return weight * (vol - lb) * (vol - lb);
  } else if (vol > ub) {
    return weight * (vol - ub) * (vol - ub);
  }
  return 0.0;
}

static __device__ __forceinline__ void chiralViolationGrad(const double* pos,
                                                           const int     idx1,
                                                           const int     idx2,
                                                           const int     idx3,
                                                           const int     idx4,
                                                           const double  lb,
                                                           const double  ub,
                                                           const double  weight,
                                                           const int     dimension,
                                                           double*       grad) {
  const int posIdx1 = idx1 * dimension;
  const int posIdx2 = idx2 * dimension;
  const int posIdx3 = idx3 * dimension;
  const int posIdx4 = idx4 * dimension;

  double v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z;
  double vol = calcChiralVolume(posIdx1, posIdx2, posIdx3, posIdx4, pos, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z);

  if (vol < lb || vol > ub) {
    double preFactor;
    if (vol < lb) {
      preFactor = weight * (vol - lb);
    } else {
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

static __device__ __forceinline__ double fourthDimEnergy(const double* pos,
                                                         const int     idx,
                                                         const double  weight,
                                                         const int     dimension) {
  if (dimension != 4) {
    return 0.0;
  }
  const int    posIdx    = idx * dimension;
  const double fourthVal = pos[posIdx + 3];
  return weight * fourthVal * fourthVal;
}

static __device__ __forceinline__ void fourthDimGrad(const double* pos,
                                                     const int     idx,
                                                     const double  weight,
                                                     const int     dimension,
                                                     double*       grad) {
  if (dimension != 4) {
    return;
  }
  const int    posIdx    = idx * dimension;
  const double fourthVal = pos[posIdx + 3];
  atomicAdd(&grad[posIdx + 3], weight * fourthVal);
}

// Consolidated per-molecule energy calculation
static __device__ __inline__ double molEnergy(const EnergyForceContribsDevicePtr& terms,
                                              const BatchedIndicesDevicePtr&      systemIndices,
                                              const double*                       coords,
                                              const int                           molIdx,
                                              const int                           dimension,
                                              const int                           tid,
                                              const int                           stride) {
  const int     atomStart = systemIndices.atomStarts[molIdx];
  const double* molCoords = coords + atomStart * dimension;

  double energy = 0.0;

  const auto& [d_idx1s, d_idx2s, d_ub2s, d_lb2s, d_weights] = terms.distTerms;
  const int distStart                                       = systemIndices.distTermStarts[molIdx];
  const int distEnd                                         = systemIndices.distTermStarts[molIdx + 1];
  for (int i = distStart + tid; i < distEnd; i += stride) {
    const int localIdx1 = d_idx1s[i] - atomStart;
    const int localIdx2 = d_idx2s[i] - atomStart;
    energy += distViolationEnergy(molCoords, localIdx1, localIdx2, d_lb2s[i], d_ub2s[i], d_weights[i], dimension);
  }

  const auto& [c_idx1s, c_idx2s, c_idx3s, c_idx4s, c_volUppers, c_volLowers, c_weights] = terms.chiralTerms;
  const int chiralStart                                                                 = systemIndices.chiralTermStarts[molIdx];
  const int chiralEnd = systemIndices.chiralTermStarts[molIdx + 1];
  for (int i = chiralStart + tid; i < chiralEnd; i += stride) {
    const int localIdx1 = c_idx1s[i] - atomStart;
    const int localIdx2 = c_idx2s[i] - atomStart;
    const int localIdx3 = c_idx3s[i] - atomStart;
    const int localIdx4 = c_idx4s[i] - atomStart;
    energy += chiralViolationEnergy(molCoords,
                                    localIdx1,
                                    localIdx2,
                                    localIdx3,
                                    localIdx4,
                                    c_volLowers[i],
                                    c_volUppers[i],
                                    c_weights[i],
                                    dimension);
  }

  const auto& [f_idxs, f_weights] = terms.fourthTerms;
  const int fourthStart          = systemIndices.fourthTermStarts[molIdx];
  const int fourthEnd            = systemIndices.fourthTermStarts[molIdx + 1];
  for (int i = fourthStart + tid; i < fourthEnd; i += stride) {
    const int localIdx = f_idxs[i] - atomStart;
    energy += fourthDimEnergy(molCoords, localIdx, f_weights[i], dimension);
  }

  return energy;
}

// Consolidated per-molecule gradient calculation
static __device__ __inline__ void molGrad(const EnergyForceContribsDevicePtr& terms,
                                          const BatchedIndicesDevicePtr&      systemIndices,
                                          const double*                       coords,
                                          double*                             grad,
                                          const int                           molIdx,
                                          const int                           dimension,
                                          const int                           tid,
                                          const int                           stride) {
  const int     atomStart = systemIndices.atomStarts[molIdx];
  const double* molCoords = coords + atomStart * dimension;
  double*       molGrad   = grad;  // grad is already offset by caller (see combinedGradKernel)

  const auto& [d_idx1s, d_idx2s, d_ub2s, d_lb2s, d_weights] = terms.distTerms;
  const int distStart                                       = systemIndices.distTermStarts[molIdx];
  const int distEnd                                         = systemIndices.distTermStarts[molIdx + 1];
  for (int i = distStart + tid; i < distEnd; i += stride) {
    const int localIdx1 = d_idx1s[i] - atomStart;
    const int localIdx2 = d_idx2s[i] - atomStart;
    distViolationGrad(molCoords, localIdx1, localIdx2, d_lb2s[i], d_ub2s[i], d_weights[i], dimension, molGrad);
  }

  const auto& [c_idx1s, c_idx2s, c_idx3s, c_idx4s, c_volUppers, c_volLowers, c_weights] = terms.chiralTerms;
  const int chiralStart = systemIndices.chiralTermStarts[molIdx];
  const int chiralEnd   = systemIndices.chiralTermStarts[molIdx + 1];
  for (int i = chiralStart + tid; i < chiralEnd; i += stride) {
    const int localIdx1 = c_idx1s[i] - atomStart;
    const int localIdx2 = c_idx2s[i] - atomStart;
    const int localIdx3 = c_idx3s[i] - atomStart;
    const int localIdx4 = c_idx4s[i] - atomStart;
    chiralViolationGrad(molCoords,
                        localIdx1,
                        localIdx2,
                        localIdx3,
                        localIdx4,
                        c_volLowers[i],
                        c_volUppers[i],
                        c_weights[i],
                        dimension,
                        molGrad);
  }

  const auto& [f_idxs, f_weights] = terms.fourthTerms;
  const int fourthStart          = systemIndices.fourthTermStarts[molIdx];
  const int fourthEnd            = systemIndices.fourthTermStarts[molIdx + 1];
  for (int i = fourthStart + tid; i < fourthEnd; i += stride) {
    const int localIdx = f_idxs[i] - atomStart;
    fourthDimGrad(molCoords, localIdx, f_weights[i], dimension, molGrad);
  }
}

// ETK (Experimental Torsion Knowledge) pointer structs
// Device pointer structs are now defined in dist_geom_kernels.h

// Helper device functions for ETK energy calculations
static __device__ __forceinline__ double calcTorsionEnergyM6(const double* forceConstants,
                                                             const int*    signs,
                                                             const double  cosPhi) {
  const double cosPhi2 = cosPhi * cosPhi;
  const double cosPhi3 = cosPhi * cosPhi2;
  const double cosPhi4 = cosPhi * cosPhi3;
  const double cosPhi5 = cosPhi * cosPhi4;
  const double cosPhi6 = cosPhi * cosPhi5;

  const double cos2Phi = 2.0 * cosPhi2 - 1.0;
  const double cos3Phi = 4.0 * cosPhi3 - 3.0 * cosPhi;
  const double cos4Phi = 8.0 * cosPhi4 - 8.0 * cosPhi2 + 1.0;
  const double cos5Phi = 16.0 * cosPhi5 - 20.0 * cosPhi3 + 5.0 * cosPhi;
  const double cos6Phi = 32.0 * cosPhi6 - 48.0 * cosPhi4 + 18.0 * cosPhi2 - 1.0;

  return (forceConstants[0] * (1.0 + signs[0] * cosPhi) + forceConstants[1] * (1.0 + signs[1] * cos2Phi) +
          forceConstants[2] * (1.0 + signs[2] * cos3Phi) + forceConstants[3] * (1.0 + signs[3] * cos4Phi) +
          forceConstants[4] * (1.0 + signs[4] * cos5Phi) + forceConstants[5] * (1.0 + signs[5] * cos6Phi));
}

static __device__ __forceinline__ double calcTorsionCosPhi(const double* pos,
                                                           const int     posIdx1,
                                                           const int     posIdx2,
                                                           const int     posIdx3,
                                                           const int     posIdx4) {
  double r1x = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  double r1y = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  double r1z = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  double r2x = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  double r2y = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  double r2z = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  double r3x = pos[posIdx2 + 0] - pos[posIdx3 + 0];
  double r3y = pos[posIdx2 + 1] - pos[posIdx3 + 1];
  double r3z = pos[posIdx2 + 2] - pos[posIdx3 + 2];

  double r4x = pos[posIdx4 + 0] - pos[posIdx3 + 0];
  double r4y = pos[posIdx4 + 1] - pos[posIdx3 + 1];
  double r4z = pos[posIdx4 + 2] - pos[posIdx3 + 2];

  double t1x, t1y, t1z;
  crossProduct(r1x, r1y, r1z, r2x, r2y, r2z, t1x, t1y, t1z);

  double t2x, t2y, t2z;
  crossProduct(r3x, r3y, r3z, r4x, r4y, r4z, t2x, t2y, t2z);

  double t1_len = sqrt(t1x * t1x + t1y * t1y + t1z * t1z);
  double t2_len = sqrt(t2x * t2x + t2y * t2y + t2z * t2z);

  if (isDoubleZero(t1_len) || isDoubleZero(t2_len)) {
    return 0.0;
  }

  double cosPhi = dotProduct(t1x, t1y, t1z, t2x, t2y, t2z) / (t1_len * t2_len);
  clipToOne(cosPhi);
  return cosPhi;
}

static __device__ __forceinline__ double torsionAngleEnergy(const double* pos,
                                                            const int     idx1,
                                                            const int     idx2,
                                                            const int     idx3,
                                                            const int     idx4,
                                                            const double* forceConstants,
                                                            const int*    signs) {
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;
  const int posIdx4 = idx4 * 4;

  double cosPhi = calcTorsionCosPhi(pos, posIdx1, posIdx2, posIdx3, posIdx4);
  return calcTorsionEnergyM6(forceConstants, signs, cosPhi);
}

static __device__ __forceinline__ double calcInversionCosY(const double* pos,
                                                           const int     posIdx1,
                                                           const int     posIdx2,
                                                           const int     posIdx3,
                                                           const int     posIdx4) {
  constexpr double inversionZeroTol = 1.0e-16;

  double rJIx = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  double rJIy = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  double rJIz = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  double rJKx = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  double rJKy = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  double rJKz = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  double rJLx = pos[posIdx4 + 0] - pos[posIdx2 + 0];
  double rJLy = pos[posIdx4 + 1] - pos[posIdx2 + 1];
  double rJLz = pos[posIdx4 + 2] - pos[posIdx2 + 2];

  double l2JI = rJIx * rJIx + rJIy * rJIy + rJIz * rJIz;
  double l2JK = rJKx * rJKx + rJKy * rJKy + rJKz * rJKz;
  double l2JL = rJLx * rJLx + rJLy * rJLy + rJLz * rJLz;

  if (l2JI < inversionZeroTol || l2JK < inversionZeroTol || l2JL < inversionZeroTol) {
    return 0.0;
  }

  double nx, ny, nz;
  crossProduct(rJIx, rJIy, rJIz, rJKx, rJKy, rJKz, nx, ny, nz);

  double norm_factor = sqrt(l2JI) * sqrt(l2JK);
  nx /= norm_factor;
  ny /= norm_factor;
  nz /= norm_factor;

  double l2n = nx * nx + ny * ny + nz * nz;
  if (l2n < inversionZeroTol) {
    return 0.0;
  }

  return dotProduct(nx, ny, nz, rJLx, rJLy, rJLz) / (sqrt(l2JL) * sqrt(l2n));
}

static __device__ __forceinline__ double inversionEnergy(const double* pos,
                                                         const int     idx1,
                                                         const int     idx2,
                                                         const int     idx3,
                                                         const int     idx4,
                                                         const double  C0,
                                                         const double  C1,
                                                         const double  C2,
                                                         const double  forceConstant) {
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;
  const int posIdx4 = idx4 * 4;

  double cosY = calcInversionCosY(pos, posIdx1, posIdx2, posIdx3, posIdx4);

  const double sinYSq = 1.0 - cosY * cosY;
  const double sinY   = ((sinYSq > 0.0) ? sqrt(sinYSq) : 0.0);
  const double cos2W  = 2.0 * sinY * sinY - 1.0;

  return forceConstant * (C0 + C1 * sinY + C2 * cos2W);
}

static __device__ __forceinline__ double distanceConstraintEnergy(const double* pos,
                                                                  const int     idx1,
                                                                  const int     idx2,
                                                                  const double  minLen,
                                                                  const double  maxLen,
                                                                  const double  forceConstant) {
  const int    posIdx1   = idx1 * 4;
  const int    posIdx2   = idx2 * 4;
  const double distance2 = distanceSquaredPosIdx(pos, posIdx1, posIdx2, 3);

  const double minLen2 = minLen * minLen;
  const double maxLen2 = maxLen * maxLen;

  double difference = 0.0;
  if (distance2 < minLen2) {
    difference = minLen - sqrt(distance2);
  } else if (distance2 > maxLen2) {
    difference = sqrt(distance2) - maxLen;
  } else {
    return 0.0;
  }

  return 0.5 * forceConstant * difference * difference;
}

static __device__ __forceinline__ double computeAngleTerm(const double angle,
                                                          const double minAngle,
                                                          const double maxAngle) {
  double angleTerm = 0.0;
  if (angle < minAngle) {
    angleTerm = angle - minAngle;
  } else if (angle > maxAngle) {
    angleTerm = angle - maxAngle;
  }
  return angleTerm;
}

static __device__ __forceinline__ double angleConstraintEnergy(const double* pos,
                                                               const int     idx1,
                                                               const int     idx2,
                                                               const int     idx3,
                                                               const double  minAngle,
                                                               const double  maxAngle,
                                                               const double  forceConstant) {
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;

  double dx1 = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  double dy1 = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  double dz1 = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  double dx2 = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  double dy2 = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  double dz2 = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  const double dist1Sq = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
  const double dist2Sq = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
  const double dist1   = sqrt(dist1Sq);
  const double dist2   = sqrt(dist2Sq);

  if (isDoubleZero(dist1) || isDoubleZero(dist2)) {
    return 0.0;
  }

  const double dot      = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const double cosTheta = clamp(dot / (dist1 * dist2), -1.0, 1.0);
  const double angle    = RAD2DEG * acos(cosTheta);

  const double angleTerm = computeAngleTerm(angle, minAngle, maxAngle);
  return forceConstant * angleTerm * angleTerm;
}

// Consolidated per-molecule ETK energy calculation
static __device__ __inline__ double molEnergyETK(const Energy3DForceContribsDevicePtr& terms,
                                                 const BatchedIndices3DDevicePtr&      systemIndices,
                                                 const double*                         coords,
                                                 const int                             molIdx,
                                                 const int                             tid,
                                                 const int                             stride) {
  const int     atomStart = systemIndices.atomStarts[molIdx];
  const double* molCoords = coords + atomStart * 4;  // ETK uses 4D coordinates

  double energy = 0.0;

  // Experimental torsion terms
  const auto& [t_idx1s, t_idx2s, t_idx3s, t_idx4s, t_forceConstants, t_signs] = terms.experimentalTorsionTerms;
  const int torsionStart                                                      = systemIndices.experimentalTorsionTermStarts[molIdx];
  const int torsionEnd = systemIndices.experimentalTorsionTermStarts[molIdx + 1];
  for (int i = torsionStart + tid; i < torsionEnd; i += stride) {
    const int localIdx1 = t_idx1s[i] - atomStart;
    const int localIdx2 = t_idx2s[i] - atomStart;
    const int localIdx3 = t_idx3s[i] - atomStart;
    const int localIdx4 = t_idx4s[i] - atomStart;
    energy += torsionAngleEnergy(molCoords, localIdx1, localIdx2, localIdx3, localIdx4, &t_forceConstants[i * 6], &t_signs[i * 6]);
  }

  // Improper torsion terms
  const auto& [i_idx1s, i_idx2s, i_idx3s, i_idx4s, i_at2AtomicNum, i_isCBoundToO, i_C0, i_C1, i_C2, i_forceConstant] =
    terms.improperTorsionTerms;
  const int improperStart = systemIndices.improperTorsionTermStarts[molIdx];
  const int improperEnd   = systemIndices.improperTorsionTermStarts[molIdx + 1];
  for (int i = improperStart + tid; i < improperEnd; i += stride) {
    const int localIdx1 = i_idx1s[i] - atomStart;
    const int localIdx2 = i_idx2s[i] - atomStart;
    const int localIdx3 = i_idx3s[i] - atomStart;
    const int localIdx4 = i_idx4s[i] - atomStart;
    energy += inversionEnergy(molCoords, localIdx1, localIdx2, localIdx3, localIdx4, i_C0[i], i_C1[i], i_C2[i], i_forceConstant[i]);
  }

  // 1-2 distance terms
  const auto& [d12_idx1s, d12_idx2s, d12_minLen, d12_maxLen, d12_forceConstant] = terms.dist12Terms;
  const int dist12Start                                                         = systemIndices.dist12TermStarts[molIdx];
  const int dist12End                                                           = systemIndices.dist12TermStarts[molIdx + 1];
  for (int i = dist12Start + tid; i < dist12End; i += stride) {
    const int localIdx1 = d12_idx1s[i] - atomStart;
    const int localIdx2 = d12_idx2s[i] - atomStart;
    energy += distanceConstraintEnergy(molCoords, localIdx1, localIdx2, d12_minLen[i], d12_maxLen[i], d12_forceConstant[i]);
  }

  // 1-3 distance terms
  const auto& [d13_idx1s, d13_idx2s, d13_minLen, d13_maxLen, d13_forceConstant] = terms.dist13Terms;
  const int dist13Start                                                         = systemIndices.dist13TermStarts[molIdx];
  const int dist13End                                                           = systemIndices.dist13TermStarts[molIdx + 1];
  for (int i = dist13Start + tid; i < dist13End; i += stride) {
    const int localIdx1 = d13_idx1s[i] - atomStart;
    const int localIdx2 = d13_idx2s[i] - atomStart;
    energy += distanceConstraintEnergy(molCoords, localIdx1, localIdx2, d13_minLen[i], d13_maxLen[i], d13_forceConstant[i]);
  }

  // 1-3 angle terms
  const auto& [a13_idx1s, a13_idx2s, a13_idx3s, a13_minAngle, a13_maxAngle] = terms.angle13Terms;
  const int angle13Start                                                    = systemIndices.angle13TermStarts[molIdx];
  const int angle13End                                                      = systemIndices.angle13TermStarts[molIdx + 1];
  constexpr double defaultAngleForceConstant = 1.0;
  for (int i = angle13Start + tid; i < angle13End; i += stride) {
    const int localIdx1 = a13_idx1s[i] - atomStart;
    const int localIdx2 = a13_idx2s[i] - atomStart;
    const int localIdx3 = a13_idx3s[i] - atomStart;
    energy += angleConstraintEnergy(molCoords,
                                    localIdx1,
                                    localIdx2,
                                    localIdx3,
                                    a13_minAngle[i],
                                    a13_maxAngle[i],
                                    defaultAngleForceConstant);
  }

  // Long range distance terms
  const auto& [dlr_idx1s, dlr_idx2s, dlr_minLen, dlr_maxLen, dlr_forceConstant] = terms.longRangeDistTerms;
  const int distLRStart                                                         = systemIndices.longRangeDistTermStarts[molIdx];
  const int distLREnd                                                           = systemIndices.longRangeDistTermStarts[molIdx + 1];
  for (int i = distLRStart + tid; i < distLREnd; i += stride) {
    const int localIdx1 = dlr_idx1s[i] - atomStart;
    const int localIdx2 = dlr_idx2s[i] - atomStart;
    energy += distanceConstraintEnergy(molCoords, localIdx1, localIdx2, dlr_minLen[i], dlr_maxLen[i], dlr_forceConstant[i]);
  }

  return energy;
}

// =============================================================================
// ETK Gradient Device Functions (defined once, used by both batched and per-mol kernels)
// =============================================================================

// Helper function to calculate torsion gradients
__device__ __forceinline__ void calcTorsionGrad(const double* r,  // 4 vectors of 3 components each
                                                const double* t,  // 2 vectors of 3 components each
                                                const double* d,  // 2 lengths
                                                double*       g,  // 4 gradient vectors of 3 components each
                                                const double  sinTerm,
                                                const double  cosPhi) {
  // Calculate dCos_dT
  double dCos_dT[6];
  dCos_dT[0] = 1.0 / d[0] * (t[3] - cosPhi * t[0]);  // t[1].x - cosPhi * t[0].x
  dCos_dT[1] = 1.0 / d[0] * (t[4] - cosPhi * t[1]);  // t[1].y - cosPhi * t[0].y
  dCos_dT[2] = 1.0 / d[0] * (t[5] - cosPhi * t[2]);  // t[1].z - cosPhi * t[0].z
  dCos_dT[3] = 1.0 / d[1] * (t[0] - cosPhi * t[3]);  // t[0].x - cosPhi * t[1].x
  dCos_dT[4] = 1.0 / d[1] * (t[1] - cosPhi * t[4]);  // t[0].y - cosPhi * t[1].y
  dCos_dT[5] = 1.0 / d[1] * (t[2] - cosPhi * t[5]);  // t[0].z - cosPhi * t[1].z

  // Calculate gradients for each atom
  // Atom 1 (i)
  g[0] = sinTerm * (dCos_dT[2] * r[4] - dCos_dT[1] * r[5]);  // x
  g[1] = sinTerm * (dCos_dT[0] * r[5] - dCos_dT[2] * r[3]);  // y
  g[2] = sinTerm * (dCos_dT[1] * r[3] - dCos_dT[0] * r[4]);  // z

  // Atom 2 (j)
  g[3] = sinTerm *
         (dCos_dT[1] * (r[5] - r[2]) + dCos_dT[2] * (r[1] - r[4]) + dCos_dT[4] * (-r[11]) + dCos_dT[5] * (r[10]));  // x
  g[4] = sinTerm *
         (dCos_dT[0] * (r[2] - r[5]) + dCos_dT[2] * (r[3] - r[0]) + dCos_dT[3] * (r[11]) + dCos_dT[5] * (-r[9]));  // y
  g[5] = sinTerm *
         (dCos_dT[0] * (r[4] - r[1]) + dCos_dT[1] * (r[0] - r[3]) + dCos_dT[3] * (-r[10]) + dCos_dT[4] * (r[9]));  // z

  // Atom 3 (k)
  g[6] = sinTerm *
         (dCos_dT[1] * (r[2]) + dCos_dT[2] * (-r[1]) + dCos_dT[4] * (r[11] - r[8]) + dCos_dT[5] * (r[7] - r[10]));  // x
  g[7] = sinTerm *
         (dCos_dT[0] * (-r[2]) + dCos_dT[2] * (r[0]) + dCos_dT[3] * (r[8] - r[11]) + dCos_dT[5] * (r[9] - r[6]));  // y
  g[8] = sinTerm *
         (dCos_dT[0] * (r[1]) + dCos_dT[1] * (-r[0]) + dCos_dT[3] * (r[10] - r[7]) + dCos_dT[4] * (r[6] - r[9]));  // z

  // Atom 4 (l)
  g[9]  = sinTerm * (dCos_dT[4] * r[8] - dCos_dT[5] * r[7]);  // x
  g[10] = sinTerm * (dCos_dT[5] * r[6] - dCos_dT[3] * r[8]);  // y
  g[11] = sinTerm * (dCos_dT[3] * r[7] - dCos_dT[4] * r[6]);  // z
}

// Experimental torsion angle gradient
static __device__ __forceinline__ void torsionAngleGrad(const double* pos,
                                                        const int     idx1,
                                                        const int     idx2,
                                                        const int     idx3,
                                                        const int     idx4,
                                                        const double* forceConstants,  // 6 components
                                                        const int*    signs,           // 6 components
                                                        double*       grad) {
  // Get positions for all four atoms
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;
  const int posIdx4 = idx4 * 4;

  // Calculate vectors
  double r[12];                                 // 4 vectors of 3 components each
  r[0]  = pos[posIdx1 + 0] - pos[posIdx2 + 0];  // r1.x
  r[1]  = pos[posIdx1 + 1] - pos[posIdx2 + 1];  // r1.y
  r[2]  = pos[posIdx1 + 2] - pos[posIdx2 + 2];  // r1.z
  r[3]  = pos[posIdx3 + 0] - pos[posIdx2 + 0];  // r2.x
  r[4]  = pos[posIdx3 + 1] - pos[posIdx2 + 1];  // r2.y
  r[5]  = pos[posIdx3 + 2] - pos[posIdx2 + 2];  // r2.z
  r[6]  = pos[posIdx2 + 0] - pos[posIdx3 + 0];  // r3.x
  r[7]  = pos[posIdx2 + 1] - pos[posIdx3 + 1];  // r3.y
  r[8]  = pos[posIdx2 + 2] - pos[posIdx3 + 2];  // r3.z
  r[9]  = pos[posIdx4 + 0] - pos[posIdx3 + 0];  // r4.x
  r[10] = pos[posIdx4 + 1] - pos[posIdx3 + 1];  // r4.y
  r[11] = pos[posIdx4 + 2] - pos[posIdx3 + 2];  // r4.z

  // Calculate cross products
  double t[6];  // 2 vectors of 3 components each
  crossProduct(r[0], r[1], r[2], r[3], r[4], r[5], t[0], t[1], t[2]);
  crossProduct(r[6], r[7], r[8], r[9], r[10], r[11], t[3], t[4], t[5]);

  // Calculate lengths
  double d[2];
  d[0] = sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
  d[1] = sqrt(t[3] * t[3] + t[4] * t[4] + t[5] * t[5]);

  if (isDoubleZero(d[0]) || isDoubleZero(d[1])) {
    return;
  }

  // Normalize vectors
  t[0] /= d[0];
  t[1] /= d[0];
  t[2] /= d[0];
  t[3] /= d[1];
  t[4] /= d[1];
  t[5] /= d[1];

  // Calculate cosine of torsion angle
  double cosPhi = dotProduct(t[0], t[1], t[2], t[3], t[4], t[5]);
  clipToOne(cosPhi);

  // Calculate sinPhi
  const double sinPhiSq = 1.0 - cosPhi * cosPhi;
  const double sinPhi   = ((sinPhiSq > 0.0) ? sqrt(sinPhiSq) : 0.0);

  // Calculate derivatives
  const double cosPhi2 = cosPhi * cosPhi;
  const double cosPhi3 = cosPhi * cosPhi2;
  const double cosPhi4 = cosPhi * cosPhi3;
  const double cosPhi5 = cosPhi * cosPhi4;

  // Calculate dE/dPhi
  const double dE_dPhi =
    (-forceConstants[0] * signs[0] * sinPhi - 2.0 * forceConstants[1] * signs[1] * (2.0 * cosPhi * sinPhi) -
     3.0 * forceConstants[2] * signs[2] * (4.0 * cosPhi2 * sinPhi - sinPhi) -
     4.0 * forceConstants[3] * signs[3] * (8.0 * cosPhi3 * sinPhi - 4.0 * cosPhi * sinPhi) -
     5.0 * forceConstants[4] * signs[4] * (16.0 * cosPhi4 * sinPhi - 12.0 * cosPhi2 * sinPhi + sinPhi) -
     6.0 * forceConstants[4] * signs[4] * (32.0 * cosPhi5 * sinPhi - 32.0 * cosPhi3 * sinPhi + 6.0 * sinPhi));

  // Calculate sinTerm
  double sinTerm = -dE_dPhi * (isDoubleZero(sinPhi) ? (1.0 / cosPhi) : (1.0 / sinPhi));

  // Calculate gradients
  double g[12];  // 4 gradient vectors of 3 components each
  for (int i = 0; i < 12; ++i) {
    g[i] = 0.0;
  }
  calcTorsionGrad(r, t, d, g, sinTerm, cosPhi);

  // Add gradients to global gradient array using atomic operations
  atomicAdd(&grad[posIdx1 + 0], g[0]);
  atomicAdd(&grad[posIdx1 + 1], g[1]);
  atomicAdd(&grad[posIdx1 + 2], g[2]);
  atomicAdd(&grad[posIdx2 + 0], g[3]);
  atomicAdd(&grad[posIdx2 + 1], g[4]);
  atomicAdd(&grad[posIdx2 + 2], g[5]);
  atomicAdd(&grad[posIdx3 + 0], g[6]);
  atomicAdd(&grad[posIdx3 + 1], g[7]);
  atomicAdd(&grad[posIdx3 + 2], g[8]);
  atomicAdd(&grad[posIdx4 + 0], g[9]);
  atomicAdd(&grad[posIdx4 + 1], g[10]);
  atomicAdd(&grad[posIdx4 + 2], g[11]);
}

// Improper torsion (inversion) gradient
static __device__ __forceinline__ void inversionGrad(const double* pos,
                                                     const int     idx1,
                                                     const int     idx2,
                                                     const int     idx3,
                                                     const int     idx4,
                                                     const double  C0,
                                                     const double  C1,
                                                     const double  C2,
                                                     const double  forceConstant,
                                                     double*       grad) {
  // Get positions for all four atoms
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;
  const int posIdx4 = idx4 * 4;

  // Calculate vectors
  double rJIx = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  double rJIy = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  double rJIz = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  double rJKx = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  double rJKy = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  double rJKz = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  double rJLx = pos[posIdx4 + 0] - pos[posIdx2 + 0];
  double rJLy = pos[posIdx4 + 1] - pos[posIdx2 + 1];
  double rJLz = pos[posIdx4 + 2] - pos[posIdx2 + 2];

  // Calculate lengths
  double dJI = sqrt(rJIx * rJIx + rJIy * rJIy + rJIz * rJIz);
  double dJK = sqrt(rJKx * rJKx + rJKy * rJKy + rJKz * rJKz);
  double dJL = sqrt(rJLx * rJLx + rJLy * rJLy + rJLz * rJLz);

  // Check for zero lengths
  if (isDoubleZero(dJI) || isDoubleZero(dJK) || isDoubleZero(dJL)) {
    return;
  }

  // Normalize vectors
  rJIx /= dJI;
  rJIy /= dJI;
  rJIz /= dJI;
  rJKx /= dJK;
  rJKy /= dJK;
  rJKz /= dJK;
  rJLx /= dJL;
  rJLy /= dJL;
  rJLz /= dJL;

  // Calculate n = (-rJI) × rJK
  double nx, ny, nz;
  crossProduct(-rJIx, -rJIy, -rJIz, rJKx, rJKy, rJKz, nx, ny, nz);

  // Normalize n
  double n_len = sqrt(nx * nx + ny * ny + nz * nz);
  nx /= n_len;
  ny /= n_len;
  nz /= n_len;

  // Calculate cosY and clamp
  double cosY = dotProduct(nx, ny, nz, rJLx, rJLy, rJLz);
  clipToOne(cosY);

  // Calculate sinY
  const double sinYSq = 1.0 - cosY * cosY;
  const double sinY   = fmax(sqrt(sinYSq), 1.0e-8);

  // Calculate cosTheta and clamp
  double cosTheta = dotProduct(rJIx, rJIy, rJIz, rJKx, rJKy, rJKz);
  clipToOne(cosTheta);

  // Calculate sinTheta
  const double sinThetaSq = 1.0 - cosTheta * cosTheta;
  const double sinTheta   = fmax(sqrt(sinThetaSq), 1.0e-8);

  // Calculate dE_dW
  const double dE_dW = -forceConstant * (C1 * cosY - 4.0 * C2 * cosY * sinY);

  // Calculate cross products for gradient terms
  double t1x, t1y, t1z;  // rJL × rJK
  crossProduct(rJLx, rJLy, rJLz, rJKx, rJKy, rJKz, t1x, t1y, t1z);

  double t2x, t2y, t2z;  // rJI × rJL
  crossProduct(rJIx, rJIy, rJIz, rJLx, rJLy, rJLz, t2x, t2y, t2z);

  double t3x, t3y, t3z;  // rJK × rJI
  crossProduct(rJKx, rJKy, rJKz, rJIx, rJIy, rJIz, t3x, t3y, t3z);

  // Calculate terms for gradient
  const double term1 = sinY * sinTheta;
  const double term2 = cosY / (sinY * sinThetaSq);

  // Calculate gradient components for each atom
  double tg1[3] = {(t1x / term1 - (rJIx - rJKx * cosTheta) * term2) / dJI,
                   (t1y / term1 - (rJIy - rJKy * cosTheta) * term2) / dJI,
                   (t1z / term1 - (rJIz - rJKz * cosTheta) * term2) / dJI};

  double tg3[3] = {(t2x / term1 - (rJKx - rJIx * cosTheta) * term2) / dJK,
                   (t2y / term1 - (rJKy - rJIy * cosTheta) * term2) / dJK,
                   (t2z / term1 - (rJKz - rJIz * cosTheta) * term2) / dJK};

  double tg4[3] = {(t3x / term1 - rJLx * cosY / sinY) / dJL,
                   (t3y / term1 - rJLy * cosY / sinY) / dJL,
                   (t3z / term1 - rJLz * cosY / sinY) / dJL};

  // Add gradients to global gradient array using atomic operations
  atomicAdd(&grad[posIdx1 + 0], dE_dW * tg1[0]);
  atomicAdd(&grad[posIdx1 + 1], dE_dW * tg1[1]);
  atomicAdd(&grad[posIdx1 + 2], dE_dW * tg1[2]);

  atomicAdd(&grad[posIdx2 + 0], -dE_dW * (tg1[0] + tg3[0] + tg4[0]));
  atomicAdd(&grad[posIdx2 + 1], -dE_dW * (tg1[1] + tg3[1] + tg4[1]));
  atomicAdd(&grad[posIdx2 + 2], -dE_dW * (tg1[2] + tg3[2] + tg4[2]));

  atomicAdd(&grad[posIdx3 + 0], dE_dW * tg3[0]);
  atomicAdd(&grad[posIdx3 + 1], dE_dW * tg3[1]);
  atomicAdd(&grad[posIdx3 + 2], dE_dW * tg3[2]);

  atomicAdd(&grad[posIdx4 + 0], dE_dW * tg4[0]);
  atomicAdd(&grad[posIdx4 + 1], dE_dW * tg4[1]);
  atomicAdd(&grad[posIdx4 + 2], dE_dW * tg4[2]);
}

// Distance constraint gradient
static __device__ __forceinline__ void distanceConstraintGrad(const double* pos,
                                                              const int     idx1,
                                                              const int     idx2,
                                                              const double  minLen,
                                                              const double  maxLen,
                                                              const double  forceConstant,
                                                              double*       grad) {
  const double minLen2 = minLen * minLen;
  const double maxLen2 = maxLen * maxLen;
  const int    posIdx1 = idx1 * 4;
  const int    posIdx2 = idx2 * 4;

  // Calculate squared distance
  const double distance2 = distanceSquaredPosIdx(pos, posIdx1, posIdx2, 3);

  // Check if distance is outside bounds
  double preFactor = 0.0;
  double distance  = 0.0;
  if (distance2 < minLen2) {
    distance  = sqrt(distance2);
    preFactor = distance - minLen;
  } else if (distance2 > maxLen2) {
    distance  = sqrt(distance2);
    preFactor = distance - maxLen;
  } else {
    return;  // Distance within bounds, no gradient contribution
  }

  // Calculate final preFactor
  preFactor *= forceConstant;
  preFactor /= fmax(1.0e-8, distance);

  // Calculate and accumulate gradients for each component
  for (int i = 0; i < 3; i++) {
    const double dGrad = preFactor * (pos[posIdx1 + i] - pos[posIdx2 + i]);
    atomicAdd(&grad[posIdx1 + i], dGrad);
    atomicAdd(&grad[posIdx2 + i], -dGrad);
  }
}

// Angle constraint gradient
static __device__ __forceinline__ void angleConstraintGrad(const double* pos,
                                                           const int     idx1,
                                                           const int     idx2,
                                                           const int     idx3,
                                                           const double  minAngle,
                                                           const double  maxAngle,
                                                           const double  forceConstant,
                                                           double*       grad) {
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
  const double angleTerm = computeAngleTerm(angle, minAngle, maxAngle);

  // Calculate dE_dTheta
  const double dE_dTheta = 2.0 * RAD2DEG * forceConstant * angleTerm;

  // Calculate cross product rp = r2 × r1
  double rpx, rpy, rpz;
  crossProduct(r2x, r2y, r2z, r1x, r1y, r1z, rpx, rpy, rpz);

  // Calculate length of rp and prefactor
  const double rpLengthSq = rpx * rpx + rpy * rpy + rpz * rpz;
  const double rpLength   = sqrt(rpLengthSq);
  const double prefactor  = dE_dTheta / fmax(1.0e-5, rpLength);

  // Calculate t factors
  const double t1 = -prefactor / r1LengthSq;
  const double t2 = prefactor / r2LengthSq;

  // Calculate cross products for gradients
  double dedp1x, dedp1y, dedp1z;  // r1 × rp
  crossProduct(r1x, r1y, r1z, rpx, rpy, rpz, dedp1x, dedp1y, dedp1z);

  double dedp3x, dedp3y, dedp3z;  // r2 × rp
  crossProduct(r2x, r2y, r2z, rpx, rpy, rpz, dedp3x, dedp3y, dedp3z);

  // Scale the cross products by t factors
  dedp1x *= t1;
  dedp1y *= t1;
  dedp1z *= t1;

  dedp3x *= t2;
  dedp3y *= t2;
  dedp3z *= t2;

  // Calculate middle point gradient as negative sum of other two
  const double dedp2x = -(dedp1x + dedp3x);
  const double dedp2y = -(dedp1y + dedp3y);
  const double dedp2z = -(dedp1z + dedp3z);

  // Accumulate gradients using atomic operations
  atomicAdd(&grad[posIdx1 + 0], dedp1x);
  atomicAdd(&grad[posIdx1 + 1], dedp1y);
  atomicAdd(&grad[posIdx1 + 2], dedp1z);

  atomicAdd(&grad[posIdx2 + 0], dedp2x);
  atomicAdd(&grad[posIdx2 + 1], dedp2y);
  atomicAdd(&grad[posIdx2 + 2], dedp2z);

  atomicAdd(&grad[posIdx3 + 0], dedp3x);
  atomicAdd(&grad[posIdx3 + 1], dedp3y);
  atomicAdd(&grad[posIdx3 + 2], dedp3z);
}

// Consolidated per-molecule ETK gradient calculation
// Note: grad pointer should already be offset to molecule start by caller
static __device__ __inline__ void molGradETK(const Energy3DForceContribsDevicePtr& terms,
                                              const BatchedIndices3DDevicePtr&      systemIndices,
                                              const double*                         coords,
                                              double*                               grad,
                                              const int                             molIdx,
                                              const int                             tid,
                                              const int                             stride) {
  const int     atomStart = systemIndices.atomStarts[molIdx];
  const double* molCoords = coords + atomStart * 4;  // ETK uses 4D coordinates

  // Experimental torsion terms
  const auto& [t_idx1s, t_idx2s, t_idx3s, t_idx4s, t_forceConstants, t_signs] = terms.experimentalTorsionTerms;
  const int torsionStart = systemIndices.experimentalTorsionTermStarts[molIdx];
  const int torsionEnd   = systemIndices.experimentalTorsionTermStarts[molIdx + 1];
  for (int i = torsionStart + tid; i < torsionEnd; i += stride) {
    const int localIdx1 = t_idx1s[i] - atomStart;
    const int localIdx2 = t_idx2s[i] - atomStart;
    const int localIdx3 = t_idx3s[i] - atomStart;
    const int localIdx4 = t_idx4s[i] - atomStart;
    torsionAngleGrad(molCoords, localIdx1, localIdx2, localIdx3, localIdx4, &t_forceConstants[i * 6], &t_signs[i * 6], grad);
  }

  // Improper torsion (inversion) terms
  const auto& [i_idx1s, i_idx2s, i_idx3s, i_idx4s, i_at2AtomicNum, i_isCBoundToO, i_C0, i_C1, i_C2, i_forceConstant] =
    terms.improperTorsionTerms;
  const int improperStart = systemIndices.improperTorsionTermStarts[molIdx];
  const int improperEnd   = systemIndices.improperTorsionTermStarts[molIdx + 1];
  for (int i = improperStart + tid; i < improperEnd; i += stride) {
    const int localIdx1 = i_idx1s[i] - atomStart;
    const int localIdx2 = i_idx2s[i] - atomStart;
    const int localIdx3 = i_idx3s[i] - atomStart;
    const int localIdx4 = i_idx4s[i] - atomStart;
    inversionGrad(molCoords, localIdx1, localIdx2, localIdx3, localIdx4, i_C0[i], i_C1[i], i_C2[i], i_forceConstant[i], grad);
  }

  // 1-2 distance terms
  const auto& [d12_idx1s, d12_idx2s, d12_minLen, d12_maxLen, d12_forceConstant] = terms.dist12Terms;
  const int dist12Start                                                         = systemIndices.dist12TermStarts[molIdx];
  const int dist12End                                                           = systemIndices.dist12TermStarts[molIdx + 1];
  for (int i = dist12Start + tid; i < dist12End; i += stride) {
    const int localIdx1 = d12_idx1s[i] - atomStart;
    const int localIdx2 = d12_idx2s[i] - atomStart;
    distanceConstraintGrad(molCoords, localIdx1, localIdx2, d12_minLen[i], d12_maxLen[i], d12_forceConstant[i], grad);
  }

  // 1-3 distance terms
  const auto& [d13_idx1s, d13_idx2s, d13_minLen, d13_maxLen, d13_forceConstant] = terms.dist13Terms;
  const int dist13Start                                                         = systemIndices.dist13TermStarts[molIdx];
  const int dist13End                                                           = systemIndices.dist13TermStarts[molIdx + 1];
  for (int i = dist13Start + tid; i < dist13End; i += stride) {
    const int localIdx1 = d13_idx1s[i] - atomStart;
    const int localIdx2 = d13_idx2s[i] - atomStart;
    distanceConstraintGrad(molCoords, localIdx1, localIdx2, d13_minLen[i], d13_maxLen[i], d13_forceConstant[i], grad);
  }

  // 1-3 angle terms
  const auto& [a13_idx1s, a13_idx2s, a13_idx3s, a13_minAngle, a13_maxAngle] = terms.angle13Terms;
  const int angle13Start                                                    = systemIndices.angle13TermStarts[molIdx];
  const int angle13End                                                      = systemIndices.angle13TermStarts[molIdx + 1];
  constexpr double defaultAngleForceConstant = 1.0;
  for (int i = angle13Start + tid; i < angle13End; i += stride) {
    const int localIdx1 = a13_idx1s[i] - atomStart;
    const int localIdx2 = a13_idx2s[i] - atomStart;
    const int localIdx3 = a13_idx3s[i] - atomStart;
    angleConstraintGrad(molCoords, localIdx1, localIdx2, localIdx3, a13_minAngle[i], a13_maxAngle[i], defaultAngleForceConstant, grad);
  }

  // Long range distance terms
  const auto& [dlr_idx1s, dlr_idx2s, dlr_minLen, dlr_maxLen, dlr_forceConstant] = terms.longRangeDistTerms;
  const int distLRStart                                                         = systemIndices.longRangeDistTermStarts[molIdx];
  const int distLREnd                                                           = systemIndices.longRangeDistTermStarts[molIdx + 1];
  for (int i = distLRStart + tid; i < distLREnd; i += stride) {
    const int localIdx1 = dlr_idx1s[i] - atomStart;
    const int localIdx2 = dlr_idx2s[i] - atomStart;
    distanceConstraintGrad(molCoords, localIdx1, localIdx2, dlr_minLen[i], dlr_maxLen[i], dlr_forceConstant[i], grad);
  }
}

}  // namespace DistGeom
}  // namespace nvMolKit

#endif  // NVMOLKIT_DISTGEOM_KERNELS_DEVICE_CUH

