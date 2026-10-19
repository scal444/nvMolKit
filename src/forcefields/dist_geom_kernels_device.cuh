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

// Instantiated by dist_geom_kernels_device_dispatch.cuh for each arithmetic type.
// DG_REAL and the DG_* math macros must be defined before inclusion.
// --------------
// DG terms
// --------------

//! Distance-violation energy from the squared distance.
static __device__ __forceinline__ DG_REAL distViolationEnergyD2(const DG_REAL distance2,
                                                                const DG_REAL lb2,
                                                                const DG_REAL ub2,
                                                                const DG_REAL weight) {
  DG_REAL val = DG_REAL(0.0);
  if (distance2 > ub2) {
    val = (distance2 / ub2) - DG_REAL(1.0);
  } else if (distance2 < lb2) {
    val = ((2 * lb2) / (lb2 + distance2)) - DG_REAL(1.0);
  }
  if (val > DG_REAL(0.0)) {
    return weight * val * val;
  }
  return DG_REAL(0.0);
}

template <int dimension, int posStride = dimension>
static __device__ __forceinline__ DG_REAL distViolationEnergy(const DG_REAL* pos,
                                                              const int      idx1,
                                                              const int      idx2,
                                                              const DG_REAL  lb2,
                                                              const DG_REAL  ub2,
                                                              const DG_REAL  weight) {
  const int     posIdx1   = idx1 * posStride;
  const int     posIdx2   = idx2 * posStride;
  const DG_REAL distance2 = distanceSquaredPosIdx<dimension>(pos, posIdx1, posIdx2);
  return distViolationEnergyD2(distance2, lb2, ub2, weight);
}

//! Per-term gradient of a DG distance violation: returns false for a satisfied term, otherwise writes the atom-1
//! contribution d (atom 2 receives -d). The factors are single-precision in every mode.
template <int dimension, int posStride = dimension>
static __device__ __forceinline__ bool distViolationGradTerm(const DG_REAL* pos,
                                                             const int      idx1,
                                                             const int      idx2,
                                                             const DG_REAL  lb2,
                                                             const DG_REAL  ub2,
                                                             const DG_REAL  weight,
                                                             float*         d) {
  const int   posIdx1   = idx1 * posStride;
  const int   posIdx2   = idx2 * posStride;
  const float distance2 = distanceSquaredPosIdx<dimension>(pos, posIdx1, posIdx2);
  float       preFactor = 0.0f;
  if (distance2 > ub2) {
    preFactor = 4.f * ((distance2 / ub2) - 1.0f) / ub2;
  } else if (distance2 < lb2) {
    const float l2d2 = distance2 + lb2;
    preFactor        = 8.f * lb2 * (1.f - 2.0f * lb2 / l2d2) / (l2d2 * l2d2);
  } else {
    return false;
  }
  d[0] = weight * preFactor * (pos[posIdx1 + 0] - pos[posIdx2 + 0]);
  d[1] = weight * preFactor * (pos[posIdx1 + 1] - pos[posIdx2 + 1]);
  d[2] = weight * preFactor * (pos[posIdx1 + 2] - pos[posIdx2 + 2]);
  if constexpr (dimension == 4) {
    d[3] = weight * preFactor * (pos[posIdx1 + 3] - pos[posIdx2 + 3]);
  }
  return true;
}

template <int dimension>
static __device__ __forceinline__ void distViolationGrad(const DG_REAL* pos,
                                                         const int      idx1,
                                                         const int      idx2,
                                                         const DG_REAL  lb2,
                                                         const DG_REAL  ub2,
                                                         const DG_REAL  weight,
                                                         DG_REAL*       grad) {
  float d[4];
  if (!distViolationGradTerm<dimension>(pos, idx1, idx2, lb2, ub2, weight, d)) {
    return;
  }
#pragma unroll
  for (int c = 0; c < dimension; ++c) {
    atomicAdd(&grad[idx1 * dimension + c], d[c]);
    atomicAdd(&grad[idx2 * dimension + c], -d[c]);
  }
}

template <typename T>
static __device__ __forceinline__ T calcChiralVolume(const int&     posIdx1,
                                                     const int&     posIdx2,
                                                     const int&     posIdx3,
                                                     const int&     posIdx4,
                                                     const DG_REAL* pos,
                                                     T&             v1x,
                                                     T&             v1y,
                                                     T&             v1z,
                                                     T&             v2x,
                                                     T&             v2y,
                                                     T&             v2z,
                                                     T&             v3x,
                                                     T&             v3y,
                                                     T&             v3z) {
  v1x = pos[posIdx1 + 0] - pos[posIdx4 + 0];
  v1y = pos[posIdx1 + 1] - pos[posIdx4 + 1];
  v1z = pos[posIdx1 + 2] - pos[posIdx4 + 2];

  v2x = pos[posIdx2 + 0] - pos[posIdx4 + 0];
  v2y = pos[posIdx2 + 1] - pos[posIdx4 + 1];
  v2z = pos[posIdx2 + 2] - pos[posIdx4 + 2];

  v3x = pos[posIdx3 + 0] - pos[posIdx4 + 0];
  v3y = pos[posIdx3 + 1] - pos[posIdx4 + 1];
  v3z = pos[posIdx3 + 2] - pos[posIdx4 + 2];

  T v2v3x, v2v3y, v2v3z;
  crossProduct(v2x, v2y, v2z, v3x, v3y, v3z, v2v3x, v2v3y, v2v3z);
  T vol = dotProduct(v1x, v1y, v1z, v2v3x, v2v3y, v2v3z);
  return vol;
}

template <int dimension>
static __device__ __forceinline__ DG_REAL chiralViolationEnergy(const DG_REAL* pos,
                                                                const int      idx1,
                                                                const int      idx2,
                                                                const int      idx3,
                                                                const int      idx4,
                                                                const DG_REAL  lb,
                                                                const DG_REAL  ub,
                                                                const DG_REAL  weight) {
  const int posIdx1 = idx1 * dimension;
  const int posIdx2 = idx2 * dimension;
  const int posIdx3 = idx3 * dimension;
  const int posIdx4 = idx4 * dimension;

  DG_REAL v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z;
  DG_REAL vol = calcChiralVolume(posIdx1, posIdx2, posIdx3, posIdx4, pos, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z);

  if (vol < lb) {
    return weight * (vol - lb) * (vol - lb);
  }
  if (vol > ub) {
    return weight * (vol - ub) * (vol - ub);
  }
  return DG_REAL(0.0);
}

template <int dimension>
static __device__ __forceinline__ void chiralViolationGrad(const DG_REAL* pos,
                                                           const int      idx1,
                                                           const int      idx2,
                                                           const int      idx3,
                                                           const int      idx4,
                                                           const DG_REAL  lb,
                                                           const DG_REAL  ub,
                                                           const DG_REAL  weight,
                                                           DG_REAL*       grad) {
  const int posIdx1 = idx1 * dimension;
  const int posIdx2 = idx2 * dimension;
  const int posIdx3 = idx3 * dimension;
  const int posIdx4 = idx4 * dimension;

  float v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z;
  float vol = calcChiralVolume(posIdx1, posIdx2, posIdx3, posIdx4, pos, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z);

  if (vol < lb || vol > ub) {
    float preFactor;
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

    float x1 = pos[posIdx1 + 0];
    float y1 = pos[posIdx1 + 1];
    float z1 = pos[posIdx1 + 2];
    float x2 = pos[posIdx2 + 0];
    float y2 = pos[posIdx2 + 1];
    float z2 = pos[posIdx2 + 2];
    float x3 = pos[posIdx3 + 0];
    float y3 = pos[posIdx3 + 1];
    float z3 = pos[posIdx3 + 2];
    atomicAdd(&grad[posIdx4 + 0], preFactor * (z1 * (y2 - y3) + z2 * (y3 - y1) + z3 * (y1 - y2)));
    atomicAdd(&grad[posIdx4 + 1], preFactor * (x1 * (z2 - z3) + x2 * (z3 - z1) + x3 * (z1 - z2)));
    atomicAdd(&grad[posIdx4 + 2], preFactor * (y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2)));
  }
}

template <int dimension>
static __device__ __forceinline__ DG_REAL fourthDimEnergy(const DG_REAL* pos, const int idx, const DG_REAL weight) {
  if constexpr (dimension != 4) {
    return DG_REAL(0.0);
  }
  const int     posIdx    = idx * dimension;
  const DG_REAL fourthVal = pos[posIdx + 3];
  return weight * fourthVal * fourthVal;
}

template <int dimension>
static __device__ __forceinline__ void fourthDimGrad(const DG_REAL* pos,
                                                     const int      idx,
                                                     const DG_REAL  weight,
                                                     DG_REAL*       grad) {
  if constexpr (dimension != 4) {
    return;
  }
  const int     posIdx    = idx * dimension;
  const DG_REAL fourthVal = pos[posIdx + 3];
  atomicAdd(&grad[posIdx + 3], weight * fourthVal);
}

// ----------------------
// ETK Terms
// ----------------------

static __device__ __forceinline__ float calcTorsionEnergyM6(const DG_PARAM_REAL* forceConstants,
                                                            const int*           signs,
                                                            const DG_REAL        cosPhi) {
  const float cosPhi2 = cosPhi * cosPhi;
  const float cosPhi3 = cosPhi * cosPhi2;
  const float cosPhi4 = cosPhi * cosPhi3;
  const float cosPhi5 = cosPhi * cosPhi4;
  const float cosPhi6 = cosPhi * cosPhi5;

  const float cos2Phi = 2.0f * cosPhi2 - 1.0f;
  const float cos3Phi = 4.0f * cosPhi3 - 3.0f * cosPhi;
  const float cos4Phi = 8.0f * cosPhi4 - 8.0f * cosPhi2 + 1.0f;
  const float cos5Phi = 16.0f * cosPhi5 - 20.0f * cosPhi3 + 5.0f * cosPhi;
  const float cos6Phi = 32.0f * cosPhi6 - 48.0f * cosPhi4 + 18.0f * cosPhi2 - 1.0f;

  return (forceConstants[0] * (1.0f + signs[0] * cosPhi) + forceConstants[1] * (1.0f + signs[1] * cos2Phi) +
          forceConstants[2] * (1.0f + signs[2] * cos3Phi) + forceConstants[3] * (1.0f + signs[3] * cos4Phi) +
          forceConstants[4] * (1.0f + signs[4] * cos5Phi) + forceConstants[5] * (1.0f + signs[5] * cos6Phi));
}

static __device__ __forceinline__ DG_REAL
calcTorsionCosPhi(const DG_REAL* pos, const int posIdx1, const int posIdx2, const int posIdx3, const int posIdx4) {
  DG_REAL r1x = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  DG_REAL r1y = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  DG_REAL r1z = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  DG_REAL r2x = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  DG_REAL r2y = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  DG_REAL r2z = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  DG_REAL r3x = pos[posIdx2 + 0] - pos[posIdx3 + 0];
  DG_REAL r3y = pos[posIdx2 + 1] - pos[posIdx3 + 1];
  DG_REAL r3z = pos[posIdx2 + 2] - pos[posIdx3 + 2];

  DG_REAL r4x = pos[posIdx4 + 0] - pos[posIdx3 + 0];
  DG_REAL r4y = pos[posIdx4 + 1] - pos[posIdx3 + 1];
  DG_REAL r4z = pos[posIdx4 + 2] - pos[posIdx3 + 2];

  DG_REAL t1x, t1y, t1z;
  crossProduct(r1x, r1y, r1z, r2x, r2y, r2z, t1x, t1y, t1z);

  DG_REAL t2x, t2y, t2z;
  crossProduct(r3x, r3y, r3z, r4x, r4y, r4z, t2x, t2y, t2z);

  const DG_REAL t1_lenSquared      = t1x * t1x + t1y * t1y + t1z * t1z;
  const DG_REAL t2_lenSquared      = t2x * t2x + t2y * t2y + t2z * t2z;
  const DG_REAL lenSquaredCombined = t1_lenSquared * t2_lenSquared;

  if (DG_IS_ZERO(lenSquaredCombined)) {
    return DG_REAL(0.0);
  }
  const DG_REAL invLenComb = DG_RSQRTF(lenSquaredCombined);
  DG_REAL       cosPhi     = dotProduct(t1x, t1y, t1z, t2x, t2y, t2z) * invLenComb;
  clipToOne(cosPhi);
  return cosPhi;
}

static __device__ __forceinline__ DG_REAL torsionAngleEnergy(const DG_REAL*       pos,
                                                             const int            idx1,
                                                             const int            idx2,
                                                             const int            idx3,
                                                             const int            idx4,
                                                             const DG_PARAM_REAL* forceConstants,
                                                             const int*           signs) {
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;
  const int posIdx4 = idx4 * 4;

  DG_REAL cosPhi = calcTorsionCosPhi(pos, posIdx1, posIdx2, posIdx3, posIdx4);
  return calcTorsionEnergyM6(forceConstants, signs, cosPhi);
}

static __device__ __forceinline__ float calcInversionCosY(const DG_REAL* pos,
                                                          const int      posIdx1,
                                                          const int      posIdx2,
                                                          const int      posIdx3,
                                                          const int      posIdx4) {
  constexpr float inversionZeroTol = 1.0e-16f;

  float rJIx = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  float rJIy = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  float rJIz = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  float rJKx = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  float rJKy = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  float rJKz = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  float rJLx = pos[posIdx4 + 0] - pos[posIdx2 + 0];
  float rJLy = pos[posIdx4 + 1] - pos[posIdx2 + 1];
  float rJLz = pos[posIdx4 + 2] - pos[posIdx2 + 2];

  float l2JI = rJIx * rJIx + rJIy * rJIy + rJIz * rJIz;
  float l2JK = rJKx * rJKx + rJKy * rJKy + rJKz * rJKz;
  float l2JL = rJLx * rJLx + rJLy * rJLy + rJLz * rJLz;

  if (l2JI < inversionZeroTol || l2JK < inversionZeroTol || l2JL < inversionZeroTol) {
    return 0.0f;
  }

  float nx, ny, nz;
  crossProduct(rJIx, rJIy, rJIz, rJKx, rJKy, rJKz, nx, ny, nz);

  const float norm_factor = DG_RSQRTF(l2JI * l2JK);
  nx *= norm_factor;
  ny *= norm_factor;
  nz *= norm_factor;

  float l2n = nx * nx + ny * ny + nz * nz;
  if (l2n < inversionZeroTol) {
    return 0.0f;
  }

  return dotProduct(nx, ny, nz, rJLx, rJLy, rJLz) * DG_RSQRTF(l2JL) * DG_RSQRTF(l2n);
}

static __device__ __forceinline__ DG_REAL inversionEnergy(const DG_REAL* pos,
                                                          const int      idx1,
                                                          const int      idx2,
                                                          const int      idx3,
                                                          const int      idx4,
                                                          const DG_REAL  C0,
                                                          const DG_REAL  C1,
                                                          const DG_REAL  C2,
                                                          const DG_REAL  forceConstant) {
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;
  const int posIdx4 = idx4 * 4;

  DG_REAL cosY = calcInversionCosY(pos, posIdx1, posIdx2, posIdx3, posIdx4);

  const DG_REAL sinYSq = DG_REAL(1.0) - cosY * cosY;
  const DG_REAL sinY   = ((sinYSq > DG_REAL(0.0)) ? DG_SQRTF(sinYSq) : DG_REAL(0.0));
  const DG_REAL cos2W  = DG_REAL(2.0) * sinY * sinY - DG_REAL(1.0);

  return forceConstant * (C0 + C1 * sinY + C2 * cos2W);
}

template <int posStride = 4>
static __device__ __forceinline__ DG_REAL distanceConstraintEnergy(const DG_REAL* pos,
                                                                   const int      idx1,
                                                                   const int      idx2,
                                                                   const DG_REAL  minLen,
                                                                   const DG_REAL  maxLen,
                                                                   const DG_REAL  forceConstant) {
  const int     posIdx1   = idx1 * posStride;
  const int     posIdx2   = idx2 * posStride;
  const DG_REAL distance2 = distanceSquaredPosIdx(pos, posIdx1, posIdx2, 3);

  const DG_REAL minLen2 = minLen * minLen;
  const DG_REAL maxLen2 = maxLen * maxLen;

  DG_REAL difference = DG_REAL(0.0);
  if (distance2 < minLen2) {
    difference = minLen - DG_SQRTF(distance2);
  } else if (distance2 > maxLen2) {
    difference = DG_SQRTF(distance2) - maxLen;
  } else {
    return DG_REAL(0.0);
  }

  return DG_REAL(0.5) * forceConstant * difference * difference;
}

static __device__ __forceinline__ DG_REAL computeAngleTerm(const DG_REAL angle,
                                                           const DG_REAL minAngle,
                                                           const DG_REAL maxAngle) {
  DG_REAL angleTerm = DG_REAL(0.0);
  if (angle < minAngle) {
    angleTerm = angle - minAngle;
  } else if (angle > maxAngle) {
    angleTerm = angle - maxAngle;
  }
  return angleTerm;
}

static __device__ __forceinline__ DG_REAL angleConstraintEnergy(const DG_REAL* pos,
                                                                const int      idx1,
                                                                const int      idx2,
                                                                const int      idx3,
                                                                const DG_REAL  minAngle,
                                                                const DG_REAL  maxAngle,
                                                                const DG_REAL  forceConstant) {
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;

  DG_REAL dx1 = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  DG_REAL dy1 = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  DG_REAL dz1 = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  DG_REAL dx2 = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  DG_REAL dy2 = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  DG_REAL dz2 = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  const DG_REAL dist1Sq  = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
  const DG_REAL dist2Sq  = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
  const DG_REAL distTerm = dist1Sq * dist2Sq;
  if (DG_IS_ZERO(distTerm)) {
    return DG_REAL(0.0);
  }

  const DG_REAL dot      = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const DG_REAL cosTheta = clamp(dot * DG_RSQRT(distTerm), -DG_REAL(1.0), DG_REAL(1.0));
  const DG_REAL angle    = RAD2DEG * DG_ACOS(cosTheta);

  const DG_REAL angleTerm = computeAngleTerm(angle, minAngle, maxAngle);
  return forceConstant * angleTerm * angleTerm;
}

static __device__ __forceinline__ void torsionAngleGrad(const DG_REAL*       pos,
                                                        const int            idx1,
                                                        const int            idx2,
                                                        const int            idx3,
                                                        const int            idx4,
                                                        const DG_PARAM_REAL* forceConstants,  // 6 components
                                                        const int*           signs,           // 6 components
                                                        DG_REAL*             grad) {
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;
  const int posIdx4 = idx4 * 4;

  // Calculate bond vectors r1 = p1-p2, r2 = p3-p2
  const DG_REAL r1x = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  const DG_REAL r1y = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  const DG_REAL r1z = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  const DG_REAL r2x = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  const DG_REAL r2y = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  const DG_REAL r2z = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  // Calculate bond vectors r3 = p2-p3, r4 = p4-p3
  const DG_REAL r3x = -r2x;
  const DG_REAL r3y = -r2y;
  const DG_REAL r3z = -r2z;

  const DG_REAL r4x = pos[posIdx4 + 0] - pos[posIdx3 + 0];
  const DG_REAL r4y = pos[posIdx4 + 1] - pos[posIdx3 + 1];
  const DG_REAL r4z = pos[posIdx4 + 2] - pos[posIdx3 + 2];

  // Calculate plane normals via cross products: t0 = r1 × r2, t1 = r3 × r4
  DG_REAL t0x, t0y, t0z;
  crossProduct(r1x, r1y, r1z, r2x, r2y, r2z, t0x, t0y, t0z);

  DG_REAL t1x, t1y, t1z;
  crossProduct(r3x, r3y, r3z, r4x, r4y, r4z, t1x, t1y, t1z);

  // Calculate lengths and check for degeneracy
  const DG_REAL d02 = t0x * t0x + t0y * t0y + t0z * t0z;
  const DG_REAL d12 = t1x * t1x + t1y * t1y + t1z * t1z;
  if (DG_IS_ZERO(d02) || DG_IS_ZERO(d12)) {
    return;
  }
  const DG_REAL inv_d0 = DG_RSQRT(t0x * t0x + t0y * t0y + t0z * t0z);
  const DG_REAL inv_d1 = DG_RSQRT(t1x * t1x + t1y * t1y + t1z * t1z);

  t0x *= inv_d0;
  t0y *= inv_d0;
  t0z *= inv_d0;
  t1x *= inv_d1;
  t1y *= inv_d1;
  t1z *= inv_d1;

  // Calculate cosine of torsion angle
  DG_REAL cosPhi = dotProduct(t0x, t0y, t0z, t1x, t1y, t1z);
  clipToOne(cosPhi);

  // Calculate sinPhi
  const DG_REAL sinPhiSq = DG_REAL(1.0) - cosPhi * cosPhi;
  const DG_REAL sinPhi   = (sinPhiSq > DG_REAL(0.0)) ? DG_SQRTF(sinPhiSq) : DG_REAL(0.0);

  // Calculate derivatives for dE/dPhi
  const DG_REAL cosPhi2 = cosPhi * cosPhi;
  const DG_REAL cosPhi3 = cosPhi * cosPhi2;
  const DG_REAL cosPhi4 = cosPhi * cosPhi3;
  const DG_REAL cosPhi5 = cosPhi * cosPhi4;

  const DG_REAL dE_dPhi =
    (-forceConstants[0] * signs[0] * sinPhi -
     DG_REAL(2.0) * forceConstants[1] * signs[1] * (DG_REAL(2.0) * cosPhi * sinPhi) -
     DG_REAL(3.0) * forceConstants[2] * signs[2] * (DG_REAL(4.0) * cosPhi2 * sinPhi - sinPhi) -
     DG_REAL(4.0) * forceConstants[3] * signs[3] * (DG_REAL(8.0) * cosPhi3 * sinPhi - DG_REAL(4.0) * cosPhi * sinPhi) -
     DG_REAL(5.0) * forceConstants[4] * signs[4] *
       (DG_REAL(16.0) * cosPhi4 * sinPhi - DG_REAL(12.0) * cosPhi2 * sinPhi + sinPhi) -
     DG_REAL(6.0) * forceConstants[5] * signs[5] *
       (DG_REAL(32.0) * cosPhi5 * sinPhi - DG_REAL(32.0) * cosPhi3 * sinPhi + DG_REAL(6.0) * cosPhi * sinPhi));

  const DG_REAL sinTerm = -dE_dPhi * (DG_IS_ZERO(sinPhi) ? (DG_REAL(1.0) / cosPhi) : (DG_REAL(1.0) / sinPhi));

  // Calculate dCos_dT components on-the-fly and compute gradients directly
  const DG_REAL dCos_dT0x = inv_d0 * (t1x - cosPhi * t0x);
  const DG_REAL dCos_dT0y = inv_d0 * (t1y - cosPhi * t0y);
  const DG_REAL dCos_dT0z = inv_d0 * (t1z - cosPhi * t0z);
  const DG_REAL dCos_dT1x = inv_d1 * (t0x - cosPhi * t1x);
  const DG_REAL dCos_dT1y = inv_d1 * (t0y - cosPhi * t1y);
  const DG_REAL dCos_dT1z = inv_d1 * (t0z - cosPhi * t1z);

  // Atom 1 gradient: grad1 = sinTerm * (dCos_dT0 × r2)
  const DG_REAL g1x = sinTerm * (dCos_dT0z * r2y - dCos_dT0y * r2z);
  const DG_REAL g1y = sinTerm * (dCos_dT0x * r2z - dCos_dT0z * r2x);
  const DG_REAL g1z = sinTerm * (dCos_dT0y * r2x - dCos_dT0x * r2y);

  // Atom 4 gradient: grad4 = sinTerm * (dCos_dT1 × r3)
  const DG_REAL g4x = sinTerm * (dCos_dT1y * r3z - dCos_dT1z * r3y);
  const DG_REAL g4y = sinTerm * (dCos_dT1z * r3x - dCos_dT1x * r3z);
  const DG_REAL g4z = sinTerm * (dCos_dT1x * r3y - dCos_dT1y * r3x);

  // Atom 2 gradient: more complex, involves both cross products
  const DG_REAL g2x =
    sinTerm * (dCos_dT0y * (r2z - r1z) + dCos_dT0z * (r1y - r2y) + dCos_dT1y * (-r4z) + dCos_dT1z * (r4y));
  const DG_REAL g2y =
    sinTerm * (dCos_dT0x * (r1z - r2z) + dCos_dT0z * (r2x - r1x) + dCos_dT1x * (r4z) + dCos_dT1z * (-r4x));
  const DG_REAL g2z =
    sinTerm * (dCos_dT0x * (r2y - r1y) + dCos_dT0y * (r1x - r2x) + dCos_dT1x * (-r4y) + dCos_dT1y * (r4x));

  // Atom 3 gradient: grad3 = -(grad1 + grad2 + grad4) by conservation
  const DG_REAL g3x =
    sinTerm * (dCos_dT0y * r1z + dCos_dT0z * (-r1y) + dCos_dT1y * (r4z - r3z) + dCos_dT1z * (r3y - r4y));
  const DG_REAL g3y =
    sinTerm * (dCos_dT0x * (-r1z) + dCos_dT0z * r1x + dCos_dT1x * (r3z - r4z) + dCos_dT1z * (r4x - r3x));
  const DG_REAL g3z =
    sinTerm * (dCos_dT0x * r1y + dCos_dT0y * (-r1x) + dCos_dT1x * (r4y - r3y) + dCos_dT1y * (r3x - r4x));

  // Add gradients using atomic operations
  atomicAdd(&grad[posIdx1 + 0], g1x);
  atomicAdd(&grad[posIdx1 + 1], g1y);
  atomicAdd(&grad[posIdx1 + 2], g1z);
  atomicAdd(&grad[posIdx2 + 0], g2x);
  atomicAdd(&grad[posIdx2 + 1], g2y);
  atomicAdd(&grad[posIdx2 + 2], g2z);
  atomicAdd(&grad[posIdx3 + 0], g3x);
  atomicAdd(&grad[posIdx3 + 1], g3y);
  atomicAdd(&grad[posIdx3 + 2], g3z);
  atomicAdd(&grad[posIdx4 + 0], g4x);
  atomicAdd(&grad[posIdx4 + 1], g4y);
  atomicAdd(&grad[posIdx4 + 2], g4z);
}

static __device__ __forceinline__ void inversionGrad(const DG_REAL* pos,
                                                     const int      idx1,
                                                     const int      idx2,
                                                     const int      idx3,
                                                     const int      idx4,
                                                     const DG_REAL  C0,
                                                     const DG_REAL  C1,
                                                     const DG_REAL  C2,
                                                     const DG_REAL  forceConstant,
                                                     DG_REAL*       grad) {
  // Get positions for all four atoms
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;
  const int posIdx4 = idx4 * 4;

  // Calculate vectors
  DG_REAL rJIx = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  DG_REAL rJIy = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  DG_REAL rJIz = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  DG_REAL rJKx = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  DG_REAL rJKy = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  DG_REAL rJKz = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  DG_REAL rJLx = pos[posIdx4 + 0] - pos[posIdx2 + 0];
  DG_REAL rJLy = pos[posIdx4 + 1] - pos[posIdx2 + 1];
  DG_REAL rJLz = pos[posIdx4 + 2] - pos[posIdx2 + 2];

  const DG_REAL dJIsq = rJIx * rJIx + rJIy * rJIy + rJIz * rJIz;
  const DG_REAL dJKsq = rJKx * rJKx + rJKy * rJKy + rJKz * rJKz;
  const DG_REAL dJLsq = rJLx * rJLx + rJLy * rJLy + rJLz * rJLz;
  if (DG_IS_ZERO(dJIsq) || DG_IS_ZERO(dJKsq) || DG_IS_ZERO(dJLsq)) {
    return;
  }
  // Calculate lengths
  float invdJI = DG_RSQRTF(dJIsq);
  float invdJK = DG_RSQRTF(dJKsq);
  float invdJL = DG_RSQRTF(dJLsq);

  // Normalize vectors
  rJIx *= invdJI;
  rJIy *= invdJI;
  rJIz *= invdJI;
  rJKx *= invdJK;
  rJKy *= invdJK;
  rJKz *= invdJK;
  rJLx *= invdJL;
  rJLy *= invdJL;
  rJLz *= invdJL;

  // Calculate n = (-rJI) × rJK
  DG_REAL nx, ny, nz;
  crossProduct(-rJIx, -rJIy, -rJIz, rJKx, rJKy, rJKz, nx, ny, nz);

  // Normalize n
  DG_REAL inv_n_len = DG_RSQRTF(nx * nx + ny * ny + nz * nz);
  nx *= inv_n_len;
  ny *= inv_n_len;
  nz *= inv_n_len;

  // Calculate cosY and clamp
  DG_REAL cosY = dotProduct(nx, ny, nz, rJLx, rJLy, rJLz);
  clipToOne(cosY);

  // Calculate sinY
  const DG_REAL sinYSq  = DG_REAL(1.0) - cosY * cosY;
  const DG_REAL rawSinY = DG_SQRTF(sinYSq);
  const DG_REAL sinY    = rawSinY > DG_REAL(1.0e-8) ? rawSinY : DG_REAL(1.0e-8);

  // Calculate cosTheta and clamp
  DG_REAL cosTheta = dotProduct(rJIx, rJIy, rJIz, rJKx, rJKy, rJKz);
  clipToOne(cosTheta);

  // Calculate sinTheta
  const DG_REAL sinThetaSq  = DG_REAL(1.0) - cosTheta * cosTheta;
  const DG_REAL rawSinTheta = DG_SQRTF(sinThetaSq);
  const DG_REAL sinTheta    = rawSinTheta > DG_REAL(1.0e-8) ? rawSinTheta : DG_REAL(1.0e-8);

  // Calculate dE_dW
  const DG_REAL dE_dW = -forceConstant * (C1 * cosY - DG_REAL(4.0) * C2 * cosY * sinY);

  // Calculate cross products for gradient terms
  DG_REAL t1x, t1y, t1z;  // rJL × rJK
  crossProduct(rJLx, rJLy, rJLz, rJKx, rJKy, rJKz, t1x, t1y, t1z);

  DG_REAL t2x, t2y, t2z;  // rJI × rJL
  crossProduct(rJIx, rJIy, rJIz, rJLx, rJLy, rJLz, t2x, t2y, t2z);

  DG_REAL t3x, t3y, t3z;  // rJK × rJI
  crossProduct(rJKx, rJKy, rJKz, rJIx, rJIy, rJIz, t3x, t3y, t3z);

  // Calculate terms for gradient
  const DG_REAL inverseTerm1   = DG_REAL(1.0) / (sinY * sinTheta);
  const DG_REAL term2          = cosY / (sinY * sinThetaSq);
  const DG_REAL cosY_over_sinY = cosY / sinY;

  // Compute gradient components on-the-fly and apply directly
  // Atom 1 gradient components
  const DG_REAL tg1x = (t1x * inverseTerm1 - (rJIx - rJKx * cosTheta) * term2) * invdJI;
  const DG_REAL tg1y = (t1y * inverseTerm1 - (rJIy - rJKy * cosTheta) * term2) * invdJI;
  const DG_REAL tg1z = (t1z * inverseTerm1 - (rJIz - rJKz * cosTheta) * term2) * invdJI;

  // Atom 3 gradient components
  const DG_REAL tg3x = (t2x * inverseTerm1 - (rJKx - rJIx * cosTheta) * term2) * invdJK;
  const DG_REAL tg3y = (t2y * inverseTerm1 - (rJKy - rJIy * cosTheta) * term2) * invdJK;
  const DG_REAL tg3z = (t2z * inverseTerm1 - (rJKz - rJIz * cosTheta) * term2) * invdJK;

  // Atom 4 gradient components
  const DG_REAL tg4x = (t3x * inverseTerm1 - rJLx * cosY_over_sinY) * invdJL;
  const DG_REAL tg4y = (t3y * inverseTerm1 - rJLy * cosY_over_sinY) * invdJL;
  const DG_REAL tg4z = (t3z * inverseTerm1 - rJLz * cosY_over_sinY) * invdJL;

  // Add gradients using atomic operations
  atomicAdd(&grad[posIdx1 + 0], dE_dW * tg1x);
  atomicAdd(&grad[posIdx1 + 1], dE_dW * tg1y);
  atomicAdd(&grad[posIdx1 + 2], dE_dW * tg1z);

  atomicAdd(&grad[posIdx2 + 0], -dE_dW * (tg1x + tg3x + tg4x));
  atomicAdd(&grad[posIdx2 + 1], -dE_dW * (tg1y + tg3y + tg4y));
  atomicAdd(&grad[posIdx2 + 2], -dE_dW * (tg1z + tg3z + tg4z));

  atomicAdd(&grad[posIdx3 + 0], dE_dW * tg3x);
  atomicAdd(&grad[posIdx3 + 1], dE_dW * tg3y);
  atomicAdd(&grad[posIdx3 + 2], dE_dW * tg3z);

  atomicAdd(&grad[posIdx4 + 0], dE_dW * tg4x);
  atomicAdd(&grad[posIdx4 + 1], dE_dW * tg4y);
  atomicAdd(&grad[posIdx4 + 2], dE_dW * tg4z);
}

//! Per-term gradient of a flat-bottom distance restraint: returns false inside the bounds, otherwise writes the
//! atom-1 xyz contribution d (atom 2 receives -d).
template <int posStride = 4>
static __device__ __forceinline__ bool distanceConstraintGradTerm(const DG_REAL* pos,
                                                                  const int      idx1,
                                                                  const int      idx2,
                                                                  const DG_REAL  minLen,
                                                                  const DG_REAL  maxLen,
                                                                  const DG_REAL  forceConstant,
                                                                  DG_REAL*       d) {
  const DG_REAL minLen2 = minLen * minLen;
  const DG_REAL maxLen2 = maxLen * maxLen;
  const int     posIdx1 = idx1 * posStride;
  const int     posIdx2 = idx2 * posStride;

  // Calculate squared distance
  const DG_REAL distance2 = distanceSquaredPosIdx(pos, posIdx1, posIdx2, 3);

  // Check if distance is outside bounds and calculate preFactor
  // Both violated bounds share one sqrt/division site (same operations as separate branches).
  const bool lower = distance2 < minLen2;
  if (!lower && !(distance2 > maxLen2)) {
    return false;  // Distance within bounds, no gradient contribution
  }
  const DG_REAL distance  = DG_SQRT(distance2);
  const DG_REAL preFactor = forceConstant * (distance - (lower ? minLen : maxLen)) / DG_FMAX(DG_REAL(1.0e-8), distance);
  for (int i = 0; i < 3; i++) {
    d[i] = preFactor * (pos[posIdx1 + i] - pos[posIdx2 + i]);
  }
  return true;
}

static __device__ __forceinline__ void distanceConstraintGrad(const DG_REAL* pos,
                                                              const int      idx1,
                                                              const int      idx2,
                                                              const DG_REAL  minLen,
                                                              const DG_REAL  maxLen,
                                                              const DG_REAL  forceConstant,
                                                              DG_REAL*       grad) {
  DG_REAL d[3];
  if (!distanceConstraintGradTerm(pos, idx1, idx2, minLen, maxLen, forceConstant, d)) {
    return;
  }
  for (int i = 0; i < 3; i++) {
    atomicAdd(&grad[idx1 * 4 + i], d[i]);
    atomicAdd(&grad[idx2 * 4 + i], -d[i]);
  }
}

// Angle constraint gradient
static __device__ __forceinline__ void angleConstraintGrad(const DG_REAL* pos,
                                                           const int      idx1,
                                                           const int      idx2,
                                                           const int      idx3,
                                                           const DG_REAL  minAngle,
                                                           const DG_REAL  maxAngle,
                                                           const DG_REAL  forceConstant,
                                                           DG_REAL*       grad) {
  // Get positions for all three atoms
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;

  // Calculate vectors r1 = p1 - p2 and r2 = p3 - p2
  DG_REAL r1x = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  DG_REAL r1y = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  DG_REAL r1z = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  DG_REAL r2x = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  DG_REAL r2y = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  DG_REAL r2z = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  // Calculate squared lengths and take max with 1.0e-5 as in RDKit
  const DG_REAL r1LengthSq = DG_FMAX(DG_REAL(1.0e-5), r1x * r1x + r1y * r1y + r1z * r1z);
  const DG_REAL r2LengthSq = DG_FMAX(DG_REAL(1.0e-5), r2x * r2x + r2y * r2y + r2z * r2z);
  const DG_REAL denom      = DG_RSQRT(r1LengthSq * r2LengthSq);

  // Calculate cosine of angle using dot product
  DG_REAL cosTheta = dotProduct(r1x, r1y, r1z, r2x, r2y, r2z) * denom;

  // Clamp cosTheta to [-1, 1]
  clipToOne(cosTheta);

  // Convert to degrees using RDKit's RAD2DEG constant
  const DG_REAL angle = RAD2DEG * DG_ACOS(cosTheta);

  // Calculate angle term using the separate device function
  const DG_REAL angleTerm = computeAngleTerm(angle, minAngle, maxAngle);

  // Calculate dE_dTheta
  const DG_REAL dE_dTheta = DG_REAL(2.0) * RAD2DEG * forceConstant * angleTerm;

  // Calculate cross product rp = r2 × r1
  DG_REAL rpx, rpy, rpz;
  crossProduct(r2x, r2y, r2z, r1x, r1y, r1z, rpx, rpy, rpz);

  // Calculate length of rp and prefactor
  const DG_REAL rpLengthSq  = DG_FMAX(rpx * rpx + rpy * rpy + rpz * rpz, 1e-10);
  const DG_REAL rpLengthInv = DG_RSQRT(rpLengthSq);
  const DG_REAL prefactor   = dE_dTheta * rpLengthInv;

  // Calculate t factors
  const DG_REAL t1 = -prefactor / r1LengthSq;
  const DG_REAL t2 = prefactor / r2LengthSq;

  // Calculate cross products for gradients
  DG_REAL dedp1x, dedp1y, dedp1z;  // r1 × rp
  crossProduct(r1x, r1y, r1z, rpx, rpy, rpz, dedp1x, dedp1y, dedp1z);

  DG_REAL dedp3x, dedp3y, dedp3z;  // r2 × rp
  crossProduct(r2x, r2y, r2z, rpx, rpy, rpz, dedp3x, dedp3y, dedp3z);

  // Scale the cross products by t factors
  dedp1x *= t1;
  dedp1y *= t1;
  dedp1z *= t1;

  dedp3x *= t2;
  dedp3y *= t2;
  dedp3z *= t2;

  // Calculate middle point gradient as negative sum of other two
  const DG_REAL dedp2x = -(dedp1x + dedp3x);
  const DG_REAL dedp2y = -(dedp1y + dedp3y);
  const DG_REAL dedp2z = -(dedp1z + dedp3z);

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

template <int dimension>
static __device__ __inline__ DG_REAL molEnergyDG(const EnergyForceContribsDevicePtr& terms,
                                                 const BatchedIndicesDevicePtr&      systemIndices,
                                                 const DG_REAL*                      molCoords,
                                                 const int                           molIdx,
                                                 const DG_REAL                       chiralWeight,
                                                 const DG_REAL                       fourthDimWeight,
                                                 const int                           tid) {
  const int atomStart = systemIndices.atomStarts[molIdx];

  DG_REAL energy = DG_REAL(0.0);

  namespace cg            = cooperative_groups;
  constexpr int WARP_SIZE = 32;
  auto          tile32    = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
  const int     laneId    = tile32.thread_rank();
  const int     warpId    = mark_warp_uniform(tile32.meta_group_rank());
  const int     numWarps  = mark_warp_uniform(tile32.meta_group_size());

  // Get term ranges
  const int distStart   = systemIndices.distTermStarts[molIdx];
  const int distEnd     = systemIndices.distTermStarts[molIdx + 1];
  const int chiralStart = systemIndices.chiralTermStarts[molIdx];
  const int chiralEnd   = systemIndices.chiralTermStarts[molIdx + 1];
  const int fourthStart = systemIndices.fourthTermStarts[molIdx];
  const int fourthEnd   = systemIndices.fourthTermStarts[molIdx + 1];

  // Get term data
  const auto& [d_idx1s, d_idx2s, d_ub2s, d_lb2s, d_weights]                  = terms.distTerms;
  const auto& [c_idx1s, c_idx2s, c_idx3s, c_idx4s, c_volUppers, c_volLowers] = terms.chiralTerms;
  const auto& [f_idxs]                                                       = terms.fourthTerms;

  const int numDist   = distEnd - distStart;
  const int numChiral = chiralEnd - chiralStart;
  const int numFourth = fourthEnd - fourthStart;

  // Calculate number of warps needed for each term type
  const int warpsForDist     = (numDist + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForChiral   = (numChiral + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForFourth   = (numFourth + WARP_SIZE - 1) / WARP_SIZE;
  const int totalWarpsNeeded = warpsForDist + warpsForChiral + warpsForFourth;

  // Each warp processes chunks in round-robin fashion
  for (int chunkIdx = warpId; chunkIdx < totalWarpsNeeded; chunkIdx += numWarps) {
    // Determine which term type this chunk belongs to
    if (chunkIdx < warpsForDist) {
      // Distance terms
      const int baseIdx = chunkIdx * WARP_SIZE;
      const int termIdx = distStart + baseIdx + laneId;
      if (baseIdx + laneId < numDist) {
        const int localIdx1 = d_idx1s[termIdx] - atomStart;
        const int localIdx2 = d_idx2s[termIdx] - atomStart;
        energy += distViolationEnergy<dimension>(molCoords,
                                                 localIdx1,
                                                 localIdx2,
                                                 d_lb2s[termIdx],
                                                 d_ub2s[termIdx],
                                                 d_weights[termIdx]);
      }
    } else if (chunkIdx < warpsForDist + warpsForChiral) {
      // Chiral terms
      const int warpOffset = chunkIdx - warpsForDist;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = chiralStart + baseIdx + laneId;
      if (baseIdx + laneId < numChiral) {
        const int localIdx1 = c_idx1s[termIdx] - atomStart;
        const int localIdx2 = c_idx2s[termIdx] - atomStart;
        const int localIdx3 = c_idx3s[termIdx] - atomStart;
        const int localIdx4 = c_idx4s[termIdx] - atomStart;
        energy += chiralViolationEnergy<dimension>(molCoords,
                                                   localIdx1,
                                                   localIdx2,
                                                   localIdx3,
                                                   localIdx4,
                                                   c_volLowers[termIdx],
                                                   c_volUppers[termIdx],
                                                   chiralWeight);
      }
    } else {
      // Fourth dimension terms
      const int warpOffset = chunkIdx - warpsForDist - warpsForChiral;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = fourthStart + baseIdx + laneId;
      if (baseIdx + laneId < numFourth) {
        const int localIdx = f_idxs[termIdx] - atomStart;
        energy += fourthDimEnergy<dimension>(molCoords, localIdx, fourthDimWeight);
      }
    }
  }

  return energy;
}

// Consolidated per-molecule gradient calculation
template <int dimension>
static __device__ __inline__ void molGradDG(const EnergyForceContribsDevicePtr& terms,
                                            const BatchedIndicesDevicePtr&      systemIndices,
                                            const DG_REAL*                      molCoords,
                                            DG_REAL*                            molGrad,
                                            const int                           molIdx,
                                            const DG_REAL                       chiralWeight,
                                            const DG_REAL                       fourthDimWeight,
                                            const int                           tid) {
  const int atomStart = systemIndices.atomStarts[molIdx];

  namespace cg            = cooperative_groups;
  constexpr int WARP_SIZE = 32;
  auto          tile32    = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
  const int     laneId    = tile32.thread_rank();
  const int     warpId    = mark_warp_uniform(tile32.meta_group_rank());
  const int     numWarps  = mark_warp_uniform(tile32.meta_group_size());

  // Get term ranges
  const int distStart   = systemIndices.distTermStarts[molIdx];
  const int distEnd     = systemIndices.distTermStarts[molIdx + 1];
  const int chiralStart = systemIndices.chiralTermStarts[molIdx];
  const int chiralEnd   = systemIndices.chiralTermStarts[molIdx + 1];
  const int fourthStart = systemIndices.fourthTermStarts[molIdx];
  const int fourthEnd   = systemIndices.fourthTermStarts[molIdx + 1];

  // Get term data
  const auto& [d_idx1s, d_idx2s, d_ub2s, d_lb2s, d_weights]                  = terms.distTerms;
  const auto& [c_idx1s, c_idx2s, c_idx3s, c_idx4s, c_volUppers, c_volLowers] = terms.chiralTerms;
  const auto& [f_idxs]                                                       = terms.fourthTerms;

  const int numDist   = distEnd - distStart;
  const int numChiral = chiralEnd - chiralStart;
  const int numFourth = fourthEnd - fourthStart;

  // Calculate number of warps needed for each term type
  const int warpsForDist     = (numDist + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForChiral   = (numChiral + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForFourth   = (numFourth + WARP_SIZE - 1) / WARP_SIZE;
  const int totalWarpsNeeded = warpsForDist + warpsForChiral + warpsForFourth;

  // Each warp processes chunks in round-robin fashion
  for (int chunkIdx = warpId; chunkIdx < totalWarpsNeeded; chunkIdx += numWarps) {
    // Determine which term type this chunk belongs to
    if (chunkIdx < warpsForDist) {
      // Distance terms
      const int baseIdx = chunkIdx * WARP_SIZE;
      const int termIdx = distStart + baseIdx + laneId;
      if (baseIdx + laneId < numDist) {
        const int localIdx1 = d_idx1s[termIdx] - atomStart;
        const int localIdx2 = d_idx2s[termIdx] - atomStart;
        distViolationGrad<dimension>(molCoords,
                                     localIdx1,
                                     localIdx2,
                                     d_lb2s[termIdx],
                                     d_ub2s[termIdx],
                                     d_weights[termIdx],
                                     molGrad);
      }
    } else if (chunkIdx < warpsForDist + warpsForChiral) {
      // Chiral terms
      const int warpOffset = chunkIdx - warpsForDist;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = chiralStart + baseIdx + laneId;
      if (baseIdx + laneId < numChiral) {
        const int localIdx1 = c_idx1s[termIdx] - atomStart;
        const int localIdx2 = c_idx2s[termIdx] - atomStart;
        const int localIdx3 = c_idx3s[termIdx] - atomStart;
        const int localIdx4 = c_idx4s[termIdx] - atomStart;
        chiralViolationGrad<dimension>(molCoords,
                                       localIdx1,
                                       localIdx2,
                                       localIdx3,
                                       localIdx4,
                                       c_volLowers[termIdx],
                                       c_volUppers[termIdx],
                                       chiralWeight,
                                       molGrad);
      }
    } else {
      // Fourth dimension terms
      const int warpOffset = chunkIdx - warpsForDist - warpsForChiral;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = fourthStart + baseIdx + laneId;
      if (baseIdx + laneId < numFourth) {
        const int localIdx = f_idxs[termIdx] - atomStart;
        fourthDimGrad<dimension>(molCoords, localIdx, fourthDimWeight, molGrad);
      }
    }
  }
}

static __device__ __inline__ DG_REAL molEnergyETK(const Energy3DForceContribsDevicePtr& terms,
                                                  const BatchedIndices3DDevicePtr&      systemIndices,
                                                  const DG_REAL*                        molCoords,
                                                  const int                             molIdx,
                                                  const int                             tid) {
  const int atomStart = systemIndices.atomStarts[molIdx];

  DG_REAL energy = DG_REAL(0.0);

  namespace cg            = cooperative_groups;
  constexpr int WARP_SIZE = 32;
  auto          tile32    = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

  const int laneId   = tile32.thread_rank();
  const int warpId   = mark_warp_uniform(tile32.meta_group_rank());
  const int numWarps = mark_warp_uniform(tile32.meta_group_size());

  // Get term ranges
  const int torsionStart  = systemIndices.experimentalTorsionTermStarts[molIdx];
  const int torsionEnd    = systemIndices.experimentalTorsionTermStarts[molIdx + 1];
  const int improperStart = systemIndices.improperTorsionTermStarts[molIdx];
  const int improperEnd   = systemIndices.improperTorsionTermStarts[molIdx + 1];
  const int dist12Start   = systemIndices.dist12TermStarts[molIdx];
  const int dist12End     = systemIndices.dist12TermStarts[molIdx + 1];
  const int dist13Start   = systemIndices.dist13TermStarts[molIdx];
  const int dist13End     = systemIndices.dist13TermStarts[molIdx + 1];
  const int angle13Start  = systemIndices.angle13TermStarts[molIdx];
  const int angle13End    = systemIndices.angle13TermStarts[molIdx + 1];
  const int distLRStart   = systemIndices.longRangeDistTermStarts[molIdx];
  const int distLREnd     = systemIndices.longRangeDistTermStarts[molIdx + 1];

  const int numTorsion  = torsionEnd - torsionStart;
  const int numImproper = improperEnd - improperStart;
  const int numDist12   = dist12End - dist12Start;
  const int numDist13   = dist13End - dist13Start;
  const int numAngle13  = angle13End - angle13Start;
  const int numDistLR   = distLREnd - distLRStart;

  // Get term data
  const auto& [t_idx1s, t_idx2s, t_idx3s, t_idx4s, t_forceConstants, t_signs] = terms.experimentalTorsionTerms;
  const auto& [i_idx1s, i_idx2s, i_idx3s, i_idx4s, i_at2AtomicNum, i_isCBoundToO, i_C0, i_C1, i_C2, i_forceConstant] =
    terms.improperTorsionTerms;
  const auto& [d12_idx1s, d12_idx2s, d12_minLen, d12_maxLen, d12_forceConstant] = terms.dist12Terms;
  const auto& [d13_idx1s, d13_idx2s, d13_minLen, d13_maxLen, d13_forceConstant] = terms.dist13Terms;
  const auto& [a13_idx1s, a13_idx2s, a13_idx3s, a13_minAngle, a13_maxAngle]     = terms.angle13Terms;
  const auto& [dlr_idx1s, dlr_idx2s, dlr_minLen, dlr_maxLen, dlr_forceConstant] = terms.longRangeDistTerms;

  constexpr DG_REAL defaultAngleForceConstant = DG_REAL(1.0);

  // Calculate number of warps needed for each term type
  const int warpsForTorsion  = (numTorsion + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForImproper = (numImproper + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForDist12   = (numDist12 + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForDist13   = (numDist13 + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForAngle13  = (numAngle13 + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForDistLR   = (numDistLR + WARP_SIZE - 1) / WARP_SIZE;
  const int totalWarpsNeeded =
    warpsForTorsion + warpsForImproper + warpsForDist12 + warpsForDist13 + warpsForAngle13 + warpsForDistLR;

  // Each warp processes chunks in round-robin fashion
  for (int chunkIdx = warpId; chunkIdx < totalWarpsNeeded; chunkIdx += numWarps) {
    // Determine which term type this chunk belongs to
    if (chunkIdx < warpsForTorsion) {
      // Torsion terms
      const int baseIdx = chunkIdx * WARP_SIZE;
      const int termIdx = torsionStart + baseIdx + laneId;
      if (baseIdx + laneId < numTorsion) {
        const int localIdx1 = t_idx1s[termIdx] - atomStart;
        const int localIdx2 = t_idx2s[termIdx] - atomStart;
        const int localIdx3 = t_idx3s[termIdx] - atomStart;
        const int localIdx4 = t_idx4s[termIdx] - atomStart;
        energy += torsionAngleEnergy(molCoords,
                                     localIdx1,
                                     localIdx2,
                                     localIdx3,
                                     localIdx4,
                                     &t_forceConstants[termIdx * 6],
                                     &t_signs[termIdx * 6]);
      }
    } else if (chunkIdx < warpsForTorsion + warpsForImproper) {
      // Improper torsion terms
      const int warpOffset = chunkIdx - warpsForTorsion;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = improperStart + baseIdx + laneId;
      if (baseIdx + laneId < numImproper) {
        const int localIdx1 = i_idx1s[termIdx] - atomStart;
        const int localIdx2 = i_idx2s[termIdx] - atomStart;
        const int localIdx3 = i_idx3s[termIdx] - atomStart;
        const int localIdx4 = i_idx4s[termIdx] - atomStart;
        energy += inversionEnergy(molCoords,
                                  localIdx1,
                                  localIdx2,
                                  localIdx3,
                                  localIdx4,
                                  i_C0[termIdx],
                                  i_C1[termIdx],
                                  i_C2[termIdx],
                                  i_forceConstant[termIdx]);
      }
    } else if (chunkIdx < warpsForTorsion + warpsForImproper + warpsForDist12) {
      // 1-2 distance terms
      const int warpOffset = chunkIdx - warpsForTorsion - warpsForImproper;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = dist12Start + baseIdx + laneId;
      if (baseIdx + laneId < numDist12) {
        const int localIdx1 = d12_idx1s[termIdx] - atomStart;
        const int localIdx2 = d12_idx2s[termIdx] - atomStart;
        energy += distanceConstraintEnergy(molCoords,
                                           localIdx1,
                                           localIdx2,
                                           d12_minLen[termIdx],
                                           d12_maxLen[termIdx],
                                           d12_forceConstant[termIdx]);
      }
    } else if (chunkIdx < warpsForTorsion + warpsForImproper + warpsForDist12 + warpsForDist13) {
      // 1-3 distance terms
      const int warpOffset = chunkIdx - warpsForTorsion - warpsForImproper - warpsForDist12;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = dist13Start + baseIdx + laneId;
      if (baseIdx + laneId < numDist13) {
        const int localIdx1 = d13_idx1s[termIdx] - atomStart;
        const int localIdx2 = d13_idx2s[termIdx] - atomStart;
        energy += distanceConstraintEnergy(molCoords,
                                           localIdx1,
                                           localIdx2,
                                           d13_minLen[termIdx],
                                           d13_maxLen[termIdx],
                                           d13_forceConstant[termIdx]);
      }
    } else if (chunkIdx < warpsForTorsion + warpsForImproper + warpsForDist12 + warpsForDist13 + warpsForAngle13) {
      // 1-3 angle terms
      const int warpOffset = chunkIdx - warpsForTorsion - warpsForImproper - warpsForDist12 - warpsForDist13;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = angle13Start + baseIdx + laneId;
      if (baseIdx + laneId < numAngle13) {
        const int localIdx1 = a13_idx1s[termIdx] - atomStart;
        const int localIdx2 = a13_idx2s[termIdx] - atomStart;
        const int localIdx3 = a13_idx3s[termIdx] - atomStart;
        energy += angleConstraintEnergy(molCoords,
                                        localIdx1,
                                        localIdx2,
                                        localIdx3,
                                        a13_minAngle[termIdx],
                                        a13_maxAngle[termIdx],
                                        defaultAngleForceConstant);
      }
    } else {
      // Long-range distance terms
      const int warpOffset =
        chunkIdx - warpsForTorsion - warpsForImproper - warpsForDist12 - warpsForDist13 - warpsForAngle13;
      const int baseIdx = warpOffset * WARP_SIZE;
      const int termIdx = distLRStart + baseIdx + laneId;
      if (baseIdx + laneId < numDistLR) {
        const int localIdx1 = dlr_idx1s[termIdx] - atomStart;
        const int localIdx2 = dlr_idx2s[termIdx] - atomStart;
        energy += distanceConstraintEnergy(molCoords,
                                           localIdx1,
                                           localIdx2,
                                           dlr_minLen[termIdx],
                                           dlr_maxLen[termIdx],
                                           dlr_forceConstant[termIdx]);
      }
    }
  }

  return energy;
}

static __device__ __inline__ void molGradETK(const Energy3DForceContribsDevicePtr& terms,
                                             const BatchedIndices3DDevicePtr&      systemIndices,
                                             const DG_REAL*                        molCoords,
                                             DG_REAL*                              grad,
                                             const int                             molIdx,
                                             const int                             tid) {
  const int atomStart = systemIndices.atomStarts[molIdx];

  namespace cg            = cooperative_groups;
  constexpr int WARP_SIZE = 32;
  auto          tile32    = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
  const int     laneId    = tile32.thread_rank();

  const int warpId   = mark_warp_uniform(tile32.meta_group_rank());
  const int numWarps = mark_warp_uniform(tile32.meta_group_size());

  // Get term ranges
  const int torsionStart  = systemIndices.experimentalTorsionTermStarts[molIdx];
  const int torsionEnd    = systemIndices.experimentalTorsionTermStarts[molIdx + 1];
  const int improperStart = systemIndices.improperTorsionTermStarts[molIdx];
  const int improperEnd   = systemIndices.improperTorsionTermStarts[molIdx + 1];
  const int dist12Start   = systemIndices.dist12TermStarts[molIdx];
  const int dist12End     = systemIndices.dist12TermStarts[molIdx + 1];
  const int dist13Start   = systemIndices.dist13TermStarts[molIdx];
  const int dist13End     = systemIndices.dist13TermStarts[molIdx + 1];
  const int angle13Start  = systemIndices.angle13TermStarts[molIdx];
  const int angle13End    = systemIndices.angle13TermStarts[molIdx + 1];
  const int distLRStart   = systemIndices.longRangeDistTermStarts[molIdx];
  const int distLREnd     = systemIndices.longRangeDistTermStarts[molIdx + 1];

  const int numTorsion  = torsionEnd - torsionStart;
  const int numImproper = improperEnd - improperStart;
  const int numDist12   = dist12End - dist12Start;
  const int numDist13   = dist13End - dist13Start;
  const int numAngle13  = angle13End - angle13Start;
  const int numDistLR   = distLREnd - distLRStart;

  // Get term data
  const auto& [t_idx1s, t_idx2s, t_idx3s, t_idx4s, t_forceConstants, t_signs] = terms.experimentalTorsionTerms;
  const auto& [i_idx1s, i_idx2s, i_idx3s, i_idx4s, i_at2AtomicNum, i_isCBoundToO, i_C0, i_C1, i_C2, i_forceConstant] =
    terms.improperTorsionTerms;
  const auto& [d12_idx1s, d12_idx2s, d12_minLen, d12_maxLen, d12_forceConstant] = terms.dist12Terms;
  const auto& [d13_idx1s, d13_idx2s, d13_minLen, d13_maxLen, d13_forceConstant] = terms.dist13Terms;
  const auto& [a13_idx1s, a13_idx2s, a13_idx3s, a13_minAngle, a13_maxAngle]     = terms.angle13Terms;
  const auto& [dlr_idx1s, dlr_idx2s, dlr_minLen, dlr_maxLen, dlr_forceConstant] = terms.longRangeDistTerms;

  constexpr DG_REAL defaultAngleForceConstant = DG_REAL(1.0);

  // Calculate number of warps needed for each term type
  const int warpsForTorsion  = (numTorsion + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForImproper = (numImproper + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForDist12   = (numDist12 + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForDist13   = (numDist13 + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForAngle13  = (numAngle13 + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForDistLR   = (numDistLR + WARP_SIZE - 1) / WARP_SIZE;
  const int totalWarpsNeeded =
    warpsForTorsion + warpsForImproper + warpsForDist12 + warpsForDist13 + warpsForAngle13 + warpsForDistLR;

  // Each warp processes chunks in round-robin fashion
  for (int chunkIdx = warpId; chunkIdx < totalWarpsNeeded; chunkIdx += numWarps) {
    // Determine which term type this chunk belongs to
    if (chunkIdx < warpsForTorsion) {
      // Torsion terms
      const int baseIdx = chunkIdx * WARP_SIZE;
      const int termIdx = torsionStart + baseIdx + laneId;
      if (baseIdx + laneId < numTorsion) {
        const int localIdx1 = t_idx1s[termIdx] - atomStart;
        const int localIdx2 = t_idx2s[termIdx] - atomStart;
        const int localIdx3 = t_idx3s[termIdx] - atomStart;
        const int localIdx4 = t_idx4s[termIdx] - atomStart;
        torsionAngleGrad(molCoords,
                         localIdx1,
                         localIdx2,
                         localIdx3,
                         localIdx4,
                         &t_forceConstants[termIdx * 6],
                         &t_signs[termIdx * 6],
                         grad);
      }
    } else if (chunkIdx < warpsForTorsion + warpsForImproper) {
      // Improper torsion terms
      const int warpOffset = chunkIdx - warpsForTorsion;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = improperStart + baseIdx + laneId;
      if (baseIdx + laneId < numImproper) {
        const int localIdx1 = i_idx1s[termIdx] - atomStart;
        const int localIdx2 = i_idx2s[termIdx] - atomStart;
        const int localIdx3 = i_idx3s[termIdx] - atomStart;
        const int localIdx4 = i_idx4s[termIdx] - atomStart;
        inversionGrad(molCoords,
                      localIdx1,
                      localIdx2,
                      localIdx3,
                      localIdx4,
                      i_C0[termIdx],
                      i_C1[termIdx],
                      i_C2[termIdx],
                      i_forceConstant[termIdx],
                      grad);
      }
    } else if (chunkIdx < warpsForTorsion + warpsForImproper + warpsForDist12) {
      // 1-2 distance terms
      const int warpOffset = chunkIdx - warpsForTorsion - warpsForImproper;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = dist12Start + baseIdx + laneId;
      if (baseIdx + laneId < numDist12) {
        const int localIdx1 = d12_idx1s[termIdx] - atomStart;
        const int localIdx2 = d12_idx2s[termIdx] - atomStart;
        distanceConstraintGrad(molCoords,
                               localIdx1,
                               localIdx2,
                               d12_minLen[termIdx],
                               d12_maxLen[termIdx],
                               d12_forceConstant[termIdx],
                               grad);
      }
    } else if (chunkIdx < warpsForTorsion + warpsForImproper + warpsForDist12 + warpsForDist13) {
      // 1-3 distance terms
      const int warpOffset = chunkIdx - warpsForTorsion - warpsForImproper - warpsForDist12;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = dist13Start + baseIdx + laneId;
      if (baseIdx + laneId < numDist13) {
        const int localIdx1 = d13_idx1s[termIdx] - atomStart;
        const int localIdx2 = d13_idx2s[termIdx] - atomStart;
        distanceConstraintGrad(molCoords,
                               localIdx1,
                               localIdx2,
                               d13_minLen[termIdx],
                               d13_maxLen[termIdx],
                               d13_forceConstant[termIdx],
                               grad);
      }
    } else if (chunkIdx < warpsForTorsion + warpsForImproper + warpsForDist12 + warpsForDist13 + warpsForAngle13) {
      // 1-3 angle terms
      const int warpOffset = chunkIdx - warpsForTorsion - warpsForImproper - warpsForDist12 - warpsForDist13;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = angle13Start + baseIdx + laneId;
      if (baseIdx + laneId < numAngle13) {
        const int localIdx1 = a13_idx1s[termIdx] - atomStart;
        const int localIdx2 = a13_idx2s[termIdx] - atomStart;
        const int localIdx3 = a13_idx3s[termIdx] - atomStart;
        angleConstraintGrad(molCoords,
                            localIdx1,
                            localIdx2,
                            localIdx3,
                            a13_minAngle[termIdx],
                            a13_maxAngle[termIdx],
                            defaultAngleForceConstant,
                            grad);
      }
    } else {
      // Long-range distance terms
      const int warpOffset =
        chunkIdx - warpsForTorsion - warpsForImproper - warpsForDist12 - warpsForDist13 - warpsForAngle13;
      const int baseIdx = warpOffset * WARP_SIZE;
      const int termIdx = distLRStart + baseIdx + laneId;
      if (baseIdx + laneId < numDistLR) {
        const int localIdx1 = dlr_idx1s[termIdx] - atomStart;
        const int localIdx2 = dlr_idx2s[termIdx] - atomStart;
        distanceConstraintGrad(molCoords,
                               localIdx1,
                               localIdx2,
                               dlr_minLen[termIdx],
                               dlr_maxLen[termIdx],
                               dlr_forceConstant[termIdx],
                               grad);
      }
    }
  }
}
// ----------------------
// Fused block-per-molecule evaluation
// ----------------------
// One thread block owns one molecule. Atom indices are made molecule-local so pos/grad can be a shared-memory tile
// or the molecule's slice of global memory. Per-term arithmetic is the shared term functions above, so every
// precision keeps its native arithmetic (including the single-precision islands of FULL).
//
// Work is split into a bulk pairwise family (DG distance violations, ETK long-range distance restraints), streamed
// as chunks of 4 consecutive terms per thread with 16-byte loads, and a merged "small" range covering every other
// family (one term per thread per pass). The first chunk/item of each thread is loaded before the coordinate tile
// is ready, so a molecule normally costs one dependent round trip for its term data.

static __device__ __forceinline__ void loadChunk4(const int* __restrict__ p,
                                                  const int  c,
                                                  const int  lo,
                                                  const int  hi,
                                                  const bool vec,
                                                  int (&v)[4]) {
  if (vec) {
    const int4 t = __ldg(reinterpret_cast<const int4*>(p + c));
    v[0]         = t.x;
    v[1]         = t.y;
    v[2]         = t.z;
    v[3]         = t.w;
  } else {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int i = c + k;
      v[k]        = (i >= lo && i < hi) ? __ldg(p + i) : 0;
    }
  }
}

static __device__ __forceinline__ void loadChunk4(const float* __restrict__ p,
                                                  const int  c,
                                                  const int  lo,
                                                  const int  hi,
                                                  const bool vec,
                                                  float (&v)[4]) {
  if (vec) {
    const float4 t = __ldg(reinterpret_cast<const float4*>(p + c));
    v[0]           = t.x;
    v[1]           = t.y;
    v[2]           = t.z;
    v[3]           = t.w;
  } else {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int i = c + k;
      v[k]        = (i >= lo && i < hi) ? __ldg(p + i) : 0.0f;
    }
  }
}

static __device__ __forceinline__ void loadChunk4(const double* __restrict__ p,
                                                  const int  c,
                                                  const int  lo,
                                                  const int  hi,
                                                  const bool vec,
                                                  double (&v)[4]) {
  if (vec) {
    const double2 t0 = __ldg(reinterpret_cast<const double2*>(p + c));
    const double2 t1 = __ldg(reinterpret_cast<const double2*>(p + c) + 1);
    v[0]             = t0.x;
    v[1]             = t0.y;
    v[2]             = t1.x;
    v[3]             = t1.y;
  } else {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int i = c + k;
      v[k]        = (i >= lo && i < hi) ? __ldg(p + i) : 0.0;
    }
  }
}

//! Bulk DG family: distance violations. Params are (lb2, ub2, weight); gradient factors are single precision.
//! kTileStride is the per-atom stride of the bulk coordinate tile. Double precision uses an odd number of values, so
//! the random atom gathers of a warp spread over every shared-memory bank (a 4-double stride maps all atoms onto a
//! quarter of them); single precision keeps the natural layout, whose 4-float atoms load as one 16-byte vector.
static constexpr bool kOddBulkTile = sizeof(DG_REAL) == 8;
template <int dimension> struct DGBulkTerm {
  static constexpr int kStride     = dimension;
  static constexpr int kComps      = dimension;
  static constexpr int kTileStride = (kOddBulkTile && dimension == 4) ? 5 : dimension;
  using Contrib                    = float;
  template <int posStride>
  static __device__ __forceinline__ DG_REAL
  energy(const DG_REAL* pos, int a, int b, DG_REAL p0, DG_REAL p1, DG_REAL p2) {
    return distViolationEnergy<dimension, posStride>(pos, a, b, p0, p1, p2);
  }
  template <int posStride>
  static __device__ __forceinline__ bool
  grad(const DG_REAL* pos, int a, int b, DG_REAL p0, DG_REAL p1, DG_REAL p2, Contrib* d) {
    return distViolationGradTerm<dimension, posStride>(pos, a, b, p0, p1, p2, d);
  }
};

//! Coordinates of atom idx in 4-value slots with 16-byte loads (pos must be 16-byte aligned).
static __device__ __forceinline__ void loadAtom4(const DG_REAL* pos, const int idx, DG_REAL (&v)[4]) {
  const DG_REAL* p = pos + idx * 4;
  if constexpr (sizeof(DG_REAL) == 4) {
    const float4 t = *reinterpret_cast<const float4*>(p);
    v[0]           = t.x;
    v[1]           = t.y;
    v[2]           = t.z;
    v[3]           = t.w;
  } else {
    const double2 t = *reinterpret_cast<const double2*>(p);
    const double2 u = *reinterpret_cast<const double2*>(p + 2);
    v[0]            = t.x;
    v[1]            = t.y;
    v[2]            = u.x;
    v[3]            = u.y;
  }
}

//! DGBulkTerm whose energy gathers both atoms from the natural 4-value layout with 16-byte loads (16-byte aligned
//! coordinates): fewer L1 gathers for DG4 energies, which read coordinates straight from global memory. Other
//! strides and the gradient are unchanged.
template <int dimension> struct DGBulkTermVec : DGBulkTerm<dimension> {
  static_assert(dimension == 4, "vector coordinate loads need 4-value slots");
  template <int posStride>
  static __device__ __forceinline__ DG_REAL
  energy(const DG_REAL* pos, int a, int b, DG_REAL p0, DG_REAL p1, DG_REAL p2) {
    if constexpr (posStride == 4) {
      DG_REAL u[4];
      DG_REAL w[4];
      loadAtom4(pos, a, u);
      loadAtom4(pos, b, w);
      const DG_REAL dx = u[0] - w[0];
      const DG_REAL dy = u[1] - w[1];
      const DG_REAL dz = u[2] - w[2];
      const DG_REAL dw = u[3] - w[3];
      DG_REAL       d2 = dx * dx + dy * dy + dz * dz;  // summation order of distanceSquaredPosIdx
      d2 += dw * dw;
      return distViolationEnergyD2(d2, p0, p1, p2);
    } else {
      return DGBulkTerm<dimension>::template energy<posStride>(pos, a, b, p0, p1, p2);
    }
  }
};

//! Bulk ETK family: flat-bottom distance restraints on xyz. Params are (minLen, maxLen, forceConstant).
struct ETKBulkTerm {
  static constexpr int kStride     = 4;
  static constexpr int kComps      = 3;
  static constexpr int kTileStride = kOddBulkTile ? 3 : 4;  //!< double: xyz only
  using Contrib                    = DG_REAL;
  template <int posStride>
  static __device__ __forceinline__ DG_REAL
  energy(const DG_REAL* pos, int a, int b, DG_REAL p0, DG_REAL p1, DG_REAL p2) {
    return distanceConstraintEnergy<posStride>(pos, a, b, p0, p1, p2);
  }
  template <int posStride>
  static __device__ __forceinline__ bool
  grad(const DG_REAL* pos, int a, int b, DG_REAL p0, DG_REAL p1, DG_REAL p2, Contrib* d) {
    return distanceConstraintGradTerm<posStride>(pos, a, b, p0, p1, p2, d);
  }
};

//! Adds one atom's bulk-term contribution.
template <class Term> static __device__ __forceinline__ void bulkGradAdd(DG_REAL* g, const DG_REAL (&v)[Term::kComps]) {
#pragma unroll
  for (int k = 0; k < Term::kComps; ++k) {
    atomicAdd(g + k, v[k]);
  }
}

struct BulkPairArrays {
  const int*     idx1;
  const int*     idx2;
  const DG_REAL* p0;
  const DG_REAL* p1;
  const DG_REAL* p2;
};

//! Four consecutive bulk terms owned by one thread. Chunks start at multiples of 4 so full chunks are 16B aligned.
struct BulkChunk {
  int     i1[4];
  int     i2[4];
  DG_REAL p0[4];
  DG_REAL p1[4];
  DG_REAL p2[4];

  __device__ __forceinline__ void load(const BulkPairArrays& a,
                                       const int             c,
                                       const int             lo,
                                       const int             hi,
                                       const bool            aligned) {
    const bool vec = aligned && c >= lo && c + 4 <= hi;
    loadChunk4(a.idx1, c, lo, hi, vec, i1);
    loadChunk4(a.idx2, c, lo, hi, vec, i2);
    loadChunk4(a.p0, c, lo, hi, vec, p0);
    loadChunk4(a.p1, c, lo, hi, vec, p1);
    loadChunk4(a.p2, c, lo, hi, vec, p2);
  }

  template <class Term, int posStride>
  __device__ __forceinline__ DG_REAL
  energy(const DG_REAL* pos, const int c, const int lo, const int hi, const int atomStart) const {
    DG_REAL e = DG_REAL(0.0);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      // Out-of-range slots evaluate atom 0 against itself (zero parameters) and are discarded, so the four terms
      // stay branch-free and independent.
      const int     t       = c + k;
      const bool    inRange = t >= lo && t < hi;
      const DG_REAL et      = Term::template energy<posStride>(pos,
                                                          inRange ? i1[k] - atomStart : 0,
                                                          inRange ? i2[k] - atomStart : 0,
                                                          p0[k],
                                                          p1[k],
                                                          p2[k]);
      e += inRange ? et : DG_REAL(0.0);
    }
    return e;
  }

  //! Atom-2 contributions are added per term; atom-1 contributions are summed while consecutive terms share atom 1
  //! (distance lists are grouped by their first atom) and flushed once per run.
  template <class Term, int posStride>
  __device__ __forceinline__ void grad(const DG_REAL* pos,
                                       DG_REAL*       g,
                                       const int      c,
                                       const int      lo,
                                       const int      hi,
                                       const int      atomStart) const {
    int     runAtom = -1;
    DG_REAL run[Term::kComps];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int t = c + k;
      if (t < lo || t >= hi) {
        continue;
      }
      typename Term::Contrib d[Term::kComps];
      const int              a = i1[k] - atomStart;
      const int              b = i2[k] - atomStart;
      if (!Term::template grad<posStride>(pos, a, b, p0[k], p1[k], p2[k], d)) {
        continue;
      }
      if (a != runAtom) {
        if (runAtom >= 0) {
          bulkGradAdd<Term>(&g[runAtom * Term::kStride], run);
        }
        runAtom = a;
#pragma unroll
        for (int q = 0; q < Term::kComps; ++q) {
          run[q] = DG_REAL(0.0);
        }
      }
      DG_REAL other[Term::kComps];
#pragma unroll
      for (int q = 0; q < Term::kComps; ++q) {
        run[q] += static_cast<DG_REAL>(d[q]);
        other[q] = -d[q];
      }
      bulkGradAdd<Term>(&g[b * Term::kStride], other);
    }
    if (runAtom >= 0) {
      bulkGradAdd<Term>(&g[runAtom * Term::kStride], run);
    }
  }
};

//! One non-bulk term: up to four atoms, six real parameters and six integer signs.
struct SmallItem {
  int     kind;
  int     a[4];
  DG_REAL p[6];
  int     s[6];
};

// DG small kinds
constexpr int kDGSmallChiral = 0;
constexpr int kDGSmallFourth = 1;

//! DG non-bulk range: chiral terms followed by fourth-dimension terms. r = [chiral0, chiral1, fourth0, fourth1].
static __device__ __forceinline__ int dgSmallCount(const int* r) {
  return (r[1] - r[0]) + (r[3] - r[2]);
}

static __device__ __forceinline__ void dgSmallLoad(const EnergyForceContribsDevicePtr& terms,
                                                   const int*                          r,
                                                   const int                           v,
                                                   const int                           atomStart,
                                                   SmallItem&                          it) {
  const int nChiral = r[1] - r[0];
  if (v < nChiral) {
    const int   i = r[0] + v;
    const auto& c = terms.chiralTerms;
    it.kind       = kDGSmallChiral;
    it.a[0]       = c.idx1[i] - atomStart;
    it.a[1]       = c.idx2[i] - atomStart;
    it.a[2]       = c.idx3[i] - atomStart;
    it.a[3]       = c.idx4[i] - atomStart;
    it.p[0]       = c.volLower[i];
    it.p[1]       = c.volUpper[i];
  } else {
    it.kind = kDGSmallFourth;
    it.a[0] = terms.fourthTerms.idx[r[2] + v - nChiral] - atomStart;
  }
}

template <int dimension>
static __device__ __forceinline__ DG_REAL
dgSmallEnergy(const SmallItem& it, const DG_REAL* pos, const DG_REAL chiralWeight, const DG_REAL fourthDimWeight) {
  if (it.kind == kDGSmallChiral) {
    return chiralViolationEnergy<dimension>(pos, it.a[0], it.a[1], it.a[2], it.a[3], it.p[0], it.p[1], chiralWeight);
  }
  return fourthDimEnergy<dimension>(pos, it.a[0], fourthDimWeight);
}

template <int dimension>
static __device__ __forceinline__ void dgSmallGrad(const SmallItem& it,
                                                   const DG_REAL*   pos,
                                                   DG_REAL*         grad,
                                                   const DG_REAL    chiralWeight,
                                                   const DG_REAL    fourthDimWeight) {
  if (it.kind == kDGSmallChiral) {
    chiralViolationGrad<dimension>(pos, it.a[0], it.a[1], it.a[2], it.a[3], it.p[0], it.p[1], chiralWeight, grad);
  } else {
    fourthDimGrad<dimension>(pos, it.a[0], fourthDimWeight, grad);
  }
}

// ETK small kinds, in range order
constexpr int kETKSmallTorsion  = 0;
constexpr int kETKSmallImproper = 1;
constexpr int kETKSmallDist12   = 2;
constexpr int kETKSmallDist13   = 3;
constexpr int kETKSmallAngle13  = 4;

//! ETK non-bulk range: torsion, improper, 1-2, 1-3 distance and 1-3 angle terms back to back.
//! r = [torsion0, torsion1, improper0, improper1, d12_0, d12_1, d13_0, d13_1, angle0, angle1] (empty when masked).
static __device__ __forceinline__ int etkSmallCount(const int* r) {
  return (r[1] - r[0]) + (r[3] - r[2]) + (r[5] - r[4]) + (r[7] - r[6]) + (r[9] - r[8]);
}

static __device__ __forceinline__ void etkSmallLoad(const Energy3DForceContribsDevicePtr& terms,
                                                    const int*                            r,
                                                    int                                   v,
                                                    const int                             atomStart,
                                                    SmallItem&                            it) {
  const int n0 = r[1] - r[0];
  const int n1 = r[3] - r[2];
  const int n2 = r[5] - r[4];
  const int n3 = r[7] - r[6];
  int       kind;
  int       i;
  if (v < n0) {
    kind = kETKSmallTorsion;
    i    = r[0] + v;
  } else if ((v -= n0) < n1) {
    kind = kETKSmallImproper;
    i    = r[2] + v;
  } else if ((v -= n1) < n2) {
    kind = kETKSmallDist12;
    i    = r[4] + v;
  } else if ((v -= n2) < n3) {
    kind = kETKSmallDist13;
    i    = r[6] + v;
  } else {
    kind = kETKSmallAngle13;
    i    = r[8] + v - n3;
  }
  it.kind = kind;
  switch (kind) {
    case kETKSmallTorsion: {
      const auto& t = terms.experimentalTorsionTerms;
      it.a[0]       = t.idx1[i] - atomStart;
      it.a[1]       = t.idx2[i] - atomStart;
      it.a[2]       = t.idx3[i] - atomStart;
      it.a[3]       = t.idx4[i] - atomStart;
#pragma unroll
      for (int q = 0; q < 6; ++q) {
        it.p[q] = t.forceConstants[i * 6 + q];
        it.s[q] = t.signs[i * 6 + q];
      }
      break;
    }
    case kETKSmallImproper: {
      const auto& t = terms.improperTorsionTerms;
      it.a[0]       = t.idx1[i] - atomStart;
      it.a[1]       = t.idx2[i] - atomStart;
      it.a[2]       = t.idx3[i] - atomStart;
      it.a[3]       = t.idx4[i] - atomStart;
      it.p[0]       = t.C0[i];
      it.p[1]       = t.C1[i];
      it.p[2]       = t.C2[i];
      it.p[3]       = t.forceConstant[i];
      break;
    }
    case kETKSmallDist12:
    case kETKSmallDist13: {
      const auto& t = kind == kETKSmallDist12 ? terms.dist12Terms : terms.dist13Terms;
      it.a[0]       = t.idx1[i] - atomStart;
      it.a[1]       = t.idx2[i] - atomStart;
      it.p[0]       = t.minLen[i];
      it.p[1]       = t.maxLen[i];
      it.p[2]       = t.forceConstant[i];
      break;
    }
    default: {
      const auto& t = terms.angle13Terms;
      it.a[0]       = t.idx1[i] - atomStart;
      it.a[1]       = t.idx2[i] - atomStart;
      it.a[2]       = t.idx3[i] - atomStart;
      it.p[0]       = t.minAngle[i];
      it.p[1]       = t.maxAngle[i];
      break;
    }
  }
}

static __device__ __forceinline__ DG_REAL etkSmallEnergy(const SmallItem& it, const DG_REAL* pos) {
  switch (it.kind) {
    case kETKSmallTorsion:
      return torsionAngleEnergy(pos, it.a[0], it.a[1], it.a[2], it.a[3], it.p, it.s);
    case kETKSmallImproper:
      return inversionEnergy(pos, it.a[0], it.a[1], it.a[2], it.a[3], it.p[0], it.p[1], it.p[2], it.p[3]);
    case kETKSmallDist12:
    case kETKSmallDist13:
      return distanceConstraintEnergy(pos, it.a[0], it.a[1], it.p[0], it.p[1], it.p[2]);
    default:
      return angleConstraintEnergy(pos, it.a[0], it.a[1], it.a[2], it.p[0], it.p[1], DG_REAL(1.0));
  }
}

static __device__ __forceinline__ void etkSmallGrad(const SmallItem& it, const DG_REAL* pos, DG_REAL* grad) {
  switch (it.kind) {
    case kETKSmallTorsion:
      torsionAngleGrad(pos, it.a[0], it.a[1], it.a[2], it.a[3], it.p, it.s, grad);
      break;
    case kETKSmallImproper:
      inversionGrad(pos, it.a[0], it.a[1], it.a[2], it.a[3], it.p[0], it.p[1], it.p[2], it.p[3], grad);
      break;
    case kETKSmallDist12:
    case kETKSmallDist13:
      distanceConstraintGrad(pos, it.a[0], it.a[1], it.p[0], it.p[1], it.p[2], grad);
      break;
    default:
      angleConstraintGrad(pos, it.a[0], it.a[1], it.a[2], it.p[0], it.p[1], DG_REAL(1.0), grad);
      break;
  }
}

//! DG family for the fused driver. Ranges: [dist0, dist1, chiral0, chiral1, fourth0, fourth1].
//! vecPos: bulk energies gather natural-layout coordinates with 16-byte loads (dimension 4, aligned coordinates).
template <int dimension, bool vecPos = false> struct DGFamily {
  EnergyForceContribsDevicePtr terms;
  BatchedIndicesDevicePtr      indices;
  DG_REAL                      chiralWeight;
  DG_REAL                      fourthDimWeight;
  bool                         aligned;  //!< bulk arrays are 16-byte aligned

  using Bulk = typename std::conditional<vecPos, DGBulkTermVec<dimension>, DGBulkTerm<dimension>>::type;
  static constexpr int kNumRanges = 6;
  static constexpr int kDimension = dimension;

  __device__ __forceinline__ void loadRanges(const int m, int* r) const {
    r[0] = indices.distTermStarts[m];
    r[1] = indices.distTermStarts[m + 1];
    r[2] = indices.chiralTermStarts[m];
    r[3] = indices.chiralTermStarts[m + 1];
    r[4] = indices.fourthTermStarts[m];
    r[5] = indices.fourthTermStarts[m + 1];
  }
  __device__ __forceinline__ BulkPairArrays bulkArrays() const {
    const auto& d = terms.distTerms;
    return {d.idx1, d.idx2, d.lb2, d.ub2, d.weight};
  }
  __device__ __forceinline__ int  smallCount(const int* r) const { return dgSmallCount(r + 2); }
  __device__ __forceinline__ void smallLoad(const int* r, const int v, const int atomStart, SmallItem& it) const {
    dgSmallLoad(terms, r + 2, v, atomStart, it);
  }
  __device__ __forceinline__ DG_REAL smallEnergy(const SmallItem& it, const DG_REAL* pos) const {
    return dgSmallEnergy<dimension>(it, pos, chiralWeight, fourthDimWeight);
  }
  __device__ __forceinline__ void smallGrad(const SmallItem& it, const DG_REAL* pos, DG_REAL* grad) const {
    dgSmallGrad<dimension>(it, pos, grad, chiralWeight, fourthDimWeight);
  }
};

//! ETK family for the fused driver (4D storage). Ranges: [longRange0, longRange1, then the small range layout of
//! etkSmallLoad]. Families outside termMask get empty ranges.
struct ETKFamily {
  Energy3DForceContribsDevicePtr terms;
  BatchedIndices3DDevicePtr      indices;
  int                            termMask;
  bool                           aligned;  //!< bulk arrays are 16-byte aligned

  using Bulk                      = ETKBulkTerm;
  static constexpr int kNumRanges = 12;
  static constexpr int kDimension = 4;

  __device__ __forceinline__ void loadRanges(const int m, int* r) const {
    const int* starts[6] = {indices.longRangeDistTermStarts,
                            indices.experimentalTorsionTermStarts,
                            indices.improperTorsionTermStarts,
                            indices.dist12TermStarts,
                            indices.dist13TermStarts,
                            indices.angle13TermStarts};
    const int  bits[6] =
      {kETKLongDistBit, kETKTorsionBit, kETKImproperBit, kETKDist12Bit, kETKDist13Bit, kETKAngle13Bit};
#pragma unroll
    for (int f = 0; f < 6; ++f) {
      if (termMask & bits[f]) {
        r[2 * f]     = starts[f][m];
        r[2 * f + 1] = starts[f][m + 1];
      } else {
        r[2 * f]     = 0;
        r[2 * f + 1] = 0;
      }
    }
  }
  __device__ __forceinline__ BulkPairArrays bulkArrays() const {
    const auto& d = terms.longRangeDistTerms;
    return {d.idx1, d.idx2, d.minLen, d.maxLen, d.forceConstant};
  }
  __device__ __forceinline__ int  smallCount(const int* r) const { return etkSmallCount(r + 2); }
  __device__ __forceinline__ void smallLoad(const int* r, const int v, const int atomStart, SmallItem& it) const {
    etkSmallLoad(terms, r + 2, v, atomStart, it);
  }
  __device__ __forceinline__ DG_REAL smallEnergy(const SmallItem& it, const DG_REAL* pos) const {
    return etkSmallEnergy(it, pos);
  }
  __device__ __forceinline__ void smallGrad(const SmallItem& it, const DG_REAL* pos, DG_REAL* grad) const {
    etkSmallGrad(it, pos, grad);
  }
};

//! Per-thread work of one molecule: the prefetched first chunk/item, then any further passes.
//! pos (natural stride) feeds the small terms; bulkPos (stride bulkPosStride) feeds the bulk terms.
template <class Family, bool isGrad, int bulkPosStride>
static __device__ __forceinline__ DG_REAL fusedEvaluate(const Family&    fam,
                                                        const int*       r,
                                                        const BulkChunk& chunk,
                                                        const SmallItem& item,
                                                        const int        nSmall,
                                                        const int        atomStart,
                                                        const DG_REAL*   pos,
                                                        const DG_REAL*   bulkPos,
                                                        DG_REAL*         grad) {
  using Bulk         = typename Family::Bulk;
  const int tid      = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lo       = r[0];
  const int hi       = r[1];
  const int c0       = (lo & ~3) + 4 * tid;
  const int bulkStep = 4 * nthreads;
  DG_REAL   energy   = DG_REAL(0.0);
  // Bulk chunk first so its registers are dead before the (register-heavy) small-term math.
  if (c0 < hi) {
    if constexpr (isGrad) {
      chunk.template grad<Bulk, bulkPosStride>(bulkPos, grad, c0, lo, hi, atomStart);
    } else {
      energy += chunk.template energy<Bulk, bulkPosStride>(bulkPos, c0, lo, hi, atomStart);
    }
  }
  if (tid < nSmall) {
    if constexpr (isGrad) {
      fam.smallGrad(item, pos, grad);
    } else {
      energy += fam.smallEnergy(item, pos);
    }
  }
  // Further passes (large molecules): load the next chunk and item together so they share one round trip.
  const BulkPairArrays bulk = fam.bulkArrays();
  int                  c    = c0 + bulkStep;
  int                  v    = tid + nthreads;
  while (c < hi || v < nSmall) {
    BulkChunk nextChunk;
    SmallItem nextItem;
    if (c < hi) {
      nextChunk.load(bulk, c, lo, hi, fam.aligned);
    }
    if (v < nSmall) {
      fam.smallLoad(r, v, atomStart, nextItem);
    }
    if (c < hi) {
      if constexpr (isGrad) {
        nextChunk.template grad<Bulk, bulkPosStride>(bulkPos, grad, c, lo, hi, atomStart);
      } else {
        energy += nextChunk.template energy<Bulk, bulkPosStride>(bulkPos, c, lo, hi, atomStart);
      }
    }
    if (v < nSmall) {
      if constexpr (isGrad) {
        fam.smallGrad(nextItem, pos, grad);
      } else {
        energy += fam.smallEnergy(nextItem, pos);
      }
    }
    c += bulkStep;
    v += nthreads;
  }
  return energy;
}

template <typename T> static __device__ __forceinline__ T fusedWarpSum(T v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
  }
  return v;
}

//! Block-wide sum; the result is valid in thread 0.
static __device__ __forceinline__ DG_REAL fusedBlockSum(DG_REAL v, DG_REAL* scratch) {
  v              = fusedWarpSum(v);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    scratch[warp] = v;
  }
  __syncthreads();
  DG_REAL total = DG_REAL(0.0);
  if (warp == 0) {
    total = lane < static_cast<int>(blockDim.x >> 5) ? scratch[lane] : DG_REAL(0.0);
    total = fusedWarpSum(total);
  }
  return total;
}

//! Evaluates molecule blockIdx.x and updates its output in place (see launchFusedEnergyKernel for the contract).
//!   energy: active -> sum (isSingle) or out + sum; inactive -> 0 (isSingle) or unchanged.
//!   gradient: active -> sum (overwriteGrad) or out + sum; inactive -> unchanged.
//! sPos holds tileVals values; larger molecules read coordinates from global memory. Gradients always accumulate in
//! the molecule's slice of global memory.
template <class Family, bool isGrad, bool isSingle, bool overwriteGrad>
static __device__ __forceinline__ void fusedMolecule(const Family& fam,
                                                     const int* __restrict__ atomStarts,
                                                     const uint8_t* __restrict__ active,
                                                     const DG_REAL* __restrict__ coords,
                                                     DG_REAL* __restrict__ out,
                                                     DG_REAL*  sPos,
                                                     DG_REAL*  sBulk,
                                                     DG_REAL*  sRed,
                                                     const int tileVals) {
  constexpr int dimension  = Family::kDimension;
  using Bulk               = typename Family::Bulk;
  //! Bulk terms read a separate tile when their conflict-light stride differs from the natural layout.
  constexpr bool kBulkTile = Bulk::kTileStride != dimension;
  const int      molIdx    = blockIdx.x;
  const int      tid       = threadIdx.x;
  const int      nthreads  = blockDim.x;

  // Round trip 1: everything addressed by the molecule index alone.
  const int  atomStart = atomStarts[molIdx];
  const int  atomEnd   = atomStarts[molIdx + 1];
  const bool isActive  = active == nullptr || (isSingle ? active[molIdx] != 0 : active[molIdx] == 1);
  int        r[Family::kNumRanges];
  fam.loadRanges(molIdx, r);
  DG_REAL priorEnergy = DG_REAL(0.0);
  if constexpr (!isGrad && !isSingle) {
    priorEnergy = out[molIdx];
  }
  const int numVals = (atomEnd - atomStart) * dimension;
  const int offset  = atomStart * dimension;

  if (!isActive) {
    if constexpr (!isGrad && isSingle) {
      if (tid == 0) {
        out[molIdx] = DG_REAL(0.0);
      }
    }
    return;
  }

  // Round trip 2: coordinates plus the first bulk chunk and small item of every thread.
  const int lo = r[0];
  const int hi = r[1];
  const int c0 = (lo & ~3) + 4 * tid;
  BulkChunk chunk;
  if (c0 < hi) {
    chunk.load(fam.bulkArrays(), c0, lo, hi, fam.aligned);
  }
  const int nSmall = fam.smallCount(r);
  SmallItem item;
  if (tid < nSmall) {
    fam.smallLoad(r, tid, atomStart, item);
  }

  if (numVals <= tileVals) {
    // Coordinates come from the shared tile. Gradients accumulate directly in this molecule's output slice with
    // global atomics; when overwriting, the slice is zeroed first and the barrier orders those stores
    // before every reduction of the block.
    for (int i = tid; i < numVals; i += nthreads) {
      const DG_REAL v = coords[offset + i];
      sPos[i]         = v;
      if constexpr (kBulkTile) {
        const int atom = i / dimension;
        const int comp = i - atom * dimension;
        if (comp < Bulk::kComps) {
          sBulk[atom * Bulk::kTileStride + comp] = v;
        }
      }
      if constexpr (isGrad && overwriteGrad) {
        out[offset + i] = DG_REAL(0.0);
      }
    }
    __syncthreads();
    const DG_REAL energy = fusedEvaluate<Family, isGrad, Bulk::kTileStride>(fam,
                                                                            r,
                                                                            chunk,
                                                                            item,
                                                                            nSmall,
                                                                            atomStart,
                                                                            sPos,
                                                                            kBulkTile ? sBulk : sPos,
                                                                            isGrad ? out + offset : nullptr);
    if constexpr (!isGrad) {
      const DG_REAL total = fusedBlockSum(energy, sRed);
      if (tid == 0) {
        out[molIdx] = priorEnergy + total;
      }
    }
  } else {
    // Molecule larger than the shared tile: coordinates from global memory as well.
    if constexpr (isGrad) {
      if constexpr (overwriteGrad) {
        for (int i = tid; i < numVals; i += nthreads) {
          out[offset + i] = DG_REAL(0.0);
        }
        __syncthreads();
      }
      fusedEvaluate<Family, isGrad, Bulk::kStride>(fam,
                                                   r,
                                                   chunk,
                                                   item,
                                                   nSmall,
                                                   atomStart,
                                                   coords + offset,
                                                   coords + offset,
                                                   out + offset);
    } else {
      const DG_REAL energy = fusedEvaluate<Family, isGrad, Bulk::kStride>(fam,
                                                                          r,
                                                                          chunk,
                                                                          item,
                                                                          nSmall,
                                                                          atomStart,
                                                                          coords + offset,
                                                                          coords + offset,
                                                                          nullptr);
      const DG_REAL total  = fusedBlockSum(energy, sRed);
      if (tid == 0) {
        out[molIdx] = priorEnergy + total;
      }
    }
  }
}
