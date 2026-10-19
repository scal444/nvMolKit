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

// Instantiated by mmff_kernels_device_dispatch.cuh for each arithmetic type.

using namespace nvMolKit::FFKernelUtils;

//! A molecule's coordinates staged in shared memory as x, y and z arrays of Stride values each, so that a warp reading
//! the coordinates of consecutive atoms touches consecutive banks (interleaved xyz spreads 32 atoms over 96 words).
//! Device code reads coordinates through coordOf, which takes this or an interleaved xyz pointer.
template <typename T, int Stride> struct PlanarCoords {
  T* p;
};

template <typename T> static __device__ __forceinline__ T coordOf(const T* pos, const int idx, const int c) {
  return pos[3 * idx + c];
}

template <typename T, int Stride>
static __device__ __forceinline__ T coordOf(const PlanarCoords<T, Stride>& pos, const int idx, const int c) {
  return pos.p[c * Stride + idx];
}

//! distanceSquared and distanceSquaredWithComponents of kernel_utils.cuh (same arithmetic) through coordOf.
static __device__ __forceinline__ real coordDistanceSquared(const auto& pos, const int idx1, const int idx2) {
  const real dx = coordOf(pos, idx1, 0) - coordOf(pos, idx2, 0);
  const real dy = coordOf(pos, idx1, 1) - coordOf(pos, idx2, 1);
  const real dz = coordOf(pos, idx1, 2) - coordOf(pos, idx2, 2);
  return dx * dx + dy * dy + dz * dz;
}

static __device__ __forceinline__ real
coordDistanceSquaredWithComponents(const auto& pos, const int idx1, const int idx2, real& dx, real& dy, real& dz) {
  dx = coordOf(pos, idx1, 0) - coordOf(pos, idx2, 0);
  dy = coordOf(pos, idx1, 1) - coordOf(pos, idx2, 1);
  dz = coordOf(pos, idx1, 2) - coordOf(pos, idx2, 2);
  return dx * dx + dy * dy + dz * dz;
}

constexpr real degreeToRadian = static_cast<real>(M_PI) / real{180.0};
constexpr real radianToDegree = real{180.0} / static_cast<real>(M_PI);

//! Gradient sink that atomically adds one atom's (x, y, z) contribution to a flat xyz gradient buffer.
template <typename T> struct AtomicGradSink {
  T*                              grad;
  __device__ __forceinline__ void operator()(const int idx, const real x, const real y, const real z) const {
    atomicAdd(&grad[3 * idx + 0], x);
    atomicAdd(&grad[3 * idx + 1], y);
    atomicAdd(&grad[3 * idx + 2], z);
  }
  //! Adds one component (0, 1 or 2 for x, y or z) of an atom's contribution.
  __device__ __forceinline__ void component(const int idx, const int c, const real v) const {
    atomicAdd(&grad[3 * idx + c], v);
  }
};

template <typename T> static __device__ __forceinline__ AtomicGradSink<T> atomicGradSink(T* grad) {
  return AtomicGradSink<T>{grad};
}

//! Gradient sink that atomically adds into a planar gradient buffer holding the x, y and z components of the
//! molecule's atoms in three consecutive arrays of stride atoms. In global memory, a warp's scattered updates of one
//! component then fall into a third of the cache lines that the interleaved layout spreads them over; in shared memory,
//! consecutive atoms fall into consecutive banks.
template <typename T> struct PlanarAtomicGradSink {
  T*                              grad;
  int                             stride;
  __device__ __forceinline__ void operator()(const int idx, const real x, const real y, const real z) const {
    atomicAdd(&grad[idx], x);
    atomicAdd(&grad[stride + idx], y);
    atomicAdd(&grad[2 * stride + idx], z);
  }
  __device__ __forceinline__ void component(const int idx, const int c, const real v) const {
    atomicAdd(&grad[c * stride + idx], v);
  }
};

namespace rdkit_ports {

static __device__ __forceinline__ void oopGradTo(const auto& pos,
                                                 const int   idx1,
                                                 const int   idx2,
                                                 const int   idx3,
                                                 const int   idx4,
                                                 const real  koop,
                                                 const auto& sink) {
  constexpr real prefactor = real{143.9325} * degreeToRadian;

  real dJIx = coordOf(pos, idx1, 0) - coordOf(pos, idx2, 0);
  real dJIy = coordOf(pos, idx1, 1) - coordOf(pos, idx2, 1);
  real dJIz = coordOf(pos, idx1, 2) - coordOf(pos, idx2, 2);

  real dJKx = coordOf(pos, idx3, 0) - coordOf(pos, idx2, 0);
  real dJKy = coordOf(pos, idx3, 1) - coordOf(pos, idx2, 1);
  real dJKz = coordOf(pos, idx3, 2) - coordOf(pos, idx2, 2);

  real dJLx = coordOf(pos, idx4, 0) - coordOf(pos, idx2, 0);
  real dJLy = coordOf(pos, idx4, 1) - coordOf(pos, idx2, 1);
  real dJLz = coordOf(pos, idx4, 2) - coordOf(pos, idx2, 2);

  const real invdJI = NVMOLKIT_RSQRT(dJIx * dJIx + dJIy * dJIy + dJIz * dJIz);
  const real invdJK = NVMOLKIT_RSQRT(dJKx * dJKx + dJKy * dJKy + dJKz * dJKz);
  const real invdJL = NVMOLKIT_RSQRT(dJLx * dJLx + dJLy * dJLy + dJLz * dJLz);

  dJIx *= invdJI;
  dJIy *= invdJI;
  dJIz *= invdJI;
  dJKx *= invdJK;
  dJKy *= invdJK;
  dJKz *= invdJK;
  dJLx *= invdJL;
  dJLy *= invdJL;
  dJLz *= invdJL;

  real normalJIKx, normalJIKy, normalJIKz;
  crossProduct(-dJIx, -dJIy, -dJIz, dJKx, dJKy, dJKz, normalJIKx, normalJIKy, normalJIKz);
  const real invNormLength =
    NVMOLKIT_RSQRT(normalJIKx * normalJIKx + normalJIKy * normalJIKy + normalJIKz * normalJIKz);
  normalJIKx *= invNormLength;
  normalJIKy *= invNormLength;
  normalJIKz *= invNormLength;

  const real sinChi    = clamp(dotProduct(dJLx, dJLy, dJLz, normalJIKx, normalJIKy, normalJIKz), real{-1.0}, real{1.0});
  const real cosChiSq  = real{1.0} - sinChi * sinChi;
  const real invCosChi = cosChiSq > 0 ? NVMOLKIT_RSQRT(cosChiSq) : real{1.0e8};
  const real chi       = radianToDegree * NVMOLKIT_ASIN(sinChi);
  const real cosTheta  = clamp(dotProduct(dJIx, dJIy, dJIz, dJKx, dJKy, dJKz), real{-1.0}, real{1.0});

  real invSinTheta = NVMOLKIT_RSQRT(NVMOLKIT_FMAX(real{1.0} - cosTheta * cosTheta, real{1.0e-8}));

  real dE_dChi = prefactor * koop * chi;
  real t1x, t1y, t1z, t2x, t2y, t2z, t3x, t3y, t3z;
  crossProduct(dJLx, dJLy, dJLz, dJKx, dJKy, dJKz, t1x, t1y, t1z);
  crossProduct(dJIx, dJIy, dJIz, dJLx, dJLy, dJLz, t2x, t2y, t2z);
  crossProduct(dJKx, dJKy, dJKz, dJIx, dJIy, dJIz, t3x, t3y, t3z);

  real term1  = invCosChi * invSinTheta;
  real term2  = sinChi * invCosChi * (invSinTheta * invSinTheta);
  real tg1[3] = {(t1x * term1 - (dJIx - dJKx * cosTheta) * term2) * invdJI,
                 (t1y * term1 - (dJIy - dJKy * cosTheta) * term2) * invdJI,
                 (t1z * term1 - (dJIz - dJKz * cosTheta) * term2) * invdJI};
  real tg3[3] = {(t2x * term1 - (dJKx - dJIx * cosTheta) * term2) * invdJK,
                 (t2y * term1 - (dJKy - dJIy * cosTheta) * term2) * invdJK,
                 (t2z * term1 - (dJKz - dJIz * cosTheta) * term2) * invdJK};
  real tg4[3] = {(t3x * term1 - dJLx * sinChi * invCosChi) * invdJL,
                 (t3y * term1 - dJLy * sinChi * invCosChi) * invdJL,
                 (t3z * term1 - dJLz * sinChi * invCosChi) * invdJL};

  sink(idx1, dE_dChi * tg1[0], dE_dChi * tg1[1], dE_dChi * tg1[2]);
  sink(idx2,
       -dE_dChi * (tg1[0] + tg3[0] + tg4[0]),
       -dE_dChi * (tg1[1] + tg3[1] + tg4[1]),
       -dE_dChi * (tg1[2] + tg3[2] + tg4[2]));
  sink(idx3, dE_dChi * tg3[0], dE_dChi * tg3[1], dE_dChi * tg3[2]);
  sink(idx4, dE_dChi * tg4[0], dE_dChi * tg4[1], dE_dChi * tg4[2]);
}

static __device__ __forceinline__ void
oopGrad(const auto& pos, const int idx1, const int idx2, const int idx3, const int idx4, const real koop, auto* grad) {
  oopGradTo(pos, idx1, idx2, idx3, idx4, koop, atomicGradSink(grad));
}

static __device__ __forceinline__ void torsionGradTo(const auto& pos,
                                                     const int   idx1,
                                                     const int   idx2,
                                                     const int   idx3,
                                                     const int   idx4,
                                                     const real  V1,
                                                     const real  V2,
                                                     const real  V3,
                                                     const auto& sink) {
  // P1 - P2
  const real dx1 = coordOf(pos, idx1, 0) - coordOf(pos, idx2, 0);
  const real dy1 = coordOf(pos, idx1, 1) - coordOf(pos, idx2, 1);
  const real dz1 = coordOf(pos, idx1, 2) - coordOf(pos, idx2, 2);

  // P3 - P2
  const real dx2 = coordOf(pos, idx3, 0) - coordOf(pos, idx2, 0);
  const real dy2 = coordOf(pos, idx3, 1) - coordOf(pos, idx2, 1);
  const real dz2 = coordOf(pos, idx3, 2) - coordOf(pos, idx2, 2);

  // P4 - P3
  const real dx4 = coordOf(pos, idx4, 0) - coordOf(pos, idx3, 0);
  const real dy4 = coordOf(pos, idx4, 1) - coordOf(pos, idx3, 1);
  const real dz4 = coordOf(pos, idx4, 2) - coordOf(pos, idx3, 2);

  real cross1x, cross1y, cross1z;
  crossProduct(dx1, dy1, dz1, dx2, dy2, dz2, cross1x, cross1y, cross1z);
  const real invNorm1 =
    NVMOLKIT_FMIN(NVMOLKIT_RSQRT(cross1x * cross1x + cross1y * cross1y + cross1z * cross1z), real{1.0e5});
  cross1x *= invNorm1;
  cross1y *= invNorm1;
  cross1z *= invNorm1;

  real cross2x, cross2y, cross2z;
  // Use -dx2, -dy2, -dz2 directly instead of storing dx3, dy3, dz3
  crossProduct(-dx2, -dy2, -dz2, dx4, dy4, dz4, cross2x, cross2y, cross2z);
  const real invNorm2 =
    NVMOLKIT_FMIN(NVMOLKIT_RSQRT(cross2x * cross2x + cross2y * cross2y + cross2z * cross2z), real{1.0e5});
  cross2x *= invNorm2;
  cross2y *= invNorm2;
  cross2z *= invNorm2;

  const real cosPhi = clamp(dotProduct(cross1x, cross1y, cross1z, cross2x, cross2y, cross2z), -real{1.0}, real{1.0});

  const real sinPhiSq = 1.0f - cosPhi * cosPhi;
  real       sinTerm  = real{0.0};
  if (sinPhiSq > real{0.0}) {
    const real sin2Phi = 2.0f * cosPhi;
    const real sin3Phi = 3.0f - 4.0f * sinPhiSq;
    sinTerm            = 0.5f * (V1 - 2.0f * V2 * sin2Phi + 3.0f * V3 * sin3Phi);
  }

  real dCos_dT0 = invNorm1 * (cross2x - cosPhi * cross1x);
  real dCos_dT1 = invNorm1 * (cross2y - cosPhi * cross1y);
  real dCos_dT2 = invNorm1 * (cross2z - cosPhi * cross1z);

  sink(idx1,
       sinTerm * (dCos_dT2 * dy2 - dCos_dT1 * dz2),
       sinTerm * (dCos_dT0 * dz2 - dCos_dT2 * dx2),
       sinTerm * (dCos_dT1 * dx2 - dCos_dT0 * dy2));

  // idx3 and idx4 gradients - reuse variables dCos_dT0-2 for dCos_dT3-5
  const real dCos_dT3 = invNorm2 * (cross1x - cosPhi * cross2x);
  const real dCos_dT4 = invNorm2 * (cross1y - cosPhi * cross2y);
  const real dCos_dT5 = invNorm2 * (cross1z - cosPhi * cross2z);

  sink(idx2,
       sinTerm * (dCos_dT1 * (dz2 - dz1) + dCos_dT2 * (dy1 - dy2) + dCos_dT4 * (-dz4) + dCos_dT5 * (dy4)),
       sinTerm * (dCos_dT0 * (dz1 - dz2) + dCos_dT2 * (dx2 - dx1) + dCos_dT3 * (dz4) + dCos_dT5 * (-dx4)),
       sinTerm * (dCos_dT0 * (dy2 - dy1) + dCos_dT1 * (dx1 - dx2) + dCos_dT3 * (-dy4) + dCos_dT4 * (dx4)));

  sink(idx3,
       sinTerm * (dCos_dT1 * (dz1) + dCos_dT2 * (-dy1) + dCos_dT4 * (dz4 + dz2) + dCos_dT5 * (-dy4 - dy2)),
       sinTerm * (dCos_dT0 * (-dz1) + dCos_dT2 * (dx1) + dCos_dT3 * (-dz4 - dz2) + dCos_dT5 * (dx4 + dx2)),
       sinTerm * (dCos_dT0 * (dy1) + dCos_dT1 * (-dx1) + dCos_dT3 * (dy4 + dy2) + dCos_dT4 * (-dx4 - dx2)));

  sink(idx4,
       sinTerm * (dCos_dT4 * (-dz2) - dCos_dT5 * (-dy2)),
       sinTerm * (dCos_dT5 * (-dx2) - dCos_dT3 * (-dz2)),
       sinTerm * (dCos_dT3 * (-dy2) - dCos_dT4 * (-dx2)));
}

static __device__ __forceinline__ void torsionGrad(const auto& pos,
                                                   const int   idx1,
                                                   const int   idx2,
                                                   const int   idx3,
                                                   const int   idx4,
                                                   const real  V1,
                                                   const real  V2,
                                                   const real  V3,
                                                   auto*       grad) {
  torsionGradTo(pos, idx1, idx2, idx3, idx4, V1, V2, V3, atomicGradSink(grad));
}
//! Gradient of one vdW pair with respect to idx1; the idx2 gradient is its negation.
static __device__ __forceinline__ void vdwGradVec(const auto& pos,
                                                  const int   idx1,
                                                  const int   idx2,
                                                  const real  R_ij_star,
                                                  const real  wellDepth,
                                                  real&       term1x,
                                                  real&       term1y,
                                                  real&       term1z) {
  constexpr real vdw1   = real{1.07};
  constexpr real vdw1m1 = vdw1 - real{1.0};
  constexpr real vdw2   = real{1.12};
  constexpr real vdw2m1 = vdw2 - real{1.0};
  constexpr real vdw2t7 = vdw2 * real{7.0};

  const real invDistance = NVMOLKIT_RSQRT(coordDistanceSquared(pos, idx1, idx2));
  const real distance    = 1.0f / invDistance;

  const real invRIJStar = 1.0f / R_ij_star;

  const real q         = distance * invRIJStar;
  const real q2        = q * q;
  const real q6        = q2 * q2 * q2;
  const real q7        = q6 * q;
  const real q7pvdw2m1 = q7 + vdw2m1;
  const real invQ7Term = 1.0f / q7pvdw2m1;
  const real t         = vdw1 / (q + vdw1 - real{1.0});
  const real t2        = t * t;
  const real t7        = t2 * t2 * t2 * t;
  const real dE_dr     = wellDepth * invRIJStar * t7 *
                     (-vdw2t7 * q6 * invQ7Term * invQ7Term + ((-vdw2t7 * invQ7Term + real{14.0}) / (q + vdw1m1)));

  if (distance <= real{0.0}) {
    term1x = R_ij_star * 0.01f;
    term1y = R_ij_star * 0.01f;
    term1z = R_ij_star * 0.01f;
  } else {
    term1x = dE_dr * (coordOf(pos, idx1, 0) - coordOf(pos, idx2, 0)) * invDistance;
    term1y = dE_dr * (coordOf(pos, idx1, 1) - coordOf(pos, idx2, 1)) * invDistance;
    term1z = dE_dr * (coordOf(pos, idx1, 2) - coordOf(pos, idx2, 2)) * invDistance;
  }
}

static __device__ __forceinline__ void vDWGrad(const auto& pos,
                                               const int   idx1,
                                               const int   idx2,
                                               const real  R_ij_star,
                                               const real  wellDepth,
                                               auto*       grad) {
  real term1x, term1y, term1z;
  vdwGradVec(pos, idx1, idx2, R_ij_star, wellDepth, term1x, term1y, term1z);

  atomicAdd(&grad[3 * idx1 + 0], term1x);
  atomicAdd(&grad[3 * idx1 + 1], term1y);
  atomicAdd(&grad[3 * idx1 + 2], term1z);

  atomicAdd(&grad[3 * idx2 + 0], -term1x);
  atomicAdd(&grad[3 * idx2 + 1], -term1y);
  atomicAdd(&grad[3 * idx2 + 2], -term1z);
}

}  // namespace rdkit_ports

static __device__ __forceinline__ real
bondStretchEnergy(const auto& pos, const int idx1, const int idx2, const real r0, const real kb) {
  constexpr real prefactor           = real{143.9325} / real{2.0};
  constexpr real csFactorDist        = -real{2.0};
  constexpr real csFactorDistSquared = real{7.0} / real{12.0} * csFactorDist * csFactorDist;

  const real distSquared = coordDistanceSquared(pos, idx1, idx2);
  const real distance    = NVMOLKIT_SQRT(static_cast<real>(distSquared));

  const real deltaR  = distance - r0;
  const real deltaR2 = deltaR * deltaR;
  return prefactor * kb * deltaR2 * (real{1.0} + csFactorDist * deltaR + csFactorDistSquared * deltaR2);
}

static __device__ __forceinline__ void bondStretchGradTo(const auto& pos,
                                                         const int   idx1,
                                                         const int   idx2,
                                                         const real  r0,
                                                         const real  kb,
                                                         const auto& sink) {
  constexpr real c1                          = real{143.9325};
  constexpr real cs                          = -real{2.0};
  constexpr real csFactorTimesSecondConstant = cs * real{1.5};
  constexpr real lastFactor                  = real{2.0} * real{7.0} / real{12.0} * cs * cs;

  real       dx, dy, dz;
  const real distanceSquared = coordDistanceSquaredWithComponents(pos, idx1, idx2, dx, dy, dz);
  const real invDist         = NVMOLKIT_RSQRT(distanceSquared);
  const real distance        = real{1.0} / invDist;
  const real deltaR          = distance - r0;

  const real de_dr =
    c1 * kb * deltaR * (real{1.0} + csFactorTimesSecondConstant * deltaR + lastFactor * deltaR * deltaR);

  real dE_dx, dE_dy, dE_dz;
  if (distance > real{0.0}) {
    dE_dx = de_dr * dx * invDist;
    dE_dy = de_dr * dy * invDist;
    dE_dz = de_dr * dz * invDist;
  } else {
    dE_dx = kb * real{0.01};
    dE_dy = kb * real{0.01};
    dE_dz = kb * real{0.01};
  }

  sink(idx1, dE_dx, dE_dy, dE_dz);

  sink(idx2, -dE_dx, -dE_dy, -dE_dz);
}

static __device__ __forceinline__ void bondStretchGrad(const auto& pos,
                                                       const int   idx1,
                                                       const int   idx2,
                                                       const real  r0,
                                                       const real  kb,
                                                       auto*       grad) {
  bondStretchGradTo(pos, idx1, idx2, r0, kb, atomicGradSink(grad));
}

static __device__ __forceinline__ real angleBendEnergy(const auto& pos,
                                                       const int   idx1,
                                                       const int   idx2,
                                                       const int   idx3,
                                                       const real  theta0,
                                                       const real  ka,
                                                       const bool  isLinear) {
  constexpr real prefactor = real{0.5} * real{143.9325} * degreeToRadian * degreeToRadian;
  constexpr real cb        = -real{0.4} * degreeToRadian;

  real       dx1, dy1, dz1, dx2, dy2, dz2;
  const real dist1Squared = coordDistanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const real dist2Squared = coordDistanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const real dist1        = NVMOLKIT_RSQRT(dist1Squared);
  const real dist2        = NVMOLKIT_RSQRT(dist2Squared);

  const real dot         = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const real cosTheta    = clamp(dot * (dist1 * dist2), real{-1.0}, real{1.0});
  const real theta       = radianToDegree * NVMOLKIT_ACOS(cosTheta);
  const real deltaTheta  = theta - theta0;
  const real deltaTheta2 = deltaTheta * deltaTheta;

  if (isLinear) {
    constexpr real linearPrefactor = real{143.9325};
    return linearPrefactor * ka * (real{1.0} + cosTheta);
  }
  return prefactor * ka * deltaTheta2 * (real{1.0} + cb * deltaTheta);
}

static __device__ __forceinline__ void angleBendGradTo(const int   idx1,
                                                       const int   idx2,
                                                       const int   idx3,
                                                       const real  theta0,
                                                       const real  ka,
                                                       const bool  isLinear,
                                                       const auto& pos,
                                                       const auto& sink) {
  constexpr real c1       = real{143.9325} * degreeToRadian;
  constexpr real cbFactor = -real{0.006981317} * real{1.5};
  real           dx1, dy1, dz1, dx2, dy2, dz2;
  const real     dist1Squared = coordDistanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const real     dist2Squared = coordDistanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const real     invDist1     = NVMOLKIT_RSQRT(dist1Squared);
  const real     invDist2     = NVMOLKIT_RSQRT(dist2Squared);

  const real dot        = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const real cosTheta   = clamp(dot * invDist1 * invDist2, -real{1.0}, real{1.0});
  const real sinThetaSq = real{1.0} - cosTheta * cosTheta;
  if (NVMOLKIT_IS_ZERO(sinThetaSq) || NVMOLKIT_IS_ZERO(dist1Squared) || NVMOLKIT_IS_ZERO(dist2Squared)) {
    return;
  }

  const real invNegSinTheta = -NVMOLKIT_RSQRT(sinThetaSq);
  const real theta          = radianToDegree * NVMOLKIT_ACOS(cosTheta);
  const real deltaTheta     = theta - theta0;

  real de_dDeltaTheta;

  if (isLinear) {
    constexpr real linearPrefactor = real{143.9325};
    de_dDeltaTheta                 = -linearPrefactor * ka * NVMOLKIT_SQRT(real{1.0} - (cosTheta * cosTheta));
  } else {
    de_dDeltaTheta = c1 * ka * deltaTheta * (real{1.0} + cbFactor * deltaTheta);
  }

  const real dxnorm1 = dx1 * invDist1;
  const real dynorm1 = dy1 * invDist1;
  const real dznorm1 = dz1 * invDist1;
  const real dxnorm2 = dx2 * invDist2;
  const real dynorm2 = dy2 * invDist2;
  const real dznorm2 = dz2 * invDist2;

  const real intermediate1 = invDist1 * (dxnorm2 - cosTheta * dxnorm1);
  const real intermediate2 = invDist1 * (dynorm2 - cosTheta * dynorm1);
  const real intermediate3 = invDist1 * (dznorm2 - cosTheta * dznorm1);
  const real intermediate4 = invDist2 * (dxnorm1 - cosTheta * dxnorm2);
  const real intermediate5 = invDist2 * (dynorm1 - cosTheta * dynorm2);
  const real intermediate6 = invDist2 * (dznorm1 - cosTheta * dznorm2);

  const real constantFactor = de_dDeltaTheta * invNegSinTheta;

  sink(idx1, constantFactor * intermediate1, constantFactor * intermediate2, constantFactor * intermediate3);

  sink(idx2,
       constantFactor * (-intermediate1 - intermediate4),
       constantFactor * (-intermediate2 - intermediate5),
       constantFactor * (-intermediate3 - intermediate6));

  sink(idx3, constantFactor * intermediate4, constantFactor * intermediate5, constantFactor * intermediate6);
}

static __device__ __forceinline__ void angleBendGrad(const int   idx1,
                                                     const int   idx2,
                                                     const int   idx3,
                                                     const real  theta0,
                                                     const real  ka,
                                                     const bool  isLinear,
                                                     const auto& pos,
                                                     auto*       grad) {
  angleBendGradTo(idx1, idx2, idx3, theta0, ka, isLinear, pos, atomicGradSink(grad));
}

static __device__ __forceinline__ real bendStretchEnergy(const auto& pos,
                                                         const int   idx1,
                                                         const int   idx2,
                                                         const int   idx3,
                                                         const real  theta0,
                                                         const real  restLen1,
                                                         const real  restLen2,
                                                         const real  forceConst1,
                                                         const real  forceConst2) {
  constexpr real prefactor = real{2.51210};

  real       dx1, dy1, dz1, dx2, dy2, dz2;
  const real dist1Squared = coordDistanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const real dist2Squared = coordDistanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const real dist1        = NVMOLKIT_SQRT(dist1Squared);
  const real dist2        = NVMOLKIT_SQRT(dist2Squared);

  const real dot      = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const real cosTheta = clamp(dot / (dist1 * dist2), real{-1.0}, real{1.0});
  const real theta    = 180 / static_cast<real>(M_PI) * NVMOLKIT_ACOS(cosTheta);

  const real deltaTheta = theta - theta0;
  const real deltaR1    = dist1 - restLen1;
  const real deltaR2    = dist2 - restLen2;

  return prefactor * deltaTheta * (deltaR1 * forceConst1 + deltaR2 * forceConst2);
}

static __device__ __forceinline__ void bendStretchGradTo(const auto& pos,
                                                         const int   idx1,
                                                         const int   idx2,
                                                         const int   idx3,
                                                         const real  theta0,
                                                         const real  restLen1,
                                                         const real  restLen2,
                                                         const real  forceConst1,
                                                         const real  forceConst2,
                                                         const auto& sink) {
  constexpr real prefactor = real{143.9325} * static_cast<real>(M_PI) / real{180.0};

  real       dx1, dy1, dz1, dx2, dy2, dz2;
  const real dist1Squared = coordDistanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const real dist2Squared = coordDistanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const real dist1        = NVMOLKIT_SQRT(dist1Squared);
  const real dist2        = NVMOLKIT_SQRT(dist2Squared);
  const real invDist1     = real{1.0} / dist1;
  const real invDist2     = real{1.0} / dist2;
  const real dot          = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const real cosTheta     = clamp(dot * invDist1 * invDist2, real{-1.0}, real{1.0});
  const real invSinTheta  = NVMOLKIT_FMIN(NVMOLKIT_RSQRT(real{1.0} - cosTheta * cosTheta), real{1.0e8});

  constexpr real bondFactor = real{180.0} / static_cast<real>(M_PI);
  const real     theta      = bondFactor * NVMOLKIT_ACOS(cosTheta);

  const real deltaTheta = theta - theta0;
  const real deltaR1    = dist1 - restLen1;
  const real deltaR2    = dist2 - restLen2;

  const real bondEnergyTerm = bondFactor * (forceConst1 * deltaR1 + forceConst2 * deltaR2);

  const real scaledDx1 = dx1 * invDist1;
  const real scaledDy1 = dy1 * invDist1;
  const real scaledDz1 = dz1 * invDist1;
  const real scaledDx2 = dx2 * invDist2;
  const real scaledDy2 = dy2 * invDist2;
  const real scaledDz2 = dz2 * invDist2;

  const real intermediate1 = invDist1 * (scaledDx2 - cosTheta * scaledDx1);
  const real intermediate2 = invDist1 * (scaledDy2 - cosTheta * scaledDy1);
  const real intermediate3 = invDist1 * (scaledDz2 - cosTheta * scaledDz1);
  const real intermediate4 = invDist2 * (scaledDx1 - cosTheta * scaledDx2);
  const real intermediate5 = invDist2 * (scaledDy1 - cosTheta * scaledDy2);
  const real intermediate6 = invDist2 * (scaledDz1 - cosTheta * scaledDz2);

  const real bondEnergyTimesInvSinTheta = bondEnergyTerm * invSinTheta;

  const real gradx1 = prefactor * (deltaTheta * scaledDx1 * forceConst1 - intermediate1 * bondEnergyTimesInvSinTheta);
  const real grady1 = prefactor * (deltaTheta * scaledDy1 * forceConst1 - intermediate2 * bondEnergyTimesInvSinTheta);
  const real gradz1 = prefactor * (deltaTheta * scaledDz1 * forceConst1 - intermediate3 * bondEnergyTimesInvSinTheta);

  const real gradx2 = prefactor * (-deltaTheta * (scaledDx1 * forceConst1 + scaledDx2 * forceConst2) +
                                   (intermediate1 + intermediate4) * bondEnergyTimesInvSinTheta);
  const real grady2 = prefactor * (-deltaTheta * (scaledDy1 * forceConst1 + scaledDy2 * forceConst2) +
                                   (intermediate2 + intermediate5) * bondEnergyTimesInvSinTheta);
  const real gradz2 = prefactor * (-deltaTheta * (scaledDz1 * forceConst1 + scaledDz2 * forceConst2) +
                                   (intermediate3 + intermediate6) * bondEnergyTimesInvSinTheta);

  const real gradx3 = prefactor * (deltaTheta * scaledDx2 * forceConst2 - intermediate4 * bondEnergyTimesInvSinTheta);
  const real grady3 = prefactor * (deltaTheta * scaledDy2 * forceConst2 - intermediate5 * bondEnergyTimesInvSinTheta);
  const real gradz3 = prefactor * (deltaTheta * scaledDz2 * forceConst2 - intermediate6 * bondEnergyTimesInvSinTheta);

  sink(idx1, gradx1, grady1, gradz1);

  sink(idx3, gradx3, grady3, gradz3);

  sink(idx2, gradx2, grady2, gradz2);
}

static __device__ __forceinline__ void bendStretchGrad(const auto& pos,
                                                       const int   idx1,
                                                       const int   idx2,
                                                       const int   idx3,
                                                       const real  theta0,
                                                       const real  restLen1,
                                                       const real  restLen2,
                                                       const real  forceConst1,
                                                       const real  forceConst2,
                                                       auto*       grad) {
  bendStretchGradTo(pos, idx1, idx2, idx3, theta0, restLen1, restLen2, forceConst1, forceConst2, atomicGradSink(grad));
}

static __device__ __forceinline__ real
oopBendEnergy(const auto& pos, const int idx1, const int idx2, const int idx3, const int idx4, const real koop) {
  constexpr real prefactor = real{0.5} * real{143.9325} * degreeToRadian * degreeToRadian;

  real       dxji, dyji, dzji, dxjk, dyjk, dzjk, dxjl, dyjl, dzjl;
  const real distSquaredJI = coordDistanceSquaredWithComponents(pos, idx1, idx2, dxji, dyji, dzji);
  const real distSquaredJK = coordDistanceSquaredWithComponents(pos, idx3, idx2, dxjk, dyjk, dzjk);
  const real distSquaredJL = coordDistanceSquaredWithComponents(pos, idx4, idx2, dxjl, dyjl, dzjl);

  const real invDistJI = NVMOLKIT_RSQRT(distSquaredJI);
  const real invDistJK = NVMOLKIT_RSQRT(distSquaredJK);
  const real invDistJL = NVMOLKIT_RSQRT(distSquaredJL);

  const real scaledDxJI = dxji * invDistJI;
  const real scaledDyJI = dyji * invDistJI;
  const real scaledDzJI = dzji * invDistJI;

  const real scaledDxJK = dxjk * invDistJK;
  const real scaledDyJK = dyjk * invDistJK;
  const real scaledDzJK = dzjk * invDistJK;

  const real scaledDxJL = dxjl * invDistJL;
  const real scaledDyJL = dyjl * invDistJL;
  const real scaledDzJL = dzjl * invDistJL;

  real crossX, crossY, crossZ;
  crossProduct(scaledDxJI, scaledDyJI, scaledDzJI, scaledDxJK, scaledDyJK, scaledDzJK, crossX, crossY, crossZ);
  const real invDistCross = NVMOLKIT_RSQRT(crossX * crossX + crossY * crossY + crossZ * crossZ);

  const real scaledCrossX = crossX * invDistCross;
  const real scaledCrossY = crossY * invDistCross;
  const real scaledCrossZ = crossZ * invDistCross;

  const real dotProduct = scaledCrossX * scaledDxJL + scaledCrossY * scaledDyJL + scaledCrossZ * scaledDzJL;
  const real chi        = radianToDegree * NVMOLKIT_ASIN(clamp(dotProduct, real{-1.0}, real{1.0}));

  return prefactor * koop * chi * chi;
}

static __device__ __forceinline__ real torsionEnergy(const auto& pos,
                                                     const int   idx1,
                                                     const int   idx2,
                                                     const int   idx3,
                                                     const int   idx4,
                                                     const real  V1,
                                                     const real  V2,
                                                     const real  V3) {
  const real dxIJ = coordOf(pos, idx1, 0) - coordOf(pos, idx2, 0);
  const real dyIJ = coordOf(pos, idx1, 1) - coordOf(pos, idx2, 1);
  const real dzIJ = coordOf(pos, idx1, 2) - coordOf(pos, idx2, 2);

  const real dxKJ = coordOf(pos, idx3, 0) - coordOf(pos, idx2, 0);
  const real dyKJ = coordOf(pos, idx3, 1) - coordOf(pos, idx2, 1);
  const real dzKJ = coordOf(pos, idx3, 2) - coordOf(pos, idx2, 2);

  const real dxLK = coordOf(pos, idx4, 0) - coordOf(pos, idx3, 0);
  const real dyLK = coordOf(pos, idx4, 1) - coordOf(pos, idx3, 1);
  const real dzLK = coordOf(pos, idx4, 2) - coordOf(pos, idx3, 2);

  const real crossIJKJx = dyIJ * dzKJ - dzIJ * dyKJ;
  const real crossIJKJy = dzIJ * dxKJ - dxIJ * dzKJ;
  const real crossIJKJz = dxIJ * dyKJ - dyIJ * dxKJ;

  const real crossJKLKx = -dyKJ * dzLK + dzKJ * dyLK;
  const real crossJKLKy = -dzKJ * dxLK + dxKJ * dzLK;
  const real crossJKLKz = -dxKJ * dyLK + dyKJ * dxLK;

  const real invCross1Norm =
    NVMOLKIT_RSQRT(crossIJKJx * crossIJKJx + crossIJKJy * crossIJKJy + crossIJKJz * crossIJKJz);
  const real invCross2Norm =
    NVMOLKIT_RSQRT(crossJKLKx * crossJKLKx + crossJKLKy * crossJKLKy + crossJKLKz * crossJKLKz);

  const real dotProduct = crossIJKJx * crossJKLKx + crossIJKJy * crossJKLKy + crossIJKJz * crossJKLKz;
  const real cosPhi     = dotProduct * invCross1Norm * invCross2Norm;
  const real phi        = NVMOLKIT_ACOS(clamp(cosPhi, real{-1.0}, real{1.0}));

  return real{0.5} * (V1 * (real{1.0} + cosPhi) + V2 * (real{1.0} - NVMOLKIT_COS(real{2.0} * phi)) +
                      V3 * (real{1.0} + NVMOLKIT_COS(real{3.0} * phi)));
}

static __device__ __forceinline__ real
vdwEnergy(const auto& pos, const int idx1, const int idx2, const real R_ij_star, const real wellDepth) {
  real R_ij_star2 = R_ij_star * R_ij_star;
  real R_ij_star7 = R_ij_star2 * R_ij_star2 * R_ij_star2 * R_ij_star;

  const real epsilon = wellDepth;

  const real distSquared = coordDistanceSquared(pos, idx1, idx2);
  const real dist        = NVMOLKIT_SQRT(distSquared);
  const real dist7       = distSquared * distSquared * distSquared * dist;

  const real term1        = real{1.07} * R_ij_star / (dist + real{0.07} * R_ij_star);
  const real term1Squared = term1 * term1;
  const real term1_7th    = term1Squared * term1Squared * term1Squared * term1;

  const real term2Fraction = real{1.12} * R_ij_star7 / (dist7 + real{0.12} * R_ij_star7);

  return epsilon * term1_7th * (term2Fraction - real{2.0});
}

static __device__ __forceinline__ real eleEnergy(const auto& pos,
                                                 const int   idx1,
                                                 const int   idx2,
                                                 const real  chargeTerm,
                                                 const int   dielModel,
                                                 const bool  is1_4) {
  constexpr real prefactor         = real{332.0716};
  constexpr real bufferingConstant = real{0.05};
  const real     distSquared       = coordDistanceSquared(pos, idx1, idx2);
  real           distTerm          = NVMOLKIT_SQRT(distSquared) + bufferingConstant;
  if (dielModel == 2) {
    distTerm *= distTerm;
  }
  real energy = prefactor * chargeTerm / (distTerm);
  if (is1_4) {
    energy *= real{0.75};
  }
  return energy;
}

//! Gradient of one electrostatic pair with respect to idx1; the idx2 gradient is its negation.
static __device__ __forceinline__ void eleGradVec(const auto& pos,
                                                  const int   idx1,
                                                  const int   idx2,
                                                  const real  chargeTerm,
                                                  const int   dielModel,
                                                  const bool  is1_4,
                                                  real&       dE_dx,
                                                  real&       dE_dy,
                                                  real&       dE_dz) {
  constexpr real prefactor         = real{332.0716};
  constexpr real bufferingConstant = real{0.05};

  const real distSquared = coordDistanceSquared(pos, idx1, idx2);
  const real invDistance = NVMOLKIT_RSQRT(distSquared);
  const real distance    = real{1.0} / invDistance;
  const real rBuf        = distance + bufferingConstant;
  real       numerator   = -prefactor * chargeTerm;
  // E_1 = q / (r+b)         -> dE/dr = -q / (r+b)^2
  // E_2 = q / (r+b)^2       -> dE/dr = -2q / (r+b)^3
  real       denominator = rBuf * rBuf;
  if (dielModel == 2) {
    numerator *= 2;
    denominator *= rBuf;
  }

  real dE_dr = numerator / denominator;
  if (is1_4) {
    dE_dr *= real{0.75};
  }

  dE_dx = dE_dr * (coordOf(pos, idx1, 0) - coordOf(pos, idx2, 0)) * invDistance;
  dE_dy = dE_dr * (coordOf(pos, idx1, 1) - coordOf(pos, idx2, 1)) * invDistance;
  dE_dz = dE_dr * (coordOf(pos, idx1, 2) - coordOf(pos, idx2, 2)) * invDistance;
}

static __device__ __forceinline__ void eleGrad(const auto& pos,
                                               const int   idx1,
                                               const int   idx2,
                                               const real  chargeTerm,
                                               const int   dielModel,
                                               const bool  is1_4,
                                               auto*       grad) {
  real dE_dx, dE_dy, dE_dz;
  eleGradVec(pos, idx1, idx2, chargeTerm, dielModel, is1_4, dE_dx, dE_dy, dE_dz);

  atomicAdd(&grad[3 * idx1 + 0], dE_dx);
  atomicAdd(&grad[3 * idx1 + 1], dE_dy);
  atomicAdd(&grad[3 * idx1 + 2], dE_dz);

  atomicAdd(&grad[3 * idx2 + 0], -dE_dx);
  atomicAdd(&grad[3 * idx2 + 1], -dE_dy);
  atomicAdd(&grad[3 * idx2 + 2], -dE_dz);
}

static __device__ __forceinline__ real normalizeAngleDeg(real angleDeg) {
  angleDeg = NVMOLKIT_FMOD(angleDeg, real{360.0});
  if (angleDeg < real{-180.0}) {
    angleDeg += real{360.0};
  } else if (angleDeg > real{180.0}) {
    angleDeg -= real{360.0};
  }
  return angleDeg;
}

static __device__ __forceinline__ real distanceConstraintEnergy(const auto& pos,
                                                                const int   idx1,
                                                                const int   idx2,
                                                                const real  minLen,
                                                                const real  maxLen,
                                                                const real  forceConstant) {
  const real distance2Val = coordDistanceSquared(pos, idx1, idx2);
  real       difference   = real{0.0};
  if (distance2Val < minLen * minLen) {
    difference = minLen - NVMOLKIT_SQRT(distance2Val);
  } else if (distance2Val > maxLen * maxLen) {
    difference = NVMOLKIT_SQRT(distance2Val) - maxLen;
  } else {
    return real{0.0};
  }
  return real{0.5} * forceConstant * difference * difference;
}

static __device__ __forceinline__ void distanceConstraintGradTo(const auto& pos,
                                                                const int   idx1,
                                                                const int   idx2,
                                                                const real  minLen,
                                                                const real  maxLen,
                                                                const real  forceConstant,
                                                                const auto& sink) {
  const real distance2Val = coordDistanceSquared(pos, idx1, idx2);
  real       preFactor    = real{0.0};
  real       distance     = real{0.0};
  if (distance2Val < minLen * minLen) {
    distance  = NVMOLKIT_SQRT(distance2Val);
    preFactor = distance - minLen;
  } else if (distance2Val > maxLen * maxLen) {
    distance  = NVMOLKIT_SQRT(distance2Val);
    preFactor = distance - maxLen;
  } else {
    return;
  }
  preFactor *= forceConstant;
  preFactor /= NVMOLKIT_FMAX(real{1.0e-8}, distance);
  const real dGradX = preFactor * (coordOf(pos, idx1, 0) - coordOf(pos, idx2, 0));
  const real dGradY = preFactor * (coordOf(pos, idx1, 1) - coordOf(pos, idx2, 1));
  const real dGradZ = preFactor * (coordOf(pos, idx1, 2) - coordOf(pos, idx2, 2));
  sink(idx1, dGradX, dGradY, dGradZ);
  sink(idx2, -dGradX, -dGradY, -dGradZ);
}

static __device__ __forceinline__ void distanceConstraintGrad(const auto& pos,
                                                              const int   idx1,
                                                              const int   idx2,
                                                              const real  minLen,
                                                              const real  maxLen,
                                                              const real  forceConstant,
                                                              auto*       grad) {
  distanceConstraintGradTo(pos, idx1, idx2, minLen, maxLen, forceConstant, atomicGradSink(grad));
}

static __device__ __forceinline__ real positionConstraintEnergy(const auto& pos,
                                                                const int   idx,
                                                                const real  refX,
                                                                const real  refY,
                                                                const real  refZ,
                                                                const real  maxDispl,
                                                                const real  forceConstant) {
  const real dx       = coordOf(pos, idx, 0) - refX;
  const real dy       = coordOf(pos, idx, 1) - refY;
  const real dz       = coordOf(pos, idx, 2) - refZ;
  const real dist     = NVMOLKIT_SQRT(dx * dx + dy * dy + dz * dz);
  const real distTerm = NVMOLKIT_FMAX(dist - maxDispl, real{0.0});
  return real{0.5} * forceConstant * distTerm * distTerm;
}

static __device__ __forceinline__ void positionConstraintGradTo(const auto& pos,
                                                                const int   idx,
                                                                const real  refX,
                                                                const real  refY,
                                                                const real  refZ,
                                                                const real  maxDispl,
                                                                const real  forceConstant,
                                                                const auto& sink) {
  const real dx   = coordOf(pos, idx, 0) - refX;
  const real dy   = coordOf(pos, idx, 1) - refY;
  const real dz   = coordOf(pos, idx, 2) - refZ;
  const real dist = NVMOLKIT_SQRT(dx * dx + dy * dy + dz * dz);
  if (dist <= maxDispl) {
    return;
  }
  const real preFactor = (dist - maxDispl) * forceConstant / NVMOLKIT_FMAX(dist, real{1.0e-8});
  sink(idx, preFactor * dx, preFactor * dy, preFactor * dz);
}

static __device__ __forceinline__ void positionConstraintGrad(const auto& pos,
                                                              const int   idx,
                                                              const real  refX,
                                                              const real  refY,
                                                              const real  refZ,
                                                              const real  maxDispl,
                                                              const real  forceConstant,
                                                              auto*       grad) {
  positionConstraintGradTo(pos, idx, refX, refY, refZ, maxDispl, forceConstant, atomicGradSink(grad));
}

static __device__ __forceinline__ real computeAngleConstraintTerm(const real angle,
                                                                  const real minAngleDeg,
                                                                  const real maxAngleDeg) {
  real angleTerm = real{0.0};
  if (angle < minAngleDeg) {
    angleTerm = angle - minAngleDeg;
  } else if (angle > maxAngleDeg) {
    angleTerm = angle - maxAngleDeg;
  }
  return angleTerm;
}

static __device__ __forceinline__ real angleConstraintEnergy(const auto& pos,
                                                             const int   idx1,
                                                             const int   idx2,
                                                             const int   idx3,
                                                             const real  minAngleDeg,
                                                             const real  maxAngleDeg,
                                                             const real  forceConstant) {
  const real p1x = coordOf(pos, idx1, 0);
  const real p1y = coordOf(pos, idx1, 1);
  const real p1z = coordOf(pos, idx1, 2);
  const real p2x = coordOf(pos, idx2, 0);
  const real p2y = coordOf(pos, idx2, 1);
  const real p2z = coordOf(pos, idx2, 2);
  const real p3x = coordOf(pos, idx3, 0);
  const real p3y = coordOf(pos, idx3, 1);
  const real p3z = coordOf(pos, idx3, 2);

  const real r1x        = p1x - p2x;
  const real r1y        = p1y - p2y;
  const real r1z        = p1z - p2z;
  const real r2x        = p3x - p2x;
  const real r2y        = p3y - p2y;
  const real r2z        = p3z - p2z;
  const real rLengthSq1 = NVMOLKIT_FMAX(real{1.0e-5}, r1x * r1x + r1y * r1y + r1z * r1z);
  const real rLengthSq2 = NVMOLKIT_FMAX(real{1.0e-5}, r2x * r2x + r2y * r2y + r2z * r2z);
  real       cosTheta   = (r1x * r2x + r1y * r2y + r1z * r2z) / NVMOLKIT_SQRT(rLengthSq1 * rLengthSq2);
  cosTheta              = clamp(cosTheta, real{-1.0}, real{1.0});
  const real angle      = radianToDegree * NVMOLKIT_ACOS(cosTheta);
  const real angleTerm  = computeAngleConstraintTerm(angle, minAngleDeg, maxAngleDeg);
  return forceConstant * angleTerm * angleTerm;
}

static __device__ __forceinline__ void angleConstraintGradTo(const auto& pos,
                                                             const int   idx1,
                                                             const int   idx2,
                                                             const int   idx3,
                                                             const real  minAngleDeg,
                                                             const real  maxAngleDeg,
                                                             const real  forceConstant,
                                                             const auto& sink) {
  const real p1x = coordOf(pos, idx1, 0);
  const real p1y = coordOf(pos, idx1, 1);
  const real p1z = coordOf(pos, idx1, 2);
  const real p2x = coordOf(pos, idx2, 0);
  const real p2y = coordOf(pos, idx2, 1);
  const real p2z = coordOf(pos, idx2, 2);
  const real p3x = coordOf(pos, idx3, 0);
  const real p3y = coordOf(pos, idx3, 1);
  const real p3z = coordOf(pos, idx3, 2);

  const real r1x        = p1x - p2x;
  const real r1y        = p1y - p2y;
  const real r1z        = p1z - p2z;
  const real r2x        = p3x - p2x;
  const real r2y        = p3y - p2y;
  const real r2z        = p3z - p2z;
  const real rLengthSq1 = NVMOLKIT_FMAX(real{1.0e-5}, r1x * r1x + r1y * r1y + r1z * r1z);
  const real rLengthSq2 = NVMOLKIT_FMAX(real{1.0e-5}, r2x * r2x + r2y * r2y + r2z * r2z);
  const real invDist1   = rsqrt(rLengthSq1);
  const real invDist2   = rsqrt(rLengthSq2);
  real       cosTheta   = (r1x * r2x + r1y * r2y + r1z * r2z) * invDist1 * invDist2;
  cosTheta              = clamp(cosTheta, real{-1.0}, real{1.0});
  const real angle      = radianToDegree * NVMOLKIT_ACOS(cosTheta);
  const real angleTerm  = computeAngleConstraintTerm(angle, minAngleDeg, maxAngleDeg);
  if (NVMOLKIT_IS_ZERO(angleTerm)) {
    return;
  }

  const real rpX       = r2y * r1z - r2z * r1y;
  const real rpY       = r2z * r1x - r2x * r1z;
  const real rpZ       = r2x * r1y - r2y * r1x;
  const real rpLength  = NVMOLKIT_FMAX(real{1.0e-5}, NVMOLKIT_SQRT(rpX * rpX + rpY * rpY + rpZ * rpZ));
  const real dE_dTheta = real{2.0} * radianToDegree * forceConstant * angleTerm;
  const real prefactor = dE_dTheta / rpLength;
  const real t0        = -prefactor / rLengthSq1;
  const real t1        = prefactor / rLengthSq2;

  const real c0x = r1y * rpZ - r1z * rpY;
  const real c0y = r1z * rpX - r1x * rpZ;
  const real c0z = r1x * rpY - r1y * rpX;
  const real c1x = r2y * rpZ - r2z * rpY;
  const real c1y = r2z * rpX - r2x * rpZ;
  const real c1z = r2x * rpY - r2y * rpX;

  const real dedp0x = c0x * t0;
  const real dedp0y = c0y * t0;
  const real dedp0z = c0z * t0;
  const real dedp2x = c1x * t1;
  const real dedp2y = c1y * t1;
  const real dedp2z = c1z * t1;
  const real dedp1x = -dedp0x - dedp2x;
  const real dedp1y = -dedp0y - dedp2y;
  const real dedp1z = -dedp0z - dedp2z;

  sink(idx1, dedp0x, dedp0y, dedp0z);
  sink(idx2, dedp1x, dedp1y, dedp1z);
  sink(idx3, dedp2x, dedp2y, dedp2z);
}

static __device__ __forceinline__ void angleConstraintGrad(const auto& pos,
                                                           const int   idx1,
                                                           const int   idx2,
                                                           const int   idx3,
                                                           const real  minAngleDeg,
                                                           const real  maxAngleDeg,
                                                           const real  forceConstant,
                                                           auto*       grad) {
  angleConstraintGradTo(pos, idx1, idx2, idx3, minAngleDeg, maxAngleDeg, forceConstant, atomicGradSink(grad));
}

static __device__ __forceinline__ real computeDihedralConstraintTerm(real       dihedral,
                                                                     const real minDihedralDeg,
                                                                     const real maxDihedralDeg) {
  real dihedralTarget = dihedral;
  if (!(dihedral > minDihedralDeg && dihedral < maxDihedralDeg) &&
      !(dihedral > minDihedralDeg && minDihedralDeg > maxDihedralDeg) &&
      !(dihedral < maxDihedralDeg && minDihedralDeg > maxDihedralDeg)) {
    real dihedralMinTarget = normalizeAngleDeg(dihedral - minDihedralDeg);
    real dihedralMaxTarget = normalizeAngleDeg(dihedral - maxDihedralDeg);
    if (fabs(dihedralMinTarget) < fabs(dihedralMaxTarget)) {
      dihedralTarget = minDihedralDeg;
    } else {
      dihedralTarget = maxDihedralDeg;
    }
  }
  return normalizeAngleDeg(dihedral - dihedralTarget);
}

static __device__ __forceinline__ real computeSignedDihedral(const auto& pos,
                                                             const int   idx1,
                                                             const int   idx2,
                                                             const int   idx3,
                                                             const int   idx4,
                                                             real*       cosPhiOut = nullptr,
                                                             real        r[4][3]   = nullptr,
                                                             real        t[2][3]   = nullptr,
                                                             real        d[2]      = nullptr) {
  real localR[4][3];
  real localT[2][3];
  real localD[2];
  if (r == nullptr) {
    r = localR;
  }
  if (t == nullptr) {
    t = localT;
  }
  if (d == nullptr) {
    d = localD;
  }
  r[0][0] = coordOf(pos, idx1, 0) - coordOf(pos, idx2, 0);
  r[0][1] = coordOf(pos, idx1, 1) - coordOf(pos, idx2, 1);
  r[0][2] = coordOf(pos, idx1, 2) - coordOf(pos, idx2, 2);
  r[1][0] = coordOf(pos, idx3, 0) - coordOf(pos, idx2, 0);
  r[1][1] = coordOf(pos, idx3, 1) - coordOf(pos, idx2, 1);
  r[1][2] = coordOf(pos, idx3, 2) - coordOf(pos, idx2, 2);
  r[2][0] = -r[1][0];
  r[2][1] = -r[1][1];
  r[2][2] = -r[1][2];
  r[3][0] = coordOf(pos, idx4, 0) - coordOf(pos, idx3, 0);
  r[3][1] = coordOf(pos, idx4, 1) - coordOf(pos, idx3, 1);
  r[3][2] = coordOf(pos, idx4, 2) - coordOf(pos, idx3, 2);

  crossProduct(r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], t[0][0], t[0][1], t[0][2]);
  d[0] = NVMOLKIT_FMAX(NVMOLKIT_SQRT(t[0][0] * t[0][0] + t[0][1] * t[0][1] + t[0][2] * t[0][2]), real{1.0e-5});
  t[0][0] /= d[0];
  t[0][1] /= d[0];
  t[0][2] /= d[0];
  crossProduct(r[2][0], r[2][1], r[2][2], r[3][0], r[3][1], r[3][2], t[1][0], t[1][1], t[1][2]);
  d[1] = NVMOLKIT_FMAX(NVMOLKIT_SQRT(t[1][0] * t[1][0] + t[1][1] * t[1][1] + t[1][2] * t[1][2]), real{1.0e-5});
  t[1][0] /= d[1];
  t[1][1] /= d[1];
  t[1][2] /= d[1];
  const real cosPhi = clamp(t[0][0] * t[1][0] + t[0][1] * t[1][1] + t[0][2] * t[1][2], real{-1.0}, real{1.0});
  if (cosPhiOut != nullptr) {
    *cosPhiOut = cosPhi;
  }
  real mX, mY, mZ;
  crossProduct(t[0][0], t[0][1], t[0][2], r[1][0], r[1][1], r[1][2], mX, mY, mZ);
  const real mLength = NVMOLKIT_FMAX(NVMOLKIT_SQRT(mX * mX + mY * mY + mZ * mZ), real{1.0e-5});
  return -NVMOLKIT_ATAN2((mX * t[1][0] + mY * t[1][1] + mZ * t[1][2]) / mLength, cosPhi);
}

static __device__ __forceinline__ real torsionConstraintEnergy(const auto& pos,
                                                               const int   idx1,
                                                               const int   idx2,
                                                               const int   idx3,
                                                               const int   idx4,
                                                               const real  minDihedralDeg,
                                                               const real  maxDihedralDeg,
                                                               const real  forceConstant) {
  const real dihedral     = radianToDegree * computeSignedDihedral(pos, idx1, idx2, idx3, idx4);
  const real dihedralTerm = computeDihedralConstraintTerm(dihedral, minDihedralDeg, maxDihedralDeg);
  return forceConstant * dihedralTerm * dihedralTerm;
}

static __device__ __forceinline__ void torsionConstraintGradTo(const auto& pos,
                                                               const int   idx1,
                                                               const int   idx2,
                                                               const int   idx3,
                                                               const int   idx4,
                                                               const real  minDihedralDeg,
                                                               const real  maxDihedralDeg,
                                                               const real  forceConstant,
                                                               const auto& sink) {
  real       r[4][3];
  real       t[2][3];
  real       d[2];
  const real dihedral     = radianToDegree * computeSignedDihedral(pos, idx1, idx2, idx3, idx4, nullptr, r, t, d);
  const real dihedralTerm = computeDihedralConstraintTerm(dihedral, minDihedralDeg, maxDihedralDeg);
  if (NVMOLKIT_IS_ZERO(dihedralTerm)) {
    return;
  }
  const real dE_dPhi = real{2.0} * radianToDegree * forceConstant * dihedralTerm;

  const real d23       = NVMOLKIT_SQRT(coordDistanceSquared(pos, idx2, idx3));
  const real prefactor = dE_dPhi / NVMOLKIT_FMAX(d23, real{1.0e-8});

  real tt0[3], tt1[3];
  crossProduct(r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], tt0[0], tt0[1], tt0[2]);
  crossProduct(r[2][0], r[2][1], r[2][2], r[3][0], r[3][1], r[3][2], tt1[0], tt1[1], tt1[2]);
  const real tt0LenSq = NVMOLKIT_FMAX(tt0[0] * tt0[0] + tt0[1] * tt0[1] + tt0[2] * tt0[2], real{1.0e-8});
  const real tt1LenSq = NVMOLKIT_FMAX(tt1[0] * tt1[0] + tt1[1] * tt1[1] + tt1[2] * tt1[2], real{1.0e-8});

  real tmp0[3], tmp1[3];
  crossProduct(tt0[0], tt0[1], tt0[2], r[2][0], r[2][1], r[2][2], tmp0[0], tmp0[1], tmp0[2]);
  crossProduct(tt1[0], tt1[1], tt1[2], r[1][0], r[1][1], r[1][2], tmp1[0], tmp1[1], tmp1[2]);
  const real dedt0[3] = {tmp0[0] / tt0LenSq * prefactor,
                         tmp0[1] / tt0LenSq * prefactor,
                         tmp0[2] / tt0LenSq * prefactor};
  const real dedt1[3] = {tmp1[0] / tt1LenSq * prefactor,
                         tmp1[1] / tt1LenSq * prefactor,
                         tmp1[2] / tt1LenSq * prefactor};

  const real r31[3] = {static_cast<real>(coordOf(pos, idx3, 0)) - static_cast<real>(coordOf(pos, idx1, 0)),
                       static_cast<real>(coordOf(pos, idx3, 1)) - static_cast<real>(coordOf(pos, idx1, 1)),
                       static_cast<real>(coordOf(pos, idx3, 2)) - static_cast<real>(coordOf(pos, idx1, 2))};
  const real r42[3] = {static_cast<real>(coordOf(pos, idx4, 0)) - static_cast<real>(coordOf(pos, idx2, 0)),
                       static_cast<real>(coordOf(pos, idx4, 1)) - static_cast<real>(coordOf(pos, idx2, 1)),
                       static_cast<real>(coordOf(pos, idx4, 2)) - static_cast<real>(coordOf(pos, idx2, 2))};

  real dedp0[3], dedp1[3], dedp2[3], dedp3[3];
  crossProduct(r[2][0], r[2][1], r[2][2], dedt0[0], dedt0[1], dedt0[2], dedp0[0], dedp0[1], dedp0[2]);

  real r31Cross[3], r3Cross[3];
  crossProduct(r31[0], r31[1], r31[2], dedt0[0], dedt0[1], dedt0[2], r31Cross[0], r31Cross[1], r31Cross[2]);
  crossProduct(r[3][0], r[3][1], r[3][2], dedt1[0], dedt1[1], dedt1[2], r3Cross[0], r3Cross[1], r3Cross[2]);
  dedp1[0] = r31Cross[0] - r3Cross[0];
  dedp1[1] = r31Cross[1] - r3Cross[1];
  dedp1[2] = r31Cross[2] - r3Cross[2];

  real r0Cross[3], r42Cross[3];
  crossProduct(r[0][0], r[0][1], r[0][2], dedt0[0], dedt0[1], dedt0[2], r0Cross[0], r0Cross[1], r0Cross[2]);
  crossProduct(r42[0], r42[1], r42[2], dedt1[0], dedt1[1], dedt1[2], r42Cross[0], r42Cross[1], r42Cross[2]);
  dedp2[0] = r0Cross[0] + r42Cross[0];
  dedp2[1] = r0Cross[1] + r42Cross[1];
  dedp2[2] = r0Cross[2] + r42Cross[2];

  crossProduct(r[2][0], r[2][1], r[2][2], dedt1[0], dedt1[1], dedt1[2], dedp3[0], dedp3[1], dedp3[2]);

  sink(idx1, dedp0[0], dedp0[1], dedp0[2]);
  sink(idx2, dedp1[0], dedp1[1], dedp1[2]);
  sink(idx3, dedp2[0], dedp2[1], dedp2[2]);
  sink(idx4, dedp3[0], dedp3[1], dedp3[2]);
}

static __device__ __forceinline__ void torsionConstraintGrad(const auto& pos,
                                                             const int   idx1,
                                                             const int   idx2,
                                                             const int   idx3,
                                                             const int   idx4,
                                                             const real  minDihedralDeg,
                                                             const real  maxDihedralDeg,
                                                             const real  forceConstant,
                                                             auto*       grad) {
  torsionConstraintGradTo(pos,
                          idx1,
                          idx2,
                          idx3,
                          idx4,
                          minDihedralDeg,
                          maxDihedralDeg,
                          forceConstant,
                          atomicGradSink(grad));
}

template <int stride, bool HasConstraints, typename Terms>
static __device__ __inline__ real molEnergy(const Terms&                   terms,
                                            const BatchedIndicesDevicePtr& systemIndices,
                                            const auto*                    molCoords,
                                            const int                      molIdx,
                                            const int                      tid) {
  const int atomStart = systemIndices.atomStarts[molIdx];

  real energy = real{0.0};

  const auto& [idx1s, idx2s, r0s, kbs] = terms.bondTerms;
  const int bondStart                  = systemIndices.bondTermStarts[molIdx];
  const int bondEnd                    = systemIndices.bondTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = bondStart + tid; i < bondEnd; i += stride) {
    const int localIdx1 = idx1s[i] - atomStart;
    const int localIdx2 = idx2s[i] - atomStart;
    energy += bondStretchEnergy(molCoords, localIdx1, localIdx2, r0s[i], kbs[i]);
  }

  const auto& [a_idx1s, a_idx2s, a_idx3s, theta0s, kas, isLinears] = terms.angleTerms;
  const int angleStart                                             = systemIndices.angleTermStarts[molIdx];
  const int angleEnd                                               = systemIndices.angleTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = angleStart + tid; i < angleEnd; i += stride) {
    const int  localIdx1 = a_idx1s[i] - atomStart;
    const int  localIdx2 = a_idx2s[i] - atomStart;
    const int  localIdx3 = a_idx3s[i] - atomStart;
    const bool isLinear  = static_cast<bool>(isLinears[i]);
    energy += angleBendEnergy(molCoords, localIdx1, localIdx2, localIdx3, theta0s[i], kas[i], isLinear);
  }

  const auto& [bs_idx1s, bs_idx2s, bs_idx3s, bs_theta0s, restLen1s, restLen2s, forceConst1s, forceConst2s] =
    terms.bendTerms;
  const int bendStart = systemIndices.bendTermStarts[molIdx];
  const int bendEnd   = systemIndices.bendTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = bendStart + tid; i < bendEnd; i += stride) {
    const int localIdx1 = bs_idx1s[i] - atomStart;
    const int localIdx2 = bs_idx2s[i] - atomStart;
    const int localIdx3 = bs_idx3s[i] - atomStart;
    energy += bendStretchEnergy(molCoords,
                                localIdx1,
                                localIdx2,
                                localIdx3,
                                bs_theta0s[i],
                                restLen1s[i],
                                restLen2s[i],
                                forceConst1s[i],
                                forceConst2s[i]);
  }

  const auto& [o_idx1s, o_idx2s, o_idx3s, o_idx4s, koops] = terms.oopTerms;
  const int oopStart                                      = systemIndices.oopTermStarts[molIdx];
  const int oopEnd                                        = systemIndices.oopTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = oopStart + tid; i < oopEnd; i += stride) {
    const int localIdx1 = o_idx1s[i] - atomStart;
    const int localIdx2 = o_idx2s[i] - atomStart;
    const int localIdx3 = o_idx3s[i] - atomStart;
    const int localIdx4 = o_idx4s[i] - atomStart;
    energy += oopBendEnergy(molCoords, localIdx1, localIdx2, localIdx3, localIdx4, koops[i]);
  }

  const auto& [t_idx1s, t_idx2s, t_idx3s, t_idx4s, V1s, V2s, V3s] = terms.torsionTerms;
  const int torsionStart                                          = systemIndices.torsionTermStarts[molIdx];
  const int torsionEnd                                            = systemIndices.torsionTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = torsionStart + tid; i < torsionEnd; i += stride) {
    const int localIdx1 = t_idx1s[i] - atomStart;
    const int localIdx2 = t_idx2s[i] - atomStart;
    const int localIdx3 = t_idx3s[i] - atomStart;
    const int localIdx4 = t_idx4s[i] - atomStart;
    energy += torsionEnergy(molCoords, localIdx1, localIdx2, localIdx3, localIdx4, V1s[i], V2s[i], V3s[i]);
  }

  const auto& [v_idx1s, v_idx2s, R_ij_stars, wellDepths] = terms.vdwTerms;
  const int vdwStart                                     = systemIndices.vdwTermStarts[molIdx];
  const int vdwEnd                                       = systemIndices.vdwTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = vdwStart + tid; i < vdwEnd; i += stride) {
    const int localIdx1 = v_idx1s[i] - atomStart;
    const int localIdx2 = v_idx2s[i] - atomStart;
    energy += vdwEnergy(molCoords, localIdx1, localIdx2, R_ij_stars[i], wellDepths[i]);
  }

  const auto& [e_idx1s, e_idx2s, chargeTerms, dielModels, is1_4s] = terms.eleTerms;
  const int eleStart                                              = systemIndices.eleTermStarts[molIdx];
  const int eleEnd                                                = systemIndices.eleTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = eleStart + tid; i < eleEnd; i += stride) {
    const int  localIdx1 = e_idx1s[i] - atomStart;
    const int  localIdx2 = e_idx2s[i] - atomStart;
    const int  dielModel = static_cast<int>(dielModels[i]);
    const bool is14      = is1_4s[i] > 0;
    energy += eleEnergy(molCoords, localIdx1, localIdx2, chargeTerms[i], dielModel, is14);
  }

  if constexpr (HasConstraints) {
    const auto& [dc_idx1s, dc_idx2s, minLens, maxLens, dcForceConstants] = terms.distanceConstraintTerms;
    const int dcStart = systemIndices.distanceConstraintTermStarts[molIdx];
    const int dcEnd   = systemIndices.distanceConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = dcStart + tid; i < dcEnd; i += stride) {
      const int localIdx1 = dc_idx1s[i] - atomStart;
      const int localIdx2 = dc_idx2s[i] - atomStart;
      energy += distanceConstraintEnergy(molCoords, localIdx1, localIdx2, minLens[i], maxLens[i], dcForceConstants[i]);
    }

    const auto& [pc_idxs, refXs, refYs, refZs, maxDispls, pcForceConstants] = terms.positionConstraintTerms;
    const int pcStart = systemIndices.positionConstraintTermStarts[molIdx];
    const int pcEnd   = systemIndices.positionConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = pcStart + tid; i < pcEnd; i += stride) {
      const int localIdx = pc_idxs[i] - atomStart;
      energy +=
        positionConstraintEnergy(molCoords, localIdx, refXs[i], refYs[i], refZs[i], maxDispls[i], pcForceConstants[i]);
    }

    const auto& [ac_idx1s, ac_idx2s, ac_idx3s, minAngleDegs, maxAngleDegs, acForceConstants] =
      terms.angleConstraintTerms;
    const int acStart = systemIndices.angleConstraintTermStarts[molIdx];
    const int acEnd   = systemIndices.angleConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = acStart + tid; i < acEnd; i += stride) {
      const int localIdx1 = ac_idx1s[i] - atomStart;
      const int localIdx2 = ac_idx2s[i] - atomStart;
      const int localIdx3 = ac_idx3s[i] - atomStart;
      energy += angleConstraintEnergy(molCoords,
                                      localIdx1,
                                      localIdx2,
                                      localIdx3,
                                      minAngleDegs[i],
                                      maxAngleDegs[i],
                                      acForceConstants[i]);
    }

    const auto& [tc_idx1s, tc_idx2s, tc_idx3s, tc_idx4s, minDihedralDegs, maxDihedralDegs, tcForceConstants] =
      terms.torsionConstraintTerms;
    const int tcStart = systemIndices.torsionConstraintTermStarts[molIdx];
    const int tcEnd   = systemIndices.torsionConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = tcStart + tid; i < tcEnd; i += stride) {
      const int localIdx1 = tc_idx1s[i] - atomStart;
      const int localIdx2 = tc_idx2s[i] - atomStart;
      const int localIdx3 = tc_idx3s[i] - atomStart;
      const int localIdx4 = tc_idx4s[i] - atomStart;
      energy += torsionConstraintEnergy(molCoords,
                                        localIdx1,
                                        localIdx2,
                                        localIdx3,
                                        localIdx4,
                                        minDihedralDegs[i],
                                        maxDihedralDegs[i],
                                        tcForceConstants[i]);
    }
  }

  return energy;
}

template <int stride, bool HasConstraints, typename Terms>
static __device__ __inline__ void molGrad(const Terms&                   terms,
                                          const BatchedIndicesDevicePtr& systemIndices,
                                          const auto*                    molCoords,
                                          auto*                          grad,
                                          const int                      molIdx,
                                          const int                      tid) {
  const int atomStart = systemIndices.atomStarts[molIdx];

  const auto& [idx1s, idx2s, r0s, kbs] = terms.bondTerms;
  const int bondStart                  = systemIndices.bondTermStarts[molIdx];
  const int bondEnd                    = systemIndices.bondTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = bondStart + tid; i < bondEnd; i += stride) {
    const int localIdx1 = idx1s[i] - atomStart;
    const int localIdx2 = idx2s[i] - atomStart;
    bondStretchGrad(molCoords, localIdx1, localIdx2, r0s[i], kbs[i], grad);
  }

  const auto& [a_idx1s, a_idx2s, a_idx3s, theta0s, kas, isLinears] = terms.angleTerms;
  const int angleStart                                             = systemIndices.angleTermStarts[molIdx];
  const int angleEnd                                               = systemIndices.angleTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = angleStart + tid; i < angleEnd; i += stride) {
    const int  localIdx1 = a_idx1s[i] - atomStart;
    const int  localIdx2 = a_idx2s[i] - atomStart;
    const int  localIdx3 = a_idx3s[i] - atomStart;
    const bool isLinear  = static_cast<bool>(isLinears[i]);
    angleBendGrad(localIdx1, localIdx2, localIdx3, theta0s[i], kas[i], isLinear, molCoords, grad);
  }

  const auto& [bs_idx1s, bs_idx2s, bs_idx3s, bs_theta0s, restLen1s, restLen2s, forceConst1s, forceConst2s] =
    terms.bendTerms;
  const int bendStart = systemIndices.bendTermStarts[molIdx];
  const int bendEnd   = systemIndices.bendTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = bendStart + tid; i < bendEnd; i += stride) {
    const int localIdx1 = bs_idx1s[i] - atomStart;
    const int localIdx2 = bs_idx2s[i] - atomStart;
    const int localIdx3 = bs_idx3s[i] - atomStart;
    bendStretchGrad(molCoords,
                    localIdx1,
                    localIdx2,
                    localIdx3,
                    bs_theta0s[i],
                    restLen1s[i],
                    restLen2s[i],
                    forceConst1s[i],
                    forceConst2s[i],
                    grad);
  }

  const auto& [o_idx1s, o_idx2s, o_idx3s, o_idx4s, koops] = terms.oopTerms;
  const int oopStart                                      = systemIndices.oopTermStarts[molIdx];
  const int oopEnd                                        = systemIndices.oopTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = oopStart + tid; i < oopEnd; i += stride) {
    const int localIdx1 = o_idx1s[i] - atomStart;
    const int localIdx2 = o_idx2s[i] - atomStart;
    const int localIdx3 = o_idx3s[i] - atomStart;
    const int localIdx4 = o_idx4s[i] - atomStart;
    rdkit_ports::oopGrad(molCoords, localIdx1, localIdx2, localIdx3, localIdx4, koops[i], grad);
  }

  const auto& [t_idx1s, t_idx2s, t_idx3s, t_idx4s, V1s, V2s, V3s] = terms.torsionTerms;
  const int torsionStart                                          = systemIndices.torsionTermStarts[molIdx];
  const int torsionEnd                                            = systemIndices.torsionTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = torsionStart + tid; i < torsionEnd; i += stride) {
    const int localIdx1 = t_idx1s[i] - atomStart;
    const int localIdx2 = t_idx2s[i] - atomStart;
    const int localIdx3 = t_idx3s[i] - atomStart;
    const int localIdx4 = t_idx4s[i] - atomStart;
    rdkit_ports::torsionGrad(molCoords, localIdx1, localIdx2, localIdx3, localIdx4, V1s[i], V2s[i], V3s[i], grad);
  }

  const auto& [v_idx1s, v_idx2s, R_ij_stars, wellDepths] = terms.vdwTerms;
  const int vdwStart                                     = systemIndices.vdwTermStarts[molIdx];
  const int vdwEnd                                       = systemIndices.vdwTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = vdwStart + tid; i < vdwEnd; i += stride) {
    const int localIdx1 = v_idx1s[i] - atomStart;
    const int localIdx2 = v_idx2s[i] - atomStart;
    rdkit_ports::vDWGrad(molCoords, localIdx1, localIdx2, R_ij_stars[i], wellDepths[i], grad);
  }

  const auto& [e_idx1s, e_idx2s, chargeTerms, dielModels, is1_4s] = terms.eleTerms;
  const int eleStart                                              = systemIndices.eleTermStarts[molIdx];
  const int eleEnd                                                = systemIndices.eleTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = eleStart + tid; i < eleEnd; i += stride) {
    const int  localIdx1 = e_idx1s[i] - atomStart;
    const int  localIdx2 = e_idx2s[i] - atomStart;
    const bool is14      = is1_4s[i] > 0;
    eleGrad(molCoords, localIdx1, localIdx2, chargeTerms[i], dielModels[i], is14, grad);
  }

  if constexpr (HasConstraints) {
    const auto& [dc_idx1s, dc_idx2s, minLens, maxLens, dcForceConstants] = terms.distanceConstraintTerms;
    const int dcStart = systemIndices.distanceConstraintTermStarts[molIdx];
    const int dcEnd   = systemIndices.distanceConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = dcStart + tid; i < dcEnd; i += stride) {
      const int localIdx1 = dc_idx1s[i] - atomStart;
      const int localIdx2 = dc_idx2s[i] - atomStart;
      distanceConstraintGrad(molCoords, localIdx1, localIdx2, minLens[i], maxLens[i], dcForceConstants[i], grad);
    }

    const auto& [pc_idxs, refXs, refYs, refZs, maxDispls, pcForceConstants] = terms.positionConstraintTerms;
    const int pcStart = systemIndices.positionConstraintTermStarts[molIdx];
    const int pcEnd   = systemIndices.positionConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = pcStart + tid; i < pcEnd; i += stride) {
      const int localIdx = pc_idxs[i] - atomStart;
      positionConstraintGrad(molCoords,
                             localIdx,
                             refXs[i],
                             refYs[i],
                             refZs[i],
                             maxDispls[i],
                             pcForceConstants[i],
                             grad);
    }

    const auto& [ac_idx1s, ac_idx2s, ac_idx3s, minAngleDegs, maxAngleDegs, acForceConstants] =
      terms.angleConstraintTerms;
    const int acStart = systemIndices.angleConstraintTermStarts[molIdx];
    const int acEnd   = systemIndices.angleConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = acStart + tid; i < acEnd; i += stride) {
      const int localIdx1 = ac_idx1s[i] - atomStart;
      const int localIdx2 = ac_idx2s[i] - atomStart;
      const int localIdx3 = ac_idx3s[i] - atomStart;
      angleConstraintGrad(molCoords,
                          localIdx1,
                          localIdx2,
                          localIdx3,
                          minAngleDegs[i],
                          maxAngleDegs[i],
                          acForceConstants[i],
                          grad);
    }

    const auto& [tc_idx1s, tc_idx2s, tc_idx3s, tc_idx4s, minDihedralDegs, maxDihedralDegs, tcForceConstants] =
      terms.torsionConstraintTerms;
    const int tcStart = systemIndices.torsionConstraintTermStarts[molIdx];
    const int tcEnd   = systemIndices.torsionConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = tcStart + tid; i < tcEnd; i += stride) {
      const int localIdx1 = tc_idx1s[i] - atomStart;
      const int localIdx2 = tc_idx2s[i] - atomStart;
      const int localIdx3 = tc_idx3s[i] - atomStart;
      const int localIdx4 = tc_idx4s[i] - atomStart;
      torsionConstraintGrad(molCoords,
                            localIdx1,
                            localIdx2,
                            localIdx3,
                            localIdx4,
                            minDihedralDegs[i],
                            maxDihedralDegs[i],
                            tcForceConstants[i],
                            grad);
    }
  }
}

// -----------------------------------------------------------------------------------------------------------------
// Block-cooperative evaluation used by the block-per-molecule energy and gradient kernels.
//
// Each term range is cut into warp-sized chunks that are dealt round-robin to the block's warps. Energy keeps lanes
// contiguous for coalesced term loads; the gradient path may transpose bonded chunks to reduce atomic contention.
// -----------------------------------------------------------------------------------------------------------------

constexpr int      kWarpSize     = 32;
constexpr unsigned kFullWarpMask = 0xffffffffu;

//! Calls body(offset, valid) for this warp's chunks of one term type holding count terms. Chunks are numbered
//! continuously across term types (chunkBase counts the chunks of the preceding types, and is advanced past this type)
//! and dealt round-robin to the warps of the block. With Transposed, lane l of chunk c takes term c + l * numChunks
//! instead of 32 * c + l, which spreads runs of consecutive terms sharing an atom over different chunks and so lowers
//! atomic contention. All lanes of a warp run the same iterations.
template <int NumWarps, bool Transposed, typename Body>
static __device__ __forceinline__ void forEachTermChunk(int& chunkBase, const int count, Body&& body) {
  const int numChunks = (count + kWarpSize - 1) / kWarpSize;
  const int warp      = threadIdx.x / kWarpSize;
  const int lane      = threadIdx.x % kWarpSize;
#pragma unroll 1
  for (int c = (warp - chunkBase % NumWarps + NumWarps) % NumWarps; c < numChunks; c += NumWarps) {
    const int offset = Transposed ? c + lane * numChunks : c * kWarpSize + lane;
    body(offset, offset < count);
  }
  chunkBase += numChunks;
}

//! forEachTermChunk with contiguous lanes, split into load(offset) -> data and compute(data, valid) so the next
//! chunk's term data is fetched while the current chunk is evaluated. Offsets passed to load are clamped into range.
template <int NumWarps, typename Load, typename Compute>
static __device__ __forceinline__ void forEachTermChunkPrefetched(int&      chunkBase,
                                                                  const int count,
                                                                  Load&&    load,
                                                                  Compute&& compute) {
  const int numChunks = (count + kWarpSize - 1) / kWarpSize;
  const int warp      = threadIdx.x / kWarpSize;
  const int lane      = threadIdx.x % kWarpSize;
  int       c         = (warp - chunkBase % NumWarps + NumWarps) % NumWarps;
  chunkBase += numChunks;
  if (c >= numChunks) {
    return;
  }
  auto data = load(min(c * kWarpSize + lane, count - 1));
#pragma unroll 1
  while (true) {
    const int  next     = c + NumWarps;
    const bool hasNext  = next < numChunks;
    auto       nextData = data;
    if (hasNext) {
      nextData = load(min(next * kWarpSize + lane, count - 1));
    }
    compute(data, c * kWarpSize + lane < count);
    if (!hasNext) {
      break;
    }
    data = nextData;
    c    = next;
  }
}

//! forEachTermChunkPrefetched unrolled into two halves that alternate the roles of the two data buffers, so the
//! prefetched data is never copied between registers. Used by the energy pair loops; the gradient pair loops keep the
//! rolled form, whose smaller body schedules better around their shuffles and atomics.
template <int NumWarps, typename Load, typename Compute>
static __device__ __forceinline__ void forEachTermChunkPrefetchedPingPong(int&      chunkBase,
                                                                          const int count,
                                                                          Load&&    load,
                                                                          Compute&& compute) {
  const int numChunks = (count + kWarpSize - 1) / kWarpSize;
  const int warp      = threadIdx.x / kWarpSize;
  const int lane      = threadIdx.x % kWarpSize;
  int       c         = (warp - chunkBase % NumWarps + NumWarps) % NumWarps;
  chunkBase += numChunks;
  if (c >= numChunks) {
    return;
  }
  auto dataA = load(min(c * kWarpSize + lane, count - 1));
  auto dataB = dataA;
#pragma unroll 1
  while (true) {
    // Offsets are clamped into the range, so the loads past the last chunk are safe and need no branch.
    int next = c + NumWarps;
    dataB    = load(min(next * kWarpSize + lane, count - 1));
    compute(dataA, c * kWarpSize + lane < count);
    if (next >= numChunks) {
      break;
    }
    c     = next;
    next  = c + NumWarps;
    dataA = load(min(next * kWarpSize + lane, count - 1));
    compute(dataB, c * kWarpSize + lane < count);
    if (next >= numChunks) {
      break;
    }
    c = next;
  }
}

//! Inclusive sums of (x, y, z) over runs of consecutive lanes holding the same key. Must be called by all 32 lanes.
//! Returns true on the last lane of each run, which then holds the total of its run.
template <bool EarlyExit>
static __device__ __forceinline__ bool warpSegmentedSum3(const int key, real& x, real& y, real& z) {
  const int      lane     = threadIdx.x % kWarpSize;
  const int      prevKey  = __shfl_up_sync(kFullWarpMask, key, 1);
  const unsigned heads    = __ballot_sync(kFullWarpMask, lane == 0 || prevKey != key);
  // First lane of this lane's run: the highest run head at or below this lane.
  const int      runStart = kWarpSize - 1 - __clz(heads & (kFullWarpMask >> (kWarpSize - 1 - lane)));
#pragma unroll
  for (int delta = 1; delta < kWarpSize; delta <<= 1) {
    if constexpr (EarlyExit) {
      // Stop once no lane's run reaches back delta lanes.
      if (!__any_sync(kFullWarpMask, lane - delta >= runStart)) {
        break;
      }
    }
    const real ox = __shfl_up_sync(kFullWarpMask, x, delta);
    const real oy = __shfl_up_sync(kFullWarpMask, y, delta);
    const real oz = __shfl_up_sync(kFullWarpMask, z, delta);
    // Adds the shuffled partial sums only within the run, through a 0/1 weight instead of selects on every word.
    const real w  = lane - delta >= runStart ? real{1.0} : real{0.0};
    x             = fma(w, ox, x);
    y             = fma(w, oy, y);
    z             = fma(w, oz, z);
  }
  return lane == kWarpSize - 1 || ((heads >> lane) & 2u) != 0;
}

//! Adds a pair gradient (+g on idx1, -g on idx2) for pair terms whose idx1 is sorted within the term list. Lanes
//! sharing idx1 are combined first so that only one lane per atom updates idx1. Must be called by all 32 lanes.
template <bool EarlyExit>
static __device__ __forceinline__ void accumulateSortedKeyGrad(const bool  valid,
                                                               const int   idx1,
                                                               real        gx,
                                                               real        gy,
                                                               real        gz,
                                                               const auto& sink) {
  const int      lane       = threadIdx.x % kWarpSize;
  const int      key        = valid ? idx1 : -1 - lane;
  const int      prevKey    = __shfl_up_sync(kFullWarpMask, key, 1);
  const unsigned heads      = __ballot_sync(kFullWarpMask, lane == 0 || prevKey != key);
  const unsigned validMask  = __ballot_sync(kFullWarpMask, valid);
  const unsigned laterHeads = heads & ~1u;
  if (validMask == kFullWarpMask && (laterHeads & (laterHeads - 1u)) == 0) {
    // A full chunk with at most two runs, lanes [0, split) and [split, 32), as in most chunks of the long sorted vdW
    // runs. Both runs' (x, y, z) totals are reduced at once by a butterfly that halves the values each lane carries at
    // every step (8 shuffled values instead of the 15 of the segmented scan), after which lane l holds component
    // (l >> 2) & 3 (3 is padding) of run (l >> 4) & 1 and one reduction per (run, component) lane adds it.
    const int  split   = laterHeads != 0u ? __ffs(laterHeads) - 1 : kWarpSize;
    const bool upper16 = (lane & 16) != 0;
    // Lanes l and l ^ 16: the lower half keeps run A's values and the upper half run B's.
    const real keep    = upper16 == (lane >= split) ? real{1.0} : real{0.0};
    real       kx      = keep * gx;
    real       ky      = keep * gy;
    real       kz      = keep * gz;
    kx += __shfl_xor_sync(kFullWarpMask, gx - kx, 16);
    ky += __shfl_xor_sync(kFullWarpMask, gy - ky, 16);
    kz += __shfl_xor_sync(kFullWarpMask, gz - kz, 16);
    // Lanes l and l ^ 8: the lower half keeps (x, y) and the upper half (z, padding).
    const bool upper8 = (lane & 8) != 0;
    real       k0     = upper8 ? kz : kx;
    real       k1     = upper8 ? real{0.0} : ky;
    k0 += __shfl_xor_sync(kFullWarpMask, upper8 ? kx : kz, 8);
    k1 += __shfl_xor_sync(kFullWarpMask, upper8 ? ky : real{0.0}, 8);
    // Lanes l and l ^ 4: one value per lane from here on.
    const bool upper4 = (lane & 4) != 0;
    real       v      = upper4 ? k1 : k0;
    v += __shfl_xor_sync(kFullWarpMask, upper4 ? k0 : k1, 4);
    v += __shfl_xor_sync(kFullWarpMask, v, 2);
    v += __shfl_xor_sync(kFullWarpMask, v, 1);
    const int  atom   = __shfl_sync(kFullWarpMask, idx1, upper16 ? (split & (kWarpSize - 1)) : 0);
    const bool hasRun = !upper16 || split < kWarpSize;
    if constexpr (requires { sink.component(0, 0, v); }) {
      const int c = (lane >> 2) & 3;
      if ((lane & 3) == 0 && c < 3 && hasRun) {
        sink.component(atom, c, v);
      }
    } else {
      const int  runLane = lane & 16;
      const real x       = __shfl_sync(kFullWarpMask, v, runLane);
      const real y       = __shfl_sync(kFullWarpMask, v, runLane + 4);
      const real z       = __shfl_sync(kFullWarpMask, v, runLane + 8);
      if ((lane & 15) == 0 && hasRun) {
        sink(atom, x, y, z);
      }
    }
    return;
  }
  if (warpSegmentedSum3<EarlyExit>(key, gx, gy, gz) && valid) {
    sink(idx1, gx, gy, gz);
  }
}

template <bool EarlyExit>
static __device__ __forceinline__ void accumulateSortedPairGrad(const bool  valid,
                                                                const int   idx1,
                                                                const int   idx2,
                                                                real        gx,
                                                                real        gy,
                                                                real        gz,
                                                                const auto& sink) {
  if (valid) {
    sink(idx2, -gx, -gy, -gz);
  }
  accumulateSortedKeyGrad<EarlyExit>(valid, idx1, gx, gy, gz, sink);
}

//! Raw per-term inputs of the pair terms, loaded one chunk ahead of their evaluation. Nothing may consume the loaded
//! values until the chunk is evaluated, otherwise the warp would wait for the loads right away.
struct VdwTermData {
  int  idx1;
  int  idx2;
  real R_ij_star;
  real wellDepth;
};

struct EleTermData {
  int          idx1;
  int          idx2;
  real         chargeTerm;
  std::uint8_t dielModel;
  std::uint8_t is1_4;
};

template <typename Terms> static __device__ __forceinline__ VdwTermData loadVdwTerm(const Terms& terms, const int i) {
  const auto& t = terms.vdwTerms;
  return VdwTermData{t.idx1[i], t.idx2[i], t.R_ij_star[i], t.wellDepth[i]};
}

template <typename Terms> static __device__ __forceinline__ EleTermData loadEleTerm(const Terms& terms, const int i) {
  const auto& t = terms.eleTerms;
  return EleTermData{t.idx1[i], t.idx2[i], t.chargeTerm[i], t.dielModel[i], t.is1_4[i]};
}

//! Per-thread share of one molecule's energy, evaluated cooperatively by a block of BlockSize threads.
template <int BlockSize, bool HasConstraints, typename Terms>
static __device__ __forceinline__ real molEnergyBlock(const Terms& terms, const int* ranges, const auto molCoords) {
  constexpr int NumWarps  = BlockSize / kWarpSize;
  const int     atomStart = ranges[0];
  real          energy    = real{0.0};
  int           chunkBase = 0;

  {
    const auto& t     = terms.bondTerms;
    const int   start = ranges[2];
    const int   count = ranges[3] - start;
    forEachTermChunk<NumWarps, false>(chunkBase, count, [&](const int k, const bool valid) {
      if (valid) {
        const int i = start + k;
        energy += bondStretchEnergy(molCoords, t.idx1[i] - atomStart, t.idx2[i] - atomStart, t.r0[i], t.kb[i]);
      }
    });
  }
  {
    const auto& t     = terms.angleTerms;
    const int   start = ranges[4];
    const int   count = ranges[5] - start;
    forEachTermChunk<NumWarps, false>(chunkBase, count, [&](const int k, const bool valid) {
      if (valid) {
        const int i = start + k;
        energy += angleBendEnergy(molCoords,
                                  t.idx1[i] - atomStart,
                                  t.idx2[i] - atomStart,
                                  t.idx3[i] - atomStart,
                                  t.theta0[i],
                                  t.ka[i],
                                  static_cast<bool>(t.isLinear[i]));
      }
    });
  }
  {
    const auto& t     = terms.bendTerms;
    const int   start = ranges[6];
    const int   count = ranges[7] - start;
    forEachTermChunk<NumWarps, false>(chunkBase, count, [&](const int k, const bool valid) {
      if (valid) {
        const int i = start + k;
        energy += bendStretchEnergy(molCoords,
                                    t.idx1[i] - atomStart,
                                    t.idx2[i] - atomStart,
                                    t.idx3[i] - atomStart,
                                    t.theta0[i],
                                    t.restLen1[i],
                                    t.restLen2[i],
                                    t.forceConst1[i],
                                    t.forceConst2[i]);
      }
    });
  }
  {
    const auto& t     = terms.oopTerms;
    const int   start = ranges[8];
    const int   count = ranges[9] - start;
    forEachTermChunk<NumWarps, false>(chunkBase, count, [&](const int k, const bool valid) {
      if (valid) {
        const int i = start + k;
        energy += oopBendEnergy(molCoords,
                                t.idx1[i] - atomStart,
                                t.idx2[i] - atomStart,
                                t.idx3[i] - atomStart,
                                t.idx4[i] - atomStart,
                                t.koop[i]);
      }
    });
  }
  {
    const auto& t     = terms.torsionTerms;
    const int   start = ranges[10];
    const int   count = ranges[11] - start;
    forEachTermChunk<NumWarps, false>(chunkBase, count, [&](const int k, const bool valid) {
      if (valid) {
        const int i = start + k;
        energy += torsionEnergy(molCoords,
                                t.idx1[i] - atomStart,
                                t.idx2[i] - atomStart,
                                t.idx3[i] - atomStart,
                                t.idx4[i] - atomStart,
                                t.V1[i],
                                t.V2[i],
                                t.V3[i]);
      }
    });
  }
  {
    const int start = ranges[12];
    const int count = ranges[13] - start;
    forEachTermChunkPrefetchedPingPong<NumWarps>(
      chunkBase,
      count,
      [&](const int k) { return loadVdwTerm(terms, start + k); },
      [&](const VdwTermData& d, const bool valid) {
        // Lanes past the range evaluate the (clamped) last term and discard it, without a branch.
        const real e = vdwEnergy(molCoords, d.idx1 - atomStart, d.idx2 - atomStart, d.R_ij_star, d.wellDepth);
        energy += valid ? e : real{0.0};
      });
  }
  {
    const int start = ranges[14];
    const int count = ranges[15] - start;
    forEachTermChunkPrefetchedPingPong<NumWarps>(
      chunkBase,
      count,
      [&](const int k) { return loadEleTerm(terms, start + k); },
      [&](const EleTermData& d, const bool valid) {
        const real e = eleEnergy(molCoords,
                                 d.idx1 - atomStart,
                                 d.idx2 - atomStart,
                                 d.chargeTerm,
                                 static_cast<int>(d.dielModel),
                                 d.is1_4 > 0);
        energy += valid ? e : real{0.0};
      });
  }

  if constexpr (HasConstraints) {
    {
      const auto& t     = terms.distanceConstraintTerms;
      const int   start = ranges[16];
      const int   count = ranges[17] - start;
      forEachTermChunk<NumWarps, false>(chunkBase, count, [&](const int k, const bool valid) {
        if (valid) {
          const int i = start + k;
          energy += distanceConstraintEnergy(molCoords,
                                             t.idx1[i] - atomStart,
                                             t.idx2[i] - atomStart,
                                             t.minLen[i],
                                             t.maxLen[i],
                                             t.forceConstant[i]);
        }
      });
    }
    {
      const auto& t     = terms.positionConstraintTerms;
      const int   start = ranges[18];
      const int   count = ranges[19] - start;
      forEachTermChunk<NumWarps, false>(chunkBase, count, [&](const int k, const bool valid) {
        if (valid) {
          const int i = start + k;
          energy += positionConstraintEnergy(molCoords,
                                             t.idx[i] - atomStart,
                                             t.refX[i],
                                             t.refY[i],
                                             t.refZ[i],
                                             t.maxDispl[i],
                                             t.forceConstant[i]);
        }
      });
    }
    {
      const auto& t     = terms.angleConstraintTerms;
      const int   start = ranges[20];
      const int   count = ranges[21] - start;
      forEachTermChunk<NumWarps, false>(chunkBase, count, [&](const int k, const bool valid) {
        if (valid) {
          const int i = start + k;
          energy += angleConstraintEnergy(molCoords,
                                          t.idx1[i] - atomStart,
                                          t.idx2[i] - atomStart,
                                          t.idx3[i] - atomStart,
                                          t.minAngleDeg[i],
                                          t.maxAngleDeg[i],
                                          t.forceConstant[i]);
        }
      });
    }
    {
      const auto& t     = terms.torsionConstraintTerms;
      const int   start = ranges[22];
      const int   count = ranges[23] - start;
      forEachTermChunk<NumWarps, false>(chunkBase, count, [&](const int k, const bool valid) {
        if (valid) {
          const int i = start + k;
          energy += torsionConstraintEnergy(molCoords,
                                            t.idx1[i] - atomStart,
                                            t.idx2[i] - atomStart,
                                            t.idx3[i] - atomStart,
                                            t.idx4[i] - atomStart,
                                            t.minDihedralDeg[i],
                                            t.maxDihedralDeg[i],
                                            t.forceConstant[i]);
        }
      });
    }
  }
  return energy;
}

//! Accumulates one molecule's gradient through sink (which adds into a buffer zeroed by the caller), evaluated
//! cooperatively by a block of BlockSize threads.
template <int BlockSize, bool HasConstraints, typename Terms, typename Sink>
static __device__ __forceinline__ void molGradBlock(const Terms& terms,
                                                    const int*   ranges,
                                                    const auto   molCoords,
                                                    const Sink&  sink) {
  constexpr int NumWarps  = BlockSize / kWarpSize;
  const int     atomStart = ranges[0];
  int           chunkBase = 0;

  {
    const auto& t     = terms.bondTerms;
    const int   start = ranges[2];
    const int   count = ranges[3] - start;
    forEachTermChunk<NumWarps, true>(chunkBase, count, [&](const int k, const bool valid) {
      if (valid) {
        const int i = start + k;
        bondStretchGradTo(molCoords, t.idx1[i] - atomStart, t.idx2[i] - atomStart, t.r0[i], t.kb[i], sink);
      }
    });
  }
  {
    const auto& t     = terms.angleTerms;
    const int   start = ranges[4];
    const int   count = ranges[5] - start;
    forEachTermChunk<NumWarps, true>(chunkBase, count, [&](const int k, const bool valid) {
      if (valid) {
        const int i = start + k;
        angleBendGradTo(t.idx1[i] - atomStart,
                        t.idx2[i] - atomStart,
                        t.idx3[i] - atomStart,
                        t.theta0[i],
                        t.ka[i],
                        static_cast<bool>(t.isLinear[i]),
                        molCoords,
                        sink);
      }
    });
  }
  {
    const auto& t     = terms.bendTerms;
    const int   start = ranges[6];
    const int   count = ranges[7] - start;
    forEachTermChunk<NumWarps, true>(chunkBase, count, [&](const int k, const bool valid) {
      if (valid) {
        const int i = start + k;
        bendStretchGradTo(molCoords,
                          t.idx1[i] - atomStart,
                          t.idx2[i] - atomStart,
                          t.idx3[i] - atomStart,
                          t.theta0[i],
                          t.restLen1[i],
                          t.restLen2[i],
                          t.forceConst1[i],
                          t.forceConst2[i],
                          sink);
      }
    });
  }
  {
    const auto& t     = terms.oopTerms;
    const int   start = ranges[8];
    const int   count = ranges[9] - start;
    forEachTermChunk<NumWarps, true>(chunkBase, count, [&](const int k, const bool valid) {
      if (valid) {
        const int i = start + k;
        rdkit_ports::oopGradTo(molCoords,
                               t.idx1[i] - atomStart,
                               t.idx2[i] - atomStart,
                               t.idx3[i] - atomStart,
                               t.idx4[i] - atomStart,
                               t.koop[i],
                               sink);
      }
    });
  }
  {
    const auto& t     = terms.torsionTerms;
    const int   start = ranges[10];
    const int   count = ranges[11] - start;
    forEachTermChunk<NumWarps, true>(chunkBase, count, [&](const int k, const bool valid) {
      if (valid) {
        const int i = start + k;
        rdkit_ports::torsionGradTo(molCoords,
                                   t.idx1[i] - atomStart,
                                   t.idx2[i] - atomStart,
                                   t.idx3[i] - atomStart,
                                   t.idx4[i] - atomStart,
                                   t.V1[i],
                                   t.V2[i],
                                   t.V3[i],
                                   sink);
      }
    });
  }
  {
    const int start = ranges[12];
    const int count = ranges[13] - start;
    forEachTermChunkPrefetched<NumWarps>(
      chunkBase,
      count,
      [&](const int k) { return loadVdwTerm(terms, start + k); },
      [&](const VdwTermData& d, const bool valid) {
        const int localIdx1 = d.idx1 - atomStart;
        const int localIdx2 = d.idx2 - atomStart;
        real      gx = real{0.0}, gy = real{0.0}, gz = real{0.0};
        if (valid) {
          rdkit_ports::vdwGradVec(molCoords, localIdx1, localIdx2, d.R_ij_star, d.wellDepth, gx, gy, gz);
        }
        accumulateSortedPairGrad<false>(valid, localIdx1, localIdx2, gx, gy, gz, sink);
      });
  }
  {
    const int start = ranges[14];
    const int count = ranges[15] - start;
    forEachTermChunkPrefetched<NumWarps>(
      chunkBase,
      count,
      [&](const int k) { return loadEleTerm(terms, start + k); },
      [&](const EleTermData& d, const bool valid) {
        const int localIdx1 = d.idx1 - atomStart;
        const int localIdx2 = d.idx2 - atomStart;
        real      gx = real{0.0}, gy = real{0.0}, gz = real{0.0};
        if (valid) {
          eleGradVec(molCoords,
                     localIdx1,
                     localIdx2,
                     d.chargeTerm,
                     static_cast<int>(d.dielModel),
                     d.is1_4 > 0,
                     gx,
                     gy,
                     gz);
        }
        accumulateSortedPairGrad<true>(valid, localIdx1, localIdx2, gx, gy, gz, sink);
      });
  }

  if constexpr (HasConstraints) {
    {
      const auto& t     = terms.distanceConstraintTerms;
      const int   start = ranges[16];
      const int   count = ranges[17] - start;
      forEachTermChunk<NumWarps, true>(chunkBase, count, [&](const int k, const bool valid) {
        if (valid) {
          const int i = start + k;
          distanceConstraintGradTo(molCoords,
                                   t.idx1[i] - atomStart,
                                   t.idx2[i] - atomStart,
                                   t.minLen[i],
                                   t.maxLen[i],
                                   t.forceConstant[i],
                                   sink);
        }
      });
    }
    {
      const auto& t     = terms.positionConstraintTerms;
      const int   start = ranges[18];
      const int   count = ranges[19] - start;
      forEachTermChunk<NumWarps, true>(chunkBase, count, [&](const int k, const bool valid) {
        if (valid) {
          const int i = start + k;
          positionConstraintGradTo(molCoords,
                                   t.idx[i] - atomStart,
                                   t.refX[i],
                                   t.refY[i],
                                   t.refZ[i],
                                   t.maxDispl[i],
                                   t.forceConstant[i],
                                   sink);
        }
      });
    }
    {
      const auto& t     = terms.angleConstraintTerms;
      const int   start = ranges[20];
      const int   count = ranges[21] - start;
      forEachTermChunk<NumWarps, true>(chunkBase, count, [&](const int k, const bool valid) {
        if (valid) {
          const int i = start + k;
          angleConstraintGradTo(molCoords,
                                t.idx1[i] - atomStart,
                                t.idx2[i] - atomStart,
                                t.idx3[i] - atomStart,
                                t.minAngleDeg[i],
                                t.maxAngleDeg[i],
                                t.forceConstant[i],
                                sink);
        }
      });
    }
    {
      const auto& t     = terms.torsionConstraintTerms;
      const int   start = ranges[22];
      const int   count = ranges[23] - start;
      forEachTermChunk<NumWarps, true>(chunkBase, count, [&](const int k, const bool valid) {
        if (valid) {
          const int i = start + k;
          torsionConstraintGradTo(molCoords,
                                  t.idx1[i] - atomStart,
                                  t.idx2[i] - atomStart,
                                  t.idx3[i] - atomStart,
                                  t.idx4[i] - atomStart,
                                  t.minDihedralDeg[i],
                                  t.maxDihedralDeg[i],
                                  t.forceConstant[i],
                                  sink);
        }
      });
    }
  }
}
