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

#ifndef NVMOLKIT_MMFF_KERNELS_DEVICE_CUH
#define NVMOLKIT_MMFF_KERNELS_DEVICE_CUH

#include "mmff_kernels.h"
#include "kernel_utils.cuh"

// Constants used by MMFF energy/gradient functions
namespace {
constexpr double degreeToRadian = M_PI / 180.0;
constexpr double radianToDegree = 180.0 / M_PI;
}  // namespace

using namespace nvMolKit::FFKernelUtils;

// rdkit_ports namespace functions (header-only implementations)
namespace rdkit_ports {

static __device__ __forceinline__ void oopGrad(const double* pos,
                                        const int     idx1,
                                        const int     idx2,
                                        const int     idx3,
                                        const int     idx4,
                                        const double  koop,
                                        double*       grad) {
  constexpr double prefactor = 143.9325 * degreeToRadian;

  double dJIx = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  double dJIy = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  double dJIz = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];

  double dJKx = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  double dJKy = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  double dJKz = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];

  double dJLx = pos[3 * idx4 + 0] - pos[3 * idx2 + 0];
  double dJLy = pos[3 * idx4 + 1] - pos[3 * idx2 + 1];
  double dJLz = pos[3 * idx4 + 2] - pos[3 * idx2 + 2];

  const double invdJI = rsqrtf(dJIx * dJIx + dJIy * dJIy + dJIz * dJIz);
  const double invdJK = rsqrtf(dJKx * dJKx + dJKy * dJKy + dJKz * dJKz);
  const double invdJL = rsqrtf(dJLx * dJLx + dJLy * dJLy + dJLz * dJLz);

  dJIx *= invdJI;
  dJIy *= invdJI;
  dJIz *= invdJI;
  dJKx *= invdJK;
  dJKy *= invdJK;
  dJKz *= invdJK;
  dJLx *= invdJL;
  dJLy *= invdJL;
  dJLz *= invdJL;

  double normalJIKx, normalJIKy, normalJIKz;
  crossProduct(-dJIx, -dJIy, -dJIz, dJKx, dJKy, dJKz, normalJIKx, normalJIKy, normalJIKz);
  const double invNormLength = rsqrtf(normalJIKx * normalJIKx + normalJIKy * normalJIKy + normalJIKz * normalJIKz);
  normalJIKx *= invNormLength;
  normalJIKy *= invNormLength;
  normalJIKz *= invNormLength;

  const double sinChi   = clamp(dotProduct(dJLx, dJLy, dJLz, normalJIKx, normalJIKy, normalJIKz), -1.0, 1.0);
  const double cosChiSq = 1.0 - sinChi * sinChi;
  const double invCosChi = cosChiSq > 0 ? rsqrtf(cosChiSq) : 1.0e8;
  const double chi      = radianToDegree * asin(sinChi);
  const double cosTheta = clamp(dotProduct(dJIx, dJIy, dJIz, dJKx, dJKy, dJKz), -1.0, 1.0);

  double invSinTheta = rsqrtf(fmax(1.0 - cosTheta * cosTheta, 1.0e-8));

  double dE_dChi = prefactor * koop * chi;
  double t1x, t1y, t1z, t2x, t2y, t2z, t3x, t3y, t3z;
  crossProduct(dJLx, dJLy, dJLz, dJKx, dJKy, dJKz, t1x, t1y, t1z);
  crossProduct(dJIx, dJIy, dJIz, dJLx, dJLy, dJLz, t2x, t2y, t2z);
  crossProduct(dJKx, dJKy, dJKz, dJIx, dJIy, dJIz, t3x, t3y, t3z);

  double term1  = invCosChi * invSinTheta;
  double term2  = sinChi * invCosChi *  (invSinTheta * invSinTheta);
  double tg1[3] = {(t1x * term1 - (dJIx - dJKx * cosTheta) * term2) * invdJI,
                   (t1y * term1 - (dJIy - dJKy * cosTheta) * term2) * invdJI,
                   (t1z * term1 - (dJIz - dJKz * cosTheta) * term2) * invdJI};
  double tg3[3] = {(t2x * term1 - (dJKx - dJIx * cosTheta) * term2) * invdJK,
                   (t2y * term1 - (dJKy - dJIy * cosTheta) * term2) * invdJK,
                   (t2z * term1 - (dJKz - dJIz * cosTheta) * term2) * invdJK};
  double tg4[3] = {(t3x * term1 - dJLx * sinChi * invCosChi) * invdJL,
                   (t3y * term1 - dJLy * sinChi * invCosChi) * invdJL,
                   (t3z * term1 - dJLz * sinChi * invCosChi) * invdJL};

  atomicAdd(&grad[3 * idx1 + 0], dE_dChi * tg1[0]);
  atomicAdd(&grad[3 * idx1 + 1], dE_dChi * tg1[1]);
  atomicAdd(&grad[3 * idx1 + 2], dE_dChi * tg1[2]);
  atomicAdd(&grad[3 * idx2 + 0], -dE_dChi * (tg1[0] + tg3[0] + tg4[0]));
  atomicAdd(&grad[3 * idx2 + 1], -dE_dChi * (tg1[1] + tg3[1] + tg4[1]));
  atomicAdd(&grad[3 * idx2 + 2], -dE_dChi * (tg1[2] + tg3[2] + tg4[2]));
  atomicAdd(&grad[3 * idx3 + 0], dE_dChi * tg3[0]);
  atomicAdd(&grad[3 * idx3 + 1], dE_dChi * tg3[1]);
  atomicAdd(&grad[3 * idx3 + 2], dE_dChi * tg3[2]);
  atomicAdd(&grad[3 * idx4 + 0], dE_dChi * tg4[0]);
  atomicAdd(&grad[3 * idx4 + 1], dE_dChi * tg4[1]);
  atomicAdd(&grad[3 * idx4 + 2], dE_dChi * tg4[2]);
}

static __device__ __forceinline__ void torsionGrad(const double* pos,
                                            const int     idx1,
                                            const int     idx2,
                                            const int     idx3,
                                            const int     idx4,
                                            const double  V1,
                                            const double  V2,
                                            const double  V3,
                                            double*       grad) {
  double dx1, dy1, dz1, dx2, dy2, dz2, dx3, dy3, dz3, dx4, dy4, dz4;

  // P1 - P2
  dx1 = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  dy1 = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  dz1 = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];

  // P3 - P2
  dx2 = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  dy2 = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  dz2 = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];

  // P2 - P3
  dx3 = -dx2;
  dy3 = -dy2;
  dz3 = -dz2;

  // P4 - P3
  dx4 = pos[3 * idx4 + 0] - pos[3 * idx3 + 0];
  dy4 = pos[3 * idx4 + 1] - pos[3 * idx3 + 1];
  dz4 = pos[3 * idx4 + 2] - pos[3 * idx3 + 2];

  double cross1x, cross1y, cross1z, cross2x, cross2y, cross2z;
  crossProduct(dx1, dy1, dz1, dx2, dy2, dz2, cross1x, cross1y, cross1z);
  const double norm1 = fmax(sqrt(cross1x * cross1x + cross1y * cross1y + cross1z * cross1z), 1.0e-5);
  cross1x /= norm1;
  cross1y /= norm1;
  cross1z /= norm1;

  crossProduct(dx3, dy3, dz3, dx4, dy4, dz4, cross2x, cross2y, cross2z);
  const double norm2 = fmax(sqrt(cross2x * cross2x + cross2y * cross2y + cross2z * cross2z), 1.0e-5);
  cross2x /= norm2;
  cross2y /= norm2;
  cross2z /= norm2;

  const double dot    = dotProduct(cross1x, cross1y, cross1z, cross2x, cross2y, cross2z);
  const double cosPhi = clamp(dot, -1.0, 1.0);

  double cross3x, cross3y, cross3z;
  crossProduct(cross1x, cross1y, cross1z, dx2, dy2, dz2, cross3x, cross3y, cross3z);

  const double sinPhiSq = 1.0 - cosPhi * cosPhi;
  const double sinPhi   = ((sinPhiSq > 0.0) ? sqrt(sinPhiSq) : 0.0);
  const double sin2Phi  = 2.0 * sinPhi * cosPhi;
  const double sin3Phi  = 3.0 * sinPhi - 4.0 * sinPhi * sinPhiSq;
  const double dE_dPhi  = 0.5 * (-V1 * sinPhi + 2.0 * V2 * sin2Phi - 3.0 * V3 * sin3Phi);
  const double sinTerm  = -dE_dPhi * (isDoubleZero(sinPhi) ? (1.0 / cosPhi) : (1.0 / sinPhi));

  double dCos_dT[6] = {1.0 / norm1 * (cross2x - cosPhi * cross1x),
                       1.0 / norm1 * (cross2y - cosPhi * cross1y),
                       1.0 / norm1 * (cross2z - cosPhi * cross1z),
                       1.0 / norm2 * (cross1x - cosPhi * cross2x),
                       1.0 / norm2 * (cross1y - cosPhi * cross2y),
                       1.0 / norm2 * (cross1z - cosPhi * cross2z)};

  atomicAdd(&grad[3 * idx1 + 0], sinTerm * (dCos_dT[2] * dy2 - dCos_dT[1] * dz2));
  atomicAdd(&grad[3 * idx1 + 1], sinTerm * (dCos_dT[0] * dz2 - dCos_dT[2] * dx2));
  atomicAdd(&grad[3 * idx1 + 2], sinTerm * (dCos_dT[1] * dx2 - dCos_dT[0] * dy2));

  atomicAdd(&grad[3 * idx2 + 0],
            sinTerm * (dCos_dT[1] * (dz2 - dz1) + dCos_dT[2] * (dy1 - dy2) + dCos_dT[4] * (-dz4) + dCos_dT[5] * (dy4)));
  atomicAdd(&grad[3 * idx2 + 1],
            sinTerm * (dCos_dT[0] * (dz1 - dz2) + dCos_dT[2] * (dx2 - dx1) + dCos_dT[3] * (dz4) + dCos_dT[5] * (-dx4)));
  atomicAdd(&grad[3 * idx2 + 2],
            sinTerm * (dCos_dT[0] * (dy2 - dy1) + dCos_dT[1] * (dx1 - dx2) + dCos_dT[3] * (-dy4) + dCos_dT[4] * (dx4)));

  atomicAdd(&grad[3 * idx3 + 0],
            sinTerm * (dCos_dT[1] * (dz1) + dCos_dT[2] * (-dy1) + dCos_dT[4] * (dz4 - dz3) + dCos_dT[5] * (dy3 - dy4)));
  atomicAdd(&grad[3 * idx3 + 1],
            sinTerm * (dCos_dT[0] * (-dz1) + dCos_dT[2] * (dx1) + dCos_dT[3] * (dz3 - dz4) + dCos_dT[5] * (dx4 - dx3)));
  atomicAdd(&grad[3 * idx3 + 2],
            sinTerm * (dCos_dT[0] * (dy1) + dCos_dT[1] * (-dx1) + dCos_dT[3] * (dy4 - dy3) + dCos_dT[4] * (dx3 - dx4)));

  atomicAdd(&grad[3 * idx4 + 0], sinTerm * (dCos_dT[4] * dz3 - dCos_dT[5] * dy3));
  atomicAdd(&grad[3 * idx4 + 1], sinTerm * (dCos_dT[5] * dx3 - dCos_dT[3] * dz3));
  atomicAdd(&grad[3 * idx4 + 2], sinTerm * (dCos_dT[3] * dy3 - dCos_dT[4] * dx3));
}

static __device__ __forceinline__ void vDWGrad(const double* pos,
                                        const int     idx1,
                                        const int     idx2,
                                        const double  R_ij_star,
                                        const double  wellDepth,
                                        double*       grad) {
  constexpr double vdw1   = 1.07;
  constexpr double vdw1m1 = vdw1 - 1.0;
  constexpr double vdw2   = 1.12;
  constexpr double vdw2m1 = vdw2 - 1.0;
  constexpr double vdw2t7 = vdw2 * 7.0;

  const double distance = sqrt(distanceSquared(pos, idx1, idx2));

  const double q         = distance / R_ij_star;
  const double q2        = q * q;
  const double q6        = q2 * q2 * q2;
  const double q7        = q6 * q;
  const double q7pvdw2m1 = q7 + vdw2m1;
  const double t         = vdw1 / (q + vdw1 - 1.0);
  const double t2        = t * t;
  const double t7        = t2 * t2 * t2 * t;
  const double dE_dr     = wellDepth / R_ij_star * t7 *
                       (-vdw2t7 * q6 / (q7pvdw2m1 * q7pvdw2m1) + ((-vdw2t7 / q7pvdw2m1 + 14.0) / (q + vdw1m1)));

  double term1x, term1y, term1z;
  if (distance <= 0.0) {
    term1x = R_ij_star * 0.01;
    term1y = R_ij_star * 0.01;
    term1z = R_ij_star * 0.01;
  } else {
    term1x = dE_dr * (pos[3 * idx1 + 0] - pos[3 * idx2 + 0]) / distance;
    term1y = dE_dr * (pos[3 * idx1 + 1] - pos[3 * idx2 + 1]) / distance;
    term1z = dE_dr * (pos[3 * idx1 + 2] - pos[3 * idx2 + 2]) / distance;
  }

  atomicAdd(&grad[3 * idx1 + 0], term1x);
  atomicAdd(&grad[3 * idx1 + 1], term1y);
  atomicAdd(&grad[3 * idx1 + 2], term1z);

  atomicAdd(&grad[3 * idx2 + 0], -term1x);
  atomicAdd(&grad[3 * idx2 + 1], -term1y);
  atomicAdd(&grad[3 * idx2 + 2], -term1z);
}

}  // namespace rdkit_ports

// Device helper functions (header-only implementations)
static __device__ __forceinline__ double bondStretchEnergy(const double* pos,
                                                           const int     idx1,
                                                           const int     idx2,
                                                           const double  r0,
                                                           const double  kb) {
  constexpr double prefactor           = 143.9325 / 2.0;
  constexpr double csFactorDist        = -2.0;
  constexpr double csFactorDistSquared = 7.0 / 12.0 * csFactorDist * csFactorDist;

  const double distSquared = distanceSquared(pos, idx1, idx2);
  const double distance    = sqrt(distSquared);

  const double deltaR  = distance - r0;
  const double deltaR2 = deltaR * deltaR;
  return prefactor * kb * deltaR2 * (1.0 + csFactorDist * deltaR + csFactorDistSquared * deltaR2);
}

static __device__ __forceinline__ void bondStretchGrad(const double* pos,
                                                       const int     idx1,
                                                       const int     idx2,
                                                       const double  r0,
                                                       const double  kb,
                                                       double*       grad) {
  constexpr double c1                          = 143.9325;
  constexpr double cs                          = -2.0;
  constexpr double csFactorTimesSecondConstant = cs * 1.5;
  constexpr double lastFactor                  = 2.0 * 7.0 / 12.0 * cs * cs;

  double       dx, dy, dz;
  const double distanceSquared = distanceSquaredWithComponents(pos, idx1, idx2, dx, dy, dz);
  const double distance        = sqrt(distanceSquared);
  const double deltaR          = distance - r0;

  const double de_dr = c1 * kb * deltaR * (1.0 + csFactorTimesSecondConstant * deltaR + lastFactor * deltaR * deltaR);

  const double invDist = 1.0 / distance;
  double       dE_dx, dE_dy, dE_dz;
  if (distance > 0.0) {
    dE_dx = de_dr * dx * invDist;
    dE_dy = de_dr * dy * invDist;
    dE_dz = de_dr * dz * invDist;
  } else {
    dE_dx = kb * 0.01;
    dE_dy = kb * 0.01;
    dE_dz = kb * 0.01;
  }

  atomicAdd(&grad[3 * idx1 + 0], dE_dx);
  atomicAdd(&grad[3 * idx1 + 1], dE_dy);
  atomicAdd(&grad[3 * idx1 + 2], dE_dz);

  atomicAdd(&grad[3 * idx2 + 0], -dE_dx);
  atomicAdd(&grad[3 * idx2 + 1], -dE_dy);
  atomicAdd(&grad[3 * idx2 + 2], -dE_dz);
}

static __device__ __forceinline__ double angleBendEnergy(const double* pos,
                                                         const int     idx1,
                                                         const int     idx2,
                                                         const int     idx3,
                                                         const double  theta0,
                                                         const double  ka,
                                                         const bool    isLinear) {
  constexpr double prefactor = 0.5 * 143.9325 * degreeToRadian * degreeToRadian;
  constexpr double cb        = -0.4 * degreeToRadian;

  float       dx1, dy1, dz1, dx2, dy2, dz2;
  const float dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const float dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const float dist1        = sqrtf(dist1Squared);
  const float dist2        = sqrtf(dist2Squared);

  const float  dot         = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const double cosTheta    = clamp(dot / (dist1 * dist2), -1.0, 1.0);
  const double theta       = radianToDegree * acos(cosTheta);
  const double deltaTheta  = theta - theta0;
  const double deltaTheta2 = deltaTheta * deltaTheta;

  if (isLinear) {
    constexpr double linearPrefactor = 143.9325;
    return linearPrefactor * ka * (1.0 + cosTheta);
  }
  return prefactor * ka * deltaTheta2 * (1.0 + cb * deltaTheta);
}

static __device__ __forceinline__ void angleBendGrad(const int     idx1,
                                                     const int     idx2,
                                                     const int     idx3,
                                                     const double  theta0,
                                                     const double  ka,
                                                     const bool    isLinear,
                                                     const double* pos,
                                                     double*       grad) {
  constexpr double c1       = 143.9325 * degreeToRadian;
  constexpr double cbFactor = -0.006981317 * 1.5;

  double       dx1, dy1, dz1, dx2, dy2, dz2;
  const double dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const double dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const double dist1        = sqrt(dist1Squared);
  const double dist2        = sqrt(dist2Squared);

  const double dot         = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const double cosTheta    = clamp(dot / (dist1 * dist2), -1.0, 1.0);
  const double sinThetaSq  = 1.0 - cosTheta * cosTheta;
  const double negSinTheta = -(fmax(((sinThetaSq > 0.0) ? sqrt(sinThetaSq) : 0.0), 1.0e-8));
  const double theta       = radianToDegree * acos(cosTheta);
  const double deltaTheta  = theta - theta0;

  double de_dDeltaTheta;

  if (isLinear) {
    constexpr double linearPrefactor = 143.9325;
    de_dDeltaTheta                   = -linearPrefactor * ka * sqrt(1.0 - (cosTheta * cosTheta));
  } else {
    de_dDeltaTheta = c1 * ka * deltaTheta * (1.0 + cbFactor * deltaTheta);
  }

  if (isDoubleZero(dist1) || isDoubleZero(dist2)) {
    return;
  }

  const double invDist1 = 1.0 / dist1;
  const double invDist2 = 1.0 / dist2;

  const double dxnorm1 = dx1 * invDist1;
  const double dynorm1 = dy1 * invDist1;
  const double dznorm1 = dz1 * invDist1;
  const double dxnorm2 = dx2 * invDist2;
  const double dynorm2 = dy2 * invDist2;
  const double dznorm2 = dz2 * invDist2;

  const double intermediate1 = invDist1 * (dxnorm2 - cosTheta * dxnorm1);
  const double intermediate2 = invDist1 * (dynorm2 - cosTheta * dynorm1);
  const double intermediate3 = invDist1 * (dznorm2 - cosTheta * dznorm1);
  const double intermediate4 = invDist2 * (dxnorm1 - cosTheta * dxnorm2);
  const double intermediate5 = invDist2 * (dynorm1 - cosTheta * dynorm2);
  const double intermediate6 = invDist2 * (dznorm1 - cosTheta * dznorm2);

  if (isDoubleZero(negSinTheta)) {
    return;
  }
  const double constantFactor = de_dDeltaTheta / negSinTheta;

  atomicAdd(&grad[3 * idx1 + 0], constantFactor * intermediate1);
  atomicAdd(&grad[3 * idx1 + 1], constantFactor * intermediate2);
  atomicAdd(&grad[3 * idx1 + 2], constantFactor * intermediate3);

  atomicAdd(&grad[3 * idx2 + 0], constantFactor * (-intermediate1 - intermediate4));
  atomicAdd(&grad[3 * idx2 + 1], constantFactor * (-intermediate2 - intermediate5));
  atomicAdd(&grad[3 * idx2 + 2], constantFactor * (-intermediate3 - intermediate6));

  atomicAdd(&grad[3 * idx3 + 0], constantFactor * intermediate4);
  atomicAdd(&grad[3 * idx3 + 1], constantFactor * intermediate5);
  atomicAdd(&grad[3 * idx3 + 2], constantFactor * intermediate6);
}

static __device__ __forceinline__ double bendStretchEnergy(const double* pos,
                                                           const int     idx1,
                                                           const int     idx2,
                                                           const int     idx3,
                                                           const double  theta0,
                                                           const double  restLen1,
                                                           const double  restLen2,
                                                           const double  forceConst1,
                                                           const double  forceConst2) {
  constexpr double prefactor = 2.51210;

  float       dx1, dy1, dz1, dx2, dy2, dz2;
  const float dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const float dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const float dist1        = sqrtf(dist1Squared);
  const float dist2        = sqrtf(dist2Squared);

  const float  dot      = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const double cosTheta = clamp(dot / (dist1 * dist2), -1.0, 1.0);
  const double theta    = 180 / M_PI * acos(cosTheta);

  const double deltaTheta = theta - theta0;
  const double deltaR1    = dist1 - restLen1;
  const double deltaR2    = dist2 - restLen2;

  return prefactor * deltaTheta * (deltaR1 * forceConst1 + deltaR2 * forceConst2);
}

static __device__ __forceinline__ void bendStretchGrad(const double* pos,
                                                       const int     idx1,
                                                       const int     idx2,
                                                       const int     idx3,
                                                       const double  theta0,
                                                       const double  restLen1,
                                                       const double  restLen2,
                                                       const double  forceConst1,
                                                       const double  forceConst2,
                                                       double*       grad) {
  constexpr double prefactor = 143.9325 * M_PI / 180.0;

  double       dx1, dy1, dz1, dx2, dy2, dz2;
  const double dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const double dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const double dist1        = sqrt(dist1Squared);
  const double dist2        = sqrt(dist2Squared);

  const double dot      = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const double cosTheta = clamp(dot / (dist1 * dist2), -1.0, 1.0);
  const double sinTheta = fmax(sqrt(1.0 - cosTheta * cosTheta), 1.0e-8);

  const double theta = 180 / M_PI * acos(cosTheta);

  const double deltaTheta = theta - theta0;
  const double deltaR1    = dist1 - restLen1;
  const double deltaR2    = dist2 - restLen2;

  const double bondEnergyTerm = 180.0 / M_PI * (forceConst1 * deltaR1 + forceConst2 * deltaR2);

  const double invDist1 = 1.0 / dist1;
  const double invDist2 = 1.0 / dist2;

  const double scaledDx1 = dx1 * invDist1;
  const double scaledDy1 = dy1 * invDist1;
  const double scaledDz1 = dz1 * invDist1;
  const double scaledDx2 = dx2 * invDist2;
  const double scaledDy2 = dy2 * invDist2;
  const double scaledDz2 = dz2 * invDist2;

  const double intermediate1 = invDist1 * (scaledDx2 - cosTheta * scaledDx1);
  const double intermediate2 = invDist1 * (scaledDy2 - cosTheta * scaledDy1);
  const double intermediate3 = invDist1 * (scaledDz2 - cosTheta * scaledDz1);
  const double intermediate4 = invDist2 * (scaledDx1 - cosTheta * scaledDx2);
  const double intermediate5 = invDist2 * (scaledDy1 - cosTheta * scaledDy2);
  const double intermediate6 = invDist2 * (scaledDz1 - cosTheta * scaledDz2);

  const double gradx1 = prefactor * (deltaTheta * scaledDx1 * forceConst1 - intermediate1 * bondEnergyTerm / sinTheta);
  const double grady1 = prefactor * (deltaTheta * scaledDy1 * forceConst1 - intermediate2 * bondEnergyTerm / sinTheta);
  const double gradz1 = prefactor * (deltaTheta * scaledDz1 * forceConst1 - intermediate3 * bondEnergyTerm / sinTheta);

  const double gradx2 = prefactor * (-deltaTheta * (scaledDx1 * forceConst1 + scaledDx2 * forceConst2) +
                                     (intermediate1 + intermediate4) * bondEnergyTerm / sinTheta);
  const double grady2 = prefactor * (-deltaTheta * (scaledDy1 * forceConst1 + scaledDy2 * forceConst2) +
                                     (intermediate2 + intermediate5) * bondEnergyTerm / sinTheta);
  const double gradz2 = prefactor * (-deltaTheta * (scaledDz1 * forceConst1 + scaledDz2 * forceConst2) +
                                     (intermediate3 + intermediate6) * bondEnergyTerm / sinTheta);

  const double gradx3 = prefactor * (deltaTheta * scaledDx2 * forceConst2 - intermediate4 * bondEnergyTerm / sinTheta);
  const double grady3 = prefactor * (deltaTheta * scaledDy2 * forceConst2 - intermediate5 * bondEnergyTerm / sinTheta);
  const double gradz3 = prefactor * (deltaTheta * scaledDz2 * forceConst2 - intermediate6 * bondEnergyTerm / sinTheta);

  atomicAdd(&grad[3 * idx1 + 0], gradx1);
  atomicAdd(&grad[3 * idx1 + 1], grady1);
  atomicAdd(&grad[3 * idx1 + 2], gradz1);

  atomicAdd(&grad[3 * idx3 + 0], gradx3);
  atomicAdd(&grad[3 * idx3 + 1], grady3);
  atomicAdd(&grad[3 * idx3 + 2], gradz3);

  atomicAdd(&grad[3 * idx2 + 0], gradx2);
  atomicAdd(&grad[3 * idx2 + 1], grady2);
  atomicAdd(&grad[3 * idx2 + 2], gradz2);
}

static __device__ __forceinline__ double oopBendEnergy(const double* pos,
                                                       const int     idx1,
                                                       const int     idx2,
                                                       const int     idx3,
                                                       const int     idx4,
                                                       const double  koop) {
  constexpr double prefactor = 0.5 * 143.9325 * degreeToRadian * degreeToRadian;

  double       dxji, dyji, dzji, dxjk, dyjk, dzjk, dxjl, dyjl, dzjl;
  const double distSquaredJI = distanceSquaredWithComponents(pos, idx1, idx2, dxji, dyji, dzji);
  const double distSquaredJK = distanceSquaredWithComponents(pos, idx3, idx2, dxjk, dyjk, dzjk);
  const double distSquaredJL = distanceSquaredWithComponents(pos, idx4, idx2, dxjl, dyjl, dzjl);

  const double distJI = sqrt(distSquaredJI);
  const double distJK = sqrt(distSquaredJK);
  const double distJL = sqrt(distSquaredJL);

  const double scaledDxJI = dxji / distJI;
  const double scaledDyJI = dyji / distJI;
  const double scaledDzJI = dzji / distJI;

  const double scaledDxJK = dxjk / distJK;
  const double scaledDyJK = dyjk / distJK;
  const double scaledDzJK = dzjk / distJK;

  const double scaledDxJL = dxjl / distJL;
  const double scaledDyJL = dyjl / distJL;
  const double scaledDzJL = dzjl / distJL;

  double crossX, crossY, crossZ;
  crossProduct(scaledDxJI, scaledDyJI, scaledDzJI, scaledDxJK, scaledDyJK, scaledDzJK, crossX, crossY, crossZ);
  const double distCross = sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ);

  const double scaledCrossX = crossX / distCross;
  const double scaledCrossY = crossY / distCross;
  const double scaledCrossZ = crossZ / distCross;

  const double dotProduct = scaledCrossX * scaledDxJL + scaledCrossY * scaledDyJL + scaledCrossZ * scaledDzJL;
  const double chi        = radianToDegree * asin(clamp(dotProduct, -1.0, 1.0));

  return prefactor * koop * chi * chi;
}

static __device__ __forceinline__ double torsionEnergy(const double* pos,
                                                       const int     idx1,
                                                       const int     idx2,
                                                       const int     idx3,
                                                       const int     idx4,
                                                       const double  V1,
                                                       const double  V2,
                                                       const double  V3) {
  const double dxIJ = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  const double dyIJ = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  const double dzIJ = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];

  const double dxKJ = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  const double dyKJ = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  const double dzKJ = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];

  const double dxLK = pos[3 * idx4 + 0] - pos[3 * idx3 + 0];
  const double dyLK = pos[3 * idx4 + 1] - pos[3 * idx3 + 1];
  const double dzLK = pos[3 * idx4 + 2] - pos[3 * idx3 + 2];

  const double crossIJKJx = dyIJ * dzKJ - dzIJ * dyKJ;
  const double crossIJKJy = dzIJ * dxKJ - dxIJ * dzKJ;
  const double crossIJKJz = dxIJ * dyKJ - dyIJ * dxKJ;

  const double crossJKLKx = -dyKJ * dzLK + dzKJ * dyLK;
  const double crossJKLKy = -dzKJ * dxLK + dxKJ * dzLK;
  const double crossJKLKz = -dxKJ * dyLK + dyKJ * dxLK;

  const double cross1Norm = sqrt(crossIJKJx * crossIJKJx + crossIJKJy * crossIJKJy + crossIJKJz * crossIJKJz);
  const double cross2Norm = sqrt(crossJKLKx * crossJKLKx + crossJKLKy * crossJKLKy + crossJKLKz * crossJKLKz);

  const double dotProduct = crossIJKJx * crossJKLKx + crossIJKJy * crossJKLKy + crossIJKJz * crossJKLKz;
  const double cosPhi     = dotProduct / (cross1Norm * cross2Norm);
  const double phi        = acos(clamp(cosPhi, -1.0, 1.0));

  return 0.5 * (V1 * (1.0 + cosPhi) + V2 * (1.0 - cos(2.0 * phi)) + V3 * (1.0 + cos(3.0 * phi)));
}

static __device__ __forceinline__ double vdwEnergy(const double* pos,
                                                   const int     idx1,
                                                   const int     idx2,
                                                   const double  R_ij_star,
                                                   const double  wellDepth) {
  double R_ij_star2 = R_ij_star * R_ij_star;
  double R_ij_star7 = R_ij_star2 * R_ij_star2 * R_ij_star2 * R_ij_star;

  const double epsilon = wellDepth;

  const double distSquared = distanceSquared(pos, idx1, idx2);
  const double dist        = sqrt(distSquared);
  const double dist7       = distSquared * distSquared * distSquared * dist;

  const double term1        = 1.07 * R_ij_star / (dist + 0.07 * R_ij_star);
  const double term1Squared = term1 * term1;
  const double term1_7th    = term1Squared * term1Squared * term1Squared * term1;

  const double term2Fraction = 1.12 * R_ij_star7 / (dist7 + 0.12 * R_ij_star7);

  return epsilon * term1_7th * (term2Fraction - 2.0);
}

static __device__ __forceinline__ double eleEnergy(const double* pos,
                                                   const int     idx1,
                                                   const int     idx2,
                                                   const double  chargeTerm,
                                                   const int     dielModel,
                                                   const bool    is1_4) {
  constexpr double prefactor         = 332.0716;
  constexpr double bufferingConstant = 0.05;
  const double     distSquared       = distanceSquared(pos, idx1, idx2);
  double           distTerm          = sqrt(distSquared) + bufferingConstant;
  if (dielModel == 2) {
    distTerm *= distTerm;
  }
  double energy = prefactor * chargeTerm / (distTerm);
  if (is1_4) {
    energy *= 0.75;
  }
  return energy;
}

static __device__ __forceinline__ void eleGrad(const double* pos,
                                               const int     idx1,
                                               const int     idx2,
                                               const double  chargeTerm,
                                               const int     dielModel,
                                               const bool    is1_4,
                                               double*       grad) {
  constexpr double prefactor         = 332.0716;
  constexpr double bufferingConstant = 0.05;

  const double distSquared = distanceSquared(pos, idx1, idx2);
  const double distance    = sqrt(distSquared);
  double       distTerm    = distance + bufferingConstant;
  double       numerator   = -prefactor * chargeTerm;

  if (dielModel == 2) {
    distTerm *= distTerm;
    numerator *= 2;
  }

  double dE_dr = numerator / (distTerm * distTerm);
  if (is1_4) {
    dE_dr *= 0.75;
  }

  const double dE_dx = dE_dr * (pos[3 * idx1 + 0] - pos[3 * idx2 + 0]) / distance;
  const double dE_dy = dE_dr * (pos[3 * idx1 + 1] - pos[3 * idx2 + 1]) / distance;
  const double dE_dz = dE_dr * (pos[3 * idx1 + 2] - pos[3 * idx2 + 2]) / distance;

  atomicAdd(&grad[3 * idx1 + 0], dE_dx);
  atomicAdd(&grad[3 * idx1 + 1], dE_dy);
  atomicAdd(&grad[3 * idx1 + 2], dE_dz);

  atomicAdd(&grad[3 * idx2 + 0], -dE_dx);
  atomicAdd(&grad[3 * idx2 + 1], -dE_dy);
  atomicAdd(&grad[3 * idx2 + 2], -dE_dz);
}

namespace nvMolKit {
namespace MMFF {

static __device__ __inline__ double molEnergy(const EnergyForceContribsDevicePtr& terms,
                                               const BatchedIndicesDevicePtr&      systemIndices,
                                               const double*                       coords,
                                               const int                           molIdx,
                                               const int                           tid,
                                               const int                           stride) {
  const int     atomStart = systemIndices.atomStarts[molIdx];
  const double* molCoords = coords + atomStart * 3;

  double energy = 0.0;

  const auto& [idx1s, idx2s, r0s, kbs] = terms.bondTerms;
  const int bondStart                  = systemIndices.bondTermStarts[molIdx];
  const int bondEnd                    = systemIndices.bondTermStarts[molIdx + 1];
  for (int i = bondStart + tid; i < bondEnd; i += stride) {
    const int localIdx1 = idx1s[i] - atomStart;
    const int localIdx2 = idx2s[i] - atomStart;
    energy += bondStretchEnergy(molCoords, localIdx1, localIdx2, r0s[i], kbs[i]);
  }

  const auto& [a_idx1s, a_idx2s, a_idx3s, theta0s, kas, isLinears] = terms.angleTerms;
  const int angleStart                                             = systemIndices.angleTermStarts[molIdx];
  const int angleEnd                                               = systemIndices.angleTermStarts[molIdx + 1];
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
  for (int i = vdwStart + tid; i < vdwEnd; i += stride) {
    const int localIdx1 = v_idx1s[i] - atomStart;
    const int localIdx2 = v_idx2s[i] - atomStart;
    energy += vdwEnergy(molCoords, localIdx1, localIdx2, R_ij_stars[i], wellDepths[i]);
  }

  const auto& [e_idx1s, e_idx2s, chargeTerms, dielModels, is1_4s] = terms.eleTerms;
  const int eleStart                                              = systemIndices.eleTermStarts[molIdx];
  const int eleEnd                                                = systemIndices.eleTermStarts[molIdx + 1];
  for (int i = eleStart + tid; i < eleEnd; i += stride) {
    const int  localIdx1 = e_idx1s[i] - atomStart;
    const int  localIdx2 = e_idx2s[i] - atomStart;
    const int  dielModel = static_cast<int>(dielModels[i]);
    const bool is14      = is1_4s[i] > 0;
    energy += eleEnergy(molCoords, localIdx1, localIdx2, chargeTerms[i], dielModel, is14);
  }

  return energy;
}

static __device__ __inline__ void molGrad(const EnergyForceContribsDevicePtr& terms,
                                          const BatchedIndicesDevicePtr&      systemIndices,
                                          const double*                       coords,
                                          double*                             grad,
                                          const int                           molIdx,
                                          const int                           tid,
                                          const int                           stride) {
  const int     atomStart = systemIndices.atomStarts[molIdx];
  const double* molCoords = coords + atomStart * 3;

  const auto& [idx1s, idx2s, r0s, kbs] = terms.bondTerms;
  const int bondStart                  = systemIndices.bondTermStarts[molIdx];
  const int bondEnd                    = systemIndices.bondTermStarts[molIdx + 1];
  for (int i = bondStart + tid; i < bondEnd; i += stride) {
    const int localIdx1 = idx1s[i] - atomStart;
    const int localIdx2 = idx2s[i] - atomStart;
    bondStretchGrad(molCoords, localIdx1, localIdx2, r0s[i], kbs[i], grad);
  }

  const auto& [a_idx1s, a_idx2s, a_idx3s, theta0s, kas, isLinears] = terms.angleTerms;
  const int angleStart                                             = systemIndices.angleTermStarts[molIdx];
  const int angleEnd                                               = systemIndices.angleTermStarts[molIdx + 1];
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
  for (int i = vdwStart + tid; i < vdwEnd; i += stride) {
    const int localIdx1 = v_idx1s[i] - atomStart;
    const int localIdx2 = v_idx2s[i] - atomStart;
    rdkit_ports::vDWGrad(molCoords, localIdx1, localIdx2, R_ij_stars[i], wellDepths[i], grad);
  }

  const auto& [e_idx1s, e_idx2s, chargeTerms, dielModels, is1_4s] = terms.eleTerms;
  const int eleStart                                              = systemIndices.eleTermStarts[molIdx];
  const int eleEnd                                                = systemIndices.eleTermStarts[molIdx + 1];
  for (int i = eleStart + tid; i < eleEnd; i += stride) {
    const int  localIdx1 = e_idx1s[i] - atomStart;
    const int  localIdx2 = e_idx2s[i] - atomStart;
    const bool is14      = is1_4s[i] > 0;
    eleGrad(molCoords, localIdx1, localIdx2, chargeTerms[i], dielModels[i], is14, grad);
  }
}

}  // namespace MMFF
}  // namespace nvMolKit

#endif  // NVMOLKIT_MMFF_KERNELS_DEVICE_CUH
