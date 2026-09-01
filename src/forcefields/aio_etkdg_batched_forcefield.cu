// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <numbers>

#include "src/forcefields/aio_etkdg_batched_forcefield.h"
#include "src/forcefields/dist_geom_kernels_device.cuh"

namespace nvMolKit {
namespace {

constexpr int kBlockSize = 256;

__device__ __forceinline__ bool systemIsActive(const int systemIdx, const uint8_t* activeSystemMask) {
  return activeSystemMask == nullptr || activeSystemMask[systemIdx] != 0;
}

__device__ __forceinline__ double planarityEnergy(const double* positions,
                                                  const int     idx1,
                                                  const int     idx2,
                                                  const int     idx3,
                                                  const int     idx4,
                                                  const double  forceConstant) {
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;
  const int posIdx4 = idx4 * 4;

  double rJIx = positions[posIdx1] - positions[posIdx2];
  double rJIy = positions[posIdx1 + 1] - positions[posIdx2 + 1];
  double rJIz = positions[posIdx1 + 2] - positions[posIdx2 + 2];
  double rJKx = positions[posIdx3] - positions[posIdx2];
  double rJKy = positions[posIdx3 + 1] - positions[posIdx2 + 1];
  double rJKz = positions[posIdx3 + 2] - positions[posIdx2 + 2];
  double rJLx = positions[posIdx4] - positions[posIdx2];
  double rJLy = positions[posIdx4 + 1] - positions[posIdx2 + 1];
  double rJLz = positions[posIdx4 + 2] - positions[posIdx2 + 2];

  const double dJI2 = rJIx * rJIx + rJIy * rJIy + rJIz * rJIz;
  const double dJK2 = rJKx * rJKx + rJKy * rJKy + rJKz * rJKz;
  const double dJL2 = rJLx * rJLx + rJLy * rJLy + rJLz * rJLz;
  if (isDoubleZero(dJI2) || isDoubleZero(dJK2) || isDoubleZero(dJL2)) {
    return 0.0;
  }

  const double invJI = rsqrt(dJI2);
  const double invJK = rsqrt(dJK2);
  const double invJL = rsqrt(dJL2);
  rJIx *= invJI;
  rJIy *= invJI;
  rJIz *= invJI;
  rJKx *= invJK;
  rJKy *= invJK;
  rJKz *= invJK;
  rJLx *= invJL;
  rJLy *= invJL;
  rJLz *= invJL;

  double nx, ny, nz;
  crossProduct(rJIx, rJIy, rJIz, rJKx, rJKy, rJKz, nx, ny, nz);
  const double normalLength2 = nx * nx + ny * ny + nz * nz;
  if (isDoubleZero(normalLength2)) {
    return 0.0;
  }
  const double invNormalLength = rsqrt(normalLength2);
  nx *= invNormalLength;
  ny *= invNormalLength;
  nz *= invNormalLength;

  const double sinChi = clamp(dotProduct(nx, ny, nz, rJLx, rJLy, rJLz), -1.0, 1.0);
  const double chi    = asin(sinChi) * 180.0 / std::numbers::pi;
  return 0.5 * forceConstant * chi * chi;
}

template <int component>
__device__ __forceinline__ void planarityGradientComponent(const double* positions,
                                                           const int     idx1,
                                                           const int     idx2,
                                                           const int     idx3,
                                                           const int     idx4,
                                                           const double  forceConstant,
                                                           double*       gradient) {
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;
  const int posIdx4 = idx4 * 4;

  double rJIx = positions[posIdx1] - positions[posIdx2];
  double rJIy = positions[posIdx1 + 1] - positions[posIdx2 + 1];
  double rJIz = positions[posIdx1 + 2] - positions[posIdx2 + 2];
  double rJKx = positions[posIdx3] - positions[posIdx2];
  double rJKy = positions[posIdx3 + 1] - positions[posIdx2 + 1];
  double rJKz = positions[posIdx3 + 2] - positions[posIdx2 + 2];
  double rJLx = positions[posIdx4] - positions[posIdx2];
  double rJLy = positions[posIdx4 + 1] - positions[posIdx2 + 1];
  double rJLz = positions[posIdx4 + 2] - positions[posIdx2 + 2];

  const double dJI2 = rJIx * rJIx + rJIy * rJIy + rJIz * rJIz;
  const double dJK2 = rJKx * rJKx + rJKy * rJKy + rJKz * rJKz;
  const double dJL2 = rJLx * rJLx + rJLy * rJLy + rJLz * rJLz;
  if (isDoubleZero(dJI2) || isDoubleZero(dJK2) || isDoubleZero(dJL2)) {
    return;
  }

  const double invJI = rsqrt(dJI2);
  const double invJK = rsqrt(dJK2);
  const double invJL = rsqrt(dJL2);
  rJIx *= invJI;
  rJIy *= invJI;
  rJIz *= invJI;
  rJKx *= invJK;
  rJKy *= invJK;
  rJKz *= invJK;
  rJLx *= invJL;
  rJLy *= invJL;
  rJLz *= invJL;

  double nx, ny, nz;
  crossProduct(-rJIx, -rJIy, -rJIz, rJKx, rJKy, rJKz, nx, ny, nz);
  const double normalLength2 = nx * nx + ny * ny + nz * nz;
  if (isDoubleZero(normalLength2)) {
    return;
  }
  const double invNormalLength = rsqrt(normalLength2);
  nx *= invNormalLength;
  ny *= invNormalLength;
  nz *= invNormalLength;

  const double sinChi    = clamp(dotProduct(rJLx, rJLy, rJLz, nx, ny, nz), -1.0, 1.0);
  const double cosChiSq  = 1.0 - sinChi * sinChi;
  const double cosChi    = fmax(sqrt(fmax(cosChiSq, 0.0)), 1.0e-8);
  const double chi       = asin(sinChi) * 180.0 / std::numbers::pi;
  const double cosTheta  = clamp(dotProduct(rJIx, rJIy, rJIz, rJKx, rJKy, rJKz), -1.0, 1.0);
  const double sinTheta2 = fmax(1.0 - cosTheta * cosTheta, 1.0e-8);
  const double sinTheta  = fmax(sqrt(sinTheta2), 1.0e-8);
  const double dEdChi    = forceConstant * chi * 180.0 / std::numbers::pi;

  const double inverseTerm1 = 1.0 / (cosChi * sinTheta);
  const double term2        = sinChi / (cosChi * sinTheta2);
  const double term4        = sinChi / cosChi;
  double       tx, ty, tz;
  double       vx, vy, vz;
  double       scale;
  int          targetPosIdx;
  if constexpr (component == 0) {
    crossProduct(rJLx, rJLy, rJLz, rJKx, rJKy, rJKz, tx, ty, tz);
    vx           = rJIx - rJKx * cosTheta;
    vy           = rJIy - rJKy * cosTheta;
    vz           = rJIz - rJKz * cosTheta;
    scale        = invJI;
    targetPosIdx = posIdx1;
  } else if constexpr (component == 1) {
    crossProduct(rJIx, rJIy, rJIz, rJLx, rJLy, rJLz, tx, ty, tz);
    vx           = rJKx - rJIx * cosTheta;
    vy           = rJKy - rJIy * cosTheta;
    vz           = rJKz - rJIz * cosTheta;
    scale        = invJK;
    targetPosIdx = posIdx3;
  } else {
    crossProduct(rJKx, rJKy, rJKz, rJIx, rJIy, rJIz, tx, ty, tz);
    vx           = rJLx;
    vy           = rJLy;
    vz           = rJLz;
    scale        = invJL;
    targetPosIdx = posIdx4;
  }
  const double secondaryTerm = component == 2 ? term4 : term2;
  const double gx            = dEdChi * (tx * inverseTerm1 - vx * secondaryTerm) * scale;
  const double gy            = dEdChi * (ty * inverseTerm1 - vy * secondaryTerm) * scale;
  const double gz            = dEdChi * (tz * inverseTerm1 - vz * secondaryTerm) * scale;
  atomicAdd(&gradient[targetPosIdx], gx);
  atomicAdd(&gradient[targetPosIdx + 1], gy);
  atomicAdd(&gradient[targetPosIdx + 2], gz);
  atomicAdd(&gradient[posIdx2], -gx);
  atomicAdd(&gradient[posIdx2 + 1], -gy);
  atomicAdd(&gradient[posIdx2 + 2], -gz);
}

__global__ void harmonicEnergyKernel(const int      numTerms,
                                     const int*     idx1,
                                     const int*     idx2,
                                     const double*  minLen,
                                     const double*  maxLen,
                                     const double*  forceConstant,
                                     const int*     systemIdx,
                                     const double*  positions,
                                     const uint8_t* activeSystemMask,
                                     double*        energyOuts) {
  const int termIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (termIdx < numTerms && systemIsActive(systemIdx[termIdx], activeSystemMask)) {
    const double energy = DistGeom::distanceConstraintEnergy(positions,
                                                             idx1[termIdx],
                                                             idx2[termIdx],
                                                             minLen[termIdx],
                                                             maxLen[termIdx],
                                                             forceConstant[termIdx]);
    atomicAdd(&energyOuts[systemIdx[termIdx]], energy);
  }
}

__global__ void harmonicGradientKernel(const int      numTerms,
                                       const int*     idx1,
                                       const int*     idx2,
                                       const double*  minLen,
                                       const double*  maxLen,
                                       const double*  forceConstant,
                                       const int*     systemIdx,
                                       const double*  positions,
                                       const uint8_t* activeSystemMask,
                                       double*        gradient) {
  const int termIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (termIdx < numTerms && systemIsActive(systemIdx[termIdx], activeSystemMask)) {
    DistGeom::distanceConstraintGrad(positions,
                                     idx1[termIdx],
                                     idx2[termIdx],
                                     minLen[termIdx],
                                     maxLen[termIdx],
                                     forceConstant[termIdx],
                                     gradient);
  }
}

__global__ void angleEnergyKernel(const int      numTerms,
                                  const int*     idx1,
                                  const int*     idx2,
                                  const int*     idx3,
                                  const double*  minAngle,
                                  const double*  maxAngle,
                                  const double*  forceConstant,
                                  const int*     systemIdx,
                                  const double*  positions,
                                  const uint8_t* activeSystemMask,
                                  double*        energyOuts) {
  const int termIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (termIdx < numTerms && systemIsActive(systemIdx[termIdx], activeSystemMask)) {
    const double energy = DistGeom::angleConstraintEnergy(positions,
                                                          idx1[termIdx],
                                                          idx2[termIdx],
                                                          idx3[termIdx],
                                                          minAngle[termIdx],
                                                          maxAngle[termIdx],
                                                          forceConstant[termIdx]);
    atomicAdd(&energyOuts[systemIdx[termIdx]], energy);
  }
}

__global__ void angleGradientKernel(const int      numTerms,
                                    const int*     idx1,
                                    const int*     idx2,
                                    const int*     idx3,
                                    const double*  minAngle,
                                    const double*  maxAngle,
                                    const double*  forceConstant,
                                    const int*     systemIdx,
                                    const double*  positions,
                                    const uint8_t* activeSystemMask,
                                    double*        gradient) {
  const int termIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (termIdx < numTerms && systemIsActive(systemIdx[termIdx], activeSystemMask)) {
    DistGeom::angleConstraintGrad(positions,
                                  idx1[termIdx],
                                  idx2[termIdx],
                                  idx3[termIdx],
                                  minAngle[termIdx],
                                  maxAngle[termIdx],
                                  forceConstant[termIdx],
                                  gradient);
  }
}

__global__ void torsionEnergyKernel(const int      numTerms,
                                    const int*     idx1,
                                    const int*     idx2,
                                    const int*     idx3,
                                    const int*     idx4,
                                    const double*  forceConstants,
                                    const int*     signs,
                                    const int*     systemIdx,
                                    const double*  positions,
                                    const uint8_t* activeSystemMask,
                                    double*        energyOuts) {
  const int termIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (termIdx < numTerms && systemIsActive(systemIdx[termIdx], activeSystemMask)) {
    const double energy = DistGeom::torsionAngleEnergy(positions,
                                                       idx1[termIdx],
                                                       idx2[termIdx],
                                                       idx3[termIdx],
                                                       idx4[termIdx],
                                                       forceConstants + termIdx * 6,
                                                       signs + termIdx * 6);
    atomicAdd(&energyOuts[systemIdx[termIdx]], energy);
  }
}

__global__ void torsionGradientKernel(const int      numTerms,
                                      const int*     idx1,
                                      const int*     idx2,
                                      const int*     idx3,
                                      const int*     idx4,
                                      const double*  forceConstants,
                                      const int*     signs,
                                      const int*     systemIdx,
                                      const double*  positions,
                                      const uint8_t* activeSystemMask,
                                      double*        gradient) {
  const int termIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (termIdx < numTerms && systemIsActive(systemIdx[termIdx], activeSystemMask)) {
    DistGeom::torsionAngleGrad(positions,
                               idx1[termIdx],
                               idx2[termIdx],
                               idx3[termIdx],
                               idx4[termIdx],
                               forceConstants + termIdx * 6,
                               signs + termIdx * 6,
                               gradient);
  }
}

__global__ void planarityEnergyKernel(const int      numTerms,
                                      const int*     idx1,
                                      const int*     idx2,
                                      const int*     idx3,
                                      const int*     idx4,
                                      const double*  forceConstant,
                                      const int*     systemIdx,
                                      const double*  positions,
                                      const uint8_t* activeSystemMask,
                                      double*        energyOuts) {
  const int termIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (termIdx < numTerms && systemIsActive(systemIdx[termIdx], activeSystemMask)) {
    const double energy =
      planarityEnergy(positions, idx1[termIdx], idx2[termIdx], idx3[termIdx], idx4[termIdx], forceConstant[termIdx]);
    atomicAdd(&energyOuts[systemIdx[termIdx]], energy);
  }
}

template <int component>
__global__ void planarityGradientKernel(const int      numTerms,
                                        const int*     idx1,
                                        const int*     idx2,
                                        const int*     idx3,
                                        const int*     idx4,
                                        const double*  forceConstant,
                                        const int*     systemIdx,
                                        const double*  positions,
                                        const uint8_t* activeSystemMask,
                                        double*        gradient) {
  const int termIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (termIdx < numTerms && systemIsActive(systemIdx[termIdx], activeSystemMask)) {
    planarityGradientComponent<component>(positions,
                                          idx1[termIdx],
                                          idx2[termIdx],
                                          idx3[termIdx],
                                          idx4[termIdx],
                                          forceConstant[termIdx],
                                          gradient);
  }
}

template <typename T>
void setStreamAndUpload(AsyncDeviceVector<T>& destination, const std::vector<T>& source, const cudaStream_t stream) {
  destination.setStream(stream);
  destination.setFromVector(source);
}

}  // namespace

AllInOneETKDGBatchedForcefield::AllInOneETKDGBatchedForcefield(
  const DistGeom::BatchedMolecularSystemHost&             dgSystemHost,
  const std::vector<DistGeom::AllInOneForceContribsHost>& systemContribs,
  const std::vector<int>&                                 atomStartsHost,
  BatchedForcefieldMetadata                               metadata,
  const cudaStream_t                                      stream)
    : BatchedForcefield(ForceFieldType::AIO_ETKDG, 4, atomStartsHost, nullptr, metadata),
      dgForcefield_(dgSystemHost, atomStartsHost, 1.0, 2.15, metadata, stream) {
  std::vector<int>    harmonicIdx1;
  std::vector<int>    harmonicIdx2;
  std::vector<double> harmonicMinLen;
  std::vector<double> harmonicMaxLen;
  std::vector<double> harmonicForceConstant;
  std::vector<int>    harmonicSystemIdx;
  std::vector<int>    angleIdx1;
  std::vector<int>    angleIdx2;
  std::vector<int>    angleIdx3;
  std::vector<double> angleMin;
  std::vector<double> angleMax;
  std::vector<double> angleForceConstant;
  std::vector<int>    angleSystemIdx;
  std::vector<int>    torsionIdx1;
  std::vector<int>    torsionIdx2;
  std::vector<int>    torsionIdx3;
  std::vector<int>    torsionIdx4;
  std::vector<double> torsionForceConstants;
  std::vector<int>    torsionSigns;
  std::vector<int>    torsionSystemIdx;
  std::vector<int>    planarityIdx1;
  std::vector<int>    planarityIdx2;
  std::vector<int>    planarityIdx3;
  std::vector<int>    planarityIdx4;
  std::vector<double> planarityForceConstant;
  std::vector<int>    planaritySystemIdx;

  for (int systemIdx = 0; systemIdx < static_cast<int>(systemContribs.size()); ++systemIdx) {
    const int   atomOffset = atomStartsHost[systemIdx];
    const auto& contribs   = systemContribs[systemIdx];
    for (std::size_t termIdx = 0; termIdx < contribs.harmonicDistanceTerms.idx1.size(); ++termIdx) {
      harmonicIdx1.push_back(contribs.harmonicDistanceTerms.idx1[termIdx] + atomOffset);
      harmonicIdx2.push_back(contribs.harmonicDistanceTerms.idx2[termIdx] + atomOffset);
      harmonicMinLen.push_back(contribs.harmonicDistanceTerms.minLen[termIdx]);
      harmonicMaxLen.push_back(contribs.harmonicDistanceTerms.maxLen[termIdx]);
      harmonicForceConstant.push_back(contribs.harmonicDistanceTerms.forceConstant[termIdx]);
      harmonicSystemIdx.push_back(systemIdx);
    }
    for (std::size_t termIdx = 0; termIdx < contribs.angleTerms.idx1.size(); ++termIdx) {
      angleIdx1.push_back(contribs.angleTerms.idx1[termIdx] + atomOffset);
      angleIdx2.push_back(contribs.angleTerms.idx2[termIdx] + atomOffset);
      angleIdx3.push_back(contribs.angleTerms.idx3[termIdx] + atomOffset);
      angleMin.push_back(contribs.angleTerms.minAngle[termIdx]);
      angleMax.push_back(contribs.angleTerms.maxAngle[termIdx]);
      angleForceConstant.push_back(contribs.angleTerms.forceConstant[termIdx]);
      angleSystemIdx.push_back(systemIdx);
    }
    for (std::size_t termIdx = 0; termIdx < contribs.experimentalTorsionTerms.idx1.size(); ++termIdx) {
      torsionIdx1.push_back(contribs.experimentalTorsionTerms.idx1[termIdx] + atomOffset);
      torsionIdx2.push_back(contribs.experimentalTorsionTerms.idx2[termIdx] + atomOffset);
      torsionIdx3.push_back(contribs.experimentalTorsionTerms.idx3[termIdx] + atomOffset);
      torsionIdx4.push_back(contribs.experimentalTorsionTerms.idx4[termIdx] + atomOffset);
      torsionSystemIdx.push_back(systemIdx);
      for (int component = 0; component < 6; ++component) {
        torsionForceConstants.push_back(contribs.experimentalTorsionTerms.forceConstants[termIdx * 6 + component]);
        torsionSigns.push_back(contribs.experimentalTorsionTerms.signs[termIdx * 6 + component]);
      }
    }
    for (std::size_t termIdx = 0; termIdx < contribs.planarityTerms.idx1.size(); ++termIdx) {
      planarityIdx1.push_back(contribs.planarityTerms.idx1[termIdx] + atomOffset);
      planarityIdx2.push_back(contribs.planarityTerms.idx2[termIdx] + atomOffset);
      planarityIdx3.push_back(contribs.planarityTerms.idx3[termIdx] + atomOffset);
      planarityIdx4.push_back(contribs.planarityTerms.idx4[termIdx] + atomOffset);
      planarityForceConstant.push_back(contribs.planarityTerms.forceConstant[termIdx]);
      planaritySystemIdx.push_back(systemIdx);
    }
  }

  setStreamAndUpload(atomStartsDevice_, atomStartsHost, stream);
  setAtomStartsDevice(atomStartsDevice_.data());
  setStreamAndUpload(harmonicIdx1_, harmonicIdx1, stream);
  setStreamAndUpload(harmonicIdx2_, harmonicIdx2, stream);
  setStreamAndUpload(harmonicMinLen_, harmonicMinLen, stream);
  setStreamAndUpload(harmonicMaxLen_, harmonicMaxLen, stream);
  setStreamAndUpload(harmonicForceConstant_, harmonicForceConstant, stream);
  setStreamAndUpload(harmonicSystemIdx_, harmonicSystemIdx, stream);
  setStreamAndUpload(angleIdx1_, angleIdx1, stream);
  setStreamAndUpload(angleIdx2_, angleIdx2, stream);
  setStreamAndUpload(angleIdx3_, angleIdx3, stream);
  setStreamAndUpload(angleMin_, angleMin, stream);
  setStreamAndUpload(angleMax_, angleMax, stream);
  setStreamAndUpload(angleForceConstant_, angleForceConstant, stream);
  setStreamAndUpload(angleSystemIdx_, angleSystemIdx, stream);
  setStreamAndUpload(torsionIdx1_, torsionIdx1, stream);
  setStreamAndUpload(torsionIdx2_, torsionIdx2, stream);
  setStreamAndUpload(torsionIdx3_, torsionIdx3, stream);
  setStreamAndUpload(torsionIdx4_, torsionIdx4, stream);
  setStreamAndUpload(torsionForceConstants_, torsionForceConstants, stream);
  setStreamAndUpload(torsionSigns_, torsionSigns, stream);
  setStreamAndUpload(torsionSystemIdx_, torsionSystemIdx, stream);
  setStreamAndUpload(planarityIdx1_, planarityIdx1, stream);
  setStreamAndUpload(planarityIdx2_, planarityIdx2, stream);
  setStreamAndUpload(planarityIdx3_, planarityIdx3, stream);
  setStreamAndUpload(planarityIdx4_, planarityIdx4, stream);
  setStreamAndUpload(planarityForceConstant_, planarityForceConstant, stream);
  setStreamAndUpload(planaritySystemIdx_, planaritySystemIdx, stream);
}

cudaError_t AllInOneETKDGBatchedForcefield::computeEnergy(double*            energyOuts,
                                                          const double*      positions,
                                                          const uint8_t*     activeSystemMask,
                                                          const cudaStream_t stream) {
  cudaError_t error = dgForcefield_.computeEnergy(energyOuts, positions, activeSystemMask, stream);
  if (error != cudaSuccess) {
    return error;
  }
  if (harmonicIdx1_.size() > 0) {
    harmonicEnergyKernel<<<(harmonicIdx1_.size() + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
      harmonicIdx1_.size(),
      harmonicIdx1_.data(),
      harmonicIdx2_.data(),
      harmonicMinLen_.data(),
      harmonicMaxLen_.data(),
      harmonicForceConstant_.data(),
      harmonicSystemIdx_.data(),
      positions,
      activeSystemMask,
      energyOuts);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess && angleIdx1_.size() > 0) {
    angleEnergyKernel<<<(angleIdx1_.size() + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
      angleIdx1_.size(),
      angleIdx1_.data(),
      angleIdx2_.data(),
      angleIdx3_.data(),
      angleMin_.data(),
      angleMax_.data(),
      angleForceConstant_.data(),
      angleSystemIdx_.data(),
      positions,
      activeSystemMask,
      energyOuts);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess && torsionTermsEnabled_ && torsionIdx1_.size() > 0) {
    torsionEnergyKernel<<<(torsionIdx1_.size() + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
      torsionIdx1_.size(),
      torsionIdx1_.data(),
      torsionIdx2_.data(),
      torsionIdx3_.data(),
      torsionIdx4_.data(),
      torsionForceConstants_.data(),
      torsionSigns_.data(),
      torsionSystemIdx_.data(),
      positions,
      activeSystemMask,
      energyOuts);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess && torsionTermsEnabled_ && planarityIdx1_.size() > 0) {
    planarityEnergyKernel<<<(planarityIdx1_.size() + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
      planarityIdx1_.size(),
      planarityIdx1_.data(),
      planarityIdx2_.data(),
      planarityIdx3_.data(),
      planarityIdx4_.data(),
      planarityForceConstant_.data(),
      planaritySystemIdx_.data(),
      positions,
      activeSystemMask,
      energyOuts);
    error = cudaGetLastError();
  }
  return error;
}

cudaError_t AllInOneETKDGBatchedForcefield::computeGradients(double*            grad,
                                                             const double*      positions,
                                                             const uint8_t*     activeSystemMask,
                                                             const cudaStream_t stream) {
  cudaError_t error = dgForcefield_.computeGradients(grad, positions, activeSystemMask, stream);
  if (error != cudaSuccess) {
    return error;
  }
  if (harmonicIdx1_.size() > 0) {
    harmonicGradientKernel<<<(harmonicIdx1_.size() + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
      harmonicIdx1_.size(),
      harmonicIdx1_.data(),
      harmonicIdx2_.data(),
      harmonicMinLen_.data(),
      harmonicMaxLen_.data(),
      harmonicForceConstant_.data(),
      harmonicSystemIdx_.data(),
      positions,
      activeSystemMask,
      grad);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess && angleIdx1_.size() > 0) {
    angleGradientKernel<<<(angleIdx1_.size() + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
      angleIdx1_.size(),
      angleIdx1_.data(),
      angleIdx2_.data(),
      angleIdx3_.data(),
      angleMin_.data(),
      angleMax_.data(),
      angleForceConstant_.data(),
      angleSystemIdx_.data(),
      positions,
      activeSystemMask,
      grad);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess && torsionTermsEnabled_ && torsionIdx1_.size() > 0) {
    torsionGradientKernel<<<(torsionIdx1_.size() + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
      torsionIdx1_.size(),
      torsionIdx1_.data(),
      torsionIdx2_.data(),
      torsionIdx3_.data(),
      torsionIdx4_.data(),
      torsionForceConstants_.data(),
      torsionSigns_.data(),
      torsionSystemIdx_.data(),
      positions,
      activeSystemMask,
      grad);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess && torsionTermsEnabled_ && planarityIdx1_.size() > 0) {
    planarityGradientKernel<0>
      <<<(planarityIdx1_.size() + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(planarityIdx1_.size(),
                                                                                         planarityIdx1_.data(),
                                                                                         planarityIdx2_.data(),
                                                                                         planarityIdx3_.data(),
                                                                                         planarityIdx4_.data(),
                                                                                         planarityForceConstant_.data(),
                                                                                         planaritySystemIdx_.data(),
                                                                                         positions,
                                                                                         activeSystemMask,
                                                                                         grad);
    error = cudaGetLastError();
    if (error == cudaSuccess) {
      planarityGradientKernel<1><<<(planarityIdx1_.size() + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        planarityIdx1_.size(),
        planarityIdx1_.data(),
        planarityIdx2_.data(),
        planarityIdx3_.data(),
        planarityIdx4_.data(),
        planarityForceConstant_.data(),
        planaritySystemIdx_.data(),
        positions,
        activeSystemMask,
        grad);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      planarityGradientKernel<2><<<(planarityIdx1_.size() + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        planarityIdx1_.size(),
        planarityIdx1_.data(),
        planarityIdx2_.data(),
        planarityIdx3_.data(),
        planarityIdx4_.data(),
        planarityForceConstant_.data(),
        planaritySystemIdx_.data(),
        positions,
        activeSystemMask,
        grad);
      error = cudaGetLastError();
    }
  }
  return error;
}

}  // namespace nvMolKit
