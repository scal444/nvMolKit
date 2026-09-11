// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "src/forcefields/mmff_batched_forcefield.h"
#include "src/forcefields/mmff_kernels.h"
#include "src/utils/device_convert.cuh"

namespace nvMolKit {

namespace {
void allocateEnergyScratch(const MMFF::BatchedMolecularSystemHost& molSystemHost,
                           MMFF::BatchedMolecularDeviceBuffers&    systemDevice) {
  systemDevice.energyBuffer.resize(molSystemHost.indices.energyBufferStarts.back());
  systemDevice.energyBuffer.zero();
}

template <typename Buffers, typename coordinateT>
cudaError_t launchEnergy(Buffers&           buffers,
                         int                numMols,
                         const coordinateT* positions,
                         double*            energies,
                         const uint8_t*     activeSystemMask,
                         bool               computeInFloat,
                         bool               reduceInFloat,
                         cudaStream_t       stream) {
  return MMFF::launchBlockPerMolEnergyKernel(numMols,
                                             MMFF::toEnergyForceContribsDevicePtr(buffers),
                                             MMFF::toBatchedIndicesDevicePtr(buffers),
                                             positions,
                                             energies,
                                             MMFF::batchHasConstraints(buffers.contribs),
                                             computeInFloat,
                                             reduceInFloat,
                                             stream,
                                             activeSystemMask);
}

template <typename Buffers, typename coordinateT, typename storageT>
cudaError_t launchGrad(Buffers&           buffers,
                       int                numMols,
                       const coordinateT* positions,
                       storageT*          gradients,
                       const uint8_t*     activeSystemMask,
                       bool               computeInFloat,
                       cudaStream_t       stream) {
  return MMFF::launchBlockPerMolGradKernel(numMols,
                                           MMFF::toEnergyForceContribsDevicePtr(buffers),
                                           MMFF::toBatchedIndicesDevicePtr(buffers),
                                           positions,
                                           gradients,
                                           MMFF::batchHasConstraints(buffers.contribs),
                                           computeInFloat,
                                           stream,
                                           activeSystemMask);
}
}  // namespace

MMFFBatchedForcefield::MMFFBatchedForcefield(const MMFF::BatchedMolecularSystemHost& molSystemHost,
                                             BatchedForcefieldMetadata               metadata,
                                             const cudaStream_t                      stream,
                                             const PrecisionOptions                  precision)
    : BatchedForcefield(ForceFieldType::MMFF, 3, molSystemHost.indices.atomStarts, nullptr, std::move(metadata)) {
  forcefieldCoordinateStorageInFloat_ = usesFloatForcefieldCoordinates(precision);
  forcefieldGradientStorageInFloat_   = usesFloatForcefieldGradients(precision);
  computeInFloat_                     = usesFloatForcefieldCompute(precision);
  reduceInFloat_                      = usesFloatReduction(precision);
  positionsFloat_.setStream(stream);
  gradientsFloat_.setStream(stream);
  if (forcefieldCoordinateStorageInFloat_)
    positionsFloat_.resize(totalPositions());
  if (forcefieldGradientStorageInFloat_)
    gradientsFloat_.resize(totalPositions());
  if (usesFloatForcefield(precision)) {
    auto& buffers = systemDevice_.emplace<MMFF::BatchedMolecularDeviceBuffersF32Params>();
    MMFF::setStreams(buffers, stream);
    MMFF::sendContribsAndIndicesToDevice(molSystemHost, buffers);
    setAtomStartsDevice(buffers.indices.atomStarts.data());
  } else {
    auto& buffers = systemDevice_.emplace<MMFF::BatchedMolecularDeviceBuffers>();
    MMFF::setStreams(buffers, stream);
    MMFF::sendContribsAndIndicesToDevice(molSystemHost, buffers);
    allocateEnergyScratch(molSystemHost, buffers);
    setAtomStartsDevice(buffers.indices.atomStarts.data());
  }
}

cudaError_t MMFFBatchedForcefield::computeEnergyFloat(double*        energyOuts,
                                                      const float*   positions,
                                                      const uint8_t* activeSystemMask,
                                                      cudaStream_t   stream) {
  return std::visit(
    [&](auto& buffers) {
      return launchEnergy(buffers,
                          numMolecules(),
                          positions,
                          energyOuts,
                          activeSystemMask,
                          computeInFloat_,
                          reduceInFloat_,
                          stream);
    },
    systemDevice_);
}

cudaError_t MMFFBatchedForcefield::computeGradientsFloat(float*         grad,
                                                         const float*   positions,
                                                         const uint8_t* activeSystemMask,
                                                         cudaStream_t   stream) {
  return std::visit(
    [&](auto& buffers) {
      return launchGrad(buffers, numMolecules(), positions, grad, activeSystemMask, computeInFloat_, stream);
    },
    systemDevice_);
}

cudaError_t MMFFBatchedForcefield::computeEnergy(double*        energyOuts,
                                                 const double*  positions,
                                                 const uint8_t* activeSystemMask,
                                                 cudaStream_t   stream) {
  if (forcefieldCoordinateStorageInFloat_) {
    const auto err = detail::convertDeviceArray(positionsFloat_.data(), positions, totalPositions(), stream);
    return err == cudaSuccess ? computeEnergyFloat(energyOuts, positionsFloat_.data(), activeSystemMask, stream) : err;
  }
  return std::visit(
    [&](auto& buffers) {
      return launchEnergy(buffers,
                          numMolecules(),
                          positions,
                          energyOuts,
                          activeSystemMask,
                          computeInFloat_,
                          reduceInFloat_,
                          stream);
    },
    systemDevice_);
}

cudaError_t MMFFBatchedForcefield::computeGradients(double*        grad,
                                                    const double*  positions,
                                                    const uint8_t* activeSystemMask,
                                                    cudaStream_t   stream) {
  const float* positionsF = nullptr;
  if (forcefieldCoordinateStorageInFloat_) {
    const auto err = detail::convertDeviceArray(positionsFloat_.data(), positions, totalPositions(), stream);
    if (err != cudaSuccess)
      return err;
    positionsF = positionsFloat_.data();
  }
  cudaError_t err;
  if (forcefieldGradientStorageInFloat_) {
    err = std::visit(
      [&](auto& buffers) {
        return positionsF != nullptr ? launchGrad(buffers,
                                                  numMolecules(),
                                                  positionsF,
                                                  gradientsFloat_.data(),
                                                  activeSystemMask,
                                                  computeInFloat_,
                                                  stream) :
                                       launchGrad(buffers,
                                                  numMolecules(),
                                                  positions,
                                                  gradientsFloat_.data(),
                                                  activeSystemMask,
                                                  computeInFloat_,
                                                  stream);
      },
      systemDevice_);
    if (err == cudaSuccess)
      err = detail::convertDeviceArray(grad, gradientsFloat_.data(), totalPositions(), stream);
  } else {
    err = std::visit(
      [&](auto& buffers) {
        return positionsF != nullptr ?
                 launchGrad(buffers, numMolecules(), positionsF, grad, activeSystemMask, computeInFloat_, stream) :
                 launchGrad(buffers, numMolecules(), positions, grad, activeSystemMask, computeInFloat_, stream);
      },
      systemDevice_);
  }
  return err;
}

}  // namespace nvMolKit
