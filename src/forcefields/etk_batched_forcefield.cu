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

#include "src/forcefields/dist_geom_kernels.h"
#include "src/forcefields/etk_batched_forcefield.h"
#include "src/utils/device_convert.cuh"

namespace nvMolKit {

namespace {
void allocateEnergyScratch(const DistGeom::BatchedMolecularSystem3DHost& molSystemHost,
                           DistGeom::BatchedMolecular3DDeviceBuffers&    systemDevice) {
  systemDevice.energyBuffer.resize(molSystemHost.indices.energyBufferStarts.back());
  systemDevice.energyBuffer.zero();
}
}  // namespace

ETKBatchedForcefield::ETKBatchedForcefield(const DistGeom::BatchedMolecularSystem3DHost& molSystemHost,
                                           const std::vector<int>&                       atomStartsHost,
                                           const bool                                    useBasicKnowledge,
                                           BatchedForcefieldMetadata                     metadata,
                                           const cudaStream_t                            stream,
                                           const PrecisionOptions                        precision)
    : BatchedForcefield(ForceFieldType::ETK, 4, atomStartsHost, nullptr, std::move(metadata)),
      term_(useBasicKnowledge ? DistGeom::ETKTerm::ALL : DistGeom::ETKTerm::PLAIN) {
  const auto resolved       = resolvePrecisionOptions(precision);
  computeInFloat_           = isFloat32(resolved.forcefieldCompute);
  reduceInFloat_            = isFloat32(resolved.reductionCompute);
  coordinateStorageInFloat_ = isFloat32(resolved.forcefieldCoordinateStorage);
  gradientStorageInFloat_   = isFloat32(resolved.forcefieldGradientStorage);
  atomStartsDevice_.setStream(stream);
  positionsFloat_.setStream(stream);
  gradientsFloat_.setStream(stream);
  positionsComputeDouble_.setStream(stream);
  gradientsComputeDouble_.setStream(stream);
  positionsFloat_.resize(totalPositions());
  gradientsFloat_.resize(totalPositions());
  positionsComputeDouble_.resize(totalPositions());
  gradientsComputeDouble_.resize(totalPositions());
  if (isFloat32(resolved.forcefieldParameterStorage)) {
    auto& buffers = systemDevice_.emplace<DistGeom::BatchedMolecular3DDeviceBuffersF32Params>();
    DistGeom::setStreams(buffers, stream);
    DistGeom::sendContribsAndIndicesToDevice3D(molSystemHost, buffers);
  } else {
    auto& buffers = systemDevice_.emplace<DistGeom::BatchedMolecular3DDeviceBuffers>();
    DistGeom::setStreams(buffers, stream);
    DistGeom::sendContribsAndIndicesToDevice3D(molSystemHost, buffers);
    allocateEnergyScratch(molSystemHost, buffers);
  }
  atomStartsDevice_.setFromVector(atomStartsHost);
  setAtomStartsDevice(atomStartsDevice_.data());
}

cudaError_t ETKBatchedForcefield::computeEnergy(double*        energyOuts,
                                                const double*  positions,
                                                const uint8_t* activeSystemMask,
                                                cudaStream_t   stream) {
  if (computeInFloat_ || coordinateStorageInFloat_) {
    const auto err = detail::convertDeviceArray(positionsFloat_.data(), positions, totalPositions(), stream);
    return err == cudaSuccess ? computeEnergyFloat(energyOuts, positionsFloat_.data(), activeSystemMask, stream) : err;
  }
  if (!reduceInFloat_)
    if (auto* buffers = std::get_if<DistGeom::BatchedMolecular3DDeviceBuffers>(&systemDevice_))
      return DistGeom::computeEnergyETK(*buffers,
                                        energyOuts,
                                        atomStartsDevice_.data(),
                                        positions,
                                        activeSystemMask,
                                        positions,
                                        term_,
                                        stream);
  return std::visit(
    [&](const auto& buffers) {
      return DistGeom::launchBlockPerMolEnergyKernelETKTyped(
        numMolecules(),
        DistGeom::toEnergy3DForceContribsDevicePtr(buffers),
        DistGeom::toBatchedIndices3DDevicePtr(buffers, atomStartsDevice_.data()),
        positions,
        energyOuts,
        reduceInFloat_,
        activeSystemMask,
        stream);
    },
    systemDevice_);
}

cudaError_t ETKBatchedForcefield::computeGradients(double*        grad,
                                                   const double*  positions,
                                                   const uint8_t* activeSystemMask,
                                                   cudaStream_t   stream) {
  if (computeInFloat_ || coordinateStorageInFloat_ || gradientStorageInFloat_) {
    auto err = detail::convertDeviceArray(positionsFloat_.data(), positions, totalPositions(), stream);
    if (err != cudaSuccess)
      return err;
    err = computeGradientsFloat(gradientsFloat_.data(), positionsFloat_.data(), activeSystemMask, stream);
    return err == cudaSuccess ? detail::convertDeviceArray(grad, gradientsFloat_.data(), totalPositions(), stream) :
                                err;
  }
  if (auto* buffers = std::get_if<DistGeom::BatchedMolecular3DDeviceBuffers>(&systemDevice_))
    return DistGeom::computeGradientsETK(*buffers,
                                         grad,
                                         atomStartsDevice_.data(),
                                         positions,
                                         activeSystemMask,
                                         term_,
                                         stream);
  const auto& buffers = std::get<DistGeom::BatchedMolecular3DDeviceBuffersF32Params>(systemDevice_);
  return DistGeom::launchBlockPerMolGradKernelETK(
    numMolecules(),
    DistGeom::toEnergy3DForceContribsDevicePtr(buffers),
    DistGeom::toBatchedIndices3DDevicePtr(buffers, atomStartsDevice_.data()),
    positions,
    grad,
    activeSystemMask,
    stream);
}

cudaError_t ETKBatchedForcefield::computePlanarEnergy(double*        energyOuts,
                                                      const double*  positions,
                                                      const uint8_t* activeSystemMask,
                                                      cudaStream_t   stream) {
  if (computeInFloat_) {
    const auto err = detail::convertDeviceArray(positionsFloat_.data(), positions, totalPositions(), stream);
    if (err != cudaSuccess)
      return err;
    return std::visit(
      [&](const auto& buffers) {
        return DistGeom::launchPlanarEnergyKernelETKF32(
          numMolecules(),
          DistGeom::toEnergy3DForceContribsDevicePtr(buffers),
          DistGeom::toBatchedIndices3DDevicePtr(buffers, atomStartsDevice_.data()),
          positionsFloat_.data(),
          energyOuts,
          reduceInFloat_,
          activeSystemMask,
          stream);
      },
      systemDevice_);
  }
  if (!coordinateStorageInFloat_ && !reduceInFloat_) {
    if (auto* buffers = std::get_if<DistGeom::BatchedMolecular3DDeviceBuffers>(&systemDevice_))
      return DistGeom::computePlanarEnergy(*buffers,
                                           energyOuts,
                                           atomStartsDevice_.data(),
                                           positions,
                                           activeSystemMask,
                                           positions,
                                           stream);
  }
  const double* computePositions = positions;
  if (coordinateStorageInFloat_) {
    auto err = detail::convertDeviceArray(positionsFloat_.data(), positions, totalPositions(), stream);
    if (err != cudaSuccess)
      return err;
    err = detail::convertDeviceArray(positionsComputeDouble_.data(), positionsFloat_.data(), totalPositions(), stream);
    if (err != cudaSuccess)
      return err;
    computePositions = positionsComputeDouble_.data();
  }
  return std::visit(
    [&](const auto& buffers) {
      return DistGeom::launchPlanarEnergyKernelETKTyped(
        numMolecules(),
        DistGeom::toEnergy3DForceContribsDevicePtr(buffers),
        DistGeom::toBatchedIndices3DDevicePtr(buffers, atomStartsDevice_.data()),
        computePositions,
        energyOuts,
        reduceInFloat_,
        activeSystemMask,
        stream);
    },
    systemDevice_);
}

cudaError_t ETKBatchedForcefield::computeEnergyFloat(double*        energyOuts,
                                                     const float*   positions,
                                                     const uint8_t* activeSystemMask,
                                                     cudaStream_t   stream) {
  if (!computeInFloat_) {
    const auto err = detail::convertDeviceArray(positionsComputeDouble_.data(), positions, totalPositions(), stream);
    if (err != cudaSuccess)
      return err;
    return std::visit(
      [&](const auto& buffers) {
        return DistGeom::launchBlockPerMolEnergyKernelETKTyped(
          numMolecules(),
          DistGeom::toEnergy3DForceContribsDevicePtr(buffers),
          DistGeom::toBatchedIndices3DDevicePtr(buffers, atomStartsDevice_.data()),
          positionsComputeDouble_.data(),
          energyOuts,
          reduceInFloat_,
          activeSystemMask,
          stream);
      },
      systemDevice_);
  }
  return std::visit(
    [&](const auto& buffers) {
      return DistGeom::launchBlockPerMolEnergyKernelETKF32(
        numMolecules(),
        DistGeom::toEnergy3DForceContribsDevicePtr(buffers),
        DistGeom::toBatchedIndices3DDevicePtr(buffers, atomStartsDevice_.data()),
        positions,
        energyOuts,
        reduceInFloat_,
        activeSystemMask,
        stream);
    },
    systemDevice_);
}

cudaError_t ETKBatchedForcefield::computeGradientsFloat(float*         grad,
                                                        const float*   positions,
                                                        const uint8_t* activeSystemMask,
                                                        cudaStream_t   stream) {
  if (!computeInFloat_) {
    auto err = detail::convertDeviceArray(positionsComputeDouble_.data(), positions, totalPositions(), stream);
    if (err != cudaSuccess)
      return err;
    err = std::visit(
      [&](const auto& buffers) {
        return DistGeom::launchBlockPerMolGradKernelETK(
          numMolecules(),
          DistGeom::toEnergy3DForceContribsDevicePtr(buffers),
          DistGeom::toBatchedIndices3DDevicePtr(buffers, atomStartsDevice_.data()),
          positionsComputeDouble_.data(),
          gradientsComputeDouble_.data(),
          activeSystemMask,
          stream);
      },
      systemDevice_);
    return err == cudaSuccess ?
             detail::convertDeviceArray(grad, gradientsComputeDouble_.data(), totalPositions(), stream) :
             err;
  }
  return std::visit(
    [&](const auto& buffers) {
      return DistGeom::launchBlockPerMolGradKernelETKF32(
        numMolecules(),
        DistGeom::toEnergy3DForceContribsDevicePtr(buffers),
        DistGeom::toBatchedIndices3DDevicePtr(buffers, atomStartsDevice_.data()),
        positions,
        grad,
        activeSystemMask,
        stream);
    },
    systemDevice_);
}

}  // namespace nvMolKit
