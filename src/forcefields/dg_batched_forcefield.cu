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

#include "src/forcefields/dg_batched_forcefield.h"
#include "src/forcefields/dist_geom_kernels.h"
#include "src/utils/device_convert.cuh"

namespace nvMolKit {

namespace {
void allocateEnergyScratch(const DistGeom::BatchedMolecularSystemHost& molSystemHost,
                           DistGeom::BatchedMolecularDeviceBuffers&    systemDevice) {
  systemDevice.energyBuffer.resize(molSystemHost.indices.energyBufferStarts.back());
  systemDevice.energyBuffer.zero();
  systemDevice.dimension = molSystemHost.dimension;
}
}  // namespace

DGBatchedForcefield::DGBatchedForcefield(const DistGeom::BatchedMolecularSystemHost& molSystemHost,
                                         const std::vector<int>&                     atomStartsHost,
                                         const double                                chiralWeight,
                                         const double                                fourthDimWeight,
                                         BatchedForcefieldMetadata                   metadata,
                                         const cudaStream_t                          stream,
                                         const PrecisionOptions                      precision)
    : BatchedForcefield(ForceFieldType::DG, molSystemHost.dimension, atomStartsHost, nullptr, std::move(metadata)),
      chiralWeight_(chiralWeight),
      fourthDimWeight_(fourthDimWeight) {
  const auto resolved       = resolvePrecisionOptions(precision);
  coordinateStorageInFloat_ = isFloat32(resolved.forcefieldCoordinateStorage);
  gradientStorageInFloat_   = isFloat32(resolved.forcefieldGradientStorage);
  computeInFloat_           = isFloat32(resolved.forcefieldCompute);
  reduceInFloat_            = isFloat32(resolved.reductionCompute);
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
    auto& buffers = systemDevice_.emplace<DistGeom::BatchedMolecularDeviceBuffersF32Params>();
    DistGeom::setStreams(buffers, stream);
    DistGeom::sendContribsAndIndicesToDevice(molSystemHost, buffers);
    buffers.dimension = molSystemHost.dimension;
  } else {
    auto& buffers = systemDevice_.emplace<DistGeom::BatchedMolecularDeviceBuffers>();
    DistGeom::setStreams(buffers, stream);
    DistGeom::sendContribsAndIndicesToDevice(molSystemHost, buffers);
    allocateEnergyScratch(molSystemHost, buffers);
  }
  atomStartsDevice_.setFromVector(atomStartsHost);
  setAtomStartsDevice(atomStartsDevice_.data());
}

cudaError_t DGBatchedForcefield::computeEnergy(double*        energyOuts,
                                               const double*  positions,
                                               const uint8_t* activeSystemMask,
                                               cudaStream_t   stream) {
  if (computeInFloat_ || coordinateStorageInFloat_) {
    const auto err = detail::convertDeviceArray(positionsFloat_.data(), positions, totalPositions(), stream);
    return err == cudaSuccess ? computeEnergyFloat(energyOuts, positionsFloat_.data(), activeSystemMask, stream) : err;
  }
  if (!reduceInFloat_)
    if (auto* buffers = std::get_if<DistGeom::BatchedMolecularDeviceBuffers>(&systemDevice_))
      return DistGeom::computeEnergy(*buffers,
                                     energyOuts,
                                     atomStartsDevice_.data(),
                                     positions,
                                     chiralWeight_,
                                     fourthDimWeight_,
                                     activeSystemMask,
                                     positions,
                                     stream);
  return std::visit(
    [&](const auto& buffers) {
      return DistGeom::launchBlockPerMolEnergyKernelTyped(
        numMolecules(),
        DistGeom::toEnergyForceContribsDevicePtr(buffers),
        DistGeom::toBatchedIndicesDevicePtr(buffers, atomStartsDevice_.data()),
        positions,
        energyOuts,
        dataDim(),
        chiralWeight_,
        fourthDimWeight_,
        reduceInFloat_,
        activeSystemMask,
        stream);
    },
    systemDevice_);
}

cudaError_t DGBatchedForcefield::computeGradients(double*        grad,
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
  if (auto* buffers = std::get_if<DistGeom::BatchedMolecularDeviceBuffers>(&systemDevice_))
    return DistGeom::computeGradients(*buffers,
                                      grad,
                                      atomStartsDevice_.data(),
                                      positions,
                                      chiralWeight_,
                                      fourthDimWeight_,
                                      activeSystemMask,
                                      stream);
  const auto& buffers = std::get<DistGeom::BatchedMolecularDeviceBuffersF32Params>(systemDevice_);
  return DistGeom::launchBlockPerMolGradKernel(numMolecules(),
                                               DistGeom::toEnergyForceContribsDevicePtr(buffers),
                                               DistGeom::toBatchedIndicesDevicePtr(buffers, atomStartsDevice_.data()),
                                               positions,
                                               grad,
                                               dataDim(),
                                               chiralWeight_,
                                               fourthDimWeight_,
                                               activeSystemMask,
                                               stream);
}

cudaError_t DGBatchedForcefield::computeEnergyFloat(double*        energyOuts,
                                                    const float*   positions,
                                                    const uint8_t* activeSystemMask,
                                                    cudaStream_t   stream) {
  if (!computeInFloat_) {
    const auto err = detail::convertDeviceArray(positionsComputeDouble_.data(), positions, totalPositions(), stream);
    if (err != cudaSuccess)
      return err;
    return std::visit(
      [&](const auto& buffers) {
        return DistGeom::launchBlockPerMolEnergyKernelTyped(
          numMolecules(),
          DistGeom::toEnergyForceContribsDevicePtr(buffers),
          DistGeom::toBatchedIndicesDevicePtr(buffers, atomStartsDevice_.data()),
          positionsComputeDouble_.data(),
          energyOuts,
          dataDim(),
          chiralWeight_,
          fourthDimWeight_,
          reduceInFloat_,
          activeSystemMask,
          stream);
      },
      systemDevice_);
  }
  return std::visit(
    [&](const auto& buffers) {
      return DistGeom::launchBlockPerMolEnergyKernelF32(
        numMolecules(),
        DistGeom::toEnergyForceContribsDevicePtr(buffers),
        DistGeom::toBatchedIndicesDevicePtr(buffers, atomStartsDevice_.data()),
        positions,
        energyOuts,
        dataDim(),
        static_cast<float>(chiralWeight_),
        static_cast<float>(fourthDimWeight_),
        reduceInFloat_,
        activeSystemMask,
        stream);
    },
    systemDevice_);
}

cudaError_t DGBatchedForcefield::computeGradientsFloat(float*         grad,
                                                       const float*   positions,
                                                       const uint8_t* activeSystemMask,
                                                       cudaStream_t   stream) {
  if (!computeInFloat_) {
    auto err = detail::convertDeviceArray(positionsComputeDouble_.data(), positions, totalPositions(), stream);
    if (err != cudaSuccess)
      return err;
    err = std::visit(
      [&](const auto& buffers) {
        return DistGeom::launchBlockPerMolGradKernel(
          numMolecules(),
          DistGeom::toEnergyForceContribsDevicePtr(buffers),
          DistGeom::toBatchedIndicesDevicePtr(buffers, atomStartsDevice_.data()),
          positionsComputeDouble_.data(),
          gradientsComputeDouble_.data(),
          dataDim(),
          chiralWeight_,
          fourthDimWeight_,
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
      return DistGeom::launchBlockPerMolGradKernelF32(
        numMolecules(),
        DistGeom::toEnergyForceContribsDevicePtr(buffers),
        DistGeom::toBatchedIndicesDevicePtr(buffers, atomStartsDevice_.data()),
        positions,
        grad,
        dataDim(),
        static_cast<float>(chiralWeight_),
        static_cast<float>(fourthDimWeight_),
        activeSystemMask,
        stream);
    },
    systemDevice_);
}

}  // namespace nvMolKit
