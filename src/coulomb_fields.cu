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

#include <algorithm>
#include <cmath>
#include <limits>

#include "src/coulomb_fields.h"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit {
namespace {

constexpr int    kBlockSize        = 128;
// RDKit RDMIF CoulombDetail::prefactor converts e^2/Angstrom to kJ/mol.
constexpr double kCoulombPrefactor = 1 / (4.0 * 3.141592 * 8.854188) * 1.602 * 1.602 * 6.02214129 * 10000;

__device__ __forceinline__ double burialDistance(const size_t neighbors) {
  if (neighbors <= 6)
    return 0.0;
  switch (neighbors) {
    case 7:
      return 0.4;
    case 8:
      return 0.9;
    case 9:
      return 1.4;
    case 10:
      return 1.9;
    case 11:
      return 2.6;
    default:
      return 4.0;
  }
}

__device__ __forceinline__ double squaredDistance(const double  x,
                                                  const double  y,
                                                  const double  z,
                                                  const double* position) {
  const double dx = x - position[0];
  const double dy = y - position[1];
  const double dz = z - position[2];
  return dx * dx + dy * dy + dz * dz;
}

__global__ void coulombAtomBurialKernel(const double* positions,
                                        const size_t* offsets,
                                        double*       burial,
                                        const int     moleculeCount) {
  for (size_t molecule = blockIdx.x; molecule < static_cast<size_t>(moleculeCount); molecule += gridDim.x) {
    const size_t begin = offsets[molecule];
    const size_t end   = offsets[molecule + 1];
    for (size_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
      const double x         = positions[3 * atom];
      const double y         = positions[3 * atom + 1];
      const double z         = positions[3 * atom + 2];
      size_t       neighbors = 0;
      for (size_t other = begin; other < end; ++other)
        if (other != atom && squaredDistance(x, y, z, positions + 3 * other) < 16.0)
          ++neighbors;
      burial[atom] = burialDistance(neighbors);
    }
  }
}

template <bool Dielectric>
__global__ void coulombFieldsKernel(const double* positions,
                                    const double* charges,
                                    const size_t* offsets,
                                    const double* points,
                                    const double* burial,
                                    double*       output,
                                    const size_t  pointCount,
                                    const size_t  outputCount,
                                    const double  alpha,
                                    const double  cutoffSquared,
                                    const double  thresholdSquared,
                                    const double  scale,
                                    const double  dielectricFactor,
                                    const bool    absoluteValue) {
  for (size_t item = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; item < outputCount;
       item += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const size_t molecule    = item / pointCount;
    const size_t point       = item % pointCount;
    const size_t begin       = offsets[molecule];
    const size_t end         = offsets[molecule + 1];
    const double x           = points[3 * point];
    const double y           = points[3 * point + 1];
    const double z           = points[3 * point + 2];
    double       queryBurial = 0.0;
    if constexpr (Dielectric) {
      size_t neighbors = 0;
      for (size_t atom = begin; atom < end; ++atom)
        if (squaredDistance(x, y, z, positions + 3 * atom) < 16.0)
          ++neighbors;
      queryBurial = burialDistance(neighbors);
    }
    double energy = 0.0;
    for (size_t atom = begin; atom < end; ++atom) {
      const double distanceSquared = squaredDistance(x, y, z, positions + 3 * atom);
      if (distanceSquared < thresholdSquared) {
        const double softenedDistance = alpha >= 1e-6 ? alpha + distanceSquared : fmax(distanceSquared, cutoffSquared);
        double       interaction      = 1.0 / sqrt(softenedDistance);
        if constexpr (Dielectric)
          interaction += dielectricFactor / sqrt(softenedDistance + 4.0 * burial[atom] * queryBurial);
        energy += charges[atom] * interaction;
      }
    }
    energy *= scale;
    output[item] = absoluteValue ? -fabs(energy) : energy;
  }
}

}  // namespace

void coulombFieldsGpu(cuda::std::span<const double> positions,
                      cuda::std::span<const double> charges,
                      cuda::std::span<const size_t> offsets,
                      cuda::std::span<const double> points,
                      cuda::std::span<double>       burial,
                      cuda::std::span<double>       output,
                      const CoulombFieldOptions&    options,
                      const int                     moleculeCount,
                      cudaStream_t                  stream) {
  if (output.empty())
    return;
  const size_t pointCount = points.size() / 3;
  const int    blocks     = static_cast<int>(std::min<size_t>((output.size() + kBlockSize - 1) / kBlockSize, 65535));
  const double thresholdSquared =
    options.threshold < 0 ? std::numeric_limits<double>::infinity() : options.threshold * options.threshold;
  double cutoffSquared = options.cutoff * options.cutoff;
  if (options.dielectric) {
    if (cutoffSquared < 1e-12)
      cutoffSquared = 1e-6;
    coulombAtomBurialKernel<<<std::min(moleculeCount, 65535), kBlockSize, 0, stream>>>(positions.data(),
                                                                                       offsets.data(),
                                                                                       burial.data(),
                                                                                       moleculeCount);
    cudaCheckError(cudaGetLastError());
    const double factor = (options.xi - options.epsilon) / (options.xi + options.epsilon);
    coulombFieldsKernel<true>
      <<<blocks, kBlockSize, 0, stream>>>(positions.data(),
                                          charges.data(),
                                          offsets.data(),
                                          points.data(),
                                          burial.data(),
                                          output.data(),
                                          pointCount,
                                          output.size(),
                                          options.alpha,
                                          cutoffSquared,
                                          thresholdSquared,
                                          kCoulombPrefactor * (1.0 / options.xi) * options.probeCharge,
                                          factor,
                                          options.absoluteValue);
  } else {
    if (cutoffSquared < 1e-6)
      cutoffSquared = 0.001;
    coulombFieldsKernel<false><<<blocks, kBlockSize, 0, stream>>>(positions.data(),
                                                                  charges.data(),
                                                                  offsets.data(),
                                                                  points.data(),
                                                                  nullptr,
                                                                  output.data(),
                                                                  pointCount,
                                                                  output.size(),
                                                                  options.alpha,
                                                                  cutoffSquared,
                                                                  thresholdSquared,
                                                                  kCoulombPrefactor * options.probeCharge,
                                                                  0.0,
                                                                  options.absoluteValue);
  }
  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit
