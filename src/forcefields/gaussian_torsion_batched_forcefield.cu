// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <math_constants.h>

#include <numbers>
#include <stdexcept>

#include "src/forcefields/dist_geom_kernels_device_dispatch.cuh"
#include "src/forcefields/gaussian_torsion_batched_forcefield.h"

namespace nvMolKit {
namespace {

constexpr int kBlockSize = 128;

template <bool gradient>
__global__ void gaussianTorsionKernel(int            count,
                                      int            dimension,
                                      const int*     idx1,
                                      const int*     idx2,
                                      const int*     idx3,
                                      const int*     idx4,
                                      const int*     systemIdx,
                                      const int*     tableStarts,
                                      const int*     tableSizes,
                                      const double*  scaling,
                                      const double*  tables,
                                      const double*  positions,
                                      const uint8_t* activeSystems,
                                      double*        output) {
  const int term = blockIdx.x * blockDim.x + threadIdx.x;
  if (term >= count || (activeSystems != nullptr && !activeSystems[systemIdx[term]])) {
    return;
  }
  const int    pos1 = dimension * idx1[term];
  const int    pos2 = dimension * idx2[term];
  const int    pos3 = dimension * idx3[term];
  const int    pos4 = dimension * idx4[term];
  const double r1x  = positions[pos1] - positions[pos2];
  const double r1y  = positions[pos1 + 1] - positions[pos2 + 1];
  const double r1z  = positions[pos1 + 2] - positions[pos2 + 2];
  const double r2x  = positions[pos3] - positions[pos2];
  const double r2y  = positions[pos3 + 1] - positions[pos2 + 1];
  const double r2z  = positions[pos3 + 2] - positions[pos2 + 2];
  const double r4x  = positions[pos4] - positions[pos3];
  const double r4y  = positions[pos4 + 1] - positions[pos3 + 1];
  const double r4z  = positions[pos4 + 2] - positions[pos3 + 2];
  double       n1x, n1y, n1z, n2x, n2y, n2z;
  crossProduct(r1x, r1y, r1z, r2x, r2y, r2z, n1x, n1y, n1z);
  crossProduct(-r2x, -r2y, -r2z, r4x, r4y, r4z, n2x, n2y, n2z);
  const double norm1      = sqrt(n1x * n1x + n1y * n1y + n1z * n1z);
  const double norm2      = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
  const bool   degenerate = isDoubleZero(norm1) || isDoubleZero(norm2);
  if constexpr (gradient) {
    if (degenerate) {
      return;
    }
  }
  double cosine = degenerate ? 0.0 : (n1x * n2x + n1y * n2y + n1z * n2z) / (norm1 * norm2);
  if (!isfinite(cosine)) {
    if constexpr (!gradient) {
      atomicAdd(output + systemIdx[term], CUDART_NAN);
    }
    return;
  }
  cosine                   = clamp(cosine, -1.0, 1.0);
  const double normalized  = acos(cosine) * (tableSizes[term] - 1) / std::numbers::pi;
  const int    cell        = min(static_cast<int>(normalized), tableSizes[term] - 2);
  const double fraction    = normalized - cell;
  const int    tableOffset = tableStarts[term] + cell;
  const double value = scaling[term] * ((1.0 - fraction) * tables[tableOffset] + fraction * tables[tableOffset + 1]);
  if constexpr (!gradient) {
    atomicAdd(output + systemIdx[term], value);
  } else {
    // RDKit interpolates dE/dphi independently of its interpolated energy table.
    const double factor = -value / fmax(sqrt(fmax(1.0 - cosine * cosine, 0.0)), 1e-8);
    n1x /= norm1;
    n1y /= norm1;
    n1z /= norm1;
    n2x /= norm2;
    n2y /= norm2;
    n2z /= norm2;
    const double proj1x = factor * (n2x - cosine * n1x) / norm1;
    const double proj1y = factor * (n2y - cosine * n1y) / norm1;
    const double proj1z = factor * (n2z - cosine * n1z) / norm1;
    const double proj2x = factor * (n1x - cosine * n2x) / norm2;
    const double proj2y = factor * (n1y - cosine * n2y) / norm2;
    const double proj2z = factor * (n1z - cosine * n2z) / norm2;
    atomicAdd(output + pos1, proj1z * r2y - proj1y * r2z);
    atomicAdd(output + pos1 + 1, proj1x * r2z - proj1z * r2x);
    atomicAdd(output + pos1 + 2, proj1y * r2x - proj1x * r2y);
    atomicAdd(output + pos2, proj1y * (r2z - r1z) + proj1z * (r1y - r2y) - proj2y * r4z + proj2z * r4y);
    atomicAdd(output + pos2 + 1, proj1x * (r1z - r2z) + proj1z * (r2x - r1x) + proj2x * r4z - proj2z * r4x);
    atomicAdd(output + pos2 + 2, proj1x * (r2y - r1y) + proj1y * (r1x - r2x) - proj2x * r4y + proj2y * r4x);
    atomicAdd(output + pos3, proj1y * r1z - proj1z * r1y + proj2y * (r4z + r2z) + proj2z * (-r2y - r4y));
    atomicAdd(output + pos3 + 1, -proj1x * r1z + proj1z * r1x + proj2x * (-r2z - r4z) + proj2z * (r4x + r2x));
    atomicAdd(output + pos3 + 2, proj1x * r1y - proj1y * r1x + proj2x * (r4y + r2y) + proj2y * (-r2x - r4x));
    atomicAdd(output + pos4, -proj2y * r2z + proj2z * r2y);
    atomicAdd(output + pos4 + 1, -proj2z * r2x + proj2x * r2z);
    atomicAdd(output + pos4 + 2, -proj2x * r2y + proj2y * r2x);
  }
}

}  // namespace

GaussianTorsionBatchedForcefield::GaussianTorsionBatchedForcefield(
  BatchedForcefield&                           base,
  const DistGeom::GaussianTorsionContribsHost& contributions,
  cudaStream_t                                 stream)
    : BatchedForcefield(ForceFieldType::GAUSSIAN_ETKDG,
                        base.dataDim(),
                        base.atomStartsHost(),
                        base.atomStartsDevice(),
                        base.metadata()),
      base_(base) {
  const auto count = contributions.idx1.size();
  if (contributions.idx2.size() != count || contributions.idx3.size() != count || contributions.idx4.size() != count ||
      contributions.systemIdx.size() != count || contributions.scaling.size() != count ||
      contributions.tableStarts.size() != count || contributions.tableSizes.size() != count ||
      contributions.energies.size() != contributions.derivatives.size()) {
    throw std::invalid_argument("Inconsistent Gaussian torsion contribution sizes");
  }
  for (std::size_t term = 0; term < count; ++term) {
    const int system = contributions.systemIdx[term];
    const int start  = contributions.tableStarts[term];
    const int size   = contributions.tableSizes[term];
    if (system < 0 || system >= numMolecules() || start < 0 || size < 2 ||
        static_cast<std::size_t>(start) + size > contributions.energies.size()) {
      throw std::invalid_argument("Invalid Gaussian torsion table or system index");
    }
    for (const int atom :
         {contributions.idx1[term], contributions.idx2[term], contributions.idx3[term], contributions.idx4[term]}) {
      if (atom < atomStartsHost()[system] || atom >= atomStartsHost()[system + 1]) {
        throw std::invalid_argument("Gaussian torsion atom lies outside its system");
      }
    }
  }
  const auto upload = [stream]<typename T>(AsyncDeviceVector<T>& target, const std::vector<T>& source) {
    target.setStream(stream);
    target.setFromVector(source);
  };
  upload(idx1_, contributions.idx1);
  upload(idx2_, contributions.idx2);
  upload(idx3_, contributions.idx3);
  upload(idx4_, contributions.idx4);
  upload(systemIdx_, contributions.systemIdx);
  upload(tableStarts_, contributions.tableStarts);
  upload(tableSizes_, contributions.tableSizes);
  upload(scaling_, contributions.scaling);
  upload(energies_, contributions.energies);
  upload(derivatives_, contributions.derivatives);
  cudaCheckError(cudaStreamSynchronize(stream));
}

cudaError_t GaussianTorsionBatchedForcefield::computeEnergy(double*        energies,
                                                            const double*  positions,
                                                            const uint8_t* activeSystems,
                                                            cudaStream_t   stream) {
  const auto error = base_.computeEnergy(energies, positions, activeSystems, stream);
  if (error != cudaSuccess || !enabled_ || idx1_.size() == 0) {
    return error;
  }
  gaussianTorsionKernel<false>
    <<<(idx1_.size() + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(idx1_.size(),
                                                                              dataDim(),
                                                                              idx1_.data(),
                                                                              idx2_.data(),
                                                                              idx3_.data(),
                                                                              idx4_.data(),
                                                                              systemIdx_.data(),
                                                                              tableStarts_.data(),
                                                                              tableSizes_.data(),
                                                                              scaling_.data(),
                                                                              energies_.data(),
                                                                              positions,
                                                                              activeSystems,
                                                                              energies);
  return cudaGetLastError();
}

cudaError_t GaussianTorsionBatchedForcefield::computeGradients(double*        gradients,
                                                               const double*  positions,
                                                               const uint8_t* activeSystems,
                                                               cudaStream_t   stream) {
  const auto error = base_.computeGradients(gradients, positions, activeSystems, stream);
  if (error != cudaSuccess || !enabled_ || idx1_.size() == 0) {
    return error;
  }
  gaussianTorsionKernel<true>
    <<<(idx1_.size() + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(idx1_.size(),
                                                                              dataDim(),
                                                                              idx1_.data(),
                                                                              idx2_.data(),
                                                                              idx3_.data(),
                                                                              idx4_.data(),
                                                                              systemIdx_.data(),
                                                                              tableStarts_.data(),
                                                                              tableSizes_.data(),
                                                                              scaling_.data(),
                                                                              derivatives_.data(),
                                                                              positions,
                                                                              activeSystems,
                                                                              gradients);
  return cudaGetLastError();
}

}  // namespace nvMolKit
