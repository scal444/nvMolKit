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

#include <cub/cub.cuh>

#include "src/forcefields/kernel_utils.cuh"
#include "src/forcefields/uff_kernels.h"
#include "src/forcefields/uff_kernels_device.cuh"

using namespace nvMolKit::UFF::fp64;

using namespace nvMolKit::FFKernelUtils;

namespace {

__global__ void bondStretchEnergyKernel(const int     numBonds,
                                        const int*    idx1,
                                        const int*    idx2,
                                        const double* restLen,
                                        const double* forceConstant,
                                        const double* pos,
                                        double*       energyBuffer,
                                        const int*    energyBufferStarts,
                                        const int*    atomBatchMap,
                                        const int*    termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numBonds) {
    const double energy    = uffBondStretchEnergy(pos, idx1[idx], idx2[idx], restLen[idx], forceConstant[idx]);
    const int    batchIdx  = atomBatchMap[idx1[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void bondStretchGradKernel(const int     numBonds,
                                      const int*    idx1,
                                      const int*    idx2,
                                      const double* restLen,
                                      const double* forceConstant,
                                      const double* pos,
                                      double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numBonds) {
    uffBondStretchGrad(pos, idx1[idx], idx2[idx], restLen[idx], forceConstant[idx], grad);
  }
}

__global__ void angleBendEnergyKernel(const int      numAngles,
                                      const int*     idx1,
                                      const int*     idx2,
                                      const int*     idx3,
                                      const double*  theta0,
                                      const double*  forceConstant,
                                      const uint8_t* order,
                                      const double*  C0,
                                      const double*  C1,
                                      const double*  C2,
                                      const double*  pos,
                                      double*        energyBuffer,
                                      const int*     energyBufferStarts,
                                      const int*     atomBatchMap,
                                      const int*     termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numAngles) {
    const double energy    = uffAngleBendEnergy(pos,
                                             idx1[idx],
                                             idx2[idx],
                                             idx3[idx],
                                             theta0[idx],
                                             forceConstant[idx],
                                             order[idx],
                                             C0[idx],
                                             C1[idx],
                                             C2[idx]);
    const int    batchIdx  = atomBatchMap[idx1[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void angleBendGradKernel(const int      numAngles,
                                    const int*     idx1,
                                    const int*     idx2,
                                    const int*     idx3,
                                    const double*  theta0,
                                    const double*  forceConstant,
                                    const uint8_t* order,
                                    const double*  C0,
                                    const double*  C1,
                                    const double*  C2,
                                    const double*  pos,
                                    double*        grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numAngles) {
    uffAngleBendGrad(pos,
                     idx1[idx],
                     idx2[idx],
                     idx3[idx],
                     theta0[idx],
                     forceConstant[idx],
                     order[idx],
                     C0[idx],
                     C1[idx],
                     C2[idx],
                     grad);
  }
}

__global__ void torsionEnergyKernel(const int      numTorsions,
                                    const int*     idx1,
                                    const int*     idx2,
                                    const int*     idx3,
                                    const int*     idx4,
                                    const double*  forceConstant,
                                    const uint8_t* order,
                                    const double*  cosTerm,
                                    const double*  pos,
                                    double*        energyBuffer,
                                    const int*     energyBufferStarts,
                                    const int*     atomBatchMap,
                                    const int*     termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTorsions) {
    const double energy =
      uffTorsionEnergy(pos, idx1[idx], idx2[idx], idx3[idx], idx4[idx], forceConstant[idx], order[idx], cosTerm[idx]);
    const int batchIdx  = atomBatchMap[idx1[idx]];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void torsionGradKernel(const int      numTorsions,
                                  const int*     idx1,
                                  const int*     idx2,
                                  const int*     idx3,
                                  const int*     idx4,
                                  const double*  forceConstant,
                                  const uint8_t* order,
                                  const double*  cosTerm,
                                  const double*  pos,
                                  double*        grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTorsions) {
    uffTorsionGrad(pos, idx1[idx], idx2[idx], idx3[idx], idx4[idx], forceConstant[idx], order[idx], cosTerm[idx], grad);
  }
}

__global__ void inversionEnergyKernel(const int     numInversions,
                                      const int*    idx1,
                                      const int*    idx2,
                                      const int*    idx3,
                                      const int*    idx4,
                                      const double* forceConstant,
                                      const double* C0,
                                      const double* C1,
                                      const double* C2,
                                      const double* pos,
                                      double*       energyBuffer,
                                      const int*    energyBufferStarts,
                                      const int*    atomBatchMap,
                                      const int*    termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numInversions) {
    const double energy    = uffInversionEnergy(pos,
                                             idx1[idx],
                                             idx2[idx],
                                             idx3[idx],
                                             idx4[idx],
                                             forceConstant[idx],
                                             C0[idx],
                                             C1[idx],
                                             C2[idx]);
    const int    batchIdx  = atomBatchMap[idx1[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void inversionGradKernel(const int     numInversions,
                                    const int*    idx1,
                                    const int*    idx2,
                                    const int*    idx3,
                                    const int*    idx4,
                                    const double* forceConstant,
                                    const double* C1,
                                    const double* C2,
                                    const double* pos,
                                    double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numInversions) {
    uffInversionGrad(pos, idx1[idx], idx2[idx], idx3[idx], idx4[idx], forceConstant[idx], C1[idx], C2[idx], grad);
  }
}

__global__ void vdwEnergyKernel(const int     numVdws,
                                const int*    idx1,
                                const int*    idx2,
                                const double* x_ij,
                                const double* wellDepth,
                                const double* threshold,
                                const double* pos,
                                double*       energyBuffer,
                                const int*    energyBufferStarts,
                                const int*    atomBatchMap,
                                const int*    termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numVdws) {
    const double energy    = uffVdwEnergy(pos, idx1[idx], idx2[idx], x_ij[idx], wellDepth[idx], threshold[idx]);
    const int    batchIdx  = atomBatchMap[idx1[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void vdwGradKernel(const int     numVdws,
                              const int*    idx1,
                              const int*    idx2,
                              const double* x_ij,
                              const double* wellDepth,
                              const double* threshold,
                              const double* pos,
                              double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numVdws) {
    uffVdwGrad(pos, idx1[idx], idx2[idx], x_ij[idx], wellDepth[idx], threshold[idx], grad);
  }
}

constexpr int blockSizePerMol = 128;

template <bool HasConstraints, typename real, typename reduceT, typename Terms, typename storageT>
__global__ void combinedEnergiesKernel(const Terms*                                  terms,
                                       const nvMolKit::UFF::BatchedIndicesDevicePtr* systemIndices,
                                       const storageT*                               coords,
                                       double*                                       energies,
                                       const uint8_t*                                activeSystemMask) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;

  if (activeSystemMask != nullptr && activeSystemMask[molIdx] == 0) {
    if (tid == 0)
      energies[molIdx] = 0.0;
    return;
  }

  const int       atomStart = systemIndices->atomStarts[molIdx];
  const storageT* molCoords = coords + atomStart * 3;
  real            threadEnergy;
  if constexpr (std::is_same_v<real, float>)
    threadEnergy =
      nvMolKit::UFF::fp32::molEnergy<blockSizePerMol, HasConstraints>(*terms, *systemIndices, molCoords, molIdx, tid);
  else
    threadEnergy =
      nvMolKit::UFF::fp64::molEnergy<blockSizePerMol, HasConstraints>(*terms, *systemIndices, molCoords, molIdx, tid);

  using BlockReduce = cub::BlockReduce<reduceT, blockSizePerMol>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  const reduceT blockEnergy = BlockReduce(tempStorage).Sum(static_cast<reduceT>(threadEnergy));

  if (tid == 0) {
    energies[molIdx] = blockEnergy;
  }
}

template <bool HasConstraints, typename real, typename Terms, typename coordinateT, typename storageT>
__global__ void combinedGradKernel(const Terms*                                  terms,
                                   const nvMolKit::UFF::BatchedIndicesDevicePtr* systemIndices,
                                   const coordinateT*                            coords,
                                   storageT*                                     grad,
                                   const uint8_t*                                activeSystemMask) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;

  const int atomStart = systemIndices->atomStarts[molIdx];
  const int atomEnd   = systemIndices->atomStarts[molIdx + 1];
  const int numAtoms  = atomEnd - atomStart;

  if (activeSystemMask != nullptr && activeSystemMask[molIdx] == 0) {
    for (int i = tid; i < numAtoms * 3; i += blockSizePerMol)
      grad[atomStart * 3 + i] = 0.0;
    return;
  }

  constexpr int       maxAtomSize = 256;
  __shared__ storageT accumGrad[maxAtomSize * 3];

  const bool useSharedMem = numAtoms <= maxAtomSize;
  storageT*  molGradBase  = useSharedMem ? accumGrad : grad + atomStart * 3;

  for (int i = tid; i < numAtoms * 3; i += blockSizePerMol) {
    molGradBase[i] = 0.0;
  }
  __syncthreads();

  const coordinateT* molCoords = coords + atomStart * 3;
  if constexpr (std::is_same_v<real, float>)
    nvMolKit::UFF::fp32::molGrad<blockSizePerMol, HasConstraints>(*terms,
                                                                  *systemIndices,
                                                                  molCoords,
                                                                  molGradBase,
                                                                  molIdx,
                                                                  tid);
  else
    nvMolKit::UFF::fp64::molGrad<blockSizePerMol, HasConstraints>(*terms,
                                                                  *systemIndices,
                                                                  molCoords,
                                                                  molGradBase,
                                                                  molIdx,
                                                                  tid);
  __syncthreads();

  if (useSharedMem) {
    storageT* globalGrad = grad + atomStart * 3;
    for (int i = tid; i < numAtoms * 3; i += blockSizePerMol) {
      globalGrad[i] = molGradBase[i];
    }
  }
}

}  // namespace

namespace nvMolKit {
namespace UFF {

cudaError_t launchBondStretchEnergyKernel(int           numBonds,
                                          const int*    idx1,
                                          const int*    idx2,
                                          const double* restLen,
                                          const double* forceConstant,
                                          const double* pos,
                                          double*       energyBuffer,
                                          const int*    energyBufferStarts,
                                          const int*    atomBatchMap,
                                          const int*    termBatchStarts,
                                          cudaStream_t  stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numBonds + blockSize - 1) / blockSize;
  bondStretchEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numBonds,
                                                               idx1,
                                                               idx2,
                                                               restLen,
                                                               forceConstant,
                                                               pos,
                                                               energyBuffer,
                                                               energyBufferStarts,
                                                               atomBatchMap,
                                                               termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchBondStretchGradientKernel(int           numBonds,
                                            const int*    idx1,
                                            const int*    idx2,
                                            const double* restLen,
                                            const double* forceConstant,
                                            const double* pos,
                                            double*       grad,
                                            cudaStream_t  stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numBonds + blockSize - 1) / blockSize;
  bondStretchGradKernel<<<numBlocks, blockSize, 0, stream>>>(numBonds, idx1, idx2, restLen, forceConstant, pos, grad);
  return cudaGetLastError();
}

cudaError_t launchAngleBendEnergyKernel(int            numAngles,
                                        const int*     idx1,
                                        const int*     idx2,
                                        const int*     idx3,
                                        const double*  theta0,
                                        const double*  forceConstant,
                                        const uint8_t* order,
                                        const double*  C0,
                                        const double*  C1,
                                        const double*  C2,
                                        const double*  pos,
                                        double*        energyBuffer,
                                        const int*     energyBufferStarts,
                                        const int*     atomBatchMap,
                                        const int*     termBatchStarts,
                                        cudaStream_t   stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngles + blockSize - 1) / blockSize;
  angleBendEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numAngles,
                                                             idx1,
                                                             idx2,
                                                             idx3,
                                                             theta0,
                                                             forceConstant,
                                                             order,
                                                             C0,
                                                             C1,
                                                             C2,
                                                             pos,
                                                             energyBuffer,
                                                             energyBufferStarts,
                                                             atomBatchMap,
                                                             termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchAngleBendGradientKernel(int            numAngles,
                                          const int*     idx1,
                                          const int*     idx2,
                                          const int*     idx3,
                                          const double*  theta0,
                                          const double*  forceConstant,
                                          const uint8_t* order,
                                          const double*  C0,
                                          const double*  C1,
                                          const double*  C2,
                                          const double*  pos,
                                          double*        grad,
                                          cudaStream_t   stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngles + blockSize - 1) / blockSize;
  angleBendGradKernel<<<numBlocks, blockSize, 0, stream>>>(numAngles,
                                                           idx1,
                                                           idx2,
                                                           idx3,
                                                           theta0,
                                                           forceConstant,
                                                           order,
                                                           C0,
                                                           C1,
                                                           C2,
                                                           pos,
                                                           grad);
  return cudaGetLastError();
}

cudaError_t launchTorsionEnergyKernel(int            numTorsions,
                                      const int*     idx1,
                                      const int*     idx2,
                                      const int*     idx3,
                                      const int*     idx4,
                                      const double*  forceConstant,
                                      const uint8_t* order,
                                      const double*  cosTerm,
                                      const double*  pos,
                                      double*        energyBuffer,
                                      const int*     energyBufferStarts,
                                      const int*     atomBatchMap,
                                      const int*     termBatchStarts,
                                      cudaStream_t   stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numTorsions + blockSize - 1) / blockSize;
  torsionEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numTorsions,
                                                           idx1,
                                                           idx2,
                                                           idx3,
                                                           idx4,
                                                           forceConstant,
                                                           order,
                                                           cosTerm,
                                                           pos,
                                                           energyBuffer,
                                                           energyBufferStarts,
                                                           atomBatchMap,
                                                           termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchTorsionGradientKernel(int            numTorsions,
                                        const int*     idx1,
                                        const int*     idx2,
                                        const int*     idx3,
                                        const int*     idx4,
                                        const double*  forceConstant,
                                        const uint8_t* order,
                                        const double*  cosTerm,
                                        const double*  pos,
                                        double*        grad,
                                        cudaStream_t   stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numTorsions + blockSize - 1) / blockSize;
  torsionGradKernel<<<numBlocks, blockSize, 0, stream>>>(numTorsions,
                                                         idx1,
                                                         idx2,
                                                         idx3,
                                                         idx4,
                                                         forceConstant,
                                                         order,
                                                         cosTerm,
                                                         pos,
                                                         grad);
  return cudaGetLastError();
}

cudaError_t launchInversionEnergyKernel(int           numInversions,
                                        const int*    idx1,
                                        const int*    idx2,
                                        const int*    idx3,
                                        const int*    idx4,
                                        const double* forceConstant,
                                        const double* C0,
                                        const double* C1,
                                        const double* C2,
                                        const double* pos,
                                        double*       energyBuffer,
                                        const int*    energyBufferStarts,
                                        const int*    atomBatchMap,
                                        const int*    termBatchStarts,
                                        cudaStream_t  stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numInversions + blockSize - 1) / blockSize;
  inversionEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numInversions,
                                                             idx1,
                                                             idx2,
                                                             idx3,
                                                             idx4,
                                                             forceConstant,
                                                             C0,
                                                             C1,
                                                             C2,
                                                             pos,
                                                             energyBuffer,
                                                             energyBufferStarts,
                                                             atomBatchMap,
                                                             termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchInversionGradientKernel(int           numInversions,
                                          const int*    idx1,
                                          const int*    idx2,
                                          const int*    idx3,
                                          const int*    idx4,
                                          const double* forceConstant,
                                          const double* C0,
                                          const double* C1,
                                          const double* C2,
                                          const double* pos,
                                          double*       grad,
                                          cudaStream_t  stream) {
  (void)C0;
  constexpr int blockSize = 256;
  const int     numBlocks = (numInversions + blockSize - 1) / blockSize;
  inversionGradKernel<<<numBlocks, blockSize, 0, stream>>>(numInversions,
                                                           idx1,
                                                           idx2,
                                                           idx3,
                                                           idx4,
                                                           forceConstant,
                                                           C1,
                                                           C2,
                                                           pos,
                                                           grad);
  return cudaGetLastError();
}

cudaError_t launchVdwEnergyKernel(int           numVdws,
                                  const int*    idx1,
                                  const int*    idx2,
                                  const double* x_ij,
                                  const double* wellDepth,
                                  const double* threshold,
                                  const double* pos,
                                  double*       energyBuffer,
                                  const int*    energyBufferStarts,
                                  const int*    atomBatchMap,
                                  const int*    termBatchStarts,
                                  cudaStream_t  stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numVdws + blockSize - 1) / blockSize;
  vdwEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numVdws,
                                                       idx1,
                                                       idx2,
                                                       x_ij,
                                                       wellDepth,
                                                       threshold,
                                                       pos,
                                                       energyBuffer,
                                                       energyBufferStarts,
                                                       atomBatchMap,
                                                       termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchVdwGradientKernel(int           numVdws,
                                    const int*    idx1,
                                    const int*    idx2,
                                    const double* x_ij,
                                    const double* wellDepth,
                                    const double* threshold,
                                    const double* pos,
                                    double*       grad,
                                    cudaStream_t  stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numVdws + blockSize - 1) / blockSize;
  vdwGradKernel<<<numBlocks, blockSize, 0, stream>>>(numVdws, idx1, idx2, x_ij, wellDepth, threshold, pos, grad);
  return cudaGetLastError();
}

template <typename Terms, typename CoordinateScalar>
cudaError_t launchBlockPerMolEnergyKernelImpl(int                            numMols,
                                              const Terms&                   terms,
                                              const BatchedIndicesDevicePtr& systemIndices,
                                              const CoordinateScalar*        coords,
                                              double*                        energies,
                                              bool                           hasConstraints,
                                              bool                           computeInFloat,
                                              bool                           reduceInFloat,
                                              cudaStream_t                   stream,
                                              const uint8_t*                 activeSystemMask) {
  const AsyncDevicePtr<Terms>                   devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndicesDevicePtr> devSysIdx(systemIndices, stream);
#define NVMOLKIT_LAUNCH_UFF_ENERGY(HasConstraints, real, reduceT) \
  combinedEnergiesKernel<HasConstraints, real, reduceT>           \
    <<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(), devSysIdx.data(), coords, energies, activeSystemMask)
  if (hasConstraints) {
    if (computeInFloat && reduceInFloat)
      NVMOLKIT_LAUNCH_UFF_ENERGY(true, float, float);
    else if (computeInFloat)
      NVMOLKIT_LAUNCH_UFF_ENERGY(true, float, double);
    else if (reduceInFloat)
      NVMOLKIT_LAUNCH_UFF_ENERGY(true, double, float);
    else
      NVMOLKIT_LAUNCH_UFF_ENERGY(true, double, double);
  } else {
    if (computeInFloat && reduceInFloat)
      NVMOLKIT_LAUNCH_UFF_ENERGY(false, float, float);
    else if (computeInFloat)
      NVMOLKIT_LAUNCH_UFF_ENERGY(false, float, double);
    else if (reduceInFloat)
      NVMOLKIT_LAUNCH_UFF_ENERGY(false, double, float);
    else
      NVMOLKIT_LAUNCH_UFF_ENERGY(false, double, double);
  }
#undef NVMOLKIT_LAUNCH_UFF_ENERGY
  return cudaGetLastError();
}

template <typename Terms, typename coordinateT, typename storageT>
cudaError_t launchBlockPerMolGradKernelImpl(int                            numMols,
                                            const Terms&                   terms,
                                            const BatchedIndicesDevicePtr& systemIndices,
                                            const coordinateT*             coords,
                                            storageT*                      grad,
                                            bool                           hasConstraints,
                                            bool                           computeInFloat,
                                            cudaStream_t                   stream,
                                            const uint8_t*                 activeSystemMask) {
  const AsyncDevicePtr<Terms>                   devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndicesDevicePtr> devSysIdx(systemIndices, stream);
  if (hasConstraints) {
    if (computeInFloat)
      combinedGradKernel<true, float>
        <<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(), devSysIdx.data(), coords, grad, activeSystemMask);
    else
      combinedGradKernel<true, double>
        <<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(), devSysIdx.data(), coords, grad, activeSystemMask);
  } else {
    if (computeInFloat)
      combinedGradKernel<false, float>
        <<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(), devSysIdx.data(), coords, grad, activeSystemMask);
    else
      combinedGradKernel<false, double>
        <<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(), devSysIdx.data(), coords, grad, activeSystemMask);
  }
  return cudaGetLastError();
}

#define NVMOLKIT_DEFINE_UFF_ENERGY_LAUNCHER(TermsType, CoordinateType)                     \
  cudaError_t launchBlockPerMolEnergyKernel(int                            numMols,        \
                                            const TermsType&               terms,          \
                                            const BatchedIndicesDevicePtr& indices,        \
                                            const CoordinateType*          coords,         \
                                            double*                        energies,       \
                                            bool                           hasConstraints, \
                                            bool                           computeInFloat, \
                                            bool                           reduceInFloat,  \
                                            cudaStream_t                   stream,         \
                                            const uint8_t*                 activeSystemMask) {             \
    return launchBlockPerMolEnergyKernelImpl(numMols,                                      \
                                             terms,                                        \
                                             indices,                                      \
                                             coords,                                       \
                                             energies,                                     \
                                             hasConstraints,                               \
                                             computeInFloat,                               \
                                             reduceInFloat,                                \
                                             stream,                                       \
                                             activeSystemMask);                            \
  }

#define NVMOLKIT_DEFINE_UFF_GRAD_LAUNCHER(TermsType, CoordinateType, StorageType)        \
  cudaError_t launchBlockPerMolGradKernel(int                            numMols,        \
                                          const TermsType&               terms,          \
                                          const BatchedIndicesDevicePtr& indices,        \
                                          const CoordinateType*          coords,         \
                                          StorageType*                   grad,           \
                                          bool                           hasConstraints, \
                                          bool                           computeInFloat, \
                                          cudaStream_t                   stream,         \
                                          const uint8_t*                 activeSystemMask) {             \
    return launchBlockPerMolGradKernelImpl(numMols,                                      \
                                           terms,                                        \
                                           indices,                                      \
                                           coords,                                       \
                                           grad,                                         \
                                           hasConstraints,                               \
                                           computeInFloat,                               \
                                           stream,                                       \
                                           activeSystemMask);                            \
  }

#define NVMOLKIT_DEFINE_UFF_LAUNCHERS_FOR_TERMS(TermsType)     \
  NVMOLKIT_DEFINE_UFF_ENERGY_LAUNCHER(TermsType, double)       \
  NVMOLKIT_DEFINE_UFF_ENERGY_LAUNCHER(TermsType, float)        \
  NVMOLKIT_DEFINE_UFF_GRAD_LAUNCHER(TermsType, double, double) \
  NVMOLKIT_DEFINE_UFF_GRAD_LAUNCHER(TermsType, double, float)  \
  NVMOLKIT_DEFINE_UFF_GRAD_LAUNCHER(TermsType, float, double)  \
  NVMOLKIT_DEFINE_UFF_GRAD_LAUNCHER(TermsType, float, float)

NVMOLKIT_DEFINE_UFF_LAUNCHERS_FOR_TERMS(EnergyForceContribsDevicePtr)
NVMOLKIT_DEFINE_UFF_LAUNCHERS_FOR_TERMS(EnergyForceContribsDevicePtrF32)
#undef NVMOLKIT_DEFINE_UFF_LAUNCHERS_FOR_TERMS
#undef NVMOLKIT_DEFINE_UFF_GRAD_LAUNCHER
#undef NVMOLKIT_DEFINE_UFF_ENERGY_LAUNCHER

}  // namespace UFF
}  // namespace nvMolKit
