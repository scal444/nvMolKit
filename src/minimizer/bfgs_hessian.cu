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

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cub/cub.cuh>
#include <vector>

#include "src/minimizer/bfgs_hessian.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"
#include "src/utils/host_vector.h"
namespace cg = cooperative_groups;

namespace nvMolKit {

namespace {

constexpr int warpSize  = 32;
constexpr int blockSize = 512;
constexpr int numWarp   = blockSize / warpSize;
constexpr int maxAtom   = 256;

__device__ __forceinline__ void computeBfgsSums(const double*                    dGrad,
                                                const double*                    xi,
                                                const double*                    hessDGrad,
                                                double*                          facShared,
                                                double*                          faeShared,
                                                double*                          sumDGradShared,
                                                double*                          sumXiShared,
                                                cg::thread_block_tile<warpSize>& warp,
                                                int                              warpIdx,
                                                int                              laneIdx,
                                                int                              dim) {
  double sumTerm = 0.0;
  if (warpIdx == 0) {
    for (int i = laneIdx; i < dim; i += warpSize) {
      sumTerm += dGrad[i] * xi[i];
    }
    cg::reduce_store_async(warp, &facShared[0], sumTerm, cg::plus<double>{});
  } else if (warpIdx == 1) {
    for (int i = laneIdx; i < dim; i += warpSize) {
      sumTerm += dGrad[i] * hessDGrad[i];
    }
    cg::reduce_store_async(warp, &faeShared[0], sumTerm, cg::plus<double>{});
  } else if (warpIdx == 2) {
    for (int i = laneIdx; i < dim; i += warpSize) {
      sumTerm += dGrad[i] * dGrad[i];
    }
    cg::reduce_store_async(warp, &sumDGradShared[0], sumTerm, cg::plus<double>{});
  } else if (warpIdx == 3) {
    for (int i = laneIdx; i < dim; i += warpSize) {
      sumTerm += xi[i] * xi[i];
    }
    cg::reduce_store_async(warp, &sumXiShared[0], sumTerm, cg::plus<double>{});
  }
}

__device__ __forceinline__ void computeUpdateFlag(int     idxWithinSystem,
                                                  double* facShared,
                                                  double* faeShared,
                                                  double* fadShared,
                                                  double* alphaShared,
                                                  double* sumDGradShared,
                                                  double* sumXiShared,
                                                  bool*   needUpdateInverseHessian) {
  if (idxWithinSystem == 0) {
    constexpr double EPS                  = 3e-8;
    const double     sumXi                = sumXiShared[0];
    const double     sumDGrad             = sumDGradShared[0];
    const double     fac                  = facShared[0];
    const double     fae                  = faeShared[0];
    bool             updateInverseHessian = fac > sqrt(EPS * sumDGrad * sumXi);
    if (updateInverseHessian) {
      facShared[0] = 1.0 / fac;
      fadShared[0] = 1.0 / fae;
      if (alphaShared != nullptr) {
        alphaShared[0] = facShared[0] + fae * facShared[0] * facShared[0];
      }
    }
    needUpdateInverseHessian[0] = updateInverseHessian;
  }
}

template <int groupWidth> __device__ __forceinline__ double reduceGroup(double value, unsigned int groupMask) {
#pragma unroll
  for (int delta = groupWidth / 2; delta > 0; delta /= 2) {
    value += __shfl_down_sync(groupMask, value, delta, groupWidth);
  }
  return value;
}

__device__ __forceinline__ double reduceWarp(double value) {
#pragma unroll
  for (int delta = warpSize / 2; delta > 0; delta /= 2) {
    value += __shfl_down_sync(0xffffffff, value, delta);
  }
  return value;
}

// Shared-memory optimized kernel used when all systems have <= maxAtom atoms
template <int dataDim, int blockThreads, int groupWidth>
__global__ void updateInverseHessianBFGSBatchKernelShared(const int16_t* statuses,
                                                          const int*     atomStarts,
                                                          const int*     hessianStarts,
                                                          double*        invHessians,
                                                          double*        dGrads,
                                                          double*        xis,
                                                          double*        hessDGrads,
                                                          const double*  grads,
                                                          const int*     activeSystemIndices) {
  constexpr int rowGroups = blockThreads / groupWidth;

  __shared__ double scalars[5];
  __shared__ bool   needUpdateInverseHessian;

  __shared__ __align__(16) double cachedDGrads[dataDim * maxAtom];
  __shared__ __align__(16) double cachedHessDGrads[dataDim * maxAtom];
  __shared__ __align__(16) double cachedHessGrads[dataDim * maxAtom];
  __shared__ __align__(16) double cachedXis[dataDim * maxAtom];
  __shared__ __align__(16) double cachedGrads[dataDim * maxAtom];

  const int sysIdx = activeSystemIndices[gridDim.x - 1 - blockIdx.x];
  if (statuses != nullptr && statuses[sysIdx] == 0) {
    return;
  }

  const int          idxWithinSystem = threadIdx.x;
  const int          laneIdx         = idxWithinSystem & (groupWidth - 1);
  const int          groupIdx        = idxWithinSystem / groupWidth;
  const unsigned int groupMask = ((1u << groupWidth) - 1u) << ((idxWithinSystem & (warpSize - 1)) & ~(groupWidth - 1));

  // Get local pointers. Note that inverse hessian is dim indexed but the atomStart-based ones are * dataDim
  const int           atomOffset      = atomStarts[sysIdx];
  const int           dim             = dataDim * (atomStarts[sysIdx + 1] - atomOffset);
  double* const       invHessianLocal = &invHessians[hessianStarts[sysIdx]];
  const int           absAtomOffset   = atomOffset * dataDim;
  double* const       localDGrad      = &dGrads[absAtomOffset];
  double* const       localHessDGrad  = &hessDGrads[absAtomOffset];
  double* const       localXi         = &xis[absAtomOffset];
  const double* const localGrad       = &grads[absAtomOffset];

  // Load dGrads, Xi, grads into shared memory
  for (int i = idxWithinSystem; i < dim; i += blockThreads) {
    cachedDGrads[i] = localDGrad[i];
    cachedXis[i]    = localXi[i];
    cachedGrads[i]  = localGrad[i];
  }

  __syncthreads();

  // Compute H*dGrad and H*grad together while each Hessian value is resident.
  for (int row = dim - 1 - groupIdx; row >= 0; row -= rowGroups) {
    double hessDGrad = 0.0;
    double hessGrad  = 0.0;

    if constexpr (dataDim == 4) {
      const double2* hessianPairs =
        reinterpret_cast<const double2*>(invHessianLocal + static_cast<long long>(row) * dim);
      const double2* dGradPairs = reinterpret_cast<const double2*>(cachedDGrads);
      const double2* gradPairs  = reinterpret_cast<const double2*>(cachedGrads);
      for (int colPair = laneIdx; colPair < dim / 2; colPair += groupWidth) {
        const double2 hessianValue = hessianPairs[colPair];
        const double2 dGrad        = dGradPairs[colPair];
        const double2 grad         = gradPairs[colPair];
        hessDGrad                  = fma(hessianValue.x, dGrad.x, hessDGrad);
        hessDGrad                  = fma(hessianValue.y, dGrad.y, hessDGrad);
        hessGrad                   = fma(hessianValue.x, grad.x, hessGrad);
        hessGrad                   = fma(hessianValue.y, grad.y, hessGrad);
      }
    } else {
      for (int col = laneIdx; col < dim; col += groupWidth) {
        const double hessianValue = invHessianLocal[row * dim + col];
        hessDGrad                 = fma(hessianValue, cachedDGrads[col], hessDGrad);
        hessGrad                  = fma(hessianValue, cachedGrads[col], hessGrad);
      }
    }

    hessDGrad = reduceGroup<groupWidth>(hessDGrad, groupMask);
    hessGrad  = reduceGroup<groupWidth>(hessGrad, groupMask);
    if (laneIdx == 0) {
      cachedHessDGrads[row] = hessDGrad;
      cachedHessGrads[row]  = hessGrad;
    }
  }

  __syncthreads();

  if (idxWithinSystem < warpSize) {
    double fac           = 0.0;
    double fae           = 0.0;
    double sumDGrad      = 0.0;
    double sumXi         = 0.0;
    double xiGrad        = 0.0;
    double hessDGradGrad = 0.0;
    for (int i = idxWithinSystem; i < dim; i += warpSize) {
      const double dGrad     = cachedDGrads[i];
      const double xi        = cachedXis[i];
      const double hessDGrad = cachedHessDGrads[i];
      const double grad      = cachedGrads[i];
      fac                    = fma(dGrad, xi, fac);
      fae                    = fma(dGrad, hessDGrad, fae);
      sumDGrad               = fma(dGrad, dGrad, sumDGrad);
      sumXi                  = fma(xi, xi, sumXi);
      xiGrad                 = fma(xi, grad, xiGrad);
      hessDGradGrad          = fma(hessDGrad, grad, hessDGradGrad);
    }
    fac           = reduceWarp(fac);
    fae           = reduceWarp(fae);
    sumDGrad      = reduceWarp(sumDGrad);
    sumXi         = reduceWarp(sumXi);
    xiGrad        = reduceWarp(xiGrad);
    hessDGradGrad = reduceWarp(hessDGradGrad);
    if (idxWithinSystem == 0) {
      constexpr double EPS     = 3e-8;
      needUpdateInverseHessian = fac > sqrt(EPS * sumDGrad * sumXi);
      if (needUpdateInverseHessian) {
        const double inverseFac = 1.0 / fac;
        scalars[0]              = inverseFac;
        scalars[1]              = 1.0 / fae;
        scalars[2]              = inverseFac + fae * inverseFac * inverseFac;
        scalars[3]              = xiGrad;
        scalars[4]              = hessDGradGrad;
      }
    }
  }

  __syncthreads();

  for (int i = idxWithinSystem; i < dim; i += blockThreads) {
    localHessDGrad[i] = cachedHessDGrads[i];
  }

  if (needUpdateInverseHessian) {
    // Update dGrads, Inverse Hessian, and Xi
    const double inverseFac    = scalars[0];
    const double inverseFae    = scalars[1];
    const double alpha         = scalars[2];
    const double xiGrad        = scalars[3];
    const double hessDGradGrad = scalars[4];

    for (int i = idxWithinSystem; i < dim; i += blockThreads) {
      cachedDGrads[i] = inverseFac * cachedXis[i] - inverseFae * cachedHessDGrads[i];
      localXi[i]      = -cachedHessGrads[i] - alpha * cachedXis[i] * xiGrad +
                   inverseFac * (cachedXis[i] * hessDGradGrad + cachedHessDGrads[i] * xiGrad);
    }

    __syncthreads();

    for (int i = idxWithinSystem; i < dim; i += blockThreads) {
      localDGrad[i] = cachedDGrads[i];
    }

    if constexpr (dataDim == 4) {
      const int      rowWidthPairs  = dim / 2;
      const int      totalPairs     = dim * rowWidthPairs;
      double2*       hessianPairs   = reinterpret_cast<double2*>(invHessianLocal);
      const double2* xiPairs        = reinterpret_cast<const double2*>(cachedXis);
      const double2* hessDGradPairs = reinterpret_cast<const double2*>(cachedHessDGrads);
      for (int pair = idxWithinSystem; pair < totalPairs; pair += blockThreads) {
        const int     row          = pair / rowWidthPairs;
        const int     colPair      = pair - row * rowWidthPairs;
        const double  alphaXi      = alpha * cachedXis[row];
        const double  facHessDGrad = inverseFac * cachedHessDGrads[row];
        const double  facXi        = inverseFac * cachedXis[row];
        const double2 xi           = xiPairs[colPair];
        const double2 hessDGrad    = hessDGradPairs[colPair];
        double2       hessian      = hessianPairs[pair];
        hessian.x += alphaXi * xi.x - facHessDGrad * xi.x - facXi * hessDGrad.x;
        hessian.y += alphaXi * xi.y - facHessDGrad * xi.y - facXi * hessDGrad.y;
        hessianPairs[pair] = hessian;
      }
    } else {
      const int totalElements = dim * dim;
      for (int element = idxWithinSystem; element < totalElements; element += blockThreads) {
        const int row = element / dim;
        const int col = element - row * dim;
        invHessianLocal[element] +=
          alpha * cachedXis[row] * cachedXis[col] -
          inverseFac * (cachedHessDGrads[row] * cachedXis[col] + cachedXis[row] * cachedHessDGrads[col]);
      }
    }
  } else {
    // Update Xi Only
    for (int i = idxWithinSystem; i < dim; i += blockThreads) {
      localXi[i] = -cachedHessGrads[i];
    }
  }
}

// Global-memory variant that avoids fixed-size shared arrays; safe for large molecules
template <int dataDim>
__global__ void updateInverseHessianBFGSBatchKernelGlobal(const int16_t* statuses,
                                                          const int*     atomStarts,
                                                          const int*     hessianStarts,
                                                          double*        invHessians,
                                                          double*        dGrads,
                                                          double*        xis,
                                                          double*        hessDGrads,
                                                          const double*  grads,
                                                          const int*     activeSystemIndices) {
  __shared__ double facShared[1];
  __shared__ double faeShared[1];
  __shared__ double fadShared[1];
  __shared__ double sumDGradShared[1];
  __shared__ double sumXiShared[1];
  __shared__ bool   needUpdateInverseHessian[1];

  const int sysIdx = activeSystemIndices[blockIdx.x];
  if (statuses != nullptr && statuses[sysIdx] == 0) {
    return;
  }

  cg::thread_block                block           = cg::this_thread_block();
  cg::thread_block_tile<warpSize> warp            = cg::tiled_partition<warpSize>(block);
  const int                       idxWithinSystem = threadIdx.x;
  const int                       warpIdx         = idxWithinSystem / warpSize;
  const int                       laneIdx         = idxWithinSystem % warpSize;

  // Get local pointers. Note that inverse hessian is dim indexed but the atomStart-based ones are * dataDim
  const int           atomOffset      = atomStarts[sysIdx];
  const int           dim             = dataDim * (atomStarts[sysIdx + 1] - atomOffset);
  double* const       invHessianLocal = &invHessians[hessianStarts[sysIdx]];
  const int           absAtomOffset   = atomOffset * dataDim;
  double* const       localDGrad      = &dGrads[absAtomOffset];
  double* const       localHessDGrad  = &hessDGrads[absAtomOffset];
  double* const       localXi         = &xis[absAtomOffset];
  const double* const localGrad       = &grads[absAtomOffset];

  // Compute hessDGrads directly into global memory
  for (int row = warpIdx; row < dim; row += numWarp) {
    double dotProduct = 0.0;

    for (int col = laneIdx; col < dim; col += warpSize) {
      dotProduct += invHessianLocal[row * dim + col] * localDGrad[col];
    }

    cg::reduce_store_async(warp, &localHessDGrad[row], dotProduct, cg::plus<double>{});
  }

  block.sync();

  // Compute BFGS sums using global memory
  computeBfgsSums(localDGrad,
                  localXi,
                  localHessDGrad,
                  facShared,
                  faeShared,
                  sumDGradShared,
                  sumXiShared,
                  warp,
                  warpIdx,
                  laneIdx,
                  dim);

  block.sync();

  computeUpdateFlag(idxWithinSystem,
                    facShared,
                    faeShared,
                    fadShared,
                    nullptr,
                    sumDGradShared,
                    sumXiShared,
                    needUpdateInverseHessian);

  block.sync();

  if (needUpdateInverseHessian[0]) {
    const double fac = facShared[0];
    const double fae = faeShared[0];
    const double fad = fadShared[0];

    // Update dGrad with snapshot values; do not touch Xi yet.
    for (int i = idxWithinSystem; i < dim; i += blockSize) {
      const double dval = fac * localXi[i] - fad * localHessDGrad[i];
      localDGrad[i]     = dval;
    }

    block.sync();

    // Update inverse Hessian using OLD Xi snapshot (still in localXi)
    for (int row = warpIdx; row < dim; row += numWarp) {
      const double pxi  = fac * localXi[row];
      const double hdgi = fad * localHessDGrad[row];
      const double dgi  = fae * localDGrad[row];

      for (int col = laneIdx; col < dim; col += warpSize) {
        const double pxj       = localXi[col];
        const double hdgj      = localHessDGrad[col];
        const double dgj       = localDGrad[col];
        double       new_value = pxi * pxj - hdgi * hdgj + dgi * dgj;
        new_value += invHessianLocal[row * dim + col];
        invHessianLocal[row * dim + col] = new_value;
      }
    }

    block.sync();

    // Now compute Xi = -H_new * grad using the updated inverse Hessian
    for (int row = warpIdx; row < dim; row += numWarp) {
      double dotProduct = 0.0;
      for (int col = laneIdx; col < dim; col += warpSize) {
        dotProduct -= invHessianLocal[row * dim + col] * localGrad[col];
      }
      cg::reduce_store_async(warp, &localXi[row], dotProduct, cg::plus<double>{});
    }
  } else {
    // Xi update only
    for (int row = warpIdx; row < dim; row += numWarp) {
      double dotProduct = 0.0;

      for (int col = laneIdx; col < dim; col += warpSize) {
        dotProduct -= invHessianLocal[row * dim + col] * localGrad[col];
      }

      cg::reduce_store_async(warp, &localXi[row], dotProduct, cg::plus<double>{});
    }
  }
}

template <int dataDim>
void launchSharedBfgsKernel(int            numActiveSystems,
                            long long      averageHessianElements,
                            const int16_t* statuses,
                            const int*     atomStarts,
                            const int*     hessianStarts,
                            double*        invHessians,
                            double*        dGrads,
                            double*        xis,
                            double*        hessDGrads,
                            const double*  grads,
                            const int*     activeSystemIndices,
                            cudaStream_t   stream) {
#define LAUNCH_SHARED_BFGS(blockThreads, groupWidth)                           \
  updateInverseHessianBFGSBatchKernelShared<dataDim, blockThreads, groupWidth> \
    <<<numActiveSystems, blockThreads, 0, stream>>>(statuses,                  \
                                                    atomStarts,                \
                                                    hessianStarts,             \
                                                    invHessians,               \
                                                    dGrads,                    \
                                                    xis,                       \
                                                    hessDGrads,                \
                                                    grads,                     \
                                                    activeSystemIndices)

  if (numActiveSystems <= 512) {
    LAUNCH_SHARED_BFGS(1024, 16);
  } else if (averageHessianElements <= 12000) {
    LAUNCH_SHARED_BFGS(128, 4);
  } else {
    LAUNCH_SHARED_BFGS(512, 8);
  }

#undef LAUNCH_SHARED_BFGS
}

}  // namespace

void updateInverseHessianBFGSBatch(int            numActiveSystems,
                                   long long      numHessianElements,
                                   const int16_t* statuses,
                                   const int*     hessianStarts,
                                   const int*     atomStarts,
                                   double*        invHessians,
                                   double*        dGrads,
                                   double*        xis,
                                   double*        hessDGrads,
                                   const double*  grads,
                                   int            dataDim,
                                   bool           hasLargeMolecule,
                                   const int*     activeSystemIndices,
                                   cudaStream_t   stream) {
  const long long averageHessianElements = numHessianElements / numActiveSystems;

  if (dataDim == 3) {
    if (hasLargeMolecule) {
      updateInverseHessianBFGSBatchKernelGlobal<3><<<numActiveSystems, blockSize, 0, stream>>>(statuses,
                                                                                               atomStarts,
                                                                                               hessianStarts,
                                                                                               invHessians,
                                                                                               dGrads,
                                                                                               xis,
                                                                                               hessDGrads,
                                                                                               grads,
                                                                                               activeSystemIndices);
    } else {
      launchSharedBfgsKernel<3>(numActiveSystems,
                                averageHessianElements,
                                statuses,
                                atomStarts,
                                hessianStarts,
                                invHessians,
                                dGrads,
                                xis,
                                hessDGrads,
                                grads,
                                activeSystemIndices,
                                stream);
    }
  } else if (dataDim == 4) {
    if (hasLargeMolecule) {
      updateInverseHessianBFGSBatchKernelGlobal<4><<<numActiveSystems, blockSize, 0, stream>>>(statuses,
                                                                                               atomStarts,
                                                                                               hessianStarts,
                                                                                               invHessians,
                                                                                               dGrads,
                                                                                               xis,
                                                                                               hessDGrads,
                                                                                               grads,
                                                                                               activeSystemIndices);
    } else {
      launchSharedBfgsKernel<4>(numActiveSystems,
                                averageHessianElements,
                                statuses,
                                atomStarts,
                                hessianStarts,
                                invHessians,
                                dGrads,
                                xis,
                                hessDGrads,
                                grads,
                                activeSystemIndices,
                                stream);
    }
  } else {
    throw std::runtime_error("Unsupported data dimension: " + std::to_string(dataDim));
  }

  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit
