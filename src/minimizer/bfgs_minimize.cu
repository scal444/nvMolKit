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

#include <cub/cub.cuh>
#include <numeric>

#include "src/forcefields/batched_forcefield.h"
#include "src/forcefields/dist_geom.h"
#include "src/forcefields/dist_geom_kernels.h"
#include "src/forcefields/mmff.h"
#include "src/forcefields/mmff_kernels.h"
#include "src/minimizer/bfgs_hessian.h"
#include "src/minimizer/bfgs_minimize.h"
#include "src/minimizer/bfgs_minimize_permol_kernels.h"
#include "src/utils/cub_helpers.cuh"
#include "src/utils/device_vector.h"
#include "src/utils/nvtx.h"
#include "versions.h"

namespace nvMolKit {
constexpr double FUNCTOL = 1e-4;  //!< Default tolerance for function convergence in the minimizer
constexpr double MOVETOL = 1e-7;  //!< Default tolerance for x changes in the minimizer

namespace {
template <typename real> __device__ __forceinline__ real bfgsSqrt(real value) {
  if constexpr (cuda::std::is_same_v<real, float>)
    return sqrtf(value);
  else
    return sqrt(value);
}
template <typename real> __device__ __forceinline__ real bfgsAbs(real value) {
  if constexpr (cuda::std::is_same_v<real, float>)
    return fabsf(value);
  else
    return fabs(value);
}
BfgsBackend resolveBackend(BfgsBackend backend, const std::vector<int>& atomStartsHost) {
  if (backend != BfgsBackend::HYBRID) {
    return backend;
  }
  for (size_t i = 0; i + 1 < atomStartsHost.size(); ++i) {
    if (atomStartsHost[i + 1] - atomStartsHost[i] > kHybridBackendAtomThreshold) {
      return BfgsBackend::BATCHED;
    }
  }
  return BfgsBackend::PER_MOLECULE;
}

}  // namespace

// TODO - consolidate this to device vector code. We don't want CUDA in the device vector
// header so we'll need to specialize for a few types and instantiate them in the cu file.
template <typename T> __global__ void setAllKernel(const int numElements, T value, T* dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = value;
  }
}

// Specialized version for copying from uint8_t to int16_t
__global__ void copyActiveToStatusKernel(const int numElements, const uint8_t* src, int16_t* dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = static_cast<int16_t>(src[idx]);
  }
}

template <typename T> void setAll(AsyncDeviceVector<T>& vec, const T& value) {
  const int          numElements = vec.size();
  const cudaStream_t stream      = vec.stream();
  if (numElements == 0) {
    return;
  }
  const int blockSize = 128;
  const int numBlocks = (numElements + blockSize - 1) / blockSize;
  setAllKernel<<<numBlocks, blockSize, 0, stream>>>(numElements, value, vec.data());
  cudaCheckError(cudaGetLastError());
}

// Scale direction vector, get slope and test values.
template <typename real, typename reduceT, typename stateT>
__global__ void initializeLineSearchKernel(const int16_t* statuses,
                                           const double*  oldPositions,
                                           const double*  grads,
                                           const int*     atomStarts,
                                           const stateT*  maxSteps,
                                           stateT*        dirs,
                                           stateT*        slopes,
                                           stateT*        lambdaMins,
                                           const int*     activeSystemIndices,
                                           const int      DIM) {
  const int  sysIdx        = activeSystemIndices[blockIdx.x];
  const int  idxInSys      = threadIdx.x;
  const bool isFirstThread = threadIdx.x == 0;

  if (statuses[sysIdx] == 0) {
    return;
  }

  const int     numTerms  = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const double* posStart  = &oldPositions[atomStarts[sysIdx] * DIM];
  const double* gradStart = &grads[atomStarts[sysIdx] * DIM];
  stateT*       dirStart  = &dirs[atomStarts[sysIdx] * DIM];

  using BlockReduce = cub::BlockReduce<reduceT, 128>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ reduceT                           dirSum[1];

  // ---------------------------------
  //  Scale direction vector if needed
  // ---------------------------------
  reduceT sumSquaredLocal = 0;
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    const real dx = static_cast<real>(dirStart[i]);
    sumSquaredLocal += static_cast<reduceT>(dx * dx);
  }
  reduceT blockSum = BlockReduce(tempStorage).Sum(sumSquaredLocal);
  if (isFirstThread) {
    dirSum[0] = bfgsSqrt(blockSum);
  }
  __syncthreads();
  if (dirSum[0] > maxSteps[sysIdx]) {
    for (int i = idxInSys; i < numTerms; i += blockDim.x) {
      const real scaled =
        static_cast<real>(dirStart[i]) * (static_cast<real>(maxSteps[sysIdx]) / static_cast<real>(dirSum[0]));
      dirStart[i] = static_cast<stateT>(scaled);
    }
  }

  // -------------------------
  // Set slope, check validity
  // -------------------------
  reduceT localSum = 0;
  // Each thread computes its partial sum
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    localSum += static_cast<reduceT>(static_cast<real>(dirStart[i]) * static_cast<real>(gradStart[i]));
  }

  // Perform block-wide reduction to compute the total sum
  blockSum = BlockReduce(tempStorage).Sum(localSum);
  __syncthreads();
  // The first thread in the block writes the result

  if (isFirstThread) {
    slopes[sysIdx] = static_cast<stateT>(blockSum);
  }

  // ----------------------
  // Compute initial lambda
  // ----------------------
  reduceT localMax = 0;
  // Each thread computes its local maximum
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    const real temp = bfgsAbs(static_cast<real>(dirStart[i])) / max(bfgsAbs(static_cast<real>(posStart[i])), real{1});
    if (temp > localMax) {
      localMax = temp;
    }
  }
  // Perform block-wide reduction to find the maximum
  reduceT blockMax = BlockReduce(tempStorage).Reduce(localMax, cubMax());

  // The first thread in the block writes the result
  if (isFirstThread) {
    lambdaMins[sysIdx] = static_cast<stateT>(static_cast<reduceT>(MOVETOL) / blockMax);
  }
}

template <typename stateT>
__global__ void setLineStatusAndEnergyFromGlobalKernel(const int numSystems,
                                                       const int16_t* __restrict__ statuses,
                                                       int16_t* __restrict__ lineSearchStatus,
                                                       const double* __restrict__ srcEnergies,
                                                       stateT* __restrict__ destEnergies,
                                                       stateT* __restrict__ lineSearchLambdas) {
  const int sysIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (sysIdx < numSystems) {
    const int16_t status      = statuses[sysIdx];
    lineSearchStatus[sysIdx]  = status == 0 ? 0 : -2;
    destEnergies[sysIdx]      = static_cast<stateT>(srcEnergies[sysIdx]);
    lineSearchLambdas[sysIdx] = stateT{1};
  }
}

void BfgsBatchMinimizer::doLineSearchSetup(const double* srcEnergies) {
  const int     numblocks      = (numSystems_ + 128 - 1) / 128;
  constexpr int blockSizeSetup = 128;
#define NVMOLKIT_LAUNCH_BFGS_LINE_SETUP(real, reduceT, stateT, maxSteps, dirs, slopes, lambdaMins) \
  initializeLineSearchKernel<real, reduceT, stateT>                                                \
    <<<numUnfinishedSystems_, blockSizeSetup, 0, stream_>>>(statuses_.data(),                      \
                                                            positionsDevice,                       \
                                                            gradDevice,                            \
                                                            atomStartsDevice,                      \
                                                            maxSteps,                              \
                                                            dirs,                                  \
                                                            slopes,                                \
                                                            lambdaMins,                            \
                                                            activeSystemIndices_.data(),           \
                                                            dataDim_)
  if (usesFloatMinimizerState(precision_)) {
    setLineStatusAndEnergyFromGlobalKernel<float><<<numblocks, 128, 0, stream_>>>(numSystems_,
                                                                                  statuses_.data(),
                                                                                  lineSearchStatus_.data(),
                                                                                  srcEnergies,
                                                                                  lineSearchStoredEnergyFloat_.data(),
                                                                                  lineSearchLambdasFloat_.data());
    if (usesFloatMinimizerCompute(precision_)) {
      if (usesFloatReduction(precision_))
        NVMOLKIT_LAUNCH_BFGS_LINE_SETUP(float,
                                        float,
                                        float,
                                        lineSearchMaxStepsFloat_.data(),
                                        lineSearchDirFloat_.data(),
                                        lineSearchSlopeFloat_.data(),
                                        lineSearchLambdaMinsFloat_.data());
      else
        NVMOLKIT_LAUNCH_BFGS_LINE_SETUP(float,
                                        double,
                                        float,
                                        lineSearchMaxStepsFloat_.data(),
                                        lineSearchDirFloat_.data(),
                                        lineSearchSlopeFloat_.data(),
                                        lineSearchLambdaMinsFloat_.data());
    } else if (usesFloatReduction(precision_))
      NVMOLKIT_LAUNCH_BFGS_LINE_SETUP(double,
                                      float,
                                      float,
                                      lineSearchMaxStepsFloat_.data(),
                                      lineSearchDirFloat_.data(),
                                      lineSearchSlopeFloat_.data(),
                                      lineSearchLambdaMinsFloat_.data());
    else
      NVMOLKIT_LAUNCH_BFGS_LINE_SETUP(double,
                                      double,
                                      float,
                                      lineSearchMaxStepsFloat_.data(),
                                      lineSearchDirFloat_.data(),
                                      lineSearchSlopeFloat_.data(),
                                      lineSearchLambdaMinsFloat_.data());
  } else {
    setLineStatusAndEnergyFromGlobalKernel<double><<<numblocks, 128, 0, stream_>>>(numSystems_,
                                                                                   statuses_.data(),
                                                                                   lineSearchStatus_.data(),
                                                                                   srcEnergies,
                                                                                   lineSearchStoredEnergy_.data(),
                                                                                   lineSearchLambdas_.data());
    if (usesFloatMinimizerCompute(precision_)) {
      if (usesFloatReduction(precision_))
        NVMOLKIT_LAUNCH_BFGS_LINE_SETUP(float,
                                        float,
                                        double,
                                        lineSearchMaxSteps_.data(),
                                        lineSearchDir_.data(),
                                        lineSearchSlope_.data(),
                                        lineSearchLambdaMins_.data());
      else
        NVMOLKIT_LAUNCH_BFGS_LINE_SETUP(float,
                                        double,
                                        double,
                                        lineSearchMaxSteps_.data(),
                                        lineSearchDir_.data(),
                                        lineSearchSlope_.data(),
                                        lineSearchLambdaMins_.data());
    } else if (usesFloatReduction(precision_))
      NVMOLKIT_LAUNCH_BFGS_LINE_SETUP(double,
                                      float,
                                      double,
                                      lineSearchMaxSteps_.data(),
                                      lineSearchDir_.data(),
                                      lineSearchSlope_.data(),
                                      lineSearchLambdaMins_.data());
    else
      NVMOLKIT_LAUNCH_BFGS_LINE_SETUP(double,
                                      double,
                                      double,
                                      lineSearchMaxSteps_.data(),
                                      lineSearchDir_.data(),
                                      lineSearchSlope_.data(),
                                      lineSearchLambdaMins_.data());
  }
#undef NVMOLKIT_LAUNCH_BFGS_LINE_SETUP
  cudaCheckError(cudaGetLastError());
}

template <typename real, typename stateT>
__global__ void lineSearchPerturbKernel(const int*    atomStarts,
                                        const double* refPos,
                                        const stateT* dirs,
                                        const stateT* lambdas,
                                        const stateT* lambdaMins,
                                        stateT*       statePositions,
                                        double*       evalPositions,
                                        int16_t*      statuses,
                                        const int*    activeSystemIndices,
                                        const int     DIM) {
  const int     sysIdx        = activeSystemIndices[blockIdx.x];
  const int     idxInSys      = threadIdx.x;
  const int     numTerms      = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const stateT* dirStart      = &dirs[atomStarts[sysIdx] * DIM];
  const double* oldPosStart   = &refPos[atomStarts[sysIdx] * DIM];
  stateT*       statePosStart = &statePositions[atomStarts[sysIdx] * DIM];
  double*       evalPosStart  = &evalPositions[atomStarts[sysIdx] * DIM];
  const bool    isFirstThread = threadIdx.x == 0;

  const int16_t status = statuses[sysIdx];
  if (status != -2) {
    // Case where we've already converged or failed.
    return;
  }
  const real lambda    = static_cast<real>(lambdas[sysIdx]);
  const real lambdaMin = static_cast<real>(lambdaMins[sysIdx]);

  if (lambda < lambdaMin) {
    if (isFirstThread) {
      statuses[sysIdx] = 1;
    }
    return;
  }

  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    const real value = static_cast<real>(oldPosStart[i]) + lambda * static_cast<real>(dirStart[i]);
    statePosStart[i] = static_cast<stateT>(value);
    evalPosStart[i]  = static_cast<double>(static_cast<stateT>(value));
  }
}

void BfgsBatchMinimizer::doLineSearchPerturb() {
  if (usesFloatMinimizerCompute(precision_))
    if (usesFloatMinimizerState(precision_))
      lineSearchPerturbKernel<float, float>
        <<<numUnfinishedSystems_, 128, 0, stream_>>>(atomStartsDevice,
                                                     positionsDevice,
                                                     lineSearchDirFloat_.data(),
                                                     lineSearchLambdasFloat_.data(),
                                                     lineSearchLambdaMinsFloat_.data(),
                                                     scratchPositionsFloat_.data(),
                                                     scratchPositions_.data(),
                                                     lineSearchStatus_.data(),
                                                     activeSystemIndices_.data(),
                                                     dataDim_);
    else
      lineSearchPerturbKernel<float, double><<<numUnfinishedSystems_, 128, 0, stream_>>>(atomStartsDevice,
                                                                                         positionsDevice,
                                                                                         lineSearchDir_.data(),
                                                                                         lineSearchLambdas_.data(),
                                                                                         lineSearchLambdaMins_.data(),
                                                                                         scratchPositions_.data(),
                                                                                         scratchPositions_.data(),
                                                                                         lineSearchStatus_.data(),
                                                                                         activeSystemIndices_.data(),
                                                                                         dataDim_);
  else if (usesFloatMinimizerState(precision_))
    lineSearchPerturbKernel<double, float>
      <<<numUnfinishedSystems_, 128, 0, stream_>>>(atomStartsDevice,
                                                   positionsDevice,
                                                   lineSearchDirFloat_.data(),
                                                   lineSearchLambdasFloat_.data(),
                                                   lineSearchLambdaMinsFloat_.data(),
                                                   scratchPositionsFloat_.data(),
                                                   scratchPositions_.data(),
                                                   lineSearchStatus_.data(),
                                                   activeSystemIndices_.data(),
                                                   dataDim_);
  else
    lineSearchPerturbKernel<double, double><<<numUnfinishedSystems_, 128, 0, stream_>>>(atomStartsDevice,
                                                                                        positionsDevice,
                                                                                        lineSearchDir_.data(),
                                                                                        lineSearchLambdas_.data(),
                                                                                        lineSearchLambdaMins_.data(),
                                                                                        scratchPositions_.data(),
                                                                                        scratchPositions_.data(),
                                                                                        lineSearchStatus_.data(),
                                                                                        activeSystemIndices_.data(),
                                                                                        dataDim_);
  cudaCheckError(cudaGetLastError());
}

template <typename real, typename stateT>
__global__ void lineSearchPostEnergyKernel(const int     numSystems,
                                           const bool    isFirstIter,
                                           const stateT* prevE,  // oldval
                                           const double* newE,   // newval
                                           const stateT* slopes,
                                           stateT*       eScratch,  // val2
                                           stateT*       lambdas,
                                           stateT*       lambda2s,
                                           int16_t*      statuses) {
  const int sysIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (sysIdx >= numSystems) {
    return;
  }
  // Finished run.
  if (statuses[sysIdx] != -2) {
    return;
  }

  const real slope  = static_cast<real>(slopes[sysIdx]);
  const real newVal = static_cast<real>(newE[sysIdx]);
  const real oldVal = static_cast<real>(prevE[sysIdx]);
  const real lambda = static_cast<real>(lambdas[sysIdx]);
  if (newVal - oldVal <= static_cast<real>(FUNCTOL) * lambda * slope) {
    // we're converged on the function:
    statuses[sysIdx] = 0;
    return;
  }
  // if we made it this far, we need to backtrack:
  real tmpLambda;
  if (isFirstIter) {
    // it's the first step:
    tmpLambda = -slope / (real{2} * (newVal - oldVal - slope));
  } else {
    const real val2    = static_cast<real>(eScratch[sysIdx]);
    const real lambda2 = static_cast<real>(lambda2s[sysIdx]);
    real       rhs1    = newVal - oldVal - lambda * slope;
    real       rhs2    = val2 - oldVal - lambda2 * slope;
    real       a       = (rhs1 / (lambda * lambda) - rhs2 / (lambda2 * lambda2)) / (lambda - lambda2);
    real       b = (-lambda2 * rhs1 / (lambda * lambda) + lambda * rhs2 / (lambda2 * lambda2)) / (lambda - lambda2);
    if (a == real{0}) {
      tmpLambda = -slope / (real{2} * b);
    } else {
      real disc = b * b - real{3} * a * slope;
      if (disc < real{0}) {
        tmpLambda = real{0.5} * lambda;
      } else if (b <= real{0}) {
        tmpLambda = (-b + bfgsSqrt(disc)) / (real{3} * a);
      } else {
        tmpLambda = -slope / (b + bfgsSqrt(disc));
      }
    }
    if (tmpLambda > real{0.5} * lambda) {
      tmpLambda = real{0.5} * lambda;
    }
  }
  lambda2s[sysIdx] = static_cast<stateT>(lambda);
  eScratch[sysIdx] = static_cast<stateT>(newVal);
  lambdas[sysIdx]  = static_cast<stateT>(max(tmpLambda, real{0.1} * lambda));
}

void BfgsBatchMinimizer::doLineSearchPostEnergy(const int iter) {
  const int numBlocks = (numSystems_ + 127) / 128;
#define NVMOLKIT_LAUNCH_LINE_POST(real, stateT, stored, slope, scratch, lambda, lambda2)     \
  lineSearchPostEnergyKernel<real, stateT><<<numBlocks, 128, 0, stream_>>>(numSystems_,      \
                                                                           iter == 0,        \
                                                                           stored,           \
                                                                           energyOutsDevice, \
                                                                           slope,            \
                                                                           scratch,          \
                                                                           lambda,           \
                                                                           lambda2,          \
                                                                           lineSearchStatus_.data())
  if (usesFloatMinimizerState(precision_)) {
    if (usesFloatMinimizerCompute(precision_))
      NVMOLKIT_LAUNCH_LINE_POST(float,
                                float,
                                lineSearchStoredEnergyFloat_.data(),
                                lineSearchSlopeFloat_.data(),
                                lineSearchEnergyScratchFloat_.data(),
                                lineSearchLambdasFloat_.data(),
                                lineSearchLambdas2Float_.data());
    else
      NVMOLKIT_LAUNCH_LINE_POST(double,
                                float,
                                lineSearchStoredEnergyFloat_.data(),
                                lineSearchSlopeFloat_.data(),
                                lineSearchEnergyScratchFloat_.data(),
                                lineSearchLambdasFloat_.data(),
                                lineSearchLambdas2Float_.data());
  } else if (usesFloatMinimizerCompute(precision_))
    NVMOLKIT_LAUNCH_LINE_POST(float,
                              double,
                              lineSearchStoredEnergy_.data(),
                              lineSearchSlope_.data(),
                              lineSearchEnergyScratch_.data(),
                              lineSearchLambdas_.data(),
                              lineSearchLambdas2_.data());
  else
    NVMOLKIT_LAUNCH_LINE_POST(double,
                              double,
                              lineSearchStoredEnergy_.data(),
                              lineSearchSlope_.data(),
                              lineSearchEnergyScratch_.data(),
                              lineSearchLambdas_.data(),
                              lineSearchLambdas2_.data());
#undef NVMOLKIT_LAUNCH_LINE_POST
  cudaCheckError(cudaGetLastError());
}

template <typename stateT>
__global__ void lineSearchPostLoopKernel(const int*    atomStarts,
                                         int16_t*      statuses,
                                         const double* oldPos,
                                         stateT*       statePos,
                                         double*       evalPos,
                                         const int*    activeSystemIndices,
                                         const int     DIM) {
  const int     sysIdx      = activeSystemIndices[blockIdx.x];
  const int     idxInSys    = threadIdx.x;
  const int     numTerms    = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const double* oldPosStart = &oldPos[atomStarts[sysIdx] * DIM];
  stateT*       stateStart  = &statePos[atomStarts[sysIdx] * DIM];
  double*       evalStart   = &evalPos[atomStarts[sysIdx] * DIM];

  // Special handling of statuses needed here, to reproduce RDKit behavior which has either early returns
  // or loop breaks depending on the status. Note that "-2" is not a status in the RDKit code, but for us
  // it means reached the end of the loop, and should be a -1.
  const int16_t status              = statuses[sysIdx];
  // These are the two cases in the RDKit loop where this end section triggers. A -1 or 0 exits the function.
  const bool    needUpdatePositions = status == -2 || status == 1;
  // Match RDKit for end of loop case.
  if (status == -2) {
    if (threadIdx.x == 0) {
      statuses[sysIdx] = -1;
    }
  }
  if (needUpdatePositions) {
    for (int i = idxInSys; i < numTerms; i += blockDim.x) {
      stateStart[i] = static_cast<stateT>(oldPosStart[i]);
      evalStart[i]  = oldPosStart[i];
    }
  }
}

void BfgsBatchMinimizer::doLineSearchPostLoop() {
  if (usesFloatMinimizerState(precision_))
    lineSearchPostLoopKernel<float><<<numUnfinishedSystems_, 128, 0, stream_>>>(atomStartsDevice,
                                                                                lineSearchStatus_.data(),
                                                                                positionsDevice,
                                                                                scratchPositionsFloat_.data(),
                                                                                scratchPositions_.data(),
                                                                                activeSystemIndices_.data(),
                                                                                dataDim_);
  else
    lineSearchPostLoopKernel<double><<<numUnfinishedSystems_, 128, 0, stream_>>>(atomStartsDevice,
                                                                                 lineSearchStatus_.data(),
                                                                                 positionsDevice,
                                                                                 scratchPositions_.data(),
                                                                                 scratchPositions_.data(),
                                                                                 activeSystemIndices_.data(),
                                                                                 dataDim_);
  cudaCheckError(cudaGetLastError());
}

struct NotEqualToMinusTwoFunctor {
  __host__ __device__ int operator()(const int16_t& x) const { return x != -2 ? 1 : 0; }
};

struct EqualsZeroFunctor {
  __host__ __device__ int operator()(const int16_t& x) const { return x == 0; }
};

BfgsBatchMinimizer::BfgsBatchMinimizer(const int        dataDim,
                                       DebugLevel       debugLevel,
                                       bool             scaleGrads,
                                       cudaStream_t     stream,
                                       BfgsBackend      backend,
                                       PrecisionOptions precision)
    : countFinished_(0, stream) {
  debugLevel_ = debugLevel;
  dataDim_    = dataDim;
  scaleGrads_ = scaleGrads;
  stream_     = stream;
  // The batched kernels carry independent real/reduceT/storageT dispatch.
  // The fused backend remains the legacy specialization until it gains the
  // same policy surface, so precision experiments use the complete path.
  backend_    = bfgsPrecisionRequiresBatchedBackend(precision) ? BfgsBackend::BATCHED : backend;
  precision_  = precision;
  // For HYBRID, we need to support both paths, so initialize for both
  if (backend_ == BfgsBackend::BATCHED || backend_ == BfgsBackend::HYBRID) {
    loopStatusHost_.resize(1);
  }

  if (stream_ != nullptr) {
    activeSystemIndices_.setStream(stream_);
    allSystemIndices_.setStream(stream_);

    scratchPositions_.setStream(stream_);
    scratchPositionsFloat_.setStream(stream_);
    statuses_.setStream(stream_);

    lineSearchDir_.setStream(stream_);
    lineSearchDirFloat_.setStream(stream_);
    lineSearchStatus_.setStream(stream_);
    lineSearchLambdaMins_.setStream(stream_);
    lineSearchLambdaMinsFloat_.setStream(stream_);
    lineSearchLambdas_.setStream(stream_);
    lineSearchLambdasFloat_.setStream(stream_);
    lineSearchLambdas2_.setStream(stream_);
    lineSearchLambdas2Float_.setStream(stream_);
    lineSearchSlope_.setStream(stream_);
    lineSearchSlopeFloat_.setStream(stream_);
    lineSearchMaxSteps_.setStream(stream_);
    lineSearchMaxStepsFloat_.setStream(stream_);
    countTempStorage_.setStream(stream_);
    countFinished_.setStream(stream_);
    lineSearchStoredEnergy_.setStream(stream_);
    lineSearchStoredEnergyFloat_.setStream(stream_);
    lineSearchEnergyScratch_.setStream(stream_);
    lineSearchEnergyScratchFloat_.setStream(stream_);
    lineSearchEnergyOut_.setStream(stream_);

    finalEnergies_.setStream(stream_);

    hessianStarts_.setStream(stream_);
    scratchGrad_.setStream(stream_);
    scratchGradFloat_.setStream(stream_);
    gradScales_.setStream(stream_);
    gradScalesFloat_.setStream(stream_);
    inverseHessian_.setStream(stream_);
    inverseHessianFloat_.setStream(stream_);
    hessDGrad_.setStream(stream_);
    hessDGradFloat_.setStream(stream_);
    scratchBuffersDevice_.setStream(stream_);
    activeMolIdsDevice_.setStream(stream_);
  }
  // Allocate device array for per-molecule backend (also needed for HYBRID which might use it)
  if (backend_ == BfgsBackend::PER_MOLECULE || backend_ == BfgsBackend::HYBRID) {
    scratchBuffersDevice_.resize(5);
  }
}
BfgsBatchMinimizer::~BfgsBatchMinimizer() = default;

BfgsBackend BfgsBatchMinimizer::resolveBackend(const std::vector<int>& atomStartsHost) const {
  return nvMolKit::resolveBackend(backend_, atomStartsHost);
}

void BfgsBatchMinimizer::initialize(const std::vector<int>& atomStartsHost,
                                    const int*              atomStarts,
                                    double*                 positions,
                                    double*                 grad,
                                    double*                 energyOuts,
                                    BfgsBackend             effectiveBackend,
                                    const uint8_t*          activeThisStage) {
  atomStartsDevice = atomStarts;
  positionsDevice  = positions;
  gradDevice       = grad;
  energyOutsDevice = energyOuts;

  const int numSystems = atomStartsHost.size() - 1;
  activeHost_.resize(numSystems);
  convergenceHost_.resize(numSystems);
  scratchBufferPointersHost_.resize(5);

  statuses_.resize(numSystems);
  if (activeThisStage) {
    // Copy activeThisStage to statuses_ with type conversion
    const int blockSize = 128;
    const int numBlocks = (numSystems + blockSize - 1) / blockSize;
    copyActiveToStatusKernel<<<numBlocks, blockSize, 0, stream_>>>(numSystems, activeThisStage, statuses_.data());
  } else {
    // Default initialization to all 1s
    setAll(statuses_, static_cast<int16_t>(1));
  }
  cudaCheckError(cudaGetLastError());

  numSystems_     = numSystems;
  numAtomsTotal_  = atomStartsHost.back();
  hasLargeSystem_ = false;

  if (effectiveBackend == BfgsBackend::PER_MOLECULE) {
    // Copy activeThisStage to host for CPU-side filtering (using pinned memory)
    std::fill_n(activeHost_.begin(), numSystems, 1);
    if (activeThisStage) {
      cudaCheckError(cudaMemcpyAsync(activeHost_.data(),
                                     activeThisStage,
                                     numSystems * sizeof(uint8_t),
                                     cudaMemcpyDeviceToHost,
                                     stream_));
      cudaCheckError(cudaStreamSynchronize(stream_));
    }

    activeMolIds_.clear();
    maxAtomsInBatch_ = 0;

    for (int i = 0; i < numSystems_; ++i) {
      if (activeHost_[i] == 0) {
        continue;
      }

      const int numAtoms = atomStartsHost[i + 1] - atomStartsHost[i];
      activeMolIds_.push_back(i);

      if (numAtoms > maxAtomsInBatch_) {
        maxAtomsInBatch_ = numAtoms;
      }
      if (numAtoms > 256) {
        hasLargeSystem_ = true;
      }
    }

    // Transfer active molecule list to device
    if (!activeMolIds_.empty()) {
      activeMolIdsDevice_.resize(activeMolIds_.size());
      activeMolIdsDevice_.setFromVector(activeMolIds_);
    }
  } else {
    // Original logic for batched backend
    for (int i = 0; i < numSystems_; ++i) {
      const int numAtoms = atomStartsHost[i + 1] - atomStartsHost[i];
      if (numAtoms > 256) {
        hasLargeSystem_ = true;
        break;
      }
    }
  }

  activeSystemIndices_.resize(numSystems_);
  allSystemIndices_.resize(numSystems_);
  systemIndicesHost_.resize(numSystems_);
  std::iota(systemIndicesHost_.begin(), systemIndicesHost_.end(), 0);
  allSystemIndices_.setFromVector(systemIndicesHost_);
  activeSystemIndices_.setFromVector(systemIndicesHost_);

  hessianStartsHost_.clear();
  hessianStartsHost_.reserve(numSystems + 1);
  hessianStartsHost_.push_back(0);
  for (int i = 0; i < numSystems; ++i) {
    const int numAtoms = atomStartsHost[i + 1] - atomStartsHost[i];
    // Note - hessian starts is total term based, not atom based.
    const int numTerms = (dataDim_ * numAtoms) * (dataDim_ * numAtoms);
    hessianStartsHost_.push_back(hessianStartsHost_.back() + numTerms);
  }
  hessianStarts_.resize(numSystems + 1);
  hessianStarts_.setFromVector(hessianStartsHost_);
  if (usesFloatHessian(precision_)) {
    inverseHessian_.resize(0);
    inverseHessianFloat_.resize(hessianStartsHost_.back());
    inverseHessianFloat_.zero();
  } else {
    inverseHessianFloat_.resize(0);
    inverseHessian_.resize(hessianStartsHost_.back());
    inverseHessian_.zero();
  }

  const int numStateTerms = atomStartsHost.back() * dataDim_;
  // This double buffer is the forcefield evaluation bridge in both state modes.
  scratchPositions_.resize(numStateTerms);
  scratchPositions_.zero();
  lineSearchStatus_.resize(numSystems);
  if (usesFloatMinimizerState(precision_)) {
    scratchPositionsFloat_.resize(numStateTerms);
    scratchPositionsFloat_.zero();
    lineSearchDirFloat_.resize(numStateTerms);
    scratchGradFloat_.resize(numStateTerms);
    hessDGradFloat_.resize(numStateTerms);
    hessDGradFloat_.zero();
    gradScalesFloat_.resize(numSystems);
    lineSearchLambdaMinsFloat_.resize(numSystems);
    lineSearchLambdasFloat_.resize(numSystems);
    lineSearchLambdas2Float_.resize(numSystems);
    lineSearchSlopeFloat_.resize(numSystems);
    lineSearchMaxStepsFloat_.resize(numSystems);
    lineSearchStoredEnergyFloat_.resize(numSystems);
    lineSearchEnergyScratchFloat_.resize(numSystems);

    lineSearchDir_.resize(0);
    scratchGrad_.resize(0);
    hessDGrad_.resize(0);
    gradScales_.resize(0);
    lineSearchLambdaMins_.resize(0);
    lineSearchLambdas_.resize(0);
    lineSearchLambdas2_.resize(0);
    lineSearchSlope_.resize(0);
    lineSearchMaxSteps_.resize(0);
    lineSearchStoredEnergy_.resize(0);
    lineSearchEnergyScratch_.resize(0);
  } else {
    scratchPositionsFloat_.resize(0);
    lineSearchDirFloat_.resize(0);
    scratchGradFloat_.resize(0);
    hessDGradFloat_.resize(0);
    gradScalesFloat_.resize(0);
    lineSearchLambdaMinsFloat_.resize(0);
    lineSearchLambdasFloat_.resize(0);
    lineSearchLambdas2Float_.resize(0);
    lineSearchSlopeFloat_.resize(0);
    lineSearchMaxStepsFloat_.resize(0);
    lineSearchStoredEnergyFloat_.resize(0);
    lineSearchEnergyScratchFloat_.resize(0);

    lineSearchDir_.resize(numStateTerms);
    scratchGrad_.resize(numStateTerms);
    hessDGrad_.resize(numStateTerms);
    hessDGrad_.zero();
    gradScales_.resize(numSystems);
    lineSearchLambdaMins_.resize(numSystems);
    lineSearchLambdas_.resize(numSystems);
    lineSearchLambdas2_.resize(numSystems);
    lineSearchSlope_.resize(numSystems);
    lineSearchMaxSteps_.resize(numSystems);
    lineSearchStoredEnergy_.resize(numSystems);
    lineSearchEnergyScratch_.resize(numSystems);
  }

  // Compute needed reduction storage.
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::TransformReduce(nullptr,
                                     temp_storage_bytes,
                                     lineSearchStatus_.data(),
                                     countFinished_.data(),
                                     lineSearchStatus_.size(),
                                     cubSum(),
                                     NotEqualToMinusTwoFunctor(),
                                     0,
                                     stream_);
  countTempStorage_.resize(temp_storage_bytes);

  cub::DeviceSelect::Flagged(nullptr,
                             temp_storage_bytes,
                             allSystemIndices_.data(),
                             statuses_.data(),
                             activeSystemIndices_.data(),
                             countFinished_.data(),
                             statuses_.size(),
                             stream_);

  if (temp_storage_bytes > countTempStorage_.size()) {
    countTempStorage_.zero();
    countTempStorage_.resize(temp_storage_bytes);
  }
}

template <typename storageT>
__global__ void populateHessianIdentityKernel(const int* hessianStarts,
                                              const int* atomStarts,
                                              storageT*  inverseHessian,
                                              const int  DIM) {
  const int sysIdx        = blockIdx.x;
  const int idxInSys      = threadIdx.x;
  const int writeStartIdx = hessianStarts[sysIdx];
  const int numTerms      = hessianStarts[sysIdx + 1] - hessianStarts[sysIdx];
  const int numAtoms      = atomStarts[sysIdx + 1] - atomStarts[sysIdx];
  const int rowLength     = DIM * numAtoms;

  for (int i = idxInSys; i < rowLength; i += blockDim.x) {
    if (i < numTerms) {
      inverseHessian[writeStartIdx + i * rowLength + i] = static_cast<storageT>(1.0);
    }
  }
}

void BfgsBatchMinimizer::setHessianToIdentity() {
  constexpr int blockDim  = 128;
  const int     numBlocks = hessianStarts_.size() - 1;
  if (usesFloatHessian(precision_)) {
    inverseHessianFloat_.zero();
    populateHessianIdentityKernel<<<numBlocks, blockDim, 0, stream_>>>(hessianStarts_.data(),
                                                                       atomStartsDevice,
                                                                       inverseHessianFloat_.data(),
                                                                       dataDim_);
  } else {
    inverseHessian_.zero();
    populateHessianIdentityKernel<<<numBlocks, blockDim, 0, stream_>>>(hessianStarts_.data(),
                                                                       atomStartsDevice,
                                                                       inverseHessian_.data(),
                                                                       dataDim_);
  }
  cudaCheckError(cudaGetLastError());
}

template <typename real, typename reduceT, typename stateT>
__global__ void setMaxStepKernel(const int* atomStarts, const double* positions, stateT* maxSteps, const int DIM) {
  const int  sysIdx        = blockIdx.x;
  const int  idxInSys      = threadIdx.x;
  const bool isFirstThread = threadIdx.x == 0;

  const int     numTerms = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const double* posStart = &positions[atomStarts[sysIdx] * DIM];

  reduceT sumSquaredPos = 0;
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    const real x = static_cast<real>(posStart[i]);
    sumSquaredPos += static_cast<reduceT>(x * x);
  }

  using BlockReduce = cub::BlockReduce<reduceT, 128>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  const reduceT squaredSum = BlockReduce(tempStorage).Sum(sumSquaredPos);
  if (isFirstThread) {
    constexpr real maxStepFactor = real{100};
    maxSteps[sysIdx] =
      static_cast<stateT>(maxStepFactor * max(static_cast<real>(bfgsSqrt(squaredSum)), static_cast<real>(numTerms)));
  }
}

void BfgsBatchMinimizer::setMaxStep() {
#define NVMOLKIT_LAUNCH_MAX_STEP(real, reduceT, stateT, output) \
  setMaxStepKernel<real, reduceT, stateT>                       \
    <<<numSystems_, 128, 0, stream_>>>(atomStartsDevice, positionsDevice, output, dataDim_)
  if (usesFloatMinimizerState(precision_)) {
    if (usesFloatMinimizerCompute(precision_)) {
      if (usesFloatReduction(precision_))
        NVMOLKIT_LAUNCH_MAX_STEP(float, float, float, lineSearchMaxStepsFloat_.data());
      else
        NVMOLKIT_LAUNCH_MAX_STEP(float, double, float, lineSearchMaxStepsFloat_.data());
    } else if (usesFloatReduction(precision_))
      NVMOLKIT_LAUNCH_MAX_STEP(double, float, float, lineSearchMaxStepsFloat_.data());
    else
      NVMOLKIT_LAUNCH_MAX_STEP(double, double, float, lineSearchMaxStepsFloat_.data());
  } else if (usesFloatMinimizerCompute(precision_)) {
    if (usesFloatReduction(precision_))
      NVMOLKIT_LAUNCH_MAX_STEP(float, float, double, lineSearchMaxSteps_.data());
    else
      NVMOLKIT_LAUNCH_MAX_STEP(float, double, double, lineSearchMaxSteps_.data());
  } else {
    if (usesFloatReduction(precision_))
      NVMOLKIT_LAUNCH_MAX_STEP(double, float, double, lineSearchMaxSteps_.data());
    else
      NVMOLKIT_LAUNCH_MAX_STEP(double, double, double, lineSearchMaxSteps_.data());
  }
#undef NVMOLKIT_LAUNCH_MAX_STEP
  cudaCheckError(cudaGetLastError());
}

namespace {

template <typename stateT> __global__ void copyAndNegate(const int numElements, const double* src, stateT* dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = static_cast<stateT>(-src[idx]);
  }
}

void prepareScratchBuffers(AsyncDeviceVector<double>&  grad,
                           AsyncDeviceVector<double>&  lineSearchDir,
                           AsyncDeviceVector<double>&  scratchPositions,
                           AsyncDeviceVector<double>&  hessDGrad,
                           AsyncDeviceVector<double>&  scratchGrad,
                           AsyncDeviceVector<double*>& scratchBuffersDevice,
                           PinnedHostVector<double*>&  scratchBufferPointersHost,
                           cudaStream_t                stream) {
  scratchBufferPointersHost[0] = grad.data();
  scratchBufferPointersHost[1] = lineSearchDir.data();
  scratchBufferPointersHost[2] = scratchPositions.data();
  scratchBufferPointersHost[3] = hessDGrad.data();
  scratchBufferPointersHost[4] = scratchGrad.data();

  cudaCheckError(cudaMemcpyAsync(scratchBuffersDevice.data(),
                                 scratchBufferPointersHost.data(),
                                 5 * sizeof(double*),
                                 cudaMemcpyHostToDevice,
                                 stream));
}

bool checkConvergence(const std::vector<int>&     activeMolIds,
                      AsyncDeviceVector<int16_t>& statuses,
                      PinnedHostVector<int16_t>&  convergenceHost,
                      const int                   numSystems,
                      cudaStream_t                stream) {
  statuses.copyToHost(convergenceHost.data(), numSystems);
  cudaCheckError(cudaStreamSynchronize(stream));

  for (const int molIdx : activeMolIds) {
    if (convergenceHost[molIdx] != 0) {
      return true;
    }
  }
  return false;
}

}  // namespace

int BfgsBatchMinimizer::lineSearchCountFinished() const {
  size_t temp_storage_bytes = countTempStorage_.size();
  cub::DeviceReduce::TransformReduce(countTempStorage_.data(),
                                     temp_storage_bytes,
                                     lineSearchStatus_.data(),
                                     countFinished_.data(),
                                     lineSearchStatus_.size(),
                                     cubSum(),
                                     NotEqualToMinusTwoFunctor(),
                                     0,
                                     stream_);
  int& finishedHost = loopStatusHost_[0];
  countFinished_.get(finishedHost);
  cudaStreamSynchronize(stream_);
  return finishedHost;
}

int BfgsBatchMinimizer::compactAndCountConverged() const {
  const ScopedNvtxRange bfgsCompactAndCountConverged("BfgsBatchMinimizer::compactAndCountConverged");
  size_t                temp_storage_bytes = countTempStorage_.size();

  cudaCheckError(cub::DeviceSelect::Flagged(countTempStorage_.data(),
                                            temp_storage_bytes,
                                            allSystemIndices_.data(),
                                            statuses_.data(),
                                            activeSystemIndices_.data(),
                                            countFinished_.data(),
                                            statuses_.size(),
                                            stream_));
  // std::vector<int> allHost(numSystems_);
  // std::vector<int> allCompact(numSystems_);
  // std::vector<int16_t> statusHost(numSystems_);
  // statuses_.copyToHost(statusHost);
  // allSystemIndices_.copyToHost(allHost);
  // activeSystemIndices_.copyToHost(allCompact);
  int& unfinishedHost = loopStatusHost_[0];
  countFinished_.get(unfinishedHost);
  cudaStreamSynchronize(stream_);
  numUnfinishedSystems_ = unfinishedHost;
  return numSystems_ - unfinishedHost;
}

template <typename real, typename reduceT, typename stateT>
__global__ void setDirectionKernel(const int*    atomStarts,
                                   const stateT* positionsFromLineSearch,
                                   const double* grads,
                                   stateT*       xis,
                                   double*       positions,
                                   stateT*       dGrads,
                                   int16_t*      statuses,
                                   const int*    activeSystemIndices,
                                   const int     DIM,
                                   const bool    fixedSteps) {
  const int sysIdx          = activeSystemIndices[blockIdx.x];
  const int idxWithinSystem = threadIdx.x;
  const int numTerms        = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const int startIdx        = atomStarts[sysIdx] * DIM;

  if (statuses[sysIdx] == 0) {
    return;
  }

  stateT*       localXi            = &xis[startIdx];
  double*       localPos           = &positions[startIdx];
  const stateT* localPosLineSearch = &positionsFromLineSearch[startIdx];
  const double* localGrad          = &grads[startIdx];
  stateT*       localDGrad         = &dGrads[startIdx];

  reduceT localMax = 0;
  for (int i = idxWithinSystem; i < numTerms; i += blockDim.x) {
    const real xi = static_cast<real>(localPosLineSearch[i]) - static_cast<real>(localPos[i]);
    localXi[i]    = static_cast<stateT>(xi);
    localPos[i]   = static_cast<double>(localPosLineSearch[i]);
    localDGrad[i] = static_cast<stateT>(localGrad[i]);

    const real temp = bfgsAbs(xi) / max(bfgsAbs(static_cast<real>(localPos[i])), real{1});
    // TODO we could have a better thread distribution pattern for the local Max.
    if (temp > localMax) {
      localMax = temp;
    }
  }

  __shared__ typename cub::BlockReduce<reduceT, 128>::TempStorage tempStorage;
  const reduceT     blockMax = cub::BlockReduce<reduceT, 128>(tempStorage).Reduce(localMax, cubMax());
  constexpr reduceT TOLX     = static_cast<reduceT>(4. * 3e-8);
  if (!fixedSteps && idxWithinSystem == 0 && blockMax < TOLX) {
    // Converged
    statuses[sysIdx] = 0;
  }
}

void BfgsBatchMinimizer::setDirection() {
  const ScopedNvtxRange bfgsSetDirection("BfgsBatchMinimizer::setDirection");
#define NVMOLKIT_LAUNCH_SET_DIRECTION(real, reduceT, stateT, scratchPos, dirs, scratchGrad)                          \
  setDirectionKernel<real, reduceT, stateT><<<numUnfinishedSystems_, 128, 0, stream_>>>(atomStartsDevice,            \
                                                                                        scratchPos,                  \
                                                                                        gradDevice,                  \
                                                                                        dirs,                        \
                                                                                        positionsDevice,             \
                                                                                        scratchGrad,                 \
                                                                                        statuses_.data(),            \
                                                                                        activeSystemIndices_.data(), \
                                                                                        dataDim_,                    \
                                                                                        fixedSteps_)
  if (usesFloatMinimizerState(precision_)) {
    if (usesFloatMinimizerCompute(precision_)) {
      if (usesFloatReduction(precision_))
        NVMOLKIT_LAUNCH_SET_DIRECTION(float,
                                      float,
                                      float,
                                      scratchPositionsFloat_.data(),
                                      lineSearchDirFloat_.data(),
                                      scratchGradFloat_.data());
      else
        NVMOLKIT_LAUNCH_SET_DIRECTION(float,
                                      double,
                                      float,
                                      scratchPositionsFloat_.data(),
                                      lineSearchDirFloat_.data(),
                                      scratchGradFloat_.data());
    } else if (usesFloatReduction(precision_))
      NVMOLKIT_LAUNCH_SET_DIRECTION(double,
                                    float,
                                    float,
                                    scratchPositionsFloat_.data(),
                                    lineSearchDirFloat_.data(),
                                    scratchGradFloat_.data());
    else
      NVMOLKIT_LAUNCH_SET_DIRECTION(double,
                                    double,
                                    float,
                                    scratchPositionsFloat_.data(),
                                    lineSearchDirFloat_.data(),
                                    scratchGradFloat_.data());
  } else if (usesFloatMinimizerCompute(precision_)) {
    if (usesFloatReduction(precision_))
      NVMOLKIT_LAUNCH_SET_DIRECTION(float,
                                    float,
                                    double,
                                    scratchPositions_.data(),
                                    lineSearchDir_.data(),
                                    scratchGrad_.data());
    else
      NVMOLKIT_LAUNCH_SET_DIRECTION(float,
                                    double,
                                    double,
                                    scratchPositions_.data(),
                                    lineSearchDir_.data(),
                                    scratchGrad_.data());
  } else {
    if (usesFloatReduction(precision_))
      NVMOLKIT_LAUNCH_SET_DIRECTION(double,
                                    float,
                                    double,
                                    scratchPositions_.data(),
                                    lineSearchDir_.data(),
                                    scratchGrad_.data());
    else
      NVMOLKIT_LAUNCH_SET_DIRECTION(double,
                                    double,
                                    double,
                                    scratchPositions_.data(),
                                    lineSearchDir_.data(),
                                    scratchGrad_.data());
  }
#undef NVMOLKIT_LAUNCH_SET_DIRECTION
  cudaCheckError(cudaGetLastError());
}

// Mirrors RDKit's ForceField::minimize gradient cap (calcGradient in
// Code/ForceField/ForceField.cpp). RDKit historically tracked the signed max of
// gradient components; commit 5b1d04d23 (RDKit 2025.09) switched to |grad|.
// Follow whichever rule the linked RDKit uses so weighted MMFF/UFF minimization
// trajectories agree with the host reference.
template <bool scaleGrads, typename real, typename reduceT, typename stateT>
__global__ void scaleGradKernel(const int16_t* statuses,
                                const int*     atomStarts,
                                double*        grads,
                                stateT*        gradScales,
                                const int*     activeSystemIndices,
                                const int      DIM) {
  constexpr bool kRdkitHasGradScaleFix =
    RDKIT_VERSION_MAJOR > 2025 || (RDKIT_VERSION_MAJOR == 2025 && RDKIT_VERSION_MINOR >= 9);
  const int sysIdx          = activeSystemIndices == nullptr ? blockIdx.x : activeSystemIndices[blockIdx.x];
  const int idxWithinSystem = threadIdx.x;
  const int numTerms        = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);

  if (statuses[sysIdx] == 0) {
    return;
  }

  double* localGrad = &grads[atomStarts[sysIdx] * DIM];

  reduceT            maxGrad   = kRdkitHasGradScaleFix ? reduceT{0} : reduceT{-1e8};
  real               gradScale = scaleGrads ? real{0.1} : real{1};
  __shared__ reduceT distributedMax[1];
  if (idxWithinSystem == 0) {
    distributedMax[0] = -1.0;  // See note at start at function, this will work for now.
  }

  for (int i = idxWithinSystem; i < numTerms; i += blockDim.x) {
    const real scaled = static_cast<real>(localGrad[i]) * gradScale;
    localGrad[i]      = scaled;
    const real cmp    = kRdkitHasGradScaleFix ? bfgsAbs(scaled) : scaled;
    if (cmp > maxGrad) {
      maxGrad = cmp;
    }
  }

  __shared__ typename cub::BlockReduce<reduceT, 128>::TempStorage tempStorage;
  const reduceT blockMax = cub::BlockReduce<reduceT, 128>(tempStorage).Reduce(maxGrad, cubMax());

  if (idxWithinSystem == 0) {
    distributedMax[0] = blockMax;
  }
  __syncthreads();
  maxGrad = distributedMax[0];

  if (scaleGrads && maxGrad > 10.0) {
    while (maxGrad * gradScale > 10.0) {
      gradScale *= .5;
    }
    for (int i = idxWithinSystem; i < numTerms; i += blockDim.x) {
      localGrad[i] = static_cast<real>(localGrad[i]) * gradScale;
    }
  }
  if (idxWithinSystem == 0) {
    gradScales[sysIdx] = static_cast<stateT>(gradScale);
  }
}

void BfgsBatchMinimizer::scaleGrad(const bool preLoop) {
  const int  numSystems          = preLoop ? numSystems_ : numUnfinishedSystems_;
  const int* activeSystemIndices = preLoop ? nullptr : activeSystemIndices_.data();
#define NVMOLKIT_LAUNCH_SCALE_GRAD(Scale, real, reduceT, stateT, gradScales)                          \
  scaleGradKernel<Scale, real, reduceT, stateT><<<numSystems, 128, 0, stream_>>>(statuses_.data(),    \
                                                                                 atomStartsDevice,    \
                                                                                 gradDevice,          \
                                                                                 gradScales,          \
                                                                                 activeSystemIndices, \
                                                                                 dataDim_)
  const bool floatState = usesFloatMinimizerState(precision_);
  if (usesFloatMinimizerCompute(precision_)) {
    if (usesFloatReduction(precision_)) {
      if (scaleGrads_)
        if (floatState)
          NVMOLKIT_LAUNCH_SCALE_GRAD(true, float, float, float, gradScalesFloat_.data());
        else
          NVMOLKIT_LAUNCH_SCALE_GRAD(true, float, float, double, gradScales_.data());
      else if (floatState)
        NVMOLKIT_LAUNCH_SCALE_GRAD(false, float, float, float, gradScalesFloat_.data());
      else
        NVMOLKIT_LAUNCH_SCALE_GRAD(false, float, float, double, gradScales_.data());
    } else {
      if (scaleGrads_)
        if (floatState)
          NVMOLKIT_LAUNCH_SCALE_GRAD(true, float, double, float, gradScalesFloat_.data());
        else
          NVMOLKIT_LAUNCH_SCALE_GRAD(true, float, double, double, gradScales_.data());
      else if (floatState)
        NVMOLKIT_LAUNCH_SCALE_GRAD(false, float, double, float, gradScalesFloat_.data());
      else
        NVMOLKIT_LAUNCH_SCALE_GRAD(false, float, double, double, gradScales_.data());
    }
  } else if (usesFloatReduction(precision_)) {
    if (scaleGrads_)
      if (floatState)
        NVMOLKIT_LAUNCH_SCALE_GRAD(true, double, float, float, gradScalesFloat_.data());
      else
        NVMOLKIT_LAUNCH_SCALE_GRAD(true, double, float, double, gradScales_.data());
    else if (floatState)
      NVMOLKIT_LAUNCH_SCALE_GRAD(false, double, float, float, gradScalesFloat_.data());
    else
      NVMOLKIT_LAUNCH_SCALE_GRAD(false, double, float, double, gradScales_.data());
  } else {
    if (scaleGrads_)
      if (floatState)
        NVMOLKIT_LAUNCH_SCALE_GRAD(true, double, double, float, gradScalesFloat_.data());
      else
        NVMOLKIT_LAUNCH_SCALE_GRAD(true, double, double, double, gradScales_.data());
    else if (floatState)
      NVMOLKIT_LAUNCH_SCALE_GRAD(false, double, double, float, gradScalesFloat_.data());
    else
      NVMOLKIT_LAUNCH_SCALE_GRAD(false, double, double, double, gradScales_.data());
  }
#undef NVMOLKIT_LAUNCH_SCALE_GRAD
}

template <typename real, typename reduceT, typename stateT>
__global__ void updateDGradKernel(const double  gradTol,
                                  const int*    atomStarts,
                                  const double* energies,
                                  const stateT* gradScales,
                                  const double* grads,
                                  const double* positions,
                                  stateT*       dGrads,
                                  int16_t*      statuses,
                                  const int*    activeSystemIndices,
                                  const int     DIM,
                                  const bool    fixedSteps) {
  const int sysIdx          = activeSystemIndices[blockIdx.x];
  const int idxWithinSystem = threadIdx.x;
  const int numTerms        = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const int startIdx        = atomStarts[sysIdx] * DIM;

  if (statuses[sysIdx] == 0) {
    return;
  }

  const double* localGrad = &grads[startIdx];

  const double* localPos   = &positions[startIdx];
  stateT*       localDGrad = &dGrads[startIdx];

  reduceT localMax = 0;

  for (int i = idxWithinSystem; i < numTerms; i += blockDim.x) {
    const real gradValue = static_cast<real>(localGrad[i]);
    localDGrad[i]        = static_cast<stateT>(gradValue - static_cast<real>(localDGrad[i]));
    const real temp      = bfgsAbs(gradValue) * max(bfgsAbs(static_cast<real>(localPos[i])), real{1});
    // TODO we could have a better thread distribution pattern for the local Max.
    if (temp > localMax) {
      localMax = temp;
    }
  }
  __shared__ typename cub::BlockReduce<reduceT, 128>::TempStorage tempStorage;
  reduceT blockMax = cub::BlockReduce<reduceT, 128>(tempStorage).Reduce(localMax, cubMax());

  if (idxWithinSystem == 0) {
    // rdkit/rdkit#9298 (merged RDKit 2026.03) fixed the signed-energy denominator bug:
    // raw negative energy clamped the denominator to 1, artificially tightening gradTol.
    // Use |energy| when linked against a fixed RDKit; keep signed otherwise for parity.
    constexpr bool kRdkitHasGradDenomFix =
      RDKIT_VERSION_MAJOR > 2026 || (RDKIT_VERSION_MAJOR == 2026 && RDKIT_VERSION_MINOR >= 3);
    const real energyValue = static_cast<real>(energies[sysIdx]);
    const real energyMag   = kRdkitHasGradDenomFix ? bfgsAbs(energyValue) : energyValue;
    const real term        = max(energyMag * static_cast<real>(gradScales[sysIdx]), real{1});
    blockMax /= term;
    if (!fixedSteps && blockMax < static_cast<reduceT>(gradTol)) {
      // Converged
      statuses[sysIdx] = 0;
    }
  }
}

void BfgsBatchMinimizer::updateDGrad() {
  const ScopedNvtxRange bfgsUpdateDGrad("BfgsBatchMinimizer::updateDGrad");
#define NVMOLKIT_LAUNCH_UPDATE_DGRAD(real, reduceT, stateT, gradScales, dGrads)                                     \
  updateDGradKernel<real, reduceT, stateT><<<numUnfinishedSystems_, 128, 0, stream_>>>(gradTol_,                    \
                                                                                       atomStartsDevice,            \
                                                                                       energyOutsDevice,            \
                                                                                       gradScales,                  \
                                                                                       gradDevice,                  \
                                                                                       positionsDevice,             \
                                                                                       dGrads,                      \
                                                                                       statuses_.data(),            \
                                                                                       activeSystemIndices_.data(), \
                                                                                       dataDim_,                    \
                                                                                       fixedSteps_)
  if (usesFloatMinimizerState(precision_)) {
    if (usesFloatMinimizerCompute(precision_)) {
      if (usesFloatReduction(precision_))
        NVMOLKIT_LAUNCH_UPDATE_DGRAD(float, float, float, gradScalesFloat_.data(), scratchGradFloat_.data());
      else
        NVMOLKIT_LAUNCH_UPDATE_DGRAD(float, double, float, gradScalesFloat_.data(), scratchGradFloat_.data());
    } else if (usesFloatReduction(precision_))
      NVMOLKIT_LAUNCH_UPDATE_DGRAD(double, float, float, gradScalesFloat_.data(), scratchGradFloat_.data());
    else
      NVMOLKIT_LAUNCH_UPDATE_DGRAD(double, double, float, gradScalesFloat_.data(), scratchGradFloat_.data());
  } else if (usesFloatMinimizerCompute(precision_)) {
    if (usesFloatReduction(precision_))
      NVMOLKIT_LAUNCH_UPDATE_DGRAD(float, float, double, gradScales_.data(), scratchGrad_.data());
    else
      NVMOLKIT_LAUNCH_UPDATE_DGRAD(float, double, double, gradScales_.data(), scratchGrad_.data());
  } else {
    if (usesFloatReduction(precision_))
      NVMOLKIT_LAUNCH_UPDATE_DGRAD(double, float, double, gradScales_.data(), scratchGrad_.data());
    else
      NVMOLKIT_LAUNCH_UPDATE_DGRAD(double, double, double, gradScales_.data(), scratchGrad_.data());
  }
#undef NVMOLKIT_LAUNCH_UPDATE_DGRAD
  cudaCheckError(cudaGetLastError());
}

template <typename hessianT, typename stateT>
void updateHessianState(int            numUnfinishedSystems,
                        const int16_t* statuses,
                        const int*     hessianStarts,
                        const int*     atomStarts,
                        hessianT*      inverseHessian,
                        stateT*        dGrads,
                        stateT*        dirs,
                        stateT*        hessDGrads,
                        const double*  grads,
                        int            dataDim,
                        bool           largeMol,
                        const int*     activeSystemIndices,
                        cudaStream_t   stream,
                        bool           computeInFloat,
                        bool           reduceInFloat) {
  nvMolKit::updateInverseHessianBFGSBatch(numUnfinishedSystems,
                                          statuses,
                                          hessianStarts,
                                          atomStarts,
                                          inverseHessian,
                                          dGrads,
                                          dirs,
                                          hessDGrads,
                                          grads,
                                          dataDim,
                                          largeMol,
                                          activeSystemIndices,
                                          stream,
                                          computeInFloat,
                                          reduceInFloat);
}

void BfgsBatchMinimizer::updateHessian() {
  const ScopedNvtxRange bfgsUpdateHessian("BfgsBatchMinimizer::updateHessian");
  // Determine if any active system exceeds the shared-memory-optimized limit
  bool                  largeMol = hasLargeSystem_;
#define NVMOLKIT_UPDATE_HESSIAN(hessian, dGrads, dirs, hessDGrads) \
  updateHessianState(numUnfinishedSystems_,                        \
                     statuses_.data(),                             \
                     hessianStarts_.data(),                        \
                     atomStartsDevice,                             \
                     hessian,                                      \
                     dGrads,                                       \
                     dirs,                                         \
                     hessDGrads,                                   \
                     gradDevice,                                   \
                     dataDim_,                                     \
                     largeMol,                                     \
                     activeSystemIndices_.data(),                  \
                     stream_,                                      \
                     usesFloatMinimizerCompute(precision_),        \
                     usesFloatReduction(precision_))
  if (usesFloatMinimizerState(precision_)) {
    if (usesFloatHessian(precision_))
      NVMOLKIT_UPDATE_HESSIAN(inverseHessianFloat_.data(),
                              scratchGradFloat_.data(),
                              lineSearchDirFloat_.data(),
                              hessDGradFloat_.data());
    else
      NVMOLKIT_UPDATE_HESSIAN(inverseHessian_.data(),
                              scratchGradFloat_.data(),
                              lineSearchDirFloat_.data(),
                              hessDGradFloat_.data());
  } else if (usesFloatHessian(precision_))
    NVMOLKIT_UPDATE_HESSIAN(inverseHessianFloat_.data(), scratchGrad_.data(), lineSearchDir_.data(), hessDGrad_.data());
  else
    NVMOLKIT_UPDATE_HESSIAN(inverseHessian_.data(), scratchGrad_.data(), lineSearchDir_.data(), hessDGrad_.data());
#undef NVMOLKIT_UPDATE_HESSIAN
}

void BfgsBatchMinimizer::collectDebugData() {
  if (debugLevel_ != DebugLevel::STEPWISE) {
    return;
  }

  // Copy energies and statuses to host for debugging.
  std::vector<int16_t> statusesHost(numSystems_);
  std::vector<double>  energiesHost(numSystems_);
  cudaCheckError(cudaMemcpyAsync(statusesHost.data(),
                                 statuses_.data(),
                                 numSystems_ * sizeof(int16_t),
                                 cudaMemcpyDeviceToHost,
                                 stream_));
  cudaCheckError(cudaMemcpyAsync(energiesHost.data(),
                                 energyOutsDevice,
                                 numSystems_ * sizeof(double),
                                 cudaMemcpyDeviceToHost,
                                 stream_));
  cudaCheckError(cudaStreamSynchronize(stream_));

  stepwiseStatuses.push_back(std::move(statusesHost));
  stepwiseEnergies.push_back(std::move(energiesHost));
}

__global__ void incrementIterationCountsKernel(const int  numActive,
                                               const int* activeSystemIndices,
                                               int*       iterationCounts) {
  const int activeIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (activeIdx < numActive) {
    ++iterationCounts[activeSystemIndices[activeIdx]];
  }
}

bool BfgsBatchMinimizer::minimize(const int                  numIters,
                                  const double               gradTol,
                                  const std::vector<int>&    atomStartsHost,
                                  const int*                 atomStarts,
                                  AsyncDeviceVector<double>& positions,
                                  AsyncDeviceVector<double>& grad,
                                  AsyncDeviceVector<double>& energyOuts,
                                  EnergyFunctor              eFunc,
                                  GradFunctor                gFunc,
                                  const uint8_t*             activeThisStage) {
  gradTol_             = gradTol;
  const int numSystems = atomStartsHost.size() - 1;

  if (backend_ == BfgsBackend::PER_MOLECULE) {
    throw std::runtime_error(
      "PER_MOLECULE backend is only supported through the forcefield-specific entry points. "
      "Use minimizeWithMMFF(), minimizeWithETK(), minimizeWithDG(), or switch to BATCHED backend.");
  }

  {
    const ScopedNvtxRange bfgsFullInitialize("BfgsBatchMinimizer::fullInitialize");
    initialize(atomStartsHost,
               atomStarts,
               positions.data(),
               grad.data(),
               energyOuts.data(),
               BfgsBackend::BATCHED,
               activeThisStage);

    iterationCounts_.resize(numSystems);
    iterationCounts_.zero();

    setHessianToIdentity();

    energyOuts.zero();
    eFunc(nullptr);
    grad.zero();
    gFunc();
    scaleGrad(/*preLoop=*/true);

    collectDebugData();
    if (usesFloatMinimizerState(precision_))
      copyAndInvert(grad, lineSearchDirFloat_);
    else
      copyAndInvert(grad, lineSearchDir_);
    setMaxStep();
  }

  for (int currIter = 0; currIter < numIters && compactAndCountConverged() < numSystems; currIter++) {
    const int countBlocks = (numUnfinishedSystems_ + 127) / 128;
    incrementIterationCountsKernel<<<countBlocks, 128, 0, stream_>>>(numUnfinishedSystems_,
                                                                     activeSystemIndices_.data(),
                                                                     iterationCounts_.data());
    {
      const ScopedNvtxRange bfgsLineSearch("BfgsBatchMinimizer::lineSearch");
      doLineSearchSetup(energyOuts.data());

      int              lineSearchIter         = 0;
      constexpr double MAX_ITER_LINEAR_SEARCH = 1000;
      while (lineSearchIter < MAX_ITER_LINEAR_SEARCH && lineSearchCountFinished() < numSystems) {
        doLineSearchPerturb();
        energyOuts.zero();
        eFunc(scratchPositions_.data());
        doLineSearchPostEnergy(lineSearchIter);
        lineSearchIter++;
      }
      doLineSearchPostLoop();
    }
    setDirection();

    {
      const ScopedNvtxRange bfgsGetAndScaleGrad("BfgsBatchMinimizer::getAndScaleGrad");
      grad.zero();
      gFunc();
      scaleGrad(/*preLoop=*/false);
    }

    updateDGrad();
    updateHessian();
    collectDebugData();
  }

  energyOuts.zero();
  eFunc(nullptr);
  return compactAndCountConverged() == numSystems ? 0 : 1;
}

bool BfgsBatchMinimizer::minimize(const int                  numIters,
                                  const double               gradTol,
                                  BatchedForcefield&         ff,
                                  AsyncDeviceVector<double>& positions,
                                  AsyncDeviceVector<double>& grad,
                                  AsyncDeviceVector<double>& energyOuts,
                                  const uint8_t*             activeSystemMask) {
  const auto& atomStartsHost = ff.atomStartsHost();

  if (resolveBackend(atomStartsHost) != BfgsBackend::BATCHED) {
    throw std::runtime_error("BatchedForcefield minimization is only supported on the BATCHED backend");
  }

  auto eFunc = [&](const double* evalPositions) {
    const double* positionsToEvaluate = evalPositions != nullptr ? evalPositions : positions.data();
    ff.computeEnergy(energyOuts.data(), positionsToEvaluate, activeSystemMask, stream_);
  };
  auto gFunc = [&]() { ff.computeGradients(grad.data(), positions.data(), activeSystemMask, stream_); };

  return minimize(numIters,
                  gradTol,
                  atomStartsHost,
                  ff.atomStartsDevice(),
                  positions,
                  grad,
                  energyOuts,
                  eFunc,
                  gFunc,
                  activeSystemMask);
}

template <typename DeviceBuffers>
bool BfgsBatchMinimizer::minimizeWithMMFFImpl(const int               numIters,
                                              const double            gradTol,
                                              const std::vector<int>& atomStartsHost,
                                              DeviceBuffers&          systemDevice,
                                              const uint8_t*          activeThisStage) {
  const int         numSystems       = atomStartsHost.size() - 1;
  const BfgsBackend effectiveBackend = resolveBackend(atomStartsHost);

  if (effectiveBackend == BfgsBackend::BATCHED) {
    throw std::runtime_error("Use minimize(..., BatchedForcefield&) for batched MMFF minimization");
  }

  initialize(atomStartsHost,
             systemDevice.indices.atomStarts.data(),
             systemDevice.positions.data(),
             systemDevice.grad.data(),
             systemDevice.energyOuts.data(),
             effectiveBackend,
             activeThisStage);

  setHessianToIdentity();

  const ScopedNvtxRange bfgsPerMolecule("BfgsBatchMinimizer::perMoleculeMinimize");

  prepareScratchBuffers(systemDevice.grad,
                        lineSearchDir_,
                        scratchPositions_,
                        hessDGrad_,
                        scratchGrad_,
                        scratchBuffersDevice_,
                        scratchBufferPointersHost_,
                        stream_);

  auto terms         = MMFF::toEnergyForceContribsDevicePtr(systemDevice);
  auto systemIndices = MMFF::toBatchedIndicesDevicePtr(systemDevice);

  cudaError_t err;
  if (usesFloatHessian(precision_)) {
    err = launchBfgsMinimizePerMolKernel(static_cast<int>(activeMolIds_.size()),
                                         activeMolIdsDevice_.data(),
                                         maxAtomsInBatch_,
                                         systemDevice.indices.atomStarts.data(),
                                         hessianStarts_.data(),
                                         numIters,
                                         gradTol,
                                         scaleGrads_,
                                         terms,
                                         systemIndices,
                                         systemDevice.positions.data(),
                                         systemDevice.grad.data(),
                                         inverseHessianFloat_.data(),
                                         scratchBuffersDevice_.data(),
                                         systemDevice.energyOuts.data(),
                                         MMFF::batchHasConstraints(systemDevice.contribs),
                                         statuses_.data(),
                                         stream_);
  } else {
    err = launchBfgsMinimizePerMolKernel(static_cast<int>(activeMolIds_.size()),
                                         activeMolIdsDevice_.data(),
                                         maxAtomsInBatch_,
                                         systemDevice.indices.atomStarts.data(),
                                         hessianStarts_.data(),
                                         numIters,
                                         gradTol,
                                         scaleGrads_,
                                         terms,
                                         systemIndices,
                                         systemDevice.positions.data(),
                                         systemDevice.grad.data(),
                                         inverseHessian_.data(),
                                         scratchBuffersDevice_.data(),
                                         systemDevice.energyOuts.data(),
                                         MMFF::batchHasConstraints(systemDevice.contribs),
                                         statuses_.data(),
                                         stream_);
  }

  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Per-molecule BFGS kernel failed: ") + cudaGetErrorString(err));
  }

  return checkConvergence(activeMolIds_, statuses_, convergenceHost_, numSystems, stream_);
}

bool BfgsBatchMinimizer::minimizeWithMMFF(const int                            numIters,
                                          const double                         gradTol,
                                          const std::vector<int>&              atomStartsHost,
                                          MMFF::BatchedMolecularDeviceBuffers& systemDevice,
                                          const uint8_t*                       activeThisStage) {
  return minimizeWithMMFFImpl(numIters, gradTol, atomStartsHost, systemDevice, activeThisStage);
}

bool BfgsBatchMinimizer::minimizeWithMMFF(const int                                     numIters,
                                          const double                                  gradTol,
                                          const std::vector<int>&                       atomStartsHost,
                                          MMFF::BatchedMolecularDeviceBuffersF32Params& systemDevice,
                                          const uint8_t*                                activeThisStage) {
  return minimizeWithMMFFImpl(numIters, gradTol, atomStartsHost, systemDevice, activeThisStage);
}

bool BfgsBatchMinimizer::minimizeWithETK(const int                                  numIters,
                                         const double                               gradTol,
                                         const std::vector<int>&                    atomStartsHost,
                                         const AsyncDeviceVector<int>&              atomStarts,
                                         AsyncDeviceVector<double>&                 positions,
                                         DistGeom::BatchedMolecular3DDeviceBuffers& systemDevice,
                                         const uint8_t*                             activeThisStage) {
  const int         numSystems       = atomStartsHost.size() - 1;
  const BfgsBackend effectiveBackend = resolveBackend(atomStartsHost);

  if (effectiveBackend == BfgsBackend::BATCHED) {
    throw std::runtime_error("Use minimize(..., BatchedForcefield&) for batched ETK minimization");
  }

  initialize(atomStartsHost,
             atomStarts.data(),
             positions.data(),
             systemDevice.grad.data(),
             systemDevice.energyOuts.data(),
             effectiveBackend,
             activeThisStage);

  setHessianToIdentity();

  const ScopedNvtxRange bfgsPerMoleculeETK("BfgsBatchMinimizer::perMoleculeMinimizeETK");

  prepareScratchBuffers(systemDevice.grad,
                        lineSearchDir_,
                        scratchPositions_,
                        hessDGrad_,
                        scratchGrad_,
                        scratchBuffersDevice_,
                        scratchBufferPointersHost_,
                        stream_);

  auto terms         = DistGeom::toEnergy3DForceContribsDevicePtr(systemDevice);
  auto systemIndices = DistGeom::toBatchedIndices3DDevicePtr(systemDevice, atomStarts.data());

  cudaError_t err;
  if (usesFloatHessian(precision_)) {
    err = launchBfgsMinimizePerMolKernelETK(static_cast<int>(activeMolIds_.size()),
                                            activeMolIdsDevice_.data(),
                                            maxAtomsInBatch_,
                                            atomStarts.data(),
                                            hessianStarts_.data(),
                                            numIters,
                                            gradTol,
                                            scaleGrads_,
                                            terms,
                                            systemIndices,
                                            positions.data(),
                                            systemDevice.grad.data(),
                                            inverseHessianFloat_.data(),
                                            scratchBuffersDevice_.data(),
                                            systemDevice.energyOuts.data(),
                                            statuses_.data(),
                                            stream_);
  } else {
    err = launchBfgsMinimizePerMolKernelETK(static_cast<int>(activeMolIds_.size()),
                                            activeMolIdsDevice_.data(),
                                            maxAtomsInBatch_,
                                            atomStarts.data(),
                                            hessianStarts_.data(),
                                            numIters,
                                            gradTol,
                                            scaleGrads_,
                                            terms,
                                            systemIndices,
                                            positions.data(),
                                            systemDevice.grad.data(),
                                            inverseHessian_.data(),
                                            scratchBuffersDevice_.data(),
                                            systemDevice.energyOuts.data(),
                                            statuses_.data(),
                                            stream_);
  }

  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Per-molecule BFGS ETK kernel failed: ") + cudaGetErrorString(err));
  }

  return checkConvergence(activeMolIds_, statuses_, convergenceHost_, numSystems, stream_);
}

bool BfgsBatchMinimizer::minimizeWithDG(const int                                numIters,
                                        const double                             gradTol,
                                        const std::vector<int>&                  atomStartsHost,
                                        const AsyncDeviceVector<int>&            atomStarts,
                                        AsyncDeviceVector<double>&               positions,
                                        DistGeom::BatchedMolecularDeviceBuffers& systemDevice,
                                        double                                   chiralWeight,
                                        double                                   fourthDimWeight,
                                        const uint8_t*                           activeThisStage) {
  const int numSystems = atomStartsHost.size() - 1;

  if (dataDim_ != 4) {
    throw std::runtime_error("minimizeWithDG requires BfgsBatchMinimizer to be constructed with dataDim=4");
  }

  const BfgsBackend effectiveBackend = resolveBackend(atomStartsHost);

  if (effectiveBackend == BfgsBackend::BATCHED) {
    throw std::runtime_error("Use minimize(..., BatchedForcefield&) for batched DG minimization");
  }

  initialize(atomStartsHost,
             atomStarts.data(),
             positions.data(),
             systemDevice.grad.data(),
             systemDevice.energyOuts.data(),
             effectiveBackend,
             activeThisStage);

  setHessianToIdentity();

  const ScopedNvtxRange bfgsPerMoleculeDG("BfgsBatchMinimizer::perMoleculeMinimizeDG");

  prepareScratchBuffers(systemDevice.grad,
                        lineSearchDir_,
                        scratchPositions_,
                        hessDGrad_,
                        scratchGrad_,
                        scratchBuffersDevice_,
                        scratchBufferPointersHost_,
                        stream_);

  auto terms         = DistGeom::toEnergyForceContribsDevicePtr(systemDevice);
  auto systemIndices = DistGeom::toBatchedIndicesDevicePtr(systemDevice, atomStarts.data());

  cudaError_t err;
  if (usesFloatHessian(precision_)) {
    err = launchBfgsMinimizePerMolKernelDG(static_cast<int>(activeMolIds_.size()),
                                           activeMolIdsDevice_.data(),
                                           maxAtomsInBatch_,
                                           atomStarts.data(),
                                           hessianStarts_.data(),
                                           numIters,
                                           gradTol,
                                           scaleGrads_,
                                           terms,
                                           systemIndices,
                                           positions.data(),
                                           systemDevice.grad.data(),
                                           inverseHessianFloat_.data(),
                                           scratchBuffersDevice_.data(),
                                           systemDevice.energyOuts.data(),
                                           chiralWeight,
                                           fourthDimWeight,
                                           statuses_.data(),
                                           stream_);
  } else {
    err = launchBfgsMinimizePerMolKernelDG(static_cast<int>(activeMolIds_.size()),
                                           activeMolIdsDevice_.data(),
                                           maxAtomsInBatch_,
                                           atomStarts.data(),
                                           hessianStarts_.data(),
                                           numIters,
                                           gradTol,
                                           scaleGrads_,
                                           terms,
                                           systemIndices,
                                           positions.data(),
                                           systemDevice.grad.data(),
                                           inverseHessian_.data(),
                                           scratchBuffersDevice_.data(),
                                           systemDevice.energyOuts.data(),
                                           chiralWeight,
                                           fourthDimWeight,
                                           statuses_.data(),
                                           stream_);
  }

  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Per-molecule BFGS DG kernel failed: ") + cudaGetErrorString(err));
  }

  return checkConvergence(activeMolIds_, statuses_, convergenceHost_, numSystems, stream_);
}

template <typename stateT>
void copyAndInvertImpl(const AsyncDeviceVector<double>& src, AsyncDeviceVector<stateT>& dst) {
  const size_t numElements = src.size();
  cudaStream_t stream      = dst.stream();
  if (numElements == 0) {
    return;
  }
  if (dst.size() != numElements) {
    throw std::runtime_error("Destination vector size does not match source vector size:" +
                             std::to_string(numElements) + " vs " + std::to_string(dst.size()));
  }
  const int blockSize = 128;
  const int numBlocks = (numElements + blockSize - 1) / blockSize;
  copyAndNegate<<<numBlocks, blockSize, 0, stream>>>(numElements, src.data(), dst.data());
  cudaCheckError(cudaGetLastError());
}
void copyAndInvert(const AsyncDeviceVector<double>& src, AsyncDeviceVector<double>& dst) {
  copyAndInvertImpl(src, dst);
}
void copyAndInvert(const AsyncDeviceVector<double>& src, AsyncDeviceVector<float>& dst) {
  copyAndInvertImpl(src, dst);
}
}  // namespace nvMolKit
