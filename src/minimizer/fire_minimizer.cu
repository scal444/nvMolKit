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
#include <cstdlib>
#include <cub/cub.cuh>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "src/forcefields/batched_forcefield.h"
#include "src/forcefields/mmff.h"
#include "src/forcefields/mmff_kernels.h"
#include "src/minimizer/fire_minimize_permol_kernels.h"
#include "src/minimizer/fire_minimizer.h"
#include "src/utils/nvtx.h"

namespace nvMolKit {

namespace {

constexpr int kFireBlockSize = 256;

// Acceleration conversion factor:
//   1 kcal/mol/Å applied to 1 amu produces 4.184 * 100 Å/ps^2
// Derivation: 1 kcal/mol = 4184 J/mol -> 6.9477e-21 J/molecule.
//             F = 6.9477e-21 / 1e-10 m  = 6.9477e-11 N.
//             a = F / m = 6.9477e-11 N / 1.66054e-27 kg = 4.184e16 m/s^2.
//             Convert m/s^2 -> Å/ps^2: multiply by 1e10/1e24 = 1e-14.
//             a = 418.4 Å/ps^2.
constexpr double kForceKcalMolPerAng_PerAmu_to_AngPerPs2 = 4.184 * 100.0;

template <typename T> __global__ void setAllKernel(const int numElements, const T value, T* dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = value;
  }
}

template <typename T> void setAll(AsyncDeviceVector<T>& vec, const T& value) {
  const int numElements = static_cast<int>(vec.size());
  if (numElements == 0) {
    return;
  }
  constexpr int blockSize = 128;
  const int     numBlocks = (numElements + blockSize - 1) / blockSize;
  setAllKernel<<<numBlocks, blockSize, 0, vec.stream()>>>(numElements, value, vec.data());
  cudaCheckError(cudaGetLastError());
}

template <typename T>
__device__ __forceinline__ cuda::std::span<T> getSystemSpan(const cuda::std::span<T>         data,
                                                            const cuda::std::span<const int> atomStarts,
                                                            const int                        sysIdx,
                                                            const int                        dataDim) {
  return data.subspan(atomStarts[sysIdx] * dataDim, (atomStarts[sysIdx + 1] - atomStarts[sysIdx]) * dataDim);
}

template <typename real> __device__ __forceinline__ real fireSqrt(real value) {
  if constexpr (cuda::std::is_same_v<real, float>)
    return sqrtf(value);
  else
    return sqrt(value);
}

template <typename real> __device__ __forceinline__ real firePow(real base, int exponent) {
  if constexpr (cuda::std::is_same_v<real, float>)
    return powf(base, static_cast<float>(exponent));
  else
    return pow(base, static_cast<double>(exponent));
}

//! Packed read-only kernel parameters.
struct FireKernelParams {
  double dtIncrementFactor;
  double dtDecrementFactor;
  double minDt;
  double maxDt;
  double dMax;
  double alphaStart;
  double alphaDecrementFactor;
  double gradTol;
  int    nMinForIncrease;
};

//! \brief Pre-kick stage of FIRE 2.0. One block per system.
//!
//! Reads the gradient at the *current* position, runs the convergence check, runs the
//! dt/alpha/nsteps state machine, and applies the half-step-back-with-velocity-zero
//! when power < 0. Does not touch positions otherwise. After this kernel runs the
//! caller must re-evaluate the gradient at the new positions before invoking
//! ::firePostKickKernel. This split mirrors ASE FIRE2 which evaluates forces twice per
//! iteration (once before the state machine and again after any half-step-back).
template <typename real, typename reduceT, typename storageT>
__global__ void firePreKickKernel(const cuda::std::span<const int>    atomStarts,
                                  const cuda::std::span<const int>    activeIndices,
                                  const cuda::std::span<double>       x,
                                  const cuda::std::span<storageT>     v,
                                  const cuda::std::span<const double> f,
                                  const int                           dataDim,
                                  const cuda::std::span<storageT>     alphas,
                                  const cuda::std::span<storageT>     dts,
                                  const cuda::std::span<int>          nStepsPositive,
                                  const FireKernelParams              params,
                                  const bool                          takeHalfStepBack,
                                  const bool                          isFirstStep,
                                  uint8_t*                            statuses,
                                  uint8_t*                            convergeReason,
                                  const cuda::std::span<double>       debugPowers) {
  using BlockReduce = cub::BlockReduce<reduceT, kFireBlockSize>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ reduceT                           sharedScalar0;
  __shared__ reduceT                           sharedScalar1;
  __shared__ real                              sharedDt;
  __shared__ bool                              sharedConverged;

  const int sysIdx = activeIndices[blockIdx.x];
  if (statuses[sysIdx] == 0) {
    return;
  }

  const auto vSys = getSystemSpan(v, atomStarts, sysIdx, dataDim);
  const auto fSys = getSystemSpan(f, atomStarts, sysIdx, dataDim);
  const auto xSys = getSystemSpan(x, atomStarts, sysIdx, dataDim);

  if (threadIdx.x == 0) {
    sharedDt        = dts[sysIdx];
    sharedConverged = false;
  }
  __syncthreads();

  const real dtIn    = sharedDt;
  const real alphaIn = static_cast<real>(alphas[sysIdx]);
  const int  nstepIn = nStepsPositive[sysIdx];

  reduceT power  = 0;
  reduceT gradSq = 0;
  for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
    const real fi = static_cast<real>(fSys[i]);
    if (!isFirstStep) {
      power += static_cast<reduceT>(static_cast<real>(vSys[i]) * -fi);
    }
    gradSq += static_cast<reduceT>(fi * fi);
  }
  reduceT powerSum = 0;
  if (!isFirstStep) {
    powerSum = BlockReduce(tempStorage).Sum(power);
    __syncthreads();
  }
  const reduceT gradSqSum = BlockReduce(tempStorage).Sum(gradSq);
  if (threadIdx.x == 0) {
    sharedScalar0 = powerSum;
    sharedScalar1 = gradSqSum;
  }
  __syncthreads();
  const reduceT powerShared  = sharedScalar0;
  const reduceT gradSqShared = sharedScalar1;

  if (threadIdx.x == 0) {
    if (fireSqrt(gradSqShared) <= static_cast<reduceT>(params.gradTol)) {
      sharedConverged        = true;
      statuses[sysIdx]       = 0;
      convergeReason[sysIdx] = 1;
    }
  }
  __syncthreads();
  if (sharedConverged) {
    return;
  }

  if (threadIdx.x == 0 && !isFirstStep) {
    if (!debugPowers.empty()) {
      debugPowers[sysIdx] = powerShared;
    }

    real newDt     = dtIn;
    real newAlpha  = alphaIn;
    int  newNsteps = nstepIn;

    if (powerShared >= 0.0) {
      newNsteps = nstepIn + 1;
      if (newNsteps > params.nMinForIncrease) {
        newDt    = min(dtIn * static_cast<real>(params.dtIncrementFactor), static_cast<real>(params.maxDt));
        newAlpha = alphaIn * static_cast<real>(params.alphaDecrementFactor);
      }
    } else {
      newNsteps = 0;
      newAlpha  = static_cast<real>(params.alphaStart);
      newDt     = max(dtIn * static_cast<real>(params.dtDecrementFactor), static_cast<real>(params.minDt));
    }

    sharedDt               = newDt;
    dts[sysIdx]            = newDt;
    alphas[sysIdx]         = newAlpha;
    nStepsPositive[sysIdx] = newNsteps;
  }
  __syncthreads();

  const real dt = sharedDt;

  // Negative-power half-step-back must read the freshly-updated nstepsPositive=0 marker
  // to decide if it fires. Equivalently: it fires iff power < 0 on a non-first step.
  const bool negative = !isFirstStep && (powerShared < 0.0);
  if (negative) {
    for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
      if (takeHalfStepBack) {
        xSys[i] = static_cast<real>(xSys[i]) - real{0.5} * dt * static_cast<real>(vSys[i]);
      }
      vSys[i] = 0.0;
    }
    __syncthreads();
  }
}

//! \brief Post-kick stage of FIRE 2.0. One block per system.
//!
//! Reads the *new* gradient at positions produced by ::firePreKickKernel and applies the
//! semi-implicit Euler kick (v += dt*F), the FIRE mixer, and the displacement clip /
//! position update. Per-system dt/alpha/nstepsPositive are read back from the device
//! buffers populated by the pre-kick.
template <typename real, typename reduceT, typename storageT>
__global__ void firePostKickKernel(const cuda::std::span<const int>      atomStarts,
                                   const cuda::std::span<const int>      activeIndices,
                                   const cuda::std::span<double>         x,
                                   const cuda::std::span<storageT>       v,
                                   const cuda::std::span<const double>   f,
                                   const cuda::std::span<const double>   masses,
                                   const int                             dataDim,
                                   const cuda::std::span<const storageT> alphas,
                                   const cuda::std::span<const storageT> dts,
                                   const cuda::std::span<const int>      nStepsPositive,
                                   const FireKernelParams                params,
                                   const bool                            useAbc,
                                   const uint8_t*                        statuses) {
  using BlockReduce = cub::BlockReduce<reduceT, kFireBlockSize>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ reduceT                           sharedScalar0;

  const int sysIdx = activeIndices[blockIdx.x];
  if (statuses[sysIdx] == 0) {
    return;
  }

  const auto vSys = getSystemSpan(v, atomStarts, sysIdx, dataDim);
  const auto fSys = getSystemSpan(f, atomStarts, sysIdx, dataDim);
  const auto xSys = getSystemSpan(x, atomStarts, sysIdx, dataDim);

  cuda::std::span<const double> massSys;
  const bool                    massEnabled = !masses.empty();
  if (massEnabled) {
    const int atomStart = atomStarts[sysIdx];
    const int atomCount = atomStarts[sysIdx + 1] - atomStart;
    massSys             = masses.subspan(atomStart, atomCount);
  }

  const real dt     = static_cast<real>(dts[sysIdx]);
  const real alpha  = static_cast<real>(alphas[sysIdx]);
  const int  nsteps = nStepsPositive[sysIdx];

  // Kick: v += dt * F. With mass weighting (real MD-style integration), F is converted
  // to physical Å/ps^2 via kForceKcalMolPerAng_PerAmu_to_AngPerPs2 and divided by
  // per-atom mass. Without mass weighting we mirror ASE FIRE2: v += dt * F where F is
  // the raw force in the calculator's native units (no implicit unit conversion).
  reduceT vSqAccum    = 0;
  reduceT gradSqAccum = 0;
  for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
    real accel;
    if (massEnabled) {
      const real accelMag = -static_cast<real>(fSys[i]) * static_cast<real>(kForceKcalMolPerAng_PerAmu_to_AngPerPs2);
      const int  coordIdx = i / dataDim;
      accel               = accelMag / static_cast<real>(massSys[coordIdx]);
    } else {
      accel = -static_cast<real>(fSys[i]);
    }
    const real newV = static_cast<real>(vSys[i]) + dt * accel;
    vSys[i]         = newV;
    vSqAccum += static_cast<reduceT>(newV * newV);
    const real fi = static_cast<real>(fSys[i]);
    gradSqAccum += static_cast<reduceT>(fi * fi);
  }
  const reduceT vSqReduced = BlockReduce(tempStorage).Sum(vSqAccum);
  if (threadIdx.x == 0) {
    sharedScalar0 = vSqReduced;
  }
  __syncthreads();
  const reduceT vSqSum        = sharedScalar0;
  const reduceT gradSqReduced = BlockReduce(tempStorage).Sum(gradSqAccum);
  if (threadIdx.x == 0) {
    sharedScalar0 = gradSqReduced;
  }
  __syncthreads();
  const reduceT gradSqSum = sharedScalar0;

  // Mixer: v = (1 - alpha) * v + alpha * |v|/|grad| * (-grad). With ABC,
  // multiply by 1 / (1 - (1 - alpha)^(N+1)).
  const real mixCoef1 = real{1} - alpha;
  const real mixCoef2 =
    (gradSqSum > reduceT{1e-30}) ? (alpha * static_cast<real>(fireSqrt(vSqSum) / fireSqrt(gradSqSum))) : real{0};
  real abcMult = 1;
  if (useAbc) {
    const real oneMinusA = real{1} - max(alpha, real{1e-10});
    const real pow_term  = firePow(oneMinusA, nsteps + 1);
    const real denom     = real{1} - pow_term;
    abcMult              = (denom > real{1e-30}) ? (real{1} / denom) : real{1};
  }

  for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
    const real vMix = mixCoef1 * static_cast<real>(vSys[i]) + mixCoef2 * (-static_cast<real>(fSys[i]));
    vSys[i]         = abcMult * vMix;
  }
  __syncthreads();

  // Displacement handling: ABC clips per-component v to ±dMax/dt, non-ABC
  // norm-clips dr (without modifying v) before the position update.
  real drScale = 1;
  if (useAbc) {
    if (params.dMax > 0.0) {
      const real maxV = static_cast<real>(params.dMax) / dt;
      for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
        const real clamped = max(-maxV, min(maxV, static_cast<real>(vSys[i])));
        vSys[i]            = clamped;
      }
      __syncthreads();
    }
  } else {
    if (params.dMax > 0.0) {
      reduceT drSqAccum = 0;
      for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
        const real dri = dt * static_cast<real>(vSys[i]);
        drSqAccum += static_cast<reduceT>(dri * dri);
      }
      const reduceT drSqReduced = BlockReduce(tempStorage).Sum(drSqAccum);
      if (threadIdx.x == 0) {
        sharedScalar0 = drSqReduced;
      }
      __syncthreads();
      const real drNorm = static_cast<real>(fireSqrt(sharedScalar0));
      if (drNorm > params.dMax) {
        drScale = params.dMax / drNorm;
      }
    }
  }

  for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
    xSys[i] = static_cast<real>(xSys[i]) + drScale * dt * static_cast<real>(vSys[i]);
  }
}

//! \brief Energy-plateau stuck detection.
//!
//! Launched with one block per active system; each block reads its own scalar energy and
//! per-system streak state. When the windowed extrema relative spread falls below
//! @p relTol, the streak counter increments; otherwise the window resets to the current
//! sample. Reaching @p streakLimit declares the system converged (status = 0).
__global__ void fireStuckCheckKernel(cuda::std::span<const int>    activeSystemIndices,
                                     cuda::std::span<const double> energies,
                                     cuda::std::span<double>       energyMinStreak,
                                     cuda::std::span<double>       energyMaxStreak,
                                     cuda::std::span<int32_t>      stuckStreak,
                                     uint8_t*                      statuses,
                                     uint8_t*                      convergeReason,
                                     const double                  relTol,
                                     const int                     streakLimit) {
  if (threadIdx.x != 0) {
    return;
  }
  const int sysIdx = activeSystemIndices[blockIdx.x];
  if (statuses[sysIdx] == 0) {
    return;
  }
  const double energy = energies[sysIdx];
  double       newMin = fmin(energyMinStreak[sysIdx], energy);
  double       newMax = fmax(energyMaxStreak[sysIdx], energy);
  const double denom  = fmax(fabs(energy), 1.0);
  if ((newMax - newMin) <= relTol * denom) {
    const int32_t streak = stuckStreak[sysIdx] + 1;
    stuckStreak[sysIdx]  = streak;
    if (streak >= streakLimit) {
      statuses[sysIdx]       = 0;
      convergeReason[sysIdx] = 2;
    }
    energyMinStreak[sysIdx] = newMin;
    energyMaxStreak[sysIdx] = newMax;
  } else {
    stuckStreak[sysIdx]     = 1;
    energyMinStreak[sysIdx] = energy;
    energyMaxStreak[sysIdx] = energy;
  }
}

}  // namespace

FireBatchMinimizer::FireBatchMinimizer(const int          dataDim,
                                       const FireOptions& options,
                                       cudaStream_t       stream,
                                       const bool         debugMode,
                                       const FireBackend  backend,
                                       PrecisionOptions   precision)
    : dataDim_(dataDim),
      fireOptions_(options),
      stream_(stream),
      debugMode_(debugMode),
      backend_((usesFloatForcefieldCoordinates(precision) || usesFloatMinimizerCompute(precision) ||
                usesFloatForcefieldCompute(precision)) ?
                 FireBackend::BATCHED :
                 backend),
      precision_(precision) {
  velocities_.setStream(stream_);
  velocitiesFloat_.setStream(stream_);
  statuses_.setStream(stream_);
  dt_.setStream(stream_);
  alpha_.setStream(stream_);
  dtFloat_.setStream(stream_);
  alphaFloat_.setStream(stream_);
  allSystemIndices_.setStream(stream_);
  activeSystemIndices_.setStream(stream_);
  numStepsWithPositivePower_.setStream(stream_);
  countUnfinished_.setStream(stream_);
  countTempStorage_.setStream(stream_);
  masses_.setStream(stream_);
  debugPowers_.setStream(stream_);
  energyMinStreak_.setStream(stream_);
  energyMaxStreak_.setStream(stream_);
  stuckStreak_.setStream(stream_);
  convergeReason_.setStream(stream_);
  activeMolIdsDevice_.setStream(stream_);
  loopStatusHost_.resize(1);
  loopStatusHost_[0] = 0;
}

FireBackend FireBatchMinimizer::resolveBackend(const std::vector<int>& atomStartsHost) const {
  if (backend_ != FireBackend::HYBRID) {
    return backend_;
  }
  for (size_t i = 0; i + 1 < atomStartsHost.size(); ++i) {
    if (atomStartsHost[i + 1] - atomStartsHost[i] > kHybridFireBackendAtomThreshold) {
      return FireBackend::BATCHED;
    }
  }
  return FireBackend::PER_MOLECULE;
}

void FireBatchMinimizer::setMasses(const std::vector<double>& masses) {
  hostMasses_ = masses;
}

void FireBatchMinimizer::resetContinuationCache() {
  hasInitializedBatch_   = false;
  cachedNumSystems_      = -1;
  cachedTotalAtoms_      = -1;
  cachedActiveThisStage_ = nullptr;
  cachedMasses_          = nullptr;
}

void FireBatchMinimizer::setConvergencePollInterval(const int interval) {
  if (interval < 1) {
    throw std::invalid_argument("FireBatchMinimizer poll interval must be >= 1");
  }
  convergencePollInterval_ = interval;
}

void FireBatchMinimizer::initialize(const std::vector<int>& atomStartsHost,
                                    const double*           masses,
                                    const uint8_t*          activeThisStage,
                                    const FireBackend       effectiveBackend) {
  step_                = 0;
  const int totalAtoms = atomStartsHost.back();
  const int numSystems = static_cast<int>(atomStartsHost.size()) - 1;

  // Continuation cache only applies to the BATCHED backend; the per-mol path
  // resets per-system state on every call.
  const bool isContinuation = effectiveBackend == FireBackend::BATCHED && hasInitializedBatch_ &&
                              cachedNumSystems_ == numSystems && cachedTotalAtoms_ == totalAtoms &&
                              cachedActiveThisStage_ == activeThisStage && cachedMasses_ == masses;

  numSystems_ = numSystems;

  if (usesFloatMinimizerState(precision_)) {
    velocities_.resize(0);
    velocitiesFloat_.resize(static_cast<size_t>(totalAtoms) * dataDim_);
    velocitiesFloat_.zero();
  } else {
    velocitiesFloat_.resize(0);
    velocities_.resize(static_cast<size_t>(totalAtoms) * dataDim_);
    velocities_.zero();
  }

  if (fireOptions_.useMass && masses != nullptr) {
    masses_.resize(totalAtoms);
    cudaCheckError(cudaMemcpyAsync(masses_.data(), masses, totalAtoms * sizeof(double), cudaMemcpyDefault, stream_));
  } else if (fireOptions_.useMass && !hostMasses_.empty()) {
    if (hostMasses_.size() != static_cast<size_t>(totalAtoms)) {
      throw std::runtime_error("Stored masses size does not match atom count");
    }
    masses_.setFromVector(hostMasses_);
  } else {
    masses_.resize(0);
  }

  statuses_.resize(numSystems);
  if (!isContinuation) {
    if (activeThisStage != nullptr) {
      cudaCheckError(
        cudaMemcpyAsync(statuses_.data(), activeThisStage, numSystems * sizeof(uint8_t), cudaMemcpyDefault, stream_));
    } else {
      setAll(statuses_, static_cast<uint8_t>(1));
    }
  }

  numStepsWithPositivePower_.resize(numSystems);
  numStepsWithPositivePower_.zero();
  if (usesFloatMinimizerState(precision_)) {
    alpha_.resize(0);
    dt_.resize(0);
    alphaFloat_.resize(numSystems);
    setAll(alphaFloat_, static_cast<float>(fireOptions_.alphaInit));
    dtFloat_.resize(numSystems);
    setAll(dtFloat_, static_cast<float>(fireOptions_.dtInit));
  } else {
    alphaFloat_.resize(0);
    dtFloat_.resize(0);
    alpha_.resize(numSystems);
    setAll(alpha_, fireOptions_.alphaInit);
    dt_.resize(numSystems);
    setAll(dt_, fireOptions_.dtInit);
  }

  if (effectiveBackend == FireBackend::PER_MOLECULE) {
    energyMinStreak_.resize(0);
    energyMaxStreak_.resize(0);
    stuckStreak_.resize(0);
    convergeReason_.resize(0);
    debugPowers_.resize(0);
    debugOutputs_.clear();
    activeSystemIndices_.resize(0);
    allSystemIndices_.resize(0);
    pollsSinceLastEnergyEval_ = 0;

    activeHost_.resize(numSystems);
    convergenceHost_.resize(numSystems);
    std::fill_n(activeHost_.begin(), numSystems, 1);
    if (activeThisStage != nullptr) {
      cudaCheckError(cudaMemcpyAsync(activeHost_.data(),
                                     activeThisStage,
                                     numSystems * sizeof(uint8_t),
                                     cudaMemcpyDeviceToHost,
                                     stream_));
      cudaCheckError(cudaStreamSynchronize(stream_));
    }
    activeMolIds_.clear();
    maxAtomsInBatch_ = 0;
    for (int sysIdx = 0; sysIdx < numSystems; ++sysIdx) {
      if (activeHost_[sysIdx] == 0) {
        continue;
      }
      activeMolIds_.push_back(sysIdx);
      const int numAtoms = atomStartsHost[sysIdx + 1] - atomStartsHost[sysIdx];
      if (numAtoms > maxAtomsInBatch_) {
        maxAtomsInBatch_ = numAtoms;
      }
    }
    if (!activeMolIds_.empty()) {
      activeMolIdsDevice_.resize(activeMolIds_.size());
      activeMolIdsDevice_.setFromVector(activeMolIds_);
    }

    hasInitializedBatch_    = false;
    cachedNumSystems_       = -1;
    cachedTotalAtoms_       = -1;
    cachedActiveThisStage_  = nullptr;
    cachedMasses_           = nullptr;
    lastKnownNumUnfinished_ = static_cast<int>(activeMolIds_.size());
    return;
  }

  activeSystemIndices_.resize(numSystems);
  allSystemIndices_.resize(numSystems);
  std::vector<int> indicesHost(numSystems);
  std::iota(indicesHost.begin(), indicesHost.end(), 0);
  allSystemIndices_.setFromVector(indicesHost);
  activeSystemIndices_.setFromVector(indicesHost);

  if (fireOptions_.stuckDetectionEnabled) {
    energyMinStreak_.resize(numSystems);
    energyMaxStreak_.resize(numSystems);
    stuckStreak_.resize(numSystems);
    if (!isContinuation) {
      setAll(energyMinStreak_, std::numeric_limits<double>::infinity());
      setAll(energyMaxStreak_, -std::numeric_limits<double>::infinity());
      stuckStreak_.zero();
    }
  } else {
    energyMinStreak_.resize(0);
    energyMaxStreak_.resize(0);
    stuckStreak_.resize(0);
  }
  pollsSinceLastEnergyEval_ = 0;

  convergeReason_.resize(numSystems);
  if (!isContinuation) {
    convergeReason_.zero();
  }

  hasInitializedBatch_   = true;
  cachedNumSystems_      = numSystems;
  cachedTotalAtoms_      = totalAtoms;
  cachedActiveThisStage_ = activeThisStage;
  cachedMasses_          = masses;

  size_t tempStorageBytes = 0;
  cudaCheckError(cub::DeviceSelect::Flagged(nullptr,
                                            tempStorageBytes,
                                            allSystemIndices_.data(),
                                            statuses_.data(),
                                            activeSystemIndices_.data(),
                                            countUnfinished_.data(),
                                            allSystemIndices_.size(),
                                            stream_));
  if (tempStorageBytes > countTempStorage_.size()) {
    countTempStorage_.resize(tempStorageBytes);
  }

  // Filter activeSystemIndices_ to the initial mask so the first kernel launch
  // already skips any caller-marked-inactive systems and uses the right block count.
  compactActiveAsync();
  lastKnownNumUnfinished_ = readbackNumUnfinished();

  if (debugMode_) {
    debugPowers_.resize(numSystems);
    debugPowers_.zero();
    debugOutputs_.assign(numSystems, FireDebugOutput{});
  } else {
    debugPowers_.resize(0);
    debugOutputs_.clear();
  }
}

void FireBatchMinimizer::compactActiveAsync() {
  size_t tempStorageBytes = countTempStorage_.size();
  cudaCheckError(cub::DeviceSelect::Flagged(countTempStorage_.data(),
                                            tempStorageBytes,
                                            allSystemIndices_.data(),
                                            statuses_.data(),
                                            activeSystemIndices_.data(),
                                            countUnfinished_.data(),
                                            allSystemIndices_.size(),
                                            stream_));
}

int FireBatchMinimizer::readbackNumUnfinished() {
  int& host = loopStatusHost_[0];
  countUnfinished_.get(host);
  cudaCheckError(cudaStreamSynchronize(stream_));
  return host;
}

namespace {
FireKernelParams buildKernelParams(const FireOptions& opts, const double gradTol) {
  FireKernelParams params{};
  params.dtIncrementFactor    = opts.timeStepIncrement;
  params.dtDecrementFactor    = opts.timeStepDecrement;
  params.minDt                = opts.dtInit * opts.dtMinFactor;
  params.maxDt                = opts.dtInit * opts.dtMaxFactor;
  params.dMax                 = opts.dMax;
  params.alphaStart           = opts.alphaInit;
  params.alphaDecrementFactor = opts.alphaDecrement;
  params.gradTol              = gradTol;
  params.nMinForIncrease      = opts.nMinForIncrease;
  return params;
}
}  // namespace

void FireBatchMinimizer::launchPreKick(const double                  gradTol,
                                       const AsyncDeviceVector<int>& atomStarts,
                                       AsyncDeviceVector<double>&    positions,
                                       AsyncDeviceVector<double>&    grad,
                                       const int                     launchBlocks,
                                       const bool                    isFirstStep) {
  if (launchBlocks <= 0) {
    return;
  }

  cuda::std::span<double> debugPowersSpan;
  if (debugMode_ && debugPowers_.size() > 0) {
    debugPowersSpan = cuda::std::span<double>(debugPowers_.data(), debugPowers_.size());
  }

  const FireKernelParams params = buildKernelParams(fireOptions_, gradTol);

  const auto atomStartsSpan = cuda::std::span<const int>(atomStarts.data(), atomStarts.size());
  const auto activeSpan     = cuda::std::span<const int>(activeSystemIndices_.data(), activeSystemIndices_.size());
  const auto positionsSpan  = cuda::std::span<double>(positions.data(), positions.size());
  const auto gradSpan       = cuda::std::span<const double>(grad.data(), grad.size());
  const auto positiveStepsSpan =
    cuda::std::span<int>(numStepsWithPositivePower_.data(), numStepsWithPositivePower_.size());
#define NVMOLKIT_LAUNCH_FIRE_PRE(real, reduceT, storageT, VSpan, AlphaSpan, DtSpan) \
  firePreKickKernel<real, reduceT, storageT>                                        \
    <<<launchBlocks, kFireBlockSize, 0, stream_>>>(atomStartsSpan,                  \
                                                   activeSpan,                      \
                                                   positionsSpan,                   \
                                                   VSpan,                           \
                                                   gradSpan,                        \
                                                   dataDim_,                        \
                                                   AlphaSpan,                       \
                                                   DtSpan,                          \
                                                   positiveStepsSpan,               \
                                                   params,                          \
                                                   fireOptions_.takeHalfStepBack,   \
                                                   isFirstStep,                     \
                                                   statuses_.data(),                \
                                                   convergeReason_.data(),          \
                                                   debugPowersSpan)
  if (usesFloatMinimizerState(precision_)) {
    const auto v = cuda::std::span<float>(velocitiesFloat_.data(), velocitiesFloat_.size());
    const auto a = cuda::std::span<float>(alphaFloat_.data(), alphaFloat_.size());
    const auto d = cuda::std::span<float>(dtFloat_.data(), dtFloat_.size());
    if (usesFloatMinimizerCompute(precision_)) {
      if (usesFloatReduction(precision_))
        NVMOLKIT_LAUNCH_FIRE_PRE(float, float, float, v, a, d);
      else
        NVMOLKIT_LAUNCH_FIRE_PRE(float, double, float, v, a, d);
    } else {
      if (usesFloatReduction(precision_))
        NVMOLKIT_LAUNCH_FIRE_PRE(double, float, float, v, a, d);
      else
        NVMOLKIT_LAUNCH_FIRE_PRE(double, double, float, v, a, d);
    }
  } else {
    const auto v = cuda::std::span<double>(velocities_.data(), velocities_.size());
    const auto a = cuda::std::span<double>(alpha_.data(), alpha_.size());
    const auto d = cuda::std::span<double>(dt_.data(), dt_.size());
    if (usesFloatMinimizerCompute(precision_)) {
      if (usesFloatReduction(precision_))
        NVMOLKIT_LAUNCH_FIRE_PRE(float, float, double, v, a, d);
      else
        NVMOLKIT_LAUNCH_FIRE_PRE(float, double, double, v, a, d);
    } else {
      if (usesFloatReduction(precision_))
        NVMOLKIT_LAUNCH_FIRE_PRE(double, float, double, v, a, d);
      else
        NVMOLKIT_LAUNCH_FIRE_PRE(double, double, double, v, a, d);
    }
  }
#undef NVMOLKIT_LAUNCH_FIRE_PRE
  cudaCheckError(cudaGetLastError());
}

void FireBatchMinimizer::launchPostKick(const double                  gradTol,
                                        const AsyncDeviceVector<int>& atomStarts,
                                        AsyncDeviceVector<double>&    positions,
                                        AsyncDeviceVector<double>&    grad,
                                        const int                     launchBlocks) {
  if (launchBlocks <= 0) {
    return;
  }

  cuda::std::span<const double> massesSpan;
  if (masses_.size() > 0) {
    massesSpan = cuda::std::span<const double>(masses_.data(), masses_.size());
  }

  const FireKernelParams params = buildKernelParams(fireOptions_, gradTol);

  const auto atomStartsSpan = cuda::std::span<const int>(atomStarts.data(), atomStarts.size());
  const auto activeSpan     = cuda::std::span<const int>(activeSystemIndices_.data(), activeSystemIndices_.size());
  const auto positionsSpan  = cuda::std::span<double>(positions.data(), positions.size());
  const auto gradSpan       = cuda::std::span<const double>(grad.data(), grad.size());
  const auto positiveStepsSpan =
    cuda::std::span<const int>(numStepsWithPositivePower_.data(), numStepsWithPositivePower_.size());
#define NVMOLKIT_LAUNCH_FIRE_POST(real, reduceT, storageT, VSpan, AlphaSpan, DtSpan) \
  firePostKickKernel<real, reduceT, storageT>                                        \
    <<<launchBlocks, kFireBlockSize, 0, stream_>>>(atomStartsSpan,                   \
                                                   activeSpan,                       \
                                                   positionsSpan,                    \
                                                   VSpan,                            \
                                                   gradSpan,                         \
                                                   massesSpan,                       \
                                                   dataDim_,                         \
                                                   AlphaSpan,                        \
                                                   DtSpan,                           \
                                                   positiveStepsSpan,                \
                                                   params,                           \
                                                   fireOptions_.abcCorrection,       \
                                                   statuses_.data())
  if (usesFloatMinimizerState(precision_)) {
    const auto v = cuda::std::span<float>(velocitiesFloat_.data(), velocitiesFloat_.size());
    const auto a = cuda::std::span<const float>(alphaFloat_.data(), alphaFloat_.size());
    const auto d = cuda::std::span<const float>(dtFloat_.data(), dtFloat_.size());
    if (usesFloatMinimizerCompute(precision_)) {
      if (usesFloatReduction(precision_))
        NVMOLKIT_LAUNCH_FIRE_POST(float, float, float, v, a, d);
      else
        NVMOLKIT_LAUNCH_FIRE_POST(float, double, float, v, a, d);
    } else {
      if (usesFloatReduction(precision_))
        NVMOLKIT_LAUNCH_FIRE_POST(double, float, float, v, a, d);
      else
        NVMOLKIT_LAUNCH_FIRE_POST(double, double, float, v, a, d);
    }
  } else {
    const auto v = cuda::std::span<double>(velocities_.data(), velocities_.size());
    const auto a = cuda::std::span<const double>(alpha_.data(), alpha_.size());
    const auto d = cuda::std::span<const double>(dt_.data(), dt_.size());
    if (usesFloatMinimizerCompute(precision_)) {
      if (usesFloatReduction(precision_))
        NVMOLKIT_LAUNCH_FIRE_POST(float, float, double, v, a, d);
      else
        NVMOLKIT_LAUNCH_FIRE_POST(float, double, double, v, a, d);
    } else {
      if (usesFloatReduction(precision_))
        NVMOLKIT_LAUNCH_FIRE_POST(double, float, double, v, a, d);
      else
        NVMOLKIT_LAUNCH_FIRE_POST(double, double, double, v, a, d);
    }
  }
#undef NVMOLKIT_LAUNCH_FIRE_POST
  cudaCheckError(cudaGetLastError());
}

bool FireBatchMinimizer::step(const double                  gradTol,
                              const AsyncDeviceVector<int>& atomStarts,
                              AsyncDeviceVector<double>&    positions,
                              AsyncDeviceVector<double>&    grad,
                              const GradFunctor&            gFunc) {
  const ScopedNvtxRange stepRange("FireBatchMinimizer::step");
  {
    const ScopedNvtxRange gradRange("FIRE pre-kick gradient");
    grad.zero();
    gFunc();
  }
  const bool isFirstStep = (step_ == 0);
  {
    const ScopedNvtxRange preKickRange("FIRE preKick");
    launchPreKick(gradTol, atomStarts, positions, grad, lastKnownNumUnfinished_, isFirstStep);
  }
  {
    const ScopedNvtxRange gradRange("FIRE post-kick gradient");
    grad.zero();
    gFunc();
  }
  {
    const ScopedNvtxRange postKickRange("FIRE postKick");
    launchPostKick(gradTol, atomStarts, positions, grad, lastKnownNumUnfinished_);
  }
  compactActiveAsync();
  lastKnownNumUnfinished_ = readbackNumUnfinished();
  step_++;
  return lastKnownNumUnfinished_ == 0;
}

namespace {

template <typename T> std::vector<double> debugDump(const AsyncDeviceVector<T>& vec) {
  std::vector<T> stored(vec.size());
  if (vec.size() == 0) {
    return {};
  }
  vec.copyToHost(stored);
  cudaCheckError(cudaStreamSynchronize(vec.stream()));
  return std::vector<double>(stored.begin(), stored.end());
}

}  // namespace

bool FireBatchMinimizer::minimize(const int                                   numIters,
                                  const double                                gradTol,
                                  const std::vector<int>&                     atomStartsHost,
                                  const AsyncDeviceVector<int>&               atomStarts,
                                  AsyncDeviceVector<double>&                  positions,
                                  AsyncDeviceVector<double>&                  grad,
                                  [[maybe_unused]] AsyncDeviceVector<double>& energyOuts,
                                  [[maybe_unused]] AsyncDeviceVector<double>& energyBuffer,
                                  EnergyFunctor                               eFunc,
                                  const GradFunctor                           gFunc,
                                  const uint8_t*                              activeThisStage) {
  const ScopedNvtxRange minimizeRange("FireBatchMinimizer::minimize (batched)");
  initialize(atomStartsHost, nullptr, activeThisStage, FireBackend::BATCHED);

  for (int iter = 0; iter < numIters; ++iter) {
    if (debugMode_) {
      energyBuffer.zero();
      energyOuts.zero();
      eFunc(positions.data());

      const std::vector<double> energies = debugDump(energyOuts);
      const std::vector<double> powers   = debugDump(debugPowers_);
      const std::vector<double> alphas =
        usesFloatMinimizerState(precision_) ? debugDump(alphaFloat_) : debugDump(alpha_);
      const std::vector<double> dts = usesFloatMinimizerState(precision_) ? debugDump(dtFloat_) : debugDump(dt_);

      for (size_t sysIdx = 0; sysIdx < energies.size(); ++sysIdx) {
        debugOutputs_[sysIdx].energies.push_back(energies[sysIdx]);
        debugOutputs_[sysIdx].powers.push_back(powers[sysIdx]);
        debugOutputs_[sysIdx].alphas.push_back(alphas[sysIdx]);
        debugOutputs_[sysIdx].dt.push_back(dts[sysIdx]);
      }
    }

    if (lastKnownNumUnfinished_ == 0) {
      return true;
    }

    grad.zero();
    gFunc();

    const bool isFirstStep = (step_ == 0);
    launchPreKick(gradTol, atomStarts, positions, grad, lastKnownNumUnfinished_, isFirstStep);
    grad.zero();
    gFunc();
    launchPostKick(gradTol, atomStarts, positions, grad, lastKnownNumUnfinished_);
    compactActiveAsync();

    const bool poll = debugMode_ || ((iter + 1) % convergencePollInterval_ == 0) || (iter + 1 == numIters);
    if (poll) {
      const int activeBeforeStuckCheck = lastKnownNumUnfinished_;
      if (fireOptions_.stuckDetectionEnabled && activeBeforeStuckCheck > 0) {
        ++pollsSinceLastEnergyEval_;
        if (pollsSinceLastEnergyEval_ >= fireOptions_.stuckEvalEveryNPolls) {
          pollsSinceLastEnergyEval_ = 0;
          energyOuts.zero();
          eFunc(nullptr);
          fireStuckCheckKernel<<<activeBeforeStuckCheck, 1, 0, stream_>>>(
            cuda::std::span<const int>(activeSystemIndices_.data(), activeSystemIndices_.size()),
            cuda::std::span<const double>(energyOuts.data(), energyOuts.size()),
            cuda::std::span<double>(energyMinStreak_.data(), energyMinStreak_.size()),
            cuda::std::span<double>(energyMaxStreak_.data(), energyMaxStreak_.size()),
            cuda::std::span<int32_t>(stuckStreak_.data(), stuckStreak_.size()),
            statuses_.data(),
            convergeReason_.data(),
            fireOptions_.stuckEnergyRelTol,
            fireOptions_.stuckStreakLength);
          cudaCheckError(cudaGetLastError());
          compactActiveAsync();
        }
      }
      lastKnownNumUnfinished_ = readbackNumUnfinished();
    }

    step_++;
  }

  if (lastKnownNumUnfinished_ != 0) {
    lastKnownNumUnfinished_ = readbackNumUnfinished();
  }

  static const bool diagVerbose = []() {
    const char* env = std::getenv("NVMOLKIT_FIRE_DIAG");
    return env != nullptr && env[0] != '0';
  }();
  if (diagVerbose) {
    std::vector<uint8_t> reasons(numSystems_);
    convergeReason_.copyToHost(reasons.data(), numSystems_);
    cudaCheckError(cudaStreamSynchronize(stream_));
    int byGrad  = 0;
    int byStuck = 0;
    for (uint8_t reason : reasons) {
      if (reason == 1) {
        ++byGrad;
      } else if (reason == 2) {
        ++byStuck;
      }
    }
    std::cerr << "[FIRE-diag] systems=" << numSystems_ << " iters=" << step_ << " converged_grad=" << byGrad
              << " converged_stuck=" << byStuck << " unfinished=" << lastKnownNumUnfinished_ << '\n';
  }

  return lastKnownNumUnfinished_ == 0;
}

bool FireBatchMinimizer::minimize(const int                  numIters,
                                  const double               gradTol,
                                  BatchedForcefield&         ff,
                                  AsyncDeviceVector<double>& positions,
                                  AsyncDeviceVector<double>& grad,
                                  AsyncDeviceVector<double>& energyOuts,
                                  const uint8_t*             activeSystemMask) {
  const ScopedNvtxRange minimizeRange("FireBatchMinimizer::minimize (BatchedForcefield)");
  const auto&           atomStartsHost = ff.atomStartsHost();

  AsyncDeviceVector<double> energyBuffer;
  energyBuffer.setStream(stream_);
  energyBuffer.resize(energyOuts.size());
  energyBuffer.zero();

  auto eFunc = [&](const double* evalPositions) {
    const double* positionsToEvaluate = evalPositions != nullptr ? evalPositions : positions.data();
    ff.computeEnergy(energyOuts.data(), positionsToEvaluate, activeSystemMask, stream_);
  };
  auto gFunc = [&]() { ff.computeGradients(grad.data(), positions.data(), activeSystemMask, stream_); };

  // The BatchMinimizer interface requires owning storage for device atom offsets.
  AsyncDeviceVector<int> atomStartsDeviceMirror;
  atomStartsDeviceMirror.setStream(stream_);
  atomStartsDeviceMirror.setFromArray(ff.atomStartsHost().data(), atomStartsHost.size());

  return minimize(numIters,
                  gradTol,
                  atomStartsHost,
                  atomStartsDeviceMirror,
                  positions,
                  grad,
                  energyOuts,
                  energyBuffer,
                  eFunc,
                  gFunc,
                  activeSystemMask);
}

bool FireBatchMinimizer::checkPerMolConvergence() {
  if (numSystems_ == 0) {
    return true;
  }
  statuses_.copyToHost(convergenceHost_.data(), numSystems_);
  cudaCheckError(cudaStreamSynchronize(stream_));
  for (const int molIdx : activeMolIds_) {
    if (convergenceHost_[molIdx] != 0) {
      return false;
    }
  }
  return true;
}

namespace {
void requireStuckDetectionDisabled(const FireOptions& options) {
  if (options.stuckDetectionEnabled) {
    throw std::runtime_error(
      "FireBatchMinimizer per-molecule backend does not support FireOptions::stuckDetectionEnabled");
  }
}
}  // namespace

template <typename DeviceBuffers>
bool FireBatchMinimizer::minimizeWithMMFFImpl(const int               numIters,
                                              const double            gradTol,
                                              const std::vector<int>& atomStartsHost,
                                              DeviceBuffers&          systemDevice,
                                              const uint8_t*          activeThisStage) {
  const ScopedNvtxRange perMolRange("FireBatchMinimizer::perMoleculeMinimizeMMFF");
  requireStuckDetectionDisabled(fireOptions_);

  const FireBackend effectiveBackend = resolveBackend(atomStartsHost);
  if (effectiveBackend == FireBackend::BATCHED) {
    throw std::runtime_error(
      "FireBatchMinimizer::minimizeWithMMFF requires PER_MOLECULE backend (or HYBRID resolving "
      "to PER_MOLECULE); use minimize(..., BatchedForcefield&) for batched MMFF minimization");
  }

  initialize(atomStartsHost, /*masses=*/nullptr, activeThisStage, effectiveBackend);

  if (activeMolIds_.empty()) {
    return true;
  }

  auto       terms          = MMFF::toEnergyForceContribsDevicePtr(systemDevice);
  auto       systemIndices  = MMFF::toBatchedIndicesDevicePtr(systemDevice);
  const bool hasConstraints = MMFF::batchHasConstraints(systemDevice.contribs);

  cudaError_t err;
  if (usesFloatMinimizerState(precision_)) {
    err = launchFirePerMolKernel(static_cast<int>(activeMolIds_.size()),
                                 activeMolIdsDevice_.data(),
                                 maxAtomsInBatch_,
                                 systemDevice.indices.atomStarts.data(),
                                 fireOptions_,
                                 numIters,
                                 gradTol,
                                 terms,
                                 systemIndices,
                                 hasConstraints,
                                 systemDevice.positions.data(),
                                 systemDevice.grad.data(),
                                 velocitiesFloat_.data(),
                                 alphaFloat_.data(),
                                 dtFloat_.data(),
                                 numStepsWithPositivePower_.data(),
                                 masses_.size() > 0 ? masses_.data() : nullptr,
                                 systemDevice.energyOuts.data(),
                                 statuses_.data(),
                                 stream_);
  } else {
    err = launchFirePerMolKernel(static_cast<int>(activeMolIds_.size()),
                                 activeMolIdsDevice_.data(),
                                 maxAtomsInBatch_,
                                 systemDevice.indices.atomStarts.data(),
                                 fireOptions_,
                                 numIters,
                                 gradTol,
                                 terms,
                                 systemIndices,
                                 hasConstraints,
                                 systemDevice.positions.data(),
                                 systemDevice.grad.data(),
                                 velocities_.data(),
                                 alpha_.data(),
                                 dt_.data(),
                                 numStepsWithPositivePower_.data(),
                                 masses_.size() > 0 ? masses_.data() : nullptr,
                                 systemDevice.energyOuts.data(),
                                 statuses_.data(),
                                 stream_);
  }
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Per-molecule FIRE MMFF kernel failed: ") + cudaGetErrorString(err));
  }
  return checkPerMolConvergence();
}

bool FireBatchMinimizer::minimizeWithMMFF(const int                            numIters,
                                          const double                         gradTol,
                                          const std::vector<int>&              atomStartsHost,
                                          MMFF::BatchedMolecularDeviceBuffers& systemDevice,
                                          const uint8_t*                       activeThisStage) {
  return minimizeWithMMFFImpl(numIters, gradTol, atomStartsHost, systemDevice, activeThisStage);
}

bool FireBatchMinimizer::minimizeWithMMFF(const int                                     numIters,
                                          const double                                  gradTol,
                                          const std::vector<int>&                       atomStartsHost,
                                          MMFF::BatchedMolecularDeviceBuffersF32Params& systemDevice,
                                          const uint8_t*                                activeThisStage) {
  return minimizeWithMMFFImpl(numIters, gradTol, atomStartsHost, systemDevice, activeThisStage);
}

FireInternalState FireBatchMinimizer::snapshotInternalState() const {
  FireInternalState snap;
  if (usesFloatMinimizerState(precision_)) {
    snap.velocities = debugDump(velocitiesFloat_);
    snap.dt         = debugDump(dtFloat_);
    snap.alpha      = debugDump(alphaFloat_);
  } else {
    snap.velocities = debugDump(velocities_);
    snap.dt         = debugDump(dt_);
    snap.alpha      = debugDump(alpha_);
  }
  snap.nStepsPositive.resize(numStepsWithPositivePower_.size());
  if (numStepsWithPositivePower_.size() > 0) {
    numStepsWithPositivePower_.copyToHost(snap.nStepsPositive);
  }
  snap.statuses.resize(statuses_.size());
  if (statuses_.size() > 0) {
    statuses_.copyToHost(snap.statuses);
  }
  cudaCheckError(cudaStreamSynchronize(stream_));
  return snap;
}

}  // namespace nvMolKit
