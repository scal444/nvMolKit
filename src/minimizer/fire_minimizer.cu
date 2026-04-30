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

#include <algorithm>
#include <cub/cub.cuh>
#include <numeric>
#include <stdexcept>

#include "../forcefields/batched_forcefield.h"
#include "fire_minimizer.h"

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
// This replaces the previous (incorrect) energy-only kCalMolToEV factor.
constexpr double kForceKcalMolPerAng_PerAmu_to_AngPerPs2 = 4.184 * 100.0;

template <typename T>
__global__ void setAllKernel(const int numElements, const T value, T* dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = value;
  }
}

template <typename T>
void setAll(AsyncDeviceVector<T>& vec, const T& value) {
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
  return data.subspan(atomStarts[sysIdx] * dataDim,
                      (atomStarts[sysIdx + 1] - atomStarts[sysIdx]) * dataDim);
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

//! \brief Single-block-per-system FIRE 2.0 kernel.
//!
//! Implements the ASE FIRE2 control flow with one gradient evaluation per
//! iteration (ASE re-evaluates the gradient after the negative-power
//! half-step-back; we reuse the same gradient for the post-half-step kick,
//! avoiding a second device round-trip per iteration).
//!  1. Compute power = v . (-grad) and |grad|^2.
//!  2. Convergence check on |grad|.
//!  3. State machine: positive-power -> grow dt + shrink alpha after Nmin
//!     consecutive accepted steps. Negative-power -> shrink dt, reset alpha,
//!     half-step backward (with the post-decrement dt) and zero v.
//!  4. Kick: v += dt * force / m (force = -grad converted to Å/ps^2).
//!  5. Mixer: v = (1 - alpha) * v + alpha * |v|/|grad| * (-grad). With ABC,
//!     multiply by 1 / (1 - (1 - alpha)^(N+1)).
//!  6. Displacement clip: ABC clips per-component v to ±dMax/dt (so dr per
//!     component is bounded). Non-ABC scales dr (not v) so that |dr| <= dMax.
//!  7. Position update: x += dr.
__global__ void fireKernel(const cuda::std::span<const int>    atomStarts,
                           const cuda::std::span<const int>    activeIndices,
                           const cuda::std::span<double>       x,
                           const cuda::std::span<double>       v,
                           const cuda::std::span<const double> f,
                           const cuda::std::span<const double> masses,
                           const int                           dataDim,
                           const cuda::std::span<double>       alphas,
                           const cuda::std::span<double>       dts,
                           const cuda::std::span<int>          nStepsPositive,
                           const FireKernelParams              params,
                           const bool                          useAbc,
                           const bool                          takeHalfStepBack,
                           const bool                          isFirstStep,
                           uint8_t*                            statuses,
                           const cuda::std::span<double>       debugPowers) {
  using BlockReduce = cub::BlockReduce<double, kFireBlockSize>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ double                            sharedScalar0;
  __shared__ double                            sharedScalar1;
  __shared__ double                            sharedDt;
  __shared__ double                            sharedAlpha;
  __shared__ int                               sharedNsteps;
  __shared__ bool                              sharedNegative;
  __shared__ bool                              sharedConverged;

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

  if (threadIdx.x == 0) {
    sharedDt        = dts[sysIdx];
    sharedAlpha     = alphas[sysIdx];
    sharedNsteps    = nStepsPositive[sysIdx];
    sharedNegative  = false;
    sharedConverged = false;
  }
  __syncthreads();

  const double dtIn    = sharedDt;
  const double alphaIn = sharedAlpha;
  const int    nstepIn = sharedNsteps;

  double power  = 0.0;
  double gradSq = 0.0;
  for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
    const double fi = fSys[i];
    if (!isFirstStep) {
      power += vSys[i] * -fi;
    }
    gradSq += fi * fi;
  }
  double powerSum = 0.0;
  if (!isFirstStep) {
    powerSum = BlockReduce(tempStorage).Sum(power);
    __syncthreads();
  }
  const double gradSqSum = BlockReduce(tempStorage).Sum(gradSq);
  if (threadIdx.x == 0) {
    sharedScalar0 = powerSum;
    sharedScalar1 = gradSqSum;
  }
  __syncthreads();
  const double powerShared  = sharedScalar0;
  const double gradSqShared = sharedScalar1;

  if (threadIdx.x == 0) {
    if (sqrt(gradSqShared) <= params.gradTol) {
      sharedConverged   = true;
      statuses[sysIdx]  = 0;
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

    double newDt     = dtIn;
    double newAlpha  = alphaIn;
    int    newNsteps = nstepIn;
    bool   negative  = false;

    if (powerShared >= 0.0) {
      newNsteps = nstepIn + 1;
      if (newNsteps > params.nMinForIncrease) {
        newDt    = fmin(dtIn * params.dtIncrementFactor, params.maxDt);
        newAlpha = alphaIn * params.alphaDecrementFactor;
      }
    } else {
      negative  = true;
      newNsteps = 0;
      newAlpha  = params.alphaStart;
      newDt     = fmax(dtIn * params.dtDecrementFactor, params.minDt);
    }

    sharedDt              = newDt;
    sharedAlpha           = newAlpha;
    sharedNsteps          = newNsteps;
    sharedNegative        = negative;
    dts[sysIdx]           = newDt;
    alphas[sysIdx]        = newAlpha;
    nStepsPositive[sysIdx] = newNsteps;
  }
  __syncthreads();

  const double dt       = sharedDt;
  const double alpha    = sharedAlpha;
  const int    nsteps   = sharedNsteps;
  const bool   negative = sharedNegative;

  if (negative) {
    for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
      if (takeHalfStepBack) {
        xSys[i] -= 0.5 * dt * vSys[i];
      }
      vSys[i] = 0.0;
    }
    __syncthreads();
  }

  // Kick: v += dt * force / m, force = -grad * accel-conv.
  double vSqAccum = 0.0;
  for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
    const double accelMag = -fSys[i] * kForceKcalMolPerAng_PerAmu_to_AngPerPs2;
    double       accel;
    if (massEnabled) {
      const int coordIdx = i / dataDim;
      accel              = accelMag / massSys[coordIdx];
    } else {
      accel = accelMag;
    }
    const double newV = vSys[i] + dt * accel;
    vSys[i]           = newV;
    vSqAccum += newV * newV;
  }
  const double vSqReduced = BlockReduce(tempStorage).Sum(vSqAccum);
  if (threadIdx.x == 0) {
    sharedScalar0 = vSqReduced;
  }
  __syncthreads();
  const double vSqSum = sharedScalar0;

  // Mixer: v = (1 - alpha) * v + alpha * |v|/|grad| * (-grad). With ABC,
  // multiply by 1 / (1 - (1 - alpha)^(N+1)).
  const double mixCoef1 = 1.0 - alpha;
  const double mixCoef2 = (gradSqShared > 1e-30) ? (alpha * sqrt(vSqSum) / sqrt(gradSqShared)) : 0.0;
  double       abcMult  = 1.0;
  if (useAbc) {
    const double oneMinusA = 1.0 - fmax(alpha, 1e-10);
    const double pow_term  = pow(oneMinusA, static_cast<double>(nsteps + 1));
    const double denom     = 1.0 - pow_term;
    abcMult                = (denom > 1e-30) ? (1.0 / denom) : 1.0;
  }

  for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
    const double vMix = mixCoef1 * vSys[i] + mixCoef2 * (-fSys[i]);
    vSys[i]           = abcMult * vMix;
  }
  __syncthreads();

  // Displacement handling: ABC clips per-component v to ±dMax/dt, non-ABC
  // norm-clips dr (without modifying v) before the position update.
  double drScale = 1.0;
  if (useAbc) {
    if (params.dMax > 0.0) {
      const double maxV = params.dMax / dt;
      for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
        const double clamped = fmax(-maxV, fmin(maxV, vSys[i]));
        vSys[i]              = clamped;
      }
      __syncthreads();
    }
  } else {
    if (params.dMax > 0.0) {
      double drSqAccum = 0.0;
      for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
        const double dri = dt * vSys[i];
        drSqAccum += dri * dri;
      }
      const double drSqReduced = BlockReduce(tempStorage).Sum(drSqAccum);
      if (threadIdx.x == 0) {
        sharedScalar0 = drSqReduced;
      }
      __syncthreads();
      const double drNorm = sqrt(sharedScalar0);
      if (drNorm > params.dMax) {
        drScale = params.dMax / drNorm;
      }
    }
  }

  for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
    xSys[i] += drScale * dt * vSys[i];
  }
}

}  // namespace

FireBatchMinimizer::FireBatchMinimizer(const int          dataDim,
                                       const FireOptions& options,
                                       cudaStream_t       stream,
                                       const bool         debugMode)
    : dataDim_(dataDim),
      fireOptions_(options),
      stream_(stream),
      debugMode_(debugMode) {
  velocities_.setStream(stream_);
  statuses_.setStream(stream_);
  dt_.setStream(stream_);
  alpha_.setStream(stream_);
  allSystemIndices_.setStream(stream_);
  activeSystemIndices_.setStream(stream_);
  numStepsWithPositivePower_.setStream(stream_);
  countUnfinished_.setStream(stream_);
  countTempStorage_.setStream(stream_);
  masses_.setStream(stream_);
  debugPowers_.setStream(stream_);
  loopStatusHost_.resize(1);
  loopStatusHost_[0] = 0;
}

void FireBatchMinimizer::setMasses(const std::vector<double>& masses) {
  hostMasses_ = masses;
}

void FireBatchMinimizer::setConvergencePollInterval(const int interval) {
  if (interval < 1) {
    throw std::invalid_argument("FireBatchMinimizer poll interval must be >= 1");
  }
  convergencePollInterval_ = interval;
}

void FireBatchMinimizer::initialize(const std::vector<int>& atomStartsHost,
                                    const double*           masses,
                                    const uint8_t*          activeThisStage) {
  step_ = 0;
  const int totalAtoms = atomStartsHost.back();
  const int numSystems = static_cast<int>(atomStartsHost.size()) - 1;
  numSystems_          = numSystems;

  velocities_.resize(static_cast<size_t>(totalAtoms) * dataDim_);
  velocities_.zero();

  if (fireOptions_.useMass && masses != nullptr) {
    masses_.resize(totalAtoms);
    cudaCheckError(
      cudaMemcpyAsync(masses_.data(), masses, totalAtoms * sizeof(double), cudaMemcpyDefault, stream_));
  } else if (fireOptions_.useMass && !hostMasses_.empty()) {
    if (hostMasses_.size() != static_cast<size_t>(totalAtoms)) {
      throw std::runtime_error("Stored masses size does not match atom count");
    }
    masses_.setFromVector(hostMasses_);
  } else {
    masses_.resize(0);
  }

  statuses_.resize(numSystems);
  if (activeThisStage != nullptr) {
    cudaCheckError(
      cudaMemcpyAsync(statuses_.data(), activeThisStage, numSystems * sizeof(uint8_t), cudaMemcpyDefault, stream_));
  } else {
    setAll(statuses_, static_cast<uint8_t>(1));
  }

  activeSystemIndices_.resize(numSystems);
  allSystemIndices_.resize(numSystems);
  std::vector<int> indicesHost(numSystems);
  std::iota(indicesHost.begin(), indicesHost.end(), 0);
  allSystemIndices_.setFromVector(indicesHost);
  activeSystemIndices_.setFromVector(indicesHost);

  numStepsWithPositivePower_.resize(numSystems);
  numStepsWithPositivePower_.zero();
  alpha_.resize(numSystems);
  setAll(alpha_, fireOptions_.alphaInit);
  dt_.resize(numSystems);
  setAll(dt_, fireOptions_.dtInit);

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

void FireBatchMinimizer::launchFireKernel(const double                  gradTol,
                                          const AsyncDeviceVector<int>& atomStarts,
                                          AsyncDeviceVector<double>&    positions,
                                          AsyncDeviceVector<double>&    grad,
                                          const int                     launchBlocks,
                                          const bool                    isFirstStep) {
  if (launchBlocks <= 0) {
    return;
  }

  cuda::std::span<const double> massesSpan;
  if (masses_.size() > 0) {
    massesSpan = cuda::std::span<const double>(masses_.data(), masses_.size());
  }

  cuda::std::span<double> debugPowersSpan;
  if (debugMode_ && debugPowers_.size() > 0) {
    debugPowersSpan = cuda::std::span<double>(debugPowers_.data(), debugPowers_.size());
  }

  FireKernelParams params{};
  params.dtIncrementFactor    = fireOptions_.timeStepIncrement;
  params.dtDecrementFactor    = fireOptions_.timeStepDecrement;
  params.minDt                = fireOptions_.dtInit * fireOptions_.dtMinFactor;
  params.maxDt                = fireOptions_.dtInit * fireOptions_.dtMaxFactor;
  params.dMax                 = fireOptions_.dMax;
  params.alphaStart           = fireOptions_.alphaInit;
  params.alphaDecrementFactor = fireOptions_.alphaDecrement;
  params.gradTol              = gradTol;
  params.nMinForIncrease      = fireOptions_.nMinForIncrease;

  fireKernel<<<launchBlocks, kFireBlockSize, 0, stream_>>>(
    cuda::std::span<const int>(atomStarts.data(), atomStarts.size()),
    cuda::std::span<const int>(activeSystemIndices_.data(), activeSystemIndices_.size()),
    cuda::std::span<double>(positions.data(), positions.size()),
    cuda::std::span<double>(velocities_.data(), velocities_.size()),
    cuda::std::span<const double>(grad.data(), grad.size()),
    massesSpan,
    dataDim_,
    cuda::std::span<double>(alpha_.data(), alpha_.size()),
    cuda::std::span<double>(dt_.data(), dt_.size()),
    cuda::std::span<int>(numStepsWithPositivePower_.data(), numStepsWithPositivePower_.size()),
    params,
    fireOptions_.abcCorrection,
    fireOptions_.takeHalfStepBack,
    isFirstStep,
    statuses_.data(),
    debugPowersSpan);
  cudaCheckError(cudaGetLastError());
}

bool FireBatchMinimizer::step(const double                  gradTol,
                              const AsyncDeviceVector<int>& atomStarts,
                              AsyncDeviceVector<double>&    positions,
                              AsyncDeviceVector<double>&    grad,
                              const GradFunctor&            gFunc) {
  grad.zero();
  gFunc();
  const bool isFirstStep = (step_ == 0);
  launchFireKernel(gradTol, atomStarts, positions, grad, lastKnownNumUnfinished_, isFirstStep);
  compactActiveAsync();
  lastKnownNumUnfinished_ = readbackNumUnfinished();
  step_++;
  return lastKnownNumUnfinished_ == 0;
}

namespace {

std::vector<double> debugDump(const AsyncDeviceVector<double>& vec) {
  std::vector<double> result(vec.size());
  if (vec.size() == 0) {
    return result;
  }
  vec.copyToHost(result);
  cudaCheckError(cudaStreamSynchronize(vec.stream()));
  return result;
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
  initialize(atomStartsHost, nullptr, activeThisStage);

  for (int iter = 0; iter < numIters; ++iter) {
    if (debugMode_) {
      energyBuffer.zero();
      energyOuts.zero();
      eFunc(positions.data());

      const std::vector<double> energies = debugDump(energyOuts);
      const std::vector<double> powers   = debugDump(debugPowers_);
      const std::vector<double> alphas   = debugDump(alpha_);
      const std::vector<double> dts      = debugDump(dt_);

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
    launchFireKernel(gradTol, atomStarts, positions, grad, lastKnownNumUnfinished_, isFirstStep);
    compactActiveAsync();

    const bool poll = debugMode_ ||
                      ((iter + 1) % convergencePollInterval_ == 0) ||
                      (iter + 1 == numIters);
    if (poll) {
      lastKnownNumUnfinished_ = readbackNumUnfinished();
    }

    step_++;
  }

  if (lastKnownNumUnfinished_ != 0) {
    lastKnownNumUnfinished_ = readbackNumUnfinished();
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
  const auto& atomStartsHost = ff.atomStartsHost();

  AsyncDeviceVector<double> energyBuffer;
  energyBuffer.setStream(stream_);
  energyBuffer.resize(energyOuts.size());
  energyBuffer.zero();

  auto eFunc = [&](const double* evalPositions) {
    const double* positionsToEvaluate = evalPositions != nullptr ? evalPositions : positions.data();
    ff.computeEnergy(energyOuts.data(), positionsToEvaluate, activeSystemMask, stream_);
  };
  auto gFunc = [&]() { ff.computeGradients(grad.data(), positions.data(), activeSystemMask, stream_); };

  // The base minimize() expects an AsyncDeviceVector<int>& for atomStarts but
  // only ever calls .data() and .size() on it. Wrap the raw device pointer in
  // a non-owning shim by allocating a tiny mirror.
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

FireInternalState FireBatchMinimizer::snapshotInternalState() const {
  FireInternalState snap;
  snap.velocities.resize(velocities_.size());
  if (velocities_.size() > 0) {
    velocities_.copyToHost(snap.velocities);
  }
  snap.dt.resize(dt_.size());
  if (dt_.size() > 0) {
    dt_.copyToHost(snap.dt);
  }
  snap.alpha.resize(alpha_.size());
  if (alpha_.size() > 0) {
    alpha_.copyToHost(snap.alpha);
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
