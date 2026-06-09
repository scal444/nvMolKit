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

#include "src/forcefields/dist_geom_kernels.h"
#include "src/forcefields/dist_geom_kernels_device.cuh"
#include "src/forcefields/mmff_kernels.h"
#include "src/forcefields/mmff_kernels_device.cuh"
#include "src/minimizer/bfgs_types.h"
#include "src/minimizer/fire_minimize_permol_kernels.h"
#include "src/utils/cub_helpers.cuh"
#include "src/utils/device_vector.h"

namespace nvMolKit {

namespace {

constexpr int kFirePerMolBlockSize = 128;

//! Acceleration conversion factor: 1 kcal/mol/Å applied to 1 amu produces 4.184 * 100 Å/ps^2.
//! Mirrors @c kForceKcalMolPerAng_PerAmu_to_AngPerPs2 in the batched FIRE implementation.
constexpr double kForceKcalMolPerAng_PerAmu_to_AngPerPs2 = 4.184 * 100.0;

//! Packed read-only kernel parameters mirroring ::FireKernelParams in the batched
//! implementation (kept private to this TU to avoid coupling to the batched header).
struct FirePerMolKernelParams {
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

FirePerMolKernelParams buildKernelParams(const FireOptions& opts, const double gradTol) {
  FirePerMolKernelParams params{};
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

//! Compile-time data dimensionality lookup mirroring the BFGS per-mol kernel.
template <ForceFieldType FFType> struct DataDimTraits;

template <> struct DataDimTraits<ForceFieldType::MMFF> {
  static constexpr int value = 3;
};

template <> struct DataDimTraits<ForceFieldType::ETK> {
  static constexpr int value = 4;
};

template <> struct DataDimTraits<ForceFieldType::DG> {
  static constexpr int value = 4;
};

//! Block-wide gradient evaluation dispatched on the force-field type.
template <ForceFieldType FFType, bool HasConstraints, int dataDim, typename TermsType, typename IndicesType>
__device__ __forceinline__ void evalMolGrad([[maybe_unused]] const TermsType&   terms,
                                            [[maybe_unused]] const IndicesType& systemIndices,
                                            [[maybe_unused]] const double*      molCoords,
                                            [[maybe_unused]] double*            grad,
                                            [[maybe_unused]] const int          molIdx,
                                            [[maybe_unused]] const int          tid,
                                            [[maybe_unused]] const double       chiralWeight,
                                            [[maybe_unused]] const double       fourthDimWeight) {
  if constexpr (FFType == ForceFieldType::MMFF) {
    MMFF::molGrad<kFirePerMolBlockSize, HasConstraints>(terms, systemIndices, molCoords, grad, molIdx, tid);
  } else if constexpr (FFType == ForceFieldType::ETK) {
    DistGeom::molGradETK(terms, systemIndices, molCoords, grad, molIdx, tid);
  } else {  // DG
    DistGeom::molGradDG<dataDim>(terms, systemIndices, molCoords, grad, molIdx, chiralWeight, fourthDimWeight, tid);
  }
}

//! Block-wide energy evaluation dispatched on the force-field type.
template <ForceFieldType FFType, bool HasConstraints, int dataDim, typename TermsType, typename IndicesType>
__device__ __forceinline__ double evalMolEnergy([[maybe_unused]] const TermsType&   terms,
                                                [[maybe_unused]] const IndicesType& systemIndices,
                                                [[maybe_unused]] const double*      molCoords,
                                                [[maybe_unused]] const int          molIdx,
                                                [[maybe_unused]] const int          tid,
                                                [[maybe_unused]] const double       chiralWeight,
                                                [[maybe_unused]] const double       fourthDimWeight) {
  if constexpr (FFType == ForceFieldType::MMFF) {
    return MMFF::molEnergy<kFirePerMolBlockSize, HasConstraints>(terms, systemIndices, molCoords, molIdx, tid);
  } else if constexpr (FFType == ForceFieldType::ETK) {
    return DistGeom::molEnergyETK(terms, systemIndices, molCoords, molIdx, tid);
  } else {  // DG
    return DistGeom::molEnergyDG<dataDim>(terms, systemIndices, molCoords, molIdx, chiralWeight, fourthDimWeight, tid);
  }
}

template <int            MaxAtoms,
          bool           UseSharedMem,
          ForceFieldType FFType,
          bool           HasConstraints,
          typename TermsType,
          typename IndicesType>
__launch_bounds__(kFirePerMolBlockSize)
  __global__ void firePerMolKernel(const int                     numIters,
                                   const FirePerMolKernelParams  params,
                                   const bool                    takeHalfStepBack,
                                   const bool                    useAbc,
                                   const bool                    useMass,
                                   const TermsType*              terms,
                                   const IndicesType*            systemIndices,
                                   const int*                    molIdList,
                                   const int*                    atomStarts,
                                   double*                       positions,
                                   double*                       grad,
                                   double*                       velocities,
                                   double*                       alphas,
                                   double*                       dts,
                                   int*                          nStepsPositive,
                                   const double*                 masses,
                                   double*                       energyOuts,
                                   uint8_t*                      statuses,
                                   [[maybe_unused]] const double chiralWeight,
                                   [[maybe_unused]] const double fourthDimWeight) {
  const int molIdx = molIdList[blockIdx.x];
  const int tid    = threadIdx.x;

  if (statuses[molIdx] == 0) {
    return;
  }

  const int atomStart = atomStarts[molIdx];
  const int atomEnd   = atomStarts[molIdx + 1];
  const int numAtoms  = atomEnd - atomStart;

  constexpr int dataDim  = DataDimTraits<FFType>::value;
  constexpr int maxTerms = MaxAtoms * dataDim;
  const int     numTerms = dataDim * numAtoms;

  // Pointers to working memory (shared or global).
  double*       localPos;
  double*       localVel;
  double*       localGrad;
  double* const globalPos  = positions + atomStart * dataDim;
  double* const globalVel  = velocities + atomStart * dataDim;
  double* const globalGrad = grad + atomStart * dataDim;

  if constexpr (UseSharedMem) {
    __shared__ double sharedPos[maxTerms];
    __shared__ double sharedVel[maxTerms];
    __shared__ double sharedGrad[maxTerms];
    localPos  = sharedPos;
    localVel  = sharedVel;
    localGrad = sharedGrad;
    for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
      localPos[i]  = globalPos[i];
      localVel[i]  = globalVel[i];
      localGrad[i] = 0.0;
    }
    __syncthreads();
  } else {
    localPos  = globalPos;
    localVel  = globalVel;
    localGrad = globalGrad;
    for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
      localGrad[i] = 0.0;
    }
    __syncthreads();
  }

  const double* massSys = useMass ? (masses + atomStart) : nullptr;

  using BlockReduce = cub::BlockReduce<double, kFirePerMolBlockSize>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  __shared__ double sharedDt;
  __shared__ double sharedAlpha;
  __shared__ int    sharedNsteps;
  __shared__ double sharedScalar0;
  __shared__ double sharedScalar1;
  __shared__ bool   sharedConverged;
  __shared__ bool   sharedNegative;

  if (tid == 0) {
    sharedDt        = dts[molIdx];
    sharedAlpha     = alphas[molIdx];
    sharedNsteps    = nStepsPositive[molIdx];
    sharedConverged = false;
  }
  __syncthreads();

  // -------------------- main FIRE loop --------------------
  for (int iter = 0; iter < numIters; ++iter) {
    const bool isFirstStep = (iter == 0);

    // Evaluate gradient at current positions (use pre-zeroed buffer for atomicAdd accumulation).
    if (!isFirstStep) {
      for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
        localGrad[i] = 0.0;
      }
      __syncthreads();
    }
    evalMolGrad<FFType, HasConstraints, dataDim>(*terms,
                                                 *systemIndices,
                                                 localPos,
                                                 localGrad,
                                                 molIdx,
                                                 tid,
                                                 chiralWeight,
                                                 fourthDimWeight);
    __syncthreads();

    // -------------------- pre-kick: convergence + dt/alpha state machine --------------------
    double power  = 0.0;
    double gradSq = 0.0;
    for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
      const double fi = localGrad[i];
      if (!isFirstStep) {
        power += localVel[i] * -fi;
      }
      gradSq += fi * fi;
    }
    double powerSum = 0.0;
    if (!isFirstStep) {
      powerSum = BlockReduce(tempStorage).Sum(power);
      __syncthreads();
    }
    const double gradSqSum = BlockReduce(tempStorage).Sum(gradSq);
    if (tid == 0) {
      sharedScalar0 = powerSum;
      sharedScalar1 = gradSqSum;
    }
    __syncthreads();
    const double powerShared  = sharedScalar0;
    const double gradSqShared = sharedScalar1;

    if (tid == 0) {
      if (sqrt(gradSqShared) <= params.gradTol) {
        sharedConverged  = true;
        statuses[molIdx] = 0;
      }
    }
    __syncthreads();
    if (sharedConverged) {
      break;
    }

    if (tid == 0 && !isFirstStep) {
      double newDt     = sharedDt;
      double newAlpha  = sharedAlpha;
      int    newNsteps = sharedNsteps;
      if (powerShared >= 0.0) {
        newNsteps = sharedNsteps + 1;
        if (newNsteps > params.nMinForIncrease) {
          newDt    = fmin(sharedDt * params.dtIncrementFactor, params.maxDt);
          newAlpha = sharedAlpha * params.alphaDecrementFactor;
        }
      } else {
        newNsteps = 0;
        newAlpha  = params.alphaStart;
        newDt     = fmax(sharedDt * params.dtDecrementFactor, params.minDt);
      }
      sharedDt     = newDt;
      sharedAlpha  = newAlpha;
      sharedNsteps = newNsteps;
    }
    __syncthreads();

    const bool negative = !isFirstStep && (powerShared < 0.0);
    if (tid == 0) {
      sharedNegative = negative;
    }
    __syncthreads();

    if (sharedNegative) {
      const double dtNow = sharedDt;
      for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
        if (takeHalfStepBack) {
          localPos[i] -= 0.5 * dtNow * localVel[i];
        }
        localVel[i] = 0.0;
      }
      __syncthreads();
      // Re-evaluate gradient at the rolled-back position before the post-kick.
      for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
        localGrad[i] = 0.0;
      }
      __syncthreads();
      evalMolGrad<FFType, HasConstraints, dataDim>(*terms,
                                                   *systemIndices,
                                                   localPos,
                                                   localGrad,
                                                   molIdx,
                                                   tid,
                                                   chiralWeight,
                                                   fourthDimWeight);
      __syncthreads();
    }

    // -------------------- post-kick: integrate v, mixer, displacement clip, position update --------------------
    const double dt     = sharedDt;
    const double alpha  = sharedAlpha;
    const int    nsteps = sharedNsteps;

    double vSqAccum    = 0.0;
    double gradSqAccum = 0.0;
    for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
      double accel;
      if (useMass) {
        const double accelMag = -localGrad[i] * kForceKcalMolPerAng_PerAmu_to_AngPerPs2;
        const int    coordIdx = i / dataDim;
        accel                 = accelMag / massSys[coordIdx];
      } else {
        accel = -localGrad[i];
      }
      const double newV = localVel[i] + dt * accel;
      localVel[i]       = newV;
      vSqAccum += newV * newV;
      gradSqAccum += localGrad[i] * localGrad[i];
    }
    const double vSqReduced = BlockReduce(tempStorage).Sum(vSqAccum);
    if (tid == 0) {
      sharedScalar0 = vSqReduced;
    }
    __syncthreads();
    const double vSqSum        = sharedScalar0;
    const double gradSqReduced = BlockReduce(tempStorage).Sum(gradSqAccum);
    if (tid == 0) {
      sharedScalar0 = gradSqReduced;
    }
    __syncthreads();
    const double gradSqSum2 = sharedScalar0;

    const double mixCoef1 = 1.0 - alpha;
    const double mixCoef2 = (gradSqSum2 > 1e-30) ? (alpha * sqrt(vSqSum) / sqrt(gradSqSum2)) : 0.0;
    double       abcMult  = 1.0;
    if (useAbc) {
      const double oneMinusA = 1.0 - fmax(alpha, 1e-10);
      const double pow_term  = pow(oneMinusA, static_cast<double>(nsteps + 1));
      const double denom     = 1.0 - pow_term;
      abcMult                = (denom > 1e-30) ? (1.0 / denom) : 1.0;
    }

    for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
      const double vMix = mixCoef1 * localVel[i] + mixCoef2 * (-localGrad[i]);
      localVel[i]       = abcMult * vMix;
    }
    __syncthreads();

    double drScale = 1.0;
    if (useAbc) {
      if (params.dMax > 0.0) {
        const double maxV = params.dMax / dt;
        for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
          const double clamped = fmax(-maxV, fmin(maxV, localVel[i]));
          localVel[i]          = clamped;
        }
        __syncthreads();
      }
    } else {
      if (params.dMax > 0.0) {
        double drSqAccum = 0.0;
        for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
          const double dri = dt * localVel[i];
          drSqAccum += dri * dri;
        }
        const double drSqReduced = BlockReduce(tempStorage).Sum(drSqAccum);
        if (tid == 0) {
          sharedScalar0 = drSqReduced;
        }
        __syncthreads();
        const double drNorm = sqrt(sharedScalar0);
        if (drNorm > params.dMax) {
          drScale = params.dMax / drNorm;
        }
      }
    }

    for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
      localPos[i] += drScale * dt * localVel[i];
    }
    __syncthreads();
  }

  // Write back per-system state.
  if (tid == 0) {
    dts[molIdx]            = sharedDt;
    alphas[molIdx]         = sharedAlpha;
    nStepsPositive[molIdx] = sharedNsteps;
  }

  if constexpr (UseSharedMem) {
    for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
      globalPos[i] = localPos[i];
      globalVel[i] = localVel[i];
    }
  }
  __syncthreads();

  // Final energy at the converged position.
  const double finalThreadEnergy = evalMolEnergy<FFType, HasConstraints, dataDim>(*terms,
                                                                                  *systemIndices,
                                                                                  localPos,
                                                                                  molIdx,
                                                                                  tid,
                                                                                  chiralWeight,
                                                                                  fourthDimWeight);
  const double finalEnergy       = BlockReduce(tempStorage).Sum(finalThreadEnergy);
  if (tid == 0) {
    energyOuts[molIdx] = finalEnergy;
  }
}

template <int            MaxAtoms,
          bool           UseSharedMem,
          ForceFieldType FFType,
          bool           HasConstraints,
          typename TermsType,
          typename IndicesType>
cudaError_t launchKernelForSize(const int                     numMols,
                                const int*                    molIdList,
                                const FirePerMolKernelParams& params,
                                const bool                    takeHalfStepBack,
                                const bool                    useAbc,
                                const bool                    useMass,
                                const TermsType*              devTerms,
                                const IndicesType*            devSysIdx,
                                const int*                    atomStarts,
                                const int                     numIters,
                                double*                       positions,
                                double*                       grad,
                                double*                       velocities,
                                double*                       alphas,
                                double*                       dts,
                                int*                          nStepsPositive,
                                const double*                 masses,
                                double*                       energyOuts,
                                uint8_t*                      statuses,
                                const double                  chiralWeight,
                                const double                  fourthDimWeight,
                                const cudaStream_t            stream) {
  if (numMols == 0) {
    return cudaSuccess;
  }
  firePerMolKernel<MaxAtoms, UseSharedMem, FFType, HasConstraints, TermsType, IndicesType>
    <<<numMols, kFirePerMolBlockSize, 0, stream>>>(numIters,
                                                   params,
                                                   takeHalfStepBack,
                                                   useAbc,
                                                   useMass,
                                                   devTerms,
                                                   devSysIdx,
                                                   molIdList,
                                                   atomStarts,
                                                   positions,
                                                   grad,
                                                   velocities,
                                                   alphas,
                                                   dts,
                                                   nStepsPositive,
                                                   masses,
                                                   energyOuts,
                                                   statuses,
                                                   chiralWeight,
                                                   fourthDimWeight);
  return cudaGetLastError();
}

template <ForceFieldType FFType, bool HasConstraints, typename TermsType, typename IndicesType>
cudaError_t dispatchByMaxAtoms(const int                     numMols,
                               const int*                    molIds,
                               const int                     maxAtoms,
                               const int*                    atomStarts,
                               const FirePerMolKernelParams& params,
                               const bool                    takeHalfStepBack,
                               const bool                    useAbc,
                               const bool                    useMass,
                               const TermsType*              devTerms,
                               const IndicesType*            devSysIdx,
                               const int                     numIters,
                               double*                       positions,
                               double*                       grad,
                               double*                       velocities,
                               double*                       alphas,
                               double*                       dts,
                               int*                          nStepsPositive,
                               const double*                 masses,
                               double*                       energyOuts,
                               uint8_t*                      statuses,
                               const double                  chiralWeight,
                               const double                  fourthDimWeight,
                               const cudaStream_t            stream) {
  auto launchBucket = [&](auto bucketTag) {
    constexpr int  kBucket    = decltype(bucketTag)::value;
    constexpr bool kUseShared = (kBucket <= 128);
    return launchKernelForSize<kBucket, kUseShared, FFType, HasConstraints>(numMols,
                                                                            molIds,
                                                                            params,
                                                                            takeHalfStepBack,
                                                                            useAbc,
                                                                            useMass,
                                                                            devTerms,
                                                                            devSysIdx,
                                                                            atomStarts,
                                                                            numIters,
                                                                            positions,
                                                                            grad,
                                                                            velocities,
                                                                            alphas,
                                                                            dts,
                                                                            nStepsPositive,
                                                                            masses,
                                                                            energyOuts,
                                                                            statuses,
                                                                            chiralWeight,
                                                                            fourthDimWeight,
                                                                            stream);
  };

  if (maxAtoms <= 32) {
    return launchBucket(std::integral_constant<int, 32>{});
  }
  if (maxAtoms <= 64) {
    return launchBucket(std::integral_constant<int, 64>{});
  }
  if (maxAtoms <= 96) {
    return launchBucket(std::integral_constant<int, 96>{});
  }
  if (maxAtoms <= 128) {
    return launchBucket(std::integral_constant<int, 128>{});
  }
  if (maxAtoms <= 256) {
    return launchBucket(std::integral_constant<int, 256>{});
  }
  return launchBucket(std::integral_constant<int, 2048>{});
}

}  // namespace

cudaError_t launchFirePerMolKernel(int                                       numMols,
                                   const int*                                molIds,
                                   int                                       maxAtoms,
                                   const int*                                atomStarts,
                                   const FireOptions&                        fireOptions,
                                   int                                       numIters,
                                   double                                    gradTol,
                                   const MMFF::EnergyForceContribsDevicePtr& terms,
                                   const MMFF::BatchedIndicesDevicePtr&      systemIndices,
                                   const bool                                hasConstraints,
                                   double*                                   positions,
                                   double*                                   grad,
                                   double*                                   velocities,
                                   double*                                   alphas,
                                   double*                                   dts,
                                   int*                                      nStepsPositive,
                                   const double*                             masses,
                                   double*                                   energyOuts,
                                   uint8_t*                                  statuses,
                                   cudaStream_t                              stream) {
  if (numMols == 0) {
    return cudaSuccess;
  }
  const AsyncDevicePtr<MMFF::EnergyForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<MMFF::BatchedIndicesDevicePtr>      devSysIdx(systemIndices, stream);
  const FirePerMolKernelParams                             params  = buildKernelParams(fireOptions, gradTol);
  const bool                                               useMass = fireOptions.useMass && masses != nullptr;
  if (hasConstraints) {
    return dispatchByMaxAtoms<ForceFieldType::MMFF, true>(numMols,
                                                          molIds,
                                                          maxAtoms,
                                                          atomStarts,
                                                          params,
                                                          fireOptions.takeHalfStepBack,
                                                          fireOptions.abcCorrection,
                                                          useMass,
                                                          devTerms.data(),
                                                          devSysIdx.data(),
                                                          numIters,
                                                          positions,
                                                          grad,
                                                          velocities,
                                                          alphas,
                                                          dts,
                                                          nStepsPositive,
                                                          masses,
                                                          energyOuts,
                                                          statuses,
                                                          /*chiralWeight=*/1.0,
                                                          /*fourthDimWeight=*/1.0,
                                                          stream);
  }
  return dispatchByMaxAtoms<ForceFieldType::MMFF, false>(numMols,
                                                         molIds,
                                                         maxAtoms,
                                                         atomStarts,
                                                         params,
                                                         fireOptions.takeHalfStepBack,
                                                         fireOptions.abcCorrection,
                                                         useMass,
                                                         devTerms.data(),
                                                         devSysIdx.data(),
                                                         numIters,
                                                         positions,
                                                         grad,
                                                         velocities,
                                                         alphas,
                                                         dts,
                                                         nStepsPositive,
                                                         masses,
                                                         energyOuts,
                                                         statuses,
                                                         /*chiralWeight=*/1.0,
                                                         /*fourthDimWeight=*/1.0,
                                                         stream);
}

cudaError_t launchFirePerMolKernelETK(int                                             numMols,
                                      const int*                                      molIds,
                                      int                                             maxAtoms,
                                      const int*                                      atomStarts,
                                      const FireOptions&                              fireOptions,
                                      int                                             numIters,
                                      double                                          gradTol,
                                      const DistGeom::Energy3DForceContribsDevicePtr& terms,
                                      const DistGeom::BatchedIndices3DDevicePtr&      systemIndices,
                                      double*                                         positions,
                                      double*                                         grad,
                                      double*                                         velocities,
                                      double*                                         alphas,
                                      double*                                         dts,
                                      int*                                            nStepsPositive,
                                      const double*                                   masses,
                                      double*                                         energyOuts,
                                      uint8_t*                                        statuses,
                                      cudaStream_t                                    stream) {
  if (numMols == 0) {
    return cudaSuccess;
  }
  const AsyncDevicePtr<DistGeom::Energy3DForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<DistGeom::BatchedIndices3DDevicePtr>      devSysIdx(systemIndices, stream);
  const FirePerMolKernelParams                                   params  = buildKernelParams(fireOptions, gradTol);
  const bool                                                     useMass = fireOptions.useMass && masses != nullptr;
  return dispatchByMaxAtoms<ForceFieldType::ETK, false>(numMols,
                                                        molIds,
                                                        maxAtoms,
                                                        atomStarts,
                                                        params,
                                                        fireOptions.takeHalfStepBack,
                                                        fireOptions.abcCorrection,
                                                        useMass,
                                                        devTerms.data(),
                                                        devSysIdx.data(),
                                                        numIters,
                                                        positions,
                                                        grad,
                                                        velocities,
                                                        alphas,
                                                        dts,
                                                        nStepsPositive,
                                                        masses,
                                                        energyOuts,
                                                        statuses,
                                                        /*chiralWeight=*/1.0,
                                                        /*fourthDimWeight=*/1.0,
                                                        stream);
}

cudaError_t launchFirePerMolKernelDG(int                                           numMols,
                                     const int*                                    molIds,
                                     int                                           maxAtoms,
                                     const int*                                    atomStarts,
                                     const FireOptions&                            fireOptions,
                                     int                                           numIters,
                                     double                                        gradTol,
                                     const DistGeom::EnergyForceContribsDevicePtr& terms,
                                     const DistGeom::BatchedIndicesDevicePtr&      systemIndices,
                                     double*                                       positions,
                                     double*                                       grad,
                                     double*                                       velocities,
                                     double*                                       alphas,
                                     double*                                       dts,
                                     int*                                          nStepsPositive,
                                     const double*                                 masses,
                                     double*                                       energyOuts,
                                     double                                        chiralWeight,
                                     double                                        fourthDimWeight,
                                     uint8_t*                                      statuses,
                                     cudaStream_t                                  stream) {
  if (numMols == 0) {
    return cudaSuccess;
  }
  const AsyncDevicePtr<DistGeom::EnergyForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<DistGeom::BatchedIndicesDevicePtr>      devSysIdx(systemIndices, stream);
  const FirePerMolKernelParams                                 params  = buildKernelParams(fireOptions, gradTol);
  const bool                                                   useMass = fireOptions.useMass && masses != nullptr;
  return dispatchByMaxAtoms<ForceFieldType::DG, false>(numMols,
                                                       molIds,
                                                       maxAtoms,
                                                       atomStarts,
                                                       params,
                                                       fireOptions.takeHalfStepBack,
                                                       fireOptions.abcCorrection,
                                                       useMass,
                                                       devTerms.data(),
                                                       devSysIdx.data(),
                                                       numIters,
                                                       positions,
                                                       grad,
                                                       velocities,
                                                       alphas,
                                                       dts,
                                                       nStepsPositive,
                                                       masses,
                                                       energyOuts,
                                                       statuses,
                                                       chiralWeight,
                                                       fourthDimWeight,
                                                       stream);
}

}  // namespace nvMolKit
