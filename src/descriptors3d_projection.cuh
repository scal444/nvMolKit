// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_PROJECTION_CUH
#define NVMOLKIT_DESCRIPTORS3D_PROJECTION_CUH

#include <cmath>

#include "src/descriptors3d.h"
#include "src/descriptors3d_kernel.cuh"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit::descriptors3d_detail {

template <typename Real> struct ProjectionState {
  Real matrix[9];
  Real eigenvectors[9];
  Real eigenvalues[3];
  Real rotationCosine;
  Real rotationSine;
  int  swapEigenpairs;
};

template <typename Real>
__device__ __forceinline__ void jacobiRotateProjection(ProjectionState<Real>& state,
                                                       const int              first,
                                                       const int              second,
                                                       const int              remaining,
                                                       const int              laneInGroup,
                                                       const unsigned         warpMask) {
  if (laneInGroup == 0) {
    Real& diagonalFirst  = state.matrix[first * 3 + first];
    Real& diagonalSecond = state.matrix[second * 3 + second];
    Real& offDiagonal    = state.matrix[first * 3 + second];
    if (offDiagonal == Real(0)) {
      state.rotationCosine = Real(1);
      state.rotationSine   = Real(0);
    } else {
      const Real theta     = (diagonalSecond - diagonalFirst) / (Real(2) * offDiagonal);
      const Real tangent   = fabs(theta) > Real(1e18) ?
                               Real(0.5) / theta :
                               copysign(Real(1), theta) / (fabs(theta) + sqrt(theta * theta + Real(1)));
      state.rotationCosine = Real(1) / sqrt(tangent * tangent + Real(1));
      state.rotationSine   = tangent * state.rotationCosine;
      diagonalFirst -= tangent * offDiagonal;
      diagonalSecond += tangent * offDiagonal;
      offDiagonal                      = Real(0);
      state.matrix[second * 3 + first] = Real(0);
    }
  }
  __syncwarp(warpMask);

  if (laneInGroup == 0) {
    const Real remainingFirst            = state.matrix[remaining * 3 + first];
    const Real remainingSecond           = state.matrix[remaining * 3 + second];
    state.matrix[remaining * 3 + first]  = state.rotationCosine * remainingFirst - state.rotationSine * remainingSecond;
    state.matrix[first * 3 + remaining]  = state.matrix[remaining * 3 + first];
    state.matrix[remaining * 3 + second] = state.rotationSine * remainingFirst + state.rotationCosine * remainingSecond;
    state.matrix[second * 3 + remaining] = state.matrix[remaining * 3 + second];
  }
  if (laneInGroup < 3) {
    const int  row                       = laneInGroup;
    const Real vectorFirst               = state.eigenvectors[row * 3 + first];
    const Real vectorSecond              = state.eigenvectors[row * 3 + second];
    state.eigenvectors[row * 3 + first]  = state.rotationCosine * vectorFirst - state.rotationSine * vectorSecond;
    state.eigenvectors[row * 3 + second] = state.rotationSine * vectorFirst + state.rotationCosine * vectorSecond;
  }
  __syncwarp(warpMask);
}

template <typename Real>
__device__ __forceinline__ void swapProjectionEigenpairs(ProjectionState<Real>& state,
                                                         const int              first,
                                                         const int              second,
                                                         const int              laneInGroup,
                                                         const unsigned         warpMask) {
  if (laneInGroup == 0) {
    state.swapEigenpairs = state.eigenvalues[second] > state.eigenvalues[first];
    if (state.swapEigenpairs) {
      const Real eigenvalue     = state.eigenvalues[first];
      state.eigenvalues[first]  = state.eigenvalues[second];
      state.eigenvalues[second] = eigenvalue;
    }
  }
  __syncwarp(warpMask);
  if (state.swapEigenpairs && laneInGroup < 3) {
    const int  row                       = laneInGroup;
    const Real component                 = state.eigenvectors[row * 3 + first];
    state.eigenvectors[row * 3 + first]  = state.eigenvectors[row * 3 + second];
    state.eigenvectors[row * 3 + second] = component;
  }
  __syncwarp(warpMask);
}

template <typename Real>
__device__ __forceinline__ void diagonalizeProjection(ProjectionState<Real>& state,
                                                      const int              laneInGroup,
                                                      const unsigned         warpMask) {
  for (int i = laneInGroup; i < 9; i += kGroupSize) {
    state.eigenvectors[i] = i == 0 || i == 4 || i == 8 ? Real(1) : Real(0);
  }
  __syncwarp(warpMask);

  constexpr int kMaxSweeps = 16;
  for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
    jacobiRotateProjection(state, 0, 1, 2, laneInGroup, warpMask);
    jacobiRotateProjection(state, 0, 2, 1, laneInGroup, warpMask);
    jacobiRotateProjection(state, 1, 2, 0, laneInGroup, warpMask);
  }

  if (laneInGroup < 3) {
    state.eigenvalues[laneInGroup] = fabs(state.matrix[laneInGroup * 3 + laneInGroup]);
  }
  __syncwarp(warpMask);
  swapProjectionEigenpairs(state, 0, 1, laneInGroup, warpMask);
  swapProjectionEigenpairs(state, 0, 2, laneInGroup, warpMask);
  swapProjectionEigenpairs(state, 1, 2, laneInGroup, warpMask);
}

template <typename Real>
__device__ __forceinline__ void jacobiRotateProjectionSerial(ProjectionState<Real>& state,
                                                             const int              first,
                                                             const int              second,
                                                             const int              remaining) {
  Real& diagonalFirst  = state.matrix[first * 3 + first];
  Real& diagonalSecond = state.matrix[second * 3 + second];
  Real& offDiagonal    = state.matrix[first * 3 + second];
  if (offDiagonal == Real(0)) {
    return;
  }
  const Real theta   = (diagonalSecond - diagonalFirst) / (Real(2) * offDiagonal);
  const Real tangent = fabs(theta) > Real(1e18) ?
                         Real(0.5) / theta :
                         copysign(Real(1), theta) / (fabs(theta) + sqrt(theta * theta + Real(1)));
  const Real cosine  = Real(1) / sqrt(tangent * tangent + Real(1));
  const Real sine    = tangent * cosine;
  diagonalFirst -= tangent * offDiagonal;
  diagonalSecond += tangent * offDiagonal;
  offDiagonal                          = Real(0);
  state.matrix[second * 3 + first]     = Real(0);
  const Real remainingFirst            = state.matrix[remaining * 3 + first];
  const Real remainingSecond           = state.matrix[remaining * 3 + second];
  state.matrix[remaining * 3 + first]  = cosine * remainingFirst - sine * remainingSecond;
  state.matrix[first * 3 + remaining]  = state.matrix[remaining * 3 + first];
  state.matrix[remaining * 3 + second] = sine * remainingFirst + cosine * remainingSecond;
  state.matrix[second * 3 + remaining] = state.matrix[remaining * 3 + second];
  for (int row = 0; row < 3; ++row) {
    const Real vectorFirst               = state.eigenvectors[row * 3 + first];
    const Real vectorSecond              = state.eigenvectors[row * 3 + second];
    state.eigenvectors[row * 3 + first]  = cosine * vectorFirst - sine * vectorSecond;
    state.eigenvectors[row * 3 + second] = sine * vectorFirst + cosine * vectorSecond;
  }
}

template <typename Real>
__device__ __forceinline__ void swapProjectionEigenpairsSerial(ProjectionState<Real>& state,
                                                               const int              first,
                                                               const int              second) {
  const Real eigenvalue     = state.eigenvalues[first];
  state.eigenvalues[first]  = state.eigenvalues[second];
  state.eigenvalues[second] = eigenvalue;
  for (int row = 0; row < 3; ++row) {
    const Real component                 = state.eigenvectors[row * 3 + first];
    state.eigenvectors[row * 3 + first]  = state.eigenvectors[row * 3 + second];
    state.eigenvectors[row * 3 + second] = component;
  }
}

template <typename Real> __device__ __forceinline__ void diagonalizeProjectionSerial(ProjectionState<Real>& state) {
  for (int i = 0; i < 9; ++i) {
    state.eigenvectors[i] = Real(0);
  }
  state.eigenvectors[0]    = Real(1);
  state.eigenvectors[4]    = Real(1);
  state.eigenvectors[8]    = Real(1);
  constexpr int kMaxSweeps = 16;
  for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
    const Real offDiagonal =
      state.matrix[1] * state.matrix[1] + state.matrix[2] * state.matrix[2] + state.matrix[5] * state.matrix[5];
    const Real diagonal =
      state.matrix[0] * state.matrix[0] + state.matrix[4] * state.matrix[4] + state.matrix[8] * state.matrix[8];
    if (!(offDiagonal > Real(1e-14) * Real(1e-14) * diagonal)) {
      break;
    }
    jacobiRotateProjectionSerial(state, 0, 1, 2);
    jacobiRotateProjectionSerial(state, 0, 2, 1);
    jacobiRotateProjectionSerial(state, 1, 2, 0);
  }
  state.eigenvalues[0] = fabs(state.matrix[0]);
  state.eigenvalues[1] = fabs(state.matrix[4]);
  state.eigenvalues[2] = fabs(state.matrix[8]);
  if (state.eigenvalues[1] > state.eigenvalues[0]) {
    swapProjectionEigenpairsSerial(state, 0, 1);
  }
  if (state.eigenvalues[2] > state.eigenvalues[0]) {
    swapProjectionEigenpairsSerial(state, 0, 2);
  }
  if (state.eigenvalues[2] > state.eigenvalues[1]) {
    swapProjectionEigenpairsSerial(state, 1, 2);
  }
}

template <typename Real>
__device__ __forceinline__ void computeProjectionCentroid(const ConformerAtoms& atoms,
                                                          const int             laneInGroup,
                                                          Real&                 centroidX,
                                                          Real&                 centroidY,
                                                          Real&                 centroidZ) {
  Real sumX = 0;
  Real sumY = 0;
  Real sumZ = 0;
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    sumX += static_cast<Real>(atoms.positions[atomIdx * 3 + 0]);
    sumY += static_cast<Real>(atoms.positions[atomIdx * 3 + 1]);
    sumZ += static_cast<Real>(atoms.positions[atomIdx * 3 + 2]);
  }
  const Real inverseAtoms = Real(1) / static_cast<Real>(atoms.numAtoms > 0 ? atoms.numAtoms : 1);
  centroidX               = groupAllReduceSum(sumX) * inverseAtoms;
  centroidY               = groupAllReduceSum(sumY) * inverseAtoms;
  centroidZ               = groupAllReduceSum(sumZ) * inverseAtoms;
}

template <int kAxis, typename Real>
__device__ __forceinline__ Real centeredProjectionCoordinate(const ConformerAtoms& atoms,
                                                             const int             atomIdx,
                                                             const Real            centroidX,
                                                             const Real            centroidY,
                                                             const Real            centroidZ) {
  if constexpr (kAxis == 0) {
    return static_cast<Real>(atoms.positions[atomIdx * 3 + 0]) - centroidX;
  } else if constexpr (kAxis == 1) {
    return static_cast<Real>(atoms.positions[atomIdx * 3 + 1]) - centroidY;
  } else {
    return static_cast<Real>(atoms.positions[atomIdx * 3 + 2]) - centroidZ;
  }
}

template <int kFirstAxis, int kSecondAxis, typename Real>
__device__ __forceinline__ Real computeCovarianceTerm(const ConformerAtoms& atoms,
                                                      const double*         weights,
                                                      const int             laneInGroup,
                                                      const Real            centroidX,
                                                      const Real            centroidY,
                                                      const Real            centroidZ) {
  Real sum = 0;
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    const Real weight = weights == nullptr ? Real(1) : static_cast<Real>(weights[atomIdx]);
    const Real first  = centeredProjectionCoordinate<kFirstAxis>(atoms, atomIdx, centroidX, centroidY, centroidZ);
    const Real second = centeredProjectionCoordinate<kSecondAxis>(atoms, atomIdx, centroidX, centroidY, centroidZ);
    sum += weight * first * second;
  }
  return groupAllReduceSum(sum);
}

template <typename Real, bool kCooperativeDiagonalization>
__device__ __forceinline__ void computeProjectionPca(const ConformerAtoms&  atoms,
                                                     const double*          weights,
                                                     const int              laneInGroup,
                                                     const Real             centroidX,
                                                     const Real             centroidY,
                                                     const Real             centroidZ,
                                                     const unsigned         warpMask,
                                                     ProjectionState<Real>& state) {
  Real inverseWeight;
  if (weights == nullptr) {
    inverseWeight = Real(1) / static_cast<Real>(atoms.numAtoms > 0 ? atoms.numAtoms : 1);
  } else {
    Real totalWeight = 0;
    for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
      totalWeight += static_cast<Real>(weights[atomIdx]);
    }
    totalWeight   = groupAllReduceSum(totalWeight);
    inverseWeight = fabs(totalWeight) < Real(1e-4) ? Real(1) : Real(1) / totalWeight;
  }

  Real covariance = computeCovarianceTerm<0, 0>(atoms, weights, laneInGroup, centroidX, centroidY, centroidZ);
  if (laneInGroup == 0) {
    state.matrix[0] = covariance * inverseWeight;
  }
  covariance = computeCovarianceTerm<0, 1>(atoms, weights, laneInGroup, centroidX, centroidY, centroidZ);
  if (laneInGroup == 0) {
    state.matrix[1] = covariance * inverseWeight;
    state.matrix[3] = state.matrix[1];
  }
  covariance = computeCovarianceTerm<0, 2>(atoms, weights, laneInGroup, centroidX, centroidY, centroidZ);
  if (laneInGroup == 0) {
    state.matrix[2] = covariance * inverseWeight;
    state.matrix[6] = state.matrix[2];
  }
  covariance = computeCovarianceTerm<1, 1>(atoms, weights, laneInGroup, centroidX, centroidY, centroidZ);
  if (laneInGroup == 0) {
    state.matrix[4] = covariance * inverseWeight;
  }
  covariance = computeCovarianceTerm<1, 2>(atoms, weights, laneInGroup, centroidX, centroidY, centroidZ);
  if (laneInGroup == 0) {
    state.matrix[5] = covariance * inverseWeight;
    state.matrix[7] = state.matrix[5];
  }
  covariance = computeCovarianceTerm<2, 2>(atoms, weights, laneInGroup, centroidX, centroidY, centroidZ);
  if (laneInGroup == 0) {
    state.matrix[8] = covariance * inverseWeight;
  }
  __syncwarp(warpMask);
  if constexpr (kCooperativeDiagonalization) {
    diagonalizeProjection(state, laneInGroup, warpMask);
  } else {
    if (laneInGroup == 0) {
      diagonalizeProjectionSerial(state);
    }
    __syncwarp(warpMask);
  }
}

template <typename Real>
__device__ __forceinline__ Real projectionScore(const ConformerAtoms&        atoms,
                                                const int                    atomIdx,
                                                const Real                   centroidX,
                                                const Real                   centroidY,
                                                const Real                   centroidZ,
                                                const ProjectionState<Real>& state,
                                                const int                    axis) {
  const Real x = static_cast<Real>(atoms.positions[atomIdx * 3 + 0]) - centroidX;
  const Real y = static_cast<Real>(atoms.positions[atomIdx * 3 + 1]) - centroidY;
  const Real z = static_cast<Real>(atoms.positions[atomIdx * 3 + 2]) - centroidZ;
  return x * state.eigenvectors[axis] + y * state.eigenvectors[3 + axis] + z * state.eigenvectors[6 + axis];
}

template <typename Real> __device__ __forceinline__ Real roundWhim(const Real value) {
  return round(value * Real(1000)) / Real(1000);
}

template <typename Real>
__device__ __forceinline__ Real computeWhimGamma(const ConformerAtoms&        atoms,
                                                 const int                    laneInGroup,
                                                 const Real                   centroidX,
                                                 const Real                   centroidY,
                                                 const Real                   centroidZ,
                                                 const ProjectionState<Real>& state,
                                                 const int                    axis,
                                                 const Real                   threshold) {
  Real symmetricCount  = 0;
  Real asymmetricCount = 0;
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    const Real score       = roundWhim(projectionScore(atoms, atomIdx, centroidX, centroidY, centroidZ, state, axis));
    bool       hasOpposite = false;
    for (int otherIdx = 0; otherIdx < atoms.numAtoms; ++otherIdx) {
      if (otherIdx == atomIdx) {
        continue;
      }
      const Real otherScore = roundWhim(projectionScore(atoms, otherIdx, centroidX, centroidY, centroidZ, state, axis));
      if (fabs(score + otherScore) <= threshold) {
        hasOpposite = true;
        break;
      }
    }
    if (hasOpposite || fabs(score) < threshold) {
      symmetricCount += Real(1);
    } else {
      asymmetricCount += Real(1);
    }
  }
  symmetricCount      = groupAllReduceSum(symmetricCount);
  asymmetricCount     = groupAllReduceSum(asymmetricCount);
  const Real numAtoms = static_cast<Real>(atoms.numAtoms);
  Real       inverseGamma;
  if (symmetricCount == Real(0)) {
    inverseGamma = Real(1) - (asymmetricCount / numAtoms) * log(Real(1) / numAtoms) / log(Real(2));
  } else {
    inverseGamma = Real(1) - ((symmetricCount / numAtoms) * log(symmetricCount / numAtoms) / log(Real(2)) +
                              (asymmetricCount / numAtoms) * log(Real(1) / numAtoms) / log(Real(2)));
  }
  return Real(1) / inverseGamma;
}

template <typename Real, typename OutputReal>
__device__ __forceinline__ void writeWhimChannel(const ConformerAtoms&        atoms,
                                                 const int                    laneInGroup,
                                                 const int                    conformerIdx,
                                                 const int                    channel,
                                                 const Real                   centroidX,
                                                 const Real                   centroidY,
                                                 const Real                   centroidZ,
                                                 const Real                   threshold,
                                                 const ProjectionState<Real>& state,
                                                 OutputReal*                  output) {
  const Real first  = state.eigenvalues[0];
  const Real second = state.eigenvalues[1];
  const Real third  = state.eigenvalues[2];
  const Real total  = first + second + third;

  Real fourthMomentFirst  = 0;
  Real fourthMomentSecond = 0;
  Real fourthMomentThird  = 0;
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    const Real scoreFirst  = projectionScore(atoms, atomIdx, centroidX, centroidY, centroidZ, state, 0);
    const Real scoreSecond = projectionScore(atoms, atomIdx, centroidX, centroidY, centroidZ, state, 1);
    const Real scoreThird  = projectionScore(atoms, atomIdx, centroidX, centroidY, centroidZ, state, 2);
    fourthMomentFirst += scoreFirst * scoreFirst * scoreFirst * scoreFirst;
    fourthMomentSecond += scoreSecond * scoreSecond * scoreSecond * scoreSecond;
    fourthMomentThird += scoreThird * scoreThird * scoreThird * scoreThird;
  }
  fourthMomentFirst   = groupAllReduceSum(fourthMomentFirst);
  fourthMomentSecond  = groupAllReduceSum(fourthMomentSecond);
  fourthMomentThird   = groupAllReduceSum(fourthMomentThird);
  const Real numAtoms = static_cast<Real>(atoms.numAtoms);
  const Real e1       = fourthMomentFirst > Real(0) ? numAtoms * first * first / fourthMomentFirst : Real(0);
  const Real e2       = fourthMomentSecond > Real(0) ? numAtoms * second * second / fourthMomentSecond : Real(0);
  const Real e3       = fourthMomentThird > Real(0) ? numAtoms * third * third / fourthMomentThird : Real(0);

  if (laneInGroup == 0) {
    OutputReal* row          = output + static_cast<size_t>(conformerIdx) * kNumWhimProperties;
    const int   channelStart = channel * 11;
    row[channelStart + 0]    = roundWhim(first);
    row[channelStart + 1]    = roundWhim(second);
    row[channelStart + 2]    = roundWhim(third);
    row[channelStart + 3]    = roundWhim(first / total);
    row[channelStart + 4]    = roundWhim(second / total);
    row[channelStart + 8]    = roundWhim(e1);
    row[channelStart + 9]    = roundWhim(e2);
    row[channelStart + 10]   = roundWhim(e3);
    row[77 + channel]        = roundWhim(total);
    row[84 + channel]        = roundWhim(first * second + first * third + second * third);
    const Real anisotropy =
      Real(0.75) * (fabs(first / total - Real(1) / Real(3)) + fabs(second / total - Real(1) / Real(3)) +
                    fabs(third / total - Real(1) / Real(3)));
    row[93 + channel]  = roundWhim(anisotropy);
    row[100 + channel] = roundWhim((e1 + e2 + e3) / Real(3));
    row[107 + channel] = roundWhim(total + first * second + first * third + second * third + first * second * third);
  }

  Real gammaProduct = Real(1);
  for (int axis = 0; axis < 3; ++axis) {
    const Real gamma = computeWhimGamma(atoms, laneInGroup, centroidX, centroidY, centroidZ, state, axis, threshold);
    gammaProduct *= gamma;
    if (laneInGroup == 0) {
      output[static_cast<size_t>(conformerIdx) * kNumWhimProperties + channel * 11 + 5 + axis] = roundWhim(gamma);
    }
  }
  if (laneInGroup == 0 && channel < 2) {
    output[static_cast<size_t>(conformerIdx) * kNumWhimProperties + 91 + channel] =
      roundWhim(pow(gammaProduct, Real(1) / Real(3)));
  }
}

template <typename OutputReal, typename ComputeReal, bool kComputeWhim>
__global__ void projection3DKernel(const DeviceCoordView coordinates,
                                   const double* __restrict__ whimAtomWeights,
                                   const int8_t* __restrict__ conformerIs3D,
                                   const int32_t* __restrict__ moleculeAtomStarts,
                                   OutputReal* __restrict__ pbfOutput,
                                   OutputReal* __restrict__ whimOutput,
                                   const ComputeReal whimThreshold) {
  __shared__ ProjectionState<ComputeReal> states[kConformersPerBlock];

  const int      lane         = static_cast<int>(threadIdx.x) % kWarpSize;
  const int      laneInGroup  = lane % kGroupSize;
  const int      groupInBlock = static_cast<int>(threadIdx.x) / kGroupSize;
  const int      warpStart = (blockIdx.x * kWarpsPerBlock + static_cast<int>(threadIdx.x) / kWarpSize) * kGroupsPerWarp;
  const int      conformerIdx = warpStart + lane / kGroupSize;
  const unsigned warpMask     = __activemask();
  if (warpStart >= coordinates.numConformers) {
    return;
  }

  const ConformerAtoms atoms = loadConformer(coordinates, nullptr, moleculeAtomStarts, conformerIdx);
  ComputeReal          centroidX;
  ComputeReal          centroidY;
  ComputeReal          centroidZ;
  computeProjectionCentroid(atoms, laneInGroup, centroidX, centroidY, centroidZ);

  ProjectionState<ComputeReal>& state = states[groupInBlock];
  computeProjectionPca<ComputeReal,
                       !kComputeWhim>(atoms, nullptr, laneInGroup, centroidX, centroidY, centroidZ, warpMask, state);

  if (pbfOutput != nullptr) {
    ComputeReal distanceSum = 0;
    for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
      distanceSum += fabs(projectionScore(atoms, atomIdx, centroidX, centroidY, centroidZ, state, 2));
    }
    distanceSum = groupAllReduceSum(distanceSum);
    if (laneInGroup == 0 && conformerIdx < coordinates.numConformers) {
      const bool is3D = conformerIs3D == nullptr || conformerIs3D[conformerIdx] != 0;
      pbfOutput[conformerIdx] =
        !atoms.valid                ? static_cast<OutputReal>(nan("")) :
        atoms.numAtoms < 4 || !is3D ? OutputReal(0) :
                                      static_cast<OutputReal>(distanceSum / static_cast<ComputeReal>(atoms.numAtoms));
    }
  }

  if constexpr (kComputeWhim) {
    writeWhimChannel(atoms,
                     laneInGroup,
                     conformerIdx,
                     0,
                     centroidX,
                     centroidY,
                     centroidZ,
                     whimThreshold,
                     state,
                     whimOutput);
    const int moleculeIdx       = atoms.valid ? coordinates.molIndices[conformerIdx] : 0;
    const int moleculeAtomStart = atoms.valid ? moleculeAtomStarts[moleculeIdx] : 0;
    const int totalAtoms        = moleculeAtomStarts[coordinates.nMols];
    for (int channel = 1; channel < 7; ++channel) {
      const double* weights = whimAtomWeights + static_cast<size_t>(channel - 1) * totalAtoms + moleculeAtomStart;
      computeProjectionPca<ComputeReal,
                           false>(atoms, weights, laneInGroup, centroidX, centroidY, centroidZ, warpMask, state);
      writeWhimChannel(atoms,
                       laneInGroup,
                       conformerIdx,
                       channel,
                       centroidX,
                       centroidY,
                       centroidZ,
                       whimThreshold,
                       state,
                       whimOutput);
    }
    if (!atoms.valid) {
      for (int valueIdx = laneInGroup; valueIdx < kNumWhimProperties; valueIdx += kGroupSize) {
        whimOutput[static_cast<size_t>(conformerIdx) * kNumWhimProperties + valueIdx] =
          static_cast<OutputReal>(nan(""));
      }
    }
  }
}

template <typename Real>
void launchProjectionProperties(const DeviceCoordView& coordinates,
                                const double*          whimAtomWeights,
                                const int8_t*          conformerIs3D,
                                const int32_t*         moleculeAtomStarts,
                                Real*                  pbfOutput,
                                Real*                  whimOutput,
                                const double           whimThreshold,
                                const cudaStream_t     stream) {
  if (coordinates.numConformers == 0 || (pbfOutput == nullptr && whimOutput == nullptr)) {
    return;
  }
  const int numBlocks = (coordinates.numConformers + kConformersPerBlock - 1) / kConformersPerBlock;
  if (whimOutput != nullptr) {
    projection3DKernel<Real, double, true><<<numBlocks, kBlockSize, 0, stream>>>(coordinates,
                                                                                 whimAtomWeights,
                                                                                 conformerIs3D,
                                                                                 moleculeAtomStarts,
                                                                                 pbfOutput,
                                                                                 whimOutput,
                                                                                 whimThreshold);
  } else {
    projection3DKernel<Real, Real, false><<<numBlocks, kBlockSize, 0, stream>>>(coordinates,
                                                                                nullptr,
                                                                                conformerIs3D,
                                                                                moleculeAtomStarts,
                                                                                pbfOutput,
                                                                                nullptr,
                                                                                static_cast<Real>(whimThreshold));
  }
  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_PROJECTION_CUH
