// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_PROJECTION_CUH
#define NVMOLKIT_DESCRIPTORS3D_PROJECTION_CUH

#include <algorithm>
#include <cmath>

#include "src/descriptors3d.h"
#include "src/descriptors3d_kernel.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/symmetric_eigenvalues_3x3.cuh"

namespace nvMolKit::descriptors3d_detail {

//! Coordinate PCA of one conformer. Every lane of a group holds an identical copy.
template <typename Real> struct ProjectionState {
  //! float64 for every Real; see centeredPosition().
  double centroidX;
  double centroidY;
  double centroidZ;
  //! Covariance eigenvalue magnitudes in descending order.
  Real   eigenvalues[3];
  //! Row-major; column i is the unit axis of eigenvalues[i].
  Real   eigenvectors[9];
};

//! Group-collective. Unweighted centroid, shared by every PCA channel.
template <typename Real>
__device__ __forceinline__ void computeProjectionCentroid(const ConformerAtoms&  atoms,
                                                          const int              laneInGroup,
                                                          ProjectionState<Real>& state) {
  double sumX = 0;
  double sumY = 0;
  double sumZ = 0;
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    sumX += atoms.positions[atomIdx * 3 + 0];
    sumY += atoms.positions[atomIdx * 3 + 1];
    sumZ += atoms.positions[atomIdx * 3 + 2];
  }
  const double inverseAtoms = 1.0 / static_cast<double>(atoms.numAtoms > 0 ? atoms.numAtoms : 1);
  state.centroidX           = groupAllReduceSum(sumX) * inverseAtoms;
  state.centroidY           = groupAllReduceSum(sumY) * inverseAtoms;
  state.centroidZ           = groupAllReduceSum(sumZ) * inverseAtoms;
}

template <typename Real>
__device__ __forceinline__ void centeredCoordinates(const ConformerAtoms&        atoms,
                                                    const int                    atomIdx,
                                                    const ProjectionState<Real>& state,
                                                    Real&                        x,
                                                    Real&                        y,
                                                    Real&                        z) {
  centeredPosition(atoms.positions, atomIdx, state.centroidX, state.centroidY, state.centroidZ, x, y, z);
}

template <typename Real>
__device__ __forceinline__ void swapEigenpairs(ProjectionState<Real>& state, const int first, const int second) {
  const Real value          = state.eigenvalues[first];
  state.eigenvalues[first]  = state.eigenvalues[second];
  state.eigenvalues[second] = value;
  for (int row = 0; row < 3; ++row) {
    const Real component                 = state.eigenvectors[row * 3 + first];
    state.eigenvectors[row * 3 + first]  = state.eigenvectors[row * 3 + second];
    state.eigenvectors[row * 3 + second] = component;
  }
}

/**
 * @brief Group-collective. Covariance of the centered coordinates about ProjectionState's centroid, in
 *        the order xx, xy, xz, yy, yz, zz. Every lane receives the same values.
 *
 * @p weights (one per atom) scales each atom's contribution and normalizes by their sum, falling back to
 * a plain sum when that total is near zero. A null @p weights gives the unweighted covariance normalized
 * by the atom count.
 */
template <typename Real>
__device__ __forceinline__ void computeProjectionCovariance(const ConformerAtoms&        atoms,
                                                            const double*                weights,
                                                            const int                    laneInGroup,
                                                            const ProjectionState<Real>& state,
                                                            Real (&covariance)[6]) {
  Real totalWeight = 0;
  Real xx          = 0;
  Real xy          = 0;
  Real xz          = 0;
  Real yy          = 0;
  Real yz          = 0;
  Real zz          = 0;
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    const Real weight = weights == nullptr ? Real(1) : static_cast<Real>(weights[atomIdx]);
    Real       x;
    Real       y;
    Real       z;
    centeredCoordinates(atoms, atomIdx, state, x, y, z);
    totalWeight += weight;
    xx += weight * x * x;
    xy += weight * x * y;
    xz += weight * x * z;
    yy += weight * y * y;
    yz += weight * y * z;
    zz += weight * z * z;
  }
  totalWeight = groupAllReduceSum(totalWeight);
  Real inverseWeight;
  if (weights == nullptr) {
    inverseWeight = Real(1) / static_cast<Real>(atoms.numAtoms > 0 ? atoms.numAtoms : 1);
  } else {
    inverseWeight = fabs(totalWeight) < Real(1e-4) ? Real(1) : Real(1) / totalWeight;
  }
  covariance[0] = groupAllReduceSum(xx) * inverseWeight;
  covariance[1] = groupAllReduceSum(xy) * inverseWeight;
  covariance[2] = groupAllReduceSum(xz) * inverseWeight;
  covariance[3] = groupAllReduceSum(yy) * inverseWeight;
  covariance[4] = groupAllReduceSum(yz) * inverseWeight;
  covariance[5] = groupAllReduceSum(zz) * inverseWeight;
}

//! Eigensystem of @p covariance into @p state: eigenvalue magnitudes in descending order with their axes.
template <typename Real>
__device__ __forceinline__ void diagonalizeProjection(const Real (&covariance)[6], ProjectionState<Real>& state) {
  symmetricEigensystemJacobi3x3(covariance[0],
                                covariance[1],
                                covariance[2],
                                covariance[3],
                                covariance[4],
                                covariance[5],
                                state.eigenvalues,
                                state.eigenvectors);
  for (int axis = 0; axis < 3; ++axis) {
    state.eigenvalues[axis] = fabs(state.eigenvalues[axis]);
  }
  if (state.eigenvalues[1] > state.eigenvalues[0]) {
    swapEigenpairs(state, 0, 1);
  }
  if (state.eigenvalues[2] > state.eigenvalues[0]) {
    swapEigenpairs(state, 0, 2);
  }
  if (state.eigenvalues[2] > state.eigenvalues[1]) {
    swapEigenpairs(state, 1, 2);
  }
}

//! Group-collective. computeProjectionCovariance() followed by diagonalizeProjection(); every lane
//! diagonalizes its own identical copy, so the eigensystem is available group-wide.
template <typename Real>
__device__ __forceinline__ void computeProjectionPca(const ConformerAtoms&  atoms,
                                                     const double*          weights,
                                                     const int              laneInGroup,
                                                     ProjectionState<Real>& state) {
  Real covariance[6];
  computeProjectionCovariance(atoms, weights, laneInGroup, state, covariance);
  diagonalizeProjection(covariance, state);
}

//! Centered coordinate of atom @p atomIdx along principal axis @p axis.
template <typename Real>
__device__ __forceinline__ Real
projectionScore(const ConformerAtoms& atoms, const int atomIdx, const ProjectionState<Real>& state, const int axis) {
  Real x;
  Real y;
  Real z;
  centeredCoordinates(atoms, atomIdx, state, x, y, z);
  return x * state.eigenvectors[axis] + y * state.eigenvectors[3 + axis] + z * state.eigenvectors[6 + axis];
}

//! Group-collective. Mean absolute distance of the atoms from the plane of best fit, i.e. their mean
//! absolute score along the smallest principal axis of the unweighted PCA.
template <typename Real>
__device__ __forceinline__ Real computePbf(const ConformerAtoms&        atoms,
                                           const int                    laneInGroup,
                                           const ProjectionState<Real>& state) {
  Real distanceSum = 0;
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    distanceSum += fabs(projectionScore(atoms, atomIdx, state, 2));
  }
  return groupAllReduceSum(distanceSum) / static_cast<Real>(atoms.numAtoms > 0 ? atoms.numAtoms : 1);
}

/**
 * @brief `a / 1000.0`, correctly rounded, without a general float64 division.
 *
 * One Newton correction of the product with the rounded reciprocal. Exhaustively verified bit-identical
 * to IEEE division for every int32 @p a.
 */
__device__ __forceinline__ double thousandthsToUnits(const int32_t a) {
  constexpr double kInverseThousand = 1.0 / 1000.0;
  const double     numerator        = static_cast<double>(a);
  const double     quotient         = numerator * kInverseThousand;
  return fma(fma(-quotient, 1000.0, numerator), kInverseThousand, quotient);
}

//! roundThousandths() for float64, bit-identical, dividing via thousandthsToUnits() in the int32 range.
__device__ __forceinline__ double roundWhimThousandths(const double value) {
  const double thousandths = round(value * 1000.0);
  if (fabs(thousandths) < 2147483648.0) {
    return copysign(thousandthsToUnits(static_cast<int32_t>(thousandths)), thousandths);  // Keeps -0.
  }
  return thousandths / 1000.0;
}

/**
 * @brief RDKit's symmetry comparisons on projections rounded to thousandths.
 *
 * RDKit compares `fabs(a + b) <= threshold` and `fabs(a) < threshold` on float64 values `k / 1000.0`.
 * On integer thousandths these reduce to integer comparisons against the threshold in thousandths,
 * except where a magnitude equals a threshold lying (within float64 noise) on the integer grid.
 *
 * FP64 required: the on-threshold evaluation. Rounded projections live on the same 0.001 grid as the
 * default threshold, so many pair sums land exactly on it, and RDKit's result there depends on how
 * `k / 1000.0` is represented in float64. Evaluating those cases in float32 flips them.
 */
struct WhimThreshold {
  double   value;           //!< Threshold in the projection's units.
  int64_t  mirroredLimit;   //!< Largest off-boundary |a + b| (thousandths) that counts as mirrored.
  int64_t  centeredLimit;   //!< Largest off-boundary |a| (thousandths) that counts as centered.
  int64_t  boundary;        //!< Magnitude decided by RDKit's float64 expression, or -1 if none.
  uint32_t candidateLimit;  //!< mayMirror() bound: every |a + b| that mirrored() can accept.
  bool     exhaustive;      //!< candidateLimit would overflow the filter, so every pair is a candidate.

  __host__ __device__ static WhimThreshold make(const double value) {
    // Rounded int32 projections sum to below 2^32 in magnitude; any larger threshold accepts them all.
    constexpr double  kCap               = 1099511627776.0;  // 2^40
    const double      thousandths        = fmin(value * 1000.0, kCap);
    const double      nearest            = rint(thousandths);
    // float64 noise in `a / 1000.0 + b / 1000.0` stays far below 1e-6 thousandths for int32 inputs.
    const bool        onGrid             = fabs(thousandths - nearest) <= 1e-6;
    const auto        floorLimit         = static_cast<int64_t>(floor(thousandths));
    const int64_t     boundary           = onGrid ? static_cast<int64_t>(nearest) : -1;
    const int64_t     candidate          = boundary > floorLimit ? boundary : floorLimit;
    constexpr int64_t kMaxCandidateLimit = int64_t{1} << 30;
    return WhimThreshold{value,
                         floorLimit,
                         onGrid ? static_cast<int64_t>(nearest) - 1 : floorLimit,
                         boundary,
                         static_cast<uint32_t>(candidate < kMaxCandidateLimit ? candidate : kMaxCandidateLimit),
                         candidate > kMaxCandidateLimit};
  }

  /**
   * @brief Cheap filter for mirrored(): false only if mirrored() is false.
   *
   * `|a + b| <= candidateLimit` as one unsigned comparison. int32 sums wrap modulo 2^32, which can only
   * admit extra candidates; mirrored() decides them.
   */
  __device__ __forceinline__ bool mayMirror(const int32_t a, const int32_t b) const {
    return exhaustive || static_cast<uint32_t>(a) + static_cast<uint32_t>(b) + candidateLimit <= 2u * candidateLimit;
  }

  //! `fabs(a / 1000.0 + b / 1000.0) <= value` for rounded projections @p a and @p b.
  __device__ __forceinline__ bool mirrored(const int32_t a, const int32_t b) const {
    const int64_t sum = llabs(static_cast<int64_t>(a) + static_cast<int64_t>(b));
    if (sum == boundary) {
      return mirroredOnBoundary(a, b, value);
    }
    return sum <= mirroredLimit;
  }

  //! `fabs(a / 1000.0) < value` for rounded projection @p a.
  __device__ __forceinline__ bool centered(const int32_t a) const {
    const int64_t magnitude = llabs(static_cast<int64_t>(a));
    if (magnitude == boundary) {
      return centeredOnBoundary(a, value);
    }
    return magnitude <= centeredLimit;
  }

  // The on-boundary evaluations stay out of line: inlined, the compiler hoists their float64 work out of
  // the pair loop and pays it for every atom rather than for the boundary hits.
  __device__ __noinline__ static bool mirroredOnBoundary(const int32_t a, const int32_t b, const double value) {
    return fabs(thousandthsToUnits(a) + thousandthsToUnits(b)) <= value;
  }

  __device__ __noinline__ static bool centeredOnBoundary(const int32_t a, const double value) {
    return fabs(thousandthsToUnits(a)) < value;
  }
};

/**
 * @brief Arithmetic type of WHIM's PCA and projections, in every precision mode.
 *
 * FP64 required: WHIM's inverse kurtosis, `n * lambda^2 / sum(score^4)` per axis, is scale-invariant.
 * On the near-zero axis of a planar or near-planar conformer it depends only on out-of-plane deviations
 * far below float32 resolution relative to the molecule's extent, so a float32 covariance, eigensystem or
 * projection turns it (and the kurtosis means built from it) into rounding noise. Every other WHIM term is
 * accurate in float32.
 */
using WhimReal = double;

//! WHIM's unweighted channel followed by the atom-property channels.
constexpr int kNumWhimChannels = 1 + kNumAtomPropertyChannels;
//! (channel, principal axis) pairs of WHIM's symmetry search.
constexpr int kNumWhimAxes     = 3 * kNumWhimChannels;

//! Stores @p value narrowed to the output precision @p Real.
template <typename Real> __device__ __forceinline__ void storeOutput(Real* destination, const double value) {
  *destination = static_cast<Real>(value);
}

//! WHIM symmetry term from the symmetric atom count along one axis; @p logInverseAtoms is log(1 / numAtoms).
__device__ __forceinline__ WhimReal whimGamma(const int      symmetricAtoms,
                                              const WhimReal numAtoms,
                                              const WhimReal logInverseAtoms) {
  const WhimReal symmetricCount  = static_cast<WhimReal>(symmetricAtoms);
  const WhimReal asymmetricCount = numAtoms - symmetricCount;
  const WhimReal symmetricShare  = symmetricCount / numAtoms;
  // Without symmetric atoms RDKit drops the first term; with log(1) = 0 it adds an exact zero instead, which
  // leaves the result bit-identical and the lanes of a warp on one path.
  const WhimReal symmetricTerm =
    symmetricShare * log(symmetricAtoms == 0 ? WhimReal(1) : symmetricShare) / log(WhimReal(2));
  const WhimReal inverseGamma =
    WhimReal(1) - (symmetricTerm + (asymmetricCount / numAtoms) * logInverseAtoms / log(WhimReal(2)));
  return WhimReal(1) / inverseGamma;
}

/*
 * One conformer's 114-value WHIM row, in RDKit's CalcWHIM layout: 11 directional values per channel
 * (eigenvalues, two eigenvalue ratios, three symmetry terms, three inverse kurtoses), then 7 totals, 7 pairwise
 * products, 2 symmetry means, 7 anisotropies, 7 kurtosis means and 7 combined terms. Values are computed in
 * WhimReal and stored as @p Real by three writers: per (channel, axis) pair, per
 * channel from its eigenvalues alone, and per channel from its axes' terms.
 */

//! Writes (channel, axis) pair @p item's eigenvalue magnitude, symmetry term and inverse kurtosis; returns the
//! last two in @p gamma and @p kurtosis. @p logInverseAtoms is log(1 / numAtoms).
template <typename Real>
__device__ __forceinline__ void writeWhimAxisTerms(const int      item,
                                                   const WhimReal lambda,
                                                   const WhimReal moment,
                                                   const int      symmetricAtoms,
                                                   const int      numAtomsInt,
                                                   const WhimReal logInverseAtoms,
                                                   Real*          row,
                                                   WhimReal&      gamma,
                                                   WhimReal&      kurtosis) {
  const WhimReal numAtoms = static_cast<WhimReal>(numAtomsInt);
  const int      channel  = item / 3;
  const int      axis     = item % 3;
  kurtosis                = moment > WhimReal(0) ? numAtoms * lambda * lambda / moment : WhimReal(0);
  gamma                   = whimGamma(symmetricAtoms, numAtoms, logInverseAtoms);
  storeOutput<Real>(row + channel * 11 + axis, roundWhimThousandths(lambda));
  storeOutput<Real>(row + channel * 11 + 5 + axis, roundWhimThousandths(gamma));
  storeOutput<Real>(row + channel * 11 + 8 + axis, roundWhimThousandths(kurtosis));
}

//! Writes channel @p channel's values that depend on its eigenvalue magnitudes alone.
template <typename Real>
__device__ __forceinline__ void writeWhimEigenvalueTerms(const int channel,
                                                         const WhimReal (&eigenvalues)[3],
                                                         Real* row) {
  const WhimReal first  = eigenvalues[0];
  const WhimReal second = eigenvalues[1];
  const WhimReal third  = eigenvalues[2];
  const WhimReal total  = first + second + third;
  storeOutput<Real>(row + channel * 11 + 3, roundWhimThousandths(first / total));
  storeOutput<Real>(row + channel * 11 + 4, roundWhimThousandths(second / total));
  storeOutput<Real>(row + 77 + channel, roundWhimThousandths(total));
  storeOutput<Real>(row + 84 + channel, roundWhimThousandths(first * second + first * third + second * third));
  const WhimReal anisotropy = WhimReal(0.75) * (fabs(first / total - WhimReal(1) / WhimReal(3)) +
                                                fabs(second / total - WhimReal(1) / WhimReal(3)) +
                                                fabs(third / total - WhimReal(1) / WhimReal(3)));
  storeOutput<Real>(row + 93 + channel, roundWhimThousandths(anisotropy));
  storeOutput<Real>(
    row + 107 + channel,
    roundWhimThousandths(total + first * second + first * third + second * third + first * second * third));
}

//! Writes channel @p channel's kurtosis mean and, for the first two channels, symmetry mean from its axes'
//! @p gammas and @p kurtoses.
template <typename Real>
__device__ __forceinline__ void writeWhimMeans(const int channel,
                                               const WhimReal (&gammas)[3],
                                               const WhimReal (&kurtoses)[3],
                                               Real* row) {
  storeOutput<Real>(row + 100 + channel, roundWhimThousandths((kurtoses[0] + kurtoses[1] + kurtoses[2]) / WhimReal(3)));
  if (channel < 2) {
    const WhimReal gammaProduct = WhimReal(1) * gammas[0] * gammas[1] * gammas[2];
    storeOutput<Real>(row + 91 + channel, roundWhimThousandths(pow(gammaProduct, WhimReal(1) / WhimReal(3))));
  }
}

/**
 * @brief Warp-collective. Sums each of a lane's 8 @p values over the warp, halving the values each lane
 *        carries at every exchange. Returns the total of value `(lane >> 2) & 7`.
 */
template <typename T> __device__ __forceinline__ T warpTransposedSum8(const T (&values)[8], const int lane) {
  T half[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const bool upper = (lane & 16) != 0;
    const T    send  = upper ? values[k] : values[k + 4];
    const T    keep  = upper ? values[k + 4] : values[k];
    half[k]          = keep + __shfl_xor_sync(0xffffffffu, send, 16);
  }
  T quarter[2];
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const bool upper = (lane & 8) != 0;
    const T    send  = upper ? half[k] : half[k + 2];
    const T    keep  = upper ? half[k + 2] : half[k];
    quarter[k]       = keep + __shfl_xor_sync(0xffffffffu, send, 8);
  }
  const bool upper = (lane & 4) != 0;
  T value = (upper ? quarter[1] : quarter[0]) + __shfl_xor_sync(0xffffffffu, upper ? quarter[0] : quarter[1], 4);
  value += __shfl_xor_sync(0xffffffffu, value, 2);
  value += __shfl_xor_sync(0xffffffffu, value, 1);
  return value;
}

//! Warp-collective. warpTransposedSum8() for 4 @p values: returns the total of value `(lane >> 3) & 3`.
template <typename T> __device__ __forceinline__ T warpTransposedSum4(const T (&values)[4], const int lane) {
  T half[2];
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const bool upper = (lane & 16) != 0;
    const T    send  = upper ? values[k] : values[k + 2];
    const T    keep  = upper ? values[k + 2] : values[k];
    half[k]          = keep + __shfl_xor_sync(0xffffffffu, send, 16);
  }
  const bool upper = (lane & 8) != 0;
  T          value = (upper ? half[1] : half[0]) + __shfl_xor_sync(0xffffffffu, upper ? half[0] : half[1], 8);
  value += __shfl_xor_sync(0xffffffffu, value, 4);
  value += __shfl_xor_sync(0xffffffffu, value, 2);
  value += __shfl_xor_sync(0xffffffffu, value, 1);
  return value;
}

/**
 * @brief Warp-collective. Sums of one PCA channel over the conformer's atoms, in the order total weight,
 *        xx, xy, xz, yy, yz, zz (computeProjectionCovariance() before normalization); returns sum
 *        `(lane >> 2) & 7` (see warpTransposedSum8()).
 *
 * Coordinates come from @p centered (float64, one array per axis of @p stride atoms) and are narrowed to
 * @p Real; @p weights (one per atom) scales each atom's contribution, and null weights every atom by one.
 */
template <typename Real>
__device__ __forceinline__ Real accumulateCovarianceSums(const int     numAtoms,
                                                         const int     lane,
                                                         const double* weights,
                                                         const double* centered,
                                                         const int     stride) {
  Real totalWeight = 0;
  Real xx          = 0;
  Real xy          = 0;
  Real xz          = 0;
  Real yy          = 0;
  Real yz          = 0;
  Real zz          = 0;
  if (weights == nullptr) {
    // A unit weight leaves every product exact, so these are the weighted sums for weight one.
    for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
      const Real x = static_cast<Real>(centered[atomIdx]);
      const Real y = static_cast<Real>(centered[stride + atomIdx]);
      const Real z = static_cast<Real>(centered[2 * stride + atomIdx]);
      totalWeight += Real(1);
      xx += x * x;
      xy += x * y;
      xz += x * z;
      yy += y * y;
      yz += y * z;
      zz += z * z;
    }
  } else {
    for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
      const Real weight = static_cast<Real>(weights[atomIdx]);
      const Real x      = static_cast<Real>(centered[atomIdx]);
      const Real y      = static_cast<Real>(centered[stride + atomIdx]);
      const Real z      = static_cast<Real>(centered[2 * stride + atomIdx]);
      totalWeight += weight;
      xx += weight * x * x;
      xy += weight * x * y;
      xz += weight * x * z;
      yy += weight * y * y;
      yz += weight * y * z;
      zz += weight * z * z;
    }
  }
  const Real partials[8] = {totalWeight, xx, xy, xz, yy, yz, zz, Real(0)};
  return warpTransposedSum8(partials, lane);
}

/**
 * @brief Covariance from accumulateCovarianceSums() output, normalized as computeProjectionCovariance()
 *        does: by the atom count for @p unweighted sums, else by the total weight unless it is near zero.
 */
template <typename Real>
__device__ __forceinline__ void normalizeCovariance(const Real (&sums)[7],
                                                    const bool unweighted,
                                                    const int  numAtoms,
                                                    Real (&covariance)[6]) {
  Real inverseWeight;
  if (unweighted) {
    inverseWeight = Real(1) / static_cast<Real>(numAtoms > 0 ? numAtoms : 1);
  } else {
    inverseWeight = fabs(sums[0]) < Real(1e-4) ? Real(1) : Real(1) / sums[0];
  }
  for (int i = 0; i < 6; ++i) {
    covariance[i] = sums[i + 1] * inverseWeight;
  }
}

/**
 * @brief Warp-collective. Number of atoms whose rounded projection @p values (one axis of one channel) is
 *        centered, or mirrored by another atom within the threshold, comparing every pair as RDKit does.
 *        Returns the warp total.
 */
__device__ __forceinline__ int countWhimSymmetric(const int32_t*       values,
                                                  const int            numAtoms,
                                                  const int            lane,
                                                  const WhimThreshold& threshold) {
  int symmetric = 0;
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    const int32_t own   = values[atomIdx];
    bool          found = threshold.centered(own);
    for (int otherIdx = 0; otherIdx < numAtoms && !found; ++otherIdx) {
      const int32_t other = values[otherIdx];
      found               = otherIdx != atomIdx && threshold.mayMirror(own, other) && threshold.mirrored(own, other);
    }
    symmetric += found ? 1 : 0;
  }
  return groupAllReduceSum<kWarpSize>(symmetric);
}

//! Warps per block of the WHIM kernel, which computes one conformer per block, for grids of several waves.
constexpr int kWhimWarps     = 4;
//! Blocks per SM the register budget of that kernel allows (shared memory permitting).
constexpr int kWhimMinBlocks = 8;
//! PCA slots of the WHIM kernel: WHIM's channels, then PBF's own PCA when it computes in another precision.
constexpr int kWhimPcaSlots  = kNumWhimChannels + 1;
static_assert(kWhimPcaSlots <= kWarpSize && kNumWhimAxes <= kWarpSize, "per-lane PCA slots and WHIM axes");

//! Per-block values of the WHIM kernel.
template <int kWarps> struct WhimBlockShared {
  double centroidPartials[kWarps][3];
  double logInverseAtoms;  //!< log(1 / numAtoms), for the symmetry terms.
  union {
    double covarianceSums[kWhimPcaSlots][7];  //!< Until the eigensolves.
    struct {
      double gammas[kNumWhimChannels][3];
      double kurtoses[kNumWhimChannels][3];
    } axisTerms;  //!< After the symmetry search.
  };
  double eigenvalues[kWhimPcaSlots][3];
  double eigenvectors[kWhimPcaSlots][9];
  double fourthMoments[kNumWhimChannels][3];
  int    symmetricCounts[kNumWhimAxes];  //!< Symmetric atoms of each (channel, axis) pair.
};

/**
 * @brief Per-conformer scratch layout of the WHIM kernel for rows of up to `stride` atoms, in elements from
 *        its start: float64 centered coordinates (x, y, z arrays) and atom-property weights (one array per
 *        channel), then int32 rounded projections (one array per channel and axis).
 */
struct WhimScratchLayout {
  int stride;

  __host__ __device__ static WhimScratchLayout make(const int maxAtoms) {
    const int atoms = maxAtoms > 1 ? maxAtoms : 1;
    return WhimScratchLayout{(atoms + 3) / 4 * 4};  // Keeps every array 16-byte aligned.
  }
  __host__ __device__ size_t doubles() const { return static_cast<size_t>(3 + kNumAtomPropertyChannels) * stride; }
  __host__ __device__ size_t ints() const { return static_cast<size_t>(kNumWhimAxes) * stride; }
  __host__ __device__ size_t bytes() const { return doubles() * sizeof(double) + ints() * sizeof(int32_t); }
};

/**
 * @brief In-bounds coordinate range of a conformer, as loadConformer() resolves it before checking the
 *        molecule's atom count: null positions and zero atoms when the conformer index, molecule index or
 *        atom range is out of bounds. Reading it first lets the coordinates load alongside that check.
 */
struct ConformerRange {
  const double* positions   = nullptr;
  int           moleculeIdx = 0;
  int           numAtoms    = 0;
};

__device__ __forceinline__ ConformerRange loadConformerRange(const DeviceCoordView& coordinates,
                                                             const int              conformerIdx) {
  ConformerRange range;
  if (conformerIdx >= coordinates.numConformers) {
    return range;
  }
  const int     moleculeIdx = coordinates.molIndices[conformerIdx];
  const int64_t atomStart   = coordinates.atomStarts[conformerIdx];
  const int64_t atomStop    = coordinates.atomStarts[conformerIdx + 1];
  if (moleculeIdx < 0 || moleculeIdx >= coordinates.nMols || atomStart < 0 || atomStop <= atomStart ||
      atomStop > coordinates.numAtoms) {
    return range;
  }
  range.positions   = coordinates.positions + static_cast<size_t>(atomStart) * 3;
  range.moleculeIdx = moleculeIdx;
  range.numAtoms    = static_cast<int>(atomStop - atomStart);
  return range;
}

/**
 * @brief PBF and WHIM of conformer `conformerBegin + blockIdx.x`, one block per conformer.
 *
 * The block centers the coordinates in float64 into the scratch, then each warp sums the covariance of its
 * PCA slots: WHIM's unweighted and atom-property channels in WhimReal and, when PBF is requested, PBF's
 * unweighted PCA in @p Real.
 * One lane per slot diagonalizes. Warps then project and round each channel's coordinates, PBF is written,
 * and the warps share the symmetry searches of the (channel, axis) pairs. Finally the first warp writes the
 * WHIM row. The scratch (see WhimScratchLayout) is shared memory, or with @p kGlobalScratch a block per
 * conformer of @p globalScratch. Rows longer than @p whimMaxAtoms are treated as invalid, and invalid rows
 * produce NaN.
 */
template <int kWarps, typename Real, bool kComputePbf, bool kGlobalScratch>
__global__ void __launch_bounds__(kWarps* kWarpSize, kWhimMinBlocks) whimKernel(const DeviceCoordView coordinates,
                                                                                const Property3DDeviceInputs inputs,
                                                                                const int conformerBegin,
                                                                                Real* __restrict__ pbfOutput,
                                                                                Real* __restrict__ whimOutput,
                                                                                char* __restrict__ globalScratch,
                                                                                const int               whimMaxAtoms,
                                                                                const WhimScratchLayout layout,
                                                                                const WhimThreshold     threshold) {
  extern __shared__ double dynamicScratch[];
  __shared__ WhimBlockShared<kWarps> shared;
  constexpr int                      kWhimThreads = kWarps * kWarpSize;
  static_assert(kWarps >= 3, "the eigensolve phase uses three warps");
  constexpr bool kSeparatePbf = kComputePbf;
  constexpr int  kPcaSlots    = kNumWhimChannels + (kSeparatePbf ? 1 : 0);
  //! Slot whose eigenvectors PBF projects on.
  constexpr int  kPbfSlot     = kSeparatePbf ? kNumWhimChannels : 0;
  const int      tid          = static_cast<int>(threadIdx.x);
  const int      lane         = tid % kWarpSize;
  const int      warp         = tid / kWarpSize;
  const int      conformerIdx = conformerBegin + static_cast<int>(blockIdx.x);
  const int      stride       = layout.stride;

  // The coordinate range, then the molecule's atom range alongside the coordinates (see loadConformer()).
  const ConformerRange range      = loadConformerRange(coordinates, conformerIdx);
  const int            totalAtoms = inputs.moleculeAtomStarts[coordinates.nMols];
  const bool is3D     = !kComputePbf || inputs.conformerIs3D == nullptr || inputs.conformerIs3D[conformerIdx] != 0;
  const int  numAtoms = range.numAtoms;
  int        moleculeAtomStart = 0;
  int        moleculeAtoms     = -1;
  if (range.positions != nullptr) {
    moleculeAtomStart = inputs.moleculeAtomStarts[range.moleculeIdx];
    moleculeAtoms     = inputs.moleculeAtomStarts[range.moleculeIdx + 1] - moleculeAtomStart;
  }
  double ownX = 0;
  double ownY = 0;
  double ownZ = 0;
  if (tid < numAtoms) {
    ownX = range.positions[tid * 3 + 0];
    ownY = range.positions[tid * 3 + 1];
    ownZ = range.positions[tid * 3 + 2];
  }
  Real* const row = whimOutput + static_cast<size_t>(conformerIdx) * kNumWhimProperties;
  if (moleculeAtoms != numAtoms || numAtoms > whimMaxAtoms) {  // Invalid; uniform across the block.
    for (int valueIdx = tid; valueIdx < kNumWhimProperties; valueIdx += kWhimThreads) {
      storeOutput<Real>(row + valueIdx, nan(""));
    }
    if (kComputePbf && tid == 0) {
      storeOutput<Real>(pbfOutput + conformerIdx, nan(""));
    }
    return;
  }

  double* scratch;
  if constexpr (kGlobalScratch) {
    scratch = reinterpret_cast<double*>(globalScratch + static_cast<size_t>(blockIdx.x) * layout.bytes());
  } else {
    scratch = dynamicScratch;
  }
  double* const  centered    = scratch;
  double* const  weights     = scratch + 3 * stride;
  int32_t* const scores      = reinterpret_cast<int32_t*>(scratch + layout.doubles());
  const double*  atomWeights = inputs.atomPropertyWeights + moleculeAtomStart;

  // Unweighted centroid in float64, shared by every PCA slot; the weights load meanwhile.
  double ownWeights[kNumAtomPropertyChannels];
  if (tid < numAtoms) {
    for (int channel = 0; channel < kNumAtomPropertyChannels; ++channel) {
      ownWeights[channel] = atomWeights[static_cast<size_t>(channel) * totalAtoms + tid];
    }
  }
  {
    double sumX = ownX;
    double sumY = ownY;
    double sumZ = ownZ;
    for (int atomIdx = tid + kWhimThreads; atomIdx < numAtoms; atomIdx += kWhimThreads) {
      sumX += range.positions[atomIdx * 3 + 0];
      sumY += range.positions[atomIdx * 3 + 1];
      sumZ += range.positions[atomIdx * 3 + 2];
    }
    const double partials[4] = {sumX, sumY, sumZ, 0.0};
    const double total       = warpTransposedSum4(partials, lane);
    if ((lane & 7) == 0 && (lane >> 3) < 3) {
      shared.centroidPartials[warp][lane >> 3] = total;
    }
  }
  __syncthreads();
  {
    double sumX = 0;
    double sumY = 0;
    double sumZ = 0;
    for (int w = 0; w < kWarps; ++w) {
      sumX += shared.centroidPartials[w][0];
      sumY += shared.centroidPartials[w][1];
      sumZ += shared.centroidPartials[w][2];
    }
    const double inverseAtoms = 1.0 / static_cast<double>(numAtoms);
    const double centroidX    = sumX * inverseAtoms;
    const double centroidY    = sumY * inverseAtoms;
    const double centroidZ    = sumZ * inverseAtoms;
    if (tid < numAtoms) {
      centered[tid]              = ownX - centroidX;
      centered[stride + tid]     = ownY - centroidY;
      centered[2 * stride + tid] = ownZ - centroidZ;
      for (int channel = 0; channel < kNumAtomPropertyChannels; ++channel) {
        weights[channel * stride + tid] = ownWeights[channel];
      }
    }
    for (int atomIdx = tid + kWhimThreads; atomIdx < numAtoms; atomIdx += kWhimThreads) {
      centeredPosition(range.positions,
                       atomIdx,
                       centroidX,
                       centroidY,
                       centroidZ,
                       centered[atomIdx],
                       centered[stride + atomIdx],
                       centered[2 * stride + atomIdx]);
      for (int channel = 0; channel < kNumAtomPropertyChannels; ++channel) {
        weights[channel * stride + atomIdx] = atomWeights[static_cast<size_t>(channel) * totalAtoms + atomIdx];
      }
    }
  }
  __syncthreads();

  // Covariance sums, one warp per slot.
  for (int slot = warp; slot < kPcaSlots; slot += kWarps) {
    const int sumIdx = lane >> 2;
    if (slot < kNumWhimChannels) {
      const WhimReal sum = accumulateCovarianceSums<WhimReal>(numAtoms,
                                                              lane,
                                                              slot == 0 ? nullptr : weights + (slot - 1) * stride,
                                                              centered,
                                                              stride);
      if ((lane & 3) == 0 && sumIdx < 7) {
        shared.covarianceSums[slot][sumIdx] = sum;
      }
    } else {
      const Real sum = accumulateCovarianceSums<Real>(numAtoms, lane, nullptr, centered, stride);
      if ((lane & 3) == 0 && sumIdx < 7) {
        shared.covarianceSums[slot][sumIdx] = sum;
      }
    }
  }
  __syncthreads();

  // Eigensystems: lane c of the first warp diagonalizes WHIM channel c, and the second warp's first lane PBF's
  // own PCA, apart from the first warp's lanes so that the two precisions do not run one after the other.
  // Meanwhile the third warp takes the logarithm the symmetry terms share.
  if (warp == 0 && lane < kNumWhimChannels) {
    WhimReal sums[7];
    for (int i = 0; i < 7; ++i) {
      sums[i] = shared.covarianceSums[lane][i];
    }
    WhimReal covariance[6];
    normalizeCovariance(sums, lane == 0, numAtoms, covariance);
    ProjectionState<WhimReal> state;
    diagonalizeProjection(covariance, state);
    for (int i = 0; i < 3; ++i) {
      shared.eigenvalues[lane][i] = state.eigenvalues[i];
    }
    for (int i = 0; i < 9; ++i) {
      shared.eigenvectors[lane][i] = state.eigenvectors[i];
    }
  } else if (kSeparatePbf && warp == 1 && lane == 0) {
    Real sums[7];
    for (int i = 0; i < 7; ++i) {
      sums[i] = static_cast<Real>(shared.covarianceSums[kPbfSlot][i]);
    }
    Real covariance[6];
    normalizeCovariance(sums, true, numAtoms, covariance);
    ProjectionState<Real> state;
    diagonalizeProjection(covariance, state);
    for (int i = 0; i < 9; ++i) {
      shared.eigenvectors[kPbfSlot][i] = state.eigenvectors[i];
    }
  } else if (warp == 2 && lane == 0) {
    shared.logInverseAtoms = log(WhimReal(1) / static_cast<WhimReal>(numAtoms));
  }
  __syncthreads();

  // Projections: one warp per channel rounds the scores and sums their fourth powers; PBF takes the last slot.
  constexpr int kProjectionSlots = kNumWhimChannels + (kComputePbf ? 1 : 0);
  for (int slot = warp; slot < kProjectionSlots; slot += kWarps) {
    if (slot < kNumWhimChannels) {
      WhimReal axes[9];
      for (int i = 0; i < 9; ++i) {
        axes[i] = shared.eigenvectors[slot][i];
      }
      int32_t* const channelScores    = scores + 3 * slot * stride;
      WhimReal       fourthMoments[4] = {0, 0, 0, 0};
      for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
        const WhimReal x = centered[atomIdx];
        const WhimReal y = centered[stride + atomIdx];
        const WhimReal z = centered[2 * stride + atomIdx];
        for (int axis = 0; axis < 3; ++axis) {
          const WhimReal score = x * axes[axis] + y * axes[3 + axis] + z * axes[6 + axis];
          fourthMoments[axis] += score * score * score * score;
          const int32_t rounded                  = static_cast<int32_t>(round(score * WhimReal(1000)));
          channelScores[axis * stride + atomIdx] = rounded;
        }
      }
      const WhimReal moment = warpTransposedSum4(fourthMoments, lane);
      if ((lane & 7) == 0 && (lane >> 3) < 3) {
        shared.fourthMoments[slot][lane >> 3] = moment;
      }
    } else {
      // Mean absolute distance from the plane of best fit, as computePbf().
      const Real axisX       = static_cast<Real>(shared.eigenvectors[kPbfSlot][2]);
      const Real axisY       = static_cast<Real>(shared.eigenvectors[kPbfSlot][5]);
      const Real axisZ       = static_cast<Real>(shared.eigenvectors[kPbfSlot][8]);
      Real       distanceSum = 0;
      for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
        const Real x = static_cast<Real>(centered[atomIdx]);
        const Real y = static_cast<Real>(centered[stride + atomIdx]);
        const Real z = static_cast<Real>(centered[2 * stride + atomIdx]);
        distanceSum += fabs(x * axisX + y * axisY + z * axisZ);
      }
      const Real pbf = groupAllReduceSum<kWarpSize>(distanceSum) / static_cast<Real>(numAtoms);
      if (lane == 0) {
        storeOutput<Real>(pbfOutput + conformerIdx, numAtoms < 4 || !is3D ? Real(0) : pbf);
      }
    }
  }
  __syncthreads();

  // Symmetry search, one (channel, axis) pair per warp at a time.
  for (int item = warp; item < kNumWhimAxes; item += kWarps) {
    const int total = countWhimSymmetric(scores + item * stride, numAtoms, lane, threshold);
    if (lane == 0) {
      shared.symmetricCounts[item] = total;
    }
  }
  // The last warp, which has no more pairs than the first, writes the eigenvalue-only channel terms meanwhile.
  if (warp == kWarps - 1 && lane < kNumWhimChannels) {
    writeWhimEigenvalueTerms<Real>(lane, shared.eigenvalues[lane], row);
  }
  __syncthreads();
  if (warp != 0) {
    return;
  }
  // The first warp writes the terms of every (channel, axis) pair, a lane each, then each channel's means.
  static_assert(kNumWhimAxes <= kWarpSize, "a lane per (channel, axis) pair");
  if (lane < kNumWhimAxes) {
    const int channel = lane / 3;
    const int axis    = lane % 3;
    WhimReal  gamma;
    WhimReal  kurtosis;
    writeWhimAxisTerms<Real>(lane,
                             shared.eigenvalues[channel][axis],
                             shared.fourthMoments[channel][axis],
                             shared.symmetricCounts[lane],
                             numAtoms,
                             shared.logInverseAtoms,
                             row,
                             gamma,
                             kurtosis);
    shared.axisTerms.gammas[channel][axis]   = gamma;
    shared.axisTerms.kurtoses[channel][axis] = kurtosis;
  }
  __syncwarp();
  if (lane < kNumWhimChannels) {
    writeWhimMeans<Real>(lane, shared.axisTerms.gammas[lane], shared.axisTerms.kurtoses[lane], row);
  }
}

/**
 * @brief PBF of conformers [0, numConformers); one group of lanes per conformer, an unweighted PCA in @p Real.
 */
template <typename Real>
__global__ void pbfGroupKernel(const DeviceCoordView        coordinates,
                               const Property3DDeviceInputs inputs,
                               Real* __restrict__ pbfOutput) {
  const int lane         = static_cast<int>(threadIdx.x) % kWarpSize;
  const int laneInGroup  = lane % kGroupSize;
  const int warpStart    = (blockIdx.x * kWarpsPerBlock + static_cast<int>(threadIdx.x) / kWarpSize) * kGroupsPerWarp;
  const int conformerIdx = warpStart + lane / kGroupSize;
  if (warpStart >= coordinates.numConformers) {
    return;  // Uniform across the warp; partially filled warps keep every lane for the shuffles.
  }
  const bool           inRange = conformerIdx < coordinates.numConformers;
  const ConformerAtoms atoms =
    loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, inRange ? conformerIdx : coordinates.numConformers);
  ProjectionState<Real> state;
  computeProjectionCentroid(atoms, laneInGroup, state);
  computeProjectionPca(atoms, nullptr, laneInGroup, state);
  const Real pbf = computePbf(atoms, laneInGroup, state);
  if (inRange && laneInGroup == 0) {
    const bool is3D         = inputs.conformerIs3D == nullptr || inputs.conformerIs3D[conformerIdx] != 0;
    pbfOutput[conformerIdx] = !atoms.valid ? static_cast<Real>(nan("")) : atoms.numAtoms < 4 || !is3D ? Real(0) : pbf;
  }
}

//! Upper bound on the WHIM kernel's global scratch, used when a row's scratch exceeds shared memory.
constexpr size_t kWhimScratchBudgetBytes = size_t{256} << 20;
//! Dynamic shared memory every device grants a block without opting in.
constexpr size_t kDefaultSharedBytes     = size_t{48} << 10;

/**
 * @brief Whether a shared-scratch WHIM kernel fits @p scratchBytes of dynamic shared memory beside its
 *        @p staticBytes on @p device; opts @p kernel in beyond the default allowance there when that is needed.
 */
template <typename Kernel>
bool configureWhimSharedScratch(Kernel* const kernel,
                                const size_t  scratchBytes,
                                const size_t  staticBytes,
                                const int     device) {
  if (scratchBytes + staticBytes <= kDefaultSharedBytes) {
    return true;
  }
  int optInSize = 0;
  cudaCheckError(cudaDeviceGetAttribute(&optInSize, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  if (scratchBytes + staticBytes > static_cast<size_t>(optInSize)) {
    return false;
  }
  cudaCheckError(
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(scratchBytes)));
  return true;
}

template <typename Real, bool kComputePbf>
void launchWhim(const DeviceCoordView&        coordinates,
                const Property3DDeviceInputs& inputs,
                const WhimThreshold&          threshold,
                Real*                         pbfOutput,
                Real*                         whimOutput,
                const cudaStream_t            stream) {
  const int               numConformers = coordinates.numConformers;
  const int               maxAtoms      = std::max(inputs.maxMoleculeAtoms, 1);
  const WhimScratchLayout layout        = WhimScratchLayout::make(maxAtoms);
  int                     device        = 0;
  cudaCheckError(cudaGetDevice(&device));

  const size_t scratchBytes = layout.bytes();
  auto* const  sharedKernel = whimKernel<kWhimWarps, Real, kComputePbf, false>;
  const bool   sharedFits =
    configureWhimSharedScratch(sharedKernel, scratchBytes, sizeof(WhimBlockShared<kWhimWarps>), device);
  if (sharedFits) {
    sharedKernel<<<numConformers, kWhimWarps * kWarpSize, scratchBytes, stream>>>(coordinates,
                                                                                  inputs,
                                                                                  0,
                                                                                  pbfOutput,
                                                                                  whimOutput,
                                                                                  nullptr,
                                                                                  maxAtoms,
                                                                                  layout,
                                                                                  threshold);
    cudaCheckError(cudaGetLastError());
    return;
  }
  // Rows too long for shared memory use global scratch, in conformer chunks within the budget.
  const size_t chunkSize =
    std::clamp<size_t>(kWhimScratchBudgetBytes / scratchBytes, 1, static_cast<size_t>(numConformers));
  AsyncDeviceVector<char> scratch(chunkSize * scratchBytes, stream);
  for (int begin = 0; begin < numConformers; begin += static_cast<int>(chunkSize)) {
    const int end = std::min(numConformers, begin + static_cast<int>(chunkSize));
    whimKernel<kWhimWarps, Real, kComputePbf, true><<<end - begin, kWhimWarps * kWarpSize, 0, stream>>>(coordinates,
                                                                                                        inputs,
                                                                                                        begin,
                                                                                                        pbfOutput,
                                                                                                        whimOutput,
                                                                                                        scratch.data(),
                                                                                                        maxAtoms,
                                                                                                        layout,
                                                                                                        threshold);
    cudaCheckError(cudaGetLastError());
  }
}

/**
 * @brief Launches the projection kernels when PBF or WHIM is requested. PBF computes in @p Real; WHIM's
 *        PCA computes in WhimReal. Both store @p Real values.
 */
template <typename Real>
void launchProjectionProperties(const DeviceCoordView&        coordinates,
                                const Property3DDeviceInputs& inputs,
                                const WhimOptions&            whim,
                                Real*                         pbfOutput,
                                Real*                         whimOutput,
                                const cudaStream_t            stream) {
  const int numConformers = coordinates.numConformers;
  if (numConformers == 0 || (pbfOutput == nullptr && whimOutput == nullptr)) {
    return;
  }
  if (whimOutput == nullptr) {
    const int numBlocks = (numConformers + kConformersPerBlock - 1) / kConformersPerBlock;
    pbfGroupKernel<Real><<<numBlocks, kBlockSize, 0, stream>>>(coordinates, inputs, pbfOutput);
    cudaCheckError(cudaGetLastError());
    return;
  }
  const WhimThreshold threshold = WhimThreshold::make(whim.threshold);
  if (pbfOutput != nullptr) {
    launchWhim<Real, true>(coordinates, inputs, threshold, pbfOutput, whimOutput, stream);
  } else {
    launchWhim<Real, false>(coordinates, inputs, threshold, pbfOutput, whimOutput, stream);
  }
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_PROJECTION_CUH
