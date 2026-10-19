// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_PROJECTION_CUH
#define NVMOLKIT_DESCRIPTORS3D_PROJECTION_CUH

#include <algorithm>
#include <cmath>
#include <type_traits>

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
__host__ __device__ __forceinline__ double thousandthsToUnits(const int32_t a) {
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
//! Widest bitmap search window radius: the 2 r + 1 window bits fit one word.
constexpr int kWhimMaxWindowRadius = 15;

struct WhimThreshold {
  double   value;             //!< Threshold in the projection's units.
  int64_t  mirroredLimit;     //!< Largest off-boundary |a + b| (thousandths) that counts as mirrored.
  int64_t  centeredLimit;     //!< Largest off-boundary |a| (thousandths) that counts as centered.
  int64_t  boundary;          //!< Magnitude decided by RDKit's float64 expression, or -1 if none.
  uint32_t candidateLimit;    //!< mayMirror() bound: every |a + b| that mirrored() can accept.
  bool     exhaustive;        //!< candidateLimit would overflow the filter, so every pair is a candidate.
  int      bucketShift;       //!< Smallest s with 2^s > 2 * candidateLimit: buckets `a >> s` of the symmetry search.
  int      windowRadius;      //!< Bitmap search window: mirror images +- windowRadius, the boundary included.
  uint32_t definiteMask;      //!< Window bits (bit k is mirror - windowRadius + k) mirrored() accepts outright.
  bool     windowFits;        //!< The bitmap search applies: not exhaustive, windowRadius <= kWhimMaxWindowRadius.
  bool     boundaryCentered;  //!< centered() of a magnitude on the boundary, which depends on the magnitude alone.

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
    const int64_t     candidateLimit     = candidate < kMaxCandidateLimit ? candidate : kMaxCandidateLimit;
    int               bucketShift        = 0;
    while ((int64_t{1} << bucketShift) <= 2 * candidateLimit) {
      ++bucketShift;
    }
    // mirrored() accepts |a + b| <= definite outright; on the grid that is boundary - 1, and the window adds
    // the two boundary values.
    const int64_t definite     = boundary == floorLimit ? floorLimit - 1 : floorLimit;
    const int64_t radius       = boundary >= 0 ? boundary : definite;
    const bool    windowFits   = candidate <= kMaxCandidateLimit && radius >= 0 && radius <= kWhimMaxWindowRadius;
    uint32_t      definiteMask = 0;
    if (windowFits) {
      definiteMask =
        boundary >= 0 ? (radius > 0 ? ((1u << (2 * radius - 1)) - 1u) << 1 : 0u) : (1u << (2 * radius + 1)) - 1u;
    }
    return WhimThreshold{value,
                         floorLimit,
                         onGrid ? static_cast<int64_t>(nearest) - 1 : floorLimit,
                         boundary,
                         static_cast<uint32_t>(candidateLimit),
                         candidate > kMaxCandidateLimit,
                         bucketShift,
                         static_cast<int>(windowFits ? radius : 0),
                         definiteMask,
                         windowFits,
                         boundary >= 0 && boundary < (int64_t{1} << 31) &&
                           fabs(thousandthsToUnits(static_cast<int32_t>(boundary))) < value};
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

//! Maximum over the warp, returned to every lane.
__device__ __forceinline__ uint32_t warpMax(uint32_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __reduce_max_sync(0xffffffffu, value);
#else
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    const uint32_t other = __shfl_xor_sync(0xffffffffu, value, offset);
    value                = other > value ? other : value;
  }
  return value;
#endif
}

//! Magnitude of a rounded projection, exact for every int32.
__device__ __forceinline__ uint32_t whimMagnitude(const int32_t value) {
  return value < 0 ? 0u - static_cast<uint32_t>(value) : static_cast<uint32_t>(value);
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

//! Hash slots of the symmetry search for @p numAtoms atoms: a power of two of at least twice the atoms.
__host__ __device__ inline int whimHashSlots(const int numAtoms) {
  int slots = kWarpSize;
  while (slots < 2 * numAtoms) {
    slots *= 2;
  }
  return slots;
}

/**
 * @brief Warp-collective. Per lane, the number of its atoms (lane, lane + 32, ...) whose rounded projection
 *        @p values (one axis of one channel) is centered, or mirrored by another atom within the threshold.
 *
 * Matches RDKit's pairwise comparison without visiting every pair: atoms are hashed by bucket
 * `value >> bucketShift` into @p head (@p hashSlots chains threaded through @p next), and each atom walks only
 * the (at most two) buckets that can hold a value within candidateLimit of its mirror image, where the exact
 * comparison decides. Exhaustive thresholds compare every pair.
 */
__device__ __forceinline__ int countWhimSymmetric(const int32_t*       values,
                                                  const int            numAtoms,
                                                  const int            lane,
                                                  const WhimThreshold& threshold,
                                                  const int            hashSlots,
                                                  int32_t*             head,
                                                  int32_t*             next) {
  int symmetric = 0;
  if (threshold.exhaustive) {
    for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
      const int32_t own   = values[atomIdx];
      bool          found = threshold.centered(own);
      for (int otherIdx = 0; otherIdx < numAtoms && !found; ++otherIdx) {
        found = otherIdx != atomIdx && threshold.mirrored(own, values[otherIdx]);
      }
      symmetric += found ? 1 : 0;
    }
    return symmetric;
  }
  const int     mask  = hashSlots - 1;
  const int     shift = threshold.bucketShift;
  const int64_t limit = threshold.candidateLimit;
  for (int slot = lane; slot < hashSlots; slot += kWarpSize) {
    head[slot] = -1;
  }
  __syncwarp();
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    const int bucket = static_cast<int>((static_cast<int64_t>(values[atomIdx]) >> shift) & mask);
    next[atomIdx]    = atomicExch(&head[bucket], atomIdx);
  }
  __syncwarp();
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    const int32_t own   = values[atomIdx];
    bool          found = threshold.centered(own);
    if (!found) {
      const int64_t mirror = -static_cast<int64_t>(own);
      const int64_t last   = (mirror + limit) >> shift;
      for (int64_t bucket = (mirror - limit) >> shift; bucket <= last && !found; ++bucket) {
        for (int otherIdx = head[bucket & mask]; otherIdx >= 0 && !found; otherIdx = next[otherIdx]) {
          const int32_t other = values[otherIdx];
          const int64_t sum   = static_cast<int64_t>(own) + other;
          found               = otherIdx != atomIdx && sum >= -limit && sum <= limit && threshold.mirrored(own, other);
        }
      }
    }
    symmetric += found ? 1 : 0;
  }
  __syncwarp();  // Every lane is done with the chains before the next search resets them.
  return symmetric;
}

//! Words of each warp's value bitmap in WHIM's symmetry search: a rounded projection v with
//! |v| < kWhimBitmapRange owns bit v + kWhimBitmapOffset, between kWhimBitmapPadWords zero words on each side
//! so that every search window lies inside the bitmap.
constexpr int kWhimBitmapPadWords = 2;
constexpr int kWhimBitmapRange    = 8192;
constexpr int kWhimBitmapWords    = 2 * kWhimBitmapRange / kWarpSize + 2 * kWhimBitmapPadWords;
constexpr int kWhimBitmapOffset   = kWhimBitmapRange + kWhimBitmapPadWords * kWarpSize;
static_assert(kWhimMaxWindowRadius < kWhimBitmapPadWords * kWarpSize, "windows stay inside the bitmap");

/**
 * @brief The case of whimBitmapSymmetric() too rare for the inline path, out of line so it stays short: an atom
 *        whose own bit lies in its window, compared with every other atom as countWhimSymmetric() does for
 *        exhaustive thresholds. Only a zero threshold reaches it with an atom not already centered.
 */
__device__ __noinline__ bool whimSymmetricRare(const int32_t  own,
                                               const int      radius,
                                               const int      boundary,
                                               const int64_t  mirroredLimit,
                                               const double   value,
                                               const int32_t* values,
                                               const int      numAtoms,
                                               const int      atomIdx) {
  const int magnitude = own < 0 ? -own : own;
  if (magnitude == boundary && WhimThreshold::centeredOnBoundary(own, value)) {
    return true;
  }
  if (2 * magnitude > radius) {
    return false;  // The window bits decide; see whimBitmapSymmetric().
  }
  for (int otherIdx = 0; otherIdx < numAtoms; ++otherIdx) {
    const int32_t other = values[otherIdx];
    const int64_t sum   = llabs(static_cast<int64_t>(own) + static_cast<int64_t>(other));
    if (otherIdx != atomIdx &&
        (sum == boundary ? WhimThreshold::mirroredOnBoundary(own, other, value) : sum <= mirroredLimit)) {
      return true;
    }
  }
  return false;
}

//! WhimThreshold's bitmap-search constants, held in registers through the symmetry searches.
struct WhimBitmapParams {
  int      windowBase;  //!< Bit of the window's lowest value for own value 0: mirror images sit at windowBase - own.
  int      radius;
  int      boundary;
  int      centeredLimit;
  uint32_t definiteMask;
  uint32_t boundaryMask;
  int64_t  mirroredLimit;
  double   value;
  //! Every value whose own bit lies in its window (|v| <= radius / 2) is centered, so that case never decides.
  bool     selfCentered;

  __device__ static WhimBitmapParams make(const WhimThreshold& threshold) {
    const int boundary = static_cast<int>(threshold.boundary);
    const int radius   = threshold.windowRadius;
    const int centeredLimit =
      static_cast<int>(threshold.boundaryCentered ? threshold.boundary : threshold.centeredLimit);
    return WhimBitmapParams{kWhimBitmapOffset - radius,
                            radius,
                            boundary,
                            centeredLimit,
                            threshold.definiteMask,
                            boundary >= 0 ? (1u | (1u << (2 * radius))) : 0u,
                            threshold.mirroredLimit,
                            threshold.value,
                            centeredLimit >= radius / 2};
  }
};

/**
 * @brief Whether rounded projection @p own of atom @p atomIdx is centered, or mirrored by another atom, given
 *        @p bitmap holding every atom's value (see countWhimSymmetricBitmap()). False for an inactive lane.
 *
 * One funnel shift reads the window around the mirror image -own: a definite bit means mirrored, and a
 * boundary bit takes RDKit's float64 mirrored test inline (symmetric pairs land on the boundary often). RDKit's
 * centered test on the boundary magnitude is one value per threshold, folded into centeredLimit (see
 * WhimThreshold::boundaryCentered), and an own bit inside the window goes to whimSymmetricRare(). With
 * @p kSelfCentered (WhimBitmapParams::selfCentered) such a value is centered, so that case is compiled out.
 */
template <bool kSelfCentered>
__device__ __forceinline__ bool whimBitmapSymmetric(const int32_t           own,
                                                    const bool              active,
                                                    const int               atomIdx,
                                                    const WhimBitmapParams& params,
                                                    const uint32_t*         bitmap,
                                                    const int32_t*          values,
                                                    const int               numAtoms) {
  const int       magnitude    = own < 0 ? -own : own;
  const int       start        = params.windowBase - own;
  const uint32_t* words        = bitmap + (start >> 5);
  const uint32_t  window       = __funnelshift_r(words[0], words[1], start & 31);
  const bool      selfInWindow = !kSelfCentered && 2 * magnitude <= params.radius;
  bool            found = magnitude <= params.centeredLimit || (!selfInWindow && (window & params.definiteMask) != 0);
  if (active && !found) {
    if (!selfInWindow && (window & params.boundaryMask) != 0) {
      const double ownUnits = thousandthsToUnits(own);
      found = ((window & 1u) != 0 && fabs(ownUnits + thousandthsToUnits(-own - params.boundary)) <= params.value) ||
              (((window >> (2 * params.radius)) & 1u) != 0 &&
               fabs(ownUnits + thousandthsToUnits(-own + params.boundary)) <= params.value);
    }
    if (!found && selfInWindow) {
      found = whimSymmetricRare(own,
                                params.radius,
                                params.boundary,
                                params.mirroredLimit,
                                params.value,
                                values,
                                numAtoms,
                                atomIdx);
    }
  }
  return active && found;
}

/**
 * @brief Warp-collective countWhimSymmetric() for rounded projections of magnitude below kWhimBitmapRange and
 *        thresholds whose window fits (WhimThreshold::windowFits), for at most @p kChunks * 32 atoms, which the
 *        lanes keep in registers. Returns the warp total.
 *
 * Every value sets its bit in @p bitmap (all zero on entry and on return), then each atom reads the window
 * around its mirror image (whimBitmapSymmetric()). Lanes past the last atom set and clear bit 0, in padding
 * that no window reaches, so every lane runs the same instructions.
 */
template <int kChunks, bool kSelfCentered>
__device__ __forceinline__ int countWhimSymmetricBitmap(const int32_t*          values,
                                                        const int               numAtoms,
                                                        const int               lane,
                                                        const WhimBitmapParams& params,
                                                        uint32_t*               bitmap) {
  static_assert(kWhimBitmapOffset - kWhimBitmapRange - kWhimMaxWindowRadius >= kWarpSize, "bit 0 is unread");
  int32_t own[kChunks];
  int     bit[kChunks];
#pragma unroll
  for (int k = 0; k < kChunks; ++k) {
    const bool active = lane + k * kWarpSize < numAtoms;
    own[k]            = active ? values[lane + k * kWarpSize] : 0;
    bit[k]            = active ? own[k] + kWhimBitmapOffset : 0;
    atomicOr(&bitmap[bit[k] >> 5], 1u << (bit[k] & 31));
  }
  __syncwarp();
  int symmetric = 0;
#pragma unroll
  for (int k = 0; k < kChunks; ++k) {
    const int  atomIdx = lane + k * kWarpSize;
    const bool found =
      whimBitmapSymmetric<kSelfCentered>(own[k], atomIdx < numAtoms, atomIdx, params, bitmap, values, numAtoms);
    symmetric += __popc(__ballot_sync(0xffffffffu, found));
  }
  __syncwarp();
#pragma unroll
  for (int k = 0; k < kChunks; ++k) {
    bitmap[bit[k] >> 5] = 0;
  }
  __syncwarp();
  return symmetric;
}

//! countWhimSymmetricBitmap() for any number of atoms, rereading the values.
template <bool kSelfCentered>
__device__ __forceinline__ int countWhimSymmetricBitmapLoop(const int32_t*          values,
                                                            const int               numAtoms,
                                                            const int               lane,
                                                            const WhimBitmapParams& params,
                                                            uint32_t*               bitmap) {
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    const int bit = values[atomIdx] + kWhimBitmapOffset;
    atomicOr(&bitmap[bit >> 5], 1u << (bit & 31));
  }
  __syncwarp();
  int symmetric = 0;
  for (int base = 0; base < numAtoms; base += kWarpSize) {
    const int     atomIdx = base + lane;
    const bool    active  = atomIdx < numAtoms;
    const int32_t own     = active ? values[atomIdx] : 0;
    symmetric +=
      __popc(__ballot_sync(0xffffffffu,
                           whimBitmapSymmetric<kSelfCentered>(own, active, atomIdx, params, bitmap, values, numAtoms)));
  }
  __syncwarp();
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    bitmap[(values[atomIdx] + kWhimBitmapOffset) >> 5] = 0;
  }
  __syncwarp();
  return symmetric;
}

/**
 * @brief Warp-collective. Symmetric atom counts of the (channel, axis) pairs `warp`, `warp + kWarps`, ...: the
 *        warp's k-th pair's count is returned to lane k. @p kChunks selects countWhimSymmetricBitmap()'s
 *        variant (0: the loop) for pairs whose bit of @p bitmapItems is set; the others take the hash chains,
 *        whose region is zeroed again after.
 */
template <int kChunks, int kWarps, bool kSelfCentered>
__device__ __forceinline__ int searchWarpWhimAxes(const int32_t*          scores,
                                                  const int               stride,
                                                  const int               numAtoms,
                                                  const int               warp,
                                                  const int               lane,
                                                  const uint32_t          bitmapItems,
                                                  const WhimBitmapParams& params,
                                                  const WhimThreshold&    threshold,
                                                  int32_t*                search) {
  int ownCount = 0;
  for (int item = warp, k = 0; item < kNumWhimAxes; item += kWarps, ++k) {
    const int32_t* const values = scores + item * stride;
    int                  total;
    if (((bitmapItems >> item) & 1u) != 0) {
      if constexpr (kChunks > 0) {
        total = countWhimSymmetricBitmap<kChunks, kSelfCentered>(values,
                                                                 numAtoms,
                                                                 lane,
                                                                 params,
                                                                 reinterpret_cast<uint32_t*>(search));
      } else {
        total = countWhimSymmetricBitmapLoop<kSelfCentered>(values,
                                                            numAtoms,
                                                            lane,
                                                            params,
                                                            reinterpret_cast<uint32_t*>(search));
      }
    } else {
      const int hashSlots = whimHashSlots(numAtoms);
      total               = groupAllReduceSum<kWarpSize>(
        countWhimSymmetric(values, numAtoms, lane, threshold, hashSlots, search, search + hashSlots));
      for (int i = lane; i < hashSlots + numAtoms; i += kWarpSize) {
        search[i] = 0;
      }
      __syncwarp();
    }
    ownCount = lane == k ? total : ownCount;
  }
  return ownCount;
}

//! searchWarpWhimAxes() with the register-resident bitmap variant that fits @p numAtoms, uniform across the block.
template <int kWarps, bool kSelfCentered>
__device__ __forceinline__ int searchWhimAxes(const int32_t*          scores,
                                              const int               stride,
                                              const int               numAtoms,
                                              const int               warp,
                                              const int               lane,
                                              const uint32_t          bitmapItems,
                                              const WhimBitmapParams& params,
                                              const WhimThreshold&    threshold,
                                              int32_t*                search) {
  if (numAtoms <= kWarpSize) {
    return searchWarpWhimAxes<1, kWarps, kSelfCentered>(scores,
                                                        stride,
                                                        numAtoms,
                                                        warp,
                                                        lane,
                                                        bitmapItems,
                                                        params,
                                                        threshold,
                                                        search);
  }
  if (numAtoms <= 2 * kWarpSize) {
    return searchWarpWhimAxes<2, kWarps, kSelfCentered>(scores,
                                                        stride,
                                                        numAtoms,
                                                        warp,
                                                        lane,
                                                        bitmapItems,
                                                        params,
                                                        threshold,
                                                        search);
  }
  if (numAtoms <= 4 * kWarpSize) {
    return searchWarpWhimAxes<4, kWarps, kSelfCentered>(scores,
                                                        stride,
                                                        numAtoms,
                                                        warp,
                                                        lane,
                                                        bitmapItems,
                                                        params,
                                                        threshold,
                                                        search);
  }
  return searchWarpWhimAxes<0, kWarps, kSelfCentered>(scores,
                                                      stride,
                                                      numAtoms,
                                                      warp,
                                                      lane,
                                                      bitmapItems,
                                                      params,
                                                      threshold,
                                                      search);
}

//! Warps per block of the WHIM kernel, which computes one conformer per block, for grids of several waves.
constexpr int                     kWhimWarps     = 4;
//! Blocks per SM the register budget of that kernel allows (shared memory permitting).
constexpr int                     kWhimMinBlocks = 8;
//! Warps per block of the WHIM kernel for a grid that fits one resident wave: a warp per projection slot (WHIM's
//! channels, then PBF's), so those phases run concurrently; four blocks per SM keep its register budget.
__host__ __device__ constexpr int whimWaveWarps(const bool computePbf) {
  return kNumWhimChannels + (computePbf ? 1 : 0);
}
constexpr int kWhimWaveMinBlocks = 4;
//! PCA slots of the WHIM kernel: WHIM's channels, then PBF's own PCA when it computes in another precision.
constexpr int kWhimPcaSlots      = kNumWhimChannels + 1;
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
  double   eigenvalues[kWhimPcaSlots][3];
  double   eigenvectors[kWhimPcaSlots][9];
  double   fourthMoments[kNumWhimChannels][3];
  uint32_t maxMagnitudes[kNumWhimChannels];  //!< Largest rounded projection magnitude of each channel.
  int      symmetricCounts[kNumWhimAxes];    //!< Symmetric atoms of each (channel, axis) pair.
};

/**
 * @brief Per-conformer scratch layout of the WHIM kernel for rows of up to `stride` atoms, in elements from
 *        its start: float64 centered coordinates (x, y, z arrays) and atom-property weights (one array per
 *        channel), then int32 rounded projections (one array per channel and axis) and each warp's symmetry
 *        search region of `searchInts`: the value bitmap (kWhimBitmapWords), or the hash heads
 *        (whimHashSlots(stride)) followed by the chain links (stride).
 */
struct WhimScratchLayout {
  int stride;
  int searchInts;

  __host__ __device__ static WhimScratchLayout make(const int maxAtoms) {
    const int atoms  = maxAtoms > 1 ? maxAtoms : 1;
    const int stride = (atoms + 3) / 4 * 4;  // Keeps every array 16-byte aligned.
    const int hash   = whimHashSlots(stride) + stride;
    return WhimScratchLayout{stride, hash > kWhimBitmapWords ? hash : kWhimBitmapWords};
  }
  __host__ __device__ size_t doubles() const { return static_cast<size_t>(3 + kNumAtomPropertyChannels) * stride; }
  __host__ __device__ size_t ints(const int warps) const {
    return static_cast<size_t>(kNumWhimAxes) * stride + static_cast<size_t>(warps) * searchInts;
  }
  __host__ __device__ size_t bytes(const int warps) const {
    return doubles() * sizeof(double) + ints(warps) * sizeof(int32_t);
  }
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
 * PCA slots: WHIM's unweighted and atom-property channels in WhimReal and, when PBF is requested in another
 * precision, PBF's unweighted PCA in @p Real (in float64 PBF shares WHIM's identical unweighted channel).
 * One lane per slot diagonalizes. Warps then project and round each channel's coordinates, PBF is written,
 * and the warps share the symmetry searches of the (channel, axis) pairs. Finally the first warp writes the
 * WHIM row. The scratch (see WhimScratchLayout) is shared memory, or with @p kGlobalScratch a block per
 * conformer of @p globalScratch. Rows longer than @p whimMaxAtoms are treated as invalid, and invalid rows
 * produce NaN.
 */
template <int kWarps, typename Real, bool kComputePbf, bool kGlobalScratch>
__global__ void __launch_bounds__(kWarps* kWarpSize, kWarps == kWhimWarps ? kWhimMinBlocks : kWhimWaveMinBlocks)
  whimKernel(const DeviceCoordView        coordinates,
             const Property3DDeviceInputs inputs,
             const int                    conformerBegin,
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
  constexpr bool kSeparatePbf = kComputePbf && !std::is_same_v<Real, WhimReal>;
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
    scratch = reinterpret_cast<double*>(globalScratch + static_cast<size_t>(blockIdx.x) * layout.bytes(kWarps));
  } else {
    scratch = dynamicScratch;
  }
  double* const  centered    = scratch;
  double* const  weights     = scratch + 3 * stride;
  int32_t* const scores      = reinterpret_cast<int32_t*>(scratch + layout.doubles());
  int32_t* const search      = scores + kNumWhimAxes * stride + warp * layout.searchInts;
  const double*  atomWeights = inputs.atomPropertyWeights + moleculeAtomStart;
  // The symmetry search expects a zero region; searchInts is a multiple of 4 and the region 16-byte aligned.
  for (int i = lane; i < layout.searchInts / 4; i += kWarpSize) {
    reinterpret_cast<int4*>(search)[i] = make_int4(0, 0, 0, 0);
  }

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
      uint32_t       maxMagnitude     = 0;
      for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
        const WhimReal x = centered[atomIdx];
        const WhimReal y = centered[stride + atomIdx];
        const WhimReal z = centered[2 * stride + atomIdx];
        for (int axis = 0; axis < 3; ++axis) {
          const WhimReal score = x * axes[axis] + y * axes[3 + axis] + z * axes[6 + axis];
          fourthMoments[axis] += score * score * score * score;
          const int32_t rounded                  = static_cast<int32_t>(round(score * WhimReal(1000)));
          channelScores[axis * stride + atomIdx] = rounded;
          const uint32_t magnitude               = whimMagnitude(rounded);
          maxMagnitude                           = magnitude > maxMagnitude ? magnitude : maxMagnitude;
        }
      }
      const WhimReal moment = warpTransposedSum4(fourthMoments, lane);
      if ((lane & 7) == 0 && (lane >> 3) < 3) {
        shared.fourthMoments[slot][lane >> 3] = moment;
      }
      maxMagnitude = warpMax(maxMagnitude);
      if (lane == 0) {
        shared.maxMagnitudes[slot] = maxMagnitude;
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

  // Symmetry search, one (channel, axis) pair per warp at a time: the bitmap search when the channel's
  // projections and the threshold window fit it, else the hash chains. The atom count, uniform across the
  // block, picks the variant once.
  {
    const WhimBitmapParams bitmapParams = WhimBitmapParams::make(threshold);
    uint32_t               bitmapItems  = 0;
    if (threshold.windowFits) {
      for (int channel = 0; channel < kNumWhimChannels; ++channel) {
        bitmapItems |=
          shared.maxMagnitudes[channel] < static_cast<uint32_t>(kWhimBitmapRange) ? 7u << (3 * channel) : 0u;
      }
    }
    // The threshold, uniform across the grid, decides whether the self-in-window case can arise.
    const int ownCount =
      bitmapParams.selfCentered ?
        searchWhimAxes<kWarps,
                       true>(scores, stride, numAtoms, warp, lane, bitmapItems, bitmapParams, threshold, search) :
        searchWhimAxes<kWarps,
                       false>(scores, stride, numAtoms, warp, lane, bitmapItems, bitmapParams, threshold, search);
    if (warp + lane * kWarps < kNumWhimAxes) {
      shared.symmetricCounts[warp + lane * kWarps] = ownCount;
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

/*
 * Batched WHIM: one block of kBatchWarps warps computes kConformers conformers, so that the work no single
 * conformer fills a warp with is packed across conformers. Atom-parallel phases use groups of
 * kBatchGroupLanes lanes per (conformer, slot) task; the eigensolves and the per-axis output terms take a lane
 * each; the symmetry searches are spread over the warps one (conformer, channel, axis) item at a time.
 */

//! Warps per block of the batched WHIM kernel.
constexpr int kBatchWarps      = 4;
constexpr int kBatchThreads    = kBatchWarps * kWarpSize;
//! Lanes of a covariance or projection task, and tasks per warp.
constexpr int kBatchGroupLanes = 8;
constexpr int kBatchGroups     = kWarpSize / kBatchGroupLanes;
//! Covariance slots per conformer: WHIM's channels, then PBF's own PCA when it computes in another precision.
constexpr int kBatchSlots      = 8;
static_assert(kWhimPcaSlots <= kBatchSlots && kBatchSlots == 2 * kBatchGroups, "two covariance passes per conformer");
//! Atoms per conformer row of the batched kernel's int32 fallback search, which reuses a warp's bitmap.
constexpr int kBatchMaxAtoms = kWhimBitmapWords;

//! Per-block values of the batched WHIM kernel, per conformer of the batch.
template <int kConformers> struct WhimBatchShared {
  union {
    double covarianceSums[kConformers][kBatchSlots][8];  //!< Total weight, xx, xy, xz, yy, yz, zz per slot.
    struct {
      double gammas[kConformers][kNumWhimChannels][3];
      double kurtoses[kConformers][kNumWhimChannels][3];
    } axisTerms;  //!< After the eigensolves.
  };
  double   eigen[kConformers][kNumWhimChannels][12];  //!< Row-major eigenvectors, then the eigenvalues.
  double   pbfAxis[kConformers][3];                   //!< PBF's plane normal, when PBF has its own PCA.
  double   fourthMoments[kConformers][kNumWhimAxes];
  double   centroids[kConformers][3];
  int64_t  atomStart[kConformers];  //!< First coordinate row.
  double   logInverseAtoms[kConformers];
  uint32_t maxMagnitudes[kConformers][kNumWhimAxes];
  int      symmetricCounts[kConformers][kNumWhimAxes];
  int      numAtoms[kConformers];  //!< Zero for a conformer that is out of range or invalid.
  int      moleculeAtomStart[kConformers];
  int      is3D[kConformers];
};

/**
 * @brief Dynamic shared memory of the batched WHIM kernel for rows of up to `stride` atoms: each warp's value
 *        bitmap, then each warp's int16 rounded projections of its groups' current tasks (one array per group
 *        and principal axis).
 */
struct WhimBatchLayout {
  int stride;

  __host__ __device__ static WhimBatchLayout make(const int maxAtoms) {
    const int atoms = maxAtoms > 1 ? maxAtoms : 1;
    return WhimBatchLayout{(atoms + 7) / 8 * 8};  // Keeps every array 16-byte aligned.
  }
  __host__ __device__ size_t warpScoreInts16() const { return static_cast<size_t>(kBatchGroups) * 3 * stride; }
  __host__ __device__ size_t bytes() const {
    return static_cast<size_t>(kBatchWarps) *
           (kWhimBitmapWords * sizeof(uint32_t) + warpScoreInts16() * sizeof(int16_t));
  }
};

/**
 * @brief Collective over aligned groups of kBatchGroupLanes lanes. Sums each of a lane's 8 @p values over its
 *        group, halving the values each lane carries at every exchange; returns the total of value `lane & 7`.
 */
template <typename T> __device__ __forceinline__ T groupTransposedSum8(const T (&values)[8], const int lane) {
  T half[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const bool upper = (lane & 4) != 0;
    const T    send  = upper ? values[k] : values[k + 4];
    const T    keep  = upper ? values[k + 4] : values[k];
    half[k]          = keep + __shfl_xor_sync(0xffffffffu, send, 4);
  }
  T quarter[2];
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const bool upper = (lane & 2) != 0;
    const T    send  = upper ? half[k] : half[k + 2];
    const T    keep  = upper ? half[k + 2] : half[k];
    quarter[k]       = keep + __shfl_xor_sync(0xffffffffu, send, 2);
  }
  const bool upper = (lane & 1) != 0;
  return (upper ? quarter[1] : quarter[0]) + __shfl_xor_sync(0xffffffffu, upper ? quarter[0] : quarter[1], 1);
}

//! groupTransposedSum8() for 4 @p values: returns the group total of value `(lane >> 1) & 3`.
template <typename T> __device__ __forceinline__ T groupTransposedSum4(const T (&values)[4], const int lane) {
  T half[2];
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const bool upper = (lane & 4) != 0;
    const T    send  = upper ? values[k] : values[k + 2];
    const T    keep  = upper ? values[k + 2] : values[k];
    half[k]          = keep + __shfl_xor_sync(0xffffffffu, send, 4);
  }
  const bool upper = (lane & 2) != 0;
  T          value = (upper ? half[1] : half[0]) + __shfl_xor_sync(0xffffffffu, upper ? half[0] : half[1], 2);
  return value + __shfl_xor_sync(0xffffffffu, value, 1);
}

//! Group-collective sum over aligned groups of kBatchGroupLanes lanes, returned to every lane of the group.
template <typename T> __device__ __forceinline__ T groupButterflySum(T value) {
  value += __shfl_xor_sync(0xffffffffu, value, 4);
  value += __shfl_xor_sync(0xffffffffu, value, 2);
  value += __shfl_xor_sync(0xffffffffu, value, 1);
  return value;
}

//! WHIM's projection of centered atom (@p x, @p y, @p z) on the principal axis (@p axisX, @p axisY, @p axisZ).
__device__ __forceinline__ WhimReal whimScore(const WhimReal x,
                                              const WhimReal y,
                                              const WhimReal z,
                                              const WhimReal axisX,
                                              const WhimReal axisY,
                                              const WhimReal axisZ) {
  return x * axisX + y * axisY + z * axisZ;
}

__device__ __forceinline__ int32_t whimRounded(const WhimReal score) {
  return static_cast<int32_t>(round(score * WhimReal(1000)));
}

//! The batched kernel's window test of value @p own: whether an atom with it is symmetric.
__device__ __forceinline__ bool whimBatchSymmetric(const int32_t           own,
                                                   const bool              active,
                                                   const WhimBitmapParams& params,
                                                   const uint32_t*         bitmap) {
  const int       magnitude = own < 0 ? -own : own;
  const int       start     = params.windowBase - own;
  const uint32_t* words     = bitmap + (start >> 5);
  const uint32_t  window    = __funnelshift_r(words[0], words[1], start & 31);
  bool            found     = magnitude <= params.centeredLimit || (window & params.definiteMask) != 0;
  if (active && !found && (window & params.boundaryMask) != 0) {
    const double ownUnits = thousandthsToUnits(own);
    found = ((window & 1u) != 0 && fabs(ownUnits + thousandthsToUnits(-own - params.boundary)) <= params.value) ||
            (((window >> (2 * params.radius)) & 1u) != 0 &&
             fabs(ownUnits + thousandthsToUnits(-own + params.boundary)) <= params.value);
  }
  return active && found;
}

/**
 * @brief countWhimSymmetricBatchLoop() for rows of at most @p kChunks warp widths of atoms, each lane's values held
 *        in registers through the set, search and clear passes. Lanes past the last atom set and clear bit 0,
 *        in padding.
 */
template <int kChunks>
__device__ __forceinline__ int countWhimSymmetricBatchChunks(const int16_t*          values,
                                                             const int               numAtoms,
                                                             const int               lane,
                                                             const WhimBitmapParams& params,
                                                             uint32_t*               bitmap) {
  int32_t own[kChunks];
  int     bit[kChunks];
#pragma unroll
  for (int k = 0; k < kChunks; ++k) {
    const int atomIdx = lane + k * kWarpSize;
    own[k]            = atomIdx < numAtoms ? values[atomIdx] : 0;
    bit[k]            = atomIdx < numAtoms ? own[k] + kWhimBitmapOffset : 0;
  }
#pragma unroll
  for (int k = 0; k < kChunks; ++k) {
    atomicOr(&bitmap[bit[k] >> 5], 1u << (bit[k] & 31));
  }
  __syncwarp();
  int symmetric = 0;
#pragma unroll
  for (int k = 0; k < kChunks; ++k) {
    const bool active = lane + k * kWarpSize < numAtoms;
    symmetric += __popc(__ballot_sync(0xffffffffu, whimBatchSymmetric(own[k], active, params, bitmap)));
  }
  __syncwarp();
#pragma unroll
  for (int k = 0; k < kChunks; ++k) {
    bitmap[bit[k] >> 5] = 0;
  }
  __syncwarp();
  return symmetric;
}

/**
 * @brief Warp-collective symmetric atom count of one (channel, axis) pair from its rounded projections
 *        @p values (|v| < kWhimBitmapRange) via @p bitmap (all zero on entry and on return), for rows of more
 *        than countWhimSymmetricBatchChunks()'s atoms.
 *
 * whimBitmapSymmetric() for thresholds whose centered test covers every atom whose own bit lies in its window
 * (see whimBatchSupported()), so the window bits and RDKit's float64 boundary test decide every other atom.
 */
__device__ __forceinline__ int countWhimSymmetricBatchLoop(const int16_t*          values,
                                                           const int               numAtoms,
                                                           const int               lane,
                                                           const WhimBitmapParams& params,
                                                           uint32_t*               bitmap) {
  int symmetric = 0;
#pragma unroll 1
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    const int bit = values[atomIdx] + kWhimBitmapOffset;
    atomicOr(&bitmap[bit >> 5], 1u << (bit & 31));
  }
  __syncwarp();
#pragma unroll 1
  for (int base = 0; base < numAtoms; base += kWarpSize) {
    const int     atomIdx = base + lane;
    const bool    active  = atomIdx < numAtoms;
    const int32_t own     = active ? values[atomIdx] : 0;
    symmetric += __popc(__ballot_sync(0xffffffffu, whimBatchSymmetric(own, active, params, bitmap)));
  }
  __syncwarp();
#pragma unroll 1
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    bitmap[(values[atomIdx] + kWhimBitmapOffset) >> 5] = 0;
  }
  __syncwarp();
  return symmetric;
}

/**
 * @brief Warp-collective symmetric atom count of principal axis (@p axisX, @p axisY, @p axisZ) of a conformer
 *        whose rounded projections exceed the bitmap: recomputed in int32 into @p values (at least @p numAtoms
 *        words, zero on entry and on return) and compared pairwise as countWhimSymmetric() does for exhaustive
 *        thresholds. Out of line: no conformer of a typical batch takes it.
 */
__device__ __noinline__ int countWhimSymmetricPairwise(const double*       positions,
                                                       const double        centroidX,
                                                       const double        centroidY,
                                                       const double        centroidZ,
                                                       const WhimReal      axisX,
                                                       const WhimReal      axisY,
                                                       const WhimReal      axisZ,
                                                       const int           numAtoms,
                                                       const int           lane,
                                                       const WhimThreshold threshold,
                                                       int32_t*            values) {
#pragma unroll 1
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    WhimReal x;
    WhimReal y;
    WhimReal z;
    centeredPosition(positions, atomIdx, centroidX, centroidY, centroidZ, x, y, z);
    values[atomIdx] = whimRounded(whimScore(x, y, z, axisX, axisY, axisZ));
  }
  __syncwarp();
  int symmetric = 0;
#pragma unroll 1
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    const int32_t own   = values[atomIdx];
    bool          found = threshold.centered(own);
#pragma unroll 1
    for (int otherIdx = 0; otherIdx < numAtoms && !found; ++otherIdx) {
      found = otherIdx != atomIdx && threshold.mirrored(own, values[otherIdx]);
    }
    symmetric += found ? 1 : 0;
  }
  const int total = groupAllReduceSum<kWarpSize>(symmetric);
  __syncwarp();
#pragma unroll 1
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    values[atomIdx] = 0;
  }
  __syncwarp();
  return total;
}

/**
 * @brief Warp-collective symmetric atom counts of the three principal axes of one (conformer, channel) of the
 *        batched WHIM kernel whose rounded projections all fit the bitmap, written by lane 0 to @p counts:
 *        countWhimSymmetricBatchChunks() for rows of at most @p kChunks warp widths of atoms on the int16 scores
 *        @p values (an array of @p stride per axis).
 */
template <int kChunks>
__device__ __forceinline__ void countWhimChannelBatch(const int16_t*          values,
                                                      const int               stride,
                                                      const int               numAtoms,
                                                      const int               lane,
                                                      const WhimBitmapParams& params,
                                                      uint32_t*               bitmap,
                                                      int*                    counts) {
#pragma unroll 1
  for (int axis = 0; axis < 3; ++axis) {
    const int total = countWhimSymmetricBatchChunks<kChunks>(values + axis * stride, numAtoms, lane, params, bitmap);
    if (lane == 0) {
      counts[axis] = total;
    }
  }
}

/**
 * @brief Whether the batched WHIM kernel handles @p threshold: its window fits the bitmap search, and every atom
 *        whose own bit can lie in its window is centered (see countWhimSymmetricBatchLoop()).
 */
inline bool whimBatchSupported(const WhimThreshold& threshold) {
  if (!threshold.windowFits || threshold.exhaustive) {
    return false;
  }
  const int64_t centeredLimit = threshold.boundaryCentered ? threshold.boundary : threshold.centeredLimit;
  return threshold.windowRadius / 2 <= centeredLimit;
}

/**
 * @brief PBF and WHIM of conformers `kConformers * blockIdx.x ...`, a batch per block (see WhimBatchShared).
 *
 * Phases, each ending in a block barrier: (0) one warp per conformer centers the coordinates in float64;
 * (1) groups of kBatchGroupLanes lanes sum the covariance of each (conformer, slot); (2) a lane per
 * (conformer, slot) diagonalizes; (3) groups project and round each (conformer, channel), PBF included;
 * (4) warps take the (conformer, channel, axis) symmetry searches in turn; (5) a lane per output term writes
 * the WHIM rows. Rows longer than @p whimMaxAtoms are invalid, and invalid rows produce NaN.
 */
template <int kConformers, typename Real, bool kComputePbf>
__global__ void __launch_bounds__(kBatchThreads, 8) whimBatchKernel(const DeviceCoordView        coordinates,
                                                                    const Property3DDeviceInputs inputs,
                                                                    Real* __restrict__ pbfOutput,
                                                                    Real* __restrict__ whimOutput,
                                                                    const int             whimMaxAtoms,
                                                                    const WhimBatchLayout layout,
                                                                    const WhimThreshold   threshold) {
  extern __shared__ double dynamicScratch[];
  __shared__ WhimBatchShared<kConformers> shared;
  constexpr bool                          kSeparatePbf   = kComputePbf && !std::is_same_v<Real, WhimReal>;
  const int                               tid            = static_cast<int>(threadIdx.x);
  const int                               lane           = tid % kWarpSize;
  const int                               warp           = tid / kWarpSize;
  const int                               group          = lane / kBatchGroupLanes;
  const int                               laneInGroup    = lane % kBatchGroupLanes;
  const int                               stride         = layout.stride;
  const int                               firstConformer = static_cast<int>(blockIdx.x) * kConformers;
  uint32_t* const bitmap = reinterpret_cast<uint32_t*>(dynamicScratch) + warp * kWhimBitmapWords;
  int16_t* const  warpScores =
    reinterpret_cast<int16_t*>(reinterpret_cast<uint32_t*>(dynamicScratch) + kBatchWarps * kWhimBitmapWords) +
    warp * layout.warpScoreInts16();
  const double* const positions = coordinates.positions;
  static_assert(kWhimBitmapWords % 4 == 0, "16-byte aligned bitmaps");
  const int totalAtoms = inputs.moleculeAtomStarts[coordinates.nMols];

// Phase 0: one warp per conformer validates it and centers its coordinates; every warp clears its bitmap.
#pragma unroll 1
  for (int i = lane; i < kWhimBitmapWords / 4; i += kWarpSize) {
    reinterpret_cast<uint4*>(bitmap)[i] = make_uint4(0, 0, 0, 0);
  }
#pragma unroll 1
  for (int i = tid; i < kConformers * kNumWhimAxes; i += kBatchThreads) {
    (&shared.maxMagnitudes[0][0])[i] = 0;
  }
#pragma unroll 1
  for (int c = warp; c < kConformers; c += kBatchWarps) {
    const int conformerIdx = firstConformer + c;
    if (conformerIdx >= coordinates.numConformers) {
      if (lane == 0) {
        shared.numAtoms[c] = 0;
      }
      continue;
    }
    const ConformerRange range         = loadConformerRange(coordinates, conformerIdx);
    const int            numAtoms      = range.numAtoms;
    int                  moleculeStart = 0;
    int                  moleculeAtoms = -1;
    if (range.positions != nullptr) {
      moleculeStart = inputs.moleculeAtomStarts[range.moleculeIdx];
      moleculeAtoms = inputs.moleculeAtomStarts[range.moleculeIdx + 1] - moleculeStart;
    }
    if (moleculeAtoms != numAtoms || numAtoms > whimMaxAtoms) {  // Invalid; uniform across the warp.
      Real* const row = whimOutput + static_cast<size_t>(conformerIdx) * kNumWhimProperties;
#pragma unroll 1
      for (int valueIdx = lane; valueIdx < kNumWhimProperties; valueIdx += kWarpSize) {
        storeOutput<Real>(row + valueIdx, nan(""));
      }
      if (kComputePbf && lane == 0) {
        storeOutput<Real>(pbfOutput + conformerIdx, nan(""));
      }
      if (lane == 0) {
        shared.numAtoms[c] = 0;
      }
      continue;
    }
    double sumX = 0;
    double sumY = 0;
    double sumZ = 0;
#pragma unroll 1
    for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
      sumX += range.positions[atomIdx * 3 + 0];
      sumY += range.positions[atomIdx * 3 + 1];
      sumZ += range.positions[atomIdx * 3 + 2];
    }
    const double partials[4]  = {sumX, sumY, sumZ, 0.0};
    const double total        = warpTransposedSum4(partials, lane);
    const double inverseAtoms = 1.0 / static_cast<double>(numAtoms);
    const double centroidX    = __shfl_sync(0xffffffffu, total, 0) * inverseAtoms;
    const double centroidY    = __shfl_sync(0xffffffffu, total, 8) * inverseAtoms;
    const double centroidZ    = __shfl_sync(0xffffffffu, total, 16) * inverseAtoms;
    if (lane == 0) {
      shared.centroids[c][0]      = centroidX;
      shared.centroids[c][1]      = centroidY;
      shared.centroids[c][2]      = centroidZ;
      shared.atomStart[c]         = (range.positions - positions) / 3;
      shared.numAtoms[c]          = numAtoms;
      shared.moleculeAtomStart[c] = moleculeStart;
      shared.is3D[c] = !kComputePbf || inputs.conformerIs3D == nullptr || inputs.conformerIs3D[conformerIdx] != 0;
    }
  }
  __syncthreads();

// Phase 1: covariance sums, a group per (conformer, slot); a warp takes four slots of one conformer per pass.
#pragma unroll 1
  for (int half = warp; half < 2 * kConformers; half += kBatchWarps) {
    const int     c            = half >> 1;
    const int     slot         = (half & 1) * kBatchGroups + group;
    const int     numAtoms     = shared.numAtoms[c];
    const double* rowPositions = positions + 3 * shared.atomStart[c];
    const double  centroidX    = shared.centroids[c][0];
    const double  centroidY    = shared.centroids[c][1];
    const double  centroidZ    = shared.centroids[c][2];
    WhimReal      sums[8]      = {0, 0, 0, 0, 0, 0, 0, 0};
    // With its own PCA, PBF's float sums ride along the unweighted channel's (the first group of an even half):
    // the same atoms per lane and the same centered values narrowed to Real, so the same sums as a loop of their own.
    Real          pbfSums[8]   = {0, 0, 0, 0, 0, 0, 0, 0};
    if (slot < kNumWhimChannels) {
      // A unit weight leaves every product exact, so the unweighted channel takes the weighted path.
      const double* weights = inputs.atomPropertyWeights + shared.moleculeAtomStart[c] +
                              static_cast<size_t>(slot > 0 ? slot - 1 : 0) * totalAtoms;
#pragma unroll 1
      for (int atomIdx = laneInGroup; atomIdx < numAtoms; atomIdx += kBatchGroupLanes) {
        const WhimReal weight = slot > 0 ? weights[atomIdx] : WhimReal(1);
        WhimReal       x;
        WhimReal       y;
        WhimReal       z;
        centeredPosition(rowPositions, atomIdx, centroidX, centroidY, centroidZ, x, y, z);
        sums[0] += weight;
        sums[1] += weight * x * x;
        sums[2] += weight * x * y;
        sums[3] += weight * x * z;
        sums[4] += weight * y * y;
        sums[5] += weight * y * z;
        sums[6] += weight * z * z;
        if constexpr (kSeparatePbf) {
          if (slot == 0) {
            const Real pbfX = static_cast<Real>(x);
            const Real pbfY = static_cast<Real>(y);
            const Real pbfZ = static_cast<Real>(z);
            pbfSums[0] += Real(1);
            pbfSums[1] += pbfX * pbfX;
            pbfSums[2] += pbfX * pbfY;
            pbfSums[3] += pbfX * pbfZ;
            pbfSums[4] += pbfY * pbfY;
            pbfSums[5] += pbfY * pbfZ;
            pbfSums[6] += pbfZ * pbfZ;
          }
        }
      }
    }
    const WhimReal sum = groupTransposedSum8(sums, lane);
    if (slot < kNumWhimChannels && laneInGroup < 7) {
      shared.covarianceSums[c][slot][laneInGroup] = sum;
    }
    if constexpr (kSeparatePbf) {
      if ((half & 1) == 0) {  // Uniform across the warp.
        const Real pbfSum = groupTransposedSum8(pbfSums, lane);
        if (slot == 0 && laneInGroup < 7) {
          shared.covarianceSums[c][kNumWhimChannels][laneInGroup] = pbfSum;
        }
      }
    }
  }
  __syncthreads();

  // Phase 2: a lane per (conformer, channel) eigensystem; the next warp PBF's own eigensystems, then the
  // logarithms the symmetry terms share.
  constexpr int kEigenWarps = (kConformers * kNumWhimChannels + kWarpSize - 1) / kWarpSize;
  constexpr int kPbfWarp    = kEigenWarps;
  constexpr int kLogWarp    = kEigenWarps + (kSeparatePbf ? 1 : 0);
  static_assert(kLogWarp < kBatchWarps && kConformers <= kWarpSize, "phase 2 fits the block");
  if (tid < kConformers * kNumWhimChannels) {
    const int c       = tid / kNumWhimChannels;
    const int channel = tid % kNumWhimChannels;
    if (shared.numAtoms[c] > 0) {
      WhimReal sums[7];
      for (int i = 0; i < 7; ++i) {
        sums[i] = shared.covarianceSums[c][channel][i];
      }
      WhimReal covariance[6];
      normalizeCovariance(sums, channel == 0, shared.numAtoms[c], covariance);
      ProjectionState<WhimReal> state;
      diagonalizeProjection(covariance, state);
      for (int i = 0; i < 9; ++i) {
        shared.eigen[c][channel][i] = state.eigenvectors[i];
      }
      for (int i = 0; i < 3; ++i) {
        shared.eigen[c][channel][9 + i] = state.eigenvalues[i];
      }
    }
  } else if (kSeparatePbf && warp == kPbfWarp && lane < kConformers) {
    if (shared.numAtoms[lane] > 0) {
      Real sums[7];
      for (int i = 0; i < 7; ++i) {
        sums[i] = static_cast<Real>(shared.covarianceSums[lane][kNumWhimChannels][i]);
      }
      Real covariance[6];
      normalizeCovariance(sums, true, shared.numAtoms[lane], covariance);
      ProjectionState<Real> state;
      diagonalizeProjection(covariance, state);
      shared.pbfAxis[lane][0] = state.eigenvectors[2];
      shared.pbfAxis[lane][1] = state.eigenvectors[5];
      shared.pbfAxis[lane][2] = state.eigenvectors[8];
    }
  } else if (warp == kLogWarp && lane < kConformers) {
    if (shared.numAtoms[lane] > 0) {
      shared.logInverseAtoms[lane] = log(WhimReal(1) / static_cast<WhimReal>(shared.numAtoms[lane]));
    }
  }
  __syncthreads();

  // Phase 3: projections and symmetry searches. Each pass, a group per (conformer, channel) task and, with its
  // own PCA, PBF's projects into the warp's score buffer (rounded scores, their fourth-power sums and largest
  // magnitudes; in float64 PBF shares channel 0's projections), then the warp searches its tasks' (channel,
  // axis) pairs in turn. Projections too large for the bitmap are recomputed in int32 and compared pairwise.
  // Tasks are the (conformer, channel) pairs, then with its own PCA each conformer's PBF projection; they are
  // dealt round-robin over the warps, so that every warp gets the same number of searches (within one channel) and
  // a mix of the batch's conformers, and each pass a warp's groups take its next four.
  constexpr int          kProjectionTasks = kNumWhimChannels + (kSeparatePbf ? 1 : 0);
  constexpr int          kWhimTasks       = kConformers * kNumWhimChannels;
  constexpr int          kTasksPerPass    = kBatchWarps * kBatchGroups;
  const WhimBitmapParams params           = WhimBitmapParams::make(threshold);
#pragma unroll 1
  for (int pass = 0; pass * kTasksPerPass < kConformers * kProjectionTasks; ++pass) {
    const int     task         = (pass * kBatchGroups + group) * kBatchWarps + warp;
    const bool    whimTask     = task < kWhimTasks;
    const int     c            = whimTask ? task / kNumWhimChannels : task - kWhimTasks;
    const int     channel      = whimTask ? task % kNumWhimChannels : kNumWhimChannels;
    const int     numAtoms     = task < kConformers * kProjectionTasks ? shared.numAtoms[c] : 0;
    const double* rowPositions = positions;
    double        centroidX    = 0;
    double        centroidY    = 0;
    double        centroidZ    = 0;
    if (numAtoms > 0) {
      rowPositions = positions + 3 * shared.atomStart[c];
      centroidX    = shared.centroids[c][0];
      centroidY    = shared.centroids[c][1];
      centroidZ    = shared.centroids[c][2];
    }
    WhimReal moments[4] = {0, 0, 0, 0};
    Real     pbfSum     = 0;
    if (channel < kNumWhimChannels) {
      WhimReal axes[9];
      if (numAtoms > 0) {
        for (int i = 0; i < 9; ++i) {
          axes[i] = shared.eigen[c][channel][i];
        }
      }
      int16_t* const channelScores = warpScores + group * 3 * stride;
      uint32_t       magnitudes[3] = {0, 0, 0};
#pragma unroll 1
      for (int atomIdx = laneInGroup; atomIdx < numAtoms; atomIdx += kBatchGroupLanes) {
        WhimReal x;
        WhimReal y;
        WhimReal z;
        centeredPosition(rowPositions, atomIdx, centroidX, centroidY, centroidZ, x, y, z);
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
          const WhimReal score = whimScore(x, y, z, axes[axis], axes[3 + axis], axes[6 + axis]);
          moments[axis] += score * score * score * score;
          const int32_t  rounded   = whimRounded(score);
          const uint32_t magnitude = whimMagnitude(rounded);
          channelScores[axis * stride + atomIdx] =
            static_cast<int16_t>(magnitude < static_cast<uint32_t>(kWhimBitmapRange) ? rounded : 0);
          magnitudes[axis] = magnitude > magnitudes[axis] ? magnitude : magnitudes[axis];
          if constexpr (kComputePbf && !kSeparatePbf) {
            if (axis == 2) {
              pbfSum += fabs(score);
            }
          }
        }
      }
      if (numAtoms > 0) {
        for (int axis = 0; axis < 3; ++axis) {
          atomicMax(&shared.maxMagnitudes[c][3 * channel + axis], magnitudes[axis]);
        }
      }
    } else if constexpr (kSeparatePbf) {
      // Mean absolute distance from the plane of best fit, as computePbf().
      Real axisX = 0;
      Real axisY = 0;
      Real axisZ = 0;
      if (numAtoms > 0) {
        axisX = static_cast<Real>(shared.pbfAxis[c][0]);
        axisY = static_cast<Real>(shared.pbfAxis[c][1]);
        axisZ = static_cast<Real>(shared.pbfAxis[c][2]);
      }
#pragma unroll 1
      for (int atomIdx = laneInGroup; atomIdx < numAtoms; atomIdx += kBatchGroupLanes) {
        Real x;
        Real y;
        Real z;
        centeredPosition(rowPositions, atomIdx, centroidX, centroidY, centroidZ, x, y, z);
        pbfSum += fabs(x * axisX + y * axisY + z * axisZ);
      }
    }
    const WhimReal moment    = groupTransposedSum4(moments, lane);
    const int      momentIdx = (laneInGroup >> 1) & 3;
    if (channel < kNumWhimChannels && numAtoms > 0 && (laneInGroup & 1) == 0 && momentIdx < 3) {
      shared.fourthMoments[c][3 * channel + momentIdx] = moment;
    }
    if constexpr (kComputePbf) {
      const Real pbfTotal = groupButterflySum(pbfSum);
      const bool pbfTask  = kSeparatePbf ? channel == kNumWhimChannels : channel == 0;
      if (pbfTask && numAtoms > 0 && laneInGroup == 0) {
        const Real pbf = pbfTotal / static_cast<Real>(numAtoms);
        storeOutput<Real>(pbfOutput + firstConformer + c, numAtoms < 4 || !shared.is3D[c] ? Real(0) : pbf);
      }
    }
    __syncwarp();
    // The warp searches its WHIM tasks' axes, choosing the bitmap search's variant once per task.
#pragma unroll 1
    for (int warpTask = 0; warpTask < kBatchGroups; ++warpTask) {
      const int searchTask = (pass * kBatchGroups + warpTask) * kBatchWarps + warp;
      if (searchTask >= kWhimTasks) {
        break;  // The PBF tasks follow every WHIM task.
      }
      const int searchConformer = searchTask / kNumWhimChannels;
      const int searchChannel   = searchTask % kNumWhimChannels;
      const int searchAtoms     = shared.numAtoms[searchConformer];
      if (searchAtoms == 0) {
        continue;
      }
      const int16_t* const  values  = warpScores + warpTask * 3 * stride;
      int* const            counts  = &shared.symmetricCounts[searchConformer][3 * searchChannel];
      const uint32_t* const maxima  = &shared.maxMagnitudes[searchConformer][3 * searchChannel];
      const uint32_t        largest = maxima[0] > maxima[1] ? (maxima[0] > maxima[2] ? maxima[0] : maxima[2]) :
                                                              (maxima[1] > maxima[2] ? maxima[1] : maxima[2]);
      if (largest < static_cast<uint32_t>(kWhimBitmapRange) && searchAtoms <= 4 * kWarpSize) {
        if (searchAtoms <= 2 * kWarpSize) {
          if (searchAtoms <= kWarpSize) {
            countWhimChannelBatch<1>(values, stride, searchAtoms, lane, params, bitmap, counts);
          } else {
            countWhimChannelBatch<2>(values, stride, searchAtoms, lane, params, bitmap, counts);
          }
        } else if (searchAtoms <= 3 * kWarpSize) {
          countWhimChannelBatch<3>(values, stride, searchAtoms, lane, params, bitmap, counts);
        } else {
          countWhimChannelBatch<4>(values, stride, searchAtoms, lane, params, bitmap, counts);
        }
        continue;
      }
      // Longer rows, or projections too large for the bitmap: recomputed in int32 and compared pairwise.
#pragma unroll 1
      for (int axis = 0; axis < 3; ++axis) {
        int total;
        if (maxima[axis] < static_cast<uint32_t>(kWhimBitmapRange)) {
          total = countWhimSymmetricBatchLoop(values + axis * stride, searchAtoms, lane, params, bitmap);
        } else {
          const double* const axes = shared.eigen[searchConformer][searchChannel];
          total                    = countWhimSymmetricPairwise(positions + 3 * shared.atomStart[searchConformer],
                                             shared.centroids[searchConformer][0],
                                             shared.centroids[searchConformer][1],
                                             shared.centroids[searchConformer][2],
                                             axes[axis],
                                             axes[3 + axis],
                                             axes[6 + axis],
                                             searchAtoms,
                                             lane,
                                             threshold,
                                             reinterpret_cast<int32_t*>(bitmap));
        }
        if (lane == 0) {
          counts[axis] = total;
        }
      }
    }
    __syncwarp();  // The buffer is searched before the next pass overwrites it.
  }
  __syncthreads();

  // Phase 5: a lane per (conformer, channel, axis) writes its terms; the last warp the channels' eigenvalue
  // terms meanwhile. Then a lane per (conformer, channel) writes the means.
  constexpr int kEigenvalueTermBase = (kConformers * kNumWhimAxes + kWarpSize - 1) / kWarpSize * kWarpSize;
  static_assert(kEigenvalueTermBase + kConformers * kNumWhimChannels <= kBatchThreads, "phase 5 fits the block");
  if (tid < kConformers * kNumWhimAxes) {
    const int c        = tid / kNumWhimAxes;
    const int pair     = tid % kNumWhimAxes;
    const int numAtoms = shared.numAtoms[c];
    if (numAtoms > 0) {
      const int channel = pair / 3;
      const int axis    = pair % 3;
      WhimReal  gamma;
      WhimReal  kurtosis;
      writeWhimAxisTerms<Real>(pair,
                               shared.eigen[c][channel][9 + axis],
                               shared.fourthMoments[c][pair],
                               shared.symmetricCounts[c][pair],
                               numAtoms,
                               shared.logInverseAtoms[c],
                               whimOutput + static_cast<size_t>(firstConformer + c) * kNumWhimProperties,
                               gamma,
                               kurtosis);
      shared.axisTerms.gammas[c][channel][axis]   = gamma;
      shared.axisTerms.kurtoses[c][channel][axis] = kurtosis;
    }
  } else if (tid >= kEigenvalueTermBase && tid < kEigenvalueTermBase + kConformers * kNumWhimChannels) {
    const int c       = (tid - kEigenvalueTermBase) / kNumWhimChannels;
    const int channel = (tid - kEigenvalueTermBase) % kNumWhimChannels;
    if (shared.numAtoms[c] > 0) {
      const WhimReal eigenvalues[3] = {shared.eigen[c][channel][9],
                                       shared.eigen[c][channel][10],
                                       shared.eigen[c][channel][11]};
      writeWhimEigenvalueTerms<Real>(channel,
                                     eigenvalues,
                                     whimOutput + static_cast<size_t>(firstConformer + c) * kNumWhimProperties);
    }
  }
  __syncthreads();
  if (tid < kConformers * kNumWhimChannels) {
    const int c       = tid / kNumWhimChannels;
    const int channel = tid % kNumWhimChannels;
    if (shared.numAtoms[c] > 0) {
      writeWhimMeans<Real>(channel,
                           shared.axisTerms.gammas[c][channel],
                           shared.axisTerms.kurtoses[c][channel],
                           whimOutput + static_cast<size_t>(firstConformer + c) * kNumWhimProperties);
    }
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

//! Launches whimBatchKernel() with @p kConformers per block when its shared memory fits the device; returns
//! whether it did.
template <int kConformers, typename Real, bool kComputePbf>
bool launchWhimBatch(const DeviceCoordView&        coordinates,
                     const Property3DDeviceInputs& inputs,
                     const WhimThreshold&          threshold,
                     Real*                         pbfOutput,
                     Real*                         whimOutput,
                     const int                     maxAtoms,
                     const int                     device,
                     const cudaStream_t            stream) {
  const WhimBatchLayout layout = WhimBatchLayout::make(maxAtoms);
  auto* const           kernel = whimBatchKernel<kConformers, Real, kComputePbf>;
  const size_t          bytes  = layout.bytes();
  if (!configureWhimSharedScratch(kernel, bytes, sizeof(WhimBatchShared<kConformers>), device)) {
    return false;
  }
  const int blocks = (coordinates.numConformers + kConformers - 1) / kConformers;
  kernel<<<blocks, kBatchThreads, bytes, stream>>>(coordinates,
                                                   inputs,
                                                   pbfOutput,
                                                   whimOutput,
                                                   maxAtoms,
                                                   layout,
                                                   threshold);
  cudaCheckError(cudaGetLastError());
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

  // A grid that the current device holds in one wave of wide blocks takes a warp per projection slot.
  constexpr int kWaveWarps       = whimWaveWarps(kComputePbf);
  auto* const   waveKernel       = whimKernel<kWaveWarps, Real, kComputePbf, false>;
  const size_t  waveScratchBytes = layout.bytes(kWaveWarps);
  if (configureWhimSharedScratch(waveKernel, waveScratchBytes, sizeof(WhimBlockShared<kWaveWarps>), device)) {
    int numSMs       = 0;
    int activeBlocks = 0;
    cudaCheckError(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
    cudaCheckError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks,
                                                                 waveKernel,
                                                                 kWaveWarps * kWarpSize,
                                                                 waveScratchBytes));
    if (static_cast<int64_t>(numConformers) <= static_cast<int64_t>(numSMs) * activeBlocks) {
      waveKernel<<<numConformers, kWaveWarps * kWarpSize, waveScratchBytes, stream>>>(coordinates,
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
  }

  const size_t scratchBytes = layout.bytes(kWhimWarps);
  auto* const  sharedKernel = whimKernel<kWhimWarps, Real, kComputePbf, false>;
  const bool   sharedFits =
    configureWhimSharedScratch(sharedKernel, scratchBytes, sizeof(WhimBlockShared<kWhimWarps>), device);

  // Grids filling at least half a resident wave of four-conformer batched blocks pack four conformers into
  // each; grids that a block per conformer would take more than one wave for, but two-conformer batched
  // blocks take in one, pack two (half the batched block's passes, so about half its latency).
  if (whimBatchSupported(threshold) && maxAtoms <= kBatchMaxAtoms) {
    constexpr int kBatchConformers = 4;
    auto* const   batchKernel      = whimBatchKernel<kBatchConformers, Real, kComputePbf>;
    const size_t  batchBytes       = WhimBatchLayout::make(maxAtoms).bytes();
    int           numSMs           = 0;
    int           batchBlocks      = 0;
    cudaCheckError(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
    if (configureWhimSharedScratch(batchKernel, batchBytes, sizeof(WhimBatchShared<kBatchConformers>), device)) {
      cudaCheckError(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&batchBlocks, batchKernel, kBatchThreads, batchBytes));
    }
    if (2 * static_cast<int64_t>(numConformers) > static_cast<int64_t>(numSMs) * batchBlocks * kBatchConformers &&
        launchWhimBatch<kBatchConformers, Real, kComputePbf>(coordinates,
                                                             inputs,
                                                             threshold,
                                                             pbfOutput,
                                                             whimOutput,
                                                             maxAtoms,
                                                             device,
                                                             stream)) {
      return;
    }
    constexpr int kPairConformers = 2;
    auto* const   pairKernel      = whimBatchKernel<kPairConformers, Real, kComputePbf>;
    int           sharedBlocks    = 0;
    int           pairBlocks      = 0;
    if (sharedFits) {
      cudaCheckError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&sharedBlocks,
                                                                   sharedKernel,
                                                                   kWhimWarps * kWarpSize,
                                                                   scratchBytes));
    }
    if (configureWhimSharedScratch(pairKernel, batchBytes, sizeof(WhimBatchShared<kPairConformers>), device)) {
      cudaCheckError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&pairBlocks, pairKernel, kBatchThreads, batchBytes));
    }
    if (static_cast<int64_t>(numConformers) > static_cast<int64_t>(numSMs) * sharedBlocks &&
        static_cast<int64_t>(numConformers) <= static_cast<int64_t>(numSMs) * pairBlocks * kPairConformers &&
        launchWhimBatch<kPairConformers, Real, kComputePbf>(coordinates,
                                                            inputs,
                                                            threshold,
                                                            pbfOutput,
                                                            whimOutput,
                                                            maxAtoms,
                                                            device,
                                                            stream)) {
      return;
    }
  }

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
