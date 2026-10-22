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

//! Centered coordinates of atom @p atomIdx, from @p centered when it is non-null.
template <typename Real>
__device__ __forceinline__ void loadCentered(const ConformerAtoms&        atoms,
                                             const int                    atomIdx,
                                             const ProjectionState<Real>& state,
                                             const Real*                  centered,
                                             Real&                        x,
                                             Real&                        y,
                                             Real&                        z) {
  if (centered != nullptr) {
    x = centered[atomIdx * 3 + 0];
    y = centered[atomIdx * 3 + 1];
    z = centered[atomIdx * 3 + 2];
  } else {
    centeredCoordinates(atoms, atomIdx, state, x, y, z);
  }
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
 * by the atom count. A non-null @p centered (xyz per atom, as centeredCoordinates() gives them) replaces
 * centering the positions.
 */
template <typename Real>
__device__ __forceinline__ void computeProjectionCovariance(const ConformerAtoms&        atoms,
                                                            const double*                weights,
                                                            const int                    laneInGroup,
                                                            const ProjectionState<Real>& state,
                                                            Real (&covariance)[6],
                                                            const Real* centered = nullptr) {
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
    loadCentered(atoms, atomIdx, state, centered, x, y, z);
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

/**
 * @brief Jacobi rotation annihilating @p apq of a symmetric 3x3 matrix (@p arp, @p arq: the remaining index's
 *        entries), rotating columns @p p and @p q of the row-major @p eigenvectors with it.
 *
 * The same rotation as symmetricEigensystemJacobi3x3()'s (the smaller angle, t = tan in [-1, 1]), from its
 * half-angle relations rather than theta = (aqq - app) / 2 apq: with d = aqq - app, u = sgn(d) 2 apq and
 * r = sqrt(d^2 + u^2), 1 + t^2 = 2 r / (|d| + r), so one square root and one reciprocal square root give
 * c = (|d| + r) / sqrt(2 r (|d| + r)), s = u / sqrt(2 r (|d| + r)) and t = s 2 r / sqrt(2 r (|d| + r)), where
 * the theta form needs three divisions and two square roots in sequence. Accuracy is unchanged: c^2 + s^2 = 1
 * to rounding, so the iteration stays backward stable in @p Real.
 */
template <typename Real>
__device__ __forceinline__ void projectionJacobiRotate(Real&       app,
                                                       Real&       aqq,
                                                       Real&       apq,
                                                       Real&       arp,
                                                       Real&       arq,
                                                       Real* const eigenvectors,
                                                       const int   p,
                                                       const int   q) {
  if (apq == Real(0)) {
    return;
  }
  const Real d       = aqq - app;
  const Real u       = copysign(Real(2), d) * apq;
  const Real r       = sqrt(d * d + u * u);
  const Real sum     = fabs(d) + r;
  const Real inverse = rsqrt(Real(2) * r * sum);
  if (!(inverse < cuda::std::numeric_limits<Real>::infinity())) {
    return;  // d^2 + u^2 underflowed: apq is negligible.
  }
  const Real c         = sum * inverse;
  const Real s         = u * inverse;
  const Real tTimesApq = s * (Real(2) * r * inverse) * apq;
  app -= tTimesApq;
  aqq += tTimesApq;
  apq           = Real(0);
  const Real rp = arp;
  const Real rq = arq;
  arp           = c * rp - s * rq;
  arq           = s * rp + c * rq;
#pragma unroll
  for (int row = 0; row < 3; ++row) {
    const Real vp             = eigenvectors[row * 3 + p];
    const Real vq             = eigenvectors[row * 3 + q];
    eigenvectors[row * 3 + p] = c * vp - s * vq;
    eigenvectors[row * 3 + q] = s * vp + c * vq;
  }
}

//! Cyclic Jacobi eigensystem of a symmetric 3x3 matrix, as symmetricEigensystemJacobi3x3() (same sweeps and
//! stopping rule) with projectionJacobiRotate(): unsorted @p eigenvalues, column i of @p eigenvectors paired
//! with eigenvalues[i].
template <typename Real>
__device__ __forceinline__ void projectionEigensystem3x3(Real a00,
                                                         Real a01,
                                                         Real a02,
                                                         Real a11,
                                                         Real a12,
                                                         Real a22,
                                                         Real (&eigenvalues)[3],
                                                         Real (&eigenvectors)[9]) {
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    eigenvectors[i] = (i % 4 == 0) ? Real(1) : Real(0);
  }
  constexpr int  kMaxSweeps = 16;
  constexpr Real kEpsilon   = cuda::std::numeric_limits<Real>::epsilon();
  for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
    const Real offDiagonal = a01 * a01 + a02 * a02 + a12 * a12;
    const Real diagonal    = a00 * a00 + a11 * a11 + a22 * a22;
    if (!(offDiagonal > kEpsilon * kEpsilon * diagonal)) {
      break;
    }
    projectionJacobiRotate(a00, a11, a01, a02, a12, eigenvectors, 0, 1);  // r = 2
    projectionJacobiRotate(a00, a22, a02, a01, a12, eigenvectors, 0, 2);  // r = 1
    projectionJacobiRotate(a11, a22, a12, a01, a02, eigenvectors, 1, 2);  // r = 0
  }
  eigenvalues[0] = a00;
  eigenvalues[1] = a11;
  eigenvalues[2] = a22;
}

//! Replaces @p state's eigenvalues by their magnitudes and sorts them in descending order with their axes.
template <typename Real> __device__ __forceinline__ void sortProjectionEigenpairs(ProjectionState<Real>& state) {
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

//! Eigensystem of @p covariance into @p state: eigenvalue magnitudes in descending order with their axes.
//! computeProjectionPca() uses it, so other descriptors that share that PCA (GETAWAY's planarity test) keep
//! symmetricEigensystemJacobi3x3()'s results.
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
  sortProjectionEigenpairs(state);
}

//! diagonalizeProjection() with projectionEigensystem3x3()'s shorter rotation, for WHIM and PBF.
template <typename Real>
__device__ __forceinline__ void diagonalizeProjectionFast(const Real (&covariance)[6], ProjectionState<Real>& state) {
  projectionEigensystem3x3(covariance[0],
                           covariance[1],
                           covariance[2],
                           covariance[3],
                           covariance[4],
                           covariance[5],
                           state.eigenvalues,
                           state.eigenvectors);
  sortProjectionEigenpairs(state);
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

//! computePbf() along the third column of the row-major @p axes, from @p centered coordinates.
template <typename Real>
__device__ __forceinline__ Real
computePbf(const ConformerAtoms& atoms, const int laneInGroup, const Real (&axes)[9], const Real* centered) {
  Real distanceSum = 0;
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    const Real x = centered[atomIdx * 3 + 0];
    const Real y = centered[atomIdx * 3 + 1];
    const Real z = centered[atomIdx * 3 + 2];
    distanceSum += fabs(x * axes[2] + y * axes[5] + z * axes[8]);
  }
  return groupAllReduceSum(distanceSum) / static_cast<Real>(atoms.numAtoms > 0 ? atoms.numAtoms : 1);
}

/**
 * @brief Block-collective. The smallest principal axis of each group's @p covariance into @p state's third
 *        eigenvector column, for computePbf().
 *
 * One lane of the block's first warp diagonalizes each group's covariance, so the block's eigensolves take one
 * warp's instructions instead of every lane solving its group's copy.
 */
template <typename Real>
__device__ __forceinline__ void diagonalizePbfInBlock(const Real (&covariance)[6],
                                                      const int              laneInGroup,
                                                      ProjectionState<Real>& state) {
  __shared__ Real sharedPca[6][kConformersPerBlock];
  const int       group = static_cast<int>(threadIdx.x) / kGroupSize;
  if (laneInGroup == 0) {
    for (int i = 0; i < 6; ++i) {
      sharedPca[i][group] = covariance[i];
    }
  }
  __syncthreads();
  if (threadIdx.x < kConformersPerBlock) {
    const int             solved              = static_cast<int>(threadIdx.x);
    const Real            solvedCovariance[6] = {sharedPca[0][solved],
                                                 sharedPca[1][solved],
                                                 sharedPca[2][solved],
                                                 sharedPca[3][solved],
                                                 sharedPca[4][solved],
                                                 sharedPca[5][solved]};
    ProjectionState<Real> solvedState;
    diagonalizeProjectionFast(solvedCovariance, solvedState);
    for (int row = 0; row < 3; ++row) {
      sharedPca[row][solved] = solvedState.eigenvectors[row * 3 + 2];
    }
  }
  __syncthreads();
  for (int row = 0; row < 3; ++row) {
    state.eigenvectors[row * 3 + 2] = sharedPca[row][group];
  }
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

//! WHIM's unweighted channel followed by the atom-property channels. Lane c of a group owns channel c.
constexpr int kNumWhimChannels = 1 + kNumAtomPropertyChannels;
static_assert(kNumWhimChannels <= kGroupSize, "each WHIM channel needs its own lane");

/**
 * @brief Per-conformer WHIM scratch: `stride` int4 rounded projections (x, y, z = principal axes), then
 *        3 * `stride` float64 centered coordinates. `stride` is even so every conformer's block stays
 *        16-byte aligned.
 */
struct WhimScratch {
  int4*     scores;
  WhimReal* centered;
};
constexpr size_t kWhimScratchBytesPerAtom = sizeof(int4) + 3 * sizeof(WhimReal);

//! Atoms per conformer block in the WHIM scratch for rows of up to @p maxAtoms atoms.
__host__ __device__ inline int whimScratchStride(const int maxAtoms) {
  const int atoms = maxAtoms > 1 ? maxAtoms : 1;
  return atoms + atoms % 2;
}

__device__ __forceinline__ WhimScratch whimScratchFor(int4* base, const int conformerOffset, const int stride) {
  int4* scores = reinterpret_cast<int4*>(reinterpret_cast<char*>(base) +
                                         static_cast<size_t>(conformerOffset) * kWhimScratchBytesPerAtom * stride);
  return WhimScratch{scores, reinterpret_cast<WhimReal*>(scores + stride)};
}

//! Per-axis sums of one WHIM channel over the conformer's atoms.
struct WhimChannelSums {
  WhimReal fourthMoments[3];    //!< Sum of score^4 along each principal axis.
  int      symmetricCounts[3];  //!< Atoms whose rounded projection is centered or mirrored.
};

/**
 * @brief Group-collective. Projects every atom on the principal @p axes (row-major, column i is axis i),
 *        then counts per axis the atoms whose rounded projection is centered, or mirrored by another atom
 *        within the threshold. Every lane receives the same sums.
 *
 * The pairwise search compares every atom with every other, so each projection is rounded once to
 * integer thousandths in `scratch.scores`; `scratch.centered` holds the centered coordinates (@p groupMask
 * selects the group's lanes). One pass over the other atoms serves all three axes: an integer filter
 * rejects nearly every pair, and only candidates go through the exact comparison.
 */
__device__ __forceinline__ WhimChannelSums accumulateWhimChannel(const ConformerAtoms& atoms,
                                                                 const int             laneInGroup,
                                                                 const unsigned        groupMask,
                                                                 const WhimReal (&axes)[9],
                                                                 const WhimThreshold& threshold,
                                                                 const WhimScratch&   scratch) {
  int4* const     scores = scratch.scores;
  WhimChannelSums sums{};
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    const WhimReal x = scratch.centered[atomIdx * 3 + 0];
    const WhimReal y = scratch.centered[atomIdx * 3 + 1];
    const WhimReal z = scratch.centered[atomIdx * 3 + 2];
    int32_t        rounded[3];
    for (int axis = 0; axis < 3; ++axis) {
      const WhimReal score = x * axes[axis] + y * axes[3 + axis] + z * axes[6 + axis];
      sums.fourthMoments[axis] += score * score * score * score;
      rounded[axis] = static_cast<int32_t>(round(score * WhimReal(1000)));
    }
    scores[atomIdx] = make_int4(rounded[0], rounded[1], rounded[2], 0);
  }
  for (int axis = 0; axis < 3; ++axis) {
    sums.fourthMoments[axis] = groupAllReduceSum(sums.fourthMoments[axis]);
  }
  __syncwarp(groupMask);

  int symmetric[3] = {0, 0, 0};
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    const int4 own    = scores[atomIdx];
    bool       foundX = threshold.centered(own.x);
    bool       foundY = threshold.centered(own.y);
    bool       foundZ = threshold.centered(own.z);
    for (int otherIdx = 0; otherIdx < atoms.numAtoms && !(foundX && foundY && foundZ); ++otherIdx) {
      const int4 other      = scores[otherIdx];
      const bool candidateX = !foundX && threshold.mayMirror(own.x, other.x);
      const bool candidateY = !foundY && threshold.mayMirror(own.y, other.y);
      const bool candidateZ = !foundZ && threshold.mayMirror(own.z, other.z);
      if ((candidateX || candidateY || candidateZ) && otherIdx != atomIdx) {
        foundX = foundX || (candidateX && threshold.mirrored(own.x, other.x));
        foundY = foundY || (candidateY && threshold.mirrored(own.y, other.y));
        foundZ = foundZ || (candidateZ && threshold.mirrored(own.z, other.z));
      }
    }
    symmetric[0] += foundX ? 1 : 0;
    symmetric[1] += foundY ? 1 : 0;
    symmetric[2] += foundZ ? 1 : 0;
  }
  for (int axis = 0; axis < 3; ++axis) {
    sums.symmetricCounts[axis] = groupAllReduceSum(symmetric[axis]);
  }
  __syncwarp(groupMask);  // Every lane is done reading before the next channel overwrites the scores.
  return sums;
}

//! WHIM symmetry term from the symmetric atom count along one axis; @p logInverseAtoms is log(1 / numAtoms).
__device__ __forceinline__ WhimReal whimGamma(const int      symmetricAtoms,
                                              const WhimReal numAtoms,
                                              const WhimReal logInverseAtoms) {
  const WhimReal symmetricCount  = static_cast<WhimReal>(symmetricAtoms);
  const WhimReal asymmetricCount = numAtoms - symmetricCount;
  WhimReal       inverseGamma;
  if (symmetricCount == WhimReal(0)) {
    inverseGamma = WhimReal(1) - (asymmetricCount / numAtoms) * logInverseAtoms / log(WhimReal(2));
  } else {
    inverseGamma = WhimReal(1) - ((symmetricCount / numAtoms) * log(symmetricCount / numAtoms) / log(WhimReal(2)) +
                                  (asymmetricCount / numAtoms) * logInverseAtoms / log(WhimReal(2)));
  }
  return WhimReal(1) / inverseGamma;
}

/**
 * @brief Writes WHIM channel @p channel of one conformer's 114-value row from its eigenvalues and sums.
 *
 * Layout matches RDKit's CalcWHIM: 11 directional values per channel (eigenvalues, two eigenvalue ratios,
 * three symmetry terms, three inverse kurtoses), then 7 totals, 7 pairwise products, 2 symmetry means,
 * 7 anisotropies, 7 kurtosis means and 7 combined terms.
 */
template <typename OutputReal>
__device__ __forceinline__ void writeWhimChannel(const int                        channel,
                                                 const ProjectionState<WhimReal>& state,
                                                 const WhimChannelSums&           sums,
                                                 const int                        numAtomsInt,
                                                 OutputReal*                      row) {
  const WhimReal first    = state.eigenvalues[0];
  const WhimReal second   = state.eigenvalues[1];
  const WhimReal third    = state.eigenvalues[2];
  const WhimReal total    = first + second + third;
  const WhimReal numAtoms = static_cast<WhimReal>(numAtomsInt);
  const WhimReal e1 =
    sums.fourthMoments[0] > WhimReal(0) ? numAtoms * first * first / sums.fourthMoments[0] : WhimReal(0);
  const WhimReal e2 =
    sums.fourthMoments[1] > WhimReal(0) ? numAtoms * second * second / sums.fourthMoments[1] : WhimReal(0);
  const WhimReal e3 =
    sums.fourthMoments[2] > WhimReal(0) ? numAtoms * third * third / sums.fourthMoments[2] : WhimReal(0);

  const int channelStart    = channel * 11;
  row[channelStart + 0]     = roundWhimThousandths(first);
  row[channelStart + 1]     = roundWhimThousandths(second);
  row[channelStart + 2]     = roundWhimThousandths(third);
  row[channelStart + 3]     = roundWhimThousandths(first / total);
  row[channelStart + 4]     = roundWhimThousandths(second / total);
  row[channelStart + 8]     = roundWhimThousandths(e1);
  row[channelStart + 9]     = roundWhimThousandths(e2);
  row[channelStart + 10]    = roundWhimThousandths(e3);
  row[77 + channel]         = roundWhimThousandths(total);
  row[84 + channel]         = roundWhimThousandths(first * second + first * third + second * third);
  const WhimReal anisotropy = WhimReal(0.75) * (fabs(first / total - WhimReal(1) / WhimReal(3)) +
                                                fabs(second / total - WhimReal(1) / WhimReal(3)) +
                                                fabs(third / total - WhimReal(1) / WhimReal(3)));
  row[93 + channel]         = roundWhimThousandths(anisotropy);
  row[100 + channel]        = roundWhimThousandths((e1 + e2 + e3) / WhimReal(3));
  row[107 + channel] =
    roundWhimThousandths(total + first * second + first * third + second * third + first * second * third);

  const WhimReal logInverseAtoms = log(WhimReal(1) / numAtoms);
  WhimReal       gammaProduct    = WhimReal(1);
  for (int axis = 0; axis < 3; ++axis) {
    const WhimReal gamma = whimGamma(sums.symmetricCounts[axis], numAtoms, logInverseAtoms);
    gammaProduct *= gamma;
    row[channelStart + 5 + axis] = roundWhimThousandths(gamma);
  }
  if (channel < 2) {
    row[91 + channel] = roundWhimThousandths(pow(gammaProduct, WhimReal(1) / WhimReal(3)));
  }
}

/**
 * @brief Group-collective. All WHIM channels of one conformer's row.
 *
 * Each channel's covariance is reduced across the group, and lane c keeps channel c's. Every lane then
 * diagonalizes its own channel, so the group solves the channels concurrently rather than each lane
 * repeating all of them. Channel by channel, the owning lane broadcasts its principal axes for the
 * group-wide projection pass, and finally each lane writes its channel's values when @p writeRow.
 *
 * With @p measurePlane, returns PBF along the unweighted channel's axes (its PCA is PBF's in float64);
 * otherwise zero.
 */
template <typename OutputReal>
__device__ __forceinline__ WhimReal computeWhim(const ConformerAtoms&         atoms,
                                                const Property3DDeviceInputs& inputs,
                                                const int                     moleculeIdx,
                                                const int                     totalAtoms,
                                                const int                     laneInGroup,
                                                const unsigned                groupMask,
                                                const WhimThreshold&          threshold,
                                                const bool                    writeRow,
                                                OutputReal*                   row,
                                                const WhimScratch&            scratch,
                                                const bool                    measurePlane) {
  ProjectionState<WhimReal> state;
  computeProjectionCentroid(atoms, laneInGroup, state);
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    centeredCoordinates(atoms,
                        atomIdx,
                        state,
                        scratch.centered[atomIdx * 3 + 0],
                        scratch.centered[atomIdx * 3 + 1],
                        scratch.centered[atomIdx * 3 + 2]);
  }
  __syncwarp(groupMask);
  const int moleculeAtomStart = inputs.moleculeAtomStarts[moleculeIdx];

  WhimReal ownCovariance[6] = {};  // Zero for the lane without a channel; it diagonalizes trivially.
  for (int channel = 0; channel < kNumWhimChannels; ++channel) {
    const double* weights =
      channel == 0 ? nullptr :
                     inputs.atomPropertyWeights + static_cast<size_t>(channel - 1) * totalAtoms + moleculeAtomStart;
    WhimReal covariance[6];
    computeProjectionCovariance(atoms, weights, laneInGroup, state, covariance, scratch.centered);
    if (laneInGroup == channel) {
      for (int i = 0; i < 6; ++i) {
        ownCovariance[i] = covariance[i];
      }
    }
  }
  diagonalizeProjectionFast(ownCovariance, state);
  __syncwarp(groupMask);  // Channel eigensolvers can finish after different numbers of sweeps.

  WhimChannelSums ownSums{};
  WhimReal        planeDistance = 0;
  for (int channel = 0; channel < kNumWhimChannels; ++channel) {
    WhimReal axes[9];
    for (int i = 0; i < 9; ++i) {
      axes[i] = __shfl_sync(groupMask, state.eigenvectors[i], channel, kGroupSize);
    }
    if (channel == 0 && measurePlane) {
      planeDistance = computePbf(atoms, laneInGroup, axes, scratch.centered);
    }
    const WhimChannelSums sums = accumulateWhimChannel(atoms, laneInGroup, groupMask, axes, threshold, scratch);
    if (laneInGroup == channel) {
      ownSums = sums;
    }
  }
  if (writeRow && laneInGroup < kNumWhimChannels) {
    writeWhimChannel(laneInGroup, state, ownSums, atoms.numAtoms, row);
  }
  return planeDistance;
}

/**
 * @brief PBF and WHIM for conformers [@p conformerBegin, @p conformerEnd); one group of lanes per conformer.
 *
 * PBF uses an unweighted PCA in @p OutputReal, taken from WHIM's unweighted channel when both run in
 * float64. WHIM runs its own PCA in WhimReal for the unweighted channel and each atom-property channel,
 * whose weights are stored channel-major in `inputs.atomPropertyWeights` (`kNumAtomPropertyChannels`
 * blocks of one value per molecule atom), and uses @p whimScratch: one WhimScratch block of
 * whimScratchStride(@p whimMaxAtoms) atoms per conformer of the range. Rows longer than @p whimMaxAtoms are
 * treated as invalid. Outputs are @p OutputReal.
 */
template <typename OutputReal, bool kComputeWhim>
__global__ void projection3DKernel(const DeviceCoordView        coordinates,
                                   const Property3DDeviceInputs inputs,
                                   const int                    conformerBegin,
                                   const int                    conformerEnd,
                                   OutputReal* __restrict__ pbfOutput,
                                   OutputReal* __restrict__ whimOutput,
                                   int4* __restrict__ whimScratch,
                                   const int           whimMaxAtoms,
                                   const WhimThreshold whimThreshold) {
  const int lane        = static_cast<int>(threadIdx.x) % kWarpSize;
  const int laneInGroup = lane % kGroupSize;
  const int warpStart =
    conformerBegin + (blockIdx.x * kWarpsPerBlock + static_cast<int>(threadIdx.x) / kWarpSize) * kGroupsPerWarp;
  const int  conformerIdx = warpStart + lane / kGroupSize;
  // Lanes past the end keep no atoms but stay for the shuffles and PBF's block barriers.
  const bool inRange      = conformerIdx < conformerEnd;

  ConformerAtoms atoms =
    loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, inRange ? conformerIdx : coordinates.numConformers);
  if constexpr (kComputeWhim) {
    if (atoms.numAtoms > whimMaxAtoms) {
      atoms = ConformerAtoms{};
    }
  }

  // In float64, PBF's PCA is WHIM's unweighted channel; WHIM measures the plane distances along its axes.
  constexpr bool kPbfFromWhim = kComputeWhim && std::is_same_v<OutputReal, WhimReal>;
  const auto     writePbf     = [&](const OutputReal pbf) {
    if (inRange && laneInGroup == 0) {
      const bool is3D         = inputs.conformerIs3D == nullptr || inputs.conformerIs3D[conformerIdx] != 0;
      pbfOutput[conformerIdx] = !atoms.valid                ? static_cast<OutputReal>(nan("")) :
                                atoms.numAtoms < 4 || !is3D ? OutputReal(0) :
                                                              pbf;
    }
  };
  if (pbfOutput != nullptr && !kPbfFromWhim) {
    ProjectionState<OutputReal> pbfState;
    computeProjectionCentroid(atoms, laneInGroup, pbfState);
    OutputReal covariance[6];
    computeProjectionCovariance(atoms, nullptr, laneInGroup, pbfState, covariance);
    diagonalizePbfInBlock(covariance, laneInGroup, pbfState);
    writePbf(computePbf(atoms, laneInGroup, pbfState));
  }

  if constexpr (kComputeWhim) {
    const unsigned    groupMask = 0xffu << (lane - laneInGroup);
    const WhimScratch scratch =
      whimScratchFor(whimScratch, inRange ? conformerIdx - conformerBegin : 0, whimScratchStride(whimMaxAtoms));
    OutputReal*    row = inRange ? whimOutput + static_cast<size_t>(conformerIdx) * kNumWhimProperties : nullptr;
    const WhimReal planeDistance = computeWhim(atoms,
                                               inputs,
                                               atoms.valid ? coordinates.molIndices[conformerIdx] : 0,
                                               inputs.moleculeAtomStarts[coordinates.nMols],
                                               laneInGroup,
                                               groupMask,
                                               whimThreshold,
                                               inRange && atoms.valid,
                                               row,
                                               scratch,
                                               kPbfFromWhim && pbfOutput != nullptr);
    if (kPbfFromWhim && pbfOutput != nullptr) {
      writePbf(static_cast<OutputReal>(planeDistance));
    }
    if (inRange && !atoms.valid) {
      for (int valueIdx = laneInGroup; valueIdx < kNumWhimProperties; valueIdx += kGroupSize) {
        row[valueIdx] = static_cast<OutputReal>(nan(""));
      }
    }
  }
}

//! Upper bound on the WHIM symmetry-search scratch; larger batches run in conformer chunks.
constexpr size_t kWhimScratchBudgetBytes = size_t{256} << 20;

/**
 * @brief Launches the projection kernel when PBF or WHIM is requested. PBF computes in @p Real; WHIM's
 *        PCA computes in WhimReal. Both write @p Real outputs.
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
    projection3DKernel<Real, false><<<numBlocks, kBlockSize, 0, stream>>>(coordinates,
                                                                          inputs,
                                                                          0,
                                                                          numConformers,
                                                                          pbfOutput,
                                                                          nullptr,
                                                                          nullptr,
                                                                          0,
                                                                          WhimThreshold::make(whim.threshold));
    cudaCheckError(cudaGetLastError());
    return;
  }

  const int    maxAtoms          = std::max(inputs.maxMoleculeAtoms, 1);
  const size_t bytesPerConformer = kWhimScratchBytesPerAtom * static_cast<size_t>(whimScratchStride(maxAtoms));
  const size_t chunkSize =
    std::clamp<size_t>(kWhimScratchBudgetBytes / bytesPerConformer, 1, static_cast<size_t>(numConformers));
  AsyncDeviceVector<int4> scratch(chunkSize * bytesPerConformer / sizeof(int4), stream);
  for (int begin = 0; begin < numConformers; begin += static_cast<int>(chunkSize)) {
    const int end       = std::min(numConformers, begin + static_cast<int>(chunkSize));
    const int numBlocks = (end - begin + kConformersPerBlock - 1) / kConformersPerBlock;
    projection3DKernel<Real, true><<<numBlocks, kBlockSize, 0, stream>>>(coordinates,
                                                                         inputs,
                                                                         begin,
                                                                         end,
                                                                         pbfOutput,
                                                                         whimOutput,
                                                                         scratch.data(),
                                                                         maxAtoms,
                                                                         WhimThreshold::make(whim.threshold));
    cudaCheckError(cudaGetLastError());
  }
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_PROJECTION_CUH
