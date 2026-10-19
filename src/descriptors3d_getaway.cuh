// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_GETAWAY_CUH
#define NVMOLKIT_DESCRIPTORS3D_GETAWAY_CUH

#include <cmath>
#include <cstdint>

#include "src/descriptors3d.h"
#include "src/descriptors3d_kernel.cuh"
#include "src/descriptors3d_projection.cuh"
#include "src/descriptors3d_topology.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"
#include "src/utils/symmetric_eigenvalues_3x3.cuh"

namespace nvMolKit::descriptors3d_detail {

//! Unweighted channel plus the atom-property channels, in RDKit's output order (u, m, v, e, p, i, s).
constexpr int kNumGetawayChannels = kNumAtomPropertyChannels + 1;
//! Topological lags with their own H, HATS and R values (1-8); farther or disconnected pairs only enter the
//! totals. One lane of each kGroupSize-lane group owns each lag.
constexpr int kNumGetawayLags     = 8;
static_assert(kNumGetawayLags == kGroupSize);
//! Pair streams per warp: each kGroupSize-lane group walks interleaved rows of the atom-pair triangle.
constexpr int kGetawayPairStreams = kWarpSize / kGroupSize;

//! Values per channel in the H/HATS block: H0..H8, HT, HATS0..HATS8, HATST.
constexpr int kGetawayHBlock    = 2 * (kNumGetawayLags + 2);
//! Values per channel in the R block: R1..R8, RT, R1+..R8+, RT+.
constexpr int kGetawayRBlock    = 2 * (kNumGetawayLags + 1);
constexpr int kGetawayHStart    = 4;
constexpr int kGetawayRconIndex = kGetawayHStart + kNumGetawayChannels * kGetawayHBlock;
constexpr int kGetawayRStart    = kGetawayRconIndex + 3;
static_assert(kGetawayRStart + kNumGetawayChannels * kGetawayRBlock == kNumGetawayProperties);

//! Singular values of X^T X at or below this are dropped from its pseudo-inverse, as in RDKit's GetPinv.
constexpr double kGetawayPinvTolerance = 1e-3;
//! PBF below this makes RDKit treat the molecule as two-dimensional for HIC.
constexpr double kGetawayPlanarPbf     = 1e-5;
//! Upper bound on power-iteration steps for REIG.
constexpr int    kGetawayMaxIterations = 1000;

/**
 * @brief Per-atom scratch row, written once per conformer as the compute type: scaled projections y (three
 *        values, H(j, k) = y_j . y_k), leverage h = H(j, j), sqrt(h), the atom-property channels, then the
 *        centered coordinates.
 */
constexpr int kGetawayRowY        = 0;
constexpr int kGetawayRowLeverage = 3;
constexpr int kGetawayRowRootLev  = 4;
constexpr int kGetawayRowWeights  = 5;
constexpr int kGetawayRowCentered = kGetawayRowWeights + kNumAtomPropertyChannels;
constexpr int kGetawayRowStride   = kGetawayRowCentered + 3;

// ITH and ISH group the heavy-atom leverages by their decimal digits, not their numeric values: RDKit rounds
// each leverage to `precision` significant digits (round_to_n_digits_), prints it with C++ stream output, and
// its IsClose2 compares substrings of that printed text. Reproducing RDKit's groups exactly therefore means
// reproducing that rounding and that text, which the helpers below do for the values leverages can take.

//! Exact powers of ten representable in float64.
__device__ __forceinline__ double exactPowerOfTen(const int exponent) {
  constexpr double kPowers[] = {1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
                                1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};
  return kPowers[exponent];
}

//! @p value * 10^@p exponent; exact scaling factors within float64's exact powers of ten.
__device__ __forceinline__ double scaleByPowerOfTen(double value, int exponent) {
  while (exponent > 22) {
    value *= 1e22;
    exponent -= 22;
  }
  while (exponent < -22) {
    value /= 1e22;
    exponent += 22;
  }
  return exponent >= 0 ? value * exactPowerOfTen(exponent) : value / exactPowerOfTen(-exponent);
}

/**
 * @brief RDKit's round_to_n_digits_: `atof(sprintf("%.*g", digits, value))` for a non-negative @p value.
 *
 * Also returns the rounded value's significand (@p mantissa, exactly @p digits digits unless the value is 0)
 * and decimal exponent, from which rdkitDecimalText() rebuilds its printed text. Exact ties round to even, as glibc
 * does for exactly representable halves; values within float64 rounding of a tie may differ from printf.
 */
__device__ __forceinline__ double roundToSignificantDigits(const double value,
                                                           const int    digits,
                                                           int64_t&     mantissa,
                                                           int&         exponent) {
  mantissa = 0;
  exponent = 0;
  if (!(value > 0.0) || !isfinite(value)) {
    return value;  // 0 stays 0; NaN and infinity stay as they are.
  }
  exponent             = static_cast<int>(floor(log10(value)));
  const double lowest  = exactPowerOfTen(digits - 1);
  const double highest = exactPowerOfTen(digits);
  double       scaled  = scaleByPowerOfTen(value, digits - 1 - exponent);
  if (scaled < lowest) {
    --exponent;
    scaled = scaleByPowerOfTen(value, digits - 1 - exponent);
  } else if (scaled >= highest) {
    ++exponent;
    scaled = scaleByPowerOfTen(value, digits - 1 - exponent);
  }
  mantissa = llrint(scaled);
  if (mantissa == static_cast<int64_t>(highest)) {
    mantissa /= 10;
    ++exponent;
  }
  return scaleByPowerOfTen(static_cast<double>(mantissa), exponent - (digits - 1));
}

//! Decimal text of a printed value, as RDKit's IsClose2 compares it.
struct DecimalText {
  char chars[24];
  int  length;
};

__device__ __forceinline__ void appendChar(DecimalText& text, const char character) {
  text.chars[text.length++] = character;
}

/**
 * @brief The decimal text RDKit's IsClose2 compares for the rounded leverage
 *        v = @p mantissa * 10^(@p exponent - @p digits + 1) from roundToSignificantDigits().
 *
 * IsClose2 prints with default C++ stream formatting: up to 6 significant digits with trailing zeros removed,
 * fixed notation for decimal exponents -4 to 5 (e.g. "0.034", "1") and scientific notation otherwise
 * (e.g. "1.8e-06"). With @p digits <= 6 every significant digit of v is printed.
 */
__device__ __forceinline__ void rdkitDecimalText(const int64_t mantissa,
                                                 const int     exponent,
                                                 const int     digits,
                                                 DecimalText&  text) {
  text.length = 0;
  if (mantissa == 0) {
    appendChar(text, '0');
    return;
  }
  char    significand[8];
  int     significandLength = 0;
  int64_t remainder         = mantissa;
  for (int position = digits - 1; position >= 0; --position) {
    significand[position] = static_cast<char>('0' + remainder % 10);
    remainder /= 10;
  }
  significandLength = digits;
  while (significandLength > 1 && significand[significandLength - 1] == '0') {
    --significandLength;
  }
  if (exponent < -4 || exponent >= 6) {
    appendChar(text, significand[0]);
    if (significandLength > 1) {
      appendChar(text, '.');
      for (int position = 1; position < significandLength; ++position) {
        appendChar(text, significand[position]);
      }
    }
    appendChar(text, 'e');
    appendChar(text, exponent < 0 ? '-' : '+');
    const int magnitude = exponent < 0 ? -exponent : exponent;
    if (magnitude >= 100) {
      appendChar(text, static_cast<char>('0' + magnitude / 100));
    }
    appendChar(text, static_cast<char>('0' + (magnitude / 10) % 10));
    appendChar(text, static_cast<char>('0' + magnitude % 10));
    return;
  }
  if (exponent >= 0) {
    for (int position = 0; position <= exponent; ++position) {
      appendChar(text, position < significandLength ? significand[position] : '0');
    }
    if (significandLength > exponent + 1) {
      appendChar(text, '.');
      for (int position = exponent + 1; position < significandLength; ++position) {
        appendChar(text, significand[position]);
      }
    }
    return;
  }
  appendChar(text, '0');
  appendChar(text, '.');
  for (int zero = 0; zero < -exponent - 1; ++zero) {
    appendChar(text, '0');
  }
  for (int position = 0; position < significandLength; ++position) {
    appendChar(text, significand[position]);
  }
}

//! `text.substr(text.find('.') + 1)`: the text after the first '.', or all of it when there is none.
__device__ __forceinline__ DecimalText textAfterPoint(const DecimalText& text) {
  int start = 0;
  for (int position = 0; position < text.length; ++position) {
    if (text.chars[position] == '.') {
      start = position + 1;
      break;
    }
  }
  DecimalText suffix;
  suffix.length = 0;
  for (int position = start; position < text.length; ++position) {
    appendChar(suffix, text.chars[position]);
  }
  return suffix;
}

//! RDKit's countZeros: leading '0' characters.
__device__ __forceinline__ int countLeadingZeros(const DecimalText& text) {
  int zeros = 0;
  while (zeros < text.length && text.chars[zeros] == '0') {
    ++zeros;
  }
  return zeros;
}

//! C atoi on the text: leading decimal digits, 0 if there are none.
__device__ __forceinline__ int64_t leadingInteger(const DecimalText& text) {
  int64_t value = 0;
  for (int position = 0; position < text.length && text.chars[position] >= '0' && text.chars[position] <= '9';
       ++position) {
    value = value * 10 + (text.chars[position] - '0');
  }
  return value;
}

//! What RDKit's IsClose2 reads from a rounded leverage's printed fraction text (textAfterPoint()): its
//! leading '0' count, its length, its leading integer (C atoi) and whether it is all digits, in which case
//! appending '0' multiplies that integer by 10.
struct ClusterKey {
  int     zeros;
  int     length;
  int64_t leading;
  bool    allDigits;
};

//! Packs a ClusterKey into 64 bits: leading integer (below 2^40) in the low bits, then length, zeros and the
//! all-digits flag.
__device__ __forceinline__ int64_t packClusterKey(const ClusterKey& key) {
  return key.leading | (int64_t{key.length} << 40) | (int64_t{key.zeros} << 48) |
         (int64_t{key.allDigits ? 1 : 0} << 56);
}

__device__ __forceinline__ ClusterKey unpackClusterKey(const int64_t packed) {
  return {static_cast<int>((packed >> 48) & 0xff),
          static_cast<int>((packed >> 40) & 0xff),
          packed & ((int64_t{1} << 40) - 1),
          ((packed >> 56) & 1) != 0};
}

//! Rounds @p leverage to @p digits significant digits (into @p rounded) and returns the ClusterKey of its
//! printed text.
__device__ __forceinline__ ClusterKey clusterKey(const double leverage, const int digits, double& rounded) {
  int64_t mantissa;
  int     exponent;
  rounded = roundToSignificantDigits(leverage, digits, mantissa, exponent);
  DecimalText printed;
  rdkitDecimalText(mantissa, exponent, digits, printed);
  const DecimalText fraction = textAfterPoint(printed);
  ClusterKey        key{countLeadingZeros(fraction), fraction.length, leadingInteger(fraction), true};
  for (int position = 0; position < fraction.length; ++position) {
    key.allDigits = key.allDigits && fraction.chars[position] >= '0' && fraction.chars[position] <= '9';
  }
  return key;
}

//! The effect of appending '0' to the text a ClusterKey describes.
__device__ __forceinline__ void appendZero(ClusterKey& key) {
  ++key.length;
  if (key.allDigits) {
    key.leading *= 10;
  }
}

/**
 * @brief RDKit's IsClose2 on the keys of two printed values: equal leading-zero counts pad the shorter text
 *        once with '0'; texts of equal length are close when their leading integers differ by less than 2.
 */
__device__ __forceinline__ bool rdkitIsClose(ClusterKey a, ClusterKey b) {
  if (a.zeros == b.zeros) {
    if (a.length > b.length) {
      appendZero(b);
    }
    if (a.length < b.length) {
      appendZero(a);
    }
  }
  if (a.length != b.length) {
    return false;
  }
  const int64_t difference = a.leading - b.leading;
  return difference > -2 && difference < 2;
}

/**
 * @brief Warp-collective. ITH and ISH from the heavy-atom leverages, porting RDKit's getGETAWAYDesc: each
 *        leverage rounded to @p digits significant digits, sorted in descending order and grouped by RDKit's
 *        clusterArray2 (a greedy scan with IsClose2 comparisons of the printed values), then
 *        ITH = n log2 n - sum(c log2 c) over the group sizes c and ISH = ITH / (n log2 n).
 *
 * The lanes round, key and rank the heavy atoms (equal values print identically, so their relative order
 * does not matter); lane 0 then runs the greedy scan on the sorted keys and alone receives @p ith and @p ish.
 * Both are NaN when any heavy-atom leverage is not finite. @p heavyValues and @p unsortedKeys hold one entry and @p
 * sortedKeys one entry per heavy atom.
 */
__device__ __forceinline__ void computeInformationIndices(const double*  leverages,
                                                          const uint8_t* heavyAtomFlags,
                                                          const int      numAtoms,
                                                          const int      digits,
                                                          const int      lane,
                                                          double*        heavyValues,
                                                          int64_t*       unsortedKeys,
                                                          int64_t*       sortedKeys,
                                                          double&        ith,
                                                          double&        ish) {
  const auto warp      = laneTile<kWarpSize>();
  int        numHeavy  = 0;
  bool       nonFinite = false;
  for (int base = 0; base < numAtoms; base += kWarpSize) {
    const int      atomIdx = base + lane;
    const bool     isHeavy = atomIdx < numAtoms && heavyAtomFlags[atomIdx] != 0;
    const unsigned heavy   = warp.ballot(isHeavy);
    nonFinite              = nonFinite || (isHeavy && !isfinite(leverages[atomIdx]));
    if (isHeavy) {
      const int  slot = numHeavy + __popc(heavy & ((1u << lane) - 1u));
      double     rounded;
      const auto key     = clusterKey(leverages[atomIdx], digits, rounded);
      heavyValues[slot]  = rounded;
      unsortedKeys[slot] = packClusterKey(key);
    }
    numHeavy += __popc(heavy);
  }
  // Non-finite leverages (from non-finite coordinates) have no rank or printed digits to cluster.
  if (warp.any(nonFinite)) {
    ith = nan("");
    ish = nan("");
    return;
  }
  warp.sync();
  for (int slot = lane; slot < numHeavy; slot += kWarpSize) {
    const double value = heavyValues[slot];
    int          rank  = 0;
    for (int other = 0; other < numHeavy; ++other) {
      const double otherValue = heavyValues[other];
      rank += otherValue > value || (otherValue == value && other < slot) ? 1 : 0;
    }
    sortedKeys[rank] = unsortedKeys[slot];
  }
  warp.sync();
  if (lane != 0) {
    return;
  }
  const double heavyCount = numHeavy;
  const double total0     = heavyCount * log(heavyCount) / log(2.0);
  double       total      = total0;
  auto         store      = [&](const int count) {
    const double size = count;
    total -= size * log(size) / log(2.0);
  };
  int head  = 0;
  int count = 0;
  while (head < numHeavy) {
    const ClusterKey front = unpackClusterKey(sortedKeys[head]);
    ++head;
    ++count;
    const int remaining = numHeavy - head;
    if (remaining == 0) {
      store(count);
    }
    for (int offset = 0; offset < remaining; ++offset) {
      if (rdkitIsClose(front, unpackClusterKey(sortedKeys[head + offset]))) {
        ++count;
      } else {
        store(count);
        head += offset;
        count = 0;
        break;
      }
      if (offset == remaining - 1) {
        store(count);
        ++head;
        count = 0;
      }
    }
  }
  ith = total;
  ish = total / total0;
}

/**
 * @brief Warp-collective. Leverages of the conformer's atoms: the diagonal of H = X (X^T X)^+ X^T for the
 *        centered coordinates X, with RDKit's pseudo-inverse (singular values at or below
 *        kGetawayPinvTolerance dropped). Writes each atom's scratch row and float64 leverage.
 *
 * FP64 required: leverages decide ITH and ISH through their rounding to `precision` significant digits and
 * RDKit's digit-string clustering, so float32 leverages would move values across rounding boundaries; the
 * pseudo-inverse cutoff decides which axes count for (near-)planar conformers. This is O(atoms) work.
 */
template <typename Real>
__device__ __forceinline__ void computeLeverages(const ConformerAtoms& atoms,
                                                 const double*         propertyWeights,
                                                 const int             totalAtoms,
                                                 const int             lane,
                                                 Real*                 rows,
                                                 double*               leverages) {
  double sumX = 0;
  double sumY = 0;
  double sumZ = 0;
  for (int atomIdx = lane; atomIdx < atoms.numAtoms; atomIdx += kWarpSize) {
    sumX += atoms.positions[atomIdx * 3 + 0];
    sumY += atoms.positions[atomIdx * 3 + 1];
    sumZ += atoms.positions[atomIdx * 3 + 2];
  }
  const double inverseAtoms = 1.0 / static_cast<double>(atoms.numAtoms);
  const double centroidX    = groupAllReduceSum<kWarpSize>(sumX) * inverseAtoms;
  const double centroidY    = groupAllReduceSum<kWarpSize>(sumY) * inverseAtoms;
  const double centroidZ    = groupAllReduceSum<kWarpSize>(sumZ) * inverseAtoms;

  double moments[6] = {};
  for (int atomIdx = lane; atomIdx < atoms.numAtoms; atomIdx += kWarpSize) {
    const double x = atoms.positions[atomIdx * 3 + 0] - centroidX;
    const double y = atoms.positions[atomIdx * 3 + 1] - centroidY;
    const double z = atoms.positions[atomIdx * 3 + 2] - centroidZ;
    moments[0] += x * x;
    moments[1] += x * y;
    moments[2] += x * z;
    moments[3] += y * y;
    moments[4] += y * z;
    moments[5] += z * z;
  }
  for (double& moment : moments) {
    moment = groupAllReduceSum<kWarpSize>(moment);
  }
  double eigenvalues[3];
  double eigenvectors[9];
  symmetricEigensystemJacobi3x3(moments[0],
                                moments[1],
                                moments[2],
                                moments[3],
                                moments[4],
                                moments[5],
                                eigenvalues,
                                eigenvectors);
  double inverseRoots[3];
  for (int axis = 0; axis < 3; ++axis) {
    const double singular = fabs(eigenvalues[axis]);
    inverseRoots[axis]    = singular > kGetawayPinvTolerance ? 1.0 / sqrt(singular) : 0.0;
  }

  for (int atomIdx = lane; atomIdx < atoms.numAtoms; atomIdx += kWarpSize) {
    const double x        = atoms.positions[atomIdx * 3 + 0] - centroidX;
    const double y        = atoms.positions[atomIdx * 3 + 1] - centroidY;
    const double z        = atoms.positions[atomIdx * 3 + 2] - centroidZ;
    Real* const  row      = rows + atomIdx * kGetawayRowStride;
    double       leverage = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const double scaled =
        (x * eigenvectors[axis] + y * eigenvectors[3 + axis] + z * eigenvectors[6 + axis]) * inverseRoots[axis];
      row[kGetawayRowY + axis] = static_cast<Real>(scaled);
      leverage += scaled * scaled;
    }
    leverages[atomIdx]       = leverage;
    row[kGetawayRowLeverage] = static_cast<Real>(leverage);
    row[kGetawayRowRootLev]  = static_cast<Real>(sqrt(leverage));
    for (int channel = 0; channel < kNumAtomPropertyChannels; ++channel) {
      row[kGetawayRowWeights + channel] =
        static_cast<Real>(propertyWeights[static_cast<size_t>(channel) * totalAtoms + atomIdx]);
    }
    row[kGetawayRowCentered + 0] = static_cast<Real>(x);
    row[kGetawayRowCentered + 1] = static_cast<Real>(y);
    row[kGetawayRowCentered + 2] = static_cast<Real>(z);
  }
  __syncwarp();
}

//! Atom-property weight of channel @p channel (0 is the unweighted channel) from a scratch row.
template <typename Real> __device__ __forceinline__ Real getawayWeight(const Real* row, const int channel) {
  return channel == 0 ? Real(1) : row[kGetawayRowWeights + channel - 1];
}

//! Distance between two atoms from their scratch rows' centered coordinates.
template <typename Real> __device__ __forceinline__ Real getawayDistance(const Real* first, const Real* second) {
  const Real dx = first[kGetawayRowCentered + 0] - second[kGetawayRowCentered + 0];
  const Real dy = first[kGetawayRowCentered + 1] - second[kGetawayRowCentered + 1];
  const Real dz = first[kGetawayRowCentered + 2] - second[kGetawayRowCentered + 2];
  return sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * @brief Warp-collective. (R v)_i = sqrt(h_i) * sum_{k != i} sqrt(h_k) v_k / r_ik for every atom i into
 *        @p product; returns v^T R v and |R v|^2, reduced over the warp. Rows are computed on the fly.
 */
template <typename Real>
__device__ __forceinline__ void multiplyByInfluenceMatrix(const Real* rows,
                                                          const int   numAtoms,
                                                          const int   lane,
                                                          const Real* vector,
                                                          Real*       product,
                                                          Real&       rayleighNumerator,
                                                          Real&       productNormSquared) {
  Real numerator = 0;
  Real norm      = 0;
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    const Real* rowI = rows + atomIdx * kGetawayRowStride;
    Real        sum  = 0;
    for (int other = 0; other < numAtoms; ++other) {
      if (other != atomIdx) {
        const Real* rowK = rows + other * kGetawayRowStride;
        sum += rowK[kGetawayRowRootLev] * vector[other] / getawayDistance(rowI, rowK);
      }
    }
    const Real value = rowI[kGetawayRowRootLev] * sum;
    product[atomIdx] = value;
    numerator += vector[atomIdx] * value;
    norm += value * value;
  }
  rayleighNumerator  = groupAllReduceSum<kWarpSize>(numerator);
  productNormSquared = groupAllReduceSum<kWarpSize>(norm);
  __syncwarp();
}

// ---- Fast path: one CTA per conformer ------------------------------------------------------------------------
//
// The CTA keeps the conformer's atom rows (structure of arrays) and R = sqrt(h_j h_k) / r_jk in shared memory.
// R is stored once per unordered pair in a circulant layout: slot (d, i) holds R(i, (i + d) mod n) for
// d = 1..n/2, so every row of the power-iteration matvec reads consecutive slots and consecutive vector
// entries (conflict-free) while storing only n * floor(n / 2) values. Pair totals run over those slots with
// every thread busy; the lag 1-8 sums run over per-molecule lag lists (getawayMoleculePrepassKernel()), so no
// lane scans pairs it does not own. Conformers whose molecule has no bond-distance table or does not fit the
// shared-memory capacity fall back to getaway3DKernel().

constexpr int kGetawayFastBlockSize   = 160;  // four power-iteration warps plus one for ITH
//! Molecules of up to kGetawaySmallMaxAtoms atoms run in CTAs of kGetawaySmallBlockSize threads.
constexpr int kGetawaySmallBlockSize  = 64;
constexpr int kGetawaySmallMaxAtoms   = 64;
//! Molecules of up to kGetawayWarpMaxAtoms atoms run one warp per conformer (getawayWarpKernel()), each lane owning
//! an atom and half of its row of R in registers; larger molecules use CTAs.
constexpr int kGetawayWarpMaxAtoms    = 24;
constexpr int kGetawayPrepassBlock    = 256;
//! Per-molecule lag-list starts: lag l's pairs are [starts[l - 1], starts[l]) for l = 1..kNumGetawayLags.
constexpr int kGetawayLagStartsStride = kNumGetawayLags + 1;

//! Structure-of-arrays atom rows of the fast path, each numAtoms long.
constexpr int kGetawayFastY         = 0;  // three arrays
constexpr int kGetawayFastLeverage  = 3;
constexpr int kGetawayFastRootLev   = 4;
constexpr int kGetawayFastWeights   = 5;  // kNumAtomPropertyChannels arrays
constexpr int kGetawayFastCentered  = kGetawayFastWeights + kNumAtomPropertyChannels;  // three arrays
constexpr int kGetawayFastRowArrays = kGetawayFastCentered + 3;

//! Sums reduced together after the pair loop: H, HATS and R totals, then lag-0 H and HATS, per channel.
constexpr int kGetawayFastTotals    = 5 * kNumGetawayChannels;
//! Most warps a fast-path CTA may have.
constexpr int kGetawayMaxFastWarps  = kGetawayFastBlockSize / kWarpSize;
//! Broadcast float64 scalars in the reduction scratch: HIC's own dimension, then the leverage projection.
constexpr int kGetawayFastScalars   = 16;
//! Atom-row arrays the fast path reserves: the per-atom rows, then one more so the power-iteration iterates (4n
//! values) fit over the centered coordinates, which nothing reads once R's slots hold the inverse distances.
constexpr int kGetawayFastRowRegion = kGetawayFastCentered + 4;

__host__ __device__ inline size_t getawayAlign16(const size_t bytes) {
  return (bytes + 15) & ~static_cast<size_t>(15);
}

//! Byte offsets within the reduction scratch of a fast-path CTA of @p warps warps.
struct GetawayFastReduce {
  size_t
    power;      //!< two parities * warps * 4 power-iteration partials; before it, (warps + 1) * kGetawayFastTotals sums
  size_t rMax;  //!< kNumGetawayLags * kNumGetawayChannels lag maxima
  size_t scalars;  //!< kGetawayFastScalars broadcast values
  size_t total;
};

__host__ __device__ inline GetawayFastReduce getawayFastReduce(const int warps) {
  GetawayFastReduce reduce;
  reduce.power   = getawayAlign16(static_cast<size_t>(warps + 1) * kGetawayFastTotals * sizeof(double));
  reduce.rMax    = getawayAlign16(reduce.power + static_cast<size_t>(2 * warps * 4) * sizeof(double));
  reduce.scalars = getawayAlign16(reduce.rMax + kNumGetawayLags * kNumGetawayChannels * sizeof(double));
  reduce.total   = getawayAlign16(reduce.scalars + kGetawayFastScalars * sizeof(double));
  return reduce;
}

//! Byte offsets of the fast kernel's shared-memory sections for molecules of up to @p capacity atoms, in CTAs of
//! @p warps warps.
struct GetawayFastLayout {
  size_t leverages;
  size_t rows;
  size_t matrix;
  size_t heavyValues;
  size_t keys;
  size_t reduce;
  size_t total;
};

template <typename Real>
__host__ __device__ inline GetawayFastLayout getawayFastLayout(const int capacity, const int warps) {
  GetawayFastLayout layout;
  const size_t      atoms = static_cast<size_t>(capacity);
  size_t            bytes = 0;
  layout.leverages        = bytes;
  bytes                   = getawayAlign16(bytes + sizeof(double) * atoms);
  layout.rows             = bytes;
  bytes                   = getawayAlign16(bytes + sizeof(Real) * kGetawayFastRowRegion * atoms);
  layout.matrix           = bytes;
  bytes                   = getawayAlign16(bytes + sizeof(Real) * (atoms / 2) * atoms);
  layout.heavyValues      = bytes;
  bytes                   = getawayAlign16(bytes + sizeof(double) * atoms);
  layout.keys             = bytes;
  bytes                   = getawayAlign16(bytes + sizeof(int64_t) * 2 * atoms);
  layout.reduce           = bytes;
  bytes                   = getawayAlign16(bytes + getawayFastReduce(warps).total);
  layout.total            = bytes;
  return layout;
}

//! Whether getawayFastKernel() computes conformer @p conformerIdx (invalid conformers included).
__device__ __forceinline__ bool getawayFastPathOwns(const DeviceCoordView&        coordinates,
                                                    const Property3DDeviceInputs& inputs,
                                                    const BondDistanceTable&      bondTable,
                                                    const int                     fastCapacity,
                                                    const int                     conformerIdx) {
  if (fastCapacity <= 0) {
    return false;
  }
  const ConformerAtoms atoms = loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, conformerIdx);
  if (!atoms.valid) {
    return true;
  }
  return atoms.numAtoms <= fastCapacity &&
         moleculeBondDistances(bondTable, coordinates.molIndices[conformerIdx]) != nullptr;
}

//! Barrier 1 over the first @p threads threads of the CTA (a multiple of kWarpSize).
__device__ __forceinline__ void getawayWarpsBarrier(const int threads) {
  asm volatile("bar.sync 1, %0;" ::"r"(threads) : "memory");
}

//! HIC's dimension from RDKit's PBF of @p defaultAtoms (group-collective over kGroupSize lanes).
__device__ __forceinline__ double getawayHicDimension(const ConformerAtoms& defaultAtoms,
                                                      const bool            is3D,
                                                      const int             laneInGroup) {
  ProjectionState<double> plane;
  computeProjectionCentroid(defaultAtoms, laneInGroup, plane);
  computeProjectionPca(defaultAtoms, nullptr, laneInGroup, plane);
  const double pbf = computePbf(defaultAtoms, laneInGroup, plane);
  return (defaultAtoms.numAtoms < 4 || !is3D ? 0.0 : pbf) < kGetawayPlanarPbf ? 2.0 : 3.0;
}

// ---- Work lists: the prepass routes every fast-path conformer to the kernel that computes it -----------------
//
// Each fast-path kernel walks only its own bucket's list (CTAs, or warps for getawayWarpKernel(), striding over
// the entries), so no CTA spends time on a conformer another kernel computes. The CTA buckets are split into size
// classes, each with its own list: the kernels take the classes in order, largest molecules first, so the longest
// conformers start first and the shortest fill the tail.
constexpr int kGetawaySizeClasses = 8;
constexpr int kGetawayLargeBucket = 0;  //!< getawayFastKernel() with largeCapacity: kGetawaySizeClasses lists
constexpr int kGetawaySmallBucket = kGetawaySizeClasses;      //!< getawayFastKernel() with smallCapacity: as many lists
constexpr int kGetawayWarpBucket  = 2 * kGetawaySizeClasses;  //!< getawayWarpKernel(), including invalid conformers
constexpr int kGetawayBuckets     = kGetawayWarpBucket + 1;   //!< lists in all

struct GetawayWorkLists {
  int* counts          = nullptr;  //!< kGetawayBuckets list lengths (zeroed before the prepass)
  int* conformers      = nullptr;  //!< list b at conformers + b * stride
  int  stride          = 0;
  int  smallCapacity   = 0;  //!< conformers of kGetawayWarpMaxAtoms < n <= smallCapacity take the small bucket
  int  largeCapacity   = 0;  //!< conformers of smallCapacity < n <= largeCapacity atoms take the large bucket
  int  smallClassWidth = 1;  //!< atoms per size class of the small bucket, counted down from smallCapacity
  int  largeClassWidth = 1;  //!< atoms per size class of the large bucket, counted down from largeCapacity
};

//! Size class of a molecule @p gap atoms below its bucket's capacity, with @p width atoms per class.
__device__ __forceinline__ int getawaySizeClass(const int gap, const int width) {
  const int sizeClass = gap / width;
  return sizeClass < kGetawaySizeClasses - 1 ? sizeClass : kGetawaySizeClasses - 1;
}

//! List (bucket and size class) of conformer @p conformerIdx, or -1 when getaway3DKernel() computes it.
__device__ __forceinline__ int getawayBucket(const DeviceCoordView&        coordinates,
                                             const Property3DDeviceInputs& inputs,
                                             const BondDistanceTable&      bondTable,
                                             const GetawayWorkLists&       lists,
                                             const int                     conformerIdx) {
  const ConformerAtoms atoms = loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, conformerIdx);
  if (!atoms.valid) {
    return kGetawayWarpBucket;
  }
  if (moleculeBondDistances(bondTable, coordinates.molIndices[conformerIdx]) == nullptr) {
    return -1;
  }
  const int n = atoms.numAtoms;
  if (n <= kGetawayWarpMaxAtoms) {
    return kGetawayWarpBucket;
  }
  if (n <= lists.smallCapacity) {
    return kGetawaySmallBucket + getawaySizeClass(lists.smallCapacity - n, lists.smallClassWidth);
  }
  return n <= lists.largeCapacity ?
           kGetawayLargeBucket + getawaySizeClass(lists.largeCapacity - n, lists.largeClassWidth) :
           -1;
}

constexpr int kGetawayListThreads   = kGetawayPrepassBlock - kWarpSize;
//! Bond-distance bytes the prepass stages in shared memory: every pair of a molecule of up to 128 atoms (about
//! 8 KB, enough for drug-like molecules with explicit hydrogens). Larger molecules read the table directly.
constexpr int kGetawayPrepassStaged = 128 * 127 / 2;

/**
 * @brief Per molecule (one CTA each): HIC's dimension from the default conformer (0 when it is not valid, in
 *        which case each conformer uses its own), and the molecule's pairs with bond distance 1-8 grouped by
 *        distance into @p lagPairs (packed j | k << 16, j < k) from the molecule's table offset. The last warp
 *        computes the dimension while the other warps build the lists (barrier 1 among them only); list thread t
 *        takes a contiguous range of table entries, so each lag list comes out in pair order. Blocks past the
 *        molecules route one conformer per thread to its work list.
 */

__global__ void __launch_bounds__(kGetawayPrepassBlock)
  getawayMoleculePrepassKernel(const DeviceCoordView        coordinates,
                               const Property3DDeviceInputs inputs,
                               const BondDistanceTable      bondTable,
                               const GetawayWorkLists       lists,
                               uint32_t* __restrict__ lagPairs,
                               int32_t* __restrict__ lagStarts,
                               double* __restrict__ dimensions) {
  __shared__ int     counts[kNumGetawayLags][kGetawayListThreads];
  __shared__ int     lagOffsets[kNumGetawayLags + 1];
  __shared__ uint8_t stagedLags[kGetawayPrepassStaged];
  const int          moleculeIdx = blockIdx.x;
  const int          tid         = static_cast<int>(threadIdx.x);
  const int          lane        = tid % kWarpSize;
  const int          warp        = tid / kWarpSize;
  if (moleculeIdx >= coordinates.nMols) {
    // Routing blocks: one conformer per thread, appended to its bucket's list (warp-aggregated).
    const int      conformerIdx = (moleculeIdx - coordinates.nMols) * kGetawayPrepassBlock + tid;
    const int      bucket       = conformerIdx < coordinates.numConformers ?
                                    getawayBucket(coordinates, inputs, bondTable, lists, conformerIdx) :
                                    -1;
    // Lanes of one list share a leader, and every leader's atomic issues in the same instruction, so the warp
    // waits for one round trip however many lists it touches.
    const unsigned members      = __match_any_sync(0xffffffffu, bucket);
    const int      leader       = __ffs(static_cast<int>(members)) - 1;
    int            base         = 0;
    if (bucket >= 0 && lane == leader) {
      base = atomicAdd(lists.counts + bucket, __popc(members));
    }
    base = __shfl_sync(0xffffffffu, base, leader);
    if (bucket >= 0) {
      lists.conformers[bucket * lists.stride + base + __popc(members & ((1u << lane) - 1u))] = conformerIdx;
    }
    return;
  }
  if (tid >= kGetawayListThreads) {
    const int      defaultRow = inputs.defaultConformerRows != nullptr ? inputs.defaultConformerRows[moleculeIdx] : -1;
    ConformerAtoms defaultAtoms =
      defaultRow >= 0 ? loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, defaultRow) : ConformerAtoms{};
    double dimension = 0.0;
    if (defaultAtoms.valid) {
      const bool is3D = inputs.conformerIs3D == nullptr || inputs.conformerIs3D[defaultRow] != 0;
      dimension       = getawayHicDimension(defaultAtoms, is3D, lane % kGroupSize);
    }
    if (lane == 0) {
      dimensions[moleculeIdx] = dimension;
    }
    return;
  }
  const uint8_t* table = moleculeBondDistances(bondTable, moleculeIdx);
  if (table == nullptr) {
    return;  // Uniform across the list threads; this molecule's conformers take the fallback kernel.
  }
  const int      numAtoms = inputs.moleculeAtomStarts[moleculeIdx + 1] - inputs.moleculeAtomStarts[moleculeIdx];
  const int64_t  numPairs = static_cast<int64_t>(numAtoms) * (numAtoms - 1) / 2;
  // Stage the table bytes in shared memory with independent loads when they fit; larger molecules read them
  // from global memory directly.
  const bool     staged   = numPairs <= kGetawayPrepassStaged;
  const uint8_t* lags     = table;
  if (staged) {
#pragma unroll 8
    for (int pair = tid; pair < numPairs; pair += kGetawayListThreads) {
      stagedLags[pair] = table[pair];
    }
    getawayWarpsBarrier(kGetawayListThreads);
    lags = stagedLags;
  }
  // Thread t owns the contiguous pair range [begin, end), so each lag list comes out in pair order.
  const int64_t perThread = (numPairs + kGetawayListThreads - 1) / kGetawayListThreads;
  const int64_t begin     = perThread * tid < numPairs ? perThread * tid : numPairs;
  const int64_t end       = begin + perThread < numPairs ? begin + perThread : numPairs;
  {
    int local[kNumGetawayLags] = {};
    for (int64_t pair = begin; pair < end; ++pair) {
      const int lag = lags[pair];
#pragma unroll
      for (int l = 0; l < kNumGetawayLags; ++l) {
        local[l] += lag == l + 1 ? 1 : 0;
      }
    }
#pragma unroll
    for (int l = 0; l < kNumGetawayLags; ++l) {
      counts[l][tid] = local[l];
    }
  }
  getawayWarpsBarrier(kGetawayListThreads);
  // Exclusive scan of each lag's per-thread counts (one warp per lag).
  constexpr int kListWarps = kGetawayListThreads / kWarpSize;
  constexpr int kPerLane   = kGetawayListThreads / kWarpSize;
  for (int lag = warp; lag < kNumGetawayLags; lag += kListWarps) {
    int local[kPerLane];
    int sum = 0;
#pragma unroll
    for (int q = 0; q < kPerLane; ++q) {
      local[q] = sum;
      sum += counts[lag][lane * kPerLane + q];
    }
    int inclusive = sum;
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
      inclusive += lane >= offset ? other : 0;
    }
    const int exclusive = inclusive - sum;
#pragma unroll
    for (int q = 0; q < kPerLane; ++q) {
      counts[lag][lane * kPerLane + q] = exclusive + local[q];
    }
    if (lane == kWarpSize - 1) {
      lagOffsets[lag + 1] = inclusive;
    }
  }
  getawayWarpsBarrier(kGetawayListThreads);
  if (tid == 0) {
    lagOffsets[0] = 0;
    for (int lag = 0; lag < kNumGetawayLags; ++lag) {
      lagOffsets[lag + 1] += lagOffsets[lag];
    }
    for (int lag = 0; lag <= kNumGetawayLags; ++lag) {
      lagStarts[moleculeIdx * kGetawayLagStartsStride + lag] = lagOffsets[lag];
    }
  }
  getawayWarpsBarrier(kGetawayListThreads);
  if (begin >= end) {
    return;
  }
  // Row j and column k of pair begin: the largest j with bondTableRowStart(j) <= begin.
  int j =
    static_cast<int>(numAtoms - 0.5 - sqrt((numAtoms - 0.5) * (numAtoms - 0.5) - 2.0 * static_cast<double>(begin)));
  j = j < 0 ? 0 : (j > numAtoms - 2 ? numAtoms - 2 : j);
  while (j > 0 && bondTableRowStart(j, numAtoms) > begin) {
    --j;
  }
  while (j < numAtoms - 2 && bondTableRowStart(j + 1, numAtoms) <= begin) {
    ++j;
  }
  int       k            = static_cast<int>(begin - bondTableRowStart(j, numAtoms)) + j + 1;
  uint32_t* moleculeList = lagPairs + bondTable.starts[moleculeIdx];
  int       position[kNumGetawayLags];
#pragma unroll
  for (int lag = 0; lag < kNumGetawayLags; ++lag) {
    position[lag] = lagOffsets[lag] + counts[lag][tid];
  }
  for (int64_t pair = begin; pair < end; ++pair) {
    const int lag = lags[pair];
    if (lag >= 1 && lag <= kNumGetawayLags) {
      const uint32_t packed = static_cast<uint32_t>(j) | (static_cast<uint32_t>(k) << 16);
#pragma unroll
      for (int l = 0; l < kNumGetawayLags; ++l) {
        if (l == lag - 1) {
          moleculeList[position[l]++] = packed;
        }
      }
    }
    if (++k == numAtoms) {
      ++j;
      k = j + 1;
    }
  }
}

//! Block-collective sum of @p values over the CTA of @p kWarps warps, returned to every thread.
template <int kWarps, int kCount, typename T>
__device__ __forceinline__ void getawayBlockSum(T (&values)[kCount], T* scratch, const int lane, const int warp) {
#pragma unroll
  for (int q = 0; q < kCount; ++q) {
    values[q] = groupAllReduceSum<kWarpSize>(values[q]);
  }
  if (lane == 0) {
#pragma unroll
    for (int q = 0; q < kCount; ++q) {
      scratch[warp * kCount + q] = values[q];
    }
  }
  __syncthreads();
#pragma unroll
  for (int q = 0; q < kCount; ++q) {
    T sum = scratch[q];
    for (int w = 1; w < kWarps; ++w) {
      sum += scratch[w * kCount + q];
    }
    values[q] = sum;
  }
  __syncthreads();
}

/**
 * @brief Sum of @p values over the CTA's first @p numWarps warps, returned to each of their threads. Callers
 *        alternate between two @p scratch buffers of (CTA warps) * kCount values on consecutive calls.
 */
template <int kCount, typename T>
__device__ __forceinline__ void getawayWarpsSum(T (&values)[kCount],
                                                T*        scratch,
                                                const int lane,
                                                const int warp,
                                                const int numWarps) {
#pragma unroll
  for (int q = 0; q < kCount; ++q) {
    values[q] = groupAllReduceSum<kWarpSize>(values[q]);
  }
  if (numWarps == 1) {
    __syncwarp();
    return;
  }
  if (lane == 0) {
#pragma unroll
    for (int q = 0; q < kCount; ++q) {
      scratch[warp * kCount + q] = values[q];
    }
  }
  getawayWarpsBarrier(numWarps * kWarpSize);
#pragma unroll
  for (int q = 0; q < kCount; ++q) {
    T sum = scratch[q];
    for (int w = 1; w < numWarps; ++w) {
      sum += scratch[w * kCount + q];
    }
    values[q] = sum;
  }
}

/**
 * @brief Warp-collective transpose reduction of 32 values per lane: on return lane l holds the combination over
 *        the warp of every lane's @p values[l] (a sum, or the maximum for indices at or above @p firstMax).
 *
 * Each of the five steps exchanges half of the remaining values with the partner lane, so the whole reduction
 * takes 31 shuffles instead of 5 per value; the combination order is fixed, so results are deterministic.
 */
template <typename T>
__device__ __forceinline__ T getawayTransposeReduce(T (&values)[kWarpSize], const int lane, const int firstMax) {
#pragma unroll
  for (int step = kWarpSize / 2; step >= 1; step >>= 1) {
    const bool upper = (lane & step) != 0;
#pragma unroll
    for (int k = 0; k < step; ++k) {
      const T   send  = upper ? values[k] : values[k + step];
      const T   keep  = upper ? values[k + step] : values[k];
      const T   other = __shfl_xor_sync(0xffffffffu, send, step);
      const int index = (lane & ~(2 * step - 1) & (kWarpSize - 1)) | (upper ? step : 0) | k;
      values[k]       = index >= firstMax ? (other > keep ? other : keep) : keep + other;
    }
  }
  return values[0];
}

/**
 * @brief Row @p i of R times @p vector (R's circulant slots in @p matrix), or R's row sum when @p vector is
 *        null.
 */
template <typename Real>
__device__ __forceinline__ Real
getawayRowProduct(const Real* matrix, const Real* vector, const int numAtoms, const int i) {
  const int half = (numAtoms - 1) / 2;
  Real      sum  = 0;
  if (vector == nullptr) {
    Real backSum = 0;
    for (int d = 1; d <= half; ++d) {
      int back = i - d;
      back += back < 0 ? numAtoms : 0;
      sum += matrix[(d - 1) * numAtoms + i];
      backSum += matrix[(d - 1) * numAtoms + back];
    }
    sum += backSum;
    if ((numAtoms & 1) == 0 && numAtoms >= 2) {
      sum += matrix[(numAtoms / 2 - 1) * numAtoms + i];
    }
    return sum;
  }
  // @p vector holds the iterate twice (vector[k] = v[k mod n] for k < 2n), so the forward read vector[i + d] and
  // the backward read vector[i - d + n] need no wrap. Two independent accumulators halve the dependent FMA chain.
  Real        backSum  = 0;
  const Real* forwardV = vector + i;
  const Real* backV    = vector + i + numAtoms;
  const Real* slots    = matrix + i;  // slot (d, i) for the current d
  for (int d = 1; d <= half; ++d) {
    const int back = i - d < 0 ? numAtoms - d : -d;  // slot (d, (i - d) mod n) relative to slot (d, i)
    sum += slots[0] * forwardV[d];
    backSum += slots[back] * backV[-d];
    slots += numAtoms;
  }
  if ((numAtoms & 1) == 0 && numAtoms >= 2) {
    sum += matrix[(numAtoms / 2 - 1) * numAtoms + i] * forwardV[numAtoms / 2];
  }
  return sum + backSum;
}

/**
 * @brief CTA-collective. RDKit's GETAWAY for conformer @p conformerIdx of at most @p fastCapacity atoms, with
 *        the CTA's @p kThreads threads; same results as getaway3DKernel() up to summation order. See the fast-path
 *        notes above.
 */
template <typename Real, int kThreads>
__device__ __forceinline__ void getawayFastConformer(const int                     conformerIdx,
                                                     const DeviceCoordView&        coordinates,
                                                     const Property3DDeviceInputs& inputs,
                                                     const int                     precisionDigits,
                                                     const int                     fastCapacity,
                                                     const BondDistanceTable&      bondTable,
                                                     const uint32_t* __restrict__ lagPairs,
                                                     const int32_t* __restrict__ lagStarts,
                                                     const double* __restrict__ dimensions,
                                                     Real* __restrict__ output) {
  constexpr int kWarps = kThreads / kWarpSize;
  static_assert(kWarps <= kGetawayMaxFastWarps);
  extern __shared__ __align__(16) unsigned char getawayShared[];
  const int                                     tid  = static_cast<int>(threadIdx.x);
  const int                                     lane = tid % kWarpSize;
  const int                                     warp = tid / kWarpSize;
  Real* const          row   = output + static_cast<size_t>(conformerIdx) * kNumGetawayProperties;
  const ConformerAtoms atoms = loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, conformerIdx);
  if (!atoms.valid) {
    for (int valueIdx = tid; valueIdx < kNumGetawayProperties; valueIdx += kThreads) {
      row[valueIdx] = static_cast<Real>(nan(""));
    }
    return;
  }
  const int moleculeIdx = coordinates.molIndices[conformerIdx];
  const int n           = atoms.numAtoms;
  if (n > fastCapacity || moleculeBondDistances(bondTable, moleculeIdx) == nullptr) {
    return;  // Never listed; getaway3DKernel() computes it.
  }
  const int moleculeStart = inputs.moleculeAtomStarts[moleculeIdx];
  const int totalAtoms    = inputs.moleculeAtomStarts[coordinates.nMols];

  const GetawayFastLayout layout    = getawayFastLayout<Real>(fastCapacity, kWarps);
  const GetawayFastReduce offsets   = getawayFastReduce(kWarps);
  double* const           leverages = reinterpret_cast<double*>(getawayShared + layout.leverages);
  Real* const             rows      = reinterpret_cast<Real*>(getawayShared + layout.rows);
  Real* const             matrix    = reinterpret_cast<Real*>(getawayShared + layout.matrix);
  Real* const   vectors = rows + kGetawayFastCentered * n;  // over the centered coordinates (see kGetawayFastRowRegion)
  double* const heavyValues   = reinterpret_cast<double*>(getawayShared + layout.heavyValues);
  int64_t* const keys         = reinterpret_cast<int64_t*>(getawayShared + layout.keys);
  unsigned char* reduce       = getawayShared + layout.reduce;
  Real* const    sumScratch   = reinterpret_cast<Real*>(reduce);
  double* const  sumScratchD  = reinterpret_cast<double*>(reduce);
  Real* const    powerScratch = reinterpret_cast<Real*>(reduce + offsets.power);
  Real* const    rMaxScratch  = reinterpret_cast<Real*>(reduce + offsets.rMax);
  double* const  scalars      = reinterpret_cast<double*>(reduce + offsets.scalars);

  // HIC's dimension: per molecule from the prepass, or this conformer's own when the default is not valid.
  double dimension = dimensions[moleculeIdx];
  if (!(dimension > 0.0)) {
    if (warp == 0) {
      const bool   is3D = inputs.conformerIs3D == nullptr || inputs.conformerIs3D[conformerIdx] != 0;
      const double own  = getawayHicDimension(atoms, is3D, lane % kGroupSize);
      if (lane == 0) {
        scalars[0] = own;
      }
    }
    __syncthreads();
    dimension = scalars[0];
  }

  // Leverages: centroid and moments in float64, then the pseudo-inverse projection (see computeLeverages()).
  double centroid[3] = {0.0, 0.0, 0.0};
  for (int atomIdx = tid; atomIdx < n; atomIdx += kThreads) {
    centroid[0] += atoms.positions[atomIdx * 3 + 0];
    centroid[1] += atoms.positions[atomIdx * 3 + 1];
    centroid[2] += atoms.positions[atomIdx * 3 + 2];
  }
  getawayBlockSum<kWarps>(centroid, sumScratchD, lane, warp);
  const double inverseAtoms = 1.0 / static_cast<double>(n);
  const double centroidX    = centroid[0] * inverseAtoms;
  const double centroidY    = centroid[1] * inverseAtoms;
  const double centroidZ    = centroid[2] * inverseAtoms;
  double       moments[6]   = {};
  for (int atomIdx = tid; atomIdx < n; atomIdx += kThreads) {
    const double x                                 = atoms.positions[atomIdx * 3 + 0] - centroidX;
    const double y                                 = atoms.positions[atomIdx * 3 + 1] - centroidY;
    const double z                                 = atoms.positions[atomIdx * 3 + 2] - centroidZ;
    rows[kGetawayFastCentered * n + atomIdx]       = static_cast<Real>(x);
    rows[(kGetawayFastCentered + 1) * n + atomIdx] = static_cast<Real>(y);
    rows[(kGetawayFastCentered + 2) * n + atomIdx] = static_cast<Real>(z);
    moments[0] += x * x;
    moments[1] += x * y;
    moments[2] += x * z;
    moments[3] += y * y;
    moments[4] += y * z;
    moments[5] += z * z;
  }
  getawayBlockSum<kWarps>(moments, sumScratchD, lane, warp);

  // The last warp solves the moments' eigensystem (a long serial float64 chain) and broadcasts the projection
  // through shared memory while the other warps fill R's slots with the inverse distances 1 / r_jk, which the pair
  // loop scales by sqrt(h_j h_k) once the leverages are known.
  constexpr int kDistanceWarps = kWarps > 1 ? kWarps - 1 : 1;
  double* const projection     = scalars + 1;  // inverse roots, then the row-major eigenvectors
  if (kWarps == 1 || warp == kWarps - 1) {
    double eigenvalues[3];
    double eigenvectors[9];
    symmetricEigensystemJacobi3x3(moments[0],
                                  moments[1],
                                  moments[2],
                                  moments[3],
                                  moments[4],
                                  moments[5],
                                  eigenvalues,
                                  eigenvectors);
    if (lane == 0) {
      for (int axis = 0; axis < 3; ++axis) {
        const double singular = fabs(eigenvalues[axis]);
        projection[axis]      = singular > kGetawayPinvTolerance ? 1.0 / sqrt(singular) : 0.0;
      }
      for (int entry = 0; entry < 9; ++entry) {
        projection[3 + entry] = eigenvectors[entry];
      }
    }
  }
  if (warp < kDistanceWarps) {
    constexpr int kDistanceThreads = kDistanceWarps * kWarpSize;
    const Real*   cx               = rows + kGetawayFastCentered * n;
    const Real*   cy               = rows + (kGetawayFastCentered + 1) * n;
    const Real*   cz               = rows + (kGetawayFastCentered + 2) * n;
    const int     slots            = (n / 2) * n;
    const int     stepD            = kDistanceThreads / n;
    const int     stepI            = kDistanceThreads % n;
    int           d                = tid / n + 1;
    int           i                = tid % n;
#pragma unroll 2
    for (int slot = tid; slot < slots; slot += kDistanceThreads) {
      int k = i + d;
      k -= k >= n ? n : 0;
      const Real dx = cx[i] - cx[k];
      const Real dy = cy[i] - cy[k];
      const Real dz = cz[i] - cz[k];
      matrix[slot]  = Real(1) / sqrt(dx * dx + dy * dy + dz * dz);
      i += stepI;
      d += stepD;
      if (i >= n) {
        i -= n;
        ++d;
      }
    }
  }
  __syncthreads();
  double inverseRoots[3];
  double eigenvectors[9];
  for (int axis = 0; axis < 3; ++axis) {
    inverseRoots[axis] = projection[axis];
  }
  for (int entry = 0; entry < 9; ++entry) {
    eigenvectors[entry] = projection[3 + entry];
  }

  // Per-atom rows, plus lag 0 (H0, HATS0 over atoms with h > 0) and HIC.
  Real   sums[kGetawayFastTotals] = {};  // H, HATS, R totals; lag-0 H, HATS
  double hic                      = 0;
  for (int atomIdx = tid; atomIdx < n; atomIdx += kThreads) {
    const double x        = atoms.positions[atomIdx * 3 + 0] - centroidX;
    const double y        = atoms.positions[atomIdx * 3 + 1] - centroidY;
    const double z        = atoms.positions[atomIdx * 3 + 2] - centroidZ;
    double       leverage = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const double scaled =
        (x * eigenvectors[axis] + y * eigenvectors[3 + axis] + z * eigenvectors[6 + axis]) * inverseRoots[axis];
      rows[(kGetawayFastY + axis) * n + atomIdx] = static_cast<Real>(scaled);
      leverage += scaled * scaled;
    }
    leverages[atomIdx]                       = leverage;
    const Real realLeverage                  = static_cast<Real>(leverage);
    rows[kGetawayFastLeverage * n + atomIdx] = realLeverage;
    rows[kGetawayFastRootLev * n + atomIdx]  = static_cast<Real>(sqrt(leverage));
    hic -= leverage / dimension * log(leverage / dimension) / log(2.0);
#pragma unroll
    for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
      Real weight = Real(1);
      if (channel > 0) {
        weight = static_cast<Real>(
          inputs.atomPropertyWeights[static_cast<size_t>(channel - 1) * totalAtoms + moleculeStart + atomIdx]);
        rows[(kGetawayFastWeights + channel - 1) * n + atomIdx] = weight;
      }
      if (leverage > 0.0) {
        sums[3 * kNumGetawayChannels + channel] += weight * weight * realLeverage;
        sums[4 * kNumGetawayChannels + channel] += weight * weight * realLeverage * realLeverage;
      }
    }
  }
  __syncthreads();

  // Pair totals over R's circulant slots; every slot gets R, each unordered pair is summed once.
  {
    const Real* y0      = rows + kGetawayFastY * n;
    const Real* y1      = rows + (kGetawayFastY + 1) * n;
    const Real* y2      = rows + (kGetawayFastY + 2) * n;
    const Real* lev     = rows + kGetawayFastLeverage * n;
    const Real* rootLev = rows + kGetawayFastRootLev * n;
    const Real* weights = rows + kGetawayFastWeights * n;
    const int   slots   = (n / 2) * n;
    const int   stepD   = kThreads / n;
    const int   stepI   = kThreads % n;
    int         d       = tid / n + 1;
    int         i       = tid % n;
#pragma unroll 2
    for (int slot = tid; slot < slots; slot += kThreads) {
      int k = i + d;
      k -= k >= n ? n : 0;
      const Real rTerm = rootLev[i] * rootLev[k] * matrix[slot];
      matrix[slot]     = rTerm;
      if (2 * d < n || i < d) {
        const Real hatsTerm  = lev[i] * lev[k];
        const Real influence = y0[i] * y0[k] + y1[i] * y1[k] + y2[i] * y2[k];
#pragma unroll
        for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
          const Real product = channel == 0 ? Real(1) : weights[(channel - 1) * n + i] * weights[(channel - 1) * n + k];
          if (influence > Real(0)) {
            sums[channel] += product * influence;
          }
          sums[kNumGetawayChannels + channel] += product * hatsTerm;
          sums[2 * kNumGetawayChannels + channel] += product * rTerm;
        }
      }
      i += stepI;
      d += stepD;
      if (i >= n) {
        i -= n;
        ++d;
      }
    }
  }
  {
    double hicSum[1] = {hic};
    getawayBlockSum<kWarps>(hicSum, sumScratchD, lane, warp);
    hic = hicSum[0];
  }
  {
    // Transpose-reduce the first kWarpSize sums within each warp (lane q holds sum q), the rest by butterflies;
    // then thread q adds the warps' partials of sum q in warp order into totals[q].
    Real* const totals = sumScratch + kWarps * kGetawayFastTotals;
    Real        head[kWarpSize];
#pragma unroll
    for (int q = 0; q < kWarpSize; ++q) {
      head[q] = sums[q];
    }
    sumScratch[warp * kGetawayFastTotals + lane] = getawayTransposeReduce(head, lane, kWarpSize);
#pragma unroll
    for (int q = kWarpSize; q < kGetawayFastTotals; ++q) {
      const Real tail = groupAllReduceSum<kWarpSize>(sums[q]);
      if (lane == q - kWarpSize) {
        sumScratch[warp * kGetawayFastTotals + q] = tail;
      }
    }
    __syncthreads();
    if (tid < kGetawayFastTotals) {
      Real total = sumScratch[tid];
      for (int w = 1; w < kWarps; ++w) {
        total += sumScratch[w * kGetawayFastTotals + tid];
      }
      totals[tid] = total;
    }
    __syncthreads();
    if (tid < kNumGetawayChannels) {
      const int   channel         = tid;
      Real* const hBlock          = row + kGetawayHStart + channel * kGetawayHBlock;
      Real* const rBlock          = row + kGetawayRStart + channel * kGetawayRBlock;
      const Real  lagZeroH        = totals[3 * kNumGetawayChannels + channel];
      const Real  lagZeroHats     = totals[4 * kNumGetawayChannels + channel];
      hBlock[0]                   = roundThousandths(lagZeroH);
      hBlock[kNumGetawayLags + 1] = roundThousandths(lagZeroH + Real(2) * totals[channel]);
      hBlock[kNumGetawayLags + 2] = roundThousandths(lagZeroHats);
      hBlock[kGetawayHBlock - 1]  = roundThousandths(lagZeroHats + Real(2) * totals[kNumGetawayChannels + channel]);
      rBlock[kNumGetawayLags]     = roundThousandths(Real(2) * totals[2 * kNumGetawayChannels + channel]);
    }
  }

  // REIG by power iteration on R on the first powerWarps warps; the first product with ones gives R's row sums
  // for RARS and RCON. ITH, ISH and HGM run on the last warp at the same time when it is not needed for REIG.
  // When the spare-warp split would need two rows per thread (n above (kWarps - 1) warps of rows), every warp
  // iterates and warp 0 computes ITH afterwards instead.
  const int powerWarps =
    n <= kWarpSize || kWarps == 1 ? 1 : (kWarps > 2 && n > (kWarps - 1) * kWarpSize ? kWarps : kWarps - 1);
  const int ithWarp      = powerWarps < kWarps ? kWarps - 1 : 0;
  const int powerThreads = powerWarps * kWarpSize;
  // A spare ITH warp of a CTA with more than two warps skips the lags and starts ITH at once; the other warps
  // take the lags between them. No CTA barrier follows the lags: each warp moves on to REIG or ITH as soon as its
  // own lags are done, and RT+ (which needs every lag's maximum) is written after the barrier at the end.
  const int lagWarps     = kWarps > 2 && ithWarp != 0 ? kWarps - 1 : kWarps;

  // Lags 1-8 from the molecule's lag lists, one warp per lag.
  if (warp < lagWarps) {
    const Real*     y0      = rows + kGetawayFastY * n;
    const Real*     y1      = rows + (kGetawayFastY + 1) * n;
    const Real*     y2      = rows + (kGetawayFastY + 2) * n;
    const Real*     lev     = rows + kGetawayFastLeverage * n;
    const Real*     weights = rows + kGetawayFastWeights * n;
    const int32_t*  starts  = lagStarts + moleculeIdx * kGetawayLagStartsStride;
    const uint32_t* pairs   = lagPairs + bondTable.starts[moleculeIdx];
    for (int lag = warp + 1; lag <= kNumGetawayLags; lag += lagWarps) {
      Real      lagH[kNumGetawayChannels]    = {};
      Real      lagHats[kNumGetawayChannels] = {};
      Real      lagR[kNumGetawayChannels]    = {};
      Real      lagRMax[kNumGetawayChannels] = {};
      const int stop                         = starts[lag];
#pragma unroll 4
      for (int entry = starts[lag - 1] + lane; entry < stop; entry += kWarpSize) {
        const uint32_t packed   = pairs[entry];
        const int      j        = static_cast<int>(packed & 0xffffu);
        const int      k        = static_cast<int>(packed >> 16);
        const int      distance = k - j;
        const Real     rTerm = 2 * distance <= n ? matrix[(distance - 1) * n + j] : matrix[(n - distance - 1) * n + k];
        const Real     hatsTerm  = lev[j] * lev[k];
        const Real     influence = y0[j] * y0[k] + y1[j] * y1[k] + y2[j] * y2[k];
#pragma unroll
        for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
          const Real product = channel == 0 ? Real(1) : weights[(channel - 1) * n + j] * weights[(channel - 1) * n + k];
          lagHats[channel] += product * hatsTerm;
          if (influence > Real(0)) {
            lagH[channel] += product * influence;
          }
          const Real weighted = product * rTerm;
          lagR[channel] += weighted;
          if (weighted > lagRMax[channel]) {
            lagRMax[channel] = weighted;
          }
        }
      }
      // Lane q * 7 + c receives quantity q (H, HATS, R, R max) of channel c.
      Real packed[kWarpSize];
#pragma unroll
      for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
        packed[channel]                           = lagH[channel];
        packed[kNumGetawayChannels + channel]     = lagHats[channel];
        packed[2 * kNumGetawayChannels + channel] = lagR[channel];
        packed[3 * kNumGetawayChannels + channel] = lagRMax[channel];
      }
#pragma unroll
      for (int pad = 4 * kNumGetawayChannels; pad < kWarpSize; ++pad) {
        packed[pad] = Real(0);
      }
      const Real total = getawayTransposeReduce(packed, lane, 3 * kNumGetawayChannels);
      if (lane < 4 * kNumGetawayChannels) {
        const int   quantity = lane / kNumGetawayChannels;
        const int   channel  = lane % kNumGetawayChannels;
        Real* const hBlock   = row + kGetawayHStart + channel * kGetawayHBlock;
        Real* const rBlock   = row + kGetawayRStart + channel * kGetawayRBlock;
        const Real  rounded  = roundThousandths(total);
        if (quantity == 0) {
          hBlock[lag] = rounded;
        } else if (quantity == 1) {
          hBlock[kNumGetawayLags + 2 + lag] = rounded;
        } else if (quantity == 2) {
          rBlock[lag - 1] = rounded;
        } else {
          rBlock[kNumGetawayLags + lag]                          = rounded;
          rMaxScratch[(lag - 1) * kNumGetawayChannels + channel] = total;
        }
      }
    }
  }

  if (warp < powerWarps) {
    constexpr int kParityStride = kWarps * 4;
    // Two iterate buffers of 2n values each (the iterate twice, see getawayRowProduct()).
    Real* const   next          = vectors + 2 * n;
    Real          first[3]      = {0, 0, 0};  // v^T R v, |R v|^2, sum of row sums
    for (int atomIdx = tid; atomIdx < n; atomIdx += powerThreads) {
      const Real value  = getawayRowProduct<Real>(matrix, nullptr, n, atomIdx);
      next[atomIdx]     = value;
      next[atomIdx + n] = value;
      first[0] += value;
      first[1] += value * value;
      first[2] += value;
    }
    getawayWarpsSum(first, powerScratch, lane, warp, powerWarps);
    Real       numerator   = first[0];
    Real       normSquared = first[1];
    const Real rars        = first[2] / static_cast<Real>(n);
    Real       rcon[1]     = {0};
    for (int atomIdx = tid; atomIdx < n; atomIdx += powerThreads) {
      for (int edge = inputs.bondNeighborStarts[moleculeStart + atomIdx];
           edge < inputs.bondNeighborStarts[moleculeStart + atomIdx + 1];
           ++edge) {
        const int neighbor = inputs.bondNeighbors[edge];
        if (neighbor > atomIdx) {
          rcon[0] += sqrt(next[atomIdx] * next[neighbor]);
        }
      }
    }
    getawayWarpsSum(rcon, powerScratch + kParityStride, lane, warp, powerWarps);
    Real           eigenvalue = numerator / static_cast<Real>(n);
    constexpr Real kTolerance = sizeof(Real) == sizeof(double) ? Real(1e-13) : Real(2e-7);
    int            source     = 1;
    for (int iteration = 0; iteration < kGetawayMaxIterations && normSquared > Real(0); ++iteration) {
      const Real  inverseNorm = Real(1) / sqrt(normSquared);
      const Real* current     = vectors + source * 2 * n;
      Real*       product     = vectors + (1 - source) * 2 * n;
      Real        partial[2]  = {0, 0};
      for (int atomIdx = tid; atomIdx < n; atomIdx += powerThreads) {
        const Real value     = getawayRowProduct(matrix, current, n, atomIdx) * inverseNorm;
        product[atomIdx]     = value;
        product[atomIdx + n] = value;
        partial[0] += current[atomIdx] * inverseNorm * value;
        partial[1] += value * value;
      }
      getawayWarpsSum(partial, powerScratch + (iteration & 1) * kParityStride, lane, warp, powerWarps);
      numerator           = partial[0];
      normSquared         = partial[1];
      source              = 1 - source;
      const Real previous = eigenvalue;
      eigenvalue          = numerator;
      if (fabs(eigenvalue - previous) <= kTolerance * fabs(eigenvalue)) {
        break;
      }
    }
    if (!(normSquared > Real(0))) {
      eigenvalue = Real(0);
    }
    if (tid == 0) {
      row[kGetawayRconIndex + 0] = roundThousandths(rcon[0]);
      row[kGetawayRconIndex + 1] = roundThousandths(rars);
      row[kGetawayRconIndex + 2] = roundThousandths(eigenvalue);
    }
  }

  if (warp == ithWarp) {
    double ith = 0;
    double ish = 0;
    computeInformationIndices(leverages,
                              inputs.heavyAtomFlags + moleculeStart,
                              n,
                              precisionDigits,
                              lane,
                              heavyValues,
                              keys,
                              keys + n,
                              ith,
                              ish);
    if (lane == 0) {
      double leverageProduct = 1.0;
      for (int atomIdx = 0; atomIdx < n; ++atomIdx) {
        leverageProduct *= leverages[atomIdx];
      }
      const double hgm = 100.0 * pow(leverageProduct, 1.0 / n);
      row[0]           = static_cast<Real>(roundThousandths(ith));
      row[1]           = static_cast<Real>(roundThousandths(ish));
      row[2]           = static_cast<Real>(roundThousandths(hic));
      row[3]           = static_cast<Real>(roundThousandths(hgm));
    }
  }

  // RT+: every lag's maximum is in rMaxScratch once all warps are here.
  __syncthreads();
  if (tid < kNumGetawayChannels) {
    Real rMax = 0;
    for (int lag = 0; lag < kNumGetawayLags; ++lag) {
      const Real value = rMaxScratch[lag * kNumGetawayChannels + tid];
      rMax             = value > rMax ? value : rMax;
    }
    row[kGetawayRStart + tid * kGetawayRBlock + kGetawayRBlock - 1] = roundThousandths(rMax);
  }
}

//! Register budget of getawayFastKernel(), in resident CTAs per SM. On devices with about 228 KB of shared memory
//! per SM, the float64 layouts fit seven small (64-thread) CTAs for molecules of up to 64 atoms, or about three large
//! (160-thread) CTAs for molecules near 100 atoms, so registers should not be what limits residency.
template <int kThreads> __host__ __device__ constexpr int getawayFastMinBlocks() {
  return kThreads >= kGetawayFastBlockSize ? 3 : 7;
}

/**
 * @brief RDKit's GETAWAY for the conformers of the size-class lists starting at @p bucket, one CTA of @p kThreads
 *        threads per conformer (getawayFastConformer()). CTA b computes entries b, b + gridDim.x, ... of the class
 *        lists concatenated in order, largest molecules first.
 */
template <typename Real, int kThreads>
__global__ void __launch_bounds__(kThreads, getawayFastMinBlocks<kThreads>())
  getawayFastKernel(const DeviceCoordView        coordinates,
                    const Property3DDeviceInputs inputs,
                    const int                    precisionDigits,
                    const int                    fastCapacity,
                    const BondDistanceTable      bondTable,
                    const GetawayWorkLists       lists,
                    const int                    bucket,
                    const uint32_t* __restrict__ lagPairs,
                    const int32_t* __restrict__ lagStarts,
                    const double* __restrict__ dimensions,
                    Real* __restrict__ output) {
  int sizeClass = 0;
  int skipped   = 0;  // entries of the classes before sizeClass
  for (int next = static_cast<int>(blockIdx.x); sizeClass < kGetawaySizeClasses;) {
    const int list = bucket + sizeClass;
    __syncthreads();  // Every thread is done with the previous conformer's shared memory.
    const int position = next - skipped;
    const int count    = lists.counts[list];
    if (position >= count) {
      skipped += count;
      ++sizeClass;
      continue;
    }
    getawayFastConformer<Real, kThreads>(lists.conformers[static_cast<size_t>(list) * lists.stride + position],
                                         coordinates,
                                         inputs,
                                         precisionDigits,
                                         fastCapacity,
                                         bondTable,
                                         lagPairs,
                                         lagStarts,
                                         dimensions,
                                         output);
    next += static_cast<int>(gridDim.x);
  }
}

/**
 * @brief Collective over aligned groups of @p kLanes lanes: transpose reduction of @p kCount values per lane.
 *        On return lane g of each group (g = lane % kLanes) holds, in values[0, kCount / kLanes), the group's
 *        combination of indices g * (kCount / kLanes) + [0, kCount / kLanes): sums, or maxima for indices at or
 *        above @p firstMax. Every lane of the warp must call it; the combination order is fixed.
 */
template <int kLanes, int kCount, typename T>
__device__ __forceinline__ void getawayGroupTransposeReduce(T (&values)[kCount], const int lane, const int firstMax) {
  static_assert(kCount % kLanes == 0 && (kLanes & (kLanes - 1)) == 0 && kLanes <= kWarpSize);
  constexpr int kPerLane = kCount / kLanes;
#pragma unroll
  for (int step = kLanes / 2; step >= 1; step >>= 1) {
    const bool upper = (lane & step) != 0;
    const int  width = kPerLane * step;
    const int  base  = ((lane & (kLanes - 1)) & ~(2 * step - 1)) * kPerLane + (upper ? width : 0);
#pragma unroll
    for (int k = 0; k < width; ++k) {
      const T send  = upper ? values[k] : values[k + width];
      const T keep  = upper ? values[k + width] : values[k];
      const T other = __shfl_xor_sync(0xffffffffu, send, step);
      values[k]     = base + k >= firstMax ? (other > keep ? other : keep) : keep + other;
    }
  }
}

// ---- Warp path: one warp per conformer of at most kGetawayWarpMaxAtoms atoms -------------------------------------
//
// Lane i owns atom i. Atom rows and R's circulant slots live in the warp's own shared memory; R's forward and
// backward entries of row i (R(i, i + d) and R(i, i - d), d = 1..n/2) are also held in registers, so each
// power-iteration product reads only the iterate (wrap-free from a doubled copy). No CTA barriers: the CTA's
// warps are independent conformers.

constexpr int kGetawayWarpHalf     = kGetawayWarpMaxAtoms / 2;
constexpr int kGetawayWarpCtaWarps = 4;

template <typename Real> struct GetawayWarpShared {
  double  leverages[kGetawayWarpMaxAtoms];
  double  heavyValues[kGetawayWarpMaxAtoms];
  int64_t keys[2 * kGetawayWarpMaxAtoms];
  Real    rows[kGetawayFastRowArrays][kGetawayWarpMaxAtoms];
  Real    matrix[kGetawayWarpHalf][kGetawayWarpMaxAtoms];  //!< matrix[d - 1][i] = R(i, (i + d) mod n)
  Real    iterate[2 * kGetawayWarpMaxAtoms];               //!< iterate[k] = v[k mod n] for k < 2n
  Real    rowSums[kGetawayWarpMaxAtoms];
  Real    totals[2 * kNumGetawayChannels];  //!< lag-0 H and HATS per channel
};

/**
 * @brief Warp-collective. RDKit's GETAWAY for a valid conformer of at most kGetawayWarpMaxAtoms atoms whose molecule
 * has bond-distance table entries, or the NaN row of an invalid conformer. Same results as getaway3DKernel() up to
 * summation order (its leverages, HIC and RCON reductions are identical).
 */
template <typename Real>
__device__ __forceinline__ void getawayWarpConformer(const int                     conformerIdx,
                                                     GetawayWarpShared<Real>&      shared,
                                                     const DeviceCoordView&        coordinates,
                                                     const Property3DDeviceInputs& inputs,
                                                     const int                     precisionDigits,
                                                     const BondDistanceTable&      bondTable,
                                                     const uint32_t* __restrict__ lagPairs,
                                                     const int32_t* __restrict__ lagStarts,
                                                     const double* __restrict__ dimensions,
                                                     Real* __restrict__ output) {
  const int            lane  = static_cast<int>(threadIdx.x) % kWarpSize;
  Real* const          row   = output + static_cast<size_t>(conformerIdx) * kNumGetawayProperties;
  const ConformerAtoms atoms = loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, conformerIdx);
  if (!atoms.valid) {
    for (int valueIdx = lane; valueIdx < kNumGetawayProperties; valueIdx += kWarpSize) {
      row[valueIdx] = static_cast<Real>(nan(""));
    }
    return;
  }
  const int n           = atoms.numAtoms;
  const int moleculeIdx = coordinates.molIndices[conformerIdx];
  if (n > kGetawayWarpMaxAtoms || moleculeBondDistances(bondTable, moleculeIdx) == nullptr) {
    return;  // Another kernel computes it.
  }
  const int  moleculeStart = inputs.moleculeAtomStarts[moleculeIdx];
  const int  totalAtoms    = inputs.moleculeAtomStarts[coordinates.nMols];
  const bool active        = lane < n;
  const int  half          = n / 2;

  double dimension = dimensions[moleculeIdx];
  if (!(dimension > 0.0)) {
    const bool is3D = inputs.conformerIs3D == nullptr || inputs.conformerIs3D[conformerIdx] != 0;
    dimension       = getawayHicDimension(atoms, is3D, lane % kGroupSize);
  }
  for (int k = lane; k < 2 * kGetawayWarpMaxAtoms; k += kWarpSize) {
    shared.iterate[k] = Real(0);
  }

  // Leverages exactly as computeLeverages(): one atom per lane, float64 centroid, moments and Jacobi.
  double px = 0.0;
  double py = 0.0;
  double pz = 0.0;
  if (active) {
    px = atoms.positions[lane * 3 + 0];
    py = atoms.positions[lane * 3 + 1];
    pz = atoms.positions[lane * 3 + 2];
  }
  const double inverseAtoms = 1.0 / static_cast<double>(n);
  const double centroidX    = groupAllReduceSum<kWarpSize>(px) * inverseAtoms;
  const double centroidY    = groupAllReduceSum<kWarpSize>(py) * inverseAtoms;
  const double centroidZ    = groupAllReduceSum<kWarpSize>(pz) * inverseAtoms;
  const double x            = active ? px - centroidX : 0.0;
  const double y            = active ? py - centroidY : 0.0;
  const double z            = active ? pz - centroidZ : 0.0;
  double       moments[6]   = {x * x, x * y, x * z, y * y, y * z, z * z};
  for (double& moment : moments) {
    moment = groupAllReduceSum<kWarpSize>(moment);
  }
  double eigenvalues[3];
  double eigenvectors[9];
  symmetricEigensystemJacobi3x3(moments[0],
                                moments[1],
                                moments[2],
                                moments[3],
                                moments[4],
                                moments[5],
                                eigenvalues,
                                eigenvectors);
  double leverage = 0;
  Real   ownY[3];
  for (int axis = 0; axis < 3; ++axis) {
    const double singular    = fabs(eigenvalues[axis]);
    const double inverseRoot = singular > kGetawayPinvTolerance ? 1.0 / sqrt(singular) : 0.0;
    const double scaled =
      (x * eigenvectors[axis] + y * eigenvectors[3 + axis] + z * eigenvectors[6 + axis]) * inverseRoot;
    ownY[axis] = static_cast<Real>(scaled);
    leverage += scaled * scaled;
  }
  const Real ownLeverage = static_cast<Real>(leverage);
  const Real ownRootLev  = static_cast<Real>(sqrt(leverage));
  const Real ownX        = static_cast<Real>(x);
  const Real ownYc       = static_cast<Real>(y);
  const Real ownZ        = static_cast<Real>(z);
  Real       ownWeights[kNumAtomPropertyChannels];
  for (int channel = 0; channel < kNumAtomPropertyChannels; ++channel) {
    ownWeights[channel] =
      active ? static_cast<Real>(
                 inputs.atomPropertyWeights[static_cast<size_t>(channel) * totalAtoms + moleculeStart + lane]) :
               Real(0);
  }
  if (active) {
    shared.leverages[lane] = leverage;
    for (int axis = 0; axis < 3; ++axis) {
      shared.rows[kGetawayFastY + axis][lane] = ownY[axis];
    }
    shared.rows[kGetawayFastLeverage][lane]     = ownLeverage;
    shared.rows[kGetawayFastRootLev][lane]      = ownRootLev;
    shared.rows[kGetawayFastCentered + 0][lane] = ownX;
    shared.rows[kGetawayFastCentered + 1][lane] = ownYc;
    shared.rows[kGetawayFastCentered + 2][lane] = ownZ;
    for (int channel = 0; channel < kNumAtomPropertyChannels; ++channel) {
      shared.rows[kGetawayFastWeights + channel][lane] = ownWeights[channel];
    }
  }

  // HIC and lag 0 (H0 = sum w^2 h, HATS0 = sum w^2 h^2 over atoms with h > 0), reduced as getaway3DKernel() does.
  double hic = 0.0;
  if (active) {
    hic -= leverage / dimension * log(leverage / dimension) / log(2.0);
  }
  hic = groupAllReduceSum<kWarpSize>(hic);
  {
    Real lagZero[kWarpSize];
#pragma unroll
    for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
      const Real weight                      = channel == 0 ? Real(1) : ownWeights[channel - 1];
      const bool counts                      = active && leverage > 0.0;
      lagZero[channel]                       = counts ? weight * weight * ownLeverage : Real(0);
      lagZero[kNumGetawayChannels + channel] = counts ? weight * weight * ownLeverage * ownLeverage : Real(0);
    }
#pragma unroll
    for (int pad = 2 * kNumGetawayChannels; pad < kWarpSize; ++pad) {
      lagZero[pad] = Real(0);
    }
    const Real total = getawayTransposeReduce(lagZero, lane, kWarpSize);
    if (lane < 2 * kNumGetawayChannels) {
      shared.totals[lane] = total;
    }
  }
  __syncwarp();

  // Pair totals over the circulant slots (d, i): R(i, i + d), counting each unordered pair once.
  {
    Real sums[3 * kNumGetawayChannels];
#pragma unroll
    for (int q = 0; q < 3 * kNumGetawayChannels; ++q) {
      sums[q] = Real(0);
    }
#pragma unroll
    for (int d = 1; d <= kGetawayWarpHalf; ++d) {
      if (d <= half) {
        int k = lane + d;
        k -= k >= n ? n : 0;
        if (active) {
          const Real dx    = ownX - shared.rows[kGetawayFastCentered + 0][k];
          const Real dy    = ownYc - shared.rows[kGetawayFastCentered + 1][k];
          const Real dz    = ownZ - shared.rows[kGetawayFastCentered + 2][k];
          const Real rTerm = ownRootLev * shared.rows[kGetawayFastRootLev][k] / sqrt(dx * dx + dy * dy + dz * dz);
          shared.matrix[d - 1][lane] = rTerm;
          if (2 * d < n || lane < d) {
            const Real hatsTerm  = ownLeverage * shared.rows[kGetawayFastLeverage][k];
            const Real influence = ownY[0] * shared.rows[kGetawayFastY + 0][k] +
                                   ownY[1] * shared.rows[kGetawayFastY + 1][k] +
                                   ownY[2] * shared.rows[kGetawayFastY + 2][k];
#pragma unroll
            for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
              const Real product =
                channel == 0 ? Real(1) : ownWeights[channel - 1] * shared.rows[kGetawayFastWeights + channel - 1][k];
              if (influence > Real(0)) {
                sums[channel] += product * influence;
              }
              sums[kNumGetawayChannels + channel] += product * hatsTerm;
              sums[2 * kNumGetawayChannels + channel] += product * rTerm;
            }
          }
        }
      }
    }
    Real packed[kWarpSize];
#pragma unroll
    for (int q = 0; q < kWarpSize; ++q) {
      packed[q] = q < 3 * kNumGetawayChannels ? sums[q] : Real(0);
    }
    const Real total = getawayTransposeReduce(packed, lane, kWarpSize);
    __syncwarp();
    // Lane c receives channel c's H, HATS and R totals from lanes c, 7 + c and 14 + c.
    const Real hatsTotal = __shfl_sync(0xffffffffu, total, (lane + kNumGetawayChannels) & (kWarpSize - 1));
    const Real rTotal    = __shfl_sync(0xffffffffu, total, (lane + 2 * kNumGetawayChannels) & (kWarpSize - 1));
    if (lane < kNumGetawayChannels) {
      const int   channel         = lane;
      Real* const hBlock          = row + kGetawayHStart + channel * kGetawayHBlock;
      Real* const rBlock          = row + kGetawayRStart + channel * kGetawayRBlock;
      const Real  lagZeroH        = shared.totals[channel];
      const Real  lagZeroHats     = shared.totals[kNumGetawayChannels + channel];
      hBlock[0]                   = roundThousandths(lagZeroH);
      hBlock[kNumGetawayLags + 1] = roundThousandths(lagZeroH + Real(2) * total);
      hBlock[kNumGetawayLags + 2] = roundThousandths(lagZeroHats);
      hBlock[kGetawayHBlock - 1]  = roundThousandths(lagZeroHats + Real(2) * hatsTotal);
      rBlock[kNumGetawayLags]     = roundThousandths(Real(2) * rTotal);
    }
  }
  // Lags 1-8 from the molecule's lag lists: lane group g of four takes lag g + 1, then a transpose reduction
  // leaves quantity (lane % 4) of every channel in each lane: H, HATS, R sums and the R maximum.
  {
    const int32_t*  starts = lagStarts + moleculeIdx * kGetawayLagStartsStride;
    const uint32_t* pairs  = lagPairs + bondTable.starts[moleculeIdx];
    const int       lag    = lane / 4 + 1;
    Real            lagSums[4 * kNumGetawayChannels];
#pragma unroll
    for (int q = 0; q < 4 * kNumGetawayChannels; ++q) {
      lagSums[q] = Real(0);
    }
    const int stop = starts[lag];
#pragma unroll 4
    for (int entry = starts[lag - 1] + (lane & 3); entry < stop; entry += 4) {
      const uint32_t packed   = pairs[entry];
      const int      j        = static_cast<int>(packed & 0xffffu);
      const int      k        = static_cast<int>(packed >> 16);
      const int      distance = k - j;
      const Real     rTerm    = 2 * distance <= n ? shared.matrix[distance - 1][j] : shared.matrix[n - distance - 1][k];
      const Real     hatsTerm = shared.rows[kGetawayFastLeverage][j] * shared.rows[kGetawayFastLeverage][k];
      const Real     influence = shared.rows[kGetawayFastY + 0][j] * shared.rows[kGetawayFastY + 0][k] +
                             shared.rows[kGetawayFastY + 1][j] * shared.rows[kGetawayFastY + 1][k] +
                             shared.rows[kGetawayFastY + 2][j] * shared.rows[kGetawayFastY + 2][k];
#pragma unroll
      for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
        const Real product = channel == 0 ? Real(1) :
                                            shared.rows[kGetawayFastWeights + channel - 1][j] *
                                              shared.rows[kGetawayFastWeights + channel - 1][k];
        if (influence > Real(0)) {
          lagSums[channel] += product * influence;
        }
        lagSums[kNumGetawayChannels + channel] += product * hatsTerm;
        const Real weighted = product * rTerm;
        lagSums[2 * kNumGetawayChannels + channel] += weighted;
        if (weighted > lagSums[3 * kNumGetawayChannels + channel]) {
          lagSums[3 * kNumGetawayChannels + channel] = weighted;
        }
      }
    }
    getawayGroupTransposeReduce<4>(lagSums, lane, 3 * kNumGetawayChannels);
    const int quantity = lane & 3;
    // RT+: the maximum over lags of each channel's lag maximum (lanes with quantity 3, one per lag).
    Real      overall[kNumGetawayChannels];
#pragma unroll
    for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
      Real value = lagSums[channel];
      for (int offset = 4; offset < kWarpSize; offset <<= 1) {
        const Real other = __shfl_xor_sync(0xffffffffu, value, offset);
        value            = other > value ? other : value;
      }
      overall[channel] = value;
    }
#pragma unroll
    for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
      Real* const hBlock  = row + kGetawayHStart + channel * kGetawayHBlock;
      Real* const rBlock  = row + kGetawayRStart + channel * kGetawayRBlock;
      const Real  rounded = roundThousandths(lagSums[channel]);
      if (quantity == 0) {
        hBlock[lag] = rounded;
      } else if (quantity == 1) {
        hBlock[kNumGetawayLags + 2 + lag] = rounded;
      } else if (quantity == 2) {
        rBlock[lag - 1] = rounded;
      } else {
        rBlock[kNumGetawayLags + lag] = rounded;
        if (lag == 1) {
          rBlock[kGetawayRBlock - 1] = roundThousandths(overall[channel]);
        }
      }
    }
  }

  // Row i of R in registers for the power iteration: forward entries R(i, i + d) from the slots, backward
  // entries R(i, i - d) = R(i - d, i) from lane i - d; for even n the d = n/2 entry is the forward one.
  Real      forward[kGetawayWarpHalf];
  Real      backward[kGetawayWarpHalf];
  const int backHalf = (n & 1) != 0 ? half : half - 1;
#pragma unroll
  for (int d = 1; d <= kGetawayWarpHalf; ++d) {
    forward[d - 1] = active && d <= half ? shared.matrix[d - 1][lane] : Real(0);
    int source     = lane - d;
    source += source < 0 ? n : 0;
    const Real value = __shfl_sync(0xffffffffu, forward[d - 1], source & (kWarpSize - 1));
    backward[d - 1]  = active && d <= backHalf ? value : Real(0);
  }

  // REIG by power iteration on R (as getaway3DKernel()); the first product with ones gives the row sums.
  const auto multiply = [&]() {
    Real sum     = 0;
    Real backSum = 0;
    if (!active) {
      return Real(0);
    }
#pragma unroll
    for (int d = 1; d <= kGetawayWarpHalf; ++d) {
      if (d <= half) {
        sum += forward[d - 1] * shared.iterate[lane + d];
        backSum += backward[d - 1] * shared.iterate[lane - d + n];
      }
    }
    return sum + backSum;
  };
  if (active) {
    shared.iterate[lane]     = Real(1);
    shared.iterate[lane + n] = Real(1);
  }
  __syncwarp();
  Real product     = multiply();
  Real numerator   = groupAllReduceSum<kWarpSize>(product);
  Real normSquared = groupAllReduceSum<kWarpSize>(product * product);
  if (active) {
    shared.rowSums[lane] = product;
  }
  __syncwarp();
  Real rcon = 0;
  if (active) {
    for (int edge = inputs.bondNeighborStarts[moleculeStart + lane];
         edge < inputs.bondNeighborStarts[moleculeStart + lane + 1];
         ++edge) {
      const int neighbor = inputs.bondNeighbors[edge];
      if (neighbor > lane) {
        rcon += sqrt(product * shared.rowSums[neighbor]);
      }
    }
  }
  rcon                      = groupAllReduceSum<kWarpSize>(rcon);
  const Real     rars       = numerator / static_cast<Real>(n);
  Real           eigenvalue = numerator / static_cast<Real>(n);
  constexpr Real kTolerance = sizeof(Real) == sizeof(double) ? Real(1e-13) : Real(2e-7);
  for (int iteration = 0; iteration < kGetawayMaxIterations && normSquared > Real(0); ++iteration) {
    const Real inverseNorm = Real(1) / sqrt(normSquared);
    const Real current     = product * inverseNorm;
    if (active) {
      shared.iterate[lane]     = current;
      shared.iterate[lane + n] = current;
    }
    __syncwarp();
    product             = multiply();
    numerator           = groupAllReduceSum<kWarpSize>(active ? current * product : Real(0));
    normSquared         = groupAllReduceSum<kWarpSize>(product * product);
    const Real previous = eigenvalue;
    eigenvalue          = numerator;
    __syncwarp();
    if (fabs(eigenvalue - previous) <= kTolerance * fabs(eigenvalue)) {
      break;
    }
  }
  if (!(normSquared > Real(0))) {
    eigenvalue = Real(0);
  }

  double ith = 0;
  double ish = 0;
  computeInformationIndices(shared.leverages,
                            inputs.heavyAtomFlags + moleculeStart,
                            n,
                            precisionDigits,
                            lane,
                            shared.heavyValues,
                            shared.keys,
                            shared.keys + n,
                            ith,
                            ish);
  if (lane == 0) {
    double leverageProduct = 1.0;
    for (int atomIdx = 0; atomIdx < n; ++atomIdx) {
      leverageProduct *= shared.leverages[atomIdx];
    }
    const double hgm           = 100.0 * pow(leverageProduct, 1.0 / n);
    row[0]                     = static_cast<Real>(roundThousandths(ith));
    row[1]                     = static_cast<Real>(roundThousandths(ish));
    row[2]                     = static_cast<Real>(roundThousandths(hic));
    row[3]                     = static_cast<Real>(roundThousandths(hgm));
    row[kGetawayRconIndex + 0] = roundThousandths(rcon);
    row[kGetawayRconIndex + 1] = roundThousandths(rars);
    row[kGetawayRconIndex + 2] = roundThousandths(eigenvalue);
  }
}

/**
 * @brief The warp bucket's conformers, one warp per conformer (getawayWarpConformer()): warp w of CTA b computes
 *        list entries b * kGetawayWarpCtaWarps + w, then every (gridDim.x * kGetawayWarpCtaWarps)-th one after it.
 */
template <typename Real>
__global__ void __launch_bounds__(kGetawayWarpCtaWarps* kWarpSize)
  getawayWarpKernel(const DeviceCoordView        coordinates,
                    const Property3DDeviceInputs inputs,
                    const int                    precisionDigits,
                    const BondDistanceTable      bondTable,
                    const GetawayWorkLists       lists,
                    const uint32_t* __restrict__ lagPairs,
                    const int32_t* __restrict__ lagStarts,
                    const double* __restrict__ dimensions,
                    Real* __restrict__ output) {
  __shared__ GetawayWarpShared<Real> sharedWarps[kGetawayWarpCtaWarps];
  const int                          count = lists.counts[kGetawayWarpBucket];
  const int*               entries         = lists.conformers + static_cast<size_t>(kGetawayWarpBucket) * lists.stride;
  GetawayWarpShared<Real>& shared          = sharedWarps[threadIdx.x / kWarpSize];
  for (int position = static_cast<int>(blockIdx.x * kGetawayWarpCtaWarps + threadIdx.x / kWarpSize); position < count;
       position += static_cast<int>(gridDim.x) * kGetawayWarpCtaWarps) {
    getawayWarpConformer<Real>(entries[position],
                               shared,
                               coordinates,
                               inputs,
                               precisionDigits,
                               bondTable,
                               lagPairs,
                               lagStarts,
                               dimensions,
                               output);
    __syncwarp();  // Every lane is done with the conformer's shared memory.
  }
}

/**
 * @brief RDKit's GETAWAY for every conformer; one warp per conformer.
 *
 * Per conformer: leverages and scratch rows (computeLeverages()); per-atom H0/HATS0 sums, HIC (dimension from
 * the PBF of the molecule's default conformer, as RDKit's PBF(mol) call does), HGM (sequential product, so
 * underflow matches RDKit) and ITH/ISH (computeInformationIndices()); then the pair loop: each kGroupSize-lane
 * group walks interleaved rows j of the pair triangle, finds row j's bond distances (capped at
 * kNumGetawayLags) with searchBondDepths() into its @p bondDepths array, and the lane owning a pair's lag
 * accumulates its H, HATS and R terms; pairs farther apart or disconnected enter the totals only. REIG, the largest
 * singular value of the nonnegative symmetric R, is its Perron eigenvalue, found by power iteration whose first product
 * (with ones) gives the row sums for RARS and RCON.
 */
template <typename Real>
__global__ void getaway3DKernel(const DeviceCoordView        coordinates,
                                const Property3DDeviceInputs inputs,
                                const int                    precisionDigits,
                                Real* __restrict__ rows,
                                double* __restrict__ leverageScratch,
                                double* __restrict__ sortScratch,
                                int64_t* __restrict__ keyScratch,
                                Real* __restrict__ iterateScratch,
                                uint8_t* __restrict__ bondDepths,
                                const BondDistanceTable bondTable,
                                const int               fastCapacity,
                                Real* __restrict__ output) {
  const int lane         = static_cast<int>(threadIdx.x) % kWarpSize;
  const int laneInGroup  = lane % kGroupSize;
  const int stream       = lane / kGroupSize;
  const int conformerIdx = blockIdx.x * kWarpsPerBlock + static_cast<int>(threadIdx.x) / kWarpSize;
  if (conformerIdx >= coordinates.numConformers) {
    return;  // Uniform across the warp.
  }
  if (getawayFastPathOwns(coordinates, inputs, bondTable, fastCapacity, conformerIdx)) {
    return;  // Computed by getawayFastKernel().
  }
  Real* const          row   = output + static_cast<size_t>(conformerIdx) * kNumGetawayProperties;
  const ConformerAtoms atoms = loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, conformerIdx);
  if (!atoms.valid) {
    for (int valueIdx = lane; valueIdx < kNumGetawayProperties; valueIdx += kWarpSize) {
      row[valueIdx] = static_cast<Real>(nan(""));
    }
    return;
  }
  const int     numAtoms      = atoms.numAtoms;
  const int     moleculeIdx   = coordinates.molIndices[conformerIdx];
  const int     moleculeStart = inputs.moleculeAtomStarts[moleculeIdx];
  const int     totalAtoms    = inputs.moleculeAtomStarts[coordinates.nMols];
  const size_t  atomOffset    = static_cast<size_t>(atoms.positions - coordinates.positions) / 3;
  Real* const   atomRows      = rows + atomOffset * kGetawayRowStride;
  double* const leverages     = leverageScratch + atomOffset;
  computeLeverages(atoms, inputs.atomPropertyWeights + moleculeStart, totalAtoms, lane, atomRows, leverages);

  // HIC's dimension: RDKit's PBF of the molecule's default conformer below kGetawayPlanarPbf means 2D.
  int            defaultRow = inputs.defaultConformerRows != nullptr ? inputs.defaultConformerRows[moleculeIdx] : -1;
  ConformerAtoms defaultAtoms =
    defaultRow >= 0 ? loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, defaultRow) : ConformerAtoms{};
  if (!defaultAtoms.valid) {
    defaultRow   = conformerIdx;
    defaultAtoms = atoms;
  }
  ProjectionState<double> plane;
  computeProjectionCentroid(defaultAtoms, laneInGroup, plane);
  computeProjectionPca(defaultAtoms, nullptr, laneInGroup, plane);
  const double pbf       = computePbf(defaultAtoms, laneInGroup, plane);
  const bool   is3D      = inputs.conformerIs3D == nullptr || inputs.conformerIs3D[defaultRow] != 0;
  const double dimension = (defaultAtoms.numAtoms < 4 || !is3D ? 0.0 : pbf) < kGetawayPlanarPbf ? 2.0 : 3.0;

  // Pair loop: lane l owns lag l + 1; pairs beyond kNumGetawayLags (or disconnected) go to lane k % kGroupSize.
  Real lagH[kNumGetawayChannels]    = {};
  Real lagHats[kNumGetawayChannels] = {};
  Real lagR[kNumGetawayChannels]    = {};
  Real lagRMax[kNumGetawayChannels] = {};
  Real farH[kNumGetawayChannels]    = {};
  Real farHats[kNumGetawayChannels] = {};
  Real farR[kNumGetawayChannels]    = {};
  {
    const int32_t* neighborStarts = inputs.bondNeighborStarts + moleculeStart;
    uint8_t*       depth     = bondDepths + atomOffset * kGetawayPairStreams + static_cast<size_t>(stream) * numAtoms;
    const unsigned groupMask = 0xffu << (stream * kGroupSize);
    const uint8_t* moleculeTable = moleculeBondDistances(bondTable, moleculeIdx);
    for (int j = stream; j < numAtoms - 1; j += kGetawayPairStreams) {
      // Row j's bond distances, indexed by k - lagBase: the molecule's table row, or a search into depth.
      const uint8_t* lags    = depth;
      int            lagBase = 0;
      if (moleculeTable != nullptr) {
        lags    = moleculeTable + bondTableRowStart(j, numAtoms);
        lagBase = j + 1;
      } else {
        searchBondDepths<kNumGetawayLags>(neighborStarts,
                                          inputs.bondNeighbors,
                                          numAtoms,
                                          j,
                                          laneInGroup,
                                          groupMask,
                                          depth);
      }
      const Real* rowJ = atomRows + j * kGetawayRowStride;
      for (int k = j + 1; k < numAtoms; ++k) {
        const int  lag   = lags[k - lagBase];
        const bool near  = lag <= kNumGetawayLags;
        const bool owner = near ? lag - 1 == laneInGroup : k % kGroupSize == laneInGroup;
        if (!owner) {
          continue;
        }
        const Real* rowK      = atomRows + k * kGetawayRowStride;
        const Real  hatsTerm  = rowJ[kGetawayRowLeverage] * rowK[kGetawayRowLeverage];
        const Real  influence = rowJ[kGetawayRowY + 0] * rowK[kGetawayRowY + 0] +
                               rowJ[kGetawayRowY + 1] * rowK[kGetawayRowY + 1] +
                               rowJ[kGetawayRowY + 2] * rowK[kGetawayRowY + 2];
        const Real rTerm = rowJ[kGetawayRowRootLev] * rowK[kGetawayRowRootLev] / getawayDistance(rowJ, rowK);
        for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
          const Real product = getawayWeight(rowJ, channel) * getawayWeight(rowK, channel);
          if (near) {
            lagHats[channel] += product * hatsTerm;
            if (influence > Real(0)) {
              lagH[channel] += product * influence;
            }
            const Real weighted = product * rTerm;
            lagR[channel] += weighted;
            if (weighted > lagRMax[channel]) {
              lagRMax[channel] = weighted;
            }
          } else {
            farHats[channel] += product * hatsTerm;
            if (influence > Real(0)) {
              farH[channel] += product * influence;
            }
            farR[channel] += product * rTerm;
          }
        }
      }
    }
  }
  for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
    lagH[channel]    = sumAcrossGroups(lagH[channel]);
    lagHats[channel] = sumAcrossGroups(lagHats[channel]);
    lagR[channel]    = sumAcrossGroups(lagR[channel]);
    lagRMax[channel] = maxAcrossGroups(lagRMax[channel]);
    farH[channel]    = sumAcrossGroups(farH[channel]);
    farHats[channel] = sumAcrossGroups(farHats[channel]);
    farR[channel]    = sumAcrossGroups(farR[channel]);
  }

  // Lag 0: H0 = sum w^2 h and HATS0 = sum w^2 h^2 over atoms with h > 0; HIC from every leverage. Computed
  // after the pair loop, and the H and R blocks written before REIG, so neither set of sums holds registers
  // through the other's loop (168 -> 128 registers in float32).
  Real   lagZeroH[kNumGetawayChannels]    = {};
  Real   lagZeroHats[kNumGetawayChannels] = {};
  double hic                              = 0;
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    const Real*  atomRow  = atomRows + atomIdx * kGetawayRowStride;
    const Real   leverage = atomRow[kGetawayRowLeverage];
    const double exact    = leverages[atomIdx];
    hic -= exact / dimension * log(exact / dimension) / log(2.0);
    if (exact > 0.0) {
      for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
        const Real weight = getawayWeight(atomRow, channel);
        lagZeroH[channel] += weight * weight * leverage;
        lagZeroHats[channel] += weight * weight * leverage * leverage;
      }
    }
  }
  hic = groupAllReduceSum<kWarpSize>(hic);
  for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
    lagZeroH[channel]    = groupAllReduceSum<kWarpSize>(lagZeroH[channel]);
    lagZeroHats[channel] = groupAllReduceSum<kWarpSize>(lagZeroHats[channel]);
  }

  if (stream == 0) {
    for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
      Real* const hBlock     = row + kGetawayHStart + channel * kGetawayHBlock;
      Real* const rBlock     = row + kGetawayRStart + channel * kGetawayRBlock;
      const Real  lagHSum    = groupAllReduceSum(lagH[channel]);
      const Real  lagHatsSum = groupAllReduceSum(lagHats[channel]);
      const Real  lagRSum    = groupAllReduceSum(lagR[channel]);
      const Real  farHSum    = groupAllReduceSum(farH[channel]);
      const Real  farHatsSum = groupAllReduceSum(farHats[channel]);
      const Real  farRSum    = groupAllReduceSum(farR[channel]);
      const Real  rMax =
        cooperative_groups::reduce(laneTile<kGroupSize>(), lagRMax[channel], cooperative_groups::greater<Real>());
      const int lagIdx                     = laneInGroup + 1;
      hBlock[lagIdx]                       = roundThousandths(lagH[channel]);
      hBlock[kNumGetawayLags + 2 + lagIdx] = roundThousandths(lagHats[channel]);
      rBlock[lagIdx - 1]                   = roundThousandths(lagR[channel]);
      rBlock[kNumGetawayLags + lagIdx]     = roundThousandths(lagRMax[channel]);
      if (laneInGroup == 0) {
        hBlock[0]                   = roundThousandths(lagZeroH[channel]);
        hBlock[kNumGetawayLags + 1] = roundThousandths(lagZeroH[channel] + Real(2) * (lagHSum + farHSum));
        hBlock[kNumGetawayLags + 2] = roundThousandths(lagZeroHats[channel]);
        hBlock[kGetawayHBlock - 1]  = roundThousandths(lagZeroHats[channel] + Real(2) * (lagHatsSum + farHatsSum));
        rBlock[kNumGetawayLags]     = roundThousandths(Real(2) * (lagRSum + farRSum));
        rBlock[kGetawayRBlock - 1]  = roundThousandths(rMax);
      }
    }
  }

  // REIG by power iteration on R; the first product with ones gives R's row sums for RARS and RCON.
  Real* const current = iterateScratch + atomOffset * 2;
  Real* const next    = current + numAtoms;
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    current[atomIdx] = Real(1);
  }
  __syncwarp();
  Real numerator;
  Real normSquared;
  multiplyByInfluenceMatrix(atomRows, numAtoms, lane, current, next, numerator, normSquared);
  Real rowSum = 0;
  Real rcon   = 0;
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    rowSum += next[atomIdx];
    for (int edge = inputs.bondNeighborStarts[moleculeStart + atomIdx];
         edge < inputs.bondNeighborStarts[moleculeStart + atomIdx + 1];
         ++edge) {
      const int neighbor = inputs.bondNeighbors[edge];
      if (neighbor > atomIdx) {
        rcon += sqrt(next[atomIdx] * next[neighbor]);
      }
    }
  }
  const Real rars           = groupAllReduceSum<kWarpSize>(rowSum) / static_cast<Real>(numAtoms);
  rcon                      = groupAllReduceSum<kWarpSize>(rcon);
  Real           eigenvalue = numerator / static_cast<Real>(numAtoms);
  constexpr Real kTolerance = sizeof(Real) == sizeof(double) ? Real(1e-13) : Real(2e-7);
  for (int iteration = 0; iteration < kGetawayMaxIterations && normSquared > Real(0); ++iteration) {
    const Real inverseNorm = Real(1) / sqrt(normSquared);
    for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
      current[atomIdx] = next[atomIdx] * inverseNorm;
    }
    __syncwarp();
    multiplyByInfluenceMatrix(atomRows, numAtoms, lane, current, next, numerator, normSquared);
    const Real previous = eigenvalue;
    eigenvalue          = numerator;
    if (fabs(eigenvalue - previous) <= kTolerance * fabs(eigenvalue)) {
      break;
    }
  }
  if (!(normSquared > Real(0))) {
    eigenvalue = Real(0);
  }

  double         ith  = 0;
  double         ish  = 0;
  int64_t* const keys = keyScratch + atomOffset * 2;
  computeInformationIndices(leverages,
                            inputs.heavyAtomFlags + moleculeStart,
                            numAtoms,
                            precisionDigits,
                            lane,
                            sortScratch + atomOffset,
                            keys,
                            keys + numAtoms,
                            ith,
                            ish);
  if (lane == 0) {
    double leverageProduct = 1.0;
    for (int atomIdx = 0; atomIdx < numAtoms; ++atomIdx) {
      leverageProduct *= leverages[atomIdx];
    }
    const double hgm           = 100.0 * pow(leverageProduct, 1.0 / numAtoms);
    row[0]                     = static_cast<Real>(roundThousandths(ith));
    row[1]                     = static_cast<Real>(roundThousandths(ish));
    row[2]                     = static_cast<Real>(roundThousandths(hic));
    row[3]                     = static_cast<Real>(roundThousandths(hgm));
    row[kGetawayRconIndex + 0] = roundThousandths(rcon);
    row[kGetawayRconIndex + 1] = roundThousandths(rars);
    row[kGetawayRconIndex + 2] = roundThousandths(eigenvalue);
  }
}

/**
 * @brief Launches GETAWAY. Molecules with bond-distance table entries that fit the device's opt-in shared
 *        memory run getawayFastKernel() (one CTA per conformer) after getawayMoleculePrepassKernel(); any other
 *        conformer runs getaway3DKernel(), launched only when the batch metadata cannot rule such conformers
 *        out. getaway3DKernel()'s scratch per atom: one kGetawayRowStride row of @p Real, a float64 leverage and
 *        rounded-leverage slot, two ITH/ISH sort keys, two power-iteration values and one bond-depth byte per
 *        pair stream.
 */
template <typename Real>
void launchGetawayProperties(const DeviceCoordView&        coordinates,
                             const Property3DDeviceInputs& inputs,
                             const GetawayOptions&         options,
                             const BondDistanceTable&      bondTable,
                             Real*                         output,
                             const cudaStream_t            stream) {
  const int numConformers = coordinates.numConformers;
  if (numConformers == 0 || output == nullptr) {
    return;
  }
  // Molecules of up to kGetawaySmallMaxAtoms atoms run in small CTAs with small shared memory, so more of them
  // are resident per SM; larger ones run in a second launch with shared memory sized for the largest molecule.
  const int maxAtoms      = inputs.maxMoleculeAtoms;
  int       smallCapacity = 0;
  int       largeCapacity = 0;
  if (bondTable.distances != nullptr && maxAtoms >= 1 && coordinates.nMols > 0) {
    int device = 0;
    cudaCheckError(cudaGetDevice(&device));
    int optin = 0;
    cudaCheckError(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    // Largest capacity (at most limit) whose layout fits; the layout grows monotonically with capacity.
    const auto fit = [optin](const int limit, const int warps) {
      int low  = 0;
      int high = limit;
      while (low < high) {
        const int mid = low + (high - low + 1) / 2;
        if (getawayFastLayout<Real>(mid, warps).total <= static_cast<size_t>(optin)) {
          low = mid;
        } else {
          high = mid - 1;
        }
      }
      return low;
    };
    smallCapacity =
      fit(maxAtoms < kGetawaySmallMaxAtoms ? maxAtoms : kGetawaySmallMaxAtoms, kGetawaySmallBlockSize / kWarpSize);
    largeCapacity = maxAtoms > smallCapacity ? fit(maxAtoms, kGetawayFastBlockSize / kWarpSize) : smallCapacity;
  }
  int        fastCapacity = largeCapacity > smallCapacity ? largeCapacity : smallCapacity;
  // getawayWarpKernel() takes every table-covered conformer of at most kGetawayWarpMaxAtoms atoms.
  const bool warpPath     = fastCapacity >= 1;
  if (warpPath && fastCapacity < kGetawayWarpMaxAtoms) {
    fastCapacity = kGetawayWarpMaxAtoms;
  }
  // Every conformer is on the fast path when every molecule fits and has table entries.
  const bool fastCoversAll = fastCapacity >= 1 && fastCapacity >= maxAtoms && inputs.moleculeAtomPairs > 0 &&
                             inputs.moleculeAtomPairs <= bondTable.capacity;

  AsyncDeviceVector<uint8_t> fastWorkspace;
  if (fastCapacity >= 1) {
    const int    nMols        = coordinates.nMols;
    const size_t mols         = static_cast<size_t>(nMols);
    const size_t conformers   = static_cast<size_t>(numConformers);
    const size_t pairCapacity = static_cast<size_t>(bondTable.capacity > 0 ? bondTable.capacity : 1);

    // One stream-ordered allocation for the five disjoint fast-path arrays. Ordering the double region first
    // leaves every following offset naturally aligned without padding.
    const size_t dimensionsOffset  = 0;
    const size_t lagPairsOffset    = dimensionsOffset + mols * sizeof(double);
    const size_t lagStartsOffset   = lagPairsOffset + pairCapacity * sizeof(uint32_t);
    const size_t queuesOffset      = lagStartsOffset + mols * kGetawayLagStartsStride * sizeof(int32_t);
    const size_t listEntriesOffset = queuesOffset + kGetawayBuckets * sizeof(int);
    const size_t workspaceBytes    = listEntriesOffset + conformers * kGetawayBuckets * sizeof(int);
    fastWorkspace                  = AsyncDeviceVector<uint8_t>(workspaceBytes, stream);
    uint8_t* const  base           = fastWorkspace.data();
    double* const   dimensions     = reinterpret_cast<double*>(base + dimensionsOffset);
    uint32_t* const lagPairs       = reinterpret_cast<uint32_t*>(base + lagPairsOffset);
    int32_t* const  lagStarts      = reinterpret_cast<int32_t*>(base + lagStartsOffset);
    int* const      queues         = reinterpret_cast<int*>(base + queuesOffset);
    int* const      listEntries    = reinterpret_cast<int*>(base + listEntriesOffset);

    cudaCheckError(cudaMemsetAsync(queues, 0, kGetawayBuckets * sizeof(int), stream));
    GetawayWorkLists lists;
    lists.counts          = queues;
    lists.conformers      = listEntries;
    lists.stride          = numConformers;
    lists.smallCapacity   = smallCapacity > kGetawayWarpMaxAtoms ? smallCapacity : kGetawayWarpMaxAtoms;
    lists.largeCapacity   = largeCapacity > smallCapacity ? largeCapacity : lists.smallCapacity;
    const auto classWidth = [](const int span) {
      const int width = (span + kGetawaySizeClasses - 1) / kGetawaySizeClasses;
      return width > 1 ? width : 1;
    };
    lists.smallClassWidth = classWidth(lists.smallCapacity - kGetawayWarpMaxAtoms);
    lists.largeClassWidth = classWidth(lists.largeCapacity - lists.smallCapacity);
    const int routeBlocks = (numConformers + kGetawayPrepassBlock - 1) / kGetawayPrepassBlock;
    getawayMoleculePrepassKernel<<<nMols + routeBlocks, kGetawayPrepassBlock, 0, stream>>>(coordinates,
                                                                                           inputs,
                                                                                           bondTable,
                                                                                           lists,
                                                                                           lagPairs,
                                                                                           lagStarts,
                                                                                           dimensions);
    cudaCheckError(cudaGetLastError());
    // Largest molecules first, with only as many CTAs as can be resident at once, so every CTA starts immediately:
    // the warp path next, then the small-CTA path. Those two run one CTA (warp) per listed conformer, the listed
    // ones first, so CTAs past the list end exit at once.
    int device = 0;
    cudaCheckError(cudaGetDevice(&device));
    int multiprocessors = 0;
    cudaCheckError(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    const int precision = static_cast<int>(options.precision);
    if (largeCapacity > smallCapacity) {
      // As many CTAs as can be resident, each walking the class lists with a static stride.
      const auto   kernel      = getawayFastKernel<Real, kGetawayFastBlockSize>;
      const size_t sharedBytes = getawayFastLayout<Real>(largeCapacity, kGetawayFastBlockSize / kWarpSize).total;
      cudaCheckError(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes)));
      cudaCheckError(cudaFuncSetAttribute(kernel,
                                          cudaFuncAttributePreferredSharedMemoryCarveout,
                                          static_cast<int>(cudaSharedmemCarveoutMaxShared)));
      int perMultiprocessor = 0;
      cudaCheckError(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perMultiprocessor, kernel, kGetawayFastBlockSize, sharedBytes));
      const int64_t resident = static_cast<int64_t>(perMultiprocessor > 0 ? perMultiprocessor : 1) *
                               (multiprocessors > 0 ? multiprocessors : 1);
      const int blocks = static_cast<int>(resident < numConformers ? resident : numConformers);
      kernel<<<blocks, kGetawayFastBlockSize, sharedBytes, stream>>>(coordinates,
                                                                     inputs,
                                                                     precision,
                                                                     largeCapacity,
                                                                     bondTable,
                                                                     lists,
                                                                     kGetawayLargeBucket,
                                                                     static_cast<const uint32_t*>(lagPairs),
                                                                     static_cast<const int32_t*>(lagStarts),
                                                                     static_cast<const double*>(dimensions),
                                                                     output);
      cudaCheckError(cudaGetLastError());
    }
    if (warpPath) {
      const int warpBlocks = (numConformers + kGetawayWarpCtaWarps - 1) / kGetawayWarpCtaWarps;
      getawayWarpKernel<Real>
        <<<warpBlocks, kGetawayWarpCtaWarps * kWarpSize, 0, stream>>>(coordinates,
                                                                      inputs,
                                                                      precision,
                                                                      bondTable,
                                                                      lists,
                                                                      static_cast<const uint32_t*>(lagPairs),
                                                                      static_cast<const int32_t*>(lagStarts),
                                                                      static_cast<const double*>(dimensions),
                                                                      output);
      cudaCheckError(cudaGetLastError());
    }
    if (smallCapacity > kGetawayWarpMaxAtoms) {
      const auto   kernel      = getawayFastKernel<Real, kGetawaySmallBlockSize>;
      const size_t sharedBytes = getawayFastLayout<Real>(smallCapacity, kGetawaySmallBlockSize / kWarpSize).total;
      cudaCheckError(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes)));
      cudaCheckError(cudaFuncSetAttribute(kernel,
                                          cudaFuncAttributePreferredSharedMemoryCarveout,
                                          static_cast<int>(cudaSharedmemCarveoutMaxShared)));
      kernel<<<numConformers, kGetawaySmallBlockSize, sharedBytes, stream>>>(coordinates,
                                                                             inputs,
                                                                             precision,
                                                                             smallCapacity,
                                                                             bondTable,
                                                                             lists,
                                                                             kGetawaySmallBucket,
                                                                             static_cast<const uint32_t*>(lagPairs),
                                                                             static_cast<const int32_t*>(lagStarts),
                                                                             static_cast<const double*>(dimensions),
                                                                             output);
      cudaCheckError(cudaGetLastError());
    }
  }
  if (fastCoversAll) {
    return;
  }

  const size_t               numAtoms = static_cast<size_t>(coordinates.numAtoms);
  AsyncDeviceVector<Real>    rows(numAtoms * kGetawayRowStride, stream);
  AsyncDeviceVector<double>  leverages(numAtoms, stream);
  AsyncDeviceVector<double>  sorted(numAtoms, stream);
  AsyncDeviceVector<int64_t> keys(numAtoms * 2, stream);
  AsyncDeviceVector<Real>    iterates(numAtoms * 2, stream);
  AsyncDeviceVector<uint8_t> depths(numAtoms * kGetawayPairStreams, stream);
  const int                  numBlocks = (numConformers + kWarpsPerBlock - 1) / kWarpsPerBlock;
  getaway3DKernel<Real><<<numBlocks, kBlockSize, 0, stream>>>(coordinates,
                                                              inputs,
                                                              static_cast<int>(options.precision),
                                                              rows.data(),
                                                              leverages.data(),
                                                              sorted.data(),
                                                              keys.data(),
                                                              iterates.data(),
                                                              depths.data(),
                                                              bondTable,
                                                              fastCapacity,
                                                              output);
  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_GETAWAY_CUH
