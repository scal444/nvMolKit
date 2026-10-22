// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_GETAWAY_CUH
#define NVMOLKIT_DESCRIPTORS3D_GETAWAY_CUH

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

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
//! totals.
constexpr int kNumGetawayLags     = 8;

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
 * @brief Per-atom arrays of a conformer's working set, each numAtoms long and written once per conformer as the
 *        compute type: scaled projections y (three arrays, H(j, k) = y_j . y_k), leverage h = H(j, j), sqrt(h), the
 *        atom-property channels, then the centered coordinates.
 */
constexpr int kGetawayRowY        = 0;
constexpr int kGetawayRowLeverage = 3;
constexpr int kGetawayRowRootLev  = 4;
constexpr int kGetawayRowWeights  = 5;
constexpr int kGetawayRowCentered = kGetawayRowWeights + kNumAtomPropertyChannels;
constexpr int kGetawayRowArrays   = kGetawayRowCentered + 3;

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

// ---- One block per conformer, working set in shared memory -----------------------------------------------------
//
// The block keeps the conformer's per-atom arrays and R = sqrt(h_j h_k) / r_jk in shared memory, so every distance
// is computed once instead of on each pass over the pair triangle (the pair sums and every product of REIG's power
// iteration). R is stored once per unordered pair in a circulant layout: slot (d - 1) * n + i holds R(i, (i + d) mod n)
// for d = 1..n/2, so consecutive threads read consecutive slots and vector entries in the power iteration. The lag 1-8
// sums run over per-lag lists of slots built from the molecule's bond distances, one warp per lag, so no thread
// scans pairs at other lags. Molecules whose working set does not fit a block's shared memory run the same kernel
// with the working set in global memory.

//! Threads per GETAWAY block; each block computes one conformer at a time.
constexpr int kGetawayBlockSize = 64;
constexpr int kGetawayWarps     = kGetawayBlockSize / kWarpSize;
static_assert(kGetawayWarps >= 2, "the eigensystem and ITH each run on a warp of their own beside other work");
constexpr int kGetawayGroups = kGetawayBlockSize / kGroupSize;

//! Sums of the pair pass: H, HATS and R over all pairs, then lag-0 H and HATS, per channel.
constexpr int kGetawayTotals = 5 * kNumGetawayChannels;

//! float64 slots of the block's scratch: per-warp partial sums, broadcast scalars (HIC's dimension, the inverse roots,
//! the eigenvectors), then each lag's per-channel H, HATS and R sums and R maximum.
constexpr int kGetawayScratchSums    = 0;
constexpr int kGetawayScratchScalars = kGetawayScratchSums + kGetawayWarps * kGetawayTotals;
constexpr int kGetawayScratchLagSums = kGetawayScratchScalars + 16;
constexpr int kGetawayScratchDoubles = kGetawayScratchLagSums + 4 * kNumGetawayLags * kNumGetawayChannels;
//! Lag-list starts: entry (lag - 1) * kGetawayWarps + warp starts that warp's part of the lag's list; the last is the
//! total.
constexpr int kGetawayListStarts     = kNumGetawayLags * kGetawayWarps + 1;

//! Global-memory working sets allocated at once for molecules too large for shared memory.
constexpr size_t kGetawayWorkspaceBudget = size_t{256} << 20;

//! Byte offsets of the working set of a conformer of numAtoms atoms.
struct GetawayLayout {
  size_t scratch;
  size_t listStarts;
  size_t leverages;
  size_t heavyValues;
  size_t keys;
  size_t rows;
  size_t iterates;
  size_t matrix;
  size_t lagSlots;
  size_t lags;
  size_t depths;
  size_t total;
};

//! Reserves @p size bytes at @p bytes (16-byte aligned) and returns their offset.
__host__ __device__ inline size_t getawayPlace(size_t& bytes, const size_t size) {
  const size_t offset = bytes;
  bytes               = (bytes + size + 15) & ~static_cast<size_t>(15);
  return offset;
}

template <typename Real> __host__ __device__ inline GetawayLayout getawayLayout(const int numAtoms) {
  const size_t  n     = static_cast<size_t>(numAtoms);
  const size_t  pairs = n > 1 ? n * (n - 1) / 2 : 0;
  GetawayLayout layout;
  size_t        bytes = 0;
  layout.scratch      = getawayPlace(bytes, sizeof(double) * kGetawayScratchDoubles);
  layout.listStarts   = getawayPlace(bytes, sizeof(int) * kGetawayListStarts);
  layout.leverages    = getawayPlace(bytes, sizeof(double) * n);
  layout.heavyValues  = getawayPlace(bytes, sizeof(double) * n);
  layout.keys         = getawayPlace(bytes, sizeof(int64_t) * 2 * n);
  layout.rows         = getawayPlace(bytes, sizeof(Real) * kGetawayRowArrays * n);
  layout.iterates     = getawayPlace(bytes, sizeof(Real) * 2 * n);
  layout.matrix       = getawayPlace(bytes, sizeof(Real) * n * (n / 2));
  layout.lagSlots     = getawayPlace(bytes, sizeof(uint32_t) * pairs);
  layout.lags         = getawayPlace(bytes, pairs);
  layout.depths       = getawayPlace(bytes, static_cast<size_t>(kGetawayGroups) * n);
  layout.total        = bytes;
  return layout;
}

//! Block-collective sum of @p values, returned to every thread; @p scratch holds kGetawayWarps * kCount values.
template <int kCount, typename T>
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
    for (int other = 1; other < kGetawayWarps; ++other) {
      sum += scratch[other * kCount + q];
    }
    values[q] = sum;
  }
  __syncthreads();
}

/**
 * @brief Warp-collective transpose reduction of kWarpSize values per lane: lane l returns the combination over the
 *        warp of every lane's @p values[l], a sum, or the maximum for l at or above @p firstMax.
 *
 * Each of the five steps exchanges half of the remaining values with the partner lane, so the reduction takes 31
 * shuffles instead of 5 per value, in a fixed combination order.
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

//! Walks the circulant slots start, start + stride, ... of a conformer of n atoms, tracking slot (d - 1) * n + i's
//! offset d and row i.
struct GetawaySlotWalk {
  int slot;
  int d;
  int i;
  int stepD;
  int stepI;
  int n;

  __device__ GetawaySlotWalk(const int start, const int stride, const int numAtoms)
      : slot(start),
        d(start / numAtoms + 1),
        i(start % numAtoms),
        stepD(stride / numAtoms),
        stepI(stride % numAtoms),
        n(numAtoms) {}

  __device__ void advance(const int stride) {
    slot += stride;
    i += stepI;
    d += stepD;
    if (i >= n) {
      i -= n;
      ++d;
    }
  }

  //! Column (i + d) mod n of the slot.
  __device__ int partner() const { return i + d >= n ? i + d - n : i + d; }

  //! Whether the slot is the one that represents its unordered pair (for even n, offset n / 2 appears twice).
  __device__ bool unique() const { return 2 * d < n || i < d; }
};

//! Index of pair (@p a, @p b), a != b, in a molecule's row-major bond-distance triangle.
__device__ __forceinline__ int getawayPairIndex(const int a, const int b, const int numAtoms) {
  const int j = a < b ? a : b;
  const int k = a < b ? b : a;
  return static_cast<int>(bondTableRowStart(j, numAtoms)) + k - j - 1;
}

/**
 * @brief Row @p i of R times @p vector, from R's circulant slots in @p matrix: forward slots (d - 1) * n + i hold
 *        R(i, i + d) and backward slots (d - 1) * n + (i - d) hold R(i - d, i).
 */
template <typename Real>
__device__ __forceinline__ Real getawayRowProduct(const Real* matrix, const Real* vector, const int n, const int i) {
  Real forward  = 0;
  Real backward = 0;
  int  ahead    = i;
  int  behind   = i;
  for (int d = 1; 2 * d < n; ++d) {
    ahead  = ahead + 1 == n ? 0 : ahead + 1;
    behind = behind == 0 ? n - 1 : behind - 1;
    forward += matrix[(d - 1) * n + i] * vector[ahead];
    backward += matrix[(d - 1) * n + behind] * vector[behind];
  }
  if (n % 2 == 0 && n >= 2) {
    forward += matrix[(n / 2 - 1) * n + i] * vector[ahead + 1 == n ? 0 : ahead + 1];
  }
  return forward + backward;
}

/**
 * @brief Group-collective. HIC's dimension from RDKit's PBF of the conformer at coordinate row @p row: 2 when the PBF
 *        is below kGetawayPlanarPbf (or the conformer is not 3D or has fewer than four atoms), else 3.
 */
__device__ __forceinline__ double getawayDimension(const ConformerAtoms&         atoms,
                                                   const int                     row,
                                                   const Property3DDeviceInputs& inputs,
                                                   const int                     laneInGroup) {
  ProjectionState<double> plane;
  computeProjectionCentroid(atoms, laneInGroup, plane);
  computeProjectionPca(atoms, nullptr, laneInGroup, plane);
  const double pbf  = computePbf(atoms, laneInGroup, plane);
  const bool   is3D = inputs.conformerIs3D == nullptr || inputs.conformerIs3D[row] != 0;
  return (atoms.numAtoms < 4 || !is3D ? 0.0 : pbf) < kGetawayPlanarPbf ? 2.0 : 3.0;
}

/**
 * @brief HIC's dimension of each molecule from its default conformer, as RDKit's PBF(mol) call decides it, one
 *        kGroupSize-lane group per molecule; 0 when the molecule has no valid default conformer, in which case each
 *        conformer uses its own.
 */
__global__ void getawayDimensionKernel(const DeviceCoordView        coordinates,
                                       const Property3DDeviceInputs inputs,
                                       double* __restrict__ dimensions) {
  const int moleculeIdx = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / kGroupSize);
  if (moleculeIdx >= coordinates.nMols) {
    return;  // Uniform across the group.
  }
  const int defaultRow = inputs.defaultConformerRows != nullptr ? inputs.defaultConformerRows[moleculeIdx] : -1;
  const ConformerAtoms atoms =
    defaultRow >= 0 ? loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, defaultRow) : ConformerAtoms{};
  const double dimension =
    atoms.valid ? getawayDimension(atoms, defaultRow, inputs, static_cast<int>(threadIdx.x) % kGroupSize) : 0.0;
  if (threadIdx.x % kGroupSize == 0) {
    dimensions[moleculeIdx] = dimension;
  }
}

/**
 * @brief Block-collective. RDKit's GETAWAY for conformer @p conformerIdx with its working set at @p workspace (see
 *        getawayLayout()), when its atom count is in (@p minAtoms, @p maxAtoms]; invalid conformers get NaN rows
 *        when @p minAtoms is 0.
 *
 * Steps: centroid and second moments (float64); the last warp solves the moments' eigensystem (and computes HIC's
 * dimension when the molecule has none from getawayDimensionKernel()) while the other warps fill R's slots with
 * inverse distances and stage the bond distances (the molecule's table entries, or per-row searches); leverages and
 * per-atom arrays, with lag 0 and HIC; the pair pass, which scales R's slots by sqrt(h_j h_k), sums H, HATS and R over
 * all pairs and counts each warp's pairs at lags 1-8; the lag lists; the lag sums, one warp per lag; then REIG by
 * power iteration on R on the first warp, whose first product (with ones) gives the row sums for RARS and RCON, while
 * the second writes the lag blocks and computes ITH, ISH (computeInformationIndices()) and HGM.
 *
 * FP64 required: leverages decide ITH and ISH through their rounding to `precision` significant digits and RDKit's
 * digit-string clustering, so float32 leverages would move values across rounding boundaries; the pseudo-inverse
 * cutoff decides which axes count for (near-)planar conformers. This is O(atoms) work.
 */
template <typename Real>
__device__ __forceinline__ void getawayConformer(const int                     conformerIdx,
                                                 const DeviceCoordView&        coordinates,
                                                 const Property3DDeviceInputs& inputs,
                                                 const int                     precisionDigits,
                                                 const BondDistanceTable&      bondTable,
                                                 const double* __restrict__ dimensions,
                                                 const int            minAtoms,
                                                 const int            maxAtoms,
                                                 unsigned char* const workspace,
                                                 Real* __restrict__ output) {
  const int            tid   = static_cast<int>(threadIdx.x);
  const int            lane  = tid % kWarpSize;
  const int            warp  = tid / kWarpSize;
  Real* const          row   = output + static_cast<size_t>(conformerIdx) * kNumGetawayProperties;
  const ConformerAtoms atoms = loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, conformerIdx);
  if (!atoms.valid) {
    if (minAtoms == 0) {
      for (int valueIdx = tid; valueIdx < kNumGetawayProperties; valueIdx += kGetawayBlockSize) {
        row[valueIdx] = static_cast<Real>(nan(""));
      }
    }
    return;
  }
  const int n = atoms.numAtoms;
  if (n <= minAtoms || n > maxAtoms) {
    return;  // Uniform across the block; the other launch computes it.
  }
  const int moleculeIdx   = coordinates.molIndices[conformerIdx];
  const int moleculeStart = inputs.moleculeAtomStarts[moleculeIdx];
  const int totalAtoms    = inputs.moleculeAtomStarts[coordinates.nMols];
  const int numPairs      = n * (n - 1) / 2;
  const int numSlots      = n * (n / 2);

  const GetawayLayout layout     = getawayLayout<Real>(n);
  double* const       scratch    = reinterpret_cast<double*>(workspace + layout.scratch);
  Real* const         sumScratch = reinterpret_cast<Real*>(scratch + kGetawayScratchSums);
  double* const       scalars    = scratch + kGetawayScratchScalars;
  Real* const         lagSums    = reinterpret_cast<Real*>(scratch + kGetawayScratchLagSums);
  int* const          listStarts = reinterpret_cast<int*>(workspace + layout.listStarts);
  double* const       leverages  = reinterpret_cast<double*>(workspace + layout.leverages);
  Real* const         rows       = reinterpret_cast<Real*>(workspace + layout.rows);
  Real* const         iterates   = reinterpret_cast<Real*>(workspace + layout.iterates);
  Real* const         matrix     = reinterpret_cast<Real*>(workspace + layout.matrix);
  uint32_t* const     lagSlots   = reinterpret_cast<uint32_t*>(workspace + layout.lagSlots);
  uint8_t* const      lags       = workspace + layout.lags;
  const Real*         y0         = rows + kGetawayRowY * n;
  const Real*         y1         = rows + (kGetawayRowY + 1) * n;
  const Real*         y2         = rows + (kGetawayRowY + 2) * n;
  const Real*         lev        = rows + kGetawayRowLeverage * n;
  const Real*         rootLev    = rows + kGetawayRowRootLev * n;
  const Real*         weights    = rows + kGetawayRowWeights * n;
  Real* const         centered   = rows + kGetawayRowCentered * n;

  // Centroid and second moments in float64.
  double centroid[3] = {0.0, 0.0, 0.0};
  for (int atomIdx = tid; atomIdx < n; atomIdx += kGetawayBlockSize) {
    centroid[0] += atoms.positions[atomIdx * 3 + 0];
    centroid[1] += atoms.positions[atomIdx * 3 + 1];
    centroid[2] += atoms.positions[atomIdx * 3 + 2];
  }
  getawayBlockSum(centroid, scratch, lane, warp);
  const double inverseAtoms = 1.0 / static_cast<double>(n);
  const double centroidX    = centroid[0] * inverseAtoms;
  const double centroidY    = centroid[1] * inverseAtoms;
  const double centroidZ    = centroid[2] * inverseAtoms;
  double       moments[6]   = {};
  for (int atomIdx = tid; atomIdx < n; atomIdx += kGetawayBlockSize) {
    const double x            = atoms.positions[atomIdx * 3 + 0] - centroidX;
    const double y            = atoms.positions[atomIdx * 3 + 1] - centroidY;
    const double z            = atoms.positions[atomIdx * 3 + 2] - centroidZ;
    centered[atomIdx]         = static_cast<Real>(x);
    centered[n + atomIdx]     = static_cast<Real>(y);
    centered[2 * n + atomIdx] = static_cast<Real>(z);
    moments[0] += x * x;
    moments[1] += x * y;
    moments[2] += x * z;
    moments[3] += y * y;
    moments[4] += y * z;
    moments[5] += z * z;
  }
  getawayBlockSum(moments, scratch, lane, warp);

  if (warp == kGetawayWarps - 1) {
    // HIC's dimension: the molecule's (getawayDimensionKernel()), or this conformer's own when the molecule's default
    // conformer is not valid.
    double dimension = dimensions[moleculeIdx];
    if (!(dimension > 0.0)) {
      dimension = getawayDimension(atoms, conformerIdx, inputs, lane % kGroupSize);
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
    if (lane == 0) {
      scalars[0] = dimension;
      for (int axis = 0; axis < 3; ++axis) {
        const double singular = fabs(eigenvalues[axis]);
        scalars[1 + axis]     = singular > kGetawayPinvTolerance ? 1.0 / sqrt(singular) : 0.0;
      }
      for (int entry = 0; entry < 9; ++entry) {
        scalars[4 + entry] = eigenvectors[entry];
      }
    }
  } else {
    // Inverse distances into R's slots, then the bond distances of every pair.
    constexpr int kThreads = (kGetawayWarps - 1) * kWarpSize;
    for (GetawaySlotWalk walk(tid, kThreads, n); walk.slot < numSlots; walk.advance(kThreads)) {
      const int  k      = walk.partner();
      const Real dx     = centered[walk.i] - centered[k];
      const Real dy     = centered[n + walk.i] - centered[n + k];
      const Real dz     = centered[2 * n + walk.i] - centered[2 * n + k];
      matrix[walk.slot] = Real(1) / sqrt(dx * dx + dy * dy + dz * dz);
    }
    const uint8_t* table = moleculeBondDistances(bondTable, moleculeIdx);
    if (table != nullptr) {
      for (int pair = tid; pair < numPairs; pair += kThreads) {
        lags[pair] = table[pair];
      }
    } else {
      constexpr int  kSearchGroups = kThreads / kGroupSize;
      const int      group         = tid / kGroupSize;
      const unsigned groupMask     = 0xffu << (lane / kGroupSize * kGroupSize);
      uint8_t* const depth         = workspace + layout.depths + static_cast<size_t>(group) * n;
      for (int j = group; j < n - 1; j += kSearchGroups) {
        searchBondDepths<kNumGetawayLags>(inputs.bondNeighborStarts + moleculeStart,
                                          inputs.bondNeighbors,
                                          n,
                                          j,
                                          tid % kGroupSize,
                                          groupMask,
                                          depth);
        uint8_t* const rowLags = lags + bondTableRowStart(j, n) - j - 1;
        for (int k = j + 1 + tid % kGroupSize; k < n; k += kGroupSize) {
          rowLags[k] = depth[k];
        }
      }
    }
  }
  __syncthreads();

  // Leverages and per-atom arrays, lag 0 (H0, HATS0 over atoms with h > 0) and HIC; the power iteration's first
  // iterate is all ones.
  const double dimension = scalars[0];
  double       inverseRoots[3];
  double       eigenvectors[9];
  for (int axis = 0; axis < 3; ++axis) {
    inverseRoots[axis] = scalars[1 + axis];
  }
  for (int entry = 0; entry < 9; ++entry) {
    eigenvectors[entry] = scalars[4 + entry];
  }
  Real   sums[kGetawayTotals] = {};
  double hic                  = 0;
  for (int atomIdx = tid; atomIdx < n; atomIdx += kGetawayBlockSize) {
    const double x        = atoms.positions[atomIdx * 3 + 0] - centroidX;
    const double y        = atoms.positions[atomIdx * 3 + 1] - centroidY;
    const double z        = atoms.positions[atomIdx * 3 + 2] - centroidZ;
    double       leverage = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const double scaled =
        (x * eigenvectors[axis] + y * eigenvectors[3 + axis] + z * eigenvectors[6 + axis]) * inverseRoots[axis];
      rows[(kGetawayRowY + axis) * n + atomIdx] = static_cast<Real>(scaled);
      leverage += scaled * scaled;
    }
    leverages[atomIdx]                      = leverage;
    const Real realLeverage                 = static_cast<Real>(leverage);
    rows[kGetawayRowLeverage * n + atomIdx] = realLeverage;
    rows[kGetawayRowRootLev * n + atomIdx]  = static_cast<Real>(sqrt(leverage));
    iterates[atomIdx]                       = Real(1);
    hic -= leverage / dimension * log(leverage / dimension) / log(2.0);
#pragma unroll
    for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
      Real weight = Real(1);
      if (channel > 0) {
        weight = static_cast<Real>(
          inputs.atomPropertyWeights[static_cast<size_t>(channel - 1) * totalAtoms + moleculeStart + atomIdx]);
        rows[(kGetawayRowWeights + channel - 1) * n + atomIdx] = weight;
      }
      if (leverage > 0.0) {
        sums[3 * kNumGetawayChannels + channel] += weight * weight * realLeverage;
        sums[4 * kNumGetawayChannels + channel] += weight * weight * realLeverage * realLeverage;
      }
    }
  }
  __syncthreads();

  // Pair pass: R's slots scaled to sqrt(h_j h_k) / r_jk; each unordered pair enters the totals once and is counted at
  // its lag. Warps walk whole rounds of slots so the lag ballots see every lane.
  int lagCounts[kNumGetawayLags] = {};
  for (GetawaySlotWalk walk(tid, kGetawayBlockSize, n); walk.slot - lane < numSlots; walk.advance(kGetawayBlockSize)) {
    int lag = 0;
    if (walk.slot < numSlots) {
      const int  i      = walk.i;
      const int  k      = walk.partner();
      const Real rTerm  = rootLev[i] * rootLev[k] * matrix[walk.slot];
      matrix[walk.slot] = rTerm;
      if (walk.unique()) {
        lag                  = lags[getawayPairIndex(i, k, n)];
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
    }
#pragma unroll
    for (int l = 0; l < kNumGetawayLags; ++l) {
      lagCounts[l] += __popc(__ballot_sync(0xffffffffu, lag == l + 1));
    }
  }
  if (lane == 0) {
    for (int l = 0; l < kNumGetawayLags; ++l) {
      listStarts[l * kGetawayWarps + warp] = lagCounts[l];
    }
  }
  {
    double hicSum[1] = {hic};
    getawayBlockSum(hicSum, scratch, lane, warp);
    hic = hicSum[0];
  }
  getawayBlockSum(sums, sumScratch, lane, warp);
  if (tid == 0) {
    int start = 0;
    for (int entry = 0; entry < kGetawayListStarts; ++entry) {
      const int count   = entry < kGetawayListStarts - 1 ? listStarts[entry] : 0;
      listStarts[entry] = start;
      start += count;
    }
  }
  if (tid < kNumGetawayChannels) {
    const int   channel         = tid;
    Real* const hBlock          = row + kGetawayHStart + channel * kGetawayHBlock;
    Real* const rBlock          = row + kGetawayRStart + channel * kGetawayRBlock;
    const Real  lagZeroH        = sums[3 * kNumGetawayChannels + channel];
    const Real  lagZeroHats     = sums[4 * kNumGetawayChannels + channel];
    hBlock[0]                   = roundThousandths(lagZeroH);
    hBlock[kNumGetawayLags + 1] = roundThousandths(lagZeroH + Real(2) * sums[channel]);
    hBlock[kNumGetawayLags + 2] = roundThousandths(lagZeroHats);
    hBlock[kGetawayHBlock - 1]  = roundThousandths(lagZeroHats + Real(2) * sums[kNumGetawayChannels + channel]);
    rBlock[kNumGetawayLags]     = roundThousandths(Real(2) * sums[2 * kNumGetawayChannels + channel]);
  }
  __syncthreads();

  // Lag lists: each warp writes its pairs of each lag, in walk order, from its start in listStarts. Entries pack the
  // slot's row i and offset d.
  {
    int position[kNumGetawayLags];
#pragma unroll
    for (int l = 0; l < kNumGetawayLags; ++l) {
      position[l] = listStarts[l * kGetawayWarps + warp];
    }
    const unsigned lanesBefore = (1u << lane) - 1u;
    for (GetawaySlotWalk walk(tid, kGetawayBlockSize, n); walk.slot - lane < numSlots;
         walk.advance(kGetawayBlockSize)) {
      const int lag = walk.slot < numSlots && walk.unique() ? lags[getawayPairIndex(walk.i, walk.partner(), n)] : 0;
#pragma unroll
      for (int l = 0; l < kNumGetawayLags; ++l) {
        const unsigned members = __ballot_sync(0xffffffffu, lag == l + 1);
        if (lag == l + 1) {
          lagSlots[position[l] + __popc(members & lanesBefore)] =
            static_cast<uint32_t>(walk.i) | (static_cast<uint32_t>(walk.d) << 16);
        }
        position[l] += __popc(members);
      }
    }
  }
  __syncthreads();

  // Lags 1-8 from the lag lists, one warp per lag.
  for (int lag = warp + 1; lag <= kNumGetawayLags; lag += kGetawayWarps) {
    Real      lagH[kNumGetawayChannels]    = {};
    Real      lagHats[kNumGetawayChannels] = {};
    Real      lagR[kNumGetawayChannels]    = {};
    Real      lagRMax[kNumGetawayChannels] = {};
    const int stop                         = listStarts[lag * kGetawayWarps];
    for (int entry = listStarts[(lag - 1) * kGetawayWarps] + lane; entry < stop; entry += kWarpSize) {
      const uint32_t packed    = lagSlots[entry];
      const int      i         = static_cast<int>(packed & 0xffffu);
      const int      d         = static_cast<int>(packed >> 16);
      const int      k         = i + d >= n ? i + d - n : i + d;
      const Real     rTerm     = matrix[(d - 1) * n + i];
      const Real     hatsTerm  = lev[i] * lev[k];
      const Real     influence = y0[i] * y0[k] + y1[i] * y1[k] + y2[i] * y2[k];
#pragma unroll
      for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
        const Real product = channel == 0 ? Real(1) : weights[(channel - 1) * n + i] * weights[(channel - 1) * n + k];
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
    // Lane q * 7 + c receives quantity q (H, HATS, R, R maximum) of channel c.
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
      lagSums[(lane / kNumGetawayChannels * kNumGetawayLags + lag - 1) * kNumGetawayChannels +
              lane % kNumGetawayChannels] = total;
    }
  }

  // Every lag's sums are in lagSums. REIG by power iteration on R runs on the first warp while the second computes
  // ITH, ISH, HGM and the H and R blocks. The power iteration's first product (with ones) gives R's row sums for RARS
  // and RCON; products are taken with the unnormalized iterate and scaled after.
  __syncthreads();
  if (warp == 0) {
    Real rowSums     = 0;
    Real normSquared = 0;
    for (int atomIdx = lane; atomIdx < n; atomIdx += kWarpSize) {
      const Real value      = getawayRowProduct(matrix, iterates, n, atomIdx);
      iterates[n + atomIdx] = value;
      rowSums += value;
      normSquared += value * value;
    }
    rowSums     = groupAllReduceSum<kWarpSize>(rowSums);
    normSquared = groupAllReduceSum<kWarpSize>(normSquared);
    __syncwarp();
    Real rcon = 0;
    for (int atomIdx = lane; atomIdx < n; atomIdx += kWarpSize) {
      for (int edge = inputs.bondNeighborStarts[moleculeStart + atomIdx];
           edge < inputs.bondNeighborStarts[moleculeStart + atomIdx + 1];
           ++edge) {
        const int neighbor = inputs.bondNeighbors[edge];
        if (neighbor > atomIdx) {
          rcon += sqrt(iterates[n + atomIdx] * iterates[n + neighbor]);
        }
      }
    }
    rcon                      = groupAllReduceSum<kWarpSize>(rcon);
    const Real     rars       = rowSums / static_cast<Real>(n);
    Real           eigenvalue = rowSums / static_cast<Real>(n);
    constexpr Real kTolerance = sizeof(Real) == sizeof(double) ? Real(1e-13) : Real(2e-7);
    int            source     = 1;
    for (int iteration = 0; iteration < kGetawayMaxIterations && normSquared > Real(0); ++iteration) {
      const Real  inverseNorm = Real(1) / sqrt(normSquared);
      const Real* current     = iterates + source * n;
      Real* const product     = iterates + (1 - source) * n;
      Real        numerator   = 0;
      normSquared             = 0;
      for (int atomIdx = lane; atomIdx < n; atomIdx += kWarpSize) {
        const Real value = getawayRowProduct(matrix, current, n, atomIdx) * inverseNorm;
        product[atomIdx] = value;
        numerator += current[atomIdx] * inverseNorm * value;
        normSquared += value * value;
      }
      numerator   = groupAllReduceSum<kWarpSize>(numerator);
      normSquared = groupAllReduceSum<kWarpSize>(normSquared);
      __syncwarp();
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
    if (lane == 0) {
      row[kGetawayRconIndex + 0] = roundThousandths(rcon);
      row[kGetawayRconIndex + 1] = roundThousandths(rars);
      row[kGetawayRconIndex + 2] = roundThousandths(eigenvalue);
    }
  } else if (warp == 1) {
    if (lane < kNumGetawayChannels) {
      // Lags 1-8 and RT+, the largest lag maximum.
      const int   channel = lane;
      Real* const hBlock  = row + kGetawayHStart + channel * kGetawayHBlock;
      Real* const rBlock  = row + kGetawayRStart + channel * kGetawayRBlock;
      Real        rMax    = 0;
      for (int lag = 1; lag <= kNumGetawayLags; ++lag) {
        const Real* lagValues             = lagSums + (lag - 1) * kNumGetawayChannels + channel;
        const Real  lagRMax               = lagValues[3 * kNumGetawayLags * kNumGetawayChannels];
        hBlock[lag]                       = roundThousandths(lagValues[0]);
        hBlock[kNumGetawayLags + 2 + lag] = roundThousandths(lagValues[kNumGetawayLags * kNumGetawayChannels]);
        rBlock[lag - 1]                   = roundThousandths(lagValues[2 * kNumGetawayLags * kNumGetawayChannels]);
        rBlock[kNumGetawayLags + lag]     = roundThousandths(lagRMax);
        rMax                              = lagRMax > rMax ? lagRMax : rMax;
      }
      rBlock[kGetawayRBlock - 1] = roundThousandths(rMax);
    }
    double         ith  = 0;
    double         ish  = 0;
    int64_t* const keys = reinterpret_cast<int64_t*>(workspace + layout.keys);
    computeInformationIndices(leverages,
                              inputs.heavyAtomFlags + moleculeStart,
                              n,
                              precisionDigits,
                              lane,
                              reinterpret_cast<double*>(workspace + layout.heavyValues),
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
}

/**
 * @brief RDKit's GETAWAY, one block per conformer (getawayConformer()). With @p globalWorkspace null each block's
 *        working set is its dynamic shared memory; otherwise block b uses @p workspaceBytes bytes at
 *        @p globalWorkspace + b * @p workspaceBytes and strides over the conformers.
 */
template <typename Real>
__global__ void getaway3DKernel(const DeviceCoordView        coordinates,
                                const Property3DDeviceInputs inputs,
                                const int                    precisionDigits,
                                const BondDistanceTable      bondTable,
                                const double* __restrict__ dimensions,
                                const int minAtoms,
                                const int maxAtoms,
                                unsigned char* __restrict__ globalWorkspace,
                                const size_t workspaceBytes,
                                Real* __restrict__ output) {
  extern __shared__ __align__(16) unsigned char getawayShared[];
  unsigned char* const                          workspace =
    globalWorkspace != nullptr ? globalWorkspace + blockIdx.x * workspaceBytes : getawayShared;
  for (int conformerIdx = static_cast<int>(blockIdx.x); conformerIdx < coordinates.numConformers;
       conformerIdx += static_cast<int>(gridDim.x)) {
    __syncthreads();  // The previous conformer is done with the working set.
    getawayConformer(conformerIdx,
                     coordinates,
                     inputs,
                     precisionDigits,
                     bondTable,
                     dimensions,
                     minAtoms,
                     maxAtoms,
                     workspace,
                     output);
  }
}

/**
 * @brief Launches the GETAWAY kernel: conformers whose working set (getawayLayout()) fits a block's shared memory
 *        there, one block each, and any larger ones with their working sets in global memory, as many blocks at a
 *        time as kGetawayWorkspaceBudget allows (at least one).
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
  int maxAtoms = inputs.maxMoleculeAtoms;
  if (maxAtoms <= 0) {
    // Unknown largest molecule: read the atom ranges back.
    std::vector<int32_t> starts(static_cast<size_t>(coordinates.nMols) + 1);
    cudaCheckError(cudaMemcpyAsync(starts.data(),
                                   inputs.moleculeAtomStarts,
                                   starts.size() * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost,
                                   stream));
    cudaCheckError(cudaStreamSynchronize(stream));
    for (size_t moleculeIdx = 0; moleculeIdx + 1 < starts.size(); ++moleculeIdx) {
      maxAtoms = std::max(maxAtoms, starts[moleculeIdx + 1] - starts[moleculeIdx]);
    }
  }
  AsyncDeviceVector<double> dimensions(static_cast<size_t>(coordinates.nMols), stream);
  if (coordinates.nMols > 0) {
    getawayDimensionKernel<<<(coordinates.nMols + kConformersPerBlock - 1) / kConformersPerBlock,
                             kBlockSize,
                             0,
                             stream>>>(coordinates, inputs, dimensions.data());
    cudaCheckError(cudaGetLastError());
  }
  int device = 0;
  cudaCheckError(cudaGetDevice(&device));
  int sharedLimit = 0;
  cudaCheckError(cudaDeviceGetAttribute(&sharedLimit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  int sharedAtoms = std::max(maxAtoms, 1);
  while (sharedAtoms > 1 && getawayLayout<Real>(sharedAtoms).total > static_cast<size_t>(sharedLimit)) {
    --sharedAtoms;
  }
  const size_t sharedBytes = getawayLayout<Real>(sharedAtoms).total;
  cudaCheckError(cudaFuncSetAttribute(getaway3DKernel<Real>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(sharedBytes)));
  const int precisionDigits = static_cast<int>(options.precision);
  getaway3DKernel<Real><<<numConformers, kGetawayBlockSize, sharedBytes, stream>>>(coordinates,
                                                                                   inputs,
                                                                                   precisionDigits,
                                                                                   bondTable,
                                                                                   dimensions.data(),
                                                                                   0,
                                                                                   sharedAtoms,
                                                                                   nullptr,
                                                                                   0,
                                                                                   output);
  cudaCheckError(cudaGetLastError());
  if (sharedAtoms >= maxAtoms) {
    return;
  }
  const size_t workspaceBytes = getawayLayout<Real>(maxAtoms).total;
  int          numSms         = 0;
  cudaCheckError(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device));
  const size_t budgetBlocks = std::max<size_t>(kGetawayWorkspaceBudget / workspaceBytes, 1);
  const int    numBlocks    = static_cast<int>(
    std::min<size_t>({static_cast<size_t>(numConformers), static_cast<size_t>(2 * numSms), budgetBlocks}));
  AsyncDeviceVector<uint8_t> workspace(static_cast<size_t>(numBlocks) * workspaceBytes, stream);
  getaway3DKernel<Real><<<numBlocks, kGetawayBlockSize, 0, stream>>>(coordinates,
                                                                     inputs,
                                                                     precisionDigits,
                                                                     bondTable,
                                                                     dimensions.data(),
                                                                     sharedAtoms,
                                                                     maxAtoms,
                                                                     workspace.data(),
                                                                     workspaceBytes,
                                                                     output);
  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_GETAWAY_CUH
