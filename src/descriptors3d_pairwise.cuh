// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_PAIRWISE_CUH
#define NVMOLKIT_DESCRIPTORS3D_PAIRWISE_CUH

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "src/descriptors3d.h"
#include "src/descriptors3d_kernel.cuh"
#include "src/descriptors3d_topology.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"

namespace nvMolKit::descriptors3d_detail {

//! Unweighted channel plus the atom-property channels, in RDKit's output order (u, m, v, e, p, i, s).
constexpr int kNumPairwiseChannels = kNumAtomPropertyChannels + 1;
constexpr int kNumRdfRadii         = kNumRdfProperties / kNumPairwiseChannels;
constexpr int kNumMorseScatterings = kNumMorseProperties / kNumPairwiseChannels;
//! AUTOCORR3D adds RDKit's relative covalent radius after the pairwise channels (u, m, v, e, p, i, s, r).
constexpr int kNumAutocorrChannels = kNumPairwiseChannels + 1;
constexpr int kNumAutocorrLags     = kNumAutocorr3DProperties / kNumAutocorrChannels;
static_assert(kNumRdfRadii <= kNumMorseScatterings && kNumMorseScatterings == kWarpSize);
static_assert(kNumAutocorrChannels == kGroupSize);

//! Pairwise properties served by one kernel instantiation; unrequested ones compile out.
enum PairwiseSet : unsigned {
  kPairwiseRdf        = 1u,
  kPairwiseMorse      = 2u,
  kPairwiseAutocorr3D = 4u,
};

/**
 * Lanes sharing one pair record for RDF and MORSE; each owns kNumMorseScatterings / lanes bins of all seven channels.
 * Float RDF alone gives each lane eight bins, which keeps four times as many sums per record load. Every other set
 * gives it four, whose smaller sums leave the registers for a fifth or sixth block per SM; double lanes own four bins
 * as well (one MMA fragment per eight bins).
 */
template <typename Real, unsigned kSet> __host__ __device__ constexpr int binLanes() {
  return std::is_same_v<Real, float> && kSet == kPairwiseRdf ? 4 : 8;
}

//! One block per (conformer, slice); the last slice of a split conformer to finish combines the slices' sums.
constexpr int                          kPairwiseThreads   = 128;
constexpr int                          kPairwiseWarps     = kPairwiseThreads / kWarpSize;
constexpr int                          kPairwiseGroups    = kPairwiseThreads / kGroupSize;
//! Most blocks a conformer splits over; double's heavier pairs warrant finer splits.
template <typename Real> constexpr int kPairwiseMaxSlices = std::is_same_v<Real, float> ? 4 : 8;
//! Shared memory for atom rows; conformers with more atoms read their rows from global scratch.
constexpr size_t                       kPairwiseRowBudget = size_t{96} << 10;
//! Shared memory for a molecule's bond-distance entries (one byte per pair); larger molecules read the table.
constexpr int                          kPairwiseLagBudget = 8 << 10;
//! Fewest kWarpSize-pair batches a slice takes before another of the conformer's blocks joins it.
constexpr int                          kMinSliceBatches   = 24;

/**
 * Per-atom row written once per conformer: centered x, y, z, the atom-property channels (the last one MORSE's
 * I-state), RDF's I-state, then AUTOCORR3D's covalent radius, all as the compute type. The stride keeps rows on
 * 16-byte vectors and the rows read by a quarter warp on distinct banks.
 */
constexpr int                          kRowWeights        = 3;
constexpr int                          kRowIStateDrag     = kRowWeights + kNumAtomPropertyChannels;
constexpr int                          kRowCovalentRadius = kRowIStateDrag + 1;
constexpr int                          kRowValues         = 12;
template <typename Real> constexpr int kRowStride         = std::is_same_v<Real, float> ? 12 : 14;
static_assert(kRowCovalentRadius < kRowValues);

//! Values of @p Real per 16-byte vector.
template <typename Real> constexpr int kVector             = 16 / static_cast<int>(sizeof(Real));
//! Consecutive record slots sharing a swizzle: one MMA step's records.
constexpr int                          kMmaPairsPerSwizzle = 4;

//! Loads @p kCount values (a whole number of 16-byte vectors) from 16-byte aligned memory.
template <int kCount, typename Real, int kSize>
__device__ __forceinline__ void loadValues(const Real* source, Real (&values)[kSize]) {
  static_assert(kCount <= kSize && kCount % kVector<Real> == 0);
#pragma unroll
  for (int value = 0; value < kCount; value += kVector<Real>) {
    if constexpr (std::is_same_v<Real, float>) {
      const float4 vector = *reinterpret_cast<const float4*>(source + value);
      values[value]       = vector.x;
      values[value + 1]   = vector.y;
      values[value + 2]   = vector.z;
      values[value + 3]   = vector.w;
    } else {
      const double2 vector = *reinterpret_cast<const double2*>(source + value);
      values[value]        = vector.x;
      values[value + 1]    = vector.y;
    }
  }
}

//! Stores @p kCount values (a whole number of 16-byte vectors) to 16-byte aligned memory.
template <int kCount, typename Real, int kSize>
__device__ __forceinline__ void storeValues(Real* target, const Real (&values)[kSize]) {
  static_assert(kCount <= kSize && kCount % kVector<Real> == 0);
#pragma unroll
  for (int value = 0; value < kCount; value += kVector<Real>) {
    if constexpr (std::is_same_v<Real, float>) {
      *reinterpret_cast<float4*>(target + value) =
        make_float4(values[value], values[value + 1], values[value + 2], values[value + 3]);
    } else {
      *reinterpret_cast<double2*>(target + value) = make_double2(values[value], values[value + 1]);
    }
  }
}

//! Stores @p kCount zeros (a whole number of 16-byte vectors) to 16-byte aligned memory.
template <int kCount, typename Real> __device__ __forceinline__ void storeZeros(Real* target) {
  static_assert(kCount % kVector<Real> == 0);
#pragma unroll
  for (int value = 0; value < kCount; value += kVector<Real>) {
    if constexpr (std::is_same_v<Real, float>) {
      *reinterpret_cast<float4*>(target + value) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
      *reinterpret_cast<double2*>(target + value) = make_double2(0.0, 0.0);
    }
  }
}

/**
 * Double records are swizzled: 16-byte vector v of the record in slot s is stored at vector
 * v ^ ((s / 4) % 2), so the eight lanes of a quarter warp storing their records at once hit distinct banks. The
 * four records an MMA step reads share @p swizzle = (step % 2). Float records are not swizzled.
 */
template <typename Real> __device__ __forceinline__ int recordIndex(const int value, const int swizzle) {
  if constexpr (std::is_same_v<Real, float>) {
    return value;
  } else {
    return value ^ (swizzle << 1);
  }
}

//! Swizzle of the record in slot @p slot; see recordIndex().
__device__ __forceinline__ int recordSwizzle(const int slot) {
  return (slot / kMmaPairsPerSwizzle) & 1;
}

//! Stores @p kCount doubles of a record swizzled by @p swizzle; see recordIndex().
template <int kCount, int kSize>
__device__ __forceinline__ void storeSwizzled(double* target, const double (&values)[kSize], const int swizzle) {
  static_assert(kCount <= kSize && kCount % 2 == 0);
#pragma unroll
  for (int vector = 0; vector < kCount / 2; ++vector) {
    *reinterpret_cast<double2*>(target + 2 * (vector ^ swizzle)) =
      make_double2(values[2 * vector], values[2 * vector + 1]);
  }
}

//! An int carried in a record slot of the compute type.
template <typename Real> __device__ __forceinline__ Real intAsReal(const int value) {
  if constexpr (std::is_same_v<Real, float>) {
    return __int_as_float(value);
  } else {
    return __longlong_as_double(static_cast<long long>(value));
  }
}
template <typename Real> __device__ __forceinline__ int realAsInt(const Real value) {
  if constexpr (std::is_same_v<Real, float>) {
    return __float_as_int(value);
  } else {
    return static_cast<int>(__double_as_longlong(value));
  }
}

/**
 * @brief acc0 += a0 * b0 and acc1 += a1 * b1, each one fused multiply-add; for float, one paired instruction.
 */
template <typename Real>
__device__ __forceinline__ void fmaPair(Real&      acc0,
                                        Real&      acc1,
                                        const Real a0,
                                        const Real a1,
                                        const Real b0,
                                        const Real b1) {
  if constexpr (std::is_same_v<Real, float>) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    const float2 sum = __ffma2_rn(make_float2(a0, a1), make_float2(b0, b1), make_float2(acc0, acc1));
    acc0             = sum.x;
    acc1             = sum.y;
#else
    acc0 = fmaf(a0, b0, acc0);
    acc1 = fmaf(a1, b1, acc1);
#endif
  } else {
    acc0 = fma(a0, b0, acc0);
    acc1 = fma(a1, b1, acc1);
  }
}

//! acc[slot][channel] += products[channel] * terms[slot] for every bin slot and channel.
template <typename Real, int kSlots>
__device__ __forceinline__ void accumulateOuterProduct(Real (&acc)[kSlots][kNumPairwiseChannels],
                                                       const Real (&products)[kNumPairwiseChannels],
                                                       const Real (&terms)[kSlots]) {
  static_assert(kSlots % 2 == 0);
#pragma unroll
  for (int slot = 0; slot < kSlots; slot += 2) {
#pragma unroll
    for (int channel = 0; channel < kNumPairwiseChannels; ++channel) {
      fmaPair(acc[slot][channel],
              acc[slot + 1][channel],
              products[channel],
              products[channel],
              terms[slot],
              terms[slot + 1]);
    }
  }
}

/**
 * @brief Pair p of a conformer's n (n - 1) / 2 atom pairs, enumerated circulantly so every row has the same length:
 *        row j holds pairs (j, j + 1 + t mod n) for t < (n - 1) / 2, then, for even n, pairs (j, j + n / 2) for
 *        j < n / 2 follow all rows.
 */
//! A lane's pair and, while the pair is circulant, its row and column packed as row << 16 | column. The cursor
//! advances by one kWarpSize-pair batch using integer additions instead of decoding the quotient again.
struct PairCursor {
  int pair      = 0;
  int rowColumn = 0;
};

struct PairIndexing {
  int   numAtoms         = 0;
  int   rowLength        = 0;
  int   circulantPairs   = 0;
  float inverseRowLength = 0.0f;
  int   batchStep        = 0;  //!< kWarpSize pairs as whole rows plus columns, packed like PairCursor::rowColumn.

  __device__ __forceinline__ explicit PairIndexing(const int atoms)
      : numAtoms(atoms),
        rowLength((atoms - 1) / 2),
        circulantPairs(atoms * ((atoms - 1) / 2)),
        inverseRowLength(atoms > 2 ? 1.0f / static_cast<float>((atoms - 1) / 2) : 0.0f),
        batchStep(atoms > 2 ? ((kWarpSize / ((atoms - 1) / 2)) << 16) | (kWarpSize % ((atoms - 1) / 2)) : 0) {}

  __device__ __forceinline__ PairCursor cursor(const int pair) const {
    PairCursor at;
    at.pair = pair;
    if (pair < circulantPairs) {
      // A float quotient, off by at most one, corrected to the exact one.
      int row      = static_cast<int>(static_cast<float>(pair) * inverseRowLength);
      int column   = pair - row * rowLength;
      row          = column < 0 ? row - 1 : (column >= rowLength ? row + 1 : row);
      column       = column < 0 ? column + rowLength : (column >= rowLength ? column - rowLength : column);
      at.rowColumn = (row << 16) | column;
    }
    return at;
  }

  __device__ __forceinline__ PairCursor advance(PairCursor at) const {
    at.pair += kWarpSize;
    at.rowColumn += batchStep;
    at.rowColumn += (at.rowColumn & 0xffff) >= rowLength ? (1 << 16) - rowLength : 0;
    return at;
  }

  __device__ __forceinline__ void atomsAt(const PairCursor& at, int& j, int& k) const {
    if (at.pair < circulantPairs) {
      j = at.rowColumn >> 16;
      k = j + 1 + (at.rowColumn & 0xffff);
      k = k >= numAtoms ? k - numAtoms : k;
    } else {
      j = at.pair - circulantPairs;
      k = j + numAtoms / 2;
    }
  }
};

//! What every pass over a conformer's atom pairs reads: its rows, its slice of the pairs and its bond distances.
template <typename Real> struct PairwiseConformer {
  const Real*    rows = nullptr;  //!< kRowStride<Real> values per atom.
  PairIndexing   indexing{0};
  int            firstPair = 0;
  int            lastPair  = 0;
  //! The molecule's bond-distance entries (in shared memory when they fit), or null to search per row.
  const uint8_t* lagTable  = nullptr;
  //! Whether one warp takes every batch (a packed small conformer) rather than sharing them with the block.
  bool           solo      = false;
};

//! Pair (j, k)'s geometry: distance and, for float rows, each atom's leading row values.
template <typename Real, int kValues> struct PairGeometry {
  Real first[kValues];
  Real second[kValues];
  Real distance;
};

template <typename Real, int kValues>
__device__ __forceinline__ PairGeometry<Real, kValues> pairGeometry(const Real* rows, const int j, const int k) {
  PairGeometry<Real, kValues> pair;
  loadValues<kValues>(rows + j * kRowStride<Real>, pair.first);
  loadValues<kValues>(rows + k * kRowStride<Real>, pair.second);
  const Real dx = pair.first[0] - pair.second[0];
  const Real dy = pair.first[1] - pair.second[1];
  const Real dz = pair.first[2] - pair.second[2];
  pair.distance = sqrt(dx * dx + dy * dy + dz * dz);
  return pair;
}

//! Records one double m8n8k4 MMA step takes: its four K columns are four atom pairs.
constexpr int kMmaPairs = 4;

/**
 * @brief acc += A B for one float64 m8n8k4 MMA tile: lane l supplies A(l / 4, l % 4) = @p a and B(l % 4, l / 4) = @p b,
 *        and holds D(l / 4, 2 (l % 4) + i) in acc[i].
 */
__device__ __forceinline__ void mmaAccumulate(double (&acc)[2], const double a, const double b) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%0, %1};"
               : "+d"(acc[0]), "+d"(acc[1])
               : "d"(a), "d"(b));
}

/**
 * @brief MORSE bins 1-31: bin s adds w_j * w_k * sin(s r) / (s r) per channel. Records hold the weight products over
 *        r; the 1 / s factor is applied to the finished sums. Bin 0 is computed from the weights in float64, see
 *        PairwiseStats.
 *
 * Float records carry r; the kLanes lanes of a record own scattering values l, l + kLanes, ... and each takes
 * sin(s r) for its own. Double records carry sin(l r) for l = 0..8 and 2 cos(8 r), from one double sincos by the
 * three-term recurrence; double sums are m8n8k4 MMA fragments (channels x pairs times pairs x scattering values), lane
 * l continuing sin(l' r), l' = l / 4, of its pair l % 4 with sin((l' + 8) r) = 2 cos(8 r) sin(l' r) - sin((l' - 8) r).
 * In double the recurrence stays within a few ulp of sin(s r) over these few steps.
 */
template <typename Real, int kBinLanes = 8> struct MorsePass {
  static constexpr bool kFloat          = std::is_same_v<Real, float>;
  static constexpr bool kMma            = !kFloat;
  static constexpr int  kLanes          = kBinLanes;
  static constexpr int  kSlots          = kNumMorseScatterings / kBinLanes;
  static constexpr int  kRecordsPerStep = kMma ? kMmaPairs : kWarpSize / kLanes;
  static constexpr int  kRowLoad        = kFloat ? 12 : 10;
  static constexpr int  kSines          = 8;
  //! Float: Q0..Q6, r. Double: Q0..Q6, 2 cos(8 r), sin(l r) for l = 0..8.
  static constexpr int  kRecordValues   = kFloat ? 8 : 18;
  //! Keeps the records a warp reads at once on distinct banks.
  static constexpr int  kRecordStride   = kFloat ? 12 : 20;
  //! Values a null record zeroes: all a consumer reads.
  static constexpr int  kNullValues     = kRecordStride;
  //! Every pair contributes, so batches keep a record per lane and overlap the next batch's records with the sums.
  static constexpr bool kPipelined      = true;
  static constexpr bool kSorted         = false;
  static constexpr int  kWidth          = kNumMorseProperties;
  static constexpr int  kBins           = kNumMorseScatterings;
  using Accumulators =
    std::conditional_t<kMma, double[kNumMorseScatterings / 8][2], Real[kSlots][kNumPairwiseChannels]>;
  static_assert(kFloat || kSlots * 8 == kNumMorseScatterings);

  //! Float: scattering value of slot @p slot of lane @p laneInGroup.
  __device__ __forceinline__ static int binOf(const int laneInGroup, const int slot) {
    return laneInGroup + kLanes * slot;
  }

  __device__ __forceinline__ static bool compute(const PairwiseConformer<Real>& conformer,
                                                 const int                      j,
                                                 const int                      k,
                                                 Real (&record)[kRecordValues]) {
    const auto pair     = pairGeometry<Real, kRowLoad>(conformer.rows, j, k);
    const Real r        = pair.distance;
    const Real inverseR = Real(1) / r;
    record[0]           = inverseR;
#pragma unroll
    for (int channel = 1; channel < kNumPairwiseChannels; ++channel) {
      record[channel] = pair.first[kRowWeights + channel - 1] * pair.second[kRowWeights + channel - 1] * inverseR;
    }
    if constexpr (kFloat) {
      record[kNumPairwiseChannels] = r;
    } else {
      // sin(l r) by the three-term recurrence sin((l + 1) r) = 2 cos(r) sin(l r) - sin((l - 1) r) from one double
      // sincos, and 2 cos(8 r) = 2 - 4 sin^2(4 r).
      double sine;
      double cosine;
      sincos(r, &sine, &cosine);
      const double twiceCosine = 2.0 * cosine;
      record[kSineOffset]      = 0.0;
      record[kSineOffset + 1]  = sine;
#pragma unroll
      for (int l = 2; l <= kSines; ++l) {
        record[kSineOffset + l] = fma(twiceCosine, record[kSineOffset + l - 1], -record[kSineOffset + l - 2]);
      }
      const double sineHalf        = record[kSineOffset + kSines / 2];
      record[kNumPairwiseChannels] = fma(-4.0 * sineHalf, sineHalf, 2.0);
    }
    return true;
  }

  __device__ __forceinline__ static void store(Real* target, const Real (&record)[kRecordValues]) {
    storeValues<kRecordValues>(target, record);
  }
  __device__ __forceinline__ static void prepare(Real*, int) {}
  __device__ __forceinline__ static void release(Real*, const Real (&)[kRecordValues]) {}
  //! A pipelined batch's slot: the pair's record, or a null record.
  __device__ __forceinline__ static void storeSlot(Real* target,
                                                   const Real (&record)[kRecordValues],
                                                   const bool contributes,
                                                   const int  swizzle) {
    if (!contributes) {
      storeZeros<kNullValues>(target);
    } else if constexpr (kFloat) {
      store(target, record);
    } else {
      storeSwizzled<kRecordValues>(target, record, swizzle);
    }
  }

  //! One step: the kRecordsPerStep records from @p records, double ones swizzled by @p swizzle.
  __device__ __forceinline__ static void consume(const Real*   records,
                                                 const int     lane,
                                                 Accumulators& acc,
                                                 const int     swizzle) {
    if constexpr (kFloat) {
      const Real* record      = records + (lane / kLanes) * kRecordStride;
      const int   laneInGroup = lane % kLanes;
      Real        head[kNumPairwiseChannels + 1];
      loadValues<kNumPairwiseChannels + 1>(record, head);
      Real products[kNumPairwiseChannels];
#pragma unroll
      for (int channel = 0; channel < kNumPairwiseChannels; ++channel) {
        products[channel] = head[channel];
      }
      // Each slot's sin(s r) on its own: independent special-function results keep the dependency chains short.
      Real sines[kSlots];
#pragma unroll
      for (int slot = 0; slot < kSlots; ++slot) {
        sines[slot] = sinf(static_cast<float>(binOf(laneInGroup, slot)) * head[kNumPairwiseChannels]);
      }
      accumulateOuterProduct(acc, products, sines);
    } else {
      const double* record      = records + (lane % kMmaPairs) * kRecordStride;
      const int     row         = lane / kMmaPairs;
      const double  product     = record[recordIndex<Real>(min(row, kNumPairwiseChannels - 1), swizzle)];
      const double  twiceCosine = record[recordIndex<Real>(kNumPairwiseChannels, swizzle)];
      double        sines[kSlots];
      sines[0] = record[recordIndex<Real>(kSineOffset + row, swizzle)];
      // sin((l - 8) r) = -sin((8 - l) r).
      sines[1] = fma(twiceCosine, sines[0], record[recordIndex<Real>(kSineOffset + kSines - row, swizzle)]);
#pragma unroll
      for (int slot = 2; slot < kSlots; ++slot) {
        sines[slot] = fma(twiceCosine, sines[slot - 1], -sines[slot - 2]);
      }
      // A's row 7 is padding.
      const double a = row < kNumPairwiseChannels ? product : 0.0;
#pragma unroll
      for (int slot = 0; slot < kSlots; ++slot) {
        mmaAccumulate(acc[slot], a, sines[slot]);
      }
    }
  }

 private:
  static constexpr int kSineOffset = kNumPairwiseChannels + 1;
  static_assert(kFloat || kSineOffset + kSines + 1 <= kRecordValues);
};

//! exp(-25 (2 b + 1)) for b = 0..3: RdfPass's double window steps.
__constant__ double kRdfStepFactors[4] = {1.3887943864964021e-11,
                                          2.6786369618080778e-33,
                                          5.166420632837861e-55,
                                          9.964733010103672e-77};

/**
 * @brief RDF bins: bin i adds w_j * w_k * exp(-100 (R_i - r)^2) with R_i = 1 + 0.5 i per channel, the I-state
 *        channel taking RDF's I-state weights.
 *
 * The exponent reaches -100 (float's flush to zero starts near -87) within one radius step of 0.5 A on either side of
 * the radii bracketing r, so a record carries the terms of the four bins floor(2 (r - 1)) - 1 .. + 2 only: every other
 * bin's term is zero in float and below exp(-100) in double. Float records spread the window over a zeroed term per
 * bin and lane l of a record's kLanes lanes owns the kSlots consecutive bins from kSlots l.
 *
 * Double sums are m8n8k4 MMAs (channels x pairs times pairs x bins) over kRdfTiles eight-bin tiles that overlap by
 * four bins: tile t covers bins 4 t .. 4 t + 7, so every window fits the tile of its start clamped to [0, 26], divided
 * by four. A batch's records are stored sorted by tile (kSorted), and each MMA step takes four records of one tile:
 * each pair costs about a quarter MMA instead of one per tile its step's pairs reach. Records place the window's terms
 * at their tile positions, zeros elsewhere; double records are stored transposed (see storeTransposed()).
 */
constexpr int kRdfTiles    = 7;
constexpr int kRdfTileBins = 8;
//! Class of a pair that reaches no bin.
constexpr int kRdfNoTile   = kRdfTiles;

template <typename Real, int kBinLanes = 8> struct RdfPass {
  static constexpr bool kFloat          = std::is_same_v<Real, float>;
  static constexpr bool kMma            = !kFloat;
  //! Double batches store records sorted by tile; see runPairwisePass().
  static constexpr bool kSorted         = kMma;
  static constexpr int  kLanes          = kBinLanes;
  static constexpr int  kSlots          = kNumMorseScatterings / kBinLanes;
  static constexpr int  kRecordsPerStep = kMma ? kMmaPairs : kWarpSize / kLanes;
  static constexpr int  kWindow         = 4;
  static constexpr int  kRowLoad        = kFloat ? 12 : 10;
  //! Products of channels 0-6 (channel 0's is 1, the last is RDF's I-state), window start (int), then the terms.
  static constexpr int  kStartOffset    = kNumPairwiseChannels;
  static constexpr int  kTermOffset     = 8;
  static constexpr int  kTerms          = kFloat ? kNumMorseScatterings : kRdfTileBins;
  static constexpr int  kRecordValues   = kTermOffset + kWindow;
  //! Float: keeps the records a warp reads at once on distinct banks. Double: values per transposed slot, the products
  //! (row 7 of A zero) then the tile's terms.
  static constexpr int  kRecordStride   = kFloat ? kTermOffset + kTerms + 4 : kTermOffset + kRdfTileBins;
  //! Sorted batches read up to kRecordsPerStep - 1 slots past their last record.
  static constexpr int  kRecordSlots    = kSorted ? kWarpSize + kRecordsPerStep : kWarpSize;
  //! Values a null record zeroes; float terms stay zeroed between records.
  static constexpr int  kNullValues     = kFloat ? kTermOffset : kRecordStride;
  //! Every pair contributes, so batches keep a record per lane and overlap the next batch's records with the sums.
  static constexpr bool kPipelined      = !kSorted;
  static constexpr int  kWidth          = kNumRdfProperties;
  static constexpr int  kBins           = kNumRdfRadii;
  using Accumulators = std::conditional_t<kMma, double[kRdfTiles][2], Real[kSlots][kNumPairwiseChannels]>;
  static_assert(kRowIStateDrag < kRowLoad);
  static_assert(kRdfTileBins * (kRdfTiles + 1) / 2 >= kNumRdfRadii);

  //! Float: bin of slot @p slot of lane @p laneInGroup.
  __device__ __forceinline__ static int binOf(const int laneInGroup, const int slot) {
    return laneInGroup * kSlots + slot;
  }

  //! Window start of a record.
  __device__ __forceinline__ static int startOf(const Real (&record)[kRecordValues]) {
    return realAsInt(record[kStartOffset]);
  }

  //! Tile of a contributing record's window, or kRdfNoTile when its window reaches no bin.
  __device__ __forceinline__ static int tileOf(const Real (&record)[kRecordValues]) {
    const int start = startOf(record);
    return start < kNumRdfRadii ? min(max(start, 0), kNumRdfRadii - kWindow) / 4 : kRdfNoTile;
  }

  __device__ __forceinline__ static bool compute(const PairwiseConformer<Real>& conformer,
                                                 const int                      j,
                                                 const int                      k,
                                                 Real (&record)[kRecordValues]) {
    const auto pair = pairGeometry<Real, kRowLoad>(conformer.rows, j, k);
    const Real r    = pair.distance;
    // Radius index floor(2 (r - 1)); clamped so out-of-range and non-finite distances keep int arithmetic defined.
    int        start;
    if constexpr (kFloat) {
      start = __float2int_rd((r - Real(1)) * Real(2)) - 1;
    } else {
      start = __double2int_rd((r - Real(1)) * Real(2)) - 1;
    }
    start     = min(max(start, -kWindow), kNumRdfRadii);
    record[0] = Real(1);
#pragma unroll
    for (int channel = 1; channel < kNumPairwiseChannels - 1; ++channel) {
      record[channel] = pair.first[kRowWeights + channel - 1] * pair.second[kRowWeights + channel - 1];
    }
    record[kNumPairwiseChannels - 1] = pair.first[kRowIStateDrag] * pair.second[kRowIStateDrag];
    record[kStartOffset]             = intAsReal<Real>(start);
    const Real offset                = Real(1) + Real(0.5) * static_cast<Real>(start) - r;
    if constexpr (kFloat) {
#pragma unroll
      for (int bin = 0; bin < kWindow; ++bin) {
        const Real binOffset      = Real(1) + Real(0.5) * static_cast<Real>(start + bin) - r;
        record[kTermOffset + bin] = exp(Real(-100) * binOffset * binOffset);
      }
    } else {
      // exp(-100 (d + 0.5 (b + 1))^2) = exp(-100 (d + 0.5 b)^2) exp(-100 d) exp(-25 (2 b + 1)) from the window's first
      // offset d in (-1, -0.5]: two double exp() for the four terms, each step a few ulp. A window clamped past the
      // last radius has no bins and stays zero (exp(-100 d) would overflow).
      const bool   binned = start < kNumRdfRadii;
      const double growth = binned ? exp(-100.0 * offset) : 0.0;
      double       term   = binned ? exp(-100.0 * offset * offset) : 0.0;
#pragma unroll
      for (int bin = 0; bin < kWindow; ++bin) {
        record[kTermOffset + bin] = term;
        term *= growth * kRdfStepFactors[bin];
      }
    }
    return true;
  }

  //! Float records keep their terms zeroed between uses (see prepare() and release()), so a record writes only the
  //! window's four terms at their bins.
  __device__ __forceinline__ static void store(Real* target, const Real (&record)[kRecordValues]) {
    if constexpr (kFloat) {
      const int start = startOf(record);
      storeValues<kTermOffset>(target, record);
#pragma unroll
      for (int bin = 0; bin < kWindow; ++bin) {
        if (static_cast<unsigned>(start + bin) < static_cast<unsigned>(kTerms)) {
          target[kTermOffset + start + bin] = record[kTermOffset + bin];
        }
      }
    }
  }

  /**
   * @brief Double: stores a record at sorted slot @p slot, transposed (value v of slot s at v * kRecordSlots + s): the
   *        channel products (row 7 of A zero) and the tile's eight terms, the window's at their positions and zeros
   *        elsewhere. The lanes' slots are a permutation, so each value's stores fill one contiguous row, and a step's
   *        reads (four consecutive slots of eight rows) take few shared-memory wavefronts.
   */
  __device__ __forceinline__ static void storeTransposed(Real*     records,
                                                         const int slot,
                                                         const Real (&record)[kRecordValues]) {
    if constexpr (kMma) {
      Real* const column = records + slot;
#pragma unroll
      for (int channel = 0; channel < kNumPairwiseChannels; ++channel) {
        column[channel * kRecordSlots] = record[channel];
      }
      column[kNumPairwiseChannels * kRecordSlots] = Real(0);
#pragma unroll
      for (int position = 0; position < kRdfTileBins; ++position) {
        column[(kTermOffset + position) * kRecordSlots] = Real(0);
      }
      // Window position of bin start + bin in its tile; positions outside the tile (bins below 0 or past 31) drop.
      const int first = startOf(record) - 4 * tileOf(record);
#pragma unroll
      for (int bin = 0; bin < kWindow; ++bin) {
        if (static_cast<unsigned>(first + bin) < static_cast<unsigned>(kRdfTileBins)) {
          column[(kTermOffset + first + bin) * kRecordSlots] = record[kTermOffset + bin];
        }
      }
    }
  }

  //! Zeros the records before the pass's first record: float terms, and every double slot a step may read.
  __device__ __forceinline__ static void prepare(Real* records, const int lane) {
    for (int value = lane * kVector<Real>; value < kRecordSlots * kRecordStride; value += kWarpSize * kVector<Real>) {
      storeZeros<kVector<Real>>(records + value);
    }
  }

  //! Clears the terms store() wrote once every lane has consumed the record.
  __device__ __forceinline__ static void release(Real* target, const Real (&record)[kRecordValues]) {
    if constexpr (kFloat) {
      const int start = startOf(record);
#pragma unroll
      for (int bin = 0; bin < kWindow; ++bin) {
        if (static_cast<unsigned>(start + bin) < static_cast<unsigned>(kTerms)) {
          target[kTermOffset + start + bin] = Real(0);
        }
      }
    }
  }

  //! A pipelined batch's slot: the pair's record, or a null record.
  __device__ __forceinline__ static void storeSlot(Real* target,
                                                   const Real (&record)[kRecordValues],
                                                   const bool contributes,
                                                   const int) {
    if (contributes) {
      store(target, record);
    } else {
      storeZeros<kNullValues>(target);
    }
  }

  //! Float: one step, the kRecordsPerStep records from @p records.
  __device__ __forceinline__ static void consume(const Real* records, const int lane, Accumulators& acc, const int) {
    if constexpr (kFloat) {
      const Real* record      = records + (lane / kLanes) * kRecordStride;
      const int   laneInGroup = lane % kLanes;
      Real        head[kTermOffset];
      loadValues<kTermOffset>(record, head);
      Real products[kNumPairwiseChannels];
#pragma unroll
      for (int channel = 0; channel < kNumPairwiseChannels; ++channel) {
        products[channel] = head[channel];
      }
      Real terms[kSlots];
      loadValues<kSlots>(record + kTermOffset + binOf(laneInGroup, 0), terms);
      accumulateOuterProduct(acc, products, terms);
    }
  }

  /**
   * @brief Double: one MMA step of tile @p tile, the records from slot @p first; slots from @p end on belong to the
   *        next tile and contribute zero.
   */
  __device__ __forceinline__ static void consumeTile(const Real* records,
                                                     const int   first,
                                                     const int   end,
                                                     const int   lane,
                                                     double (&acc)[2]) {
    if constexpr (kMma) {
      const int     slot   = first + lane % kMmaPairs;
      const double* column = records + slot;
      const int     row    = lane / kMmaPairs;
      const bool    valid  = slot < end;
      const double  a      = valid ? column[row * kRecordSlots] : 0.0;
      const double  b      = valid ? column[(kTermOffset + row) * kRecordSlots] : 0.0;
      mmaAccumulate(acc, a, b);
    }
  }
};

/**
 * @brief AUTOCORR3D lags: lag l (1-10) adds w_j * w_k * r over pairs whose bond-count distance is l, for the MORSE
 *        channels plus covalent radius.
 *
 * Float compacts each batch to the pairs within kNumAutocorrLags bonds; lane l of a record's kLanes lanes owns lags
 * l + 1 and l + 1 + kLanes for every channel, so each record updates one lane's sums. Double keeps every pair's record
 * (pairs farther apart get null records) and sums with m8n8k4 MMAs: channels x pairs times pairs x lags, B holding r
 * in the pair's lag column; the second tile (lags 9-10) runs only when a pair of the step needs it.
 */
template <typename Real> struct AutocorrPass {
  static constexpr bool kFloat          = std::is_same_v<Real, float>;
  static constexpr bool kMma            = !kFloat;
  static constexpr int  kLanes          = kGroupSize;
  static constexpr int  kRecordsPerStep = kMma ? kMmaPairs : kWarpSize / kLanes;
  static constexpr int  kRowLoad        = 12;
  //! Products of channels 0-7, r, lag (int), padding.
  static constexpr int  kRecordValues   = kFloat ? 12 : 10;
  //! Keeps the records a warp reads at once on distinct banks.
  static constexpr int  kRecordStride   = kFloat ? 12 : 12;
  static constexpr int  kNullValues     = kRecordStride;
  //! Double overlaps the next batch's records with the sums; float compacts batches instead.
  static constexpr bool kPipelined      = kMma;
  static constexpr bool kSorted         = false;
  static constexpr int  kWidth          = kNumAutocorr3DProperties;
  static constexpr int  kLagSlots       = (kNumAutocorrLags + kLanes - 1) / kLanes;
  using Accumulators = std::conditional_t<kMma, double[kLagSlots][2], Real[kLagSlots][kNumAutocorrChannels]>;
  static_assert(kLagSlots == 2 && kNumAutocorrChannels % 2 == 0 && kNumAutocorrChannels == 8);

  __device__ __forceinline__ static bool compute(const PairwiseConformer<Real>& conformer,
                                                 const int                      j,
                                                 const int                      k,
                                                 Real (&record)[kRecordValues]) {
    const int a   = min(j, k);
    const int b   = max(j, k);
    const int lag = conformer.lagTable[bondTableRowStart(a, conformer.indexing.numAtoms) + (b - a - 1)];
    if (lag < 1 || lag > kNumAutocorrLags) {
      return false;
    }
    const auto pair = pairGeometry<Real, kRowLoad>(conformer.rows, j, k);
    pairProducts(pair.first, pair.second, record);
    record[kNumAutocorrChannels]     = pair.distance;
    record[kNumAutocorrChannels + 1] = intAsReal<Real>(lag);
#pragma unroll
    for (int value = kNumAutocorrChannels + 2; value < kRecordValues; ++value) {
      record[value] = Real(0);
    }
    return true;
  }

  //! Per-channel weight products of a pair: 1, the atom-property channels, then the covalent radius.
  template <int kSize, int kTarget>
  __device__ __forceinline__ static void pairProducts(const Real (&first)[kSize],
                                                      const Real (&second)[kSize],
                                                      Real (&products)[kTarget]) {
    products[0] = Real(1);
#pragma unroll
    for (int channel = 1; channel < kNumPairwiseChannels; ++channel) {
      products[channel] = first[kRowWeights + channel - 1] * second[kRowWeights + channel - 1];
    }
    products[kNumPairwiseChannels] = first[kRowCovalentRadius] * second[kRowCovalentRadius];
  }

  __device__ __forceinline__ static void store(Real* target, const Real (&record)[kRecordValues]) {
    storeValues<kRecordValues>(target, record);
  }
  __device__ __forceinline__ static void prepare(Real*, int) {}
  __device__ __forceinline__ static void release(Real*, const Real (&)[kRecordValues]) {}
  //! A pipelined batch's slot: the pair's record, or a null record (lag 0, which no lag column takes).
  __device__ __forceinline__ static void storeSlot(Real* target,
                                                   const Real (&record)[kRecordValues],
                                                   const bool contributes,
                                                   const int  swizzle) {
    if (!contributes) {
      storeZeros<kNullValues>(target);
    } else if constexpr (kFloat) {
      store(target, record);
    } else {
      storeSwizzled<kRecordValues>(target, record, swizzle);
    }
  }

  //! Warp-collective step: the kRecordsPerStep records from @p records, double ones swizzled by @p swizzle. Null
  //! records (lag 0) update nothing.
  __device__ __forceinline__ static void consume(const Real*   records,
                                                 const int     lane,
                                                 Accumulators& acc,
                                                 const int     swizzle) {
    if constexpr (kMma) {
      const double* record = records + (lane % kMmaPairs) * kRecordStride;
      const int     row    = lane / kMmaPairs;
      double        tail[kVector<double>];
      loadValues<kVector<double>>(record + recordIndex<Real>(kNumAutocorrChannels, swizzle), tail);
      const int    lag     = realAsInt(tail[1]);
      const double product = record[recordIndex<Real>(row, swizzle)];
      mmaAccumulate(acc[0], product, lag - 1 == row ? tail[0] : 0.0);
      if (__any_sync(0xffffffffu, lag > kLanes)) {
        mmaAccumulate(acc[1], product, lag - 1 - kLanes == row ? tail[0] : 0.0);
      }
    } else {
      const Real* record      = records + (lane / kLanes) * kRecordStride;
      const int   laneInGroup = lane % kLanes;
      Real        products[kNumAutocorrChannels];
      loadValues<kNumAutocorrChannels>(record, products);
      Real tail[kVector<Real>];
      loadValues<kVector<Real>>(record + kNumAutocorrChannels, tail);
      const int offset = realAsInt(tail[1]) - 1 - laneInGroup;
      accumulateLag(acc, products, tail[0], offset);
      if (__any_sync(0xffffffffu, offset == kLanes)) {
        if (offset == kLanes) {
          accumulateChannels(acc[1], products, tail[0]);
        }
      }
    }
  }

  //! Adds the pair to the lane's lag slot 0 when @p offset (lag - 1 - laneInGroup) is 0.
  template <typename Sums>
  __device__ __forceinline__ static void accumulateLag(Sums& acc,
                                                       const Real (&products)[kNumAutocorrChannels],
                                                       const Real r,
                                                       const int  offset) {
    if (offset == 0) {
      accumulateChannels(acc[0], products, r);
    }
  }

  //! sums[c] += products[c] * r for every channel.
  __device__ __forceinline__ static void accumulateChannels(Real (&sums)[kNumAutocorrChannels],
                                                            const Real (&products)[kNumAutocorrChannels],
                                                            const Real r) {
#pragma unroll
    for (int channel = 0; channel < kNumAutocorrChannels; channel += 2) {
      fmaPair(sums[channel], sums[channel + 1], products[channel], products[channel + 1], r, r);
    }
  }
};

//! std::max for constexpr functions that device code also calls.
__host__ __device__ constexpr int maxValues(const int a, const int b) {
  return a > b ? a : b;
}

//! Record region of one warp, in values of the compute type: the records of the widest pass.
template <typename Real, unsigned kSet> __host__ __device__ constexpr int recordRegion() {
  int values = 0;
  if constexpr ((kSet & kPairwiseRdf) != 0) {
    values = maxValues(values, RdfPass<Real>::kRecordSlots * RdfPass<Real>::kRecordStride);
  }
  if constexpr ((kSet & kPairwiseMorse) != 0) {
    values = maxValues(values, kWarpSize * MorsePass<Real>::kRecordStride);
  }
  if constexpr ((kSet & kPairwiseAutocorr3D) != 0) {
    values = maxValues(values, kWarpSize * AutocorrPass<Real>::kRecordStride);
  }
  return maxValues(values, kNumMorseProperties);
}

//! Output values of the properties in @p kSet, in output order: RDF, MORSE, AUTOCORR3D.
template <unsigned kSet> struct PairwiseWidths {
  static constexpr int kRdf      = (kSet & kPairwiseRdf) != 0 ? kNumRdfProperties : 0;
  static constexpr int kMorse    = (kSet & kPairwiseMorse) != 0 ? kNumMorseProperties : 0;
  static constexpr int kAutocorr = (kSet & kPairwiseAutocorr3D) != 0 ? kNumAutocorr3DProperties : 0;
  static constexpr int kTotal    = kRdf + kMorse + kAutocorr;
};

//! Per-conformer values gathered while staging the atom rows.
struct PairwiseStats {
  //! MORSE bin 0 per channel, sum over pairs j < k of w_j * w_k, as ((sum w)^2 - sum w^2) / 2 in float64.
  double   morseZeroBin[kNumPairwiseChannels];
  //! Per warp: centroid sums, then the MORSE weight sums and sums of squares.
  double   warpSums[kPairwiseWarps][16];
  double   centroid[3];
  //! Per warp, bit c: AUTOCORR3D channel c has a NaN weight (bit 0: a non-finite coordinate).
  unsigned warpFlags[kPairwiseWarps];
  unsigned autocorrNaN;
  //! Whether this block combines the conformer's slices.
  bool     lastSlice;
  //! Per warp of a block of packed small conformers: its conformer's MORSE bin 0 and AUTOCORR3D NaN channels.
  double   soloZeroBin[kPairwiseWarps][kNumPairwiseChannels];
  unsigned soloNaN[kPairwiseWarps];
};

template <typename Real, unsigned kSet, int kSumRows> struct PairwiseSharedLayoutOf {
  static __host__ __device__ constexpr size_t align(const size_t bytes) { return (bytes + 15) / 16 * 16; }
  //! Each warp's sums of every property, or with one row the block's (see kPairwiseCompactSums).
  static constexpr int kSumStride = (PairwiseWidths<kSet>::kTotal + kVector<Real> - 1) / kVector<Real> * kVector<Real>;
  static constexpr size_t kPartialOffset = align(sizeof(PairwiseStats));
  static constexpr size_t kRecordOffset =
    kPartialOffset + align(static_cast<size_t>(kSumRows) * kSumStride * sizeof(Real));
  static constexpr size_t kRowOffset =
    kRecordOffset + static_cast<size_t>(kPairwiseWarps) * recordRegion<Real, kSet>() * sizeof(Real);
  static __host__ __device__ constexpr size_t rowBytes(const int sharedAtoms) {
    return align(static_cast<size_t>(sharedAtoms) * kRowStride<Real> * sizeof(Real));
  }
  //! AUTOCORR3D copies its molecule's bond-distance entries after the rows when they fit.
  static constexpr size_t                     kLagBytes = (kSet & kPairwiseAutocorr3D) != 0 ? kPairwiseLagBudget : 0;
  static __host__ __device__ constexpr size_t bytes(const int sharedAtoms) {
    return kRowOffset + rowBytes(sharedAtoms) + kLagBytes;
  }
};

//! Atoms per conformer the occupancy target assumes; larger conformers' rows take more shared memory.
constexpr int kOccupancyAtoms = 128;
//! Most blocks per SM worth reserving registers for: beyond them the register budget would spill. Double records and
//! RDF's window terms need the registers of a sixth block.
template <typename Real, unsigned kSet>
constexpr int kPairwiseMaxBlocks =
  std::is_same_v<Real, double> || (kSet & kPairwiseRdf) != 0 || kSet == kPairwiseMorse ? 5 : 6;

//! Blocks per SM that @p bytes of shared memory per block allow at kOccupancyAtoms, at most kPairwiseMaxBlocks.
template <typename Real, unsigned kSet> __host__ __device__ constexpr int sharedMemoryBlocks(const size_t bytes) {
  const int blocks = static_cast<int>((size_t{228} << 10) / (bytes + (size_t{1} << 10)));
  return blocks > kPairwiseMaxBlocks<Real, kSet> ? kPairwiseMaxBlocks<Real, kSet> : blocks;
}

/**
 * Whether the block keeps one compact row of sums instead of one per warp: each pass leaves its warp sums in the
 * warp's records, which the pass no longer needs, and the block adds them up at once. That costs two block barriers per
 * pass, so only sets whose occupancy the smaller layout raises use it.
 */
template <typename Real, unsigned kSet>
constexpr bool kPairwiseCompactSums =
  sharedMemoryBlocks<Real, kSet>(PairwiseSharedLayoutOf<Real, kSet, 1>::bytes(kOccupancyAtoms)) >
  sharedMemoryBlocks<Real, kSet>(PairwiseSharedLayoutOf<Real, kSet, kPairwiseWarps>::bytes(kOccupancyAtoms));

template <typename Real, unsigned kSet>
using PairwiseSharedLayout = PairwiseSharedLayoutOf<Real, kSet, kPairwiseCompactSums<Real, kSet> ? 1 : kPairwiseWarps>;

/**
 * @brief Most atom rows a block stages in shared memory on a device allowing @p sharedPerBlock bytes per block: within
 *        kPairwiseRowBudget, and small enough that every set's layout fits.
 */
template <typename Real> int pairwiseRowCap(const size_t sharedPerBlock) {
  constexpr size_t kOverheads[] = {
    PairwiseSharedLayout<Real, kPairwiseRdf>::bytes(0),
    PairwiseSharedLayout<Real, kPairwiseMorse>::bytes(0),
    PairwiseSharedLayout<Real, kPairwiseRdf | kPairwiseMorse>::bytes(0),
    PairwiseSharedLayout<Real, kPairwiseAutocorr3D>::bytes(0),
    PairwiseSharedLayout<Real, kPairwiseRdf | kPairwiseAutocorr3D>::bytes(0),
    PairwiseSharedLayout<Real, kPairwiseMorse | kPairwiseAutocorr3D>::bytes(0),
    PairwiseSharedLayout<Real, kPairwiseRdf | kPairwiseMorse | kPairwiseAutocorr3D>::bytes(0),
  };
  size_t overhead = 0;
  for (const size_t bytes : kOverheads) {
    overhead = std::max(overhead, bytes);
  }
  constexpr size_t kRowSize  = kRowStride<Real> * sizeof(Real);
  // rowBytes() rounds up to 16 bytes.
  const size_t     available = sharedPerBlock > overhead + 16 ? sharedPerBlock - overhead - 16 : 0;
  return static_cast<int>(std::min(kPairwiseRowBudget, available) / kRowSize);
}

/**
 * @brief Blocks per SM the register use must allow: as many as the instantiation's shared memory fits at
 *        kOccupancyAtoms, between 4 and kPairwiseMaxBlocks. The kernel is latency-bound, so registers left unused by an
 *        occupancy the shared memory cannot reach would be wasted.
 */
template <typename Real, unsigned kSet> __host__ __device__ constexpr int pairwiseMinBlocks() {
  if constexpr (binLanes<Real, kSet>() == 4) {
    return 4;  // Eight bins per lane: the sums take the registers of a fifth block.
  }
  constexpr int kBlocks = sharedMemoryBlocks<Real, kSet>(PairwiseSharedLayout<Real, kSet>::bytes(kOccupancyAtoms));
  return kBlocks < 4 ? 4 : kBlocks;
}

/**
 * @brief Sums each lane's bin slots over the warp's groups of @p kLanes lanes (lanes of equal laneInGroup),
 *        reduce-scattered: afterwards group g holds bin slot g of every channel in @p result.
 */
template <int kLanes, typename Real, int kSlots>
__device__ __forceinline__ void reduceBinsAcrossGroups(const Real (&acc)[kSlots][kNumPairwiseChannels],
                                                       const int lane,
                                                       Real (&result)[kNumPairwiseChannels]) {
  static_assert(kSlots * kLanes == kWarpSize);
  Real current[kSlots][kNumPairwiseChannels];
#pragma unroll
  for (int slot = 0; slot < kSlots; ++slot) {
#pragma unroll
    for (int channel = 0; channel < kNumPairwiseChannels; ++channel) {
      current[slot][channel] = acc[slot][channel];
    }
  }
#pragma unroll
  for (int width = kSlots / 2, offset = kWarpSize / 2; width >= 1; width /= 2, offset /= 2) {
    const bool upper = (lane & offset) != 0;
#pragma unroll
    for (int slot = 0; slot < width; ++slot) {
#pragma unroll
      for (int channel = 0; channel < kNumPairwiseChannels; ++channel) {
        const Real keep        = upper ? current[slot + width][channel] : current[slot][channel];
        const Real send        = upper ? current[slot][channel] : current[slot + width][channel];
        current[slot][channel] = keep + __shfl_xor_sync(0xffffffffu, send, offset);
      }
    }
  }
#pragma unroll
  for (int channel = 0; channel < kNumPairwiseChannels; ++channel) {
    result[channel] = current[0][channel];
  }
}

//! Writes one warp's bin sums, laid out like the property's output row (channel-major), to @p target.
template <typename Pass, typename Real>
__device__ __forceinline__ void writeWarpBins(const typename Pass::Accumulators& acc, const int lane, Real* target) {
  Real sums[kNumPairwiseChannels];
  reduceBinsAcrossGroups<Pass::kLanes>(acc, lane, sums);
  const int bin = Pass::binOf(lane % Pass::kLanes, lane / Pass::kLanes);
  if (bin < Pass::kBins) {
#pragma unroll
    for (int channel = 0; channel < kNumPairwiseChannels; ++channel) {
      target[channel * Pass::kBins + bin] = sums[channel];
    }
  }
}

//! Writes one warp's MMA fragments, laid out like the property's output row (channel-major), to @p target.
template <typename Pass>
__device__ __forceinline__ void writeMmaBins(const typename Pass::Accumulators& acc, const int lane, double* target) {
  const int channel = lane / kMmaPairs;
#pragma unroll
  for (int tile = 0; tile < kNumMorseScatterings / 8; ++tile) {
#pragma unroll
    for (int column = 0; column < 2; ++column) {
      const int bin = 8 * tile + 2 * (lane % kMmaPairs) + column;
      if (channel < kNumPairwiseChannels && bin < Pass::kBins) {
        target[channel * Pass::kBins + bin] = acc[tile][column];
      }
    }
  }
}

//! Writes one warp's RDF tile fragments (lane l: channel l / 4, tile positions 2 (l % 4) + i), laid out like the output
//! row (channel-major), to @p target. Bin 4 t + p (p < 4) adds tile t - 1's position p + 4, held two lanes up.
__device__ __forceinline__ void writeRdfTiles(const double (&acc)[kRdfTiles][2], const int lane, double* target) {
  const int  channel  = lane / kMmaPairs;
  const int  quad     = lane % kMmaPairs;
  const bool written  = channel < kNumPairwiseChannels;
  double     carry[2] = {0.0, 0.0};
#pragma unroll
  for (int tile = 0; tile < kRdfTiles; ++tile) {
#pragma unroll
    for (int column = 0; column < 2; ++column) {
      const double upper = __shfl_down_sync(0xffffffffu, acc[tile][column], 2);
      if (quad < 2 && written) {
        target[channel * kNumRdfRadii + 4 * tile + 2 * quad + column] = acc[tile][column] + carry[column];
      }
      carry[column] = upper;
    }
  }
  // Bins 28 and 29: positions 4 and 5 of the last tile.
  if (quad == 0 && written) {
    target[channel * kNumRdfRadii + 4 * kRdfTiles]     = carry[0];
    target[channel * kNumRdfRadii + 4 * kRdfTiles + 1] = carry[1];
  }
}

//! A batch's tile classes (kRdfNoTile for none) as three ballots, one per bit of the class.
struct TileBallots {
  unsigned bits[3];

  //! Lanes of class @p tile.
  __device__ __forceinline__ unsigned lanesOf(const int tile) const {
    return ((tile & 1) != 0 ? bits[0] : ~bits[0]) & ((tile & 2) != 0 ? bits[1] : ~bits[1]) &
           ((tile & 4) != 0 ? bits[2] : ~bits[2]);
  }
};
static_assert(kRdfNoTile < 8);

//! Warp-collective. Stores this lane's record (of tile @p tile, or none for kRdfNoTile) at its slot in the batch's
//! records sorted by tile, lanes in order within a tile, and returns the batch's classes in @p ballots.
template <typename Pass, typename Real>
__device__ __forceinline__ void storeSorted(Real* records,
                                            const Real (&record)[Pass::kRecordValues],
                                            const int    tile,
                                            const int    lane,
                                            TileBallots& ballots) {
  // Lanes of a lower class, from the most significant bit down, then lower lanes of the same class.
  unsigned same  = 0xffffffffu;
  unsigned lower = 0u;
#pragma unroll
  for (int bit = 2; bit >= 0; --bit) {
    const bool     set  = ((tile >> bit) & 1) != 0;
    const unsigned mask = __ballot_sync(0xffffffffu, set);
    ballots.bits[bit]   = mask;
    lower |= set ? same & ~mask : 0u;
    same &= set ? mask : ~mask;
  }
  if (tile < kRdfTiles) {
    Pass::storeTransposed(records, __popc(lower | (same & ((1u << lane) - 1u))), record);
  }
}

//! Writes one warp's AUTOCORR3D MMA fragments (lane l: channel l / 4, lags 2 (l % 4) + i + 1 of each tile), laid out
//! like the output row (channel-major), to @p target.
__device__ __forceinline__ void writeMmaLags(const double (&acc)[2][2], const int lane, double* target) {
  const int channel = lane / kMmaPairs;
#pragma unroll
  for (int tile = 0; tile < 2; ++tile) {
#pragma unroll
    for (int column = 0; column < 2; ++column) {
      const int lag = 8 * tile + 2 * (lane % kMmaPairs) + column;
      if (lag < kNumAutocorrLags) {
        target[channel * kNumAutocorrLags + lag] = acc[tile][column];
      }
    }
  }
}

/**
 * @brief Writes one warp's AUTOCORR3D sums, laid out like the output row (channel-major), to @p target. The sums are
 *        reduce-scattered over the groups: group g ends with lag slot g / 2 of channels 4 (g % 2) .. + 3.
 */
template <typename Real>
__device__ __forceinline__ void writeWarpLags(const Real (&acc)[AutocorrPass<Real>::kLagSlots][kNumAutocorrChannels],
                                              const int lane,
                                              Real*     target) {
  static_assert(kWarpSize / kGroupSize == 4);
  constexpr int kHalf = kNumAutocorrChannels / 2;
  const int     group = lane / kGroupSize;
  const bool    high  = group >= 2;
  const bool    odd   = (group & 1) != 0;
  Real          half[kNumAutocorrChannels];
#pragma unroll
  for (int channel = 0; channel < kNumAutocorrChannels; ++channel) {
    const Real keep = high ? acc[1][channel] : acc[0][channel];
    const Real send = high ? acc[0][channel] : acc[1][channel];
    half[channel]   = keep + __shfl_xor_sync(0xffffffffu, send, 2 * kGroupSize);
  }
  const int lag = lane % kGroupSize + (high ? kGroupSize : 0);
#pragma unroll
  for (int channel = 0; channel < kHalf; ++channel) {
    const Real keep = odd ? half[channel + kHalf] : half[channel];
    const Real send = odd ? half[channel] : half[channel + kHalf];
    const Real sum  = keep + __shfl_xor_sync(0xffffffffu, send, kGroupSize);
    if (lag < kNumAutocorrLags) {
      target[((odd ? kHalf : 0) + channel) * kNumAutocorrLags + lag] = sum;
    }
  }
}

//! Writes zeros over one warp's sums of a property it took no pairs of.
template <typename Real> __device__ __forceinline__ void clearWarpSums(const int lane, const int width, Real* target) {
  for (int index = lane; index < width; index += kWarpSize) {
    target[index] = Real(0);
  }
}

/**
 * @brief Computes pair @p pair's record without branching on it: pairs past the slice compute the slice's first pair
 *        and report that they do not contribute.
 */
template <typename Pass, typename Real>
__device__ __forceinline__ bool computePair(const PairwiseConformer<Real>& conformer,
                                            const PairCursor&              cursor,
                                            const bool                     inRange,
                                            Real (&record)[Pass::kRecordValues]) {
  const bool       valid = inRange && cursor.pair < conformer.lastPair;
  const PairCursor safe  = valid ? cursor : conformer.indexing.cursor(conformer.firstPair);
  int              j;
  int              k;
  conformer.indexing.atomsAt(safe, j, k);
  return Pass::compute(conformer, j, k, record) && valid;
}

//! Fewest kWarpSize-pair batches a warp takes before another warp joins the slice.
constexpr int kMinWarpBatches = 3;

//! The batches of the slice's pairs warp @p warp takes: whole batches in contiguous shares, from the first warp, so
//! small slices leave later warps idle rather than splitting a few batches over every warp.
template <typename Real>
__device__ __forceinline__ int2 warpBatches(const PairwiseConformer<Real>& conformer,
                                            const int                      warp,
                                            const bool                     reversed) {
  const int numBatches = (conformer.lastPair - conformer.firstPair + kWarpSize - 1) / kWarpSize;
  if (conformer.solo) {
    return make_int2(0, numBatches);
  }
  const int activeWarps = min(kPairwiseWarps, max(1, numBatches / kMinWarpBatches));
  if (warp >= activeWarps) {
    return make_int2(0, 0);
  }
  const int share = reversed ? activeWarps - 1 - warp : warp;
  return make_int2(numBatches * share / activeWarps, numBatches * (share + 1) / activeWarps);
}

/**
 * @brief Warp-collective per warp of the block. Walks the warp's batches (warpBatches()) of the conformer's slice of
 *        atom pairs for one property: every lane writes one pair's record (Pass::compute()) to the warp's records,
 *        compacted to the pairs that contribute and padded with zero records to whole steps, then the warp's groups
 *        of Pass::kLanes lanes take one record each per step, each lane accumulating the bins it owns. Each warp's sums
 *        go to its own @p warpSums row, laid out like the property's output row.
 *
 * Passes whose every pair contributes (Pass::kPipelined) instead keep a full batch of records, lane t's in slot t,
 * and compute the next batch's records in registers while consuming the current one, so the records' long dependency
 * chains overlap the accumulation.
 */
template <typename Real, typename Pass>
__device__ __forceinline__ void runPairwisePass(const PairwiseConformer<Real>& conformer,
                                                Real*                          records,
                                                const int                      regionSize,
                                                const bool                     reversed,
                                                Real*                          warpSums) {
  constexpr int kRecordsPerStep = Pass::kRecordsPerStep;
  const int     lane            = static_cast<int>(threadIdx.x) % kWarpSize;
  const int     warp            = static_cast<int>(threadIdx.x) / kWarpSize;
  Real*         warpRecords     = records + warp * regionSize;

  typename Pass::Accumulators acc;
  {
    Real* values = reinterpret_cast<Real*>(&acc);
#pragma unroll
    for (int value = 0; value < static_cast<int>(sizeof(acc) / sizeof(Real)); ++value) {
      values[value] = Real(0);
    }
  }
  const int2 batches = warpBatches(conformer, warp, reversed);
  if (batches.x < batches.y) {
    Pass::prepare(warpRecords, lane);
    __syncwarp();
  }
  PairCursor cursor = conformer.indexing.cursor(conformer.firstPair + batches.x * kWarpSize + lane);
  if constexpr (Pass::kSorted) {
    // Batches sorted by tile: each step takes kRecordsPerStep records of one tile into that tile's sums.
    Real        record[Pass::kRecordValues];
    TileBallots tileBallots;
    if (batches.x < batches.y) {
      const bool contributes = computePair<Pass>(conformer, cursor, true, record);
      storeSorted<Pass>(warpRecords, record, contributes ? Pass::tileOf(record) : kRdfNoTile, lane, tileBallots);
      __syncwarp();
    }
    for (int batch = batches.x; batch < batches.y; ++batch) {
      const PairCursor nextCursor = conformer.indexing.advance(cursor);
      Real             next[Pass::kRecordValues];
      const bool       nextContributes = computePair<Pass>(conformer, nextCursor, batch + 1 < batches.y, next);
      int              first           = 0;
#pragma unroll
      for (int tile = 0; tile < kRdfTiles; ++tile) {
        const int end = first + __popc(tileBallots.lanesOf(tile));
        for (int slot = first; slot < end; slot += kRecordsPerStep) {
          Pass::consumeTile(warpRecords, slot, end, lane, acc[tile]);
        }
        first = end;
      }
      __syncwarp();
      storeSorted<Pass>(warpRecords, next, nextContributes ? Pass::tileOf(next) : kRdfNoTile, lane, tileBallots);
      __syncwarp();
      cursor = nextCursor;
    }
  } else if constexpr (Pass::kPipelined) {
    Real* const own = warpRecords + lane * Pass::kRecordStride;
    Real        record[Pass::kRecordValues];
    bool        contributes = false;
    const int   swizzle     = recordSwizzle(lane);
    if (batches.x < batches.y) {
      contributes = computePair<Pass>(conformer, cursor, true, record);
      Pass::storeSlot(own, record, contributes, swizzle);
      __syncwarp();
    }
    for (int batch = batches.x; batch < batches.y; ++batch) {
      const PairCursor nextCursor = conformer.indexing.advance(cursor);
      Real             next[Pass::kRecordValues];
      const bool       nextContributes = computePair<Pass>(conformer, nextCursor, batch + 1 < batches.y, next);
#pragma unroll
      for (int step = 0; step < kWarpSize / kRecordsPerStep; ++step) {
        Pass::consume(warpRecords + step * kRecordsPerStep * Pass::kRecordStride,
                      lane,
                      acc,
                      recordSwizzle(step * kRecordsPerStep));
      }
      __syncwarp();
      if (contributes) {
        Pass::release(own, record);
      }
      Pass::storeSlot(own, next, nextContributes, swizzle);
#pragma unroll
      for (int value = 0; value < Pass::kRecordValues; ++value) {
        record[value] = next[value];
      }
      contributes = nextContributes;
      __syncwarp();
      cursor = nextCursor;
    }
  } else {
    for (int batch = batches.x; batch < batches.y; ++batch, cursor = conformer.indexing.advance(cursor)) {
      bool contributes = false;
      Real record[Pass::kRecordValues];
      if (cursor.pair < conformer.lastPair) {
        int j;
        int k;
        conformer.indexing.atomsAt(cursor, j, k);
        contributes = Pass::compute(conformer, j, k, record);
      }
      const unsigned contributing = __ballot_sync(0xffffffffu, contributes);
      const int      count        = __popc(contributing);
      const int      steps        = (count + kRecordsPerStep - 1) / kRecordsPerStep;
      Real* const    own          = warpRecords + __popc(contributing & ((1u << lane) - 1u)) * Pass::kRecordStride;
      if (contributes) {
        Pass::store(own, record);
      }
      if (lane < steps * kRecordsPerStep - count) {
        storeZeros<Pass::kNullValues>(warpRecords + (count + lane) * Pass::kRecordStride);
      }
      __syncwarp();
#pragma unroll 2
      for (int step = 0; step < steps; ++step) {
        Pass::consume(warpRecords + step * kRecordsPerStep * Pass::kRecordStride, lane, acc, 0);
      }
      __syncwarp();
      if (contributes) {
        Pass::release(own, record);
      }
    }
  }

  if (batches.x < batches.y) {
    if constexpr (std::is_same_v<Pass, AutocorrPass<Real>> && Pass::kMma) {
      writeMmaLags(acc, lane, warpSums);
    } else if constexpr (std::is_same_v<Pass, AutocorrPass<Real>>) {
      writeWarpLags(acc, lane, warpSums);
    } else if constexpr (Pass::kSorted) {
      writeRdfTiles(acc, lane, warpSums);
    } else if constexpr (Pass::kMma) {
      writeMmaBins<Pass>(acc, lane, warpSums);
    } else {
      writeWarpBins<Pass>(acc, lane, warpSums);
    }
  } else {
    clearWarpSums(lane, Pass::kWidth, warpSums);
  }
}

/**
 * @brief Warp-collective per warp of the block. AUTOCORR3D for a molecule without bond-distance table entries: each
 * group takes rows j of the atom-pair triangle, finds the row's bond distances with searchBondDepths() into its own @p
 * depth array, then accumulates pairs (j, k > j) into the lane owning the pair's lag.
 */
template <typename Real>
__device__ __forceinline__ void runAutocorrSearch(const PairwiseConformer<Real>& conformer,
                                                  const int32_t*                 neighborStarts,
                                                  const int32_t*                 neighbors,
                                                  uint8_t*                       depth,
                                                  const int                      slice,
                                                  const int                      slices,
                                                  Real*                          warpSums) {
  const int lane        = static_cast<int>(threadIdx.x) % kWarpSize;
  const int group       = lane / kGroupSize;
  const int laneInGroup = lane % kGroupSize;
  const int numAtoms    = conformer.indexing.numAtoms;
  const int stream      = slice * kPairwiseGroups + static_cast<int>(threadIdx.x) / kGroupSize;
  uint8_t*  groupDepth  = depth + static_cast<size_t>(threadIdx.x / kGroupSize) * numAtoms;
  Real      acc[AutocorrPass<Real>::kLagSlots][kNumAutocorrChannels] = {};
  for (int j = stream; j < numAtoms - 1; j += slices * kPairwiseGroups) {
    searchBondDepths<kNumAutocorrLags>(neighborStarts,
                                       neighbors,
                                       numAtoms,
                                       j,
                                       laneInGroup,
                                       0xffu << (group * kGroupSize),
                                       groupDepth);
    Real first[kRowValues];
    loadValues<kRowValues>(conformer.rows + j * kRowStride<Real>, first);
    for (int k = j + 1; k < numAtoms; ++k) {
      const int lag = groupDepth[k];
      if (lag >= 1 && lag <= kNumAutocorrLags) {
        Real second[kRowValues];
        loadValues<kRowValues>(conformer.rows + k * kRowStride<Real>, second);
        const Real dx = first[0] - second[0];
        const Real dy = first[1] - second[1];
        const Real dz = first[2] - second[2];
        const Real r  = sqrt(dx * dx + dy * dy + dz * dz);
        Real       products[kNumAutocorrChannels];
        AutocorrPass<Real>::pairProducts(first, second, products);
        const int offset = lag - 1 - laneInGroup;
        AutocorrPass<Real>::accumulateLag(acc, products, r, offset);
        if (offset == kGroupSize) {
          AutocorrPass<Real>::accumulateChannels(acc[1], products, r);
        }
      }
    }
  }
  writeWarpLags(acc, lane, warpSums);
}

constexpr int kReduceScatterValues = kWarpSize / 2;

/**
 * @brief Warp-collective. Sums each of @p values over the warp's lanes, reduce-scattered: returns the sum of value
 *        lane / 2 (lanes 2 i and 2 i + 1 both hold value i).
 */
template <typename T>
__device__ __forceinline__ T warpReduceScatter(const T (&values)[kReduceScatterValues], const int lane) {
  T current[kReduceScatterValues];
#pragma unroll
  for (int value = 0; value < kReduceScatterValues; ++value) {
    current[value] = values[value];
  }
#pragma unroll
  for (int width = kReduceScatterValues / 2, offset = kWarpSize / 2; width >= 1; width /= 2, offset /= 2) {
    const bool upper = (lane & offset) != 0;
#pragma unroll
    for (int value = 0; value < width; ++value) {
      const T keep   = upper ? current[value + width] : current[value];
      const T send   = upper ? current[value] : current[value + width];
      current[value] = keep + __shfl_xor_sync(0xffffffffu, send, offset);
    }
  }
  return current[0] + __shfl_xor_sync(0xffffffffu, current[0], 1);
}

/**
 * @brief Block-collective. Writes the conformer's atom rows (coordinates centered on the unweighted centroid in
 *        float64, see centeredPosition(), then the channel weights) to @p rows, and its PairwiseStats.
 */
template <typename Real, unsigned kSet>
__device__ __forceinline__ void stageConformer(const ConformerAtoms& atoms,
                                               const double*         propertyWeights,
                                               const int             totalAtoms,
                                               const double*         iStateDrag,
                                               const double*         radius,
                                               Real*                 rows,
                                               PairwiseStats&        stats) {
  constexpr bool kRdf      = (kSet & kPairwiseRdf) != 0;
  constexpr bool kMorse    = (kSet & kPairwiseMorse) != 0;
  constexpr bool kAutocorr = (kSet & kPairwiseAutocorr3D) != 0;
  constexpr int  kSums     = kMorse ? 3 + 2 * kNumAtomPropertyChannels : 3;
  static_assert(kSums <= kReduceScatterValues);
  const int lane        = static_cast<int>(threadIdx.x) % kWarpSize;
  const int warp        = static_cast<int>(threadIdx.x) / kWarpSize;
  const int numAtoms    = atoms.numAtoms;
  const int activeWarps = min(kPairwiseWarps, (numAtoms + kWarpSize - 1) / kWarpSize);
  if (warp < activeWarps) {
    double   sums[kReduceScatterValues] = {};
    unsigned flags                      = 0;
    for (int atomIdx = static_cast<int>(threadIdx.x); atomIdx < numAtoms; atomIdx += kPairwiseThreads) {
#pragma unroll
      for (int axis = 0; axis < 3; ++axis) {
        const double coordinate = atoms.positions[atomIdx * 3 + axis];
        sums[axis] += coordinate;
        if constexpr (kAutocorr) {
          flags |= isfinite(coordinate) ? 0u : 1u;
        }
      }
      if constexpr (kMorse || kAutocorr) {
#pragma unroll
        for (int channel = 0; channel < kNumAtomPropertyChannels; ++channel) {
          const double weight = propertyWeights[static_cast<size_t>(channel) * totalAtoms + atomIdx];
          if constexpr (kMorse) {
            sums[3 + channel] += weight;
            sums[3 + kNumAtomPropertyChannels + channel] += weight * weight;
          }
          if constexpr (kAutocorr) {
            flags |= isnan(weight) ? 2u << channel : 0u;
          }
        }
      }
      if constexpr (kAutocorr) {
        flags |= isnan(radius[atomIdx]) ? 1u << kNumPairwiseChannels : 0u;
      }
    }
    const double total = warpReduceScatter(sums, lane);
    flags              = __reduce_or_sync(0xffffffffu, flags);
    if ((lane & 1) == 0 && lane / 2 < kSums) {
      stats.warpSums[warp][lane / 2] = total;
    }
    if (lane == 0) {
      stats.warpFlags[warp] = flags;
    }
  }
  __syncthreads();
  // Every thread combines the warps' sums itself, in warp order.
  double   centroid[3] = {};
  unsigned flags       = 0;
  for (int w = 0; w < activeWarps; ++w) {
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
      centroid[axis] += stats.warpSums[w][axis];
    }
    flags |= stats.warpFlags[w];
  }
  if constexpr (kMorse) {
    if (threadIdx.x < kNumPairwiseChannels) {
      if (threadIdx.x == 0) {
        const double n        = numAtoms;
        stats.morseZeroBin[0] = 0.5 * n * (n - 1.0);
      } else {
        double sum     = 0.0;
        double squares = 0.0;
        for (int w = 0; w < activeWarps; ++w) {
          sum += stats.warpSums[w][3 + threadIdx.x - 1];
          squares += stats.warpSums[w][3 + kNumAtomPropertyChannels + threadIdx.x - 1];
        }
        stats.morseZeroBin[threadIdx.x] = 0.5 * (sum * sum - squares);
      }
    }
  }
  if (threadIdx.x == 0) {
    // A non-finite coordinate makes every channel NaN.
    stats.autocorrNaN = (flags & 1u) != 0 ? (1u << kNumAutocorrChannels) - 1u : flags;
  }
  const double inverseAtoms = 1.0 / static_cast<double>(numAtoms);
  const double centroidX    = centroid[0] * inverseAtoms;
  const double centroidY    = centroid[1] * inverseAtoms;
  const double centroidZ    = centroid[2] * inverseAtoms;
  for (int atomIdx = static_cast<int>(threadIdx.x); atomIdx < numAtoms; atomIdx += kPairwiseThreads) {
    Real row[kRowValues];
    centeredPosition(atoms.positions, atomIdx, centroidX, centroidY, centroidZ, row[0], row[1], row[2]);
#pragma unroll
    for (int channel = 0; channel < kNumAtomPropertyChannels; ++channel) {
      row[kRowWeights + channel] =
        static_cast<Real>(propertyWeights[static_cast<size_t>(channel) * totalAtoms + atomIdx]);
    }
    row[kRowIStateDrag]     = kRdf ? static_cast<Real>(iStateDrag[atomIdx]) : Real(0);
    row[kRowCovalentRadius] = kAutocorr ? static_cast<Real>(radius[atomIdx]) : Real(0);
    row[kRowValues - 1]     = Real(0);
    storeValues<kRowValues>(rows + atomIdx * kRowStride<Real>, row);
  }
  __syncthreads();
}

/**
 * @brief Warp-collective. stageConformer() for a conformer one warp takes alone: writes its atom rows to @p rows, its
 *        MORSE bin 0 to @p zeroBin and its AUTOCORR3D NaN channels to @p autocorrNaN, with the same sums in the same
 *        order as a one-warp stageConformer().
 */
template <typename Real, unsigned kSet>
__device__ __forceinline__ void stageConformerSolo(const ConformerAtoms& atoms,
                                                   const double*         propertyWeights,
                                                   const int             totalAtoms,
                                                   const double*         iStateDrag,
                                                   const double*         radius,
                                                   Real*                 rows,
                                                   double*               zeroBin,
                                                   unsigned&             autocorrNaN) {
  constexpr bool kRdf                       = (kSet & kPairwiseRdf) != 0;
  constexpr bool kMorse                     = (kSet & kPairwiseMorse) != 0;
  constexpr bool kAutocorr                  = (kSet & kPairwiseAutocorr3D) != 0;
  const int      lane                       = static_cast<int>(threadIdx.x) % kWarpSize;
  const int      numAtoms                   = atoms.numAtoms;
  double         sums[kReduceScatterValues] = {};
  unsigned       flags                      = 0;
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
      const double coordinate = atoms.positions[atomIdx * 3 + axis];
      sums[axis] += coordinate;
      if constexpr (kAutocorr) {
        flags |= isfinite(coordinate) ? 0u : 1u;
      }
    }
    if constexpr (kMorse || kAutocorr) {
#pragma unroll
      for (int channel = 0; channel < kNumAtomPropertyChannels; ++channel) {
        const double weight = propertyWeights[static_cast<size_t>(channel) * totalAtoms + atomIdx];
        if constexpr (kMorse) {
          sums[3 + channel] += weight;
          sums[3 + kNumAtomPropertyChannels + channel] += weight * weight;
        }
        if constexpr (kAutocorr) {
          flags |= isnan(weight) ? 2u << channel : 0u;
        }
      }
    }
    if constexpr (kAutocorr) {
      flags |= isnan(radius[atomIdx]) ? 1u << kNumPairwiseChannels : 0u;
    }
  }
  // Lanes 2 v and 2 v + 1 hold value v's sum.
  const double total = warpReduceScatter(sums, lane);
  flags              = __reduce_or_sync(0xffffffffu, flags);
  if constexpr (kMorse) {
#pragma unroll
    for (int channel = 1; channel < kNumPairwiseChannels; ++channel) {
      const double sum     = __shfl_sync(0xffffffffu, total, 2 * (3 + channel - 1));
      const double squares = __shfl_sync(0xffffffffu, total, 2 * (3 + kNumAtomPropertyChannels + channel - 1));
      if (lane == 0) {
        zeroBin[channel] = 0.5 * (sum * sum - squares);
      }
    }
    if (lane == 0) {
      const double n = numAtoms;
      zeroBin[0]     = 0.5 * n * (n - 1.0);
    }
  }
  if (lane == 0) {
    // A non-finite coordinate makes every channel NaN.
    autocorrNaN = (flags & 1u) != 0 ? (1u << kNumAutocorrChannels) - 1u : flags;
  }
  const double inverseAtoms = 1.0 / static_cast<double>(numAtoms);
  const double centroidX    = __shfl_sync(0xffffffffu, total, 0) * inverseAtoms;
  const double centroidY    = __shfl_sync(0xffffffffu, total, 2) * inverseAtoms;
  const double centroidZ    = __shfl_sync(0xffffffffu, total, 4) * inverseAtoms;
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    Real row[kRowValues];
    centeredPosition(atoms.positions, atomIdx, centroidX, centroidY, centroidZ, row[0], row[1], row[2]);
#pragma unroll
    for (int channel = 0; channel < kNumAtomPropertyChannels; ++channel) {
      row[kRowWeights + channel] =
        static_cast<Real>(propertyWeights[static_cast<size_t>(channel) * totalAtoms + atomIdx]);
    }
    row[kRowIStateDrag]     = kRdf ? static_cast<Real>(iStateDrag[atomIdx]) : Real(0);
    row[kRowCovalentRadius] = kAutocorr ? static_cast<Real>(radius[atomIdx]) : Real(0);
    row[kRowValues - 1]     = Real(0);
    storeValues<kRowValues>(rows + atomIdx * kRowStride<Real>, row);
  }
  __syncwarp();
}

//! Stores output value @p index of the properties in @p kSet (output order) from its finished @p sum, rounded like
//! RDKit, or NaN for an invalid conformer.
template <typename Real, unsigned kSet>
__device__ __forceinline__ void storeFinished(const int      index,
                                              const Real     sum,
                                              const bool     valid,
                                              const int      conformerIdx,
                                              const int      numAtoms,
                                              const double*  morseZeroBin,
                                              const unsigned autocorrNaN,
                                              Real*          rdfOutput,
                                              Real*          morseOutput,
                                              Real*          autocorrOutput) {
  using Widths       = PairwiseWidths<kSet>;
  const Real invalid = static_cast<Real>(nan(""));
  if (index < Widths::kRdf) {
    rdfOutput[static_cast<size_t>(conformerIdx) * kNumRdfProperties + index] =
      valid ? static_cast<Real>(roundThousandths(sum)) : invalid;
  } else if (index < Widths::kRdf + Widths::kMorse) {
    const int local      = index - Widths::kRdf;
    const int channel    = local / kNumMorseScatterings;
    const int scattering = local % kNumMorseScatterings;
    Real      value      = invalid;
    if (valid) {
      value = scattering == 0 ? static_cast<Real>(roundThousandths(morseZeroBin[channel])) :
                                static_cast<Real>(roundThousandths(sum * (Real(1) / static_cast<Real>(scattering))));
    }
    morseOutput[static_cast<size_t>(conformerIdx) * kNumMorseProperties + local] = value;
  } else {
    const int    local        = index - Widths::kRdf - Widths::kMorse;
    const int    channel      = local / kNumAutocorrLags;
    const double orderedPairs = static_cast<double>(numAtoms) * (numAtoms - 1);
    // One atom gives 0 / 0 = NaN, as in RDKit; a channel with a NaN weight reports 0, as RDKit does.
    const double total        = ((autocorrNaN >> channel) & 1u) != 0 ? 0.0 : 2.0 * static_cast<double>(sum);
    autocorrOutput[static_cast<size_t>(conformerIdx) * kNumAutocorr3DProperties + local] =
      valid ? static_cast<Real>(roundThousandths(total / orderedPairs)) : invalid;
  }
}

/**
 * @brief Block-collective. Sums the warps' sums in warp order and, for a conformer split over several slices, the
 *        slices' sums in slice order: every slice publishes its block's sums to @p sliceSums and takes a ticket, and
 * the last slice to arrive combines them. The combining block writes each finished value, rounded like RDKit, or NaN
 * for an invalid conformer.
 */
template <typename Real, unsigned kSet>
__device__ __forceinline__ void finishPairwise(const int      conformerIdx,
                                               const int      slice,
                                               const int      slices,
                                               const int      sliceStride,
                                               const bool     valid,
                                               const int      numAtoms,
                                               PairwiseStats& stats,
                                               const Real*    warpSums,
                                               Real*          sliceSums,
                                               int32_t*       tickets,
                                               Real*          rdfOutput,
                                               Real*          morseOutput,
                                               Real*          autocorrOutput) {
  using Widths           = PairwiseWidths<kSet>;
  constexpr int kStride  = PairwiseSharedLayout<Real, kSet>::kSumStride;
  // The block's sum of output value @p index: its compact row, or its warps' rows in warp order.
  const auto    blockSum = [&](const int index) {
    if constexpr (kPairwiseCompactSums<Real, kSet>) {
      return warpSums[index];
    } else {
      Real sum = Real(0);
#pragma unroll
      for (int warp = 0; warp < kPairwiseWarps; ++warp) {
        sum += warpSums[warp * kStride + index];
      }
      return sum;
    }
  };
  __syncthreads();
  Real* const conformerSums =
    slices > 1 ? sliceSums + static_cast<size_t>(conformerIdx) * sliceStride * Widths::kTotal : nullptr;
  if (slices > 1) {
    for (int index = static_cast<int>(threadIdx.x); index < Widths::kTotal; index += kPairwiseThreads) {
      __stcg(conformerSums + slice * Widths::kTotal + index, blockSum(index));
    }
    // Publishes the sums before the ticket.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      stats.lastSlice = atomicAdd(tickets + conformerIdx, 1) == slices - 1;
    }
    __syncthreads();
    if (!stats.lastSlice) {
      return;
    }
    __threadfence();
  }
  for (int index = static_cast<int>(threadIdx.x); index < Widths::kTotal; index += kPairwiseThreads) {
    Real sum = Real(0);
    if (valid) {
      if (slices == 1) {
        sum = blockSum(index);
      } else {
        for (int other = 0; other < slices; ++other) {
          sum += __ldcg(conformerSums + other * Widths::kTotal + index);
        }
      }
    }
    storeFinished<Real, kSet>(index,
                              sum,
                              valid,
                              conformerIdx,
                              numAtoms,
                              stats.morseZeroBin,
                              stats.autocorrNaN,
                              rdfOutput,
                              morseOutput,
                              autocorrOutput);
  }
}

//! Lets the next kernel in the stream, launched with programmatic stream serialization, start before this grid ends.
__device__ __forceinline__ void allowDependentLaunch() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

//! Waits until the grids this one depends on (programmatic stream serialization) have finished and their writes are
//! visible.
__device__ __forceinline__ void waitForPrerequisiteGrids() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

//! Threads and size classes of workOrderKernel().
constexpr int kOrderThreads       = 1024;
constexpr int kOrderClasses       = 128;
//! Atom pairs per size class.
constexpr int kOrderClassPairs    = 64;
//! Conformers whose sizes each thread loads at once.
constexpr int kOrderUnroll        = 8;
//! Batches from this many conformers pack small conformers one per warp (see launchPairwiseProperties()).
constexpr int kOrderMinConformers = 2048;
//! A work item packs conformer << 8 | slices << kItemSliceBits | slice.
constexpr int kItemSliceBits      = 4;
constexpr int kItemSliceMask      = (1 << kItemSliceBits) - 1;

//! Blocks a conformer of @p numPairs atom pairs splits over, at most @p slices: only conformers with enough pairs use
//! more than one.
__host__ __device__ __forceinline__ int activeSlicesOf(const int numPairs, const int slices) {
  const int byPairs = (numPairs + kWarpSize - 1) / kWarpSize / kMinSliceBatches;
  return byPairs < 1 ? 1 : (byPairs < slices ? byPairs : slices);
}

/**
 * @brief Zeros the split conformers' @p tickets (when given) and lists the batch's work items by descending size class
 *        (kOrderClassPairs atom pairs per slice), then -1 for the rest of the grid: the pairwise kernel's blocks, which
 *        the GPU starts in index order, take the largest work first, the last blocks to start are short, and no block
 *        is launched for a slice its conformer does not use. Conformers of at most @p smallAtoms atoms are listed in
 *        @p small instead, also largest first, and each item after the (conformer, slice) items packs kPairwiseWarps
 *        of them, one per warp. The order within a class is arbitrary: it only schedules the work.
 */
__global__ void __launch_bounds__(kOrderThreads) workOrderKernel(const DeviceCoordView coordinates,
                                                                 const int             slices,
                                                                 const int             smallAtoms,
                                                                 int32_t*              items,
                                                                 int32_t*              small,
                                                                 int32_t*              tickets) {
  constexpr int  kWarps = kOrderThreads / kWarpSize;
  //! Per warp, its item count of each class (large classes, then small classes), then its first position in each.
  //! Private counts keep the atomics off the few classes every warp shares.
  __shared__ int warpOffsets[kWarps][2 * kOrderClasses];
  __shared__ int scanTotals[kWarps];
  __shared__ int totals[2];
  // The pairwise kernel launches now and waits for this grid's results (launchPairwiseSet()).
  allowDependentLaunch();
  const int  thread        = static_cast<int>(threadIdx.x);
  const int  warp          = thread / kWarpSize;
  const int  lane          = thread % kWarpSize;
  const int  numConformers = coordinates.numConformers;
  // (class key, items) of a conformer, from its atom count alone; invalid conformers take one slice.
  const auto sizeOf        = [&](const int conformerIdx) {
    const int64_t numAtoms = coordinates.atomStarts[conformerIdx + 1] - coordinates.atomStarts[conformerIdx];
    const int64_t pairs64  = numAtoms > 1 ? numAtoms * (numAtoms - 1) / 2 : 0;
    const int     numPairs = static_cast<int>(pairs64 < (int64_t{1} << 30) ? pairs64 : (int64_t{1} << 30));
    const bool    isSmall  = numAtoms <= smallAtoms;
    const int     active   = isSmall ? 1 : activeSlicesOf(numPairs, slices);
    const int     perSlice = (numPairs + active - 1) / active;
    const int     key      = kOrderClasses - 1 - min(perSlice / kOrderClassPairs, kOrderClasses - 1);
    return make_int2(isSmall ? kOrderClasses + key : key, active);
  };
  for (int index = lane; index < 2 * kOrderClasses; index += kWarpSize) {
    warpOffsets[warp][index] = 0;
  }
  __syncwarp();
  for (int base = thread; base < numConformers; base += kOrderThreads * kOrderUnroll) {
    int2 sizes[kOrderUnroll];
#pragma unroll
    for (int u = 0; u < kOrderUnroll; ++u) {
      const int conformerIdx = base + u * kOrderThreads;
      sizes[u]               = conformerIdx < numConformers ? sizeOf(conformerIdx) : make_int2(0, 0);
    }
#pragma unroll
    for (int u = 0; u < kOrderUnroll; ++u) {
      if (sizes[u].y > 0) {
        atomicAdd(&warpOffsets[warp][sizes[u].x], sizes[u].y);
        if (tickets != nullptr) {
          tickets[base + u * kOrderThreads] = 0;
        }
      }
    }
  }
  __syncthreads();
  // Exclusive scan of the counts in (class, warp) order: each thread takes kPerThread consecutive warps of one class.
  constexpr int kPerThread = kWarps * 2 * kOrderClasses / kOrderThreads;
  static_assert(kWarps % kPerThread == 0 && kPerThread * kOrderThreads == kWarps * 2 * kOrderClasses);
  const int scanClass = thread / (kWarps / kPerThread);
  const int scanWarp  = thread % (kWarps / kPerThread) * kPerThread;
  int       counts[kPerThread];
  int       sum = 0;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    counts[i] = warpOffsets[scanWarp + i][scanClass];
    sum += counts[i];
  }
  int inclusive = sum;
#pragma unroll
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
    inclusive += lane >= offset ? other : 0;
  }
  if (lane == kWarpSize - 1) {
    scanTotals[warp] = inclusive;
  }
  __syncthreads();
  if (warp == 0) {
    const int total = scanTotals[lane];
    int       scan  = total;
#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, scan, offset);
      scan += lane >= offset ? other : 0;
    }
    scanTotals[lane] = scan - total;
    // The large classes are the first half of the (class, warp) order.
    if (lane == kWarps / 2) {
      totals[0] = scan - total;
    }
    if (lane == kWarps - 1) {
      totals[1] = scan;
    }
  }
  __syncthreads();
  int start = scanTotals[warp] + inclusive - sum;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    warpOffsets[scanWarp + i][scanClass] = start;
    start += counts[i];
  }
  __syncthreads();
  const int largeItems = totals[0];
  const int smallCount = totals[1] - largeItems;
  for (int base = thread; base < numConformers; base += kOrderThreads * kOrderUnroll) {
    int2 sizes[kOrderUnroll];
#pragma unroll
    for (int u = 0; u < kOrderUnroll; ++u) {
      const int conformerIdx = base + u * kOrderThreads;
      sizes[u]               = conformerIdx < numConformers ? sizeOf(conformerIdx) : make_int2(0, 0);
    }
#pragma unroll
    for (int u = 0; u < kOrderUnroll; ++u) {
      if (sizes[u].y > 0) {
        const int conformerIdx = base + u * kOrderThreads;
        const int position     = atomicAdd(&warpOffsets[warp][sizes[u].x], sizes[u].y);
        if (sizes[u].x >= kOrderClasses) {
          small[position - largeItems] = conformerIdx;
        } else {
          for (int slice = 0; slice < sizes[u].y; ++slice) {
            items[position + slice] = conformerIdx << 8 | sizes[u].y << kItemSliceBits | slice;
          }
        }
      }
    }
  }
  // Packed items carry no slice count; their group's conformers are small[kPairwiseWarps g ..], -1 past the last.
  const int groups = (smallCount + kPairwiseWarps - 1) / kPairwiseWarps;
  for (int index = smallCount + thread; index < groups * kPairwiseWarps; index += kOrderThreads) {
    small[index] = -1;
  }
  const int gridItems = numConformers * slices;
  for (int index = largeItems + thread; index < gridItems; index += kOrderThreads) {
    const int group = index - largeItems;
    items[index]    = group < groups ? group << 8 : -1;
  }
}

//! Global scratch for conformers whose rows exceed the shared budget, and for per-row bond-distance searches.
template <typename Real> struct PairwiseScratch {
  Real*          rows        = nullptr;  //!< kRowStride<Real> values per atom per slice.
  uint8_t*       depths      = nullptr;  //!< kPairwiseGroups bytes per atom per slice.
  const int32_t* order       = nullptr;  //!< Work item of each block (workOrderKernel()).
  const int32_t* small       = nullptr;  //!< Packed small conformers, kPairwiseWarps per packed item, or -1.
  Real*          sliceSums   = nullptr;  //!< Per conformer and slice, a split conformer's block sums.
  int32_t*       tickets     = nullptr;  //!< Per conformer, slices finished; zero before the launch.
  int            sharedAtoms = 0;        //!< Largest conformer whose rows are staged in shared memory.
};

/**
 * @brief Warp-collective per warp of the block. Runs each property's pass, then calls @p afterPass(offset, width, sums)
 *        with the property's place in the output order and the warp's sums: in the warp's @p warpSums row, or with
 *        compact sums (kPairwiseCompactSums) in its records, which the next pass overwrites.
 */
template <typename Real, unsigned kSet, bool kSharedRows, typename AfterPass>
__device__ __forceinline__ void runPairwisePasses(PairwiseConformer<Real>       conformer,
                                                  const Property3DDeviceInputs& inputs,
                                                  const int                     moleculeStart,
                                                  uint8_t*                      depth,
                                                  const int                     slice,
                                                  const int                     slices,
                                                  Real*                         records,
                                                  Real*                         warpSums,
                                                  AfterPass&&                   afterPass) {
  using Widths                     = PairwiseWidths<kSet>;
  constexpr int  kRegion           = recordRegion<Real, kSet>();
  constexpr bool kCompact          = kPairwiseCompactSums<Real, kSet>;
  Real* const    ownRecords        = records + (threadIdx.x / kWarpSize) * kRegion;
  // Consecutive passes give their larger warp shares to opposite ends of the block.
  constexpr bool kMorseReversed    = (kSet & kPairwiseRdf) != 0;
  constexpr bool kAutocorrReversed = ((kSet & kPairwiseRdf) != 0) != ((kSet & kPairwiseMorse) != 0);
  if constexpr ((kSet & kPairwiseRdf) != 0) {
    Real* const sums = kCompact ? ownRecords : warpSums;
    runPairwisePass<Real, RdfPass<Real, binLanes<Real, kSet>()>>(conformer, records, kRegion, false, sums);
    afterPass(0, Widths::kRdf, sums);
  }
  if constexpr ((kSet & kPairwiseMorse) != 0) {
    Real* const sums = kCompact ? ownRecords : warpSums + Widths::kRdf;
    runPairwisePass<Real, MorsePass<Real, binLanes<Real, kSet>()>>(conformer, records, kRegion, kMorseReversed, sums);
    afterPass(Widths::kRdf, Widths::kMorse, sums);
  }
  if constexpr ((kSet & kPairwiseAutocorr3D) != 0) {
    Real* const sums = kCompact ? ownRecords : warpSums + Widths::kRdf + Widths::kMorse;
    if (conformer.lagTable != nullptr) {
      runPairwisePass<Real, AutocorrPass<Real>>(conformer, records, kRegion, kAutocorrReversed, sums);
    } else {
      runAutocorrSearch(conformer,
                        inputs.bondNeighborStarts + moleculeStart,
                        inputs.bondNeighbors,
                        depth,
                        slice,
                        slices,
                        sums);
    }
    afterPass(Widths::kRdf + Widths::kMorse, Widths::kAutocorr, sums);
  }
}

/**
 * @brief Warp-collective. One packed small conformer taken by this warp alone, with no block barriers: stages its rows
 *        in the warp's quarter of the shared rows (and its bond distances in the warp's quarter of the shared lags),
 *        runs every pass over all its pairs into the warp's sums row, and writes its finished values.
 */
template <typename Real, unsigned kSet>
__device__ __forceinline__ void pairwiseSolo(const DeviceCoordView&        coordinates,
                                             const Property3DDeviceInputs& inputs,
                                             const BondDistanceTable&      bondTable,
                                             const int                     sharedAtoms,
                                             const int                     conformerIdx,
                                             PairwiseStats&                stats,
                                             Real*                         warpSums,
                                             Real*                         records,
                                             Real*                         sharedRows,
                                             Real*                         rdfOutput,
                                             Real*                         morseOutput,
                                             Real*                         autocorrOutput) {
  using Layout                      = PairwiseSharedLayout<Real, kSet>;
  using Widths                      = PairwiseWidths<kSet>;
  constexpr bool       kAutocorr    = (kSet & kPairwiseAutocorr3D) != 0;
  const int            lane         = static_cast<int>(threadIdx.x) % kWarpSize;
  const int            warp         = static_cast<int>(threadIdx.x) / kWarpSize;
  const ConformerAtoms atoms        = loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, conformerIdx);
  constexpr bool       kCompact     = kPairwiseCompactSums<Real, kSet>;
  // Compact sums keep one block row: a packed warp stores each pass's values as soon as the pass ends instead.
  Real* const          sums         = kCompact ? nullptr : warpSums + warp * Layout::kSumStride;
  const int            quarterAtoms = sharedAtoms / kPairwiseWarps;
  if (atoms.valid) {
    const int               numAtoms      = atoms.numAtoms;
    const int               numPairs      = numAtoms * (numAtoms - 1) / 2;
    const int               moleculeIdx   = coordinates.molIndices[conformerIdx];
    const int               moleculeStart = inputs.moleculeAtomStarts[moleculeIdx];
    const int               totalAtoms    = inputs.moleculeAtomStarts[coordinates.nMols];
    Real* const             rows          = sharedRows + warp * quarterAtoms * kRowStride<Real>;
    PairwiseConformer<Real> conformer;
    conformer.indexing  = PairIndexing(numAtoms);
    conformer.firstPair = 0;
    conformer.lastPair  = numPairs;
    conformer.solo      = true;
    conformer.rows      = rows;
    conformer.lagTable  = kAutocorr ? moleculeBondDistances(bondTable, moleculeIdx) : nullptr;
    if constexpr (kAutocorr) {
      // The staging's warp barrier publishes the copy.
      constexpr int kLagShare = kPairwiseLagBudget / kPairwiseWarps;
      if (conformer.lagTable != nullptr && numPairs <= kLagShare) {
        uint8_t* const sharedLags =
          reinterpret_cast<uint8_t*>(sharedRows) + Layout::rowBytes(sharedAtoms) + warp * kLagShare;
        for (int entry = lane; entry < numPairs; entry += kWarpSize) {
          sharedLags[entry] = __ldg(conformer.lagTable + entry);
        }
        conformer.lagTable = sharedLags;
      }
    }
    stageConformerSolo<Real, kSet>(atoms,
                                   inputs.atomPropertyWeights + moleculeStart,
                                   totalAtoms,
                                   (kSet & kPairwiseRdf) != 0 ? inputs.iStateDragWeights + moleculeStart : nullptr,
                                   kAutocorr ? inputs.covalentRadiusWeights + moleculeStart : nullptr,
                                   rows,
                                   stats.soloZeroBin[warp],
                                   stats.soloNaN[warp]);
    runPairwisePasses<Real, kSet, true>(conformer,
                                        inputs,
                                        moleculeStart,
                                        nullptr,
                                        0,
                                        1,
                                        records,
                                        sums,
                                        [&](const int offset, const int width, const Real* passSums) {
                                          if constexpr (kCompact) {
                                            __syncwarp();
                                            for (int index = lane; index < width; index += kWarpSize) {
                                              storeFinished<Real, kSet>(offset + index,
                                                                        passSums[index],
                                                                        true,
                                                                        conformerIdx,
                                                                        numAtoms,
                                                                        stats.soloZeroBin[warp],
                                                                        stats.soloNaN[warp],
                                                                        rdfOutput,
                                                                        morseOutput,
                                                                        autocorrOutput);
                                            }
                                            // The next pass overwrites the records only once every lane has read them.
                                            __syncwarp();
                                          }
                                        });
    __syncwarp();
    if constexpr (kCompact) {
      return;
    }
  }
  for (int index = lane; index < Widths::kTotal; index += kWarpSize) {
    storeFinished<Real, kSet>(index,
                              atoms.valid ? sums[index] : Real(0),
                              atoms.valid,
                              conformerIdx,
                              atoms.numAtoms,
                              stats.soloZeroBin[warp],
                              stats.soloNaN[warp],
                              rdfOutput,
                              morseOutput,
                              autocorrOutput);
  }
}

/**
 * @brief The pairwise properties in @p kSet (PairwiseSet bits) of one slice of one conformer per block; a conformer's
 *        @p slices consecutive blocks split its atom pairs evenly (as many as its size warrants, see kMinSliceBatches).
 *
 * RDF bin i adds exp(-100 (R_i - r)^2) with R_i = 1 + 0.5 i; MORSE bin i adds sin(i r) / (i r), or 1 for i = 0. Each
 * term is scaled by the pair product of the channel's atom weights. RDF's I-state channel uses
 * `inputs.iStateDragWeights`; MORSE's uses the I-state block of `inputs.atomPropertyWeights`, matching RDKit.
 *
 * AUTOCORR3D lag l (1-10) adds r * w_j * w_k over pairs whose bond-count distance is l, for the MORSE channels plus
 * covalent radius, with distances from the shared bond-distance table (or per-row searches for molecules without
 * entries). RDKit evaluates each sum as w^T (B_l .* D) w, so a NaN weight anywhere in a channel, or a non-finite
 * coordinate anywhere in the conformer, makes every affected lag NaN, which RDKit then reports as 0; the kernel
 * reproduces that per channel. Sums count both orders of each pair and are divided by n (n - 1).
 *
 * The block first stages one row per atom (coordinates centered in float64, then the channel weights), so the pair
 * passes read only @p Real values; each property then takes one pass over the slice's pairs (runPairwisePass()).
 */
template <typename Real, unsigned kSet>
__global__ void __launch_bounds__(kPairwiseThreads, (pairwiseMinBlocks<Real, kSet>()))
  pairwise3DKernel(const DeviceCoordView        coordinates,
                   const Property3DDeviceInputs inputs,
                   const BondDistanceTable      bondTable,
                   const PairwiseScratch<Real>  scratch,
                   const int                    slices,
                   Real* __restrict__ rdfOutput,
                   Real* __restrict__ morseOutput,
                   Real* __restrict__ autocorrOutput) {
  using Layout = PairwiseSharedLayout<Real, kSet>;
  extern __shared__ __align__(16) unsigned char pairwiseShared[];
  PairwiseStats&                                stats = *reinterpret_cast<PairwiseStats*>(pairwiseShared);
  Real* const warpSums   = reinterpret_cast<Real*>(pairwiseShared + Layout::kPartialOffset);
  Real* const records    = reinterpret_cast<Real*>(pairwiseShared + Layout::kRecordOffset);
  Real* const sharedRows = reinterpret_cast<Real*>(pairwiseShared + Layout::kRowOffset);

  // Launched while workOrderKernel() runs: its work items are visible after the wait. Items past the batch's last one
  // exit.
  waitForPrerequisiteGrids();
  const int item = scratch.order[blockIdx.x];
  if (item < 0) {
    return;
  }
  const int conformerIdx = item >> 8;
  const int activeSlices = (item >> kItemSliceBits) & kItemSliceMask;
  const int slice        = item & kItemSliceMask;
  if (activeSlices == 0) {
    // A packed item: each warp takes one small conformer alone.
    const int smallConformer = scratch.small[conformerIdx * kPairwiseWarps + static_cast<int>(threadIdx.x) / kWarpSize];
    if (smallConformer >= 0) {
      pairwiseSolo<Real, kSet>(coordinates,
                               inputs,
                               bondTable,
                               scratch.sharedAtoms,
                               smallConformer,
                               stats,
                               warpSums,
                               records,
                               sharedRows,
                               rdfOutput,
                               morseOutput,
                               autocorrOutput);
    }
    return;
  }
  const ConformerAtoms atoms    = loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, conformerIdx);
  const int            numPairs = atoms.valid ? atoms.numAtoms * (atoms.numAtoms - 1) / 2 : 0;
  if (atoms.valid) {
    const int     numAtoms      = atoms.numAtoms;
    const int64_t atomOffset    = (atoms.positions - coordinates.positions) / 3;
    const int     moleculeIdx   = coordinates.molIndices[conformerIdx];
    const int     moleculeStart = inputs.moleculeAtomStarts[moleculeIdx];
    const int     totalAtoms    = inputs.moleculeAtomStarts[coordinates.nMols];
    const bool    useSharedRows = numAtoms <= scratch.sharedAtoms;
    Real* const   rows =
      useSharedRows ? sharedRows :
                        scratch.rows + (atomOffset * slices + static_cast<int64_t>(slice) * numAtoms) * kRowStride<Real>;
    PairwiseConformer<Real> conformer;
    conformer.indexing  = PairIndexing(numAtoms);
    conformer.firstPair = static_cast<int>(static_cast<int64_t>(numPairs) * slice / activeSlices);
    conformer.lastPair  = static_cast<int>(static_cast<int64_t>(numPairs) * (slice + 1) / activeSlices);
    conformer.lagTable  = (kSet & kPairwiseAutocorr3D) != 0 ? moleculeBondDistances(bondTable, moleculeIdx) : nullptr;
    if constexpr ((kSet & kPairwiseAutocorr3D) != 0) {
      // The staging's barriers publish the copy.
      if (conformer.lagTable != nullptr && numPairs <= kPairwiseLagBudget) {
        uint8_t* const sharedLags = reinterpret_cast<uint8_t*>(sharedRows) + Layout::rowBytes(scratch.sharedAtoms);
        for (int entry = static_cast<int>(threadIdx.x); entry < numPairs; entry += kPairwiseThreads) {
          sharedLags[entry] = __ldg(conformer.lagTable + entry);
        }
        conformer.lagTable = sharedLags;
      }
    }
    stageConformer<Real, kSet>(
      atoms,
      inputs.atomPropertyWeights + moleculeStart,
      totalAtoms,
      (kSet & kPairwiseRdf) != 0 ? inputs.iStateDragWeights + moleculeStart : nullptr,
      (kSet & kPairwiseAutocorr3D) != 0 ? inputs.covalentRadiusWeights + moleculeStart : nullptr,
      rows,
      stats);
    uint8_t* const depth =
      scratch.depths == nullptr ?
        nullptr :
        scratch.depths + (atomOffset * slices + static_cast<int64_t>(slice) * numAtoms) * kPairwiseGroups;
    constexpr bool kCompact     = kPairwiseCompactSums<Real, kSet>;
    Real* const    warpRows     = kCompact ? warpSums : warpSums + (threadIdx.x / kWarpSize) * Layout::kSumStride;
    // Compact sums: the block adds up each pass's warp sums, in warp order, into its one row before the next pass.
    const auto     combineWarps = [&](const int offset, const int width, const Real*) {
      if constexpr (kCompact) {
        constexpr int kRegion = recordRegion<Real, kSet>();
        __syncthreads();
        for (int index = static_cast<int>(threadIdx.x); index < width; index += kPairwiseThreads) {
          Real sum = Real(0);
#pragma unroll
          for (int warp = 0; warp < kPairwiseWarps; ++warp) {
            sum += records[warp * kRegion + index];
          }
          warpSums[offset + index] = sum;
        }
        __syncthreads();
      }
    };
    if (useSharedRows) {
      conformer.rows = sharedRows;
      runPairwisePasses<Real, kSet, true>(conformer,
                                          inputs,
                                          moleculeStart,
                                          depth,
                                          slice,
                                          activeSlices,
                                          records,
                                          warpRows,
                                          combineWarps);
    } else {
      conformer.rows = rows;
      runPairwisePasses<Real, kSet, false>(conformer,
                                           inputs,
                                           moleculeStart,
                                           depth,
                                           slice,
                                           activeSlices,
                                           records,
                                           warpRows,
                                           combineWarps);
    }
  }
  finishPairwise<Real, kSet>(conformerIdx,
                             slice,
                             activeSlices,
                             slices,
                             atoms.valid,
                             atoms.numAtoms,
                             stats,
                             warpSums,
                             scratch.sliceSums,
                             scratch.tickets,
                             rdfOutput,
                             morseOutput,
                             autocorrOutput);
}

//! Output rows of the pairwise properties; null for properties that were not requested.
template <typename Real> struct PairwiseOutputs {
  Real* rdf        = nullptr;
  Real* morse      = nullptr;
  Real* autocorr3D = nullptr;
};

/**
 * @brief Blocks per conformer: the batch's largest conformer splits until a slice is no more than the average work of
 *        one of @p blockSlots resident blocks, so it does not finish long after the rest; kMinSliceBatches decides how
 *        many of them each conformer uses. The average comes from the fewest atom pairs the batch can have,
 *        numAtoms^2 / (2 numConformers) (all conformers of equal size), so splits err towards more slices. Float pairs
 *        are cheap enough that, from kOrderMinConformers conformers, the extra blocks and slice combines cost more than
 *        the shorter last conformer saves, so float batches that large, like batches without a known largest
 *        molecule, keep conformers whole.
 */
template <typename Real>
int pairwiseSlices(const DeviceCoordView& coordinates, const int maxAtoms, const int blockSlots) {
  if (maxAtoms <= 1 || (std::is_same_v<Real, float> && coordinates.numConformers >= kOrderMinConformers)) {
    return coordinates.numConformers >= kOrderMinConformers ?
             1 :
             (coordinates.numConformers >= 1024 ? 2 : kPairwiseMaxSlices<Real>);
  }
  const double atoms      = static_cast<double>(coordinates.numAtoms);
  const double leastPairs = atoms * atoms / (2.0 * coordinates.numConformers);
  const double largest    = 0.5 * maxAtoms * (maxAtoms - 1.0);
  const double slices     = std::ceil(largest * blockSlots / std::max(leastPairs, 1.0));
  return static_cast<int>(std::min<double>(std::max(slices, 1.0), kPairwiseMaxSlices<Real>));
}

template <typename Real, unsigned kSet>
void launchPairwiseSet(const DeviceCoordView&        coordinates,
                       const Property3DDeviceInputs& inputs,
                       const PairwiseScratch<Real>&  scratch,
                       const int                     slices,
                       const BondDistanceTable&      bondTable,
                       const PairwiseOutputs<Real>&  outputs,
                       const cudaStream_t            stream) {
  auto* const  kernel      = pairwise3DKernel<Real, kSet>;
  const size_t sharedBytes = PairwiseSharedLayout<Real, kSet>::bytes(scratch.sharedAtoms);
  // Set per launch: the attribute is per device, and the layout depends on the batch's largest molecule.
  cudaCheckError(
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes)));
  // Programmatic stream serialization: the blocks start while workOrderKernel() runs and wait for its work items.
  cudaLaunchAttribute attribute;
  attribute.id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config                            = {};
  config.gridDim          = dim3(static_cast<unsigned>(coordinates.numConformers) * slices);
  config.blockDim         = dim3(kPairwiseThreads);
  config.dynamicSmemBytes = sharedBytes;
  config.stream           = stream;
  config.attrs            = &attribute;
  config.numAttrs         = 1;
  cudaCheckError(cudaLaunchKernelEx(&config,
                                    kernel,
                                    coordinates,
                                    inputs,
                                    bondTable,
                                    scratch,
                                    slices,
                                    outputs.rdf,
                                    outputs.morse,
                                    outputs.autocorr3D));
}

//! Launches the pairwise kernel when RDF, MORSE or AUTOCORR3D is requested; one launch serves all of them.
template <typename Real>
void launchPairwiseProperties(const DeviceCoordView&        coordinates,
                              const Property3DDeviceInputs& inputs,
                              const PairwiseOutputs<Real>&  outputs,
                              const BondDistanceTable&      bondTable,
                              const cudaStream_t            stream) {
  const unsigned set = (outputs.rdf != nullptr ? kPairwiseRdf : 0u) | (outputs.morse != nullptr ? kPairwiseMorse : 0u) |
                       (outputs.autocorr3D != nullptr ? kPairwiseAutocorr3D : 0u);
  if (coordinates.numConformers == 0 || set == 0) {
    return;
  }
  const int maxAtoms       = inputs.maxMoleculeAtoms;
  int       device         = 0;
  int       sms            = 0;
  int       sharedPerBlock = 0;
  cudaCheckError(cudaGetDevice(&device));
  cudaCheckError(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  cudaCheckError(cudaDeviceGetAttribute(&sharedPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  // Resident blocks of the pairwise kernel, at the fewest blocks per SM any instantiation allows.
  const int             blockSlots = sms * 4;
  const int             slices     = pairwiseSlices<Real>(coordinates, maxAtoms, blockSlots);
  const int             rowCap     = pairwiseRowCap<Real>(static_cast<size_t>(sharedPerBlock));
  // An unknown largest molecule stages up to 256 atoms in shared memory and reads larger ones from global scratch.
  PairwiseScratch<Real> scratch;
  scratch.sharedAtoms = maxAtoms > 0 ? std::min(maxAtoms, rowCap) : std::min(256, rowCap);
  AsyncDeviceVector<Real> rowScratch(maxAtoms > 0 && maxAtoms <= scratch.sharedAtoms ?
                                       0 :
                                       static_cast<size_t>(coordinates.numAtoms) * slices * kRowStride<Real>,
                                     stream);
  scratch.rows              = rowScratch.data();
  // Per-row searches need depth arrays unless the table provably covers every molecule.
  const int64_t maxPairs    = static_cast<int64_t>(std::max(maxAtoms, 0)) * (maxAtoms - 1) / 2;
  const int64_t neededPairs = inputs.moleculeAtomPairs > 0 ? inputs.moleculeAtomPairs : maxPairs * coordinates.nMols;
  const bool    tableCovers = bondTable.distances != nullptr && maxAtoms > 0 && bondTable.capacity >= neededPairs;
  AsyncDeviceVector<uint8_t> depthScratch((set & kPairwiseAutocorr3D) != 0 && !tableCovers ?
                                            static_cast<size_t>(coordinates.numAtoms) * slices * kPairwiseGroups :
                                            0,
                                          stream);
  scratch.depths = depthScratch.data();
  // Split conformers combine their slices through global sums; every conformer's ticket starts at zero.
  AsyncDeviceVector<Real> sliceSumStorage(
    slices > 1 ? static_cast<size_t>(coordinates.numConformers) * slices *
                   PairwiseWidths<kPairwiseRdf | kPairwiseMorse | kPairwiseAutocorr3D>::kTotal :
                 0,
    stream);
  AsyncDeviceVector<int32_t> ticketStorage(slices > 1 ? static_cast<size_t>(coordinates.numConformers) : 0, stream);
  if (slices > 1) {
    scratch.sliceSums = sliceSumStorage.data();
    scratch.tickets   = ticketStorage.data();
  }
  // Every batch starts its largest slices first and launches no block for a slice its conformer does not use. Batches
  // of kOrderMinConformers or more pack small conformers, whose rows fit a quarter of the shared rows, one per warp,
  // unless AUTOCORR3D may need per-row searches (a block-collective fallback). Smaller batches keep every conformer on
  // a whole block, which shortens the last blocks to finish.
  const bool packs =
    coordinates.numConformers >= kOrderMinConformers && ((set & kPairwiseAutocorr3D) == 0 || tableCovers);
  const int                  smallAtoms = packs ? scratch.sharedAtoms / kPairwiseWarps : 0;
  const size_t               gridItems  = static_cast<size_t>(coordinates.numConformers) * slices;
  AsyncDeviceVector<int32_t> orderStorage(gridItems + coordinates.numConformers + kPairwiseWarps, stream);
  scratch.order = orderStorage.data();
  scratch.small = orderStorage.data() + gridItems;
  workOrderKernel<<<1, kOrderThreads, 0, stream>>>(coordinates,
                                                   slices,
                                                   smallAtoms,
                                                   orderStorage.data(),
                                                   orderStorage.data() + gridItems,
                                                   scratch.tickets);
  cudaCheckError(cudaGetLastError());
  switch (set) {
    case kPairwiseRdf:
      launchPairwiseSet<Real, kPairwiseRdf>(coordinates, inputs, scratch, slices, bondTable, outputs, stream);
      break;
    case kPairwiseMorse:
      launchPairwiseSet<Real, kPairwiseMorse>(coordinates, inputs, scratch, slices, bondTable, outputs, stream);
      break;
    case kPairwiseRdf | kPairwiseMorse:
      launchPairwiseSet<Real, kPairwiseRdf | kPairwiseMorse>(coordinates,
                                                             inputs,
                                                             scratch,
                                                             slices,
                                                             bondTable,
                                                             outputs,
                                                             stream);
      break;
    case kPairwiseAutocorr3D:
      launchPairwiseSet<Real, kPairwiseAutocorr3D>(coordinates, inputs, scratch, slices, bondTable, outputs, stream);
      break;
    case kPairwiseRdf | kPairwiseAutocorr3D:
      launchPairwiseSet<Real, kPairwiseRdf | kPairwiseAutocorr3D>(coordinates,
                                                                  inputs,
                                                                  scratch,
                                                                  slices,
                                                                  bondTable,
                                                                  outputs,
                                                                  stream);
      break;
    case kPairwiseMorse | kPairwiseAutocorr3D:
      launchPairwiseSet<Real, kPairwiseMorse | kPairwiseAutocorr3D>(coordinates,
                                                                    inputs,
                                                                    scratch,
                                                                    slices,
                                                                    bondTable,
                                                                    outputs,
                                                                    stream);
      break;
    default:
      launchPairwiseSet<Real, kPairwiseRdf | kPairwiseMorse | kPairwiseAutocorr3D>(coordinates,
                                                                                   inputs,
                                                                                   scratch,
                                                                                   slices,
                                                                                   bondTable,
                                                                                   outputs,
                                                                                   stream);
      break;
  }
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_PAIRWISE_CUH
