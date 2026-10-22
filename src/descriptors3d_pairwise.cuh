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

//! Double sums use the m8n8k4 float64 MMA, which needs sm_80; older devices accumulate them like float sums.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
constexpr bool kDoubleMma = false;
#else
constexpr bool kDoubleMma = true;
#endif

//! Per-atom scratch row written once per conformer: centered x, y, z, the atom-property channels (the last one
//! MORSE's I-state), RDF's I-state, then AUTOCORR3D's covalent radius and padding to whole 16-byte vectors, all as the
//! compute type.
constexpr int kRowWeights        = 3;
constexpr int kRowIStateDrag     = kRowWeights + kNumAtomPropertyChannels;
constexpr int kRowCovalentRadius = kRowIStateDrag + 1;
constexpr int kRowValues         = 12;
static_assert(kRowCovalentRadius < kRowValues);

//! Values of @p Real per 16-byte vector.
template <typename Real> constexpr int kVector = 16 / static_cast<int>(sizeof(Real));

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

//! acc[slot][channel] += products[channel] * terms[slot] for every bin slot and channel.
template <typename Real, int kSlots>
__device__ __forceinline__ void accumulateOuterProduct(Real (&acc)[kSlots][kNumPairwiseChannels],
                                                       const Real (&products)[kNumPairwiseChannels],
                                                       const Real (&terms)[kSlots]) {
#pragma unroll
  for (int slot = 0; slot < kSlots; ++slot) {
#pragma unroll
    for (int channel = 0; channel < kNumPairwiseChannels; ++channel) {
      acc[slot][channel] = fma(products[channel], terms[slot], acc[slot][channel]);
    }
  }
}

//! A lane's atom pair (j, k > j), walked in row-major order of the pair triangle.
struct PairCursor {
  int pair = 0;
  int j    = 0;
  int k    = 0;

  //! Moves @p steps pairs on along the rows of a conformer of @p numAtoms atoms.
  __device__ __forceinline__ void advance(const int steps, const int numAtoms) {
    pair += steps;
    k += steps;
    while (k >= numAtoms && j < numAtoms - 2) {
      k -= numAtoms - j - 2;
      ++j;
    }
  }
};

//! What every pass over a conformer's atom pairs reads: its rows, its pairs and its bond distances.
template <typename Real> struct PairwiseConformer {
  const Real*    rows     = nullptr;  //!< kRowValues values per atom.
  int            numAtoms = 0;
  int            numPairs = 0;
  //! The molecule's bond-distance entries, or null to search per row.
  const uint8_t* lagTable = nullptr;
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
  loadValues<kValues>(rows + j * kRowValues, pair.first);
  loadValues<kValues>(rows + k * kRowValues, pair.second);
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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%0, %1};"
               : "+d"(acc[0]), "+d"(acc[1])
               : "d"(a), "d"(b));
#else
  __trap();  // Not reached: kDoubleMma is false here.
#endif
}

/**
 * @brief MORSE bins 1-31: bin s adds w_j * w_k * sin(s r) / (s r) per channel. Records hold the weight products over
 *        r; the 1 / s factor is applied to the finished sums. Bin 0 is computed from the weights in float64, see
 *        stageConformer().
 *
 * Float records carry r; the kLanes lanes of a record own scattering values l, l + kLanes, ... and each takes
 * sin(s r) for its own. Double records carry sin(l r) for l = 0..8 and 2 cos(8 r), from one double sincos by the
 * three-term recurrence; double sums are m8n8k4 MMA fragments (channels x pairs times pairs x scattering values), lane
 * l continuing sin(l' r), l' = l / 4, of its pair l % 4 with sin((l' + 8) r) = 2 cos(8 r) sin(l' r) - sin((l' - 8) r).
 * In double the recurrence stays within a few ulp of sin(s r) over these few steps.
 */
template <typename Real> struct MorsePass {
  static constexpr bool kFloat          = std::is_same_v<Real, float>;
  static constexpr bool kMma            = !kFloat && kDoubleMma;
  static constexpr int  kLanes          = kGroupSize;
  static constexpr int  kSlots          = kNumMorseScatterings / kLanes;
  static constexpr int  kRecordsPerStep = kMma ? kMmaPairs : kWarpSize / kLanes;
  static constexpr int  kRowLoad        = kFloat ? 12 : 10;
  static constexpr int  kSines          = 8;
  //! Float: Q0..Q6, r. Double: Q0..Q6, 2 cos(8 r), sin(l r) for l = 0..8.
  static constexpr int  kRecordValues   = !kMma ? 8 : 18;
  //! Keeps the records a warp reads at once on distinct banks.
  static constexpr int  kRecordStride   = !kMma ? 12 : 20;
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
    if constexpr (!kMma) {
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
                                                   const bool contributes) {
    if (contributes) {
      store(target, record);
    } else {
      storeZeros<kNullValues>(target);
    }
  }

  //! One step: the kRecordsPerStep records from @p records.
  __device__ __forceinline__ static void consume(const Real* records, const int lane, Accumulators& acc) {
    if constexpr (!kMma) {
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
        if constexpr (kFloat) {
          sines[slot] = sinf(static_cast<float>(binOf(laneInGroup, slot)) * head[kNumPairwiseChannels]);
        } else {
          sines[slot] = sin(static_cast<double>(binOf(laneInGroup, slot)) * head[kNumPairwiseChannels]);
        }
      }
      accumulateOuterProduct(acc, products, sines);
    } else {
      const double* record      = records + (lane % kMmaPairs) * kRecordStride;
      const int     row         = lane / kMmaPairs;
      const double  product     = record[min(row, kNumPairwiseChannels - 1)];
      const double  twiceCosine = record[kNumPairwiseChannels];
      double        sines[kSlots];
      sines[0] = record[kSineOffset + row];
      // sin((l - 8) r) = -sin((8 - l) r).
      sines[1] = fma(twiceCosine, sines[0], record[kSineOffset + kSines - row]);
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
  static_assert(!kMma || kSineOffset + kSines + 1 <= kRecordValues);
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

template <typename Real> struct RdfPass {
  static constexpr bool kFloat          = std::is_same_v<Real, float>;
  static constexpr bool kMma            = !kFloat && kDoubleMma;
  //! Double batches store records sorted by tile; see runPairwisePass().
  static constexpr bool kSorted         = kMma;
  static constexpr int  kLanes          = kGroupSize;
  static constexpr int  kSlots          = kNumMorseScatterings / kLanes;
  static constexpr int  kRecordsPerStep = kMma ? kMmaPairs : kWarpSize / kLanes;
  static constexpr int  kWindow         = 4;
  static constexpr int  kRowLoad        = kFloat ? 12 : 10;
  //! Products of channels 0-6 (channel 0's is 1, the last is RDF's I-state), window start (int), then the terms.
  static constexpr int  kStartOffset    = kNumPairwiseChannels;
  static constexpr int  kTermOffset     = 8;
  static constexpr int  kTerms          = !kMma ? kNumMorseScatterings : kRdfTileBins;
  static constexpr int  kRecordValues   = kTermOffset + kWindow;
  //! Float: keeps the records a warp reads at once on distinct banks. Double: values per transposed slot, the products
  //! (row 7 of A zero) then the tile's terms.
  static constexpr int  kRecordStride   = !kMma ? kTermOffset + kTerms + 4 : kTermOffset + kRdfTileBins;
  //! Sorted batches read up to kRecordsPerStep - 1 slots past their last record.
  static constexpr int  kRecordSlots    = kSorted ? kWarpSize + kRecordsPerStep : kWarpSize;
  //! Values a null record zeroes; float terms stay zeroed between records.
  static constexpr int  kNullValues     = !kMma ? kTermOffset : kRecordStride;
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
    if constexpr (!kMma) {
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
    if constexpr (!kMma) {
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
                                                   const bool contributes) {
    if (contributes) {
      store(target, record);
    } else {
      storeZeros<kNullValues>(target);
    }
  }

  //! Float: one step, the kRecordsPerStep records from @p records.
  __device__ __forceinline__ static void consume(const Real* records, const int lane, Accumulators& acc) {
    if constexpr (!kMma) {
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
 * Batches are compacted to the pairs within kNumAutocorrLags bonds. Float lane l of a record's kLanes lanes owns lags
 * l + 1 and l + 1 + kLanes for every channel, so each record updates one lane's sums. Double sums are m8n8k4 MMAs:
 * channels x pairs times pairs x lags, B holding r in the pair's lag column; the second tile (lags 9-10) runs only
 * when a pair of the step needs it.
 */
template <typename Real> struct AutocorrPass {
  static constexpr bool kFloat          = std::is_same_v<Real, float>;
  static constexpr bool kMma            = !kFloat && kDoubleMma;
  static constexpr int  kLanes          = kGroupSize;
  static constexpr int  kRecordsPerStep = kMma ? kMmaPairs : kWarpSize / kLanes;
  static constexpr int  kRowLoad        = 12;
  //! Products of channels 0-7, r, lag (int), padding.
  static constexpr int  kRecordValues   = kFloat ? 12 : 10;
  //! Keeps the records a warp reads at once on distinct banks.
  static constexpr int  kRecordStride   = 12;
  static constexpr int  kNullValues     = kRecordStride;
  static constexpr bool kPipelined      = false;
  static constexpr bool kSorted         = false;
  static constexpr int  kWidth          = kNumAutocorr3DProperties;
  static constexpr int  kLagSlots       = (kNumAutocorrLags + kLanes - 1) / kLanes;
  using Accumulators = std::conditional_t<kMma, double[kLagSlots][2], Real[kLagSlots][kNumAutocorrChannels]>;
  static_assert(kLagSlots == 2 && kNumAutocorrChannels == 8);

  __device__ __forceinline__ static bool compute(const PairwiseConformer<Real>& conformer,
                                                 const int                      j,
                                                 const int                      k,
                                                 Real (&record)[kRecordValues]) {
    const int lag = conformer.lagTable[bondTableRowStart(j, conformer.numAtoms) + (k - j - 1)];
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
  //! Warp-collective step: the kRecordsPerStep records from @p records. Null records (lag 0) update nothing.
  __device__ __forceinline__ static void consume(const Real* records, const int lane, Accumulators& acc) {
    if constexpr (kMma) {
      const double* record = records + (lane % kMmaPairs) * kRecordStride;
      const int     row    = lane / kMmaPairs;
      double        tail[kVector<double>];
      loadValues<kVector<double>>(record + kNumAutocorrChannels, tail);
      const int    lag     = realAsInt(tail[1]);
      const double product = record[row];
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
    for (int channel = 0; channel < kNumAutocorrChannels; ++channel) {
      sums[channel] = fma(products[channel], r, sums[channel]);
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
  return values;
}

//! Output values of the properties in @p kSet, in output order: RDF, MORSE, AUTOCORR3D.
template <unsigned kSet> struct PairwiseWidths {
  static constexpr int kRdf      = (kSet & kPairwiseRdf) != 0 ? kNumRdfProperties : 0;
  static constexpr int kMorse    = (kSet & kPairwiseMorse) != 0 ? kNumMorseProperties : 0;
  static constexpr int kAutocorr = (kSet & kPairwiseAutocorr3D) != 0 ? kNumAutocorr3DProperties : 0;
  static constexpr int kTotal    = kRdf + kMorse + kAutocorr;
};

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

//! Writes one warp's bin sums, laid out like the property's output row (channel-major), to @p store(index, sum).
template <typename Pass, typename Store>
__device__ __forceinline__ void writeWarpBins(const typename Pass::Accumulators& acc, const int lane, Store&& store) {
  std::remove_cv_t<std::remove_reference_t<decltype(acc[0][0])>> sums[kNumPairwiseChannels];
  reduceBinsAcrossGroups<Pass::kLanes>(acc, lane, sums);
  const int bin = Pass::binOf(lane % Pass::kLanes, lane / Pass::kLanes);
  if (bin < Pass::kBins) {
#pragma unroll
    for (int channel = 0; channel < kNumPairwiseChannels; ++channel) {
      store(channel * Pass::kBins + bin, sums[channel]);
    }
  }
}

//! Writes one warp's MMA fragments, laid out like the property's output row (channel-major), to @p store(index, sum).
template <typename Pass, typename Store>
__device__ __forceinline__ void writeMmaBins(const typename Pass::Accumulators& acc, const int lane, Store&& store) {
  const int channel = lane / kMmaPairs;
#pragma unroll
  for (int tile = 0; tile < kNumMorseScatterings / 8; ++tile) {
#pragma unroll
    for (int column = 0; column < 2; ++column) {
      const int bin = 8 * tile + 2 * (lane % kMmaPairs) + column;
      if (channel < kNumPairwiseChannels && bin < Pass::kBins) {
        store(channel * Pass::kBins + bin, acc[tile][column]);
      }
    }
  }
}

//! Writes one warp's RDF tile fragments (lane l: channel l / 4, tile positions 2 (l % 4) + i), laid out like the output
//! row (channel-major), to @p store(index, sum). Bin 4 t + p (p < 4) adds tile t - 1's position p + 4, held two lanes
//! up.
template <typename Store>
__device__ __forceinline__ void writeRdfTiles(const double (&acc)[kRdfTiles][2], const int lane, Store&& store) {
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
        store(channel * kNumRdfRadii + 4 * tile + 2 * quad + column, acc[tile][column] + carry[column]);
      }
      carry[column] = upper;
    }
  }
  // Bins 28 and 29: positions 4 and 5 of the last tile.
  if (quad == 0 && written) {
    store(channel * kNumRdfRadii + 4 * kRdfTiles, carry[0]);
    store(channel * kNumRdfRadii + 4 * kRdfTiles + 1, carry[1]);
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
//! like the output row (channel-major), to @p store(index, sum).
template <typename Store>
__device__ __forceinline__ void writeMmaLags(const double (&acc)[2][2], const int lane, Store&& store) {
  const int channel = lane / kMmaPairs;
#pragma unroll
  for (int tile = 0; tile < 2; ++tile) {
#pragma unroll
    for (int column = 0; column < 2; ++column) {
      const int lag = 8 * tile + 2 * (lane % kMmaPairs) + column;
      if (lag < kNumAutocorrLags) {
        store(channel * kNumAutocorrLags + lag, acc[tile][column]);
      }
    }
  }
}

/**
 * @brief Writes one warp's AUTOCORR3D sums, laid out like the output row (channel-major), to @p store(index, sum). The
 *        sums are reduce-scattered over the groups: group g ends with lag slot g / 2 of channels 4 (g % 2) .. + 3.
 */
template <typename Real, typename Store>
__device__ __forceinline__ void writeWarpLags(const Real (&acc)[AutocorrPass<Real>::kLagSlots][kNumAutocorrChannels],
                                              const int lane,
                                              Store&&   store) {
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
      store(((odd ? kHalf : 0) + channel) * kNumAutocorrLags + lag, sum);
    }
  }
}

/**
 * @brief Computes pair @p cursor's record without branching on it: pairs past the conformer's last pair compute its
 *        first pair and report that they do not contribute.
 */
template <typename Pass, typename Real>
__device__ __forceinline__ bool computePair(const PairwiseConformer<Real>& conformer,
                                            const PairCursor&              cursor,
                                            Real (&record)[Pass::kRecordValues]) {
  const bool valid = cursor.pair < conformer.numPairs;
  return Pass::compute(conformer, valid ? cursor.j : 0, valid ? cursor.k : 1, record) && valid;
}

/**
 * @brief Warp-collective. Walks the conformer's atom pairs in kWarpSize-pair batches for one property: every lane
 *        writes one pair's record (Pass::compute()) to the warp's @p records, compacted to the pairs that contribute
 *        and padded with zero records to whole steps, then the warp's groups of Pass::kLanes lanes take one record
 *        each per step, each lane accumulating the bins it owns. The sums go to @p store(index, sum), indexed like the
 *        property's output row.
 *
 * Passes whose every pair contributes (Pass::kPipelined) instead keep a full batch of records, lane t's in slot t,
 * and compute the next batch's records in registers while consuming the current one, so the records' long dependency
 * chains overlap the accumulation.
 */
template <typename Real, typename Pass, typename Store>
__device__ __forceinline__ void runPairwisePass(const PairwiseConformer<Real>& conformer,
                                                Real*                          records,
                                                Store&&                        store) {
  constexpr int kRecordsPerStep = Pass::kRecordsPerStep;
  const int     lane            = static_cast<int>(threadIdx.x) % kWarpSize;
  const int     numBatches      = (conformer.numPairs + kWarpSize - 1) / kWarpSize;
  if (numBatches == 0) {
    for (int index = lane; index < Pass::kWidth; index += kWarpSize) {
      store(index, Real(0));
    }
    return;
  }

  typename Pass::Accumulators acc;
  {
    Real* values = reinterpret_cast<Real*>(&acc);
#pragma unroll
    for (int value = 0; value < static_cast<int>(sizeof(acc) / sizeof(Real)); ++value) {
      values[value] = Real(0);
    }
  }
  Pass::prepare(records, lane);
  __syncwarp();
  PairCursor cursor;
  cursor.k = 1;
  cursor.advance(lane, conformer.numAtoms);
  if constexpr (Pass::kSorted) {
    // Batches sorted by tile: each step takes kRecordsPerStep records of one tile into that tile's sums.
    Real        record[Pass::kRecordValues];
    TileBallots tileBallots;
    const bool  contributes = computePair<Pass>(conformer, cursor, record);
    storeSorted<Pass>(records, record, contributes ? Pass::tileOf(record) : kRdfNoTile, lane, tileBallots);
    __syncwarp();
    for (int batch = 0; batch < numBatches; ++batch) {
      PairCursor nextCursor = cursor;
      nextCursor.advance(kWarpSize, conformer.numAtoms);
      Real       next[Pass::kRecordValues];
      const bool nextContributes = computePair<Pass>(conformer, nextCursor, next);
      int        first           = 0;
#pragma unroll
      for (int tile = 0; tile < kRdfTiles; ++tile) {
        const int end = first + __popc(tileBallots.lanesOf(tile));
        for (int slot = first; slot < end; slot += kRecordsPerStep) {
          Pass::consumeTile(records, slot, end, lane, acc[tile]);
        }
        first = end;
      }
      __syncwarp();
      storeSorted<Pass>(records, next, nextContributes ? Pass::tileOf(next) : kRdfNoTile, lane, tileBallots);
      __syncwarp();
      cursor = nextCursor;
    }
  } else if constexpr (Pass::kPipelined) {
    Real* const own = records + lane * Pass::kRecordStride;
    Real        record[Pass::kRecordValues];
    bool        contributes = computePair<Pass>(conformer, cursor, record);
    Pass::storeSlot(own, record, contributes);
    __syncwarp();
    for (int batch = 0; batch < numBatches; ++batch) {
      PairCursor nextCursor = cursor;
      nextCursor.advance(kWarpSize, conformer.numAtoms);
      Real       next[Pass::kRecordValues];
      const bool nextContributes = computePair<Pass>(conformer, nextCursor, next);
#pragma unroll
      for (int step = 0; step < kWarpSize / kRecordsPerStep; ++step) {
        Pass::consume(records + step * kRecordsPerStep * Pass::kRecordStride, lane, acc);
      }
      __syncwarp();
      if (contributes) {
        Pass::release(own, record);
      }
      Pass::storeSlot(own, next, nextContributes);
#pragma unroll
      for (int value = 0; value < Pass::kRecordValues; ++value) {
        record[value] = next[value];
      }
      contributes = nextContributes;
      __syncwarp();
      cursor = nextCursor;
    }
  } else {
    for (int batch = 0; batch < numBatches; ++batch, cursor.advance(kWarpSize, conformer.numAtoms)) {
      bool contributes = false;
      Real record[Pass::kRecordValues];
      if (cursor.pair < conformer.numPairs) {
        contributes = Pass::compute(conformer, cursor.j, cursor.k, record);
      }
      const unsigned contributing = __ballot_sync(0xffffffffu, contributes);
      const int      count        = __popc(contributing);
      const int      steps        = (count + kRecordsPerStep - 1) / kRecordsPerStep;
      Real* const    own          = records + __popc(contributing & ((1u << lane) - 1u)) * Pass::kRecordStride;
      if (contributes) {
        Pass::store(own, record);
      }
      if (lane < steps * kRecordsPerStep - count) {
        storeZeros<Pass::kNullValues>(records + (count + lane) * Pass::kRecordStride);
      }
      __syncwarp();
#pragma unroll 2
      for (int step = 0; step < steps; ++step) {
        Pass::consume(records + step * kRecordsPerStep * Pass::kRecordStride, lane, acc);
      }
      __syncwarp();
      if (contributes) {
        Pass::release(own, record);
      }
    }
  }

  if constexpr (std::is_same_v<Pass, AutocorrPass<Real>> && Pass::kMma) {
    writeMmaLags(acc, lane, store);
  } else if constexpr (std::is_same_v<Pass, AutocorrPass<Real>>) {
    writeWarpLags(acc, lane, store);
  } else if constexpr (Pass::kSorted) {
    writeRdfTiles(acc, lane, store);
  } else if constexpr (Pass::kMma) {
    writeMmaBins<Pass>(acc, lane, store);
  } else {
    writeWarpBins<Pass>(acc, lane, store);
  }
}

/**
 * @brief Warp-collective. AUTOCORR3D for a molecule without bond-distance table entries: each group takes rows j of
 *        the atom-pair triangle, finds the row's bond distances with searchBondDepths() into its own @p depth array
 *        (kGroupsPerWarp per conformer, laid out by atom), then accumulates pairs (j, k > j) into the lane owning the
 *        pair's lag.
 */
template <typename Real, typename Store>
__device__ __forceinline__ void runAutocorrSearch(const PairwiseConformer<Real>& conformer,
                                                  const int32_t*                 neighborStarts,
                                                  const int32_t*                 neighbors,
                                                  uint8_t*                       depth,
                                                  Store&&                        store) {
  const int lane                                                     = static_cast<int>(threadIdx.x) % kWarpSize;
  const int group                                                    = lane / kGroupSize;
  const int laneInGroup                                              = lane % kGroupSize;
  const int numAtoms                                                 = conformer.numAtoms;
  uint8_t*  groupDepth                                               = depth + static_cast<size_t>(group) * numAtoms;
  Real      acc[AutocorrPass<Real>::kLagSlots][kNumAutocorrChannels] = {};
  for (int j = group; j < numAtoms - 1; j += kGroupsPerWarp) {
    searchBondDepths<kNumAutocorrLags>(neighborStarts,
                                       neighbors,
                                       numAtoms,
                                       j,
                                       laneInGroup,
                                       0xffu << (group * kGroupSize),
                                       groupDepth);
    Real first[kRowValues];
    loadValues<kRowValues>(conformer.rows + j * kRowValues, first);
    for (int k = j + 1; k < numAtoms; ++k) {
      const int lag = groupDepth[k];
      if (lag >= 1 && lag <= kNumAutocorrLags) {
        Real second[kRowValues];
        loadValues<kRowValues>(conformer.rows + k * kRowValues, second);
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
  writeWarpLags(acc, lane, store);
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
 * @brief Warp-collective. Writes the conformer's atom rows (coordinates centered on the unweighted centroid in
 *        float64, see centeredPosition(), then the channel weights) to @p rows, MORSE bin 0 per channel (sum over pairs
 *        j < k of w_j * w_k, as ((sum w)^2 - sum w^2) / 2 in float64) to @p zeroBin, and returns AUTOCORR3D's NaN
 *        channels as bits (all of them for a non-finite coordinate).
 *
 * MORSE bin 0 needs float64: every term is a positive weight product, so the sum grows to ~1e5 for drug-sized
 * molecules and float32 accumulation over the pairs drifted by up to 19 (measured on ChEMBL conformers).
 */
template <typename Real, unsigned kSet>
__device__ __forceinline__ unsigned stageConformer(const ConformerAtoms& atoms,
                                                   const double*         propertyWeights,
                                                   const int             totalAtoms,
                                                   const double*         iStateDrag,
                                                   const double*         radius,
                                                   Real*                 rows,
                                                   double*               zeroBin) {
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
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    flags |= __shfl_xor_sync(0xffffffffu, flags, offset);
  }
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
    storeValues<kRowValues>(rows + atomIdx * kRowValues, row);
  }
  // Makes the rows and bin 0 visible to every lane.
  __syncwarp();
  // A non-finite coordinate makes every channel NaN.
  return (flags & 1u) != 0 ? (1u << kNumAutocorrChannels) - 1u : flags;
}

/**
 * @brief The pairwise properties in @p kSet (PairwiseSet bits) for every conformer; one warp per conformer.
 *
 * RDF bin i adds exp(-100 (R_i - r)^2) with R_i = 1 + 0.5 i; MORSE bin i adds sin(i r) / (i r), or 1 for i = 0. Each
 * term is scaled by the pair product of the channel's atom weights. RDF's I-state channel uses
 * `inputs.iStateDragWeights`; MORSE's uses the I-state block of `inputs.atomPropertyWeights`, matching RDKit.
 *
 * AUTOCORR3D lag l (1-10) adds r * w_j * w_k over pairs whose bond-count distance is l, for the MORSE channels plus
 * covalent radius, with distances from the shared bond-distance table (or per-row searches into @p bondDepths,
 * kGroupsPerWarp arrays per conformer laid out by atom, for molecules without entries). RDKit evaluates each sum as
 * w^T (B_l .* D) w, so a NaN weight anywhere in a channel, or a non-finite coordinate anywhere in the conformer, makes
 * every affected lag NaN, which RDKit then reports as 0; the kernel reproduces that per channel. Sums count both
 * orders of each pair and are divided by n (n - 1).
 *
 * The warp first writes one @p scratch row per atom (coordinates centered in float64, then the channel weights), so
 * the pair passes read only @p Real values; each property then takes one pass over the conformer's pairs
 * (runPairwisePass()) through the warp's shared-memory records.
 */
template <typename Real, unsigned kSet>
__global__ void pairwise3DKernel(const DeviceCoordView        coordinates,
                                 const Property3DDeviceInputs inputs,
                                 const BondDistanceTable      bondTable,
                                 Real* __restrict__ scratch,
                                 uint8_t* __restrict__ bondDepths,
                                 Real* __restrict__ rdfOutput,
                                 Real* __restrict__ morseOutput,
                                 Real* __restrict__ autocorrOutput) {
  constexpr bool kRdf      = (kSet & kPairwiseRdf) != 0;
  constexpr bool kMorse    = (kSet & kPairwiseMorse) != 0;
  constexpr bool kAutocorr = (kSet & kPairwiseAutocorr3D) != 0;
  constexpr int  kRegion   = recordRegion<Real, kSet>();
  __shared__ __align__(16) Real allRecords[kWarpsPerBlock][kRegion];
  __shared__ double             zeroBins[kWarpsPerBlock][kNumPairwiseChannels];
  const int                     warp         = static_cast<int>(threadIdx.x) / kWarpSize;
  const int                     lane         = static_cast<int>(threadIdx.x) % kWarpSize;
  const int                     conformerIdx = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
  if (conformerIdx >= coordinates.numConformers) {
    return;  // Uniform across the warp.
  }
  const ConformerAtoms atoms  = loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, conformerIdx);
  Real* const          rdfRow = kRdf ? rdfOutput + static_cast<size_t>(conformerIdx) * kNumRdfProperties : nullptr;
  Real* const morseRow = kMorse ? morseOutput + static_cast<size_t>(conformerIdx) * kNumMorseProperties : nullptr;
  Real* const autocorrRow =
    kAutocorr ? autocorrOutput + static_cast<size_t>(conformerIdx) * kNumAutocorr3DProperties : nullptr;
  if (!atoms.valid) {
    const Real invalid = static_cast<Real>(nan(""));
    for (int index = lane; index < PairwiseWidths<kSet>::kTotal; index += kWarpSize) {
      if (index < PairwiseWidths<kSet>::kRdf) {
        rdfRow[index] = invalid;
      } else if (index < PairwiseWidths<kSet>::kRdf + PairwiseWidths<kSet>::kMorse) {
        morseRow[index - PairwiseWidths<kSet>::kRdf] = invalid;
      } else {
        autocorrRow[index - PairwiseWidths<kSet>::kRdf - PairwiseWidths<kSet>::kMorse] = invalid;
      }
    }
    return;
  }

  const int      numAtoms      = atoms.numAtoms;
  const int64_t  atomOffset    = (atoms.positions - coordinates.positions) / 3;
  const int      moleculeIdx   = coordinates.molIndices[conformerIdx];
  const int      moleculeStart = inputs.moleculeAtomStarts[moleculeIdx];
  const int      totalAtoms    = inputs.moleculeAtomStarts[coordinates.nMols];
  Real* const    rows          = scratch + atomOffset * kRowValues;
  double* const  zeroBin       = zeroBins[warp];
  const unsigned autocorrNaN =
    stageConformer<Real, kSet>(atoms,
                               inputs.atomPropertyWeights + moleculeStart,
                               totalAtoms,
                               kRdf ? inputs.iStateDragWeights + moleculeStart : nullptr,
                               kAutocorr ? inputs.covalentRadiusWeights + moleculeStart : nullptr,
                               rows,
                               zeroBin);
  PairwiseConformer<Real> conformer;
  conformer.rows      = rows;
  conformer.numAtoms  = numAtoms;
  conformer.numPairs  = numAtoms * (numAtoms - 1) / 2;
  conformer.lagTable  = kAutocorr ? moleculeBondDistances(bondTable, moleculeIdx) : nullptr;
  Real* const records = allRecords[warp];

  if constexpr (kRdf) {
    runPairwisePass<Real, RdfPass<Real>>(conformer, records, [&](const int index, const Real sum) {
      rdfRow[index] = static_cast<Real>(roundThousandths(sum));
    });
  }
  if constexpr (kMorse) {
    runPairwisePass<Real, MorsePass<Real>>(conformer, records, [&](const int index, const Real sum) {
      const int scattering = index % kNumMorseScatterings;
      morseRow[index]      = scattering == 0 ?
                               static_cast<Real>(roundThousandths(zeroBin[index / kNumMorseScatterings])) :
                               static_cast<Real>(roundThousandths(sum * (Real(1) / static_cast<Real>(scattering))));
    });
  }
  if constexpr (kAutocorr) {
    const double orderedPairs = static_cast<double>(numAtoms) * (numAtoms - 1);
    // One atom gives 0 / 0 = NaN, as in RDKit; a channel with a NaN weight reports 0, as RDKit does.
    const auto   store        = [&](const int index, const Real sum) {
      const double total =
        ((autocorrNaN >> (index / kNumAutocorrLags)) & 1u) != 0 ? 0.0 : 2.0 * static_cast<double>(sum);
      autocorrRow[index] = static_cast<Real>(roundThousandths(total / orderedPairs));
    };
    if (conformer.lagTable != nullptr) {
      runPairwisePass<Real, AutocorrPass<Real>>(conformer, records, store);
    } else {
      runAutocorrSearch(conformer,
                        inputs.bondNeighborStarts + moleculeStart,
                        inputs.bondNeighbors,
                        bondDepths + atomOffset * kGroupsPerWarp,
                        store);
    }
  }
}

//! Output rows of the pairwise properties; null for properties that were not requested.
template <typename Real> struct PairwiseOutputs {
  Real* rdf        = nullptr;
  Real* morse      = nullptr;
  Real* autocorr3D = nullptr;
};

template <typename Real, unsigned kSet>
void launchPairwiseSet(const DeviceCoordView&        coordinates,
                       const Property3DDeviceInputs& inputs,
                       Real*                         scratch,
                       uint8_t*                      bondDepths,
                       const BondDistanceTable&      bondTable,
                       const PairwiseOutputs<Real>&  outputs,
                       const cudaStream_t            stream) {
  const int numBlocks = (coordinates.numConformers + kWarpsPerBlock - 1) / kWarpsPerBlock;
  pairwise3DKernel<Real, kSet><<<numBlocks, kBlockSize, 0, stream>>>(coordinates,
                                                                     inputs,
                                                                     bondTable,
                                                                     scratch,
                                                                     bondDepths,
                                                                     outputs.rdf,
                                                                     outputs.morse,
                                                                     outputs.autocorr3D);
  cudaCheckError(cudaGetLastError());
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
  AsyncDeviceVector<Real>    scratch(static_cast<size_t>(coordinates.numAtoms) * kRowValues, stream);
  // One bond-depth array per group per conformer, only when AUTOCORR3D is requested.
  AsyncDeviceVector<uint8_t> bondDepthScratch(
    (set & kPairwiseAutocorr3D) != 0 ? static_cast<size_t>(coordinates.numAtoms) * kGroupsPerWarp : 0,
    stream);
  Real* const    rows   = scratch.data();
  uint8_t* const depths = bondDepthScratch.data();
  switch (set) {
    case kPairwiseRdf:
      launchPairwiseSet<Real, kPairwiseRdf>(coordinates, inputs, rows, depths, bondTable, outputs, stream);
      break;
    case kPairwiseMorse:
      launchPairwiseSet<Real, kPairwiseMorse>(coordinates, inputs, rows, depths, bondTable, outputs, stream);
      break;
    case kPairwiseRdf | kPairwiseMorse:
      launchPairwiseSet<Real, kPairwiseRdf | kPairwiseMorse>(coordinates,
                                                             inputs,
                                                             rows,
                                                             depths,
                                                             bondTable,
                                                             outputs,
                                                             stream);
      break;
    case kPairwiseAutocorr3D:
      launchPairwiseSet<Real, kPairwiseAutocorr3D>(coordinates, inputs, rows, depths, bondTable, outputs, stream);
      break;
    case kPairwiseRdf | kPairwiseAutocorr3D:
      launchPairwiseSet<Real, kPairwiseRdf | kPairwiseAutocorr3D>(coordinates,
                                                                  inputs,
                                                                  rows,
                                                                  depths,
                                                                  bondTable,
                                                                  outputs,
                                                                  stream);
      break;
    case kPairwiseMorse | kPairwiseAutocorr3D:
      launchPairwiseSet<Real, kPairwiseMorse | kPairwiseAutocorr3D>(coordinates,
                                                                    inputs,
                                                                    rows,
                                                                    depths,
                                                                    bondTable,
                                                                    outputs,
                                                                    stream);
      break;
    default:
      launchPairwiseSet<Real, kPairwiseRdf | kPairwiseMorse | kPairwiseAutocorr3D>(coordinates,
                                                                                   inputs,
                                                                                   rows,
                                                                                   depths,
                                                                                   bondTable,
                                                                                   outputs,
                                                                                   stream);
      break;
  }
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_PAIRWISE_CUH
