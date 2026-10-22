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

/**
 * @brief MORSE bins 1-31: bin s adds w_j * w_k * sin(s r) / (s r) per channel. Records hold the weight products over
 *        r; the 1 / s factor is applied to the finished sums. Bin 0 is computed from the weights in float64, see
 *        stageConformer().
 *
 * Records carry r; the kLanes lanes of a record own scattering values l, l + kLanes, ... and each takes sin(s r) for
 * its own.
 */
template <typename Real> struct MorsePass {
  static constexpr bool kFloat          = std::is_same_v<Real, float>;
  static constexpr int  kLanes          = kGroupSize;
  static constexpr int  kSlots          = kNumMorseScatterings / kLanes;
  static constexpr int  kRecordsPerStep = kWarpSize / kLanes;
  static constexpr int  kRowLoad        = kFloat ? 12 : 10;
  //! Q0..Q6, r.
  static constexpr int  kRecordValues   = 8;
  //! Keeps the records a warp reads at once on distinct banks.
  static constexpr int  kRecordStride   = 12;
  //! Values a null record zeroes: all a consumer reads.
  static constexpr int  kNullValues     = kRecordStride;
  //! Every pair contributes, so batches keep a record per lane and overlap the next batch's records with the sums.
  static constexpr bool kPipelined      = true;
  static constexpr int  kWidth          = kNumMorseProperties;
  static constexpr int  kBins           = kNumMorseScatterings;
  using Accumulators                    = Real[kSlots][kNumPairwiseChannels];

  //! Scattering value of slot @p slot of lane @p laneInGroup.
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
    record[kNumPairwiseChannels] = r;
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
  }
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
 * bin's term is zero in float and below exp(-100) in double. Records spread the window over a zeroed term per bin
 * and lane l of a record's kLanes lanes owns the kSlots consecutive bins from kSlots l.
 */
template <typename Real> struct RdfPass {
  static constexpr bool kFloat          = std::is_same_v<Real, float>;
  static constexpr int  kLanes          = kGroupSize;
  static constexpr int  kSlots          = kNumMorseScatterings / kLanes;
  static constexpr int  kRecordsPerStep = kWarpSize / kLanes;
  static constexpr int  kWindow         = 4;
  static constexpr int  kRowLoad        = kFloat ? 12 : 10;
  //! Products of channels 0-6 (channel 0's is 1, the last is RDF's I-state), window start (int), then the terms.
  static constexpr int  kStartOffset    = kNumPairwiseChannels;
  static constexpr int  kTermOffset     = 8;
  static constexpr int  kTerms          = kNumMorseScatterings;
  static constexpr int  kRecordValues   = kTermOffset + kWindow;
  //! Keeps the records a warp reads at once on distinct banks.
  static constexpr int  kRecordStride   = kTermOffset + kTerms + 4;
  //! Values a null record zeroes; terms stay zeroed between records.
  static constexpr int  kNullValues     = kTermOffset;
  //! Every pair contributes, so batches keep a record per lane and overlap the next batch's records with the sums.
  static constexpr bool kPipelined      = true;
  static constexpr int  kWidth          = kNumRdfProperties;
  static constexpr int  kBins           = kNumRdfRadii;
  using Accumulators                    = Real[kSlots][kNumPairwiseChannels];
  static_assert(kRowIStateDrag < kRowLoad);

  //! Bin of slot @p slot of lane @p laneInGroup.
  __device__ __forceinline__ static int binOf(const int laneInGroup, const int slot) {
    return laneInGroup * kSlots + slot;
  }

  //! Window start of a record.
  __device__ __forceinline__ static int startOf(const Real (&record)[kRecordValues]) {
    return realAsInt(record[kStartOffset]);
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

  //! Records keep their terms zeroed between uses (see prepare() and release()), so a record writes only the window's
  //! four terms at their bins.
  __device__ __forceinline__ static void store(Real* target, const Real (&record)[kRecordValues]) {
    const int start = startOf(record);
    storeValues<kTermOffset>(target, record);
#pragma unroll
    for (int bin = 0; bin < kWindow; ++bin) {
      if (static_cast<unsigned>(start + bin) < static_cast<unsigned>(kTerms)) {
        target[kTermOffset + start + bin] = record[kTermOffset + bin];
      }
    }
  }

  //! Zeros the records' terms before the pass's first record.
  __device__ __forceinline__ static void prepare(Real* records, const int lane) {
    for (int value = lane * kVector<Real>; value < kWarpSize * kRecordStride; value += kWarpSize * kVector<Real>) {
      storeZeros<kVector<Real>>(records + value);
    }
  }

  //! Clears the terms store() wrote once every lane has consumed the record.
  __device__ __forceinline__ static void release(Real* target, const Real (&record)[kRecordValues]) {
    const int start = startOf(record);
#pragma unroll
    for (int bin = 0; bin < kWindow; ++bin) {
      if (static_cast<unsigned>(start + bin) < static_cast<unsigned>(kTerms)) {
        target[kTermOffset + start + bin] = Real(0);
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

  //! One step: the kRecordsPerStep records from @p records.
  __device__ __forceinline__ static void consume(const Real* records, const int lane, Accumulators& acc) {
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
};

/**
 * @brief AUTOCORR3D lags: lag l (1-10) adds w_j * w_k * r over pairs whose bond-count distance is l, for the MORSE
 *        channels plus covalent radius.
 *
 * Batches are compacted to the pairs within kNumAutocorrLags bonds. Lane l of a record's kLanes lanes owns lags l + 1
 * and l + 1 + kLanes for every channel, so each record updates one lane's sums.
 */
template <typename Real> struct AutocorrPass {
  static constexpr bool kFloat          = std::is_same_v<Real, float>;
  static constexpr int  kLanes          = kGroupSize;
  static constexpr int  kRecordsPerStep = kWarpSize / kLanes;
  static constexpr int  kRowLoad        = 12;
  //! Products of channels 0-7, r, lag (int), padding.
  static constexpr int  kRecordValues   = kFloat ? 12 : 10;
  //! Keeps the records a warp reads at once on distinct banks.
  static constexpr int  kRecordStride   = 12;
  static constexpr int  kNullValues     = kRecordStride;
  static constexpr bool kPipelined      = false;
  static constexpr int  kWidth          = kNumAutocorr3DProperties;
  static constexpr int  kLagSlots       = (kNumAutocorrLags + kLanes - 1) / kLanes;
  using Accumulators                    = Real[kLagSlots][kNumAutocorrChannels];
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
    values = maxValues(values, kWarpSize * RdfPass<Real>::kRecordStride);
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
  if constexpr (Pass::kPipelined) {
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

  if constexpr (std::is_same_v<Pass, AutocorrPass<Real>>) {
    writeWarpLags(acc, lane, store);
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
