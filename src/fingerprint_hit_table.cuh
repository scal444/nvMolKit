// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_FINGERPRINT_HIT_TABLE_CUH
#define NVMOLKIT_FINGERPRINT_HIT_TABLE_CUH

#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>

#include "src/fingerprint_similarity_device.cuh"

// Sparse threshold tests between a tile of up to 32 source fingerprints and many candidates.
//
// A hit table in shared memory maps every bit position to the mask of tile sources that set it. A candidate looks up
// the masks of its set bits, listed once per fingerprint, and adds them into bit-sliced counters (bit k of slice s is
// bit s of the intersection with source k), so each 32-bit operation advances the intersections with all 32 sources.
// Real fingerprints are sparse, so this costs a few operations per set bit instead of a population count per word and
// source. Only sources whose intersection reaches a conservative lower bound for the threshold are then tested
// exactly, with the same arithmetic as the dense path, so results are unchanged.

namespace nvMolKit::detail {

//! Set-bit positions listed per fingerprint, in groups of eight padded with the sentinel position numWords * 32, whose
//! table entry is zero. Fingerprints with more set bits are staged and tested by word.
constexpr int         kPositionCapacity = 128;
constexpr int         kPositionGroups   = kPositionCapacity / 8;
//! Bits per sliced counter; larger intersections saturate into the overflow mask and are counted exactly.
constexpr int         kSlices           = 6;
//! Largest hit table kept in shared memory.
constexpr std::size_t kMaxHitTableBytes = 40 * 1024;

//! Shared-memory bytes of a hit table: one mask per bit position and the sentinel, then the 32 source bit counts, their
//! minimum, and the mask of valid sources.
__host__ __device__ inline std::size_t hitTableBytes(const int numWords) {
  return (static_cast<std::size_t>(numWords) * 32 + 1 + 32 + 2) * sizeof(std::uint32_t);
}

//! Whether fingerprints of @p numWords words can be listed by 16-bit position and tested on a shared hit table.
inline bool hitTableSupported(const int numWords) {
  return numWords * 32 < 0xFFFF && hitTableBytes(numWords) <= kMaxHitTableBytes;
}

/**
 * Lists the set-bit positions of every fingerprint with at most kPositionCapacity bits, one warp per fingerprint.
 * Lists are stored by group of eight, then fingerprint, so adjacent candidates later load adjacent uint4 values.
 */
static __global__ void fingerprintPositionsKernel(const std::uint32_t* fingerprints,
                                                  std::uint16_t*       positions,
                                                  const int            numItems,
                                                  const int            numWords) {
  const int lane  = static_cast<int>(threadIdx.x) & 31;
  const int warps = static_cast<int>(gridDim.x * blockDim.x) >> 5;
  for (int item = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) >> 5; item < numItems; item += warps) {
    const std::uint32_t* row   = fingerprints + static_cast<std::size_t>(item) * numWords;
    int                  count = 0;
    for (int base = 0; base < numWords; base += 32) {
      const int     word  = base + lane;
      std::uint32_t bits  = word < numWords ? row[word] : 0U;
      const int     added = __popc(bits);
      int           scan  = added;
      for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffffU, scan, offset);
        scan += lane >= offset ? other : 0;
      }
      for (int slot = count + scan - added; bits != 0U && slot < kPositionCapacity; bits &= bits - 1U, ++slot) {
        positions[(static_cast<std::size_t>(slot / 8) * numItems + item) * 8 + slot % 8] =
          static_cast<std::uint16_t>(word * 32 + __ffs(static_cast<int>(bits)) - 1);
      }
      count += __shfl_sync(0xffffffffU, scan, 31);
    }
    const int slot = count + lane;
    if (slot < kPositionCapacity && slot < ((count + 7) & ~7)) {
      positions[(static_cast<std::size_t>(slot / 8) * numItems + item) * 8 + slot % 8] =
        static_cast<std::uint16_t>(numWords * 32);
    }
  }
}

//! Intersections of one candidate with up to 32 sources, bit-sliced.
struct SlicedCounts {
  std::uint32_t slice[kSlices] = {};
  std::uint32_t overflow       = 0U;
};

//! Carry-save addition of two masks into @p sum; @p carry receives the masks of the next weight.
__device__ __forceinline__ void carrySave(std::uint32_t&      carry,
                                          std::uint32_t&      sum,
                                          const std::uint32_t a,
                                          const std::uint32_t b) {
  carry = (sum & a) | (sum & b) | (a & b);
  sum   = sum ^ a ^ b;
}

//! Adds eight source masks to the counters.
__device__ __forceinline__ void addMasks(SlicedCounts& counts, const std::uint32_t (&masks)[8]) {
  std::uint32_t twos[4];
  carrySave(twos[0], counts.slice[0], masks[0], masks[1]);
  carrySave(twos[1], counts.slice[0], masks[2], masks[3]);
  carrySave(twos[2], counts.slice[0], masks[4], masks[5]);
  carrySave(twos[3], counts.slice[0], masks[6], masks[7]);
  std::uint32_t fours[2];
  carrySave(fours[0], counts.slice[1], twos[0], twos[1]);
  carrySave(fours[1], counts.slice[1], twos[2], twos[3]);
  std::uint32_t carry = 0U;
  carrySave(carry, counts.slice[2], fours[0], fours[1]);
#pragma unroll
  for (int level = 3; level < kSlices; ++level) {
    const std::uint32_t next = counts.slice[level] & carry;
    counts.slice[level] ^= carry;
    carry = next;
  }
  counts.overflow |= carry;
}

//! Mask of sources whose count is at least @p minimum; overflowed sources reach any minimum.
__device__ __forceinline__ std::uint32_t countsAtLeast(const SlicedCounts& counts, const int minimum) {
  constexpr int kLargest = (1 << kSlices) - 1;
  const int     bound    = minimum < 0 ? 0 : (minimum > kLargest ? kLargest : minimum);
  std::uint32_t greater  = 0U;
  std::uint32_t equal    = ~0U;
#pragma unroll
  for (int level = kSlices - 1; level >= 0; --level) {
    const std::uint32_t boundBit = 0U - static_cast<std::uint32_t>((bound >> level) & 1);
    greater |= equal & counts.slice[level] & ~boundBit;
    equal &= ~(counts.slice[level] ^ boundBit);
  }
  return greater | equal | counts.overflow;
}

//! Count of one source that did not overflow.
__device__ __forceinline__ int countOf(const SlicedCounts& counts, const int source) {
  int count = 0;
#pragma unroll
  for (int level = 0; level < kSlices; ++level) {
    count |= static_cast<int>((counts.slice[level] >> source) & 1U) << level;
  }
  return count;
}

/**
 * Smallest intersection with which fingerprints of these bit counts can reach @p similarity, lowered by one so that
 * rounding in the exact test can never accept a pair this bound drops. @p sourceCount may be the smallest count of a
 * tile, since the bound grows with it.
 */
template <FingerprintSimilarityMetric Metric>
__device__ __forceinline__ int minimumIntersection(const int   candidateCount,
                                                   const int   sourceCount,
                                                   const float similarity) {
  if (!(similarity > 0.0F)) {
    return 0;
  }
  float bound;
  if constexpr (Metric == FingerprintSimilarityMetric::Tanimoto) {
    // I / (a + b - I) >= s  <=>  I >= s (a + b) / (1 + s)
    bound = similarity * (static_cast<float>(candidateCount) + static_cast<float>(sourceCount)) / (1.0F + similarity);
  } else {
    // I / sqrt(a b) >= s
    bound = similarity * sqrtf(static_cast<float>(candidateCount) * static_cast<float>(sourceCount));
  }
  return static_cast<int>(floorf(bound)) - 1;
}

/**
 * Block-cooperative: stages the hit table of up to 32 @p sources (in shared memory; invalid entries are -1) at
 * @p table, which holds hitTableBytes(numWords). Listed sources are staged from their positions, the others by word.
 * Ends synchronized.
 */
__device__ inline void stageHitTable(std::uint32_t*       table,
                                     const std::uint32_t* fingerprints,
                                     const int*           bitCounts,
                                     const std::uint16_t* positions,
                                     const int            numItems,
                                     const int            numWords,
                                     const int*           sources,
                                     const int            numSources) {
  const int entries = numWords * 32 + 1;
  int*      counts  = reinterpret_cast<int*>(table + entries);
  for (int index = static_cast<int>(threadIdx.x); index < entries; index += static_cast<int>(blockDim.x)) {
    table[index] = 0U;
  }
  if (threadIdx.x < 32) {
    const int  lane    = static_cast<int>(threadIdx.x);
    const bool valid   = lane < numSources && sources[lane] >= 0;
    const int  count   = valid ? bitCounts[sources[lane]] : 0;
    int        minimum = valid ? count : INT_MAX;
    for (int offset = 16; offset > 0; offset >>= 1) {
      minimum = min(minimum, __shfl_xor_sync(0xffffffffU, minimum, offset));
    }
    const std::uint32_t present = __ballot_sync(0xffffffffU, valid);
    counts[lane]                = count;
    if (lane == 0) {
      counts[32] = minimum;
      counts[33] = static_cast<int>(present);
    }
  }
  __syncthreads();
  const auto* valid = reinterpret_cast<const std::uint32_t*>(counts + 33);
  // One thread per group of eight listed positions.
  for (int index = static_cast<int>(threadIdx.x); index < 32 * kPositionGroups; index += static_cast<int>(blockDim.x)) {
    const int ordinal = index / kPositionGroups;
    const int group   = index % kPositionGroups;
    if ((*valid >> ordinal & 1U) == 0U || counts[ordinal] > kPositionCapacity || group * 8 >= counts[ordinal]) {
      continue;
    }
    const uint4 packed =
      reinterpret_cast<const uint4*>(positions)[static_cast<std::size_t>(group) * numItems + sources[ordinal]];
    const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int half = 0; half < 8; ++half) {
      // Padding holds the sentinel position, whose entry stays clear.
      const int position = static_cast<int>((words[half / 2] >> (16 * (half % 2))) & 0xFFFFU);
      if (position < entries - 1) {
        atomicOr(table + position, 1U << ordinal);
      }
    }
  }
  // Sources with more set bits than a list holds are staged by word, eight threads per source.
  for (int index = static_cast<int>(threadIdx.x); index < 32 * 8; index += static_cast<int>(blockDim.x)) {
    const int ordinal = index / 8;
    if ((*valid >> ordinal & 1U) == 0U || counts[ordinal] <= kPositionCapacity) {
      continue;
    }
    const std::uint32_t* row = fingerprints + static_cast<std::size_t>(sources[ordinal]) * numWords;
    for (int word = index % 8; word < numWords; word += 8) {
      for (std::uint32_t bits = row[word]; bits != 0U; bits &= bits - 1U) {
        atomicOr(table + word * 32 + __ffs(static_cast<int>(bits)) - 1, 1U << ordinal);
      }
    }
  }
  __syncthreads();
}

/**
 * Bit-sliced intersections of @p candidate with the staged sources. A candidate whose positions are not listed gets
 * every valid source marked as overflowed, so its intersections are counted by word.
 */
__device__ inline SlicedCounts hitTableCounts(const std::uint32_t* table,
                                              const std::uint16_t* positions,
                                              const int            numItems,
                                              const int            numWords,
                                              const int            candidate,
                                              const int            candidateCount) {
  SlicedCounts sliced;
  if (candidateCount > kPositionCapacity) {
    sliced.overflow = static_cast<std::uint32_t>(reinterpret_cast<const int*>(table + numWords * 32 + 1)[33]);
    return sliced;
  }
  const auto* list   = reinterpret_cast<const uint4*>(positions) + candidate;
  const int   groups = (candidateCount + 7) / 8;
  for (int group = 0; group < groups; ++group) {
    const uint4         packed   = list[static_cast<std::size_t>(group) * numItems];
    const std::uint32_t masks[8] = {table[packed.x & 0xFFFFU],
                                    table[packed.x >> 16],
                                    table[packed.y & 0xFFFFU],
                                    table[packed.y >> 16],
                                    table[packed.z & 0xFFFFU],
                                    table[packed.z >> 16],
                                    table[packed.w & 0xFFFFU],
                                    table[packed.w >> 16]};
    addMasks(sliced, masks);
  }
  return sliced;
}

//! Exact intersection of the candidate row with staged source @p ordinal, counted by word if it overflowed.
__device__ __forceinline__ int hitTableIntersection(const SlicedCounts&  sliced,
                                                    const int            ordinal,
                                                    const std::uint32_t* source,
                                                    const std::uint32_t* row,
                                                    const int            numWords) {
  if ((sliced.overflow >> ordinal & 1U) == 0U) {
    return countOf(sliced, ordinal);
  }
  int common = 0;
  for (int word = 0; word < numWords; ++word) {
    common += __popc(source[word] & row[word]);
  }
  return common;
}

/**
 * Mask of the staged sources within @p similarity of @p candidate under @p Metric, with the same test as
 * fingerprintSimilarityCanReach and fingerprintSimilarityAtLeast. Only sources whose sliced count reaches
 * minimumIntersection are tested.
 */
template <FingerprintSimilarityMetric Metric>
__device__ inline std::uint32_t hitTableMask(const std::uint32_t* table,
                                             const std::uint32_t* fingerprints,
                                             const std::uint16_t* positions,
                                             const int            numItems,
                                             const int            numWords,
                                             const int*           sources,
                                             const int            candidate,
                                             const int            candidateCount,
                                             const float          similarity) {
  const int*         counts = reinterpret_cast<const int*>(table + numWords * 32 + 1);
  const SlicedCounts sliced = hitTableCounts(table, positions, numItems, numWords, candidate, candidateCount);
  std::uint32_t flagged = countsAtLeast(sliced, minimumIntersection<Metric>(candidateCount, counts[32], similarity)) &
                          static_cast<std::uint32_t>(counts[33]);
  const std::uint32_t* row  = fingerprints + static_cast<std::size_t>(candidate) * numWords;
  std::uint32_t        hits = 0U;
  for (; flagged != 0U; flagged &= flagged - 1U) {
    const int ordinal = __ffs(static_cast<int>(flagged)) - 1;
    const int common  = hitTableIntersection(sliced,
                                            ordinal,
                                            fingerprints + static_cast<std::size_t>(sources[ordinal]) * numWords,
                                            row,
                                            numWords);
    if (fingerprintSimilarityCanReach<Metric>(counts[ordinal], candidateCount, similarity) &&
        fingerprintSimilarityAtLeast<Metric>(common, counts[ordinal], candidateCount, similarity)) {
      hits |= 1U << ordinal;
    }
  }
  return hits;
}

}  // namespace nvMolKit::detail

#endif  // NVMOLKIT_FINGERPRINT_HIT_TABLE_CUH
