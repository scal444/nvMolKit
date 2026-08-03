// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FMCS_CUDA_FMCS_SEARCH_SUPPORT_CUH
#define FMCS_CUDA_FMCS_SEARCH_SUPPORT_CUH

#include <cooperative_groups.h>

#include <cstdint>

#include "src/mcs/fmcs_cuda/fmcs_grow.cuh"
#include "src/mcs/fmcs_cuda/fmcs_match.cuh"
#include "src/mcs/fmcs_cuda/fmcs_match_tables.cuh"
#include "src/mcs/fmcs_cuda/fmcs_seed.cuh"
#include "src/mcs/fmcs_cuda/fmcs_seed_queue.cuh"
#include "src/mcs/fmcs_cuda/fmcs_topology.cuh"
#include "src/mcs/mcs_common/mcs_cooperative_copy.cuh"

namespace mcs {
namespace fmcs {

namespace cg = cooperative_groups;

/// Per-pair descriptor passed to the kernel.  Non-owning: pointers refer
/// into host-uploaded device buffers.  The caller chooses the smaller
/// input as the query (fMCS only enumerates subgraphs of the query) and
/// records the choice in @c swapped so host-side expansion can un-swap
/// the result mappings.
struct DevicePerPairInput {
  int queryNumAtoms  = 0;
  int queryNumBonds  = 0;
  int targetNumAtoms = 0;
  int targetNumBonds = 0;

  const std::uint32_t* queryRowOffsets    = nullptr;
  const std::uint32_t* queryColIndices    = nullptr;
  /// Parallel to @c queryColIndices: undirected bond id for each CSR entry.
  const std::uint32_t* queryBondIndices   = nullptr;
  /// Packed (u << 16 | v), one entry per undirected bond, ordered to
  /// match the bond dimension of the match tables (@ref enumerateBonds).
  const std::uint32_t* queryBondEndpoints = nullptr;

  const std::uint32_t* targetRowOffsets    = nullptr;
  const std::uint32_t* targetColIndices    = nullptr;
  const std::uint32_t* targetBondIndices   = nullptr;
  const std::uint32_t* targetBondEndpoints = nullptr;

  PairMatchTablesDevice tables;

  bool swapped = false;
};

/// Fixed-size device-writable result.  mcs::MCSResult uses std::vector and
/// cannot be constructed on the device, so the kernel fills this POD and
/// the host expands it.
///
/// @c bondMapA[i] / @c bondMapB[i] hold the query / target bond index of
/// the i-th matched edge in the common subgraph.  The host recovers
/// (u, v) endpoints by indexing the @c bondEndpoints arrays it already
/// uploaded for each pair -- avoids duplicating endpoint data here at
/// 4 * maxBonds bytes per result.
template <int maxAtoms, int maxBonds> struct DeviceMCSResult {
  int  numCommonVertices = 0;
  int  numCommonEdges    = 0;
  bool timedOut          = false;
  bool overflowed        = false;

  /// mappingA[i] = query atom idx, mappingB[i] = target atom idx of
  /// the i-th matched vertex.
  uint8_t mappingA[maxAtoms];
  uint8_t mappingB[maxAtoms];
  /// bondMapA[i] = query bond idx, bondMapB[i] = target bond idx of
  /// the i-th matched edge.
  uint8_t bondMapA[maxBonds];
  uint8_t bondMapB[maxBonds];
};

/// Within-thread: standard SplitMix64 finalizer.  Used both as the
/// per-step mix in @ref mappingHashWithinThread and as a final scramble
/// when slot-deriving in @ref DeviceMatchCache.
__device__ __forceinline__ std::uint64_t splitMix64(std::uint64_t x) {
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

/// Per-block open-addressed hash table of 64-bit keys.  Stores successful
/// (seed shape, embedding) keys only; failures are never cached so they
/// cannot poison alternative lineages.
///
/// The backing key array lives in global memory (one per-block slab
/// handed in via @ref init); only the cache header (storage pointer +
/// capacity) lives in the block's shared memory.  This keeps the
/// per-block shared footprint a few bytes instead of @c 8 * Capacity,
/// preserving SM occupancy -- 4096 keys would otherwise eat 32 KB of
/// shared memory and cap the SM at ~2 blocks.  Per-block global cost is
/// 8 * Capacity bytes; the L1/L2 hierarchy amortizes the slower probe
/// path on hot keys.
///
/// Empty-slot sentinel is the all-zero key.  @ref mappingHashWithinThread
/// is required to never return 0 (it post-processes a 0 hash to 1) so
/// any non-empty slot is unambiguously occupied.
///
/// @ref capacity must be a power of two; the slot index is computed via
/// a mask, not a modulo.  The power-of-two constraint is asserted at
/// the kernel-instantiation site.
///
/// Probe / insert are intentionally @c WithinThread (single-lane).
/// Warp-cooperative variants (@c probeCooperative / @c insertCooperative
/// that check 32 slots per round via @c group.ballot) would amortize
/// the linear-probe traversal across the whole warp, but are only a
/// win at high cache load -- expected probe length is roughly
/// @c 1 / (1 - load_factor), and at our drug-like fmcs workloads the
/// cache stays at 5-25% load (expected length ~1).  Revisit if
/// profiling shows the cache on the hot path or if we shrink
/// @c kFmcsCacheCapacity.
struct DeviceMatchCache {
  std::uint64_t* keys     = nullptr;
  int            capacity = 0;

  /// Lane-0 init: bind this header to a caller-allocated global slab.
  __device__ __forceinline__ void init(std::uint64_t* storage, int slabCapacity) {
    keys     = storage;
    capacity = slabCapacity;
  }

  /// Cooperative: zero the (global) table across @p group.
  template <class GroupT> __device__ __forceinline__ void zeroCooperative(const GroupT& group) {
    const int rank  = static_cast<int>(group.thread_rank());
    const int total = static_cast<int>(group.num_threads());
    for (int i = rank; i < capacity; i += total) {
      keys[i] = 0;
    }
  }

  /// Within-thread: single lane attempts to insert @p key via linear
  /// probing.  Idempotent on duplicate inserts.  Returns false only on
  /// a fully-occupied table (no slot found after probing every entry);
  /// callers may safely ignore the false return -- a missed insert
  /// just costs a re-run of the matching call on a future visit, which
  /// is by design (the cache is best-effort).
  ///
  /// 0 is reserved as the empty-slot sentinel and is rejected here as
  /// a defensive guard.  @ref mappingHashWithinThread post-processes
  /// any 0 hash to 1 so this branch shouldn't fire in normal use, but
  /// without it inserting the sentinel would be indistinguishable
  /// from no insert at all.
  ///
  /// Concurrency: the global atomicCAS makes concurrent inserts of
  /// distinct keys safe.  Concurrent inserts of the SAME key may both
  /// observe the slot transitioning from empty -> @p key; both return
  /// true (idempotent).
  __device__ __forceinline__ bool insertWithinThread(std::uint64_t key) {
    if (key == 0ULL)
      return false;
    const int mask = capacity - 1;
    int       slot = static_cast<int>(splitMix64(key)) & mask;
    for (int attempt = 0; attempt < capacity; ++attempt) {
      const std::uint64_t prev =
        atomicCAS(reinterpret_cast<unsigned long long*>(&keys[slot]), 0ULL, static_cast<unsigned long long>(key));
      if (prev == 0ULL)
        return true;  // claimed empty slot
      if (prev == key)
        return true;  // already present (idempotent)
      slot = (slot + 1) & mask;
    }
    return false;
  }

  /// Within-thread: single lane checks whether @p key is in the table.
  /// Linear-probes from @p key's hashed slot until either an empty
  /// slot (would-have-been-here -> miss) or a key match (hit).
  /// Empty-slot termination is checked first so a probe of the
  /// reserved 0 sentinel doesn't false-hit on a fresh cache.
  __device__ __forceinline__ bool probeWithinThread(std::uint64_t key) const {
    if (key == 0ULL)
      return false;
    const int mask = capacity - 1;
    int       slot = static_cast<int>(splitMix64(key)) & mask;
    for (int attempt = 0; attempt < capacity; ++attempt) {
      const std::uint64_t entry = keys[slot];
      if (entry == 0ULL)
        return false;
      if (entry == key)
        return true;
      slot = (slot + 1) & mask;
    }
    return false;
  }
};

/// Within-thread: SplitMix64 cumulative mix over canonical-ordered seed
/// state and recorded mapping pairs.  The key covers:
///   - seed atoms + atom mapping,
///   - last-added atom frontier,
///   - seed bonds + bond mapping,
///   - excluded query bonds.
/// We walk set bits in increasing order (via @c __ffs / @c __ffsll),
/// which canonicalizes the iteration regardless of the order in which
/// atoms or bonds were actually added to the seed.  Two queue entries
/// with the same grow state and recorded embedding therefore produce the
/// same hash, which is the property the cache relies on for cross-lineage
/// success deduplication.
///
/// Atom mappings are part of the key because a single undirected bond
/// mapping does not determine endpoint orientation.  Without the atom
/// contribution, the two Phase-1 orientations of the same (query bond,
/// target bond) alias in the cache and one can incorrectly prune the
/// other before it has a chance to grow.
///
/// Frontier/exclusion state is also part of the key because it determines
/// future grow candidates.  A mapped subgraph reached through a different
/// last-added frontier is not interchangeable for search purposes.
///
/// The hash NEVER returns 0; if the natural mix produces 0 (rare:
/// 1-in-2^64) we remap to 1 so the cache's empty-slot sentinel stays
/// unambiguous.  See @ref DeviceMatchCache.
template <int maxAtoms, int maxBonds, int maxTA, int maxTB>
__device__ __forceinline__ std::uint64_t mappingHashWithinThread(
  const Seed<maxAtoms, maxBonds>&                      seed,
  const MatchResult<maxAtoms, maxBonds, maxTA, maxTB>& match) {
  using SeedT                    = Seed<maxAtoms, maxBonds>;
  using AtomWord                 = typename SeedT::atom_word_type;
  using BondWord                 = typename SeedT::bond_word_type;
  constexpr int kAtomBitsPerWord = SeedT::kAtomBitsPerWord;
  constexpr int kAtomWords       = SeedT::kAtomWords;
  constexpr int kBondBitsPerWord = SeedT::kBondBitsPerWord;
  constexpr int kBondWords       = SeedT::kBondWords;

  std::uint64_t           hash             = 0;
  constexpr std::uint64_t kAtomDomain      = 0x9e3779b97f4a7c15ULL;
  constexpr std::uint64_t kLastAddedDomain = 0xd6e8feb86659fd93ULL;
  constexpr std::uint64_t kBondDomain      = 0xbf58476d1ce4e5b9ULL;
  constexpr std::uint64_t kExcludedDomain  = 0x94d049bb133111ebULL;

  for (int wordIdx = 0; wordIdx < kAtomWords; ++wordIdx) {
    AtomWord remaining = seed.atoms[wordIdx];
    while (remaining != 0) {
      int bitPosInWord;
      if constexpr (sizeof(AtomWord) == 4) {
        bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
      } else {
        bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
      }
      const int queryAtomIdx = wordIdx * kAtomBitsPerWord + bitPosInWord;
      remaining &= remaining - 1;

      const std::uint8_t targetAtomIdx = match.targetAtomIdx[queryAtomIdx];
      if (targetAtomIdx == kUnmappedTargetIdx)
        continue;

      const std::uint64_t pair =
        (static_cast<std::uint64_t>(queryAtomIdx + 1) << 8) | static_cast<std::uint64_t>(targetAtomIdx);
      hash = splitMix64(hash + kAtomDomain + pair);
    }
  }

  for (int wordIdx = 0; wordIdx < kAtomWords; ++wordIdx) {
    AtomWord remaining = seed.lastAddedAtoms[wordIdx];
    while (remaining != 0) {
      int bitPosInWord;
      if constexpr (sizeof(AtomWord) == 4) {
        bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
      } else {
        bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
      }
      const int queryAtomIdx = wordIdx * kAtomBitsPerWord + bitPosInWord;
      remaining &= remaining - 1;
      const std::uint64_t token = static_cast<std::uint64_t>(queryAtomIdx + 1);
      hash                      = splitMix64(hash + kLastAddedDomain + token);
    }
  }

  for (int wordIdx = 0; wordIdx < kBondWords; ++wordIdx) {
    BondWord remaining = seed.bonds[wordIdx];
    while (remaining != 0) {
      int bitPosInWord;
      if constexpr (sizeof(BondWord) == 4) {
        bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
      } else {
        bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
      }
      const int queryBondIdx = wordIdx * kBondBitsPerWord + bitPosInWord;
      remaining &= remaining - 1;

      const std::uint8_t targetBondIdx = match.targetBondIdx[queryBondIdx];
      if (targetBondIdx == kUnmappedTargetIdx)
        continue;

      // Pack (queryBondIdx, targetBondIdx) into a uint64 with no
      // overlap (target index is 8 bits; query index is 16-bit-safe
      // since maxBonds <= 128).  The query field is biased by +1 so the
      // packed token is never zero: the accumulator starts at 0 and
      // splitMix64(0) == 0, so a zero token (which a raw q=0,t=0 bond
      // would produce) would leave the accumulator at 0 and make that
      // bond invisible, aliasing a seed onto one that omits it.
      const std::uint64_t pair =
        (static_cast<std::uint64_t>(queryBondIdx + 1) << 8) | static_cast<std::uint64_t>(targetBondIdx);
      hash = splitMix64(hash + kBondDomain + pair);
    }
  }

  for (int wordIdx = 0; wordIdx < kBondWords; ++wordIdx) {
    BondWord remaining = seed.excludedBonds[wordIdx];
    while (remaining != 0) {
      int bitPosInWord;
      if constexpr (sizeof(BondWord) == 4) {
        bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
      } else {
        bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
      }
      const int queryBondIdx = wordIdx * kBondBitsPerWord + bitPosInWord;
      remaining &= remaining - 1;
      const std::uint64_t token = static_cast<std::uint64_t>(queryBondIdx + 1);
      hash                      = splitMix64(hash + kExcludedDomain + token);
    }
  }
  return hash == 0ULL ? 1ULL : hash;
}

template <class GroupT, class QueuedT>
__device__ __forceinline__ void updateIncumbentCooperative(const GroupT&  group,
                                                           const QueuedT& candidate,
                                                           QueuedT&       best,
                                                           unsigned int*  bestScore,
                                                           int*           bestCopyLock) {
  const int groupRank  = static_cast<int>(group.thread_rank());
  int       locked     = 0;
  int       shouldCopy = 0;
  if (groupRank == 0) {
    const unsigned int candidateScore =
      (static_cast<unsigned int>(candidate.seed.numBonds) << 16) | static_cast<unsigned int>(candidate.seed.numAtoms);
    unsigned int prev = *bestScore;
    bool         won  = false;
    while (candidateScore > prev) {
      const unsigned int seen = atomicCAS(bestScore, prev, candidateScore);
      if (seen == prev) {
        won = true;
        break;
      }
      prev = seen;
    }
    if (won) {
      while (atomicCAS(bestCopyLock, 0, 1) != 0) {
      }
      locked     = 1;
      shouldCopy = (candidateScore == *bestScore) ? 1 : 0;
    }
  }
  locked     = group.shfl(locked, 0);
  shouldCopy = group.shfl(shouldCopy, 0);
  if (shouldCopy) {
    warpCopy(group, &best, &candidate, sizeof(QueuedT));
  }
  group.sync();
  if (groupRank == 0 && locked) {
    __threadfence_block();
    atomicExch(bestCopyLock, 0);
  }
  group.sync();
}

template <int maxAtoms, int maxBonds, int maxTA, int maxTB, class GroupT>
__device__ __forceinline__ bool checkSeedMatchAndAppendCooperative(
  const GroupT&                                 group,
  QueuedSeed<maxAtoms, maxBonds, maxTA, maxTB>& candidate,
  const DeviceCsrView&                          queryTopology,
  const DeviceCsrView&                          targetTopology,
  const PairMatchTablesDevice&                  tables,
  FmcsSubstructureScratch<maxAtoms, maxTA>&     scratch) {
  bool ok = false;
  if (!candidate.match.empty) {
    ok = tryMatchIncrementalGreedyCooperative(group,
                                              candidate.seed,
                                              queryTopology,
                                              targetTopology,
                                              tables,
                                              candidate.match);
  }

  if (!ok) {
    ok = matchSeedSubstructureCooperative(group,
                                          candidate.seed,
                                          queryTopology,
                                          targetTopology,
                                          tables,
                                          candidate.match,
                                          scratch);
    group.sync();
  }
  return ok;
}

__device__ __forceinline__ bool isPowerOfTwo64(unsigned long long value) {
  return value != 0ULL && (value & (value - 1ULL)) == 0ULL;
}

template <class GroupT, class QueuedT>
__device__ __forceinline__ bool insertSortedByBondsCooperative(const GroupT&                         group,
                                                               SeedQueue<QueuedT, ThreadBlockScope>& queue,
                                                               const QueuedT&                        element) {
  const int groupRank = static_cast<int>(group.thread_rank());
  int       oldSize   = 0;
  int       insertAt  = 0;
  int       ok        = 1;
  if (groupRank == 0) {
    oldSize  = queue.size();
    ok       = oldSize < queue.capacity() ? 1 : 0;
    insertAt = oldSize;
    if (ok) {
      for (int i = 0; i < oldSize; ++i) {
        if (queue.slot(i).seed.numBonds < element.seed.numBonds) {
          insertAt = i;
          break;
        }
      }
    }
  }
  oldSize  = group.shfl(oldSize, 0);
  insertAt = group.shfl(insertAt, 0);
  ok       = group.shfl(ok, 0);
  if (!ok)
    return false;

  for (int i = oldSize; i > insertAt; --i) {
    warpCopy(group, &queue.slot(i), &queue.slot(i - 1), sizeof(QueuedT));
    group.sync();
  }
  warpCopy(group, &queue.slot(insertAt), &element, sizeof(QueuedT));
  group.sync();
  if (groupRank == 0)
    queue.setSizeWithinThread(oldSize + 1);
  group.sync();
  return true;
}

template <class GroupT, class QueuedT>
__device__ __forceinline__ bool popFrontCooperative(const GroupT&                         group,
                                                    SeedQueue<QueuedT, ThreadBlockScope>& queue,
                                                    QueuedT&                              outElement) {
  const int groupRank = static_cast<int>(group.thread_rank());
  int       oldSize   = 0;
  if (groupRank == 0)
    oldSize = queue.size();
  oldSize = group.shfl(oldSize, 0);
  if (oldSize <= 0)
    return false;

  warpCopy(group, &outElement, &queue.slot(0), sizeof(QueuedT));
  group.sync();
  for (int i = 1; i < oldSize; ++i) {
    warpCopy(group, &queue.slot(i - 1), &queue.slot(i), sizeof(QueuedT));
    group.sync();
  }
  if (groupRank == 0)
    queue.setSizeWithinThread(oldSize - 1);
  group.sync();
  return true;
}

template <class GroupT, class QueuedT>
__device__ __forceinline__ bool pushBackCooperative(const GroupT&                         group,
                                                    SeedQueue<QueuedT, ThreadBlockScope>& queue,
                                                    const QueuedT&                        element) {
  const int slot = queue.batchReserveCooperative(group, 1);
  if (slot < 0)
    return false;
  warpCopy(group, &queue.slot(slot), &element, sizeof(QueuedT));
  group.sync();
  return true;
}

template <class GroupT, class QueuedT>
__device__ __forceinline__ bool popBackCooperative(const GroupT&                         group,
                                                   SeedQueue<QueuedT, ThreadBlockScope>& queue,
                                                   QueuedT&                              outElement) {
  const int oldTop = queue.popReserveCooperative(group);
  if (oldTop < 0)
    return false;
  warpCopy(group, &outElement, &queue.slot(oldTop - 1), sizeof(QueuedT));
  group.sync();
  return true;
}

template <class GroupT>
__device__ __forceinline__ unsigned int readBestScoreCooperative(const GroupT& group, const unsigned int* bestScore) {
  unsigned int score = 0;
  if (group.thread_rank() == 0)
    score = *bestScore;
  return group.shfl(score, 0);
}

template <class GroupT> __device__ __forceinline__ bool readFlagCooperative(const GroupT& group, const bool* flag) {
  int value = 0;
  if (group.thread_rank() == 0)
    value = *flag ? 1 : 0;
  return group.shfl(value, 0) != 0;
}

template <int maxAtoms, int maxBonds, class GroupT>
__device__ __forceinline__ void seedComputeRemainingSizeRdkitCooperative(
  const GroupT&                                      group,
  Seed<maxAtoms, maxBonds>&                          seed,
  const DeviceCsrView&                               queryTopology,
  std::uint8_t*                                      atomStack,
  typename Seed<maxAtoms, maxBonds>::atom_word_type* visitedAtoms,
  typename Seed<maxAtoms, maxBonds>::bond_word_type* visitedBonds,
  int*                                               stackSize) {
  using SeedT                    = Seed<maxAtoms, maxBonds>;
  using AtomWord                 = typename SeedT::atom_word_type;
  using BondWord                 = typename SeedT::bond_word_type;
  constexpr int kAtomBitsPerWord = SeedT::kAtomBitsPerWord;
  constexpr int kBondBitsPerWord = SeedT::kBondBitsPerWord;

  if (group.thread_rank() == 0) {
    seed.remainingAtoms = 0;
    seed.remainingBonds = 0;
    *stackSize          = 0;
    for (int i = 0; i < SeedT::kAtomWords; ++i) {
      visitedAtoms[i] = seed.atoms[i];
    }
    for (int i = 0; i < SeedT::kBondWords; ++i) {
      visitedBonds[i] = seed.excludedBonds[i];
    }

    for (int wordIdx = 0; wordIdx < SeedT::kAtomWords; ++wordIdx) {
      AtomWord remaining = seed.lastAddedAtoms[wordIdx];
      while (remaining != 0) {
        int bitPosInWord;
        if constexpr (sizeof(AtomWord) == 4) {
          bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
        } else {
          bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
        }
        const int atomIdx = wordIdx * kAtomBitsPerWord + bitPosInWord;
        remaining &= remaining - 1;

        for (int bondIdx = 0; bondIdx < queryTopology.numBonds; ++bondIdx) {
          const int      bondWordIdx = bondIdx / kBondBitsPerWord;
          const BondWord bondMask    = static_cast<BondWord>(1) << (bondIdx % kBondBitsPerWord);
          if ((visitedBonds[bondWordIdx] & bondMask) != 0)
            continue;

          const std::uint32_t endpoints = queryTopology.bondEndpoints[bondIdx];
          const int           endpointU = static_cast<int>(endpoints >> kBondEndpointShift);
          const int           endpointV = static_cast<int>(endpoints & kBondEndpointMask);
          int                 otherAtom = -1;
          if (endpointU == atomIdx) {
            otherAtom = endpointV;
          } else if (endpointV == atomIdx) {
            otherAtom = endpointU;
          } else {
            continue;
          }

          visitedBonds[bondWordIdx] |= bondMask;
          seed.remainingBonds += 1;
          const int      atomWordIdx = otherAtom / kAtomBitsPerWord;
          const AtomWord atomMask    = static_cast<AtomWord>(1) << (otherAtom % kAtomBitsPerWord);
          if ((visitedAtoms[atomWordIdx] & atomMask) == 0) {
            visitedAtoms[atomWordIdx] |= atomMask;
            seed.remainingAtoms += 1;
            atomStack[(*stackSize)++] = static_cast<std::uint8_t>(otherAtom);
          }
        }
      }
    }

    while (*stackSize > 0) {
      const int atomIdx = atomStack[--(*stackSize)];
      for (int bondIdx = 0; bondIdx < queryTopology.numBonds; ++bondIdx) {
        const int      bondWordIdx = bondIdx / kBondBitsPerWord;
        const BondWord bondMask    = static_cast<BondWord>(1) << (bondIdx % kBondBitsPerWord);
        if ((visitedBonds[bondWordIdx] & bondMask) != 0)
          continue;

        const std::uint32_t endpoints = queryTopology.bondEndpoints[bondIdx];
        const int           endpointU = static_cast<int>(endpoints >> kBondEndpointShift);
        const int           endpointV = static_cast<int>(endpoints & kBondEndpointMask);
        int                 otherAtom = -1;
        if (endpointU == atomIdx) {
          otherAtom = endpointV;
        } else if (endpointV == atomIdx) {
          otherAtom = endpointU;
        } else {
          continue;
        }

        visitedBonds[bondWordIdx] |= bondMask;
        seed.remainingBonds += 1;
        const int      atomWordIdx = otherAtom / kAtomBitsPerWord;
        const AtomWord atomMask    = static_cast<AtomWord>(1) << (otherAtom % kAtomBitsPerWord);
        if ((visitedAtoms[atomWordIdx] & atomMask) == 0) {
          visitedAtoms[atomWordIdx] |= atomMask;
          seed.remainingAtoms += 1;
          atomStack[(*stackSize)++] = static_cast<std::uint8_t>(otherAtom);
        }
      }
    }
  }
  group.sync();
}

/// Per-block seed worklist capacity.  The backing slab lives in global
/// memory; only the cursor/header lives in shared memory.  Approach 1 uses
/// atomic LIFO push/pop so multiple warp groups can own grow work
/// concurrently.
constexpr int kFmcsQueueCapacity = 4096;
/// Legacy success-only mapping cache capacity.  The active RDKit-parity
/// kernel ignores the cache path; this constant remains for the standalone
/// DeviceMatchCache unit tests until an RDKit-equivalent cache is added.
constexpr int kFmcsCacheCapacity = 4096;
static_assert((kFmcsCacheCapacity & (kFmcsCacheCapacity - 1)) == 0, "kFmcsCacheCapacity must be a power of two");
/// Block / cooperative-group sizing.  Phase 2 partitions each block into warp
/// groups; each group pops one seed at a time from the block worklist and
/// cooperates on matching, remaining-size checks, seed copying, and fallback
/// substructure search.  Supported block sizes are compile-time kernel
/// specializations selected at launch time.
constexpr int kFmcsDefaultBlockSize = 128;
constexpr int kFmcsGroupSize        = 32;
static_assert(kFmcsGroupSize <= 32, "kFmcsGroupSize must be <= 32 (warp shuffle / ballot scope)");
static_assert((kFmcsGroupSize & (kFmcsGroupSize - 1)) == 0, "kFmcsGroupSize must be a power of two");

template <int blockThreads> struct FmcsBlockConfig {
  static_assert(blockThreads == 64 || blockThreads == 128 || blockThreads == 256 || blockThreads == 512,
                "fMCS block size must be 64, 128, 256, or 512");
  static_assert(blockThreads % kFmcsGroupSize == 0, "fMCS block size must be a multiple of kFmcsGroupSize");
  static constexpr int numGroups = blockThreads / kFmcsGroupSize;
};

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_SEARCH_SUPPORT_CUH
