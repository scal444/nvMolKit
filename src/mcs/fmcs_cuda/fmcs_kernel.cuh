// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef FMCS_CUDA_FMCS_KERNEL_CUH
#define FMCS_CUDA_FMCS_KERNEL_CUH

#include "fmcs_cuda/fmcs_debug.cuh"
#include "fmcs_cuda/fmcs_grow.cuh"
#include "fmcs_cuda/fmcs_match.cuh"
#include "fmcs_cuda/fmcs_match_tables.cuh"
#include "fmcs_cuda/fmcs_seed.cuh"
#include "fmcs_cuda/fmcs_seed_queue.cuh"
#include "mcs_common/mcs_cooperative_copy.cuh"

#include <cooperative_groups.h>

#include <cstdint>

namespace mcs {
namespace fmcs {

namespace cg = cooperative_groups;

/// Per-pair descriptor passed to the kernel.  Non-owning: pointers refer
/// into host-uploaded device buffers.  The caller chooses the smaller
/// input as the query (fMCS only enumerates subgraphs of the query) and
/// records the choice in @c swapped so host-side expansion can un-swap
/// the result mappings.
struct DevicePerPairInput {
  int queryNumAtoms = 0;
  int queryNumBonds = 0;
  int targetNumAtoms = 0;
  int targetNumBonds = 0;

  const std::uint32_t* queryRowOffsets = nullptr;
  const std::uint32_t* queryColIndices = nullptr;
  /// Packed (u << 16 | v), one entry per undirected bond, ordered to
  /// match the bond dimension of the match tables (@ref enumerateBonds).
  const std::uint32_t* queryBondEndpoints = nullptr;

  const std::uint32_t* targetRowOffsets = nullptr;
  const std::uint32_t* targetColIndices = nullptr;
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
template<int maxAtoms, int maxBonds>
struct DeviceMCSResult {
  int numCommonVertices = 0;
  int numCommonEdges    = 0;
  bool timedOut         = false;
  bool overflowed       = false;

  /// mappingA[i] = query atom idx, mappingB[i] = target atom idx of
  /// the i-th matched vertex.
  uint8_t mappingA[maxAtoms];
  uint8_t mappingB[maxAtoms];
  /// bondMapA[i] = query bond idx, bondMapB[i] = target bond idx of
  /// the i-th matched edge.
  uint8_t bondMapA[maxBonds];
  uint8_t bondMapB[maxBonds];
};

/// Non-owning view over one side's CSR + bond-endpoint arrays.  Passed to
/// matcher/grow helpers as the @c TargetTopology / @c QueryTopology.
struct DeviceCsrView {
  const std::uint32_t* rowOffsets    = nullptr;
  const std::uint32_t* colIndices    = nullptr;
  const std::uint32_t* bondEndpoints = nullptr;
  int numAtoms = 0;
  int numBonds = 0;
};

struct FmcsMeasureStats {
  unsigned int phase2Iters = 0;
  unsigned int initialSeeds = 0;
  unsigned int mismatchedInitialSeeds = 0;
  unsigned int popped = 0;
  unsigned int seedChecks = 0;
  unsigned int matchCalls = 0;
  unsigned int matchFound = 0;
  unsigned int boundRejected = 0;
  unsigned int expanded = 0;
  unsigned int fillZero = 0;
  unsigned int stage0Attempts = 0;
  unsigned int stage0Success = 0;
  unsigned int stage1Attempts = 0;
  unsigned int stage1Success = 0;
  unsigned int stage2Attempts = 0;
  unsigned int stage2Success = 0;
  unsigned int individualBondExcluded = 0;
  unsigned int fastAttempts = 0;
  unsigned int fastSuccess = 0;
  unsigned int fallbackCalls = 0;
  unsigned int fallbackSuccess = 0;
  unsigned int fallbackFail = 0;
  unsigned int fallbackOverflow = 0;
  unsigned int maxQueue = 0;
  unsigned int forcedExit = 0;
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
  template<class GroupT>
  __device__ __forceinline__ void zeroCooperative(const GroupT& group) {
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
    if (key == 0ULL) return false;
    const int mask = capacity - 1;
    int slot = static_cast<int>(splitMix64(key)) & mask;
    for (int attempt = 0; attempt < capacity; ++attempt) {
      const std::uint64_t prev = atomicCAS(
          reinterpret_cast<unsigned long long*>(&keys[slot]),
          0ULL,
          static_cast<unsigned long long>(key));
      if (prev == 0ULL) return true;   // claimed empty slot
      if (prev == key)  return true;   // already present (idempotent)
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
    if (key == 0ULL) return false;
    const int mask = capacity - 1;
    int slot = static_cast<int>(splitMix64(key)) & mask;
    for (int attempt = 0; attempt < capacity; ++attempt) {
      const std::uint64_t entry = keys[slot];
      if (entry == 0ULL) return false;
      if (entry == key)  return true;
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
template<int maxAtoms, int maxBonds, int maxTA, int maxTB>
__device__ __forceinline__ std::uint64_t mappingHashWithinThread(
    const Seed<maxAtoms, maxBonds>& seed,
    const MatchResult<maxAtoms, maxBonds, maxTA, maxTB>& match) {
  using SeedT = Seed<maxAtoms, maxBonds>;
  using AtomWord = typename SeedT::atom_word_type;
  using BondWord = typename SeedT::bond_word_type;
  constexpr int kAtomBitsPerWord = SeedT::kAtomBitsPerWord;
  constexpr int kAtomWords       = SeedT::kAtomWords;
  constexpr int kBondBitsPerWord = SeedT::kBondBitsPerWord;
  constexpr int kBondWords       = SeedT::kBondWords;

  std::uint64_t hash = 0;
  constexpr std::uint64_t kAtomDomain = 0x9e3779b97f4a7c15ULL;
  constexpr std::uint64_t kLastAddedDomain = 0xd6e8feb86659fd93ULL;
  constexpr std::uint64_t kBondDomain = 0xbf58476d1ce4e5b9ULL;
  constexpr std::uint64_t kExcludedDomain = 0x94d049bb133111ebULL;

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
      if (targetAtomIdx == kUnmappedTargetIdx) continue;

      const std::uint64_t pair =
          (static_cast<std::uint64_t>(queryAtomIdx + 1) << 8) |
           static_cast<std::uint64_t>(targetAtomIdx);
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
      hash = splitMix64(hash + kLastAddedDomain + token);
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
      if (targetBondIdx == kUnmappedTargetIdx) continue;

      // Pack (queryBondIdx, targetBondIdx) into a uint64 with no
      // overlap (target index is 8 bits; query index is 16-bit-safe
      // since maxBonds <= 128).  The query field is biased by +1 so the
      // packed token is never zero: the accumulator starts at 0 and
      // splitMix64(0) == 0, so a zero token (which a raw q=0,t=0 bond
      // would produce) would leave the accumulator at 0 and make that
      // bond invisible, aliasing a seed onto one that omits it.
      const std::uint64_t pair =
          (static_cast<std::uint64_t>(queryBondIdx + 1) << 8) |
           static_cast<std::uint64_t>(targetBondIdx);
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
      hash = splitMix64(hash + kExcludedDomain + token);
    }
  }
  return hash == 0ULL ? 1ULL : hash;
}

template<class GroupT, class QueuedT>
__device__ __forceinline__ void updateIncumbentCooperative(
    const GroupT& group,
    const QueuedT& candidate,
    QueuedT& best,
    unsigned int* bestScore,
    int* bestCopyLock) {
  const int groupRank = static_cast<int>(group.thread_rank());
  int locked = 0;
  int shouldCopy = 0;
  if (groupRank == 0) {
    const unsigned int candidateScore =
        (static_cast<unsigned int>(candidate.seed.numBonds) << 16) |
         static_cast<unsigned int>(candidate.seed.numAtoms);
    unsigned int prev = *bestScore;
    bool won = false;
    while (candidateScore > prev) {
      const unsigned int seen =
          atomicCAS(bestScore, prev, candidateScore);
      if (seen == prev) {
        won = true;
        break;
      }
      prev = seen;
    }
    if (won) {
      while (atomicCAS(bestCopyLock, 0, 1) != 0) {}
      locked = 1;
      shouldCopy = (candidateScore == *bestScore) ? 1 : 0;
    }
  }
  locked = group.shfl(locked, 0);
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

template<int maxAtoms, int maxBonds, int maxTA, int maxTB,
         class QueryTopology, class TargetTopology, class GroupT>
__device__ __forceinline__ bool checkSeedMatchAndAppendCooperative(
    const GroupT& group,
    QueuedSeed<maxAtoms, maxBonds, maxTA, maxTB>& candidate,
    const QueryTopology& queryTopology,
    const TargetTopology& targetTopology,
    const PairMatchTablesDevice& tables,
    FmcsSubstructureScratch<maxAtoms, maxTA>& scratch,
    int* scratchLock,
    std::uint8_t* partialStorage,
    int partialCapacity,
    bool* overflowedFlag,
    FmcsMeasureStats& stats) {
  const int groupRank = static_cast<int>(group.thread_rank());
  if constexpr (kFmcsMeasure) {
    if (groupRank == 0) {
      atomicAdd(&stats.seedChecks, 1u);
      atomicAdd(&stats.matchCalls, 1u);
    }
  }

  bool ok = false;
  if (!candidate.match.empty) {
    if constexpr (kFmcsMeasure) {
      if (groupRank == 0) atomicAdd(&stats.fastAttempts, 1u);
    }
    ok = matchIncrementalFastCooperative(
        group, candidate.seed, queryTopology, targetTopology, tables,
        candidate.match);
    if constexpr (kFmcsMeasure) {
      if (groupRank == 0 && ok) atomicAdd(&stats.fastSuccess, 1u);
    }
  }

  if (!ok) {
    if constexpr (kFmcsMeasure) {
      if (groupRank == 0) atomicAdd(&stats.fallbackCalls, 1u);
    }
    if (groupRank == 0) {
      while (atomicCAS(scratchLock, 0, 1) != 0) {}
    }
    group.sync();
    const bool overflowBefore =
        overflowedFlag != nullptr ? *overflowedFlag : false;
    ok = matchSeedSubstructureCooperative(
        group, candidate.seed, queryTopology, targetTopology, tables,
        candidate.match, scratch, partialStorage, partialCapacity,
        overflowedFlag);
    group.sync();
    if (groupRank == 0) {
      atomicExch(scratchLock, 0);
    }
    group.sync();
    if constexpr (kFmcsMeasure) {
      if (groupRank == 0) {
        if (ok) {
          atomicAdd(&stats.fallbackSuccess, 1u);
        } else if (overflowedFlag != nullptr && *overflowedFlag &&
                   !overflowBefore) {
          atomicAdd(&stats.fallbackOverflow, 1u);
        } else {
          atomicAdd(&stats.fallbackFail, 1u);
        }
      }
    }
  }

  if constexpr (kFmcsMeasure) {
    if (groupRank == 0 && ok) atomicAdd(&stats.matchFound, 1u);
  }
  return ok;
}

__device__ __forceinline__ bool isPowerOfTwo64(unsigned long long value) {
  return value != 0ULL && (value & (value - 1ULL)) == 0ULL;
}

template<class GroupT, class QueuedT>
__device__ __forceinline__ bool insertSortedByBondsCooperative(
    const GroupT& group,
    SeedQueue<QueuedT, ThreadBlockScope>& queue,
    const QueuedT& element) {
  const int groupRank = static_cast<int>(group.thread_rank());
  int oldSize = 0;
  int insertAt = 0;
  int ok = 1;
  if (groupRank == 0) {
    oldSize = queue.size();
    ok = oldSize < queue.capacity() ? 1 : 0;
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
  oldSize = group.shfl(oldSize, 0);
  insertAt = group.shfl(insertAt, 0);
  ok = group.shfl(ok, 0);
  if (!ok) return false;

  for (int i = oldSize; i > insertAt; --i) {
    warpCopy(group, &queue.slot(i), &queue.slot(i - 1), sizeof(QueuedT));
    group.sync();
  }
  warpCopy(group, &queue.slot(insertAt), &element, sizeof(QueuedT));
  group.sync();
  if (groupRank == 0) queue.setSizeWithinThread(oldSize + 1);
  group.sync();
  return true;
}

template<class GroupT, class QueuedT>
__device__ __forceinline__ bool popFrontCooperative(
    const GroupT& group,
    SeedQueue<QueuedT, ThreadBlockScope>& queue,
    QueuedT& outElement) {
  const int groupRank = static_cast<int>(group.thread_rank());
  int oldSize = 0;
  if (groupRank == 0) oldSize = queue.size();
  oldSize = group.shfl(oldSize, 0);
  if (oldSize <= 0) return false;

  warpCopy(group, &outElement, &queue.slot(0), sizeof(QueuedT));
  group.sync();
  for (int i = 1; i < oldSize; ++i) {
    warpCopy(group, &queue.slot(i - 1), &queue.slot(i), sizeof(QueuedT));
    group.sync();
  }
  if (groupRank == 0) queue.setSizeWithinThread(oldSize - 1);
  group.sync();
  return true;
}

template<int maxAtoms, int maxBonds, class QueryTopology, class GroupT>
__device__ __forceinline__ void seedComputeRemainingSizeRdkitCooperative(
    const GroupT& group,
    Seed<maxAtoms, maxBonds>& seed,
    const QueryTopology& queryTopology,
    std::uint8_t* atomStack,
    typename Seed<maxAtoms, maxBonds>::atom_word_type* visitedAtoms,
    typename Seed<maxAtoms, maxBonds>::bond_word_type* visitedBonds,
    int* stackSize) {
  using SeedT = Seed<maxAtoms, maxBonds>;
  using AtomWord = typename SeedT::atom_word_type;
  using BondWord = typename SeedT::bond_word_type;
  constexpr int kAtomBitsPerWord = SeedT::kAtomBitsPerWord;
  constexpr int kBondBitsPerWord = SeedT::kBondBitsPerWord;

  if (group.thread_rank() == 0) {
    seed.remainingAtoms = 0;
    seed.remainingBonds = 0;
    *stackSize = 0;
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
          bitPosInWord =
              __ffsll(static_cast<unsigned long long>(remaining)) - 1;
        }
        const int atomIdx = wordIdx * kAtomBitsPerWord + bitPosInWord;
        remaining &= remaining - 1;

        for (int bondIdx = 0; bondIdx < queryTopology.numBonds; ++bondIdx) {
          const int bondWordIdx = bondIdx / kBondBitsPerWord;
          const BondWord bondMask =
              static_cast<BondWord>(1) << (bondIdx % kBondBitsPerWord);
          if ((visitedBonds[bondWordIdx] & bondMask) != 0) continue;

          const std::uint32_t endpoints =
              queryTopology.bondEndpoints[bondIdx];
          const int endpointU =
              static_cast<int>(endpoints >> kBondEndpointShift);
          const int endpointV =
              static_cast<int>(endpoints & kBondEndpointMask);
          int otherAtom = -1;
          if (endpointU == atomIdx) {
            otherAtom = endpointV;
          } else if (endpointV == atomIdx) {
            otherAtom = endpointU;
          } else {
            continue;
          }

          visitedBonds[bondWordIdx] |= bondMask;
          seed.remainingBonds += 1;
          const int atomWordIdx = otherAtom / kAtomBitsPerWord;
          const AtomWord atomMask =
              static_cast<AtomWord>(1) << (otherAtom % kAtomBitsPerWord);
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
        const int bondWordIdx = bondIdx / kBondBitsPerWord;
        const BondWord bondMask =
            static_cast<BondWord>(1) << (bondIdx % kBondBitsPerWord);
        if ((visitedBonds[bondWordIdx] & bondMask) != 0) continue;

        const std::uint32_t endpoints =
            queryTopology.bondEndpoints[bondIdx];
        const int endpointU =
            static_cast<int>(endpoints >> kBondEndpointShift);
        const int endpointV =
            static_cast<int>(endpoints & kBondEndpointMask);
        int otherAtom = -1;
        if (endpointU == atomIdx) {
          otherAtom = endpointV;
        } else if (endpointV == atomIdx) {
          otherAtom = endpointU;
        } else {
          continue;
        }

        visitedBonds[bondWordIdx] |= bondMask;
        seed.remainingBonds += 1;
        const int atomWordIdx = otherAtom / kAtomBitsPerWord;
        const AtomWord atomMask =
            static_cast<AtomWord>(1) << (otherAtom % kAtomBitsPerWord);
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

/// Per-block sorted seed-set capacity.  The backing slab lives in global
/// memory; only the cursor/header lives in shared memory.  The active
/// RDKit-shaped kernel creates one initial seed per query bond and inserts
/// accepted children into a bond-count-sorted worklist, so this no longer
/// needs to cover queryBond * targetBond * orientation embeddings.
constexpr int kFmcsQueueCapacity = 4096;
/// Per-block substructure fallback partial capacity, expressed as
/// max-sized partial entries per ping-pong half.  The bodies live in
/// global memory as raw uint8 mappings; runtime effective capacity is
/// larger for smaller seeds because each partial uses only
/// seed.numAtoms bytes.
constexpr int kFmcsSubstructurePartialCapacity = 4096;
/// Legacy success-only mapping cache capacity.  The active RDKit-parity
/// kernel ignores the cache path; this constant remains for the standalone
/// DeviceMatchCache unit tests until an RDKit-equivalent cache is added.
constexpr int kFmcsCacheCapacity = 4096;
static_assert((kFmcsCacheCapacity & (kFmcsCacheCapacity - 1)) == 0,
              "kFmcsCacheCapacity must be a power of two");
/// Block / cooperative-group sizing.  The active RDKit-parity Phase 2
/// serializes the sorted worklist through group 0 so pop order matches
/// RDKit's SeedSet::pop/front behavior; lanes in that group still cooperate
/// on matching, remaining-size checks, and seed copying.  The per-group shared
/// arrays are kept because the helper layer supports cooperative groups, but
/// only group 0 owns grow work in the current kernel.
/// @c kFmcsGroupSize must evenly divide @c kFmcsBlockSize and (for the
/// warp-shuffle / ballot primitives we use) be a power of two <= 32.
constexpr int kFmcsBlockSize    = 128;
constexpr int kFmcsGroupSize    = 32;
constexpr int kFmcsNumGroups    = kFmcsBlockSize / kFmcsGroupSize;
static_assert(kFmcsBlockSize % kFmcsGroupSize == 0,
              "kFmcsBlockSize must be a multiple of kFmcsGroupSize");
static_assert(kFmcsGroupSize <= 32,
              "kFmcsGroupSize must be <= 32 (warp shuffle / ballot scope)");

/// One CUDA block per pair.  Three-phase seed-grow search:
///   Phase 1: RDKit makeInitialSeeds() analogue.  Build one query-bond
///   seed at a time, run checkIfMatchAndAppend() via substructure search,
///   store one witness MatchResult, and carry RDKit's initial
///   ExcludedBonds prefix behavior.
///   Phase 2: group 0 pops the front seed from the RDKit-style sorted queue
///   and cooperatively processes the RDKit Seed::grow() stages:
///   canGrowBiggerThan, fillNewBonds, Stage 0 all-outgoing-bonds child,
///   Stage 1 individual-bond pruning, and Stage 2 subset enumeration.
///   Exit on empty queue, timeout, or queue overflow.
///   Phase 3: block lane 0 writes the incumbent into DeviceMCSResult.
///
/// @p queueStorageAll points at a host-allocated global-memory slab of
/// @p queueCapacity * numPairs @c QueuedSeed entries; this block uses
/// the slice starting at @p queueStorageAll[blockIdx.x * queueCapacity].
///
/// @p cacheStorageAll and @p cacheCapacity are currently ignored.  They remain
/// in the signature while the old cache scaffolding is still compiled for unit
/// tests; the active RDKit-parity kernel does not allocate or probe it.
template<int maxAtoms, int maxBonds, class Policy>
__global__ void fmcsKernel(
    const DevicePerPairInput* __restrict__ pairs,
    DeviceMCSResult<maxAtoms, maxBonds>* __restrict__ results,
    QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>* __restrict__ queueStorageAll,
    std::uint64_t* __restrict__ cacheStorageAll,
    std::uint8_t* __restrict__ substructureStorageAll,
    int queueCapacity,
    int cacheCapacity,
    int substructurePartialCapacity,
    int numPairs,
    std::uint32_t timeoutUs) {
  (void)timeoutUs;
  (void)cacheStorageAll;
  (void)cacheCapacity;

  const int pairIdx = blockIdx.x;
  if (pairIdx >= numPairs) return;

  auto block = cg::this_thread_block();
  const DevicePerPairInput& pair = pairs[pairIdx];

  if constexpr (kFmcsDebug) {
    if (pairIdx == kFmcsDebugPairIdx && block.thread_rank() == 0) {
      printf("[fmcs] kernel start: pair=%d q=(a%d,b%d) t=(a%d,b%d)\n",
             pairIdx, pair.queryNumAtoms, pair.queryNumBonds,
             pair.targetNumAtoms, pair.targetNumBonds);
    }
  }

  using QueuedT = QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>;
  using SubstructureScratchT = FmcsSubstructureScratch<maxAtoms, maxAtoms>;
  constexpr int kMaxNewBondsForTier = maxBonds;

  // Block-shared resources: the queue, the incumbent, and the early-exit
  // flags are visible to every group.  Cross-group accesses use atomics
  // for queue and incumbent updates; the bool flags tolerate benign races
  // (stale reads only cause extra work or duplicate flag-true writes).
  __shared__ SeedQueue<QueuedT, ThreadBlockScope> queue;
  __shared__ __align__(16) unsigned char bestStorage[sizeof(QueuedT)];
  QueuedT& best = *reinterpret_cast<QueuedT*>(bestStorage);
  // Atomic incumbent score: high 16 bits = numBonds, low 16 bits =
  // numAtoms.  Groups race through atomicCAS on this single int to
  // claim the right to write @c best.
  __shared__ unsigned int bestScore;
  __shared__ int bestCopyLock;
  __shared__ DeviceCsrView queryView;
  __shared__ DeviceCsrView targetView;
  __shared__ bool overflowed;
  __shared__ bool timedOut;
  __shared__ SubstructureScratchT substructureScratch;
  __shared__ int substructureScratchLock;
  __shared__ FmcsMeasureStats measureStats;

  // Cooperative Phase 2 working state.  Only group 0 currently owns RDKit
  // grow work so the sorted queue is processed in the same order as RDKit;
  // the arrays stay per-group to keep helper signatures group-local and to
  // avoid local storage if we later add an explicitly measured parallel mode.
  __shared__ __align__(16) unsigned char
      currentStorage[sizeof(QueuedT) * kFmcsNumGroups];
  __shared__ __align__(16) unsigned char
      biggestStorage[sizeof(QueuedT) * kFmcsNumGroups];
  QueuedT* current = reinterpret_cast<QueuedT*>(currentStorage);
  QueuedT* biggest = reinterpret_cast<QueuedT*>(biggestStorage);
  __shared__ NewBond newBondsArr[kFmcsNumGroups][kMaxNewBondsForTier];
  __shared__ int     newBondCount[kFmcsNumGroups];
  __shared__ bool    popped[kFmcsNumGroups];
  __shared__ bool    stage0Ok[kFmcsNumGroups];
  __shared__ std::uint8_t remainingAtomStack[kFmcsNumGroups][maxAtoms];
  __shared__ typename Seed<maxAtoms, maxBonds>::atom_word_type
      remainingVisitedAtoms[kFmcsNumGroups][Seed<maxAtoms, maxBonds>::kAtomWords];
  __shared__ typename Seed<maxAtoms, maxBonds>::bond_word_type
      remainingVisitedBonds[kFmcsNumGroups][Seed<maxAtoms, maxBonds>::kBondWords];
  __shared__ int remainingStackSize[kFmcsNumGroups];
  __shared__ typename Seed<maxAtoms, maxBonds>::bond_word_type
      initialExcludedBonds[Seed<maxAtoms, maxBonds>::kBondWords];

  QueuedT* myQueueStorage =
      queueStorageAll + static_cast<size_t>(pairIdx) * queueCapacity;
  std::uint8_t* mySubstructureStorage =
      substructureStorageAll +
      static_cast<size_t>(pairIdx) * 2u *
          static_cast<size_t>(substructurePartialCapacity) *
          static_cast<size_t>(maxAtoms);

  if (block.thread_rank() == 0) {
    queue.init(myQueueStorage, queueCapacity);
    seedClearWithinThread(best.seed);
    matchResultClearWithinThread(best.match);
    bestScore = 0;
    bestCopyLock = 0;
    substructureScratchLock = 0;
    if constexpr (kFmcsMeasure) {
      measureStats = FmcsMeasureStats{};
    }
    overflowed = false;
    timedOut = false;

    queryView.rowOffsets    = pair.queryRowOffsets;
    queryView.colIndices    = pair.queryColIndices;
    queryView.bondEndpoints = pair.queryBondEndpoints;
    queryView.numAtoms      = pair.queryNumAtoms;
    queryView.numBonds      = pair.queryNumBonds;

    targetView.rowOffsets    = pair.targetRowOffsets;
    targetView.colIndices    = pair.targetColIndices;
    targetView.bondEndpoints = pair.targetBondEndpoints;
    targetView.numAtoms      = pair.targetNumAtoms;
    targetView.numBonds      = pair.targetNumBonds;
  }
  block.sync();

  if constexpr (kFmcsDebug) {
    if (pairIdx == kFmcsDebugPairIdx && block.thread_rank() == 0) {
      printf("[fmcs] init done\n");
    }
  }

  auto group = cg::tiled_partition<kFmcsGroupSize>(block);
  const int groupId   = static_cast<int>(block.thread_rank()) / kFmcsGroupSize;
  const int groupRank = static_cast<int>(group.thread_rank());
  QueuedT& myCurrent  = current[groupId];
  QueuedT& myBiggest  = biggest[groupId];
  NewBond* myNewBonds = newBondsArr[groupId];
  std::uint8_t* myRemainingAtomStack = remainingAtomStack[groupId];
  auto* myRemainingVisitedAtoms = remainingVisitedAtoms[groupId];
  auto* myRemainingVisitedBonds = remainingVisitedBonds[groupId];

  // ---- Phase 1: RDKit makeInitialSeeds() analogue ----
  // RDKit creates one initial seed per query bond, not one per target
  // embedding.  Each candidate goes through checkIfMatchAndAppend(), which
  // runs substructure matching and stores one witness MatchResult on success.
  // Initial ExcludedBonds is prefix-like: later initial seeds exclude earlier
  // query bonds, and a mismatched initial bond is also excluded from seeds
  // already admitted.  Group 0 handles this serial state; the substructure
  // check remains cooperative across that group's lanes.
  if (block.thread_rank() < Seed<maxAtoms, maxBonds>::kBondWords) {
    initialExcludedBonds[block.thread_rank()] = 0;
  }
  block.sync();

  if (groupId == 0) {
    for (int qBond = 0; qBond < pair.queryNumBonds && !overflowed; ++qBond) {
      if (groupRank == 0) {
        seedClearWithinThread(myCurrent.seed);
        matchResultClearWithinThread(myCurrent.match);
        for (int wordIdx = 0;
             wordIdx < Seed<maxAtoms, maxBonds>::kBondWords;
             ++wordIdx) {
          myCurrent.seed.excludedBonds[wordIdx] =
              initialExcludedBonds[wordIdx];
        }

        const std::uint32_t queryEndpoints =
            queryView.bondEndpoints[qBond];
        const int queryEndpointU =
            static_cast<int>(queryEndpoints >> kBondEndpointShift);
        const int queryEndpointV =
            static_cast<int>(queryEndpoints & kBondEndpointMask);
        seedAddBondWithinThread(myCurrent.seed, qBond);
        seedAddAtomWithinThread(myCurrent.seed, queryEndpointU);
        seedAddAtomWithinThread(myCurrent.seed, queryEndpointV);
        myCurrent.seed.growingStage = kSeedGrowStageOuter;
        if constexpr (kFmcsMeasure) {
          atomicAdd(&measureStats.initialSeeds, 1u);
        }
      }
      group.sync();
      seedComputeRemainingSizeRdkitCooperative(
          group, myCurrent.seed, queryView, myRemainingAtomStack,
          myRemainingVisitedAtoms, myRemainingVisitedBonds,
          &remainingStackSize[groupId]);

      const bool matched = checkSeedMatchAndAppendCooperative(
          group, myCurrent, queryView, targetView, pair.tables,
          substructureScratch, &substructureScratchLock,
          mySubstructureStorage, substructurePartialCapacity, &overflowed,
          measureStats);
      if (matched) {
        updateIncumbentCooperative(
            group, myCurrent, best, &bestScore, &bestCopyLock);
        if (!insertSortedByBondsCooperative(group, queue, myCurrent)) {
          overflowed = true;
        }
      } else if (groupRank == 0) {
        const int queuedSeeds = queue.size();
        for (int i = 0; i < queuedSeeds; ++i) {
          seedExcludeBondWithinThread(queue.slot(i).seed, qBond);
        }
        if constexpr (kFmcsMeasure) {
          atomicAdd(&measureStats.mismatchedInitialSeeds, 1u);
        }
      }
      group.sync();

      if (groupRank == 0) {
        const int wordIdx = qBond / Seed<maxAtoms, maxBonds>::kBondBitsPerWord;
        const auto mask =
            static_cast<typename Seed<maxAtoms, maxBonds>::bond_word_type>(1)
            << (qBond % Seed<maxAtoms, maxBonds>::kBondBitsPerWord);
        initialExcludedBonds[wordIdx] |= mask;
      }
      group.sync();
    }
  }

  block.sync();

  if constexpr (kFmcsDebug) {
    if (pairIdx == kFmcsDebugPairIdx && block.thread_rank() == 0) {
      printf("[fmcs] phase1 done: acceptedInitialSeeds=%d queryBonds=%d overflowed=%d\n",
             queue.size(), pair.queryNumBonds, static_cast<int>(overflowed));
    }
  }

  // ---- Phase 2: RDKit Seed::grow() analogue ----

  // RDKit's growSeeds() repeatedly pops the front of a sorted SeedSet.
  // To preserve that ordering, only group 0 pops and grows a seed.  The
  // remaining groups still reach the block-level rendezvous each iteration so
  // all shared-memory state stays synchronized.
  [[maybe_unused]] int debugIter = 0;
  while (true) {
    if (overflowed || timedOut) break;

    if constexpr (kFmcsMeasure) {
      if (block.thread_rank() == 0) {
        measureStats.phase2Iters += 1;
        const unsigned int qsz = static_cast<unsigned int>(queue.size());
        if (qsz > measureStats.maxQueue) measureStats.maxQueue = qsz;
      }
    }

    if constexpr (kFmcsDebug) {
      if (pairIdx == kFmcsDebugPairIdx && block.thread_rank() == 0) {
        printf("[fmcs][iter %d] queueSize=%d best=(b%d,a%d)\n",
               debugIter, queue.size(),
               static_cast<int>(best.seed.numBonds),
               static_cast<int>(best.seed.numAtoms));
      }
    }

    const bool poppedThisGroup =
        groupId == 0 ? popFrontCooperative(group, queue, myCurrent) : false;
    if (groupRank == 0) {
      popped[groupId] = poppedThisGroup;
      if constexpr (kFmcsMeasure) {
        if (popped[groupId]) atomicAdd(&measureStats.popped, 1u);
      }
    }
    group.sync();

    do {
      if (!popped[groupId]) break;

      if constexpr (kFmcsDebug) {
        if (pairIdx == kFmcsDebugPairIdx && groupRank == 0) {
          printf("[fmcs][grp %d iter %d] A: popped seed (b%d,a%d)\n",
                 groupId, debugIter, static_cast<int>(myCurrent.seed.numBonds),
                 static_cast<int>(myCurrent.seed.numAtoms));
        }
      }

      // RDKit growSeeds() increments TotalSteps before Seed::grow(), then
      // Seed::grow() first checks canGrowBiggerThan().
      if constexpr (kFmcsMeasure) {
        if (groupRank == 0) atomicAdd(&measureStats.expanded, 1u);
      }

      const unsigned int scoreSnapshot = bestScore;
      const int bestBondsSnapshot = static_cast<int>(scoreSnapshot >> 16);
      const int bestAtomsSnapshot =
          static_cast<int>(scoreSnapshot & 0xFFFFu);
      if (!seedCanGrowBiggerThanWithinThread(
              myCurrent.seed, bestBondsSnapshot, bestAtomsSnapshot)) {
        if constexpr (kFmcsMeasure) {
          if (groupRank == 0) atomicAdd(&measureStats.boundRejected, 1u);
        }
        break;
      }

      updateIncumbentCooperative(
          group, myCurrent, best, &bestScore, &bestCopyLock);

      if constexpr (kFmcsDebug) {
        if (pairIdx == kFmcsDebugPairIdx && groupRank == 0) {
          printf("[fmcs][grp %d iter %d] B: after incumbent sync\n",
                 groupId, debugIter);
        }
      }

      const bool fillOk = fillNewBondsCooperative(
          group, myCurrent.seed, queryView, myNewBonds,
          &newBondCount[groupId], kMaxNewBondsForTier);
      group.sync();
      if (!fillOk) {
        if (groupRank == 0) overflowed = true;
        break;
      }

      if constexpr (kFmcsDebug) {
        if (pairIdx == kFmcsDebugPairIdx && groupRank == 0) {
          printf("[fmcs][grp %d iter %d] C: after fillNewBonds, count=%d stage=%u\n",
                 groupId, debugIter, newBondCount[groupId],
                 static_cast<unsigned int>(myCurrent.seed.growingStage));
        }
      }
      if (newBondCount[groupId] == 0) {
        if constexpr (kFmcsMeasure) {
          if (groupRank == 0) atomicAdd(&measureStats.fillZero, 1u);
        }
        break;
      }

      bool runInnerStage =
          myCurrent.seed.growingStage != kSeedGrowStageOuter;

      // RDKit Seed::grow() stage 0: build the child containing all newly
      // discovered outgoing bonds and run checkIfMatchAndAppend().  If this
      // all-bonds child matches and there is more than one new bond, RDKit
      // returns immediately with the parent left at GrowingStage=1; the next
      // outer grow loop resumes the parent at the singleton/subset stage.
      if (myCurrent.seed.growingStage == kSeedGrowStageOuter) {
        if constexpr (kFmcsMeasure) {
          if (groupRank == 0) atomicAdd(&measureStats.stage0Attempts, 1u);
        }
        if constexpr (kFmcsDebug) {
          if (pairIdx == kFmcsDebugPairIdx && groupRank == 0) {
            printf("[fmcs][grp %d iter %d] D: stage0 begin\n", groupId, debugIter);
          }
        }
        warpCopy(group, &myBiggest, &myCurrent, sizeof(QueuedT));
        group.sync();
        if (groupRank == 0) {
          seedBeginGrowStepWithinThread(myBiggest.seed);
          myBiggest.seed.growingStage = kSeedGrowStageOuter;
          for (int i = 0; i < newBondCount[groupId]; ++i) {
            seedAddNewBondWithinThread(myBiggest.seed, myNewBonds[i]);
          }
        }
        group.sync();
        seedComputeRemainingSizeRdkitCooperative(
            group, myBiggest.seed, queryView, myRemainingAtomStack,
            myRemainingVisitedAtoms, myRemainingVisitedBonds,
            &remainingStackSize[groupId]);

        const unsigned int childScoreSnapshot = bestScore;
        const int childBestBonds =
            static_cast<int>(childScoreSnapshot >> 16);
        const int childBestAtoms =
            static_cast<int>(childScoreSnapshot & 0xFFFFu);
        if (!seedCanGrowBiggerThanWithinThread(
                myBiggest.seed, childBestBonds, childBestAtoms)) {
          if constexpr (kFmcsMeasure) {
            if (groupRank == 0) atomicAdd(&measureStats.boundRejected, 1u);
          }
          break;
        }

        const bool ok = checkSeedMatchAndAppendCooperative(
            group, myBiggest, queryView, targetView, pair.tables,
            substructureScratch, &substructureScratchLock,
            mySubstructureStorage, substructurePartialCapacity, &overflowed,
            measureStats);
        if (groupRank == 0) stage0Ok[groupId] = ok;
        group.sync();

        if constexpr (kFmcsDebug) {
          if (pairIdx == kFmcsDebugPairIdx && groupRank == 0) {
            printf("[fmcs][grp %d iter %d] E: stage0 after match, ok=%d\n",
                   groupId, debugIter, static_cast<int>(stage0Ok[groupId]));
          }
        }
        if (stage0Ok[groupId]) {
          if constexpr (kFmcsMeasure) {
            if (groupRank == 0) atomicAdd(&measureStats.stage0Success, 1u);
          }
          updateIncumbentCooperative(
              group, myBiggest, best, &bestScore, &bestCopyLock);
          if (!insertSortedByBondsCooperative(group, queue, myBiggest)) {
            overflowed = true;
          }
          group.sync();
          if (newBondCount[groupId] > 1) {
            if (groupRank == 0) {
              myCurrent.seed.growingStage = kSeedGrowStageInner;
            }
            group.sync();
            if (!insertSortedByBondsCooperative(group, queue, myCurrent)) {
              overflowed = true;
            }
            group.sync();
          }
          break;
        }
        if (newBondCount[groupId] == 1) break;
        runInnerStage = true;
      }

      if (!runInnerStage) break;

      // RDKit Seed::grow() stage 1: try every individual outgoing bond.
      // A failed individual match excludes that NewBond from later subset
      // enumeration and increments IndividualBondExcluded.
      for (int i = 0; i < newBondCount[groupId]; ++i) {
        if (!myNewBonds[i].alive) continue;

        warpCopy(group, &myBiggest, &myCurrent, sizeof(QueuedT));
        group.sync();

        if (groupRank == 0) {
          seedBeginGrowStepWithinThread(myBiggest.seed);
          myBiggest.seed.growingStage = kSeedGrowStageOuter;
          seedAddNewBondWithinThread(myBiggest.seed, myNewBonds[i]);
        }
        group.sync();
        seedComputeRemainingSizeRdkitCooperative(
            group, myBiggest.seed, queryView, myRemainingAtomStack,
            myRemainingVisitedAtoms, myRemainingVisitedBonds,
            &remainingStackSize[groupId]);

        if constexpr (kFmcsMeasure) {
          if (groupRank == 0) atomicAdd(&measureStats.stage1Attempts, 1u);
        }

        const unsigned int childScoreSnapshot = bestScore;
        const int childBestBonds =
            static_cast<int>(childScoreSnapshot >> 16);
        const int childBestAtoms =
            static_cast<int>(childScoreSnapshot & 0xFFFFu);
        if (!seedCanGrowBiggerThanWithinThread(
                myBiggest.seed, childBestBonds, childBestAtoms)) {
          if constexpr (kFmcsMeasure) {
            if (groupRank == 0) atomicAdd(&measureStats.boundRejected, 1u);
          }
          continue;
        }

        const bool ok = checkSeedMatchAndAppendCooperative(
            group, myBiggest, queryView, targetView, pair.tables,
            substructureScratch, &substructureScratchLock,
            mySubstructureStorage, substructurePartialCapacity, &overflowed,
            measureStats);
        if constexpr (kFmcsMeasure) {
          if (groupRank == 0 && ok) atomicAdd(&measureStats.stage1Success, 1u);
        }
        if (ok) {
          updateIncumbentCooperative(
              group, myBiggest, best, &bestScore, &bestCopyLock);
          if (!insertSortedByBondsCooperative(group, queue, myBiggest)) {
            overflowed = true;
          }
        } else if (groupRank == 0) {
          myNewBonds[i].alive = false;
          if constexpr (kFmcsMeasure) {
            atomicAdd(&measureStats.individualBondExcluded, 1u);
          }
        }
        group.sync();
      }

      // RDKit Seed::grow() stage 2: enumerate all non-singleton subsets of
      // the surviving NewBonds.  Singletons were handled by Stage 1; the
      // all-bonds subset was handled by Stage 0 unless Stage 1 erased one or
      // more individual bonds, in which case the all-surviving-bonds subset is
      // new work and must be checked.
      int aliveCount = 0;
      int aliveOverflow = 0;
      if (groupRank == 0) {
        for (int i = 0; i < newBondCount[groupId]; ++i) {
          if (myNewBonds[i].alive) ++aliveCount;
        }
        aliveOverflow = aliveCount > 63 ? 1 : 0;
        if (aliveOverflow) overflowed = true;
      }
      aliveCount = group.shfl(aliveCount, 0);
      aliveOverflow = group.shfl(aliveOverflow, 0);
      if (aliveOverflow) break;
      if (aliveCount > 1) {
        unsigned int erasedCount = 0;
        if (groupRank == 0) {
          erasedCount =
              static_cast<unsigned int>(newBondCount[groupId] - aliveCount);
        }
        erasedCount = group.shfl(erasedCount, 0);
        const unsigned long long maxComposition =
            (1ULL << aliveCount) - 1ULL;
        for (unsigned long long composition = maxComposition;
             composition != 0ULL;
             --composition) {
          if (isPowerOfTwo64(composition)) continue;
          if (erasedCount == 0 && composition == maxComposition) continue;

          warpCopy(group, &myBiggest, &myCurrent, sizeof(QueuedT));
          group.sync();
          if (groupRank == 0) {
            seedBeginGrowStepWithinThread(myBiggest.seed);
            myBiggest.seed.growingStage = kSeedGrowStageOuter;
            int aliveBit = 0;
            for (int i = 0; i < newBondCount[groupId]; ++i) {
              if (!myNewBonds[i].alive) continue;
              if ((composition & (1ULL << aliveBit)) != 0ULL) {
                seedAddNewBondWithinThread(myBiggest.seed, myNewBonds[i]);
              }
              ++aliveBit;
            }
          }
          group.sync();
          seedComputeRemainingSizeRdkitCooperative(
              group, myBiggest.seed, queryView, myRemainingAtomStack,
              myRemainingVisitedAtoms, myRemainingVisitedBonds,
              &remainingStackSize[groupId]);

          if constexpr (kFmcsMeasure) {
            if (groupRank == 0) atomicAdd(&measureStats.stage2Attempts, 1u);
          }
          const unsigned int childScoreSnapshot = bestScore;
          const int childBestBonds =
              static_cast<int>(childScoreSnapshot >> 16);
          const int childBestAtoms =
              static_cast<int>(childScoreSnapshot & 0xFFFFu);
          if (!seedCanGrowBiggerThanWithinThread(
                  myBiggest.seed, childBestBonds, childBestAtoms)) {
            if constexpr (kFmcsMeasure) {
              if (groupRank == 0) atomicAdd(&measureStats.boundRejected, 1u);
            }
            continue;
          }

          const bool ok = checkSeedMatchAndAppendCooperative(
              group, myBiggest, queryView, targetView, pair.tables,
              substructureScratch, &substructureScratchLock,
              mySubstructureStorage, substructurePartialCapacity, &overflowed,
              measureStats);
          if (ok) {
            if constexpr (kFmcsMeasure) {
              if (groupRank == 0) atomicAdd(&measureStats.stage2Success, 1u);
            }
            updateIncumbentCooperative(
                group, myBiggest, best, &bestScore, &bestCopyLock);
            if (!insertSortedByBondsCooperative(group, queue, myBiggest)) {
              overflowed = true;
            }
          }
          group.sync();
          if (overflowed) break;
        }
      }

      if constexpr (kFmcsDebug) {
        if (pairIdx == kFmcsDebugPairIdx && groupRank == 0) {
          printf("[fmcs][grp %d iter %d] K: inner stages done\n",
                 groupId, debugIter);
        }
      }
    } while (false);

    if constexpr (kFmcsDebug) {
      if (pairIdx == kFmcsDebugPairIdx && groupRank == 0) {
        printf("[fmcs][grp %d iter %d] L: reached end-of-iter block.sync\n",
               groupId, debugIter);
      }
    }

    // End-of-iteration rendezvous: every group must reach here before
    // group 0 decides whether the sorted queue is empty.
    block.sync();

    if constexpr (kFmcsDebug) {
      if (pairIdx == kFmcsDebugPairIdx && block.thread_rank() == 0) {
        printf("[fmcs][iter %d] M: passed end-of-iter block.sync, queueSize=%d\n",
               debugIter, queue.size());
      }
    }
    if (queue.empty()) break;

    if constexpr (kFmcsDebug) {
      ++debugIter;
      if (debugIter >= kFmcsDebugMaxIters) {
        if (pairIdx == kFmcsDebugPairIdx && block.thread_rank() == 0) {
          printf("[fmcs] WATCHDOG: hit %d iters, queueSize=%d -- forcing exit\n",
                 debugIter, queue.size());
        }
        break;
      }
    }
    if constexpr (kFmcsMeasure) {
      ++debugIter;
      if (debugIter >= kFmcsMeasureMaxIters) {
        if (block.thread_rank() == 0) {
          measureStats.forcedExit = 1;
          timedOut = true;
        }
        break;
      }
    }
  }

  if constexpr (kFmcsDebug) {
    if (pairIdx == kFmcsDebugPairIdx && block.thread_rank() == 0) {
      printf("[fmcs] phase2 exit: iters=%d queueSize=%d overflowed=%d "
             "timedOut=%d best=(b%d,a%d)\n",
             debugIter, queue.size(), static_cast<int>(overflowed),
             static_cast<int>(timedOut),
             static_cast<int>(best.seed.numBonds),
             static_cast<int>(best.seed.numAtoms));
    }
  }
  if constexpr (kFmcsMeasure) {
    if (pairIdx == kFmcsDebugPairIdx && block.thread_rank() == 0) {
      printf("[fmcs][measure] iters=%u initial=%u/%u popped=%u expanded=%u "
             "seedChecks=%u match=%u/%u boundReject=%u indivExcluded=%u "
             "fillZero=%u stage0=%u/%u stage1=%u/%u stage2=%u/%u "
             "fast=%u/%u slow=%u success=%u fail=%u overflow=%u "
             "maxQueue=%u forcedExit=%u overflowed=%d timedOut=%d "
             "best=(b%d,a%d) queue=%d\n",
             measureStats.phase2Iters,
             measureStats.initialSeeds - measureStats.mismatchedInitialSeeds,
             measureStats.initialSeeds,
             measureStats.popped,
             measureStats.expanded,
             measureStats.seedChecks,
             measureStats.matchFound,
             measureStats.matchCalls,
             measureStats.boundRejected,
             measureStats.individualBondExcluded,
             measureStats.fillZero,
             measureStats.stage0Success,
             measureStats.stage0Attempts,
             measureStats.stage1Success,
             measureStats.stage1Attempts,
             measureStats.stage2Success,
             measureStats.stage2Attempts,
             measureStats.fastSuccess,
             measureStats.fastAttempts,
             measureStats.fallbackCalls,
             measureStats.fallbackSuccess,
             measureStats.fallbackFail,
             measureStats.fallbackOverflow,
             measureStats.maxQueue,
             measureStats.forcedExit,
             static_cast<int>(overflowed),
             static_cast<int>(timedOut),
             static_cast<int>(best.seed.numBonds),
             static_cast<int>(best.seed.numAtoms),
             queue.size());
    }
  }

  // ---- Phase 3: writeback ----
  block.sync();
  if (block.thread_rank() == 0) {
    auto& dst = results[pairIdx];
    dst.numCommonVertices = best.seed.numAtoms;
    dst.numCommonEdges    = best.seed.numBonds;
    dst.timedOut          = timedOut;
    dst.overflowed        = overflowed;

    // Walk set bits of best.seed.atoms (in increasing query atom idx
    // order, via __ffs/__ffsll) to fill mappingA/B.
    using BestSeedT  = decltype(best.seed);
    using AtomWord   = typename BestSeedT::atom_word_type;
    constexpr int kAtomBitsPerWord = BestSeedT::kAtomBitsPerWord;
    constexpr int kAtomWords       = BestSeedT::kAtomWords;
    int outIdx = 0;
    for (int wordIdx = 0; wordIdx < kAtomWords; ++wordIdx) {
      AtomWord remaining = best.seed.atoms[wordIdx];
      while (remaining != 0) {
        int bitPosInWord;
        if constexpr (sizeof(AtomWord) == 4) {
          bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
        } else {
          bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
        }
        const int queryAtomIdx = wordIdx * kAtomBitsPerWord + bitPosInWord;
        remaining &= remaining - 1;
        dst.mappingA[outIdx] = static_cast<std::uint8_t>(queryAtomIdx);
        dst.mappingB[outIdx] = best.match.targetAtomIdx[queryAtomIdx];
        ++outIdx;
      }
    }

    // Walk set bits of best.seed.bonds to fill bondMapA/B.
    using BondWord = typename BestSeedT::bond_word_type;
    constexpr int kBondBitsPerWord = BestSeedT::kBondBitsPerWord;
    constexpr int kBondWords       = BestSeedT::kBondWords;
    outIdx = 0;
    for (int wordIdx = 0; wordIdx < kBondWords; ++wordIdx) {
      BondWord remaining = best.seed.bonds[wordIdx];
      while (remaining != 0) {
        int bitPosInWord;
        if constexpr (sizeof(BondWord) == 4) {
          bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
        } else {
          bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
        }
        const int queryBondIdx = wordIdx * kBondBitsPerWord + bitPosInWord;
        remaining &= remaining - 1;
        dst.bondMapA[outIdx] = static_cast<std::uint8_t>(queryBondIdx);
        dst.bondMapB[outIdx] = best.match.targetBondIdx[queryBondIdx];
        ++outIdx;
      }
    }
  }
}

enum class MaxSizeTier {
  k16,
  k32,
  k64,
  k128,
};

/// Smallest tier whose bitset width covers both counts.  Returns -1 when
/// the largest tier (128) does not fit; the caller must flag
/// @ref MCSResult::overflowed.
inline int pickMaxSizeTier(int numAtoms, int numBonds) {
  const int need = numAtoms > numBonds ? numAtoms : numBonds;
  if (need <= 16) return 0;
  if (need <= 32) return 1;
  if (need <= 64) return 2;
  if (need <= 128) return 3;
  return -1;
}

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_KERNEL_CUH
