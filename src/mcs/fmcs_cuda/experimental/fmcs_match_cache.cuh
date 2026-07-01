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
//
// EXPERIMENTAL: cross-lineage success cache for the fMCS seed-grow search.
// The active RDKit-parity kernel does not consult this cache; the code is
// retained here for a future RDKit-equivalent caching path and is exercised
// only by the standalone unit tests.

#ifndef FMCS_CUDA_EXPERIMENTAL_FMCS_MATCH_CACHE_CUH
#define FMCS_CUDA_EXPERIMENTAL_FMCS_MATCH_CACHE_CUH

#include "fmcs_cuda/fmcs_seed.cuh"

#include <cstdint>

namespace mcs {
namespace fmcs {

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

/// Legacy success-only mapping cache capacity.  The active RDKit-parity
/// kernel ignores the cache path; this constant remains for the standalone
/// DeviceMatchCache unit tests until an RDKit-equivalent cache is added.
constexpr int kFmcsCacheCapacity = 4096;
static_assert((kFmcsCacheCapacity & (kFmcsCacheCapacity - 1)) == 0,
              "kFmcsCacheCapacity must be a power of two");

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_EXPERIMENTAL_FMCS_MATCH_CACHE_CUH
