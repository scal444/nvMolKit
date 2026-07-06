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

#ifndef FMCS_CUDA_FMCS_MATCH_CUH
#define FMCS_CUDA_FMCS_MATCH_CUH

#include "fmcs_cuda/fmcs_match_tables.cuh"
#include "fmcs_cuda/fmcs_seed.cuh"

#include <cstdint>

namespace mcs {
namespace fmcs {

// ---------------------------------------------------------------------------
// Shared matching primitives
// ---------------------------------------------------------------------------

/// Bond endpoints are stored across the kernel as a single uint32 with
/// the u-endpoint atom index in the high 16 bits and the v-endpoint
/// atom index in the low 16 bits.  Tiers cap maxAtoms at 128 so 16 bits
/// per index is plenty.
constexpr int          kBondEndpointShift = 16;
constexpr std::uint32_t kBondEndpointMask  = 0xFFFFu;
constexpr int          kFallbackAdjacencySubwarpSize = 4;

// FIXME(group-size): this hardcodes the full 32-lane mask and broadcasts from
// warp lane 0. That is correct ONLY while kFmcsGroupSize == 32 and the
// cg::tiled_partition<32> groups are warp-aligned (one group == one warp), so
// "lane 0 of the warp" == "rank 0 of the group". fmcs_config.cuh currently
// allows kFmcsGroupSize to be any power of two <= 32; if it is ever reduced
// (e.g. to 16, giving two sub-warp groups per warp), this would broadcast the
// value from group 0's lane 0 to BOTH groups in the warp -> silent wrong
// pruning/stage decisions with no error. Either add
// static_assert(kFmcsGroupSize == 32) to pin the invariant, or route this
// through group.shfl(input, 0) so it follows the actual group scope.
__device__ __forceinline__ int mark_warp_uniform(const int input) {
  return __shfl_sync(0xffffffffu, input, 0);
}

__device__ __forceinline__ int reserveSharedCounterSlotWarpAggregated(
    int* counter, int* overflowed, const int capacity) {
  const unsigned int activeMask = __activemask();
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int leaderLane = __ffs(activeMask) - 1;
  const unsigned int laneMaskLt =
      lane == 0 ? 0u : ((1u << static_cast<unsigned int>(lane)) - 1u);
  const int offset = __popc(activeMask & laneMaskLt);
  int base = 0;
  if (lane == leaderLane) {
    base = atomicAdd(counter, __popc(activeMask));
  }
  base = __shfl_sync(activeMask, base, leaderLane);
  const int slot = base + offset;
  if (slot >= capacity) {
    atomicExch(overflowed, 1);
    return -1;
  }
  return slot;
}

/// Resolved target endpoints for a successful single-(query bond, target
/// bond, orientation) compatibility check.  Populated by
/// @ref matchSingleBondWithinThread on success only; contents are
/// unspecified on failure.  @c targetAtomU is the target atom that the
/// query bond's u endpoint was mapped to (likewise V).
struct SingleBondMatch {
  uint8_t targetAtomU;
  uint8_t targetAtomV;
};

/// Invariant: a topology either has compile-time adjacency with valid CSR
/// arrays, or carries no CSR pointers at all.  The matchers' non-adjacency
/// paths scan bondEndpoints directly and never consult CSR pointers.
template<class Topology>
__host__ __device__ constexpr bool topologyHasAdjacencyBondIndices() {
  if constexpr (requires { Topology::kHasAdjacencyBondIndices; }) {
    return Topology::kHasAdjacencyBondIndices;
  } else {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Fast incremental matching
// ---------------------------------------------------------------------------

/// Within-thread: per-lane single-(query bond, target bond, orientation)
/// compatibility check used by Phase 1 initial-seed enumeration.  Writes
/// resolved target atom indices for the two endpoints of @p queryBondIdx
/// into @p outMatch and returns true on success; on false the caller
/// should not read @p outMatch.
///
/// @p reversed selects the orientation: when false, the query bond's u
/// endpoint maps to the target bond's u endpoint; when true, to the
/// target bond's v endpoint.
///
/// @p queryTopology and @p targetTopology must expose a
/// @c bondEndpoints array of packed (u<<16 | v) entries.
template<class QueryTopology, class TargetTopology>
__device__ __forceinline__ bool matchSingleBondWithinThread(
    const int queryBondIdx,
    const int targetBondIdx,
    const bool reversed,
    const QueryTopology& queryTopology,
    const TargetTopology& targetTopology,
    const PairMatchTablesDevice& tables,
    SingleBondMatch& outMatch) {
  // Cheap bond-table check first; if the bond labels are incompatible
  // we never need to touch the atom table or decode endpoints.
  if (!tables.bonds.testBit(queryBondIdx, targetBondIdx)) return false;

  // Decode the packed (u<<16 | v) endpoint encoding for both bonds.
  const std::uint32_t queryEndpoints  = queryTopology.bondEndpoints[queryBondIdx];
  const int           queryEndpointU  =
      static_cast<int>(queryEndpoints >> kBondEndpointShift);
  const int           queryEndpointV  =
      static_cast<int>(queryEndpoints & kBondEndpointMask);

  const std::uint32_t targetEndpoints = targetTopology.bondEndpoints[targetBondIdx];
  const int           targetEndpointU =
      static_cast<int>(targetEndpoints >> kBondEndpointShift);
  const int           targetEndpointV =
      static_cast<int>(targetEndpoints & kBondEndpointMask);

  // Pick which target endpoint the query's u maps to; v gets the other.
  // reversed=false: queryU -> targetU, queryV -> targetV.
  // reversed=true:  queryU -> targetV, queryV -> targetU.
  const int targetForQueryU = reversed ? targetEndpointV : targetEndpointU;
  const int targetForQueryV = reversed ? targetEndpointU : targetEndpointV;

  // Both atom-pairings must be label-compatible; either failure rejects
  // this orientation.
  if (!tables.atoms.testBit(queryEndpointU, targetForQueryU)) return false;
  if (!tables.atoms.testBit(queryEndpointV, targetForQueryV)) return false;

  outMatch.targetAtomU = static_cast<uint8_t>(targetForQueryU);
  outMatch.targetAtomV = static_cast<uint8_t>(targetForQueryV);
  return true;
}

/// Cooperative: extend @p match by every query bond in @p seed.bonds
/// whose @c match.targetBondIdx[q] is still @ref kUnmappedTargetIdx
/// (i.e., unmapped by the parent's recorded embedding).  For each such
/// bond:
///   - Both endpoints already mapped -> ring-closing case.  The lanes
///     of @p group scan target bonds in parallel for one whose endpoint
///     pair exactly matches the mapped (queryU, queryV) target atoms
///     and is unvisited, with the bond-match-table bit set.  First
///     compatible target bond commits.
///   - Exactly one endpoint mapped -> atom-adding case.  Lane-parallel
///     scan for a target bond incident to the mapped target atom whose
///     other end is unvisited, atom-table-compatible with the unmapped
///     query atom, and bond-table-compatible.  First compatible
///     candidate commits both the new bond mapping and the new atom
///     mapping, and marks both visited.
///   - Both endpoints unmapped -> defensive fail (shouldn't occur on
///     well-formed seeds, where Phase 1 maps both initial atoms before
///     pushing).
/// Any bond that fails to extend causes the function to return false;
/// @p match is left in an unspecified state and the caller should
/// discard the seed.
template<int maxAtoms, int maxBonds, int maxTA, int maxTB,
         class QueryTopology, class TargetTopology, class GroupT>
__device__ __forceinline__ bool matchIncrementalFastCooperative(
    const GroupT& group,
    const Seed<maxAtoms, maxBonds>& seed,
    const QueryTopology& queryTopology,
    const TargetTopology& targetTopology,
    const PairMatchTablesDevice& tables,
    MatchResult<maxAtoms, maxBonds, maxTA, maxTB>& match) {
  using SeedT  = Seed<maxAtoms, maxBonds>;
  using MatchT = MatchResult<maxAtoms, maxBonds, maxTA, maxTB>;
  using BondWord = typename SeedT::bond_word_type;

  constexpr int kBondBitsPerWord = SeedT::kBondBitsPerWord;
  constexpr int kBondWords       = SeedT::kBondWords;
  constexpr int kTargetAtomBitsPerWord = MatchT::kTargetAtomBitsPerWord;
  constexpr int kTargetBondBitsPerWord = MatchT::kTargetBondBitsPerWord;
  using TargetAtomWord = typename MatchT::target_atom_word;
  using TargetBondWord = typename MatchT::target_bond_word;

  const int laneRank  = static_cast<int>(group.thread_rank());
  const int laneCount = static_cast<int>(group.num_threads());

  // Outer loop walks set bits of seed.bonds via __ffs/__ffsll.  Each
  // iteration handles one query bond q; if q is already mapped (parent
  // saw it) we skip, otherwise we extend the match by one bond +
  // possibly one atom.
  for (int wordIdx = 0; wordIdx < kBondWords; ++wordIdx) {
    BondWord remainingBondBits = seed.bonds[wordIdx];
    while (remainingBondBits != 0) {
      int bitPosInWord;
      if constexpr (sizeof(BondWord) == 4) {
        bitPosInWord = __ffs(static_cast<unsigned int>(remainingBondBits)) - 1;
      } else {
        bitPosInWord = __ffsll(static_cast<unsigned long long>(remainingBondBits)) - 1;
      }
      const int queryBondIdx = wordIdx * kBondBitsPerWord + bitPosInWord;
      remainingBondBits &= remainingBondBits - 1;  // clear lowest set bit

      // Skip bonds already mapped by the parent's recorded embedding.  Keep
      // this control decision uniform across the group; diverging before the
      // later ballot/shuffle would make the winning lane undefined.
      int mappedTargetBond = kUnmappedTargetIdx;
      if (laneRank == 0) mappedTargetBond = match.targetBondIdx[queryBondIdx];
      mappedTargetBond = group.shfl(mappedTargetBond, 0);
      if (mappedTargetBond != kUnmappedTargetIdx) continue;

      // Decode this bond's query endpoints from the packed (u<<16 | v).
      const std::uint32_t queryEndpoints =
          queryTopology.bondEndpoints[queryBondIdx];
      const int queryEndpointU =
          static_cast<int>(queryEndpoints >> kBondEndpointShift);
      const int queryEndpointV =
          static_cast<int>(queryEndpoints & kBondEndpointMask);

      // Look up the parent's atom mapping for both endpoints; either
      // may already be mapped (from an earlier bond) or still unmapped
      // (this bond is the one bringing it in).
      int targetForQueryU = kUnmappedTargetIdx;
      int targetForQueryV = kUnmappedTargetIdx;
      if (laneRank == 0) {
        targetForQueryU = match.targetAtomIdx[queryEndpointU];
        targetForQueryV = match.targetAtomIdx[queryEndpointV];
      }
      targetForQueryU = group.shfl(targetForQueryU, 0);
      targetForQueryV = group.shfl(targetForQueryV, 0);
      const bool queryUIsMapped = targetForQueryU != kUnmappedTargetIdx;
      const bool queryVIsMapped = targetForQueryV != kUnmappedTargetIdx;

      // Both endpoints unmapped means the seed is missing an earlier
      // bond that should have brought one of them in.  Phase 1 always
      // anchors both initial atoms before pushing, so this never hits
      // for well-formed seeds; defensive return false.
      if (!queryUIsMapped && !queryVIsMapped) return false;

      // Each lane records its first compatible target bond, and (for
      // the atom-adding case) the resulting new target atom.  After
      // the per-lane scan, the warp ballots and the lowest-rank winner
      // is broadcast as the committed extension.
      int chosenTargetBond = -1;
      int chosenTargetAtomForUnmapped = -1;  // -1 means ring-closing.

      if (queryUIsMapped && queryVIsMapped) {
        // Ring-closing: the new bond connects two atoms that are both
        // already in the seed's mapping.  Find a target bond whose
        // endpoints are exactly the pair { targetForQueryU, targetForQueryV }.
        const int srcTargetAtom = targetForQueryU;
        const int dstTargetAtom = targetForQueryV;
        if constexpr (topologyHasAdjacencyBondIndices<TargetTopology>()) {
          if (srcTargetAtom >= 0 && srcTargetAtom < targetTopology.numAtoms) {
            const int begin =
                static_cast<int>(targetTopology.rowOffsets[srcTargetAtom]);
            const int end =
                static_cast<int>(targetTopology.rowOffsets[srcTargetAtom + 1]);
            for (int adjIdx = begin + laneRank;
                 adjIdx < end && chosenTargetBond < 0;
                 adjIdx += laneCount) {
              const int otherTargetAtom =
                  static_cast<int>(targetTopology.colIndices[adjIdx]);
              if (otherTargetAtom != dstTargetAtom) continue;
              const int targetBondIdx =
                  static_cast<int>(targetTopology.bondIndices[adjIdx]);
              if (targetBondIdx < 0 ||
                  targetBondIdx >= targetTopology.numBonds ||
                  targetBondIdx >= maxTB) {
                continue;
              }
              const TargetBondWord visitedBondsWord =
                  match.visitedTargetBonds[targetBondIdx / kTargetBondBitsPerWord];
              if ((visitedBondsWord >> (targetBondIdx % kTargetBondBitsPerWord)) & 1) {
                continue;
              }
              if (!tables.bonds.testBit(queryBondIdx, targetBondIdx)) continue;
              chosenTargetBond = targetBondIdx;
            }
          }
        } else {
          for (int targetBondIdx = laneRank;
               targetBondIdx < targetTopology.numBonds && chosenTargetBond < 0;
               targetBondIdx += laneCount) {
            // Skip target bonds already used by the parent's match.
            const TargetBondWord visitedBondsWord =
                match.visitedTargetBonds[targetBondIdx / kTargetBondBitsPerWord];
            if ((visitedBondsWord >> (targetBondIdx % kTargetBondBitsPerWord)) & 1) {
              continue;
            }
            const std::uint32_t targetEndpoints =
                targetTopology.bondEndpoints[targetBondIdx];
            const int targetEndpointU =
                static_cast<int>(targetEndpoints >> kBondEndpointShift);
            const int targetEndpointV =
                static_cast<int>(targetEndpoints & kBondEndpointMask);
            // Match either orientation -- target bonds are undirected.
            const bool endpointsMatch =
                (targetEndpointU == srcTargetAtom &&
                 targetEndpointV == dstTargetAtom) ||
                (targetEndpointU == dstTargetAtom &&
                 targetEndpointV == srcTargetAtom);
            if (!endpointsMatch) continue;
            if (!tables.bonds.testBit(queryBondIdx, targetBondIdx)) continue;
            chosenTargetBond = targetBondIdx;
          }
        }
      } else {
        // Atom-adding: one query endpoint (`src`) is already mapped to
        // a target atom; the other (`dst`) is what we're trying to
        // place.  Scan target bonds incident to srcTargetAtom for one
        // whose other endpoint is unvisited, atom-table-compatible
        // with the unmapped query atom, and bond-table-compatible.
        const int unmappedQueryAtom =
            queryUIsMapped ? queryEndpointV : queryEndpointU;
        const int srcTargetAtom     = queryUIsMapped ? targetForQueryU
                                                     : targetForQueryV;
        if constexpr (topologyHasAdjacencyBondIndices<TargetTopology>()) {
          if (srcTargetAtom >= 0 && srcTargetAtom < targetTopology.numAtoms) {
            const int begin =
                static_cast<int>(targetTopology.rowOffsets[srcTargetAtom]);
            const int end =
                static_cast<int>(targetTopology.rowOffsets[srcTargetAtom + 1]);
            for (int adjIdx = begin + laneRank;
                 adjIdx < end && chosenTargetBond < 0;
                 adjIdx += laneCount) {
              const int targetBondIdx =
                  static_cast<int>(targetTopology.bondIndices[adjIdx]);
              if (targetBondIdx < 0 ||
                  targetBondIdx >= targetTopology.numBonds ||
                  targetBondIdx >= maxTB) {
                continue;
              }
              const TargetBondWord visitedBondsWord =
                  match.visitedTargetBonds[targetBondIdx / kTargetBondBitsPerWord];
              if ((visitedBondsWord >> (targetBondIdx % kTargetBondBitsPerWord)) & 1) {
                continue;
              }
              const int candidateTargetAtom =
                  static_cast<int>(targetTopology.colIndices[adjIdx]);
              if (candidateTargetAtom < 0 ||
                  candidateTargetAtom >= targetTopology.numAtoms ||
                  candidateTargetAtom >= maxTA) {
                continue;
              }
              const TargetAtomWord visitedAtomsWord =
                  match.visitedTargetAtoms[candidateTargetAtom / kTargetAtomBitsPerWord];
              if ((visitedAtomsWord >>
                   (candidateTargetAtom % kTargetAtomBitsPerWord)) & 1) {
                continue;
              }
              if (!tables.bonds.testBit(queryBondIdx, targetBondIdx)) continue;
              if (!tables.atoms.testBit(unmappedQueryAtom, candidateTargetAtom)) continue;
              chosenTargetBond            = targetBondIdx;
              chosenTargetAtomForUnmapped = candidateTargetAtom;
            }
          }
        } else {
          for (int targetBondIdx = laneRank;
               targetBondIdx < targetTopology.numBonds && chosenTargetBond < 0;
               targetBondIdx += laneCount) {
            const TargetBondWord visitedBondsWord =
                match.visitedTargetBonds[targetBondIdx / kTargetBondBitsPerWord];
            if ((visitedBondsWord >> (targetBondIdx % kTargetBondBitsPerWord)) & 1) {
              continue;
            }
            const std::uint32_t targetEndpoints =
                targetTopology.bondEndpoints[targetBondIdx];
            const int targetEndpointU =
                static_cast<int>(targetEndpoints >> kBondEndpointShift);
            const int targetEndpointV =
                static_cast<int>(targetEndpoints & kBondEndpointMask);
            // Identify the candidate target atom on the far side of the
            // bond from srcTargetAtom; skip bonds not incident to it.
            int candidateTargetAtom;
            if (targetEndpointU == srcTargetAtom) {
              candidateTargetAtom = targetEndpointV;
            } else if (targetEndpointV == srcTargetAtom) {
              candidateTargetAtom = targetEndpointU;
            } else {
              continue;
            }
            // Candidate target atom must not already be in the embedding.
            const TargetAtomWord visitedAtomsWord =
                match.visitedTargetAtoms[candidateTargetAtom / kTargetAtomBitsPerWord];
            if ((visitedAtomsWord >>
                 (candidateTargetAtom % kTargetAtomBitsPerWord)) & 1) {
              continue;
            }
            if (!tables.bonds.testBit(queryBondIdx, targetBondIdx)) continue;
            if (!tables.atoms.testBit(unmappedQueryAtom, candidateTargetAtom)) continue;
            chosenTargetBond            = targetBondIdx;
            chosenTargetAtomForUnmapped = candidateTargetAtom;
          }
        }
      }

      // Warp-wide pick: ballot for any lane with a candidate, broadcast
      // the lowest-rank winner's choice to all lanes.  No candidate
      // anywhere -> this bond can't be extended -> abandon the seed.
      const unsigned ballot = group.ballot(chosenTargetBond >= 0 ? 1u : 0u);
      if (ballot == 0u) return false;
      const int firstWinningLane = __ffs(ballot) - 1;
      const int committedTargetBond =
          group.shfl(chosenTargetBond, firstWinningLane);
      const int committedTargetAtom =
          group.shfl(chosenTargetAtomForUnmapped, firstWinningLane);
      if (committedTargetBond < 0 ||
          committedTargetBond >= targetTopology.numBonds ||
          committedTargetBond >= maxTB) {
        return false;
      }
      if (committedTargetAtom < -1 ||
          committedTargetAtom >= targetTopology.numAtoms ||
          committedTargetAtom >= maxTA) {
        return false;
      }

      // Lane 0 commits the new mapping into the shared MatchResult after all
      // lanes have finished reading the previous match state.
      group.sync();
      if (laneRank == 0) {
        match.targetBondIdx[queryBondIdx] =
            static_cast<std::uint8_t>(committedTargetBond);
        match.visitedTargetBonds[committedTargetBond / kTargetBondBitsPerWord] |=
            static_cast<TargetBondWord>(1)
            << (committedTargetBond % kTargetBondBitsPerWord);
        match.matchedBondSize += 1;
        match.empty = false;
        if (committedTargetAtom >= 0) {
          // Atom-adding case: also commit the new atom mapping.
          const int unmappedQueryAtom =
              queryUIsMapped ? queryEndpointV : queryEndpointU;
          match.targetAtomIdx[unmappedQueryAtom] =
              static_cast<std::uint8_t>(committedTargetAtom);
          match.visitedTargetAtoms[committedTargetAtom / kTargetAtomBitsPerWord] |=
              static_cast<TargetAtomWord>(1)
              << (committedTargetAtom % kTargetAtomBitsPerWord);
          match.matchedAtomSize += 1;
        }
      }
      group.sync();
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Full substructure fallback
// ---------------------------------------------------------------------------

template<int maxAtoms, int maxBonds, int maxTargetAtoms>
struct FmcsSubstructureScratch {
  // Scratch for the RDKit checkIfMatchAndAppend fallback.  This is deliberately
  // caller-owned shared memory, not function-local state: tier-128 scratch is
  // too large to risk compiler-created stack/local memory in the matcher.
  std::uint8_t seedAtomList[maxAtoms];
  std::uint8_t seedAtoms[maxAtoms];
  std::uint8_t seedDegree[maxAtoms];
  std::uint8_t mappedSeedNeighborCount[maxAtoms];
  std::uint16_t seedNeighborOffset[maxAtoms + 1];
  std::uint8_t seedNeighborAtom[2 * maxBonds];
  std::uint16_t seedNeighborBond[2 * maxBonds];
  std::uint8_t targetDegree[maxTargetAtoms];
  std::uint8_t orderedQueryAtom[maxAtoms];
  std::uint8_t queryOrderPos[maxAtoms];
  std::uint8_t targetAtomForQuery[maxAtoms];
  int currentCount;
  int nextCount;
  int found;
  int overflowed;
};

__device__ __forceinline__ void setOverflowedFlagWithinThread(int* flag) {
  if (flag != nullptr) atomicExch(flag, 1);
}

__device__ __forceinline__ void setOverflowedFlagWithinThread(bool* flag) {
  if (flag != nullptr) *flag = true;
}

template<class TargetTopology>
__device__ __forceinline__ bool findTargetBondBetweenAtomsWithinThread(
    const int targetAtomA,
    const int targetAtomB,
    const int queryBondIdx,
    const TargetTopology& targetTopology,
    const PairMatchTablesDevice& tables,
    int& outTargetBondIdx) {
  if constexpr (topologyHasAdjacencyBondIndices<TargetTopology>()) {
    if (targetAtomA < 0 || targetAtomA >= targetTopology.numAtoms) {
      return false;
    }
    const int begin = static_cast<int>(targetTopology.rowOffsets[targetAtomA]);
    const int end = static_cast<int>(targetTopology.rowOffsets[targetAtomA + 1]);
    for (int adjIdx = begin; adjIdx < end; ++adjIdx) {
      const int otherTargetAtom =
          static_cast<int>(targetTopology.colIndices[adjIdx]);
      if (otherTargetAtom != targetAtomB) continue;
      const int targetBondIdx =
          static_cast<int>(targetTopology.bondIndices[adjIdx]);
      if (targetBondIdx < 0 || targetBondIdx >= targetTopology.numBonds) {
        continue;
      }
      if (!tables.bonds.testBit(queryBondIdx, targetBondIdx)) continue;
      outTargetBondIdx = targetBondIdx;
      return true;
    }
    return false;
  } else {
    for (int targetBondIdx = 0;
         targetBondIdx < targetTopology.numBonds;
         ++targetBondIdx) {
      const std::uint32_t targetEndpoints =
          targetTopology.bondEndpoints[targetBondIdx];
      const int targetEndpointU =
          static_cast<int>(targetEndpoints >> kBondEndpointShift);
      const int targetEndpointV =
          static_cast<int>(targetEndpoints & kBondEndpointMask);
      const bool endpointsMatch =
          (targetEndpointU == targetAtomA && targetEndpointV == targetAtomB) ||
          (targetEndpointU == targetAtomB && targetEndpointV == targetAtomA);
      if (!endpointsMatch) continue;
      if (!tables.bonds.testBit(queryBondIdx, targetBondIdx)) continue;
      outTargetBondIdx = targetBondIdx;
      return true;
    }
    return false;
  }
}

template<int maxAtoms, int maxBonds, int maxTA, int maxTB,
         class QueryTopology, class TargetTopology>
__device__ __forceinline__ bool rebuildMatchFromSubstructureMappingWithinThread(
    const Seed<maxAtoms, maxBonds>& seed,
    const QueryTopology& queryTopology,
    const TargetTopology& targetTopology,
    const PairMatchTablesDevice& tables,
    MatchResult<maxAtoms, maxBonds, maxTA, maxTB>& match,
    FmcsSubstructureScratch<maxAtoms, maxBonds, maxTA>& scratch) {
  using SeedT  = Seed<maxAtoms, maxBonds>;
  using MatchT = MatchResult<maxAtoms, maxBonds, maxTA, maxTB>;
  using BondWord = typename SeedT::bond_word_type;
  using TargetBondWord = typename MatchT::target_bond_word;
  using TargetAtomWord = typename MatchT::target_atom_word;

  constexpr int kBondBitsPerWord = SeedT::kBondBitsPerWord;
  constexpr int kBondWords       = SeedT::kBondWords;
  constexpr int kTargetAtomBitsPerWord = MatchT::kTargetAtomBitsPerWord;
  constexpr int kTargetBondBitsPerWord = MatchT::kTargetBondBitsPerWord;

  matchResultClearWithinThread(match);
  for (int i = 0; i < seed.numAtoms; ++i) {
    const int queryAtomIdx = scratch.seedAtoms[i];
    const int targetAtomIdx = scratch.targetAtomForQuery[queryAtomIdx];
    if (targetAtomIdx == kUnmappedTargetIdx) return false;
    match.targetAtomIdx[queryAtomIdx] =
        static_cast<std::uint8_t>(targetAtomIdx);
    match.visitedTargetAtoms[targetAtomIdx / kTargetAtomBitsPerWord] |=
        static_cast<TargetAtomWord>(1)
        << (targetAtomIdx % kTargetAtomBitsPerWord);
  }
  match.matchedAtomSize = seed.numAtoms;

  int matchedBondCount = 0;
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

      const std::uint32_t queryEndpoints =
          queryTopology.bondEndpoints[queryBondIdx];
      const int queryEndpointU =
          static_cast<int>(queryEndpoints >> kBondEndpointShift);
      const int queryEndpointV =
          static_cast<int>(queryEndpoints & kBondEndpointMask);
      const int targetEndpointU = match.targetAtomIdx[queryEndpointU];
      const int targetEndpointV = match.targetAtomIdx[queryEndpointV];
      if (targetEndpointU == kUnmappedTargetIdx ||
          targetEndpointV == kUnmappedTargetIdx) {
        matchResultClearWithinThread(match);
        return false;
      }

      int targetBondIdx = -1;
      if (!findTargetBondBetweenAtomsWithinThread(
              targetEndpointU, targetEndpointV, queryBondIdx,
              targetTopology, tables, targetBondIdx)) {
        matchResultClearWithinThread(match);
        return false;
      }
      const TargetBondWord visitedWord =
          match.visitedTargetBonds[targetBondIdx / kTargetBondBitsPerWord];
      if ((visitedWord >> (targetBondIdx % kTargetBondBitsPerWord)) & 1) {
        matchResultClearWithinThread(match);
        return false;
      }
      match.targetBondIdx[queryBondIdx] =
          static_cast<std::uint8_t>(targetBondIdx);
      match.visitedTargetBonds[targetBondIdx / kTargetBondBitsPerWord] |=
          static_cast<TargetBondWord>(1)
          << (targetBondIdx % kTargetBondBitsPerWord);
      ++matchedBondCount;
    }
  }
  match.matchedBondSize = static_cast<std::uint16_t>(matchedBondCount);
  if (matchedBondCount != seed.numBonds) {
    matchResultClearWithinThread(match);
    return false;
  }
  match.empty = false;
  return true;
}

template<int maxAtoms, int maxBonds, int maxTA, class TargetTopology, class GroupT>
__device__ __forceinline__ void initializeSeedSubstructureScratchCooperative(
    const GroupT& group,
    const TargetTopology& targetTopology,
    FmcsSubstructureScratch<maxAtoms, maxBonds, maxTA>& scratch) {
  const int laneRank  = static_cast<int>(group.thread_rank());
  const int laneCount = static_cast<int>(group.num_threads());

  for (int i = laneRank; i < maxAtoms; i += laneCount) {
    scratch.seedDegree[i] = 0;
    scratch.mappedSeedNeighborCount[i] = 0;
    scratch.orderedQueryAtom[i] = 0;
    scratch.queryOrderPos[i] = kUnmappedTargetIdx;
    scratch.targetAtomForQuery[i] = kUnmappedTargetIdx;
  }

  if constexpr (topologyHasAdjacencyBondIndices<TargetTopology>()) {
    for (int targetAtomIdx = laneRank;
         targetAtomIdx < targetTopology.numAtoms;
         targetAtomIdx += laneCount) {
      scratch.targetDegree[targetAtomIdx] = static_cast<std::uint8_t>(
          targetTopology.rowOffsets[targetAtomIdx + 1] -
          targetTopology.rowOffsets[targetAtomIdx]);
    }
  } else {
    for (int i = laneRank; i < maxTA; i += laneCount) {
      scratch.targetDegree[i] = 0;
    }
    group.sync();
    if (laneRank == 0) {
      for (int targetBondIdx = 0;
           targetBondIdx < targetTopology.numBonds;
           ++targetBondIdx) {
        const std::uint32_t targetEndpoints =
            targetTopology.bondEndpoints[targetBondIdx];
        const int targetEndpointU =
            static_cast<int>(targetEndpoints >> kBondEndpointShift);
        const int targetEndpointV =
            static_cast<int>(targetEndpoints & kBondEndpointMask);
        ++scratch.targetDegree[targetEndpointU];
        ++scratch.targetDegree[targetEndpointV];
      }
    }
  }

  if (laneRank == 0) {
    scratch.currentCount = 0;
    scratch.nextCount = 0;
    scratch.found = 0;
    scratch.overflowed = 0;
  }
  group.sync();
}

template<int maxAtoms, int maxBonds, int maxTA>
__device__ __forceinline__ void incrementMappedSeedNeighborCountsWithinThread(
    FmcsSubstructureScratch<maxAtoms, maxBonds, maxTA>& scratch,
    const int queryAtomIdx) {
  const int begin = static_cast<int>(scratch.seedNeighborOffset[queryAtomIdx]);
  const int end =
      static_cast<int>(scratch.seedNeighborOffset[queryAtomIdx + 1]);
  for (int adjIdx = begin; adjIdx < end; ++adjIdx) {
    const int otherQueryAtom = scratch.seedNeighborAtom[adjIdx];
    if (!scratch.orderedQueryAtom[otherQueryAtom]) {
      ++scratch.mappedSeedNeighborCount[otherQueryAtom];
    }
  }
}

template<int maxAtoms, int maxBonds, int maxTA, class TargetTopology, class GroupT>
__device__ __forceinline__ int countCandidateTargetAtomsCooperative(
    const GroupT& group,
    const int queryAtomIdx,
    const TargetTopology& targetTopology,
    const PairMatchTablesDevice& tables,
    const FmcsSubstructureScratch<maxAtoms, maxBonds, maxTA>& scratch) {
  const int laneRank  = static_cast<int>(group.thread_rank());
  const int laneCount = static_cast<int>(group.num_threads());
  const int requiredDegree =
      static_cast<int>(scratch.seedDegree[queryAtomIdx]);

  int candidateCount = 0;
  for (int targetBase = 0;
       targetBase < targetTopology.numAtoms;
       targetBase += laneCount) {
    const int targetAtomIdx = targetBase + laneRank;
    bool compatible = targetAtomIdx < targetTopology.numAtoms;
    if (compatible) {
      compatible =
          scratch.targetDegree[targetAtomIdx] >= requiredDegree &&
          tables.atoms.testBit(queryAtomIdx, targetAtomIdx);
    }
    const unsigned ballot = group.ballot(compatible ? 1u : 0u);
    if (laneRank == 0) candidateCount += __popc(ballot);
  }
  candidateCount = group.shfl(candidateCount, 0);
  return candidateCount;
}

template<int maxAtoms, int maxBonds, int maxTA,
         class QueryTopology, class TargetTopology, class GroupT>
__device__ __forceinline__ bool prepareSeedSubstructureSearchCooperative(
    const GroupT& group,
    const Seed<maxAtoms, maxBonds>& seed,
    const QueryTopology& queryTopology,
    const TargetTopology& targetTopology,
    const PairMatchTablesDevice& tables,
    FmcsSubstructureScratch<maxAtoms, maxBonds, maxTA>& scratch,
    int& numSeedAtoms) {
  const int laneRank = static_cast<int>(group.thread_rank());
  using SeedT = Seed<maxAtoms, maxBonds>;
  using AtomWord = typename SeedT::atom_word_type;
  using BondWord = typename SeedT::bond_word_type;

  constexpr int kAtomBitsPerWord = SeedT::kAtomBitsPerWord;
  constexpr int kAtomWords       = SeedT::kAtomWords;
  constexpr int kBondBitsPerWord = SeedT::kBondBitsPerWord;
  constexpr int kBondWords       = SeedT::kBondWords;

  int prepareOk = 1;
  if (laneRank == 0) {
    if (seed.numAtoms > targetTopology.numAtoms ||
        seed.numBonds > targetTopology.numBonds) {
      prepareOk = 0;
    } else {
      numSeedAtoms = 0;
      for (int wordIdx = 0; wordIdx < kAtomWords; ++wordIdx) {
        AtomWord remaining = seed.atoms[wordIdx];
        while (remaining != 0) {
          int bitPosInWord;
          if constexpr (sizeof(AtomWord) == 4) {
            bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
          } else {
            bitPosInWord =
                __ffsll(static_cast<unsigned long long>(remaining)) - 1;
          }
          const int queryAtomIdx = wordIdx * kAtomBitsPerWord + bitPosInWord;
          remaining &= remaining - 1;
          if (queryAtomIdx < queryTopology.numAtoms) {
            scratch.seedAtomList[numSeedAtoms++] =
                static_cast<std::uint8_t>(queryAtomIdx);
          }
        }
      }
      if (numSeedAtoms != seed.numAtoms) prepareOk = 0;

      for (int wordIdx = 0; wordIdx < kBondWords; ++wordIdx) {
        BondWord remaining = seed.bonds[wordIdx];
        while (remaining != 0) {
          int bitPosInWord;
          if constexpr (sizeof(BondWord) == 4) {
            bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
          } else {
            bitPosInWord =
                __ffsll(static_cast<unsigned long long>(remaining)) - 1;
          }
          const int queryBondIdx = wordIdx * kBondBitsPerWord + bitPosInWord;
          remaining &= remaining - 1;

          const std::uint32_t queryEndpoints =
              queryTopology.bondEndpoints[queryBondIdx];
          const int queryEndpointU =
              static_cast<int>(queryEndpoints >> kBondEndpointShift);
          const int queryEndpointV =
              static_cast<int>(queryEndpoints & kBondEndpointMask);
          ++scratch.seedDegree[queryEndpointU];
          ++scratch.seedDegree[queryEndpointV];
        }
      }

      int seedNeighborEntries = 0;
      for (int queryAtomIdx = 0; queryAtomIdx < maxAtoms; ++queryAtomIdx) {
        scratch.seedNeighborOffset[queryAtomIdx] =
            static_cast<std::uint16_t>(seedNeighborEntries);
        seedNeighborEntries += scratch.seedDegree[queryAtomIdx];
        scratch.mappedSeedNeighborCount[queryAtomIdx] = 0;
      }
      scratch.seedNeighborOffset[maxAtoms] =
          static_cast<std::uint16_t>(seedNeighborEntries);
      if (seedNeighborEntries > 2 * maxBonds) {
        prepareOk = 0;
      } else {
        for (int wordIdx = 0; wordIdx < kBondWords; ++wordIdx) {
          BondWord remaining = seed.bonds[wordIdx];
          while (remaining != 0) {
            int bitPosInWord;
            if constexpr (sizeof(BondWord) == 4) {
              bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
            } else {
              bitPosInWord =
                  __ffsll(static_cast<unsigned long long>(remaining)) - 1;
            }
            const int queryBondIdx =
                wordIdx * kBondBitsPerWord + bitPosInWord;
            remaining &= remaining - 1;

            const std::uint32_t queryEndpoints =
                queryTopology.bondEndpoints[queryBondIdx];
            const int queryEndpointU =
                static_cast<int>(queryEndpoints >> kBondEndpointShift);
            const int queryEndpointV =
                static_cast<int>(queryEndpoints & kBondEndpointMask);
            const int slotU =
                scratch.seedNeighborOffset[queryEndpointU] +
                scratch.mappedSeedNeighborCount[queryEndpointU]++;
            scratch.seedNeighborAtom[slotU] =
                static_cast<std::uint8_t>(queryEndpointV);
            scratch.seedNeighborBond[slotU] =
                static_cast<std::uint16_t>(queryBondIdx);
            const int slotV =
                scratch.seedNeighborOffset[queryEndpointV] +
                scratch.mappedSeedNeighborCount[queryEndpointV]++;
            scratch.seedNeighborAtom[slotV] =
                static_cast<std::uint8_t>(queryEndpointU);
            scratch.seedNeighborBond[slotV] =
                static_cast<std::uint16_t>(queryBondIdx);
          }
        }
        for (int queryAtomIdx = 0; queryAtomIdx < maxAtoms; ++queryAtomIdx) {
          scratch.mappedSeedNeighborCount[queryAtomIdx] = 0;
        }
      }
    }
  }
  prepareOk = group.shfl(prepareOk, 0);
  numSeedAtoms = group.shfl(numSeedAtoms, 0);
  group.sync();
  if (!prepareOk) return false;

  for (int orderPos = 0; orderPos < numSeedAtoms; ++orderPos) {
    int bestAtom = -1;
    int bestListIdx = -1;
    int bestMappedNeighborCount = -1;
    int bestDegree = -1;
    int bestCandidateCount = maxTA + 1;
    const int remainingSeedAtomCount = numSeedAtoms - orderPos;

    for (int atomListIdx = 0; atomListIdx < remainingSeedAtomCount;
         ++atomListIdx) {
      int queryAtomIdx = -1;
      int mappedNeighborCount = 0;
      int shouldCount = 0;

      if (laneRank == 0) {
        queryAtomIdx = scratch.seedAtomList[atomListIdx];
        mappedNeighborCount =
            scratch.mappedSeedNeighborCount[queryAtomIdx];
        shouldCount = orderPos == 0 || mappedNeighborCount != 0;
      }
      queryAtomIdx = group.shfl(queryAtomIdx, 0);
      mappedNeighborCount = group.shfl(mappedNeighborCount, 0);
      shouldCount = group.shfl(shouldCount, 0);
      if (!shouldCount) continue;

      int cachedCandidateCount = kUnmappedTargetIdx;
      if (laneRank == 0) {
        // Reused only during ordering; mappings are written after a match is found.
        cachedCandidateCount = scratch.targetAtomForQuery[queryAtomIdx];
      }
      cachedCandidateCount = group.shfl(cachedCandidateCount, 0);
      int candidateCount = cachedCandidateCount;
      if (cachedCandidateCount == kUnmappedTargetIdx) {
        candidateCount = countCandidateTargetAtomsCooperative(
            group, queryAtomIdx, targetTopology, tables, scratch);
        if (laneRank == 0) {
          scratch.targetAtomForQuery[queryAtomIdx] =
              static_cast<std::uint8_t>(candidateCount);
        }
      }
      if (laneRank == 0) {
        if (candidateCount == 0) {
          prepareOk = 0;
        } else {
          const int degree = scratch.seedDegree[queryAtomIdx];
          const bool better =
              bestAtom < 0 ||
              mappedNeighborCount > bestMappedNeighborCount ||
              (mappedNeighborCount == bestMappedNeighborCount &&
               degree > bestDegree) ||
              (mappedNeighborCount == bestMappedNeighborCount &&
               degree == bestDegree &&
               candidateCount < bestCandidateCount) ||
              (mappedNeighborCount == bestMappedNeighborCount &&
               degree == bestDegree &&
               candidateCount == bestCandidateCount &&
               queryAtomIdx < bestAtom);
          if (better) {
            bestAtom = queryAtomIdx;
            bestListIdx = atomListIdx;
            bestMappedNeighborCount = mappedNeighborCount;
            bestDegree = degree;
            bestCandidateCount = candidateCount;
          }
        }
      }
      prepareOk = group.shfl(prepareOk, 0);
      if (!prepareOk) break;
    }
    if (!prepareOk) break;

    if (laneRank == 0 && bestAtom < 0 && remainingSeedAtomCount > 0) {
      bestAtom = scratch.seedAtomList[0];
      bestListIdx = 0;
    }
    if (laneRank == 0) {
      if (bestAtom < 0) {
        prepareOk = 0;
      } else {
        scratch.seedAtomList[bestListIdx] =
            scratch.seedAtomList[remainingSeedAtomCount - 1];
        scratch.seedAtoms[orderPos] = static_cast<std::uint8_t>(bestAtom);
        scratch.orderedQueryAtom[bestAtom] = 1;
        scratch.queryOrderPos[bestAtom] = static_cast<std::uint8_t>(orderPos);
        incrementMappedSeedNeighborCountsWithinThread(scratch, bestAtom);
      }
    }
    prepareOk = group.shfl(prepareOk, 0);
    if (!prepareOk) break;
  }
  return prepareOk != 0;
}

__device__ __forceinline__ bool partialUsesTargetAtomWithinThread(
    const std::uint8_t* partial,
    const int depth,
    const int targetAtomIdx) {
  for (int i = 0; i < depth; ++i) {
    if (partial[i] == targetAtomIdx) return true;
  }
  return false;
}

template<int maxAtoms, int maxBonds, int maxTA>
__device__ __forceinline__ bool findMappedQueryNeighborWithinThread(
    const FmcsSubstructureScratch<maxAtoms, maxBonds, maxTA>& scratch,
    const int depth,
    const int queryAtomIdx,
    int& outNeighborOrderPos) {
  outNeighborOrderPos = -1;
  const int begin = static_cast<int>(scratch.seedNeighborOffset[queryAtomIdx]);
  const int end =
      static_cast<int>(scratch.seedNeighborOffset[queryAtomIdx + 1]);
  for (int adjIdx = begin; adjIdx < end; ++adjIdx) {
    const int otherQueryAtom = scratch.seedNeighborAtom[adjIdx];
    const int otherOrderPos = scratch.queryOrderPos[otherQueryAtom];
    if (otherOrderPos != kUnmappedTargetIdx && otherOrderPos < depth) {
      outNeighborOrderPos = otherOrderPos;
      return true;
    }
  }
  return false;
}

template<int maxAtoms, int maxBonds, int maxTA, class TargetTopology>
__device__ __forceinline__ bool substructurePartialEdgeConsistentWithinThread(
    const TargetTopology& targetTopology,
    const PairMatchTablesDevice& tables,
    const FmcsSubstructureScratch<maxAtoms, maxBonds, maxTA>& scratch,
    const std::uint8_t* partial,
    const int depth,
    const int queryAtomIdx,
    const int targetAtomIdx) {
  const int begin = static_cast<int>(scratch.seedNeighborOffset[queryAtomIdx]);
  const int end =
      static_cast<int>(scratch.seedNeighborOffset[queryAtomIdx + 1]);
  for (int adjIdx = begin; adjIdx < end; ++adjIdx) {
    const int otherQueryAtom = scratch.seedNeighborAtom[adjIdx];
    const int otherOrderPos = scratch.queryOrderPos[otherQueryAtom];
    if (otherOrderPos == kUnmappedTargetIdx || otherOrderPos >= depth) {
      continue;
    }
    const int otherTargetAtom = partial[otherOrderPos];
    const int queryBondIdx = scratch.seedNeighborBond[adjIdx];

    int targetBondIdx = -1;
    if (!findTargetBondBetweenAtomsWithinThread(
            targetAtomIdx, otherTargetAtom, queryBondIdx,
            targetTopology, tables, targetBondIdx)) {
      return false;
    }
  }
  return true;
}

template<int maxAtoms, int maxBonds, int maxTA, class TargetTopology>
__device__ __forceinline__ void tryCommitFinalSubstructurePartialWithinThread(
    const TargetTopology& targetTopology,
    const PairMatchTablesDevice& tables,
    FmcsSubstructureScratch<maxAtoms, maxBonds, maxTA>& scratch,
    const std::uint8_t* partial,
    const int depth,
    const int queryAtomIdx,
    const int targetAtomIdx) {
  if (scratch.targetDegree[targetAtomIdx] < scratch.seedDegree[queryAtomIdx]) {
    return;
  }
  if (!tables.atoms.testBit(queryAtomIdx, targetAtomIdx)) return;
  if (partialUsesTargetAtomWithinThread(partial, depth, targetAtomIdx)) {
    return;
  }
  if (!substructurePartialEdgeConsistentWithinThread(
          targetTopology, tables, scratch, partial, depth, queryAtomIdx,
          targetAtomIdx)) {
    return;
  }

  if (atomicCAS(&scratch.found, 0, 1) == 0) {
    for (int orderPos = 0; orderPos < depth; ++orderPos) {
      const int mappedQueryAtom = scratch.seedAtoms[orderPos];
      scratch.targetAtomForQuery[mappedQueryAtom] = partial[orderPos];
    }
    scratch.targetAtomForQuery[queryAtomIdx] =
        static_cast<std::uint8_t>(targetAtomIdx);
  }
}

template<int maxAtoms, int maxBonds, int maxTA, class TargetTopology>
__device__ __forceinline__ void tryAppendSubstructurePartialWithinThread(
    const TargetTopology& targetTopology,
    const PairMatchTablesDevice& tables,
    FmcsSubstructureScratch<maxAtoms, maxBonds, maxTA>& scratch,
    const std::uint8_t* partial,
    const int depth,
    const int queryAtomIdx,
    const int targetAtomIdx,
    const int stride,
    std::uint8_t* nextPartials,
    const int effectiveCapacity) {
  if (scratch.targetDegree[targetAtomIdx] < scratch.seedDegree[queryAtomIdx]) {
    return;
  }
  if (!tables.atoms.testBit(queryAtomIdx, targetAtomIdx)) return;
  if (partialUsesTargetAtomWithinThread(partial, depth, targetAtomIdx)) {
    return;
  }
  if (!substructurePartialEdgeConsistentWithinThread(
          targetTopology, tables, scratch, partial, depth, queryAtomIdx,
          targetAtomIdx)) {
    return;
  }

  const int slot = reserveSharedCounterSlotWarpAggregated(
      &scratch.nextCount, &scratch.overflowed, effectiveCapacity);
  if (slot >= 0) {
    std::uint8_t* next = nextPartials + slot * stride;
    for (int orderPos = 0; orderPos < depth; ++orderPos) {
      next[orderPos] = partial[orderPos];
    }
    next[depth] = static_cast<std::uint8_t>(targetAtomIdx);
  }
}

// Leaf dispatch for one (partial, targetAtomIdx) candidate: at the final depth
// a consistent candidate commits the full mapping, otherwise it is appended to
// the next-depth partial buffer.
template<bool Final, int maxAtoms, int maxBonds, int maxTA, class TargetTopology>
__device__ __forceinline__ void trySubstructureCandidateWithinThread(
    const TargetTopology& targetTopology,
    const PairMatchTablesDevice& tables,
    FmcsSubstructureScratch<maxAtoms, maxBonds, maxTA>& scratch,
    const std::uint8_t* partial,
    const int depth,
    const int queryAtomIdx,
    const int targetAtomIdx,
    const int stride,
    std::uint8_t* nextPartials,
    const int effectiveCapacity) {
  if constexpr (Final) {
    tryCommitFinalSubstructurePartialWithinThread(
        targetTopology, tables, scratch, partial, depth, queryAtomIdx,
        targetAtomIdx);
  } else {
    tryAppendSubstructurePartialWithinThread(
        targetTopology, tables, scratch, partial, depth, queryAtomIdx,
        targetAtomIdx, stride, nextPartials, effectiveCapacity);
  }
}

// Expands every partial at one depth using the neighbor-order decision hoisted
// by lane 0 (hoistedNeighborOrderPos; kUnmappedTargetIdx means "no mapped
// neighbor / no adjacency" and forces the full target scan).
//
// CONCURRENCY: this helper contains NO collective operations, and none may be
// added. Every loop bound here is data-dependent and differs across lanes
// (numPartials, targetScanEnd, the shared scratch.found early-exit, the subwarp
// `continue`), so any group.sync/shfl/ballot inside would be a partial-group
// barrier deadlock. The caller's group.sync() that publishes the hoisted
// decision must stay *before* this call. See
// analysis/fmcs_duplicated_match_impl.md.
template<bool Final, int maxAtoms, int maxBonds, int maxTA, class TargetTopology>
__device__ __forceinline__ void expandSubstructurePartialsHoistedCooperative(
    const int laneRank,
    const int laneCount,
    const TargetTopology& targetTopology,
    const PairMatchTablesDevice& tables,
    FmcsSubstructureScratch<maxAtoms, maxBonds, maxTA>& scratch,
    const std::uint8_t* currentPartials,
    const int numPartials,
    const int stride,
    const int depth,
    const int queryAtomIdx,
    const int hoistedNeighborOrderPos,
    std::uint8_t* nextPartials,
    const int effectiveCapacity) {
  if (hoistedNeighborOrderPos != kUnmappedTargetIdx) {
    constexpr int subwarpSize = kFallbackAdjacencySubwarpSize;
    const int subwarpRank = laneRank / subwarpSize;
    const int sublaneRank = laneRank - subwarpRank * subwarpSize;
    const int subwarpCount = laneCount / subwarpSize;
    for (int partialBase = 0;
         partialBase < numPartials && scratch.found == 0;
         partialBase += subwarpCount) {
      const int partialIdx = partialBase + subwarpRank;
      if (partialIdx >= numPartials) continue;
      const std::uint8_t* partial = currentPartials + partialIdx * stride;
      const int mappedTargetAtom = partial[hoistedNeighborOrderPos];
      const int targetScanBegin =
          static_cast<int>(targetTopology.rowOffsets[mappedTargetAtom]);
      const int targetScanEnd =
          static_cast<int>(targetTopology.rowOffsets[mappedTargetAtom + 1]);
      for (int targetScanIdx = targetScanBegin + sublaneRank;
           targetScanIdx < targetScanEnd && scratch.found == 0;
           targetScanIdx += subwarpSize) {
        const int targetAtomIdx =
            static_cast<int>(targetTopology.colIndices[targetScanIdx]);
        trySubstructureCandidateWithinThread<Final>(
            targetTopology, tables, scratch, partial, depth, queryAtomIdx,
            targetAtomIdx, stride, nextPartials, effectiveCapacity);
      }
    }
  } else {
    for (int partialIdx = 0;
         partialIdx < numPartials && scratch.found == 0;
         ++partialIdx) {
      const std::uint8_t* partial = currentPartials + partialIdx * stride;
      for (int targetAtomIdx = laneRank;
           targetAtomIdx < targetTopology.numAtoms && scratch.found == 0;
           targetAtomIdx += laneCount) {
        trySubstructureCandidateWithinThread<Final>(
            targetTopology, tables, scratch, partial, depth, queryAtomIdx,
            targetAtomIdx, stride, nextPartials, effectiveCapacity);
      }
    }
  }
}

// Non-hoisted (stats/measure builds) counterpart: every lane recomputes the
// neighbor-order / adjacency decision per partial, inline.
//
// CONCURRENCY: same rule as the hoisted expansion above — NO collective
// operations in here; the loop bounds diverge across lanes.
template<bool Final, int maxAtoms, int maxBonds, int maxTA, class TargetTopology>
__device__ __forceinline__ void expandSubstructurePartialsRecomputeCooperative(
    const int laneRank,
    const int laneCount,
    const TargetTopology& targetTopology,
    const PairMatchTablesDevice& tables,
    FmcsSubstructureScratch<maxAtoms, maxBonds, maxTA>& scratch,
    const std::uint8_t* currentPartials,
    const int numPartials,
    const int stride,
    const int depth,
    const int queryAtomIdx,
    std::uint8_t* nextPartials,
    const int effectiveCapacity) {
  for (int partialIdx = 0;
       partialIdx < numPartials && scratch.found == 0;
       ++partialIdx) {
    const std::uint8_t* partial = currentPartials + partialIdx * stride;
    int neighborOrderPos = -1;
    const bool hasMappedNeighbor =
        findMappedQueryNeighborWithinThread(
            scratch, depth, queryAtomIdx, neighborOrderPos);
    const bool scanAdjacency =
        hasMappedNeighbor &&
        targetTopology.rowOffsets != nullptr &&
        targetTopology.colIndices != nullptr;
    const int targetScanBegin =
        scanAdjacency ? static_cast<int>(
                            targetTopology.rowOffsets[partial[neighborOrderPos]])
                      : 0;
    const int targetScanEnd =
        scanAdjacency ? static_cast<int>(
                            targetTopology.rowOffsets[partial[neighborOrderPos] + 1])
                      : targetTopology.numAtoms;

    for (int targetScanIdx = targetScanBegin + laneRank;
         targetScanIdx < targetScanEnd && scratch.found == 0;
         targetScanIdx += laneCount) {
      const int targetAtomIdx =
          scanAdjacency
              ? static_cast<int>(targetTopology.colIndices[targetScanIdx])
              : targetScanIdx;
      trySubstructureCandidateWithinThread<Final>(
          targetTopology, tables, scratch, partial, depth, queryAtomIdx,
          targetAtomIdx, stride, nextPartials, effectiveCapacity);
    }
  }
}

template<bool HoistNeighborOrder = true,
         int maxAtoms, int maxBonds, int maxTA, int maxTB,
         class QueryTopology, class TargetTopology, class GroupT,
         class OverflowFlagT>
__device__ __forceinline__ bool matchSeedSubstructureCooperative(
    const GroupT& group,
    const Seed<maxAtoms, maxBonds>& seed,
    const QueryTopology& queryTopology,
    const TargetTopology& targetTopology,
    const PairMatchTablesDevice& tables,
    MatchResult<maxAtoms, maxBonds, maxTA, maxTB>& match,
    FmcsSubstructureScratch<maxAtoms, maxBonds, maxTA>& scratch,
    std::uint8_t* partialStorage,
    int partialCapacity,
    OverflowFlagT* overflowedFlag) {
  const int laneRank  = static_cast<int>(group.thread_rank());
  const int laneCount = static_cast<int>(group.num_threads());

  int numSeedAtoms = 0;
  int prepared = 0;
  int prepareOk = 1;
  if (laneRank == 0) {
    matchResultClearWithinThread(match);
  }
  if (seed.numAtoms != 0) {
    initializeSeedSubstructureScratchCooperative(
        group, targetTopology, scratch);
    prepareOk = prepareSeedSubstructureSearchCooperative(
        group, seed, queryTopology, targetTopology, tables, scratch,
        numSeedAtoms)
        ? 1
        : 0;
  }
  if (laneRank == 0) {
    if (seed.numAtoms == 0) {
      prepared = (seed.numBonds == 0) ? 2 : -1;
    } else {
      prepared = prepareOk ? 1 : -1;
    }
    if ((partialStorage == nullptr || partialCapacity <= 0) && prepared == 1) {
      scratch.overflowed = 1;
      prepared = -1;
    }
  }
  group.sync();
  prepared = group.shfl(prepared, 0);
  numSeedAtoms = group.shfl(numSeedAtoms, 0);
  if (prepared == 2) return true;
  if (prepared < 0) {
    if (laneRank == 0 && scratch.overflowed && overflowedFlag != nullptr) {
      setOverflowedFlagWithinThread(overflowedFlag);
    }
    return false;
  }

  const int stride = numSeedAtoms;
  const int halfBytes = partialCapacity * maxAtoms;
  const int effectiveCapacity = halfBytes / stride;
  std::uint8_t* currentPartials = partialStorage;
  std::uint8_t* nextPartials = partialStorage + halfBytes;

  const int firstQueryAtom = scratch.seedAtoms[0];
  for (int targetAtomIdx = laneRank;
       targetAtomIdx < targetTopology.numAtoms;
       targetAtomIdx += laneCount) {
    if (scratch.targetDegree[targetAtomIdx] < scratch.seedDegree[firstQueryAtom]) {
      continue;
    }
    if (!tables.atoms.testBit(firstQueryAtom, targetAtomIdx)) continue;

    const int slot = reserveSharedCounterSlotWarpAggregated(
        &scratch.currentCount, &scratch.overflowed, effectiveCapacity);
    if (slot >= 0) {
      currentPartials[slot * stride] = static_cast<std::uint8_t>(targetAtomIdx);
    }
  }
  group.sync();

  if (numSeedAtoms == 1) {
    if (scratch.currentCount > 0) {
      if (laneRank == 0) {
        scratch.targetAtomForQuery[firstQueryAtom] = currentPartials[0];
        prepared = rebuildMatchFromSubstructureMappingWithinThread(
            seed, queryTopology, targetTopology, tables, match, scratch) ? 1 : -1;
      }
      group.sync();
      prepared = group.shfl(prepared, 0);
      return prepared == 1;
    }
    return false;
  }

  for (int depth = 1; depth < numSeedAtoms; ++depth) {
    if (scratch.currentCount == 0 || scratch.found != 0) break;
    if (laneRank == 0) scratch.nextCount = 0;
    group.sync();

    const int queryAtomIdx = scratch.seedAtoms[depth];
    const int numPartials =
        scratch.currentCount < effectiveCapacity
            ? scratch.currentCount
            : effectiveCapacity;
    const bool finalDepth = depth == numSeedAtoms - 1;
    if constexpr (HoistNeighborOrder) {
      if (laneRank == 0) {
        int neighborOrderPos = -1;
        const bool hasMappedNeighbor =
            findMappedQueryNeighborWithinThread(
                scratch, depth, queryAtomIdx, neighborOrderPos);
        scratch.orderedQueryAtom[queryAtomIdx] =
            hasMappedNeighbor &&
                    targetTopology.rowOffsets != nullptr &&
                    targetTopology.colIndices != nullptr
                ? static_cast<std::uint8_t>(neighborOrderPos)
                : kUnmappedTargetIdx;
      }
      group.sync();
    }

    // The expansion helpers below contain no collective operations; the only
    // collective op per depth is the hoist-publishing group.sync() above, which
    // every lane reaches before the divergent per-partial loops. Keep it that
    // way — see analysis/fmcs_duplicated_match_impl.md.
    if constexpr (HoistNeighborOrder) {
      const int hoistedNeighborOrderPos =
          static_cast<int>(scratch.orderedQueryAtom[queryAtomIdx]);
      if (finalDepth) {
        expandSubstructurePartialsHoistedCooperative<true>(
            laneRank, laneCount, targetTopology, tables, scratch,
            currentPartials, numPartials, stride, depth, queryAtomIdx,
            hoistedNeighborOrderPos, nextPartials, effectiveCapacity);
      } else {
        expandSubstructurePartialsHoistedCooperative<false>(
            laneRank, laneCount, targetTopology, tables, scratch,
            currentPartials, numPartials, stride, depth, queryAtomIdx,
            hoistedNeighborOrderPos, nextPartials, effectiveCapacity);
      }
    } else {
      if (finalDepth) {
        expandSubstructurePartialsRecomputeCooperative<true>(
            laneRank, laneCount, targetTopology, tables, scratch,
            currentPartials, numPartials, stride, depth, queryAtomIdx,
            nextPartials, effectiveCapacity);
      } else {
        expandSubstructurePartialsRecomputeCooperative<false>(
            laneRank, laneCount, targetTopology, tables, scratch,
            currentPartials, numPartials, stride, depth, queryAtomIdx,
            nextPartials, effectiveCapacity);
      }
    }
    group.sync();

    if (scratch.found != 0) break;
    if (finalDepth) break;
    if (laneRank == 0) scratch.currentCount = scratch.nextCount;
    std::uint8_t* tmp = currentPartials;
    currentPartials = nextPartials;
    nextPartials = tmp;
    group.sync();
  }

  int found = scratch.found;
  if (found != 0) {
    if (laneRank == 0) {
      found = rebuildMatchFromSubstructureMappingWithinThread(
          seed, queryTopology, targetTopology, tables, match, scratch) ? 1 : 0;
    }
    group.sync();
    found = group.shfl(found, 0);
    return found != 0;
  }

  if (laneRank == 0) {
    matchResultClearWithinThread(match);
    if (scratch.overflowed && overflowedFlag != nullptr) {
      setOverflowedFlagWithinThread(overflowedFlag);
    }
  }
  group.sync();
  return false;
}

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_MATCH_CUH
