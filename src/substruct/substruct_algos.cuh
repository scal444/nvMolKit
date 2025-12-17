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

#ifndef NVMOLKIT_SUBSTRUCT_ALGOS_CUH
#define NVMOLKIT_SUBSTRUCT_ALGOS_CUH

#include <cooperative_groups.h>
#include <cstdint>

#include "flat_bit_vect.h"
#include "molecules_device.cuh"
#include "substruct_types.h"

namespace nvMolKit {

// =============================================================================
// Shared Constants
// =============================================================================

constexpr int  kWarpSize      = 32;
constexpr bool kDebugWUS      = false;  ///< Enable debug output in warpUnifiedSearchGPU

// =============================================================================
// Helper function for checking if target atom is used in mapping
// =============================================================================

/**
 * @brief Check if a target atom is already used in a partial mapping.
 *
 * Iterates through the mapping array to check if targetAtom appears.
 * This is O(numQueryAtoms) but avoids the need for a separate bitmask,
 * saving significant shared memory.
 *
 * @param mapping The partial mapping array (mapping[q] = target atom, -1 if unassigned)
 * @param numQueryAtoms Number of query atoms
 * @param targetAtom Target atom to check
 * @return true if targetAtom is already used in the mapping
 */
__device__ __forceinline__ bool isTargetUsedInMapping(const int8_t* mapping, int numQueryAtoms, int targetAtom) {
  for (int q = 0; q < numQueryAtoms; ++q) {
    if (mapping[q] == targetAtom) {
      return true;
    }
  }
  return false;
}

// =============================================================================
// VF2 Data Structures
// =============================================================================

/**
 * @brief State for VF2 iterative search (per-warp in shared memory).
 *
 * Maintains partial match and exploration stack for DFS backtracking.
 */
struct VF2State {
  int8_t mapping[kMaxQueryAtoms];       ///< mapping[q] = target atom idx, -1 if unassigned
  int8_t candidateIdx[kMaxQueryAtoms];  ///< Current candidate index at each stack level
  int    depth;                         ///< Current recursion depth (0 to numQueryAtoms-1)
  int    matchCount;                    ///< Number of complete matches found
  int    numQueryAtoms;                 ///< Cached for isTargetUsed check

  __device__ __forceinline__ void init(int nQueryAtoms) {
    for (int i = 0; i < kMaxQueryAtoms; ++i) {
      mapping[i]      = -1;
      candidateIdx[i] = 0;
    }
    depth         = 0;
    matchCount    = 0;
    numQueryAtoms = nQueryAtoms;
  }

  __device__ __forceinline__ bool isTargetUsed(int targetIdx) const {
    return isTargetUsedInMapping(mapping, numQueryAtoms, targetIdx);
  }
};

// =============================================================================
// GSI/BFS Data Structures
// =============================================================================

/**
 * @brief Partial match for BFS-style algorithms.
 *
 * Represents a partial mapping from query atoms to target atoms.
 * Stored compactly for queue-based BFS exploration.
 */
struct PartialMatch {
  int8_t mapping[kMaxQueryAtoms];  ///< mapping[q] = target atom, -1 if unassigned
  int8_t nextQueryAtom;            ///< Next query atom to extend

  __device__ __forceinline__ void init() {
    for (int i = 0; i < kMaxQueryAtoms; ++i) {
      mapping[i] = -1;
    }
    nextQueryAtom = 0;
  }
};

/**
 * @brief Candidate list for a query atom.
 *
 * Precomputed from label matrix for efficient iteration.
 */
struct CandidateList {
  int8_t candidates[kMaxTargetAtoms];  ///< Target atoms that can match this query atom
  int    count;                        ///< Number of valid candidates
};

// =============================================================================
// Edge Consistency Checking
// =============================================================================

/**
 * @brief Check if a query bond type matches a target bond type.
 *
 * Handles "any bond" (type 0) which matches any target bond type.
 *
 * @param queryBondType Query bond type (0 = any, 1 = single, 2 = double, etc.)
 * @param targetBondType Target bond type
 * @return true if bond types are compatible
 */
__device__ __forceinline__ bool bondTypeMatches(int queryBondType, int targetBondType) {
  // Query bond type 0 (UNSPECIFIED) means "any bond" - matches everything
  if (queryBondType == 0) {
    return true;
  }
  return queryBondType == targetBondType;
}

/**
 * @brief Check if ring bond constraints are satisfied.
 *
 * @param queryFlags Bond query flags (BondQueryIsRingBond, BondQueryNotRingBond)
 * @param targetIsInRing Whether the target bond is in a ring
 * @return true if ring constraints are satisfied
 */
__device__ __forceinline__ bool ringBondConstraintsSatisfied(uint8_t queryFlags, bool targetIsInRing) {
  // Check ring bond constraint
  if (queryFlags & BondQueryIsRingBond) {
    if (!targetIsInRing) {
      return false;
    }
  }
  if (queryFlags & BondQueryNotRingBond) {
    if (targetIsInRing) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Check if extending partial match with (queryAtom -> targetAtom) is edge-consistent.
 *
 * For each already-matched neighbor of queryAtom in the query graph, verify that
 * the corresponding edge exists in the target graph between targetAtom and the
 * mapped neighbor.
 *
 * @param target Target molecule view
 * @param query Query molecule view
 * @param mapping Current partial mapping (query -> target)
 * @param queryAtom Query atom being matched
 * @param targetAtom Candidate target atom
 * @return true if edge consistency is satisfied
 */
__device__ __forceinline__ bool checkEdgeConsistency(const MoleculeView& target,
                                                     const MoleculeView& query,
                                                     const int8_t*       mapping,
                                                     int                 queryAtom,
                                                     int                 targetAtom) {
  const int  queryDegree      = query.getAtomDegree(queryAtom);
  const bool hasBondQueryData = query.hasBondQueryData();

  for (int i = 0; i < queryDegree; ++i) {
    const int neighborQueryAtom = query.getNeighborAtomIdx(queryAtom, i);

    // Only check neighbors that are already mapped
    if (mapping[neighborQueryAtom] < 0) {
      continue;
    }

    const int neighborTargetAtom = mapping[neighborQueryAtom];
    const int queryBondIdx       = query.getNeighborBondIdx(queryAtom, i);

    // Get query bond info
    int     queryBondType  = query.getBond(queryBondIdx, threadIdx.x, blockIdx.x).bondType;
    uint8_t queryBondFlags = 0;
    if (hasBondQueryData) {
      const BondQueryData& bqd = query.getBondQuery(queryBondIdx);
      queryBondType  = bqd.bondType;
      queryBondFlags = bqd.queryFlags;
    }

    // Check if targetAtom has an edge to neighborTargetAtom with compatible bond
    bool      foundEdge    = false;
    const int targetDegree = target.getAtomDegree(targetAtom);

    for (int j = 0; j < targetDegree; ++j) {
      if (target.getNeighborAtomIdx(targetAtom, j) == neighborTargetAtom) {
        const int       targetBondIdx = target.getNeighborBondIdx(targetAtom, j);
        const BondData& targetBond    = target.getBond(targetBondIdx, threadIdx.x, blockIdx.x);

        // Check bond type compatibility
        if (queryBondFlags & BondQuerySingleOrAromatic) {
          // SingleOrAromaticBond: only match single (1) or aromatic (7, 12) bonds
          const int tbt = targetBond.bondType;
          if (tbt != 1 && tbt != 7 && tbt != 12) {
            continue;
          }
        } else if (!bondTypeMatches(queryBondType, targetBond.bondType)) {
          continue;
        }

        // Check ring bond constraints if present
        if (queryBondFlags & (BondQueryIsRingBond | BondQueryNotRingBond)) {
          bool targetIsInRing = (targetBond.isInRing != 0);
          if (!ringBondConstraintsSatisfied(queryBondFlags, targetIsInRing)) {
            continue;
          }
        }

        foundEdge = true;
        break;
      }
    }

    if (!foundEdge) {
      return false;
    }
  }

  return true;
}

// =============================================================================
// VF2 Algorithm Implementation
// =============================================================================

/**
 * @brief VF2 iterative DFS search for subgraph isomorphism.
 *
 * Each warp explores from a different starting target atom for query atom 0.
 * Uses explicit stack to avoid recursion and maintain uniform control flow.
 *
 * @tparam MaxTargetAtoms Maximum target atoms (for label matrix sizing)
 * @tparam MaxQueryAtoms Maximum query atoms (for label matrix sizing)
 * @param target Target molecule view
 * @param query Query molecule view
 * @param labelMatrix Precomputed label compatibility matrix
 * @param state VF2 state in shared memory (per warp)
 * @param startingTargetAtom Starting target atom for this warp (query atom 0)
 * @param matchCount Output: number of matches found
 * @param reportedCount Output: number of matches written (capped)
 * @param matchIndices Output: match index buffer
 * @param maxMatches Maximum matches to write
 * @param matchOffset Offset into matchIndices for this pair
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ void vf2SearchGPU(const MoleculeView&                                   target,
                             const MoleculeView&                                   query,
                             const BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>& labelMatrix,
                             VF2State&                                             state,
                             int                                                   startingTargetAtom,
                             int*                                                  matchCount,
                             int*                                                  reportedCount,
                             int16_t*                                              matchIndices,
                             int                                                   maxMatches,
                             int                                                   matchOffset) {
  namespace cg = cooperative_groups;
  auto tile32  = cg::tiled_partition<32>(cg::this_thread_block());
  const int laneId = tile32.thread_rank();

  const int numQueryAtoms  = query.numAtoms;
  const int numTargetAtoms = target.numAtoms;

  // Only lane 0 does the DFS logic; other lanes assist with parallel candidate evaluation
  if (laneId != 0) {
    return;
  }

  // Check if starting atom is valid
  if (startingTargetAtom >= numTargetAtoms || !labelMatrix.get(startingTargetAtom, 0)) {
    return;
  }

  state.init(numQueryAtoms);
  state.mapping[0] = static_cast<int8_t>(startingTargetAtom);
  state.depth = 1;

  // Iterative DFS
  while (state.depth > 0) {
    if (state.depth == numQueryAtoms) {
      // Complete match found
      const int currentMatchCount = atomicAdd(matchCount, 1);

      if (currentMatchCount < maxMatches) {
        // Write match to output
        const int writeOffset = matchOffset + currentMatchCount * numQueryAtoms;
        for (int q = 0; q < numQueryAtoms; ++q) {
          matchIndices[writeOffset + q] = state.mapping[q];
        }
        atomicAdd(reportedCount, 1);
      }

      // Backtrack to find more matches
      --state.depth;
      if (state.depth > 0) {
        state.mapping[state.depth] = -1;
        ++state.candidateIdx[state.depth];
      }
      continue;
    }

    const int currentQueryAtom = state.depth;
    int8_t&   candIdx          = state.candidateIdx[state.depth];

    // Find next valid candidate
    bool foundCandidate = false;
    while (candIdx < numTargetAtoms) {
      const int candidateTarget = candIdx;

      // Check feasibility
      const bool labelOk    = labelMatrix.get(candidateTarget, currentQueryAtom);
      const bool notUsed    = !state.isTargetUsed(candidateTarget);
      const bool edgeOk     = labelOk && notUsed && 
                              checkEdgeConsistency(target, query, state.mapping, currentQueryAtom, candidateTarget);

      if (edgeOk) {
        // Extend match
        state.mapping[currentQueryAtom] = static_cast<int8_t>(candidateTarget);
        state.candidateIdx[state.depth + 1] = 0;
        ++state.depth;
        foundCandidate = true;
        break;
      }

      ++candIdx;
    }

    if (!foundCandidate) {
      // Backtrack
      state.candidateIdx[state.depth] = 0;
      --state.depth;
      if (state.depth > 0) {
        state.mapping[state.depth] = -1;
        ++state.candidateIdx[state.depth];
      } else if (state.depth == 0) {
        // Exhausted this starting point
        break;
      }
    }
  }
}

// =============================================================================
// GSI BFS Algorithm Implementation
// =============================================================================

/**
 * @brief GSI-inspired BFS level-by-level search.
 *
 * Processes query atoms in order, extending all partial matches at each level.
 * Uses preallocation strategy to avoid two-step output scheme.
 *
 * @tparam MaxTargetAtoms Maximum target atoms
 * @tparam MaxQueryAtoms Maximum query atoms
 * @param target Target molecule view
 * @param query Query molecule view
 * @param labelMatrix Precomputed label compatibility matrix
 * @param sharedPartials Shared memory for partial matches (ping-pong buffers)
 * @param matchCount Output: number of matches found
 * @param reportedCount Output: number of matches written
 * @param matchIndices Output buffer
 * @param maxMatches Maximum matches to write
 * @param matchOffset Offset into output
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ void gsiBFSSearchGPU(const MoleculeView&                                   target,
                                const MoleculeView&                                   query,
                                const BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>& labelMatrix,
                                PartialMatch*                                         sharedPartials,
                                int                                                   maxPartials,
                                int*                                                  matchCount,
                                int*                                                  reportedCount,
                                int16_t*                                              matchIndices,
                                int                                                   maxMatches,
                                int                                                   matchOffset) {
  namespace cg = cooperative_groups;
  auto block   = cg::this_thread_block();
  auto tile32  = cg::tiled_partition<32>(block);
  
  const int tid       = block.thread_rank();
  const int laneId    = tile32.thread_rank();
  const int warpId    = tile32.meta_group_rank();
  const int numWarps  = tile32.meta_group_size();

  const int numQueryAtoms  = query.numAtoms;
  const int numTargetAtoms = target.numAtoms;

  // Use ping-pong buffers for BFS levels
  __shared__ int currentCount;
  __shared__ int nextCount;

  if (tid == 0) {
    currentCount = 0;
    nextCount    = 0;
  }
  block.sync();

  // Initialize level 0: all candidates for query atom 0
  // If query has only 1 atom, these are complete matches
  const bool singleAtomQuery = (numQueryAtoms == 1);

  for (int t = tid; t < numTargetAtoms; t += block.size()) {
    if (labelMatrix.get(t, 0)) {
      if (singleAtomQuery) {
        // Single-atom query: record as complete match
        const int matchIdx = atomicAdd(matchCount, 1);
        if (matchIdx < maxMatches) {
          matchIndices[matchOffset + matchIdx] = static_cast<int16_t>(t);
          atomicAdd(reportedCount, 1);
        }
      } else {
        // Multi-atom query: add to partial matches for BFS
        const int slot = atomicAdd(&currentCount, 1);
        if (slot < maxPartials) {
          sharedPartials[slot].init();
          sharedPartials[slot].mapping[0]    = static_cast<int8_t>(t);
          sharedPartials[slot].nextQueryAtom = 1;
        }
      }
    }
  }
  block.sync();

  // Early exit for single-atom queries
  if (singleAtomQuery) {
    return;
  }

  // BFS levels
  for (int level = 1; level < numQueryAtoms; ++level) {
    const int queryAtom   = level;
    const int numPartials = min(currentCount, maxPartials);

    if (tid == 0) {
      nextCount = 0;
    }
    block.sync();

    // Each warp processes partial matches in round-robin
    for (int pIdx = warpId; pIdx < numPartials; pIdx += numWarps) {
      const PartialMatch& partial = sharedPartials[pIdx];

      // Each lane evaluates a different target candidate
      for (int tBase = 0; tBase < numTargetAtoms; tBase += kWarpSize) {
        const int t = tBase + laneId;
        
        bool valid = false;
        if (t < numTargetAtoms) {
          const bool labelOk = labelMatrix.get(t, queryAtom);
          const bool notUsed = !isTargetUsedInMapping(partial.mapping, numQueryAtoms, t);
          valid = labelOk && notUsed && 
                  checkEdgeConsistency(target, query, partial.mapping, queryAtom, t);
        }

        // Count valid extensions using ballot
        const uint32_t validMask = __ballot_sync(0xFFFFFFFF, valid);
        const int      validCount = __popc(validMask);

        if (validCount > 0 && laneId == 0) {
          // Reserve slots for valid extensions
          // Note: simplified - full impl would use exclusive scan
        }

        // Write valid extensions
        if (valid) {
          if (level == numQueryAtoms - 1) {
            // Complete match
            const int matchIdx = atomicAdd(matchCount, 1);
            if (matchIdx < maxMatches) {
              const int writeOffset = matchOffset + matchIdx * numQueryAtoms;
              for (int q = 0; q < numQueryAtoms; ++q) {
                matchIndices[writeOffset + q] = (q == queryAtom) ? static_cast<int16_t>(t) 
                                                                  : partial.mapping[q];
              }
              atomicAdd(reportedCount, 1);
            }
          } else {
            // Add to next level
            const int slot = atomicAdd(&nextCount, 1);
            if (slot < maxPartials) {
              PartialMatch& next = sharedPartials[maxPartials + slot];
              for (int q = 0; q < numQueryAtoms; ++q) {
                next.mapping[q] = partial.mapping[q];
              }
              next.mapping[queryAtom] = static_cast<int8_t>(t);
              next.nextQueryAtom      = static_cast<int8_t>(queryAtom + 1);
            }
          }
        }
      }
    }
    block.sync();

    // Swap buffers (copy next to current)
    if (tid == 0) {
      currentCount = nextCount;
    }
    block.sync();

    // Copy next level partials to start of buffer
    const int toCopy = min(nextCount, maxPartials);
    for (int i = tid; i < toCopy; i += block.size()) {
      sharedPartials[i] = sharedPartials[maxPartials + i];
    }
    block.sync();
  }
}

// =============================================================================
// Warp-Unified Search (WUS) Algorithm Implementation
// =============================================================================

/**
 * @brief Novel warp-unified BFS search exploiting ballot operations.
 *
 * Precomputes candidate lists from label matrix, then uses warp-collective
 * operations for candidate evaluation with minimal divergence.
 *
 * @tparam MaxTargetAtoms Maximum target atoms
 * @tparam MaxQueryAtoms Maximum query atoms
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ void warpUnifiedSearchGPU(const MoleculeView&                                   target,
                                     const MoleculeView&                                   query,
                                     const BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>& labelMatrix,
                                     CandidateList*                                        sharedCandidates,
                                     PartialMatch*                                         workQueue,
                                     int                                                   maxQueueSize,
                                     int*                                                  matchCount,
                                     int*                                                  reportedCount,
                                     int16_t*                                              matchIndices,
                                     int                                                   maxMatches,
                                     int                                                   matchOffset) {
  namespace cg = cooperative_groups;
  auto block   = cg::this_thread_block();
  auto tile32  = cg::tiled_partition<32>(block);

  const int tid      = block.thread_rank();
  const int laneId   = tile32.thread_rank();
  const int warpId   = tile32.meta_group_rank();
  const int numWarps = tile32.meta_group_size();

  const int numQueryAtoms  = query.numAtoms;
  const int numTargetAtoms = target.numAtoms;

  if constexpr (kDebugWUS) {
    if (tid == 0) {
      printf("[WUS] numQueryAtoms=%d, numTargetAtoms=%d, maxQueueSize=%d, maxMatches=%d\n",
             numQueryAtoms, numTargetAtoms, maxQueueSize, maxMatches);
      printf("[WUS] query.hasBondQueryData()=%d\n", query.hasBondQueryData() ? 1 : 0);
    }
    block.sync();
  }

  // Phase 1: Precompute candidate lists from label matrix
  for (int q = tid; q < numQueryAtoms; q += block.size()) {
    CandidateList& list = sharedCandidates[q];
    list.count = 0;
    for (int t = 0; t < numTargetAtoms; ++t) {
      if (labelMatrix.get(t, q)) {
        list.candidates[list.count++] = static_cast<int8_t>(t);
      }
    }
  }
  block.sync();

  if constexpr (kDebugWUS) {
    if (tid == 0) {
      for (int q = 0; q < numQueryAtoms; ++q) {
        printf("[WUS] Phase1: query atom %d has %d candidates: ", q, sharedCandidates[q].count);
        for (int i = 0; i < sharedCandidates[q].count && i < 10; ++i) {
          printf("%d ", (int)sharedCandidates[q].candidates[i]);
        }
        if (sharedCandidates[q].count > 10) printf("...");
        printf("\n");
      }
    }
    block.sync();
  }

  // Phase 2: Initialize work queue with candidates for query atom 0
  __shared__ int queueHead;
  __shared__ int queueTail;

  if (tid == 0) {
    queueHead = 0;
    queueTail = 0;
  }
  block.sync();

  const CandidateList& q0Candidates = sharedCandidates[0];
  const bool           singleAtomQuery = (numQueryAtoms == 1);

  for (int i = tid; i < q0Candidates.count; i += block.size()) {
    if (singleAtomQuery) {
      // Single-atom query: record as complete match
      const int matchIdx = atomicAdd(matchCount, 1);
      if (matchIdx < maxMatches) {
        matchIndices[matchOffset + matchIdx] = static_cast<int16_t>(q0Candidates.candidates[i]);
        atomicAdd(reportedCount, 1);
      }
    } else {
      const int slot = atomicAdd(&queueTail, 1);
      if (slot < maxQueueSize) {
        workQueue[slot].init();
        workQueue[slot].mapping[0]    = q0Candidates.candidates[i];
        workQueue[slot].nextQueryAtom = 1;
      }
    }
  }
  block.sync();

  // Early exit for single-atom queries
  if (singleAtomQuery) {
    return;
  }

  // Phase 3: BFS processing
  // Shared variables declared outside loop to avoid race conditions
  __shared__ int workAvailable;
  __shared__ int warpWorkIdx[32];  // Max 32 warps
  __shared__ int iterCount;

  if constexpr (kDebugWUS) {
    if (tid == 0) {
      iterCount = 0;
    }
    block.sync();
  }

  __shared__ int snapshotQueueTail;  // Snapshot of queueTail at start of iteration

  while (true) {
    if (tid == 0) {
      snapshotQueueTail = queueTail;  // Snapshot BEFORE checking workAvailable
      workAvailable = (queueHead < snapshotQueueTail) ? 1 : 0;
      if constexpr (kDebugWUS) {
        ++iterCount;
      }
    }
    block.sync();

    if constexpr (kDebugWUS) {
      if (tid == 0 && iterCount <= 10) {
        printf("[WUS] Phase3 iter %d: queueHead=%d, queueTail=%d, snapshotQueueTail=%d, workAvailable=%d\n",
               iterCount, queueHead, queueTail, snapshotQueueTail, workAvailable);
      }
      block.sync();
    }

    if (!workAvailable) {
      break;
    }

    // Each warp tries to claim a work item using CAS to avoid over-incrementing queueHead
    if (laneId == 0 && warpId < 32) {
      warpWorkIdx[warpId] = -1;  // Default to invalid
      int oldHead = queueHead;
      while (oldHead < snapshotQueueTail) {
        int newHead = atomicCAS(&queueHead, oldHead, oldHead + 1);
        if (newHead == oldHead) {
          // Successfully claimed slot
          warpWorkIdx[warpId] = oldHead;
          break;
        }
        oldHead = newHead;  // Retry with updated value
      }
    }
    tile32.sync();

    const int myWorkIdx = warpWorkIdx[warpId];

    // Use hasWork flag instead of continue to ensure all threads hit sync
    const bool hasWork = (myWorkIdx >= 0 && myWorkIdx < maxQueueSize);

    // Copy work item to registers to avoid races with concurrent writes to workQueue
    int8_t localMapping[kMaxQueryAtoms];
    int    localQueryAtom = 0;
    for (int q = 0; q < kMaxQueryAtoms; ++q) {
      localMapping[q] = -1;
    }

    if (hasWork) {
      const PartialMatch& work = workQueue[myWorkIdx];
      localQueryAtom = work.nextQueryAtom;
      for (int q = 0; q < numQueryAtoms; ++q) {
        localMapping[q] = work.mapping[q];
      }
    }

    if constexpr (kDebugWUS) {
      if (warpId == 0 && laneId == 0 && iterCount <= 10) {
        printf("[WUS] Phase3 iter %d warp0: hasWork=%d, myWorkIdx=%d, localQueryAtom=%d\n",
               iterCount, hasWork ? 1 : 0, myWorkIdx, localQueryAtom);
        if (hasWork) {
          printf("[WUS]   localMapping: ");
          for (int q = 0; q < numQueryAtoms; ++q) {
            printf("%d ", (int)localMapping[q]);
          }
          printf("\n");
          printf("[WUS]   candidates for queryAtom %d: count=%d\n",
                 localQueryAtom, sharedCandidates[localQueryAtom].count);
        }
      }
    }

    // Ensure all reads from workQueue complete before any writes
    block.sync();

    if (hasWork && localQueryAtom < numQueryAtoms) {
      const CandidateList& candidates = sharedCandidates[localQueryAtom];

      // Warp-parallel candidate evaluation
      for (int cBase = 0; cBase < candidates.count; cBase += kWarpSize) {
        const int cIdx = cBase + laneId;

        bool valid      = false;
        int  targetAtom = -1;

        if (cIdx < candidates.count) {
          targetAtom         = candidates.candidates[cIdx];
          const bool notUsed = !isTargetUsedInMapping(localMapping, numQueryAtoms, targetAtom);
          const bool edgeOk  = checkEdgeConsistency(target, query, localMapping, localQueryAtom, targetAtom);
          valid = notUsed && edgeOk;

          if constexpr (kDebugWUS) {
            if (warpId == 0 && cIdx < 5 && iterCount <= 5) {
              printf("[WUS] Phase3 iter %d cand %d: targetAtom=%d, notUsed=%d, edgeOk=%d, valid=%d\n",
                     iterCount, cIdx, targetAtom, notUsed ? 1 : 0, edgeOk ? 1 : 0, valid ? 1 : 0);
            }
          }
        }

        // Use ballot to find valid lanes
        const uint32_t validMask = __ballot_sync(0xFFFFFFFF, valid);

        // Each valid lane writes its result
        if (valid) {
          if (localQueryAtom == numQueryAtoms - 1) {
            // Complete match
            const int matchIdx = atomicAdd(matchCount, 1);
            if constexpr (kDebugWUS) {
              printf("[WUS] MATCH FOUND! matchIdx=%d, mapping: ", matchIdx);
              for (int q = 0; q < numQueryAtoms; ++q) {
                printf("%d ", (q == localQueryAtom) ? targetAtom : (int)localMapping[q]);
              }
              printf("\n");
            }
            if (matchIdx < maxMatches) {
              const int writeOffset = matchOffset + matchIdx * numQueryAtoms;
              for (int q = 0; q < numQueryAtoms; ++q) {
                matchIndices[writeOffset + q] =
                  (q == localQueryAtom) ? static_cast<int16_t>(targetAtom) : localMapping[q];
              }
              atomicAdd(reportedCount, 1);
            }
          } else {
            // Enqueue for next level
            const int slot = atomicAdd(&queueTail, 1);
            if constexpr (kDebugWUS) {
              if (warpId == 0 && laneId < 5 && iterCount <= 5) {
                printf("[WUS] Phase3 iter %d: enqueueing at slot %d for queryAtom %d, targetAtom=%d\n",
                       iterCount, slot, localQueryAtom + 1, targetAtom);
              }
            }
            if (slot < maxQueueSize) {
              PartialMatch& next = workQueue[slot];
              for (int q = 0; q < numQueryAtoms; ++q) {
                next.mapping[q] = localMapping[q];
              }
              next.mapping[localQueryAtom] = static_cast<int8_t>(targetAtom);
              next.nextQueryAtom           = static_cast<int8_t>(localQueryAtom + 1);
            }
          }
        }
      }
    }

    // All threads sync before next iteration to ensure queue updates are visible
    block.sync();
  }

  if constexpr (kDebugWUS) {
    if (tid == 0) {
      printf("[WUS] DONE: total iterations=%d, final matchCount=%d, reportedCount=%d\n",
             iterCount, *matchCount, *reportedCount);
    }
  }
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_ALGOS_CUH

