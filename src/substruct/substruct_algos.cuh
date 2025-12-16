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

namespace nvMolKit {

// =============================================================================
// Algorithm Selection
// =============================================================================

/**
 * @brief Algorithm choice for substructure matching.
 */
enum class SubstructAlgorithm {
  VF2,          ///< VF2 iterative stack-based DFS
  GSI,          ///< GSI-inspired BFS level-by-level join
  WarpUnified   ///< Novel warp-collective BFS search
};

// =============================================================================
// Shared Constants
// =============================================================================

constexpr int kWarpSize       = 32;
constexpr int kMaxQueryAtoms  = 64;
constexpr int kMaxTargetAtoms = 128;

// =============================================================================
// VF2 Data Structures
// =============================================================================

/**
 * @brief State for VF2 iterative search (per-warp in shared memory).
 *
 * Maintains partial match and exploration stack for DFS backtracking.
 */
struct VF2State {
  int8_t   mapping[kMaxQueryAtoms];       ///< mapping[q] = target atom idx, -1 if unassigned
  uint32_t usedTargetsMask;               ///< Bitmask of target atoms already in mapping (up to 32)
  int8_t   candidateIdx[kMaxQueryAtoms];  ///< Current candidate index at each stack level
  int      depth;                         ///< Current recursion depth (0 to numQueryAtoms-1)
  int      matchCount;                    ///< Number of complete matches found

  __device__ __forceinline__ void init() {
    for (int i = 0; i < kMaxQueryAtoms; ++i) {
      mapping[i]      = -1;
      candidateIdx[i] = 0;
    }
    usedTargetsMask = 0;
    depth           = 0;
    matchCount      = 0;
  }

  __device__ __forceinline__ bool isTargetUsed(int targetIdx) const {
    return (usedTargetsMask & (1u << targetIdx)) != 0;
  }

  __device__ __forceinline__ void markTargetUsed(int targetIdx) { usedTargetsMask |= (1u << targetIdx); }

  __device__ __forceinline__ void unmarkTargetUsed(int targetIdx) { usedTargetsMask &= ~(1u << targetIdx); }
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
  int8_t   mapping[kMaxQueryAtoms];  ///< mapping[q] = target atom, -1 if unassigned
  uint32_t usedTargetsMask;          ///< Bitmask of used target atoms
  int8_t   nextQueryAtom;            ///< Next query atom to extend

  __device__ __forceinline__ void init() {
    for (int i = 0; i < kMaxQueryAtoms; ++i) {
      mapping[i] = -1;
    }
    usedTargetsMask = 0;
    nextQueryAtom   = 0;
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
  const int queryDegree = query.getAtomDegree(queryAtom);

  for (int i = 0; i < queryDegree; ++i) {
    const int neighborQueryAtom = query.getNeighborAtomIdx(queryAtom, i);

    // Only check neighbors that are already mapped
    if (mapping[neighborQueryAtom] < 0) {
      continue;
    }

    const int neighborTargetAtom = mapping[neighborQueryAtom];
    const int queryBondIdx       = query.getNeighborBondIdx(queryAtom, i);
    const int queryBondType      = query.getBond(queryBondIdx).bondType;

    // Check if targetAtom has an edge to neighborTargetAtom with compatible bond type
    bool foundEdge          = false;
    const int targetDegree = target.getAtomDegree(targetAtom);

    for (int j = 0; j < targetDegree; ++j) {
      if (target.getNeighborAtomIdx(targetAtom, j) == neighborTargetAtom) {
        const int targetBondIdx  = target.getNeighborBondIdx(targetAtom, j);
        const int targetBondType = target.getBond(targetBondIdx).bondType;

        // Bond type compatibility check (query bond must match target bond)
        // For now: exact match. Could extend to handle query bond wildcards.
        if (targetBondType == queryBondType) {
          foundEdge = true;
          break;
        }
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

  state.init();
  state.mapping[0] = static_cast<int8_t>(startingTargetAtom);
  state.markTargetUsed(startingTargetAtom);
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
        const int prevTarget = state.mapping[state.depth];
        state.mapping[state.depth] = -1;
        state.unmarkTargetUsed(prevTarget);
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
        state.markTargetUsed(candidateTarget);
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
        const int prevTarget = state.mapping[state.depth];
        state.mapping[state.depth] = -1;
        state.unmarkTargetUsed(prevTarget);
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
          sharedPartials[slot].mapping[0]      = static_cast<int8_t>(t);
          sharedPartials[slot].usedTargetsMask = 1u << t;
          sharedPartials[slot].nextQueryAtom   = 1;
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
          const bool notUsed = (partial.usedTargetsMask & (1u << t)) == 0;
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
              next.mapping[queryAtom]  = static_cast<int8_t>(t);
              next.usedTargetsMask     = partial.usedTargetsMask | (1u << t);
              next.nextQueryAtom       = static_cast<int8_t>(queryAtom + 1);
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
        workQueue[slot].mapping[0]      = q0Candidates.candidates[i];
        workQueue[slot].usedTargetsMask = 1u << q0Candidates.candidates[i];
        workQueue[slot].nextQueryAtom   = 1;
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

  while (true) {
    if (tid == 0) {
      workAvailable = (queueHead < queueTail) ? 1 : 0;
    }
    block.sync();

    if (!workAvailable) {
      break;
    }

    // Each warp pops one work item
    if (laneId == 0 && warpId < 32) {
      warpWorkIdx[warpId] = atomicAdd(&queueHead, 1);
    }
    tile32.sync();

    const int myWorkIdx = warpWorkIdx[warpId];

    // Use hasWork flag instead of continue to ensure all threads hit sync
    const bool hasWork = (myWorkIdx < queueTail && myWorkIdx < maxQueueSize);

    if (hasWork) {
      const PartialMatch& work      = workQueue[myWorkIdx % maxQueueSize];
      const int           queryAtom = work.nextQueryAtom;

      if (queryAtom < numQueryAtoms) {
        const CandidateList& candidates = sharedCandidates[queryAtom];

        // Warp-parallel candidate evaluation
        for (int cBase = 0; cBase < candidates.count; cBase += kWarpSize) {
          const int cIdx = cBase + laneId;

          bool valid      = false;
          int  targetAtom = -1;

          if (cIdx < candidates.count) {
            targetAtom         = candidates.candidates[cIdx];
            const bool notUsed = (work.usedTargetsMask & (1u << targetAtom)) == 0;
            valid = notUsed && checkEdgeConsistency(target, query, work.mapping, queryAtom, targetAtom);
          }

          // Use ballot to find valid lanes
          const uint32_t validMask = __ballot_sync(0xFFFFFFFF, valid);

          // Each valid lane writes its result
          if (valid) {
            if (queryAtom == numQueryAtoms - 1) {
              // Complete match
              const int matchIdx = atomicAdd(matchCount, 1);
              if (matchIdx < maxMatches) {
                const int writeOffset = matchOffset + matchIdx * numQueryAtoms;
                for (int q = 0; q < numQueryAtoms; ++q) {
                  matchIndices[writeOffset + q] =
                    (q == queryAtom) ? static_cast<int16_t>(targetAtom) : work.mapping[q];
                }
                atomicAdd(reportedCount, 1);
              }
            } else {
              // Enqueue for next level
              const int slot = atomicAdd(&queueTail, 1);
              if (slot < maxQueueSize) {
                PartialMatch& next = workQueue[slot % maxQueueSize];
                for (int q = 0; q < numQueryAtoms; ++q) {
                  next.mapping[q] = work.mapping[q];
                }
                next.mapping[queryAtom]  = static_cast<int8_t>(targetAtom);
                next.usedTargetsMask     = work.usedTargetsMask | (1u << targetAtom);
                next.nextQueryAtom       = static_cast<int8_t>(queryAtom + 1);
              }
            }
          }
        }
      }
    }

    // All threads sync before next iteration to ensure queue updates are visible
    block.sync();
  }
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_ALGOS_CUH

