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

#include "device_timings.cuh"
#include "flat_bit_vect.h"
#include "global_pool.cuh"
#include "molecules_device.cuh"
#include "substruct_debug.h"
#include "substruct_types.h"

namespace nvMolKit {

// =============================================================================
// Shared Constants
// =============================================================================

constexpr int kWarpSize = 32;

// =============================================================================
// Output Mode for Substructure Search
// =============================================================================

/**
 * @brief Determines how matches are recorded.
 */
enum class SubstructOutputMode {
  StoreMatches,  ///< Store full match mappings to output buffer
  PaintBits      ///< Paint recursive match bit for first matched atom (no storage)
};

/**
 * @brief Parameters for paint mode output.
 */
struct PaintModeParams {
  uint32_t* recursiveBits;  ///< Buffer to paint bits into [maxTargetAtoms per pair]
  int       patternId;      ///< Bit position (0-31) to set
  int       maxTargetAtoms; ///< Stride for indexing recursiveBits
  int       outputPairIdx;  ///< Which pair's buffer to write to
};

// =============================================================================
// Helper function for checking if target atom is used in mapping
// =============================================================================

/**
 * @brief Check if a target atom is already used in a partial mapping.
 *
 * Only checks mapping[0..depth-1] since query atoms are processed in order.
 * This avoids the need for -1 sentinel initialization.
 *
 * @param mapping The partial mapping array
 * @param depth Current search depth (number of assigned query atoms)
 * @param targetAtom Target atom to check
 * @return true if targetAtom is already used in the mapping
 */
__device__ __forceinline__ bool isTargetUsedInMapping(const int8_t* mapping, int depth, int targetAtom) {
  for (int q = 0; q < depth; ++q) {
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
  int8_t mapping[kMaxQueryAtoms];       ///< mapping[q] = target atom idx (only [0..depth-1] valid)
  int8_t candidateIdx[kMaxQueryAtoms];  ///< Current candidate index at each stack level
  int    depth;                         ///< Current recursion depth (0 to numQueryAtoms-1)
  int    matchCount;                    ///< Number of complete matches found

  __device__ __forceinline__ void init() {
    // No need to initialize mapping - we only read [0..depth-1] which are always written first
    for (int i = 0; i < kMaxQueryAtoms; ++i) {
      candidateIdx[i] = 0;
    }
    depth      = 0;
    matchCount = 0;
  }

  __device__ __forceinline__ bool isTargetUsed(int targetIdx) const {
    return isTargetUsedInMapping(mapping, depth, targetIdx);
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
 *
 * Complete matches are never stored in the queue, so we only need
 * kMaxQueryAtoms - 1 slots (matching atoms 0 through numQueryAtoms-2).
 */
struct PartialMatch {
  int8_t mapping[kMaxQueryAtoms - 1];  ///< mapping[q] = target atom (only [0..nextQueryAtom-1] valid)
  int8_t nextQueryAtom;                ///< Next query atom to extend (also serves as depth)
};
static_assert(sizeof(PartialMatch) == kMaxQueryAtoms, "PartialMatch must be kMaxQueryAtoms bytes");

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
 * @param queryAtom Query atom being matched (also the current depth)
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
  // queryAtom is the current depth - atoms 0..queryAtom-1 are already mapped
  const int  depth            = queryAtom;

  for (int i = 0; i < queryDegree; ++i) {
    const int neighborQueryAtom = query.getNeighborAtomIdx(queryAtom, i);

    // Only check neighbors that are already mapped (index < current depth)
    if (neighborQueryAtom >= depth) {
      continue;
    }

    const int neighborTargetAtom = mapping[neighborQueryAtom];
    const int queryBondIdx       = query.getNeighborBondIdx(queryAtom, i);

    // Get query bond info
    int      queryBondType       = query.getBond(queryBondIdx, threadIdx.x, blockIdx.x).bondType;
    uint8_t  queryBondFlags      = 0;
    uint16_t queryAllowedBondTypes = 0;
    if (hasBondQueryData) {
      const BondQueryData& bqd = query.getBondQuery(queryBondIdx);
      queryBondType       = bqd.bondType;
      queryBondFlags      = bqd.queryFlags;
      queryAllowedBondTypes = bqd.allowedBondTypes;
    }

    // Check if targetAtom has an edge to neighborTargetAtom with compatible bond
    bool      foundEdge    = false;
    const int targetDegree = target.getAtomDegree(targetAtom);

    for (int j = 0; j < targetDegree; ++j) {
      if (target.getNeighborAtomIdx(targetAtom, j) == neighborTargetAtom) {
        const int       targetBondIdx = target.getNeighborBondIdx(targetAtom, j);
        const BondData& targetBond    = target.getBond(targetBondIdx, threadIdx.x, blockIdx.x);

        if constexpr (kDebugEdgeConsistency) {
          if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("[EdgeCheck] q%d->q%d (t%d->t%d): qBondType=%d, tBondType=%d, qFlags=0x%x, qAllowed=0x%x, tInRing=%d\n",
                   queryAtom, neighborQueryAtom, targetAtom, neighborTargetAtom,
                   queryBondType, targetBond.bondType, queryBondFlags, queryAllowedBondTypes, targetBond.isInRing);
          }
        }

        // Check bond type compatibility
        if (queryBondFlags & BondQueryNeverMatches) {
          // Impossible constraint (e.g., single AND aromatic) - never matches
          if constexpr (kDebugEdgeConsistency) {
            if (threadIdx.x == 0 && blockIdx.x == 0) {
              printf("[EdgeCheck]   -> FAIL: NeverMatches flag set\n");
            }
          }
          continue;
        } else if (queryBondFlags & BondQueryUseBondMask) {
          // Bond OR pattern: check if target bond type is in allowed mask
          const int tbt = targetBond.bondType;
          if (tbt < 0 || tbt >= 16 || !(queryAllowedBondTypes & (1u << tbt))) {
            if constexpr (kDebugEdgeConsistency) {
              if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("[EdgeCheck]   -> FAIL: bondMask check (tbt=%d not in mask 0x%x)\n", tbt, queryAllowedBondTypes);
              }
            }
            continue;
          }
        } else if (!bondTypeMatches(queryBondType, targetBond.bondType)) {
          if constexpr (kDebugEdgeConsistency) {
            if (threadIdx.x == 0 && blockIdx.x == 0) {
              printf("[EdgeCheck]   -> FAIL: bondType mismatch (%d != %d)\n", queryBondType, targetBond.bondType);
            }
          }
          continue;
        }

        // Check ring bond constraints if present
        if (queryBondFlags & (BondQueryIsRingBond | BondQueryNotRingBond)) {
          bool targetIsInRing = (targetBond.isInRing != 0);
          if (!ringBondConstraintsSatisfied(queryBondFlags, targetIsInRing)) {
            if constexpr (kDebugEdgeConsistency) {
              if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("[EdgeCheck]   -> FAIL: ring constraint (flags=0x%x, targetInRing=%d)\n", queryBondFlags, targetIsInRing);
              }
            }
            continue;
          }
        }

        if constexpr (kDebugEdgeConsistency) {
          if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("[EdgeCheck]   -> OK\n");
          }
        }
        foundEdge = true;
        break;
      }
    }

    if (!foundEdge) {
      if constexpr (kDebugEdgeConsistency) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
          printf("[EdgeCheck] q%d->q%d (t%d->t%d): NO EDGE FOUND in target\n",
                 queryAtom, neighborQueryAtom, targetAtom, neighborTargetAtom);
        }
      }
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
 * @param maxMatchesToFind Stop searching after this many matches (-1 = no limit)
 * @param countOnly If true, count matches but don't store them
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
                             int                                                   matchOffset,
                             int                                                   maxMatchesToFind = -1,
                             bool                                                  countOnly = false) {
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
  state.depth = 1;

  const bool hasEarlyExitLimit = (maxMatchesToFind >= 0);

  // Iterative DFS
  while (state.depth > 0) {
    // Check early exit
    if (hasEarlyExitLimit && *matchCount >= maxMatchesToFind) {
      break;
    }

    if (state.depth == numQueryAtoms) {
      // Complete match found
      const int currentMatchCount = atomicAdd(matchCount, 1);

      if (!countOnly && currentMatchCount < maxMatches) {
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
 * Uses shared memory with pre-allocated per-block overflow buffers.
 *
 * @tparam MaxTargetAtoms Maximum target atoms
 * @tparam MaxQueryAtoms Maximum query atoms
 * @tparam OutputMode How to record matches (StoreMatches or PaintBits)
 * @param target Target molecule view
 * @param query Query molecule view
 * @param labelMatrix Precomputed label compatibility matrix
 * @param sharedPartials Shared memory for partial matches (ping-pong buffers)
 * @param maxPartials Max partials in shared memory
 * @param overflowA Pre-allocated overflow buffer A (ping)
 * @param overflowB Pre-allocated overflow buffer B (pong)
 * @param maxOverflow Entries per overflow buffer
 * @param matchCount Output: number of matches found
 * @param reportedCount Output: number of matches written (StoreMatches) or painted (PaintBits)
 * @param matchIndices Output buffer (only used in StoreMatches mode)
 * @param maxMatches Maximum matches to write (only used in StoreMatches mode)
 * @param matchOffset Offset into output (only used in StoreMatches mode)
 * @param paintParams Paint mode parameters (only used in PaintBits mode)
 * @param maxMatchesToFind Stop searching after this many matches (-1 = no limit)
 * @param countOnly If true, count matches but don't store them
 * @param timings Optional timing data collection
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms, 
          SubstructOutputMode OutputMode = SubstructOutputMode::StoreMatches>
__device__ void gsiBFSSearchGPU(const MoleculeView&                                   target,
                                const MoleculeView&                                   query,
                                const BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>& labelMatrix,
                                PartialMatch*                                         sharedPartials,
                                int                                                   maxPartials,
                                PartialMatch*                                         overflowA,
                                PartialMatch*                                         overflowB,
                                int                                                   maxOverflow,
                                int*                                                  matchCount,
                                int*                                                  reportedCount,
                                int16_t*                                              matchIndices,
                                int                                                   maxMatches,
                                int                                                   matchOffset,
                                PaintModeParams                                       paintParams = {},
                                int                                                   maxMatchesToFind = -1,
                                bool                                                  countOnly = false,
                                DeviceTimingsData*                                    timings = nullptr) {
  long long int t_start;
  DEVICE_TIMING_START(timings, 0, t_start);

  const int tid    = threadIdx.x;
  const int laneId = tid % 32;
  const int warpId = tid / 32;
  constexpr int numWarps = kThreadsPerBlock / 32;

  const int numQueryAtoms  = query.numAtoms;
  const int numTargetAtoms = target.numAtoms;

  // Runtime-sized partial matches: stride = numQueryAtoms bytes per partial.
  // This allows fitting more partials when queries have fewer atoms.
  const int stride = numQueryAtoms;

  // Treat buffers as raw bytes for runtime-sized access
  int8_t* sharedBytes   = reinterpret_cast<int8_t*>(sharedPartials);
  int8_t* overflowABytes = reinterpret_cast<int8_t*>(overflowA);
  int8_t* overflowBBytes = reinterpret_cast<int8_t*>(overflowB);

  // Compute effective capacities based on runtime stride
  const int sharedBytesPerHalf    = maxPartials * sizeof(PartialMatch);
  const int effectiveMaxPartials  = sharedBytesPerHalf / stride;
  const int overflowBytes         = maxOverflow * sizeof(PartialMatch);
  const int effectiveMaxOverflow  = overflowBytes / stride;
  const int maxTotal              = effectiveMaxPartials + effectiveMaxOverflow;

  // Ping-pong byte pointers for shared memory halves
  int8_t* currentShared = sharedBytes;
  int8_t* nextShared    = sharedBytes + sharedBytesPerHalf;

  // Ping-pong overflow byte pointers
  int8_t* currentOverflow = overflowABytes;
  int8_t* nextOverflow    = overflowBBytes;

  __shared__ int currentCount;
  __shared__ int nextCount;
  __shared__ int earlyExitFlag;

  if (tid == 0) {
    currentCount  = 0;
    nextCount     = 0;
    earlyExitFlag = 0;
  }
  __syncthreads();

  DEVICE_TIMING_END(timings, 0, t_start);
  DEVICE_TIMING_START(timings, 1, t_start);

  if constexpr (kDebugGSI) {
    if (tid == 0) {
      printf("[GSI] numQueryAtoms=%d, numTargetAtoms=%d, effectiveMaxPartials=%d, effectiveMaxOverflow=%d, maxTotal=%d\n",
             numQueryAtoms, numTargetAtoms, effectiveMaxPartials, effectiveMaxOverflow, maxTotal);
      printf("[GSI] query.hasBondQueryData()=%d, query.hasQueryTrees()=%d\n",
             query.hasBondQueryData() ? 1 : 0, query.hasQueryTrees() ? 1 : 0);
    }
    __syncthreads();
  }

  // Initialize level 0: all candidates for query atom 0
  const bool singleAtomQuery = (numQueryAtoms == 1);
  const bool hasEarlyExitLimit = (maxMatchesToFind >= 0);

  for (int t = tid; t < numTargetAtoms; t += blockDim.x) {
    // Check early exit before processing
    if (hasEarlyExitLimit && earlyExitFlag) {
      break;
    }
    if (labelMatrix.get(t, 0)) {
      if (singleAtomQuery) {
        const int matchIdx = atomicAdd(matchCount, 1);
        // Check if we should signal early exit
        if (hasEarlyExitLimit && matchIdx + 1 >= maxMatchesToFind) {
          earlyExitFlag = 1;
        }
        if constexpr (OutputMode == SubstructOutputMode::StoreMatches) {
          if (!countOnly && matchIdx < maxMatches) {
            matchIndices[matchOffset + matchIdx] = static_cast<int16_t>(t);
            atomicAdd(reportedCount, 1);
          }
        } else {
          // Paint mode: set bit for this target atom
          atomicOr(&paintParams.recursiveBits[paintParams.outputPairIdx * paintParams.maxTargetAtoms + t],
                   1u << paintParams.patternId);
          atomicAdd(reportedCount, 1);
          if constexpr (kDebugGSI) {
            printf("[GSI Paint] pairIdx=%d, targetAtom=%d, patternId=%d, bit=0x%x\n",
                   paintParams.outputPairIdx, t, paintParams.patternId, 1u << paintParams.patternId);
          }
        }
      } else {
        const int slot = atomicAdd(&currentCount, 1);
        if (slot < effectiveMaxPartials) {
          int8_t* p = currentShared + slot * stride;
          p[0]          = static_cast<int8_t>(t);
          p[stride - 1] = 1;  // nextQueryAtom
        } else if (slot < maxTotal) {
          if constexpr (kDebugGSI) {
            if (slot == effectiveMaxPartials) {
              printf("[GSI] Level 0: spilling to overflow buffer (slot=%d)\n", slot);
            }
          }
          const int overflowSlot = slot - effectiveMaxPartials;
          int8_t* p = currentOverflow + overflowSlot * stride;
          p[0]          = static_cast<int8_t>(t);
          p[stride - 1] = 1;  // nextQueryAtom
        } else {
          if constexpr (kDebugGSI) {
            if (slot == maxTotal) {
              printf("[GSI] WARNING: Level 0 OVERFLOW EXHAUSTED! slot=%d >= maxTotal=%d\n", slot, maxTotal);
            }
          }
        }
      }
    }
  }
  __syncthreads();

  if constexpr (kDebugGSI) {
    if (tid == 0) {
      printf("[GSI] Level 0: %d candidates for query atom 0", currentCount);
      if (currentCount > effectiveMaxPartials) {
        printf(" (%d in shared, %d in overflow)", effectiveMaxPartials, currentCount - effectiveMaxPartials);
      }
      printf("\n");
      printf("[GSI] Level 0 candidates (shared): ");
      for (int i = 0; i < min(currentCount, effectiveMaxPartials) && i < 10; ++i) {
        printf("%d ", (int)currentShared[i * stride]);
      }
      if (currentCount > 10) printf("...");
      printf("\n");
    }
    __syncthreads();
  }

  DEVICE_TIMING_END(timings, 1, t_start);

  if (singleAtomQuery) {
    return;
  }

  DEVICE_TIMING_START(timings, 2, t_start);

  // BFS levels
  for (int level = 1; level < numQueryAtoms; ++level) {
    // Check early exit at start of level
    if (hasEarlyExitLimit && earlyExitFlag) {
      break;
    }

    const int queryAtom   = level;
    const int numPartials = min(currentCount, maxTotal);

    if constexpr (kDebugGSI) {
      if (tid == 0) {
        printf("[GSI] === Level %d: processing %d partials for queryAtom %d ===\n",
               level, numPartials, queryAtom);
      }
      __syncthreads();
    }

    if (tid == 0) {
      nextCount = 0;
    }
    __syncthreads();

    __shared__ int debugValidTotal;
    __shared__ int debugCheckedTotal;
    if constexpr (kDebugGSI) {
      if (tid == 0) {
        debugValidTotal   = 0;
        debugCheckedTotal = 0;
      }
      __syncthreads();
    }

    // Each warp processes partial matches in round-robin
    for (int pIdx = warpId; pIdx < numPartials; pIdx += numWarps) {
      // Get pointer to partial's mapping (raw byte access)
      const int8_t* partial = (pIdx < effectiveMaxPartials)
                                  ? (currentShared + pIdx * stride)
                                  : (currentOverflow + (pIdx - effectiveMaxPartials) * stride);

      if constexpr (kDebugGSI) {
        if (pIdx == 0 && laneId == 0 && level <= 5) {
          printf("[GSI] Level %d partial 0 mapping: ", level);
          for (int q = 0; q < numQueryAtoms; ++q) {
            printf("%d ", (int)partial[q]);
          }
          printf("\n");
        }
      }

      for (int tBase = 0; tBase < numTargetAtoms; tBase += kWarpSize) {
        const int t = tBase + laneId;
        
        bool valid   = false;
        bool labelOk = false;
        bool notUsed = false;
        bool edgeOk  = false;
        
        if (t < numTargetAtoms) {
          labelOk = labelMatrix.get(t, queryAtom);
          notUsed = !isTargetUsedInMapping(partial, queryAtom, t);
          edgeOk  = checkEdgeConsistency(target, query, partial, queryAtom, t);
          valid   = labelOk && notUsed && edgeOk;
          
          if constexpr (kDebugGSI) {
            atomicAdd(&debugCheckedTotal, 1);
            if (valid) atomicAdd(&debugValidTotal, 1);
          }
        }

        if constexpr (kDebugGSI) {
          if (pIdx == 0 && t < 5 && level <= 3) {
            printf("[GSI] Level %d partial 0 cand %d: labelOk=%d, notUsed=%d, edgeOk=%d, valid=%d\n",
                   level, t, labelOk ? 1 : 0, notUsed ? 1 : 0, edgeOk ? 1 : 0, valid ? 1 : 0);
          }
        }
        
        if (valid) {
          if (level == numQueryAtoms - 1) {
            const int matchIdx = atomicAdd(matchCount, 1);
            // Check if we should signal early exit
            if (hasEarlyExitLimit && matchIdx + 1 >= maxMatchesToFind) {
              earlyExitFlag = 1;
            }
            if constexpr (kDebugGSI) {
              printf("[GSI] MATCH FOUND! matchIdx=%d, mapping: ", matchIdx);
              for (int q = 0; q < numQueryAtoms; ++q) {
                printf("%d ", (q == queryAtom) ? t : (int)partial[q]);
              }
              printf("\n");
            }
            if constexpr (OutputMode == SubstructOutputMode::StoreMatches) {
              if (!countOnly && matchIdx < maxMatches) {
                const int writeOffset = matchOffset + matchIdx * numQueryAtoms;
                for (int q = 0; q < numQueryAtoms; ++q) {
                  matchIndices[writeOffset + q] = (q == queryAtom) ? static_cast<int16_t>(t)
                                                                   : partial[q];
                }
                atomicAdd(reportedCount, 1);
              }
            } else {
              // Paint mode: set bit for first target atom in this match (mapping[0])
              const int firstTargetAtom = partial[0];
              atomicOr(&paintParams.recursiveBits[paintParams.outputPairIdx * paintParams.maxTargetAtoms + firstTargetAtom],
                       1u << paintParams.patternId);
              atomicAdd(reportedCount, 1);
            }
          } else {
            const int slot = atomicAdd(&nextCount, 1);
            if constexpr (kDebugGSI) {
              if (slot < 3 && level <= 3) {
                printf("[GSI] Level %d: enqueueing slot %d, queryAtom %d -> target %d\n",
                       level, slot, queryAtom, t);
              }
            }
            // Write to shared memory ping-pong or global overflow
            // Only copy [0..queryAtom-1] from parent, then set [queryAtom]
            if (slot < effectiveMaxPartials) {
              int8_t* next = nextShared + slot * stride;
              for (int q = 0; q < queryAtom; ++q) {
                next[q] = partial[q];
              }
              next[queryAtom]    = static_cast<int8_t>(t);
              next[stride - 1]   = static_cast<int8_t>(queryAtom + 1);  // nextQueryAtom
            } else if (slot < maxTotal) {
              if constexpr (kDebugGSI) {
                if (slot == effectiveMaxPartials) {
                  printf("[GSI] Level %d: spilling to overflow buffer (slot=%d)\n", level, slot);
                }
              }
              const int overflowSlot = slot - effectiveMaxPartials;
              int8_t* next = nextOverflow + overflowSlot * stride;
              for (int q = 0; q < queryAtom; ++q) {
                next[q] = partial[q];
              }
              next[queryAtom]    = static_cast<int8_t>(t);
              next[stride - 1]   = static_cast<int8_t>(queryAtom + 1);  // nextQueryAtom
            } else {
              if constexpr (kDebugGSI) {
                if (slot == maxTotal) {
                  printf("[GSI] WARNING: Level %d OVERFLOW EXHAUSTED! slot=%d >= maxTotal=%d\n",
                         level, slot, maxTotal);
                }
              }
            }
          }
        }
      }
    }
    __syncthreads();

    if constexpr (kDebugGSI) {
      if (tid == 0) {
        printf("[GSI] Level %d summary: checked=%d, valid=%d, nextCount=%d",
               level, debugCheckedTotal, debugValidTotal, nextCount);
        if (nextCount > effectiveMaxPartials) {
          printf(" (%d in shared, %d in overflow)", effectiveMaxPartials, nextCount - effectiveMaxPartials);
        }
        if (nextCount > maxTotal) {
          printf(" [OVERFLOW LOST %d]", nextCount - maxTotal);
        }
        printf("\n");
      }
      __syncthreads();
    }

    // Swap buffers (just swap pointers, no copy needed)
    if (tid == 0) {
      currentCount = nextCount;
    }
    // Swap shared pointers (each thread has its own copy)
    int8_t* tmpShared = currentShared;
    currentShared     = nextShared;
    nextShared        = tmpShared;
    // Swap overflow pointers
    int8_t* tmpOverflow = currentOverflow;
    currentOverflow     = nextOverflow;
    nextOverflow        = tmpOverflow;

    __syncthreads();

    if constexpr (kDebugGSI) {
      if (tid == 0 && currentCount == 0) {
        printf("[GSI] Level %d: no more partials, stopping early\n", level);
      }
    }
    if (currentCount == 0) {
      break;
    }
  }

  DEVICE_TIMING_END(timings, 2, t_start);

  if constexpr (kDebugGSI) {
    if (tid == 0) {
      printf("[GSI] DONE: final matchCount=%d, reportedCount=%d\n", *matchCount, *reportedCount);
    }
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
 * Uses pre-allocated per-block overflow buffer when shared queue fills.
 *
 * @tparam MaxTargetAtoms Maximum target atoms
 * @tparam MaxQueryAtoms Maximum query atoms
 * @tparam OutputMode How to record matches (StoreMatches or PaintBits)
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms,
          SubstructOutputMode OutputMode = SubstructOutputMode::StoreMatches>
__device__ void warpUnifiedSearchGPU(const MoleculeView&                                   target,
                                     const MoleculeView&                                   query,
                                     const BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>& labelMatrix,
                                     CandidateList*                                        sharedCandidates,
                                     PartialMatch*                                         workQueue,
                                     int                                                   maxQueueSize,
                                     PartialMatch*                                         overflowBuffer,
                                     int                                                   maxOverflow,
                                     int*                                                  matchCount,
                                     int*                                                  reportedCount,
                                     int16_t*                                              matchIndices,
                                     int                                                   maxMatches,
                                     int                                                   matchOffset,
                                     PaintModeParams                                       paintParams = {},
                                     int                                                   maxMatchesToFind = -1,
                                     bool                                                  countOnly = false) {
  namespace cg = cooperative_groups;
  auto block   = cg::this_thread_block();
  auto tile32  = cg::tiled_partition<32>(block);

  const int tid      = block.thread_rank();
  const int laneId   = tile32.thread_rank();
  const int warpId   = tile32.meta_group_rank();
  constexpr int numWarps = kThreadsPerBlock / kWarpSize;

  const int numQueryAtoms  = query.numAtoms;
  const int numTargetAtoms = target.numAtoms;
  const int maxTotal       = maxQueueSize + maxOverflow;

  // Pre-allocated overflow (no dynamic allocation needed)
  PartialMatch* globalOverflow = overflowBuffer;

  // Queue state
  __shared__ int queueHead;
  __shared__ int queueTail;
  __shared__ int earlyExitFlag;

  const bool hasEarlyExitLimit = (maxMatchesToFind >= 0);

  if (tid == 0) {
    queueHead     = 0;
    queueTail     = 0;
    earlyExitFlag = 0;
  }
  block.sync();

  if constexpr (kDebugWUS) {
    if (tid == 0) {
      printf("[WUS] numQueryAtoms=%d, numTargetAtoms=%d, maxQueueSize=%d, maxOverflow=%d, maxTotal=%d\n",
             numQueryAtoms, numTargetAtoms, maxQueueSize, maxOverflow, maxTotal);
      printf("[WUS] query.hasBondQueryData()=%d, query.hasQueryTrees()=%d\n",
             query.hasBondQueryData() ? 1 : 0, query.hasQueryTrees() ? 1 : 0);
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
  const CandidateList& q0Candidates    = sharedCandidates[0];
  const bool           singleAtomQuery = (numQueryAtoms == 1);

  for (int i = tid; i < q0Candidates.count; i += block.size()) {
    // Check early exit before processing
    if (hasEarlyExitLimit && earlyExitFlag) {
      break;
    }
    if (singleAtomQuery) {
      // Single-atom query: record as complete match
      const int matchIdx = atomicAdd(matchCount, 1);
      // Check if we should signal early exit
      if (hasEarlyExitLimit && matchIdx + 1 >= maxMatchesToFind) {
        earlyExitFlag = 1;
      }
      const int targetAtom = q0Candidates.candidates[i];
      if constexpr (OutputMode == SubstructOutputMode::StoreMatches) {
        if (!countOnly && matchIdx < maxMatches) {
          matchIndices[matchOffset + matchIdx] = static_cast<int16_t>(targetAtom);
          atomicAdd(reportedCount, 1);
        }
      } else {
        // Paint mode: set bit for this target atom
        atomicOr(&paintParams.recursiveBits[paintParams.outputPairIdx * paintParams.maxTargetAtoms + targetAtom],
                 1u << paintParams.patternId);
        atomicAdd(reportedCount, 1);
      }
    } else {
      const int slot = atomicAdd(&queueTail, 1);
      if (slot < maxQueueSize) {
        workQueue[slot].mapping[0]    = q0Candidates.candidates[i];
        workQueue[slot].nextQueryAtom = 1;
      } else if (slot < maxTotal) {
        if constexpr (kDebugWUS) {
          if (slot == maxQueueSize) {
            printf("[WUS] Phase2: spilling to overflow buffer (slot=%d)\n", slot);
          }
        }
        const int overflowSlot = slot - maxQueueSize;
        globalOverflow[overflowSlot].mapping[0]    = q0Candidates.candidates[i];
        globalOverflow[overflowSlot].nextQueryAtom = 1;
      } else {
        if constexpr (kDebugWUS) {
          if (slot == maxTotal) {
            printf("[WUS] WARNING: Phase2 OVERFLOW EXHAUSTED! slot=%d >= maxTotal=%d\n", slot, maxTotal);
          }
        }
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
  __shared__ int snapshotQueueTail;

  if (tid == 0) {
    iterCount = 0;
  }
  block.sync();

  while (true) {
    if (tid == 0) {
      snapshotQueueTail = queueTail;
      // Check for early exit or no more work
      workAvailable = (queueHead < snapshotQueueTail && !(hasEarlyExitLimit && earlyExitFlag)) ? 1 : 0;
      if constexpr (kDebugWUS) {
        ++iterCount;
      }
    }
    block.sync();

    if constexpr (kDebugWUS) {
      if (tid == 0 && iterCount <= 10) {
        printf("[WUS] Phase3 iter %d: queueHead=%d, queueTail=%d (overflow=%d), workAvailable=%d\n",
               iterCount, queueHead, queueTail, 
               (queueTail > maxQueueSize) ? queueTail - maxQueueSize : 0, workAvailable);
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
    const bool hasWork = (myWorkIdx >= 0 && myWorkIdx < maxTotal);

    // Copy work item to registers to avoid races with concurrent writes to workQueue
    // Only [0..localQueryAtom-1] are valid - no need to init or copy beyond that
    int8_t localMapping[kMaxQueryAtoms];
    int    localQueryAtom = 0;

    if (hasWork) {
      // Read from shared queue or overflow buffer based on index
      const PartialMatch& work = (myWorkIdx < maxQueueSize) 
                                   ? workQueue[myWorkIdx] 
                                   : globalOverflow[myWorkIdx - maxQueueSize];
      localQueryAtom = work.nextQueryAtom;
      for (int q = 0; q < localQueryAtom; ++q) {
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
          const bool notUsed = !isTargetUsedInMapping(localMapping, localQueryAtom, targetAtom);
          const bool edgeOk  = checkEdgeConsistency(target, query, localMapping, localQueryAtom, targetAtom);
          valid = notUsed && edgeOk;

          if constexpr (kDebugWUS) {
            if (warpId == 0 && cIdx < 5 && iterCount <= 5) {
              printf("[WUS] Phase3 iter %d cand %d: targetAtom=%d, notUsed=%d, edgeOk=%d, valid=%d\n",
                     iterCount, cIdx, targetAtom, notUsed ? 1 : 0, edgeOk ? 1 : 0, valid ? 1 : 0);
            }
          }
        }


        // Each valid lane writes its result
        if (valid) {
          if (localQueryAtom == numQueryAtoms - 1) {
            // Complete match
            const int matchIdx = atomicAdd(matchCount, 1);
            // Check if we should signal early exit
            if (hasEarlyExitLimit && matchIdx + 1 >= maxMatchesToFind) {
              earlyExitFlag = 1;
            }
            if constexpr (kDebugWUS) {
              printf("[WUS] MATCH FOUND! matchIdx=%d, mapping: ", matchIdx);
              for (int q = 0; q < numQueryAtoms; ++q) {
                printf("%d ", (q == localQueryAtom) ? targetAtom : (int)localMapping[q]);
              }
              printf("\n");
            }
            if constexpr (OutputMode == SubstructOutputMode::StoreMatches) {
              if (!countOnly && matchIdx < maxMatches) {
                const int writeOffset = matchOffset + matchIdx * numQueryAtoms;
                for (int q = 0; q < numQueryAtoms; ++q) {
                  matchIndices[writeOffset + q] =
                    (q == localQueryAtom) ? static_cast<int16_t>(targetAtom) : localMapping[q];
                }
                atomicAdd(reportedCount, 1);
              }
            } else {
              // Paint mode: set bit for first target atom in this match (localMapping[0])
              const int firstTargetAtom = localMapping[0];
              atomicOr(&paintParams.recursiveBits[paintParams.outputPairIdx * paintParams.maxTargetAtoms + firstTargetAtom],
                       1u << paintParams.patternId);
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
            // Only copy [0..localQueryAtom-1] from parent, then set [localQueryAtom]
            if (slot < maxQueueSize) {
              PartialMatch& next = workQueue[slot];
              for (int q = 0; q < localQueryAtom; ++q) {
                next.mapping[q] = localMapping[q];
              }
              next.mapping[localQueryAtom] = static_cast<int8_t>(targetAtom);
              next.nextQueryAtom           = static_cast<int8_t>(localQueryAtom + 1);
            } else if (slot < maxTotal) {
              if constexpr (kDebugWUS) {
                if (slot == maxQueueSize) {
                  printf("[WUS] Spilling to overflow buffer (slot=%d)\n", slot);
                }
              }
              const int overflowSlot = slot - maxQueueSize;
              PartialMatch& next = globalOverflow[overflowSlot];
              for (int q = 0; q < localQueryAtom; ++q) {
                next.mapping[q] = localMapping[q];
              }
              next.mapping[localQueryAtom] = static_cast<int8_t>(targetAtom);
              next.nextQueryAtom           = static_cast<int8_t>(localQueryAtom + 1);
            } else {
              if constexpr (kDebugWUS) {
                if (slot == maxTotal) {
                  printf("[WUS] WARNING: OVERFLOW EXHAUSTED! slot=%d >= maxTotal=%d\n", slot, maxTotal);
                }
              }
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
      printf("[WUS] DONE: total iterations=%d, final matchCount=%d, reportedCount=%d",
             iterCount, *matchCount, *reportedCount);
      if (queueTail > maxQueueSize) {
        printf(" (used overflow: %d entries)", min(queueTail, maxTotal) - maxQueueSize);
      }
      if (queueTail > maxTotal) {
        printf(" [OVERFLOW LOST %d]", queueTail - maxTotal);
      }
      printf("\n");
    }
  }
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_ALGOS_CUH

