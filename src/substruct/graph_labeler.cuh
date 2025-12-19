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

#ifndef NVMOLKIT_GRAPH_LABELER_CUH
#define NVMOLKIT_GRAPH_LABELER_CUH

#include <cooperative_groups.h>

#include "atom_data_packed.h"
#include "boolean_tree.cuh"
#include "flat_bit_vect.h"
#include "molecules.h"
#include "molecules_device.cuh"

namespace nvMolKit {

// =============================================================================
// GPU-Optimized Matching Functions (branchless, warp-uniform)
// =============================================================================

/**
 * @brief Branchless atom matching using packed data and precomputed masks.
 *
 * Uses bitwise mask-and-compare to check all atom properties in just 4 instructions
 * (2 AND, 2 CMP). All threads in a warp execute identical instructions regardless
 * of which fields are being compared, eliminating warp divergence.
 *
 * @param target Packed target atom data
 * @param queryMask Precomputed query mask with expected values
 * @return true if target matches all specified query fields
 */
__device__ __forceinline__ bool atomMatchesOptimized(const AtomDataPacked& target, const AtomQueryMask& queryMask) {
  return atomMatchesPacked(target, queryMask);
}

/**
 * @brief Check if target atom has sufficient bonds using precomputed counts.
 *
 * Uses precomputed bond type counts to avoid runtime bond traversal.
 * Uniform comparison across all bond types eliminates divergent loops.
 *
 * @param targetCounts Precomputed bond type counts for target atom
 * @param queryCounts Precomputed bond type counts for query atom
 * @return true if target has >= bonds of each type compared to query
 */
__device__ __forceinline__ bool bondCountsMatchOptimized(const BondTypeCounts& targetCounts,
                                                         const BondTypeCounts& queryCounts) {
  return bondCountsMatchPacked(targetCounts, queryCounts);
}

/**
 * @brief Combined optimized match check for a (target, query) atom pair.
 *
 * Checks both atom properties and bond counts using precomputed data.
 * Fully branchless and warp-uniform.
 *
 * @param target Target molecule view
 * @param targetAtomIdx Index of target atom
 * @param query Query molecule view (must have query masks populated)
 * @param queryAtomIdx Index of query atom
 * @return true if target atom can match query atom
 */
__device__ __forceinline__ bool atomPairMatchesOptimized(const MoleculeView& target,
                                                         int                 targetAtomIdx,
                                                         const MoleculeView& query,
                                                         int                 queryAtomIdx) {
  const AtomDataPacked& targetPacked = target.getAtomPacked(targetAtomIdx);
  const AtomQueryMask&  queryMask    = query.getQueryMask(queryAtomIdx);
  const BondTypeCounts& targetBonds  = target.getBondTypeCounts(targetAtomIdx);
  const BondTypeCounts& queryBonds   = query.getBondTypeCounts(queryAtomIdx);

  return atomMatchesOptimized(targetPacked, queryMask) && bondCountsMatchOptimized(targetBonds, queryBonds);
}

/**
 * @brief Match using boolean expression tree for compound queries (OR/NOT).
 *
 * Evaluates the full boolean expression tree for query atoms that have
 * compound queries. For label matrix purposes, only checks atom properties
 * (not bond counts) to match RDKit's queryAtom->Match() semantics.
 *
 * @param target Target molecule view
 * @param targetAtomIdx Index of target atom
 * @param query Query molecule view (must have query trees populated)
 * @param queryAtomIdx Index of query atom
 * @param recursiveMatchBits Per-pair recursive match bits for this target atom (32 bits for patterns 0-31)
 * @return true if target atom's properties match the compound query expression
 */
__device__ __forceinline__ bool atomPairMatchesWithTree(const MoleculeView& target,
                                                        int                 targetAtomIdx,
                                                        const MoleculeView& query,
                                                        int                 queryAtomIdx,
                                                        uint32_t            recursiveMatchBits = 0) {
  const AtomDataPacked&   targetPacked   = target.getAtomPacked(targetAtomIdx);
  const AtomQueryTree&    tree           = query.getQueryTree(queryAtomIdx);
  const BoolInstruction*  instructions   = query.getQueryInstructions(queryAtomIdx);
  const AtomQueryMask*    leafMasks      = query.getQueryLeafMasks(queryAtomIdx);

  // For label matrix, only check atom properties (not bond counts)
  // Bond connectivity is verified during actual substructure search
  return evaluateBoolTree<false>(&targetPacked, nullptr, leafMasks, nullptr, instructions, tree, recursiveMatchBits);
}

// =============================================================================
// Warp-Parallel Label Matrix Population
// =============================================================================

/**
 * @brief Populate label matrix using warp-level parallelism with shared memory.
 *
 * This is the GPU-optimized version that:
 * 1. Loads query data cooperatively into shared memory
 * 2. Processes (target, query) pairs in warp-sized chunks
 * 3. Uses branchless matching via packed data and precomputed masks
 * 4. Writes results with atomic bit operations
 *
 * Requirements:
 * - Must be called with a full thread block (multiple warps)
 * - Shared memory must be pre-allocated for query data
 * - Both target and query must have packed data populated
 *
 * @tparam MaxTargetAtoms Maximum number of atoms in target graph
 * @tparam MaxQueryAtoms Maximum number of atoms in query graph
 * @param target Target molecule view with packed data
 * @param query Query molecule view with query masks and packed data
 * @param labelMatrix Output 2D bit matrix view (in shared memory)
 * @param sharedQueryPacked Shared memory buffer for query packed data [MaxQueryAtoms]
 * @param sharedQueryMasks Shared memory buffer for query masks [MaxQueryAtoms]
 * @param sharedQueryBondCounts Shared memory buffer for query bond counts [MaxQueryAtoms]
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ __forceinline__ void populateLabelMatrixWarpParallel(const MoleculeView&                             target,
                                                const MoleculeView&                             query,
                                                BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>& labelMatrix,
                                                AtomDataPacked*                                 sharedQueryPacked,
                                                AtomQueryMask*                                  sharedQueryMasks,
                                                BondTypeCounts*                                 sharedQueryBondCounts) {
  namespace cg = cooperative_groups;

  // Get thread block and warp information
  auto      block      = cg::this_thread_block();
  auto      tile32     = cg::tiled_partition<32>(block);
  const int tid        = block.thread_rank();
  const int numThreads = block.size();
  const int laneId     = tile32.thread_rank();
  const int warpId     = tile32.meta_group_rank();
  const int numWarps   = tile32.meta_group_size();

  const int numQueryAtoms  = query.numAtoms;
  const int numTargetAtoms = target.numAtoms;

  // Step 1: Parallel clear of label matrix
  labelMatrix.clearParallel(tid, numThreads);
  block.sync();

  // Step 2: Cooperative load of query data into shared memory
  for (int q = tid; q < numQueryAtoms; q += numThreads) {
    sharedQueryPacked[q] = query.getAtomPacked(q);
    sharedQueryMasks[q]  = query.getQueryMask(q);
  }
  block.sync();

  // Step 3: Warp-parallel processing of (target, query) pairs
  const int numPairs    = numTargetAtoms * numQueryAtoms;
  const int warpsNeeded = (numPairs + 31) / 32;

  for (int chunkIdx = warpId; chunkIdx < warpsNeeded; chunkIdx += numWarps) {
    const int pairIdx = chunkIdx * 32 + laneId;

    // All lanes compute, but only valid pairs write
    const bool validPair = (pairIdx < numPairs);

    // Compute indices (uniform division/modulo across warp)
    const int targetIdx = validPair ? (pairIdx / numQueryAtoms) : 0;
    const int queryIdx  = validPair ? (pairIdx % numQueryAtoms) : 0;

    // Load target data from global memory
    const AtomDataPacked targetPacked = validPair ? target.getAtomPacked(targetIdx) : AtomDataPacked{};

    // Load query data from shared memory
    const AtomQueryMask queryMask = sharedQueryMasks[queryIdx];

    // Branchless atom property matching only (bond counts not checked for label matrix)
    const bool atomMatch = atomMatchesPacked(targetPacked, queryMask);
    const bool matches   = validPair && atomMatch;

    // Write result atomically (only matching pairs write)
    if (matches) {
      labelMatrix.setAtomic(targetIdx, queryIdx);
    }
  }
}

/**
 * @brief Simplified warp-parallel label matrix population without explicit shared memory args.
 *
 * This version declares shared memory internally. Use when you don't need to
 * share the query data with other kernel logic.
 *
 * @tparam MaxTargetAtoms Maximum number of atoms in target graph
 * @tparam MaxQueryAtoms Maximum number of atoms in query graph
 * @param target Target molecule view with packed data
 * @param query Query molecule view with query masks and packed data
 * @param labelMatrix Output 2D bit matrix view
 * @param pairRecursiveBits Per-pair recursive match bits indexed by [targetAtomIdx], or nullptr if none
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ __forceinline__ void populateLabelMatrixOptimized(const MoleculeView&                             target,
                                             const MoleculeView&                             query,
                                             BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>& labelMatrix,
                                             const uint32_t*                                 pairRecursiveBits = nullptr) {
  // Check if query has boolean trees (compound queries with OR/NOT)
  if (query.hasQueryTrees()) {
    // Use boolean tree evaluation for compound queries
    namespace cg = cooperative_groups;
    auto      block      = cg::this_thread_block();
    const int tid        = block.thread_rank();
    const int numThreads = block.size();

    const int numQueryAtoms  = query.numAtoms;
    const int numTargetAtoms = target.numAtoms;

    // Clear label matrix
    labelMatrix.clearParallel(tid, numThreads);
    block.sync();

    // Process all (target, query) pairs
    const int numPairs = numTargetAtoms * numQueryAtoms;
    for (int pairIdx = tid; pairIdx < numPairs; pairIdx += numThreads) {
      const int targetIdx = pairIdx / numQueryAtoms;
      const int queryIdx  = pairIdx % numQueryAtoms;

      const uint32_t recursiveBits = pairRecursiveBits ? pairRecursiveBits[targetIdx] : 0;
      if (atomPairMatchesWithTree(target, targetIdx, query, queryIdx, recursiveBits)) {
        labelMatrix.setAtomic(targetIdx, queryIdx);
      }
    }
  } else {
    // Use fast path for simple AND-only queries
    __shared__ AtomDataPacked sharedQueryPacked[MaxQueryAtoms];
    __shared__ AtomQueryMask  sharedQueryMasks[MaxQueryAtoms];
    __shared__ BondTypeCounts sharedQueryBondCounts[MaxQueryAtoms];

    populateLabelMatrixWarpParallel<MaxTargetAtoms, MaxQueryAtoms>(target,
                                                                   query,
                                                                   labelMatrix,
                                                                   sharedQueryPacked,
                                                                   sharedQueryMasks,
                                                                   sharedQueryBondCounts);
  }
}

// =============================================================================
// Legacy Matching Functions (kept for compatibility)
// =============================================================================

/**
 * @brief Check if a target atom matches a query atom based on query flags.
 *
 * Compares the target atom's properties against the query atom's properties,
 * but only for the fields specified in the query flags bitmask.
 *
 * @param target The target atom data
 * @param query The query atom data (contains values to match against)
 * @param queryFlags Bitmask of AtomQueryFlags indicating which fields to compare
 * @return true if target matches query for all specified fields
 */
__device__ __forceinline__ bool atomMatches(const AtomData& target, const AtomData& query, AtomQuery queryFlags) {
  if (queryFlags & AtomQueryAtomicNum) {
    if (target.atomicNum != query.atomicNum) {
      return false;
    }
  }
  if (queryFlags & AtomQueryNumExplicitHs) {
    if (target.numExplicitHs != query.numExplicitHs) {
      return false;
    }
  }
  if (queryFlags & AtomQueryExplicitValence) {
    if (target.explicitValence != query.explicitValence) {
      return false;
    }
  }
  if (queryFlags & AtomQueryImplicitValence) {
    if (target.implicitValence != query.implicitValence) {
      return false;
    }
  }
  if (queryFlags & AtomQueryFormalCharge) {
    if (target.formalCharge != query.formalCharge) {
      return false;
    }
  }
  if (queryFlags & AtomQueryChiralTag) {
    if (target.chiralTag != query.chiralTag) {
      return false;
    }
  }
  if (queryFlags & AtomQueryNumRadicalElectrons) {
    if (target.numRadicalElectrons != query.numRadicalElectrons) {
      return false;
    }
  }
  if (queryFlags & AtomQueryHybridization) {
    if (target.hybridization != query.hybridization) {
      return false;
    }
  }
  if (queryFlags & AtomQueryMinRingSize) {
    if (target.minRingSize != query.minRingSize) {
      return false;
    }
  }
  if (queryFlags & AtomQueryNumRings) {
    if (target.numRings != query.numRings) {
      return false;
    }
  }
  if (queryFlags & AtomQueryTotalValence) {
    if (target.totalValence != query.totalValence) {
      return false;
    }
  }
  if (queryFlags & AtomQueryIsAromatic) {
    if (!target.isAromatic) {
      return false;
    }
  }
  if (queryFlags & AtomQueryIsAliphatic) {
    if (target.isAromatic) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Check if target atom has at least as many bonds of each type as query atom.
 *
 * For each bond type present in the query atom's neighborhood, verifies that the
 * target atom has at least as many bonds of that type.
 *
 * @param target The target molecule view
 * @param targetAtomIdx Index of the atom in the target molecule
 * @param query The query molecule view
 * @param queryAtomIdx Index of the atom in the query molecule
 * @return true if target has sufficient bonds of each type
 */
__device__ __forceinline__ bool bondCountsMatch(const MoleculeView& target,
                                                int                 targetAtomIdx,
                                                const MoleculeView& query,
                                                int                 queryAtomIdx) {
  // Use precomputed bond type counts - no need to recompute at runtime
  const BondTypeCounts& targetCounts = target.getBondTypeCounts(targetAtomIdx);
  const BondTypeCounts& queryCounts  = query.getBondTypeCounts(queryAtomIdx);
  return targetCounts.canMatchQuery(queryCounts);
}

/**
 * @brief Populate the label matrix for substructure matching.
 *
 * Given a target molecule and a query molecule, populates a 2D bit matrix where
 * bit[i][j] is set if target atom i could potentially match query atom j.
 * A match requires:
 * 1. The atom properties match according to the query flags
 * 2. The target atom has at least as many bonds of each type as the query atom
 *
 * @tparam MaxTargetAtoms Maximum number of atoms in target graph
 * @tparam MaxQueryAtoms Maximum number of atoms in query graph
 * @param target The target molecule view
 * @param query The query molecule view
 * @param labelMatrix Output 2D bit matrix view [target_atoms x query_atoms]
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ __forceinline__ void populateLabelMatrix(const MoleculeView&                             target,
                                    const MoleculeView&                             query,
                                    BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>& labelMatrix) {
  // Clear the matrix first
  labelMatrix.clear();

  // For each target atom
  for (int targetIdx = 0; targetIdx < target.numAtoms; ++targetIdx) {
    const AtomData& targetAtom = target.getAtom(targetIdx);

    // For each query atom
    for (int queryIdx = 0; queryIdx < query.numAtoms; ++queryIdx) {
      const AtomData& queryAtom  = query.getAtom(queryIdx);
      const AtomQuery queryFlags = query.getAtomQuery(queryIdx);

      // Check if atoms match
      if (!atomMatches(targetAtom, queryAtom, queryFlags)) {
        continue;
      }

      // Check if bond counts match
      if (!bondCountsMatch(target, targetIdx, query, queryIdx)) {
        continue;
      }

      // Set the bit - this target atom can match this query atom
      labelMatrix.set(targetIdx, queryIdx, true);
    }
  }
}

/**
 * @brief Populate the label matrix for a single target atom (parallelizable version).
 *
 * This version processes a single target atom, making it suitable for parallel
 * execution where each thread handles one target atom.
 *
 * @tparam MaxTargetAtoms Maximum number of atoms in target graph
 * @tparam MaxQueryAtoms Maximum number of atoms in query graph
 * @param target The target molecule view
 * @param targetAtomIdx Index of the target atom to process
 * @param query The query molecule view
 * @param labelMatrix Output 2D bit matrix view [target_atoms x query_atoms]
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ __forceinline__ void populateLabelMatrixForAtom(const MoleculeView&                             target,
                                           int                                             targetAtomIdx,
                                           const MoleculeView&                             query,
                                           BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>& labelMatrix) {
  if (targetAtomIdx >= target.numAtoms) {
    return;
  }

  const AtomData& targetAtom = target.getAtom(targetAtomIdx);

  // For each query atom
  for (int queryIdx = 0; queryIdx < query.numAtoms; ++queryIdx) {
    const AtomData& queryAtom  = query.getAtom(queryIdx);
    const AtomQuery queryFlags = query.getAtomQuery(queryIdx);

    // Only set bits for matches; assumes matrix is pre-cleared
    if (atomMatches(targetAtom, queryAtom, queryFlags) && bondCountsMatch(target, targetAtomIdx, query, queryIdx)) {
      labelMatrix.set(targetAtomIdx, queryIdx, true);
    }
  }
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_GRAPH_LABELER_CUH
