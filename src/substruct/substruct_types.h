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

#ifndef NVMOLKIT_SUBSTRUCT_TYPES_H
#define NVMOLKIT_SUBSTRUCT_TYPES_H

#include <cstdint>
#include <vector>

namespace nvMolKit {

/**
 * @brief Algorithm choice for substructure matching.
 */
enum class SubstructAlgorithm {
  VF2,          ///< VF2 iterative stack-based DFS
  GSI,          ///< GSI-inspired BFS level-by-level join
  WarpUnified   ///< Novel warp-collective BFS search
};

/**
 * @brief Host-side results from batch substructure matching.
 *
 * For M targets x N queries (all-to-all matching), stores:
 * - Match counts for each pair (actual count, may exceed buffer)
 * - Reported counts for each pair (clamped to buffer capacity)
 * - Per-pair offsets into the flattened match index array
 * - Flattened match mappings (query atom -> target atom indices)
 */
struct SubstructMatchResultsHost {
  int numTargets = 0;
  int numQueries = 0;

  /// Actual match count per (target, query) pair [numTargets * numQueries]
  /// May exceed buffer capacity - use for detecting overflow
  std::vector<int> matchCounts;

  /// Reported (stored) match count per pair [numTargets * numQueries]
  /// Clamped to maxMatchesPerPair
  std::vector<int> reportedCounts;

  /// Offset into matchIndices for each pair [numTargets * numQueries + 1]
  /// matchIndices for pair i start at pairMatchStarts[i]
  std::vector<int> pairMatchStarts;

  /// Flattened match mappings
  /// Each match is numQueryAtoms consecutive int16_t values
  /// matchIndices[j] = target atom index that query atom (j % numQueryAtoms) maps to
  std::vector<int16_t> matchIndices;

  /// Maximum matches stored per pair (used to detect overflow)
  int maxMatchesPerPair = 0;

  /**
   * @brief Get pair index for (targetIdx, queryIdx).
   */
  [[nodiscard]] int pairIndex(int targetIdx, int queryIdx) const { return targetIdx * numQueries + queryIdx; }

  /**
   * @brief Check if pair had more matches than could be stored.
   */
  [[nodiscard]] bool hasOverflow(int targetIdx, int queryIdx) const {
    const int idx = pairIndex(targetIdx, queryIdx);
    return matchCounts[idx] > reportedCounts[idx];
  }
};

/// Maximum scratch space for boolean expression evaluation per query atom.
/// Complex SMARTS patterns with many OR branches can require significant scratch space.
/// E.g., [C,N,O,S,F,Cl,Br,I,...] with N alternatives needs 2N-1 slots (N leaves + N-1 ORs).
/// 256 supports up to ~128 OR alternatives per atom.
constexpr int kMaxBoolScratchSize = 256;

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_TYPES_H

