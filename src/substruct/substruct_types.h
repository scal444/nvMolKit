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
 * @brief Accumulated results from substructure matching.
 *
 * Dynamically allocated nested vector format: matches[targetIdx][queryIdx][matchIdx]
 * is a vector of target atom indices for that match.
 *
 * Memory is allocated proportional to actual matches found, avoiding the
 * worst-case pre-allocation required by CSR formats.
 */
struct SubstructSearchResults {
  /// matches[t][q] = vector of matches for target t against query q
  /// Each match is a vector<int> of target atom indices (one per query atom)
  std::vector<std::vector<std::vector<std::vector<int>>>> matches;

  /// actualMatchCounts[t][q] = total matches found (may exceed stored if capped)
  std::vector<std::vector<int>> actualMatchCounts;

  int numTargets = 0;
  int numQueries = 0;

  void resize(int nTargets, int nQueries) {
    numTargets = nTargets;
    numQueries = nQueries;
    matches.assign(nTargets, std::vector<std::vector<std::vector<int>>>(nQueries));
    actualMatchCounts.assign(nTargets, std::vector<int>(nQueries, 0));
  }

  /// Compute flat pair index (for compatibility with CSR-style access patterns)
  [[nodiscard]] int pairIndex(int targetIdx, int queryIdx) const {
    return targetIdx * numQueries + queryIdx;
  }

  /// Check if pair had more matches than could be stored
  [[nodiscard]] bool hasOverflow(int targetIdx, int queryIdx) const {
    return actualMatchCounts[targetIdx][queryIdx] >
           static_cast<int>(matches[targetIdx][queryIdx].size());
  }

  /// Number of matches stored for this pair
  [[nodiscard]] int matchCount(int targetIdx, int queryIdx) const {
    return static_cast<int>(matches[targetIdx][queryIdx].size());
  }

  /// Actual number of matches found (may exceed stored count)
  [[nodiscard]] int actualCount(int targetIdx, int queryIdx) const {
    return actualMatchCounts[targetIdx][queryIdx];
  }

  /// Get the matches for a (target, query) pair
  [[nodiscard]] const std::vector<std::vector<int>>& getMatches(int targetIdx, int queryIdx) const {
    return matches[targetIdx][queryIdx];
  }
};

/// Maximum scratch space for boolean expression evaluation per query atom.
/// Complex SMARTS patterns with many OR branches can require significant scratch space.
/// E.g., [C,N,O,S,F,Cl,Br,I,...] with N alternatives needs 2N-1 slots (N leaves + N-1 ORs).
/// 256 supports up to ~128 OR alternatives per atom.
constexpr int kMaxBoolScratchSize = 256;

constexpr std::size_t kMaxTargetAtoms = 128;
constexpr std::size_t kMaxQueryAtoms  = 64;

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_TYPES_H

