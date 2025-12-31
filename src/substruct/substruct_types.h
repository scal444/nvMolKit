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
#include <unordered_map>
#include <vector>

namespace nvMolKit {

constexpr int kThreadsPerBlock = 256;

/**
 * @brief Algorithm choice for substructure matching.
 */
enum class SubstructAlgorithm {
  VF2,          ///< VF2 iterative stack-based DFS
  GSI,          ///< GSI-inspired BFS level-by-level join
  WarpUnified   ///< Novel warp-collective BFS search
};

/**
 * @brief Configuration for substructure search execution.
 *
 * Controls threading and batching behavior. Default configuration is single-threaded
 * for deterministic behavior and simpler debugging.
 */
struct SubstructSearchConfig {
  int  batchSize           = 1024;   ///< Number of (target, query) pairs per GPU batch
  int  workerThreads       = 1;      ///< Number of GPU runner threads (1 = single-threaded)
  int  preprocessorThreads = 0;      ///< Number of CPU preprocessor threads (0 = inline preprocessing)
  int  slotsPerRunner      = 3;      ///< Slots per runner for inline mode (1-8, higher = more overlap)
  bool presort             = true;   ///< Sort molecules by atom count (largest first) for GPU efficiency
};

/**
 * @brief Accumulated results from substructure matching.
 *
 * Uses sparse storage: only pairs with matches are stored.
 * Memory is allocated proportional to actual matches found.
 */
struct SubstructSearchResults {
  /// Sparse storage: pairIndex -> vector of matches
  /// Each match is a vector<int> of target atom indices (one per query atom)
  std::unordered_map<int, std::vector<std::vector<int>>> matches;

  /// Sparse storage: pairIndex -> actual match count (may exceed stored if capped)
  std::unordered_map<int, int> actualMatchCounts;

  int numTargets = 0;
  int numQueries = 0;

  void resize(int nTargets, int nQueries) {
    numTargets = nTargets;
    numQueries = nQueries;
    matches.clear();
    actualMatchCounts.clear();
    // Reserve based on expected sparsity (assume ~10% of pairs have matches)
    const size_t expectedPairs = static_cast<size_t>(nTargets) * nQueries / 10 + 1;
    matches.reserve(expectedPairs);
    actualMatchCounts.reserve(expectedPairs);
  }

  /// Compute flat pair index
  [[nodiscard]] int pairIndex(int targetIdx, int queryIdx) const {
    return targetIdx * numQueries + queryIdx;
  }

  /// Check if pair had more matches than could be stored
  [[nodiscard]] bool hasOverflow(int targetIdx, int queryIdx) const {
    const int idx = pairIndex(targetIdx, queryIdx);
    auto countIt = actualMatchCounts.find(idx);
    if (countIt == actualMatchCounts.end()) return false;
    auto matchIt = matches.find(idx);
    if (matchIt == matches.end()) return countIt->second > 0;
    return countIt->second > static_cast<int>(matchIt->second.size());
  }

  /// Number of matches stored for this pair
  [[nodiscard]] int matchCount(int targetIdx, int queryIdx) const {
    auto it = matches.find(pairIndex(targetIdx, queryIdx));
    return (it != matches.end()) ? static_cast<int>(it->second.size()) : 0;
  }

  /// Actual number of matches found (may exceed stored count)
  [[nodiscard]] int actualCount(int targetIdx, int queryIdx) const {
    auto it = actualMatchCounts.find(pairIndex(targetIdx, queryIdx));
    return (it != actualMatchCounts.end()) ? it->second : 0;
  }

  /// Get the matches for a (target, query) pair (returns empty if none)
  [[nodiscard]] const std::vector<std::vector<int>>& getMatches(int targetIdx, int queryIdx) const {
    static const std::vector<std::vector<int>> empty;
    auto it = matches.find(pairIndex(targetIdx, queryIdx));
    return (it != matches.end()) ? it->second : empty;
  }

  /// Mutable access to matches (creates entry if needed)
  std::vector<std::vector<int>>& getMatchesMut(int targetIdx, int queryIdx) {
    return matches[pairIndex(targetIdx, queryIdx)];
  }

  /// Add to actual match count
  void addActualCount(int targetIdx, int queryIdx, int count) {
    actualMatchCounts[pairIndex(targetIdx, queryIdx)] += count;
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

