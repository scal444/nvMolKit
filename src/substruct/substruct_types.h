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
  GSI           ///< GSI-inspired BFS level-by-level join
};

/**
 * @brief Configuration for substructure search execution.
 *
 * Controls threading, batching, and multi-GPU behavior. Default configuration is
 * single-threaded for deterministic behavior and simpler debugging.
 *
 * Multi-GPU mode: When gpuIds is non-empty, work is distributed across the specified
 * GPUs using round-robin assignment. Each GPU gets workerThreads workers, so total
 * worker threads = workerThreads * gpuIds.size().
 */
struct SubstructSearchConfig {
  int  batchSize           = 1024;   ///< Number of (target, query) pairs per GPU batch
  int  workerThreads       = 1;      ///< Number of GPU runner threads per GPU
  int  preprocessingThreads = 0;     ///< CPU threads for input preprocessing (0 = single-threaded)
  int  slotsPerRunner      = 3;      ///< Batch slots per runner thread (1-8, higher = more overlap)
  bool presort             = true;   ///< Sort molecules by atom count (largest first) for GPU efficiency
  std::vector<int> gpuIds;           ///< GPU device IDs to use (empty = current device only)
  int  maxMatches          = -1;     ///< Max matches to store per pair (-1 = no limit, 0 = count only)
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

/**
 * @brief Results from hasSubstructMatch - boolean per (target, query) pair.
 *
 * More efficient than full match enumeration when only existence is needed.
 * Uses uint8_t instead of bool to avoid std::vector<bool> specialization issues.
 */
struct HasSubstructMatchResults {
  std::vector<uint8_t> hasMatch;  ///< Flattened [target * numQueries + query], 0=false, non-zero=true
  int numTargets = 0;
  int numQueries = 0;

  void resize(int nTargets, int nQueries) {
    numTargets = nTargets;
    numQueries = nQueries;
    hasMatch.assign(static_cast<size_t>(nTargets) * nQueries, 0);
  }

  /// Compute flat pair index
  [[nodiscard]] int pairIndex(int targetIdx, int queryIdx) const {
    return targetIdx * numQueries + queryIdx;
  }

  /// Check if target contains query as substructure
  [[nodiscard]] bool matches(int targetIdx, int queryIdx) const {
    return hasMatch[pairIndex(targetIdx, queryIdx)] != 0;
  }

  /// Set match result for a pair
  void setMatch(int targetIdx, int queryIdx, bool value) {
    hasMatch[pairIndex(targetIdx, queryIdx)] = value ? 1 : 0;
  }
};

/// Maximum scratch space for boolean expression evaluation per query atom.
/// Complex SMARTS patterns with many OR branches can require significant scratch space.
/// E.g., [C,N,O,S,F,Cl,Br,I,...] with N alternatives needs 2N-1 slots (N leaves + N-1 ORs).
/// 256 supports up to ~128 OR alternatives per atom.
constexpr int kMaxBoolScratchSize = 256;

constexpr std::size_t kMaxTargetAtoms = 128;
constexpr std::size_t kMaxQueryAtoms  = 64;

/**
 * @brief Entry representing a (target, query) pair that needs RDKit fallback processing.
 *
 * Used when GPU processing cannot handle a pair, either due to:
 * - Target molecule exceeding kMaxTargetAtoms
 * - Output buffer overflow during GPU matching
 */
struct RDKitFallbackEntry {
  int originalTargetIdx;  ///< Index in the original input targets vector
  int originalQueryIdx;   ///< Index in the original input queries vector

  bool operator<(const RDKitFallbackEntry& other) const {
    if (originalTargetIdx != other.originalTargetIdx) {
      return originalTargetIdx < other.originalTargetIdx;
    }
    return originalQueryIdx < other.originalQueryIdx;
  }
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_TYPES_H

