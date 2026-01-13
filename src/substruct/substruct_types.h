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
 * Controls threading, batching, and multi-GPU behavior. Default configuration uses
 * autoselect (-1) which determines optimal thread counts based on hardware.
 *
 * Multi-GPU mode: When gpuIds is non-empty, work is distributed across the specified
 * GPUs using round-robin assignment. Each GPU gets workerThreads workers, so total
 * worker threads = workerThreads * gpuIds.size().
 *
 * Threading autoselect (-1 for any thread count):
 * - preprocessingThreads: uses hardware_concurrency
 * - workerThreads: min(4, hardware_concurrency / numGpus)
 *
 * RDKit fallback (for oversized molecules or overflow) is processed opportunistically
 * by preprocessing threads while waiting for GPU work to complete.
 */
struct SubstructSearchConfig {
  int  batchSize            = 1024;  ///< Number of (target, query) pairs per GPU batch
  int  macroBatchMinibatches = 500;  ///< Number of mini-batches per macro-batch for target preprocessing overlap (<=1 disables macro batching)
  int  workerThreads        = -1;    ///< GPU runner threads per GPU (-1 = autoselect)
  int  preprocessingThreads = -1;    ///< CPU threads for preprocessing and opportunistic RDKit fallback (-1 = autoselect)
  int  executorsPerRunner   = -1;    ///< GPU executors per runner thread (-1 = auto: 3 for single runner, 2 otherwise)
  bool presort              = true;  ///< Sort molecules by atom count (largest first) for GPU efficiency
  std::vector<int> gpuIds;           ///< GPU device IDs to use (empty = current device only)
  int  maxMatches           = 0;     ///< Max matches per pair (0 = unlimited, like RDKit)
  bool uniquify             = false; ///< Remove duplicate matches differing only in atom enumeration order
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

  int numTargets = 0;
  int numQueries = 0;

  void resize(int nTargets, int nQueries) {
    numTargets = nTargets;
    numQueries = nQueries;
    matches.clear();
    // Reserve based on expected sparsity (assume ~10% of pairs have matches)
    const size_t expectedPairs = static_cast<size_t>(nTargets) * nQueries / 10 + 1;
    matches.reserve(expectedPairs);
  }

  /// Compute flat pair index
  [[nodiscard]] int pairIndex(int targetIdx, int queryIdx) const {
    return targetIdx * numQueries + queryIdx;
  }

  /// Number of matches for this pair
  [[nodiscard]] int matchCount(int targetIdx, int queryIdx) const {
    auto it = matches.find(pairIndex(targetIdx, queryIdx));
    return (it != matches.end()) ? static_cast<int>(it->second.size()) : 0;
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

/// Total bits in a label matrix (target × query)
constexpr std::size_t kLabelMatrixBits = kMaxTargetAtoms * kMaxQueryAtoms;

/// Number of 32-bit words per label matrix
constexpr std::size_t kLabelMatrixWords = kLabelMatrixBits / 32;

// =============================================================================
// Partial Match Structure (for GSI algorithm queue)
// =============================================================================

/**
 * @brief Partial match for BFS-style algorithms.
 *
 * @tparam MaxQueryAtoms Maximum query atoms for array sizing
 *
 * Represents a partial mapping from query atoms to target atoms.
 * Stored compactly for queue-based BFS exploration.
 *
 * Complete matches are never stored in the queue, so we only need
 * MaxQueryAtoms - 1 slots (matching atoms 0 through numQueryAtoms-2).
 */
template <std::size_t MaxQueryAtoms = kMaxQueryAtoms>
struct PartialMatchT {
  static constexpr std::size_t kMaxQueryAtomsValue = MaxQueryAtoms;
  int8_t mapping[MaxQueryAtoms - 1];  ///< mapping[q] = target atom (only [0..nextQueryAtom-1] valid)
  int8_t nextQueryAtom;               ///< Next query atom to extend (also serves as depth)
};

/// Type alias for max-sized partial match (backward compatibility)
using PartialMatch = PartialMatchT<kMaxQueryAtoms>;
static_assert(sizeof(PartialMatch) == kMaxQueryAtoms, "PartialMatch must be kMaxQueryAtoms bytes");

// =============================================================================
// Template Configuration for Kernel Dispatch
// =============================================================================

/**
 * @brief Enumeration of valid template configurations.
 *
 * Each configuration specifies (MaxTargetAtoms, MaxQueryAtoms, MaxBondsPerAtom).
 * Valid combinations require Target >= Query. Total: 24 configurations.
 *
 * Naming: Config_T{targets}_Q{queries}_B{bonds}
 */
enum class SubstructTemplateConfig : uint8_t {
  // Target 32, Query 16
  Config_T32_Q16_B4 = 0,
  Config_T32_Q16_B6,
  Config_T32_Q16_B8,
  // Target 32, Query 32
  Config_T32_Q32_B4,
  Config_T32_Q32_B6,
  Config_T32_Q32_B8,
  // Target 64, Query 16
  Config_T64_Q16_B4,
  Config_T64_Q16_B6,
  Config_T64_Q16_B8,
  // Target 64, Query 32
  Config_T64_Q32_B4,
  Config_T64_Q32_B6,
  Config_T64_Q32_B8,
  // Target 64, Query 64
  Config_T64_Q64_B4,
  Config_T64_Q64_B6,
  Config_T64_Q64_B8,
  // Target 128, Query 16
  Config_T128_Q16_B4,
  Config_T128_Q16_B6,
  Config_T128_Q16_B8,
  // Target 128, Query 32
  Config_T128_Q32_B4,
  Config_T128_Q32_B6,
  Config_T128_Q32_B8,
  // Target 128, Query 64
  Config_T128_Q64_B4,
  Config_T128_Q64_B6,
  Config_T128_Q64_B8,

  NumConfigs  ///< Total number of configurations (24)
};

constexpr int kNumTemplateConfigs = static_cast<int>(SubstructTemplateConfig::NumConfigs);

/**
 * @brief Compile-time properties for a template configuration.
 */
struct TemplateConfigProperties {
  int maxTargetAtoms;
  int maxQueryAtoms;
  int maxBondsPerAtom;
  std::size_t labelMatrixBits;
  std::size_t labelMatrixWords;
};

/**
 * @brief Get properties for a template configuration at compile time.
 */
constexpr TemplateConfigProperties getTemplateConfigProperties(SubstructTemplateConfig config) {
  switch (config) {
    case SubstructTemplateConfig::Config_T32_Q16_B4:  return {32, 16, 4, 32*16, 32*16/32};
    case SubstructTemplateConfig::Config_T32_Q16_B6:  return {32, 16, 6, 32*16, 32*16/32};
    case SubstructTemplateConfig::Config_T32_Q16_B8:  return {32, 16, 8, 32*16, 32*16/32};
    case SubstructTemplateConfig::Config_T32_Q32_B4:  return {32, 32, 4, 32*32, 32*32/32};
    case SubstructTemplateConfig::Config_T32_Q32_B6:  return {32, 32, 6, 32*32, 32*32/32};
    case SubstructTemplateConfig::Config_T32_Q32_B8:  return {32, 32, 8, 32*32, 32*32/32};
    case SubstructTemplateConfig::Config_T64_Q16_B4:  return {64, 16, 4, 64*16, 64*16/32};
    case SubstructTemplateConfig::Config_T64_Q16_B6:  return {64, 16, 6, 64*16, 64*16/32};
    case SubstructTemplateConfig::Config_T64_Q16_B8:  return {64, 16, 8, 64*16, 64*16/32};
    case SubstructTemplateConfig::Config_T64_Q32_B4:  return {64, 32, 4, 64*32, 64*32/32};
    case SubstructTemplateConfig::Config_T64_Q32_B6:  return {64, 32, 6, 64*32, 64*32/32};
    case SubstructTemplateConfig::Config_T64_Q32_B8:  return {64, 32, 8, 64*32, 64*32/32};
    case SubstructTemplateConfig::Config_T64_Q64_B4:  return {64, 64, 4, 64*64, 64*64/32};
    case SubstructTemplateConfig::Config_T64_Q64_B6:  return {64, 64, 6, 64*64, 64*64/32};
    case SubstructTemplateConfig::Config_T64_Q64_B8:  return {64, 64, 8, 64*64, 64*64/32};
    case SubstructTemplateConfig::Config_T128_Q16_B4: return {128, 16, 4, 128*16, 128*16/32};
    case SubstructTemplateConfig::Config_T128_Q16_B6: return {128, 16, 6, 128*16, 128*16/32};
    case SubstructTemplateConfig::Config_T128_Q16_B8: return {128, 16, 8, 128*16, 128*16/32};
    case SubstructTemplateConfig::Config_T128_Q32_B4: return {128, 32, 4, 128*32, 128*32/32};
    case SubstructTemplateConfig::Config_T128_Q32_B6: return {128, 32, 6, 128*32, 128*32/32};
    case SubstructTemplateConfig::Config_T128_Q32_B8: return {128, 32, 8, 128*32, 128*32/32};
    case SubstructTemplateConfig::Config_T128_Q64_B4: return {128, 64, 4, 128*64, 128*64/32};
    case SubstructTemplateConfig::Config_T128_Q64_B6: return {128, 64, 6, 128*64, 128*64/32};
    case SubstructTemplateConfig::Config_T128_Q64_B8: return {128, 64, 8, 128*64, 128*64/32};
    default: return {128, 64, 8, 128*64, 128*64/32};  // fallback to max
  }
}

/**
 * @brief Compute label matrix words for a given (targets, queries) pair.
 */
constexpr std::size_t computeLabelMatrixWords(int maxTargetAtoms, int maxQueryAtoms) {
  return static_cast<std::size_t>(maxTargetAtoms) * maxQueryAtoms / 32;
}

/**
 * @brief Select the smallest template configuration that fits the given sizes.
 *
 * Finds the configuration with the smallest resource requirements (label matrix size,
 * partial match size) that can handle molecules with the specified maximums.
 *
 * @param maxTargetAtoms Maximum target atoms in the batch (1-128)
 * @param maxQueryAtoms Maximum query atoms in the batch (1-64)
 * @param maxBondsPerAtom Maximum bonds per atom in the batch (1-8)
 * @return The smallest fitting template configuration
 */
inline SubstructTemplateConfig selectTemplateConfig(int maxTargetAtoms, int maxQueryAtoms, int maxBondsPerAtom) {
  // Clamp maxBondsPerAtom to valid template values: 4, 6, or 8
  int bondConfig;
  if (maxBondsPerAtom <= 4) {
    bondConfig = 4;
  } else if (maxBondsPerAtom <= 6) {
    bondConfig = 6;
  } else {
    bondConfig = 8;
  }

  // Select target size tier
  int targetTier;
  if (maxTargetAtoms <= 32) {
    targetTier = 32;
  } else if (maxTargetAtoms <= 64) {
    targetTier = 64;
  } else {
    targetTier = 128;
  }

  // Select query size tier (must be <= target tier)
  int queryTier;
  if (maxQueryAtoms <= 16) {
    queryTier = 16;
  } else if (maxQueryAtoms <= 32) {
    queryTier = 32;
  } else {
    queryTier = 64;
  }

  // Ensure query <= target
  if (queryTier > targetTier) {
    queryTier = targetTier;
  }

  // Map to config enum
  if (targetTier == 32) {
    if (queryTier == 16) {
      if (bondConfig == 4) return SubstructTemplateConfig::Config_T32_Q16_B4;
      if (bondConfig == 6) return SubstructTemplateConfig::Config_T32_Q16_B6;
      return SubstructTemplateConfig::Config_T32_Q16_B8;
    } else {  // queryTier == 32
      if (bondConfig == 4) return SubstructTemplateConfig::Config_T32_Q32_B4;
      if (bondConfig == 6) return SubstructTemplateConfig::Config_T32_Q32_B6;
      return SubstructTemplateConfig::Config_T32_Q32_B8;
    }
  } else if (targetTier == 64) {
    if (queryTier == 16) {
      if (bondConfig == 4) return SubstructTemplateConfig::Config_T64_Q16_B4;
      if (bondConfig == 6) return SubstructTemplateConfig::Config_T64_Q16_B6;
      return SubstructTemplateConfig::Config_T64_Q16_B8;
    } else if (queryTier == 32) {
      if (bondConfig == 4) return SubstructTemplateConfig::Config_T64_Q32_B4;
      if (bondConfig == 6) return SubstructTemplateConfig::Config_T64_Q32_B6;
      return SubstructTemplateConfig::Config_T64_Q32_B8;
    } else {  // queryTier == 64
      if (bondConfig == 4) return SubstructTemplateConfig::Config_T64_Q64_B4;
      if (bondConfig == 6) return SubstructTemplateConfig::Config_T64_Q64_B6;
      return SubstructTemplateConfig::Config_T64_Q64_B8;
    }
  } else {  // targetTier == 128
    if (queryTier == 16) {
      if (bondConfig == 4) return SubstructTemplateConfig::Config_T128_Q16_B4;
      if (bondConfig == 6) return SubstructTemplateConfig::Config_T128_Q16_B6;
      return SubstructTemplateConfig::Config_T128_Q16_B8;
    } else if (queryTier == 32) {
      if (bondConfig == 4) return SubstructTemplateConfig::Config_T128_Q32_B4;
      if (bondConfig == 6) return SubstructTemplateConfig::Config_T128_Q32_B6;
      return SubstructTemplateConfig::Config_T128_Q32_B8;
    } else {  // queryTier == 64
      if (bondConfig == 4) return SubstructTemplateConfig::Config_T128_Q64_B4;
      if (bondConfig == 6) return SubstructTemplateConfig::Config_T128_Q64_B6;
      return SubstructTemplateConfig::Config_T128_Q64_B8;
    }
  }
}

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
