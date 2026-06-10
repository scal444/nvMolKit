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

#ifndef MCS_COMMON_MCS_TYPES_CUH
#define MCS_COMMON_MCS_TYPES_CUH

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef __CUDACC__
#define MCS_HOST_DEVICE __host__ __device__
#else
#define MCS_HOST_DEVICE
#endif

namespace mcs {

// ---------------------------------------------------------------------------
// Search / objective enums and spec
// ---------------------------------------------------------------------------

enum class SearchKind {
  kMCIS,
  kMCES,
};

enum class ObjectiveKind {
  kMaxVertices,
  kMaxEdges,
};

struct SearchSpec {
  SearchKind kind = SearchKind::kMCIS;
  ObjectiveKind objective = ObjectiveKind::kMaxVertices;
  /// When true, restrict the search to connected common subgraphs: once the
  /// partial mapping is non-empty, only label classes carrying the
  /// isAdjacent flag (i.e. inherited from the adjacent side of a prior
  /// split) contribute to bounds and are eligible for branching.  This
  /// matches the rdFMCS default of returning a single connected fragment.
  bool requireConnected = false;
};

// ---------------------------------------------------------------------------
// Score types -- used by the host-side winner selection to compare results.
// ---------------------------------------------------------------------------

struct Score {
  int primary = 0;
  int secondary = 0;

  bool operator>(const Score& other) const {
    return primary > other.primary ||
           (primary == other.primary && secondary > other.secondary);
  }
};

// ---------------------------------------------------------------------------
// NullLabel -- tag type indicating "no labels".
// ---------------------------------------------------------------------------

struct NullLabel {};

template<typename T>
inline constexpr bool is_null_label_v = std::is_same_v<T, NullLabel>;

// ---------------------------------------------------------------------------
// NullCompat -- generic always-true compatibility functor.
// ---------------------------------------------------------------------------

struct NullCompat {
  template<typename... Args>
  MCS_HOST_DEVICE bool operator()(Args...) const { return true; }
};

/**
 * @brief CSR (Compressed Sparse Row) graph representation.
 *
 * Stores an undirected graph where each undirected edge (u,v) appears twice
 * in the adjacency structure: once under u and once under v.
 */
struct Graph {
  int              numVertices = 0;
  int              numEdges    = 0;  ///< Count of undirected edges.
  std::vector<size_t> rowOffsets;       ///< Size = numVertices + 1.
  std::vector<size_t> colIndices;       ///< Size = 2 * numEdges (symmetric).
};

/**
 * @brief Build a CSR Graph from a vertex count and undirected edge list.
 */
Graph buildGraphFromEdges(size_t numVertices,
                          const std::vector<std::pair<size_t, size_t>>& edges);

/**
 * @brief Result of a maximum common substructure computation.
 *
 * The ``whitneyAmbiguous`` flag signals that the MCES (line-graph
 * reduction) search returned a size-N LG-MCIS whose matched LG-vertex
 * set cannot be decoded to a consistent vertex-level correspondence
 * on BOTH sides simultaneously.  When this happens the decoder in
 * ``mcesResultFromWinner`` drops the inconsistent pairs via an
 * iterative MAX-2SAT approximation, so the returned
 * ``numCommonEdges`` is the size of the largest Whitney-consistent
 * subset of the search's match rather than the search's
 * ``winner.nMappings``.  The returned match IS a valid common edge
 * subgraph of both inputs, but it may be smaller than the true MCES
 * (which a Whitney-aware search would have found).
 *
 * This manifests almost exclusively on unlabeled inputs -- under
 * non-trivial vertex/edge labels the per-pair canonicalisation of
 * (edge-label, sorted endpoint-vertex-labels) in
 * ``buildLabeledLineGraphPair`` eliminates the structural ambiguity
 * in practice.  Callers that need a guaranteed-exact MCES on
 * unlabeled inputs should check this flag and fall back to a
 * Whitney-aware solver on flagged pairs.
 */
struct MCSResult {
  int numCommonVertices = 0;
  int numCommonEdges    = 0;
  bool timedOut         = false;
  bool killed           = false;
  bool overflowed       = false;
  /// MCES-only.  True iff the line-graph MCIS result contained pairs
  /// with contradictory orientation constraints and at least one pair
  /// had to be dropped by the decoder.  ``numCommonEdges`` is then
  /// the max Whitney-consistent subset size, which is a valid lower
  /// bound on the true MCES but not guaranteed equal to it.
  bool whitneyAmbiguous = false;

  /// Vertex mappings: mappingA[i] <-> mappingB[i] in the common subgraph.
  std::vector<size_t> mappingA;
  std::vector<size_t> mappingB;

  /// Edge mappings (populated for MCES mode):
  /// edgeMappingA[i] = (u, v) in graphA, edgeMappingB[i] = (u, v) in graphB.
  std::vector<std::pair<size_t, size_t>> edgeMappingA;
  std::vector<std::pair<size_t, size_t>> edgeMappingB;
};

#ifdef MCS_PROFILE_WARPS
/**
 * @brief Per-warp profiling counters collected during kernel execution.
 *
 * One instance per warp, written by lane 0 only.
 */
struct WarpProfileCounters {
  long long dfsSteps;
  long long stackPushOps;
  long long stackPushItems;
  long long stackPopOps;
  long long stackPopFails;
  long long pushOverflows;
  long long steals;
  long long activeClocks;
  long long idleClocks;
  long long idleIters;
  int       peakStackTop;
  long long stackTopSum;
  long long loopIters;
  long long pathsExhausted;
};
#endif

}  // namespace mcs

#endif  // MCS_COMMON_MCS_TYPES_CUH
