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

#ifndef NVMOLKIT_SUBSTRUCTURE_SEARCH_CUH
#define NVMOLKIT_SUBSTRUCTURE_SEARCH_CUH

#include <cuda_runtime.h>

#include <vector>

#include "device_vector.h"
#include "molecules.h"

namespace nvMolKit {

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

/**
 * @brief POD view into device-side substructure match results.
 *
 * Passed to kernels by value. All pointers are device memory.
 */
struct SubstructMatchResultsDeviceView {
  int* matchCounts;      ///< [numPairs] actual match count per pair
  int* reportedCounts;   ///< [numPairs] stored match count per pair
  int* pairMatchStarts;  ///< [numPairs + 1] offset into matchIndices

  int16_t* matchIndices;  ///< Flattened match mappings

  int numTargets;
  int numQueries;
  int maxMatchesPerPair;  ///< Buffer capacity per pair

  /// Number of query atoms (stride for match indices)
  const int* queryAtomCounts;  ///< [numQueries] atoms per query molecule

  __device__ __forceinline__ int pairIndex(int targetIdx, int queryIdx) const {
    return targetIdx * numQueries + queryIdx;
  }
};

/**
 * @brief Device-side storage for substructure match results.
 *
 * Owns device memory and provides views for kernel access.
 */
class SubstructMatchResultsDevice {
 public:
  SubstructMatchResultsDevice() = default;
  explicit SubstructMatchResultsDevice(cudaStream_t stream) : stream_(stream) {}

  /**
   * @brief Allocate output buffers for batch matching.
   *
   * Buffer sizing: each (target, query) pair gets space for up to
   * maxMatchesPerPair matches. Each match requires numQueryAtoms int16_t values.
   *
   * @param numTargets Number of target molecules
   * @param numQueries Number of query molecules
   * @param queryAtomCounts Number of atoms in each query molecule
   * @param maxMatchesPerPair Maximum matches to store per pair (typically target atom count)
   */
  void allocate(int                     numTargets,
                int                     numQueries,
                const std::vector<int>& queryAtomCounts,
                const std::vector<int>& maxMatchesPerPairVec);

  /**
   * @brief Copy results from device to host.
   */
  void copyToHost(SubstructMatchResultsHost& host) const;

  /**
   * @brief Get a view suitable for passing to CUDA kernels.
   */
  [[nodiscard]] SubstructMatchResultsDeviceView view() const;

  void setStream(cudaStream_t stream);

 private:
  cudaStream_t stream_ = nullptr;

  int numTargets_ = 0;
  int numQueries_ = 0;

  AsyncDeviceVector<int>     matchCounts_;
  AsyncDeviceVector<int>     reportedCounts_;
  AsyncDeviceVector<int>     pairMatchStarts_;
  AsyncDeviceVector<int16_t> matchIndices_;
  AsyncDeviceVector<int>     queryAtomCounts_;

  std::vector<int> hostPairMatchStarts_;
  std::vector<int> hostQueryAtomCounts_;
  int              totalMatchIndices_ = 0;
};

/**
 * @brief Perform batch substructure matching on GPU.
 *
 * Matches each target molecule against each query molecule (all-to-all).
 * Results are stored in device memory and can be copied to host.
 *
 * @param targetsDevice Device-resident target molecules (use addToBatch to build)
 * @param queriesDevice Device-resident query molecules (use addQueryToBatch to build)
 * @param targetsHost Host-side target data (for atom counts)
 * @param queriesHost Host-side query data (for atom counts)
 * @param results Output storage (will be allocated)
 * @param stream CUDA stream for async operations
 */
void getSubstructMatches(const MoleculesDevice&       targetsDevice,
                         const MoleculesDevice&       queriesDevice,
                         const MoleculesHost&         targetsHost,
                         const MoleculesHost&         queriesHost,
                         SubstructMatchResultsDevice& results,
                         cudaStream_t                 stream);

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCTURE_SEARCH_CUH

