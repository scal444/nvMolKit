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
#include "substruct_algos.cuh"
#include "substruct_types.h"

namespace nvMolKit {

// SubstructAlgorithm and SubstructMatchResultsHost are defined in substruct_types.h

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

  // Global memory overflow buffers for search algorithms
  PartialMatch* overflowBuffer;     ///< [overflowBatchSize * overflowSize * 2] ping-pong overflow
  int           overflowSize;       ///< Overflow capacity per pair (per buffer)
  int           overflowBatchSize;  ///< Number of pairs overflow is allocated for
  int           pairOffset;         ///< Offset for current batch (blockIdx.x + pairOffset = actual pair)

  __device__ __forceinline__ int pairIndex(int targetIdx, int queryIdx) const {
    return targetIdx * numQueries + queryIdx;
  }

  /// Get overflow buffer for a block within current batch (ping-pong: bufferIdx 0 or 1)
  __device__ __forceinline__ PartialMatch* getOverflowBuffer(int blockIdxInBatch, int bufferIdx) const {
    return overflowBuffer + (blockIdxInBatch * 2 + bufferIdx) * overflowSize;
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

  // Global memory overflow buffers for search algorithms
  AsyncDeviceVector<PartialMatch> overflowBuffer_;
  int                             overflowSize_ = 0;

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
 * @param algorithm Algorithm to use for matching
 * @param stream CUDA stream for async operations
 */
void getSubstructMatches(MoleculesDevice&             targetsDevice,
                         const MoleculesDevice&       queriesDevice,
                         const MoleculesHost&         targetsHost,
                         const MoleculesHost&         queriesHost,
                         SubstructMatchResultsDevice& results,
                         SubstructAlgorithm           algorithm,
                         cudaStream_t                 stream);

/**
 * @brief Paint recursive SMARTS match bits onto target atom labels.
 *
 * Given substructure match results from recursive pattern matching, sets the
 * corresponding recursive match bits on each target atom that matched.
 *
 * @param targetsDevice Device-resident target molecules (will be modified)
 * @param results Match results from running substruct match on recursive patterns
 * @param patternIds Vector mapping query index to pattern ID (for each pattern query)
 * @param stream CUDA stream for async operations
 */
void paintRecursiveMatchBits(MoleculesDevice&                   targetsDevice,
                             const SubstructMatchResultsDevice& results,
                             const std::vector<int>&            patternIds,
                             cudaStream_t                       stream);

/**
 * @brief Preprocess recursive SMARTS patterns for a batch of queries.
 *
 * For queries containing recursive SMARTS ($(...)), this function:
 * 1. Extracts all recursive patterns from the queries
 * 2. Runs substruct matching for each pattern against all targets
 * 3. Paints the recursive match bits on target atoms
 *
 * Call this before getSubstructMatches for queries with recursive SMARTS.
 *
 * @param targetsDevice Device-resident target molecules (will be modified)
 * @param targetsHost Host-side target data
 * @param recursiveInfo Extracted recursive pattern information
 * @param stream CUDA stream for async operations
 */
void preprocessRecursiveSmarts(MoleculesDevice&             targetsDevice,
                               const MoleculesHost&         targetsHost,
                               const RecursivePatternInfo&  recursiveInfo,
                               cudaStream_t                 stream);

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCTURE_SEARCH_CUH

