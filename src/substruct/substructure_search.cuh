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

  // Pre-allocated per-block overflow buffers (simple chunked DeviceVector)
  PartialMatch* overflowBuffer;       ///< Base pointer to overflow storage
  int           overflowEntriesPerBuffer;  ///< Entries per buffer (kOverflowEntriesPerBuffer)
  int           overflowBuffersPerBlock;   ///< 2 for GSI ping-pong, 1 for WUS

  // Per-pair recursive match bits: [numPairs * maxTargetAtoms] with 32 bits per atom
  uint32_t* recursiveMatchBits;  ///< Indexed by pairIdx * maxTargetAtoms + atomIdx
  int       maxTargetAtoms;      ///< Stride for recursiveMatchBits indexing

  __device__ __forceinline__ int pairIndex(int targetIdx, int queryIdx) const {
    return targetIdx * numQueries + queryIdx;
  }

  /// Get overflow buffer for this block (GSI ping-pong: bufferIdx 0 or 1)
  __device__ __forceinline__ PartialMatch* getOverflowBuffer(int bufferIdx = 0) const {
    return overflowBuffer + (blockIdx.x * overflowBuffersPerBlock + bufferIdx) * overflowEntriesPerBuffer;
  }

  /// Get overflow buffer capacity (entries per buffer)
  __device__ __forceinline__ int getOverflowCapacity() const {
    return overflowEntriesPerBuffer;
  }

  /// Get recursive match bits for a specific (pair, atom) combination
  __device__ __forceinline__ uint32_t getRecursiveMatchBits(int pairIdx, int atomIdx) const {
    return recursiveMatchBits[pairIdx * maxTargetAtoms + atomIdx];
  }

  /// Set a recursive match bit for a specific (pair, atom, pattern) combination
  __device__ __forceinline__ void setRecursiveMatchBit(int pairIdx, int atomIdx, int patternId) const {
    if (patternId < 32) {
      atomicOr(&recursiveMatchBits[pairIdx * maxTargetAtoms + atomIdx], 1u << patternId);
    }
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
   * @param maxMatchesPerPairVec Maximum matches to store per pair (typically target atom count)
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

  /**
   * @brief Allocate overflow buffers for a batch size.
   *
   * @param batchSize Number of blocks to allocate overflow for
   * @param numBuffersPerBlock 2 for GSI (ping-pong), 1 for WUS
   */
  void allocateOverflow(int batchSize, int numBuffersPerBlock);

  /**
   * @brief Allocate batch-sized recursive match bits buffer.
   *
   * @param batchSize Number of pairs in the batch
   * @param maxTargetAtoms Max atoms per target (stride for indexing)
   */
  void allocateBatchRecursiveBits(int batchSize, int maxTargetAtoms);

  /**
   * @brief Zero the recursive match bits buffer for a new batch.
   */
  void zeroRecursiveBits();

 private:
  cudaStream_t stream_ = nullptr;

  int numTargets_ = 0;
  int numQueries_ = 0;

  AsyncDeviceVector<int>     matchCounts_;
  AsyncDeviceVector<int>     reportedCounts_;
  AsyncDeviceVector<int>     pairMatchStarts_;
  AsyncDeviceVector<int16_t> matchIndices_;
  AsyncDeviceVector<int>     queryAtomCounts_;

  // Pre-allocated per-block overflow buffers (simple chunked storage)
  AsyncDeviceVector<PartialMatch> overflowBuffer_;
  int overflowBuffersPerBlock_ = 0;

  // Per-pair recursive match bits storage
  AsyncDeviceVector<uint32_t> recursiveMatchBits_;
  int                         maxTargetAtoms_ = 0;

  std::vector<int> hostPairMatchStarts_;
  std::vector<int> hostQueryAtomCounts_;
  int              totalMatchIndices_ = 0;
};

/**
 * @brief Perform batch substructure matching on GPU.
 *
 * Matches each target molecule against each query molecule (all-to-all).
 * Results are stored in device memory and can be copied to host.
 * Processing is divided into batches to limit GPU memory usage.
 *
 * @param targetsDevice Device-resident target molecules (use addToBatch to build)
 * @param queriesDevice Device-resident query molecules (use addQueryToBatch to build)
 * @param targetsHost Host-side target data (for atom counts)
 * @param queriesHost Host-side query data (for atom counts)
 * @param results Output storage (will be allocated)
 * @param algorithm Algorithm to use for matching
 * @param stream CUDA stream for async operations
 * @param batchSize Number of (target, query) pairs to process per batch (default 1024)
 */
void getSubstructMatches(MoleculesDevice&             targetsDevice,
                         const MoleculesDevice&       queriesDevice,
                         const MoleculesHost&         targetsHost,
                         const MoleculesHost&         queriesHost,
                         SubstructMatchResultsDevice& results,
                         SubstructAlgorithm           algorithm,
                         cudaStream_t                 stream,
                         int                          batchSize = 1024);

/**
 * @brief Preprocess recursive SMARTS patterns for a single query within a batch.
 *
 * Uses fused paint mode to directly paint recursive match bits during
 * pattern matching, avoiding overflow issues from intermediate storage.
 *
 * @param targetsDevice Device-resident target molecules
 * @param targetsHost Host-side target data
 * @param recursiveInfo Extracted recursive pattern information for this query
 * @param outputResults The main results buffer where recursiveMatchBits will be written
 * @param mainQueryIdx Index of the main query whose pair storage should be updated
 * @param numQueries Total number of queries (for computing pair indices)
 * @param batchPairOffset Global pair index where current batch starts
 * @param batchSize Number of pairs in this batch
 * @param algorithm Algorithm to use for matching
 * @param stream CUDA stream for async operations
 */
void preprocessRecursiveSmarts(const MoleculesDevice&             targetsDevice,
                               const MoleculesHost&         targetsHost,
                               const RecursivePatternInfo&  recursiveInfo,
                               const SubstructMatchResultsDevice& outputResults,
                               int                          mainQueryIdx,
                               int                          numQueries,
                               int                          batchPairOffset,
                               int                          batchSize,
                               SubstructAlgorithm           algorithm,
                               cudaStream_t                 stream);

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCTURE_SEARCH_CUH

