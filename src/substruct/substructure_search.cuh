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

#include "flat_bit_vect.h"
#include "molecules.h"
#include "substruct_algos.cuh"
#include "substruct_types.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit

namespace nvMolKit {

// Forward declarations
class MoleculesDevice;

/// Constants for label matrix sizing
constexpr std::size_t kLabelMaxTargetAtoms = 128;
constexpr std::size_t kLabelMaxQueryAtoms  = 64;
constexpr std::size_t kLabelMatrixBits     = kLabelMaxTargetAtoms * kLabelMaxQueryAtoms;

/// Storage type for a single label matrix
using LabelMatrixStorage = FlatBitVect<kLabelMatrixBits>;

/// Number of uint32_t words per label matrix
constexpr std::size_t kLabelMatrixWords = LabelMatrixStorage::kStorageCount;

/**
 * @brief POD view into batch-local device-side substructure match results.
 *
 * Passed to kernels by value. All pointers are device memory.
 * All indexing is batch-local (0 to batchSize-1).
 */
struct SubstructMatchResultsDeviceView {
  int* matchCounts;      ///< [batchSize] actual match count per pair
  int* reportedCounts;   ///< [batchSize] stored match count per pair
  int* pairMatchStarts;  ///< [batchSize + 1] batch-local offset into matchIndices

  int16_t* matchIndices;  ///< Flattened match mappings (batch-local)

  int numQueries;         ///< Total queries (for decoding global pair indices)

  /// Number of query atoms (stride for match indices)
  const int* queryAtomCounts;  ///< [numQueries] atoms per query molecule

  // Pre-allocated per-block overflow buffers (simple chunked DeviceVector)
  PartialMatch* overflowBuffer;       ///< Base pointer to overflow storage
  int           overflowEntriesPerBuffer;  ///< Entries per buffer (kOverflowEntriesPerBuffer)
  int           overflowBuffersPerBlock;   ///< 2 for GSI ping-pong, 1 for WUS

  // Per-pair recursive match bits: [batchSize * maxTargetAtoms] with 32 bits per atom
  uint32_t* recursiveMatchBits;  ///< Indexed by batchLocalIdx * maxTargetAtoms + atomIdx
  int       maxTargetAtoms;      ///< Stride for recursiveMatchBits indexing

  // Pre-computed label matrices: [batchSize] label matrices in global memory
  uint32_t* labelMatrixBuffer;  ///< Indexed by batchLocalIdx * kLabelMatrixWords

  /// Get pointer to label matrix for a batch-local pair index
  __device__ __forceinline__ uint32_t* getLabelMatrixPtr(int batchLocalIdx) const {
    return labelMatrixBuffer + batchLocalIdx * kLabelMatrixWords;
  }

  /// Get overflow buffer for this block (GSI ping-pong: bufferIdx 0 or 1)
  __device__ __forceinline__ PartialMatch* getOverflowBuffer(int bufferIdx = 0) const {
    return overflowBuffer + (blockIdx.x * overflowBuffersPerBlock + bufferIdx) * overflowEntriesPerBuffer;
  }

  /// Get overflow buffer capacity (entries per buffer)
  __device__ __forceinline__ int getOverflowCapacity() const {
    return overflowEntriesPerBuffer;
  }

  /// Get recursive match bits for a batch-local (pair, atom) combination
  __device__ __forceinline__ uint32_t getRecursiveMatchBits(int batchLocalIdx, int atomIdx) const {
    return recursiveMatchBits[batchLocalIdx * maxTargetAtoms + atomIdx];
  }

  /// Set a recursive match bit for a batch-local (pair, atom, pattern) combination
  __device__ __forceinline__ void setRecursiveMatchBit(int batchLocalIdx, int atomIdx, int patternId) const {
    if (patternId < 32) {
      atomicOr(&recursiveMatchBits[batchLocalIdx * maxTargetAtoms + atomIdx], 1u << patternId);
    }
  }
};

namespace detail {

/**
 * @brief Internal: Perform batch substructure matching on GPU.
 *
 * Prefer using getSubstructMatches(const std::vector<const RDKit::ROMol*>&, ...) instead.
 *
 * @param targetSortOrder If non-empty, maps sorted index -> original index for targets
 * @param querySortOrder If non-empty, maps sorted index -> original index for queries
 */
void getSubstructMatches(MoleculesDevice&              targetsDevice,
                         const MoleculesDevice&        queriesDevice,
                         const MoleculesHost&          targetsHost,
                         const MoleculesHost&          queriesHost,
                         SubstructSearchResults&       results,
                         SubstructAlgorithm            algorithm,
                         cudaStream_t                  stream,
                         const SubstructSearchConfig&  config            = SubstructSearchConfig{},
                         const std::vector<int>&       targetSortOrder   = {},
                         const std::vector<int>&       querySortOrder    = {});

}  // namespace detail

/**
 * @brief Perform batch substructure matching on GPU.
 *
 * Molecules are sorted by atom count (largest first) for improved GPU efficiency,
 * and results are returned in the original input order.
 *
 * @param targets Vector of target molecule pointers
 * @param queries Vector of query molecule pointers (typically from SMARTS)
 * @param results Output: matches[target][query][match] = vector of target atom indices
 * @param algorithm Algorithm to use for matching
 * @param stream CUDA stream for async operations
 * @param config Execution configuration (threading, batching). Defaults to single-threaded.
 */
void getSubstructMatches(const std::vector<const RDKit::ROMol*>& targets,
                         const std::vector<const RDKit::ROMol*>& queries,
                         SubstructSearchResults&                 results,
                         SubstructAlgorithm                      algorithm,
                         cudaStream_t                            stream,
                         const SubstructSearchConfig&            config = SubstructSearchConfig{});

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCTURE_SEARCH_CUH
