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

#include <array>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "device.h"
#include "device_vector.h"
#include "flat_bit_vect.h"
#include "molecules.h"
#include "pinned_buffer_pool.h"
#include "substruct_algos.cuh"
#include "substruct_types.h"

namespace nvMolKit {

// SubstructAlgorithm and SubstructSearchResults are defined in substruct_types.h

/// Constants for label matrix sizing (must match substructure_search.cu)
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

/**
 * @brief Batch-local device-side storage for substructure match results.
 *
 * Owns device memory for a single batch and provides views for kernel access.
 * Results are copied back to host after each batch and accumulated.
 */
class BatchResultsDevice {
 public:
  BatchResultsDevice() = default;
  explicit BatchResultsDevice(cudaStream_t stream) : stream_(stream) { setStream(stream); }

  /**
   * @brief Allocate batch-local buffers for a specific batch.
   *
   * @param batchSize Number of pairs in this batch
   * @param batchPairMatchStarts Batch-local offsets into matchIndices [batchSize + 1]
   * @param totalBatchMatchIndices Total match indices capacity for this batch
   * @param numQueries Total number of queries (for kernel view)
   * @param maxTargetAtoms Max atoms per target (stride for recursiveMatchBits)
   * @param numBuffersPerBlock Overflow buffers per block (2 for GSI, 1 for WUS)
   */
  void allocateBatch(int         batchSize,
                     const int*  batchPairMatchStarts,
                     int         totalBatchMatchIndices,
                     int         numQueries,
                     int         maxTargetAtoms,
                     int         numBuffersPerBlock);

  /**
   * @brief Get a view suitable for passing to CUDA kernels.
   */
  [[nodiscard]] SubstructMatchResultsDeviceView view() const;

  void setStream(cudaStream_t stream);

  /**
   * @brief Zero the recursive match bits buffer for a new batch.
   */
  void zeroRecursiveBits();

  /**
   * @brief Copy batch results to raw pinned memory pointers.
   *
   * @param hostMatchCounts Output: match counts for this batch [batchSize]
   * @param hostReportedCounts Output: reported counts for this batch [batchSize]
   * @param hostMatchIndices Output: match indices for this batch
   */
  void copyBatchToHost(int*     hostMatchCounts,
                       int*     hostReportedCounts,
                       int16_t* hostMatchIndices) const;

  void setQueryAtomCounts(const int* queryAtomCounts, size_t count);

  [[nodiscard]] int batchSize() const { return batchSize_; }
  [[nodiscard]] int maxTargetAtoms() const { return maxTargetAtoms_; }
  [[nodiscard]] uint32_t* recursiveMatchBits() { return recursiveMatchBits_.data(); }

 private:
  cudaStream_t stream_ = nullptr;

  int batchSize_    = 0;
  int numQueries_   = 0;
  int maxTargetAtoms_ = 0;

  AsyncDeviceVector<int>     matchCounts_;
  AsyncDeviceVector<int>     reportedCounts_;
  AsyncDeviceVector<int>     pairMatchStarts_;
  AsyncDeviceVector<int16_t> matchIndices_;
  AsyncDeviceVector<int>     queryAtomCounts_;

  AsyncDeviceVector<PartialMatch> overflowBuffer_;
  int overflowBuffersPerBlock_ = 0;

  AsyncDeviceVector<uint32_t> recursiveMatchBits_;

  AsyncDeviceVector<uint32_t> labelMatrixBuffer_;

  int totalBatchMatchIndices_ = 0;
};

// Forward declaration for function signatures below
struct LeafSubpatterns;

/**
 * @brief Perform batch substructure matching on GPU.
 *
 * Returns results in a dynamically allocated nested vector format.
 * Memory is proportional to actual matches, avoiding worst-case pre-allocation.
 *
 * @param targetsDevice Device-resident target molecules (use addToBatch to build)
 * @param queriesDevice Device-resident query molecules (use addQueryToBatch to build)
 * @param targetsHost Host-side target data (for atom counts)
 * @param queriesHost Host-side query data (for atom counts)
 * @param leafSubpatterns Pre-built leaf subpatterns for recursive SMARTS (or empty)
 * @param results Output: matches[target][query][match] = vector of target atom indices
 * @param algorithm Algorithm to use for matching
 * @param stream CUDA stream for async operations
 * @param batchSize Number of pairs per batch (default 1024).
 * @param numRunners Number of GPU runner threads (default 2).
 * @param numPreprocessors Number of CPU preprocessor threads (0 = runners preprocess inline).
 */
void getSubstructMatches(MoleculesDevice&           targetsDevice,
                         const MoleculesDevice&     queriesDevice,
                         const MoleculesHost&       targetsHost,
                         const MoleculesHost&       queriesHost,
                         const LeafSubpatterns&     leafSubpatterns,
                         SubstructSearchResults&    results,
                         SubstructAlgorithm         algorithm,
                         cudaStream_t               stream,
                         int                        batchSize = 1024,
                         int                        numRunners = 2,
                         int                        numPreprocessors = 0);

/**
 * @brief Convenience overload that builds LeafSubpatterns internally.
 *
 * Builds leaf subpatterns from queriesHost before matching. For repeated calls
 * with the same queries, prefer the overload accepting pre-built LeafSubpatterns.
 */
void getSubstructMatches(MoleculesDevice&           targetsDevice,
                         const MoleculesDevice&     queriesDevice,
                         const MoleculesHost&       targetsHost,
                         const MoleculesHost&       queriesHost,
                         SubstructSearchResults&    results,
                         SubstructAlgorithm         algorithm,
                         cudaStream_t               stream,
                         int                        batchSize = 1024,
                         int                        numRunners = 2,
                         int                        numPreprocessors = 0);

// BatchedPatternEntry is defined in pinned_buffer_pool.h

/**
 * @brief Scratch buffers for recursive SMARTS preprocessing.
 *
 * Reusable device memory to avoid repeated alloc/free between kernels.
 * For nested patterns, intermediateBits holds results from child levels
 * that become input for parent patterns.
 *
 * The patternsAtDepthHost pointer can reference memory from the consolidated
 * pinned buffer, or fallback to an owned allocation if not set.
 */
struct RecursiveScratchBuffers {
  AsyncDeviceVector<BatchedPatternEntry> patternEntries;
  AsyncDeviceVector<PartialMatch>        overflow;
  AsyncDeviceVector<uint32_t>            labelMatrixBuffer;
  AsyncDeviceVector<uint32_t>            intermediateBits;  ///< Child pattern results for nested recursion
  BatchedPatternEntry*                   patternsAtDepthHost = nullptr;  ///< Points into consolidated buffer or ownedBuffer
  int                                    patternsAtDepthHostCapacity = 0;
  ScopedCudaEvent                        patternsAtDepthHostCopyDone;  ///< Guards reuse of patternsAtDepthHost
  bool                                   patternsAtDepthHostCopyPending = false;

  explicit RecursiveScratchBuffers(cudaStream_t stream) 
      : patternEntries(), overflow(), labelMatrixBuffer(), intermediateBits(),
        patternsAtDepthHostCopyDone(), patternsAtDepthHostCopyPending(false) {
    patternEntries.setStream(stream);
    overflow.setStream(stream);
    labelMatrixBuffer.setStream(stream);
    intermediateBits.setStream(stream);
  }

  ~RecursiveScratchBuffers() {
    if (ownsBuffer_ && patternsAtDepthHost != nullptr) {
      cudaFreeHost(patternsAtDepthHost);
    }
  }

  RecursiveScratchBuffers(const RecursiveScratchBuffers&)            = delete;
  RecursiveScratchBuffers& operator=(const RecursiveScratchBuffers&) = delete;
  RecursiveScratchBuffers(RecursiveScratchBuffers&&)                 = delete;
  RecursiveScratchBuffers& operator=(RecursiveScratchBuffers&&)      = delete;

  void setStream(cudaStream_t stream) {
    patternEntries.setStream(stream);
    overflow.setStream(stream);
    labelMatrixBuffer.setStream(stream);
    intermediateBits.setStream(stream);
  }

  void setPinnedBuffer(BatchedPatternEntry* ptr, int capacity) {
    if (ownsBuffer_ && patternsAtDepthHost != nullptr) {
      cudaFreeHost(patternsAtDepthHost);
    }
    patternsAtDepthHost         = ptr;
    patternsAtDepthHostCapacity = capacity;
    ownsBuffer_                 = false;
  }

  /**
   * @brief Ensure pinned buffer capacity, allocating if needed.
   *
   * If the consolidated buffer is too small, allocates a separate owned buffer.
   */
  void ensureCapacity(int requiredCapacity) {
    if (patternsAtDepthHostCapacity >= requiredCapacity) {
      return;
    }
    // Need to allocate (or reallocate) an owned buffer
    if (ownsBuffer_ && patternsAtDepthHost != nullptr) {
      cudaFreeHost(patternsAtDepthHost);
    }
    const int newCapacity = static_cast<int>(requiredCapacity * 1.5);
    cudaCheckError(cudaMallocHost(&patternsAtDepthHost, newCapacity * sizeof(BatchedPatternEntry)));
    patternsAtDepthHostCapacity = newCapacity;
    ownsBuffer_                 = true;
  }

 private:
  bool ownsBuffer_ = false;
};

/**
 * @brief Key for mapping (queryIdx, patternId) to leaf subpattern molecule index.
 */
struct LeafSubpatternKey {
  int queryIdx;
  int patternId;

  bool operator==(const LeafSubpatternKey& other) const {
    return queryIdx == other.queryIdx && patternId == other.patternId;
  }
};

/**
 * @brief Hash function for LeafSubpatternKey.
 */
struct LeafSubpatternKeyHash {
  std::size_t operator()(const LeafSubpatternKey& key) const {
    return std::hash<int>()(key.queryIdx) ^ (std::hash<int>()(key.patternId) << 16);
  }
};

/**
 * @brief Pre-built collection of all recursive SMARTS leaf subpatterns.
 *
 * Contains all recursive patterns from all queries, built once before batch
 * processing begins. Kernels access patterns by molecule index via the
 * patternIndexMap lookup.
 *
 * The device-side data is shared (read-only) across all worker threads.
 */
struct LeafSubpatterns {
  std::unordered_map<LeafSubpatternKey, int, LeafSubpatternKeyHash> patternIndexMap;
  MoleculesHost   patternsHost;
  MoleculesDevice patternsDevice;

  /// Precomputed pattern entries per query, organized by depth.
  /// perQueryPatterns[queryIdx][depth] = vector of BatchedPatternEntry
  std::vector<std::array<std::vector<BatchedPatternEntry>, kMaxRecursionDepth + 1>> perQueryPatterns;
  
  /// Max recursion depth per query (0 if no recursive patterns)
  std::vector<int> perQueryMaxDepth;

  LeafSubpatterns() = default;

  /**
   * @brief Build all leaf subpatterns from all queries.
   *
   * Iterates through all queries and extracts all recursive patterns,
   * building them into a single MoleculesHost batch. Must be called
   * before batch processing begins.
   *
   * @param queriesHost Host-side query data containing recursivePatterns
   */
  void buildAllPatterns(const MoleculesHost& queriesHost);

  /**
   * @brief Upload patterns to device.
   *
   * @param stream CUDA stream for async operations
   */
  void syncToDevice(cudaStream_t stream);

  /**
   * @brief Look up a pattern's molecule index.
   *
   * @param queryIdx Index of the query containing the pattern
   * @param patternId Pattern ID within the query
   * @return The molecule index in patternsHost/patternsDevice, or -1 if not found
   */
  [[nodiscard]] int getPatternIndex(int queryIdx, int patternId) const {
    LeafSubpatternKey key{queryIdx, patternId};
    auto it = patternIndexMap.find(key);
    return (it != patternIndexMap.end()) ? it->second : -1;
  }

  /**
   * @brief Check if any patterns were built.
   */
  [[nodiscard]] bool empty() const { return patternIndexMap.empty(); }

  /**
   * @brief Get the number of patterns.
   */
  [[nodiscard]] size_t size() const { return patternIndexMap.size(); }

  /**
   * @brief Get view for kernel access.
   */
  [[nodiscard]] MoleculesDeviceView view() const { return patternsDevice.view(); }
};

// kMaxRecursionDepth is defined in pinned_buffer_pool.h

/**
 * @brief Two-stream pipeline context for overlapping recursive preprocessing with matching.
 *
 * Uses a high-priority stream for recursive paint operations and a low-priority
 * stream for main query matching. Events synchronize pairs that depend on
 * recursive preprocessing results.
 *
 * Host-side pinned buffers are now referenced via pointers into the consolidated
 * buffer rather than owned allocations.
 */
struct TwoStreamPipelineContext {
  ScopedStreamWithPriority recursiveStream;  ///< High priority stream for paint kernels

  /// Low priority streams for match kernels at depth > 0.
  /// Depth 0 uses the main ctx.stream. Depths 1..kMaxRecursionDepth each get their own stream
  /// so matching at different depths can overlap.
  std::array<ScopedStreamWithPriority, kMaxRecursionDepth> matchStreams;

  std::array<ScopedCudaEvent, kMaxRecursionDepth> depthEvents;

  ScopedCudaEvent recursiveDoneEvent;  ///< Signaled when recursive stream work completes
  std::array<ScopedCudaEvent, kMaxRecursionDepth> matchDoneEvents;  ///< Signaled when match stream work completes

  /// Matching: global pair indices for each depth group (depth 0..kMaxRecursionDepth)
  std::array<AsyncDeviceVector<int>, kMaxRecursionDepth + 1> matchGlobalPairIndices;

  /// Matching: batch-local indices for each depth group (depth 0..kMaxRecursionDepth)
  std::array<AsyncDeviceVector<int>, kMaxRecursionDepth + 1> matchBatchLocalIndices;

  /// Host-side schedule: pairs to match after each depth level completes
  std::array<std::vector<int>, kMaxRecursionDepth + 1> matchPairsHost;

  /// Pointers to pinned buffers for H2D transfers (reference consolidated buffer)
  std::array<int*, kMaxRecursionDepth + 1> matchGlobalPairIndicesHost = {};
  std::array<int*, kMaxRecursionDepth + 1> matchBatchLocalIndicesHost = {};
  int perDepthCapacity = 0;

  int maxDepthInBatch = 0;

  /**
   * @brief Construct pipeline context with priority streams.
   *
   * The recursive stream gets high priority (lower numerical value),
   * match streams get low priority (higher numerical value).
   * 
   * @param workerIdx Worker thread index for unique stream naming
   */
  explicit TwoStreamPipelineContext(int workerIdx = 0);

  /**
   * @brief Set pointers to consolidated pinned buffer regions.
   */
  void setPinnedBuffers(const std::array<int*, kMaxRecursionDepth + 1>& globalPairPtrs,
                        const std::array<int*, kMaxRecursionDepth + 1>& batchLocalPtrs,
                        int capacity) {
    matchGlobalPairIndicesHost = globalPairPtrs;
    matchBatchLocalIndicesHost = batchLocalPtrs;
    perDepthCapacity           = capacity;
  }
};

/**
 * @brief Preprocess ALL recursive SMARTS patterns for a batch.
 *
 * Uses pre-built leaf subpatterns to run paint kernels for all recursive patterns
 * that affect pairs in the current batch. Optionally records events after each
 * depth level for two-stream pipeline synchronization.
 *
 * @param targetsDevice Device-resident target molecules
 * @param targetsHost Host-side target data
 * @param queriesHost Host-side query data (contains recursivePatterns per query)
 * @param leafSubpatterns Pre-built leaf subpattern molecules (device-resident)
 * @param batchResults The batch results buffer where recursiveMatchBits will be written
 * @param numQueries Total number of queries (for computing pair indices)
 * @param batchPairOffset Global pair index where current batch starts
 * @param batchSize Number of pairs in this batch
 * @param algorithm Algorithm to use for matching
 * @param stream CUDA stream for async operations
 * @param scratch Reusable scratch buffers (avoids alloc/free between kernels)
 * @param scratchPatternEntries Vector to store pattern entries for the batch
 * @param depthEvents Array of events to record after each depth level, or nullptr
 * @param numDepthEvents Number of events in the array (typically kMaxRecursionDepth)
 */
void preprocessRecursiveSmartsBatchedWithEvents(const MoleculesDevice&            targetsDevice,
                                                const MoleculesHost&              queriesHost,
                                                const LeafSubpatterns&            leafSubpatterns,
                                                BatchResultsDevice&               batchResults,
                                                int                               numQueries,
                                                int                               batchPairOffset,
                                                int                               batchSize,
                                                SubstructAlgorithm                algorithm,
                                                cudaStream_t                      stream,
                                                RecursiveScratchBuffers&          scratch,
                                                std::vector<BatchedPatternEntry>& scratchPatternEntries,
                                                cudaEvent_t*                      depthEvents,
                                                int                               numDepthEvents);

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCTURE_SEARCH_CUH

