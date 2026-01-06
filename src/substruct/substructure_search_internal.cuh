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

#ifndef NVMOLKIT_SUBSTRUCTURE_SEARCH_INTERNAL_CUH
#define NVMOLKIT_SUBSTRUCTURE_SEARCH_INTERNAL_CUH

/**
 * @file substructure_search_internal.cuh
 * @brief Internal implementation details for substructure search.
 *
 * This header exposes internal types and functions needed for testing.
 * Not part of the public API - may change without notice.
 */

#include <cuda_runtime.h>

#include <array>
#include <unordered_map>
#include <vector>

#include "cuda_error_check.h"
#include "device.h"
#include "device_vector.h"
#include "molecules.h"
#include "molecules_device.cuh"
#include "pinned_buffer_pool.h"
#include "substructure_search.cuh"

namespace nvMolKit {

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

/**
 * @brief Scratch buffers for recursive SMARTS preprocessing.
 *
 * Reusable device memory to avoid repeated alloc/free between kernels.
 * For nested patterns, intermediateBits holds results from child levels
 * that become input for parent patterns.
 *
 * Uses double-buffered pinned memory for pattern entries to avoid CPU stalls
 * waiting for H2D copies to complete. While one buffer is being copied, the
 * other can be filled with the next sub-batch's data.
 */
struct RecursiveScratchBuffers {
  AsyncDeviceVector<BatchedPatternEntry> patternEntries;
  AsyncDeviceVector<PartialMatch>        overflow;
  AsyncDeviceVector<uint32_t>            labelMatrixBuffer;
  AsyncDeviceVector<uint32_t>            intermediateBits;  ///< Child pattern results for nested recursion

  /// Double-buffered pinned pattern entries for overlap
  std::array<BatchedPatternEntry*, 2>    patternsAtDepthHost = {nullptr, nullptr};
  std::array<int, 2>                     patternsAtDepthHostCapacity = {0, 0};
  std::array<ScopedCudaEvent, 2>         patternsAtDepthHostCopyDone;
  std::array<bool, 2>                    patternsAtDepthHostCopyPending = {false, false};
  int                                    currentPatternBuffer = 0;  ///< Index of buffer to fill next

  explicit RecursiveScratchBuffers(cudaStream_t stream) 
      : patternEntries(), overflow(), labelMatrixBuffer(), intermediateBits() {
    patternEntries.setStream(stream);
    overflow.setStream(stream);
    labelMatrixBuffer.setStream(stream);
    intermediateBits.setStream(stream);
  }

  ~RecursiveScratchBuffers() {
    for (int i = 0; i < 2; ++i) {
      if (ownsBuffer_[i] && patternsAtDepthHost[i] != nullptr) {
        cudaFreeHost(patternsAtDepthHost[i]);
      }
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

  void setPinnedBuffer(const std::array<BatchedPatternEntry*, 2>& ptrs, int capacity) {
    for (int i = 0; i < 2; ++i) {
      if (ownsBuffer_[i] && patternsAtDepthHost[i] != nullptr) {
        cudaFreeHost(patternsAtDepthHost[i]);
      }
      patternsAtDepthHost[i]         = ptrs[i];
      patternsAtDepthHostCapacity[i] = capacity;
      ownsBuffer_[i]                 = false;
    }
  }

  /**
   * @brief Allocate owned pinned buffers with given capacity.
   * For tests and standalone usage.
   */
  void allocateBuffers(int capacity) {
    for (int i = 0; i < 2; ++i) {
      if (ownsBuffer_[i] && patternsAtDepthHost[i] != nullptr) {
        cudaFreeHost(patternsAtDepthHost[i]);
      }
      cudaCheckError(cudaMallocHost(&patternsAtDepthHost[i], capacity * sizeof(BatchedPatternEntry)));
      patternsAtDepthHostCapacity[i] = capacity;
      ownsBuffer_[i]                 = true;
    }
  }

  /**
   * @brief Get the current buffer index and advance to next for double-buffering.
   */
  int acquireBufferIndex() {
    int idx = currentPatternBuffer;
    currentPatternBuffer = 1 - currentPatternBuffer;
    return idx;
  }

  /**
   * @brief Wait for a specific buffer's copy to complete if pending.
   */
  void waitForBuffer(int bufferIdx) {
    if (patternsAtDepthHostCopyPending[bufferIdx]) {
      cudaCheckError(cudaEventSynchronize(patternsAtDepthHostCopyDone[bufferIdx].event()));
      patternsAtDepthHostCopyPending[bufferIdx] = false;
    }
  }

  /**
   * @brief Record that a copy has been initiated on a buffer.
   */
  void recordCopy(int bufferIdx, cudaStream_t stream) {
    cudaCheckError(cudaEventRecord(patternsAtDepthHostCopyDone[bufferIdx].event(), stream));
    patternsAtDepthHostCopyPending[bufferIdx] = true;
  }

  /**
   * @brief Check that pinned buffer has sufficient capacity.
   * @throws std::runtime_error if capacity is exceeded or buffer not initialized
   */
  void ensureCapacity(int bufferIdx, int requiredCapacity) {
    if (patternsAtDepthHostCapacity[bufferIdx] >= requiredCapacity) {
      return;
    }
    throw std::runtime_error(
        "Recursive SMARTS pattern count (" + std::to_string(requiredCapacity) +
        ") exceeds pre-allocated capacity (" + std::to_string(patternsAtDepthHostCapacity[bufferIdx]) +
        "). Ensure buffers are properly initialized.");
  }

 private:
  std::array<bool, 2> ownsBuffer_ = {false, false};
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
   * @param numBuffersPerBlock Overflow buffers per block (2 for GSI)
   * @param maxMatchesToFind Stop searching after this many matches (-1 = no limit)
   * @param countOnly If true, count matches but don't store them
   */
  void allocateBatch(int         batchSize,
                     const int*  batchPairMatchStarts,
                     int         totalBatchMatchIndices,
                     int         numQueries,
                     int         maxTargetAtoms,
                     int         numBuffersPerBlock,
                     int         maxMatchesToFind = -1,
                     bool        countOnly = false);

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

  // Early exit control
  int  maxMatchesToFind_ = -1;
  bool countOnly_        = false;
};

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

  /// Low priority stream for match kernels at depth > 0.
  /// Depth 0 uses the main ctx.stream.
  ScopedStreamWithPriority postRecursionStream;

  std::array<ScopedCudaEvent, kMaxRecursionDepth> depthEvents;

  ScopedCudaEvent recursiveDoneEvent;      ///< Signaled when recursive stream work completes
  ScopedCudaEvent postRecursionDoneEvent;  ///< Signaled when post-recursion stream work completes

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
   * post-recursion stream gets low priority (higher numerical value).
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

#endif  // NVMOLKIT_SUBSTRUCTURE_SEARCH_INTERNAL_CUH

