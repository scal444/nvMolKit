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
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

#include "cuda_error_check.h"
#include "host_vector.h"
#include "nvtx.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit
#include "device.h"
#include "device_vector.h"
#include "molecules.h"
#include "molecules_device.cuh"
#include "pinned_buffer_pool.h"
#include "substruct_algos.cuh"
#include "substructure_search.h"

namespace nvMolKit {

// Forward declarations for friend function
struct DeviceTimingsData;

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
 * @brief Mini-batch-local device-side storage for substructure match results.
 *
 * Owns device memory for a single mini-batch and provides views for kernel access.
 * Results are copied back to host after each mini-batch and accumulated.
 */
class MiniBatchResultsDevice {
  friend void launchSubstructMatchKernel(SubstructAlgorithm, MoleculesDeviceView, MoleculesDeviceView,
                                         const MiniBatchResultsDevice&, const int*, int, int,
                                         const int*, DeviceTimingsData*, cudaStream_t);
 public:
  MiniBatchResultsDevice() = default;
  explicit MiniBatchResultsDevice(cudaStream_t stream) : stream_(stream) { setStream(stream); }

  /**
   * @brief Allocate mini-batch-local buffers for a specific mini-batch.
   *
   * @param miniBatchSize Number of pairs in this mini-batch
   * @param miniBatchPairMatchStarts Mini-batch-local offsets into matchIndices [miniBatchSize + 1]
   * @param totalMiniBatchMatchIndices Total match indices capacity for this mini-batch
   * @param numQueries Total number of queries (for kernel view)
   * @param maxTargetAtoms Max atoms per target (stride for recursiveMatchBits)
   * @param numBuffersPerBlock Overflow buffers per block (2 for GSI)
   * @param maxMatchesToFind Stop searching after this many matches (-1 = no limit)
   * @param countOnly If true, count matches but don't store them
   */
  void allocateMiniBatch(int         miniBatchSize,
                         const int*  miniBatchPairMatchStarts,
                         int         totalMiniBatchMatchIndices,
                     int         numQueries,
                     int         maxTargetAtoms,
                     int         numBuffersPerBlock,
                     int         maxMatchesToFind = -1,
                     bool        countOnly = false);

  void setStream(cudaStream_t stream);

  /**
   * @brief Zero the recursive match bits buffer for a new mini-batch.
   */
  void zeroRecursiveBits();

  /**
   * @brief Copy mini-batch results to raw pinned memory pointers.
   *
   * @param hostMatchCounts Output: match counts for this mini-batch [miniBatchSize]
   * @param hostReportedCounts Output: reported counts for this mini-batch [miniBatchSize]
   * @param hostMatchIndices Output: match indices for this mini-batch
   */
  void copyMiniBatchToHost(int*     hostMatchCounts,
                           int*     hostReportedCounts,
                           int16_t* hostMatchIndices) const;

  void setQueryAtomCounts(const int* queryAtomCounts, size_t count);

  [[nodiscard]] int miniBatchSize() const { return miniBatchSize_; }
  [[nodiscard]] int numQueries() const { return numQueries_; }
  [[nodiscard]] int maxTargetAtoms() const { return maxTargetAtoms_; }
  [[nodiscard]] uint32_t* recursiveMatchBits() { return recursiveMatchBits_.data(); }
  [[nodiscard]] uint32_t* labelMatrixBuffer() { return labelMatrixBuffer_.data(); }

 private:
  cudaStream_t stream_ = nullptr;

  int miniBatchSize_    = 0;
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

  int totalMiniBatchMatchIndices_ = 0;

  // Early exit control
  int  maxMatchesToFind_ = -1;
  bool countOnly_        = false;
};

/**
 * @brief Pipeline context for overlapping recursive preprocessing with matching.
 *
 * Uses a high-priority stream for recursive paint operations and a low-priority
 * stream for main query matching. Events synchronize pairs that depend on
 * recursive preprocessing results.
 *
 * Host-side pinned buffers are now referenced via pointers into the consolidated
 * buffer rather than owned allocations.
 */
struct RecursivePipelineContext {
  ScopedStreamWithPriority recursiveStream;  ///< High priority stream for paint kernels

  /// Low priority stream for match kernels at depth > 0.
  /// Depth 0 uses the main ctx.stream.
  ScopedStreamWithPriority postRecursionStream;

  std::array<ScopedCudaEvent, kMaxRecursionDepth> depthEvents;

  ScopedCudaEvent recursiveDoneEvent;      ///< Signaled when recursive stream work completes
  ScopedCudaEvent postRecursionDoneEvent;  ///< Signaled when post-recursion stream work completes

  /// Matching: global pair indices for each depth group (depth 0..kMaxRecursionDepth)
  std::array<AsyncDeviceVector<int>, kMaxRecursionDepth + 1> matchGlobalPairIndices;

  /// Matching: mini-batch-local indices for each depth group (depth 0..kMaxRecursionDepth)
  std::array<AsyncDeviceVector<int>, kMaxRecursionDepth + 1> matchMiniBatchLocalIndices;

  /// Host-side schedule: pairs to match after each depth level completes
  std::array<std::vector<int>, kMaxRecursionDepth + 1> matchPairsHost;

  /// Pointers to pinned buffers for H2D transfers (reference consolidated buffer)
  std::array<int*, kMaxRecursionDepth + 1> matchGlobalPairIndicesHost = {};
  std::array<int*, kMaxRecursionDepth + 1> matchMiniBatchLocalIndicesHost = {};
  int perDepthCapacity = 0;

  int maxDepthInMiniBatch = 0;

  /**
   * @brief Construct pipeline context with priority streams.
   *
   * The recursive stream gets high priority (lower numerical value),
   * post-recursion stream gets low priority (higher numerical value).
   * 
   * @param executorIdx Executor index for unique stream naming
   */
  explicit RecursivePipelineContext(int executorIdx = 0);

  /**
   * @brief Set pointers to consolidated pinned buffer regions.
   */
  void setPinnedBuffers(const std::array<int*, kMaxRecursionDepth + 1>& globalPairPtrs,
                        const std::array<int*, kMaxRecursionDepth + 1>& miniBatchLocalPtrs,
                        int capacity) {
    matchGlobalPairIndicesHost = globalPairPtrs;
    matchMiniBatchLocalIndicesHost = miniBatchLocalPtrs;
    perDepthCapacity           = capacity;
  }
};

/**
 * @brief Preprocess ALL recursive SMARTS patterns for a mini-batch.
 *
 * Uses pre-built leaf subpatterns to run paint kernels for all recursive patterns
 * that affect pairs in the current mini-batch. Optionally records events after each
 * depth level for pipeline synchronization.
 *
 * @param targetsDevice Device-resident target molecules
 * @param queriesHost Host-side query data (contains recursivePatterns per query)
 * @param leafSubpatterns Pre-built leaf subpattern molecules (device-resident)
 * @param miniBatchResults The mini-batch results buffer where recursiveMatchBits will be written
 * @param numQueries Total number of queries (for computing pair indices)
 * @param miniBatchPairOffset Global pair index where current mini-batch starts
 * @param miniBatchSize Number of pairs in this mini-batch
 * @param algorithm Algorithm to use for matching
 * @param stream CUDA stream for async operations
 * @param scratch Reusable scratch buffers (avoids alloc/free between kernels)
 * @param scratchPatternEntries Vector to store pattern entries for the mini-batch
 * @param depthEvents Array of events to record after each depth level, or nullptr
 * @param numDepthEvents Number of events in the array (typically kMaxRecursionDepth)
 */
void preprocessRecursiveSmartsBatchedWithEvents(const MoleculesDevice&            targetsDevice,
                                                const MoleculesHost&              queriesHost,
                                                const LeafSubpatterns&            leafSubpatterns,
                                                MiniBatchResultsDevice&           miniBatchResults,
                                                int                               numQueries,
                                                int                               miniBatchPairOffset,
                                                int                               miniBatchSize,
                                                SubstructAlgorithm                algorithm,
                                                cudaStream_t                      stream,
                                                RecursiveScratchBuffers&          scratch,
                                                std::vector<BatchedPatternEntry>& scratchPatternEntries,
                                                cudaEvent_t*                      depthEvents,
                                                int                               numDepthEvents);

// =============================================================================
// RDKit Fallback Processing
// =============================================================================

/**
 * @brief Process a single (target, query) pair using RDKit's CPU implementation.
 *
 * Used as fallback for oversized targets or overflow cases.
 */
void processWithRDKitFallback(const RDKit::ROMol*     target,
                              const RDKit::ROMol*     query,
                              int                     targetIdx,
                              int                     queryIdx,
                              SubstructSearchResults& results,
                              std::mutex&             resultsMutex,
                              int                     maxMatches);

/**
 * @brief Thread-safe queue for RDKit fallback processing.
 *
 * Worker threads wait on a condition variable and consume entries as they arrive.
 * Supports concurrent producers (GPU batch accumulators) and consumers (RDKit workers).
 */
class RDKitFallbackQueue {
 public:
  RDKitFallbackQueue(const std::vector<const RDKit::ROMol*>* targets,
                     const std::vector<const RDKit::ROMol*>* queries,
                     SubstructSearchResults*                 results,
                     std::mutex*                             resultsMutex,
                     int                                     maxMatches);

  void enqueue(const std::vector<RDKitFallbackEntry>& entries);
  void enqueue(const RDKitFallbackEntry& entry);

  void registerProducer();
  void unregisterProducer();
  void shutdown();
  void workerLoop();

  [[nodiscard]] size_t processedCount() const;
  std::vector<RDKitFallbackEntry> drainToVector();
  std::mutex& getResultsMutex();
  bool tryProcessOne();

  template <typename Predicate>
  int processWhileWaiting(Predicate shouldStop) {
    int processed = 0;
    while (!shouldStop()) {
      if (!tryProcessOne()) {
        break;
      }
      ++processed;
    }
    return processed;
  }

  [[nodiscard]] bool hasWork() const;

 private:
  void processEntry(const RDKitFallbackEntry& entry);

  const std::vector<const RDKit::ROMol*>* targets_;
  const std::vector<const RDKit::ROMol*>* queries_;
  SubstructSearchResults*                 results_;
  std::mutex*                             resultsMutex_;
  int                                     maxMatches_;

  mutable std::mutex      mutex_;
  std::condition_variable cv_;
  std::queue<RDKitFallbackEntry> queue_;
  std::atomic<size_t>     queueSize_{0};
  bool                    shutdown_;
  int                     activeProducers_;
  std::atomic<size_t>     processedCount_{0};
};

/**
 * @brief RAII helper to register/unregister as a producer on the fallback queue.
 */
class FallbackQueueProducerGuard {
 public:
  explicit FallbackQueueProducerGuard(RDKitFallbackQueue* queue) : queue_(queue) {
    if (queue_) queue_->registerProducer();
  }
  ~FallbackQueueProducerGuard() {
    if (queue_) queue_->unregisterProducer();
  }
  FallbackQueueProducerGuard(const FallbackQueueProducerGuard&) = delete;
  FallbackQueueProducerGuard& operator=(const FallbackQueueProducerGuard&) = delete;
 private:
  RDKitFallbackQueue* queue_;
};

// =============================================================================
// Thread Worker Context
// =============================================================================

/**
 * @brief Per-worker context for substructure search threads.
 *
 * Contains cached data about queries and targets that's reused across mini-batches.
 */
struct ThreadWorkerContext {
  PinnedHostVector<int> queryAtomCounts;
  std::vector<int> targetAtomCounts;
  std::vector<int> queryDepths;
  std::vector<int> queryMaxDepths;
  std::vector<int8_t> queryHasPatterns;
  const std::vector<int>* targetSortOrder = nullptr;
  const std::vector<int>* querySortOrder  = nullptr;
  int numTargets     = 0;
  int numQueries     = 0;
  int maxTargetAtoms = 0;
  int maxMatches     = 0;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCTURE_SEARCH_INTERNAL_CUH

