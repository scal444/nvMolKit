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

#ifndef NVMOLKIT_PINNED_BUFFER_POOL_H
#define NVMOLKIT_PINNED_BUFFER_POOL_H

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "cuda_error_check.h"

namespace nvMolKit {

constexpr int kMaxRecursionDepth = 4;

/**
 * @brief Per-pattern metadata for batched recursive preprocessing kernel.
 *
 * Each entry describes one recursive pattern in the combined batch:
 * which main query it belongs to, what bit to paint, and where the
 * pattern data starts in the combined pattern batch.
 */
struct BatchedPatternEntry {
  int mainQueryIdx;     ///< Index of the main query this pattern belongs to
  int patternId;        ///< Bit position (0-31) to paint for this pattern
  int patternMolIdx;    ///< Index into the combined patterns MoleculesDevice
  int depth;            ///< Nesting depth (0=leaf, higher=parent of children)
  int localIdInParent;  ///< Bit position in parent's input (for nested patterns)
};

/**
 * @brief Consolidated pinned memory buffer for a single GpuExecutor.
 *
 * Combines all pinned host memory allocations into a single cudaMallocHost call,
 * then partitions the memory into separate logical regions accessed via raw pointers.
 * This reduces allocation overhead from ~16 separate cudaMallocHost calls to 1.
 */
struct ConsolidatedPinnedBuffer {
  char*  basePtr   = nullptr;
  size_t totalSize = 0;

  // GpuExecutor buffers
  int*     pairIndices          = nullptr;
  int*     miniBatchPairMatchStarts = nullptr;
  int*     matchCounts          = nullptr;
  int*     reportedCounts       = nullptr;
  int16_t* matchIndices         = nullptr;

  // RecursivePipelineContext buffers (kMaxRecursionDepth + 1 = 5 arrays each)
  std::array<int*, kMaxRecursionDepth + 1> matchGlobalPairIndicesHost = {};
  std::array<int*, kMaxRecursionDepth + 1> matchBatchLocalIndicesHost = {};

  // RecursiveScratchBuffers double-buffered pattern entries
  std::array<BatchedPatternEntry*, 2> patternsAtDepthHost = {nullptr, nullptr};

  // Capacities for bounds checking
  int pairIndicesCapacity   = 0;
  int matchIndicesCapacity  = 0;
  int perDepthCapacity      = 0;
  int patternsCapacity      = 0;

  ConsolidatedPinnedBuffer() = default;

  ConsolidatedPinnedBuffer(const ConsolidatedPinnedBuffer&)            = delete;
  ConsolidatedPinnedBuffer& operator=(const ConsolidatedPinnedBuffer&) = delete;

  ConsolidatedPinnedBuffer(ConsolidatedPinnedBuffer&& other) noexcept;
  ConsolidatedPinnedBuffer& operator=(ConsolidatedPinnedBuffer&& other) noexcept;
  ~ConsolidatedPinnedBuffer();

  /**
   * @brief Compute the size needed for one consolidated buffer.
   */
  static size_t computeSize(int maxBatchSize, int maxMatchIndicesEstimate, int maxPatternsPerDepth);

  /**
   * @brief Allocate consolidated pinned memory and partition into regions.
   *
   * @param maxBatchSize Maximum number of pairs per batch
   * @param maxMatchIndicesEstimate Maximum total match indices (batchSize * maxTargetAtoms * maxQueryAtoms)
   * @param maxPatternsPerDepth Maximum patterns per depth level for recursive SMARTS
   */
  void allocate(int maxBatchSize, int maxMatchIndicesEstimate, int maxPatternsPerDepth);

  /**
   * @brief Assign from externally-allocated memory (no ownership).
   *
   * The caller is responsible for freeing the memory. This buffer will not
   * free it on destruction.
   */
  void assignExternal(char* externalPtr, int maxBatchSize, int maxMatchIndicesEstimate, int maxPatternsPerDepth);

  [[nodiscard]] bool isAllocated() const { return basePtr != nullptr; }
  [[nodiscard]] bool ownsMemory() const { return ownsMemory_; }

 private:
  bool ownsMemory_ = true;
};

/**
 * @brief Async resource cleanup manager using a background thread.
 *
 * Enables "fire and forget" cleanup pattern where the calling function can
 * return immediately while resource destruction happens in the background.
 * Uses a singleton pattern with a dedicated cleanup thread.
 */
class AsyncResourceCleaner {
 public:
  static AsyncResourceCleaner& instance();

  /**
   * @brief Schedule a generic cleanup task.
   *
   * The task will be executed in the background cleanup thread.
   */
  void scheduleCleanup(std::function<void()> task);

  /**
   * @brief Schedule cleanup of a consolidated pinned buffer.
   *
   * Takes ownership of the buffer and destroys it in the background.
   */
  void scheduleBufferCleanup(std::unique_ptr<ConsolidatedPinnedBuffer> buffer);

  /**
   * @brief Wait for all pending cleanup tasks to complete.
   *
   * Useful for ensuring cleanup is done before program exit or tests.
   */
  void flush();

  AsyncResourceCleaner(const AsyncResourceCleaner&)            = delete;
  AsyncResourceCleaner& operator=(const AsyncResourceCleaner&) = delete;

 private:
  AsyncResourceCleaner();
  ~AsyncResourceCleaner();

  void cleanupThreadFunc();

  std::thread                             cleanupThread_;
  std::queue<std::function<void()>>       pendingTasks_;
  std::queue<std::unique_ptr<ConsolidatedPinnedBuffer>> pendingBuffers_;
  std::mutex                              mutex_;
  std::condition_variable                 cv_;
  std::condition_variable                 flushCv_;
  std::atomic<bool>                       shutdown_{false};
  std::atomic<int>                        pendingCount_{0};
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_PINNED_BUFFER_POOL_H

