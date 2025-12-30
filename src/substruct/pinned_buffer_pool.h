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
 * @brief Consolidated pinned memory buffer for a single BatchSlot.
 *
 * Combines all pinned host memory allocations into a single cudaMallocHost call,
 * then partitions the memory into separate logical regions accessed via raw pointers.
 * This reduces allocation overhead from ~16 separate cudaMallocHost calls to 1.
 */
struct ConsolidatedPinnedBuffer {
  char*  basePtr   = nullptr;
  size_t totalSize = 0;

  // BatchSlot buffers
  int*     pairIndices          = nullptr;
  int*     batchPairMatchStarts = nullptr;
  int*     matchCounts          = nullptr;
  int*     reportedCounts       = nullptr;
  int16_t* matchIndices         = nullptr;

  // TwoStreamPipelineContext buffers (kMaxRecursionDepth + 1 = 5 arrays each)
  std::array<int*, kMaxRecursionDepth + 1> matchGlobalPairIndicesHost = {};
  std::array<int*, kMaxRecursionDepth + 1> matchBatchLocalIndicesHost = {};

  // RecursiveScratchBuffers buffer
  BatchedPatternEntry* patternsAtDepthHost = nullptr;

  // Capacities for bounds checking
  int pairIndicesCapacity   = 0;
  int matchIndicesCapacity  = 0;
  int perDepthCapacity      = 0;
  int patternsCapacity      = 0;

  ConsolidatedPinnedBuffer() = default;

  ConsolidatedPinnedBuffer(const ConsolidatedPinnedBuffer&)            = delete;
  ConsolidatedPinnedBuffer& operator=(const ConsolidatedPinnedBuffer&) = delete;

  ConsolidatedPinnedBuffer(ConsolidatedPinnedBuffer&& other) noexcept
      : basePtr(other.basePtr),
        totalSize(other.totalSize),
        pairIndices(other.pairIndices),
        batchPairMatchStarts(other.batchPairMatchStarts),
        matchCounts(other.matchCounts),
        reportedCounts(other.reportedCounts),
        matchIndices(other.matchIndices),
        matchGlobalPairIndicesHost(other.matchGlobalPairIndicesHost),
        matchBatchLocalIndicesHost(other.matchBatchLocalIndicesHost),
        patternsAtDepthHost(other.patternsAtDepthHost),
        pairIndicesCapacity(other.pairIndicesCapacity),
        matchIndicesCapacity(other.matchIndicesCapacity),
        perDepthCapacity(other.perDepthCapacity),
        patternsCapacity(other.patternsCapacity) {
    other.basePtr   = nullptr;
    other.totalSize = 0;
  }

  ConsolidatedPinnedBuffer& operator=(ConsolidatedPinnedBuffer&& other) noexcept {
    if (this != &other) {
      if (basePtr != nullptr) {
        cudaFreeHost(basePtr);
      }
      basePtr                    = other.basePtr;
      totalSize                  = other.totalSize;
      pairIndices                = other.pairIndices;
      batchPairMatchStarts       = other.batchPairMatchStarts;
      matchCounts                = other.matchCounts;
      reportedCounts             = other.reportedCounts;
      matchIndices               = other.matchIndices;
      matchGlobalPairIndicesHost = other.matchGlobalPairIndicesHost;
      matchBatchLocalIndicesHost = other.matchBatchLocalIndicesHost;
      patternsAtDepthHost        = other.patternsAtDepthHost;
      pairIndicesCapacity        = other.pairIndicesCapacity;
      matchIndicesCapacity       = other.matchIndicesCapacity;
      perDepthCapacity           = other.perDepthCapacity;
      patternsCapacity           = other.patternsCapacity;

      other.basePtr   = nullptr;
      other.totalSize = 0;
    }
    return *this;
  }

  ~ConsolidatedPinnedBuffer() {
    if (basePtr != nullptr) {
      cudaFreeHost(basePtr);
    }
  }

  /**
   * @brief Allocate consolidated pinned memory and partition into regions.
   *
   * @param maxBatchSize Maximum number of pairs per batch
   * @param maxMatchIndicesEstimate Maximum total match indices (batchSize * maxTargetAtoms * maxQueryAtoms)
   * @param maxPatternsPerDepth Maximum patterns per depth level for recursive SMARTS
   */
  void allocate(int maxBatchSize, int maxMatchIndicesEstimate, int maxPatternsPerDepth);

  [[nodiscard]] bool isAllocated() const { return basePtr != nullptr; }
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

// Forward declaration for BatchSlot (defined in substructure_search.cu)
struct BatchSlot;

/**
 * @brief Thread-safe bounded queue for prepared BatchSlots.
 *
 * Used to pass preprocessed batches from preprocessor threads to runner threads.
 * Blocks on enqueue when full, blocks on dequeue when empty.
 */
class PreparedBatchQueue {
 public:
  explicit PreparedBatchQueue(int capacity) : capacity_(capacity) {}

  /**
   * @brief Enqueue a prepared batch slot. Blocks if queue is full.
   */
  void enqueue(BatchSlot* slot) {
    std::unique_lock<std::mutex> lock(mutex_);
    notFull_.wait(lock, [this] { return queue_.size() < static_cast<size_t>(capacity_) || shutdown_; });
    if (shutdown_) return;
    queue_.push(slot);
    notEmpty_.notify_one();
  }

  /**
   * @brief Dequeue a prepared batch slot. Blocks if queue is empty.
   * @return BatchSlot pointer, or nullptr if shutdown was called
   */
  BatchSlot* dequeue() {
    std::unique_lock<std::mutex> lock(mutex_);
    notEmpty_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
    if (queue_.empty()) return nullptr;
    BatchSlot* slot = queue_.front();
    queue_.pop();
    notFull_.notify_one();
    return slot;
  }

  /**
   * @brief Signal shutdown, waking all waiting threads.
   */
  void shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = true;
    notEmpty_.notify_all();
    notFull_.notify_all();
  }

  PreparedBatchQueue(const PreparedBatchQueue&)            = delete;
  PreparedBatchQueue& operator=(const PreparedBatchQueue&) = delete;

 private:
  std::queue<BatchSlot*>  queue_;
  std::mutex              mutex_;
  std::condition_variable notEmpty_;
  std::condition_variable notFull_;
  int                     capacity_;
  bool                    shutdown_ = false;
};

/**
 * @brief Pool of reusable BatchSlot pointers with blocking acquire/release.
 *
 * Manages a free list of BatchSlot pointers. Does not own the slots - 
 * ownership remains with the caller who populates the pool.
 */
class BatchSlotPool {
 public:
  BatchSlotPool() = default;

  /**
   * @brief Initialize the pool with slot pointers.
   * @param slotPtrs Vector of raw pointers to BatchSlots (caller retains ownership)
   */
  void initialize(const std::vector<BatchSlot*>& slotPtrs) {
    for (auto* slot : slotPtrs) {
      freeList_.push(slot);
    }
    size_ = slotPtrs.size();
  }

  /**
   * @brief Acquire a free slot. Blocks if none available.
   */
  BatchSlot* acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !freeList_.empty() || shutdown_; });
    if (freeList_.empty()) return nullptr;
    BatchSlot* slot = freeList_.front();
    freeList_.pop();
    return slot;
  }

  /**
   * @brief Release a slot back to the pool.
   */
  void release(BatchSlot* slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    freeList_.push(slot);
    cv_.notify_one();
  }

  /**
   * @brief Signal shutdown, waking all waiting threads.
   */
  void shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = true;
    cv_.notify_all();
  }

  [[nodiscard]] size_t size() const { return size_; }

  BatchSlotPool(const BatchSlotPool&)            = delete;
  BatchSlotPool& operator=(const BatchSlotPool&) = delete;

 private:
  std::queue<BatchSlot*>  freeList_;
  std::mutex              mutex_;
  std::condition_variable cv_;
  size_t                  size_     = 0;
  bool                    shutdown_ = false;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_PINNED_BUFFER_POOL_H

