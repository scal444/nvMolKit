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

#include "pinned_buffer_pool.h"

namespace nvMolKit {

namespace {
constexpr size_t kAlignment = 64;

size_t alignUp(size_t offset, size_t alignment) {
  return (offset + alignment - 1) & ~(alignment - 1);
}
}  // namespace

void ConsolidatedPinnedBuffer::allocate(int maxBatchSize, int maxMatchIndicesEstimate, int maxPatternsPerDepth) {
  if (basePtr != nullptr) {
    cudaFreeHost(basePtr);
    basePtr = nullptr;
  }

  size_t offset = 0;

  // pairIndices: int[maxBatchSize]
  size_t pairIndicesOff = offset;
  offset += sizeof(int) * maxBatchSize;
  offset = alignUp(offset, kAlignment);

  // batchPairMatchStarts: int[maxBatchSize + 1]
  size_t batchMatchStartsOff = offset;
  offset += sizeof(int) * (maxBatchSize + 1);
  offset = alignUp(offset, kAlignment);

  // matchCounts: int[maxBatchSize]
  size_t matchCountsOff = offset;
  offset += sizeof(int) * maxBatchSize;
  offset = alignUp(offset, kAlignment);

  // reportedCounts: int[maxBatchSize]
  size_t reportedCountsOff = offset;
  offset += sizeof(int) * maxBatchSize;
  offset = alignUp(offset, kAlignment);

  // matchIndices: int16_t[maxMatchIndicesEstimate]
  size_t matchIndicesOff = offset;
  offset += sizeof(int16_t) * maxMatchIndicesEstimate;
  offset = alignUp(offset, kAlignment);

  // matchGlobalPairIndicesHost and matchBatchLocalIndicesHost: (kMaxRecursionDepth + 1) arrays each
  std::array<size_t, kMaxRecursionDepth + 1> globalPairOff = {};
  std::array<size_t, kMaxRecursionDepth + 1> batchLocalOff = {};
  for (int i = 0; i <= kMaxRecursionDepth; ++i) {
    globalPairOff[i] = offset;
    offset += sizeof(int) * maxBatchSize;
    offset = alignUp(offset, kAlignment);

    batchLocalOff[i] = offset;
    offset += sizeof(int) * maxBatchSize;
    offset = alignUp(offset, kAlignment);
  }

  // patternsAtDepthHost: 2 x BatchedPatternEntry[maxPatternsPerDepth] (double-buffered)
  std::array<size_t, 2> patternsOff = {};
  for (int i = 0; i < 2; ++i) {
    patternsOff[i] = offset;
    offset += sizeof(BatchedPatternEntry) * maxPatternsPerDepth;
    offset = alignUp(offset, kAlignment);
  }

  totalSize = offset;
  cudaCheckError(cudaMallocHost(&basePtr, totalSize));

  // Assign pointers
  pairIndices          = reinterpret_cast<int*>(basePtr + pairIndicesOff);
  batchPairMatchStarts = reinterpret_cast<int*>(basePtr + batchMatchStartsOff);
  matchCounts          = reinterpret_cast<int*>(basePtr + matchCountsOff);
  reportedCounts       = reinterpret_cast<int*>(basePtr + reportedCountsOff);
  matchIndices         = reinterpret_cast<int16_t*>(basePtr + matchIndicesOff);

  for (int i = 0; i <= kMaxRecursionDepth; ++i) {
    matchGlobalPairIndicesHost[i] = reinterpret_cast<int*>(basePtr + globalPairOff[i]);
    matchBatchLocalIndicesHost[i] = reinterpret_cast<int*>(basePtr + batchLocalOff[i]);
  }

  for (int i = 0; i < 2; ++i) {
    patternsAtDepthHost[i] = reinterpret_cast<BatchedPatternEntry*>(basePtr + patternsOff[i]);
  }

  // Store capacities
  pairIndicesCapacity  = maxBatchSize;
  matchIndicesCapacity = maxMatchIndicesEstimate;
  perDepthCapacity     = maxBatchSize;
  patternsCapacity     = maxPatternsPerDepth;
}

// =============================================================================
// AsyncResourceCleaner Implementation
// =============================================================================

AsyncResourceCleaner& AsyncResourceCleaner::instance() {
  static AsyncResourceCleaner inst;
  return inst;
}

AsyncResourceCleaner::AsyncResourceCleaner() {
  cleanupThread_ = std::thread(&AsyncResourceCleaner::cleanupThreadFunc, this);
}

AsyncResourceCleaner::~AsyncResourceCleaner() {
  shutdown_ = true;
  cv_.notify_all();
  if (cleanupThread_.joinable()) {
    cleanupThread_.join();
  }
}

void AsyncResourceCleaner::cleanupThreadFunc() {
  while (true) {
    std::function<void()>                         task;
    std::unique_ptr<ConsolidatedPinnedBuffer> buffer;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return shutdown_ || !pendingTasks_.empty() || !pendingBuffers_.empty(); });

      if (shutdown_ && pendingTasks_.empty() && pendingBuffers_.empty()) {
        return;
      }

      if (!pendingTasks_.empty()) {
        task = std::move(pendingTasks_.front());
        pendingTasks_.pop();
      }

      if (!pendingBuffers_.empty()) {
        buffer = std::move(pendingBuffers_.front());
        pendingBuffers_.pop();
      }
    }

    if (task) {
      task();
      --pendingCount_;
    }

    // buffer destructor runs here, freeing pinned memory
    if (buffer) {
      buffer.reset();
      --pendingCount_;
    }

    flushCv_.notify_all();
  }
}

void AsyncResourceCleaner::scheduleCleanup(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++pendingCount_;
    pendingTasks_.push(std::move(task));
  }
  cv_.notify_one();
}

void AsyncResourceCleaner::scheduleBufferCleanup(std::unique_ptr<ConsolidatedPinnedBuffer> buffer) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++pendingCount_;
    pendingBuffers_.push(std::move(buffer));
  }
  cv_.notify_one();
}

void AsyncResourceCleaner::flush() {
  std::unique_lock<std::mutex> lock(mutex_);
  flushCv_.wait(lock, [this] { return pendingCount_ == 0; });
}

}  // namespace nvMolKit

