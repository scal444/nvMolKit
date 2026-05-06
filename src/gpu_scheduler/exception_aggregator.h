// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMOLKIT_GPU_SCHEDULER_EXCEPTION_AGGREGATOR_H
#define NVMOLKIT_GPU_SCHEDULER_EXCEPTION_AGGREGATOR_H

#include <atomic>
#include <exception>
#include <mutex>
#include <utility>
#include <vector>

namespace nvMolKit::gpu_scheduler {

/**
 * @brief Collects exceptions thrown by worker threads and rethrows the first one.
 *
 * Worker threads call recordAndAbort() with std::current_exception() inside
 * a catch (...) block. The first recorded exception is preserved exactly.
 * Subsequent exceptions are still recorded so destructors can observe them
 * via storedCount(), but rethrow() always rethrows the first.
 *
 * The aborted() flag is checked from worker hot loops to bail out early
 * once any sibling has failed.
 */
class ExceptionAggregator {
 public:
  ExceptionAggregator() = default;

  ExceptionAggregator(const ExceptionAggregator&)            = delete;
  ExceptionAggregator& operator=(const ExceptionAggregator&) = delete;
  ExceptionAggregator(ExceptionAggregator&&)                 = delete;
  ExceptionAggregator& operator=(ExceptionAggregator&&)      = delete;

  /**
   * @brief Record an exception and set the abort flag.
   *
   * Safe to call from any thread. Always sets aborted() even if a previous
   * exception was already stored.
   */
  void recordAndAbort(std::exception_ptr ex) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stored_.push_back(std::move(ex));
    }
    aborted_.store(true, std::memory_order_release);
  }

  /// Hot-path test for early bailout from worker loops.
  [[nodiscard]] bool aborted() const noexcept { return aborted_.load(std::memory_order_acquire); }

  /// Total number of exceptions recorded (for diagnostics).
  [[nodiscard]] std::size_t storedCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stored_.size();
  }

  /**
   * @brief Rethrow the first stored exception, if any.
   *
   * Call this on the main thread after joining all workers. No-op if
   * no exception was recorded.
   */
  void rethrowIfAny() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stored_.empty()) {
      std::rethrow_exception(stored_.front());
    }
  }

 private:
  mutable std::mutex              mutex_;
  std::vector<std::exception_ptr> stored_;
  std::atomic<bool>               aborted_{false};
};

}  // namespace nvMolKit::gpu_scheduler

#endif  // NVMOLKIT_GPU_SCHEDULER_EXCEPTION_AGGREGATOR_H
