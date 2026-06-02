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

#ifndef NVMOLKIT_GPU_SCHEDULER_CPU_FALLBACK_QUEUE_H
#define NVMOLKIT_GPU_SCHEDULER_CPU_FALLBACK_QUEUE_H

#include <functional>
#include <mutex>
#include <utility>
#include <vector>

#include "src/utils/thread_safe_queue.h"

namespace nvMolKit::gpu_scheduler {

/**
 * @brief MPMC queue of CPU-fallback entries, drained opportunistically.
 *
 * Generalizes substructure's RDKitFallbackQueue. A workload that supports a
 * CPU code path for inputs the GPU can't handle (oversized molecules,
 * algorithm bailouts, etc.) constructs one of these and hands it to its
 * Workload::preprocess() and Workload::postprocess() implementations.
 *
 * Producers must register/unregister via FallbackProducerGuard. When the last
 * producer leaves and the queue is empty, the queue closes and waiting
 * consumers exit. This matches the substructure model where preprocessing
 * threads enqueue oversize-target work and GPU runner threads opportunistically
 * drain it whenever they're waiting on cudaEventSynchronize.
 *
 * @tparam Entry  The work item. Must be movable. Typically a small struct
 *                naming a (target, query) pair or a single molecule index.
 */
template <typename Entry> class CpuFallbackQueue {
 public:
  using Handler = std::function<void(const Entry&)>;

  /**
   * @brief Construct the queue with the per-entry processing function.
   *
   * @param handler  Called once per dequeued entry on whichever thread is
   *                 draining the queue. Must be thread-safe (typically grabs
   *                 a results mutex internally).
   */
  explicit CpuFallbackQueue(Handler handler) : handler_(std::move(handler)) {}

  CpuFallbackQueue(const CpuFallbackQueue&)            = delete;
  CpuFallbackQueue& operator=(const CpuFallbackQueue&) = delete;
  CpuFallbackQueue(CpuFallbackQueue&&)                 = delete;
  CpuFallbackQueue& operator=(CpuFallbackQueue&&)      = delete;

  /// Enqueue a single entry. Safe to call without a producer guard, but
  /// callers should typically hold one so the queue doesn't close mid-stream.
  void enqueue(Entry entry) { queue_.push(std::move(entry)); }

  /// Bulk enqueue. The container is moved-from (each element moved).
  template <typename Container> void enqueueBatch(Container&& entries) {
    queue_.pushBatch(std::forward<Container>(entries));
  }

  /**
   * @brief Pop one entry and invoke the handler if available.
   *
   * Non-blocking. Returns true if an entry was processed, false if the queue
   * was empty (regardless of whether it's closed).
   */
  bool tryProcessOne() {
    auto entry = queue_.tryPop();
    if (!entry) {
      return false;
    }
    handler_(*entry);
    return true;
  }

  /// Returns true if the queue currently has at least one entry. Snapshot only.
  [[nodiscard]] bool hasWork() const { return !queue_.empty(); }

  /// Number of producer-registered threads. For diagnostics.
  [[nodiscard]] int activeProducers() const {
    std::lock_guard<std::mutex> lock(producerMutex_);
    return activeProducers_;
  }

 private:
  template <typename E> friend class FallbackProducerGuard;

  void registerProducer() {
    std::lock_guard<std::mutex> lock(producerMutex_);
    ++activeProducers_;
  }

  void unregisterProducer() {
    {
      std::lock_guard<std::mutex> lock(producerMutex_);
      --activeProducers_;
      if (activeProducers_ != 0) {
        return;
      }
    }
    queue_.close();
  }

  ThreadSafeQueue<Entry> queue_;
  Handler                handler_;
  mutable std::mutex     producerMutex_;
  int                    activeProducers_ = 0;
};

/**
 * @brief RAII helper that registers a producer for the lifetime of the scope.
 *
 * Threads that may enqueue fallback entries hold one of these. The queue
 * closes once the last guard is destroyed, so consumers waiting on the queue
 * exit cleanly without needing an explicit shutdown signal.
 *
 * The guard is movable so it can be installed lazily into a thread-local
 * context (e.g. on the first preprocess() call). A default-constructed guard
 * is inert; assigning a fresh one transfers registration.
 */
template <typename Entry> class FallbackProducerGuard {
 public:
  FallbackProducerGuard() noexcept = default;

  explicit FallbackProducerGuard(CpuFallbackQueue<Entry>* queue) : queue_(queue) {
    if (queue_ != nullptr) {
      queue_->registerProducer();
    }
  }

  ~FallbackProducerGuard() { releaseIfHeld(); }

  FallbackProducerGuard(const FallbackProducerGuard&)            = delete;
  FallbackProducerGuard& operator=(const FallbackProducerGuard&) = delete;

  FallbackProducerGuard(FallbackProducerGuard&& other) noexcept : queue_(other.queue_) { other.queue_ = nullptr; }

  FallbackProducerGuard& operator=(FallbackProducerGuard&& other) noexcept {
    if (this != &other) {
      releaseIfHeld();
      queue_       = other.queue_;
      other.queue_ = nullptr;
    }
    return *this;
  }

  /// True if this guard currently owns a producer registration.
  [[nodiscard]] bool registered() const noexcept { return queue_ != nullptr; }

 private:
  void releaseIfHeld() noexcept {
    if (queue_ != nullptr) {
      queue_->unregisterProducer();
      queue_ = nullptr;
    }
  }

  CpuFallbackQueue<Entry>* queue_ = nullptr;
};

}  // namespace nvMolKit::gpu_scheduler

#endif  // NVMOLKIT_GPU_SCHEDULER_CPU_FALLBACK_QUEUE_H
