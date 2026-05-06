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

#ifndef NVMOLKIT_GPU_SCHEDULER_PIPELINE_H
#define NVMOLKIT_GPU_SCHEDULER_PIPELINE_H

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "config.h"
#include "device.h"
#include "exception_aggregator.h"
#include "thread_safe_queue.h"

namespace nvMolKit::gpu_scheduler {

/**
 * @brief Half-open input range claimed atomically by a preprocessing thread.
 */
struct IndexRange {
  int start = 0;
  int end   = 0;

  [[nodiscard]] int size() const { return end - start; }
};

/**
 * @brief Workload defines `static void onAbort(Inputs&)`.
 *
 * Used by the pipeline to optionally unblock workload-owned primitives after a
 * worker failure. Workloads omit `onAbort` entirely if they do not need it.
 */
template <class Workload>
concept WorkloadHasOnAbort = requires(typename Workload::Inputs& inputs) {
  { Workload::onAbort(inputs) } -> std::same_as<void>;
};

/**
 * @brief Workload defines `static PreprocThreadContext makePreprocCtx(Inputs&)`.
 *
 * When absent, each preprocessor thread uses a default-constructed
 * `PreprocThreadContext`.
 */
template <class Workload>
concept WorkloadHasMakePreprocCtx = requires(typename Workload::Inputs& inputs) {
  { Workload::makePreprocCtx(inputs) } -> std::same_as<typename Workload::PreprocThreadContext>;
};

/**
 * @brief Workload defines `static RunnerThreadContext makeRunnerCtx(Inputs&)`.
 *
 * When absent, each runner thread uses a default-constructed
 * `RunnerThreadContext`.
 */
template <class Workload>
concept WorkloadHasMakeRunnerCtx = requires(typename Workload::Inputs& inputs) {
  { Workload::makeRunnerCtx(inputs) } -> std::same_as<typename Workload::RunnerThreadContext>;
};

// =============================================================================
// Workload contract
// =============================================================================
//
// Pipeline<Workload> is parameterized by a Workload trait struct that the
// caller defines. The Workload provides the following types and static
// methods:
//
//   struct Workload {
//     // Data shared across all threads. Pipeline holds a reference; the
//     // workload owns it and may put pinned buffer pools, results sinks,
//     // CPU-fallback queues, etc. on it.
//     using Inputs = ...;
//
//     // One unit of work produced by preprocess() and consumed by
//     // dispatchAndCopyBack(). Owned by std::unique_ptr inside the queue.
//     using PreparedBatch = ...;
//
//     // Per-CUDA-device read-only state (e.g. uploaded query patterns,
//     // device-resident reference data). Constructed once per device.
//     using PerGpuState = ...;
//
//     // Per-runner-slot mutable state (streams, events, device buffers).
//     // Each runner owns slotsPerWorker of these.
//     using GpuSlotState = ...;
//
//     // Per-thread helpers. Constructed at thread start (with the device
//     // active for runner threads, before any user code runs); destroyed at
//     // thread exit. Use this to install RAII guards that need to outlive
//     // every preprocess()/dispatch()/postprocess() call on a thread (e.g.
//     // FallbackProducerGuards keeping a CpuFallbackQueue open across the
//     // thread's lifetime).
//     using PreprocThreadContext = ...;
//     using RunnerThreadContext  = ...;
//
//     // Optional. Construct the per-thread context. Called once per thread,
//     // on that thread, before any work begins. If omitted, the context type
//     // is default-constructed. Satisfy gpu_scheduler::WorkloadHasMakePreprocCtx /
//     // WorkloadHasMakeRunnerCtx by defining the corresponding static method.
//     static PreprocThreadContext makePreprocCtx(Inputs&);
//     static RunnerThreadContext  makeRunnerCtx(Inputs&);
//
//     // Total input unit count. Drives the global atomic claim counter.
//     static int totalUnits(Inputs&);
//
//     // Number of input units one preprocess() call should claim at a time.
//     // The workload computes this from its own batchSize knob.
//     static int unitsPerPreprocBatch(Inputs&);
//
//     // Build PreparedBatch(es) for a claimed range and push each via pushBatch.
//     // May push zero, one, or many batches per call (e.g. when the workload
//     // splits a claim into multiple GPU-sized mini-batches). May also route
//     // some inputs through a CPU fallback queue owned by Inputs.
//     //
//     // Should call ExceptionAggregator::aborted() periodically inside long
//     // loops to bail out early if a sibling thread has failed. The Pipeline
//     // automatically catches exceptions thrown by this method.
//     template <class PushFn>
//     static void preprocess(Inputs&,
//                            IndexRange,
//                            PreprocThreadContext&,
//                            PushFn pushBatch);
//
//     // Construct per-device state. Called once per entry in Config::gpuIds
//     // on the coordinator thread, with the device active via WithDevice.
//     // Returned by std::unique_ptr so the workload can host non-movable types
//     // (e.g. anything wrapping ScopedStream / ScopedCudaEvent).
//     static std::unique_ptr<PerGpuState> makePerGpuState(Inputs&, int gpuId);
//
//     // Construct one runner-slot state (streams, device buffers, etc.).
//     // Called slotsPerWorker times per runner thread, with the device active.
//     static std::unique_ptr<GpuSlotState> makeSlotState(Inputs&, PerGpuState&, int gpuId);
//
//     // Issue H2D copies, kernels, and D2H copies all stream-async on the
//     // slot's primary stream (slot.primaryStream() must return a valid
//     // cudaStream_t — see GpuSlotConcept). Pipeline records a cudaEvent on
//     // that stream after this call returns and waits on the event before
//     // calling postprocess.
//     static void dispatchAndCopyBack(GpuSlotState&,
//                                     PerGpuState&,
//                                     PreparedBatch&,
//                                     Inputs&,
//                                     RunnerThreadContext&);
//
//     // Merge results into the workload's shared output. Runs after the
//     // event from dispatchAndCopyBack has fired. The workload owns any
//     // mutex needed to serialize with siblings.
//     static void postprocess(GpuSlotState&,
//                             PreparedBatch&,
//                             Inputs&,
//                             RunnerThreadContext&);
//
//     // Optional. Called once if any worker thread throws. Runs on the
//     // thread that detected the abort. Use this to unblock any custom
//     // blocking primitives the workload owns (e.g. shutdown a pinned
//     // buffer pool whose acquire() is blocking a preprocessor thread).
//     // Satisfy gpu_scheduler::WorkloadHasOnAbort by defining this method.
//     static void onAbort(Inputs&);
//   };
//
// The slot type must expose its primary stream so the pipeline can record
// the completion event. We require:
//
//   cudaStream_t Workload::GpuSlotState::primaryStream() const;
//   cudaEvent_t  Workload::GpuSlotState::completionEvent() const;
//
// (The slot owns the event; the pipeline records and waits on it.)
// =============================================================================

namespace detail {

/**
 * @brief One in-flight slot waiting on its completionEvent to fire.
 *
 * The runner ring stores these and drains the head when it's full.
 */
template <class Workload> struct InFlightSlot {
  typename Workload::GpuSlotState*                  state = nullptr;
  std::unique_ptr<typename Workload::PreparedBatch> batch;
};

}  // namespace detail

/**
 * @brief Reusable preprocess-pool / per-GPU-coord / per-runner-slot pipeline.
 *
 * This is a pure orchestration scaffold. All workload-specific logic lives
 * behind the Workload trait struct. See the Workload contract block above.
 *
 * Usage:
 *   MyWorkload::Inputs inputs = ...;
 *   gpu_scheduler::Config config;
 *   config.gpuIds = {0, 1};
 *   gpu_scheduler::Pipeline<MyWorkload> pipeline(config, inputs);
 *   pipeline.run();    // blocks; rethrows worker exceptions on the caller
 */
template <class Workload> class Pipeline {
 public:
  using Inputs               = typename Workload::Inputs;
  using PreparedBatch        = typename Workload::PreparedBatch;
  using PerGpuState          = typename Workload::PerGpuState;
  using GpuSlotState         = typename Workload::GpuSlotState;
  using PreprocThreadContext = typename Workload::PreprocThreadContext;
  using RunnerThreadContext  = typename Workload::RunnerThreadContext;

  /**
   * @brief Construct with user config and a reference to workload Inputs.
   *
   * The Inputs reference must outlive run(). Pipeline does not copy or own it.
   */
  Pipeline(const Config& config, Inputs& inputs) : config_(config), inputs_(inputs) {}

  Pipeline(const Pipeline&)            = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  Pipeline(Pipeline&&)                 = delete;
  Pipeline& operator=(Pipeline&&)      = delete;

  /**
   * @brief Run the pipeline to completion.
   *
   * Blocks until every input unit has been preprocessed and every dispatched
   * batch has been postprocessed. If any worker thread (preprocessor or
   * runner) throws, the exception is recorded, all sibling threads are
   * signaled to abort, and the first exception is rethrown from run() after
   * joining.
   */
  void run() {
    int currentDevice = 0;
    cudaCheckError(cudaGetDevice(&currentDevice));

    const ResolvedConfig resolved = resolve(config_, std::thread::hardware_concurrency(), currentDevice);

    const int totalUnits = Workload::totalUnits(inputs_);
    if (totalUnits <= 0) {
      return;
    }
    const int unitsPerClaim = std::max(1, Workload::unitsPerPreprocBatch(inputs_));

    const int numGpus       = static_cast<int>(resolved.gpuIds.size());
    const int totalRunners  = std::min(resolved.workerThreadsPerGpu * numGpus, totalUnits);
    if (totalRunners <= 0) {
      return;
    }

    // Distribute runners across GPUs (round-robin remainder).
    std::vector<int> runnersPerGpu(numGpus, totalRunners / numGpus);
    for (int i = 0; i < totalRunners % numGpus; ++i) {
      runnersPerGpu[i]++;
    }

    ThreadSafeQueue<std::unique_ptr<PreparedBatch>> queue;
    std::atomic<int>                                nextUnit{0};
    ExceptionAggregator                             errors;

    // Coordinators must be launched before preprocessors so the queue has
    // consumers ready. Each coordinator owns its PerGpuState and its runner
    // threads.
    std::vector<std::thread> coordinators;
    coordinators.reserve(static_cast<size_t>(numGpus));
    for (int gpuIdx = 0; gpuIdx < numGpus; ++gpuIdx) {
      const int deviceId  = resolved.gpuIds[gpuIdx];
      const int numRunners = runnersPerGpu[gpuIdx];
      if (numRunners == 0) {
        continue;
      }
      coordinators.emplace_back([this, deviceId, numRunners, slotsPerWorker = resolved.slotsPerWorker, &queue, &errors] {
        runCoordinator(deviceId, numRunners, slotsPerWorker, queue, errors);
      });
    }

    std::vector<std::thread> preprocessors;
    preprocessors.reserve(static_cast<size_t>(resolved.globalPreprocessingThreads));
    for (int i = 0; i < resolved.globalPreprocessingThreads; ++i) {
      preprocessors.emplace_back(
        [this, totalUnits, unitsPerClaim, &nextUnit, &queue, &errors] {
          runPreprocessor(totalUnits, unitsPerClaim, nextUnit, queue, errors);
        });
    }

    for (auto& thread : preprocessors) {
      thread.join();
    }
    queue.close();
    for (auto& thread : coordinators) {
      thread.join();
    }

    errors.rethrowIfAny();
  }

 private:
  void runPreprocessor(int                                              totalUnits,
                       int                                              unitsPerClaim,
                       std::atomic<int>&                                nextUnit,
                       ThreadSafeQueue<std::unique_ptr<PreparedBatch>>& queue,
                       ExceptionAggregator&                             errors) {
    try {
      PreprocThreadContext ctx{};
      if constexpr (WorkloadHasMakePreprocCtx<Workload>) {
        ctx = Workload::makePreprocCtx(inputs_);
      }
      auto pushBatch = [&queue](std::unique_ptr<PreparedBatch> batch) {
        if (batch) {
          queue.push(std::move(batch));
        }
      };

      while (true) {
        if (errors.aborted()) {
          break;
        }
        const int start = nextUnit.fetch_add(unitsPerClaim, std::memory_order_relaxed);
        if (start >= totalUnits) {
          break;
        }
        const int  end = std::min(start + unitsPerClaim, totalUnits);
        IndexRange range{start, end};
        Workload::preprocess(inputs_, range, ctx, pushBatch);
      }
    } catch (...) {
      errors.recordAndAbort(std::current_exception());
      queue.close();
      if constexpr (WorkloadHasOnAbort<Workload>) {
        Workload::onAbort(inputs_);
      }
    }
  }

  void runCoordinator(int                                              deviceId,
                      int                                              numRunners,
                      int                                              slotsPerWorker,
                      ThreadSafeQueue<std::unique_ptr<PreparedBatch>>& queue,
                      ExceptionAggregator&                             errors) {
    try {
      const WithDevice setDevice(deviceId);

      std::unique_ptr<PerGpuState> perGpuState = Workload::makePerGpuState(inputs_, deviceId);

      // Each runner owns slotsPerWorker GpuSlotState instances. Pre-allocate
      // them all on the coordinator so device construction order is
      // deterministic, then hand pointer slices to the runner threads.
      const int totalSlots = numRunners * slotsPerWorker;
      std::vector<std::unique_ptr<GpuSlotState>> slots;
      slots.reserve(static_cast<size_t>(totalSlots));
      for (int s = 0; s < totalSlots; ++s) {
        slots.push_back(Workload::makeSlotState(inputs_, *perGpuState, deviceId));
      }

      std::vector<std::thread> runners;
      runners.reserve(static_cast<size_t>(numRunners));
      for (int r = 0; r < numRunners; ++r) {
        std::vector<GpuSlotState*> mySlots;
        mySlots.reserve(static_cast<size_t>(slotsPerWorker));
        for (int s = 0; s < slotsPerWorker; ++s) {
          mySlots.push_back(slots[r * slotsPerWorker + s].get());
        }
        runners.emplace_back([this, deviceId, mySlots = std::move(mySlots), pgs = perGpuState.get(), &queue, &errors] {
          runRunner(deviceId, mySlots, *pgs, queue, errors);
        });
      }

      for (auto& runner : runners) {
        runner.join();
      }
    } catch (...) {
      errors.recordAndAbort(std::current_exception());
      queue.close();
      if constexpr (WorkloadHasOnAbort<Workload>) {
        Workload::onAbort(inputs_);
      }
    }
  }

  void runRunner(int                                              deviceId,
                 std::vector<GpuSlotState*>                       slots,
                 PerGpuState&                                     perGpuState,
                 ThreadSafeQueue<std::unique_ptr<PreparedBatch>>& queue,
                 ExceptionAggregator&                             errors) {
    try {
      const WithDevice setDevice(deviceId);
      RunnerThreadContext ctx{};
      if constexpr (WorkloadHasMakeRunnerCtx<Workload>) {
        ctx = Workload::makeRunnerCtx(inputs_);
      }

      const int                                  numSlots = static_cast<int>(slots.size());
      std::vector<detail::InFlightSlot<Workload>> pending(static_cast<size_t>(numSlots));
      int pendingHead  = 0;
      int pendingTail  = 0;
      int pendingCount = 0;

      auto drainOne = [&]() {
        auto& head = pending[static_cast<size_t>(pendingHead)];
        cudaCheckError(cudaEventSynchronize(head.state->completionEvent()));
        Workload::postprocess(*head.state, *head.batch, inputs_, ctx);
        head.batch.reset();
        head.state  = nullptr;
        pendingHead = (pendingHead + 1) % numSlots;
        --pendingCount;
      };

      while (true) {
        if (errors.aborted() && pendingCount == 0) {
          break;
        }

        if (pendingCount == numSlots) {
          drainOne();
          continue;
        }

        std::unique_ptr<PreparedBatch> batch;
        if (pendingCount > 0) {
          // Slots are partially filled: don't block on the queue if it's
          // momentarily empty; instead drain the oldest in-flight batch so
          // the GPU stays busy.
          auto opt = queue.tryPop();
          if (!opt) {
            drainOne();
            continue;
          }
          batch = std::move(*opt);
        } else {
          // Fully drained: it's safe to block; nothing useful to do until
          // either a new batch arrives or the queue closes.
          auto opt = queue.pop();
          if (!opt) {
            break;
          }
          batch = std::move(*opt);
        }

        GpuSlotState* slot = slots[static_cast<size_t>(pendingTail)];
        Workload::dispatchAndCopyBack(*slot, perGpuState, *batch, inputs_, ctx);
        cudaCheckError(cudaEventRecord(slot->completionEvent(), slot->primaryStream()));

        pending[static_cast<size_t>(pendingTail)].state = slot;
        pending[static_cast<size_t>(pendingTail)].batch = std::move(batch);
        pendingTail                                     = (pendingTail + 1) % numSlots;
        ++pendingCount;
      }

      while (pendingCount > 0) {
        drainOne();
      }
    } catch (...) {
      errors.recordAndAbort(std::current_exception());
      queue.close();
      if constexpr (WorkloadHasOnAbort<Workload>) {
        Workload::onAbort(inputs_);
      }
    }
  }

  const Config& config_;
  Inputs&       inputs_;
};

}  // namespace nvMolKit::gpu_scheduler

#endif  // NVMOLKIT_GPU_SCHEDULER_PIPELINE_H
