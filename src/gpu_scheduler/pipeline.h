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
#include <cstddef>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "src/gpu_scheduler/config.h"
#include "src/gpu_scheduler/exception_aggregator.h"
#include "src/gpu_scheduler/workload.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "src/utils/thread_safe_queue.h"

namespace nvMolKit::gpu_scheduler {

namespace detail {

/**
 * @brief One in-flight slot waiting on its completionEvent to fire.
 */
struct InFlightSlot {
  GpuSlotState*                  state = nullptr;
  std::unique_ptr<PreparedBatch> batch;
};

}  // namespace detail

/**
 * @brief Reusable preprocess-pool / per-GPU-coord / per-runner-slot pipeline.
 *
 * This is orchestration-only: all workload-specific preprocessing, GPU state,
 * dispatch, and result merge behavior lives behind the Workload interface.
 */
class Pipeline {
 public:
  /**
   * @brief Construct with user config and a workload implementation.
   *
   * The workload reference must outlive run(). Pipeline does not copy or own it.
   */
  Pipeline(const Config& config, Workload& workload) : config_(config), workload_(workload) {}

  Pipeline(const Pipeline&)            = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  Pipeline(Pipeline&&)                 = delete;
  Pipeline& operator=(Pipeline&&)      = delete;

  /**
   * @brief Run the pipeline to completion.
   *
   * Blocks until every input unit has been preprocessed and every dispatched
   * batch has been postprocessed. If any worker thread throws, the exception
   * is recorded, all sibling threads are signaled to abort, and the first
   * exception is rethrown from run() after joining.
   */
  void run() {
    int currentDevice = 0;
    cudaCheckError(cudaGetDevice(&currentDevice));

    const ResolvedConfig resolved = resolve(config_, std::thread::hardware_concurrency(), currentDevice);

    const int totalUnits = workload_.totalUnits();
    if (totalUnits <= 0) {
      return;
    }
    const int unitsPerClaim = std::max(1, workload_.unitsPerPreprocBatch());

    const int numGpus      = static_cast<int>(resolved.gpuIds.size());
    const int totalRunners = std::min(resolved.workerThreadsPerGpu * numGpus, totalUnits);
    if (totalRunners <= 0) {
      return;
    }

    std::vector<int> runnersPerGpu(numGpus, totalRunners / numGpus);
    for (int i = 0; i < totalRunners % numGpus; ++i) {
      runnersPerGpu[i]++;
    }

    ThreadSafeQueue<std::unique_ptr<PreparedBatch>> queue;
    std::atomic<int>                                nextUnit{0};
    ExceptionAggregator                             errors;

    // Coordinators are launched before preprocessors so the queue has
    // consumers ready. Each coordinator owns its PerGpuState and runner slots.
    std::vector<std::thread> coordinators;
    coordinators.reserve(static_cast<size_t>(numGpus));
    for (int gpuIdx = 0; gpuIdx < numGpus; ++gpuIdx) {
      const int deviceId   = resolved.gpuIds[gpuIdx];
      const int numRunners = runnersPerGpu[gpuIdx];
      if (numRunners == 0) {
        continue;
      }
      coordinators.emplace_back(
        [this, deviceId, numRunners, slotsPerWorker = resolved.slotsPerWorker, &queue, &errors] {
          runCoordinator(deviceId, numRunners, slotsPerWorker, queue, errors);
        });
    }

    std::vector<std::thread> preprocessors;
    preprocessors.reserve(static_cast<size_t>(resolved.globalPreprocessingThreads));
    for (int i = 0; i < resolved.globalPreprocessingThreads; ++i) {
      preprocessors.emplace_back([this, totalUnits, unitsPerClaim, &nextUnit, &queue, &errors] {
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
  void abort(ThreadSafeQueue<std::unique_ptr<PreparedBatch>>& queue, ExceptionAggregator& errors) {
    errors.recordAndAbort(std::current_exception());
    queue.close();
    try {
      workload_.onAbort();
    } catch (...) {
      errors.recordAndAbort(std::current_exception());
    }
  }

  void runPreprocessor(int                                              totalUnits,
                       int                                              unitsPerClaim,
                       std::atomic<int>&                                nextUnit,
                       ThreadSafeQueue<std::unique_ptr<PreparedBatch>>& queue,
                       ExceptionAggregator&                             errors) {
    try {
      std::unique_ptr<PreprocThreadContext> ctx       = workload_.makePreprocCtx();
      const PushBatch                       pushBatch = [&queue](std::unique_ptr<PreparedBatch> batch) {
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
        workload_.preprocess(range, *ctx, pushBatch);
      }
    } catch (...) {
      abort(queue, errors);
    }
  }

  void runCoordinator(int                                              deviceId,
                      int                                              numRunners,
                      int                                              slotsPerWorker,
                      ThreadSafeQueue<std::unique_ptr<PreparedBatch>>& queue,
                      ExceptionAggregator&                             errors) {
    try {
      const WithDevice setDevice(deviceId);

      std::unique_ptr<PerGpuState> perGpuState = workload_.makePerGpuState(deviceId);

      const int                                  totalSlots = numRunners * slotsPerWorker;
      std::vector<std::unique_ptr<GpuSlotState>> slots;
      slots.reserve(static_cast<size_t>(totalSlots));
      for (int s = 0; s < totalSlots; ++s) {
        slots.push_back(workload_.makeSlotState(*perGpuState, deviceId));
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
      abort(queue, errors);
    }
  }

  void runRunner(int                                              deviceId,
                 std::vector<GpuSlotState*>                       slots,
                 PerGpuState&                                     perGpuState,
                 ThreadSafeQueue<std::unique_ptr<PreparedBatch>>& queue,
                 ExceptionAggregator&                             errors) {
    try {
      const WithDevice setDevice(deviceId);

      std::unique_ptr<RunnerThreadContext> ctx = workload_.makeRunnerCtx();

      const int                         numSlots = static_cast<int>(slots.size());
      std::vector<detail::InFlightSlot> pending(static_cast<size_t>(numSlots));
      int                               pendingHead  = 0;
      int                               pendingTail  = 0;
      int                               pendingCount = 0;

      auto drainOne = [&]() {
        auto& head = pending[static_cast<size_t>(pendingHead)];
        cudaCheckError(cudaEventSynchronize(head.state->completionEvent()));
        workload_.postprocess(*head.state, *head.batch, *ctx);
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
          auto opt = queue.tryPop();
          if (!opt) {
            drainOne();
            continue;
          }
          batch = std::move(*opt);
        } else {
          auto opt = queue.pop();
          if (!opt) {
            break;
          }
          batch = std::move(*opt);
        }

        GpuSlotState* slot = slots[static_cast<size_t>(pendingTail)];
        workload_.dispatchAndCopyBack(*slot, perGpuState, *batch, *ctx);
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
      abort(queue, errors);
    }
  }

  const Config& config_;
  Workload&     workload_;
};

}  // namespace nvMolKit::gpu_scheduler

#endif  // NVMOLKIT_GPU_SCHEDULER_PIPELINE_H
