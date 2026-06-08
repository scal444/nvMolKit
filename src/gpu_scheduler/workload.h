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

#ifndef NVMOLKIT_GPU_SCHEDULER_WORKLOAD_H
#define NVMOLKIT_GPU_SCHEDULER_WORKLOAD_H

#include <cuda_runtime.h>

#include <functional>
#include <memory>

namespace nvMolKit::gpu_scheduler {

/**
 * @brief Half-open input range claimed atomically by a preprocessing thread.
 */
struct IndexRange {
  int start = 0;
  int end   = 0;

  [[nodiscard]] int size() const { return end - start; }
};

struct PreparedBatch {
  virtual ~PreparedBatch() = default;
};

struct PerGpuState {
  virtual ~PerGpuState() = default;
};

struct GpuSlotState {
  virtual ~GpuSlotState() = default;

  [[nodiscard]] virtual cudaStream_t primaryStream() const   = 0;
  [[nodiscard]] virtual cudaEvent_t  completionEvent() const = 0;
};

struct PreprocThreadContext {
  virtual ~PreprocThreadContext() = default;
};

struct RunnerThreadContext {
  virtual ~RunnerThreadContext() = default;
};

using PushBatch = std::function<void(std::unique_ptr<PreparedBatch>)>;

/**
 * @brief Runtime-polymorphic workload interface consumed by Pipeline.
 *
 * Workload implementations own their domain-specific Inputs reference and
 * downcast scheduler base objects at this boundary. Pipeline stays concerned
 * only with thread orchestration, queueing, slot completion, and abort wiring.
 */
class Workload {
 public:
  virtual ~Workload() = default;

  [[nodiscard]] virtual int totalUnits() const           = 0;
  [[nodiscard]] virtual int unitsPerPreprocBatch() const = 0;

  [[nodiscard]] virtual std::unique_ptr<PreprocThreadContext> makePreprocCtx() {
    return std::make_unique<PreprocThreadContext>();
  }
  [[nodiscard]] virtual std::unique_ptr<RunnerThreadContext> makeRunnerCtx() {
    return std::make_unique<RunnerThreadContext>();
  }

  virtual void preprocess(IndexRange range, PreprocThreadContext& ctx, const PushBatch& pushBatch) = 0;

  [[nodiscard]] virtual std::unique_ptr<PerGpuState>  makePerGpuState(int gpuId)                         = 0;
  [[nodiscard]] virtual std::unique_ptr<GpuSlotState> makeSlotState(PerGpuState& perGpuState, int gpuId) = 0;

  virtual void dispatchAndCopyBack(GpuSlotState&        slot,
                                   PerGpuState&         perGpuState,
                                   PreparedBatch&       batch,
                                   RunnerThreadContext& ctx)                                   = 0;
  virtual void postprocess(GpuSlotState& slot, PreparedBatch& batch, RunnerThreadContext& ctx) = 0;

  virtual void onAbort() {}
};

}  // namespace nvMolKit::gpu_scheduler

#endif  // NVMOLKIT_GPU_SCHEDULER_WORKLOAD_H
