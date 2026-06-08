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

#ifndef NVMOLKIT_SUBSTRUCT_WORKLOAD_H
#define NVMOLKIT_SUBSTRUCT_WORKLOAD_H

#include <GraphMol/ROMol.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "src/gpu_scheduler/cpu_fallback_queue.h"
#include "src/gpu_scheduler/workload.h"
#include "src/substruct/gpu_executor.h"
#include "src/substruct/minibatch_planner.h"
#include "src/substruct/molecules.h"
#include "src/substruct/molecules_device.cuh"
#include "src/substruct/pinned_buffer_pool.h"
#include "src/substruct/prepared_mini_batch.h"
#include "src/substruct/recursive_preprocessor.h"
#include "src/substruct/substruct_kernels.h"
#include "src/substruct/substruct_launch_config.h"
#include "src/substruct/substruct_results.h"
#include "src/substruct/substruct_search_internal.h"
#include "src/substruct/thread_worker_context.h"
#include "src/utils/nvtx.h"

namespace nvMolKit {

/**
 * @brief Inputs shared across all SubstructWorkload threads.
 *
 * Owned by the caller (typically getSubstructMatchesImpl) and held by
 * reference inside SubstructWorkload. Threads read these without
 * synchronization (read-only fields) or via the embedded mutex
 * (results-related fields).
 */
struct SubstructInputs {
  // Read-only inputs.
  const std::vector<const RDKit::ROMol*>* targets               = nullptr;
  const std::vector<const RDKit::ROMol*>* queriesRdkit          = nullptr;  // for fallback handler
  const MoleculesHost*                    queriesHost           = nullptr;
  const MoleculesDevice*                  queriesDevice         = nullptr;  // primary GPU's view
  const RecursivePatternPreprocessor*     recursivePreprocessor = nullptr;  // primary GPU's view
  const QueryPreprocessContext*           queryContext          = nullptr;
  SubstructAlgorithm                      algorithm             = SubstructAlgorithm::VF2;
  int                                     primaryDeviceId       = 0;     // device active when queriesDevice was built
  int                                     batchSize             = 1024;  // pairs per GPU mini-batch
  int                                     maxMatches            = 0;
  bool                                    countOnly             = false;

  // Result sinks (exactly one is non-null at a time).
  SubstructSearchResults*   results      = nullptr;
  HasSubstructMatchResults* boolResults  = nullptr;
  std::vector<int>*         countResults = nullptr;
  std::mutex*               resultsMutex = nullptr;  // shared with fallback queue

  // Workload-owned pinned buffer pool. Re-owned by the workload because
  // each pipeline-managed buffer carries substructure-specific layout.
  PinnedHostBufferPool* bufferPool = nullptr;

  // Optional CPU fallback queue. May be nullptr if every input is GPU-eligible.
  gpu_scheduler::CpuFallbackQueue<RDKitFallbackEntry>* fallbackQueue = nullptr;

  // Per-mini-batch derived sizes (computed once at setup, read everywhere).
  int targetsPerBatch     = 0;
  int maxPairsPerBatch    = 0;
  int maxPatternsPerDepth = 0;
};

/**
 * @brief Per-CUDA-device read-only state for substructure search.
 *
 * For the primary GPU (deviceId == primaryDeviceId), reuses the queriesDevice
 * and recursivePreprocessor that getSubstructMatchesImpl already built. For
 * any other GPU, allocates its own copies via copyFromHost / buildPatterns,
 * since neither type can be shared across CUDA contexts.
 */
struct SubstructPerGpu : gpu_scheduler::PerGpuState {
  // Either points to the caller-owned (primary) state, or to the locally-owned
  // unique_ptr below.
  const MoleculesDevice*              queriesDevice         = nullptr;
  const RecursivePatternPreprocessor* recursivePreprocessor = nullptr;

  std::unique_ptr<MoleculesDevice>              localQueries;
  std::unique_ptr<RecursivePatternPreprocessor> localPreprocessor;
};

/**
 * @brief Per-thread context for preprocessor threads.
 *
 * Caches reusable scratch vectors so we don't reallocate on every claim.
 * CPU fallback producers are registered on runner threads only
 * (`SubstructWorkload::makeRunnerCtx`) so the fallback queue stays open until
 * GPU-side overflow paths can enqueue.
 */
struct SubstructPreprocCtx : gpu_scheduler::PreprocThreadContext {
  std::vector<const RDKit::ROMol*> batchTargets;
  std::vector<int>                 batchOriginalIndices;
  std::vector<RDKitFallbackEntry>  fallbackEntries;
};

/**
 * @brief Per-runner-thread context: holds the fallback producer registration.
 */
struct SubstructRunnerCtx : gpu_scheduler::RunnerThreadContext {
  gpu_scheduler::FallbackProducerGuard<RDKitFallbackEntry> fallbackGuard;
};

/**
 * @brief Runtime-polymorphic substructure workload consumed by gpu_scheduler::Pipeline.
 *
 * Each method is a thin shim around existing substructure code. The actual
 * preprocess/dispatch/postprocess implementations live in substruct_workload.cu.
 */
class SubstructWorkload : public gpu_scheduler::Workload {
 public:
  using Inputs               = SubstructInputs;
  using PreparedBatch        = PreparedMiniBatch;
  using PerGpuState          = SubstructPerGpu;
  using GpuSlotState         = GpuExecutor;
  using PreprocThreadContext = SubstructPreprocCtx;
  using RunnerThreadContext  = SubstructRunnerCtx;

  explicit SubstructWorkload(Inputs& inputs) : inputs_(inputs) {}

  int totalUnits() const override;
  int unitsPerPreprocBatch() const override;

  std::unique_ptr<gpu_scheduler::PreprocThreadContext> makePreprocCtx() override;
  std::unique_ptr<gpu_scheduler::RunnerThreadContext>  makeRunnerCtx() override;

  void preprocess(gpu_scheduler::IndexRange            range,
                  gpu_scheduler::PreprocThreadContext& ctx,
                  const gpu_scheduler::PushBatch&      pushBatch) override;

  std::unique_ptr<gpu_scheduler::PerGpuState>  makePerGpuState(int gpuId) override;
  std::unique_ptr<gpu_scheduler::GpuSlotState> makeSlotState(gpu_scheduler::PerGpuState& pgs, int gpuId) override;

  void dispatchAndCopyBack(gpu_scheduler::GpuSlotState&        slot,
                           gpu_scheduler::PerGpuState&         pgs,
                           gpu_scheduler::PreparedBatch&       batch,
                           gpu_scheduler::RunnerThreadContext& ctx) override;

  void postprocess(gpu_scheduler::GpuSlotState&        slot,
                   gpu_scheduler::PreparedBatch&       batch,
                   gpu_scheduler::RunnerThreadContext& ctx) override;

  void onAbort() override;

 private:
  Inputs& inputs_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_WORKLOAD_H
