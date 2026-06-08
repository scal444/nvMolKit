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
#include "src/gpu_scheduler/pipeline.h"
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
 * Owned by the caller (typically getSubstructMatchesImpl) and held by reference
 * inside Pipeline<SubstructWorkload>. Threads read these without
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
struct SubstructPerGpu {
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
struct SubstructPreprocCtx {
  std::vector<const RDKit::ROMol*> batchTargets;
  std::vector<int>                 batchOriginalIndices;
  std::vector<RDKitFallbackEntry>  fallbackEntries;
};

/**
 * @brief Per-runner-thread context: holds the fallback producer registration.
 */
struct SubstructRunnerCtx {
  gpu_scheduler::FallbackProducerGuard<RDKitFallbackEntry> fallbackGuard;
};

/**
 * @brief Trait struct wired into Pipeline<SubstructWorkload>.
 *
 * Each method is a thin shim around existing substructure code. The actual
 * preprocess/dispatch/postprocess implementations live in substruct_workload.cu.
 */
struct SubstructWorkload {
  using Inputs               = SubstructInputs;
  using PreparedBatch        = PreparedMiniBatch;
  using PerGpuState          = SubstructPerGpu;
  using GpuSlotState         = GpuExecutor;
  using PreprocThreadContext = SubstructPreprocCtx;
  using RunnerThreadContext  = SubstructRunnerCtx;

  static int totalUnits(Inputs& inputs);
  static int unitsPerPreprocBatch(Inputs& inputs);

  static RunnerThreadContext makeRunnerCtx(Inputs& inputs);

  /// `preprocess` is the one trait method that has to live in the header:
  /// the pipeline instantiates it per-runner with a caller-supplied `PushFn`
  /// (a lambda type) that the pipeline owns. All other methods are defined
  /// in substruct_workload.cu.
  template <class PushFn>
  static void preprocess(Inputs& inputs, gpu_scheduler::IndexRange range, PreprocThreadContext& ctx, PushFn pushBatch) {
    ScopedNvtxRange  claimRange("SubstructWorkload::preprocess");
    MiniBatchPlanner planner;

    const auto& targets         = *inputs.targets;
    const auto& queryContext    = *inputs.queryContext;
    const int   numQueries      = queryContext.numQueries;
    const auto& leafSubpatterns = inputs.recursivePreprocessor->leafSubpatterns();

    ctx.batchTargets.clear();
    ctx.batchOriginalIndices.clear();
    ctx.fallbackEntries.clear();
    ctx.batchTargets.reserve(static_cast<size_t>(range.size()));
    ctx.batchOriginalIndices.reserve(static_cast<size_t>(range.size()));

    for (int i = range.start; i < range.end; ++i) {
      const RDKit::ROMol* target        = targets[i];
      const unsigned int  atomCount     = target->getNumAtoms();
      const bool          needsFallback = (atomCount > kMaxTargetAtoms) || requiresRDKitFallback(target);
      if (needsFallback) {
        for (int q = 0; q < numQueries; ++q) {
          ctx.fallbackEntries.push_back({i, q});
        }
        continue;
      }
      ctx.batchTargets.push_back(target);
      ctx.batchOriginalIndices.push_back(i);
    }

    if (!ctx.fallbackEntries.empty() && inputs.fallbackQueue != nullptr) {
      inputs.fallbackQueue->enqueueBatch(std::move(ctx.fallbackEntries));
      ctx.fallbackEntries.clear();
    }

    if (ctx.batchTargets.empty()) {
      if (inputs.fallbackQueue != nullptr) {
        inputs.fallbackQueue->tryProcessOne();
      }
      return;
    }

    MoleculesHost    targetsHost;
    std::vector<int> emptySortOrder;
    buildTargetBatchParallelInto(targetsHost, 1, ctx.batchTargets, emptySortOrder);

    auto sharedTargetsHost     = std::make_shared<MoleculesHost>(std::move(targetsHost));
    auto sharedOriginalIndices = std::make_shared<std::vector<int>>(std::move(ctx.batchOriginalIndices));
    ctx.batchOriginalIndices.clear();
    ctx.batchOriginalIndices.reserve(static_cast<size_t>(range.size()));

    const int numBatchTargets  = static_cast<int>(sharedOriginalIndices->size());
    auto      sharedAtomCounts = std::make_shared<std::vector<int>>(static_cast<size_t>(numBatchTargets));

    int localMaxTargetAtoms  = 0;
    int localMaxBondsPerAtom = 0;
    for (int targetIdx = 0; targetIdx < numBatchTargets; ++targetIdx) {
      const int atomStart            = sharedTargetsHost->batchAtomStarts[targetIdx];
      const int atomEnd              = sharedTargetsHost->batchAtomStarts[targetIdx + 1];
      const int atoms                = atomEnd - atomStart;
      (*sharedAtomCounts)[targetIdx] = atoms;
      localMaxTargetAtoms            = std::max(localMaxTargetAtoms, atoms);
      for (int a = atomStart; a < atomEnd; ++a) {
        localMaxBondsPerAtom =
          std::max(localMaxBondsPerAtom, static_cast<int>(sharedTargetsHost->targetAtomBonds[a].degree));
      }
    }

    const int totalPairs       = numBatchTargets * numQueries;
    const int maxPairsPerBatch = inputs.maxPairsPerBatch;

    // Releases the buffer back to the pool if we throw between acquire and push.
    struct BufferReleaseGuard {
      PinnedHostBufferPool* pool   = nullptr;
      PinnedHostBuffer*     buffer = nullptr;
      ~BufferReleaseGuard() {
        if (pool && buffer) {
          pool->release(buffer);
        }
      }
      void release() { buffer = nullptr; }
    };

    for (int pairOffset = 0; pairOffset < totalPairs; pairOffset += maxPairsPerBatch) {
      PinnedHostBuffer* buffer = nullptr;
      {
        ScopedNvtxRange waitRange("Wait for pinned buffer", NvtxColor::kRed);
        buffer = inputs.bufferPool->acquire();
      }
      if (buffer == nullptr) {
        // Pool was shutdown via onAbort; bail out cleanly.
        return;
      }
      BufferReleaseGuard releaseGuard{inputs.bufferPool, buffer};

      auto batch                   = std::make_unique<PreparedMiniBatch>();
      batch->pinnedBuffer          = buffer;
      batch->pool                  = inputs.bufferPool;
      batch->targetsHost           = sharedTargetsHost;
      batch->targetOriginalIndices = sharedOriginalIndices;
      batch->targetAtomCounts      = sharedAtomCounts;

      batch->ctx.queryAtomCounts       = queryContext.queryAtomCounts.data();
      batch->ctx.queryPipelineDepths   = queryContext.queryPipelineDepths.data();
      batch->ctx.queryMaxDepths        = queryContext.queryMaxDepths.data();
      batch->ctx.queryHasPatterns      = queryContext.queryHasPatterns.data();
      batch->ctx.targetAtomCounts      = sharedAtomCounts.get();
      batch->ctx.targetOriginalIndices = sharedOriginalIndices.get();
      batch->ctx.numTargets            = numBatchTargets;
      batch->ctx.numQueries            = numQueries;
      batch->ctx.maxTargetAtoms        = localMaxTargetAtoms;
      batch->ctx.maxQueryAtoms         = queryContext.maxQueryAtoms;
      batch->ctx.maxBondsPerAtom       = localMaxBondsPerAtom;
      batch->ctx.maxMatches            = inputs.maxMatches;
      batch->ctx.countOnly             = inputs.countOnly;
      const int templateTargetAtoms    = std::max(localMaxTargetAtoms, queryContext.maxQueryAtoms);
      batch->ctx.templateConfig =
        selectTemplateConfig(templateTargetAtoms, queryContext.maxQueryAtoms, localMaxBondsPerAtom);

      planner.prepareMiniBatch(batch->plan, *buffer, batch->ctx, leafSubpatterns, pairOffset, maxPairsPerBatch);

      releaseGuard.release();
      pushBatch(std::move(batch));
    }

    if (inputs.fallbackQueue != nullptr && inputs.fallbackQueue->hasWork()) {
      inputs.fallbackQueue->tryProcessOne();
    }
  }

  static std::unique_ptr<PerGpuState>  makePerGpuState(Inputs& inputs, int gpuId);
  static std::unique_ptr<GpuSlotState> makeSlotState(Inputs& inputs, PerGpuState& pgs, int gpuId);

  static void dispatchAndCopyBack(GpuSlotState&        slot,
                                  PerGpuState&         pgs,
                                  PreparedBatch&       batch,
                                  Inputs&              inputs,
                                  RunnerThreadContext& ctx);

  static void postprocess(GpuSlotState& slot, PreparedBatch& batch, Inputs& inputs, RunnerThreadContext& ctx);

  static void onAbort(Inputs& inputs);
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_WORKLOAD_H
