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

#include <algorithm>
#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "src/substruct/molecules_device.cuh"
#include "src/substruct/recursive_preprocessor.h"
#include "src/substruct/substruct_kernels.h"
#include "src/substruct/substruct_search_internal.h"
#include "src/substruct/substruct_workload.h"
#include "src/substruct/thread_worker_context.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/nvtx.h"

namespace nvMolKit {

namespace {

// Mirrors the original substruct_search.cu helpers for launching the per-mini-batch
// kernel sequence. Lives in the workload TU so the only call site is dispatchAndCopyBack.

void launchLabelAndMatchInternal(int                        numPairsInGroup,
                                 GpuExecutor&               executor,
                                 const ThreadWorkerContext& ctx,
                                 MoleculesDevice&           targetsDevice,
                                 const MoleculesDevice&     queriesDevice,
                                 SubstructAlgorithm         algorithm,
                                 cudaStream_t               stream,
                                 int                        depthGroupIdx) {
  if (numPairsInGroup == 0) {
    return;
  }

  const int* globalPairIndicesDev     = executor.consolidatedBuffer.matchGlobalPairIndices(depthGroupIdx);
  const int* miniBatchLocalIndicesDev = executor.consolidatedBuffer.matchBatchLocalIndices(depthGroupIdx);

  launchLabelMatrixKernel(ctx.templateConfig,
                          targetsDevice.view<MoleculeType::Target>(),
                          queriesDevice.view<MoleculeType::Query>(),
                          globalPairIndicesDev,
                          numPairsInGroup,
                          ctx.numQueries,
                          executor.deviceResults.labelMatrixBuffer(),
                          executor.deviceResults.recursiveMatchBits(),
                          executor.deviceResults.maxTargetAtoms(),
                          miniBatchLocalIndicesDev,
                          stream);

  launchSubstructMatchKernel(ctx.templateConfig,
                             algorithm,
                             targetsDevice.view<MoleculeType::Target>(),
                             queriesDevice.view<MoleculeType::Query>(),
                             executor.deviceResults,
                             globalPairIndicesDev,
                             numPairsInGroup,
                             ctx.numQueries,
                             miniBatchLocalIndicesDev,
                             nullptr,
                             stream);
}

void uploadAndLaunchMiniBatchInternal(GpuExecutor&                        executor,
                                      const ThreadWorkerContext&          ctx,
                                      MoleculesDevice&                    targetsDevice,
                                      const MoleculesDevice&              queriesDevice,
                                      const RecursivePatternPreprocessor& recursivePreprocessor,
                                      SubstructAlgorithm                  algorithm) {
  cudaStream_t executorStream     = executor.stream();
  const int    numBuffersPerBlock = (algorithm == SubstructAlgorithm::GSI) ? 2 : 1;

  if (executor.plan.maxPipelineDepthInMiniBatch == 0) {
    const int maxMatchesToFind = ctx.maxMatches > 0 ? ctx.maxMatches : -1;
    executor.deviceResults.allocateMiniBatch(executor.plan.numPairsInMiniBatch,
                                             executor.consolidatedBuffer.miniBatchPairMatchStarts(),
                                             executor.plan.totalMatchIndices,
                                             ctx.numQueries,
                                             ctx.maxTargetAtoms,
                                             numBuffersPerBlock,
                                             maxMatchesToFind,
                                             ctx.countOnly);
    executor.deviceResults.setQueryAtomCounts(ctx.queryAtomCounts, ctx.numQueries);

    launchLabelMatrixKernel(ctx.templateConfig,
                            targetsDevice.view<MoleculeType::Target>(),
                            queriesDevice.view<MoleculeType::Query>(),
                            executor.consolidatedBuffer.pairIndices(),
                            executor.plan.numPairsInMiniBatch,
                            ctx.numQueries,
                            executor.deviceResults.labelMatrixBuffer(),
                            nullptr,
                            executor.deviceResults.maxTargetAtoms(),
                            nullptr,
                            executorStream);

    launchSubstructMatchKernel(ctx.templateConfig,
                               algorithm,
                               targetsDevice.view<MoleculeType::Target>(),
                               queriesDevice.view<MoleculeType::Query>(),
                               executor.deviceResults,
                               executor.consolidatedBuffer.pairIndices(),
                               executor.plan.numPairsInMiniBatch,
                               ctx.numQueries,
                               nullptr,
                               nullptr,
                               executorStream);
    return;
  }

  cudaStream_t recursiveStream = executor.recursiveStream.stream();

  const int maxMatchesToFind = ctx.maxMatches > 0 ? ctx.maxMatches : -1;
  executor.deviceResults.allocateMiniBatch(executor.plan.numPairsInMiniBatch,
                                           executor.consolidatedBuffer.miniBatchPairMatchStarts(),
                                           executor.plan.totalMatchIndices,
                                           ctx.numQueries,
                                           ctx.maxTargetAtoms,
                                           numBuffersPerBlock,
                                           maxMatchesToFind,
                                           ctx.countOnly);
  executor.deviceResults.setQueryAtomCounts(ctx.queryAtomCounts, ctx.numQueries);

  cudaCheckError(cudaEventRecord(executor.allocDoneEvent.event(), executorStream));

  cudaCheckError(cudaStreamWaitEvent(recursiveStream, executor.allocDoneEvent.event(), 0));

  std::array<cudaEvent_t, kMaxSmartsNestingDepth> depthEventPtrs;
  for (int i = 0; i < kMaxSmartsNestingDepth; ++i) {
    depthEventPtrs[i] = executor.depthEvents[i].event();
  }

  recursivePreprocessor.preprocessMiniBatch(ctx.templateConfig,
                                            targetsDevice,
                                            executor.deviceResults,
                                            ctx.numQueries,
                                            executor.plan.miniBatchPairOffset,
                                            executor.plan.numPairsInMiniBatch,
                                            algorithm,
                                            recursiveStream,
                                            executor.recursiveScratch,
                                            *executor.plan.patternsAtDepth,
                                            executor.plan.recursiveMaxDepth,
                                            executor.plan.firstTargetInMiniBatch,
                                            executor.plan.numTargetsInMiniBatch,
                                            depthEventPtrs.data(),
                                            kMaxSmartsNestingDepth);

  launchLabelAndMatchInternal(executor.plan.matchPairsCounts[0],
                              executor,
                              ctx,
                              targetsDevice,
                              queriesDevice,
                              algorithm,
                              executorStream,
                              0);

  cudaStream_t postStream = executor.postRecursionStream.stream();
  cudaCheckError(cudaStreamWaitEvent(postStream, executor.allocDoneEvent.event(), 0));

  for (int depth = 1; depth <= executor.plan.maxPipelineDepthInMiniBatch; ++depth) {
    cudaCheckError(cudaStreamWaitEvent(postStream, depthEventPtrs[depth - 1], 0));
    launchLabelAndMatchInternal(executor.plan.matchPairsCounts[depth],
                                executor,
                                ctx,
                                targetsDevice,
                                queriesDevice,
                                algorithm,
                                postStream,
                                depth);
  }
  cudaCheckError(cudaEventRecord(executor.postRecursionDoneEvent.event(), postStream));

  cudaCheckError(cudaEventRecord(executor.recursiveDoneEvent.event(), recursiveStream));
  cudaCheckError(cudaStreamWaitEvent(executorStream, executor.recursiveDoneEvent.event(), 0));
  cudaCheckError(cudaStreamWaitEvent(executorStream, executor.postRecursionDoneEvent.event(), 0));
}

}  // namespace

int SubstructWorkload::totalUnits() const {
  return static_cast<int>(inputs_.targets->size());
}

int SubstructWorkload::unitsPerPreprocBatch() const {
  return inputs_.targetsPerBatch;
}

std::unique_ptr<gpu_scheduler::PerGpuState> SubstructWorkload::makePerGpuState(int gpuId) {
  auto state = std::make_unique<SubstructPerGpu>();
  if (gpuId == inputs_.primaryDeviceId) {
    state->queriesDevice         = inputs_.queriesDevice;
    state->recursivePreprocessor = inputs_.recursivePreprocessor;
  } else {
    state->localQueries = std::make_unique<MoleculesDevice>();
    state->localQueries->copyFromHost(*inputs_.queriesHost);
    state->localPreprocessor = std::make_unique<RecursivePatternPreprocessor>();
    state->localPreprocessor->buildPatterns(*inputs_.queriesHost);
    state->localPreprocessor->syncToDevice(nullptr);
    state->queriesDevice         = state->localQueries.get();
    state->recursivePreprocessor = state->localPreprocessor.get();
  }
  return state;
}

std::unique_ptr<gpu_scheduler::GpuSlotState> SubstructWorkload::makeSlotState(gpu_scheduler::PerGpuState& /*pgs*/,
                                                                              int gpuId) {
  // Slot index isn't meaningful with the new pipeline (executors are pooled
  // per-coordinator); pass 0 for telemetry.
  auto executor = std::make_unique<GpuExecutor>(0, gpuId);
  executor->initializeForStream();
  return executor;
}

std::unique_ptr<gpu_scheduler::PreprocThreadContext> SubstructWorkload::makePreprocCtx() {
  return std::make_unique<PreprocThreadContext>();
}

std::unique_ptr<gpu_scheduler::RunnerThreadContext> SubstructWorkload::makeRunnerCtx() {
  auto ctx = std::make_unique<RunnerThreadContext>();
  if (inputs_.fallbackQueue != nullptr) {
    ctx->fallbackGuard = gpu_scheduler::FallbackProducerGuard<RDKitFallbackEntry>(inputs_.fallbackQueue);
  }
  return ctx;
}

void SubstructWorkload::preprocess(gpu_scheduler::IndexRange            range,
                                   gpu_scheduler::PreprocThreadContext& ctxBase,
                                   const gpu_scheduler::PushBatch&      pushBatch) {
  ScopedNvtxRange  claimRange("SubstructWorkload::preprocess");
  MiniBatchPlanner planner;
  auto&            inputs = inputs_;
  auto&            ctx    = static_cast<PreprocThreadContext&>(ctxBase);

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

void SubstructWorkload::dispatchAndCopyBack(gpu_scheduler::GpuSlotState&  slotBase,
                                            gpu_scheduler::PerGpuState&   pgsBase,
                                            gpu_scheduler::PreparedBatch& batchBase,
                                            gpu_scheduler::RunnerThreadContext& /*ctxBase*/) {
  ScopedNvtxRange dispatchRange("SubstructWorkload::dispatchAndCopyBack");
  auto&           slot   = static_cast<GpuSlotState&>(slotBase);
  auto&           pgs    = static_cast<PerGpuState&>(pgsBase);
  auto&           batch  = static_cast<PreparedBatch&>(batchBase);
  auto&           inputs = inputs_;

  slot.applyMiniBatchPlan(std::move(batch.plan));
  slot.recursiveScratch.setPinnedBuffer(batch.pinnedBuffer->patternsAtDepthHost,
                                        static_cast<int>(batch.pinnedBuffer->patternsAtDepthHost[0].size()));

  slot.consolidatedBuffer.allocate(batch.pinnedBuffer->consolidated.maxBatchSize, slot.stream());
  slot.consolidatedBuffer.copyFromHost(*batch.pinnedBuffer, slot.stream());

  slot.targetsDevice.copyFromHost(*batch.targetsHost, slot.stream());
  cudaCheckError(cudaEventRecord(slot.targetsReadyEvent.event(), slot.stream()));
  cudaCheckError(cudaStreamWaitEvent(slot.recursiveStream.stream(), slot.targetsReadyEvent.event(), 0));
  cudaCheckError(cudaStreamWaitEvent(slot.postRecursionStream.stream(), slot.targetsReadyEvent.event(), 0));

  uploadAndLaunchMiniBatchInternal(slot,
                                   batch.ctx,
                                   slot.targetsDevice,
                                   *pgs.queriesDevice,
                                   *pgs.recursivePreprocessor,
                                   inputs.algorithm);

  if (inputs.countOnly) {
    initiateCountsOnlyCopyToHost(slot, *batch.pinnedBuffer);
  } else {
    initiateResultsCopyToHost(slot, *batch.pinnedBuffer);
  }
}

void SubstructWorkload::postprocess(gpu_scheduler::GpuSlotState&  slotBase,
                                    gpu_scheduler::PreparedBatch& batchBase,
                                    gpu_scheduler::RunnerThreadContext& /*ctxBase*/) {
  ScopedNvtxRange accumRange("SubstructWorkload::postprocess");
  auto&           slot   = static_cast<GpuSlotState&>(slotBase);
  auto&           batch  = static_cast<PreparedBatch&>(batchBase);
  auto&           inputs = inputs_;

  if (inputs.boolResults != nullptr) {
    accumulateMiniBatchResultsBoolean(slot, batch.ctx, *inputs.boolResults, *inputs.resultsMutex, *batch.pinnedBuffer);
  } else if (inputs.countResults != nullptr) {
    accumulateMiniBatchResultsCounts(slot, batch.ctx, *inputs.countResults, *inputs.resultsMutex, *batch.pinnedBuffer);
  } else {
    // The accumulator routes overflow / oversize results to the fallback queue
    // when the GPU couldn't hold the full match list.
    accumulateMiniBatchResults(slot,
                               batch.ctx,
                               *inputs.results,
                               *inputs.resultsMutex,
                               *batch.pinnedBuffer,
                               inputs.fallbackQueue);
  }

  // Hand the pinned buffer back to the pool eagerly so other preprocessor
  // threads waiting on acquire() can make progress before the rest of the
  // postprocess work (fallback drain) runs. Clearing pinnedBuffer opts out of
  // ~PreparedMiniBatch's auto-release.
  inputs.bufferPool->release(batch.pinnedBuffer);
  batch.pinnedBuffer = nullptr;
  batch.pool         = nullptr;

  if (inputs.fallbackQueue != nullptr) {
    inputs.fallbackQueue->tryProcessOne();
  }
}

void SubstructWorkload::onAbort() {
  if (inputs_.bufferPool != nullptr) {
    inputs_.bufferPool->shutdown();
  }
}

}  // namespace nvMolKit
