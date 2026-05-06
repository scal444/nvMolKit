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

#include "substruct_workload.h"

#include <array>

#include "cuda_error_check.h"
#include "molecules_device.cuh"
#include "nvtx.h"
#include "recursive_preprocessor.h"
#include "substruct_kernels.h"
#include "substruct_search_internal.h"
#include "thread_worker_context.h"

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

int SubstructWorkload::totalUnits(Inputs& inputs) {
  return static_cast<int>(inputs.targets->size());
}

int SubstructWorkload::unitsPerPreprocBatch(Inputs& inputs) {
  return inputs.targetsPerBatch;
}

std::unique_ptr<SubstructWorkload::PerGpuState> SubstructWorkload::makePerGpuState(Inputs& inputs, int gpuId) {
  auto state = std::make_unique<SubstructPerGpu>();
  if (gpuId == inputs.primaryDeviceId) {
    state->queriesDevice         = inputs.queriesDevice;
    state->recursivePreprocessor = inputs.recursivePreprocessor;
  } else {
    state->localQueries = std::make_unique<MoleculesDevice>();
    state->localQueries->copyFromHost(*inputs.queriesHost);
    state->localPreprocessor = std::make_unique<RecursivePatternPreprocessor>();
    state->localPreprocessor->buildPatterns(*inputs.queriesHost);
    state->localPreprocessor->syncToDevice(nullptr);
    state->queriesDevice         = state->localQueries.get();
    state->recursivePreprocessor = state->localPreprocessor.get();
  }
  return state;
}

std::unique_ptr<SubstructWorkload::GpuSlotState> SubstructWorkload::makeSlotState(Inputs& /*inputs*/,
                                                                                  PerGpuState& /*pgs*/,
                                                                                  int gpuId) {
  // Slot index isn't meaningful with the new pipeline (executors are pooled
  // per-coordinator); pass 0 for telemetry.
  auto executor = std::make_unique<GpuExecutor>(0, gpuId);
  executor->initializeForStream();
  return executor;
}

SubstructWorkload::RunnerThreadContext SubstructWorkload::makeRunnerCtx(Inputs& inputs) {
  RunnerThreadContext ctx;
  if (inputs.fallbackQueue != nullptr) {
    ctx.fallbackGuard = gpu_scheduler::FallbackProducerGuard<RDKitFallbackEntry>(inputs.fallbackQueue);
  }
  return ctx;
}

void SubstructWorkload::dispatchAndCopyBack(GpuSlotState&        slot,
                                            PerGpuState&         pgs,
                                            PreparedBatch&       batch,
                                            Inputs&              inputs,
                                            RunnerThreadContext& /*ctx*/) {
  ScopedNvtxRange dispatchRange("SubstructWorkload::dispatchAndCopyBack");

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

void SubstructWorkload::postprocess(GpuSlotState&        slot,
                                    PreparedBatch&       batch,
                                    Inputs&              inputs,
                                    RunnerThreadContext& /*ctx*/) {
  ScopedNvtxRange accumRange("SubstructWorkload::postprocess");

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

void SubstructWorkload::onAbort(Inputs& inputs) {
  if (inputs.bufferPool != nullptr) {
    inputs.bufferPool->shutdown();
  }
}

}  // namespace nvMolKit
