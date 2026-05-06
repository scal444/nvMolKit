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

#ifndef NVMOLKIT_SUBSTRUCT_WORKLOAD_INL_H
#define NVMOLKIT_SUBSTRUCT_WORKLOAD_INL_H

#include <GraphMol/ROMol.h>

#include <algorithm>
#include <memory>
#include <utility>

#include "minibatch_planner.h"
#include "molecules.h"
#include "nvtx.h"
#include "substruct_kernels.h"
#include "substruct_launch_config.h"
#include "thread_worker_context.h"

namespace nvMolKit {

template <class PushFn>
void SubstructWorkload::preprocess(Inputs&                   inputs,
                                   gpu_scheduler::IndexRange range,
                                   PreprocThreadContext&     ctx,
                                   PushFn                    pushBatch) {
  ScopedNvtxRange claimRange("SubstructWorkload::preprocess");
  MiniBatchPlanner planner;

  const auto& targets        = *inputs.targets;
  const auto& queryContext   = *inputs.queryContext;
  const int   numQueries     = queryContext.numQueries;
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
    const int atomStart       = sharedTargetsHost->batchAtomStarts[targetIdx];
    const int atomEnd         = sharedTargetsHost->batchAtomStarts[targetIdx + 1];
    const int atoms           = atomEnd - atomStart;
    (*sharedAtomCounts)[targetIdx] = atoms;
    localMaxTargetAtoms       = std::max(localMaxTargetAtoms, atoms);
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

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_WORKLOAD_INL_H
