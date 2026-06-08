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

#include <GraphMol/ROMol.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "src/gpu_scheduler/config.h"
#include "src/gpu_scheduler/cpu_fallback_queue.h"
#include "src/gpu_scheduler/pipeline.h"
#include "src/substruct/molecules_device.cuh"
#include "src/substruct/pinned_buffer_pool.h"
#include "src/substruct/recursive_preprocessor.h"
#include "src/substruct/substruct_search.h"
#include "src/substruct/substruct_search_internal.h"
#include "src/substruct/substruct_workload.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/nvtx.h"

namespace nvMolKit {

namespace {

/**
 * @brief Per-entry handler bound into the workload's CpuFallbackQueue.
 *
 * Carries the target/query lists and result sinks; called by whichever thread
 * happens to drain a fallback entry (preprocessor or runner).
 */
struct FallbackHandlerState {
  const std::vector<const RDKit::ROMol*>* targets      = nullptr;
  const std::vector<const RDKit::ROMol*>* queries      = nullptr;
  SubstructSearchResults*                 results      = nullptr;
  HasSubstructMatchResults*               boolResults  = nullptr;
  std::vector<int>*                       countResults = nullptr;
  std::mutex*                             resultsMutex = nullptr;
  int                                     maxMatches   = 0;
};

void processFallbackEntry(const FallbackHandlerState& state, const RDKitFallbackEntry& entry) {
  ScopedNvtxRange     range("RDKit fallback T" + std::to_string(entry.originalTargetIdx) + "/Q" +
                        std::to_string(entry.originalQueryIdx));
  const RDKit::ROMol* target              = (*state.targets)[entry.originalTargetIdx];
  const RDKit::ROMol* query               = (*state.queries)[entry.originalQueryIdx];
  const int           effectiveMaxMatches = state.boolResults ? 1 : state.maxMatches;
  processWithRDKitFallback(target,
                           query,
                           entry.originalTargetIdx,
                           entry.originalQueryIdx,
                           *state.results,
                           *state.resultsMutex,
                           effectiveMaxMatches,
                           state.boolResults,
                           state.countResults);
}

/**
 * @brief Postprocess: collapse equivalent matches that differ only in atom enumeration.
 */
void uniquifyResults(SubstructSearchResults& results) {
  ScopedNvtxRange uniquifyRange("uniquifyResults");

  std::set<std::vector<int>>    seenSorted;
  std::vector<std::vector<int>> uniqueMatches;
  std::vector<int>              sortedMatch;

  for (auto& [pairIdx, matchList] : results.matches) {
    if (matchList.size() <= 1) {
      continue;
    }

    seenSorted.clear();
    uniqueMatches.clear();
    uniqueMatches.reserve(matchList.size());

    for (auto& match : matchList) {
      sortedMatch.assign(match.begin(), match.end());
      std::sort(sortedMatch.begin(), sortedMatch.end());

      if (seenSorted.insert(sortedMatch).second) {
        uniqueMatches.push_back(std::move(match));
      }
    }

    matchList = std::move(uniqueMatches);
  }
}

}  // namespace

/**
 * @brief Compute effective thread counts using autoselect logic.
 *
 * Kept as a free function in the namespace because tests reference it directly.
 * Internally now defers to gpu_scheduler::resolve(); behavior matches the
 * original computeEffectiveThreadCounts to byte.
 */
void computeEffectiveThreadCounts(const SubstructSearchConfig& config,
                                  int                          numGpus,
                                  int&                         effectivePreprocessingThreads,
                                  int&                         effectiveWorkerThreads) {
  const int hwThreads        = static_cast<int>(std::thread::hardware_concurrency());
  const int effectiveNumGpus = std::max(1, numGpus);

  effectivePreprocessingThreads =
    (config.preprocessingThreads == -1) ? hwThreads : std::max(1, config.preprocessingThreads);

  effectiveWorkerThreads = (config.workerThreads == -1) ? std::min(4, std::max(1, hwThreads / effectiveNumGpus)) :
                                                          std::max(1, config.workerThreads);
}

namespace {

void getSubstructMatchesImpl(const std::vector<const RDKit::ROMol*>& targets,
                             const std::vector<const RDKit::ROMol*>& queries,
                             SubstructSearchResults&                 results,
                             SubstructAlgorithm                      algorithm,
                             cudaStream_t                            stream,
                             const SubstructSearchConfig&            config,
                             HasSubstructMatchResults*               boolResults,
                             std::vector<int>*                       countResults) {
  const int numTargets = static_cast<int>(targets.size());
  const int numQueries = static_cast<int>(queries.size());

  if (numTargets == 0 || numQueries == 0) {
    results.resize(numTargets, numQueries);
    return;
  }

  std::vector<int> gpuIds        = config.gpuIds;
  int              currentDevice = 0;
  cudaCheckError(cudaGetDevice(&currentDevice));
  if (gpuIds.empty()) {
    gpuIds.push_back(currentDevice);
  }
  const int numGpus = static_cast<int>(gpuIds.size());

  int effectivePreprocessingThreads = 0;
  int effectiveWorkerThreads        = 0;
  computeEffectiveThreadCounts(config, numGpus, effectivePreprocessingThreads, effectiveWorkerThreads);

  ScopedNvtxRange overloadRange(
    "getSubstructMatches T=" + std::to_string(numTargets) + " Q=" + std::to_string(numQueries) +
    " batch=" + std::to_string(config.batchSize) + " prep=" + std::to_string(effectivePreprocessingThreads) +
    " workers=" + std::to_string(effectiveWorkerThreads) + " gpus=" + std::to_string(numGpus));

  results.resize(numTargets, numQueries);

  ScopedNvtxRange  buildRange2("Build host query data structures");
  std::vector<int> emptySortOrder;
  MoleculesHost    queriesHost = buildQueryBatchParallel(queries, emptySortOrder, effectivePreprocessingThreads);
  buildRange2.pop();

  ScopedNvtxRange buildRange3("Build device query data structures");
  MoleculesDevice queriesDevice(stream);
  buildRange3.pop();

  ScopedNvtxRange buildRange4("Copy queries to device");
  queriesDevice.copyFromHost(queriesHost);
  buildRange4.pop();

  ScopedNvtxRange              leafRange("Build LeafSubpatterns");
  RecursivePatternPreprocessor recursivePreprocessor;
  recursivePreprocessor.buildPatterns(queriesHost);
  recursivePreprocessor.syncToDevice(stream);
  leafRange.pop();

  // Workers use independent streams; the caller stream needs to fully publish
  // queries before any worker reads them.
  cudaCheckError(cudaStreamSynchronize(stream));

  const LeafSubpatterns& leafSubpatterns = recursivePreprocessor.leafSubpatterns();
  QueryPreprocessContext queryContext;
  queryContext.numQueries = numQueries;
  queryContext.queryAtomCounts.resize(numQueries);
  queryContext.queryPipelineDepths.resize(numQueries);
  queryContext.queryMaxDepths.resize(numQueries);
  queryContext.queryHasPatterns.resize(numQueries);
  queryContext.queryNeedsFallback.resize(numQueries, 0);

  const int precomputedSize      = static_cast<int>(leafSubpatterns.perQueryPatterns.size());
  const int perQueryMaxDepthSize = static_cast<int>(leafSubpatterns.perQueryMaxDepth.size());

  int maxQueryAtoms = 0;
#pragma omp parallel num_threads(effectivePreprocessingThreads) reduction(max : maxQueryAtoms)
  {
#pragma omp for nowait
    for (int q = 0; q < numQueries; ++q) {
      const int atomStart             = queriesHost.batchAtomStarts[q];
      const int atomEnd               = queriesHost.batchAtomStarts[q + 1];
      const int atomCount             = atomEnd - atomStart;
      queryContext.queryAtomCounts[q] = atomCount;

      const int maxDepth = (q < perQueryMaxDepthSize) ? leafSubpatterns.perQueryMaxDepth[q] : 0;

      if (maxDepth >= kMaxSmartsNestingDepth) {
        queryContext.queryNeedsFallback[q]  = 1;
        queryContext.queryPipelineDepths[q] = 0;
        queryContext.queryMaxDepths[q]      = 0;
        queryContext.queryHasPatterns[q]    = 0;
      } else {
        const int depth                     = getQueryPipelineDepth(queriesHost, q);
        queryContext.queryPipelineDepths[q] = depth;
        queryContext.queryMaxDepths[q]      = maxDepth;

        const bool hasPatterns =
          (q < precomputedSize) && (maxDepth > 0 || !leafSubpatterns.perQueryPatterns[q][0].empty());
        queryContext.queryHasPatterns[q] = hasPatterns ? 1 : 0;
      }

      maxQueryAtoms = std::max(maxQueryAtoms, atomCount);
    }
  }
  queryContext.maxQueryAtoms = maxQueryAtoms;

  std::mutex resultsMutex;

  // Build the workload-owned CPU fallback queue.
  FallbackHandlerState
    handlerState{&targets, &queries, &results, boolResults, countResults, &resultsMutex, config.maxMatches};
  gpu_scheduler::CpuFallbackQueue<RDKitFallbackEntry> fallbackQueue(
    [handlerState](const RDKitFallbackEntry& entry) { processFallbackEntry(handlerState, entry); });

  // Enqueue all (target, query) pairs for queries that exceed recursion depth limit.
  // No producer guard needed yet - the queue stays open as long as no producer
  // has registered-then-unregistered, which won't happen until the pipeline runs.
  {
    std::vector<RDKitFallbackEntry> depthFallbackEntries;
    for (int q = 0; q < numQueries; ++q) {
      if (queryContext.queryNeedsFallback[q]) {
        for (int t = 0; t < numTargets; ++t) {
          depthFallbackEntries.push_back({t, q});
        }
      }
    }
    if (!depthFallbackEntries.empty()) {
      fallbackQueue.enqueueBatch(std::move(depthFallbackEntries));
    }
  }

  // Compute per-mini-batch sizes (matches the original substructure logic).
  const int targetsPerBatch     = std::max(1, config.batchSize / numQueries);
  const int maxPairsPerBatch    = std::max(1, config.batchSize);
  int       maxPatternsPerDepth = 256;
  for (int d = 0; d <= kMaxSmartsNestingDepth; ++d) {
    int patternsAtThisDepth = 0;
    for (size_t q = 0; q < leafSubpatterns.perQueryPatterns.size(); ++q) {
      patternsAtThisDepth += static_cast<int>(leafSubpatterns.perQueryPatterns[q][d].size());
    }
    maxPatternsPerDepth = std::max(maxPatternsPerDepth, patternsAtThisDepth);
  }

  const bool countOnly = (boolResults != nullptr) || (countResults != nullptr);
  size_t     maxMatchIndicesPerMiniBatch;
  if (countOnly) {
    maxMatchIndicesPerMiniBatch = 0;
  } else if (config.maxMatches > 0) {
    maxMatchIndicesPerMiniBatch =
      static_cast<size_t>(maxPairsPerBatch) * config.maxMatches * queryContext.maxQueryAtoms;
  } else {
    maxMatchIndicesPerMiniBatch = static_cast<size_t>(maxPairsPerBatch) * kMaxTargetAtoms * queryContext.maxQueryAtoms;
  }

  // Determine the orchestration shape and apply the same RAM cap as before.
  int slotsPerWorker;
  if (config.executorsPerRunner == -1) {
    const int totalRunners = effectiveWorkerThreads * numGpus;
    slotsPerWorker         = (totalRunners == 1) ? 3 : 2;
  } else if (config.executorsPerRunner < 1 || config.executorsPerRunner > 8) {
    throw std::invalid_argument("executorsPerRunner must be -1 (auto) or between 1 and 8");
  } else {
    slotsPerWorker = config.executorsPerRunner;
  }

  const int    poolSize = std::max(1, effectivePreprocessingThreads) * 2;
  const size_t perBufferSize =
    computePinnedHostBufferBytes(maxPairsPerBatch, static_cast<int>(maxMatchIndicesPerMiniBatch), maxPatternsPerDepth);
  const size_t totalPinnedBytes = static_cast<size_t>(poolSize) * perBufferSize;
  const long   pages            = sysconf(_SC_PHYS_PAGES);
  const long   pageSize         = sysconf(_SC_PAGE_SIZE);
  const size_t systemRam        = static_cast<size_t>(pages) * static_cast<size_t>(pageSize);
  const size_t maxAllowed       = systemRam / 4;
  if (totalPinnedBytes > maxAllowed) {
    throw std::runtime_error("Substructure search would require " + std::to_string(totalPinnedBytes / (1024 * 1024)) +
                             " MB of pinned memory, exceeding 1/4 of system RAM (" +
                             std::to_string(maxAllowed / (1024 * 1024)) +
                             " MB). Reduce workerThreads, executorsPerRunner, or batchSize.");
  }

  PinnedHostBufferPool bufferPool;
  bufferPool.initialize(poolSize, maxPairsPerBatch, static_cast<int>(maxMatchIndicesPerMiniBatch), maxPatternsPerDepth);

  // Wire everything into the workload Inputs and run the pipeline.
  SubstructInputs inputs;
  inputs.targets               = &targets;
  inputs.queriesRdkit          = &queries;
  inputs.queriesHost           = &queriesHost;
  inputs.queriesDevice         = &queriesDevice;
  inputs.recursivePreprocessor = &recursivePreprocessor;
  inputs.queryContext          = &queryContext;
  inputs.algorithm             = algorithm;
  inputs.primaryDeviceId       = currentDevice;
  inputs.batchSize             = config.batchSize;
  inputs.maxMatches            = config.maxMatches;
  inputs.countOnly             = countOnly;
  inputs.results               = &results;
  inputs.boolResults           = boolResults;
  inputs.countResults          = countResults;
  inputs.resultsMutex          = &resultsMutex;
  inputs.bufferPool            = &bufferPool;
  inputs.fallbackQueue         = &fallbackQueue;
  inputs.targetsPerBatch       = targetsPerBatch;
  inputs.maxPairsPerBatch      = maxPairsPerBatch;
  inputs.maxPatternsPerDepth   = maxPatternsPerDepth;

  gpu_scheduler::Config schedulerConfig;
  schedulerConfig.workerThreadsPerGpu        = effectiveWorkerThreads;
  schedulerConfig.globalPreprocessingThreads = effectivePreprocessingThreads;
  schedulerConfig.slotsPerWorker             = slotsPerWorker;
  schedulerConfig.gpuIds                     = gpuIds;

  SubstructWorkload       workload(inputs);
  gpu_scheduler::Pipeline pipeline(schedulerConfig, workload);
  pipeline.run();

  // Drain any residual fallback entries (e.g. depth-limit overflow that wasn't
  // claimed by an opportunistic drain inside the pipeline).
  while (fallbackQueue.tryProcessOne()) {
  }

  if (!boolResults && config.uniquify) {
    uniquifyResults(results);
  }

  cudaCheckError(cudaGetLastError());
}

}  // anonymous namespace

void getSubstructMatches(const std::vector<const RDKit::ROMol*>& targets,
                         const std::vector<const RDKit::ROMol*>& queries,
                         SubstructSearchResults&                 results,
                         SubstructAlgorithm                      algorithm,
                         cudaStream_t                            stream,
                         const SubstructSearchConfig&            config) {
  getSubstructMatchesImpl(targets, queries, results, algorithm, stream, config, nullptr, nullptr);
}

void countSubstructMatches(const std::vector<const RDKit::ROMol*>& targets,
                           const std::vector<const RDKit::ROMol*>& queries,
                           std::vector<int>&                       counts,
                           SubstructAlgorithm                      algorithm,
                           cudaStream_t                            stream,
                           const SubstructSearchConfig&            config) {
  const int numTargets = static_cast<int>(targets.size());
  const int numQueries = static_cast<int>(queries.size());

  counts.assign(static_cast<size_t>(numTargets) * numQueries, 0);

  SubstructSearchResults matchResults;
  SubstructSearchConfig  countConfig = config;
  countConfig.maxMatches             = 0;

  getSubstructMatchesImpl(targets, queries, matchResults, algorithm, stream, countConfig, nullptr, &counts);
}

void hasSubstructMatch(const std::vector<const RDKit::ROMol*>& targets,
                       const std::vector<const RDKit::ROMol*>& queries,
                       HasSubstructMatchResults&               results,
                       SubstructAlgorithm                      algorithm,
                       cudaStream_t                            stream,
                       const SubstructSearchConfig&            config) {
  const int numTargets = static_cast<int>(targets.size());
  const int numQueries = static_cast<int>(queries.size());

  ScopedNvtxRange overloadRange("hasSubstructMatch T=" + std::to_string(numTargets) +
                                " Q=" + std::to_string(numQueries));

  results.resize(numTargets, numQueries);
  if (numTargets == 0 || numQueries == 0) {
    return;
  }

  SubstructSearchResults matchResults;
  SubstructSearchConfig  hasMatchConfig = config;
  hasMatchConfig.maxMatches             = 1;
  getSubstructMatchesImpl(targets, queries, matchResults, algorithm, stream, hasMatchConfig, &results, nullptr);

  for (auto& [pairIdx, matches] : matchResults.matches) {
    if (!matches.empty()) {
      results.hasMatch[static_cast<size_t>(pairIdx)] = 1;
    }
  }
}

}  // namespace nvMolKit
