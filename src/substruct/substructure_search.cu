// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "substructure_search.h"
#include "substructure_search_internal.cuh"
#include "substruct_kernels.h"

#include <GraphMol/ROMol.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include <omp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <vector>

#include "cuda_error_check.h"
#include "host_vector.h"
#include "global_pool.cuh"
#include "graph_labeler.cuh"
#include "molecules_device.cuh"
#include "pinned_buffer_pool.h"
#include "sm_shared_mem_config.cuh"
#include "substruct_algos.cuh"
#include "substruct_debug.h"
#include "nvtx.h"

namespace nvMolKit {

namespace {

void runMacroBatchedSubstructSearch(const std::vector<const RDKit::ROMol*>& gpuTargets,
                                    const std::vector<int>&                gpuTargetIndices,
                                    const std::vector<unsigned int>&       gpuTargetAtomCounts,
                                    const MoleculesHost&                   queriesHost,
                                    const MoleculesDevice&                 queriesDevice,
                                    const LeafSubpatterns&                 leafSubpatterns,
                                    SubstructSearchResults&                results,
                                    SubstructAlgorithm                     algorithm,
                                    cudaStream_t                           stream,
                                    const SubstructSearchConfig&           config,
                                    const std::vector<int>&                querySortOrder,
                                    int                                    effectivePreprocessingThreads,
                                    RDKitFallbackQueue*                    fallbackQueue,
                                    HasSubstructMatchResults*              boolResults = nullptr,
                                    std::vector<int>*                      countResults = nullptr);

}  // anonymous namespace

// =============================================================================
// RDKit Fallback Implementation
// =============================================================================

void processWithRDKitFallback(const RDKit::ROMol*       target,
                              const RDKit::ROMol*       query,
                              int                       targetIdx,
                              int                       queryIdx,
                              SubstructSearchResults&   results,
                              std::mutex&               resultsMutex,
                              int                       maxMatches,
                              HasSubstructMatchResults* boolResults,
                              std::vector<int>*         countResults) {
  RDKit::SubstructMatchParameters params;
  params.uniquify = false;
  params.maxMatches = (maxMatches > 0) ? static_cast<unsigned int>(maxMatches) : 0;
  params.useChirality = false;
  params.useQueryQueryMatches = false;

  std::vector<RDKit::MatchVectType> rdkitMatches = RDKit::SubstructMatch(*target, *query, params);

  const int matchCount = static_cast<int>(rdkitMatches.size());
  if (matchCount == 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(resultsMutex);

  if (boolResults) {
    boolResults->setMatch(targetIdx, queryIdx, true);
  } else if (countResults) {
    const int pairIdx = targetIdx * results.numQueries + queryIdx;
    (*countResults)[pairIdx] = matchCount;
  } else {
    std::vector<std::vector<int>> convertedMatches;
    convertedMatches.reserve(rdkitMatches.size());
    for (const auto& match : rdkitMatches) {
      std::vector<int> mapping(match.size());
      for (size_t i = 0; i < match.size(); ++i) {
        mapping[i] = match[i].second;
      }
      convertedMatches.push_back(std::move(mapping));
    }

    auto& targetMatches = results.getMatchesMut(targetIdx, queryIdx);
    targetMatches.insert(targetMatches.end(),
                         std::make_move_iterator(convertedMatches.begin()),
                         std::make_move_iterator(convertedMatches.end()));
  }
}

RDKitFallbackQueue::RDKitFallbackQueue(const std::vector<const RDKit::ROMol*>* targets,
                                       const std::vector<const RDKit::ROMol*>* queries,
                                       SubstructSearchResults*                 results,
                                       std::mutex*                             resultsMutex,
                                       int                                     maxMatches,
                                       HasSubstructMatchResults*               boolResults,
                                       std::vector<int>*                       countResults)
    : targets_(targets),
      queries_(queries),
      results_(results),
      boolResults_(boolResults),
      countResults_(countResults),
      resultsMutex_(resultsMutex),
      maxMatches_(maxMatches),
      shutdown_(false),
      activeProducers_(0) {}

void RDKitFallbackQueue::enqueue(const std::vector<RDKitFallbackEntry>& entries) {
  if (entries.empty()) return;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& entry : entries) {
      queue_.push(entry);
    }
    queueSize_.store(queue_.size(), std::memory_order_release);
  }
  cv_.notify_all();
}

void RDKitFallbackQueue::enqueue(const RDKitFallbackEntry& entry) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(entry);
    queueSize_.fetch_add(1, std::memory_order_release);
  }
  cv_.notify_one();
}

void RDKitFallbackQueue::registerProducer() {
  std::lock_guard<std::mutex> lock(mutex_);
  ++activeProducers_;
}

void RDKitFallbackQueue::unregisterProducer() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    --activeProducers_;
  }
  cv_.notify_all();
}

void RDKitFallbackQueue::shutdown() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = true;
  }
  cv_.notify_all();
}

void RDKitFallbackQueue::workerLoop() {
  while (true) {
    RDKitFallbackEntry entry;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] {
        return !queue_.empty() || shutdown_ || (activeProducers_ == 0 && queue_.empty());
      });

      if (queue_.empty()) {
        if (shutdown_ || activeProducers_ == 0) {
          return;
        }
        continue;
      }

      entry = queue_.front();
      queue_.pop();
      queueSize_.fetch_sub(1, std::memory_order_release);
    }

    processEntry(entry);
  }
}

size_t RDKitFallbackQueue::processedCount() const {
  return processedCount_.load(std::memory_order_relaxed);
}

std::vector<RDKitFallbackEntry> RDKitFallbackQueue::drainToVector() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<RDKitFallbackEntry> result;
  result.reserve(queue_.size());
  while (!queue_.empty()) {
    result.push_back(queue_.front());
    queue_.pop();
  }
  queueSize_.store(0, std::memory_order_release);
  return result;
}

std::mutex& RDKitFallbackQueue::getResultsMutex() { return *resultsMutex_; }

bool RDKitFallbackQueue::tryProcessOne() {
  RDKitFallbackEntry entry;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    entry = queue_.front();
    queue_.pop();
    queueSize_.fetch_sub(1, std::memory_order_release);
  }
  processEntry(entry);
  return true;
}

bool RDKitFallbackQueue::hasWork() const {
  return queueSize_.load(std::memory_order_relaxed) > 0;
}

void RDKitFallbackQueue::processEntry(const RDKitFallbackEntry& entry) {
  ScopedNvtxRange pairRange("RDKit fallback T" + std::to_string(entry.originalTargetIdx) + 
                            "/Q" + std::to_string(entry.originalQueryIdx));

  const RDKit::ROMol* target = (*targets_)[entry.originalTargetIdx];
  const RDKit::ROMol* query  = (*queries_)[entry.originalQueryIdx];

  const int effectiveMaxMatches = boolResults_ ? 1 : maxMatches_;
  processWithRDKitFallback(target, query, entry.originalTargetIdx, entry.originalQueryIdx,
                           *results_, *resultsMutex_, effectiveMaxMatches, boolResults_, countResults_);

  processedCount_.fetch_add(1, std::memory_order_relaxed);
}

// =============================================================================
// Pipelined Batch Processing Types (internal, but needs external linkage for forward decl)
// =============================================================================

std::pair<int, int> getStreamPriorityRange();

struct GpuExecutor {
  int miniBatchStart        = 0;
  int numPairsInMiniBatch   = 0;
  int totalMatchIndices = 0;

  // Pointers into consolidated pinned buffer (not owned)
  int*     pairIndicesHost          = nullptr;
  int*     miniBatchPairMatchStarts = nullptr;
  int*     matchCountsHost          = nullptr;
  int*     reportedCountsHost       = nullptr;
  int16_t* matchIndicesHost         = nullptr;

  // Precomputed recursive mini-batch setup (populated by prepareRecursiveMiniBatchOnCPU)
  int recursiveMaxDepth       = 0;
  int firstTargetInMiniBatch  = 0;
  int numTargetsInMiniBatch   = 0;
  std::array<std::vector<BatchedPatternEntry>, kMaxRecursionDepth + 1> patternsAtDepth;

  // Streams and events (declared first so they're destroyed last)
  ScopedStream             computeStream;
  ScopedCudaEvent          copyDoneEvent;
  ScopedCudaEvent          allocDoneEvent;

  // Recursive pipeline (inlined from RecursivePipelineContext)
  ScopedStreamWithPriority recursiveStream;
  ScopedStreamWithPriority postRecursionStream;
  std::array<ScopedCudaEvent, kMaxRecursionDepth> depthEvents;
  ScopedCudaEvent          recursiveDoneEvent;
  ScopedCudaEvent          postRecursionDoneEvent;
  std::array<AsyncDeviceVector<int>, kMaxRecursionDepth + 1> matchGlobalPairIndices;
  std::array<AsyncDeviceVector<int>, kMaxRecursionDepth + 1> matchMiniBatchLocalIndices;
  std::array<int*, kMaxRecursionDepth + 1> matchGlobalPairIndicesHost = {};
  std::array<int*, kMaxRecursionDepth + 1> matchMiniBatchLocalIndicesHost = {};
  std::array<int, kMaxRecursionDepth + 1> matchPairsCounts = {};
  int perDepthCapacity = 0;
  int maxDepthInMiniBatch = 0;

  RecursiveScratchBuffers  recursiveScratch;
  MiniBatchResultsDevice   deviceResults;
  AsyncDeviceVector<int>   pairIndicesDev;
  
  int deviceId = 0;  ///< GPU device ID this executor is assigned to

  GpuExecutor(int executorIdx, int gpuDeviceId) 
      : computeStream(("executor" + std::to_string(executorIdx) + "_mainStream").c_str()),
        recursiveStream(getStreamPriorityRange().first, 
                        ("executor" + std::to_string(executorIdx) + "_priorityRecursiveStream").c_str()),
        postRecursionStream(getStreamPriorityRange().second,
                            ("executor" + std::to_string(executorIdx) + "_postRecursionStream").c_str()),
        recursiveScratch(nullptr),
        deviceId(gpuDeviceId) {}

  void initializeForStream() {
    cudaStream_t s = computeStream.stream();
    cudaStream_t recStream = recursiveStream.stream();
    deviceResults.setStream(s);
    pairIndicesDev.setStream(s);
    recursiveScratch.setStream(recStream);
  }

  cudaStream_t stream() const { return computeStream.stream(); }

  /**
   * @brief Bind pointers from consolidated pinned buffer.
   */
  void bindPinnedBuffer(ConsolidatedPinnedBuffer& buffer) {
    pairIndicesHost          = buffer.pairIndices;
    miniBatchPairMatchStarts = buffer.miniBatchPairMatchStarts;
    matchCountsHost          = buffer.matchCounts;
    reportedCountsHost       = buffer.reportedCounts;
    matchIndicesHost         = buffer.matchIndices;

    matchGlobalPairIndicesHost = buffer.matchGlobalPairIndicesHost;
    matchMiniBatchLocalIndicesHost = buffer.matchBatchLocalIndicesHost;
    perDepthCapacity = buffer.perDepthCapacity;

    recursiveScratch.setPinnedBuffer(buffer.patternsAtDepthHost, buffer.patternsCapacity);
  }
};

// =============================================================================
// LeafSubpatterns Implementation
// =============================================================================

void LeafSubpatterns::buildAllPatterns(const MoleculesHost& queriesHost) {
  ScopedNvtxRange buildRange("LeafSubpatterns::buildAllPatterns");

  const int numQueries = static_cast<int>(queriesHost.numMolecules());

  // First pass: build pattern molecules and register in patternIndexMap
  for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
    if (queryIdx >= static_cast<int>(queriesHost.recursivePatterns.size())) {
      continue;
    }

    const auto& recursiveInfo = queriesHost.recursivePatterns[queryIdx];
    if (recursiveInfo.empty()) {
      continue;
    }

    for (const auto& entry : recursiveInfo.patterns) {
      if (entry.queryMol == nullptr) {
        continue;
      }

      LeafSubpatternKey key{queryIdx, entry.patternId};
      if (patternIndexMap.find(key) != patternIndexMap.end()) {
        continue;
      }

      int molIdx = static_cast<int>(patternsHost.numMolecules());

      std::vector<std::pair<int, int>> childrenByLocalId;
      for (const auto& p : recursiveInfo.patterns) {
        if (p.parentPatternId == entry.patternId) {
          childrenByLocalId.emplace_back(p.localIdInParent, p.patternId);
        }
      }
      std::sort(childrenByLocalId.begin(), childrenByLocalId.end());

      std::vector<int> childPatternIds;
      for (const auto& [localId, childId] : childrenByLocalId) {
        childPatternIds.push_back(childId);
      }

      if constexpr (kDebugPaintRecursive) {
        printf("[LeafSubpatterns] buildAllPatterns: queryIdx=%d, patternId=%d, found %zu children: [",
               queryIdx, entry.patternId, childPatternIds.size());
        for (size_t i = 0; i < childPatternIds.size(); ++i) {
          printf("%d%s", childPatternIds[i], i + 1 < childPatternIds.size() ? "," : "");
        }
        printf("]\n");
      }

      if (childPatternIds.empty()) {
        addQueryToBatch(entry.queryMol, patternsHost);
      } else {
        addQueryToBatch(entry.queryMol, patternsHost, childPatternIds);
      }

      patternIndexMap[key] = molIdx;
    }
  }

  // Second pass: build precomputed BatchedPatternEntry structures
  perQueryPatterns.resize(numQueries);
  perQueryMaxDepth.resize(numQueries, 0);

  for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
    if (queryIdx >= static_cast<int>(queriesHost.recursivePatterns.size())) {
      continue;
    }

    const auto& recursiveInfo = queriesHost.recursivePatterns[queryIdx];
    if (recursiveInfo.empty()) {
      continue;
    }

    perQueryMaxDepth[queryIdx] = recursiveInfo.maxDepth;

    for (const auto& entry : recursiveInfo.patterns) {
      if (entry.queryMol == nullptr) {
        continue;
      }

      const int patternMolIdx = getPatternIndex(queryIdx, entry.patternId);
      if (patternMolIdx < 0) {
        continue;
      }

      BatchedPatternEntry batchEntry;
      batchEntry.mainQueryIdx    = queryIdx;
      batchEntry.patternId       = entry.patternId;
      batchEntry.patternMolIdx   = patternMolIdx;
      batchEntry.depth           = entry.depth;
      batchEntry.localIdInParent = entry.localIdInParent;

      perQueryPatterns[queryIdx][entry.depth].push_back(batchEntry);
    }
  }
}

void LeafSubpatterns::syncToDevice(cudaStream_t stream) {
  ScopedNvtxRange syncRange("LeafSubpatterns::syncToDevice");
  
  if (!patternsHost.numMolecules()) {
    return;
  }
  patternsDevice.copyFromHost(patternsHost, stream);
}

// =============================================================================
// Stream Priority Helper
// =============================================================================

std::pair<int, int> getStreamPriorityRange() {
  int leastPriority    = 0;
  int greatestPriority = 0;
  cudaCheckError(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  return {greatestPriority, leastPriority};
}

// =============================================================================
// MiniBatchResultsDevice Implementation
// =============================================================================

void MiniBatchResultsDevice::setStream(cudaStream_t stream) {
  stream_ = stream;
  matchCounts_.setStream(stream);
  reportedCounts_.setStream(stream);
  pairMatchStarts_.setStream(stream);
  matchIndices_.setStream(stream);
  queryAtomCounts_.setStream(stream);
  overflowBuffer_.setStream(stream);
  recursiveMatchBits_.setStream(stream);
  labelMatrixBuffer_.setStream(stream);
}

void MiniBatchResultsDevice::allocateMiniBatch(int        miniBatchSize,
                                               const int* miniBatchPairMatchStarts,
                                               int        totalMiniBatchMatchIndices,
                                               int        numQueries,
                                               int        maxTargetAtoms,
                                               int        numBuffersPerBlock,
                                               int        maxMatchesToFind,
                                               bool       countOnly) {
  ScopedNvtxRange allocRange("MiniBatchResultsDevice::allocateMiniBatch");
  
  miniBatchSize_              = miniBatchSize;
  numQueries_             = numQueries;
  maxTargetAtoms_         = maxTargetAtoms;
  totalMiniBatchMatchIndices_ = countOnly ? 0 : totalMiniBatchMatchIndices;
  overflowBuffersPerBlock_ = numBuffersPerBlock;
  maxMatchesToFind_       = maxMatchesToFind;
  countOnly_              = countOnly;

  if (matchCounts_.size() < static_cast<size_t>(miniBatchSize)) {
    matchCounts_.resize(static_cast<size_t>(miniBatchSize * 1.5));
  }

  if (!countOnly) {
    if (reportedCounts_.size() < static_cast<size_t>(miniBatchSize)) {
      reportedCounts_.resize(static_cast<size_t>(miniBatchSize * 1.5));
    }

    if (pairMatchStarts_.size() < static_cast<size_t>(miniBatchSize + 1)) {
      pairMatchStarts_.resize(static_cast<size_t>((miniBatchSize + 1) * 1.5));
    }
    pairMatchStarts_.copyFromHost(miniBatchPairMatchStarts, miniBatchSize + 1);

    if (matchIndices_.size() < static_cast<size_t>(totalMiniBatchMatchIndices)) {
      matchIndices_.resize(static_cast<size_t>(totalMiniBatchMatchIndices) * 3 / 2);
    }
  }

  const int overflowEntries = miniBatchSize * numBuffersPerBlock * kOverflowEntriesPerBuffer;
  if (overflowBuffer_.size() < static_cast<size_t>(overflowEntries)) {
    overflowBuffer_.resize(static_cast<size_t>(overflowEntries * 1.5));
  }

  const size_t recursiveBitsSize = static_cast<size_t>(miniBatchSize) * maxTargetAtoms;
  if (recursiveMatchBits_.size() < recursiveBitsSize) {
    recursiveMatchBits_.resize(static_cast<size_t>(recursiveBitsSize * 1.5));
  }
  recursiveMatchBits_.zero();

  const size_t labelMatrixSize = static_cast<size_t>(miniBatchSize) * kLabelMatrixWords;
  if (labelMatrixBuffer_.size() < labelMatrixSize) {
    labelMatrixBuffer_.resize(static_cast<size_t>(labelMatrixSize * 1.5));
  }
}

void MiniBatchResultsDevice::setQueryAtomCounts(const int* queryAtomCounts, size_t count) {
  if (queryAtomCounts_.size() < count) {
    queryAtomCounts_.resize(static_cast<size_t>(count * 1.5));
  }
  queryAtomCounts_.copyFromHost(queryAtomCounts, count);
}

void MiniBatchResultsDevice::zeroRecursiveBits() {
  recursiveMatchBits_.zero();
}

void MiniBatchResultsDevice::copyMiniBatchToHost(int*     hostMatchCounts,
                                                 int*     hostReportedCounts,
                                                 int16_t* hostMatchIndices) const {
  matchCounts_.copyToHost(hostMatchCounts, miniBatchSize_);
  reportedCounts_.copyToHost(hostReportedCounts, miniBatchSize_);
  matchIndices_.copyToHost(hostMatchIndices, totalMiniBatchMatchIndices_);
}

void MiniBatchResultsDevice::copyCountsOnlyToHost(int* hostMatchCounts) const {
  matchCounts_.copyToHost(hostMatchCounts, miniBatchSize_);
}

// =============================================================================
// Pipelined Batch Processing Implementation
// =============================================================================

namespace {

/**
 * @brief Precompute the pipeline schedule for a mini-batch.
 *
 * Groups pairs by their query's recursion depth and populates the host-side
 * index vectors for the recursive stream and match stream.
 *
 * @param executor GPU executor to populate with schedule
 * @param ctx Worker context with cached query depths
 * @param numPairsInMiniBatch Number of pairs in the mini-batch
 * @param miniBatchStart Global pair index where the mini-batch starts
 */
void precomputePipelineSchedule(GpuExecutor&               executor,
                                const ThreadWorkerContext& ctx,
                                int                        numPairsInMiniBatch,
                                int                        miniBatchStart) {
  ScopedNvtxRange scheduleRange("CPU: precomputePipelineSchedule");
  int maxDepth = 0;

  executor.matchPairsCounts.fill(0);

  int queryIdx = miniBatchStart % ctx.numQueries;
  for (int i = 0; i < numPairsInMiniBatch; ++i) {
    const int depth = ctx.queryDepths[queryIdx];
    const int offset = executor.matchPairsCounts[depth]++;

    executor.matchGlobalPairIndicesHost[depth][offset] = executor.pairIndicesHost[i];
    executor.matchMiniBatchLocalIndicesHost[depth][offset] = i;

    if (depth > maxDepth) {
      maxDepth = depth;
    }

    if (++queryIdx >= ctx.numQueries) {
      queryIdx = 0;
    }
  }
  executor.maxDepthInMiniBatch = maxDepth;
}

void prepareRecursiveMiniBatchOnCPU(GpuExecutor&               executor,
                                    const ThreadWorkerContext& ctx,
                                    const LeafSubpatterns&     leafSubpatterns) {
  ScopedNvtxRange prepRecRange("prepareRecursiveMiniBatchOnCPU");

  precomputePipelineSchedule(executor, ctx, executor.numPairsInMiniBatch, executor.miniBatchStart);

  for (auto& vec : executor.patternsAtDepth) {
    vec.clear();
  }

  const int numUniqueQueries = std::min(executor.numPairsInMiniBatch, ctx.numQueries);

  int maxDepth = 0;
  int queryIdx = executor.miniBatchStart % ctx.numQueries;
  for (int i = 0; i < numUniqueQueries; ++i) {
    if (ctx.queryHasPatterns[queryIdx]) {
      const int queryMaxDepth = ctx.queryMaxDepths[queryIdx];
      if (queryMaxDepth > maxDepth) {
        maxDepth = queryMaxDepth;
      }

      for (int d = 0; d <= queryMaxDepth; ++d) {
        const auto& srcEntries = leafSubpatterns.perQueryPatterns[queryIdx][d];
        auto& destEntries = executor.patternsAtDepth[d];
        destEntries.insert(destEntries.end(), srcEntries.begin(), srcEntries.end());
      }
    }

    if (++queryIdx >= ctx.numQueries) {
      queryIdx = 0;
    }
  }
  executor.recursiveMaxDepth = maxDepth;

  executor.firstTargetInMiniBatch = executor.miniBatchStart / ctx.numQueries;
  const int lastTargetInMiniBatch = (executor.miniBatchStart + executor.numPairsInMiniBatch - 1) / ctx.numQueries;
  executor.numTargetsInMiniBatch = lastTargetInMiniBatch - executor.firstTargetInMiniBatch + 1;
}

void prepareMiniBatchOnCPU(GpuExecutor&                 executor,
                           const ThreadWorkerContext&   ctx,
                           const MoleculesHost&         queriesHost,
                           const LeafSubpatterns&       leafSubpatterns,
                           int                          miniBatchStart,
                           int                          maxPairsInMiniBatch) {
  ScopedNvtxRange prepRange("prepareMiniBatchOnCPU");

  const int numPairs = ctx.numTargets * ctx.numQueries;
  const int miniBatchEnd = std::min(miniBatchStart + maxPairsInMiniBatch, numPairs);
  const int numPairsInMiniBatch = miniBatchEnd - miniBatchStart;

  executor.miniBatchStart       = miniBatchStart;
  executor.numPairsInMiniBatch  = numPairsInMiniBatch;

  const bool useMaxMatchesLimit = ctx.maxMatches > 0;
  int sortedTargetIdx = miniBatchStart / ctx.numQueries;
  int sortedQueryIdx  = miniBatchStart % ctx.numQueries;

  executor.miniBatchPairMatchStarts[0] = 0;
  for (int i = 0; i < numPairsInMiniBatch; ++i) {
    const int targetAtoms = ctx.targetAtomCounts[sortedTargetIdx];
    const int queryAtoms  = ctx.queryAtomCounts[sortedQueryIdx];
    const int pairCapacity = useMaxMatchesLimit
        ? (ctx.maxMatches * queryAtoms)
        : (targetAtoms * queryAtoms);
    executor.miniBatchPairMatchStarts[i + 1] = executor.miniBatchPairMatchStarts[i] + pairCapacity;
    executor.pairIndicesHost[i] = miniBatchStart + i;

    if (++sortedQueryIdx >= ctx.numQueries) {
      sortedQueryIdx = 0;
      ++sortedTargetIdx;
    }
  }
  executor.totalMatchIndices = executor.miniBatchPairMatchStarts[numPairsInMiniBatch];

  prepareRecursiveMiniBatchOnCPU(executor, ctx, leafSubpatterns);
}

/**
 * @brief Launch label matrix and match kernels for a subset of pairs.
 */
void launchLabelAndMatch(int                          numPairsInGroup,
                         GpuExecutor&                 executor,
                         const ThreadWorkerContext&   ctx,
                         MoleculesDevice&             targetsDevice,
                         const MoleculesDevice&       queriesDevice,
                         SubstructAlgorithm           algorithm,
                         cudaStream_t                 stream,
                         int                          depthGroupIdx) {
  ScopedNvtxRange launchRange("launchLabelAndMatch depth=" + std::to_string(depthGroupIdx));
  
  if (numPairsInGroup == 0) {
    return;
  }

  int* globalPairIndicesHost = executor.matchGlobalPairIndicesHost[depthGroupIdx];
  int* miniBatchLocalIndicesHostPtr = executor.matchMiniBatchLocalIndicesHost[depthGroupIdx];

  auto& globalPairIndicesDev = executor.matchGlobalPairIndices[depthGroupIdx];
  auto& miniBatchLocalIndicesDev = executor.matchMiniBatchLocalIndices[depthGroupIdx];

  globalPairIndicesDev.setStream(stream);
  if (globalPairIndicesDev.size() < static_cast<size_t>(numPairsInGroup)) {
    globalPairIndicesDev.resize(static_cast<size_t>(numPairsInGroup * 1.5));
  }
  globalPairIndicesDev.copyFromHost(globalPairIndicesHost, numPairsInGroup);

  miniBatchLocalIndicesDev.setStream(stream);
  if (miniBatchLocalIndicesDev.size() < static_cast<size_t>(numPairsInGroup)) {
    miniBatchLocalIndicesDev.resize(static_cast<size_t>(numPairsInGroup * 1.5));
  }
  miniBatchLocalIndicesDev.copyFromHost(miniBatchLocalIndicesHostPtr, numPairsInGroup);

  launchLabelMatrixKernel(
    ctx.templateConfig,
    targetsDevice.view(),
    queriesDevice.view(),
    globalPairIndicesDev.data(),
    numPairsInGroup,
    ctx.numQueries,
    executor.deviceResults.labelMatrixBuffer(),
    executor.deviceResults.recursiveMatchBits(),
    executor.deviceResults.maxTargetAtoms(),
    miniBatchLocalIndicesDev.data(),
    stream);

  launchSubstructMatchKernel(
    ctx.templateConfig,
    algorithm,
    targetsDevice.view(),
    queriesDevice.view(),
    executor.deviceResults,
    globalPairIndicesDev.data(),
    numPairsInGroup,
    ctx.numQueries,
    miniBatchLocalIndicesDev.data(),
    nullptr,
    stream);
}

void launchRecursivePaintKernels(
    SubstructTemplateConfig                                                  templateConfig,
    const MoleculesDevice&                                                   targetsDevice,
    const LeafSubpatterns&                                                   leafSubpatterns,
    MiniBatchResultsDevice&                                                  miniBatchResults,
    int                                                                      numQueries,
    int                                                                      miniBatchPairOffset,
    int                                                                      miniBatchSize,
    SubstructAlgorithm                                                       algorithm,
    cudaStream_t                                                             stream,
    RecursiveScratchBuffers&                                                 scratch,
    const std::array<std::vector<BatchedPatternEntry>, kMaxRecursionDepth + 1>& patternsAtDepth,
    int                                                                      maxDepth,
    int                                                                      firstTargetInMiniBatch,
    int                                                                      numTargetsInMiniBatch,
    cudaEvent_t*                                                             depthEvents,
    int                                                                      numDepthEvents) {
  ScopedNvtxRange processRecursiveRange("launchRecursivePaintKernels");

  scratch.setStream(stream);

  constexpr int gsiBuffersPerBlock = 2;

  const int maxPaintPairsPerSubBatch = std::max(miniBatchSize, 1024);

  for (int currentDepth = 0; currentDepth <= maxDepth; ++currentDepth) {
    ScopedNvtxRange depthRange("Process recursive depth level " + std::to_string(currentDepth));

    const auto& patternsForDepth = patternsAtDepth[currentDepth];

    if (patternsForDepth.empty()) {
      if (currentDepth < numDepthEvents && depthEvents != nullptr) {
        cudaCheckError(cudaEventRecord(depthEvents[currentDepth], stream));
      }
      continue;
    }

    const size_t numPatterns = patternsForDepth.size();
    const int patternsPerSubBatch = std::max(1, maxPaintPairsPerSubBatch / numTargetsInMiniBatch);

    for (size_t patternStart = 0; patternStart < numPatterns; patternStart += patternsPerSubBatch) {
      ScopedNvtxRange subBatchRange("Process sub-batch " + std::to_string(patternStart));
      
      const size_t patternEnd            = std::min(patternStart + patternsPerSubBatch, numPatterns);
      const size_t numPatternsInSubBatch = patternEnd - patternStart;
      const size_t numBlocksInSubBatch   = numTargetsInMiniBatch * numPatternsInSubBatch;

      ScopedNvtxRange prepareRange("GPU: Upload pattern entries");
      const int bufferIdx = scratch.acquireBufferIndex();
      scratch.waitForBuffer(bufferIdx);
      scratch.ensureCapacity(bufferIdx, static_cast<int>(numPatternsInSubBatch));
      for (size_t i = 0; i < numPatternsInSubBatch; ++i) {
        scratch.patternsAtDepthHost[bufferIdx][i] = patternsForDepth[patternStart + i];
      }
      prepareRange.pop();

      const int buffersPerBlock = gsiBuffersPerBlock;
      const size_t overflowNeeded = numBlocksInSubBatch * buffersPerBlock * kOverflowEntriesPerBuffer;

      if (scratch.overflow.size() < overflowNeeded) {
        scratch.overflow.resize(static_cast<size_t>(overflowNeeded * 1.5));
      }

      const size_t labelMatrixNeeded = numBlocksInSubBatch * kLabelMatrixWords;
      if (scratch.labelMatrixBuffer.size() < labelMatrixNeeded) {
        scratch.labelMatrixBuffer.resize(static_cast<size_t>(labelMatrixNeeded * 1.5));
      }

      if (scratch.patternEntries.size() < numPatternsInSubBatch) {
        scratch.patternEntries.resize(static_cast<size_t>(numPatternsInSubBatch * 1.5));
      }
      
      scratch.patternEntries.copyFromHost(scratch.patternsAtDepthHost[bufferIdx], numPatternsInSubBatch);
      scratch.recordCopy(bufferIdx, scratch.patternEntries.stream());

      const uint32_t* recursiveBitsForLabel = (currentDepth > 0) ? miniBatchResults.recursiveMatchBits() : nullptr;

      launchLabelMatrixPaintKernel(
        templateConfig,
        targetsDevice.view(),
        leafSubpatterns.view(),
        scratch.patternEntries.data(),
        static_cast<int>(numPatternsInSubBatch),
        numBlocksInSubBatch,
        numQueries,
        miniBatchPairOffset,
        miniBatchSize,
        scratch.labelMatrixBuffer.data(),
        firstTargetInMiniBatch,
        recursiveBitsForLabel,
        miniBatchResults.maxTargetAtoms(),
        stream);

      launchSubstructPaintKernel(
        templateConfig,
        algorithm,
        targetsDevice.view(),
        leafSubpatterns.view(),
        scratch.patternEntries.data(),
        static_cast<int>(numPatternsInSubBatch),
        numBlocksInSubBatch,
        miniBatchResults.recursiveMatchBits(),
        miniBatchResults.maxTargetAtoms(),
        numQueries,
        0, 0,
        miniBatchPairOffset,
        miniBatchSize,
        scratch.overflow.data(),
        scratch.overflow.data(),
        kOverflowEntriesPerBuffer,
        scratch.labelMatrixBuffer.data(),
        firstTargetInMiniBatch,
        stream);
    }

    if (currentDepth < numDepthEvents && depthEvents != nullptr) {
      cudaCheckError(cudaEventRecord(depthEvents[currentDepth], stream));
    }
  }

  cudaCheckError(cudaGetLastError());
}

void uploadAndLaunchMiniBatch(GpuExecutor&               executor,
                              const ThreadWorkerContext& ctx,
                              MoleculesDevice&           targetsDevice,
                              const MoleculesDevice&     queriesDevice,
                              const LeafSubpatterns&     leafSubpatterns,
                              SubstructAlgorithm         algorithm) {
  ScopedNvtxRange uploadRange("uploadAndLaunchMiniBatch");

  cudaStream_t executorStream = executor.stream();
  const int numBuffersPerBlock = (algorithm == SubstructAlgorithm::GSI) ? 2 : 1;

  if (executor.maxDepthInMiniBatch == 0) {
    ScopedNvtxRange nonRecursiveRange("Non-recursive path");
    
    const int maxMatchesToFind = ctx.maxMatches > 0 ? ctx.maxMatches : -1;
    executor.deviceResults.allocateMiniBatch(executor.numPairsInMiniBatch,
                                             executor.miniBatchPairMatchStarts,
                                             executor.totalMatchIndices,
                                             ctx.numQueries,
                                             ctx.maxTargetAtoms,
                                             numBuffersPerBlock,
                                             maxMatchesToFind,
                                             ctx.countOnly);
    executor.deviceResults.setQueryAtomCounts(ctx.queryAtomCounts.data(), ctx.numQueries);

    if (executor.pairIndicesDev.size() < static_cast<size_t>(executor.numPairsInMiniBatch)) {
      executor.pairIndicesDev.resize(static_cast<size_t>(executor.numPairsInMiniBatch * 1.5));
    }
    executor.pairIndicesDev.copyFromHost(executor.pairIndicesHost, executor.numPairsInMiniBatch);

    launchLabelMatrixKernel(
      ctx.templateConfig,
      targetsDevice.view(),
      queriesDevice.view(),
      executor.pairIndicesDev.data(),
      executor.numPairsInMiniBatch,
      ctx.numQueries,
      executor.deviceResults.labelMatrixBuffer(),
      executor.deviceResults.recursiveMatchBits(),
      executor.deviceResults.maxTargetAtoms(),
      nullptr,
      executorStream);

    launchSubstructMatchKernel(
      ctx.templateConfig,
      algorithm,
      targetsDevice.view(),
      queriesDevice.view(),
      executor.deviceResults,
      executor.pairIndicesDev.data(),
      executor.numPairsInMiniBatch,
      ctx.numQueries,
      nullptr,
      nullptr,
      executorStream);
    return;
  }

  ScopedNvtxRange multiStreamRange("Multi-stream recursive pipeline");

  cudaStream_t recursiveStream = executor.recursiveStream.stream();

  const int maxMatchesToFind = ctx.maxMatches > 0 ? ctx.maxMatches : -1;
  executor.deviceResults.allocateMiniBatch(executor.numPairsInMiniBatch,
                                           executor.miniBatchPairMatchStarts,
                                           executor.totalMatchIndices,
                                           ctx.numQueries,
                                           ctx.maxTargetAtoms,
                                           numBuffersPerBlock,
                                           maxMatchesToFind,
                                           ctx.countOnly);
  executor.deviceResults.setQueryAtomCounts(ctx.queryAtomCounts.data(), ctx.numQueries);

  cudaCheckError(cudaEventRecord(executor.allocDoneEvent.event(), executorStream));
  
  ScopedNvtxRange waitAllocRange("Wait: recursiveStream waits for alloc");
  cudaCheckError(cudaStreamWaitEvent(recursiveStream, executor.allocDoneEvent.event(), 0));
  waitAllocRange.pop();

  std::array<cudaEvent_t, kMaxRecursionDepth> depthEventPtrs;
  for (int i = 0; i < kMaxRecursionDepth; ++i) {
    depthEventPtrs[i] = executor.depthEvents[i].event();
  }

  ScopedNvtxRange preprocRange("launchRecursivePaintKernels (recursiveStream)");
  launchRecursivePaintKernels(ctx.templateConfig, targetsDevice, leafSubpatterns,
                              executor.deviceResults, ctx.numQueries,
                              executor.miniBatchStart, executor.numPairsInMiniBatch,
                              algorithm, recursiveStream,
                              executor.recursiveScratch,
                              executor.patternsAtDepth,
                              executor.recursiveMaxDepth,
                              executor.firstTargetInMiniBatch,
                              executor.numTargetsInMiniBatch,
                              depthEventPtrs.data(),
                              kMaxRecursionDepth);
  preprocRange.pop();

  ScopedNvtxRange depth0Range("Match depth-0 pairs (executorStream)");
  launchLabelAndMatch(executor.matchPairsCounts[0], executor, ctx, targetsDevice, queriesDevice,
                      algorithm, executorStream, 0);
  depth0Range.pop();

  cudaStream_t postStream = executor.postRecursionStream.stream();
  cudaCheckError(cudaStreamWaitEvent(postStream, executor.allocDoneEvent.event(), 0));

  for (int depth = 1; depth <= executor.maxDepthInMiniBatch; ++depth) {
    ScopedNvtxRange depthRange("Match depth-" + std::to_string(depth) + " pairs (postRecursionStream)");

    ScopedNvtxRange waitRange("Wait: postRecursionStream waits for depth event");
    cudaCheckError(cudaStreamWaitEvent(postStream, depthEventPtrs[depth - 1], 0));
    waitRange.pop();

    launchLabelAndMatch(executor.matchPairsCounts[depth], executor, ctx, targetsDevice, queriesDevice,
                        algorithm, postStream, depth);
  }
  cudaCheckError(cudaEventRecord(executor.postRecursionDoneEvent.event(), postStream));

  cudaCheckError(cudaEventRecord(executor.recursiveDoneEvent.event(), recursiveStream));
  cudaCheckError(cudaStreamWaitEvent(executorStream, executor.recursiveDoneEvent.event(), 0));
  cudaCheckError(cudaStreamWaitEvent(executorStream, executor.postRecursionDoneEvent.event(), 0));
}

void initiateResultsCopyToHost(GpuExecutor& executor) {
  ScopedNvtxRange copyRange("initiateResultsCopyToHost");
  executor.deviceResults.copyMiniBatchToHost(executor.matchCountsHost, executor.reportedCountsHost, executor.matchIndicesHost);
  cudaCheckError(cudaEventRecord(executor.copyDoneEvent.event(), executor.stream()));
}

struct PairUpdate {
  int targetIdx;
  int queryIdx;
  int miniBatchLocalOffset;
  int reportedMatches;
  int queryAtoms;
};

void accumulateMiniBatchResults(GpuExecutor& executor,
  const ThreadWorkerContext& ctx,
  SubstructSearchResults& results,
  std::mutex& resultsMutex,
  RDKitFallbackQueue* fallbackQueue = nullptr) {
  ScopedNvtxRange accumRange("accumulateMiniBatchResults");

  const bool hasTargetSort = ctx.targetSortOrder != nullptr;
  const bool hasQuerySort = ctx.querySortOrder != nullptr;

  // Phase 1: Build update list without any locks

  std::vector<PairUpdate> updates;
  updates.reserve(executor.numPairsInMiniBatch);

  for (int i = 0; i < executor.numPairsInMiniBatch; ++i) {
    const int pairIdxInMacrobatch = executor.miniBatchStart + i;
    const int sortedTargetIdx = pairIdxInMacrobatch / ctx.numQueries;
    const int sortedQueryIdx  = pairIdxInMacrobatch % ctx.numQueries;

    const int targetIdx = hasTargetSort ? (*ctx.targetSortOrder)[sortedTargetIdx] : sortedTargetIdx;
    const int queryIdx  = hasQuerySort ? (*ctx.querySortOrder)[sortedQueryIdx] : sortedQueryIdx;

    const int queryAtoms      = ctx.queryAtomCounts[sortedQueryIdx];
    const int actualMatches   = executor.matchCountsHost[i];
    const int reportedMatches = executor.reportedCountsHost[i];

    const bool isBufferOverflow = (actualMatches > reportedMatches) && (ctx.maxMatches == 0);
    if (isBufferOverflow && fallbackQueue != nullptr) {
      fallbackQueue->enqueue({targetIdx, queryIdx});
      continue;
    }

    if (reportedMatches > 0) {
      updates.push_back({targetIdx, queryIdx, executor.miniBatchPairMatchStarts[i], reportedMatches, queryAtoms});
    }
  }

  if (updates.empty()) { 
    return; 
  }

  // Phase 2: Single lock acquisition to get all hash map references
  std::vector<std::vector<std::vector<int>>*> matchRefs;
  matchRefs.reserve(updates.size());

  {
    std::lock_guard<std::mutex> lock(resultsMutex);
    for (const auto& u : updates) {
      matchRefs.push_back(&results.getMatchesMut(u.targetIdx, u.queryIdx));
    }
  }
  // Lock released here

  // Phase 3: Copy all match data without holding the lock
  // Safe because mini-batches have non-overlapping pair indices
  for (size_t i = 0; i < updates.size(); ++i) {
    const auto& u = updates[i];
    auto& targetMatches = *matchRefs[i];

    targetMatches.reserve(targetMatches.size() + u.reportedMatches);
    const int16_t* src = executor.matchIndicesHost + u.miniBatchLocalOffset;

    for (int m = 0; m < u.reportedMatches; ++m) {
      auto& match = targetMatches.emplace_back(u.queryAtoms);
    for (int a = 0; a < u.queryAtoms; ++a) {
      match[a] = src[m * u.queryAtoms + a];
    }
    }
  }
}

void initiateCountsOnlyCopyToHost(GpuExecutor& executor) {
  ScopedNvtxRange copyRange("initiateCountsOnlyCopyToHost");
  executor.deviceResults.copyCountsOnlyToHost(executor.matchCountsHost);
  cudaCheckError(cudaEventRecord(executor.copyDoneEvent.event(), executor.stream()));
}

void accumulateMiniBatchResultsBoolean(GpuExecutor&              executor,
                                       const ThreadWorkerContext& ctx,
                                       HasSubstructMatchResults&  results,
                                       std::mutex&                resultsMutex) {
  ScopedNvtxRange accumRange("accumulateMiniBatchResultsBoolean");

  const bool hasTargetSort = ctx.targetSortOrder != nullptr;
  const bool hasQuerySort  = ctx.querySortOrder != nullptr;

  std::lock_guard<std::mutex> lock(resultsMutex);

  for (int i = 0; i < executor.numPairsInMiniBatch; ++i) {
    if (executor.matchCountsHost[i] == 0) {
      continue;
    }

    const int pairIdxInMacrobatch = executor.miniBatchStart + i;
    const int sortedTargetIdx     = pairIdxInMacrobatch / ctx.numQueries;
    const int sortedQueryIdx      = pairIdxInMacrobatch % ctx.numQueries;

    const int targetIdx = hasTargetSort ? (*ctx.targetSortOrder)[sortedTargetIdx] : sortedTargetIdx;
    const int queryIdx  = hasQuerySort ? (*ctx.querySortOrder)[sortedQueryIdx] : sortedQueryIdx;

    results.setMatch(targetIdx, queryIdx, true);
  }
}

void accumulateMiniBatchResultsCounts(GpuExecutor&              executor,
                                      const ThreadWorkerContext& ctx,
                                      std::vector<int>&          counts,
                                      std::mutex&                resultsMutex) {
  ScopedNvtxRange accumRange("accumulateMiniBatchResultsCounts");

  const bool hasTargetSort = ctx.targetSortOrder != nullptr;
  const bool hasQuerySort  = ctx.querySortOrder != nullptr;

  std::lock_guard<std::mutex> lock(resultsMutex);

  for (int i = 0; i < executor.numPairsInMiniBatch; ++i) {
    const int pairIdxInMacrobatch = executor.miniBatchStart + i;
    const int sortedTargetIdx     = pairIdxInMacrobatch / ctx.numQueries;
    const int sortedQueryIdx      = pairIdxInMacrobatch % ctx.numQueries;

    const int targetIdx = hasTargetSort ? (*ctx.targetSortOrder)[sortedTargetIdx] : sortedTargetIdx;
    const int queryIdx  = hasQuerySort ? (*ctx.querySortOrder)[sortedQueryIdx] : sortedQueryIdx;

    const int pairIdx = targetIdx * ctx.numQueries + queryIdx;
    counts[pairIdx] = executor.matchCountsHost[i];
  }
}

constexpr int kMaxExecutorsPerRunner = 8;

template <typename AccumulateFunc>
void runnerWorkerInlineCountOnly(int                               workerIdx,
                                 const ThreadWorkerContext&        ctx,
                                 MoleculesDevice&                  targetsDevice,
                                 const MoleculesDevice&            queriesDevice,
                                 const MoleculesHost&              queriesHost,
                                 const LeafSubpatterns&            leafSubpatterns,
                                 SubstructAlgorithm                algorithm,
                                 cudaEvent_t                       upstreamReadyEvent,
                                 std::atomic<int>&                 nextMiniBatchIdx,
                                 int                               totalNumMiniBatches,
                                 int                               effectiveMiniBatchSize,
                                 int                               deviceId,
                                 std::vector<GpuExecutor*>         executors,
                                 AccumulateFunc&&                  accumulate,
                                 std::exception_ptr&               exceptionPtr) {
  try {
    ScopedNvtxRange workerRange("runnerWorkerInlineCountOnly " + std::to_string(workerIdx) + " GPU" + std::to_string(deviceId));
    const WithDevice setDevice(deviceId);

    const int executorsPerRunner = static_cast<int>(executors.size());
    const int numPairs = ctx.numTargets * ctx.numQueries;

    std::array<GpuExecutor*, kMaxExecutorsPerRunner> pendingExecutors{};
    int pendingHead  = 0;
    int pendingTail  = 0;
    int pendingCount = 0;

    auto drainOneExecutor = [&]() {
      GpuExecutor* oldest = pendingExecutors[pendingHead];
      ScopedNvtxRange waitRange("Wait for D2H copy");
      cudaCheckError(cudaEventSynchronize(oldest->copyDoneEvent.event()));
      waitRange.pop();

      ScopedNvtxRange accumRange("Accumulate mini-batch counts");
      accumulate(*oldest);
      accumRange.pop();

      pendingHead = (pendingHead + 1) % executorsPerRunner;
      --pendingCount;
    };

    int localMiniBatchCount = 0;

    while (true) {
      const int miniBatchIdx = nextMiniBatchIdx.fetch_add(1, std::memory_order_relaxed);
      if (miniBatchIdx >= totalNumMiniBatches) break;

      const int miniBatchStart = miniBatchIdx * effectiveMiniBatchSize;
      if (miniBatchStart >= numPairs) break;

      if (pendingCount == executorsPerRunner) {
        drainOneExecutor();
      }

      GpuExecutor* executor = executors[pendingTail];

      if (upstreamReadyEvent != nullptr && localMiniBatchCount < executorsPerRunner) {
        cudaCheckError(cudaStreamWaitEvent(executor->stream(), upstreamReadyEvent, 0));
        cudaCheckError(cudaStreamWaitEvent(executor->recursiveStream.stream(), upstreamReadyEvent, 0));
        cudaCheckError(cudaStreamWaitEvent(executor->postRecursionStream.stream(), upstreamReadyEvent, 0));
      }
      ++localMiniBatchCount;

      prepareMiniBatchOnCPU(*executor, ctx, queriesHost, leafSubpatterns, miniBatchStart, effectiveMiniBatchSize);

      ScopedNvtxRange launchRange("GPU launch mini-batch " + std::to_string(miniBatchIdx));
      uploadAndLaunchMiniBatch(*executor, ctx, targetsDevice, queriesDevice, leafSubpatterns, algorithm);
      initiateCountsOnlyCopyToHost(*executor);
      launchRange.pop();

      pendingExecutors[pendingTail] = executor;
      pendingTail = (pendingTail + 1) % executorsPerRunner;
      ++pendingCount;
    }

    while (pendingCount > 0) {
      drainOneExecutor();
    }
  } catch (...) {
    exceptionPtr = std::current_exception();
  }
}

/**
 * @brief Inline runner with thread-local executors and deferred accumulation.
 *
 * Uses N-buffering with deferred accumulation: only blocks when all
 * executors are in-flight. This maximizes GPU utilization by keeping mini-batches
 * queued while waiting for D2H copies.
 *
 * @param executorsPerRunner Number of executors assigned to this runner (2-8)
 * @param deviceId GPU device ID to use for this worker
 */
void runnerWorkerInline(int                               workerIdx,
                        const ThreadWorkerContext&        ctx,
                        MoleculesDevice&                  targetsDevice,
                        const MoleculesDevice&            queriesDevice,
                        const MoleculesHost&              queriesHost,
                        const LeafSubpatterns&            leafSubpatterns,
                        SubstructSearchResults&           results,
                        std::mutex&                       resultsMutex,
                        SubstructAlgorithm                algorithm,
                        cudaEvent_t                       upstreamReadyEvent,
                        std::atomic<int>&                 nextMiniBatchIdx,
                        int                               totalNumMiniBatches,
                        int                               effectiveMiniBatchSize,
                        int                               deviceId,
                        std::vector<GpuExecutor*>         executors,
                        std::exception_ptr&               exceptionPtr,
                        RDKitFallbackQueue*               fallbackQueue) {
  try {
    FallbackQueueProducerGuard producerGuard(fallbackQueue);
    ScopedNvtxRange workerRange("runnerWorkerInline " + std::to_string(workerIdx) + " GPU" + std::to_string(deviceId));
    const WithDevice setDevice(deviceId);

    const int executorsPerRunner = static_cast<int>(executors.size());
    const int numPairs = ctx.numTargets * ctx.numQueries;

    std::array<GpuExecutor*, kMaxExecutorsPerRunner> pendingExecutors{};
    int pendingHead  = 0;
    int pendingTail  = 0;
    int pendingCount = 0;

    auto drainOneExecutor = [&]() {
      GpuExecutor* oldest = pendingExecutors[pendingHead];
      ScopedNvtxRange waitRange("Wait for D2H copy");
      cudaCheckError(cudaEventSynchronize(oldest->copyDoneEvent.event()));
      waitRange.pop();

      ScopedNvtxRange accumRange("Accumulate mini-batch");
      accumulateMiniBatchResults(*oldest, ctx, results, resultsMutex, fallbackQueue);
      accumRange.pop();

      pendingHead = (pendingHead + 1) % executorsPerRunner;
      --pendingCount;
    };

    int localMiniBatchCount = 0;

    while (true) {
      const int miniBatchIdx = nextMiniBatchIdx.fetch_add(1, std::memory_order_relaxed);
      if (miniBatchIdx >= totalNumMiniBatches) break;

      const int miniBatchStart = miniBatchIdx * effectiveMiniBatchSize;
      if (miniBatchStart >= numPairs) break;

      if (pendingCount == executorsPerRunner) {
        drainOneExecutor();
      }

      GpuExecutor* executor = executors[pendingTail];

      if (upstreamReadyEvent != nullptr && localMiniBatchCount < executorsPerRunner) {
        cudaCheckError(cudaStreamWaitEvent(executor->stream(), upstreamReadyEvent, 0));
        cudaCheckError(cudaStreamWaitEvent(executor->recursiveStream.stream(), upstreamReadyEvent, 0));
        cudaCheckError(cudaStreamWaitEvent(executor->postRecursionStream.stream(), upstreamReadyEvent, 0));
      }
      ++localMiniBatchCount;

      prepareMiniBatchOnCPU(*executor, ctx, queriesHost, leafSubpatterns, miniBatchStart, effectiveMiniBatchSize);

      ScopedNvtxRange launchRange("GPU launch mini-batch " + std::to_string(miniBatchIdx));
      uploadAndLaunchMiniBatch(*executor, ctx, targetsDevice, queriesDevice, leafSubpatterns, algorithm);
      initiateResultsCopyToHost(*executor);
      launchRange.pop();

      pendingExecutors[pendingTail] = executor;
      pendingTail = (pendingTail + 1) % executorsPerRunner;
      ++pendingCount;
    }

    while (pendingCount > 0) {
      drainOneExecutor();
    }
  } catch (...) {
    exceptionPtr = std::current_exception();
  }
}

/**
 * @brief Boolean output variant of runnerWorkerInline.
 *
 * Optimized for hasSubstructMatch: uses countOnly mode to skip match index
 * storage/transfer, only copies match counts, and directly populates boolean results.
 */
void runnerWorkerInlineBoolean(int                               workerIdx,
                               const ThreadWorkerContext&        ctx,
                               MoleculesDevice&                  targetsDevice,
                               const MoleculesDevice&            queriesDevice,
                               const MoleculesHost&              queriesHost,
                               const LeafSubpatterns&            leafSubpatterns,
                               HasSubstructMatchResults&         results,
                               std::mutex&                       resultsMutex,
                               SubstructAlgorithm                algorithm,
                               cudaEvent_t                       upstreamReadyEvent,
                               std::atomic<int>&                 nextMiniBatchIdx,
                               int                               totalNumMiniBatches,
                               int                               effectiveMiniBatchSize,
                               int                               deviceId,
                               std::vector<GpuExecutor*>         executors,
                               std::exception_ptr&               exceptionPtr) {
  auto accumulate = [&](GpuExecutor& executor) {
    accumulateMiniBatchResultsBoolean(executor, ctx, results, resultsMutex);
  };
  runnerWorkerInlineCountOnly(workerIdx, ctx, targetsDevice, queriesDevice, queriesHost, leafSubpatterns,
                              algorithm, upstreamReadyEvent, nextMiniBatchIdx, totalNumMiniBatches,
                              effectiveMiniBatchSize, deviceId, std::move(executors),
                              accumulate, exceptionPtr);
}

void runnerWorkerInlineCounts(int                               workerIdx,
                              const ThreadWorkerContext&        ctx,
                              MoleculesDevice&                  targetsDevice,
                              const MoleculesDevice&            queriesDevice,
                              const MoleculesHost&              queriesHost,
                              const LeafSubpatterns&            leafSubpatterns,
                              std::vector<int>&                 counts,
                              std::mutex&                       resultsMutex,
                              SubstructAlgorithm                algorithm,
                              cudaEvent_t                       upstreamReadyEvent,
                              std::atomic<int>&                 nextMiniBatchIdx,
                              int                               totalNumMiniBatches,
                              int                               effectiveMiniBatchSize,
                              int                               deviceId,
                              std::vector<GpuExecutor*>         executors,
                              std::exception_ptr&               exceptionPtr) {
  auto accumulate = [&](GpuExecutor& executor) {
    accumulateMiniBatchResultsCounts(executor, ctx, counts, resultsMutex);
  };
  runnerWorkerInlineCountOnly(workerIdx, ctx, targetsDevice, queriesDevice, queriesHost, leafSubpatterns,
                              algorithm, upstreamReadyEvent, nextMiniBatchIdx, totalNumMiniBatches,
                              effectiveMiniBatchSize, deviceId, std::move(executors),
                              accumulate, exceptionPtr);
}

}  // namespace

// =============================================================================
// Main API
// =============================================================================

namespace {

void runMacroBatchedSubstructSearch(const std::vector<const RDKit::ROMol*>& gpuTargets,
                                    const std::vector<int>&                gpuTargetIndices,
                                    const std::vector<unsigned int>&       gpuTargetAtomCounts,
                                    const MoleculesHost&                   queriesHost,
                                    const MoleculesDevice&                 queriesDevice,
                                    const LeafSubpatterns&                 leafSubpatterns,
                                    SubstructSearchResults&                results,
                                    SubstructAlgorithm                     algorithm,
                                    cudaStream_t                           stream,
                                    const SubstructSearchConfig&           config,
                                    const std::vector<int>&                querySortOrder,
                                    int                                    effectivePreprocessingThreads,
                                    RDKitFallbackQueue*                    fallbackQueue,
                                    HasSubstructMatchResults*              boolResults,
                                    std::vector<int>*                      countResults) {
  const bool countOnly = (boolResults != nullptr) || (countResults != nullptr);
  const char* rangeLabel = boolResults
      ? "runMacroBatchedHasSubstructMatch"
      : (countResults ? "runMacroBatchedCountSubstructMatches" : "runMacroBatchedSubstructSearch");
  ScopedNvtxRange e2eRange(rangeLabel);

  const int numGpuTargets = static_cast<int>(gpuTargets.size());
  const int numQueries    = static_cast<int>(queriesHost.numMolecules());
  if (numGpuTargets == 0 || numQueries == 0) {
    return;
  }

  // Determine GPU list: empty gpuIds = current device only
  std::vector<int> gpuIds = config.gpuIds;
  int currentDevice = 0;
  cudaCheckError(cudaGetDevice(&currentDevice));
  if (gpuIds.empty()) {
    gpuIds.push_back(currentDevice);
  }
  const int numGpus = static_cast<int>(gpuIds.size());

  // Macro partitioning (targets in original order, no target straddles macros)
  const int macroMinibatches = std::max(1, config.macroBatchMinibatches);
  const int64_t macroPairsTarget = static_cast<int64_t>(config.batchSize) * static_cast<int64_t>(macroMinibatches);
  int targetsPerMacro = numGpuTargets;
  if (macroMinibatches > 1) {
    targetsPerMacro = static_cast<int>((macroPairsTarget + numQueries - 1) / numQueries);
    targetsPerMacro = std::max(1, std::min(targetsPerMacro, numGpuTargets));
  }
  const int numMacros = (numGpuTargets + targetsPerMacro - 1) / targetsPerMacro;

  // Precompute query metadata once (query preprocessing is kept up-front).
  const int precomputedSize      = static_cast<int>(leafSubpatterns.perQueryPatterns.size());
  const int perQueryMaxDepthSize = static_cast<int>(leafSubpatterns.perQueryMaxDepth.size());

  std::vector<int>     queryAtomCountsHost(numQueries);
  std::vector<int>     queryDepthsHost(numQueries);
  std::vector<int>     queryMaxDepthsHost(numQueries);
  std::vector<int8_t>  queryHasPatternsHost(numQueries);

  int maxQueryAtoms = 0;
  int maxDepthSeen  = 0;

#pragma omp parallel num_threads(effectivePreprocessingThreads) reduction(max:maxQueryAtoms, maxDepthSeen)
  {
#pragma omp for nowait
    for (int q = 0; q < numQueries; ++q) {
      const int atomStart    = queriesHost.batchAtomStarts[q];
      const int atomEnd      = queriesHost.batchAtomStarts[q + 1];
      const int atomCount    = atomEnd - atomStart;
      queryAtomCountsHost[q] = atomCount;

      const int depth        = getQueryRecursionDepth(queriesHost, q);
      queryDepthsHost[q]     = depth;
      maxDepthSeen           = std::max(maxDepthSeen, depth);

      const int maxDepth     = (q < perQueryMaxDepthSize) ? leafSubpatterns.perQueryMaxDepth[q] : 0;
      queryMaxDepthsHost[q]  = maxDepth;

      const bool hasPatterns = (q < precomputedSize) &&
                               (maxDepth > 0 || !leafSubpatterns.perQueryPatterns[q][0].empty());
      queryHasPatternsHost[q] = hasPatterns ? 1 : 0;

      maxQueryAtoms          = std::max(maxQueryAtoms, atomCount);
    }
  }

  if (maxDepthSeen > kMaxRecursionDepth) {
    throw std::runtime_error("Recursive SMARTS depth " + std::to_string(maxDepthSeen) +
                             " exceeds maximum supported depth of " +
                             std::to_string(kMaxRecursionDepth));
  }

  // Precompute max patterns per depth across all queries for pinned buffer sizing.
  int maxPatternsPerDepth = 256;
  for (int d = 0; d <= kMaxRecursionDepth; ++d) {
    int patternsAtThisDepth = 0;
    for (size_t q = 0; q < leafSubpatterns.perQueryPatterns.size(); ++q) {
      patternsAtThisDepth += static_cast<int>(leafSubpatterns.perQueryPatterns[q][d].size());
    }
    maxPatternsPerDepth = std::max(maxPatternsPerDepth, patternsAtThisDepth);
  }

  // Use global maximum target atoms for sizing pinned buffers once.
  int globalMaxTargetAtoms = 0;
  for (size_t i = 0; i < gpuTargetAtomCounts.size(); ++i) {
    globalMaxTargetAtoms = std::max(globalMaxTargetAtoms, static_cast<int>(gpuTargetAtomCounts[i]));
  }

  // Pinned buffers are sized for the worst-case mini-batch size within a macro.
  const int maxPairsInMacro = std::min(numGpuTargets, targetsPerMacro) * numQueries;
  const int pinnedMiniBatchSize = std::min(config.batchSize, maxPairsInMacro);

  size_t maxMatchIndicesPerMiniBatch;
  if (countOnly) {
    maxMatchIndicesPerMiniBatch = 0;
  } else if (config.maxMatches > 0) {
    maxMatchIndicesPerMiniBatch = static_cast<size_t>(pinnedMiniBatchSize) * config.maxMatches * maxQueryAtoms;
  } else {
    maxMatchIndicesPerMiniBatch = static_cast<size_t>(pinnedMiniBatchSize) * globalMaxTargetAtoms * maxQueryAtoms;
  }

  // Determine runners and slots using the same logic as getSubstructMatchesImpl, but based on worst-case macro.
  const int requestedNumRunners = config.workerThreads;
  const int numPairsWorstCase = std::min(numGpuTargets, targetsPerMacro) * numQueries;
  const int effectiveMiniBatchSizeWorstCase = std::min(config.batchSize, numPairsWorstCase);
  const int totalNumMiniBatchesWorstCase = (numPairsWorstCase + effectiveMiniBatchSizeWorstCase - 1) / effectiveMiniBatchSizeWorstCase;

  const int runnersPerGpu = std::max(1, requestedNumRunners);
  const int totalRunners  = std::min(runnersPerGpu * numGpus, totalNumMiniBatchesWorstCase);
  const int numRunners    = totalRunners;

  int executorsPerRunner;
  if (config.executorsPerRunner == -1) {
    executorsPerRunner = (numRunners == 1) ? 3 : 2;
  } else if (config.executorsPerRunner < 1 || config.executorsPerRunner > kMaxExecutorsPerRunner) {
    throw std::invalid_argument("executorsPerRunner must be -1 (auto) or between 1 and " +
                                std::to_string(kMaxExecutorsPerRunner));
  } else {
    executorsPerRunner = config.executorsPerRunner;
  }

  std::vector<int> workersPerGpu(numGpus, numRunners / numGpus);
  for (int i = 0; i < numRunners % numGpus; ++i) {
    workersPerGpu[i]++;
  }

  // Compute pinned memory footprint (worst-case macro) and allocate once.
  const int totalExecutors = numRunners * executorsPerRunner;
  const size_t perExecutorSize = ConsolidatedPinnedBuffer::computeSize(
      pinnedMiniBatchSize, static_cast<int>(maxMatchIndicesPerMiniBatch), maxPatternsPerDepth);
  const size_t totalPinnedBytes = static_cast<size_t>(totalExecutors) * perExecutorSize;

  const long pages    = sysconf(_SC_PHYS_PAGES);
  const long pageSize = sysconf(_SC_PAGE_SIZE);
  const size_t systemRam  = static_cast<size_t>(pages) * static_cast<size_t>(pageSize);
  const size_t maxAllowed = systemRam / 4;
  if (totalPinnedBytes > maxAllowed) {
    throw std::runtime_error(
        "Substructure search would require " + std::to_string(totalPinnedBytes / (1024 * 1024)) +
        " MB of pinned memory, exceeding 1/4 of system RAM (" +
        std::to_string(maxAllowed / (1024 * 1024)) + " MB). "
        "Reduce workerThreads, executorsPerRunner, or batchSize.");
  }

  ScopedNvtxRange allocRange("CPU: Allocate all pinned buffers (macro)");
  char* megaBuffer = nullptr;
  cudaCheckError(cudaMallocHost(&megaBuffer, totalPinnedBytes));

  std::vector<ConsolidatedPinnedBuffer> pinnedBuffers(totalExecutors);
  for (int i = 0; i < totalExecutors; ++i) {
    char* executorPtr = megaBuffer + i * perExecutorSize;
    pinnedBuffers[i].assignExternal(executorPtr, pinnedMiniBatchSize, static_cast<int>(maxMatchIndicesPerMiniBatch), maxPatternsPerDepth);
  }
  allocRange.pop();

  struct MacroData {
    MoleculesHost    targetsHost;
    std::vector<int> sortedToOriginal;   ///< sorted target idx -> original target idx (full input)
    ThreadWorkerContext ctx;             ///< fully-populated context for this macro
    int totalNumMiniBatches    = 0;
    int effectiveMiniBatchSize = 0;
  };

  const int preprocessingThreads = effectivePreprocessingThreads;

  auto initializeMacroContextQueries = [&](ThreadWorkerContext& ctx) {
    ctx.numQueries     = numQueries;
    ctx.querySortOrder = querySortOrder.empty() ? nullptr : &querySortOrder;
    ctx.maxMatches     = config.maxMatches;
    ctx.countOnly      = countOnly;

    ctx.queryAtomCounts.resize(static_cast<size_t>(numQueries * 1.5));
    ctx.queryDepths.resize(numQueries);
    ctx.queryMaxDepths.resize(numQueries);
    ctx.queryHasPatterns.resize(numQueries);

    for (int q = 0; q < numQueries; ++q) {
      ctx.queryAtomCounts[q]  = queryAtomCountsHost[q];
      ctx.queryDepths[q]      = queryDepthsHost[q];
      ctx.queryMaxDepths[q]   = queryMaxDepthsHost[q];
      ctx.queryHasPatterns[q] = queryHasPatternsHost[q];
    }
  };

  auto buildMacro = [&](int macroIdx, MacroData& out) {
    ScopedNvtxRange macroBuildRange("CPU: Build macro " + std::to_string(macroIdx));
    const int t0 = macroIdx * targetsPerMacro;
    const int t1 = std::min(t0 + targetsPerMacro, numGpuTargets);
    const int n  = t1 - t0;

    std::vector<const RDKit::ROMol*> macroTargets;
    std::vector<int> macroBuildOrder;
    {
      ScopedNvtxRange setupRange("CPU: Macro setup and sort");
      macroTargets.reserve(static_cast<size_t>(n));
      for (int i = 0; i < n; ++i) {
        macroTargets.push_back(gpuTargets[t0 + i]);
      }

      if (config.presort) {
        macroBuildOrder.resize(n);
        std::iota(macroBuildOrder.begin(), macroBuildOrder.end(), 0);
        std::sort(macroBuildOrder.begin(), macroBuildOrder.end(), [&](int a, int b) {
          return gpuTargetAtomCounts[t0 + a] > gpuTargetAtomCounts[t0 + b];
        });
      }
    }

    buildTargetBatchParallelInto(out.targetsHost, preprocessingThreads, macroTargets, macroBuildOrder);

    {
      ScopedNvtxRange postRange("CPU: Macro context setup");
      out.sortedToOriginal.resize(static_cast<size_t>(n));
      for (int sortedIdx = 0; sortedIdx < n; ++sortedIdx) {
        const int macroLocalIdx = macroBuildOrder.empty() ? sortedIdx : macroBuildOrder[sortedIdx];
        out.sortedToOriginal[sortedIdx] = gpuTargetIndices[t0 + macroLocalIdx];
      }

      out.ctx.numTargets      = n;
      out.ctx.targetSortOrder = &out.sortedToOriginal;

      out.ctx.targetAtomCounts.resize(n);
      int localMaxTargetAtoms = 0;
      int localMaxBondsPerAtom = 0;
      for (int t = 0; t < n; ++t) {
        const int atomStart = out.targetsHost.batchAtomStarts[t];
        const int atomEnd   = out.targetsHost.batchAtomStarts[t + 1];
        const int atoms     = atomEnd - atomStart;
        out.ctx.targetAtomCounts[t] = atoms;
        localMaxTargetAtoms = std::max(localMaxTargetAtoms, atoms);
        for (int a = atomStart; a < atomEnd; ++a) {
          localMaxBondsPerAtom = std::max(localMaxBondsPerAtom, static_cast<int>(out.targetsHost.targetAtomBonds[a].degree));
        }
      }
      out.ctx.maxTargetAtoms = localMaxTargetAtoms;
      out.ctx.maxQueryAtoms = maxQueryAtoms;
      out.ctx.maxBondsPerAtom = localMaxBondsPerAtom;
      out.ctx.templateConfig = selectTemplateConfig(localMaxTargetAtoms, maxQueryAtoms, localMaxBondsPerAtom);

      const int numPairs = n * numQueries;
      out.effectiveMiniBatchSize = std::min(config.batchSize, numPairs);
      out.totalNumMiniBatches    = (numPairs + out.effectiveMiniBatchSize - 1) / out.effectiveMiniBatchSize;
    }
  };

  // Global macro dispatch state.
  std::mutex              macroMutex;
  std::condition_variable macroCv;
  int                     macroEpoch = 0;
  bool                    shutdown   = false;
  MacroData*              currentMacro = nullptr;

  std::mutex              doneMutex;
  std::condition_variable doneCv;
  std::atomic<int>        gpusDone{0};

  std::atomic<int> nextMiniBatchIdx(0);

  // Use the fallback queue's mutex if available (ensures GPU batch accumulation
  // and fallback processing use the same mutex to avoid race conditions)
  std::mutex localResultsMutex;
  std::mutex& resultsMutex = fallbackQueue ? fallbackQueue->getResultsMutex() : localResultsMutex;

  std::vector<std::exception_ptr> exceptions(numRunners);

  ScopedNvtxRange launchRange("CPU: Launch GPU coordinators (macro)");
  std::vector<std::thread> gpuThreads;
  gpuThreads.reserve(numGpus);
  int activeGpus = 0;

  int executorOffset = 0;
  int workerIdOffset = 0;
  for (int g = 0; g < numGpus; ++g) {
    const int numWorkersThisGpu = workersPerGpu[g];
    if (numWorkersThisGpu == 0) {
      continue;
    }
    ++activeGpus;

    const int deviceId = gpuIds[g];
    const int startWorkerIdx = workerIdOffset;
    const int startExecutorIdx = executorOffset;
    const int numExecutorsThisGpu = numWorkersThisGpu * executorsPerRunner;
    workerIdOffset += numWorkersThisGpu;
    executorOffset += numExecutorsThisGpu;

    std::vector<ConsolidatedPinnedBuffer*> gpuBufferPtrs;
    gpuBufferPtrs.reserve(numExecutorsThisGpu);
    for (int i = 0; i < numExecutorsThisGpu; ++i) {
      gpuBufferPtrs.push_back(&pinnedBuffers[startExecutorIdx + i]);
    }

    gpuThreads.emplace_back([=, &macroMutex, &macroCv, &macroEpoch, &shutdown, &currentMacro,
                             &doneMutex, &doneCv, &gpusDone,
                             &queriesHost, &queriesDevice, &leafSubpatterns, &results, &resultsMutex,
                             &nextMiniBatchIdx, &exceptions]() mutable {
      try {
        ScopedNvtxRange coordRange("GPU" + std::to_string(deviceId) + " coordinator (macro)");
        const WithDevice setDevice(deviceId);

        // Executors must be declared before device objects so streams outlive async memory.
        std::vector<std::unique_ptr<GpuExecutor>> executors;
        executors.reserve(gpuBufferPtrs.size());
        for (size_t i = 0; i < gpuBufferPtrs.size(); ++i) {
          auto executor = std::make_unique<GpuExecutor>(startWorkerIdx * executorsPerRunner + static_cast<int>(i), deviceId);
          executor->bindPinnedBuffer(*gpuBufferPtrs[i]);
          executor->initializeForStream();
          executors.push_back(std::move(executor));
        }

        std::unique_ptr<MoleculesDevice> localTargets;
        std::unique_ptr<MoleculesDevice> localQueries;
        std::unique_ptr<LeafSubpatterns> localLeafPatterns;

        MoleculesDevice* targetsPtr = nullptr;
        const MoleculesDevice* queriesPtr = &queriesDevice;
        const LeafSubpatterns* leafPtr = &leafSubpatterns;

        if (deviceId != currentDevice) {
          localTargets = std::make_unique<MoleculesDevice>();
          localQueries = std::make_unique<MoleculesDevice>();
          localQueries->copyFromHost(queriesHost);
          localLeafPatterns = std::make_unique<LeafSubpatterns>();
          localLeafPatterns->buildAllPatterns(queriesHost);
          localLeafPatterns->syncToDevice(nullptr);
          targetsPtr = localTargets.get();
          queriesPtr = localQueries.get();
          leafPtr = localLeafPatterns.get();
        } else {
          localTargets = std::make_unique<MoleculesDevice>();
          targetsPtr = localTargets.get();
        }

        // Per-GPU worker dispatch
        std::mutex              localMutex;
        std::condition_variable localCv;
        int                     localEpoch = 0;
        bool                    localShutdown = false;
        const ThreadWorkerContext* localCtx = nullptr;
        int                     localTotalMiniBatches = 0;
        int                     localMiniBatchSize = 0;
        cudaEvent_t             localUpstreamEvent = nullptr;
        int                     localWorkersRemaining = 0;

        ScopedCudaEvent upstreamEventStorage;

        auto workerLoop = [&](int globalIdx, const std::vector<GpuExecutor*>& workerExecutors) {
          int seenEpoch = 0;
          while (true) {
            const ThreadWorkerContext* ctxPtr = nullptr;
            int totalMiniBatches = 0;
            int miniBatchSize = 0;
            cudaEvent_t upstreamEvent = nullptr;

            {
              ScopedNvtxRange waitRange("Worker " + std::to_string(globalIdx) + " wait for next batch");
              std::unique_lock<std::mutex> lock(localMutex);
              localCv.wait(lock, [&]() { return localEpoch != seenEpoch || localShutdown; });
              if (localShutdown) {
                return;
              }
              seenEpoch     = localEpoch;
              ctxPtr        = localCtx;
              totalMiniBatches  = localTotalMiniBatches;
              miniBatchSize     = localMiniBatchSize;
              upstreamEvent = localUpstreamEvent;
            }

            if (boolResults) {
              runnerWorkerInlineBoolean(globalIdx,
                                 std::cref(*ctxPtr),
                                 std::ref(*targetsPtr),
                                 std::cref(*queriesPtr),
                                 std::cref(queriesHost),
                                 std::cref(*leafPtr),
                                 std::ref(*boolResults),
                                 std::ref(resultsMutex),
                                 algorithm,
                                 upstreamEvent,
                                 std::ref(nextMiniBatchIdx),
                                 totalMiniBatches,
                                 miniBatchSize,
                                 deviceId,
                                 workerExecutors,
                                 std::ref(exceptions[globalIdx]));
            } else if (countResults) {
              runnerWorkerInlineCounts(globalIdx,
                                 std::cref(*ctxPtr),
                                 std::ref(*targetsPtr),
                                 std::cref(*queriesPtr),
                                 std::cref(queriesHost),
                                 std::cref(*leafPtr),
                                 std::ref(*countResults),
                                 std::ref(resultsMutex),
                                 algorithm,
                                 upstreamEvent,
                                 std::ref(nextMiniBatchIdx),
                                 totalMiniBatches,
                                 miniBatchSize,
                                 deviceId,
                                 workerExecutors,
                                 std::ref(exceptions[globalIdx]));
            } else {
              runnerWorkerInline(globalIdx,
                                 std::cref(*ctxPtr),
                                 std::ref(*targetsPtr),
                                 std::cref(*queriesPtr),
                                 std::cref(queriesHost),
                                 std::cref(*leafPtr),
                                 std::ref(results),
                                 std::ref(resultsMutex),
                                 algorithm,
                                 upstreamEvent,
                                 std::ref(nextMiniBatchIdx),
                                 totalMiniBatches,
                                 miniBatchSize,
                                 deviceId,
                                 workerExecutors,
                                 std::ref(exceptions[globalIdx]),
                                 fallbackQueue);
            }

            {
              std::lock_guard<std::mutex> lock(localMutex);
              --localWorkersRemaining;
              if (localWorkersRemaining == 0) {
                localCv.notify_all();
              }
            }
          }
        };

        std::vector<std::thread> workers;
        workers.reserve(numWorkersThisGpu);
        for (int w = 0; w < numWorkersThisGpu; ++w) {
          const int globalIdx = startWorkerIdx + w;
          std::vector<GpuExecutor*> workerExecutors;
          workerExecutors.reserve(executorsPerRunner);
          for (int s = 0; s < executorsPerRunner; ++s) {
            workerExecutors.push_back(executors[w * executorsPerRunner + s].get());
          }
          workers.emplace_back(workerLoop, globalIdx, workerExecutors);
        }

        int seenMacroEpoch = 0;
        while (true) {
          MacroData* macro = nullptr;
          int epoch = 0;

          {
            std::unique_lock<std::mutex> lock(macroMutex);
            macroCv.wait(lock, [&]() { return macroEpoch != seenMacroEpoch || shutdown; });
            if (shutdown) {
              break;
            }
            epoch = macroEpoch;
            macro = currentMacro;
            seenMacroEpoch = epoch;
          }

          // Copy macro targets to this GPU and produce an upstream event for slots to wait on.
          cudaStream_t copyStream = executors.front()->stream();
          targetsPtr->copyFromHost(macro->targetsHost, copyStream);
          cudaCheckError(cudaEventRecord(upstreamEventStorage.event(), copyStream));

          // Reset local worker counter and publish macro parameters to workers.
          {
            std::lock_guard<std::mutex> lock(localMutex);
            localCtx           = &macro->ctx;
            localTotalMiniBatches  = macro->totalNumMiniBatches;
            localMiniBatchSize     = macro->effectiveMiniBatchSize;
            localUpstreamEvent = upstreamEventStorage.event();
            localWorkersRemaining = numWorkersThisGpu;
            localEpoch = epoch;
          }
          localCv.notify_all();

          // Wait for all local workers to finish this macro.
          {
            std::unique_lock<std::mutex> lock(localMutex);
            localCv.wait(lock, [&]() { return localWorkersRemaining == 0; });
          }

          // Notify global done.
          gpusDone.fetch_add(1, std::memory_order_release);
          doneCv.notify_one();
        }

        // Shutdown local workers.
        {
          std::lock_guard<std::mutex> lock(localMutex);
          localShutdown = true;
        }
        localCv.notify_all();
        for (auto& w : workers) {
          w.join();
        }
      } catch (...) {
        exceptions[startWorkerIdx] = std::current_exception();
      }
    });
  }
  launchRange.pop();

  // Double-buffered macro build + run loop on the main thread.
  MacroData buffers[2];
  initializeMacroContextQueries(buffers[0].ctx);
  initializeMacroContextQueries(buffers[1].ctx);

  // Build first macro (blocking) so workers can start quickly.
  buildMacro(0, buffers[0]);

  for (int macroIdx = 0; macroIdx < numMacros; ++macroIdx) {
    MacroData& current = buffers[macroIdx % 2];

    // Reset global batch counter for this macro, then publish new macro epoch.
    nextMiniBatchIdx.store(0, std::memory_order_relaxed);
    gpusDone.store(0, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(macroMutex);
      currentMacro = &current;
      ++macroEpoch;
    }
    macroCv.notify_all();

    // Build next macro while GPUs are working on current.
    if (macroIdx + 1 < numMacros) {
      buildMacro(macroIdx + 1, buffers[(macroIdx + 1) % 2]);
    }

    // Process RDKit fallback work while waiting for GPUs to finish.
    // This hides fallback latency in the preprocessing thread's idle time.
    // Use OMP parallel to leverage preprocessing threads for fallback processing.
    if (fallbackQueue != nullptr && fallbackQueue->hasWork()) {
      ScopedNvtxRange fallbackWhileWaitingRange("Process RDKit fallback while waiting");
      
      // Use OMP threads to process fallback work until GPUs are done or queue is empty
      #pragma omp parallel num_threads(effectivePreprocessingThreads)
      {
        while (true) {
          // Lock-free check if GPUs are done
          if (gpusDone.load(std::memory_order_acquire) >= activeGpus) {
            break;
          }
          
          // Try to process one fallback entry
          if (!fallbackQueue->tryProcessOne()) {
            // Queue empty - exit this worker
            break;
          }
        }
      }
      
      // Ensure GPUs have finished (in case queue emptied before GPUs were done)
      {
        std::unique_lock<std::mutex> lock(doneMutex);
        doneCv.wait(lock, [&]() { return gpusDone.load(std::memory_order_acquire) >= activeGpus; });
      }
    } else {
      std::unique_lock<std::mutex> lock(doneMutex);
      doneCv.wait(lock, [&]() { return gpusDone.load(std::memory_order_acquire) >= activeGpus; });
    }
  }

  // Shut down coordinators.
  {
    std::lock_guard<std::mutex> lock(macroMutex);
    shutdown = true;
  }
  macroCv.notify_all();

  ScopedNvtxRange joinRange("CPU: Join GPU coordinators (macro)");
  for (auto& t : gpuThreads) {
    t.join();
  }
  joinRange.pop();

  cudaFreeHost(megaBuffer);

  for (const auto& ex : exceptions) {
    if (ex) {
      std::rethrow_exception(ex);
    }
  }

  cudaCheckError(cudaGetLastError());
}

}  // namespace

// =============================================================================
// Recursive SMARTS Preprocessing
// =============================================================================

void preprocessRecursiveSmartsBatchedWithEvents(SubstructTemplateConfig           templateConfig,
                                                const MoleculesDevice&            targetsDevice,
                                                const MoleculesHost&              queriesHost,
                                                const LeafSubpatterns&            leafSubpatterns,
                                                MiniBatchResultsDevice&           miniBatchResults,
                                                const int                         numQueries,
                                                const int                         miniBatchPairOffset,
                                                const int                         miniBatchSize,
                                                const SubstructAlgorithm          algorithm,
                                                cudaStream_t                      stream,
                                                RecursiveScratchBuffers&          scratch,
                                                std::vector<BatchedPatternEntry>& scratchPatternEntries,
                                                cudaEvent_t*                      depthEvents,
                                                int                               numDepthEvents) {
  ScopedNvtxRange processRecursiveRange("Process recursive mini-batch with events");
  
  // Configure kernels for max shared memory carveout (once per process)
  configureSubstructKernelsSharedMem();
  
  ScopedNvtxRange processRecursiveRangeSetup("Process recursive mini-batch setup");

  scratch.setStream(stream);

  std::vector<BatchedPatternEntry>& patternEntriesHost = scratchPatternEntries;
  patternEntriesHost.clear();

  const int firstQueryInMiniBatch = miniBatchPairOffset % numQueries;
  const int numUniqueQueries      = std::min(miniBatchSize, numQueries);
  const int recursivePatternsSize = static_cast<int>(queriesHost.recursivePatterns.size());

  int maxDepth = 0;
  for (int i = 0; i < numUniqueQueries; ++i) {
    const int queryIdx = (firstQueryInMiniBatch + i) % numQueries;

    if (queryIdx >= recursivePatternsSize) {
      continue;
    }

    const auto& recursiveInfo = queriesHost.recursivePatterns[queryIdx];
    if (recursiveInfo.empty()) {
      continue;
    }

    maxDepth = std::max(maxDepth, recursiveInfo.maxDepth);

    for (const auto& entry : recursiveInfo.patterns) {
      if (entry.queryMol == nullptr) {
        continue;
      }

      const int patternMolIdx = leafSubpatterns.getPatternIndex(queryIdx, entry.patternId);
      if (patternMolIdx < 0) {
        throw std::runtime_error("Pattern not found in pre-built LeafSubpatterns: queryIdx=" +
                                 std::to_string(queryIdx) + ", patternId=" + std::to_string(entry.patternId));
      }

      BatchedPatternEntry& batchEntry = patternEntriesHost.emplace_back();
      batchEntry.mainQueryIdx    = queryIdx;
      batchEntry.patternId       = entry.patternId;
      batchEntry.patternMolIdx   = patternMolIdx;
      batchEntry.depth           = entry.depth;
      batchEntry.localIdInParent = entry.localIdInParent;
    }
  }

  if (patternEntriesHost.empty()) {
    return;
  }

  const int firstTargetInMiniBatch = miniBatchPairOffset / numQueries;
  const int lastTargetInMiniBatch  = (miniBatchPairOffset + miniBatchSize - 1) / numQueries;
  const int numTargetsInMiniBatch  = lastTargetInMiniBatch - firstTargetInMiniBatch + 1;

  constexpr int gsiBuffersPerBlock = 2;

  const int maxPaintPairsPerSubBatch = std::max(miniBatchSize, 1024);
  processRecursiveRangeSetup.pop();

  for (int currentDepth = 0; currentDepth <= maxDepth; ++currentDepth) {
    ScopedNvtxRange depthRange("Process recursive depth level " + std::to_string(currentDepth));

    std::vector<BatchedPatternEntry> patternsAtDepth;
    for (const auto& entry : patternEntriesHost) {
      if (entry.depth == currentDepth) {
        patternsAtDepth.push_back(entry);
      }
    }

    if (patternsAtDepth.empty()) {
      if (currentDepth < numDepthEvents && depthEvents != nullptr) {
        cudaCheckError(cudaEventRecord(depthEvents[currentDepth], stream));
      }
      continue;
    }

    const size_t numPatterns = patternsAtDepth.size();
    const int patternsPerSubBatch = std::max(1, maxPaintPairsPerSubBatch / numTargetsInMiniBatch);

    for (size_t patternStart = 0; patternStart < numPatterns; patternStart += patternsPerSubBatch) {
      ScopedNvtxRange subBatchRange("Process sub-batch " + std::to_string(patternStart));
      
      const size_t patternEnd            = std::min(patternStart + patternsPerSubBatch, numPatterns);
      const size_t numPatternsInSubBatch = patternEnd - patternStart;
      const size_t numBlocksInSubBatch   = numTargetsInMiniBatch * numPatternsInSubBatch;

      ScopedNvtxRange prepareRange("CPU: Prepare pattern entries");
      const int bufferIdx = scratch.acquireBufferIndex();
      scratch.waitForBuffer(bufferIdx);
      scratch.ensureCapacity(bufferIdx, static_cast<int>(numPatternsInSubBatch));
      for (size_t i = 0; i < numPatternsInSubBatch; ++i) {
        scratch.patternsAtDepthHost[bufferIdx][i] = patternsAtDepth[patternStart + i];
      }
      prepareRange.pop();

      const int buffersPerBlock = gsiBuffersPerBlock;
      const size_t overflowNeeded = numBlocksInSubBatch * buffersPerBlock * kOverflowEntriesPerBuffer;

      if (scratch.overflow.size() < overflowNeeded) {
        scratch.overflow.resize(static_cast<size_t>(overflowNeeded * 1.5));
      }

      const size_t labelMatrixNeeded = numBlocksInSubBatch * kLabelMatrixWords;
      if (scratch.labelMatrixBuffer.size() < labelMatrixNeeded) {
        scratch.labelMatrixBuffer.resize(static_cast<size_t>(labelMatrixNeeded * 1.5));
      }

      if (scratch.patternEntries.size() < numPatternsInSubBatch) {
        scratch.patternEntries.resize(static_cast<size_t>(numPatternsInSubBatch * 1.5));
      }
      
      scratch.patternEntries.copyFromHost(scratch.patternsAtDepthHost[bufferIdx], numPatternsInSubBatch);
      scratch.recordCopy(bufferIdx, scratch.patternEntries.stream());

      const uint32_t* recursiveBitsForLabel = (currentDepth > 0) ? miniBatchResults.recursiveMatchBits() : nullptr;

      launchLabelMatrixPaintKernel(
        templateConfig,
        targetsDevice.view(),
        leafSubpatterns.view(),
        scratch.patternEntries.data(),
        static_cast<int>(numPatternsInSubBatch),
        numBlocksInSubBatch,
        numQueries,
        miniBatchPairOffset,
        miniBatchSize,
        scratch.labelMatrixBuffer.data(),
        firstTargetInMiniBatch,
        recursiveBitsForLabel,
        miniBatchResults.maxTargetAtoms(),
        stream);

      launchSubstructPaintKernel(
        templateConfig,
        algorithm,
        targetsDevice.view(),
        leafSubpatterns.view(),
        scratch.patternEntries.data(),
        static_cast<int>(numPatternsInSubBatch),
        numBlocksInSubBatch,
        miniBatchResults.recursiveMatchBits(),
        miniBatchResults.maxTargetAtoms(),
        numQueries,
        0, 0,
        miniBatchPairOffset,
        miniBatchSize,
        scratch.overflow.data(),
        scratch.overflow.data(),
        kOverflowEntriesPerBuffer,
        scratch.labelMatrixBuffer.data(),
        firstTargetInMiniBatch,
        stream);
    }

    if (currentDepth < numDepthEvents && depthEvents != nullptr) {
      cudaCheckError(cudaEventRecord(depthEvents[currentDepth], stream));
    }
  }

  cudaCheckError(cudaGetLastError());
}

namespace {

/**
 * @brief Process RDKit fallback queue.
 *
 * Runs all (target, query) pairs in the fallback queue using RDKit CPU implementation.
 * Called on main thread after GPU workers have been dispatched.
 *
 * @param reason Human-readable reason for fallback (shown in profiler)
 */
void processRDKitFallbackQueue(const std::vector<const RDKit::ROMol*>& targets,
                               const std::vector<const RDKit::ROMol*>& queries,
                               const std::vector<RDKitFallbackEntry>&  fallbackQueue,
                               SubstructSearchResults&                 results,
                               std::mutex&                             resultsMutex,
                               int                                     maxMatches,
                               const char*                             reason) {
  ScopedNvtxRange fallbackRange("RDKIT FALLBACK: " + std::string(reason) + 
                                " (" + std::to_string(fallbackQueue.size()) + " pairs)");

  for (const auto& entry : fallbackQueue) {
    const RDKit::ROMol* target = targets[entry.originalTargetIdx];
    const RDKit::ROMol* query  = queries[entry.originalQueryIdx];
    
    processWithRDKitFallback(target, query, entry.originalTargetIdx, entry.originalQueryIdx,
                             results, resultsMutex, maxMatches);
  }
}

/**
 * @brief Remove duplicate matches that differ only in atom enumeration order.
 *
 * Two matches are considered duplicates if they map query atoms to the same set
 * of target atoms, regardless of the ordering. For example, with query "CCC" on
 * cyclohexane, matches (0,1,2) and (2,1,0) would be considered duplicates since
 * they both involve target atoms {0,1,2}.
 *
 * This is a postprocessing step applied after all matches are collected.
 */
void uniquifyResults(SubstructSearchResults& results) {
  ScopedNvtxRange uniquifyRange("uniquifyResults");

  std::set<std::vector<int>> seenSorted;
  std::vector<std::vector<int>> uniqueMatches;
  std::vector<int> sortedMatch;

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

    if (uniqueMatches.size() < matchList.size()) {
      matchList = std::move(uniqueMatches);
    }
  }
}

}  // namespace

/**
 * @brief Compute effective thread counts using autoselect logic.
 *
 * When a config value is -1 (autoselect):
 * - preprocessingThreads: uses hardware_concurrency
 * - workerThreads (per GPU): min(4, hardware_concurrency / numGpus)
 */
void computeEffectiveThreadCounts(const SubstructSearchConfig& config,
                                  int                          numGpus,
                                  int&                         effectivePreprocessingThreads,
                                  int&                         effectiveWorkerThreads) {
  const int hwThreads = static_cast<int>(std::thread::hardware_concurrency());
  const int effectiveNumGpus = std::max(1, numGpus);

  effectivePreprocessingThreads = (config.preprocessingThreads == -1)
      ? hwThreads
      : std::max(1, config.preprocessingThreads);

  effectiveWorkerThreads = (config.workerThreads == -1)
      ? std::min(4, std::max(1, hwThreads / effectiveNumGpus))
      : std::max(1, config.workerThreads);
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

  std::vector<int> gpuIds = config.gpuIds;
  if (gpuIds.empty()) {
    int currentDevice = 0;
    cudaCheckError(cudaGetDevice(&currentDevice));
    gpuIds.push_back(currentDevice);
  }
  const int numGpus = static_cast<int>(gpuIds.size());

  int effectivePreprocessingThreads, effectiveWorkerThreads;
  computeEffectiveThreadCounts(config, numGpus,
                               effectivePreprocessingThreads,
                               effectiveWorkerThreads);

  ScopedNvtxRange overloadRange(
      "getSubstructMatches T=" + std::to_string(numTargets) +
      " Q=" + std::to_string(numQueries) +
      " batch=" + std::to_string(config.batchSize) +
      " prep=" + std::to_string(effectivePreprocessingThreads) +
      " workers=" + std::to_string(effectiveWorkerThreads) +
      " gpus=" + std::to_string(numGpus));

  SubstructSearchConfig effectiveConfig = config;
  effectiveConfig.preprocessingThreads = effectivePreprocessingThreads;
  effectiveConfig.workerThreads = effectiveWorkerThreads;
  effectiveConfig.gpuIds = gpuIds;

  ScopedNvtxRange preprocessRange("Preprocess molecules");
  std::vector<unsigned int> targetAtomCounts(numTargets);
  std::vector<unsigned int> queryAtomCounts(numQueries);
  std::vector<uint8_t> needsFallback(numTargets);

#pragma omp parallel num_threads(effectivePreprocessingThreads)
  {
#pragma omp for nowait
    for (int i = 0; i < numTargets; ++i) {
      targetAtomCounts[i] = targets[i]->getNumAtoms();
      needsFallback[i] = (targetAtomCounts[i] > kMaxTargetAtoms) || requiresRDKitFallback(targets[i]);
    }
#pragma omp for
    for (int i = 0; i < numQueries; ++i) {
      queryAtomCounts[i] = queries[i]->getNumAtoms();
    }
  }

  std::vector<RDKitFallbackEntry> fallbackTargets;
  std::vector<int>                gpuTargetIndices;
  std::vector<const RDKit::ROMol*> gpuTargets;
  int                             numFallbackTargets = 0;

  gpuTargetIndices.reserve(numTargets);
  gpuTargets.reserve(numTargets);

  for (int i = 0; i < numTargets; ++i) {
    if (needsFallback[i]) {
      ++numFallbackTargets;
      for (int q = 0; q < numQueries; ++q) {
        fallbackTargets.push_back({i, q});
      }
    } else {
      gpuTargetIndices.push_back(i);
      gpuTargets.push_back(targets[i]);
    }
  }
  preprocessRange.pop();
  
  if (numFallbackTargets > 0) {
    ScopedNvtxRange warnRange("WARNING: " + std::to_string(numFallbackTargets) + 
                              " targets will use RDKit fallback");
  }

  // Initialize results for all original targets
  results.resize(numTargets, numQueries);

  // If no GPU-processable targets, just run RDKit fallback (no queue needed)
  if (gpuTargets.empty()) {
    ScopedNvtxRange allFallbackRange("ALL TARGETS - full RDKit fallback");
    std::mutex resultsMutex;
    RDKitFallbackQueue fallbackQueue(&targets, &queries, &results, &resultsMutex, config.maxMatches, boolResults, countResults);
    fallbackQueue.enqueue(fallbackTargets);
    while (fallbackQueue.tryProcessOne()) {}
    return;
  }

  // Compute atom counts for GPU-processable targets
  std::vector<unsigned int> gpuTargetAtomCounts(gpuTargets.size());
  for (size_t i = 0; i < gpuTargets.size(); ++i) {
    gpuTargetAtomCounts[i] = targetAtomCounts[gpuTargetIndices[i]];
  }

  std::vector<int> querySortOrder;
  const int numGpuTargets = static_cast<int>(gpuTargets.size());

  if (config.presort) {
    ScopedNvtxRange sortRange("Compute sort ordering");

    querySortOrder.resize(numQueries);
    std::iota(querySortOrder.begin(), querySortOrder.end(), 0);

    std::sort(querySortOrder.begin(), querySortOrder.end(), [&](int a, int b) {
      return queryAtomCounts[a] > queryAtomCounts[b];
    });
  }

  ScopedNvtxRange buildRange2("Build host query data structures");
  MoleculesHost queriesHost = buildQueryBatchParallel(queries, querySortOrder, effectivePreprocessingThreads);
  buildRange2.pop();

  ScopedNvtxRange buildRange3("Build device query data structures");
  MoleculesDevice queriesDevice(stream);
  buildRange3.pop();

  ScopedNvtxRange buildRange4("Copy queries to device");
  queriesDevice.copyFromHost(queriesHost);
  buildRange4.pop();

  ScopedNvtxRange leafRange("Build LeafSubpatterns");
  LeafSubpatterns leafSubpatterns;
  leafSubpatterns.buildAllPatterns(queriesHost);
  leafSubpatterns.syncToDevice(stream);
  leafRange.pop();

  // Ensure queries and patterns are fully copied before workers start using them.
  // Workers use different streams, so we need an explicit sync here.
  cudaCheckError(cudaStreamSynchronize(stream));

  // Mutex shared between GPU batch accumulation and fallback queue processing
  std::mutex resultsMutex;
  
  // Create fallback queue to collect overflow from GPU processing.
  // Preprocessing threads will opportunistically process entries while waiting for GPUs.
  RDKitFallbackQueue fallbackQueue(&targets, &queries, &results, &resultsMutex, config.maxMatches, boolResults, countResults);

  // Enqueue oversized target fallbacks - processed opportunistically during macro batch loop
  if (!fallbackTargets.empty()) {
    ScopedNvtxRange enqueueRange("Enqueue oversized targets for opportunistic processing");
    fallbackQueue.enqueue(fallbackTargets);
  }

  // Macro-batch overlap:
  // Run target preprocessing (macro-batches) overlapped with persistent GPU worker threads.
  // Preprocessing threads will process RDKit fallback work while waiting for GPUs.
  runMacroBatchedSubstructSearch(gpuTargets,
                                gpuTargetIndices,
                                gpuTargetAtomCounts,
                                queriesHost,
                                queriesDevice,
                                leafSubpatterns,
                                results,
                                algorithm,
                                stream,
                                effectiveConfig,
                                querySortOrder,
                                effectivePreprocessingThreads,
                                &fallbackQueue,
                                boolResults,
                                countResults);

  // Process any remaining fallback entries after GPU work completes.
  // This handles overflow entries added during GPU processing and any
  // oversized targets not processed during the wait periods.
  while (fallbackQueue.tryProcessOne()) {}

  if (!boolResults && config.uniquify) {
    uniquifyResults(results);
  }
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
  SubstructSearchConfig countConfig = config;
  countConfig.maxMatches = 0;

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

  ScopedNvtxRange overloadRange(
      "hasSubstructMatch T=" + std::to_string(numTargets) +
      " Q=" + std::to_string(numQueries));

  results.resize(numTargets, numQueries);

  if (numTargets == 0 || numQueries == 0) {
    return;
  }

  SubstructSearchConfig hasMatchConfig = config;
  hasMatchConfig.maxMatches = 1;

  SubstructSearchResults matchResults;
  getSubstructMatchesImpl(targets, queries, matchResults, algorithm, stream, hasMatchConfig, &results, nullptr);

  for (auto& [pairIdx, matches] : matchResults.matches) {
    if (!matches.empty()) {
      results.hasMatch[pairIdx] = 1;
    }
  }
}

}  // namespace nvMolKit

