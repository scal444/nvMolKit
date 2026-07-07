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

#include "src/mcs/mcs_search.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "src/mcs/fmcs_cuda/fmcs.cuh"
#include "src/mcs/mcs_compile_flags.h"
#include "src/mcs/mcs_rdkit_adapter.h"
#include "src/utils/device.h"
#include "src/utils/nvtx.h"

namespace nvMolKit {
namespace {

using mcs::fmcs::LabeledGraph;

constexpr int kMaxMCSExecutorsPerRunner = 8;

struct PreparedGpuPair {
  size_t       resultIdx = 0;
  size_t       molIdxA   = 0;
  size_t       molIdxB   = 0;
  LabeledGraph graphA;
  LabeledGraph graphB;
};

void checkCuda(cudaError_t err, const char* context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("MCS dispatch CUDA error at ") + context + ": " + cudaGetErrorString(err));
  }
}

std::vector<int> resolveGpuIds(const MCSParameters& params) {
  std::vector<int> gpuIds = params.gpuIds;
  if (gpuIds.empty()) {
    int currentDevice = 0;
    checkCuda(cudaGetDevice(&currentDevice), "cudaGetDevice");
    gpuIds.push_back(currentDevice);
  }
  return gpuIds;
}

void computeEffectiveThreadCounts(const MCSParameters& params,
                                  int                  numGpus,
                                  int&                 effectivePreprocessingThreads,
                                  int&                 effectiveWorkerThreads) {
  const int hwThreads        = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  const int effectiveNumGpus = std::max(1, numGpus);

  effectivePreprocessingThreads =
    (params.preprocessingThreads == -1) ? hwThreads : std::max(1, params.preprocessingThreads);
  effectiveWorkerThreads = (params.workerThreads == -1) ? std::min(4, std::max(1, hwThreads / effectiveNumGpus)) :
                                                          std::max(1, params.workerThreads);
}

int effectiveExecutorsPerRunner(const MCSParameters& params, int totalRunners, cudaStream_t stream) {
  if (params.executorsPerRunner == -1) {
    return stream != nullptr ? 1 : (totalRunners == 1 ? 2 : 1);
  }
  if (params.executorsPerRunner < 1 || params.executorsPerRunner > kMaxMCSExecutorsPerRunner) {
    throw std::invalid_argument("MCS executorsPerRunner must be -1 (auto) or between 1 and " +
                                std::to_string(kMaxMCSExecutorsPerRunner));
  }
  return params.executorsPerRunner;
}

std::vector<PreparedGpuPair> prepareGpuPairs(const std::vector<const RDKit::ROMol*>& mols,
                                             const std::vector<MCSPair>&             pairs,
                                             const MCSParameters&                    params,
                                             int                                     preprocessingThreads,
                                             std::vector<MCSResult>&                 results) {
  std::vector<std::unique_ptr<PreparedGpuPair>> prepared(pairs.size());
  if (pairs.empty()) {
    return {};
  }

  ScopedNvtxRange prepareRange("CPU: Prepare GPU pairs P=" + std::to_string(pairs.size()));

  std::atomic<size_t> nextPair{0};
  std::atomic<bool>   abort{false};
  std::exception_ptr  firstException;
  std::mutex          exceptionMutex;

  auto setException = [&](std::exception_ptr ex) {
    std::lock_guard<std::mutex> lock(exceptionMutex);
    if (!firstException) {
      firstException = ex;
    }
    abort.store(true, std::memory_order_release);
  };

  auto prepareOne = [&](size_t i) {
    const auto [idxA, idxB] = pairs[i];
    if (idxA >= mols.size() || idxB >= mols.size()) {
      throw std::runtime_error("findMCSBatch pair index out of range");
    }
    const RDKit::ROMol* molA = mols[idxA];
    const RDKit::ROMol* molB = mols[idxB];
    if (molA == nullptr || molB == nullptr) {
      throw std::runtime_error("findMCSBatch pair references a null molecule pointer");
    }

    std::string fallbackReason;
    if (mcs_detail::shouldFallbackToRDKit(*molA, *molB, params, fallbackReason)) {
      if (!params.allowRDKitFallback) {
        throw std::runtime_error("RDKit fallback is disabled: " + fallbackReason);
      }
      results[i] = mcs_detail::runRDKitFallback(*molA, *molB, params);
      return;
    }

    auto graphs = mcs_detail::buildLabeledGraphPair(*molA, *molB, params);

    auto item       = std::make_unique<PreparedGpuPair>();
    item->resultIdx = i;
    item->molIdxA   = idxA;
    item->molIdxB   = idxB;
    item->graphA    = std::move(graphs.graphA);
    item->graphB    = std::move(graphs.graphB);
    prepared[i]     = std::move(item);
  };

  const int threadCount = std::min<int>(std::max(1, preprocessingThreads), static_cast<int>(pairs.size()));
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(threadCount));
  for (int t = 0; t < threadCount; ++t) {
    workers.emplace_back([&, t]() {
      ScopedNvtxRange threadRange("Prepare pairs thread " + std::to_string(t));
      while (!abort.load(std::memory_order_acquire)) {
        const size_t i = nextPair.fetch_add(1, std::memory_order_relaxed);
        if (i >= pairs.size()) {
          break;
        }
        try {
          prepareOne(i);
        } catch (...) {
          setException(std::current_exception());
          break;
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  if (firstException) {
    std::rethrow_exception(firstException);
  }

  std::vector<PreparedGpuPair> gpuPairs;
  gpuPairs.reserve(pairs.size());
  for (auto& item : prepared) {
    if (item) {
      gpuPairs.push_back(std::move(*item));
    }
  }
  return gpuPairs;
}

void runGpuPairs(std::vector<PreparedGpuPair>&           gpuPairs,
                 const std::vector<const RDKit::ROMol*>& mols,
                 const MCSParameters&                    params,
                 const std::vector<int>&                 gpuIds,
                 int                                     effectiveWorkerThreads,
                 int                                     effectiveExecutorsPerRunner,
                 cudaStream_t                            stream,
                 std::vector<MCSResult>&                 results) {
  if (gpuPairs.empty()) {
    return;
  }

  ScopedNvtxRange dispatchRange("Dispatch GPU pairs P=" + std::to_string(gpuPairs.size()), NvtxColor::kGreen);

  const int numGpus      = static_cast<int>(gpuIds.size());
  const int totalRunners = std::max(1, numGpus * std::max(1, effectiveWorkerThreads));
  if (stream != nullptr && totalRunners > 1) {
    throw std::invalid_argument("MCS multi-worker or multi-GPU dispatch does not support an external CUDA stream");
  }

  const size_t activeRunners = std::min<size_t>(static_cast<size_t>(totalRunners), gpuPairs.size());
  std::vector<std::vector<PreparedGpuPair>> runnerWork(activeRunners);
  for (size_t i = 0; i < gpuPairs.size(); ++i) {
    runnerWork[i % activeRunners].push_back(std::move(gpuPairs[i]));
  }

  auto runOneRunner = [&](size_t runnerIdx) {
    auto& work = runnerWork[runnerIdx];
    if (work.empty()) {
      return;
    }

    const int       deviceId = gpuIds[runnerIdx % static_cast<size_t>(numGpus)];
    ScopedNvtxRange runnerRange("MCS runner " + std::to_string(runnerIdx) + " GPU" + std::to_string(deviceId));
    std::unique_ptr<WithDevice> setDevice;
    if (stream == nullptr) {
      setDevice = std::make_unique<WithDevice>(deviceId);
    }

    std::vector<LabeledGraph> gpuGraphsA;
    std::vector<LabeledGraph> gpuGraphsB;
    std::vector<size_t>       resultIndices;
    std::vector<size_t>       molIndicesA;
    std::vector<size_t>       molIndicesB;
    gpuGraphsA.reserve(work.size());
    gpuGraphsB.reserve(work.size());
    resultIndices.reserve(work.size());
    molIndicesA.reserve(work.size());
    molIndicesB.reserve(work.size());
    for (auto& item : work) {
      resultIndices.push_back(item.resultIdx);
      molIndicesA.push_back(item.molIdxA);
      molIndicesB.push_back(item.molIdxB);
      gpuGraphsA.push_back(std::move(item.graphA));
      gpuGraphsB.push_back(std::move(item.graphB));
    }

    mcs::fmcs::Parameters fmcsParams;
    fmcsParams.batchSize = params.batchSize;
    fmcsParams.blockSize = params.blockSize;
    switch (params.scratchLocation) {
      case MCSScratchLocation::Shared:
        fmcsParams.scratchLocation = mcs::fmcs::FmcsScratchLocation::Shared;
        break;
      case MCSScratchLocation::Global:
        fmcsParams.scratchLocation = mcs::fmcs::FmcsScratchLocation::Global;
        break;
      case MCSScratchLocation::Auto:
        fmcsParams.scratchLocation = mcs::fmcs::FmcsScratchLocation::Auto;
        break;
    }
    fmcsParams.executorsPerRunner = effectiveExecutorsPerRunner;
    fmcsParams.matchVertexLabels  = mcs_detail::usesAtomLabels(params);
    fmcsParams.matchEdgeLabels    = mcs_detail::usesBondLabels(params);
    fmcsParams.completeRingsOnly =
      params.atomCompareParameters.completeRingsOnly || params.bondCompareParameters.completeRingsOnly;
    fmcsParams.timeoutMs = static_cast<float>(params.timeoutSeconds) * 1000.0f;

    std::vector<float>                      gpuTimesMs;
    std::vector<mcs::fmcs::ExecutionStats>  gpuTimingStats;
    std::vector<mcs::fmcs::ExecutionStats>  gpuStats;
    std::vector<float>*                     gpuTimesPtr       = nullptr;
    std::vector<mcs::fmcs::ExecutionStats>* gpuTimingStatsPtr = nullptr;
    std::vector<mcs::fmcs::ExecutionStats>* gpuStatsPtr       = nullptr;
    if constexpr (kMCSCollectTimingsEnabled) {
      if (params.collectTimings) {
        gpuTimesPtr       = &gpuTimesMs;
        gpuTimingStatsPtr = &gpuTimingStats;
      }
    }
    if constexpr (kMCSCollectStatsEnabled) {
      if (params.collectStats) {
        gpuStatsPtr = &gpuStats;
      }
    }
    auto            gpuResults = mcs::fmcs::findMCESfMCSBatchLabeled(gpuGraphsA,
                                                          gpuGraphsB,
                                                          fmcsParams,
                                                          gpuTimesPtr,
                                                          stream,
                                                          gpuStatsPtr,
                                                          gpuTimingStatsPtr);
    ScopedNvtxRange convertRange("Convert GPU results");
    for (size_t gpuIdx = 0; gpuIdx < gpuResults.size(); ++gpuIdx) {
      const size_t resultIdx    = resultIndices[gpuIdx];
      const size_t idxA         = molIndicesA[gpuIdx];
      const size_t idxB         = molIndicesB[gpuIdx];
      const float  gpuElapsedMs = gpuTimesPtr != nullptr && gpuIdx < gpuTimesMs.size() ? gpuTimesMs[gpuIdx] : 0.0f;
      if (gpuResults[gpuIdx].overflowed) {
        if (!params.allowRDKitFallback) {
          throw std::runtime_error("RDKit fallback is disabled: GPU MCS path overflowed");
        }
        auto fallback = mcs_detail::runRDKitFallback(*mols[idxA], *mols[idxB], params);
        fallback.elapsedMs += gpuElapsedMs;
        if constexpr (kMCSCollectStatsEnabled) {
          if (gpuStatsPtr != nullptr && gpuIdx < gpuStats.size()) {
            fallback.hasExecutionStats = true;
            fallback.executionStats    = mcs_detail::convertExecutionStats(gpuStats[gpuIdx]);
          }
        }
        if constexpr (kMCSCollectTimingsEnabled) {
          if (gpuTimingStatsPtr != nullptr && gpuIdx < gpuTimingStats.size()) {
            fallback.hasKernelTimings = true;
            fallback.kernelTimings    = mcs_detail::convertExecutionStats(gpuTimingStats[gpuIdx]);
          }
        }
        results[resultIdx] = std::move(fallback);
      } else {
        const mcs::fmcs::ExecutionStats* timingStatsPtr = nullptr;
        if constexpr (kMCSCollectTimingsEnabled) {
          timingStatsPtr =
            gpuTimingStatsPtr != nullptr && gpuIdx < gpuTimingStats.size() ? &gpuTimingStats[gpuIdx] : nullptr;
        }
        const mcs::fmcs::ExecutionStats* statsPtr = nullptr;
        if constexpr (kMCSCollectStatsEnabled) {
          statsPtr = gpuStatsPtr != nullptr && gpuIdx < gpuStats.size() ? &gpuStats[gpuIdx] : nullptr;
        }
        results[resultIdx] = mcs_detail::convertGpuResult(*mols[idxA],
                                                          *mols[idxB],
                                                          gpuResults[gpuIdx],
                                                          params,
                                                          gpuElapsedMs,
                                                          timingStatsPtr,
                                                          statsPtr);
      }
    }
  };

  if (activeRunners == 1) {
    runOneRunner(0);
    return;
  }

  std::atomic<bool>  abort{false};
  std::exception_ptr firstException;
  std::mutex         exceptionMutex;
  auto               setException = [&](std::exception_ptr ex) {
    std::lock_guard<std::mutex> lock(exceptionMutex);
    if (!firstException) {
      firstException = ex;
    }
    abort.store(true, std::memory_order_release);
  };

  std::vector<std::thread> runners;
  runners.reserve(activeRunners);
  for (size_t runnerIdx = 0; runnerIdx < activeRunners; ++runnerIdx) {
    runners.emplace_back([&, runnerIdx]() {
      if (abort.load(std::memory_order_acquire)) {
        return;
      }
      try {
        runOneRunner(runnerIdx);
      } catch (...) {
        setException(std::current_exception());
      }
    });
  }
  for (auto& runner : runners) {
    runner.join();
  }
  if (firstException) {
    std::rethrow_exception(firstException);
  }
}

}  // namespace

std::vector<MCSResult> findMCSBatch(const std::vector<const RDKit::ROMol*>& mols,
                                    const std::vector<MCSPair>&             pairs,
                                    cudaStream_t                            stream,
                                    const MCSParameters&                    params) {
  if (params.collectTimings && !kMCSCollectTimingsEnabled) {
    throw std::runtime_error("fMCS timing instrumentation is not instantiated in this build");
  }
  if (params.collectStats && !kMCSCollectStatsEnabled) {
    throw std::runtime_error("fMCS stat instrumentation is not instantiated in this build");
  }
  std::vector<MCSResult> results(pairs.size());
  if (pairs.empty())
    return results;

  const auto gpuIds                        = resolveGpuIds(params);
  int        effectivePreprocessingThreads = 1;
  int        effectiveWorkerThreads        = 1;
  computeEffectiveThreadCounts(params,
                               static_cast<int>(gpuIds.size()),
                               effectivePreprocessingThreads,
                               effectiveWorkerThreads);
  if (stream != nullptr) {
    if ((params.workerThreads != -1 && effectiveWorkerThreads > 1) || gpuIds.size() > 1) {
      throw std::invalid_argument("MCS multi-worker or multi-GPU dispatch does not support an external CUDA stream");
    }
    effectiveWorkerThreads = 1;
  }

  const int totalRunners       = std::max(1, static_cast<int>(gpuIds.size()) * effectiveWorkerThreads);
  const int effectiveExecutors = effectiveExecutorsPerRunner(params, totalRunners, stream);

  ScopedNvtxRange entryRange(
    "findMCSBatch P=" + std::to_string(pairs.size()) + " batch=" + std::to_string(params.batchSize) +
      " prep=" + std::to_string(effectivePreprocessingThreads) + " workers=" + std::to_string(effectiveWorkerThreads) +
      " gpus=" + std::to_string(gpuIds.size()) + " executors=" + std::to_string(effectiveExecutors),
    NvtxColor::kBlue);

  auto gpuPairs = prepareGpuPairs(mols, pairs, params, effectivePreprocessingThreads, results);
  runGpuPairs(gpuPairs, mols, params, gpuIds, effectiveWorkerThreads, effectiveExecutors, stream, results);

  return results;
}

std::vector<MCSResult> findMCSBatch(const std::vector<const RDKit::ROMol*>& molsA,
                                    const std::vector<const RDKit::ROMol*>& molsB,
                                    cudaStream_t                            stream,
                                    const MCSParameters&                    params) {
  if (molsA.size() != molsB.size()) {
    throw std::runtime_error("findMCSBatch requires equal-sized molecule arrays");
  }

  std::vector<const RDKit::ROMol*> mols;
  mols.reserve(molsA.size() + molsB.size());
  mols.insert(mols.end(), molsA.begin(), molsA.end());
  mols.insert(mols.end(), molsB.begin(), molsB.end());

  std::vector<MCSPair> pairs;
  pairs.reserve(molsA.size());
  for (size_t i = 0; i < molsA.size(); ++i) {
    pairs.emplace_back(i, molsA.size() + i);
  }
  return findMCSBatch(mols, pairs, stream, params);
}

std::vector<MCSResult> findMCSAllPairs(const std::vector<const RDKit::ROMol*>& mols,
                                       MCSAllPairsOptions                      options,
                                       cudaStream_t                            stream,
                                       const MCSParameters&                    params) {
  std::vector<MCSPair> pairs;
  const size_t         n = mols.size();
  if (options.upperTriangle) {
    const size_t maxPairs = options.includeDiagonal ? (n * (n + 1)) / 2 : (n > 1 ? (n * (n - 1)) / 2 : 0);
    pairs.reserve(maxPairs);
    for (size_t i = 0; i < n; ++i) {
      const size_t begin = options.includeDiagonal ? i : i + 1;
      for (size_t j = begin; j < n; ++j) {
        pairs.emplace_back(i, j);
      }
    }
  } else {
    pairs.reserve(options.includeDiagonal ? n * n : (n > 1 ? n * (n - 1) : 0));
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        if (!options.includeDiagonal && i == j) {
          continue;
        }
        pairs.emplace_back(i, j);
      }
    }
  }
  return findMCSBatch(mols, pairs, stream, params);
}

}  // namespace nvMolKit
