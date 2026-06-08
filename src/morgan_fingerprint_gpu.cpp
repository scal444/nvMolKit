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

#include "src/morgan_fingerprint_gpu.h"

#include <DataStructs/ExplicitBitVect.h>
#include <GraphMol/ROMol.h>
#include <omp.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

#include "src/data_structures/flat_bit_vect.h"
#include "src/gpu_scheduler/config.h"
#include "src/gpu_scheduler/cpu_fallback_queue.h"
#include "src/gpu_scheduler/pipeline.h"
#include "src/morgan_fingerprint_common.h"
#include "src/morgan_fingerprint_cpu.h"
#include "src/morgan_fingerprint_kernels.h"
#include "src/morgan_workload.h"
#include "src/utils/nvtx.h"
#include "src/utils/openmp_helpers.h"

namespace nvMolKit {

namespace {

constexpr int kDefaultGpuBatchSize = 2048;

template <int fpSize>
void extractResultsFromGPUBatch(const AsyncDeviceVector<FlatBitVect<fpSize>>& outputBuffer,
                                std::vector<FlatBitVect<fpSize>>&             result) {
  outputBuffer.copyToHost(result);
}

template <int nBits>
void populateResults(const std::vector<FlatBitVect<nBits>>&         resultsGpuVec,
                     std::vector<std::unique_ptr<ExplicitBitVect>>& results,
                     const int                                      numThreads) {
  detail::OpenMPExceptionRegistry exceptionRegistry;
#pragma omp parallel for default(none) num_threads(numThreads) shared(resultsGpuVec, results, exceptionRegistry)
  for (size_t vecIndex = 0; vecIndex < resultsGpuVec.size(); ++vecIndex) {
    try {
      const auto& tempResult    = resultsGpuVec[vecIndex];
      auto        resultBitVect = std::make_unique<ExplicitBitVect>(nBits);

      for (int i = 0; i < nBits; i++) {
        if (tempResult[i]) {
          resultBitVect->setBit(i);
        }
      }
      results[vecIndex] = std::move(resultBitVect);
    } catch (...) {
      exceptionRegistry.store(std::current_exception());
    }
  }
  exceptionRegistry.rethrow();
}

template <int fpSize>
FlatBitVect<fpSize> processSingleLargeMolecule(const RDKit::ROMol& mol, const std::uint32_t maxRadius) {
  auto                fingerprint = internal::getFingerprintImpl(mol, maxRadius, std::uint32_t(fpSize));
  FlatBitVect<fpSize> flatBitVect(false);
  for (int bitId = 0; bitId < fpSize; bitId++) {
    flatBitVect.setBit(bitId, fingerprint->getBit(bitId));
  }
  return flatBitVect;
}

/**
 * @brief Bucket every input mol by atom/bond budget into the four work lists.
 */
void bucketMolecules(const std::vector<const RDKit::ROMol*>& mols,
                     std::vector<int>&                       work32,
                     std::vector<int>&                       work64,
                     std::vector<int>&                       work128,
                     std::vector<int>&                       workLarge) {
  for (int i = 0; i < static_cast<int>(mols.size()); ++i) {
    const auto& mol = *mols[i];
    if (mol.getNumAtoms() < 32 && mol.getNumBonds() < 32) {
      work32.push_back(i);
    } else if (mol.getNumAtoms() < 64 && mol.getNumBonds() < 64) {
      work64.push_back(i);
    } else if (mol.getNumAtoms() < 128 && mol.getNumBonds() < 128) {
      work128.push_back(i);
    } else {
      workLarge.push_back(i);
    }
  }
}

inline int countMiniBatches(int numMols, int chunkSize) {
  if (numMols <= 0) {
    return 0;
  }
  return (numMols + chunkSize - 1) / chunkSize;
}

/**
 * @brief Run the pipeline iff there is any GPU work to do.
 *
 * Returns true if Pipeline::run() was invoked, false if there were no
 * GPU-eligible mini-batches.
 */
template <int fpSize> bool runPipelineIfAnyGpuWork(MorganInputs<fpSize>& inputs) {
  const int totalGpuUnits = inputs.numMiniBatches32 + inputs.numMiniBatches64 + inputs.numMiniBatches128;
  if (totalGpuUnits == 0) {
    return false;
  }
  gpu_scheduler::Config config;
  // Single GPU for now (matches pre-port behavior). Multi-GPU support requires
  // per-device output aggregation and is left as a follow-up.
  config.gpuIds = {inputs.primaryDeviceId};
  gpu_scheduler::Pipeline<MorganWorkload<fpSize>> pipeline(config, inputs);
  pipeline.run();
  return true;
}

template <int fpSize>
AsyncDeviceVector<FlatBitVect<fpSize>> computeFingerprintsCuImpl(const std::vector<const RDKit::ROMol*>& mols,
                                                                 const int                               maxRadius,
                                                                 const size_t dispatchChunkSizeInit,
                                                                 cudaStream_t stream = nullptr) {
  ScopedNvtxRange allocRange("MorganFPBatchAllocation");
  const size_t    numMols           = mols.size();
  auto            outputAccumulator = AsyncDeviceVector<FlatBitVect<fpSize>>(numMols, stream);
  cudaCheckError(cudaMemsetAsync(outputAccumulator.data(), 0, numMols * sizeof(FlatBitVect<fpSize>), stream));

  if (numMols == 0) {
    return outputAccumulator;
  }

  int currentDevice = 0;
  cudaCheckError(cudaGetDevice(&currentDevice));

  const int dispatchChunkSize = std::max(1, static_cast<int>(std::min(dispatchChunkSizeInit, numMols)));

  MorganInputs<fpSize> inputs;
  inputs.mols              = &mols;
  inputs.radius            = maxRadius;
  inputs.dispatchChunkSize = dispatchChunkSize;
  inputs.outputAccumulator = &outputAccumulator;
  inputs.primaryStream     = stream;
  inputs.primaryDeviceId   = currentDevice;

  bucketMolecules(mols, inputs.work32, inputs.work64, inputs.work128, inputs.workLarge);
  inputs.numMiniBatches32  = countMiniBatches(static_cast<int>(inputs.work32.size()), dispatchChunkSize);
  inputs.numMiniBatches64  = countMiniBatches(static_cast<int>(inputs.work64.size()), dispatchChunkSize);
  inputs.numMiniBatches128 = countMiniBatches(static_cast<int>(inputs.work128.size()), dispatchChunkSize);

  // The fallback queue handles the large-molecule CPU path. The handler grabs
  // the mol index, computes the fingerprint via the RDKit reference path, and
  // writes the result directly into the output accumulator.
  std::mutex outputMutex;
  auto       handler = [&](const int molIdx) {
    auto                        fingerprint = processSingleLargeMolecule<fpSize>(*mols[molIdx], maxRadius);
    std::lock_guard<std::mutex> lock(outputMutex);
    outputAccumulator.copyFromHost(&fingerprint, 1, 0, molIdx);
  };
  gpu_scheduler::CpuFallbackQueue<int> fallbackQueue(handler);
  inputs.fallbackQueue = &fallbackQueue;

  // Seed the fallback queue with all large-mol indices. Runner threads drain
  // entries opportunistically inside postprocess; whatever's left is drained
  // on the calling thread after run() returns.
  for (const int molIdx : inputs.workLarge) {
    fallbackQueue.enqueue(molIdx);
  }

  allocRange.pop();

  runPipelineIfAnyGpuWork<fpSize>(inputs);

  // Drain any fallback entries that weren't claimed during runner waits.
  // (When no runners ran at all - pure large-mol input - this is the only drain.)
  while (fallbackQueue.tryProcessOne()) {
  }

  return outputAccumulator;
}

std::vector<std::unique_ptr<ExplicitBitVect>> getFingerprintsCu(const std::vector<const RDKit::ROMol*>& mols,
                                                                const std::uint32_t                     maxRadius,
                                                                const std::uint64_t                     fpSize,
                                                                const size_t                            batchSize,
                                                                const int                               numThreads) {
  if (mols.empty()) {
    return {};
  }
  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  switch (fpSize) {
    case 4096: {
      auto                           gpuResult = computeFingerprintsCuImpl<4096>(mols, maxRadius, batchSize);
      std::vector<FlatBitVect<4096>> resultsGpuVec(gpuResult.size());
      std::vector<std::unique_ptr<ExplicitBitVect>> results(gpuResult.size());
      cudaCheckError(cudaDeviceSynchronize());
      extractResultsFromGPUBatch<4096>(gpuResult, resultsGpuVec);
      cudaCheckError(cudaDeviceSynchronize());
      populateResults<4096>(resultsGpuVec, results, numThreads);
      return results;
    }
    case 2048: {
      auto                           gpuResult = computeFingerprintsCuImpl<2048>(mols, maxRadius, batchSize);
      std::vector<FlatBitVect<2048>> resultsGpuVec(gpuResult.size());
      std::vector<std::unique_ptr<ExplicitBitVect>> results(gpuResult.size());
      cudaCheckError(cudaDeviceSynchronize());
      extractResultsFromGPUBatch<2048>(gpuResult, resultsGpuVec);
      cudaCheckError(cudaDeviceSynchronize());
      populateResults<2048>(resultsGpuVec, results, numThreads);
      return results;
    }
    case 1024: {
      auto                           gpuResult = computeFingerprintsCuImpl<1024>(mols, maxRadius, batchSize);
      std::vector<FlatBitVect<1024>> resultsGpuVec(gpuResult.size());
      std::vector<std::unique_ptr<ExplicitBitVect>> results(gpuResult.size());
      cudaCheckError(cudaDeviceSynchronize());
      extractResultsFromGPUBatch<1024>(gpuResult, resultsGpuVec);
      cudaCheckError(cudaDeviceSynchronize());
      populateResults<1024>(resultsGpuVec, results, numThreads);
      return results;
    }
    case 512: {
      auto                          gpuResult = computeFingerprintsCuImpl<512>(mols, maxRadius, batchSize);
      std::vector<FlatBitVect<512>> resultsGpuVec(gpuResult.size());
      std::vector<std::unique_ptr<ExplicitBitVect>> results(gpuResult.size());
      cudaCheckError(cudaDeviceSynchronize());
      extractResultsFromGPUBatch<512>(gpuResult, resultsGpuVec);
      cudaCheckError(cudaDeviceSynchronize());
      populateResults<512>(resultsGpuVec, results, numThreads);
      return results;
    }
    case 256: {
      auto                          gpuResult = computeFingerprintsCuImpl<256>(mols, maxRadius, batchSize);
      std::vector<FlatBitVect<256>> resultsGpuVec(gpuResult.size());
      std::vector<std::unique_ptr<ExplicitBitVect>> results(gpuResult.size());
      cudaCheckError(cudaDeviceSynchronize());
      extractResultsFromGPUBatch<256>(gpuResult, resultsGpuVec);
      cudaCheckError(cudaDeviceSynchronize());
      populateResults<256>(resultsGpuVec, results, numThreads);
      return results;
    }
    case 128: {
      auto                          gpuResult = computeFingerprintsCuImpl<128>(mols, maxRadius, batchSize);
      std::vector<FlatBitVect<128>> resultsGpuVec(gpuResult.size());
      std::vector<std::unique_ptr<ExplicitBitVect>> results(gpuResult.size());
      cudaCheckError(cudaDeviceSynchronize());
      extractResultsFromGPUBatch<128>(gpuResult, resultsGpuVec);
      cudaCheckError(cudaDeviceSynchronize());
      populateResults<128>(resultsGpuVec, results, numThreads);
      return results;
    }
    default:
      throw std::runtime_error("Unsupported fingerprint size" + std::to_string(fpSize) +
                               ", must be multiple of 2 between 128 and 4096");
  }
  // NOLINTEND
}

}  // namespace

MorganFingerprintGpuGenerator::MorganFingerprintGpuGenerator(std::uint32_t radius, std::uint32_t fpSize)
    : radius_(radius),
      fpSize_(fpSize) {}

MorganFingerprintGpuGenerator::~MorganFingerprintGpuGenerator() = default;

std::unique_ptr<ExplicitBitVect> MorganFingerprintGpuGenerator::GetFingerprint(
  const RDKit::ROMol&                      mol,
  std::optional<FingerprintComputeOptions> computeOptions) {
  std::vector<const RDKit::ROMol*> molView;
  molView.push_back(&mol);
  return std::move(GetFingerprints(molView, computeOptions)[0]);
}

std::vector<std::unique_ptr<ExplicitBitVect>> MorganFingerprintGpuGenerator::GetFingerprints(
  const std::vector<const RDKit::ROMol*>&  mols,
  std::optional<FingerprintComputeOptions> computeOptions) {
  const FingerprintComputeOptions options = computeOptions.value_or(FingerprintComputeOptions());
  return getFingerprintsCu(mols,
                           radius_,
                           fpSize_,
                           options.gpuBatchSize.value_or(kDefaultGpuBatchSize),
                           options.numCpuThreads.value_or(omp_get_max_threads()));
}

template <int nBits>
AsyncDeviceVector<FlatBitVect<nBits>> MorganFingerprintGpuGenerator::GetFingerprintsGpuBuffer(
  const std::vector<const RDKit::ROMol*>&  mols,
  cudaStream_t                             stream,
  std::optional<FingerprintComputeOptions> computeOptions) {
  const FingerprintComputeOptions options = computeOptions.value_or(FingerprintComputeOptions());
  if (options.backend != FingerprintComputeBackend::GPU) {
    throw std::runtime_error("GPU results requested but GPU backend is not selected");
  }
  if (mols.empty()) {
    return AsyncDeviceVector<FlatBitVect<nBits>>();
  }
  const size_t batchSize = options.gpuBatchSize.value_or(kDefaultGpuBatchSize);
  return computeFingerprintsCuImpl<nBits>(mols, radius_, batchSize, stream);
}

#define DEFINE_TEMPLATE(fpSize)                                                                                      \
  template AsyncDeviceVector<FlatBitVect<(fpSize)>> MorganFingerprintGpuGenerator::GetFingerprintsGpuBuffer<fpSize>( \
    const std::vector<const RDKit::ROMol*>&  mols,                                                                   \
    cudaStream_t                             stream,                                                                 \
    std::optional<FingerprintComputeOptions> options);
DEFINE_TEMPLATE(128)
DEFINE_TEMPLATE(256)
DEFINE_TEMPLATE(512)
DEFINE_TEMPLATE(1024)
DEFINE_TEMPLATE(2048)
DEFINE_TEMPLATE(4096)
#undef DEFINE_TEMPLATE

}  // namespace nvMolKit
