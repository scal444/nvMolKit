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

#include "bfgs_common.h"

#include <GraphMol/Conformer.h>
#include <GraphMol/ROMol.h>
#include <omp.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

#include "device.h"

namespace nvMolKit {

void ThreadLocalBuffers::ensureCapacity(const size_t positionsSize, const size_t energiesSize) {
  constexpr double extraCapacityFactor = 1.3;
  const auto       newSize             = static_cast<size_t>(static_cast<double>(positionsSize) * extraCapacityFactor);
  if (positions.size() < positionsSize) {
    positions.resize(newSize);
  }
  if (energies.size() < energiesSize) {
    energies.resize(static_cast<size_t>(static_cast<double>(energiesSize) * extraCapacityFactor));
  }
  if (initialPositions.size() < positionsSize) {
    initialPositions.resize(newSize);
  }
}

gpu_scheduler::Config configFromHardwareOptions(const BatchHardwareOptions& perfOptions) {
  gpu_scheduler::Config config;
  config.gpuIds = perfOptions.gpuIds;
  if (config.gpuIds.empty()) {
    const int numDevices = countCudaDevices();
    if (numDevices == 0) {
      throw std::runtime_error("No CUDA devices found");
    }
    config.gpuIds.resize(numDevices);
    std::iota(config.gpuIds.begin(), config.gpuIds.end(), 0);
  }

  // workerThreadsPerGpu maps to BatchHardwareOptions::batchesPerGpu. When
  // batchesPerGpu is unset, fall back to distributing all available host
  // threads across the chosen GPUs (matches the legacy default).
  if (perfOptions.batchesPerGpu > 0) {
    config.workerThreadsPerGpu = perfOptions.batchesPerGpu;
  } else {
    const int hwThreads        = std::max(1, omp_get_max_threads());
    const int numGpus          = static_cast<int>(config.gpuIds.size());
    config.workerThreadsPerGpu = std::max(1, hwThreads / numGpus);
  }

  // The legacy OpenMP loop did not pipeline batches per worker (each
  // iteration owned its GPU work end-to-end), so default slotsPerWorker to 1
  // to preserve per-runner concurrency characteristics.
  config.slotsPerWorker = 1;

  if (perfOptions.preprocessingThreads > 0) {
    config.globalPreprocessingThreads = perfOptions.preprocessingThreads;
  }
  return config;
}

int resolveBatchSize(const BatchHardwareOptions& perfOptions, const int totalConformers) {
  const int requested = perfOptions.batchSize == -1 ? 500 : perfOptions.batchSize;
  return requested <= 0 ? totalConformers : requested;
}

std::vector<ConformerInfo> flattenConformers(const std::vector<RDKit::ROMol*>& mols,
                                             std::vector<std::vector<double>>& moleculeEnergies) {
  moleculeEnergies.resize(mols.size());
  std::vector<ConformerInfo> allConformers;
  for (size_t molIdx = 0; molIdx < mols.size(); ++molIdx) {
    auto* mol = mols[molIdx];
    if (mol == nullptr) {
      throw std::invalid_argument("Invalid molecule pointer at index " + std::to_string(molIdx));
    }
    moleculeEnergies[molIdx].resize(mol->getNumConformers());
    size_t confIdx = 0;
    for (auto confIter = mol->beginConformers(); confIter != mol->endConformers(); ++confIter, ++confIdx) {
      allConformers.push_back({mol, molIdx, &(**confIter), static_cast<int>((*confIter)->getId()), confIdx});
    }
  }
  return allConformers;
}

void writeBackResults(const std::vector<ConformerInfo>& batchConformers,
                      const std::vector<uint32_t>&      conformerAtomStarts,
                      const ThreadLocalBuffers&         buffers,
                      std::vector<std::vector<double>>& moleculeEnergies) {
  for (size_t i = 0; i < batchConformers.size(); ++i) {
    const auto&    confInfo     = batchConformers[i];
    const uint32_t numAtoms     = confInfo.mol->getNumAtoms();
    const uint32_t atomStartIdx = conformerAtomStarts[i];
    for (uint32_t j = 0; j < numAtoms; ++j) {
      confInfo.conformer->setAtomPos(j,
                                     RDGeom::Point3D(buffers.positions[3 * (atomStartIdx + j) + 0],
                                                     buffers.positions[3 * (atomStartIdx + j) + 1],
                                                     buffers.positions[3 * (atomStartIdx + j) + 2]));
    }
    moleculeEnergies[confInfo.molIdx][confInfo.confIdx] = buffers.energies[i];
  }
}

}  // namespace nvMolKit
