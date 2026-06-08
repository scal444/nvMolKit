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

#include "src/minimizer/bfgs_mmff.h"

#include <GraphMol/ROMol.h>

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "src/conformer/ff_device_collect.h"
#include "src/gpu_scheduler/pipeline.h"
#include "src/minimizer/bfgs_common.h"
#include "src/minimizer/mmff_workload.h"
#include "src/utils/device.h"
#include "src/utils/nvtx.h"

namespace nvMolKit::MMFF {

std::vector<std::vector<double>> MMFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>& mols,
                                                                const int                   maxIters,
                                                                const MMFFProperties&       properties,
                                                                const BatchHardwareOptions& perfOptions,
                                                                const BfgsBackend           backend) {
  return MMFFOptimizeMoleculesConfsBfgs(mols,
                                        maxIters,
                                        std::vector<MMFFProperties>(mols.size(), properties),
                                        perfOptions,
                                        backend);
}

MMFFMinimizeResult MMFFMinimizeMoleculesConfs(std::vector<RDKit::ROMol*>&                                  mols,
                                              const int                                                    maxIters,
                                              const double                                                 gradTol,
                                              const std::vector<MMFFProperties>&                           properties,
                                              const std::vector<ForceFieldConstraints::PerMolConstraints>& constraints,
                                              const BatchHardwareOptions&                                  perfOptions,
                                              const BfgsBackend                                            backend,
                                              const CoordinateOutput                                       output,
                                              int                                                          targetGpu,
                                              const DeviceCoordResult* deviceInput) {
  ScopedNvtxRange fullRange("BFGS MMFF Minimize Molecules Confs");

  if (properties.size() != mols.size()) {
    throw std::invalid_argument("Expected one MMFFProperties entry per molecule");
  }
  if (!constraints.empty() && constraints.size() != mols.size()) {
    throw std::invalid_argument("Expected one PerMolConstraints entry per molecule");
  }
  if (deviceInput != nullptr) {
    const bool anyConstraint =
      std::any_of(constraints.begin(), constraints.end(), [](const auto& perMol) { return !perMol.empty(); });
    if (anyConstraint) {
      throw std::invalid_argument(
        "Device input coordinates not supported with custom constraints. "
        "Use the RDKit Mol + Conformer path to apply constraints "
        "(call MMFFMinimizeMoleculesConfs without deviceInput; the constraints anchor positions are "
        "read from each mol's RDKit conformer at force-field construction time).");
    }
  }

  const bool deviceOutput = output == CoordinateOutput::DEVICE;
  const auto config       = configFromHardwareOptions(perfOptions);

  if (deviceOutput) {
    if (targetGpu < 0) {
      targetGpu = config.gpuIds.empty() ? 0 : config.gpuIds.front();
    }
    if (std::find(config.gpuIds.begin(), config.gpuIds.end(), targetGpu) == config.gpuIds.end()) {
      throw std::invalid_argument(
        "targetGpu " + std::to_string(targetGpu) +
        " is not in the configured set of execution GPUs; pass it via perfOptions.gpuIds first.");
    }
  }

  std::vector<std::vector<double>> moleculeEnergies;
  const auto                       allConformers = flattenConformers(mols, moleculeEnergies);

  std::vector<std::vector<int8_t>> moleculeConverged(mols.size());
  for (size_t i = 0; i < mols.size(); ++i) {
    moleculeConverged[i].resize(moleculeEnergies[i].size(), 0);
  }

  // Build & validate the device-input index up front so per-batch broadcasts can do an O(1)
  // source lookup for each batch slot. The validation also catches mismatched conformer
  // counts/labels before we start any GPU work.
  detail::DeviceInputIndex deviceInputIndex;
  if (deviceInput != nullptr) {
    deviceInputIndex = detail::buildDeviceInputIndex(*deviceInput, allConformers);
  }

  if (allConformers.empty()) {
    if (deviceOutput) {
      std::vector<detail::DeviceCoordCollector> emptyCollectors;
      return {{}, {}, detail::finalizeOnTarget(emptyCollectors, targetGpu, static_cast<int>(mols.size()))};
    }
    return {moleculeEnergies, moleculeConverged, std::nullopt};
  }

  const int effectiveBatchSize = resolveBatchSize(perfOptions, static_cast<int>(allConformers.size()));
  const int numRunners         = static_cast<int>(config.gpuIds.size()) * config.workerThreadsPerGpu;

  // Declared in this order so destruction runs `deviceCollectors` first (its
  // AsyncDeviceVectors free async on the collector streams) and then the
  // runner streams (each ScopedStream's destructor calls cudaStreamDestroy).
  std::vector<std::unique_ptr<ScopedStream>> runnerStreams(deviceOutput ? static_cast<size_t>(numRunners) : 0);
  std::vector<detail::DeviceCoordCollector>  deviceCollectors(deviceOutput ? static_cast<size_t>(numRunners) : 0);

  std::mutex outputMutex;
  MmffInputs inputs;
  inputs.allConformers     = &allConformers;
  inputs.properties        = &properties;
  inputs.constraints       = &constraints;
  inputs.maxIters          = maxIters;
  inputs.gradTol           = gradTol;
  inputs.backend           = backend;
  inputs.batchSize         = effectiveBatchSize;
  inputs.output            = output;
  inputs.deviceInput       = deviceInput;
  inputs.deviceInputIndex  = deviceInput != nullptr ? &deviceInputIndex : nullptr;
  inputs.deviceCollectors  = deviceOutput ? &deviceCollectors : nullptr;
  inputs.runnerStreams     = deviceOutput ? &runnerStreams : nullptr;
  inputs.moleculeEnergies  = &moleculeEnergies;
  inputs.moleculeConverged = &moleculeConverged;
  inputs.outputMutex       = &outputMutex;

  MmffWorkload            workload(inputs);
  gpu_scheduler::Pipeline pipeline(config, workload);
  pipeline.run();

  if (deviceOutput) {
    // Some runners may not have processed any batches (e.g. when totalUnits <
    // numRunners), leaving their collector slot unused. Trim those off before
    // finalize so finalizeOnTarget doesn't see uninitialized gpuId == -1.
    const int claimed = inputs.nextRunnerIdx.load(std::memory_order_relaxed);
    deviceCollectors.resize(static_cast<size_t>(claimed));
    return {{}, {}, detail::finalizeOnTarget(deviceCollectors, targetGpu, static_cast<int>(mols.size()))};
  }
  return {moleculeEnergies, moleculeConverged, std::nullopt};
}

std::vector<std::vector<double>> MMFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>&        mols,
                                                                const int                          maxIters,
                                                                const std::vector<MMFFProperties>& properties,
                                                                const BatchHardwareOptions&        perfOptions,
                                                                const BfgsBackend                  backend) {
  return MMFFMinimizeMoleculesConfs(mols, maxIters, 1e-4, properties, {}, perfOptions, backend).energies;
}

}  // namespace nvMolKit::MMFF
