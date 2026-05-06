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

#include "bfgs_mmff.h"

#include <GraphMol/ROMol.h>

#include <mutex>
#include <vector>

#include "bfgs_common.h"
#include "mmff_workload.h"
#include "nvtx.h"
#include "pipeline.h"

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
                                              const BfgsBackend                                            backend) {
  ScopedNvtxRange fullRange("BFGS MMFF Minimize Molecules Confs");

  if (properties.size() != mols.size()) {
    throw std::invalid_argument("Expected one MMFFProperties entry per molecule");
  }
  if (!constraints.empty() && constraints.size() != mols.size()) {
    throw std::invalid_argument("Expected one PerMolConstraints entry per molecule");
  }

  std::vector<std::vector<double>> moleculeEnergies;
  const auto                       allConformers = flattenConformers(mols, moleculeEnergies);

  std::vector<std::vector<int8_t>> moleculeConverged(mols.size());
  for (size_t i = 0; i < mols.size(); ++i) {
    moleculeConverged[i].resize(moleculeEnergies[i].size(), 0);
  }

  if (allConformers.empty()) {
    return {moleculeEnergies, moleculeConverged};
  }

  const int effectiveBatchSize = resolveBatchSize(perfOptions, static_cast<int>(allConformers.size()));

  std::mutex outputMutex;
  MmffInputs inputs;
  inputs.allConformers     = &allConformers;
  inputs.properties        = &properties;
  inputs.constraints       = &constraints;
  inputs.maxIters          = maxIters;
  inputs.gradTol           = gradTol;
  inputs.backend           = backend;
  inputs.batchSize         = effectiveBatchSize;
  inputs.moleculeEnergies  = &moleculeEnergies;
  inputs.moleculeConverged = &moleculeConverged;
  inputs.outputMutex       = &outputMutex;

  const auto config = configFromHardwareOptions(perfOptions);
  gpu_scheduler::Pipeline<MmffWorkload> pipeline(config, inputs);
  pipeline.run();

  return {moleculeEnergies, moleculeConverged};
}

std::vector<std::vector<double>> MMFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>&        mols,
                                                                const int                          maxIters,
                                                                const std::vector<MMFFProperties>& properties,
                                                                const BatchHardwareOptions&        perfOptions,
                                                                const BfgsBackend                  backend) {
  return MMFFMinimizeMoleculesConfs(mols, maxIters, 1e-4, properties, {}, perfOptions, backend).energies;
}

}  // namespace nvMolKit::MMFF
