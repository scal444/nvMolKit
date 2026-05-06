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

#include "bfgs_uff.h"

#include <GraphMol/ROMol.h>

#include <mutex>
#include <vector>

#include "bfgs_common.h"
#include "nvtx.h"
#include "pipeline.h"
#include "uff_workload.h"

namespace nvMolKit::UFF {

UFFMinimizeResult UFFMinimizeMoleculesConfs(std::vector<RDKit::ROMol*>& mols,
                                            const int                   maxIters,
                                            const double                gradTol,
                                            const std::vector<double>&  vdwThresholds,
                                            const std::vector<bool>&    ignoreInterfragInteractions,
                                            const std::vector<ForceFieldConstraints::PerMolConstraints>& constraints,
                                            const BatchHardwareOptions&                                  perfOptions) {
  ScopedNvtxRange fullRange("BFGS UFF Minimize Molecules Confs");

  if (vdwThresholds.size() != mols.size()) {
    throw std::invalid_argument("Expected one vdw threshold per molecule");
  }
  if (ignoreInterfragInteractions.size() != mols.size()) {
    throw std::invalid_argument("Expected one interfragment interaction flag per molecule");
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
  UffInputs  inputs;
  inputs.allConformers               = &allConformers;
  inputs.vdwThresholds               = &vdwThresholds;
  inputs.ignoreInterfragInteractions = &ignoreInterfragInteractions;
  inputs.constraints                 = &constraints;
  inputs.maxIters                    = maxIters;
  inputs.gradTol                     = gradTol;
  inputs.batchSize                   = effectiveBatchSize;
  inputs.moleculeEnergies            = &moleculeEnergies;
  inputs.moleculeConverged           = &moleculeConverged;
  inputs.outputMutex                 = &outputMutex;

  const auto config = configFromHardwareOptions(perfOptions);
  gpu_scheduler::Pipeline<UffWorkload> pipeline(config, inputs);
  pipeline.run();

  return {moleculeEnergies, moleculeConverged};
}

std::vector<std::vector<double>> UFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>& mols,
                                                               const int                   maxIters,
                                                               const std::vector<double>&  vdwThresholds,
                                                               const std::vector<bool>&    ignoreInterfragInteractions,
                                                               const BatchHardwareOptions& perfOptions) {
  return UFFMinimizeMoleculesConfs(mols, maxIters, 1e-4, vdwThresholds, ignoreInterfragInteractions, {}, perfOptions)
    .energies;
}

}  // namespace nvMolKit::UFF
