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

#include <GraphMol/DistGeomHelpers/Embedder.h>

#include "dist_geom.h"
#include "dist_geom_flattened_builder.h"
#include "etkdg_impl.h"
#include "etkdg_stage_fourthdimminimization.h"
#include "forcefields/kernel_utils.cuh"
#include "minimizer/bfgs_distgeom.h"

using ::nvMolKit::detail::ETKDGContext;
using ::nvMolKit::detail::ETKDGStage;

namespace nvMolKit {

namespace detail {
FourthDimMinimizeStage::FourthDimMinimizeStage(const std::vector<const RDKit::ROMol*>&     mols,
                                               const std::vector<EmbedArgs>&               eargs,
                                               const RDKit::DGeomHelpers::EmbedParameters& embedParam,
                                               ETKDGContext&                               ctx,
                                               BfgsBatchMinimizer&                         minimizer,
                                               const cudaStream_t                          stream)
    : embedParam_(embedParam),
      minimizer_(minimizer),
      stream_(stream) {
  if (mols.size() != eargs.size()) {
    throw std::runtime_error("Number of molecules and embed args must be the same");
  }
  setStreams(molSystemDevice, stream_);

  // Process each molecule
  for (size_t i = 0; i < mols.size(); ++i) {
    const auto&      mol      = mols[i];
    const auto&      embedArg = eargs[i];
    const auto&      numAtoms = mol->getNumAtoms();
    auto             ffParams = nvMolKit::DistGeom::constructForceFieldContribs(embedArg.dim,
                                                                    *embedArg.mmat,
                                                                    embedArg.chiralCenters,
                                                                    0.2,
                                                                    1.0,
                                                                    nullptr,
                                                                    embedParam_.basinThresh);
    // Get atom numbers
    std::vector<int> atomNumbers;
    atomNumbers.reserve(numAtoms);
    for (const auto& atom : mol->atoms()) {
      atomNumbers.push_back(atom->getAtomicNum());
    }

    // Add to molecular system
    nvMolKit::DistGeom::addMoleculeToMolecularSystem(ffParams,
                                                     numAtoms,
                                                     embedArg.dim,
                                                     ctx.systemHost.atomStarts,
                                                     molSystemHost,
                                                     &atomNumbers);
  }
  nvMolKit::DistGeom::sendContribsAndIndicesToDevice(molSystemHost, molSystemDevice);
  nvMolKit::DistGeom::setupDeviceBuffers(molSystemHost,
                                         molSystemDevice,
                                         ctx.systemHost.positions,
                                         ctx.systemHost.atomStarts.size() - 1);
}

void FourthDimMinimizeStage::execute(ETKDGContext& ctx) {
  // Setup device buffers for minimization
  DistGeom::setupDeviceBuffers(molSystemHost,
                               molSystemDevice,
                               ctx.systemHost.positions,
                               static_cast<int>(ctx.systemHost.atomStarts.size() - 1));

  const size_t numAtoms = ctx.systemHost.atomStarts.back();
  const size_t numPos   = ctx.systemHost.positions.size();
  const int    dim      = (numPos == numAtoms * 3) ? 3 : 4;

  // Allocate energy buffer (not used for DG but required by interface)
  AsyncDeviceVector<double> energyBuffer(0, stream_);

  // Use shared minimizer with repeat-until-converged
  constexpr int maxIters = 200;
  if (minimizer_.backend() == BfgsBackend::BATCHED) {
    // BATCHED backend: use generic minimize() with energy/gradient functors
    // Allocate intermediate buffers before minimization
    DistGeom::allocateIntermediateBuffers(molSystemHost, molSystemDevice);
    
    auto eFunc = [&](const double* positions) {
      DistGeom::computeEnergy(molSystemDevice,
                              ctx.systemDevice.atomStarts,
                              ctx.systemDevice.positions,
                              ctx.activeThisStage.data(),
                              positions,
                              stream_);
    };

    auto gFunc = [&]() {
      DistGeom::computeGradients(molSystemDevice,
                                 ctx.systemDevice.atomStarts,
                                 ctx.systemDevice.positions,
                                 ctx.activeThisStage.data(),
                                 stream_);
    };

    bool needsMore = minimizer_.minimize(maxIters,
                                        embedParam_.optimizerForceTol,
                                        ctx.systemHost.atomStarts,
                                        ctx.systemDevice.atomStarts,
                                        ctx.systemDevice.positions,
                                        molSystemDevice.grad,
                                        molSystemDevice.energyOuts,
                                        molSystemDevice.energyBuffer,
                                        eFunc,
                                        gFunc,
                                        ctx.activeThisStage.data());

    // Repeat until converged
    while (needsMore) {
      needsMore = minimizer_.minimize(maxIters,
                                     embedParam_.optimizerForceTol,
                                     ctx.systemHost.atomStarts,
                                     ctx.systemDevice.atomStarts,
                                     ctx.systemDevice.positions,
                                     molSystemDevice.grad,
                                     molSystemDevice.energyOuts,
                                     molSystemDevice.energyBuffer,
                                     eFunc,
                                     gFunc,
                                     ctx.activeThisStage.data());
    }
  } else {
    // PER_MOLECULE backend: use specialized minimizeWithDG()
    auto terms         = DistGeom::toEnergyForceContribsDevicePtr(molSystemDevice);
    auto systemIndices = DistGeom::toBatchedIndicesDevicePtr(molSystemDevice, ctx.systemDevice.atomStarts.data());

    bool needsMore = minimizer_.minimizeWithDG(maxIters,
                                               embedParam_.optimizerForceTol,
                                               ctx.systemHost.atomStarts,
                                               ctx.systemDevice.atomStarts,
                                               ctx.systemDevice.positions,
                                               molSystemDevice.grad,
                                               molSystemDevice.energyOuts,
                                               energyBuffer,
                                               terms,
                                               systemIndices,
                                               ctx.activeThisStage.data());

    // Repeat until converged
    while (needsMore) {
      needsMore = minimizer_.minimizeWithDG(maxIters,
                                            embedParam_.optimizerForceTol,
                                            ctx.systemHost.atomStarts,
                                            ctx.systemDevice.atomStarts,
                                            ctx.systemDevice.positions,
                                            molSystemDevice.grad,
                                            molSystemDevice.energyOuts,
                                            energyBuffer,
                                            terms,
                                            systemIndices,
                                            ctx.activeThisStage.data());
    }
  }
}

}  // namespace detail
}  // namespace nvMolKit
