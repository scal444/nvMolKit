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

#include <gmock/gmock.h>
// clang-format off
// Bug in RDKit, includes need to be ordered.
#include <GraphMol/ROMol.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
// clang-format on
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "../rdkit_extensions/mmff_flattened_builder.h"
#include "fire_minimizer.h"
#include "mmff.h"
#include "mmff_batched_forcefield.h"
#include "test_utils.h"

using ::nvMolKit::MMFF::BatchedMolecularDeviceBuffers;
using ::nvMolKit::MMFF::BatchedMolecularSystemHost;

namespace {

void perturbConformer(RDKit::Conformer& conf, const float delta = 0.1, const int seed = 0) {
  std::mt19937                          gen(seed);
  std::uniform_real_distribution<float> dist(-delta, delta);
  for (unsigned int i = 0; i < conf.getNumAtoms(); ++i) {
    RDGeom::Point3D pos = conf.getAtomPos(i);
    pos.x += delta * dist(gen);
    pos.y += delta * dist(gen);
    pos.z += delta * dist(gen);
    conf.setAtomPos(i, pos);
  }
}

struct PerMolFireFixture {
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  BatchedMolecularSystemHost                 systemHost;
  BatchedMolecularDeviceBuffers              systemDevice;

  void setup(int numMols) {
    getMols(getTestDataFolderPath() + "/MMFF94_dative.sdf", mols, numMols);
    int runningSeed = 0;
    for (const auto& mol : mols) {
      perturbConformer(mol->getConformer(), 0.3, runningSeed++);
      std::vector<double> positions(3 * mol->getNumAtoms());
      for (unsigned int i = 0; i < mol->getNumAtoms(); ++i) {
        const RDGeom::Point3D pos = mol->getConformer().getAtomPos(i);
        positions[3 * i]          = pos.x;
        positions[3 * i + 1]      = pos.y;
        positions[3 * i + 2]      = pos.z;
      }
      const auto ffParams = nvMolKit::MMFF::constructForcefieldContribs(*mol);
      nvMolKit::MMFF::addMoleculeToBatch(ffParams, positions, systemHost);
    }
    nvMolKit::MMFF::sendContribsAndIndicesToDevice(systemHost, systemDevice);
    nvMolKit::MMFF::allocateIntermediateBuffers(systemHost, systemDevice);
    systemDevice.energyOuts.zero();
    systemDevice.positions.setFromVector(systemHost.positions);
    systemDevice.grad.resize(systemDevice.positions.size());
    systemDevice.grad.zero();
  }
};

std::vector<double> computeReferenceEnergies(const std::vector<std::unique_ptr<RDKit::ROMol>>& mols) {
  std::vector<double> energies;
  energies.reserve(mols.size());
  for (const auto& mol : mols) {
    const auto                                     molProps = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol);
    const std::unique_ptr<ForceFields::ForceField> molFF(RDKit::MMFF::constructForceField(*mol, molProps.get()));
    molFF->initialize();
    molFF->minimize(500, 1e-4);
    energies.push_back(molFF->calcEnergy());
  }
  return energies;
}

}  // namespace

TEST(FireMinimizerPerMolMMFF, ConvergesNearReference) {
  PerMolFireFixture fixture;
  fixture.setup(/*numMols=*/4);
  const std::vector<double> refEnergies = computeReferenceEnergies(fixture.mols);

  nvMolKit::FireOptions options{};
  options.useMass               = false;
  options.stuckDetectionEnabled = false;
  options.gradTol               = 1e-3;
  options.dtInit                = 0.05;
  options.dMax                  = 0.2;

  nvMolKit::FireBatchMinimizer minimizer(/*dataDim=*/3,
                                         options,
                                         /*stream=*/nullptr,
                                         /*debugMode=*/false,
                                         nvMolKit::FireBackend::PER_MOLECULE);
  // Run several extension cycles since FIRE typically needs many more steps than BFGS.
  bool                         converged = minimizer.minimizeWithMMFF(/*numIters=*/2000,
                                              /*gradTol=*/options.gradTol,
                                              fixture.systemHost.indices.atomStarts,
                                              fixture.systemDevice);
  for (int extension = 0; !converged && extension < 4; ++extension) {
    converged = minimizer.minimizeWithMMFF(/*numIters=*/2000,
                                           /*gradTol=*/options.gradTol,
                                           fixture.systemHost.indices.atomStarts,
                                           fixture.systemDevice);
  }

  std::vector<double> energiesHost(fixture.systemDevice.energyOuts.size());
  fixture.systemDevice.energyOuts.copyToHost(energiesHost);
  cudaDeviceSynchronize();

  // Each per-mol-FIRE-minimized energy should be within an absolute or relative
  // tolerance of RDKit's BFGS-minimized reference.
  for (size_t i = 0; i < energiesHost.size(); ++i) {
    const double tolerance = 1.0 + 0.05 * std::abs(refEnergies[i]);
    EXPECT_NEAR(energiesHost[i], refEnergies[i], tolerance) << "system " << i;
  }
}

TEST(FireMinimizerPerMolMMFF, ThrowsWhenStuckDetectionEnabled) {
  PerMolFireFixture fixture;
  fixture.setup(/*numMols=*/1);

  nvMolKit::FireOptions options{};
  options.stuckDetectionEnabled = true;
  options.gradTol               = 1e-3;

  nvMolKit::FireBatchMinimizer minimizer(/*dataDim=*/3,
                                         options,
                                         /*stream=*/nullptr,
                                         /*debugMode=*/false,
                                         nvMolKit::FireBackend::PER_MOLECULE);
  EXPECT_THROW(minimizer.minimizeWithMMFF(/*numIters=*/10,
                                          /*gradTol=*/options.gradTol,
                                          fixture.systemHost.indices.atomStarts,
                                          fixture.systemDevice),
               std::runtime_error);
}

TEST(FireMinimizerPerMolMMFF, ResolveBackendHybridSwitchesAtThreshold) {
  nvMolKit::FireOptions options{};
  options.stuckDetectionEnabled = false;
  nvMolKit::FireBatchMinimizer minimizer(/*dataDim=*/3,
                                         options,
                                         /*stream=*/nullptr,
                                         /*debugMode=*/false,
                                         nvMolKit::FireBackend::HYBRID);
  // Atom counts well below the hybrid threshold should select PER_MOLECULE.
  EXPECT_EQ(minimizer.resolveBackend({0, 5, 10, 30}), nvMolKit::FireBackend::PER_MOLECULE);
  // A molecule above the threshold should force BATCHED.
  EXPECT_EQ(minimizer.resolveBackend({0, 5, 10, 200}), nvMolKit::FireBackend::BATCHED);
}
