// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <DistGeom/BoundsMatrix.h>
#include <ForceField/ForceField.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <limits>
#include <numbers>

#include "rdkit_extensions/dist_geom_flattened_builder.h"
#include "src/etkdg_stage_aio_minimization.h"
#include "src/etkdg_stage_etk_minimization.h"
#include "src/forcefields/dg_batched_forcefield.h"
#include "src/forcefields/gaussian_torsion_batched_forcefield.h"
#include "src/utils/device.h"
#include "tests/test_utils.h"
#include "versions.h"

#if RDKIT_ETKDG_2026_09_API
#include <GraphMol/ForceFieldHelpers/CrystalFF/GaussianTorsionAngleContribs.h>

namespace {

ForceFields::CrystalFF::CrystalFFDetails makeGaussianDetails(int tableSize, double scaling) {
  ForceFields::CrystalFF::CrystalFFDetails details{};
  details.torsionParamKind = ForceFields::CrystalFF::TorsionParamKind::Gaussian;
  details.expTorsionAtoms  = {
    {0, 1, 2, 3},
    {3, 2, 1, 0}
  };
  details.expTorsionAngles = {
    ForceFields::CrystalFF::GaussianExp_T{{1.0}, {0.0}, {1.0}, scaling},
    ForceFields::CrystalFF::GaussianExp_T{{1.0}, {0.0}, {1.0},   100.0}
  };
  details.torsionIdx = {0, 0};
  details.phiToEnergy.resize(1);
  details.phiToGrad.resize(1);
  for (int index = 0; index < tableSize; ++index) {
    const double angle = index * std::numbers::pi / (tableSize - 1);
    details.phiToEnergy[0].push_back(1.3 - cos(angle) + 0.2 * cos(2 * angle));
    details.phiToGrad[0].push_back(sin(angle) * (0.7 + cos(angle)));
  }
  return details;
}

}  // namespace

TEST(GaussianTorsions, MatchesRdkitLookupGradientsMasksAndStream) {
  using namespace nvMolKit;
  using namespace nvMolKit::DistGeom;
  const ScopedStream      stream;
  const auto              cudaStream = stream.stream();
  const std::vector<int>  atomStarts{0, 4, 8};
  EnergyForceContribsHost distanceContribs;
  distanceContribs.distTerms.idx1   = {0};
  distanceContribs.distTerms.idx2   = {1};
  distanceContribs.distTerms.lb2    = {4.0};
  distanceContribs.distTerms.ub2    = {4.0};
  distanceContribs.distTerms.weight = {1.0};
  BatchedMolecularSystemHost distanceSystem;
  preallocateEstimatedBatch(distanceContribs, distanceSystem, 2);
  for (int system = 0; system < 2; ++system) {
    addMoleculeToMolecularSystem(distanceContribs, 4, 4, atomStarts, distanceSystem);
  }
  DGBatchedForcefield base(distanceSystem, atomStarts, 1.0, 1.0, {}, cudaStream);
  for (int tableSize : {2, 3, 180, 361}) {
    SCOPED_TRACE(tableSize);
    auto       first  = makeGaussianDetails(tableSize, 0.05);
    auto       second = makeGaussianDetails(tableSize, 1.5);
    const auto packed = constructGaussianTorsionContribs({&first, &second}, atomStarts);
    ASSERT_EQ(packed.idx1.size(), 2);
    EXPECT_EQ(packed.energies.size(), tableSize);
    EXPECT_EQ(packed.tableStarts, (std::vector<int>{0, 0}));
    GaussianTorsionBatchedForcefield              actual(base, packed, cudaStream);
    std::vector<std::unique_ptr<RDGeom::PointND>> pointOwners;
    for (int atom = 0; atom < 4; ++atom) {
      pointOwners.push_back(std::make_unique<RDGeom::PointND>(4));
    }
    std::vector<std::unique_ptr<ForceFields::ForceField>> references;
    for (double scaling : {0.05, 1.5}) {
      auto reference = std::make_unique<ForceFields::ForceField>(4);
      for (const auto& point : pointOwners) {
        reference->positions().push_back(point.get());
      }
      auto contribution = std::make_unique<ForceFields::CrystalFF::GaussianTorsionAngleContribs>(reference.get());
      contribution->addContrib(0, 1, 2, 3, first.phiToEnergy[0], first.phiToGrad[0], scaling);
      reference->contribs().push_back(std::move(contribution));
      reference->initialize();
      references.push_back(std::move(reference));
    }
    AsyncDeviceVector<double>  positionsDevice, energyDevice, gradientDevice;
    AsyncDeviceVector<uint8_t> mask;
    positionsDevice.setStream(cudaStream);
    energyDevice.setStream(cudaStream);
    gradientDevice.setStream(cudaStream);
    mask.setStream(cudaStream);
    energyDevice.resize(2);
    gradientDevice.resize(32);
    const std::vector<double>
      angles{-std::numbers::pi, -2.1, -1e-8, 0.0, 1e-8, 0.47, 2.3, std::numbers::pi - 1e-8, std::numbers::pi};
    for (std::size_t geometry = 0; geometry <= angles.size(); ++geometry) {
      SCOPED_TRACE(geometry);
      const double angle = geometry == angles.size() ? 0.0 : angles[geometry];
      std::vector<double>
        positions{0.0, 1.0, 0.0, 0.1, 0.0, 0.0, 0.0, -0.2, 1.0, 0.0, 0.0, 0.3, 1.0, cos(angle), sin(angle), -0.4};
      if (geometry == angles.size()) {
        positions[1] = 0.0;
      }
      const auto firstPositions = positions;
      positions.insert(positions.end(), firstPositions.begin(), firstPositions.end());
      positionsDevice.setFromVector(positions);
      for (const std::vector<uint8_t>& active : {
             std::vector<uint8_t>{},
             {1, 1},
             {1, 0},
             {0, 1},
             {0, 0}
      }) {
        mask.setFromVector(active);
        const uint8_t*      activeDevice = active.empty() ? nullptr : mask.data();
        std::vector<double> baseEnergy(2), baseGradient(32);
        energyDevice.zero();
        gradientDevice.zero();
        CHECK_CUDA_RETURN(base.computeEnergy(energyDevice.data(), positionsDevice.data(), activeDevice, cudaStream));
        CHECK_CUDA_RETURN(
          base.computeGradients(gradientDevice.data(), positionsDevice.data(), activeDevice, cudaStream));
        energyDevice.copyToHost(baseEnergy);
        gradientDevice.copyToHost(baseGradient);
        CHECK_CUDA_RETURN(cudaStreamSynchronize(cudaStream));
        for (bool enabled : {false, true}) {
          actual.setEnabled(enabled);
          energyDevice.zero();
          gradientDevice.zero();
          CHECK_CUDA_RETURN(
            actual.computeEnergy(energyDevice.data(), positionsDevice.data(), activeDevice, cudaStream));
          CHECK_CUDA_RETURN(
            actual.computeGradients(gradientDevice.data(), positionsDevice.data(), activeDevice, cudaStream));
          std::vector<double> energies(2), gradients(32);
          energyDevice.copyToHost(energies);
          gradientDevice.copyToHost(gradients);
          CHECK_CUDA_RETURN(cudaStreamSynchronize(cudaStream));
          for (int system = 0; system < 2; ++system) {
            const bool   participates = enabled && (active.empty() || active[system]);
            const double expected = participates ? references[system]->calcEnergy(positions.data() + system * 16) : 0.0;
            EXPECT_NEAR(energies[system] - baseEnergy[system], expected, 2e-6 * std::max(1.0, std::abs(expected)));
            std::vector<double> expectedGradient(16, 0.0);
            if (participates) {
              references[system]->calcGrad(positions.data() + system * 16, expectedGradient.data());
            }
            for (int coordinate = 0; coordinate < 16; ++coordinate) {
              EXPECT_NEAR(gradients[system * 16 + coordinate] - baseGradient[system * 16 + coordinate],
                          expectedGradient[coordinate],
                          2e-5 * std::max(1.0, std::abs(expectedGradient[coordinate])))
                << "system=" << system << " coordinate=" << coordinate;
            }
          }
        }
      }
    }
  }
}

TEST(GaussianTorsions, BothRefinementStagesMinimizeGaussianPotential) {
  using namespace nvMolKit;
  std::unique_ptr<RDKit::RWMol> molecule(RDKit::SmilesToMol("CCCC"));
  ASSERT_NE(molecule, nullptr);
  std::vector<nvMolKit::detail::EmbedArgs> arguments(1);
  arguments[0].dim  = 4;
  arguments[0].mmat = std::make_unique<::DistGeom::BoundsMatrix>(4);
  RDKit::DGeomHelpers::initBoundsMat(arguments[0].mmat);
  auto& details                 = arguments[0].etkdgDetails;
  details                       = makeGaussianDetails(361, 1.0);
  details.boundsMatForceScaling = 1.0;
  details.bonds                 = {
    {0, 1},
    {1, 2},
    {2, 3}
  };
  details.angles = {
    {0, 1, 2, 0},
    {1, 2, 3, 0}
  };
  details.atomNums = {6, 6, 6, 6};
  for (int index = 0; index < 361; ++index) {
    const double angle            = index * std::numbers::pi / 360;
    details.phiToEnergy[0][index] = 1.0 - cos(angle);
    details.phiToGrad[0][index]   = sin(angle);
  }
  std::vector<std::unique_ptr<RDGeom::PointND>> points;
  ForceFields::ForceField                       reference(4);
  for (int atom = 0; atom < 4; ++atom) {
    points.push_back(std::make_unique<RDGeom::PointND>(4));
    reference.positions().push_back(points.back().get());
  }
  auto contribution = std::make_unique<ForceFields::CrystalFF::GaussianTorsionAngleContribs>(&reference);
  contribution->addContrib(0, 1, 2, 3, details.phiToEnergy[0], details.phiToGrad[0], 1.0);
  reference.contribs().push_back(std::move(contribution));
  reference.initialize();
  std::vector<double> initial{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0};
  EXPECT_NEAR(reference.calcEnergy(initial.data()), 1.0, 1e-12);
  for (bool legacy : {false, true}) {
    SCOPED_TRACE(legacy);
    nvMolKit::detail::ETKDGContext context;
    context.nTotalSystems         = 1;
    context.systemHost.positions  = initial;
    context.systemHost.atomStarts = {0, 4};
    nvMolKit::DistGeom::sendContextToDevice(initial,
                                            context.systemDevice.positions,
                                            context.systemHost.atomStarts,
                                            context.systemDevice.atomStarts);
    context.activeThisStage.setFromVector({1});
    context.failedThisStage.setFromVector({0});
    auto parameters                    = RDKit::DGeomHelpers::ETKDGv4;
    parameters.useLegacyImplementation = legacy;
    parameters.useBasicKnowledge       = false;
    parameters.optimizerForceTol       = 1e-5;
    BfgsBatchMinimizer                            minimizer(4, DebugLevel::NONE, true, nullptr, BfgsBackend::BATCHED);
    std::unique_ptr<nvMolKit::detail::ETKDGStage> stage;
    if (legacy) {
      stage = std::make_unique<nvMolKit::detail::ETKMinimizationStage>(std::vector<const RDKit::ROMol*>{molecule.get()},
                                                                       arguments,
                                                                       parameters,
                                                                       context,
                                                                       minimizer);
    } else {
      stage = std::make_unique<nvMolKit::detail::ETKDGAllInOneMinimizationStage>(
        std::vector<const RDKit::ROMol*>{molecule.get()},
        arguments,
        parameters,
        context,
        minimizer);
    }
    stage->execute(context);
    std::vector<double>  positions(initial.size());
    std::vector<uint8_t> failures(1);
    context.systemDevice.positions.copyToHost(positions);
    context.failedThisStage.copyToHost(failures);
    CHECK_CUDA_RETURN(cudaDeviceSynchronize());
    EXPECT_EQ(failures[0], 0);
    EXPECT_LT(reference.calcEnergy(positions.data()), 1e-3);
  }
}

TEST(GaussianTorsions, RejectsMalformedMetadata) {
  using nvMolKit::DistGeom::constructGaussianTorsionContribs;
  auto details = makeGaussianDetails(180, 1.0);
  EXPECT_THROW(constructGaussianTorsionContribs({&details}, {0}), std::invalid_argument);
  EXPECT_THROW(constructGaussianTorsionContribs({nullptr}, {0, 4}), std::invalid_argument);
  details.torsionIdx.pop_back();
  EXPECT_THROW(constructGaussianTorsionContribs({&details}, {0, 4}), std::invalid_argument);
  details.torsionIdx.push_back(0);
  details.expTorsionAtoms[0][3] = 4;
  EXPECT_THROW(constructGaussianTorsionContribs({&details}, {0, 4}), std::invalid_argument);
  details.expTorsionAtoms[0][3] = 3;
  details.phiToGrad[0].pop_back();
  EXPECT_THROW(constructGaussianTorsionContribs({&details}, {0, 4}), std::invalid_argument);
  EXPECT_TRUE(constructGaussianTorsionContribs({}, {0}).idx1.empty());
}
#endif

TEST(GaussianTorsions, EmptyAndInvalidDeviceContributions) {
  using namespace nvMolKit;
  nvMolKit::DistGeom::BatchedMolecularSystemHost distanceSystem;
  const std::vector<int>                         atomStarts{0, 4};
  nvMolKit::DistGeom::addMoleculeToMolecularSystem({}, 4, 4, atomStarts, distanceSystem);
  DGBatchedForcefield                             base(distanceSystem, atomStarts, 1.0, 1.0);
  nvMolKit::DistGeom::GaussianTorsionContribsHost data;
  EXPECT_NO_THROW((GaussianTorsionBatchedForcefield(base, data)));
  data.idx1 = {0};
  EXPECT_THROW((GaussianTorsionBatchedForcefield(base, data)), std::invalid_argument);
  data.idx2        = {1};
  data.idx3        = {2};
  data.idx4        = {3};
  data.systemIdx   = {0};
  data.scaling     = {1.0};
  data.tableStarts = {0};
  data.tableSizes  = {1};
  data.energies    = {1.0};
  data.derivatives = {0.0};
  EXPECT_THROW((GaussianTorsionBatchedForcefield(base, data)), std::invalid_argument);
  data.tableSizes = {2};
  data.energies.push_back(0.0);
  data.derivatives.push_back(0.0);
  data.idx4[0] = 4;
  EXPECT_THROW((GaussianTorsionBatchedForcefield(base, data)), std::invalid_argument);
  data.idx4[0]      = 3;
  data.systemIdx[0] = 1;
  EXPECT_THROW((GaussianTorsionBatchedForcefield(base, data)), std::invalid_argument);
  data.systemIdx[0] = 0;
  GaussianTorsionBatchedForcefield actual(base, data);
  AsyncDeviceVector<double>        positionsDevice, energiesDevice(1);
  for (double invalid : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    std::vector<double> positions{0.0, invalid, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    positionsDevice.setFromVector(positions);
    energiesDevice.zero();
    CHECK_CUDA_RETURN(actual.computeEnergy(energiesDevice.data(), positionsDevice.data()));
    std::vector<double> energies(1);
    energiesDevice.copyToHost(energies);
    CHECK_CUDA_RETURN(cudaDeviceSynchronize());
    EXPECT_FALSE(std::isfinite(energies[0]));
  }
}
