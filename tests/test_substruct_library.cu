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

#include <GraphMol/MolOps.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/substruct/substruct_library.h"
#include "src/utils/device.h"

namespace {

using MoleculeId = unsigned int;

std::unique_ptr<RDKit::ROMol> molFromSmiles(const std::string& smiles) {
  auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
  EXPECT_NE(mol, nullptr) << "Failed to parse SMILES: " << smiles;
  return mol;
}

std::unique_ptr<RDKit::ROMol> queryFromSmarts(const std::string& smarts) {
  auto query = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
  EXPECT_NE(query, nullptr) << "Failed to parse SMARTS: " << smarts;
  return query;
}

std::vector<MoleculeId> rdkitMatchingIds(const std::vector<std::unique_ptr<RDKit::ROMol>>& targets,
                                         const RDKit::ROMol&                               query) {
  std::vector<MoleculeId> expected;
  for (std::size_t targetIdx = 0; targetIdx < targets.size(); ++targetIdx) {
    RDKit::MatchVectType match;
    if (RDKit::SubstructMatch(*targets[targetIdx], query, match)) {
      expected.push_back(static_cast<MoleculeId>(targetIdx));
    }
  }
  return expected;
}

std::string linearAlkane(std::size_t numAtoms) {
  return std::string(numAtoms, 'C');
}

std::unique_ptr<RDKit::ROMol> highDegreeMolecule() {
  RDKit::SmilesParserParams params;
  params.sanitize = false;
  auto mol        = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol("[Fe](C)(C)(C)(C)(C)(C)(C)(C)C", params));
  if (mol) {
    RDKit::MolOps::symmetrizeSSSR(*mol);
  }
  return mol;
}

TEST(SubstructLibraryState, RejectsQueriesBeforeFirstFinalize) {
  nvMolKit::SubstructLibrary library(2);
  auto                       target = molFromSmiles("CCO");
  auto                       query  = queryFromSmarts("CO");

  ASSERT_NE(target, nullptr);
  ASSERT_NE(query, nullptr);
  EXPECT_EQ(library.addMol(*target), 0U);
  EXPECT_EQ(library.size(), 0U);
  EXPECT_EQ(library.pendingSize(), 1U);

  EXPECT_THROW(static_cast<void>(library.getMatches(*query)), std::logic_error);
  EXPECT_THROW(static_cast<void>(library.countMatches(*query)), std::logic_error);
  EXPECT_THROW(static_cast<void>(library.hasMatch(*query)), std::logic_error);
}

TEST(SubstructLibraryState, FinalizedEmptyLibraryHasEmptyResults) {
  nvMolKit::SubstructLibrary library;
  auto                       query = queryFromSmarts("C");

  ASSERT_NE(query, nullptr);
  EXPECT_NO_THROW(library.finalize());
  EXPECT_EQ(library.size(), 0U);
  EXPECT_EQ(library.pendingSize(), 0U);
  EXPECT_TRUE(library.getMatches(*query).empty());
  EXPECT_EQ(library.countMatches(*query), 0U);
  EXPECT_FALSE(library.hasMatch(*query));

  EXPECT_NO_THROW(library.finalize());
  EXPECT_TRUE(library.getMatches(*query).empty());
}

TEST(SubstructLibraryState, PublishesPendingMoleculesAtomicallyAcrossGenerations) {
  nvMolKit::SubstructLibrary library(2);
  auto                       ethanol        = molFromSmiles("CCO");
  auto                       benzene        = molFromSmiles("c1ccccc1");
  auto                       phenol         = molFromSmiles("Oc1ccccc1");
  auto                       aromaticCarbon = queryFromSmarts("c");

  ASSERT_NE(ethanol, nullptr);
  ASSERT_NE(benzene, nullptr);
  ASSERT_NE(phenol, nullptr);
  ASSERT_NE(aromaticCarbon, nullptr);

  EXPECT_EQ(library.addMol(*ethanol), 0U);
  EXPECT_EQ(library.addMol(*benzene), 1U);
  library.finalize();
  EXPECT_EQ(library.size(), 2U);
  EXPECT_EQ(library.pendingSize(), 0U);
  EXPECT_EQ(library.getMatches(*aromaticCarbon), std::vector<MoleculeId>({1U}));

  EXPECT_EQ(library.addMol(*phenol), 2U);
  EXPECT_EQ(library.size(), 2U);
  EXPECT_EQ(library.pendingSize(), 1U);

  EXPECT_EQ(library.getMatches(*aromaticCarbon), std::vector<MoleculeId>({1U}));
  EXPECT_EQ(library.countMatches(*aromaticCarbon), 1U);

  library.finalize();
  EXPECT_EQ(library.size(), 3U);
  EXPECT_EQ(library.pendingSize(), 0U);
  EXPECT_EQ(library.getMatches(*aromaticCarbon), std::vector<MoleculeId>({1U, 2U}));

  library.finalize();
  EXPECT_EQ(library.size(), 3U);
  EXPECT_EQ(library.getMatches(*aromaticCarbon), std::vector<MoleculeId>({1U, 2U}));
}

TEST(SubstructLibraryState, BulkAddPreservesStableIdsAcrossGenerations) {
  nvMolKit::SubstructSearchConfig config;
  config.preprocessingThreads = 4;
  nvMolKit::SubstructLibrary                 library(2, config);
  std::vector<std::unique_ptr<RDKit::ROMol>> targets;
  std::vector<const RDKit::ROMol*>           pointers;
  for (const auto& smiles : {"CC", "O", "CCC", "N", "c1ccccc1"}) {
    targets.push_back(molFromSmiles(smiles));
    ASSERT_NE(targets.back(), nullptr);
    pointers.push_back(targets.back().get());
  }

  EXPECT_EQ(library.addMols(pointers), std::vector<MoleculeId>({0U, 1U, 2U, 3U, 4U}));
  EXPECT_EQ(library.pendingSize(), 5U);
  library.finalize();
  EXPECT_EQ(library.size(), 5U);

  auto extra = molFromSmiles("CO");
  ASSERT_NE(extra, nullptr);
  EXPECT_EQ(library.addMol(*extra), 5U);
  library.finalize();
  auto query = queryFromSmarts("[#6]");
  ASSERT_NE(query, nullptr);
  EXPECT_EQ(library.getMatches(*query), std::vector<MoleculeId>({0U, 2U, 4U, 5U}));
}

TEST(SubstructLibraryResults, PreservesInsertionOrderAndAppliesExactLimits) {
  nvMolKit::SubstructLibrary                 library(2);
  std::vector<std::unique_ptr<RDKit::ROMol>> targets;
  for (const auto& smiles : {"CC", "O", "CCC", "N", "c1ccccc1", "CO"}) {
    targets.push_back(molFromSmiles(smiles));
    ASSERT_NE(targets.back(), nullptr);
    EXPECT_EQ(library.addMol(*targets.back()), targets.size() - 1);
  }
  auto query = queryFromSmarts("[#6]");
  ASSERT_NE(query, nullptr);
  library.finalize();

  const std::vector<MoleculeId> allExpected{0U, 2U, 4U, 5U};
  EXPECT_EQ(library.getMatches(*query), allExpected);
  EXPECT_EQ(library.getMatches(*query, 0), std::vector<MoleculeId>{});
  EXPECT_EQ(library.getMatches(*query, 1), std::vector<MoleculeId>({0U}));
  EXPECT_EQ(library.getMatches(*query, 3), std::vector<MoleculeId>({0U, 2U, 4U}));
  EXPECT_EQ(library.getMatches(*query, 20), allExpected);
  EXPECT_EQ(library.countMatches(*query), allExpected.size());
  EXPECT_TRUE(library.hasMatch(*query));

  auto noMatch = queryFromSmarts("[Si]");
  ASSERT_NE(noMatch, nullptr);
  EXPECT_TRUE(library.getMatches(*noMatch).empty());
  EXPECT_EQ(library.countMatches(*noMatch), 0U);
  EXPECT_FALSE(library.hasMatch(*noMatch));
}

TEST(SubstructLibraryResults, MatchesRDKitAcrossRepresentativeQuerySemantics) {
  nvMolKit::SubstructLibrary                 library(3);
  std::vector<std::unique_ptr<RDKit::ROMol>> targets;
  for (const auto& smiles :
       {"CCO", "CC(=O)C", "c1ccccc1", "C1CCCCC1", "C[N+](C)(C)C", "CC(=O)[O-]", "[Na+].[Cl-]", "CCOC(=O)c1ccccc1O"}) {
    targets.push_back(molFromSmiles(smiles));
    ASSERT_NE(targets.back(), nullptr);
    EXPECT_EQ(library.addMol(*targets.back()), targets.size() - 1);
  }
  library.finalize();

  for (const auto& smarts :
       {"[#6]", "C=O", "c1ccccc1", "[R]", "[N+]", "[#8;H1]", "[$([CX3]=[OX1])]", "[Cl-]", "[Si]"}) {
    auto query = queryFromSmarts(smarts);
    ASSERT_NE(query, nullptr);
    const auto expected = rdkitMatchingIds(targets, *query);

    EXPECT_EQ(library.getMatches(*query), expected) << "SMARTS: " << smarts;
    EXPECT_EQ(library.countMatches(*query), expected.size()) << "SMARTS: " << smarts;
    EXPECT_EQ(library.hasMatch(*query), !expected.empty()) << "SMARTS: " << smarts;
  }
}

TEST(SubstructLibraryOwnership, DoesNotDependOnInputMoleculeLifetime) {
  nvMolKit::SubstructLibrary library(1);
  {
    auto temporary = molFromSmiles("CC(=O)Oc1ccccc1C(=O)O");
    ASSERT_NE(temporary, nullptr);
    EXPECT_EQ(library.addMol(*temporary), 0U);
  }

  auto query = queryFromSmarts("C(=O)O");
  ASSERT_NE(query, nullptr);
  library.finalize();
  EXPECT_EQ(library.getMatches(*query), std::vector<MoleculeId>({0U}));
}

TEST(SubstructLibraryFallback, HandlesTargetsOutsideGpuPackingLimits) {
  nvMolKit::SubstructLibrary library(1);
  auto                       maximumSizedTarget = molFromSmiles(linearAlkane(128));
  auto                       oversizedTarget    = molFromSmiles(linearAlkane(129));
  auto                       highDegreeTarget   = highDegreeMolecule();
  auto                       query              = queryFromSmarts("CCCC");

  ASSERT_NE(maximumSizedTarget, nullptr);
  ASSERT_NE(oversizedTarget, nullptr);
  ASSERT_NE(highDegreeTarget, nullptr);
  ASSERT_GT(highDegreeTarget->getAtomWithIdx(0)->getDegree(), 8U);
  ASSERT_NE(query, nullptr);
  EXPECT_EQ(library.addMol(*maximumSizedTarget), 0U);
  EXPECT_EQ(library.addMol(*oversizedTarget), 1U);
  EXPECT_EQ(library.addMol(*highDegreeTarget), 2U);
  library.finalize();

  EXPECT_EQ(library.getMatches(*query), std::vector<MoleculeId>({0U, 1U}));
  EXPECT_EQ(library.countMatches(*query), 2U);
  EXPECT_TRUE(library.hasMatch(*query));

  auto ironQuery = queryFromSmarts("[Fe]");
  ASSERT_NE(ironQuery, nullptr);
  EXPECT_EQ(library.getMatches(*ironQuery), std::vector<MoleculeId>({2U}));
}

TEST(SubstructLibraryConfiguration, SupportsProductionBackendsAndRejectsInvalidConstruction) {
  EXPECT_THROW(nvMolKit::SubstructLibrary(0), std::invalid_argument);

  nvMolKit::SubstructSearchConfig vf2Config;
  vf2Config.algorithm = nvMolKit::SubstructAlgorithm::VF2;
  EXPECT_THROW(nvMolKit::SubstructLibrary(8, vf2Config), std::invalid_argument);

  nvMolKit::SubstructSearchConfig duplicateGpuConfig;
  duplicateGpuConfig.gpuIds = {0, 0};
  EXPECT_THROW(nvMolKit::SubstructLibrary(8, duplicateGpuConfig), std::invalid_argument);

  nvMolKit::SubstructSearchConfig invalidGpuConfig;
  invalidGpuConfig.gpuIds = {std::numeric_limits<int>::max()};
  nvMolKit::SubstructLibrary invalidGpuLibrary(8, invalidGpuConfig);
  EXPECT_THROW(invalidGpuLibrary.finalize(), std::invalid_argument);

  auto target = molFromSmiles("CCOC(=O)C");
  auto query  = queryFromSmarts("C=O");
  ASSERT_NE(target, nullptr);
  ASSERT_NE(query, nullptr);

  for (const auto algorithm : {nvMolKit::SubstructAlgorithm::GSI, nvMolKit::SubstructAlgorithm::DFS}) {
    nvMolKit::SubstructSearchConfig config;
    config.algorithm            = algorithm;
    config.workerThreads        = 1;
    config.preprocessingThreads = 1;
    nvMolKit::SubstructLibrary library(2, config);
    EXPECT_EQ(library.addMol(*target), 0U);
    library.finalize();
    EXPECT_EQ(library.getMatches(*query), std::vector<MoleculeId>({0U}));
  }
}

TEST(SubstructLibraryMultiGpu, ShardsTargetsAndMergesEveryOperationInInsertionOrder) {
  int deviceCount = 0;
  ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
  if (deviceCount < 2) {
    GTEST_SKIP() << "Multi-GPU library test requires at least two CUDA devices";
  }

  nvMolKit::SubstructSearchConfig config;
  config.gpuIds               = {0, 1};
  config.workerThreads        = 1;
  config.preprocessingThreads = 4;
  nvMolKit::SubstructLibrary                 library(2, config);
  std::vector<std::unique_ptr<RDKit::ROMol>> targets;
  std::vector<const RDKit::ROMol*>           pointers;
  for (const auto& smiles : {"CC", "O", "CCC", "N", "c1ccccc1", "CO", "C=O", "[Si]"}) {
    targets.push_back(molFromSmiles(smiles));
    ASSERT_NE(targets.back(), nullptr);
    pointers.push_back(targets.back().get());
  }
  EXPECT_EQ(library.addMols(pointers), std::vector<MoleculeId>({0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U}));
  library.finalize();

  auto carbon = queryFromSmarts("[#6]");
  ASSERT_NE(carbon, nullptr);
  const std::vector<MoleculeId> expected{0U, 2U, 4U, 5U, 6U};
  EXPECT_EQ(library.getMatches(*carbon), expected);
  EXPECT_EQ(library.getMatches(*carbon, 3), std::vector<MoleculeId>({0U, 2U, 4U}));
  EXPECT_EQ(library.countMatches(*carbon), expected.size());
  EXPECT_TRUE(library.hasMatch(*carbon));

  auto phosphorus = queryFromSmarts("[P]");
  ASSERT_NE(phosphorus, nullptr);
  EXPECT_TRUE(library.getMatches(*phosphorus).empty());
  EXPECT_EQ(library.countMatches(*phosphorus), 0U);
  EXPECT_FALSE(library.hasMatch(*phosphorus));
}

TEST(SubstructLibraryConcurrency, AllowsConcurrentQueriesOfACommittedGeneration) {
  nvMolKit::SubstructSearchConfig config;
  config.workerThreads        = 1;
  config.preprocessingThreads = 1;
  nvMolKit::SubstructLibrary library(2, config);
  for (const auto& smiles : {"CCO", "c1ccccc1", "CC(=O)O", "N"}) {
    auto target = molFromSmiles(smiles);
    ASSERT_NE(target, nullptr);
    EXPECT_LT(library.addMol(*target), 4U);
  }
  library.finalize();

  auto carbonyl = queryFromSmarts("C=O");
  auto aromatic = queryFromSmarts("c");
  ASSERT_NE(carbonyl, nullptr);
  ASSERT_NE(aromatic, nullptr);

  auto carbonylFuture = std::async(std::launch::async, [&] { return library.getMatches(*carbonyl); });
  auto aromaticFuture = std::async(std::launch::async, [&] { return library.getMatches(*aromatic); });
  EXPECT_EQ(carbonylFuture.get(), std::vector<MoleculeId>({2U}));
  EXPECT_EQ(aromaticFuture.get(), std::vector<MoleculeId>({1U}));
}

TEST(SubstructLibraryStreams, SupportsFinalizeAndQueriesOnANondefaultStream) {
  nvMolKit::ScopedStream     stream("SubstructLibraryStreams");
  nvMolKit::SubstructLibrary library(2);
  auto                       target = molFromSmiles("CCOC(=O)C");
  auto                       query  = queryFromSmarts("C=O");
  ASSERT_NE(target, nullptr);
  ASSERT_NE(query, nullptr);

  EXPECT_EQ(library.addMol(*target), 0U);
  library.finalize(stream.stream());
  EXPECT_EQ(library.getMatches(*query, -1, stream.stream()), std::vector<MoleculeId>({0U}));
  EXPECT_EQ(library.countMatches(*query, stream.stream()), 1U);
  EXPECT_TRUE(library.hasMatch(*query, stream.stream()));
}

}  // namespace
