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

#include <GraphMol/ROMol.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "cuda_error_check.h"
#include "device.h"
#include "substructure_search.cuh"
#include "test_utils.h"
#include "testutils/mol_data.h"
#include "testutils/substruct_validation.h"

using nvMolKit::addQueryToBatch;
using nvMolKit::addToBatch;
using nvMolKit::algorithmName;
using nvMolKit::checkReturnCode;
using nvMolKit::getSubstructMatches;
using nvMolKit::MoleculesDevice;
using nvMolKit::MoleculesHost;
using nvMolKit::ScopedStream;
using nvMolKit::SubstructAlgorithm;
using nvMolKit::SubstructMatchResultsDevice;
using nvMolKit::SubstructMatchResultsHost;
using nvMolKit::testing::readSmartsFileWithStrings;
using nvMolKit::testing::readSmilesFileWithStrings;
using nvMolKit::printValidationResultDetailed;
using nvMolKit::validateAgainstRDKit;

namespace {

constexpr size_t kMaxAtoms  = 128;
constexpr size_t kNumSmiles = 10;

}  // namespace

class SubstructureIntegrationTest : public ::testing::TestWithParam<SubstructAlgorithm> {
 protected:
  ScopedStream stream_;
  std::string  testDataPath_;

  void SetUp() override { testDataPath_ = getTestDataFolderPath(); }

  SubstructAlgorithm algorithm() const { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(AllAlgorithms,
                         SubstructureIntegrationTest,
                         ::testing::Values(
                                           SubstructAlgorithm::WarpUnified, SubstructAlgorithm::GSI),
                         [](const ::testing::TestParamInfo<SubstructAlgorithm>& info) {
                           return algorithmName(info.param);
                         });

TEST_P(SubstructureIntegrationTest, ChemblVsAlertCollection) {
  const std::string smilesPath = testDataPath_ + "/chembl_1k.smi";
  const std::string smartsPath = testDataPath_ + "/SMARTS/pwalters_alert_collection_supported.txt";

  ASSERT_TRUE(std::filesystem::exists(smilesPath)) << "SMILES file not found: " << smilesPath;
  ASSERT_TRUE(std::filesystem::exists(smartsPath)) << "SMARTS file not found: " << smartsPath;

  auto [targetMols, targetSmiles] = readSmilesFileWithStrings(smilesPath, kNumSmiles, kMaxAtoms);
  auto [queryMols, querySmarts]   = readSmartsFileWithStrings(smartsPath);

  ASSERT_FALSE(targetMols.empty()) << "No target molecules loaded";
  ASSERT_FALSE(queryMols.empty()) << "No query patterns loaded";

  ASSERT_LE(targetMols.size(), kNumSmiles) << "Loaded more targets than requested";

  MoleculesHost targetsHost;
  MoleculesHost queriesHost;

  for (const auto& mol : targetMols) {
    addToBatch(mol.get(), targetsHost);
  }

  for (const auto& mol : queryMols) {
    addQueryToBatch(mol.get(), queriesHost);
  }

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost, resultsDevice, algorithm(),
                      stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  EXPECT_EQ(resultsHost.numTargets, static_cast<int>(targetMols.size()));
  EXPECT_EQ(resultsHost.numQueries, static_cast<int>(queryMols.size()));

  auto validationResult = validateAgainstRDKit(resultsHost, targetMols, queryMols);

  if (!validationResult.allMatch) {
    printValidationResultDetailed(validationResult, resultsHost, targetMols, queryMols, targetSmiles, querySmarts,
                                  algorithmName(algorithm()));
  }

  EXPECT_TRUE(validationResult.allMatch)
    << "GPU results do not match RDKit for algorithm " << algorithmName(algorithm())
    << ". Count mismatches: " << validationResult.mismatchedPairs
    << ", Mapping mismatches: " << validationResult.wrongMappingPairs
    << " / " << validationResult.totalPairs << " total pairs";
}
