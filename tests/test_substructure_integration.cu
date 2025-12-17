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

#include <GraphMol/QueryAtom.h>
#include <GraphMol/ROMol.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "cuda_error_check.h"
#include "device.h"
#include "flat_bit_vect.h"
#include "graph_labeler.cuh"
#include "molecules_device.cuh"
#include "substructure_search.cuh"
#include "test_utils.h"
#include "testutils/mol_data.h"
#include "testutils/substruct_validation.h"

using nvMolKit::addQueryToBatch;
using nvMolKit::addToBatch;
using nvMolKit::algorithmName;
using nvMolKit::AsyncDeviceVector;
using nvMolKit::BitMatrix2DView;
using nvMolKit::checkReturnCode;
using nvMolKit::FlatBitVect;
using nvMolKit::getSubstructMatches;
using nvMolKit::kMaxQueryAtoms;
using nvMolKit::kMaxTargetAtoms;
using nvMolKit::MoleculesDevice;
using nvMolKit::MoleculesHost;
using nvMolKit::printValidationResultDetailed;
using nvMolKit::ScopedStream;
using nvMolKit::SubstructAlgorithm;
using nvMolKit::SubstructMatchResultsDevice;
using nvMolKit::SubstructMatchResultsHost;
using nvMolKit::testing::readSmartsFileWithStrings;
using nvMolKit::testing::readSmilesFileWithStrings;
using nvMolKit::validateAgainstRDKit;

namespace {

constexpr size_t kMaxAtoms  = 128;
constexpr size_t kNumSmiles = 100;

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

  const int numTargets = resultsHost.numTargets;
  const int numQueries = resultsHost.numQueries;

  std::vector<int64_t> totalMatchesPerQuery(numQueries, 0);
  int64_t              grandTotalMatches = 0;

  for (int q = 0; q < numQueries; ++q) {
    for (int t = 0; t < numTargets; ++t) {
      const int pairIdx = t * numQueries + q;
      totalMatchesPerQuery[q] += resultsHost.matchCounts[pairIdx];
    }
    grandTotalMatches += totalMatchesPerQuery[q];
  }

  std::vector<int> zeroMatchQueries;
  for (int q = 0; q < numQueries; ++q) {
    if (totalMatchesPerQuery[q] == 0) {
      zeroMatchQueries.push_back(q);
    }
  }

  std::cout << "[" << algorithmName(algorithm()) << "] Query statistics:\n"
            << "  Total queries: " << numQueries << "\n"
            << "  Total targets: " << numTargets << "\n"
            << "  Grand total matches: " << grandTotalMatches << "\n"
            << "  Queries with 0 matches: " << zeroMatchQueries.size() << "\n";

  if (!zeroMatchQueries.empty()) {
    std::cout << "  Zero-match queries:\n";
    const size_t maxToShow = 20;
    for (size_t i = 0; i < std::min(zeroMatchQueries.size(), maxToShow); ++i) {
      const int q = zeroMatchQueries[i];
      std::cout << "    [" << q << "]: " << querySmarts[q] << "\n";
    }
    if (zeroMatchQueries.size() > maxToShow) {
      std::cout << "    ... and " << (zeroMatchQueries.size() - maxToShow) << " more\n";
    }
  }

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

// =============================================================================
// Label Matrix Integration Test (not parameterized - label matrix is shared)
// =============================================================================

using LabelMatrixStorage = FlatBitVect<kMaxTargetAtoms * kMaxQueryAtoms>;
using LabelMatrixView    = BitMatrix2DView<kMaxTargetAtoms, kMaxQueryAtoms>;

template <std::size_t MaxTarget, std::size_t MaxQuery>
__global__ void populateLabelMatrixKernelForIntegration(nvMolKit::MoleculesDeviceView targetsView,
                                                        int                          targetIdx,
                                                        nvMolKit::MoleculesDeviceView queriesView,
                                                        int                          queryIdx,
                                                        LabelMatrixStorage*          output) {
  nvMolKit::MoleculeView target = nvMolKit::getMolecule(targetsView, targetIdx);
  nvMolKit::MoleculeView query  = nvMolKit::getMolecule(queriesView, queryIdx);

  BitMatrix2DView<MaxTarget, MaxQuery> view(*output);
  nvMolKit::populateLabelMatrixOptimized<MaxTarget, MaxQuery>(target, query, view);
}

class LabelMatrixIntegrationTest : public ::testing::Test {
 protected:
  ScopedStream stream_;
  std::string  testDataPath_;

  void SetUp() override { testDataPath_ = getTestDataFolderPath(); }
};

TEST_F(LabelMatrixIntegrationTest, ChemblVsAlertCollectionLabelMatrix) {
  const std::string smilesPath = testDataPath_ + "/chembl_1k.smi";
  const std::string smartsPath = testDataPath_ + "/SMARTS/pwalters_alert_collection_supported.txt";

  ASSERT_TRUE(std::filesystem::exists(smilesPath)) << "SMILES file not found: " << smilesPath;
  ASSERT_TRUE(std::filesystem::exists(smartsPath)) << "SMARTS file not found: " << smartsPath;

  auto [targetMols, targetSmiles] = readSmilesFileWithStrings(smilesPath, kNumSmiles, kMaxAtoms);
  auto [queryMols, querySmarts]   = readSmartsFileWithStrings(smartsPath);

  ASSERT_FALSE(targetMols.empty()) << "No target molecules loaded";
  ASSERT_FALSE(queryMols.empty()) << "No query patterns loaded";

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

  AsyncDeviceVector<LabelMatrixStorage> matrixDev(1, stream_.stream());

  const int numTargets = static_cast<int>(targetMols.size());
  const int numQueries = static_cast<int>(queryMols.size());

  int totalPairs = 0;
  int totalMismatches = 0;
  int totalFalsePositives = 0;
  int totalFalseNegatives = 0;

  std::vector<std::tuple<int, int, int, int>> fpPairs;
  std::vector<std::tuple<int, int, int, int, int, int, bool, bool>> fpDetails;

  for (int t = 0; t < numTargets; ++t) {
    for (int q = 0; q < numQueries; ++q) {
      ++totalPairs;

      LabelMatrixStorage hostMatrix(false);
      matrixDev.setFromVector(std::vector<LabelMatrixStorage>{hostMatrix});

      populateLabelMatrixKernelForIntegration<kMaxTargetAtoms, kMaxQueryAtoms>
        <<<1, 128, 0, stream_.stream()>>>(targetsDevice.view(), t, queriesDevice.view(), q, matrixDev.data());
      cudaCheckError(cudaGetLastError());

      std::vector<LabelMatrixStorage> resultMatrix(1);
      matrixDev.copyToHost(resultMatrix);
      cudaCheckError(cudaStreamSynchronize(stream_.stream()));

      LabelMatrixView view(resultMatrix[0]);

      const int numTargetAtoms = static_cast<int>(targetMols[t]->getNumAtoms());
      const int numQueryAtoms  = static_cast<int>(queryMols[q]->getNumAtoms());

      int pairMismatches = 0;
      int pairFalsePositives = 0;
      int pairFalseNegatives = 0;

      for (int ta = 0; ta < numTargetAtoms; ++ta) {
        const auto* targetAtom = targetMols[t]->getAtomWithIdx(ta);
        for (int qa = 0; qa < numQueryAtoms; ++qa) {
          const auto* queryAtom = queryMols[q]->getAtomWithIdx(qa);

          bool rdkitResult = false;
          if (queryAtom->hasQuery()) {
            rdkitResult = queryAtom->Match(targetAtom);
          } else {
            rdkitResult = (targetAtom->getAtomicNum() == queryAtom->getAtomicNum());
          }

          bool gpuResult = view.get(ta, qa);

          if (gpuResult && !rdkitResult) {
            ++pairFalsePositives;
            ++pairMismatches;
            if (fpDetails.size() < 20) {
              fpDetails.push_back({t, q, ta, qa,
                                   targetAtom->getAtomicNum(),
                                   queryAtom->getAtomicNum(),
                                   targetAtom->getIsAromatic(),
                                   queryAtom->getIsAromatic()});
            }
          } else if (!gpuResult && rdkitResult) {
            ++pairFalseNegatives;
            ++pairMismatches;
          }
        }
      }

      totalMismatches += pairMismatches;
      totalFalsePositives += pairFalsePositives;
      totalFalseNegatives += pairFalseNegatives;

      if (pairFalsePositives > 0 && fpPairs.size() < 10) {
        fpPairs.emplace_back(t, q, pairFalsePositives, pairFalseNegatives);
      }
    }
  }

  std::cout << "Label matrix integration test:\n"
            << "  Total pairs tested: " << totalPairs << "\n"
            << "  Total atom-level mismatches: " << totalMismatches << "\n"
            << "  False positives (GPU yes, RDKit no): " << totalFalsePositives << "\n"
            << "  False negatives (GPU no, RDKit yes): " << totalFalseNegatives << "\n";

  if (!fpPairs.empty()) {
    std::cout << "  Pairs with FALSE POSITIVES:\n";
    for (const auto& [t, q, fp, fn] : fpPairs) {
      std::cout << "    Target[" << t << "] x Query[" << q << "]: "
                << fp << " false positives\n"
                << "      Target: " << targetSmiles[t].substr(0, 80) << "...\n"
                << "      Query: " << querySmarts[q] << "\n";
    }
  }

  if (!fpDetails.empty()) {
    std::cout << "  False positive atom details:\n";
    for (const auto& [t, q, ta, qa, tAtomNum, qAtomNum, tArom, qArom] : fpDetails) {
      std::cout << "    T[" << t << "].atom" << ta << " (Z=" << tAtomNum << ",arom=" << tArom << ")"
                << " vs Q[" << q << "].atom" << qa << " (Z=" << qAtomNum << ",arom=" << qArom << ")\n";
    }
  }

  EXPECT_EQ(totalFalsePositives, 0)
    << "GPU label matrix has false positives (marks atoms as compatible when RDKit says no)";
}
