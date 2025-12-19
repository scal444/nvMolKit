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
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <climits>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
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

std::unique_ptr<RDKit::ROMol> makeSmartsQuery(const std::string& smarts) {
  auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
  EXPECT_NE(mol, nullptr) << "Failed to parse SMARTS: " << smarts;
  return mol;
}

struct DatasetConfig {
  const char* smartsFile;
  const char* name;
};

constexpr DatasetConfig kDatasets[] = {
  {"pwalters_alert_collection_supported.txt", "PwaltersAlertCollection"},
  {"openbabel_functional_groups_supported.txt", "OpenBabelFunctionalGroups"},
  {"BMS_2006_filter_supported.txt", "BMS2006Filter"},
  {"rdkit_fragment_descriptors_supported.txt", "RDKitFragmentDescriptors"},
  {"rdkit_tautomer_transforms_supported.txt", "RDKitTautomerTransforms"},
  {"rdkit_torsionPreferences_v2_supported.txt", "RDKitTorsionPreferencesV2"},
  {"rdkit_torsionPreferences_smallrings_supported.txt", "RDKitTorsionPreferencesSmallRings"},
  {"rdkit_pattern_fingerprint_supported.txt", "RDKitPatternFingerprints"},
  {"rdkit_torsionPreferences_macrocycles_supported.txt", "RDKitTorsionPreferencesMacrocycles"},
  {"RLewis_smarts_supported.txt", "RLewisSMARTS"},
  {"wehi_pains_supported.txt", "WEHIPAINS"},
};

struct SmallestRepro {
  int t = -1;
  int q = -1;
};

struct SmallestRepros {
  SmallestRepro smallestSum;
  SmallestRepro smallestQ;
  SmallestRepro smallestT;

  bool allSame() const {
    return smallestSum.t == smallestQ.t && smallestSum.q == smallestQ.q &&
           smallestSum.t == smallestT.t && smallestSum.q == smallestT.q;
  }

  bool hasAny() const { return smallestSum.t >= 0; }
};

template <typename PairContainer, typename GetT, typename GetQ>
SmallestRepros findSmallestRepros(const PairContainer& pairs, GetT getT, GetQ getQ) {
  SmallestRepros result;
  int            minSum = INT_MAX;
  int            minQ   = INT_MAX;
  int            minT   = INT_MAX;

  for (const auto& pair : pairs) {
    const int t   = getT(pair);
    const int q   = getQ(pair);
    const int sum = t + q;

    if (sum < minSum) {
      minSum              = sum;
      result.smallestSum  = {t, q};
    }
    if (q < minQ) {
      minQ              = q;
      result.smallestQ  = {t, q};
    }
    if (t < minT) {
      minT              = t;
      result.smallestT  = {t, q};
    }
  }
  return result;
}

void printMatches(const std::string& label, const std::vector<std::vector<int>>& matches) {
  std::cout << "    " << label << ": ";
  if (matches.empty()) {
    std::cout << "(none)\n";
    return;
  }
  std::cout << matches.size() << " match(es)\n";
  for (size_t i = 0; i < matches.size(); ++i) {
    std::cout << "      [" << i << "]: {";
    for (size_t j = 0; j < matches[i].size(); ++j) {
      if (j > 0) std::cout << ", ";
      std::cout << matches[i][j];
    }
    std::cout << "}\n";
  }
}

std::vector<std::vector<int>> extractGpuMatches(const SubstructMatchResultsHost& results,
                                                int                              targetIdx,
                                                int                              queryIdx,
                                                int                              numQueryAtoms) {
  const int pairIdx       = results.pairIndex(targetIdx, queryIdx);
  const int reportedCount = results.reportedCounts[pairIdx];
  const int startOffset   = results.pairMatchStarts[pairIdx];

  std::vector<std::vector<int>> gpuMatches;
  gpuMatches.reserve(reportedCount);

  for (int m = 0; m < reportedCount; ++m) {
    std::vector<int> mapping(numQueryAtoms);
    for (int a = 0; a < numQueryAtoms; ++a) {
      mapping[a] = results.matchIndices[startOffset + m * numQueryAtoms + a];
    }
    gpuMatches.push_back(std::move(mapping));
  }

  return gpuMatches;
}

void printSmallestRepro(const char*                                       label,
                        const SmallestRepro&                              r,
                        const std::vector<std::string>&                   targetSmiles,
                        const std::vector<std::string>&                   querySmarts,
                        const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                        const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                        const SubstructMatchResultsHost&                  gpuResults) {
  std::cout << "  --- " << label << " (t=" << r.t << " q=" << r.q << " sum=" << (r.t + r.q) << ") ---\n";
  std::cout << "  Target[" << r.t << "]: " << targetSmiles[r.t] << "\n";
  std::cout << "  Query[" << r.q << "]:  " << querySmarts[r.q] << "\n";

  auto rdkitMatches = nvMolKit::getRDKitSubstructMatches(*targetMols[r.t], *queryMols[r.q], false);
  printMatches("Expected (RDKit)", rdkitMatches);

  const int numQueryAtoms = static_cast<int>(queryMols[r.q]->getNumAtoms());
  auto      gpuMatches    = extractGpuMatches(gpuResults, r.t, r.q, numQueryAtoms);
  printMatches("Actual (GPU)", gpuMatches);

  std::cout << "\n";
}

void printSmallestRepros(const SmallestRepros&                             repros,
                         const std::vector<std::string>&                   targetSmiles,
                         const std::vector<std::string>&                   querySmarts,
                         const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                         const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                         const SubstructMatchResultsHost&                  gpuResults,
                         const std::string&                                category) {
  if (!repros.hasAny()) return;

  std::cout << "\n=== Smallest " << category << " repros ===\n";

  if (repros.allSame()) {
    printSmallestRepro("smallest (all criteria)", repros.smallestSum, targetSmiles, querySmarts,
                       targetMols, queryMols, gpuResults);
  } else {
    printSmallestRepro("smallest sum (t+q)", repros.smallestSum, targetSmiles, querySmarts,
                       targetMols, queryMols, gpuResults);
    if (repros.smallestQ.t != repros.smallestSum.t || repros.smallestQ.q != repros.smallestSum.q) {
      printSmallestRepro("smallest q", repros.smallestQ, targetSmiles, querySmarts,
                         targetMols, queryMols, gpuResults);
    }
    if (repros.smallestT.t != repros.smallestSum.t || repros.smallestT.q != repros.smallestSum.q) {
      if (repros.smallestT.t != repros.smallestQ.t || repros.smallestT.q != repros.smallestQ.q) {
        printSmallestRepro("smallest t", repros.smallestT, targetSmiles, querySmarts,
                           targetMols, queryMols, gpuResults);
      }
    }
  }
}

void printSmallestReproSimple(const char*                     label,
                              const SmallestRepro&            r,
                              const std::vector<std::string>& targetSmiles,
                              const std::vector<std::string>& querySmarts,
                              int                             fp,
                              int                             fn) {
  std::cout << "  --- " << label << " (t=" << r.t << " q=" << r.q << " sum=" << (r.t + r.q) << ") ---\n";
  std::cout << "  Target[" << r.t << "]: " << targetSmiles[r.t] << "\n";
  std::cout << "  Query[" << r.q << "]:  " << querySmarts[r.q] << "\n";
  std::cout << "    FP=" << fp << " FN=" << fn << "\n\n";
}

template <typename GetFPFN>
void printSmallestReprosSimple(const SmallestRepros&           repros,
                               const std::vector<std::string>& targetSmiles,
                               const std::vector<std::string>& querySmarts,
                               const std::string&              category,
                               GetFPFN                         getFPFN) {
  if (!repros.hasAny()) return;

  std::cout << "\n=== Smallest " << category << " repros ===\n";

  auto printOne = [&](const char* label, const SmallestRepro& r) {
    auto [fp, fn] = getFPFN(r.t, r.q);
    printSmallestReproSimple(label, r, targetSmiles, querySmarts, fp, fn);
  };

  if (repros.allSame()) {
    printOne("smallest (all criteria)", repros.smallestSum);
  } else {
    printOne("smallest sum (t+q)", repros.smallestSum);
    if (repros.smallestQ.t != repros.smallestSum.t || repros.smallestQ.q != repros.smallestSum.q) {
      printOne("smallest q", repros.smallestQ);
    }
    if (repros.smallestT.t != repros.smallestSum.t || repros.smallestT.q != repros.smallestSum.q) {
      if (repros.smallestT.t != repros.smallestQ.t || repros.smallestT.q != repros.smallestQ.q) {
        printOne("smallest t", repros.smallestT);
      }
    }
  }
}

}  // namespace

using SubstructParams = std::tuple<SubstructAlgorithm, DatasetConfig>;

class SubstructureIntegrationTest : public ::testing::TestWithParam<SubstructParams> {
 protected:
  ScopedStream stream_;
  std::string  testDataPath_;

  void SetUp() override { testDataPath_ = getTestDataFolderPath(); }

  SubstructAlgorithm algorithm() const { return std::get<0>(GetParam()); }
  const DatasetConfig& dataset() const { return std::get<1>(GetParam()); }
};

INSTANTIATE_TEST_SUITE_P(
  AllCombinations,
  SubstructureIntegrationTest,
  ::testing::Combine(
    ::testing::Values(SubstructAlgorithm::GSI),
    ::testing::ValuesIn(kDatasets)),
  [](const ::testing::TestParamInfo<SubstructParams>& info) {
    return std::string(algorithmName(std::get<0>(info.param))) + "_" +
           std::get<1>(info.param).name;
  });

TEST_P(SubstructureIntegrationTest, ChemblVsSmarts) {
  const std::string smilesPath = testDataPath_ + "/chembl_1k.smi";
  const std::string smartsPath = testDataPath_ + "/SMARTS/" + dataset().smartsFile;

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

    if (!validationResult.mismatches.empty()) {
      auto repros = findSmallestRepros(
        validationResult.mismatches,
        [](const auto& m) { return std::get<0>(m); },
        [](const auto& m) { return std::get<1>(m); });
      printSmallestRepros(repros, targetSmiles, querySmarts, targetMols, queryMols, resultsHost, "count mismatch");
    }

    if (!validationResult.mappingMismatches.empty()) {
      auto repros = findSmallestRepros(
        validationResult.mappingMismatches,
        [](const auto& m) { return m.first; },
        [](const auto& m) { return m.second; });
      printSmallestRepros(repros, targetSmiles, querySmarts, targetMols, queryMols, resultsHost, "mapping mismatch");
    }
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

class LabelMatrixIntegrationTest : public ::testing::TestWithParam<DatasetConfig> {
 protected:
  ScopedStream stream_;
  std::string  testDataPath_;

  void SetUp() override { testDataPath_ = getTestDataFolderPath(); }

  const DatasetConfig& dataset() const { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(
  AllDatasets,
  LabelMatrixIntegrationTest,
  ::testing::ValuesIn(kDatasets),
  [](const ::testing::TestParamInfo<DatasetConfig>& info) {
    return std::string(info.param.name);
  });

TEST_P(LabelMatrixIntegrationTest, ChemblVsSmartsLabelMatrix) {
  const std::string smilesPath = testDataPath_ + "/chembl_1k.smi";
  const std::string smartsPath = testDataPath_ + "/SMARTS/" + dataset().smartsFile;

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

  if (!fpPairs.empty()) {
    std::map<std::pair<int, int>, std::pair<int, int>> fpInfo;
    for (const auto& [t, q, fp, fn] : fpPairs) {
      fpInfo[{t, q}] = {fp, fn};
    }
    auto repros = findSmallestRepros(
      fpPairs,
      [](const auto& p) { return std::get<0>(p); },
      [](const auto& p) { return std::get<1>(p); });
    printSmallestReprosSimple(repros, targetSmiles, querySmarts, "false positive",
      [&](int t, int q) -> std::pair<int, int> {
        auto it = fpInfo.find({t, q});
        return it != fpInfo.end() ? it->second : std::pair{0, 0};
      });
  }

  EXPECT_EQ(totalFalsePositives, 0)
    << "GPU label matrix has false positives (marks atoms as compatible when RDKit says no)";
}

// =============================================================================
// Recursive SMARTS Tests
// =============================================================================

TEST(RecursiveSmartsTest, HasRecursiveSmartsDetection) {
  auto nonRecursive = makeSmartsQuery("[CH3]");
  EXPECT_FALSE(nvMolKit::hasRecursiveSmarts(nonRecursive.get()));

  auto recursive = makeSmartsQuery("[$([OH])]");
  EXPECT_TRUE(nvMolKit::hasRecursiveSmarts(recursive.get()));
}

TEST(RecursiveSmartsTest, ExtractSimplePattern) {
  auto query = makeSmartsQuery("[$([OH])]");
  auto info = nvMolKit::extractRecursivePatterns(query.get());

  EXPECT_EQ(info.size(), 1);
  EXPECT_TRUE(info.hasRecursivePatterns);
}

TEST(RecursiveSmartsTest, ExtractMultiplePatterns) {
  auto query = makeSmartsQuery("[$([OH]),$([NH2])]");
  auto info = nvMolKit::extractRecursivePatterns(query.get());

  EXPECT_EQ(info.size(), 2);
}

TEST(RecursiveSmartsTest, PatternIdsAreSequential) {
  auto query = makeSmartsQuery("[$([C]),$([N]),$([O])]");
  auto info = nvMolKit::extractRecursivePatterns(query.get());

  EXPECT_EQ(info.size(), 3);
  EXPECT_EQ(info.patterns[0].patternId, 0);
  EXPECT_EQ(info.patterns[1].patternId, 1);
  EXPECT_EQ(info.patterns[2].patternId, 2);
}

TEST(RecursiveSmartsTest, TooManyPatternsThrows) {
  std::string smarts = "[C";
  for (int i = 0; i < nvMolKit::RecursivePatternInfo::kMaxPatterns + 1; ++i) {
    smarts += ";$(*-N)";
  }
  smarts += "]";

  auto query = makeSmartsQuery(smarts);
  EXPECT_THROW(nvMolKit::extractRecursivePatterns(query.get()), std::runtime_error);
}
