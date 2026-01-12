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
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <climits>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "device.h"
#include "graph_labeler.cuh"
#include "substructure_search.cuh"
#include "test_utils.h"
#include "testutils/mol_data.h"
#include "testutils/substruct_validation.h"

using nvMolKit::countCudaDevices;

using nvMolKit::algorithmName;
using nvMolKit::getSubstructMatches;
using nvMolKit::printValidationResultDetailed;
using nvMolKit::ScopedStream;
using nvMolKit::SubstructAlgorithm;
using nvMolKit::SubstructSearchConfig;
using nvMolKit::SubstructSearchResults;
using nvMolKit::testing::readSmartsFileWithStrings;
using nvMolKit::testing::readSmilesFileWithStrings;
using nvMolKit::validateAgainstRDKit;

std::vector<const RDKit::ROMol*> getRawPtrs(const std::vector<std::unique_ptr<RDKit::ROMol>>& mols) {
  std::vector<const RDKit::ROMol*> ptrs;
  ptrs.reserve(mols.size());
  for (const auto& m : mols) {
    ptrs.push_back(m.get());
  }
  return ptrs;
}

namespace {

constexpr size_t kMaxAtoms  = 128;
constexpr size_t kNumSmiles = 300;

std::unique_ptr<RDKit::ROMol> makeSmartsQuery(const std::string& smarts) {
  auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
  EXPECT_NE(mol, nullptr) << "Failed to parse SMARTS: " << smarts;
  return mol;
}

struct DatasetConfig {
  const char* smartsFile;
  const char* name;
};

struct ThreadingConfig {
  nvMolKit::SubstructSearchConfig config;
  const char*                     name;
};

std::vector<int> getAllGpuIds() {
  const int numDevices = countCudaDevices();
  std::vector<int> ids;
  ids.reserve(numDevices);
  for (int i = 0; i < numDevices; ++i) {
    ids.push_back(i);
  }
  return ids;
}

const ThreadingConfig kThreadingConfigs[] = {
  {nvMolKit::SubstructSearchConfig{1024, 100, 1, 1, 0}, "SingleThreaded"},
  {nvMolKit::SubstructSearchConfig{1024, 100, 2, 4, 0}, "MultiThreaded"},
  {nvMolKit::SubstructSearchConfig{1024, 100, -1, -1, -1}, "Autoselect"},
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

std::vector<std::vector<int>> extractGpuMatches(const SubstructSearchResults& results,
                                                int                           targetIdx,
                                                int                           queryIdx,
                                                int /* numQueryAtoms */) {
  return results.getMatches(targetIdx, queryIdx);
}

void printSmallestRepro(const char*                                       label,
                        const SmallestRepro&                              r,
                        const std::vector<std::string>&                   targetSmiles,
                        const std::vector<std::string>&                   querySmarts,
                        const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                        const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                        const SubstructSearchResults&                     gpuResults) {
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
                         const SubstructSearchResults&                     gpuResults,
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

}  // namespace

using SubstructParams = std::tuple<SubstructAlgorithm, DatasetConfig, ThreadingConfig>;

class SubstructureIntegrationTest : public ::testing::TestWithParam<SubstructParams> {
 protected:
  ScopedStream stream_;
  std::string  testDataPath_;

  void SetUp() override { testDataPath_ = getTestDataFolderPath(); }

  SubstructAlgorithm algorithm() const { return std::get<0>(GetParam()); }
  const DatasetConfig& dataset() const { return std::get<1>(GetParam()); }
  const ThreadingConfig& threading() const { return std::get<2>(GetParam()); }
};

INSTANTIATE_TEST_SUITE_P(
  AllCombinations,
  SubstructureIntegrationTest,
  ::testing::Combine(
    ::testing::Values(SubstructAlgorithm::GSI),
    ::testing::ValuesIn(kDatasets),
    ::testing::ValuesIn(kThreadingConfigs)),
  [](const ::testing::TestParamInfo<SubstructParams>& info) {
    return std::string(algorithmName(std::get<0>(info.param))) + "_" +
           std::get<1>(info.param).name + "_" +
           std::get<2>(info.param).name;
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

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(),
                      stream_.stream(), threading().config);

  EXPECT_EQ(results.numTargets, static_cast<int>(targetMols.size()));
  EXPECT_EQ(results.numQueries, static_cast<int>(queryMols.size()));

  const int numTargets = results.numTargets;
  const int numQueries = results.numQueries;

  std::vector<int64_t> totalMatchesPerQuery(numQueries, 0);
  int64_t              grandTotalMatches = 0;

  for (int q = 0; q < numQueries; ++q) {
    for (int t = 0; t < numTargets; ++t) {
      totalMatchesPerQuery[q] += results.matchCount(t, q);
    }
    grandTotalMatches += totalMatchesPerQuery[q];
  }

  std::vector<int> zeroMatchQueries;
  for (int q = 0; q < numQueries; ++q) {
    if (totalMatchesPerQuery[q] == 0) {
      zeroMatchQueries.push_back(q);
    }
  }

  const int numGpus = threading().config.gpuIds.empty() ? 1 : static_cast<int>(threading().config.gpuIds.size());
  std::cout << "[" << algorithmName(algorithm()) << ", " << threading().name << "] Query statistics:\n"
            << "  Threading: workerThreads=" << threading().config.workerThreads
            << ", preprocessingThreads=" << threading().config.preprocessingThreads
            << ", rdkitFallbackThreads=" << threading().config.rdkitFallbackThreads
            << ", " << numGpus << " GPU(s)\n"
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

  auto validationResult = validateAgainstRDKit(results, targetMols, queryMols);

  if (!validationResult.allMatch) {
    printValidationResultDetailed(validationResult, results, targetMols, queryMols, targetSmiles, querySmarts,
                                  algorithmName(algorithm()));

    if (!validationResult.mismatches.empty()) {
      auto repros = findSmallestRepros(
          validationResult.mismatches,
          [](const auto& m) { return std::get<0>(m); },
          [](const auto& m) { return std::get<1>(m); });
      printSmallestRepros(repros, targetSmiles, querySmarts, targetMols, queryMols, results, "count mismatch");
    }

    if (!validationResult.mappingMismatches.empty()) {
      auto repros = findSmallestRepros(
          validationResult.mappingMismatches,
          [](const auto& m) { return m.first; },
          [](const auto& m) { return m.second; });
      printSmallestRepros(repros, targetSmiles, querySmarts, targetMols, queryMols, results, "mapping mismatch");
    }
  }

  EXPECT_TRUE(validationResult.allMatch)
    << "GPU results do not match RDKit for algorithm " << algorithmName(algorithm())
    << ". Count mismatches: " << validationResult.mismatchedPairs
    << ", Mapping mismatches: " << validationResult.wrongMappingPairs
    << " / " << validationResult.totalPairs << " total pairs";
}

// =============================================================================
// Multi-GPU Tests
// =============================================================================

class MultiGpuSubstructTest : public ::testing::Test {
 protected:
  ScopedStream stream_;
  std::string  testDataPath_;

  void SetUp() override {
    testDataPath_ = getTestDataFolderPath();
    const int numDevices = countCudaDevices();
    if (numDevices < 2) {
      GTEST_SKIP() << "Multi-GPU test requires at least 2 GPUs, found " << numDevices;
    }
  }
};

TEST_F(MultiGpuSubstructTest, MultiGpuMatchesSingleGpu) {
  const std::string smilesPath = testDataPath_ + "/chembl_1k.smi";
  const std::string smartsPath = testDataPath_ + "/SMARTS/rdkit_fragment_descriptors_supported.txt";

  ASSERT_TRUE(std::filesystem::exists(smilesPath)) << "SMILES file not found: " << smilesPath;
  ASSERT_TRUE(std::filesystem::exists(smartsPath)) << "SMARTS file not found: " << smartsPath;

  auto [targetMols, targetSmiles] = readSmilesFileWithStrings(smilesPath, kNumSmiles, kMaxAtoms);
  auto [queryMols, querySmarts]   = readSmartsFileWithStrings(smartsPath);

  ASSERT_FALSE(targetMols.empty()) << "No target molecules loaded";
  ASSERT_FALSE(queryMols.empty()) << "No query patterns loaded";

  auto targetPtrs = getRawPtrs(targetMols);
  auto queryPtrs  = getRawPtrs(queryMols);

  // Run single-GPU
  SubstructSearchConfig singleGpuConfig;
  singleGpuConfig.batchSize     = 1024;
  singleGpuConfig.workerThreads = 2;
  
  SubstructSearchResults singleGpuResults;
  getSubstructMatches(targetPtrs, queryPtrs, singleGpuResults, SubstructAlgorithm::GSI,
                      stream_.stream(), singleGpuConfig);

  // Run multi-GPU
  SubstructSearchConfig multiGpuConfig;
  multiGpuConfig.batchSize     = 1024;
  multiGpuConfig.workerThreads = 2;
  multiGpuConfig.gpuIds        = getAllGpuIds();

  SubstructSearchResults multiGpuResults;
  getSubstructMatches(targetPtrs, queryPtrs, multiGpuResults, SubstructAlgorithm::GSI,
                      stream_.stream(), multiGpuConfig);

  const int numGpus = static_cast<int>(multiGpuConfig.gpuIds.size());
  std::cout << "[MultiGPU] Using " << numGpus << " GPUs with " 
            << multiGpuConfig.workerThreads << " workers each\n";

  // Compare results
  EXPECT_EQ(singleGpuResults.numTargets, multiGpuResults.numTargets);
  EXPECT_EQ(singleGpuResults.numQueries, multiGpuResults.numQueries);

  int64_t singleGpuTotal = 0;
  int64_t multiGpuTotal  = 0;
  int     mismatches     = 0;

  for (int t = 0; t < singleGpuResults.numTargets; ++t) {
    for (int q = 0; q < singleGpuResults.numQueries; ++q) {
      const int singleCount = singleGpuResults.matchCount(t, q);
      const int multiCount  = multiGpuResults.matchCount(t, q);
      singleGpuTotal += singleCount;
      multiGpuTotal += multiCount;
      if (singleCount != multiCount) {
        ++mismatches;
      }
    }
  }

  std::cout << "[MultiGPU] Single-GPU total matches: " << singleGpuTotal << "\n";
  std::cout << "[MultiGPU] Multi-GPU total matches: " << multiGpuTotal << "\n";
  std::cout << "[MultiGPU] Mismatched pairs: " << mismatches << "\n";

  EXPECT_EQ(singleGpuTotal, multiGpuTotal) << "Total match counts differ between single and multi-GPU";
  EXPECT_EQ(mismatches, 0) << "Some pairs have different match counts";
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
