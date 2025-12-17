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

#include <memory>
#include <set>
#include <vector>

#include "cuda_error_check.h"
#include "device.h"
#include "substructure_search.cuh"
#include "testutils/substruct_validation.h"

using nvMolKit::addQueryToBatch;
using nvMolKit::addToBatch;
using nvMolKit::algorithmName;
using nvMolKit::checkReturnCode;
using nvMolKit::getRDKitSubstructMatches;
using nvMolKit::getSubstructMatches;
using nvMolKit::MoleculesDevice;
using nvMolKit::MoleculesHost;
using nvMolKit::ScopedStream;
using nvMolKit::SubstructAlgorithm;
using nvMolKit::SubstructMatchResultsDevice;
using nvMolKit::SubstructMatchResultsHost;

namespace {

std::unique_ptr<RDKit::ROMol> makeMolFromSmiles(const std::string& smiles) {
  return std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
}

std::unique_ptr<RDKit::ROMol> makeMolFromSmarts(const std::string& smarts) {
  return std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
}

}  // namespace

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class SubstructureSearchTest : public ::testing::TestWithParam<SubstructAlgorithm> {
 protected:
  ScopedStream stream_;

  SubstructAlgorithm algorithm() const { return GetParam(); }

  void SetUp() override {}

  /**
   * @brief Build target and query batches from SMILES/SMARTS strings.
   */
  void buildBatches(const std::vector<std::string>&             targetSmiles,
                    const std::vector<std::string>&             querySmarts,
                    MoleculesHost&                              targetsHost,
                    MoleculesHost&                              queriesHost,
                    std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                    std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols) {
    targetMols.clear();
    queryMols.clear();

    for (const auto& smiles : targetSmiles) {
      auto mol = makeMolFromSmiles(smiles);
      ASSERT_NE(mol, nullptr) << "Failed to parse target SMILES: " << smiles;
      addToBatch(mol.get(), targetsHost);
      targetMols.push_back(std::move(mol));
    }

    for (const auto& smarts : querySmarts) {
      auto mol = makeMolFromSmarts(smarts);
      ASSERT_NE(mol, nullptr) << "Failed to parse query SMARTS: " << smarts;
      addQueryToBatch(mol.get(), queriesHost);
      queryMols.push_back(std::move(mol));
    }
  }

  /**
   * @brief Extract GPU matches for a (target, query) pair from results.
   */
  static std::vector<std::vector<int>> extractGpuMatches(const SubstructMatchResultsHost& results,
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

  /**
   * @brief Compare two sets of matches (order-independent).
   */
  static bool matchSetsEqual(const std::vector<std::vector<int>>& gpuMatches,
                             const std::vector<std::vector<int>>& rdkitMatches) {
    if (gpuMatches.size() != rdkitMatches.size()) {
      return false;
    }

    std::set<std::vector<int>> gpuSet(gpuMatches.begin(), gpuMatches.end());
    std::set<std::vector<int>> rdkitSet(rdkitMatches.begin(), rdkitMatches.end());

    return gpuSet == rdkitSet;
  }

  /**
   * @brief Verify GPU matches RDKit for a single (target, query) pair.
   *
   * Checks both match count AND actual atom mappings.
   */
  void expectMatchesRDKit(const SubstructMatchResultsHost&   results,
                          const RDKit::ROMol&                target,
                          const RDKit::ROMol&                query,
                          int                                targetIdx,
                          int                                queryIdx,
                          const std::string&                 description = "") {
    const auto rdkitMatches    = getRDKitSubstructMatches(target, query, false);
    const int  pairIdx         = results.pairIndex(targetIdx, queryIdx);
    const int  gpuMatchCount   = results.matchCounts[pairIdx];
    const int  rdkitMatchCount = static_cast<int>(rdkitMatches.size());

    std::string context = description.empty() ? "" : " (" + description + ")";

    EXPECT_EQ(gpuMatchCount, rdkitMatchCount)
      << "Match count mismatch" << context << " using " << algorithmName(algorithm())
      << ": GPU=" << gpuMatchCount << ", RDKit=" << rdkitMatchCount;

    if (gpuMatchCount == rdkitMatchCount && gpuMatchCount > 0 && !results.hasOverflow(targetIdx, queryIdx)) {
      const int  numQueryAtoms = static_cast<int>(query.getNumAtoms());
      const auto gpuMatches    = extractGpuMatches(results, targetIdx, queryIdx, numQueryAtoms);

      EXPECT_TRUE(matchSetsEqual(gpuMatches, rdkitMatches))
        << "Match indices mismatch" << context << " using " << algorithmName(algorithm())
        << ": counts match (" << gpuMatchCount << ") but atom mappings differ";
    }
  }

  /**
   * @brief Compare GPU results against RDKit ground truth (with uniquify=false).
   *
   * @param results GPU results
   * @param targetMols Target molecules for RDKit comparison
   * @param queryMols Query molecules for RDKit comparison
   * @param expectMatch If true, expect tests to pass; if false, expect current failures
   * @param allowOverflow If false, assert that no pairs overflowed
   */
  void compareWithRDKit(const SubstructMatchResultsHost&                  results,
                        const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                        const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                        bool                                              expectMatch   = false,
                        bool                                              allowOverflow = false) {
    for (int t = 0; t < results.numTargets; ++t) {
      for (int q = 0; q < results.numQueries; ++q) {
        // Use uniquify=false to match our non-uniquifying GPU algorithm
        const auto rdkitMatches    = getRDKitSubstructMatches(*targetMols[t], *queryMols[q], false);
        const int  pairIdx         = results.pairIndex(t, q);
        const int  gpuMatchCount   = results.matchCounts[pairIdx];
        const int  rdkitMatchCount = static_cast<int>(rdkitMatches.size());

        if (!allowOverflow) {
          EXPECT_FALSE(results.hasOverflow(t, q))
            << "Unexpected overflow for target " << t << ", query " << q << " using algorithm "
            << algorithmName(algorithm()) << ": actual=" << gpuMatchCount
            << ", reported=" << results.reportedCounts[pairIdx];
        }

        if (expectMatch) {
          EXPECT_EQ(gpuMatchCount, rdkitMatchCount)
            << "Match count mismatch for target " << t << ", query " << q << " using algorithm "
            << algorithmName(algorithm()) << ": GPU=" << gpuMatchCount << ", RDKit=" << rdkitMatchCount;

          // Also verify actual match indices if counts match and no overflow
          if (gpuMatchCount == rdkitMatchCount && gpuMatchCount > 0 && !results.hasOverflow(t, q)) {
            const int  numQueryAtoms = static_cast<int>(queryMols[q]->getNumAtoms());
            const auto gpuMatches    = extractGpuMatches(results, t, q, numQueryAtoms);

            EXPECT_TRUE(matchSetsEqual(gpuMatches, rdkitMatches))
              << "Match indices mismatch for target " << t << ", query " << q << " using algorithm "
              << algorithmName(algorithm()) << ": counts match (" << gpuMatchCount
              << ") but atom mappings differ";
          }
        }
      }
    }
  }
};

// Instantiate parameterized tests for all algorithms
INSTANTIATE_TEST_SUITE_P(AllAlgorithms,
                         SubstructureSearchTest,
                         ::testing::Values(SubstructAlgorithm::VF2,
                                           SubstructAlgorithm::GSI,
                                           SubstructAlgorithm::WarpUnified),
                         [](const ::testing::TestParamInfo<SubstructAlgorithm>& info) {
                           return algorithmName(info.param);
                         });

// =============================================================================
// Basic Tests - Run with all algorithms
// =============================================================================

TEST_P(SubstructureSearchTest, SingleTargetSingleQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  buildBatches({"CCO"}, {"C"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  EXPECT_EQ(resultsHost.numTargets, 1);
  EXPECT_EQ(resultsHost.numQueries, 1);

  // Compare with RDKit - expect match once algorithms are working
  compareWithRDKit(resultsHost, targetMols, queryMols, true);
}

TEST_P(SubstructureSearchTest, MultipleTargetsSingleQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  buildBatches({"CCO", "CCCC", "c1ccccc1"}, {"C"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  EXPECT_EQ(resultsHost.numTargets, 3);
  EXPECT_EQ(resultsHost.numQueries, 1);

  compareWithRDKit(resultsHost, targetMols, queryMols, true);
}

TEST_P(SubstructureSearchTest, SingleTargetMultipleQueries) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  buildBatches({"CCO"}, {"C", "O", "CC"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  EXPECT_EQ(resultsHost.numTargets, 1);
  EXPECT_EQ(resultsHost.numQueries, 3);

  compareWithRDKit(resultsHost, targetMols, queryMols, true);
}

TEST_P(SubstructureSearchTest, BatchAllToAll) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Use molecules where non-unique matches won't overflow buffer
  // Buffer size = target atom count. CC query gives 2*(N-1) matches for N-carbon chain.
  // To avoid overflow: need 2*(N-1) <= N, i.e., N >= 2. But also need margin for branching.
  // Use single-atom queries and aromatic targets which have limited matches.
  buildBatches({"CCO", "c1ccccc1", "c1ccc(O)cc1", "CCN"},  // 4 targets: 3, 6, 7, 3 atoms
               {"C", "O", "c", "N"},                        // 4 single-atom queries
               targetsHost,
               queriesHost,
               targetMols,
               queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  EXPECT_EQ(resultsHost.numTargets, 4);
  EXPECT_EQ(resultsHost.numQueries, 4);
  EXPECT_EQ(static_cast<int>(resultsHost.matchCounts.size()), 16);

  compareWithRDKit(resultsHost, targetMols, queryMols, true);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_P(SubstructureSearchTest, NoMatchPossible) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Carbon chain vs nitrogen query - no match possible
  buildBatches({"CCCC"}, {"N"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "CCCC with N query");
}

TEST_P(SubstructureSearchTest, AromaticVsAliphatic) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Benzene (aromatic) vs aliphatic carbon query
  buildBatches({"c1ccccc1"}, {"C"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "benzene with aliphatic C query");
}

TEST_P(SubstructureSearchTest, LargerMolecule) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Caffeine as a larger test case
  buildBatches({"Cn1cnc2c1c(=O)n(c(=O)n2C)C"}, {"c", "N", "C"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  EXPECT_EQ(resultsHost.numTargets, 1);
  EXPECT_EQ(resultsHost.numQueries, 3);

  compareWithRDKit(resultsHost, targetMols, queryMols, true);
}

TEST_P(SubstructureSearchTest, BufferAllocationCorrect) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Different sized molecules to test buffer allocation
  // Use single-atom queries to avoid overflow from non-unique CC matching
  buildBatches({"C", "CCC", "CCCCC"},  // 1, 3, 5 atoms
               {"C", "N"},              // 1, 1 query atoms
               targetsHost,
               queriesHost,
               targetMols,
               queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  // Verify pairMatchStarts offsets are correctly computed
  // Both queries have 1 atom each (C and N)
  EXPECT_EQ(resultsHost.pairMatchStarts[0], 0);
  EXPECT_EQ(resultsHost.pairMatchStarts[1], 1);   // 0 + 1*1   (target 0, query 0)
  EXPECT_EQ(resultsHost.pairMatchStarts[2], 2);   // 1 + 1*1   (target 0, query 1)
  EXPECT_EQ(resultsHost.pairMatchStarts[3], 5);   // 2 + 3*1   (target 1, query 0)
  EXPECT_EQ(resultsHost.pairMatchStarts[4], 8);   // 5 + 3*1   (target 1, query 1)
  EXPECT_EQ(resultsHost.pairMatchStarts[5], 13);  // 8 + 5*1   (target 2, query 0)
  EXPECT_EQ(resultsHost.pairMatchStarts[6], 18);  // 13 + 5*1  (target 2, query 1)
}

TEST_P(SubstructureSearchTest, MultiAtomQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test multi-atom queries
  // CCO with CC: 2 non-unique matches (0,1) and (1,0)
  // Use larger buffer (3 atoms) so no overflow
  buildBatches({"CCO"},    // 3 atoms
               {"CC"},     // 2 atom query - should get 2 matches with uniquify=false
               targetsHost,
               queriesHost,
               targetMols,
               queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "CCO with CC query");
}

TEST_P(SubstructureSearchTest, ThreeAtomQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test 3-atom query: CCOCC with COC
  // Should get 2 matches: (1,2,3) and (3,2,1) - both directions through the ether
  buildBatches({"CCOCC"},   // 5 atoms - diethyl ether
               {"COC"},     // 3 atom query
               targetsHost,
               queriesHost,
               targetMols,
               queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "CCOCC with COC query");
}

TEST_P(SubstructureSearchTest, ExpectedOverflow) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // CCCCCC (6 atoms) with CC query has 10 non-unique matches
  // Buffer sized to target atoms (6), so this should overflow
  buildBatches({"CCCCCC"}, {"CC"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  // RDKit returns 10 non-unique matches
  auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(rdkitMatches.size(), 10u);

  // GPU should report actual count of 10, but only store 6
  EXPECT_TRUE(resultsHost.hasOverflow(0, 0))
    << "Expected overflow for hexane/CC pair";
  EXPECT_EQ(resultsHost.matchCounts[0], 10)
    << "GPU should count all 10 matches";
  EXPECT_EQ(resultsHost.reportedCounts[0], 6)
    << "GPU should only store 6 matches (buffer limit)";
}

// =============================================================================
// Non-Parameterized RDKit Reference Tests
// =============================================================================

class RDKitReferenceTest : public ::testing::Test {};

TEST_F(RDKitReferenceTest, HexaneCCMatches) {
  // Example from user: CCCCCC target, CC query
  // With uniquify=true: 5 matches
  // With uniquify=false: 10 matches
  auto target = makeMolFromSmiles("CCCCCC");
  auto query  = makeMolFromSmarts("CC");

  auto matchesUnique    = getRDKitSubstructMatches(*target, *query, true);
  auto matchesNonUnique = getRDKitSubstructMatches(*target, *query, false);

  EXPECT_EQ(matchesUnique.size(), 5u);
  EXPECT_EQ(matchesNonUnique.size(), 10u);
}

// =============================================================================
// Compound Query Tests (OR/NOT support)
// =============================================================================

TEST_P(SubstructureSearchTest, OrQueryMatchesBothTypes) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test OR query: [C,N] should match both carbons and nitrogens
  // CCN has 2 carbons and 1 nitrogen, so [C,N] should match all 3 atoms
  buildBatches({"CCN"}, {"[C,N]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "[C,N] in CCN");
}

TEST_P(SubstructureSearchTest, OrQuerySelectiveMatch) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test OR query: [N,O] should match nitrogens and oxygens but not carbons
  // CCO has 2 carbons and 1 oxygen, so [N,O] should match only the oxygen (1 atom)
  buildBatches({"CCO"}, {"[N,O]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "[N,O] in CCO");
}

TEST_P(SubstructureSearchTest, NotQueryExcludesAtom) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test NOT query: [!C] should match everything except carbon
  // CCO has 2 carbons and 1 oxygen, so [!C] should match only the oxygen
  buildBatches({"CCO"}, {"[!C]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "[!C] in CCO");
}

TEST_P(SubstructureSearchTest, NotQueryMatchesMultiple) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test NOT query: [!C] in molecule with multiple non-carbons
  // CCNO has 2 carbons, 1 nitrogen, and 1 oxygen, so [!C] should match 2 atoms
  buildBatches({"CCNO"}, {"[!C]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "[!C] in CCNO");
}

TEST_P(SubstructureSearchTest, MultiAtomOrQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test OR query with multi-atom pattern: [C,N][C,N]
  // CCN should match CC, CN, NC, and would match NN if present
  buildBatches({"CCN"}, {"[C,N][C,N]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "[C,N][C,N] in CCN");
}

TEST_P(SubstructureSearchTest, ThreeWayOrQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test 3-way OR: [C,N,O] should match all of C, N, and O
  // CCNO has all three types, should match 4 atoms
  buildBatches({"CCNO"}, {"[C,N,O]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "[C,N,O] in CCNO");
}

TEST_P(SubstructureSearchTest, NestedAndOrQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test nested AND/OR: [C,N;!R1] = (C OR N) AND NOT(in 1 ring)
  // In "CCN" (no rings), all 3 atoms should match
  // In "C1CC1N" (cyclopropane + N), only N should match (ring carbons fail !R1)
  // In "C1CCC1" (cyclobutane), 0 atoms match (all ring carbons fail !R1)
  buildBatches({"CCN", "C1CC1N", "C1CCC1"}, {"[C,N;!R1]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "[C,N;!R1] in CCN");
  expectMatchesRDKit(resultsHost, *targetMols[1], *queryMols[0], 1, 0, "[C,N;!R1] in C1CC1N");
  expectMatchesRDKit(resultsHost, *targetMols[2], *queryMols[0], 2, 0, "[C,N;!R1] in C1CCC1");
}

TEST_P(SubstructureSearchTest, DeepNestedOrAndOrQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test deep nesting: [C,N;R1,O] (complex SMARTS with multiple operators)
  // In cyclopentane C1CCCC1: ring carbons match
  // In "CCCCO": behavior depends on SMARTS precedence rules
  buildBatches({"C1CCCC1", "CCCCO"}, {"[C,N;R1,O]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "[C,N;R1,O] in C1CCCC1");
  expectMatchesRDKit(resultsHost, *targetMols[1], *queryMols[0], 1, 0, "[C,N;R1,O] in CCCCO");
}

TEST_P(SubstructureSearchTest, MultipleNotWithAndQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [!C;!N] = NOT(C) AND NOT(N) - matches anything except C or N
  // In "CCNO": only O matches
  buildBatches({"CCNO"}, {"[!C;!N]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "[!C;!N] in CCNO");
}

TEST_P(SubstructureSearchTest, NotWithOrQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [!C,!N] = NOT(C) OR NOT(N) - matches anything except C AND N
  // C atoms: NOT(C)=false, NOT(N)=true => true
  // N atom: NOT(C)=true, NOT(N)=false => true
  // O atom: NOT(C)=true, NOT(N)=true => true
  // All 4 atoms match
  buildBatches({"CCNO"}, {"[!C,!N]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "[!C,!N] in CCNO");
}

TEST_P(SubstructureSearchTest, SimpleAndNotQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [C;!R1] = C AND NOT(in 1 ring)
  // In C1CC1CCN (cyclopropane with chain): ring carbons (0,1,2) excluded, chain carbons (3,4) match
  buildBatches({"C1CC1CCN"}, {"[C;!R1]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "[C;!R1] in C1CC1CCN");
}

TEST_P(SubstructureSearchTest, BondedOrAtomQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test bonded pattern with OR atoms: [C,N]-[O,S]
  // In "CCNO": C-N-O, so N-O bond matches (N matches [C,N], O matches [O,S])
  // In "CCSO": C-S-O, so S-O bond matches if S in query... wait, S doesn't match [C,N]
  // Actually "CCS": C-C-S, no match for [C,N]-[O,S] since S doesn't connect to O
  // Use "CCO" (ethanol): C-C-O, C matches [C,N], O matches [O,S], so C-O matches
  buildBatches({"CCO", "CCS"}, {"[C,N]-[O,S]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "[C,N]-[O,S] in CCO");
  expectMatchesRDKit(resultsHost, *targetMols[1], *queryMols[0], 1, 0, "[C,N]-[O,S] in CCS");
}

TEST_P(SubstructureSearchTest, MultiAtomMixedBooleanQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test multi-atom query with different boolean logic per atom: [C,N]-[!O]
  // First atom is OR, second is NOT
  // In "CCO": C-C bond matches (C for [C,N], C for [!O]), C-O doesn't match (!O fails)
  // In "CCN": C-C and C-N bonds all match
  buildBatches({"CCO", "CCN"}, {"[C,N]-[!O]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "[C,N]-[!O] in CCO");
  expectMatchesRDKit(resultsHost, *targetMols[1], *queryMols[0], 1, 0, "[C,N]-[!O] in CCN");
}

TEST_P(SubstructureSearchTest, ThreeAtomNestedBooleanQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test 3-atom pattern with nested boolean: [C,N]-[!O]-[C,O]
  // In "CCCCO": C-C-C-C-O, should find C-C-C and C-C-O patterns
  buildBatches({"CCCCO"}, {"[C,N]-[!O]-[C,O]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "[C,N]-[!O]-[C,O] in CCCCO");
}

TEST_P(SubstructureSearchTest, AromaticOrQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test aromatic OR: [c,n] should match aromatic carbons and nitrogens
  // In pyridine "c1ccncc1": 5 aromatic carbons + 1 aromatic nitrogen = 6 matches
  buildBatches({"c1ccncc1"}, {"[c,n]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "[c,n] in pyridine");
}

TEST_P(SubstructureSearchTest, AromaticNotQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test aromatic NOT: [!n] should match anything except aromatic nitrogen
  // In pyridine "c1ccncc1": 5 aromatic carbons match, nitrogen doesn't
  buildBatches({"c1ccncc1"}, {"[!n]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  expectMatchesRDKit(resultsHost, *targetMols[0], *queryMols[0], 0, 0, "[!n] in pyridine");
}

TEST_P(SubstructureSearchTest, AromaticRingPatternWithOr) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test aromatic ring pattern with OR: c1[c,n]cccc1 (benzene or pyridine-like ring)
  // In benzene "c1ccccc1": all carbons form 6-ring, position 1 matches [c,n]
  // In pyridine "c1ccncc1": position 1 is nitrogen which matches [c,n]
  buildBatches({"c1ccccc1", "c1ccncc1"}, {"c1[c,n]cccc1"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  // Benzene should match (symmetric, many automorphisms)
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[0], static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for c1[c,n]cccc1 in benzene using " << algorithmName(algorithm());

  // Pyridine should match
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[1], static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for c1[c,n]cccc1 in pyridine using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, AnyRingMembershipQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [R] any ring membership query
  // C1CCC1C: 4 ring atoms, 1 non-ring atom
  // CCCCC: no ring atoms
  buildBatches({"C1CCC1C", "CCCCC"}, {"[R]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  // Cyclobutane with methyl: 4 ring atoms match [R]
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[0], static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [R] in C1CCC1C using " << algorithmName(algorithm());

  // Pentane: no ring atoms, should get 0 matches
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[1], static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [R] in CCCCC using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, AnyRingSizeQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [r] any ring size query (same semantics as [R])
  // c1ccccc1: 6 ring atoms
  // CCCCC: no ring atoms
  buildBatches({"c1ccccc1", "CCCCC"}, {"[r]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  // Benzene: 6 ring atoms match [r]
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[0], static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [r] in benzene using " << algorithmName(algorithm());

  // Pentane: no ring atoms, should get 0 matches
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[1], static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [r] in CCCCC using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, AnyRingCombinedWithAtomType) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [C;R] carbon in any ring
  // C1CCC1C: 4 ring carbons, 1 non-ring carbon
  // c1ccccc1: aromatic carbons (not aliphatic C)
  buildBatches({"C1CCC1C", "c1ccccc1"}, {"[C;R]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  // Cyclobutane with methyl: 4 aliphatic ring carbons match [C;R]
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[0], static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [C;R] in C1CCC1C using " << algorithmName(algorithm());

  // Benzene: aromatic carbons don't match aliphatic C
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[1], static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [C;R] in benzene using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, IsotopeCarbon13Query) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [13C] isotope query
  // [13C]CC: one carbon-13 atom
  // CC: natural abundance carbons (isotope = 0)
  buildBatches({"[13C]CC", "CC"}, {"[13C]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  // First target has one 13C
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[0], static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [13C] in [13C]CC using " << algorithmName(algorithm());

  // Second target has no isotope labels
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[1], static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [13C] in CC using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, IsotopeDeuteriumQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [2H] deuterium query
  // [2H]C([2H])([2H])[2H]: deuterated methane with 4 deuterium atoms
  // C: regular methane (no explicit H with isotope)
  buildBatches({"[2H]C([2H])([2H])[2H]", "C"}, {"[2H]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  // First target has 4 deuterium atoms
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[0], static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [2H] in CD4 using " << algorithmName(algorithm());

  // Second target has no deuterium
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[1], static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [2H] in CH4 using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, IsotopeNitrogen15Query) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [15N] nitrogen-15 query
  // [15N]CC: nitrogen-15 labeled
  // NCC: natural abundance nitrogen
  buildBatches({"[15N]CC", "NCC"}, {"[15N]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  // First target has one 15N
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[0], static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [15N] in [15N]CC using " << algorithmName(algorithm());

  // Second target has no isotope labels
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[1], static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [15N] in NCC using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, DegreeQueryD0) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [D0] degree query - atom with no explicit bonds
  // C: methane - single atom (degree 0)
  // CC: ethane - both atoms have degree 1
  buildBatches({"C", "CC"}, {"[D0]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[0], static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [D0] in methane using " << algorithmName(algorithm());

  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[1], static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [D0] in ethane using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, DegreeQueryD1) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [D1] degree query - terminal atoms
  // CC: ethane - both atoms have degree 1
  // CCC: propane - 2 terminal atoms with degree 1
  buildBatches({"CC", "CCC"}, {"[D1]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[0], static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [D1] in ethane using " << algorithmName(algorithm());

  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[1], static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [D1] in propane using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, DegreeQueryD3) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [D3] degree query
  // CC(C)C: isobutane - central carbon has degree 3
  // CCC: propane - no degree 3 atoms
  buildBatches({"CC(C)C", "CCC"}, {"[D3]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  // Isobutane: one atom with degree 3
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[0], static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [D3] in isobutane using " << algorithmName(algorithm());

  // Propane: no degree 3 atoms
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[1], static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [D3] in propane using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, TotalConnectivityQueryX1) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [X1] total connectivity query
  // [H][H]: H2 molecule - each H has X1 (1 bond + 0 H = 1)
  // CC: ethane - carbons have X4
  buildBatches({"[H][H]", "CC"}, {"[X1]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[0], static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [X1] in H2 using " << algorithmName(algorithm());

  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[1], static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [X1] in ethane using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, TotalConnectivityQueryX2) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [X2] total connectivity query
  // C#C: acetylene - carbons have X2 (1 bond + 1 H = 2)
  // CC: ethane - carbons have X4
  buildBatches({"C#C", "CC"}, {"[X2]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[0], static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [X2] in acetylene using " << algorithmName(algorithm());

  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[1], static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [X2] in ethane using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, TotalConnectivityQueryX3) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [X3] total connectivity query
  // C=C: ethene - carbons have X3 (1 bond + 2 H = 3)
  // CC: ethane - carbons have X4
  buildBatches({"C=C", "CC"}, {"[X3]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[0], static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [X3] in ethene using " << algorithmName(algorithm());

  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[1], static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [X3] in ethane using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, TotalConnectivityQueryX4) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [X4] total connectivity query
  // CC: ethane - all carbons have X4 (1 bond + 3 H = 4)
  // C=C: ethene - carbons have X3 (1 bond + 2 H = 3)
  buildBatches({"CC", "C=C"}, {"[X4]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  // Ethane: 2 atoms with X4
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[0], static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [X4] in ethane using " << algorithmName(algorithm());

  // Ethene: no X4 atoms
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[1], static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [X4] in ethene using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, DegreeWithAtomTypeQuery) {
  MoleculesHost                              targetsHost;
  MoleculesHost                              queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [CD3] carbon with degree 3
  // CC(C)C: isobutane
  // CN(C)C: trimethylamine - nitrogen has degree 3, not carbon
  buildBatches({"CC(C)C", "CN(C)C"}, {"[CD3]"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, algorithm(), stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  // Isobutane: one carbon with degree 3
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[0], static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [CD3] in isobutane using " << algorithmName(algorithm());

  // Trimethylamine: no carbon with degree 3 (N has degree 3)
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(resultsHost.matchCounts[1], static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [CD3] in trimethylamine using " << algorithmName(algorithm());
}
