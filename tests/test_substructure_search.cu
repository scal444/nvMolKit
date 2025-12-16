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

  // RDKit also returns 0 matches here
  auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0]);
  EXPECT_EQ(rdkitMatches.size(), 0u);

  // GPU should also return 0
  EXPECT_EQ(resultsHost.matchCounts[0], 0);
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

  // RDKit: benzene has no aliphatic carbons
  auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0]);
  EXPECT_EQ(rdkitMatches.size(), 0u);

  EXPECT_EQ(resultsHost.matchCounts[0], 0);
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

  // Should get 2 matches with uniquify=false
  auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(rdkitMatches.size(), 2u) << "RDKit should find 2 non-unique matches";

  // Check GPU result
  EXPECT_EQ(resultsHost.matchCounts[0], 2)
    << "GPU should find 2 matches for CCO with CC using " << algorithmName(algorithm());
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

  // Should get 2 matches with uniquify=false
  auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(rdkitMatches.size(), 2u) << "RDKit should find 2 non-unique matches for COC in CCOCC";

  // Check GPU result
  EXPECT_EQ(resultsHost.matchCounts[0], 2)
    << "GPU should find 2 matches for CCOCC with COC using " << algorithmName(algorithm());
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
