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
#include <GraphMol/Substruct/SubstructMatch.h>
#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <vector>

#include "cuda_error_check.h"
#include "device.h"
#include "substructure_search.cuh"

using nvMolKit::addQueryToBatch;
using nvMolKit::addToBatch;
using nvMolKit::checkReturnCode;
using nvMolKit::getSubstructMatches;
using nvMolKit::MoleculesDevice;
using nvMolKit::MoleculesHost;
using nvMolKit::ScopedStream;
using nvMolKit::SubstructMatchResultsDevice;
using nvMolKit::SubstructMatchResultsHost;

namespace {

std::unique_ptr<RDKit::ROMol> makeMolFromSmiles(const std::string& smiles) {
  auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
  return mol;
}

std::unique_ptr<RDKit::ROMol> makeMolFromSmarts(const std::string& smarts) {
  auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
  return mol;
}

/**
 * @brief Get substructure matches using RDKit as ground truth.
 *
 * @param target Target molecule (SMILES)
 * @param query Query molecule (SMARTS)
 * @param uniquify If true, return only unique matches
 * @return Vector of matches, where each match is a vector of target atom indices
 */
std::vector<std::vector<int>> getRDKitMatches(const RDKit::ROMol& target,
                                              const RDKit::ROMol& query,
                                              bool                uniquify = true) {
  RDKit::SubstructMatchParameters params;
  params.uniquify = uniquify;

  std::vector<RDKit::MatchVectType> matches = RDKit::SubstructMatch(target, query, params);

  std::vector<std::vector<int>> result;
  result.reserve(matches.size());

  for (const auto& match : matches) {
    std::vector<int> mapping(match.size());
    for (size_t i = 0; i < match.size(); ++i) {
      mapping[match[i].first] = match[i].second;
    }
    result.push_back(std::move(mapping));
  }

  return result;
}

}  // namespace

// =============================================================================
// Test Fixture
// =============================================================================

class SubstructureSearchTest : public ::testing::Test {
 protected:
  ScopedStream stream_;

  void SetUp() override {}

  /**
   * @brief Build target and query batches from SMILES/SMARTS strings.
   */
  void buildBatches(const std::vector<std::string>& targetSmiles,
                    const std::vector<std::string>& querySmarts,
                    MoleculesHost&                  targetsHost,
                    MoleculesHost&                  queriesHost,
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
   * @brief Compare GPU results against RDKit ground truth.
   *
   * @param results GPU results
   * @param targetMols Target molecules for RDKit comparison
   * @param queryMols Query molecules for RDKit comparison
   * @param expectMatch If true, expect tests to pass; if false, expect current failures
   */
  void compareWithRDKit(const SubstructMatchResultsHost&                  results,
                        const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                        const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                        bool                                              expectMatch = false) {
    for (int t = 0; t < results.numTargets; ++t) {
      for (int q = 0; q < results.numQueries; ++q) {
        const auto rdkitMatches = getRDKitMatches(*targetMols[t], *queryMols[q]);
        const int  pairIdx      = results.pairIndex(t, q);

        const int gpuMatchCount = results.matchCounts[pairIdx];
        const int rdkitMatchCount = static_cast<int>(rdkitMatches.size());

        if (expectMatch) {
          EXPECT_EQ(gpuMatchCount, rdkitMatchCount)
            << "Match count mismatch for target " << t << ", query " << q
            << ": GPU=" << gpuMatchCount << ", RDKit=" << rdkitMatchCount;
        } else {
          // Currently expected to fail - just log the discrepancy
          if (gpuMatchCount != rdkitMatchCount) {
            // Expected: search not yet implemented, GPU returns 0
            EXPECT_EQ(gpuMatchCount, 0)
              << "GPU should return 0 matches (search not implemented)";
          }
        }
      }
    }
  }
};

// =============================================================================
// Basic Tests - Expected to fail until search is implemented
// =============================================================================

TEST_F(SubstructureSearchTest, SingleTargetSingleQuery) {
  MoleculesHost                             targetsHost;
  MoleculesHost                             queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  buildBatches({"CCO"}, {"C"}, targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  // Verify structure
  EXPECT_EQ(resultsHost.numTargets, 1);
  EXPECT_EQ(resultsHost.numQueries, 1);

  // RDKit comparison (expect failure - search not implemented)
  compareWithRDKit(resultsHost, targetMols, queryMols, false);

  // Verify GPU returns 0 (placeholder)
  EXPECT_EQ(resultsHost.matchCounts[0], 0);
}

TEST_F(SubstructureSearchTest, MultipleTargetsSingleQuery) {
  MoleculesHost                             targetsHost;
  MoleculesHost                             queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  buildBatches({"CCO", "CCCC", "c1ccccc1"}, {"C"}, 
               targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  EXPECT_EQ(resultsHost.numTargets, 3);
  EXPECT_EQ(resultsHost.numQueries, 1);

  compareWithRDKit(resultsHost, targetMols, queryMols, false);
}

TEST_F(SubstructureSearchTest, SingleTargetMultipleQueries) {
  MoleculesHost                             targetsHost;
  MoleculesHost                             queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  buildBatches({"CCO"}, {"C", "O", "CC"},
               targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  EXPECT_EQ(resultsHost.numTargets, 1);
  EXPECT_EQ(resultsHost.numQueries, 3);

  compareWithRDKit(resultsHost, targetMols, queryMols, false);
}

TEST_F(SubstructureSearchTest, BatchAllToAll) {
  MoleculesHost                             targetsHost;
  MoleculesHost                             queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  buildBatches(
    {"CCO", "CCCC", "c1ccccc1", "CC(C)C"},  // 4 targets
    {"C", "O", "CC", "c"},                   // 4 queries
    targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  EXPECT_EQ(resultsHost.numTargets, 4);
  EXPECT_EQ(resultsHost.numQueries, 4);

  // Should have 16 pairs
  EXPECT_EQ(static_cast<int>(resultsHost.matchCounts.size()), 16);

  compareWithRDKit(resultsHost, targetMols, queryMols, false);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(SubstructureSearchTest, NoMatchPossible) {
  MoleculesHost                             targetsHost;
  MoleculesHost                             queriesHost;
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
                      resultsDevice, stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  // RDKit also returns 0 matches here
  auto rdkitMatches = getRDKitMatches(*targetMols[0], *queryMols[0]);
  EXPECT_EQ(rdkitMatches.size(), 0u);

  // GPU should also return 0 (which it does as placeholder anyway)
  EXPECT_EQ(resultsHost.matchCounts[0], 0);
}

TEST_F(SubstructureSearchTest, AromaticVsAliphatic) {
  MoleculesHost                             targetsHost;
  MoleculesHost                             queriesHost;
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
                      resultsDevice, stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  // RDKit: benzene has no aliphatic carbons
  auto rdkitMatches = getRDKitMatches(*targetMols[0], *queryMols[0]);
  EXPECT_EQ(rdkitMatches.size(), 0u);

  // GPU placeholder returns 0 - this is actually correct for this case!
  EXPECT_EQ(resultsHost.matchCounts[0], 0);
}

// =============================================================================
// RDKit Ground Truth Reference Tests
// =============================================================================

TEST_F(SubstructureSearchTest, RDKitReferenceHexaneCC) {
  // Example from user: CCCCCC target, CC query
  // With uniquify=true: 5 matches
  // With uniquify=false: 10 matches
  auto target = makeMolFromSmiles("CCCCCC");
  auto query  = makeMolFromSmarts("CC");

  auto matchesUnique    = getRDKitMatches(*target, *query, true);
  auto matchesNonUnique = getRDKitMatches(*target, *query, false);

  EXPECT_EQ(matchesUnique.size(), 5u);
  EXPECT_EQ(matchesNonUnique.size(), 10u);
}

TEST_F(SubstructureSearchTest, LargerMolecule) {
  MoleculesHost                             targetsHost;
  MoleculesHost                             queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Caffeine as a larger test case
  buildBatches({"Cn1cnc2c1c(=O)n(c(=O)n2C)C"}, {"c", "N", "C"},
               targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  EXPECT_EQ(resultsHost.numTargets, 1);
  EXPECT_EQ(resultsHost.numQueries, 3);

  compareWithRDKit(resultsHost, targetMols, queryMols, false);
}

// =============================================================================
// Buffer Allocation Tests
// =============================================================================

TEST_F(SubstructureSearchTest, BufferAllocationCorrect) {
  MoleculesHost                             targetsHost;
  MoleculesHost                             queriesHost;
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Different sized molecules to test buffer allocation
  buildBatches(
    {"C", "CCC", "CCCCC"},       // 1, 3, 5 atoms
    {"C", "CC"},                  // 1, 2 query atoms
    targetsHost, queriesHost, targetMols, queryMols);

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  SubstructMatchResultsDevice resultsDevice(stream_.stream());
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsDevice, stream_.stream());

  SubstructMatchResultsHost resultsHost;
  resultsDevice.copyToHost(resultsHost);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  // Verify pairMatchStarts offsets are correctly computed
  // Pair (0,0): target=1 atom, query=1 atom -> capacity = 1 * 1 = 1
  // Pair (0,1): target=1 atom, query=2 atoms -> capacity = 1 * 2 = 2
  // Pair (1,0): target=3 atoms, query=1 atom -> capacity = 3 * 1 = 3
  // Pair (1,1): target=3 atoms, query=2 atoms -> capacity = 3 * 2 = 6
  // Pair (2,0): target=5 atoms, query=1 atom -> capacity = 5 * 1 = 5
  // Pair (2,1): target=5 atoms, query=2 atoms -> capacity = 5 * 2 = 10

  EXPECT_EQ(resultsHost.pairMatchStarts[0], 0);
  EXPECT_EQ(resultsHost.pairMatchStarts[1], 1);   // 0 + 1
  EXPECT_EQ(resultsHost.pairMatchStarts[2], 3);   // 1 + 2
  EXPECT_EQ(resultsHost.pairMatchStarts[3], 6);   // 3 + 3
  EXPECT_EQ(resultsHost.pairMatchStarts[4], 12);  // 6 + 6
  EXPECT_EQ(resultsHost.pairMatchStarts[5], 17);  // 12 + 5
  EXPECT_EQ(resultsHost.pairMatchStarts[6], 27);  // 17 + 10
}

