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

#include <gmock/gmock.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmartsWrite.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

#include "cuda_error_check.h"
#include "device.h"
#include "molecules_device.cuh"

using nvMolKit::AsyncDeviceVector;
using nvMolKit::AtomQuery;
using nvMolKit::AtomQueryAtomicNum;
using nvMolKit::AtomQueryFormalCharge;
using nvMolKit::AtomQueryHybridization;
using nvMolKit::AtomQueryIsAliphatic;
using nvMolKit::AtomQueryIsAromatic;
using nvMolKit::AtomQueryIsInRing;
using nvMolKit::AtomQueryIsotope;
using nvMolKit::AtomQueryDegree;
using nvMolKit::AtomQueryTotalConnectivity;
using nvMolKit::AtomQueryMinRingSize;
using nvMolKit::AtomQueryNone;
using nvMolKit::AtomQueryNumExplicitHs;
using nvMolKit::AtomQueryNumRings;
using nvMolKit::AtomQueryTotalValence;
using nvMolKit::checkReturnCode;
using nvMolKit::getMolecule;
using nvMolKit::MoleculesDevice;
using nvMolKit::MoleculesDeviceView;
using nvMolKit::MoleculesHost;
using nvMolKit::MoleculeView;
using nvMolKit::ScopedStream;

namespace {

std::unique_ptr<RDKit::ROMol> makeQuery(const std::string& smarts) {
  auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
  EXPECT_NE(mol, nullptr) << "Failed to parse SMARTS: " << smarts;
  return mol;
}

__global__ void readAtomQueriesKernel(MoleculesDeviceView view, int molIdx, int* results) {
  const MoleculeView mol     = getMolecule(view, molIdx);
  const int          atomIdx = threadIdx.x;
  if (atomIdx >= mol.numAtoms) {
    return;
  }
  results[atomIdx] = static_cast<int>(mol.getAtomQuery(atomIdx));
}

}  // namespace

// =============================================================================
// Query Parsing Tests
// =============================================================================

struct QueryTestCase {
  std::string            smarts;
  std::vector<AtomQuery> expectedQueries;
};

class QueryParsingTest : public ::testing::TestWithParam<QueryTestCase> {};

TEST_P(QueryParsingTest, QueryTypeMatchesExpected) {
  const auto& testCase = GetParam();
  auto        mol      = makeQuery(testCase.smarts);
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  nvMolKit::addQueryToBatch(mol.get(), batch);

  ASSERT_EQ(batch.numMolecules(), 1);
  ASSERT_EQ(batch.atomQueries.size(), testCase.expectedQueries.size())
    << "Query count mismatch for SMARTS: " << testCase.smarts;

  for (size_t i = 0; i < testCase.expectedQueries.size(); ++i) {
    EXPECT_EQ(batch.atomQueries[i], testCase.expectedQueries[i])
      << "Query type mismatch at atom " << i << " for SMARTS: " << testCase.smarts
      << " expected: " << testCase.expectedQueries[i] << " got: " << batch.atomQueries[i];
  }
}

TEST_P(QueryParsingTest, QueryTypeMatchesOnDevice) {
  const auto& testCase = GetParam();
  auto        mol      = makeQuery(testCase.smarts);
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  nvMolKit::addQueryToBatch(mol.get(), batch);

  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch);

  const int              numAtoms = static_cast<int>(testCase.expectedQueries.size());
  AsyncDeviceVector<int> resultsDev(numAtoms, stream.stream());

  readAtomQueriesKernel<<<1, numAtoms, 0, stream.stream()>>>(device.view(), 0, resultsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(numAtoms);
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < numAtoms; ++i) {
    EXPECT_EQ(results[i], static_cast<int>(testCase.expectedQueries[i]))
      << "Device query type mismatch at atom " << i << " for SMARTS: " << testCase.smarts;
  }
}

// Convenience constants for common combinations
constexpr AtomQuery kAliphaticCarbon   = AtomQueryAtomicNum | AtomQueryIsAliphatic;
constexpr AtomQuery kAromaticCarbon    = AtomQueryAtomicNum | AtomQueryIsAromatic;
constexpr AtomQuery kAliphaticNitrogen = AtomQueryAtomicNum | AtomQueryIsAliphatic;
constexpr AtomQuery kAromaticNitrogen  = AtomQueryAtomicNum | AtomQueryIsAromatic;
constexpr AtomQuery kAliphaticOxygen   = AtomQueryAtomicNum | AtomQueryIsAliphatic;

INSTANTIATE_TEST_SUITE_P(
  SimpleQueries,
  QueryParsingTest,
  ::testing::Values(
    // Single atom queries by atomic number only
    QueryTestCase{
      "[#6]",
      {AtomQueryAtomicNum}
},
    QueryTestCase{"[#7]", {AtomQueryAtomicNum}},
    QueryTestCase{"[#8]", {AtomQueryAtomicNum}},

    // Multi-atom queries by atomic number
    QueryTestCase{"[#6][#6]", {AtomQueryAtomicNum, AtomQueryAtomicNum}},
    QueryTestCase{"[#6][#7][#8]", {AtomQueryAtomicNum, AtomQueryAtomicNum, AtomQueryAtomicNum}},

    // Ring queries
    QueryTestCase{"[R1]", {AtomQueryNumRings}},

    // Formal charge queries
    QueryTestCase{"[+1]", {AtomQueryFormalCharge}},
    QueryTestCase{"[-1]", {AtomQueryFormalCharge}},

    // Hybridization
    QueryTestCase{"[^3]", {AtomQueryHybridization}},

    // Aliphatic atoms (C, N, O) - these are AtomAnd of atomic number + aliphatic
    QueryTestCase{"C", {kAliphaticCarbon}},
    QueryTestCase{"N", {kAliphaticNitrogen}},
    QueryTestCase{"O", {kAliphaticOxygen}},
    QueryTestCase{"CC", {kAliphaticCarbon, kAliphaticCarbon}},
    QueryTestCase{"CCO", {kAliphaticCarbon, kAliphaticCarbon, kAliphaticOxygen}},

    // Aromatic atoms (c, n) - these are AtomAnd of atomic number + aromatic
    QueryTestCase{"c", {kAromaticCarbon}},
    QueryTestCase{"n", {kAromaticNitrogen}},
    QueryTestCase{"cc", {kAromaticCarbon, kAromaticCarbon}},

    // Benzene - all aromatic carbons
    QueryTestCase{
      "c1ccccc1",
      {kAromaticCarbon, kAromaticCarbon, kAromaticCarbon, kAromaticCarbon, kAromaticCarbon, kAromaticCarbon}},

    // Combined query: aliphatic carbon with H count
    QueryTestCase{"[CH3]", {AtomQueryAtomicNum | AtomQueryIsAliphatic | AtomQueryNumExplicitHs}},
    QueryTestCase{"[CH2]", {AtomQueryAtomicNum | AtomQueryIsAliphatic | AtomQueryNumExplicitHs}},

    // Ring size queries (r = smallest ring size)
    QueryTestCase{"[r5]", {AtomQueryMinRingSize}},
    QueryTestCase{"[r6]", {AtomQueryMinRingSize}},

    // Explicit AND with ampersand (&) - high precedence
    QueryTestCase{"[C&R1]", {AtomQueryAtomicNum | AtomQueryIsAliphatic | AtomQueryNumRings}},
    QueryTestCase{"[C&H3]", {AtomQueryAtomicNum | AtomQueryIsAliphatic | AtomQueryNumExplicitHs}},
    QueryTestCase{"[c&R1]", {AtomQueryAtomicNum | AtomQueryIsAromatic | AtomQueryNumRings}},
    QueryTestCase{"[#6&R1]", {AtomQueryAtomicNum | AtomQueryNumRings}},

    // Explicit AND with semicolon (;) - low precedence
    QueryTestCase{"[C;H3]", {AtomQueryAtomicNum | AtomQueryIsAliphatic | AtomQueryNumExplicitHs}},
    QueryTestCase{"[c;R1]", {AtomQueryAtomicNum | AtomQueryIsAromatic | AtomQueryNumRings}},
    QueryTestCase{"[#6;R1]", {AtomQueryAtomicNum | AtomQueryNumRings}},

    // Multiple ring query
    QueryTestCase{"[R2]", {AtomQueryNumRings}},

    // Combining ring membership with ring size
    QueryTestCase{"[R1;r6]", {AtomQueryNumRings | AtomQueryMinRingSize}},

    // Multiple chained ANDs (3+ conditions)
    QueryTestCase{"[c&R1&r6]", {AtomQueryAtomicNum | AtomQueryIsAromatic | AtomQueryNumRings | AtomQueryMinRingSize}},
    QueryTestCase{"[#6;R1;r5]", {AtomQueryAtomicNum | AtomQueryNumRings | AtomQueryMinRingSize}},
    QueryTestCase{"[C&R1&^3]", {AtomQueryAtomicNum | AtomQueryIsAliphatic | AtomQueryNumRings | AtomQueryHybridization}},

    // Total valence queries [v]
    QueryTestCase{"[v4]", {AtomQueryTotalValence}},
    QueryTestCase{"[v3]", {AtomQueryTotalValence}},
    QueryTestCase{"[C&v4]", {AtomQueryAtomicNum | AtomQueryIsAliphatic | AtomQueryTotalValence}},
    QueryTestCase{"[N;v3]", {AtomQueryAtomicNum | AtomQueryIsAliphatic | AtomQueryTotalValence}}),
  [](const ::testing::TestParamInfo<QueryTestCase>& info) {
    std::string name;
    for (char c : info.param.smarts) {
      if (std::isalnum(c)) {
        name += c;
      } else if (c == '#') {
        name += "Num";
      } else if (c == '+') {
        name += "Plus";
      } else if (c == '-') {
        name += "Minus";
      } else if (c == '^') {
        name += "Hyb";
      } else if (c == '[') {
        name += "L";
      } else if (c == ']') {
        name += "R";
      } else if (c == '&') {
        name += "And";
      } else if (c == ';') {
        name += "Semi";
      } else {
        name += '_';
      }
    }
    return name;
  });

// =============================================================================
// Compound Query Tests (OR/NOT)
// =============================================================================

struct CompoundQueryTestCase {
  std::string smarts;
  int         numAtoms;          ///< Expected number of atoms in query
  int         atom0NumLeaves;    ///< Expected numLeaves for first atom's tree
  int         atom0MinInstrs;    ///< Minimum expected instructions for first atom
};

class CompoundQueryParsingTest : public ::testing::TestWithParam<CompoundQueryTestCase> {};

TEST_P(CompoundQueryParsingTest, TreeStructureMatchesExpected) {
  const auto& testCase = GetParam();
  auto        mol      = makeQuery(testCase.smarts);
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  nvMolKit::addQueryToBatch(mol.get(), batch);

  ASSERT_EQ(batch.numMolecules(), 1);
  ASSERT_EQ(static_cast<int>(batch.atomQueryTrees.size()), testCase.numAtoms)
    << "Atom count mismatch for SMARTS: " << testCase.smarts;

  EXPECT_EQ(batch.atomQueryTrees[0].numLeaves, testCase.atom0NumLeaves)
    << "Leaf count mismatch at atom 0 for SMARTS: " << testCase.smarts;

  EXPECT_GE(batch.atomQueryTrees[0].numInstructions, testCase.atom0MinInstrs)
    << "Instruction count too low at atom 0 for SMARTS: " << testCase.smarts;
}

TEST_P(CompoundQueryParsingTest, TreeStructureMatchesOnDevice) {
  const auto& testCase = GetParam();
  auto        mol      = makeQuery(testCase.smarts);
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  nvMolKit::addQueryToBatch(mol.get(), batch);

  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch);

  // Verify device view has query trees populated
  auto view = device.view();
  EXPECT_NE(view.atomQueryTrees, nullptr)
    << "Device should have query trees for SMARTS: " << testCase.smarts;
  EXPECT_NE(view.queryInstructions, nullptr)
    << "Device should have query instructions for SMARTS: " << testCase.smarts;
  EXPECT_NE(view.queryLeafMasks, nullptr)
    << "Device should have query leaf masks for SMARTS: " << testCase.smarts;
}

INSTANTIATE_TEST_SUITE_P(
  OrQueries,
  CompoundQueryParsingTest,
  ::testing::Values(
    // Simple OR queries
    CompoundQueryTestCase{"[C,N]", 1, 2, 3},       // 2 leaves + 1 OR
    CompoundQueryTestCase{"[N,O]", 1, 2, 3},       // 2 leaves + 1 OR
    CompoundQueryTestCase{"[C,N,O]", 1, 3, 5},     // 3 leaves + 2 ORs
    CompoundQueryTestCase{"[C,N,O,S]", 1, 4, 7},   // 4 leaves + 3 ORs

    // OR with aromatic/aliphatic variants
    CompoundQueryTestCase{"[c,n]", 1, 2, 3},       // aromatic c OR n
    CompoundQueryTestCase{"[C,c]", 1, 2, 3},       // aliphatic C OR aromatic c

    // Multi-atom OR queries
    CompoundQueryTestCase{"[C,N][C,N]", 2, 2, 3},  // two atoms, each with OR

    // OR combined with other properties
    CompoundQueryTestCase{"[C,N;R1]", 1, 3, 5}),   // (C or N) AND R1: 3 leaves + OR + AND
  [](const ::testing::TestParamInfo<CompoundQueryTestCase>& info) {
    std::string name;
    for (char c : info.param.smarts) {
      if (std::isalnum(c)) {
        name += c;
      } else if (c == ',') {
        name += "Or";
      } else if (c == '[') {
        name += "L";
      } else if (c == ']') {
        name += "R";
      } else if (c == ';') {
        name += "Semi";
      } else {
        name += '_';
      }
    }
    return name;
  });

INSTANTIATE_TEST_SUITE_P(
  NotQueries,
  CompoundQueryParsingTest,
  ::testing::Values(
    // Simple NOT queries
    CompoundQueryTestCase{"[!C]", 1, 1, 2},        // 1 leaf + 1 NOT
    CompoundQueryTestCase{"[!N]", 1, 1, 2},        // 1 leaf + 1 NOT
    CompoundQueryTestCase{"[!c]", 1, 1, 2},        // NOT aromatic carbon

    // NOT with specific properties
    CompoundQueryTestCase{"[!R1]", 1, 1, 2},       // NOT in exactly 1 ring
    CompoundQueryTestCase{"[!r6]", 1, 1, 2},       // NOT in 6-membered ring

    // AND with NOT
    CompoundQueryTestCase{"[C;!R1]", 1, 2, 4},     // C AND NOT(R1): 2 leaves + NOT + AND
    CompoundQueryTestCase{"[N;!R1]", 1, 2, 4},     // N AND NOT(R1)

    // Multi-atom NOT queries
    CompoundQueryTestCase{"[!C][!N]", 2, 1, 2}),   // two atoms with NOT
  [](const ::testing::TestParamInfo<CompoundQueryTestCase>& info) {
    std::string name;
    for (char c : info.param.smarts) {
      if (std::isalnum(c)) {
        name += c;
      } else if (c == '!') {
        name += "Not";
      } else if (c == '[') {
        name += "L";
      } else if (c == ']') {
        name += "R";
      } else if (c == ';') {
        name += "Semi";
      } else {
        name += '_';
      }
    }
    return name;
  });

INSTANTIATE_TEST_SUITE_P(
  CombinedQueries,
  CompoundQueryParsingTest,
  ::testing::Values(
    // OR combined with NOT
    CompoundQueryTestCase{"[!C,!N]", 1, 2, 5},     // NOT(C) OR NOT(N): 2 leaves + 2 NOTs + OR

    // Complex combinations with multiple levels
    CompoundQueryTestCase{"[C,N;!R1]", 1, 3, 6},   // (C OR N) AND NOT(R1): 3 leaves + OR + NOT + AND

    // Nested alternating AND/OR: (C AND R1) OR N  (semicolon binds tighter due to left-to-right)
    CompoundQueryTestCase{"[C;R1,N]", 1, 3, 5},    // 3 leaves + AND + OR

    // Multiple ORs with AND: (C OR N OR O) AND R1
    CompoundQueryTestCase{"[C,N,O;R1]", 1, 4, 7},  // 4 leaves + 2 ORs + AND

    // Multiple ANDs with OR: C AND (R1 OR R2) - expressed as [C&R1,C&R2] workaround
    // Actually [C;R1,R2] = (C AND R1) OR R2
    CompoundQueryTestCase{"[C;R1,R2]", 1, 3, 5},   // 3 leaves + AND + OR

    // Deep nesting: ((C OR N) AND R1) OR O
    CompoundQueryTestCase{"[C,N;R1,O]", 1, 4, 7},  // 4 leaves + OR + AND + OR

    // Multiple NOTs with AND
    CompoundQueryTestCase{"[!C;!N]", 1, 2, 5},     // NOT(C) AND NOT(N): 2 leaves + 2 NOTs + AND

    // Triple nesting: (C OR N) AND (R1 OR R2) - need explicit grouping via semicolons
    // [C,N;R1,R2] actually parses as ((C OR N) AND R1) OR R2 due to left-to-right
    CompoundQueryTestCase{"[C,N;R1;R2]", 1, 4, 7}),// (C OR N) AND R1 AND R2: 4 leaves + OR + 2 ANDs
  [](const ::testing::TestParamInfo<CompoundQueryTestCase>& info) {
    std::string name;
    for (char c : info.param.smarts) {
      if (std::isalnum(c)) {
        name += c;
      } else if (c == ',') {
        name += "Or";
      } else if (c == '!') {
        name += "Not";
      } else if (c == '[') {
        name += "L";
      } else if (c == ']') {
        name += "R";
      } else if (c == ';') {
        name += "Semi";
      } else {
        name += '_';
      }
    }
    return name;
  });

// =============================================================================
// Unsupported Composite Query Tests (should throw)
// =============================================================================


TEST(QueryCompositeTest, RingConnectivityQueryThrows) {
  // [x2] ring connectivity query
  auto mol = makeQuery("[x2]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addQueryToBatch(mol.get(), batch), std::runtime_error);
}

TEST(QueryCompositeTest, ImplicitHCountQueryThrows) {
  // [h1] implicit H count query
  auto mol = makeQuery("[h1]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addQueryToBatch(mol.get(), batch), std::runtime_error);
}

TEST(QueryCompositeTest, ChiralityQueryThrows) {
  // [@] chirality query
  auto mol = makeQuery("[C@H](F)(Cl)Br");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addQueryToBatch(mol.get(), batch), std::runtime_error);
}

TEST(QueryCompositeTest, ExcessiveOrBranchesThrows) {
  // Create a SMARTS with many OR alternatives that exceeds kMaxBoolScratchSize.
  // Each alternative needs 1 leaf + 1 OR (except the first), so N alternatives
  // need N leaves + (N-1) ORs = 2N-1 scratch slots.
  // With kMaxBoolScratchSize=128, we need at least 65 alternatives to overflow.
  std::string smarts = "[#1";  // Start with hydrogen
  for (int i = 2; i <= 200; ++i) {
    smarts += ",#" + std::to_string(i);  // Add element 2-100 as OR alternatives
  }
  smarts += "]";  // 100 alternatives = 199 scratch slots needed

  auto mol = makeQuery(smarts);
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addQueryToBatch(mol.get(), batch), std::runtime_error);
}

TEST(QueryCompositeTest, FragmentQueryThrows) {
  auto mol = makeQuery("C.C");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addQueryToBatch(mol.get(), batch), std::runtime_error);
}

TEST(QueryCompositeTest, MultiFragmentQueryThrows) {
  auto mol = makeQuery("C[O;D1].C[O;D1].C[O;D1]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addQueryToBatch(mol.get(), batch), std::runtime_error);
}

TEST(QueryCompositeTest, WildcardAtomSucceeds) {
  // [*] wildcard atom - matches any atom
  // This is supported as it produces AtomNull which returns AtomQueryNone
  auto mol = makeQuery("[*]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.atomQueries[0], AtomQueryNone);
}

TEST(QueryCompositeTest, AnyAromaticAtomSucceeds) {
  // [a] any aromatic atom
  auto mol = makeQuery("[a]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.atomQueries[0], AtomQueryIsAromatic);
}

TEST(QueryCompositeTest, AnyAliphaticAtomSucceeds) {
  // [A] any aliphatic atom
  auto mol = makeQuery("[A]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.atomQueries[0], AtomQueryIsAliphatic);
}

TEST(QueryCompositeTest, AnyRingCountSucceeds) {
  // [R] any ring count - atom in at least one ring
  auto mol = makeQuery("[R]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.atomQueries[0], AtomQueryIsInRing);
}

TEST(QueryCompositeTest, AnyRingSizeSucceeds) {
  // [r] any ring size - atom in any ring
  auto mol = makeQuery("[r]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.atomQueries[0], AtomQueryIsInRing);
}

TEST(QueryCompositeTest, AnyRingWithAtomTypeSucceeds) {
  // [C;R] carbon in any ring
  auto mol = makeQuery("[C;R]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  // Should have both atom type and ring flags
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryAtomicNum);
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryIsInRing);
}

TEST(QueryCompositeTest, IsotopeQuerySucceeds) {
  // [13C] carbon-13 isotope
  auto mol = makeQuery("[13C]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryIsotope);
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryAtomicNum);
}

TEST(QueryCompositeTest, DeuteriumQuerySucceeds) {
  // [2H] deuterium
  auto mol = makeQuery("[2H]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryIsotope);
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryAtomicNum);
}

TEST(QueryCompositeTest, TritiumQuerySucceeds) {
  // [3H] tritium
  auto mol = makeQuery("[3H]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryIsotope);
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryAtomicNum);
}

TEST(QueryCompositeTest, DegreeQueryD0Succeeds) {
  // [D0] degree 0 (no explicit bonds)
  auto mol = makeQuery("[D0]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryDegree);
}

TEST(QueryCompositeTest, DegreeQueryD1Succeeds) {
  // [D1] degree 1 (terminal atom)
  auto mol = makeQuery("[D1]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryDegree);
}

TEST(QueryCompositeTest, DegreeQueryD3Succeeds) {
  // [D3] degree 3 (3 explicit bonds)
  auto mol = makeQuery("[D3]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryDegree);
}

TEST(QueryCompositeTest, DegreeWithAtomTypeSucceeds) {
  // [CD3] carbon with degree 3
  auto mol = makeQuery("[CD3]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryDegree);
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryAtomicNum);
}

TEST(QueryCompositeTest, TotalConnectivityQueryX1Succeeds) {
  // [X1] total connectivity 1
  auto mol = makeQuery("[X1]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryTotalConnectivity);
}

TEST(QueryCompositeTest, TotalConnectivityQueryX2Succeeds) {
  // [X2] total connectivity 2
  auto mol = makeQuery("[X2]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryTotalConnectivity);
}

TEST(QueryCompositeTest, TotalConnectivityQueryX4Succeeds) {
  // [X4] total connectivity 4 (degree + H count)
  auto mol = makeQuery("[X4]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryTotalConnectivity);
}

TEST(QueryCompositeTest, TotalConnectivityWithAtomTypeSucceeds) {
  // [CX4] carbon with total connectivity 4
  auto mol = makeQuery("[CX4]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryTotalConnectivity);
  EXPECT_TRUE(batch.atomQueries[0] & AtomQueryAtomicNum);
}

TEST(QueryCompositeTest, AnyBondSucceeds) {
  // C~C uses "any bond" (~) which should be supported
  auto mol = makeQuery("C~C");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.numMolecules(), 1);
  // Each carbon should have 1 "any" bond
  EXPECT_EQ(batch.bondTypeCounts[0].any, 1);
  EXPECT_EQ(batch.bondTypeCounts[1].any, 1);
}

TEST(QueryCompositeTest, MixedBondTypesSucceeds) {
  // C~C-C has both "any" bond and single bond
  auto mol = makeQuery("C~C-C");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.numMolecules(), 1);
  // First C: 1 any bond
  EXPECT_EQ(batch.bondTypeCounts[0].any, 1);
  EXPECT_EQ(batch.bondTypeCounts[0].single, 0);
  // Middle C: 1 any bond + 1 single bond
  EXPECT_EQ(batch.bondTypeCounts[1].any, 1);
  EXPECT_EQ(batch.bondTypeCounts[1].single, 1);
  // Last C: 1 single bond
  EXPECT_EQ(batch.bondTypeCounts[2].any, 0);
  EXPECT_EQ(batch.bondTypeCounts[2].single, 1);
}

// =============================================================================
// Batch with Multiple Query Molecules
// =============================================================================

TEST(QueryBatchTest, MultipleQueriesInBatch) {
  auto q1 = makeQuery("[#6]");
  auto q2 = makeQuery("[#7]");
  auto q3 = makeQuery("[#8]");

  MoleculesHost batch;
  nvMolKit::addQueryToBatch(q1.get(), batch);
  nvMolKit::addQueryToBatch(q2.get(), batch);
  nvMolKit::addQueryToBatch(q3.get(), batch);

  EXPECT_EQ(batch.numMolecules(), 3);
  EXPECT_EQ(batch.atomQueries.size(), 3);

  EXPECT_EQ(batch.atomQueries[0], AtomQueryAtomicNum);
  EXPECT_EQ(batch.atomQueries[1], AtomQueryAtomicNum);
  EXPECT_EQ(batch.atomQueries[2], AtomQueryAtomicNum);
}

TEST(QueryBatchTest, MultipleQueriesOnDevice) {
  auto q1 = makeQuery("[#6][#6]");  // 2 atoms
  auto q2 = makeQuery("[#7]");      // 1 atom
  auto q3 = makeQuery("[#8][#8]");  // 2 atoms

  MoleculesHost batch;
  nvMolKit::addQueryToBatch(q1.get(), batch);
  nvMolKit::addQueryToBatch(q2.get(), batch);
  nvMolKit::addQueryToBatch(q3.get(), batch);

  EXPECT_EQ(batch.numMolecules(), 3);
  EXPECT_EQ(batch.atomQueries.size(), 5);

  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch);

  // Test first molecule (2 atoms)
  {
    AsyncDeviceVector<int> resultsDev(2, stream.stream());
    readAtomQueriesKernel<<<1, 2, 0, stream.stream()>>>(device.view(), 0, resultsDev.data());
    cudaCheckError(cudaGetLastError());

    std::vector<int> results(2);
    resultsDev.copyToHost(results);
    cudaCheckError(cudaStreamSynchronize(stream.stream()));

    EXPECT_EQ(results[0], static_cast<int>(AtomQueryAtomicNum));
    EXPECT_EQ(results[1], static_cast<int>(AtomQueryAtomicNum));
  }

  // Test second molecule (1 atom)
  {
    AsyncDeviceVector<int> resultsDev(1, stream.stream());
    readAtomQueriesKernel<<<1, 1, 0, stream.stream()>>>(device.view(), 1, resultsDev.data());
    cudaCheckError(cudaGetLastError());

    std::vector<int> results(1);
    resultsDev.copyToHost(results);
    cudaCheckError(cudaStreamSynchronize(stream.stream()));

    EXPECT_EQ(results[0], static_cast<int>(AtomQueryAtomicNum));
  }

  // Test third molecule (2 atoms)
  {
    AsyncDeviceVector<int> resultsDev(2, stream.stream());
    readAtomQueriesKernel<<<1, 2, 0, stream.stream()>>>(device.view(), 2, resultsDev.data());
    cudaCheckError(cudaGetLastError());

    std::vector<int> results(2);
    resultsDev.copyToHost(results);
    cudaCheckError(cudaStreamSynchronize(stream.stream()));

    EXPECT_EQ(results[0], static_cast<int>(AtomQueryAtomicNum));
    EXPECT_EQ(results[1], static_cast<int>(AtomQueryAtomicNum));
  }
}

TEST(QueryBatchTest, MixedAliphaticAromaticQueries) {
  // Indole-like pattern: aromatic ring fused with aliphatic
  auto q = makeQuery("c1ccccc1C");  // benzene with aliphatic carbon

  MoleculesHost batch;
  nvMolKit::addQueryToBatch(q.get(), batch);

  EXPECT_EQ(batch.numMolecules(), 1);
  EXPECT_EQ(batch.atomQueries.size(), 7);

  // First 6 atoms are aromatic carbons
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(batch.atomQueries[i], kAromaticCarbon) << "Atom " << i << " should be aromatic carbon";
  }
  // Last atom is aliphatic carbon
  EXPECT_EQ(batch.atomQueries[6], kAliphaticCarbon) << "Atom 6 should be aliphatic carbon";
}

// =============================================================================
// Recursive SMARTS Pattern Extraction Tests
// =============================================================================

TEST(RecursivePatternExtraction, NoRecursivePatterns) {
  auto q = makeQuery("[CH3]");
  
  EXPECT_FALSE(nvMolKit::hasRecursiveSmarts(q.get()));
  
  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_TRUE(info.empty());
  EXPECT_EQ(info.size(), 0);
  EXPECT_FALSE(info.hasRecursivePatterns);
}

TEST(RecursivePatternExtraction, SimpleRecursivePattern) {
  auto q = makeQuery("[$([OH])]");
  
  EXPECT_TRUE(nvMolKit::hasRecursiveSmarts(q.get()));
  
  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_FALSE(info.empty());
  EXPECT_EQ(info.size(), 1);
  EXPECT_TRUE(info.hasRecursivePatterns);
  
  EXPECT_EQ(info.patterns[0].queryAtomIdx, 0);
  EXPECT_EQ(info.patterns[0].patternId, 0);
  EXPECT_NE(info.patterns[0].queryMol, nullptr);
}

TEST(RecursivePatternExtraction, MultipleRecursivePatterns) {
  auto q = makeQuery("[$([OH]),$([NH2])]");
  
  EXPECT_TRUE(nvMolKit::hasRecursiveSmarts(q.get()));
  
  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_EQ(info.size(), 2);
  
  EXPECT_EQ(info.patterns[0].patternId, 0);
  EXPECT_EQ(info.patterns[1].patternId, 1);
  EXPECT_EQ(info.patterns[0].queryAtomIdx, 0);
  EXPECT_EQ(info.patterns[1].queryAtomIdx, 0);
}

TEST(RecursivePatternExtraction, RecursivePatternsOnDifferentAtoms) {
  auto q = makeQuery("[$([OH])]-[$([C]=O)]");
  
  EXPECT_TRUE(nvMolKit::hasRecursiveSmarts(q.get()));
  
  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_EQ(info.size(), 2);
  
  std::set<int> atomIndices;
  for (const auto& pattern : info.patterns) {
    atomIndices.insert(pattern.queryAtomIdx);
  }
  EXPECT_EQ(atomIndices.size(), 2);
}

TEST(RecursivePatternExtraction, MixedRecursiveAndNonRecursive) {
  auto q = makeQuery("C[$([OH])]");
  
  EXPECT_TRUE(nvMolKit::hasRecursiveSmarts(q.get()));
  
  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_EQ(info.size(), 1);
  EXPECT_EQ(info.patterns[0].queryAtomIdx, 1);
}

TEST(RecursivePatternExtraction, ComplexRecursivePattern) {
  auto q = makeQuery("[$([CX3]=[OX1]),$([CX3+]-[OX1-])]");
  
  EXPECT_TRUE(nvMolKit::hasRecursiveSmarts(q.get()));
  
  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_EQ(info.size(), 2);
}

TEST(RecursivePatternExtraction, MaxPatternsAllowed) {
  auto q = makeQuery("[$([C]),$([N]),$([O]),$([S]),$([F]),$([Cl]),$([Br]),$([I])]");
  
  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_EQ(info.size(), 8);
  
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(info.patterns[i].patternId, i);
  }
}

TEST(RecursivePatternExtraction, TooManyPatternsThrows) {
  auto q = makeQuery("[$([C]),$([N]),$([O]),$([S]),$([F]),$([Cl]),$([Br]),$([I]),$([P])]");
  
  EXPECT_THROW(nvMolKit::extractRecursivePatterns(q.get()), std::runtime_error);
}

TEST(RecursivePatternExtraction, NullMolecule) {
  EXPECT_FALSE(nvMolKit::hasRecursiveSmarts(nullptr));
  
  auto info = nvMolKit::extractRecursivePatterns(nullptr);
  EXPECT_TRUE(info.empty());
}
