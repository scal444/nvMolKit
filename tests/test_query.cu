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
    // Note: [R] without a count is not supported; use [R1], [R2], etc.
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
// Unsupported Composite Query Tests (OR/XOR should throw)
// =============================================================================

TEST(QueryCompositeTest, OrQueryThrows) {
  // [C,N] creates an AtomOr query
  auto mol = makeQuery("[C,N]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addQueryToBatch(mol.get(), batch), std::runtime_error);
}

TEST(QueryCompositeTest, RecursiveSmartsThrows) {
  // $(*C) recursive SMARTS
  auto mol = makeQuery("[$(*C)]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addQueryToBatch(mol.get(), batch), std::runtime_error);
}

TEST(QueryCompositeTest, NegationThrows) {
  // [!C] negated query
  auto mol = makeQuery("[!C]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addQueryToBatch(mol.get(), batch), std::runtime_error);
}

TEST(QueryCompositeTest, DegreeQueryThrows) {
  // [D3] explicit degree query
  auto mol = makeQuery("[D3]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addQueryToBatch(mol.get(), batch), std::runtime_error);
}

TEST(QueryCompositeTest, TotalConnectivityQueryThrows) {
  // [X3] total connectivity query
  auto mol = makeQuery("[X3]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addQueryToBatch(mol.get(), batch), std::runtime_error);
}

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

TEST(QueryCompositeTest, IsotopeQueryThrows) {
  // [13C] isotope/mass query
  auto mol = makeQuery("[13C]");
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

TEST(QueryCompositeTest, BareRingQueryThrows) {
  // [R] without a count is not supported - use [R1], [R2], etc.
  auto mol = makeQuery("[R]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addQueryToBatch(mol.get(), batch), std::runtime_error);
}

TEST(QueryCompositeTest, BareRingInAndQueryThrows) {
  // [C&R] contains unsupported [R] - use [C&R1] instead
  auto mol = makeQuery("[C&R]");
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
