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
#include <vector>

#include "atom_data_packed.h"
#include "cuda_error_check.h"
#include "device.h"
#include "graph_labeler.cuh"
#include "molecules_device.cuh"

using nvMolKit::addQueryToBatch;
using nvMolKit::addToBatch;
using nvMolKit::AsyncDeviceVector;
using nvMolKit::AtomQuery;
using nvMolKit::AtomQueryAtomicNum;
using nvMolKit::AtomQueryIsAliphatic;
using nvMolKit::AtomQueryIsAromatic;
using nvMolKit::BitMatrix2DView;
using nvMolKit::checkReturnCode;
using nvMolKit::FlatBitVect;
using nvMolKit::getMolecule;
using nvMolKit::MoleculesDevice;
using nvMolKit::MoleculesDeviceView;
using nvMolKit::MoleculesHost;
using nvMolKit::MoleculeView;
using nvMolKit::ScopedStream;

namespace {

std::unique_ptr<RDKit::ROMol> makeMolFromSmiles(const std::string& smiles) {
  auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
  return mol;
}

std::unique_ptr<RDKit::ROMol> makeMolFromSmarts(const std::string& smarts) {
  auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
  return mol;
}

constexpr std::size_t kMaxTargetAtoms = 128;
constexpr std::size_t kMaxQueryAtoms  = 64;
using LabelMatrixStorage              = FlatBitVect<kMaxTargetAtoms * kMaxQueryAtoms>;
using LabelMatrixView                 = BitMatrix2DView<kMaxTargetAtoms, kMaxQueryAtoms>;

}  // namespace

// =============================================================================
// atomMatches Tests
// =============================================================================

__global__ void atomMatchesKernel(const nvMolKit::AtomData* target,
                                  const nvMolKit::AtomData* query,
                                  AtomQuery                 queryFlags,
                                  uint8_t*                  result) {
  *result = nvMolKit::atomMatches(*target, *query, queryFlags);
}

TEST(GraphLabelerAtomMatches, SameAtomicNumber) {
  ScopedStream stream;

  nvMolKit::AtomData target{};
  target.atomicNum = 6;

  nvMolKit::AtomData query{};
  query.atomicNum = 6;

  AsyncDeviceVector<nvMolKit::AtomData> targetDev(1, stream.stream());
  AsyncDeviceVector<nvMolKit::AtomData> queryDev(1, stream.stream());
  AsyncDeviceVector<uint8_t>            resultDev(1, stream.stream());

  targetDev.setFromVector(std::vector<nvMolKit::AtomData>{target});
  queryDev.setFromVector(std::vector<nvMolKit::AtomData>{query});

  atomMatchesKernel<<<1, 1, 0, stream.stream()>>>(targetDev.data(),
                                                  queryDev.data(),
                                                  AtomQueryAtomicNum,
                                                  resultDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<uint8_t> result(1);
  resultDev.copyToHost(result);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  EXPECT_TRUE(result[0]);
}

TEST(GraphLabelerAtomMatches, DifferentAtomicNumber) {
  ScopedStream stream;

  nvMolKit::AtomData target{};
  target.atomicNum = 6;

  nvMolKit::AtomData query{};
  query.atomicNum = 7;

  AsyncDeviceVector<nvMolKit::AtomData> targetDev(1, stream.stream());
  AsyncDeviceVector<nvMolKit::AtomData> queryDev(1, stream.stream());
  AsyncDeviceVector<uint8_t>            resultDev(1, stream.stream());

  targetDev.setFromVector(std::vector<nvMolKit::AtomData>{target});
  queryDev.setFromVector(std::vector<nvMolKit::AtomData>{query});

  atomMatchesKernel<<<1, 1, 0, stream.stream()>>>(targetDev.data(),
                                                  queryDev.data(),
                                                  AtomQueryAtomicNum,
                                                  resultDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<uint8_t> result(1);
  resultDev.copyToHost(result);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  EXPECT_FALSE(result[0]);
}

TEST(GraphLabelerAtomMatches, AromaticMatch) {
  ScopedStream stream;

  nvMolKit::AtomData target{};
  target.atomicNum  = 6;
  target.isAromatic = true;

  nvMolKit::AtomData query{};
  query.atomicNum  = 6;
  query.isAromatic = true;

  AsyncDeviceVector<nvMolKit::AtomData> targetDev(1, stream.stream());
  AsyncDeviceVector<nvMolKit::AtomData> queryDev(1, stream.stream());
  AsyncDeviceVector<uint8_t>            resultDev(1, stream.stream());

  targetDev.setFromVector(std::vector<nvMolKit::AtomData>{target});
  queryDev.setFromVector(std::vector<nvMolKit::AtomData>{query});

  // Query requires aromatic
  atomMatchesKernel<<<1, 1, 0, stream.stream()>>>(targetDev.data(),
                                                  queryDev.data(),
                                                  AtomQueryAtomicNum | AtomQueryIsAromatic,
                                                  resultDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<uint8_t> result(1);
  resultDev.copyToHost(result);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  EXPECT_TRUE(result[0]);
}

TEST(GraphLabelerAtomMatches, AromaticMismatch) {
  ScopedStream stream;

  nvMolKit::AtomData target{};
  target.atomicNum  = 6;
  target.isAromatic = false;

  nvMolKit::AtomData query{};
  query.atomicNum  = 6;
  query.isAromatic = true;

  AsyncDeviceVector<nvMolKit::AtomData> targetDev(1, stream.stream());
  AsyncDeviceVector<nvMolKit::AtomData> queryDev(1, stream.stream());
  AsyncDeviceVector<uint8_t>            resultDev(1, stream.stream());

  targetDev.setFromVector(std::vector<nvMolKit::AtomData>{target});
  queryDev.setFromVector(std::vector<nvMolKit::AtomData>{query});

  // Query requires aromatic, target is aliphatic
  atomMatchesKernel<<<1, 1, 0, stream.stream()>>>(targetDev.data(),
                                                  queryDev.data(),
                                                  AtomQueryAtomicNum | AtomQueryIsAromatic,
                                                  resultDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<uint8_t> result(1);
  resultDev.copyToHost(result);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  EXPECT_FALSE(result[0]);
}

TEST(GraphLabelerAtomMatches, AliphaticMatch) {
  ScopedStream stream;

  nvMolKit::AtomData target{};
  target.atomicNum  = 6;
  target.isAromatic = false;

  nvMolKit::AtomData query{};
  query.atomicNum  = 6;
  query.isAromatic = false;

  AsyncDeviceVector<nvMolKit::AtomData> targetDev(1, stream.stream());
  AsyncDeviceVector<nvMolKit::AtomData> queryDev(1, stream.stream());
  AsyncDeviceVector<uint8_t>            resultDev(1, stream.stream());

  targetDev.setFromVector(std::vector<nvMolKit::AtomData>{target});
  queryDev.setFromVector(std::vector<nvMolKit::AtomData>{query});

  // Query requires aliphatic
  atomMatchesKernel<<<1, 1, 0, stream.stream()>>>(targetDev.data(),
                                                  queryDev.data(),
                                                  AtomQueryAtomicNum | AtomQueryIsAliphatic,
                                                  resultDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<uint8_t> result(1);
  resultDev.copyToHost(result);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  EXPECT_TRUE(result[0]);
}

TEST(GraphLabelerAtomMatches, NoQueryFlags) {
  ScopedStream stream;

  // With no query flags, any atom matches any atom
  nvMolKit::AtomData target{};
  target.atomicNum    = 6;
  target.formalCharge = -1;

  nvMolKit::AtomData query{};
  query.atomicNum    = 8;
  query.formalCharge = 2;

  AsyncDeviceVector<nvMolKit::AtomData> targetDev(1, stream.stream());
  AsyncDeviceVector<nvMolKit::AtomData> queryDev(1, stream.stream());
  AsyncDeviceVector<uint8_t>            resultDev(1, stream.stream());

  targetDev.setFromVector(std::vector<nvMolKit::AtomData>{target});
  queryDev.setFromVector(std::vector<nvMolKit::AtomData>{query});

  atomMatchesKernel<<<1, 1, 0, stream.stream()>>>(targetDev.data(), queryDev.data(), 0, resultDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<uint8_t> result(1);
  resultDev.copyToHost(result);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  EXPECT_TRUE(result[0]);
}

// =============================================================================
// Full Graph Labeling Tests with Real Molecules
// =============================================================================

template <std::size_t MaxTarget, std::size_t MaxQuery>
__global__ void populateLabelMatrixKernel(MoleculesDeviceView                targetBatch,
                                          int                                targetMolIdx,
                                          MoleculesDeviceView                queryBatch,
                                          int                                queryMolIdx,
                                          FlatBitVect<MaxTarget * MaxQuery>* matrix) {
  MoleculeView                         target = getMolecule(targetBatch, targetMolIdx);
  MoleculeView                         query  = getMolecule(queryBatch, queryMolIdx);
  BitMatrix2DView<MaxTarget, MaxQuery> view(matrix);
  nvMolKit::populateLabelMatrix<MaxTarget, MaxQuery>(target, query, view);
}

class GraphLabelerTest : public ::testing::Test {
 protected:
  ScopedStream stream_;

  void SetUp() override {}

  void runLabelingTest(const std::string&                 targetSmiles,
                       const std::string&                 querySmarts,
                       std::vector<std::vector<uint8_t>>& expectedMatrix) {
    auto targetMol = makeMolFromSmiles(targetSmiles);
    auto queryMol  = makeMolFromSmarts(querySmarts);
    ASSERT_NE(targetMol, nullptr) << "Failed to parse target: " << targetSmiles;
    ASSERT_NE(queryMol, nullptr) << "Failed to parse query: " << querySmarts;

    MoleculesHost targetHost;
    MoleculesHost queryHost;
    addToBatch(targetMol.get(), targetHost);
    addQueryToBatch(queryMol.get(), queryHost);

    MoleculesDevice targetDevice(stream_.stream());
    MoleculesDevice queryDevice(stream_.stream());
    targetDevice.copyFromHost(targetHost);
    queryDevice.copyFromHost(queryHost);

    AsyncDeviceVector<LabelMatrixStorage> matrixDev(1, stream_.stream());
    const LabelMatrixStorage              hostMatrix(false);
    matrixDev.setFromVector(std::vector<LabelMatrixStorage>{hostMatrix});

    populateLabelMatrixKernel<kMaxTargetAtoms, kMaxQueryAtoms>
      <<<1, 1, 0, stream_.stream()>>>(targetDevice.view(), 0, queryDevice.view(), 0, matrixDev.data());
    cudaCheckError(cudaGetLastError());

    std::vector<LabelMatrixStorage> resultMatrix(1);
    matrixDev.copyToHost(resultMatrix);
    cudaCheckError(cudaStreamSynchronize(stream_.stream()));

    const LabelMatrixView view(resultMatrix[0]);

    const int numTargetAtoms = static_cast<int>(targetHost.totalAtoms());
    const int numQueryAtoms  = static_cast<int>(queryHost.totalAtoms());

    ASSERT_EQ(expectedMatrix.size(), numTargetAtoms);
    for (int i = 0; i < numTargetAtoms; ++i) {
      ASSERT_EQ(expectedMatrix[i].size(), numQueryAtoms);
      for (int j = 0; j < numQueryAtoms; ++j) {
        EXPECT_EQ(view.get(i, j), expectedMatrix[i][j]) << "Mismatch at target atom " << i << ", query atom " << j
                                                        << " for target=" << targetSmiles << ", query=" << querySmarts;
      }
    }
  }
};

TEST_F(GraphLabelerTest, EthaneQueryCarbon) {
  // Target: CC (ethane) - 2 carbons, each with 1 bond
  // Query: C (single aliphatic carbon)
  // Both target atoms should match the query
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // Target atom 0 matches query atom 0
    {true}   // Target atom 1 matches query atom 0
  };
  runLabelingTest("CC", "C", expected);
}

TEST_F(GraphLabelerTest, EthaneQueryNitrogen) {
  // Target: CC (ethane)
  // Query: N (nitrogen)
  // No target atoms should match
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}};
  runLabelingTest("CC", "N", expected);
}

TEST_F(GraphLabelerTest, EthanolQueryOxygen) {
  // Target: CCO (ethanol) - C, C, O
  // Query: O (oxygen)
  // Only the oxygen should match
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // First C
    {false},  // Second C
    {true}    // O
  };
  runLabelingTest("CCO", "O", expected);
}

TEST_F(GraphLabelerTest, BenzeneQueryAromaticCarbon) {
  // Target: c1ccccc1 (benzene) - 6 aromatic carbons
  // Query: c (aromatic carbon)
  // All 6 should match
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {true}, {true}, {true}};
  runLabelingTest("c1ccccc1", "c", expected);
}

TEST_F(GraphLabelerTest, BenzeneQueryAliphaticCarbon) {
  // Target: c1ccccc1 (benzene) - 6 aromatic carbons
  // Query: C (aliphatic carbon)
  // None should match (aromatic vs aliphatic)
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {false}, {false}, {false}, {false}};
  runLabelingTest("c1ccccc1", "C", expected);
}

TEST_F(GraphLabelerTest, PropaneQueryCC) {
  // Target: CCC (propane) - C0-C1-C2
  // Query: CC (two carbons bonded)
  // C0 can match Q0 or Q1 (1 bond each side of query)
  // C1 can match Q0 or Q1 (2 bonds, enough for either end)
  // C2 can match Q0 or Q1 (1 bond each side of query)
  std::vector<std::vector<uint8_t>> expected = {
    {true, true}, // C0: 1 bond, matches both query atoms (each has 1 bond)
    {true, true}, // C1: 2 bonds, matches both query atoms
    {true, true}  // C2: 1 bond, matches both query atoms
  };
  runLabelingTest("CCC", "CC", expected);
}

TEST_F(GraphLabelerTest, MethaneQueryCC) {
  // Target: C (methane) - 1 carbon with 0 heavy-atom bonds
  // Query: CC (two bonded carbons) - each has 1 bond
  // Methane's carbon has 0 bonds, can't match query atoms with 1 bond
  std::vector<std::vector<uint8_t>> expected = {
    {false, false}
  };
  runLabelingTest("C", "CC", expected);
}

TEST_F(GraphLabelerTest, TolueneQueryAromaticCarbon) {
  // Target: Cc1ccccc1 (toluene) - 1 aliphatic C + 6 aromatic c
  // Query: c (aromatic carbon)
  // Only the 6 aromatic carbons should match
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // Methyl carbon (aliphatic)
    {true},   // Aromatic
    {true},   // Aromatic
    {true},   // Aromatic
    {true},   // Aromatic
    {true},   // Aromatic
    {true}    // Aromatic
  };
  runLabelingTest("Cc1ccccc1", "c", expected);
}

TEST_F(GraphLabelerTest, PyridineQueryAromaticNitrogen) {
  // Target: c1ccncc1 (pyridine) - 5 aromatic C, 1 aromatic N
  // Query: n (aromatic nitrogen)
  // Only the nitrogen should match
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c
    {false},  // c
    {false},  // c
    {true},   // n
    {false},  // c
    {false}   // c
  };
  runLabelingTest("c1ccncc1", "n", expected);
}

TEST_F(GraphLabelerTest, AtomicNumberOnlyQuery) {
  // Target: CCO
  // Query: [#6] (any carbon regardless of aromaticity)
  // Both carbons should match
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // C
    {true},  // C
    {false}  // O
  };
  runLabelingTest("CCO", "[#6]", expected);
}

// =============================================================================
// Parallel Per-Atom Labeling Test
// =============================================================================

template <std::size_t MaxTarget, std::size_t MaxQuery>
__global__ void populateLabelMatrixParallelKernel(MoleculesDeviceView                targetBatch,
                                                  int                                targetMolIdx,
                                                  MoleculesDeviceView                queryBatch,
                                                  int                                queryMolIdx,
                                                  FlatBitVect<MaxTarget * MaxQuery>* matrix) {
  MoleculeView                         target = getMolecule(targetBatch, targetMolIdx);
  MoleculeView                         query  = getMolecule(queryBatch, queryMolIdx);
  BitMatrix2DView<MaxTarget, MaxQuery> view(matrix);

  // Clear first (only thread 0)
  if (threadIdx.x == 0) {
    view.clear();
  }
  __syncthreads();

  // Each thread handles one target atom
  int targetAtomIdx = threadIdx.x;
  nvMolKit::populateLabelMatrixForAtom<MaxTarget, MaxQuery>(target, targetAtomIdx, query, view);
}

TEST_F(GraphLabelerTest, ParallelLabeling) {
  auto targetMol = makeMolFromSmiles("c1ccccc1");  // benzene
  auto queryMol  = makeMolFromSmarts("c");
  ASSERT_NE(targetMol, nullptr);
  ASSERT_NE(queryMol, nullptr);

  MoleculesHost targetHost;
  MoleculesHost queryHost;
  addToBatch(targetMol.get(), targetHost);
  addQueryToBatch(queryMol.get(), queryHost);

  MoleculesDevice targetDevice(stream_.stream());
  MoleculesDevice queryDevice(stream_.stream());
  targetDevice.copyFromHost(targetHost);
  queryDevice.copyFromHost(queryHost);

  AsyncDeviceVector<LabelMatrixStorage> matrixDev(1, stream_.stream());
  LabelMatrixStorage                    hostMatrix(false);
  matrixDev.setFromVector(std::vector<LabelMatrixStorage>{hostMatrix});

  // Launch with enough threads for all target atoms
  const int numTargetAtoms = static_cast<int>(targetHost.totalAtoms());
  populateLabelMatrixParallelKernel<kMaxTargetAtoms, kMaxQueryAtoms>
    <<<1, numTargetAtoms, 0, stream_.stream()>>>(targetDevice.view(), 0, queryDevice.view(), 0, matrixDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<LabelMatrixStorage> resultMatrix(1);
  matrixDev.copyToHost(resultMatrix);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  LabelMatrixView view(resultMatrix[0]);

  // All 6 aromatic carbons should match the aromatic carbon query
  for (int i = 0; i < 6; ++i) {
    EXPECT_TRUE(view.get(i, 0)) << "Atom " << i << " should match aromatic carbon query";
  }
}

// =============================================================================
// Shared Memory Labeling Test
// =============================================================================

template <std::size_t MaxTarget, std::size_t MaxQuery>
__global__ void populateLabelMatrixSharedMemKernel(MoleculesDeviceView targetBatch,
                                                   int                 targetMolIdx,
                                                   MoleculesDeviceView queryBatch,
                                                   int                 queryMolIdx,
                                                   uint8_t*            outputBits,
                                                   int                 numTargetAtoms,
                                                   int                 numQueryAtoms) {
  __shared__ FlatBitVect<MaxTarget * MaxQuery> sharedMatrix;
  BitMatrix2DView<MaxTarget, MaxQuery>         view(&sharedMatrix);

  MoleculeView target = getMolecule(targetBatch, targetMolIdx);
  MoleculeView query  = getMolecule(queryBatch, queryMolIdx);

  // Thread 0 does the labeling
  if (threadIdx.x == 0) {
    nvMolKit::populateLabelMatrix<MaxTarget, MaxQuery>(target, query, view);
  }
  __syncthreads();

  // All threads copy out their portion
  // Note: The matrix uses linearIndex(row, col) = row * MaxQuery + col
  int idx = threadIdx.x;
  if (idx < numTargetAtoms * numQueryAtoms) {
    int targetIdx   = idx / numQueryAtoms;
    int queryIdx    = idx % numQueryAtoms;
    outputBits[idx] = sharedMatrix[targetIdx * MaxQuery + queryIdx];
  }
}

TEST_F(GraphLabelerTest, SharedMemoryLabeling) {
  auto targetMol = makeMolFromSmiles("CCO");
  auto queryMol  = makeMolFromSmarts("C");
  ASSERT_NE(targetMol, nullptr);
  ASSERT_NE(queryMol, nullptr);

  MoleculesHost targetHost;
  MoleculesHost queryHost;
  addToBatch(targetMol.get(), targetHost);
  addQueryToBatch(queryMol.get(), queryHost);

  MoleculesDevice targetDevice(stream_.stream());
  MoleculesDevice queryDevice(stream_.stream());
  targetDevice.copyFromHost(targetHost);
  queryDevice.copyFromHost(queryHost);

  const int numTargetAtoms = static_cast<int>(targetHost.totalAtoms());
  const int numQueryAtoms  = static_cast<int>(queryHost.totalAtoms());

  AsyncDeviceVector<uint8_t> outputDev(numTargetAtoms * numQueryAtoms, stream_.stream());

  populateLabelMatrixSharedMemKernel<kMaxTargetAtoms, kMaxQueryAtoms>
    <<<1, 32, 0, stream_.stream()>>>(targetDevice.view(),
                                     0,
                                     queryDevice.view(),
                                     0,
                                     outputDev.data(),
                                     numTargetAtoms,
                                     numQueryAtoms);
  cudaCheckError(cudaGetLastError());

  std::vector<uint8_t> output(numTargetAtoms * numQueryAtoms);
  outputDev.copyToHost(output);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  // CCO with query C: first two atoms (carbons) should match, third (oxygen) should not
  EXPECT_TRUE(output[0]);   // C matches C
  EXPECT_TRUE(output[1]);   // C matches C
  EXPECT_FALSE(output[2]);  // O doesn't match C
}

// =============================================================================
// Bond Count Matching Tests
// =============================================================================

TEST_F(GraphLabelerTest, BondCountsPreventMatch) {
  // Target: C (methane - 0 bonds to heavy atoms)
  // Query: CC (each carbon has 1 bond)
  // Methane's carbon can't match because it has fewer bonds
  std::vector<std::vector<uint8_t>> expected = {
    {false, false}
  };
  runLabelingTest("C", "CC", expected);
}

TEST_F(GraphLabelerTest, CentralCarbonHasMoreBonds) {
  // Target: CC(C)C (isobutane) - central carbon has 3 bonds
  // Query: CC (each carbon has 1 bond)
  // All carbons should match query since they all have >= 1 bond
  std::vector<std::vector<uint8_t>> expected = {
    {true, true}, // Terminal C (1 bond)
    {true, true}, // Central C (3 bonds)
    {true, true}, // Terminal C (1 bond)
    {true, true}  // Terminal C (1 bond)
  };
  runLabelingTest("CC(C)C", "CC", expected);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(GraphLabelerTest, SingleAtomTargetAndQuery) {
  std::vector<std::vector<uint8_t>> expected = {{true}};
  runLabelingTest("C", "C", expected);
}

TEST_F(GraphLabelerTest, SingleAtomNoMatch) {
  std::vector<std::vector<uint8_t>> expected = {{false}};
  runLabelingTest("C", "N", expected);
}

// =============================================================================
// Hydrogen Count Query Tests
// =============================================================================

// Note: SMARTS H count queries like [CH3] check total H count (explicit + implicit).
// For molecules with explicit Hs in SMILES, these work correctly.

TEST_F(GraphLabelerTest, HCountExplicitHydrogens) {
  // Target with explicit Hs: [CH4] (methane with explicit Hs)
  // Query: [CH4] (carbon with 4 hydrogens)
  std::vector<std::vector<uint8_t>> expected = {
    {true}   // C with 4 explicit Hs
  };
  runLabelingTest("[CH4]", "[CH4]", expected);
}

TEST_F(GraphLabelerTest, HCountExplicitMismatch) {
  // Target with explicit Hs: [CH4] (methane)
  // Query: [CH3] (carbon with 3 hydrogens)
  std::vector<std::vector<uint8_t>> expected = {
    {false}   // C with 4 Hs doesn't match [CH3]
  };
  runLabelingTest("[CH4]", "[CH3]", expected);
}

// =============================================================================
// Formal Charge Query Tests
// =============================================================================

TEST_F(GraphLabelerTest, FormalChargePositive) {
  // Target: [NH4+] (ammonium)
  // Query: [+1] (any atom with +1 charge)
  std::vector<std::vector<uint8_t>> expected = {
    {true}   // N has +1 charge
  };
  runLabelingTest("[NH4+]", "[+1]", expected);
}

TEST_F(GraphLabelerTest, FormalChargeNegative) {
  // Target: [O-]C=O (formate)
  // Query: [-1] (any atom with -1 charge)
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // O- has -1 charge
    {false},  // C has 0 charge
    {false}   // O has 0 charge
  };
  runLabelingTest("[O-]C=O", "[-1]", expected);
}

TEST_F(GraphLabelerTest, FormalChargeNeutralMolecule) {
  // Target: CCO (ethanol - all neutral)
  // Query: [+1] (any atom with +1 charge)
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // C
    {false},  // C
    {false}   // O
  };
  runLabelingTest("CCO", "[+1]", expected);
}

// =============================================================================
// Hybridization Query Tests
// =============================================================================

TEST_F(GraphLabelerTest, HybridizationSP3) {
  // Target: CCO (ethanol - all sp3)
  // Query: [^3] (sp3 hybridized)
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // C sp3
    {true},   // C sp3
    {true}    // O sp3
  };
  runLabelingTest("CCO", "[^3]", expected);
}

TEST_F(GraphLabelerTest, HybridizationSP2) {
  // Target: C=CC (propene)
  // Query: [^2] (sp2 hybridized)
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // C= sp2
    {true},   // =C sp2
    {false}   // C sp3
  };
  runLabelingTest("C=CC", "[^2]", expected);
}

TEST_F(GraphLabelerTest, HybridizationSP) {
  // Target: C#CC (propyne)
  // Query: [^1] (sp hybridized)
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // C# sp
    {true},   // #C sp
    {false}   // C sp3
  };
  runLabelingTest("C#CC", "[^1]", expected);
}

// =============================================================================
// Ring Membership Query Tests
// =============================================================================

TEST_F(GraphLabelerTest, RingMembershipAnyRing) {
  // Target: C1CCC1C (cyclobutane with methyl)
  // Query: [R1] (atom in exactly 1 ring)
  // Note: [R] without a number uses >= semantics which we don't support; use [R1] instead
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // Ring C
    {true},   // Ring C
    {true},   // Ring C
    {true},   // Ring C
    {false}   // Methyl C (not in ring)
  };
  runLabelingTest("C1CCC1C", "[R1]", expected);
}

TEST_F(GraphLabelerTest, RingMembershipNotInRing) {
  // Target: C1CCC1C (cyclobutane with methyl)
  // Query: [R0] (atom not in any ring)
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // Ring C
    {false},  // Ring C
    {false},  // Ring C
    {false},  // Ring C
    {true}    // Methyl C (not in ring)
  };
  runLabelingTest("C1CCC1C", "[R0]", expected);
}

TEST_F(GraphLabelerTest, RingMembershipExactlyOneRing) {
  // Target: c1ccccc1 (benzene - each atom in 1 ring)
  // Query: [R1] (atom in exactly 1 ring)
  std::vector<std::vector<uint8_t>> expected = {
    {true}, {true}, {true}, {true}, {true}, {true}
  };
  runLabelingTest("c1ccccc1", "[R1]", expected);
}

TEST_F(GraphLabelerTest, RingMembershipTwoRings) {
  // Target: c1ccc2ccccc2c1 (naphthalene - fused atoms in 2 rings)
  // Query: [R2] (atom in exactly 2 rings)
  // Atom order: c0,c1,c2,c3(fusion),c4,c5,c6,c7,c8(fusion),c9
  // Fusion atoms 3 and 8 are in 2 rings
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c0
    {false},  // c1
    {false},  // c2
    {true},   // c3 (fusion)
    {false},  // c4
    {false},  // c5
    {false},  // c6
    {false},  // c7
    {true},   // c8 (fusion)
    {false}   // c9
  };
  runLabelingTest("c1ccc2ccccc2c1", "[R2]", expected);
}

TEST_F(GraphLabelerTest, RingMembershipIndole) {
  // Target: c1ccc2[nH]ccc2c1 (indole - 6-membered benzene fused with 5-membered pyrrole)
  // Query: [R2] (atom in exactly 2 rings)
  // Atom order: c0,c1,c2,c3(fusion),[nH]4,c5,c6,c7(fusion),c8
  // Fusion atoms 3 and 7 are in 2 rings
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c0 (benzene only)
    {false},  // c1 (benzene only)
    {false},  // c2 (benzene only)
    {true},   // c3 (fusion - benzene/pyrrole)
    {false},  // nH (pyrrole only)
    {false},  // c5 (pyrrole only)
    {false},  // c6 (pyrrole only)
    {true},   // c7 (fusion - benzene/pyrrole)
    {false}   // c8 (benzene only)
  };
  runLabelingTest("c1ccc2[nH]ccc2c1", "[R2]", expected);
}

TEST_F(GraphLabelerTest, RingMembershipIndoleSingleRing) {
  // Target: c1ccc2[nH]ccc2c1 (indole)
  // Query: [R1] (atom in exactly 1 ring)
  // Non-fusion atoms are in exactly 1 ring
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // c0 (benzene only)
    {true},   // c1 (benzene only)
    {true},   // c2 (benzene only)
    {false},  // c3 (fusion - in 2 rings)
    {true},   // nH (pyrrole only)
    {true},   // c5 (pyrrole only)
    {true},   // c6 (pyrrole only)
    {false},  // c7 (fusion - in 2 rings)
    {true}    // c8 (benzene only)
  };
  runLabelingTest("c1ccc2[nH]ccc2c1", "[R1]", expected);
}

// =============================================================================
// Ring Size Query Tests
// =============================================================================

TEST_F(GraphLabelerTest, RingSizeFive) {
  // Target: C1CCCC1 (cyclopentane)
  // Query: [r5] (atom in 5-membered ring)
  std::vector<std::vector<uint8_t>> expected = {
    {true}, {true}, {true}, {true}, {true}
  };
  runLabelingTest("C1CCCC1", "[r5]", expected);
}

TEST_F(GraphLabelerTest, RingSizeSix) {
  // Target: c1ccccc1 (benzene)
  // Query: [r6] (atom in 6-membered ring)
  std::vector<std::vector<uint8_t>> expected = {
    {true}, {true}, {true}, {true}, {true}, {true}
  };
  runLabelingTest("c1ccccc1", "[r6]", expected);
}

TEST_F(GraphLabelerTest, RingSizeMismatch) {
  // Target: C1CCCC1 (cyclopentane - 5-membered)
  // Query: [r6] (atom in 6-membered ring)
  std::vector<std::vector<uint8_t>> expected = {
    {false}, {false}, {false}, {false}, {false}
  };
  runLabelingTest("C1CCCC1", "[r6]", expected);
}

TEST_F(GraphLabelerTest, RingSizeIndoleFiveMembered) {
  // Target: c1ccc2[nH]ccc2c1 (indole)
  // Query: [r5] (atom in 5-membered ring)
  // Pyrrole atoms (including fusion) have min ring size 5
  // Benzene-only atoms have min ring size 6
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c0 (benzene, min size 6)
    {false},  // c1 (benzene, min size 6)
    {false},  // c2 (benzene, min size 6)
    {true},   // c3 (fusion, min size 5 from pyrrole)
    {true},   // nH (pyrrole, min size 5)
    {true},   // c5 (pyrrole, min size 5)
    {true},   // c6 (pyrrole, min size 5)
    {true},   // c7 (fusion, min size 5 from pyrrole)
    {false}   // c8 (benzene, min size 6)
  };
  runLabelingTest("c1ccc2[nH]ccc2c1", "[r5]", expected);
}

TEST_F(GraphLabelerTest, RingSizeIndoleSixMembered) {
  // Target: c1ccc2[nH]ccc2c1 (indole)
  // Query: [r6] (atom in 6-membered ring)
  // Only benzene-only atoms have min ring size 6
  // Fusion and pyrrole atoms have min ring size 5
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // c0 (benzene, min size 6)
    {true},   // c1 (benzene, min size 6)
    {true},   // c2 (benzene, min size 6)
    {false},  // c3 (fusion, min size 5)
    {false},  // nH (pyrrole, min size 5)
    {false},  // c5 (pyrrole, min size 5)
    {false},  // c6 (pyrrole, min size 5)
    {false},  // c7 (fusion, min size 5)
    {true}    // c8 (benzene, min size 6)
  };
  runLabelingTest("c1ccc2[nH]ccc2c1", "[r6]", expected);
}

// =============================================================================
// Wildcard and Any Aromaticity Query Tests
// =============================================================================

TEST_F(GraphLabelerTest, WildcardMatchesAll) {
  // Target: CCO (ethanol)
  // Query: [*] (any atom)
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // C
    {true},   // C
    {true}    // O
  };
  runLabelingTest("CCO", "[*]", expected);
}

TEST_F(GraphLabelerTest, WildcardMatchesMixed) {
  // Target: c1ccccc1C (toluene - mixed aromaticity)
  // Query: [*] (any atom)
  std::vector<std::vector<uint8_t>> expected = {
    {true}, {true}, {true}, {true}, {true}, {true}, {true}
  };
  runLabelingTest("c1ccccc1C", "[*]", expected);
}

TEST_F(GraphLabelerTest, AnyAromaticAtom) {
  // Target: c1ccccc1C (toluene)
  // Query: [a] (any aromatic atom)
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // c
    {true},   // c
    {true},   // c
    {true},   // c
    {true},   // c
    {true},   // c
    {false}   // C (aliphatic)
  };
  runLabelingTest("c1ccccc1C", "[a]", expected);
}

TEST_F(GraphLabelerTest, AnyAliphaticAtom) {
  // Target: c1ccccc1C (toluene)
  // Query: [A] (any aliphatic atom)
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c
    {false},  // c
    {false},  // c
    {false},  // c
    {false},  // c
    {false},  // c
    {true}    // C (aliphatic)
  };
  runLabelingTest("c1ccccc1C", "[A]", expected);
}

// =============================================================================
// Explicit AND Query Tests
// =============================================================================

TEST_F(GraphLabelerTest, ExplicitAndAmpersandCarbonInRing) {
  // Target: C1CCC1C (cyclobutane with methyl)
  // Query: [C&R1] (aliphatic carbon AND in exactly 1 ring)
  // Note: [R] uses >= semantics; use [R1] for exact matching
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // Ring C
    {true},   // Ring C
    {true},   // Ring C
    {true},   // Ring C
    {false}   // Methyl C (not in ring)
  };
  runLabelingTest("C1CCC1C", "[C&R1]", expected);
}

TEST_F(GraphLabelerTest, ExplicitAndSemicolonAromaticInRing) {
  // Target: c1ccccc1C (toluene)
  // Query: [c;R1] (aromatic carbon AND in exactly 1 ring)
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // c in ring
    {true},   // c in ring
    {true},   // c in ring
    {true},   // c in ring
    {true},   // c in ring
    {true},   // c in ring
    {false}   // C (aliphatic, not matching aromatic query)
  };
  runLabelingTest("c1ccccc1C", "[c;R1]", expected);
}

TEST_F(GraphLabelerTest, ExplicitAndAtomicNumRing) {
  // Target: c1ccncc1 (pyridine)
  // Query: [#6;R1] (carbon AND in exactly 1 ring)
  // Note: [R] uses >= semantics; use [R1] for exact matching
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // c
    {true},   // c
    {true},   // c
    {false},  // n (nitrogen, not carbon)
    {true},   // c
    {true}    // c
  };
  runLabelingTest("c1ccncc1", "[#6;R1]", expected);
}

TEST_F(GraphLabelerTest, ExplicitAndHCountAromaticity) {
  // Target: c1ccccc1[CH3] (toluene with explicit methyl Hs)
  // Query: [CH3] - aliphatic carbon with 3 Hs
  // Note: H count matching requires explicit Hs in target SMILES
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c (aromatic)
    {false},  // c
    {false},  // c
    {false},  // c
    {false},  // c
    {false},  // c
    {true}    // [CH3] (methyl with explicit Hs)
  };
  runLabelingTest("c1ccccc1[CH3]", "[CH3]", expected);
}

TEST_F(GraphLabelerTest, CombinedRingMembershipAndSize) {
  // Target: c1ccccc1 (benzene)
  // Query: [R1;r6] (in exactly 1 ring AND ring size 6)
  std::vector<std::vector<uint8_t>> expected = {
    {true}, {true}, {true}, {true}, {true}, {true}
  };
  runLabelingTest("c1ccccc1", "[R1;r6]", expected);
}

// =============================================================================
// Multiple Chained AND Query Tests (3+ conditions)
// =============================================================================

TEST_F(GraphLabelerTest, TripleAndAromaticRingRingSize) {
  // Target: c1ccccc1 (benzene) - aromatic, in 1 ring, ring size 6
  // Query: [c&R1&r6] (aromatic carbon AND in 1 ring AND ring size 6)
  std::vector<std::vector<uint8_t>> expected = {
    {true}, {true}, {true}, {true}, {true}, {true}
  };
  runLabelingTest("c1ccccc1", "[c&R1&r6]", expected);
}

TEST_F(GraphLabelerTest, TripleAndNoMatch) {
  // Target: c1ccccc1 (benzene) - 6-membered ring
  // Query: [c&R1&r5] (aromatic carbon AND in 1 ring AND ring size 5)
  // No match because benzene is 6-membered, not 5
  std::vector<std::vector<uint8_t>> expected = {
    {false}, {false}, {false}, {false}, {false}, {false}
  };
  runLabelingTest("c1ccccc1", "[c&R1&r5]", expected);
}

TEST_F(GraphLabelerTest, TripleAndCyclopentane) {
  // Target: C1CCCC1 (cyclopentane) - aliphatic, in 1 ring, ring size 5, sp3
  // Query: [C&R1&r5] (aliphatic carbon AND in 1 ring AND ring size 5)
  std::vector<std::vector<uint8_t>> expected = {
    {true}, {true}, {true}, {true}, {true}
  };
  runLabelingTest("C1CCCC1", "[C&R1&r5]", expected);
}

TEST_F(GraphLabelerTest, TripleAndWithHybridization) {
  // Target: C1CCCC1 (cyclopentane) - all sp3
  // Query: [C&R1&^3] (aliphatic carbon AND in 1 ring AND sp3)
  std::vector<std::vector<uint8_t>> expected = {
    {true}, {true}, {true}, {true}, {true}
  };
  runLabelingTest("C1CCCC1", "[C&R1&^3]", expected);
}

TEST_F(GraphLabelerTest, TripleAndSemicolonSyntax) {
  // Target: C1CCCC1 (cyclopentane)
  // Query: [#6;R1;r5] (carbon AND in 1 ring AND ring size 5) using semicolon
  std::vector<std::vector<uint8_t>> expected = {
    {true}, {true}, {true}, {true}, {true}
  };
  runLabelingTest("C1CCCC1", "[#6;R1;r5]", expected);
}

TEST_F(GraphLabelerTest, TripleAndIndolePyrrole) {
  // Target: c1ccc2[nH]ccc2c1 (indole)
  // Query: [c&R1&r5] (aromatic carbon AND in exactly 1 ring AND ring size 5)
  // Only pyrrole carbons (not at fusion) match: atoms 5, 6
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c0 (benzene, r6)
    {false},  // c1 (benzene, r6)
    {false},  // c2 (benzene, r6)
    {false},  // c3 (fusion, R2)
    {false},  // nH (nitrogen, not carbon)
    {true},   // c5 (pyrrole, R1, r5)
    {true},   // c6 (pyrrole, R1, r5)
    {false},  // c7 (fusion, R2)
    {false}   // c8 (benzene, r6)
  };
  runLabelingTest("c1ccc2[nH]ccc2c1", "[c&R1&r5]", expected);
}

// =============================================================================
// Total Valence Tests
// =============================================================================

TEST_F(GraphLabelerTest, TotalValenceMethane) {
  // Methane: C with 4 hydrogens, total valence = 4
  std::vector<std::vector<uint8_t>> expected = {{true}};
  runLabelingTest("C", "[v4]", expected);
}

TEST_F(GraphLabelerTest, TotalValenceAmmonia) {
  // Ammonia: N with 3 hydrogens, total valence = 3
  std::vector<std::vector<uint8_t>> expected = {{true}};
  runLabelingTest("N", "[v3]", expected);
}

TEST_F(GraphLabelerTest, TotalValenceWater) {
  // Water: O with 2 hydrogens, total valence = 2
  std::vector<std::vector<uint8_t>> expected = {{true}};
  runLabelingTest("O", "[v2]", expected);
}

TEST_F(GraphLabelerTest, TotalValenceMismatch) {
  // Methane has valence 4, not 3
  std::vector<std::vector<uint8_t>> expected = {{false}};
  runLabelingTest("C", "[v3]", expected);
}

TEST_F(GraphLabelerTest, TotalValenceWithAtomType) {
  // Carbon with valence 4
  std::vector<std::vector<uint8_t>> expected = {{true}};
  runLabelingTest("C", "[C&v4]", expected);
}

TEST_F(GraphLabelerTest, TotalValenceEthane) {
  // Ethane: two carbons, each with valence 4
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}};
  runLabelingTest("CC", "[v4]", expected);
}

TEST_F(GraphLabelerTest, TotalValenceEthene) {
  // Ethene: two sp2 carbons, each with valence 4 (double bond counts as 2)
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}};
  runLabelingTest("C=C", "[v4]", expected);
}

TEST_F(GraphLabelerTest, TotalValenceFormaldehyde) {
  // Formaldehyde CH2O: C has valence 4, O has valence 2
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // C (valence 4)
    {false}  // O (valence 2)
  };
  runLabelingTest("C=O", "[v4]", expected);
}

// =============================================================================
// GPU-Optimized Warp-Parallel Labeling Tests
// =============================================================================

template <std::size_t MaxTarget, std::size_t MaxQuery>
__global__ void populateLabelMatrixOptimizedKernel(MoleculesDeviceView                targetBatch,
                                                   int                                targetMolIdx,
                                                   MoleculesDeviceView                queryBatch,
                                                   int                                queryMolIdx,
                                                   FlatBitVect<MaxTarget * MaxQuery>* matrix) {
  MoleculeView                         target = getMolecule(targetBatch, targetMolIdx);
  MoleculeView                         query  = getMolecule(queryBatch, queryMolIdx);
  BitMatrix2DView<MaxTarget, MaxQuery> view(matrix);

  nvMolKit::populateLabelMatrixOptimized<MaxTarget, MaxQuery>(target, query, view);
}

TEST_F(GraphLabelerTest, OptimizedWarpParallelLabeling) {
  auto targetMol = makeMolFromSmiles("c1ccccc1");  // benzene
  auto queryMol  = makeMolFromSmarts("c");
  ASSERT_NE(targetMol, nullptr);
  ASSERT_NE(queryMol, nullptr);

  MoleculesHost targetHost;
  MoleculesHost queryHost;
  addToBatch(targetMol.get(), targetHost);
  addQueryToBatch(queryMol.get(), queryHost);

  MoleculesDevice targetDevice(stream_.stream());
  MoleculesDevice queryDevice(stream_.stream());
  targetDevice.copyFromHost(targetHost);
  queryDevice.copyFromHost(queryHost);

  AsyncDeviceVector<LabelMatrixStorage> matrixDev(1, stream_.stream());
  LabelMatrixStorage                    hostMatrix(false);
  matrixDev.setFromVector(std::vector<LabelMatrixStorage>{hostMatrix});

  // Launch with multiple warps (128 threads = 4 warps)
  populateLabelMatrixOptimizedKernel<kMaxTargetAtoms, kMaxQueryAtoms>
    <<<1, 128, 0, stream_.stream()>>>(targetDevice.view(), 0, queryDevice.view(), 0, matrixDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<LabelMatrixStorage> resultMatrix(1);
  matrixDev.copyToHost(resultMatrix);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  LabelMatrixView view(resultMatrix[0]);

  // All 6 aromatic carbons should match the aromatic carbon query
  for (int i = 0; i < 6; ++i) {
    EXPECT_TRUE(view.get(i, 0)) << "Atom " << i << " should match aromatic carbon query";
  }
}

TEST_F(GraphLabelerTest, OptimizedLabelingMultiAtomQuery) {
  // Target: CCC (propane) - 3 carbons
  // Query: CC (two bonded carbons)
  auto targetMol = makeMolFromSmiles("CCC");
  auto queryMol  = makeMolFromSmarts("CC");
  ASSERT_NE(targetMol, nullptr);
  ASSERT_NE(queryMol, nullptr);

  MoleculesHost targetHost;
  MoleculesHost queryHost;
  addToBatch(targetMol.get(), targetHost);
  addQueryToBatch(queryMol.get(), queryHost);

  MoleculesDevice targetDevice(stream_.stream());
  MoleculesDevice queryDevice(stream_.stream());
  targetDevice.copyFromHost(targetHost);
  queryDevice.copyFromHost(queryHost);

  AsyncDeviceVector<LabelMatrixStorage> matrixDev(1, stream_.stream());
  LabelMatrixStorage                    hostMatrix(false);
  matrixDev.setFromVector(std::vector<LabelMatrixStorage>{hostMatrix});

  populateLabelMatrixOptimizedKernel<kMaxTargetAtoms, kMaxQueryAtoms>
    <<<1, 128, 0, stream_.stream()>>>(targetDevice.view(), 0, queryDevice.view(), 0, matrixDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<LabelMatrixStorage> resultMatrix(1);
  matrixDev.copyToHost(resultMatrix);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  LabelMatrixView view(resultMatrix[0]);

  // All 3 carbons should match both query atoms (each has >= 1 bond)
  for (int t = 0; t < 3; ++t) {
    for (int q = 0; q < 2; ++q) {
      EXPECT_TRUE(view.get(t, q)) << "Target atom " << t << " should match query atom " << q;
    }
  }
}

TEST_F(GraphLabelerTest, OptimizedLabelingNoMatches) {
  // Target: C (methane) - single carbon with 0 bonds to heavy atoms
  // Query: CC (two bonded carbons, each with 1 bond)
  auto targetMol = makeMolFromSmiles("C");
  auto queryMol  = makeMolFromSmarts("CC");
  ASSERT_NE(targetMol, nullptr);
  ASSERT_NE(queryMol, nullptr);

  MoleculesHost targetHost;
  MoleculesHost queryHost;
  addToBatch(targetMol.get(), targetHost);
  addQueryToBatch(queryMol.get(), queryHost);

  MoleculesDevice targetDevice(stream_.stream());
  MoleculesDevice queryDevice(stream_.stream());
  targetDevice.copyFromHost(targetHost);
  queryDevice.copyFromHost(queryHost);

  AsyncDeviceVector<LabelMatrixStorage> matrixDev(1, stream_.stream());
  LabelMatrixStorage                    hostMatrix(false);
  matrixDev.setFromVector(std::vector<LabelMatrixStorage>{hostMatrix});

  populateLabelMatrixOptimizedKernel<kMaxTargetAtoms, kMaxQueryAtoms>
    <<<1, 128, 0, stream_.stream()>>>(targetDevice.view(), 0, queryDevice.view(), 0, matrixDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<LabelMatrixStorage> resultMatrix(1);
  matrixDev.copyToHost(resultMatrix);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  LabelMatrixView view(resultMatrix[0]);

  // Methane's carbon has 0 bonds, can't match query atoms with 1 bond
  EXPECT_FALSE(view.get(0, 0));
  EXPECT_FALSE(view.get(0, 1));
}

TEST_F(GraphLabelerTest, OptimizedLabelingMixedMatch) {
  // Target: CCO (ethanol) - 2 carbons, 1 oxygen
  // Query: O (oxygen)
  auto targetMol = makeMolFromSmiles("CCO");
  auto queryMol  = makeMolFromSmarts("O");
  ASSERT_NE(targetMol, nullptr);
  ASSERT_NE(queryMol, nullptr);

  MoleculesHost targetHost;
  MoleculesHost queryHost;
  addToBatch(targetMol.get(), targetHost);
  addQueryToBatch(queryMol.get(), queryHost);

  MoleculesDevice targetDevice(stream_.stream());
  MoleculesDevice queryDevice(stream_.stream());
  targetDevice.copyFromHost(targetHost);
  queryDevice.copyFromHost(queryHost);

  AsyncDeviceVector<LabelMatrixStorage> matrixDev(1, stream_.stream());
  LabelMatrixStorage                    hostMatrix(false);
  matrixDev.setFromVector(std::vector<LabelMatrixStorage>{hostMatrix});

  populateLabelMatrixOptimizedKernel<kMaxTargetAtoms, kMaxQueryAtoms>
    <<<1, 128, 0, stream_.stream()>>>(targetDevice.view(), 0, queryDevice.view(), 0, matrixDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<LabelMatrixStorage> resultMatrix(1);
  matrixDev.copyToHost(resultMatrix);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  LabelMatrixView view(resultMatrix[0]);

  // Only oxygen should match
  EXPECT_FALSE(view.get(0, 0));  // C doesn't match O
  EXPECT_FALSE(view.get(1, 0));  // C doesn't match O
  EXPECT_TRUE(view.get(2, 0));   // O matches O
}

// =============================================================================
// Packed Atom Data Tests
// =============================================================================

TEST(AtomDataPackedTest, PackedDataRoundTrip) {
  nvMolKit::AtomDataPacked packed;
  packed.setAtomicNum(6);
  packed.setNumExplicitHs(2);
  packed.setExplicitValence(4);
  packed.setImplicitValence(0);
  packed.setFormalCharge(-1);
  packed.setChiralTag(1);
  packed.setNumRadicalElectrons(0);
  packed.setHybridization(3);
  packed.setMinRingSize(6);
  packed.setNumRings(1);
  packed.setIsAromatic(true);

  EXPECT_EQ(packed.atomicNum(), 6);
  EXPECT_EQ(packed.numExplicitHs(), 2);
  EXPECT_EQ(packed.explicitValence(), 4);
  EXPECT_EQ(packed.implicitValence(), 0);
  EXPECT_EQ(packed.formalCharge(), -1);
  EXPECT_EQ(packed.chiralTag(), 1);
  EXPECT_EQ(packed.numRadicalElectrons(), 0);
  EXPECT_EQ(packed.hybridization(), 3);
  EXPECT_EQ(packed.minRingSize(), 6);
  EXPECT_EQ(packed.numRings(), 1);
  EXPECT_TRUE(packed.isAromatic());
}

TEST(AtomQueryMaskTest, BuildQueryMaskAtomicNum) {
  nvMolKit::AtomDataPacked queryAtom;
  queryAtom.setAtomicNum(6);

  nvMolKit::AtomQueryMask mask = nvMolKit::buildQueryMask(queryAtom, nvMolKit::AtomQueryAtomicNum);

  // Mask should have 0xFF in byte 0 (atomicNum position)
  EXPECT_EQ(mask.maskLo & 0xFF, 0xFF);
  EXPECT_EQ(mask.expectedLo & 0xFF, 6);

  // Rest should be zero
  EXPECT_EQ(mask.maskHi, 0);
  EXPECT_EQ(mask.expectedHi, 0);
}

TEST(AtomQueryMaskTest, BuildQueryMaskAromatic) {
  nvMolKit::AtomDataPacked queryAtom;
  queryAtom.setIsAromatic(true);

  nvMolKit::AtomQueryMask mask = nvMolKit::buildQueryMask(queryAtom, nvMolKit::AtomQueryIsAromatic);

  // Mask should have 0xFF in byte 2 of hi (isAromatic position)
  EXPECT_EQ((mask.maskHi >> 16) & 0xFF, 0xFF);
  EXPECT_EQ((mask.expectedHi >> 16) & 0xFF, 0x01);  // expect true

  // Rest should be zero
  EXPECT_EQ(mask.maskLo, 0);
  EXPECT_EQ(mask.expectedLo, 0);
}

TEST(BondTypeCountsTest, SufficientBonds) {
  nvMolKit::BondTypeCounts target;
  target.single  = 2;  // 2 single bonds
  target.double_ = 1;  // 1 double bond

  nvMolKit::BondTypeCounts query;
  query.single  = 1;  // 1 single bond
  query.double_ = 1;  // 1 double bond

  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, InsufficientBonds) {
  nvMolKit::BondTypeCounts target;
  target.single = 1;  // 1 single bond

  nvMolKit::BondTypeCounts query;
  query.single = 2;  // 2 single bonds needed

  EXPECT_FALSE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, AromaticBonds) {
  nvMolKit::BondTypeCounts target;
  target.aromatic = 2;  // 2 aromatic bonds

  nvMolKit::BondTypeCounts query;
  query.aromatic = 2;  // need 2 aromatic bonds

  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, AromaticInsufficientBonds) {
  nvMolKit::BondTypeCounts target;
  target.aromatic = 1;  // 1 aromatic bond

  nvMolKit::BondTypeCounts query;
  query.aromatic = 2;  // need 2 aromatic bonds

  EXPECT_FALSE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, AnyBondMatchesRemaining) {
  // Target has 2 single, 1 double = 3 total
  nvMolKit::BondTypeCounts target;
  target.single  = 2;
  target.double_ = 1;

  // Query needs 1 single, 1 "any" = 2 total
  nvMolKit::BondTypeCounts query;
  query.single = 1;
  query.any    = 1;  // "any" bond from SMARTS (~)

  // Should match: 1 single satisfied, "any" can use remaining (1 single or 1 double)
  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, AnyBondInsufficientTotal) {
  // Target has only 1 single bond
  nvMolKit::BondTypeCounts target;
  target.single = 1;

  // Query needs 1 single + 1 "any" = 2 total
  nvMolKit::BondTypeCounts query;
  query.single = 1;
  query.any    = 1;

  // Should fail: after satisfying single requirement, no bonds left for "any"
  EXPECT_FALSE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, AnyBondMatchesAromatic) {
  // Target has 2 aromatic bonds
  nvMolKit::BondTypeCounts target;
  target.aromatic = 2;

  // Query needs 1 "any" = 1 total
  nvMolKit::BondTypeCounts query;
  query.any = 1;

  // Should match: "any" can match the aromatic bond
  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedSingleDoubleTargetHasMore) {
  // Target has 2 single, 2 double
  nvMolKit::BondTypeCounts target;
  target.single  = 2;
  target.double_ = 2;

  // Query needs 1 single, 1 double
  nvMolKit::BondTypeCounts query;
  query.single  = 1;
  query.double_ = 1;

  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedSingleDoubleQueryNeedsMore) {
  // Target has 1 single, 1 double
  nvMolKit::BondTypeCounts target;
  target.single  = 1;
  target.double_ = 1;

  // Query needs 2 single, 1 double - not enough singles
  nvMolKit::BondTypeCounts query;
  query.single  = 2;
  query.double_ = 1;

  EXPECT_FALSE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedSingleTripleTargetHasMore) {
  // Target has 3 single, 1 triple
  nvMolKit::BondTypeCounts target;
  target.single = 3;
  target.triple = 1;

  // Query needs 1 single, 1 triple
  nvMolKit::BondTypeCounts query;
  query.single = 1;
  query.triple = 1;

  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedSingleTripleQueryNeedsMore) {
  // Target has 1 single, 1 triple
  nvMolKit::BondTypeCounts target;
  target.single = 1;
  target.triple = 1;

  // Query needs 1 single, 2 triple - not enough triples
  nvMolKit::BondTypeCounts query;
  query.single = 1;
  query.triple = 2;

  EXPECT_FALSE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedDoubleAromaticTargetHasMore) {
  // Target has 1 double, 2 aromatic
  nvMolKit::BondTypeCounts target;
  target.double_  = 1;
  target.aromatic = 2;

  // Query needs 1 double, 1 aromatic
  nvMolKit::BondTypeCounts query;
  query.double_  = 1;
  query.aromatic = 1;

  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedDoubleAromaticQueryNeedsMore) {
  // Target has 1 double, 1 aromatic
  nvMolKit::BondTypeCounts target;
  target.double_  = 1;
  target.aromatic = 1;

  // Query needs 2 double, 1 aromatic - not enough doubles
  nvMolKit::BondTypeCounts query;
  query.double_  = 2;
  query.aromatic = 1;

  EXPECT_FALSE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedAllTypesPass) {
  // Target has 2 of each type
  nvMolKit::BondTypeCounts target;
  target.single   = 2;
  target.double_  = 2;
  target.triple   = 2;
  target.aromatic = 2;

  // Query needs 1 of each type
  nvMolKit::BondTypeCounts query;
  query.single   = 1;
  query.double_  = 1;
  query.triple   = 1;
  query.aromatic = 1;

  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedAllTypesFailOnOneType) {
  // Target has 2 single, 2 double, 2 triple, but only 0 aromatic
  nvMolKit::BondTypeCounts target;
  target.single   = 2;
  target.double_  = 2;
  target.triple   = 2;
  target.aromatic = 0;

  // Query needs 1 of each - will fail on aromatic
  nvMolKit::BondTypeCounts query;
  query.single   = 1;
  query.double_  = 1;
  query.triple   = 1;
  query.aromatic = 1;

  EXPECT_FALSE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedWithAnyPass) {
  // Target has 1 single, 1 double, 1 aromatic = 3 total
  nvMolKit::BondTypeCounts target;
  target.single   = 1;
  target.double_  = 1;
  target.aromatic = 1;

  // Query needs 1 single, 1 any = 2 total
  nvMolKit::BondTypeCounts query;
  query.single = 1;
  query.any    = 1;

  // Should pass: single satisfied, any can match double or aromatic
  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedWithAnyFailTotal) {
  // Target has 1 single, 1 double = 2 total
  nvMolKit::BondTypeCounts target;
  target.single  = 1;
  target.double_ = 1;

  // Query needs 1 single, 1 double, 1 any = 3 total
  nvMolKit::BondTypeCounts query;
  query.single  = 1;
  query.double_ = 1;
  query.any     = 1;

  // Should fail: specific types satisfied but not enough total for any
  EXPECT_FALSE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedWithAnyFailSpecific) {
  // Target has 2 single, 1 aromatic = 3 total
  nvMolKit::BondTypeCounts target;
  target.single   = 2;
  target.aromatic = 1;

  // Query needs 1 double, 1 any = 2 total
  nvMolKit::BondTypeCounts query;
  query.double_ = 1;
  query.any     = 1;

  // Should fail: no double bonds in target, even though total is enough
  EXPECT_FALSE(target.canMatchQuery(query));
}
