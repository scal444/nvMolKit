// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
//
// Unit tests for the per-helper device functions in fmcs_cuda/.  Each test
// launches a tiny __global__ driver that constructs inputs, calls one
// helper, and copies results back to host memory for assertion.  This
// file is populated incrementally as Steps 1-5 land real implementations.

#include <cooperative_groups.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <set>
#include <vector>

#include "fmcs_cuda/fmcs_grow.cuh"
#include "fmcs_cuda/fmcs_kernel.cuh"
#include "fmcs_cuda/fmcs_match.cuh"
#include "fmcs_cuda/fmcs_match_tables.cuh"
#include "fmcs_cuda/fmcs_seed.cuh"
#include "fmcs_cuda/fmcs_seed_queue.cuh"
#include "src/utils/device_vector.h"

namespace {

using nvMolKit::AsyncDevicePtr;
using nvMolKit::AsyncDeviceVector;

using mcs::fmcs::MatchResult;
using mcs::fmcs::Seed;

}  // namespace

// ---------------------------------------------------------------------------
// matchSingleBondWithinThread / matchIncrementalFastCooperative
// ---------------------------------------------------------------------------

namespace {

using mcs::fmcs::MatchTableDevice;
using mcs::fmcs::PairMatchTablesDevice;
using mcs::fmcs::SingleBondMatch;

// Tiny CSR-view used by the match helpers via duck-typing; satisfies the
// QueryTopology / TargetTopology template requirement (bondEndpoints +
// numAtoms / numBonds), with optional CSR adjacency fields.
struct TestCsrView {
  static constexpr bool kHasAdjacencyBondIndices = false;

  const std::uint32_t* bondEndpoints = nullptr;
  int                  numAtoms      = 0;
  int                  numBonds      = 0;
  const std::uint32_t* rowOffsets    = nullptr;
  const std::uint32_t* colIndices    = nullptr;
  const std::uint32_t* bondIndices   = nullptr;
};

// Match tables staged on the host and uploaded to device memory on
// demand.  Both atom and bond tables are 32-bit row-packed bitmasks
// (one bit per (q, t) pair).  Tests mutate the host staging vectors
// via the set*Bit helpers; device() uploads (if dirty) and returns the
// device view to pass to kernels.
struct ManagedMatchTables {
  std::vector<std::uint32_t>       atomHost;
  std::vector<std::uint32_t>       bondHost;
  AsyncDeviceVector<std::uint32_t> atomData;
  AsyncDeviceVector<std::uint32_t> bondData;
  int                              qNumAtoms = 0;
  int                              qNumBonds = 0;

  void allocate(int qAtoms, int tAtoms, int qBonds, int tBonds) {
    qNumAtoms                 = qAtoms;
    qNumBonds                 = qBonds;
    const int atomWordsPerRow = (tAtoms + 31) / 32;
    const int bondWordsPerRow = (tBonds + 31) / 32;
    atomHost.assign(qAtoms * atomWordsPerRow, 0);
    bondHost.assign(qBonds * bondWordsPerRow, 0);
    dev_.atoms = MatchTableDevice{nullptr, qAtoms, tAtoms, atomWordsPerRow};
    dev_.bonds = MatchTableDevice{nullptr, qBonds, tBonds, bondWordsPerRow};
    dirty_     = true;
  }

  void setAtomBit(int qAtom, int tAtom) {
    atomHost[qAtom * dev_.atoms.wordsPerRow + tAtom / 32] |= (1u << (tAtom % 32));
    dirty_ = true;
  }
  void setBondBit(int qBond, int tBond) {
    bondHost[qBond * dev_.bonds.wordsPerRow + tBond / 32] |= (1u << (tBond % 32));
    dirty_ = true;
  }
  void setAllAtomBits() {
    for (int q = 0; q < dev_.atoms.nRows; ++q)
      for (int t = 0; t < dev_.atoms.nCols; ++t)
        setAtomBit(q, t);
  }
  void setAllBondBits() {
    for (int q = 0; q < dev_.bonds.nRows; ++q)
      for (int t = 0; t < dev_.bonds.nCols; ++t)
        setBondBit(q, t);
  }

  PairMatchTablesDevice device() {
    if (dirty_) {
      atomData.setFromVector(atomHost);
      bondData.setFromVector(bondHost);
      dev_.atoms.data = atomData.data();
      dev_.bonds.data = bondData.data();
      dirty_          = false;
    }
    return dev_;
  }

 private:
  PairMatchTablesDevice dev_{};
  bool                  dirty_ = true;
};

AsyncDeviceVector<std::uint32_t> makeBondEndpointsDevice(const std::vector<std::pair<int, int>>& edges) {
  std::vector<std::uint32_t> host(edges.size());
  for (size_t i = 0; i < edges.size(); ++i) {
    host[i] = (static_cast<std::uint32_t>(edges[i].first) << 16) | static_cast<std::uint32_t>(edges[i].second);
  }
  AsyncDeviceVector<std::uint32_t> dev(edges.size());
  dev.copyFromHost(host);
  return dev;
}

// ---- matchSingleBondWithinThread ----

struct SingleBondTestOut {
  bool            ok;
  SingleBondMatch match;
};

__global__ void matchSingleBondDriver(int                   qBondIdx,
                                      int                   tBondIdx,
                                      bool                  reversed,
                                      const std::uint32_t*  qBondEndpoints,
                                      int                   qNumAtoms,
                                      int                   qNumBonds,
                                      const std::uint32_t*  tBondEndpoints,
                                      int                   tNumAtoms,
                                      int                   tNumBonds,
                                      PairMatchTablesDevice tables,
                                      SingleBondTestOut*    out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  TestCsrView     qView{qBondEndpoints, qNumAtoms, qNumBonds};
  TestCsrView     tView{tBondEndpoints, tNumAtoms, tNumBonds};
  SingleBondMatch sm{};
  out->ok    = mcs::fmcs::matchSingleBondWithinThread(qBondIdx, tBondIdx, reversed, qView, tView, tables, sm);
  out->match = sm;
}

}  // namespace

TEST(FMCSUnit, MatchSingleBondForwardOrientation) {
  // Query bond (0,1), target bond (0,1).  All atoms / bonds compatible.
  auto               qBE = makeBondEndpointsDevice({
    {0, 1}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 1}
  });
  ManagedMatchTables tables;
  tables.allocate(2, 2, 1, 1);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<SingleBondTestOut> d_out;
  matchSingleBondDriver<<<1, 1>>>(0,
                                  0,
                                  /*reversed=*/false,
                                  qBE.data(),
                                  2,
                                  1,
                                  tBE.data(),
                                  2,
                                  1,
                                  tables.device(),
                                  d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SingleBondTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_EQ(out.match.targetAtomU, 0u);  // qU=0 -> tU=0
  EXPECT_EQ(out.match.targetAtomV, 1u);  // qV=1 -> tV=1
}

TEST(FMCSUnit, MatchSingleBondReverseOrientation) {
  auto               qBE = makeBondEndpointsDevice({
    {0, 1}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 1}
  });
  ManagedMatchTables tables;
  tables.allocate(2, 2, 1, 1);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<SingleBondTestOut> d_out;
  matchSingleBondDriver<<<1, 1>>>(0,
                                  0,
                                  /*reversed=*/true,
                                  qBE.data(),
                                  2,
                                  1,
                                  tBE.data(),
                                  2,
                                  1,
                                  tables.device(),
                                  d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SingleBondTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_EQ(out.match.targetAtomU, 1u);  // qU=0 -> tV=1 (reversed)
  EXPECT_EQ(out.match.targetAtomV, 0u);  // qV=1 -> tU=0
}

TEST(FMCSUnit, MatchSingleBondBondTableRejection) {
  auto               qBE = makeBondEndpointsDevice({
    {0, 1}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 1}
  });
  ManagedMatchTables tables;
  tables.allocate(2, 2, 1, 1);
  tables.setAllAtomBits();
  // Deliberately leave bondData all zero -> bond-table reject.

  AsyncDevicePtr<SingleBondTestOut> d_out;
  matchSingleBondDriver<<<1, 1>>>(0,
                                  0,
                                  /*reversed=*/false,
                                  qBE.data(),
                                  2,
                                  1,
                                  tBE.data(),
                                  2,
                                  1,
                                  tables.device(),
                                  d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SingleBondTestOut out{};
  d_out.get(out);

  EXPECT_FALSE(out.ok);
}

TEST(FMCSUnit, MatchSingleBondAtomTableRejection) {
  auto               qBE = makeBondEndpointsDevice({
    {0, 1}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 1}
  });
  ManagedMatchTables tables;
  tables.allocate(2, 2, 1, 1);
  tables.setAllBondBits();
  // Atom 0 compatible with target atom 0, but atom 1 is incompatible
  // with target atom 1 -> forward orientation rejects on second atom.
  tables.setAtomBit(0, 0);
  tables.setAtomBit(0, 1);
  tables.setAtomBit(1, 0);
  // Note: (1, 1) deliberately left unset.

  AsyncDevicePtr<SingleBondTestOut> d_out;
  matchSingleBondDriver<<<1, 1>>>(0,
                                  0,
                                  /*reversed=*/false,
                                  qBE.data(),
                                  2,
                                  1,
                                  tBE.data(),
                                  2,
                                  1,
                                  tables.device(),
                                  d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SingleBondTestOut out{};
  d_out.get(out);

  EXPECT_FALSE(out.ok);
}

// ---- matchIncrementalFastCooperative ----

namespace {

// Helper: build a parent MatchResult that records the mapping
// {qAtom[i] -> tAtom[i]} and {qBond[i] -> tBond[i]} from caller-supplied
// parallel arrays.  Used by the incremental tests to set up "what the
// parent already had matched" before adding new bonds.
template <int maxA, int maxB, int maxTA, int maxTB>
__device__ __forceinline__ void buildParentMatch(mcs::fmcs::MatchResult<maxA, maxB, maxTA, maxTB>& match,
                                                 const int*                                        qAtoms,
                                                 const int*                                        tAtoms,
                                                 int                                               nAtomMaps,
                                                 const int*                                        qBonds,
                                                 const int*                                        tBonds,
                                                 int                                               nBondMaps) {
  using MatchT = mcs::fmcs::MatchResult<maxA, maxB, maxTA, maxTB>;
  mcs::fmcs::matchResultClearWithinThread(match);
  for (int i = 0; i < nAtomMaps; ++i) {
    match.targetAtomIdx[qAtoms[i]] = static_cast<std::uint8_t>(tAtoms[i]);
    const int t                    = tAtoms[i];
    match.visitedTargetAtoms[t / MatchT::kTargetAtomBitsPerWord] |= (typename MatchT::target_atom_word{1})
                                                                 << (t % MatchT::kTargetAtomBitsPerWord);
  }
  for (int i = 0; i < nBondMaps; ++i) {
    match.targetBondIdx[qBonds[i]] = static_cast<std::uint8_t>(tBonds[i]);
    const int t                    = tBonds[i];
    match.visitedTargetBonds[t / MatchT::kTargetBondBitsPerWord] |= (typename MatchT::target_bond_word{1})
                                                                 << (t % MatchT::kTargetBondBitsPerWord);
  }
  match.matchedAtomSize = static_cast<std::uint16_t>(nAtomMaps);
  match.matchedBondSize = static_cast<std::uint16_t>(nBondMaps);
  match.empty           = (nAtomMaps == 0 && nBondMaps == 0);
}

}  // namespace

namespace mcs_fmcs_incremental_test {

using QueuedT16 = mcs::fmcs::QueuedSeed<16, 16, 16, 16>;

struct IncrementalTestOut {
  bool      ok;
  QueuedT16 child;
};

// One-warp driver: builds parent match in shared mem, then constructs
// the child seed (parent + new bonds) and runs
// matchIncrementalFastCooperative on it.
__global__ void matchIncrementalAtomAddingDriver(const std::uint32_t*  qBE,
                                                 int                   qNumAtoms,
                                                 int                   qNumBonds,
                                                 const std::uint32_t*  tBE,
                                                 int                   tNumAtoms,
                                                 int                   tNumBonds,
                                                 PairMatchTablesDevice tables,
                                                 IncrementalTestOut*   out) {
  __shared__ QueuedT16 child;
  if (threadIdx.x == 0) {
    mcs::fmcs::seedClearWithinThread(child.seed);
    // Parent: bond (0,1) mapped 0->0, 1->1, bond 0->target bond 0.
    int qA[] = {0, 1};
    int tA[] = {0, 1};
    int qB[] = {0};
    int tB[] = {0};
    buildParentMatch(child.match, qA, tA, 2, qB, tB, 1);
    // Child seed has bonds {0, 1} and atoms {0, 1, 2}.  Bond 1 is the
    // new atom-adding bond (qU=1 already mapped, qV=2 unmapped).
    mcs::fmcs::seedAddBondWithinThread(child.seed, 0);
    mcs::fmcs::seedAddBondWithinThread(child.seed, 1);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 0);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 1);
    mcs::fmcs::seedBeginGrowStepWithinThread(child.seed);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 2);
  }
  __syncthreads();

  auto        block = cooperative_groups::this_thread_block();
  auto        warp  = cooperative_groups::tiled_partition<32>(block);
  TestCsrView qView{qBE, qNumAtoms, qNumBonds};
  TestCsrView tView{tBE, tNumAtoms, tNumBonds};
  bool        ok = mcs::fmcs::matchIncrementalFastCooperative(warp, child.seed, qView, tView, tables, child.match);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok    = ok;
    out->child = child;
  }
}

__global__ void matchIncrementalRingClosingDriver(const std::uint32_t*  qBE,
                                                  int                   qNumAtoms,
                                                  int                   qNumBonds,
                                                  const std::uint32_t*  tBE,
                                                  int                   tNumAtoms,
                                                  int                   tNumBonds,
                                                  PairMatchTablesDevice tables,
                                                  IncrementalTestOut*   out) {
  __shared__ QueuedT16 child;
  if (threadIdx.x == 0) {
    mcs::fmcs::seedClearWithinThread(child.seed);
    // 4-atom square query: bonds (0,1), (1,2), (2,3), (0,3).  Parent
    // has all 4 atoms mapped via 3 path bonds; child adds the
    // ring-closing 4th bond (0,3).
    int qA[] = {0, 1, 2, 3};
    int tA[] = {0, 1, 2, 3};
    int qB[] = {0, 1, 2};
    int tB[] = {0, 1, 2};
    buildParentMatch(child.match, qA, tA, 4, qB, tB, 3);
    for (int b : {0, 1, 2, 3})
      mcs::fmcs::seedAddBondWithinThread(child.seed, b);
    for (int a : {0, 1, 2, 3})
      mcs::fmcs::seedAddAtomWithinThread(child.seed, a);
  }
  __syncthreads();

  auto        block = cooperative_groups::this_thread_block();
  auto        warp  = cooperative_groups::tiled_partition<32>(block);
  TestCsrView qView{qBE, qNumAtoms, qNumBonds};
  TestCsrView tView{tBE, tNumAtoms, tNumBonds};
  bool        ok = mcs::fmcs::matchIncrementalFastCooperative(warp, child.seed, qView, tView, tables, child.match);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok    = ok;
    out->child = child;
  }
}

__global__ void matchIncrementalVisitedConflictDriver(const std::uint32_t*  qBE,
                                                      int                   qNumAtoms,
                                                      int                   qNumBonds,
                                                      const std::uint32_t*  tBE,
                                                      int                   tNumAtoms,
                                                      int                   tNumBonds,
                                                      PairMatchTablesDevice tables,
                                                      IncrementalTestOut*   out) {
  __shared__ QueuedT16 child;
  if (threadIdx.x == 0) {
    mcs::fmcs::seedClearWithinThread(child.seed);
    // Query: atoms 0,1,2; bonds (0,1), (0,2).  Target: atoms 0,1; one
    // bond (0,1).  Parent has bond (0,1) mapped (qU=0->t=0, qV=1->t=1).
    // Now child wants atom-adding bond (0,2), but the only target bond
    // out of t=0 is (0,1) and t=1 is already visited -> must fail.
    int qA[] = {0, 1};
    int tA[] = {0, 1};
    int qB[] = {0};
    int tB[] = {0};
    buildParentMatch(child.match, qA, tA, 2, qB, tB, 1);
    mcs::fmcs::seedAddBondWithinThread(child.seed, 0);
    mcs::fmcs::seedAddBondWithinThread(child.seed, 1);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 0);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 1);
    mcs::fmcs::seedBeginGrowStepWithinThread(child.seed);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 2);
  }
  __syncthreads();

  auto        block = cooperative_groups::this_thread_block();
  auto        warp  = cooperative_groups::tiled_partition<32>(block);
  TestCsrView qView{qBE, qNumAtoms, qNumBonds};
  TestCsrView tView{tBE, tNumAtoms, tNumBonds};
  bool        ok = mcs::fmcs::matchIncrementalFastCooperative(warp, child.seed, qView, tView, tables, child.match);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok    = ok;
    out->child = child;
  }
}

__global__ void matchIncrementalTwoBondChainDriver(const std::uint32_t*  qBE,
                                                   int                   qNumAtoms,
                                                   int                   qNumBonds,
                                                   const std::uint32_t*  tBE,
                                                   int                   tNumAtoms,
                                                   int                   tNumBonds,
                                                   PairMatchTablesDevice tables,
                                                   IncrementalTestOut*   out) {
  __shared__ QueuedT16 child;
  if (threadIdx.x == 0) {
    mcs::fmcs::seedClearWithinThread(child.seed);
    // 4-atom path 0-1-2-3.  Parent has only bond (0,1) mapped.  Child
    // adds bonds (1,2) and (2,3) in one matchIncrementalFast call.
    int qA[] = {0, 1};
    int tA[] = {0, 1};
    int qB[] = {0};
    int tB[] = {0};
    buildParentMatch(child.match, qA, tA, 2, qB, tB, 1);
    mcs::fmcs::seedAddBondWithinThread(child.seed, 0);
    mcs::fmcs::seedAddBondWithinThread(child.seed, 1);
    mcs::fmcs::seedAddBondWithinThread(child.seed, 2);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 0);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 1);
    mcs::fmcs::seedBeginGrowStepWithinThread(child.seed);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 2);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 3);
  }
  __syncthreads();

  auto        block = cooperative_groups::this_thread_block();
  auto        warp  = cooperative_groups::tiled_partition<32>(block);
  TestCsrView qView{qBE, qNumAtoms, qNumBonds};
  TestCsrView tView{tBE, tNumAtoms, tNumBonds};
  bool        ok = mcs::fmcs::matchIncrementalFastCooperative(warp, child.seed, qView, tView, tables, child.match);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok    = ok;
    out->child = child;
  }
}

}  // namespace mcs_fmcs_incremental_test

TEST(FMCSUnit, MatchIncrementalFastAtomAdding) {
  using mcs_fmcs_incremental_test::IncrementalTestOut;
  using mcs_fmcs_incremental_test::matchIncrementalAtomAddingDriver;

  // Query/target are both a 3-atom path 0-1-2 with bonds (0,1), (1,2).
  auto               qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2}
  });
  ManagedMatchTables tables;
  tables.allocate(3, 3, 2, 2);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<IncrementalTestOut> d_out;
  matchIncrementalAtomAddingDriver<<<1, 32>>>(qBE.data(), 3, 2, tBE.data(), 3, 2, tables.device(), d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  IncrementalTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_EQ(out.child.match.targetBondIdx[1], 1u);
  EXPECT_EQ(out.child.match.targetAtomIdx[2], 2u);
  EXPECT_EQ(out.child.match.matchedBondSize, 2);
  EXPECT_EQ(out.child.match.matchedAtomSize, 3);
}

TEST(FMCSUnit, MatchIncrementalFastRingClosing) {
  using mcs_fmcs_incremental_test::IncrementalTestOut;
  using mcs_fmcs_incremental_test::matchIncrementalRingClosingDriver;

  // 4-atom square with one diagonal-free closure.  Bonds: (0,1) (1,2)
  // (2,3) (0,3).
  auto               qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2},
    {2, 3},
    {0, 3}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2},
    {2, 3},
    {0, 3}
  });
  ManagedMatchTables tables;
  tables.allocate(4, 4, 4, 4);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<IncrementalTestOut> d_out;
  matchIncrementalRingClosingDriver<<<1, 32>>>(qBE.data(), 4, 4, tBE.data(), 4, 4, tables.device(), d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  IncrementalTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_EQ(out.child.match.targetBondIdx[3], 3u);
  EXPECT_EQ(out.child.match.matchedBondSize, 4);
  // No new atoms added by the ring-closing bond.
  EXPECT_EQ(out.child.match.matchedAtomSize, 4);
}

TEST(FMCSUnit, MatchIncrementalFastVisitedConflictFails) {
  using mcs_fmcs_incremental_test::IncrementalTestOut;
  using mcs_fmcs_incremental_test::matchIncrementalVisitedConflictDriver;

  // Query: 3 atoms / 2 bonds.  Target: 2 atoms / 1 bond.  Parent has
  // bond (0,1) mapped; trying to extend with bond (0,2) forces atom 2
  // onto target atom 1, which is already visited.
  auto               qBE = makeBondEndpointsDevice({
    {0, 1},
    {0, 2}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 1}
  });
  ManagedMatchTables tables;
  tables.allocate(3, 2, 2, 1);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<IncrementalTestOut> d_out;
  matchIncrementalVisitedConflictDriver<<<1, 32>>>(qBE.data(), 3, 2, tBE.data(), 2, 1, tables.device(), d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  IncrementalTestOut out{};
  d_out.get(out);

  EXPECT_FALSE(out.ok);
}

TEST(FMCSUnit, MatchIncrementalFastTwoBondChain) {
  using mcs_fmcs_incremental_test::IncrementalTestOut;
  using mcs_fmcs_incremental_test::matchIncrementalTwoBondChainDriver;

  // Both sides are the 4-atom path 0-1-2-3 with bonds 0=(0,1), 1=(1,2),
  // 2=(2,3).
  auto               qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2},
    {2, 3}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2},
    {2, 3}
  });
  ManagedMatchTables tables;
  tables.allocate(4, 4, 3, 3);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<IncrementalTestOut> d_out;
  matchIncrementalTwoBondChainDriver<<<1, 32>>>(qBE.data(), 4, 3, tBE.data(), 4, 3, tables.device(), d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  IncrementalTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_EQ(out.child.match.targetBondIdx[1], 1u);
  EXPECT_EQ(out.child.match.targetBondIdx[2], 2u);
  EXPECT_EQ(out.child.match.targetAtomIdx[2], 2u);
  EXPECT_EQ(out.child.match.targetAtomIdx[3], 3u);
  EXPECT_EQ(out.child.match.matchedBondSize, 3);
  EXPECT_EQ(out.child.match.matchedAtomSize, 4);
}

// ---- full seed substructure fallback ----

namespace mcs_fmcs_substructure_test {

using QueuedT16 = mcs::fmcs::QueuedSeed<16, 16, 16, 16>;

struct SubstructureTestOut {
  bool      ok;
  bool      overflowed;
  QueuedT16 child;
};

constexpr int kTestSubstructurePartialCapacity = 64;

__device__ __forceinline__ void addMaskSeed(QueuedT16& child, std::uint32_t atomMask, std::uint32_t bondMask) {
  mcs::fmcs::seedClearWithinThread(child.seed);
  mcs::fmcs::matchResultClearWithinThread(child.match);
  for (int a = 0; a < 16; ++a) {
    if ((atomMask >> a) & 1u) {
      mcs::fmcs::seedAddAtomWithinThread(child.seed, a);
    }
  }
  for (int b = 0; b < 16; ++b) {
    if ((bondMask >> b) & 1u) {
      mcs::fmcs::seedAddBondWithinThread(child.seed, b);
    }
  }
}

__global__ void matchSubstructureMaskDriver(const std::uint32_t*  qBE,
                                            int                   qNumAtoms,
                                            int                   qNumBonds,
                                            const std::uint32_t*  tBE,
                                            int                   tNumAtoms,
                                            int                   tNumBonds,
                                            PairMatchTablesDevice tables,
                                            std::uint32_t         atomMask,
                                            std::uint32_t         bondMask,
                                            std::uint8_t*         partialStorage,
                                            int                   partialCapacity,
                                            SubstructureTestOut*  out) {
  __shared__ QueuedT16 child;
  __shared__ mcs::fmcs::FmcsSubstructureScratch<16, 16> scratch;
  if (threadIdx.x == 0) {
    addMaskSeed(child, atomMask, bondMask);
  }
  __syncthreads();

  auto        block = cooperative_groups::this_thread_block();
  auto        warp  = cooperative_groups::tiled_partition<32>(block);
  TestCsrView qView{qBE, qNumAtoms, qNumBonds};
  TestCsrView tView{tBE, tNumAtoms, tNumBonds};
  bool        overflowed = false;
  bool        ok         = mcs::fmcs::matchSeedSubstructureCooperative(warp,
                                                        child.seed,
                                                        qView,
                                                        tView,
                                                        tables,
                                                        child.match,
                                                        scratch,
                                                        partialStorage,
                                                        partialCapacity,
                                                        &overflowed);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok         = ok;
    out->overflowed = overflowed;
    out->child      = child;
  }
}

__global__ void matchFallbackBadParentDriver(const std::uint32_t*  qBE,
                                             int                   qNumAtoms,
                                             int                   qNumBonds,
                                             const std::uint32_t*  tBE,
                                             int                   tNumAtoms,
                                             int                   tNumBonds,
                                             PairMatchTablesDevice tables,
                                             std::uint8_t*         partialStorage,
                                             int                   partialCapacity,
                                             SubstructureTestOut*  out) {
  __shared__ QueuedT16 child;
  __shared__ mcs::fmcs::FmcsSubstructureScratch<16, 16> scratch;
  __shared__ int                                        scratchLock;
  if (threadIdx.x == 0) {
    // Query seed is the 4-edge path inside the triangle-with-leaves
    // repro: query bonds 1,2,3,4 and all five atoms.  The stored parent
    // match maps q bond 1 = (0,2) onto target bond 0 = (0,3), which is
    // locally valid but blocks q bond 2 = (0,4) in the fast extender.
    addMaskSeed(child, /*atomMask=*/0x1Fu, /*bondMask=*/0x1Eu);
    mcs::fmcs::matchResultClearWithinThread(child.match);
    using MatchT                 = decltype(child.match);
    child.match.targetAtomIdx[0] = 0;
    child.match.targetAtomIdx[2] = 3;
    child.match.visitedTargetAtoms[0 / MatchT::kTargetAtomBitsPerWord] |= typename MatchT::target_atom_word{1}
                                                                       << (0 % MatchT::kTargetAtomBitsPerWord);
    child.match.visitedTargetAtoms[3 / MatchT::kTargetAtomBitsPerWord] |= typename MatchT::target_atom_word{1}
                                                                       << (3 % MatchT::kTargetAtomBitsPerWord);
    child.match.targetBondIdx[1] = 0;
    child.match.visitedTargetBonds[0 / MatchT::kTargetBondBitsPerWord] |= typename MatchT::target_bond_word{1}
                                                                       << (0 % MatchT::kTargetBondBitsPerWord);
    child.match.matchedAtomSize = 2;
    child.match.matchedBondSize = 1;
    child.match.empty           = false;
    scratchLock                 = 0;
  }
  __syncthreads();

  auto        block = cooperative_groups::this_thread_block();
  auto        warp  = cooperative_groups::tiled_partition<32>(block);
  TestCsrView qView{qBE, qNumAtoms, qNumBonds};
  TestCsrView tView{tBE, tNumAtoms, tNumBonds};
  bool        overflowed = false;
  bool        ok         = mcs::fmcs::matchSeedWithSubstructureFallbackCooperative(warp,
                                                                    child.seed,
                                                                    qView,
                                                                    tView,
                                                                    tables,
                                                                    child.match,
                                                                    scratch,
                                                                    &scratchLock,
                                                                    partialStorage,
                                                                    partialCapacity,
                                                                    &overflowed);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok         = ok;
    out->overflowed = overflowed;
    out->child      = child;
  }
}

}  // namespace mcs_fmcs_substructure_test

TEST(FMCSUnit, MatchSeedSubstructurePath) {
  using namespace mcs_fmcs_substructure_test;

  auto               qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2},
    {2, 3}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2},
    {2, 3},
    {3, 4}
  });
  ManagedMatchTables tables;
  tables.allocate(4, 5, 3, 4);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<SubstructureTestOut> d_out;
  AsyncDeviceVector<std::uint8_t>     partials(2 * kTestSubstructurePartialCapacity * 16);
  matchSubstructureMaskDriver<<<1, 32>>>(qBE.data(),
                                         4,
                                         3,
                                         tBE.data(),
                                         5,
                                         4,
                                         tables.device(),
                                         /*atomMask=*/0xFu,
                                         /*bondMask=*/0x7u,
                                         partials.data(),
                                         kTestSubstructurePartialCapacity,
                                         d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SubstructureTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_FALSE(out.overflowed);
  EXPECT_EQ(out.child.match.matchedAtomSize, 4);
  EXPECT_EQ(out.child.match.matchedBondSize, 3);
  for (int q = 0; q < 4; ++q) {
    EXPECT_NE(out.child.match.targetAtomIdx[q], mcs::fmcs::kUnmappedTargetIdx);
  }
  for (int q = 0; q < 3; ++q) {
    EXPECT_NE(out.child.match.targetBondIdx[q], mcs::fmcs::kUnmappedTargetIdx);
  }
}

TEST(FMCSUnit, MatchSeedSubstructureRejectsNoMatch) {
  using namespace mcs_fmcs_substructure_test;

  auto               qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2},
    {0, 2}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2}
  });
  ManagedMatchTables tables;
  tables.allocate(3, 3, 3, 2);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<SubstructureTestOut> d_out;
  AsyncDeviceVector<std::uint8_t>     partials(2 * kTestSubstructurePartialCapacity * 16);
  matchSubstructureMaskDriver<<<1, 32>>>(qBE.data(),
                                         3,
                                         3,
                                         tBE.data(),
                                         3,
                                         2,
                                         tables.device(),
                                         /*atomMask=*/0x7u,
                                         /*bondMask=*/0x7u,
                                         partials.data(),
                                         kTestSubstructurePartialCapacity,
                                         d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SubstructureTestOut out{};
  d_out.get(out);

  EXPECT_FALSE(out.ok);
  EXPECT_FALSE(out.overflowed);
  EXPECT_TRUE(out.child.match.empty);
}

TEST(FMCSUnit, MatchSeedSubstructureRespectsAtomTable) {
  using namespace mcs_fmcs_substructure_test;

  auto               qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2}
  });
  ManagedMatchTables tables;
  tables.allocate(3, 3, 2, 2);
  tables.setAtomBit(0, 0);
  tables.setAtomBit(1, 1);
  tables.setAtomBit(2, 2);
  tables.setAllBondBits();

  AsyncDevicePtr<SubstructureTestOut> d_out;
  AsyncDeviceVector<std::uint8_t>     partials(2 * kTestSubstructurePartialCapacity * 16);
  matchSubstructureMaskDriver<<<1, 32>>>(qBE.data(),
                                         3,
                                         2,
                                         tBE.data(),
                                         3,
                                         2,
                                         tables.device(),
                                         /*atomMask=*/0x7u,
                                         /*bondMask=*/0x3u,
                                         partials.data(),
                                         kTestSubstructurePartialCapacity,
                                         d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SubstructureTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_FALSE(out.overflowed);
  EXPECT_EQ(out.child.match.targetAtomIdx[0], 0u);
  EXPECT_EQ(out.child.match.targetAtomIdx[1], 1u);
  EXPECT_EQ(out.child.match.targetAtomIdx[2], 2u);
  EXPECT_EQ(out.child.match.matchedAtomSize, 3);
  EXPECT_EQ(out.child.match.matchedBondSize, 2);
}

TEST(FMCSUnit, MatchSeedSubstructureRespectsBondTable) {
  using namespace mcs_fmcs_substructure_test;

  auto               qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2},
    {0, 2}
  });
  ManagedMatchTables tables;
  tables.allocate(3, 3, 2, 3);
  tables.setAllAtomBits();
  tables.setBondBit(0, 0);
  tables.setBondBit(1, 1);

  AsyncDevicePtr<SubstructureTestOut> d_out;
  AsyncDeviceVector<std::uint8_t>     partials(2 * kTestSubstructurePartialCapacity * 16);
  matchSubstructureMaskDriver<<<1, 32>>>(qBE.data(),
                                         3,
                                         2,
                                         tBE.data(),
                                         3,
                                         3,
                                         tables.device(),
                                         /*atomMask=*/0x7u,
                                         /*bondMask=*/0x3u,
                                         partials.data(),
                                         kTestSubstructurePartialCapacity,
                                         d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SubstructureTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_FALSE(out.overflowed);
  EXPECT_EQ(out.child.match.targetBondIdx[0], 0u);
  EXPECT_EQ(out.child.match.targetBondIdx[1], 1u);
  EXPECT_EQ(out.child.match.matchedAtomSize, 3);
  EXPECT_EQ(out.child.match.matchedBondSize, 2);
}

TEST(FMCSUnit, MatchSeedSubstructureFindsPathInsideTriangleWithLeaves) {
  using namespace mcs_fmcs_substructure_test;

  auto               qBE = makeBondEndpointsDevice({
    {0, 1},
    {0, 2},
    {0, 4},
    {1, 2},
    {1, 3}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 3},
    {1, 2},
    {1, 4},
    {2, 3}
  });
  ManagedMatchTables tables;
  tables.allocate(5, 5, 5, 4);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<SubstructureTestOut> d_out;
  AsyncDeviceVector<std::uint8_t>     partials(2 * kTestSubstructurePartialCapacity * 16);
  matchSubstructureMaskDriver<<<1, 32>>>(qBE.data(),
                                         5,
                                         5,
                                         tBE.data(),
                                         5,
                                         4,
                                         tables.device(),
                                         /*atomMask=*/0x1Fu,
                                         /*bondMask=*/0x1Eu,
                                         partials.data(),
                                         kTestSubstructurePartialCapacity,
                                         d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SubstructureTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_FALSE(out.overflowed);
  EXPECT_EQ(out.child.match.matchedAtomSize, 5);
  EXPECT_EQ(out.child.match.matchedBondSize, 4);
  EXPECT_EQ(out.child.match.targetBondIdx[0], mcs::fmcs::kUnmappedTargetIdx);
  for (int q : {1, 2, 3, 4}) {
    EXPECT_NE(out.child.match.targetBondIdx[q], mcs::fmcs::kUnmappedTargetIdx);
  }
}

TEST(FMCSUnit, MatchSeedFallbackRebuildsAfterGreedyFailure) {
  using namespace mcs_fmcs_substructure_test;

  auto               qBE = makeBondEndpointsDevice({
    {0, 1},
    {0, 2},
    {0, 4},
    {1, 2},
    {1, 3}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 3},
    {1, 2},
    {1, 4},
    {2, 3}
  });
  ManagedMatchTables tables;
  tables.allocate(5, 5, 5, 4);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<SubstructureTestOut> d_out;
  AsyncDeviceVector<std::uint8_t>     partials(2 * kTestSubstructurePartialCapacity * 16);
  matchFallbackBadParentDriver<<<1, 32>>>(qBE.data(),
                                          5,
                                          5,
                                          tBE.data(),
                                          5,
                                          4,
                                          tables.device(),
                                          partials.data(),
                                          kTestSubstructurePartialCapacity,
                                          d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SubstructureTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_FALSE(out.overflowed);
  EXPECT_EQ(out.child.match.matchedAtomSize, 5);
  EXPECT_EQ(out.child.match.matchedBondSize, 4);
  for (int q = 0; q < 5; ++q) {
    EXPECT_NE(out.child.match.targetAtomIdx[q], mcs::fmcs::kUnmappedTargetIdx);
  }
  for (int q : {1, 2, 3, 4}) {
    EXPECT_NE(out.child.match.targetBondIdx[q], mcs::fmcs::kUnmappedTargetIdx);
  }
}

// ---------------------------------------------------------------------------
// fillNewBondsCooperative / pruneIndividualBondsCooperative
// ---------------------------------------------------------------------------

namespace mcs_fmcs_grow_test {

using mcs::fmcs::NewBond;
using SeedT   = mcs::fmcs::Seed<16, 16>;
using QueuedT = mcs::fmcs::QueuedSeed<16, 16, 16, 16>;

constexpr int kMaxNewBonds = 8;

struct FillNewBondsOut {
  NewBond bonds[kMaxNewBonds];
  int     count;
  bool    ok;
};

// Driver: caller-provided seed-setup function fills the seed before
// calling fillNewBondsCooperative.  Templated on the setup so each
// test can hand-construct its own scenario.
template <class SeedSetup>
__device__ __forceinline__ void fillNewBondsRun(SeedSetup&&          setup,
                                                const std::uint32_t* qBondEndpoints,
                                                int                  qNumAtoms,
                                                int                  qNumBonds,
                                                int                  maxNewBonds,
                                                FillNewBondsOut*     out) {
  __shared__ SeedT   seed;
  __shared__ NewBond bonds[kMaxNewBonds];
  __shared__ int     count;

  if (threadIdx.x == 0) {
    mcs::fmcs::seedClearWithinThread(seed);
    setup(seed);
  }
  __syncthreads();

  auto        block = cooperative_groups::this_thread_block();
  auto        warp  = cooperative_groups::tiled_partition<32>(block);
  TestCsrView qView{qBondEndpoints, qNumAtoms, qNumBonds};
  bool        ok = mcs::fmcs::fillNewBondsCooperative(warp, seed, qView, bonds, &count, maxNewBonds);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok    = ok;
    out->count = count;
    for (int i = 0; i < kMaxNewBonds; ++i)
      out->bonds[i] = bonds[i];
  }
}

// Test 1: 3-atom path query (atoms 0-1-2, bonds (0,1), (1,2)).  Seed
// holds just atom 1, marked last-added.  Both bonds touch atom 1;
// each should become an atom-adding NewBond.
__global__ void fillNewBondsAtomAddingDriver(const std::uint32_t* qBE,
                                             int                  qNumAtoms,
                                             int                  qNumBonds,
                                             FillNewBondsOut*     out) {
  fillNewBondsRun([] __device__(SeedT & seed) { mcs::fmcs::seedAddAtomWithinThread(seed, 1); },
                  qBE,
                  qNumAtoms,
                  qNumBonds,
                  kMaxNewBonds,
                  out);
}

// Test 2: same query, seed has bond 0 in excludedBonds.  Only bond 1
// should appear in the output.
__global__ void fillNewBondsExcludedSkippedDriver(const std::uint32_t* qBE,
                                                  int                  qNumAtoms,
                                                  int                  qNumBonds,
                                                  FillNewBondsOut*     out) {
  fillNewBondsRun(
    [] __device__(SeedT & seed) {
      mcs::fmcs::seedAddAtomWithinThread(seed, 1);
      // Mark bond 0 as excluded but DO NOT add it to seed.bonds.
      // The exclusion alone is what fillNewBonds checks.
      using BondWord     = SeedT::bond_word_type;
      constexpr int kBPW = SeedT::kBondBitsPerWord;
      seed.excludedBonds[0 / kBPW] |= static_cast<BondWord>(1) << (0 % kBPW);
    },
    qBE,
    qNumAtoms,
    qNumBonds,
    kMaxNewBonds,
    out);
}

// Test 3: 3-atom triangle query (atoms 0,1,2; bonds (0,1), (1,2), (0,2)).
// Seed holds atoms 0 and 1 (both last-added) plus bond (0,1).  Bond 2
// = (0,2) is atom-adding (atom 2 not in seed).  More importantly, this
// test verifies the seed.atoms vs seed.lastAddedAtoms split: if we
// only mark atom 1 as "last added" (atom 0 stays in seed.atoms but
// NOT in lastAddedAtoms), bond (0,2) should NOT be reported because
// neither endpoint is "newly added".  Bond (1,2) SHOULD be reported.
__global__ void fillNewBondsLastAddedFilteringDriver(const std::uint32_t* qBE,
                                                     int                  qNumAtoms,
                                                     int                  qNumBonds,
                                                     FillNewBondsOut*     out) {
  fillNewBondsRun(
    [] __device__(SeedT & seed) {
      // Atom 0 is "old" (in seed but not last-added).
      seed.atoms[0] |= 1u << 0;
      seed.numAtoms = 1;
      // Begin a grow step boundary.
      mcs::fmcs::seedBeginGrowStepWithinThread(seed);
      // Atom 1 is "newly added".
      mcs::fmcs::seedAddAtomWithinThread(seed, 1);
      // Bond 0 already mapped (in seed and excluded).
      mcs::fmcs::seedAddBondWithinThread(seed, 0);
    },
    qBE,
    qNumAtoms,
    qNumBonds,
    kMaxNewBonds,
    out);
}

// Test 4: ring-closing.  Seed has atoms 0, 1, 2 all marked last-added,
// bond 0 = (0,1) and bond 1 = (1,2) already in excludedBonds.  Bond 2
// = (0,2) closes the ring: both endpoints in seed.
__global__ void fillNewBondsRingClosingDriver(const std::uint32_t* qBE,
                                              int                  qNumAtoms,
                                              int                  qNumBonds,
                                              FillNewBondsOut*     out) {
  fillNewBondsRun(
    [] __device__(SeedT & seed) {
      mcs::fmcs::seedAddAtomWithinThread(seed, 0);
      mcs::fmcs::seedAddAtomWithinThread(seed, 1);
      mcs::fmcs::seedAddAtomWithinThread(seed, 2);
      mcs::fmcs::seedAddBondWithinThread(seed, 0);
      mcs::fmcs::seedAddBondWithinThread(seed, 1);
      // Bond 2 = (0,2) is the only candidate; it's ring-closing
      // since both 0 and 2 are in seed.atoms.
    },
    qBE,
    qNumAtoms,
    qNumBonds,
    kMaxNewBonds,
    out);
}

// Test 5: overflow.  4-atom path with seed = {atom 0 newly-added}.
// Query bonds 0..3 all touch atom 0 (star graph: bonds (0,1) (0,2)
// (0,3) (0,4)).  maxNewBonds = 2 -> first 2 win the race, function
// returns false, count clamped at 2.
__global__ void fillNewBondsOverflowDriver(const std::uint32_t* qBE,
                                           int                  qNumAtoms,
                                           int                  qNumBonds,
                                           FillNewBondsOut*     out) {
  fillNewBondsRun([] __device__(SeedT & seed) { mcs::fmcs::seedAddAtomWithinThread(seed, 0); },
                  qBE,
                  qNumAtoms,
                  qNumBonds,
                  /*maxNewBonds=*/2,
                  out);
}

}  // namespace mcs_fmcs_grow_test

TEST(FMCSUnit, FillNewBondsAtomAddingFromBoundary) {
  using namespace mcs_fmcs_grow_test;
  // Path 0-1-2: bonds (0,1), (1,2).
  auto                            qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2}
  });
  AsyncDevicePtr<FillNewBondsOut> d_out;

  fillNewBondsAtomAddingDriver<<<1, 32>>>(qBE.data(), 3, 2, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  FillNewBondsOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_EQ(out.count, 2);
  // Both bonds appeared; race-order is non-deterministic so collect
  // by bondIdx.
  std::set<int> seenBonds;
  for (int i = 0; i < out.count; ++i) {
    seenBonds.insert(out.bonds[i].bondIdx);
    EXPECT_EQ(out.bonds[i].endAtomSeedIdx, NewBond::kNotInSeed);
    EXPECT_TRUE(out.bonds[i].alive);
  }
  EXPECT_TRUE(seenBonds.count(0));
  EXPECT_TRUE(seenBonds.count(1));
}

TEST(FMCSUnit, FillNewBondsExcludedSkipped) {
  using namespace mcs_fmcs_grow_test;
  auto                            qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2}
  });
  AsyncDevicePtr<FillNewBondsOut> d_out;

  fillNewBondsExcludedSkippedDriver<<<1, 32>>>(qBE.data(), 3, 2, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  FillNewBondsOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_EQ(out.count, 1);
  EXPECT_EQ(out.bonds[0].bondIdx, 1);  // bond 0 excluded; only bond 1 left
}

TEST(FMCSUnit, FillNewBondsLastAddedFiltering) {
  using namespace mcs_fmcs_grow_test;
  // Triangle: bonds 0=(0,1), 1=(1,2), 2=(0,2).
  auto                            qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2},
    {0, 2}
  });
  AsyncDevicePtr<FillNewBondsOut> d_out;

  fillNewBondsLastAddedFilteringDriver<<<1, 32>>>(qBE.data(), 3, 3, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  FillNewBondsOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  // Bond 0 is excluded.  Bond 1 = (1,2) touches newly-added atom 1.
  // Bond 2 = (0,2) touches atom 0 (NOT newly-added) and atom 2 (also
  // not in seed) -> neither endpoint newly-added, must be skipped.
  EXPECT_EQ(out.count, 1);
  EXPECT_EQ(out.bonds[0].bondIdx, 1);
  EXPECT_EQ(out.bonds[0].endAtomSeedIdx, NewBond::kNotInSeed);
  EXPECT_EQ(out.bonds[0].newAtomIdx, 2);  // atom 2 is the unmapped end
}

TEST(FMCSUnit, FillNewBondsRingClosing) {
  using namespace mcs_fmcs_grow_test;
  // Triangle: bonds 0=(0,1), 1=(1,2), 2=(0,2).
  auto                            qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2},
    {0, 2}
  });
  AsyncDevicePtr<FillNewBondsOut> d_out;

  fillNewBondsRingClosingDriver<<<1, 32>>>(qBE.data(), 3, 3, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  FillNewBondsOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  // Only bond 2 = (0,2) is candidate (others are excluded).  Both
  // endpoints in seed -> ring-closing.
  EXPECT_EQ(out.count, 1);
  EXPECT_EQ(out.bonds[0].bondIdx, 2);
  EXPECT_NE(out.bonds[0].endAtomSeedIdx, NewBond::kNotInSeed);
}

TEST(FMCSUnit, FillNewBondsOverflowReturnsFalse) {
  using namespace mcs_fmcs_grow_test;
  // Star graph: 4 bonds from atom 0 -> 1, 2, 3, 4.  numAtoms=5.
  auto                            qBE = makeBondEndpointsDevice({
    {0, 1},
    {0, 2},
    {0, 3},
    {0, 4}
  });
  AsyncDevicePtr<FillNewBondsOut> d_out;

  fillNewBondsOverflowDriver<<<1, 32>>>(qBE.data(), 5, 4, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  FillNewBondsOut out{};
  d_out.get(out);

  EXPECT_FALSE(out.ok);
  // count is clamped to maxNewBonds (=2).
  EXPECT_EQ(out.count, 2);
}

// ---- pruneIndividualBondsCooperative ----

namespace mcs_fmcs_prune_test {

using mcs::fmcs::NewBond;
using mcs_fmcs_grow_test::QueuedT;
using mcs_fmcs_grow_test::SeedT;

constexpr int kMaxNewBondsPrune = 8;

struct PruneOut {
  // Identity of children that survived matchFn and reached childSink.
  // We record the child seed's bondIdx-of-the-newly-added bond (the
  // first set bit beyond the parent's bond bitset).  That uniquely
  // identifies which NewBond the child was built from.
  int survivorBondIdx[kMaxNewBondsPrune];
  int numSurvivors;
  int numMatchAttempts;
};

__global__ void pruneStage1Driver(const NewBond* hostBonds, int nBonds, PruneOut* out) {
  __shared__ std::uint64_t parentStorage[(sizeof(QueuedT) + sizeof(std::uint64_t) - 1) / sizeof(std::uint64_t)];
  __shared__ std::uint64_t workspaceStorage[(sizeof(QueuedT) + sizeof(std::uint64_t) - 1) / sizeof(std::uint64_t)];
  __shared__ NewBond       bonds[kMaxNewBondsPrune];
  __shared__ int           matchAttempts;
  __shared__ int           survivorIdx;
  QueuedT&                 parent    = *reinterpret_cast<QueuedT*>(parentStorage);
  QueuedT&                 workspace = *reinterpret_cast<QueuedT*>(workspaceStorage);

  if (threadIdx.x == 0) {
    mcs::fmcs::seedClearWithinThread(parent.seed);
    mcs::fmcs::matchResultClearWithinThread(parent.match);
    // Parent has bond 0 already in its bitset (so pruneIndividualBonds
    // appears as "extending past bond 0" for tracking).
    mcs::fmcs::seedAddAtomWithinThread(parent.seed, 0);
    mcs::fmcs::seedAddAtomWithinThread(parent.seed, 1);
    mcs::fmcs::seedAddBondWithinThread(parent.seed, 0);
    for (int i = 0; i < nBonds; ++i)
      bonds[i] = hostBonds[i];
    matchAttempts = 0;
    survivorIdx   = 0;
    for (int i = 0; i < kMaxNewBondsPrune; ++i) {
      out->survivorBondIdx[i] = -1;
    }
  }
  __syncthreads();

  auto block = cooperative_groups::this_thread_block();
  auto warp  = cooperative_groups::tiled_partition<32>(block);

  // Stub matchFn: returns true iff the new bond's bondIdx is even.
  // Counts attempts so the test can verify every bond was tried.
  auto matchFn = [&] __device__(QueuedT & child) -> bool {
    if (warp.thread_rank() == 0)
      ++matchAttempts;
    warp.sync();
    // Find which bondIdx the workspace's seed has beyond the parent's.
    // Parent has only bond 0 set; child has one more.
    int childBond = -1;
    for (int b = 0; b < 16; ++b) {
      using BondWord            = SeedT::bond_word_type;
      const BondWord childWord  = child.seed.bonds[0];
      const BondWord parentWord = parent.seed.bonds[0];
      const BondWord newBits    = childWord & ~parentWord;
      if ((newBits >> b) & 1) {
        childBond = b;
        break;
      }
    }
    return (childBond % 2) == 0;
  };
  auto childSink = [&] __device__(QueuedT & child) {
    if (warp.thread_rank() == 0) {
      // Record child's new-bond identity.
      using BondWord           = SeedT::bond_word_type;
      const BondWord newBits   = child.seed.bonds[0] & ~parent.seed.bonds[0];
      int            childBond = -1;
      for (int b = 0; b < 16; ++b) {
        if ((newBits >> b) & 1) {
          childBond = b;
          break;
        }
      }
      out->survivorBondIdx[survivorIdx++] = childBond;
    }
    warp.sync();
  };
  mcs::fmcs::pruneIndividualBondsCooperative(warp, parent, workspace, bonds, nBonds, matchFn, childSink);

  __syncthreads();
  if (threadIdx.x == 0) {
    out->numMatchAttempts = matchAttempts;
    out->numSurvivors     = survivorIdx;
  }
}

}  // namespace mcs_fmcs_prune_test

TEST(FMCSUnit, PruneIndividualBondsStage1OnlySurvivorsSinked) {
  using namespace mcs_fmcs_prune_test;
  // 4 candidate bonds with bondIdx 1, 2, 3, 4.  Stub matchFn passes
  // even bondIdx (-> 2, 4 succeed; 1, 3 fail).  All 4 must be tried
  // (matchAttempts == 4); only 2 survivors should reach childSink.
  const std::vector<NewBond> hostBonds = {
    NewBond{1, 5, NewBond::kNotInSeed, true},
    NewBond{2, 6, NewBond::kNotInSeed, true},
    NewBond{3, 7, NewBond::kNotInSeed, true},
    NewBond{4, 8, NewBond::kNotInSeed, true},
  };
  AsyncDeviceVector<NewBond> d_bonds(hostBonds.size());
  d_bonds.copyFromHost(hostBonds);

  AsyncDevicePtr<PruneOut> d_out;
  pruneStage1Driver<<<1, 32>>>(d_bonds.data(), 4, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  PruneOut out{};
  d_out.get(out);

  EXPECT_EQ(out.numMatchAttempts, 4);  // all four bonds were tried
  EXPECT_EQ(out.numSurvivors, 2);
  std::set<int> survivors{out.survivorBondIdx[0], out.survivorBondIdx[1]};
  EXPECT_TRUE(survivors.count(2));
  EXPECT_TRUE(survivors.count(4));
  EXPECT_FALSE(survivors.count(1));
  EXPECT_FALSE(survivors.count(3));
}

// ---------------------------------------------------------------------------
// mappingHashWithinThread / DeviceMatchCache
// ---------------------------------------------------------------------------

namespace mcs_fmcs_cache_test {

using mcs::fmcs::DeviceMatchCache;
using HashSeedT  = mcs::fmcs::Seed<16, 16>;
using HashMatchT = mcs::fmcs::MatchResult<16, 16, 16, 16>;

// Hash test: build a seed with bonds added in some order, target bond
// mappings filled in, return the hash.  Used for the determinism +
// order-invariance + non-zero properties.
struct HashOut {
  std::uint64_t hash;
};

__device__ __forceinline__ void buildHashFixture(HashSeedT&  seed,
                                                 HashMatchT& match,
                                                 const int*  qBonds,
                                                 const int*  tBonds,
                                                 int         n) {
  mcs::fmcs::seedClearWithinThread(seed);
  mcs::fmcs::matchResultClearWithinThread(match);
  for (int i = 0; i < n; ++i) {
    mcs::fmcs::seedAddBondWithinThread(seed, qBonds[i]);
    match.targetBondIdx[qBonds[i]] = static_cast<std::uint8_t>(tBonds[i]);
  }
  match.empty = false;
}

__global__ void hashOrderInvariantDriver(HashOut* outA, HashOut* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeedT  seedA, seedB;
  HashMatchT matchA, matchB;
  // Same set of (q, t) mappings, different addition order.  Hash
  // walks seed.bonds in q-increasing order via __ffs, so the
  // resulting hash should be identical.
  int        qsA[] = {3, 1, 5};
  int        tsA[] = {7, 2, 9};
  int        qsB[] = {1, 5, 3};
  int        tsB[] = {2, 9, 7};
  buildHashFixture(seedA, matchA, qsA, tsA, 3);
  buildHashFixture(seedB, matchB, qsB, tsB, 3);
  outA->hash = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

__global__ void hashDistinctMappingsDiffer(HashOut* outA, HashOut* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeedT  seedA, seedB;
  HashMatchT matchA, matchB;
  // Same bonds, different target mappings -> hashes should differ.
  int        qsA[] = {0, 1, 2};
  int        tsA[] = {0, 1, 2};
  int        qsB[] = {0, 1, 2};
  int        tsB[] = {2, 1, 0};
  buildHashFixture(seedA, matchA, qsA, tsA, 3);
  buildHashFixture(seedB, matchB, qsB, tsB, 3);
  outA->hash = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

__global__ void hashEmptyIsNonZero(HashOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeedT  seed;
  HashMatchT match;
  mcs::fmcs::seedClearWithinThread(seed);
  mcs::fmcs::matchResultClearWithinThread(match);
  // Empty seed.bonds -> hash is post-processed from 0 to 1.
  out->hash = mcs::fmcs::mappingHashWithinThread(seed, match);
}

// ---- Cache insert / probe driver kernels ----

struct CacheInsertProbeOut {
  bool insertOk;
  bool probeFoundInserted;
  bool probeMissedOther;
};

__global__ void cacheInsertProbeBasicDriver(std::uint64_t* keys, int capacity, CacheInsertProbeOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  __shared__ DeviceMatchCache cache;
  cache.init(keys, capacity);
  // Cooperative zero w/ a 1-thread "group" works because we already
  // pre-zeroed `keys` from the host; nothing to do here.
  out->insertOk           = cache.insertWithinThread(0xDEADBEEFCAFEBABEULL);
  out->probeFoundInserted = cache.probeWithinThread(0xDEADBEEFCAFEBABEULL);
  out->probeMissedOther   = cache.probeWithinThread(0x0123456789ABCDEFULL);
}

struct CacheCollisionOut {
  bool insertOkA;
  bool insertOkB;
  bool foundA;
  bool foundB;
  // Auxiliary: did the second insert pick a different physical slot?
  // We test indirectly: both keys present and findable proves it.
};

__global__ void cacheLinearProbeCollisionDriver(std::uint64_t* keys, int capacity, CacheCollisionOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  __shared__ DeviceMatchCache cache;
  cache.init(keys, capacity);
  // Construct two keys that hash to the same initial slot.  The slot
  // is splitMix64(key) & (capacity - 1).  We need two keys A and B
  // such that splitMix64(A) % capacity == splitMix64(B) % capacity,
  // but A != B.  Easiest: with capacity = 8, the slot is the low 3
  // bits of splitMix64(key).  Hand-pick two keys whose hash low-3-bits
  // collide.  We brute-force it on the host below; here we just take
  // whatever two keys the host wrote into the wire format.
  // (Done in the host helper -- see CacheLinearProbeCollision test.)
  out->insertOkA = cache.insertWithinThread(0xAAAAAAAAAAAAAAAAULL);
  out->insertOkB = cache.insertWithinThread(0xBBBBBBBBBBBBBBBBULL);
  out->foundA    = cache.probeWithinThread(0xAAAAAAAAAAAAAAAAULL);
  out->foundB    = cache.probeWithinThread(0xBBBBBBBBBBBBBBBBULL);
}

struct CacheFullDropOut {
  int  successfulInserts;
  bool finalInsertOk;       // expected: false (table full)
  bool seenFirstAfterFull;  // probe still finds the first insert
};

__global__ void cacheFullDropDriver(std::uint64_t* keys, int capacity, CacheFullDropOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  __shared__ DeviceMatchCache cache;
  cache.init(keys, capacity);
  int n = 0;
  for (int i = 0; i < capacity; ++i) {
    // Use distinct non-zero keys.  Note: distinctness in the *hash*
    // input space; the cache hashes internally.
    const std::uint64_t key = static_cast<std::uint64_t>(i) + 1;
    if (cache.insertWithinThread(key))
      ++n;
  }
  out->successfulInserts  = n;
  // Capacity is hit; one more must fail.
  out->finalInsertOk      = cache.insertWithinThread(0xFFFFFFFFFFFFFFFFULL);
  // First-key probe still works.
  out->seenFirstAfterFull = cache.probeWithinThread(1ULL);
}

}  // namespace mcs_fmcs_cache_test

TEST(FMCSUnit, MappingHashOrderInvariant) {
  using namespace mcs_fmcs_cache_test;
  AsyncDevicePtr<HashOut> d_dA;
  AsyncDevicePtr<HashOut> d_dB;
  hashOrderInvariantDriver<<<1, 1>>>(d_dA.data(), d_dB.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut dA{};
  d_dA.get(dA);
  HashOut dB{};
  d_dB.get(dB);

  EXPECT_EQ(dA.hash, dB.hash) << "Same (q,t) set must hash identically.";
  EXPECT_NE(dA.hash, 0ULL);
}

TEST(FMCSUnit, MappingHashDistinctMappingsDiffer) {
  using namespace mcs_fmcs_cache_test;
  AsyncDevicePtr<HashOut> d_dA;
  AsyncDevicePtr<HashOut> d_dB;
  hashDistinctMappingsDiffer<<<1, 1>>>(d_dA.data(), d_dB.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut dA{};
  d_dA.get(dA);
  HashOut dB{};
  d_dB.get(dB);

  EXPECT_NE(dA.hash, dB.hash) << "Different target mappings should not collide on a 64-bit hash.";
}

TEST(FMCSUnit, MappingHashEmptySeedReturnsNonZero) {
  using namespace mcs_fmcs_cache_test;
  AsyncDevicePtr<HashOut> d_d;
  hashEmptyIsNonZero<<<1, 1>>>(d_d.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut d{};
  d_d.get(d);

  // 0 is reserved as the cache's empty-slot sentinel; mappingHash
  // post-processes a 0 hash to 1.
  EXPECT_EQ(d.hash, 1ULL);
}

TEST(FMCSUnit, CacheInsertProbeBasic) {
  using namespace mcs_fmcs_cache_test;
  constexpr int                    kCap = 16;
  AsyncDeviceVector<std::uint64_t> d_keys(kCap);
  d_keys.zero();
  AsyncDevicePtr<CacheInsertProbeOut> d_out;

  cacheInsertProbeBasicDriver<<<1, 1>>>(d_keys.data(), kCap, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  CacheInsertProbeOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.insertOk);
  EXPECT_TRUE(out.probeFoundInserted);
  EXPECT_FALSE(out.probeMissedOther);
}

TEST(FMCSUnit, CacheLinearProbeCollision) {
  using namespace mcs_fmcs_cache_test;
  constexpr int                    kCap = 8;
  AsyncDeviceVector<std::uint64_t> d_keys(kCap);
  d_keys.zero();
  AsyncDevicePtr<CacheCollisionOut> d_out;

  cacheLinearProbeCollisionDriver<<<1, 1>>>(d_keys.data(), kCap, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  CacheCollisionOut out{};
  d_out.get(out);

  // Both inserts succeed regardless of whether they happen to collide
  // on the initial slot -- linear probing finds a different slot
  // when the first is taken.  Both probes find their key.
  EXPECT_TRUE(out.insertOkA);
  EXPECT_TRUE(out.insertOkB);
  EXPECT_TRUE(out.foundA);
  EXPECT_TRUE(out.foundB);

  // Sanity: at least one slot in keys[] equals each inserted key.
  std::vector<std::uint64_t> keys(kCap);
  d_keys.copyToHost(keys);
  bool seenA = false, seenB = false;
  for (int i = 0; i < kCap; ++i) {
    if (keys[i] == 0xAAAAAAAAAAAAAAAAULL)
      seenA = true;
    if (keys[i] == 0xBBBBBBBBBBBBBBBBULL)
      seenB = true;
  }
  EXPECT_TRUE(seenA);
  EXPECT_TRUE(seenB);
}

TEST(FMCSUnit, CacheFullTableDropsAdditionalInsert) {
  using namespace mcs_fmcs_cache_test;
  constexpr int                    kCap = 16;  // small but power-of-two
  AsyncDeviceVector<std::uint64_t> d_keys(kCap);
  d_keys.zero();
  AsyncDevicePtr<CacheFullDropOut> d_out;

  cacheFullDropDriver<<<1, 1>>>(d_keys.data(), kCap, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  CacheFullDropOut out{};
  d_out.get(out);

  EXPECT_EQ(out.successfulInserts, kCap) << "Distinct keys should fill all slots.";
  EXPECT_FALSE(out.finalInsertOk) << "Insert into a full table must return false.";
  EXPECT_TRUE(out.seenFirstAfterFull) << "Existing entries must remain visible after a failed insert.";
}

// ---------------------------------------------------------------------------
// Step 5 gap-coverage tests: multi-word hash, idempotent insert, probe of
// empty cache, single-bond hash, extreme indices, cooperative zero.
// ---------------------------------------------------------------------------

namespace mcs_fmcs_cache_gap_test {

using mcs::fmcs::DeviceMatchCache;
using HashSeed128  = mcs::fmcs::Seed<128, 128>;
using HashMatch128 = mcs::fmcs::MatchResult<128, 128, 128, 128>;
using HashSeed16   = mcs::fmcs::Seed<16, 16>;
using HashMatch16  = mcs::fmcs::MatchResult<16, 16, 16, 16>;

struct HashOut64 {
  std::uint64_t hash;
};

// Multi-word path: tier 128 has kBondWords = 2 (two uint64 words),
// exercising the outer wordIdx loop AND __ffsll on each.  Place bonds
// in BOTH words so the iteration must cross the word boundary.
__global__ void hashMultiWordPathDriver(HashOut64* outA, HashOut64* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeed128  seedA, seedB;
  HashMatch128 matchA, matchB;
  mcs::fmcs::seedClearWithinThread(seedA);
  mcs::fmcs::seedClearWithinThread(seedB);
  mcs::fmcs::matchResultClearWithinThread(matchA);
  mcs::fmcs::matchResultClearWithinThread(matchB);
  // bonds in word 0 and word 1.  Same set, different addition order.
  for (int q : {5, 70, 33, 100}) {
    mcs::fmcs::seedAddBondWithinThread(seedA, q);
    matchA.targetBondIdx[q] = static_cast<std::uint8_t>(q);
  }
  for (int q : {100, 5, 33, 70}) {
    mcs::fmcs::seedAddBondWithinThread(seedB, q);
    matchB.targetBondIdx[q] = static_cast<std::uint8_t>(q);
  }
  matchA.empty = false;
  matchB.empty = false;
  outA->hash   = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash   = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

// Single-bond minimal seed.  Should produce a non-zero, deterministic
// hash distinct from the empty-seed sentinel (1).
__global__ void hashSingleBondDriver(HashOut64* outA, HashOut64* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeed16  seedA, seedB;
  HashMatch16 matchA, matchB;
  mcs::fmcs::seedClearWithinThread(seedA);
  mcs::fmcs::seedClearWithinThread(seedB);
  mcs::fmcs::matchResultClearWithinThread(matchA);
  mcs::fmcs::matchResultClearWithinThread(matchB);
  // Determinism: two seeds with the same single (q=4, t=2) mapping.
  mcs::fmcs::seedAddBondWithinThread(seedA, 4);
  matchA.targetBondIdx[4] = 2;
  matchA.empty            = false;
  mcs::fmcs::seedAddBondWithinThread(seedB, 4);
  matchB.targetBondIdx[4] = 2;
  matchB.empty            = false;
  outA->hash              = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash              = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

// Same single query bond and target bond, opposite endpoint orientation.
// These must hash differently or the success cache can prune one Phase-1
// orientation because the other orientation was already seen.
__global__ void hashSingleBondOrientationsDifferDriver(HashOut64* outA, HashOut64* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeed16  seedA, seedB;
  HashMatch16 matchA, matchB;
  mcs::fmcs::seedClearWithinThread(seedA);
  mcs::fmcs::seedClearWithinThread(seedB);
  mcs::fmcs::matchResultClearWithinThread(matchA);
  mcs::fmcs::matchResultClearWithinThread(matchB);

  mcs::fmcs::seedAddAtomWithinThread(seedA, 0);
  mcs::fmcs::seedAddAtomWithinThread(seedA, 1);
  mcs::fmcs::seedAddBondWithinThread(seedA, 0);
  matchA.targetAtomIdx[0] = 4;
  matchA.targetAtomIdx[1] = 5;
  matchA.targetBondIdx[0] = 3;
  matchA.empty            = false;

  mcs::fmcs::seedAddAtomWithinThread(seedB, 0);
  mcs::fmcs::seedAddAtomWithinThread(seedB, 1);
  mcs::fmcs::seedAddBondWithinThread(seedB, 0);
  matchB.targetAtomIdx[0] = 5;
  matchB.targetAtomIdx[1] = 4;
  matchB.targetBondIdx[0] = 3;
  matchB.empty            = false;

  outA->hash = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

__global__ void hashLastAddedFrontierDiffersDriver(HashOut64* outA, HashOut64* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeed16  seedA, seedB;
  HashMatch16 matchA, matchB;
  mcs::fmcs::seedClearWithinThread(seedA);
  mcs::fmcs::seedClearWithinThread(seedB);
  mcs::fmcs::matchResultClearWithinThread(matchA);
  mcs::fmcs::matchResultClearWithinThread(matchB);

  for (int atom : {0, 1, 2}) {
    mcs::fmcs::seedAddAtomWithinThread(seedA, atom);
    mcs::fmcs::seedAddAtomWithinThread(seedB, atom);
    matchA.targetAtomIdx[atom] = static_cast<std::uint8_t>(atom + 4);
    matchB.targetAtomIdx[atom] = static_cast<std::uint8_t>(atom + 4);
  }
  for (int bond : {0, 1}) {
    mcs::fmcs::seedAddBondWithinThread(seedA, bond);
    mcs::fmcs::seedAddBondWithinThread(seedB, bond);
    matchA.targetBondIdx[bond] = static_cast<std::uint8_t>(bond + 6);
    matchB.targetBondIdx[bond] = static_cast<std::uint8_t>(bond + 6);
  }
  for (int i = 0; i < HashSeed16::kAtomWords; ++i) {
    seedA.lastAddedAtoms[i] = 0;
    seedB.lastAddedAtoms[i] = 0;
  }
  seedA.lastAddedAtoms[0] = static_cast<HashSeed16::atom_word_type>(1) << 1;
  seedB.lastAddedAtoms[0] = static_cast<HashSeed16::atom_word_type>(1) << 2;
  matchA.empty            = false;
  matchB.empty            = false;

  outA->hash = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

__global__ void hashExcludedBondsDifferDriver(HashOut64* outA, HashOut64* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeed16  seedA, seedB;
  HashMatch16 matchA, matchB;
  mcs::fmcs::seedClearWithinThread(seedA);
  mcs::fmcs::seedClearWithinThread(seedB);
  mcs::fmcs::matchResultClearWithinThread(matchA);
  mcs::fmcs::matchResultClearWithinThread(matchB);

  for (int atom : {0, 1, 2}) {
    mcs::fmcs::seedAddAtomWithinThread(seedA, atom);
    mcs::fmcs::seedAddAtomWithinThread(seedB, atom);
    matchA.targetAtomIdx[atom] = static_cast<std::uint8_t>(atom + 4);
    matchB.targetAtomIdx[atom] = static_cast<std::uint8_t>(atom + 4);
  }
  for (int bond : {0, 1}) {
    mcs::fmcs::seedAddBondWithinThread(seedA, bond);
    mcs::fmcs::seedAddBondWithinThread(seedB, bond);
    matchA.targetBondIdx[bond] = static_cast<std::uint8_t>(bond + 6);
    matchB.targetBondIdx[bond] = static_cast<std::uint8_t>(bond + 6);
  }
  seedB.excludedBonds[0] |= static_cast<HashSeed16::bond_word_type>(1) << 3;
  matchA.empty = false;
  matchB.empty = false;

  outA->hash = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

// Extreme indices: q = 127, t = 127 are the largest values supported
// (tier-128 cap on maxBonds).  Verify the (q << 8 | t) packing
// doesn't overflow and the high-q __ffsll iteration finds the bit.
__global__ void hashExtremeIndicesDriver(HashOut64* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeed128  seed;
  HashMatch128 match;
  mcs::fmcs::seedClearWithinThread(seed);
  mcs::fmcs::matchResultClearWithinThread(match);
  mcs::fmcs::seedAddBondWithinThread(seed, 127);
  match.targetBondIdx[127] = 127;
  match.empty              = false;
  out->hash                = mcs::fmcs::mappingHashWithinThread(seed, match);
}

// The (q=0, t=0) bond packs to a zero token; with a zero-initialized
// accumulator and splitMix64(0) == 0 it would be swallowed, aliasing a
// seed that contains it onto one that omits it.  A strict superset that
// adds the q0->t0 bond must therefore hash differently from the subset
// without it.
__global__ void hashZeroPairBondVisibleDriver(HashOut64* outWithout, HashOut64* outWith) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeed16  seedWithout, seedWith;
  HashMatch16 matchWithout, matchWith;
  mcs::fmcs::seedClearWithinThread(seedWithout);
  mcs::fmcs::seedClearWithinThread(seedWith);
  mcs::fmcs::matchResultClearWithinThread(matchWithout);
  mcs::fmcs::matchResultClearWithinThread(matchWith);
  mcs::fmcs::seedAddBondWithinThread(seedWithout, 1);
  matchWithout.targetBondIdx[1] = 1;
  matchWithout.empty            = false;
  mcs::fmcs::seedAddBondWithinThread(seedWith, 0);
  mcs::fmcs::seedAddBondWithinThread(seedWith, 1);
  matchWith.targetBondIdx[0] = 0;
  matchWith.targetBondIdx[1] = 1;
  matchWith.empty            = false;
  outWithout->hash           = mcs::fmcs::mappingHashWithinThread(seedWithout, matchWithout);
  outWith->hash              = mcs::fmcs::mappingHashWithinThread(seedWith, matchWith);
}

// ---- Cache idempotent insert ----

struct CacheIdempotentOut {
  bool firstInsert;
  bool secondInsert;
  int  occupiedSlots;  // post-state count of non-zero slots
};

__global__ void cacheIdempotentInsertDriver(std::uint64_t* keys, int capacity, CacheIdempotentOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  __shared__ DeviceMatchCache cache;
  cache.init(keys, capacity);
  const std::uint64_t key = 0xC0FFEEC0FFEEC0FFULL;
  out->firstInsert        = cache.insertWithinThread(key);
  out->secondInsert       = cache.insertWithinThread(key);
  int occupied            = 0;
  for (int i = 0; i < capacity; ++i)
    if (keys[i] != 0ULL)
      ++occupied;
  out->occupiedSlots = occupied;
}

// ---- Probe of fresh empty cache ----

struct ProbeEmptyOut {
  bool probedZeroKey;
  bool probedNonZeroKey;
};

__global__ void cacheProbeEmptyDriver(std::uint64_t* keys, int capacity, ProbeEmptyOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  __shared__ DeviceMatchCache cache;
  cache.init(keys, capacity);
  // Cache was zero-initialized by the host.  Probing any key should
  // miss immediately at the first slot (it's empty).
  out->probedZeroKey    = cache.probeWithinThread(0ULL);
  out->probedNonZeroKey = cache.probeWithinThread(0xABCDABCD12341234ULL);
}

// ---- Cooperative zero ----

__global__ void cacheZeroCooperativeDriver(std::uint64_t* keys, int capacity) {
  __shared__ DeviceMatchCache cache;
  if (threadIdx.x == 0)
    cache.init(keys, capacity);
  __syncthreads();
  auto block = cooperative_groups::this_thread_block();
  cache.zeroCooperative(block);
}

}  // namespace mcs_fmcs_cache_gap_test

TEST(FMCSUnit, MappingHashMultiWordPath) {
  using namespace mcs_fmcs_cache_gap_test;
  AsyncDevicePtr<HashOut64> d_dA;
  AsyncDevicePtr<HashOut64> d_dB;
  hashMultiWordPathDriver<<<1, 1>>>(d_dA.data(), d_dB.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut64 dA{};
  d_dA.get(dA);
  HashOut64 dB{};
  d_dB.get(dB);

  EXPECT_NE(dA.hash, 0ULL);
  EXPECT_EQ(dA.hash, dB.hash) << "Multi-word seed.bonds must be canonicalized regardless of "
                                 "addition order, even when bonds straddle a word boundary.";
}

TEST(FMCSUnit, MappingHashSingleBondDeterministic) {
  using namespace mcs_fmcs_cache_gap_test;
  AsyncDevicePtr<HashOut64> d_dA;
  AsyncDevicePtr<HashOut64> d_dB;
  hashSingleBondDriver<<<1, 1>>>(d_dA.data(), d_dB.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut64 dA{};
  d_dA.get(dA);
  HashOut64 dB{};
  d_dB.get(dB);

  EXPECT_NE(dA.hash, 0ULL);
  EXPECT_NE(dA.hash, 1ULL) << "Should differ from the empty-seed sentinel";
  EXPECT_EQ(dA.hash, dB.hash);
}

TEST(FMCSUnit, MappingHashSingleBondOrientationsDiffer) {
  using namespace mcs_fmcs_cache_gap_test;
  AsyncDevicePtr<HashOut64> d_dA;
  AsyncDevicePtr<HashOut64> d_dB;
  hashSingleBondOrientationsDifferDriver<<<1, 1>>>(d_dA.data(), d_dB.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut64 dA{};
  d_dA.get(dA);
  HashOut64 dB{};
  d_dB.get(dB);

  EXPECT_NE(dA.hash, dB.hash) << "Opposite orientations of the same single bond must not alias.";
}

TEST(FMCSUnit, MappingHashLastAddedFrontierDiffers) {
  using namespace mcs_fmcs_cache_gap_test;
  AsyncDevicePtr<HashOut64> d_dA;
  AsyncDevicePtr<HashOut64> d_dB;
  hashLastAddedFrontierDiffersDriver<<<1, 1>>>(d_dA.data(), d_dB.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut64 dA{};
  d_dA.get(dA);
  HashOut64 dB{};
  d_dB.get(dB);

  EXPECT_NE(dA.hash, dB.hash) << "Same mapped subgraph with different last-added frontier "
                                 "must not cache-alias.";
}

TEST(FMCSUnit, MappingHashExcludedBondsDiffer) {
  using namespace mcs_fmcs_cache_gap_test;
  AsyncDevicePtr<HashOut64> d_dA;
  AsyncDevicePtr<HashOut64> d_dB;
  hashExcludedBondsDifferDriver<<<1, 1>>>(d_dA.data(), d_dB.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut64 dA{};
  d_dA.get(dA);
  HashOut64 dB{};
  d_dB.get(dB);

  EXPECT_NE(dA.hash, dB.hash) << "Same mapped subgraph with different excluded bonds must not "
                                 "cache-alias.";
}

TEST(FMCSUnit, MappingHashExtremeIndicesFitInPacking) {
  using namespace mcs_fmcs_cache_gap_test;
  AsyncDevicePtr<HashOut64> d_d;
  hashExtremeIndicesDriver<<<1, 1>>>(d_d.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut64 d{};
  d_d.get(d);

  // The packed pair (127 << 8) | 127 = 0x7F7F fits comfortably in
  // 64 bits, and __ffsll finds bit 63 of word 1 correctly.  Just
  // assert non-zero (i.e., we got past the hash post-process and
  // didn't hit a degenerate result).
  EXPECT_NE(d.hash, 0ULL);
}

TEST(FMCSUnit, MappingHashZeroPairBondIsVisible) {
  using namespace mcs_fmcs_cache_gap_test;
  AsyncDevicePtr<HashOut64> d_without;
  AsyncDevicePtr<HashOut64> d_with;
  hashZeroPairBondVisibleDriver<<<1, 1>>>(d_without.data(), d_with.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut64 without{};
  d_without.get(without);
  HashOut64 with{};
  d_with.get(with);

  EXPECT_NE(with.hash, without.hash) << "A (q0->t0) bond must perturb the hash; otherwise a full seed "
                                        "aliases onto a sub-seed and the success cache falsely dedups it.";
  EXPECT_NE(with.hash, 0ULL);
}

TEST(FMCSUnit, CacheInsertIsIdempotent) {
  using namespace mcs_fmcs_cache_gap_test;
  constexpr int                    kCap = 16;
  AsyncDeviceVector<std::uint64_t> d_keys(kCap);
  d_keys.zero();
  AsyncDevicePtr<CacheIdempotentOut> d_out;

  cacheIdempotentInsertDriver<<<1, 1>>>(d_keys.data(), kCap, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  CacheIdempotentOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.firstInsert);
  EXPECT_TRUE(out.secondInsert);
  EXPECT_EQ(out.occupiedSlots, 1) << "Repeated insert of the same key must not consume a second slot.";
}

TEST(FMCSUnit, CacheProbeEmptyCacheMisses) {
  using namespace mcs_fmcs_cache_gap_test;
  constexpr int                    kCap = 16;
  AsyncDeviceVector<std::uint64_t> d_keys(kCap);
  d_keys.zero();
  AsyncDevicePtr<ProbeEmptyOut> d_out;

  cacheProbeEmptyDriver<<<1, 1>>>(d_keys.data(), kCap, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ProbeEmptyOut out{};
  d_out.get(out);

  EXPECT_FALSE(out.probedZeroKey) << "0 is the empty-slot sentinel; probing it must miss.";
  EXPECT_FALSE(out.probedNonZeroKey);
}

TEST(FMCSUnit, CacheZeroCooperativeWipesAllSlots) {
  using namespace mcs_fmcs_cache_gap_test;
  constexpr int                    kCap = 64;
  AsyncDeviceVector<std::uint64_t> d_keys(kCap);
  // Pre-fill with non-zero garbage so the cooperative zero has work
  // to do.
  std::vector<std::uint64_t>       keys(kCap);
  for (int i = 0; i < kCap; ++i)
    keys[i] = 0xDEADDEAD00000000ULL | i;
  d_keys.copyFromHost(keys);

  cacheZeroCooperativeDriver<<<1, 32>>>(d_keys.data(), kCap);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  d_keys.copyToHost(keys);
  for (int i = 0; i < kCap; ++i) {
    EXPECT_EQ(keys[i], 0ULL) << "slot " << i << " not zeroed";
  }
}
