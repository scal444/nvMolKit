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
//
// Unit tests for the per-helper device functions in fmcs_cuda/.  Each test
// launches a tiny __global__ driver that constructs inputs, calls one
// helper, and copies results back to host memory for assertion.  This
// file is populated incrementally as Steps 1-5 land real implementations.

#include "fmcs_cuda/fmcs_grow.cuh"
#include "fmcs_cuda/fmcs_kernel.cuh"
#include "fmcs_cuda/fmcs_match.cuh"
#include "fmcs_cuda/fmcs_match_tables.cuh"
#include "fmcs_cuda/fmcs_seed.cuh"
#include "fmcs_cuda/fmcs_seed_queue.cuh"

#include <cooperative_groups.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <set>
#include <vector>

namespace {

using mcs::fmcs::MatchResult;
using mcs::fmcs::Seed;

template<typename T>
T* mallocManaged() {
  T* ptr = nullptr;
  if (cudaMallocManaged(&ptr, sizeof(T)) != cudaSuccess) return nullptr;
  return ptr;
}

// ---------------------------------------------------------------------------
// matchResultClearWithinThread
// ---------------------------------------------------------------------------

__global__ void matchResultClearDriverKernel(
    MatchResult<16, 16, 16, 16>* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  MatchResult<16, 16, 16, 16> m;
  for (int i = 0; i < 16; ++i) m.targetAtomIdx[i] = 7;
  for (int i = 0; i < 16; ++i) m.targetBondIdx[i] = 7;
  // Pre-populate visited bitsets too so we can verify they're zeroed.
  m.visitedTargetAtoms[0] = 0xDEADBEEFu;
  m.visitedTargetBonds[0] = 0xCAFEBABEu;
  m.matchedAtomSize = 13;
  m.matchedBondSize = 11;
  m.empty = false;
  mcs::fmcs::matchResultClearWithinThread(m);
  *out = m;
}

}  // namespace

TEST(FMCSUnit, MatchResultClearZeroesEverything) {
  MatchResult<16, 16, 16, 16>* d_out = mallocManaged<MatchResult<16, 16, 16, 16>>();
  ASSERT_NE(d_out, nullptr);

  matchResultClearDriverKernel<<<1, 1>>>(d_out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(d_out->targetAtomIdx[i], 0xFFu);
    EXPECT_EQ(d_out->targetBondIdx[i], 0xFFu);
  }
  EXPECT_EQ(d_out->visitedTargetAtoms[0], 0u);
  EXPECT_EQ(d_out->visitedTargetBonds[0], 0u);
  EXPECT_TRUE(d_out->empty);
  EXPECT_EQ(d_out->matchedAtomSize, 0);
  EXPECT_EQ(d_out->matchedBondSize, 0);

  cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// seedAddAtomWithinThread
// ---------------------------------------------------------------------------

namespace {

__global__ void seedAddAtomDriverKernel(Seed<16, 16>* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  Seed<16, 16> seed{};
  mcs::fmcs::seedAddAtomWithinThread(seed, 3);
  mcs::fmcs::seedAddAtomWithinThread(seed, 5);
  mcs::fmcs::seedAddAtomWithinThread(seed, 12);
  *out = seed;
}

}  // namespace

TEST(FMCSUnit, SeedAddAtomSetsBitsAndCount) {
  Seed<16, 16>* d_out = mallocManaged<Seed<16, 16>>();
  ASSERT_NE(d_out, nullptr);

  seedAddAtomDriverKernel<<<1, 1>>>(d_out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Tier 16 packs the atom bitset into one uint32 word.
  const std::uint32_t expected =
      (1u << 3) | (1u << 5) | (1u << 12);
  EXPECT_EQ(d_out->atoms[0], expected);
  EXPECT_EQ(d_out->numAtoms, 3);
  // No bond / excludedBonds touched.
  EXPECT_EQ(d_out->bonds[0], 0u);
  EXPECT_EQ(d_out->excludedBonds[0], 0u);
  EXPECT_EQ(d_out->numBonds, 0);

  cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// seedAddBondWithinThread
// ---------------------------------------------------------------------------

namespace {

__global__ void seedAddBondDriverKernel(Seed<16, 16>* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  Seed<16, 16> seed{};
  mcs::fmcs::seedAddBondWithinThread(seed, 2);
  mcs::fmcs::seedAddBondWithinThread(seed, 7);
  *out = seed;
}

}  // namespace

TEST(FMCSUnit, SeedAddBondSetsBondsAndExcludedAndCount) {
  Seed<16, 16>* d_out = mallocManaged<Seed<16, 16>>();
  ASSERT_NE(d_out, nullptr);

  seedAddBondDriverKernel<<<1, 1>>>(d_out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const std::uint32_t expected = (1u << 2) | (1u << 7);
  EXPECT_EQ(d_out->bonds[0], expected);
  EXPECT_EQ(d_out->excludedBonds[0], expected);
  EXPECT_EQ(d_out->numBonds, 2);
  // Atom side untouched.
  EXPECT_EQ(d_out->atoms[0], 0u);
  EXPECT_EQ(d_out->numAtoms, 0);

  cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// seedBeginGrowStepWithinThread
// ---------------------------------------------------------------------------

namespace {

__global__ void seedBeginGrowStepDriverKernel(Seed<16, 16>* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  Seed<16, 16> seed{};
  mcs::fmcs::seedAddAtomWithinThread(seed, 0);
  mcs::fmcs::seedAddAtomWithinThread(seed, 1);
  mcs::fmcs::seedAddAtomWithinThread(seed, 2);
  // Boundary should snap to numAtoms=3 here.
  mcs::fmcs::seedBeginGrowStepWithinThread(seed);
  // Subsequent adds advance numAtoms but not the boundary.
  mcs::fmcs::seedAddAtomWithinThread(seed, 5);
  mcs::fmcs::seedAddAtomWithinThread(seed, 9);
  *out = seed;
}

}  // namespace

TEST(FMCSUnit, SeedBeginGrowStepSnapshotsNumAtoms) {
  Seed<16, 16>* d_out = mallocManaged<Seed<16, 16>>();
  ASSERT_NE(d_out, nullptr);

  seedBeginGrowStepDriverKernel<<<1, 1>>>(d_out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(d_out->numAtoms, 5);
  EXPECT_EQ(d_out->lastAddedAtomsBegin, 3);

  cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// seedMarkLastAddedAtomWithinThread
// ---------------------------------------------------------------------------

namespace {

__global__ void seedMarkLastAddedAtomDriverKernel(Seed<16, 16>* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  Seed<16, 16> seed{};
  mcs::fmcs::seedAddAtomWithinThread(seed, 0);
  mcs::fmcs::seedAddAtomWithinThread(seed, 1);
  mcs::fmcs::seedBeginGrowStepWithinThread(seed);
  mcs::fmcs::seedMarkLastAddedAtomWithinThread(seed, 0);
  *out = seed;
}

}  // namespace

TEST(FMCSUnit, SeedMarkLastAddedAtomDoesNotChangeSeedAtomsOrCount) {
  Seed<16, 16>* d_out = mallocManaged<Seed<16, 16>>();
  ASSERT_NE(d_out, nullptr);

  seedMarkLastAddedAtomDriverKernel<<<1, 1>>>(d_out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(d_out->atoms[0], 0x3u);
  EXPECT_EQ(d_out->lastAddedAtoms[0], 0x1u);
  EXPECT_EQ(d_out->numAtoms, 2);
  EXPECT_EQ(d_out->lastAddedAtomsBegin, 2);

  cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// seedCanGrowBiggerThanWithinThread
// ---------------------------------------------------------------------------

namespace {

struct CanGrowResults {
  bool moreBonds;          // possible bonds 8 vs best 5: true
  bool fewerBonds;         // possible bonds 5 vs best 8: false
  bool tiedBondsMoreAtoms; // possible bonds 8 best 8, atoms tie-break: true
  bool tiedBondsFewerAtoms;
  bool tiedBondsTiedAtoms;
};

__global__ void seedCanGrowDriverKernel(CanGrowResults* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  Seed<16, 16> seed{};
  seed.numBonds       = 5;
  seed.remainingBonds = 3;   // possibleBonds = 8
  seed.numAtoms       = 4;
  seed.remainingAtoms = 2;   // possibleAtoms = 6

  out->moreBonds =
      mcs::fmcs::seedCanGrowBiggerThanWithinThread(seed, /*bestBonds=*/5, /*bestAtoms=*/0);
  out->fewerBonds =
      mcs::fmcs::seedCanGrowBiggerThanWithinThread(seed, /*bestBonds=*/9, /*bestAtoms=*/0);
  out->tiedBondsMoreAtoms =
      mcs::fmcs::seedCanGrowBiggerThanWithinThread(seed, /*bestBonds=*/8, /*bestAtoms=*/5);
  out->tiedBondsFewerAtoms =
      mcs::fmcs::seedCanGrowBiggerThanWithinThread(seed, /*bestBonds=*/8, /*bestAtoms=*/7);
  out->tiedBondsTiedAtoms =
      mcs::fmcs::seedCanGrowBiggerThanWithinThread(seed, /*bestBonds=*/8, /*bestAtoms=*/6);
}

}  // namespace

TEST(FMCSUnit, SeedCanGrowBiggerThanBoundCheck) {
  CanGrowResults* d_out = mallocManaged<CanGrowResults>();
  ASSERT_NE(d_out, nullptr);

  seedCanGrowDriverKernel<<<1, 1>>>(d_out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(d_out->moreBonds);
  EXPECT_FALSE(d_out->fewerBonds);
  EXPECT_TRUE(d_out->tiedBondsMoreAtoms);
  EXPECT_FALSE(d_out->tiedBondsFewerAtoms);
  EXPECT_FALSE(d_out->tiedBondsTiedAtoms);  // strictly bigger required

  cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// seedComputeRemainingSizeWithinThread
// ---------------------------------------------------------------------------

namespace {

struct FakeQueryTopology {
  int numAtoms;
  int numBonds;
};

__global__ void seedComputeRemainingDriverKernel16(Seed<16, 16>* out,
                                                   FakeQueryTopology topo) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  Seed<16, 16> seed{};
  mcs::fmcs::seedAddAtomWithinThread(seed, 0);
  mcs::fmcs::seedAddAtomWithinThread(seed, 1);
  // Three excluded bonds at indices 0, 2, 3.
  mcs::fmcs::seedAddBondWithinThread(seed, 0);
  mcs::fmcs::seedAddBondWithinThread(seed, 2);
  mcs::fmcs::seedAddBondWithinThread(seed, 3);
  mcs::fmcs::seedComputeRemainingSizeWithinThread(seed, topo);
  *out = seed;
}

__global__ void seedComputeRemainingDriverKernel64(Seed<64, 64>* out,
                                                   FakeQueryTopology topo) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  Seed<64, 64> seed{};
  // Spread bond exclusions across the uint64 word (and into a second
  // word would require maxBonds > 64; we exercise just the popcll path).
  for (int b : {0, 17, 33, 60}) {
    mcs::fmcs::seedAddBondWithinThread(seed, b);
  }
  for (int a : {0, 1, 5, 17, 32}) {
    mcs::fmcs::seedAddAtomWithinThread(seed, a);
  }
  mcs::fmcs::seedComputeRemainingSizeWithinThread(seed, topo);
  *out = seed;
}

}  // namespace

TEST(FMCSUnit, SeedComputeRemainingSizeLooseBoundUint32Word) {
  Seed<16, 16>* d_out = mallocManaged<Seed<16, 16>>();
  ASSERT_NE(d_out, nullptr);

  FakeQueryTopology topo{/*numAtoms=*/10, /*numBonds=*/15};
  seedComputeRemainingDriverKernel16<<<1, 1>>>(d_out, topo);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(d_out->numAtoms, 2);
  EXPECT_EQ(d_out->numBonds, 3);
  EXPECT_EQ(d_out->remainingAtoms, 10 - 2);
  EXPECT_EQ(d_out->remainingBonds, 15 - 3);

  cudaFree(d_out);
}

TEST(FMCSUnit, SeedComputeRemainingSizeLooseBoundUint64Word) {
  Seed<64, 64>* d_out = mallocManaged<Seed<64, 64>>();
  ASSERT_NE(d_out, nullptr);

  FakeQueryTopology topo{/*numAtoms=*/40, /*numBonds=*/50};
  seedComputeRemainingDriverKernel64<<<1, 1>>>(d_out, topo);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(d_out->numAtoms, 5);
  EXPECT_EQ(d_out->numBonds, 4);
  EXPECT_EQ(d_out->remainingAtoms, 40 - 5);
  EXPECT_EQ(d_out->remainingBonds, 50 - 4);

  cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// Multi-word and boundary edge cases
// ---------------------------------------------------------------------------

namespace {

__global__ void seedAddAtomMultiWordDriverKernel(Seed<128, 128>* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  Seed<128, 128> seed{};
  // Atoms at the four corners of the 2-uint64-word atom bitset:
  //   0 (word 0, bit 0), 63 (word 0, bit 63),
  //  64 (word 1, bit 0), 127 (word 1, bit 63).
  mcs::fmcs::seedAddAtomWithinThread(seed, 0);
  mcs::fmcs::seedAddAtomWithinThread(seed, 63);
  mcs::fmcs::seedAddAtomWithinThread(seed, 64);
  mcs::fmcs::seedAddAtomWithinThread(seed, 127);
  *out = seed;
}

__global__ void seedAddBondMultiWordDriverKernel(Seed<128, 128>* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  Seed<128, 128> seed{};
  mcs::fmcs::seedAddBondWithinThread(seed, 0);
  mcs::fmcs::seedAddBondWithinThread(seed, 63);
  mcs::fmcs::seedAddBondWithinThread(seed, 64);
  mcs::fmcs::seedAddBondWithinThread(seed, 127);
  *out = seed;
}

__global__ void seedAddAtomBoundaryDriverKernel(Seed<16, 16>* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  Seed<16, 16> seed{};
  mcs::fmcs::seedAddAtomWithinThread(seed, 0);
  mcs::fmcs::seedAddAtomWithinThread(seed, 15);  // tier-16 max valid atom idx
  *out = seed;
}

}  // namespace

TEST(FMCSUnit, SeedAddAtomCrossesWordBoundary) {
  Seed<128, 128>* d_out = mallocManaged<Seed<128, 128>>();
  ASSERT_NE(d_out, nullptr);

  seedAddAtomMultiWordDriverKernel<<<1, 1>>>(d_out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const std::uint64_t bit0  = std::uint64_t{1} << 0;
  const std::uint64_t bit63 = std::uint64_t{1} << 63;
  EXPECT_EQ(d_out->atoms[0], bit0 | bit63);
  EXPECT_EQ(d_out->atoms[1], bit0 | bit63);
  EXPECT_EQ(d_out->numAtoms, 4);

  cudaFree(d_out);
}

TEST(FMCSUnit, SeedAddBondCrossesWordBoundary) {
  Seed<128, 128>* d_out = mallocManaged<Seed<128, 128>>();
  ASSERT_NE(d_out, nullptr);

  seedAddBondMultiWordDriverKernel<<<1, 1>>>(d_out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const std::uint64_t bit0  = std::uint64_t{1} << 0;
  const std::uint64_t bit63 = std::uint64_t{1} << 63;
  const std::uint64_t expected = bit0 | bit63;
  EXPECT_EQ(d_out->bonds[0], expected);
  EXPECT_EQ(d_out->bonds[1], expected);
  EXPECT_EQ(d_out->excludedBonds[0], expected);
  EXPECT_EQ(d_out->excludedBonds[1], expected);
  EXPECT_EQ(d_out->numBonds, 4);

  cudaFree(d_out);
}

TEST(FMCSUnit, SeedAddAtomBoundaryIndices) {
  Seed<16, 16>* d_out = mallocManaged<Seed<16, 16>>();
  ASSERT_NE(d_out, nullptr);

  seedAddAtomBoundaryDriverKernel<<<1, 1>>>(d_out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(d_out->atoms[0], (1u << 0) | (1u << 15));
  EXPECT_EQ(d_out->numAtoms, 2);

  cudaFree(d_out);
}

namespace {

struct CanGrowFromEmptyResults {
  bool nonEmptyVsZero;  // possibleBonds=2 vs (0,0): true
  bool emptyVsZero;     // possibleBonds=0 vs (0,0): false (strict >)
};

__global__ void seedCanGrowFromEmptyDriverKernel(CanGrowFromEmptyResults* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  Seed<16, 16> live{};
  live.numBonds       = 1;
  live.remainingBonds = 1;
  out->nonEmptyVsZero =
      mcs::fmcs::seedCanGrowBiggerThanWithinThread(live, 0, 0);
  Seed<16, 16> empty{};
  out->emptyVsZero =
      mcs::fmcs::seedCanGrowBiggerThanWithinThread(empty, 0, 0);
}

}  // namespace

TEST(FMCSUnit, SeedCanGrowBiggerThanFromEmptyIncumbent) {
  CanGrowFromEmptyResults* d_out = mallocManaged<CanGrowFromEmptyResults>();
  ASSERT_NE(d_out, nullptr);

  seedCanGrowFromEmptyDriverKernel<<<1, 1>>>(d_out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(d_out->nonEmptyVsZero);
  EXPECT_FALSE(d_out->emptyVsZero);

  cudaFree(d_out);
}

namespace {

__global__ void seedComputeRemainingEmptyDriverKernel(Seed<16, 16>* out,
                                                      FakeQueryTopology topo) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  Seed<16, 16> seed{};  // No atoms, no bonds, no excludedBonds.
  mcs::fmcs::seedComputeRemainingSizeWithinThread(seed, topo);
  *out = seed;
}

__global__ void seedComputeRemainingFullDriverKernel(Seed<16, 16>* out,
                                                     FakeQueryTopology topo) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  Seed<16, 16> seed{};
  // Exclude every bond up to topo.numBonds-1.
  for (int b = 0; b < topo.numBonds; ++b) {
    mcs::fmcs::seedAddBondWithinThread(seed, b);
  }
  mcs::fmcs::seedComputeRemainingSizeWithinThread(seed, topo);
  *out = seed;
}

__global__ void seedComputeRemainingMultiWordDriverKernel(Seed<128, 128>* out,
                                                          FakeQueryTopology topo) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  Seed<128, 128> seed{};
  // One excluded bond in each of the two uint64 words.
  mcs::fmcs::seedAddBondWithinThread(seed, 5);    // word 0
  mcs::fmcs::seedAddBondWithinThread(seed, 100);  // word 1
  mcs::fmcs::seedComputeRemainingSizeWithinThread(seed, topo);
  *out = seed;
}

}  // namespace

TEST(FMCSUnit, SeedComputeRemainingSizeEmptySeed) {
  Seed<16, 16>* d_out = mallocManaged<Seed<16, 16>>();
  ASSERT_NE(d_out, nullptr);

  FakeQueryTopology topo{/*numAtoms=*/12, /*numBonds=*/9};
  seedComputeRemainingEmptyDriverKernel<<<1, 1>>>(d_out, topo);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(d_out->remainingAtoms, 12);
  EXPECT_EQ(d_out->remainingBonds, 9);

  cudaFree(d_out);
}

TEST(FMCSUnit, SeedComputeRemainingSizeAllBondsExcluded) {
  Seed<16, 16>* d_out = mallocManaged<Seed<16, 16>>();
  ASSERT_NE(d_out, nullptr);

  FakeQueryTopology topo{/*numAtoms=*/10, /*numBonds=*/12};
  seedComputeRemainingFullDriverKernel<<<1, 1>>>(d_out, topo);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(d_out->numBonds, 12);
  EXPECT_EQ(d_out->remainingBonds, 0);
  // numAtoms still 0 since we only added bonds.
  EXPECT_EQ(d_out->remainingAtoms, 10);

  cudaFree(d_out);
}

TEST(FMCSUnit, SeedComputeRemainingSizeMultiWordPopcount) {
  Seed<128, 128>* d_out = mallocManaged<Seed<128, 128>>();
  ASSERT_NE(d_out, nullptr);

  FakeQueryTopology topo{/*numAtoms=*/100, /*numBonds=*/120};
  seedComputeRemainingMultiWordDriverKernel<<<1, 1>>>(d_out, topo);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Two excluded bonds total -- one in each word.
  EXPECT_EQ(d_out->remainingBonds, 120 - 2);
  EXPECT_EQ(d_out->remainingAtoms, 100);

  cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// SeedQueue: pushWithinThread / popWithinThread / batchReserveCooperative
// ---------------------------------------------------------------------------

namespace {

namespace cg = cooperative_groups;

using mcs::fmcs::SeedQueue;
using mcs::fmcs::ThreadBlockScope;

// Single-thread driver: push 5 ints, pop 5, record the popped order so
// the host can verify LIFO.
__global__ void queuePushPopLIFODriver(int* poppedOrder, int* outSizeAfterPushes) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  __shared__ int storage[8];
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  queue.init(storage, 8);
  for (int v : {10, 20, 30, 40, 50}) {
    bool ok = queue.pushWithinThread(v);
    (void)ok;
  }
  *outSizeAfterPushes = queue.size();
  for (int i = 0; i < 5; ++i) {
    int popped = -1;
    bool ok = queue.popWithinThread(popped);
    poppedOrder[i] = ok ? popped : -1;
  }
}

__global__ void queueOverflowDriver(bool* pushOk, int* finalSize) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  __shared__ int storage[4];
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  queue.init(storage, 4);
  for (int i = 0; i < 5; ++i) {
    pushOk[i] = queue.pushWithinThread(100 + i);
  }
  *finalSize = queue.size();
}

__global__ void queuePopFromEmptyDriver(bool* popOk, int* outVal) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  __shared__ int storage[4];
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  queue.init(storage, 4);
  int v = 12345;
  *popOk = queue.popWithinThread(v);
  *outVal = v;
}

// Multi-thread (1 warp = 32 threads): each lane pushes its own payload.
// Verifies CAS-based pushes don't lose any concurrent inserts.
__global__ void queueConcurrentPushesDriver(int* outStorage,
                                            int* outSize) {
  __shared__ int storage[64];
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  if (threadIdx.x == 0) queue.init(storage, 64);
  __syncthreads();

  bool ok = queue.pushWithinThread(1000 + static_cast<int>(threadIdx.x));
  __syncthreads();
  (void)ok;

  if (threadIdx.x == 0) {
    *outSize = queue.size();
    for (int i = 0; i < queue.size(); ++i) outStorage[i] = storage[i];
  }
}

// Multi-thread: 32 lanes do a single batchReserveCooperative call,
// each lane writes its own payload into the reserved slot.  Verifies
// the cooperative reservation broadcasts correctly and all 32 slots
// land contiguously.
__global__ void queueBatchReserveDriver(int* outStorage,
                                        int* outStart,
                                        int* outSize) {
  __shared__ int storage[64];
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  if (threadIdx.x == 0) queue.init(storage, 64);
  __syncthreads();

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);
  const int laneId = static_cast<int>(warp.thread_rank());
  const int start = queue.batchReserveCooperative(warp, 32);
  if (start >= 0) {
    queue.slot(start + laneId) = 2000 + laneId;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    *outStart = start;
    *outSize  = queue.size();
    for (int i = 0; i < queue.size(); ++i) outStorage[i] = storage[i];
  }
}

__global__ void queueBatchReserveOverflowDriver(int* outStart, int* outSize) {
  __shared__ int storage[10];
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  if (threadIdx.x == 0) queue.init(storage, 10);
  __syncthreads();

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);
  const int start = queue.batchReserveCooperative(warp, 15);
  __syncthreads();

  if (threadIdx.x == 0) {
    *outStart = start;
    *outSize = queue.size();
  }
}

// Capacity 10, pre-push 7 elements, then request 5 slots: only 3 free.
// Reservation must fail atomically (no partial reservation, no slots
// consumed) -- partial fit must be treated as a hard failure.
__global__ void queueBatchReservePartialFitDriver(int* outStart,
                                                  int* outSizeBefore,
                                                  int* outSizeAfter) {
  __shared__ int storage[10];
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  if (threadIdx.x == 0) {
    queue.init(storage, 10);
    for (int i = 0; i < 7; ++i) queue.pushWithinThread(i);
  }
  __syncthreads();

  if (threadIdx.x == 0) *outSizeBefore = queue.size();
  __syncthreads();

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);
  const int start = queue.batchReserveCooperative(warp, 5);
  __syncthreads();

  if (threadIdx.x == 0) {
    *outStart = start;
    *outSizeAfter = queue.size();
  }
}

}  // namespace

TEST(FMCSUnit, QueuePushPopLIFO) {
  int* d_order  = nullptr;
  int* d_size   = nullptr;
  ASSERT_EQ(cudaMallocManaged(&d_order, sizeof(int) * 5), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&d_size, sizeof(int)), cudaSuccess);

  queuePushPopLIFODriver<<<1, 1>>>(d_order, d_size);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(*d_size, 5);
  // LIFO: first popped should be the last pushed.
  EXPECT_EQ(d_order[0], 50);
  EXPECT_EQ(d_order[1], 40);
  EXPECT_EQ(d_order[2], 30);
  EXPECT_EQ(d_order[3], 20);
  EXPECT_EQ(d_order[4], 10);

  cudaFree(d_order);
  cudaFree(d_size);
}

TEST(FMCSUnit, QueuePushOverflowReturnsFalse) {
  bool* d_ok   = nullptr;
  int*  d_size = nullptr;
  ASSERT_EQ(cudaMallocManaged(&d_ok, sizeof(bool) * 5), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&d_size, sizeof(int)), cudaSuccess);

  queueOverflowDriver<<<1, 1>>>(d_ok, d_size);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(d_ok[0]);
  EXPECT_TRUE(d_ok[1]);
  EXPECT_TRUE(d_ok[2]);
  EXPECT_TRUE(d_ok[3]);
  EXPECT_FALSE(d_ok[4]);   // 5th push should fail; capacity == 4.
  EXPECT_EQ(*d_size, 4);

  cudaFree(d_ok);
  cudaFree(d_size);
}

TEST(FMCSUnit, QueuePopFromEmptyReturnsFalse) {
  bool* d_ok  = nullptr;
  int*  d_val = nullptr;
  ASSERT_EQ(cudaMallocManaged(&d_ok, sizeof(bool)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&d_val, sizeof(int)), cudaSuccess);

  queuePopFromEmptyDriver<<<1, 1>>>(d_ok, d_val);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_FALSE(*d_ok);
  EXPECT_EQ(*d_val, 12345);  // Caller's outElement should be untouched.

  cudaFree(d_ok);
  cudaFree(d_val);
}

TEST(FMCSUnit, QueueConcurrentPushesAllLand) {
  int* d_storage = nullptr;
  int* d_size    = nullptr;
  ASSERT_EQ(cudaMallocManaged(&d_storage, sizeof(int) * 64), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&d_size, sizeof(int)), cudaSuccess);

  queueConcurrentPushesDriver<<<1, 32>>>(d_storage, d_size);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  ASSERT_EQ(*d_size, 32);
  // All 32 lanes' payloads should be present exactly once.  Order is
  // non-deterministic across CAS races; check by set membership.
  std::set<int> seen;
  for (int i = 0; i < 32; ++i) seen.insert(d_storage[i]);
  EXPECT_EQ(seen.size(), 32u);
  for (int i = 0; i < 32; ++i) {
    EXPECT_TRUE(seen.count(1000 + i)) << "missing lane " << i;
  }

  cudaFree(d_storage);
  cudaFree(d_size);
}

TEST(FMCSUnit, QueueBatchReserveCooperative) {
  int* d_storage = nullptr;
  int* d_start   = nullptr;
  int* d_size    = nullptr;
  ASSERT_EQ(cudaMallocManaged(&d_storage, sizeof(int) * 64), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&d_start, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&d_size, sizeof(int)), cudaSuccess);

  queueBatchReserveDriver<<<1, 32>>>(d_storage, d_start, d_size);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(*d_start, 0);
  EXPECT_EQ(*d_size, 32);
  // Each lane wrote 2000 + laneId into slot start + laneId; storage[i]
  // should equal 2000 + i for i in [0, 32).
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(d_storage[i], 2000 + i);
  }

  cudaFree(d_storage);
  cudaFree(d_start);
  cudaFree(d_size);
}

TEST(FMCSUnit, QueueBatchReserveOverflowReturnsNegativeOne) {
  int* d_start = nullptr;
  int* d_size  = nullptr;
  ASSERT_EQ(cudaMallocManaged(&d_start, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&d_size, sizeof(int)), cudaSuccess);

  queueBatchReserveOverflowDriver<<<1, 32>>>(d_start, d_size);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(*d_start, -1);
  EXPECT_EQ(*d_size, 0);  // No slots consumed on overflow.

  cudaFree(d_start);
  cudaFree(d_size);
}

TEST(FMCSUnit, QueueBatchReservePartialFitFails) {
  int* d_start       = nullptr;
  int* d_sizeBefore  = nullptr;
  int* d_sizeAfter   = nullptr;
  ASSERT_EQ(cudaMallocManaged(&d_start, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&d_sizeBefore, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&d_sizeAfter, sizeof(int)), cudaSuccess);

  queueBatchReservePartialFitDriver<<<1, 32>>>(d_start, d_sizeBefore, d_sizeAfter);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(*d_sizeBefore, 7);
  EXPECT_EQ(*d_start, -1);    // Reservation must fail atomically.
  EXPECT_EQ(*d_sizeAfter, 7); // No partial slots consumed; size unchanged.

  cudaFree(d_start);
  cudaFree(d_sizeBefore);
  cudaFree(d_sizeAfter);
}

// ---------------------------------------------------------------------------
// matchSingleBondWithinThread / matchIncrementalFastCooperative
// ---------------------------------------------------------------------------

namespace {

using mcs::fmcs::SingleBondMatch;
using mcs::fmcs::PairMatchTablesDevice;
using mcs::fmcs::MatchTableDevice;

// Tiny CSR-view used by the match helpers via duck-typing; satisfies the
// QueryTopology / TargetTopology template requirement (bondEndpoints +
// numAtoms / numBonds), with optional CSR adjacency fields.
struct TestCsrView {
  static constexpr bool kHasAdjacencyBondIndices = false;

  const std::uint32_t* bondEndpoints = nullptr;
  int numAtoms = 0;
  int numBonds = 0;
  const std::uint32_t* rowOffsets = nullptr;
  const std::uint32_t* colIndices = nullptr;
  const std::uint32_t* bondIndices = nullptr;
};

// RAII wrapper for managed-memory match tables.  Both atom and bond
// tables are 32-bit row-packed bitmasks (one bit per (q, t) pair).
struct ManagedMatchTables {
  std::uint32_t* atomData = nullptr;
  std::uint32_t* bondData = nullptr;
  PairMatchTablesDevice device{};
  int qNumAtoms = 0;
  int qNumBonds = 0;

  void allocate(int qAtoms, int tAtoms, int qBonds, int tBonds) {
    qNumAtoms = qAtoms;
    qNumBonds = qBonds;
    const int atomWordsPerRow = (tAtoms + 31) / 32;
    const int bondWordsPerRow = (tBonds + 31) / 32;
    EXPECT_EQ(cudaMallocManaged(&atomData,
                                qAtoms * atomWordsPerRow * sizeof(std::uint32_t)),
              cudaSuccess);
    EXPECT_EQ(cudaMallocManaged(&bondData,
                                qBonds * bondWordsPerRow * sizeof(std::uint32_t)),
              cudaSuccess);
    std::memset(atomData, 0, qAtoms * atomWordsPerRow * sizeof(std::uint32_t));
    std::memset(bondData, 0, qBonds * bondWordsPerRow * sizeof(std::uint32_t));
    device.atoms = MatchTableDevice{atomData, qAtoms, tAtoms, atomWordsPerRow};
    device.bonds = MatchTableDevice{bondData, qBonds, tBonds, bondWordsPerRow};
  }

  void setAtomBit(int qAtom, int tAtom) {
    atomData[qAtom * device.atoms.wordsPerRow + tAtom / 32] |=
        (1u << (tAtom % 32));
  }
  void setBondBit(int qBond, int tBond) {
    bondData[qBond * device.bonds.wordsPerRow + tBond / 32] |=
        (1u << (tBond % 32));
  }
  void setAllAtomBits() {
    for (int q = 0; q < device.atoms.nRows; ++q)
      for (int t = 0; t < device.atoms.nCols; ++t)
        setAtomBit(q, t);
  }
  void setAllBondBits() {
    for (int q = 0; q < device.bonds.nRows; ++q)
      for (int t = 0; t < device.bonds.nCols; ++t)
        setBondBit(q, t);
  }

  ~ManagedMatchTables() {
    if (atomData) cudaFree(atomData);
    if (bondData) cudaFree(bondData);
  }
};

std::uint32_t* allocBondEndpointsManaged(
    const std::vector<std::pair<int, int>>& edges) {
  std::uint32_t* p = nullptr;
  EXPECT_EQ(cudaMallocManaged(&p, edges.size() * sizeof(std::uint32_t)),
            cudaSuccess);
  for (size_t i = 0; i < edges.size(); ++i) {
    p[i] = (static_cast<std::uint32_t>(edges[i].first)  << 16) |
            static_cast<std::uint32_t>(edges[i].second);
  }
  return p;
}

// ---- matchSingleBondWithinThread ----

struct SingleBondTestOut {
  bool ok;
  SingleBondMatch match;
};

__global__ void matchSingleBondDriver(
    int qBondIdx, int tBondIdx, bool reversed,
    const std::uint32_t* qBondEndpoints, int qNumAtoms, int qNumBonds,
    const std::uint32_t* tBondEndpoints, int tNumAtoms, int tNumBonds,
    PairMatchTablesDevice tables,
    SingleBondTestOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  TestCsrView qView{qBondEndpoints, qNumAtoms, qNumBonds};
  TestCsrView tView{tBondEndpoints, tNumAtoms, tNumBonds};
  SingleBondMatch sm{};
  out->ok = mcs::fmcs::matchSingleBondWithinThread(
      qBondIdx, tBondIdx, reversed, qView, tView, tables, sm);
  out->match = sm;
}

}  // namespace

TEST(FMCSUnit, MatchSingleBondForwardOrientation) {
  // Query bond (0,1), target bond (0,1).  All atoms / bonds compatible.
  std::uint32_t* qBE = allocBondEndpointsManaged({{0, 1}});
  std::uint32_t* tBE = allocBondEndpointsManaged({{0, 1}});
  ManagedMatchTables tables;
  tables.allocate(2, 2, 1, 1);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  SingleBondTestOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(SingleBondTestOut)), cudaSuccess);
  matchSingleBondDriver<<<1, 1>>>(0, 0, /*reversed=*/false, qBE, 2, 1, tBE, 2, 1,
                                  tables.device, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(out->ok);
  EXPECT_EQ(out->match.targetAtomU, 0u);  // qU=0 -> tU=0
  EXPECT_EQ(out->match.targetAtomV, 1u);  // qV=1 -> tV=1

  cudaFree(out);
  cudaFree(qBE);
  cudaFree(tBE);
}

TEST(FMCSUnit, MatchSingleBondReverseOrientation) {
  std::uint32_t* qBE = allocBondEndpointsManaged({{0, 1}});
  std::uint32_t* tBE = allocBondEndpointsManaged({{0, 1}});
  ManagedMatchTables tables;
  tables.allocate(2, 2, 1, 1);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  SingleBondTestOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(SingleBondTestOut)), cudaSuccess);
  matchSingleBondDriver<<<1, 1>>>(0, 0, /*reversed=*/true, qBE, 2, 1, tBE, 2, 1,
                                  tables.device, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(out->ok);
  EXPECT_EQ(out->match.targetAtomU, 1u);  // qU=0 -> tV=1 (reversed)
  EXPECT_EQ(out->match.targetAtomV, 0u);  // qV=1 -> tU=0

  cudaFree(out);
  cudaFree(qBE);
  cudaFree(tBE);
}

TEST(FMCSUnit, MatchSingleBondBondTableRejection) {
  std::uint32_t* qBE = allocBondEndpointsManaged({{0, 1}});
  std::uint32_t* tBE = allocBondEndpointsManaged({{0, 1}});
  ManagedMatchTables tables;
  tables.allocate(2, 2, 1, 1);
  tables.setAllAtomBits();
  // Deliberately leave bondData all zero -> bond-table reject.

  SingleBondTestOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(SingleBondTestOut)), cudaSuccess);
  matchSingleBondDriver<<<1, 1>>>(0, 0, /*reversed=*/false, qBE, 2, 1, tBE, 2, 1,
                                  tables.device, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_FALSE(out->ok);

  cudaFree(out);
  cudaFree(qBE);
  cudaFree(tBE);
}

TEST(FMCSUnit, MatchSingleBondAtomTableRejection) {
  std::uint32_t* qBE = allocBondEndpointsManaged({{0, 1}});
  std::uint32_t* tBE = allocBondEndpointsManaged({{0, 1}});
  ManagedMatchTables tables;
  tables.allocate(2, 2, 1, 1);
  tables.setAllBondBits();
  // Atom 0 compatible with target atom 0, but atom 1 is incompatible
  // with target atom 1 -> forward orientation rejects on second atom.
  tables.setAtomBit(0, 0);
  tables.setAtomBit(0, 1);
  tables.setAtomBit(1, 0);
  // Note: (1, 1) deliberately left unset.

  SingleBondTestOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(SingleBondTestOut)), cudaSuccess);
  matchSingleBondDriver<<<1, 1>>>(0, 0, /*reversed=*/false, qBE, 2, 1, tBE, 2, 1,
                                  tables.device, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_FALSE(out->ok);

  cudaFree(out);
  cudaFree(qBE);
  cudaFree(tBE);
}

// ---- matchIncrementalFastCooperative ----

namespace {

// Helper: build a parent MatchResult that records the mapping
// {qAtom[i] -> tAtom[i]} and {qBond[i] -> tBond[i]} from caller-supplied
// parallel arrays.  Used by the incremental tests to set up "what the
// parent already had matched" before adding new bonds.
template<int maxA, int maxB, int maxTA, int maxTB>
__device__ __forceinline__ void buildParentMatch(
    mcs::fmcs::MatchResult<maxA, maxB, maxTA, maxTB>& match,
    const int* qAtoms, const int* tAtoms, int nAtomMaps,
    const int* qBonds, const int* tBonds, int nBondMaps) {
  using MatchT = mcs::fmcs::MatchResult<maxA, maxB, maxTA, maxTB>;
  mcs::fmcs::matchResultClearWithinThread(match);
  for (int i = 0; i < nAtomMaps; ++i) {
    match.targetAtomIdx[qAtoms[i]] = static_cast<std::uint8_t>(tAtoms[i]);
    const int t = tAtoms[i];
    match.visitedTargetAtoms[t / MatchT::kTargetAtomBitsPerWord] |=
        (typename MatchT::target_atom_word{1}) << (t % MatchT::kTargetAtomBitsPerWord);
  }
  for (int i = 0; i < nBondMaps; ++i) {
    match.targetBondIdx[qBonds[i]] = static_cast<std::uint8_t>(tBonds[i]);
    const int t = tBonds[i];
    match.visitedTargetBonds[t / MatchT::kTargetBondBitsPerWord] |=
        (typename MatchT::target_bond_word{1}) << (t % MatchT::kTargetBondBitsPerWord);
  }
  match.matchedAtomSize = static_cast<std::uint16_t>(nAtomMaps);
  match.matchedBondSize = static_cast<std::uint16_t>(nBondMaps);
  match.empty = (nAtomMaps == 0 && nBondMaps == 0);
}

}  // namespace

namespace mcs_fmcs_incremental_test {

using QueuedT16 = mcs::fmcs::QueuedSeed<16, 16, 16, 16>;

struct IncrementalTestOut {
  bool ok;
  QueuedT16 child;
};

// One-warp driver: builds parent match in shared mem, then constructs
// the child seed (parent + new bonds) and runs
// matchIncrementalFastCooperative on it.
__global__ void matchIncrementalAtomAddingDriver(
    const std::uint32_t* qBE, int qNumAtoms, int qNumBonds,
    const std::uint32_t* tBE, int tNumAtoms, int tNumBonds,
    PairMatchTablesDevice tables,
    IncrementalTestOut* out) {
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

  auto block = cooperative_groups::this_thread_block();
  auto warp = cooperative_groups::tiled_partition<32>(block);
  TestCsrView qView{qBE, qNumAtoms, qNumBonds};
  TestCsrView tView{tBE, tNumAtoms, tNumBonds};
  bool ok = mcs::fmcs::matchIncrementalFastCooperative(
      warp, child.seed, qView, tView, tables, child.match);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok = ok;
    out->child = child;
  }
}

__global__ void matchIncrementalRingClosingDriver(
    const std::uint32_t* qBE, int qNumAtoms, int qNumBonds,
    const std::uint32_t* tBE, int tNumAtoms, int tNumBonds,
    PairMatchTablesDevice tables,
    IncrementalTestOut* out) {
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
    for (int b : {0, 1, 2, 3}) mcs::fmcs::seedAddBondWithinThread(child.seed, b);
    for (int a : {0, 1, 2, 3}) mcs::fmcs::seedAddAtomWithinThread(child.seed, a);
  }
  __syncthreads();

  auto block = cooperative_groups::this_thread_block();
  auto warp = cooperative_groups::tiled_partition<32>(block);
  TestCsrView qView{qBE, qNumAtoms, qNumBonds};
  TestCsrView tView{tBE, tNumAtoms, tNumBonds};
  bool ok = mcs::fmcs::matchIncrementalFastCooperative(
      warp, child.seed, qView, tView, tables, child.match);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok = ok;
    out->child = child;
  }
}

__global__ void matchIncrementalVisitedConflictDriver(
    const std::uint32_t* qBE, int qNumAtoms, int qNumBonds,
    const std::uint32_t* tBE, int tNumAtoms, int tNumBonds,
    PairMatchTablesDevice tables,
    IncrementalTestOut* out) {
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

  auto block = cooperative_groups::this_thread_block();
  auto warp = cooperative_groups::tiled_partition<32>(block);
  TestCsrView qView{qBE, qNumAtoms, qNumBonds};
  TestCsrView tView{tBE, tNumAtoms, tNumBonds};
  bool ok = mcs::fmcs::matchIncrementalFastCooperative(
      warp, child.seed, qView, tView, tables, child.match);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok = ok;
    out->child = child;
  }
}

__global__ void matchIncrementalTwoBondChainDriver(
    const std::uint32_t* qBE, int qNumAtoms, int qNumBonds,
    const std::uint32_t* tBE, int tNumAtoms, int tNumBonds,
    PairMatchTablesDevice tables,
    IncrementalTestOut* out) {
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

  auto block = cooperative_groups::this_thread_block();
  auto warp = cooperative_groups::tiled_partition<32>(block);
  TestCsrView qView{qBE, qNumAtoms, qNumBonds};
  TestCsrView tView{tBE, tNumAtoms, tNumBonds};
  bool ok = mcs::fmcs::matchIncrementalFastCooperative(
      warp, child.seed, qView, tView, tables, child.match);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok = ok;
    out->child = child;
  }
}

}  // namespace mcs_fmcs_incremental_test

TEST(FMCSUnit, MatchIncrementalFastAtomAdding) {
  using mcs_fmcs_incremental_test::IncrementalTestOut;
  using mcs_fmcs_incremental_test::matchIncrementalAtomAddingDriver;

  // Query/target are both a 3-atom path 0-1-2 with bonds (0,1), (1,2).
  std::uint32_t* qBE = allocBondEndpointsManaged({{0, 1}, {1, 2}});
  std::uint32_t* tBE = allocBondEndpointsManaged({{0, 1}, {1, 2}});
  ManagedMatchTables tables;
  tables.allocate(3, 3, 2, 2);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  IncrementalTestOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(IncrementalTestOut)), cudaSuccess);
  matchIncrementalAtomAddingDriver<<<1, 32>>>(qBE, 3, 2, tBE, 3, 2,
                                              tables.device, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(out->ok);
  EXPECT_EQ(out->child.match.targetBondIdx[1], 1u);
  EXPECT_EQ(out->child.match.targetAtomIdx[2], 2u);
  EXPECT_EQ(out->child.match.matchedBondSize, 2);
  EXPECT_EQ(out->child.match.matchedAtomSize, 3);

  cudaFree(out);
  cudaFree(qBE);
  cudaFree(tBE);
}

TEST(FMCSUnit, MatchIncrementalFastRingClosing) {
  using mcs_fmcs_incremental_test::IncrementalTestOut;
  using mcs_fmcs_incremental_test::matchIncrementalRingClosingDriver;

  // 4-atom square with one diagonal-free closure.  Bonds: (0,1) (1,2)
  // (2,3) (0,3).
  std::uint32_t* qBE = allocBondEndpointsManaged({{0,1}, {1,2}, {2,3}, {0,3}});
  std::uint32_t* tBE = allocBondEndpointsManaged({{0,1}, {1,2}, {2,3}, {0,3}});
  ManagedMatchTables tables;
  tables.allocate(4, 4, 4, 4);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  IncrementalTestOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(IncrementalTestOut)), cudaSuccess);
  matchIncrementalRingClosingDriver<<<1, 32>>>(qBE, 4, 4, tBE, 4, 4,
                                               tables.device, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(out->ok);
  EXPECT_EQ(out->child.match.targetBondIdx[3], 3u);
  EXPECT_EQ(out->child.match.matchedBondSize, 4);
  // No new atoms added by the ring-closing bond.
  EXPECT_EQ(out->child.match.matchedAtomSize, 4);

  cudaFree(out);
  cudaFree(qBE);
  cudaFree(tBE);
}

TEST(FMCSUnit, MatchIncrementalFastVisitedConflictFails) {
  using mcs_fmcs_incremental_test::IncrementalTestOut;
  using mcs_fmcs_incremental_test::matchIncrementalVisitedConflictDriver;

  // Query: 3 atoms / 2 bonds.  Target: 2 atoms / 1 bond.  Parent has
  // bond (0,1) mapped; trying to extend with bond (0,2) forces atom 2
  // onto target atom 1, which is already visited.
  std::uint32_t* qBE = allocBondEndpointsManaged({{0,1}, {0,2}});
  std::uint32_t* tBE = allocBondEndpointsManaged({{0,1}});
  ManagedMatchTables tables;
  tables.allocate(3, 2, 2, 1);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  IncrementalTestOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(IncrementalTestOut)), cudaSuccess);
  matchIncrementalVisitedConflictDriver<<<1, 32>>>(qBE, 3, 2, tBE, 2, 1,
                                                   tables.device, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_FALSE(out->ok);

  cudaFree(out);
  cudaFree(qBE);
  cudaFree(tBE);
}

TEST(FMCSUnit, MatchIncrementalFastTwoBondChain) {
  using mcs_fmcs_incremental_test::IncrementalTestOut;
  using mcs_fmcs_incremental_test::matchIncrementalTwoBondChainDriver;

  // Both sides are the 4-atom path 0-1-2-3 with bonds 0=(0,1), 1=(1,2),
  // 2=(2,3).
  std::uint32_t* qBE = allocBondEndpointsManaged({{0,1}, {1,2}, {2,3}});
  std::uint32_t* tBE = allocBondEndpointsManaged({{0,1}, {1,2}, {2,3}});
  ManagedMatchTables tables;
  tables.allocate(4, 4, 3, 3);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  IncrementalTestOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(IncrementalTestOut)), cudaSuccess);
  matchIncrementalTwoBondChainDriver<<<1, 32>>>(qBE, 4, 3, tBE, 4, 3,
                                                tables.device, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(out->ok);
  EXPECT_EQ(out->child.match.targetBondIdx[1], 1u);
  EXPECT_EQ(out->child.match.targetBondIdx[2], 2u);
  EXPECT_EQ(out->child.match.targetAtomIdx[2], 2u);
  EXPECT_EQ(out->child.match.targetAtomIdx[3], 3u);
  EXPECT_EQ(out->child.match.matchedBondSize, 3);
  EXPECT_EQ(out->child.match.matchedAtomSize, 4);

  cudaFree(out);
  cudaFree(qBE);
  cudaFree(tBE);
}

// ---- full seed substructure fallback ----

namespace mcs_fmcs_substructure_test {

using QueuedT16 = mcs::fmcs::QueuedSeed<16, 16, 16, 16>;

struct SubstructureTestOut {
  bool ok;
  bool overflowed;
  QueuedT16 child;
};

constexpr int kTestSubstructurePartialCapacity = 64;

std::uint8_t* allocSubstructurePartialsManaged() {
  std::uint8_t* p = nullptr;
  EXPECT_EQ(cudaMallocManaged(
                &p, 2 * kTestSubstructurePartialCapacity * 16 *
                        sizeof(std::uint8_t)),
            cudaSuccess);
  return p;
}

__device__ __forceinline__ void addMaskSeed(QueuedT16& child,
                                            std::uint32_t atomMask,
                                            std::uint32_t bondMask) {
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

__global__ void matchSubstructureMaskDriver(
    const std::uint32_t* qBE, int qNumAtoms, int qNumBonds,
    const std::uint32_t* tBE, int tNumAtoms, int tNumBonds,
    PairMatchTablesDevice tables,
    std::uint32_t atomMask,
    std::uint32_t bondMask,
    std::uint8_t* partialStorage,
    int partialCapacity,
    SubstructureTestOut* out) {
  __shared__ QueuedT16 child;
  __shared__ mcs::fmcs::FmcsSubstructureScratch<16, 16> scratch;
  if (threadIdx.x == 0) {
    addMaskSeed(child, atomMask, bondMask);
  }
  __syncthreads();

  auto block = cooperative_groups::this_thread_block();
  auto warp = cooperative_groups::tiled_partition<32>(block);
  TestCsrView qView{qBE, qNumAtoms, qNumBonds};
  TestCsrView tView{tBE, tNumAtoms, tNumBonds};
  bool overflowed = false;
  bool ok = mcs::fmcs::matchSeedSubstructureCooperative(
      warp, child.seed, qView, tView, tables, child.match, scratch,
      partialStorage, partialCapacity, &overflowed);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok = ok;
    out->overflowed = overflowed;
    out->child = child;
  }
}

__global__ void matchFallbackBadParentDriver(
    const std::uint32_t* qBE, int qNumAtoms, int qNumBonds,
    const std::uint32_t* tBE, int tNumAtoms, int tNumBonds,
    PairMatchTablesDevice tables,
    std::uint8_t* partialStorage,
    int partialCapacity,
    SubstructureTestOut* out) {
  __shared__ QueuedT16 child;
  __shared__ mcs::fmcs::FmcsSubstructureScratch<16, 16> scratch;
  __shared__ int scratchLock;
  if (threadIdx.x == 0) {
    // Query seed is the 4-edge path inside the triangle-with-leaves
    // repro: query bonds 1,2,3,4 and all five atoms.  The stored parent
    // match maps q bond 1 = (0,2) onto target bond 0 = (0,3), which is
    // locally valid but blocks q bond 2 = (0,4) in the fast extender.
    addMaskSeed(child, /*atomMask=*/0x1Fu, /*bondMask=*/0x1Eu);
    mcs::fmcs::matchResultClearWithinThread(child.match);
    using MatchT = decltype(child.match);
    child.match.targetAtomIdx[0] = 0;
    child.match.targetAtomIdx[2] = 3;
    child.match.visitedTargetAtoms[0 / MatchT::kTargetAtomBitsPerWord] |=
        typename MatchT::target_atom_word{1}
        << (0 % MatchT::kTargetAtomBitsPerWord);
    child.match.visitedTargetAtoms[3 / MatchT::kTargetAtomBitsPerWord] |=
        typename MatchT::target_atom_word{1}
        << (3 % MatchT::kTargetAtomBitsPerWord);
    child.match.targetBondIdx[1] = 0;
    child.match.visitedTargetBonds[0 / MatchT::kTargetBondBitsPerWord] |=
        typename MatchT::target_bond_word{1}
        << (0 % MatchT::kTargetBondBitsPerWord);
    child.match.matchedAtomSize = 2;
    child.match.matchedBondSize = 1;
    child.match.empty = false;
    scratchLock = 0;
  }
  __syncthreads();

  auto block = cooperative_groups::this_thread_block();
  auto warp = cooperative_groups::tiled_partition<32>(block);
  TestCsrView qView{qBE, qNumAtoms, qNumBonds};
  TestCsrView tView{tBE, tNumAtoms, tNumBonds};
  bool overflowed = false;
  bool ok = mcs::fmcs::matchSeedWithSubstructureFallbackCooperative(
      warp, child.seed, qView, tView, tables, child.match, scratch,
      &scratchLock, partialStorage, partialCapacity, &overflowed);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok = ok;
    out->overflowed = overflowed;
    out->child = child;
  }
}

}  // namespace mcs_fmcs_substructure_test

TEST(FMCSUnit, MatchSeedSubstructurePath) {
  using namespace mcs_fmcs_substructure_test;

  std::uint32_t* qBE =
      allocBondEndpointsManaged({{0, 1}, {1, 2}, {2, 3}});
  std::uint32_t* tBE =
      allocBondEndpointsManaged({{0, 1}, {1, 2}, {2, 3}, {3, 4}});
  ManagedMatchTables tables;
  tables.allocate(4, 5, 3, 4);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  SubstructureTestOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(SubstructureTestOut)), cudaSuccess);
  std::uint8_t* partials = allocSubstructurePartialsManaged();
  matchSubstructureMaskDriver<<<1, 32>>>(
      qBE, 4, 3, tBE, 5, 4, tables.device,
      /*atomMask=*/0xFu, /*bondMask=*/0x7u,
      partials, kTestSubstructurePartialCapacity, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(out->ok);
  EXPECT_FALSE(out->overflowed);
  EXPECT_EQ(out->child.match.matchedAtomSize, 4);
  EXPECT_EQ(out->child.match.matchedBondSize, 3);
  for (int q = 0; q < 4; ++q) {
    EXPECT_NE(out->child.match.targetAtomIdx[q], mcs::fmcs::kUnmappedTargetIdx);
  }
  for (int q = 0; q < 3; ++q) {
    EXPECT_NE(out->child.match.targetBondIdx[q], mcs::fmcs::kUnmappedTargetIdx);
  }

  cudaFree(out);
  cudaFree(partials);
  cudaFree(qBE);
  cudaFree(tBE);
}

TEST(FMCSUnit, MatchSeedSubstructureRejectsNoMatch) {
  using namespace mcs_fmcs_substructure_test;

  std::uint32_t* qBE =
      allocBondEndpointsManaged({{0, 1}, {1, 2}, {0, 2}});
  std::uint32_t* tBE =
      allocBondEndpointsManaged({{0, 1}, {1, 2}});
  ManagedMatchTables tables;
  tables.allocate(3, 3, 3, 2);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  SubstructureTestOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(SubstructureTestOut)), cudaSuccess);
  std::uint8_t* partials = allocSubstructurePartialsManaged();
  matchSubstructureMaskDriver<<<1, 32>>>(
      qBE, 3, 3, tBE, 3, 2, tables.device,
      /*atomMask=*/0x7u, /*bondMask=*/0x7u,
      partials, kTestSubstructurePartialCapacity, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_FALSE(out->ok);
  EXPECT_FALSE(out->overflowed);
  EXPECT_TRUE(out->child.match.empty);

  cudaFree(out);
  cudaFree(partials);
  cudaFree(qBE);
  cudaFree(tBE);
}

TEST(FMCSUnit, MatchSeedSubstructureRespectsAtomTable) {
  using namespace mcs_fmcs_substructure_test;

  std::uint32_t* qBE =
      allocBondEndpointsManaged({{0, 1}, {1, 2}});
  std::uint32_t* tBE =
      allocBondEndpointsManaged({{0, 1}, {1, 2}});
  ManagedMatchTables tables;
  tables.allocate(3, 3, 2, 2);
  tables.setAtomBit(0, 0);
  tables.setAtomBit(1, 1);
  tables.setAtomBit(2, 2);
  tables.setAllBondBits();

  SubstructureTestOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(SubstructureTestOut)), cudaSuccess);
  std::uint8_t* partials = allocSubstructurePartialsManaged();
  matchSubstructureMaskDriver<<<1, 32>>>(
      qBE, 3, 2, tBE, 3, 2, tables.device,
      /*atomMask=*/0x7u, /*bondMask=*/0x3u,
      partials, kTestSubstructurePartialCapacity, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(out->ok);
  EXPECT_FALSE(out->overflowed);
  EXPECT_EQ(out->child.match.targetAtomIdx[0], 0u);
  EXPECT_EQ(out->child.match.targetAtomIdx[1], 1u);
  EXPECT_EQ(out->child.match.targetAtomIdx[2], 2u);
  EXPECT_EQ(out->child.match.matchedAtomSize, 3);
  EXPECT_EQ(out->child.match.matchedBondSize, 2);

  cudaFree(out);
  cudaFree(partials);
  cudaFree(qBE);
  cudaFree(tBE);
}

TEST(FMCSUnit, MatchSeedSubstructureRespectsBondTable) {
  using namespace mcs_fmcs_substructure_test;

  std::uint32_t* qBE =
      allocBondEndpointsManaged({{0, 1}, {1, 2}});
  std::uint32_t* tBE =
      allocBondEndpointsManaged({{0, 1}, {1, 2}, {0, 2}});
  ManagedMatchTables tables;
  tables.allocate(3, 3, 2, 3);
  tables.setAllAtomBits();
  tables.setBondBit(0, 0);
  tables.setBondBit(1, 1);

  SubstructureTestOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(SubstructureTestOut)), cudaSuccess);
  std::uint8_t* partials = allocSubstructurePartialsManaged();
  matchSubstructureMaskDriver<<<1, 32>>>(
      qBE, 3, 2, tBE, 3, 3, tables.device,
      /*atomMask=*/0x7u, /*bondMask=*/0x3u,
      partials, kTestSubstructurePartialCapacity, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(out->ok);
  EXPECT_FALSE(out->overflowed);
  EXPECT_EQ(out->child.match.targetBondIdx[0], 0u);
  EXPECT_EQ(out->child.match.targetBondIdx[1], 1u);
  EXPECT_EQ(out->child.match.matchedAtomSize, 3);
  EXPECT_EQ(out->child.match.matchedBondSize, 2);

  cudaFree(out);
  cudaFree(partials);
  cudaFree(qBE);
  cudaFree(tBE);
}

TEST(FMCSUnit, MatchSeedSubstructureFindsPathInsideTriangleWithLeaves) {
  using namespace mcs_fmcs_substructure_test;

  std::uint32_t* qBE =
      allocBondEndpointsManaged({{0, 1}, {0, 2}, {0, 4}, {1, 2}, {1, 3}});
  std::uint32_t* tBE =
      allocBondEndpointsManaged({{0, 3}, {1, 2}, {1, 4}, {2, 3}});
  ManagedMatchTables tables;
  tables.allocate(5, 5, 5, 4);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  SubstructureTestOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(SubstructureTestOut)), cudaSuccess);
  std::uint8_t* partials = allocSubstructurePartialsManaged();
  matchSubstructureMaskDriver<<<1, 32>>>(
      qBE, 5, 5, tBE, 5, 4, tables.device,
      /*atomMask=*/0x1Fu, /*bondMask=*/0x1Eu,
      partials, kTestSubstructurePartialCapacity, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(out->ok);
  EXPECT_FALSE(out->overflowed);
  EXPECT_EQ(out->child.match.matchedAtomSize, 5);
  EXPECT_EQ(out->child.match.matchedBondSize, 4);
  EXPECT_EQ(out->child.match.targetBondIdx[0], mcs::fmcs::kUnmappedTargetIdx);
  for (int q : {1, 2, 3, 4}) {
    EXPECT_NE(out->child.match.targetBondIdx[q], mcs::fmcs::kUnmappedTargetIdx);
  }

  cudaFree(out);
  cudaFree(partials);
  cudaFree(qBE);
  cudaFree(tBE);
}

TEST(FMCSUnit, MatchSeedFallbackRebuildsAfterGreedyFailure) {
  using namespace mcs_fmcs_substructure_test;

  std::uint32_t* qBE =
      allocBondEndpointsManaged({{0, 1}, {0, 2}, {0, 4}, {1, 2}, {1, 3}});
  std::uint32_t* tBE =
      allocBondEndpointsManaged({{0, 3}, {1, 2}, {1, 4}, {2, 3}});
  ManagedMatchTables tables;
  tables.allocate(5, 5, 5, 4);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  SubstructureTestOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(SubstructureTestOut)), cudaSuccess);
  std::uint8_t* partials = allocSubstructurePartialsManaged();
  matchFallbackBadParentDriver<<<1, 32>>>(
      qBE, 5, 5, tBE, 5, 4, tables.device,
      partials, kTestSubstructurePartialCapacity, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(out->ok);
  EXPECT_FALSE(out->overflowed);
  EXPECT_EQ(out->child.match.matchedAtomSize, 5);
  EXPECT_EQ(out->child.match.matchedBondSize, 4);
  for (int q = 0; q < 5; ++q) {
    EXPECT_NE(out->child.match.targetAtomIdx[q], mcs::fmcs::kUnmappedTargetIdx);
  }
  for (int q : {1, 2, 3, 4}) {
    EXPECT_NE(out->child.match.targetBondIdx[q], mcs::fmcs::kUnmappedTargetIdx);
  }

  cudaFree(out);
  cudaFree(partials);
  cudaFree(qBE);
  cudaFree(tBE);
}

// ---------------------------------------------------------------------------
// fillNewBondsCooperative / pruneIndividualBondsCooperative
// ---------------------------------------------------------------------------

namespace mcs_fmcs_grow_test {

using mcs::fmcs::NewBond;
using SeedT = mcs::fmcs::Seed<16, 16>;
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
template<class SeedSetup>
__device__ __forceinline__ void fillNewBondsRun(
    SeedSetup&& setup,
    const std::uint32_t* qBondEndpoints, int qNumAtoms, int qNumBonds,
    int maxNewBonds,
    FillNewBondsOut* out) {
  __shared__ SeedT seed;
  __shared__ NewBond bonds[kMaxNewBonds];
  __shared__ int     count;

  if (threadIdx.x == 0) {
    mcs::fmcs::seedClearWithinThread(seed);
    setup(seed);
  }
  __syncthreads();

  auto block = cooperative_groups::this_thread_block();
  auto warp  = cooperative_groups::tiled_partition<32>(block);
  TestCsrView qView{qBondEndpoints, qNumAtoms, qNumBonds};
  bool ok = mcs::fmcs::fillNewBondsCooperative(warp, seed, qView,
                                               bonds, &count, maxNewBonds);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok    = ok;
    out->count = count;
    for (int i = 0; i < kMaxNewBonds; ++i) out->bonds[i] = bonds[i];
  }
}

// Test 1: 3-atom path query (atoms 0-1-2, bonds (0,1), (1,2)).  Seed
// holds just atom 1, marked last-added.  Both bonds touch atom 1;
// each should become an atom-adding NewBond.
__global__ void fillNewBondsAtomAddingDriver(
    const std::uint32_t* qBE, int qNumAtoms, int qNumBonds,
    FillNewBondsOut* out) {
  fillNewBondsRun(
      [] __device__ (SeedT& seed) {
        mcs::fmcs::seedAddAtomWithinThread(seed, 1);
      },
      qBE, qNumAtoms, qNumBonds, kMaxNewBonds, out);
}

// Test 2: same query, seed has bond 0 in excludedBonds.  Only bond 1
// should appear in the output.
__global__ void fillNewBondsExcludedSkippedDriver(
    const std::uint32_t* qBE, int qNumAtoms, int qNumBonds,
    FillNewBondsOut* out) {
  fillNewBondsRun(
      [] __device__ (SeedT& seed) {
        mcs::fmcs::seedAddAtomWithinThread(seed, 1);
        // Mark bond 0 as excluded but DO NOT add it to seed.bonds.
        // The exclusion alone is what fillNewBonds checks.
        using BondWord = SeedT::bond_word_type;
        constexpr int kBPW = SeedT::kBondBitsPerWord;
        seed.excludedBonds[0 / kBPW] |=
            static_cast<BondWord>(1) << (0 % kBPW);
      },
      qBE, qNumAtoms, qNumBonds, kMaxNewBonds, out);
}

// Test 3: 3-atom triangle query (atoms 0,1,2; bonds (0,1), (1,2), (0,2)).
// Seed holds atoms 0 and 1 (both last-added) plus bond (0,1).  Bond 2
// = (0,2) is atom-adding (atom 2 not in seed).  More importantly, this
// test verifies the seed.atoms vs seed.lastAddedAtoms split: if we
// only mark atom 1 as "last added" (atom 0 stays in seed.atoms but
// NOT in lastAddedAtoms), bond (0,2) should NOT be reported because
// neither endpoint is "newly added".  Bond (1,2) SHOULD be reported.
__global__ void fillNewBondsLastAddedFilteringDriver(
    const std::uint32_t* qBE, int qNumAtoms, int qNumBonds,
    FillNewBondsOut* out) {
  fillNewBondsRun(
      [] __device__ (SeedT& seed) {
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
      qBE, qNumAtoms, qNumBonds, kMaxNewBonds, out);
}

// Test 4: ring-closing.  Seed has atoms 0, 1, 2 all marked last-added,
// bond 0 = (0,1) and bond 1 = (1,2) already in excludedBonds.  Bond 2
// = (0,2) closes the ring: both endpoints in seed.
__global__ void fillNewBondsRingClosingDriver(
    const std::uint32_t* qBE, int qNumAtoms, int qNumBonds,
    FillNewBondsOut* out) {
  fillNewBondsRun(
      [] __device__ (SeedT& seed) {
        mcs::fmcs::seedAddAtomWithinThread(seed, 0);
        mcs::fmcs::seedAddAtomWithinThread(seed, 1);
        mcs::fmcs::seedAddAtomWithinThread(seed, 2);
        mcs::fmcs::seedAddBondWithinThread(seed, 0);
        mcs::fmcs::seedAddBondWithinThread(seed, 1);
        // Bond 2 = (0,2) is the only candidate; it's ring-closing
        // since both 0 and 2 are in seed.atoms.
      },
      qBE, qNumAtoms, qNumBonds, kMaxNewBonds, out);
}

// Test 5: overflow.  4-atom path with seed = {atom 0 newly-added}.
// Query bonds 0..3 all touch atom 0 (star graph: bonds (0,1) (0,2)
// (0,3) (0,4)).  maxNewBonds = 2 -> first 2 win the race, function
// returns false, count clamped at 2.
__global__ void fillNewBondsOverflowDriver(
    const std::uint32_t* qBE, int qNumAtoms, int qNumBonds,
    FillNewBondsOut* out) {
  fillNewBondsRun(
      [] __device__ (SeedT& seed) {
        mcs::fmcs::seedAddAtomWithinThread(seed, 0);
      },
      qBE, qNumAtoms, qNumBonds, /*maxNewBonds=*/2, out);
}

}  // namespace mcs_fmcs_grow_test

TEST(FMCSUnit, FillNewBondsAtomAddingFromBoundary) {
  using namespace mcs_fmcs_grow_test;
  // Path 0-1-2: bonds (0,1), (1,2).
  std::uint32_t* qBE = allocBondEndpointsManaged({{0, 1}, {1, 2}});
  FillNewBondsOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(FillNewBondsOut)), cudaSuccess);

  fillNewBondsAtomAddingDriver<<<1, 32>>>(qBE, 3, 2, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(out->ok);
  EXPECT_EQ(out->count, 2);
  // Both bonds appeared; race-order is non-deterministic so collect
  // by bondIdx.
  std::set<int> seenBonds;
  for (int i = 0; i < out->count; ++i) {
    seenBonds.insert(out->bonds[i].bondIdx);
    EXPECT_EQ(out->bonds[i].endAtomSeedIdx, NewBond::kNotInSeed);
    EXPECT_TRUE(out->bonds[i].alive);
  }
  EXPECT_TRUE(seenBonds.count(0));
  EXPECT_TRUE(seenBonds.count(1));

  cudaFree(out);
  cudaFree(qBE);
}

TEST(FMCSUnit, FillNewBondsExcludedSkipped) {
  using namespace mcs_fmcs_grow_test;
  std::uint32_t* qBE = allocBondEndpointsManaged({{0, 1}, {1, 2}});
  FillNewBondsOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(FillNewBondsOut)), cudaSuccess);

  fillNewBondsExcludedSkippedDriver<<<1, 32>>>(qBE, 3, 2, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(out->ok);
  EXPECT_EQ(out->count, 1);
  EXPECT_EQ(out->bonds[0].bondIdx, 1);  // bond 0 excluded; only bond 1 left

  cudaFree(out);
  cudaFree(qBE);
}

TEST(FMCSUnit, FillNewBondsLastAddedFiltering) {
  using namespace mcs_fmcs_grow_test;
  // Triangle: bonds 0=(0,1), 1=(1,2), 2=(0,2).
  std::uint32_t* qBE = allocBondEndpointsManaged({{0, 1}, {1, 2}, {0, 2}});
  FillNewBondsOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(FillNewBondsOut)), cudaSuccess);

  fillNewBondsLastAddedFilteringDriver<<<1, 32>>>(qBE, 3, 3, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(out->ok);
  // Bond 0 is excluded.  Bond 1 = (1,2) touches newly-added atom 1.
  // Bond 2 = (0,2) touches atom 0 (NOT newly-added) and atom 2 (also
  // not in seed) -> neither endpoint newly-added, must be skipped.
  EXPECT_EQ(out->count, 1);
  EXPECT_EQ(out->bonds[0].bondIdx, 1);
  EXPECT_EQ(out->bonds[0].endAtomSeedIdx, NewBond::kNotInSeed);
  EXPECT_EQ(out->bonds[0].newAtomIdx, 2);  // atom 2 is the unmapped end

  cudaFree(out);
  cudaFree(qBE);
}

TEST(FMCSUnit, FillNewBondsRingClosing) {
  using namespace mcs_fmcs_grow_test;
  // Triangle: bonds 0=(0,1), 1=(1,2), 2=(0,2).
  std::uint32_t* qBE = allocBondEndpointsManaged({{0, 1}, {1, 2}, {0, 2}});
  FillNewBondsOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(FillNewBondsOut)), cudaSuccess);

  fillNewBondsRingClosingDriver<<<1, 32>>>(qBE, 3, 3, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(out->ok);
  // Only bond 2 = (0,2) is candidate (others are excluded).  Both
  // endpoints in seed -> ring-closing.
  EXPECT_EQ(out->count, 1);
  EXPECT_EQ(out->bonds[0].bondIdx, 2);
  EXPECT_NE(out->bonds[0].endAtomSeedIdx, NewBond::kNotInSeed);

  cudaFree(out);
  cudaFree(qBE);
}

TEST(FMCSUnit, FillNewBondsOverflowReturnsFalse) {
  using namespace mcs_fmcs_grow_test;
  // Star graph: 4 bonds from atom 0 -> 1, 2, 3, 4.  numAtoms=5.
  std::uint32_t* qBE =
      allocBondEndpointsManaged({{0, 1}, {0, 2}, {0, 3}, {0, 4}});
  FillNewBondsOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(FillNewBondsOut)), cudaSuccess);

  fillNewBondsOverflowDriver<<<1, 32>>>(qBE, 5, 4, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_FALSE(out->ok);
  // count is clamped to maxNewBonds (=2).
  EXPECT_EQ(out->count, 2);

  cudaFree(out);
  cudaFree(qBE);
}

// ---- pruneIndividualBondsCooperative ----

namespace mcs_fmcs_prune_test {

using mcs_fmcs_grow_test::QueuedT;
using mcs_fmcs_grow_test::SeedT;
using mcs::fmcs::NewBond;

constexpr int kMaxNewBondsPrune = 8;

struct PruneOut {
  // Identity of children that survived matchFn and reached childSink.
  // We record the child seed's bondIdx-of-the-newly-added bond (the
  // first set bit beyond the parent's bond bitset).  That uniquely
  // identifies which NewBond the child was built from.
  int     survivorBondIdx[kMaxNewBondsPrune];
  int     numSurvivors;
  int     numMatchAttempts;
};

__global__ void pruneStage1Driver(
    const NewBond* hostBonds, int nBonds,
    PruneOut* out) {
  __shared__ std::uint64_t parentStorage[(sizeof(QueuedT) + sizeof(std::uint64_t) - 1) / sizeof(std::uint64_t)];
  __shared__ std::uint64_t workspaceStorage[(sizeof(QueuedT) + sizeof(std::uint64_t) - 1) / sizeof(std::uint64_t)];
  __shared__ NewBond bonds[kMaxNewBondsPrune];
  __shared__ int matchAttempts;
  __shared__ int survivorIdx;
  QueuedT& parent    = *reinterpret_cast<QueuedT*>(parentStorage);
  QueuedT& workspace = *reinterpret_cast<QueuedT*>(workspaceStorage);

  if (threadIdx.x == 0) {
    mcs::fmcs::seedClearWithinThread(parent.seed);
    mcs::fmcs::matchResultClearWithinThread(parent.match);
    // Parent has bond 0 already in its bitset (so pruneIndividualBonds
    // appears as "extending past bond 0" for tracking).
    mcs::fmcs::seedAddAtomWithinThread(parent.seed, 0);
    mcs::fmcs::seedAddAtomWithinThread(parent.seed, 1);
    mcs::fmcs::seedAddBondWithinThread(parent.seed, 0);
    for (int i = 0; i < nBonds; ++i) bonds[i] = hostBonds[i];
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
  auto matchFn = [&] __device__ (QueuedT& child) -> bool {
    if (warp.thread_rank() == 0) ++matchAttempts;
    warp.sync();
    // Find which bondIdx the workspace's seed has beyond the parent's.
    // Parent has only bond 0 set; child has one more.
    int childBond = -1;
    for (int b = 0; b < 16; ++b) {
      using BondWord = SeedT::bond_word_type;
      const BondWord childWord = child.seed.bonds[0];
      const BondWord parentWord = parent.seed.bonds[0];
      const BondWord newBits = childWord & ~parentWord;
      if ((newBits >> b) & 1) { childBond = b; break; }
    }
    return (childBond % 2) == 0;
  };
  auto childSink = [&] __device__ (QueuedT& child) {
    if (warp.thread_rank() == 0) {
      // Record child's new-bond identity.
      using BondWord = SeedT::bond_word_type;
      const BondWord newBits =
          child.seed.bonds[0] & ~parent.seed.bonds[0];
      int childBond = -1;
      for (int b = 0; b < 16; ++b) {
        if ((newBits >> b) & 1) { childBond = b; break; }
      }
      out->survivorBondIdx[survivorIdx++] = childBond;
    }
    warp.sync();
  };
  mcs::fmcs::pruneIndividualBondsCooperative(
      warp, parent, workspace, bonds, nBonds, matchFn, childSink);

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
  NewBond* hostBonds = nullptr;
  ASSERT_EQ(cudaMallocManaged(&hostBonds, sizeof(NewBond) * 4), cudaSuccess);
  hostBonds[0] = NewBond{1, 5, NewBond::kNotInSeed, true, 1};
  hostBonds[1] = NewBond{2, 6, NewBond::kNotInSeed, true, 1};
  hostBonds[2] = NewBond{3, 7, NewBond::kNotInSeed, true, 1};
  hostBonds[3] = NewBond{4, 8, NewBond::kNotInSeed, true, 1};

  PruneOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(PruneOut)), cudaSuccess);
  pruneStage1Driver<<<1, 32>>>(hostBonds, 4, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(out->numMatchAttempts, 4);  // all four bonds were tried
  EXPECT_EQ(out->numSurvivors, 2);
  std::set<int> survivors{out->survivorBondIdx[0], out->survivorBondIdx[1]};
  EXPECT_TRUE(survivors.count(2));
  EXPECT_TRUE(survivors.count(4));
  EXPECT_FALSE(survivors.count(1));
  EXPECT_FALSE(survivors.count(3));

  cudaFree(hostBonds);
  cudaFree(out);
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

__device__ __forceinline__ void buildHashFixture(
    HashSeedT& seed, HashMatchT& match,
    const int* qBonds, const int* tBonds, int n) {
  mcs::fmcs::seedClearWithinThread(seed);
  mcs::fmcs::matchResultClearWithinThread(match);
  for (int i = 0; i < n; ++i) {
    mcs::fmcs::seedAddBondWithinThread(seed, qBonds[i]);
    match.targetBondIdx[qBonds[i]] = static_cast<std::uint8_t>(tBonds[i]);
  }
  match.empty = false;
}

__global__ void hashOrderInvariantDriver(HashOut* outA, HashOut* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  HashSeedT seedA, seedB;
  HashMatchT matchA, matchB;
  // Same set of (q, t) mappings, different addition order.  Hash
  // walks seed.bonds in q-increasing order via __ffs, so the
  // resulting hash should be identical.
  int qsA[] = {3, 1, 5};
  int tsA[] = {7, 2, 9};
  int qsB[] = {1, 5, 3};
  int tsB[] = {2, 9, 7};
  buildHashFixture(seedA, matchA, qsA, tsA, 3);
  buildHashFixture(seedB, matchB, qsB, tsB, 3);
  outA->hash = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

__global__ void hashDistinctMappingsDiffer(HashOut* outA, HashOut* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  HashSeedT seedA, seedB;
  HashMatchT matchA, matchB;
  // Same bonds, different target mappings -> hashes should differ.
  int qsA[] = {0, 1, 2};
  int tsA[] = {0, 1, 2};
  int qsB[] = {0, 1, 2};
  int tsB[] = {2, 1, 0};
  buildHashFixture(seedA, matchA, qsA, tsA, 3);
  buildHashFixture(seedB, matchB, qsB, tsB, 3);
  outA->hash = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

__global__ void hashEmptyIsNonZero(HashOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  HashSeedT seed;
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

__global__ void cacheInsertProbeBasicDriver(
    std::uint64_t* keys, int capacity, CacheInsertProbeOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
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

__global__ void cacheLinearProbeCollisionDriver(
    std::uint64_t* keys, int capacity, CacheCollisionOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
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
  bool finalInsertOk;     // expected: false (table full)
  bool seenFirstAfterFull; // probe still finds the first insert
};

__global__ void cacheFullDropDriver(
    std::uint64_t* keys, int capacity, CacheFullDropOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  __shared__ DeviceMatchCache cache;
  cache.init(keys, capacity);
  int n = 0;
  for (int i = 0; i < capacity; ++i) {
    // Use distinct non-zero keys.  Note: distinctness in the *hash*
    // input space; the cache hashes internally.
    const std::uint64_t key = static_cast<std::uint64_t>(i) + 1;
    if (cache.insertWithinThread(key)) ++n;
  }
  out->successfulInserts = n;
  // Capacity is hit; one more must fail.
  out->finalInsertOk = cache.insertWithinThread(0xFFFFFFFFFFFFFFFFULL);
  // First-key probe still works.
  out->seenFirstAfterFull = cache.probeWithinThread(1ULL);
}

}  // namespace mcs_fmcs_cache_test

TEST(FMCSUnit, MappingHashOrderInvariant) {
  using namespace mcs_fmcs_cache_test;
  HashOut* dA = nullptr; HashOut* dB = nullptr;
  ASSERT_EQ(cudaMallocManaged(&dA, sizeof(HashOut)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&dB, sizeof(HashOut)), cudaSuccess);
  hashOrderInvariantDriver<<<1, 1>>>(dA, dB);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(dA->hash, dB->hash) << "Same (q,t) set must hash identically.";
  EXPECT_NE(dA->hash, 0ULL);

  cudaFree(dA);
  cudaFree(dB);
}

TEST(FMCSUnit, MappingHashDistinctMappingsDiffer) {
  using namespace mcs_fmcs_cache_test;
  HashOut* dA = nullptr; HashOut* dB = nullptr;
  ASSERT_EQ(cudaMallocManaged(&dA, sizeof(HashOut)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&dB, sizeof(HashOut)), cudaSuccess);
  hashDistinctMappingsDiffer<<<1, 1>>>(dA, dB);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_NE(dA->hash, dB->hash)
      << "Different target mappings should not collide on a 64-bit hash.";

  cudaFree(dA);
  cudaFree(dB);
}

TEST(FMCSUnit, MappingHashEmptySeedReturnsNonZero) {
  using namespace mcs_fmcs_cache_test;
  HashOut* d = nullptr;
  ASSERT_EQ(cudaMallocManaged(&d, sizeof(HashOut)), cudaSuccess);
  hashEmptyIsNonZero<<<1, 1>>>(d);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // 0 is reserved as the cache's empty-slot sentinel; mappingHash
  // post-processes a 0 hash to 1.
  EXPECT_EQ(d->hash, 1ULL);

  cudaFree(d);
}

TEST(FMCSUnit, CacheInsertProbeBasic) {
  using namespace mcs_fmcs_cache_test;
  constexpr int kCap = 16;
  std::uint64_t* keys = nullptr;
  ASSERT_EQ(cudaMallocManaged(&keys, sizeof(std::uint64_t) * kCap), cudaSuccess);
  std::memset(keys, 0, sizeof(std::uint64_t) * kCap);
  CacheInsertProbeOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(CacheInsertProbeOut)), cudaSuccess);

  cacheInsertProbeBasicDriver<<<1, 1>>>(keys, kCap, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(out->insertOk);
  EXPECT_TRUE(out->probeFoundInserted);
  EXPECT_FALSE(out->probeMissedOther);

  cudaFree(keys);
  cudaFree(out);
}

TEST(FMCSUnit, CacheLinearProbeCollision) {
  using namespace mcs_fmcs_cache_test;
  constexpr int kCap = 8;
  std::uint64_t* keys = nullptr;
  ASSERT_EQ(cudaMallocManaged(&keys, sizeof(std::uint64_t) * kCap), cudaSuccess);
  std::memset(keys, 0, sizeof(std::uint64_t) * kCap);
  CacheCollisionOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(CacheCollisionOut)), cudaSuccess);

  cacheLinearProbeCollisionDriver<<<1, 1>>>(keys, kCap, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Both inserts succeed regardless of whether they happen to collide
  // on the initial slot -- linear probing finds a different slot
  // when the first is taken.  Both probes find their key.
  EXPECT_TRUE(out->insertOkA);
  EXPECT_TRUE(out->insertOkB);
  EXPECT_TRUE(out->foundA);
  EXPECT_TRUE(out->foundB);

  // Sanity: at least one slot in keys[] equals each inserted key.
  bool seenA = false, seenB = false;
  for (int i = 0; i < kCap; ++i) {
    if (keys[i] == 0xAAAAAAAAAAAAAAAAULL) seenA = true;
    if (keys[i] == 0xBBBBBBBBBBBBBBBBULL) seenB = true;
  }
  EXPECT_TRUE(seenA);
  EXPECT_TRUE(seenB);

  cudaFree(keys);
  cudaFree(out);
}

TEST(FMCSUnit, CacheFullTableDropsAdditionalInsert) {
  using namespace mcs_fmcs_cache_test;
  constexpr int kCap = 16;  // small but power-of-two
  std::uint64_t* keys = nullptr;
  ASSERT_EQ(cudaMallocManaged(&keys, sizeof(std::uint64_t) * kCap), cudaSuccess);
  std::memset(keys, 0, sizeof(std::uint64_t) * kCap);
  CacheFullDropOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(CacheFullDropOut)), cudaSuccess);

  cacheFullDropDriver<<<1, 1>>>(keys, kCap, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(out->successfulInserts, kCap)
      << "Distinct keys should fill all slots.";
  EXPECT_FALSE(out->finalInsertOk)
      << "Insert into a full table must return false.";
  EXPECT_TRUE(out->seenFirstAfterFull)
      << "Existing entries must remain visible after a failed insert.";

  cudaFree(keys);
  cudaFree(out);
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
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  HashSeed128 seedA, seedB;
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
  outA->hash = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

// Single-bond minimal seed.  Should produce a non-zero, deterministic
// hash distinct from the empty-seed sentinel (1).
__global__ void hashSingleBondDriver(HashOut64* outA, HashOut64* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  HashSeed16 seedA, seedB;
  HashMatch16 matchA, matchB;
  mcs::fmcs::seedClearWithinThread(seedA);
  mcs::fmcs::seedClearWithinThread(seedB);
  mcs::fmcs::matchResultClearWithinThread(matchA);
  mcs::fmcs::matchResultClearWithinThread(matchB);
  // Determinism: two seeds with the same single (q=4, t=2) mapping.
  mcs::fmcs::seedAddBondWithinThread(seedA, 4);
  matchA.targetBondIdx[4] = 2;
  matchA.empty = false;
  mcs::fmcs::seedAddBondWithinThread(seedB, 4);
  matchB.targetBondIdx[4] = 2;
  matchB.empty = false;
  outA->hash = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

// Same single query bond and target bond, opposite endpoint orientation.
// These must hash differently or the success cache can prune one Phase-1
// orientation because the other orientation was already seen.
__global__ void hashSingleBondOrientationsDifferDriver(HashOut64* outA,
                                                       HashOut64* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  HashSeed16 seedA, seedB;
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
  matchA.empty = false;

  mcs::fmcs::seedAddAtomWithinThread(seedB, 0);
  mcs::fmcs::seedAddAtomWithinThread(seedB, 1);
  mcs::fmcs::seedAddBondWithinThread(seedB, 0);
  matchB.targetAtomIdx[0] = 5;
  matchB.targetAtomIdx[1] = 4;
  matchB.targetBondIdx[0] = 3;
  matchB.empty = false;

  outA->hash = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

__global__ void hashLastAddedFrontierDiffersDriver(HashOut64* outA,
                                                   HashOut64* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  HashSeed16 seedA, seedB;
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
  matchA.empty = false;
  matchB.empty = false;

  outA->hash = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

__global__ void hashExcludedBondsDifferDriver(HashOut64* outA,
                                              HashOut64* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  HashSeed16 seedA, seedB;
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
  seedB.excludedBonds[0] |=
      static_cast<HashSeed16::bond_word_type>(1) << 3;
  matchA.empty = false;
  matchB.empty = false;

  outA->hash = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

// Extreme indices: q = 127, t = 127 are the largest values supported
// (tier-128 cap on maxBonds).  Verify the (q << 8 | t) packing
// doesn't overflow and the high-q __ffsll iteration finds the bit.
__global__ void hashExtremeIndicesDriver(HashOut64* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  HashSeed128 seed;
  HashMatch128 match;
  mcs::fmcs::seedClearWithinThread(seed);
  mcs::fmcs::matchResultClearWithinThread(match);
  mcs::fmcs::seedAddBondWithinThread(seed, 127);
  match.targetBondIdx[127] = 127;
  match.empty = false;
  out->hash = mcs::fmcs::mappingHashWithinThread(seed, match);
}

// The (q=0, t=0) bond packs to a zero token; with a zero-initialized
// accumulator and splitMix64(0) == 0 it would be swallowed, aliasing a
// seed that contains it onto one that omits it.  A strict superset that
// adds the q0->t0 bond must therefore hash differently from the subset
// without it.
__global__ void hashZeroPairBondVisibleDriver(HashOut64* outWithout,
                                              HashOut64* outWith) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  HashSeed16 seedWithout, seedWith;
  HashMatch16 matchWithout, matchWith;
  mcs::fmcs::seedClearWithinThread(seedWithout);
  mcs::fmcs::seedClearWithinThread(seedWith);
  mcs::fmcs::matchResultClearWithinThread(matchWithout);
  mcs::fmcs::matchResultClearWithinThread(matchWith);
  mcs::fmcs::seedAddBondWithinThread(seedWithout, 1);
  matchWithout.targetBondIdx[1] = 1;
  matchWithout.empty = false;
  mcs::fmcs::seedAddBondWithinThread(seedWith, 0);
  mcs::fmcs::seedAddBondWithinThread(seedWith, 1);
  matchWith.targetBondIdx[0] = 0;
  matchWith.targetBondIdx[1] = 1;
  matchWith.empty = false;
  outWithout->hash = mcs::fmcs::mappingHashWithinThread(seedWithout, matchWithout);
  outWith->hash    = mcs::fmcs::mappingHashWithinThread(seedWith, matchWith);
}

// ---- Cache idempotent insert ----

struct CacheIdempotentOut {
  bool firstInsert;
  bool secondInsert;
  int  occupiedSlots;  // post-state count of non-zero slots
};

__global__ void cacheIdempotentInsertDriver(
    std::uint64_t* keys, int capacity, CacheIdempotentOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  __shared__ DeviceMatchCache cache;
  cache.init(keys, capacity);
  const std::uint64_t key = 0xC0FFEEC0FFEEC0FFULL;
  out->firstInsert  = cache.insertWithinThread(key);
  out->secondInsert = cache.insertWithinThread(key);
  int occupied = 0;
  for (int i = 0; i < capacity; ++i) if (keys[i] != 0ULL) ++occupied;
  out->occupiedSlots = occupied;
}

// ---- Probe of fresh empty cache ----

struct ProbeEmptyOut {
  bool probedZeroKey;
  bool probedNonZeroKey;
};

__global__ void cacheProbeEmptyDriver(
    std::uint64_t* keys, int capacity, ProbeEmptyOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  __shared__ DeviceMatchCache cache;
  cache.init(keys, capacity);
  // Cache was zero-initialized by the host.  Probing any key should
  // miss immediately at the first slot (it's empty).
  out->probedZeroKey    = cache.probeWithinThread(0ULL);
  out->probedNonZeroKey = cache.probeWithinThread(0xABCDABCD12341234ULL);
}

// ---- Cooperative zero ----

__global__ void cacheZeroCooperativeDriver(
    std::uint64_t* keys, int capacity) {
  __shared__ DeviceMatchCache cache;
  if (threadIdx.x == 0) cache.init(keys, capacity);
  __syncthreads();
  auto block = cooperative_groups::this_thread_block();
  cache.zeroCooperative(block);
}

}  // namespace mcs_fmcs_cache_gap_test

TEST(FMCSUnit, MappingHashMultiWordPath) {
  using namespace mcs_fmcs_cache_gap_test;
  HashOut64* dA = nullptr; HashOut64* dB = nullptr;
  ASSERT_EQ(cudaMallocManaged(&dA, sizeof(HashOut64)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&dB, sizeof(HashOut64)), cudaSuccess);
  hashMultiWordPathDriver<<<1, 1>>>(dA, dB);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_NE(dA->hash, 0ULL);
  EXPECT_EQ(dA->hash, dB->hash)
      << "Multi-word seed.bonds must be canonicalized regardless of "
         "addition order, even when bonds straddle a word boundary.";

  cudaFree(dA);
  cudaFree(dB);
}

TEST(FMCSUnit, MappingHashSingleBondDeterministic) {
  using namespace mcs_fmcs_cache_gap_test;
  HashOut64* dA = nullptr; HashOut64* dB = nullptr;
  ASSERT_EQ(cudaMallocManaged(&dA, sizeof(HashOut64)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&dB, sizeof(HashOut64)), cudaSuccess);
  hashSingleBondDriver<<<1, 1>>>(dA, dB);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_NE(dA->hash, 0ULL);
  EXPECT_NE(dA->hash, 1ULL) << "Should differ from the empty-seed sentinel";
  EXPECT_EQ(dA->hash, dB->hash);

  cudaFree(dA);
  cudaFree(dB);
}

TEST(FMCSUnit, MappingHashSingleBondOrientationsDiffer) {
  using namespace mcs_fmcs_cache_gap_test;
  HashOut64* dA = nullptr; HashOut64* dB = nullptr;
  ASSERT_EQ(cudaMallocManaged(&dA, sizeof(HashOut64)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&dB, sizeof(HashOut64)), cudaSuccess);
  hashSingleBondOrientationsDifferDriver<<<1, 1>>>(dA, dB);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_NE(dA->hash, dB->hash)
      << "Opposite orientations of the same single bond must not alias.";

  cudaFree(dA);
  cudaFree(dB);
}

TEST(FMCSUnit, MappingHashLastAddedFrontierDiffers) {
  using namespace mcs_fmcs_cache_gap_test;
  HashOut64* dA = nullptr; HashOut64* dB = nullptr;
  ASSERT_EQ(cudaMallocManaged(&dA, sizeof(HashOut64)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&dB, sizeof(HashOut64)), cudaSuccess);
  hashLastAddedFrontierDiffersDriver<<<1, 1>>>(dA, dB);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_NE(dA->hash, dB->hash)
      << "Same mapped subgraph with different last-added frontier "
         "must not cache-alias.";

  cudaFree(dA);
  cudaFree(dB);
}

TEST(FMCSUnit, MappingHashExcludedBondsDiffer) {
  using namespace mcs_fmcs_cache_gap_test;
  HashOut64* dA = nullptr; HashOut64* dB = nullptr;
  ASSERT_EQ(cudaMallocManaged(&dA, sizeof(HashOut64)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&dB, sizeof(HashOut64)), cudaSuccess);
  hashExcludedBondsDifferDriver<<<1, 1>>>(dA, dB);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_NE(dA->hash, dB->hash)
      << "Same mapped subgraph with different excluded bonds must not "
         "cache-alias.";

  cudaFree(dA);
  cudaFree(dB);
}

TEST(FMCSUnit, MappingHashExtremeIndicesFitInPacking) {
  using namespace mcs_fmcs_cache_gap_test;
  HashOut64* d = nullptr;
  ASSERT_EQ(cudaMallocManaged(&d, sizeof(HashOut64)), cudaSuccess);
  hashExtremeIndicesDriver<<<1, 1>>>(d);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // The packed pair (127 << 8) | 127 = 0x7F7F fits comfortably in
  // 64 bits, and __ffsll finds bit 63 of word 1 correctly.  Just
  // assert non-zero (i.e., we got past the hash post-process and
  // didn't hit a degenerate result).
  EXPECT_NE(d->hash, 0ULL);

  cudaFree(d);
}

TEST(FMCSUnit, MappingHashZeroPairBondIsVisible) {
  using namespace mcs_fmcs_cache_gap_test;
  HashOut64* without = nullptr; HashOut64* with = nullptr;
  ASSERT_EQ(cudaMallocManaged(&without, sizeof(HashOut64)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&with, sizeof(HashOut64)), cudaSuccess);
  hashZeroPairBondVisibleDriver<<<1, 1>>>(without, with);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_NE(with->hash, without->hash)
      << "A (q0->t0) bond must perturb the hash; otherwise a full seed "
         "aliases onto a sub-seed and the success cache falsely dedups it.";
  EXPECT_NE(with->hash, 0ULL);

  cudaFree(without);
  cudaFree(with);
}

TEST(FMCSUnit, CacheInsertIsIdempotent) {
  using namespace mcs_fmcs_cache_gap_test;
  constexpr int kCap = 16;
  std::uint64_t* keys = nullptr;
  ASSERT_EQ(cudaMallocManaged(&keys, sizeof(std::uint64_t) * kCap), cudaSuccess);
  std::memset(keys, 0, sizeof(std::uint64_t) * kCap);
  CacheIdempotentOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(CacheIdempotentOut)), cudaSuccess);

  cacheIdempotentInsertDriver<<<1, 1>>>(keys, kCap, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(out->firstInsert);
  EXPECT_TRUE(out->secondInsert);
  EXPECT_EQ(out->occupiedSlots, 1)
      << "Repeated insert of the same key must not consume a second slot.";

  cudaFree(keys);
  cudaFree(out);
}

TEST(FMCSUnit, CacheProbeEmptyCacheMisses) {
  using namespace mcs_fmcs_cache_gap_test;
  constexpr int kCap = 16;
  std::uint64_t* keys = nullptr;
  ASSERT_EQ(cudaMallocManaged(&keys, sizeof(std::uint64_t) * kCap), cudaSuccess);
  std::memset(keys, 0, sizeof(std::uint64_t) * kCap);
  ProbeEmptyOut* out = nullptr;
  ASSERT_EQ(cudaMallocManaged(&out, sizeof(ProbeEmptyOut)), cudaSuccess);

  cacheProbeEmptyDriver<<<1, 1>>>(keys, kCap, out);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_FALSE(out->probedZeroKey)
      << "0 is the empty-slot sentinel; probing it must miss.";
  EXPECT_FALSE(out->probedNonZeroKey);

  cudaFree(keys);
  cudaFree(out);
}

TEST(FMCSUnit, CacheZeroCooperativeWipesAllSlots) {
  using namespace mcs_fmcs_cache_gap_test;
  constexpr int kCap = 64;
  std::uint64_t* keys = nullptr;
  ASSERT_EQ(cudaMallocManaged(&keys, sizeof(std::uint64_t) * kCap), cudaSuccess);
  // Pre-fill with non-zero garbage so the cooperative zero has work
  // to do.
  for (int i = 0; i < kCap; ++i) keys[i] = 0xDEADDEAD00000000ULL | i;

  cacheZeroCooperativeDriver<<<1, 32>>>(keys, kCap);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  for (int i = 0; i < kCap; ++i) {
    EXPECT_EQ(keys[i], 0ULL) << "slot " << i << " not zeroed";
  }

  cudaFree(keys);
}
