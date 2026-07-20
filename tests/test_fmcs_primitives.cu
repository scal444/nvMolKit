// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>

#include "fmcs_cuda/fmcs_seed.cuh"
#include "fmcs_cuda/fmcs_seed_queue.cuh"

namespace {

using mcs::fmcs::Seed;
using mcs::fmcs::SeedQueue;
using mcs::fmcs::ThreadBlockScope;

struct QueueResult {
  int  first;
  int  second;
  bool empty;
  bool overflowAccepted;
};

__global__ void seedMutationKernel(Seed<128, 128>* seed) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  *seed = {};
  mcs::fmcs::seedAddAtomWithinThread(*seed, 3);
  mcs::fmcs::seedAddAtomWithinThread(*seed, 67);
  mcs::fmcs::seedAddBondWithinThread(*seed, 7);
  mcs::fmcs::seedAddBondWithinThread(*seed, 71);
}

__global__ void queueKernel(int* storage, QueueResult* result) {
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  if (threadIdx.x == 0) {
    queue.init(storage, 2);
    queue.pushWithinThread(11);
    queue.pushWithinThread(22);
    result->overflowAccepted = queue.pushWithinThread(33);
    queue.popWithinThread(result->first);
    queue.popWithinThread(result->second);
    int ignored{};
    result->empty = !queue.popWithinThread(ignored);
  }
}

}  // namespace

TEST(FMCSPrimitives, SeedMutationCrossesBitsetWords) {
  Seed<128, 128>* seed = nullptr;
  ASSERT_EQ(cudaMallocManaged(&seed, sizeof(*seed)), cudaSuccess);

  seedMutationKernel<<<1, 1>>>(seed);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(seed->numAtoms, 2);
  EXPECT_EQ(seed->atoms[0], 1ULL << 3);
  EXPECT_EQ(seed->atoms[1], 1ULL << 3);
  EXPECT_EQ(seed->numBonds, 2);
  EXPECT_EQ(seed->bonds[0], 1ULL << 7);
  EXPECT_EQ(seed->bonds[1], 1ULL << 7);
  EXPECT_EQ(seed->excludedBonds[0], seed->bonds[0]);
  EXPECT_EQ(seed->excludedBonds[1], seed->bonds[1]);

  EXPECT_EQ(cudaFree(seed), cudaSuccess);
}

TEST(FMCSPrimitives, QueueIsBoundedLifo) {
  int*         storage = nullptr;
  QueueResult* result  = nullptr;
  ASSERT_EQ(cudaMallocManaged(&storage, 2 * sizeof(*storage)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&result, sizeof(*result)), cudaSuccess);

  queueKernel<<<1, 1>>>(storage, result);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_FALSE(result->overflowAccepted);
  EXPECT_EQ(result->first, 22);
  EXPECT_EQ(result->second, 11);
  EXPECT_TRUE(result->empty);

  EXPECT_EQ(cudaFree(result), cudaSuccess);
  EXPECT_EQ(cudaFree(storage), cudaSuccess);
}
