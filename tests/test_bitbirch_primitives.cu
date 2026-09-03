// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "src/bitbirch.h"
#include "src/bitbirch_common.cuh"
#include "src/utils/device_vector.h"

namespace {

struct PrimitiveResults {
  bool   centroidBits[5];
  double isim;
  bool   exactThreshold;
  bool   rejectsAboveThreshold;
  bool   singletonTolerance;
  bool   tolerancePass;
  bool   toleranceFail;
};

__global__ void evaluatePrimitivesKernel(PrimitiveResults* output) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  PrimitiveResults result{};
  for (std::uint64_t linearSum = 0; linearSum < 5; ++linearSum) {
    result.centroidBits[linearSum] = nvMolKit::bitbirch::majorityCentroidBit(linearSum, 4);
  }
  nvMolKit::bitbirch::ISimTanimotoTerms terms{};
  nvMolKit::bitbirch::accumulateISimTanimotoTerm(terms, 2, 3);
  nvMolKit::bitbirch::accumulateISimTanimotoTerm(terms, 1, 3);
  result.isim                  = nvMolKit::bitbirch::isimTanimoto(terms, 3);
  result.exactThreshold        = nvMolKit::bitbirch::isimTanimotoAtLeast(terms, 3, 0.2);
  result.rejectsAboveThreshold = !nvMolKit::bitbirch::isimTanimotoAtLeast(terms, 3, 0.2000001);
  result.singletonTolerance    = nvMolKit::bitbirch::singletonToleranceAllows(1.0, 0.1, 1, 0.0);
  result.tolerancePass         = nvMolKit::bitbirch::singletonToleranceAllows(0.8, 0.7, 4, 0.3);
  result.toleranceFail         = !nvMolKit::bitbirch::singletonToleranceAllows(0.8, 0.7, 4, 0.2);
  *output                      = result;
}

TEST(BitBirchPrimitives, DeviceEvaluationMatchesPaperEquations) {
  nvMolKit::AsyncDevicePtr<PrimitiveResults> deviceResults;
  evaluatePrimitivesKernel<<<1, 1>>>(deviceResults.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  PrimitiveResults result{};
  deviceResults.get(result);
  EXPECT_FALSE(result.centroidBits[0]);
  EXPECT_FALSE(result.centroidBits[1]);
  EXPECT_TRUE(result.centroidBits[2]);
  EXPECT_TRUE(result.centroidBits[3]);
  EXPECT_TRUE(result.centroidBits[4]);
  EXPECT_DOUBLE_EQ(result.isim, 0.2);
  EXPECT_TRUE(result.exactThreshold);
  EXPECT_TRUE(result.rejectsAboveThreshold);
  EXPECT_TRUE(result.singletonTolerance);
  EXPECT_TRUE(result.tolerancePass);
  EXPECT_TRUE(result.toleranceFail);
}

TEST(BitBirchPrimitives, HostEdgeCaseConventionsAreExplicit) {
  nvMolKit::bitbirch::ISimTanimotoTerms empty{};
  EXPECT_DOUBLE_EQ(nvMolKit::bitbirch::isimTanimoto(empty, 0), 1.0);
  EXPECT_DOUBLE_EQ(nvMolKit::bitbirch::isimTanimoto(empty, 1), 1.0);
  EXPECT_DOUBLE_EQ(nvMolKit::bitbirch::isimTanimoto(empty, 8), 1.0);

  nvMolKit::bitbirch::ISimTanimotoTerms large{};
  const std::uint64_t                   count = std::uint64_t{1} << 40;
  nvMolKit::bitbirch::accumulateISimTanimotoTerm(large, count, count);
  nvMolKit::bitbirch::accumulateISimTanimotoTerm(large, count / 2, count);
  EXPECT_TRUE(std::isfinite(large.commonPairs));
  EXPECT_TRUE(std::isfinite(large.mismatches));
  EXPECT_GT(nvMolKit::bitbirch::isimTanimoto(large, count), 0.0);
  EXPECT_LT(nvMolKit::bitbirch::isimTanimoto(large, count), 1.0);
}

TEST(BitBirchValidation, RejectsNonfiniteOptionsBeforeLaunching) {
  const cuda::std::span<const std::uint32_t> empty;
  EXPECT_THROW(nvMolKit::bitBirchSerialGpu(empty, 0, 1, std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);
  EXPECT_THROW(nvMolKit::bitBirchSerialGpu(empty,
                                           0,
                                           1,
                                           0.5,
                                           3,
                                           nvMolKit::BitBirchMergeCriterion::Diameter,
                                           std::numeric_limits<double>::infinity()),
               std::invalid_argument);
  EXPECT_THROW(nvMolKit::bitBirchSerialGpu(empty, 0, 1, 0.5, 3, static_cast<nvMolKit::BitBirchMergeCriterion>(255)),
               std::invalid_argument);
}

TEST(BitBirchSerial, MergesAtThresholdAndReturnsMajorityCentroids) {
  const std::vector<std::uint32_t>           fingerprints{0b0011U, 0b0001U, 0b1100U, 0b0100U};
  nvMolKit::AsyncDeviceVector<std::uint32_t> deviceFingerprints(fingerprints.size());
  deviceFingerprints.copyFromHost(fingerprints);

  auto             result = nvMolKit::bitBirchSerialGpu({deviceFingerprints.data(), deviceFingerprints.size()},
                                            4,
                                            1,
                                            0.5,
                                            3,
                                            nvMolKit::BitBirchMergeCriterion::Diameter,
                                            0.05,
                                            true);
  std::vector<int> labels(4);
  std::vector<std::uint32_t> centroids(result.centroids.size());
  result.clusterIds.copyToHost(labels);
  result.centroids.copyToHost(centroids);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(result.numClusters, 2);
  EXPECT_EQ(labels, (std::vector<int>{0, 0, 1, 1}));
  ASSERT_GE(centroids.size(), 2);
  EXPECT_EQ(centroids[0], 0b0011U);
  EXPECT_EQ(centroids[1], 0b1100U);
}

TEST(BitBirchSerial, CascadingSplitsPreserveDeterministicSingletonLabels) {
  std::vector<std::uint32_t> fingerprints(24);
  for (int index = 0; index < 24; ++index) {
    fingerprints[index] = std::uint32_t{1} << index;
  }
  nvMolKit::AsyncDeviceVector<std::uint32_t> deviceFingerprints(fingerprints.size());
  deviceFingerprints.copyFromHost(fingerprints);

  auto             result = nvMolKit::bitBirchSerialGpu({deviceFingerprints.data(), deviceFingerprints.size()},
                                            24,
                                            1,
                                            0.9,
                                            3,
                                            nvMolKit::BitBirchMergeCriterion::Diameter);
  std::vector<int> labels(24);
  result.clusterIds.copyToHost(labels);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(result.numClusters, 24);
  for (int index = 0; index < 24; ++index) {
    EXPECT_EQ(labels[index], index);
  }
}

TEST(BitBirchPartitioned, MergesPartialTreeSummariesAndReturnsCentroids) {
  const std::vector<std::uint32_t>           fingerprints{0b0011U, 0b0001U, 0b1100U, 0b0100U};
  nvMolKit::AsyncDeviceVector<std::uint32_t> deviceFingerprints(fingerprints.size());
  deviceFingerprints.copyFromHost(fingerprints);

  auto                       result = nvMolKit::bitBirchGpu({deviceFingerprints.data(), deviceFingerprints.size()},
                                      4,
                                      1,
                                      0.5,
                                      3,
                                      nvMolKit::BitBirchMergeCriterion::Diameter,
                                      0.05,
                                      2,
                                      true);
  std::vector<int>           labels(4);
  std::vector<std::uint32_t> centroids(result.centroids.size());
  result.clusterIds.copyToHost(labels);
  result.centroids.copyToHost(centroids);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(result.numClusters, 2);
  EXPECT_EQ(labels, (std::vector<int>{0, 0, 1, 1}));
  EXPECT_EQ(centroids[0], 0b0011U);
  EXPECT_EQ(centroids[1], 0b1100U);
}

TEST(BitBirchPartitioned, SplitHeavyPartitionsAssignEveryInput) {
  std::vector<std::uint32_t> fingerprints(24);
  for (int index = 0; index < 24; ++index) {
    fingerprints[index] = std::uint32_t{1} << index;
  }
  nvMolKit::AsyncDeviceVector<std::uint32_t> deviceFingerprints(fingerprints.size());
  deviceFingerprints.copyFromHost(fingerprints);

  auto             result = nvMolKit::bitBirchGpu({deviceFingerprints.data(), deviceFingerprints.size()},
                                      24,
                                      1,
                                      0.9,
                                      3,
                                      nvMolKit::BitBirchMergeCriterion::Diameter,
                                      0.05,
                                      4);
  std::vector<int> labels(24);
  result.clusterIds.copyToHost(labels);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(result.numClusters, 24);
  for (int index = 0; index < 24; ++index) {
    EXPECT_EQ(labels[index], index);
  }
}

TEST(BitBirchSerial, BalancedSplitsStayWithinLinearPoolOnDeepSingletonTree) {
  constexpr int              numFingerprints = 512;
  constexpr int              numWords        = numFingerprints / 32;
  std::vector<std::uint32_t> fingerprints(static_cast<std::size_t>(numFingerprints) * numWords, 0);
  for (int index = 0; index < numFingerprints; ++index) {
    fingerprints[static_cast<std::size_t>(index) * numWords + index / 32] = std::uint32_t{1} << (index % 32);
  }
  nvMolKit::AsyncDeviceVector<std::uint32_t> deviceFingerprints(fingerprints.size());
  deviceFingerprints.copyFromHost(fingerprints);

  auto             result = nvMolKit::bitBirchSerialGpu({deviceFingerprints.data(), deviceFingerprints.size()},
                                            numFingerprints,
                                            numWords,
                                            1.0,
                                            3,
                                            nvMolKit::BitBirchMergeCriterion::Diameter);
  std::vector<int> labels(numFingerprints);
  result.clusterIds.copyToHost(labels);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(result.numClusters, numFingerprints);
  for (int index = 0; index < numFingerprints; ++index) {
    EXPECT_EQ(labels[index], index);
  }
}

TEST(BitBirchPartitioned, NonDivisiblePartitionsCrossUint8ToUint16SummaryWidths) {
  constexpr int              numFingerprints = 300;
  constexpr int              numWords        = (numFingerprints + 31) / 32;
  std::vector<std::uint32_t> fingerprints(static_cast<std::size_t>(numFingerprints) * numWords, 0);
  for (int index = 0; index < numFingerprints; ++index) {
    fingerprints[static_cast<std::size_t>(index) * numWords + index / 32] = std::uint32_t{1} << (index % 32);
  }
  nvMolKit::AsyncDeviceVector<std::uint32_t> deviceFingerprints(fingerprints.size());
  deviceFingerprints.copyFromHost(fingerprints);

  auto             result = nvMolKit::bitBirchGpu({deviceFingerprints.data(), deviceFingerprints.size()},
                                      numFingerprints,
                                      numWords,
                                      1.0,
                                      3,
                                      nvMolKit::BitBirchMergeCriterion::Diameter,
                                      0.05,
                                      7);
  std::vector<int> labels(numFingerprints);
  result.clusterIds.copyToHost(labels);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(result.numClusters, numFingerprints);
  for (int index = 0; index < numFingerprints; ++index) {
    EXPECT_EQ(labels[index], index);
  }
}

}  // namespace
