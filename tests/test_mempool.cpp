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

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "src/utils/device_vector.h"
#include "src/utils/mempool.h"

using namespace nvMolKit;

namespace {

cudaMemPool_t defaultPool() {
  int device = 0;
  EXPECT_EQ(cudaGetDevice(&device), cudaSuccess);
  cudaMemPool_t pool = nullptr;
  EXPECT_EQ(cudaDeviceGetDefaultMemPool(&pool, device), cudaSuccess);
  return pool;
}

std::uint64_t releaseThreshold(cudaMemPool_t pool) {
  std::uint64_t value = 0;
  EXPECT_EQ(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &value), cudaSuccess);
  return value;
}

void setReleaseThreshold(cudaMemPool_t pool, std::uint64_t value) {
  EXPECT_EQ(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &value), cudaSuccess);
}

//! Sets the environment variable while alive and restores its previous value afterwards.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* previous = std::getenv(name)) {
      previous_ = previous;
    }
    setenv(name, value, 1);
  }
  ScopedEnv(const ScopedEnv&)            = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;
  ~ScopedEnv() {
    if (previous_.has_value()) {
      setenv(name_, previous_->c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char*                name_;
  std::optional<std::string> previous_;
};

//! Restores the default pool's release threshold when the test ends.
class MemPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_     = defaultPool();
    original_ = releaseThreshold(pool_);
  }
  void TearDown() override { setReleaseThreshold(pool_, original_); }

  cudaMemPool_t pool_     = nullptr;
  std::uint64_t original_ = 0;
};

}  // namespace

TEST(MemPoolThresholdParse, UnsetOrEmptyUsesDefault) {
  EXPECT_EQ(parseMemPoolReleaseThreshold(nullptr), kDefaultMemPoolReleaseThreshold);
  EXPECT_EQ(parseMemPoolReleaseThreshold(""), kDefaultMemPoolReleaseThreshold);
}

TEST(MemPoolThresholdParse, ByteCount) {
  EXPECT_EQ(parseMemPoolReleaseThreshold("1048576"), std::uint64_t{1048576});
  EXPECT_EQ(parseMemPoolReleaseThreshold("18446744073709551615"), UINT64_MAX);
}

TEST(MemPoolThresholdParse, ZeroOptsOut) {
  EXPECT_EQ(parseMemPoolReleaseThreshold("0"), std::nullopt);
}

TEST(MemPoolThresholdParse, InvalidLeavesPoolUntouched) {
  EXPECT_EQ(parseMemPoolReleaseThreshold("abc"), std::nullopt);
  EXPECT_EQ(parseMemPoolReleaseThreshold("12MB"), std::nullopt);
  EXPECT_EQ(parseMemPoolReleaseThreshold("-1"), std::nullopt);
  EXPECT_EQ(parseMemPoolReleaseThreshold(" -1"), std::nullopt);
  EXPECT_EQ(parseMemPoolReleaseThreshold("\t-1"), std::nullopt);
  EXPECT_EQ(parseMemPoolReleaseThreshold("+1"), std::nullopt);
  EXPECT_EQ(parseMemPoolReleaseThreshold(" 1"), std::nullopt);
  EXPECT_EQ(parseMemPoolReleaseThreshold("1 "), std::nullopt);
  EXPECT_EQ(parseMemPoolReleaseThreshold("99999999999999999999999"), std::nullopt);
}

TEST_F(MemPoolTest, FirstAllocationSetsThreshold) {
  if (std::getenv(kMemPoolReleaseThresholdEnvVar) != nullptr) {
    GTEST_SKIP() << kMemPoolReleaseThresholdEnvVar << " is set";
  }
  if (original_ != 0) {
    GTEST_SKIP() << "Default pool was already configured in this process";
  }
  std::vector<std::thread> workers;
  for (int index = 0; index < 8; ++index) {
    workers.emplace_back([] { AsyncDeviceVector<int> values(16); });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  EXPECT_EQ(releaseThreshold(pool_), kDefaultMemPoolReleaseThreshold);
}

TEST_F(MemPoolTest, UserThresholdIsPreserved) {
  constexpr std::uint64_t userThreshold = 12345;
  setReleaseThreshold(pool_, userThreshold);
  AsyncDeviceVector<int> vec(16);
  int                    device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  configureDefaultMemPool(device);
  EXPECT_EQ(releaseThreshold(pool_), userThreshold);
}

TEST_F(MemPoolTest, UnsetThresholdIsConfigured) {
  setReleaseThreshold(pool_, 0);
  const ScopedEnv env(kMemPoolReleaseThresholdEnvVar, "4194304");
  int             device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  configureDefaultMemPool(device);
  EXPECT_EQ(releaseThreshold(pool_), std::uint64_t{4194304});
}

TEST_F(MemPoolTest, EnvZeroLeavesPoolUntouched) {
  setReleaseThreshold(pool_, 0);
  const ScopedEnv env(kMemPoolReleaseThresholdEnvVar, "0");
  int             device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  configureDefaultMemPool(device);
  EXPECT_EQ(releaseThreshold(pool_), 0U);
}

TEST_F(MemPoolTest, FreedMemoryIsRetainedAndCountedIdle) {
  setReleaseThreshold(pool_, kDefaultMemPoolReleaseThreshold);
  constexpr std::size_t bytes = std::size_t{8} << 20;
  { AsyncDeviceVector<char> vec(bytes); }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const std::size_t idleBytes = getMemPoolIdleBytes();
  ASSERT_GE(idleBytes, bytes);
  AsyncDeviceVector<char> reused(bytes);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_LE(getMemPoolIdleBytes(), idleBytes - bytes);
}
