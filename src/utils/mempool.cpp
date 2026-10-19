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

#include "src/utils/mempool.h"

#include <cuda_runtime.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace nvMolKit {

namespace {

//! Clears the error a failed query leaves behind so later cudaGetLastError checks do not report it.
bool succeeded(const cudaError_t code) {
  if (code != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    return false;
  }
  return true;
}

}  // namespace

std::optional<std::uint64_t> parseMemPoolReleaseThreshold(const char* value) {
  if (value == nullptr || *value == '\0') {
    return kDefaultMemPoolReleaseThreshold;
  }
  char* end                       = nullptr;
  errno                           = 0;
  // NOLINTNEXTLINE(google-runtime-int)
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || *end != '\0' || *value < '0' || *value > '9') {
    std::fprintf(stderr,
                 "nvMolKit: ignoring invalid %s value '%s'; the CUDA memory pool is left unchanged\n",
                 kMemPoolReleaseThresholdEnvVar,
                 value);
    return std::nullopt;
  }
  if (parsed == 0) {
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(parsed);
}

void configureDefaultMemPool(const int device) {
  const std::optional<std::uint64_t> threshold =
    parseMemPoolReleaseThreshold(std::getenv(kMemPoolReleaseThresholdEnvVar));
  if (!threshold.has_value()) {
    return;
  }
  int supported = 0;
  if (!succeeded(cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device)) || supported == 0) {
    return;
  }
  // Only the CUDA default pool is touched. PyTorch and CuPy allocate from their own pools by default; a library or
  // user that configures this pool sets a non-zero threshold, which is kept.
  cudaMemPool_t pool = nullptr;
  if (!succeeded(cudaDeviceGetDefaultMemPool(&pool, device))) {
    return;
  }
  std::uint64_t current = 0;
  if (!succeeded(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &current)) || current != 0) {
    return;
  }
  std::uint64_t value = *threshold;
  static_cast<void>(succeeded(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &value)));
}

void ensureDefaultMemPoolConfigured() {
  static const int numDevices = [] {
    int count = 0;
    return succeeded(cudaGetDeviceCount(&count)) ? count : 0;
  }();
  static std::vector<std::once_flag> configured(numDevices);
  int                                device = 0;
  if (!succeeded(cudaGetDevice(&device)) || device < 0 || device >= numDevices) {
    return;
  }
  std::call_once(configured[device], configureDefaultMemPool, device);
}

std::size_t getMemPoolIdleBytes() {
  int device = 0;
  if (!succeeded(cudaGetDevice(&device))) {
    return 0;
  }
  cudaMemPool_t pool = nullptr;
  if (!succeeded(cudaDeviceGetMemPool(&pool, device))) {
    return 0;
  }
  std::uint64_t reserved = 0;
  std::uint64_t used     = 0;
  if (!succeeded(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved)) ||
      !succeeded(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used))) {
    return 0;
  }
  return reserved > used ? static_cast<std::size_t>(reserved - used) : 0;
}

}  // namespace nvMolKit
