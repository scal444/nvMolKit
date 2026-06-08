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

#ifndef NVMOLKIT_GPU_SCHEDULER_CONFIG_H
#define NVMOLKIT_GPU_SCHEDULER_CONFIG_H

#include <algorithm>
#include <thread>
#include <vector>

namespace nvMolKit::gpu_scheduler {

/**
 * @brief Orchestration-only knobs for gpu_scheduler::Pipeline.
 *
 * Workload-specific sizing (batch size, max matches, force-field tolerances,
 * fingerprint radius, etc.) lives on the workload's own configuration struct
 * and is reached via Workload methods, never through Config. The Pipeline
 * itself only needs to know how many threads to spin up, which devices to
 * target, and how many in-flight batches each runner is allowed to juggle.
 *
 * All thread-count fields accept -1 to request autoselect. Resolved values
 * are returned by resolve().
 */
struct Config {
  /// Per-GPU runner thread count. Each runner thread owns slotsPerWorker
  /// in-flight GPU "slots". -1 = autoselect (min(4, hardware_concurrency / numGpus)).
  int workerThreadsPerGpu = -1;

  /// Total preprocessing thread count, shared across all GPUs.
  /// -1 = autoselect (hardware_concurrency).
  int globalPreprocessingThreads = -1;

  /// Per-runner in-flight batch cap. Each slot owns one cudaStream
  /// (or stream group) and a set of device buffers. -1 = autoselect
  /// (3 if there's a single runner globally, 2 otherwise).
  int slotsPerWorker = -1;

  /// CUDA device IDs to dispatch onto. Empty = current device only.
  std::vector<int> gpuIds;
};

/**
 * @brief Concrete (non-negative) values resolved from a Config.
 *
 * Pipeline resolves once at the start of run() and stores the resolved view
 * internally; this struct is also exposed for unit testing and for workloads
 * that want to size their own pinned buffer pools.
 */
struct ResolvedConfig {
  int              workerThreadsPerGpu        = 0;
  int              globalPreprocessingThreads = 0;
  int              slotsPerWorker             = 0;
  std::vector<int> gpuIds;
};

/**
 * @brief Resolve any -1 fields against the provided defaults.
 *
 * @param config   User-supplied configuration (may contain -1 fields).
 * @param hardwareConcurrency  Result of std::thread::hardware_concurrency()
 *                             at the call site (parameterized for testability).
 * @param currentDevice        CUDA device to use when config.gpuIds is empty.
 */
inline ResolvedConfig resolve(const Config& config, unsigned int hardwareConcurrency, int currentDevice) {
  ResolvedConfig out;

  out.gpuIds = config.gpuIds;
  if (out.gpuIds.empty()) {
    out.gpuIds.push_back(currentDevice);
  }
  const int numGpus = static_cast<int>(out.gpuIds.size());

  const int hwThreads = std::max(1, static_cast<int>(hardwareConcurrency));
  if (config.workerThreadsPerGpu == -1) {
    out.workerThreadsPerGpu = std::max(1, std::min(4, hwThreads / numGpus));
  } else {
    out.workerThreadsPerGpu = std::max(1, config.workerThreadsPerGpu);
  }

  if (config.globalPreprocessingThreads == -1) {
    out.globalPreprocessingThreads = hwThreads;
  } else {
    out.globalPreprocessingThreads = std::max(1, config.globalPreprocessingThreads);
  }

  if (config.slotsPerWorker == -1) {
    const int totalRunners = out.workerThreadsPerGpu * numGpus;
    out.slotsPerWorker     = (totalRunners == 1) ? 3 : 2;
  } else {
    out.slotsPerWorker = std::max(1, config.slotsPerWorker);
  }

  return out;
}

}  // namespace nvMolKit::gpu_scheduler

#endif  // NVMOLKIT_GPU_SCHEDULER_CONFIG_H
