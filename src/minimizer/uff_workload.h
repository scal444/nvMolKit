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

#ifndef NVMOLKIT_UFF_WORKLOAD_H
#define NVMOLKIT_UFF_WORKLOAD_H

#include <GraphMol/ROMol.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "rdkit_extensions/uff_flattened_builder.h"
#include "src/conformer/conformer_info.h"
#include "src/conformer/device_coord_collector.h"
#include "src/conformer/device_coord_result.h"
#include "src/conformer/ff_device_collect.h"
#include "src/forcefields/ff_utils.h"
#include "src/forcefields/forcefield_constraints.h"
#include "src/forcefields/uff.h"
#include "src/forcefields/uff_batched_forcefield.h"
#include "src/gpu_scheduler/workload.h"
#include "src/minimizer/bfgs_common.h"
#include "src/minimizer/bfgs_minimize.h"
#include "src/minimizer/bfgs_types.h"
#include "src/utils/device.h"
#include "src/utils/nvtx.h"

namespace nvMolKit::UFF {

/**
 * @brief Inputs shared across all UFF workload threads.
 *
 * Owned by `UFFMinimizeMoleculesConfs` and held by reference inside
 * UffWorkload. Per-molecule UFF parameters (`vdwThresholds`,
 * `ignoreInterfragInteractions`, `constraints`) are read-only; the result
 * vectors are written under `outputMutex` from postprocess.
 */
struct UffInputs {
  // Read-only inputs.
  const std::vector<ConformerInfo>*                            allConformers               = nullptr;
  const std::vector<double>*                                   vdwThresholds               = nullptr;
  const std::vector<bool>*                                     ignoreInterfragInteractions = nullptr;
  const std::vector<ForceFieldConstraints::PerMolConstraints>* constraints                 = nullptr;
  int                                                          maxIters                    = 200;
  double                                                       gradTol                     = 1e-4;
  /// Number of conformers per pipeline mini-batch.
  int                                                          batchSize                   = 0;

  /// See MmffInputs for the equivalent device-input / device-output fields;
  /// UFF mirrors MMFF here, with no per-batch backend selection (UFF always
  /// uses the BATCHED kernels).
  CoordinateOutput                            output           = CoordinateOutput::RDKIT_CONFORMERS;
  const DeviceCoordResult*                    deviceInput      = nullptr;
  const detail::DeviceInputIndex*             deviceInputIndex = nullptr;
  std::vector<detail::DeviceCoordCollector>*  deviceCollectors = nullptr;
  /// See MmffInputs::runnerStreams. Same lifetime rules apply.
  std::vector<std::unique_ptr<ScopedStream>>* runnerStreams    = nullptr;
  std::atomic<int>                            nextRunnerIdx{0};

  // RDKIT_CONFORMERS-mode result sinks (mutex-protected).
  std::vector<std::vector<double>>* moleculeEnergies  = nullptr;
  std::vector<std::vector<int8_t>>* moleculeConverged = nullptr;
  std::mutex*                       outputMutex       = nullptr;
};

/**
 * @brief One UFF mini-batch: a contiguous slice of the flattened
 * conformer list plus the host-side batched system built for it.
 */
struct UffBatch : gpu_scheduler::PreparedBatch {
  std::vector<ConformerInfo> conformers;
  std::vector<std::uint32_t> conformerAtomStarts;
  BatchedMolecularSystemHost systemHost;
  BatchedForcefieldMetadata  metadata;
  /// Per-conformer BFGS convergence statuses, populated by dispatch (queued
  /// async D2H on the slot stream) and consumed by postprocess. Empty in
  /// DEVICE output mode (the device-side collector reads statuses directly).
  std::vector<int16_t>       statusesHost;

  /// Per-batch precomputed source-conformer indices and atom counts used for
  /// device-input broadcasting; populated in preprocess only when
  /// `Inputs.deviceInput != nullptr`.
  std::vector<int> batchSrcIndices;
  std::vector<int> batchAtomCounts;
};

/**
 * @brief Per-runner-slot device state (stream + completion event + reusable
 * pinned host buffers for D2H transfers).
 */
struct UffSlot : gpu_scheduler::GpuSlotState {
  ScopedStream       stream;
  ScopedCudaEvent    completion;
  ThreadLocalBuffers buffers;

  cudaStream_t primaryStream() const override { return stream.stream(); }
  cudaEvent_t  completionEvent() const override { return completion.event(); }
};

/**
 * @brief Per-CUDA-device read-only state. Empty for UFF.
 */
struct UffPerGpu : gpu_scheduler::PerGpuState {
  int deviceId = -1;
};

/**
 * @brief Runtime-polymorphic UFF workload consumed by gpu_scheduler::Pipeline.
 *
 * Lifecycle mirrors MmffWorkload: preprocess builds the host system from a
 * conformer slice (UFF rebuilds parameters per conformer because they
 * depend on the conformer id and per-molecule vdw / interfrag flags),
 * dispatch uploads + runs BFGS via UFFBatchedForcefield + queues async
 * D2H copies, postprocess writes positions and statuses back under the
 * shared outputMutex.
 */
class UffWorkload : public gpu_scheduler::Workload {
 public:
  using Inputs        = UffInputs;
  using PreparedBatch = UffBatch;
  using PerGpuState   = UffPerGpu;
  using GpuSlotState  = UffSlot;
  /// No per-thread preproc state needed.
  struct PreprocThreadContext : gpu_scheduler::PreprocThreadContext {};
  /// Per-runner-thread context. In DEVICE output mode, `collectorIdx` holds
  /// the runner's exclusive slot into `Inputs.deviceCollectors`, claimed in
  /// `makeRunnerCtx` via `Inputs.nextRunnerIdx`.
  struct RunnerThreadContext : gpu_scheduler::RunnerThreadContext {
    int collectorIdx = -1;
  };

  explicit UffWorkload(Inputs& inputs) : inputs_(inputs) {}

  int totalUnits() const override;
  int unitsPerPreprocBatch() const override;

  std::unique_ptr<gpu_scheduler::RunnerThreadContext> makeRunnerCtx() override;

  void preprocess(gpu_scheduler::IndexRange            range,
                  gpu_scheduler::PreprocThreadContext& ctx,
                  const gpu_scheduler::PushBatch&      pushBatch) override;

  std::unique_ptr<gpu_scheduler::PerGpuState>  makePerGpuState(int gpuId) override;
  std::unique_ptr<gpu_scheduler::GpuSlotState> makeSlotState(gpu_scheduler::PerGpuState& pgs, int gpuId) override;

  void dispatchAndCopyBack(gpu_scheduler::GpuSlotState&        slot,
                           gpu_scheduler::PerGpuState&         pgs,
                           gpu_scheduler::PreparedBatch&       batch,
                           gpu_scheduler::RunnerThreadContext& ctx) override;

  void postprocess(gpu_scheduler::GpuSlotState&        slot,
                   gpu_scheduler::PreparedBatch&       batch,
                   gpu_scheduler::RunnerThreadContext& ctx) override;

 private:
  Inputs& inputs_;
};

}  // namespace nvMolKit::UFF

#endif  // NVMOLKIT_UFF_WORKLOAD_H
