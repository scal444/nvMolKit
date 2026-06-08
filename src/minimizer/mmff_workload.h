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

#ifndef NVMOLKIT_MMFF_WORKLOAD_H
#define NVMOLKIT_MMFF_WORKLOAD_H

#include <GraphMol/ROMol.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "rdkit_extensions/mmff_flattened_builder.h"
#include "src/conformer/conformer_info.h"
#include "src/conformer/device_coord_collector.h"
#include "src/conformer/device_coord_result.h"
#include "src/conformer/ff_device_collect.h"
#include "src/forcefields/ff_utils.h"
#include "src/forcefields/forcefield_constraints.h"
#include "src/forcefields/mmff.h"
#include "src/forcefields/mmff_batched_forcefield.h"
#include "src/forcefields/mmff_properties.h"
#include "src/gpu_scheduler/workload.h"
#include "src/minimizer/bfgs_common.h"
#include "src/minimizer/bfgs_minimize.h"
#include "src/minimizer/bfgs_types.h"
#include "src/utils/device.h"
#include "src/utils/nvtx.h"

namespace nvMolKit::MMFF {

/**
 * @brief Inputs shared across all MMFF workload threads.
 *
 * Owned by `MMFFMinimizeMoleculesConfs` and held by reference inside
 * MmffWorkload. The `mols` / `properties` / `constraints` fields are
 * read-only; the result vectors are written under `outputMutex` from
 * postprocess.
 */
struct MmffInputs {
  // Read-only inputs.
  const std::vector<ConformerInfo>*                            allConformers = nullptr;
  const std::vector<MMFFProperties>*                           properties    = nullptr;
  const std::vector<ForceFieldConstraints::PerMolConstraints>* constraints   = nullptr;
  int                                                          maxIters      = 200;
  double                                                       gradTol       = 1e-4;
  BfgsBackend                                                  backend       = BfgsBackend::HYBRID;
  /// Number of conformers per pipeline mini-batch.
  int                                                          batchSize     = 0;

  /// Output mode selector. RDKIT_CONFORMERS writes positions and energies back
  /// into RDKit conformers and `moleculeEnergies` / `moleculeConverged`. DEVICE
  /// leaves positions/energies on the GPU and accumulates per-runner partials
  /// into `deviceCollectors` for later stitching by `finalizeOnTarget`.
  CoordinateOutput output = CoordinateOutput::RDKIT_CONFORMERS;

  /// Optional on-device starting coordinates. When non-null, each batch
  /// broadcasts these into its slot's positions buffer (via
  /// detail::broadcastDeviceInputBatch) after the initial host->device copy,
  /// using the matching `deviceInputIndex`. The (molIdx, confIdx) labels must
  /// match `allConformers` one-to-one (validated by the caller).
  const DeviceCoordResult*        deviceInput      = nullptr;
  const detail::DeviceInputIndex* deviceInputIndex = nullptr;

  /// Per-runner-thread device-output accumulators. The caller sizes this to
  /// the total number of runner threads (workerThreadsPerGpu * gpuIds.size()).
  /// Each runner claims a unique slot via `nextRunnerIdx` on first dispatch
  /// and appends its batch outputs there.
  std::vector<detail::DeviceCoordCollector>*  deviceCollectors = nullptr;
  /// Per-runner-thread long-lived streams used as collector streams. Sized
  /// alongside `deviceCollectors` and declared in the caller frame *before*
  /// it, so the streams outlive the collectors during destruction (the
  /// collectors' AsyncDeviceVectors free async on these streams).
  std::vector<std::unique_ptr<ScopedStream>>* runnerStreams    = nullptr;
  std::atomic<int>                            nextRunnerIdx{0};

  // RDKIT_CONFORMERS-mode result sinks (mutex-protected).
  std::vector<std::vector<double>>* moleculeEnergies  = nullptr;
  std::vector<std::vector<int8_t>>* moleculeConverged = nullptr;
  std::mutex*                       outputMutex       = nullptr;
};

/**
 * @brief One MMFF mini-batch: a contiguous slice of the flattened
 * conformer list plus the host-side batched system built for it.
 */
struct MmffBatch : gpu_scheduler::PreparedBatch {
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
 * @brief Per-runner-slot device state.
 *
 * Each slot owns one stream / completion event plus pinned host buffers
 * for D2H transfers; the BFGS minimizer and forcefield are constructed
 * fresh per batch (matching the pre-port code) but use this slot's stream.
 */
struct MmffSlot : gpu_scheduler::GpuSlotState {
  ScopedStream       stream;
  ScopedCudaEvent    completion;
  ThreadLocalBuffers buffers;

  cudaStream_t primaryStream() const override { return stream.stream(); }
  cudaEvent_t  completionEvent() const override { return completion.event(); }
};

/**
 * @brief Per-CUDA-device read-only state. Empty for MMFF; each batch is
 * self-contained (forcefield params live on the slot for the duration of
 * one dispatch).
 */
struct MmffPerGpu : gpu_scheduler::PerGpuState {
  int deviceId = -1;
};

/**
 * @brief Runtime-polymorphic MMFF workload consumed by gpu_scheduler::Pipeline.
 *
 * Lifecycle:
 *   - preprocess: takes a [start,end) slice of the flat conformer list,
 *     constructs MMFF forcefield contributions for each unique molecule
 *     (cached for the duration of the call), builds the
 *     BatchedMolecularSystemHost + metadata, and pushes one MmffBatch
 *     downstream.
 *   - dispatchAndCopyBack: uploads the host system, runs BFGS using the
 *     selected backend (BATCHED via MMFFBatchedForcefield, otherwise the
 *     per-molecule kernel path), and queues async D2H copies of positions,
 *     energies, and statuses. The slot's completion event is recorded
 *     after the copies, so postprocess sees ready data.
 *   - postprocess: writes positions back into the RDKit conformers and
 *     records per-conformer energies / convergence flags under outputMutex.
 */
class MmffWorkload : public gpu_scheduler::Workload {
 public:
  using Inputs        = MmffInputs;
  using PreparedBatch = MmffBatch;
  using PerGpuState   = MmffPerGpu;
  using GpuSlotState  = MmffSlot;
  /// No per-thread preproc state needed.
  struct PreprocThreadContext : gpu_scheduler::PreprocThreadContext {};
  /// Per-runner-thread context. In DEVICE output mode, `collectorIdx` holds
  /// the runner's exclusive slot into `Inputs.deviceCollectors`, claimed in
  /// `makeRunnerCtx` via `Inputs.nextRunnerIdx`.
  struct RunnerThreadContext : gpu_scheduler::RunnerThreadContext {
    int collectorIdx = -1;
  };

  explicit MmffWorkload(Inputs& inputs) : inputs_(inputs) {}

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

}  // namespace nvMolKit::MMFF

#endif  // NVMOLKIT_MMFF_WORKLOAD_H
