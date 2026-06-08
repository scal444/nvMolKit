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
#include "src/gpu_scheduler/pipeline.h"
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
 * Pipeline<UffWorkload>. Per-molecule UFF parameters (`vdwThresholds`,
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
struct UffBatch {
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
struct UffSlot {
  ScopedStream       stream;
  ScopedCudaEvent    completion;
  ThreadLocalBuffers buffers;

  cudaStream_t primaryStream() const { return stream.stream(); }
  cudaEvent_t  completionEvent() const { return completion.event(); }
};

/**
 * @brief Per-CUDA-device read-only state. Empty for UFF.
 */
struct UffPerGpu {
  int deviceId = -1;
};

/**
 * @brief Trait struct wired into Pipeline<UffWorkload>.
 *
 * Lifecycle mirrors MmffWorkload: preprocess builds the host system from a
 * conformer slice (UFF rebuilds parameters per conformer because they
 * depend on the conformer id and per-molecule vdw / interfrag flags),
 * dispatch uploads + runs BFGS via UFFBatchedForcefield + queues async
 * D2H copies, postprocess writes positions and statuses back under the
 * shared outputMutex.
 */
struct UffWorkload {
  using Inputs        = UffInputs;
  using PreparedBatch = UffBatch;
  using PerGpuState   = UffPerGpu;
  using GpuSlotState  = UffSlot;
  /// No per-thread preproc state needed.
  struct PreprocThreadContext {};
  /// Per-runner-thread context. In DEVICE output mode, `collectorIdx` holds
  /// the runner's exclusive slot into `Inputs.deviceCollectors`, claimed in
  /// `makeRunnerCtx` via `Inputs.nextRunnerIdx`.
  struct RunnerThreadContext {
    int collectorIdx = -1;
  };

  static int totalUnits(Inputs& inputs);
  static int unitsPerPreprocBatch(Inputs& inputs);

  static RunnerThreadContext makeRunnerCtx(Inputs& inputs);

  /// `preprocess` is the one trait method that has to be defined in the
  /// header: the pipeline instantiates it per-runner with a caller-supplied
  /// `PushFn` (a lambda type) that the pipeline owns.
  template <class PushFn>
  static void preprocess(Inputs& inputs, gpu_scheduler::IndexRange range, PreprocThreadContext&, PushFn pushBatch) {
    ScopedNvtxRange preprocRange("UffWorkload::preprocess");

    auto batch = std::make_unique<UffBatch>();
    batch->conformers.assign(inputs.allConformers->begin() + range.start, inputs.allConformers->begin() + range.end);

    std::uint32_t       currentAtomOffset = 0;
    std::vector<double> pos;

    for (const auto& confInfo : batch->conformers) {
      const std::uint32_t numAtoms = confInfo.mol->getNumAtoms();
      batch->conformerAtomStarts.push_back(currentAtomOffset);
      currentAtomOffset += numAtoms;

      confPosToVect(*confInfo.conformer, pos);
      auto ffParams = constructForcefieldContribs(*confInfo.mol,
                                                  (*inputs.vdwThresholds)[confInfo.molIdx],
                                                  confInfo.conformerId,
                                                  (*inputs.ignoreInterfragInteractions)[confInfo.molIdx]);
      if (!inputs.constraints->empty()) {
        (*inputs.constraints)[confInfo.molIdx].applyTo(ffParams, pos);
      }
      addMoleculeToBatch(ffParams,
                         pos,
                         batch->systemHost,
                         batch->metadata,
                         confInfo.molIdx,
                         static_cast<int>(confInfo.confIdx));
    }

    if (inputs.deviceInput != nullptr) {
      batch->batchSrcIndices.reserve(batch->conformers.size());
      batch->batchAtomCounts.reserve(batch->conformers.size());
      for (size_t k = 0; k < batch->conformers.size(); ++k) {
        const size_t srcSlot = static_cast<size_t>(range.start) + k;
        batch->batchSrcIndices.push_back(inputs.deviceInputIndex->conformerIndexBy[srcSlot]);
        batch->batchAtomCounts.push_back(static_cast<int>(batch->conformers[k].mol->getNumAtoms()));
      }
    }

    pushBatch(std::move(batch));
  }

  static std::unique_ptr<PerGpuState>  makePerGpuState(Inputs& inputs, int gpuId);
  static std::unique_ptr<GpuSlotState> makeSlotState(Inputs& inputs, PerGpuState& pgs, int gpuId);

  static void dispatchAndCopyBack(GpuSlotState&        slot,
                                  PerGpuState&         pgs,
                                  PreparedBatch&       batch,
                                  Inputs&              inputs,
                                  RunnerThreadContext& ctx);

  static void postprocess(GpuSlotState& slot, PreparedBatch& batch, Inputs& inputs, RunnerThreadContext& ctx);
};

}  // namespace nvMolKit::UFF

#endif  // NVMOLKIT_UFF_WORKLOAD_H
