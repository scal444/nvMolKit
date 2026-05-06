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

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "bfgs_common.h"
#include "bfgs_minimize.h"
#include "bfgs_types.h"
#include "conformer_info.h"
#include "device.h"
#include "forcefield_constraints.h"
#include "pipeline.h"
#include "uff.h"
#include "uff_batched_forcefield.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit

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
  int batchSize = 0;

  // Result sinks (mutex-protected).
  std::vector<std::vector<double>>* moleculeEnergies  = nullptr;
  std::vector<std::vector<int8_t>>* moleculeConverged = nullptr;
  std::mutex*                       outputMutex       = nullptr;
};

/**
 * @brief One UFF mini-batch: a contiguous slice of the flattened
 * conformer list plus the host-side batched system built for it.
 *
 * Built in preprocess and consumed in dispatchAndCopyBack.
 */
struct UffBatch {
  std::vector<ConformerInfo>    conformers;
  std::vector<std::uint32_t>    conformerAtomStarts;
  BatchedMolecularSystemHost    systemHost;
  BatchedForcefieldMetadata     metadata;
  /// Per-conformer BFGS convergence statuses, populated by dispatch (queued
  /// async D2H on the slot stream) and consumed by postprocess.
  std::vector<int16_t>          statusesHost;
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
  using Inputs               = UffInputs;
  using PreparedBatch        = UffBatch;
  using PerGpuState          = UffPerGpu;
  using GpuSlotState         = UffSlot;
  /// No per-thread preproc state needed.
  struct PreprocThreadContext {};
  /// No per-thread runner state needed.
  struct RunnerThreadContext {};

  static int totalUnits(Inputs& inputs);
  static int unitsPerPreprocBatch(Inputs& inputs);

  template <class PushFn>
  static void preprocess(Inputs&                   inputs,
                         gpu_scheduler::IndexRange range,
                         PreprocThreadContext& /*ctx*/,
                         PushFn pushBatch);

  static std::unique_ptr<PerGpuState>  makePerGpuState(Inputs& inputs, int gpuId);
  static std::unique_ptr<GpuSlotState> makeSlotState(Inputs& inputs, PerGpuState& pgs, int gpuId);

  static void dispatchAndCopyBack(GpuSlotState&        slot,
                                  PerGpuState&         pgs,
                                  PreparedBatch&       batch,
                                  Inputs&              inputs,
                                  RunnerThreadContext& ctx);

  static void postprocess(GpuSlotState&        slot,
                          PreparedBatch&       batch,
                          Inputs&              inputs,
                          RunnerThreadContext& ctx);
};

}  // namespace nvMolKit::UFF

#include "uff_workload_inl.h"

#endif  // NVMOLKIT_UFF_WORKLOAD_H
