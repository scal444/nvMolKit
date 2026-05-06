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
#include "mmff_batched_forcefield.h"
#include "mmff.h"
#include "mmff_properties.h"
#include "pipeline.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit

namespace nvMolKit::MMFF {

/**
 * @brief Inputs shared across all MMFF workload threads.
 *
 * Owned by `MMFFMinimizeMoleculesConfs` and held by reference inside
 * Pipeline<MmffWorkload>. The `mols` / `properties` / `constraints` fields
 * are read-only; the result vectors are written under `outputMutex` from
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
  int batchSize = 0;

  // Result sinks (mutex-protected).
  std::vector<std::vector<double>>* moleculeEnergies   = nullptr;
  std::vector<std::vector<int8_t>>* moleculeConverged  = nullptr;
  std::mutex*                       outputMutex        = nullptr;
};

/**
 * @brief One MMFF mini-batch: a contiguous slice of the flattened
 * conformer list plus the host-side batched system built for it.
 *
 * Built in preprocess and consumed in dispatchAndCopyBack.
 */
struct MmffBatch {
  std::vector<ConformerInfo>    conformers;
  std::vector<std::uint32_t>    conformerAtomStarts;
  BatchedMolecularSystemHost    systemHost;
  BatchedForcefieldMetadata     metadata;
  /// Per-conformer BFGS convergence statuses, populated by dispatch (queued
  /// async D2H on the slot stream) and consumed by postprocess.
  std::vector<int16_t>          statusesHost;
};

/**
 * @brief Per-runner-slot device state.
 *
 * Each slot owns one stream / completion event plus pinned host buffers
 * for D2H transfers; the BFGS minimizer and forcefield are constructed
 * fresh per batch (matching the pre-port code) but use this slot's stream.
 */
struct MmffSlot {
  ScopedStream       stream;
  ScopedCudaEvent    completion;
  ThreadLocalBuffers buffers;

  cudaStream_t primaryStream() const { return stream.stream(); }
  cudaEvent_t  completionEvent() const { return completion.event(); }
};

/**
 * @brief Per-CUDA-device read-only state. Empty for MMFF; each batch is
 * self-contained (forcefield params live on the slot for the duration of
 * one dispatch).
 */
struct MmffPerGpu {
  int deviceId = -1;
};

/**
 * @brief Trait struct wired into Pipeline<MmffWorkload>.
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
struct MmffWorkload {
  using Inputs               = MmffInputs;
  using PreparedBatch        = MmffBatch;
  using PerGpuState          = MmffPerGpu;
  using GpuSlotState         = MmffSlot;
  /// No per-thread preproc state needed.
  struct PreprocThreadContext {};
  /// No per-thread runner state needed (no fallback queue).
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

}  // namespace nvMolKit::MMFF

#include "mmff_workload_inl.h"

#endif  // NVMOLKIT_MMFF_WORKLOAD_H
