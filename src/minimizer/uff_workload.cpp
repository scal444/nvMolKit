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

#include "src/minimizer/uff_workload.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "src/conformer/ff_device_collect.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/nvtx.h"

namespace nvMolKit::UFF {

int UffWorkload::totalUnits() const {
  return static_cast<int>(inputs_.allConformers->size());
}

int UffWorkload::unitsPerPreprocBatch() const {
  return std::max(1, inputs_.batchSize);
}

std::unique_ptr<gpu_scheduler::RunnerThreadContext> UffWorkload::makeRunnerCtx() {
  auto ctx = std::make_unique<RunnerThreadContext>();
  if (inputs_.output == CoordinateOutput::DEVICE) {
    ctx->collectorIdx = inputs_.nextRunnerIdx.fetch_add(1, std::memory_order_relaxed);
  }
  return ctx;
}

void UffWorkload::preprocess(gpu_scheduler::IndexRange range,
                             gpu_scheduler::PreprocThreadContext& /*ctx*/,
                             const gpu_scheduler::PushBatch& pushBatch) {
  ScopedNvtxRange preprocRange("UffWorkload::preprocess");
  auto&           inputs = inputs_;

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

std::unique_ptr<gpu_scheduler::PerGpuState> UffWorkload::makePerGpuState(int gpuId) {
  auto pgs      = std::make_unique<UffPerGpu>();
  pgs->deviceId = gpuId;
  return pgs;
}

std::unique_ptr<gpu_scheduler::GpuSlotState> UffWorkload::makeSlotState(gpu_scheduler::PerGpuState& /*pgs*/,
                                                                        int /*gpuId*/) {
  return std::make_unique<UffSlot>();
}

void UffWorkload::dispatchAndCopyBack(gpu_scheduler::GpuSlotState&        slotBase,
                                      gpu_scheduler::PerGpuState&         pgsBase,
                                      gpu_scheduler::PreparedBatch&       batchBase,
                                      gpu_scheduler::RunnerThreadContext& ctxBase) {
  ScopedNvtxRange dispatchRange("UffWorkload::dispatchAndCopyBack");
  auto&           slot   = static_cast<GpuSlotState&>(slotBase);
  auto&           pgs    = static_cast<PerGpuState&>(pgsBase);
  auto&           batch  = static_cast<PreparedBatch&>(batchBase);
  auto&           ctx    = static_cast<RunnerThreadContext&>(ctxBase);
  auto&           inputs = inputs_;

  const cudaStream_t stream         = slot.primaryStream();
  const size_t       numAtomsTotal  = batch.systemHost.positions.size();
  const size_t       numConformers  = batch.conformers.size();
  const bool         deviceOutput   = inputs.output == CoordinateOutput::DEVICE;
  const bool         useDeviceInput = inputs.deviceInput != nullptr;

  // Lazily initialize this runner's collector on first dispatch. See the
  // matching block in mmff_workload.cpp for why we use a separate, longer-
  // lived stream rather than the slot stream.
  if (deviceOutput) {
    auto& collector = (*inputs.deviceCollectors)[ctx.collectorIdx];
    if (collector.gpuId < 0) {
      auto& runnerStream                 = (*inputs.runnerStreams)[ctx.collectorIdx];
      runnerStream                       = std::make_unique<ScopedStream>();
      const cudaStream_t collectorStream = runnerStream->stream();
      collector.gpuId                    = pgs.deviceId;
      collector.stream                   = collectorStream;
      collector.positions.setStream(collectorStream);
      collector.energies.setStream(collectorStream);
      collector.converged.setStream(collectorStream);
    }
  }

  slot.buffers.ensureCapacity(numAtomsTotal, numConformers);
  std::copy(batch.systemHost.positions.begin(),
            batch.systemHost.positions.end(),
            slot.buffers.initialPositions.begin());

  UFFBatchedForcefield      forcefield(batch.systemHost, batch.metadata, stream);
  AsyncDeviceVector<double> positionsDevice;
  AsyncDeviceVector<double> gradDevice;
  AsyncDeviceVector<double> energyOutsDevice;
  positionsDevice.setStream(stream);
  gradDevice.setStream(stream);
  energyOutsDevice.setStream(stream);
  positionsDevice.resize(numAtomsTotal);
  positionsDevice.copyFromHost(slot.buffers.initialPositions.data(), numAtomsTotal);
  if (useDeviceInput) {
    detail::broadcastDeviceInputBatch(*inputs.deviceInput,
                                      *inputs.deviceInputIndex,
                                      batch.batchSrcIndices,
                                      batch.batchAtomCounts,
                                      pgs.deviceId,
                                      stream,
                                      positionsDevice);
  }
  gradDevice.resize(numAtomsTotal);
  gradDevice.zero();
  energyOutsDevice.resize(numConformers);
  energyOutsDevice.zero();

  BfgsBatchMinimizer bfgsMinimizer(/*dataDim=*/3, DebugLevel::NONE, true, stream, BfgsBackend::BATCHED);
  bfgsMinimizer.minimize(inputs.maxIters, inputs.gradTol, forcefield, positionsDevice, gradDevice, energyOutsDevice);

  if (deviceOutput) {
    detail::appendBatch(batch.conformers,
                        positionsDevice,
                        energyOutsDevice,
                        bfgsMinimizer.statuses_,
                        (*inputs.deviceCollectors)[ctx.collectorIdx]);
  } else {
    positionsDevice.copyToHost(slot.buffers.positions.data(), numAtomsTotal);
    energyOutsDevice.copyToHost(slot.buffers.energies.data(), numConformers);

    batch.statusesHost.assign(numConformers, 0);
    bfgsMinimizer.statuses_.copyToHost(batch.statusesHost.data(), numConformers);
  }
}

void UffWorkload::postprocess(gpu_scheduler::GpuSlotState&  slotBase,
                              gpu_scheduler::PreparedBatch& batchBase,
                              gpu_scheduler::RunnerThreadContext& /*ctxBase*/) {
  auto& slot   = static_cast<GpuSlotState&>(slotBase);
  auto& batch  = static_cast<PreparedBatch&>(batchBase);
  auto& inputs = inputs_;
  if (inputs.output == CoordinateOutput::DEVICE) {
    return;
  }

  ScopedNvtxRange writebackRange("UffWorkload::postprocess");

  std::lock_guard<std::mutex> lock(*inputs.outputMutex);
  writeBackResults(batch.conformers, batch.conformerAtomStarts, slot.buffers, *inputs.moleculeEnergies);
  for (size_t i = 0; i < batch.conformers.size(); ++i) {
    const auto& confInfo                                           = batch.conformers[i];
    (*inputs.moleculeConverged)[confInfo.molIdx][confInfo.confIdx] = static_cast<int8_t>(batch.statusesHost[i] == 0);
  }
}

}  // namespace nvMolKit::UFF
