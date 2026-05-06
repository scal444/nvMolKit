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

#ifndef NVMOLKIT_UFF_WORKLOAD_INL_H
#define NVMOLKIT_UFF_WORKLOAD_INL_H

#include <GraphMol/ROMol.h>

#include <algorithm>
#include <utility>

#include "cuda_error_check.h"
#include "ff_utils.h"
#include "nvtx.h"
#include "uff_flattened_builder.h"

namespace nvMolKit::UFF {

inline int UffWorkload::totalUnits(Inputs& inputs) {
  return static_cast<int>(inputs.allConformers->size());
}

inline int UffWorkload::unitsPerPreprocBatch(Inputs& inputs) {
  return std::max(1, inputs.batchSize);
}

template <class PushFn>
void UffWorkload::preprocess(Inputs&                   inputs,
                             gpu_scheduler::IndexRange range,
                             PreprocThreadContext& /*ctx*/,
                             PushFn pushBatch) {
  ScopedNvtxRange preprocRange("UffWorkload::preprocess");

  auto batch = std::make_unique<UffBatch>();
  batch->conformers.assign(inputs.allConformers->begin() + range.start,
                           inputs.allConformers->begin() + range.end);

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

  pushBatch(std::move(batch));
}

inline std::unique_ptr<UffWorkload::PerGpuState>
UffWorkload::makePerGpuState(Inputs& /*inputs*/, int gpuId) {
  auto pgs      = std::make_unique<UffPerGpu>();
  pgs->deviceId = gpuId;
  return pgs;
}

inline std::unique_ptr<UffWorkload::GpuSlotState>
UffWorkload::makeSlotState(Inputs& /*inputs*/, PerGpuState& /*pgs*/, int /*gpuId*/) {
  return std::make_unique<UffSlot>();
}

inline void UffWorkload::dispatchAndCopyBack(GpuSlotState&        slot,
                                             PerGpuState&         /*pgs*/,
                                             PreparedBatch&       batch,
                                             Inputs&              inputs,
                                             RunnerThreadContext& /*ctx*/) {
  ScopedNvtxRange dispatchRange("UffWorkload::dispatchAndCopyBack");

  const cudaStream_t stream        = slot.primaryStream();
  const size_t       numAtomsTotal = batch.systemHost.positions.size();
  const size_t       numConformers = batch.conformers.size();

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
  gradDevice.resize(numAtomsTotal);
  gradDevice.zero();
  energyOutsDevice.resize(numConformers);
  energyOutsDevice.zero();

  BfgsBatchMinimizer bfgsMinimizer(/*dataDim=*/3, DebugLevel::NONE, true, stream, BfgsBackend::BATCHED);
  bfgsMinimizer.minimize(inputs.maxIters, inputs.gradTol, forcefield, positionsDevice, gradDevice, energyOutsDevice);

  positionsDevice.copyToHost(slot.buffers.positions.data(), numAtomsTotal);
  energyOutsDevice.copyToHost(slot.buffers.energies.data(), numConformers);

  batch.statusesHost.assign(numConformers, 0);
  bfgsMinimizer.statuses_.copyToHost(batch.statusesHost.data(), numConformers);
}

inline void UffWorkload::postprocess(GpuSlotState&        slot,
                                     PreparedBatch&       batch,
                                     Inputs&              inputs,
                                     RunnerThreadContext& /*ctx*/) {
  ScopedNvtxRange writebackRange("UffWorkload::postprocess");

  std::lock_guard<std::mutex> lock(*inputs.outputMutex);
  writeBackResults(batch.conformers, batch.conformerAtomStarts, slot.buffers, *inputs.moleculeEnergies);
  for (size_t i = 0; i < batch.conformers.size(); ++i) {
    const auto& confInfo                                  = batch.conformers[i];
    (*inputs.moleculeConverged)[confInfo.molIdx][confInfo.confIdx] =
      static_cast<int8_t>(batch.statusesHost[i] == 0);
  }
}

}  // namespace nvMolKit::UFF

#endif  // NVMOLKIT_UFF_WORKLOAD_INL_H
