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

#ifndef NVMOLKIT_MMFF_WORKLOAD_INL_H
#define NVMOLKIT_MMFF_WORKLOAD_INL_H

#include <GraphMol/ROMol.h>

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "cuda_error_check.h"
#include "ff_utils.h"
#include "mmff_flattened_builder.h"
#include "nvtx.h"

namespace nvMolKit::MMFF {

inline int MmffWorkload::totalUnits(Inputs& inputs) {
  return static_cast<int>(inputs.allConformers->size());
}

inline int MmffWorkload::unitsPerPreprocBatch(Inputs& inputs) {
  return std::max(1, inputs.batchSize);
}

template <class PushFn>
void MmffWorkload::preprocess(Inputs&                   inputs,
                              gpu_scheduler::IndexRange range,
                              PreprocThreadContext& /*ctx*/,
                              PushFn pushBatch) {
  ScopedNvtxRange preprocRange("MmffWorkload::preprocess");

  // Cached MMFF contributions per molecule, scoped to this preprocess call.
  // The same molecule may have multiple conformers in this batch; we build
  // its contributions once and clone-with-positions per conformer.
  struct CachedMoleculeData {
    EnergyForceContribsHost ffParams;
  };
  std::unordered_map<RDKit::ROMol*, CachedMoleculeData> moleculeCache;

  auto batch = std::make_unique<MmffBatch>();
  batch->conformers.assign(inputs.allConformers->begin() + range.start,
                           inputs.allConformers->begin() + range.end);

  std::uint32_t      currentAtomOffset = 0;
  std::vector<double> pos;

  for (const auto& confInfo : batch->conformers) {
    auto*               mol      = confInfo.mol;
    const std::uint32_t numAtoms = mol->getNumAtoms();

    auto it = moleculeCache.find(mol);
    if (it == moleculeCache.end()) {
      ScopedNvtxRange    cacheRange("Preprocess single molecule");
      CachedMoleculeData cached;
      cached.ffParams = constructForcefieldContribs(*mol, (*inputs.properties)[confInfo.molIdx]);
      it              = moleculeCache.insert({mol, std::move(cached)}).first;
    }

    batch->conformerAtomStarts.push_back(currentAtomOffset);
    currentAtomOffset += numAtoms;

    confPosToVect(*confInfo.conformer, pos);

    auto contribs = it->second.ffParams;
    if (!inputs.constraints->empty()) {
      (*inputs.constraints)[confInfo.molIdx].applyTo(contribs, pos);
    }
    addMoleculeToBatch(contribs, pos, batch->systemHost, &batch->metadata, confInfo.molIdx, confInfo.confIdx);
  }

  pushBatch(std::move(batch));
}

inline std::unique_ptr<MmffWorkload::PerGpuState>
MmffWorkload::makePerGpuState(Inputs& /*inputs*/, int gpuId) {
  auto pgs      = std::make_unique<MmffPerGpu>();
  pgs->deviceId = gpuId;
  return pgs;
}

inline std::unique_ptr<MmffWorkload::GpuSlotState>
MmffWorkload::makeSlotState(Inputs& /*inputs*/, PerGpuState& /*pgs*/, int /*gpuId*/) {
  return std::make_unique<MmffSlot>();
}

inline void MmffWorkload::dispatchAndCopyBack(GpuSlotState&        slot,
                                              PerGpuState&         /*pgs*/,
                                              PreparedBatch&       batch,
                                              Inputs&              inputs,
                                              RunnerThreadContext& /*ctx*/) {
  ScopedNvtxRange dispatchRange("MmffWorkload::dispatchAndCopyBack");

  const cudaStream_t stream         = slot.primaryStream();
  const size_t       numAtomsTotal  = batch.systemHost.positions.size();
  const size_t       numConformers  = batch.conformers.size();

  slot.buffers.ensureCapacity(numAtomsTotal, numConformers);
  std::copy(batch.systemHost.positions.begin(),
            batch.systemHost.positions.end(),
            slot.buffers.initialPositions.begin());

  // Per-batch BFGS minimizer; its internal device buffers are stream-ordered
  // so the destructor's async frees ride after the D2H copies queued below.
  BfgsBatchMinimizer bfgsMinimizer(/*dataDim=*/3, DebugLevel::NONE, true, stream, inputs.backend);
  const auto         effectiveBackend = bfgsMinimizer.resolveBackend(batch.systemHost.indices.atomStarts);

  if (effectiveBackend == BfgsBackend::BATCHED) {
    MMFFBatchedForcefield     forcefield(batch.systemHost, batch.metadata, stream);
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

    bfgsMinimizer.minimize(inputs.maxIters, inputs.gradTol, forcefield, positionsDevice, gradDevice, energyOutsDevice);

    positionsDevice.copyToHost(slot.buffers.positions.data(), numAtomsTotal);
    energyOutsDevice.copyToHost(slot.buffers.energies.data(), numConformers);
  } else {
    BatchedMolecularDeviceBuffers systemDevice;
    sendContribsAndIndicesToDevice(batch.systemHost, systemDevice);
    setStreams(systemDevice, stream);
    allocateIntermediateBuffers(batch.systemHost, systemDevice);
    systemDevice.positions.resize(numAtomsTotal);
    systemDevice.positions.copyFromHost(slot.buffers.initialPositions.data(), numAtomsTotal);
    systemDevice.grad.resize(numAtomsTotal);
    systemDevice.grad.zero();

    bfgsMinimizer.minimizeWithMMFF(inputs.maxIters, inputs.gradTol, batch.systemHost.indices.atomStarts, systemDevice);

    systemDevice.positions.copyToHost(slot.buffers.positions.data(), numAtomsTotal);
    systemDevice.energyOuts.copyToHost(slot.buffers.energies.data(), numConformers);
  }

  // Stash the per-conformer convergence statuses on the batch so postprocess
  // can read them after the slot's completion event has fired.
  batch.statusesHost.assign(numConformers, 0);
  bfgsMinimizer.statuses_.copyToHost(batch.statusesHost.data(), numConformers);
}

inline void MmffWorkload::postprocess(GpuSlotState&        slot,
                                      PreparedBatch&       batch,
                                      Inputs&              inputs,
                                      RunnerThreadContext& /*ctx*/) {
  ScopedNvtxRange writebackRange("MmffWorkload::postprocess");

  std::lock_guard<std::mutex> lock(*inputs.outputMutex);
  writeBackResults(batch.conformers, batch.conformerAtomStarts, slot.buffers, *inputs.moleculeEnergies);
  for (size_t i = 0; i < batch.conformers.size(); ++i) {
    const auto& confInfo                                  = batch.conformers[i];
    (*inputs.moleculeConverged)[confInfo.molIdx][confInfo.confIdx] =
      static_cast<int8_t>(batch.statusesHost[i] == 0);
  }
}

}  // namespace nvMolKit::MMFF

#endif  // NVMOLKIT_MMFF_WORKLOAD_INL_H
