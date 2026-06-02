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

int UffWorkload::totalUnits(Inputs& inputs) {
  return static_cast<int>(inputs.allConformers->size());
}

int UffWorkload::unitsPerPreprocBatch(Inputs& inputs) {
  return std::max(1, inputs.batchSize);
}

UffWorkload::RunnerThreadContext UffWorkload::makeRunnerCtx(Inputs& inputs) {
  RunnerThreadContext ctx;
  if (inputs.output == CoordinateOutput::DEVICE) {
    ctx.collectorIdx = inputs.nextRunnerIdx.fetch_add(1, std::memory_order_relaxed);
  }
  return ctx;
}

std::unique_ptr<UffWorkload::PerGpuState> UffWorkload::makePerGpuState(Inputs& /*inputs*/, int gpuId) {
  auto pgs      = std::make_unique<UffPerGpu>();
  pgs->deviceId = gpuId;
  return pgs;
}

std::unique_ptr<UffWorkload::GpuSlotState> UffWorkload::makeSlotState(Inputs& /*inputs*/,
                                                                     PerGpuState& /*pgs*/,
                                                                     int /*gpuId*/) {
  return std::make_unique<UffSlot>();
}

void UffWorkload::dispatchAndCopyBack(GpuSlotState&        slot,
                                      PerGpuState&         pgs,
                                      PreparedBatch&       batch,
                                      Inputs&              inputs,
                                      RunnerThreadContext& ctx) {
  ScopedNvtxRange dispatchRange("UffWorkload::dispatchAndCopyBack");

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

void UffWorkload::postprocess(GpuSlotState&  slot,
                              PreparedBatch& batch,
                              Inputs&        inputs,
                              RunnerThreadContext& /*ctx*/) {
  if (inputs.output == CoordinateOutput::DEVICE) {
    return;
  }

  ScopedNvtxRange writebackRange("UffWorkload::postprocess");

  std::lock_guard<std::mutex> lock(*inputs.outputMutex);
  writeBackResults(batch.conformers, batch.conformerAtomStarts, slot.buffers, *inputs.moleculeEnergies);
  for (size_t i = 0; i < batch.conformers.size(); ++i) {
    const auto& confInfo                                          = batch.conformers[i];
    (*inputs.moleculeConverged)[confInfo.molIdx][confInfo.confIdx] =
      static_cast<int8_t>(batch.statusesHost[i] == 0);
  }
}

}  // namespace nvMolKit::UFF
