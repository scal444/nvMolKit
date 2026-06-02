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

#include "src/minimizer/mmff_workload.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "src/conformer/ff_device_collect.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/nvtx.h"

namespace nvMolKit::MMFF {

int MmffWorkload::totalUnits(Inputs& inputs) {
  return static_cast<int>(inputs.allConformers->size());
}

int MmffWorkload::unitsPerPreprocBatch(Inputs& inputs) {
  return std::max(1, inputs.batchSize);
}

MmffWorkload::RunnerThreadContext MmffWorkload::makeRunnerCtx(Inputs& inputs) {
  RunnerThreadContext ctx;
  // Each runner thread claims one exclusive collector slot when the workload
  // is producing device-side output; the slot is initialized lazily on the
  // first dispatch (where the GPU stream is known).
  if (inputs.output == CoordinateOutput::DEVICE) {
    ctx.collectorIdx = inputs.nextRunnerIdx.fetch_add(1, std::memory_order_relaxed);
  }
  return ctx;
}

std::unique_ptr<MmffWorkload::PerGpuState> MmffWorkload::makePerGpuState(Inputs& /*inputs*/, int gpuId) {
  auto pgs      = std::make_unique<MmffPerGpu>();
  pgs->deviceId = gpuId;
  return pgs;
}

std::unique_ptr<MmffWorkload::GpuSlotState> MmffWorkload::makeSlotState(Inputs& /*inputs*/,
                                                                       PerGpuState& /*pgs*/,
                                                                       int /*gpuId*/) {
  return std::make_unique<MmffSlot>();
}

void MmffWorkload::dispatchAndCopyBack(GpuSlotState&        slot,
                                       PerGpuState&         pgs,
                                       PreparedBatch&       batch,
                                       Inputs&              inputs,
                                       RunnerThreadContext& ctx) {
  ScopedNvtxRange dispatchRange("MmffWorkload::dispatchAndCopyBack");

  const cudaStream_t stream         = slot.primaryStream();
  const size_t       numAtomsTotal  = batch.systemHost.positions.size();
  const size_t       numConformers  = batch.conformers.size();
  const bool         deviceOutput   = inputs.output == CoordinateOutput::DEVICE;
  const bool         useDeviceInput = inputs.deviceInput != nullptr;

  // Lazily initialize this runner's collector on first dispatch. The collector
  // gets its OWN long-lived stream (owned by Inputs.runnerStreams), distinct
  // from the slot stream. The slot stream dies when pipeline.run() returns,
  // but the collector and its buffers outlive that (they're freed by the
  // caller after finalizeOnTarget runs), so using the slot stream would
  // dangle. appendBatch synchronizes the slot stream's writes onto the
  // collector stream via cross-stream events.
  if (deviceOutput) {
    auto& collector = (*inputs.deviceCollectors)[ctx.collectorIdx];
    if (collector.gpuId < 0) {
      auto& runnerStream = (*inputs.runnerStreams)[ctx.collectorIdx];
      runnerStream       = std::make_unique<ScopedStream>();
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

  // Per-batch BFGS minimizer; its internal device buffers are stream-ordered
  // so the destructor's async frees ride after the D2H copies queued below.
  BfgsBatchMinimizer bfgsMinimizer(/*dataDim=*/3, DebugLevel::NONE, true, stream, inputs.backend);
  const auto         effectiveBackend = bfgsMinimizer.resolveBackend(batch.systemHost.indices.atomStarts);

  const AsyncDeviceVector<double>* finalPositions = nullptr;
  const AsyncDeviceVector<double>* finalEnergies  = nullptr;
  AsyncDeviceVector<double>        positionsDevice;
  AsyncDeviceVector<double>        gradDevice;
  AsyncDeviceVector<double>        energyOutsDevice;
  BatchedMolecularDeviceBuffers    systemDevice;

  if (effectiveBackend == BfgsBackend::BATCHED) {
    MMFFBatchedForcefield forcefield(batch.systemHost, batch.metadata, stream);
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

    bfgsMinimizer.minimize(inputs.maxIters, inputs.gradTol, forcefield, positionsDevice, gradDevice, energyOutsDevice);

    finalPositions = &positionsDevice;
    finalEnergies  = &energyOutsDevice;

    if (!deviceOutput) {
      positionsDevice.copyToHost(slot.buffers.positions.data(), numAtomsTotal);
      energyOutsDevice.copyToHost(slot.buffers.energies.data(), numConformers);
    }
  } else {
    // setStreams must run before sendContribsAndIndicesToDevice; the latter
    // queues memcpys onto each system-device buffer's stream, which only
    // becomes the slot stream after setStreams.
    setStreams(systemDevice, stream);
    sendContribsAndIndicesToDevice(batch.systemHost, systemDevice);
    allocateIntermediateBuffers(batch.systemHost, systemDevice);
    systemDevice.positions.resize(numAtomsTotal);
    systemDevice.positions.copyFromHost(slot.buffers.initialPositions.data(), numAtomsTotal);
    if (useDeviceInput) {
      detail::broadcastDeviceInputBatch(*inputs.deviceInput,
                                        *inputs.deviceInputIndex,
                                        batch.batchSrcIndices,
                                        batch.batchAtomCounts,
                                        pgs.deviceId,
                                        stream,
                                        systemDevice.positions);
    }
    systemDevice.grad.resize(numAtomsTotal);
    systemDevice.grad.zero();

    bfgsMinimizer.minimizeWithMMFF(inputs.maxIters, inputs.gradTol, batch.systemHost.indices.atomStarts, systemDevice);

    finalPositions = &systemDevice.positions;
    finalEnergies  = &systemDevice.energyOuts;

    if (!deviceOutput) {
      systemDevice.positions.copyToHost(slot.buffers.positions.data(), numAtomsTotal);
      systemDevice.energyOuts.copyToHost(slot.buffers.energies.data(), numConformers);
    }
  }

  if (deviceOutput) {
    // appendBatch downloads statuses and syncs the collector stream internally,
    // so we don't need to queue any host-visible copies before returning.
    detail::appendBatch(batch.conformers,
                        *finalPositions,
                        *finalEnergies,
                        bfgsMinimizer.statuses_,
                        (*inputs.deviceCollectors)[ctx.collectorIdx]);
  } else {
    // Stash the per-conformer convergence statuses on the batch so postprocess
    // can read them after the slot's completion event has fired. The download
    // uses the slot stream; the pipeline waits on the completion event before
    // calling postprocess so the data is host-visible by then.
    batch.statusesHost.assign(numConformers, 0);
    bfgsMinimizer.statuses_.copyToHost(batch.statusesHost.data(), numConformers);
  }
}

void MmffWorkload::postprocess(GpuSlotState&  slot,
                               PreparedBatch& batch,
                               Inputs&        inputs,
                               RunnerThreadContext& /*ctx*/) {
  // In DEVICE output mode dispatch already appended into the collector and
  // there is no host-side state to merge per batch.
  if (inputs.output == CoordinateOutput::DEVICE) {
    return;
  }

  ScopedNvtxRange writebackRange("MmffWorkload::postprocess");

  std::lock_guard<std::mutex> lock(*inputs.outputMutex);
  writeBackResults(batch.conformers, batch.conformerAtomStarts, slot.buffers, *inputs.moleculeEnergies);
  for (size_t i = 0; i < batch.conformers.size(); ++i) {
    const auto& confInfo                                          = batch.conformers[i];
    (*inputs.moleculeConverged)[confInfo.molIdx][confInfo.confIdx] =
      static_cast<int8_t>(batch.statusesHost[i] == 0);
  }
}

}  // namespace nvMolKit::MMFF
