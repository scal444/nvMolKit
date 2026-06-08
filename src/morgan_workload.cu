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

#include <memory>
#include <stdexcept>
#include <string>

#include "src/morgan_fingerprint_kernels.h"
#include "src/morgan_workload.h"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit {

namespace detail {

std::pair<MorganBucket, int> unitToBucketSlice(int unitIdx, int numBatches32, int numBatches64, int /*numBatches128*/) {
  if (unitIdx < numBatches32) {
    return {MorganBucket::kAtoms32, unitIdx};
  }
  unitIdx -= numBatches32;
  if (unitIdx < numBatches64) {
    return {MorganBucket::kAtoms64, unitIdx};
  }
  unitIdx -= numBatches64;
  return {MorganBucket::kAtoms128, unitIdx};
}

int bucketAtomCapacity(MorganBucket bucket) {
  switch (bucket) {
    case MorganBucket::kAtoms32:
      return 32;
    case MorganBucket::kAtoms64:
      return 64;
    case MorganBucket::kAtoms128:
      return 128;
  }
  return 0;
}

}  // namespace detail

template <int fpSize> int MorganWorkload<fpSize>::totalUnits(Inputs& inputs) {
  return inputs.numMiniBatches32 + inputs.numMiniBatches64 + inputs.numMiniBatches128;
}

template <int fpSize> int MorganWorkload<fpSize>::unitsPerPreprocBatch(Inputs& /*inputs*/) {
  return 1;
}

template <int fpSize>
typename MorganWorkload<fpSize>::RunnerThreadContext MorganWorkload<fpSize>::makeRunnerCtx(Inputs& inputs) {
  RunnerThreadContext ctx;
  if (inputs.fallbackQueue != nullptr) {
    ctx.fallbackGuard = gpu_scheduler::FallbackProducerGuard<int>(inputs.fallbackQueue);
  }
  return ctx;
}

template <int fpSize>
std::unique_ptr<typename MorganWorkload<fpSize>::PerGpuState> MorganWorkload<fpSize>::makePerGpuState(
  Inputs& /*inputs*/,
  int gpuId) {
  auto pgs      = std::make_unique<MorganPerGpu>();
  pgs->deviceId = gpuId;
  return pgs;
}

template <int fpSize>
std::unique_ptr<typename MorganWorkload<fpSize>::GpuSlotState>
MorganWorkload<fpSize>::makeSlotState(Inputs& inputs, PerGpuState& /*pgs*/, int /*gpuId*/) {
  auto         slot   = std::make_unique<MorganSlot>();
  cudaStream_t stream = slot->stream.stream();

  slot->gpuBuffers32  = std::make_unique<MorganGPUBuffersBatch>();
  slot->gpuBuffers64  = std::make_unique<MorganGPUBuffersBatch>();
  slot->gpuBuffers128 = std::make_unique<MorganGPUBuffersBatch>();

  const size_t chunkSize = static_cast<size_t>(inputs.dispatchChunkSize);
  const int    radius    = inputs.radius;

  auto allocate = [&](MorganGPUBuffersBatch& buffers, int atomCapacity) {
    buffers.atomInvariants = AsyncDeviceVector<std::uint32_t>(chunkSize * static_cast<size_t>(atomCapacity), stream);
    buffers.bondInvariants = AsyncDeviceVector<std::uint32_t>(chunkSize * static_cast<size_t>(atomCapacity), stream);
    buffers.bondIndices    = AsyncDeviceVector<std::int16_t>(
      chunkSize * static_cast<size_t>(atomCapacity) * static_cast<size_t>(kMaxBondsPerAtom),
      stream);
    buffers.bondOtherAtomIndices = AsyncDeviceVector<std::int16_t>(
      chunkSize * static_cast<size_t>(atomCapacity) * static_cast<size_t>(kMaxBondsPerAtom),
      stream);
    buffers.nAtomsPerMol  = AsyncDeviceVector<std::int16_t>(chunkSize, stream);
    buffers.outputIndices = AsyncDeviceVector<int>(chunkSize, stream);
    switch (atomCapacity) {
      case 32:
        buffers.allSeenNeighborhoods32 =
          AsyncDeviceVector<FlatBitVect<32>>(chunkSize * 32 * static_cast<size_t>(radius + 1), stream);
        buffers.allSeenNeighborhoods32.zero();
        break;
      case 64:
        buffers.allSeenNeighborhoods64 =
          AsyncDeviceVector<FlatBitVect<64>>(chunkSize * 64 * static_cast<size_t>(radius + 1), stream);
        buffers.allSeenNeighborhoods64.zero();
        break;
      case 128:
        buffers.allSeenNeighborhoods128 =
          AsyncDeviceVector<FlatBitVect<128>>(chunkSize * 128 * static_cast<size_t>(radius + 1), stream);
        buffers.allSeenNeighborhoods128.zero();
        break;
      default:
        throw std::runtime_error("Unsupported atom capacity for Morgan slot: " + std::to_string(atomCapacity));
    }
  };

  allocate(*slot->gpuBuffers32, 32);
  allocate(*slot->gpuBuffers64, 64);
  allocate(*slot->gpuBuffers128, 128);

  return slot;
}

template <int fpSize>
void MorganWorkload<fpSize>::dispatchAndCopyBack(GpuSlotState& slot,
                                                 PerGpuState& /*pgs*/,
                                                 PreparedBatch& batch,
                                                 Inputs&        inputs,
                                                 RunnerThreadContext& /*ctx*/) {
  ScopedNvtxRange dispatchRange("MorganWorkload::dispatchAndCopyBack");
  cudaStream_t    stream       = slot.primaryStream();
  const int       atomCapacity = detail::bucketAtomCapacity(batch.bucket);

  MorganGPUBuffersBatch* buffers = nullptr;
  switch (batch.bucket) {
    case MorganBucket::kAtoms32:
      buffers = slot.gpuBuffers32.get();
      break;
    case MorganBucket::kAtoms64:
      buffers = slot.gpuBuffers64.get();
      break;
    case MorganBucket::kAtoms128:
      buffers = slot.gpuBuffers128.get();
      break;
  }

  // All device buffers were allocated to dispatchChunkSize; the host-side
  // batch arrays are padded to match (see preprocess). Copy the full padded
  // length so the spans the kernel sees match what it expects.
  const size_t paddedMols = static_cast<size_t>(inputs.dispatchChunkSize);
  const size_t bondTotal  = paddedMols * static_cast<size_t>(atomCapacity) * static_cast<size_t>(kMaxBondsPerAtom);

  buffers->atomInvariants.copyFromHost(batch.atomInvariants.data(), paddedMols * static_cast<size_t>(atomCapacity));
  buffers->bondInvariants.copyFromHost(batch.bondInvariants.data(), paddedMols * static_cast<size_t>(atomCapacity));
  buffers->bondIndices.copyFromHost(batch.bondIndices.data(), bondTotal);
  buffers->bondOtherAtomIndices.copyFromHost(batch.bondOtherAtomIndices.data(), bondTotal);
  buffers->nAtomsPerMol.copyFromHost(batch.nAtomsPerMol.data(), paddedMols);
  buffers->outputIndices.copyFromHost(batch.molIndices.data(), paddedMols);

  launchMorganFingerprintKernelBatch<fpSize>(*buffers,
                                             *inputs.outputAccumulator,
                                             static_cast<size_t>(inputs.radius),
                                             atomCapacity,
                                             batch.scopedChunkSize,
                                             stream);
}

template <int fpSize>
void MorganWorkload<fpSize>::postprocess(GpuSlotState& /*slot*/,
                                         PreparedBatch& /*batch*/,
                                         Inputs& inputs,
                                         RunnerThreadContext& /*ctx*/) {
  if (inputs.fallbackQueue != nullptr) {
    inputs.fallbackQueue->tryProcessOne();
  }
}

#define DEFINE_MORGAN_WORKLOAD(fpSize) template struct MorganWorkload<fpSize>;
DEFINE_MORGAN_WORKLOAD(128)
DEFINE_MORGAN_WORKLOAD(256)
DEFINE_MORGAN_WORKLOAD(512)
DEFINE_MORGAN_WORKLOAD(1024)
DEFINE_MORGAN_WORKLOAD(2048)
DEFINE_MORGAN_WORKLOAD(4096)
#undef DEFINE_MORGAN_WORKLOAD

}  // namespace nvMolKit
