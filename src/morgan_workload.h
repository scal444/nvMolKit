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

#ifndef NVMOLKIT_MORGAN_WORKLOAD_H
#define NVMOLKIT_MORGAN_WORKLOAD_H

#include <GraphMol/ROMol.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "src/gpu_scheduler/cpu_fallback_queue.h"
#include "src/gpu_scheduler/pipeline.h"
#include "src/morgan_fingerprint_common.h"
#include "src/morgan_fingerprint_kernels.h"
#include "src/utils/device.h"
#include "src/utils/nvtx.h"

namespace nvMolKit {

/**
 * @brief Bucket key for Morgan mini-batches. Each bucket maps to a fixed
 * power-of-two atom/bond capacity and is dispatched to a different
 * MorganGPUBuffersBatch on a slot.
 */
enum class MorganBucket : int {
  kAtoms32  = 0,
  kAtoms64  = 1,
  kAtoms128 = 2,
};

namespace detail {

/**
 * @brief Map a flat unit index to a (bucket, sliceIndex) pair.
 *
 * Units are laid out as [32-bucket batches][64-bucket batches][128-bucket batches].
 */
std::pair<MorganBucket, int> unitToBucketSlice(int unitIdx, int numBatches32, int numBatches64, int numBatches128);

int bucketAtomCapacity(MorganBucket bucket);

}  // namespace detail

/**
 * @brief Inputs shared across all Morgan workload threads.
 *
 * Owned by `getFingerprintsCu` and held by reference inside
 * Pipeline<MorganWorkload<fpSize>>. Threads read most fields without
 * synchronization (read-only); the output accumulator is written via
 * indexed `cudaMemcpyAsync` from postprocess and is single-writer-per-index.
 */
template <int fpSize> struct MorganInputs {
  const std::vector<const RDKit::ROMol*>* mols   = nullptr;
  int                                     radius = 0;
  /// Pre-bucketed mol indices, computed by the caller before run().
  std::vector<int>                        work32;
  std::vector<int>                        work64;
  std::vector<int>                        work128;
  std::vector<int>                        workLarge;

  /// Maximum mols per mini-batch. Each "unit" handed to preprocess produces
  /// one mini-batch of up to this many mols from one bucket.
  int dispatchChunkSize = 0;

  /// Master output accumulator. Lives on the primary device (the one active
  /// when getFingerprintsCu was called) and on `primaryStream`.
  AsyncDeviceVector<FlatBitVect<fpSize>>* outputAccumulator = nullptr;
  cudaStream_t                            primaryStream     = nullptr;
  int                                     primaryDeviceId   = 0;

  /// Fallback queue of large-mol indices: those over the 128-atom budget that
  /// can't ride the GPU kernel and are processed via the RDKit CPU path.
  gpu_scheduler::CpuFallbackQueue<int>* fallbackQueue = nullptr;

  /// Derived schedule, populated by the caller before `Pipeline::run()`.
  int numMiniBatches32  = 0;
  int numMiniBatches64  = 0;
  int numMiniBatches128 = 0;
};

/**
 * @brief One mini-batch of Morgan work for a single bucket.
 *
 * Owns nothing: the pinned host buffers come from per-thread context, get
 * filled in preprocess, then sent to the GPU during dispatch. The slot
 * provides the device-side scratch (`MorganGPUBuffersBatch`).
 */
struct MorganBatch {
  MorganBucket     bucket          = MorganBucket::kAtoms32;
  /// Number of real mols. Host arrays are padded to dispatchChunkSize so the
  /// device-side spans line up with what the kernel expects, but only the
  /// first `scopedChunkSize` slots carry meaningful data.
  int              scopedChunkSize = 0;
  std::vector<int> molIndices;  // padded to dispatchChunkSize; trailing zeros

  // Host scratch, filled in preprocess, consumed in dispatch. All sized to
  // dispatchChunkSize * (atomCapacity factor).
  std::vector<std::uint32_t> atomInvariants;
  std::vector<std::uint32_t> bondInvariants;
  std::vector<std::int16_t>  bondIndices;
  std::vector<std::int16_t>  bondOtherAtomIndices;
  std::vector<std::int16_t>  nAtomsPerMol;
};

/**
 * @brief Per-runner-slot device state: streams, events, GPU scratch buffers.
 *
 * One slot per "in-flight" batch on a runner. Holds the bucket-specific
 * MorganGPUBuffersBatch instances and reuses them across batches.
 */
struct MorganSlot {
  ScopedStream    stream;
  ScopedCudaEvent completion;

  std::unique_ptr<MorganGPUBuffersBatch> gpuBuffers32;
  std::unique_ptr<MorganGPUBuffersBatch> gpuBuffers64;
  std::unique_ptr<MorganGPUBuffersBatch> gpuBuffers128;

  cudaStream_t primaryStream() const { return stream.stream(); }
  cudaEvent_t  completionEvent() const { return completion.event(); }
};

/**
 * @brief Per-CUDA-device read-only state. No persistent data is needed
 * beyond the device id; each slot owns its own scratch.
 */
struct MorganPerGpu {
  int deviceId = -1;
};

/**
 * @brief Per-runner-thread context: holds the fallback producer guard so the
 * fallback queue stays open as long as any runner thread is alive (and
 * therefore can still drain it during cudaEventSynchronize waits).
 */
struct MorganRunnerCtx {
  gpu_scheduler::FallbackProducerGuard<int> fallbackGuard;
};

/**
 * @brief Trait struct wired into `Pipeline<MorganWorkload<fpSize>>`.
 *
 * The fpSize is a template parameter because the GPU kernel and output
 * accumulator are templated on it. The caller (`getFingerprintsCu`)
 * dispatches on fpSize at runtime once and instantiates the right
 * pipeline.
 *
 * Lifecycle:
 *   - preprocess: pulls mols from the bucket implied by the unit index,
 *     computes invariants directly into the batch's host scratch, pushes
 *     the batch downstream.
 *   - dispatchAndCopyBack: copies host scratch into the slot's bucket-
 *     specific MorganGPUBuffersBatch and launches the kernel; the kernel
 *     writes directly into the master output accumulator using each
 *     mol's global index.
 *   - postprocess: nothing per-batch; large-mol fallback work is drained
 *     opportunistically here.
 */
template <int fpSize> struct MorganWorkload {
  using Inputs        = MorganInputs<fpSize>;
  using PreparedBatch = MorganBatch;
  using PerGpuState   = MorganPerGpu;
  using GpuSlotState  = MorganSlot;
  /// No per-thread preproc state; preprocess only consumes inputs and
  /// produces batches. The empty type keeps the pipeline contract uniform.
  struct PreprocThreadContext {};
  using RunnerThreadContext = MorganRunnerCtx;

  static int totalUnits(Inputs& inputs);
  static int unitsPerPreprocBatch(Inputs& inputs);

  static RunnerThreadContext makeRunnerCtx(Inputs& inputs);

  /// `preprocess` is the one trait method that has to be defined in the
  /// header: the pipeline instantiates it per-runner with a caller-supplied
  /// `PushFn` (a lambda type) that the pipeline owns. The non-template
  /// MorganWorkload<fpSize> methods live in morgan_workload.cu with
  /// explicit instantiations for the supported fpSize values.
  template <class PushFn>
  static void preprocess(Inputs& inputs, gpu_scheduler::IndexRange range, PreprocThreadContext&, PushFn pushBatch) {
    ScopedNvtxRange claimRange("MorganWorkload::preprocess");

    for (int unitIdx = range.start; unitIdx < range.end; ++unitIdx) {
      auto [bucket, sliceIdx] =
        detail::unitToBucketSlice(unitIdx, inputs.numMiniBatches32, inputs.numMiniBatches64, inputs.numMiniBatches128);
      const std::vector<int>* bucketWork = nullptr;
      switch (bucket) {
        case MorganBucket::kAtoms32:
          bucketWork = &inputs.work32;
          break;
        case MorganBucket::kAtoms64:
          bucketWork = &inputs.work64;
          break;
        case MorganBucket::kAtoms128:
          bucketWork = &inputs.work128;
          break;
      }

      const int chunkSize = inputs.dispatchChunkSize;
      const int start     = sliceIdx * chunkSize;
      const int end       = std::min(start + chunkSize, static_cast<int>(bucketWork->size()));
      if (start >= end) {
        continue;
      }
      const int numMolsInBatch = end - start;
      const int atomCapacity   = detail::bucketAtomCapacity(bucket);
      // The kernel reads `nAtomsPerMolArray.size()` to determine the batch
      // extent and only early-exits on `nAtomsPerMol[i] == 0`. Other device
      // buffers must therefore be sized to the same dispatchChunkSize so the
      // span sizes line up. Trailing entries are zeroed and harmless.
      const int paddedSize     = inputs.dispatchChunkSize;

      auto batch             = std::make_unique<MorganBatch>();
      batch->bucket          = bucket;
      batch->scopedChunkSize = numMolsInBatch;
      batch->molIndices.assign(bucketWork->begin() + start, bucketWork->begin() + end);
      // Pad mol indices to dispatchChunkSize so the H2D copy is uniform with
      // the other arrays. Trailing slots are unused (nAtoms==0 early-exits).
      batch->molIndices.resize(static_cast<size_t>(paddedSize), 0);

      batch->atomInvariants.assign(static_cast<size_t>(atomCapacity * paddedSize), 0);
      batch->bondInvariants.assign(static_cast<size_t>(atomCapacity * paddedSize), 0);
      batch->bondIndices.assign(static_cast<size_t>(atomCapacity * paddedSize * kMaxBondsPerAtom),
                                static_cast<std::int16_t>(-1));
      batch->bondOtherAtomIndices.assign(static_cast<size_t>(atomCapacity * paddedSize * kMaxBondsPerAtom),
                                         static_cast<std::int16_t>(-1));
      batch->nAtomsPerMol.assign(static_cast<size_t>(paddedSize), 0);

      std::vector<const RDKit::ROMol*> molsView;
      molsView.reserve(static_cast<size_t>(numMolsInBatch));
      for (int j = 0; j < numMolsInBatch; ++j) {
        const RDKit::ROMol* mol = (*inputs.mols)[batch->molIndices[j]];
        molsView.push_back(mol);
        batch->nAtomsPerMol[j] = static_cast<std::int16_t>(mol->getNumAtoms());
      }

      {
        ScopedNvtxRange invarsRange("Compute invariants");
        MorganInvariantsGenerator::ComputeInvariantsInto(molsView,
                                                         static_cast<size_t>(atomCapacity),
                                                         batch->atomInvariants.data(),
                                                         batch->bondInvariants.data(),
                                                         batch->bondIndices.data(),
                                                         batch->bondOtherAtomIndices.data());
      }

      pushBatch(std::move(batch));
    }
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

}  // namespace nvMolKit

#endif  // NVMOLKIT_MORGAN_WORKLOAD_H
