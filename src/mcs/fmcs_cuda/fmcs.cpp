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

#include "fmcs_cuda/fmcs.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "fmcs_cuda/fmcs_debug.cuh"
#include "fmcs_cuda/fmcs_launch.cuh"
#include "fmcs_cuda/fmcs_match_tables.cuh"
#include "fmcs_cuda/fmcs_policy.cuh"
#include "fmcs_cuda/fmcs_tiers.cuh"
#include "src/mcs/mcs_compile_flags.h"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"
#include "src/utils/gpu_executor_ring.h"
#include "src/utils/nvtx.h"
#include "src/utils/pinned_host_allocator.h"

namespace mcs {
namespace fmcs {

namespace {

void checkCuda(cudaError_t err, const char* context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("fMCS dispatch CUDA error at ") + context + ": " + cudaGetErrorString(err));
  }
}

constexpr size_t kTransferAlignment = 256;

size_t alignTransferOffset(const size_t offset) {
  return (offset + kTransferAlignment - 1) & ~(kTransferAlignment - 1);
}

void requireTransferCapacity(size_t required, size_t capacity, const char* context) {
  if (required > capacity) {
    throw std::runtime_error(std::string("fMCS ") + context + " staging buffer is too small: required " +
                             std::to_string(required) + " bytes, capacity " + std::to_string(capacity) + " bytes");
  }
}

/// Grow a per-executor scratch slab to at least @p bytes, reusing it when it is
/// already large enough.  Old contents are discarded (scratch is rewritten by
/// each kernel), and the old allocation is released before the new one so the
/// two never co-reside -- these slabs reach multiple GiB at large tiers.
/// Resolve the scratch-location policy for one (blockThreads, tier) pair.
/// Auto puts only 512 @ tier-128 in global memory (where static shared cannot
/// fit) and keeps every other config on fast shared memory.  Explicit Shared
/// at 512 @ tier-128 is rejected here rather than hitting a missing kernel
/// instantiation downstream.
template <int blockThreads, int maxAtoms> FmcsScratchLocation resolveScratchLocation(FmcsScratchLocation requested) {
  constexpr bool kTier128At512 = (blockThreads == 512 && maxAtoms == 128);
  if (requested == FmcsScratchLocation::Auto) {
    return kTier128At512 ? FmcsScratchLocation::Global : FmcsScratchLocation::Shared;
  }
  if (requested == FmcsScratchLocation::Shared && kTier128At512) {
    throw std::invalid_argument(
      "fMCS scratchLocation=shared cannot satisfy blockSize 512 at tier-128 "
      "(needs ~70 KB static shared > 48 KB); use scratchLocation=global or auto");
  }
  return requested;
}

void ensureScratchCapacity(nvMolKit::AsyncDeviceVector<std::uint8_t>& buffer, size_t bytes) {
  if (buffer.size() >= bytes) {
    return;
  }
  buffer.resize(0);
  buffer.resize(bytes);
}

struct FmcsOutputLayout {
  size_t resultsOffset     = 0;
  size_t resultsBytes      = 0;
  size_t elapsedOffset     = 0;
  size_t elapsedBytes      = 0;
  size_t timingStatsOffset = 0;
  size_t timingStatsBytes  = 0;
  size_t statsOffset       = 0;
  size_t statsBytes        = 0;
  size_t totalBytes        = 0;
};

struct FmcsExecutorBufferSizing {
  size_t inputBytes  = 0;
  size_t outputBytes = 0;
};

struct StagedChunkInput {
  DevicePerPairInput* dPairInputs = nullptr;
  size_t              bytes       = 0;
  size_t              numPairs    = 0;
};

/// uint32 host-side mirror of a CSR @ref Graph plus packed bond metadata
/// matching @ref enumerateBonds ordering.
struct PackedGraphHost {
  std::vector<uint32_t> rowOffsets;
  std::vector<uint32_t> colIndices;
  /// Per-adjacency entry bond id, parallel to @c colIndices.
  std::vector<uint32_t> bondIndices;
  std::vector<uint32_t> bondEndpoints;  // (u << 16) | v, u < v
  std::vector<uint32_t> ringBondFlags;  // 1 iff the bond belongs to a cycle
};

PackedGraphHost packGraph(const Graph& g) {
  constexpr uint32_t kUnsetBondIndex = 0xFFFFFFFFu;
  PackedGraphHost    out;
  out.rowOffsets.reserve(g.rowOffsets.size());
  for (size_t v : g.rowOffsets)
    out.rowOffsets.push_back(static_cast<uint32_t>(v));
  out.colIndices.reserve(g.colIndices.size());
  for (size_t v : g.colIndices)
    out.colIndices.push_back(static_cast<uint32_t>(v));
  out.bondIndices.assign(g.colIndices.size(), kUnsetBondIndex);
  out.bondEndpoints.reserve(static_cast<size_t>(g.numEdges));
  for (int u = 0; u < g.numVertices; ++u) {
    const size_t begin = g.rowOffsets[u];
    const size_t end   = g.rowOffsets[u + 1];
    for (size_t k = begin; k < end; ++k) {
      const int v = static_cast<int>(g.colIndices[k]);
      if (u < v) {
        const auto bondIdx = static_cast<uint32_t>(out.bondEndpoints.size());
        out.bondEndpoints.push_back((static_cast<uint32_t>(u) << 16) | static_cast<uint32_t>(v));
        out.bondIndices[k] = bondIdx;

        bool foundReverse = false;
        for (size_t rk = g.rowOffsets[v]; rk < g.rowOffsets[v + 1]; ++rk) {
          if (static_cast<int>(g.colIndices[rk]) == u && out.bondIndices[rk] == kUnsetBondIndex) {
            out.bondIndices[rk] = bondIdx;
            foundReverse        = true;
            break;
          }
        }
        if (!foundReverse) {
          throw std::runtime_error("fMCS CSR graph is missing reverse edge");
        }
      }
    }
  }
  if (out.bondEndpoints.size() != static_cast<size_t>(g.numEdges)) {
    throw std::runtime_error("fMCS CSR graph edge count is inconsistent");
  }
  out.ringBondFlags.assign(out.bondEndpoints.size(), 0);
  std::vector<unsigned char> visited(static_cast<size_t>(g.numVertices));
  std::vector<int>           stack;
  stack.reserve(static_cast<size_t>(g.numVertices));
  for (size_t excluded = 0; excluded < out.bondEndpoints.size(); ++excluded) {
    std::fill(visited.begin(), visited.end(), 0);
    stack.clear();
    const uint32_t endpoints           = out.bondEndpoints[excluded];
    const int      from                = static_cast<int>(endpoints >> 16);
    const int      goal                = static_cast<int>(endpoints & 0xFFFFu);
    visited[static_cast<size_t>(from)] = 1;
    stack.push_back(from);
    while (!stack.empty() && !visited[static_cast<size_t>(goal)]) {
      const int atom = stack.back();
      stack.pop_back();
      for (size_t k = g.rowOffsets[atom]; k < g.rowOffsets[atom + 1]; ++k) {
        if (out.bondIndices[k] == excluded)
          continue;
        const int next = static_cast<int>(g.colIndices[k]);
        if (!visited[static_cast<size_t>(next)]) {
          visited[static_cast<size_t>(next)] = 1;
          stack.push_back(next);
        }
      }
    }
    out.ringBondFlags[excluded] = visited[static_cast<size_t>(goal)] ? 1u : 0u;
  }
  return out;
}

/// Host-side descriptor assembled per pair.  @c queryGraph points at the
/// smaller input (fMCS enumerates subgraphs of the query), and
/// @c swapped records whether sides were flipped so host-side expansion
/// can un-swap the result mappings.
struct HostPairDescriptor {
  bool swapped           = false;
  bool overflowed        = false;
  bool completeRingsOnly = false;

  const Graph* queryGraph  = nullptr;
  const Graph* targetGraph = nullptr;

  PackedGraphHost     packedQuery;
  PackedGraphHost     packedTarget;
  PairMatchTablesHost tables;

  /// Maps to @ref pickMaxSizeTier: 0=16, 1=32, 2=64, 3=128; -1 on overflow.
  int tier = -1;
};

template <class Policy, class InputT>
HostPairDescriptor buildPairDescriptor(const InputT& sideA, const InputT& sideB, const Parameters& params) {
  HostPairDescriptor desc;
  desc.completeRingsOnly = params.completeRingsOnly;
  const Graph& gA        = [&]() -> const Graph& {
    if constexpr (std::is_same_v<InputT, Graph>) {
      return sideA;
    } else {
      return sideA.graph;
    }
  }();
  const Graph& gB = [&]() -> const Graph& {
    if constexpr (std::is_same_v<InputT, Graph>) {
      return sideB;
    } else {
      return sideB.graph;
    }
  }();

  const bool pickA = gA.numEdges <= gB.numEdges;
  desc.swapped     = !pickA;
  const InputT& q  = pickA ? sideA : sideB;
  const InputT& t  = pickA ? sideB : sideA;
  const Graph&  qG = pickA ? gA : gB;
  const Graph&  tG = pickA ? gB : gA;

  desc.queryGraph   = &qG;
  desc.targetGraph  = &tG;
  desc.packedQuery  = packGraph(qG);
  desc.packedTarget = packGraph(tG);

  Policy::buildAtomMatchTable(q, t, desc.tables.atoms, params.matchVertexLabels);
  Policy::buildBondMatchTable(q, t, desc.tables.bonds, params.matchEdgeLabels);

  const int tier  = pickMaxSizeTier(std::max(qG.numVertices, tG.numVertices), std::max(qG.numEdges, tG.numEdges));
  desc.tier       = tier;
  desc.overflowed = (tier < 0);
  return desc;
}

/// Non-owning device views for one tier sub-batch.  @c dResultsBuffer is
/// typed at launch time (per @c maxAtoms / @c maxBonds), so it is held as an
/// opaque pointer here.  All backing storage (inputs, results, queue, and
/// substructure-fallback scratch) lives in executor-persistent
/// @ref nvMolKit::AsyncDeviceVector buffers that are reused across chunks.
struct BatchDeviceBuffers {
  // Non-owning pointers into executor-persistent input/output buffers.
  DevicePerPairInput* dPairInputs    = nullptr;
  void*               dResultsBuffer = nullptr;
  unsigned long long* dElapsedClocks = nullptr;
  ExecutionStats*     dTimingStats   = nullptr;
  ExecutionStats*     dStats         = nullptr;
  FmcsOutputLayout    outputLayout;
};

constexpr int kMaxFmcsExecutorsPerRunner = 8;

/// Default per-tier chunk size used when the caller does not request an
/// explicit @c batchSize.  Per-pair device scratch (queue + substructure
/// fallback storage) is multiple megabytes, so an unbounded chunk of all
/// pairs exhausts device memory at large pair counts.  Bounding the chunk
/// keeps peak resident scratch independent of the total pair count.
constexpr size_t kDefaultFmcsChunkSize = 512;

struct FmcsExecutor {
  std::unique_ptr<nvMolKit::ScopedStream>   ownedStream;
  cudaStream_t                              stream = nullptr;
  nvMolKit::ScopedCudaEvent                 copyDoneEvent;
  BatchDeviceBuffers                        bufs;
  nvMolKit::PinnedHostAllocator             inputStagingAllocator;
  nvMolKit::PinnedHostAllocator             outputStagingAllocator;
  nvMolKit::PinnedHostView<std::uint8_t>    inputStaging;
  nvMolKit::PinnedHostView<std::uint8_t>    outputStaging;
  nvMolKit::AsyncDeviceVector<std::uint8_t> inputDevice;
  nvMolKit::AsyncDeviceVector<std::uint8_t> outputDevice;
  nvMolKit::AsyncDeviceVector<std::uint8_t> queueDevice;
  nvMolKit::AsyncDeviceVector<std::uint8_t> substructureDevice;
  // Global substructure-scratch slab; size 0 (unused) unless a tier resolves
  // to scratchLocation == Global.
  nvMolKit::AsyncDeviceVector<std::uint8_t> scratchDevice;

  FmcsExecutor(int                             executorIdx,
               cudaStream_t                    externalStream,
               bool                            useExternalStream,
               const FmcsExecutorBufferSizing& bufferSizing) {
    if (useExternalStream) {
      stream = externalStream;
    } else {
      const std::string streamName = "fmcs_executor_" + std::to_string(executorIdx);
      ownedStream                  = std::make_unique<nvMolKit::ScopedStream>(streamName.c_str());
      stream                       = ownedStream->stream();
    }

    inputStagingAllocator.preallocate(bufferSizing.inputBytes);
    outputStagingAllocator.preallocate(bufferSizing.outputBytes);
    inputStaging  = inputStagingAllocator.allocate<std::uint8_t>(bufferSizing.inputBytes);
    outputStaging = outputStagingAllocator.allocate<std::uint8_t>(bufferSizing.outputBytes);
    inputDevice.setStream(stream);
    outputDevice.setStream(stream);
    // Queue and substructure scratch are sized lazily on first use and grown
    // only when a larger chunk arrives, so an executor never reserves more than
    // the largest chunk it actually processes.
    queueDevice.setStream(stream);
    substructureDevice.setStream(stream);
    scratchDevice.setStream(stream);
    inputDevice.resize(bufferSizing.inputBytes);
    outputDevice.resize(bufferSizing.outputBytes);
  }
};

int validateRequestedExecutorCount(const Parameters& params) {
  if (params.executorsPerRunner < 1 || params.executorsPerRunner > kMaxFmcsExecutorsPerRunner) {
    throw std::invalid_argument("fMCS executorsPerRunner must be between 1 and " +
                                std::to_string(kMaxFmcsExecutorsPerRunner));
  }
  return params.executorsPerRunner;
}

int validateRequestedBlockSize(const Parameters& params) {
  if (params.blockSize == 128 || params.blockSize == 512) {
    return params.blockSize;
  }
  throw std::invalid_argument("fMCS blockSize must be one of 128 or 512");
}

size_t countChunkTransferWords(const std::vector<HostPairDescriptor*>& descs) {
  size_t totalWords = 0;
  for (auto* d : descs) {
    totalWords += d->tables.atoms.data.size();
    totalWords += d->tables.bonds.data.size();
    totalWords += d->packedQuery.rowOffsets.size();
    totalWords += d->packedQuery.colIndices.size();
    totalWords += d->packedQuery.bondIndices.size();
    totalWords += d->packedQuery.bondEndpoints.size();
    totalWords += d->packedQuery.ringBondFlags.size();
    totalWords += d->packedTarget.rowOffsets.size();
    totalWords += d->packedTarget.colIndices.size();
    totalWords += d->packedTarget.bondIndices.size();
    totalWords += d->packedTarget.bondEndpoints.size();
    totalWords += d->packedTarget.ringBondFlags.size();
  }
  return totalWords;
}

size_t computeChunkInputBytes(const std::vector<HostPairDescriptor*>& descs) {
  const size_t pairBytes = descs.size() * sizeof(DevicePerPairInput);
  return alignTransferOffset(pairBytes) + countChunkTransferWords(descs) * sizeof(std::uint32_t);
}

template <int maxAtoms, int maxBonds>
FmcsOutputLayout computeOutputLayout(size_t numPairs, bool collectTimings, bool collectStats) {
  FmcsOutputLayout layout;
  size_t           offset = 0;

  offset               = alignTransferOffset(offset);
  layout.resultsOffset = offset;
  layout.resultsBytes  = numPairs * sizeof(DeviceMCSResult<maxAtoms, maxBonds>);
  offset += layout.resultsBytes;

  if (collectTimings) {
    offset               = alignTransferOffset(offset);
    layout.elapsedOffset = offset;
    layout.elapsedBytes  = numPairs * sizeof(unsigned long long);
    offset += layout.elapsedBytes;

    if (!collectStats) {
      offset                   = alignTransferOffset(offset);
      layout.timingStatsOffset = offset;
      layout.timingStatsBytes  = numPairs * sizeof(ExecutionStats);
      offset += layout.timingStatsBytes;
    }
  }

  if (collectStats) {
    offset             = alignTransferOffset(offset);
    layout.statsOffset = offset;
    layout.statsBytes  = numPairs * sizeof(ExecutionStats);
    offset += layout.statsBytes;
  }

  layout.totalBytes = offset;
  return layout;
}

StagedChunkInput stageChunkInput(const std::vector<HostPairDescriptor*>&    descs,
                                 nvMolKit::PinnedHostView<std::uint8_t>&    hostStaging,
                                 nvMolKit::AsyncDeviceVector<std::uint8_t>& deviceStaging,
                                 cudaStream_t                               stream) {
  nvMolKit::ScopedNvtxRange stageRange("fMCS: Stage chunk input (H2D)");
  const size_t              numPairs = descs.size();
  const size_t              bytes    = computeChunkInputBytes(descs);
  requireTransferCapacity(bytes, hostStaging.size(), "input host");
  requireTransferCapacity(bytes, deviceStaging.size(), "input device");

  auto* hostBase   = hostStaging.data();
  auto* deviceBase = deviceStaging.data();
  std::memset(hostBase, 0, bytes);
  auto*        hostPairs   = reinterpret_cast<DevicePerPairInput*>(hostBase);
  auto*        devicePairs = reinterpret_cast<DevicePerPairInput*>(deviceBase);
  const size_t wordsOffset = alignTransferOffset(numPairs * sizeof(DevicePerPairInput));
  auto*        hostWords   = reinterpret_cast<std::uint32_t*>(hostBase + wordsOffset);
  auto*        deviceWords = reinterpret_cast<std::uint32_t*>(deviceBase + wordsOffset);

  size_t cursor   = 0;
  auto   stageVec = [&](const std::vector<std::uint32_t>& v) -> const std::uint32_t* {
    if (v.empty())
      return nullptr;
    std::uint32_t* hostDst = hostWords + cursor;
    std::memcpy(hostDst, v.data(), v.size() * sizeof(std::uint32_t));
    const std::uint32_t* deviceDst = deviceWords + cursor;
    cursor += v.size();
    return deviceDst;
  };

  auto stageTable = [&](const MatchTableHost& h) {
    MatchTableDevice d;
    d.data        = stageVec(h.data);
    d.nRows       = h.nRows;
    d.nCols       = h.nCols;
    d.wordsPerRow = h.wordsPerRow;
    return d;
  };

  for (size_t i = 0; i < descs.size(); ++i) {
    const auto&         d = *descs[i];
    DevicePerPairInput& p = hostPairs[i];
    p                     = DevicePerPairInput{};
    p.queryNumAtoms       = d.queryGraph->numVertices;
    p.queryNumBonds       = d.queryGraph->numEdges;
    p.targetNumAtoms      = d.targetGraph->numVertices;
    p.targetNumBonds      = d.targetGraph->numEdges;
    p.tables.atoms        = stageTable(d.tables.atoms);
    p.tables.bonds        = stageTable(d.tables.bonds);
    p.queryRowOffsets     = stageVec(d.packedQuery.rowOffsets);
    p.queryColIndices     = stageVec(d.packedQuery.colIndices);
    p.queryBondIndices    = stageVec(d.packedQuery.bondIndices);
    p.queryBondEndpoints  = stageVec(d.packedQuery.bondEndpoints);
    p.queryRingBondFlags  = stageVec(d.packedQuery.ringBondFlags);
    p.targetRowOffsets    = stageVec(d.packedTarget.rowOffsets);
    p.targetColIndices    = stageVec(d.packedTarget.colIndices);
    p.targetBondIndices   = stageVec(d.packedTarget.bondIndices);
    p.targetBondEndpoints = stageVec(d.packedTarget.bondEndpoints);
    p.targetRingBondFlags = stageVec(d.packedTarget.ringBondFlags);
    p.swapped             = d.swapped;
    p.completeRingsOnly   = d.completeRingsOnly;
  }

  checkCuda(cudaMemcpyAsync(deviceBase, hostBase, bytes, cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync (fMCS staged input)");
  return {devicePairs, bytes, numPairs};
}

template <int blockThreads, int maxAtoms, int maxBonds>
void launchTierAsync(const StagedChunkInput&                    stagedInput,
                     const Parameters&                          params,
                     cudaStream_t                               stream,
                     BatchDeviceBuffers&                        bufs,
                     nvMolKit::PinnedHostView<std::uint8_t>&    hostOutput,
                     nvMolKit::AsyncDeviceVector<std::uint8_t>& deviceOutput,
                     nvMolKit::AsyncDeviceVector<std::uint8_t>& queueStorage,
                     nvMolKit::AsyncDeviceVector<std::uint8_t>& substructureStorage,
                     nvMolKit::AsyncDeviceVector<std::uint8_t>& scratchStorage,
                     bool                                       collectTimings,
                     bool                                       collectStats) {
  const int numPairs = static_cast<int>(stagedInput.numPairs);
  if (numPairs == 0)
    return;

  const FmcsScratchLocation scratchLocation = resolveScratchLocation<blockThreads, maxAtoms>(params.scratchLocation);

  nvMolKit::ScopedNvtxRange launchRange("fMCS: Launch tier kernel maxAtoms=" + std::to_string(maxAtoms) +
                                        " pairs=" + std::to_string(numPairs));

  auto layout = computeOutputLayout<maxAtoms, maxBonds>(stagedInput.numPairs, collectTimings, collectStats);
  requireTransferCapacity(layout.totalBytes, hostOutput.size(), "output host");
  requireTransferCapacity(layout.totalBytes, deviceOutput.size(), "output device");

  auto* outputBase    = deviceOutput.data();
  auto* dResults      = reinterpret_cast<DeviceMCSResult<maxAtoms, maxBonds>*>(outputBase + layout.resultsOffset);
  bufs.dPairInputs    = stagedInput.dPairInputs;
  bufs.dResultsBuffer = dResults;
  bufs.dElapsedClocks =
    collectTimings ? reinterpret_cast<unsigned long long*>(outputBase + layout.elapsedOffset) : nullptr;
  bufs.dTimingStats =
    layout.timingStatsBytes > 0 ? reinterpret_cast<ExecutionStats*>(outputBase + layout.timingStatsOffset) : nullptr;
  bufs.dStats       = collectStats ? reinterpret_cast<ExecutionStats*>(outputBase + layout.statsOffset) : nullptr;
  bufs.outputLayout = layout;

  const size_t queueBytes = fmcsQueueStorageBytes<maxAtoms, maxBonds>(stagedInput.numPairs);
  ensureScratchCapacity(queueStorage, queueBytes);
  void* dQueue = queueStorage.data();

  const size_t substructureBytes = fmcsSubstructureStorageBytes<blockThreads, maxAtoms>(stagedInput.numPairs);
  ensureScratchCapacity(substructureStorage, substructureBytes);
  std::uint8_t* dSubstructure = substructureStorage.data();

  void* dScratch = nullptr;
  if (scratchLocation == FmcsScratchLocation::Global) {
    const size_t scratchBytes = fmcsScratchStorageBytes<blockThreads, maxAtoms, maxBonds>(stagedInput.numPairs);
    ensureScratchCapacity(scratchStorage, scratchBytes);
    dScratch = scratchStorage.data();
  }

  unsigned long long timeoutClocks = 0;
  if (params.timeoutMs > 0.0f) {
    int device = 0;
    checkCuda(cudaGetDevice(&device), "cudaGetDevice (timeout)");
    int clockRateKHz = 0;
    checkCuda(cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, device),
              "cudaDeviceGetAttribute (clock rate)");
    timeoutClocks = static_cast<unsigned long long>(
      std::max(1.0, static_cast<double>(params.timeoutMs) * static_cast<double>(clockRateKHz)));
  }

  if constexpr (kFmcsDebug) {
    std::fprintf(stderr,
                 "[fmcs][host] launching tier maxAtoms=%d maxBonds=%d numPairs=%d\n",
                 maxAtoms,
                 maxBonds,
                 numPairs);
  }
  if constexpr (blockThreads == 128) {
    launchFmcsKernel128<maxAtoms, maxBonds>(bufs.dPairInputs,
                                            dResults,
                                            dQueue,
                                            dSubstructure,
                                            dScratch,
                                            scratchLocation,
                                            bufs.dElapsedClocks,
                                            bufs.dTimingStats,
                                            bufs.dStats,
                                            fmcsQueueCapacity(),
                                            fmcsSubstructurePartialCapacity(),
                                            numPairs,
                                            timeoutClocks,
                                            stream);
  } else if constexpr (blockThreads == 512) {
    launchFmcsKernel512<maxAtoms, maxBonds>(bufs.dPairInputs,
                                            dResults,
                                            dQueue,
                                            dSubstructure,
                                            dScratch,
                                            scratchLocation,
                                            bufs.dElapsedClocks,
                                            bufs.dTimingStats,
                                            bufs.dStats,
                                            fmcsQueueCapacity(),
                                            fmcsSubstructurePartialCapacity(),
                                            numPairs,
                                            timeoutClocks,
                                            stream);
  } else {
    static_assert(blockThreads == 128 || blockThreads == 512, "fMCS block size must be 128 or 512");
  }
  checkCuda(cudaGetLastError(), "fmcsKernel launch");
  if constexpr (kFmcsDebug) {
    std::fprintf(stderr, "[fmcs][host] launch ok, synchronizing...\n");
  }

  checkCuda(cudaMemcpyAsync(hostOutput.data(), deviceOutput.data(), layout.totalBytes, cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync (fMCS staged output)");
  if constexpr (kFmcsDebug) {
    std::fprintf(stderr, "[fmcs][host] tier maxAtoms=%d copy scheduled\n", maxAtoms);
  }
}

/// Translate fixed-size DeviceMCSResult into MCSResult, un-swapping the
/// mappings if @p swapped is set.  The device result stores edges as
/// (queryBondIdx, targetBondIdx) pairs; we recover (u, v) endpoints
/// here by indexing the per-pair @c bondEndpoints arrays the host
/// already has.
template <int maxAtoms, int maxBonds>
MCSResult expandDeviceResult(const DeviceMCSResult<maxAtoms, maxBonds>& dr,
                             const std::vector<std::uint32_t>&          queryBondEndpoints,
                             const std::vector<std::uint32_t>&          targetBondEndpoints,
                             bool                                       swapped) {
  MCSResult r;
  r.numCommonVertices = dr.numCommonVertices;
  r.numCommonEdges    = dr.numCommonEdges;
  r.timedOut          = dr.timedOut;
  r.overflowed        = dr.overflowed;
  r.mappingA.reserve(static_cast<size_t>(dr.numCommonVertices));
  r.mappingB.reserve(static_cast<size_t>(dr.numCommonVertices));
  for (int i = 0; i < dr.numCommonVertices; ++i) {
    const size_t a = dr.mappingA[i];
    const size_t b = dr.mappingB[i];
    if (swapped) {
      r.mappingA.push_back(b);
      r.mappingB.push_back(a);
    } else {
      r.mappingA.push_back(a);
      r.mappingB.push_back(b);
    }
  }
  r.edgeMappingA.reserve(static_cast<size_t>(dr.numCommonEdges));
  r.edgeMappingB.reserve(static_cast<size_t>(dr.numCommonEdges));
  for (int i = 0; i < dr.numCommonEdges; ++i) {
    const std::uint32_t       qPacked = queryBondEndpoints[dr.bondMapA[i]];
    const std::uint32_t       tPacked = targetBondEndpoints[dr.bondMapB[i]];
    std::pair<size_t, size_t> eA{static_cast<size_t>(qPacked >> 16), static_cast<size_t>(qPacked & 0xFFFFu)};
    std::pair<size_t, size_t> eB{static_cast<size_t>(tPacked >> 16), static_cast<size_t>(tPacked & 0xFFFFu)};
    if (swapped)
      std::swap(eA, eB);
    r.edgeMappingA.push_back(eA);
    r.edgeMappingB.push_back(eB);
  }
  return r;
}

template <int maxAtoms, int maxBonds, class Policy> struct TierChunk {
  std::vector<int>                 resultIndices;
  std::vector<HostPairDescriptor*> tierDescs;
  size_t                           inputBytes = 0;
};

template <class Policy>
using TierChunkVariant = std::variant<std::unique_ptr<TierChunk<16, 16, Policy>>,
                                      std::unique_ptr<TierChunk<32, 32, Policy>>,
                                      std::unique_ptr<TierChunk<64, 64, Policy>>,
                                      std::unique_ptr<TierChunk<128, 128, Policy>>>;

template <int maxAtoms, int maxBonds, class Policy>
std::unique_ptr<TierChunk<maxAtoms, maxBonds, Policy>> makeTierChunk(const std::vector<HostPairDescriptor*>& descs,
                                                                     const std::vector<int>&                 indices,
                                                                     size_t                                  begin,
                                                                     size_t                                  end) {
  auto chunk = std::make_unique<TierChunk<maxAtoms, maxBonds, Policy>>();
  chunk->resultIndices.reserve(end - begin);
  chunk->tierDescs.reserve(end - begin);
  for (size_t pos = begin; pos < end; ++pos) {
    const int idx = indices[pos];
    chunk->resultIndices.push_back(idx);
    chunk->tierDescs.push_back(descs[idx]);
  }
  chunk->inputBytes = computeChunkInputBytes(chunk->tierDescs);
  return chunk;
}

template <int maxAtoms, int maxBonds, class Policy>
void enqueueTierChunks(const std::vector<HostPairDescriptor*>&              descs,
                       const std::vector<int>&                              indices,
                       size_t                                               chunkSize,
                       nvMolKit::ThreadSafeQueue<TierChunkVariant<Policy>>& chunkQueue,
                       FmcsExecutorBufferSizing&                            bufferSizing,
                       bool                                                 collectTimings,
                       bool                                                 collectStats) {
  if (indices.empty())
    return;

  for (size_t begin = 0; begin < indices.size(); begin += chunkSize) {
    const size_t end         = std::min(begin + chunkSize, indices.size());
    auto         chunk       = makeTierChunk<maxAtoms, maxBonds, Policy>(descs, indices, begin, end);
    bufferSizing.inputBytes  = std::max(bufferSizing.inputBytes, chunk->inputBytes);
    bufferSizing.outputBytes = std::max(
      bufferSizing.outputBytes,
      computeOutputLayout<maxAtoms, maxBonds>(chunk->tierDescs.size(), collectTimings, collectStats).totalBytes);
    TierChunkVariant<Policy> item = std::move(chunk);
    chunkQueue.push(std::move(item));
  }
}

template <int blockThreads, int maxAtoms, int maxBonds, class Policy>
void launchTierChunk(FmcsExecutor&                                           executor,
                     std::unique_ptr<TierChunk<maxAtoms, maxBonds, Policy>>& chunk,
                     const Parameters&                                       params,
                     bool                                                    collectTimings,
                     bool                                                    collectStats) {
  cudaStream_t     executorStream = executor.stream;
  StagedChunkInput stagedInput =
    stageChunkInput(chunk->tierDescs, executor.inputStaging, executor.inputDevice, executorStream);

  launchTierAsync<blockThreads, maxAtoms, maxBonds>(stagedInput,
                                                    params,
                                                    executorStream,
                                                    executor.bufs,
                                                    executor.outputStaging,
                                                    executor.outputDevice,
                                                    executor.queueDevice,
                                                    executor.substructureDevice,
                                                    executor.scratchDevice,
                                                    collectTimings,
                                                    collectStats);

  checkCuda(cudaEventRecord(executor.copyDoneEvent.event(), executorStream), "cudaEventRecord (fMCS chunk copy done)");
}

template <int maxAtoms, int maxBonds, class Policy>
void drainTierChunk(FmcsExecutor&                                           executor,
                    std::unique_ptr<TierChunk<maxAtoms, maxBonds, Policy>>& chunk,
                    std::vector<MCSResult>&                                 outResults,
                    std::vector<float>*                                     perPairTimesMs,
                    std::vector<ExecutionStats>*                            perPairTimingStats,
                    std::vector<ExecutionStats>*                            perPairStats,
                    float                                                   clockRateKHz) {
  nvMolKit::ScopedNvtxRange waitRange("Wait: fMCS chunk copy done", nvMolKit::NvtxColor::kRed);
  checkCuda(cudaEventSynchronize(executor.copyDoneEvent.event()), "cudaEventSynchronize (fMCS chunk copy done)");
  waitRange.pop();

  nvMolKit::ScopedNvtxRange expandRange("fMCS: Expand chunk results");
  const auto&               layout = executor.bufs.outputLayout;
  const auto*               hostResults =
    reinterpret_cast<const DeviceMCSResult<maxAtoms, maxBonds>*>(executor.outputStaging.data() + layout.resultsOffset);
  const auto* hostElapsedClocks =
    layout.elapsedBytes > 0 ?
      reinterpret_cast<const unsigned long long*>(executor.outputStaging.data() + layout.elapsedOffset) :
      nullptr;
  const auto* hostStats =
    layout.statsBytes > 0 ?
      reinterpret_cast<const ExecutionStats*>(executor.outputStaging.data() + layout.statsOffset) :
      nullptr;
  const auto* hostTimingStats =
    layout.timingStatsBytes > 0 ?
      reinterpret_cast<const ExecutionStats*>(executor.outputStaging.data() + layout.timingStatsOffset) :
      hostStats;

  for (size_t k = 0; k < chunk->resultIndices.size(); ++k) {
    const int resultIdx = chunk->resultIndices[k];
    outResults[static_cast<size_t>(resultIdx)] =
      expandDeviceResult<maxAtoms, maxBonds>(hostResults[k],
                                             chunk->tierDescs[k]->packedQuery.bondEndpoints,
                                             chunk->tierDescs[k]->packedTarget.bondEndpoints,
                                             chunk->tierDescs[k]->swapped);
    if (perPairTimesMs != nullptr && hostElapsedClocks != nullptr) {
      (*perPairTimesMs)[static_cast<size_t>(resultIdx)] =
        static_cast<float>(static_cast<double>(hostElapsedClocks[k]) / static_cast<double>(clockRateKHz));
    }
    if (perPairTimingStats != nullptr && hostTimingStats != nullptr) {
      (*perPairTimingStats)[static_cast<size_t>(resultIdx)] = hostTimingStats[k];
    }
    if (perPairStats != nullptr && hostStats != nullptr) {
      (*perPairStats)[static_cast<size_t>(resultIdx)] = hostStats[k];
    }
  }
  expandRange.pop();
}

template <int blockThreads, class Policy>
void runTierChunks(nvMolKit::ThreadSafeQueue<TierChunkVariant<Policy>>& chunkQueue,
                   size_t                                               numChunks,
                   const FmcsExecutorBufferSizing&                      bufferSizing,
                   const Parameters&                                    params,
                   cudaStream_t                                         stream,
                   std::vector<MCSResult>&                              outResults,
                   std::vector<float>*                                  perPairTimesMs,
                   std::vector<ExecutionStats>*                         perPairTimingStats,
                   std::vector<ExecutionStats>*                         perPairStats) {
  if (numChunks == 0)
    return;

  const bool collectTimings = perPairTimesMs != nullptr || perPairTimingStats != nullptr;
  const bool collectStats   = perPairStats != nullptr;
  float      clockRateKHz   = 0.0f;
  if (collectTimings) {
    int device = 0;
    checkCuda(cudaGetDevice(&device), "cudaGetDevice (timing conversion)");
    int clockRateKHzInt = 0;
    checkCuda(cudaDeviceGetAttribute(&clockRateKHzInt, cudaDevAttrClockRate, device),
              "cudaDeviceGetAttribute (timing conversion clock rate)");
    clockRateKHz = static_cast<float>(clockRateKHzInt);
  }

  const int executorCount =
    static_cast<int>(std::min<size_t>(static_cast<size_t>(validateRequestedExecutorCount(params)), numChunks));
  if (executorCount > 1 && stream != nullptr) {
    throw std::invalid_argument("fMCS multi-executor dispatch does not support an external CUDA stream");
  }

  nvMolKit::ScopedNvtxRange                  allocRange("fMCS: Allocate executors n=" + std::to_string(executorCount));
  std::vector<std::unique_ptr<FmcsExecutor>> executorStorage;
  executorStorage.reserve(static_cast<size_t>(executorCount));
  std::vector<FmcsExecutor*> executors;
  executors.reserve(static_cast<size_t>(executorCount));
  // A null `stream` is the legacy default stream, not a caller-supplied
  // stream: adopting it would funnel every worker thread onto one serializing
  // queue.  Only adopt an external stream when the caller actually passed one;
  // otherwise each executor gets its own non-blocking stream so concurrent
  // workers overlap on the device.
  const bool useExternalStream = executorCount == 1 && stream != nullptr;
  for (int i = 0; i < executorCount; ++i) {
    auto executor = std::make_unique<FmcsExecutor>(i, stream, useExternalStream, bufferSizing);
    executors.push_back(executor.get());
    executorStorage.push_back(std::move(executor));
  }
  allocRange.pop();

  // Tier-128 at blockSize 512 is supported via global substructure scratch
  // (resolved per tier in launchTierAsync), so no tier is gated here.
  auto launchChunk = [&](FmcsExecutor& executor, TierChunkVariant<Policy>& chunk) {
    std::visit(
      [&](auto& typedChunk) {
        if (typedChunk) {
          launchTierChunk<blockThreads>(executor, typedChunk, params, collectTimings, collectStats);
        }
      },
      chunk);
  };

  auto drainChunk = [&](FmcsExecutor& executor, TierChunkVariant<Policy>& chunk) {
    std::visit(
      [&](auto& typedChunk) {
        if (typedChunk) {
          drainTierChunk(executor,
                         typedChunk,
                         outResults,
                         perPairTimesMs,
                         perPairTimingStats,
                         perPairStats,
                         clockRateKHz);
        }
      },
      chunk);
  };

  nvMolKit::runQueuedExecutorRing(executors, chunkQueue, launchChunk, drainChunk);
}

template <int blockThreads, class Policy, class InputT>
std::vector<MCSResult> runBatchWithBlockSize(const std::vector<InputT>&   a,
                                             const std::vector<InputT>&   b,
                                             Parameters                   params,
                                             std::vector<float>*          perPairTimesMs,
                                             std::vector<ExecutionStats>* perPairTimingStats,
                                             std::vector<ExecutionStats>* perPairStats,
                                             cudaStream_t                 stream) {
  if (a.size() != b.size()) {
    throw std::runtime_error("fMCS batch: graphsA and graphsB must have equal length");
  }
  const size_t           N = a.size();
  std::vector<MCSResult> results(N);
  if (perPairTimesMs != nullptr) {
    perPairTimesMs->assign(N, 0.0f);
  }
  if (perPairTimingStats != nullptr) {
    perPairTimingStats->assign(N, ExecutionStats{});
  }
  if (perPairStats != nullptr) {
    perPairStats->assign(N, ExecutionStats{});
  }
  if (N == 0)
    return results;

  nvMolKit::ScopedNvtxRange       descRange("fMCS: Build pair descriptors N=" + std::to_string(N));
  std::vector<HostPairDescriptor> descs(N);
  std::array<std::vector<int>, 4> tierIndices;
  for (size_t i = 0; i < N; ++i) {
    descs[i] = buildPairDescriptor<Policy, InputT>(a[i], b[i], params);
    if (descs[i].overflowed) {
      MCSResult r;
      r.overflowed = true;
      results[i]   = r;
      continue;
    }
    tierIndices[descs[i].tier].push_back(static_cast<int>(i));
  }
  descRange.pop();
  if constexpr (blockThreads == 512) {
    // Tier-128 at 512 threads now runs via global substructure scratch
    // (resolved per tier in launchTierAsync).  Only reject the one combination
    // that cannot be satisfied: an explicit request for shared placement.
    if (!tierIndices[3].empty() && params.scratchLocation == FmcsScratchLocation::Shared) {
      throw std::invalid_argument(
        "fMCS scratchLocation=shared cannot satisfy blockSize 512 at tier-128 "
        "(needs ~70 KB static shared > 48 KB); use scratchLocation=global or auto");
    }
  }

  std::vector<HostPairDescriptor*> descPtrs(N);
  for (size_t i = 0; i < N; ++i)
    descPtrs[i] = &descs[i];

  const size_t chunkSize = params.batchSize > 0 ? static_cast<size_t>(params.batchSize) : kDefaultFmcsChunkSize;
  size_t       numChunks = 0;
  for (const auto& indices : tierIndices) {
    if (!indices.empty()) {
      numChunks += (indices.size() + chunkSize - 1) / chunkSize;
    }
  }

  nvMolKit::ScopedNvtxRange                           enqueueRange("fMCS: Enqueue tier chunks");
  nvMolKit::ThreadSafeQueue<TierChunkVariant<Policy>> chunkQueue;
  FmcsExecutorBufferSizing                            bufferSizing;
  const bool collectTimings = perPairTimesMs != nullptr || perPairTimingStats != nullptr;
  enqueueTierChunks<16, 16, Policy>(descPtrs,
                                    tierIndices[0],
                                    chunkSize,
                                    chunkQueue,
                                    bufferSizing,
                                    collectTimings,
                                    perPairStats != nullptr);
  enqueueTierChunks<32, 32, Policy>(descPtrs,
                                    tierIndices[1],
                                    chunkSize,
                                    chunkQueue,
                                    bufferSizing,
                                    collectTimings,
                                    perPairStats != nullptr);
  enqueueTierChunks<64, 64, Policy>(descPtrs,
                                    tierIndices[2],
                                    chunkSize,
                                    chunkQueue,
                                    bufferSizing,
                                    collectTimings,
                                    perPairStats != nullptr);
  enqueueTierChunks<128, 128, Policy>(descPtrs,
                                      tierIndices[3],
                                      chunkSize,
                                      chunkQueue,
                                      bufferSizing,
                                      collectTimings,
                                      perPairStats != nullptr);
  chunkQueue.close();
  enqueueRange.pop();

  nvMolKit::ScopedNvtxRange runRange("fMCS: Run tier chunks chunks=" + std::to_string(numChunks));
  runTierChunks<blockThreads, Policy>(chunkQueue,
                                      numChunks,
                                      bufferSizing,
                                      params,
                                      stream,
                                      results,
                                      perPairTimesMs,
                                      perPairTimingStats,
                                      perPairStats);

  return results;
}

template <class Policy, class InputT>
std::vector<MCSResult> runBatch(const std::vector<InputT>&   a,
                                const std::vector<InputT>&   b,
                                Parameters                   params,
                                std::vector<float>*          perPairTimesMs,
                                std::vector<ExecutionStats>* perPairTimingStats,
                                std::vector<ExecutionStats>* perPairStats,
                                cudaStream_t                 stream) {
  switch (validateRequestedBlockSize(params)) {
    case 128:
      return runBatchWithBlockSize<128, Policy, InputT>(a,
                                                        b,
                                                        params,
                                                        perPairTimesMs,
                                                        perPairTimingStats,
                                                        perPairStats,
                                                        stream);
    case 512:
      return runBatchWithBlockSize<512, Policy, InputT>(a,
                                                        b,
                                                        params,
                                                        perPairTimesMs,
                                                        perPairTimingStats,
                                                        perPairStats,
                                                        stream);
  }
  throw std::logic_error("unreachable fMCS blockSize dispatch");
}

template <class Policy, class InputT>
std::vector<MCSResult> runBatchWithInstrumentation(const std::vector<InputT>&   a,
                                                   const std::vector<InputT>&   b,
                                                   Parameters                   params,
                                                   std::vector<float>*          perPairTimesMs,
                                                   std::vector<ExecutionStats>* perPairStats,
                                                   std::vector<ExecutionStats>* perPairTimingStats,
                                                   cudaStream_t                 stream) {
  if ((perPairTimesMs != nullptr || perPairTimingStats != nullptr) && !nvMolKit::kMCSCollectTimingsEnabled) {
    throw std::runtime_error("fMCS timing instrumentation is not instantiated in this build");
  }
  if (perPairStats != nullptr && !nvMolKit::kMCSCollectStatsEnabled) {
    throw std::runtime_error("fMCS stat instrumentation is not instantiated in this build");
  }

  return runBatch<Policy, InputT>(a, b, params, perPairTimesMs, perPairTimingStats, perPairStats, stream);
}

}  // namespace

std::vector<MCSResult> findMCESfMCSBatch(const std::vector<Graph>&    graphsA,
                                         const std::vector<Graph>&    graphsB,
                                         Parameters                   params,
                                         std::vector<float>*          perPairTimesMs,
                                         cudaStream_t                 stream,
                                         std::vector<ExecutionStats>* perPairStats,
                                         std::vector<ExecutionStats>* perPairTimingStats) {
  nvMolKit::ScopedNvtxRange entryRange(
    "findMCESfMCSBatch N=" + std::to_string(graphsA.size()) + " block=" + std::to_string(params.blockSize),
    nvMolKit::NvtxColor::kCyan);
  return runBatchWithInstrumentation<UnlabeledFmcsPolicy, Graph>(graphsA,
                                                                 graphsB,
                                                                 params,
                                                                 perPairTimesMs,
                                                                 perPairStats,
                                                                 perPairTimingStats,
                                                                 stream);
}

std::vector<MCSResult> findMCESfMCSBatchLabeled(const std::vector<LabeledGraph>& graphsA,
                                                const std::vector<LabeledGraph>& graphsB,
                                                Parameters                       params,
                                                std::vector<float>*              perPairTimesMs,
                                                cudaStream_t                     stream,
                                                std::vector<ExecutionStats>*     perPairStats,
                                                std::vector<ExecutionStats>*     perPairTimingStats) {
  nvMolKit::ScopedNvtxRange entryRange(
    "findMCESfMCSBatchLabeled N=" + std::to_string(graphsA.size()) + " block=" + std::to_string(params.blockSize),
    nvMolKit::NvtxColor::kCyan);
  return runBatchWithInstrumentation<LabeledFmcsPolicy, LabeledGraph>(graphsA,
                                                                      graphsB,
                                                                      params,
                                                                      perPairTimesMs,
                                                                      perPairStats,
                                                                      perPairTimingStats,
                                                                      stream);
}

}  // namespace fmcs
}  // namespace mcs
