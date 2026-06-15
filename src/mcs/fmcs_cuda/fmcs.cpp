// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "fmcs_cuda/fmcs_debug.cuh"
#include "fmcs_cuda/fmcs_launch.cuh"
#include "fmcs_cuda/fmcs_match_tables.cuh"
#include "fmcs_cuda/fmcs_policy.cuh"
#include "fmcs_cuda/fmcs_tiers.cuh"

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

#include "src/utils/device.h"
#include "src/utils/device_vector.h"
#include "src/utils/gpu_executor_ring.h"
#include "src/utils/pinned_host_allocator.h"

namespace mcs {
namespace fmcs {

namespace {

void checkCuda(cudaError_t err, const char* context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("fMCS dispatch CUDA error at ")
                             + context + ": "
                             + cudaGetErrorString(err));
  }
}

constexpr size_t kTransferAlignment = 256;

size_t alignTransferOffset(const size_t offset) {
  return (offset + kTransferAlignment - 1) & ~(kTransferAlignment - 1);
}

void requireTransferCapacity(size_t required, size_t capacity, const char* context) {
  if (required > capacity) {
    throw std::runtime_error(std::string("fMCS ") + context +
                             " staging buffer is too small: required " +
                             std::to_string(required) + " bytes, capacity " +
                             std::to_string(capacity) + " bytes");
  }
}

struct FmcsOutputLayout {
  size_t resultsOffset = 0;
  size_t resultsBytes = 0;
  size_t elapsedOffset = 0;
  size_t elapsedBytes = 0;
  size_t statsOffset = 0;
  size_t statsBytes = 0;
  size_t totalBytes = 0;
};

struct FmcsExecutorBufferSizing {
  size_t inputBytes = 0;
  size_t outputBytes = 0;
};

struct StagedChunkInput {
  DevicePerPairInput* dPairInputs = nullptr;
  size_t bytes = 0;
  size_t numPairs = 0;
};

/// uint32 host-side mirror of a CSR @ref Graph plus packed bond metadata
/// matching @ref enumerateBonds ordering.
struct PackedGraphHost {
  std::vector<uint32_t> rowOffsets;
  std::vector<uint32_t> colIndices;
  /// Per-adjacency entry bond id, parallel to @c colIndices.
  std::vector<uint32_t> bondIndices;
  std::vector<uint32_t> bondEndpoints;  // (u << 16) | v, u < v
};

PackedGraphHost packGraph(const Graph& g) {
  constexpr uint32_t kUnsetBondIndex = 0xFFFFFFFFu;
  PackedGraphHost out;
  out.rowOffsets.reserve(g.rowOffsets.size());
  for (size_t v : g.rowOffsets) out.rowOffsets.push_back(static_cast<uint32_t>(v));
  out.colIndices.reserve(g.colIndices.size());
  for (size_t v : g.colIndices) out.colIndices.push_back(static_cast<uint32_t>(v));
  out.bondIndices.assign(g.colIndices.size(), kUnsetBondIndex);
  out.bondEndpoints.reserve(static_cast<size_t>(g.numEdges));
  for (int u = 0; u < g.numVertices; ++u) {
    const size_t begin = g.rowOffsets[u];
    const size_t end   = g.rowOffsets[u + 1];
    for (size_t k = begin; k < end; ++k) {
      const int v = static_cast<int>(g.colIndices[k]);
      if (u < v) {
        const auto bondIdx =
            static_cast<uint32_t>(out.bondEndpoints.size());
        out.bondEndpoints.push_back(
            (static_cast<uint32_t>(u) << 16) | static_cast<uint32_t>(v));
        out.bondIndices[k] = bondIdx;

        bool foundReverse = false;
        for (size_t rk = g.rowOffsets[v]; rk < g.rowOffsets[v + 1]; ++rk) {
          if (static_cast<int>(g.colIndices[rk]) == u &&
              out.bondIndices[rk] == kUnsetBondIndex) {
            out.bondIndices[rk] = bondIdx;
            foundReverse = true;
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
  return out;
}

/// Host-side descriptor assembled per pair.  @c queryGraph points at the
/// smaller input (fMCS enumerates subgraphs of the query), and
/// @c swapped records whether sides were flipped so host-side expansion
/// can un-swap the result mappings.
struct HostPairDescriptor {
  bool swapped = false;
  bool overflowed = false;

  const Graph* queryGraph = nullptr;
  const Graph* targetGraph = nullptr;

  PackedGraphHost packedQuery;
  PackedGraphHost packedTarget;
  PairMatchTablesHost tables;

  /// Maps to @ref pickMaxSizeTier: 0=16, 1=32, 2=64, 3=128; -1 on overflow.
  int tier = -1;
};

template<class Policy, class InputT>
HostPairDescriptor buildPairDescriptor(const InputT& sideA,
                                       const InputT& sideB,
                                       const Parameters& params) {
  HostPairDescriptor desc;
  const Graph& gA = [&]() -> const Graph& {
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
  desc.swapped = !pickA;
  const InputT& q = pickA ? sideA : sideB;
  const InputT& t = pickA ? sideB : sideA;
  const Graph& qG = pickA ? gA : gB;
  const Graph& tG = pickA ? gB : gA;

  desc.queryGraph  = &qG;
  desc.targetGraph = &tG;
  desc.packedQuery  = packGraph(qG);
  desc.packedTarget = packGraph(tG);

  Policy::buildAtomMatchTable(
      q, t, desc.tables.atoms, params.matchVertexLabels);
  Policy::buildBondMatchTable(
      q, t, desc.tables.bonds, params.matchEdgeLabels);

  const int tier = pickMaxSizeTier(
      std::max(qG.numVertices, tG.numVertices),
      std::max(qG.numEdges, tG.numEdges));
  desc.tier = tier;
  desc.overflowed = (tier < 0);
  return desc;
}

/// Owning device allocations for one tier sub-batch.  @c dResultsBuffer
/// and @c dQueueStorage are typed at launch time (per @c maxAtoms /
/// @c maxBonds), so they are held as opaque pointers here.  The queue
/// storage is one contiguous global-memory slab of
/// @c kFmcsQueueCapacity * numPairs QueuedSeed entries; per-block slices
/// are taken inside the kernel via @c blockIdx.x.
struct BatchDeviceBuffers {
  // Non-owning pointers into executor-persistent input/output buffers.
  DevicePerPairInput* dPairInputs = nullptr;
  void* dResultsBuffer  = nullptr;
  unsigned long long* dElapsedClocks = nullptr;
  ExecutionStats* dStats = nullptr;
  FmcsOutputLayout outputLayout;

  // Per-chunk scratch allocations.
  void* dQueueStorage   = nullptr;
  void* dCacheStorage   = nullptr;
  void* dSubstructureStorage = nullptr;
};

constexpr int kMaxFmcsExecutorsPerRunner = 8;

struct FmcsExecutor {
  std::unique_ptr<nvMolKit::ScopedStream> ownedStream;
  cudaStream_t                            stream = nullptr;
  nvMolKit::ScopedCudaEvent               copyDoneEvent;
  BatchDeviceBuffers                      bufs;
  nvMolKit::PinnedHostAllocator           inputStagingAllocator;
  nvMolKit::PinnedHostAllocator           outputStagingAllocator;
  nvMolKit::PinnedHostView<std::uint8_t>  inputStaging;
  nvMolKit::PinnedHostView<std::uint8_t>  outputStaging;
  nvMolKit::AsyncDeviceVector<std::uint8_t> inputDevice;
  nvMolKit::AsyncDeviceVector<std::uint8_t> outputDevice;

  FmcsExecutor(int executorIdx,
               cudaStream_t externalStream,
               bool useExternalStream,
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
    inputStaging = inputStagingAllocator.allocate<std::uint8_t>(bufferSizing.inputBytes);
    outputStaging = outputStagingAllocator.allocate<std::uint8_t>(bufferSizing.outputBytes);
    inputDevice.setStream(stream);
    outputDevice.setStream(stream);
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

void freeBatchBuffers(BatchDeviceBuffers& bufs, cudaStream_t stream) {
  if (bufs.dQueueStorage) {
    cudaFreeAsync(bufs.dQueueStorage, stream);
    bufs.dQueueStorage = nullptr;
  }
  if (bufs.dCacheStorage) {
    cudaFreeAsync(bufs.dCacheStorage, stream);
    bufs.dCacheStorage = nullptr;
  }
  if (bufs.dSubstructureStorage) {
    cudaFreeAsync(bufs.dSubstructureStorage, stream);
    bufs.dSubstructureStorage = nullptr;
  }
  bufs.dPairInputs = nullptr;
  bufs.dResultsBuffer = nullptr;
  bufs.dElapsedClocks = nullptr;
  bufs.dStats = nullptr;
  bufs.outputLayout = {};
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
    totalWords += d->packedTarget.rowOffsets.size();
    totalWords += d->packedTarget.colIndices.size();
    totalWords += d->packedTarget.bondIndices.size();
    totalWords += d->packedTarget.bondEndpoints.size();
  }
  return totalWords;
}

size_t computeChunkInputBytes(const std::vector<HostPairDescriptor*>& descs) {
  const size_t pairBytes = descs.size() * sizeof(DevicePerPairInput);
  return alignTransferOffset(pairBytes) +
         countChunkTransferWords(descs) * sizeof(std::uint32_t);
}

template<int maxAtoms, int maxBonds>
FmcsOutputLayout computeOutputLayout(size_t numPairs) {
  FmcsOutputLayout layout;
  size_t offset = 0;

  offset = alignTransferOffset(offset);
  layout.resultsOffset = offset;
  layout.resultsBytes =
      numPairs * sizeof(DeviceMCSResult<maxAtoms, maxBonds>);
  offset += layout.resultsBytes;

  layout.totalBytes = offset;
  return layout;
}

StagedChunkInput stageChunkInput(
    const std::vector<HostPairDescriptor*>& descs,
    nvMolKit::PinnedHostView<std::uint8_t>& hostStaging,
    nvMolKit::AsyncDeviceVector<std::uint8_t>& deviceStaging,
    cudaStream_t stream) {
  const size_t numPairs = descs.size();
  const size_t bytes = computeChunkInputBytes(descs);
  requireTransferCapacity(bytes, hostStaging.size(), "input host");
  requireTransferCapacity(bytes, deviceStaging.size(), "input device");

  auto* hostBase = hostStaging.data();
  auto* deviceBase = deviceStaging.data();
  auto* hostPairs = reinterpret_cast<DevicePerPairInput*>(hostBase);
  auto* devicePairs = reinterpret_cast<DevicePerPairInput*>(deviceBase);
  const size_t wordsOffset =
      alignTransferOffset(numPairs * sizeof(DevicePerPairInput));
  auto* hostWords = reinterpret_cast<std::uint32_t*>(hostBase + wordsOffset);
  auto* deviceWords = reinterpret_cast<std::uint32_t*>(deviceBase + wordsOffset);

  size_t cursor = 0;
  auto stageVec = [&](const std::vector<std::uint32_t>& v) -> const std::uint32_t* {
    if (v.empty()) return nullptr;
    std::uint32_t* hostDst = hostWords + cursor;
    std::memcpy(hostDst, v.data(), v.size() * sizeof(std::uint32_t));
    const std::uint32_t* deviceDst = deviceWords + cursor;
    cursor += v.size();
    return deviceDst;
  };

  auto stageTable = [&](const MatchTableHost& h) {
    MatchTableDevice d;
    d.data = stageVec(h.data);
    d.nRows = h.nRows;
    d.nCols = h.nCols;
    d.wordsPerRow = h.wordsPerRow;
    return d;
  };

  for (size_t i = 0; i < descs.size(); ++i) {
    const auto& d = *descs[i];
    DevicePerPairInput& p = hostPairs[i];
    p = DevicePerPairInput{};
    p.queryNumAtoms  = d.queryGraph->numVertices;
    p.queryNumBonds  = d.queryGraph->numEdges;
    p.targetNumAtoms = d.targetGraph->numVertices;
    p.targetNumBonds = d.targetGraph->numEdges;
    p.tables.atoms = stageTable(d.tables.atoms);
    p.tables.bonds = stageTable(d.tables.bonds);
    p.queryRowOffsets    = stageVec(d.packedQuery.rowOffsets);
    p.queryColIndices    = stageVec(d.packedQuery.colIndices);
    p.queryBondIndices   = stageVec(d.packedQuery.bondIndices);
    p.queryBondEndpoints = stageVec(d.packedQuery.bondEndpoints);
    p.targetRowOffsets    = stageVec(d.packedTarget.rowOffsets);
    p.targetColIndices    = stageVec(d.packedTarget.colIndices);
    p.targetBondIndices   = stageVec(d.packedTarget.bondIndices);
    p.targetBondEndpoints = stageVec(d.packedTarget.bondEndpoints);
    p.swapped  = d.swapped;
  }

  checkCuda(cudaMemcpyAsync(deviceBase,
                            hostBase,
                            bytes,
                            cudaMemcpyHostToDevice,
                            stream),
            "cudaMemcpyAsync (fMCS staged input)");
  return {devicePairs, bytes, numPairs};
}

template<int blockThreads, int maxAtoms, int maxBonds>
void launchTierAsync(
    const StagedChunkInput& stagedInput,
    const Parameters& params,
    cudaStream_t stream,
    BatchDeviceBuffers& bufs,
    nvMolKit::PinnedHostView<std::uint8_t>& hostOutput,
    nvMolKit::AsyncDeviceVector<std::uint8_t>& deviceOutput) {
  const int numPairs = static_cast<int>(stagedInput.numPairs);
  if (numPairs == 0) return;

  auto layout = computeOutputLayout<maxAtoms, maxBonds>(stagedInput.numPairs);
  requireTransferCapacity(layout.totalBytes, hostOutput.size(), "output host");
  requireTransferCapacity(layout.totalBytes, deviceOutput.size(), "output device");

  auto* outputBase = deviceOutput.data();
  auto* dResults = reinterpret_cast<DeviceMCSResult<maxAtoms, maxBonds>*>(
      outputBase + layout.resultsOffset);
  bufs.dPairInputs = stagedInput.dPairInputs;
  bufs.dResultsBuffer = dResults;
  bufs.outputLayout = layout;

  void* dQueue = nullptr;
  const size_t queueBytes =
      fmcsQueueStorageBytes<maxAtoms, maxBonds>(stagedInput.numPairs);
  checkCuda(cudaMallocAsync(&dQueue,
                            queueBytes, stream),
            "cudaMallocAsync (queue storage)");
  bufs.dQueueStorage = dQueue;

  std::uint8_t* dSubstructure = nullptr;
  const size_t substructureBytes =
      fmcsSubstructureStorageBytes<blockThreads, maxAtoms>(
          stagedInput.numPairs);
  checkCuda(cudaMallocAsync(reinterpret_cast<void**>(&dSubstructure),
                            substructureBytes, stream),
            "cudaMallocAsync (substructure storage)");
  bufs.dSubstructureStorage = dSubstructure;

  unsigned long long timeoutClocks = 0;
  if (params.timeoutMs > 0.0f) {
    int device = 0;
    checkCuda(cudaGetDevice(&device), "cudaGetDevice (timeout)");
    int clockRateKHz = 0;
    checkCuda(cudaDeviceGetAttribute(
                  &clockRateKHz, cudaDevAttrClockRate, device),
              "cudaDeviceGetAttribute (clock rate)");
    timeoutClocks = static_cast<unsigned long long>(
        std::max(1.0, static_cast<double>(params.timeoutMs) *
                          static_cast<double>(clockRateKHz)));
  }

  if constexpr (kFmcsDebug) {
    std::fprintf(stderr,
                 "[fmcs][host] launching tier maxAtoms=%d maxBonds=%d numPairs=%d\n",
                 maxAtoms, maxBonds, numPairs);
  }
  if constexpr (blockThreads == 128) {
    launchFmcsKernel128<maxAtoms, maxBonds>(
        bufs.dPairInputs, dResults, dQueue, dSubstructure,
        fmcsQueueCapacity(), fmcsSubstructurePartialCapacity(),
        numPairs, timeoutClocks, stream);
  } else if constexpr (blockThreads == 512) {
    launchFmcsKernel512<maxAtoms, maxBonds>(
        bufs.dPairInputs, dResults, dQueue, dSubstructure,
        fmcsQueueCapacity(), fmcsSubstructurePartialCapacity(),
        numPairs, timeoutClocks, stream);
  } else {
    static_assert(blockThreads == 128 || blockThreads == 512,
                  "fMCS block size must be 128 or 512");
  }
  checkCuda(cudaGetLastError(), "fmcsKernel launch");
  if constexpr (kFmcsDebug) {
    std::fprintf(stderr, "[fmcs][host] launch ok, synchronizing...\n");
  }

  checkCuda(cudaMemcpyAsync(hostOutput.data(), deviceOutput.data(),
                            layout.totalBytes,
                            cudaMemcpyDeviceToHost, stream),
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
template<int maxAtoms, int maxBonds>
MCSResult expandDeviceResult(
    const DeviceMCSResult<maxAtoms, maxBonds>& dr,
    const std::vector<std::uint32_t>& queryBondEndpoints,
    const std::vector<std::uint32_t>& targetBondEndpoints,
    bool swapped) {
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
    const std::uint32_t qPacked = queryBondEndpoints[dr.bondMapA[i]];
    const std::uint32_t tPacked = targetBondEndpoints[dr.bondMapB[i]];
    std::pair<size_t, size_t> eA{
        static_cast<size_t>(qPacked >> 16),
        static_cast<size_t>(qPacked & 0xFFFFu)};
    std::pair<size_t, size_t> eB{
        static_cast<size_t>(tPacked >> 16),
        static_cast<size_t>(tPacked & 0xFFFFu)};
    if (swapped) std::swap(eA, eB);
    r.edgeMappingA.push_back(eA);
    r.edgeMappingB.push_back(eB);
  }
  return r;
}

template<int maxAtoms, int maxBonds, class Policy>
struct TierChunk {
  std::vector<int>                 resultIndices;
  std::vector<HostPairDescriptor*> tierDescs;
  size_t inputBytes = 0;
};

template<class Policy>
using TierChunkVariant = std::variant<
    std::unique_ptr<TierChunk<16, 16, Policy>>,
    std::unique_ptr<TierChunk<32, 32, Policy>>,
    std::unique_ptr<TierChunk<64, 64, Policy>>,
    std::unique_ptr<TierChunk<128, 128, Policy>>>;

template<int maxAtoms, int maxBonds, class Policy>
std::unique_ptr<TierChunk<maxAtoms, maxBonds, Policy>> makeTierChunk(
    const std::vector<HostPairDescriptor*>& descs,
    const std::vector<int>& indices,
    size_t begin,
    size_t end) {
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

template<int maxAtoms, int maxBonds, class Policy>
void enqueueTierChunks(
    const std::vector<HostPairDescriptor*>& descs,
    const std::vector<int>& indices,
    size_t chunkSize,
    nvMolKit::ThreadSafeQueue<TierChunkVariant<Policy>>& chunkQueue,
    FmcsExecutorBufferSizing& bufferSizing) {
  if (indices.empty()) return;

  for (size_t begin = 0; begin < indices.size(); begin += chunkSize) {
    const size_t end = std::min(begin + chunkSize, indices.size());
    auto chunk = makeTierChunk<maxAtoms, maxBonds, Policy>(
        descs, indices, begin, end);
    bufferSizing.inputBytes = std::max(bufferSizing.inputBytes,
                                       chunk->inputBytes);
    bufferSizing.outputBytes = std::max(
        bufferSizing.outputBytes,
        computeOutputLayout<maxAtoms, maxBonds>(
            chunk->tierDescs.size()).totalBytes);
    TierChunkVariant<Policy> item = std::move(chunk);
    chunkQueue.push(std::move(item));
  }
}

template<int blockThreads, int maxAtoms, int maxBonds, class Policy>
void launchTierChunk(
    FmcsExecutor& executor,
    std::unique_ptr<TierChunk<maxAtoms, maxBonds, Policy>>& chunk,
    const Parameters& params) {
  cudaStream_t executorStream = executor.stream;
  try {
    StagedChunkInput stagedInput = stageChunkInput(
        chunk->tierDescs, executor.inputStaging, executor.inputDevice,
        executorStream);

    launchTierAsync<blockThreads, maxAtoms, maxBonds>(
        stagedInput,
        params,
        executorStream,
        executor.bufs,
        executor.outputStaging,
        executor.outputDevice);

    checkCuda(cudaEventRecord(executor.copyDoneEvent.event(), executorStream),
              "cudaEventRecord (fMCS chunk copy done)");
  } catch (...) {
    freeBatchBuffers(executor.bufs, executorStream);
    throw;
  }
}

template<int maxAtoms, int maxBonds, class Policy>
void drainTierChunk(
    FmcsExecutor& executor,
    std::unique_ptr<TierChunk<maxAtoms, maxBonds, Policy>>& chunk,
    std::vector<MCSResult>& outResults) {
  checkCuda(cudaEventSynchronize(executor.copyDoneEvent.event()),
            "cudaEventSynchronize (fMCS chunk copy done)");
  const auto& layout = executor.bufs.outputLayout;
  const auto* hostResults =
      reinterpret_cast<const DeviceMCSResult<maxAtoms, maxBonds>*>(
          executor.outputStaging.data() + layout.resultsOffset);

  for (size_t k = 0; k < chunk->resultIndices.size(); ++k) {
    outResults[chunk->resultIndices[k]] = expandDeviceResult<maxAtoms, maxBonds>(
        hostResults[k],
        chunk->tierDescs[k]->packedQuery.bondEndpoints,
        chunk->tierDescs[k]->packedTarget.bondEndpoints,
        chunk->tierDescs[k]->swapped);
  }
  freeBatchBuffers(executor.bufs, executor.stream);
}

template<int blockThreads, class Policy>
void runTierChunks(
    nvMolKit::ThreadSafeQueue<TierChunkVariant<Policy>>& chunkQueue,
    size_t numChunks,
    const FmcsExecutorBufferSizing& bufferSizing,
    const Parameters& params,
    cudaStream_t stream,
    std::vector<MCSResult>& outResults) {
  if (numChunks == 0) return;

  const int executorCount = static_cast<int>(
      std::min<size_t>(static_cast<size_t>(validateRequestedExecutorCount(params)), numChunks));
  if (executorCount > 1 && stream != nullptr) {
    throw std::invalid_argument("fMCS multi-executor dispatch does not support an external CUDA stream");
  }

  std::vector<std::unique_ptr<FmcsExecutor>> executorStorage;
  executorStorage.reserve(static_cast<size_t>(executorCount));
  std::vector<FmcsExecutor*> executors;
  executors.reserve(static_cast<size_t>(executorCount));
  const bool useExternalStream = executorCount == 1;
  for (int i = 0; i < executorCount; ++i) {
    auto executor = std::make_unique<FmcsExecutor>(
        i, stream, useExternalStream, bufferSizing);
    executors.push_back(executor.get());
    executorStorage.push_back(std::move(executor));
  }

  auto launchChunk = [&](FmcsExecutor& executor, TierChunkVariant<Policy>& chunk) {
    std::visit(
        [&](auto& typedChunk) {
          if (typedChunk) {
            using ChunkPtrT = std::decay_t<decltype(typedChunk)>;
            using Tier128ChunkPtr = std::unique_ptr<TierChunk<128, 128, Policy>>;
            if constexpr (blockThreads == 512 &&
                          std::is_same_v<ChunkPtrT, Tier128ChunkPtr>) {
              throw std::invalid_argument(
                  "fMCS blockSize 512 supports maxSize tiers up to 64");
            } else {
              launchTierChunk<blockThreads>(executor, typedChunk, params);
            }
          }
        },
        chunk);
  };

  auto drainChunk = [&](FmcsExecutor& executor, TierChunkVariant<Policy>& chunk) {
    std::visit(
        [&](auto& typedChunk) {
          if (typedChunk) {
            using ChunkPtrT = std::decay_t<decltype(typedChunk)>;
            using Tier128ChunkPtr = std::unique_ptr<TierChunk<128, 128, Policy>>;
            if constexpr (blockThreads == 512 &&
                          std::is_same_v<ChunkPtrT, Tier128ChunkPtr>) {
              throw std::invalid_argument(
                  "fMCS blockSize 512 supports maxSize tiers up to 64");
            } else {
              drainTierChunk(executor, typedChunk, outResults);
            }
          }
        },
        chunk);
  };

  nvMolKit::runQueuedExecutorRing(executors, chunkQueue, launchChunk, drainChunk);
}

template<int blockThreads, class Policy, class InputT>
std::vector<MCSResult> runBatchWithBlockSize(
    const std::vector<InputT>& a,
    const std::vector<InputT>& b,
    Parameters params,
    cudaStream_t stream) {
  if (a.size() != b.size()) {
    throw std::runtime_error(
        "fMCS batch: graphsA and graphsB must have equal length");
  }
  const size_t N = a.size();
  std::vector<MCSResult> results(N);
  if (N == 0) return results;

  std::vector<HostPairDescriptor> descs(N);
  std::array<std::vector<int>, 4> tierIndices;
  for (size_t i = 0; i < N; ++i) {
    descs[i] = buildPairDescriptor<Policy, InputT>(a[i], b[i], params);
    if (descs[i].overflowed) {
      MCSResult r;
      r.overflowed = true;
      results[i] = r;
      continue;
    }
    tierIndices[descs[i].tier].push_back(static_cast<int>(i));
  }
  if constexpr (blockThreads == 512) {
    if (!tierIndices[3].empty()) {
      throw std::invalid_argument(
          "fMCS blockSize 512 supports maxSize tiers up to 64");
    }
  }

  std::vector<HostPairDescriptor*> descPtrs(N);
  for (size_t i = 0; i < N; ++i) descPtrs[i] = &descs[i];

  const size_t chunkSize = params.batchSize > 0
      ? static_cast<size_t>(params.batchSize)
      : N;
  size_t numChunks = 0;
  for (const auto& indices : tierIndices) {
    if (!indices.empty()) {
      numChunks += (indices.size() + chunkSize - 1) / chunkSize;
    }
  }

  nvMolKit::ThreadSafeQueue<TierChunkVariant<Policy>> chunkQueue;
  FmcsExecutorBufferSizing bufferSizing;
  enqueueTierChunks<16, 16, Policy>(
      descPtrs, tierIndices[0], chunkSize, chunkQueue, bufferSizing);
  enqueueTierChunks<32, 32, Policy>(
      descPtrs, tierIndices[1], chunkSize, chunkQueue, bufferSizing);
  enqueueTierChunks<64, 64, Policy>(
      descPtrs, tierIndices[2], chunkSize, chunkQueue, bufferSizing);
  enqueueTierChunks<128, 128, Policy>(
      descPtrs, tierIndices[3], chunkSize, chunkQueue, bufferSizing);
  chunkQueue.close();
  runTierChunks<blockThreads, Policy>(
      chunkQueue, numChunks, bufferSizing, params, stream, results);

  return results;
}

template<class Policy, class InputT>
std::vector<MCSResult> runBatch(
    const std::vector<InputT>& a,
    const std::vector<InputT>& b,
    Parameters params,
    cudaStream_t stream) {
  switch (validateRequestedBlockSize(params)) {
    case 128:
      return runBatchWithBlockSize<128, Policy, InputT>(
          a, b, params, stream);
    case 512:
      return runBatchWithBlockSize<512, Policy, InputT>(
          a, b, params, stream);
  }
  throw std::logic_error("unreachable fMCS blockSize dispatch");
}

template<class Policy, class InputT>
std::vector<MCSResult> runBatchWithInstrumentation(
    const std::vector<InputT>& a,
    const std::vector<InputT>& b,
    Parameters params,
    std::vector<float>* perPairTimesMs,
    std::vector<ExecutionStats>* perPairStats,
    cudaStream_t stream) {
  if (perPairTimesMs != nullptr || perPairStats != nullptr) {
    throw std::runtime_error(
        "fMCS timing/stat instrumentation is not instantiated in this build");
  }

  return runBatch<Policy, InputT>(a, b, params, stream);
}

}  // namespace

std::vector<MCSResult> findMCESfMCSBatch(
    const std::vector<Graph>& graphsA,
    const std::vector<Graph>& graphsB,
    Parameters params,
    std::vector<float>* perPairTimesMs,
    cudaStream_t stream,
    std::vector<ExecutionStats>* perPairStats) {
  return runBatchWithInstrumentation<NullFMCSPolicy, Graph>(
      graphsA, graphsB, params, perPairTimesMs, perPairStats, stream);
}

std::vector<MCSResult> findMCESfMCSBatchLabeled(
    const std::vector<benchmark::MiviaGraphData>& graphsA,
    const std::vector<benchmark::MiviaGraphData>& graphsB,
    Parameters params,
    std::vector<float>* perPairTimesMs,
    cudaStream_t stream,
    std::vector<ExecutionStats>* perPairStats) {
  return runBatchWithInstrumentation<LabeledFMCSPolicy, benchmark::MiviaGraphData>(
      graphsA, graphsB, params, perPairTimesMs, perPairStats, stream);
}

}  // namespace fmcs
}  // namespace mcs
