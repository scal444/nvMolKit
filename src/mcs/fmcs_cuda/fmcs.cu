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
#include "fmcs_cuda/fmcs_kernel.cuh"
#include "fmcs_cuda/fmcs_match_tables.cuh"
#include "fmcs_cuda/fmcs_policy.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/utils/device.h"
#include "src/utils/gpu_executor_ring.h"

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

/// uint32 host-side mirror of a CSR @ref Graph plus a packed
/// @c bondEndpoints array matching @ref enumerateBonds ordering.
struct PackedGraphHost {
  std::vector<uint32_t> rowOffsets;
  std::vector<uint32_t> colIndices;
  std::vector<uint32_t> bondEndpoints;  // (u << 16) | v, u < v
};

PackedGraphHost packGraph(const Graph& g) {
  PackedGraphHost out;
  out.rowOffsets.reserve(g.rowOffsets.size());
  for (size_t v : g.rowOffsets) out.rowOffsets.push_back(static_cast<uint32_t>(v));
  out.colIndices.reserve(g.colIndices.size());
  for (size_t v : g.colIndices) out.colIndices.push_back(static_cast<uint32_t>(v));
  out.bondEndpoints.reserve(static_cast<size_t>(g.numEdges));
  for (int u = 0; u < g.numVertices; ++u) {
    const size_t begin = g.rowOffsets[u];
    const size_t end   = g.rowOffsets[u + 1];
    for (size_t k = begin; k < end; ++k) {
      const int v = static_cast<int>(g.colIndices[k]);
      if (u < v) {
        out.bondEndpoints.push_back(
            (static_cast<uint32_t>(u) << 16) | static_cast<uint32_t>(v));
      }
    }
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
  void* matchTablesBuffer = nullptr;
  size_t matchTablesBufferBytes = 0;
  void* csrBuffer = nullptr;
  size_t csrBufferBytes = 0;
  DevicePerPairInput* dPairInputs = nullptr;
  void* dResultsBuffer  = nullptr;
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

  FmcsExecutor(int executorIdx, cudaStream_t externalStream, bool useExternalStream) {
    if (useExternalStream) {
      stream = externalStream;
      return;
    }

    const std::string streamName = "fmcs_executor_" + std::to_string(executorIdx);
    ownedStream                  = std::make_unique<nvMolKit::ScopedStream>(streamName.c_str());
    stream                       = ownedStream->stream();
  }
};

int validateRequestedExecutorCount(const Parameters& params) {
  if (params.executorsPerRunner < 1 || params.executorsPerRunner > kMaxFmcsExecutorsPerRunner) {
    throw std::invalid_argument("fMCS executorsPerRunner must be between 1 and " +
                                std::to_string(kMaxFmcsExecutorsPerRunner));
  }
  return params.executorsPerRunner;
}

void freeBatchBuffers(BatchDeviceBuffers& bufs, cudaStream_t stream) {
  if (bufs.matchTablesBuffer) {
    freePairMatchTablesBuffer(bufs.matchTablesBuffer, stream);
    bufs.matchTablesBuffer = nullptr;
  }
  if (bufs.csrBuffer) {
    cudaFreeAsync(bufs.csrBuffer, stream);
    bufs.csrBuffer = nullptr;
  }
  if (bufs.dPairInputs) {
    cudaFreeAsync(bufs.dPairInputs, stream);
    bufs.dPairInputs = nullptr;
  }
  if (bufs.dResultsBuffer) {
    cudaFreeAsync(bufs.dResultsBuffer, stream);
    bufs.dResultsBuffer = nullptr;
  }
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
}

/// Upload every pair's CSR and bond-endpoint arrays into one contiguous
/// device buffer, returning per-pair descriptors whose pointers refer into
/// that buffer.
std::vector<DevicePerPairInput> uploadCsrAndAssemblePairInputs(
    const std::vector<HostPairDescriptor*>& descs,
    const std::vector<PairMatchTablesDevice>& tablesDev,
    cudaStream_t stream,
    void** outBuf,
    size_t* outBufBytes) {
  size_t totalWords = 0;
  for (auto* d : descs) {
    totalWords += d->packedQuery.rowOffsets.size();
    totalWords += d->packedQuery.colIndices.size();
    totalWords += d->packedQuery.bondEndpoints.size();
    totalWords += d->packedTarget.rowOffsets.size();
    totalWords += d->packedTarget.colIndices.size();
    totalWords += d->packedTarget.bondEndpoints.size();
  }

  void* buf = nullptr;
  if (totalWords > 0) {
    checkCuda(cudaMallocAsync(&buf, totalWords * sizeof(uint32_t), stream),
              "cudaMallocAsync (CSR)");
  }
  uint32_t* base = reinterpret_cast<uint32_t*>(buf);

  std::vector<DevicePerPairInput> out(descs.size());
  size_t cursor = 0;
  auto uploadVec = [&](const std::vector<uint32_t>& v) -> uint32_t* {
    if (v.empty()) return nullptr;
    uint32_t* dst = base + cursor;
    checkCuda(cudaMemcpyAsync(dst, v.data(), v.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync (CSR)");
    cursor += v.size();
    return dst;
  };

  for (size_t i = 0; i < descs.size(); ++i) {
    const auto& d = *descs[i];
    DevicePerPairInput& p = out[i];
    p.queryNumAtoms  = d.queryGraph->numVertices;
    p.queryNumBonds  = d.queryGraph->numEdges;
    p.targetNumAtoms = d.targetGraph->numVertices;
    p.targetNumBonds = d.targetGraph->numEdges;
    p.queryRowOffsets    = uploadVec(d.packedQuery.rowOffsets);
    p.queryColIndices    = uploadVec(d.packedQuery.colIndices);
    p.queryBondEndpoints = uploadVec(d.packedQuery.bondEndpoints);
    p.targetRowOffsets    = uploadVec(d.packedTarget.rowOffsets);
    p.targetColIndices    = uploadVec(d.packedTarget.colIndices);
    p.targetBondEndpoints = uploadVec(d.packedTarget.bondEndpoints);
    p.tables   = tablesDev[i];
    p.swapped  = d.swapped;
  }

  if (outBuf) *outBuf = buf;
  if (outBufBytes) *outBufBytes = totalWords * sizeof(uint32_t);
  return out;
}

template<int maxAtoms, int maxBonds, class Policy>
void launchTierAsync(
    const std::vector<DevicePerPairInput>& hostPairInputs,
    const Parameters& params,
    cudaStream_t stream,
    BatchDeviceBuffers& bufs,
    std::vector<DeviceMCSResult<maxAtoms, maxBonds>>& hostResults) {
  const int numPairs = static_cast<int>(hostPairInputs.size());
  if (numPairs == 0) return;

  checkCuda(cudaMallocAsync(reinterpret_cast<void**>(&bufs.dPairInputs),
                            numPairs * sizeof(DevicePerPairInput), stream),
            "cudaMallocAsync (pair inputs)");
  checkCuda(cudaMemcpyAsync(bufs.dPairInputs, hostPairInputs.data(),
                            numPairs * sizeof(DevicePerPairInput),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync (pair inputs)");

  DeviceMCSResult<maxAtoms, maxBonds>* dResults = nullptr;
  const size_t resultsBytes =
      static_cast<size_t>(numPairs) * sizeof(DeviceMCSResult<maxAtoms, maxBonds>);
  checkCuda(cudaMallocAsync(reinterpret_cast<void**>(&dResults),
                            resultsBytes, stream),
            "cudaMallocAsync (results)");
  bufs.dResultsBuffer = dResults;

  using QueuedT = QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>;
  QueuedT* dQueue = nullptr;
  const size_t queueBytes =
      static_cast<size_t>(numPairs) * kFmcsQueueCapacity * sizeof(QueuedT);
  checkCuda(cudaMallocAsync(reinterpret_cast<void**>(&dQueue),
                            queueBytes, stream),
            "cudaMallocAsync (queue storage)");
  bufs.dQueueStorage = dQueue;

  std::uint8_t* dSubstructure = nullptr;
  const size_t substructureBytes =
      static_cast<size_t>(numPairs) * 2u *
      static_cast<size_t>(kFmcsSubstructurePartialCapacity) *
      static_cast<size_t>(maxAtoms) * sizeof(std::uint8_t);
  checkCuda(cudaMallocAsync(reinterpret_cast<void**>(&dSubstructure),
                            substructureBytes, stream),
            "cudaMallocAsync (substructure storage)");
  bufs.dSubstructureStorage = dSubstructure;

  const uint32_t timeoutUs = params.timeoutMs > 0
      ? static_cast<uint32_t>(params.timeoutMs * 1000.0f)
      : 0u;

  // Block size is a compile-time constant (kFmcsBlockSize) so the
  // kernel's per-group state arrays can be sized statically.  The
  // host-side @c params.blockSize is currently advisory; v1 always
  // launches with kFmcsBlockSize.
  dim3 grid(static_cast<unsigned>(numPairs));
  dim3 block(static_cast<unsigned>(kFmcsBlockSize));
  if constexpr (kFmcsDebug) {
    std::fprintf(stderr,
                 "[fmcs][host] launching tier maxAtoms=%d maxBonds=%d numPairs=%d\n",
                 maxAtoms, maxBonds, numPairs);
  }
  fmcsKernel<maxAtoms, maxBonds, Policy>
      <<<grid, block, 0, stream>>>(
          bufs.dPairInputs, dResults,
          dQueue, nullptr, dSubstructure,
          kFmcsQueueCapacity, 0,
          kFmcsSubstructurePartialCapacity,
          numPairs, timeoutUs);
  checkCuda(cudaGetLastError(), "fmcsKernel launch");
  if constexpr (kFmcsDebug) {
    std::fprintf(stderr, "[fmcs][host] launch ok, synchronizing...\n");
  }

  hostResults.resize(static_cast<size_t>(numPairs));
  checkCuda(cudaMemcpyAsync(hostResults.data(), dResults, resultsBytes,
                            cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync (results)");
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
  std::vector<PairMatchTablesHost> hostTables;
  std::vector<DevicePerPairInput>  hostPairInputs;
  std::vector<DeviceMCSResult<maxAtoms, maxBonds>> hostResults;
};

template<int maxAtoms, int maxBonds, class Policy>
std::unique_ptr<TierChunk<maxAtoms, maxBonds, Policy>> makeTierChunk(
    const std::vector<HostPairDescriptor*>& descs,
    const std::vector<int>& indices,
    size_t begin,
    size_t end) {
  auto chunk = std::make_unique<TierChunk<maxAtoms, maxBonds, Policy>>();
  chunk->resultIndices.reserve(end - begin);
  chunk->tierDescs.reserve(end - begin);
  chunk->hostTables.reserve(end - begin);
  for (size_t pos = begin; pos < end; ++pos) {
    const int idx = indices[pos];
    chunk->resultIndices.push_back(idx);
    chunk->tierDescs.push_back(descs[idx]);
    chunk->hostTables.push_back(descs[idx]->tables);
  }
  return chunk;
}

template<int maxAtoms, int maxBonds, class Policy>
void runTier(
    const std::vector<HostPairDescriptor*>& descs,
    const std::vector<int>& indices,
    const Parameters& params,
    cudaStream_t stream,
    std::vector<MCSResult>& outResults) {
  if (indices.empty()) return;

  const size_t chunkSize = params.batchSize > 0
      ? static_cast<size_t>(params.batchSize)
      : indices.size();
  const size_t numChunks = (indices.size() + chunkSize - 1) / chunkSize;
  const int executorCount =
      static_cast<int>(std::min<size_t>(static_cast<size_t>(validateRequestedExecutorCount(params)), numChunks));
  if (executorCount > 1 && stream != nullptr) {
    throw std::invalid_argument("fMCS multi-executor dispatch does not support an external CUDA stream");
  }

  nvMolKit::ThreadSafeQueue<std::unique_ptr<TierChunk<maxAtoms, maxBonds, Policy>>> chunkQueue;
  for (size_t begin = 0; begin < indices.size(); begin += chunkSize) {
    const size_t end = std::min(begin + chunkSize, indices.size());
    chunkQueue.push(makeTierChunk<maxAtoms, maxBonds, Policy>(descs, indices, begin, end));
  }
  chunkQueue.close();

  std::vector<std::unique_ptr<FmcsExecutor>> executorStorage;
  executorStorage.reserve(static_cast<size_t>(executorCount));
  std::vector<FmcsExecutor*> executors;
  executors.reserve(static_cast<size_t>(executorCount));
  const bool useExternalStream = executorCount == 1;
  for (int i = 0; i < executorCount; ++i) {
    auto executor = std::make_unique<FmcsExecutor>(i, stream, useExternalStream);
    executors.push_back(executor.get());
    executorStorage.push_back(std::move(executor));
  }

  auto launchChunk = [&](FmcsExecutor& executor, std::unique_ptr<TierChunk<maxAtoms, maxBonds, Policy>>& chunk) {
    cudaStream_t executorStream = executor.stream;
    try {
      auto tablesDev = uploadPairMatchTables(
          chunk->hostTables, executorStream,
          &executor.bufs.matchTablesBuffer, &executor.bufs.matchTablesBufferBytes);

      chunk->hostPairInputs = uploadCsrAndAssemblePairInputs(
          chunk->tierDescs, tablesDev, executorStream,
          &executor.bufs.csrBuffer, &executor.bufs.csrBufferBytes);

      launchTierAsync<maxAtoms, maxBonds, Policy>(
          chunk->hostPairInputs, params, executorStream, executor.bufs, chunk->hostResults);

      checkCuda(cudaEventRecord(executor.copyDoneEvent.event(), executorStream),
                "cudaEventRecord (fMCS chunk copy done)");
    } catch (...) {
      freeBatchBuffers(executor.bufs, executorStream);
      throw;
    }
  };

  auto drainChunk = [&](FmcsExecutor& executor, std::unique_ptr<TierChunk<maxAtoms, maxBonds, Policy>>& chunk) {
    checkCuda(cudaEventSynchronize(executor.copyDoneEvent.event()),
              "cudaEventSynchronize (fMCS chunk copy done)");
    for (size_t k = 0; k < chunk->hostResults.size(); ++k) {
      outResults[chunk->resultIndices[k]] = expandDeviceResult<maxAtoms, maxBonds>(
          chunk->hostResults[k],
          chunk->tierDescs[k]->packedQuery.bondEndpoints,
          chunk->tierDescs[k]->packedTarget.bondEndpoints,
          chunk->tierDescs[k]->swapped);
    }
    freeBatchBuffers(executor.bufs, executor.stream);
  };

  nvMolKit::runQueuedExecutorRing(executors, chunkQueue, launchChunk, drainChunk);
}

template<class Policy, class InputT>
std::vector<MCSResult> runBatch(
    const std::vector<InputT>& a,
    const std::vector<InputT>& b,
    Parameters params,
    std::vector<float>* perPairTimesMs,
    cudaStream_t stream) {
  if (a.size() != b.size()) {
    throw std::runtime_error(
        "fMCS batch: graphsA and graphsB must have equal length");
  }
  const size_t N = a.size();
  std::vector<MCSResult> results(N);
  if (perPairTimesMs) {
    perPairTimesMs->assign(N, 0.0f);
  }
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

  std::vector<HostPairDescriptor*> descPtrs(N);
  for (size_t i = 0; i < N; ++i) descPtrs[i] = &descs[i];

  runTier<16, 16, Policy>(descPtrs, tierIndices[0], params, stream, results);
  runTier<32, 32, Policy>(descPtrs, tierIndices[1], params, stream, results);
  runTier<64, 64, Policy>(descPtrs, tierIndices[2], params, stream, results);
  runTier<128, 128, Policy>(descPtrs, tierIndices[3], params, stream, results);

  return results;
}

}  // namespace

std::vector<MCSResult> findMCESfMCSBatch(
    const std::vector<Graph>& graphsA,
    const std::vector<Graph>& graphsB,
    Parameters params,
    std::vector<float>* perPairTimesMs,
    cudaStream_t stream) {
  return runBatch<NullFMCSPolicy, Graph>(
      graphsA, graphsB, params, perPairTimesMs, stream);
}

std::vector<MCSResult> findMCESfMCSBatchLabeled(
    const std::vector<benchmark::MiviaGraphData>& graphsA,
    const std::vector<benchmark::MiviaGraphData>& graphsB,
    Parameters params,
    std::vector<float>* perPairTimesMs,
    cudaStream_t stream) {
  return runBatch<LabeledFMCSPolicy, benchmark::MiviaGraphData>(
      graphsA, graphsB, params, perPairTimesMs, stream);
}

}  // namespace fmcs
}  // namespace mcs
