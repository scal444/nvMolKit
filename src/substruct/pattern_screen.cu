// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <thrust/iterator/counting_iterator.h>

#include <algorithm>

#include "src/substruct/pattern_screen.h"
#include "src/utils/cub_helpers.cuh"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit {

namespace {

struct PatternScreenPredicate {
  const std::uint64_t* targetWords;
  const int*           batchAtomStarts;
  int                  numTargets;
  PatternScreenQuery   query;

  __device__ __forceinline__ bool operator()(int target) const {
    if (batchAtomStarts[target + 1] - batchAtomStarts[target] < query.numAtoms) {
      return false;
    }
    if (targetWords == nullptr) {
      return true;
    }
    for (int index = 0; index < query.numWords; ++index) {
      const std::uint64_t queryWord  = query.words[index];
      const std::size_t   wordOffset = static_cast<std::size_t>(query.wordIndices[index]) * numTargets;
      if ((targetWords[wordOffset + target] & queryWord) != queryWord) {
        return false;
      }
    }
    return true;
  }
};

std::size_t selectTempBytes(int numTargets, cudaStream_t stream) {
  std::size_t tempBytes = 0;
  cudaCheckError(cub::DeviceSelect::If(nullptr,
                                       tempBytes,
                                       thrust::counting_iterator<int>(0),
                                       static_cast<int*>(nullptr),
                                       static_cast<int*>(nullptr),
                                       numTargets,
                                       PatternScreenPredicate{},
                                       stream));
  return tempBytes;
}

}  // namespace

PatternScreenWorkspace::PatternScreenWorkspace(int deviceId) : deviceId_(deviceId), stream_("pattern screen") {
  indices_.setStream(stream_.stream());
  counts_.setStream(stream_.stream());
  tempStorage_.setStream(stream_.stream());
}

PatternScreenWorkspace::~PatternScreenWorkspace() noexcept {
  cudaFreeHost(hostCounts_);
  cudaFreeHost(hostIndices_);
}

void PatternScreenWorkspace::prepare(std::size_t numChunks, std::size_t totalTargets, std::size_t maxChunkTargets) {
  if (counts_.size() < numChunks) {
    counts_.resize(numChunks);
  }
  if (indices_.size() < totalTargets) {
    indices_.resize(totalTargets);
  }
  const std::size_t tempBytes =
    selectTempBytes(static_cast<int>(std::max<std::size_t>(1, maxChunkTargets)), stream_.stream());
  if (tempStorage_.size() < tempBytes) {
    tempStorage_.resize(tempBytes);
  }
  // Chunks without packed targets are never screened; they must read as empty.
  cudaCheckError(cudaMemsetAsync(counts_.data(), 0, numChunks * sizeof(int), stream_.stream()));
  if (hostCountsSize_ < numChunks) {
    cudaCheckError(cudaFreeHost(hostCounts_));
    hostCounts_ = nullptr;
    cudaCheckError(cudaMallocHost(&hostCounts_, numChunks * sizeof(int)));
    hostCountsSize_ = numChunks;
  }
  if (hostIndicesSize_ < totalTargets) {
    cudaCheckError(cudaFreeHost(hostIndices_));
    hostIndices_ = nullptr;
    cudaCheckError(cudaMallocHost(&hostIndices_, totalTargets * sizeof(int)));
    hostIndicesSize_ = totalTargets;
  }
}

void PatternScreenWorkspace::enqueueChunk(std::size_t               chunkIndex,
                                          std::size_t               targetOffset,
                                          const std::uint64_t*      targetWords,
                                          const int*                batchAtomStarts,
                                          int                       numTargets,
                                          const PatternScreenQuery& query) {
  std::size_t tempBytes = tempStorage_.size();
  cudaCheckError(cub::DeviceSelect::If(tempStorage_.data(),
                                       tempBytes,
                                       thrust::counting_iterator<int>(0),
                                       indices_.data() + targetOffset,
                                       counts_.data() + chunkIndex,
                                       numTargets,
                                       PatternScreenPredicate{targetWords, batchAtomStarts, numTargets, query},
                                       stream_.stream()));
}

void PatternScreenWorkspace::collect(std::size_t numChunks, const std::vector<std::size_t>& targetOffsets) {
  cudaStream_t stream = stream_.stream();
  cudaCheckError(cudaMemcpyAsync(hostCounts_, counts_.data(), numChunks * sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaCheckError(cudaStreamSynchronize(stream));
  bool anySelected = false;
  for (std::size_t chunk = 0; chunk < numChunks; ++chunk) {
    const int selected = hostCounts_[chunk];
    if (selected > 0) {
      anySelected = true;
      cudaCheckError(cudaMemcpyAsync(hostIndices_ + targetOffsets[chunk],
                                     indices_.data() + targetOffsets[chunk],
                                     static_cast<std::size_t>(selected) * sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     stream));
    }
  }
  if (anySelected) {
    cudaCheckError(cudaStreamSynchronize(stream));
  }
}

}  // namespace nvMolKit
