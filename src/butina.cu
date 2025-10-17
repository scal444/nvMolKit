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

#include <cub/cub.cuh>

#include "butina.h"
#include "data_structures/flat_bit_vect.h"
#include "host_vector.h"
#include "nvtx.h"
namespace nvMolKit {

namespace {
constexpr int blockSizeCount               = 256;
constexpr int kAssignedAsSingletonSentinel = std::numeric_limits<int>::max() - 1;
constexpr int kMinLoopSizeForAssignment    = 3;
constexpr int kBitVectWidth = 32;


// TODO can we do some dynamic assignment as lower numbers of rows execute due to being done.
__global__ void butinaKernelCountClusterSize(const cuda::std::span<const FlatBitVect<kBitVectWidth>> hitMatrix,
                                             const cuda::std::span<int>           clusters,
                                             const cuda::std::span<int>           clusterSizes) {
  const auto tid       = static_cast<int>(threadIdx.x);
  const auto pointIdx  = static_cast<int>(blockIdx.x);
  const auto numPoints = static_cast<int>(clusters.size());

  if (clusters[pointIdx] >= 0) {
    clusterSizes[pointIdx] = 0;
    return;
  }

  const int hitMatDim = (clusters.size() - 1) / kBitVectWidth + 1;

  const auto hits       = hitMatrix.subspan(pointIdx * hitMatDim, numPoints);
  int                                  localCount = 0;

  constexpr int kLoadsPerLoop = 4;
  using BlockLoad = cub::BlockLoad<int, kLoadsPerLoop * blockSizeCount, kLoadsPerLoop, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  __shared__ BlockLoad::TempStorage temp_storage;
  int blockClusterData[kLoadsPerLoop];
  for (int i = tid; i < hitMatDim; i += blockSizeCount) {
    const FlatBitVect<kBitVectWidth> bitVect = hits[i];
    const int count = min(numPoints - i * kBitVectWidth, kBitVectWidth);

    for (int j = 0; j < count; j+= kLoadsPerLoop) {
      BlockLoad(temp_storage).Load(clusters.data() + i * kBitVectWidth + j * kLoadsPerLoop, blockClusterData, count);
      const bool isNeighbor = bitVect[j];
      if (isNeighbor) {
        const int cluster = blockClusterData[j % kLoadsPerLoop];
        if (cluster < 0) {
          localCount++;
        }
      }
      __syncthreads(); // To reuse BlockLoad shared memory.
    }
  }

  __shared__ cub::BlockReduce<int, blockSizeCount>::TempStorage tempStorage;
  const int totalCount = cub::BlockReduce<int, blockSizeCount>(tempStorage).Sum(localCount);
  if (tid == 0) {
    // printf("Count %d for pt %d\n", totalCount, pointIdx);
    clusterSizes[pointIdx] = totalCount;
    if (totalCount < 2) {
      // Note that this would be a data race between writing this cluster[] and another thread reading it[]. However,
      // the hit check should preclude any singleton entry from being read by anything other than its own
      // thread.
      clusters[pointIdx] = kAssignedAsSingletonSentinel;
    }
  }
}

__global__ void butinaWriteClusterValue(const cuda::std::span<const FlatBitVect<kBitVectWidth>> hitMatrix,
                                        const cuda::std::span<int>           clusters,
                                        const int*                           centralIdx,
                                        const int*                           clusterIdx,
                                        const int*                           maxClusterSize) {
  const auto numPoints = static_cast<int>(clusters.size());
  const auto tid       = static_cast<int>(threadIdx.x + (blockIdx.x * blockDim.x));
  const int  clusterSz = *maxClusterSize;
  if (clusterSz < kMinLoopSizeForAssignment) {
    return;
  }
  const int pointIdx = *centralIdx;
  if (pointIdx < 0) {
    return;
  }
  const int clusterVal = *clusterIdx;
  if (tid == 0) {
    // printf("Cluster %d centroid is element %d\n", clusterVal, pointIdx);
  }

  const int hitMatDim = (clusters.size() - 1) / kBitVectWidth + 1;

  const cuda::std::span<const FlatBitVect<kBitVectWidth>> hits = hitMatrix.subspan(pointIdx * hitMatDim, hitMatDim);
  if (tid < numPoints) {
    if (const bool isNeighbor = hits[tid / kBitVectWidth][tid % kBitVectWidth]; isNeighbor > 0) {
      if (const int cluster = clusters[tid]; cluster < 0) {
        // printf("CLUSTER: %d to item %d total size %d\n", clusterVal, tid, clusterSz);
        clusters[tid] = clusterVal;
      }
    }
  }
}

__global__ void bumpClusterIdxKernel(int* clusterIdx, const int* lastClusterSize) {
  if (const auto tid = static_cast<int>(threadIdx.x); tid == 0) {
    if (*lastClusterSize >= kMinLoopSizeForAssignment) {
      clusterIdx[tid] += 1;
    }
  }
}

__global__ void pairDoubletKernels(const cuda::std::span<const FlatBitVect<kBitVectWidth>> hitMatrix,
                                   const cuda::std::span<int>           clusters,
                                   const cuda::std::span<const int>     clusterSizes) {
  const auto tid       = static_cast<int>(threadIdx.x);
  const auto pointIdx  = static_cast<int>(blockIdx.x);

  if (clusterSizes[pointIdx] != 2) {
    return;
  }
  const int hitMatDim = (clusters.size() - 1) / kBitVectWidth + 1;

  const cuda::std::span<const FlatBitVect<kBitVectWidth>> hits = hitMatrix.subspan(pointIdx * hitMatDim, hitMatDim);
  // Loop up to point IDX so that only one of the pairs does the write. The followup kernel will set both values
  for (int i = tid; i < pointIdx; i += blockSizeCount) {
    const bool isNeighbor = hits[i / kBitVectWidth][i % kBitVectWidth];
    if (i != pointIdx && isNeighbor && clusterSizes[i] == 2) {
      clusters[pointIdx] = kAssignedAsSingletonSentinel - 1 - i;  // Mark as paired with i
      // printf("Pairing point %d with %d\n", pointIdx, i);
      break;
    }
  }
}

// TODO This could be parallelized with a shared clustderIdx++ to atomicAdd.
__global__ void assignDoubletIdsKernel(const cuda::std::span<int> clusters, int* nextClusterIdx) {
  int       clusterIdx           = *nextClusterIdx;
  const int expectedDoubletRange = kAssignedAsSingletonSentinel - clusters.size();
  for (int i = static_cast<int>(clusters.size()) - 1; i >= 0; i--) {
    const int clustId = clusters[i];
    if (clustId >= expectedDoubletRange && clustId < kAssignedAsSingletonSentinel) {
      int otherIdx       = kAssignedAsSingletonSentinel - 1 - clustId;
      clusters[i]        = clusterIdx;
      clusters[otherIdx] = clusterIdx;
      // printf("CLUSTER: Assigning doublet cluster %d to items %d and %d\n", clusterIdx, i, otherIdx);
      clusterIdx++;
    }
  }
  *nextClusterIdx = clusterIdx;
}

// TODO This could be parallelized with a shared clustderIdx++ to atomicAdd.
__global__ void assignSingletonIdsKernel(const cuda::std::span<int> clusters, const int* nextClusterIdx) {
  int clusterIdx = *nextClusterIdx;
  for (int i = static_cast<int>(clusters.size()) - 1; i >= 0; i--) {
    const int clustId = clusters[i];
    if (clustId < 0 || clustId == kAssignedAsSingletonSentinel) {
      // printf("CLUSTER: Assigning singleton %d to item %d\n", clusterIdx, i);
      clusters[i] = clusterIdx;
      clusterIdx++;
    }
  }
}

constexpr int   argMaxBlockSize = 512;

__global__ void lastArgMax(const cuda::std::span<const int> values, int* outVal, int* outIdx) {
  int            maxVal = cuda::std::numeric_limits<int>::min();
  int            maxID  = -1;
  __shared__ int foundMaxVal[argMaxBlockSize];
  __shared__ int foundMaxIds[argMaxBlockSize];
  const auto     tid = static_cast<int>(threadIdx.x);
  for (int i = tid; i < values.size(); i += argMaxBlockSize) {
    if (const int val = values[i]; val >= maxVal) {
      maxID  = i;
      maxVal = val;
    }
  }
  foundMaxVal[tid] = maxVal;
  foundMaxIds[tid] = maxID;

  __shared__ cub::BlockReduce<int, argMaxBlockSize>::TempStorage storage;
  const int actualMaxVal = cub::BlockReduce<int, argMaxBlockSize>(storage).Reduce(maxVal, cub::Max());
  __syncthreads();  // For shared memory write of maxVal and maxID
  if (tid == 0) {
    *outVal = actualMaxVal;
    for (int i = argMaxBlockSize - 1; i >= 0; i--) {
      if (foundMaxVal[i] == actualMaxVal) {
        *outIdx = foundMaxIds[i];
        // printf("Found max of %d at %d\n", actualMaxVal, *outIdx);
        break;
      }
    }
  }
}

// TODO - consolidate this to device vector code.
template <typename T> __global__ void setAllKernel(const size_t numElements, T value, T* dst) {
  const size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = value;
  }
}
template <typename T> void setAll(const cuda::std::span<T>& vec, const T& value, cudaStream_t stream) {
  const size_t numElements = vec.size();
  if (numElements == 0) {
    return;
  }
  constexpr int blockSize = 128;
  const size_t  numBlocks = (numElements + blockSize - 1) / blockSize;
  setAllKernel<<<numBlocks, blockSize, 0, stream>>>(numElements, value, vec.data());
  cudaCheckError(cudaGetLastError());
}

void innerButinaLoop(const int                            numPoints,
                     const cuda::std::span<const FlatBitVect<kBitVectWidth>> hitMatrix,
                     const cuda::std::span<int>           clusters,
                      const cuda::std::span<FlatBitVect<32>> clusterIsSet,
                     const cuda::std::span<int>           clusterSizesSpan,
                     const AsyncDevicePtr<int>&           maxIndex,
                     const AsyncDevicePtr<int>&           maxValue,
                     const AsyncDevicePtr<int>&           clusterIdx,
                     PinnedHostVector<int>&               maxCluster,
                     cudaStream_t                         stream) {
  const int numBlocksFlat = ((static_cast<int>(clusterSizesSpan.size()) - 1) / blockSizeCount) + 1;

  butinaKernelCountClusterSize<<<numPoints, blockSizeCount, 0, stream>>>(hitMatrix, clusters, clusterSizesSpan);
  cudaCheckError(cudaGetLastError());
  // cudaCheckError(cub::DeviceReduce::ArgMax(tempStorage.data(),
  //                                          tempStorageBytes,
  //                                          clusterSizes.data(),
  //                                          maxValue.data(),
  //                                          maxIndex.data(),
  //                                          numPoints,
  //                                          stream));

  lastArgMax<<<1, argMaxBlockSize, 0, stream>>>(clusterSizesSpan, maxValue.data(), maxIndex.data());
  cudaCheckError(cudaGetLastError());
  butinaWriteClusterValue<<<numBlocksFlat, blockSizeCount, 0, stream>>>(hitMatrix,
                                                                        clusters,
                                                                        maxIndex.data(),
                                                                        clusterIdx.data(),
                                                                        maxValue.data());
  cudaCheckError(cudaGetLastError());
  bumpClusterIdxKernel<<<1, 1, 0, stream>>>(clusterIdx.data(), maxValue.data());
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaMemcpyAsync(maxCluster.data(), maxValue.data(), sizeof(int), cudaMemcpyDefault, stream));
  cudaStreamSynchronize(stream);
  int maxV = maxCluster[0];
  // printf("End of loop max cluster size: %d\n", maxV);
}


}  // namespace
void butinaGpu(const cuda::std::span<const FlatBitVect<kBitVectWidth>> hitMatrix,
               const cuda::std::span<int>           clusters,
               cudaStream_t                         stream,
               const bool                           useGraph) {
  ScopedNvtxRange setupRange("Butina Setup");
  const size_t    numPoints = clusters.size();
  setAll(clusters, -1, stream);
  if (const size_t matSize = hitMatrix.size(); numPoints * numPoints != matSize) {
    throw std::runtime_error("Butina size mismatch" + std::to_string(numPoints) +
                             " points^2 != " + std::to_string(matSize) + " neighbor matrix size");
  }
  AsyncDeviceVector<int> clusterSizes(clusters.size(), stream);
  clusterSizes.zero();

  const AsyncDevicePtr<int> maxIndex(-1, stream);
  const AsyncDevicePtr<int> maxValue(std::numeric_limits<int>::max(), stream);
  const AsyncDevicePtr<int> clusterIdx(0, stream);

  const int numBitFieldClusterIsSet = (clusters.size() -1) / kBitVectWidth + 1;
  AsyncDeviceVector<FlatBitVect<32>> clusterIsSet(numBitFieldClusterIsSet, stream);
  auto clusterIsSetSpan = toSpan(clusterIsSet);
  clusterIsSet.zero();
  PinnedHostVector<int>     maxCluster(1);
  maxCluster[0] = std::numeric_limits<int>::max();
  // size_t tempStorageBytes = 0;
  // cub::DeviceReduce::ArgMax(nullptr,
  //                           tempStorageBytes,
  //                           clusterSizes.data(),
  //                           maxValue.data(),
  //                           maxIndex.data(),
  //                           static_cast<int64_t>(numPoints),
  //                           stream);
  // AsyncDeviceVector<uint8_t> const tempStorage(tempStorageBytes, stream);
  setupRange.pop();
  const auto clusterSizesSpan = toSpan(clusterSizes);

  while (maxCluster[0] >= kMinLoopSizeForAssignment) {
    ScopedNvtxRange loopRange("Butina Loop");
    innerButinaLoop(numPoints,
                    hitMatrix,
                    clusters,
                    clusterIsSetSpan,
                    clusterSizesSpan,
                    maxIndex,
                    maxValue,
                    clusterIdx,
                    maxCluster,
                    stream);
  }
  // printf("Exiting loop\n");
  pairDoubletKernels<<<numPoints, blockSizeCount, 0, stream>>>(hitMatrix, clusters, clusterSizesSpan);
  assignDoubletIdsKernel<<<1, 1, 0, stream>>>(clusters, clusterIdx.data());
  assignSingletonIdsKernel<<<1, 1, 0, stream>>>(clusters, clusterIdx.data());
  cudaStreamSynchronize(stream);
}

namespace {
constexpr int kBlockSizeHitAssign = 128;
constexpr int kLoadsPerLoop = 4;
__global__ void assignHitsMatrixFromDoubleMatrix(const cuda::std::span<const double> distanceMatrix,
                                                  const cuda::std::span<FlatBitVect<kBitVectWidth>>  hitMatrix,
                                                  const double                    cutoff,
                                                  const int hitMatrixDim,
                                                  const int distMatrixDim) {
  using BlockLoad = cub::BlockLoad<double, kLoadsPerLoop * kBlockSizeHitAssign, kLoadsPerLoop, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  __shared__ BlockLoad::TempStorage temp_storage;

  const auto tid       = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int row = tid / hitMatrixDim;
  const int col = tid % hitMatrixDim;

  const int startIdx = row * distMatrixDim + col * kBitVectWidth;
  const int count = min(distMatrixDim - col * kBitVectWidth, kBitVectWidth);
  const auto distanceSpan = distanceMatrix.subspan(startIdx, count);


  FlatBitVect<kBitVectWidth> bitVect(false);
  double blockData[kLoadsPerLoop];
  for (int loadLoop = 0; loadLoop < kBitVectWidth/ kLoadsPerLoop; ++loadLoop) {
    BlockLoad(temp_storage).Load(distanceSpan.data() + loadLoop * kLoadsPerLoop, blockData, count);
    for (int i = 0; i < kLoadsPerLoop; i++) {
      if (blockData[i] < cutoff) {
        bitVect.setBit(i, true);
      }
    }
    __syncthreads(); // for BlockLoad shared memory reuse
  }


  hitMatrix[tid] = bitVect;

}


}  // namespace

void butinaGpu(const cuda::std::span<const double> distanceMatrix,
               const cuda::std::span<int>          clusters,
               const double                        cutoff,
               cudaStream_t                        stream,
               const bool                          useGraph) {
  // Pad the matrix out to bitvect size
  const int bitVectDim = (clusters.size() - 1) / kBitVectWidth + 1;
  AsyncDeviceVector<FlatBitVect<kBitVectWidth>> hitMatrix(bitVectDim * bitVectDim , stream);
  const int numBlocks = hitMatrix.size();
  assignHitsMatrixFromDoubleMatrix<<<numBlocks, kBlockSizeHitAssign, 0, stream>>>(distanceMatrix, toSpan(hitMatrix), cutoff, bitVectDim, clusters.size());
  cudaCheckError(cudaGetLastError());
  butinaGpu(toSpan(hitMatrix), clusters, stream, useGraph);
}

}  // namespace nvMolKit