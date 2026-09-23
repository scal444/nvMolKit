// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/diversity_picker_algorithms.cuh"
#include "src/diversity_pickers.h"
#include "src/fingerprint_similarity_device.cuh"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit {
namespace detail {
namespace {

__global__ void findNextActiveKernel(const std::uint8_t* active, const int numItems, const int start, int* next) {
  __shared__ int found;
  for (int base = start; base < numItems; base += static_cast<int>(blockDim.x)) {
    const int  index = base + static_cast<int>(threadIdx.x);
    const bool hit   = index < numItems && active[index] != 0;
    if (__syncthreads_or(hit)) {
      if (threadIdx.x == 0) {
        found = numItems;
      }
      __syncthreads();
      if (hit) {
        atomicMin(&found, index);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        *next = found;
      }
      return;
    }
  }
  if (threadIdx.x == 0) {
    *next = numItems;
  }
}

__global__ void fillDoublesKernel(double* values, const int numItems, const double value) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < numItems) {
    values[index] = value;
  }
}

__global__ void markIndicesKernel(const int* indices, const int count, std::uint8_t* flags) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < count) {
    flags[indices[index]] = 1;
  }
}

__global__ void recordMaxMinPickKernel(const double*       candidateDistance,
                                       const std::int64_t* candidateIndex,
                                       const double        threshold,
                                       const int           position,
                                       int*                picks,
                                       int*                count,
                                       double*             lastDistance) {
  const bool stopped = *count != position || (threshold >= 0.0 && *candidateDistance <= threshold);
  if (stopped) {
    picks[position] = -1;
    return;
  }
  picks[position] = static_cast<int>(*candidateIndex);
  *lastDistance   = *candidateDistance;
  *count          = position + 1;
}

template <typename Op>
__global__ void matrixDistancesKernel(const cuda::std::span<const double> distances,
                                      const int                           numItems,
                                      const int*                          sourcePtr,
                                      const Op                            op) {
  const int source    = *sourcePtr;
  const int candidate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (!validSource(source, numItems) || candidate >= numItems || op.skip(candidate)) {
    return;
  }
  op.apply(candidate, distances[static_cast<std::size_t>(source) * numItems + candidate]);
}

__global__ void fingerprintBitCountsKernel(const cuda::std::span<const std::uint32_t> fingerprints,
                                           int*                                       bitCounts,
                                           const int                                  numItems,
                                           const int                                  numWords) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= numItems) {
    return;
  }
  int count = 0;
  for (int word = 0; word < numWords; ++word) {
    count += __popc(fingerprints[static_cast<std::size_t>(index) * numWords + word]);
  }
  bitCounts[index] = count;
}

template <FingerprintSimilarityMetric Metric, typename Op>
__global__ void fingerprintDistancesKernel(const cuda::std::span<const std::uint32_t> fingerprints,
                                           const int*                                 bitCounts,
                                           const int                                  numItems,
                                           const int                                  numWords,
                                           const int*                                 sourcePtr,
                                           const Op                                   op) {
  const int source    = *sourcePtr;
  const int candidate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (!validSource(source, numItems) || candidate >= numItems || op.skip(candidate)) {
    return;
  }
  int intersection = 0;
  for (int word = 0; word < numWords; ++word) {
    intersection += __popc(fingerprints[static_cast<std::size_t>(source) * numWords + word] &
                           fingerprints[static_cast<std::size_t>(candidate) * numWords + word]);
  }
  const double similarity =
    fingerprintSimilarity<Metric, double>(intersection, bitCounts[source], bitCounts[candidate]);
  op.apply(candidate, 1.0 - similarity);
}

class MatrixDistanceProvider {
 public:
  MatrixDistanceProvider(const cuda::std::span<const double> distances, const int numItems)
      : distances_(distances),
        numItems_(numItems) {}

  int size() const { return numItems_; }

  template <typename Op> void forEachDistance(const int* source, const Op& op, cudaStream_t stream) const {
    matrixDistancesKernel<<<pickerGridSize(numItems_), kPickerBlockSize, 0, stream>>>(distances_,
                                                                                      numItems_,
                                                                                      source,
                                                                                      op);
    cudaCheckError(cudaGetLastError());
  }

 private:
  cuda::std::span<const double> distances_;
  int                           numItems_;
};

template <FingerprintSimilarityMetric Metric> class FingerprintDistanceProvider {
 public:
  FingerprintDistanceProvider(const cuda::std::span<const std::uint32_t> fingerprints,
                              const int                                  numItems,
                              const int                                  numWords,
                              cudaStream_t                               stream)
      : fingerprints_(fingerprints),
        numItems_(numItems),
        numWords_(numWords),
        bitCounts_(numItems, stream) {
    if (numItems > 0) {
      fingerprintBitCountsKernel<<<pickerGridSize(numItems), kPickerBlockSize, 0, stream>>>(fingerprints_,
                                                                                            bitCounts_.data(),
                                                                                            numItems_,
                                                                                            numWords_);
      cudaCheckError(cudaGetLastError());
    }
  }

  int size() const { return numItems_; }

  template <typename Op> void forEachDistance(const int* source, const Op& op, cudaStream_t stream) const {
    fingerprintDistancesKernel<Metric><<<pickerGridSize(numItems_), kPickerBlockSize, 0, stream>>>(fingerprints_,
                                                                                                   bitCounts_.data(),
                                                                                                   numItems_,
                                                                                                   numWords_,
                                                                                                   source,
                                                                                                   op);
    cudaCheckError(cudaGetLastError());
  }

 private:
  cuda::std::span<const std::uint32_t> fingerprints_;
  int                                  numItems_;
  int                                  numWords_;
  AsyncDeviceVector<int>               bitCounts_;
};

void validateDistanceMatrix(const cuda::std::span<const double> distanceMatrix, const int numItems) {
  if (numItems < 0 || distanceMatrix.size() != static_cast<std::size_t>(numItems) * numItems) {
    throw std::invalid_argument("Distance matrix buffer size does not match its square shape");
  }
}

void validateMatrixCutoff(const double cutoff) {
  if (!std::isfinite(cutoff) || cutoff < 0.0) {
    throw std::invalid_argument("cutoff must be finite and non-negative");
  }
}

void validateFingerprintInput(const cuda::std::span<const std::uint32_t> fingerprints,
                              const int                                  numItems,
                              const int                                  numWords) {
  if (numItems < 0 || numWords <= 0 ||
      fingerprints.size() != static_cast<std::size_t>(numItems) * static_cast<std::size_t>(numWords)) {
    throw std::invalid_argument("Fingerprint buffer size does not match its shape");
  }
}

template <typename Function>
auto withFingerprintProvider(const cuda::std::span<const std::uint32_t> fingerprints,
                             const int                                  numItems,
                             const int                                  numWords,
                             const FingerprintSimilarityMetric          metric,
                             cudaStream_t                               stream,
                             Function&&                                 function) {
  validateFingerprintInput(fingerprints, numItems, numWords);
  if (metric == FingerprintSimilarityMetric::Tanimoto) {
    FingerprintDistanceProvider<FingerprintSimilarityMetric::Tanimoto> provider(fingerprints,
                                                                                numItems,
                                                                                numWords,
                                                                                stream);
    return function(provider);
  }
  if (metric == FingerprintSimilarityMetric::Cosine) {
    FingerprintDistanceProvider<FingerprintSimilarityMetric::Cosine> provider(fingerprints, numItems, numWords, stream);
    return function(provider);
  }
  throw std::invalid_argument("Unsupported fingerprint similarity metric");
}

}  // namespace

void launchFindNextActive(const std::uint8_t* active,
                          const int           numItems,
                          const int           start,
                          int*                next,
                          cudaStream_t        stream) {
  findNextActiveKernel<<<1, kPickerBlockSize, 0, stream>>>(active, numItems, start, next);
  cudaCheckError(cudaGetLastError());
}

void launchFillDoubles(double* values, const int numItems, const double value, cudaStream_t stream) {
  if (numItems == 0) {
    return;
  }
  fillDoublesKernel<<<pickerGridSize(numItems), kPickerBlockSize, 0, stream>>>(values, numItems, value);
  cudaCheckError(cudaGetLastError());
}

void launchMarkIndices(const int* indices, const int count, std::uint8_t* flags, cudaStream_t stream) {
  if (count == 0) {
    return;
  }
  markIndicesKernel<<<pickerGridSize(count), kPickerBlockSize, 0, stream>>>(indices, count, flags);
  cudaCheckError(cudaGetLastError());
}

void launchRecordMaxMinPick(const double*       candidateDistance,
                            const std::int64_t* candidateIndex,
                            const double        threshold,
                            const int           position,
                            int*                picks,
                            int*                count,
                            double*             lastDistance,
                            cudaStream_t        stream) {
  recordMaxMinPickKernel<<<1, 1, 0, stream>>>(candidateDistance,
                                              candidateIndex,
                                              threshold,
                                              position,
                                              picks,
                                              count,
                                              lastDistance);
  cudaCheckError(cudaGetLastError());
}

void validateFirstPicks(const std::vector<int>& firstPicks, const int numItems) {
  std::vector<std::uint8_t> seen(numItems, 0);
  for (const int pick : firstPicks) {
    if (pick < 0 || pick >= numItems) {
      throw std::invalid_argument("first_picks contains an index outside the input pool");
    }
    if (seen[pick] != 0) {
      throw std::invalid_argument("first_picks must not contain duplicate indices");
    }
    seen[pick] = 1;
  }
}

void validateMaxMinThreshold(const double threshold, const double maximum) {
  if (threshold == -1.0) {
    return;
  }
  if (!std::isfinite(threshold) || threshold < 0.0 || threshold > maximum) {
    throw std::invalid_argument("threshold must be finite and in the supported distance range");
  }
}

void validateUnitCutoff(const double cutoff) {
  if (!(cutoff >= 0.0 && cutoff <= 1.0)) {
    throw std::invalid_argument("cutoff must be in [0, 1]");
  }
}

int randomFirstPick(const int poolSize, const int seed) {
  // Matches RDKit's MaxMinPicker so seeded runs select the same first item.
  boost::mt19937 generator;
  if (seed >= 0) {
    generator.seed(static_cast<boost::mt19937::result_type>(seed));
  } else {
    generator.seed(std::random_device()());
  }
  boost::uniform_int<>                                            distribution(0, poolSize - 1);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<>> source(generator, distribution);
  return source();
}

ClusteringResult buildClusteringResult(const std::vector<int>& labels, const std::vector<int>& centroids) {
  const int                 numClusters = static_cast<int>(centroids.size());
  std::vector<std::int64_t> sizes(numClusters, 0);
  for (const int label : labels) {
    ++sizes[label];
  }

  std::vector<int> order(numClusters);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&sizes](const int left, const int right) {
    return sizes[left] > sizes[right];
  });
  std::vector<int> remap(numClusters);
  for (int newId = 0; newId < numClusters; ++newId) {
    remap[order[newId]] = newId;
  }

  ClusteringResult result;
  result.clusterIds.resize(labels.size());
  result.centroids.resize(numClusters);
  result.clusterSizes.resize(numClusters);
  std::transform(labels.begin(), labels.end(), result.clusterIds.begin(), [&remap](const int label) {
    return remap[label];
  });
  for (int newId = 0; newId < numClusters; ++newId) {
    const int oldId            = order[newId];
    result.centroids[newId]    = centroids[oldId];
    result.clusterSizes[newId] = sizes[oldId];
  }
  return result;
}

}  // namespace detail

PickerResult leaderFromDistanceMatrix(const cuda::std::span<const double> distanceMatrix,
                                      const int                           numItems,
                                      const double                        cutoff,
                                      const int                           pickSize,
                                      const std::vector<int>&             firstPicks,
                                      cudaStream_t                        stream) {
  detail::validateDistanceMatrix(distanceMatrix, numItems);
  detail::validateMatrixCutoff(cutoff);
  detail::MatrixDistanceProvider provider(distanceMatrix, numItems);
  return detail::leaderPick(provider, cutoff, pickSize, firstPicks, nullptr, stream);
}

PickerResult maxMinFromDistanceMatrix(const cuda::std::span<const double> distanceMatrix,
                                      const int                           numItems,
                                      const int                           pickSize,
                                      const std::vector<int>&             firstPicks,
                                      const int                           seed,
                                      const double                        threshold,
                                      cudaStream_t                        stream) {
  detail::validateDistanceMatrix(distanceMatrix, numItems);
  detail::validateMaxMinThreshold(threshold, std::numeric_limits<double>::max());
  detail::MatrixDistanceProvider provider(distanceMatrix, numItems);
  return detail::maxMinPick(provider, pickSize, firstPicks, seed, threshold, stream);
}

ClusteringResult diseFromDistanceMatrix(const cuda::std::span<const double> distanceMatrix,
                                        const int                           numItems,
                                        const double                        cutoff,
                                        const bool                          nearestAssignment,
                                        cudaStream_t                        stream) {
  detail::validateDistanceMatrix(distanceMatrix, numItems);
  detail::validateMatrixCutoff(cutoff);
  detail::MatrixDistanceProvider provider(distanceMatrix, numItems);
  return detail::diseCluster(provider, cutoff, nearestAssignment, stream);
}

PickerResult fusedLeaderGpu(const cuda::std::span<const std::uint32_t> fingerprints,
                            const int                                  numFingerprints,
                            const int                                  numWords,
                            const double                               cutoff,
                            const FingerprintSimilarityMetric          metric,
                            const int                                  pickSize,
                            const std::vector<int>&                    firstPicks,
                            cudaStream_t                               stream) {
  detail::validateUnitCutoff(cutoff);
  return detail::withFingerprintProvider(fingerprints, numFingerprints, numWords, metric, stream, [&](auto& provider) {
    return detail::leaderPick(provider, cutoff, pickSize, firstPicks, nullptr, stream);
  });
}

PickerResult fusedMaxMinGpu(const cuda::std::span<const std::uint32_t> fingerprints,
                            const int                                  numFingerprints,
                            const int                                  numWords,
                            const int                                  pickSize,
                            const FingerprintSimilarityMetric          metric,
                            const std::vector<int>&                    firstPicks,
                            const int                                  seed,
                            const double                               threshold,
                            cudaStream_t                               stream) {
  detail::validateMaxMinThreshold(threshold, 1.0);
  return detail::withFingerprintProvider(fingerprints, numFingerprints, numWords, metric, stream, [&](auto& provider) {
    return detail::maxMinPick(provider, pickSize, firstPicks, seed, threshold, stream);
  });
}

ClusteringResult fusedDiseGpu(const cuda::std::span<const std::uint32_t> fingerprints,
                              const int                                  numFingerprints,
                              const int                                  numWords,
                              const double                               cutoff,
                              const FingerprintSimilarityMetric          metric,
                              const bool                                 nearestAssignment,
                              cudaStream_t                               stream) {
  detail::validateUnitCutoff(cutoff);
  return detail::withFingerprintProvider(fingerprints, numFingerprints, numWords, metric, stream, [&](auto& provider) {
    return detail::diseCluster(provider, cutoff, nearestAssignment, stream);
  });
}

}  // namespace nvMolKit
