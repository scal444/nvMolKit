// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cub/device/device_reduce.cuh>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/diversity_pickers.h"
#include "src/fingerprint_similarity_device.cuh"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit {
namespace {

constexpr int kBlockSize = 256;

__global__ void initializeActiveKernel(std::uint8_t* active, const int numItems) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < numItems) {
    active[index] = 1;
  }
}

__global__ void findFirstActiveKernel(const std::uint8_t* active, const int numItems, int* firstActive) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < numItems && active[index]) {
    atomicMin(firstActive, index);
  }
}

__global__ void fingerprintBitCountsKernel(cuda::std::span<const std::uint32_t> fingerprints,
                                           int*                                 bitCounts,
                                           const int                            numItems,
                                           const int                            numWords) {
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

template <FingerprintSimilarityMetric Metric>
__device__ __forceinline__ double fingerprintPairSimilarity(cuda::std::span<const std::uint32_t> fingerprints,
                                                            const int*                           bitCounts,
                                                            const int                            left,
                                                            const int                            right,
                                                            const int                            numWords) {
  int intersection = 0;
  for (int word = 0; word < numWords; ++word) {
    const auto leftWord  = fingerprints[static_cast<std::size_t>(left) * numWords + word];
    const auto rightWord = fingerprints[static_cast<std::size_t>(right) * numWords + word];
    intersection += __popc(leftWord & rightWord);
  }
  return detail::fingerprintSimilarity<Metric, double>(intersection, bitCounts[left], bitCounts[right]);
}

__global__ void suppressMatrixCandidatesKernel(cuda::std::span<const double> distanceMatrix,
                                               std::uint8_t*                 active,
                                               const int                     numItems,
                                               const int                     leader,
                                               const double                  cutoff) {
  const int candidate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (candidate >= numItems || !active[candidate]) {
    return;
  }
  // Selection uniqueness is an algorithm invariant and must not depend on a
  // caller-provided matrix having an exact zero on its diagonal.
  if (candidate == leader || distanceMatrix[static_cast<std::size_t>(leader) * numItems + candidate] <= cutoff) {
    active[candidate] = 0;
  }
}

template <FingerprintSimilarityMetric Metric>
__global__ void suppressFingerprintCandidatesKernel(cuda::std::span<const std::uint32_t> fingerprints,
                                                    const int*                           bitCounts,
                                                    std::uint8_t*                        active,
                                                    const int                            numItems,
                                                    const int                            numWords,
                                                    const int                            leader,
                                                    const double                         similarityThreshold) {
  const int candidate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (candidate >= numItems || !active[candidate]) {
    return;
  }
  // A selected leader must always leave the active set.  This cannot be
  // inferred from the metric: cosine similarity deliberately defines two
  // empty fingerprints as zero similarity.
  if (candidate == leader) {
    active[candidate] = 0;
    return;
  }
  const double similarity = fingerprintPairSimilarity<Metric>(fingerprints, bitCounts, leader, candidate, numWords);
  if (similarity >= similarityThreshold) {
    active[candidate] = 0;
  }
}

__global__ void initializeMinimumDistancesKernel(double* minimumDistances, std::uint8_t* selected, const int numItems) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < numItems) {
    minimumDistances[index] = DBL_MAX;
    selected[index]         = 0;
  }
}

__global__ void updateMatrixMinimumDistancesKernel(cuda::std::span<const double> distanceMatrix,
                                                   double*                       minimumDistances,
                                                   std::uint8_t*                 selected,
                                                   const int                     numItems,
                                                   const int                     pick) {
  const int candidate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (candidate >= numItems) {
    return;
  }
  if (candidate == pick) {
    selected[candidate]         = 1;
    minimumDistances[candidate] = -1.0;
    return;
  }
  if (!selected[candidate]) {
    minimumDistances[candidate] =
      min(minimumDistances[candidate], distanceMatrix[static_cast<std::size_t>(candidate) * numItems + pick]);
  }
}

template <FingerprintSimilarityMetric Metric>
__global__ void updateFingerprintMinimumDistancesKernel(cuda::std::span<const std::uint32_t> fingerprints,
                                                        const int*                           bitCounts,
                                                        double*                              minimumDistances,
                                                        std::uint8_t*                        selected,
                                                        const int                            numItems,
                                                        const int                            numWords,
                                                        const int                            pick) {
  const int candidate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (candidate >= numItems) {
    return;
  }
  if (candidate == pick) {
    selected[candidate]         = 1;
    minimumDistances[candidate] = -1.0;
    return;
  }
  if (!selected[candidate]) {
    const double similarity     = fingerprintPairSimilarity<Metric>(fingerprints, bitCounts, candidate, pick, numWords);
    minimumDistances[candidate] = min(minimumDistances[candidate], 1.0 - similarity);
  }
}

__global__ void assignMatrixClustersKernel(cuda::std::span<const double> distanceMatrix,
                                           const int*                    centroids,
                                           int*                          labels,
                                           const int                     numItems,
                                           const int                     numClusters,
                                           const double                  cutoff,
                                           const bool                    nearestAssignment) {
  const int candidate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (candidate >= numItems) {
    return;
  }

  for (int cluster = 0; cluster < numClusters; ++cluster) {
    if (centroids[cluster] == candidate) {
      labels[candidate] = cluster;
      return;
    }
  }

  int    bestCluster  = 0;
  double bestDistance = distanceMatrix[static_cast<std::size_t>(centroids[0]) * numItems + candidate];
  if (!nearestAssignment && bestDistance <= cutoff) {
    labels[candidate] = 0;
    return;
  }
  for (int cluster = 1; cluster < numClusters; ++cluster) {
    const double distance = distanceMatrix[static_cast<std::size_t>(centroids[cluster]) * numItems + candidate];
    if (!nearestAssignment && distance <= cutoff) {
      labels[candidate] = cluster;
      return;
    }
    if (nearestAssignment && distance < bestDistance) {
      bestDistance = distance;
      bestCluster  = cluster;
    }
  }
  labels[candidate] = bestCluster;
}

template <FingerprintSimilarityMetric Metric>
__global__ void assignFingerprintClustersKernel(cuda::std::span<const std::uint32_t> fingerprints,
                                                const int*                           bitCounts,
                                                const int*                           centroids,
                                                int*                                 labels,
                                                const int                            numItems,
                                                const int                            numWords,
                                                const int                            numClusters,
                                                const double                         similarityThreshold,
                                                const bool                           nearestAssignment) {
  const int candidate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (candidate >= numItems) {
    return;
  }

  for (int cluster = 0; cluster < numClusters; ++cluster) {
    if (centroids[cluster] == candidate) {
      labels[candidate] = cluster;
      return;
    }
  }

  int    bestCluster    = 0;
  double bestSimilarity = fingerprintPairSimilarity<Metric>(fingerprints, bitCounts, centroids[0], candidate, numWords);
  if (!nearestAssignment && bestSimilarity >= similarityThreshold) {
    labels[candidate] = 0;
    return;
  }
  for (int cluster = 1; cluster < numClusters; ++cluster) {
    const double similarity =
      fingerprintPairSimilarity<Metric>(fingerprints, bitCounts, centroids[cluster], candidate, numWords);
    if (!nearestAssignment && similarity >= similarityThreshold) {
      labels[candidate] = cluster;
      return;
    }
    if (nearestAssignment && similarity > bestSimilarity) {
      bestSimilarity = similarity;
      bestCluster    = cluster;
    }
  }
  labels[candidate] = bestCluster;
}

void validateFirstPicks(const std::vector<int>& firstPicks, const int numItems) {
  std::vector<std::uint8_t> seen(numItems, 0);
  for (const int pick : firstPicks) {
    if (pick < 0 || pick >= numItems) {
      throw std::invalid_argument("first_picks contains an index outside the input pool");
    }
    if (seen[pick]) {
      throw std::invalid_argument("first_picks must not contain duplicate indices");
    }
    seen[pick] = 1;
  }
}

void validateDistanceMatrix(const cuda::std::span<const double> distanceMatrix, const int numItems) {
  if (numItems < 0 || distanceMatrix.size() != static_cast<std::size_t>(numItems) * numItems) {
    throw std::invalid_argument("Distance matrix buffer size does not match its square shape");
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

void validateMaxMinThreshold(const double threshold, const double maximum) {
  if (threshold == -1.0) {
    return;
  }
  if (!std::isfinite(threshold) || threshold < 0.0 || threshold > maximum) {
    throw std::invalid_argument("threshold must be finite and in the supported distance range");
  }
}

int randomFirstPick(const int poolSize, const int seed) {
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

PickerResult makePickerResult(const std::vector<int>& picks, const double lastDistance, cudaStream_t stream) {
  PickerResult result{AsyncDeviceVector<int>(picks.size(), stream), static_cast<int>(picks.size()), lastDistance};
  if (!picks.empty()) {
    result.indices.copyFromHost(picks);
    cudaCheckError(cudaStreamSynchronize(stream));
  }
  return result;
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

class MatrixDistanceProvider {
 public:
  MatrixDistanceProvider(cuda::std::span<const double> distances, const int numItems)
      : distances_(distances),
        numItems_(numItems) {}

  void suppress(std::uint8_t* active, const int leader, const double cutoff, cudaStream_t stream) const {
    suppressMatrixCandidatesKernel<<<(numItems_ + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(distances_,
                                                                                                         active,
                                                                                                         numItems_,
                                                                                                         leader,
                                                                                                         cutoff);
    cudaCheckError(cudaGetLastError());
  }

  void updateMinimumDistances(double*       minimumDistances,
                              std::uint8_t* selected,
                              const int     pick,
                              cudaStream_t  stream) const {
    updateMatrixMinimumDistancesKernel<<<(numItems_ + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
      distances_,
      minimumDistances,
      selected,
      numItems_,
      pick);
    cudaCheckError(cudaGetLastError());
  }

  void assign(const int*   centroids,
              int*         labels,
              const int    numClusters,
              const double cutoff,
              const bool   nearestAssignment,
              cudaStream_t stream) const {
    assignMatrixClustersKernel<<<(numItems_ + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(distances_,
                                                                                                     centroids,
                                                                                                     labels,
                                                                                                     numItems_,
                                                                                                     numClusters,
                                                                                                     cutoff,
                                                                                                     nearestAssignment);
    cudaCheckError(cudaGetLastError());
  }

 private:
  cuda::std::span<const double> distances_;
  int                           numItems_;
};

template <FingerprintSimilarityMetric Metric> class FingerprintSimilarityProvider {
 public:
  FingerprintSimilarityProvider(cuda::std::span<const std::uint32_t> fingerprints,
                                const int                            numItems,
                                const int                            numWords,
                                cudaStream_t                         stream)
      : fingerprints_(fingerprints),
        numItems_(numItems),
        numWords_(numWords),
        bitCounts_(numItems, stream) {
    if (numItems > 0) {
      fingerprintBitCountsKernel<<<(numItems + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(fingerprints_,
                                                                                                      bitCounts_.data(),
                                                                                                      numItems_,
                                                                                                      numWords_);
      cudaCheckError(cudaGetLastError());
    }
  }

  void suppress(std::uint8_t* active, const int leader, const double cutoff, cudaStream_t stream) const {
    suppressFingerprintCandidatesKernel<Metric>
      <<<(numItems_ + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(fingerprints_,
                                                                             bitCounts_.data(),
                                                                             active,
                                                                             numItems_,
                                                                             numWords_,
                                                                             leader,
                                                                             1.0 - cutoff);
    cudaCheckError(cudaGetLastError());
  }

  void updateMinimumDistances(double*       minimumDistances,
                              std::uint8_t* selected,
                              const int     pick,
                              cudaStream_t  stream) const {
    updateFingerprintMinimumDistancesKernel<Metric>
      <<<(numItems_ + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(fingerprints_,
                                                                             bitCounts_.data(),
                                                                             minimumDistances,
                                                                             selected,
                                                                             numItems_,
                                                                             numWords_,
                                                                             pick);
    cudaCheckError(cudaGetLastError());
  }

  void assign(const int*   centroids,
              int*         labels,
              const int    numClusters,
              const double cutoff,
              const bool   nearestAssignment,
              cudaStream_t stream) const {
    assignFingerprintClustersKernel<Metric>
      <<<(numItems_ + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(fingerprints_,
                                                                             bitCounts_.data(),
                                                                             centroids,
                                                                             labels,
                                                                             numItems_,
                                                                             numWords_,
                                                                             numClusters,
                                                                             1.0 - cutoff,
                                                                             nearestAssignment);
    cudaCheckError(cudaGetLastError());
  }

 private:
  cuda::std::span<const std::uint32_t> fingerprints_;
  int                                  numItems_;
  int                                  numWords_;
  AsyncDeviceVector<int>               bitCounts_;
};

template <typename Provider>
PickerResult leaderImpl(const Provider&         provider,
                        const int               numItems,
                        const double            cutoff,
                        const int               pickSize,
                        const std::vector<int>& firstPicks,
                        cudaStream_t            stream) {
  validateFirstPicks(firstPicks, numItems);
  if (pickSize < 0 || pickSize > numItems) {
    throw std::invalid_argument("pick_size must be between 0 and the input size");
  }
  if (numItems == 0) {
    return makePickerResult({}, -1.0, stream);
  }

  const int                       limit = pickSize == 0 ? numItems : pickSize;
  AsyncDeviceVector<std::uint8_t> active(numItems, stream);
  initializeActiveKernel<<<(numItems + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(active.data(), numItems);
  cudaCheckError(cudaGetLastError());

  std::vector<int> picks;
  picks.reserve(std::max(limit, static_cast<int>(firstPicks.size())));
  for (const int pick : firstPicks) {
    picks.push_back(pick);
    provider.suppress(active.data(), pick, cutoff, stream);
  }

  AsyncDevicePtr<int> next(numItems, stream);
  while (static_cast<int>(picks.size()) < limit) {
    next.set(numItems);
    findFirstActiveKernel<<<(numItems + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(active.data(),
                                                                                               numItems,
                                                                                               next.data());
    cudaCheckError(cudaGetLastError());
    int nextHost = numItems;
    next.get(nextHost);
    cudaCheckError(cudaStreamSynchronize(stream));
    if (nextHost == numItems) {
      break;
    }
    picks.push_back(nextHost);
    provider.suppress(active.data(), nextHost, cutoff, stream);
  }
  return makePickerResult(picks, -1.0, stream);
}

template <typename Provider>
PickerResult maxMinImpl(const Provider&         provider,
                        const int               numItems,
                        const int               pickSize,
                        const std::vector<int>& firstPicks,
                        const int               seed,
                        const double            threshold,
                        cudaStream_t            stream) {
  validateFirstPicks(firstPicks, numItems);
  if (pickSize <= 0 || pickSize > numItems) {
    throw std::invalid_argument("pick_size must be positive and no larger than the input size");
  }
  if (numItems == 0) {
    throw std::invalid_argument("cannot pick from an empty input");
  }

  AsyncDeviceVector<double>       minimumDistances(numItems, stream);
  AsyncDeviceVector<std::uint8_t> selected(numItems, stream);
  initializeMinimumDistancesKernel<<<(numItems + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
    minimumDistances.data(),
    selected.data(),
    numItems);
  cudaCheckError(cudaGetLastError());

  std::vector<int> picks = firstPicks;
  if (picks.empty()) {
    picks.push_back(randomFirstPick(numItems, seed));
  }
  for (const int pick : picks) {
    provider.updateMinimumDistances(minimumDistances.data(), selected.data(), pick, stream);
  }
  if (static_cast<int>(picks.size()) >= pickSize) {
    return makePickerResult(picks, -1.0, stream);
  }

  std::size_t                  reductionBytes = 0;
  AsyncDevicePtr<double>       nextDistance(-1.0, stream);
  AsyncDevicePtr<std::int64_t> nextIndex(-1, stream);
  cudaCheckError(cub::DeviceReduce::ArgMax(nullptr,
                                           reductionBytes,
                                           minimumDistances.data(),
                                           nextDistance.data(),
                                           nextIndex.data(),
                                           numItems,
                                           stream));
  AsyncDeviceVector<std::uint8_t> reductionStorage(reductionBytes, stream);

  double lastDistance = -1.0;
  while (static_cast<int>(picks.size()) < pickSize) {
    cudaCheckError(cub::DeviceReduce::ArgMax(reductionStorage.data(),
                                             reductionBytes,
                                             minimumDistances.data(),
                                             nextDistance.data(),
                                             nextIndex.data(),
                                             numItems,
                                             stream));
    double       distanceHost = -1.0;
    std::int64_t indexHost    = -1;
    nextDistance.get(distanceHost);
    nextIndex.get(indexHost);
    cudaCheckError(cudaStreamSynchronize(stream));
    if (threshold >= 0.0 && distanceHost <= threshold) {
      break;
    }
    const int selectedIndex = static_cast<int>(indexHost);
    picks.push_back(selectedIndex);
    lastDistance = distanceHost;
    provider.updateMinimumDistances(minimumDistances.data(), selected.data(), selectedIndex, stream);
  }
  return makePickerResult(picks, lastDistance, stream);
}

template <typename Provider>
ClusteringResult diseImpl(const Provider& provider,
                          const int       numItems,
                          const double    cutoff,
                          const bool      nearestAssignment,
                          cudaStream_t    stream) {
  if (numItems == 0) {
    return {};
  }
  auto                   leaders = leaderImpl(provider, numItems, cutoff, 0, {}, stream);
  AsyncDeviceVector<int> labels(numItems, stream);
  provider.assign(leaders.indices.data(), labels.data(), leaders.count, cutoff, nearestAssignment, stream);
  std::vector<int> labelsHost(numItems);
  std::vector<int> centroidsHost(leaders.count);
  labels.copyToHost(labelsHost.data(), labelsHost.size());
  leaders.indices.copyToHost(centroidsHost.data(), centroidsHost.size());
  cudaCheckError(cudaStreamSynchronize(stream));
  return buildClusteringResult(labelsHost, centroidsHost);
}

template <FingerprintSimilarityMetric Metric>
PickerResult fusedLeaderImpl(cuda::std::span<const std::uint32_t> fingerprints,
                             const int                            numItems,
                             const int                            numWords,
                             const double                         cutoff,
                             const int                            pickSize,
                             const std::vector<int>&              firstPicks,
                             cudaStream_t                         stream) {
  FingerprintSimilarityProvider<Metric> provider(fingerprints, numItems, numWords, stream);
  return leaderImpl(provider, numItems, cutoff, pickSize, firstPicks, stream);
}

template <FingerprintSimilarityMetric Metric>
PickerResult fusedMaxMinImpl(cuda::std::span<const std::uint32_t> fingerprints,
                             const int                            numItems,
                             const int                            numWords,
                             const int                            pickSize,
                             const std::vector<int>&              firstPicks,
                             const int                            seed,
                             const double                         threshold,
                             cudaStream_t                         stream) {
  FingerprintSimilarityProvider<Metric> provider(fingerprints, numItems, numWords, stream);
  return maxMinImpl(provider, numItems, pickSize, firstPicks, seed, threshold, stream);
}

template <FingerprintSimilarityMetric Metric>
ClusteringResult fusedDiseImpl(cuda::std::span<const std::uint32_t> fingerprints,
                               const int                            numItems,
                               const int                            numWords,
                               const double                         cutoff,
                               const bool                           nearestAssignment,
                               cudaStream_t                         stream) {
  FingerprintSimilarityProvider<Metric> provider(fingerprints, numItems, numWords, stream);
  return diseImpl(provider, numItems, cutoff, nearestAssignment, stream);
}

}  // namespace

PickerResult leaderFromDistanceMatrix(const cuda::std::span<const double> distanceMatrix,
                                      const int                           numItems,
                                      const double                        cutoff,
                                      const int                           pickSize,
                                      const std::vector<int>&             firstPicks,
                                      cudaStream_t                        stream) {
  validateDistanceMatrix(distanceMatrix, numItems);
  if (!std::isfinite(cutoff) || cutoff < 0.0) {
    throw std::invalid_argument("cutoff must be finite and non-negative");
  }
  return leaderImpl(MatrixDistanceProvider(distanceMatrix, numItems), numItems, cutoff, pickSize, firstPicks, stream);
}

PickerResult fusedLeaderGpu(const cuda::std::span<const std::uint32_t> fingerprints,
                            const int                                  numFingerprints,
                            const int                                  numWords,
                            const double                               cutoff,
                            const FingerprintSimilarityMetric          metric,
                            const int                                  pickSize,
                            const std::vector<int>&                    firstPicks,
                            cudaStream_t                               stream) {
  validateFingerprintInput(fingerprints, numFingerprints, numWords);
  if (!(cutoff >= 0.0 && cutoff <= 1.0)) {
    throw std::invalid_argument("cutoff must be in [0, 1]");
  }
  if (metric == FingerprintSimilarityMetric::Tanimoto) {
    return fusedLeaderImpl<FingerprintSimilarityMetric::Tanimoto>(fingerprints,
                                                                  numFingerprints,
                                                                  numWords,
                                                                  cutoff,
                                                                  pickSize,
                                                                  firstPicks,
                                                                  stream);
  }
  if (metric == FingerprintSimilarityMetric::Cosine) {
    return fusedLeaderImpl<FingerprintSimilarityMetric::Cosine>(fingerprints,
                                                                numFingerprints,
                                                                numWords,
                                                                cutoff,
                                                                pickSize,
                                                                firstPicks,
                                                                stream);
  }
  throw std::invalid_argument("Unsupported fingerprint similarity metric");
}

PickerResult maxMinFromDistanceMatrix(const cuda::std::span<const double> distanceMatrix,
                                      const int                           numItems,
                                      const int                           pickSize,
                                      const std::vector<int>&             firstPicks,
                                      const int                           seed,
                                      const double                        threshold,
                                      cudaStream_t                        stream) {
  validateDistanceMatrix(distanceMatrix, numItems);
  validateMaxMinThreshold(threshold, std::numeric_limits<double>::max());
  return maxMinImpl(MatrixDistanceProvider(distanceMatrix, numItems),
                    numItems,
                    pickSize,
                    firstPicks,
                    seed,
                    threshold,
                    stream);
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
  validateFingerprintInput(fingerprints, numFingerprints, numWords);
  validateMaxMinThreshold(threshold, 1.0);
  if (metric == FingerprintSimilarityMetric::Tanimoto) {
    return fusedMaxMinImpl<FingerprintSimilarityMetric::Tanimoto>(fingerprints,
                                                                  numFingerprints,
                                                                  numWords,
                                                                  pickSize,
                                                                  firstPicks,
                                                                  seed,
                                                                  threshold,
                                                                  stream);
  }
  if (metric == FingerprintSimilarityMetric::Cosine) {
    return fusedMaxMinImpl<FingerprintSimilarityMetric::Cosine>(fingerprints,
                                                                numFingerprints,
                                                                numWords,
                                                                pickSize,
                                                                firstPicks,
                                                                seed,
                                                                threshold,
                                                                stream);
  }
  throw std::invalid_argument("Unsupported fingerprint similarity metric");
}

ClusteringResult diseFromDistanceMatrix(const cuda::std::span<const double> distanceMatrix,
                                        const int                           numItems,
                                        const double                        cutoff,
                                        const bool                          nearestAssignment,
                                        cudaStream_t                        stream) {
  validateDistanceMatrix(distanceMatrix, numItems);
  if (!std::isfinite(cutoff) || cutoff < 0.0) {
    throw std::invalid_argument("cutoff must be finite and non-negative");
  }
  return diseImpl(MatrixDistanceProvider(distanceMatrix, numItems), numItems, cutoff, nearestAssignment, stream);
}

ClusteringResult fusedDiseGpu(const cuda::std::span<const std::uint32_t> fingerprints,
                              const int                                  numFingerprints,
                              const int                                  numWords,
                              const double                               cutoff,
                              const FingerprintSimilarityMetric          metric,
                              const bool                                 nearestAssignment,
                              cudaStream_t                               stream) {
  validateFingerprintInput(fingerprints, numFingerprints, numWords);
  if (!(cutoff >= 0.0 && cutoff <= 1.0)) {
    throw std::invalid_argument("cutoff must be in [0, 1]");
  }
  if (metric == FingerprintSimilarityMetric::Tanimoto) {
    return fusedDiseImpl<FingerprintSimilarityMetric::Tanimoto>(fingerprints,
                                                                numFingerprints,
                                                                numWords,
                                                                cutoff,
                                                                nearestAssignment,
                                                                stream);
  }
  if (metric == FingerprintSimilarityMetric::Cosine) {
    return fusedDiseImpl<FingerprintSimilarityMetric::Cosine>(fingerprints,
                                                              numFingerprints,
                                                              numWords,
                                                              cutoff,
                                                              nearestAssignment,
                                                              stream);
  }
  throw std::invalid_argument("Unsupported fingerprint similarity metric");
}

}  // namespace nvMolKit
