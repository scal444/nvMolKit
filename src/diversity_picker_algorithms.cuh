// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DIVERSITY_PICKER_ALGORITHMS_CUH
#define NVMOLKIT_DIVERSITY_PICKER_ALGORITHMS_CUH

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cub/device/device_reduce.cuh>
#include <stdexcept>
#include <vector>

#include "src/clustering_result.h"
#include "src/diversity_pickers.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"
#include "src/utils/host_vector.h"

// Leader, MaxMin, and DISE are written once against a distance provider. A provider implements
//
//   int size() const;
//   template <typename Op> void forEachDistance(const int* source, const Op& op, cudaStream_t stream);
//
// forEachDistance evaluates distance(source -> candidate) for each candidate where op.skip(candidate) is false
// and passes it to op.apply(candidate, distance), including candidate == source. `source` is read from device
// memory so the selection loops can enqueue work without host round trips; a source outside [0, size()) makes the
// call a no-op.

namespace nvMolKit::detail {

constexpr int kPickerBlockSize = 256;

inline int pickerGridSize(const int numItems) {
  return (numItems + kPickerBlockSize - 1) / kPickerBlockSize;
}

__device__ __forceinline__ bool validSource(const int source, const int numItems) {
  return source >= 0 && source < numItems;
}

//! Removes candidates within the cutoff of a new leader, optionally recording that leader's ordinal as their label.
struct SuppressOp {
  std::uint8_t* active;
  int*          labels;
  const int*    source;
  int           leaderOrdinal;
  double        cutoff;

  __device__ bool skip(const int candidate) const { return active[candidate] == 0; }
  __device__ void apply(const int candidate, const double distance) const {
    // Leaders always leave the active set, independent of the diagonal or a metric's empty-input convention.
    if (candidate == *source || distance <= cutoff) {
      active[candidate] = 0;
      if (labels != nullptr) {
        labels[candidate] = leaderOrdinal;
      }
    }
  }
};

//! Folds a new pick into each unselected candidate's distance to its nearest pick.
struct MinDistanceOp {
  double*       minimumDistances;
  std::uint8_t* selected;
  const int*    source;

  __device__ bool skip(const int candidate) const { return selected[candidate] != 0; }
  __device__ void apply(const int candidate, const double distance) const {
    if (candidate == *source) {
      selected[candidate]         = 1;
      minimumDistances[candidate] = -DBL_MAX;
    } else {
      minimumDistances[candidate] = fmin(minimumDistances[candidate], distance);
    }
  }
};

//! Reassigns non-centroids to a centroid that is strictly nearer than any centroid visited before it.
struct NearestOp {
  double*             bestDistances;
  int*                labels;
  const std::uint8_t* isCentroid;
  int                 clusterOrdinal;

  __device__ bool skip(const int candidate) const { return isCentroid[candidate] != 0; }
  __device__ void apply(const int candidate, const double distance) const {
    if (distance < bestDistances[candidate]) {
      bestDistances[candidate] = distance;
      labels[candidate]        = clusterOrdinal;
    }
  }
};

// Implemented in diversity_pickers.cu.
void launchFindNextActive(const std::uint8_t* active, int numItems, int start, int* next, cudaStream_t stream);
void launchFillDoubles(double* values, int numItems, double value, cudaStream_t stream);
void launchMarkIndices(const int* indices, int count, std::uint8_t* flags, cudaStream_t stream);
void launchRecordMaxMinPick(const double*       candidateDistance,
                            const std::int64_t* candidateIndex,
                            double              threshold,
                            int                 position,
                            int*                picks,
                            int*                count,
                            double*             lastDistance,
                            cudaStream_t        stream);
void validateFirstPicks(const std::vector<int>& firstPicks, int numItems);
void validateMaxMinThreshold(double threshold, double maximum);
void validateUnitCutoff(double cutoff);
int  randomFirstPick(int poolSize, int seed);
ClusteringResult buildClusteringResult(const std::vector<int>& labels, const std::vector<int>& centroids);

//! Returns the first @p count picks as an exactly sized result.
inline PickerResult makePickerResult(const AsyncDeviceVector<int>& picks,
                                     const int                     count,
                                     const double                  lastDistance,
                                     cudaStream_t                  stream) {
  PickerResult result{AsyncDeviceVector<int>(count, stream), lastDistance};
  if (count > 0) {
    cudaCheckError(
      cudaMemcpyAsync(result.indices.data(), picks.data(), count * sizeof(int), cudaMemcpyDeviceToDevice, stream));
  }
  return result;
}

template <typename Provider>
PickerResult leaderPick(Provider&               provider,
                        const double            cutoff,
                        const int               pickSize,
                        const std::vector<int>& firstPicks,
                        int*                    labels,
                        cudaStream_t            stream) {
  const int numItems = provider.size();
  validateFirstPicks(firstPicks, numItems);
  if (pickSize < 0 || pickSize > numItems) {
    throw std::invalid_argument("pick_size must be between 0 and the input size");
  }
  if (numItems == 0) {
    return PickerResult{AsyncDeviceVector<int>(0, stream), -1.0};
  }

  const int                       limit    = pickSize == 0 ? numItems : pickSize;
  const auto                      capacity = std::max<std::size_t>(limit, firstPicks.size());
  AsyncDeviceVector<int>          picks(capacity, stream);
  AsyncDeviceVector<std::uint8_t> active(numItems, stream);
  cudaCheckError(cudaMemsetAsync(active.data(), 1, numItems, stream));
  if (!firstPicks.empty()) {
    picks.copyFromHost(firstPicks, firstPicks.size());
  }

  int count = 0;
  for (; count < static_cast<int>(firstPicks.size()); ++count) {
    provider.forEachDistance(picks.data() + count,
                             SuppressOp{active.data(), labels, picks.data() + count, count, cutoff},
                             stream);
  }

  // After any forced picks, leaders are the first remaining candidate in input order, so each scan resumes after
  // the previous leader. The suppression pass is enqueued before the host reads the pick; an exhausted scan yields
  // numItems, which providers treat as a no-op source.
  PinnedHostVector<int> nextHost(1);
  int                   scanStart = 0;
  while (count < limit) {
    int* next = picks.data() + count;
    launchFindNextActive(active.data(), numItems, scanStart, next, stream);
    provider.forEachDistance(next, SuppressOp{active.data(), labels, next, count, cutoff}, stream);
    cudaCheckError(cudaMemcpyAsync(nextHost.data(), next, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaStreamSynchronize(stream));
    if (nextHost[0] >= numItems) {
      break;
    }
    scanStart = nextHost[0] + 1;
    ++count;
  }
  return makePickerResult(picks, count, -1.0, stream);
}

template <typename Provider>
PickerResult maxMinPick(Provider&               provider,
                        const int               pickSize,
                        const std::vector<int>& firstPicks,
                        const int               seed,
                        const double            threshold,
                        cudaStream_t            stream) {
  const int numItems = provider.size();
  validateFirstPicks(firstPicks, numItems);
  if (pickSize <= 0 || pickSize > numItems) {
    throw std::invalid_argument("pick_size must be positive and no larger than the input size");
  }

  const std::vector<int> initial = firstPicks.empty() ? std::vector<int>{randomFirstPick(numItems, seed)} : firstPicks;
  const int              numInitial = static_cast<int>(initial.size());
  AsyncDeviceVector<int> picks(std::max(pickSize, numInitial), stream);
  picks.copyFromHost(initial, initial.size());

  AsyncDeviceVector<double>       minimumDistances(numItems, stream);
  AsyncDeviceVector<std::uint8_t> selected(numItems, stream);
  launchFillDoubles(minimumDistances.data(), numItems, DBL_MAX, stream);
  cudaCheckError(cudaMemsetAsync(selected.data(), 0, numItems, stream));
  for (int position = 0; position < numInitial; ++position) {
    const int* source = picks.data() + position;
    provider.forEachDistance(source, MinDistanceOp{minimumDistances.data(), selected.data(), source}, stream);
  }

  AsyncDevicePtr<int>          count(numInitial, stream);
  AsyncDevicePtr<double>       lastDistance(-1.0, stream);
  AsyncDevicePtr<double>       candidateDistance(-1.0, stream);
  AsyncDevicePtr<std::int64_t> candidateIndex(-1, stream);
  std::size_t                  reductionBytes = 0;
  cudaCheckError(cub::DeviceReduce::ArgMax(nullptr,
                                           reductionBytes,
                                           minimumDistances.data(),
                                           candidateDistance.data(),
                                           candidateIndex.data(),
                                           numItems,
                                           stream));
  AsyncDeviceVector<std::uint8_t> reductionStorage(reductionBytes, stream);

  // The whole greedy loop is enqueued without host synchronization. Once the threshold stops selection, the
  // recorded pick for each remaining position is -1 and the provider passes become no-ops.
  for (int position = numInitial; position < pickSize; ++position) {
    cudaCheckError(cub::DeviceReduce::ArgMax(reductionStorage.data(),
                                             reductionBytes,
                                             minimumDistances.data(),
                                             candidateDistance.data(),
                                             candidateIndex.data(),
                                             numItems,
                                             stream));
    launchRecordMaxMinPick(candidateDistance.data(),
                           candidateIndex.data(),
                           threshold,
                           position,
                           picks.data(),
                           count.data(),
                           lastDistance.data(),
                           stream);
    const int* source = picks.data() + position;
    provider.forEachDistance(source, MinDistanceOp{minimumDistances.data(), selected.data(), source}, stream);
  }

  int    countHost        = numInitial;
  double lastDistanceHost = -1.0;
  count.get(countHost);
  lastDistance.get(lastDistanceHost);
  cudaCheckError(cudaStreamSynchronize(stream));
  return makePickerResult(picks, countHost, lastDistanceHost, stream);
}

template <typename Provider>
ClusteringResult diseCluster(Provider&    provider,
                             const double cutoff,
                             const bool   nearestAssignment,
                             cudaStream_t stream) {
  const int numItems = provider.size();
  if (numItems == 0) {
    return {};
  }

  // Sphere exclusion records the first excluding leader for every item, which is the "first" assignment.
  AsyncDeviceVector<int> labels(numItems, stream);
  auto                   leaders      = leaderPick(provider, cutoff, 0, {}, labels.data(), stream);
  const int              numCentroids = static_cast<int>(leaders.indices.size());

  if (nearestAssignment) {
    AsyncDeviceVector<std::uint8_t> isCentroid(numItems, stream);
    AsyncDeviceVector<double>       bestDistances(numItems, stream);
    cudaCheckError(cudaMemsetAsync(isCentroid.data(), 0, numItems, stream));
    launchMarkIndices(leaders.indices.data(), numCentroids, isCentroid.data(), stream);
    launchFillDoubles(bestDistances.data(), numItems, DBL_MAX, stream);
    for (int cluster = 0; cluster < numCentroids; ++cluster) {
      provider.forEachDistance(leaders.indices.data() + cluster,
                               NearestOp{bestDistances.data(), labels.data(), isCentroid.data(), cluster},
                               stream);
    }
  }

  std::vector<int> labelsHost(numItems);
  std::vector<int> centroidsHost(numCentroids);
  labels.copyToHost(labelsHost);
  leaders.indices.copyToHost(centroidsHost);
  cudaCheckError(cudaStreamSynchronize(stream));
  return buildClusteringResult(labelsHost, centroidsHost);
}

}  // namespace nvMolKit::detail

#endif  // NVMOLKIT_DIVERSITY_PICKER_ALGORITHMS_CUH
