// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/bitbirch.h"
#include "src/bitbirch_common.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/nvtx.h"

namespace nvMolKit {
namespace {

enum class BitBirchStatus : int {
  Success,
  NodeCapacity,
  EntryCapacity,
  SummaryCapacity,
  InvalidTree
};

constexpr int summaryEntriesPerPage = 4096;

template <typename Component> class PagedSummaryArena {
 public:
  PagedSummaryArena(const int numBits, const int numWords, const cudaStream_t stream)
      : numBits_(numBits),
        numWords_(numWords),
        stream_(stream) {
    linearSumPagePointers_.setStream(stream);
    centroidPagePointers_.setStream(stream);
  }

  void reserve(const int entries) {
    const int requiredPages = (entries + summaryEntriesPerPage - 1) / summaryEntriesPerPage;
    if (requiredPages <= static_cast<int>(linearSumPages_.size())) {
      return;
    }
    while (static_cast<int>(linearSumPages_.size()) < requiredPages) {
      linearSumPages_.emplace_back(static_cast<std::size_t>(summaryEntriesPerPage) * numBits_, stream_);
      centroidPages_.emplace_back(static_cast<std::size_t>(summaryEntriesPerPage) * numWords_, stream_);
    }
    std::vector<Component*> linearSumPointers;
    std::vector<std::uint32_t*> centroidPointers;
    linearSumPointers.reserve(linearSumPages_.size());
    centroidPointers.reserve(centroidPages_.size());
    for (auto& page : linearSumPages_) {
      linearSumPointers.push_back(page.data());
    }
    for (auto& page : centroidPages_) {
      centroidPointers.push_back(page.data());
    }
    linearSumPagePointers_.setFromVector(linearSumPointers);
    centroidPagePointers_.setFromVector(centroidPointers);
  }

  Component** linearSumPages() const noexcept { return linearSumPagePointers_.data(); }
  std::uint32_t** centroidPages() const noexcept { return centroidPagePointers_.data(); }
  int capacity() const noexcept { return static_cast<int>(linearSumPages_.size()) * summaryEntriesPerPage; }

 private:
  int                                                  numBits_;
  int                                                  numWords_;
  cudaStream_t                                         stream_;
  std::vector<AsyncDeviceVector<Component>>            linearSumPages_;
  std::vector<AsyncDeviceVector<std::uint32_t>>        centroidPages_;
  AsyncDeviceVector<Component*>                        linearSumPagePointers_;
  AsyncDeviceVector<std::uint32_t*>                    centroidPagePointers_;
};

template <typename Component> struct TreeStorage {
  const std::uint32_t* fingerprints;
  int*            nodeHeads;
  int*            nodeSizes;
  int*            nodeParents;
  std::uint8_t*   nodeLeaves;
  int*            entryNext;
  int*            entryChildren;
  std::uint32_t*  entryCounts;
  int*            entrySummarySlots;
  int*            entryFingerprintIndices;
  Component**     linearSumPages;
  std::uint32_t** centroidPages;
  int*            entryClusterIds;
  int*            labels;
  std::uint32_t*  centroids;
  int*            root;
  int*            nodeCursor;
  int*            entryCursor;
  int*            summaryCursor;
  int*            numClusters;
  BitBirchStatus* status;
  int             maxNodes;
  int             maxEntries;
  int             maxSummaries;
  int             numWords;
  int             numBits;
};

template <typename Component>
__device__ __forceinline__ Component& materializedLinearSum(TreeStorage<Component>& storage,
                                                            const int               entry,
                                                            const int               bit) {
  const int slot = storage.entrySummarySlots[entry];
  return storage.linearSumPages[slot / summaryEntriesPerPage]
                               [static_cast<std::size_t>(slot % summaryEntriesPerPage) * storage.numBits + bit];
}

template <typename Component>
__device__ __forceinline__ Component linearSum(const TreeStorage<Component>& storage, const int entry, const int bit) {
  const int fingerprintIndex = storage.entryFingerprintIndices[entry];
  if (fingerprintIndex >= 0) {
    const std::uint32_t word =
      storage.fingerprints[static_cast<std::size_t>(fingerprintIndex) * storage.numWords + bit / 32];
    return static_cast<Component>((word >> (bit % 32)) & 1U);
  }
  const int slot = storage.entrySummarySlots[entry];
  return storage.linearSumPages[slot / summaryEntriesPerPage]
                               [static_cast<std::size_t>(slot % summaryEntriesPerPage) * storage.numBits + bit];
}

template <typename Component>
__device__ __forceinline__ int allocateNode(TreeStorage<Component>& storage, const bool leaf, const int parent) {
  const int node = (*storage.nodeCursor)++;
  if (node >= storage.maxNodes) {
    *storage.status = BitBirchStatus::NodeCapacity;
    return -1;
  }
  storage.nodeHeads[node]   = -1;
  storage.nodeSizes[node]   = 0;
  storage.nodeParents[node] = parent;
  storage.nodeLeaves[node]  = leaf;
  return node;
}

template <typename Component> __device__ __forceinline__ int allocateEntry(TreeStorage<Component>& storage) {
  const int entry = (*storage.entryCursor)++;
  if (entry >= storage.maxEntries) {
    *storage.status = BitBirchStatus::EntryCapacity;
    return -1;
  }
  storage.entryNext[entry]     = -1;
  storage.entryChildren[entry] = -1;
  storage.entryCounts[entry]   = 0;
  storage.entrySummarySlots[entry]       = -1;
  storage.entryFingerprintIndices[entry] = -1;
  return entry;
}

template <typename Component>
__device__ __forceinline__ bool materializeEntry(TreeStorage<Component>& storage, const int entry) {
  if (storage.entrySummarySlots[entry] >= 0) {
    return true;
  }
  const int slot = atomicAdd(storage.summaryCursor, 1);
  if (slot >= storage.maxSummaries) {
    *storage.status = BitBirchStatus::SummaryCapacity;
    return false;
  }
  const int fingerprintIndex              = storage.entryFingerprintIndices[entry];
  storage.entrySummarySlots[entry]       = slot;
  for (int bit = 0; bit < storage.numBits; ++bit) {
    Component value = 0;
    if (fingerprintIndex >= 0) {
      const std::uint32_t word =
        storage.fingerprints[static_cast<std::size_t>(fingerprintIndex) * storage.numWords + bit / 32];
      value = static_cast<Component>((word >> (bit % 32)) & 1U);
    }
    materializedLinearSum(storage, entry, bit) = value;
  }
  if (fingerprintIndex >= 0) {
    for (int word = 0; word < storage.numWords; ++word) {
      storage.centroidPages[slot / summaryEntriesPerPage]
                           [static_cast<std::size_t>(slot % summaryEntriesPerPage) * storage.numWords + word] =
        storage.fingerprints[static_cast<std::size_t>(fingerprintIndex) * storage.numWords + word];
    }
  }
  storage.entryFingerprintIndices[entry] = -1;
  return true;
}

template <typename Component>
__device__ __forceinline__ void appendEntry(TreeStorage<Component>& storage, const int node, const int entry) {
  if (storage.nodeHeads[node] < 0) {
    storage.nodeHeads[node] = entry;
  } else {
    int tail = storage.nodeHeads[node];
    while (storage.entryNext[tail] >= 0) {
      tail = storage.entryNext[tail];
    }
    storage.entryNext[tail] = entry;
  }
  storage.entryNext[entry] = -1;
  ++storage.nodeSizes[node];
}

template <typename Component>
__device__ __forceinline__ std::uint32_t centroidWord(const TreeStorage<Component>& storage,
                                                      const int                     entry,
                                                      const int                     word) {
  const int fingerprintIndex = storage.entryFingerprintIndices[entry];
  if (fingerprintIndex >= 0) {
    return storage.fingerprints[static_cast<std::size_t>(fingerprintIndex) * storage.numWords + word];
  }
  const int slot = storage.entrySummarySlots[entry];
  return storage.centroidPages[slot / summaryEntriesPerPage]
                              [static_cast<std::size_t>(slot % summaryEntriesPerPage) * storage.numWords + word];
}

template <typename Component>
__device__ __forceinline__ void refreshCentroid(TreeStorage<Component>& storage, const int entry) {
  if (storage.entryFingerprintIndices[entry] >= 0) {
    return;
  }
  const auto count = static_cast<std::uint64_t>(storage.entryCounts[entry]);
  for (int word = 0; word < storage.numWords; ++word) {
    std::uint32_t result = 0;
    for (int bitInWord = 0; bitInWord < 32; ++bitInWord) {
      const int bit = word * 32 + bitInWord;
      if (bitbirch::majorityCentroidBit(static_cast<std::uint64_t>(linearSum(storage, entry, bit)), count)) {
        result |= std::uint32_t{1} << bitInWord;
      }
    }
    const int slot = storage.entrySummarySlots[entry];
    storage.centroidPages[slot / summaryEntriesPerPage]
                         [static_cast<std::size_t>(slot % summaryEntriesPerPage) * storage.numWords + word] = result;
  }
}

template <typename Component>
__device__ __forceinline__ double entryToFingerprintSimilarity(const TreeStorage<Component>& storage,
                                                               const int                     entry,
                                                               const std::uint32_t*          fingerprint) {
  int intersection = 0;
  int unionCount   = 0;
  for (int word = 0; word < storage.numWords; ++word) {
    const std::uint32_t centroid = centroidWord(storage, entry, word);
    intersection += __popc(centroid & fingerprint[word]);
    unionCount += __popc(centroid | fingerprint[word]);
  }
  return unionCount > 0 ? static_cast<double>(intersection) / unionCount : 1.0;
}

template <typename Component>
__device__ __forceinline__ double entrySimilarity(const TreeStorage<Component>& storage, const int lhs, const int rhs) {
  int intersection = 0;
  int unionCount   = 0;
  for (int word = 0; word < storage.numWords; ++word) {
    const std::uint32_t lhsWord = centroidWord(storage, lhs, word);
    const std::uint32_t rhsWord = centroidWord(storage, rhs, word);
    intersection += __popc(lhsWord & rhsWord);
    unionCount += __popc(lhsWord | rhsWord);
  }
  return unionCount > 0 ? static_cast<double>(intersection) / unionCount : 1.0;
}

template <typename Component>
__device__ __forceinline__ int closestEntry(const TreeStorage<Component>& storage,
                                            const int                     node,
                                            const std::uint32_t*          fingerprint) {
  int    best           = storage.nodeHeads[node];
  double bestSimilarity = -1.0;
  for (int entry = storage.nodeHeads[node]; entry >= 0; entry = storage.entryNext[entry]) {
    const double similarity = entryToFingerprintSimilarity(storage, entry, fingerprint);
    if (similarity > bestSimilarity) {
      bestSimilarity = similarity;
      best           = entry;
    }
  }
  return best;
}

template <typename Component>
__device__ __forceinline__ bitbirch::ISimTanimotoTerms entryISimTerms(const TreeStorage<Component>& storage,
                                                                      const int                     entry) {
  bitbirch::ISimTanimotoTerms terms{};
  const auto                  count = static_cast<std::uint64_t>(storage.entryCounts[entry]);
  for (int bit = 0; bit < storage.numBits; ++bit) {
    bitbirch::accumulateISimTanimotoTerm(terms, static_cast<std::uint64_t>(linearSum(storage, entry, bit)), count);
  }
  return terms;
}

template <typename Component>
__device__ __forceinline__ bitbirch::ISimTanimotoTerms combinedISimTerms(const TreeStorage<Component>& storage,
                                                                         const int                     entry,
                                                                         const std::uint32_t*          fingerprint) {
  bitbirch::ISimTanimotoTerms terms{};
  const auto                  combinedCount = static_cast<std::uint64_t>(storage.entryCounts[entry]) + 1;
  for (int bit = 0; bit < storage.numBits; ++bit) {
    const std::uint32_t word = fingerprint[bit / 32];
    const auto component     = static_cast<std::uint64_t>(linearSum(storage, entry, bit)) + ((word >> (bit % 32)) & 1U);
    bitbirch::accumulateISimTanimotoTerm(terms, component, combinedCount);
  }
  return terms;
}

template <typename Component>
__device__ __forceinline__ void initializeLeafEntry(TreeStorage<Component>& storage,
                                                    const int               entry,
                                                    const int               fingerprintIndex) {
  storage.entryCounts[entry] = 1;
  storage.entryFingerprintIndices[entry] = fingerprintIndex;
}

template <typename Component>
__device__ __forceinline__ void addFingerprint(TreeStorage<Component>& storage,
                                               const int               entry,
                                               const std::uint32_t*    fingerprint) {
  if (!materializeEntry(storage, entry)) {
    return;
  }
  ++storage.entryCounts[entry];
  for (int bit = 0; bit < storage.numBits; ++bit) {
    materializedLinearSum(storage, entry, bit) +=
      static_cast<Component>((fingerprint[bit / 32] >> (bit % 32)) & 1U);
  }
  refreshCentroid(storage, entry);
}

template <typename Component>
__device__ __forceinline__ void summarizeNode(TreeStorage<Component>& storage, const int node, const int targetEntry) {
  if (!materializeEntry(storage, targetEntry)) {
    return;
  }
  std::uint32_t count = 0;
  for (int bit = 0; bit < storage.numBits; ++bit) {
    materializedLinearSum(storage, targetEntry, bit) = 0;
  }
  for (int entry = storage.nodeHeads[node]; entry >= 0; entry = storage.entryNext[entry]) {
    count += storage.entryCounts[entry];
    for (int bit = 0; bit < storage.numBits; ++bit) {
      materializedLinearSum(storage, targetEntry, bit) += linearSum(storage, entry, bit);
    }
  }
  storage.entryCounts[targetEntry] = count;
  refreshCentroid(storage, targetEntry);
}

template <typename Component>
__device__ __forceinline__ int parentEntry(const TreeStorage<Component>& storage, const int node) {
  const int parent = storage.nodeParents[node];
  if (parent < 0) {
    return -1;
  }
  for (int entry = storage.nodeHeads[parent]; entry >= 0; entry = storage.entryNext[entry]) {
    if (storage.entryChildren[entry] == node) {
      return entry;
    }
  }
  return -1;
}

template <typename Component>
__device__ __forceinline__ bool refreshAncestors(TreeStorage<Component>& storage, int node) {
  while (storage.nodeParents[node] >= 0) {
    const int entry = parentEntry(storage, node);
    if (entry < 0) {
      *storage.status = BitBirchStatus::InvalidTree;
      return false;
    }
    summarizeNode(storage, node, entry);
    node = storage.nodeParents[node];
  }
  return true;
}

template <typename Component>
__device__ __forceinline__ void appendDetachedEntry(TreeStorage<Component>& storage,
                                                    const int               node,
                                                    int&                    tail,
                                                    const int               entry) {
  if (tail < 0) {
    storage.nodeHeads[node] = entry;
  } else {
    storage.entryNext[tail] = entry;
  }
  tail                     = entry;
  storage.entryNext[entry] = -1;
  ++storage.nodeSizes[node];
}

template <typename Component>
__device__ __forceinline__ int splitNodeWithSeeds(TreeStorage<Component>& storage,
                                                  const int               node,
                                                  const int               branchingFactor,
                                                  const int               lhsSeed,
                                                  const int               rhsSeed) {
  const int oldHead  = storage.nodeHeads[node];
  const int maxGroup = (storage.nodeSizes[node] + 1) / 2;
  const int sibling  = allocateNode(storage, storage.nodeLeaves[node] != 0, storage.nodeParents[node]);
  if (sibling < 0) {
    return -1;
  }
  storage.nodeHeads[node] = -1;
  storage.nodeSizes[node] = 0;
  int lhsTail             = -1;
  int rhsTail             = -1;
  int lhsAssigned         = 1;
  int rhsAssigned         = 1;
  for (int entry = oldHead; entry >= 0;) {
    const int next = storage.entryNext[entry];
    if (entry != lhsSeed && entry != rhsSeed) {
      const double lhsSimilarity = entrySimilarity(storage, entry, lhsSeed);
      const double rhsSimilarity = entrySimilarity(storage, entry, rhsSeed);
      bool assignLeft = lhsSimilarity > rhsSimilarity || (lhsSimilarity == rhsSimilarity && lhsAssigned <= rhsAssigned);
      if (lhsAssigned >= maxGroup) {
        assignLeft = false;
      } else if (rhsAssigned >= maxGroup) {
        assignLeft = true;
      }
      if (assignLeft) {
        appendDetachedEntry(storage, node, lhsTail, entry);
        ++lhsAssigned;
      } else {
        appendDetachedEntry(storage, sibling, rhsTail, entry);
        ++rhsAssigned;
      }
    }
    entry = next;
  }
  storage.entryNext[lhsSeed] = storage.nodeHeads[node];
  storage.nodeHeads[node]    = lhsSeed;
  ++storage.nodeSizes[node];
  storage.entryNext[rhsSeed] = storage.nodeHeads[sibling];
  storage.nodeHeads[sibling] = rhsSeed;
  ++storage.nodeSizes[sibling];
  if (!storage.nodeLeaves[node]) {
    for (int entry = storage.nodeHeads[node]; entry >= 0; entry = storage.entryNext[entry]) {
      storage.nodeParents[storage.entryChildren[entry]] = node;
    }
    for (int entry = storage.nodeHeads[sibling]; entry >= 0; entry = storage.entryNext[entry]) {
      storage.nodeParents[storage.entryChildren[entry]] = sibling;
    }
  }

  const int parent = storage.nodeParents[node];
  if (parent < 0) {
    const int newRoot  = allocateNode(storage, false, -1);
    const int lhsEntry = allocateEntry(storage);
    const int rhsEntry = allocateEntry(storage);
    if (newRoot < 0 || lhsEntry < 0 || rhsEntry < 0) {
      return -1;
    }
    storage.nodeParents[node]       = newRoot;
    storage.nodeParents[sibling]    = newRoot;
    storage.entryChildren[lhsEntry] = node;
    storage.entryChildren[rhsEntry] = sibling;
    summarizeNode(storage, node, lhsEntry);
    summarizeNode(storage, sibling, rhsEntry);
    appendEntry(storage, newRoot, lhsEntry);
    appendEntry(storage, newRoot, rhsEntry);
    *storage.root = newRoot;
    return newRoot;
  }

  const int oldParentEntry = parentEntry(storage, node);
  const int siblingEntry   = allocateEntry(storage);
  if (oldParentEntry < 0 || siblingEntry < 0) {
    *storage.status = oldParentEntry < 0 ? BitBirchStatus::InvalidTree : *storage.status;
    return -1;
  }
  summarizeNode(storage, node, oldParentEntry);
  storage.entryChildren[siblingEntry] = sibling;
  summarizeNode(storage, sibling, siblingEntry);
  appendEntry(storage, parent, siblingEntry);
  return parent;
}

template <typename Component>
__device__ __forceinline__ int splitNode(TreeStorage<Component>& storage, int node, const int branchingFactor) {
  while (storage.nodeSizes[node] > branchingFactor) {
    int    lhsSeed        = storage.nodeHeads[node];
    int    rhsSeed        = storage.entryNext[lhsSeed];
    double seedSimilarity = entrySimilarity(storage, lhsSeed, rhsSeed);
    for (int lhs = storage.nodeHeads[node]; lhs >= 0; lhs = storage.entryNext[lhs]) {
      for (int rhs = storage.entryNext[lhs]; rhs >= 0; rhs = storage.entryNext[rhs]) {
        const double similarity = entrySimilarity(storage, lhs, rhs);
        if (similarity < seedSimilarity) {
          seedSimilarity = similarity;
          lhsSeed        = lhs;
          rhsSeed        = rhs;
        }
      }
    }
    const int parent = storage.nodeParents[node];
    node             = splitNodeWithSeeds(storage, node, branchingFactor, lhsSeed, rhsSeed);
    if (node < 0 || parent < 0) {
      return node;
    }
  }
  return node;
}

constexpr int cooperativeBlockSize = 256;

struct CooperativeScratch {
  int       nodeEntries[cooperativeBlockSize];
  int       entries[cooperativeBlockSize];
  int       otherEntries[cooperativeBlockSize];
  long long orders[cooperativeBlockSize];
  double    values[cooperativeBlockSize];
  double    otherValues[cooperativeBlockSize];
  int       next;
  int       count;
  int       bestEntry;
  int       selectedEntry;
  int       sourceEntry;
  int       materializeSlot;
  int       materializeFingerprintIndex;
  int       node;
  int       success;
  double    bestValue;
  double    accumulatedValue;
  double    accumulatedOtherValue;
};

template <typename Component>
__device__ __forceinline__ bool cooperativeMaterializeEntry(TreeStorage<Component>& storage,
                                                            const int               entry,
                                                            CooperativeScratch&     scratch) {
  if (storage.entrySummarySlots[entry] >= 0) {
    return true;
  }
  if (threadIdx.x == 0) {
    scratch.materializeFingerprintIndex = storage.entryFingerprintIndices[entry];
    scratch.materializeSlot             = atomicAdd(storage.summaryCursor, 1);
    scratch.success                     = scratch.materializeSlot < storage.maxSummaries;
    if (scratch.success) {
      storage.entrySummarySlots[entry] = scratch.materializeSlot;
    } else {
      *storage.status = BitBirchStatus::SummaryCapacity;
    }
  }
  __syncthreads();
  if (!scratch.success) {
    return false;
  }
  for (int bit = threadIdx.x; bit < storage.numBits; bit += blockDim.x) {
    Component value = 0;
    if (scratch.materializeFingerprintIndex >= 0) {
      const std::uint32_t word =
        storage.fingerprints[static_cast<std::size_t>(scratch.materializeFingerprintIndex) * storage.numWords +
                             bit / 32];
      value = static_cast<Component>((word >> (bit % 32)) & 1U);
    }
    materializedLinearSum(storage, entry, bit) = value;
  }
  for (int word = threadIdx.x; word < storage.numWords; word += blockDim.x) {
    if (scratch.materializeFingerprintIndex >= 0) {
      const int slot = storage.entrySummarySlots[entry];
      storage.centroidPages[slot / summaryEntriesPerPage]
                           [static_cast<std::size_t>(slot % summaryEntriesPerPage) * storage.numWords + word] =
        storage.fingerprints[static_cast<std::size_t>(scratch.materializeFingerprintIndex) * storage.numWords + word];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    storage.entryFingerprintIndices[entry] = -1;
  }
  __syncthreads();
  return true;
}

__device__ __forceinline__ void cooperativeAccumulateISimTerms(double              commonPairs,
                                                                double              mismatches,
                                                                CooperativeScratch& scratch) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    commonPairs += __shfl_down_sync(0xffffffffU, commonPairs, offset);
    mismatches += __shfl_down_sync(0xffffffffU, mismatches, offset);
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  if (lane == 0) {
    scratch.values[warp]      = commonPairs;
    scratch.otherValues[warp] = mismatches;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    for (int index = 0; index < numWarps; ++index) {
      scratch.accumulatedValue += scratch.values[index];
      scratch.accumulatedOtherValue += scratch.otherValues[index];
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void cooperativeUpdateBestEntry(CooperativeScratch& scratch) {
  const bool active = threadIdx.x < scratch.count;
  double     value  = active ? scratch.values[threadIdx.x] : -1.0;
  int        order  = active ? threadIdx.x : INT_MAX;
  int        entry  = active ? scratch.entries[threadIdx.x] : -1;
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    const double otherValue = __shfl_down_sync(0xffffffffU, value, offset);
    const int    otherOrder = __shfl_down_sync(0xffffffffU, order, offset);
    const int    otherEntry = __shfl_down_sync(0xffffffffU, entry, offset);
    if (otherValue > value || (otherValue == value && otherOrder < order)) {
      value = otherValue;
      order = otherOrder;
      entry = otherEntry;
    }
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  if (lane == 0) {
    scratch.otherValues[warp] = value;
    scratch.orders[warp]      = order;
    scratch.nodeEntries[warp] = entry;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    double    bestValue = scratch.otherValues[0];
    long long bestOrder = scratch.orders[0];
    int       bestEntry = scratch.nodeEntries[0];
    for (int index = 1; index < numWarps; ++index) {
      if (scratch.otherValues[index] > bestValue ||
          (scratch.otherValues[index] == bestValue && scratch.orders[index] < bestOrder)) {
        bestValue = scratch.otherValues[index];
        bestOrder = scratch.orders[index];
        bestEntry = scratch.nodeEntries[index];
      }
    }
    if (bestValue > scratch.bestValue) {
      scratch.bestValue = bestValue;
      scratch.bestEntry = bestEntry;
    }
  }
  __syncthreads();
}

template <typename Component>
__device__ __forceinline__ int cooperativeClosestEntry(const TreeStorage<Component>& storage,
                                                       const int                     node,
                                                       const std::uint32_t*          fingerprint,
                                                       CooperativeScratch&           scratch) {
  if (threadIdx.x == 0) {
    scratch.next      = storage.nodeHeads[node];
    scratch.bestEntry = scratch.next;
    scratch.bestValue = -1.0;
  }
  __syncthreads();
  while (true) {
    if (threadIdx.x == 0) {
      int entry     = scratch.next;
      scratch.count = 0;
      while (entry >= 0 && scratch.count < blockDim.x) {
        scratch.entries[scratch.count++] = entry;
        entry                            = storage.entryNext[entry];
      }
      scratch.next = entry;
    }
    __syncthreads();
    if (scratch.count == 0) {
      break;
    }
    if (threadIdx.x < scratch.count) {
      scratch.values[threadIdx.x] = entryToFingerprintSimilarity(storage, scratch.entries[threadIdx.x], fingerprint);
    }
    cooperativeUpdateBestEntry(scratch);
  }
  return scratch.bestEntry;
}

template <typename Component>
__device__ __forceinline__ bitbirch::ISimTanimotoTerms cooperativeCombinedISimTerms(
  const TreeStorage<Component>& storage,
  const int                     entry,
  const std::uint32_t*          fingerprint,
  CooperativeScratch&           scratch) {
  if (threadIdx.x == 0) {
    scratch.accumulatedValue      = 0.0;
    scratch.accumulatedOtherValue = 0.0;
  }
  __syncthreads();
  const auto combinedCount = static_cast<std::uint64_t>(storage.entryCounts[entry]) + 1;
  for (int base = 0; base < storage.numBits; base += blockDim.x) {
    const int bit = base + threadIdx.x;
    double    commonPairs = 0.0;
    double    mismatches  = 0.0;
    if (bit < storage.numBits) {
      const std::uint32_t word = fingerprint[bit / 32];
      const auto component = static_cast<std::uint64_t>(linearSum(storage, entry, bit)) + ((word >> (bit % 32)) & 1U);
      bitbirch::ISimTanimotoTerms terms{};
      bitbirch::accumulateISimTanimotoTerm(terms, component, combinedCount);
      commonPairs = terms.commonPairs;
      mismatches  = terms.mismatches;
    }
    cooperativeAccumulateISimTerms(commonPairs, mismatches, scratch);
  }
  return {scratch.accumulatedValue, scratch.accumulatedOtherValue};
}

template <typename Component>
__device__ __forceinline__ void cooperativeRefreshCentroid(TreeStorage<Component>& storage, const int entry) {
  if (storage.entryFingerprintIndices[entry] >= 0) {
    __syncthreads();
    return;
  }
  const auto count = static_cast<std::uint64_t>(storage.entryCounts[entry]);
  for (int word = threadIdx.x; word < storage.numWords; word += blockDim.x) {
    std::uint32_t result = 0;
    for (int bitInWord = 0; bitInWord < 32; ++bitInWord) {
      const int bit = word * 32 + bitInWord;
      if (bitbirch::majorityCentroidBit(static_cast<std::uint64_t>(linearSum(storage, entry, bit)), count)) {
        result |= std::uint32_t{1} << bitInWord;
      }
    }
    const int slot = storage.entrySummarySlots[entry];
    storage.centroidPages[slot / summaryEntriesPerPage]
                         [static_cast<std::size_t>(slot % summaryEntriesPerPage) * storage.numWords + word] = result;
  }
  __syncthreads();
}

template <typename Component>
__device__ __forceinline__ void cooperativeInitializeLeafEntry(TreeStorage<Component>& storage,
                                                               const int               entry,
                                                               const int               fingerprintIndex) {
  if (threadIdx.x == 0) {
    storage.entryCounts[entry]             = 1;
    storage.entryFingerprintIndices[entry] = fingerprintIndex;
  }
  __syncthreads();
}

template <typename Component>
__device__ __forceinline__ void cooperativeAddFingerprint(TreeStorage<Component>& storage,
                                                          const int               entry,
                                                          const std::uint32_t*    fingerprint,
                                                          CooperativeScratch&     scratch) {
  if (!cooperativeMaterializeEntry(storage, entry, scratch)) {
    return;
  }
  if (threadIdx.x == 0) {
    ++storage.entryCounts[entry];
  }
  for (int bit = threadIdx.x; bit < storage.numBits; bit += blockDim.x) {
    materializedLinearSum(storage, entry, bit) +=
      static_cast<Component>((fingerprint[bit / 32] >> (bit % 32)) & 1U);
  }
  __syncthreads();
  cooperativeRefreshCentroid(storage, entry);
}

template <typename Component>
__device__ __forceinline__ void cooperativeSummarizeNode(TreeStorage<Component>& storage,
                                                         const int               node,
                                                         const int               targetEntry,
                                                         CooperativeScratch&     scratch) {
  if (!cooperativeMaterializeEntry(storage, targetEntry, scratch)) {
    return;
  }
  if (threadIdx.x == 0) {
    std::uint32_t count = 0;
    for (int entry = storage.nodeHeads[node]; entry >= 0; entry = storage.entryNext[entry]) {
      count += storage.entryCounts[entry];
    }
    storage.entryCounts[targetEntry] = count;
  }
  for (int bit = threadIdx.x; bit < storage.numBits; bit += blockDim.x) {
    Component sum = 0;
    for (int entry = storage.nodeHeads[node]; entry >= 0; entry = storage.entryNext[entry]) {
      sum += linearSum(storage, entry, bit);
    }
    materializedLinearSum(storage, targetEntry, bit) = sum;
  }
  __syncthreads();
  cooperativeRefreshCentroid(storage, targetEntry);
}

template <typename Component>
__device__ __forceinline__ bool cooperativeRefreshAncestors(TreeStorage<Component>& storage,
                                                            int                     node,
                                                            CooperativeScratch&     scratch) {
  while (storage.nodeParents[node] >= 0) {
    if (threadIdx.x == 0) {
      scratch.selectedEntry = parentEntry(storage, node);
      scratch.success       = scratch.selectedEntry >= 0;
      if (!scratch.success) {
        *storage.status = BitBirchStatus::InvalidTree;
      }
    }
    __syncthreads();
    if (!scratch.success) {
      return false;
    }
    cooperativeSummarizeNode(storage, node, scratch.selectedEntry, scratch);
    node = storage.nodeParents[node];
  }
  return true;
}

template <typename Component>
__device__ __forceinline__ int cooperativeSplitNode(TreeStorage<Component>& storage,
                                                    int                     node,
                                                    const int               branchingFactor,
                                                    CooperativeScratch&     scratch) {
  while (storage.nodeSizes[node] > branchingFactor) {
    double    localBest      = 2.0;
    long long localBestOrder = LLONG_MAX;
    int       localLhs       = -1;
    int       localRhs       = -1;
    long long order          = 0;
    if (storage.nodeSizes[node] <= blockDim.x) {
      if (threadIdx.x == 0) {
        int index = 0;
        for (int entry = storage.nodeHeads[node]; entry >= 0; entry = storage.entryNext[entry]) {
          scratch.nodeEntries[index++] = entry;
        }
        scratch.count = index;
      }
      __syncthreads();
      for (int lhsIndex = 0; lhsIndex < scratch.count; ++lhsIndex) {
        for (int rhsIndex = lhsIndex + 1; rhsIndex < scratch.count; ++rhsIndex, ++order) {
          if (order % blockDim.x == threadIdx.x) {
            const int    lhs        = scratch.nodeEntries[lhsIndex];
            const int    rhs        = scratch.nodeEntries[rhsIndex];
            const double similarity = entrySimilarity(storage, lhs, rhs);
            if (similarity < localBest) {
              localBest      = similarity;
              localBestOrder = order;
              localLhs       = lhs;
              localRhs       = rhs;
            }
          }
        }
      }
    } else {
      for (int lhs = storage.nodeHeads[node]; lhs >= 0; lhs = storage.entryNext[lhs]) {
        for (int rhs = storage.entryNext[lhs]; rhs >= 0; rhs = storage.entryNext[rhs], ++order) {
          if (order % blockDim.x == threadIdx.x) {
            const double similarity = entrySimilarity(storage, lhs, rhs);
            if (similarity < localBest) {
              localBest      = similarity;
              localBestOrder = order;
              localLhs       = lhs;
              localRhs       = rhs;
            }
          }
        }
      }
    }
    scratch.values[threadIdx.x]       = localBest;
    scratch.orders[threadIdx.x]       = localBestOrder;
    scratch.entries[threadIdx.x]      = localLhs;
    scratch.otherEntries[threadIdx.x] = localRhs;
    __syncthreads();
    if (threadIdx.x == 0) {
      double    best      = 2.0;
      long long bestOrder = LLONG_MAX;
      int       lhsSeed   = -1;
      int       rhsSeed   = -1;
      for (int index = 0; index < blockDim.x; ++index) {
        if (scratch.values[index] < best || (scratch.values[index] == best && scratch.orders[index] < bestOrder)) {
          best      = scratch.values[index];
          bestOrder = scratch.orders[index];
          lhsSeed   = scratch.entries[index];
          rhsSeed   = scratch.otherEntries[index];
        }
      }
      const int oldParent = storage.nodeParents[node];
      scratch.node        = splitNodeWithSeeds(storage, node, branchingFactor, lhsSeed, rhsSeed);
      scratch.success     = scratch.node >= 0;
      scratch.count       = oldParent < 0;
    }
    __syncthreads();
    if (!scratch.success || scratch.count) {
      return scratch.node;
    }
    node = scratch.node;
  }
  return node;
}

template <typename Component>
__device__ bool cooperativeBuildFingerprintRange(const std::uint32_t*         fingerprints,
                                                 const int                    begin,
                                                 const int                    end,
                                                 const double                 threshold,
                                                 const int                    branchingFactor,
                                                 const BitBirchMergeCriterion mergeCriterion,
                                                 const double                 tolerance,
                                                 const bool                   initialize,
                                                 TreeStorage<Component>&      storage,
                                                 CooperativeScratch&          scratch) {
  if (threadIdx.x == 0) {
    if (initialize) {
      *storage.status      = BitBirchStatus::Success;
      *storage.nodeCursor  = 0;
      *storage.entryCursor = 0;
      *storage.numClusters = 0;
      *storage.root        = allocateNode(storage, true, -1);
      scratch.success      = *storage.root >= 0;
    } else {
      scratch.success = *storage.status == BitBirchStatus::Success;
    }
  }
  __syncthreads();

  for (int fingerprintIndex = begin; fingerprintIndex < end && scratch.success; ++fingerprintIndex) {
    const std::uint32_t* fingerprint = fingerprints + static_cast<std::size_t>(fingerprintIndex) * storage.numWords;
    if (threadIdx.x == 0) {
      scratch.node = *storage.root;
    }
    __syncthreads();
    while (!storage.nodeLeaves[scratch.node]) {
      const int selected = cooperativeClosestEntry(storage, scratch.node, fingerprint, scratch);
      if (threadIdx.x == 0) {
        scratch.node    = storage.entryChildren[selected];
        scratch.success = scratch.node >= 0;
        if (!scratch.success) {
          *storage.status = BitBirchStatus::InvalidTree;
        }
      }
      __syncthreads();
      if (!scratch.success) {
        return false;
      }
    }

    if (storage.nodeHeads[scratch.node] >= 0) {
      const int selected = cooperativeClosestEntry(storage, scratch.node, fingerprint, scratch);
      if (threadIdx.x == 0) {
        scratch.selectedEntry = selected;
      }
    } else if (threadIdx.x == 0) {
      scratch.selectedEntry = -1;
    }
    __syncthreads();

    bool merge = false;
    if (scratch.selectedEntry >= 0) {
      const auto combinedTerms = cooperativeCombinedISimTerms(storage, scratch.selectedEntry, fingerprint, scratch);
      if (threadIdx.x == 0) {
        const auto combinedCount = static_cast<std::uint64_t>(storage.entryCounts[scratch.selectedEntry]) + 1;
        merge                    = bitbirch::isimTanimotoAtLeast(combinedTerms, combinedCount, threshold);
        if (merge && mergeCriterion == BitBirchMergeCriterion::ToleranceDiameter) {
          const auto   oldCount     = static_cast<std::uint64_t>(storage.entryCounts[scratch.selectedEntry]);
          const double oldISim      = bitbirch::isimTanimoto(entryISimTerms(storage, scratch.selectedEntry), oldCount);
          const double combinedISim = bitbirch::isimTanimoto(combinedTerms, combinedCount);
          merge                     = bitbirch::singletonToleranceAllows(oldISim, combinedISim, oldCount, tolerance);
        }
        scratch.count = merge;
      }
      __syncthreads();
      merge = scratch.count != 0;
    }

    if (merge) {
      cooperativeAddFingerprint(storage, scratch.selectedEntry, fingerprint, scratch);
    } else {
      if (threadIdx.x == 0) {
        scratch.selectedEntry = allocateEntry(storage);
        scratch.success       = scratch.selectedEntry >= 0;
      }
      __syncthreads();
      if (!scratch.success) {
        return false;
      }
      cooperativeInitializeLeafEntry(storage, scratch.selectedEntry, fingerprintIndex);
      if (threadIdx.x == 0) {
        appendEntry(storage, scratch.node, scratch.selectedEntry);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      storage.labels[fingerprintIndex] = scratch.selectedEntry;
    }
    __syncthreads();
    cooperativeRefreshAncestors(storage, scratch.node, scratch);
    if (!scratch.success) {
      return false;
    }
    if (storage.nodeSizes[scratch.node] > branchingFactor) {
      const int changedNode = cooperativeSplitNode(storage, scratch.node, branchingFactor, scratch);
      if (changedNode >= 0) {
        cooperativeRefreshAncestors(storage, changedNode, scratch);
      } else if (threadIdx.x == 0) {
        scratch.success = false;
      }
      __syncthreads();
    }
  }
  return scratch.success;
}

template <typename Component>
__device__ bool buildFingerprintRange(const std::uint32_t*         fingerprints,
                                      const int                    begin,
                                      const int                    end,
                                      const double                 threshold,
                                      const int                    branchingFactor,
                                      const BitBirchMergeCriterion mergeCriterion,
                                      const double                 tolerance,
                                      const bool                   initialize,
                                      TreeStorage<Component>&      storage) {
  if (initialize) {
    *storage.status       = BitBirchStatus::Success;
    *storage.nodeCursor   = 0;
    *storage.entryCursor  = 0;
    *storage.numClusters  = 0;
    const int initialRoot = allocateNode(storage, true, -1);
    *storage.root         = initialRoot;
  }

  for (int fingerprintIndex = begin; fingerprintIndex < end; ++fingerprintIndex) {
    const std::uint32_t* fingerprint = fingerprints + static_cast<std::size_t>(fingerprintIndex) * storage.numWords;
    int                  node        = *storage.root;
    while (!storage.nodeLeaves[node]) {
      const int entry = closestEntry(storage, node, fingerprint);
      node            = storage.entryChildren[entry];
      if (node < 0) {
        *storage.status = BitBirchStatus::InvalidTree;
        return false;
      }
    }

    int  selectedEntry = storage.nodeHeads[node] >= 0 ? closestEntry(storage, node, fingerprint) : -1;
    bool merge         = false;
    if (selectedEntry >= 0) {
      const auto combinedTerms = combinedISimTerms(storage, selectedEntry, fingerprint);
      const auto combinedCount = static_cast<std::uint64_t>(storage.entryCounts[selectedEntry]) + 1;
      merge                    = bitbirch::isimTanimotoAtLeast(combinedTerms, combinedCount, threshold);
      if (merge && mergeCriterion == BitBirchMergeCriterion::ToleranceDiameter) {
        const auto   oldCount     = static_cast<std::uint64_t>(storage.entryCounts[selectedEntry]);
        const double oldISim      = bitbirch::isimTanimoto(entryISimTerms(storage, selectedEntry), oldCount);
        const double combinedISim = bitbirch::isimTanimoto(combinedTerms, combinedCount);
        merge                     = bitbirch::singletonToleranceAllows(oldISim, combinedISim, oldCount, tolerance);
      }
    }

    if (merge) {
      addFingerprint(storage, selectedEntry, fingerprint);
    } else {
      selectedEntry = allocateEntry(storage);
      if (selectedEntry < 0) {
        return false;
      }
      initializeLeafEntry(storage, selectedEntry, fingerprintIndex);
      appendEntry(storage, node, selectedEntry);
    }
    storage.labels[fingerprintIndex] = selectedEntry;
    if (!refreshAncestors(storage, node)) {
      return false;
    }
    if (storage.nodeSizes[node] > branchingFactor) {
      const int changedNode = splitNode(storage, node, branchingFactor);
      if (changedNode < 0 || !refreshAncestors(storage, changedNode)) {
        return false;
      }
    }
  }
  return true;
}

template <typename Component>
__device__ void compactLabels(const int begin, const int end, TreeStorage<Component>& storage) {
  for (int entry = 0; entry < *storage.entryCursor; ++entry) {
    storage.entryClusterIds[entry] = -1;
  }
  int clusterCount = 0;
  for (int fingerprintIndex = begin; fingerprintIndex < end; ++fingerprintIndex) {
    const int entry     = storage.labels[fingerprintIndex];
    int       clusterId = storage.entryClusterIds[entry];
    if (clusterId < 0) {
      clusterId                      = clusterCount++;
      storage.entryClusterIds[entry] = clusterId;
      if (storage.centroids != nullptr) {
        for (int word = 0; word < storage.numWords; ++word) {
          storage.centroids[static_cast<std::size_t>(clusterId) * storage.numWords + word] =
            centroidWord(storage, entry, word);
        }
      }
    }
    storage.labels[fingerprintIndex] = clusterId;
  }
  *storage.numClusters = clusterCount;
}

template <typename Component>
__global__ void bitBirchSerialKernel(const std::uint32_t*         fingerprints,
                                     const int                    begin,
                                     const int                    end,
                                     const int                    numFingerprints,
                                     const double                 threshold,
                                     const int                    branchingFactor,
                                     const BitBirchMergeCriterion mergeCriterion,
                                     const double                 tolerance,
                                     const bool                   initialize,
                                     const bool                   finalize,
                                     TreeStorage<Component>       storage) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  if (buildFingerprintRange(fingerprints,
                            begin,
                            end,
                            threshold,
                            branchingFactor,
                            mergeCriterion,
                            tolerance,
                            initialize,
                            storage) && finalize) {
    compactLabels(0, numFingerprints, storage);
  }
}

template <typename Component>
__device__ TreeStorage<Component> partitionStorage(TreeStorage<Component> storage,
                                                   const int              partition,
                                                   const int              nodeStride,
                                                   const int              entryStride) {
  storage.nodeHeads += partition * nodeStride;
  storage.nodeSizes += partition * nodeStride;
  storage.nodeParents += partition * nodeStride;
  storage.nodeLeaves += partition * nodeStride;
  storage.entryNext += partition * entryStride;
  storage.entryChildren += partition * entryStride;
  storage.entryCounts += partition * entryStride;
  storage.entrySummarySlots += partition * entryStride;
  storage.entryFingerprintIndices += partition * entryStride;
  if (storage.entryClusterIds != nullptr) {
    storage.entryClusterIds += partition * entryStride;
  }
  storage.root += partition;
  storage.nodeCursor += partition;
  storage.entryCursor += partition;
  storage.numClusters += partition;
  storage.status += partition;
  storage.centroids  = nullptr;
  storage.maxNodes   = nodeStride;
  storage.maxEntries = entryStride;
  return storage;
}

template <typename Component>
__global__ void bitBirchPartialTreesKernel(const std::uint32_t*         fingerprints,
                                           const int                    numFingerprints,
                                           const int                    partitionSize,
                                           const int                    partitionOffset,
                                           const int                    batchSize,
                                           const bool                   initialize,
                                           const int                    nodeStride,
                                           const int                    entryStride,
                                           const double                 threshold,
                                           const int                    branchingFactor,
                                           const BitBirchMergeCriterion mergeCriterion,
                                           const double                 tolerance,
                                           TreeStorage<Component>       storage) {
  __shared__ CooperativeScratch scratch;
  const int                     partition = blockIdx.x;
  const int                     partitionBegin = partition * partitionSize;
  const int                     begin          = partitionBegin + partitionOffset;
  const int                     end = min(min(begin + batchSize, partitionBegin + partitionSize), numFingerprints);
  auto                          local     = partitionStorage(storage, partition, nodeStride, entryStride);
  cooperativeBuildFingerprintRange(fingerprints,
                                   begin,
                                   end,
                                   threshold,
                                   branchingFactor,
                                   mergeCriterion,
                                   tolerance,
                                   initialize,
                                   local,
                                   scratch);
}

template <typename SummaryComponent>
__device__ __forceinline__ std::uint32_t summaryCentroidWord(const SummaryComponent* linearSums,
                                                             const std::uint32_t     count,
                                                             const int               word) {
  std::uint32_t result = 0;
  for (int bitInWord = 0; bitInWord < 32; ++bitInWord) {
    if (bitbirch::majorityCentroidBit(linearSums[word * 32 + bitInWord], count)) {
      result |= std::uint32_t{1} << bitInWord;
    }
  }
  return result;
}

template <typename Component, typename SummaryComponent>
__device__ __forceinline__ double entryToSummarySimilarity(const TreeStorage<Component>& storage,
                                                           const int                     entry,
                                                           const TreeStorage<SummaryComponent>& sourceStorage,
                                                           const int                           sourceEntry) {
  int intersection = 0;
  int unionCount   = 0;
  for (int word = 0; word < storage.numWords; ++word) {
    const std::uint32_t lhs = centroidWord(storage, entry, word);
    const std::uint32_t rhs = centroidWord(sourceStorage, sourceEntry, word);
    intersection += __popc(lhs & rhs);
    unionCount += __popc(lhs | rhs);
  }
  return unionCount > 0 ? static_cast<double>(intersection) / unionCount : 1.0;
}

template <typename Component, typename SummaryComponent>
__device__ __forceinline__ int cooperativeClosestSummaryEntry(const TreeStorage<Component>& storage,
                                                              const int                     node,
                                                              const TreeStorage<SummaryComponent>& sourceStorage,
                                                              const int                           sourceEntry,
                                                              CooperativeScratch&           scratch) {
  if (threadIdx.x == 0) {
    scratch.next      = storage.nodeHeads[node];
    scratch.bestEntry = scratch.next;
    scratch.bestValue = -1.0;
  }
  __syncthreads();
  while (true) {
    if (threadIdx.x == 0) {
      int entry     = scratch.next;
      scratch.count = 0;
      while (entry >= 0 && scratch.count < blockDim.x) {
        scratch.entries[scratch.count++] = entry;
        entry                            = storage.entryNext[entry];
      }
      scratch.next = entry;
    }
    __syncthreads();
    if (scratch.count == 0) {
      break;
    }
    if (threadIdx.x < scratch.count) {
      scratch.values[threadIdx.x] =
        entryToSummarySimilarity(storage, scratch.entries[threadIdx.x], sourceStorage, sourceEntry);
    }
    cooperativeUpdateBestEntry(scratch);
  }
  return scratch.bestEntry;
}

template <typename Component, typename SummaryComponent>
__device__ __forceinline__ bitbirch::ISimTanimotoTerms cooperativeCombinedSummaryISimTerms(
  const TreeStorage<Component>& storage,
  const int                     entry,
  const TreeStorage<SummaryComponent>& sourceStorage,
  const int                           sourceEntry,
  const std::uint32_t           candidateCount,
  CooperativeScratch&           scratch) {
  if (threadIdx.x == 0) {
    scratch.accumulatedValue      = 0.0;
    scratch.accumulatedOtherValue = 0.0;
  }
  __syncthreads();
  const auto combinedCount = static_cast<std::uint64_t>(storage.entryCounts[entry]) + candidateCount;
  for (int base = 0; base < storage.numBits; base += blockDim.x) {
    const int bit = base + threadIdx.x;
    double    commonPairs = 0.0;
    double    mismatches  = 0.0;
    if (bit < storage.numBits) {
      const auto component =
        static_cast<std::uint64_t>(linearSum(storage, entry, bit)) +
        static_cast<std::uint64_t>(linearSum(sourceStorage, sourceEntry, bit));
      bitbirch::ISimTanimotoTerms terms{};
      bitbirch::accumulateISimTanimotoTerm(terms, component, combinedCount);
      commonPairs = terms.commonPairs;
      mismatches  = terms.mismatches;
    }
    cooperativeAccumulateISimTerms(commonPairs, mismatches, scratch);
  }
  return {scratch.accumulatedValue, scratch.accumulatedOtherValue};
}

template <typename Component, typename SummaryComponent>
__device__ int cooperativeInsertSummary(TreeStorage<Component>& storage,
                                        const TreeStorage<SummaryComponent>& sourceStorage,
                                        const int                         sourceEntry,
                                        const std::uint32_t     candidateCount,
                                        const double            threshold,
                                        const int               branchingFactor,
                                        CooperativeScratch&     scratch) {
  if (threadIdx.x == 0) {
    scratch.node = *storage.root;
  }
  __syncthreads();
  while (!storage.nodeLeaves[scratch.node]) {
    const int entry = cooperativeClosestSummaryEntry(storage, scratch.node, sourceStorage, sourceEntry, scratch);
    if (threadIdx.x == 0) {
      scratch.node    = storage.entryChildren[entry];
      scratch.success = scratch.node >= 0;
      if (!scratch.success) {
        *storage.status = BitBirchStatus::InvalidTree;
      }
    }
    __syncthreads();
    if (!scratch.success) {
      return -1;
    }
  }

  if (storage.nodeHeads[scratch.node] >= 0) {
    const int entry = cooperativeClosestSummaryEntry(storage, scratch.node, sourceStorage, sourceEntry, scratch);
    if (threadIdx.x == 0) {
      scratch.selectedEntry = entry;
    }
  } else if (threadIdx.x == 0) {
    scratch.selectedEntry = -1;
  }
  __syncthreads();

  bool merge = false;
  if (scratch.selectedEntry >= 0) {
    const auto combinedTerms =
      cooperativeCombinedSummaryISimTerms(
        storage, scratch.selectedEntry, sourceStorage, sourceEntry, candidateCount, scratch);
    if (threadIdx.x == 0) {
      const auto combinedCount =
        static_cast<std::uint64_t>(storage.entryCounts[scratch.selectedEntry]) + candidateCount;
      scratch.count = bitbirch::isimTanimotoAtLeast(combinedTerms, combinedCount, threshold);
    }
    __syncthreads();
    merge = scratch.count != 0;
  }

  if (merge) {
    if (!cooperativeMaterializeEntry(storage, scratch.selectedEntry, scratch)) {
      return -1;
    }
    if (threadIdx.x == 0) {
      storage.entryCounts[scratch.selectedEntry] += candidateCount;
    }
    for (int bit = threadIdx.x; bit < storage.numBits; bit += blockDim.x) {
      materializedLinearSum(storage, scratch.selectedEntry, bit) +=
        static_cast<Component>(linearSum(sourceStorage, sourceEntry, bit));
    }
    __syncthreads();
    cooperativeRefreshCentroid(storage, scratch.selectedEntry);
  } else {
    if (threadIdx.x == 0) {
      scratch.selectedEntry = allocateEntry(storage);
      scratch.success       = scratch.selectedEntry >= 0;
      if (scratch.success) {
        storage.entryCounts[scratch.selectedEntry] = candidateCount;
      }
    }
    __syncthreads();
    if (!scratch.success) {
      return -1;
    }
    if (candidateCount == 1 && sourceStorage.entryFingerprintIndices[sourceEntry] >= 0) {
      if (threadIdx.x == 0) {
        storage.entryFingerprintIndices[scratch.selectedEntry] =
          sourceStorage.entryFingerprintIndices[sourceEntry];
      }
      __syncthreads();
    } else {
      if (!cooperativeMaterializeEntry(storage, scratch.selectedEntry, scratch)) {
        return -1;
      }
      for (int bit = threadIdx.x; bit < storage.numBits; bit += blockDim.x) {
        materializedLinearSum(storage, scratch.selectedEntry, bit) =
          static_cast<Component>(linearSum(sourceStorage, sourceEntry, bit));
      }
      __syncthreads();
      cooperativeRefreshCentroid(storage, scratch.selectedEntry);
    }
    if (threadIdx.x == 0) {
      appendEntry(storage, scratch.node, scratch.selectedEntry);
    }
    __syncthreads();
  }
  const int insertedEntry = scratch.selectedEntry;
  cooperativeRefreshAncestors(storage, scratch.node, scratch);
  if (!scratch.success) {
    return -1;
  }
  if (storage.nodeSizes[scratch.node] > branchingFactor) {
    const int changedNode = cooperativeSplitNode(storage, scratch.node, branchingFactor, scratch);
    if (changedNode >= 0) {
      cooperativeRefreshAncestors(storage, changedNode, scratch);
    } else if (threadIdx.x == 0) {
      scratch.success = false;
    }
    __syncthreads();
    if (!scratch.success) {
      return -1;
    }
  }
  return insertedEntry;
}

template <typename Component, typename PartialComponent>
__global__ void bitBirchMergePartialTreesKernel(const int               numFingerprints,
                                                const int               partitionSize,
                                                const int               partialEntryStride,
                                                const int               totalPartialEntries,
                                                TreeStorage<PartialComponent> sourceStorage,
                                                const double            threshold,
                                                const int               branchingFactor,
                                                int*                    partialToFinal,
                                                int*                    uniqueSourceEntries,
                                                TreeStorage<Component>  storage) {
  if (blockIdx.x != 0) {
    return;
  }
  __shared__ CooperativeScratch scratch;
  if (threadIdx.x == 0) {
    *storage.status      = BitBirchStatus::Success;
    *storage.nodeCursor  = 0;
    *storage.entryCursor = 0;
    *storage.numClusters = 0;
    *storage.root        = allocateNode(storage, true, -1);
    scratch.success      = *storage.root >= 0;
  }
  for (int entry = threadIdx.x; entry < totalPartialEntries; entry += blockDim.x) {
    partialToFinal[entry] = -1;
  }
  __syncthreads();

  __shared__ int numUniqueSources;
  if (threadIdx.x == 0) {
    numUniqueSources = 0;
    for (int fingerprintIndex = 0; fingerprintIndex < numFingerprints; ++fingerprintIndex) {
      const int partition   = fingerprintIndex / partitionSize;
      const int localEntry  = storage.labels[fingerprintIndex];
      const int sourceEntry = partition * partialEntryStride + localEntry;
      if (partialToFinal[sourceEntry] == -1) {
        partialToFinal[sourceEntry]              = -2;
        uniqueSourceEntries[numUniqueSources++] = sourceEntry;
      }
    }
  }
  __syncthreads();

  for (int uniqueIndex = 0; uniqueIndex < numUniqueSources; ++uniqueIndex) {
    if (threadIdx.x == 0) {
      scratch.sourceEntry = uniqueSourceEntries[uniqueIndex];
    }
    __syncthreads();
    const int finalEntry = cooperativeInsertSummary(storage,
                                                    sourceStorage,
                                                    scratch.sourceEntry,
                                                    sourceStorage.entryCounts[scratch.sourceEntry],
                                                    threshold,
                                                    branchingFactor,
                                                    scratch);
    if (threadIdx.x == 0) {
      scratch.selectedEntry = finalEntry;
    }
    __syncthreads();
    if (scratch.selectedEntry < 0) {
      return;
    }
    if (threadIdx.x == 0) {
      partialToFinal[scratch.sourceEntry] = scratch.selectedEntry;
    }
  }
  __syncthreads();
  for (int fingerprintIndex = threadIdx.x; fingerprintIndex < numFingerprints; fingerprintIndex += blockDim.x) {
    const int partition                = fingerprintIndex / partitionSize;
    const int localEntry               = storage.labels[fingerprintIndex];
    const int sourceEntry              = partition * partialEntryStride + localEntry;
    storage.labels[fingerprintIndex] = partialToFinal[sourceEntry];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    compactLabels(0, numFingerprints, storage);
  }
}

template <typename Component, typename InputComponent>
__global__ void bitBirchMergeTreeGroupsKernel(const int              numFingerprints,
                                              const int              inputPartitionSize,
                                              const int              inputEntryStride,
                                              const int              numInputTrees,
                                              const int              mergeFanIn,
                                              const int              partitionOffset,
                                              const int              batchSize,
                                              const bool             initialize,
                                              TreeStorage<InputComponent> inputStorage,
                                              const double           threshold,
                                              const int              branchingFactor,
                                              int*                   inputToOutput,
                                              const int              outputNodeStride,
                                              const int              outputEntryStride,
                                              TreeStorage<Component> outputStorage) {
  __shared__ CooperativeScratch scratch;
  const int                     outputTree        = blockIdx.x;
  const int                     firstInputTree    = outputTree * mergeFanIn;
  const int                     pastLastInputTree = min(firstInputTree + mergeFanIn, numInputTrees);
  const int                     groupBegin        = firstInputTree * inputPartitionSize;
  const int                     groupEnd          = min(pastLastInputTree * inputPartitionSize, numFingerprints);
  const int                     begin             = groupBegin + partitionOffset;
  const int                     end               = min(begin + batchSize, groupEnd);
  auto local = partitionStorage(outputStorage, outputTree, outputNodeStride, outputEntryStride);

  if (initialize) {
    if (threadIdx.x == 0) {
      *local.status      = BitBirchStatus::Success;
      *local.nodeCursor  = 0;
      *local.entryCursor = 0;
      *local.numClusters = 0;
      *local.root        = allocateNode(local, true, -1);
      scratch.success    = *local.root >= 0;
    }
    for (int inputEntry = firstInputTree * inputEntryStride + threadIdx.x;
         inputEntry < pastLastInputTree * inputEntryStride;
         inputEntry += blockDim.x) {
      inputToOutput[inputEntry] = -1;
    }
  }
  __syncthreads();

  for (int fingerprintIndex = begin; fingerprintIndex < end; ++fingerprintIndex) {
    if (threadIdx.x == 0) {
      const int inputTree   = fingerprintIndex / inputPartitionSize;
      const int localEntry  = local.labels[fingerprintIndex];
      scratch.sourceEntry   = inputTree * inputEntryStride + localEntry;
      scratch.selectedEntry = inputToOutput[scratch.sourceEntry];
    }
    __syncthreads();
    if (scratch.selectedEntry < 0) {
      const int outputEntry = cooperativeInsertSummary(local,
                                                       inputStorage,
                                                       scratch.sourceEntry,
                                                       inputStorage.entryCounts[scratch.sourceEntry],
                                                       threshold,
                                                       branchingFactor,
                                                       scratch);
      if (threadIdx.x == 0) {
        scratch.selectedEntry = outputEntry;
      }
      __syncthreads();
      if (scratch.selectedEntry < 0) {
        return;
      }
      if (threadIdx.x == 0) {
        inputToOutput[scratch.sourceEntry] = scratch.selectedEntry;
      }
    }
    if (threadIdx.x == 0) {
      local.labels[fingerprintIndex] = scratch.selectedEntry;
    }
    __syncthreads();
  }
}

template <typename Component>
__global__ void bitBirchIndexForestClustersKernel(const int              numFingerprints,
                                                  const int              partitionSize,
                                                  const int              nodeStride,
                                                  const int              entryStride,
                                                  TreeStorage<Component> storage) {
  if (threadIdx.x != 0) {
    return;
  }
  const int tree  = blockIdx.x;
  const int begin = tree * partitionSize;
  const int end   = min(begin + partitionSize, numFingerprints);
  auto      local = partitionStorage(storage, tree, nodeStride, entryStride);
  if (*local.status != BitBirchStatus::Success) {
    return;
  }
  for (int entry = 0; entry < *local.entryCursor; ++entry) {
    local.entryClusterIds[entry] = -1;
  }
  int clusterCount = 0;
  for (int fingerprintIndex = begin; fingerprintIndex < end; ++fingerprintIndex) {
    const int entry = local.labels[fingerprintIndex];
    if (local.entryClusterIds[entry] < 0) {
      local.entryClusterIds[entry] = clusterCount++;
    }
  }
  *local.numClusters = clusterCount;
}

template <typename Component>
__global__ void bitBirchFinalizeForestKernel(const int              numFingerprints,
                                             const int              partitionSize,
                                             const int              nodeStride,
                                             const int              entryStride,
                                             const int*             clusterOffsets,
                                             std::uint32_t*         outputCentroids,
                                             TreeStorage<Component> storage) {
  const int tree   = blockIdx.x;
  const int begin  = tree * partitionSize;
  const int end    = min(begin + partitionSize, numFingerprints);
  const int offset = clusterOffsets[tree];
  auto      local  = partitionStorage(storage, tree, nodeStride, entryStride);

  for (int fingerprintIndex = begin + threadIdx.x; fingerprintIndex < end; fingerprintIndex += blockDim.x) {
    const int entry                = local.labels[fingerprintIndex];
    local.labels[fingerprintIndex] = offset + local.entryClusterIds[entry];
  }
  if (outputCentroids == nullptr) {
    return;
  }
  const auto flatSize = static_cast<std::size_t>(*local.entryCursor) * local.numWords;
  for (std::size_t flatIndex = threadIdx.x; flatIndex < flatSize; flatIndex += blockDim.x) {
    const int entry     = static_cast<int>(flatIndex / local.numWords);
    const int word      = static_cast<int>(flatIndex % local.numWords);
    const int clusterId = local.entryClusterIds[entry];
    if (clusterId >= 0) {
      outputCentroids[static_cast<std::size_t>(offset + clusterId) * local.numWords + word] =
        centroidWord(local, entry, word);
    }
  }
}

template <typename Component>
BitBirchResult launchSerial(const cuda::std::span<const std::uint32_t> fingerprints,
                            const int                                  numFingerprints,
                            const int                                  numWords,
                            const double                               threshold,
                            const int                                  branchingFactor,
                            const BitBirchMergeCriterion               mergeCriterion,
                            const double                               tolerance,
                            const bool                                 returnCentroids,
                            const cudaStream_t                         stream) {
  const int maxNodes   = 2 * numFingerprints + 8;
  const int maxEntries = 3 * numFingerprints + 8;
  const int numBits    = numWords * 32;

  AsyncDeviceVector<int>           nodeHeads(maxNodes, stream);
  AsyncDeviceVector<int>           nodeSizes(maxNodes, stream);
  AsyncDeviceVector<int>           nodeParents(maxNodes, stream);
  AsyncDeviceVector<std::uint8_t>  nodeLeaves(maxNodes, stream);
  AsyncDeviceVector<int>           entryNext(maxEntries, stream);
  AsyncDeviceVector<int>           entryChildren(maxEntries, stream);
  AsyncDeviceVector<std::uint32_t> entryCounts(maxEntries, stream);
  AsyncDeviceVector<int>           entrySummarySlots(maxEntries, stream);
  AsyncDeviceVector<int>           entryFingerprintIndices(maxEntries, stream);
  AsyncDeviceVector<int>           entryClusterIds(maxEntries, stream);
  PagedSummaryArena<Component>     summaryArena(numBits, numWords, stream);
  AsyncDevicePtr<int>              root(0, stream);
  AsyncDevicePtr<int>              nodeCursor(0, stream);
  AsyncDevicePtr<int>              entryCursor(0, stream);
  AsyncDevicePtr<int>              summaryCursor(0, stream);
  AsyncDevicePtr<int>              numClusters(0, stream);
  AsyncDevicePtr<BitBirchStatus>   status(BitBirchStatus::Success, stream);
  summaryArena.reserve(summaryEntriesPerPage);

  BitBirchResult result{
    AsyncDeviceVector<int>(numFingerprints, stream),
    AsyncDeviceVector<std::uint32_t>(returnCentroids ? static_cast<std::size_t>(numFingerprints) * numWords : 0,
                                     stream),
    0,
    numWords};
  TreeStorage<Component> storage{fingerprints.data(),
                                 nodeHeads.data(),
                                 nodeSizes.data(),
                                 nodeParents.data(),
                                 nodeLeaves.data(),
                                 entryNext.data(),
                                 entryChildren.data(),
                                 entryCounts.data(),
                                 entrySummarySlots.data(),
                                 entryFingerprintIndices.data(),
                                 summaryArena.linearSumPages(),
                                 summaryArena.centroidPages(),
                                 entryClusterIds.data(),
                                 result.clusterIds.data(),
                                 returnCentroids ? result.centroids.data() : nullptr,
                                 root.data(),
                                 nodeCursor.data(),
                                 entryCursor.data(),
                                 summaryCursor.data(),
                                 numClusters.data(),
                                 status.data(),
                                 maxNodes,
                                 maxEntries,
                                 summaryArena.capacity(),
                                 numWords,
                                 numBits};
  BitBirchStatus hostStatus{};
  constexpr int  buildBatchSize               = 256;
  constexpr int  reservedSummarySlotsPerInput = 8;
  int            hostSummaryCursor            = 0;
  for (int begin = 0; begin < numFingerprints; begin += buildBatchSize) {
    const int end = std::min(begin + buildBatchSize, numFingerprints);
    const auto requiredSummaries = static_cast<std::int64_t>(hostSummaryCursor) +
                                   static_cast<std::int64_t>(reservedSummarySlotsPerInput) * (end - begin) + 4;
    if (requiredSummaries > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("BitBIRCH serial summary workspace exceeds the supported index range");
    }
    summaryArena.reserve(static_cast<int>(requiredSummaries));
    storage.linearSumPages = summaryArena.linearSumPages();
    storage.centroidPages  = summaryArena.centroidPages();
    storage.maxSummaries   = summaryArena.capacity();
    bitBirchSerialKernel<<<1, 1, 0, stream>>>(fingerprints.data(),
                                              begin,
                                              end,
                                              numFingerprints,
                                              threshold,
                                              branchingFactor,
                                              mergeCriterion,
                                              tolerance,
                                              begin == 0,
                                              end == numFingerprints,
                                              storage);
    cudaCheckError(cudaGetLastError());
    summaryCursor.get(hostSummaryCursor);
    status.get(hostStatus);
    cudaCheckError(cudaStreamSynchronize(stream));
    if (hostStatus != BitBirchStatus::Success) {
      break;
    }
  }
  numClusters.get(result.numClusters);
  cudaCheckError(cudaStreamSynchronize(stream));
  if (hostStatus != BitBirchStatus::Success) {
    throw std::runtime_error("BitBIRCH serial tree capacity or structural failure (status " +
                             std::to_string(static_cast<int>(hostStatus)) + ")");
  }
  return result;
}

template <typename PartialComponent, typename MergeComponent, typename Component>
BitBirchResult launchPartitioned(const cuda::std::span<const std::uint32_t> fingerprints,
                                 const int                                  numFingerprints,
                                 const int                                  numWords,
                                 const double                               threshold,
                                 const int                                  branchingFactor,
                                 const BitBirchMergeCriterion               mergeCriterion,
                                 const double                               tolerance,
                                 const int                                  numPartitions,
                                 const bool                                 returnCentroids,
                                 const cudaStream_t                         stream) {
  const ScopedNvtxRange partitionedRange("BitBIRCH partitioned tree");
  const int             partitionSize = (numFingerprints + numPartitions - 1) / numPartitions;
  const int             nodeStride    = 2 * partitionSize + 8;
  const int             entryStride   = 3 * partitionSize + 8;
  const int             numBits       = numWords * 32;
  const auto            totalNodes    = static_cast<std::size_t>(numPartitions) * nodeStride;
  const auto            totalEntries  = static_cast<std::size_t>(numPartitions) * entryStride;
  if (totalEntries > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("BitBIRCH partition workspace exceeds the supported index range");
  }

  AsyncDeviceVector<int>              partialNodeHeads(totalNodes, stream);
  AsyncDeviceVector<int>              partialNodeSizes(totalNodes, stream);
  AsyncDeviceVector<int>              partialNodeParents(totalNodes, stream);
  AsyncDeviceVector<std::uint8_t>     partialNodeLeaves(totalNodes, stream);
  AsyncDeviceVector<int>              partialEntryNext(totalEntries, stream);
  AsyncDeviceVector<int>              partialEntryChildren(totalEntries, stream);
  AsyncDeviceVector<std::uint32_t>    partialEntryCounts(totalEntries, stream);
  AsyncDeviceVector<int>              partialEntrySummarySlots(totalEntries, stream);
  AsyncDeviceVector<int>              partialEntryFingerprintIndices(totalEntries, stream);
  AsyncDeviceVector<int>              partialEntryClusterIds(totalEntries, stream);
  AsyncDeviceVector<int>              partialRoots(numPartitions, stream);
  AsyncDeviceVector<int>              partialNodeCursors(numPartitions, stream);
  AsyncDeviceVector<int>              partialEntryCursors(numPartitions, stream);
  AsyncDeviceVector<int>              partialClusterCounts(numPartitions, stream);
  AsyncDeviceVector<BitBirchStatus>   partialStatuses(numPartitions, stream);
  AsyncDevicePtr<int>                 partialSummaryCursor(0, stream);
  PagedSummaryArena<PartialComponent> partialSummaryArena(numBits, numWords, stream);
  partialSummaryArena.reserve(summaryEntriesPerPage);

  BitBirchResult result{
    AsyncDeviceVector<int>(numFingerprints, stream),
    AsyncDeviceVector<std::uint32_t>(returnCentroids ? static_cast<std::size_t>(numFingerprints) * numWords : 0,
                                     stream),
    0,
    numWords};
  TreeStorage<PartialComponent> partialStorage{fingerprints.data(),
                                               partialNodeHeads.data(),
                                               partialNodeSizes.data(),
                                               partialNodeParents.data(),
                                               partialNodeLeaves.data(),
                                               partialEntryNext.data(),
                                               partialEntryChildren.data(),
                                               partialEntryCounts.data(),
                                               partialEntrySummarySlots.data(),
                                               partialEntryFingerprintIndices.data(),
                                               partialSummaryArena.linearSumPages(),
                                               partialSummaryArena.centroidPages(),
                                               partialEntryClusterIds.data(),
                                               result.clusterIds.data(),
                                               nullptr,
                                               partialRoots.data(),
                                               partialNodeCursors.data(),
                                               partialEntryCursors.data(),
                                               partialSummaryCursor.data(),
                                               partialClusterCounts.data(),
                                               partialStatuses.data(),
                                               nodeStride,
                                               entryStride,
                                               partialSummaryArena.capacity(),
                                               numWords,
                                               numBits};
  const int     mergeFanIn           = numPartitions > 128 ? 4 : 2;
  const bool    useIntermediateMerge = numPartitions > mergeFanIn;
  std::vector<BitBirchStatus> hostPartialStatuses(numPartitions);
  std::vector<int>            hostPartialClusterCounts(numPartitions);
  int                         totalPartialClusters = 0;
  {
    const ScopedNvtxRange partialRange("BitBIRCH partial-tree construction");
    constexpr int buildBatchSize                 = 256;
    constexpr int reservedSummarySlotsPerInput   = 8;
    int           hostPartialSummaryCursor       = 0;
    for (int partitionOffset = 0; partitionOffset < partitionSize; partitionOffset += buildBatchSize) {
      int batchInputs = 0;
      for (int partition = 0; partition < numPartitions; ++partition) {
        const int begin = partition * partitionSize + partitionOffset;
        const int end = std::min(std::min(begin + buildBatchSize, (partition + 1) * partitionSize), numFingerprints);
        batchInputs += std::max(0, end - begin);
      }
      const auto requiredSummaries = static_cast<std::int64_t>(hostPartialSummaryCursor) +
                                     static_cast<std::int64_t>(reservedSummarySlotsPerInput) * batchInputs +
                                     4 * numPartitions;
      if (requiredSummaries > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("BitBIRCH partial summary workspace exceeds the supported index range");
      }
      partialSummaryArena.reserve(static_cast<int>(requiredSummaries));
      partialStorage.linearSumPages = partialSummaryArena.linearSumPages();
      partialStorage.centroidPages  = partialSummaryArena.centroidPages();
      partialStorage.maxSummaries   = partialSummaryArena.capacity();
      bitBirchPartialTreesKernel<<<numPartitions, cooperativeBlockSize, 0, stream>>>(fingerprints.data(),
                                                                                     numFingerprints,
                                                                                     partitionSize,
                                                                                     partitionOffset,
                                                                                     buildBatchSize,
                                                                                     partitionOffset == 0,
                                                                                     nodeStride,
                                                                                     entryStride,
                                                                                     threshold,
                                                                                     branchingFactor,
                                                                                     mergeCriterion,
                                                                                     tolerance,
                                                                                     partialStorage);
      cudaCheckError(cudaGetLastError());
      partialSummaryCursor.get(hostPartialSummaryCursor);
      partialStatuses.copyToHost(hostPartialStatuses);
      cudaCheckError(cudaStreamSynchronize(stream));
      for (const auto status : hostPartialStatuses) {
        if (status != BitBirchStatus::Success) {
          throw std::runtime_error("BitBIRCH partial-tree capacity or structural failure (status " +
                                   std::to_string(static_cast<int>(status)) + ")");
        }
      }
    }
    bitBirchIndexForestClustersKernel<<<numPartitions, 1, 0, stream>>>(numFingerprints,
                                                                       partitionSize,
                                                                       nodeStride,
                                                                       entryStride,
                                                                       partialStorage);
    cudaCheckError(cudaGetLastError());
    partialStatuses.copyToHost(hostPartialStatuses);
    partialClusterCounts.copyToHost(hostPartialClusterCounts);
    cudaCheckError(cudaStreamSynchronize(stream));
    for (const int clusterCount : hostPartialClusterCounts) {
      totalPartialClusters += clusterCount;
    }
    constexpr int earlyForestNumerator   = 9;
    constexpr int earlyForestDenominator = 10;
    const bool    earlySparseForest =
      useIntermediateMerge && static_cast<std::int64_t>(totalPartialClusters) * earlyForestDenominator >=
                                static_cast<std::int64_t>(numFingerprints) * earlyForestNumerator;
    if (earlySparseForest) {
      const ScopedNvtxRange finalizeRange("BitBIRCH early sparse-forest finalization");
      std::vector<int>      clusterOffsets(numPartitions);
      int                   offset = 0;
      for (int tree = 0; tree < numPartitions; ++tree) {
        clusterOffsets[tree] = offset;
        offset += hostPartialClusterCounts[tree];
      }
      AsyncDeviceVector<int> deviceClusterOffsets(numPartitions, stream);
      deviceClusterOffsets.copyFromHost(clusterOffsets);
      bitBirchFinalizeForestKernel<<<numPartitions, cooperativeBlockSize, 0, stream>>>(
        numFingerprints,
        partitionSize,
        nodeStride,
        entryStride,
        deviceClusterOffsets.data(),
        returnCentroids ? result.centroids.data() : nullptr,
        partialStorage);
      cudaCheckError(cudaGetLastError());
      cudaCheckError(cudaStreamSynchronize(stream));
      result.numClusters = totalPartialClusters;
      return result;
    }
  }

  const int     numMergeTrees = useIntermediateMerge ? (numPartitions + mergeFanIn - 1) / mergeFanIn : numPartitions;
  const int     mergePartitionSize = useIntermediateMerge ? partitionSize * mergeFanIn : partitionSize;
  const int     mergeNodeStride    = 2 * mergePartitionSize + 8;
  const int     mergeEntryStride   = 3 * mergePartitionSize + 8;
  const auto    mergeTotalNodes    = static_cast<std::size_t>(numMergeTrees) * mergeNodeStride;
  const auto    mergeTotalEntries  = static_cast<std::size_t>(numMergeTrees) * mergeEntryStride;
  if (mergeTotalEntries > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("BitBIRCH intermediate merge workspace exceeds the supported index range");
  }

  AsyncDeviceVector<int>           mergeNodeHeads(useIntermediateMerge ? mergeTotalNodes : 0, stream);
  AsyncDeviceVector<int>           mergeNodeSizes(useIntermediateMerge ? mergeTotalNodes : 0, stream);
  AsyncDeviceVector<int>           mergeNodeParents(useIntermediateMerge ? mergeTotalNodes : 0, stream);
  AsyncDeviceVector<std::uint8_t>  mergeNodeLeaves(useIntermediateMerge ? mergeTotalNodes : 0, stream);
  AsyncDeviceVector<int>           mergeEntryNext(useIntermediateMerge ? mergeTotalEntries : 0, stream);
  AsyncDeviceVector<int>           mergeEntryChildren(useIntermediateMerge ? mergeTotalEntries : 0, stream);
  AsyncDeviceVector<std::uint32_t> mergeEntryCounts(useIntermediateMerge ? mergeTotalEntries : 0, stream);
  AsyncDeviceVector<int>           mergeEntrySummarySlots(useIntermediateMerge ? mergeTotalEntries : 0, stream);
  AsyncDeviceVector<int>           mergeEntryFingerprintIndices(useIntermediateMerge ? mergeTotalEntries : 0, stream);
  AsyncDeviceVector<int>           mergeEntryClusterIds(useIntermediateMerge ? mergeTotalEntries : 0, stream);
  AsyncDeviceVector<int>           mergeRoots(useIntermediateMerge ? numMergeTrees : 0, stream);
  AsyncDeviceVector<int>           mergeNodeCursors(useIntermediateMerge ? numMergeTrees : 0, stream);
  AsyncDeviceVector<int>           mergeEntryCursors(useIntermediateMerge ? numMergeTrees : 0, stream);
  AsyncDeviceVector<int>           mergeClusterCounts(useIntermediateMerge ? numMergeTrees : 0, stream);
  AsyncDeviceVector<BitBirchStatus> mergeStatuses(useIntermediateMerge ? numMergeTrees : 0, stream);
  AsyncDeviceVector<int>            partialToMerge(useIntermediateMerge ? totalEntries : 0, stream);
  AsyncDevicePtr<int>               mergeSummaryCursor(0, stream);
  PagedSummaryArena<MergeComponent> mergeSummaryArena(numBits, numWords, stream);
  int totalMergeClusters = 0;
  if (useIntermediateMerge) {
    mergeSummaryArena.reserve(summaryEntriesPerPage);
  }
  TreeStorage<MergeComponent>       mergeStorage{fingerprints.data(),
                                      mergeNodeHeads.data(),
                                      mergeNodeSizes.data(),
                                      mergeNodeParents.data(),
                                      mergeNodeLeaves.data(),
                                      mergeEntryNext.data(),
                                      mergeEntryChildren.data(),
                                      mergeEntryCounts.data(),
                                      mergeEntrySummarySlots.data(),
                                      mergeEntryFingerprintIndices.data(),
                                      mergeSummaryArena.linearSumPages(),
                                      mergeSummaryArena.centroidPages(),
                                      mergeEntryClusterIds.data(),
                                      result.clusterIds.data(),
                                      nullptr,
                                      mergeRoots.data(),
                                      mergeNodeCursors.data(),
                                      mergeEntryCursors.data(),
                                      mergeSummaryCursor.data(),
                                      mergeClusterCounts.data(),
                                      mergeStatuses.data(),
                                      mergeNodeStride,
                                      mergeEntryStride,
                                      mergeSummaryArena.capacity(),
                                      numWords,
                                      numBits};
  if (useIntermediateMerge) {
    const ScopedNvtxRange intermediateRange("BitBIRCH merge round 1");
    constexpr int buildBatchSize               = 256;
    constexpr int reservedSummarySlotsPerInput = 8;
    int           hostMergeSummaryCursor       = 0;
    std::vector<BitBirchStatus> hostMergeStatuses(numMergeTrees);
    for (int partitionOffset = 0; partitionOffset < mergePartitionSize; partitionOffset += buildBatchSize) {
      int batchInputs = 0;
      for (int tree = 0; tree < numMergeTrees; ++tree) {
        const int begin = tree * mergePartitionSize + partitionOffset;
        const int end = std::min(std::min(begin + buildBatchSize, (tree + 1) * mergePartitionSize), numFingerprints);
        batchInputs += std::max(0, end - begin);
      }
      const auto requiredSummaries = static_cast<std::int64_t>(hostMergeSummaryCursor) +
                                     static_cast<std::int64_t>(reservedSummarySlotsPerInput) * batchInputs +
                                     4 * numMergeTrees;
      if (requiredSummaries > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("BitBIRCH merge summary workspace exceeds the supported index range");
      }
      mergeSummaryArena.reserve(static_cast<int>(requiredSummaries));
      mergeStorage.linearSumPages = mergeSummaryArena.linearSumPages();
      mergeStorage.centroidPages  = mergeSummaryArena.centroidPages();
      mergeStorage.maxSummaries   = mergeSummaryArena.capacity();
      bitBirchMergeTreeGroupsKernel<MergeComponent, PartialComponent>
        <<<numMergeTrees, cooperativeBlockSize, 0, stream>>>(numFingerprints,
                                                             partitionSize,
                                                             entryStride,
                                                             numPartitions,
                                                             mergeFanIn,
                                                             partitionOffset,
                                                             buildBatchSize,
                                                             partitionOffset == 0,
                                                             partialStorage,
                                                             threshold,
                                                             branchingFactor,
                                                             partialToMerge.data(),
                                                             mergeNodeStride,
                                                             mergeEntryStride,
                                                             mergeStorage);
      cudaCheckError(cudaGetLastError());
      mergeSummaryCursor.get(hostMergeSummaryCursor);
      mergeStatuses.copyToHost(hostMergeStatuses);
      cudaCheckError(cudaStreamSynchronize(stream));
      for (const auto status : hostMergeStatuses) {
        if (status != BitBirchStatus::Success) {
          throw std::runtime_error("BitBIRCH intermediate merge-tree capacity or structural failure (status " +
                                   std::to_string(static_cast<int>(status)) + ")");
        }
      }
    }
    bitBirchIndexForestClustersKernel<<<numMergeTrees, 1, 0, stream>>>(numFingerprints,
                                                                       mergePartitionSize,
                                                                       mergeNodeStride,
                                                                       mergeEntryStride,
                                                                       mergeStorage);
    cudaCheckError(cudaGetLastError());
    std::vector<int>            hostMergeClusterCounts(numMergeTrees);
    mergeStatuses.copyToHost(hostMergeStatuses);
    mergeClusterCounts.copyToHost(hostMergeClusterCounts);
    cudaCheckError(cudaStreamSynchronize(stream));
    for (const auto status : hostMergeStatuses) {
      if (status != BitBirchStatus::Success) {
        throw std::runtime_error("BitBIRCH intermediate merge-tree capacity or structural failure (status " +
                                 std::to_string(static_cast<int>(status)) + ")");
      }
    }
    std::vector<int> clusterOffsets(numMergeTrees);
    for (int tree = 0; tree < numMergeTrees; ++tree) {
      clusterOffsets[tree] = totalMergeClusters;
      totalMergeClusters += hostMergeClusterCounts[tree];
    }
    constexpr int sparseForestNumerator   = 1;
    constexpr int sparseForestDenominator = 2;
    const bool    sparseForest =
      totalMergeClusters > std::numeric_limits<std::uint16_t>::max() ||
      static_cast<std::int64_t>(totalMergeClusters) * sparseForestDenominator >=
        static_cast<std::int64_t>(numFingerprints) * sparseForestNumerator;
    if (sparseForest) {
      const ScopedNvtxRange  finalizeRange("BitBIRCH sparse-forest finalization");
      AsyncDeviceVector<int> deviceClusterOffsets(numMergeTrees, stream);
      deviceClusterOffsets.copyFromHost(clusterOffsets);
      bitBirchFinalizeForestKernel<<<numMergeTrees, cooperativeBlockSize, 0, stream>>>(
        numFingerprints,
        mergePartitionSize,
        mergeNodeStride,
        mergeEntryStride,
        deviceClusterOffsets.data(),
        returnCentroids ? result.centroids.data() : nullptr,
        mergeStorage);
      cudaCheckError(cudaGetLastError());
      cudaCheckError(cudaStreamSynchronize(stream));
      result.numClusters = totalMergeClusters;
      return result;
    }
  }

  const int                        finalMaxNodes   = 2 * numFingerprints + 8;
  const int                        finalMaxEntries = 3 * numFingerprints + 8;
  AsyncDeviceVector<int>           finalNodeHeads(finalMaxNodes, stream);
  AsyncDeviceVector<int>           finalNodeSizes(finalMaxNodes, stream);
  AsyncDeviceVector<int>           finalNodeParents(finalMaxNodes, stream);
  AsyncDeviceVector<std::uint8_t>  finalNodeLeaves(finalMaxNodes, stream);
  AsyncDeviceVector<int>           finalEntryNext(finalMaxEntries, stream);
  AsyncDeviceVector<int>           finalEntryChildren(finalMaxEntries, stream);
  AsyncDeviceVector<std::uint32_t> finalEntryCounts(finalMaxEntries, stream);
  AsyncDeviceVector<int>           finalEntrySummarySlots(finalMaxEntries, stream);
  AsyncDeviceVector<int>           finalEntryFingerprintIndices(finalMaxEntries, stream);
  AsyncDeviceVector<int>           finalEntryClusterIds(finalMaxEntries, stream);
  AsyncDeviceVector<int>           partialToFinal(useIntermediateMerge ? mergeTotalEntries : totalEntries, stream);
  AsyncDeviceVector<int>           uniqueSourceEntries(numFingerprints, stream);
  AsyncDevicePtr<int>              finalRoot(0, stream);
  AsyncDevicePtr<int>              finalNodeCursor(0, stream);
  AsyncDevicePtr<int>              finalEntryCursor(0, stream);
  AsyncDevicePtr<int>              finalSummaryCursor(0, stream);
  AsyncDevicePtr<int>              finalClusterCount(0, stream);
  AsyncDevicePtr<BitBirchStatus>   finalStatus(BitBirchStatus::Success, stream);
  PagedSummaryArena<Component>     finalSummaryArena(numBits, numWords, stream);
  const int                        finalSourceClusters = useIntermediateMerge ? totalMergeClusters : totalPartialClusters;
  const auto finalSummaryCapacity = std::min<std::size_t>(
    finalMaxEntries, 2 * static_cast<std::size_t>(finalSourceClusters) + 4);
  finalSummaryArena.reserve(static_cast<int>(finalSummaryCapacity));
  TreeStorage<Component>           finalStorage{fingerprints.data(),
                                      finalNodeHeads.data(),
                                      finalNodeSizes.data(),
                                      finalNodeParents.data(),
                                      finalNodeLeaves.data(),
                                      finalEntryNext.data(),
                                      finalEntryChildren.data(),
                                      finalEntryCounts.data(),
                                      finalEntrySummarySlots.data(),
                                      finalEntryFingerprintIndices.data(),
                                      finalSummaryArena.linearSumPages(),
                                      finalSummaryArena.centroidPages(),
                                      finalEntryClusterIds.data(),
                                      result.clusterIds.data(),
                                      returnCentroids ? result.centroids.data() : nullptr,
                                      finalRoot.data(),
                                      finalNodeCursor.data(),
                                      finalEntryCursor.data(),
                                      finalSummaryCursor.data(),
                                      finalClusterCount.data(),
                                      finalStatus.data(),
                                      finalMaxNodes,
                                      finalMaxEntries,
                                      finalSummaryArena.capacity(),
                                      numWords,
                                      numBits};
  BitBirchStatus                   hostFinalStatus{};
  {
    const ScopedNvtxRange mergeRange(useIntermediateMerge ? "BitBIRCH merge round 2" : "BitBIRCH merge round 1");
    if (useIntermediateMerge) {
      bitBirchMergePartialTreesKernel<Component, MergeComponent>
        <<<1, cooperativeBlockSize, 0, stream>>>(numFingerprints,
                                                 mergePartitionSize,
                                                 mergeEntryStride,
                                                 static_cast<int>(mergeTotalEntries),
                                                 mergeStorage,
                                                 threshold,
                                                 branchingFactor,
                                                 partialToFinal.data(),
                                                 uniqueSourceEntries.data(),
                                                 finalStorage);
    } else {
      bitBirchMergePartialTreesKernel<Component, PartialComponent>
        <<<1, cooperativeBlockSize, 0, stream>>>(numFingerprints,
                                                 partitionSize,
                                                 entryStride,
                                                 static_cast<int>(totalEntries),
                                                 partialStorage,
                                                 threshold,
                                                 branchingFactor,
                                                 partialToFinal.data(),
                                                 uniqueSourceEntries.data(),
                                                 finalStorage);
    }
    cudaCheckError(cudaGetLastError());
    finalStatus.get(hostFinalStatus);
    finalClusterCount.get(result.numClusters);
    cudaCheckError(cudaStreamSynchronize(stream));
  }
  if (hostFinalStatus != BitBirchStatus::Success) {
    throw std::runtime_error("BitBIRCH merge-tree capacity or structural failure (status " +
                             std::to_string(static_cast<int>(hostFinalStatus)) + ")");
  }
  return result;
}

template <typename PartialComponent, typename Component>
BitBirchResult launchPartitionedForMerge(const cuda::std::span<const std::uint32_t> fingerprints,
                                         const int                                  numFingerprints,
                                         const int                                  numWords,
                                         const double                               threshold,
                                         const int                                  branchingFactor,
                                         const BitBirchMergeCriterion               mergeCriterion,
                                         const double                               tolerance,
                                         const int                                  numPartitions,
                                         const bool                                 returnCentroids,
                                         const cudaStream_t                         stream) {
  const int partitionSize = (numFingerprints + numPartitions - 1) / numPartitions;
  const int mergeFanIn     = numPartitions > 128 ? 4 : 2;
  const int maxMergeCount  = std::min(numFingerprints, partitionSize * mergeFanIn);
  if constexpr (sizeof(Component) == sizeof(std::uint8_t)) {
    return launchPartitioned<PartialComponent, std::uint8_t, Component>(fingerprints,
                                                                        numFingerprints,
                                                                        numWords,
                                                                        threshold,
                                                                        branchingFactor,
                                                                        mergeCriterion,
                                                                        tolerance,
                                                                        numPartitions,
                                                                        returnCentroids,
                                                                        stream);
  }
  if (maxMergeCount <= std::numeric_limits<std::uint8_t>::max()) {
    return launchPartitioned<PartialComponent, std::uint8_t, Component>(fingerprints,
                                                                        numFingerprints,
                                                                        numWords,
                                                                        threshold,
                                                                        branchingFactor,
                                                                        mergeCriterion,
                                                                        tolerance,
                                                                        numPartitions,
                                                                        returnCentroids,
                                                                        stream);
  }
  if (maxMergeCount <= std::numeric_limits<std::uint16_t>::max()) {
    return launchPartitioned<PartialComponent, std::uint16_t, Component>(fingerprints,
                                                                         numFingerprints,
                                                                         numWords,
                                                                         threshold,
                                                                         branchingFactor,
                                                                         mergeCriterion,
                                                                         tolerance,
                                                                         numPartitions,
                                                                         returnCentroids,
                                                                         stream);
  }
  if constexpr (sizeof(Component) == sizeof(std::uint16_t)) {
    throw std::logic_error("BitBIRCH merge component dispatch exceeded the final component width");
  } else {
    return launchPartitioned<PartialComponent, std::uint32_t, Component>(fingerprints,
                                                                         numFingerprints,
                                                                         numWords,
                                                                         threshold,
                                                                         branchingFactor,
                                                                         mergeCriterion,
                                                                         tolerance,
                                                                         numPartitions,
                                                                         returnCentroids,
                                                                         stream);
  }
}

template <typename Component>
BitBirchResult launchPartitionedForFinal(const cuda::std::span<const std::uint32_t> fingerprints,
                                         const int                                  numFingerprints,
                                         const int                                  numWords,
                                         const double                               threshold,
                                         const int                                  branchingFactor,
                                         const BitBirchMergeCriterion               mergeCriterion,
                                         const double                               tolerance,
                                         const int                                  numPartitions,
                                         const bool                                 returnCentroids,
                                         const cudaStream_t                         stream) {
  const int partitionSize = (numFingerprints + numPartitions - 1) / numPartitions;
  if constexpr (sizeof(Component) == sizeof(std::uint8_t)) {
    return launchPartitionedForMerge<std::uint8_t, Component>(fingerprints,
                                                               numFingerprints,
                                                               numWords,
                                                               threshold,
                                                               branchingFactor,
                                                               mergeCriterion,
                                                               tolerance,
                                                               numPartitions,
                                                               returnCentroids,
                                                               stream);
  } else {
    if (partitionSize <= std::numeric_limits<std::uint8_t>::max()) {
      return launchPartitionedForMerge<std::uint8_t, Component>(fingerprints,
                                                                 numFingerprints,
                                                                 numWords,
                                                                 threshold,
                                                                 branchingFactor,
                                                                 mergeCriterion,
                                                                 tolerance,
                                                                 numPartitions,
                                                                 returnCentroids,
                                                                 stream);
    }
    if constexpr (sizeof(Component) == sizeof(std::uint16_t)) {
      return launchPartitionedForMerge<std::uint16_t, Component>(fingerprints,
                                                                  numFingerprints,
                                                                  numWords,
                                                                  threshold,
                                                                  branchingFactor,
                                                                  mergeCriterion,
                                                                  tolerance,
                                                                  numPartitions,
                                                                  returnCentroids,
                                                                  stream);
    } else {
      if (partitionSize <= std::numeric_limits<std::uint16_t>::max()) {
        return launchPartitionedForMerge<std::uint16_t, Component>(fingerprints,
                                                                    numFingerprints,
                                                                    numWords,
                                                                    threshold,
                                                                    branchingFactor,
                                                                    mergeCriterion,
                                                                    tolerance,
                                                                    numPartitions,
                                                                    returnCentroids,
                                                                    stream);
      }
      return launchPartitionedForMerge<std::uint32_t, Component>(fingerprints,
                                                                  numFingerprints,
                                                                  numWords,
                                                                  threshold,
                                                                  branchingFactor,
                                                                  mergeCriterion,
                                                                  tolerance,
                                                                  numPartitions,
                                                                  returnCentroids,
                                                                  stream);
    }
  }
}

}  // namespace

BitBirchResult bitBirchSerialGpu(const cuda::std::span<const std::uint32_t> fingerprints,
                                 const int                                  numFingerprints,
                                 const int                                  numWords,
                                 const double                               threshold,
                                 const int                                  branchingFactor,
                                 const BitBirchMergeCriterion               mergeCriterion,
                                 const double                               tolerance,
                                 const bool                                 returnCentroids,
                                 const cudaStream_t                         stream) {
  const ScopedNvtxRange range("BitBIRCH serial tree");
  if (numFingerprints < 0 || numWords <= 0 ||
      fingerprints.size() != static_cast<std::size_t>(numFingerprints) * numWords) {
    throw std::invalid_argument("BitBIRCH fingerprints shape is invalid");
  }
  if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
    throw std::invalid_argument("BitBIRCH threshold must be in [0, 1]");
  }
  if (branchingFactor < 3) {
    throw std::invalid_argument("BitBIRCH branching factor must be at least 3");
  }
  if (mergeCriterion != BitBirchMergeCriterion::Diameter &&
      mergeCriterion != BitBirchMergeCriterion::ToleranceDiameter) {
    throw std::invalid_argument("BitBIRCH merge criterion is invalid");
  }
  if (numFingerprints > (std::numeric_limits<int>::max() - 8) / 3 || numWords > std::numeric_limits<int>::max() / 32) {
    throw std::invalid_argument("BitBIRCH input dimensions exceed the supported index range");
  }
  if (!std::isfinite(tolerance) || tolerance < 0.0) {
    throw std::invalid_argument("BitBIRCH tolerance must be nonnegative");
  }
  if (numFingerprints == 0) {
    return {AsyncDeviceVector<int>(0, stream), AsyncDeviceVector<std::uint32_t>(0, stream), 0, numWords};
  }
  if (numFingerprints <= std::numeric_limits<std::uint8_t>::max()) {
    return launchSerial<std::uint8_t>(fingerprints,
                                      numFingerprints,
                                      numWords,
                                      threshold,
                                      branchingFactor,
                                      mergeCriterion,
                                      tolerance,
                                      returnCentroids,
                                      stream);
  }
  if (numFingerprints <= std::numeric_limits<std::uint16_t>::max()) {
    return launchSerial<std::uint16_t>(fingerprints,
                                       numFingerprints,
                                       numWords,
                                       threshold,
                                       branchingFactor,
                                       mergeCriterion,
                                       tolerance,
                                       returnCentroids,
                                       stream);
  }
  return launchSerial<std::uint32_t>(fingerprints,
                                     numFingerprints,
                                     numWords,
                                     threshold,
                                     branchingFactor,
                                     mergeCriterion,
                                     tolerance,
                                     returnCentroids,
                                     stream);
}

BitBirchResult bitBirchGpu(const cuda::std::span<const std::uint32_t> fingerprints,
                           const int                                  numFingerprints,
                           const int                                  numWords,
                           const double                               threshold,
                           const int                                  branchingFactor,
                           const BitBirchMergeCriterion               mergeCriterion,
                           const double                               tolerance,
                           const int                                  numPartitions,
                           const bool                                 returnCentroids,
                           const cudaStream_t                         stream) {
  if (numPartitions < 1 || (numFingerprints > 0 && numPartitions > numFingerprints)) {
    throw std::invalid_argument("BitBIRCH numPartitions must be between 1 and the number of fingerprints");
  }
  if (numPartitions == 1 || numFingerprints == 0) {
    return bitBirchSerialGpu(fingerprints,
                             numFingerprints,
                             numWords,
                             threshold,
                             branchingFactor,
                             mergeCriterion,
                             tolerance,
                             returnCentroids,
                             stream);
  }
  if (mergeCriterion == BitBirchMergeCriterion::ToleranceDiameter) {
    throw std::invalid_argument("BitBIRCH tolerance-diameter merging currently requires numPartitions=1");
  }
  if (mergeCriterion != BitBirchMergeCriterion::Diameter) {
    throw std::invalid_argument("BitBIRCH merge criterion is invalid");
  }
  if (numFingerprints > (std::numeric_limits<int>::max() - 8) / 3 || numWords <= 0 ||
      numWords > std::numeric_limits<int>::max() / 32 ||
      fingerprints.size() != static_cast<std::size_t>(numFingerprints) * numWords) {
    throw std::invalid_argument("BitBIRCH fingerprints shape or dimensions are invalid");
  }
  if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0 || branchingFactor < 3 ||
      !std::isfinite(tolerance) || tolerance < 0.0) {
    throw std::invalid_argument("BitBIRCH clustering options are invalid");
  }
  if (numFingerprints <= std::numeric_limits<std::uint8_t>::max()) {
    return launchPartitionedForFinal<std::uint8_t>(fingerprints,
                                                   numFingerprints,
                                                   numWords,
                                                   threshold,
                                                   branchingFactor,
                                                   mergeCriterion,
                                                   tolerance,
                                                   numPartitions,
                                                   returnCentroids,
                                                   stream);
  }
  if (numFingerprints <= std::numeric_limits<std::uint16_t>::max()) {
    return launchPartitionedForFinal<std::uint16_t>(fingerprints,
                                                    numFingerprints,
                                                    numWords,
                                                    threshold,
                                                    branchingFactor,
                                                    mergeCriterion,
                                                    tolerance,
                                                    numPartitions,
                                                    returnCentroids,
                                                    stream);
  }
  return launchPartitionedForFinal<std::uint32_t>(fingerprints,
                                                  numFingerprints,
                                                  numWords,
                                                  threshold,
                                                  branchingFactor,
                                                  mergeCriterion,
                                                  tolerance,
                                                  numPartitions,
                                                  returnCentroids,
                                                  stream);
}

}  // namespace nvMolKit
