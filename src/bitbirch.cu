// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

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
  InvalidTree
};

template <typename Component> struct TreeStorage {
  int*            nodeHeads;
  int*            nodeSizes;
  int*            nodeParents;
  std::uint8_t*   nodeLeaves;
  int*            entryNext;
  int*            entryChildren;
  std::uint32_t*  entryCounts;
  Component*      entryLinearSums;
  int*            entryClusterIds;
  int*            labels;
  std::uint32_t*  centroids;
  int*            root;
  int*            nodeCursor;
  int*            entryCursor;
  int*            numClusters;
  BitBirchStatus* status;
  int             maxNodes;
  int             maxEntries;
  int             numWords;
  int             numBits;
};

template <typename Component>
__device__ __forceinline__ Component& linearSum(TreeStorage<Component>& storage, const int entry, const int bit) {
  return storage.entryLinearSums[static_cast<std::size_t>(entry) * storage.numBits + bit];
}

template <typename Component>
__device__ __forceinline__ Component linearSum(const TreeStorage<Component>& storage, const int entry, const int bit) {
  return storage.entryLinearSums[static_cast<std::size_t>(entry) * storage.numBits + bit];
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
  return entry;
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
  std::uint32_t result = 0;
  const auto    count  = static_cast<std::uint64_t>(storage.entryCounts[entry]);
  for (int bitInWord = 0; bitInWord < 32; ++bitInWord) {
    const int bit = word * 32 + bitInWord;
    if (bitbirch::majorityCentroidBit(static_cast<std::uint64_t>(linearSum(storage, entry, bit)), count)) {
      result |= std::uint32_t{1} << bitInWord;
    }
  }
  return result;
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
                                                    const std::uint32_t*    fingerprint) {
  storage.entryCounts[entry] = 1;
  for (int bit = 0; bit < storage.numBits; ++bit) {
    linearSum(storage, entry, bit) = static_cast<Component>((fingerprint[bit / 32] >> (bit % 32)) & 1U);
  }
}

template <typename Component>
__device__ __forceinline__ void addFingerprint(TreeStorage<Component>& storage,
                                               const int               entry,
                                               const std::uint32_t*    fingerprint) {
  ++storage.entryCounts[entry];
  for (int bit = 0; bit < storage.numBits; ++bit) {
    linearSum(storage, entry, bit) += static_cast<Component>((fingerprint[bit / 32] >> (bit % 32)) & 1U);
  }
}

template <typename Component>
__device__ __forceinline__ void summarizeNode(TreeStorage<Component>& storage, const int node, const int targetEntry) {
  std::uint32_t count = 0;
  for (int bit = 0; bit < storage.numBits; ++bit) {
    linearSum(storage, targetEntry, bit) = 0;
  }
  for (int entry = storage.nodeHeads[node]; entry >= 0; entry = storage.entryNext[entry]) {
    count += storage.entryCounts[entry];
    for (int bit = 0; bit < storage.numBits; ++bit) {
      linearSum(storage, targetEntry, bit) += linearSum(storage, entry, bit);
    }
  }
  storage.entryCounts[targetEntry] = count;
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
        bool         assignLeft =
          lhsSimilarity > rhsSimilarity || (lhsSimilarity == rhsSimilarity && lhsAssigned <= rhsAssigned);
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
    node = parent;
  }
  return node;
}

template <typename Component>
__device__ bool buildFingerprintRange(const std::uint32_t*         fingerprints,
                                      const int                    begin,
                                      const int                    end,
                                      const double                 threshold,
                                      const int                    branchingFactor,
                                      const BitBirchMergeCriterion mergeCriterion,
                                      const double                 tolerance,
                                      TreeStorage<Component>&      storage) {
  *storage.status       = BitBirchStatus::Success;
  *storage.nodeCursor   = 0;
  *storage.entryCursor  = 0;
  *storage.numClusters  = 0;
  const int initialRoot = allocateNode(storage, true, -1);
  *storage.root         = initialRoot;

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
      initializeLeafEntry(storage, selectedEntry, fingerprint);
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
                                     const int                    numFingerprints,
                                     const double                 threshold,
                                     const int                    branchingFactor,
                                     const BitBirchMergeCriterion mergeCriterion,
                                     const double                 tolerance,
                                     TreeStorage<Component>       storage) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  if (buildFingerprintRange(fingerprints,
                            0,
                            numFingerprints,
                            threshold,
                            branchingFactor,
                            mergeCriterion,
                            tolerance,
                            storage)) {
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
  storage.entryLinearSums += static_cast<std::size_t>(partition) * entryStride * storage.numBits;
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
                                           const int                    nodeStride,
                                           const int                    entryStride,
                                           const double                 threshold,
                                           const int                    branchingFactor,
                                           const BitBirchMergeCriterion mergeCriterion,
                                           const double                 tolerance,
                                           TreeStorage<Component>       storage) {
  if (threadIdx.x != 0) {
    return;
  }
  const int partition = blockIdx.x;
  const int begin     = partition * partitionSize;
  const int end       = min(begin + partitionSize, numFingerprints);
  auto      local     = partitionStorage(storage, partition, nodeStride, entryStride);
  buildFingerprintRange(fingerprints, begin, end, threshold, branchingFactor, mergeCriterion, tolerance, local);
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
                                                           const SummaryComponent*       candidateSums,
                                                           const std::uint32_t           candidateCount) {
  int intersection = 0;
  int unionCount   = 0;
  for (int word = 0; word < storage.numWords; ++word) {
    const std::uint32_t lhs = centroidWord(storage, entry, word);
    const std::uint32_t rhs = summaryCentroidWord(candidateSums, candidateCount, word);
    intersection += __popc(lhs & rhs);
    unionCount += __popc(lhs | rhs);
  }
  return unionCount > 0 ? static_cast<double>(intersection) / unionCount : 1.0;
}

template <typename Component, typename SummaryComponent>
__device__ __forceinline__ int closestSummaryEntry(const TreeStorage<Component>& storage,
                                                   const int                     node,
                                                   const SummaryComponent*       candidateSums,
                                                   const std::uint32_t           candidateCount) {
  int    best           = storage.nodeHeads[node];
  double bestSimilarity = -1.0;
  for (int entry = storage.nodeHeads[node]; entry >= 0; entry = storage.entryNext[entry]) {
    const double similarity = entryToSummarySimilarity(storage, entry, candidateSums, candidateCount);
    if (similarity > bestSimilarity) {
      bestSimilarity = similarity;
      best           = entry;
    }
  }
  return best;
}

template <typename Component, typename SummaryComponent>
__device__ __forceinline__ bitbirch::ISimTanimotoTerms combinedSummaryISimTerms(const TreeStorage<Component>& storage,
                                                                                const int                     entry,
                                                                                const SummaryComponent* candidateSums,
                                                                                const std::uint32_t candidateCount) {
  bitbirch::ISimTanimotoTerms terms{};
  const auto                  combinedCount = static_cast<std::uint64_t>(storage.entryCounts[entry]) + candidateCount;
  for (int bit = 0; bit < storage.numBits; ++bit) {
    const auto component =
      static_cast<std::uint64_t>(linearSum(storage, entry, bit)) + static_cast<std::uint64_t>(candidateSums[bit]);
    bitbirch::accumulateISimTanimotoTerm(terms, component, combinedCount);
  }
  return terms;
}

template <typename Component, typename SummaryComponent>
__device__ int insertSummary(TreeStorage<Component>&      storage,
                             const SummaryComponent*      candidateSums,
                             const std::uint32_t          candidateCount,
                             const double                 threshold,
                             const int                    branchingFactor,
                             const BitBirchMergeCriterion mergeCriterion,
                             const double                 tolerance) {
  int node = *storage.root;
  while (!storage.nodeLeaves[node]) {
    const int entry = closestSummaryEntry(storage, node, candidateSums, candidateCount);
    node            = storage.entryChildren[entry];
    if (node < 0) {
      *storage.status = BitBirchStatus::InvalidTree;
      return -1;
    }
  }

  int selectedEntry =
    storage.nodeHeads[node] >= 0 ? closestSummaryEntry(storage, node, candidateSums, candidateCount) : -1;
  bool merge = false;
  if (selectedEntry >= 0) {
    const auto combinedTerms = combinedSummaryISimTerms(storage, selectedEntry, candidateSums, candidateCount);
    const auto combinedCount = static_cast<std::uint64_t>(storage.entryCounts[selectedEntry]) + candidateCount;
    merge                    = bitbirch::isimTanimotoAtLeast(combinedTerms, combinedCount, threshold);
    if (merge && mergeCriterion == BitBirchMergeCriterion::ToleranceDiameter) {
      const auto   oldCount     = static_cast<std::uint64_t>(storage.entryCounts[selectedEntry]);
      const double oldISim      = bitbirch::isimTanimoto(entryISimTerms(storage, selectedEntry), oldCount);
      const double combinedISim = bitbirch::isimTanimoto(combinedTerms, combinedCount);
      merge                     = bitbirch::singletonToleranceAllows(oldISim, combinedISim, oldCount, tolerance);
    }
  }

  if (merge) {
    storage.entryCounts[selectedEntry] += candidateCount;
    for (int bit = 0; bit < storage.numBits; ++bit) {
      linearSum(storage, selectedEntry, bit) += static_cast<Component>(candidateSums[bit]);
    }
  } else {
    selectedEntry = allocateEntry(storage);
    if (selectedEntry < 0) {
      return -1;
    }
    storage.entryCounts[selectedEntry] = candidateCount;
    for (int bit = 0; bit < storage.numBits; ++bit) {
      linearSum(storage, selectedEntry, bit) = static_cast<Component>(candidateSums[bit]);
    }
    appendEntry(storage, node, selectedEntry);
  }
  if (!refreshAncestors(storage, node)) {
    return -1;
  }
  if (storage.nodeSizes[node] > branchingFactor) {
    const int changedNode = splitNode(storage, node, branchingFactor);
    if (changedNode < 0 || !refreshAncestors(storage, changedNode)) {
      return -1;
    }
  }
  return selectedEntry;
}

template <typename Component, typename PartialComponent>
__global__ void bitBirchMergePartialTreesKernel(const int                    numFingerprints,
                                                const int                    partitionSize,
                                                const int                    partialEntryStride,
                                                const int                    totalPartialEntries,
                                                const std::uint32_t*         partialCounts,
                                                const PartialComponent*      partialLinearSums,
                                                const double                 threshold,
                                                const int                    branchingFactor,
                                                const BitBirchMergeCriterion mergeCriterion,
                                                const double                 tolerance,
                                                int*                         partialToFinal,
                                                TreeStorage<Component>       storage) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  *storage.status       = BitBirchStatus::Success;
  *storage.nodeCursor   = 0;
  *storage.entryCursor  = 0;
  *storage.numClusters  = 0;
  const int initialRoot = allocateNode(storage, true, -1);
  *storage.root         = initialRoot;
  for (int entry = 0; entry < totalPartialEntries; ++entry) {
    partialToFinal[entry] = -1;
  }

  for (int fingerprintIndex = 0; fingerprintIndex < numFingerprints; ++fingerprintIndex) {
    const int partition   = fingerprintIndex / partitionSize;
    const int localEntry  = storage.labels[fingerprintIndex];
    const int sourceEntry = partition * partialEntryStride + localEntry;
    int       finalEntry  = partialToFinal[sourceEntry];
    if (finalEntry < 0) {
      const PartialComponent* candidateSums =
        partialLinearSums + static_cast<std::size_t>(sourceEntry) * storage.numBits;
      finalEntry = insertSummary(storage,
                                 candidateSums,
                                 partialCounts[sourceEntry],
                                 threshold,
                                 branchingFactor,
                                 mergeCriterion,
                                 tolerance);
      if (finalEntry < 0) {
        return;
      }
      partialToFinal[sourceEntry] = finalEntry;
    }
    storage.labels[fingerprintIndex] = finalEntry;
  }
  compactLabels(0, numFingerprints, storage);
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
  AsyncDeviceVector<Component>     entryLinearSums(static_cast<std::size_t>(maxEntries) * numBits, stream);
  AsyncDeviceVector<int>           entryClusterIds(maxEntries, stream);
  AsyncDevicePtr<int>              root(0, stream);
  AsyncDevicePtr<int>              nodeCursor(0, stream);
  AsyncDevicePtr<int>              entryCursor(0, stream);
  AsyncDevicePtr<int>              numClusters(0, stream);
  AsyncDevicePtr<BitBirchStatus>   status(BitBirchStatus::Success, stream);

  BitBirchResult result{
    AsyncDeviceVector<int>(numFingerprints, stream),
    AsyncDeviceVector<std::uint32_t>(returnCentroids ? static_cast<std::size_t>(numFingerprints) * numWords : 0,
                                     stream),
    0,
    numWords};
  TreeStorage<Component> storage{nodeHeads.data(),
                                 nodeSizes.data(),
                                 nodeParents.data(),
                                 nodeLeaves.data(),
                                 entryNext.data(),
                                 entryChildren.data(),
                                 entryCounts.data(),
                                 entryLinearSums.data(),
                                 entryClusterIds.data(),
                                 result.clusterIds.data(),
                                 returnCentroids ? result.centroids.data() : nullptr,
                                 root.data(),
                                 nodeCursor.data(),
                                 entryCursor.data(),
                                 numClusters.data(),
                                 status.data(),
                                 maxNodes,
                                 maxEntries,
                                 numWords,
                                 numBits};
  bitBirchSerialKernel<<<1, 1, 0, stream>>>(fingerprints.data(),
                                            numFingerprints,
                                            threshold,
                                            branchingFactor,
                                            mergeCriterion,
                                            tolerance,
                                            storage);
  cudaCheckError(cudaGetLastError());

  BitBirchStatus hostStatus{};
  status.get(hostStatus);
  numClusters.get(result.numClusters);
  cudaCheckError(cudaStreamSynchronize(stream));
  if (hostStatus != BitBirchStatus::Success) {
    throw std::runtime_error("BitBIRCH serial tree capacity or structural failure (status " +
                             std::to_string(static_cast<int>(hostStatus)) + ")");
  }
  return result;
}

template <typename PartialComponent, typename Component>
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
  AsyncDeviceVector<PartialComponent> partialEntryLinearSums(totalEntries * numBits, stream);
  AsyncDeviceVector<int>              partialRoots(numPartitions, stream);
  AsyncDeviceVector<int>              partialNodeCursors(numPartitions, stream);
  AsyncDeviceVector<int>              partialEntryCursors(numPartitions, stream);
  AsyncDeviceVector<int>              partialClusterCounts(numPartitions, stream);
  AsyncDeviceVector<BitBirchStatus>   partialStatuses(numPartitions, stream);

  BitBirchResult result{
    AsyncDeviceVector<int>(numFingerprints, stream),
    AsyncDeviceVector<std::uint32_t>(returnCentroids ? static_cast<std::size_t>(numFingerprints) * numWords : 0,
                                     stream),
    0,
    numWords};
  TreeStorage<PartialComponent> partialStorage{partialNodeHeads.data(),
                                               partialNodeSizes.data(),
                                               partialNodeParents.data(),
                                               partialNodeLeaves.data(),
                                               partialEntryNext.data(),
                                               partialEntryChildren.data(),
                                               partialEntryCounts.data(),
                                               partialEntryLinearSums.data(),
                                               nullptr,
                                               result.clusterIds.data(),
                                               nullptr,
                                               partialRoots.data(),
                                               partialNodeCursors.data(),
                                               partialEntryCursors.data(),
                                               partialClusterCounts.data(),
                                               partialStatuses.data(),
                                               nodeStride,
                                               entryStride,
                                               numWords,
                                               numBits};
  {
    const ScopedNvtxRange partialRange("BitBIRCH partial-tree construction");
    bitBirchPartialTreesKernel<<<numPartitions, 1, 0, stream>>>(fingerprints.data(),
                                                                numFingerprints,
                                                                partitionSize,
                                                                nodeStride,
                                                                entryStride,
                                                                threshold,
                                                                branchingFactor,
                                                                mergeCriterion,
                                                                tolerance,
                                                                partialStorage);
    cudaCheckError(cudaGetLastError());
    std::vector<BitBirchStatus> hostPartialStatuses(numPartitions);
    partialStatuses.copyToHost(hostPartialStatuses);
    cudaCheckError(cudaStreamSynchronize(stream));
    for (const auto status : hostPartialStatuses) {
      if (status != BitBirchStatus::Success) {
        throw std::runtime_error("BitBIRCH partial-tree capacity or structural failure (status " +
                                 std::to_string(static_cast<int>(status)) + ")");
      }
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
  AsyncDeviceVector<Component>     finalEntryLinearSums(static_cast<std::size_t>(finalMaxEntries) * numBits, stream);
  AsyncDeviceVector<int>           finalEntryClusterIds(finalMaxEntries, stream);
  AsyncDeviceVector<int>           partialToFinal(totalEntries, stream);
  AsyncDevicePtr<int>              finalRoot(0, stream);
  AsyncDevicePtr<int>              finalNodeCursor(0, stream);
  AsyncDevicePtr<int>              finalEntryCursor(0, stream);
  AsyncDevicePtr<int>              finalClusterCount(0, stream);
  AsyncDevicePtr<BitBirchStatus>   finalStatus(BitBirchStatus::Success, stream);
  TreeStorage<Component>           finalStorage{finalNodeHeads.data(),
                                      finalNodeSizes.data(),
                                      finalNodeParents.data(),
                                      finalNodeLeaves.data(),
                                      finalEntryNext.data(),
                                      finalEntryChildren.data(),
                                      finalEntryCounts.data(),
                                      finalEntryLinearSums.data(),
                                      finalEntryClusterIds.data(),
                                      result.clusterIds.data(),
                                      returnCentroids ? result.centroids.data() : nullptr,
                                      finalRoot.data(),
                                      finalNodeCursor.data(),
                                      finalEntryCursor.data(),
                                      finalClusterCount.data(),
                                      finalStatus.data(),
                                      finalMaxNodes,
                                      finalMaxEntries,
                                      numWords,
                                      numBits};
  BitBirchStatus                   hostFinalStatus{};
  {
    const ScopedNvtxRange mergeRange("BitBIRCH merge round 1");
    bitBirchMergePartialTreesKernel<Component, PartialComponent><<<1, 1, 0, stream>>>(numFingerprints,
                                                                                      partitionSize,
                                                                                      entryStride,
                                                                                      static_cast<int>(totalEntries),
                                                                                      partialEntryCounts.data(),
                                                                                      partialEntryLinearSums.data(),
                                                                                      threshold,
                                                                                      branchingFactor,
                                                                                      mergeCriterion,
                                                                                      tolerance,
                                                                                      partialToFinal.data(),
                                                                                      finalStorage);
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
    return launchPartitioned<std::uint8_t, Component>(fingerprints,
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
      return launchPartitioned<std::uint8_t, Component>(fingerprints,
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
      return launchPartitioned<std::uint16_t, Component>(fingerprints,
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
        return launchPartitioned<std::uint16_t, Component>(fingerprints,
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
      return launchPartitioned<std::uint32_t, Component>(fingerprints,
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
