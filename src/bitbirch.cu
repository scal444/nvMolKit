// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/bitbirch.h"
#include "src/bitbirch_common.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/host_vector.h"
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

constexpr int summaryEntriesPerPage     = 4096;
constexpr int summaryBuildBatchSize     = 32;
// Preserve block-level parallelism across hierarchy rounds. A smaller tail is
// worthwhile only when the remaining summaries fit a bounded global merge.
constexpr int minimumParallelMergeTrees = 32;
constexpr int serialTailSummaryBudget   = 4096;

bool hierarchyRoundFitsWorkBudget(const int numTrees, const int totalClusters, const int mergeFanIn) {
  const int nextNumTrees = (numTrees + mergeFanIn - 1) / mergeFanIn;
  return nextNumTrees >= minimumParallelMergeTrees || totalClusters <= serialTailSummaryBudget;
}

template <typename Component> class PagedSummaryArena {
 public:
  PagedSummaryArena(const int numBits, const int numWords, const cudaStream_t stream, const std::size_t cacheBytes = 0)
      : numBits_(numBits),
        numWords_(numWords),
        stream_(stream),
        cachePages_(cacheBytes / (sizeof(Component) * static_cast<std::size_t>(summaryEntriesPerPage) * numBits)) {
    if (cacheBytes > 0 && cachePages_ == 0) {
      throw std::invalid_argument("summary_cache_bytes must fit at least one BF-sum page");
    }
    linearSumPagePointers_.setStream(stream);
    centroidPagePointers_.setStream(stream);
    pageHits_.setStream(stream);
  }

  ~PagedSummaryArena() {
    // CPU backing must outlive every kernel/transfer, including exception paths.
    if (cachePages_ > 0) {
      cudaStreamSynchronize(stream_);
    }
  }

  void reserve(const int entries) {
    const int requiredPages =
      static_cast<int>((static_cast<std::int64_t>(entries) + summaryEntriesPerPage - 1) / summaryEntriesPerPage);
    const auto oldPages = centroidPages_.size();
    if (requiredPages <= static_cast<int>(oldPages)) {
      return;
    }
    while (static_cast<int>(centroidPages_.size()) < requiredPages) {
      if (cachePages_ == 0) {
        linearSumPages_.emplace_back(pageElements(), stream_);
      } else {
        hostSumPages_.emplace_back(pageElements(), Component{0});
        Component* mapped = nullptr;
        cudaCheckError(cudaHostGetDevicePointer(&mapped, hostSumPages_.back().data(), 0));
        mappedHostPages_.push_back(mapped);
        pageToCache_.push_back(-1);
        pageScores_.push_back(0);
        if (linearSumPages_.size() < cachePages_) {
          const int slot = static_cast<int>(linearSumPages_.size());
          const int page = static_cast<int>(centroidPages_.size());
          linearSumPages_.emplace_back(pageElements(), stream_);
          pageToCache_[page] = slot;
          cacheToPage_.push_back(page);
          cudaCheckError(cudaMemcpyAsync(linearSumPages_.back().data(),
                                         hostSumPages_.back().data(),
                                         pageBytes(),
                                         cudaMemcpyHostToDevice,
                                         stream_));
        }
      }
      centroidPages_.emplace_back(static_cast<std::size_t>(summaryEntriesPerPage) * numWords_, stream_);
    }
    if (cachePages_ > 0) {
      pageHits_.resize(requiredPages);
      cudaCheckError(
        cudaMemsetAsync(pageHits_.data() + oldPages, 0, (requiredPages - oldPages) * sizeof(std::uint32_t), stream_));
    }
    updatePointers();
  }

  // Called only after all writers in the logical batch have completed. CPU
  // backing is stale for cached pages; always write back before reusing a slot.
  // Cold misses use mapped CPU backing, so cache capacity never changes routing,
  // proposal groups, service order, or the partition itself.
  void rotate() {
    if (cachePages_ == 0 || centroidPages_.size() <= cachePages_) {
      return;
    }
    std::vector<std::uint32_t> hits(centroidPages_.size());
    pageHits_.copyToHost(hits);
    cudaCheckError(cudaStreamSynchronize(stream_));
    cudaCheckError(cudaMemsetAsync(pageHits_.data(), 0, hits.size() * sizeof(std::uint32_t), stream_));
    std::vector<int> order(hits.size());
    std::iota(order.begin(), order.end(), 0);
    for (std::size_t page = 0; page < hits.size(); ++page) {
      pageScores_[page] = pageScores_[page] / 2 + hits[page];
    }
    std::stable_sort(order.begin(), order.end(), [this](const int left, const int right) {
      if (pageScores_[left] != pageScores_[right]) {
        return pageScores_[left] > pageScores_[right];
      }
      return pageToCache_[left] >= 0 && pageToCache_[right] < 0;
    });
    order.resize(linearSumPages_.size());
    std::vector<bool> retain(hits.size(), false);
    for (const int page : order) {
      retain[page] = true;
    }
    bool changed = false;
    for (const int page : order) {
      if (pageToCache_[page] >= 0) {
        continue;
      }
      int slot = 0;
      while (retain[cacheToPage_[slot]]) {
        ++slot;
      }
      const int oldPage = cacheToPage_[slot];
      cudaCheckError(cudaMemcpyAsync(hostSumPages_[oldPage].data(),
                                     linearSumPages_[slot].data(),
                                     pageBytes(),
                                     cudaMemcpyDeviceToHost,
                                     stream_));
      cudaCheckError(cudaMemcpyAsync(linearSumPages_[slot].data(),
                                     hostSumPages_[page].data(),
                                     pageBytes(),
                                     cudaMemcpyHostToDevice,
                                     stream_));
      pageToCache_[oldPage] = -1;
      pageToCache_[page]    = slot;
      cacheToPage_[slot]    = page;
      changed               = true;
    }
    if (changed) {
      updatePointers();
    }
  }

  std::uint32_t*  pageHits() const noexcept { return cachePages_ > 0 ? pageHits_.data() : nullptr; }
  Component**     linearSumPages() const noexcept { return linearSumPagePointers_.data(); }
  std::uint32_t** centroidPages() const noexcept { return centroidPagePointers_.data(); }
  int             capacity() const noexcept { return static_cast<int>(centroidPages_.size()) * summaryEntriesPerPage; }

  void clear() {
    if (cachePages_ > 0) {
      cudaCheckError(cudaStreamSynchronize(stream_));
    }
    linearSumPages_.clear();
    centroidPages_.clear();
    hostSumPages_.clear();
    mappedHostPages_.clear();
    pageToCache_.clear();
    cacheToPage_.clear();
    pageScores_.clear();
    pageHits_.resize(0);
    linearSumPagePointers_ = AsyncDeviceVector<Component*>();
    centroidPagePointers_  = AsyncDeviceVector<std::uint32_t*>();
    linearSumPagePointers_.setStream(stream_);
    centroidPagePointers_.setStream(stream_);
  }

 private:
  std::size_t pageElements() const noexcept { return static_cast<std::size_t>(summaryEntriesPerPage) * numBits_; }
  std::size_t pageBytes() const noexcept { return pageElements() * sizeof(Component); }

  void updatePointers() {
    std::vector<Component*>     linearSumPointers;
    std::vector<std::uint32_t*> centroidPointers;
    linearSumPointers.reserve(centroidPages_.size());
    centroidPointers.reserve(centroidPages_.size());
    for (std::size_t page = 0; page < centroidPages_.size(); ++page) {
      if (cachePages_ == 0) {
        linearSumPointers.push_back(linearSumPages_[page].data());
      } else {
        const int slot = pageToCache_[page];
        linearSumPointers.push_back(slot >= 0 ? linearSumPages_[slot].data() : mappedHostPages_[page]);
      }
    }
    for (auto& page : centroidPages_) {
      centroidPointers.push_back(page.data());
    }
    linearSumPagePointers_.setFromVector(linearSumPointers);
    centroidPagePointers_.setFromVector(centroidPointers);
    if (cachePages_ > 0) {
      cudaCheckError(cudaStreamSynchronize(stream_));
    }
  }

 private:
  int                                           numBits_;
  int                                           numWords_;
  cudaStream_t                                  stream_;
  std::vector<AsyncDeviceVector<Component>>     linearSumPages_;
  std::vector<AsyncDeviceVector<std::uint32_t>> centroidPages_;
  AsyncDeviceVector<Component*>                 linearSumPagePointers_;
  AsyncDeviceVector<std::uint32_t*>             centroidPagePointers_;
  std::size_t                                   cachePages_;
  std::vector<PinnedHostVector<Component>>      hostSumPages_;
  std::vector<Component*>                       mappedHostPages_;
  std::vector<int>                              pageToCache_;
  std::vector<int>                              cacheToPage_;
  std::vector<std::uint64_t>                    pageScores_;
  AsyncDeviceVector<std::uint32_t>              pageHits_;
};

// Packed singleton ownership is independent of the current input tile. It
// grows with live entry IDs instead of retaining all N input fingerprints.
class PagedFingerprintArena {
 public:
  PagedFingerprintArena(const int words, const cudaStream_t stream) : words_(words), stream_(stream) {
    pointers_.setStream(stream);
  }

  void reserve(const int entries) {
    const auto required = (static_cast<std::size_t>(entries) + summaryEntriesPerPage - 1) / summaryEntriesPerPage;
    if (required <= pages_.size()) {
      return;
    }
    while (pages_.size() < required) {
      pages_.emplace_back(static_cast<std::size_t>(summaryEntriesPerPage) * words_, stream_);
    }
    std::vector<std::uint32_t*> pointers;
    pointers.reserve(pages_.size());
    for (auto& page : pages_) {
      pointers.push_back(page.data());
    }
    pointers_.setFromVector(pointers);
    // Keep the temporary host pointer table alive until its upload completes.
    cudaCheckError(cudaStreamSynchronize(stream_));
  }

  std::uint32_t** data() const noexcept { return pointers_.data(); }

 private:
  int                                           words_;
  cudaStream_t                                  stream_;
  std::vector<AsyncDeviceVector<std::uint32_t>> pages_;
  AsyncDeviceVector<std::uint32_t*>             pointers_;
};

template <typename Component> struct TreeStorage {
  const std::uint32_t* fingerprints;
  int*                 nodeHeads;
  int*                 nodeSizes;
  int*                 nodeParents;
  std::uint8_t*        nodeLeaves;
  int*                 entryNext;
  int*                 entryChildren;
  std::uint32_t*       entryCounts;
  int*                 entrySummarySlots;
  int*                 entryFingerprintIndices;
  Component**          linearSumPages;
  std::uint32_t**      centroidPages;
  int*                 entryClusterIds;
  int*                 labels;
  std::uint32_t*       centroids;
  int*                 root;
  int*                 nodeCursor;
  int*                 entryCursor;
  int*                 summaryCursor;
  int*                 numClusters;
  BitBirchStatus*      status;
  int                  maxNodes;
  int                  maxEntries;
  int                  maxSummaries;
  int                  numWords;
  int                  numBits;
  std::uint32_t*       summaryPageHits           = nullptr;
  std::uint32_t**      singletonFingerprintPages = nullptr;
  int                  queryBegin                = 0;
};

template <typename Component> struct PartitionedForest {
  PartitionedForest(const std::uint32_t* fingerprints,
                    int*                 labels,
                    const int            numFingerprints,
                    const int            numTrees,
                    const int            partitionSize,
                    const int            numWords,
                    const cudaStream_t   stream,
                    const int            nodeCapacity      = 0,
                    const int            entryCapacity     = 0,
                    const std::size_t    summaryCacheBytes = 0)
      : numTrees(numTrees),
        partitionSize(partitionSize),
        nodeStride(nodeCapacity > 0 ? nodeCapacity : 2 * partitionSize + 8),
        entryStride(entryCapacity > 0 ? entryCapacity : 3 * partitionSize + 8),
        totalNodes(static_cast<std::size_t>(numTrees) * nodeStride),
        totalEntries(static_cast<std::size_t>(numTrees) * entryStride),
        nodeHeads(totalNodes, stream),
        nodeSizes(totalNodes, stream),
        nodeParents(totalNodes, stream),
        nodeLeaves(totalNodes, stream),
        entryNext(totalEntries, stream),
        entryChildren(totalEntries, stream),
        entryCounts(totalEntries, stream),
        entrySummarySlots(totalEntries, stream),
        entryFingerprintIndices(totalEntries, stream),
        entryClusterIds(totalEntries, stream),
        roots(numTrees, stream),
        nodeCursors(numTrees, stream),
        entryCursors(numTrees, stream),
        clusterCounts(numTrees, stream),
        statuses(numTrees, stream),
        summaryCursor(0, stream),
        summaryArena(numWords * 32, numWords, stream, summaryCacheBytes),
        fingerprints(fingerprints),
        labels(labels),
        numFingerprints(numFingerprints),
        numWords(numWords) {
    if (totalEntries > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("BitBIRCH merge workspace exceeds the supported index range");
    }
    summaryArena.reserve(summaryEntriesPerPage);
  }

  TreeStorage<Component> storage() {
    return {fingerprints,
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
            labels,
            nullptr,
            roots.data(),
            nodeCursors.data(),
            entryCursors.data(),
            summaryCursor.data(),
            clusterCounts.data(),
            statuses.data(),
            nodeStride,
            entryStride,
            summaryArena.capacity(),
            numWords,
            numWords * 32,
            summaryArena.pageHits()};
  }

  void growSingleTree(const int nodes, const int entries) {
    if (numTrees != 1) {
      throw std::logic_error("Only a single shared tree supports incremental metadata growth");
    }
    if (nodes > nodeStride) {
      nodeHeads.resize(nodes);
      nodeSizes.resize(nodes);
      nodeParents.resize(nodes);
      nodeLeaves.resize(nodes);
      nodeStride = nodes;
      totalNodes = nodes;
    }
    if (entries > entryStride) {
      entryNext.resize(entries);
      entryChildren.resize(entries);
      entryCounts.resize(entries);
      entrySummarySlots.resize(entries);
      entryFingerprintIndices.resize(entries);
      entryClusterIds.resize(entries);
      entryStride  = entries;
      totalEntries = entries;
    }
  }

  int                               numTrees;
  int                               partitionSize;
  int                               nodeStride;
  int                               entryStride;
  std::size_t                       totalNodes;
  std::size_t                       totalEntries;
  AsyncDeviceVector<int>            nodeHeads;
  AsyncDeviceVector<int>            nodeSizes;
  AsyncDeviceVector<int>            nodeParents;
  AsyncDeviceVector<std::uint8_t>   nodeLeaves;
  AsyncDeviceVector<int>            entryNext;
  AsyncDeviceVector<int>            entryChildren;
  AsyncDeviceVector<std::uint32_t>  entryCounts;
  AsyncDeviceVector<int>            entrySummarySlots;
  AsyncDeviceVector<int>            entryFingerprintIndices;
  AsyncDeviceVector<int>            entryClusterIds;
  AsyncDeviceVector<int>            roots;
  AsyncDeviceVector<int>            nodeCursors;
  AsyncDeviceVector<int>            entryCursors;
  AsyncDeviceVector<int>            clusterCounts;
  AsyncDeviceVector<BitBirchStatus> statuses;
  AsyncDevicePtr<int>               summaryCursor;
  PagedSummaryArena<Component>      summaryArena;
  const std::uint32_t*              fingerprints;
  int*                              labels;
  int                               numFingerprints;
  int                               numWords;
  std::vector<int>                  hostClusterCounts;
  int                               totalClusters = 0;
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
__device__ __forceinline__ const std::uint32_t* queryFingerprint(const TreeStorage<Component>& storage,
                                                                 const int                     molecule) {
  return storage.fingerprints + static_cast<std::size_t>(molecule - storage.queryBegin) * storage.numWords;
}

template <typename Component>
__device__ __forceinline__ std::uint32_t singletonWord(const TreeStorage<Component>& storage,
                                                       const int                     index,
                                                       const int                     word) {
  if (storage.singletonFingerprintPages != nullptr) {
    return storage
      .singletonFingerprintPages[index / summaryEntriesPerPage]
                                [static_cast<std::size_t>(index % summaryEntriesPerPage) * storage.numWords + word];
  }
  return storage.fingerprints[static_cast<std::size_t>(index) * storage.numWords + word];
}

template <typename Component>
__device__ __forceinline__ Component linearSum(const TreeStorage<Component>& storage, const int entry, const int bit) {
  const int fingerprintIndex = storage.entryFingerprintIndices[entry];
  if (fingerprintIndex >= 0) {
    const std::uint32_t word = singletonWord(storage, fingerprintIndex, bit / 32);
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
  storage.entryNext[entry]               = -1;
  storage.entryChildren[entry]           = -1;
  storage.entryCounts[entry]             = 0;
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
  const int fingerprintIndex       = storage.entryFingerprintIndices[entry];
  storage.entrySummarySlots[entry] = slot;
  for (int bit = 0; bit < storage.numBits; ++bit) {
    Component value = 0;
    if (fingerprintIndex >= 0) {
      const std::uint32_t word = singletonWord(storage, fingerprintIndex, bit / 32);
      value                    = static_cast<Component>((word >> (bit % 32)) & 1U);
    }
    materializedLinearSum(storage, entry, bit) = value;
  }
  if (fingerprintIndex >= 0) {
    for (int word = 0; word < storage.numWords; ++word) {
      storage.centroidPages[slot / summaryEntriesPerPage]
                           [static_cast<std::size_t>(slot % summaryEntriesPerPage) * storage.numWords + word] =
        singletonWord(storage, fingerprintIndex, word);
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
    return singletonWord(storage, fingerprintIndex, word);
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

// This value is used only to order centroid similarities, never for diameter
// acceptance. For at most 4096 bits, distinct a/b ratios with 0 <= a <= b <= D
// are separated by at least 1/(D*(D-1)), strictly more than the widest float
// rounding cell in [0, 1]. Correctly rounded FP32 division therefore preserves
// both strict ordering and exact ties. Wider fingerprints retain FP64 division.
__device__ __forceinline__ double routingSimilarity(const int intersection, const int unionCount, const int numBits) {
  if (unionCount == 0) {
    return 1.0;
  }
  if (numBits <= 4096) {
    return static_cast<double>(__fdiv_rn(static_cast<float>(intersection), static_cast<float>(unionCount)));
  }
  return static_cast<double>(intersection) / unionCount;
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
  return routingSimilarity(intersection, unionCount, storage.numBits);
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
  return routingSimilarity(intersection, unionCount, storage.numBits);
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
  storage.entryCounts[entry]             = 1;
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
    materializedLinearSum(storage, entry, bit) += static_cast<Component>((fingerprint[bit / 32] >> (bit % 32)) & 1U);
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
                                                  const int               rhsSeed,
                                                  const bool              refreshSummaries = true,
                                                  const std::int8_t*      splitAffinity    = nullptr) {
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
      bool assignLeft;
      if (splitAffinity != nullptr) {
        const auto affinity = splitAffinity[entry];
        assignLeft          = affinity > 0 || (affinity == 0 && lhsAssigned <= rhsAssigned);
      } else {
        const double lhsSimilarity = entrySimilarity(storage, entry, lhsSeed);
        const double rhsSimilarity = entrySimilarity(storage, entry, rhsSeed);
        assignLeft = lhsSimilarity > rhsSimilarity || (lhsSimilarity == rhsSimilarity && lhsAssigned <= rhsAssigned);
      }
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
    if (refreshSummaries) {
      summarizeNode(storage, node, lhsEntry);
      summarizeNode(storage, sibling, rhsEntry);
    }
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
  storage.entryChildren[siblingEntry] = sibling;
  if (refreshSummaries) {
    summarizeNode(storage, node, oldParentEntry);
    summarizeNode(storage, sibling, siblingEntry);
  }
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
  // All warps must read the old state before lane zero can publish a new slot.
  // Otherwise a lagging warp can take the already-materialized early return
  // while the allocating warp waits at the barrier below.
  const int existingSlot = storage.entrySummarySlots[entry];
  __syncthreads();
  if (existingSlot >= 0) {
    if (threadIdx.x == 0 && storage.summaryPageHits != nullptr) {
      atomicAdd(storage.summaryPageHits + existingSlot / summaryEntriesPerPage, 1U);
    }
    return true;
  }
  if (threadIdx.x == 0) {
    scratch.materializeFingerprintIndex = storage.entryFingerprintIndices[entry];
    scratch.materializeSlot             = atomicAdd(storage.summaryCursor, 1);
    scratch.success                     = scratch.materializeSlot < storage.maxSummaries;
    if (scratch.success) {
      storage.entrySummarySlots[entry] = scratch.materializeSlot;
      if (storage.summaryPageHits != nullptr) {
        atomicAdd(storage.summaryPageHits + scratch.materializeSlot / summaryEntriesPerPage, 1U);
      }
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
      const std::uint32_t word = singletonWord(storage, scratch.materializeFingerprintIndex, bit / 32);
      value                    = static_cast<Component>((word >> (bit % 32)) & 1U);
    }
    materializedLinearSum(storage, entry, bit) = value;
  }
  for (int word = threadIdx.x; word < storage.numWords; word += blockDim.x) {
    if (scratch.materializeFingerprintIndex >= 0) {
      const int slot = storage.entrySummarySlots[entry];
      storage.centroidPages[slot / summaryEntriesPerPage]
                           [static_cast<std::size_t>(slot % summaryEntriesPerPage) * storage.numWords + word] =
        singletonWord(storage, scratch.materializeFingerprintIndex, word);
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
    const int numWarps  = (blockDim.x + warpSize - 1) / warpSize;
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
                                                       CooperativeScratch&           scratch,
                                                       const int                     excludedEntry = -1) {
  if (threadIdx.x == 0) {
    scratch.next      = storage.nodeHeads[node];
    scratch.bestEntry = -1;
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
      const int entry = scratch.entries[threadIdx.x];
      scratch.values[threadIdx.x] =
        entry == excludedEntry ? -1.0 : entryToFingerprintSimilarity(storage, entry, fingerprint);
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
    const int slot = storage.entrySummarySlots[entry];
    if (slot >= 0 && storage.summaryPageHits != nullptr) {
      atomicAdd(storage.summaryPageHits + slot / summaryEntriesPerPage, 1U);
    }
    scratch.accumulatedValue      = 0.0;
    scratch.accumulatedOtherValue = 0.0;
  }
  __syncthreads();
  const auto combinedCount = static_cast<std::uint64_t>(storage.entryCounts[entry]) + 1;
  for (int base = 0; base < storage.numBits; base += blockDim.x) {
    const int bit         = base + threadIdx.x;
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
    storage.entryFingerprintIndices[entry] = storage.singletonFingerprintPages != nullptr ? entry : fingerprintIndex;
  }
  __syncthreads();
  if (storage.singletonFingerprintPages != nullptr) {
    for (int word = threadIdx.x; word < storage.numWords; word += blockDim.x) {
      storage
        .singletonFingerprintPages[entry / summaryEntriesPerPage]
                                  [static_cast<std::size_t>(entry % summaryEntriesPerPage) * storage.numWords + word] =
        queryFingerprint(storage, fingerprintIndex)[word];
    }
    __syncthreads();
  }
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
    materializedLinearSum(storage, entry, bit) += static_cast<Component>((fingerprint[bit / 32] >> (bit % 32)) & 1U);
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
__device__ __forceinline__ void cooperativeFindSplitSeeds(const TreeStorage<Component>& storage,
                                                          const int                     node,
                                                          CooperativeScratch&           scratch,
                                                          std::uint32_t*                centroidCache = nullptr) {
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
    if (centroidCache != nullptr) {
      // Word-major with an odd stride: pair-search reads are bank-coalesced,
      // and loading consecutive words does not collapse onto one shared bank.
      for (int item = threadIdx.x; item < scratch.count * storage.numWords; item += blockDim.x) {
        const int entryIndex = item / storage.numWords;
        const int word       = item % storage.numWords;
        centroidCache[word * (cooperativeBlockSize + 1) + entryIndex] =
          centroidWord(storage, scratch.nodeEntries[entryIndex], word);
      }
      __syncthreads();
    }
    // Distribute pairs directly instead of having every thread traverse the
    // entire triangle and discard 255/256 of its work. Preserve pair-order ties.
    for (int pair = threadIdx.x; pair < scratch.count * scratch.count; pair += blockDim.x) {
      const int lhsIndex = pair / scratch.count;
      const int rhsIndex = pair % scratch.count;
      if (rhsIndex <= lhsIndex) {
        continue;
      }
      const long long pairOrder =
        static_cast<long long>(lhsIndex) * (2 * scratch.count - lhsIndex - 1) / 2 + rhsIndex - lhsIndex - 1;
      const int lhs = scratch.nodeEntries[lhsIndex];
      const int rhs = scratch.nodeEntries[rhsIndex];
      double    similarity;
      if (centroidCache != nullptr) {
        int intersection = 0;
        int unionCount   = 0;
        for (int word = 0; word < storage.numWords; ++word) {
          const auto lhsWord = centroidCache[word * (cooperativeBlockSize + 1) + lhsIndex];
          const auto rhsWord = centroidCache[word * (cooperativeBlockSize + 1) + rhsIndex];
          intersection += __popc(lhsWord & rhsWord);
          unionCount += __popc(lhsWord | rhsWord);
        }
        similarity = routingSimilarity(intersection, unionCount, storage.numBits);
      } else {
        similarity = entrySimilarity(storage, lhs, rhs);
      }
      if (similarity < localBest || (similarity == localBest && pairOrder < localBestOrder)) {
        localBest      = similarity;
        localBestOrder = pairOrder;
        localLhs       = lhs;
        localRhs       = rhs;
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
    scratch.bestEntry     = lhsSeed;
    scratch.selectedEntry = rhsSeed;
  }
  __syncthreads();
}

template <typename Component>
__device__ __forceinline__ int cooperativeSplitNode(TreeStorage<Component>& storage,
                                                    int                     node,
                                                    const int               branchingFactor,
                                                    CooperativeScratch&     scratch,
                                                    int                     lhsSeed       = -1,
                                                    int                     rhsSeed       = -1,
                                                    const std::int8_t*      splitAffinity = nullptr) {
  while (true) {
    // Especially with precomputed seeds, lane zero otherwise reaches topology
    // writes before a lagging warp has even evaluated the loop predicate.
    const bool overflow = storage.nodeSizes[node] > branchingFactor;
    __syncthreads();
    if (!overflow) {
      break;
    }
    if (lhsSeed < 0) {
      cooperativeFindSplitSeeds(storage, node, scratch);
      lhsSeed = scratch.bestEntry;
      rhsSeed = scratch.selectedEntry;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const int oldParent = storage.nodeParents[node];
      scratch.sourceEntry = *storage.nodeCursor;  // The next allocation is the sibling node.
      scratch.node        = splitNodeWithSeeds(storage, node, branchingFactor, lhsSeed, rhsSeed, false, splitAffinity);
      scratch.success     = scratch.node >= 0;
      scratch.count       = oldParent < 0;
      if (scratch.success) {
        scratch.bestEntry     = parentEntry(storage, node);
        scratch.selectedEntry = parentEntry(storage, scratch.sourceEntry);
      }
    }
    __syncthreads();
    if (!scratch.success) {
      return scratch.node;
    }
    const int  sibling      = scratch.sourceEntry;
    const int  parent       = scratch.node;
    const int  leftSummary  = scratch.bestEntry;
    const int  rightSummary = scratch.selectedEntry;
    const bool newRoot      = scratch.count;
    cooperativeSummarizeNode(storage, node, leftSummary, scratch);
    cooperativeSummarizeNode(storage, sibling, rightSummary, scratch);
    if (newRoot || *storage.status != BitBirchStatus::Success) {
      return *storage.status == BitBirchStatus::Success ? parent : -1;
    }
    node          = parent;
    lhsSeed       = -1;
    rhsSeed       = -1;
    splitAffinity = nullptr;
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
                            storage) &&
      finalize) {
    compactLabels(0, numFingerprints, storage);
  }
}

// Shared-tree epochs have four disjoint phases: route, leaf ownership,
// bottom-up summary refresh, and structural repair. No leaf owner writes an
// ancestor BF or topology belonging to another owner.
template <typename Component> __global__ void bitBirchSharedInitializeKernel(TreeStorage<Component> storage) {
  if (threadIdx.x == 0) {
    *storage.nodeCursor  = 0;
    *storage.entryCursor = 0;
    *storage.numClusters = 0;
    *storage.status      = BitBirchStatus::Success;
    *storage.root        = allocateNode(storage, true, -1);
  }
}

template <typename Component>
__global__ void bitBirchSharedRouteKernel(const int                    begin,
                                          const int                    count,
                                          int*                         keys,
                                          int*                         values,
                                          int*                         groupKeys,
                                          BitBirchStatus*              ownerStatuses,
                                          BitBirchStatus*              groupStatuses,
                                          const bool                   filteredGroups,
                                          const int                    routingWidth,
                                          const double                 threshold,
                                          const TreeStorage<Component> storage) {
  // One cooperative block per query; all tree reads precede any leaf writes.
  __shared__ CooperativeScratch scratch;
  // A two-path beam uses bounded CTA-owned shared state, not per-thread arrays.
  // Candidate order is stable: prior beam order, then entry order within a node.
  __shared__ int                beamNodes[2];
  __shared__ int                candidateEntries[4];
  __shared__ int                candidateNodes[4];
  __shared__ double             candidateScores[4];
  __shared__ int                beamSize;
  __shared__ int                candidateCount;
  const int                     offset = blockIdx.x;
  if (offset >= count) {
    return;
  }
  const int molecule = begin + offset;
  if (threadIdx.x == 0) {
    values[offset]        = molecule;
    ownerStatuses[offset] = BitBirchStatus::Success;
    groupStatuses[offset] = BitBirchStatus::Success;
    scratch.node          = *storage.root;
    keys[offset]          = INT_MAX;
    groupKeys[offset]     = INT_MAX;
  }
  __syncthreads();
  if (storage.labels[molecule] >= 0) {
    return;
  }
  const auto* fingerprint = queryFingerprint(storage, molecule);
  if (routingWidth == 2 && !storage.nodeLeaves[scratch.node]) {
    if (threadIdx.x == 0) {
      beamNodes[0] = scratch.node;
      beamSize     = 1;
    }
    __syncthreads();
    while (true) {
      const bool leafLevel = storage.nodeLeaves[beamNodes[0]];
      const int  width     = beamSize;
      if (threadIdx.x == 0) {
        candidateCount = 0;
      }
      __syncthreads();
      for (int index = 0; index < width; ++index) {
        int excluded = -1;
        for (int rank = 0; rank < (leafLevel ? 1 : 2); ++rank) {
          const int selected = cooperativeClosestEntry(storage, beamNodes[index], fingerprint, scratch, excluded);
          // Capture the returned shared value before another warp reuses it.
          __syncthreads();
          if (selected < 0) {
            break;
          }
          if (threadIdx.x == 0) {
            candidateEntries[candidateCount]  = selected;
            candidateNodes[candidateCount]    = beamNodes[index];
            candidateScores[candidateCount++] = scratch.bestValue;
          }
          excluded = selected;
          __syncthreads();
        }
      }
      if (threadIdx.x == 0) {
        beamSize = min(2, candidateCount);
        for (int rank = 0; rank < (leafLevel ? 1 : beamSize); ++rank) {
          int best = 0;
          for (int index = 1; index < candidateCount; ++index) {
            if (candidateScores[index] > candidateScores[best]) {
              best = index;
            }
          }
          if (leafLevel) {
            scratch.node = candidateNodes[best];
          } else {
            beamNodes[rank] = storage.entryChildren[candidateEntries[best]];
          }
          candidateScores[best] = -1.0;
        }
      }
      __syncthreads();
      if (leafLevel) {
        break;
      }
    }
  } else {
    while (!storage.nodeLeaves[scratch.node]) {
      const int selected = cooperativeClosestEntry(storage, scratch.node, fingerprint, scratch);
      if (threadIdx.x == 0) {
        scratch.node = storage.entryChildren[selected];
      }
      __syncthreads();
    }
  }
  if (threadIdx.x == 0) {
    keys[offset] = scratch.node;
  }
  if (filteredGroups && storage.nodeHeads[scratch.node] >= 0) {
    const int  selected = cooperativeClosestEntry(storage, scratch.node, fingerprint, scratch);
    const auto terms    = cooperativeCombinedISimTerms(storage, selected, fingerprint, scratch);
    if (threadIdx.x == 0 && bitbirch::isimTanimotoAtLeast(terms,
                                                          static_cast<std::uint64_t>(storage.entryCounts[selected]) + 1,
                                                          threshold)) {
      groupKeys[offset] = selected;
    }
  }
}

template <typename Component>
__global__ void bitBirchSharedGroupsKernel(const int                    count,
                                           const int*                   keys,
                                           const int*                   values,
                                           const double                 threshold,
                                           Component*                   proposalSums,
                                           BitBirchStatus*              groupStatuses,
                                           const TreeStorage<Component> sharedTree) {
  const int first = blockIdx.x;
  if (first >= count || keys[first] == INT_MAX || (first > 0 && keys[first - 1] == keys[first])) {
    return;
  }
  __shared__ CooperativeScratch scratch;
  __shared__ TreeStorage<Component> storage;
  const int                         entry = keys[first];
  if (threadIdx.x == 0) {
    storage        = sharedTree;
    storage.status = groupStatuses + first;
    int end        = first + 1;
    while (end < count && keys[end] == entry) {
      ++end;
    }
    scratch.next                  = end;
    scratch.accumulatedValue      = 0;
    scratch.accumulatedOtherValue = 0;
  }
  __syncthreads();
  const int  end           = scratch.next;
  const auto combinedCount = static_cast<std::uint64_t>(storage.entryCounts[entry]) + end - first;
  Component* proposed      = proposalSums + static_cast<std::size_t>(first) * storage.numBits;
  double     common        = 0;
  double     mismatches    = 0;
  for (int bit = threadIdx.x; bit < storage.numBits; bit += blockDim.x) {
    auto sum = static_cast<std::uint64_t>(linearSum(storage, entry, bit));
    for (int offset = first; offset < end; ++offset) {
      const auto word = queryFingerprint(storage, values[offset])[bit / 32];
      sum += (word >> (bit % 32)) & 1U;
    }
    proposed[bit] = static_cast<Component>(sum);
    bitbirch::ISimTanimotoTerms terms{};
    bitbirch::accumulateISimTanimotoTerm(terms, sum, combinedCount);
    common += terms.commonPairs;
    mismatches += terms.mismatches;
  }
  cooperativeAccumulateISimTerms(common, mismatches, scratch);
  const bool merge =
    bitbirch::isimTanimotoAtLeast({scratch.accumulatedValue, scratch.accumulatedOtherValue}, combinedCount, threshold);
  // The rejection path reuses these shared accumulators immediately. Every
  // warp must capture the joint decision before any warp starts that fallback.
  __syncthreads();
  if (merge) {
    if (!cooperativeMaterializeEntry(storage, entry, scratch)) {
      return;
    }
    for (int bit = threadIdx.x; bit < storage.numBits; bit += blockDim.x) {
      materializedLinearSum(storage, entry, bit) = proposed[bit];
    }
    if (threadIdx.x == 0) {
      storage.entryCounts[entry]     = static_cast<std::uint32_t>(combinedCount);
      storage.entryClusterIds[entry] = min(storage.entryClusterIds[entry], values[first]);
    }
    __syncthreads();
    cooperativeRefreshCentroid(storage, entry);
    for (int offset = first + threadIdx.x; offset < end; offset += blockDim.x) {
      storage.labels[values[offset]] = entry;
    }
    return;
  }
  // Joint rejection cannot reject every member: retry ordered individual
  // insertions against this evolving entry, exactly as in the CPU model.
  for (int offset = first; offset < end; ++offset) {
    const int   molecule    = values[offset];
    const auto* fingerprint = queryFingerprint(storage, molecule);
    const auto  terms       = cooperativeCombinedISimTerms(storage, entry, fingerprint, scratch);
    if (threadIdx.x == 0) {
      scratch.count =
        bitbirch::isimTanimotoAtLeast(terms, static_cast<std::uint64_t>(storage.entryCounts[entry]) + 1, threshold);
    }
    __syncthreads();
    if (scratch.count) {
      cooperativeAddFingerprint(storage, entry, fingerprint, scratch);
      if (threadIdx.x == 0) {
        storage.labels[molecule]       = entry;
        storage.entryClusterIds[entry] = min(storage.entryClusterIds[entry], molecule);
      }
      __syncthreads();
    }
  }
}

template <typename Component> __device__ __forceinline__ int allocateSharedLeafEntry(TreeStorage<Component>& storage) {
  const int entry = atomicAdd(storage.entryCursor, 1);
  if (entry >= storage.maxEntries) {
    *storage.status = BitBirchStatus::EntryCapacity;
    return -1;
  }
  storage.entryNext[entry]               = -1;
  storage.entryChildren[entry]           = -1;
  storage.entryCounts[entry]             = 0;
  storage.entrySummarySlots[entry]       = -1;
  storage.entryFingerprintIndices[entry] = -1;
  return entry;
}

template <typename Component>
__global__ void bitBirchSharedLeafOwnersKernel(const int                    count,
                                               const int*                   keys,
                                               const int*                   values,
                                               const double                 threshold,
                                               const int                    branchingFactor,
                                               int*                         dirtyNodes,
                                               BitBirchStatus*              ownerStatuses,
                                               const TreeStorage<Component> sharedTree) {
  const int first = blockIdx.x;
  if (first >= count || keys[first] == INT_MAX || (first > 0 && keys[first - 1] == keys[first])) {
    return;
  }
  __shared__ CooperativeScratch scratch;
  __shared__ TreeStorage<Component> storage;
  const int                         node = keys[first];
  // Each owner has independent error storage as well as exclusive leaf data.
  if (threadIdx.x == 0) {
    storage         = sharedTree;
    storage.status  = ownerStatuses + first;
    scratch.success = true;
  }
  __syncthreads();
  for (int offset = first; offset < count && keys[offset] == node; ++offset) {
    const int molecule = values[offset];
    if (storage.labels[molecule] >= 0) {
      continue;
    }
    const auto* fingerprint = queryFingerprint(storage, molecule);
    int         selected    = -1;
    if (storage.nodeHeads[node] >= 0) {
      selected = cooperativeClosestEntry(storage, node, fingerprint, scratch);
    }
    bool merge = false;
    if (selected >= 0) {
      const auto terms = cooperativeCombinedISimTerms(storage, selected, fingerprint, scratch);
      // Publish one decision before any warp can mutate the entry count. Reading
      // that count independently immediately before the update can diverge warps.
      if (threadIdx.x == 0) {
        scratch.count = bitbirch::isimTanimotoAtLeast(terms,
                                                      static_cast<std::uint64_t>(storage.entryCounts[selected]) + 1,
                                                      threshold);
      }
      __syncthreads();
      merge = scratch.count;
    }
    if (merge) {
      cooperativeAddFingerprint(storage, selected, fingerprint, scratch);
    } else {
      if (threadIdx.x == 0) {
        scratch.selectedEntry = allocateSharedLeafEntry(storage);
        scratch.success       = scratch.selectedEntry >= 0;
      }
      __syncthreads();
      if (!scratch.success) {
        return;
      }
      selected = scratch.selectedEntry;
      cooperativeInitializeLeafEntry(storage, selected, molecule);
      if (threadIdx.x == 0) {
        storage.entryClusterIds[selected] = molecule;
        appendEntry(storage, node, selected);
      }
      __syncthreads();
    }
    if (*storage.status != BitBirchStatus::Success) {
      return;
    }
    if (threadIdx.x == 0) {
      storage.labels[molecule]          = selected;
      // A parked earlier query can reach an entry created later in its batch.
      // Track the actual earliest member, not the allocation/creation order.
      storage.entryClusterIds[selected] = min(storage.entryClusterIds[selected], molecule);
    }
    __syncthreads();
    if (storage.nodeSizes[node] > branchingFactor) {
      // Remaining queries stay unassigned and are rerouted after the split.
      break;
    }
  }
  if (threadIdx.x == 0) {
    for (int changed = node; storage.nodeParents[changed] >= 0; changed = storage.nodeParents[changed]) {
      atomicExch(dirtyNodes + changed, 1);
    }
  }
}

template <typename Component>
__global__ void bitBirchSharedRefreshLevelKernel(const int                    numNodes,
                                                 const int                    depth,
                                                 const int                    count,
                                                 const int*                   keys,
                                                 const int*                   values,
                                                 int*                         dirtyNodes,
                                                 int*                         deltaSlots,
                                                 int*                         deltaCursor,
                                                 const int                    deltaCapacity,
                                                 Component*                   deltaSums,
                                                 std::uint32_t*               deltaCounts,
                                                 BitBirchStatus*              nodeStatuses,
                                                 const TreeStorage<Component> sharedTree) {
  const int node = blockIdx.x;
  if (node >= numNodes || !dirtyNodes[node]) {
    return;
  }
  int nodeDepth = 0;
  for (int ancestor = node; sharedTree.nodeParents[ancestor] >= 0; ancestor = sharedTree.nodeParents[ancestor]) {
    ++nodeDepth;
  }
  if (nodeDepth != depth) {
    return;
  }
  __shared__ CooperativeScratch scratch;
  __shared__ TreeStorage<Component> storage;
  if (threadIdx.x == 0) {
    storage               = sharedTree;
    storage.status        = nodeStatuses + node;
    *storage.status       = BitBirchStatus::Success;
    scratch.selectedEntry = parentEntry(storage, node);
    scratch.sourceEntry   = atomicAdd(deltaCursor, 1);
    scratch.success       = scratch.selectedEntry >= 0 && scratch.sourceEntry < deltaCapacity;
    if (!scratch.success) {
      *storage.status = scratch.selectedEntry < 0 ? BitBirchStatus::InvalidTree : BitBirchStatus::SummaryCapacity;
    } else {
      deltaSlots[node]         = scratch.sourceEntry;
      std::uint32_t addedCount = 0;
      if (storage.nodeLeaves[node]) {
        int first = 0;
        int end   = count;
        while (first < end) {
          const int middle = first + (end - first) / 2;
          if (keys[middle] < node) {
            first = middle + 1;
          } else {
            end = middle;
          }
        }
        scratch.next = first;
        while (end < count && keys[end] == node) {
          addedCount += storage.labels[values[end]] >= 0;
          ++end;
        }
        scratch.count = end;
      } else {
        for (int entry = storage.nodeHeads[node]; entry >= 0; entry = storage.entryNext[entry]) {
          const int slot = deltaSlots[storage.entryChildren[entry]];
          if (slot >= 0) {
            addedCount += deltaCounts[slot];
          }
        }
      }
      deltaCounts[scratch.sourceEntry] = addedCount;
    }
  }
  __syncthreads();
  if (!scratch.success || !cooperativeMaterializeEntry(storage, scratch.selectedEntry, scratch)) {
    return;
  }
  const int target = scratch.selectedEntry;
  const int slot   = scratch.sourceEntry;
  if (threadIdx.x == 0) {
    storage.entryCounts[target] += deltaCounts[slot];
  }
  // Entry owners already updated leaf payloads. Propagate only this epoch's
  // accepted fingerprints, rather than rereading every old BF in each subtree.
  // One CTA owns each parent entry; no per-bit atomics or shared ancestor writes.
  for (int bit = threadIdx.x; bit < storage.numBits; bit += blockDim.x) {
    Component added = 0;
    if (storage.nodeLeaves[node]) {
      for (int offset = scratch.next; offset < scratch.count; ++offset) {
        const int molecule = values[offset];
        if (storage.labels[molecule] >= 0) {
          const auto word = queryFingerprint(storage, molecule)[bit / 32];
          added += static_cast<Component>((word >> (bit % 32)) & 1U);
        }
      }
    } else {
      for (int entry = storage.nodeHeads[node]; entry >= 0; entry = storage.entryNext[entry]) {
        const int childSlot = deltaSlots[storage.entryChildren[entry]];
        if (childSlot >= 0) {
          added += deltaSums[static_cast<std::size_t>(childSlot) * storage.numBits + bit];
        }
      }
    }
    deltaSums[static_cast<std::size_t>(slot) * storage.numBits + bit] = added;
    materializedLinearSum(storage, target, bit) += added;
  }
  __syncthreads();
  cooperativeRefreshCentroid(storage, target);
  if (threadIdx.x == 0) {
    dirtyNodes[node] = 0;
  }
}

template <typename Component>
__global__ void bitBirchSharedSplitSeedsKernel(const int                    count,
                                               const int*                   keys,
                                               const int                    branchingFactor,
                                               int*                         splitLeft,
                                               int*                         splitRight,
                                               std::int8_t*                 splitAffinity,
                                               const TreeStorage<Component> storage) {
  const int index = blockIdx.x;
  if (index >= count || keys[index] == INT_MAX || (index > 0 && keys[index - 1] == keys[index]) ||
      storage.nodeSizes[keys[index]] <= branchingFactor) {
    return;
  }
  // Other leaf splits cannot change this leaf's entries. Its exact seed search
  // can run concurrently; ancestor cascades still search after prior repairs.
  __shared__ CooperativeScratch scratch;
  extern __shared__ std::uint32_t centroidCache[];
  const bool                      cacheCentroids = storage.numWords <= 32 && branchingFactor < cooperativeBlockSize;
  cooperativeFindSplitSeeds(storage, keys[index], scratch, cacheCentroids ? centroidCache : nullptr);
  const int left  = scratch.bestEntry;
  const int right = scratch.selectedEntry;
  if (storage.nodeSizes[keys[index]] <= blockDim.x) {
    if (threadIdx.x < scratch.count) {
      const int    entry   = scratch.nodeEntries[threadIdx.x];
      const double lhs     = entrySimilarity(storage, entry, left);
      const double rhs     = entrySimilarity(storage, entry, right);
      splitAffinity[entry] = static_cast<std::int8_t>((lhs > rhs) - (lhs < rhs));
    }
  } else {
    int offset = 0;
    for (int entry = storage.nodeHeads[keys[index]]; entry >= 0; entry = storage.entryNext[entry], ++offset) {
      if (offset % blockDim.x == threadIdx.x) {
        const double lhs     = entrySimilarity(storage, entry, left);
        const double rhs     = entrySimilarity(storage, entry, right);
        splitAffinity[entry] = static_cast<std::int8_t>((lhs > rhs) - (lhs < rhs));
      }
    }
  }
  if (threadIdx.x == 0) {
    splitLeft[index]  = scratch.bestEntry;
    splitRight[index] = scratch.selectedEntry;
  }
}

template <typename Component>
__global__ void bitBirchSharedRepairKernel(const int              begin,
                                           const int              count,
                                           const int*             keys,
                                           const int*             splitLeft,
                                           const int*             splitRight,
                                           const std::int8_t*     splitAffinity,
                                           const BitBirchStatus*  ownerStatuses,
                                           const BitBirchStatus*  groupStatuses,
                                           const BitBirchStatus*  nodeStatuses,
                                           const int              branchingFactor,
                                           int*                   control,
                                           TreeStorage<Component> storage) {
  __shared__ CooperativeScratch scratch;
  if (threadIdx.x == 0) {
    *storage.status = BitBirchStatus::Success;
    for (int index = 0; index < count; ++index) {
      if (ownerStatuses[index] != BitBirchStatus::Success) {
        *storage.status = ownerStatuses[index];
      }
      if (groupStatuses[index] != BitBirchStatus::Success) {
        *storage.status = groupStatuses[index];
      }
    }
    for (int node = 0; node < *storage.nodeCursor; ++node) {
      if (nodeStatuses[node] != BitBirchStatus::Success) {
        *storage.status = nodeStatuses[node];
      }
    }
  }
  __syncthreads();
  for (int index = 0; index < count; ++index) {
    const int node = keys[index];
    // Immutable keys can be skipped without a barrier, unlike mutable topology.
    if (node == INT_MAX || (index > 0 && keys[index - 1] == node)) {
      continue;
    }
    const bool split = *storage.status == BitBirchStatus::Success && storage.nodeSizes[node] > branchingFactor;
    // Capture eligibility in every warp before any warp can change nodeSizes.
    __syncthreads();
    if (!split) {
      continue;
    }
    cooperativeSplitNode(storage, node, branchingFactor, scratch, splitLeft[index], splitRight[index], splitAffinity);
    // Payload refresh already made every ancestor BF current. Splitting only
    // redistributes the same members: each split updates its two child BFs,
    // while the total BF of their parent (and higher ancestors) is unchanged.
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    int pending = 0;
    for (int molecule = begin; molecule < begin + count; ++molecule) {
      pending += storage.labels[molecule] < 0;
    }
    int depth = 0;
    // Node zero remains a leaf; every leaf in this tree has equal depth.
    for (int node = 0; storage.nodeParents[node] >= 0; node = storage.nodeParents[node]) {
      ++depth;
    }
    control[0] = pending;
    control[1] = *storage.nodeCursor;
    control[2] = *storage.summaryCursor;
    control[3] = static_cast<int>(*storage.status);
    control[4] = depth;
    control[5] = *storage.entryCursor;
  }
}

template <typename Component>
__global__ void bitBirchSharedCollectLeavesKernel(const int                    count,
                                                  int*                         firstMembers,
                                                  int*                         leafEntries,
                                                  int*                         cursor,
                                                  const TreeStorage<Component> storage) {
  const int entry = blockIdx.x * blockDim.x + threadIdx.x;
  if (entry < count && storage.entryChildren[entry] < 0) {
    const int slot     = atomicAdd(cursor, 1);
    firstMembers[slot] = storage.entryClusterIds[entry];
    leafEntries[slot]  = entry;
  }
}

template <typename Component>
__global__ void bitBirchSharedClusterMapKernel(const int              count,
                                               const int*             leafEntries,
                                               TreeStorage<Component> storage) {
  const int cluster = blockIdx.x * blockDim.x + threadIdx.x;
  if (cluster < count) {
    const int entry                = leafEntries[cluster];
    storage.entryClusterIds[entry] = cluster;
    if (storage.centroids != nullptr) {
      for (int word = 0; word < storage.numWords; ++word) {
        storage.centroids[static_cast<std::size_t>(cluster) * storage.numWords + word] =
          centroidWord(storage, entry, word);
      }
    }
  }
  if (cluster == 0) {
    *storage.numClusters = count;
  }
}

template <typename Component>
__global__ void bitBirchSharedFinalizeKernel(const int count, TreeStorage<Component> storage) {
  const int molecule = blockIdx.x * blockDim.x + threadIdx.x;
  if (molecule < count) {
    storage.labels[molecule] = storage.entryClusterIds[storage.labels[molecule]];
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
  const int                     partition      = blockIdx.x;
  const int                     partitionBegin = partition * partitionSize;
  const int                     begin          = partitionBegin + partitionOffset;
  const int                     end   = min(min(begin + batchSize, partitionBegin + partitionSize), numFingerprints);
  auto                          local = partitionStorage(storage, partition, nodeStride, entryStride);
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
__device__ __forceinline__ double entryToSummarySimilarity(const TreeStorage<Component>&        storage,
                                                           const int                            entry,
                                                           const TreeStorage<SummaryComponent>& sourceStorage,
                                                           const int                            sourceEntry) {
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
__device__ __forceinline__ int cooperativeClosestSummaryEntry(const TreeStorage<Component>&        storage,
                                                              const int                            node,
                                                              const TreeStorage<SummaryComponent>& sourceStorage,
                                                              const int                            sourceEntry,
                                                              CooperativeScratch&                  scratch) {
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
  const TreeStorage<Component>&        storage,
  const int                            entry,
  const TreeStorage<SummaryComponent>& sourceStorage,
  const int                            sourceEntry,
  const std::uint32_t                  candidateCount,
  CooperativeScratch&                  scratch) {
  if (threadIdx.x == 0) {
    scratch.accumulatedValue      = 0.0;
    scratch.accumulatedOtherValue = 0.0;
  }
  __syncthreads();
  const auto combinedCount = static_cast<std::uint64_t>(storage.entryCounts[entry]) + candidateCount;
  for (int base = 0; base < storage.numBits; base += blockDim.x) {
    const int bit         = base + threadIdx.x;
    double    commonPairs = 0.0;
    double    mismatches  = 0.0;
    if (bit < storage.numBits) {
      const auto component = static_cast<std::uint64_t>(linearSum(storage, entry, bit)) +
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
__device__ int cooperativeInsertSummary(TreeStorage<Component>&              storage,
                                        const TreeStorage<SummaryComponent>& sourceStorage,
                                        const int                            sourceEntry,
                                        const std::uint32_t                  candidateCount,
                                        const double                         threshold,
                                        const int                            branchingFactor,
                                        CooperativeScratch&                  scratch) {
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
    const auto combinedTerms = cooperativeCombinedSummaryISimTerms(storage,
                                                                   scratch.selectedEntry,
                                                                   sourceStorage,
                                                                   sourceEntry,
                                                                   candidateCount,
                                                                   scratch);
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
        storage.entryFingerprintIndices[scratch.selectedEntry] = sourceStorage.entryFingerprintIndices[sourceEntry];
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

template <typename Component>
__global__ void bitBirchCollectForestEntriesKernel(const int              entryStride,
                                                   const int*             clusterOffsets,
                                                   int*                   sourceEntries,
                                                   TreeStorage<Component> storage) {
  const int tree  = blockIdx.x;
  auto      local = partitionStorage(storage, tree, 0, entryStride);
  for (int entry = threadIdx.x; entry < *local.entryCursor; entry += blockDim.x) {
    const int cluster = local.entryClusterIds[entry];
    if (cluster >= 0) {
      sourceEntries[clusterOffsets[tree] + cluster] = tree * entryStride + entry;
    }
  }
}

template <typename Component, typename SourceComponent>
__global__ void bitBirchMergeIndexedTreeGroupsKernel(const int                    numInputTrees,
                                                     const int                    mergeFanIn,
                                                     const int                    sourceOffset,
                                                     const int                    batchSize,
                                                     const bool                   initialize,
                                                     const int*                   sourceClusterOffsets,
                                                     const int*                   sourceEntries,
                                                     TreeStorage<SourceComponent> sourceStorage,
                                                     const double                 threshold,
                                                     const int                    branchingFactor,
                                                     int*                         sourceToOutput,
                                                     const int                    outputNodeStride,
                                                     const int                    outputEntryStride,
                                                     TreeStorage<Component>       outputStorage) {
  __shared__ CooperativeScratch scratch;
  const int                     outputTree        = blockIdx.x;
  const int                     firstInputTree    = outputTree * mergeFanIn;
  const int                     pastLastInputTree = min(firstInputTree + mergeFanIn, numInputTrees);
  const int                     groupBegin        = sourceClusterOffsets[firstInputTree];
  const int                     groupEnd          = sourceClusterOffsets[pastLastInputTree];
  const int                     begin             = min(groupBegin + sourceOffset, groupEnd);
  const int                     end               = min(begin + batchSize, groupEnd);
  auto local = partitionStorage(outputStorage, outputTree, outputNodeStride, outputEntryStride);

  if (initialize && threadIdx.x == 0) {
    *local.status      = BitBirchStatus::Success;
    *local.nodeCursor  = 0;
    *local.entryCursor = 0;
    *local.numClusters = 0;
    *local.root        = allocateNode(local, true, -1);
    scratch.success    = *local.root >= 0;
  } else if (threadIdx.x == 0) {
    scratch.success = *local.status == BitBirchStatus::Success;
  }
  __syncthreads();

  for (int sourceIndex = begin; sourceIndex < end; ++sourceIndex) {
    const int sourceEntry = sourceEntries[sourceIndex];
    const int outputEntry = cooperativeInsertSummary(local,
                                                     sourceStorage,
                                                     sourceEntry,
                                                     sourceStorage.entryCounts[sourceEntry],
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
      sourceToOutput[sourceEntry] = scratch.selectedEntry;
    }
  }
}

__global__ void bitBirchRemapForestLabelsKernel(const int  numFingerprints,
                                                const int  inputPartitionSize,
                                                const int  inputEntryStride,
                                                const int* sourceToOutput,
                                                int*       labels) {
  for (int fingerprintIndex = blockIdx.x * blockDim.x + threadIdx.x; fingerprintIndex < numFingerprints;
       fingerprintIndex += blockDim.x * gridDim.x) {
    const int inputTree      = fingerprintIndex / inputPartitionSize;
    const int sourceEntry    = inputTree * inputEntryStride + labels[fingerprintIndex];
    labels[fingerprintIndex] = sourceToOutput[sourceEntry];
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

template <typename Component, typename SourceComponent>
std::unique_ptr<PartitionedForest<Component>> mergeForestRound(const std::uint32_t*         fingerprints,
                                                               const int                    numFingerprints,
                                                               const int                    numWords,
                                                               const double                 threshold,
                                                               const int                    branchingFactor,
                                                               const int                    mergeFanIn,
                                                               const int                    sourceNumTrees,
                                                               const int                    sourcePartitionSize,
                                                               const int                    sourceEntryStride,
                                                               const std::size_t            sourceTotalEntries,
                                                               const std::vector<int>&      sourceClusterCounts,
                                                               TreeStorage<SourceComponent> sourceStorage,
                                                               int*                         labels,
                                                               const cudaStream_t           stream) {
  const ScopedNvtxRange range("BitBIRCH hierarchical merge round");
  std::vector<int>      sourceClusterOffsets(sourceNumTrees + 1);
  int                   totalSourceClusters = 0;
  for (int tree = 0; tree < sourceNumTrees; ++tree) {
    sourceClusterOffsets[tree] = totalSourceClusters;
    totalSourceClusters += sourceClusterCounts[tree];
  }
  sourceClusterOffsets[sourceNumTrees] = totalSourceClusters;

  AsyncDeviceVector<int> deviceSourceClusterOffsets(sourceNumTrees + 1, stream);
  AsyncDeviceVector<int> sourceEntries(totalSourceClusters, stream);
  AsyncDeviceVector<int> sourceToOutput(sourceTotalEntries, stream);
  deviceSourceClusterOffsets.copyFromHost(sourceClusterOffsets);
  bitBirchCollectForestEntriesKernel<<<sourceNumTrees, cooperativeBlockSize, 0, stream>>>(
    sourceEntryStride,
    deviceSourceClusterOffsets.data(),
    sourceEntries.data(),
    sourceStorage);
  cudaCheckError(cudaGetLastError());

  const int outputNumTrees      = (sourceNumTrees + mergeFanIn - 1) / mergeFanIn;
  const int outputPartitionSize = static_cast<int>(
    std::min<std::int64_t>(numFingerprints, static_cast<std::int64_t>(sourcePartitionSize) * mergeFanIn));
  auto output = std::make_unique<PartitionedForest<Component>>(fingerprints,
                                                               labels,
                                                               numFingerprints,
                                                               outputNumTrees,
                                                               outputPartitionSize,
                                                               numWords,
                                                               stream);

  int maxGroupClusters = 0;
  for (int outputTree = 0; outputTree < outputNumTrees; ++outputTree) {
    const int firstInputTree    = outputTree * mergeFanIn;
    const int pastLastInputTree = std::min(firstInputTree + mergeFanIn, sourceNumTrees);
    maxGroupClusters =
      std::max(maxGroupClusters, sourceClusterOffsets[pastLastInputTree] - sourceClusterOffsets[firstInputTree]);
  }

  constexpr int               reservedSummarySlotsPerInput = 8;
  int                         hostSummaryCursor            = 0;
  std::vector<BitBirchStatus> hostStatuses(outputNumTrees);
  for (int sourceOffset = 0; sourceOffset < maxGroupClusters; sourceOffset += summaryBuildBatchSize) {
    int batchInputs = 0;
    for (int outputTree = 0; outputTree < outputNumTrees; ++outputTree) {
      const int firstInputTree    = outputTree * mergeFanIn;
      const int pastLastInputTree = std::min(firstInputTree + mergeFanIn, sourceNumTrees);
      const int groupClusters     = sourceClusterOffsets[pastLastInputTree] - sourceClusterOffsets[firstInputTree];
      batchInputs += std::max(0, std::min(summaryBuildBatchSize, groupClusters - sourceOffset));
    }
    const auto requiredSummaries = static_cast<std::int64_t>(hostSummaryCursor) +
                                   static_cast<std::int64_t>(reservedSummarySlotsPerInput) * batchInputs +
                                   4 * outputNumTrees;
    if (requiredSummaries > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("BitBIRCH hierarchical summary workspace exceeds the supported index range");
    }
    output->summaryArena.reserve(static_cast<int>(requiredSummaries));
    auto outputStorage = output->storage();
    bitBirchMergeIndexedTreeGroupsKernel<Component, SourceComponent>
      <<<outputNumTrees, cooperativeBlockSize, 0, stream>>>(sourceNumTrees,
                                                            mergeFanIn,
                                                            sourceOffset,
                                                            summaryBuildBatchSize,
                                                            sourceOffset == 0,
                                                            deviceSourceClusterOffsets.data(),
                                                            sourceEntries.data(),
                                                            sourceStorage,
                                                            threshold,
                                                            branchingFactor,
                                                            sourceToOutput.data(),
                                                            output->nodeStride,
                                                            output->entryStride,
                                                            outputStorage);
    cudaCheckError(cudaGetLastError());
    output->summaryCursor.get(hostSummaryCursor);
    output->statuses.copyToHost(hostStatuses);
    cudaCheckError(cudaStreamSynchronize(stream));
    for (const auto status : hostStatuses) {
      if (status != BitBirchStatus::Success) {
        throw std::runtime_error("BitBIRCH hierarchical merge capacity or structural failure (status " +
                                 std::to_string(static_cast<int>(status)) + ")");
      }
    }
  }

  constexpr int labelBlockSize = 256;
  const int     labelBlocks    = std::min(4096, (numFingerprints + labelBlockSize - 1) / labelBlockSize);
  bitBirchRemapForestLabelsKernel<<<labelBlocks, labelBlockSize, 0, stream>>>(numFingerprints,
                                                                              sourcePartitionSize,
                                                                              sourceEntryStride,
                                                                              sourceToOutput.data(),
                                                                              labels);
  cudaCheckError(cudaGetLastError());
  auto outputStorage = output->storage();
  bitBirchIndexForestClustersKernel<<<outputNumTrees, 1, 0, stream>>>(numFingerprints,
                                                                      outputPartitionSize,
                                                                      output->nodeStride,
                                                                      output->entryStride,
                                                                      outputStorage);
  cudaCheckError(cudaGetLastError());
  output->hostClusterCounts.resize(outputNumTrees);
  output->clusterCounts.copyToHost(output->hostClusterCounts);
  output->statuses.copyToHost(hostStatuses);
  cudaCheckError(cudaStreamSynchronize(stream));
  for (int tree = 0; tree < outputNumTrees; ++tree) {
    if (hostStatuses[tree] != BitBirchStatus::Success) {
      throw std::runtime_error("BitBIRCH hierarchical merge capacity or structural failure (status " +
                               std::to_string(static_cast<int>(hostStatuses[tree])) + ")");
    }
    output->totalClusters += output->hostClusterCounts[tree];
  }
  return output;
}

template <typename Component>
void finalizeForest(const int               numFingerprints,
                    const int               partitionSize,
                    const int               nodeStride,
                    const int               entryStride,
                    const std::vector<int>& clusterCounts,
                    std::uint32_t*          outputCentroids,
                    TreeStorage<Component>  storage,
                    const cudaStream_t      stream) {
  std::vector<int> clusterOffsets(clusterCounts.size());
  int              offset = 0;
  for (std::size_t tree = 0; tree < clusterCounts.size(); ++tree) {
    clusterOffsets[tree] = offset;
    offset += clusterCounts[tree];
  }
  AsyncDeviceVector<int> deviceClusterOffsets(clusterOffsets.size(), stream);
  deviceClusterOffsets.copyFromHost(clusterOffsets);
  bitBirchFinalizeForestKernel<<<static_cast<int>(clusterCounts.size()), cooperativeBlockSize, 0, stream>>>(
    numFingerprints,
    partitionSize,
    nodeStride,
    entryStride,
    deviceClusterOffsets.data(),
    outputCentroids,
    storage);
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaStreamSynchronize(stream));
}

template <typename Component>
BitBirchResult launchShared(const cuda::std::span<const std::uint32_t> fingerprints,
                            const int                                  numFingerprints,
                            const int                                  numWords,
                            const double                               threshold,
                            const int                                  branchingFactor,
                            const int                                  insertionBatchSize,
                            const bool                                 filteredGroups,
                            const int                                  orderedPrefixSize,
                            const int                                  routingWidth,
                            const std::size_t                          summaryCacheBytes,
                            const bool                                 fingerprintsOnHost,
                            const bool                                 clusterIdsOnHost,
                            const bool                                 returnCentroids,
                            const cudaStream_t                         stream) {
  const int         batchCapacity   = std::min(insertionBatchSize, numFingerprints);
  const std::size_t splitCacheBytes = numWords <= 32 && branchingFactor < cooperativeBlockSize ?
                                        sizeof(std::uint32_t) * (cooperativeBlockSize + 1) * numWords :
                                        0;
  BitBirchResult    result{AsyncDeviceVector<int>(clusterIdsOnHost ? 0 : numFingerprints, stream),
                        AsyncDeviceVector<std::uint32_t>(0, stream),
                        0,
                        numWords};
  int*              labels = result.clusterIds.data();
  if (clusterIdsOnHost) {
    result.hostClusterIds   = PinnedHostVector<int>(numFingerprints);
    result.clusterIdsOnHost = true;
    cudaCheckError(cudaHostGetDevicePointer(&labels, result.hostClusterIds.data(), 0));
  }
  AsyncDeviceVector<std::uint32_t> inputTile(
    fingerprintsOnHost ? static_cast<std::size_t>(batchCapacity) * numWords : 0,
    stream);
  PagedFingerprintArena        ownedFingerprints(numWords, stream);
  // Balanced splits leave every non-root node with at least m entries. With
  // K leaf entries and V nodes, the tree has K + V - 1 total entries, hence
  // (m - 1)*(V - 1) <= K <= N. Padding also covers transient split allocations.
  // For B=254 this needs about N/126 nodes and 1.008*N entries, not 2*N/3*N.
  const int                    minimumNodeSize = branchingFactor / 2 + branchingFactor % 2;
  const int                    maxNodes        = numFingerprints / (minimumNodeSize - 1) + 8;
  const int                    maxEntries      = numFingerprints + maxNodes;
  const int                    initialNodes    = std::min(maxNodes, batchCapacity / (minimumNodeSize - 1) + 8);
  const int                    initialEntries  = std::min(maxEntries, batchCapacity + initialNodes);
  PartitionedForest<Component> tree(fingerprintsOnHost ? inputTile.data() : fingerprints.data(),
                                    labels,
                                    numFingerprints,
                                    1,
                                    numFingerprints,
                                    numWords,
                                    stream,
                                    initialNodes,
                                    initialEntries,
                                    summaryCacheBytes);
  AsyncDeviceVector<int>         keys(batchCapacity, stream);
  AsyncDeviceVector<int>         sortedKeys(batchCapacity, stream);
  AsyncDeviceVector<int>         values(batchCapacity, stream);
  AsyncDeviceVector<int>         sortedValues(batchCapacity, stream);
  AsyncDeviceVector<int>         groupKeys(batchCapacity, stream);
  AsyncDeviceVector<int>         sortedGroupKeys(batchCapacity, stream);
  AsyncDeviceVector<int>         sortedGroupValues(batchCapacity, stream);
  AsyncDeviceVector<int>         splitLeft(batchCapacity, stream);
  AsyncDeviceVector<int>         splitRight(batchCapacity, stream);
  AsyncDeviceVector<std::int8_t> splitAffinity(tree.totalEntries, stream);
  AsyncDeviceVector<Component>   proposalSums(
    filteredGroups ? static_cast<std::size_t>(batchCapacity) * numWords * 32 : 0,
    stream);
  AsyncDeviceVector<int>            dirtyNodes(tree.totalNodes, stream);
  AsyncDeviceVector<int>            deltaSlots(tree.totalNodes, stream);
  AsyncDeviceVector<int>            deltaCursor(1, stream);
  AsyncDeviceVector<Component>      deltaSums(0, stream);
  AsyncDeviceVector<std::uint32_t>  deltaCounts(0, stream);
  AsyncDeviceVector<BitBirchStatus> nodeStatuses(tree.totalNodes, stream);
  AsyncDeviceVector<BitBirchStatus> ownerStatuses(batchCapacity, stream);
  AsyncDeviceVector<BitBirchStatus> groupStatuses(batchCapacity, stream);
  AsyncDeviceVector<int>            control(6, stream);
  cudaCheckError(cudaMemsetAsync(dirtyNodes.data(), 0, tree.totalNodes * sizeof(int), stream));
  cudaCheckError(cudaMemsetAsync(nodeStatuses.data(), 0, tree.totalNodes * sizeof(BitBirchStatus), stream));
  auto storage = tree.storage();
  bitBirchSharedInitializeKernel<<<1, 1, 0, stream>>>(storage);
  cudaCheckError(cudaGetLastError());
  std::size_t sortBytes = 0;
  cudaCheckError(cub::DeviceRadixSort::SortPairs(nullptr,
                                                 sortBytes,
                                                 keys.data(),
                                                 sortedKeys.data(),
                                                 values.data(),
                                                 sortedValues.data(),
                                                 batchCapacity,
                                                 0,
                                                 32,
                                                 stream));
  AsyncDeviceVector<std::byte> sortScratch(sortBytes, stream);
  std::vector<int>             hostControl{0, 1, 0, 0, 0, 0};
  for (int begin = 0; begin < numFingerprints;) {
    const bool useGroups = filteredGroups && begin >= orderedPrefixSize;
    const int  boundary =
      filteredGroups && begin < orderedPrefixSize ? std::min(orderedPrefixSize, numFingerprints) : numFingerprints;
    const int count = std::min(batchCapacity, boundary - begin);
    if (fingerprintsOnHost) {
      cudaCheckError(cudaMemcpyAsync(inputTile.data(),
                                     fingerprints.data() + static_cast<std::size_t>(begin) * numWords,
                                     static_cast<std::size_t>(count) * numWords * sizeof(std::uint32_t),
                                     cudaMemcpyHostToDevice,
                                     stream));
    }
    cudaCheckError(cudaMemsetAsync(labels + begin, 0xff, count * sizeof(int), stream));
    int pending = count;
    while (pending > 0) {
      // Every non-root node has exactly one directory entry, so K = E - V + 1.
      // At most pending new leaf entries can be created before the next barrier.
      // Apply the same balanced-node bound to live K instead of allocating for
      // the all-singleton N worst case. Stable integer IDs survive buffer growth.
      const int clusterBound    = hostControl[5] - hostControl[1] + 1 + pending;
      const int requiredNodes   = clusterBound / (minimumNodeSize - 1) + 8;
      const int requiredEntries = clusterBound + requiredNodes;
      const int nodeCapacity =
        requiredNodes > tree.nodeStride ?
          static_cast<int>(
            std::min<std::int64_t>(maxNodes, std::max<std::int64_t>(requiredNodes, tree.nodeStride * 3LL / 2))) :
          tree.nodeStride;
      const int entryCapacity =
        requiredEntries > tree.entryStride ?
          static_cast<int>(
            std::min<std::int64_t>(maxEntries, std::max<std::int64_t>(requiredEntries, tree.entryStride * 3LL / 2))) :
          tree.entryStride;
      const auto oldNodeCapacity = tree.totalNodes;
      tree.growSingleTree(nodeCapacity, entryCapacity);
      if (tree.totalNodes > oldNodeCapacity) {
        dirtyNodes.resize(tree.totalNodes);
        deltaSlots.resize(tree.totalNodes);
        nodeStatuses.resize(tree.totalNodes);
        cudaCheckError(cudaMemsetAsync(dirtyNodes.data() + oldNodeCapacity,
                                       0,
                                       (tree.totalNodes - oldNodeCapacity) * sizeof(int),
                                       stream));
        cudaCheckError(cudaMemsetAsync(nodeStatuses.data() + oldNodeCapacity,
                                       0,
                                       (tree.totalNodes - oldNodeCapacity) * sizeof(BitBirchStatus),
                                       stream));
      }
      if (splitAffinity.size() < tree.totalEntries) {
        splitAffinity.resize(tree.totalEntries);
      }
      // Bound lazy materialization plus two summaries per possible split level.
      const auto required =
        static_cast<std::int64_t>(hostControl[2]) + static_cast<std::int64_t>(pending) * (2 * hostControl[4] + 5) + 4;
      if (required > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("BitBIRCH shared summary workspace exceeds index capacity");
      }
      tree.summaryArena.reserve(static_cast<int>(required));
      // At most one ancestor per depth for each pending molecule. Scratch is
      // bounded by batch size times tree height, not by total cluster count.
      const auto deltaCapacity = std::min(tree.totalNodes, static_cast<std::size_t>(pending) * hostControl[4]);
      if (deltaCounts.size() < deltaCapacity) {
        deltaCounts.resize(deltaCapacity);
        deltaSums.resize(deltaCapacity * numWords * 32);
      }
      if (deltaCapacity > 0) {
        cudaCheckError(cudaMemsetAsync(deltaSlots.data(), 0xff, hostControl[1] * sizeof(int), stream));
        cudaCheckError(cudaMemsetAsync(deltaCursor.data(), 0, sizeof(int), stream));
      }
      storage = tree.storage();
      if (fingerprintsOnHost) {
        ownedFingerprints.reserve(static_cast<int>(tree.totalEntries));
        storage.singletonFingerprintPages = ownedFingerprints.data();
        storage.queryBegin                = begin;
      }
      bitBirchSharedRouteKernel<<<count, cooperativeBlockSize, 0, stream>>>(begin,
                                                                            count,
                                                                            keys.data(),
                                                                            values.data(),
                                                                            groupKeys.data(),
                                                                            ownerStatuses.data(),
                                                                            groupStatuses.data(),
                                                                            useGroups,
                                                                            useGroups ? routingWidth : 1,
                                                                            threshold,
                                                                            storage);
      cudaCheckError(cudaGetLastError());
      if (useGroups) {
        cudaCheckError(cub::DeviceRadixSort::SortPairs(sortScratch.data(),
                                                       sortBytes,
                                                       groupKeys.data(),
                                                       sortedGroupKeys.data(),
                                                       values.data(),
                                                       sortedGroupValues.data(),
                                                       count,
                                                       0,
                                                       32,
                                                       stream));
        bitBirchSharedGroupsKernel<<<count, cooperativeBlockSize, 0, stream>>>(count,
                                                                               sortedGroupKeys.data(),
                                                                               sortedGroupValues.data(),
                                                                               threshold,
                                                                               proposalSums.data(),
                                                                               groupStatuses.data(),
                                                                               storage);
        cudaCheckError(cudaGetLastError());
      }
      cudaCheckError(cub::DeviceRadixSort::SortPairs(sortScratch.data(),
                                                     sortBytes,
                                                     keys.data(),
                                                     sortedKeys.data(),
                                                     values.data(),
                                                     sortedValues.data(),
                                                     count,
                                                     0,
                                                     32,
                                                     stream));
      bitBirchSharedLeafOwnersKernel<<<count, cooperativeBlockSize, 0, stream>>>(count,
                                                                                 sortedKeys.data(),
                                                                                 sortedValues.data(),
                                                                                 threshold,
                                                                                 branchingFactor,
                                                                                 dirtyNodes.data(),
                                                                                 ownerStatuses.data(),
                                                                                 storage);
      cudaCheckError(cudaGetLastError());
      for (int depth = hostControl[4]; depth > 0; --depth) {
        bitBirchSharedRefreshLevelKernel<<<hostControl[1], cooperativeBlockSize, 0, stream>>>(
          hostControl[1],
          depth,
          count,
          sortedKeys.data(),
          sortedValues.data(),
          dirtyNodes.data(),
          deltaSlots.data(),
          deltaCursor.data(),
          static_cast<int>(deltaCapacity),
          deltaSums.data(),
          deltaCounts.data(),
          nodeStatuses.data(),
          storage);
        cudaCheckError(cudaGetLastError());
      }
      bitBirchSharedSplitSeedsKernel<<<count, cooperativeBlockSize, splitCacheBytes, stream>>>(count,
                                                                                               sortedKeys.data(),
                                                                                               branchingFactor,
                                                                                               splitLeft.data(),
                                                                                               splitRight.data(),
                                                                                               splitAffinity.data(),
                                                                                               storage);
      cudaCheckError(cudaGetLastError());
      bitBirchSharedRepairKernel<<<1, cooperativeBlockSize, 0, stream>>>(begin,
                                                                         count,
                                                                         sortedKeys.data(),
                                                                         splitLeft.data(),
                                                                         splitRight.data(),
                                                                         splitAffinity.data(),
                                                                         ownerStatuses.data(),
                                                                         groupStatuses.data(),
                                                                         nodeStatuses.data(),
                                                                         branchingFactor,
                                                                         control.data(),
                                                                         storage);
      cudaCheckError(cudaGetLastError());
      control.copyToHost(hostControl);
      cudaCheckError(cudaStreamSynchronize(stream));
      if (hostControl[3] != static_cast<int>(BitBirchStatus::Success)) {
        throw std::runtime_error("BitBIRCH shared tree failure (status " + std::to_string(hostControl[3]) + ")");
      }
      if (hostControl[0] >= pending) {
        throw std::runtime_error("BitBIRCH shared insertion failed to make progress");
      }
      pending = hostControl[0];
    }
    begin += count;
    tree.summaryArena.rotate();
  }
  if (returnCentroids) {
    const int clusters = hostControl[5] - hostControl[1] + 1;
    result.centroids.resize(static_cast<std::size_t>(clusters) * numWords);
  }
  storage.centroids                  = returnCentroids ? result.centroids.data() : nullptr;
  // Entry owners retain their minimum input index. Sort only the K live leaf
  // entries by that index to preserve serial label numbering without an N-wide
  // first-member scan and its additional device workspace.
  const int              clusters    = hostControl[5] - hostControl[1] + 1;
  const int              liveEntries = hostControl[5];
  AsyncDeviceVector<int> firstMembers(clusters, stream);
  AsyncDeviceVector<int> leafEntries(clusters, stream);
  AsyncDeviceVector<int> sortedFirstMembers(clusters, stream);
  AsyncDeviceVector<int> sortedLeafEntries(clusters, stream);
  AsyncDeviceVector<int> leafCursor(1, stream);
  cudaCheckError(cudaMemsetAsync(leafCursor.data(), 0, sizeof(int), stream));
  const int entryBlocks = (liveEntries + cooperativeBlockSize - 1) / cooperativeBlockSize;
  bitBirchSharedCollectLeavesKernel<<<entryBlocks, cooperativeBlockSize, 0, stream>>>(liveEntries,
                                                                                      firstMembers.data(),
                                                                                      leafEntries.data(),
                                                                                      leafCursor.data(),
                                                                                      storage);
  cudaCheckError(cudaGetLastError());
  std::size_t clusterSortBytes = 0;
  cudaCheckError(cub::DeviceRadixSort::SortPairs(nullptr,
                                                 clusterSortBytes,
                                                 firstMembers.data(),
                                                 sortedFirstMembers.data(),
                                                 leafEntries.data(),
                                                 sortedLeafEntries.data(),
                                                 clusters,
                                                 0,
                                                 32,
                                                 stream));
  AsyncDeviceVector<std::byte> clusterSortScratch(clusterSortBytes, stream);
  cudaCheckError(cub::DeviceRadixSort::SortPairs(clusterSortScratch.data(),
                                                 clusterSortBytes,
                                                 firstMembers.data(),
                                                 sortedFirstMembers.data(),
                                                 leafEntries.data(),
                                                 sortedLeafEntries.data(),
                                                 clusters,
                                                 0,
                                                 32,
                                                 stream));
  const int clusterBlocks = (clusters + cooperativeBlockSize - 1) / cooperativeBlockSize;
  bitBirchSharedClusterMapKernel<<<clusterBlocks, cooperativeBlockSize, 0, stream>>>(clusters,
                                                                                     sortedLeafEntries.data(),
                                                                                     storage);
  cudaCheckError(cudaGetLastError());
  const int labelBlocks = (numFingerprints + cooperativeBlockSize - 1) / cooperativeBlockSize;
  bitBirchSharedFinalizeKernel<<<labelBlocks, cooperativeBlockSize, 0, stream>>>(numFingerprints, storage);
  cudaCheckError(cudaGetLastError());
  std::vector<int> clusterCounts(1);
  tree.clusterCounts.copyToHost(clusterCounts);
  cudaCheckError(cudaStreamSynchronize(stream));
  result.numClusters = clusterCounts[0];
  return result;
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
  BitBirchStatus         hostStatus{};
  constexpr int          buildBatchSize               = 256;
  constexpr int          reservedSummarySlotsPerInput = 8;
  int                    hostSummaryCursor            = 0;
  for (int begin = 0; begin < numFingerprints; begin += buildBatchSize) {
    const int  end               = std::min(begin + buildBatchSize, numFingerprints);
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
  std::vector<BitBirchStatus>   hostPartialStatuses(numPartitions);
  std::vector<int>              hostPartialClusterCounts(numPartitions);
  int                           totalPartialClusters = 0;
  {
    const ScopedNvtxRange partialRange("BitBIRCH partial-tree construction");
    constexpr int         reservedSummarySlotsPerInput = 8;
    int                   hostPartialSummaryCursor     = 0;
    for (int partitionOffset = 0; partitionOffset < partitionSize; partitionOffset += summaryBuildBatchSize) {
      int batchInputs = 0;
      for (int partition = 0; partition < numPartitions; ++partition) {
        const int begin = partition * partitionSize + partitionOffset;
        const int end =
          std::min(std::min(begin + summaryBuildBatchSize, (partition + 1) * partitionSize), numFingerprints);
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
                                                                                     summaryBuildBatchSize,
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
  }

  constexpr int scalableForestTreeThreshold = 32;
  // Pairwise reconciliation bounds per-tree work when partial construction
  // reveals that at least half of the inputs remain as distinct summaries.
  const bool    sparseScalingPath           = numPartitions >= scalableForestTreeThreshold &&
                                 static_cast<std::int64_t>(totalPartialClusters) * 2 >= numFingerprints;
  const int mergeFanIn = sparseScalingPath ? 2 : 4;

  auto current = mergeForestRound<Component, PartialComponent>(fingerprints.data(),
                                                               numFingerprints,
                                                               numWords,
                                                               threshold,
                                                               branchingFactor,
                                                               mergeFanIn,
                                                               numPartitions,
                                                               partitionSize,
                                                               entryStride,
                                                               totalEntries,
                                                               hostPartialClusterCounts,
                                                               partialStorage,
                                                               result.clusterIds.data(),
                                                               stream);

  partialNodeHeads               = AsyncDeviceVector<int>();
  partialNodeSizes               = AsyncDeviceVector<int>();
  partialNodeParents             = AsyncDeviceVector<int>();
  partialNodeLeaves              = AsyncDeviceVector<std::uint8_t>();
  partialEntryNext               = AsyncDeviceVector<int>();
  partialEntryChildren           = AsyncDeviceVector<int>();
  partialEntryCounts             = AsyncDeviceVector<std::uint32_t>();
  partialEntrySummarySlots       = AsyncDeviceVector<int>();
  partialEntryFingerprintIndices = AsyncDeviceVector<int>();
  partialEntryClusterIds         = AsyncDeviceVector<int>();
  partialRoots                   = AsyncDeviceVector<int>();
  partialNodeCursors             = AsyncDeviceVector<int>();
  partialEntryCursors            = AsyncDeviceVector<int>();
  partialClusterCounts           = AsyncDeviceVector<int>();
  partialStatuses                = AsyncDeviceVector<BitBirchStatus>();
  partialSummaryArena.clear();

  int        previousClusters = totalPartialClusters;
  const bool stopSparseHierarchy =
    sparseScalingPath && static_cast<std::int64_t>(current->totalClusters) * 2 >= numFingerprints;
  while (current->numTrees > 1 && current->totalClusters < previousClusters && !stopSparseHierarchy &&
         hierarchyRoundFitsWorkBudget(current->numTrees, current->totalClusters, mergeFanIn)) {
    previousClusters = current->totalClusters;
    auto next        = mergeForestRound<Component, Component>(fingerprints.data(),
                                                       numFingerprints,
                                                       numWords,
                                                       threshold,
                                                       branchingFactor,
                                                       mergeFanIn,
                                                       current->numTrees,
                                                       current->partitionSize,
                                                       current->entryStride,
                                                       current->totalEntries,
                                                       current->hostClusterCounts,
                                                       current->storage(),
                                                       result.clusterIds.data(),
                                                       stream);
    current          = std::move(next);
  }

  {
    const ScopedNvtxRange finalizeRange("BitBIRCH forest finalization");
    finalizeForest(numFingerprints,
                   current->partitionSize,
                   current->nodeStride,
                   current->entryStride,
                   current->hostClusterCounts,
                   returnCentroids ? result.centroids.data() : nullptr,
                   current->storage(),
                   stream);
  }
  result.numClusters = current->totalClusters;
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

BitBirchResult bitBirchSharedGpu(const cuda::std::span<const std::uint32_t> fingerprints,
                                 const int                                  numFingerprints,
                                 const int                                  numWords,
                                 const double                               threshold,
                                 const int                                  branchingFactor,
                                 const int                                  insertionBatchSize,
                                 const bool                                 filteredGroups,
                                 const int                                  orderedPrefixSize,
                                 const int                                  routingWidth,
                                 const std::size_t                          summaryCacheBytes,
                                 const bool                                 fingerprintsOnHost,
                                 const bool                                 clusterIdsOnHost,
                                 const bool                                 returnCentroids,
                                 const cudaStream_t                         stream) {
  const ScopedNvtxRange range("BitBIRCH shared tree");
  if (numFingerprints < 0 || numFingerprints > std::numeric_limits<int>::max() - cooperativeBlockSize ||
      numWords <= 0 || numWords > std::numeric_limits<int>::max() / 32 ||
      fingerprints.size() != static_cast<std::size_t>(numFingerprints) * numWords) {
    throw std::invalid_argument("BitBIRCH fingerprints shape or dimensions are invalid");
  }
  if (!std::isfinite(threshold) || threshold < 0 || threshold > 1 || branchingFactor < 3 || insertionBatchSize < 1 ||
      orderedPrefixSize < 0 || (routingWidth != 1 && routingWidth != 2) || (!filteredGroups && routingWidth != 1)) {
    throw std::invalid_argument("BitBIRCH shared clustering options are invalid");
  }
  const auto minimumNodeSize = branchingFactor / 2 + branchingFactor % 2;
  const auto maximumEntries  = static_cast<std::int64_t>(numFingerprints) + numFingerprints / (minimumNodeSize - 1) + 8;
  if (maximumEntries > std::numeric_limits<int>::max() - summaryEntriesPerPage) {
    throw std::invalid_argument("BitBIRCH shared metadata exceeds supported index capacity");
  }
  if (numFingerprints == 0) {
    BitBirchResult result{AsyncDeviceVector<int>(0, stream), AsyncDeviceVector<std::uint32_t>(0, stream), 0, numWords};
    result.clusterIdsOnHost = clusterIdsOnHost;
    return result;
  }
  if (numFingerprints <= std::numeric_limits<std::uint16_t>::max()) {
    return launchShared<std::uint16_t>(fingerprints,
                                       numFingerprints,
                                       numWords,
                                       threshold,
                                       branchingFactor,
                                       insertionBatchSize,
                                       filteredGroups,
                                       orderedPrefixSize,
                                       routingWidth,
                                       summaryCacheBytes,
                                       fingerprintsOnHost,
                                       clusterIdsOnHost,
                                       returnCentroids,
                                       stream);
  }
  return launchShared<std::uint32_t>(fingerprints,
                                     numFingerprints,
                                     numWords,
                                     threshold,
                                     branchingFactor,
                                     insertionBatchSize,
                                     filteredGroups,
                                     orderedPrefixSize,
                                     routingWidth,
                                     summaryCacheBytes,
                                     fingerprintsOnHost,
                                     clusterIdsOnHost,
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
