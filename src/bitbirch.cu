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

constexpr int summaryEntriesPerPage = 4096;
constexpr int orderedWarmupSize     = 16384;

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
  // service order, or the clustering result.
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
  PagedFingerprintArena(const int words, const cudaStream_t stream, const std::size_t cacheBytes = 0)
      : words_(words),
        stream_(stream),
        cachePages_(cacheBytes / (sizeof(std::uint32_t) * static_cast<std::size_t>(summaryEntriesPerPage) * words)) {
    if (cacheBytes > 0 && cachePages_ == 0) {
      throw std::invalid_argument("fingerprint_cache_bytes must fit at least one packed-fingerprint page");
    }
    pointers_.setStream(stream);
    pageHits_.setStream(stream);
  }

  ~PagedFingerprintArena() {
    if (cachePages_ > 0) {
      cudaStreamSynchronize(stream_);
    }
  }

  void reserve(const int entries) {
    const auto required     = (static_cast<std::size_t>(entries) + summaryEntriesPerPage - 1) / summaryEntriesPerPage;
    const auto oldPages     = pageToCache_.size();
    const auto currentPages = cachePages_ == 0 ? devicePages_.size() : hostPages_.size();
    if (required <= currentPages) {
      return;
    }
    while ((cachePages_ == 0 ? devicePages_.size() : hostPages_.size()) < required) {
      if (cachePages_ == 0) {
        devicePages_.emplace_back(pageElements(), stream_);
      } else {
        hostPages_.emplace_back(pageElements(), std::uint32_t{0});
        std::uint32_t* mapped = nullptr;
        cudaCheckError(cudaHostGetDevicePointer(&mapped, hostPages_.back().data(), 0));
        mappedHostPages_.push_back(mapped);
        pageToCache_.push_back(-1);
        pageScores_.push_back(0);
        if (devicePages_.size() < cachePages_) {
          const int slot = static_cast<int>(devicePages_.size());
          const int page = static_cast<int>(hostPages_.size() - 1);
          devicePages_.emplace_back(pageElements(), stream_);
          pageToCache_[page] = slot;
          cacheToPage_.push_back(page);
          cudaCheckError(cudaMemcpyAsync(devicePages_.back().data(),
                                         hostPages_.back().data(),
                                         pageBytes(),
                                         cudaMemcpyHostToDevice,
                                         stream_));
        }
      }
    }
    if (cachePages_ > 0) {
      pageHits_.resize(required);
      cudaCheckError(
        cudaMemsetAsync(pageHits_.data() + oldPages, 0, (required - oldPages) * sizeof(std::uint32_t), stream_));
    }
    updatePointers();
  }

  void rotate() {
    if (cachePages_ == 0 || hostPages_.size() <= cachePages_) {
      return;
    }
    std::vector<std::uint32_t> hits(hostPages_.size());
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
    order.resize(devicePages_.size());
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
      cudaCheckError(cudaMemcpyAsync(hostPages_[oldPage].data(),
                                     devicePages_[slot].data(),
                                     pageBytes(),
                                     cudaMemcpyDeviceToHost,
                                     stream_));
      cudaCheckError(cudaMemcpyAsync(devicePages_[slot].data(),
                                     hostPages_[page].data(),
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

  std::uint32_t* pageHits() const noexcept {
    return cachePages_ > 0 && hostPages_.size() > cachePages_ ? pageHits_.data() : nullptr;
  }
  std::uint32_t** data() const noexcept { return pointers_.data(); }

 private:
  std::size_t pageElements() const noexcept { return static_cast<std::size_t>(summaryEntriesPerPage) * words_; }
  std::size_t pageBytes() const noexcept { return pageElements() * sizeof(std::uint32_t); }

  void updatePointers() {
    std::vector<std::uint32_t*> pointers;
    const auto                  pages = cachePages_ == 0 ? devicePages_.size() : hostPages_.size();
    pointers.reserve(pages);
    for (std::size_t page = 0; page < pages; ++page) {
      if (cachePages_ == 0) {
        pointers.push_back(devicePages_[page].data());
      } else {
        const int slot = pageToCache_[page];
        pointers.push_back(slot >= 0 ? devicePages_[slot].data() : mappedHostPages_[page]);
      }
    }
    pointers_.setFromVector(pointers);
    // Keep the temporary host pointer table alive until its upload completes.
    cudaCheckError(cudaStreamSynchronize(stream_));
  }

 private:
  int                                           words_;
  cudaStream_t                                  stream_;
  std::size_t                                   cachePages_;
  std::vector<AsyncDeviceVector<std::uint32_t>> devicePages_;
  std::vector<PinnedHostVector<std::uint32_t>>  hostPages_;
  std::vector<std::uint32_t*>                   mappedHostPages_;
  std::vector<int>                              pageToCache_;
  std::vector<int>                              cacheToPage_;
  std::vector<std::uint64_t>                    pageScores_;
  AsyncDeviceVector<std::uint32_t>              pageHits_;
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
  std::uint32_t*       singletonPageHits         = nullptr;
  int                  queryBegin                = 0;
};

template <typename Component> class TreeWorkspace {
 public:
  TreeWorkspace(const std::uint32_t* fingerprints,
                int*                 labels,
                const int            nodeCapacity,
                const int            entryCapacity,
                const int            numWords,
                const cudaStream_t   stream,
                const std::size_t    summaryCacheBytes)
      : nodeCapacity(nodeCapacity),
        entryCapacity(entryCapacity),
        nodeHeads(nodeCapacity, stream),
        nodeSizes(nodeCapacity, stream),
        nodeParents(nodeCapacity, stream),
        nodeLeaves(nodeCapacity, stream),
        entryNext(entryCapacity, stream),
        entryChildren(entryCapacity, stream),
        entryCounts(entryCapacity, stream),
        entrySummarySlots(entryCapacity, stream),
        entryFingerprintIndices(entryCapacity, stream),
        entryClusterIds(entryCapacity, stream),
        root(-1, stream),
        nodeCursor(0, stream),
        entryCursor(0, stream),
        summaryCursor(0, stream),
        clusterCount(0, stream),
        status(BitBirchStatus::Success, stream),
        summaryArena(numWords * 32, numWords, stream, summaryCacheBytes),
        fingerprints(fingerprints),
        labels(labels),
        numWords(numWords) {
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
            root.data(),
            nodeCursor.data(),
            entryCursor.data(),
            summaryCursor.data(),
            clusterCount.data(),
            status.data(),
            nodeCapacity,
            entryCapacity,
            summaryArena.capacity(),
            numWords,
            numWords * 32,
            summaryArena.pageHits()};
  }

  void grow(const int nodes, const int entries) {
    if (nodes > nodeCapacity) {
      nodeHeads.resize(nodes);
      nodeSizes.resize(nodes);
      nodeParents.resize(nodes);
      nodeLeaves.resize(nodes);
      nodeCapacity = nodes;
    }
    if (entries > entryCapacity) {
      entryNext.resize(entries);
      entryChildren.resize(entries);
      entryCounts.resize(entries);
      entrySummarySlots.resize(entries);
      entryFingerprintIndices.resize(entries);
      entryClusterIds.resize(entries);
      entryCapacity = entries;
    }
  }

  int                              nodeCapacity;
  int                              entryCapacity;
  AsyncDeviceVector<int>           nodeHeads;
  AsyncDeviceVector<int>           nodeSizes;
  AsyncDeviceVector<int>           nodeParents;
  AsyncDeviceVector<std::uint8_t>  nodeLeaves;
  AsyncDeviceVector<int>           entryNext;
  AsyncDeviceVector<int>           entryChildren;
  AsyncDeviceVector<std::uint32_t> entryCounts;
  AsyncDeviceVector<int>           entrySummarySlots;
  AsyncDeviceVector<int>           entryFingerprintIndices;
  AsyncDeviceVector<int>           entryClusterIds;
  AsyncDevicePtr<int>              root;
  AsyncDevicePtr<int>              nodeCursor;
  AsyncDevicePtr<int>              entryCursor;
  AsyncDevicePtr<int>              summaryCursor;
  AsyncDevicePtr<int>              clusterCount;
  AsyncDevicePtr<BitBirchStatus>   status;
  PagedSummaryArena<Component>     summaryArena;
  const std::uint32_t*             fingerprints;
  int*                             labels;
  int                              numWords;
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
    if (storage.singletonPageHits != nullptr && word == 0 && (threadIdx.x & 31) == 0) {
      atomicAdd(storage.singletonPageHits + index / summaryEntriesPerPage, 1U);
    }
    return storage
      .singletonFingerprintPages[index / summaryEntriesPerPage]
                                [static_cast<std::size_t>(index % summaryEntriesPerPage) * storage.numWords + word];
  }
  return storage.fingerprints[static_cast<std::size_t>(index) * storage.numWords + word];
}

template <typename Component>
__device__ __forceinline__ const std::uint32_t* singletonFingerprint(const TreeStorage<Component>& storage,
                                                                     const int                     index) {
  if (storage.singletonFingerprintPages != nullptr) {
    if (storage.singletonPageHits != nullptr && (threadIdx.x & 31) == 0) {
      atomicAdd(storage.singletonPageHits + index / summaryEntriesPerPage, 1U);
    }
    return storage.singletonFingerprintPages[index / summaryEntriesPerPage] +
           static_cast<std::size_t>(index % summaryEntriesPerPage) * storage.numWords;
  }
  return storage.fingerprints + static_cast<std::size_t>(index) * storage.numWords;
}

template <typename Component>
__device__ __forceinline__ const std::uint32_t* centroidFingerprint(const TreeStorage<Component>& storage,
                                                                    const int                     entry) {
  const int fingerprintIndex = storage.entryFingerprintIndices[entry];
  if (fingerprintIndex >= 0) {
    return singletonFingerprint(storage, fingerprintIndex);
  }
  const int slot = storage.entrySummarySlots[entry];
  return storage.centroidPages[slot / summaryEntriesPerPage] +
         static_cast<std::size_t>(slot % summaryEntriesPerPage) * storage.numWords;
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
  int         intersection = 0;
  int         unionCount   = 0;
  const auto* centroid     = centroidFingerprint(storage, entry);
  if (storage.numWords % 4 == 0) {
    const auto* centroidVectors    = reinterpret_cast<const uint4*>(centroid);
    const auto* fingerprintVectors = reinterpret_cast<const uint4*>(fingerprint);
    for (int vector = 0; vector < storage.numWords / 4; ++vector) {
      const uint4 centroidWords    = centroidVectors[vector];
      const uint4 fingerprintWords = fingerprintVectors[vector];
      intersection += __popc(centroidWords.x & fingerprintWords.x);
      intersection += __popc(centroidWords.y & fingerprintWords.y);
      intersection += __popc(centroidWords.z & fingerprintWords.z);
      intersection += __popc(centroidWords.w & fingerprintWords.w);
      unionCount += __popc(centroidWords.x | fingerprintWords.x);
      unionCount += __popc(centroidWords.y | fingerprintWords.y);
      unionCount += __popc(centroidWords.z | fingerprintWords.z);
      unionCount += __popc(centroidWords.w | fingerprintWords.w);
    }
    return routingSimilarity(intersection, unionCount, storage.numBits);
  }
  for (int word = 0; word < storage.numWords; ++word) {
    intersection += __popc(centroid[word] & fingerprint[word]);
    unionCount += __popc(centroid[word] | fingerprint[word]);
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
                                                  const std::int8_t*      splitAffinity = nullptr) {
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
  appendEntry(storage, parent, siblingEntry);
  return parent;
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
  int       stagedCount;
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
  const int  numWarps      = (blockDim.x + warpSize - 1) / warpSize;
  const int  numChunks     = (storage.numBits + blockDim.x - 1) / blockDim.x;
  // Stage every warp partial and sum them once. The chunk-major, warp-minor
  // order matches per-chunk block reductions, so the FP64 result is identical.
  const bool stagePartials = numChunks * numWarps <= cooperativeBlockSize;
  for (int chunk = 0; chunk < numChunks; ++chunk) {
    const int bit         = chunk * blockDim.x + threadIdx.x;
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
    if (!stagePartials) {
      cooperativeAccumulateISimTerms(commonPairs, mismatches, scratch);
      continue;
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      commonPairs += __shfl_down_sync(0xffffffffU, commonPairs, offset);
      mismatches += __shfl_down_sync(0xffffffffU, mismatches, offset);
    }
    if (threadIdx.x % warpSize == 0) {
      scratch.values[chunk * numWarps + threadIdx.x / warpSize]      = commonPairs;
      scratch.otherValues[chunk * numWarps + threadIdx.x / warpSize] = mismatches;
    }
  }
  if (stagePartials) {
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int index = 0; index < numChunks * numWarps; ++index) {
        scratch.accumulatedValue += scratch.values[index];
        scratch.accumulatedOtherValue += scratch.otherValues[index];
      }
    }
    __syncthreads();
  }
  return {scratch.accumulatedValue, scratch.accumulatedOtherValue};
}

template <typename Component>
__device__ __forceinline__ void cooperativeRefreshCentroid(TreeStorage<Component>& storage, const int entry) {
  if (storage.entryFingerprintIndices[entry] >= 0) {
    __syncthreads();
    return;
  }
  const auto count    = static_cast<std::uint64_t>(storage.entryCounts[entry]);
  const int  slot     = storage.entrySummarySlots[entry];
  auto*      centroid = storage.centroidPages[slot / summaryEntriesPerPage] +
                   static_cast<std::size_t>(slot % summaryEntriesPerPage) * storage.numWords;
  // One thread per bit keeps sum reads coalesced; each warp covers one word.
  for (int base = 0; base < storage.numBits; base += blockDim.x) {
    const int  bit = base + threadIdx.x;
    const bool set = bit < storage.numBits &&
                     bitbirch::majorityCentroidBit(static_cast<std::uint64_t>(materializedLinearSum(storage, entry, bit)),
                                                   count);
    const std::uint32_t word = __ballot_sync(0xffffffffU, set);
    if (threadIdx.x % warpSize == 0 && bit < storage.numBits) {
      centroid[bit / 32] = word;
    }
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
__device__ __forceinline__ bool cooperativeAddFingerprint(TreeStorage<Component>& storage,
                                                          const int               entry,
                                                          const std::uint32_t*    fingerprint,
                                                          CooperativeScratch&     scratch) {
  if (!cooperativeMaterializeEntry(storage, entry, scratch)) {
    return false;
  }
  if (threadIdx.x == 0) {
    ++storage.entryCounts[entry];
  }
  for (int bit = threadIdx.x; bit < storage.numBits; bit += blockDim.x) {
    materializedLinearSum(storage, entry, bit) += static_cast<Component>((fingerprint[bit / 32] >> (bit % 32)) & 1U);
  }
  __syncthreads();
  cooperativeRefreshCentroid(storage, entry);
  return true;
}

template <typename Component>
__device__ __forceinline__ void cooperativeSummarizeNode(TreeStorage<Component>& storage,
                                                         const int               node,
                                                         const int               targetEntry,
                                                         CooperativeScratch&     scratch) {
  if (!cooperativeMaterializeEntry(storage, targetEntry, scratch)) {
    return;
  }
  // Walk the entry list once and stage each entry's storage location. Every
  // thread then sums its bits from shared memory instead of re-chasing links.
  if (threadIdx.x == 0) {
    std::uint32_t count   = 0;
    int           entries = 0;
    for (int entry = storage.nodeHeads[node]; entry >= 0; entry = storage.entryNext[entry], ++entries) {
      count += storage.entryCounts[entry];
      if (entries < cooperativeBlockSize) {
        scratch.entries[entries]      = storage.entryFingerprintIndices[entry];
        scratch.otherEntries[entries] = storage.entrySummarySlots[entry];
      }
    }
    storage.entryCounts[targetEntry] = count;
    scratch.stagedCount        = entries;
  }
  __syncthreads();
  const int numEntries = scratch.stagedCount;
  for (int bit = threadIdx.x; bit < storage.numBits; bit += blockDim.x) {
    Component sum = 0;
    if (numEntries <= cooperativeBlockSize) {
      for (int index = 0; index < numEntries; ++index) {
        const int fingerprintIndex = scratch.entries[index];
        if (fingerprintIndex >= 0) {
          sum += static_cast<Component>((singletonWord(storage, fingerprintIndex, bit / 32) >> (bit % 32)) & 1U);
        } else {
          const int slot = scratch.otherEntries[index];
          sum += storage.linearSumPages[slot / summaryEntriesPerPage]
                                       [static_cast<std::size_t>(slot % summaryEntriesPerPage) * storage.numBits + bit];
        }
      }
    } else {
      for (int entry = storage.nodeHeads[node]; entry >= 0; entry = storage.entryNext[entry]) {
        sum += linearSum(storage, entry, bit);
      }
    }
    materializedLinearSum(storage, targetEntry, bit) = sum;
  }
  __syncthreads();
  cooperativeRefreshCentroid(storage, targetEntry);
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
                                                    CooperativeScratch&     scratch) {
  while (true) {
    // Lane zero otherwise reaches topology writes before a lagging warp has
    // even evaluated the loop predicate.
    const bool overflow = storage.nodeSizes[node] > branchingFactor;
    __syncthreads();
    if (!overflow) {
      break;
    }
    cooperativeFindSplitSeeds(storage, node, scratch);
    const int lhsSeed = scratch.bestEntry;
    const int rhsSeed = scratch.selectedEntry;
    __syncthreads();
    if (threadIdx.x == 0) {
      const int oldParent = storage.nodeParents[node];
      scratch.sourceEntry = *storage.nodeCursor;  // The next allocation is the sibling node.
      scratch.node        = splitNodeWithSeeds(storage, node, branchingFactor, lhsSeed, rhsSeed);
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
    node = parent;
  }
  return node;
}

// Insertion epochs have four disjoint phases: route, leaf ownership,
// bottom-up summary refresh, and structural repair. No leaf owner writes an
// ancestor BF or topology belonging to another owner.
template <typename Component> __global__ void bitBirchInitializeKernel(TreeStorage<Component> storage) {
  if (threadIdx.x == 0) {
    *storage.nodeCursor  = 0;
    *storage.entryCursor = 0;
    *storage.numClusters = 0;
    *storage.status      = BitBirchStatus::Success;
    *storage.root        = allocateNode(storage, true, -1);
  }
}

template <typename Component>
__global__ void bitBirchRouteKernel(const int                    begin,
                                    const int                    count,
                                    int*                         keys,
                                    int*                         values,
                                    int*                         groupKeys,
                                    const bool                   enableGroups,
                                    const double                 threshold,
                                    const TreeStorage<Component> storage) {
  // One cooperative block per query; all tree reads precede any leaf writes.
  __shared__ CooperativeScratch scratch;
  const int                     offset = blockIdx.x;
  if (offset >= count) {
    return;
  }
  const int molecule = begin + offset;
  if (threadIdx.x == 0) {
    values[offset]    = molecule;
    scratch.node      = *storage.root;
    keys[offset]      = INT_MAX;
    groupKeys[offset] = INT_MAX;
  }
  __syncthreads();
  if (storage.labels[molecule] >= 0) {
    return;
  }
  const auto* fingerprint = queryFingerprint(storage, molecule);
  while (!storage.nodeLeaves[scratch.node]) {
    const int selected = cooperativeClosestEntry(storage, scratch.node, fingerprint, scratch);
    if (threadIdx.x == 0) {
      scratch.node = storage.entryChildren[selected];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    keys[offset] = scratch.node;
  }
  if (enableGroups && storage.nodeHeads[scratch.node] >= 0) {
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
__global__ void bitBirchGroupsKernel(const int                    count,
                                     const int*                   keys,
                                     const int*                   values,
                                     const double                 threshold,
                                     Component*                   proposalSums,
                                     TreeStorage<Component>       storage) {
  const int first = blockIdx.x;
  if (first >= count || keys[first] == INT_MAX || (first > 0 && keys[first - 1] == keys[first])) {
    return;
  }
  __shared__ CooperativeScratch scratch;
  const int                     entry = keys[first];
  if (threadIdx.x == 0) {
    int end = first + 1;
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
      if (!cooperativeAddFingerprint(storage, entry, fingerprint, scratch)) {
        return;
      }
      if (threadIdx.x == 0) {
        storage.labels[molecule]       = entry;
        storage.entryClusterIds[entry] = min(storage.entryClusterIds[entry], molecule);
      }
      __syncthreads();
    }
  }
}

template <typename Component>
__device__ __forceinline__ int allocateConcurrentLeafEntry(TreeStorage<Component>& storage) {
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
__global__ void bitBirchLeafOwnersKernel(const int                    count,
                                         const int*                   keys,
                                         const int*                   values,
                                         const double                 threshold,
                                         const int                    branchingFactor,
                                         const int                    leafDepth,
                                         int*                         dirtyNodes,
                                         int*                         levelNodes,
                                         int*                         levelCounts,
                                         const int                    levelCapacity,
                                         TreeStorage<Component>       storage) {
  const int first = blockIdx.x;
  if (first >= count || keys[first] == INT_MAX || (first > 0 && keys[first - 1] == keys[first])) {
    return;
  }
  __shared__ CooperativeScratch scratch;
  const int                     node = keys[first];
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
      if (!cooperativeAddFingerprint(storage, selected, fingerprint, scratch)) {
        return;
      }
    } else {
      if (threadIdx.x == 0) {
        scratch.selectedEntry = allocateConcurrentLeafEntry(storage);
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
  // Every leaf has the same depth, so each ancestor's level is known here. The
  // first owner to dirty a node lists it, keeping refresh work proportional to
  // the touched paths instead of the whole tree.
  if (threadIdx.x == 0) {
    int level = leafDepth;
    for (int changed = node; storage.nodeParents[changed] >= 0; changed = storage.nodeParents[changed], --level) {
      if (atomicExch(dirtyNodes + changed, 1) == 0) {
        levelNodes[level * levelCapacity + atomicAdd(levelCounts + level, 1)] = changed;
      }
    }
  }
}

template <typename Component>
__global__ void bitBirchRefreshLevelKernel(const int*             levelNodes,
                                           const int*             levelCount,
                                           const int              count,
                                           const int*             keys,
                                           const int*             values,
                                           int*                   dirtyNodes,
                                           int*                   deltaSlots,
                                           int*                   deltaCursor,
                                           const int              deltaCapacity,
                                           Component*             deltaSums,
                                           std::uint32_t*         deltaCounts,
                                           TreeStorage<Component> storage) {
  if (blockIdx.x >= *levelCount) {
    return;
  }
  const int                     node = levelNodes[blockIdx.x];
  __shared__ CooperativeScratch scratch;
  if (threadIdx.x == 0) {
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
        // Stage accepted molecules once; labels may live in mapped host memory.
        scratch.next = first;
        while (end < count && keys[end] == node) {
          if (storage.labels[values[end]] >= 0) {
            if (addedCount < cooperativeBlockSize) {
              scratch.entries[addedCount] = values[end];
            }
            ++addedCount;
          }
          ++end;
        }
        scratch.count       = end;
        scratch.stagedCount = static_cast<int>(addedCount);
      } else {
        // Stage the refreshed children's delta slots for the per-bit sum.
        int dirtyChildren = 0;
        for (int entry = storage.nodeHeads[node]; entry >= 0; entry = storage.entryNext[entry]) {
          const int slot = deltaSlots[storage.entryChildren[entry]];
          if (slot >= 0) {
            addedCount += deltaCounts[slot];
            if (dirtyChildren < cooperativeBlockSize) {
              scratch.entries[dirtyChildren] = slot;
            }
            ++dirtyChildren;
          }
        }
        scratch.stagedCount = dirtyChildren;
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
  const bool leaf   = storage.nodeLeaves[node];
  const bool staged = scratch.stagedCount <= cooperativeBlockSize;
  for (int bit = threadIdx.x; bit < storage.numBits; bit += blockDim.x) {
    Component added = 0;
    if (leaf && staged) {
      for (int index = 0; index < scratch.stagedCount; ++index) {
        const auto word = queryFingerprint(storage, scratch.entries[index])[bit / 32];
        added += static_cast<Component>((word >> (bit % 32)) & 1U);
      }
    } else if (leaf) {
      for (int offset = scratch.next; offset < scratch.count; ++offset) {
        const int molecule = values[offset];
        if (storage.labels[molecule] >= 0) {
          const auto word = queryFingerprint(storage, molecule)[bit / 32];
          added += static_cast<Component>((word >> (bit % 32)) & 1U);
        }
      }
    } else if (staged) {
      for (int index = 0; index < scratch.stagedCount; ++index) {
        added += deltaSums[static_cast<std::size_t>(scratch.entries[index]) * storage.numBits + bit];
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
__global__ void bitBirchSplitSeedsKernel(const int                    count,
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
__global__ void bitBirchRepairKernel(const int              count,
                                     const int*             keys,
                                     const int*             splitLeft,
                                     const int*             splitRight,
                                     const std::int8_t*     splitAffinity,
                                     const int              branchingFactor,
                                     const int              leafDepth,
                                     const int*             levelNodes,
                                     const int*             levelCounts,
                                     const int              levelCapacity,
                                     int*                   deltaSlots,
                                     int*                   summaryNodes,
                                     int*                   summaryEntries,
                                     int*                   summaryCount,
                                     TreeStorage<Component> storage) {
  __shared__ CooperativeScratch scratch;
  __shared__ int                deferred;
  // Refresh consumed this epoch's delta slots; restore them for the next one.
  for (int level = 1; level <= leafDepth; ++level) {
    for (int index = threadIdx.x; index < levelCounts[level]; index += blockDim.x) {
      deltaSlots[levelNodes[level * levelCapacity + index]] = -1;
    }
  }
  if (threadIdx.x == 0) {
    deferred = 0;
  }
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
    // Leaf topology changes stay ordered, but their two child BFs are only
    // read again by a cascading parent split. Defer them to a parallel pass.
    // Payload refresh already made every ancestor BF current, and a split
    // only redistributes members, so no ancestor above the parent changes.
    if (threadIdx.x == 0) {
      const int sibling = *storage.nodeCursor;
      const int parent =
        splitNodeWithSeeds(storage, node, branchingFactor, splitLeft[index], splitRight[index], splitAffinity);
      const int nodeEntry    = parent >= 0 ? parentEntry(storage, node) : -1;
      const int siblingEntry = parent >= 0 ? parentEntry(storage, sibling) : -1;
      if (parent >= 0 && (nodeEntry < 0 || siblingEntry < 0)) {
        *storage.status = BitBirchStatus::InvalidTree;
      }
      if (nodeEntry >= 0 && siblingEntry >= 0) {
        summaryNodes[deferred]       = node;
        summaryEntries[deferred]     = nodeEntry;
        summaryNodes[deferred + 1]   = sibling;
        summaryEntries[deferred + 1] = siblingEntry;
        deferred += 2;
      }
      scratch.node    = parent;
      scratch.success = nodeEntry >= 0 && siblingEntry >= 0 && storage.nodeSizes[parent] > branchingFactor;
    }
    __syncthreads();
    if (scratch.success) {
      const int parent = scratch.node;
      for (int index = 0; index < deferred; ++index) {
        cooperativeSummarizeNode(storage, summaryNodes[index], summaryEntries[index], scratch);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        deferred = 0;
      }
      cooperativeSplitNode(storage, parent, branchingFactor, scratch);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *summaryCount = deferred;
  }
}

template <typename Component>
__global__ void bitBirchSummarizeKernel(const int*             summaryNodes,
                                        const int*             summaryEntries,
                                        const int*             summaryCount,
                                        TreeStorage<Component> storage) {
  if (blockIdx.x >= *summaryCount) {
    return;
  }
  __shared__ CooperativeScratch scratch;
  cooperativeSummarizeNode(storage, summaryNodes[blockIdx.x], summaryEntries[blockIdx.x], scratch);
}

template <typename Component>
__global__ void bitBirchControlKernel(const int begin, const int count, int* control, TreeStorage<Component> storage) {
  __shared__ int pending;
  if (threadIdx.x == 0) {
    pending = 0;
  }
  __syncthreads();
  int localPending = 0;
  for (int molecule = begin + threadIdx.x; molecule < begin + count; molecule += blockDim.x) {
    localPending += storage.labels[molecule] < 0;
  }
  atomicAdd(&pending, localPending);
  __syncthreads();
  if (threadIdx.x == 0) {
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
__global__ void bitBirchCollectLeavesKernel(const int                    count,
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
__global__ void bitBirchClusterMapKernel(const int count, const int* leafEntries, TreeStorage<Component> storage) {
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

template <typename Component> __global__ void bitBirchFinalizeKernel(const int count, TreeStorage<Component> storage) {
  const int molecule = blockIdx.x * blockDim.x + threadIdx.x;
  if (molecule < count) {
    storage.labels[molecule] = storage.entryClusterIds[storage.labels[molecule]];
  }
}

template <typename Component>
BitBirchResult launchBitBirch(const cuda::std::span<const std::uint32_t> fingerprints,
                              const int                                  numFingerprints,
                              const int                                  numWords,
                              const double                               threshold,
                              const int                                  branchingFactor,
                              const int                                  batchSize,
                              const std::size_t                          summaryCacheBytes,
                              const std::size_t                          fingerprintCacheBytes,
                              const bool                                 fingerprintsOnHost,
                              const bool                                 clusterIdsOnHost,
                              const bool                                 returnCentroids,
                              const cudaStream_t                         stream) {
  const int         batchCapacity   = std::min(batchSize, numFingerprints);
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
  PagedFingerprintArena    ownedFingerprints(numWords, stream, fingerprintCacheBytes);
  // Balanced splits leave every non-root node with at least m entries. With
  // K leaf entries and V nodes, the tree has K + V - 1 total entries, hence
  // (m - 1)*(V - 1) <= K <= N. Padding also covers transient split allocations.
  // For B=254 this needs about N/126 nodes and 1.008*N entries, not 2*N/3*N.
  const int                minimumNodeSize = branchingFactor / 2 + branchingFactor % 2;
  const int                maxNodes        = numFingerprints / (minimumNodeSize - 1) + 8;
  const int                maxEntries      = numFingerprints + maxNodes;
  const int                initialNodes    = std::min(maxNodes, batchCapacity / (minimumNodeSize - 1) + 8);
  const int                initialEntries  = std::min(maxEntries, batchCapacity + initialNodes);
  TreeWorkspace<Component> tree(fingerprintsOnHost ? inputTile.data() : fingerprints.data(),
                                labels,
                                initialNodes,
                                initialEntries,
                                numWords,
                                stream,
                                summaryCacheBytes);
  AsyncDeviceVector<int>            keys(batchCapacity, stream);
  AsyncDeviceVector<int>            sortedKeys(batchCapacity, stream);
  AsyncDeviceVector<int>            values(batchCapacity, stream);
  AsyncDeviceVector<int>            sortedValues(batchCapacity, stream);
  AsyncDeviceVector<int>            groupKeys(batchCapacity, stream);
  AsyncDeviceVector<int>            sortedGroupKeys(batchCapacity, stream);
  AsyncDeviceVector<int>            sortedGroupValues(batchCapacity, stream);
  AsyncDeviceVector<int>            splitLeft(batchCapacity, stream);
  AsyncDeviceVector<int>            splitRight(batchCapacity, stream);
  AsyncDeviceVector<std::int8_t>    splitAffinity(tree.entryCapacity, stream);
  AsyncDeviceVector<Component>      proposalSums(static_cast<std::size_t>(batchCapacity) * numWords * 32, stream);
  AsyncDeviceVector<int>            dirtyNodes(tree.nodeCapacity, stream);
  AsyncDeviceVector<int>            deltaSlots(tree.nodeCapacity, stream);
  AsyncDeviceVector<int>            deltaCursor(1, stream);
  AsyncDeviceVector<Component>      deltaSums(0, stream);
  AsyncDeviceVector<std::uint32_t>  deltaCounts(0, stream);
  // Dirty nodes per tree level for one epoch. Each level holds at most one
  // node per distinct routed leaf, so capacity is batch size times height.
  AsyncDeviceVector<int>            levelNodes(0, stream);
  AsyncDeviceVector<int>            levelCounts(0, stream);
  // Each routed leaf splits at most once per epoch and defers two child BFs.
  AsyncDeviceVector<int>            summaryNodes(2 * static_cast<std::size_t>(batchCapacity), stream);
  AsyncDeviceVector<int>            summaryEntries(2 * static_cast<std::size_t>(batchCapacity), stream);
  AsyncDeviceVector<int>            summaryCount(1, stream);
  AsyncDeviceVector<int>            control(6, stream);
  cudaCheckError(cudaMemsetAsync(dirtyNodes.data(), 0, tree.nodeCapacity * sizeof(int), stream));
  cudaCheckError(cudaMemsetAsync(deltaSlots.data(), 0xff, tree.nodeCapacity * sizeof(int), stream));
  auto storage = tree.storage();
  bitBirchInitializeKernel<<<1, 1, 0, stream>>>(storage);
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
    const bool enableGroups = begin >= orderedWarmupSize;
    const int  boundary = begin < orderedWarmupSize ? std::min(orderedWarmupSize, numFingerprints) : numFingerprints;
    const int  count    = std::min(batchCapacity, boundary - begin);
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
        requiredNodes > tree.nodeCapacity ?
          static_cast<int>(
            std::min<std::int64_t>(maxNodes, std::max<std::int64_t>(requiredNodes, tree.nodeCapacity * 3LL / 2))) :
          tree.nodeCapacity;
      const int entryCapacity =
        requiredEntries > tree.entryCapacity ?
          static_cast<int>(
            std::min<std::int64_t>(maxEntries, std::max<std::int64_t>(requiredEntries, tree.entryCapacity * 3LL / 2))) :
          tree.entryCapacity;
      const auto oldNodeCapacity = tree.nodeCapacity;
      tree.grow(nodeCapacity, entryCapacity);
      if (tree.nodeCapacity > oldNodeCapacity) {
        dirtyNodes.resize(tree.nodeCapacity);
        deltaSlots.resize(tree.nodeCapacity);
        cudaCheckError(cudaMemsetAsync(dirtyNodes.data() + oldNodeCapacity,
                                       0,
                                       (tree.nodeCapacity - oldNodeCapacity) * sizeof(int),
                                       stream));
        cudaCheckError(cudaMemsetAsync(deltaSlots.data() + oldNodeCapacity,
                                       0xff,
                                       (tree.nodeCapacity - oldNodeCapacity) * sizeof(int),
                                       stream));
      }
      const int leafDepth = hostControl[4];
      if (levelCounts.size() < static_cast<std::size_t>(leafDepth) + 1) {
        levelCounts.resize(leafDepth + 1);
        levelNodes.resize(static_cast<std::size_t>(leafDepth + 1) * batchCapacity);
      }
      cudaCheckError(cudaMemsetAsync(levelCounts.data(), 0, levelCounts.size() * sizeof(int), stream));
      if (splitAffinity.size() < static_cast<std::size_t>(tree.entryCapacity)) {
        splitAffinity.resize(tree.entryCapacity);
      }
      // Bound lazy materialization plus two summaries per possible split level.
      const auto required =
        static_cast<std::int64_t>(hostControl[2]) + static_cast<std::int64_t>(pending) * (2 * hostControl[4] + 5) + 4;
      if (required > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("BitBIRCH summary workspace exceeds index capacity");
      }
      tree.summaryArena.reserve(static_cast<int>(required));
      // At most one ancestor per depth for each pending molecule. Scratch is
      // bounded by batch size times tree height, not by total cluster count.
      const auto deltaCapacity =
        std::min(static_cast<std::size_t>(tree.nodeCapacity), static_cast<std::size_t>(pending) * hostControl[4]);
      if (deltaCounts.size() < deltaCapacity) {
        deltaCounts.resize(deltaCapacity);
        deltaSums.resize(deltaCapacity * numWords * 32);
      }
      cudaCheckError(cudaMemsetAsync(deltaCursor.data(), 0, sizeof(int), stream));
      storage = tree.storage();
      if (fingerprintsOnHost) {
        ownedFingerprints.reserve(tree.entryCapacity);
        storage.singletonFingerprintPages = ownedFingerprints.data();
        storage.singletonPageHits         = ownedFingerprints.pageHits();
        storage.queryBegin                = begin;
      }
      bitBirchRouteKernel<<<count, cooperativeBlockSize, 0, stream>>>(begin,
                                                                      count,
                                                                      keys.data(),
                                                                      values.data(),
                                                                      groupKeys.data(),
                                                                      enableGroups,
                                                                      threshold,
                                                                      storage);
      cudaCheckError(cudaGetLastError());
      if (enableGroups) {
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
        bitBirchGroupsKernel<<<count, cooperativeBlockSize, 0, stream>>>(count,
                                                                         sortedGroupKeys.data(),
                                                                         sortedGroupValues.data(),
                                                                         threshold,
                                                                         proposalSums.data(),
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
      bitBirchLeafOwnersKernel<<<count, cooperativeBlockSize, 0, stream>>>(count,
                                                                           sortedKeys.data(),
                                                                           sortedValues.data(),
                                                                           threshold,
                                                                           branchingFactor,
                                                                           leafDepth,
                                                                           dirtyNodes.data(),
                                                                           levelNodes.data(),
                                                                           levelCounts.data(),
                                                                           batchCapacity,
                                                                           storage);
      cudaCheckError(cudaGetLastError());
      // Each level lists at most one node per routed leaf, so count blocks suffice.
      for (int level = leafDepth; level > 0; --level) {
        bitBirchRefreshLevelKernel<<<count, cooperativeBlockSize, 0, stream>>>(
          levelNodes.data() + static_cast<std::size_t>(level) * batchCapacity,
          levelCounts.data() + level,
          count,
          sortedKeys.data(),
          sortedValues.data(),
          dirtyNodes.data(),
          deltaSlots.data(),
          deltaCursor.data(),
          static_cast<int>(deltaCapacity),
          deltaSums.data(),
          deltaCounts.data(),
          storage);
        cudaCheckError(cudaGetLastError());
      }
      bitBirchSplitSeedsKernel<<<count, cooperativeBlockSize, splitCacheBytes, stream>>>(count,
                                                                                         sortedKeys.data(),
                                                                                         branchingFactor,
                                                                                         splitLeft.data(),
                                                                                         splitRight.data(),
                                                                                         splitAffinity.data(),
                                                                                         storage);
      cudaCheckError(cudaGetLastError());
      bitBirchRepairKernel<<<1, cooperativeBlockSize, 0, stream>>>(count,
                                                                   sortedKeys.data(),
                                                                   splitLeft.data(),
                                                                   splitRight.data(),
                                                                   splitAffinity.data(),
                                                                   branchingFactor,
                                                                   leafDepth,
                                                                   levelNodes.data(),
                                                                   levelCounts.data(),
                                                                   batchCapacity,
                                                                   deltaSlots.data(),
                                                                   summaryNodes.data(),
                                                                   summaryEntries.data(),
                                                                   summaryCount.data(),
                                                                   storage);
      cudaCheckError(cudaGetLastError());
      bitBirchSummarizeKernel<<<2 * count, cooperativeBlockSize, 0, stream>>>(summaryNodes.data(),
                                                                              summaryEntries.data(),
                                                                              summaryCount.data(),
                                                                              storage);
      cudaCheckError(cudaGetLastError());
      bitBirchControlKernel<<<1, cooperativeBlockSize, 0, stream>>>(begin, count, control.data(), storage);
      cudaCheckError(cudaGetLastError());
      control.copyToHost(hostControl);
      cudaCheckError(cudaStreamSynchronize(stream));
      if (hostControl[3] != static_cast<int>(BitBirchStatus::Success)) {
        throw std::runtime_error("BitBIRCH tree failure (status " + std::to_string(hostControl[3]) + ")");
      }
      if (hostControl[0] >= pending) {
        throw std::runtime_error("BitBIRCH insertion failed to make progress");
      }
      pending = hostControl[0];
    }
    begin += count;
    tree.summaryArena.rotate();
    ownedFingerprints.rotate();
  }
  if (returnCentroids) {
    const int clusters = hostControl[5] - hostControl[1] + 1;
    result.centroids.resize(static_cast<std::size_t>(clusters) * numWords);
  }
  storage.centroids                  = returnCentroids ? result.centroids.data() : nullptr;
  // Entry owners retain their minimum input index. Sort only the K live leaf
  // entries by that index to preserve first-member label numbering without an N-wide
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
  bitBirchCollectLeavesKernel<<<entryBlocks, cooperativeBlockSize, 0, stream>>>(liveEntries,
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
  bitBirchClusterMapKernel<<<clusterBlocks, cooperativeBlockSize, 0, stream>>>(clusters,
                                                                               sortedLeafEntries.data(),
                                                                               storage);
  cudaCheckError(cudaGetLastError());
  const int labelBlocks = (numFingerprints + cooperativeBlockSize - 1) / cooperativeBlockSize;
  bitBirchFinalizeKernel<<<labelBlocks, cooperativeBlockSize, 0, stream>>>(numFingerprints, storage);
  cudaCheckError(cudaGetLastError());
  int clusterCount = 0;
  tree.clusterCount.get(clusterCount);
  cudaCheckError(cudaStreamSynchronize(stream));
  result.numClusters = clusterCount;
  return result;
}

}  // namespace

BitBirchResult bitBirchGpu(const cuda::std::span<const std::uint32_t> fingerprints,
                           const int                                  numFingerprints,
                           const int                                  numWords,
                           const double                               threshold,
                           const int                                  branchingFactor,
                           const int                                  batchSize,
                           const std::size_t                          summaryCacheBytes,
                           const bool                                 fingerprintsOnHost,
                           const bool                                 clusterIdsOnHost,
                           const bool                                 returnCentroids,
                           const cudaStream_t                         stream,
                           const std::size_t                          fingerprintCacheBytes) {
  const ScopedNvtxRange range("BitBIRCH tree construction");
  if (numFingerprints < 0 || numFingerprints > std::numeric_limits<int>::max() - cooperativeBlockSize ||
      numWords <= 0 || numWords > std::numeric_limits<int>::max() / 32 ||
      fingerprints.size() != static_cast<std::size_t>(numFingerprints) * numWords) {
    throw std::invalid_argument("BitBIRCH fingerprints shape or dimensions are invalid");
  }
  if (!std::isfinite(threshold) || threshold < 0 || threshold > 1 || branchingFactor < 3 || batchSize < 1 ||
      (fingerprintCacheBytes > 0 &&
       (!fingerprintsOnHost ||
        fingerprintCacheBytes < sizeof(std::uint32_t) * static_cast<std::size_t>(summaryEntriesPerPage) * numWords))) {
    throw std::invalid_argument("BitBIRCH clustering options are invalid");
  }
  const auto minimumNodeSize = branchingFactor / 2 + branchingFactor % 2;
  const auto maximumEntries  = static_cast<std::int64_t>(numFingerprints) + numFingerprints / (minimumNodeSize - 1) + 8;
  if (maximumEntries > std::numeric_limits<int>::max() - summaryEntriesPerPage) {
    throw std::invalid_argument("BitBIRCH metadata exceeds supported index capacity");
  }
  if (numFingerprints == 0) {
    BitBirchResult result{AsyncDeviceVector<int>(0, stream), AsyncDeviceVector<std::uint32_t>(0, stream), 0, numWords};
    result.clusterIdsOnHost = clusterIdsOnHost;
    return result;
  }
  if (numFingerprints <= std::numeric_limits<std::uint16_t>::max()) {
    return launchBitBirch<std::uint16_t>(fingerprints,
                                         numFingerprints,
                                         numWords,
                                         threshold,
                                         branchingFactor,
                                         batchSize,
                                         summaryCacheBytes,
                                         fingerprintCacheBytes,
                                         fingerprintsOnHost,
                                         clusterIdsOnHost,
                                         returnCentroids,
                                         stream);
  }
  return launchBitBirch<std::uint32_t>(fingerprints,
                                       numFingerprints,
                                       numWords,
                                       threshold,
                                       branchingFactor,
                                       batchSize,
                                       summaryCacheBytes,
                                       fingerprintCacheBytes,
                                       fingerprintsOnHost,
                                       clusterIdsOnHost,
                                       returnCentroids,
                                       stream);
}

}  // namespace nvMolKit
