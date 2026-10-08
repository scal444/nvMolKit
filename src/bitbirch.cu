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
#include "src/utils/device.h"
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

//! Epoch state the host reads back after each insertion epoch.
enum EpochControl : int {
  PendingInputs,
  NodeCount,
  SummaryCount,
  TreeStatus,
  LeafDepth,
  EntryCount,
  NumEpochControls
};
constexpr int orderedWarmupSize = 16384;

// Entry-indexed storage in fixed pages of summaryEntriesPerPage entries. With
// a positive cache budget, pages live in mapped pinned host memory and the
// most-used ones are mirrored on the GPU; the device pointer table always names
// each page's current copy. Cold pages are read through the mapping, so cache
// capacity never changes what kernels compute, only where they read it.
template <typename T> class PagedEntryStorage {
 public:
  PagedEntryStorage(const std::size_t  elementsPerEntry,
                    const cudaStream_t stream,
                    const std::size_t  cacheBytes,
                    const char*        tooSmallMessage)
      : elementsPerEntry_(elementsPerEntry),
        stream_(stream),
        cachePages_(cacheBytes / pageBytes()) {
    if (cacheBytes > 0 && cachePages_ == 0) {
      throw std::invalid_argument(tooSmallMessage);
    }
    pointers_.setStream(stream);
    pageHits_.setStream(stream);
  }

  PagedEntryStorage(const PagedEntryStorage&)            = delete;
  PagedEntryStorage& operator=(const PagedEntryStorage&) = delete;

  ~PagedEntryStorage() {
    // Host backing must outlive every kernel and transfer, including on
    // exception paths.
    if (cached()) {
      cudaStreamSynchronize(stream_);
    }
  }

  void reserve(const int entries) {
    const auto required = static_cast<std::size_t>((static_cast<std::int64_t>(entries) + summaryEntriesPerPage - 1) /
                                                   summaryEntriesPerPage);
    const auto oldPages = numPages();
    if (required <= oldPages) {
      return;
    }
    while (numPages() < required) {
      if (!cached()) {
        devicePages_.emplace_back(pageElements(), stream_);
        continue;
      }
      hostPages_.emplace_back(pageElements(), T{0});
      T* mapped = nullptr;
      cudaCheckError(cudaHostGetDevicePointer(&mapped, hostPages_.back().data(), 0));
      mappedHostPages_.push_back(mapped);
      pageToSlot_.push_back(-1);
      pageScores_.push_back(0);
      if (devicePages_.size() < cachePages_) {
        const int slot = static_cast<int>(devicePages_.size());
        const int page = static_cast<int>(hostPages_.size() - 1);
        devicePages_.emplace_back(pageElements(), stream_);
        pageToSlot_[page] = slot;
        slotToPage_.push_back(page);
        cudaCheckError(cudaMemcpyAsync(devicePages_.back().data(),
                                       hostPages_.back().data(),
                                       pageBytes(),
                                       cudaMemcpyHostToDevice,
                                       stream_));
      }
    }
    if (cached()) {
      pageHits_.resize(required);
      cudaCheckError(
        cudaMemsetAsync(pageHits_.data() + oldPages, 0, (required - oldPages) * sizeof(std::uint32_t), stream_));
    }
    updatePointers();
  }

  // Called only after all writers in the logical batch have completed. Host
  // backing is stale for cached pages, so a slot is written back before reuse.
  void rotate() {
    if (!cached() || numPages() <= cachePages_) {
      return;
    }
    std::vector<std::uint32_t> hits(numPages());
    pageHits_.copyToHost(hits);
    cudaCheckError(cudaStreamSynchronize(stream_));
    cudaCheckError(cudaMemsetAsync(pageHits_.data(), 0, hits.size() * sizeof(std::uint32_t), stream_));
    for (std::size_t page = 0; page < hits.size(); ++page) {
      pageScores_[page] = pageScores_[page] / 2 + hits[page];
    }
    std::vector<int> order(hits.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [this](const int left, const int right) {
      if (pageScores_[left] != pageScores_[right]) {
        return pageScores_[left] > pageScores_[right];
      }
      return pageToSlot_[left] >= 0 && pageToSlot_[right] < 0;
    });
    order.resize(devicePages_.size());
    std::vector<bool> retain(hits.size(), false);
    for (const int page : order) {
      retain[page] = true;
    }
    bool changed = false;
    for (const int page : order) {
      if (pageToSlot_[page] >= 0) {
        continue;
      }
      int slot = 0;
      while (retain[slotToPage_[slot]]) {
        ++slot;
      }
      const int evicted = slotToPage_[slot];
      cudaCheckError(cudaMemcpyAsync(hostPages_[evicted].data(),
                                     devicePages_[slot].data(),
                                     pageBytes(),
                                     cudaMemcpyDeviceToHost,
                                     stream_));
      cudaCheckError(cudaMemcpyAsync(devicePages_[slot].data(),
                                     hostPages_[page].data(),
                                     pageBytes(),
                                     cudaMemcpyHostToDevice,
                                     stream_));
      pageToSlot_[evicted] = -1;
      pageToSlot_[page]    = slot;
      slotToPage_[slot]    = page;
      changed              = true;
    }
    if (changed) {
      updatePointers();
    }
  }

  T**            pointers() const noexcept { return pointers_.data(); }
  //! Per-page access counters that drive rotation; null when nothing is cached.
  std::uint32_t* pageHits() const noexcept { return cached() ? pageHits_.data() : nullptr; }
  std::size_t    numPages() const noexcept { return cached() ? hostPages_.size() : devicePages_.size(); }

 private:
  bool        cached() const noexcept { return cachePages_ > 0; }
  std::size_t pageElements() const noexcept {
    return static_cast<std::size_t>(summaryEntriesPerPage) * elementsPerEntry_;
  }
  std::size_t pageBytes() const noexcept { return pageElements() * sizeof(T); }

  void updatePointers() {
    std::vector<T*> pointers;
    pointers.reserve(numPages());
    for (std::size_t page = 0; page < numPages(); ++page) {
      if (!cached()) {
        pointers.push_back(devicePages_[page].data());
      } else {
        const int slot = pageToSlot_[page];
        pointers.push_back(slot >= 0 ? devicePages_[slot].data() : mappedHostPages_[page]);
      }
    }
    pointers_.setFromVector(pointers);
    if (cached()) {
      cudaCheckError(cudaStreamSynchronize(stream_));
    }
  }

  std::size_t                       elementsPerEntry_;
  cudaStream_t                      stream_;
  std::size_t                       cachePages_;
  std::vector<AsyncDeviceVector<T>> devicePages_;
  std::vector<PinnedHostVector<T>>  hostPages_;
  std::vector<T*>                   mappedHostPages_;
  std::vector<int>                  pageToSlot_;
  std::vector<int>                  slotToPage_;
  std::vector<std::uint64_t>        pageScores_;
  AsyncDeviceVector<std::uint32_t>  pageHits_;
  AsyncDeviceVector<T*>             pointers_;
};

// Bit Feature sums, optionally host-backed, plus packed centroids that always
// stay on the GPU because routing reads them for every candidate entry.
template <typename Component> class PagedSummaryArena {
 public:
  PagedSummaryArena(const int numBits, const int numWords, const cudaStream_t stream, const std::size_t cacheBytes = 0)
      : sums_(numBits, stream, cacheBytes, "summary_cache_bytes must fit at least one BF-sum page"),
        numWords_(numWords),
        stream_(stream) {
    centroidPagePointers_.setStream(stream);
  }

  void reserve(const int entries) {
    sums_.reserve(entries);
    if (centroidPages_.size() == sums_.numPages()) {
      return;
    }
    while (centroidPages_.size() < sums_.numPages()) {
      centroidPages_.emplace_back(static_cast<std::size_t>(summaryEntriesPerPage) * numWords_, stream_);
    }
    std::vector<std::uint32_t*> pointers;
    pointers.reserve(centroidPages_.size());
    for (auto& page : centroidPages_) {
      pointers.push_back(page.data());
    }
    centroidPagePointers_.setFromVector(pointers);
  }

  void            rotate() { sums_.rotate(); }
  std::uint32_t*  pageHits() const noexcept { return sums_.pageHits(); }
  Component**     linearSumPages() const noexcept { return sums_.pointers(); }
  std::uint32_t** centroidPages() const noexcept { return centroidPagePointers_.data(); }
  int             capacity() const noexcept { return static_cast<int>(centroidPages_.size()) * summaryEntriesPerPage; }

 private:
  PagedEntryStorage<Component>                  sums_;
  int                                           numWords_;
  cudaStream_t                                  stream_;
  std::vector<AsyncDeviceVector<std::uint32_t>> centroidPages_;
  AsyncDeviceVector<std::uint32_t*>             centroidPagePointers_;
};

template <typename Component> struct TreeStorage {
  const std::uint32_t* fingerprints;
  int*                 nodeHeads;
  int*                 nodeSizes;
  int*                 nodeParents;
  std::uint8_t*        nodeLeaves;
  int*                 nodeTails;
  int*                 nodeParentEntries;
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
  // Molecule index held in labels[0]. Host-labeled runs keep only the current
  // batch on the GPU; device-labeled runs keep all labels and use zero.
  int                  labelBegin                = 0;
  // Contiguous ordered entry copies for every node, kept in list order by each
  // append and split. Kernels stage entries from them instead of walking lists.
  int*                 nodeDirectories           = nullptr;
  int*                 directoryEntries          = nullptr;
  int*                 directoryCursor           = nullptr;
  int                  maxDirectories            = 0;
  int                  directoryStride           = 0;
};

template <typename Component> class TreeWorkspace {
 public:
  TreeWorkspace(const std::uint32_t* fingerprints,
                int*                 labels,
                const int            nodeCapacity,
                const int            entryCapacity,
                const int            branchingFactor,
                const int            numWords,
                const cudaStream_t   stream,
                const std::size_t    summaryCacheBytes)
      : nodeCapacity(nodeCapacity),
        directoryStride(branchingFactor + 1),
        entryCapacity(entryCapacity),
        nodeHeads(nodeCapacity, stream),
        nodeSizes(nodeCapacity, stream),
        nodeParents(nodeCapacity, stream),
        nodeLeaves(nodeCapacity, stream),
        nodeTails(nodeCapacity, stream),
        nodeParentEntries(nodeCapacity, stream),
        nodeDirectories(nodeCapacity, stream),
        directoryEntries(static_cast<std::size_t>(directoryCapacity(nodeCapacity)) * directoryStride, stream),
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
        directoryCursor(0, stream),
        summaryArena(numWords * 32, numWords, stream, summaryCacheBytes),
        fingerprints(fingerprints),
        labels(labels),
        numWords(numWords) {
    summaryArena.reserve(summaryEntriesPerPage);
  }

  TreeStorage<Component> storage() {
    TreeStorage<Component> result{fingerprints,
                                  nodeHeads.data(),
                                  nodeSizes.data(),
                                  nodeParents.data(),
                                  nodeLeaves.data(),
                                  nodeTails.data(),
                                  nodeParentEntries.data(),
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
    result.nodeDirectories  = nodeDirectories.data();
    result.directoryEntries = directoryEntries.data();
    result.directoryCursor  = directoryCursor.data();
    result.maxDirectories   = directoryCapacity(nodeCapacity);
    result.directoryStride  = directoryStride;
    return result;
  }

  void grow(const int nodes, const int entries) {
    if (nodes > nodeCapacity) {
      nodeHeads.resize(nodes);
      nodeSizes.resize(nodes);
      nodeParents.resize(nodes);
      nodeLeaves.resize(nodes);
      nodeTails.resize(nodes);
      nodeParentEntries.resize(nodes);
      nodeDirectories.resize(nodes);
      directoryEntries.resize(static_cast<std::size_t>(directoryCapacity(nodes)) * directoryStride);
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

  // Every node, leaf or internal, keeps an ordered directory of its entries.
  int directoryCapacity(const int nodes) const noexcept { return nodes + 2; }

  int                              nodeCapacity;
  int                              directoryStride;
  int                              entryCapacity;
  AsyncDeviceVector<int>           nodeHeads;
  AsyncDeviceVector<int>           nodeSizes;
  AsyncDeviceVector<int>           nodeParents;
  AsyncDeviceVector<std::uint8_t>  nodeLeaves;
  AsyncDeviceVector<int>           nodeTails;
  AsyncDeviceVector<int>           nodeParentEntries;
  AsyncDeviceVector<int>           nodeDirectories;
  AsyncDeviceVector<int>           directoryEntries;
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
  AsyncDevicePtr<int>              directoryCursor;
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
__device__ __forceinline__ int& labelOf(const TreeStorage<Component>& storage, const int molecule) {
  return storage.labels[molecule - storage.labelBegin];
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
  const int directory = (*storage.directoryCursor)++;
  if (directory >= storage.maxDirectories) {
    *storage.status = BitBirchStatus::NodeCapacity;
    return -1;
  }
  storage.nodeHeads[node]         = -1;
  storage.nodeTails[node]         = -1;
  storage.nodeSizes[node]         = 0;
  storage.nodeParents[node]       = parent;
  storage.nodeParentEntries[node] = -1;
  storage.nodeDirectories[node]   = directory;
  storage.nodeLeaves[node]        = leaf;
  return node;
}

template <typename Component>
__device__ __forceinline__ int* nodeDirectory(const TreeStorage<Component>& storage, const int node) {
  const int directory = storage.nodeDirectories[node];
  return directory < 0 ? nullptr :
                         storage.directoryEntries + static_cast<std::size_t>(directory) * storage.directoryStride;
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
  const int tail = storage.nodeTails[node];
  if (tail < 0) {
    storage.nodeHeads[node] = entry;
  } else {
    storage.entryNext[tail] = entry;
  }
  storage.entryNext[entry] = -1;
  storage.nodeTails[node]  = entry;
  if (int* directory = nodeDirectory(storage, node)) {
    directory[storage.nodeSizes[node]] = entry;
  }
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

//! Intersection and union popcounts of an entry centroid and a fingerprint.
template <typename Component>
__device__ __forceinline__ int2 entryFingerprintCounts(const TreeStorage<Component>& storage,
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
    return make_int2(intersection, unionCount);
  }
  for (int word = 0; word < storage.numWords; ++word) {
    intersection += __popc(centroid[word] & fingerprint[word]);
    unionCount += __popc(centroid[word] | fingerprint[word]);
  }
  return make_int2(intersection, unionCount);
}

template <typename Component>
__device__ __forceinline__ double entryToFingerprintSimilarity(const TreeStorage<Component>& storage,
                                                               const int                     entry,
                                                               const std::uint32_t*          fingerprint) {
  const int2 counts = entryFingerprintCounts(storage, entry, fingerprint);
  return routingSimilarity(counts.x, counts.y, storage.numBits);
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

//! Sign of entrySimilarity(entry, lhs) - entrySimilarity(entry, rhs), given both seed centroids.
//! Up to 4096 bits these are FP32 quotients, so FP32 comparisons give the same sign without FP64.
template <typename Component>
__device__ __forceinline__ std::int8_t seedAffinity(const TreeStorage<Component>& storage,
                                                    const int                     entry,
                                                    const std::uint32_t*          lhsCentroid,
                                                    const std::uint32_t*          rhsCentroid) {
  const int2 lhs = entryFingerprintCounts(storage, entry, lhsCentroid);
  const int2 rhs = entryFingerprintCounts(storage, entry, rhsCentroid);
  if (storage.numBits <= 4096) {
    const float lhsSimilarity = lhs.y == 0 ? 1.0f : __fdiv_rn(static_cast<float>(lhs.x), static_cast<float>(lhs.y));
    const float rhsSimilarity = rhs.y == 0 ? 1.0f : __fdiv_rn(static_cast<float>(rhs.x), static_cast<float>(rhs.y));
    return static_cast<std::int8_t>((lhsSimilarity > rhsSimilarity) - (lhsSimilarity < rhsSimilarity));
  }
  const double lhsSimilarity = routingSimilarity(lhs.x, lhs.y, storage.numBits);
  const double rhsSimilarity = routingSimilarity(rhs.x, rhs.y, storage.numBits);
  return static_cast<std::int8_t>((lhsSimilarity > rhsSimilarity) - (lhsSimilarity < rhsSimilarity));
}

template <typename Component>
__device__ __forceinline__ int parentEntry(const TreeStorage<Component>& storage, const int node) {
  // Splits move directory entries between nodes but never change which entry
  // points to a child, so the mapping is fixed when the child is linked.
  return storage.nodeParents[node] < 0 ? -1 : storage.nodeParentEntries[node];
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

// Link a new sibling beside its split node: either under a new root or as a
// new entry appended to the shared parent. Returns the node now owning both.
template <typename Component>
__device__ __forceinline__ int linkSplitSibling(TreeStorage<Component>& storage, const int node, const int sibling) {
  const int parent = storage.nodeParents[node];
  if (parent < 0) {
    const int newRoot  = allocateNode(storage, false, -1);
    const int lhsEntry = allocateEntry(storage);
    const int rhsEntry = allocateEntry(storage);
    if (newRoot < 0 || lhsEntry < 0 || rhsEntry < 0) {
      return -1;
    }
    storage.nodeParents[node]          = newRoot;
    storage.nodeParents[sibling]       = newRoot;
    storage.nodeParentEntries[node]    = lhsEntry;
    storage.nodeParentEntries[sibling] = rhsEntry;
    storage.entryChildren[lhsEntry]    = node;
    storage.entryChildren[rhsEntry]    = sibling;
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
  storage.nodeParentEntries[sibling]  = siblingEntry;
  appendEntry(storage, parent, siblingEntry);
  return parent;
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
  storage.nodeTails[node]    = lhsTail >= 0 ? lhsTail : lhsSeed;
  ++storage.nodeSizes[node];
  storage.entryNext[rhsSeed] = storage.nodeHeads[sibling];
  storage.nodeHeads[sibling] = rhsSeed;
  storage.nodeTails[sibling] = rhsTail >= 0 ? rhsTail : rhsSeed;
  ++storage.nodeSizes[sibling];
  const bool leaf           = storage.nodeLeaves[node] != 0;
  int*       nodeEntries    = nodeDirectory(storage, node);
  int*       siblingEntries = nodeDirectory(storage, sibling);
  int        position       = 0;
  for (int entry = storage.nodeHeads[node]; entry >= 0; entry = storage.entryNext[entry]) {
    if (!leaf) {
      storage.nodeParents[storage.entryChildren[entry]] = node;
    }
    nodeEntries[position++] = entry;
  }
  position = 0;
  for (int entry = storage.nodeHeads[sibling]; entry >= 0; entry = storage.entryNext[entry]) {
    if (!leaf) {
      storage.nodeParents[storage.entryChildren[entry]] = sibling;
    }
    siblingEntries[position++] = entry;
  }
  return linkSplitSibling(storage, node, sibling);
}

constexpr int cooperativeBlockSize = 256;
// Bits each thread accumulates at once when summing staged entries. Holding
// them in registers makes the loads for one staged entry independent.
constexpr int stagedBitsPerThread  = 8;

struct CooperativeScratch {
  int       nodeEntries[cooperativeBlockSize];
  int       entries[cooperativeBlockSize];
  int       otherEntries[cooperativeBlockSize];
  long long orders[cooperativeBlockSize];
  double    values[cooperativeBlockSize];
  double    otherValues[cooperativeBlockSize];
  int       next;         //!< Linked-list cursor while gathering entries.
  int       count;        //!< Entries gathered into entries or nodeEntries.
  int       stagedCount;  //!< Entries or delta slots staged for per-bit sums.
  int       stagedTotal;  //!< Member count summed over the staged entries.
  int       stagedSingletons;  //!< Singleton fingerprint indices staged in entries.
  int       stagedSummaries;   //!< Summary slots staged in otherEntries.
  int       bestEntry;    //!< Closest entry found by cooperativeClosestEntry.
  int       decision;     //!< Block-uniform yes/no published by thread zero.
  int       lhsSeed;      //!< Split seeds found by cooperativeFindSplitSeeds.
  int       rhsSeed;
  int       sibling;         //!< Node created by the latest split.
  int       lhsParentEntry;  //!< Directory entries summarizing the split halves.
  int       rhsParentEntry;
  int       targetEntry;  //!< Entry this block allocates or refreshes.
  int       deltaSlot;    //!< Refresh delta slot owned by this block.
  int       rangeBegin;   //!< Sorted-batch range owned by this block.
  int       rangeEnd;
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
                                                       const int*                    nodeEntries    = nullptr,
                                                       const int                     numNodeEntries = 0) {
  // A caller-staged copy of the node's ordered entries replaces the serial
  // list walk; either source yields the same candidates in the same order.
  if (threadIdx.x == 0) {
    scratch.next      = storage.nodeHeads[node];
    scratch.bestEntry = -1;
    scratch.bestValue = -1.0;
  }
  __syncthreads();
  for (int offset = 0;; offset += blockDim.x) {
    if (nodeEntries != nullptr) {
      const int chunk = max(0, min(static_cast<int>(blockDim.x), numNodeEntries - offset));
      if (threadIdx.x < chunk) {
        scratch.entries[threadIdx.x] = nodeEntries[offset + threadIdx.x];
      }
      if (threadIdx.x == 0) {
        scratch.count = chunk;
      }
    } else if (threadIdx.x == 0) {
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

// cooperativeClosestEntry over caller-staged entries without FP64 arithmetic,
// for at most 4096 bits. routingSimilarity is then a correctly rounded FP32
// quotient widened exactly to double, so comparing the FP32 values selects the
// same entry with the same first-in-order tie-break.
template <typename Component>
__device__ __forceinline__ int cooperativeClosestStagedEntry(const TreeStorage<Component>& storage,
                                                             const std::uint32_t*          fingerprint,
                                                             CooperativeScratch&           scratch,
                                                             const int*                    nodeEntries,
                                                             const int                     numNodeEntries) {
  if (threadIdx.x == 0) {
    scratch.bestEntry = -1;
    scratch.bestValue = -1.0;
  }
  __syncthreads();
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  for (int offset = 0; offset < numNodeEntries; offset += blockDim.x) {
    const bool active = offset + static_cast<int>(threadIdx.x) < numNodeEntries;
    int        entry  = active ? nodeEntries[offset + threadIdx.x] : -1;
    int        order  = active ? static_cast<int>(threadIdx.x) : INT_MAX;
    float      value  = -1.0f;
    if (active) {
      const int2 counts = entryFingerprintCounts(storage, entry, fingerprint);
      value             = counts.y == 0 ? 1.0f : __fdiv_rn(static_cast<float>(counts.x), static_cast<float>(counts.y));
    }
    for (int shift = warpSize / 2; shift > 0; shift /= 2) {
      const float otherValue = __shfl_down_sync(0xffffffffU, value, shift);
      const int   otherOrder = __shfl_down_sync(0xffffffffU, order, shift);
      const int   otherEntry = __shfl_down_sync(0xffffffffU, entry, shift);
      if (otherValue > value || (otherValue == value && otherOrder < order)) {
        value = otherValue;
        order = otherOrder;
        entry = otherEntry;
      }
    }
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
  return scratch.bestEntry;
}

//! Lanes that read one centroid together. Four keeps eight entries in flight per warp (measured best of 1-16).
constexpr int routeLanesPerEntry = 4;

// cooperativeClosestEntry over a node directory, with each centroid read by a
// lane group instead of a single thread. One-thread-per-entry loads touch 32
// cache lines per warp instruction; groups of four lanes touch eight. The similarities
// are the same integers, and ties still resolve to the lowest directory position.
// Value is float when routingSimilarity is an FP32 quotient, so comparisons
// match exactly without FP64 instructions.
template <typename Value, typename Component>
__device__ __forceinline__ int cooperativeClosestDirectoryEntry(const TreeStorage<Component>& storage,
                                                                const int*                    nodeEntries,
                                                                const int                     numNodeEntries,
                                                                const std::uint32_t*          fingerprint,
                                                                CooperativeScratch&           scratch) {
  const int      lane      = threadIdx.x % warpSize;
  const int      warp      = threadIdx.x / warpSize;
  const int      member    = threadIdx.x % routeLanesPerEntry;
  const int      numGroups = blockDim.x / routeLanesPerEntry;
  const unsigned groupMask = ((1U << routeLanesPerEntry) - 1U) << (lane - member);
  const bool     vectorize = storage.numWords % 4 == 0;
  Value          value     = -1;
  int            order     = INT_MAX;
  for (int position = threadIdx.x / routeLanesPerEntry; position < numNodeEntries; position += numGroups) {
    const std::uint32_t* centroid     = centroidFingerprint(storage, nodeEntries[position]);
    int                  intersection = 0;
    int                  unionCount   = 0;
    if (vectorize) {
      const auto* centroidVectors    = reinterpret_cast<const uint4*>(centroid);
      const auto* fingerprintVectors = reinterpret_cast<const uint4*>(fingerprint);
      for (int vector = member; vector < storage.numWords / 4; vector += routeLanesPerEntry) {
        const uint4 centroidWords    = centroidVectors[vector];
        const uint4 fingerprintWords = fingerprintVectors[vector];
        intersection += __popc(centroidWords.x & fingerprintWords.x) + __popc(centroidWords.y & fingerprintWords.y) +
                        __popc(centroidWords.z & fingerprintWords.z) + __popc(centroidWords.w & fingerprintWords.w);
        unionCount += __popc(centroidWords.x | fingerprintWords.x) + __popc(centroidWords.y | fingerprintWords.y) +
                      __popc(centroidWords.z | fingerprintWords.z) + __popc(centroidWords.w | fingerprintWords.w);
      }
    } else {
      for (int word = member; word < storage.numWords; word += routeLanesPerEntry) {
        intersection += __popc(centroid[word] & fingerprint[word]);
        unionCount += __popc(centroid[word] | fingerprint[word]);
      }
    }
    for (int offset = routeLanesPerEntry / 2; offset > 0; offset /= 2) {
      intersection += __shfl_xor_sync(groupMask, intersection, offset, routeLanesPerEntry);
      unionCount += __shfl_xor_sync(groupMask, unionCount, offset, routeLanesPerEntry);
    }
    // Each group visits ascending positions, so a strict comparison keeps the first tie.
    const Value similarity = sizeof(Value) == sizeof(float) && unionCount != 0 ?
                               __fdiv_rn(static_cast<float>(intersection), static_cast<float>(unionCount)) :
                               static_cast<Value>(routingSimilarity(intersection, unionCount, storage.numBits));
    if (member == 0 && similarity > value) {
      value = similarity;
      order = position;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    const Value otherValue = __shfl_down_sync(0xffffffffU, value, offset);
    const int   otherOrder = __shfl_down_sync(0xffffffffU, order, offset);
    if (otherValue > value || (otherValue == value && otherOrder < order)) {
      value = otherValue;
      order = otherOrder;
    }
  }
  if (lane == 0) {
    scratch.otherValues[warp] = value;
    scratch.orders[warp]      = order;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    Value bestValue = -1;
    int   bestOrder = INT_MAX;
    for (int index = 0; index < (blockDim.x + warpSize - 1) / warpSize; ++index) {
      const auto candidate = static_cast<Value>(scratch.otherValues[index]);
      if (candidate > bestValue || (candidate == bestValue && scratch.orders[index] < bestOrder)) {
        bestValue = candidate;
        bestOrder = static_cast<int>(scratch.orders[index]);
      }
    }
    scratch.bestValue = bestValue;
    scratch.bestEntry = bestOrder == INT_MAX ? -1 : nodeEntries[bestOrder];
  }
  __syncthreads();
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

// While numBits * n^2 <= 2^52 for the combined count n, every FP64 product and
// partial sum of the iSIM terms is an integer below 2^53. FP64 accumulation is
// then exact in any summation order and equals 64-bit integer totals converted
// once, so callers may sum integers instead. Larger clusters keep FP64.
__device__ __forceinline__ bool integerISimTerms(const std::uint64_t combinedCount, const int numBits) {
  return combinedCount <= (1ULL << 26) &&
         combinedCount * combinedCount <= (1ULL << 52) / static_cast<std::uint64_t>(numBits);
}

//! Block-wide sum of per-thread integer iSIM terms, returned as exact doubles.
__device__ __forceinline__ bitbirch::ISimTanimotoTerms cooperativeIntegerISimTerms(unsigned long long  commonPairs,
                                                                                    unsigned long long  mismatches,
                                                                                    CooperativeScratch& scratch) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    commonPairs += __shfl_down_sync(0xffffffffU, commonPairs, offset);
    mismatches += __shfl_down_sync(0xffffffffU, mismatches, offset);
  }
  const int numWarps = (blockDim.x + warpSize - 1) / warpSize;
  if (threadIdx.x % warpSize == 0) {
    scratch.orders[threadIdx.x / warpSize]            = static_cast<long long>(commonPairs);
    scratch.orders[numWarps + threadIdx.x / warpSize] = static_cast<long long>(mismatches);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    long long totalCommonPairs = 0;
    long long totalMismatches  = 0;
    for (int warp = 0; warp < numWarps; ++warp) {
      totalCommonPairs += scratch.orders[warp];
      totalMismatches += scratch.orders[numWarps + warp];
    }
    scratch.accumulatedValue      = static_cast<double>(totalCommonPairs);
    scratch.accumulatedOtherValue = static_cast<double>(totalMismatches);
  }
  __syncthreads();
  return {scratch.accumulatedValue, scratch.accumulatedOtherValue};
}

//! cooperativeCombinedISimTerms without FP64 arithmetic; see integerISimTerms.
template <typename Component>
__device__ __forceinline__ bitbirch::ISimTanimotoTerms cooperativeCombinedISimTermsExact(
  const TreeStorage<Component>& storage,
  const int                     entry,
  const std::uint32_t*          fingerprint,
  CooperativeScratch&           scratch) {
  const auto combinedCount = static_cast<std::uint64_t>(storage.entryCounts[entry]) + 1;
  if (!integerISimTerms(combinedCount, storage.numBits)) {
    return cooperativeCombinedISimTerms(storage, entry, fingerprint, scratch);
  }
  if (threadIdx.x == 0) {
    const int slot = storage.entrySummarySlots[entry];
    if (slot >= 0 && storage.summaryPageHits != nullptr) {
      atomicAdd(storage.summaryPageHits + slot / summaryEntriesPerPage, 1U);
    }
  }
  // Resolve the entry's row once instead of per bit, so each thread's bit
  // loads are independent. linearSum counted one singleton page hit per call
  // (word zero, lane zero); keep that count.
  const int            fingerprintIndex = storage.entryFingerprintIndices[entry];
  const std::uint32_t* singleton        = nullptr;
  const Component*     sums             = nullptr;
  if (fingerprintIndex >= 0) {
    if (storage.singletonFingerprintPages != nullptr) {
      if (storage.singletonPageHits != nullptr && threadIdx.x == 0) {
        atomicAdd(storage.singletonPageHits + fingerprintIndex / summaryEntriesPerPage, 1U);
      }
      singleton = storage.singletonFingerprintPages[fingerprintIndex / summaryEntriesPerPage] +
                  static_cast<std::size_t>(fingerprintIndex % summaryEntriesPerPage) * storage.numWords;
    } else {
      singleton = storage.fingerprints + static_cast<std::size_t>(fingerprintIndex) * storage.numWords;
    }
  } else {
    const int slot = storage.entrySummarySlots[entry];
    sums           = storage.linearSumPages[slot / summaryEntriesPerPage] +
                     static_cast<std::size_t>(slot % summaryEntriesPerPage) * storage.numBits;
  }
  unsigned long long commonPairs = 0;
  unsigned long long mismatches  = 0;
  for (int base = 0; base < storage.numBits; base += blockDim.x * stagedBitsPerThread) {
    std::uint64_t components[stagedBitsPerThread];
#pragma unroll
    for (int part = 0; part < stagedBitsPerThread; ++part) {
      const int bit = base + part * blockDim.x + threadIdx.x;
      if (bit < storage.numBits) {
        const std::uint64_t entryValue = singleton != nullptr ? (singleton[bit / 32] >> (bit % 32)) & 1U :
                                                                static_cast<std::uint64_t>(sums[bit]);
        components[part]               = entryValue + ((fingerprint[bit / 32] >> (bit % 32)) & 1U);
      } else {
        components[part] = 0;
      }
    }
#pragma unroll
    for (int part = 0; part < stagedBitsPerThread; ++part) {
      const std::uint64_t component = components[part];
      commonPairs += component * (component - 1) / 2;
      mismatches += component * (combinedCount - component);
    }
  }
  return cooperativeIntegerISimTerms(commonPairs, mismatches, scratch);
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
    const bool set =
      bit < storage.numBits &&
      bitbirch::majorityCentroidBit(static_cast<std::uint64_t>(materializedLinearSum(storage, entry, bit)), count);
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

//! Rows whose loads one thread keeps in flight while summing staged rows.
constexpr int stagedRowsInFlight = 4;

// Adds bit base + part * blockDim.x + threadIdx.x of each packed fingerprint
// row(index) to sums[part]. Rows are loaded stagedRowsInFlight at a time so
// their latencies overlap instead of serializing one row per memory round trip.
template <typename Component, typename RowAt>
__device__ __forceinline__ void accumulateStagedBits(Component (&sums)[stagedBitsPerThread],
                                                     const int numRows,
                                                     const int base,
                                                     const int numBits,
                                                     RowAt     rowAt) {
  for (int first = 0; first < numRows; first += stagedRowsInFlight) {
    std::uint32_t words[stagedRowsInFlight][stagedBitsPerThread];
#pragma unroll
    for (int row = 0; row < stagedRowsInFlight; ++row) {
      const std::uint32_t* fingerprint = first + row < numRows ? rowAt(first + row) : nullptr;
#pragma unroll
      for (int part = 0; part < stagedBitsPerThread; ++part) {
        const int bit    = base + part * blockDim.x + threadIdx.x;
        words[row][part] = fingerprint != nullptr && bit < numBits ? fingerprint[bit / 32] : 0U;
      }
    }
#pragma unroll
    for (int row = 0; row < stagedRowsInFlight; ++row) {
#pragma unroll
      for (int part = 0; part < stagedBitsPerThread; ++part) {
        const int bit = base + part * blockDim.x + threadIdx.x;
        sums[part] += static_cast<Component>((words[row][part] >> (bit % 32)) & 1U);
      }
    }
  }
}

//! accumulateStagedBits for rows of per-bit Component sums.
template <typename Component, typename RowAt>
__device__ __forceinline__ void accumulateStagedSums(Component (&sums)[stagedBitsPerThread],
                                                     const int numRows,
                                                     const int base,
                                                     const int numBits,
                                                     RowAt     rowAt) {
  for (int first = 0; first < numRows; first += stagedRowsInFlight) {
    Component values[stagedRowsInFlight][stagedBitsPerThread];
#pragma unroll
    for (int row = 0; row < stagedRowsInFlight; ++row) {
      const Component* rowSums = first + row < numRows ? rowAt(first + row) : nullptr;
#pragma unroll
      for (int part = 0; part < stagedBitsPerThread; ++part) {
        const int bit     = base + part * blockDim.x + threadIdx.x;
        values[row][part] = rowSums != nullptr && bit < numBits ? rowSums[bit] : Component{0};
      }
    }
#pragma unroll
    for (int row = 0; row < stagedRowsInFlight; ++row) {
#pragma unroll
      for (int part = 0; part < stagedBitsPerThread; ++part) {
        sums[part] += values[row][part];
      }
    }
  }
}

template <typename Component>
__device__ __forceinline__ void cooperativeSummarizeNode(TreeStorage<Component>& storage,
                                                         const int               node,
                                                         const int               targetEntry,
                                                         CooperativeScratch&     scratch) {
  if (!cooperativeMaterializeEntry(storage, targetEntry, scratch)) {
    return;
  }
  // Stage each entry's storage location from the node directory, one entry
  // per thread: singleton fingerprints and summary slots in separate lists.
  // Every thread then sums its bits from shared memory instead of re-chasing
  // links. Nodes wider than one block walk the list once.
  const int size = storage.nodeSizes[node];
  if (size <= static_cast<int>(blockDim.x)) {
    if (threadIdx.x == 0) {
      scratch.stagedTotal      = 0;
      scratch.stagedCount      = size;
      scratch.stagedSingletons = 0;
      scratch.stagedSummaries  = 0;
    }
    __syncthreads();
    std::uint32_t count = 0;
    if (threadIdx.x < size) {
      const int entry            = nodeDirectory(storage, node)[threadIdx.x];
      const int fingerprintIndex = storage.entryFingerprintIndices[entry];
      count                      = storage.entryCounts[entry];
      // Sums are integers, so the staged order of either list is irrelevant.
      if (fingerprintIndex >= 0) {
        scratch.entries[atomicAdd(&scratch.stagedSingletons, 1)] = fingerprintIndex;
      } else {
        scratch.otherEntries[atomicAdd(&scratch.stagedSummaries, 1)] = storage.entrySummarySlots[entry];
      }
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      count += __shfl_down_sync(0xffffffffU, count, offset);
    }
    if (threadIdx.x % warpSize == 0 && count != 0) {
      atomicAdd(&scratch.stagedTotal, static_cast<int>(count));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      storage.entryCounts[targetEntry] = static_cast<std::uint32_t>(scratch.stagedTotal);
    }
  } else if (threadIdx.x == 0) {
    std::uint32_t count   = 0;
    int           entries = 0;
    for (int entry = storage.nodeHeads[node]; entry >= 0; entry = storage.entryNext[entry], ++entries) {
      count += storage.entryCounts[entry];
    }
    storage.entryCounts[targetEntry] = count;
    scratch.stagedCount              = entries;
  }
  __syncthreads();
  const int numEntries = scratch.stagedCount;
  if (numEntries <= cooperativeBlockSize) {
    const int numSingletons = scratch.stagedSingletons;
    const int numSummaries  = scratch.stagedSummaries;
    // singletonWord counts one page hit per singleton read at word zero; keep
    // that accounting while the rows below are read without it.
    if (storage.singletonPageHits != nullptr && threadIdx.x == 0) {
      for (int index = 0; index < numSingletons; ++index) {
        atomicAdd(storage.singletonPageHits + scratch.entries[index] / summaryEntriesPerPage, 1U);
      }
    }
    for (int base = 0; base < storage.numBits; base += blockDim.x * stagedBitsPerThread) {
      Component sums[stagedBitsPerThread] = {};
      accumulateStagedBits(sums, numSingletons, base, storage.numBits, [&](const int index) {
        const int fingerprintIndex = scratch.entries[index];
        if (storage.singletonFingerprintPages != nullptr) {
          return static_cast<const std::uint32_t*>(
            storage.singletonFingerprintPages[fingerprintIndex / summaryEntriesPerPage] +
            static_cast<std::size_t>(fingerprintIndex % summaryEntriesPerPage) * storage.numWords);
        }
        return storage.fingerprints + static_cast<std::size_t>(fingerprintIndex) * storage.numWords;
      });
      accumulateStagedSums(sums, numSummaries, base, storage.numBits, [&](const int index) {
        const int slot = scratch.otherEntries[index];
        return static_cast<const Component*>(storage.linearSumPages[slot / summaryEntriesPerPage] +
                                             static_cast<std::size_t>(slot % summaryEntriesPerPage) * storage.numBits);
      });
#pragma unroll
      for (int part = 0; part < stagedBitsPerThread; ++part) {
        const int bit = base + part * blockDim.x + threadIdx.x;
        if (bit < storage.numBits) {
          materializedLinearSum(storage, targetEntry, bit) = sums[part];
        }
      }
    }
  } else {
    for (int bit = threadIdx.x; bit < storage.numBits; bit += blockDim.x) {
      Component sum = 0;
      for (int entry = storage.nodeHeads[node]; entry >= 0; entry = storage.entryNext[entry]) {
        sum += linearSum(storage, entry, bit);
      }
      materializedLinearSum(storage, targetEntry, bit) = sum;
    }
  }
  __syncthreads();
  cooperativeRefreshCentroid(storage, targetEntry);
}

template <typename Component>
__device__ __forceinline__ double cachedCentroidSimilarity(const TreeStorage<Component>& storage,
                                                           const std::uint32_t*          centroidCache,
                                                           const int                     lhsIndex,
                                                           const int                     rhsIndex) {
  int intersection = 0;
  int unionCount   = 0;
  for (int word = 0; word < storage.numWords; ++word) {
    const auto lhsWord = centroidCache[word * (cooperativeBlockSize + 1) + lhsIndex];
    const auto rhsWord = centroidCache[word * (cooperativeBlockSize + 1) + rhsIndex];
    intersection += __popc(lhsWord & rhsWord);
    unionCount += __popc(lhsWord | rhsWord);
  }
  return routingSimilarity(intersection, unionCount, storage.numBits);
}

// Staged nodeEntries, when given, must hold the node's entries in list order.
template <typename Component>
__device__ __forceinline__ void cooperativeFindSplitSeeds(const TreeStorage<Component>& storage,
                                                          const int                     node,
                                                          CooperativeScratch&           scratch,
                                                          std::uint32_t*                centroidCache = nullptr,
                                                          const int*                    nodeEntries   = nullptr) {
  double    localBest      = 2.0;
  long long localBestOrder = LLONG_MAX;
  int       localLhs       = -1;
  int       localRhs       = -1;
  long long order          = 0;
  if (storage.nodeSizes[node] <= blockDim.x) {
    if (nodeEntries != nullptr) {
      const int size = storage.nodeSizes[node];
      if (threadIdx.x < size) {
        scratch.nodeEntries[threadIdx.x] = nodeEntries[threadIdx.x];
      }
      if (threadIdx.x == 0) {
        scratch.count = size;
      }
    } else if (threadIdx.x == 0) {
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
      const int    lhs        = scratch.nodeEntries[lhsIndex];
      const int    rhs        = scratch.nodeEntries[rhsIndex];
      const double similarity = centroidCache != nullptr ?
                                  cachedCentroidSimilarity(storage, centroidCache, lhsIndex, rhsIndex) :
                                  entrySimilarity(storage, lhs, rhs);
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
    scratch.lhsSeed = lhsSeed;
    scratch.rhsSeed = rhsSeed;
  }
  __syncthreads();
}

template <typename Component>
__device__ __forceinline__ int cooperativeSplitNode(TreeStorage<Component>& storage,
                                                    int                     node,
                                                    const int               branchingFactor,
                                                    std::int8_t*            splitAffinity,
                                                    CooperativeScratch&     scratch,
                                                    std::uint32_t*          centroidCache = nullptr) {
  while (true) {
    // Lane zero otherwise reaches topology writes before a lagging warp has
    // even evaluated the loop predicate.
    const bool overflow = storage.nodeSizes[node] > branchingFactor;
    __syncthreads();
    if (!overflow) {
      break;
    }
    cooperativeFindSplitSeeds(storage, node, scratch, centroidCache, nodeDirectory(storage, node));
    const int lhsSeed = scratch.lhsSeed;
    const int rhsSeed = scratch.rhsSeed;
    const auto* lhsCentroid = centroidFingerprint(storage, lhsSeed);
    const auto* rhsCentroid = centroidFingerprint(storage, rhsSeed);
    // Compare every entry with both seeds in parallel; the ordered, size-capped
    // assignment in splitNodeWithSeeds then only reads the signs.
    const int* entries = nodeDirectory(storage, node);
    for (int position = threadIdx.x; position < storage.nodeSizes[node]; position += blockDim.x) {
      const int entry      = entries[position];
      splitAffinity[entry] = seedAffinity(storage, entry, lhsCentroid, rhsCentroid);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      const int oldParent = storage.nodeParents[node];
      scratch.sibling     = *storage.nodeCursor;  // The next allocation is the sibling node.
      scratch.node        = splitNodeWithSeeds(storage, node, branchingFactor, lhsSeed, rhsSeed, splitAffinity);
      scratch.success     = scratch.node >= 0;
      scratch.decision    = oldParent < 0;
      if (scratch.success) {
        scratch.lhsParentEntry = parentEntry(storage, node);
        scratch.rhsParentEntry = parentEntry(storage, scratch.sibling);
      }
    }
    __syncthreads();
    if (!scratch.success) {
      return scratch.node;
    }
    const int  sibling      = scratch.sibling;
    const int  parent       = scratch.node;
    const int  leftSummary  = scratch.lhsParentEntry;
    const int  rightSummary = scratch.rhsParentEntry;
    const bool newRoot      = scratch.decision;
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
  if (labelOf(storage, molecule) >= 0) {
    return;
  }
  const auto* fingerprint  = queryFingerprint(storage, molecule);
  const auto  closestEntry = [&](const int node) {
    const int* directory = nodeDirectory(storage, node);
    if (directory == nullptr) {
      return cooperativeClosestEntry(storage, node, fingerprint, scratch);
    }
    const int size = storage.nodeSizes[node];
    return storage.numBits <= 4096 ?
             cooperativeClosestDirectoryEntry<float>(storage, directory, size, fingerprint, scratch) :
             cooperativeClosestDirectoryEntry<double>(storage, directory, size, fingerprint, scratch);
  };
  while (!storage.nodeLeaves[scratch.node]) {
    const int selected = closestEntry(scratch.node);
    if (threadIdx.x == 0) {
      scratch.node = storage.entryChildren[selected];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    keys[offset] = scratch.node;
  }
  if (enableGroups && storage.nodeHeads[scratch.node] >= 0) {
    const int  leaf     = scratch.node;
    const int  selected = closestEntry(leaf);
    const auto terms    = cooperativeCombinedISimTermsExact(storage, selected, fingerprint, scratch);
    if (threadIdx.x == 0 && bitbirch::isimTanimotoAtLeast(terms,
                                                          static_cast<std::uint64_t>(storage.entryCounts[selected]) + 1,
                                                          threshold)) {
      groupKeys[offset] = selected;
    }
  }
}

template <typename Component>
__global__ void bitBirchGroupsKernel(const int              count,
                                     const int*             keys,
                                     const int*             values,
                                     const double           threshold,
                                     Component*             proposalSums,
                                     TreeStorage<Component> storage) {
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
    scratch.rangeEnd              = end;
    scratch.accumulatedValue      = 0;
    scratch.accumulatedOtherValue = 0;
  }
  __syncthreads();
  const int  end           = scratch.rangeEnd;
  const auto combinedCount = static_cast<std::uint64_t>(storage.entryCounts[entry]) + end - first;
  Component* proposed      = proposalSums + static_cast<std::size_t>(first) * storage.numBits;
  const bool integerTerms  = integerISimTerms(combinedCount, storage.numBits);

  double             common            = 0;
  double             mismatches        = 0;
  unsigned long long integerCommon     = 0;
  unsigned long long integerMismatches = 0;
  // Several bits per thread keep each member's word loads independent. Every
  // thread still visits its bits in ascending order, as the FP64 fallback needs.
  for (int base = 0; base < storage.numBits; base += blockDim.x * stagedBitsPerThread) {
    std::uint64_t sums[stagedBitsPerThread];
#pragma unroll
    for (int part = 0; part < stagedBitsPerThread; ++part) {
      const int bit = base + part * blockDim.x + threadIdx.x;
      sums[part]    = bit < storage.numBits ? static_cast<std::uint64_t>(linearSum(storage, entry, bit)) : 0;
    }
    for (int offset = first; offset < end; ++offset) {
      const auto* fingerprint = queryFingerprint(storage, values[offset]);
#pragma unroll
      for (int part = 0; part < stagedBitsPerThread; ++part) {
        const int bit = base + part * blockDim.x + threadIdx.x;
        if (bit < storage.numBits) {
          sums[part] += (fingerprint[bit / 32] >> (bit % 32)) & 1U;
        }
      }
    }
#pragma unroll
    for (int part = 0; part < stagedBitsPerThread; ++part) {
      const int bit = base + part * blockDim.x + threadIdx.x;
      if (bit >= storage.numBits) {
        continue;
      }
      const std::uint64_t sum = sums[part];
      proposed[bit]           = static_cast<Component>(sum);
      if (integerTerms) {
        integerCommon += sum * (sum - 1) / 2;
        integerMismatches += sum * (combinedCount - sum);
      } else {
        bitbirch::ISimTanimotoTerms terms{};
        bitbirch::accumulateISimTanimotoTerm(terms, sum, combinedCount);
        common += terms.commonPairs;
        mismatches += terms.mismatches;
      }
    }
  }
  if (integerTerms) {
    cooperativeIntegerISimTerms(integerCommon, integerMismatches, scratch);
  } else {
    cooperativeAccumulateISimTerms(common, mismatches, scratch);
  }
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
      labelOf(storage, values[offset]) = entry;
    }
    return;
  }
  for (int offset = first; offset < end; ++offset) {
    const int   molecule    = values[offset];
    const auto* fingerprint = queryFingerprint(storage, molecule);
    const auto  terms       = cooperativeCombinedISimTermsExact(storage, entry, fingerprint, scratch);
    if (threadIdx.x == 0) {
      scratch.decision =
        bitbirch::isimTanimotoAtLeast(terms, static_cast<std::uint64_t>(storage.entryCounts[entry]) + 1, threshold);
    }
    __syncthreads();
    if (scratch.decision) {
      if (!cooperativeAddFingerprint(storage, entry, fingerprint, scratch)) {
        return;
      }
      if (threadIdx.x == 0) {
        labelOf(storage, molecule)     = entry;
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
__global__ void bitBirchLeafOwnersKernel(const int              count,
                                         const int*             keys,
                                         const int*             values,
                                         const double           threshold,
                                         const int              branchingFactor,
                                         const int              leafDepth,
                                         int*                   dirtyNodes,
                                         int*                   levelNodes,
                                         int*                   levelCounts,
                                         const int              levelCapacity,
                                         int*                   splitPlans,
                                         int*                   splitQueue,
                                         int*                   splitQueueCount,
                                         TreeStorage<Component> storage) {
  const int first = blockIdx.x;
  if (first >= count || keys[first] == INT_MAX || (first > 0 && keys[first - 1] == keys[first])) {
    return;
  }
  __shared__ CooperativeScratch scratch;
  __shared__ int                leafEntries[cooperativeBlockSize];
  __shared__ int                numLeafEntries;
  __shared__ int                pendingQueries[cooperativeBlockSize];
  __shared__ int                warpPending[cooperativeBlockSize / 32];
  const int                     node         = keys[first];
  // Only this owner appends to its leaf, so one directory copy stages the
  // ordered entries for every query. Nodes wider than one block keep walking.
  const bool                    stageEntries = branchingFactor < cooperativeBlockSize;
  if (stageEntries) {
    const int  size      = storage.nodeSizes[node];
    const int* directory = nodeDirectory(storage, node);
    for (int position = threadIdx.x; position < size; position += blockDim.x) {
      leafEntries[position] = directory[position];
    }
    if (threadIdx.x == 0) {
      numLeafEntries = size;
    }
  }
  __syncthreads();
  // Stage this leaf's still-unlabeled queries in batch order, one block-wide
  // pass per chunk of the sorted run, instead of reading each key, molecule and
  // label serially before its query. Only this owner labels these molecules.
  const int lane       = threadIdx.x % warpSize;
  const int warp       = threadIdx.x / warpSize;
  bool      overflowed = false;
  for (int chunk = first; !overflowed; chunk += blockDim.x) {
    const int  offset  = chunk + threadIdx.x;
    const bool inRun   = offset < count && keys[offset] == node;
    const int  query   = inRun ? values[offset] : -1;
    const bool pending = inRun && labelOf(storage, query) < 0;
    const auto ballot  = __ballot_sync(0xffffffffU, pending);
    if (lane == 0) {
      warpPending[warp] = __popc(ballot);
    }
    const bool runEnds    = __syncthreads_or(!inRun);
    int        position   = 0;
    int        numPending = 0;
    for (int index = 0; index < static_cast<int>(blockDim.x / warpSize); ++index) {
      position += index < warp ? warpPending[index] : 0;
      numPending += warpPending[index];
    }
    if (pending) {
      pendingQueries[position + __popc(ballot & ((1U << lane) - 1U))] = query;
    }
    __syncthreads();
    for (int index = 0; index < numPending; ++index) {
      const int   molecule    = pendingQueries[index];
      const auto* fingerprint = queryFingerprint(storage, molecule);
      int         selected    = -1;
      // The staged count tracks the leaf size: only this owner appends to it.
      const bool  occupied    = stageEntries ? numLeafEntries > 0 : storage.nodeHeads[node] >= 0;
      if (occupied && stageEntries && storage.numBits <= 4096) {
        selected = cooperativeClosestStagedEntry(storage, fingerprint, scratch, leafEntries, numLeafEntries);
      } else if (occupied) {
        selected = cooperativeClosestEntry(storage,
                                           node,
                                           fingerprint,
                                           scratch,
                                           stageEntries ? leafEntries : nullptr,
                                           stageEntries ? numLeafEntries : 0);
      }
      bool merge = false;
      if (selected >= 0) {
        const auto terms = cooperativeCombinedISimTermsExact(storage, selected, fingerprint, scratch);
        // Publish one decision before any warp can mutate the entry count. Reading
        // that count independently immediately before the update can diverge warps.
        if (threadIdx.x == 0) {
          scratch.decision = bitbirch::isimTanimotoAtLeast(
            terms, static_cast<std::uint64_t>(storage.entryCounts[selected]) + 1, threshold);
        }
        __syncthreads();
        merge = scratch.decision;
      }
      if (merge) {
        if (!cooperativeAddFingerprint(storage, selected, fingerprint, scratch)) {
          return;
        }
      } else {
        if (threadIdx.x == 0) {
          scratch.targetEntry = allocateConcurrentLeafEntry(storage);
          scratch.success     = scratch.targetEntry >= 0;
        }
        __syncthreads();
        if (!scratch.success) {
          return;
        }
        selected = scratch.targetEntry;
        cooperativeInitializeLeafEntry(storage, selected, molecule);
        if (threadIdx.x == 0) {
          storage.entryClusterIds[selected] = molecule;
          appendEntry(storage, node, selected);
          if (stageEntries) {
            leafEntries[numLeafEntries++] = selected;
          }
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        labelOf(storage, molecule)        = selected;
        // A parked earlier query can reach an entry created later in its batch.
        // Track the actual earliest member, not the allocation/creation order.
        storage.entryClusterIds[selected] = min(storage.entryClusterIds[selected], molecule);
      }
      __syncthreads();
      if ((stageEntries ? numLeafEntries : storage.nodeSizes[node]) > branchingFactor) {
        // Remaining queries stay unassigned and are rerouted after the split.
        overflowed = true;
        break;
      }
    }
    if (runEnds) {
      break;
    }
    // Every warp has read this chunk's warpPending before the next chunk's writes.
    __syncthreads();
  }
  // An overflowing leaf is split this epoch; hand its ordered entries to the
  // split-seed search through the leaf's plan slot.
  if (stageEntries && numLeafEntries > branchingFactor) {
    for (int position = threadIdx.x; position < numLeafEntries; position += blockDim.x) {
      splitPlans[static_cast<std::size_t>(first) * cooperativeBlockSize + position] = leafEntries[position];
    }
    // Queue order is irrelevant: each leaf's pair search writes only its own slot.
    if (threadIdx.x == 0 && splitQueue != nullptr) {
      splitQueue[atomicAdd(splitQueueCount, 1)] = first;
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
  const bool                    leaf = storage.nodeLeaves[node];
  // Internal nodes that fit one block stage their dirty children in parallel
  // from the ordered directory instead of walking the entry list.
  const bool stageChildren = !leaf && storage.nodeSizes[node] <= static_cast<int>(blockDim.x);
  if (threadIdx.x == 0) {
    scratch.targetEntry = parentEntry(storage, node);
    scratch.deltaSlot   = atomicAdd(deltaCursor, 1);
    scratch.success     = scratch.targetEntry >= 0 && scratch.deltaSlot < deltaCapacity;
    if (!scratch.success) {
      *storage.status = scratch.targetEntry < 0 ? BitBirchStatus::InvalidTree : BitBirchStatus::SummaryCapacity;
    } else {
      deltaSlots[node]         = scratch.deltaSlot;
      std::uint32_t addedCount = 0;
      scratch.stagedTotal      = 0;
      if (leaf) {
        scratch.node        = -1;
        scratch.rangeBegin  = count;
        scratch.rangeEnd    = 0;
        scratch.stagedCount = 0;
      } else if (!stageChildren) {
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
      if (!leaf && !stageChildren) {
        deltaCounts[scratch.deltaSlot] = addedCount;
      }
    }
  }
  __syncthreads();
  if (leaf && scratch.success) {
    // Find this leaf's run in the sorted batch with one sampled pass, then
    // stage its accepted molecules block-wide. Integer sums make their staged
    // order irrelevant. Sample stride stays within one block-wide chunk.
    const int stride = (count + blockDim.x - 1) / blockDim.x;
    const int sample = threadIdx.x * stride;
    if (sample < count && keys[sample] < node) {
      atomicMax(&scratch.node, static_cast<int>(threadIdx.x));
    }
    __syncthreads();
    // scratch.node holds the last sample below node, so the run begins within
    // stride slots of start.
    const int start = scratch.node < 0 ? 0 : scratch.node * stride;
    for (int chunk = start;; chunk += blockDim.x) {
      const int offset = chunk + threadIdx.x;
      const int key    = offset < count ? keys[offset] : INT_MAX;
      if (key == node) {
        atomicMin(&scratch.rangeBegin, offset);
        atomicMax(&scratch.rangeEnd, offset + 1);
        const int molecule = values[offset];
        if (labelOf(storage, molecule) >= 0) {
          const int position = atomicAdd(&scratch.stagedCount, 1);
          if (position < cooperativeBlockSize) {
            scratch.entries[position] = molecule;
          }
        }
      }
      if (__syncthreads_or(key > node || chunk + static_cast<int>(blockDim.x) >= count)) {
        break;
      }
    }
    if (threadIdx.x == 0) {
      deltaCounts[scratch.deltaSlot] = static_cast<std::uint32_t>(scratch.stagedCount);
    }
  }
  if (stageChildren && scratch.success) {
    // Compact dirty children's delta slots in directory order.
    const int     lane      = threadIdx.x % warpSize;
    const int     warp      = threadIdx.x / warpSize;
    const int     size      = storage.nodeSizes[node];
    const int     childSlot = threadIdx.x < size ?
                                deltaSlots[storage.entryChildren[nodeDirectory(storage, node)[threadIdx.x]]] :
                                -1;
    const bool    dirty     = childSlot >= 0;
    const auto    ballot    = __ballot_sync(0xffffffffU, dirty);
    std::uint32_t added     = dirty ? deltaCounts[childSlot] : 0;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      added += __shfl_down_sync(0xffffffffU, added, offset);
    }
    if (lane == 0) {
      scratch.otherEntries[warp] = __popc(ballot);
      atomicAdd(&scratch.stagedTotal, static_cast<int>(added));
    }
    __syncthreads();
    int position = 0;
    for (int index = 0; index < warp; ++index) {
      position += scratch.otherEntries[index];
    }
    if (dirty) {
      scratch.entries[position + __popc(ballot & ((1U << lane) - 1U))] = childSlot;
    }
    if (threadIdx.x == blockDim.x - 1) {
      scratch.stagedCount            = position + __popc(ballot);
      deltaCounts[scratch.deltaSlot] = static_cast<std::uint32_t>(scratch.stagedTotal);
    }
  }
  __syncthreads();
  if (!scratch.success || !cooperativeMaterializeEntry(storage, scratch.targetEntry, scratch)) {
    return;
  }
  const int target = scratch.targetEntry;
  const int slot   = scratch.deltaSlot;
  if (threadIdx.x == 0) {
    storage.entryCounts[target] += deltaCounts[slot];
  }
  // Entry owners already updated leaf payloads. Propagate only this epoch's
  // accepted fingerprints, rather than rereading every old BF in each subtree.
  // One CTA owns each parent entry; no per-bit atomics or shared ancestor writes.
  const bool staged = scratch.stagedCount <= cooperativeBlockSize;
  if (staged) {
    for (int base = 0; base < storage.numBits; base += blockDim.x * stagedBitsPerThread) {
      Component added[stagedBitsPerThread] = {};
      if (leaf) {
        accumulateStagedBits(added, scratch.stagedCount, base, storage.numBits, [&](const int index) {
          return queryFingerprint(storage, scratch.entries[index]);
        });
      } else {
        accumulateStagedSums(added, scratch.stagedCount, base, storage.numBits, [&](const int index) {
          return static_cast<const Component*>(deltaSums +
                                               static_cast<std::size_t>(scratch.entries[index]) * storage.numBits);
        });
      }
#pragma unroll
      for (int part = 0; part < stagedBitsPerThread; ++part) {
        const int bit = base + part * blockDim.x + threadIdx.x;
        if (bit < storage.numBits) {
          deltaSums[static_cast<std::size_t>(slot) * storage.numBits + bit] = added[part];
          materializedLinearSum(storage, target, bit) += added[part];
        }
      }
    }
  } else {
    for (int bit = threadIdx.x; bit < storage.numBits; bit += blockDim.x) {
      Component added = 0;
      if (leaf) {
        for (int offset = scratch.rangeBegin; offset < scratch.rangeEnd; ++offset) {
          const int molecule = values[offset];
          if (labelOf(storage, molecule) >= 0) {
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
  }
  __syncthreads();
  cooperativeRefreshCentroid(storage, target);
  if (threadIdx.x == 0) {
    dirtyNodes[node] = 0;
  }
}

//! Most split pair-search slices one overflowing leaf is spread over.
constexpr int maxSplitSlices = 64;

//! Inverts cooperativeFindSplitSeeds' row-major upper-triangle pair order.
__device__ __forceinline__ int2 splitPairFromOrder(const long long order, const int size) {
  const auto rowStart = [size](const long long row) { return row * (2 * size - row - 1) / 2; };
  // An FP32 estimate; the integer corrections below make the row exact.
  const float span = 2.0f * size - 1.0f;
  const float root = sqrtf(fmaxf(0.0f, span * span - 8.0f * static_cast<float>(order)));
  long long   row  = static_cast<long long>((span - root) / 2.0f);
  row               = max(0LL, min(row, static_cast<long long>(size) - 2));
  while (row > 0 && rowStart(row) > order) {
    --row;
  }
  while (row + 1 < size - 1 && rowStart(row + 1) <= order) {
    ++row;
  }
  return make_int2(static_cast<int>(row), static_cast<int>(order - rowStart(row) + row + 1));
}

// Exact least-similar pair search for staged overflowing leaves, spread over
// many blocks. Each block owns a contiguous slice of the pair order and folds
// (similarity, pair order) into one key, so atomicMin keeps the minimum with
// cooperativeFindSplitSeeds' lowest-order tie-break. Similarities are
// nonnegative FP32 values, whose bit patterns order like the values.
template <typename Component>
__global__ void bitBirchSplitPairsKernel(const int*                   keys,
                                         const int                    branchingFactor,
                                         const int*                   splitQueue,
                                         const int*                   splitQueueCount,
                                         const int*                   splitPlans,
                                         unsigned long long*          splitKeys,
                                         const TreeStorage<Component> storage) {
  __shared__ int                  popcounts[cooperativeBlockSize];
  __shared__ const std::uint32_t* rows[cooperativeBlockSize];
  __shared__ unsigned long long   warpBest[cooperativeBlockSize / 32];
  extern __shared__ std::uint32_t centroidCache[];
  const int                       numSplits = *splitQueueCount;
  if (numSplits == 0) {
    return;
  }
  const int slices = max(1, min(maxSplitSlices, static_cast<int>(gridDim.x) / numSplits));
  for (int work = blockIdx.x; work < numSplits * slices; work += gridDim.x) {
    const int index = splitQueue[work / slices];
    const int size  = storage.nodeSizes[keys[index]];
    if (size <= branchingFactor) {
      continue;
    }
    const int* plan = splitPlans + static_cast<std::size_t>(index) * cooperativeBlockSize;
    // Resolve every centroid row at once, then copy words with independent,
    // coalesced loads instead of one dependent lookup chain per word.
    if (threadIdx.x < size) {
      rows[threadIdx.x] = centroidFingerprint(storage, plan[threadIdx.x]);
    }
    __syncthreads();
#pragma unroll 4
    for (int item = threadIdx.x; item < size * storage.numWords; item += blockDim.x) {
      const int entryIndex = item / storage.numWords;
      const int word       = item % storage.numWords;
      centroidCache[word * (cooperativeBlockSize + 1) + entryIndex] = rows[entryIndex][word];
    }
    __syncthreads();
    if (threadIdx.x < size) {
      int bits = 0;
      for (int word = 0; word < storage.numWords; ++word) {
        bits += __popc(centroidCache[word * (cooperativeBlockSize + 1) + threadIdx.x]);
      }
      popcounts[threadIdx.x] = bits;
    }
    __syncthreads();
    const long long    numPairs  = static_cast<long long>(size) * (size - 1) / 2;
    const long long    sliceSize = (numPairs + slices - 1) / slices;
    const long long    begin     = (work % slices) * sliceSize;
    const long long    end       = min(begin + sliceSize, numPairs);
    unsigned long long best      = ULLONG_MAX;
    int2               pair      = splitPairFromOrder(min(begin + threadIdx.x, numPairs - 1), size);
    for (long long order = begin + threadIdx.x; order < end; order += blockDim.x) {
      int intersection = 0;
      for (int word = 0; word < storage.numWords; ++word) {
        intersection += __popc(centroidCache[word * (cooperativeBlockSize + 1) + pair.x] &
                               centroidCache[word * (cooperativeBlockSize + 1) + pair.y]);
      }
      // |a | b| = |a| + |b| - |a & b|: the same integers cachedCentroidSimilarity counts,
      // and the same FP32 quotient routingSimilarity takes for at most 4096 bits.
      const int   unionCount = popcounts[pair.x] + popcounts[pair.y] - intersection;
      const float similarity =
        unionCount == 0 ? 1.0f : __fdiv_rn(static_cast<float>(intersection), static_cast<float>(unionCount));
      const auto key = (static_cast<unsigned long long>(__float_as_uint(similarity)) << 32) |
                       static_cast<unsigned long long>(order);
      best           = min(best, key);
      // Step blockDim.x pairs along the row-major upper triangle.
      pair.y += blockDim.x;
      while (pair.y >= size && pair.x < size - 2) {
        ++pair.x;
        pair.y += pair.x + 1 - size;
      }
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      best = min(best, __shfl_xor_sync(0xffffffffU, best, offset));
    }
    if (threadIdx.x % warpSize == 0) {
      warpBest[threadIdx.x / warpSize] = best;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int warp = 1; warp < blockDim.x / warpSize; ++warp) {
        best = min(best, warpBest[warp]);
      }
      if (best != ULLONG_MAX) {
        atomicMin(splitKeys + index, best);
      }
    }
    // The next work item overwrites the centroid cache.
    __syncthreads();
  }
}

template <typename Component>
__global__ void bitBirchSplitSeedsKernel(const int                    count,
                                         const int*                   keys,
                                         const int                    branchingFactor,
                                         int*                         splitLeft,
                                         int*                         splitRight,
                                         std::int8_t*                 splitAffinity,
                                         int*                         splitPlans,
                                         int*                         splitLeftSizes,
                                         unsigned long long*          splitKeys,
                                         const TreeStorage<Component> storage) {
  const int index = blockIdx.x;
  if (index >= count || keys[index] == INT_MAX || (index > 0 && keys[index - 1] == keys[index]) ||
      storage.nodeSizes[keys[index]] <= branchingFactor) {
    return;
  }
  // Other leaf splits cannot change this leaf's entries. Its exact seed search
  // can run concurrently; ancestor cascades still search after prior repairs.
  __shared__ CooperativeScratch scratch;
  __shared__ std::int8_t affinities[cooperativeBlockSize];
  __shared__ int         seedPositions[2];
  extern __shared__ std::uint32_t centroidCache[];
  // The leaf owner staged this leaf's ordered entries in its plan slot.
  int*                            plan     = splitPlans + static_cast<std::size_t>(index) * cooperativeBlockSize;
  // bitBirchSplitPairsKernel already searched this leaf's pairs when it left a key.
  const unsigned long long        splitKey = splitKeys != nullptr ? splitKeys[index] : ULLONG_MAX;
  const bool cacheCentroids = splitKeys == nullptr && storage.numWords <= 32 && branchingFactor < cooperativeBlockSize;
  if (splitKey != ULLONG_MAX) {
    const int size = storage.nodeSizes[keys[index]];
    if (threadIdx.x < size) {
      scratch.nodeEntries[threadIdx.x] = plan[threadIdx.x];
    }
    if (threadIdx.x == 0) {
      const int2 pair    = splitPairFromOrder(static_cast<long long>(splitKey & 0xffffffffULL), size);
      scratch.count      = size;
      scratch.lhsSeed    = plan[pair.x];
      scratch.rhsSeed    = plan[pair.y];
      splitKeys[index]   = ULLONG_MAX;
    }
    __syncthreads();
  } else {
    cooperativeFindSplitSeeds(storage,
                              keys[index],
                              scratch,
                              cacheCentroids ? centroidCache : nullptr,
                              branchingFactor < cooperativeBlockSize ? plan : nullptr);
  }
  const int left  = scratch.lhsSeed;
  const int right = scratch.rhsSeed;
  if (storage.nodeSizes[keys[index]] <= blockDim.x) {
    if (cacheCentroids && threadIdx.x < scratch.count) {
      const int entry = scratch.nodeEntries[threadIdx.x];
      if (entry == left) {
        seedPositions[0] = threadIdx.x;
      } else if (entry == right) {
        seedPositions[1] = threadIdx.x;
      }
    }
    __syncthreads();
    if (threadIdx.x < scratch.count && cacheCentroids) {
      const double lhs = cachedCentroidSimilarity(storage, centroidCache, threadIdx.x, seedPositions[0]);
      const double rhs = cachedCentroidSimilarity(storage, centroidCache, threadIdx.x, seedPositions[1]);
      affinities[threadIdx.x] = static_cast<std::int8_t>((lhs > rhs) - (lhs < rhs));
    } else if (threadIdx.x < scratch.count) {
      affinities[threadIdx.x] = seedAffinity(storage,
                                             scratch.nodeEntries[threadIdx.x],
                                             centroidFingerprint(storage, left),
                                             centroidFingerprint(storage, right));
    }
    __syncthreads();
    // Replay splitNodeWithSeeds' ordered, size-capped assignment on the staged
    // entries. Repair then relinks both halves without walking the leaf list.
    // The ordered halves are built in shared memory and copied out in parallel.
    if (threadIdx.x == 0) {
      const int size          = scratch.count;
      const int maxGroup      = (size + 1) / 2;
      int       lhsAssigned   = 1;
      int       rhsAssigned   = 1;
      scratch.entries[0]      = left;
      scratch.otherEntries[0] = right;
      for (int position = 0; position < size; ++position) {
        const int entry = scratch.nodeEntries[position];
        if (entry == left || entry == right) {
          continue;
        }
        bool assignLeft = affinities[position] > 0 || (affinities[position] == 0 && lhsAssigned <= rhsAssigned);
        if (lhsAssigned >= maxGroup) {
          assignLeft = false;
        } else if (rhsAssigned >= maxGroup) {
          assignLeft = true;
        }
        if (assignLeft) {
          scratch.entries[lhsAssigned++] = entry;
        } else {
          scratch.otherEntries[rhsAssigned++] = entry;
        }
      }
      scratch.stagedCount   = lhsAssigned;
      splitLeftSizes[index] = lhsAssigned;
    }
    __syncthreads();
    if (threadIdx.x < scratch.count) {
      const int leftSize = scratch.stagedCount;
      plan[threadIdx.x] =
        threadIdx.x < leftSize ? scratch.entries[threadIdx.x] : scratch.otherEntries[threadIdx.x - leftSize];
    }
  } else {
    if (threadIdx.x == 0) {
      splitLeftSizes[index] = -1;
    }
    int offset = 0;
    for (int entry = storage.nodeHeads[keys[index]]; entry >= 0; entry = storage.entryNext[entry], ++offset) {
      if (offset % blockDim.x == threadIdx.x) {
        splitAffinity[entry] =
          seedAffinity(storage, entry, centroidFingerprint(storage, left), centroidFingerprint(storage, right));
      }
    }
  }
  if (threadIdx.x == 0) {
    splitLeft[index]  = scratch.lhsSeed;
    splitRight[index] = scratch.rhsSeed;
  }
}

template <typename Component>
__global__ void bitBirchRepairKernel(const int              count,
                                     const int*             keys,
                                     const int*             splitLeft,
                                     const int*             splitRight,
                                     std::int8_t*           splitAffinity,
                                     const int*             splitPlans,
                                     const int*             splitLeftSizes,
                                     const int              branchingFactor,
                                     const int              leafDepth,
                                     const int*             levelNodes,
                                     const int*             levelCounts,
                                     const int              levelCapacity,
                                     int*                   deltaSlots,
                                     int*                   summaryNodes,
                                     int*                   summaryEntries,
                                     int*                   summaryCount,
                                     const bool             cacheCentroids,
                                     TreeStorage<Component> storage) {
  __shared__ CooperativeScratch scratch;
  __shared__ int                deferred;
  // Cascading inner-node splits search seeds from shared memory when it fits.
  extern __shared__ std::uint32_t centroidCache[];
  __shared__ int                queuedSplits;
  __shared__ int                warpSplits[cooperativeBlockSize / 32];
  __shared__ int                splitQueue[cooperativeBlockSize];
  // Refresh consumed this epoch's delta slots; restore them for the next one.
  for (int level = 1; level <= leafDepth; ++level) {
    for (int index = threadIdx.x; index < levelCounts[level]; index += blockDim.x) {
      deltaSlots[levelNodes[level * levelCapacity + index]] = -1;
    }
  }
  if (threadIdx.x == 0) {
    deferred = 0;
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  for (int base = 0; base < count; base += blockDim.x) {
    // A leaf's size changes only through its own split, so every eligibility
    // test in this window can run up front. Compact the overflowing leaves in
    // key order and pay barriers per split rather than per routed leaf.
    const int  candidate = base + threadIdx.x;
    const bool eligible  = candidate < count && keys[candidate] != INT_MAX &&
                          (candidate == 0 || keys[candidate - 1] != keys[candidate]) &&
                          storage.nodeSizes[keys[candidate]] > branchingFactor;
    const unsigned ballot = __ballot_sync(0xffffffffU, eligible);
    if (lane == 0) {
      warpSplits[warp] = __popc(ballot);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int total = 0;
      for (int index = 0; index < blockDim.x / warpSize; ++index) {
        const int splits  = warpSplits[index];
        warpSplits[index] = total;
        total += splits;
      }
      queuedSplits = total;
    }
    __syncthreads();
    if (eligible) {
      splitQueue[warpSplits[warp] + __popc(ballot & ((1U << lane) - 1U))] = candidate;
    }
    __syncthreads();
    for (int queued = 0; queued < queuedSplits && *storage.status == BitBirchStatus::Success; ++queued) {
      const int  index    = splitQueue[queued];
      const int  node     = keys[index];
      const int  size     = storage.nodeSizes[node];
      const int  leftSize = splitLeftSizes[index];
      const int* plan     = splitPlans + static_cast<std::size_t>(index) * cooperativeBlockSize;
      // Leaf topology changes stay ordered, but their two child BFs are only
      // read again by a cascading parent split. Defer them to a parallel pass.
      // Payload refresh already made every ancestor BF current, and a split
      // only redistributes members, so no ancestor above the parent changes.
      if (threadIdx.x == 0) {
        scratch.sibling = leftSize >= 0 ? allocateNode(storage, true, storage.nodeParents[node]) : *storage.nodeCursor;
      }
      __syncthreads();
      const int sibling = scratch.sibling;
      if (leftSize >= 0 && sibling >= 0) {
        int* nodeEntries    = nodeDirectory(storage, node);
        int* siblingEntries = nodeDirectory(storage, sibling);
        for (int position = threadIdx.x; position < size; position += blockDim.x) {
          const int last                    = position < leftSize ? leftSize - 1 : size - 1;
          storage.entryNext[plan[position]] = position == last ? -1 : plan[position + 1];
          if (position < leftSize) {
            nodeEntries[position] = plan[position];
          } else {
            siblingEntries[position - leftSize] = plan[position];
          }
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        int parent = -1;
        if (leftSize < 0) {
          parent =
            splitNodeWithSeeds(storage, node, branchingFactor, splitLeft[index], splitRight[index], splitAffinity);
        } else if (sibling >= 0) {
          storage.nodeHeads[node]    = plan[0];
          storage.nodeTails[node]    = plan[leftSize - 1];
          storage.nodeSizes[node]    = leftSize;
          storage.nodeHeads[sibling] = plan[leftSize];
          storage.nodeTails[sibling] = plan[size - 1];
          storage.nodeSizes[sibling] = size - leftSize;
          parent                     = linkSplitSibling(storage, node, sibling);
        }
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
        for (int pending = 0; pending < deferred; ++pending) {
          cooperativeSummarizeNode(storage, summaryNodes[pending], summaryEntries[pending], scratch);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          deferred = 0;
        }
        cooperativeSplitNode(storage,
                             parent,
                             branchingFactor,
                             splitAffinity,
                             scratch,
                             cacheCentroids ? centroidCache : nullptr);
      }
      __syncthreads();
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
    localPending += labelOf(storage, molecule) < 0;
  }
  atomicAdd(&pending, localPending);
  __syncthreads();
  if (threadIdx.x == 0) {
    int depth = 0;
    // Node zero remains a leaf; every leaf in this tree has equal depth.
    for (int node = 0; storage.nodeParents[node] >= 0; node = storage.nodeParents[node]) {
      ++depth;
    }
    control[PendingInputs] = pending;
    control[NodeCount]     = *storage.nodeCursor;
    control[SummaryCount]  = *storage.summaryCursor;
    control[TreeStatus]    = static_cast<int>(*storage.status);
    control[LeafDepth]     = depth;
    control[EntryCount]    = *storage.entryCursor;
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
    labelOf(storage, molecule) = storage.entryClusterIds[labelOf(storage, molecule)];
  }
}

//! Every non-root node has one directory entry, so leaf entries are E - V + 1.
int liveClusters(const PinnedHostVector<int>& control) {
  return control[EntryCount] - control[NodeCount] + 1;
}

template <typename Component>
BitBirchResult launchBitBirch(const cuda::std::span<const std::uint32_t> fingerprints,
                              const int                                  numFingerprints,
                              const int                                  numWords,
                              const double                               threshold,
                              const BitBirchOptions&                     options,
                              const cudaStream_t                         stream) {
  const int              branchingFactor    = options.branchingFactor;
  const bool             fingerprintsOnHost = options.fingerprintsOnHost;
  int* const             hostLabels         = options.hostClusterIds;
  const bool             returnCentroids    = options.returnCentroids;
  const int              batchCapacity      = std::min(options.batchSize, numFingerprints);
  const std::size_t      splitCacheBytes    = numWords <= 32 && branchingFactor < cooperativeBlockSize ?
                                                sizeof(std::uint32_t) * (cooperativeBlockSize + 1) * numWords :
                                                0;
  BitBirchResult         result{AsyncDeviceVector<int>(hostLabels != nullptr ? 0 : numFingerprints, stream),
                        AsyncDeviceVector<std::uint32_t>(0, stream),
                        0,
                        numWords};
  // Kernels only touch labels of the batch being inserted until finalization.
  // Host-labeled runs therefore keep one batch on the GPU and move each finished
  // batch of entry IDs through a small pinned buffer into the caller's array.
  AsyncDeviceVector<int> batchLabels(hostLabels != nullptr ? batchCapacity : 0, stream);
  PinnedHostVector<int>  labelStaging(hostLabels != nullptr ? batchCapacity : 0);
  int                    stagedBegin       = 0;
  int                    stagedCount       = 0;
  const auto             drainStagedLabels = [&]() {
    // Call only after a stream or event synchronization that follows the staging copy.
    std::copy_n(labelStaging.data(), stagedCount, hostLabels + stagedBegin);
    stagedCount = 0;
  };
  int*                             labels = hostLabels != nullptr ? batchLabels.data() : result.clusterIds.data();
  AsyncDeviceVector<std::uint32_t> inputTile(
    fingerprintsOnHost ? static_cast<std::size_t>(batchCapacity) * numWords : 0,
    stream);
  // Packed singletons retained from host input, independent of the current tile.
  PagedEntryStorage<std::uint32_t> ownedFingerprints(
    numWords,
    stream,
    options.fingerprintCacheBytes,
    "fingerprint_cache_bytes must fit at least one packed-fingerprint page");
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
                                branchingFactor,
                                numWords,
                                stream,
                                options.summaryCacheBytes);
  AsyncDeviceVector<int>         keys(batchCapacity, stream);
  AsyncDeviceVector<int>         sortedKeys(batchCapacity, stream);
  AsyncDeviceVector<int>         values(batchCapacity, stream);
  AsyncDeviceVector<int>         sortedValues(batchCapacity, stream);
  AsyncDeviceVector<int>         groupKeys(batchCapacity, stream);
  AsyncDeviceVector<int>         sortedGroupKeys(batchCapacity, stream);
  AsyncDeviceVector<int>         sortedGroupValues(batchCapacity, stream);
  AsyncDeviceVector<int>         splitLeft(batchCapacity, stream);
  AsyncDeviceVector<int>         splitRight(batchCapacity, stream);
  AsyncDeviceVector<std::int8_t> splitAffinity(tree.entryCapacity, stream);
  // Ordered halves for each overflowing leaf that fits one cooperative block.
  AsyncDeviceVector<int>         splitPlans(
    branchingFactor < cooperativeBlockSize ? static_cast<std::size_t>(batchCapacity) * cooperativeBlockSize : 0,
    stream);
  AsyncDeviceVector<int>           splitLeftSizes(batchCapacity, stream);
  // Staged leaves search split seeds across many blocks when every leaf's
  // centroids fit in shared memory and similarities are FP32 (see routingSimilarity).
  int device      = 0;
  int sharedOptin = 0;
  int numSMs      = 0;
  const std::size_t splitPairsCacheBytes = sizeof(std::uint32_t) * (cooperativeBlockSize + 1) * numWords;
  cudaCheckError(cudaGetDevice(&device));
  cudaCheckError(cudaDeviceGetAttribute(&sharedOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  cudaCheckError(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
  cudaFuncAttributes splitPairsAttributes{};
  cudaCheckError(cudaFuncGetAttributes(&splitPairsAttributes, bitBirchSplitPairsKernel<Component>));
  const bool parallelSplitSeeds = branchingFactor < cooperativeBlockSize && numWords * 32 <= 4096 &&
                                  splitPairsAttributes.sharedSizeBytes + splitPairsCacheBytes <=
                                    static_cast<std::size_t>(sharedOptin);
  int        splitPairsBlocks   = 0;
  if (parallelSplitSeeds) {
    cudaCheckError(cudaFuncSetAttribute(bitBirchSplitPairsKernel<Component>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(splitPairsCacheBytes)));
    cudaCheckError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&splitPairsBlocks,
                                                                 bitBirchSplitPairsKernel<Component>,
                                                                 cooperativeBlockSize,
                                                                 splitPairsCacheBytes));
    splitPairsBlocks = std::max(1, splitPairsBlocks) * numSMs;
  }
  // Inner nodes split inside the repair kernel hold at most one block of entries, like staged leaves.
  cudaFuncAttributes repairAttributes{};
  cudaCheckError(cudaFuncGetAttributes(&repairAttributes, bitBirchRepairKernel<Component>));
  const std::size_t repairCacheBytes =
    branchingFactor < cooperativeBlockSize &&
        repairAttributes.sharedSizeBytes + splitPairsCacheBytes <= static_cast<std::size_t>(sharedOptin) ?
      splitPairsCacheBytes :
      0;
  if (repairCacheBytes > 0) {
    cudaCheckError(cudaFuncSetAttribute(bitBirchRepairKernel<Component>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(repairCacheBytes)));
  }
  AsyncDeviceVector<int>                splitQueue(parallelSplitSeeds ? batchCapacity : 0, stream);
  AsyncDeviceVector<int>                splitQueueCount(1, stream);
  AsyncDeviceVector<unsigned long long> splitKeys(parallelSplitSeeds ? batchCapacity : 0, stream);
  if (parallelSplitSeeds) {
    cudaCheckError(cudaMemsetAsync(splitKeys.data(), 0xff, splitKeys.size() * sizeof(unsigned long long), stream));
  }
  AsyncDeviceVector<Component>     proposalSums(static_cast<std::size_t>(batchCapacity) * numWords * 32, stream);
  AsyncDeviceVector<int>           dirtyNodes(tree.nodeCapacity, stream);
  AsyncDeviceVector<int>           deltaSlots(tree.nodeCapacity, stream);
  AsyncDeviceVector<int>           deltaCursor(1, stream);
  AsyncDeviceVector<Component>     deltaSums(0, stream);
  AsyncDeviceVector<std::uint32_t> deltaCounts(0, stream);
  // Dirty nodes per tree level for one epoch. Each level holds at most one
  // node per distinct routed leaf, so capacity is batch size times height.
  AsyncDeviceVector<int>           levelNodes(0, stream);
  AsyncDeviceVector<int>           levelCounts(0, stream);
  // Each routed leaf splits at most once per epoch and defers two child BFs.
  AsyncDeviceVector<int>           summaryNodes(2 * static_cast<std::size_t>(batchCapacity), stream);
  AsyncDeviceVector<int>           summaryEntries(2 * static_cast<std::size_t>(batchCapacity), stream);
  AsyncDeviceVector<int>           summaryCount(1, stream);
  AsyncDeviceVector<int>           control(NumEpochControls, stream);
  // Pinned so the per-round readback is a true async copy that an event can
  // fence while the next round's scratch resets are already queued behind it.
  PinnedHostVector<int>            hostControl(NumEpochControls, 0);
  const ScopedCudaEvent            controlReady;
  cudaCheckError(cudaMemsetAsync(dirtyNodes.data(), 0, tree.nodeCapacity * sizeof(int), stream));
  cudaCheckError(cudaMemsetAsync(deltaSlots.data(), 0xff, tree.nodeCapacity * sizeof(int), stream));
  cudaCheckError(cudaMemsetAsync(deltaCursor.data(), 0, sizeof(int), stream));
  cudaCheckError(cudaMemsetAsync(splitQueueCount.data(), 0, sizeof(int), stream));
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
  hostControl[NodeCount] = 1;
  for (int begin = 0; begin < numFingerprints;) {
    ScopedNvtxRange batchRange("BitBIRCH batch", NvtxColor::kGrey);
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
    const int labelBegin = hostLabels != nullptr ? begin : 0;
    cudaCheckError(cudaMemsetAsync(labels + (begin - labelBegin), 0xff, count * sizeof(int), stream));
    int pending = count;
    while (pending > 0) {
      ScopedNvtxRange roundRange("BitBIRCH insertion round", NvtxColor::kCyan);
      ScopedNvtxRange workspaceRange("BitBIRCH workspace growth", NvtxColor::kGrey);
      // Every non-root node has exactly one directory entry, so K = E - V + 1.
      // At most pending new leaf entries can be created before the next barrier.
      // Apply the same balanced-node bound to live K instead of allocating for
      // the all-singleton N worst case. Stable integer IDs survive buffer growth.
      const int clusterBound    = liveClusters(hostControl) + pending;
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
      const int leafDepth = hostControl[LeafDepth];
      if (levelCounts.size() < static_cast<std::size_t>(leafDepth) + 1) {
        levelCounts.resize(leafDepth + 1);
        levelNodes.resize(static_cast<std::size_t>(leafDepth + 1) * batchCapacity);
        // Later rounds reset level counts behind the control readback.
        cudaCheckError(cudaMemsetAsync(levelCounts.data(), 0, levelCounts.size() * sizeof(int), stream));
      }
      if (splitAffinity.size() < static_cast<std::size_t>(tree.entryCapacity)) {
        splitAffinity.resize(tree.entryCapacity);
      }
      // Bound lazy materialization plus two summaries per possible split level.
      const auto required = static_cast<std::int64_t>(hostControl[SummaryCount]) +
                            static_cast<std::int64_t>(pending) * (2 * hostControl[LeafDepth] + 5) + 4;
      if (required > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("BitBIRCH summary workspace exceeds index capacity");
      }
      tree.summaryArena.reserve(static_cast<int>(required));
      // At most one ancestor per depth for each pending molecule. Scratch is
      // bounded by batch size times tree height, not by total cluster count.
      const auto deltaCapacity = std::min(static_cast<std::size_t>(tree.nodeCapacity),
                                          static_cast<std::size_t>(pending) * hostControl[LeafDepth]);
      if (deltaCounts.size() < deltaCapacity) {
        deltaCounts.resize(deltaCapacity);
        deltaSums.resize(deltaCapacity * numWords * 32);
      }
      storage            = tree.storage();
      storage.labelBegin = labelBegin;
      if (fingerprintsOnHost) {
        ownedFingerprints.reserve(tree.entryCapacity);
        storage.singletonFingerprintPages = ownedFingerprints.pointers();
        storage.singletonPageHits         = ownedFingerprints.pageHits();
        storage.queryBegin                = begin;
      }
      workspaceRange.pop();
      bitBirchRouteKernel<<<count, cooperativeBlockSize, 0, stream>>>(begin,
                                                                      count,
                                                                      keys.data(),
                                                                      values.data(),
                                                                      groupKeys.data(),
                                                                      enableGroups,
                                                                      threshold,
                                                                      storage);
      cudaCheckError(cudaGetLastError());
      // Keys are node or entry IDs below the current capacity, or INT_MAX. With
      // 2^bits > capacity, INT_MAX's low bits exceed every ID, so the stable sort
      // over only those bits yields the same permutation as a full 32-bit sort.
      const auto keyBits = [](const int capacity) {
        int bits = 1;
        while (bits < 32 && (1LL << bits) <= capacity) {
          ++bits;
        }
        return bits;
      };
      if (enableGroups) {
        cudaCheckError(cub::DeviceRadixSort::SortPairs(sortScratch.data(),
                                                       sortBytes,
                                                       groupKeys.data(),
                                                       sortedGroupKeys.data(),
                                                       values.data(),
                                                       sortedGroupValues.data(),
                                                       count,
                                                       0,
                                                       keyBits(tree.entryCapacity),
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
                                                     keyBits(tree.nodeCapacity),
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
                                                                           splitPlans.data(),
                                                                           splitQueue.data(),
                                                                           splitQueueCount.data(),
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
      if (parallelSplitSeeds) {
        bitBirchSplitPairsKernel<<<splitPairsBlocks, cooperativeBlockSize, splitPairsCacheBytes, stream>>>(
          sortedKeys.data(),
          branchingFactor,
          splitQueue.data(),
          splitQueueCount.data(),
          splitPlans.data(),
          splitKeys.data(),
          storage);
        cudaCheckError(cudaGetLastError());
      }
      bitBirchSplitSeedsKernel<<<count, cooperativeBlockSize, parallelSplitSeeds ? 0 : splitCacheBytes, stream>>>(count,
                                                                                         sortedKeys.data(),
                                                                                         branchingFactor,
                                                                                         splitLeft.data(),
                                                                                         splitRight.data(),
                                                                                         splitAffinity.data(),
                                                                                         splitPlans.data(),
                                                                                         splitLeftSizes.data(),
                                                                                         splitKeys.data(),
                                                                                         storage);
      cudaCheckError(cudaGetLastError());
      bitBirchRepairKernel<<<1, cooperativeBlockSize, repairCacheBytes, stream>>>(count,
                                                                   sortedKeys.data(),
                                                                   splitLeft.data(),
                                                                   splitRight.data(),
                                                                   splitAffinity.data(),
                                                                   splitPlans.data(),
                                                                   splitLeftSizes.data(),
                                                                   branchingFactor,
                                                                   leafDepth,
                                                                   levelNodes.data(),
                                                                   levelCounts.data(),
                                                                   batchCapacity,
                                                                   deltaSlots.data(),
                                                                   summaryNodes.data(),
                                                                   summaryEntries.data(),
                                                                   summaryCount.data(),
                                                                   repairCacheBytes > 0,
                                                                   storage);
      cudaCheckError(cudaGetLastError());
      bitBirchSummarizeKernel<<<2 * count, cooperativeBlockSize, 0, stream>>>(summaryNodes.data(),
                                                                              summaryEntries.data(),
                                                                              summaryCount.data(),
                                                                              storage);
      cudaCheckError(cudaGetLastError());
      ScopedNvtxRange controlRange("BitBIRCH control + host sync", NvtxColor::kGrey);
      bitBirchControlKernel<<<1, cooperativeBlockSize, 0, stream>>>(begin, count, control.data(), storage);
      cudaCheckError(cudaGetLastError());
      cudaCheckError(cudaMemcpyAsync(hostControl.data(),
                                     control.data(),
                                     NumEpochControls * sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     stream));
      cudaCheckError(cudaEventRecord(controlReady.event(), stream));
      // Reset the next round's per-round scratch while the host waits on control.
      cudaCheckError(cudaMemsetAsync(levelCounts.data(), 0, levelCounts.size() * sizeof(int), stream));
      cudaCheckError(cudaMemsetAsync(deltaCursor.data(), 0, sizeof(int), stream));
      cudaCheckError(cudaMemsetAsync(splitQueueCount.data(), 0, sizeof(int), stream));
      cudaCheckError(cudaEventSynchronize(controlReady.event()));
      if (stagedCount > 0) {
        drainStagedLabels();
      }
      if (hostControl[TreeStatus] != static_cast<int>(BitBirchStatus::Success)) {
        throw std::runtime_error("BitBIRCH tree failure (status " + std::to_string(hostControl[TreeStatus]) + ")");
      }
      if (hostControl[PendingInputs] >= pending) {
        throw std::runtime_error("BitBIRCH insertion failed to make progress");
      }
      pending = hostControl[PendingInputs];
    }
    if (hostLabels != nullptr) {
      // The next epoch's synchronization completes this copy before draining.
      cudaCheckError(cudaMemcpyAsync(labelStaging.data(),
                                     batchLabels.data(),
                                     static_cast<std::size_t>(count) * sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     stream));
      stagedBegin = begin;
      stagedCount = count;
    }
    begin += count;
    tree.summaryArena.rotate();
    ownedFingerprints.rotate();
  }
  ScopedNvtxRange finalizeRange("BitBIRCH finalize labels", NvtxColor::kGreen);
  if (returnCentroids) {
    const int clusters = liveClusters(hostControl);
    result.centroids.resize(static_cast<std::size_t>(clusters) * numWords);
  }
  storage.centroids                  = returnCentroids ? result.centroids.data() : nullptr;
  // Entry owners retain their minimum input index. Sort only the K live leaf
  // entries by that index to preserve first-member label numbering without an N-wide
  // first-member scan and its additional device workspace.
  const int              clusters    = liveClusters(hostControl);
  const int              liveEntries = hostControl[EntryCount];
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
  int clusterCount = 0;
  if (hostLabels == nullptr) {
    const int labelBlocks = (numFingerprints + cooperativeBlockSize - 1) / cooperativeBlockSize;
    bitBirchFinalizeKernel<<<labelBlocks, cooperativeBlockSize, 0, stream>>>(numFingerprints, storage);
    cudaCheckError(cudaGetLastError());
    tree.clusterCount.get(clusterCount);
    cudaCheckError(cudaStreamSynchronize(stream));
  } else {
    // Host labels hold leaf entry IDs; map them to first-member cluster order.
    std::vector<int> entryClusterIds(liveEntries);
    tree.entryClusterIds.copyToHost(entryClusterIds, liveEntries);
    tree.clusterCount.get(clusterCount);
    cudaCheckError(cudaStreamSynchronize(stream));
    drainStagedLabels();
    for (int molecule = 0; molecule < numFingerprints; ++molecule) {
      hostLabels[molecule] = entryClusterIds[hostLabels[molecule]];
    }
  }
  result.numClusters = clusterCount;
  return result;
}

}  // namespace

BitBirchResult bitBirchGpu(const cuda::std::span<const std::uint32_t> fingerprints,
                           const int                                  numFingerprints,
                           const int                                  numWords,
                           const double                               threshold,
                           const BitBirchOptions&                     options,
                           const cudaStream_t                         stream) {
  const ScopedNvtxRange range("BitBIRCH tree construction");
  const int             branchingFactor       = options.branchingFactor;
  const std::size_t     fingerprintCacheBytes = options.fingerprintCacheBytes;
  if (numFingerprints < 0 || numFingerprints > std::numeric_limits<int>::max() - cooperativeBlockSize ||
      numWords <= 0 || numWords > std::numeric_limits<int>::max() / 32 ||
      fingerprints.size() != static_cast<std::size_t>(numFingerprints) * numWords) {
    throw std::invalid_argument("BitBIRCH fingerprints shape or dimensions are invalid");
  }
  if (!std::isfinite(threshold) || threshold < 0 || threshold > 1 || branchingFactor < 3 || options.batchSize < 1 ||
      (fingerprintCacheBytes > 0 &&
       (!options.fingerprintsOnHost ||
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
    return result;
  }
  if (numFingerprints <= std::numeric_limits<std::uint16_t>::max()) {
    return launchBitBirch<std::uint16_t>(fingerprints, numFingerprints, numWords, threshold, options, stream);
  }
  return launchBitBirch<std::uint32_t>(fingerprints, numFingerprints, numWords, threshold, options, stream);
}

}  // namespace nvMolKit
