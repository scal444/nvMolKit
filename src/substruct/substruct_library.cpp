// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/substruct/substruct_library.h"

#include <GraphMol/ROMol.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "src/substruct/molecules.h"
#include "src/substruct/substruct_search.h"
#include "src/substruct/target_chunk.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "src/utils/nvtx.h"
#include "src/utils/openmp_helpers.h"

namespace nvMolKit {

namespace {

bool rdkitHasMatch(const RDKit::ROMol& target, const RDKit::ROMol& query) {
  RDKit::SubstructMatchParameters params;
  params.uniquify             = false;
  params.maxMatches           = 1;
  params.useChirality         = false;
  params.useQueryQueryMatches = false;
  return !RDKit::SubstructMatch(target, query, params).empty();
}

}  // namespace

class SubstructLibrary::Impl {
 public:
  Impl(std::size_t chunkSize, SubstructSearchConfig config) : chunkSize_(chunkSize), config_(std::move(config)) {
    if (chunkSize_ == 0) {
      throw std::invalid_argument("Substructure library chunk size must be greater than zero");
    }
    if (config_.algorithm == SubstructAlgorithm::VF2) {
      throw std::invalid_argument("Substructure libraries support the GSI and DFS production backends");
    }
    std::unordered_set<int> seenDevices;
    for (const int deviceId : config_.gpuIds) {
      if (deviceId < 0) {
        throw std::invalid_argument("Substructure library GPU ID must be nonnegative");
      }
      if (!seenDevices.insert(deviceId).second) {
        throw std::invalid_argument("Substructure library GPU IDs must be unique");
      }
    }
    deviceIds_ = config_.gpuIds;
    builder_   = std::make_unique<TargetChunkBuilder>(0, chunkSize_);
  }

  ~Impl() noexcept {
    destroyWorkspaces();
    destroyDeviceSets(deviceSets_);
    chunks_.clear();
    pendingChunks_.clear();
  }

  unsigned int addMol(const RDKit::ROMol& molecule) {
    auto lock = writeLock();
    if (nextId_ > std::numeric_limits<unsigned int>::max()) {
      throw std::overflow_error("Substructure library molecule ID space exhausted");
    }
    if (builder_->full()) {
      sealBuilder();
    }
    const MoleculeId id = builder_->addMol(molecule);
    ++nextId_;
    return static_cast<unsigned int>(id);
  }

  std::vector<unsigned int> addMols(const std::vector<const RDKit::ROMol*>& molecules) {
    auto lock = writeLock();
    if (molecules.empty()) {
      return {};
    }
    constexpr auto maxId = static_cast<MoleculeId>(std::numeric_limits<unsigned int>::max());
    if (nextId_ > maxId || molecules.size() > maxId - nextId_ + 1) {
      throw std::overflow_error("Substructure library molecule ID space exhausted");
    }

    // Keep every sealed chunk in global-ID order. Sealing a partially filled
    // builder here is harmless: it remains pending and a failed bulk build can
    // be retried without changing any published generation.
    sealBuilder();

    const std::size_t                         numChunks = (molecules.size() + chunkSize_ - 1) / chunkSize_;
    std::vector<std::unique_ptr<TargetChunk>> builtChunks(numChunks);
    const int                                 numThreads = constructionThreads();
    for (std::size_t chunkIndex = 0; chunkIndex < numChunks; ++chunkIndex) {
      const std::size_t                      begin = chunkIndex * chunkSize_;
      const std::size_t                      end   = std::min(molecules.size(), begin + chunkSize_);
      const std::vector<const RDKit::ROMol*> chunkMolecules(molecules.begin() + static_cast<std::ptrdiff_t>(begin),
                                                            molecules.begin() + static_cast<std::ptrdiff_t>(end));
      TargetChunkBuilder                     chunkBuilder(nextId_ + begin, end - begin);
      chunkBuilder.addMols(chunkMolecules, numThreads);
      builtChunks[chunkIndex] = chunkBuilder.seal();
    }

    std::vector<unsigned int> ids(molecules.size());
    for (std::size_t index = 0; index < molecules.size(); ++index) {
      ids[index] = static_cast<unsigned int>(nextId_ + index);
    }
    for (auto& chunk : builtChunks) {
      pendingChunks_.push_back(std::move(chunk));
    }
    nextId_ += molecules.size();
    builder_ = std::make_unique<TargetChunkBuilder>(nextId_, chunkSize_);
    return ids;
  }

  void finalize(cudaStream_t stream) {
    auto lock = writeLock();
    resolveDevices();
    deviceSets_.resize(deviceIds_.size());
    if (deviceIds_.size() > 1 && stream != nullptr) {
      throw std::invalid_argument("A single external CUDA stream cannot be used with a multi-GPU substructure library");
    }
    int deviceCount = 0;
    cudaCheckError(cudaGetDeviceCount(&deviceCount));
    for (const int deviceId : deviceIds_) {
      if (deviceId >= deviceCount) {
        throw std::invalid_argument("Substructure library GPU ID is not available");
      }
    }

    sealBuilder();

    // Place each new chunk on the device holding the fewest molecules so far.
    std::vector<std::size_t> deviceLoad(deviceIds_.size(), 0);
    for (const auto& record : chunks_) {
      deviceLoad[record.deviceIndex] += record.chunk->size();
    }
    std::vector<std::size_t> pendingDevices(pendingChunks_.size());
    for (std::size_t index = 0; index < pendingChunks_.size(); ++index) {
      pendingDevices[index] =
        static_cast<std::size_t>(std::min_element(deviceLoad.begin(), deviceLoad.end()) - deviceLoad.begin());
      deviceLoad[pendingDevices[index]] += pendingChunks_[index]->size();
    }

    // Extend each device's target set with its new chunks. The published
    // sets stay untouched until everything succeeds, so failures are retryable.
    std::vector<std::unique_ptr<DeviceTargetSet>> newSets(deviceIds_.size());
    detail::OpenMPExceptionRegistry               uploadExceptions;
#pragma omp parallel for num_threads(static_cast<int>(deviceIds_.size())) schedule(static)
    for (std::int64_t deviceIndex = 0; deviceIndex < static_cast<std::int64_t>(deviceIds_.size()); ++deviceIndex) {
      try {
        const auto                      index = static_cast<std::size_t>(deviceIndex);
        const WithDevice                device(deviceIds_[index]);
        std::vector<const TargetChunk*> deviceChunks;
        for (std::size_t chunk = 0; chunk < pendingChunks_.size(); ++chunk) {
          if (pendingDevices[chunk] == index) {
            deviceChunks.push_back(pendingChunks_[chunk].get());
          }
        }
        if (!deviceChunks.empty()) {
          newSets[index] = std::make_unique<DeviceTargetSet>(deviceSets_[index].get(), deviceChunks);
          newSets[index]->upload(validateStream(stream));
        }
      } catch (...) {
        uploadExceptions.store(std::current_exception());
      }
    }
    try {
      uploadExceptions.rethrow();
      chunks_.reserve(chunks_.size() + pendingChunks_.size());
      ensureWorkspaces();
    } catch (...) {
      destroyDeviceSets(newSets);
      throw;
    }

    for (std::size_t index = 0; index < deviceIds_.size(); ++index) {
      if (newSets[index] != nullptr) {
        std::vector<std::unique_ptr<DeviceTargetSet>> replaced(deviceIds_.size());
        replaced[index] = std::move(deviceSets_[index]);
        destroyDeviceSets(replaced);
        deviceSets_[index] = std::move(newSets[index]);
      }
    }
    for (std::size_t index = 0; index < pendingChunks_.size(); ++index) {
      committedSize_ += pendingChunks_[index]->size();
      pendingChunks_[index]->releasePackedData();
      chunks_.push_back(DeviceChunk{pendingDevices[index], std::move(pendingChunks_[index])});
    }
    pendingChunks_.clear();
    published_ = true;
  }

  [[nodiscard]] std::size_t size() const {
    auto lock = readLock();
    return committedSize_;
  }

  [[nodiscard]] std::size_t pendingSize() const {
    auto lock = readLock();
    return static_cast<std::size_t>(nextId_) - committedSize_;
  }

  [[nodiscard]] std::vector<unsigned int> getMatches(const RDKit::ROMol& query,
                                                     int                 maxResults,
                                                     cudaStream_t        stream) const {
    auto lock = readLock();
    requirePublished();
    if (maxResults == 0 || chunks_.empty()) {
      return {};
    }

    std::vector<unsigned int> matches;
    if (maxResults > 0) {
      matches.reserve(std::min<std::size_t>(committedSize_, static_cast<std::size_t>(maxResults)));
    }

    const auto perDeviceMatches = matchingIdsByDevice(query, stream, maxResults);
    for (const auto& deviceMatches : perDeviceMatches) {
      matches.insert(matches.end(), deviceMatches.begin(), deviceMatches.end());
    }
    std::sort(matches.begin(), matches.end());
    if (maxResults > 0 && matches.size() > static_cast<std::size_t>(maxResults)) {
      matches.resize(static_cast<std::size_t>(maxResults));
    }
    return matches;
  }

  [[nodiscard]] std::size_t countMatches(const RDKit::ROMol& query, cudaStream_t stream) const {
    auto lock = readLock();
    requirePublished();
    std::size_t count = 0;
    for (const auto& deviceMatches : matchingIdsByDevice(query, stream, -1)) {
      count += deviceMatches.size();
    }
    return count;
  }

  [[nodiscard]] bool hasMatch(const RDKit::ROMol& query, cudaStream_t stream) const {
    auto lock = readLock();
    requirePublished();
    for (const auto& deviceMatches : matchingIdsByDevice(query, stream, 1)) {
      if (!deviceMatches.empty()) {
        return true;
      }
    }
    return false;
  }

 private:
  struct DeviceChunk {
    std::size_t                  deviceIndex;
    std::unique_ptr<TargetChunk> chunk;
  };

  // Query state on one device, reused by every query.
  struct QueryWorkspace {
    std::shared_ptr<SubstructSearchWorkspace> search;
    // Per-target match flags, reused across queries to avoid reallocating them.
    std::vector<std::uint8_t>                 gpuMatches;
  };

  // Every call holds the library lock, so queries, additions, and finalize() run one at a time.
  [[nodiscard]] std::unique_lock<std::mutex> writeLock() const { return std::unique_lock(mutex_); }
  [[nodiscard]] std::unique_lock<std::mutex> readLock() const { return std::unique_lock(mutex_); }

  // Move the current builder's molecules, if any, to the pending chunks and start a fresh builder at nextId_.
  void sealBuilder() {
    if (builder_->empty()) {
      return;
    }
    auto nextBuilder = std::make_unique<TargetChunkBuilder>(nextId_, chunkSize_);
    pendingChunks_.reserve(pendingChunks_.size() + 1);
    pendingChunks_.push_back(builder_->seal());
    builder_ = std::move(nextBuilder);
  }

  [[nodiscard]] int constructionThreads() const {
    return config_.preprocessingThreads == -1 ? omp_get_max_threads() : std::max(1, config_.preprocessingThreads);
  }

  void resolveDevices() {
    if (deviceIds_.empty()) {
      int currentDevice = 0;
      cudaCheckError(cudaGetDevice(&currentDevice));
      deviceIds_.push_back(currentDevice);
      config_.gpuIds = deviceIds_;
    }
  }

  [[nodiscard]] cudaStream_t validateStream(cudaStream_t stream) const {
    const auto validated = acquireExternalStream(reinterpret_cast<std::uintptr_t>(stream));
    if (!validated.has_value()) {
      throw std::invalid_argument("CUDA stream does not belong to the substructure library device");
    }
    return *validated;
  }

  void requirePublished() const {
    if (!published_) {
      throw std::logic_error("Substructure library must be finalized before querying");
    }
  }

  [[nodiscard]] SubstructSearchConfig deviceConfig(std::size_t deviceIndex) const {
    SubstructSearchConfig result = config_;
    result.gpuIds                = {deviceIds_[deviceIndex]};
    if (config_.preprocessingThreads == -1) {
      // Persistent targets and a single query do not benefit from spawning a
      // machine-wide preprocessing team for every asynchronous request.
      result.preprocessingThreads = 1;
    } else {
      result.preprocessingThreads = std::max(1, config_.preprocessingThreads / static_cast<int>(deviceIds_.size()));
    }
    if (result.workerThreads == -1) {
      result.workerThreads = 1;
    }
    return result;
  }

  // Create one query workspace per device on first use; they live as long as the library.
  void ensureWorkspaces() {
    if (workspaces_.size() == deviceIds_.size()) {
      return;
    }
    std::vector<std::unique_ptr<QueryWorkspace>> created(deviceIds_.size());
    for (std::size_t index = 0; index < deviceIds_.size(); ++index) {
      const WithDevice device(deviceIds_[index]);
      created[index]         = std::make_unique<QueryWorkspace>();
      created[index]->search = makeSubstructSearchWorkspace(deviceIds_[index]);
    }
    workspaces_ = std::move(created);
  }

  [[nodiscard]] std::vector<std::vector<unsigned int>> matchingIdsByDevice(const RDKit::ROMol& query,
                                                                           cudaStream_t        stream,
                                                                           int                 maxResults) const {
    if (deviceIds_.size() > 1 && stream != nullptr) {
      throw std::invalid_argument("A single external CUDA stream cannot be used with a multi-GPU substructure library");
    }

    std::vector<std::vector<unsigned int>> results(deviceIds_.size());
    detail::OpenMPExceptionRegistry        exceptionRegistry;

#pragma omp parallel for num_threads(static_cast<int>(deviceIds_.size())) schedule(static)
    for (std::int64_t deviceIndex = 0; deviceIndex < static_cast<std::int64_t>(deviceIds_.size()); ++deviceIndex) {
      try {
        const auto       index = static_cast<std::size_t>(deviceIndex);
        const WithDevice device(deviceIds_[index]);
        cudaStream_t     deviceStream = validateStream(stream);
        const auto       localConfig  = deviceConfig(index);
        results[index] = deviceMatchingIds(index, query, deviceStream, localConfig, *workspaces_[index], maxResults);
      } catch (...) {
        exceptionRegistry.store(std::current_exception());
      }
    }
    exceptionRegistry.rethrow();
    return results;
  }

  // Matching IDs on one device in ascending order, truncated to maxResults when positive.
  [[nodiscard]] std::vector<unsigned int> deviceMatchingIds(std::size_t                  deviceIndex,
                                                            const RDKit::ROMol&          query,
                                                            cudaStream_t                 stream,
                                                            const SubstructSearchConfig& config,
                                                            QueryWorkspace&              workspace,
                                                            int                          maxResults) const {
    std::vector<unsigned int> matches;
    const DeviceTargetSet*    targets = deviceSets_[deviceIndex].get();
    if (targets != nullptr && targets->size() != 0) {
      ScopedNvtxRange            matchRange("SubstructLibrary match targets");
      std::vector<std::uint8_t>& gpuMatches = workspace.gpuMatches;
      hasSubstructMatch(targets->persistentTargets(),
                        query,
                        gpuMatches,
                        config.algorithm,
                        stream,
                        config,
                        workspace.search.get());
      if (gpuMatches.size() != targets->size()) {
        throw std::runtime_error("Persistent-target search result size does not match its target set");
      }
      const auto& ids = targets->ids();
      for (std::size_t target = 0; target < gpuMatches.size(); ++target) {
        if (gpuMatches[target] != 0) {
          matches.push_back(static_cast<unsigned int>(ids[target]));
        }
      }
    }

    // RDKit fallback targets. With a result limit already filled by GPU
    // matches, only fallback IDs below the limit's last match can change it.
    const bool        limitFilled   = maxResults > 0 && matches.size() >= static_cast<std::size_t>(maxResults);
    const auto        fallbackBound = limitFilled ?
                                        static_cast<MoleculeId>(matches[static_cast<std::size_t>(maxResults) - 1]) :
                                        std::numeric_limits<MoleculeId>::max();
    const std::size_t gpuMatchCount = matches.size();
    ScopedNvtxRange   fallbackRange("SubstructLibrary RDKit fallback targets");
    for (const auto& record : chunks_) {
      if (record.deviceIndex != deviceIndex) {
        continue;
      }
      for (const MoleculeId id : record.chunk->fallbackGlobalIds()) {
        if (id >= fallbackBound) {
          break;
        }
        if (rdkitHasMatch(record.chunk->sourceMol(id), query)) {
          matches.push_back(static_cast<unsigned int>(id));
        }
      }
    }
    if (matches.size() != gpuMatchCount) {
      std::inplace_merge(matches.begin(), matches.begin() + static_cast<std::ptrdiff_t>(gpuMatchCount), matches.end());
    }
    if (maxResults > 0 && matches.size() > static_cast<std::size_t>(maxResults)) {
      matches.resize(static_cast<std::size_t>(maxResults));
    }
    return matches;
  }

  // For each of the first count configured devices i where holds(i), call release(i) with device i current, then
  // restore the caller's device. Devices without resources are never selected, so a device ID rejected by
  // finalize() leaves no CUDA error behind.
  template <typename Holds, typename Release>
  void releaseOnDevices(std::size_t count, const Holds& holds, const Release& release) const noexcept {
    int originalDevice = -1;
    cudaGetDevice(&originalDevice);
    for (std::size_t index = 0; index < std::min(count, deviceIds_.size()); ++index) {
      if (holds(index) && cudaSetDevice(deviceIds_[index]) == cudaSuccess) {
        release(index);
      }
    }
    if (originalDevice >= 0) {
      cudaSetDevice(originalDevice);
    }
  }

  void destroyDeviceSets(std::vector<std::unique_ptr<DeviceTargetSet>>& sets) noexcept {
    releaseOnDevices(
      sets.size(),
      [&](std::size_t index) { return sets[index] != nullptr; },
      [&](std::size_t index) { sets[index].reset(); });
  }

  void destroyWorkspaces() noexcept {
    releaseOnDevices(
      workspaces_.size(),
      [&](std::size_t index) { return workspaces_[index] != nullptr; },
      [&](std::size_t index) { workspaces_[index].reset(); });
    workspaces_.clear();
  }

  const std::size_t                                    chunkSize_;
  SubstructSearchConfig                                config_;
  mutable std::mutex                                   mutex_;
  std::unique_ptr<TargetChunkBuilder>                  builder_;
  std::vector<std::unique_ptr<TargetChunk>>            pendingChunks_;
  std::vector<DeviceChunk>                             chunks_;
  std::vector<std::unique_ptr<DeviceTargetSet>>        deviceSets_;
  std::vector<int>                                     deviceIds_;
  mutable std::vector<std::unique_ptr<QueryWorkspace>> workspaces_;
  MoleculeId                                           nextId_        = 0;
  std::size_t                                          committedSize_ = 0;
  bool                                                 published_     = false;
};

SubstructLibrary::SubstructLibrary(std::size_t chunkSize, SubstructSearchConfig config)
    : impl_(std::make_unique<Impl>(chunkSize, std::move(config))) {}

SubstructLibrary::~SubstructLibrary() = default;

unsigned int SubstructLibrary::addMol(const RDKit::ROMol& molecule) {
  return impl_->addMol(molecule);
}

std::vector<unsigned int> SubstructLibrary::addMols(const std::vector<const RDKit::ROMol*>& molecules) {
  return impl_->addMols(molecules);
}

void SubstructLibrary::finalize(cudaStream_t stream) {
  impl_->finalize(stream);
}

std::size_t SubstructLibrary::size() const {
  return impl_->size();
}

std::size_t SubstructLibrary::pendingSize() const {
  return impl_->pendingSize();
}

std::vector<unsigned int> SubstructLibrary::getMatches(const RDKit::ROMol& query,
                                                       int                 maxResults,
                                                       cudaStream_t        stream) const {
  return impl_->getMatches(query, maxResults, stream);
}

std::size_t SubstructLibrary::countMatches(const RDKit::ROMol& query, cudaStream_t stream) const {
  return impl_->countMatches(query, stream);
}

bool SubstructLibrary::hasMatch(const RDKit::ROMol& query, cudaStream_t stream) const {
  return impl_->hasMatch(query, stream);
}

}  // namespace nvMolKit
