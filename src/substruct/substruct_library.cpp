// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/substruct/substruct_library.h"

#include <DataStructs/BitOps.h>
#include <DataStructs/ExplicitBitVect.h>
#include <GraphMol/Fingerprints/Fingerprints.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "src/substruct/pattern_screen.h"
#include "src/substruct/resident_target_chunk.h"
#include "src/substruct/substruct_search.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "src/utils/nvtx.h"
#include "src/utils/openmp_helpers.h"
#include "src/utils/thread_safe_queue.h"

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
  Impl(std::size_t chunkSize, SubstructSearchConfig config, bool usePatternFingerprints)
      : chunkSize_(chunkSize),
        config_(std::move(config)),
        usePatternFingerprints_(usePatternFingerprints) {
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
    builder_   = std::make_unique<TargetChunkBuilder>(0, chunkSize_, usePatternFingerprints_);
  }

  ~Impl() noexcept {
    destroyWorkspaces();
    destroyDeviceSets(deviceSets_);
    chunks_.clear();
    pendingChunks_.clear();
  }

  unsigned int addMol(const RDKit::ROMol& molecule) {
    std::unique_lock lock(mutex_);
    if (nextId_ > std::numeric_limits<unsigned int>::max()) {
      throw std::overflow_error("Substructure library molecule ID space exhausted");
    }
    if (builder_->full()) {
      auto nextBuilder = std::make_unique<TargetChunkBuilder>(nextId_, chunkSize_, usePatternFingerprints_);
      pendingChunks_.reserve(pendingChunks_.size() + 1);
      pendingChunks_.push_back(builder_->seal());
      builder_ = std::move(nextBuilder);
    }
    const MoleculeId id = builder_->addMol(molecule);
    ++nextId_;
    return static_cast<unsigned int>(id);
  }

  std::vector<unsigned int> addMols(const std::vector<const RDKit::ROMol*>& molecules) {
    std::unique_lock lock(mutex_);
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
    if (!builder_->empty()) {
      pendingChunks_.push_back(builder_->seal());
      builder_ = std::make_unique<TargetChunkBuilder>(nextId_, chunkSize_, usePatternFingerprints_);
    }

    const std::size_t                                 numChunks = (molecules.size() + chunkSize_ - 1) / chunkSize_;
    std::vector<std::unique_ptr<ResidentTargetChunk>> builtChunks(numChunks);
    const int                                         numThreads = constructionThreads();
    for (std::size_t chunkIndex = 0; chunkIndex < numChunks; ++chunkIndex) {
      const std::size_t                      begin = chunkIndex * chunkSize_;
      const std::size_t                      end   = std::min(molecules.size(), begin + chunkSize_);
      const std::vector<const RDKit::ROMol*> chunkMolecules(molecules.begin() + static_cast<std::ptrdiff_t>(begin),
                                                            molecules.begin() + static_cast<std::ptrdiff_t>(end));
      TargetChunkBuilder                     chunkBuilder(nextId_ + begin, end - begin, usePatternFingerprints_);
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
    builder_ = std::make_unique<TargetChunkBuilder>(nextId_, chunkSize_, usePatternFingerprints_);
    return ids;
  }

  void finalize(cudaStream_t stream) {
    std::unique_lock lock(mutex_);
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

    if (!builder_->empty()) {
      auto nextBuilder = std::make_unique<TargetChunkBuilder>(nextId_, chunkSize_, usePatternFingerprints_);
      pendingChunks_.reserve(pendingChunks_.size() + 1);
      pendingChunks_.push_back(builder_->seal());
      builder_ = std::move(nextBuilder);
    }

    std::vector<std::size_t> pendingDevices(pendingChunks_.size());
    for (std::size_t index = 0; index < pendingChunks_.size(); ++index) {
      pendingDevices[index] = (nextDeviceAssignment_ + index) % deviceIds_.size();
    }

    // Extend each device's resident set with its new chunks. The published
    // sets stay untouched until everything succeeds, so failures are retryable.
    std::vector<std::unique_ptr<DeviceTargetSet>> newSets(deviceIds_.size());
    detail::OpenMPExceptionRegistry               uploadExceptions;
#pragma omp parallel for num_threads(static_cast<int>(deviceIds_.size())) schedule(static)
    for (std::int64_t deviceIndex = 0; deviceIndex < static_cast<std::int64_t>(deviceIds_.size()); ++deviceIndex) {
      try {
        const auto                              index = static_cast<std::size_t>(deviceIndex);
        const WithDevice                        device(deviceIds_[index]);
        std::vector<const ResidentTargetChunk*> deviceChunks;
        for (std::size_t chunk = 0; chunk < pendingChunks_.size(); ++chunk) {
          if (pendingDevices[chunk] == index) {
            deviceChunks.push_back(pendingChunks_[chunk].get());
          }
        }
        if (!deviceChunks.empty()) {
          newSets[index] =
            std::make_unique<DeviceTargetSet>(deviceSets_[index].get(), deviceChunks, usePatternFingerprints_);
          newSets[index]->upload(validateStream(stream));
        }
      } catch (...) {
        uploadExceptions.store(std::current_exception());
      }
    }
    try {
      uploadExceptions.rethrow();
      chunks_.reserve(chunks_.size() + pendingChunks_.size());
      // Workspaces are sized against memory that already holds the new sets,
      // so configure them before publishing anything.
      configureWorkspaces();
    } catch (...) {
      destroyDeviceSets(newSets);
      restoreWorkspaces();
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
    nextDeviceAssignment_ = (nextDeviceAssignment_ + pendingChunks_.size()) % deviceIds_.size();
    pendingChunks_.clear();
    published_ = true;
  }

  [[nodiscard]] std::size_t size() const {
    std::shared_lock lock(mutex_);
    return committedSize_;
  }

  [[nodiscard]] std::size_t pendingSize() const {
    std::shared_lock lock(mutex_);
    return static_cast<std::size_t>(nextId_) - committedSize_;
  }

  [[nodiscard]] std::size_t queryConcurrency() const {
    std::shared_lock lock(mutex_);
    return queryConcurrency_;
  }

  [[nodiscard]] std::size_t batchesInFlightPerGpu() const {
    std::shared_lock lock(mutex_);
    return batchesInFlightPerGpu_;
  }

  [[nodiscard]] std::size_t workspaceBytesPerQueryPerGpu() const {
    std::shared_lock lock(mutex_);
    return workspaceBytesPerQueryPerGpu_;
  }

  [[nodiscard]] std::vector<unsigned int> getMatches(const RDKit::ROMol& query,
                                                     int                 maxResults,
                                                     cudaStream_t        stream) const {
    std::shared_lock lock(mutex_);
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
    std::shared_lock lock(mutex_);
    requirePublished();
    std::size_t count = 0;
    for (const auto& deviceMatches : matchingIdsByDevice(query, stream, -1)) {
      count += deviceMatches.size();
    }
    return count;
  }

  [[nodiscard]] bool hasMatch(const RDKit::ROMol& query, cudaStream_t stream) const {
    std::shared_lock lock(mutex_);
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
    std::size_t                          deviceIndex;
    std::unique_ptr<ResidentTargetChunk> chunk;
  };

  // One admitted query's device state: the search pipeline and the GPU screen.
  struct QueryWorkspace {
    std::shared_ptr<ResidentSubstructSearchWorkspace> search;
    std::unique_ptr<PatternScreenWorkspace>           screen;
    // Per-target match flags, kept all-zero between queries.
    std::vector<std::uint8_t>                         gpuMatches;
  };

  struct DeviceWorkspaces {
    std::vector<std::unique_ptr<QueryWorkspace>>      owned;
    std::unique_ptr<ThreadSafeQueue<QueryWorkspace*>> available;
  };

  class WorkspaceLease {
   public:
    explicit WorkspaceLease(DeviceWorkspaces& pool) : pool_(&pool) {
      const auto workspace = pool.available->pop();
      if (!workspace.has_value()) {
        throw std::runtime_error("Substructure library workspace queue is closed");
      }
      workspace_ = *workspace;
    }
    ~WorkspaceLease() {
      if (workspace_ != nullptr) {
        pool_->available->push(workspace_);
      }
    }
    WorkspaceLease(const WorkspaceLease&)                          = delete;
    WorkspaceLease&               operator=(const WorkspaceLease&) = delete;
    [[nodiscard]] QueryWorkspace* get() const { return workspace_; }

   private:
    DeviceWorkspaces* pool_      = nullptr;
    QueryWorkspace*   workspace_ = nullptr;
  };

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
    if (workspacePools_.empty()) {
      throw std::runtime_error(
        "Substructure library has no query workspaces after a failed finalize(); retry finalize()");
    }
  }

  // After a failed finalize(), give the previously published generation its
  // workspaces back. Failure here leaves queries rejected by requirePublished().
  void restoreWorkspaces() noexcept {
    if (!published_) {
      return;
    }
    try {
      configureWorkspaces();
    } catch (...) {
      destroyWorkspaces();
    }
  }

  [[nodiscard]] SubstructSearchConfig deviceConfig(std::size_t deviceIndex) const {
    SubstructSearchConfig result = config_;
    result.gpuIds                = {deviceIds_[deviceIndex]};
    if (config_.preprocessingThreads == -1) {
      // Resident targets and a single query do not benefit from spawning a
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

  void configureWorkspaces() {
    destroyWorkspaces();
    if (deviceIds_.empty()) {
      queryConcurrency_ = 0;
      return;
    }

    constexpr std::size_t memoryNumerator   = 85;
    constexpr std::size_t memoryDenominator = 100;
    std::size_t           capacity          = std::numeric_limits<std::size_t>::max();
    std::size_t           workspaceBytes    = 0;
    for (std::size_t index = 0; index < deviceIds_.size(); ++index) {
      const WithDevice device(deviceIds_[index]);
      std::size_t      freeBytes  = 0;
      std::size_t      totalBytes = 0;
      cudaCheckError(cudaMemGetInfo(&freeBytes, &totalBytes));
      const std::size_t usedBytes            = totalBytes - freeBytes;
      const std::size_t budgetBytes          = totalBytes * memoryNumerator / memoryDenominator;
      const std::size_t availableBytes       = budgetBytes > usedBytes ? budgetBytes - usedBytes : 0;
      const std::size_t deviceWorkspaceBytes = estimateResidentSubstructSearchWorkspaceBytes(deviceConfig(index));
      if (deviceWorkspaceBytes == 0 || availableBytes < deviceWorkspaceBytes) {
        throw std::runtime_error(
          "Substructure library cannot admit one query workspace below the 85% GPU-memory cutoff");
      }
      capacity       = std::min(capacity, availableBytes / deviceWorkspaceBytes);
      workspaceBytes = std::max(workspaceBytes, deviceWorkspaceBytes);
    }

    const auto config = deviceConfig(0);
    const int  executorsPerRunner =
      config.executorsPerRunner == -1 ? (config.workerThreads == 1 ? 3 : 2) : config.executorsPerRunner;
    const std::size_t executorsPerQuery =
      static_cast<std::size_t>(config.workerThreads) * static_cast<std::size_t>(executorsPerRunner);
    const std::size_t threadsPerQuery =
      deviceIds_.size() * static_cast<std::size_t>(config.preprocessingThreads + config.workerThreads + 2);
    const std::size_t hostCapacity = std::max<std::size_t>(
      1,
      static_cast<std::size_t>(omp_get_max_threads()) / std::max<std::size_t>(1, threadsPerQuery));
    const std::size_t             concurrency = std::max<std::size_t>(1, std::min(capacity, hostCapacity));
    std::vector<DeviceWorkspaces> pools(deviceIds_.size());
    try {
      for (std::size_t index = 0; index < deviceIds_.size(); ++index) {
        const WithDevice device(deviceIds_[index]);
        auto&            pool = pools[index];
        pool.available        = std::make_unique<ThreadSafeQueue<QueryWorkspace*>>();
        pool.owned.reserve(concurrency);
        for (std::size_t slot = 0; slot < concurrency; ++slot) {
          auto workspace    = std::make_unique<QueryWorkspace>();
          workspace->search = makeResidentSubstructSearchWorkspace(deviceIds_[index]);
          workspace->screen = std::make_unique<PatternScreenWorkspace>(deviceIds_[index]);
          pool.available->push(workspace.get());
          pool.owned.push_back(std::move(workspace));
        }
      }
    } catch (...) {
      destroyWorkspacePools(pools);
      throw;
    }
    workspacePools_               = std::move(pools);
    queryConcurrency_             = concurrency;
    batchesInFlightPerGpu_        = concurrency * executorsPerQuery;
    workspaceBytesPerQueryPerGpu_ = workspaceBytes;
  }

  [[nodiscard]] std::vector<std::vector<unsigned int>> matchingIdsByDevice(const RDKit::ROMol& query,
                                                                           cudaStream_t        stream,
                                                                           int                 maxResults) const {
    if (deviceIds_.size() > 1 && stream != nullptr) {
      throw std::invalid_argument("A single external CUDA stream cannot be used with a multi-GPU substructure library");
    }
    std::unique_ptr<ExplicitBitVect>  queryFingerprint;
    std::optional<PatternScreenQuery> screenQuery;
    ScopedNvtxRange                   fingerprintRange("SubstructLibrary query fingerprint");
    if (usePatternFingerprints_) {
      queryFingerprint.reset(RDKit::PatternFingerprintMol(query));
      screenQuery = makePatternScreenQuery(queryFingerprint.get(), static_cast<int>(query.getNumAtoms()));
    }
    fingerprintRange.pop();
    std::vector<std::vector<unsigned int>> results(deviceIds_.size());
    detail::OpenMPExceptionRegistry        exceptionRegistry;

#pragma omp parallel for num_threads(static_cast<int>(deviceIds_.size())) schedule(static)
    for (std::int64_t deviceIndex = 0; deviceIndex < static_cast<std::int64_t>(deviceIds_.size()); ++deviceIndex) {
      try {
        const auto       index = static_cast<std::size_t>(deviceIndex);
        const WithDevice device(deviceIds_[index]);
        cudaStream_t     deviceStream = validateStream(stream);
        const auto       localConfig  = deviceConfig(index);
        auto&            deviceResult = results[index];
        WorkspaceLease   workspace(workspacePools_[index]);
        deviceResult = deviceMatchingIds(index,
                                         query,
                                         queryFingerprint.get(),
                                         screenQuery,
                                         deviceStream,
                                         localConfig,
                                         *workspace.get(),
                                         maxResults);
      } catch (...) {
        exceptionRegistry.store(std::current_exception());
      }
    }
    exceptionRegistry.rethrow();
    return results;
  }

  // Matching IDs on one device in ascending order, truncated to maxResults when positive.
  [[nodiscard]] std::vector<unsigned int> deviceMatchingIds(std::size_t                              deviceIndex,
                                                            const RDKit::ROMol&                      query,
                                                            const ExplicitBitVect*                   queryFingerprint,
                                                            const std::optional<PatternScreenQuery>& screenQuery,
                                                            cudaStream_t                             stream,
                                                            const SubstructSearchConfig&             config,
                                                            QueryWorkspace&                          workspace,
                                                            int                                      maxResults) const {
    std::vector<unsigned int> matches;
    const DeviceTargetSet*    targets = deviceSets_[deviceIndex].get();
    if (targets != nullptr && targets->size() != 0) {
      const int               numTargets = static_cast<int>(targets->size());
      std::vector<int>        selected;
      const std::vector<int>* candidates = nullptr;
      if (screenQuery.has_value()) {
        ScopedNvtxRange    screenRange("SubstructLibrary GPU screen");
        PatternScreenQuery deviceQuery = *screenQuery;
        orderPatternScreenBits(deviceQuery, targets->patternBitFrequencies());
        PatternScreenWorkspace& screen = *workspace.screen;
        screen.screen(targets->devicePatternSlices(), targets->deviceView().batchAtomStarts, numTargets, deviceQuery);
        selected.assign(screen.indices(), screen.indices() + screen.count());
        candidates = &selected;
      }
      if (candidates == nullptr || !candidates->empty()) {
        ScopedNvtxRange            matchRange("SubstructLibrary match candidates");
        std::vector<std::uint8_t>& gpuMatches = workspace.gpuMatches;
        hasSubstructMatchResident(targets->targets(),
                                  targets->host(),
                                  targets->device(),
                                  query,
                                  gpuMatches,
                                  config.algorithm,
                                  stream,
                                  config,
                                  workspace.search.get(),
                                  candidates);
        if (gpuMatches.size() != targets->size()) {
          throw std::runtime_error("Resident substructure result size does not match its target set");
        }
        const auto& ids = targets->ids();
        if (candidates != nullptr) {
          for (const int target : *candidates) {
            auto& flag = gpuMatches[static_cast<std::size_t>(target)];
            if (flag != 0) {
              matches.push_back(static_cast<unsigned int>(ids[static_cast<std::size_t>(target)]));
              flag = 0;
            }
          }
        } else {
          for (std::size_t target = 0; target < gpuMatches.size(); ++target) {
            if (gpuMatches[target] != 0) {
              matches.push_back(static_cast<unsigned int>(ids[target]));
              gpuMatches[target] = 0;
            }
          }
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
        const auto* targetFingerprint = record.chunk->patternFingerprint(id);
        if ((queryFingerprint == nullptr || targetFingerprint == nullptr ||
             AllProbeBitsMatch(*queryFingerprint, *targetFingerprint)) &&
            rdkitHasMatch(record.chunk->sourceMol(id), query)) {
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

  void destroyDeviceSets(std::vector<std::unique_ptr<DeviceTargetSet>>& sets) noexcept {
    int originalDevice = -1;
    cudaGetDevice(&originalDevice);
    for (std::size_t index = 0; index < sets.size(); ++index) {
      if (sets[index] && index < deviceIds_.size() && cudaSetDevice(deviceIds_[index]) == cudaSuccess) {
        sets[index].reset();
      }
    }
    if (originalDevice >= 0) {
      cudaSetDevice(originalDevice);
    }
  }

  void destroyWorkspacePools(std::vector<DeviceWorkspaces>& pools) noexcept {
    int originalDevice = -1;
    cudaGetDevice(&originalDevice);
    for (std::size_t index = 0; index < pools.size(); ++index) {
      if (index < deviceIds_.size() && cudaSetDevice(deviceIds_[index]) == cudaSuccess) {
        pools[index].available.reset();
        pools[index].owned.clear();
      }
    }
    pools.clear();
    if (originalDevice >= 0) {
      cudaSetDevice(originalDevice);
    }
  }

  void destroyWorkspaces() noexcept {
    destroyWorkspacePools(workspacePools_);
    queryConcurrency_             = 0;
    batchesInFlightPerGpu_        = 0;
    workspaceBytesPerQueryPerGpu_ = 0;
  }

  const std::size_t                                 chunkSize_;
  SubstructSearchConfig                             config_;
  bool                                              usePatternFingerprints_ = true;
  mutable std::shared_mutex                         mutex_;
  std::unique_ptr<TargetChunkBuilder>               builder_;
  std::vector<std::unique_ptr<ResidentTargetChunk>> pendingChunks_;
  std::vector<DeviceChunk>                          chunks_;
  std::vector<std::unique_ptr<DeviceTargetSet>>     deviceSets_;
  std::vector<int>                                  deviceIds_;
  mutable std::vector<DeviceWorkspaces>             workspacePools_;
  MoleculeId                                        nextId_                       = 0;
  std::size_t                                       committedSize_                = 0;
  std::size_t                                       nextDeviceAssignment_         = 0;
  bool                                              published_                    = false;
  std::size_t                                       queryConcurrency_             = 0;
  std::size_t                                       batchesInFlightPerGpu_        = 0;
  std::size_t                                       workspaceBytesPerQueryPerGpu_ = 0;
};

SubstructLibrary::SubstructLibrary(std::size_t chunkSize, SubstructSearchConfig config, bool usePatternFingerprints)
    : impl_(std::make_unique<Impl>(chunkSize, std::move(config), usePatternFingerprints)) {}

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

std::size_t SubstructLibrary::queryConcurrency() const {
  return impl_->queryConcurrency();
}

std::size_t SubstructLibrary::batchesInFlightPerGpu() const {
  return impl_->batchesInFlightPerGpu();
}

std::size_t SubstructLibrary::workspaceBytesPerQueryPerGpu() const {
  return impl_->workspaceBytesPerQueryPerGpu();
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
