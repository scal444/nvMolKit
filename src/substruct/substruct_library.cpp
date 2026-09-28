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
#include <shared_mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "src/substruct/resident_target_chunk.h"
#include "src/substruct/substruct_search.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
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
    destroyDeviceChunks(chunks_);
    pendingChunks_.clear();
  }

  unsigned int addMol(const RDKit::ROMol& molecule) {
    std::unique_lock lock(mutex_);
    if (nextId_ > std::numeric_limits<unsigned int>::max()) {
      throw std::overflow_error("Substructure library molecule ID space exhausted");
    }
    if (builder_->full()) {
      auto nextBuilder = std::make_unique<TargetChunkBuilder>(nextId_, chunkSize_);
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
      builder_ = std::make_unique<TargetChunkBuilder>(nextId_, chunkSize_);
    }

    const std::size_t                                 numChunks = (molecules.size() + chunkSize_ - 1) / chunkSize_;
    std::vector<std::unique_ptr<ResidentTargetChunk>> builtChunks(numChunks);
    const int                                         numThreads = constructionThreads();
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
    std::unique_lock lock(mutex_);
    resolveDevices();
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
      auto nextBuilder = std::make_unique<TargetChunkBuilder>(nextId_, chunkSize_);
      pendingChunks_.reserve(pendingChunks_.size() + 1);
      pendingChunks_.push_back(builder_->seal());
      builder_ = std::move(nextBuilder);
    }

    // Upload disposable copies so a failed CUDA operation leaves the sealed
    // CPU generation intact and retryable.
    std::vector<std::unique_ptr<ResidentTargetChunk>> candidates;
    std::vector<std::size_t>                          candidateDevices;
    candidates.reserve(pendingChunks_.size());
    candidateDevices.resize(pendingChunks_.size());
    candidates.resize(pendingChunks_.size());
    const int numThreads = constructionThreads();
    for (std::size_t index = 0; index < pendingChunks_.size(); ++index) {
      const auto&                      pending = pendingChunks_[index];
      std::vector<const RDKit::ROMol*> chunkMolecules;
      chunkMolecules.reserve(pending->size());
      for (MoleculeId id = pending->firstId(); id < pending->endId(); ++id) {
        chunkMolecules.push_back(&pending->sourceMol(id));
      }
      TargetChunkBuilder candidateBuilder(pending->firstId(), pending->size());
      candidateBuilder.addMols(chunkMolecules, numThreads);
      candidates[index] = candidateBuilder.seal();
    }
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      candidateDevices[index] = (nextDeviceAssignment_ + index) % deviceIds_.size();
    }

    detail::OpenMPExceptionRegistry uploadExceptions;
#pragma omp parallel for num_threads(static_cast<int>(deviceIds_.size())) schedule(static)
    for (std::int64_t deviceIndex = 0; deviceIndex < static_cast<std::int64_t>(deviceIds_.size()); ++deviceIndex) {
      try {
        const WithDevice device(deviceIds_[static_cast<std::size_t>(deviceIndex)]);
        cudaStream_t     deviceStream = validateStream(stream);
        for (std::size_t index = 0; index < candidates.size(); ++index) {
          if (candidateDevices[index] == static_cast<std::size_t>(deviceIndex)) {
            candidates[index]->beginUpload(deviceStream);
          }
        }
      } catch (...) {
        uploadExceptions.store(std::current_exception());
      }
    }
    try {
      uploadExceptions.rethrow();
    } catch (...) {
      destroyCandidates(candidates, candidateDevices);
      throw;
    }

    detail::OpenMPExceptionRegistry commitExceptions;
#pragma omp parallel for num_threads(static_cast<int>(deviceIds_.size())) schedule(static)
    for (std::int64_t deviceIndex = 0; deviceIndex < static_cast<std::int64_t>(deviceIds_.size()); ++deviceIndex) {
      try {
        const WithDevice device(deviceIds_[static_cast<std::size_t>(deviceIndex)]);
        for (std::size_t index = 0; index < candidates.size(); ++index) {
          if (candidateDevices[index] == static_cast<std::size_t>(deviceIndex)) {
            candidates[index]->commit();
          }
        }
      } catch (...) {
        commitExceptions.store(std::current_exception());
      }
    }
    try {
      commitExceptions.rethrow();
    } catch (...) {
      destroyCandidates(candidates, candidateDevices);
      throw;
    }

    for (std::size_t index = 0; index < candidates.size(); ++index) {
      committedSize_ += candidates[index]->size();
      chunks_.push_back(DeviceChunk{candidateDevices[index], std::move(candidates[index])});
    }
    nextDeviceAssignment_ = (nextDeviceAssignment_ + candidates.size()) % deviceIds_.size();
    pendingChunks_.clear();
    published_ = true;
    configureWorkspaces();
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

  struct DeviceWorkspaces {
    std::vector<std::shared_ptr<ResidentSubstructSearchWorkspace>>      owned;
    std::unique_ptr<ThreadSafeQueue<ResidentSubstructSearchWorkspace*>> available;
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
    [[nodiscard]] ResidentSubstructSearchWorkspace* get() const { return workspace_; }

   private:
    DeviceWorkspaces*                 pool_      = nullptr;
    ResidentSubstructSearchWorkspace* workspace_ = nullptr;
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

    constexpr std::size_t    memoryNumerator   = 85;
    constexpr std::size_t    memoryDenominator = 100;
    std::size_t              capacity          = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> deviceCapacities(deviceIds_.size());
    for (std::size_t index = 0; index < deviceIds_.size(); ++index) {
      const WithDevice device(deviceIds_[index]);
      std::size_t      freeBytes  = 0;
      std::size_t      totalBytes = 0;
      cudaCheckError(cudaMemGetInfo(&freeBytes, &totalBytes));
      const std::size_t usedBytes      = totalBytes - freeBytes;
      const std::size_t budgetBytes    = totalBytes * memoryNumerator / memoryDenominator;
      const std::size_t availableBytes = budgetBytes > usedBytes ? budgetBytes - usedBytes : 0;
      const std::size_t workspaceBytes = estimateResidentSubstructSearchWorkspaceBytes(deviceConfig(index));
      if (workspaceBytes == 0 || availableBytes < workspaceBytes) {
        throw std::runtime_error(
          "Substructure library cannot admit one query workspace below the 85% GPU-memory cutoff");
      }
      deviceCapacities[index]       = availableBytes / workspaceBytes;
      capacity                      = std::min(capacity, deviceCapacities[index]);
      workspaceBytesPerQueryPerGpu_ = std::max(workspaceBytesPerQueryPerGpu_, workspaceBytes);
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
    queryConcurrency_      = std::max<std::size_t>(1, std::min(capacity, hostCapacity));
    batchesInFlightPerGpu_ = queryConcurrency_ * executorsPerQuery;
    workspacePools_.resize(deviceIds_.size());
    for (std::size_t index = 0; index < deviceIds_.size(); ++index) {
      const WithDevice device(deviceIds_[index]);
      auto&            pool = workspacePools_[index];
      pool.available        = std::make_unique<ThreadSafeQueue<ResidentSubstructSearchWorkspace*>>();
      pool.owned.reserve(queryConcurrency_);
      for (std::size_t slot = 0; slot < queryConcurrency_; ++slot) {
        auto workspace = makeResidentSubstructSearchWorkspace(deviceIds_[index]);
        pool.available->push(workspace.get());
        pool.owned.push_back(std::move(workspace));
      }
    }
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
        auto&            deviceResult = results[index];
        WorkspaceLease   workspace(workspacePools_[index]);
        for (const auto& record : chunks_) {
          if (record.deviceIndex != index) {
            continue;
          }
          const auto chunkMatches = matchingIds(*record.chunk, query, deviceStream, localConfig, workspace.get());
          for (const MoleculeId id : chunkMatches) {
            deviceResult.push_back(static_cast<unsigned int>(id));
            if (maxResults > 0 && deviceResult.size() == static_cast<std::size_t>(maxResults)) {
              break;
            }
          }
          if (maxResults > 0 && deviceResult.size() == static_cast<std::size_t>(maxResults)) {
            break;
          }
        }
      } catch (...) {
        exceptionRegistry.store(std::current_exception());
      }
    }
    exceptionRegistry.rethrow();
    return results;
  }

  [[nodiscard]] std::vector<MoleculeId> matchingIds(const ResidentTargetChunk&        chunk,
                                                    const RDKit::ROMol&               query,
                                                    cudaStream_t                      stream,
                                                    const SubstructSearchConfig&      config,
                                                    ResidentSubstructSearchWorkspace* workspace) const {
    std::vector<MoleculeId> matches;
    if (chunk.gpuTargetCount() != 0) {
      std::vector<std::uint8_t> gpuMatches;
      hasSubstructMatchResident(chunk.supportedTargetPtrs(),
                                chunk.packedHost(),
                                chunk.deviceStorage(),
                                query,
                                gpuMatches,
                                config.algorithm,
                                stream,
                                config,
                                workspace);
      if (gpuMatches.size() != chunk.packedGlobalIds().size()) {
        throw std::runtime_error("Resident substructure result size does not match its target chunk");
      }
      for (std::size_t index = 0; index < gpuMatches.size(); ++index) {
        if (gpuMatches[index] != 0) {
          matches.push_back(chunk.packedGlobalIds()[index]);
        }
      }
    }

    for (const MoleculeId id : chunk.fallbackGlobalIds()) {
      if (rdkitHasMatch(chunk.sourceMol(id), query)) {
        matches.push_back(id);
      }
    }
    std::sort(matches.begin(), matches.end());
    return matches;
  }

  void destroyCandidates(std::vector<std::unique_ptr<ResidentTargetChunk>>& candidates,
                         const std::vector<std::size_t>&                    candidateDevices) noexcept {
    int originalDevice = -1;
    cudaGetDevice(&originalDevice);
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      if (candidates[index] && cudaSetDevice(deviceIds_[candidateDevices[index]]) == cudaSuccess) {
        candidates[index].reset();
      }
    }
    if (originalDevice >= 0) {
      cudaSetDevice(originalDevice);
    }
  }

  void destroyDeviceChunks(std::vector<DeviceChunk>& chunks) noexcept {
    int originalDevice = -1;
    cudaGetDevice(&originalDevice);
    for (auto& record : chunks) {
      if (record.chunk && record.deviceIndex < deviceIds_.size() &&
          cudaSetDevice(deviceIds_[record.deviceIndex]) == cudaSuccess) {
        record.chunk.reset();
      }
    }
    if (originalDevice >= 0) {
      cudaSetDevice(originalDevice);
    }
    chunks.clear();
  }

  void destroyWorkspaces() noexcept {
    int originalDevice = -1;
    cudaGetDevice(&originalDevice);
    for (std::size_t index = 0; index < workspacePools_.size(); ++index) {
      if (index < deviceIds_.size() && cudaSetDevice(deviceIds_[index]) == cudaSuccess) {
        workspacePools_[index].available.reset();
        workspacePools_[index].owned.clear();
      }
    }
    workspacePools_.clear();
    queryConcurrency_             = 0;
    batchesInFlightPerGpu_        = 0;
    workspaceBytesPerQueryPerGpu_ = 0;
    if (originalDevice >= 0) {
      cudaSetDevice(originalDevice);
    }
  }

  const std::size_t                                 chunkSize_;
  SubstructSearchConfig                             config_;
  mutable std::shared_mutex                         mutex_;
  std::unique_ptr<TargetChunkBuilder>               builder_;
  std::vector<std::unique_ptr<ResidentTargetChunk>> pendingChunks_;
  std::vector<DeviceChunk>                          chunks_;
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
