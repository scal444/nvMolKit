// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/substruct/substruct_library.h"

#include <GraphMol/ROMol.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <utility>

#include "src/substruct/resident_target_chunk.h"
#include "src/substruct/substruct_search.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"

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
    if (config_.gpuIds.size() > 1) {
      throw std::invalid_argument("A resident substructure library currently supports one GPU");
    }
    if (!config_.gpuIds.empty()) {
      if (config_.gpuIds.front() < 0) {
        throw std::invalid_argument("Substructure library GPU ID must be nonnegative");
      }
      deviceId_ = config_.gpuIds.front();
    }
    builder_ = std::make_unique<TargetChunkBuilder>(0, chunkSize_);
  }

  ~Impl() noexcept {
    int  originalDevice = -1;
    bool restoreDevice  = false;
    if (!chunks_.empty() && cudaGetDevice(&originalDevice) == cudaSuccess && cudaSetDevice(deviceId_) == cudaSuccess) {
      restoreDevice = true;
    }
    chunks_.clear();
    if (restoreDevice) {
      cudaSetDevice(originalDevice);
    }
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

  void finalize(cudaStream_t stream) {
    std::unique_lock lock(mutex_);
    if (deviceId_ < 0) {
      cudaCheckError(cudaGetDevice(&deviceId_));
      config_.gpuIds = {deviceId_};
    }
    int deviceCount = 0;
    cudaCheckError(cudaGetDeviceCount(&deviceCount));
    if (deviceId_ >= deviceCount) {
      throw std::invalid_argument("Substructure library GPU ID is not available");
    }
    const WithDevice device(deviceId_);
    stream = validateStream(stream);

    if (!builder_->empty()) {
      auto nextBuilder = std::make_unique<TargetChunkBuilder>(nextId_, chunkSize_);
      pendingChunks_.reserve(pendingChunks_.size() + 1);
      pendingChunks_.push_back(builder_->seal());
      builder_ = std::move(nextBuilder);
    }

    // Upload disposable copies so a failed CUDA operation leaves the sealed
    // CPU generation intact and retryable.
    std::vector<std::unique_ptr<ResidentTargetChunk>> candidates;
    candidates.reserve(pendingChunks_.size());
    for (const auto& pending : pendingChunks_) {
      TargetChunkBuilder candidateBuilder(pending->firstId(), pending->size());
      for (MoleculeId id = pending->firstId(); id < pending->endId(); ++id) {
        candidateBuilder.addMol(pending->sourceMol(id));
      }
      candidates.push_back(candidateBuilder.seal());
    }

    for (auto& chunk : candidates) {
      chunk->beginUpload(stream);
    }
    for (auto& chunk : candidates) {
      chunk->commit();
    }
    for (auto& chunk : candidates) {
      committedSize_ += chunk->size();
      chunks_.push_back(std::move(chunk));
    }
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

  [[nodiscard]] std::vector<unsigned int> getMatches(const RDKit::ROMol& query,
                                                     int                 maxResults,
                                                     cudaStream_t        stream) const {
    std::shared_lock lock(mutex_);
    requirePublished();
    if (maxResults == 0 || chunks_.empty()) {
      return {};
    }

    const WithDevice device(deviceId_);
    stream = validateStream(stream);
    std::vector<unsigned int> matches;
    if (maxResults > 0) {
      matches.reserve(std::min<std::size_t>(committedSize_, static_cast<std::size_t>(maxResults)));
    }

    for (const auto& chunk : chunks_) {
      std::vector<MoleculeId> chunkMatches = matchingIds(*chunk, query, stream);
      for (const MoleculeId id : chunkMatches) {
        matches.push_back(static_cast<unsigned int>(id));
        if (maxResults > 0 && matches.size() == static_cast<std::size_t>(maxResults)) {
          return matches;
        }
      }
    }
    return matches;
  }

  [[nodiscard]] std::size_t countMatches(const RDKit::ROMol& query, cudaStream_t stream) const {
    std::shared_lock lock(mutex_);
    requirePublished();
    const WithDevice device(deviceId_);
    stream = validateStream(stream);

    std::size_t count = 0;
    for (const auto& chunk : chunks_) {
      count += matchingIds(*chunk, query, stream).size();
    }
    return count;
  }

  [[nodiscard]] bool hasMatch(const RDKit::ROMol& query, cudaStream_t stream) const {
    std::shared_lock lock(mutex_);
    requirePublished();
    const WithDevice device(deviceId_);
    stream = validateStream(stream);

    for (const auto& chunk : chunks_) {
      if (!matchingIds(*chunk, query, stream).empty()) {
        return true;
      }
    }
    return false;
  }

 private:
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

  [[nodiscard]] std::vector<MoleculeId> matchingIds(const ResidentTargetChunk& chunk,
                                                    const RDKit::ROMol&        query,
                                                    cudaStream_t               stream) const {
    std::vector<MoleculeId> matches;
    if (chunk.gpuTargetCount() != 0) {
      std::vector<std::uint8_t> gpuMatches;
      hasSubstructMatchResident(chunk.supportedTargetPtrs(),
                                chunk.packedHost(),
                                chunk.deviceStorage(),
                                query,
                                gpuMatches,
                                config_.algorithm,
                                stream,
                                config_);
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

  const std::size_t                                 chunkSize_;
  SubstructSearchConfig                             config_;
  mutable std::shared_mutex                         mutex_;
  std::unique_ptr<TargetChunkBuilder>               builder_;
  std::vector<std::unique_ptr<ResidentTargetChunk>> pendingChunks_;
  std::vector<std::unique_ptr<ResidentTargetChunk>> chunks_;
  MoleculeId                                        nextId_        = 0;
  std::size_t                                       committedSize_ = 0;
  bool                                              published_     = false;
  int                                               deviceId_      = -1;
};

SubstructLibrary::SubstructLibrary(std::size_t chunkSize, SubstructSearchConfig config)
    : impl_(std::make_unique<Impl>(chunkSize, std::move(config))) {}

SubstructLibrary::~SubstructLibrary() = default;

unsigned int SubstructLibrary::addMol(const RDKit::ROMol& molecule) {
  return impl_->addMol(molecule);
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
