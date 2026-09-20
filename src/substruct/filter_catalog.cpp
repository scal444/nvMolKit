// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/substruct/filter_catalog.h"

#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmartsWrite.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <utility>

#include "src/substruct/molecules.h"
#include "src/substruct/rdkit_filter_catalog_data.h"
#include "src/substruct/recursive_preprocessor.h"
#include "src/substruct/substruct_constants.h"
#include "src/substruct/substruct_search.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"

namespace nvMolKit {

namespace {

constexpr std::uint32_t kAllPresets = static_cast<std::uint32_t>(RDKit::FilterCatalogParams::ALL);

bool triggers(const RDKit::ROMol& target, const RDKit::ROMol& query, unsigned int triggerCount) {
  RDKit::SubstructMatchParameters parameters;
  parameters.uniquify             = true;
  parameters.useChirality         = false;
  parameters.useQueryQueryMatches = false;
  parameters.maxMatches           = triggerCount;
  return RDKit::SubstructMatch(target, query, parameters).size() >= triggerCount;
}

void validateTargets(const std::vector<const RDKit::ROMol*>& targets) {
  for (const RDKit::ROMol* target : targets) {
    if (target == nullptr) {
      throw std::invalid_argument("Filter catalog targets cannot contain null molecules");
    }
  }
}

}  // namespace

class FilterCatalog::Impl {
 public:
  struct OwnedEntry {
    FilterCatalogEntry            metadata;
    std::unique_ptr<RDKit::ROMol> query;
  };

  explicit Impl(SubstructSearchConfig config) : config_(std::move(config)) {
    if (config_.algorithm == SubstructAlgorithm::VF2) {
      throw std::invalid_argument("Filter catalogs support the GSI and DFS production backends");
    }
    if (config_.gpuIds.size() > 1) {
      throw std::invalid_argument("A resident filter catalog currently supports one GPU");
    }
    if (!config_.gpuIds.empty() && config_.gpuIds.front() < 0) {
      throw std::invalid_argument("Filter catalog GPU ID must be nonnegative");
    }
    if (!config_.gpuIds.empty()) {
      deviceId_ = config_.gpuIds.front();
    }
  }

  ~Impl() noexcept {
    int  originalDevice = -1;
    bool restoreDevice  = false;
    if (preparedQueries_ != nullptr && cudaGetDevice(&originalDevice) == cudaSuccess &&
        cudaSetDevice(deviceId_) == cudaSuccess) {
      restoreDevice = true;
    }
    preparedQueries_.reset();
    if (restoreDevice) {
      cudaSetDevice(originalDevice);
    }
  }

  std::size_t addPreset(FilterCatalogPreset preset) {
    std::unique_lock lock(mutex_);
    const auto       mask = static_cast<std::uint32_t>(preset);
    if (mask == 0 || (mask & ~kAllPresets) != 0) {
      throw std::invalid_argument("Unknown or empty RDKit filter catalog preset");
    }

    std::size_t added = 0;
    for (unsigned int bit = 1; bit < 32; ++bit) {
      const std::uint32_t atomicMask = 1u << bit;
      if ((mask & atomicMask) == 0) {
        continue;
      }
      const auto  rdkitPreset = static_cast<RDKit::FilterCatalogParams::FilterCatalogs>(atomicMask);
      const auto  count       = RDKit::GetNumEntries(rdkitPreset);
      const auto* data        = RDKit::GetFilterData(rdkitPreset);
      const auto  propCount   = RDKit::GetNumPropertyEntries(rdkitPreset);
      const auto* props       = RDKit::GetFilterProperties(rdkitPreset);
      if (count == 0 || data == nullptr || (propCount != 0 && props == nullptr)) {
        throw std::runtime_error("RDKit did not provide data for a selected filter catalog preset");
      }

      FilterCatalogProperties catalogProperties;
      for (unsigned int property = 0; property < propCount; ++property) {
        catalogProperties.emplace(props[property].key, props[property].value);
      }
      for (unsigned int index = 0; index < count; ++index) {
        const auto& record     = data[index];
        auto        properties = catalogProperties;
        if (record.comment != nullptr && record.comment[0] != '\0') {
          properties.emplace("Comment", record.comment);
        }
        addSmartsUnlocked(record.smarts,
                          record.name,
                          record.max == 0 ? 1 : record.max + 1,
                          std::move(properties),
                          true);
        ++added;
      }
    }
    return added;
  }

  unsigned int addEntry(const RDKit::ROMol&     query,
                        std::string             description,
                        unsigned int            triggerCount,
                        FilterCatalogProperties properties) {
    std::unique_lock lock(mutex_);
    return addEntryUnlocked(std::make_unique<RDKit::ROMol>(query),
                            RDKit::MolToSmarts(query),
                            std::move(description),
                            triggerCount,
                            std::move(properties));
  }

  unsigned int addSmarts(std::string             smarts,
                         std::string             description,
                         unsigned int            triggerCount,
                         FilterCatalogProperties properties) {
    std::unique_lock lock(mutex_);
    return addSmartsUnlocked(std::move(smarts), std::move(description), triggerCount, std::move(properties));
  }

  void finalize(cudaStream_t stream) {
    std::unique_lock                 lock(mutex_);
    std::vector<const RDKit::ROMol*> gpuQueries;
    std::vector<unsigned int>        gpuEntryIds;
    gpuQueries.reserve(entries_.size());
    gpuEntryIds.reserve(entries_.size());
    for (std::size_t index = 0; index < entries_.size(); ++index) {
      if (entries_[index].metadata.triggerCount != 1 || !canPrepareOnGpu(*entries_[index].query)) {
        continue;
      }
      gpuQueries.push_back(entries_[index].query.get());
      gpuEntryIds.push_back(static_cast<unsigned int>(index));
    }

    std::unique_ptr<PreparedSubstructQueries> candidate;
    if (!gpuQueries.empty()) {
      if (deviceId_ < 0) {
        cudaCheckError(cudaGetDevice(&deviceId_));
        config_.gpuIds = {deviceId_};
      }
      int deviceCount = 0;
      cudaCheckError(cudaGetDeviceCount(&deviceCount));
      if (deviceId_ >= deviceCount) {
        throw std::invalid_argument("Filter catalog GPU ID is not available");
      }
      const WithDevice device(deviceId_);
      stream    = validateStream(stream);
      candidate = std::make_unique<PreparedSubstructQueries>(gpuQueries, stream, config_);

      // Publish only after preparation completes so a failed upload leaves the
      // previously finalized generation available and retryable. Keep the
      // selected device current while releasing the previous device storage.
      preparedQueries_ = std::move(candidate);
      gpuEntryIds_     = std::move(gpuEntryIds);
      committedSize_   = entries_.size();
      published_       = true;
      return;
    }

    preparedQueries_ = std::move(candidate);
    gpuEntryIds_     = std::move(gpuEntryIds);
    committedSize_   = entries_.size();
    published_       = true;
  }

  [[nodiscard]] std::size_t size() const {
    std::shared_lock lock(mutex_);
    return committedSize_;
  }

  [[nodiscard]] std::size_t pendingSize() const {
    std::shared_lock lock(mutex_);
    return entries_.size() - committedSize_;
  }

  [[nodiscard]] FilterCatalogEntry getEntry(unsigned int id) const {
    std::shared_lock lock(mutex_);
    if (id >= entries_.size()) {
      throw std::out_of_range("Filter catalog entry ID is out of range");
    }
    return entries_[id].metadata;
  }

  [[nodiscard]] std::vector<std::uint8_t> hasMatch(const std::vector<const RDKit::ROMol*>& targets,
                                                   cudaStream_t                            stream) const {
    std::shared_lock lock(mutex_);
    requirePublished();
    validateTargets(targets);

    std::vector<std::uint8_t> result(targets.size(), 0);
    if (preparedQueries_ != nullptr && !targets.empty()) {
      const WithDevice device(deviceId_);
      stream = validateStream(stream);
      hasAnySubstructMatch(targets, *preparedQueries_, result, config_.algorithm, stream, config_);
    }
    const std::vector<unsigned int> fallbackIds = cpuEntryIds();
    for (std::size_t targetIndex = 0; targetIndex < targets.size(); ++targetIndex) {
      if (result[targetIndex] != 0) {
        continue;
      }
      for (const unsigned int entryIndex : fallbackIds) {
        if (triggers(*targets[targetIndex], *entries_[entryIndex].query, entries_[entryIndex].metadata.triggerCount)) {
          result[targetIndex] = 1;
          break;
        }
      }
    }
    return result;
  }

  [[nodiscard]] std::vector<int> getFirstMatch(const std::vector<const RDKit::ROMol*>& targets,
                                               cudaStream_t                            stream) const {
    std::shared_lock lock(mutex_);
    requirePublished();
    validateTargets(targets);

    std::vector<int> result(targets.size(), -1);
    if (preparedQueries_ != nullptr && !targets.empty()) {
      std::vector<int> preparedResult;
      const WithDevice device(deviceId_);
      stream = validateStream(stream);
      getFirstSubstructMatch(targets, *preparedQueries_, preparedResult, config_.algorithm, stream, config_);
      for (std::size_t index = 0; index < preparedResult.size(); ++index) {
        if (preparedResult[index] >= 0) {
          result[index] = static_cast<int>(gpuEntryIds_[static_cast<std::size_t>(preparedResult[index])]);
        }
      }
    }
    const std::vector<unsigned int> fallbackIds = cpuEntryIds();
    for (std::size_t targetIndex = 0; targetIndex < targets.size(); ++targetIndex) {
      for (const unsigned int entryIndex : fallbackIds) {
        if (result[targetIndex] >= 0 && entryIndex >= static_cast<unsigned int>(result[targetIndex])) {
          break;
        }
        if (triggers(*targets[targetIndex], *entries_[entryIndex].query, entries_[entryIndex].metadata.triggerCount)) {
          result[targetIndex] = static_cast<int>(entryIndex);
          break;
        }
      }
    }
    return result;
  }

  [[nodiscard]] std::vector<std::vector<unsigned int>> getMatches(const std::vector<const RDKit::ROMol*>& targets,
                                                                  cudaStream_t stream) const {
    std::shared_lock lock(mutex_);
    requirePublished();
    validateTargets(targets);

    std::vector<std::vector<unsigned int>> result(targets.size());
    if (preparedQueries_ != nullptr && !targets.empty()) {
      HasSubstructMatchResults gpuMatches;
      const WithDevice         device(deviceId_);
      stream = validateStream(stream);
      hasSubstructMatch(targets, *preparedQueries_, gpuMatches, config_.algorithm, stream, config_);
      for (std::size_t targetIndex = 0; targetIndex < targets.size(); ++targetIndex) {
        for (std::size_t queryIndex = 0; queryIndex < gpuEntryIds_.size(); ++queryIndex) {
          if (gpuMatches.matches(static_cast<int>(targetIndex), static_cast<int>(queryIndex))) {
            result[targetIndex].push_back(gpuEntryIds_[queryIndex]);
          }
        }
      }
    }

    const std::vector<unsigned int> fallbackIds = cpuEntryIds();
    for (std::size_t targetIndex = 0; targetIndex < targets.size(); ++targetIndex) {
      for (const unsigned int entryIndex : fallbackIds) {
        if (triggers(*targets[targetIndex], *entries_[entryIndex].query, entries_[entryIndex].metadata.triggerCount)) {
          result[targetIndex].push_back(entryIndex);
        }
      }
      std::sort(result[targetIndex].begin(), result[targetIndex].end());
    }
    return result;
  }

 private:
  unsigned int addSmartsUnlocked(std::string             smarts,
                                 std::string             description,
                                 unsigned int            triggerCount,
                                 FilterCatalogProperties properties,
                                 bool                    mergeHydrogens = false) {
    std::unique_ptr<RDKit::ROMol> query(RDKit::SmartsToMol(smarts, 0, mergeHydrogens));
    if (query == nullptr) {
      throw std::invalid_argument("Could not parse filter catalog SMARTS: " + smarts);
    }
    return addEntryUnlocked(std::move(query),
                            std::move(smarts),
                            std::move(description),
                            triggerCount,
                            std::move(properties));
  }

  unsigned int addEntryUnlocked(std::unique_ptr<RDKit::ROMol> query,
                                std::string                   smarts,
                                std::string                   description,
                                unsigned int                  triggerCount,
                                FilterCatalogProperties       properties) {
    if (triggerCount == 0) {
      throw std::invalid_argument("Filter catalog trigger count must be greater than zero");
    }
    if (query->getNumAtoms() == 0) {
      throw std::invalid_argument("Filter catalog queries must contain at least one atom");
    }
    if (entries_.size() > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
      throw std::overflow_error("Filter catalog entry ID space exhausted");
    }
    const auto id = static_cast<unsigned int>(entries_.size());
    entries_.push_back(OwnedEntry{
      FilterCatalogEntry{id, std::move(description), std::move(smarts), triggerCount, std::move(properties)},
      std::move(query)
    });
    return id;
  }

  void requirePublished() const {
    if (!published_) {
      throw std::logic_error("Filter catalog must be finalized before querying");
    }
  }

  [[nodiscard]] bool canPrepareOnGpu(const RDKit::ROMol& query) const {
    if (query.getNumAtoms() > static_cast<unsigned int>(kMaxQueryAtoms)) {
      return false;
    }
    try {
      MoleculesHost probe;
      addQueryToBatch(&query, probe);
      RecursivePatternPreprocessor recursiveProbe;
      recursiveProbe.buildPatterns(probe);
      return true;
    } catch (const std::runtime_error&) {
      return false;
    }
  }

  [[nodiscard]] std::vector<unsigned int> cpuEntryIds() const {
    std::vector<unsigned int> result;
    result.reserve(committedSize_ - gpuEntryIds_.size());
    std::size_t gpuIndex = 0;
    for (std::size_t entry = 0; entry < committedSize_; ++entry) {
      if (gpuIndex < gpuEntryIds_.size() && gpuEntryIds_[gpuIndex] == entry) {
        ++gpuIndex;
      } else {
        result.push_back(static_cast<unsigned int>(entry));
      }
    }
    return result;
  }

  [[nodiscard]] cudaStream_t validateStream(cudaStream_t stream) const {
    const auto validated = acquireExternalStream(reinterpret_cast<std::uintptr_t>(stream));
    if (!validated.has_value()) {
      throw std::invalid_argument("CUDA stream does not belong to the filter catalog device");
    }
    return *validated;
  }

  SubstructSearchConfig                     config_;
  mutable std::shared_mutex                 mutex_;
  std::vector<OwnedEntry>                   entries_;
  std::unique_ptr<PreparedSubstructQueries> preparedQueries_;
  std::vector<unsigned int>                 gpuEntryIds_;
  std::size_t                               committedSize_ = 0;
  bool                                      published_     = false;
  int                                       deviceId_      = -1;
};

FilterCatalog::FilterCatalog(SubstructSearchConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}

FilterCatalog::FilterCatalog(FilterCatalogPreset preset, SubstructSearchConfig config)
    : FilterCatalog(std::move(config)) {
  impl_->addPreset(preset);
}

FilterCatalog::~FilterCatalog() = default;

std::size_t FilterCatalog::addPreset(FilterCatalogPreset preset) {
  return impl_->addPreset(preset);
}

unsigned int FilterCatalog::addEntry(const RDKit::ROMol&     query,
                                     std::string             description,
                                     unsigned int            triggerCount,
                                     FilterCatalogProperties properties) {
  return impl_->addEntry(query, std::move(description), triggerCount, std::move(properties));
}

unsigned int FilterCatalog::addSmarts(std::string             smarts,
                                      std::string             description,
                                      unsigned int            triggerCount,
                                      FilterCatalogProperties properties) {
  return impl_->addSmarts(std::move(smarts), std::move(description), triggerCount, std::move(properties));
}

void FilterCatalog::finalize(cudaStream_t stream) {
  impl_->finalize(stream);
}

std::size_t FilterCatalog::size() const {
  return impl_->size();
}

std::size_t FilterCatalog::pendingSize() const {
  return impl_->pendingSize();
}

FilterCatalogEntry FilterCatalog::getEntry(unsigned int id) const {
  return impl_->getEntry(id);
}

std::vector<std::uint8_t> FilterCatalog::hasMatch(const std::vector<const RDKit::ROMol*>& targets,
                                                  cudaStream_t                            stream) const {
  return impl_->hasMatch(targets, stream);
}

std::vector<int> FilterCatalog::getFirstMatch(const std::vector<const RDKit::ROMol*>& targets,
                                              cudaStream_t                            stream) const {
  return impl_->getFirstMatch(targets, stream);
}

std::vector<std::vector<unsigned int>> FilterCatalog::getMatches(const std::vector<const RDKit::ROMol*>& targets,
                                                                 cudaStream_t                            stream) const {
  return impl_->getMatches(targets, stream);
}

}  // namespace nvMolKit
