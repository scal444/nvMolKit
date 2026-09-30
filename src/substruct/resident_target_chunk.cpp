// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/substruct/resident_target_chunk.h"

#include <DataStructs/ExplicitBitVect.h>
#include <GraphMol/Fingerprints/Fingerprints.h>
#include <GraphMol/ROMol.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "src/substruct/pattern_screen.h"
#include "src/substruct/substruct_constants.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/openmp_helpers.h"

namespace nvMolKit {

namespace {

void appendPackedTarget(MoleculesHost& destination, const MoleculesHost& source) {
  if (source.numMolecules() != 1) {
    throw std::logic_error("A packed target append must contain exactly one molecule");
  }

  const int atomOffset = destination.batchAtomStarts.back();
  destination.atomDataPacked.insert(destination.atomDataPacked.end(),
                                    source.atomDataPacked.begin(),
                                    source.atomDataPacked.end());
  destination.bondTypeCounts.insert(destination.bondTypeCounts.end(),
                                    source.bondTypeCounts.begin(),
                                    source.bondTypeCounts.end());
  destination.targetAtomBonds.insert(destination.targetAtomBonds.end(),
                                     source.targetAtomBonds.begin(),
                                     source.targetAtomBonds.end());
  destination.batchAtomStarts.push_back(atomOffset + source.batchAtomStarts.back());
}

}  // namespace

ResidentTargetChunk::ResidentTargetChunk(MoleculeId                                    firstId,
                                         std::vector<std::unique_ptr<RDKit::ROMol>>    sourceMolecules,
                                         MoleculesHost                                 packedHost,
                                         std::vector<MoleculeId>                       packedGlobalIds,
                                         std::vector<MoleculeId>                       fallbackGlobalIds,
                                         std::vector<std::uint8_t>                     gpuSupported,
                                         std::vector<std::unique_ptr<ExplicitBitVect>> patternFingerprints)
    : firstId_(firstId),
      sourceMolecules_(std::move(sourceMolecules)),
      packedHost_(std::move(packedHost)),
      packedGlobalIds_(std::move(packedGlobalIds)),
      fallbackGlobalIds_(std::move(fallbackGlobalIds)),
      gpuSupported_(std::move(gpuSupported)),
      patternFingerprints_(std::move(patternFingerprints)) {
  if (sourceMolecules_.size() != gpuSupported_.size()) {
    throw std::logic_error("Resident target chunk support metadata is inconsistent");
  }
  if (packedHost_.numMolecules() != packedGlobalIds_.size()) {
    throw std::logic_error("Resident target chunk packed-ID mapping is inconsistent");
  }
  if (packedGlobalIds_.size() + fallbackGlobalIds_.size() != sourceMolecules_.size()) {
    throw std::logic_error("Resident target chunk molecule classification is incomplete");
  }
  if (!patternFingerprints_.empty() && patternFingerprints_.size() != sourceMolecules_.size()) {
    throw std::logic_error("Resident target chunk pattern-fingerprint metadata is inconsistent");
  }

  supportedTargetPtrs_.reserve(packedGlobalIds_.size());
  for (MoleculeId id : packedGlobalIds_) {
    if (id < firstId_ || id >= endId()) {
      throw std::logic_error("Resident target chunk packed ID is outside its source range");
    }
    supportedTargetPtrs_.push_back(sourceMolecules_[static_cast<std::size_t>(id - firstId_)].get());
  }
  if (!patternFingerprints_.empty() && !packedGlobalIds_.empty()) {
    std::vector<const ExplicitBitVect*> packedFingerprints;
    packedFingerprints.reserve(packedGlobalIds_.size());
    for (MoleculeId id : packedGlobalIds_) {
      packedFingerprints.push_back(patternFingerprints_[static_cast<std::size_t>(id - firstId_)].get());
    }
    packedPatternWords_ = packPatternFingerprintsWordMajor(packedFingerprints);
  }
}

ResidentTargetChunk::~ResidentTargetChunk() = default;

const RDKit::ROMol& ResidentTargetChunk::sourceMol(MoleculeId id) const {
  if (id < firstId_ || id >= endId()) {
    throw std::out_of_range("Molecule ID is outside this resident target chunk");
  }
  return *sourceMolecules_[static_cast<std::size_t>(id - firstId_)];
}

bool ResidentTargetChunk::isGpuSupported(MoleculeId id) const {
  if (id < firstId_ || id >= endId()) {
    throw std::out_of_range("Molecule ID is outside this resident target chunk");
  }
  return gpuSupported_[static_cast<std::size_t>(id - firstId_)] != 0;
}

const ExplicitBitVect* ResidentTargetChunk::patternFingerprint(MoleculeId id) const {
  if (id < firstId_ || id >= endId()) {
    throw std::out_of_range("Molecule ID is outside this resident target chunk");
  }
  if (patternFingerprints_.empty()) {
    return nullptr;
  }
  return patternFingerprints_[static_cast<std::size_t>(id - firstId_)].get();
}

void ResidentTargetChunk::releasePackedData() noexcept {
  packedHost_         = MoleculesHost();
  packedPatternWords_ = std::vector<std::uint64_t>();
}

namespace {

void appendPackedTargets(MoleculesHost& destination, const MoleculesHost& source) {
  if (source.numMolecules() == 0) {
    return;
  }
  const int atomOffset = destination.batchAtomStarts.back();
  destination.atomDataPacked.insert(destination.atomDataPacked.end(),
                                    source.atomDataPacked.begin(),
                                    source.atomDataPacked.end());
  destination.bondTypeCounts.insert(destination.bondTypeCounts.end(),
                                    source.bondTypeCounts.begin(),
                                    source.bondTypeCounts.end());
  destination.targetAtomBonds.insert(destination.targetAtomBonds.end(),
                                     source.targetAtomBonds.begin(),
                                     source.targetAtomBonds.end());
  for (std::size_t molecule = 1; molecule < source.batchAtomStarts.size(); ++molecule) {
    destination.batchAtomStarts.push_back(atomOffset + source.batchAtomStarts[molecule]);
  }
}

// Copy a word-major block of count fingerprints into a word-major array of
// total fingerprints, starting at fingerprint offset.
void appendPatternWords(std::vector<std::uint64_t>&       destination,
                        std::size_t                       total,
                        std::size_t                       offset,
                        const std::vector<std::uint64_t>& source,
                        std::size_t                       count) {
  if (source.size() != count * kPatternFingerprintWords) {
    throw std::logic_error("Pattern fingerprint block does not match its packed target count");
  }
  for (int word = 0; word < kPatternFingerprintWords; ++word) {
    std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(word) * count),
                count,
                destination.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(word) * total + offset));
  }
}

}  // namespace

DeviceTargetSet::DeviceTargetSet(const DeviceTargetSet*                         base,
                                 const std::vector<const ResidentTargetChunk*>& chunks,
                                 bool                                           usePatternFingerprints) {
  std::size_t total = base == nullptr ? 0 : base->size();
  std::size_t atoms = base == nullptr ? 0 : base->host_.totalAtoms();
  for (const auto* chunk : chunks) {
    total += chunk->gpuTargetCount();
    atoms += chunk->packedHost().totalAtoms();
  }
  host_.batchAtomStarts.reserve(total + 1);
  host_.atomDataPacked.reserve(atoms);
  host_.bondTypeCounts.reserve(atoms);
  host_.targetAtomBonds.reserve(atoms);
  targets_.reserve(total);
  ids_.reserve(total);
  if (usePatternFingerprints) {
    patternWords_.resize(total * kPatternFingerprintWords);
  }

  std::size_t offset = 0;
  if (base != nullptr) {
    appendPackedTargets(host_, base->host_);
    targets_ = base->targets_;
    ids_     = base->ids_;
    if (usePatternFingerprints) {
      appendPatternWords(patternWords_, total, 0, base->patternWords_, base->size());
    }
    offset = base->size();
  }
  for (const auto* chunk : chunks) {
    if (chunk->gpuTargetCount() == 0) {
      continue;
    }
    if (!ids_.empty() && chunk->packedGlobalIds().front() <= ids_.back()) {
      throw std::logic_error("Device target sets must be extended in ascending ID order");
    }
    appendPackedTargets(host_, chunk->packedHost());
    targets_.insert(targets_.end(), chunk->supportedTargetPtrs().begin(), chunk->supportedTargetPtrs().end());
    ids_.insert(ids_.end(), chunk->packedGlobalIds().begin(), chunk->packedGlobalIds().end());
    if (usePatternFingerprints) {
      appendPatternWords(patternWords_, total, offset, chunk->packedPatternWords(), chunk->gpuTargetCount());
    }
    offset += chunk->gpuTargetCount();
  }
  if (usePatternFingerprints) {
    patternSlices_  = buildPatternBitSlices(patternWords_, total);
    bitFrequencies_ = nvMolKit::patternBitFrequencies(patternSlices_, total);
  }
}

DeviceTargetSet::~DeviceTargetSet() = default;

void DeviceTargetSet::upload(cudaStream_t stream) {
  if (ids_.empty()) {
    return;
  }
  device_ = std::make_unique<MoleculesDevice>(stream);
  device_->copyFromHost(host_, stream);
  if (!patternSlices_.empty()) {
    patternSlicesDevice_ = AsyncDeviceVector<std::uint32_t>(patternSlices_.size(), stream);
    patternSlicesDevice_.copyFromHost(patternSlices_);
  }
  cudaCheckError(cudaStreamSynchronize(stream));
  // The host slices are rebuilt from patternWords_ when the set is extended.
  patternSlices_ = std::vector<std::uint32_t>();
  // Device allocations outlive the caller-provided upload stream. Release
  // them on the default stream so their lifetime is not tied to that stream.
  device_->setStream(nullptr);
  patternSlicesDevice_.setStream(nullptr);
}

const MoleculesDevice& DeviceTargetSet::device() const {
  if (device_ == nullptr) {
    throw std::logic_error("Device target set has no uploaded targets");
  }
  return *device_;
}

TargetChunkBuilder::TargetChunkBuilder(MoleculeId firstId, std::size_t maxMolecules, bool usePatternFingerprints)
    : firstId_(firstId),
      maxMolecules_(maxMolecules),
      usePatternFingerprints_(usePatternFingerprints) {
  if (maxMolecules_ == 0) {
    throw std::invalid_argument("Target chunk capacity must be greater than zero");
  }
}

TargetChunkBuilder::~TargetChunkBuilder() = default;

MoleculeId TargetChunkBuilder::addMol(const RDKit::ROMol& mol) {
  if (sealed_) {
    throw std::logic_error("Cannot add a molecule to a sealed target chunk builder");
  }
  if (full()) {
    throw std::length_error("Target chunk is full");
  }
  if (sourceMolecules_.size() >= std::numeric_limits<MoleculeId>::max() - firstId_) {
    throw std::overflow_error("Target molecule ID space exhausted");
  }

  const MoleculeId id    = firstId_ + sourceMolecules_.size();
  auto             owned = std::make_unique<RDKit::ROMol>(mol);

  MoleculesHost packedTarget;
  bool          gpuSupported = false;
  try {
    if (owned->getNumAtoms() <= kMaxTargetAtoms && !requiresRDKitFallback(owned.get())) {
      addToBatch(owned.get(), packedTarget);
      gpuSupported = true;
    }
  } catch (const std::runtime_error&) {
    // Target packing has stricter representation limits than RDKit matching.
    // Preserve the source copy and route this ID through the CPU fallback.
    gpuSupported = false;
  }

  std::unique_ptr<ExplicitBitVect> patternFingerprint;
  if (usePatternFingerprints_) {
    patternFingerprint.reset(RDKit::PatternFingerprintMol(*owned));
  }
  sourceMolecules_.push_back(std::move(owned));
  if (usePatternFingerprints_) {
    patternFingerprints_.push_back(std::move(patternFingerprint));
  }
  gpuSupported_.push_back(static_cast<std::uint8_t>(gpuSupported));
  if (gpuSupported) {
    appendPackedTarget(packedHost_, packedTarget);
    packedGlobalIds_.push_back(id);
  } else {
    fallbackGlobalIds_.push_back(id);
  }
  return id;
}

void TargetChunkBuilder::addMols(const std::vector<const RDKit::ROMol*>& molecules, int numThreads) {
  if (sealed_) {
    throw std::logic_error("Cannot add molecules to a sealed target chunk builder");
  }
  if (!empty()) {
    throw std::logic_error("Parallel target chunk construction requires a fresh builder");
  }
  if (molecules.size() > maxMolecules_) {
    throw std::length_error("Target chunk capacity exceeded");
  }
  if (molecules.empty()) {
    return;
  }

  numThreads = std::max(1, numThreads);
  sourceMolecules_.resize(molecules.size());
  gpuSupported_.resize(molecules.size(), 0);
  if (usePatternFingerprints_) {
    patternFingerprints_.resize(molecules.size());
  }
  detail::OpenMPExceptionRegistry exceptionRegistry;

#pragma omp parallel for num_threads(numThreads) schedule(static)
  for (std::int64_t index = 0; index < static_cast<std::int64_t>(molecules.size()); ++index) {
    try {
      const RDKit::ROMol* molecule = molecules[static_cast<std::size_t>(index)];
      if (molecule == nullptr) {
        throw std::invalid_argument("Substructure library molecule cannot be null");
      }
      auto       owned     = std::make_unique<RDKit::ROMol>(*molecule);
      const bool supported = owned->getNumAtoms() <= kMaxTargetAtoms && !requiresRDKitFallback(owned.get());
      sourceMolecules_[static_cast<std::size_t>(index)] = std::move(owned);
      gpuSupported_[static_cast<std::size_t>(index)]    = static_cast<std::uint8_t>(supported);
      if (usePatternFingerprints_) {
        patternFingerprints_[static_cast<std::size_t>(index)].reset(
          RDKit::PatternFingerprintMol(*sourceMolecules_[static_cast<std::size_t>(index)]));
      }
    } catch (const std::runtime_error&) {
      // Match addMol(): representation-limit failures remain available via
      // the RDKit fallback rather than failing library construction.
      try {
        const auto position = static_cast<std::size_t>(index);
        if (sourceMolecules_[position] == nullptr && molecules[position] != nullptr) {
          sourceMolecules_[position] = std::make_unique<RDKit::ROMol>(*molecules[position]);
        }
        if (usePatternFingerprints_ && sourceMolecules_[position] != nullptr &&
            patternFingerprints_[position] == nullptr) {
          patternFingerprints_[position].reset(RDKit::PatternFingerprintMol(*sourceMolecules_[position]));
        }
        gpuSupported_[position] = 0;
      } catch (...) {
        exceptionRegistry.store(std::current_exception());
      }
    } catch (...) {
      exceptionRegistry.store(std::current_exception());
    }
  }
  exceptionRegistry.rethrow();

  std::vector<const RDKit::ROMol*> supportedMolecules;
  supportedMolecules.reserve(molecules.size());
  packedGlobalIds_.reserve(molecules.size());
  fallbackGlobalIds_.reserve(molecules.size());
  for (std::size_t index = 0; index < molecules.size(); ++index) {
    const MoleculeId id = firstId_ + index;
    if (gpuSupported_[index] != 0) {
      supportedMolecules.push_back(sourceMolecules_[index].get());
      packedGlobalIds_.push_back(id);
    } else {
      fallbackGlobalIds_.push_back(id);
    }
  }
  buildTargetBatchParallelInto(packedHost_, numThreads, supportedMolecules, {});
}

std::unique_ptr<ResidentTargetChunk> TargetChunkBuilder::seal() {
  if (sealed_) {
    throw std::logic_error("Target chunk builder has already been sealed");
  }
  auto chunk = std::make_unique<ResidentTargetChunk>(firstId_,
                                                     std::move(sourceMolecules_),
                                                     std::move(packedHost_),
                                                     std::move(packedGlobalIds_),
                                                     std::move(fallbackGlobalIds_),
                                                     std::move(gpuSupported_),
                                                     std::move(patternFingerprints_));
  sealed_    = true;
  return chunk;
}

}  // namespace nvMolKit
