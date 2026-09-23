// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/substruct/resident_target_chunk.h"

#include <GraphMol/ROMol.h>

#include <algorithm>
#include <stdexcept>
#include <utility>

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

ResidentTargetChunk::ResidentTargetChunk(MoleculeId                                 firstId,
                                         std::vector<std::unique_ptr<RDKit::ROMol>> sourceMolecules,
                                         MoleculesHost                              packedHost,
                                         std::vector<MoleculeId>                    packedGlobalIds,
                                         std::vector<MoleculeId>                    fallbackGlobalIds,
                                         std::vector<std::uint8_t>                  gpuSupported)
    : firstId_(firstId),
      sourceMolecules_(std::move(sourceMolecules)),
      packedHost_(std::move(packedHost)),
      packedGlobalIds_(std::move(packedGlobalIds)),
      fallbackGlobalIds_(std::move(fallbackGlobalIds)),
      gpuSupported_(std::move(gpuSupported)) {
  if (sourceMolecules_.size() != gpuSupported_.size()) {
    throw std::logic_error("Resident target chunk support metadata is inconsistent");
  }
  if (packedHost_.numMolecules() != packedGlobalIds_.size()) {
    throw std::logic_error("Resident target chunk packed-ID mapping is inconsistent");
  }
  if (packedGlobalIds_.size() + fallbackGlobalIds_.size() != sourceMolecules_.size()) {
    throw std::logic_error("Resident target chunk molecule classification is incomplete");
  }

  supportedTargetPtrs_.reserve(packedGlobalIds_.size());
  for (MoleculeId id : packedGlobalIds_) {
    if (id < firstId_ || id >= endId()) {
      throw std::logic_error("Resident target chunk packed ID is outside its source range");
    }
    supportedTargetPtrs_.push_back(sourceMolecules_[static_cast<std::size_t>(id - firstId_)].get());
  }
}

ResidentTargetChunk::~ResidentTargetChunk() = default;

void ResidentTargetChunk::beginUpload(cudaStream_t stream) {
  if (state_ != State::Sealed) {
    throw std::logic_error("Resident target chunk upload can only begin from the sealed state");
  }

  if (packedGlobalIds_.empty()) {
    state_ = State::Committed;
    return;
  }

  state_ = State::Uploading;
  try {
    uploadComplete_ = std::make_unique<ScopedCudaEvent>();
    packedDevice_   = std::make_unique<MoleculesDevice>(stream);
    packedDevice_->copyFromHost(packedHost_, stream);
    cudaCheckError(cudaEventRecord(uploadComplete_->event(), stream));
  } catch (...) {
    state_ = State::Failed;
    throw;
  }
}

void ResidentTargetChunk::commit() {
  if (state_ == State::Committed) {
    return;
  }
  if (state_ != State::Uploading) {
    throw std::logic_error("Resident target chunk commit requires an upload in progress");
  }

  try {
    cudaCheckError(cudaEventSynchronize(uploadComplete_->event()));
    // Device allocations outlive the caller-provided upload stream. Release
    // them on the default stream so chunk lifetime is not tied to that stream.
    packedDevice_->setStream(nullptr);
    state_ = State::Committed;
  } catch (...) {
    state_ = State::Failed;
    throw;
  }
}

void ResidentTargetChunk::finalize(cudaStream_t stream) {
  beginUpload(stream);
  commit();
}

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

TargetMoleculesDeviceView ResidentTargetChunk::deviceView() const {
  if (state_ != State::Committed) {
    throw std::logic_error("Resident target chunk device view requested before commit");
  }
  if (packedDevice_ == nullptr) {
    return TargetMoleculesDeviceView{nullptr, 0, nullptr, nullptr, nullptr};
  }
  return packedDevice_->view<MoleculeType::Target>();
}

const MoleculesDevice& ResidentTargetChunk::deviceStorage() const {
  if (state_ != State::Committed) {
    throw std::logic_error("Resident target chunk device storage requested before commit");
  }
  if (packedDevice_ == nullptr) {
    throw std::logic_error("Resident target chunk has no GPU-supported targets");
  }
  return *packedDevice_;
}

TargetChunkBuilder::TargetChunkBuilder(MoleculeId firstId, std::size_t maxMolecules)
    : firstId_(firstId),
      maxMolecules_(maxMolecules) {
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

  sourceMolecules_.push_back(std::move(owned));
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
    } catch (const std::runtime_error&) {
      // Match addMol(): representation-limit failures remain available via
      // the RDKit fallback rather than failing library construction.
      try {
        const auto position = static_cast<std::size_t>(index);
        if (sourceMolecules_[position] == nullptr && molecules[position] != nullptr) {
          sourceMolecules_[position] = std::make_unique<RDKit::ROMol>(*molecules[position]);
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
                                                     std::move(gpuSupported_));
  sealed_    = true;
  return chunk;
}

}  // namespace nvMolKit
