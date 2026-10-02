// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/substruct/target_chunk.h"

#include <GraphMol/ROMol.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "src/substruct/substruct_constants.h"
#include "src/substruct/substruct_search.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/openmp_helpers.h"

namespace nvMolKit {

TargetChunk::TargetChunk(MoleculeId                                 firstId,
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
    throw std::logic_error("Target chunk support metadata is inconsistent");
  }
  if (packedHost_.numMolecules() != packedGlobalIds_.size()) {
    throw std::logic_error("Target chunk packed-ID mapping is inconsistent");
  }
  if (packedGlobalIds_.size() + fallbackGlobalIds_.size() != sourceMolecules_.size()) {
    throw std::logic_error("Target chunk molecule classification is incomplete");
  }

  supportedTargetPtrs_.reserve(packedGlobalIds_.size());
  for (MoleculeId id : packedGlobalIds_) {
    if (id < firstId_ || id >= endId()) {
      throw std::logic_error("Target chunk packed ID is outside its source range");
    }
    supportedTargetPtrs_.push_back(sourceMolecules_[static_cast<std::size_t>(id - firstId_)].get());
  }
}

TargetChunk::~TargetChunk() = default;

std::size_t TargetChunk::offsetOf(MoleculeId id) const {
  if (id < firstId_ || id >= endId()) {
    throw std::out_of_range("Molecule ID is outside this target chunk");
  }
  return static_cast<std::size_t>(id - firstId_);
}

const RDKit::ROMol& TargetChunk::sourceMol(MoleculeId id) const {
  return *sourceMolecules_[offsetOf(id)];
}

bool TargetChunk::isGpuSupported(MoleculeId id) const {
  return gpuSupported_[offsetOf(id)] != 0;
}

void TargetChunk::releasePackedData() noexcept {
  packedHost_ = MoleculesHost();
}

DeviceTargetSet::DeviceTargetSet(const DeviceTargetSet* base, const std::vector<const TargetChunk*>& chunks) {
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

  if (base != nullptr) {
    mergeBatch(host_, base->host_);
    targets_ = base->targets_;
    ids_     = base->ids_;
  }
  for (const auto* chunk : chunks) {
    if (chunk->gpuTargetCount() == 0) {
      continue;
    }
    if (!ids_.empty() && chunk->packedGlobalIds().front() <= ids_.back()) {
      throw std::logic_error("Device target sets must be extended in ascending ID order");
    }
    mergeBatch(host_, chunk->packedHost());
    targets_.insert(targets_.end(), chunk->supportedTargetPtrs().begin(), chunk->supportedTargetPtrs().end());
    ids_.insert(ids_.end(), chunk->packedGlobalIds().begin(), chunk->packedGlobalIds().end());
  }
}

DeviceTargetSet::~DeviceTargetSet() = default;

void DeviceTargetSet::upload(cudaStream_t stream) {
  if (ids_.empty()) {
    return;
  }
  device_ = std::make_unique<MoleculesDevice>(stream);
  device_->copyFromHost(host_, stream);
  cudaCheckError(cudaStreamSynchronize(stream));
  // Device allocations outlive the caller-provided upload stream. Release
  // them on the default stream so their lifetime is not tied to that stream.
  device_->setStream(nullptr);
  persistentTargets_ = makePersistentDeviceTargets(targets_, host_, *device_);
}

const MoleculesDevice& DeviceTargetSet::device() const {
  if (device_ == nullptr) {
    throw std::logic_error("Device target set has no uploaded targets");
  }
  return *device_;
}

const PersistentDeviceTargets& DeviceTargetSet::persistentTargets() const {
  if (persistentTargets_ == nullptr) {
    throw std::logic_error("Device target set has no uploaded targets");
  }
  return *persistentTargets_;
}

namespace {

//! An owned copy of a library molecule and whether the packed target format can represent it.
struct StagedTarget {
  std::unique_ptr<RDKit::ROMol> mol;
  bool                          gpuSupported = false;
};

StagedTarget stageTarget(const RDKit::ROMol& source) {
  StagedTarget staged;
  staged.mol = std::make_unique<RDKit::ROMol>(source);
  try {
    staged.gpuSupported = staged.mol->getNumAtoms() <= kMaxTargetAtoms && !requiresRDKitFallback(staged.mol.get());
  } catch (const std::runtime_error&) {
    // Representation-limit failures keep the copy and route the target through the RDKit fallback.
    staged.gpuSupported = false;
  }
  return staged;
}

}  // namespace

TargetChunkBuilder::TargetChunkBuilder(MoleculeId firstId, std::size_t maxMolecules)
    : firstId_(firstId),
      maxMolecules_(maxMolecules) {
  if (maxMolecules_ == 0) {
    throw std::invalid_argument("Target chunk capacity must be greater than zero");
  }
}

TargetChunkBuilder::~TargetChunkBuilder() = default;

MoleculeId TargetChunkBuilder::append(std::unique_ptr<RDKit::ROMol> mol, bool gpuSupported) {
  const MoleculeId id = nextId();
  sourceMolecules_.push_back(std::move(mol));
  gpuSupported_.push_back(static_cast<std::uint8_t>(gpuSupported));
  (gpuSupported ? packedGlobalIds_ : fallbackGlobalIds_).push_back(id);
  return id;
}

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

  StagedTarget  staged = stageTarget(mol);
  MoleculesHost packedTarget;
  if (staged.gpuSupported) {
    try {
      addToBatch(staged.mol.get(), packedTarget);
    } catch (const std::runtime_error&) {
      // Target packing has stricter representation limits than RDKit matching.
      // Preserve the source copy and route this ID through the CPU fallback.
      staged.gpuSupported = false;
    }
  }
  const bool packed = staged.gpuSupported;
  const auto id     = append(std::move(staged.mol), packed);
  if (packed) {
    mergeBatch(packedHost_, packedTarget);
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
  std::vector<StagedTarget>       staged(molecules.size());
  detail::OpenMPExceptionRegistry exceptionRegistry;

#pragma omp parallel for num_threads(numThreads) schedule(static)
  for (std::int64_t index = 0; index < static_cast<std::int64_t>(molecules.size()); ++index) {
    try {
      const RDKit::ROMol* molecule = molecules[static_cast<std::size_t>(index)];
      if (molecule == nullptr) {
        throw std::invalid_argument("Substructure library molecule cannot be null");
      }
      staged[static_cast<std::size_t>(index)] = stageTarget(*molecule);
    } catch (...) {
      exceptionRegistry.store(std::current_exception());
    }
  }
  exceptionRegistry.rethrow();

  std::vector<const RDKit::ROMol*> supportedMolecules;
  supportedMolecules.reserve(molecules.size());
  sourceMolecules_.reserve(molecules.size());
  gpuSupported_.reserve(molecules.size());
  for (auto& target : staged) {
    if (target.gpuSupported) {
      supportedMolecules.push_back(target.mol.get());
    }
    append(std::move(target.mol), target.gpuSupported);
  }
  buildTargetBatchParallelInto(packedHost_, numThreads, supportedMolecules, {});
}

std::unique_ptr<TargetChunk> TargetChunkBuilder::seal() {
  if (sealed_) {
    throw std::logic_error("Target chunk builder has already been sealed");
  }
  auto chunk = std::make_unique<TargetChunk>(firstId_,
                                             std::move(sourceMolecules_),
                                             std::move(packedHost_),
                                             std::move(packedGlobalIds_),
                                             std::move(fallbackGlobalIds_),
                                             std::move(gpuSupported_));
  sealed_    = true;
  return chunk;
}

}  // namespace nvMolKit
