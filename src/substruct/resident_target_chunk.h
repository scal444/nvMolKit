// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_RESIDENT_TARGET_CHUNK_H
#define NVMOLKIT_RESIDENT_TARGET_CHUNK_H

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "src/substruct/molecules.h"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit
class ExplicitBitVect;

namespace nvMolKit {

using MoleculeId = std::uint64_t;

/**
 * @brief A sealed, host-side target chunk.
 *
 * The chunk owns an RDKit molecule copy for every ID in [firstId(), endId()).
 * GPU-supported molecules are additionally packed in packedHost(), with
 * packedGlobalIds()[i] identifying packed molecule i. Molecules which cannot
 * use the GPU representation remain available through sourceMol() and are
 * listed in fallbackGlobalIds(). Chunks are immutable except that their packed
 * copies may be released once a DeviceTargetSet has absorbed them.
 */
class ResidentTargetChunk {
 public:
  ResidentTargetChunk(const ResidentTargetChunk&)            = delete;
  ResidentTargetChunk& operator=(const ResidentTargetChunk&) = delete;
  ResidentTargetChunk(ResidentTargetChunk&&)                 = delete;
  ResidentTargetChunk& operator=(ResidentTargetChunk&&)      = delete;
  ~ResidentTargetChunk();

  ResidentTargetChunk(MoleculeId                                    firstId,
                      std::vector<std::unique_ptr<RDKit::ROMol>>    sourceMolecules,
                      MoleculesHost                                 packedHost,
                      std::vector<MoleculeId>                       packedGlobalIds,
                      std::vector<MoleculeId>                       fallbackGlobalIds,
                      std::vector<std::uint8_t>                     gpuSupported,
                      std::vector<std::unique_ptr<ExplicitBitVect>> patternFingerprints);

  [[nodiscard]] MoleculeId  firstId() const noexcept { return firstId_; }
  [[nodiscard]] MoleculeId  endId() const noexcept { return firstId_ + sourceMolecules_.size(); }
  [[nodiscard]] std::size_t size() const noexcept { return sourceMolecules_.size(); }
  [[nodiscard]] std::size_t gpuTargetCount() const noexcept { return packedGlobalIds_.size(); }
  [[nodiscard]] std::size_t fallbackCount() const noexcept { return fallbackGlobalIds_.size(); }
  [[nodiscard]] bool        empty() const noexcept { return sourceMolecules_.empty(); }

  [[nodiscard]] const RDKit::ROMol&    sourceMol(MoleculeId id) const;
  [[nodiscard]] bool                   isGpuSupported(MoleculeId id) const;
  [[nodiscard]] const ExplicitBitVect* patternFingerprint(MoleculeId id) const;

  [[nodiscard]] const MoleculesHost&                    packedHost() const noexcept { return packedHost_; }
  [[nodiscard]] const std::vector<const RDKit::ROMol*>& supportedTargetPtrs() const noexcept {
    return supportedTargetPtrs_;
  }
  [[nodiscard]] const std::vector<MoleculeId>& packedGlobalIds() const noexcept { return packedGlobalIds_; }
  [[nodiscard]] const std::vector<MoleculeId>& fallbackGlobalIds() const noexcept { return fallbackGlobalIds_; }

  /** Word-major pattern fingerprints of the packed targets; empty when fingerprints are disabled. */
  [[nodiscard]] const std::vector<std::uint64_t>& packedPatternWords() const noexcept { return packedPatternWords_; }

  /** Drop the packed copies after a DeviceTargetSet has absorbed them. */
  void releasePackedData() noexcept;

 private:
  MoleculeId                                    firstId_ = 0;
  std::vector<std::unique_ptr<RDKit::ROMol>>    sourceMolecules_;
  MoleculesHost                                 packedHost_;
  std::vector<const RDKit::ROMol*>              supportedTargetPtrs_;
  std::vector<MoleculeId>                       packedGlobalIds_;
  std::vector<MoleculeId>                       fallbackGlobalIds_;
  std::vector<std::uint8_t>                     gpuSupported_;
  std::vector<std::unique_ptr<ExplicitBitVect>> patternFingerprints_;
  std::vector<std::uint64_t>                    packedPatternWords_;
};

/**
 * @brief Every GPU-supported target on one device, resident as a single batch.
 *
 * Targets are concatenated in ascending ID order, so one screen and one match
 * launch per query cover the whole device regardless of chunk count.
 */
class DeviceTargetSet {
 public:
  /**
   * Concatenate base (may be null) with chunks, whose IDs must all follow the
   * base's. Nothing is uploaded until upload().
   */
  DeviceTargetSet(const DeviceTargetSet*                         base,
                  const std::vector<const ResidentTargetChunk*>& chunks,
                  bool                                           usePatternFingerprints);
  ~DeviceTargetSet();

  DeviceTargetSet(const DeviceTargetSet&)            = delete;
  DeviceTargetSet& operator=(const DeviceTargetSet&) = delete;

  /** Copy the set to the current device and wait for the copy. */
  void upload(cudaStream_t stream);

  [[nodiscard]] std::size_t                             size() const noexcept { return ids_.size(); }
  [[nodiscard]] const std::vector<MoleculeId>&          ids() const noexcept { return ids_; }
  [[nodiscard]] const std::vector<const RDKit::ROMol*>& targets() const noexcept { return targets_; }
  [[nodiscard]] const MoleculesHost&                    host() const noexcept { return host_; }
  [[nodiscard]] const MoleculesDevice&                  device() const;
  [[nodiscard]] TargetMoleculesDeviceView deviceView() const { return device().view<MoleculeType::Target>(); }
  /** Word-major device fingerprints, or null when fingerprints are disabled. */
  [[nodiscard]] const std::uint64_t*      devicePatternWords() const noexcept { return patternWordsDevice_.data(); }

 private:
  MoleculesHost                    host_;
  std::vector<const RDKit::ROMol*> targets_;
  std::vector<MoleculeId>          ids_;
  std::vector<std::uint64_t>       patternWords_;
  std::unique_ptr<MoleculesDevice> device_;
  AsyncDeviceVector<std::uint64_t> patternWordsDevice_;
};

/**
 * @brief Mutable CPU-side builder for one stable-ID target chunk.
 */
class TargetChunkBuilder {
 public:
  explicit TargetChunkBuilder(MoleculeId  firstId,
                              std::size_t maxMolecules           = std::numeric_limits<std::size_t>::max(),
                              bool        usePatternFingerprints = true);
  ~TargetChunkBuilder();

  TargetChunkBuilder(const TargetChunkBuilder&)            = delete;
  TargetChunkBuilder& operator=(const TargetChunkBuilder&) = delete;
  TargetChunkBuilder(TargetChunkBuilder&&)                 = delete;
  TargetChunkBuilder& operator=(TargetChunkBuilder&&)      = delete;

  /**
   * Copy and append a molecule, returning its stable global library ID.
   * Molecules which cannot be represented by the GPU packing format are kept
   * in the chunk and marked for RDKit fallback.
   */
  MoleculeId addMol(const RDKit::ROMol& mol);

  /**
   * Copy and pack a fresh builder's molecules with the requested CPU thread
   * count while preserving their contiguous stable IDs.
   */
  void addMols(const std::vector<const RDKit::ROMol*>& molecules, int numThreads);

  /** Seal the builder and transfer its storage into a resident chunk. */
  [[nodiscard]] std::unique_ptr<ResidentTargetChunk> seal();

  [[nodiscard]] MoleculeId  firstId() const noexcept { return firstId_; }
  [[nodiscard]] MoleculeId  nextId() const noexcept { return firstId_ + sourceMolecules_.size(); }
  [[nodiscard]] std::size_t size() const noexcept { return sourceMolecules_.size(); }
  [[nodiscard]] std::size_t maxMolecules() const noexcept { return maxMolecules_; }
  [[nodiscard]] bool        empty() const noexcept { return sourceMolecules_.empty(); }
  [[nodiscard]] bool        full() const noexcept { return sourceMolecules_.size() == maxMolecules_; }
  [[nodiscard]] bool        sealed() const noexcept { return sealed_; }

 private:
  MoleculeId  firstId_                = 0;
  std::size_t maxMolecules_           = 0;
  bool        sealed_                 = false;
  bool        usePatternFingerprints_ = true;

  std::vector<std::unique_ptr<RDKit::ROMol>>    sourceMolecules_;
  MoleculesHost                                 packedHost_;
  std::vector<MoleculeId>                       packedGlobalIds_;
  std::vector<MoleculeId>                       fallbackGlobalIds_;
  std::vector<std::uint8_t>                     gpuSupported_;
  std::vector<std::unique_ptr<ExplicitBitVect>> patternFingerprints_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_RESIDENT_TARGET_CHUNK_H
