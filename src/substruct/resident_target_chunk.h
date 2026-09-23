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

namespace RDKit {
class ROMol;
}  // namespace RDKit

namespace nvMolKit {

using MoleculeId = std::uint64_t;

/**
 * @brief A sealed target chunk whose GPU-supported molecules can be made resident.
 *
 * The chunk owns an RDKit molecule copy for every ID in [firstId(), endId()).
 * GPU-supported molecules are additionally packed in packedHost(), with
 * packedGlobalIds()[i] identifying packed molecule i. Molecules which cannot
 * use the GPU representation remain available through sourceMol() and are
 * listed in fallbackGlobalIds().
 *
 * Source data and ID mappings are immutable after construction. Device upload
 * is a two-step operation: beginUpload() enqueues the copy and commit() waits
 * for it before making deviceView() available to searches.
 */
class ResidentTargetChunk {
 public:
  enum class State : std::uint8_t {
    Sealed,
    Uploading,
    Committed,
    Failed,
  };

  ResidentTargetChunk(const ResidentTargetChunk&)            = delete;
  ResidentTargetChunk& operator=(const ResidentTargetChunk&) = delete;
  ResidentTargetChunk(ResidentTargetChunk&&)                 = delete;
  ResidentTargetChunk& operator=(ResidentTargetChunk&&)      = delete;
  ~ResidentTargetChunk();

  ResidentTargetChunk(MoleculeId                                 firstId,
                      std::vector<std::unique_ptr<RDKit::ROMol>> sourceMolecules,
                      MoleculesHost                              packedHost,
                      std::vector<MoleculeId>                    packedGlobalIds,
                      std::vector<MoleculeId>                    fallbackGlobalIds,
                      std::vector<std::uint8_t>                  gpuSupported);

  /** Enqueue all packed target arrays on stream without synchronizing it. */
  void beginUpload(cudaStream_t stream);

  /** Wait for a previously enqueued upload and publish the device view. */
  void commit();

  /** Convenience barrier equivalent to beginUpload(stream) followed by commit(). */
  void finalize(cudaStream_t stream);

  [[nodiscard]] State       state() const noexcept { return state_; }
  [[nodiscard]] MoleculeId  firstId() const noexcept { return firstId_; }
  [[nodiscard]] MoleculeId  endId() const noexcept { return firstId_ + sourceMolecules_.size(); }
  [[nodiscard]] std::size_t size() const noexcept { return sourceMolecules_.size(); }
  [[nodiscard]] std::size_t gpuTargetCount() const noexcept { return packedGlobalIds_.size(); }
  [[nodiscard]] std::size_t fallbackCount() const noexcept { return fallbackGlobalIds_.size(); }
  [[nodiscard]] bool        empty() const noexcept { return sourceMolecules_.empty(); }

  [[nodiscard]] const RDKit::ROMol& sourceMol(MoleculeId id) const;
  [[nodiscard]] bool                isGpuSupported(MoleculeId id) const;

  [[nodiscard]] const MoleculesHost&                    packedHost() const noexcept { return packedHost_; }
  [[nodiscard]] const std::vector<const RDKit::ROMol*>& supportedTargetPtrs() const noexcept {
    return supportedTargetPtrs_;
  }
  [[nodiscard]] const std::vector<MoleculeId>& packedGlobalIds() const noexcept { return packedGlobalIds_; }
  [[nodiscard]] const std::vector<MoleculeId>& fallbackGlobalIds() const noexcept { return fallbackGlobalIds_; }

  /**
   * @throws std::logic_error unless commit() completed successfully.
   */
  [[nodiscard]] TargetMoleculesDeviceView deviceView() const;

  /**
   * Return the owning device storage for existing resident-search entry points.
   * @throws std::logic_error unless commit() completed successfully or when
   * the chunk contains no GPU-supported targets.
   */
  [[nodiscard]] const MoleculesDevice& deviceStorage() const;

 private:
  MoleculeId                                 firstId_ = 0;
  std::vector<std::unique_ptr<RDKit::ROMol>> sourceMolecules_;
  MoleculesHost                              packedHost_;
  std::vector<const RDKit::ROMol*>           supportedTargetPtrs_;
  std::vector<MoleculeId>                    packedGlobalIds_;
  std::vector<MoleculeId>                    fallbackGlobalIds_;
  std::vector<std::uint8_t>                  gpuSupported_;

  State                            state_ = State::Sealed;
  std::unique_ptr<MoleculesDevice> packedDevice_;
  std::unique_ptr<ScopedCudaEvent> uploadComplete_;
};

/**
 * @brief Mutable CPU-side builder for one stable-ID target chunk.
 */
class TargetChunkBuilder {
 public:
  explicit TargetChunkBuilder(MoleculeId firstId, std::size_t maxMolecules = std::numeric_limits<std::size_t>::max());
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
  MoleculeId  firstId_      = 0;
  std::size_t maxMolecules_ = 0;
  bool        sealed_       = false;

  std::vector<std::unique_ptr<RDKit::ROMol>> sourceMolecules_;
  MoleculesHost                              packedHost_;
  std::vector<MoleculeId>                    packedGlobalIds_;
  std::vector<MoleculeId>                    fallbackGlobalIds_;
  std::vector<std::uint8_t>                  gpuSupported_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_RESIDENT_TARGET_CHUNK_H
