// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_PATTERN_SCREEN_H
#define NVMOLKIT_PATTERN_SCREEN_H

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/utils/device.h"
#include "src/utils/device_vector.h"

class ExplicitBitVect;

namespace nvMolKit {

//! RDKit's default pattern fingerprint is 2048 bits.
constexpr int kPatternFingerprintBits  = 2048;
constexpr int kPatternFingerprintWords = kPatternFingerprintBits / 64;

/**
 * Pack pattern fingerprints word-major: word w of fingerprint i is stored at
 * [w * fingerprints.size() + i], so a thread per target reads coalesced words.
 */
std::vector<std::uint64_t> packPatternFingerprintsWordMajor(const std::vector<const ExplicitBitVect*>& fingerprints);

//! Query-side screen parameters. Only nonzero query words are tested.
struct PatternScreenQuery {
  std::uint64_t words[kPatternFingerprintWords];
  std::uint8_t  wordIndices[kPatternFingerprintWords];
  int           numWords = 0;
  int           numAtoms = 0;
};

/**
 * Build screen parameters for a query. A null fingerprint screens by atom count only.
 */
PatternScreenQuery makePatternScreenQuery(const ExplicitBitVect* fingerprint, int numQueryAtoms);

/**
 * Per-query device and pinned host storage for screening every resident chunk
 * on one GPU. Screening is substructure-safe: a target is rejected only when
 * it has fewer atoms than the query or lacks a query pattern-fingerprint bit.
 */
class PatternScreenWorkspace {
 public:
  explicit PatternScreenWorkspace(int deviceId);
  ~PatternScreenWorkspace() noexcept;

  PatternScreenWorkspace(const PatternScreenWorkspace&)            = delete;
  PatternScreenWorkspace& operator=(const PatternScreenWorkspace&) = delete;

  /**
   * Start a query over numChunks chunks holding totalTargets targets, at most
   * maxChunkTargets per chunk: grow buffers as needed and clear chunk counts.
   */
  void prepare(std::size_t numChunks, std::size_t totalTargets, std::size_t maxChunkTargets);

  /**
   * Enqueue the screen of one chunk. Selected packed-target indices, in
   * ascending order, are written at targetOffset; chunkIndex receives the count.
   * targetWords may be null to screen by atom count only.
   */
  void enqueueChunk(std::size_t               chunkIndex,
                    std::size_t               targetOffset,
                    const std::uint64_t*      targetWords,
                    const int*                batchAtomStarts,
                    int                       numTargets,
                    const PatternScreenQuery& query);

  /** Copy results to the host and wait. After this, count() and indices() are valid. */
  void collect(std::size_t numChunks, const std::vector<std::size_t>& targetOffsets);

  [[nodiscard]] int        count(std::size_t chunkIndex) const { return hostCounts_[chunkIndex]; }
  [[nodiscard]] const int* indices(std::size_t targetOffset) const { return hostIndices_ + targetOffset; }
  [[nodiscard]] int        deviceId() const noexcept { return deviceId_; }

 private:
  int                             deviceId_;
  ScopedStream                    stream_;
  AsyncDeviceVector<int>          indices_;
  AsyncDeviceVector<int>          counts_;
  AsyncDeviceVector<std::uint8_t> tempStorage_;
  int*                            hostCounts_      = nullptr;
  int*                            hostIndices_     = nullptr;
  std::size_t                     hostCountsSize_  = 0;
  std::size_t                     hostIndicesSize_ = 0;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_PATTERN_SCREEN_H
