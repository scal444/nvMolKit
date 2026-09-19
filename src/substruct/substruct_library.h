// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_SUBSTRUCT_LIBRARY_H
#define NVMOLKIT_SUBSTRUCT_LIBRARY_H

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "src/substruct/substruct_results.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit

namespace nvMolKit {

/**
 * A persistent, single-GPU collection of target molecules for repeated
 * substructure queries.
 *
 * addMol() stages an owned copy on the CPU. finalize() uploads all pending
 * chunks and atomically publishes them to subsequent queries. Previously
 * finalized chunks remain queryable while more molecules are pending.
 * Queries may execute concurrently; addMol() and finalize() wait for active
 * queries and exclude new queries until the operation completes.
 */
class SubstructLibrary {
 public:
  explicit SubstructLibrary(std::size_t chunkSize = 65536, SubstructSearchConfig config = SubstructSearchConfig{});
  ~SubstructLibrary();

  SubstructLibrary(const SubstructLibrary&)            = delete;
  SubstructLibrary& operator=(const SubstructLibrary&) = delete;
  SubstructLibrary(SubstructLibrary&&)                 = delete;
  SubstructLibrary& operator=(SubstructLibrary&&)      = delete;

  /** Copy a molecule into the pending generation and return its stable ID. */
  unsigned int addMol(const RDKit::ROMol& molecule);

  /** Upload and publish every pending molecule. This is a synchronization barrier. */
  void finalize(cudaStream_t stream = nullptr);

  /** Number of molecules visible to queries. */
  [[nodiscard]] std::size_t size() const;

  /** Number of staged molecules not yet visible to queries. */
  [[nodiscard]] std::size_t pendingSize() const;

  /** Return matching molecule IDs in insertion order. Zero requests no results. */
  [[nodiscard]] std::vector<unsigned int> getMatches(const RDKit::ROMol& query,
                                                     int                 maxResults = -1,
                                                     cudaStream_t        stream     = nullptr) const;

  [[nodiscard]] std::size_t countMatches(const RDKit::ROMol& query, cudaStream_t stream = nullptr) const;
  [[nodiscard]] bool        hasMatch(const RDKit::ROMol& query, cudaStream_t stream = nullptr) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_LIBRARY_H
