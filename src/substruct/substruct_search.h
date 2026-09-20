// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NVMOLKIT_SUBSTRUCTURE_SEARCH_H
#define NVMOLKIT_SUBSTRUCTURE_SEARCH_H

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "src/substruct/substruct_types.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit

namespace nvMolKit {

struct MoleculesHost;
class MoleculesDevice;
struct SubstructSearchWorkspace;

/**
 * @brief Reusable packed query batch for repeated target screening.
 *
 * Construction packs the queries, uploads them, and prepares recursive SMARTS
 * once. The object is bound to the CUDA device current at construction.
 */
class PreparedSubstructQueries {
 public:
  struct Impl;

  PreparedSubstructQueries(const std::vector<const RDKit::ROMol*>& queries,
                           cudaStream_t                            stream,
                           const SubstructSearchConfig&            config = SubstructSearchConfig{});
  ~PreparedSubstructQueries();

  PreparedSubstructQueries(const PreparedSubstructQueries&)            = delete;
  PreparedSubstructQueries& operator=(const PreparedSubstructQueries&) = delete;
  PreparedSubstructQueries(PreparedSubstructQueries&&) noexcept;
  PreparedSubstructQueries& operator=(PreparedSubstructQueries&&) noexcept;

  [[nodiscard]] std::size_t size() const noexcept;

 private:
  std::unique_ptr<Impl> impl_;

  friend void hasAnySubstructMatch(const std::vector<const RDKit::ROMol*>&,
                                   const PreparedSubstructQueries&,
                                   std::vector<uint8_t>&,
                                   SubstructAlgorithm,
                                   cudaStream_t,
                                   const SubstructSearchConfig&);
  friend void getFirstSubstructMatch(const std::vector<const RDKit::ROMol*>&,
                                     const PreparedSubstructQueries&,
                                     std::vector<int>&,
                                     SubstructAlgorithm,
                                     cudaStream_t,
                                     const SubstructSearchConfig&);
  friend void hasSubstructMatch(const std::vector<const RDKit::ROMol*>&,
                                const PreparedSubstructQueries&,
                                HasSubstructMatchResults&,
                                SubstructAlgorithm,
                                cudaStream_t,
                                const SubstructSearchConfig&);
};

/**
 * @brief Perform batch substructure matching on GPU.
 *
 * Targets are processed in input order and results are returned in the same order.
 *
 * @param targets Vector of target molecule pointers
 * @param queries Vector of query molecule pointers (typically from SMARTS)
 * @param results Output: matches[target][query][match] = vector of target atom indices
 * @param algorithm Algorithm to use for matching
 * @param stream CUDA stream for async operations
 * @param config Execution configuration (threading, batching). Defaults to single-threaded.
 */
void getSubstructMatches(const std::vector<const RDKit::ROMol*>& targets,
                         const std::vector<const RDKit::ROMol*>& queries,
                         SubstructSearchResults&                 results,
                         SubstructAlgorithm                      algorithm,
                         cudaStream_t                            stream,
                         const SubstructSearchConfig&            config = SubstructSearchConfig{});

/**
 * @brief Count substructure matches per (target, query) pair.
 *
 * @param targets Vector of target molecule pointers
 * @param queries Vector of query molecule pointers (typically from SMARTS)
 * @param counts Output: flattened [target * numQueries + query] match counts
 * @param algorithm Algorithm to use for matching
 * @param stream CUDA stream for async operations
 * @param config Execution configuration (threading, batching). Defaults to single-threaded.
 */
void countSubstructMatches(const std::vector<const RDKit::ROMol*>& targets,
                           const std::vector<const RDKit::ROMol*>& queries,
                           std::vector<int>&                       counts,
                           SubstructAlgorithm                      algorithm,
                           cudaStream_t                            stream,
                           const SubstructSearchConfig&            config = SubstructSearchConfig{});

/**
 * @brief Check if targets contain queries as substructures
 *
 * @param targets Vector of target molecule pointers
 * @param queries Vector of query molecule pointers (typically from SMARTS)
 * @param results Output: boolean for each (target, query) pair
 * @param algorithm Algorithm to use for matching
 * @param stream CUDA stream for async operations
 * @param config Execution configuration (threading, batching). Defaults to single-threaded.
 */
void hasSubstructMatch(const std::vector<const RDKit::ROMol*>& targets,
                       const std::vector<const RDKit::ROMol*>& queries,
                       HasSubstructMatchResults&               results,
                       SubstructAlgorithm                      algorithm,
                       cudaStream_t                            stream,
                       const SubstructSearchConfig&            config = SubstructSearchConfig{});

/**
 * Targets already packed and uploaded to one GPU, with the per-target shapes hasSubstructMatch() needs.
 * Immutable: build one per packed batch with makePersistentDeviceTargets().
 */
struct PersistentDeviceTargets;

/**
 * @brief Describe packed targets uploaded to the current GPU for hasSubstructMatch().
 *
 * targets, targetsHost, and targetsDevice must describe the same molecules in the same order, stay alive and
 * unmodified while the batch is used, and targetsDevice must be fully uploaded. Targets above kMaxTargetAtoms atoms
 * or needing the RDKit fallback (see requiresRDKitFallback()) are rejected with std::invalid_argument.
 */
std::shared_ptr<const PersistentDeviceTargets> makePersistentDeviceTargets(
  const std::vector<const RDKit::ROMol*>& targets,
  const MoleculesHost&                    targetsHost,
  const MoleculesDevice&                  targetsDevice);

/**
 * @brief Check one query against persistent device targets.
 *
 * Runs on the GPU holding the batch; config.gpuIds must be empty or name that GPU, and a workspace must belong to
 * it. A workspace keeps executors and pinned buffers alive across calls and serves one call at a time.
 * candidateTargetIndices may select an ordered subset of targets to search; the others report no match. results is
 * resized to the batch size and holds one flag per target.
 */
void hasSubstructMatch(const PersistentDeviceTargets& batch,
                       const RDKit::ROMol&            query,
                       std::vector<uint8_t>&          results,
                       SubstructAlgorithm             algorithm,
                       cudaStream_t                   stream,
                       const SubstructSearchConfig&   config                 = SubstructSearchConfig{},
                       SubstructSearchWorkspace*      workspace              = nullptr,
                       const std::vector<int>*        candidateTargetIndices = nullptr);

/** Create reusable search state bound to one GPU for hasSubstructMatch(). */
std::shared_ptr<SubstructSearchWorkspace> makeSubstructSearchWorkspace(int deviceId);

/** Conservative device bytes one search workspace holds outside recursive queries. */
std::size_t estimateSubstructSearchWorkspaceBytes(const SubstructSearchConfig& config);

/**
 * Additional device bytes a search workspace holds while a query with
 * recursive SMARTS runs. It is released when the search returns.
 */
std::size_t estimateRecursiveScratchBytes(const SubstructSearchConfig& config);

/**
 * Return one flag per target by finding the lowest matching query ID.
 *
 * Active GPU mini-batches can skip later query IDs after a hit. Work already
 * assigned to other mini-batches or executors still completes.
 */
void hasAnySubstructMatch(const std::vector<const RDKit::ROMol*>& targets,
                          const PreparedSubstructQueries&         queries,
                          std::vector<uint8_t>&                   results,
                          SubstructAlgorithm                      algorithm,
                          cudaStream_t                            stream,
                          const SubstructSearchConfig&            config = SubstructSearchConfig{});

/** Check all target/query pairs while reusing a prepared resident query batch. */
void hasSubstructMatch(const std::vector<const RDKit::ROMol*>& targets,
                       const PreparedSubstructQueries&         queries,
                       HasSubstructMatchResults&               results,
                       SubstructAlgorithm                      algorithm,
                       cudaStream_t                            stream,
                       const SubstructSearchConfig&            config = SubstructSearchConfig{});

/** Return the lowest matching query ID per target, or -1 when none matches. */
void getFirstSubstructMatch(const std::vector<const RDKit::ROMol*>& targets,
                            const PreparedSubstructQueries&         queries,
                            std::vector<int>&                       results,
                            SubstructAlgorithm                      algorithm,
                            cudaStream_t                            stream,
                            const SubstructSearchConfig&            config = SubstructSearchConfig{});

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCTURE_SEARCH_H
