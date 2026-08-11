// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FMCS_CUDA_FMCS_PAIR_DATA_CACHE_CUH
#define FMCS_CUDA_FMCS_PAIR_DATA_CACHE_CUH

#include <cooperative_groups.h>

#include <cstddef>
#include <cstdint>

#include "src/mcs/fmcs_cuda/fmcs_match_tables.cuh"
#include "src/mcs/fmcs_cuda/fmcs_topology.cuh"

namespace mcs {
namespace fmcs {

/// Block-owned copy of the immutable topology and compatibility data for one
/// molecule pair.  Keeping the production uint32 layouts lets the existing
/// matcher consume the cached data without tier-specific compact-view paths.
template <int maxAtoms, int maxBonds> struct FmcsPairDataCache {
  static constexpr int kMaxDirectedEdges = 2 * maxBonds;
  static constexpr int kAtomTableWords   = maxAtoms * ((maxAtoms + 31) / 32);
  static constexpr int kBondTableWords   = maxBonds * ((maxBonds + 31) / 32);

  std::uint32_t queryRowOffsets[maxAtoms + 1];
  std::uint32_t queryColIndices[kMaxDirectedEdges];
  std::uint32_t queryBondIndices[kMaxDirectedEdges];
  std::uint32_t queryBondEndpoints[maxBonds];

  std::uint32_t targetRowOffsets[maxAtoms + 1];
  std::uint32_t targetColIndices[kMaxDirectedEdges];
  std::uint32_t targetBondIndices[kMaxDirectedEdges];
  std::uint32_t targetBondEndpoints[maxBonds];

  std::uint32_t atomMatchTable[kAtomTableWords];
  std::uint32_t bondMatchTable[kBondTableWords];
};

/// Cooperatively preload all pair-immutable graph and match-table data, then
/// publish ordinary production views that point at the shared copies.
template <int maxAtoms, int maxBonds, class BlockT>
__device__ __forceinline__ void initializePairDataCacheCooperative(
  const BlockT&                         block,
  int                                   queryNumAtoms,
  int                                   queryNumBonds,
  const std::uint32_t*                  queryRowOffsets,
  const std::uint32_t*                  queryColIndices,
  const std::uint32_t*                  queryBondIndices,
  const std::uint32_t*                  queryBondEndpoints,
  int                                   targetNumAtoms,
  int                                   targetNumBonds,
  const std::uint32_t*                  targetRowOffsets,
  const std::uint32_t*                  targetColIndices,
  const std::uint32_t*                  targetBondIndices,
  const std::uint32_t*                  targetBondEndpoints,
  const PairMatchTablesDevice&          inputTables,
  FmcsPairDataCache<maxAtoms, maxBonds>& cache,
  DeviceCsrView&                        queryView,
  DeviceCsrView&                        targetView,
  PairMatchTablesDevice&                cachedTables) {
  const int threadRank = static_cast<int>(block.thread_rank());
  const int blockSize  = static_cast<int>(block.size());

  const int queryDirectedEdges  = static_cast<int>(queryRowOffsets[queryNumAtoms]);
  const int targetDirectedEdges = static_cast<int>(targetRowOffsets[targetNumAtoms]);

  for (int i = threadRank; i <= queryNumAtoms; i += blockSize)
    cache.queryRowOffsets[i] = queryRowOffsets[i];
  for (int i = threadRank; i < queryDirectedEdges; i += blockSize) {
    cache.queryColIndices[i]  = queryColIndices[i];
    cache.queryBondIndices[i] = queryBondIndices[i];
  }
  for (int i = threadRank; i < queryNumBonds; i += blockSize)
    cache.queryBondEndpoints[i] = queryBondEndpoints[i];

  for (int i = threadRank; i <= targetNumAtoms; i += blockSize)
    cache.targetRowOffsets[i] = targetRowOffsets[i];
  for (int i = threadRank; i < targetDirectedEdges; i += blockSize) {
    cache.targetColIndices[i]  = targetColIndices[i];
    cache.targetBondIndices[i] = targetBondIndices[i];
  }
  for (int i = threadRank; i < targetNumBonds; i += blockSize)
    cache.targetBondEndpoints[i] = targetBondEndpoints[i];

  const int atomTableWords = inputTables.atoms.nRows * inputTables.atoms.wordsPerRow;
  for (int i = threadRank; i < atomTableWords; i += blockSize)
    cache.atomMatchTable[i] = inputTables.atoms.data[i];
  const int bondTableWords = inputTables.bonds.nRows * inputTables.bonds.wordsPerRow;
  for (int i = threadRank; i < bondTableWords; i += blockSize)
    cache.bondMatchTable[i] = inputTables.bonds.data[i];

  block.sync();
  if (threadRank == 0) {
    queryView = {cache.queryRowOffsets,
                 cache.queryColIndices,
                 cache.queryBondIndices,
                 cache.queryBondEndpoints,
                 queryNumAtoms,
                 queryNumBonds};
    targetView = {cache.targetRowOffsets,
                  cache.targetColIndices,
                  cache.targetBondIndices,
                  cache.targetBondEndpoints,
                  targetNumAtoms,
                  targetNumBonds};
    cachedTables.atoms = {
      cache.atomMatchTable, inputTables.atoms.nRows, inputTables.atoms.nCols, inputTables.atoms.wordsPerRow};
    cachedTables.bonds = {
      cache.bondMatchTable, inputTables.bonds.nRows, inputTables.bonds.nCols, inputTables.bonds.wordsPerRow};
  }
  block.sync();
}

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_PAIR_DATA_CACHE_CUH
