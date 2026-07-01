// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FMCS_CUDA_EXPERIMENTAL_FMCS_CANDIDATE_COUNT_CACHE_CUH
#define FMCS_CUDA_EXPERIMENTAL_FMCS_CANDIDATE_COUNT_CACHE_CUH

#include <cstdint>

#include "fmcs_cuda/fmcs_match.cuh"

namespace mcs {
namespace fmcs {

/// EXPERIMENTAL: cached wrapper for fallback candidate counts. The active
/// kernel recomputes counts per warp group because a block-shared cache races
/// when groups enter fallback concurrently. A future use must provide
/// group-private storage or explicit cross-group synchronization.
template <int maxAtoms, int maxBonds, int maxTA, class TargetTopology, class GroupT>
__device__ __forceinline__ int countCandidateTargetAtomsCachedCooperative(
  const GroupT&                                             group,
  const int                                                 queryAtomIdx,
  const TargetTopology&                                     targetTopology,
  const PairMatchTablesDevice&                              tables,
  const FmcsSubstructureScratch<maxAtoms, maxBonds, maxTA>& scratch,
  std::uint8_t*                                             candidateCountCache) {
  constexpr std::uint8_t kEmpty   = 0xFFu;
  const int              laneRank = static_cast<int>(group.thread_rank());
  if (candidateCountCache == nullptr) {
    return countCandidateTargetAtomsCooperative(group, queryAtomIdx, targetTopology, tables, scratch);
  }
  const int requiredDegree = static_cast<int>(scratch.seedDegree[queryAtomIdx]);
  const int cacheIdx       = queryAtomIdx * (maxAtoms + 1) + requiredDegree;

  int candidateCount = kEmpty;
  if (laneRank == 0) {
    candidateCount = static_cast<int>(candidateCountCache[cacheIdx]);
  }
  candidateCount = group.shfl(candidateCount, 0);
  if (candidateCount != kEmpty) {
    return candidateCount;
  }

  candidateCount = countCandidateTargetAtomsCooperative(group, queryAtomIdx, targetTopology, tables, scratch);
  if (laneRank == 0) {
    candidateCountCache[cacheIdx] = static_cast<std::uint8_t>(candidateCount);
  }
  return candidateCount;
}

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_EXPERIMENTAL_FMCS_CANDIDATE_COUNT_CACHE_CUH
