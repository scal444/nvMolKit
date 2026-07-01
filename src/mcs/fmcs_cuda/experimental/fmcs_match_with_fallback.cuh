// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FMCS_CUDA_EXPERIMENTAL_FMCS_MATCH_WITH_FALLBACK_CUH
#define FMCS_CUDA_EXPERIMENTAL_FMCS_MATCH_WITH_FALLBACK_CUH

#include "fmcs_cuda/fmcs_match.cuh"

namespace mcs {
namespace fmcs {

// EXPERIMENTAL: Composite matcher retained for direct testing and potential
// future callers. Production code selects its matching path explicitly.
template <int maxAtoms,
          int maxBonds,
          int maxTA,
          int maxTB,
          class QueryTopology,
          class TargetTopology,
          class GroupT,
          class OverflowFlagT>
__device__ __forceinline__ bool matchSeedWithSubstructureFallbackCooperative(
  const GroupT&                                       group,
  const Seed<maxAtoms, maxBonds>&                     seed,
  const QueryTopology&                                queryTopology,
  const TargetTopology&                               targetTopology,
  const PairMatchTablesDevice&                        tables,
  MatchResult<maxAtoms, maxBonds, maxTA, maxTB>&      match,
  FmcsSubstructureScratch<maxAtoms, maxBonds, maxTA>& scratch,
  int*                                                scratchLock,
  std::uint8_t*                                       partialStorage,
  int                                                 partialCapacity,
  OverflowFlagT*                                      overflowedFlag) {
  const bool fastOk = matchIncrementalFastCooperative(group, seed, queryTopology, targetTopology, tables, match);
  group.sync();
  if (fastOk) {
    return true;
  }
  if (group.thread_rank() == 0) {
    while (atomicCAS(scratchLock, 0, 1) != 0) {
    }
  }
  group.sync();
  const bool ok = matchSeedSubstructureCooperative<true>(group,
                                                         seed,
                                                         queryTopology,
                                                         targetTopology,
                                                         tables,
                                                         match,
                                                         scratch,
                                                         partialStorage,
                                                         partialCapacity,
                                                         overflowedFlag);
  group.sync();
  if (group.thread_rank() == 0) {
    atomicExch(scratchLock, 0);
  }
  group.sync();
  return ok;
}

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_EXPERIMENTAL_FMCS_MATCH_WITH_FALLBACK_CUH
