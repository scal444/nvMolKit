// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef MCS_COMMON_EXPERIMENTAL_MCS_COOPERATIVE_COPY_CUH
#define MCS_COMMON_EXPERIMENTAL_MCS_COOPERATIVE_COPY_CUH

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace mcs {

namespace cg = cooperative_groups;

/// EXPERIMENTAL: Warp-cooperative async copy (global -> shared only). Waits
/// immediately; callers needing overlap should use the cooperative-groups
/// operations directly.
template <typename WarpT>
__forceinline__ __device__ void warpCopyAsync(const WarpT& warp,
                                              void* __restrict__ dstShared,
                                              const void* __restrict__ srcGlobal,
                                              int nbytes) {
  cg::memcpy_async(warp, dstShared, srcGlobal, nbytes);
  cg::wait(warp);
}

/// EXPERIMENTAL: Block-cooperative coalesced copy via int4 (16 B) loads/stores.
/// The caller must synchronize the block before other warps read @p dst.
template <typename BlockT>
__forceinline__ __device__ void blockCopy(const BlockT& block,
                                          void* __restrict__ dst,
                                          const void* __restrict__ src,
                                          int nbytes) {
  const int threadRank = static_cast<int>(block.thread_rank());
  const int numThreads = static_cast<int>(block.num_threads());

  auto*       dst16         = static_cast<int4*>(dst);
  const auto* src16         = static_cast<const int4*>(src);
  const int   numWideChunks = nbytes / 16;
  for (int i = threadRank; i < numWideChunks; i += numThreads) {
    dst16[i] = src16[i];
  }

  const int tailBytes = nbytes - numWideChunks * 16;
  if (tailBytes > 0) {
    auto*       dstTail       = reinterpret_cast<int*>(dst16 + numWideChunks);
    const auto* srcTail       = reinterpret_cast<const int*>(src16 + numWideChunks);
    const int   numTailChunks = tailBytes / 4;
    for (int i = threadRank; i < numTailChunks; i += numThreads) {
      dstTail[i] = srcTail[i];
    }
  }
}

}  // namespace mcs

#endif  // MCS_COMMON_EXPERIMENTAL_MCS_COOPERATIVE_COPY_CUH
