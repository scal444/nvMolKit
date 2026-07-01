// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FMCS_CUDA_FMCS_CONFIG_CUH
#define FMCS_CUDA_FMCS_CONFIG_CUH

namespace mcs {
namespace fmcs {

/// Per-block seed worklist capacity. The backing slab lives in global
/// memory; only the cursor/header lives in shared memory.
constexpr int kFmcsQueueCapacity = 4096;

/// Per-block substructure fallback capacity, expressed as max-sized partial
/// entries per ping-pong half. Runtime capacity is larger for smaller seeds.
constexpr int kFmcsSubstructurePartialCapacity = 4096;

/// Supported block and cooperative-group sizing.
constexpr int kFmcsDefaultBlockSize = 128;
constexpr int kFmcsGroupSize        = 32;
static_assert(kFmcsGroupSize <= 32, "kFmcsGroupSize must be <= 32 (warp shuffle / ballot scope)");
static_assert((kFmcsGroupSize & (kFmcsGroupSize - 1)) == 0, "kFmcsGroupSize must be a power of two");

template <int blockThreads> struct FmcsBlockConfig {
  static_assert(blockThreads == 128 || blockThreads == 512, "fMCS block size must be 128 or 512");
  static_assert(blockThreads % kFmcsGroupSize == 0, "fMCS block size must be a multiple of kFmcsGroupSize");
  static constexpr int numGroups = blockThreads / kFmcsGroupSize;
};

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_CONFIG_CUH
