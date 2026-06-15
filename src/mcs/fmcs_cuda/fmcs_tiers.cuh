// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FMCS_CUDA_FMCS_TIERS_CUH
#define FMCS_CUDA_FMCS_TIERS_CUH

namespace mcs {
namespace fmcs {

enum class MaxSizeTier {
  k16,
  k32,
  k64,
  k128,
};

/// Smallest tier whose bitset width covers both counts. Returns -1 when
/// the largest tier (128) does not fit; the caller must flag overflow.
inline int pickMaxSizeTier(int numAtoms, int numBonds) {
  const int need = numAtoms > numBonds ? numAtoms : numBonds;
  if (need <= 16) return 0;
  if (need <= 32) return 1;
  if (need <= 64) return 2;
  if (need <= 128) return 3;
  return -1;
}

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_TIERS_CUH
