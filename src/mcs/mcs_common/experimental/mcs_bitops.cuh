// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// EXPERIMENTAL: generic host/device bit-mask helpers retained for possible
// future MCS algorithms. No production MCS source currently includes them.

#ifndef MCS_COMMON_EXPERIMENTAL_MCS_BITOPS_CUH
#define MCS_COMMON_EXPERIMENTAL_MCS_BITOPS_CUH

#include <cstdint>

namespace mcs {
namespace detail {

__host__ __device__ inline int devicePopc(uint32_t val) {
#ifdef __CUDA_ARCH__
  return __popc(val);
#else
  return __builtin_popcount(val);
#endif
}

__host__ __device__ inline int devicePopc(uint64_t val) {
#ifdef __CUDA_ARCH__
  return __popcll(val);
#else
  return __builtin_popcountll(val);
#endif
}

template<int kWords, typename WordT>
__host__ __device__ int popcount(const WordT* mask) {
  int n = 0;
  for (int w = 0; w < kWords; ++w) {
    n += devicePopc(mask[w]);
  }
  return n;
}

template<int kWords, typename WordT>
__host__ __device__ bool anySet(const WordT* mask) {
  for (int w = 0; w < kWords; ++w) {
    if (mask[w]) return true;
  }
  return false;
}

/// Count trailing zeros (undefined if val == 0).
__host__ __device__ inline int ctz(uint32_t val) {
#ifdef __CUDA_ARCH__
  return __ffs(val) - 1;
#else
  return __builtin_ctz(val);
#endif
}

__host__ __device__ inline int ctz(uint64_t val) {
#ifdef __CUDA_ARCH__
  return __ffsll(val) - 1;
#else
  return __builtin_ctzll(val);
#endif
}

/// Return the index of the lowest set bit across kWords, or -1 if empty.
template<int kWords, typename WordT>
__host__ __device__ int lowestBit(const WordT* mask) {
  constexpr int kBPW = sizeof(WordT) * 8;
  for (int w = 0; w < kWords; ++w) {
    if (mask[w]) {
      return w * kBPW + ctz(mask[w]);
    }
  }
  return -1;
}

/// Return the index of the nth set bit across kWords, or -1 if out of range.
template<int kWords, typename WordT>
__host__ __device__ int nthSetBit(const WordT* mask, int n) {
  constexpr int kBPW = sizeof(WordT) * 8;
  for (int w = 0; w < kWords; ++w) {
    WordT bits = mask[w];
    while (bits) {
      if (n == 0) {
        return w * kBPW + ctz(bits);
      }
      bits &= bits - 1;
      --n;
    }
  }
  return -1;
}

}  // namespace detail
}  // namespace mcs

#endif  // MCS_COMMON_EXPERIMENTAL_MCS_BITOPS_CUH
