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

#ifndef MCS_COMMON_MCS_GRAPH_DEVICE_CUH
#define MCS_COMMON_MCS_GRAPH_DEVICE_CUH

#include <cstdint>
#include <type_traits>

namespace mcs {

// ---------------------------------------------------------------------------
// Word-type selection: uint32_t for maxSize <= 32, uint64_t otherwise.
// ---------------------------------------------------------------------------

template<int maxSize>
using BitWord = std::conditional_t<(maxSize <= 32), uint32_t, uint64_t>;

/// Per-vertex adjacency stored as bitmasks.
/// adj[v][wi] has bit i set iff vertex (wi*kBitsPerWord + i) is adjacent to v.
template<int maxSize>
struct AdjMatrix {
  using word_type = BitWord<maxSize>;
  static constexpr int kBitsPerWord = sizeof(word_type) * 8;
  static constexpr int kWords = (maxSize + kBitsPerWord - 1) / kBitsPerWord;
  word_type data[maxSize * kWords] = {};

  __host__ __device__ const word_type* operator[](int v) const {
    return data + v * kWords;
  }
  __host__ __device__ word_type* operator[](int v) {
    return data + v * kWords;
  }
};

}  // namespace mcs

#endif  // MCS_COMMON_MCS_GRAPH_DEVICE_CUH
