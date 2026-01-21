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

#ifndef NVMOLKIT_SUBSTRUCT_LAUNCH_CONFIG_H
#define NVMOLKIT_SUBSTRUCT_LAUNCH_CONFIG_H

#include <cstddef>
#include <cstdint>

namespace nvMolKit {

enum class SubstructTemplateConfig : uint8_t;
struct TemplateConfigProperties;

/// Maximum scratch space for boolean expression evaluation per query atom.
/// Complex SMARTS patterns with many OR branches can require significant scratch space.
/// E.g., [C,N,O,S,F,Cl,Br,I,...] with N alternatives needs 2N-1 slots (N leaves + N-1 ORs).
/// 256 supports up to ~128 OR alternatives per atom.
constexpr int kMaxBoolScratchSize = 256;

constexpr std::size_t kMaxTargetAtoms = 128;
constexpr std::size_t kMaxQueryAtoms  = 64;

/// Total bits in a label matrix (target × query)
constexpr std::size_t kLabelMatrixBits = kMaxTargetAtoms * kMaxQueryAtoms;

/// Number of 32-bit words per label matrix
constexpr std::size_t kLabelMatrixWords = kLabelMatrixBits / 32;

/// Block size varies by MaxTargetAtoms to fit shared memory budget
template <std::size_t MaxTargetAtoms>
#ifdef __CUDACC__
__host__ __device__
#endif
constexpr int getBlockSizeForConfig() {
  if constexpr (MaxTargetAtoms >= 128) {
    return 256;
  }
  return 128;
}

TemplateConfigProperties getTemplateConfigProperties(SubstructTemplateConfig config);
std::size_t computeLabelMatrixWords(int maxTargetAtoms, int maxQueryAtoms);
SubstructTemplateConfig selectTemplateConfig(int maxTargetAtoms, int maxQueryAtoms, int maxBondsPerAtom);

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_LAUNCH_CONFIG_H
