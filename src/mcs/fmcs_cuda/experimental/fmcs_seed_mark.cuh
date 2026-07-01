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
//
// EXPERIMENTAL: last-added-atom marking helper reserved for grow-step
// experiments.  The active RDKit-shaped kernel derives the last-added
// frontier solely from seedAddAtomWithinThread, so this helper is not on
// the production path.

#ifndef FMCS_CUDA_EXPERIMENTAL_FMCS_SEED_MARK_CUH
#define FMCS_CUDA_EXPERIMENTAL_FMCS_SEED_MARK_CUH

#include "fmcs_cuda/fmcs_seed.cuh"

namespace mcs {
namespace fmcs {

/// Within-thread: marks an existing seed atom as part of the next boundary
/// without changing @c seed.atoms or @c numAtoms.  This is kept for targeted
/// experiments, but the RDKit-shaped kernel does not use it: RDKit's
/// LastAddedAtomsBeginIdx only scans atoms newly appended by the previous
/// grow step.
template<int maxAtoms, int maxBonds>
__device__ __forceinline__ void seedMarkLastAddedAtomWithinThread(
    Seed<maxAtoms, maxBonds>& seed,
    const int atomIdx) {
  using AtomWord = typename Seed<maxAtoms, maxBonds>::atom_word_type;
  constexpr int kBitsPerWord = Seed<maxAtoms, maxBonds>::kAtomBitsPerWord;
  const int      wordIdx = atomIdx / kBitsPerWord;
  const AtomWord mask    = static_cast<AtomWord>(1) << (atomIdx % kBitsPerWord);
  seed.lastAddedAtoms[wordIdx] |= mask;
}

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_EXPERIMENTAL_FMCS_SEED_MARK_CUH
