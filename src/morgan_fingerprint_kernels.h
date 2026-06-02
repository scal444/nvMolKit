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

#ifndef NVMOLKIT_MORGAN_FINGERPRINT_KERNELS_H
#define NVMOLKIT_MORGAN_FINGERPRINT_KERNELS_H

#include "src/data_structures/flat_bit_vect.h"
#include "src/morgan_fingerprint_common.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

//! All GPU buffers for Morgan fingerprint computation
struct MorganGPUBuffersBatch {
  AsyncDeviceVector<std::uint32_t> atomInvariants;        // Size = nMolecules * maxAtoms
  AsyncDeviceVector<std::uint32_t> bondInvariants;        // Size = nMolecules *  maxAtoms
  AsyncDeviceVector<std::int16_t>  bondIndices;           // Size = nMolecules *  maxAtoms * maxNumBonds
  AsyncDeviceVector<std::int16_t>  bondOtherAtomIndices;  // Size =nMolecules *  maxAtoms * maxNumBonds
  AsyncDeviceVector<std::int16_t>  nAtomsPerMol;          // Size = nMolecules

  AsyncDeviceVector<int> outputIndices;  // Size = nMolecules

  AsyncDeviceVector<FlatBitVect<32>>  allSeenNeighborhoods32;   // Size = nMolecules * 32 * (maxRadius + 1)
  AsyncDeviceVector<FlatBitVect<64>>  allSeenNeighborhoods64;   // Size = nMolecules * 32 * (maxRadius + 1)
  AsyncDeviceVector<FlatBitVect<128>> allSeenNeighborhoods128;  // Size = nMolecules * 32 * (maxRadius + 1)
};

template <int fpSize>
void launchMorganFingerprintKernelBatch(const MorganGPUBuffersBatch&            buffers,
                                        AsyncDeviceVector<FlatBitVect<fpSize>>& outputAccumulator,
                                        size_t                                  maxRadius,
                                        int                                     maxAtoms,
                                        int                                     nMolecules = 0,
                                        cudaStream_t                            stream     = nullptr);

}  // namespace nvMolKit

#define DEFINE_EXTERN_TEMPLATE(fpSize)                                       \
  extern template void nvMolKit::launchMorganFingerprintKernelBatch<fpSize>( \
    const nvMolKit::MorganGPUBuffersBatch&  buffers,                         \
    AsyncDeviceVector<FlatBitVect<fpSize>>& outputAccumulator,               \
    size_t                                  maxRadius,                       \
    int                                     maxAtoms,                        \
    int                                     nMolecules,                      \
    cudaStream_t                            stream);
DEFINE_EXTERN_TEMPLATE(128)
DEFINE_EXTERN_TEMPLATE(256)
DEFINE_EXTERN_TEMPLATE(512)
DEFINE_EXTERN_TEMPLATE(1024)
DEFINE_EXTERN_TEMPLATE(2048)
DEFINE_EXTERN_TEMPLATE(4096)

#endif  // NVMOLKIT_MORGAN_FINGERPRINT_KERNELS_H
