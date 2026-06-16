// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FMCS_CUDA_FMCS_LAUNCH_CUH
#define FMCS_CUDA_FMCS_LAUNCH_CUH

#include "fmcs_cuda/fmcs_kernel_types.cuh"
#include "fmcs_cuda/fmcs_stats.cuh"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace mcs {
namespace fmcs {

template<int maxAtoms, int maxBonds>
using FmcsDeviceResult = DeviceMCSResult<maxAtoms, maxBonds>;

int fmcsQueueCapacity();

int fmcsSubstructurePartialCapacity();

template<int maxAtoms, int maxBonds>
std::size_t fmcsQueueStorageBytes(std::size_t numPairs);

template<int blockThreads, int maxAtoms>
std::size_t fmcsSubstructureStorageBytes(std::size_t numPairs);

template<int maxAtoms, int maxBonds>
void launchFmcsKernel128(const DevicePerPairInput* pairs,
                         FmcsDeviceResult<maxAtoms, maxBonds>* results,
                         void* queueStorage,
                         std::uint8_t* substructureStorage,
                         unsigned long long* elapsedClocks,
                         ExecutionStats* statsOut,
                         int queueCapacity,
                         int substructurePartialCapacity,
                         int numPairs,
                         unsigned long long timeoutClocks,
                         cudaStream_t stream);

template<int maxAtoms, int maxBonds>
void launchFmcsKernel512(const DevicePerPairInput* pairs,
                         FmcsDeviceResult<maxAtoms, maxBonds>* results,
                         void* queueStorage,
                         std::uint8_t* substructureStorage,
                         unsigned long long* elapsedClocks,
                         ExecutionStats* statsOut,
                         int queueCapacity,
                         int substructurePartialCapacity,
                         int numPairs,
                         unsigned long long timeoutClocks,
                         cudaStream_t stream);

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_LAUNCH_CUH
