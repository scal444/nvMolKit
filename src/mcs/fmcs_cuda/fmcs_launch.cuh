// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FMCS_CUDA_FMCS_LAUNCH_CUH
#define FMCS_CUDA_FMCS_LAUNCH_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "fmcs_cuda/fmcs_config.cuh"
#include "fmcs_cuda/fmcs_kernel_types.cuh"
#include "fmcs_cuda/fmcs_stats.cuh"

namespace mcs {
namespace fmcs {

template <int maxAtoms, int maxBonds> using FmcsDeviceResult = DeviceMCSResult<maxAtoms, maxBonds>;

int fmcsQueueCapacity();

int fmcsSubstructurePartialCapacity();

template <int maxAtoms, int maxBonds> std::size_t fmcsQueueStorageBytes(std::size_t numPairs);

template <int blockThreads, int maxAtoms> std::size_t fmcsSubstructureStorageBytes(std::size_t numPairs);

/// Bytes for the global substructure-scratch slab used when
/// scratchLocation == Global: one FmcsSubstructureScratch per (pair, group).
template <int blockThreads, int maxAtoms, int maxBonds> std::size_t fmcsScratchStorageBytes(std::size_t numPairs);

/// Exact static-shared bytes of the selected (no-timings, no-stats) kernel,
/// via cudaFuncGetAttributes.  Used to validate the static budget and, in the
/// dynamic-shared follow-on, to size the carveout request.
template <int blockThreads, int maxAtoms, int maxBonds>
std::size_t fmcsKernelStaticSharedBytes(FmcsScratchLocation scratchLocation);

/// Dormant readiness hook (analysis/fmcs_scratch_placement_plan.md section 7):
/// opts a kernel into extended dynamic shared memory via
/// cudaFuncAttributeMaxDynamicSharedMemorySize.  All fMCS shared memory is
/// currently static, so nothing calls this yet; it becomes live only when a
/// dynamic-shared (extern __shared__) layout lands.
void configureFmcsKernelMaxDynamicSharedMem(const void* kernelFunc, std::size_t bytes);

/// @p scratchStorage / @p scratchLocation select substructure-scratch
/// placement.  Pass nullptr / Shared for the historical static-shared layout.
/// Global at blockSize 128 is instantiated only for tier-128.
template <int maxAtoms, int maxBonds>
void launchFmcsKernel128(const DevicePerPairInput*             pairs,
                         FmcsDeviceResult<maxAtoms, maxBonds>* results,
                         void*                                 queueStorage,
                         std::uint8_t*                         substructureStorage,
                         void*                                 scratchStorage,
                         FmcsScratchLocation                   scratchLocation,
                         unsigned long long*                   elapsedClocks,
                         ExecutionStats*                       timingStatsOut,
                         ExecutionStats*                       statsOut,
                         int                                   queueCapacity,
                         int                                   substructurePartialCapacity,
                         int                                   numPairs,
                         unsigned long long                    timeoutClocks,
                         cudaStream_t                          stream);

/// Tier-128 requires scratchLocation == Global (the shared layout exceeds the
/// 48 KB static cap and is not instantiated); explicit Shared throws.
template <int maxAtoms, int maxBonds>
void launchFmcsKernel512(const DevicePerPairInput*             pairs,
                         FmcsDeviceResult<maxAtoms, maxBonds>* results,
                         void*                                 queueStorage,
                         std::uint8_t*                         substructureStorage,
                         void*                                 scratchStorage,
                         FmcsScratchLocation                   scratchLocation,
                         unsigned long long*                   elapsedClocks,
                         ExecutionStats*                       timingStatsOut,
                         ExecutionStats*                       statsOut,
                         int                                   queueCapacity,
                         int                                   substructurePartialCapacity,
                         int                                   numPairs,
                         unsigned long long                    timeoutClocks,
                         cudaStream_t                          stream);

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_LAUNCH_CUH
