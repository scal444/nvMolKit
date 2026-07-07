// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include <string>

#include "fmcs_cuda/fmcs_config.cuh"
#include "fmcs_cuda/fmcs_kernel.cuh"
#include "fmcs_cuda/fmcs_launch.cuh"
#include "src/mcs/mcs_compile_flags.h"

namespace mcs {
namespace fmcs {
namespace {

template <typename KernelFunc> void configureSharedMemCarveout(KernelFunc kernel, const char* name) {
  const cudaError_t err =
    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("fMCS CUDA error configuring ") + name +
                             " shared-memory carveout: " + cudaGetErrorString(err));
  }
}

template <int                 blockThreads,
          int                 maxAtoms,
          int                 maxBonds,
          FmcsScratchLocation Scratch,
          bool                CollectTimings,
          bool                CollectStats>
void configureFmcsKernelSharedMem() {
  static const bool configured = []() {
    const std::string name = "fmcsKernel<" + std::to_string(maxAtoms) + "," + std::to_string(maxBonds) + "," +
                             std::to_string(blockThreads) + "," + (CollectTimings ? "timings" : "no-timings") + "," +
                             (CollectStats ? "stats" : "no-stats") + "," +
                             (Scratch == FmcsScratchLocation::Shared ? "shared" : "global") + ">";
    configureSharedMemCarveout(fmcsKernel<maxAtoms, maxBonds, blockThreads, CollectTimings, CollectStats, Scratch>,
                               name.c_str());
    return true;
  }();
  (void)configured;
}

template <int                 blockThreads,
          int                 maxAtoms,
          int                 maxBonds,
          FmcsScratchLocation Scratch,
          bool                CollectTimings,
          bool                CollectStats>
void launchFmcsKernelSpecialization(const DevicePerPairInput*             pairs,
                                    FmcsDeviceResult<maxAtoms, maxBonds>* results,
                                    void*                                 queueStorage,
                                    std::uint8_t*                         substructureStorage,
                                    void*                                 scratchStorage,
                                    unsigned long long*                   elapsedClocks,
                                    ExecutionStats*                       statsOut,
                                    int                                   queueCapacity,
                                    int                                   substructurePartialCapacity,
                                    int                                   numPairs,
                                    unsigned long long                    timeoutClocks,
                                    cudaStream_t                          stream) {
  configureFmcsKernelSharedMem<blockThreads, maxAtoms, maxBonds, Scratch, CollectTimings, CollectStats>();
  dim3 grid(static_cast<unsigned>(numPairs));
  dim3 block(static_cast<unsigned>(blockThreads));
  using QueuedT  = QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>;
  using ScratchT = FmcsSubstructureScratch<maxAtoms, maxBonds, maxAtoms>;
  fmcsKernel<maxAtoms, maxBonds, blockThreads, CollectTimings, CollectStats, Scratch>
    <<<grid, block, 0, stream>>>(pairs,
                                 results,
                                 static_cast<QueuedT*>(queueStorage),
                                 substructureStorage,
                                 static_cast<ScratchT*>(scratchStorage),
                                 elapsedClocks,
                                 statsOut,
                                 queueCapacity,
                                 substructurePartialCapacity,
                                 numPairs,
                                 timeoutClocks);
}

template <int blockThreads, int maxAtoms, int maxBonds, FmcsScratchLocation Scratch>
void launchFmcsKernelSelected(const DevicePerPairInput*             pairs,
                              FmcsDeviceResult<maxAtoms, maxBonds>* results,
                              void*                                 queueStorage,
                              std::uint8_t*                         substructureStorage,
                              void*                                 scratchStorage,
                              unsigned long long*                   elapsedClocks,
                              ExecutionStats*                       timingStatsOut,
                              ExecutionStats*                       statsOut,
                              int                                   queueCapacity,
                              int                                   substructurePartialCapacity,
                              int                                   numPairs,
                              unsigned long long                    timeoutClocks,
                              cudaStream_t                          stream) {
  const bool collectTimings = elapsedClocks != nullptr;
  const bool collectStats   = statsOut != nullptr;

  if (collectTimings && collectStats) {
    if constexpr (nvMolKit::kMCSCollectTimingsEnabled && nvMolKit::kMCSCollectStatsEnabled) {
      launchFmcsKernelSpecialization<blockThreads, maxAtoms, maxBonds, Scratch, true, true>(pairs,
                                                                                            results,
                                                                                            queueStorage,
                                                                                            substructureStorage,
                                                                                            scratchStorage,
                                                                                            elapsedClocks,
                                                                                            statsOut,
                                                                                            queueCapacity,
                                                                                            substructurePartialCapacity,
                                                                                            numPairs,
                                                                                            timeoutClocks,
                                                                                            stream);
      return;
    }
  } else if (collectTimings) {
    if constexpr (nvMolKit::kMCSCollectTimingsEnabled) {
      launchFmcsKernelSpecialization<blockThreads, maxAtoms, maxBonds, Scratch, true, false>(
        pairs,
        results,
        queueStorage,
        substructureStorage,
        scratchStorage,
        elapsedClocks,
        timingStatsOut,
        queueCapacity,
        substructurePartialCapacity,
        numPairs,
        timeoutClocks,
        stream);
      return;
    }
  } else if (collectStats) {
    if constexpr (nvMolKit::kMCSCollectStatsEnabled) {
      launchFmcsKernelSpecialization<blockThreads, maxAtoms, maxBonds, Scratch, false, true>(
        pairs,
        results,
        queueStorage,
        substructureStorage,
        scratchStorage,
        nullptr,
        statsOut,
        queueCapacity,
        substructurePartialCapacity,
        numPairs,
        timeoutClocks,
        stream);
      return;
    }
  } else {
    launchFmcsKernelSpecialization<blockThreads, maxAtoms, maxBonds, Scratch, false, false>(pairs,
                                                                                            results,
                                                                                            queueStorage,
                                                                                            substructureStorage,
                                                                                            scratchStorage,
                                                                                            nullptr,
                                                                                            nullptr,
                                                                                            queueCapacity,
                                                                                            substructurePartialCapacity,
                                                                                            numPairs,
                                                                                            timeoutClocks,
                                                                                            stream);
    return;
  }

  throw std::runtime_error("fMCS timing/stat instrumentation is not instantiated in this build");
}

}  // namespace

int fmcsQueueCapacity() {
  return kFmcsQueueCapacity;
}

int fmcsSubstructurePartialCapacity() {
  return kFmcsSubstructurePartialCapacity;
}

void configureFmcsKernelMaxDynamicSharedMem(const void* kernelFunc, std::size_t bytes) {
  const cudaError_t err =
    cudaFuncSetAttribute(kernelFunc, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("fMCS CUDA error configuring max dynamic shared memory: ") +
                             cudaGetErrorString(err));
  }
}

template <int blockThreads, int maxAtoms, int maxBonds>
std::size_t fmcsKernelStaticSharedBytes(FmcsScratchLocation scratchLocation) {
  cudaFuncAttributes attr{};
  cudaError_t        err = cudaSuccess;
  if (scratchLocation == FmcsScratchLocation::Global) {
    err =
      cudaFuncGetAttributes(&attr,
                            fmcsKernel<maxAtoms, maxBonds, blockThreads, false, false, FmcsScratchLocation::Global>);
  } else {
    if constexpr (blockThreads == 512 && maxAtoms == 128) {
      throw std::invalid_argument("fMCS blockSize 512 at tier-128 has no shared-scratch kernel");
    } else {
      err =
        cudaFuncGetAttributes(&attr,
                              fmcsKernel<maxAtoms, maxBonds, blockThreads, false, false, FmcsScratchLocation::Shared>);
    }
  }
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("fMCS CUDA error querying kernel attributes: ") + cudaGetErrorString(err));
  }
  return attr.sharedSizeBytes;
}

template <int maxAtoms, int maxBonds> std::size_t fmcsQueueStorageBytes(std::size_t numPairs) {
  using QueuedT = QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>;
  return numPairs * static_cast<std::size_t>(kFmcsQueueCapacity) * sizeof(QueuedT);
}

template <int blockThreads, int maxAtoms> std::size_t fmcsSubstructureStorageBytes(std::size_t numPairs) {
  return numPairs * static_cast<std::size_t>(FmcsBlockConfig<blockThreads>::numGroups) * 2u *
         static_cast<std::size_t>(kFmcsSubstructurePartialCapacity) * static_cast<std::size_t>(maxAtoms) *
         sizeof(std::uint8_t);
}

template <int blockThreads, int maxAtoms, int maxBonds> std::size_t fmcsScratchStorageBytes(std::size_t numPairs) {
  return numPairs * static_cast<std::size_t>(FmcsBlockConfig<blockThreads>::numGroups) *
         sizeof(FmcsSubstructureScratch<maxAtoms, maxBonds, maxAtoms>);
}

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
                         cudaStream_t                          stream) {
  if (scratchLocation == FmcsScratchLocation::Global) {
    // Global is instantiated at 128-block only for tier-128 (the A/B tier of
    // interest); smaller 128-block tiers stay shared-only to bound kernel count.
    if constexpr (maxAtoms == 128) {
      launchFmcsKernelSelected<128, maxAtoms, maxBonds, FmcsScratchLocation::Global>(pairs,
                                                                                     results,
                                                                                     queueStorage,
                                                                                     substructureStorage,
                                                                                     scratchStorage,
                                                                                     elapsedClocks,
                                                                                     timingStatsOut,
                                                                                     statsOut,
                                                                                     queueCapacity,
                                                                                     substructurePartialCapacity,
                                                                                     numPairs,
                                                                                     timeoutClocks,
                                                                                     stream);
      return;
    }
    throw std::invalid_argument(
      "fMCS scratchLocation=global is not instantiated for blockSize 128 "
      "below tier-128; use scratchLocation=auto or shared");
  }
  launchFmcsKernelSelected<128, maxAtoms, maxBonds, FmcsScratchLocation::Shared>(pairs,
                                                                                 results,
                                                                                 queueStorage,
                                                                                 substructureStorage,
                                                                                 nullptr,
                                                                                 elapsedClocks,
                                                                                 timingStatsOut,
                                                                                 statsOut,
                                                                                 queueCapacity,
                                                                                 substructurePartialCapacity,
                                                                                 numPairs,
                                                                                 timeoutClocks,
                                                                                 stream);
}

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
                         cudaStream_t                          stream) {
  if (scratchLocation == FmcsScratchLocation::Global) {
    launchFmcsKernelSelected<512, maxAtoms, maxBonds, FmcsScratchLocation::Global>(pairs,
                                                                                   results,
                                                                                   queueStorage,
                                                                                   substructureStorage,
                                                                                   scratchStorage,
                                                                                   elapsedClocks,
                                                                                   timingStatsOut,
                                                                                   statsOut,
                                                                                   queueCapacity,
                                                                                   substructurePartialCapacity,
                                                                                   numPairs,
                                                                                   timeoutClocks,
                                                                                   stream);
    return;
  }
  if constexpr (maxAtoms == 128) {
    // 512 threads x 16 groups of tier-128 scratch (~70 KB) cannot fit the
    // 48 KB static-shared cap, so a Shared specialization does not exist.
    throw std::invalid_argument(
      "fMCS scratchLocation=shared cannot satisfy blockSize 512 at tier-128 "
      "(needs ~70 KB static shared > 48 KB); use scratchLocation=global or auto");
  } else {
    launchFmcsKernelSelected<512, maxAtoms, maxBonds, FmcsScratchLocation::Shared>(pairs,
                                                                                   results,
                                                                                   queueStorage,
                                                                                   substructureStorage,
                                                                                   nullptr,
                                                                                   elapsedClocks,
                                                                                   timingStatsOut,
                                                                                   statsOut,
                                                                                   queueCapacity,
                                                                                   substructurePartialCapacity,
                                                                                   numPairs,
                                                                                   timeoutClocks,
                                                                                   stream);
  }
}

template void launchFmcsKernel128<16, 16>(const DevicePerPairInput*,
                                          FmcsDeviceResult<16, 16>*,
                                          void*,
                                          std::uint8_t*,
                                          void*,
                                          FmcsScratchLocation,
                                          unsigned long long*,
                                          ExecutionStats*,
                                          ExecutionStats*,
                                          int,
                                          int,
                                          int,
                                          unsigned long long,
                                          cudaStream_t);
template void launchFmcsKernel128<32, 32>(const DevicePerPairInput*,
                                          FmcsDeviceResult<32, 32>*,
                                          void*,
                                          std::uint8_t*,
                                          void*,
                                          FmcsScratchLocation,
                                          unsigned long long*,
                                          ExecutionStats*,
                                          ExecutionStats*,
                                          int,
                                          int,
                                          int,
                                          unsigned long long,
                                          cudaStream_t);
template void launchFmcsKernel128<64, 64>(const DevicePerPairInput*,
                                          FmcsDeviceResult<64, 64>*,
                                          void*,
                                          std::uint8_t*,
                                          void*,
                                          FmcsScratchLocation,
                                          unsigned long long*,
                                          ExecutionStats*,
                                          ExecutionStats*,
                                          int,
                                          int,
                                          int,
                                          unsigned long long,
                                          cudaStream_t);
template void launchFmcsKernel128<128, 128>(const DevicePerPairInput*,
                                            FmcsDeviceResult<128, 128>*,
                                            void*,
                                            std::uint8_t*,
                                            void*,
                                            FmcsScratchLocation,
                                            unsigned long long*,
                                            ExecutionStats*,
                                            ExecutionStats*,
                                            int,
                                            int,
                                            int,
                                            unsigned long long,
                                            cudaStream_t);

template void launchFmcsKernel512<16, 16>(const DevicePerPairInput*,
                                          FmcsDeviceResult<16, 16>*,
                                          void*,
                                          std::uint8_t*,
                                          void*,
                                          FmcsScratchLocation,
                                          unsigned long long*,
                                          ExecutionStats*,
                                          ExecutionStats*,
                                          int,
                                          int,
                                          int,
                                          unsigned long long,
                                          cudaStream_t);
template void launchFmcsKernel512<32, 32>(const DevicePerPairInput*,
                                          FmcsDeviceResult<32, 32>*,
                                          void*,
                                          std::uint8_t*,
                                          void*,
                                          FmcsScratchLocation,
                                          unsigned long long*,
                                          ExecutionStats*,
                                          ExecutionStats*,
                                          int,
                                          int,
                                          int,
                                          unsigned long long,
                                          cudaStream_t);
template void launchFmcsKernel512<64, 64>(const DevicePerPairInput*,
                                          FmcsDeviceResult<64, 64>*,
                                          void*,
                                          std::uint8_t*,
                                          void*,
                                          FmcsScratchLocation,
                                          unsigned long long*,
                                          ExecutionStats*,
                                          ExecutionStats*,
                                          int,
                                          int,
                                          int,
                                          unsigned long long,
                                          cudaStream_t);
template void launchFmcsKernel512<128, 128>(const DevicePerPairInput*,
                                            FmcsDeviceResult<128, 128>*,
                                            void*,
                                            std::uint8_t*,
                                            void*,
                                            FmcsScratchLocation,
                                            unsigned long long*,
                                            ExecutionStats*,
                                            ExecutionStats*,
                                            int,
                                            int,
                                            int,
                                            unsigned long long,
                                            cudaStream_t);

template std::size_t fmcsQueueStorageBytes<16, 16>(std::size_t);
template std::size_t fmcsQueueStorageBytes<32, 32>(std::size_t);
template std::size_t fmcsQueueStorageBytes<64, 64>(std::size_t);
template std::size_t fmcsQueueStorageBytes<128, 128>(std::size_t);

template std::size_t fmcsSubstructureStorageBytes<128, 16>(std::size_t);
template std::size_t fmcsSubstructureStorageBytes<128, 32>(std::size_t);
template std::size_t fmcsSubstructureStorageBytes<128, 64>(std::size_t);
template std::size_t fmcsSubstructureStorageBytes<128, 128>(std::size_t);
template std::size_t fmcsSubstructureStorageBytes<512, 16>(std::size_t);
template std::size_t fmcsSubstructureStorageBytes<512, 32>(std::size_t);
template std::size_t fmcsSubstructureStorageBytes<512, 64>(std::size_t);
template std::size_t fmcsSubstructureStorageBytes<512, 128>(std::size_t);

template std::size_t fmcsScratchStorageBytes<128, 16, 16>(std::size_t);
template std::size_t fmcsScratchStorageBytes<128, 32, 32>(std::size_t);
template std::size_t fmcsScratchStorageBytes<128, 64, 64>(std::size_t);
template std::size_t fmcsScratchStorageBytes<128, 128, 128>(std::size_t);
template std::size_t fmcsScratchStorageBytes<512, 16, 16>(std::size_t);
template std::size_t fmcsScratchStorageBytes<512, 32, 32>(std::size_t);
template std::size_t fmcsScratchStorageBytes<512, 64, 64>(std::size_t);
template std::size_t fmcsScratchStorageBytes<512, 128, 128>(std::size_t);

template std::size_t fmcsKernelStaticSharedBytes<128, 128, 128>(FmcsScratchLocation);
template std::size_t fmcsKernelStaticSharedBytes<512, 16, 16>(FmcsScratchLocation);
template std::size_t fmcsKernelStaticSharedBytes<512, 32, 32>(FmcsScratchLocation);
template std::size_t fmcsKernelStaticSharedBytes<512, 64, 64>(FmcsScratchLocation);
template std::size_t fmcsKernelStaticSharedBytes<512, 128, 128>(FmcsScratchLocation);

}  // namespace fmcs
}  // namespace mcs
