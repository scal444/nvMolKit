// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcs_cuda/fmcs_launch.cuh"

#include "fmcs_cuda/fmcs_config.cuh"
#include "fmcs_cuda/fmcs_kernel.cuh"
#include "src/mcs/mcs_compile_flags.h"

#include <stdexcept>
#include <string>

namespace mcs {
namespace fmcs {
namespace {

template <typename KernelFunc>
void configureSharedMemCarveout(KernelFunc kernel, const char* name) {
  const cudaError_t err =
      cudaFuncSetAttribute(kernel,
                           cudaFuncAttributePreferredSharedMemoryCarveout,
                           cudaSharedmemCarveoutMaxShared);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("fMCS CUDA error configuring ") +
                             name + " shared-memory carveout: " +
                             cudaGetErrorString(err));
  }
}

template<int blockThreads, int maxAtoms, int maxBonds, bool CollectTimings, bool CollectStats>
void configureFmcsKernelSharedMem() {
  static const bool configured = []() {
    const std::string name =
        "fmcsKernel<" + std::to_string(maxAtoms) + "," +
        std::to_string(maxBonds) + "," +
        std::to_string(blockThreads) + "," +
        (CollectTimings ? "timings" : "no-timings") + "," +
        (CollectStats ? "stats" : "no-stats") + ">";
    configureSharedMemCarveout(
        fmcsKernel<maxAtoms, maxBonds, blockThreads, CollectTimings, CollectStats>,
        name.c_str());
    return true;
  }();
  (void)configured;
}

template<int blockThreads, int maxAtoms, int maxBonds, bool CollectTimings, bool CollectStats>
void launchFmcsKernelSpecialization(const DevicePerPairInput* pairs,
                                    FmcsDeviceResult<maxAtoms, maxBonds>* results,
                                    void* queueStorage,
                                    std::uint8_t* substructureStorage,
                                    unsigned long long* elapsedClocks,
                                    ExecutionStats* statsOut,
                                    int queueCapacity,
                                    int substructurePartialCapacity,
                                    int numPairs,
                                    unsigned long long timeoutClocks,
                                    cudaStream_t stream) {
  configureFmcsKernelSharedMem<blockThreads, maxAtoms, maxBonds, CollectTimings, CollectStats>();
  dim3 grid(static_cast<unsigned>(numPairs));
  dim3 block(static_cast<unsigned>(blockThreads));
  using QueuedT = QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>;
  fmcsKernel<maxAtoms, maxBonds, blockThreads, CollectTimings, CollectStats>
      <<<grid, block, 0, stream>>>(
          pairs,
          results,
          static_cast<QueuedT*>(queueStorage),
          substructureStorage,
          elapsedClocks,
          statsOut,
          queueCapacity,
          substructurePartialCapacity,
          numPairs,
          timeoutClocks);
}

template<int blockThreads, int maxAtoms, int maxBonds>
void launchFmcsKernelSelected(const DevicePerPairInput* pairs,
                              FmcsDeviceResult<maxAtoms, maxBonds>* results,
                              void* queueStorage,
                              std::uint8_t* substructureStorage,
                              unsigned long long* elapsedClocks,
                              ExecutionStats* timingStatsOut,
                              ExecutionStats* statsOut,
                              int queueCapacity,
                              int substructurePartialCapacity,
                              int numPairs,
                              unsigned long long timeoutClocks,
                              cudaStream_t stream) {
  const bool collectTimings = elapsedClocks != nullptr;
  const bool collectStats = statsOut != nullptr;

  if (collectTimings && collectStats) {
    if constexpr (nvMolKit::kMCSCollectTimingsEnabled &&
                  nvMolKit::kMCSCollectStatsEnabled) {
      launchFmcsKernelSpecialization<blockThreads, maxAtoms, maxBonds, true, true>(
          pairs, results, queueStorage, substructureStorage, elapsedClocks,
          statsOut, queueCapacity, substructurePartialCapacity, numPairs,
          timeoutClocks, stream);
      return;
    }
  } else if (collectTimings) {
    if constexpr (nvMolKit::kMCSCollectTimingsEnabled) {
      launchFmcsKernelSpecialization<blockThreads, maxAtoms, maxBonds, true, false>(
          pairs, results, queueStorage, substructureStorage, elapsedClocks,
          timingStatsOut, queueCapacity, substructurePartialCapacity, numPairs,
          timeoutClocks, stream);
      return;
    }
  } else if (collectStats) {
    if constexpr (nvMolKit::kMCSCollectStatsEnabled) {
      launchFmcsKernelSpecialization<blockThreads, maxAtoms, maxBonds, false, true>(
          pairs, results, queueStorage, substructureStorage, nullptr,
          statsOut, queueCapacity, substructurePartialCapacity, numPairs,
          timeoutClocks, stream);
      return;
    }
  } else {
    launchFmcsKernelSpecialization<blockThreads, maxAtoms, maxBonds, false, false>(
        pairs, results, queueStorage, substructureStorage, nullptr, nullptr,
        queueCapacity, substructurePartialCapacity, numPairs, timeoutClocks,
        stream);
    return;
  }

  throw std::runtime_error(
      "fMCS timing/stat instrumentation is not instantiated in this build");
}

}  // namespace

int fmcsQueueCapacity() {
  return kFmcsQueueCapacity;
}

int fmcsSubstructurePartialCapacity() {
  return kFmcsSubstructurePartialCapacity;
}

template<int maxAtoms, int maxBonds>
std::size_t fmcsQueueStorageBytes(std::size_t numPairs) {
  using QueuedT = QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>;
  return numPairs * static_cast<std::size_t>(kFmcsQueueCapacity) *
         sizeof(QueuedT);
}

template<int blockThreads, int maxAtoms>
std::size_t fmcsSubstructureStorageBytes(std::size_t numPairs) {
  return numPairs *
         static_cast<std::size_t>(FmcsBlockConfig<blockThreads>::numGroups) *
         2u *
         static_cast<std::size_t>(kFmcsSubstructurePartialCapacity) *
         static_cast<std::size_t>(maxAtoms) *
         sizeof(std::uint8_t);
}

template<int maxAtoms, int maxBonds>
void launchFmcsKernel128(const DevicePerPairInput* pairs,
                         FmcsDeviceResult<maxAtoms, maxBonds>* results,
                         void* queueStorage,
                         std::uint8_t* substructureStorage,
                         unsigned long long* elapsedClocks,
                         ExecutionStats* timingStatsOut,
                         ExecutionStats* statsOut,
                         int queueCapacity,
                         int substructurePartialCapacity,
                         int numPairs,
                         unsigned long long timeoutClocks,
                         cudaStream_t stream) {
  launchFmcsKernelSelected<128, maxAtoms, maxBonds>(
      pairs,
      results,
      queueStorage,
      substructureStorage,
      elapsedClocks,
      timingStatsOut,
      statsOut,
      queueCapacity,
      substructurePartialCapacity,
      numPairs,
      timeoutClocks,
      stream);
}

template<int maxAtoms, int maxBonds>
void launchFmcsKernel512(const DevicePerPairInput* pairs,
                         FmcsDeviceResult<maxAtoms, maxBonds>* results,
                         void* queueStorage,
                         std::uint8_t* substructureStorage,
                         unsigned long long* elapsedClocks,
                         ExecutionStats* timingStatsOut,
                         ExecutionStats* statsOut,
                         int queueCapacity,
                         int substructurePartialCapacity,
                         int numPairs,
                         unsigned long long timeoutClocks,
                         cudaStream_t stream) {
  launchFmcsKernelSelected<512, maxAtoms, maxBonds>(
      pairs,
      results,
      queueStorage,
      substructureStorage,
      elapsedClocks,
      timingStatsOut,
      statsOut,
      queueCapacity,
      substructurePartialCapacity,
      numPairs,
      timeoutClocks,
      stream);
}

template void launchFmcsKernel128<16, 16>(
    const DevicePerPairInput*, FmcsDeviceResult<16, 16>*, void*,
    std::uint8_t*, unsigned long long*, ExecutionStats*, ExecutionStats*, int,
    int, int, unsigned long long, cudaStream_t);
template void launchFmcsKernel128<32, 32>(
    const DevicePerPairInput*, FmcsDeviceResult<32, 32>*, void*,
    std::uint8_t*, unsigned long long*, ExecutionStats*, ExecutionStats*, int,
    int, int, unsigned long long, cudaStream_t);
template void launchFmcsKernel128<64, 64>(
    const DevicePerPairInput*, FmcsDeviceResult<64, 64>*, void*,
    std::uint8_t*, unsigned long long*, ExecutionStats*, ExecutionStats*, int,
    int, int, unsigned long long, cudaStream_t);
template void launchFmcsKernel128<128, 128>(
    const DevicePerPairInput*, FmcsDeviceResult<128, 128>*, void*,
    std::uint8_t*, unsigned long long*, ExecutionStats*, ExecutionStats*, int,
    int, int, unsigned long long, cudaStream_t);

template void launchFmcsKernel512<16, 16>(
    const DevicePerPairInput*, FmcsDeviceResult<16, 16>*, void*,
    std::uint8_t*, unsigned long long*, ExecutionStats*, ExecutionStats*, int,
    int, int, unsigned long long, cudaStream_t);
template void launchFmcsKernel512<32, 32>(
    const DevicePerPairInput*, FmcsDeviceResult<32, 32>*, void*,
    std::uint8_t*, unsigned long long*, ExecutionStats*, ExecutionStats*, int,
    int, int, unsigned long long, cudaStream_t);
template void launchFmcsKernel512<64, 64>(
    const DevicePerPairInput*, FmcsDeviceResult<64, 64>*, void*,
    std::uint8_t*, unsigned long long*, ExecutionStats*, ExecutionStats*, int,
    int, int, unsigned long long, cudaStream_t);

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

}  // namespace fmcs
}  // namespace mcs
