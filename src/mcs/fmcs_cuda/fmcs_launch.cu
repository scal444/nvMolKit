// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcs_cuda/fmcs_launch.cuh"

#include "fmcs_cuda/fmcs_kernel.cuh"

namespace mcs {
namespace fmcs {
namespace {

template<int blockThreads, int maxAtoms, int maxBonds>
void launchFmcsKernelNoInstrumentation(const DevicePerPairInput* pairs,
                                       FmcsDeviceResult<maxAtoms, maxBonds>* results,
                                       void* queueStorage,
                                       std::uint8_t* substructureStorage,
                                       int queueCapacity,
                                       int substructurePartialCapacity,
                                       int numPairs,
                                       unsigned long long timeoutClocks,
                                       cudaStream_t stream) {
  dim3 grid(static_cast<unsigned>(numPairs));
  dim3 block(static_cast<unsigned>(blockThreads));
  using QueuedT = QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>;
  fmcsKernel<maxAtoms, maxBonds, blockThreads, false, false>
      <<<grid, block, 0, stream>>>(
          pairs,
          results,
          static_cast<QueuedT*>(queueStorage),
          nullptr,
          substructureStorage,
          nullptr,
          nullptr,
          queueCapacity,
          0,
          substructurePartialCapacity,
          numPairs,
          timeoutClocks);
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
                         int queueCapacity,
                         int substructurePartialCapacity,
                         int numPairs,
                         unsigned long long timeoutClocks,
                         cudaStream_t stream) {
  launchFmcsKernelNoInstrumentation<128, maxAtoms, maxBonds>(
      pairs,
      results,
      queueStorage,
      substructureStorage,
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
                         int queueCapacity,
                         int substructurePartialCapacity,
                         int numPairs,
                         unsigned long long timeoutClocks,
                         cudaStream_t stream) {
  launchFmcsKernelNoInstrumentation<512, maxAtoms, maxBonds>(
      pairs,
      results,
      queueStorage,
      substructureStorage,
      queueCapacity,
      substructurePartialCapacity,
      numPairs,
      timeoutClocks,
      stream);
}

template void launchFmcsKernel128<16, 16>(
    const DevicePerPairInput*, FmcsDeviceResult<16, 16>*, void*,
    std::uint8_t*, int, int, int, unsigned long long, cudaStream_t);
template void launchFmcsKernel128<32, 32>(
    const DevicePerPairInput*, FmcsDeviceResult<32, 32>*, void*,
    std::uint8_t*, int, int, int, unsigned long long, cudaStream_t);
template void launchFmcsKernel128<64, 64>(
    const DevicePerPairInput*, FmcsDeviceResult<64, 64>*, void*,
    std::uint8_t*, int, int, int, unsigned long long, cudaStream_t);
template void launchFmcsKernel128<128, 128>(
    const DevicePerPairInput*, FmcsDeviceResult<128, 128>*, void*,
    std::uint8_t*, int, int, int, unsigned long long, cudaStream_t);

template void launchFmcsKernel512<16, 16>(
    const DevicePerPairInput*, FmcsDeviceResult<16, 16>*, void*,
    std::uint8_t*, int, int, int, unsigned long long, cudaStream_t);
template void launchFmcsKernel512<32, 32>(
    const DevicePerPairInput*, FmcsDeviceResult<32, 32>*, void*,
    std::uint8_t*, int, int, int, unsigned long long, cudaStream_t);
template void launchFmcsKernel512<64, 64>(
    const DevicePerPairInput*, FmcsDeviceResult<64, 64>*, void*,
    std::uint8_t*, int, int, int, unsigned long long, cudaStream_t);

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
