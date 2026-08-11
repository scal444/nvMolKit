#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <cuda_runtime.h>
#include <cstdint>

#include "opt/fmcs_main_opt.cuh"

namespace {

using namespace fmcsopt;

template <int T>
void launchTier(const torch::Tensor& pairMeta,
                const uint32_t*      qRowB,
                const uint32_t*      qColB,
                const uint32_t*      qBndB,
                const uint32_t*      qEpB,
                const uint32_t*      tRowB,
                const uint32_t*      tColB,
                const uint32_t*      tBndB,
                const uint32_t*      tEpB,
                const uint32_t*      aTabB,
                const uint32_t*      bTabB,
                void*                metaDevRaw,
                void*                queueRaw,
                int32_t*             stats,
                uint8_t*             atomMapping,
                uint8_t*             bondMapping,
                cudaStream_t         stream) {
  const int numPairs = static_cast<int>(pairMeta.size(0));
  int32_t*  metaDev  = reinterpret_cast<int32_t*>(metaDevRaw);

  cudaError_t err = cudaMemcpyAsync(metaDev,
                                    pairMeta.data_ptr<int32_t>(),
                                    static_cast<size_t>(numPairs) * 16 * sizeof(int32_t),
                                    cudaMemcpyHostToDevice,
                                    stream);
  TORCH_CHECK(err == cudaSuccess, "pair meta copy failed: ", cudaGetErrorString(err));

  constexpr size_t kShared = sizeof(BlockShared<T>);
  static bool      configured = false;
  if (!configured) {
    cudaFuncSetAttribute(reinterpret_cast<const void*>(fmcsOptKernel<T>),
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(kShared));
    configured = true;
  }

  fmcsOptKernel<T><<<numPairs, kBlockThreads, kShared, stream>>>(metaDev,
                                                                 qRowB,
                                                                 qColB,
                                                                 qBndB,
                                                                 qEpB,
                                                                 tRowB,
                                                                 tColB,
                                                                 tBndB,
                                                                 tEpB,
                                                                 aTabB,
                                                                 bTabB,
                                                                 reinterpret_cast<QS<T>*>(queueRaw),
                                                                 stats,
                                                                 atomMapping,
                                                                 bondMapping,
                                                                 numPairs);
  err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "fMCS launch failed: ", cudaGetErrorString(err));
}

}  // namespace

void launch_fmcs_batch(torch::Tensor tierConfig,
                       torch::Tensor pairMeta,
                       torch::Tensor queryRowOffsets,
                       torch::Tensor queryColIndices,
                       torch::Tensor queryBondIndices,
                       torch::Tensor queryBondEndpoints,
                       torch::Tensor targetRowOffsets,
                       torch::Tensor targetColIndices,
                       torch::Tensor targetBondIndices,
                       torch::Tensor targetBondEndpoints,
                       torch::Tensor atomMatchWords,
                       torch::Tensor bondMatchWords,
                       torch::Tensor pairInputStorage,
                       torch::Tensor queueStorage,
                       torch::Tensor scratchStorage,
                       torch::Tensor rawResultStorage,
                       torch::Tensor stats,
                       torch::Tensor atomMapping,
                       torch::Tensor bondMapping) {
  TORCH_CHECK(tierConfig.device().is_cpu() && tierConfig.scalar_type() == torch::kInt32);
  const int tier = tierConfig.data_ptr<int32_t>()[0];
  const c10::cuda::CUDAGuard guard(queryRowOffsets.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  (void)scratchStorage;
  (void)rawResultStorage;

  const auto* qRowB = reinterpret_cast<const uint32_t*>(queryRowOffsets.data_ptr<int32_t>());
  const auto* qColB = reinterpret_cast<const uint32_t*>(queryColIndices.data_ptr<int32_t>());
  const auto* qBndB = reinterpret_cast<const uint32_t*>(queryBondIndices.data_ptr<int32_t>());
  const auto* qEpB  = reinterpret_cast<const uint32_t*>(queryBondEndpoints.data_ptr<int32_t>());
  const auto* tRowB = reinterpret_cast<const uint32_t*>(targetRowOffsets.data_ptr<int32_t>());
  const auto* tColB = reinterpret_cast<const uint32_t*>(targetColIndices.data_ptr<int32_t>());
  const auto* tBndB = reinterpret_cast<const uint32_t*>(targetBondIndices.data_ptr<int32_t>());
  const auto* tEpB  = reinterpret_cast<const uint32_t*>(targetBondEndpoints.data_ptr<int32_t>());
  const auto* aTabB = reinterpret_cast<const uint32_t*>(atomMatchWords.data_ptr<int32_t>());
  const auto* bTabB = reinterpret_cast<const uint32_t*>(bondMatchWords.data_ptr<int32_t>());

  void* metaDev = pairInputStorage.data_ptr<uint8_t>();
  void* queue   = queueStorage.data_ptr<uint8_t>();

  TORCH_CHECK(pairInputStorage.numel() >= pairMeta.numel() * 4, "pair descriptor storage too small");

#define LAUNCH(TT)                                                                                 \
  return launchTier<TT>(pairMeta, qRowB, qColB, qBndB, qEpB, tRowB, tColB, tBndB, tEpB, aTabB,     \
                        bTabB, metaDev, queue, stats.data_ptr<int32_t>(),                          \
                        atomMapping.data_ptr<uint8_t>(), bondMapping.data_ptr<uint8_t>(), stream)
  if (tier == 16)
    LAUNCH(16);
  if (tier == 32)
    LAUNCH(32);
  if (tier == 64)
    LAUNCH(64);
#undef LAUNCH
  TORCH_CHECK(false, "Unsupported fMCS tier: ", tier);
}
