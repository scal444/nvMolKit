// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_FINGERPRINT_SIMILARITY_DEVICE_CUH
#define NVMOLKIT_FINGERPRINT_SIMILARITY_DEVICE_CUH

#include <cuda_runtime.h>

#include <cmath>

#include "src/fingerprint_similarity.h"
#include "src/utils/cub_helpers.cuh"

namespace nvMolKit {

namespace detail {

template <FingerprintSimilarityMetric Metric, typename Real>
__device__ __forceinline__ Real fingerprintSimilarityFromDenominator(const int intersection, const Real denominator) {
  return denominator > Real{0} ? static_cast<Real>(intersection) / denominator :
                                 (Metric == FingerprintSimilarityMetric::Tanimoto ? Real{1} : Real{0});
}

template <FingerprintSimilarityMetric Metric, typename Real>
__device__ __forceinline__ Real fingerprintSimilarityDenominator(const int intersection,
                                                                 const int lhsBitCount,
                                                                 const int rhsBitCount) {
  if constexpr (Metric == FingerprintSimilarityMetric::Tanimoto) {
    return static_cast<Real>(lhsBitCount + rhsBitCount - intersection);
  } else if constexpr (Metric == FingerprintSimilarityMetric::Cosine) {
    const Real product = static_cast<Real>(lhsBitCount) * static_cast<Real>(rhsBitCount);
    return sqrt(product);
  } else {
    static_assert(Metric == FingerprintSimilarityMetric::Tanimoto || Metric == FingerprintSimilarityMetric::Cosine,
                  "Unsupported fingerprint similarity metric");
  }
}

template <FingerprintSimilarityMetric Metric, typename Real>
__device__ __forceinline__ Real fingerprintSimilarity(const int intersection,
                                                      const int lhsBitCount,
                                                      const int rhsBitCount) {
  return fingerprintSimilarityFromDenominator<Metric>(
    intersection,
    fingerprintSimilarityDenominator<Metric, Real>(intersection, lhsBitCount, rhsBitCount));
}

template <FingerprintSimilarityMetric Metric>
__device__ __forceinline__ bool fingerprintSimilarityCanReach(const int   lhsBitCount,
                                                              const int   rhsBitCount,
                                                              const float threshold) {
  const int minBitCount = cubMin{}(lhsBitCount, rhsBitCount);
  const int maxBitCount = cubMax{}(lhsBitCount, rhsBitCount);
  if constexpr (Metric == FingerprintSimilarityMetric::Tanimoto) {
    // The intersection cannot exceed the smaller population and the union cannot be smaller than the larger one.
    return static_cast<float>(minBitCount) >= threshold * static_cast<float>(maxBitCount);
  } else if constexpr (Metric == FingerprintSimilarityMetric::Cosine) {
    // The maximum cosine similarity is sqrt(minBitCount / maxBitCount).
    return static_cast<float>(minBitCount) >= threshold * threshold * static_cast<float>(maxBitCount);
  } else {
    static_assert(Metric == FingerprintSimilarityMetric::Tanimoto || Metric == FingerprintSimilarityMetric::Cosine,
                  "Unsupported fingerprint similarity metric");
  }
}

//! Distance 1 - similarity in correctly rounded single precision, unaffected by --use_fast_math, so fused pickers
//! agree exactly with a float32 distance matrix computed the same way.
template <FingerprintSimilarityMetric Metric>
__device__ __forceinline__ float fingerprintDistance(const int intersection,
                                                     const int lhsBitCount,
                                                     const int rhsBitCount) {
  float similarity = 0.0F;
  if constexpr (Metric == FingerprintSimilarityMetric::Tanimoto) {
    const int unionCount = lhsBitCount + rhsBitCount - intersection;
    similarity           = unionCount > 0 ? __fdiv_rn(__int2float_rn(intersection), __int2float_rn(unionCount)) : 1.0F;
  } else {
    const float norm = __fsqrt_rn(__fmul_rn(__int2float_rn(lhsBitCount), __int2float_rn(rhsBitCount)));
    similarity       = norm > 0.0F ? __fdiv_rn(__int2float_rn(intersection), norm) : 0.0F;
  }
  return 1.0F - similarity;
}

template <FingerprintSimilarityMetric Metric>
__device__ __forceinline__ bool fingerprintSimilarityAtLeast(const int   intersection,
                                                             const int   lhsBitCount,
                                                             const int   rhsBitCount,
                                                             const float threshold) {
  const float denominator = fingerprintSimilarityDenominator<Metric, float>(intersection, lhsBitCount, rhsBitCount);
  return denominator > 0.0F ? static_cast<float>(intersection) >= threshold * denominator :
                              fingerprintSimilarityFromDenominator<Metric>(intersection, denominator) >= threshold;
}

}  // namespace detail
}  // namespace nvMolKit

#endif  // NVMOLKIT_FINGERPRINT_SIMILARITY_DEVICE_CUH
