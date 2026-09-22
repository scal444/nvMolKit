// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DIVERSITY_PICKERS_H
#define NVMOLKIT_DIVERSITY_PICKERS_H

#include <cuda_runtime.h>

#include <cstdint>
#include <cuda/std/span>
#include <vector>

#include "src/clustering_result.h"
#include "src/fingerprint_similarity.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

/** Device-resident picker indices and MaxMin's final selected separation. */
struct PickerResult {
  AsyncDeviceVector<int> indices;
  int                    count        = 0;
  double                 lastDistance = -1.0;
};

PickerResult leaderFromDistanceMatrix(cuda::std::span<const double> distanceMatrix,
                                      int                           numItems,
                                      double                        cutoff,
                                      int                           pickSize,
                                      const std::vector<int>&       firstPicks = {},
                                      cudaStream_t                  stream     = nullptr);

PickerResult fusedLeaderGpu(cuda::std::span<const std::uint32_t> fingerprints,
                            int                                  numFingerprints,
                            int                                  numWords,
                            double                               cutoff,
                            FingerprintSimilarityMetric          metric,
                            int                                  pickSize,
                            const std::vector<int>&              firstPicks = {},
                            cudaStream_t                         stream     = nullptr);

PickerResult maxMinFromDistanceMatrix(cuda::std::span<const double> distanceMatrix,
                                      int                           numItems,
                                      int                           pickSize,
                                      const std::vector<int>&       firstPicks = {},
                                      int                           seed       = -1,
                                      double                        threshold  = -1.0,
                                      cudaStream_t                  stream     = nullptr);

PickerResult fusedMaxMinGpu(cuda::std::span<const std::uint32_t> fingerprints,
                            int                                  numFingerprints,
                            int                                  numWords,
                            int                                  pickSize,
                            FingerprintSimilarityMetric          metric,
                            const std::vector<int>&              firstPicks = {},
                            int                                  seed       = -1,
                            double                               threshold  = -1.0,
                            cudaStream_t                         stream     = nullptr);

ClusteringResult diseFromDistanceMatrix(cuda::std::span<const double> distanceMatrix,
                                        int                           numItems,
                                        double                        cutoff,
                                        bool                          nearestAssignment,
                                        cudaStream_t                  stream = nullptr);

ClusteringResult fusedDiseGpu(cuda::std::span<const std::uint32_t> fingerprints,
                              int                                  numFingerprints,
                              int                                  numWords,
                              double                               cutoff,
                              FingerprintSimilarityMetric          metric,
                              bool                                 nearestAssignment,
                              cudaStream_t                         stream = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_DIVERSITY_PICKERS_H
