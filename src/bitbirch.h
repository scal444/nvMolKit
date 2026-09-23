// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This implementation is independently derived from the published algorithm
// and equations in BitBIRCH (https://doi.org/10.1039/D5DD00030K), BitBIRCH
// refinement strategies (https://doi.org/10.1021/acs.jcim.5c00627), and
// BitBIRCH-Lean (https://doi.org/10.1101/2025.10.22.684015). No source from the
// GPL-licensed bblean or bitbirch packages is used.

#ifndef NVMOLKIT_BITBIRCH_H
#define NVMOLKIT_BITBIRCH_H

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cuda/std/span>

#include "src/utils/device_vector.h"
#include "src/utils/host_vector.h"

namespace nvMolKit {

struct BitBirchResult {
  AsyncDeviceVector<int>           clusterIds;
  AsyncDeviceVector<std::uint32_t> centroids;
  int                              numClusters = 0;
  int                              numWords    = 0;
  PinnedHostVector<int>            hostClusterIds{};
  bool                             clusterIdsOnHost = false;
};

/**
 * Build one BitBIRCH tree using snapshot routing and ordered leaf owners.
 * Structural changes occur only between insertion epochs, avoiding concurrent
 * topology mutation without constructing independent trees and merging them.
 */
BitBirchResult bitBirchGpu(cuda::std::span<const std::uint32_t> fingerprints,
                           int                                  numFingerprints,
                           int                                  numWords,
                           double                               threshold,
                           int                                  branchingFactor,
                           int                                  batchSize,
                           std::size_t                          summaryCacheBytes     = 0,
                           bool                                 fingerprintsOnHost    = false,
                           bool                                 clusterIdsOnHost      = false,
                           bool                                 returnCentroids       = false,
                           cudaStream_t                         stream                = nullptr,
                           std::size_t                          fingerprintCacheBytes = 0);

}  // namespace nvMolKit

#endif  // NVMOLKIT_BITBIRCH_H
