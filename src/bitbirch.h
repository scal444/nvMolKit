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

//! Tree shape, batching, and memory-placement options for bitBirchGpu.
struct BitBirchOptions {
  int         branchingFactor       = 254;    //!< Maximum entries per tree node; at least 3.
  int         batchSize             = 1024;   //!< Fingerprints routed per insertion epoch.
  std::size_t summaryCacheBytes     = 0;      //!< GPU cache budget for Bit Feature sums; zero keeps all on the GPU.
  std::size_t fingerprintCacheBytes = 0;      //!< GPU cache budget for retained singletons; needs fingerprintsOnHost.
  bool        fingerprintsOnHost    = false;  //!< Fingerprints are host memory, streamed to the GPU per batch.
  bool        clusterIdsOnHost      = false;  //!< Write labels to mapped pinned host memory.
  bool        returnCentroids       = false;  //!< Also return packed majority centroids in cluster-ID order.
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
                           const BitBirchOptions&               options = BitBirchOptions(),
                           cudaStream_t                         stream  = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_BITBIRCH_H
