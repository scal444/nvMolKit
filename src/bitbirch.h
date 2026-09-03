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

#include <cstdint>
#include <cuda/std/span>

#include "src/utils/device_vector.h"

namespace nvMolKit {

enum class BitBirchMergeCriterion : std::uint8_t {
  Diameter,
  ToleranceDiameter
};

struct BitBirchResult {
  AsyncDeviceVector<int>           clusterIds;
  AsyncDeviceVector<std::uint32_t> centroids;
  int                              numClusters = 0;
  int                              numWords    = 0;
};

/**
 * Correctness-first ordered BitBIRCH tree construction on one GPU thread.
 *
 * This serial path supplies deterministic small-workload behavior and a native
 * differential target for the parallel partial-tree implementation. Tree and
 * scratch state are index-based global allocations owned by AsyncDeviceVector.
 */
BitBirchResult bitBirchSerialGpu(cuda::std::span<const std::uint32_t> fingerprints,
                                 int                                  numFingerprints,
                                 int                                  numWords,
                                 double                               threshold,
                                 int                                  branchingFactor = 254,
                                 BitBirchMergeCriterion               mergeCriterion = BitBirchMergeCriterion::Diameter,
                                 double                               tolerance      = 0.05,
                                 bool                                 returnCentroids = false,
                                 cudaStream_t                         stream          = nullptr);

/**
 * Build ordered partial trees concurrently and merge their Bit Features.
 *
 * A value of one preserves the serial-tree result. Larger values partition
 * the ordered input into contiguous ranges, build one tree per CUDA block,
 * and insert the resulting leaf summaries into a final tree.
 */
BitBirchResult bitBirchGpu(cuda::std::span<const std::uint32_t> fingerprints,
                           int                                  numFingerprints,
                           int                                  numWords,
                           double                               threshold,
                           int                                  branchingFactor,
                           BitBirchMergeCriterion               mergeCriterion,
                           double                               tolerance,
                           int                                  numPartitions,
                           bool                                 returnCentroids = false,
                           cudaStream_t                         stream          = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_BITBIRCH_H
