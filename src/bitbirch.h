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
#include <memory>

#include "src/utils/device_vector.h"

namespace nvMolKit {

struct BitBirchResult {
  AsyncDeviceVector<int>           clusterIds;
  AsyncDeviceVector<std::uint32_t> centroids;
  int                              numClusters = 0;
  int                              numWords    = 0;
};

//! Tree shape, batching, and memory-placement options for bitBirchGpu.
struct BitBirchOptions {
  int         branchingFactor       = 254;    //!< Maximum entries per tree node; at least 3.
  int         batchSize             = 1024;   //!< Fingerprints routed per insertion epoch.
  std::size_t summaryCacheBytes     = 0;      //!< GPU cache budget for Bit Feature sums; zero keeps all on the GPU.
  std::size_t fingerprintCacheBytes = 0;      //!< GPU cache budget for retained singletons; needs fingerprintsOnHost.
  bool        fingerprintsOnHost    = false;  //!< Fingerprints are host memory, streamed to the GPU per batch.
  //! When set, labels go to this host array of numFingerprints ints instead of clusterIds. Only one batch of labels is
  //! kept on the GPU; each finished batch is staged through a small pinned buffer. Labels are cluster IDs, which stay
  //! below 2^31 for any numFingerprints (the tree's entry IDs are int32 and checked), so int32 suffices.
  int*        hostClusterIds        = nullptr;
  bool        returnCentroids       = false;  //!< Also return packed majority centroids in cluster-ID order.
};

class BitBirchTreeImpl;

/**
 * Persistent BitBIRCH tree that takes its rows in any number of chunks.
 *
 * Rows are cut into the same batches as one bitBirchGpu call over all rows
 * would make, so labels do not depend on the chunking. Chunks may be host or
 * device memory; each is read only during its insert call. Label destination
 * (options.hostClusterIds, or a device array) holds maxRows labels.
 * options.fingerprintsOnHost is ignored: singleton fingerprints are always
 * copied into the tree's own storage.
 */
class BitBirchTree {
 public:
  BitBirchTree(int                    numWords,
               double                 threshold,
               std::int64_t           maxRows,
               const BitBirchOptions& options = BitBirchOptions(),
               cudaStream_t           stream  = nullptr);
  explicit BitBirchTree(std::unique_ptr<BitBirchTreeImpl> impl);
  ~BitBirchTree();
  BitBirchTree(BitBirchTree&&) noexcept;
  BitBirchTree& operator=(BitBirchTree&&) noexcept;

  //! Appends numRows packed rows as input rows [rows(), rows() + numRows).
  void           insert(const std::uint32_t* fingerprints, std::int64_t numRows, bool onHost);
  //! Rows inserted so far.
  std::int64_t   rows() const noexcept;
  //! Clusters so far (as of the last completed insertion batch).
  int            clusters() const noexcept;
  //! Inserts any held rows, then labels every row; the tree accepts no more rows.
  BitBirchResult finalize();

 private:
  std::unique_ptr<BitBirchTreeImpl> impl_;
};

/**
 * Build one BitBIRCH tree using snapshot routing and ordered leaf owners.
 * Structural changes occur only between insertion epochs, avoiding concurrent
 * topology mutation without constructing independent trees and merging them.
 */
BitBirchResult bitBirchGpu(cuda::std::span<const std::uint32_t> fingerprints,
                           std::int64_t                         numFingerprints,
                           int                                  numWords,
                           double                               threshold,
                           const BitBirchOptions&               options = BitBirchOptions(),
                           cudaStream_t                         stream  = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_BITBIRCH_H
