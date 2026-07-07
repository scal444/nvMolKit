// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FMCS_CUDA_FMCS_CUH
#define FMCS_CUDA_FMCS_CUH

// Public API for the fMCS (seed-grow connected-MCES) CUDA solver.
//
// Algorithm reference: RDKit's rdFMCS (Novartis, 2014), re-implemented
// here as a GPU-native grow-and-verify search that enumerates the
// connected-subgraph lattice of the smaller input and matches each
// candidate against the larger input on the device.
//
// Supported feature surface: unlabeled topology, optional exact
// vertex/edge-label matching, MaximizeBonds objective, Threshold=1.0,
// connected-only result.  RDKit
// atomCompare/bondCompare modes map to caller-provided labels: CompareAny
// disables the corresponding table, while CompareElements/CompareIsotopes
// and CompareOrder/CompareOrderExact require the caller to encode those
// semantics as uint16 labels.  RingMatchesRingOnly is likewise supported
// when ring membership is encoded in atom/bond labels. CompleteRingsOnly is
// enforced as a final-candidate condition while partial rings remain growable.
// Chirality, fused-ring strictness, and Threshold < 1.0 are out of scope.

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "fmcs_cuda/fmcs_config.cuh"
#include "fmcs_cuda/fmcs_labeled_graph.h"
#include "fmcs_cuda/fmcs_stats.cuh"
#include "mcs_common/mcs_types.cuh"

namespace mcs {
namespace fmcs {

/// Algorithm-level parameters for the fMCS solver.
///
/// The objective (connected MCES, MaximizeBonds, Threshold=1) is baked
/// into the algorithm itself and is not exposed as a knob.  Callers that
/// need a different objective must use RDKit's CPU FMCS; the host dispatch
/// layer (mcs_search / mcs_rdkit_adapter) already falls back to it for
/// unsupported parameter combinations.
struct Parameters {
  /// CUDA block size for the per-pair kernel. Supported: 352 and 640.
  /// Block size 640 supports tier-128 when the substructure scratch is placed
  /// in global memory (see scratchLocation); the default Auto policy selects
  /// that automatically.
  int                 blockSize                    = kFmcsMaxBlockSizeTwoBlockOccupancy;
  /// Placement of the per-group substructure fallback scratch.  Auto keeps
  /// small/hot configs on shared memory and only moves scratch to global for
  /// block size 640 @ tier-128 (where static shared cannot fit). Explicit
  /// Shared with those configurations is rejected at dispatch.
  FmcsScratchLocation scratchLocation              = FmcsScratchLocation::Auto;
  /// Dormant readiness flag for the extended-shared-memory carveout
  /// (analysis/fmcs_scratch_placement_plan.md section 7).  Has no effect until
  /// the dynamic-shared follow-on lands; kept here so that change need not
  /// re-plumb the API.
  bool                enableExtendedSharedCarveout = false;
  /// Per-pair wall timeout in milliseconds.  0 = no timeout.
  float               timeoutMs                    = 0;
  /// Max pairs per tier chunk in the batch API.  0 = default chunk size.
  int                 batchSize                    = 0;
  /// Number of asynchronous executor streams for tier sub-batches. 1 = serial.
  int                 executorsPerRunner           = 1;
  /// Optional absolute wall deadline in seconds since epoch; pairs not
  /// started by this time are reported as timed out.  0 = disabled.
  double              wallDeadlineSec              = 0;
  /// For labeled inputs, require exact vertex-label equality.  When false,
  /// atom compatibility is CompareAny-style.
  bool                matchVertexLabels            = true;
  /// For labeled inputs, require exact edge-label equality.  When false,
  /// bond compatibility is CompareAny-style.
  bool                matchEdgeLabels              = true;
  /// Require every selected bond that belongs to an input cycle to remain in
  /// a cycle in the selected subgraph. Partial-ring search states remain
  /// growable but cannot become the incumbent.
  bool                completeRingsOnly            = false;
};

/// Find the connected MCES for a batch of unlabeled graph pairs.
///
/// `graphsA` and `graphsB` must have equal length.  For unlabeled input,
/// atom/bond compatibility is topology-only.  When a graph exceeds the maximum
/// supported maxSize, that pair's result has `overflowed` set and all counts
/// are zero.  Per-pair timing/stat output is available when the corresponding
/// NVMOLKIT_ENABLE_MCS_* compile option is enabled.
std::vector<MCSResult> findMCESfMCSBatch(const std::vector<Graph>&    graphsA,
                                         const std::vector<Graph>&    graphsB,
                                         Parameters                   params             = {},
                                         std::vector<float>*          perPairTimesMs     = nullptr,
                                         cudaStream_t                 stream             = nullptr,
                                         std::vector<ExecutionStats>* perPairStats       = nullptr,
                                         std::vector<ExecutionStats>* perPairTimingStats = nullptr);

/// Labeled variant: optional exact vertex and edge label equality.
///
/// Label semantics: with default parameters, two atoms may be paired iff
/// their `vertexLabels` agree, and two bonds may be paired iff both endpoint
/// pairs are compatible and the `edgeLabels` entries agree.  Set
/// `Parameters::matchVertexLabels`
/// or `Parameters::matchEdgeLabels` false for CompareAny-style matching on
/// that axis.
std::vector<MCSResult> findMCESfMCSBatchLabeled(const std::vector<LabeledGraph>& graphsA,
                                                const std::vector<LabeledGraph>& graphsB,
                                                Parameters                       params             = {},
                                                std::vector<float>*              perPairTimesMs     = nullptr,
                                                cudaStream_t                     stream             = nullptr,
                                                std::vector<ExecutionStats>*     perPairStats       = nullptr,
                                                std::vector<ExecutionStats>*     perPairTimingStats = nullptr);

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_CUH
