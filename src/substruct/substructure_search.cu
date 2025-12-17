// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "substructure_search.cuh"

#include <stdexcept>

#include "cuda_error_check.h"
#include "graph_labeler.cuh"
#include "molecules_device.cuh"
#include "substruct_algos.cuh"

namespace nvMolKit {

namespace {

constexpr std::size_t kMaxTargetAtoms = 128;
constexpr std::size_t kMaxQueryAtoms  = 64;

using LabelMatrixStorage = FlatBitVect<kMaxTargetAtoms * kMaxQueryAtoms>;
using LabelMatrixView    = BitMatrix2DView<kMaxTargetAtoms, kMaxQueryAtoms>;

constexpr int kMaxPartialsPerBlock = 256;
constexpr int kMaxQueueSize        = 512;

/**
 * @brief Kernel for batch substructure matching.
 *
 * One block per (target, query) pair. Performs graph labeling in shared memory,
 * then dispatches to algorithm-specific search based on template parameter.
 *
 * @tparam Algo Algorithm to use for the search phase
 */
template <SubstructAlgorithm Algo>
__global__ void substructMatchKernel(MoleculesDeviceView             targets,
                                     MoleculesDeviceView             queries,
                                     SubstructMatchResultsDeviceView results) {
  // Each block handles one (target, query) pair
  const int pairIdx   = blockIdx.x;
  const int targetIdx = pairIdx / results.numQueries;
  const int queryIdx  = pairIdx % results.numQueries;

  if (targetIdx >= targets.numMolecules || queryIdx >= queries.numMolecules) {
    return;
  }

  // Get molecule views
  const MoleculeView target = getMolecule(targets, targetIdx);
  const MoleculeView query  = getMolecule(queries, queryIdx);

  // Shared memory for label matrix
  __shared__ LabelMatrixStorage sharedLabelMatrix;
  LabelMatrixView               labelMatrix(&sharedLabelMatrix);

  // Populate label matrix using optimized warp-parallel function
  populateLabelMatrixOptimized<kMaxTargetAtoms, kMaxQueryAtoms>(target, query, labelMatrix);

  __syncthreads();

  // Get output buffer info for this pair
  const int matchOffset = results.pairMatchStarts[pairIdx];
  const int maxMatches  = (results.pairMatchStarts[pairIdx + 1] - matchOffset) / query.numAtoms;

  // Initialize result counters
  __shared__ int sharedMatchCount;
  __shared__ int sharedReportedCount;

  if (threadIdx.x == 0) {
    sharedMatchCount    = 0;
    sharedReportedCount = 0;
  }
  __syncthreads();

  // Dispatch to algorithm-specific search
  if constexpr (Algo == SubstructAlgorithm::VF2) {
    // VF2: Each warp handles a different starting target atom
    namespace cg = cooperative_groups;
    auto tile32  = cg::tiled_partition<32>(cg::this_thread_block());
    const int warpId   = tile32.meta_group_rank();
    const int numWarps = tile32.meta_group_size();

    __shared__ VF2State vf2States[4];  // Up to 4 warps

    if (warpId < 4 && tile32.thread_rank() == 0) {
      vf2States[warpId].init(query.numAtoms);
    }
    __syncthreads();

    // Each warp explores from different starting target atoms
    for (int startT = warpId; startT < target.numAtoms; startT += numWarps) {
      if (warpId < 4) {
        vf2SearchGPU<kMaxTargetAtoms, kMaxQueryAtoms>(target,
                                                      query,
                                                      labelMatrix,
                                                      vf2States[warpId],
                                                      startT,
                                                      &sharedMatchCount,
                                                      &sharedReportedCount,
                                                      results.matchIndices,
                                                      maxMatches,
                                                      matchOffset);
      }
    }

  } else if constexpr (Algo == SubstructAlgorithm::GSI) {
    // GSI: BFS level-by-level search
    __shared__ PartialMatch gsiPartials[kMaxPartialsPerBlock * 2];  // Ping-pong buffer

    gsiBFSSearchGPU<kMaxTargetAtoms, kMaxQueryAtoms>(target,
                                                     query,
                                                     labelMatrix,
                                                     gsiPartials,
                                                     kMaxPartialsPerBlock,
                                                     &sharedMatchCount,
                                                     &sharedReportedCount,
                                                     results.matchIndices,
                                                     maxMatches,
                                                     matchOffset);

  } else if constexpr (Algo == SubstructAlgorithm::WarpUnified) {
    // WUS: Warp-collective BFS with precomputed candidates
    __shared__ CandidateList wusCandidates[kMaxQueryAtoms];
    __shared__ PartialMatch  wusWorkQueue[kMaxQueueSize];

    warpUnifiedSearchGPU<kMaxTargetAtoms, kMaxQueryAtoms>(target,
                                                          query,
                                                          labelMatrix,
                                                          wusCandidates,
                                                          wusWorkQueue,
                                                          kMaxQueueSize,
                                                          &sharedMatchCount,
                                                          &sharedReportedCount,
                                                          results.matchIndices,
                                                          maxMatches,
                                                          matchOffset);
  }

  __syncthreads();

  // Write final counts to global memory
  if (threadIdx.x == 0) {
    results.matchCounts[pairIdx]    = sharedMatchCount;
    results.reportedCounts[pairIdx] = sharedReportedCount;
  }
}

}  // namespace

// =============================================================================
// SubstructMatchResultsDevice Implementation
// =============================================================================

void SubstructMatchResultsDevice::setStream(cudaStream_t stream) {
  stream_ = stream;
  matchCounts_.setStream(stream);
  reportedCounts_.setStream(stream);
  pairMatchStarts_.setStream(stream);
  matchIndices_.setStream(stream);
  queryAtomCounts_.setStream(stream);
}

void SubstructMatchResultsDevice::allocate(int                     numTargets,
                                           int                     numQueries,
                                           const std::vector<int>& queryAtomCounts,
                                           const std::vector<int>& maxMatchesPerPairVec) {
  if (numTargets <= 0 || numQueries <= 0) {
    throw std::invalid_argument("numTargets and numQueries must be positive");
  }
  if (static_cast<int>(queryAtomCounts.size()) != numQueries) {
    throw std::invalid_argument("queryAtomCounts size must equal numQueries");
  }

  numTargets_ = numTargets;
  numQueries_ = numQueries;

  const int numPairs = numTargets * numQueries;

  // Store query atom counts for host copy
  hostQueryAtomCounts_ = queryAtomCounts;

  // Compute offsets into matchIndices for each pair
  // Each pair (t, q) can store up to maxMatchesPerPairVec[t] matches
  // Each match for query q requires queryAtomCounts[q] int16_t values
  hostPairMatchStarts_.resize(numPairs + 1);
  hostPairMatchStarts_[0] = 0;

  for (int t = 0; t < numTargets; ++t) {
    for (int q = 0; q < numQueries; ++q) {
      const int pairIdx         = t * numQueries + q;
      const int maxMatches      = maxMatchesPerPairVec[t];
      const int indicesPerMatch = queryAtomCounts[q];
      const int pairCapacity    = maxMatches * indicesPerMatch;
      hostPairMatchStarts_[pairIdx + 1] = hostPairMatchStarts_[pairIdx] + pairCapacity;
    }
  }

  totalMatchIndices_ = hostPairMatchStarts_.back();

  // Allocate device memory
  matchCounts_.setStream(stream_);
  reportedCounts_.setStream(stream_);
  pairMatchStarts_.setStream(stream_);
  matchIndices_.setStream(stream_);
  queryAtomCounts_.setStream(stream_);

  matchCounts_.resize(numPairs);
  reportedCounts_.resize(numPairs);
  pairMatchStarts_.resize(numPairs + 1);
  matchIndices_.resize(totalMatchIndices_);
  queryAtomCounts_.resize(numQueries);

  // Initialize counts and indices to zero
  matchCounts_.zero();
  reportedCounts_.zero();
  matchIndices_.zero();

  // Copy offsets and query atom counts to device
  pairMatchStarts_.setFromVector(hostPairMatchStarts_);
  queryAtomCounts_.setFromVector(queryAtomCounts);
}

void SubstructMatchResultsDevice::copyToHost(SubstructMatchResultsHost& host) const {
  host.numTargets = numTargets_;
  host.numQueries = numQueries_;

  const int numPairs = numTargets_ * numQueries_;

  host.matchCounts.resize(numPairs);
  host.reportedCounts.resize(numPairs);
  host.pairMatchStarts = hostPairMatchStarts_;
  host.matchIndices.resize(totalMatchIndices_);

  matchCounts_.copyToHost(host.matchCounts);
  reportedCounts_.copyToHost(host.reportedCounts);
  matchIndices_.copyToHost(host.matchIndices);
}

SubstructMatchResultsDeviceView SubstructMatchResultsDevice::view() const {
  SubstructMatchResultsDeviceView v;
  v.matchCounts       = matchCounts_.data();
  v.reportedCounts    = reportedCounts_.data();
  v.pairMatchStarts   = pairMatchStarts_.data();
  v.matchIndices      = matchIndices_.data();
  v.numTargets        = numTargets_;
  v.numQueries        = numQueries_;
  v.queryAtomCounts   = queryAtomCounts_.data();
  v.maxMatchesPerPair = 0;  // Not used in current implementation
  return v;
}

// =============================================================================
// Main API
// =============================================================================

void getSubstructMatches(const MoleculesDevice&       targetsDevice,
                         const MoleculesDevice&       queriesDevice,
                         const MoleculesHost&         targetsHost,
                         const MoleculesHost&         queriesHost,
                         SubstructMatchResultsDevice& results,
                         SubstructAlgorithm           algorithm,
                         cudaStream_t                 stream) {
  const int numTargets = static_cast<int>(targetsHost.numMolecules());
  const int numQueries = static_cast<int>(queriesHost.numMolecules());

  if (numTargets == 0 || numQueries == 0) {
    throw std::invalid_argument("Target and query batches must not be empty");
  }

  // Compute atom counts per query molecule
  std::vector<int> queryAtomCounts(numQueries);
  for (int q = 0; q < numQueries; ++q) {
    const int atomStart = queriesHost.batchAtomStarts[q];
    const int atomEnd   = queriesHost.batchAtomStarts[q + 1];
    queryAtomCounts[q]  = atomEnd - atomStart;
  }

  // Compute max matches per target (= target atom count for now)
  std::vector<int> maxMatchesPerPair(numTargets);
  for (int t = 0; t < numTargets; ++t) {
    const int atomStart  = targetsHost.batchAtomStarts[t];
    const int atomEnd    = targetsHost.batchAtomStarts[t + 1];
    maxMatchesPerPair[t] = atomEnd - atomStart;
  }

  // Allocate result buffers
  results.setStream(stream);
  results.allocate(numTargets, numQueries, queryAtomCounts, maxMatchesPerPair);

  // Launch kernel: one block per (target, query) pair
  const int numPairs        = numTargets * numQueries;
  const int threadsPerBlock = 128;  // Multiple warps for parallel operations

  // Dispatch to algorithm-specific kernel instantiation
  switch (algorithm) {
    case SubstructAlgorithm::VF2:
      substructMatchKernel<SubstructAlgorithm::VF2><<<numPairs, threadsPerBlock, 0, stream>>>(
        targetsDevice.view(), queriesDevice.view(), results.view());
      break;

    case SubstructAlgorithm::GSI:
      substructMatchKernel<SubstructAlgorithm::GSI><<<numPairs, threadsPerBlock, 0, stream>>>(
        targetsDevice.view(), queriesDevice.view(), results.view());
      break;

    case SubstructAlgorithm::WarpUnified:
      substructMatchKernel<SubstructAlgorithm::WarpUnified><<<numPairs, threadsPerBlock, 0, stream>>>(
        targetsDevice.view(), queriesDevice.view(), results.view());
      break;
  }

  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit

