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

#include <algorithm>
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

constexpr int kMaxPartialsPerBlock = 256;   // Shared memory partials per block
constexpr int kMaxQueueSize        = 512;   // Shared memory queue size
constexpr int kGlobalOverflowSize  = 2048;   // Global memory overflow per pair (per buffer)

/**
 * @brief Kernel for batch substructure matching.
 *
 * One block per (target, query) pair. Performs graph labeling in shared memory,
 * then dispatches to algorithm-specific search based on template parameter.
 *
 * When pairIndices is null, uses standard contiguous pair indexing via pairOffset.
 * When pairIndices is provided, reads pair index directly from array.
 *
 * @tparam Algo Algorithm to use for the search phase
 * @param pairIndices Optional array of pair indices (null for contiguous mode)
 */
template <SubstructAlgorithm Algo>
__global__ void substructMatchKernel(MoleculesDeviceView             targets,
                                     MoleculesDeviceView             queries,
                                     SubstructMatchResultsDeviceView results,
                                     const int*                      pairIndices = nullptr) {
  // Get pair index - either from array or computed from offset
  const int pairIdx = pairIndices ? pairIndices[blockIdx.x] : (results.pairOffset + blockIdx.x);
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

    // Get global overflow buffers for this block within the batch
    PartialMatch* overflowA = results.getOverflowBuffer(blockIdx.x, 0);
    PartialMatch* overflowB = results.getOverflowBuffer(blockIdx.x, 1);

    gsiBFSSearchGPU<kMaxTargetAtoms, kMaxQueryAtoms>(target,
                                                     query,
                                                     labelMatrix,
                                                     gsiPartials,
                                                     kMaxPartialsPerBlock,
                                                     overflowA,
                                                     overflowB,
                                                     results.overflowSize,
                                                     &sharedMatchCount,
                                                     &sharedReportedCount,
                                                     results.matchIndices,
                                                     maxMatches,
                                                     matchOffset);

  } else if constexpr (Algo == SubstructAlgorithm::WarpUnified) {
    // WUS: Warp-collective BFS with precomputed candidates
    __shared__ CandidateList wusCandidates[kMaxQueryAtoms];
    __shared__ PartialMatch  wusWorkQueue[kMaxQueueSize];

    // Get global overflow buffers for this block within the batch
    PartialMatch* overflowQueue = results.getOverflowBuffer(blockIdx.x, 0);

    warpUnifiedSearchGPU<kMaxTargetAtoms, kMaxQueryAtoms>(target,
                                                          query,
                                                          labelMatrix,
                                                          wusCandidates,
                                                          wusWorkQueue,
                                                          kMaxQueueSize,
                                                          overflowQueue,
                                                          results.overflowSize,
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
  overflowBuffer_.setStream(stream);
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
  overflowBuffer_.setStream(stream_);

  matchCounts_.resize(numPairs);
  reportedCounts_.resize(numPairs);
  pairMatchStarts_.resize(numPairs + 1);
  matchIndices_.resize(totalMatchIndices_);
  queryAtomCounts_.resize(numQueries);

  // Allocate global memory overflow buffers for batched processing
  // Only allocate for min(numTargets, numQueries) pairs at a time to save memory
  overflowSize_ = kGlobalOverflowSize;
  const int overflowBatchSize = std::min(numTargets, numQueries);
  overflowBuffer_.resize(overflowBatchSize * 2 * overflowSize_);

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
  v.matchCounts        = matchCounts_.data();
  v.reportedCounts     = reportedCounts_.data();
  v.pairMatchStarts    = pairMatchStarts_.data();
  v.matchIndices       = matchIndices_.data();
  v.numTargets         = numTargets_;
  v.numQueries         = numQueries_;
  v.queryAtomCounts    = queryAtomCounts_.data();
  v.maxMatchesPerPair  = 0;  // Not used in current implementation
  v.overflowBuffer     = overflowBuffer_.data();
  v.overflowSize       = overflowSize_;
  v.overflowBatchSize  = static_cast<int>(overflowBuffer_.size() / (2 * overflowSize_));
  v.pairOffset         = 0;  // Default, will be set per-batch in getSubstructMatches
  return v;
}

// =============================================================================
// Main API
// =============================================================================

void getSubstructMatches(MoleculesDevice&             targetsDevice,
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

  // Step 1: Separate queries into recursive vs non-recursive
  std::vector<int> nonRecursiveQueries;
  std::vector<int> recursiveQueries;

  for (int q = 0; q < numQueries; ++q) {
    if (q < static_cast<int>(queriesHost.recursivePatterns.size()) &&
        !queriesHost.recursivePatterns[q].empty()) {
      recursiveQueries.push_back(q);
    } else {
      nonRecursiveQueries.push_back(q);
    }
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

  const int threadsPerBlock = 128;
  SubstructMatchResultsDeviceView baseView = results.view();
  const int batchSize = baseView.overflowBatchSize;

  // Helper lambda to launch batched kernel for a set of query indices
  auto launchQueryBatch = [&](const std::vector<int>& queryIndices) {
    if (queryIndices.empty()) return;

    // Build flat array of pair indices: all (target, query) pairs for these queries
    std::vector<int> pairIndices;
    pairIndices.reserve(numTargets * queryIndices.size());
    for (int q : queryIndices) {
      for (int t = 0; t < numTargets; ++t) {
        pairIndices.push_back(t * numQueries + q);
      }
    }

    // Process in batches to respect overflow buffer capacity
    AsyncDeviceVector<int> pairIndicesDev(std::min(static_cast<int>(pairIndices.size()), batchSize), stream);

    for (size_t offset = 0; offset < pairIndices.size(); offset += batchSize) {
      const int batchPairs = std::min(batchSize, static_cast<int>(pairIndices.size() - offset));

      // Copy this batch's indices to device
      pairIndicesDev.copyFromHost(pairIndices.data() + offset, batchPairs);

      switch (algorithm) {
        case SubstructAlgorithm::VF2:
          substructMatchKernel<SubstructAlgorithm::VF2><<<batchPairs, threadsPerBlock, 0, stream>>>(
            targetsDevice.view(), queriesDevice.view(), baseView, pairIndicesDev.data());
          break;
        case SubstructAlgorithm::GSI:
          substructMatchKernel<SubstructAlgorithm::GSI><<<batchPairs, threadsPerBlock, 0, stream>>>(
            targetsDevice.view(), queriesDevice.view(), baseView, pairIndicesDev.data());
          break;
        case SubstructAlgorithm::WarpUnified:
          substructMatchKernel<SubstructAlgorithm::WarpUnified><<<batchPairs, threadsPerBlock, 0, stream>>>(
            targetsDevice.view(), queriesDevice.view(), baseView, pairIndicesDev.data());
          break;
      }

      // Sync before next batch to ensure overflow buffers can be reused
      cudaCheckError(cudaStreamSynchronize(stream));
    }
  };

  // Step 2: Launch recursive subentries FIRST (preprocessing paints bits on target atoms)
  for (int q : recursiveQueries) {
    preprocessRecursiveSmarts(targetsDevice, targetsHost, queriesHost.recursivePatterns[q], stream);
  }

  // Step 3: Launch non-recursive queries (don't need painted bits)
  launchQueryBatch(nonRecursiveQueries);

  // Step 4: Launch recursive queries LAST (need the painted bits from step 2)
  launchQueryBatch(recursiveQueries);

  cudaCheckError(cudaStreamSynchronize(stream));

  cudaCheckError(cudaGetLastError());
}

// =============================================================================
// Recursive SMARTS Preprocessing
// =============================================================================

namespace {

/**
 * @brief Kernel to paint recursive match bits on target atoms.
 *
 * For each match in the results, sets the corresponding recursive pattern bit
 * on the matched target atom (atom 0 of the match, which is the atom that
 * anchored the recursive pattern).
 */
__global__ void paintRecursiveMatchBitsKernel(AtomDataPacked*                     atomDataPacked,
                                              const int*                          batchAtomStarts,
                                              const SubstructMatchResultsDeviceView results,
                                              int                                 patternId,
                                              int                                 queryIdx) {
  const int targetIdx = blockIdx.x;
  if (targetIdx >= results.numTargets) {
    return;
  }

  const int pairIdx      = targetIdx * results.numQueries + queryIdx;
  const int matchCount   = results.reportedCounts[pairIdx];
  const int matchOffset  = results.pairMatchStarts[pairIdx];
  const int queryAtoms   = results.queryAtomCounts[queryIdx];
  const int atomStart    = batchAtomStarts[targetIdx];

  for (int m = threadIdx.x; m < matchCount; m += blockDim.x) {
    const int targetAtomIdx = results.matchIndices[matchOffset + m * queryAtoms];
    if (targetAtomIdx >= 0) {
      atomDataPacked[atomStart + targetAtomIdx].setRecursiveMatchBit(patternId);
    }
  }
}

}  // namespace

void paintRecursiveMatchBits(MoleculesDevice&                   targetsDevice,
                             const SubstructMatchResultsDevice& results,
                             const std::vector<int>&            patternIds,
                             cudaStream_t                       stream) {
  if (patternIds.empty()) {
    return;
  }

  auto resultsView = results.view();
  auto targetsView = targetsDevice.view();

  const int numTargets = resultsView.numTargets;
  const int numQueries = resultsView.numQueries;

  if (numTargets == 0 || numQueries == 0) {
    return;
  }

  const int threadsPerBlock = 64;

  for (int q = 0; q < numQueries && q < static_cast<int>(patternIds.size()); ++q) {
    const int patternId = patternIds[q];
    if (patternId < 0 || patternId >= AtomDataPacked::kMaxRecursivePatterns) {
      continue;
    }

    paintRecursiveMatchBitsKernel<<<numTargets, threadsPerBlock, 0, stream>>>(
      const_cast<AtomDataPacked*>(targetsView.atomDataPacked),
      targetsView.batchAtomStarts,
      resultsView,
      patternId,
      q);
  }

  cudaCheckError(cudaGetLastError());
}

void preprocessRecursiveSmarts(MoleculesDevice&             targetsDevice,
                               const MoleculesHost&         targetsHost,
                               const RecursivePatternInfo&  recursiveInfo,
                               cudaStream_t                 stream) {
  if (recursiveInfo.empty()) {
    return;
  }

  MoleculesHost patternsHost;
  std::vector<int> patternIds;

  for (const auto& entry : recursiveInfo.patterns) {
    if (entry.queryMol != nullptr) {
      addQueryToBatch(entry.queryMol, patternsHost);
      patternIds.push_back(entry.patternId);
    }
  }

  if (patternsHost.numMolecules() == 0) {
    return;
  }

  MoleculesDevice patternsDevice(stream);
  patternsDevice.copyFromHost(patternsHost, stream);

  SubstructMatchResultsDevice results(stream);
  getSubstructMatches(targetsDevice, patternsDevice, targetsHost, patternsHost, results,
                      SubstructAlgorithm::WarpUnified, stream);

  paintRecursiveMatchBits(targetsDevice, results, patternIds, stream);
}

}  // namespace nvMolKit

