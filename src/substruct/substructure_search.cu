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
#include "global_pool.cuh"
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

constexpr bool kDebugDumpLabelMatrix = false;  ///< Dump full label matrices after recursive preprocessing
constexpr bool kDebugPaintRecursive  = false;  ///< Debug recursive bit painting kernel

/**
 * @brief Kernel for batch substructure matching.
 *
 * One block per (target, query) pair. Performs graph labeling in shared memory,
 * then dispatches to algorithm-specific search based on template parameter.
 *
 * @tparam Algo Algorithm to use for the search phase
 * @param pairIndices Array of global pair indices for this batch
 */
template <SubstructAlgorithm Algo>
__global__ void substructMatchKernel(MoleculesDeviceView             targets,
                                     MoleculesDeviceView             queries,
                                     SubstructMatchResultsDeviceView results,
                                     const int*                      pairIndices) {
  const int batchLocalIdx = blockIdx.x;
  const int pairIdx       = pairIndices[batchLocalIdx];
  const int targetIdx     = pairIdx / results.numQueries;
  const int queryIdx      = pairIdx % results.numQueries;

  if (targetIdx >= targets.numMolecules || queryIdx >= queries.numMolecules) {
    return;
  }

  // Get molecule views
  const MoleculeView target = getMolecule(targets, targetIdx);
  const MoleculeView query  = getMolecule(queries, queryIdx);

  // Shared memory for label matrix
  __shared__ LabelMatrixStorage sharedLabelMatrix;
  LabelMatrixView               labelMatrix(&sharedLabelMatrix);

  // Get pointer to per-pair recursive match bits using batch-local index
  const uint32_t* pairRecursiveBits = results.recursiveMatchBits
                                        ? &results.recursiveMatchBits[batchLocalIdx * results.maxTargetAtoms]
                                        : nullptr;

  // Populate label matrix using optimized warp-parallel function
  populateLabelMatrixOptimized<kMaxTargetAtoms, kMaxQueryAtoms>(target, query, labelMatrix, pairRecursiveBits);

  __syncthreads();

  if constexpr (kDebugDumpLabelMatrix) {
    if (threadIdx.x == 0) {
      printf("[LabelDump] pair=%d (target=%d, query=%d): targetAtoms=%d, queryAtoms=%d\n",
             pairIdx, targetIdx, queryIdx, target.numAtoms, query.numAtoms);
      printf("[LabelDump] Recursive bits per target atom:\n");
      for (int t = 0; t < target.numAtoms; ++t) {
        uint32_t bits = pairRecursiveBits ? pairRecursiveBits[t] : 0;
        printf("[LabelDump]   t%d=0x%08x\n", t, bits);
      }
      printf("[LabelDump] Label matrix (row=target, col=query, 1=compatible):\n");
      printf("[LabelDump]     ");
      for (int q = 0; q < query.numAtoms; ++q) {
        printf("q%d ", q);
      }
      printf("\n");
      for (int t = 0; t < target.numAtoms; ++t) {
        printf("[LabelDump] t%2d: ", t);
        for (int q = 0; q < query.numAtoms; ++q) {
          printf("%d  ", labelMatrix.get(t, q) ? 1 : 0);
        }
        printf("\n");
      }
    }
    __syncthreads();
  }

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
                                                     results.getOverflowBuffer(0),
                                                     results.getOverflowBuffer(1),
                                                     results.getOverflowCapacity(),
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
                                                          results.getOverflowBuffer(0),
                                                          results.getOverflowCapacity(),
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

/**
 * @brief Paint mode kernel for recursive SMARTS preprocessing.
 *
 * Instead of storing match mappings, directly paints recursive match bits
 * into the output buffer. This avoids overflow issues since painting is
 * idempotent (multiple matches with same first atom just set same bit).
 *
 * @tparam Algo Algorithm to use for the search phase
 * @param targets Target molecules
 * @param queries Query molecules (recursive patterns)
 * @param outputRecursiveBits Buffer to paint bits into (batch-sized)
 * @param maxTargetAtoms Stride for recursiveBits indexing  
 * @param outputNumQueries Number of queries in the output (main query) results
 * @param patternId Bit position to set (0-31)
 * @param mainQueryIdx Which main query these patterns belong to
 * @param batchPairOffset Global pair index where the current batch starts
 * @param batchSize Number of pairs in the current batch
 */
template <SubstructAlgorithm Algo>
__global__ void substructPaintKernel(MoleculesDeviceView targets,
                                     MoleculesDeviceView queries,
                                     uint32_t*           outputRecursiveBits,
                                     int                 maxTargetAtoms,
                                     int                 outputNumQueries,
                                     int                 patternId,
                                     int                 mainQueryIdx,
                                     int                 batchPairOffset,
                                     int                 batchSize,
                                     PartialMatch*       overflowA,
                                     PartialMatch*       overflowB,
                                     int                 overflowCapacity) {
  const int targetIdx = blockIdx.x / queries.numMolecules;
  const int queryIdx  = blockIdx.x % queries.numMolecules;

  if (targetIdx >= targets.numMolecules || queryIdx >= queries.numMolecules) {
    return;
  }

  // Compute global pair index for this (target, mainQuery) combination
  const int globalPairIdx = targetIdx * outputNumQueries + mainQueryIdx;

  // Check if this pair is within the current batch
  if (globalPairIdx < batchPairOffset || globalPairIdx >= batchPairOffset + batchSize) {
    return;
  }

  // Compute batch-local index for output
  const int batchLocalPairIdx = globalPairIdx - batchPairOffset;

  const MoleculeView target = getMolecule(targets, targetIdx);
  const MoleculeView query  = getMolecule(queries, queryIdx);

  __shared__ LabelMatrixStorage sharedLabelMatrix;
  LabelMatrixView               labelMatrix(&sharedLabelMatrix);

  populateLabelMatrixOptimized<kMaxTargetAtoms, kMaxQueryAtoms>(target, query, labelMatrix, nullptr);
  __syncthreads();

  __shared__ int sharedMatchCount;
  __shared__ int sharedReportedCount;

  if (threadIdx.x == 0) {
    sharedMatchCount    = 0;
    sharedReportedCount = 0;
  }
  __syncthreads();

  PaintModeParams paintParams;
  paintParams.recursiveBits  = outputRecursiveBits;
  paintParams.patternId      = patternId;
  paintParams.maxTargetAtoms = maxTargetAtoms;
  paintParams.outputPairIdx  = batchLocalPairIdx;

  // Get per-block overflow buffer offsets (GSI uses 2 buffers, WUS uses 1)
  constexpr int gsiBuffersPerBlock = 2;
  constexpr int wusBuffersPerBlock = 1;

  if constexpr (Algo == SubstructAlgorithm::GSI) {
    __shared__ PartialMatch gsiPartials[kMaxPartialsPerBlock * 2];

    PartialMatch* blockOverflowA = overflowA + blockIdx.x * gsiBuffersPerBlock * overflowCapacity;
    PartialMatch* blockOverflowB = blockOverflowA + overflowCapacity;

    gsiBFSSearchGPU<kMaxTargetAtoms, kMaxQueryAtoms, SubstructOutputMode::PaintBits>(
      target, query, labelMatrix,
      gsiPartials, kMaxPartialsPerBlock,
      blockOverflowA, blockOverflowB, overflowCapacity,
      &sharedMatchCount, &sharedReportedCount,
      nullptr, 0, 0,  // No match storage needed
      paintParams);

  } else if constexpr (Algo == SubstructAlgorithm::WarpUnified) {
    __shared__ CandidateList wusCandidates[kMaxQueryAtoms];
    __shared__ PartialMatch  wusWorkQueue[kMaxQueueSize];

    PartialMatch* blockOverflow = overflowA + blockIdx.x * wusBuffersPerBlock * overflowCapacity;

    warpUnifiedSearchGPU<kMaxTargetAtoms, kMaxQueryAtoms, SubstructOutputMode::PaintBits>(
      target, query, labelMatrix,
      wusCandidates, wusWorkQueue, kMaxQueueSize,
      blockOverflow, overflowCapacity,
      &sharedMatchCount, &sharedReportedCount,
      nullptr, 0, 0,  // No match storage needed
      paintParams);
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
  recursiveMatchBits_.setStream(stream);
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

  // Store max target atoms for later batch allocation
  maxTargetAtoms_ = *std::ranges::max_element(maxMatchesPerPairVec);

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
  v.matchCounts              = matchCounts_.data();
  v.reportedCounts           = reportedCounts_.data();
  v.pairMatchStarts          = pairMatchStarts_.data();
  v.matchIndices             = matchIndices_.data();
  v.numTargets               = numTargets_;
  v.numQueries               = numQueries_;
  v.queryAtomCounts          = queryAtomCounts_.data();
  v.maxMatchesPerPair        = 0;  // Not used in current implementation
  v.overflowBuffer           = overflowBuffer_.data();
  v.overflowEntriesPerBuffer = kOverflowEntriesPerBuffer;
  v.overflowBuffersPerBlock  = overflowBuffersPerBlock_;
  v.recursiveMatchBits       = recursiveMatchBits_.data();
  v.maxTargetAtoms           = maxTargetAtoms_;
  return v;
}

void SubstructMatchResultsDevice::allocateOverflow(int batchSize, int numBuffersPerBlock) {
  overflowBuffersPerBlock_ = numBuffersPerBlock;
  const int totalEntries = batchSize * numBuffersPerBlock * kOverflowEntriesPerBuffer;
  overflowBuffer_.setStream(stream_);
  overflowBuffer_.resize(totalEntries);
}

void SubstructMatchResultsDevice::allocateBatchRecursiveBits(int batchSize, int maxTargetAtoms) {
  maxTargetAtoms_ = maxTargetAtoms;
  recursiveMatchBits_.setStream(stream_);
  recursiveMatchBits_.resize(batchSize * maxTargetAtoms);
}

void SubstructMatchResultsDevice::zeroRecursiveBits() {
  recursiveMatchBits_.zero();
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
                         cudaStream_t                 stream,
                         int                          batchSize) {
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
  int maxTargetAtoms = 0;
  for (int t = 0; t < numTargets; ++t) {
    const int atomStart  = targetsHost.batchAtomStarts[t];
    const int atomEnd    = targetsHost.batchAtomStarts[t + 1];
    maxMatchesPerPair[t] = atomEnd - atomStart;
    maxTargetAtoms = std::max(maxTargetAtoms, maxMatchesPerPair[t]);
  }

  // Allocate result buffers (output - sized for all pairs)
  results.setStream(stream);
  results.allocate(numTargets, numQueries, queryAtomCounts, maxMatchesPerPair);

  // Allocate batch-sized scratch buffers
  const int numPairs = numTargets * numQueries;
  const int effectiveBatchSize = std::min(batchSize, numPairs);
  const int numBuffersPerBlock = (algorithm == SubstructAlgorithm::GSI) ? 2 : 1;

  results.allocateBatchRecursiveBits(effectiveBatchSize, maxTargetAtoms);
  results.allocateOverflow(effectiveBatchSize, numBuffersPerBlock);

  constexpr int threadsPerBlock = 128;

  // Process all pairs in batches
  for (int batchStart = 0; batchStart < numPairs; batchStart += batchSize) {
    const int batchEnd = std::min(batchStart + batchSize, numPairs);
    const int numPairsInBatch = batchEnd - batchStart;

    // Zero recursive bits for this batch
    results.zeroRecursiveBits();

    // Build pair indices for this batch (contiguous global indices)
    std::vector<int> pairIndices(numPairsInBatch);
    for (int i = 0; i < numPairsInBatch; ++i) {
      pairIndices[i] = batchStart + i;
    }

    // Preprocess recursive patterns for queries that have pairs in this batch
    // Only call once per query (the paint kernel filters by batch range internally)
    std::vector<bool> queryProcessed(numQueries, false);
    for (int i = 0; i < numPairsInBatch; ++i) {
      const int pairIdx = batchStart + i;
      const int queryIdx = pairIdx % numQueries;

      if (!queryProcessed[queryIdx] &&
          queryIdx < static_cast<int>(queriesHost.recursivePatterns.size()) &&
          !queriesHost.recursivePatterns[queryIdx].empty()) {
        preprocessRecursiveSmarts(targetsDevice, targetsHost, queriesHost.recursivePatterns[queryIdx],
                                  results, queryIdx, numQueries, batchStart, numPairsInBatch,
                                  algorithm, stream);
        queryProcessed[queryIdx] = true;
      }
    }

    // Get view after scratch allocation
    SubstructMatchResultsDeviceView batchView = results.view();

    // Copy pair indices to device
    AsyncDeviceVector<int> pairIndicesDev(numPairsInBatch, stream);
    pairIndicesDev.copyFromHost(pairIndices.data(), numPairsInBatch);

    switch (algorithm) {
      case SubstructAlgorithm::VF2:
        substructMatchKernel<SubstructAlgorithm::VF2><<<numPairsInBatch, threadsPerBlock, 0, stream>>>(
          targetsDevice.view(), queriesDevice.view(), batchView, pairIndicesDev.data());
        break;
      case SubstructAlgorithm::GSI:
        substructMatchKernel<SubstructAlgorithm::GSI><<<numPairsInBatch, threadsPerBlock, 0, stream>>>(
          targetsDevice.view(), queriesDevice.view(), batchView, pairIndicesDev.data());
        break;
      case SubstructAlgorithm::WarpUnified:
        substructMatchKernel<SubstructAlgorithm::WarpUnified><<<numPairsInBatch, threadsPerBlock, 0, stream>>>(
          targetsDevice.view(), queriesDevice.view(), batchView, pairIndicesDev.data());
        break;
    }

    cudaCheckError(cudaStreamSynchronize(stream));
  }

  cudaCheckError(cudaGetLastError());
}

// =============================================================================
// Recursive SMARTS Preprocessing
// =============================================================================

void preprocessRecursiveSmarts(const MoleculesDevice&             targetsDevice,
                               const MoleculesHost&         targetsHost,
                               const RecursivePatternInfo&  recursiveInfo,
                               const SubstructMatchResultsDevice& outputResults,
                               const int                          mainQueryIdx,
                               const int                          numQueries,
                               const int                          batchPairOffset,
                               const int                          batchSize,
                               const SubstructAlgorithm           algorithm,
                               cudaStream_t                 stream) {
  if (recursiveInfo.empty()) {
    if constexpr (kDebugPaintRecursive) {
      printf("[PreprocessRecursive] mainQueryIdx=%d: No recursive patterns\n", mainQueryIdx);
    }
    return;
  }

  if constexpr (kDebugPaintRecursive) {
    printf("[PreprocessRecursive] mainQueryIdx=%d: Processing %zu recursive patterns\n",
           mainQueryIdx, recursiveInfo.patterns.size());
  }

  const auto outputView = outputResults.view();
  const int numTargets = targetsDevice.view().numMolecules;

  // Process each recursive pattern with the paint kernel
  for (const auto& entry : recursiveInfo.patterns) {
    if (entry.queryMol == nullptr) {
      continue;
    }

    if constexpr (kDebugPaintRecursive) {
      printf("[PreprocessRecursive]   Pattern %d: launching paint kernel\n", entry.patternId);
    }

    // Build single-pattern batch
    MoleculesHost patternHost;
    addQueryToBatch(entry.queryMol, patternHost);

    MoleculesDevice patternDevice(stream);
    patternDevice.copyFromHost(patternHost, stream);

    const size_t numBlocks = numTargets * patternHost.numMolecules();
    constexpr int threadsPerBlock = 128;

    // Allocate overflow buffers (GSI uses 2 buffers per block, WUS uses 1)
    constexpr int gsiBuffersPerBlock = 2;
    constexpr int wusBuffersPerBlock = 1;
    const int overflowSizeGSI = numBlocks * gsiBuffersPerBlock * kOverflowEntriesPerBuffer;
    const int overflowSizeWUS = numBlocks * wusBuffersPerBlock * kOverflowEntriesPerBuffer;

    switch (algorithm) {
      case SubstructAlgorithm::VF2:
        // VF2 doesn't support paint mode, fall back to GSI
      case SubstructAlgorithm::GSI: {
        AsyncDeviceVector<PartialMatch> overflowBuf(overflowSizeGSI, stream);
        substructPaintKernel<SubstructAlgorithm::GSI><<<numBlocks, threadsPerBlock, 0, stream>>>(
          targetsDevice.view(),
          patternDevice.view(),
          outputView.recursiveMatchBits,
          outputView.maxTargetAtoms,
          numQueries,
          entry.patternId,
          mainQueryIdx,
          batchPairOffset,
          batchSize,
          overflowBuf.data(),
          overflowBuf.data(),  // Both pointers into same buffer, kernel offsets internally
          kOverflowEntriesPerBuffer);
        break;
      }
      case SubstructAlgorithm::WarpUnified: {
        AsyncDeviceVector<PartialMatch> overflowBuf(overflowSizeWUS, stream);
        substructPaintKernel<SubstructAlgorithm::WarpUnified><<<numBlocks, threadsPerBlock, 0, stream>>>(
          targetsDevice.view(),
          patternDevice.view(),
          outputView.recursiveMatchBits,
          outputView.maxTargetAtoms,
          numQueries,
          entry.patternId,
          mainQueryIdx,
          batchPairOffset,
          batchSize,
          overflowBuf.data(),
          overflowBuf.data(),  // WUS only uses first pointer
          kOverflowEntriesPerBuffer);
        break;
      }
    }

    cudaCheckError(cudaGetLastError());
  }
}

}  // namespace nvMolKit

