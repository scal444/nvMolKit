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
#include "nvtx.h"
namespace nvMolKit {

namespace {

constexpr std::size_t kMaxTargetAtoms = 128;
constexpr std::size_t kMaxQueryAtoms  = 64;

constexpr int threadsPerBlock = 256;

using LabelMatrixStorage = FlatBitVect<kMaxTargetAtoms * kMaxQueryAtoms>;
using LabelMatrixView    = BitMatrix2DView<kMaxTargetAtoms, kMaxQueryAtoms>;

constexpr int kMaxPartialsPerBlock = 256;   // Shared memory partials per block
constexpr int kMaxQueueSize        = 512;   // Shared memory queue size
constexpr int kWarpsPerBlock       = threadsPerBlock / 32;

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

    __shared__ VF2State vf2States[kWarpsPerBlock];

    if (tile32.thread_rank() == 0) {
      vf2States[warpId].init(query.numAtoms);
    }
    __syncthreads();

    // Each warp explores from different starting target atoms
    for (int startT = warpId; startT < target.numAtoms; startT += numWarps) {
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
 * Block indexing: blockIdx.x = targetIdx * numPatterns + patternIdx
 *
 * When patternEntries is non-null (batched mode), reads mainQueryIdx and patternId
 * from the array. Otherwise uses the scalar parameters.
 *
 * @tparam Algo Algorithm to use for the search phase
 * @param targets Target molecules
 * @param patterns Query molecules (recursive patterns)
 * @param patternEntries Per-pattern metadata array (null for single-pattern mode)
 * @param numPatterns Number of patterns (patterns.numMolecules)
 * @param outputRecursiveBits Buffer to paint bits into (batch-sized)
 * @param maxTargetAtoms Stride for recursiveBits indexing
 * @param outputNumQueries Number of queries in the output (main query) results
 * @param defaultPatternId Bit position to set (0-31), used when patternEntries is null
 * @param defaultMainQueryIdx Main query index, used when patternEntries is null
 * @param batchPairOffset Global pair index where the current batch starts
 * @param batchSize Number of pairs in the current batch
 * @param firstTargetIdx First target index to process (offset for block indexing)
 */
template <SubstructAlgorithm Algo>
__global__ void substructPaintKernel(MoleculesDeviceView         targets,
                                     MoleculesDeviceView         patterns,
                                     const BatchedPatternEntry*  patternEntries,
                                     int                         numPatterns,
                                     uint32_t*                   outputRecursiveBits,
                                     int                         maxTargetAtoms,
                                     int                         outputNumQueries,
                                     int                         defaultPatternId,
                                     int                         defaultMainQueryIdx,
                                     int                         batchPairOffset,
                                     int                         batchSize,
                                     PartialMatch*               overflowA,
                                     PartialMatch*               overflowB,
                                     int                         overflowCapacity,
                                     int                         firstTargetIdx) {
  const int localTargetIdx   = blockIdx.x / numPatterns;
  const int targetIdx        = firstTargetIdx + localTargetIdx;
  const int localPatternIdx  = blockIdx.x % numPatterns;

  if (targetIdx >= targets.numMolecules || localPatternIdx >= numPatterns) {
    return;
  }

  // Get mainQueryIdx, patternId, and actual pattern molecule index from array or scalar params
  const int mainQueryIdx  = patternEntries ? patternEntries[localPatternIdx].mainQueryIdx : defaultMainQueryIdx;
  const int patternId     = patternEntries ? patternEntries[localPatternIdx].patternId : defaultPatternId;
  const int patternMolIdx = patternEntries ? patternEntries[localPatternIdx].patternMolIdx : localPatternIdx;

  // Compute global pair index for this (target, mainQuery) combination
  const int globalPairIdx = targetIdx * outputNumQueries + mainQueryIdx;

  // Check if this pair is within the current batch
  if (globalPairIdx < batchPairOffset || globalPairIdx >= batchPairOffset + batchSize) {
    return;
  }

  // Compute batch-local index for output
  const int batchLocalPairIdx = globalPairIdx - batchPairOffset;

  const MoleculeView target  = getMolecule(targets, targetIdx);
  const MoleculeView pattern = getMolecule(patterns, patternMolIdx);

  __shared__ LabelMatrixStorage sharedLabelMatrix;
  LabelMatrixView               labelMatrix(&sharedLabelMatrix);

  populateLabelMatrixOptimized<kMaxTargetAtoms, kMaxQueryAtoms>(target, pattern, labelMatrix, nullptr);
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

  constexpr int gsiBuffersPerBlock = 2;
  constexpr int wusBuffersPerBlock = 1;

  if constexpr (Algo == SubstructAlgorithm::GSI) {
    __shared__ PartialMatch gsiPartials[kMaxPartialsPerBlock * 2];

    PartialMatch* blockOverflowA = overflowA + blockIdx.x * gsiBuffersPerBlock * overflowCapacity;
    PartialMatch* blockOverflowB = blockOverflowA + overflowCapacity;

    gsiBFSSearchGPU<kMaxTargetAtoms, kMaxQueryAtoms, SubstructOutputMode::PaintBits>(
      target, pattern, labelMatrix,
      gsiPartials, kMaxPartialsPerBlock,
      blockOverflowA, blockOverflowB, overflowCapacity,
      &sharedMatchCount, &sharedReportedCount,
      nullptr, 0, 0,
      paintParams);

  } else if constexpr (Algo == SubstructAlgorithm::WarpUnified) {
    __shared__ CandidateList wusCandidates[kMaxQueryAtoms];
    __shared__ PartialMatch  wusWorkQueue[kMaxQueueSize];

    PartialMatch* blockOverflow = overflowA + blockIdx.x * wusBuffersPerBlock * overflowCapacity;

    warpUnifiedSearchGPU<kMaxTargetAtoms, kMaxQueryAtoms, SubstructOutputMode::PaintBits>(
      target, pattern, labelMatrix,
      wusCandidates, wusWorkQueue, kMaxQueueSize,
      blockOverflow, overflowCapacity,
      &sharedMatchCount, &sharedReportedCount,
      nullptr, 0, 0,
      paintParams);
  }
}

}  // namespace

// =============================================================================
// RecursivePatternCache Implementation
// =============================================================================

int RecursivePatternCache::getOrAddPattern(int queryIdx, int patternId, const RDKit::ROMol* queryMol) {
  RecursivePatternKey key{queryIdx, patternId};
  auto                it = patternIndexMap.find(key);
  if (it != patternIndexMap.end()) {
    return it->second;
  }

  int molIdx = static_cast<int>(cachedPatterns.numMolecules());
  addQueryToBatch(queryMol, cachedPatterns);
  patternIndexMap[key] = molIdx;
  deviceNeedsUpdate    = true;
  return molIdx;
}

void RecursivePatternCache::syncToDevice(cudaStream_t stream) {
  if (deviceNeedsUpdate) {
    cachedPatternsDevice.copyFromHost(cachedPatterns, stream);
    deviceNeedsUpdate = false;
  }
}

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
  ScopedNvtxRange e2eRange("getSubstructMatches");
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


  // Reusable scratch buffers (avoid alloc/free between kernels)
  RecursiveScratchBuffers recursiveScratch(stream);
  RecursivePatternCache   patternCache(stream);
  AsyncDeviceVector<int>  pairIndicesDev;
  pairIndicesDev.setStream(stream);
  std::vector<int>                 pairIndicesHost(effectiveBatchSize);
  std::vector<BatchedPatternEntry> scratchPatternEntries;

  // Process all pairs in batches
  for (int batchStart = 0; batchStart < numPairs; batchStart += batchSize) {
    ScopedNvtxRange processBatchRange("getSubStructMatches Batch iteration");
    const int batchEnd = std::min(batchStart + batchSize, numPairs);
    const int numPairsInBatch = batchEnd - batchStart;

    // Zero recursive bits for this batch
    results.zeroRecursiveBits();

    // Build pair indices for this batch (contiguous global indices)
    for (int i = 0; i < numPairsInBatch; ++i) {
      pairIndicesHost[i] = batchStart + i;
    }

    // Preprocess all recursive patterns for this batch in a single kernel launch
    preprocessRecursiveSmartsBatched(targetsDevice, targetsHost, queriesHost,
                                     results, numQueries, batchStart, numPairsInBatch,
                                     algorithm, stream, recursiveScratch, patternCache,
                                     scratchPatternEntries);

    // Get view after scratch allocation
    SubstructMatchResultsDeviceView batchView = results.view();

    // Copy pair indices to device (resize reuses memory when possible)
    pairIndicesDev.resize(numPairsInBatch);
    pairIndicesDev.copyFromHost(pairIndicesHost.data(), numPairsInBatch);
    ScopedNvtxRange launchKernelRange("getSubStructMatches Kernel launch");

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

void preprocessRecursiveSmartsBatched(const MoleculesDevice&             targetsDevice,
                                      const MoleculesHost&               targetsHost,
                                      const MoleculesHost&               queriesHost,
                                      const SubstructMatchResultsDevice& outputResults,
                                      const int                          numQueries,
                                      const int                          batchPairOffset,
                                      const int                          batchSize,
                                      const SubstructAlgorithm           algorithm,
                                      cudaStream_t                       stream,
                                      RecursiveScratchBuffers&           scratch,
                                      RecursivePatternCache&             patternCache,
                                      std::vector<BatchedPatternEntry>&  scratchPatternEntries) {
  ScopedNvtxRange processRecursiveRange("Process recursive batch");
  ScopedNvtxRange processRecursiveRangeSetup("Process recursive batch setup");
  // Collect all recursive patterns from queries that have pairs in this batch
  std::vector<BatchedPatternEntry>& patternEntriesHost = scratchPatternEntries;
  patternEntriesHost.clear();

  // Track which queries have pairs in this batch
  std::vector<bool> queryInBatch(numQueries, false);
  for (int i = 0; i < batchSize; ++i) {
    const int pairIdx  = batchPairOffset + i;
    const int queryIdx = pairIdx % numQueries;
    queryInBatch[queryIdx] = true;
  }
  // Collect patterns from relevant queries, using cache for molecule data
  for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
    if (!queryInBatch[queryIdx]) {
      continue;
    }

    if (queryIdx >= static_cast<int>(queriesHost.recursivePatterns.size())) {
      continue;
    }

    const auto& recursiveInfo = queriesHost.recursivePatterns[queryIdx];
    if (recursiveInfo.empty()) {
      continue;
    }

    for (const auto& entry : recursiveInfo.patterns) {
      if (entry.queryMol == nullptr) {
        continue;
      }

      BatchedPatternEntry& batchEntry = patternEntriesHost.emplace_back();
      batchEntry.mainQueryIdx  = queryIdx;
      batchEntry.patternId     = entry.patternId;
      batchEntry.patternMolIdx = patternCache.getOrAddPattern(queryIdx, entry.patternId, entry.queryMol);
    }
  }

  if (patternEntriesHost.empty()) {
    if constexpr (kDebugPaintRecursive) {
      printf("[PreprocessBatched] No recursive patterns to process\n");
    }
    return;
  }
  // Only process targets that are actually in this batch of pairs
  const int firstTargetInBatch = batchPairOffset / numQueries;
  const int lastTargetInBatch  = (batchPairOffset + batchSize - 1) / numQueries;
  const int numTargetsInBatch  = lastTargetInBatch - firstTargetInBatch + 1;

  const size_t numPatterns     = patternEntriesHost.size();
  const size_t totalPaintPairs = static_cast<size_t>(numTargetsInBatch) * numPatterns;

  // Sub-batch if needed (with 8 max recursions per query, typical case is well under batchSize)
  const int maxPaintPairsPerSubBatch = std::max(batchSize, 1024);
  const int patternsPerSubBatch = std::max(1, maxPaintPairsPerSubBatch / numTargetsInBatch);

  if constexpr (kDebugPaintRecursive) {
    printf("[PreprocessBatched] Processing %zu patterns (%zu paint pairs) in sub-batches of %d patterns\n",
           numPatterns, totalPaintPairs, patternsPerSubBatch);
  }

  // Sync cached patterns to device (only copies if new patterns were added)
  patternCache.syncToDevice(stream);

  const auto outputView = outputResults.view();
  constexpr int gsiBuffersPerBlock = 2;
  constexpr int wusBuffersPerBlock = 1;
  processRecursiveRangeSetup.pop();
  // Process patterns in sub-batches (typically just one iteration when few patterns)
  for (size_t patternStart = 0; patternStart < numPatterns; patternStart += patternsPerSubBatch) {
    ScopedNvtxRange processRecursiveRangeSubBatch("Process recursive batch sub-batch");
    const size_t patternEnd            = std::min(patternStart + patternsPerSubBatch, numPatterns);
    const size_t numPatternsInSubBatch = patternEnd - patternStart;
    const size_t numBlocksInSubBatch   = numTargetsInBatch * numPatternsInSubBatch;

    // Copy pattern metadata for this sub-batch to device (only grow buffer if needed)
    if (scratch.patternEntries.size() < numPatternsInSubBatch) {
      scratch.patternEntries.resize(numPatternsInSubBatch);
    }
    scratch.patternEntries.copyFromHost(patternEntriesHost.data() + patternStart, numPatternsInSubBatch);

    // Compute overflow buffer size needed for this sub-batch
    const int buffersPerBlock = (algorithm == SubstructAlgorithm::WarpUnified) ? wusBuffersPerBlock : gsiBuffersPerBlock;
    const size_t overflowNeeded = numBlocksInSubBatch * buffersPerBlock * kOverflowEntriesPerBuffer;

    // Only grow overflow buffer if needed (resize down is a no-op anyway)
    if (scratch.overflow.size() < overflowNeeded) {
      scratch.overflow.resize(overflowNeeded);
    }
    ScopedNvtxRange processRecursiveRangeSubBatchPaint("Process recursive batch sub-batch paint");
    switch (algorithm) {
      case SubstructAlgorithm::VF2:
      case SubstructAlgorithm::GSI: {
        substructPaintKernel<SubstructAlgorithm::GSI><<<numBlocksInSubBatch, threadsPerBlock, 0, stream>>>(
          targetsDevice.view(),
          patternCache.cachedPatternsDevice.view(),
          scratch.patternEntries.data(),
          static_cast<int>(numPatternsInSubBatch),
          outputView.recursiveMatchBits,
          outputView.maxTargetAtoms,
          numQueries,
          0, 0,  // Defaults ignored when patternEntries is non-null
          batchPairOffset,
          batchSize,
          scratch.overflow.data(),
          scratch.overflow.data(),
          kOverflowEntriesPerBuffer,
          firstTargetInBatch);
        break;
      }
      case SubstructAlgorithm::WarpUnified: {
        substructPaintKernel<SubstructAlgorithm::WarpUnified><<<numBlocksInSubBatch, threadsPerBlock, 0, stream>>>(
          targetsDevice.view(),
          patternCache.cachedPatternsDevice.view(),
          scratch.patternEntries.data(),
          static_cast<int>(numPatternsInSubBatch),
          outputView.recursiveMatchBits,
          outputView.maxTargetAtoms,
          numQueries,
          0, 0,  // Defaults ignored when patternEntries is non-null
          batchPairOffset,
          batchSize,
          scratch.overflow.data(),
          scratch.overflow.data(),
          kOverflowEntriesPerBuffer,
          firstTargetInBatch);
        break;
      }
    }
  }

  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit

