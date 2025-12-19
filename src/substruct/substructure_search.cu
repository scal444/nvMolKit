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
#include "substruct_debug.h"
#include "nvtx.h"

namespace nvMolKit {

namespace {

constexpr std::size_t kMaxTargetAtoms = kLabelMaxTargetAtoms;
constexpr std::size_t kMaxQueryAtoms  = kLabelMaxQueryAtoms;

constexpr int threadsPerBlock = 256;

using LabelMatrixView = BitMatrix2DView<kMaxTargetAtoms, kMaxQueryAtoms>;

constexpr int kMaxPartialsPerBlock = 256;   // Shared memory partials per block
constexpr int kMaxQueueSize        = 512;   // Shared memory queue size
constexpr int kWarpsPerBlock       = threadsPerBlock / 32;

/**
 * @brief Compute label matrix and write to global memory.
 *
 * Core logic shared between labelMatrixKernel and labelMatrixPaintKernel.
 * Populates label matrix in shared memory, then copies to global buffer.
 *
 * @param target Target molecule view
 * @param query Query/pattern molecule view
 * @param sharedLabelMatrix Shared memory for label matrix (declared by caller)
 * @param globalOut Output pointer in global memory
 * @param pairRecursiveBits Per-pair recursive match bits, or nullptr
 */
__device__ __forceinline__ void computeLabelMatrixToGlobal(const MoleculeView&   target,
                                                           const MoleculeView&   query,
                                                           LabelMatrixStorage&   sharedLabelMatrix,
                                                           uint32_t*             globalOut,
                                                           const uint32_t*       pairRecursiveBits) {
  LabelMatrixView labelMatrix(&sharedLabelMatrix);

  populateLabelMatrixOptimized<kMaxTargetAtoms, kMaxQueryAtoms>(target, query, labelMatrix, pairRecursiveBits);
  __syncthreads();

  const uint32_t* sharedIn   = sharedLabelMatrix.cbegin();
  const int       tid        = threadIdx.x;
  const int       numThreads = blockDim.x;

  for (std::size_t i = tid; i < kLabelMatrixWords; i += numThreads) {
    globalOut[i] = sharedIn[i];
  }
}

/**
 * @brief Kernel for batch label matrix computation.
 *
 * One block per (target, query) pair. Computes label matrix and writes to global buffer.
 * This separates the labeling phase from the matching phase for better occupancy and reuse.
 */
__global__ void labelMatrixKernel(MoleculesDeviceView targets,
                                  MoleculesDeviceView queries,
                                  const int*          pairIndices,
                                  int                 numQueries,
                                  uint32_t*           labelMatrixBuffer,
                                  const uint32_t*     recursiveMatchBits,
                                  int                 maxTargetAtoms) {
  const int batchLocalIdx = blockIdx.x;
  const int pairIdx       = pairIndices[batchLocalIdx];
  const int targetIdx     = pairIdx / numQueries;
  const int queryIdx      = pairIdx % numQueries;

  if (targetIdx >= targets.numMolecules || queryIdx >= queries.numMolecules) {
    return;
  }

  const MoleculeView target = getMolecule(targets, targetIdx);
  const MoleculeView query  = getMolecule(queries, queryIdx);

  __shared__ LabelMatrixStorage sharedLabelMatrix;

  const uint32_t* pairRecursiveBits = recursiveMatchBits
                                        ? &recursiveMatchBits[batchLocalIdx * maxTargetAtoms]
                                        : nullptr;
  uint32_t* globalOut = labelMatrixBuffer + batchLocalIdx * kLabelMatrixWords;

  computeLabelMatrixToGlobal(target, query, sharedLabelMatrix, globalOut, pairRecursiveBits);
}

/**
 * @brief Kernel for label matrix computation for recursive pattern preprocessing.
 *
 * Block indexing: blockIdx.x = localTargetIdx * numPatterns + localPatternIdx
 */
__global__ void labelMatrixPaintKernel(MoleculesDeviceView        targets,
                                       MoleculesDeviceView        patterns,
                                       const BatchedPatternEntry* patternEntries,
                                       int                        numPatterns,
                                       int                        numQueries,
                                       int                        batchPairOffset,
                                       int                        batchSize,
                                       uint32_t*                  labelMatrixBuffer,
                                       int                        firstTargetIdx,
                                       const uint32_t*            recursiveMatchBits,
                                       int                        maxTargetAtoms) {
  const int localTargetIdx   = blockIdx.x / numPatterns;
  const int targetIdx        = firstTargetIdx + localTargetIdx;
  const int localPatternIdx  = blockIdx.x % numPatterns;

  if (targetIdx >= targets.numMolecules || localPatternIdx >= numPatterns) {
    return;
  }

  const int mainQueryIdx  = patternEntries[localPatternIdx].mainQueryIdx;
  const int patternMolIdx = patternEntries[localPatternIdx].patternMolIdx;

  const int globalPairIdx = targetIdx * numQueries + mainQueryIdx;

  if (globalPairIdx < batchPairOffset || globalPairIdx >= batchPairOffset + batchSize) {
    return;
  }

  const int batchLocalPairIdx = globalPairIdx - batchPairOffset;

  const MoleculeView target  = getMolecule(targets, targetIdx);
  const MoleculeView pattern = getMolecule(patterns, patternMolIdx);

  __shared__ LabelMatrixStorage sharedLabelMatrix;

  uint32_t* globalOut = labelMatrixBuffer + blockIdx.x * kLabelMatrixWords;

  // Pass per-pair recursive bits for evaluating nested patterns' boolean trees
  const uint32_t* pairBits = (recursiveMatchBits != nullptr)
                           ? recursiveMatchBits + batchLocalPairIdx * maxTargetAtoms
                           : nullptr;

  computeLabelMatrixToGlobal(target, pattern, sharedLabelMatrix, globalOut, pairBits);
}

/**
 * @brief Kernel for batch substructure matching.
 *
 * One block per (target, query) pair. Loads pre-computed label matrix from global
 * memory, then dispatches to algorithm-specific search based on template parameter.
 * All result indexing is batch-local.
 *
 * @tparam Algo Algorithm to use for the search phase
 * @param pairIndices Array of global pair indices for this batch
 * @param numQueries Number of queries (for decoding global pair indices)
 */
template <SubstructAlgorithm Algo>
__global__ void substructMatchKernel(MoleculesDeviceView             targets,
                                     MoleculesDeviceView             queries,
                                     SubstructMatchResultsDeviceView results,
                                     const int*                      pairIndices,
                                     int                             numQueries) {
  const int batchLocalIdx = blockIdx.x;
  const int pairIdx       = pairIndices[batchLocalIdx];
  const int targetIdx     = pairIdx / numQueries;
  const int queryIdx      = pairIdx % numQueries;

  if (targetIdx >= targets.numMolecules || queryIdx >= queries.numMolecules) {
    return;
  }

  // Get molecule views
  const MoleculeView target = getMolecule(targets, targetIdx);
  const MoleculeView query  = getMolecule(queries, queryIdx);

  // Shared memory for label matrix
  __shared__ LabelMatrixStorage sharedLabelMatrix;
  LabelMatrixView               labelMatrix(&sharedLabelMatrix);

  // Load label matrix from global memory (pre-computed by labelMatrixKernel)
  const uint32_t* globalIn   = results.getLabelMatrixPtr(batchLocalIdx);
  uint32_t*       sharedOut  = sharedLabelMatrix.begin();
  const int       tid        = threadIdx.x;
  const int       numThreads = blockDim.x;

  for (std::size_t i = tid; i < kLabelMatrixWords; i += numThreads) {
    sharedOut[i] = globalIn[i];
  }
  __syncthreads();

  if constexpr (kDebugDumpLabelMatrix) {
    if (threadIdx.x == 0) {
      printf("[LabelDump] pair=%d (target=%d, query=%d): targetAtoms=%d, queryAtoms=%d\n",
             pairIdx, targetIdx, queryIdx, target.numAtoms, query.numAtoms);
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

  // Get output buffer info for this pair using batch-local indexing
  const int matchOffset = results.pairMatchStarts[batchLocalIdx];
  const int maxMatches  = (results.pairMatchStarts[batchLocalIdx + 1] - matchOffset) / query.numAtoms;

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

  // Write final counts to global memory using batch-local index
  if (threadIdx.x == 0) {
    results.matchCounts[batchLocalIdx]    = sharedMatchCount;
    results.reportedCounts[batchLocalIdx] = sharedReportedCount;
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
 * @param labelMatrixBuffer Pre-computed label matrices [numBlocks * kLabelMatrixWords]
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
                                     const uint32_t*             labelMatrixBuffer,
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

  // Load label matrix from global memory (pre-computed by labelMatrixPaintKernel)
  __shared__ LabelMatrixStorage sharedLabelMatrix;
  LabelMatrixView               labelMatrix(&sharedLabelMatrix);

  const uint32_t* globalIn   = labelMatrixBuffer + blockIdx.x * kLabelMatrixWords;
  uint32_t*       sharedOut  = sharedLabelMatrix.begin();
  const int       tid        = threadIdx.x;
  const int       numThreads = blockDim.x;

  for (std::size_t i = tid; i < kLabelMatrixWords; i += numThreads) {
    sharedOut[i] = globalIn[i];
  }
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

int RecursivePatternCache::getOrAddPattern(int queryIdx, int patternId, const RDKit::ROMol* queryMol,
                                           const RecursivePatternInfo& patternInfo) {
  RecursivePatternKey key{queryIdx, patternId};
  auto                it = patternIndexMap.find(key);
  if (it != patternIndexMap.end()) {
    return it->second;
  }

  int molIdx = static_cast<int>(cachedPatterns.numMolecules());

  // Find children of this pattern (patterns whose parentPatternId matches this patternId)
  // Sort by localIdInParent to ensure correct order for RecursiveMatch instructions
  std::vector<std::pair<int, int>> childrenByLocalId;  // (localIdInParent, patternId)
  for (const auto& p : patternInfo.patterns) {
    if (p.parentPatternId == patternId) {
      childrenByLocalId.emplace_back(p.localIdInParent, p.patternId);
    }
  }
  std::sort(childrenByLocalId.begin(), childrenByLocalId.end());

  std::vector<int> childPatternIds;
  for (const auto& [localId, childId] : childrenByLocalId) {
    childPatternIds.push_back(childId);
  }

  if constexpr (kDebugPaintRecursive) {
    printf("[PatternCache] getOrAddPattern: queryIdx=%d, patternId=%d, found %zu children: [",
           queryIdx, patternId, childPatternIds.size());
    for (size_t i = 0; i < childPatternIds.size(); ++i) {
      printf("%d%s", childPatternIds[i], i + 1 < childPatternIds.size() ? "," : "");
    }
    printf("]\n");
  }

  if (childPatternIds.empty()) {
    addQueryToBatch(queryMol, cachedPatterns);
  } else {
    addQueryToBatch(queryMol, cachedPatterns, childPatternIds);
  }

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
// BatchResultsDevice Implementation
// =============================================================================

void BatchResultsDevice::setStream(cudaStream_t stream) {
  stream_ = stream;
  matchCounts_.setStream(stream);
  reportedCounts_.setStream(stream);
  pairMatchStarts_.setStream(stream);
  matchIndices_.setStream(stream);
  queryAtomCounts_.setStream(stream);
  overflowBuffer_.setStream(stream);
  recursiveMatchBits_.setStream(stream);
  labelMatrixBuffer_.setStream(stream);
}

void BatchResultsDevice::allocateBatch(int                     batchSize,
                                       const std::vector<int>& batchPairMatchStarts,
                                       int                     totalBatchMatchIndices,
                                       int                     numQueries,
                                       int                     maxTargetAtoms,
                                       int                     numBuffersPerBlock) {
  batchSize_              = batchSize;
  numQueries_             = numQueries;
  maxTargetAtoms_         = maxTargetAtoms;
  totalBatchMatchIndices_ = totalBatchMatchIndices;
  overflowBuffersPerBlock_ = numBuffersPerBlock;

  matchCounts_.resize(batchSize);
  matchCounts_.zero();

  reportedCounts_.resize(batchSize);
  reportedCounts_.zero();

  pairMatchStarts_.resize(batchSize + 1);
  pairMatchStarts_.setFromVector(batchPairMatchStarts);

  matchIndices_.resize(totalBatchMatchIndices);
  matchIndices_.zero();

  const int overflowEntries = batchSize * numBuffersPerBlock * kOverflowEntriesPerBuffer;
  overflowBuffer_.resize(overflowEntries);
  overflowBuffer_.zero();

  const size_t recursiveBitsSize = static_cast<size_t>(batchSize) * maxTargetAtoms;
  recursiveMatchBits_.resize(recursiveBitsSize);
  recursiveMatchBits_.zero();

  const size_t labelMatrixSize = static_cast<size_t>(batchSize) * kLabelMatrixWords;
  labelMatrixBuffer_.resize(labelMatrixSize);
  labelMatrixBuffer_.zero();
}

void BatchResultsDevice::setQueryAtomCounts(const std::vector<int>& queryAtomCounts) {
  queryAtomCounts_.setFromVector(queryAtomCounts);
}

SubstructMatchResultsDeviceView BatchResultsDevice::view() const {
  SubstructMatchResultsDeviceView v;
  v.matchCounts              = matchCounts_.data();
  v.reportedCounts           = reportedCounts_.data();
  v.pairMatchStarts          = pairMatchStarts_.data();
  v.matchIndices             = matchIndices_.data();
  v.numQueries               = numQueries_;
  v.queryAtomCounts          = queryAtomCounts_.data();
  v.overflowBuffer           = overflowBuffer_.data();
  v.overflowEntriesPerBuffer = kOverflowEntriesPerBuffer;
  v.overflowBuffersPerBlock  = overflowBuffersPerBlock_;
  v.recursiveMatchBits       = recursiveMatchBits_.data();
  v.maxTargetAtoms           = maxTargetAtoms_;
  v.labelMatrixBuffer        = labelMatrixBuffer_.data();
  return v;
}

void BatchResultsDevice::zeroRecursiveBits() {
  recursiveMatchBits_.zero();
}

void BatchResultsDevice::zeroLabelMatrixBuffer() {
  labelMatrixBuffer_.zero();
}

void BatchResultsDevice::copyBatchToHost(std::vector<int>&     hostMatchCounts,
                                         std::vector<int>&     hostReportedCounts,
                                         std::vector<int16_t>& hostMatchIndices) const {
  hostMatchCounts.resize(batchSize_);
  hostReportedCounts.resize(batchSize_);
  hostMatchIndices.resize(totalBatchMatchIndices_);

  matchCounts_.copyToHost(hostMatchCounts);
  reportedCounts_.copyToHost(hostReportedCounts);
  matchIndices_.copyToHost(hostMatchIndices);
}

// =============================================================================
// Batch Processing Helper
// =============================================================================

struct BatchContext {
  RecursiveScratchBuffers          recursiveScratch;
  RecursivePatternCache            patternCache;
  BatchResultsDevice               batchResults;
  AsyncDeviceVector<int>           pairIndicesDev;
  std::vector<int>                 pairIndicesHost;
  std::vector<BatchedPatternEntry> scratchPatternEntries;
  std::vector<int>                 batchMatchCounts;
  std::vector<int>                 batchReportedCounts;
  std::vector<int16_t>             batchMatchIndices;

  explicit BatchContext(cudaStream_t stream, int maxBatchSize)
      : recursiveScratch(stream), patternCache(stream), batchResults(stream), pairIndicesHost(maxBatchSize) {
    pairIndicesDev.setStream(stream);
  }
};

void processSingleBatch(MoleculesDevice&             targetsDevice,
                        const MoleculesDevice&       queriesDevice,
                        const MoleculesHost&         targetsHost,
                        const MoleculesHost&         queriesHost,
                        const std::vector<int>&      queryAtomCounts,
                        const std::vector<int>&      globalPairMatchStarts,
                        int                          numQueries,
                        int                          batchStart,
                        int                          numPairsInBatch,
                        int                          maxTargetAtoms,
                        SubstructAlgorithm           algorithm,
                        cudaStream_t                 stream,
                        BatchContext&                ctx,
                        SubstructMatchResultsHost&   accumulatedResults) {
  ScopedNvtxRange processBatchRange("processSingleBatch");

  std::vector<int> batchPairMatchStarts(numPairsInBatch + 1);
  batchPairMatchStarts[0] = 0;
  for (int i = 0; i < numPairsInBatch; ++i) {
    const int globalPairIdx = batchStart + i;
    const int pairCapacity = globalPairMatchStarts[globalPairIdx + 1] - globalPairMatchStarts[globalPairIdx];
    batchPairMatchStarts[i + 1] = batchPairMatchStarts[i] + pairCapacity;
  }
  const int totalBatchMatchIndices = batchPairMatchStarts.back();

  const int numBuffersPerBlock = (algorithm == SubstructAlgorithm::GSI) ? 2 : 1;
  ctx.batchResults.allocateBatch(numPairsInBatch, batchPairMatchStarts, totalBatchMatchIndices,
                                  numQueries, maxTargetAtoms, numBuffersPerBlock);
  ctx.batchResults.setQueryAtomCounts(queryAtomCounts);
  ctx.batchResults.zeroRecursiveBits();

  for (int i = 0; i < numPairsInBatch; ++i) {
    ctx.pairIndicesHost[i] = batchStart + i;
  }

  ctx.pairIndicesDev.resize(numPairsInBatch);
  ctx.pairIndicesDev.copyFromHost(ctx.pairIndicesHost.data(), numPairsInBatch);

  cudaCheckError(cudaStreamSynchronize(stream));

  preprocessRecursiveSmartsBatched(targetsDevice, targetsHost, queriesHost,
                                   ctx.batchResults, numQueries, batchStart, numPairsInBatch,
                                   algorithm, stream, ctx.recursiveScratch, ctx.patternCache,
                                   ctx.scratchPatternEntries);

  SubstructMatchResultsDeviceView batchView = ctx.batchResults.view();

  {
    ScopedNvtxRange labelKernelRange("LabelMatrix Kernel");
    labelMatrixKernel<<<numPairsInBatch, threadsPerBlock, 0, stream>>>(
      targetsDevice.view(),
      queriesDevice.view(),
      ctx.pairIndicesDev.data(),
      numQueries,
      batchView.labelMatrixBuffer,
      batchView.recursiveMatchBits,
      batchView.maxTargetAtoms);
  }

  {
    ScopedNvtxRange launchKernelRange("Match Kernel");
    switch (algorithm) {
      case SubstructAlgorithm::VF2:
        substructMatchKernel<SubstructAlgorithm::VF2><<<numPairsInBatch, threadsPerBlock, 0, stream>>>(
          targetsDevice.view(), queriesDevice.view(), batchView, ctx.pairIndicesDev.data(), numQueries);
        break;
      case SubstructAlgorithm::GSI:
        substructMatchKernel<SubstructAlgorithm::GSI><<<numPairsInBatch, threadsPerBlock, 0, stream>>>(
          targetsDevice.view(), queriesDevice.view(), batchView, ctx.pairIndicesDev.data(), numQueries);
        break;
      case SubstructAlgorithm::WarpUnified:
        substructMatchKernel<SubstructAlgorithm::WarpUnified><<<numPairsInBatch, threadsPerBlock, 0, stream>>>(
          targetsDevice.view(), queriesDevice.view(), batchView, ctx.pairIndicesDev.data(), numQueries);
        break;
    }
  }

  cudaCheckError(cudaStreamSynchronize(stream));

  ctx.batchResults.copyBatchToHost(ctx.batchMatchCounts, ctx.batchReportedCounts, ctx.batchMatchIndices);
  cudaCheckError(cudaStreamSynchronize(stream));

  for (int i = 0; i < numPairsInBatch; ++i) {
    const int globalPairIdx = batchStart + i;
    accumulatedResults.matchCounts[globalPairIdx]    = ctx.batchMatchCounts[i];
    accumulatedResults.reportedCounts[globalPairIdx] = ctx.batchReportedCounts[i];

    const int batchLocalOffset  = batchPairMatchStarts[i];
    const int globalOffset      = globalPairMatchStarts[globalPairIdx];
    const int numIndicesToCopy  = batchPairMatchStarts[i + 1] - batchLocalOffset;

    for (int j = 0; j < numIndicesToCopy; ++j) {
      accumulatedResults.matchIndices[globalOffset + j] = ctx.batchMatchIndices[batchLocalOffset + j];
    }
  }
}

// =============================================================================
// Main API
// =============================================================================

void getSubstructMatches(MoleculesDevice&           targetsDevice,
                         const MoleculesDevice&     queriesDevice,
                         const MoleculesHost&       targetsHost,
                         const MoleculesHost&       queriesHost,
                         SubstructMatchResultsHost& results,
                         SubstructAlgorithm         algorithm,
                         cudaStream_t               stream,
                         int                        batchSize) {
  ScopedNvtxRange e2eRange("getSubstructMatches");
  const int numTargets = static_cast<int>(targetsHost.numMolecules());
  const int numQueries = static_cast<int>(queriesHost.numMolecules());

  if (numTargets == 0 || numQueries == 0) {
    throw std::invalid_argument("Target and query batches must not be empty");
  }

  std::vector<int> queryAtomCounts(numQueries);
  for (int q = 0; q < numQueries; ++q) {
    const int atomStart = queriesHost.batchAtomStarts[q];
    const int atomEnd   = queriesHost.batchAtomStarts[q + 1];
    queryAtomCounts[q]  = atomEnd - atomStart;
  }

  std::vector<int> maxMatchesPerTarget(numTargets);
  int maxTargetAtoms = 0;
  for (int t = 0; t < numTargets; ++t) {
    const int atomStart    = targetsHost.batchAtomStarts[t];
    const int atomEnd      = targetsHost.batchAtomStarts[t + 1];
    maxMatchesPerTarget[t] = atomEnd - atomStart;
    maxTargetAtoms = std::max(maxTargetAtoms, maxMatchesPerTarget[t]);
  }

  const int numPairs = numTargets * numQueries;

  std::vector<int> globalPairMatchStarts(numPairs + 1);
  globalPairMatchStarts[0] = 0;
  for (int t = 0; t < numTargets; ++t) {
    for (int q = 0; q < numQueries; ++q) {
      const int pairIdx      = t * numQueries + q;
      const int maxMatches   = maxMatchesPerTarget[t];
      const int queryAtoms   = queryAtomCounts[q];
      const int pairCapacity = maxMatches * queryAtoms;
      globalPairMatchStarts[pairIdx + 1] = globalPairMatchStarts[pairIdx] + pairCapacity;
    }
  }
  const int totalMatchIndices = globalPairMatchStarts.back();

  results.numTargets = numTargets;
  results.numQueries = numQueries;
  results.matchCounts.resize(numPairs);
  results.reportedCounts.resize(numPairs);
  results.pairMatchStarts = globalPairMatchStarts;
  results.matchIndices.resize(totalMatchIndices);

  const int effectiveBatchSize = std::min(batchSize, numPairs);
  BatchContext ctx(stream, effectiveBatchSize);

  for (int batchStart = 0; batchStart < numPairs; batchStart += batchSize) {
    const int batchEnd        = std::min(batchStart + batchSize, numPairs);
    const int numPairsInBatch = batchEnd - batchStart;

    processSingleBatch(targetsDevice, queriesDevice, targetsHost, queriesHost,
                       queryAtomCounts, globalPairMatchStarts,
                       numQueries, batchStart, numPairsInBatch, maxTargetAtoms,
                       algorithm, stream, ctx, results);
  }

  cudaCheckError(cudaGetLastError());
}

void getSubstructMatches(MoleculesDevice&        targetsDevice,
                         const MoleculesDevice&  queriesDevice,
                         const MoleculesHost&    targetsHost,
                         const MoleculesHost&    queriesHost,
                         SubstructSearchResults& results,
                         SubstructAlgorithm      algorithm,
                         cudaStream_t            stream,
                         int                     batchSize) {
  ScopedNvtxRange e2eRange("getSubstructMatches (accumulated)");
  const int numTargets = static_cast<int>(targetsHost.numMolecules());
  const int numQueries = static_cast<int>(queriesHost.numMolecules());

  if (numTargets == 0 || numQueries == 0) {
    throw std::invalid_argument("Target and query batches must not be empty");
  }

  SubstructMatchResultsHost resultsHost;
  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      resultsHost, algorithm, stream, batchSize);

  std::vector<int> queryAtomCounts(numQueries);
  for (int q = 0; q < numQueries; ++q) {
    const int atomStart = queriesHost.batchAtomStarts[q];
    const int atomEnd   = queriesHost.batchAtomStarts[q + 1];
    queryAtomCounts[q]  = atomEnd - atomStart;
  }

  results.resize(numTargets, numQueries);

  for (int t = 0; t < numTargets; ++t) {
    for (int q = 0; q < numQueries; ++q) {
      const int pairIdx       = t * numQueries + q;
      const int numQueryAtoms = queryAtomCounts[q];
      const int reported      = resultsHost.reportedCounts[pairIdx];
      const int matchStart    = resultsHost.pairMatchStarts[pairIdx];

      results.overflowed[t][q] = (resultsHost.matchCounts[pairIdx] > reported) ? 1 : 0;

      auto& matchVec = results.matches[t][q];
      matchVec.reserve(reported);
      for (int m = 0; m < reported; ++m) {
        std::vector<int> match(numQueryAtoms);
        for (int a = 0; a < numQueryAtoms; ++a) {
          match[a] = resultsHost.matchIndices[matchStart + m * numQueryAtoms + a];
        }
        matchVec.push_back(std::move(match));
      }
    }
  }
}

// =============================================================================
// Recursive SMARTS Preprocessing
// =============================================================================

void preprocessRecursiveSmartsBatched(const MoleculesDevice&            targetsDevice,
                                      const MoleculesHost&              targetsHost,
                                      const MoleculesHost&              queriesHost,
                                      BatchResultsDevice&               batchResults,
                                      const int                         numQueries,
                                      const int                         batchPairOffset,
                                      const int                         batchSize,
                                      const SubstructAlgorithm          algorithm,
                                      cudaStream_t                      stream,
                                      RecursiveScratchBuffers&          scratch,
                                      RecursivePatternCache&            patternCache,
                                      std::vector<BatchedPatternEntry>& scratchPatternEntries) {
  ScopedNvtxRange processRecursiveRange("Process recursive batch");
  ScopedNvtxRange processRecursiveRangeSetup("Process recursive batch setup");

  std::vector<BatchedPatternEntry>& patternEntriesHost = scratchPatternEntries;
  patternEntriesHost.clear();

  std::vector<bool> queryInBatch(numQueries, false);
  for (int i = 0; i < batchSize; ++i) {
    const int pairIdx  = batchPairOffset + i;
    const int queryIdx = pairIdx % numQueries;
    queryInBatch[queryIdx] = true;
  }

  int maxDepth = 0;
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

    maxDepth = std::max(maxDepth, recursiveInfo.maxDepth);

    for (const auto& entry : recursiveInfo.patterns) {
      if (entry.queryMol == nullptr) {
        continue;
      }

      BatchedPatternEntry& batchEntry = patternEntriesHost.emplace_back();
      batchEntry.mainQueryIdx    = queryIdx;
      batchEntry.patternId       = entry.patternId;
      batchEntry.patternMolIdx   = patternCache.getOrAddPattern(queryIdx, entry.patternId, entry.queryMol, recursiveInfo);
      batchEntry.depth           = entry.depth;
      batchEntry.localIdInParent = entry.localIdInParent;
    }
  }

  if (patternEntriesHost.empty()) {
    if constexpr (kDebugPaintRecursive) {
      printf("[PreprocessBatched] No recursive patterns to process\n");
    }
    return;
  }

  if constexpr (kDebugPaintRecursive) {
    printf("[PreprocessBatched] All patterns collected:\n");
    for (const auto& p : patternEntriesHost) {
      printf("[PreprocessBatched]   mainQueryIdx=%d, patternId=%d, patternMolIdx=%d, depth=%d, localIdInParent=%d\n",
             p.mainQueryIdx, p.patternId, p.patternMolIdx, p.depth, p.localIdInParent);
    }
  }

  const int firstTargetInBatch = batchPairOffset / numQueries;
  const int lastTargetInBatch  = (batchPairOffset + batchSize - 1) / numQueries;
  const int numTargetsInBatch  = lastTargetInBatch - firstTargetInBatch + 1;

  patternCache.syncToDevice(stream);

  const auto batchView = batchResults.view();
  constexpr int gsiBuffersPerBlock = 2;
  constexpr int wusBuffersPerBlock = 1;

  const int maxPaintPairsPerSubBatch = std::max(batchSize, 1024);
  processRecursiveRangeSetup.pop();

  // For nested patterns, process level by level (depth 0 first, then 1, etc.)
  // Depth 0 patterns are leaves (no children), higher depths have children at lower depths
  for (int currentDepth = 0; currentDepth <= maxDepth; ++currentDepth) {
    ScopedNvtxRange depthRange("Process recursive depth level");

    // Filter patterns at current depth
    std::vector<BatchedPatternEntry> patternsAtDepth;
    for (const auto& entry : patternEntriesHost) {
      if (entry.depth == currentDepth) {
        patternsAtDepth.push_back(entry);
      }
    }

    if (patternsAtDepth.empty()) {
      continue;
    }

    const size_t numPatterns = patternsAtDepth.size();
    const int patternsPerSubBatch = std::max(1, maxPaintPairsPerSubBatch / numTargetsInBatch);

    if constexpr (kDebugPaintRecursive) {
      printf("[PreprocessBatched] Depth %d: Processing %zu patterns\n", currentDepth, numPatterns);
      for (const auto& p : patternsAtDepth) {
        printf("[PreprocessBatched]   patternId=%d, patternMolIdx=%d, localIdInParent=%d\n",
               p.patternId, p.patternMolIdx, p.localIdInParent);
      }
    }

    for (size_t patternStart = 0; patternStart < numPatterns; patternStart += patternsPerSubBatch) {
      ScopedNvtxRange processRecursiveRangeSubBatch("Process recursive batch sub-batch");
      const size_t patternEnd            = std::min(patternStart + patternsPerSubBatch, numPatterns);
      const size_t numPatternsInSubBatch = patternEnd - patternStart;
      const size_t numBlocksInSubBatch   = numTargetsInBatch * numPatternsInSubBatch;

      if (scratch.patternEntries.size() < numPatternsInSubBatch) {
        scratch.patternEntries.resize(numPatternsInSubBatch);
      }
      scratch.patternEntries.copyFromHost(patternsAtDepth.data() + patternStart, numPatternsInSubBatch);

      const int buffersPerBlock = (algorithm == SubstructAlgorithm::WarpUnified) ? wusBuffersPerBlock : gsiBuffersPerBlock;
      const size_t overflowNeeded = numBlocksInSubBatch * buffersPerBlock * kOverflowEntriesPerBuffer;

      if (scratch.overflow.size() < overflowNeeded) {
        scratch.overflow.resize(overflowNeeded);
      }
      scratch.overflow.zero();

      const size_t labelMatrixNeeded = numBlocksInSubBatch * kLabelMatrixWords;
      if (scratch.labelMatrixBuffer.size() < labelMatrixNeeded) {
        scratch.labelMatrixBuffer.resize(labelMatrixNeeded);
      }
      scratch.labelMatrixBuffer.zero();

      // For depth > 0 patterns, pass recursive bits from previous levels so their
      // boolean trees can check child pattern results
      const uint32_t* recursiveBitsForLabel = (currentDepth > 0) ? batchView.recursiveMatchBits : nullptr;

      {
        ScopedNvtxRange processRecursiveRangeSubBatchLabel("Process recursive batch sub-batch label");
        labelMatrixPaintKernel<<<numBlocksInSubBatch, threadsPerBlock, 0, stream>>>(
          targetsDevice.view(),
          patternCache.cachedPatternsDevice.view(),
          scratch.patternEntries.data(),
          static_cast<int>(numPatternsInSubBatch),
          numQueries,
          batchPairOffset,
          batchSize,
          scratch.labelMatrixBuffer.data(),
          firstTargetInBatch,
          recursiveBitsForLabel,
          batchView.maxTargetAtoms);
      }

      {
        ScopedNvtxRange processRecursiveRangeSubBatchPaint("Process recursive batch sub-batch paint");
        switch (algorithm) {
          case SubstructAlgorithm::VF2:
          case SubstructAlgorithm::GSI: {
            substructPaintKernel<SubstructAlgorithm::GSI><<<numBlocksInSubBatch, threadsPerBlock, 0, stream>>>(
              targetsDevice.view(),
              patternCache.cachedPatternsDevice.view(),
              scratch.patternEntries.data(),
              static_cast<int>(numPatternsInSubBatch),
              batchView.recursiveMatchBits,
              batchView.maxTargetAtoms,
              numQueries,
              0, 0,
              batchPairOffset,
              batchSize,
              scratch.overflow.data(),
              scratch.overflow.data(),
              kOverflowEntriesPerBuffer,
              scratch.labelMatrixBuffer.data(),
              firstTargetInBatch);
            break;
          }
          case SubstructAlgorithm::WarpUnified: {
            substructPaintKernel<SubstructAlgorithm::WarpUnified><<<numBlocksInSubBatch, threadsPerBlock, 0, stream>>>(
              targetsDevice.view(),
              patternCache.cachedPatternsDevice.view(),
              scratch.patternEntries.data(),
              static_cast<int>(numPatternsInSubBatch),
              batchView.recursiveMatchBits,
              batchView.maxTargetAtoms,
              numQueries,
              0, 0,
              batchPairOffset,
              batchSize,
              scratch.overflow.data(),
              scratch.overflow.data(),
              kOverflowEntriesPerBuffer,
              scratch.labelMatrixBuffer.data(),
              firstTargetInBatch);
            break;
          }
        }
      }
    }
  }

  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit

