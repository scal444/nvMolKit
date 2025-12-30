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
#include <array>
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "cuda_error_check.h"
#include "host_vector.h"
#include "global_pool.cuh"
#include "graph_labeler.cuh"
#include "molecules_device.cuh"
#include "pinned_buffer_pool.h"
#include "sm_shared_mem_config.cuh"
#include "substruct_algos.cuh"
#include "substruct_debug.h"
#include "nvtx.h"

namespace nvMolKit {

namespace {

constexpr std::size_t kMaxTargetAtoms = kLabelMaxTargetAtoms;
constexpr std::size_t kMaxQueryAtoms  = kLabelMaxQueryAtoms;    

using LabelMatrixView = BitMatrix2DView<kMaxTargetAtoms, kMaxQueryAtoms>;

// =============================================================================
// Architecture tables for compile-time shared memory sizing
// =============================================================================

/// Shared memory per SM in KiB for each compute capability
constexpr int getSharedMemPerSM_KiB(int sm) {
  if (sm >= 120) return 128;   // SM 12.0+
  if (sm >= 100) return 228;   // SM 10.0+ (Blackwell)
  if (sm >= 90)  return 228;   // SM 9.0+ (Hopper)
  if (sm == 80)  return 160;   // SM 8.0 (Ampere A100)
  return 100;                  // SM 8.6/8.9 (Ada), default
}

/// Max threads per SM for each compute capability
constexpr int getMaxThreadsPerSM(int sm) {
  if (sm >= 90)  return 2048;  // Hopper+
  if (sm == 80)  return 2048;  // A100
  if (sm >= 86)  return 1536;  // Ada/consumer Ampere
  return 1536;                 // Default
}

/// Compute max blocks per SM given block size
constexpr int getMaxBlocksPerSM(int sm, int blockSize) {
  return getMaxThreadsPerSM(sm) / blockSize;
}

/// Compute max partials that fit in shared memory budget
constexpr int computeMaxPartials(int sharedPerSM_KiB, int blocksPerSM) {
  constexpr int kLabelMatrixBytes = 1024;
  constexpr int kControlVarsBytes = 32;
  constexpr int kPartialMatchSize = sizeof(PartialMatch);
  static_assert(kPartialMatchSize == 64, "PartialMatch size changed - update shared memory calculations");
  
  const int budgetBytes = (sharedPerSM_KiB * 1024) / blocksPerSM;
  const int availableBytes = (budgetBytes * 9 / 10) - kLabelMatrixBytes - kControlVarsBytes;
  const int rawPartials = availableBytes / (kPartialMatchSize * 2);  // ping-pong
  return (rawPartials / 10) * 10;  // round to 10
}

/// Compute partials for a given SM architecture
constexpr int getMaxPartialsForSM(int sm, int blockSize) {
  return computeMaxPartials(getSharedMemPerSM_KiB(sm), getMaxBlocksPerSM(sm, blockSize));
}

// Compute at compile time based on __CUDA_ARCH__
#if defined(__CUDA_ARCH__)
constexpr int kMaxPartialsPerBlock = getMaxPartialsForSM(__CUDA_ARCH__ / 10, kThreadsPerBlock);
static_assert(getMaxThreadsPerSM(__CUDA_ARCH__ / 10) % kThreadsPerBlock == 0, 
              "kThreadsPerBlock must evenly divide max threads/SM");
#else
constexpr int kMaxPartialsPerBlock = getMaxPartialsForSM(86, kThreadsPerBlock);
#endif

constexpr int kMaxPartialsPerBlockHost = getMaxPartialsForSM(86, kThreadsPerBlock);
static_assert(getMaxThreadsPerSM(86) % kThreadsPerBlock == 0,
              "kThreadsPerBlock must evenly divide max threads/SM");
constexpr int kMaxQueueSize = kMaxPartialsPerBlockHost * 2;
constexpr int kWarpsPerBlock           = kThreadsPerBlock / 32;

/**
 * @brief Configure kernel to use maximum shared memory carveout.
 *
 * On Hopper+ architectures, shared memory is configurable via carveout.
 * This sets the kernel to prefer maximum shared memory over L1 cache.
 */
template <typename KernelFunc>
void configureSharedMemCarveout(KernelFunc kernel) {
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
}

/// Flag to ensure we only configure carveout once
inline bool& sharedMemCarveoutConfigured() {
  static bool configured = false;
  return configured;
}

/// Configure all substruct kernels for max shared memory (call once before first use)
void configureSubstructKernelsSharedMem();

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
 *
 * @param batchLocalIndices Optional mapping from blockIdx.x to batch-local index (for split launches).
 *                          If nullptr, uses blockIdx.x directly.
 */
__global__ void labelMatrixKernel(MoleculesDeviceView targets,
                                  MoleculesDeviceView queries,
                                  const int*          pairIndices,
                                  int                 numQueries,
                                  uint32_t*           labelMatrixBuffer,
                                  const uint32_t*     recursiveMatchBits,
                                  int                 maxTargetAtoms,
                                  const int*          batchLocalIndices = nullptr) {
  const int launchIdx     = blockIdx.x;
  const int batchLocalIdx = batchLocalIndices ? batchLocalIndices[launchIdx] : launchIdx;
  const int pairIdx       = pairIndices[launchIdx];
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
 * @param pairIndices Array of global pair indices for this launch
 * @param numQueries Number of queries (for decoding global pair indices)
 * @param batchLocalIndices Optional mapping from blockIdx.x to batch-local index (for split launches).
 *                          If nullptr, uses blockIdx.x directly.
 */
template <SubstructAlgorithm Algo>
__global__ void substructMatchKernel(MoleculesDeviceView             targets,
                                     MoleculesDeviceView             queries,
                                     SubstructMatchResultsDeviceView results,
                                     const int*                      pairIndices,
                                     int                             numQueries,
                                     const int*                      batchLocalIndices = nullptr,
                                     DeviceTimingsData*              timings = nullptr) {
  const int launchIdx     = blockIdx.x;
  const int batchLocalIdx = batchLocalIndices ? batchLocalIndices[launchIdx] : launchIdx;
  const int pairIdx       = pairIndices[launchIdx];
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
      vf2States[warpId].init();
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
                                                     matchOffset,
                                                     {},
                                                     timings);

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

// =============================================================================
// Shared Memory Carveout Configuration
// =============================================================================

void configureSubstructKernelsSharedMem() {
  if (sharedMemCarveoutConfigured()) return;
  
  // Configure GSI kernels for max shared memory
  configureSharedMemCarveout(substructMatchKernel<SubstructAlgorithm::GSI>);
  configureSharedMemCarveout(substructPaintKernel<SubstructAlgorithm::GSI>);
  
  // Configure WUS kernels
  // configureSharedMemCarveout(substructMatchKernel<SubstructAlgorithm::WarpUnified>);
  // configureSharedMemCarveout(substructPaintKernel<SubstructAlgorithm::WarpUnified>);
  
  // VF2 uses less shared memory but configure anyway
  // configureSharedMemCarveout(substructMatchKernel<SubstructAlgorithm::VF2>);
  
  sharedMemCarveoutConfigured() = true;
}

}  // anonymous namespace

// =============================================================================
// Pipelined Batch Processing Types (internal, but needs external linkage for forward decl)
// =============================================================================

struct BatchSlot {
  int batchStart        = 0;
  int numPairsInBatch   = 0;
  int totalMatchIndices = 0;

  // Pointers into consolidated pinned buffer (not owned)
  int*     pairIndicesHost          = nullptr;
  int*     batchPairMatchStarts     = nullptr;
  int*     matchCountsHost          = nullptr;
  int*     reportedCountsHost       = nullptr;
  int16_t* matchIndicesHost         = nullptr;


  // Precomputed recursive batch setup (populated by prepareRecursiveBatchOnCPU)
  int recursiveMaxDepth       = 0;
  int firstTargetInBatch      = 0;
  int numTargetsInBatch       = 0;
  std::array<std::vector<BatchedPatternEntry>, kMaxRecursionDepth + 1> patternsAtDepth;

  // Streams and events declared first so they're destroyed last (after resources that use them)
  ScopedStream                              computeStream;
  ScopedCudaEvent                           copyDoneEvent;
  ScopedCudaEvent                           allocDoneEvent;
  std::unique_ptr<TwoStreamPipelineContext> twoStreamCtx;
  RecursiveScratchBuffers                   recursiveScratch;
  
  BatchResultsDevice     deviceResults;
  AsyncDeviceVector<int> pairIndicesDev;

  explicit BatchSlot(int workerIdx) 
      : computeStream(("worker" + std::to_string(workerIdx) + "_mainStream").c_str()),
        recursiveScratch(nullptr) {
    twoStreamCtx = std::make_unique<TwoStreamPipelineContext>(workerIdx);
  }

  void initializeForStream() {
    cudaStream_t s = computeStream.stream();
    cudaStream_t recStream = twoStreamCtx->recursiveStream.stream();
    deviceResults.setStream(s);
    pairIndicesDev.setStream(s);
    recursiveScratch.setStream(recStream);
  }

  cudaStream_t stream() const { return computeStream.stream(); }

  /**
   * @brief Bind pointers from consolidated pinned buffer.
   *
   * Must be called before processing batches. The consolidated buffer
   * must outlive the BatchSlot.
   */
  void bindPinnedBuffer(ConsolidatedPinnedBuffer& buffer) {
    pairIndicesHost      = buffer.pairIndices;
    batchPairMatchStarts = buffer.batchPairMatchStarts;
    matchCountsHost      = buffer.matchCounts;
    reportedCountsHost   = buffer.reportedCounts;
    matchIndicesHost     = buffer.matchIndices;

    twoStreamCtx->setPinnedBuffers(buffer.matchGlobalPairIndicesHost,
                                   buffer.matchBatchLocalIndicesHost,
                                   buffer.perDepthCapacity);

    recursiveScratch.setPinnedBuffer(buffer.patternsAtDepthHost, buffer.patternsCapacity);
  }
};

namespace {

struct ThreadWorkerContext {
  PinnedHostVector<int> queryAtomCounts;
  std::vector<int> globalPairMatchStarts;
  std::vector<int> queryDepths;       ///< Cached recursion depth for each query
  std::vector<int> queryMaxDepths;    ///< Cached max recursion depth per query (from leafSubpatterns)
  std::vector<int8_t> queryHasPatterns;  ///< Whether query has any recursive patterns
  int numTargets     = 0;
  int numQueries     = 0;
  int maxTargetAtoms = 0;
};

}  // anonymous namespace

// =============================================================================
// LeafSubpatterns Implementation
// =============================================================================

void LeafSubpatterns::buildAllPatterns(const MoleculesHost& queriesHost) {
  ScopedNvtxRange buildRange("LeafSubpatterns::buildAllPatterns");

  const int numQueries = static_cast<int>(queriesHost.numMolecules());

  // First pass: build pattern molecules and register in patternIndexMap
  for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
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

      LeafSubpatternKey key{queryIdx, entry.patternId};
      if (patternIndexMap.find(key) != patternIndexMap.end()) {
        continue;
      }

      int molIdx = static_cast<int>(patternsHost.numMolecules());

      std::vector<std::pair<int, int>> childrenByLocalId;
      for (const auto& p : recursiveInfo.patterns) {
        if (p.parentPatternId == entry.patternId) {
          childrenByLocalId.emplace_back(p.localIdInParent, p.patternId);
        }
      }
      std::sort(childrenByLocalId.begin(), childrenByLocalId.end());

      std::vector<int> childPatternIds;
      for (const auto& [localId, childId] : childrenByLocalId) {
        childPatternIds.push_back(childId);
      }

      if constexpr (kDebugPaintRecursive) {
        printf("[LeafSubpatterns] buildAllPatterns: queryIdx=%d, patternId=%d, found %zu children: [",
               queryIdx, entry.patternId, childPatternIds.size());
        for (size_t i = 0; i < childPatternIds.size(); ++i) {
          printf("%d%s", childPatternIds[i], i + 1 < childPatternIds.size() ? "," : "");
        }
        printf("]\n");
      }

      if (childPatternIds.empty()) {
        addQueryToBatch(entry.queryMol, patternsHost);
      } else {
        addQueryToBatch(entry.queryMol, patternsHost, childPatternIds);
      }

      patternIndexMap[key] = molIdx;
    }
  }

  // Second pass: build precomputed BatchedPatternEntry structures
  perQueryPatterns.resize(numQueries);
  perQueryMaxDepth.resize(numQueries, 0);

  for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
    if (queryIdx >= static_cast<int>(queriesHost.recursivePatterns.size())) {
      continue;
    }

    const auto& recursiveInfo = queriesHost.recursivePatterns[queryIdx];
    if (recursiveInfo.empty()) {
      continue;
    }

    perQueryMaxDepth[queryIdx] = recursiveInfo.maxDepth;

    for (const auto& entry : recursiveInfo.patterns) {
      if (entry.queryMol == nullptr) {
        continue;
      }

      const int patternMolIdx = getPatternIndex(queryIdx, entry.patternId);
      if (patternMolIdx < 0) {
        continue;
      }

      BatchedPatternEntry batchEntry;
      batchEntry.mainQueryIdx    = queryIdx;
      batchEntry.patternId       = entry.patternId;
      batchEntry.patternMolIdx   = patternMolIdx;
      batchEntry.depth           = entry.depth;
      batchEntry.localIdInParent = entry.localIdInParent;

      perQueryPatterns[queryIdx][entry.depth].push_back(batchEntry);
    }
  }
}

void LeafSubpatterns::syncToDevice(cudaStream_t stream) {
  ScopedNvtxRange syncRange("LeafSubpatterns::syncToDevice");
  
  if (!patternsHost.numMolecules()) {
    return;
  }
  patternsDevice.copyFromHost(patternsHost, stream);
}

// =============================================================================
// TwoStreamPipelineContext Implementation
// =============================================================================

namespace {
std::pair<int, int> getStreamPriorityRange() {
  int leastPriority    = 0;
  int greatestPriority = 0;
  cudaCheckError(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  return {greatestPriority, leastPriority};
}
}  // namespace

TwoStreamPipelineContext::TwoStreamPipelineContext(int workerIdx)
    : recursiveStream(getStreamPriorityRange().first, 
                      ("worker" + std::to_string(workerIdx) + "_priorityRecursiveStream").c_str()),
      matchStreams{
          ScopedStreamWithPriority(getStreamPriorityRange().second, 
                                   ("worker" + std::to_string(workerIdx) + "_depth1FinalLabelStream").c_str()),
          ScopedStreamWithPriority(getStreamPriorityRange().second, 
                                   ("worker" + std::to_string(workerIdx) + "_depth2FinalLabelStream").c_str()),
          ScopedStreamWithPriority(getStreamPriorityRange().second, 
                                   ("worker" + std::to_string(workerIdx) + "_depth3FinalLabelStream").c_str()),
          ScopedStreamWithPriority(getStreamPriorityRange().second, 
                                   ("worker" + std::to_string(workerIdx) + "_depth4FinalLabelStream").c_str())} {}

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

void BatchResultsDevice::allocateBatch(int        batchSize,
                                       const int* batchPairMatchStarts,
                                       int        totalBatchMatchIndices,
                                       int        numQueries,
                                       int        maxTargetAtoms,
                                       int        numBuffersPerBlock) {
  ScopedNvtxRange allocRange("BatchResultsDevice::allocateBatch");
  
  batchSize_              = batchSize;
  numQueries_             = numQueries;
  maxTargetAtoms_         = maxTargetAtoms;
  totalBatchMatchIndices_ = totalBatchMatchIndices;
  overflowBuffersPerBlock_ = numBuffersPerBlock;

  if (matchCounts_.size() < static_cast<size_t>(batchSize)) {
    matchCounts_.resize(static_cast<size_t>(batchSize * 1.5));
  }

  if (reportedCounts_.size() < static_cast<size_t>(batchSize)) {
    reportedCounts_.resize(static_cast<size_t>(batchSize * 1.5));
  }

  if (pairMatchStarts_.size() < static_cast<size_t>(batchSize + 1)) {
    pairMatchStarts_.resize(static_cast<size_t>((batchSize + 1) * 1.5));
  }
  pairMatchStarts_.copyFromHost(batchPairMatchStarts, batchSize + 1);

  if (matchIndices_.size() < static_cast<size_t>(totalBatchMatchIndices)) {
    matchIndices_.resize(static_cast<size_t>(totalBatchMatchIndices * 1.5));
  }

  const int overflowEntries = batchSize * numBuffersPerBlock * kOverflowEntriesPerBuffer;
  if (overflowBuffer_.size() < static_cast<size_t>(overflowEntries)) {
    overflowBuffer_.resize(static_cast<size_t>(overflowEntries * 1.5));
  }

  const size_t recursiveBitsSize = static_cast<size_t>(batchSize) * maxTargetAtoms;
  if (recursiveMatchBits_.size() < recursiveBitsSize) {
    recursiveMatchBits_.resize(static_cast<size_t>(recursiveBitsSize * 1.5));
  }
  recursiveMatchBits_.zero();

  const size_t labelMatrixSize = static_cast<size_t>(batchSize) * kLabelMatrixWords;
  if (labelMatrixBuffer_.size() < labelMatrixSize) {
    labelMatrixBuffer_.resize(static_cast<size_t>(labelMatrixSize * 1.5));
  }
}

void BatchResultsDevice::setQueryAtomCounts(const int* queryAtomCounts, size_t count) {
  if (queryAtomCounts_.size() < count) {
    queryAtomCounts_.resize(static_cast<size_t>(count * 1.5));
  }
  queryAtomCounts_.copyFromHost(queryAtomCounts, count);
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

void BatchResultsDevice::copyBatchToHost(int*     hostMatchCounts,
                                         int*     hostReportedCounts,
                                         int16_t* hostMatchIndices) const {
  matchCounts_.copyToHost(hostMatchCounts, batchSize_);
  reportedCounts_.copyToHost(hostReportedCounts, batchSize_);
  matchIndices_.copyToHost(hostMatchIndices, totalBatchMatchIndices_);
}

// =============================================================================
// Pipelined Batch Processing Implementation
// =============================================================================

namespace {

/**
 * @brief Determine the recursion depth for a query (number of paint rounds needed).
 *
 * @param queriesHost Host-side query data
 * @param queryIdx Query index
 * @return 0 if no recursive patterns, otherwise maxDepth + 1
 */
int getQueryRecursionDepth(const MoleculesHost& queriesHost, int queryIdx) {
  if (queryIdx >= static_cast<int>(queriesHost.recursivePatterns.size())) {
    return 0;
  }
  const auto& recursiveInfo = queriesHost.recursivePatterns[queryIdx];
  if (recursiveInfo.empty()) {
    return 0;
  }
  return recursiveInfo.maxDepth + 1;
}

/**
 * @brief Precompute the two-stream pipeline schedule for a batch.
 *
 * Groups pairs by their query's recursion depth and populates the host-side
 * index vectors for the recursive stream and match stream.
 *
 * @param pipelineCtx Pipeline context to populate
 * @param ctx Worker context with cached query depths
 * @param numPairsInBatch Number of pairs in the batch
 * @param batchStart Global pair index where the batch starts
 */
void precomputePipelineSchedule(TwoStreamPipelineContext&  pipelineCtx,
                                const ThreadWorkerContext& ctx,
                                int                        numPairsInBatch,
                                int                        batchStart) {
  ScopedNvtxRange scheduleRange("CPU: precomputePipelineSchedule");
  pipelineCtx.maxDepthInBatch = 0;

  for (auto& vec : pipelineCtx.matchPairsHost) {
    vec.clear();
  }

  for (int i = 0; i < numPairsInBatch; ++i) {
    const int queryIdx = (batchStart + i) % ctx.numQueries;
    const int depth    = ctx.queryDepths[queryIdx];

    pipelineCtx.matchPairsHost[depth].push_back(i);
    pipelineCtx.maxDepthInBatch = std::max(pipelineCtx.maxDepthInBatch, depth);
  }
}

void prepareRecursiveBatchOnCPU(BatchSlot&                 slot,
                                const ThreadWorkerContext& ctx,
                                const LeafSubpatterns&     leafSubpatterns) {
  ScopedNvtxRange prepRecRange("prepareRecursiveBatchOnCPU");

  precomputePipelineSchedule(*slot.twoStreamCtx, ctx, slot.numPairsInBatch, slot.batchStart);

  for (auto& vec : slot.patternsAtDepth) {
    vec.clear();
  }

  const int firstQueryInBatch = slot.batchStart % ctx.numQueries;
  const int numUniqueQueries  = std::min(slot.numPairsInBatch, ctx.numQueries);

  slot.recursiveMaxDepth = 0;
  for (int i = 0; i < numUniqueQueries; ++i) {
    const int queryIdx = (firstQueryInBatch + i) % ctx.numQueries;

    if (!ctx.queryHasPatterns[queryIdx]) {
      continue;
    }

    const int queryMaxDepth = ctx.queryMaxDepths[queryIdx];
    slot.recursiveMaxDepth = std::max(slot.recursiveMaxDepth, queryMaxDepth);

    for (int d = 0; d <= queryMaxDepth; ++d) {
      const auto& srcEntries = leafSubpatterns.perQueryPatterns[queryIdx][d];
      auto& destEntries = slot.patternsAtDepth[d];
      destEntries.insert(destEntries.end(), srcEntries.begin(), srcEntries.end());
    }
  }

  slot.firstTargetInBatch = slot.batchStart / ctx.numQueries;
  const int lastTargetInBatch = (slot.batchStart + slot.numPairsInBatch - 1) / ctx.numQueries;
  slot.numTargetsInBatch = lastTargetInBatch - slot.firstTargetInBatch + 1;
}

void prepareBatchOnCPU(BatchSlot&                   slot,
                       const ThreadWorkerContext&   ctx,
                       const MoleculesHost&         queriesHost,
                       const LeafSubpatterns&       leafSubpatterns,
                       int                          batchStart,
                       int                          maxPairsInBatch) {
  ScopedNvtxRange prepRange("prepareBatchOnCPU");

  const int numPairs = ctx.numTargets * ctx.numQueries;
  const int batchEnd = std::min(batchStart + maxPairsInBatch, numPairs);
  const int numPairsInBatch = batchEnd - batchStart;

  slot.batchStart       = batchStart;
  slot.numPairsInBatch  = numPairsInBatch;

  slot.batchPairMatchStarts[0] = 0;
  for (int i = 0; i < numPairsInBatch; ++i) {
    const int globalPairIdx = batchStart + i;
    const int pairCapacity = ctx.globalPairMatchStarts[globalPairIdx + 1] -
                             ctx.globalPairMatchStarts[globalPairIdx];
    slot.batchPairMatchStarts[i + 1] = slot.batchPairMatchStarts[i] + pairCapacity;
  }
  slot.totalMatchIndices = slot.batchPairMatchStarts[numPairsInBatch];

  for (int i = 0; i < numPairsInBatch; ++i) {
    slot.pairIndicesHost[i] = batchStart + i;
  }

  prepareRecursiveBatchOnCPU(slot, ctx, leafSubpatterns);
}

/**
 * @brief Launch label matrix and match kernels for a subset of pairs.
 */
void launchLabelAndMatch(const std::vector<int>&      batchLocalIndices,
                         BatchSlot&                   slot,
                         const ThreadWorkerContext&   ctx,
                         MoleculesDevice&             targetsDevice,
                         const MoleculesDevice&       queriesDevice,
                         SubstructAlgorithm           algorithm,
                         cudaStream_t                 stream,
                         TwoStreamPipelineContext&    twoStreamCtx,
                         int                          depthGroupIdx) {
  ScopedNvtxRange launchRange("launchLabelAndMatch depth=" + std::to_string(depthGroupIdx));
  
  if (batchLocalIndices.empty()) {
    return;
  }

  const int numPairsInGroup = static_cast<int>(batchLocalIndices.size());

  int* globalPairIndicesHost = twoStreamCtx.matchGlobalPairIndicesHost[depthGroupIdx];
  int* batchLocalIndicesHostPtr = twoStreamCtx.matchBatchLocalIndicesHost[depthGroupIdx];
  
  ScopedNvtxRange prepareRange("CPU: Prepare host index arrays");

  for (int i = 0; i < numPairsInGroup; ++i) {
    globalPairIndicesHost[i] = slot.pairIndicesHost[batchLocalIndices[i]];
    batchLocalIndicesHostPtr[i] = batchLocalIndices[i];
  }
  prepareRange.pop();

  auto& globalPairIndicesDev = twoStreamCtx.matchGlobalPairIndices[depthGroupIdx];
  auto& batchLocalIndicesDev = twoStreamCtx.matchBatchLocalIndices[depthGroupIdx];

  globalPairIndicesDev.setStream(stream);
  if (globalPairIndicesDev.size() < static_cast<size_t>(numPairsInGroup)) {
    globalPairIndicesDev.resize(static_cast<size_t>(numPairsInGroup * 1.5));
  }
  globalPairIndicesDev.copyFromHost(globalPairIndicesHost, numPairsInGroup);

  batchLocalIndicesDev.setStream(stream);
  if (batchLocalIndicesDev.size() < static_cast<size_t>(numPairsInGroup)) {
    batchLocalIndicesDev.resize(static_cast<size_t>(numPairsInGroup * 1.5));
  }
  batchLocalIndicesDev.copyFromHost(batchLocalIndicesHostPtr, numPairsInGroup);

  SubstructMatchResultsDeviceView batchView = slot.deviceResults.view();

  labelMatrixKernel<<<numPairsInGroup, kThreadsPerBlock, 0, stream>>>(
    targetsDevice.view(),
    queriesDevice.view(),
    globalPairIndicesDev.data(),
    ctx.numQueries,
    batchView.labelMatrixBuffer,
    batchView.recursiveMatchBits,
    batchView.maxTargetAtoms,
    batchLocalIndicesDev.data());

  switch (algorithm) {
    // case SubstructAlgorithm::VF2:
    //   substructMatchKernel<SubstructAlgorithm::VF2><<<numPairsInGroup, kThreadsPerBlock, 0, stream>>>(
    //     targetsDevice.view(), queriesDevice.view(), batchView, globalPairIndicesDev.data(), ctx.numQueries,
    //     batchLocalIndicesDev.data());
    //   break;
    case SubstructAlgorithm::GSI:
    case SubstructAlgorithm::VF2:
    case SubstructAlgorithm::WarpUnified:

      substructMatchKernel<SubstructAlgorithm::GSI><<<numPairsInGroup, kThreadsPerBlock, 0, stream>>>(
        targetsDevice.view(), queriesDevice.view(), batchView, globalPairIndicesDev.data(), ctx.numQueries,
        batchLocalIndicesDev.data());
      break;
    // case SubstructAlgorithm::WarpUnified:
    //   substructMatchKernel<SubstructAlgorithm::WarpUnified><<<numPairsInGroup, kThreadsPerBlock, 0, stream>>>(
    //     targetsDevice.view(), queriesDevice.view(), batchView, globalPairIndicesDev.data(), ctx.numQueries,
    //     batchLocalIndicesDev.data());
    //   break;
  }
}

void launchRecursivePaintKernels(
    const MoleculesDevice&                                                   targetsDevice,
    const LeafSubpatterns&                                                   leafSubpatterns,
    BatchResultsDevice&                                                      batchResults,
    int                                                                      numQueries,
    int                                                                      batchPairOffset,
    int                                                                      batchSize,
    SubstructAlgorithm                                                       algorithm,
    cudaStream_t                                                             stream,
    RecursiveScratchBuffers&                                                 scratch,
    const std::array<std::vector<BatchedPatternEntry>, kMaxRecursionDepth + 1>& patternsAtDepth,
    int                                                                      maxDepth,
    int                                                                      firstTargetInBatch,
    int                                                                      numTargetsInBatch,
    cudaEvent_t*                                                             depthEvents,
    int                                                                      numDepthEvents) {
  ScopedNvtxRange processRecursiveRange("launchRecursivePaintKernels");

  scratch.setStream(stream);

  const auto batchView = batchResults.view();
  constexpr int gsiBuffersPerBlock = 2;
  constexpr int wusBuffersPerBlock = 1;

  const int maxPaintPairsPerSubBatch = std::max(batchSize, 1024);

  for (int currentDepth = 0; currentDepth <= maxDepth; ++currentDepth) {
    ScopedNvtxRange depthRange("Process recursive depth level " + std::to_string(currentDepth));

    const auto& patternsForDepth = patternsAtDepth[currentDepth];

    if (patternsForDepth.empty()) {
      if (currentDepth < numDepthEvents && depthEvents != nullptr) {
        cudaCheckError(cudaEventRecord(depthEvents[currentDepth], stream));
      }
      continue;
    }

    const size_t numPatterns = patternsForDepth.size();
    const int patternsPerSubBatch = std::max(1, maxPaintPairsPerSubBatch / numTargetsInBatch);

    for (size_t patternStart = 0; patternStart < numPatterns; patternStart += patternsPerSubBatch) {
      ScopedNvtxRange subBatchRange("Process sub-batch " + std::to_string(patternStart));
      
      const size_t patternEnd            = std::min(patternStart + patternsPerSubBatch, numPatterns);
      const size_t numPatternsInSubBatch = patternEnd - patternStart;
      const size_t numBlocksInSubBatch   = numTargetsInBatch * numPatternsInSubBatch;

      ScopedNvtxRange prepareRange("GPU: Upload pattern entries");
      const int bufferIdx = scratch.acquireBufferIndex();
      scratch.waitForBuffer(bufferIdx);
      scratch.ensureCapacity(bufferIdx, static_cast<int>(numPatternsInSubBatch));
      for (size_t i = 0; i < numPatternsInSubBatch; ++i) {
        scratch.patternsAtDepthHost[bufferIdx][i] = patternsForDepth[patternStart + i];
      }
      prepareRange.pop();

      const int buffersPerBlock = (algorithm == SubstructAlgorithm::WarpUnified) ? wusBuffersPerBlock : gsiBuffersPerBlock;
      const size_t overflowNeeded = numBlocksInSubBatch * buffersPerBlock * kOverflowEntriesPerBuffer;

      if (scratch.overflow.size() < overflowNeeded) {
        scratch.overflow.resize(static_cast<size_t>(overflowNeeded * 1.5));
      }

      const size_t labelMatrixNeeded = numBlocksInSubBatch * kLabelMatrixWords;
      if (scratch.labelMatrixBuffer.size() < labelMatrixNeeded) {
        scratch.labelMatrixBuffer.resize(static_cast<size_t>(labelMatrixNeeded * 1.5));
      }

      if (scratch.patternEntries.size() < numPatternsInSubBatch) {
        scratch.patternEntries.resize(static_cast<size_t>(numPatternsInSubBatch * 1.5));
      }
      
      scratch.patternEntries.copyFromHost(scratch.patternsAtDepthHost[bufferIdx], numPatternsInSubBatch);
      scratch.recordCopy(bufferIdx, scratch.patternEntries.stream());

      const uint32_t* recursiveBitsForLabel = (currentDepth > 0) ? batchView.recursiveMatchBits : nullptr;

      labelMatrixPaintKernel<<<numBlocksInSubBatch, kThreadsPerBlock, 0, stream>>>(
        targetsDevice.view(),
        leafSubpatterns.view(),
        scratch.patternEntries.data(),
        static_cast<int>(numPatternsInSubBatch),
        numQueries,
        batchPairOffset,
        batchSize,
        scratch.labelMatrixBuffer.data(),
        firstTargetInBatch,
        recursiveBitsForLabel,
        batchView.maxTargetAtoms);

      switch (algorithm) {
        case SubstructAlgorithm::VF2:
        case SubstructAlgorithm::GSI:
        case SubstructAlgorithm::WarpUnified: {
          substructPaintKernel<SubstructAlgorithm::GSI><<<numBlocksInSubBatch, kThreadsPerBlock, 0, stream>>>(
            targetsDevice.view(),
            leafSubpatterns.view(),
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
        // case SubstructAlgorithm::WarpUnified: {
        //   substructPaintKernel<SubstructAlgorithm::WarpUnified><<<numBlocksInSubBatch, kThreadsPerBlock, 0, stream>>>(
        //     targetsDevice.view(),
        //     leafSubpatterns.view(),
        //     scratch.patternEntries.data(),
        //     static_cast<int>(numPatternsInSubBatch),
        //     batchView.recursiveMatchBits,
        //     batchView.maxTargetAtoms,
        //     numQueries,
        //     0, 0,
        //     batchPairOffset,
        //     batchSize,
        //     scratch.overflow.data(),
        //     scratch.overflow.data(),
        //     kOverflowEntriesPerBuffer,
        //     scratch.labelMatrixBuffer.data(),
        //     firstTargetInBatch);
        //   break;
        // }
      }
    }

    if (currentDepth < numDepthEvents && depthEvents != nullptr) {
      cudaCheckError(cudaEventRecord(depthEvents[currentDepth], stream));
    }
  }

  cudaCheckError(cudaGetLastError());
}

void uploadAndLaunchBatch(BatchSlot&                 slot,
                          const ThreadWorkerContext& ctx,
                          MoleculesDevice&           targetsDevice,
                          const MoleculesDevice&     queriesDevice,
                          const LeafSubpatterns&     leafSubpatterns,
                          SubstructAlgorithm         algorithm) {
  ScopedNvtxRange uploadRange("uploadAndLaunchBatch");

  cudaStream_t slotStream = slot.stream();

  TwoStreamPipelineContext& twoStreamCtx = *slot.twoStreamCtx;
  const int numBuffersPerBlock = (algorithm == SubstructAlgorithm::GSI) ? 2 : 1;

  if (twoStreamCtx.maxDepthInBatch == 0) {
    ScopedNvtxRange nonRecursiveRange("Non-recursive path");
    
    slot.deviceResults.allocateBatch(slot.numPairsInBatch,
                                     slot.batchPairMatchStarts,
                                     slot.totalMatchIndices,
                                     ctx.numQueries,
                                     ctx.maxTargetAtoms,
                                     numBuffersPerBlock);
    slot.deviceResults.setQueryAtomCounts(ctx.queryAtomCounts.data(), ctx.numQueries);

    if (slot.pairIndicesDev.size() < static_cast<size_t>(slot.numPairsInBatch)) {
      slot.pairIndicesDev.resize(static_cast<size_t>(slot.numPairsInBatch * 1.5));
    }
    slot.pairIndicesDev.copyFromHost(slot.pairIndicesHost, slot.numPairsInBatch);

    SubstructMatchResultsDeviceView batchView = slot.deviceResults.view();

    labelMatrixKernel<<<slot.numPairsInBatch, kThreadsPerBlock, 0, slotStream>>>(
      targetsDevice.view(),
      queriesDevice.view(),
      slot.pairIndicesDev.data(),
      ctx.numQueries,
      batchView.labelMatrixBuffer,
      batchView.recursiveMatchBits,
      batchView.maxTargetAtoms);

    switch (algorithm) {
      // case SubstructAlgorithm::VF2:
      //   substructMatchKernel<SubstructAlgorithm::VF2><<<slot.numPairsInBatch, kThreadsPerBlock, 0, slotStream>>>(
      //     targetsDevice.view(), queriesDevice.view(), batchView, slot.pairIndicesDev.data(), ctx.numQueries);
      //   break;
        case SubstructAlgorithm::VF2:
        case SubstructAlgorithm::GSI:
        case SubstructAlgorithm::WarpUnified:
        substructMatchKernel<SubstructAlgorithm::GSI><<<slot.numPairsInBatch, kThreadsPerBlock, 0, slotStream>>>(
          targetsDevice.view(), queriesDevice.view(), batchView, slot.pairIndicesDev.data(), ctx.numQueries);
        break;
      // case SubstructAlgorithm::WarpUnified:
      //   substructMatchKernel<SubstructAlgorithm::WarpUnified><<<slot.numPairsInBatch, kThreadsPerBlock, 0, slotStream>>>(
      //     targetsDevice.view(), queriesDevice.view(), batchView, slot.pairIndicesDev.data(), ctx.numQueries);
      //   break;
    }
    return;
  }

  ScopedNvtxRange twoStreamRange("Multi-stream recursive pipeline");

  cudaStream_t recursiveStream = twoStreamCtx.recursiveStream.stream();

  slot.deviceResults.allocateBatch(slot.numPairsInBatch,
                                   slot.batchPairMatchStarts,
                                   slot.totalMatchIndices,
                                   ctx.numQueries,
                                   ctx.maxTargetAtoms,
                                   numBuffersPerBlock);
  slot.deviceResults.setQueryAtomCounts(ctx.queryAtomCounts.data(), ctx.numQueries);

  cudaCheckError(cudaEventRecord(slot.allocDoneEvent.event(), slotStream));
  
  ScopedNvtxRange waitAllocRange("Wait: recursiveStream waits for alloc");
  cudaCheckError(cudaStreamWaitEvent(recursiveStream, slot.allocDoneEvent.event(), 0));
  waitAllocRange.pop();

  std::array<cudaEvent_t, kMaxRecursionDepth> depthEventPtrs;
  for (int i = 0; i < kMaxRecursionDepth; ++i) {
    depthEventPtrs[i] = twoStreamCtx.depthEvents[i].event();
  }

  ScopedNvtxRange preprocRange("launchRecursivePaintKernels (recursiveStream)");
  launchRecursivePaintKernels(targetsDevice, leafSubpatterns,
                              slot.deviceResults, ctx.numQueries,
                              slot.batchStart, slot.numPairsInBatch,
                              algorithm, recursiveStream,
                              slot.recursiveScratch,
                              slot.patternsAtDepth,
                              slot.recursiveMaxDepth,
                              slot.firstTargetInBatch,
                              slot.numTargetsInBatch,
                              depthEventPtrs.data(),
                              kMaxRecursionDepth);
  preprocRange.pop();

  ScopedNvtxRange depth0Range("Match depth-0 pairs (slotStream)");
  launchLabelAndMatch(twoStreamCtx.matchPairsHost[0], slot, ctx, targetsDevice, queriesDevice,
                      algorithm, slotStream, twoStreamCtx, 0);
  depth0Range.pop();

  for (int depth = 1; depth <= twoStreamCtx.maxDepthInBatch; ++depth) {
    ScopedNvtxRange depthRange("Match depth-" + std::to_string(depth) + " pairs (matchStream " +
                               std::to_string(depth - 1) + ")");

    cudaStream_t depthStream = twoStreamCtx.matchStreams[depth - 1].stream();

    ScopedNvtxRange waitRange("Wait: matchStream waits for alloc + depth events");
    cudaCheckError(cudaStreamWaitEvent(depthStream, slot.allocDoneEvent.event(), 0));
    cudaCheckError(cudaStreamWaitEvent(depthStream, depthEventPtrs[depth - 1], 0));
    waitRange.pop();

    launchLabelAndMatch(twoStreamCtx.matchPairsHost[depth], slot, ctx, targetsDevice, queriesDevice,
                        algorithm, depthStream, twoStreamCtx, depth);

    cudaCheckError(cudaEventRecord(twoStreamCtx.matchDoneEvents[depth - 1].event(), depthStream));
  }

  cudaCheckError(cudaEventRecord(twoStreamCtx.recursiveDoneEvent.event(), recursiveStream));
  cudaCheckError(cudaStreamWaitEvent(slotStream, twoStreamCtx.recursiveDoneEvent.event(), 0));
  for (int depth = 1; depth <= twoStreamCtx.maxDepthInBatch; ++depth) {
    cudaCheckError(cudaStreamWaitEvent(slotStream, twoStreamCtx.matchDoneEvents[depth - 1].event(), 0));
  }
}

void initiateResultsCopyToHost(BatchSlot& slot) {
  ScopedNvtxRange copyRange("initiateResultsCopyToHost");
  slot.deviceResults.copyBatchToHost(slot.matchCountsHost, slot.reportedCountsHost, slot.matchIndicesHost);
  cudaCheckError(cudaEventRecord(slot.copyDoneEvent.event(), slot.stream()));
}

void accumulateBatchResults(BatchSlot&                 slot,
                            const ThreadWorkerContext& ctx,
                            SubstructSearchResults&    results,
                            std::mutex&                resultsMutex) {
  ScopedNvtxRange accumRange("accumulateBatchResults (dynamic)");

  ScopedNvtxRange waitRange("Wait for D2H copy");
  cudaCheckError(cudaEventSynchronize(slot.copyDoneEvent.event()));
  waitRange.pop();

  ScopedNvtxRange processRange("Process batch results");
  for (int i = 0; i < slot.numPairsInBatch; ++i) {
    const int globalPairIdx   = slot.batchStart + i;
    const int targetIdx       = globalPairIdx / ctx.numQueries;
    const int queryIdx        = globalPairIdx % ctx.numQueries;
    const int queryAtoms      = ctx.queryAtomCounts[queryIdx];
    const int actualMatches   = slot.matchCountsHost[i];
    const int reportedMatches = slot.reportedCountsHost[i];

    if (reportedMatches > 0) {
      const int batchLocalOffset = slot.batchPairMatchStarts[i];

      std::vector<std::vector<int>> pairMatches;
      pairMatches.reserve(reportedMatches);

      for (int m = 0; m < reportedMatches; ++m) {
        std::vector<int> match(queryAtoms);
        for (int a = 0; a < queryAtoms; ++a) {
          match[a] = slot.matchIndicesHost[batchLocalOffset + m * queryAtoms + a];
        }
        pairMatches.push_back(std::move(match));
      }

      std::lock_guard<std::mutex> lock(resultsMutex);
      auto& targetMatches = results.getMatchesMut(targetIdx, queryIdx);
      targetMatches.insert(targetMatches.end(),
                           std::make_move_iterator(pairMatches.begin()),
                           std::make_move_iterator(pairMatches.end()));
      results.addActualCount(targetIdx, queryIdx, actualMatches);
    } else if (actualMatches > 0) {
      std::lock_guard<std::mutex> lock(resultsMutex);
      results.addActualCount(targetIdx, queryIdx, actualMatches);
    }
  }
  processRange.pop();
}

/**
 * @brief Unified runner worker that handles both inline and queued preprocessing modes.
 *
 * When readyQueue is nullptr: acquires slots from pool, preprocesses inline (legacy mode).
 * When readyQueue is non-null: dequeues preprocessed slots from the queue.
 */
void runnerWorker(int                        workerIdx,
                  const ThreadWorkerContext& ctx,
                  MoleculesDevice&           targetsDevice,
                  const MoleculesDevice&     queriesDevice,
                  const MoleculesHost&       queriesHost,
                  const LeafSubpatterns&     leafSubpatterns,
                  SubstructSearchResults&    results,
                  std::mutex&                resultsMutex,
                  SubstructAlgorithm         algorithm,
                  cudaEvent_t                upstreamReadyEvent,
                  std::atomic<int>&          nextBatchIdx,
                  int                        totalNumBatches,
                  int                        effectiveBatchSize,
                  BatchSlotPool&             slotPool,
                  PreparedBatchQueue*        readyQueue,
                  std::atomic<bool>&         shutdownFlag,
                  std::exception_ptr&        exceptionPtr) {
  try {
    ScopedNvtxRange workerRange("runnerWorker " + std::to_string(workerIdx));

    const int numPairs = ctx.numTargets * ctx.numQueries;

    BatchSlot* pendingSlot = nullptr;

    while (true) {
      BatchSlot* slot;
      
      if (readyQueue) {
        ScopedNvtxRange waitRange("Wait: dequeue from preprocessor", NvtxColor::kRed);
        slot = readyQueue->dequeue();
        waitRange.pop();
        if (!slot) break;
      } else {
        const int batchIdx = nextBatchIdx.fetch_add(1, std::memory_order_relaxed);
        if (batchIdx >= totalNumBatches) break;

        const int batchStart = batchIdx * effectiveBatchSize;
        if (batchStart >= numPairs) break;

        {
          ScopedNvtxRange waitRange("Wait: acquire slot from pool");
          slot = slotPool.acquire();
        }
        if (!slot) break;

        if (upstreamReadyEvent != nullptr) {
          cudaCheckError(cudaStreamWaitEvent(slot->stream(), upstreamReadyEvent, 0));
          cudaCheckError(cudaStreamWaitEvent(slot->twoStreamCtx->recursiveStream.stream(), upstreamReadyEvent, 0));
          for (int depth = 0; depth < kMaxRecursionDepth; ++depth) {
            cudaCheckError(cudaStreamWaitEvent(slot->twoStreamCtx->matchStreams[depth].stream(), upstreamReadyEvent, 0));
          }
        }

        ScopedNvtxRange prepRange("CPU prep batch " + std::to_string(batchIdx));
        prepareBatchOnCPU(*slot, ctx, queriesHost, leafSubpatterns, batchStart, effectiveBatchSize);
        prepRange.pop();
      }

      ScopedNvtxRange launchRange("GPU launch batch " + std::to_string(slot->batchStart / effectiveBatchSize));
      uploadAndLaunchBatch(*slot, ctx, targetsDevice, queriesDevice, leafSubpatterns, algorithm);
      initiateResultsCopyToHost(*slot);
      launchRange.pop();

      if (pendingSlot) {
        ScopedNvtxRange accumRange("Accumulate batch");
        accumulateBatchResults(*pendingSlot, ctx, results, resultsMutex);
        slotPool.release(pendingSlot);
        accumRange.pop();
      }

      pendingSlot = slot;
    }

    if (pendingSlot) {
      ScopedNvtxRange accumRange("Accumulate final batch");
      accumulateBatchResults(*pendingSlot, ctx, results, resultsMutex);
      slotPool.release(pendingSlot);
      accumRange.pop();
    }
  } catch (...) {
    exceptionPtr = std::current_exception();
    // Signal shutdown to unblock other threads waiting on pool/queue
    shutdownFlag.store(true, std::memory_order_release);
    slotPool.shutdown();
    if (readyQueue) readyQueue->shutdown();
  }
}

/**
 * @brief Preprocessor worker that prepares batches and enqueues them for runners.
 *
 * Acquires slots from the pool, runs prepareBatchOnCPU, then enqueues to readyQueue.
 * Runners will dequeue and execute the GPU work.
 */
void preprocessorWorker(int                        workerIdx,
                        const ThreadWorkerContext& ctx,
                        const MoleculesHost&       queriesHost,
                        const LeafSubpatterns&     leafSubpatterns,
                        cudaEvent_t                upstreamReadyEvent,
                        std::atomic<int>&          nextBatchIdx,
                        int                        totalNumBatches,
                        int                        effectiveBatchSize,
                        BatchSlotPool&             slotPool,
                        PreparedBatchQueue&        readyQueue,
                        std::atomic<bool>&         shutdownFlag,
                        std::exception_ptr&        exceptionPtr) {
  try {
    ScopedNvtxRange workerRange("preprocessorWorker " + std::to_string(workerIdx));

    const int numPairs = ctx.numTargets * ctx.numQueries;

    while (true) {
      const int batchIdx = nextBatchIdx.fetch_add(1, std::memory_order_relaxed);
      if (batchIdx >= totalNumBatches) break;

      const int batchStart = batchIdx * effectiveBatchSize;
      if (batchStart >= numPairs) break;

      BatchSlot* slot;
      {
        ScopedNvtxRange waitRange("Wait: acquire slot from pool");
        slot = slotPool.acquire();
      }
      if (!slot) break;

      if (upstreamReadyEvent != nullptr) {
        cudaCheckError(cudaStreamWaitEvent(slot->stream(), upstreamReadyEvent, 0));
        cudaCheckError(cudaStreamWaitEvent(slot->twoStreamCtx->recursiveStream.stream(), upstreamReadyEvent, 0));
        for (int depth = 0; depth < kMaxRecursionDepth; ++depth) {
          cudaCheckError(cudaStreamWaitEvent(slot->twoStreamCtx->matchStreams[depth].stream(), upstreamReadyEvent, 0));
        }
      }

      ScopedNvtxRange prepRange("CPU prep batch " + std::to_string(batchIdx));
      prepareBatchOnCPU(*slot, ctx, queriesHost, leafSubpatterns, batchStart, effectiveBatchSize);
      prepRange.pop();

      {
        ScopedNvtxRange waitRange("Wait: enqueue to runner", NvtxColor::kRed);
        readyQueue.enqueue(slot);
      }
    }
  } catch (...) {
    exceptionPtr = std::current_exception();
    // Signal shutdown to unblock other threads waiting on pool/queue
    shutdownFlag.store(true, std::memory_order_release);
    slotPool.shutdown();
    readyQueue.shutdown();
  }
}

}  // namespace

// =============================================================================
// Main API
// =============================================================================

void getSubstructMatches(MoleculesDevice&           targetsDevice,
                         const MoleculesDevice&     queriesDevice,
                         const MoleculesHost&       targetsHost,
                         const MoleculesHost&       queriesHost,
                         const LeafSubpatterns&     leafSubpatterns,
                         SubstructSearchResults&    results,
                         SubstructAlgorithm         algorithm,
                         cudaStream_t               stream,
                         int                        batchSize,
                         int                        requestedNumRunners,
                         int                        requestedNumPreprocessors) {
  ScopedNvtxRange e2eRange("getSubstructMatches");
  
  configureSubstructKernelsSharedMem();
  
  const int numTargets = static_cast<int>(targetsHost.numMolecules());
  const int numQueries = static_cast<int>(queriesHost.numMolecules());

  if (numTargets == 0 || numQueries == 0) {
    throw std::invalid_argument("Target and query batches must not be empty");
  }

  const int numPairs           = numTargets * numQueries;
  const int effectiveBatchSize = std::min(batchSize, numPairs);
  const int totalNumBatches    = (numPairs + effectiveBatchSize - 1) / effectiveBatchSize;

  const int numRunners       = std::min(std::max(1, requestedNumRunners), totalNumBatches);
  const int numPreprocessors = std::min(std::max(0, requestedNumPreprocessors), totalNumBatches);

  ScopedNvtxRange ctxRange("CPU: ThreadWorkerContext construction");
  ThreadWorkerContext ctx;
  ctxRange.pop();

  ctx.numTargets = numTargets;
  ctx.numQueries = numQueries;

  ScopedNvtxRange metadataRange("CPU: Compute batch metadata");
  ctx.queryAtomCounts.resize(static_cast<size_t>(numQueries * 1.5));
  ctx.queryDepths.resize(numQueries);
  ctx.queryMaxDepths.resize(numQueries);
  ctx.queryHasPatterns.resize(numQueries);
  const int precomputedSize = static_cast<int>(leafSubpatterns.perQueryPatterns.size());
  int maxQueryAtoms = 0;
  for (int q = 0; q < numQueries; ++q) {
    const int atomStart     = queriesHost.batchAtomStarts[q];
    const int atomEnd       = queriesHost.batchAtomStarts[q + 1];
    ctx.queryAtomCounts[q]  = atomEnd - atomStart;
    ctx.queryDepths[q]      = getQueryRecursionDepth(queriesHost, q);
    if (ctx.queryDepths[q] > kMaxRecursionDepth) {
      throw std::runtime_error("Recursive SMARTS depth " + std::to_string(ctx.queryDepths[q]) +
                               " exceeds maximum supported depth of " +
                               std::to_string(kMaxRecursionDepth));
    }
    ctx.queryMaxDepths[q]   = (q < static_cast<int>(leafSubpatterns.perQueryMaxDepth.size()))
                                  ? leafSubpatterns.perQueryMaxDepth[q]
                                  : 0;
    ctx.queryHasPatterns[q] = (q < precomputedSize) &&
                              (ctx.queryMaxDepths[q] > 0 || !leafSubpatterns.perQueryPatterns[q][0].empty());
    maxQueryAtoms           = std::max(maxQueryAtoms, ctx.queryAtomCounts[q]);
  }

  ctx.maxTargetAtoms = 0;
  for (int t = 0; t < numTargets; ++t) {
    const int atomStart = targetsHost.batchAtomStarts[t];
    const int atomEnd   = targetsHost.batchAtomStarts[t + 1];
    ctx.maxTargetAtoms  = std::max(ctx.maxTargetAtoms, atomEnd - atomStart);
  }

  ctx.globalPairMatchStarts.resize(numPairs + 1);
  ctx.globalPairMatchStarts[0] = 0;
  for (int t = 0; t < numTargets; ++t) {
    const int targetAtoms = targetsHost.batchAtomStarts[t + 1] - targetsHost.batchAtomStarts[t];
    for (int q = 0; q < numQueries; ++q) {
      const int pairIdx      = t * numQueries + q;
      const int queryAtoms   = ctx.queryAtomCounts[q];
      const int pairCapacity = targetAtoms * queryAtoms;
      ctx.globalPairMatchStarts[pairIdx + 1] = ctx.globalPairMatchStarts[pairIdx] + pairCapacity;
    }
  }
  metadataRange.pop();

  ScopedNvtxRange resultsAllocRange("CPU: Allocate results structure");
  results.resize(numTargets, numQueries);
  resultsAllocRange.pop();

  std::atomic<int> nextBatchIdx(0);
  std::mutex       resultsMutex;

  ScopedCudaEvent upstreamReadyEvent;
  cudaCheckError(cudaEventRecord(upstreamReadyEvent.event(), stream));

  ScopedNvtxRange allocRange("CPU: Pre-allocate BatchSlots");
  const int maxMatchIndicesPerBatch = effectiveBatchSize * ctx.maxTargetAtoms * maxQueryAtoms;

  int maxPatternsPerDepth = 256;
  for (size_t q = 0; q < queriesHost.recursivePatterns.size(); ++q) {
    const auto& recInfo = queriesHost.recursivePatterns[q];
    maxPatternsPerDepth = std::max(maxPatternsPerDepth, static_cast<int>(recInfo.patterns.size()));
  }

  // Slot count:
  // - Inline mode: runners need 3 each for triple-buffering (one accumulating, one copying, one computing)
  //   This allows overlap of D2H copy wait with GPU execution and CPU accumulation
  // - Separated mode: runners hold 2 slots each (current + pending), PPs hold 1 each (preparing),
  //   plus buffer slots to keep the queue/pool from starving
  const int numSlots = (numPreprocessors > 0) 
                     ? (numRunners * 2 + numPreprocessors + std::max(2, numRunners))
                     : (numRunners * 3);
  std::vector<std::unique_ptr<ConsolidatedPinnedBuffer>> pinnedBuffers;
  std::vector<std::unique_ptr<BatchSlot>> slots;
  std::vector<BatchSlot*> slotPtrs;
  pinnedBuffers.reserve(numSlots);
  slots.reserve(numSlots);
  slotPtrs.reserve(numSlots);
  for (int i = 0; i < numSlots; ++i) {
    auto buf = std::make_unique<ConsolidatedPinnedBuffer>();
    buf->allocate(effectiveBatchSize, maxMatchIndicesPerBatch, maxPatternsPerDepth);
    
    auto slot = std::make_unique<BatchSlot>(i);
    slot->bindPinnedBuffer(*buf);
    slot->initializeForStream();
    
    slotPtrs.push_back(slot.get());
    pinnedBuffers.push_back(std::move(buf));
    slots.push_back(std::move(slot));
  }
  allocRange.pop();

  BatchSlotPool slotPool;
  slotPool.initialize(slotPtrs);

  ScopedNvtxRange threadRange("Multithreaded batch processing");
  const int totalThreads = numRunners + numPreprocessors;
  std::vector<std::thread> workers;
  std::vector<std::exception_ptr> exceptions(totalThreads);
  workers.reserve(totalThreads);

  std::unique_ptr<PreparedBatchQueue> readyQueue;
  if (numPreprocessors > 0) {
    // Queue capacity should allow PPs to stay ahead of runners
    // At minimum, buffer enough for each runner plus slack
    readyQueue = std::make_unique<PreparedBatchQueue>(numRunners * 2 + numPreprocessors);
  }

  std::atomic<bool> shutdownFlag{false};

  ScopedNvtxRange launchRange("CPU: Launch worker threads");
  
  for (int t = 0; t < numPreprocessors; ++t) {
    workers.emplace_back(preprocessorWorker,
                         t,
                         std::cref(ctx),
                         std::cref(queriesHost),
                         std::cref(leafSubpatterns),
                         upstreamReadyEvent.event(),
                         std::ref(nextBatchIdx),
                         totalNumBatches,
                         effectiveBatchSize,
                         std::ref(slotPool),
                         std::ref(*readyQueue),
                         std::ref(shutdownFlag),
                         std::ref(exceptions[t]));
  }

  for (int t = 0; t < numRunners; ++t) {
    workers.emplace_back(runnerWorker,
                         t,
                         std::cref(ctx),
                         std::ref(targetsDevice),
                         std::cref(queriesDevice),
                         std::cref(queriesHost),
                         std::cref(leafSubpatterns),
                         std::ref(results),
                         std::ref(resultsMutex),
                         algorithm,
                         upstreamReadyEvent.event(),
                         std::ref(nextBatchIdx),
                         totalNumBatches,
                         effectiveBatchSize,
                         std::ref(slotPool),
                         readyQueue.get(),
                         std::ref(shutdownFlag),
                         std::ref(exceptions[numPreprocessors + t]));
  }
  launchRange.pop();

  ScopedNvtxRange joinRange("CPU: Join worker threads");
  // Workers that fail will call shutdown() on pool/queue to unblock others.
  // Join all workers - they'll exit either normally or due to shutdown.
  for (auto& worker : workers) {
    worker.join();
  }
  // Defensive cleanup (shutdown already called if any worker failed)
  slotPool.shutdown();
  if (readyQueue) {
    readyQueue->shutdown();
  }
  joinRange.pop();

  threadRange.pop();

  for (const auto& ex : exceptions) {
    if (ex) {
      std::rethrow_exception(ex);
    }
  }

  cudaCheckError(cudaGetLastError());

  for (auto& buf : pinnedBuffers) {
    AsyncResourceCleaner::instance().scheduleBufferCleanup(std::move(buf));
  }
}

void getSubstructMatches(MoleculesDevice&           targetsDevice,
                         const MoleculesDevice&     queriesDevice,
                         const MoleculesHost&       targetsHost,
                         const MoleculesHost&       queriesHost,
                         SubstructSearchResults&    results,
                         SubstructAlgorithm         algorithm,
                         cudaStream_t               stream,
                         int                        batchSize,
                         int                        numRunners,
                         int                        numPreprocessors) {
  ScopedNvtxRange buildRange("Build LeafSubpatterns");
  LeafSubpatterns leafSubpatterns;
  leafSubpatterns.buildAllPatterns(queriesHost);
  leafSubpatterns.syncToDevice(stream);
  buildRange.pop();

  getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost,
                      leafSubpatterns, results, algorithm, stream, batchSize, numRunners, numPreprocessors);
}

// =============================================================================
// Recursive SMARTS Preprocessing
// =============================================================================

void preprocessRecursiveSmartsBatchedWithEvents(const MoleculesDevice&            targetsDevice,
                                                const MoleculesHost&              queriesHost,
                                                const LeafSubpatterns&            leafSubpatterns,
                                                BatchResultsDevice&               batchResults,
                                                const int                         numQueries,
                                                const int                         batchPairOffset,
                                                const int                         batchSize,
                                                const SubstructAlgorithm          algorithm,
                                                cudaStream_t                      stream,
                                                RecursiveScratchBuffers&          scratch,
                                                std::vector<BatchedPatternEntry>& scratchPatternEntries,
                                                cudaEvent_t*                      depthEvents,
                                                int                               numDepthEvents) {
  ScopedNvtxRange processRecursiveRange("Process recursive batch with events");
  
  // Configure kernels for max shared memory carveout (once per process)
  configureSubstructKernelsSharedMem();
  
  ScopedNvtxRange processRecursiveRangeSetup("Process recursive batch setup");

  scratch.setStream(stream);

  std::vector<BatchedPatternEntry>& patternEntriesHost = scratchPatternEntries;
  patternEntriesHost.clear();

  const int firstQueryInBatch   = batchPairOffset % numQueries;
  const int numUniqueQueries    = std::min(batchSize, numQueries);
  const int recursivePatternsSize = static_cast<int>(queriesHost.recursivePatterns.size());

  int maxDepth = 0;
  for (int i = 0; i < numUniqueQueries; ++i) {
    const int queryIdx = (firstQueryInBatch + i) % numQueries;

    if (queryIdx >= recursivePatternsSize) {
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

      const int patternMolIdx = leafSubpatterns.getPatternIndex(queryIdx, entry.patternId);
      if (patternMolIdx < 0) {
        throw std::runtime_error("Pattern not found in pre-built LeafSubpatterns: queryIdx=" +
                                 std::to_string(queryIdx) + ", patternId=" + std::to_string(entry.patternId));
      }

      BatchedPatternEntry& batchEntry = patternEntriesHost.emplace_back();
      batchEntry.mainQueryIdx    = queryIdx;
      batchEntry.patternId       = entry.patternId;
      batchEntry.patternMolIdx   = patternMolIdx;
      batchEntry.depth           = entry.depth;
      batchEntry.localIdInParent = entry.localIdInParent;
    }
  }

  if (patternEntriesHost.empty()) {
    return;
  }

  const int firstTargetInBatch = batchPairOffset / numQueries;
  const int lastTargetInBatch  = (batchPairOffset + batchSize - 1) / numQueries;
  const int numTargetsInBatch  = lastTargetInBatch - firstTargetInBatch + 1;

  scratch.setStream(stream);

  const auto batchView = batchResults.view();
  constexpr int gsiBuffersPerBlock = 2;
  constexpr int wusBuffersPerBlock = 1;

  const int maxPaintPairsPerSubBatch = std::max(batchSize, 1024);
  processRecursiveRangeSetup.pop();

  for (int currentDepth = 0; currentDepth <= maxDepth; ++currentDepth) {
    ScopedNvtxRange depthRange("Process recursive depth level " + std::to_string(currentDepth));

    std::vector<BatchedPatternEntry> patternsAtDepth;
    for (const auto& entry : patternEntriesHost) {
      if (entry.depth == currentDepth) {
        patternsAtDepth.push_back(entry);
      }
    }

    if (patternsAtDepth.empty()) {
      if (currentDepth < numDepthEvents && depthEvents != nullptr) {
        cudaCheckError(cudaEventRecord(depthEvents[currentDepth], stream));
      }
      continue;
    }

    const size_t numPatterns = patternsAtDepth.size();
    const int patternsPerSubBatch = std::max(1, maxPaintPairsPerSubBatch / numTargetsInBatch);

    for (size_t patternStart = 0; patternStart < numPatterns; patternStart += patternsPerSubBatch) {
      ScopedNvtxRange subBatchRange("Process sub-batch " + std::to_string(patternStart));
      
      const size_t patternEnd            = std::min(patternStart + patternsPerSubBatch, numPatterns);
      const size_t numPatternsInSubBatch = patternEnd - patternStart;
      const size_t numBlocksInSubBatch   = numTargetsInBatch * numPatternsInSubBatch;

      ScopedNvtxRange prepareRange("CPU: Prepare pattern entries");
      const int bufferIdx = scratch.acquireBufferIndex();
      scratch.waitForBuffer(bufferIdx);
      scratch.ensureCapacity(bufferIdx, static_cast<int>(numPatternsInSubBatch));
      for (size_t i = 0; i < numPatternsInSubBatch; ++i) {
        scratch.patternsAtDepthHost[bufferIdx][i] = patternsAtDepth[patternStart + i];
      }
      prepareRange.pop();

      const int buffersPerBlock = (algorithm == SubstructAlgorithm::WarpUnified) ? wusBuffersPerBlock : gsiBuffersPerBlock;
      const size_t overflowNeeded = numBlocksInSubBatch * buffersPerBlock * kOverflowEntriesPerBuffer;

      if (scratch.overflow.size() < overflowNeeded) {
        scratch.overflow.resize(static_cast<size_t>(overflowNeeded * 1.5));
      }

      const size_t labelMatrixNeeded = numBlocksInSubBatch * kLabelMatrixWords;
      if (scratch.labelMatrixBuffer.size() < labelMatrixNeeded) {
        scratch.labelMatrixBuffer.resize(static_cast<size_t>(labelMatrixNeeded * 1.5));
      }

      if (scratch.patternEntries.size() < numPatternsInSubBatch) {
        scratch.patternEntries.resize(static_cast<size_t>(numPatternsInSubBatch * 1.5));
      }
      
      scratch.patternEntries.copyFromHost(scratch.patternsAtDepthHost[bufferIdx], numPatternsInSubBatch);
      scratch.recordCopy(bufferIdx, scratch.patternEntries.stream());

      const uint32_t* recursiveBitsForLabel = (currentDepth > 0) ? batchView.recursiveMatchBits : nullptr;

      labelMatrixPaintKernel<<<numBlocksInSubBatch, kThreadsPerBlock, 0, stream>>>(
        targetsDevice.view(),
        leafSubpatterns.view(),
        scratch.patternEntries.data(),
        static_cast<int>(numPatternsInSubBatch),
        numQueries,
        batchPairOffset,
        batchSize,
        scratch.labelMatrixBuffer.data(),
        firstTargetInBatch,
        recursiveBitsForLabel,
        batchView.maxTargetAtoms);

      switch (algorithm) {
        case SubstructAlgorithm::VF2:
        case SubstructAlgorithm::GSI:
        case SubstructAlgorithm::WarpUnified: {
          substructPaintKernel<SubstructAlgorithm::GSI><<<numBlocksInSubBatch, kThreadsPerBlock, 0, stream>>>(
            targetsDevice.view(),
            leafSubpatterns.view(),
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
        // case SubstructAlgorithm::WarpUnified: {
        //   substructPaintKernel<SubstructAlgorithm::WarpUnified><<<numBlocksInSubBatch, kThreadsPerBlock, 0, stream>>>(
        //     targetsDevice.view(),
        //     leafSubpatterns.view(),
        //     scratch.patternEntries.data(),
        //     static_cast<int>(numPatternsInSubBatch),
        //     batchView.recursiveMatchBits,
        //     batchView.maxTargetAtoms,
        //     numQueries,
        //     0, 0,
        //     batchPairOffset,
        //     batchSize,
        //     scratch.overflow.data(),
        //     scratch.overflow.data(),
        //     kOverflowEntriesPerBuffer,
        //     scratch.labelMatrixBuffer.data(),
        //     firstTargetInBatch);
        //   break;
        // }
      }
    }

    if (currentDepth < numDepthEvents && depthEvents != nullptr) {
      cudaCheckError(cudaEventRecord(depthEvents[currentDepth], stream));
    }
  }

  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit

