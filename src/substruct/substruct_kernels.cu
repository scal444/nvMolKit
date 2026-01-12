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

#include "substruct_kernels.h"

#include "graph_labeler.cuh"
#include "molecules_device.cuh"
#include "substruct_algos.cuh"
#include "substruct_debug.h"
#include "substructure_search.cuh"

namespace nvMolKit {

namespace {

// =============================================================================
// Architecture-Specific Shared Memory Configuration
// =============================================================================

constexpr std::size_t kMaxTargetAtoms = kLabelMaxTargetAtoms;
constexpr std::size_t kMaxQueryAtoms  = kLabelMaxQueryAtoms;

using LabelMatrixView = BitMatrix2DView<kMaxTargetAtoms, kMaxQueryAtoms>;

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
constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;

// =============================================================================
// Shared Memory Carveout Configuration
// =============================================================================

template <typename KernelFunc>
void configureSharedMemCarveout(KernelFunc kernel) {
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
}

inline bool& sharedMemCarveoutConfigured() {
  static bool configured = false;
  return configured;
}

// =============================================================================
// Device Helper Functions
// =============================================================================

/**
 * @brief Compute label matrix and write to global memory.
 *
 * Core logic shared between labelMatrixKernel and labelMatrixPaintKernel.
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

// =============================================================================
// Kernel Definitions
// =============================================================================

/**
 * @brief Kernel for batch label matrix computation.
 *
 * One block per (target, query) pair. Computes label matrix and writes to global buffer.
 */
__global__ void labelMatrixKernel(MoleculesDeviceView targets,
                                  MoleculesDeviceView queries,
                                  const int*          pairIndices,
                                  int                 numQueries,
                                  uint32_t*           labelMatrixBuffer,
                                  const uint32_t*     recursiveMatchBits,
                                  int                 maxTargetAtoms,
                                  const int*          batchLocalIndices) {
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
                                       int                        miniBatchPairOffset,
                                       int                        miniBatchSize,
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

  if (globalPairIdx < miniBatchPairOffset || globalPairIdx >= miniBatchPairOffset + miniBatchSize) {
    return;
  }

  const int batchLocalPairIdx = globalPairIdx - miniBatchPairOffset;

  const MoleculeView target  = getMolecule(targets, targetIdx);
  const MoleculeView pattern = getMolecule(patterns, patternMolIdx);

  __shared__ LabelMatrixStorage sharedLabelMatrix;

  uint32_t* globalOut = labelMatrixBuffer + blockIdx.x * kLabelMatrixWords;

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
 */
template <SubstructAlgorithm Algo>
__global__ void substructMatchKernel(MoleculesDeviceView             targets,
                                     MoleculesDeviceView             queries,
                                     SubstructMatchResultsDeviceView results,
                                     const int*                      pairIndices,
                                     int                             numQueries,
                                     const int*                      batchLocalIndices,
                                     DeviceTimingsData*              timings) {
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
  LabelMatrixView               labelMatrix(&sharedLabelMatrix);

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

  const int matchOffset = results.pairMatchStarts[batchLocalIdx];
  const int maxMatches  = (results.pairMatchStarts[batchLocalIdx + 1] - matchOffset) / query.numAtoms;

  __shared__ int sharedMatchCount;
  __shared__ int sharedReportedCount;

  if (threadIdx.x == 0) {
    sharedMatchCount    = 0;
    sharedReportedCount = 0;
  }
  __syncthreads();

  const int  maxMatchesToFind = results.maxMatchesToFind;
  const bool countOnly        = results.countOnly;

  if constexpr (Algo == SubstructAlgorithm::VF2) {
    namespace cg = cooperative_groups;
    auto tile32  = cg::tiled_partition<32>(cg::this_thread_block());
    const int warpId   = tile32.meta_group_rank();
    const int numWarps = tile32.meta_group_size();

    __shared__ VF2State vf2States[kWarpsPerBlock];

    if (tile32.thread_rank() == 0) {
      vf2States[warpId].init();
    }
    __syncthreads();

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
                                                    matchOffset,
                                                    maxMatchesToFind,
                                                    countOnly);
    }

  } else if constexpr (Algo == SubstructAlgorithm::GSI) {
    __shared__ PartialMatch gsiPartials[kMaxPartialsPerBlock * 2];

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
                                                     maxMatchesToFind,
                                                     countOnly,
                                                     timings);

  }

  __syncthreads();

  if (threadIdx.x == 0) {
    results.matchCounts[batchLocalIdx]    = sharedMatchCount;
    results.reportedCounts[batchLocalIdx] = sharedReportedCount;
  }
}

/**
 * @brief Paint mode kernel for recursive SMARTS preprocessing.
 *
 * Instead of storing match mappings, directly paints recursive match bits
 * into the output buffer.
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
                                     int                         miniBatchPairOffset,
                                     int                         miniBatchSize,
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

  const int mainQueryIdx  = patternEntries ? patternEntries[localPatternIdx].mainQueryIdx : defaultMainQueryIdx;
  const int patternId     = patternEntries ? patternEntries[localPatternIdx].patternId : defaultPatternId;
  const int patternMolIdx = patternEntries ? patternEntries[localPatternIdx].patternMolIdx : localPatternIdx;

  const int globalPairIdx = targetIdx * outputNumQueries + mainQueryIdx;

  if (globalPairIdx < miniBatchPairOffset || globalPairIdx >= miniBatchPairOffset + miniBatchSize) {
    return;
  }

  const int batchLocalPairIdx = globalPairIdx - miniBatchPairOffset;

  const MoleculeView target  = getMolecule(targets, targetIdx);
  const MoleculeView pattern = getMolecule(patterns, patternMolIdx);

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
  }
}

}  // anonymous namespace

// =============================================================================
// Public Launch Wrapper Implementations
// =============================================================================

void launchLabelMatrixKernel(MoleculesDeviceView targets,
                             MoleculesDeviceView queries,
                             const int*          pairIndices,
                             int                 numPairs,
                             int                 numQueries,
                             uint32_t*           labelMatrixBuffer,
                             const uint32_t*     recursiveMatchBits,
                             int                 maxTargetAtoms,
                             const int*          batchLocalIndices,
                             cudaStream_t        stream) {
  labelMatrixKernel<<<numPairs, kThreadsPerBlock, 0, stream>>>(
      targets, queries, pairIndices, numQueries, labelMatrixBuffer,
      recursiveMatchBits, maxTargetAtoms, batchLocalIndices);
}

void launchLabelMatrixPaintKernel(MoleculesDeviceView        targets,
                                  MoleculesDeviceView        patterns,
                                  const BatchedPatternEntry* patternEntries,
                                  int                        numPatterns,
                                  int                        numBlocks,
                                  int                        numQueries,
                                  int                        miniBatchPairOffset,
                                  int                        miniBatchSize,
                                  uint32_t*                  labelMatrixBuffer,
                                  int                        firstTargetIdx,
                                  const uint32_t*            recursiveMatchBits,
                                  int                        maxTargetAtoms,
                                  cudaStream_t               stream) {
  labelMatrixPaintKernel<<<numBlocks, kThreadsPerBlock, 0, stream>>>(
      targets, patterns, patternEntries, numPatterns, numQueries,
      miniBatchPairOffset, miniBatchSize, labelMatrixBuffer, firstTargetIdx,
      recursiveMatchBits, maxTargetAtoms);
}

void launchSubstructMatchKernel(SubstructAlgorithm              algorithm,
                                MoleculesDeviceView             targets,
                                MoleculesDeviceView             queries,
                                SubstructMatchResultsDeviceView results,
                                const int*                      pairIndices,
                                int                             numPairs,
                                int                             numQueries,
                                const int*                      batchLocalIndices,
                                DeviceTimingsData*              timings,
                                cudaStream_t                    stream) {
  switch (algorithm) {
    case SubstructAlgorithm::VF2:
      substructMatchKernel<SubstructAlgorithm::VF2><<<numPairs, kThreadsPerBlock, 0, stream>>>(
          targets, queries, results, pairIndices, numQueries, batchLocalIndices, timings);
      break;
    case SubstructAlgorithm::GSI:
      substructMatchKernel<SubstructAlgorithm::GSI><<<numPairs, kThreadsPerBlock, 0, stream>>>(
          targets, queries, results, pairIndices, numQueries, batchLocalIndices, timings);
      break;
  }
}

void launchSubstructPaintKernel(SubstructAlgorithm          algorithm,
                                MoleculesDeviceView         targets,
                                MoleculesDeviceView         patterns,
                                const BatchedPatternEntry*  patternEntries,
                                int                         numPatterns,
                                int                         numBlocks,
                                uint32_t*                   outputRecursiveBits,
                                int                         maxTargetAtoms,
                                int                         outputNumQueries,
                                int                         defaultPatternId,
                                int                         defaultMainQueryIdx,
                                int                         miniBatchPairOffset,
                                int                         miniBatchSize,
                                PartialMatch*               overflowA,
                                PartialMatch*               overflowB,
                                int                         overflowCapacity,
                                const uint32_t*             labelMatrixBuffer,
                                int                         firstTargetIdx,
                                cudaStream_t                stream) {
  switch (algorithm) {
    case SubstructAlgorithm::VF2:
      // VF2 paint mode not implemented
      break;
    case SubstructAlgorithm::GSI:
      substructPaintKernel<SubstructAlgorithm::GSI><<<numBlocks, kThreadsPerBlock, 0, stream>>>(
          targets, patterns, patternEntries, numPatterns,
          outputRecursiveBits, maxTargetAtoms, outputNumQueries,
          defaultPatternId, defaultMainQueryIdx, miniBatchPairOffset, miniBatchSize,
          overflowA, overflowB, overflowCapacity, labelMatrixBuffer, firstTargetIdx);
      break;
  }
}

void configureSubstructKernelsSharedMem() {
  if (sharedMemCarveoutConfigured()) return;
  
  configureSharedMemCarveout(substructMatchKernel<SubstructAlgorithm::GSI>);
  configureSharedMemCarveout(substructPaintKernel<SubstructAlgorithm::GSI>);
  
  sharedMemCarveoutConfigured() = true;
}

}  // namespace nvMolKit

