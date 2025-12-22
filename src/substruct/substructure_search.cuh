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

#ifndef NVMOLKIT_SUBSTRUCTURE_SEARCH_CUH
#define NVMOLKIT_SUBSTRUCTURE_SEARCH_CUH

#include <cuda_runtime.h>

#include <array>
#include <unordered_map>
#include <vector>

#include "device.h"
#include "device_vector.h"
#include "flat_bit_vect.h"
#include "host_vector.h"
#include "molecules.h"
#include "substruct_algos.cuh"
#include "substruct_types.h"

namespace nvMolKit {

// SubstructAlgorithm and SubstructMatchResultsHost are defined in substruct_types.h

/// Constants for label matrix sizing (must match substructure_search.cu)
constexpr std::size_t kLabelMaxTargetAtoms = 128;
constexpr std::size_t kLabelMaxQueryAtoms  = 64;
constexpr std::size_t kLabelMatrixBits     = kLabelMaxTargetAtoms * kLabelMaxQueryAtoms;

/// Storage type for a single label matrix
using LabelMatrixStorage = FlatBitVect<kLabelMatrixBits>;

/// Number of uint32_t words per label matrix
constexpr std::size_t kLabelMatrixWords = LabelMatrixStorage::kStorageCount;

/**
 * @brief POD view into batch-local device-side substructure match results.
 *
 * Passed to kernels by value. All pointers are device memory.
 * All indexing is batch-local (0 to batchSize-1).
 */
struct SubstructMatchResultsDeviceView {
  int* matchCounts;      ///< [batchSize] actual match count per pair
  int* reportedCounts;   ///< [batchSize] stored match count per pair
  int* pairMatchStarts;  ///< [batchSize + 1] batch-local offset into matchIndices

  int16_t* matchIndices;  ///< Flattened match mappings (batch-local)

  int numQueries;         ///< Total queries (for decoding global pair indices)

  /// Number of query atoms (stride for match indices)
  const int* queryAtomCounts;  ///< [numQueries] atoms per query molecule

  // Pre-allocated per-block overflow buffers (simple chunked DeviceVector)
  PartialMatch* overflowBuffer;       ///< Base pointer to overflow storage
  int           overflowEntriesPerBuffer;  ///< Entries per buffer (kOverflowEntriesPerBuffer)
  int           overflowBuffersPerBlock;   ///< 2 for GSI ping-pong, 1 for WUS

  // Per-pair recursive match bits: [batchSize * maxTargetAtoms] with 32 bits per atom
  uint32_t* recursiveMatchBits;  ///< Indexed by batchLocalIdx * maxTargetAtoms + atomIdx
  int       maxTargetAtoms;      ///< Stride for recursiveMatchBits indexing

  // Pre-computed label matrices: [batchSize] label matrices in global memory
  uint32_t* labelMatrixBuffer;  ///< Indexed by batchLocalIdx * kLabelMatrixWords

  /// Get pointer to label matrix for a batch-local pair index
  __device__ __forceinline__ uint32_t* getLabelMatrixPtr(int batchLocalIdx) const {
    return labelMatrixBuffer + batchLocalIdx * kLabelMatrixWords;
  }

  /// Get overflow buffer for this block (GSI ping-pong: bufferIdx 0 or 1)
  __device__ __forceinline__ PartialMatch* getOverflowBuffer(int bufferIdx = 0) const {
    return overflowBuffer + (blockIdx.x * overflowBuffersPerBlock + bufferIdx) * overflowEntriesPerBuffer;
  }

  /// Get overflow buffer capacity (entries per buffer)
  __device__ __forceinline__ int getOverflowCapacity() const {
    return overflowEntriesPerBuffer;
  }

  /// Get recursive match bits for a batch-local (pair, atom) combination
  __device__ __forceinline__ uint32_t getRecursiveMatchBits(int batchLocalIdx, int atomIdx) const {
    return recursiveMatchBits[batchLocalIdx * maxTargetAtoms + atomIdx];
  }

  /// Set a recursive match bit for a batch-local (pair, atom, pattern) combination
  __device__ __forceinline__ void setRecursiveMatchBit(int batchLocalIdx, int atomIdx, int patternId) const {
    if (patternId < 32) {
      atomicOr(&recursiveMatchBits[batchLocalIdx * maxTargetAtoms + atomIdx], 1u << patternId);
    }
  }
};

/**
 * @brief Batch-local device-side storage for substructure match results.
 *
 * Owns device memory for a single batch and provides views for kernel access.
 * Results are copied back to host after each batch and accumulated.
 */
class BatchResultsDevice {
 public:
  BatchResultsDevice() = default;
  explicit BatchResultsDevice(cudaStream_t stream) : stream_(stream) { setStream(stream); }

  /**
   * @brief Allocate batch-local buffers for a specific batch.
   *
   * @param batchSize Number of pairs in this batch
   * @param batchPairMatchStarts Batch-local offsets into matchIndices [batchSize + 1]
   * @param totalBatchMatchIndices Total match indices capacity for this batch
   * @param numQueries Total number of queries (for kernel view)
   * @param maxTargetAtoms Max atoms per target (stride for recursiveMatchBits)
   * @param numBuffersPerBlock Overflow buffers per block (2 for GSI, 1 for WUS)
   */
  void allocateBatch(int         batchSize,
                     const int*  batchPairMatchStarts,
                     int         totalBatchMatchIndices,
                     int         numQueries,
                     int         maxTargetAtoms,
                     int         numBuffersPerBlock);

  /**
   * @brief Get a view suitable for passing to CUDA kernels.
   */
  [[nodiscard]] SubstructMatchResultsDeviceView view() const;

  void setStream(cudaStream_t stream);

  /**
   * @brief Zero the recursive match bits buffer for a new batch.
   */
  void zeroRecursiveBits();

  /**
   * @brief Zero the label matrix buffer for a new batch.
   */
  void zeroLabelMatrixBuffer();

  /**
   * @brief Copy batch results to host vectors.
   *
   * @param hostMatchCounts Output: match counts for this batch [batchSize]
   * @param hostReportedCounts Output: reported counts for this batch [batchSize]
   * @param hostMatchIndices Output: match indices for this batch
   */
  void copyBatchToHost(PinnedHostVector<int>&     hostMatchCounts,
                       PinnedHostVector<int>&     hostReportedCounts,
                       PinnedHostVector<int16_t>& hostMatchIndices) const;

  void setQueryAtomCounts(const std::vector<int>& queryAtomCounts);

  [[nodiscard]] int batchSize() const { return batchSize_; }
  [[nodiscard]] int maxTargetAtoms() const { return maxTargetAtoms_; }
  [[nodiscard]] uint32_t* recursiveMatchBits() { return recursiveMatchBits_.data(); }

 private:
  cudaStream_t stream_ = nullptr;

  int batchSize_    = 0;
  int numQueries_   = 0;
  int maxTargetAtoms_ = 0;

  AsyncDeviceVector<int>     matchCounts_;
  AsyncDeviceVector<int>     reportedCounts_;
  AsyncDeviceVector<int>     pairMatchStarts_;
  AsyncDeviceVector<int16_t> matchIndices_;
  AsyncDeviceVector<int>     queryAtomCounts_;

  AsyncDeviceVector<PartialMatch> overflowBuffer_;
  int overflowBuffersPerBlock_ = 0;

  AsyncDeviceVector<uint32_t> recursiveMatchBits_;

  AsyncDeviceVector<uint32_t> labelMatrixBuffer_;

  int totalBatchMatchIndices_ = 0;
};

/**
 * @brief Perform batch substructure matching on GPU with host-side CSR results.
 *
 * This overload returns results in a CSR format suitable for efficient processing.
 * Manages device memory internally.
 *
 * @param targetsDevice Device-resident target molecules (use addToBatch to build)
 * @param queriesDevice Device-resident query molecules (use addQueryToBatch to build)
 * @param targetsHost Host-side target data (for atom counts)
 * @param queriesHost Host-side query data (for atom counts)
 * @param results Host-side CSR output storage (will be populated)
 * @param algorithm Algorithm to use for matching
 * @param stream CUDA stream for async operations
 * @param batchSize Number of pairs per batch (default 1024).
 */
void getSubstructMatches(MoleculesDevice&           targetsDevice,
                         const MoleculesDevice&     queriesDevice,
                         const MoleculesHost&       targetsHost,
                         const MoleculesHost&       queriesHost,
                         SubstructMatchResultsHost& results,
                         SubstructAlgorithm         algorithm,
                         cudaStream_t               stream,
                         int                        batchSize = 1024);

/**
 * @brief Perform batch substructure matching on GPU with simple accumulated results.
 *
 * This overload returns results in an easy-to-use nested vector format.
 *
 * @param targetsDevice Device-resident target molecules (use addToBatch to build)
 * @param queriesDevice Device-resident query molecules (use addQueryToBatch to build)
 * @param targetsHost Host-side target data (for atom counts)
 * @param queriesHost Host-side query data (for atom counts)
 * @param results Output: matches[target][query][match] = vector of target atom indices
 * @param algorithm Algorithm to use for matching
 * @param stream CUDA stream for async operations
 * @param batchSize Number of pairs per batch (default 1024).
 */
void getSubstructMatches(MoleculesDevice&        targetsDevice,
                         const MoleculesDevice&  queriesDevice,
                         const MoleculesHost&    targetsHost,
                         const MoleculesHost&    queriesHost,
                         SubstructSearchResults& results,
                         SubstructAlgorithm      algorithm,
                         cudaStream_t            stream,
                         int                     batchSize = 1024);

/**
 * @brief Per-pattern metadata for batched recursive preprocessing kernel.
 *
 * Each entry describes one recursive pattern in the combined batch:
 * which main query it belongs to, what bit to paint, and where the
 * pattern data starts in the combined pattern batch.
 */
struct BatchedPatternEntry {
  int mainQueryIdx;     ///< Index of the main query this pattern belongs to
  int patternId;        ///< Bit position (0-31) to paint for this pattern
  int patternMolIdx;    ///< Index into the combined patterns MoleculesDevice
  int depth;            ///< Nesting depth (0=leaf, higher=parent of children)
  int localIdInParent;  ///< Bit position in parent's input (for nested patterns)
};

/**
 * @brief Scratch buffers for recursive SMARTS preprocessing.
 *
 * Reusable device memory to avoid repeated alloc/free between kernels.
 * For nested patterns, intermediateBits holds results from child levels
 * that become input for parent patterns.
 */
struct RecursiveScratchBuffers {
  AsyncDeviceVector<BatchedPatternEntry> patternEntries;
  AsyncDeviceVector<PartialMatch>        overflow;
  AsyncDeviceVector<uint32_t>            labelMatrixBuffer;
  AsyncDeviceVector<uint32_t>            intermediateBits;  ///< Child pattern results for nested recursion
  PinnedHostVector<BatchedPatternEntry>  patternsAtDepthHost;  ///< Pinned buffer for H2D transfers
  ScopedCudaEvent                        patternsAtDepthHostCopyDone;  ///< Guards reuse of patternsAtDepthHost
  bool                                   patternsAtDepthHostCopyPending = false;

  explicit RecursiveScratchBuffers(cudaStream_t stream) 
      : patternEntries(), overflow(), labelMatrixBuffer(), intermediateBits(), patternsAtDepthHost(),
        patternsAtDepthHostCopyDone(), patternsAtDepthHostCopyPending(false) {
    patternEntries.setStream(stream);
    overflow.setStream(stream);
    labelMatrixBuffer.setStream(stream);
    intermediateBits.setStream(stream);
  }

  void setStream(cudaStream_t stream) {
    patternEntries.setStream(stream);
    overflow.setStream(stream);
    labelMatrixBuffer.setStream(stream);
    intermediateBits.setStream(stream);
  }
};

/**
 * @brief Key for caching recursive patterns.
 */
struct RecursivePatternKey {
  int queryIdx;
  int patternId;

  bool operator==(const RecursivePatternKey& other) const {
    return queryIdx == other.queryIdx && patternId == other.patternId;
  }
};

/**
 * @brief Hash function for RecursivePatternKey.
 */
struct RecursivePatternKeyHash {
  std::size_t operator()(const RecursivePatternKey& key) const {
    return std::hash<int>()(key.queryIdx) ^ (std::hash<int>()(key.patternId) << 16);
  }
};

/**
 * @brief Cache for recursive SMARTS patterns.
 *
 * Caches patterns across batch iterations to avoid reprocessing the same
 * patterns for each batch. The cache maps (queryIdx, patternId) to the
 * molecule index in a persistent MoleculesHost/MoleculesDevice.
 */
struct RecursivePatternCache {
  std::unordered_map<RecursivePatternKey, int, RecursivePatternKeyHash> patternIndexMap;
  MoleculesHost   cachedPatterns;
  MoleculesDevice cachedPatternsDevice;
  bool            deviceNeedsUpdate = false;

  RecursivePatternCache() = default;
  explicit RecursivePatternCache(cudaStream_t stream) : cachedPatternsDevice(stream) {}

  void setStream(cudaStream_t stream) { cachedPatternsDevice.setStream(stream); }

  /**
   * @brief Look up or add a pattern to the cache.
   *
   * @param queryIdx Index of the query containing the pattern
   * @param patternId Pattern ID within the query
   * @param queryMol The pattern molecule (only used if not in cache)
   * @param patternInfo Full recursive pattern info for this query (to find children's patternIds)
   * @return The molecule index in the cached patterns batch
   */
  int getOrAddPattern(int queryIdx, int patternId, const RDKit::ROMol* queryMol,
                      const RecursivePatternInfo& patternInfo);

  /**
   * @brief Sync cached patterns to device if needed.
   *
   * @param stream CUDA stream for the copy
   */
  void syncToDevice(cudaStream_t stream);
};

/// Maximum supported recursion depth for nested recursive SMARTS patterns.
/// A query with depth N requires N paint rounds before matching can begin.
constexpr int kMaxRecursionDepth = 4;

/**
 * @brief Two-stream pipeline context for overlapping recursive preprocessing with matching.
 *
 * Uses a high-priority stream for recursive paint operations and a low-priority
 * stream for main query matching. Events synchronize pairs that depend on
 * recursive preprocessing results.
 */
struct TwoStreamPipelineContext {
  ScopedStreamWithPriority recursiveStream;  ///< High priority stream for paint kernels

  /// Low priority streams for match kernels at depth > 0.
  /// Depth 0 uses the main ctx.stream. Depths 1..kMaxRecursionDepth each get their own stream
  /// so matching at different depths can overlap.
  std::array<ScopedStreamWithPriority, kMaxRecursionDepth> matchStreams;

  std::array<ScopedCudaEvent, kMaxRecursionDepth> depthEvents;

  /// Matching: global pair indices for each depth group (depth 0..kMaxRecursionDepth)
  std::array<AsyncDeviceVector<int>, kMaxRecursionDepth + 1> matchGlobalPairIndices;

  /// Matching: batch-local indices for each depth group (depth 0..kMaxRecursionDepth)
  std::array<AsyncDeviceVector<int>, kMaxRecursionDepth + 1> matchBatchLocalIndices;

  /// Host-side schedule: pairs to match after each depth level completes
  std::array<std::vector<int>, kMaxRecursionDepth + 1> matchPairsHost;

  /// Temporary pinned buffers for H2D transfers (reused per depth)
  std::array<PinnedHostVector<int>, kMaxRecursionDepth + 1> matchGlobalPairIndicesHost;

  int maxDepthInBatch = 0;

  /**
   * @brief Construct pipeline context with priority streams.
   *
   * The recursive stream gets high priority (lower numerical value),
   * match streams get low priority (higher numerical value).
   * 
   * @param workerIdx Worker thread index for unique stream naming
   */
  explicit TwoStreamPipelineContext(int workerIdx = 0);
};

/**
 * @brief Preprocess ALL recursive SMARTS patterns for a batch in a single kernel launch.
 *
 * Collects all recursive patterns from all queries that have pairs in the batch,
 * builds a combined pattern batch, and launches a single fused paint kernel.
 *
 * @param targetsDevice Device-resident target molecules
 * @param targetsHost Host-side target data
 * @param queriesHost Host-side query data (contains recursivePatterns per query)
 * @param batchResults The batch results buffer where recursiveMatchBits will be written
 * @param numQueries Total number of queries (for computing pair indices)
 * @param batchPairOffset Global pair index where current batch starts
 * @param batchSize Number of pairs in this batch
 * @param algorithm Algorithm to use for matching
 * @param stream CUDA stream for async operations
 * @param scratch Reusable scratch buffers (avoids alloc/free between kernels)
 * @param patternCache Cache for recursive patterns (reused across batch iterations)
 * @param scratchPatternEntries Vector to store pattern entries for the batch
 */
void preprocessRecursiveSmartsBatched(const MoleculesDevice&            targetsDevice,
                                      const MoleculesHost&              targetsHost,
                                      const MoleculesHost&              queriesHost,
                                      BatchResultsDevice&               batchResults,
                                      int                               numQueries,
                                      int                               batchPairOffset,
                                      int                               batchSize,
                                      SubstructAlgorithm                algorithm,
                                      cudaStream_t                      stream,
                                      RecursiveScratchBuffers&          scratch,
                                      RecursivePatternCache&            patternCache,
                                      std::vector<BatchedPatternEntry>& scratchPatternEntries);

/**
 * @brief Preprocess recursive SMARTS patterns with event recording for two-stream pipeline.
 *
 * Same as preprocessRecursiveSmartsBatched but records events after each depth level
 * for synchronization with the match stream.
 *
 * @param depthEvents Array of events to record after each depth level (size >= maxDepth)
 * @param numDepthEvents Number of events in the array (typically kMaxRecursionDepth)
 */
void preprocessRecursiveSmartsBatchedWithEvents(const MoleculesDevice&            targetsDevice,
                                                const MoleculesHost&              targetsHost,
                                                const MoleculesHost&              queriesHost,
                                                BatchResultsDevice&               batchResults,
                                                int                               numQueries,
                                                int                               batchPairOffset,
                                                int                               batchSize,
                                                SubstructAlgorithm                algorithm,
                                                cudaStream_t                      stream,
                                                RecursiveScratchBuffers&          scratch,
                                                RecursivePatternCache&            patternCache,
                                                std::vector<BatchedPatternEntry>& scratchPatternEntries,
                                                cudaEvent_t*                      depthEvents,
                                                int                               numDepthEvents);

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCTURE_SEARCH_CUH

