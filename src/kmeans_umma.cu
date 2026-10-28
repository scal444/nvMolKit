// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Blackwell (sm_100a / sm_103a) tensor-core assignment with tcgen05 MMAs; accumulators and the bit operand live in
 * tensor memory.
 *
 * Per CTA: 128 rows (one M=128 accumulator) against every center, in tiles of 192 centers (N=192) and stages of 64
 * bits (four K=16 MMAs, A from tensor memory, B from shared memory). Warp roles:
 *   - warp 0: loads center stages with one cp.async.bulk each. Centers are packed on the device into exactly the
 *     shared-memory image the MMA reads (K-major, 128-byte swizzle), so a stage is one contiguous 24 KiB copy.
 *   - warp 1: allocates tensor memory and issues the MMAs (one thread).
 *   - warps 2-9: expand the CTA's packed bits to fp16 and store them into the A stage ring in tensor memory with
 *     tcgen05.st (thread = row = lane; two warps per lane quadrant take alternate stages). Keeping A out of shared
 *     memory left the shared-memory bandwidth to B; measured 1.36 vs 1.29 PFLOP/s with A in shared memory.
 *   - warps 10-17: epilogue. Thread = row = tensor-memory lane; two warps per quadrant each scan half of a tile's
 *     columns with tcgen05.ld, fold the metric into four independent running maxima, and merge at the end. Two
 *     accumulator buffers let the epilogue of one tile overlap the MMAs of the next.
 * Tensor memory: 2 x 192 accumulator columns + 4 x 32 A-stage columns = 512. Stages are a ring of up to four buffers
 * guarded by mbarriers (full: center bytes + one arrival per A-writer warp; empty: tcgen05.commit of the MMAs).
 */

#include <cuda_fp16.h>

#include <stdexcept>

#include "src/kmeans_kernels.cuh"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit::detail {
namespace {

constexpr int         kUmmaRows          = 128;
constexpr int         kUmmaCenters       = kUmmaBlockCenters;  // 192
constexpr int         kUmmaStageBits     = 64;
constexpr int         kUmmaMaxStages     = 4;
//! Static shared memory of the kernel (barriers, expansion table, half-row results), with margin.
constexpr std::size_t kUmmaStaticBytes   = 8 * 1024;
constexpr int         kUmmaWriterWarps   = 8;
constexpr int         kUmmaEpilogueWarps = 8;
constexpr int         kUmmaThreads       = (2 + kUmmaWriterWarps + kUmmaEpilogueWarps) * 32;
constexpr int         kUmmaAWriters      = kUmmaWriterWarps * 32;
static_assert(kUmmaAWriters == 2 * kUmmaRows, "two A writers per row, alternating stages");
constexpr int kUmmaBBytes      = kUmmaCenters * kUmmaStageBits * 2;  // 24 KiB
constexpr int kUmmaTmemColumns = 512;
//! Tensor-memory columns: two accumulator buffers, then the A stage ring (32 columns = 64 fp16 per row each).
constexpr int kUmmaAColumn     = 2 * kUmmaCenters;
constexpr int kUmmaAStageCols  = kUmmaStageBits / 2;
static_assert(kUmmaAColumn + 4 * kUmmaAStageCols <= kUmmaTmemColumns, "tensor memory budget");

#if defined(__CUDA_ARCH__) && (defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL) || \
                               defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__))
#define NVMOLKIT_KMEANS_UMMA 1
#endif

#ifdef NVMOLKIT_KMEANS_UMMA

__device__ __forceinline__ unsigned sharedAddress(const void* pointer) {
  return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void mbarrierInit(std::uint64_t* barrier, const unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(sharedAddress(barrier)), "r"(count));
}

__device__ __forceinline__ void mbarrierArrive(std::uint64_t* barrier) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(sharedAddress(barrier)) : "memory");
}

__device__ __forceinline__ void mbarrierArriveExpectTx(std::uint64_t* barrier, const unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(sharedAddress(barrier)), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbarrierWait(std::uint64_t* barrier, const unsigned parity) {
  asm volatile(
    "{\n"
    ".reg .pred done;\n"
    "WAIT_%=:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
    "@!done bra WAIT_%=;\n"
    "}\n" ::"r"(sharedAddress(barrier)),
    "r"(parity)
    : "memory");
}

__device__ __forceinline__ void bulkCopy(void*          shared,
                                         const void*    global,
                                         const unsigned bytes,
                                         std::uint64_t* barrier) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                 sharedAddress(shared)),
               "l"(global),
               "r"(bytes),
               "r"(sharedAddress(barrier))
               : "memory");
}

__device__ __forceinline__ void tmemFenceAfter() {
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
}

__device__ __forceinline__ void tmemFenceBefore() {
  asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
}

//! Shared-memory matrix descriptor: K-major, 128-byte swizzle, 8-row core groups 1024 bytes apart.
__device__ __forceinline__ std::uint64_t smemDescriptor(const unsigned address) {
  std::uint64_t descriptor = 0;
  descriptor |= static_cast<std::uint64_t>((address >> 4) & 0x3FFFU);
  descriptor |= static_cast<std::uint64_t>(1U) << 16;          // leading byte offset (unused with swizzle)
  descriptor |= static_cast<std::uint64_t>(1024U >> 4) << 32;  // stride byte offset
  descriptor |= static_cast<std::uint64_t>(1U) << 46;          // descriptor version (Blackwell)
  descriptor |= static_cast<std::uint64_t>(2U) << 61;          // SWIZZLE_128B
  return descriptor;
}

//! kind::f16 instruction descriptor: fp16 A and B (K-major), fp32 accumulator, M=128, N=256.
constexpr unsigned kInstructionDescriptor = (1U << 4) |               // D format F32
                                            (0U << 7) | (0U << 10) |  // A, B format F16
                                            (static_cast<unsigned>(kUmmaCenters >> 3) << 17) |
                                            (static_cast<unsigned>(kUmmaRows >> 4) << 24);

//! D (tensor memory) += A (tensor memory) x B (shared memory descriptor).
__device__ __forceinline__ void umma(const unsigned      tmem,
                                     const unsigned      a,
                                     const std::uint64_t b,
                                     const unsigned      accumulate) {
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %4, 0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n"
    "}\n" ::"r"(tmem),
    "r"(a),
    "l"(b),
    "r"(kInstructionDescriptor),
    "r"(accumulate));
}

__device__ __forceinline__ void ummaCommit(std::uint64_t* barrier) {
  asm volatile(
    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(sharedAddress(barrier))
    : "memory");
}

//! Issues a 32-column tensor-memory load; the registers are valid only after tmemWaitLoad on the same array.
__device__ __forceinline__ void tmemLoadAsync(unsigned (&r)[32], const unsigned address) {
  asm volatile(
    "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];\n"
    : "=r"(r[0]),
      "=r"(r[1]),
      "=r"(r[2]),
      "=r"(r[3]),
      "=r"(r[4]),
      "=r"(r[5]),
      "=r"(r[6]),
      "=r"(r[7]),
      "=r"(r[8]),
      "=r"(r[9]),
      "=r"(r[10]),
      "=r"(r[11]),
      "=r"(r[12]),
      "=r"(r[13]),
      "=r"(r[14]),
      "=r"(r[15]),
      "=r"(r[16]),
      "=r"(r[17]),
      "=r"(r[18]),
      "=r"(r[19]),
      "=r"(r[20]),
      "=r"(r[21]),
      "=r"(r[22]),
      "=r"(r[23]),
      "=r"(r[24]),
      "=r"(r[25]),
      "=r"(r[26]),
      "=r"(r[27]),
      "=r"(r[28]),
      "=r"(r[29]),
      "=r"(r[30]),
      "=r"(r[31])
    : "r"(address));
}

//! Waits for outstanding tensor-memory loads. The registers pass through the asm so no use is scheduled before it.
__device__ __forceinline__ void tmemWaitLoad(unsigned (&r)[32]) {
  asm volatile("tcgen05.wait::ld.sync.aligned;\n"
               : "+r"(r[0]),
                 "+r"(r[1]),
                 "+r"(r[2]),
                 "+r"(r[3]),
                 "+r"(r[4]),
                 "+r"(r[5]),
                 "+r"(r[6]),
                 "+r"(r[7]),
                 "+r"(r[8]),
                 "+r"(r[9]),
                 "+r"(r[10]),
                 "+r"(r[11]),
                 "+r"(r[12]),
                 "+r"(r[13]),
                 "+r"(r[14]),
                 "+r"(r[15]),
                 "+r"(r[16]),
                 "+r"(r[17]),
                 "+r"(r[18]),
                 "+r"(r[19]),
                 "+r"(r[20]),
                 "+r"(r[21]),
                 "+r"(r[22]),
                 "+r"(r[23]),
                 "+r"(r[24]),
                 "+r"(r[25]),
                 "+r"(r[26]),
                 "+r"(r[27]),
                 "+r"(r[28]),
                 "+r"(r[29]),
                 "+r"(r[30]),
                 "+r"(r[31])
               :
               : "memory");
}

//! Bits 0 and 1 of x as an fp16x2 of zeros and ones.
__device__ __forceinline__ unsigned bitsToHalf2(const unsigned x) {
  return ((x & 1U) | ((x << 15) & 0x10000U)) * 0x3C00U;
}

//! Stores 32 consecutive 32-bit columns of this thread's tensor-memory lane and waits for completion.
__device__ __forceinline__ void tmemStore32(const unsigned address, const unsigned (&r)[32]) {
  asm volatile(
    "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};\n" ::"r"(address),
    "r"(r[0]),
    "r"(r[1]),
    "r"(r[2]),
    "r"(r[3]),
    "r"(r[4]),
    "r"(r[5]),
    "r"(r[6]),
    "r"(r[7]),
    "r"(r[8]),
    "r"(r[9]),
    "r"(r[10]),
    "r"(r[11]),
    "r"(r[12]),
    "r"(r[13]),
    "r"(r[14]),
    "r"(r[15]),
    "r"(r[16]),
    "r"(r[17]),
    "r"(r[18]),
    "r"(r[19]),
    "r"(r[20]),
    "r"(r[21]),
    "r"(r[22]),
    "r"(r[23]),
    "r"(r[24]),
    "r"(r[25]),
    "r"(r[26]),
    "r"(r[27]),
    "r"(r[28]),
    "r"(r[29]),
    "r"(r[30]),
    "r"(r[31])
    : "memory");
  asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
}

#endif  // NVMOLKIT_KMEANS_UMMA

template <KMeansMetric Metric, int Stages>
__global__ void __launch_bounds__(kUmmaThreads, 1) ummaAssignKernel(const AssignArgs args, const int paddedBits) {
#ifdef NVMOLKIT_KMEANS_UMMA
  extern __shared__ __align__(1024) unsigned char sharedRaw[];
  // Dynamic shared memory is only 16-byte aligned by contract; align the stage buffers to 1024 for the swizzle.
  unsigned char* shared      = sharedRaw + ((1024U - (sharedAddress(sharedRaw) & 1023U)) & 1023U);
  unsigned char* bStages     = shared;
  const int      paddedWords = paddedBits / 32;
  auto*          rowWords    = reinterpret_cast<std::uint32_t*>(shared + Stages * kUmmaBBytes);

  __shared__ std::uint64_t full[Stages];
  __shared__ std::uint64_t empty[Stages];
  __shared__ std::uint64_t accFull[2];
  __shared__ std::uint64_t accEmpty[2];
  __shared__ unsigned      tmemBase;
  __shared__ float         halfScores[kUmmaRows];
  __shared__ int           halfIndices[kUmmaRows];
  __shared__ int           changedCount;

  const int warp     = static_cast<int>(threadIdx.x / 32);
  const int lane     = static_cast<int>(threadIdx.x % 32);
  const int blockRow = static_cast<int>(blockIdx.x) * kUmmaRows;

  for (int index = static_cast<int>(threadIdx.x); index < kUmmaRows * paddedWords; index += kUmmaThreads) {
    const int row   = index / paddedWords;
    const int word  = index % paddedWords;
    const int g     = blockRow + row;
    rowWords[index] = g < args.numRows && word < args.numWords ?
                        args.fingerprints[static_cast<std::size_t>(g) * args.numWords + word] :
                        0U;
  }
  if (threadIdx.x == 0) {
    for (int s = 0; s < Stages; ++s) {
      mbarrierInit(&full[s], 1 + kUmmaWriterWarps / 2);
      mbarrierInit(&empty[s], 1);
    }
    for (int b = 0; b < 2; ++b) {
      mbarrierInit(&accFull[b], 1);
      mbarrierInit(&accEmpty[b], kUmmaEpilogueWarps);
    }
    changedCount = 0;
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (warp == 1) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(sharedAddress(&tmemBase)),
                 "r"(kUmmaTmemColumns));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
  }
  tmemFenceBefore();
  __syncthreads();
  tmemFenceAfter();
  const unsigned tmem = tmemBase;

  const int stagesPerTile = paddedBits / kUmmaStageBits;
  const int numTiles      = (args.numClusters + kUmmaCenters - 1) / kUmmaCenters;
  const int numSteps      = numTiles * stagesPerTile;

  if (warp == 0) {
    if (lane == 0) {
      const auto* source = reinterpret_cast<const unsigned char*>(args.centers);
      for (int step = 0; step < numSteps; ++step) {
        const int stage = step % Stages;
        mbarrierWait(&empty[stage], ((step / Stages) & 1U) ^ 1U);
        mbarrierArriveExpectTx(&full[stage], kUmmaBBytes);
        bulkCopy(bStages + stage * kUmmaBBytes,
                 source + static_cast<std::size_t>(step) * kUmmaBBytes,
                 kUmmaBBytes,
                 &full[stage]);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      for (int tile = 0; tile < numTiles; ++tile) {
        const int buffer = tile & 1;
        mbarrierWait(&accEmpty[buffer], ((tile / 2) & 1U) ^ 1U);
        tmemFenceAfter();
        const unsigned accumulator = tmem + buffer * kUmmaCenters;
        for (int kStage = 0; kStage < stagesPerTile; ++kStage) {
          const int step  = tile * stagesPerTile + kStage;
          const int stage = step % Stages;
          mbarrierWait(&full[stage], (step / Stages) & 1U);
          tmemFenceAfter();
          const unsigned aBase = tmem + kUmmaAColumn + stage * kUmmaAStageCols;
          const unsigned bBase = sharedAddress(bStages + stage * kUmmaBBytes);
#pragma unroll
          for (int k = 0; k < kUmmaStageBits / 16; ++k) {
            umma(accumulator, aBase + k * 8, smemDescriptor(bBase + k * 32), kStage > 0 || k > 0);
          }
          ummaCommit(&empty[stage]);
        }
        ummaCommit(&accFull[buffer]);
      }
    }
  } else if (warp < 2 + kUmmaWriterWarps) {
    // One thread per row stores the row's 64-bit slice of the stage into tensor memory as 32 fp16x2 columns. A warp
    // may only access its lane quadrant (warp % 4), so the row is that quadrant's lane.
    const int      row     = (warp % 4) * 32 + lane;
    const unsigned rowTmem = tmem + (static_cast<unsigned>((warp % 4) * 32) << 16) + kUmmaAColumn;
    for (int step = (warp - 2) / 4; step < numSteps; step += 2) {
      const int stage  = step % Stages;
      const int kStage = step % stagesPerTile;
      mbarrierWait(&empty[stage], ((step / Stages) & 1U) ^ 1U);
      tmemFenceAfter();
      {
        const uint2 words = *reinterpret_cast<const uint2*>(rowWords + row * paddedWords + kStage * 2);
        unsigned    columns[32];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          columns[j]      = bitsToHalf2(words.x >> (2 * j));
          columns[16 + j] = bitsToHalf2(words.y >> (2 * j));
        }
        tmemStore32(rowTmem + stage * kUmmaAStageCols, columns);
      }
      tmemFenceBefore();
      __syncwarp();
      if (lane == 0) {
        mbarrierArrive(&full[stage]);
      }
    }
  } else {
    // Eight epilogue warps: two per tensor-memory lane quadrant, each scanning half of every tile's columns with
    // kChains independent running maxima (merged at the end) so the comparisons do not form one serial chain.
    constexpr int           kChains   = 4;
    const int               epilogue  = warp - 2 - kUmmaWriterWarps;
    const int               quadrant  = warp % 4;
    const int               half      = epilogue / 4;
    const int               row       = quadrant * 32 + lane;
    const int               globalRow = blockRow + row;
    const RowScorer<Metric> scorer(globalRow < args.numRows ? args.rowCounts[globalRow] : 0);
    float                   bestValue[kChains];
    float                   bestDen[kChains];
    int                     bestIndex[kChains];
#pragma unroll
    for (int c = 0; c < kChains; ++c) {
      bestValue[c] = -INFINITY;
      bestDen[c]   = 1.0F;
      bestIndex[c] = 0x7FFFFFFF;
    }
    const unsigned laneBase = tmem + (static_cast<unsigned>(quadrant * 32) << 16) + half * (kUmmaCenters / 2);
    for (int tile = 0; tile < numTiles; ++tile) {
      const int buffer = tile & 1;
      mbarrierWait(&accFull[buffer], (tile / 2) & 1U);
      tmemFenceAfter();
      constexpr int kChunks = kUmmaCenters / 2 / 32;
      unsigned      raw[2][32];
      tmemLoadAsync(raw[0], laneBase + buffer * kUmmaCenters);
#pragma unroll
      for (int chunkIndex = 0; chunkIndex < kChunks; ++chunkIndex) {
        const int chunk = chunkIndex * 32;
        tmemWaitLoad(raw[chunkIndex & 1]);
        if (chunkIndex + 1 < kChunks) {
          tmemLoadAsync(raw[(chunkIndex + 1) & 1], laneBase + buffer * kUmmaCenters + chunk + 32);
        }
        float dots[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          dots[i] = __uint_as_float(raw[chunkIndex & 1][i]);
        }
        const int   first  = tile * kUmmaCenters + half * (kUmmaCenters / 2) + chunk;
        // Norms are padded with zeros to whole tiles; padded columns are skipped below.
        const auto* norms4 = reinterpret_cast<const float4*>(args.norms + first);
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const float4 n4         = norms4[q];
          const float  colNorm[4] = {n4.x, n4.y, n4.z, n4.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int i   = q * 4 + e;
            const int c   = i % kChains;
            const int col = first + i;
            if constexpr (Metric == KMeansMetric::Tanimoto) {
              // With at least one set bit the denominator |x| + |c|^2 - x.c is positive; empty rows are scored like
              // the other kernels (an all-zero center has similarity 1).
              float numerator   = dots[i];
              float denominator = scorer.count + colNorm[e] - numerator;
              if (scorer.count == 0.0F && !(denominator > 0.0F)) {
                numerator   = 1.0F;
                denominator = 1.0F;
              }
              if (col < args.numClusters && numerator * bestDen[c] > bestValue[c] * denominator) {
                bestValue[c] = numerator;
                bestDen[c]   = denominator;
                bestIndex[c] = col;
              }
            } else {
              const float score = scorer.score(dots[i], colNorm[e]);
              if (col < args.numClusters && score > bestValue[c]) {
                bestValue[c] = score;
                bestIndex[c] = col;
              }
            }
          }
        }
      }
      tmemFenceBefore();
      __syncwarp();
      if (lane == 0) {
        mbarrierArrive(&accEmpty[buffer]);
      }
    }
    float score = -INFINITY;
    int   index = 0x7FFFFFFF;
#pragma unroll
    for (int c = 0; c < kChains; ++c) {
      const float chainScore = Metric == KMeansMetric::Tanimoto ?
                                 (bestIndex[c] == 0x7FFFFFFF ? -INFINITY : __fdiv_rn(bestValue[c], bestDen[c])) :
                                 bestValue[c];
      if (chainScore > score || (chainScore == score && bestIndex[c] < index)) {
        score = chainScore;
        index = bestIndex[c];
      }
    }
    if (half == 1) {
      halfScores[row]  = score;
      halfIndices[row] = index;
    }
    asm volatile("bar.sync 1, %0;\n" ::"n"(kUmmaEpilogueWarps * 32));
    if (half == 0 && globalRow < args.numRows) {
      const float otherScore = halfScores[row];
      const int   otherIndex = halfIndices[row];
      if (otherScore > score || (otherScore == score && otherIndex < index)) {
        score = otherScore;
        index = otherIndex;
      }
      if (storeAssignment(args, globalRow, scorer, score, index)) {
        atomicAdd(&changedCount, 1);
      }
    }
  }

  tmemFenceBefore();
  __syncthreads();
  tmemFenceAfter();
  if (warp == 1) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tmem), "r"(kUmmaTmemColumns));
  }
  if (threadIdx.x == 0 && changedCount > 0) {
    atomicAdd(args.changed, changedCount);
  }
#else
  (void)args;
  (void)paddedBits;
  __trap();
#endif
}

}  // namespace

std::size_t ummaSharedBytes(const int paddedBits, const int stages) {
  return 1024 + static_cast<std::size_t>(stages) * kUmmaBBytes + sizeof(std::uint32_t) * kUmmaRows * (paddedBits / 32);
}

int ummaStages(const int paddedBits) {
  constexpr std::size_t kLimit = 227 * 1024 - kUmmaStaticBytes;
  for (int stages = kUmmaMaxStages; stages >= 2; --stages) {
    if (ummaSharedBytes(paddedBits, stages) <= kLimit) {
      return stages;
    }
  }
  return 0;
}

bool ummaSupported(const int paddedBits) {
  int device = 0;
  int major  = 0;
  int minor  = 0;
  cudaCheckError(cudaGetDevice(&device));
  cudaCheckError(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  cudaCheckError(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  // The kernel is architecture-specific: it only runs on a device whose exact sm_100a / sm_103a code was built.
  const bool built = (major == 10 && minor == 0 && NVMOLKIT_KMEANS_UMMA_SM100) ||
                     (major == 10 && minor == 3 && NVMOLKIT_KMEANS_UMMA_SM103);
  return built && ummaStages(paddedBits) > 0;
}

template <KMeansMetric Metric>
void launchUmmaAssign(const AssignArgs& args, const int paddedBits, cudaStream_t stream) {
  if (args.numRows == 0) {
    return;
  }
  if (!ummaSupported(paddedBits)) {
    throw std::invalid_argument(
      "The tcgen05 k-means kernel needs an sm_100/sm_103 GPU and fingerprints of at most about 10k bits");
  }
  const int         stages      = ummaStages(paddedBits);
  const std::size_t sharedBytes = ummaSharedBytes(paddedBits, stages);
  auto              kernel      = stages == 4 ? ummaAssignKernel<Metric, 4> :
                                  stages == 3 ? ummaAssignKernel<Metric, 3> :
                                                ummaAssignKernel<Metric, 2>;
  cudaCheckError(
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes)));
  const int blocks = (args.numRows + kUmmaRows - 1) / kUmmaRows;
  kernel<<<blocks, kUmmaThreads, sharedBytes, stream>>>(args, paddedBits);
  cudaCheckError(cudaGetLastError());
}

template void launchUmmaAssign<KMeansMetric::Cosine>(const AssignArgs&, int, cudaStream_t);
template void launchUmmaAssign<KMeansMetric::Tanimoto>(const AssignArgs&, int, cudaStream_t);

}  // namespace nvMolKit::detail
