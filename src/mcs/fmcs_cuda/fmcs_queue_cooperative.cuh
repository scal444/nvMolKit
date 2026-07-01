// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FMCS_CUDA_FMCS_QUEUE_COOPERATIVE_CUH
#define FMCS_CUDA_FMCS_QUEUE_COOPERATIVE_CUH

#include "fmcs_cuda/fmcs_seed_queue.cuh"
#include "mcs_common/mcs_cooperative_copy.cuh"

namespace mcs {
namespace fmcs {

template <class GroupT, class T>
__device__ __forceinline__ void warpAtomicStoreWords(const GroupT& group, T* dst, const T& src) {
  static_assert(sizeof(T) % sizeof(unsigned int) == 0, "atomic word copy requires 32-bit granularity");
  auto*         dstWords = reinterpret_cast<unsigned int*>(dst);
  const auto*   srcWords = reinterpret_cast<const unsigned int*>(&src);
  constexpr int kWords   = static_cast<int>(sizeof(T) / sizeof(unsigned int));
  for (int i = static_cast<int>(group.thread_rank()); i < kWords; i += static_cast<int>(group.num_threads())) {
    atomicExch(&dstWords[i], srcWords[i]);
  }
}

template <class GroupT, class T>
__device__ __forceinline__ void warpAtomicLoadWords(const GroupT& group, T& dst, T* src) {
  static_assert(sizeof(T) % sizeof(unsigned int) == 0, "atomic word copy requires 32-bit granularity");
  auto*         dstWords = reinterpret_cast<unsigned int*>(&dst);
  auto*         srcWords = reinterpret_cast<unsigned int*>(src);
  constexpr int kWords   = static_cast<int>(sizeof(T) / sizeof(unsigned int));
  for (int i = static_cast<int>(group.thread_rank()); i < kWords; i += static_cast<int>(group.num_threads())) {
    dstWords[i] = atomicAdd(&srcWords[i], 0u);
  }
}

template <class GroupT, class QueuedT>
__device__ __forceinline__ bool pushBackCooperative(const GroupT&                         group,
                                                    SeedQueue<QueuedT, ThreadBlockScope>& queue,
                                                    const QueuedT&                        element) {
  const int slot = queue.batchReserveCooperative(group, 1);
  if (slot < 0)
    return false;
  warpCopy(group, &queue.slot(slot), &element, sizeof(QueuedT));
  group.sync();
  return true;
}

template <class GroupT>
__device__ __forceinline__ void acquireQueueLockCooperative(const GroupT& group, int* queueLock) {
  if (group.thread_rank() == 0) {
    while (atomicCAS(queueLock, 0, 1) != 0) {
    }
  }
  group.sync();
  __threadfence_block();
  group.sync();
}

template <class GroupT>
__device__ __forceinline__ void releaseQueueLockCooperative(const GroupT& group, int* queueLock) {
  group.sync();
  __threadfence_block();
  group.sync();
  if (group.thread_rank() == 0) {
    atomicExch(queueLock, 0);
  }
  group.sync();
}

template <class GroupT, class QueuedT>
__device__ __forceinline__ bool pushBackLockedCooperative(const GroupT&                         group,
                                                          SeedQueue<QueuedT, ThreadBlockScope>& queue,
                                                          const QueuedT&                        element,
                                                          int*                                  queueLock,
                                                          int*                                  overflowedFlag,
                                                          int*                                  timedOutFlag,
                                                          int*                                  doneFlag) {
  const int groupRank = static_cast<int>(group.thread_rank());
  int       oldSize   = 0;
  int       ok        = 1;
  int       skip      = 0;
  acquireQueueLockCooperative(group, queueLock);
  if (groupRank == 0) {
    skip    = (atomicAdd(overflowedFlag, 0) != 0) || (atomicAdd(timedOutFlag, 0) != 0) || (atomicAdd(doneFlag, 0) != 0);
    oldSize = queue.sizeAtomic();
    ok      = (!skip && oldSize < queue.capacity()) ? 1 : 0;
  }
  oldSize = group.shfl(oldSize, 0);
  ok      = group.shfl(ok, 0);
  skip    = group.shfl(skip, 0);
  if (ok) {
    warpAtomicStoreWords(group, &queue.slot(oldSize), element);
    group.sync();
    if (groupRank == 0)
      queue.setSizeAtomicWithinThread(oldSize + 1);
  }
  releaseQueueLockCooperative(group, queueLock);
  return skip || ok;
}

template <class GroupT, class QueuedT>
__device__ __forceinline__ bool popBackLockedOrFinishCooperative(const GroupT&                         group,
                                                                 SeedQueue<QueuedT, ThreadBlockScope>& queue,
                                                                 QueuedT&                              outElement,
                                                                 int*                                  queueLock,
                                                                 int*                                  activeGroups,
                                                                 int*                                  doneFlag,
                                                                 int*                                  overflowedFlag,
                                                                 int*                                  timedOutFlag,
                                                                 bool&                                 doneOut) {
  const int groupRank = static_cast<int>(group.thread_rank());
  int       oldTop    = 0;
  int       popped    = 0;
  int       done      = 0;
  acquireQueueLockCooperative(group, queueLock);
  if (groupRank == 0) {
    const bool aborted =
      (atomicAdd(overflowedFlag, 0) != 0) || (atomicAdd(timedOutFlag, 0) != 0) || (atomicAdd(doneFlag, 0) != 0);
    if (aborted) {
      atomicExch(doneFlag, 1);
      done = 1;
    } else {
      oldTop = queue.sizeAtomic();
      if (oldTop > 0) {
        popped = 1;
      } else if (atomicAdd(activeGroups, 0) == 0) {
        atomicExch(doneFlag, 1);
        done = 1;
      }
    }
  }
  oldTop = group.shfl(oldTop, 0);
  popped = group.shfl(popped, 0);
  done   = group.shfl(done, 0);
  if (popped) {
    warpAtomicLoadWords(group, outElement, &queue.slot(oldTop - 1));
    group.sync();
    if (groupRank == 0) {
      queue.setSizeAtomicWithinThread(oldTop - 1);
      atomicAdd(activeGroups, 1);
    }
  }
  releaseQueueLockCooperative(group, queueLock);
  doneOut = done != 0;
  return popped != 0;
}

template <class GroupT>
__device__ __forceinline__ unsigned int readBestScoreCooperative(const GroupT& group, const unsigned int* bestScore) {
  unsigned int score = 0;
  if (group.thread_rank() == 0)
    score = *bestScore;
  return group.shfl(score, 0);
}

template <class GroupT> __device__ __forceinline__ bool readFlagCooperative(const GroupT& group, int* flag) {
  int value = 0;
  if (group.thread_rank() == 0)
    value = atomicAdd(flag, 0);
  value = group.shfl(value, 0);
  return value != 0;
}

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_QUEUE_COOPERATIVE_CUH
