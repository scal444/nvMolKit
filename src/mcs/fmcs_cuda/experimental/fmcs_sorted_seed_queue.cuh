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
//
// EXPERIMENTAL: RDKit-parity, bond-count-sorted seed scheduling on top of
// SeedQueue.  The active kernel uses the LIFO push/pop stack discipline;
// these sorted-insert / pop-front operations are retained for a future
// SEED_GROW-ordered scheduling path and have no production callers.

#ifndef FMCS_CUDA_EXPERIMENTAL_FMCS_SORTED_SEED_QUEUE_CUH
#define FMCS_CUDA_EXPERIMENTAL_FMCS_SORTED_SEED_QUEUE_CUH

#include "fmcs_cuda/fmcs_seed_queue.cuh"
#include "mcs_common/mcs_cooperative_copy.cuh"

namespace mcs {
namespace fmcs {

/// Within-thread: RDKit SeedSet::add analogue for single-producer code.
/// Inserts @p element before the first stored seed with fewer bonds, thereby
/// keeping the active prefix sorted by descending bond count while preserving
/// insertion order among equal-size seeds.  This is intentionally serial and
/// shifts global-memory queue entries; use only for RDKit-parity seed-grow
/// scheduling, not for concurrent stack-style work.
template<class Element>
__device__ __forceinline__ bool insertSortedByBondsWithinThread(
    SeedQueue<Element, ThreadBlockScope>& queue,
    const Element& element) {
  const int top = queue.size();
  if (top >= queue.capacity()) return false;
  int insertAt = top;
  for (int i = 0; i < top; ++i) {
    if (queue.slot(i).seed.numBonds < element.seed.numBonds) {
      insertAt = i;
      break;
    }
  }
  for (int i = top; i > insertAt; --i) {
    queue.slot(i) = queue.slot(i - 1);
  }
  queue.slot(insertAt) = element;
  queue.setSizeWithinThread(top + 1);
  return true;
}

/// Within-thread: pop the front seed from the RDKit-style sorted list.
/// Remaining entries are shifted left to preserve order.
template<class Element>
__device__ __forceinline__ bool popFrontWithinThread(
    SeedQueue<Element, ThreadBlockScope>& queue,
    Element& outElement) {
  const int top = queue.size();
  if (top <= 0) return false;
  outElement = queue.slot(0);
  for (int i = 1; i < top; ++i) {
    queue.slot(i - 1) = queue.slot(i);
  }
  queue.setSizeWithinThread(top - 1);
  return true;
}

/// Cooperative counterpart of @ref insertSortedByBondsWithinThread: lane 0
/// locates the sorted insertion point, then the whole group shifts entries
/// and writes @p element via @ref warpCopy.
template<class GroupT, class QueuedT>
__device__ __forceinline__ bool insertSortedByBondsCooperative(
    const GroupT& group,
    SeedQueue<QueuedT, ThreadBlockScope>& queue,
    const QueuedT& element) {
  const int groupRank = static_cast<int>(group.thread_rank());
  int oldSize = 0;
  int insertAt = 0;
  int ok = 1;
  if (groupRank == 0) {
    oldSize = queue.size();
    ok = oldSize < queue.capacity() ? 1 : 0;
    insertAt = oldSize;
    if (ok) {
      for (int i = 0; i < oldSize; ++i) {
        if (queue.slot(i).seed.numBonds < element.seed.numBonds) {
          insertAt = i;
          break;
        }
      }
    }
  }
  oldSize = group.shfl(oldSize, 0);
  insertAt = group.shfl(insertAt, 0);
  ok = group.shfl(ok, 0);
  if (!ok) return false;

  for (int i = oldSize; i > insertAt; --i) {
    warpCopy(group, &queue.slot(i), &queue.slot(i - 1), sizeof(QueuedT));
    group.sync();
  }
  warpCopy(group, &queue.slot(insertAt), &element, sizeof(QueuedT));
  group.sync();
  if (groupRank == 0) queue.setSizeWithinThread(oldSize + 1);
  group.sync();
  return true;
}

/// Cooperative counterpart of @ref popFrontWithinThread: the whole group
/// copies out the front entry via @ref warpCopy and shifts the remainder
/// left.
template<class GroupT, class QueuedT>
__device__ __forceinline__ bool popFrontCooperative(
    const GroupT& group,
    SeedQueue<QueuedT, ThreadBlockScope>& queue,
    QueuedT& outElement) {
  const int groupRank = static_cast<int>(group.thread_rank());
  int oldSize = 0;
  if (groupRank == 0) oldSize = queue.size();
  oldSize = group.shfl(oldSize, 0);
  if (oldSize <= 0) return false;

  warpCopy(group, &outElement, &queue.slot(0), sizeof(QueuedT));
  group.sync();
  for (int i = 1; i < oldSize; ++i) {
    warpCopy(group, &queue.slot(i - 1), &queue.slot(i), sizeof(QueuedT));
    group.sync();
  }
  if (groupRank == 0) queue.setSizeWithinThread(oldSize - 1);
  group.sync();
  return true;
}

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_EXPERIMENTAL_FMCS_SORTED_SEED_QUEUE_CUH
