// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMOLKIT_PREPARED_MINI_BATCH_H
#define NVMOLKIT_PREPARED_MINI_BATCH_H

#include <memory>
#include <vector>

#include "src/substruct/minibatch_planner.h"
#include "src/substruct/molecules.h"
#include "src/substruct/pinned_buffer_pool.h"
#include "src/substruct/thread_worker_context.h"

namespace nvMolKit {

/**
 * @brief One mini-batch of (target, query) pairs ready for GPU dispatch.
 *
 * Produced by SubstructWorkload::preprocess and consumed by
 * SubstructWorkload::dispatchAndCopyBack / postprocess. The pinned buffer is
 * borrowed from PinnedHostBufferPool and must be released exactly once: either
 * by postprocess on the success path, or by ~PreparedMiniBatch on the error /
 * abort path (when the queue's destructor drops the in-flight unique_ptr
 * before any runner postprocesses it).
 *
 * Postprocess opts out of the auto-release by clearing pinnedBuffer to nullptr
 * after handing the buffer back to the pool itself.
 */
struct PreparedMiniBatch {
  std::shared_ptr<MoleculesHost>    targetsHost;
  std::shared_ptr<std::vector<int>> targetOriginalIndices;
  std::shared_ptr<std::vector<int>> targetAtomCounts;
  ThreadWorkerContext               ctx;
  MiniBatchPlan                     plan;
  PinnedHostBuffer*                 pinnedBuffer = nullptr;
  PinnedHostBufferPool*             pool         = nullptr;

  PreparedMiniBatch() = default;

  PreparedMiniBatch(const PreparedMiniBatch&)            = delete;
  PreparedMiniBatch& operator=(const PreparedMiniBatch&) = delete;
  PreparedMiniBatch(PreparedMiniBatch&&)                 = delete;
  PreparedMiniBatch& operator=(PreparedMiniBatch&&)      = delete;

  ~PreparedMiniBatch() {
    if (pool != nullptr && pinnedBuffer != nullptr) {
      pool->release(pinnedBuffer);
    }
  }
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_PREPARED_MINI_BATCH_H
