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

#ifndef FMCS_CUDA_FMCS_DEBUG_CUH
#define FMCS_CUDA_FMCS_DEBUG_CUH

#include <cstdio>

namespace mcs {
namespace fmcs {

/// Compile-time master switch for fMCS debug instrumentation.  Every
/// debug print and watchdog in the dispatch/kernel is guarded by
/// @c if @c constexpr @c (kFmcsDebug), so flipping this to false removes
/// all of it at compile time with zero runtime cost.
constexpr bool kFmcsDebug = false;

/// Temporary coarse measurement mode for fMCS kernel work accounting.
/// Unlike kFmcsDebug, this emits only one summary line at kernel exit.
constexpr bool kFmcsMeasure         = false;
constexpr int  kFmcsMeasureMaxIters = 8192;

/// Debug-only watchdog bound on Phase 2 grow-loop iterations.  A genuine
/// device-side infinite loop never returns control to the host, so its
/// buffered device @c printf output never flushes.  Forcing the loop to
/// exit after this many iterations lets the buffered trace flush so the
/// runaway can be diagnosed.  Only consulted when @c kFmcsDebug is true.
constexpr int kFmcsDebugMaxIters = 1000;

/// Restrict per-iteration device prints to a single pair so a batch
/// launch does not interleave output from every block.
constexpr int kFmcsDebugPairIdx = 0;

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_DEBUG_CUH
