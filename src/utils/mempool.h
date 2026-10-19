// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMOLKIT_MEMPOOL_H
#define NVMOLKIT_MEMPOOL_H

#include <cstddef>
#include <cstdint>
#include <optional>

namespace nvMolKit {

//! Environment variable that overrides the release threshold nvMolKit sets on each device's default memory pool.
//! A byte count sets the threshold, and 0 leaves the pool untouched.
inline constexpr const char* kMemPoolReleaseThresholdEnvVar = "NVMOLKIT_MEMPOOL_RELEASE_THRESHOLD";

//! Release threshold applied when the environment variable is unset. Up to this many bytes of freed stream-ordered
//! memory stay reserved in the pool across synchronizations instead of returning to the driver, so repeated small
//! calls do not pay to regrow the pool. Bounded so that idle reservations stay small next to other users of the GPU.
inline constexpr std::uint64_t kDefaultMemPoolReleaseThreshold = std::uint64_t{256} << 20;

//! Parses the environment variable value. Returns the threshold to apply, or nullopt to leave the pool untouched.
std::optional<std::uint64_t> parseMemPoolReleaseThreshold(const char* value);

//! Sets the release threshold of the given device's default memory pool, unless the pool already has a non-zero
//! threshold (set by the user or another library) or the environment variable opts out. Never throws.
void configureDefaultMemPool(int device);

//! Runs configureDefaultMemPool once per device for the current device. Cheap after the first call on a device.
void ensureDefaultMemPoolConfigured();

//! Bytes reserved by the current device's current memory pool that no allocation is using. Stream-ordered
//! allocations reuse them before requesting more memory from the driver.
std::size_t getMemPoolIdleBytes();

}  // namespace nvMolKit

#endif  // NVMOLKIT_MEMPOOL_H
