// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_MCS_COMPILE_FLAGS_H
#define NVMOLKIT_MCS_COMPILE_FLAGS_H

#ifndef NVMOLKIT_ENABLE_MCS_TIMINGS
#define NVMOLKIT_ENABLE_MCS_TIMINGS 1
#endif

#ifndef NVMOLKIT_ENABLE_MCS_STATS
#define NVMOLKIT_ENABLE_MCS_STATS 1
#endif

namespace nvMolKit {

inline constexpr bool kMCSCollectTimingsEnabled =
    NVMOLKIT_ENABLE_MCS_TIMINGS != 0;
inline constexpr bool kMCSCollectStatsEnabled =
    NVMOLKIT_ENABLE_MCS_STATS != 0;

}  // namespace nvMolKit

#endif  // NVMOLKIT_MCS_COMPILE_FLAGS_H
