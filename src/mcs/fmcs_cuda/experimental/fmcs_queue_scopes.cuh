// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FMCS_CUDA_EXPERIMENTAL_FMCS_QUEUE_SCOPES_CUH
#define FMCS_CUDA_EXPERIMENTAL_FMCS_QUEUE_SCOPES_CUH

namespace mcs::fmcs {

// EXPERIMENTAL: Reserved tags for potential distributed shared-memory and
// device-global queue implementations. Neither scope is implemented today.
struct ClusterScope {};
struct GridScope {};

}  // namespace mcs::fmcs

#endif  // FMCS_CUDA_EXPERIMENTAL_FMCS_QUEUE_SCOPES_CUH
