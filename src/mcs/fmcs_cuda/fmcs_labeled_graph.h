// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FMCS_CUDA_FMCS_LABELED_GRAPH_H
#define FMCS_CUDA_FMCS_LABELED_GRAPH_H

#include <cstdint>
#include <vector>

#include "mcs_common/mcs_types.cuh"

namespace mcs::fmcs {

/// Host-side graph input with explicit atom and bond labels.
struct LabeledGraph {
  Graph                      graph;
  std::vector<std::uint16_t> vertexLabels;
  std::vector<std::uint16_t> edgeLabels;
};

}  // namespace mcs::fmcs

#endif  // FMCS_CUDA_FMCS_LABELED_GRAPH_H
