// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_MCS_RDKIT_ADAPTER_H
#define NVMOLKIT_MCS_RDKIT_ADAPTER_H

#include <string>

#include "src/mcs/fmcs_cuda/fmcs.cuh"
#include "src/mcs/mcs_types.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit

namespace nvMolKit::mcs_detail {

struct LabeledGraphPair {
  mcs::fmcs::LabeledGraph graphA;
  mcs::fmcs::LabeledGraph graphB;
};

bool usesAtomLabels(const MCSParameters& params);

bool usesBondLabels(const MCSParameters& params);

LabeledGraphPair buildLabeledGraphPair(const RDKit::ROMol& molA, const RDKit::ROMol& molB, const MCSParameters& params);

bool shouldFallbackToRDKit(const RDKit::ROMol&  molA,
                           const RDKit::ROMol&  molB,
                           const MCSParameters& params,
                           std::string&         reason);

MCSResult runRDKitFallback(const RDKit::ROMol& molA, const RDKit::ROMol& molB, const MCSParameters& params);

MCSExecutionStats convertExecutionStats(const mcs::fmcs::ExecutionStats& in);

MCSResult convertGpuResult(const RDKit::ROMol&              molA,
                           const RDKit::ROMol&              molB,
                           const mcs::MCSResult&            gpuResult,
                           const MCSParameters&             params,
                           float                            elapsedMs,
                           const mcs::fmcs::ExecutionStats* kernelTimings,
                           const mcs::fmcs::ExecutionStats* executionStats);

}  // namespace nvMolKit::mcs_detail

#endif  // NVMOLKIT_MCS_RDKIT_ADAPTER_H
