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

#ifndef NVMOLKIT_BFGS_COMMON_H
#define NVMOLKIT_BFGS_COMMON_H

#include <cstdint>
#include <vector>

#include "../hardware_options.h"
#include "config.h"
#include "conformer_info.h"
#include "host_vector.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit

namespace nvMolKit {

//! Thread-local pinned memory buffers for async host-device transfers during BFGS minimization.
struct ThreadLocalBuffers {
  PinnedHostVector<double> positions;
  PinnedHostVector<double> energies;
  PinnedHostVector<double> initialPositions;

  void ensureCapacity(size_t positionsSize, size_t energiesSize);
};

/// Translate the public BatchHardwareOptions surface to a gpu_scheduler::Config.
/// Unset fields default to "all available GPUs", "spread host threads evenly
/// across them", and "one in-flight slot per runner" (matching the legacy
/// OpenMP behavior).
gpu_scheduler::Config configFromHardwareOptions(const BatchHardwareOptions& perfOptions);

/// Resolve the per-pipeline-batch conformer count from BatchHardwareOptions.
/// Returns the total conformer count when batchSize is unset / non-positive,
/// so the pipeline runs a single mini-batch per claim by default.
int resolveBatchSize(const BatchHardwareOptions& perfOptions, int totalConformers);

/// Flatten all conformers from all molecules into a single list, validating molecule pointers
/// and initializing the per-molecule energy output vectors.
std::vector<ConformerInfo> flattenConformers(const std::vector<RDKit::ROMol*>& mols,
                                             std::vector<std::vector<double>>& moleculeEnergies);

/// Write optimized positions and energies back from host buffers into the RDKit conformers.
void writeBackResults(const std::vector<ConformerInfo>& batchConformers,
                      const std::vector<uint32_t>&      conformerAtomStarts,
                      const ThreadLocalBuffers&         buffers,
                      std::vector<std::vector<double>>& moleculeEnergies);

}  // namespace nvMolKit

#endif  // NVMOLKIT_BFGS_COMMON_H
