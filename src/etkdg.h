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

#ifndef NVMOLKIT_ETKDG_H
#define NVMOLKIT_ETKDG_H

#include <vector>

#include "bfgs_minimize.h"
#include "etkdg_impl.h"
#include "hardware_options.h"

namespace RDKit {
class ROMol;

namespace DGeomHelpers {
struct EmbedParameters;
}  // namespace DGeomHelpers
}  // namespace RDKit

namespace nvMolKit {

//! \brief Run GPU-accelerated ETKDG conformer embedding on a batch of molecules.
//!
//! \param mols Molecules to embed. Conformers are appended in-place.
//! \param params RDKit embed parameters; @c useRandomCoords must be true.
//! \param confsPerMolecule Number of conformers to attempt per molecule.
//! \param maxIterations Maximum ETKDG iterations; @c -1 selects an automatic value.
//! \param debugMode Enable per-stage timing/output collection.
//! \param failures Optional pointer to a per-stage / per-conformer failure tally.
//! \param stageNames Optional pointer populated with the ordered stage names matching @p failures'
//!        outer dimension. Useful for labeling failure-mode plots.
//! \param hardwareOptions Batch and threading hardware configuration.
//! \param backend BFGS kernel layout selector. Only consulted when
//!        @p minimizerKind is ::MinimizerKind::BFGS; ignored for ::MinimizerKind::FIRE
//!        (FIRE always runs through the BATCHED \ref BatchedForcefield path).
//! \param minimizerKind Selects which minimization algorithm drives the
//!        distance-geometry and ETK refinement stages. Default ::MinimizerKind::BFGS
//!        preserves historical behavior.
void embedMolecules(const std::vector<RDKit::ROMol*>&           mols,
                    const RDKit::DGeomHelpers::EmbedParameters& params,
                    int                                         confsPerMolecule = 1,
                    int                                         maxIterations    = -1,
                    bool                                        debugMode        = false,
                    std::vector<std::vector<int16_t>>*          failures         = nullptr,
                    const BatchHardwareOptions&                 hardwareOptions  = {},
                    BfgsBackend                                 backend          = BfgsBackend::HYBRID,
                    MinimizerKind                               minimizerKind    = MinimizerKind::BFGS,
                    std::vector<std::string>*                   stageNames       = nullptr);

}  // namespace nvMolKit

#endif
