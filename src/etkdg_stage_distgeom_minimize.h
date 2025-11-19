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

#ifndef NVMOLKIT_ETKDG_STAGE_DISTGEOM_MINIMIZE_H
#define NVMOLKIT_ETKDG_STAGE_DISTGEOM_MINIMIZE_H

#include <GraphMol/DistGeomHelpers/Embedder.h>

#include "bfgs_minimize.h"
#include "dist_geom.h"
#include "etkdg_impl.h"

using ::nvMolKit::detail::EmbedArgs;
using ::nvMolKit::detail::ETKDGContext;
using ::nvMolKit::detail::ETKDGStage;

namespace nvMolKit {
namespace detail {

constexpr double MAX_MINIMIZED_E_PER_ATOM = 0.05;  // Maximum energy per atom threshold

/// Generic distance geometry minimization stage with configurable weights and parameters
class DistGeomMinimizeStage : public ETKDGStage {
 public:
  /**
   * @brief Construct a distance geometry minimization stage
   * 
   * @param mols Vector of molecules to minimize
   * @param eargs Vector of embed arguments
   * @param embedParam Embedding parameters
   * @param ctx ETKDG context
   * @param minimizer BFGS minimizer
   * @param chiralWeight Weight for chiral constraints
   * @param fourthDimWeight Weight for fourth dimension constraints
   * @param maxIters Maximum number of iterations per minimization cycle
   * @param checkEnergy Whether to check energy per atom after minimization
   * @param stageName Name of this stage for logging/debugging
   * @param stream CUDA stream
   */
  DistGeomMinimizeStage(const std::vector<const RDKit::ROMol*>&     mols,
                        const std::vector<EmbedArgs>&               eargs,
                        const RDKit::DGeomHelpers::EmbedParameters& embedParam,
                        ETKDGContext&                               ctx,
                        BfgsBatchMinimizer&                         minimizer,
                        double                                      chiralWeight,
                        double                                      fourthDimWeight,
                        int                                         maxIters,
                        bool                                        checkEnergy,
                        const std::string&                          stageName,
                        cudaStream_t                                stream = nullptr);
  
  void        execute(ETKDGContext& ctx) override;
  std::string name() const override { return stageName_; }

  nvMolKit::DistGeom::BatchedMolecularDeviceBuffers molSystemDevice;
  nvMolKit::DistGeom::BatchedMolecularSystemHost    molSystemHost;
  const RDKit::DGeomHelpers::EmbedParameters&       embedParam_;
  BfgsBatchMinimizer&                               minimizer_;
  int                                               maxIters_;
  bool                                              checkEnergy_;
  std::string                                       stageName_;
  cudaStream_t                                      stream_;
};

}  // namespace detail
}  // namespace nvMolKit
#endif  // NVMOLKIT_ETKDG_STAGE_DISTGEOM_MINIMIZE_H

