// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_ETKDG_STAGE_AIO_MINIMIZATION_H
#define NVMOLKIT_ETKDG_STAGE_AIO_MINIMIZATION_H

#include <GraphMol/DistGeomHelpers/Embedder.h>

#include "src/etkdg_impl.h"
#include "src/forcefields/aio_etkdg_batched_forcefield.h"
#include "src/minimizer/bfgs_minimize.h"

namespace nvMolKit::detail {

//! RDKit's all-in-one ETKDG minimization: DG first, then ET and K terms on the
//! same forcefield, followed by the unified minimization energy check.
class ETKDGAllInOneMinimizationStage final : public ETKDGStage {
 public:
  ETKDGAllInOneMinimizationStage(const std::vector<const RDKit::ROMol*>&     mols,
                                 const std::vector<EmbedArgs>&               eargs,
                                 const RDKit::DGeomHelpers::EmbedParameters& embedParams,
                                 const ETKDGContext&                         ctx,
                                 cudaStream_t                                stream = nullptr);

  void        execute(ETKDGContext& ctx) override;
  std::string name() const override { return "All-in-one ETKDG Minimization"; }

 private:
  std::vector<DistGeom::AllInOneForceContribsHost> systemContribs_;
  DistGeom::BatchedMolecularSystemHost             dgSystemHost_;
  BatchedForcefieldMetadata                        metadata_;
  BfgsBatchMinimizer                               minimizer_;
  AsyncDeviceVector<double>                        gradients_;
  AsyncDeviceVector<double>                        energies_;
  double                                           optimizerForceTol_;
  int                                              firstPhaseIterations_ = 1;
  cudaStream_t                                     stream_               = nullptr;
};

//! RDKit AIO post-minimization validation for linear-angle and improper K terms.
class ETKDGKTermCheckStage final : public ETKDGStage {
 public:
  ETKDGKTermCheckStage(const std::vector<EmbedArgs>& eargs, const ETKDGContext& ctx, cudaStream_t stream = nullptr);
  void        execute(ETKDGContext& ctx) override;
  std::string name() const override { return "K-term check"; }

 private:
  AsyncDeviceVector<int>    angleIdx1_, angleIdx2_, angleIdx3_, angleSystemIdx_;
  AsyncDeviceVector<double> angleMin_, angleMax_, angleForceConstant_;
  AsyncDeviceVector<int>    improperIdx1_, improperIdx2_, improperIdx3_, improperIdx4_, improperSystemIdx_;
  AsyncDeviceVector<double> improperC0_, improperC1_, improperC2_, improperForceConstant_;
  AsyncDeviceVector<int>    numCenters_;
  AsyncDeviceVector<double> systemEnergies_;
  cudaStream_t              stream_ = nullptr;
};

}  // namespace nvMolKit::detail

#endif  // NVMOLKIT_ETKDG_STAGE_AIO_MINIMIZATION_H
