// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_GAUSSIAN_TORSION_BATCHED_FORCEFIELD_H
#define NVMOLKIT_GAUSSIAN_TORSION_BATCHED_FORCEFIELD_H

#include "src/forcefields/batched_forcefield.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {
namespace DistGeom {

//! Gaussian torsions with shared, contiguous energy and derivative lookup tables.
struct GaussianTorsionContribsHost {
  std::vector<int>    idx1, idx2, idx3, idx4, systemIdx;
  std::vector<int>    tableStarts, tableSizes;
  std::vector<double> scaling, energies, derivatives;
};

}  // namespace DistGeom

//! Adds Gaussian torsion contributions to an existing batched force field.
//! The base force field must outlive this adapter and use the same stream for uploads and evaluation.
class GaussianTorsionBatchedForcefield final : public BatchedForcefield {
 public:
  GaussianTorsionBatchedForcefield(BatchedForcefield&                           base,
                                   const DistGeom::GaussianTorsionContribsHost& contributions,
                                   cudaStream_t                                 stream = nullptr);
  cudaError_t computeEnergy(double*        energies,
                            const double*  positions,
                            const uint8_t* activeSystems = nullptr,
                            cudaStream_t   stream        = nullptr) override;
  cudaError_t computeGradients(double*        gradients,
                               const double*  positions,
                               const uint8_t* activeSystems = nullptr,
                               cudaStream_t   stream        = nullptr) override;
  void        setEnabled(bool enabled) { enabled_ = enabled; }

 private:
  BatchedForcefield&        base_;
  AsyncDeviceVector<int>    idx1_, idx2_, idx3_, idx4_, systemIdx_, tableStarts_, tableSizes_;
  AsyncDeviceVector<double> scaling_, energies_, derivatives_;
  bool                      enabled_ = true;
};

}  // namespace nvMolKit
#endif  // NVMOLKIT_GAUSSIAN_TORSION_BATCHED_FORCEFIELD_H
