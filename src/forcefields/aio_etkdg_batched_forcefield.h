// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_AIO_ETKDG_BATCHED_FORCEFIELD_H
#define NVMOLKIT_AIO_ETKDG_BATCHED_FORCEFIELD_H

#include "src/forcefields/batched_forcefield.h"
#include "src/forcefields/dg_batched_forcefield.h"
#include "src/forcefields/dist_geom.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

//! Batched force field for RDKit's all-in-one ETKDG coordinate refinement.
class AllInOneETKDGBatchedForcefield final : public BatchedForcefield {
 public:
  AllInOneETKDGBatchedForcefield(const DistGeom::BatchedMolecularSystemHost&             dgSystemHost,
                                 const std::vector<DistGeom::AllInOneForceContribsHost>& systemContribs,
                                 const std::vector<int>&                                 atomStartsHost,
                                 BatchedForcefieldMetadata                               metadata = {},
                                 cudaStream_t                                            stream   = nullptr);

  cudaError_t computeEnergy(double*        energyOuts,
                            const double*  positions,
                            const uint8_t* activeSystemMask = nullptr,
                            cudaStream_t   stream           = nullptr) override;

  cudaError_t computeGradients(double*        grad,
                               const double*  positions,
                               const uint8_t* activeSystemMask = nullptr,
                               cudaStream_t   stream           = nullptr) override;

  //! Enables the ET torsion and K-planarity contributions for phase two.
  void setTorsionTermsEnabled(bool enabled) { torsionTermsEnabled_ = enabled; }

 private:
  DGBatchedForcefield    dgForcefield_;
  AsyncDeviceVector<int> atomStartsDevice_;

  AsyncDeviceVector<int>    harmonicIdx1_;
  AsyncDeviceVector<int>    harmonicIdx2_;
  AsyncDeviceVector<double> harmonicMinLen_;
  AsyncDeviceVector<double> harmonicMaxLen_;
  AsyncDeviceVector<double> harmonicForceConstant_;
  AsyncDeviceVector<int>    harmonicSystemIdx_;

  AsyncDeviceVector<int>    angleIdx1_;
  AsyncDeviceVector<int>    angleIdx2_;
  AsyncDeviceVector<int>    angleIdx3_;
  AsyncDeviceVector<double> angleMin_;
  AsyncDeviceVector<double> angleMax_;
  AsyncDeviceVector<double> angleForceConstant_;
  AsyncDeviceVector<int>    angleSystemIdx_;

  AsyncDeviceVector<int>    torsionIdx1_;
  AsyncDeviceVector<int>    torsionIdx2_;
  AsyncDeviceVector<int>    torsionIdx3_;
  AsyncDeviceVector<int>    torsionIdx4_;
  AsyncDeviceVector<double> torsionForceConstants_;
  AsyncDeviceVector<int>    torsionSigns_;
  AsyncDeviceVector<int>    torsionSystemIdx_;

  AsyncDeviceVector<int>    planarityIdx1_;
  AsyncDeviceVector<int>    planarityIdx2_;
  AsyncDeviceVector<int>    planarityIdx3_;
  AsyncDeviceVector<int>    planarityIdx4_;
  AsyncDeviceVector<double> planarityForceConstant_;
  AsyncDeviceVector<int>    planaritySystemIdx_;

  bool torsionTermsEnabled_ = false;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_AIO_ETKDG_BATCHED_FORCEFIELD_H
