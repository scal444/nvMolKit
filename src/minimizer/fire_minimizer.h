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

#ifndef NVMOLKIT_FIRE_MINIMIZER_H
#define NVMOLKIT_FIRE_MINIMIZER_H

#include <vector>
#include "device_vector.h"
#include "host_vector.h"
#include "minimizer_api.h"

namespace nvMolKit {

//! Integration schemes supported by the FIRE minimizer.
enum class FireIntegrationScheme {
  ExplicitEuler, //! Corresponds to FIRE v1.0. Not recommended
  SemiImplicitEuler //! Default FIRE 2.0 option in LAMMPs.
};

//! Algorithm parameters for the FIRE minimizer. Defaults taken from ASE
//! (https://gitlab.com/ase/ase/-/blob/master/ase/optimize/fire.py)
struct FireOptions {
  double dtInit      = 0.001;  //!< Initial time step in picoseconds (1 fs).
  double dtMinFactor = 0.02;    //!< Lower bound relative to dtInit (dt >= dtInit * dtMinFactor).
  double dtMaxFactor = 10.0;   //!< Upper bound relative to dtInit (dt <= dtInit * dtMaxFactor).

  double timeStepIncrement = 1.1;  //!< Factor to increase time step when conditions are met
  double timeStepDecrement = 0.5;  //!< Factor to decrease time step when conditions are not met

  int nMinForIncrease = 5;  //!< Number of steps with positive power before increasing time step

  double alphaInit      = 0.25;  //!< Initial value of alpha
  double alphaDecrement = 0.99;  //!< Factor to decrease alpha when conditions are met

  bool useMass = true;  //!< Whether to use per-atom masses if provided, or unit masses otherwise.

  double gradTol = 1e-4;  //!< Gradient tolerance for convergence checks.

  FireIntegrationScheme integrationScheme = FireIntegrationScheme::SemiImplicitEuler;

  bool takeHalfStepBack = false;  //!< Whether to take a half step back when power is negative. Turned on for FIRE 2.0.
};

//! Per-system debug output for the FIRE minimizer.
struct FireDebugOutput {
  std::vector<double> alphas;
  std::vector<double> dt;
  std::vector<double> powers;
  std::vector<double> energies;
};

class FireBatchMinimizer final : public BatchMinimizer {
 public:
  explicit FireBatchMinimizer(int                dataDim = 3,
                              const FireOptions& options = FireOptions(),
                              cudaStream_t       stream  = nullptr,
                              bool debugMode = false);
  ~FireBatchMinimizer() override = default;

  //! Initialize internal buffers for a new batch.
  //! @param atomStartsHost Offsets for the first atom of each system on the host.
  //! @param masses Optional pointer to per-atom masses; nullptr indicates unit masses.
  //! @param activeSystems Optional mask for active systems.
  void initialize(const std::vector<int>& atomStartsHost,
                  const double*           masses        = nullptr,
                  const uint8_t*          activeSystems = nullptr);

  //! Provide per-atom masses to be used on the next initialization when explicit masses are not supplied.
  //! Passing an empty vector clears previously stored masses.
  void setMasses(const std::vector<double>& masses);

  bool step(double                        gradTol,
            const AsyncDeviceVector<int>& atomStarts,
            AsyncDeviceVector<double>&    positions,
            AsyncDeviceVector<double>&    grad,
            const GradFunctor&            gFunc);

  bool minimize(int                           numIters,
                double                        gradTol,
                const std::vector<int>&       atomStartsHost,
                const AsyncDeviceVector<int>& atomStarts,
                AsyncDeviceVector<double>&    positions,
                AsyncDeviceVector<double>&    grad,
                AsyncDeviceVector<double>&    energyOuts,
                AsyncDeviceVector<double>&    energyBuffer,
                EnergyFunctor                 eFunc,
                GradFunctor                   gFunc,
                const uint8_t*                activeThisStage = nullptr) override;

  const std::vector<FireDebugOutput>& debugOutputs() const { return debugOutputs_; }

 private:
  void fireUpdate(double                        gradTol,
              const AsyncDeviceVector<int>& atomStarts,
              AsyncDeviceVector<double>&    positions,
              AsyncDeviceVector<double>&    grad);
  int  compactAndCountConverged();

  int          dataDim_;
  FireOptions  fireOptions_;
  cudaStream_t stream_;
  bool        debugMode_ = false;

  // Per atom * dim quantities
  AsyncDeviceVector<double> velocities_;
  AsyncDeviceVector<double> prevVelocities_;
  AsyncDeviceVector<double> masses_;

  // Per system quantities.
  AsyncDeviceVector<double>  dt_;
  AsyncDeviceVector<double>  alpha_;
  AsyncDeviceVector<int>     numStepsWithPositivePower_;
  AsyncDeviceVector<int>     numStepsWithNegativePower_;
  AsyncDeviceVector<uint8_t> statuses_;

  // Status trackers.
  AsyncDeviceVector<uint8_t> countTempStorage_;
  AsyncDevicePtr<int>        countUnfinished_;
  PinnedHostVector<int>      loopStatusHost_;
  AsyncDeviceVector<int>     activeSystemIndices_;
  AsyncDeviceVector<int>     allSystemIndices_;

  std::vector<double> hostMasses_;

  // Only allocated in debug mode when we need extra write buffers.
  AsyncDeviceVector<double> debugPowers_;
  std::vector<FireDebugOutput> debugOutputs_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_FIRE_MINIMIZER_H
