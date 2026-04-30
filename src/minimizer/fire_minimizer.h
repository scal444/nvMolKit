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

class BatchedForcefield;


//! \brief Algorithm parameters for the FIRE minimizer.
//!
//! Defaults match ASE FIRE 2.0
//! (https://gitlab.com/ase/ase/-/blob/master/ase/optimize/fire2.py).
//! Working units inside the kernel are kcal/mol for energy, Å for position,
//! amu for mass and ps for time.
struct FireOptions {
  double dtInit      = 0.001;  //!< Initial time step in picoseconds (1 fs).
  double dtMinFactor = 0.002;  //!< Lower bound for dt as a fraction of dtInit (matches ASE FIRE2 dtmin/dt).
  double dtMaxFactor = 10.0;   //!< Upper bound for dt as a fraction of dtInit (matches ASE FIRE2 dtmax/dt).

  //! \brief Maximum 2-norm of the per-step displacement vector dr = dt*v, in Å.
  //! Skipped when @ref abcCorrection is true (matches ASE FIRE2 behavior).
  double dMax = 0.2;

  double timeStepIncrement = 1.1;  //!< Multiplicative dt increase factor when power has been positive for nMinForIncrease steps.
  double timeStepDecrement = 0.5;  //!< Multiplicative dt decrease factor when power becomes negative.

  int nMinForIncrease = 20;  //!< Number of consecutive positive-power steps required before dt is allowed to grow.

  double alphaInit      = 0.25;  //!< Initial value of the mixing coefficient alpha.
  double alphaDecrement = 0.99;  //!< Multiplicative alpha decay applied while power stays positive.

  //! \brief When true, divide the per-coordinate force kick by the per-atom mass.
  //! Note: ASE FIRE2 implicitly uses mass = 1 in its native unit system. Enabling
  //! @ref useMass here weights the integrator by per-atom masses (a deliberate
  //! deviation from ASE).
  bool useMass = true;

  double gradTol = 1e-4;  //!< Convergence threshold on sqrt(sum(grad^2)) per system.

  //! \brief Take a half step backward when the power becomes negative
  //! (r -= 0.5 * dt * v with the post-decrement dt). Always true for ASE FIRE2;
  //! disable to recover the FIRE 1.0 reset behavior.
  bool takeHalfStepBack = true;

  //! \brief Apply the Accelerated Bias-Correction multiplier
  //! 1 / (1 - (1 - alpha)^(N+1)) to the mixer (ABC-FIRE).
  bool abcCorrection = false;
};

//! \brief Per-system per-iteration debug snapshot recorded when the minimizer is
//! constructed in debug mode.
struct FireDebugOutput {
  std::vector<double> alphas;
  std::vector<double> dt;
  std::vector<double> powers;
  std::vector<double> energies;
};

//! \brief Snapshot of internal per-system state, exposed for testing.
struct FireInternalState {
  std::vector<double> velocities;
  std::vector<double> dt;
  std::vector<double> alpha;
  std::vector<int>    nStepsPositive;
  std::vector<uint8_t> statuses;
};

//! \brief Batched FIRE 2.0 minimizer.
//!
//! Implements the ASE FIRE 2.0 algorithm with a single semi-implicit Euler
//! integrator, an optional ABC-FIRE mixer correction, and a post-mixer
//! 2-norm displacement clip. Each system in the batch maintains its own
//! per-system state (dt, alpha, nStepsPositive, velocities) and converges
//! independently. Inactive systems (passed via @p activeThisStage at
//! construction or marked converged during minimization) are not touched.
class FireBatchMinimizer final : public BatchMinimizer {
 public:
  explicit FireBatchMinimizer(int                dataDim   = 3,
                              const FireOptions& options   = FireOptions(),
                              cudaStream_t       stream    = nullptr,
                              bool               debugMode = false);
  ~FireBatchMinimizer() override = default;

  //! \brief Initialize internal buffers for a new batch.
  //! \param atomStartsHost Host offsets for the first atom of each system.
  //! \param masses Optional pointer to per-atom masses; nullptr means use any masses set via setMasses().
  //! \param activeThisStage Optional uint8_t mask (1 = active). When nullptr all systems start active.
  void initialize(const std::vector<int>& atomStartsHost,
                  const double*           masses          = nullptr,
                  const uint8_t*          activeThisStage = nullptr);

  //! \brief Provide per-atom masses to be used on the next initialization
  //! when explicit masses are not supplied. Passing an empty vector clears
  //! previously stored masses.
  void setMasses(const std::vector<double>& masses);

  //! \brief Run a single FIRE step synchronously.
  //! \return True if all (initially-active) systems are now converged.
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

  //! \brief Minimize using a BatchedForcefield directly (matches the
  //! BfgsBatchMinimizer overload). The forcefield's compute hooks are wrapped
  //! into EnergyFunctor / GradFunctor that honor @p activeSystemMask.
  bool minimize(int                        numIters,
                double                     gradTol,
                BatchedForcefield&         ff,
                AsyncDeviceVector<double>& positions,
                AsyncDeviceVector<double>& grad,
                AsyncDeviceVector<double>& energyOuts,
                const uint8_t*             activeSystemMask = nullptr);

  const std::vector<FireDebugOutput>& debugOutputs() const { return debugOutputs_; }

  //! \brief Cadence (in iterations) at which the minimize() loop reads the
  //! still-running system count back to the host. Default 8. Set to 1 to mimic
  //! the legacy synchronous behavior.
  void setConvergencePollInterval(int interval);

  //! \brief Read back internal per-system state for testing.
  FireInternalState snapshotInternalState() const;

  //! \brief Number of currently-active systems (host-side cached).
  int numActiveSystemsHost() const { return lastKnownNumUnfinished_; }

 private:
  void launchFireKernel(double                        gradTol,
                        const AsyncDeviceVector<int>& atomStarts,
                        AsyncDeviceVector<double>&    positions,
                        AsyncDeviceVector<double>&    grad,
                        int                           launchBlocks,
                        bool                          isFirstStep);
  void compactActiveAsync();
  int  readbackNumUnfinished();

  int          dataDim_;
  FireOptions  fireOptions_;
  cudaStream_t stream_;
  int          step_                    = 0;
  bool         debugMode_               = false;
  int          numSystems_              = 0;
  int          convergencePollInterval_ = 8;
  int          lastKnownNumUnfinished_  = 0;

  AsyncDeviceVector<double> velocities_;
  AsyncDeviceVector<double> masses_;

  AsyncDeviceVector<double>  dt_;
  AsyncDeviceVector<double>  alpha_;
  AsyncDeviceVector<int>     numStepsWithPositivePower_;
  AsyncDeviceVector<uint8_t> statuses_;

  AsyncDeviceVector<uint8_t> countTempStorage_;
  AsyncDevicePtr<int>        countUnfinished_;
  PinnedHostVector<int>      loopStatusHost_;
  AsyncDeviceVector<int>     activeSystemIndices_;
  AsyncDeviceVector<int>     allSystemIndices_;

  std::vector<double> hostMasses_;

  AsyncDeviceVector<double>    debugPowers_;
  std::vector<FireDebugOutput> debugOutputs_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_FIRE_MINIMIZER_H
