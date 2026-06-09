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

#include "src/minimizer/bfgs_types.h"
#include "src/minimizer/minimizer_api.h"
#include "src/utils/device_vector.h"
#include "src/utils/host_vector.h"

namespace nvMolKit {

class BatchedForcefield;

namespace MMFF {
struct BatchedMolecularDeviceBuffers;
}  // namespace MMFF

namespace DistGeom {
struct BatchedMolecularDeviceBuffers;
struct BatchedMolecular3DDeviceBuffers;
}  // namespace DistGeom

//! \brief Algorithm parameters for the FIRE minimizer.
//!
//! Defaults match ASE FIRE 2.0
//! (https://gitlab.com/ase/ase/-/blob/master/ase/optimize/fire2.py).
//! Working units inside the kernel are kcal/mol for energy, Å for position,
//! amu for mass and ps for time.
struct FireOptions {
  //! Defaults below are the optimum from the @c benchmarks/fire_optuna.py "gpu" study
  //! (stored in @c benchmarks/fire_optuna_gpu_v4.db) at @c maxIters=200 on the
  //! perturbed-MMFF dataset. They differ from the literal ASE FIRE2 reference values.
  double dtInit      = 0.0035256954965291066;   //!< Initial time step in picoseconds.
  double dtMinFactor = 0.00014570290330215527;  //!< Lower bound for dt as a fraction of dtInit.
  double dtMaxFactor = 5.3536466978846375;      //!< Upper bound for dt as a fraction of dtInit.

  //! \brief Maximum 2-norm of the per-step displacement vector dr = dt*v, in Å.
  //! Skipped when @ref abcCorrection is true (matches ASE FIRE2 behavior).
  double dMax = 0.6925293686798697;

  double timeStepIncrement =
    1.2751646491363886;  //!< Multiplicative dt increase factor when power has been positive for nMinForIncrease steps.
  double timeStepDecrement = 0.6158984212819867;  //!< Multiplicative dt decrease factor when power becomes negative.

  int nMinForIncrease = 3;  //!< Number of consecutive positive-power steps required before dt is allowed to grow.

  double alphaInit      = 0.2890058136581572;  //!< Initial value of the mixing coefficient alpha.
  double alphaDecrement = 0.9574425933142592;  //!< Multiplicative alpha decay applied while power stays positive.

  //! \brief When true, divide the per-coordinate force kick by the per-atom mass.
  //! Note: ASE FIRE2 implicitly uses mass = 1 in its native unit system. Enabling
  //! @ref useMass here weights the integrator by per-atom masses (a deliberate
  //! deviation from ASE).
  bool useMass = false;

  double gradTol = 1e-4;  //!< Convergence threshold on sqrt(sum(grad^2)) per system.

  //! \brief Take a half step backward when the power becomes negative
  //! (r -= 0.5 * dt * v with the post-decrement dt). Always true for ASE FIRE2;
  //! disable to recover the FIRE 1.0 reset behavior.
  bool takeHalfStepBack = true;

  //! \brief Apply the Accelerated Bias-Correction multiplier
  //! 1 / (1 - (1 - alpha)^(N+1)) to the mixer (ABC-FIRE).
  bool abcCorrection = false;

  //! \brief Detect "stuck" systems via energy plateau and declare them converged.
  //!
  //! FIRE has no analog of BFGS's MOVETOL/FUNCTOL exits, so a system that oscillates
  //! around a local minimum (or plateaus) without ever reaching @ref gradTol burns
  //! the full iteration budget. When enabled, the minimizer evaluates the energy at
  //! each convergence-poll boundary, tracks per-system min/max energy across a
  //! sliding window of @ref stuckStreakLength polls, and declares the system
  //! converged (status 0) once the windowed extrema satisfy
  //! @code
  //! (max - min) / max(|E_now|, 1) < stuckEnergyRelTol
  //! @endcode
  //! for @ref stuckStreakLength consecutive polls. Streak resets whenever a poll
  //! sees a relative energy change above the tolerance.
  bool   stuckDetectionEnabled = false;
  double stuckEnergyRelTol     = 1e-3;  //!< Relative |windowed extrema| / max(|E|, 1) tolerance.
  int    stuckStreakLength     = 3;     //!< Consecutive plateau polls required to declare stuck.
  int    stuckEvalEveryNPolls  = 1;     //!< Sample energy every Nth convergence poll (1 = every poll).
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
  std::vector<double>  velocities;
  std::vector<double>  dt;
  std::vector<double>  alpha;
  std::vector<int>     nStepsPositive;
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
                              bool               debugMode = false,
                              FireBackend        backend   = FireBackend::BATCHED);
  ~FireBatchMinimizer() override = default;

  //! \brief Resolve the effective backend for the provided batch under HYBRID selection.
  FireBackend resolveBackend(const std::vector<int>& atomStartsHost) const;

  //! \brief Initialize internal buffers for a new batch.
  //! \param atomStartsHost Host offsets for the first atom of each system.
  //! \param masses Optional pointer to per-atom masses; nullptr means use any masses set via setMasses().
  //! \param activeThisStage Optional uint8_t mask (1 = active). When nullptr all systems start active.
  //! \param effectiveBackend Selects which backend's auxiliary buffers to materialize.
  //!        Pass the value returned by ::resolveBackend so HYBRID is collapsed first.
  void initialize(const std::vector<int>& atomStartsHost,
                  const double*           masses           = nullptr,
                  const uint8_t*          activeThisStage  = nullptr,
                  FireBackend             effectiveBackend = FireBackend::BATCHED);

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

  //! \brief Run MMFF FIRE minimization through the per-molecule kernel.
  //! \pre Backend must be ::FireBackend::PER_MOLECULE or ::FireBackend::HYBRID resolving
  //!      to PER_MOLECULE for this batch.
  //! \pre ::FireOptions::stuckDetectionEnabled must be false (per-mol path does not
  //!      support energy-plateau detection; @c std::runtime_error is thrown otherwise).
  bool minimizeWithMMFF(int                                  numIters,
                        double                               gradTol,
                        const std::vector<int>&              atomStartsHost,
                        MMFF::BatchedMolecularDeviceBuffers& systemDevice,
                        const uint8_t*                       activeThisStage = nullptr);

  //! \brief Run ETK FIRE minimization through the per-molecule kernel.
  bool minimizeWithETK(int                                        numIters,
                       double                                     gradTol,
                       const std::vector<int>&                    atomStartsHost,
                       const AsyncDeviceVector<int>&              atomStarts,
                       AsyncDeviceVector<double>&                 positions,
                       DistGeom::BatchedMolecular3DDeviceBuffers& systemDevice,
                       const uint8_t*                             activeThisStage = nullptr);

  //! \brief Run DG FIRE minimization through the per-molecule kernel.
  bool minimizeWithDG(int                                      numIters,
                      double                                   gradTol,
                      const std::vector<int>&                  atomStartsHost,
                      const AsyncDeviceVector<int>&            atomStarts,
                      AsyncDeviceVector<double>&               positions,
                      DistGeom::BatchedMolecularDeviceBuffers& systemDevice,
                      double                                   chiralWeight,
                      double                                   fourthDimWeight,
                      const uint8_t*                           activeThisStage = nullptr);

  const std::vector<FireDebugOutput>& debugOutputs() const { return debugOutputs_; }

  //! \brief Cadence (in iterations) at which the minimize() loop reads the
  //! still-running system count back to the host. Default 8. Set to 1 to mimic
  //! the legacy synchronous behavior.
  //! \note Only the BATCHED backend uses this; per-molecule kernels iterate
  //! entirely device-side and ignore the poll interval.
  void setConvergencePollInterval(int interval);

  //! \brief Read back internal per-system state for testing.
  FireInternalState snapshotInternalState() const;

  //! \brief Number of currently-active systems (host-side cached).
  int numActiveSystemsHost() const { return lastKnownNumUnfinished_; }

  //! \brief Forget any cached batch state so the next @c initialize() call resets all
  //! per-system convergence state (statuses, streak counters, etc.). Use before starting
  //! a new minimization session on the same minimizer instance when the active-mask
  //! contents may have changed (the address comparison alone cannot detect that).
  void resetContinuationCache();

 private:
  void launchPreKick(double                        gradTol,
                     const AsyncDeviceVector<int>& atomStarts,
                     AsyncDeviceVector<double>&    positions,
                     AsyncDeviceVector<double>&    grad,
                     int                           launchBlocks,
                     bool                          isFirstStep);
  void launchPostKick(double                        gradTol,
                      const AsyncDeviceVector<int>& atomStarts,
                      AsyncDeviceVector<double>&    positions,
                      AsyncDeviceVector<double>&    grad,
                      int                           launchBlocks);
  void compactActiveAsync();
  int  readbackNumUnfinished();

  //! \brief Copy @p statuses_ to host and report whether all formerly-active
  //! systems are now converged. Mirrors the @c checkConvergence helper used by
  //! the BFGS per-mol path.
  bool checkPerMolConvergence();

  int          dataDim_;
  FireOptions  fireOptions_;
  cudaStream_t stream_;
  int          step_                    = 0;
  bool         debugMode_               = false;
  int          numSystems_              = 0;
  int          convergencePollInterval_ = 8;
  int          lastKnownNumUnfinished_  = 0;
  FireBackend  backend_                 = FireBackend::BATCHED;

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

  //! Cached last-call signature for detecting continuation calls (same batch + same active mask)
  //! from helpers like @c repeatUntilConverged. When set, @c initialize() preserves
  //! per-system convergence state (statuses, streak counters, convergeReason) so that
  //! systems that already converged in the previous call are not re-run.
  bool           hasInitializedBatch_   = false;
  int            cachedNumSystems_      = -1;
  int            cachedTotalAtoms_      = -1;
  const uint8_t* cachedActiveThisStage_ = nullptr;
  const double*  cachedMasses_          = nullptr;

  //! Per-system state for energy-plateau stuck detection. ``energyMinStreak_`` and
  //! ``energyMaxStreak_`` track the windowed extrema while ``stuckStreak_`` counts
  //! consecutive plateau polls; all reset when the relative tolerance is violated.
  AsyncDeviceVector<double>  energyMinStreak_;
  AsyncDeviceVector<double>  energyMaxStreak_;
  AsyncDeviceVector<int32_t> stuckStreak_;
  int                        pollsSinceLastEnergyEval_ = 0;

  //! Per-system convergence reason for diagnostics: 0=active, 1=grad-tol, 2=stuck-plateau.
  AsyncDeviceVector<uint8_t> convergeReason_;

  // Per-molecule kernel data (used when backend_ == PER_MOLECULE / HYBRID resolves to it).
  int                       maxAtomsInBatch_ = 0;  //!< Largest molecule in batch (for kernel dispatch).
  std::vector<int>          activeMolIds_;         //!< Active molecule IDs (host).
  AsyncDeviceVector<int>    activeMolIdsDevice_;   //!< Device copy of @c activeMolIds_.
  PinnedHostVector<uint8_t> activeHost_;           //!< Pinned scratch for caller-supplied active mask.
  PinnedHostVector<uint8_t> convergenceHost_;      //!< Pinned scratch for status readback.
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_FIRE_MINIMIZER_H
