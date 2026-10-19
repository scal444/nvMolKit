// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMOLKIT_DISTGEOM_KERNELS_H
#define NVMOLKIT_DISTGEOM_KERNELS_H

#include <cuda_runtime.h>

#include <cstdint>

namespace nvMolKit {
namespace DistGeom {
cudaError_t launchDistViolationEnergyKernel(int            numDist,
                                            const int*     idx1,
                                            const int*     idx2,
                                            const double*  lb2,
                                            const double*  ub2,
                                            const double*  weight,
                                            const double*  pos,
                                            double*        energyBuffer,
                                            const int*     energyBufferStarts,
                                            const int*     atomIdxToBatchIdx,
                                            const int*     distTermStarts,
                                            const int*     atomStarts,
                                            int            dimension,
                                            const uint8_t* activeThisStage = nullptr,
                                            cudaStream_t   stream          = 0);

cudaError_t launchDistViolationGradientKernel(int            numDist,
                                              const int*     idx1,
                                              const int*     idx2,
                                              const double*  lb2,
                                              const double*  ub2,
                                              const double*  weight,
                                              const double*  pos,
                                              double*        grad,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     atomStarts,
                                              int            dimension,
                                              const uint8_t* activeThisStage = nullptr,
                                              cudaStream_t   stream          = 0);

cudaError_t launchChiralViolationEnergyKernel(int            numChiral,
                                              const int*     idx1,
                                              const int*     idx2,
                                              const int*     idx3,
                                              const int*     idx4,
                                              const double*  volLower,
                                              const double*  volUpper,
                                              double         weight,
                                              const double*  pos,
                                              double*        energyBuffer,
                                              const int*     energyBufferStarts,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     chiralTermStarts,
                                              const int*     atomStarts,
                                              int            dimension,
                                              const uint8_t* activeThisStage = nullptr,
                                              cudaStream_t   stream          = 0);

cudaError_t launchChiralViolationGradientKernel(int            numChiral,
                                                const int*     idx1,
                                                const int*     idx2,
                                                const int*     idx3,
                                                const int*     idx4,
                                                const double*  volLower,
                                                const double*  volUpper,
                                                double         weight,
                                                const double*  pos,
                                                double*        grad,
                                                const int*     atomIdxToBatchIdx,
                                                const int*     atomStarts,
                                                int            dimension,
                                                const uint8_t* activeThisStage = nullptr,
                                                cudaStream_t   stream          = 0);

cudaError_t launchFourthDimEnergyKernel(int            numFD,
                                        const int*     idx,
                                        double         weight,
                                        const double*  pos,
                                        double*        energyBuffer,
                                        const int*     energyBufferStarts,
                                        const int*     atomIdxToBatchIdx,
                                        const int*     fourthTermStarts,
                                        const int*     atomStarts,
                                        int            dimension,
                                        const uint8_t* activeThisStage = nullptr,
                                        cudaStream_t   stream          = 0);

cudaError_t launchFourthDimGradientKernel(int            numFD,
                                          const int*     idx,
                                          double         weight,
                                          const double*  pos,
                                          double*        grad,
                                          const int*     atomIdxToBatchIdx,
                                          const int*     atomStarts,
                                          int            dimension,
                                          const uint8_t* activeThisStage = nullptr,
                                          cudaStream_t   stream          = 0);

// Experimental torsion angle contribution kernels
cudaError_t launchTorsionAngleGradientKernel(int            numTorsion,
                                             const int*     idx1,
                                             const int*     idx2,
                                             const int*     idx3,
                                             const int*     idx4,
                                             const double*  forceConstant,
                                             const int*     signs,
                                             const double*  pos,
                                             double*        grad,
                                             const int*     atomIdxToBatchIdx,
                                             const int*     atomStarts,
                                             const uint8_t* activeThisStage = nullptr,
                                             cudaStream_t   stream          = 0);

// Improper torsion (inversion) contribution kernels
cudaError_t launchInversionEnergyKernel(int            numInversion,
                                        const int*     idx1,
                                        const int*     idx2,
                                        const int*     idx3,
                                        const int*     idx4,
                                        const int*     at2AtomicNum,
                                        const uint8_t* isCBoundToO,
                                        const double*  C0,
                                        const double*  C1,
                                        const double*  C2,
                                        const double*  forceConstant,
                                        const double*  pos,
                                        double*        energyBuffer,
                                        const int*     energyBufferStarts,
                                        const int*     atomIdxToBatchIdx,
                                        const int*     inversionTermStarts,
                                        const int*     atomStarts,
                                        const uint8_t* activeThisStage = nullptr,
                                        cudaStream_t   stream          = 0);

cudaError_t launchInversionGradientKernel(int            numInversion,
                                          const int*     idx1,
                                          const int*     idx2,
                                          const int*     idx3,
                                          const int*     idx4,
                                          const int*     at2AtomicNum,
                                          const uint8_t* isCBoundToO,
                                          const double*  C0,
                                          const double*  C1,
                                          const double*  C2,
                                          const double*  forceConstant,
                                          const double*  pos,
                                          double*        grad,
                                          const int*     atomIdxToBatchIdx,
                                          const int*     atomStarts,
                                          const uint8_t* activeThisStage = nullptr,
                                          cudaStream_t   stream          = 0);

// Distance constraint contribution kernels
cudaError_t launchDistanceConstraintGradientKernel(int            numDist,
                                                   const int*     idx1,
                                                   const int*     idx2,
                                                   const double*  minLen,
                                                   const double*  maxLen,
                                                   const double*  forceConstant,
                                                   const double*  pos,
                                                   double*        grad,
                                                   const int*     atomIdxToBatchIdx,
                                                   const int*     atomStarts,
                                                   const uint8_t* activeThisStage = nullptr,
                                                   cudaStream_t   stream          = 0);

// Angle constraint contribution kernels
constexpr double defaultAngleForceConstant = 1.0;
cudaError_t      launchAngleConstraintEnergyKernel(int            numAngle,
                                                   const int*     idx1,
                                                   const int*     idx2,
                                                   const int*     idx3,
                                                   const double*  minAngle,
                                                   const double*  maxAngle,
                                                   const double*  pos,
                                                   double*        energyBuffer,
                                                   const int*     energyBufferStarts,
                                                   const int*     atomIdxToBatchIdx,
                                                   const int*     angleTermStarts,
                                                   const int*     atomStarts,
                                                   const uint8_t* activeThisStage = nullptr,
                                                   double         forceConstant   = defaultAngleForceConstant,
                                                   cudaStream_t   stream          = 0);

cudaError_t launchAngleConstraintGradientKernel(int            numAngle,
                                                const int*     idx1,
                                                const int*     idx2,
                                                const int*     idx3,
                                                const double*  minAngle,
                                                const double*  maxAngle,
                                                const double*  pos,
                                                double*        grad,
                                                const int*     atomIdxToBatchIdx,
                                                const int*     atomStarts,
                                                const uint8_t* activeThisStage = nullptr,
                                                double         forceConstant   = defaultAngleForceConstant,
                                                cudaStream_t   stream          = 0);

cudaError_t launchReduceEnergiesKernel(int            numBlocks,
                                       const double*  energyBuffer,
                                       const int*     energyBufferBlockIdxToBatchIdx,
                                       double*        outs,
                                       const uint8_t* activeThisStage = nullptr,
                                       cudaStream_t   stream          = 0);

template <typename Scalar> struct DistViolationContribTermsDevicePtrT {
  const int*    idx1;
  const int*    idx2;
  const Scalar* ub2;
  const Scalar* lb2;
  const Scalar* weight;
};

template <typename Scalar> struct ChiralViolationContribTermsDevicePtrT {
  const int*    idx1;
  const int*    idx2;
  const int*    idx3;
  const int*    idx4;
  const Scalar* volUpper;
  const Scalar* volLower;
};

struct FourthDimContribTermsDevicePtr {
  const int* idx;
};

template <typename Scalar> struct EnergyForceContribsDevicePtrT {
  DistViolationContribTermsDevicePtrT<Scalar>   distTerms;
  ChiralViolationContribTermsDevicePtrT<Scalar> chiralTerms;
  FourthDimContribTermsDevicePtr                fourthTerms;
};

struct BatchedIndicesDevicePtr {
  const int* atomStarts;
  const int* distTermStarts;
  const int* chiralTermStarts;
  const int* fourthTermStarts;
};

template <typename Scalar> struct TorsionAngleContribTermsDevicePtrT {
  const int*    idx1;
  const int*    idx2;
  const int*    idx3;
  const int*    idx4;
  const Scalar* forceConstants;
  const int*    signs;
};

template <typename Scalar> struct InversionContribTermsDevicePtrT {
  const int*     idx1;
  const int*     idx2;
  const int*     idx3;
  const int*     idx4;
  const int*     at2AtomicNum;
  const uint8_t* isCBoundToO;
  const Scalar*  C0;
  const Scalar*  C1;
  const Scalar*  C2;
  const Scalar*  forceConstant;
};

template <typename Scalar> struct DistanceConstraintContribTermsDevicePtrT {
  const int*    idx1;
  const int*    idx2;
  const Scalar* minLen;
  const Scalar* maxLen;
  const Scalar* forceConstant;
};

template <typename Scalar> struct AngleConstraintContribTermsDevicePtrT {
  const int*    idx1;
  const int*    idx2;
  const int*    idx3;
  const Scalar* minAngle;
  const Scalar* maxAngle;
};

template <typename Scalar> struct Energy3DForceContribsDevicePtrT {
  TorsionAngleContribTermsDevicePtrT<Scalar>       experimentalTorsionTerms;
  InversionContribTermsDevicePtrT<Scalar>          improperTorsionTerms;
  DistanceConstraintContribTermsDevicePtrT<Scalar> dist12Terms;
  DistanceConstraintContribTermsDevicePtrT<Scalar> dist13Terms;
  AngleConstraintContribTermsDevicePtrT<Scalar>    angle13Terms;
  DistanceConstraintContribTermsDevicePtrT<Scalar> longRangeDistTerms;
};

using DistViolationContribTermsDevicePtr      = DistViolationContribTermsDevicePtrT<double>;
using ChiralViolationContribTermsDevicePtr    = ChiralViolationContribTermsDevicePtrT<double>;
using EnergyForceContribsDevicePtr            = EnergyForceContribsDevicePtrT<double>;
using TorsionAngleContribTermsDevicePtr       = TorsionAngleContribTermsDevicePtrT<double>;
using InversionContribTermsDevicePtr          = InversionContribTermsDevicePtrT<double>;
using DistanceConstraintContribTermsDevicePtr = DistanceConstraintContribTermsDevicePtrT<double>;
using AngleConstraintContribTermsDevicePtr    = AngleConstraintContribTermsDevicePtrT<double>;
using Energy3DForceContribsDevicePtr          = Energy3DForceContribsDevicePtrT<double>;

using EnergyForceContribsDevicePtrSingle   = EnergyForceContribsDevicePtrT<float>;
using Energy3DForceContribsDevicePtrSingle = Energy3DForceContribsDevicePtrT<float>;

struct BatchedIndices3DDevicePtr {
  const int* atomStarts;
  const int* experimentalTorsionTermStarts;
  const int* improperTorsionTermStarts;
  const int* dist12TermStarts;
  const int* dist13TermStarts;
  const int* angle13TermStarts;
  const int* longRangeDistTermStarts;
};

cudaError_t launchBlockPerMolEnergyKernel(int                                 numMols,
                                          const EnergyForceContribsDevicePtr& terms,
                                          const BatchedIndicesDevicePtr&      systemIndices,
                                          const double*                       coords,
                                          double*                             energies,
                                          int                                 dimension,
                                          double                              chiralWeight,
                                          double                              fourthDimWeight,
                                          const uint8_t*                      activeThisStage = nullptr,
                                          cudaStream_t                        stream          = 0);

cudaError_t launchBlockPerMolGradKernel(int                                 numMols,
                                        const EnergyForceContribsDevicePtr& terms,
                                        const BatchedIndicesDevicePtr&      systemIndices,
                                        const double*                       coords,
                                        double*                             grad,
                                        int                                 dimension,
                                        double                              chiralWeight,
                                        double                              fourthDimWeight,
                                        const uint8_t*                      activeThisStage = nullptr,
                                        cudaStream_t                        stream          = 0);
cudaError_t launchBlockPerMolEnergyKernelETK(int                                   numMols,
                                             const Energy3DForceContribsDevicePtr& terms,
                                             const BatchedIndices3DDevicePtr&      systemIndices,
                                             const double*                         coords,
                                             double*                               energies,
                                             const uint8_t*                        activeThisStage = nullptr,
                                             cudaStream_t                          stream          = 0);

cudaError_t launchBlockPerMolGradKernelETK(int                                   numMols,
                                           const Energy3DForceContribsDevicePtr& terms,
                                           const BatchedIndices3DDevicePtr&      systemIndices,
                                           const double*                         coords,
                                           double*                               grad,
                                           const uint8_t*                        activeThisStage = nullptr,
                                           cudaStream_t                          stream          = 0);
cudaError_t launchBlockPerMolEnergyKernel(int                                       numMols,
                                          const EnergyForceContribsDevicePtrSingle& terms,
                                          const BatchedIndicesDevicePtr&            systemIndices,
                                          const float*                              coords,
                                          float*                                    energies,
                                          int                                       dimension,
                                          float                                     chiralWeight,
                                          float                                     fourthDimWeight,
                                          const uint8_t*                            activeThisStage = nullptr,
                                          cudaStream_t                              stream          = 0);
cudaError_t launchBlockPerMolGradKernel(int                                       numMols,
                                        const EnergyForceContribsDevicePtrSingle& terms,
                                        const BatchedIndicesDevicePtr&            systemIndices,
                                        const float*                              coords,
                                        float*                                    grad,
                                        int                                       dimension,
                                        float                                     chiralWeight,
                                        float                                     fourthDimWeight,
                                        const uint8_t*                            activeThisStage = nullptr,
                                        cudaStream_t                              stream          = 0);
cudaError_t launchBlockPerMolEnergyKernelETK(int                                         numMols,
                                             const Energy3DForceContribsDevicePtrSingle& terms,
                                             const BatchedIndices3DDevicePtr&            systemIndices,
                                             const float*                                coords,
                                             float*                                      energies,
                                             const uint8_t*                              activeThisStage = nullptr,
                                             cudaStream_t                                stream          = 0);
cudaError_t launchBlockPerMolGradKernelETK(int                                         numMols,
                                           const Energy3DForceContribsDevicePtrSingle& terms,
                                           const BatchedIndices3DDevicePtr&            systemIndices,
                                           const float*                                coords,
                                           float*                                      grad,
                                           const uint8_t*                              activeThisStage = nullptr,
                                           cudaStream_t                                stream          = 0);
cudaError_t launchPlanarEnergyKernelETK(int                                         numMols,
                                        const Energy3DForceContribsDevicePtrSingle& terms,
                                        const BatchedIndices3DDevicePtr&            systemIndices,
                                        const float*                                coords,
                                        float*                                      energies,
                                        const uint8_t*                              activeThisStage = nullptr,
                                        cudaStream_t                                stream          = 0);

//! Fused block-per-molecule evaluation: one launch computes every term family of each molecule and updates the
//! output in place. Term and index pointer structs are passed by value as kernel parameters.
//!
//! Double precision follows the term-parallel computeEnergy/computeGradients contract: active systems
//! (activeThisStage == 1) add their sum to energies[mol] or to their grad slice; inactive systems are left untouched.
//! Single precision follows the launchBlockPerMol* contract: inactive systems (activeThisStage == 0) get energy 0 and
//! keep their gradient; active systems overwrite their energy, overwrite their DG gradient slice and add to their ETK
//! gradient slice.
cudaError_t launchFusedEnergyKernel(int                                 numMols,
                                    const EnergyForceContribsDevicePtr& terms,
                                    const BatchedIndicesDevicePtr&      systemIndices,
                                    const double*                       coords,
                                    double*                             energies,
                                    int                                 dimension,
                                    double                              chiralWeight,
                                    double                              fourthDimWeight,
                                    const uint8_t*                      activeThisStage = nullptr,
                                    cudaStream_t                        stream          = 0);
cudaError_t launchFusedGradKernel(int                                 numMols,
                                  const EnergyForceContribsDevicePtr& terms,
                                  const BatchedIndicesDevicePtr&      systemIndices,
                                  const double*                       coords,
                                  double*                             grad,
                                  int                                 dimension,
                                  double                              chiralWeight,
                                  double                              fourthDimWeight,
                                  const uint8_t*                      activeThisStage = nullptr,
                                  cudaStream_t                        stream          = 0);
cudaError_t launchFusedEnergyKernel(int                                       numMols,
                                    const EnergyForceContribsDevicePtrSingle& terms,
                                    const BatchedIndicesDevicePtr&            systemIndices,
                                    const float*                              coords,
                                    float*                                    energies,
                                    int                                       dimension,
                                    float                                     chiralWeight,
                                    float                                     fourthDimWeight,
                                    const uint8_t*                            activeThisStage = nullptr,
                                    cudaStream_t                              stream          = 0);
cudaError_t launchFusedGradKernel(int                                       numMols,
                                  const EnergyForceContribsDevicePtrSingle& terms,
                                  const BatchedIndicesDevicePtr&            systemIndices,
                                  const float*                              coords,
                                  float*                                    grad,
                                  int                                       dimension,
                                  float                                     chiralWeight,
                                  float                                     fourthDimWeight,
                                  const uint8_t*                            activeThisStage = nullptr,
                                  cudaStream_t                              stream          = 0);

//! ETK term-family bits for the fused ETK energy termMask.
constexpr int kETKTorsionBit  = 1;
constexpr int kETKImproperBit = 2;
constexpr int kETKDist12Bit   = 4;
constexpr int kETKDist13Bit   = 8;
constexpr int kETKAngle13Bit  = 16;
constexpr int kETKLongDistBit = 32;
constexpr int kETKAllTermsMask =
  kETKTorsionBit | kETKImproperBit | kETKDist12Bit | kETKDist13Bit | kETKAngle13Bit | kETKLongDistBit;

cudaError_t launchFusedEnergyKernelETK(int                                   numMols,
                                       const Energy3DForceContribsDevicePtr& terms,
                                       const BatchedIndices3DDevicePtr&      systemIndices,
                                       const double*                         coords,
                                       double*                               energies,
                                       int                                   termMask        = kETKAllTermsMask,
                                       const uint8_t*                        activeThisStage = nullptr,
                                       cudaStream_t                          stream          = 0);
cudaError_t launchFusedEnergyKernelETK(int                                         numMols,
                                       const Energy3DForceContribsDevicePtrSingle& terms,
                                       const BatchedIndices3DDevicePtr&            systemIndices,
                                       const float*                                coords,
                                       float*                                      energies,
                                       int                                         termMask        = kETKAllTermsMask,
                                       const uint8_t*                              activeThisStage = nullptr,
                                       cudaStream_t                                stream          = 0);
//! Fused ETK gradient. The double-precision kernel is selected only on sm_100+ (see DistGeom::computeGradientsETK);
//! elsewhere the term-parallel kernels are faster.
cudaError_t launchFusedGradKernelETK(int                                   numMols,
                                     const Energy3DForceContribsDevicePtr& terms,
                                     const BatchedIndices3DDevicePtr&      systemIndices,
                                     const double*                         coords,
                                     double*                               grad,
                                     int                                   termMask        = kETKAllTermsMask,
                                     const uint8_t*                        activeThisStage = nullptr,
                                     cudaStream_t                          stream          = 0);
cudaError_t launchFusedGradKernelETK(int                                         numMols,
                                     const Energy3DForceContribsDevicePtrSingle& terms,
                                     const BatchedIndices3DDevicePtr&            systemIndices,
                                     const float*                                coords,
                                     float*                                      grad,
                                     const uint8_t*                              activeThisStage = nullptr,
                                     cudaStream_t                                stream          = 0);
//! FULL ETK gradient of the long-range distance restraints only (when selected by termMask), block per molecule:
//! adds to the grad slice of active systems. Pairs with the term-parallel kernels for the other families on large
//! batches.
cudaError_t launchFusedLongRangeGradKernelETK(int                                   numMols,
                                              const Energy3DForceContribsDevicePtr& terms,
                                              const BatchedIndices3DDevicePtr&      systemIndices,
                                              const double*                         coords,
                                              double*                               grad,
                                              int                                   termMask        = kETKAllTermsMask,
                                              const uint8_t*                        activeThisStage = nullptr,
                                              cudaStream_t                          stream          = 0);
}  // namespace DistGeom
}  // namespace nvMolKit

#endif  // NVMOLKIT_DISTGEOM_KERNELS_H
