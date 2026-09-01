// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/MolOps.h>

#include "rdkit_extensions/dist_geom_flattened_builder.h"
#include "src/etkdg_stage_aio_minimization.h"
#include "src/forcefields/dist_geom_kernels_device.cuh"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit::detail {
namespace {

constexpr int kBlockSize = 256;

__global__ void checkAllInOneEnergyKernel(const int      numSystems,
                                          const double*  energies,
                                          const int*     atomStarts,
                                          const uint8_t* activeSystems,
                                          uint8_t*       failedSystems) {
  const int systemIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (systemIdx >= numSystems || !activeSystems[systemIdx]) {
    return;
  }
  const int numAtoms = atomStarts[systemIdx + 1] - atomStarts[systemIdx];
  if (energies[systemIdx] / static_cast<double>(numAtoms) >= 0.1) {
    failedSystems[systemIdx] = 1;
  }
}

__global__ void checkKAngleTermsKernel(const int      numTerms,
                                       const int*     idx1,
                                       const int*     idx2,
                                       const int*     idx3,
                                       const double*  minAngle,
                                       const double*  maxAngle,
                                       const double*  forceConstant,
                                       const int*     systemIdx,
                                       const double*  positions,
                                       const uint8_t* activeSystems,
                                       double*        systemEnergies,
                                       uint8_t*       failedSystems) {
  const int termIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (termIdx >= numTerms || !activeSystems[systemIdx[termIdx]]) {
    return;
  }
  const double energy = DistGeom::angleConstraintEnergy(positions,
                                                        idx1[termIdx],
                                                        idx2[termIdx],
                                                        idx3[termIdx],
                                                        minAngle[termIdx],
                                                        maxAngle[termIdx],
                                                        forceConstant[termIdx]);
  atomicAdd(systemEnergies + systemIdx[termIdx], energy);
  if (energy > 0.05) {
    failedSystems[systemIdx[termIdx]] = 1;
  }
}

__global__ void checkKImproperTermsKernel(const int      numTerms,
                                          const int*     idx1,
                                          const int*     idx2,
                                          const int*     idx3,
                                          const int*     idx4,
                                          const double*  c0,
                                          const double*  c1,
                                          const double*  c2,
                                          const double*  forceConstant,
                                          const int*     systemIdx,
                                          const double*  positions,
                                          const uint8_t* activeSystems,
                                          double*        systemEnergies,
                                          uint8_t*       failedSystems) {
  const int termIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (termIdx >= numTerms || !activeSystems[systemIdx[termIdx]]) {
    return;
  }
  const double energy = DistGeom::inversionEnergy(positions,
                                                  idx1[termIdx],
                                                  idx2[termIdx],
                                                  idx3[termIdx],
                                                  idx4[termIdx],
                                                  c0[termIdx],
                                                  c1[termIdx],
                                                  c2[termIdx],
                                                  forceConstant[termIdx]);
  atomicAdd(systemEnergies + systemIdx[termIdx], energy);
  if (energy > 5.0) {
    failedSystems[systemIdx[termIdx]] = 1;
  }
}

__global__ void checkKTotalEnergyKernel(const int      numSystems,
                                        const double*  energies,
                                        const int*     numCenters,
                                        const uint8_t* activeSystems,
                                        uint8_t*       failedSystems) {
  const int systemIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (systemIdx < numSystems && activeSystems[systemIdx] && energies[systemIdx] > 0.7 * numCenters[systemIdx]) {
    failedSystems[systemIdx] = 1;
  }
}

}  // namespace

ETKDGAllInOneMinimizationStage::ETKDGAllInOneMinimizationStage(const std::vector<const RDKit::ROMol*>&     mols,
                                                               const std::vector<EmbedArgs>&               eargs,
                                                               const RDKit::DGeomHelpers::EmbedParameters& embedParams,
                                                               const ETKDGContext&                         ctx,
                                                               const cudaStream_t                          stream)
    : minimizer_(4, DebugLevel::NONE, true, stream, BfgsBackend::BATCHED),
      optimizerForceTol_(embedParams.optimizerForceTol),
      stream_(stream) {
  if (mols.size() != eargs.size()) {
    throw std::invalid_argument("Number of molecules and embed args must be the same");
  }
  gradients_.setStream(stream);
  energies_.setStream(stream);
  systemContribs_.reserve(mols.size());

  for (std::size_t systemIdx = 0; systemIdx < mols.size(); ++systemIdx) {
    const auto*   mol                       = mols[systemIdx];
    const auto&   earg                      = eargs[systemIdx];
    const double* topologicalDistanceMatrix = RDKit::MolOps::getDistanceMat(*mol);
    systemContribs_.push_back(DistGeom::constructAllInOneForceFieldContribs(earg.dim,
                                                                            *earg.mmat,
                                                                            earg.chiralCenters,
                                                                            earg.etkdgDetails,
                                                                            topologicalDistanceMatrix,
                                                                            embedParams.useExpTorsionAnglePrefs,
                                                                            embedParams.useBasicKnowledge));
    auto& contribs = systemContribs_.back().distanceGeometry;
    if (systemIdx == 0) {
      DistGeom::preallocateEstimatedBatch(contribs, dgSystemHost_, static_cast<int>(mols.size()));
    }
    DistGeom::addMoleculeToMolecularSystem(contribs,
                                           static_cast<int>(mol->getNumAtoms()),
                                           earg.dim,
                                           ctx.systemHost.atomStarts,
                                           dgSystemHost_,
                                           &metadata_,
                                           static_cast<int>(systemIdx),
                                           0);
    firstPhaseIterations_ = std::max(firstPhaseIterations_, static_cast<int>(mol->getNumHeavyAtoms()));
  }
}

void ETKDGAllInOneMinimizationStage::execute(ETKDGContext& ctx) {
  AllInOneETKDGBatchedForcefield forcefield(dgSystemHost_,
                                            systemContribs_,
                                            ctx.systemHost.atomStarts,
                                            metadata_,
                                            stream_);
  gradients_.resize(ctx.systemDevice.positions.size());
  energies_.resize(ctx.systemHost.atomStarts.size() - 1);

  forcefield.setTorsionTermsEnabled(false);
  minimizer_.minimize(firstPhaseIterations_,
                      optimizerForceTol_,
                      forcefield,
                      ctx.systemDevice.positions,
                      gradients_,
                      energies_,
                      ctx.activeThisStage.data());

  forcefield.setTorsionTermsEnabled(true);
  for (int cycle = 0; cycle < 8; ++cycle) {
    const bool needsMore = minimizer_.minimize(100,
                                               optimizerForceTol_,
                                               forcefield,
                                               ctx.systemDevice.positions,
                                               gradients_,
                                               energies_,
                                               ctx.activeThisStage.data());
    if (!needsMore) {
      break;
    }
  }

  energies_.zero();
  cudaCheckError(
    forcefield.computeEnergy(energies_.data(), ctx.systemDevice.positions.data(), ctx.activeThisStage.data(), stream_));
  const int numSystems = energies_.size();
  checkAllInOneEnergyKernel<<<(numSystems + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream_>>>(
    numSystems,
    energies_.data(),
    ctx.systemDevice.atomStarts.data(),
    ctx.activeThisStage.data(),
    ctx.failedThisStage.data());
  cudaCheckError(cudaGetLastError());
}

ETKDGKTermCheckStage::ETKDGKTermCheckStage(const std::vector<EmbedArgs>& eargs,
                                           const ETKDGContext&           ctx,
                                           const cudaStream_t            stream)
    : stream_(stream) {
  std::vector<int>    angleIdx1, angleIdx2, angleIdx3, angleSystemIdx;
  std::vector<double> angleMin, angleMax, angleForceConstant;
  std::vector<int>    improperIdx1, improperIdx2, improperIdx3, improperIdx4, improperSystemIdx;
  std::vector<double> improperC0, improperC1, improperC2, improperForceConstant;
  std::vector<int>    numCenters;
  for (std::size_t systemIdx = 0; systemIdx < eargs.size(); ++systemIdx) {
    const auto&         earg = eargs[systemIdx];
    std::vector<double> dummyPositions(earg.mmat->numRows() * 3, 0.0);
    const auto          contribs =
      DistGeom::construct3DForceFieldContribs(*earg.mmat, earg.etkdgDetails, dummyPositions, 3, true);
    const int offset = ctx.systemHost.atomStarts[systemIdx];
    for (std::size_t i = 0; i < contribs.angle13Terms.idx1.size(); ++i) {
      angleIdx1.push_back(offset + contribs.angle13Terms.idx1[i]);
      angleIdx2.push_back(offset + contribs.angle13Terms.idx2[i]);
      angleIdx3.push_back(offset + contribs.angle13Terms.idx3[i]);
      angleMin.push_back(contribs.angle13Terms.minAngle[i]);
      angleMax.push_back(contribs.angle13Terms.maxAngle[i]);
      angleForceConstant.push_back(10.0);
      angleSystemIdx.push_back(systemIdx);
    }
    const auto& improper = contribs.improperTorsionTerms;
    for (std::size_t i = 0; i < improper.idx1.size(); ++i) {
      improperIdx1.push_back(offset + improper.idx1[i]);
      improperIdx2.push_back(offset + improper.idx2[i]);
      improperIdx3.push_back(offset + improper.idx3[i]);
      improperIdx4.push_back(offset + improper.idx4[i]);
      improperC0.push_back(improper.C0[i]);
      improperC1.push_back(improper.C1[i]);
      improperC2.push_back(improper.C2[i]);
      improperForceConstant.push_back(improper.forceConstant[i]);
      improperSystemIdx.push_back(systemIdx);
    }
    int linearCenters = 0;
    for (const auto& angle : earg.etkdgDetails.angles) {
      linearCenters += angle[3] != 0;
    }
    numCenters.push_back(linearCenters + earg.etkdgDetails.improperAtoms.size());
  }
  const auto upload = [stream]<typename T>(AsyncDeviceVector<T>& destination, const std::vector<T>& source) {
    destination.setStream(stream);
    destination.setFromVector(source);
  };
  upload(angleIdx1_, angleIdx1);
  upload(angleIdx2_, angleIdx2);
  upload(angleIdx3_, angleIdx3);
  upload(angleMin_, angleMin);
  upload(angleMax_, angleMax);
  upload(angleForceConstant_, angleForceConstant);
  upload(angleSystemIdx_, angleSystemIdx);
  upload(improperIdx1_, improperIdx1);
  upload(improperIdx2_, improperIdx2);
  upload(improperIdx3_, improperIdx3);
  upload(improperIdx4_, improperIdx4);
  upload(improperC0_, improperC0);
  upload(improperC1_, improperC1);
  upload(improperC2_, improperC2);
  upload(improperForceConstant_, improperForceConstant);
  upload(improperSystemIdx_, improperSystemIdx);
  upload(numCenters_, numCenters);
  systemEnergies_.setStream(stream);
  cudaCheckError(cudaStreamSynchronize(stream));
}

void ETKDGKTermCheckStage::execute(ETKDGContext& ctx) {
  if (ctx.nTotalSystems == 0) {
    return;
  }
  systemEnergies_.resize(ctx.nTotalSystems);
  systemEnergies_.zero();
  if (angleIdx1_.size() > 0) {
    checkKAngleTermsKernel<<<(angleIdx1_.size() + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream_>>>(
      angleIdx1_.size(),
      angleIdx1_.data(),
      angleIdx2_.data(),
      angleIdx3_.data(),
      angleMin_.data(),
      angleMax_.data(),
      angleForceConstant_.data(),
      angleSystemIdx_.data(),
      ctx.systemDevice.positions.data(),
      ctx.activeThisStage.data(),
      systemEnergies_.data(),
      ctx.failedThisStage.data());
  }
  if (improperIdx1_.size() > 0) {
    checkKImproperTermsKernel<<<(improperIdx1_.size() + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream_>>>(
      improperIdx1_.size(),
      improperIdx1_.data(),
      improperIdx2_.data(),
      improperIdx3_.data(),
      improperIdx4_.data(),
      improperC0_.data(),
      improperC1_.data(),
      improperC2_.data(),
      improperForceConstant_.data(),
      improperSystemIdx_.data(),
      ctx.systemDevice.positions.data(),
      ctx.activeThisStage.data(),
      systemEnergies_.data(),
      ctx.failedThisStage.data());
  }
  checkKTotalEnergyKernel<<<(ctx.nTotalSystems + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream_>>>(
    ctx.nTotalSystems,
    systemEnergies_.data(),
    numCenters_.data(),
    ctx.activeThisStage.data(),
    ctx.failedThisStage.data());
  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit::detail
