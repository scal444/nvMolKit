// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/Conformer.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "nvmolkit/nanobind/device_result_python.h"
#include "nvmolkit/nanobind/forcefield_python_utils.h"
#include "rdkit_extensions/mmff_flattened_builder.h"
#include "rdkit_extensions/uff_flattened_builder.h"
#include "src/forcefields/ff_utils.h"
#include "src/forcefields/forcefield_constraints.h"
#include "src/forcefields/mmff_batched_forcefield.h"
#include "src/forcefields/uff_batched_forcefield.h"
#include "src/minimizer/fire_minimizer.h"
#include "src/minimizer/mmff_minimize.h"
#include "src/minimizer/uff_minimize.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

namespace ForcefieldConstraints = nvMolKit::ForceFieldConstraints;

enum class MinimizerKind {
  BFGS,
  FIRE
};

void throwIfCudaError(const cudaError_t error, const std::string& context) {
  if (error != cudaSuccess) {
    throw std::runtime_error(context + ": " + cudaGetErrorString(error));
  }
}

MinimizerKind parseMinimizerKind(const std::string& name) {
  if (name == "BFGS" || name == "bfgs") {
    return MinimizerKind::BFGS;
  }
  if (name == "FIRE" || name == "fire") {
    return MinimizerKind::FIRE;
  }
  throw std::invalid_argument("Unknown minimizerKind '" + name + "'. Expected 'BFGS' or 'FIRE'.");
}

template <typename T> std::vector<T> copyDeviceVector(nvMolKit::AsyncDeviceVector<T>& deviceVector) {
  std::vector<T> hostVector(deviceVector.size());
  deviceVector.copyToHost(hostVector);
  throwIfCudaError(cudaStreamSynchronize(deviceVector.stream()), "copyDeviceVector/sync");
  return hostVector;
}

std::vector<std::vector<double>> reshapeValues(const std::vector<double>& flatValues,
                                               const std::vector<int>&    conformerCounts) {
  std::vector<std::vector<double>> result;
  result.reserve(conformerCounts.size());
  size_t flatIndex = 0;
  for (const int conformerCount : conformerCounts) {
    auto& moleculeValues = result.emplace_back();
    moleculeValues.reserve(conformerCount);
    for (int conformerIndex = 0; conformerIndex < conformerCount; ++conformerIndex) {
      moleculeValues.push_back(flatValues[flatIndex++]);
    }
  }
  return result;
}

std::vector<std::vector<double>> splitGradients(const std::vector<double>& flatGradients,
                                                const std::vector<int>&    atomStarts) {
  std::vector<std::vector<double>> result;
  result.reserve(atomStarts.size() - 1);
  for (size_t systemIndex = 0; systemIndex + 1 < atomStarts.size(); ++systemIndex) {
    const int begin = atomStarts[systemIndex] * 3;
    const int end   = atomStarts[systemIndex + 1] * 3;
    result.emplace_back(flatGradients.begin() + begin, flatGradients.begin() + end);
  }
  return result;
}

std::vector<std::vector<std::vector<double>>> reshapeGradients(
  const std::vector<std::vector<double>>& perSystemGradients,
  const std::vector<int>&                 conformerCounts) {
  std::vector<std::vector<std::vector<double>>> result;
  result.reserve(conformerCounts.size());
  size_t systemIndex = 0;
  for (const int conformerCount : conformerCounts) {
    auto& moleculeGradients = result.emplace_back();
    moleculeGradients.reserve(conformerCount);
    for (int conformerIndex = 0; conformerIndex < conformerCount; ++conformerIndex) {
      moleculeGradients.push_back(perSystemGradients[systemIndex++]);
    }
  }
  return result;
}

void uploadConformerPositions(const std::vector<RDKit::ROMol*>&    molecules,
                              nvMolKit::AsyncDeviceVector<double>& devicePositions) {
  std::vector<double> allPositions;
  for (const auto* molecule : molecules) {
    for (auto conformer = molecule->beginConformers(); conformer != molecule->endConformers(); ++conformer) {
      std::vector<double> conformerPositions;
      nvMolKit::confPosToVect(**conformer, conformerPositions);
      allPositions.insert(allPositions.end(), conformerPositions.begin(), conformerPositions.end());
    }
  }
  devicePositions.copyFromHost(allPositions.data(), allPositions.size());
  throwIfCudaError(cudaStreamSynchronize(devicePositions.stream()), "uploadConformerPositions/sync");
}

std::vector<std::vector<double>> computeBatchedEnergy(nvMolKit::BatchedForcefield&         forcefield,
                                                      nvMolKit::AsyncDeviceVector<double>& devicePositions,
                                                      nvMolKit::AsyncDeviceVector<double>& deviceEnergies,
                                                      const std::vector<int>&              conformerCounts) {
  deviceEnergies.zero();
  throwIfCudaError(forcefield.computeEnergy(deviceEnergies.data(), devicePositions.data()), "computeEnergy");
  return reshapeValues(copyDeviceVector(deviceEnergies), conformerCounts);
}

std::vector<std::vector<std::vector<double>>> computeBatchedGradients(
  nvMolKit::BatchedForcefield&         forcefield,
  nvMolKit::AsyncDeviceVector<double>& devicePositions,
  nvMolKit::AsyncDeviceVector<double>& deviceGradients,
  const std::vector<int>&              conformerCounts) {
  deviceGradients.zero();
  throwIfCudaError(forcefield.computeGradients(deviceGradients.data(), devicePositions.data()), "computeGradients");
  return reshapeGradients(splitGradients(copyDeviceVector(deviceGradients), forcefield.atomStartsHost()),
                          conformerCounts);
}

template <typename Spec, typename Parser>
std::vector<std::vector<Spec>> extractConstraintLists(const nb::list&    outerList,
                                                      const int          expectedSize,
                                                      const Parser&      parser,
                                                      const std::string& name) {
  const size_t actualSize = nb::len(outerList);
  if (actualSize != static_cast<size_t>(expectedSize)) {
    throw std::invalid_argument("Expected " + std::to_string(expectedSize) + " entries for " + name + ", got " +
                                std::to_string(actualSize));
  }
  std::vector<std::vector<Spec>> result(expectedSize);
  for (int moleculeIndex = 0; moleculeIndex < expectedSize; ++moleculeIndex) {
    const nb::list innerList = nb::cast<nb::list>(outerList[moleculeIndex]);
    auto&          specs     = result[moleculeIndex];
    specs.reserve(nb::len(innerList));
    for (size_t constraintIndex = 0; constraintIndex < nb::len(innerList); ++constraintIndex) {
      specs.push_back(parser(nb::cast<nb::tuple>(innerList[constraintIndex])));
    }
  }
  return result;
}

ForcefieldConstraints::DistanceConstraintSpec parseDistanceConstraint(const nb::tuple& value) {
  if (nb::len(value) != 6) {
    throw std::invalid_argument("Distance constraint tuples must have 6 elements");
  }
  return {nb::cast<int>(value[0]),
          nb::cast<int>(value[1]),
          nb::cast<bool>(value[2]),
          nb::cast<double>(value[3]),
          nb::cast<double>(value[4]),
          nb::cast<double>(value[5])};
}

ForcefieldConstraints::PositionConstraintSpec parsePositionConstraint(const nb::tuple& value) {
  if (nb::len(value) != 3) {
    throw std::invalid_argument("Position constraint tuples must have 3 elements");
  }
  return {nb::cast<int>(value[0]), nb::cast<double>(value[1]), nb::cast<double>(value[2])};
}

ForcefieldConstraints::AngleConstraintSpec parseAngleConstraint(const nb::tuple& value) {
  if (nb::len(value) != 7) {
    throw std::invalid_argument("Angle constraint tuples must have 7 elements");
  }
  return {nb::cast<int>(value[0]),
          nb::cast<int>(value[1]),
          nb::cast<int>(value[2]),
          nb::cast<bool>(value[3]),
          nb::cast<double>(value[4]),
          nb::cast<double>(value[5]),
          nb::cast<double>(value[6])};
}

ForcefieldConstraints::TorsionConstraintSpec parseTorsionConstraint(const nb::tuple& value) {
  if (nb::len(value) != 8) {
    throw std::invalid_argument("Torsion constraint tuples must have 8 elements");
  }
  return {nb::cast<int>(value[0]),
          nb::cast<int>(value[1]),
          nb::cast<int>(value[2]),
          nb::cast<int>(value[3]),
          nb::cast<bool>(value[4]),
          nb::cast<double>(value[5]),
          nb::cast<double>(value[6]),
          nb::cast<double>(value[7])};
}

std::vector<ForcefieldConstraints::PerMolConstraints> extractAllConstraints(const nb::list& distanceConstraints,
                                                                            const nb::list& positionConstraints,
                                                                            const nb::list& angleConstraints,
                                                                            const nb::list& torsionConstraints,
                                                                            const int       moleculeCount) {
  const auto distances = extractConstraintLists<ForcefieldConstraints::DistanceConstraintSpec>(distanceConstraints,
                                                                                               moleculeCount,
                                                                                               parseDistanceConstraint,
                                                                                               "distance constraints");
  const auto positions = extractConstraintLists<ForcefieldConstraints::PositionConstraintSpec>(positionConstraints,
                                                                                               moleculeCount,
                                                                                               parsePositionConstraint,
                                                                                               "position constraints");
  const auto angles    = extractConstraintLists<ForcefieldConstraints::AngleConstraintSpec>(angleConstraints,
                                                                                         moleculeCount,
                                                                                         parseAngleConstraint,
                                                                                         "angle constraints");
  const auto torsions  = extractConstraintLists<ForcefieldConstraints::TorsionConstraintSpec>(torsionConstraints,
                                                                                             moleculeCount,
                                                                                             parseTorsionConstraint,
                                                                                             "torsion constraints");

  std::vector<ForcefieldConstraints::PerMolConstraints> result(moleculeCount);
  for (int moleculeIndex = 0; moleculeIndex < moleculeCount; ++moleculeIndex) {
    result[moleculeIndex] = {distances[moleculeIndex],
                             positions[moleculeIndex],
                             angles[moleculeIndex],
                             torsions[moleculeIndex]};
  }
  return result;
}

nb::list convertConvergence(const std::vector<std::vector<int8_t>>& convergence) {
  nb::list result;
  for (const auto& moleculeConvergence : convergence) {
    nb::list converted;
    for (const int8_t value : moleculeConvergence) {
      converted.append(value != 0);
    }
    result.append(converted);
  }
  return result;
}

class NativeMMFFBatchedForcefield {
 public:
  NativeMMFFBatchedForcefield(const nb::list&                       molecules,
                              const nb::list&                       properties,
                              const nb::list&                       distanceConstraints,
                              const nb::list&                       positionConstraints,
                              const nb::list&                       angleConstraints,
                              const nb::list&                       torsionConstraints,
                              const nvMolKit::BatchHardwareOptions& hardwareOptions)
      : hardwareOptions_(hardwareOptions) {
    throwIfCudaError(cudaGetDevice(&gpuId_), "MMFF wrapper/cudaGetDevice");
    molecules_              = nvMolKit::NanobindForcefield::extractMolecules(molecules);
    const int moleculeCount = static_cast<int>(molecules_.size());
    properties_             = nvMolKit::NanobindForcefield::extractMMFFPropertiesList(properties, moleculeCount);
    constraints_            = extractAllConstraints(distanceConstraints,
                                         positionConstraints,
                                         angleConstraints,
                                         torsionConstraints,
                                         moleculeCount);
    buildForcefield();
  }

  std::vector<std::vector<double>> computeEnergy() {
    return computeBatchedEnergy(*forcefield_, devicePositions_, deviceEnergies_, conformerCounts_);
  }

  std::vector<std::vector<std::vector<double>>> computeGradients() {
    return computeBatchedGradients(*forcefield_, devicePositions_, deviceGradients_, conformerCounts_);
  }

  nb::tuple minimize(const int                    maxIters,
                     const double                 gradTol,
                     const std::string&           minimizerKind,
                     const nvMolKit::FireOptions& fireOptions) {
    const auto result = parseMinimizerKind(minimizerKind) == MinimizerKind::FIRE ?
                          nvMolKit::MMFF::MMFFMinimizeMoleculesConfsFire(molecules_,
                                                                         maxIters,
                                                                         fireOptions,
                                                                         properties_,
                                                                         constraints_,
                                                                         hardwareOptions_,
                                                                         nvMolKit::FireBackend::BATCHED) :
                          nvMolKit::MMFF::MMFFMinimizeMoleculesConfs(molecules_,
                                                                     maxIters,
                                                                     gradTol,
                                                                     properties_,
                                                                     constraints_,
                                                                     hardwareOptions_);
    uploadConformerPositions(molecules_, devicePositions_);
    return nb::make_tuple(result.energies, convertConvergence(result.converged));
  }

  nb::object minimizeDevice(const int                    maxIters,
                            const double                 gradTol,
                            const int                    targetGpu,
                            const std::string&           minimizerKind,
                            const nvMolKit::FireOptions& fireOptions) {
    validateTargetGpu(targetGpu, "MMFF");
    auto result = parseMinimizerKind(minimizerKind) == MinimizerKind::FIRE ?
                    nvMolKit::MMFF::MMFFMinimizeMoleculesConfsFire(molecules_,
                                                                   maxIters,
                                                                   fireOptions,
                                                                   properties_,
                                                                   constraints_,
                                                                   hardwareOptions_,
                                                                   nvMolKit::FireBackend::BATCHED,
                                                                   nvMolKit::CoordinateOutput::DEVICE,
                                                                   gpuId_) :
                    nvMolKit::MMFF::MMFFMinimizeMoleculesConfs(molecules_,
                                                               maxIters,
                                                               gradTol,
                                                               properties_,
                                                               constraints_,
                                                               hardwareOptions_,
                                                               nvMolKit::BfgsBackend::HYBRID,
                                                               nvMolKit::CoordinateOutput::DEVICE,
                                                               gpuId_);
    if (!result.device.has_value()) {
      throw std::runtime_error("MMFFMinimizeMoleculesConfs(DEVICE) returned no device result");
    }
    refreshPositions(*result.device, "MMFF");
    return nvMolKit::nanobind_bindings::buildOwningDevice3DResult(*result.device);
  }

  int gpuId() const { return gpuId_; }

 private:
  void validateTargetGpu(const int targetGpu, const std::string& forcefieldName) const {
    if (targetGpu >= 0 && targetGpu != gpuId_) {
      throw std::invalid_argument(forcefieldName +
                                  "BatchedForcefield.minimize(output=DEVICE) does not support target_gpu != wrapper "
                                  "GPU (" +
                                  std::to_string(targetGpu) + " vs " + std::to_string(gpuId_) + ").");
    }
  }

  void refreshPositions(nvMolKit::DeviceCoordResult& deviceResult, const std::string& forcefieldName) {
    if (deviceResult.positions.size() != devicePositions_.size()) {
      throw std::runtime_error(forcefieldName + " minimizeDevice positions size does not match wrapper");
    }
    throwIfCudaError(cudaMemcpyAsync(devicePositions_.data(),
                                     deviceResult.positions.data(),
                                     devicePositions_.size() * sizeof(double),
                                     cudaMemcpyDeviceToDevice,
                                     devicePositions_.stream()),
                     forcefieldName + " minimizeDevice/positions refresh");
    throwIfCudaError(cudaStreamSynchronize(devicePositions_.stream()),
                     forcefieldName + " minimizeDevice/positions refresh sync");
  }

  void buildForcefield() {
    const int                                  moleculeCount = static_cast<int>(molecules_.size());
    nvMolKit::MMFF::BatchedMolecularSystemHost systemHost;
    nvMolKit::BatchedForcefieldMetadata        metadata;
    conformerCounts_.resize(moleculeCount);
    for (int moleculeIndex = 0; moleculeIndex < moleculeCount; ++moleculeIndex) {
      auto*      molecule          = molecules_[moleculeIndex];
      const auto baseContributions = nvMolKit::MMFF::constructForcefieldContribs(*molecule, properties_[moleculeIndex]);
      int        conformerIndex    = 0;
      for (auto conformer = molecule->beginConformers(); conformer != molecule->endConformers();
           ++conformer, ++conformerIndex) {
        std::vector<double> positions;
        nvMolKit::confPosToVect(**conformer, positions);
        auto contributions = baseContributions;
        constraints_[moleculeIndex].applyTo(contributions, positions);
        nvMolKit::MMFF::addMoleculeToBatch(contributions,
                                           positions,
                                           systemHost,
                                           &metadata,
                                           moleculeIndex,
                                           conformerIndex);
      }
      conformerCounts_[moleculeIndex] = conformerIndex;
    }
    forcefield_ = std::make_unique<nvMolKit::MMFFBatchedForcefield>(systemHost, metadata);
    devicePositions_.setFromVector(systemHost.positions);
    deviceGradients_.resize(forcefield_->totalPositions());
    deviceEnergies_.resize(forcefield_->numMolecules());
  }

  std::vector<RDKit::ROMol*>                            molecules_;
  std::vector<nvMolKit::MMFFProperties>                 properties_;
  std::vector<ForcefieldConstraints::PerMolConstraints> constraints_;
  nvMolKit::BatchHardwareOptions                        hardwareOptions_;
  std::unique_ptr<nvMolKit::MMFFBatchedForcefield>      forcefield_;
  nvMolKit::AsyncDeviceVector<double>                   devicePositions_;
  nvMolKit::AsyncDeviceVector<double>                   deviceGradients_;
  nvMolKit::AsyncDeviceVector<double>                   deviceEnergies_;
  std::vector<int>                                      conformerCounts_;
  int                                                   gpuId_ = 0;
};

class NativeUFFBatchedForcefield {
 public:
  NativeUFFBatchedForcefield(const nb::list&                       molecules,
                             const nb::list&                       vdwThresholds,
                             const nb::list&                       ignoreInterfragInteractions,
                             const nb::list&                       distanceConstraints,
                             const nb::list&                       positionConstraints,
                             const nb::list&                       angleConstraints,
                             const nb::list&                       torsionConstraints,
                             const nvMolKit::BatchHardwareOptions& hardwareOptions)
      : hardwareOptions_(hardwareOptions) {
    throwIfCudaError(cudaGetDevice(&gpuId_), "UFF wrapper/cudaGetDevice");
    molecules_              = nvMolKit::NanobindForcefield::extractMolecules(molecules);
    const int moleculeCount = static_cast<int>(molecules_.size());
    vdwThresholds_ = nvMolKit::NanobindForcefield::extractDoubleList(vdwThresholds, moleculeCount, "vdwThreshold");
    ignoreInterfragInteractions_ = nvMolKit::NanobindForcefield::extractBoolList(ignoreInterfragInteractions,
                                                                                 moleculeCount,
                                                                                 "ignoreInterfragInteractions");
    constraints_                 = extractAllConstraints(distanceConstraints,
                                         positionConstraints,
                                         angleConstraints,
                                         torsionConstraints,
                                         moleculeCount);
    buildForcefield();
  }

  std::vector<std::vector<double>> computeEnergy() {
    return computeBatchedEnergy(*forcefield_, devicePositions_, deviceEnergies_, conformerCounts_);
  }

  std::vector<std::vector<std::vector<double>>> computeGradients() {
    return computeBatchedGradients(*forcefield_, devicePositions_, deviceGradients_, conformerCounts_);
  }

  nb::tuple minimize(const int                    maxIters,
                     const double                 gradTol,
                     const std::string&           minimizerKind,
                     const nvMolKit::FireOptions& fireOptions) {
    const auto result = parseMinimizerKind(minimizerKind) == MinimizerKind::FIRE ?
                          nvMolKit::UFF::UFFMinimizeMoleculesConfsFire(molecules_,
                                                                       maxIters,
                                                                       fireOptions,
                                                                       vdwThresholds_,
                                                                       ignoreInterfragInteractions_,
                                                                       constraints_,
                                                                       hardwareOptions_) :
                          nvMolKit::UFF::UFFMinimizeMoleculesConfs(molecules_,
                                                                   maxIters,
                                                                   gradTol,
                                                                   vdwThresholds_,
                                                                   ignoreInterfragInteractions_,
                                                                   constraints_,
                                                                   hardwareOptions_);
    uploadConformerPositions(molecules_, devicePositions_);
    return nb::make_tuple(result.energies, convertConvergence(result.converged));
  }

  nb::object minimizeDevice(const int                    maxIters,
                            const double                 gradTol,
                            const int                    targetGpu,
                            const std::string&           minimizerKind,
                            const nvMolKit::FireOptions& fireOptions) {
    validateTargetGpu(targetGpu);
    auto result = parseMinimizerKind(minimizerKind) == MinimizerKind::FIRE ?
                    nvMolKit::UFF::UFFMinimizeMoleculesConfsFire(molecules_,
                                                                 maxIters,
                                                                 fireOptions,
                                                                 vdwThresholds_,
                                                                 ignoreInterfragInteractions_,
                                                                 constraints_,
                                                                 hardwareOptions_,
                                                                 nvMolKit::CoordinateOutput::DEVICE,
                                                                 gpuId_) :
                    nvMolKit::UFF::UFFMinimizeMoleculesConfs(molecules_,
                                                             maxIters,
                                                             gradTol,
                                                             vdwThresholds_,
                                                             ignoreInterfragInteractions_,
                                                             constraints_,
                                                             hardwareOptions_,
                                                             nvMolKit::CoordinateOutput::DEVICE,
                                                             gpuId_);
    if (!result.device.has_value()) {
      throw std::runtime_error("UFFMinimizeMoleculesConfs(DEVICE) returned no device result");
    }
    refreshPositions(*result.device);
    return nvMolKit::nanobind_bindings::buildOwningDevice3DResult(*result.device);
  }

  int gpuId() const { return gpuId_; }

 private:
  void validateTargetGpu(const int targetGpu) const {
    if (targetGpu >= 0 && targetGpu != gpuId_) {
      throw std::invalid_argument(
        "UFFBatchedForcefield.minimize(output=DEVICE) does not support target_gpu != wrapper GPU (" +
        std::to_string(targetGpu) + " vs " + std::to_string(gpuId_) + ").");
    }
  }

  void refreshPositions(nvMolKit::DeviceCoordResult& deviceResult) {
    if (deviceResult.positions.size() != devicePositions_.size()) {
      throw std::runtime_error("UFF minimizeDevice positions size does not match wrapper");
    }
    throwIfCudaError(cudaMemcpyAsync(devicePositions_.data(),
                                     deviceResult.positions.data(),
                                     devicePositions_.size() * sizeof(double),
                                     cudaMemcpyDeviceToDevice,
                                     devicePositions_.stream()),
                     "UFF minimizeDevice/positions refresh");
    throwIfCudaError(cudaStreamSynchronize(devicePositions_.stream()), "UFF minimizeDevice/positions refresh sync");
  }

  void buildForcefield() {
    const int                                 moleculeCount = static_cast<int>(molecules_.size());
    nvMolKit::UFF::BatchedMolecularSystemHost systemHost;
    nvMolKit::BatchedForcefieldMetadata       metadata;
    conformerCounts_.resize(moleculeCount);
    for (int moleculeIndex = 0; moleculeIndex < moleculeCount; ++moleculeIndex) {
      auto* molecule       = molecules_[moleculeIndex];
      int   conformerIndex = 0;
      for (auto conformer = molecule->beginConformers(); conformer != molecule->endConformers();
           ++conformer, ++conformerIndex) {
        std::vector<double> positions;
        nvMolKit::confPosToVect(**conformer, positions);
        auto contributions = nvMolKit::UFF::constructForcefieldContribs(*molecule,
                                                                        vdwThresholds_[moleculeIndex],
                                                                        (*conformer)->getId(),
                                                                        ignoreInterfragInteractions_[moleculeIndex]);
        constraints_[moleculeIndex].applyTo(contributions, positions);
        nvMolKit::UFF::addMoleculeToBatch(contributions,
                                          positions,
                                          systemHost,
                                          metadata,
                                          moleculeIndex,
                                          conformerIndex);
      }
      conformerCounts_[moleculeIndex] = conformerIndex;
    }
    forcefield_ = std::make_unique<nvMolKit::UFFBatchedForcefield>(systemHost, metadata);
    devicePositions_.setFromVector(systemHost.positions);
    deviceGradients_.resize(forcefield_->totalPositions());
    deviceEnergies_.resize(forcefield_->numMolecules());
  }

  std::vector<RDKit::ROMol*>                            molecules_;
  std::vector<double>                                   vdwThresholds_;
  std::vector<bool>                                     ignoreInterfragInteractions_;
  std::vector<ForcefieldConstraints::PerMolConstraints> constraints_;
  nvMolKit::BatchHardwareOptions                        hardwareOptions_;
  std::unique_ptr<nvMolKit::UFFBatchedForcefield>       forcefield_;
  nvMolKit::AsyncDeviceVector<double>                   devicePositions_;
  nvMolKit::AsyncDeviceVector<double>                   deviceGradients_;
  nvMolKit::AsyncDeviceVector<double>                   deviceEnergies_;
  std::vector<int>                                      conformerCounts_;
  int                                                   gpuId_ = 0;
};

}  // namespace

NB_MODULE(_batchedForcefield, module) {
  nb::module_::import_("nvmolkit._embedMolecules");
  nb::module_::import_("nvmolkit._types");
  nb::module_::import_("rdkit.ForceField.rdForceField");

  nb::class_<nvMolKit::MMFFProperties>(module, "MMFFProperties")
    .def(nb::init<>())
    .def_rw("variant", &nvMolKit::MMFFProperties::variant)
    .def_rw("dielectricConstant", &nvMolKit::MMFFProperties::dielectricConstant)
    .def_rw("dielectricModel", &nvMolKit::MMFFProperties::dielectricModel)
    .def_rw("nonBondedThreshold", &nvMolKit::MMFFProperties::nonBondedThreshold)
    .def_rw("ignoreInterfragInteractions", &nvMolKit::MMFFProperties::ignoreInterfragInteractions)
    .def_rw("bondTerm", &nvMolKit::MMFFProperties::bondTerm)
    .def_rw("angleTerm", &nvMolKit::MMFFProperties::angleTerm)
    .def_rw("stretchBendTerm", &nvMolKit::MMFFProperties::stretchBendTerm)
    .def_rw("oopTerm", &nvMolKit::MMFFProperties::oopTerm)
    .def_rw("torsionTerm", &nvMolKit::MMFFProperties::torsionTerm)
    .def_rw("vdwTerm", &nvMolKit::MMFFProperties::vdwTerm)
    .def_rw("eleTerm", &nvMolKit::MMFFProperties::eleTerm);

  module.def("buildMMFFPropertiesFromRDKit",
             &nvMolKit::NanobindForcefield::buildMMFFPropertiesFromRDKit,
             "rdkit_properties"_a,
             "non_bonded_threshold"_a,
             "ignore_interfrag_interactions"_a,
             "Build nvMolKit MMFF settings from RDKit MMFFMolProperties.");

  nb::class_<NativeMMFFBatchedForcefield>(module, "NativeMMFFBatchedForcefield")
    .def(nb::init<const nb::list&,
                  const nb::list&,
                  const nb::list&,
                  const nb::list&,
                  const nb::list&,
                  const nb::list&,
                  const nvMolKit::BatchHardwareOptions&>(),
         nb::keep_alive<1, 2>())
    .def("computeEnergy", &NativeMMFFBatchedForcefield::computeEnergy)
    .def("computeGradients", &NativeMMFFBatchedForcefield::computeGradients)
    .def("minimize", &NativeMMFFBatchedForcefield::minimize)
    .def("minimizeDevice", &NativeMMFFBatchedForcefield::minimizeDevice)
    .def("gpuId", &NativeMMFFBatchedForcefield::gpuId);

  nb::class_<NativeUFFBatchedForcefield>(module, "NativeUFFBatchedForcefield")
    .def(nb::init<const nb::list&,
                  const nb::list&,
                  const nb::list&,
                  const nb::list&,
                  const nb::list&,
                  const nb::list&,
                  const nb::list&,
                  const nvMolKit::BatchHardwareOptions&>(),
         nb::keep_alive<1, 2>())
    .def("computeEnergy", &NativeUFFBatchedForcefield::computeEnergy)
    .def("computeGradients", &NativeUFFBatchedForcefield::computeGradients)
    .def("minimize", &NativeUFFBatchedForcefield::minimize)
    .def("minimizeDevice", &NativeUFFBatchedForcefield::minimizeDevice)
    .def("gpuId", &NativeUFFBatchedForcefield::gpuId);
}
