// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nvmolkit/etkdg_stage_analysis.h"

#include <DistGeom/DistGeomUtils.h>
#include <ForceField/ForceField.h>
#include <Geometry/point.h>
#include <GraphMol/ROMol.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <cctype>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nvmolkit/boost_python_utils.h"
#include "src/embedder_utils.h"
#include "src/etkdg_impl.h"
#include "src/etkdg_stage_distgeom_minimize.h"
#include "src/etkdg_stage_etk_minimization.h"
#include "src/forcefields/dist_geom.h"
#include "src/minimizer/bfgs_minimize.h"
#include "src/utils/host_vector.h"

namespace bp = boost::python;

namespace nvMolKit {
namespace {

enum class AnalysisStage {
  FIRST,
  FOURTH,
  ETK
};

AnalysisStage parseStage(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::toupper(c); });
  if (name == "FIRST" || name == "DG_FIRST") {
    return AnalysisStage::FIRST;
  }
  if (name == "FOURTH" || name == "DG_FOURTH") {
    return AnalysisStage::FOURTH;
  }
  if (name == "ETK" || name == "ETK_3D") {
    return AnalysisStage::ETK;
  }
  throw std::invalid_argument("stage must be 'FIRST', 'FOURTH', or 'ETK'");
}

BfgsBackend parseBackend(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::toupper(c); });
  if (name == "BATCHED") {
    return BfgsBackend::BATCHED;
  }
  if (name == "PER_MOL" || name == "PER_MOLECULE") {
    return BfgsBackend::PER_MOLECULE;
  }
  if (name == "HYBRID") {
    return BfgsBackend::HYBRID;
  }
  throw std::invalid_argument("backend must be 'BATCHED', 'PER_MOL', or 'HYBRID'");
}

std::vector<double> extractFlatCoordinates(const bp::object& values, size_t expectedSize, size_t index) {
  std::vector<double>            result;
  bp::stl_input_iterator<double> it(values), end;
  for (; it != end; ++it) {
    result.push_back(*it);
  }
  if (result.size() != expectedSize) {
    throw std::invalid_argument("coordinates[" + std::to_string(index) + "] has " + std::to_string(result.size()) +
                                " values; expected " + std::to_string(expectedSize));
  }
  return result;
}

bp::list nestedCoordinates(const std::vector<double>& flat,
                           const std::vector<int>&    atomStarts,
                           int                        sourceDim,
                           int                        outputDim) {
  bp::list systems;
  for (size_t system = 0; system + 1 < atomStarts.size(); ++system) {
    bp::list atoms;
    for (int atom = atomStarts[system]; atom < atomStarts[system + 1]; ++atom) {
      bp::list point;
      for (int axis = 0; axis < outputDim; ++axis) {
        point.append(flat[static_cast<size_t>(atom) * sourceDim + axis]);
      }
      atoms.append(point);
    }
    systems.append(atoms);
  }
  return systems;
}

template <typename T> bp::list vectorList(const std::vector<T>& values) {
  bp::list result;
  for (const auto value : values) {
    result.append(value);
  }
  return result;
}

std::vector<std::unique_ptr<RDGeom::Point>> makePoints(const std::vector<double>& coords, int dim) {
  const size_t                                nAtoms = coords.size() / dim;
  std::vector<std::unique_ptr<RDGeom::Point>> points;
  points.reserve(nAtoms);
  for (size_t atom = 0; atom < nAtoms; ++atom) {
    std::unique_ptr<RDGeom::Point> point = dim == 3 ?
                                             std::unique_ptr<RDGeom::Point>(std::make_unique<RDGeom::Point3D>()) :
                                             std::unique_ptr<RDGeom::Point>(std::make_unique<RDGeom::PointND>(dim));
    for (int axis = 0; axis < dim; ++axis) {
      (*point)[axis] = coords[atom * dim + axis];
    }
    points.push_back(std::move(point));
  }
  return points;
}

std::unique_ptr<ForceFields::ForceField> makeReferenceField(AnalysisStage                                stage,
                                                            const detail::EmbedArgs&                     eargs,
                                                            RDKit::DGeomHelpers::EmbedParameters&        params,
                                                            std::vector<std::unique_ptr<RDGeom::Point>>& points) {
  std::unique_ptr<ForceFields::ForceField> field;
  if (stage == AnalysisStage::ETK) {
    RDGeom::Point3DPtrVect pointPtrs;
    pointPtrs.reserve(points.size());
    for (const auto& point : points) {
      pointPtrs.push_back(static_cast<RDGeom::Point3D*>(point.get()));
    }
    if (params.useBasicKnowledge) {
      field.reset(::DistGeom::construct3DForceField(*eargs.mmat, pointPtrs, eargs.etkdgDetails));
    } else {
      field.reset(::DistGeom::constructPlain3DForceField(*eargs.mmat, pointPtrs, eargs.etkdgDetails));
    }
  } else {
    RDGeom::PointPtrVect pointPtrs;
    pointPtrs.reserve(points.size());
    for (const auto& point : points) {
      pointPtrs.push_back(point.get());
    }
    const double            chiralWeight    = stage == AnalysisStage::FIRST ? 1.0 : 0.2;
    const double            fourthDimWeight = stage == AnalysisStage::FIRST ? 0.1 : 1.0;
    boost::dynamic_bitset<> fixedPoints(points.size());
    if (params.useRandomCoords && params.coordMap != nullptr) {
      for (const auto& [atomIdx, unused] : *params.coordMap) {
        static_cast<void>(unused);
        fixedPoints.set(atomIdx);
      }
    }
    field.reset(::DistGeom::constructForceField(*eargs.mmat,
                                                pointPtrs,
                                                eargs.chiralCenters,
                                                chiralWeight,
                                                fourthDimWeight,
                                                nullptr,
                                                params.basinThresh,
                                                &fixedPoints));
  }
  field->initialize();
  return field;
}

std::vector<double> flattenPoints(const std::vector<std::unique_ptr<RDGeom::Point>>& points, int dim) {
  std::vector<double> result;
  result.reserve(points.size() * dim);
  for (const auto& point : points) {
    for (int axis = 0; axis < dim; ++axis) {
      result.push_back((*point)[axis]);
    }
  }
  return result;
}

}  // namespace

bp::object analyzeETKDGStage(const bp::list&                             molecules,
                             const bp::list&                             coordinates,
                             const RDKit::DGeomHelpers::EmbedParameters& paramsIn,
                             const std::string&                          stageName,
                             const std::string&                          backendName,
                             const PrecisionOptions&                     precision,
                             bool                                        includeCpuReference) {
  auto mols = extractMolecules(molecules);
  if (mols.empty()) {
    throw std::invalid_argument("molecules must not be empty");
  }
  if (bp::len(coordinates) != static_cast<Py_ssize_t>(mols.size())) {
    throw std::invalid_argument("coordinates must contain one array per molecule");
  }

  const AnalysisStage stage      = parseStage(stageName);
  const BfgsBackend   backend    = parseBackend(backendName);
  const int           inputDim   = stage == AnalysisStage::ETK ? 3 : 4;
  constexpr int       contextDim = 4;
  auto                params     = paramsIn;

  detail::ETKDGContext             context;
  std::vector<detail::EmbedArgs>   eargs;
  std::vector<std::vector<double>> inputs;
  std::vector<const RDKit::ROMol*> constMols(mols.begin(), mols.end());
  context.nTotalSystems         = static_cast<int>(mols.size());
  context.systemHost.atomStarts = {0};
  eargs.reserve(mols.size());
  inputs.reserve(mols.size());

  for (size_t i = 0; i < mols.size(); ++i) {
    const size_t      expected = static_cast<size_t>(mols[i]->getNumAtoms()) * inputDim;
    auto              input    = extractFlatCoordinates(coordinates[i], expected, i);
    detail::EmbedArgs args;
    if (!DGeomHelpers::prepareEmbedderArgs(*mols[i], params, args, true)) {
      throw std::runtime_error("bounds smoothing failed for molecule " + std::to_string(i));
    }
    args.dim = contextDim;

    std::vector<double> coords4;
    if (inputDim == contextDim) {
      coords4 = input;
    } else {
      coords4.reserve(static_cast<size_t>(mols[i]->getNumAtoms()) * contextDim);
      for (size_t atom = 0; atom < mols[i]->getNumAtoms(); ++atom) {
        coords4.insert(coords4.end(), input.begin() + atom * 3, input.begin() + atom * 3 + 3);
        coords4.push_back(0.0);
      }
    }
    DistGeom::addMoleculeToContextWithPositions(coords4,
                                                contextDim,
                                                context.systemHost.atomStarts,
                                                context.systemHost.positions);
    inputs.push_back(std::move(input));
    eargs.push_back(std::move(args));
  }

  DistGeom::sendContextToDevice(context.systemHost.positions,
                                context.systemDevice.positions,
                                context.systemHost.atomStarts,
                                context.systemDevice.atomStarts);
  context.activeThisStage.resize(mols.size());
  context.failedThisStage.resize(mols.size());
  context.activeThisStage.copyFromHost(std::vector<uint8_t>(mols.size(), 1));
  context.failedThisStage.zero();

  BfgsBatchMinimizer                  minimizer(contextDim, DebugLevel::NONE, true, nullptr, backend, precision);
  std::unique_ptr<detail::ETKDGStage> stageRunner;
  if (stage == AnalysisStage::ETK) {
    stageRunner = std::make_unique<detail::ETKMinimizationStage>(constMols, eargs, params, context, minimizer, nullptr);
  } else {
    stageRunner = std::make_unique<detail::DistGeomMinimizeStage>(
      constMols,
      eargs,
      params,
      context,
      minimizer,
      stage == AnalysisStage::FIRST ? 1.0 : 0.2,
      stage == AnalysisStage::FIRST ? 0.1 : 1.0,
      stage == AnalysisStage::FIRST ? 400 : 200,
      stage == AnalysisStage::FIRST,
      stage == AnalysisStage::FIRST ? "First Minimization" : "Fourth Dimension Minimization");
  }
  stageRunner->execute(context);

  std::vector<double> gpuCoords(context.systemDevice.positions.size());
  context.systemDevice.positions.copyToHost(gpuCoords);
  std::vector<int16_t> gpuStatus(minimizer.statuses_.size());
  minimizer.statuses_.copyToHost(gpuStatus);
  std::vector<uint8_t> stageFailed(mols.size());
  context.failedThisStage.copyToHost(stageFailed);
  cudaCheckError(cudaDeviceSynchronize());

  std::vector<double> inputEnergies;
  std::vector<double> gpuEnergies;
  std::vector<double> cpuEnergies;
  std::vector<int>    cpuStatus;
  std::vector<double> cpuCoords;
  inputEnergies.reserve(mols.size());
  gpuEnergies.reserve(mols.size());
  if (includeCpuReference) {
    cpuEnergies.reserve(mols.size());
    cpuStatus.reserve(mols.size());
  }

  for (size_t i = 0; i < mols.size(); ++i) {
    const int atomBegin       = context.systemHost.atomStarts[i];
    const int atomEnd         = context.systemHost.atomStarts[i + 1];
    auto      referencePoints = makePoints(inputs[i], inputDim);
    auto      referenceField  = makeReferenceField(stage, eargs[i], params, referencePoints);
    inputEnergies.push_back(referenceField->calcEnergy());

    std::vector<double> gpuForScoring;
    gpuForScoring.reserve(static_cast<size_t>(atomEnd - atomBegin) * inputDim);
    for (int atom = atomBegin; atom < atomEnd; ++atom) {
      for (int axis = 0; axis < inputDim; ++axis) {
        gpuForScoring.push_back(gpuCoords[static_cast<size_t>(atom) * contextDim + axis]);
      }
    }
    gpuEnergies.push_back(referenceField->calcEnergy(gpuForScoring.data()));

    if (includeCpuReference) {
      int       status   = 0;
      const int maxIters = stage == AnalysisStage::FIRST ? 400 : (stage == AnalysisStage::FOURTH ? 200 : 300);
      if (referenceField->calcEnergy() > 1e-5) {
        do {
          status = referenceField->minimize(maxIters, params.optimizerForceTol);
        } while (stage != AnalysisStage::ETK && status != 0);
      }
      cpuStatus.push_back(status);
      cpuEnergies.push_back(referenceField->calcEnergy());
      auto systemCpuCoords = flattenPoints(referencePoints, inputDim);
      cpuCoords.insert(cpuCoords.end(), systemCpuCoords.begin(), systemCpuCoords.end());
    }
  }

  bp::dict result;
  result["stage"]           = stageName;
  result["dimension"]       = inputDim;
  result["input_energies"]  = vectorList(inputEnergies);
  result["gpu_coordinates"] = nestedCoordinates(gpuCoords, context.systemHost.atomStarts, contextDim, inputDim);
  result["gpu_energies"]    = vectorList(gpuEnergies);
  result["gpu_status"]      = vectorList(gpuStatus);
  bp::list gpuConverged;
  for (const auto status : gpuStatus) {
    gpuConverged.append(status == 0);
  }
  result["gpu_converged"] = gpuConverged;
  result["stage_failed"]  = vectorList(stageFailed);
  if (includeCpuReference) {
    result["cpu_coordinates"] = nestedCoordinates(cpuCoords, context.systemHost.atomStarts, inputDim, inputDim);
    result["cpu_energies"]    = vectorList(cpuEnergies);
    result["cpu_status"]      = vectorList(cpuStatus);
    bp::list cpuConverged;
    for (const auto status : cpuStatus) {
      cpuConverged.append(status == 0);
    }
    result["cpu_converged"] = cpuConverged;
  } else {
    result["cpu_coordinates"] = bp::object();
    result["cpu_energies"]    = bp::object();
    result["cpu_status"]      = bp::object();
    result["cpu_converged"]   = bp::object();
  }
  return result;
}

}  // namespace nvMolKit
