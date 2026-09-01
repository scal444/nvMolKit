// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nvmolkit/etkdg_stage_analysis.h"

#include <DistGeom/DistGeomUtils.h>
#include <ForceField/ForceField.h>
#include <Geometry/point.h>
#include <GraphMol/ROMol.h>
#include <Numerics/Optimizer/BFGSOpt.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <cctype>
#include <cmath>
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

struct ReferenceEnergy {
  ForceFields::ForceField* field;
  double                   operator()(double* pos) const { return field->calcEnergy(pos); }
};

struct ReferenceGradient {
  ForceFields::ForceField* field;
  double                   operator()(double* pos, double* grad) const {
    const unsigned int dim = field->numPoints() * field->dimension();
    std::fill(grad, grad + dim, 0.0);
    field->calcGrad(pos, grad);
    double maxGrad = 0.0;
    double scale   = 0.1;
    for (unsigned int i = 0; i < dim; ++i) {
      grad[i] *= scale;
      maxGrad = std::max(maxGrad, std::abs(grad[i]));
    }
    if (maxGrad > 10.0) {
      while (maxGrad * scale > 10.0) {
        scale *= 0.5;
      }
      for (unsigned int i = 0; i < dim; ++i) {
        grad[i] *= scale;
      }
    }
    return scale;
  }
};

// RDKit's public ForceField API discards BFGS' iteration count and cannot
// disable convergence. This analysis-only copy follows its scalar update loop,
// with the two convergence returns conditional on fixedSteps.
int minimizeReference(ForceFields::ForceField& field,
                      std::vector<double>&     pos,
                      double                   gradTol,
                      unsigned int             maxIts,
                      bool                     fixedSteps,
                      unsigned int&            numIters) {
  const unsigned int  dim = pos.size();
  std::vector<double> grad(dim), dGrad(dim), hessDGrad(dim), xi(dim), newPos(dim);
  std::vector<double> invHessian(static_cast<size_t>(dim) * dim, 0.0);
  ReferenceEnergy     func{&field};
  ReferenceGradient   gradFunc{&field};
  double              fp = func(pos.data());
  gradFunc(pos.data(), grad.data());
  double sum = 0.0;
  for (unsigned int i = 0; i < dim; ++i) {
    invHessian[static_cast<size_t>(i) * dim + i] = 1.0;
    xi[i]                                        = -grad[i];
    sum += pos[i] * pos[i];
  }
  const double maxStep = BFGSOpt::MAXSTEP * std::max(std::sqrt(sum), static_cast<double>(dim));
  numIters             = 0;
  for (unsigned int iter = 1; iter <= maxIts; ++iter) {
    numIters      = iter;
    int    status = -1;
    double funcVal;
    BFGSOpt::linearSearch(dim, pos.data(), fp, grad.data(), xi.data(), newPos.data(), funcVal, func, maxStep, status);
    if (status < 0 && fixedSteps) {
      std::copy(pos.begin(), pos.end(), newPos.begin());
      funcVal = fp;
    } else {
      CHECK_INVARIANT(status >= 0, "bad direction in analysis linearSearch");
    }
    fp          = funcVal;
    double test = 0.0;
    for (unsigned int i = 0; i < dim; ++i) {
      xi[i]    = newPos[i] - pos[i];
      pos[i]   = newPos[i];
      test     = std::max(test, std::abs(xi[i]) / std::max(std::abs(pos[i]), 1.0));
      dGrad[i] = grad[i];
    }
    if (!fixedSteps && test < BFGSOpt::TOLX) {
      return 0;
    }
    const double gradScale = gradFunc(pos.data(), grad.data());
    test                   = 0.0;
    const double term      = std::max(std::abs(funcVal) * gradScale, 1.0);
    for (unsigned int i = 0; i < dim; ++i) {
      test     = std::max(test, std::abs(grad[i]) * std::max(std::abs(pos[i]), 1.0));
      dGrad[i] = grad[i] - dGrad[i];
    }
    if (!fixedSteps && test / term < gradTol) {
      return 0;
    }
    double fac = 0.0, fae = 0.0, sumDGrad = 0.0, sumXi = 0.0;
    for (unsigned int i = 0; i < dim; ++i) {
      hessDGrad[i] = 0.0;
      for (unsigned int j = 0; j < dim; ++j) {
        hessDGrad[i] += invHessian[static_cast<size_t>(i) * dim + j] * dGrad[j];
      }
      fac += dGrad[i] * xi[i];
      fae += dGrad[i] * hessDGrad[i];
      sumDGrad += dGrad[i] * dGrad[i];
      sumXi += xi[i] * xi[i];
    }
    if (fac > std::sqrt(BFGSOpt::EPS * sumDGrad * sumXi)) {
      fac              = 1.0 / fac;
      const double fad = 1.0 / fae;
      for (unsigned int i = 0; i < dim; ++i) {
        dGrad[i] = fac * xi[i] - fad * hessDGrad[i];
      }
      for (unsigned int i = 0; i < dim; ++i) {
        for (unsigned int j = i; j < dim; ++j) {
          const double update = fac * xi[i] * xi[j] - fad * hessDGrad[i] * hessDGrad[j] + fae * dGrad[i] * dGrad[j];
          invHessian[static_cast<size_t>(i) * dim + j] += update;
          invHessian[static_cast<size_t>(j) * dim + i] = invHessian[static_cast<size_t>(i) * dim + j];
        }
      }
    }
    for (unsigned int i = 0; i < dim; ++i) {
      xi[i] = 0.0;
      for (unsigned int j = 0; j < dim; ++j) {
        xi[i] -= invHessian[static_cast<size_t>(i) * dim + j] * grad[j];
      }
    }
  }
  return 1;
}

}  // namespace

bp::object analyzeETKDGStage(const bp::list&                             molecules,
                             const bp::list&                             coordinates,
                             const RDKit::DGeomHelpers::EmbedParameters& paramsIn,
                             const std::string&                          stageName,
                             const std::string&                          backendName,
                             const PrecisionOptions&                     precision,
                             bool                                        includeCpuReference,
                             int                                         fixedSteps,
                             int                                         maxSteps) {
  auto mols = extractMolecules(molecules);
  if (mols.empty()) {
    throw std::invalid_argument("molecules must not be empty");
  }
  if (bp::len(coordinates) != static_cast<Py_ssize_t>(mols.size())) {
    throw std::invalid_argument("coordinates must contain one array per molecule");
  }

  const AnalysisStage stage   = parseStage(stageName);
  const BfgsBackend   backend = parseBackend(backendName);
  if (backend != BfgsBackend::BATCHED) {
    throw std::invalid_argument("stage iteration analysis requires backend='BATCHED'");
  }
  if (fixedSteps == 0 || fixedSteps < -1 || maxSteps == 0 || maxSteps < -1) {
    throw std::invalid_argument("fixedSteps and maxSteps must be positive or -1");
  }
  if (fixedSteps > 0 && maxSteps > 0) {
    throw std::invalid_argument("fixedSteps and maxSteps are mutually exclusive");
  }
  const bool    fixedMode     = fixedSteps > 0;
  const int     defaultSteps  = stage == AnalysisStage::FIRST ? 400 : (stage == AnalysisStage::FOURTH ? 200 : 300);
  const int     analysisSteps = fixedMode ? fixedSteps : (maxSteps > 0 ? maxSteps : defaultSteps);
  const int     inputDim      = stage == AnalysisStage::ETK ? 3 : 4;
  constexpr int contextDim    = 4;
  auto          params        = paramsIn;

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

  BfgsBatchMinimizer minimizer(contextDim, DebugLevel::NONE, true, nullptr, backend, precision);
  if (stage == AnalysisStage::ETK) {
    detail::ETKMinimizationStage stageRunner(constMols, eargs, params, context, minimizer, nullptr);
    stageRunner.executeAnalysis(context, analysisSteps, fixedMode);
  } else {
    detail::DistGeomMinimizeStage stageRunner(
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
    stageRunner.executeAnalysis(context, analysisSteps, fixedMode);
  }

  std::vector<double> gpuCoords(context.systemDevice.positions.size());
  context.systemDevice.positions.copyToHost(gpuCoords);
  std::vector<int16_t> gpuStatus(minimizer.statuses_.size());
  minimizer.statuses_.copyToHost(gpuStatus);
  std::vector<int> gpuIterations(minimizer.iterationCounts_.size());
  minimizer.iterationCounts_.copyToHost(gpuIterations);
  std::vector<uint8_t> stageFailed(mols.size());
  context.failedThisStage.copyToHost(stageFailed);
  cudaCheckError(cudaDeviceSynchronize());

  std::vector<double> inputEnergies;
  std::vector<double> gpuEnergies;
  std::vector<double> cpuEnergies;
  std::vector<int>    cpuStatus;
  std::vector<int>    cpuIterations;
  std::vector<double> cpuCoords;
  inputEnergies.reserve(mols.size());
  gpuEnergies.reserve(mols.size());
  if (includeCpuReference) {
    cpuEnergies.reserve(mols.size());
    cpuStatus.reserve(mols.size());
    cpuIterations.reserve(mols.size());
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
      int          status     = 0;
      unsigned int iterations = 0;
      if (fixedMode || referenceField->calcEnergy() > 1e-5) {
        auto systemCpuCoords = flattenPoints(referencePoints, inputDim);
        status               = minimizeReference(*referenceField,
                                   systemCpuCoords,
                                   params.optimizerForceTol,
                                   analysisSteps,
                                   fixedMode,
                                   iterations);
        for (size_t atom = 0; atom < referencePoints.size(); ++atom) {
          for (int axis = 0; axis < inputDim; ++axis) {
            (*referencePoints[atom])[axis] = systemCpuCoords[atom * inputDim + axis];
          }
        }
      }
      cpuStatus.push_back(status);
      cpuIterations.push_back(iterations);
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
  result["gpu_iterations"]  = vectorList(gpuIterations);
  result["fixed_steps"]     = fixedMode ? bp::object(fixedSteps) : bp::object();
  result["max_steps"]       = analysisSteps;
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
    result["cpu_iterations"]  = vectorList(cpuIterations);
    bp::list cpuConverged;
    for (const auto status : cpuStatus) {
      cpuConverged.append(status == 0);
    }
    result["cpu_converged"] = cpuConverged;
  } else {
    result["cpu_coordinates"] = bp::object();
    result["cpu_energies"]    = bp::object();
    result["cpu_status"]      = bp::object();
    result["cpu_iterations"]  = bp::object();
    result["cpu_converged"]   = bp::object();
  }
  return result;
}

}  // namespace nvMolKit
