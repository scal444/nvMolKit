#include <ForceField/ForceField.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ROMol.h>

#include <boost/python.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "device_vector.h"
#include "dg_batched_forcefield.h"
#include "dist_geom_flattened_builder.h"
#include "embedder_utils.h"
#include "etk_batched_forcefield.h"
#include "ff_utils.h"
#include "mmff_batched_forcefield.h"
#include "mmff_flattened_builder.h"
#include "mmff_properties.h"

namespace bp = boost::python;

namespace {

constexpr double kRadiansToDegrees = 180.0 / M_PI;

template <typename T> bp::list vectorToList(const std::vector<T>& vec) {
  bp::list list;
  for (const auto& value : vec) {
    list.append(value);
  }
  return list;
}

template <typename T> bp::list vectorOfVectorsToList(const std::vector<std::vector<T>>& vecOfVecs) {
  bp::list list;
  for (const auto& vec : vecOfVecs) {
    list.append(vectorToList(vec));
  }
  return list;
}

std::vector<RDKit::ROMol*> extractMolecules(const bp::list& molecules) {
  std::vector<RDKit::ROMol*> mols;
  mols.reserve(bp::len(molecules));
  for (int i = 0; i < bp::len(molecules); ++i) {
    auto* mol = bp::extract<RDKit::ROMol*>(bp::object(molecules[i]));
    if (mol == nullptr) {
      throw std::invalid_argument("Invalid molecule at index " + std::to_string(i));
    }
    mols.push_back(mol);
  }
  return mols;
}

void throwIfCudaError(const cudaError_t err, const std::string& context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(context + ": " + cudaGetErrorString(err));
  }
}

std::vector<double> copyDeviceVector(nvMolKit::AsyncDeviceVector<double>& deviceVec) {
  std::vector<double> host(deviceVec.size(), 0.0);
  deviceVec.copyToHost(host);
  cudaDeviceSynchronize();
  return host;
}

std::vector<double> pad3Dto4D(const std::vector<double>& positions3D) {
  std::vector<double> positions4D;
  positions4D.reserve(positions3D.size() / 3 * 4);
  for (size_t atomIdx = 0; atomIdx < positions3D.size() / 3; ++atomIdx) {
    positions4D.push_back(positions3D[3 * atomIdx + 0]);
    positions4D.push_back(positions3D[3 * atomIdx + 1]);
    positions4D.push_back(positions3D[3 * atomIdx + 2]);
    positions4D.push_back(100.0);
  }
  return positions4D;
}

std::vector<std::vector<double>> splitGradients(const std::vector<double>& flatGrad,
                                                const std::vector<int>&    atomStarts,
                                                const int                  dim) {
  std::vector<std::vector<double>> grads;
  grads.reserve(atomStarts.size() - 1);
  for (size_t molIdx = 0; molIdx + 1 < atomStarts.size(); ++molIdx) {
    const int start = atomStarts[molIdx] * dim;
    const int end   = atomStarts[molIdx + 1] * dim;
    grads.emplace_back(flatGrad.begin() + start, flatGrad.begin() + end);
  }
  return grads;
}

std::vector<std::vector<double>> strip4DGradientsTo3D(const std::vector<double>& flatGrad4D,
                                                       const std::vector<int>&    atomStarts) {
  std::vector<std::vector<double>> grads;
  grads.reserve(atomStarts.size() - 1);
  for (size_t molIdx = 0; molIdx + 1 < atomStarts.size(); ++molIdx) {
    std::vector<double> molGrad;
    molGrad.reserve((atomStarts[molIdx + 1] - atomStarts[molIdx]) * 3);
    for (int atomIdx = atomStarts[molIdx]; atomIdx < atomStarts[molIdx + 1]; ++atomIdx) {
      molGrad.push_back(flatGrad4D[4 * atomIdx + 0]);
      molGrad.push_back(flatGrad4D[4 * atomIdx + 1]);
      molGrad.push_back(flatGrad4D[4 * atomIdx + 2]);
    }
    grads.push_back(std::move(molGrad));
  }
  return grads;
}

nvMolKit::MMFFPropertiesNative extractMMFFProperties(const bp::object& obj) {
  nvMolKit::MMFFPropertiesNative props;
  if (obj.is_none()) {
    return props;
  }
  props.variant = bp::extract<std::string>(obj.attr("variant"));
  props.dielectricConstant = bp::extract<double>(obj.attr("dielectricConstant"));
  props.dielectricModel = bp::extract<int>(obj.attr("dielectricModel"));
  props.nonBondedThreshold = bp::extract<double>(obj.attr("nonBondedThreshold"));
  props.ignoreInterfragInteractions = bp::extract<bool>(obj.attr("ignoreInterfragInteractions"));
  props.bondTerm = bp::extract<bool>(obj.attr("bondTerm"));
  props.angleTerm = bp::extract<bool>(obj.attr("angleTerm"));
  props.stretchBendTerm = bp::extract<bool>(obj.attr("stretchBendTerm"));
  props.oopTerm = bp::extract<bool>(obj.attr("oopTerm"));
  props.torsionTerm = bp::extract<bool>(obj.attr("torsionTerm"));
  props.vdwTerm = bp::extract<bool>(obj.attr("vdwTerm"));
  props.eleTerm = bp::extract<bool>(obj.attr("eleTerm"));
  return props;
}

std::vector<nvMolKit::MMFFPropertiesNative> extractMMFFPropertiesList(const bp::list& properties, const int expectedSize) {
  if (bp::len(properties) != expectedSize) {
    throw std::invalid_argument("Expected " + std::to_string(expectedSize) + " MMFF properties objects, got " +
                                std::to_string(bp::len(properties)));
  }
  std::vector<nvMolKit::MMFFPropertiesNative> out;
  out.reserve(expectedSize);
  for (int i = 0; i < expectedSize; ++i) {
    out.push_back(extractMMFFProperties(bp::object(properties[i])));
  }
  return out;
}

std::vector<int> extractIntList(const bp::list& values, const int expectedSize, const std::string& name) {
  if (bp::len(values) != expectedSize) {
    throw std::invalid_argument("Expected " + std::to_string(expectedSize) + " values for " + name + ", got " +
                                std::to_string(bp::len(values)));
  }
  std::vector<int> out;
  out.reserve(expectedSize);
  for (int i = 0; i < expectedSize; ++i) {
    out.push_back(bp::extract<int>(bp::object(values[i])));
  }
  return out;
}

struct DistanceConstraintSpec {
  int    idx1          = -1;
  int    idx2          = -1;
  bool   relative      = false;
  double minLen        = 0.0;
  double maxLen        = 0.0;
  double forceConstant = 0.0;
};

struct PositionConstraintSpec {
  int    idx           = -1;
  double maxDispl      = 0.0;
  double forceConstant = 0.0;
};

struct AngleConstraintSpec {
  int    idx1          = -1;
  int    idx2          = -1;
  int    idx3          = -1;
  bool   relative      = false;
  double minAngleDeg   = 0.0;
  double maxAngleDeg   = 0.0;
  double forceConstant = 0.0;
};

struct TorsionConstraintSpec {
  int    idx1           = -1;
  int    idx2           = -1;
  int    idx3           = -1;
  int    idx4           = -1;
  bool   relative       = false;
  double minDihedralDeg = 0.0;
  double maxDihedralDeg = 0.0;
  double forceConstant  = 0.0;
};

template <typename Spec, typename Parser>
std::vector<std::vector<Spec>> extractConstraintLists(const bp::list& outerList,
                                                      const int       expectedSize,
                                                      const Parser&   parser,
                                                      const std::string& name) {
  if (bp::len(outerList) != expectedSize) {
    throw std::invalid_argument("Expected " + std::to_string(expectedSize) + " entries for " + name + ", got " +
                                std::to_string(bp::len(outerList)));
  }
  std::vector<std::vector<Spec>> allSpecs(expectedSize);
  for (int molIdx = 0; molIdx < expectedSize; ++molIdx) {
    const bp::list innerList = bp::extract<bp::list>(bp::object(outerList[molIdx]));
    auto&          specs     = allSpecs[molIdx];
    specs.reserve(bp::len(innerList));
    for (int j = 0; j < bp::len(innerList); ++j) {
      specs.push_back(parser(bp::extract<bp::tuple>(bp::object(innerList[j]))));
    }
  }
  return allSpecs;
}

DistanceConstraintSpec parseDistanceConstraintTuple(const bp::tuple& value) {
  if (bp::len(value) != 6) {
    throw std::invalid_argument("Distance constraint tuples must have 6 elements");
  }
  return {bp::extract<int>(value[0]),
          bp::extract<int>(value[1]),
          bp::extract<bool>(value[2]),
          bp::extract<double>(value[3]),
          bp::extract<double>(value[4]),
          bp::extract<double>(value[5])};
}

PositionConstraintSpec parsePositionConstraintTuple(const bp::tuple& value) {
  if (bp::len(value) != 3) {
    throw std::invalid_argument("Position constraint tuples must have 3 elements");
  }
  return {bp::extract<int>(value[0]), bp::extract<double>(value[1]), bp::extract<double>(value[2])};
}

AngleConstraintSpec parseAngleConstraintTuple(const bp::tuple& value) {
  if (bp::len(value) != 7) {
    throw std::invalid_argument("Angle constraint tuples must have 7 elements");
  }
  return {bp::extract<int>(value[0]),
          bp::extract<int>(value[1]),
          bp::extract<int>(value[2]),
          bp::extract<bool>(value[3]),
          bp::extract<double>(value[4]),
          bp::extract<double>(value[5]),
          bp::extract<double>(value[6])};
}

TorsionConstraintSpec parseTorsionConstraintTuple(const bp::tuple& value) {
  if (bp::len(value) != 8) {
    throw std::invalid_argument("Torsion constraint tuples must have 8 elements");
  }
  return {bp::extract<int>(value[0]),
          bp::extract<int>(value[1]),
          bp::extract<int>(value[2]),
          bp::extract<int>(value[3]),
          bp::extract<bool>(value[4]),
          bp::extract<double>(value[5]),
          bp::extract<double>(value[6]),
          bp::extract<double>(value[7])};
}

void validateAtomIndex(const int idx, const int numAtoms, const std::string& what) {
  if (idx < 0 || idx >= numAtoms) {
    throw std::out_of_range(what + " index " + std::to_string(idx) + " is out of range for molecule with " +
                            std::to_string(numAtoms) + " atoms");
  }
}

double distanceFromPositions(const std::vector<double>& positions, const int idx1, const int idx2) {
  const double dx = positions[3 * idx1 + 0] - positions[3 * idx2 + 0];
  const double dy = positions[3 * idx1 + 1] - positions[3 * idx2 + 1];
  const double dz = positions[3 * idx1 + 2] - positions[3 * idx2 + 2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double computeAngleDeg(const std::vector<double>& positions, const int idx1, const int idx2, const int idx3) {
  const double r1x = positions[3 * idx1 + 0] - positions[3 * idx2 + 0];
  const double r1y = positions[3 * idx1 + 1] - positions[3 * idx2 + 1];
  const double r1z = positions[3 * idx1 + 2] - positions[3 * idx2 + 2];
  const double r2x = positions[3 * idx3 + 0] - positions[3 * idx2 + 0];
  const double r2y = positions[3 * idx3 + 1] - positions[3 * idx2 + 1];
  const double r2z = positions[3 * idx3 + 2] - positions[3 * idx2 + 2];
  const double lengthSq1 = std::max(1.0e-5, r1x * r1x + r1y * r1y + r1z * r1z);
  const double lengthSq2 = std::max(1.0e-5, r2x * r2x + r2y * r2y + r2z * r2z);
  const double cosTheta =
    std::max(-1.0, std::min(1.0, (r1x * r2x + r1y * r2y + r1z * r2z) / std::sqrt(lengthSq1 * lengthSq2)));
  return kRadiansToDegrees * std::acos(cosTheta);
}

double normalizeAngleDeg(double angleDeg) {
  RDKit::ForceFieldsHelper::normalizeAngleDeg(angleDeg);
  return angleDeg;
}

double computeDihedralDeg(const std::vector<double>& positions, const int idx1, const int idx2, const int idx3, const int idx4) {
  double dihedral = 0.0;
  RDKit::ForceFieldsHelper::computeDihedral(positions.data(), idx1, idx2, idx3, idx4, &dihedral);
  return kRadiansToDegrees * dihedral;
}

void appendDistanceConstraint(nvMolKit::MMFF::EnergyForceContribsHost& contribs,
                              const std::vector<double>&               positions,
                              const DistanceConstraintSpec&            spec) {
  const int numAtoms = static_cast<int>(positions.size() / 3);
  validateAtomIndex(spec.idx1, numAtoms, "Distance constraint atom");
  validateAtomIndex(spec.idx2, numAtoms, "Distance constraint atom");
  double minLen = spec.minLen;
  double maxLen = spec.maxLen;
  if (maxLen < minLen) {
    throw std::invalid_argument("Distance constraint maxLen must be >= minLen");
  }
  if (spec.relative) {
    const double distance = distanceFromPositions(positions, spec.idx1, spec.idx2);
    minLen                = std::max(minLen + distance, 0.0);
    maxLen                = std::max(maxLen + distance, 0.0);
  }
  contribs.distanceConstraintTerms.idx1.push_back(spec.idx1);
  contribs.distanceConstraintTerms.idx2.push_back(spec.idx2);
  contribs.distanceConstraintTerms.minLen.push_back(minLen);
  contribs.distanceConstraintTerms.maxLen.push_back(maxLen);
  contribs.distanceConstraintTerms.forceConstant.push_back(spec.forceConstant);
}

void appendPositionConstraint(nvMolKit::MMFF::EnergyForceContribsHost& contribs,
                              const std::vector<double>&               positions,
                              const PositionConstraintSpec&            spec) {
  const int numAtoms = static_cast<int>(positions.size() / 3);
  validateAtomIndex(spec.idx, numAtoms, "Position constraint atom");
  contribs.positionConstraintTerms.idx.push_back(spec.idx);
  contribs.positionConstraintTerms.refX.push_back(positions[3 * spec.idx + 0]);
  contribs.positionConstraintTerms.refY.push_back(positions[3 * spec.idx + 1]);
  contribs.positionConstraintTerms.refZ.push_back(positions[3 * spec.idx + 2]);
  contribs.positionConstraintTerms.maxDispl.push_back(spec.maxDispl);
  contribs.positionConstraintTerms.forceConstant.push_back(spec.forceConstant);
}

void appendAngleConstraint(nvMolKit::MMFF::EnergyForceContribsHost& contribs,
                           const std::vector<double>&               positions,
                           const AngleConstraintSpec&               spec) {
  const int numAtoms = static_cast<int>(positions.size() / 3);
  validateAtomIndex(spec.idx1, numAtoms, "Angle constraint atom");
  validateAtomIndex(spec.idx2, numAtoms, "Angle constraint atom");
  validateAtomIndex(spec.idx3, numAtoms, "Angle constraint atom");
  if (spec.maxAngleDeg < spec.minAngleDeg) {
    throw std::invalid_argument("Angle constraint maxAngleDeg must be >= minAngleDeg");
  }
  double minAngleDeg = spec.minAngleDeg;
  double maxAngleDeg = spec.maxAngleDeg;
  if (spec.relative) {
    const double angle = computeAngleDeg(positions, spec.idx1, spec.idx2, spec.idx3);
    minAngleDeg += angle;
    maxAngleDeg += angle;
  }
  if (minAngleDeg < 0.0 || minAngleDeg > 180.0 || maxAngleDeg < 0.0 || maxAngleDeg > 180.0) {
    throw std::invalid_argument("Angle constraint bounds must be within [0, 180]");
  }
  contribs.angleConstraintTerms.idx1.push_back(spec.idx1);
  contribs.angleConstraintTerms.idx2.push_back(spec.idx2);
  contribs.angleConstraintTerms.idx3.push_back(spec.idx3);
  contribs.angleConstraintTerms.minAngleDeg.push_back(minAngleDeg);
  contribs.angleConstraintTerms.maxAngleDeg.push_back(maxAngleDeg);
  contribs.angleConstraintTerms.forceConstant.push_back(spec.forceConstant);
}

void appendTorsionConstraint(nvMolKit::MMFF::EnergyForceContribsHost& contribs,
                             const std::vector<double>&               positions,
                             const TorsionConstraintSpec&             spec) {
  const int numAtoms = static_cast<int>(positions.size() / 3);
  validateAtomIndex(spec.idx1, numAtoms, "Torsion constraint atom");
  validateAtomIndex(spec.idx2, numAtoms, "Torsion constraint atom");
  validateAtomIndex(spec.idx3, numAtoms, "Torsion constraint atom");
  validateAtomIndex(spec.idx4, numAtoms, "Torsion constraint atom");
  if (spec.maxDihedralDeg < spec.minDihedralDeg) {
    throw std::invalid_argument("Torsion constraint maxDihedralDeg must be >= minDihedralDeg");
  }
  double minDihedralDeg = spec.minDihedralDeg;
  double maxDihedralDeg = spec.maxDihedralDeg;
  if (spec.relative) {
    const double dihedral = computeDihedralDeg(positions, spec.idx1, spec.idx2, spec.idx3, spec.idx4);
    minDihedralDeg += dihedral;
    maxDihedralDeg += dihedral;
  }
  minDihedralDeg = normalizeAngleDeg(minDihedralDeg);
  maxDihedralDeg = normalizeAngleDeg(maxDihedralDeg);
  contribs.torsionConstraintTerms.idx1.push_back(spec.idx1);
  contribs.torsionConstraintTerms.idx2.push_back(spec.idx2);
  contribs.torsionConstraintTerms.idx3.push_back(spec.idx3);
  contribs.torsionConstraintTerms.idx4.push_back(spec.idx4);
  contribs.torsionConstraintTerms.minDihedralDeg.push_back(minDihedralDeg);
  contribs.torsionConstraintTerms.maxDihedralDeg.push_back(maxDihedralDeg);
  contribs.torsionConstraintTerms.forceConstant.push_back(spec.forceConstant);
}

class NativeMMFFBatchedForcefield {
 public:
  NativeMMFFBatchedForcefield(const bp::list& molecules,
                              const bp::list& properties,
                              const bp::list& confIds,
                              const bp::list& distanceConstraints,
                              const bp::list& positionConstraints,
                              const bp::list& angleConstraints,
                              const bp::list& torsionConstraints) {
    const auto mols     = extractMolecules(molecules);
    const int  numMols  = static_cast<int>(mols.size());
    const auto props    = extractMMFFPropertiesList(properties, numMols);
    const auto confList = extractIntList(confIds, numMols, "conf_id");
    const auto distanceConstraintLists =
      extractConstraintLists<DistanceConstraintSpec>(distanceConstraints,
                                                     numMols,
                                                     parseDistanceConstraintTuple,
                                                     "distance constraints");
    const auto positionConstraintLists =
      extractConstraintLists<PositionConstraintSpec>(positionConstraints,
                                                     numMols,
                                                     parsePositionConstraintTuple,
                                                     "position constraints");
    const auto angleConstraintLists =
      extractConstraintLists<AngleConstraintSpec>(angleConstraints, numMols, parseAngleConstraintTuple, "angle constraints");
    const auto torsionConstraintLists = extractConstraintLists<TorsionConstraintSpec>(
      torsionConstraints, numMols, parseTorsionConstraintTuple, "torsion constraints");

    for (int molIdx = 0; molIdx < numMols; ++molIdx) {
      std::vector<double> positions;
      nvMolKit::confPosToVect(*mols[molIdx], positions, confList[molIdx]);
      auto ffParams = nvMolKit::MMFF::constructForcefieldContribs(*mols[molIdx], props[molIdx], confList[molIdx]);
      for (const auto& spec : distanceConstraintLists[molIdx]) {
        appendDistanceConstraint(ffParams, positions, spec);
      }
      for (const auto& spec : positionConstraintLists[molIdx]) {
        appendPositionConstraint(ffParams, positions, spec);
      }
      for (const auto& spec : angleConstraintLists[molIdx]) {
        appendAngleConstraint(ffParams, positions, spec);
      }
      for (const auto& spec : torsionConstraintLists[molIdx]) {
        appendTorsionConstraint(ffParams, positions, spec);
      }
      nvMolKit::MMFF::addMoleculeToBatch(ffParams, positions, systemHost_);
    }

    forcefield_ = std::make_unique<nvMolKit::MMFFBatchedForcefield>(systemHost_);
    positionsDevice_.setFromVector(systemHost_.positions);
    energyOutsDevice_.resize(systemHost_.indices.atomStarts.size() - 1);
    energyOutsDevice_.zero();
    gradDevice_.resize(systemHost_.positions.size());
    gradDevice_.zero();
  }

  bp::list computeEnergy() {
    energyOutsDevice_.zero();
    throwIfCudaError(forcefield_->computeEnergy(energyOutsDevice_.data(), positionsDevice_.data()),
                     "NativeMMFFBatchedForcefield.computeEnergy");
    return vectorToList(copyDeviceVector(energyOutsDevice_));
  }

  bp::list computeGradients() {
    gradDevice_.zero();
    throwIfCudaError(forcefield_->computeGradients(gradDevice_.data(), positionsDevice_.data()),
                     "NativeMMFFBatchedForcefield.computeGradients");
    return vectorOfVectorsToList(splitGradients(copyDeviceVector(gradDevice_), systemHost_.indices.atomStarts, 3));
  }

 private:
  std::unique_ptr<nvMolKit::MMFFBatchedForcefield> forcefield_;
  nvMolKit::AsyncDeviceVector<double>              positionsDevice_;
  nvMolKit::AsyncDeviceVector<double>              energyOutsDevice_;
  nvMolKit::AsyncDeviceVector<double>              gradDevice_;
  nvMolKit::MMFF::BatchedMolecularSystemHost       systemHost_;
};

bp::list computeMMFFEnergies(const bp::list& molecules, const double nonBondedThreshold) {
  const int numMols = bp::len(molecules);
  bp::list  properties;
  bp::list  confIds;
  bp::list  distanceConstraints;
  bp::list  positionConstraints;
  bp::list  angleConstraints;
  bp::list  torsionConstraints;
  for (int i = 0; i < numMols; ++i) {
    nvMolKit::MMFFPropertiesNative props;
    props.nonBondedThreshold = nonBondedThreshold;
    properties.append(props);
    confIds.append(-1);
    distanceConstraints.append(bp::list());
    positionConstraints.append(bp::list());
    angleConstraints.append(bp::list());
    torsionConstraints.append(bp::list());
  }
  NativeMMFFBatchedForcefield forcefield(
    molecules, properties, confIds, distanceConstraints, positionConstraints, angleConstraints, torsionConstraints);
  return forcefield.computeEnergy();
}

bp::list computeMMFFGradients(const bp::list& molecules, const double nonBondedThreshold) {
  const int numMols = bp::len(molecules);
  bp::list  properties;
  bp::list  confIds;
  bp::list  distanceConstraints;
  bp::list  positionConstraints;
  bp::list  angleConstraints;
  bp::list  torsionConstraints;
  for (int i = 0; i < numMols; ++i) {
    nvMolKit::MMFFPropertiesNative props;
    props.nonBondedThreshold = nonBondedThreshold;
    properties.append(props);
    confIds.append(-1);
    distanceConstraints.append(bp::list());
    positionConstraints.append(bp::list());
    angleConstraints.append(bp::list());
    torsionConstraints.append(bp::list());
  }
  NativeMMFFBatchedForcefield forcefield(
    molecules, properties, confIds, distanceConstraints, positionConstraints, angleConstraints, torsionConstraints);
  return forcefield.computeGradients();
}

bp::list computeDGEnergies(const bp::list& molecules, RDKit::DGeomHelpers::EmbedParameters params) {
  auto mols = extractMolecules(molecules);
  nvMolKit::DistGeom::BatchedMolecularSystemHost systemHost;
  std::vector<int>                               atomStartsHost = {0};
  std::vector<double>                            positionsHost;

  for (auto* mol : mols) {
    params.useRandomCoords = true;
    nvMolKit::detail::EmbedArgs                 eargs;
    std::vector<std::unique_ptr<RDGeom::Point>> positions;
    std::unique_ptr<ForceFields::ForceField>    referenceFF;
    nvMolKit::DGeomHelpers::setupRDKitFFWithPos(mol, params, referenceFF, eargs, positions);
    auto ffParams = nvMolKit::DistGeom::constructForceFieldContribs(eargs.dim, *eargs.mmat, eargs.chiralCenters);
    nvMolKit::DistGeom::addMoleculeToBatch(ffParams, eargs.posVec, systemHost, eargs.dim, atomStartsHost, positionsHost);
  }

  nvMolKit::DGBatchedForcefield forcefield(systemHost, atomStartsHost, 1.0, 0.1);
  nvMolKit::AsyncDeviceVector<double> positionsDevice;
  nvMolKit::AsyncDeviceVector<double> energyOutsDevice;
  positionsDevice.setFromVector(positionsHost);
  energyOutsDevice.resize(atomStartsHost.size() - 1);
  energyOutsDevice.zero();
  throwIfCudaError(forcefield.computeEnergy(energyOutsDevice.data(), positionsDevice.data()), "DGComputeEnergies");
  return vectorToList(copyDeviceVector(energyOutsDevice));
}

bp::list computeDGGradients(const bp::list& molecules, RDKit::DGeomHelpers::EmbedParameters params) {
  auto mols = extractMolecules(molecules);
  nvMolKit::DistGeom::BatchedMolecularSystemHost systemHost;
  std::vector<int>                               atomStartsHost = {0};
  std::vector<double>                            positionsHost;

  for (auto* mol : mols) {
    params.useRandomCoords = true;
    nvMolKit::detail::EmbedArgs                 eargs;
    std::vector<std::unique_ptr<RDGeom::Point>> positions;
    std::unique_ptr<ForceFields::ForceField>    referenceFF;
    nvMolKit::DGeomHelpers::setupRDKitFFWithPos(mol, params, referenceFF, eargs, positions);
    auto ffParams = nvMolKit::DistGeom::constructForceFieldContribs(eargs.dim, *eargs.mmat, eargs.chiralCenters);
    nvMolKit::DistGeom::addMoleculeToBatch(ffParams, eargs.posVec, systemHost, eargs.dim, atomStartsHost, positionsHost);
  }

  nvMolKit::DGBatchedForcefield forcefield(systemHost, atomStartsHost, 1.0, 0.1);
  nvMolKit::AsyncDeviceVector<double> positionsDevice;
  nvMolKit::AsyncDeviceVector<double> gradDevice;
  positionsDevice.setFromVector(positionsHost);
  gradDevice.resize(positionsHost.size());
  gradDevice.zero();
  throwIfCudaError(forcefield.computeGradients(gradDevice.data(), positionsDevice.data()), "DGComputeGradients");
  return vectorOfVectorsToList(splitGradients(copyDeviceVector(gradDevice), atomStartsHost, 4));
}

bp::list computeETKEnergies(const bp::list& molecules, RDKit::DGeomHelpers::EmbedParameters params) {
  auto mols = extractMolecules(molecules);
  nvMolKit::DistGeom::BatchedMolecularSystem3DHost systemHost;
  std::vector<int>                                 atomStartsHost = {0};
  std::vector<double>                              positionsHost3D;

  for (auto* mol : mols) {
    params.useRandomCoords = true;
    nvMolKit::detail::EmbedArgs                 eargs;
    std::vector<std::unique_ptr<RDGeom::Point>> positions;
    std::unique_ptr<ForceFields::ForceField>    referenceFF;
    nvMolKit::DGeomHelpers::setupRDKitFFWithPos(
      mol, params, referenceFF, eargs, positions, -1, nvMolKit::DGeomHelpers::Dimensionality::DIM_3D);
    std::vector<double> positions3D;
    nvMolKit::confPosToVect(*mol, positions3D);
    auto ffParams =
      nvMolKit::DistGeom::construct3DForceFieldContribs(*eargs.mmat, eargs.etkdgDetails, positions3D, 3, params.useBasicKnowledge);
    nvMolKit::DistGeom::addMoleculeToBatch3D(ffParams, positions3D, systemHost, atomStartsHost, positionsHost3D);
  }

  nvMolKit::ETKBatchedForcefield forcefield(systemHost, atomStartsHost, params.useBasicKnowledge);
  nvMolKit::AsyncDeviceVector<double> positionsDevice;
  nvMolKit::AsyncDeviceVector<double> energyOutsDevice;
  positionsDevice.setFromVector(pad3Dto4D(positionsHost3D));
  energyOutsDevice.resize(atomStartsHost.size() - 1);
  energyOutsDevice.zero();
  throwIfCudaError(forcefield.computeEnergy(energyOutsDevice.data(), positionsDevice.data()), "ETKComputeEnergies");
  return vectorToList(copyDeviceVector(energyOutsDevice));
}

bp::list computeETKGradients(const bp::list& molecules, RDKit::DGeomHelpers::EmbedParameters params) {
  auto mols = extractMolecules(molecules);
  nvMolKit::DistGeom::BatchedMolecularSystem3DHost systemHost;
  std::vector<int>                                 atomStartsHost = {0};
  std::vector<double>                              positionsHost3D;

  for (auto* mol : mols) {
    params.useRandomCoords = true;
    nvMolKit::detail::EmbedArgs                 eargs;
    std::vector<std::unique_ptr<RDGeom::Point>> positions;
    std::unique_ptr<ForceFields::ForceField>    referenceFF;
    nvMolKit::DGeomHelpers::setupRDKitFFWithPos(
      mol, params, referenceFF, eargs, positions, -1, nvMolKit::DGeomHelpers::Dimensionality::DIM_3D);
    std::vector<double> positions3D;
    nvMolKit::confPosToVect(*mol, positions3D);
    auto ffParams =
      nvMolKit::DistGeom::construct3DForceFieldContribs(*eargs.mmat, eargs.etkdgDetails, positions3D, 3, params.useBasicKnowledge);
    nvMolKit::DistGeom::addMoleculeToBatch3D(ffParams, positions3D, systemHost, atomStartsHost, positionsHost3D);
  }

  nvMolKit::ETKBatchedForcefield forcefield(systemHost, atomStartsHost, params.useBasicKnowledge);
  nvMolKit::AsyncDeviceVector<double> positionsDevice;
  nvMolKit::AsyncDeviceVector<double> gradDevice;
  positionsDevice.setFromVector(pad3Dto4D(positionsHost3D));
  gradDevice.resize(positionsDevice.size());
  gradDevice.zero();
  throwIfCudaError(forcefield.computeGradients(gradDevice.data(), positionsDevice.data()), "ETKComputeGradients");
  return vectorOfVectorsToList(strip4DGradientsTo3D(copyDeviceVector(gradDevice), atomStartsHost));
}

}  // namespace

BOOST_PYTHON_MODULE(_batchedForcefield) {
  bp::class_<nvMolKit::MMFFPropertiesNative>("MMFFPropertiesNative")
    .def(bp::init<>())
    .def_readwrite("variant", &nvMolKit::MMFFPropertiesNative::variant)
    .def_readwrite("dielectricConstant", &nvMolKit::MMFFPropertiesNative::dielectricConstant)
    .def_readwrite("dielectricModel", &nvMolKit::MMFFPropertiesNative::dielectricModel)
    .def_readwrite("nonBondedThreshold", &nvMolKit::MMFFPropertiesNative::nonBondedThreshold)
    .def_readwrite("ignoreInterfragInteractions", &nvMolKit::MMFFPropertiesNative::ignoreInterfragInteractions)
    .def_readwrite("bondTerm", &nvMolKit::MMFFPropertiesNative::bondTerm)
    .def_readwrite("angleTerm", &nvMolKit::MMFFPropertiesNative::angleTerm)
    .def_readwrite("stretchBendTerm", &nvMolKit::MMFFPropertiesNative::stretchBendTerm)
    .def_readwrite("oopTerm", &nvMolKit::MMFFPropertiesNative::oopTerm)
    .def_readwrite("torsionTerm", &nvMolKit::MMFFPropertiesNative::torsionTerm)
    .def_readwrite("vdwTerm", &nvMolKit::MMFFPropertiesNative::vdwTerm)
    .def_readwrite("eleTerm", &nvMolKit::MMFFPropertiesNative::eleTerm);

  bp::class_<NativeMMFFBatchedForcefield, boost::noncopyable>("NativeMMFFBatchedForcefield",
                                                              bp::init<const bp::list&,
                                                                       const bp::list&,
                                                                       const bp::list&,
                                                                       const bp::list&,
                                                                       const bp::list&,
                                                                       const bp::list&,
                                                                       const bp::list&>())
    .def("computeEnergy", &NativeMMFFBatchedForcefield::computeEnergy)
    .def("computeGradients", &NativeMMFFBatchedForcefield::computeGradients);

  bp::def("MMFFComputeEnergies",
          &computeMMFFEnergies,
          (bp::arg("molecules"), bp::arg("nonBondedThreshold") = 100.0));
  bp::def("MMFFComputeGradients",
          &computeMMFFGradients,
          (bp::arg("molecules"), bp::arg("nonBondedThreshold") = 100.0));
  bp::def("DGComputeEnergies", &computeDGEnergies, (bp::arg("molecules"), bp::arg("params")));
  bp::def("DGComputeGradients", &computeDGGradients, (bp::arg("molecules"), bp::arg("params")));
  bp::def("ETKComputeEnergies", &computeETKEnergies, (bp::arg("molecules"), bp::arg("params")));
  bp::def("ETKComputeGradients", &computeETKGradients, (bp::arg("molecules"), bp::arg("params")));
}
