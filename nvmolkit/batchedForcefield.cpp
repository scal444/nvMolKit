#include <ForceField/ForceField.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/Builder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/ROMol.h>

#include <boost/python.hpp>

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

namespace bp = boost::python;

namespace {

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

std::vector<std::vector<double>> splitGradients(const std::vector<double>& flatGrad, const std::vector<int>& atomStarts, const int dim) {
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

bp::list computeMMFFEnergies(const bp::list& molecules, const double nonBondedThreshold) {
  auto mols = extractMolecules(molecules);
  nvMolKit::MMFF::BatchedMolecularSystemHost systemHost;
  for (auto* mol : mols) {
    std::vector<double> positions;
    nvMolKit::confPosToVect(*mol, positions);
    auto ffParams = nvMolKit::MMFF::constructForcefieldContribs(*mol, nonBondedThreshold);
    nvMolKit::MMFF::addMoleculeToBatch(ffParams, positions, systemHost);
  }

  nvMolKit::MMFFBatchedForcefield forcefield(systemHost);
  nvMolKit::AsyncDeviceVector<double> positionsDevice;
  nvMolKit::AsyncDeviceVector<double> energyOutsDevice;
  positionsDevice.setFromVector(systemHost.positions);
  energyOutsDevice.resize(systemHost.indices.atomStarts.size() - 1);
  energyOutsDevice.zero();
  throwIfCudaError(forcefield.computeEnergy(energyOutsDevice.data(), positionsDevice.data()), "MMFFComputeEnergies");
  return vectorToList(copyDeviceVector(energyOutsDevice));
}

bp::list computeMMFFGradients(const bp::list& molecules, const double nonBondedThreshold) {
  auto mols = extractMolecules(molecules);
  nvMolKit::MMFF::BatchedMolecularSystemHost systemHost;
  for (auto* mol : mols) {
    std::vector<double> positions;
    nvMolKit::confPosToVect(*mol, positions);
    auto ffParams = nvMolKit::MMFF::constructForcefieldContribs(*mol, nonBondedThreshold);
    nvMolKit::MMFF::addMoleculeToBatch(ffParams, positions, systemHost);
  }

  nvMolKit::MMFFBatchedForcefield forcefield(systemHost);
  nvMolKit::AsyncDeviceVector<double> positionsDevice;
  nvMolKit::AsyncDeviceVector<double> gradDevice;
  positionsDevice.setFromVector(systemHost.positions);
  gradDevice.resize(systemHost.positions.size());
  gradDevice.zero();
  throwIfCudaError(forcefield.computeGradients(gradDevice.data(), positionsDevice.data()), "MMFFComputeGradients");
  return vectorOfVectorsToList(splitGradients(copyDeviceVector(gradDevice), systemHost.indices.atomStarts, 3));
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
