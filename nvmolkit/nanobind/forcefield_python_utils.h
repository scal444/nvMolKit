// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_NANOBIND_FORCEFIELD_PYTHON_UTILS_H
#define NVMOLKIT_NANOBIND_FORCEFIELD_PYTHON_UTILS_H

#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>
#include <GraphMol/ROMol.h>
#include <nanobind/nanobind.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "src/forcefields/mmff_properties.h"

namespace nb = nanobind;

namespace nvMolKit::NanobindForcefield {

inline std::vector<RDKit::ROMol*> extractMolecules(const nb::list& molecules) {
  const size_t               moleculeCount = nb::len(molecules);
  std::vector<RDKit::ROMol*> result;
  result.reserve(moleculeCount);
  for (size_t moleculeIndex = 0; moleculeIndex < moleculeCount; ++moleculeIndex) {
    auto* molecule = nb::cast<RDKit::ROMol*>(molecules[moleculeIndex]);
    if (molecule == nullptr) {
      throw std::invalid_argument("Invalid molecule at index " + std::to_string(moleculeIndex));
    }
    result.push_back(molecule);
  }
  return result;
}

inline std::vector<double> extractDoubleList(const nb::list& values, const int expectedSize, const std::string& name) {
  const size_t actualSize = nb::len(values);
  if (actualSize != static_cast<size_t>(expectedSize)) {
    throw std::invalid_argument("Expected " + std::to_string(expectedSize) + " values for " + name + ", got " +
                                std::to_string(actualSize));
  }
  std::vector<double> result;
  result.reserve(actualSize);
  for (size_t valueIndex = 0; valueIndex < actualSize; ++valueIndex) {
    result.push_back(nb::cast<double>(values[valueIndex]));
  }
  return result;
}

inline std::vector<bool> extractBoolList(const nb::list& values, const int expectedSize, const std::string& name) {
  const size_t actualSize = nb::len(values);
  if (actualSize != static_cast<size_t>(expectedSize)) {
    throw std::invalid_argument("Expected " + std::to_string(expectedSize) + " values for " + name + ", got " +
                                std::to_string(actualSize));
  }
  std::vector<bool> result;
  result.reserve(actualSize);
  for (size_t valueIndex = 0; valueIndex < actualSize; ++valueIndex) {
    result.push_back(nb::cast<bool>(values[valueIndex]));
  }
  return result;
}

inline MMFFProperties extractMMFFProperties(const nb::handle& value,
                                            const double      nonBondedThreshold          = 100.0,
                                            const bool        ignoreInterfragInteractions = true) {
  if (value.is_none()) {
    MMFFProperties properties;
    properties.nonBondedThreshold          = nonBondedThreshold;
    properties.ignoreInterfragInteractions = ignoreInterfragInteractions;
    return properties;
  }
  return nb::cast<MMFFProperties>(value);
}

inline std::vector<MMFFProperties> extractMMFFPropertiesList(const nb::list& properties, const int moleculeCount) {
  const size_t                propertyCount = nb::len(properties);
  std::vector<MMFFProperties> result;
  result.reserve(moleculeCount);
  for (int moleculeIndex = 0; moleculeIndex < moleculeCount; ++moleculeIndex) {
    if (static_cast<size_t>(moleculeIndex) < propertyCount) {
      result.push_back(extractMMFFProperties(properties[moleculeIndex]));
    } else {
      result.emplace_back();
    }
  }
  return result;
}

inline MMFFProperties buildMMFFPropertiesFromRDKit(RDKit::MMFF::MMFFMolProperties& rdkitProperties,
                                                   const double                    nonBondedThreshold,
                                                   const bool                      ignoreInterfragInteractions) {
  MMFFProperties properties;
  properties.variant                     = rdkitProperties.getMMFFVariant();
  properties.dielectricConstant          = rdkitProperties.getMMFFDielectricConstant();
  properties.dielectricModel             = static_cast<int>(rdkitProperties.getMMFFDielectricModel());
  properties.nonBondedThreshold          = nonBondedThreshold;
  properties.ignoreInterfragInteractions = ignoreInterfragInteractions;
  properties.bondTerm                    = rdkitProperties.getMMFFBondTerm();
  properties.angleTerm                   = rdkitProperties.getMMFFAngleTerm();
  properties.stretchBendTerm             = rdkitProperties.getMMFFStretchBendTerm();
  properties.oopTerm                     = rdkitProperties.getMMFFOopTerm();
  properties.torsionTerm                 = rdkitProperties.getMMFFTorsionTerm();
  properties.vdwTerm                     = rdkitProperties.getMMFFVdWTerm();
  properties.eleTerm                     = rdkitProperties.getMMFFEleTerm();
  return properties;
}

}  // namespace nvMolKit::NanobindForcefield

#endif  // NVMOLKIT_NANOBIND_FORCEFIELD_PYTHON_UTILS_H
