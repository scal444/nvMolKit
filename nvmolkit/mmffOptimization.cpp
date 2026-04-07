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

#include <GraphMol/ROMol.h>

#include <boost/python.hpp>

#include "bfgs_mmff.h"
#include "mmff_properties.h"

namespace bp = boost::python;

template <typename T> bp::list vectorToList(const std::vector<T>& vec) {
  bp::list list;
  for (const auto& value : vec) {
    list.append(value);
  }
  return list;
}

template <typename T> bp::list vectorOfVectorsToList(const std::vector<std::vector<T>>& vecOfVecs) {
  bp::list outerList;
  for (const auto& innerVec : vecOfVecs) {
    outerList.append(vectorToList(innerVec));
  }
  return outerList;
}

static std::vector<RDKit::ROMol*> extractMolecules(const bp::list& molecules) {
  const int                  n = bp::len(molecules);
  std::vector<RDKit::ROMol*> mols;
  mols.reserve(n);
  for (int i = 0; i < n; ++i) {
    auto* mol = bp::extract<RDKit::ROMol*>(bp::object(molecules[i]))();
    if (mol == nullptr) {
      throw std::invalid_argument("Invalid molecule at index " + std::to_string(i));
    }
    mols.push_back(mol);
  }
  return mols;
}

static nvMolKit::MMFFProperties extractInternalMMFFProperties(const bp::object& obj) {
  if (obj.is_none()) {
    return {};
  }
  return bp::extract<nvMolKit::MMFFProperties>(obj);
}

static std::vector<nvMolKit::MMFFProperties> extractMMFFPropertiesList(const bp::list& propertiesList, int numMols) {
  const int                             n = bp::len(propertiesList);
  std::vector<nvMolKit::MMFFProperties> props;
  props.reserve(numMols);
  for (int i = 0; i < numMols; ++i) {
    if (i < n) {
      props.push_back(extractInternalMMFFProperties(bp::object(propertiesList[i])));
    } else {
      props.emplace_back();
    }
  }
  return props;
}

BOOST_PYTHON_MODULE(_mmffOptimization) {
  bp::class_<nvMolKit::MMFFProperties>("MMFFProperties")
    .def_readwrite("variant", &nvMolKit::MMFFProperties::variant)
    .def_readwrite("dielectricConstant", &nvMolKit::MMFFProperties::dielectricConstant)
    .def_readwrite("dielectricModel", &nvMolKit::MMFFProperties::dielectricModel)
    .def_readwrite("nonBondedThreshold", &nvMolKit::MMFFProperties::nonBondedThreshold)
    .def_readwrite("ignoreInterfragInteractions", &nvMolKit::MMFFProperties::ignoreInterfragInteractions)
    .def_readwrite("bondTerm", &nvMolKit::MMFFProperties::bondTerm)
    .def_readwrite("angleTerm", &nvMolKit::MMFFProperties::angleTerm)
    .def_readwrite("stretchBendTerm", &nvMolKit::MMFFProperties::stretchBendTerm)
    .def_readwrite("oopTerm", &nvMolKit::MMFFProperties::oopTerm)
    .def_readwrite("torsionTerm", &nvMolKit::MMFFProperties::torsionTerm)
    .def_readwrite("vdwTerm", &nvMolKit::MMFFProperties::vdwTerm)
    .def_readwrite("eleTerm", &nvMolKit::MMFFProperties::eleTerm);

  bp::def(
    "MMFFOptimizeMoleculesConfs",
    +[](const bp::list&                       molecules,
        int                                   maxIters,
        const nvMolKit::MMFFProperties&       properties,
        const nvMolKit::BatchHardwareOptions& hardwareOptions) -> bp::list {
      auto molsVec = extractMolecules(molecules);
      auto result  = nvMolKit::MMFF::MMFFOptimizeMoleculesConfsBfgs(molsVec, maxIters, properties, hardwareOptions);
      return vectorOfVectorsToList(result);
    },
    (bp::arg("molecules"),
     bp::arg("maxIters")        = 200,
     bp::arg("properties")      = nvMolKit::MMFFProperties(),
     bp::arg("hardwareOptions") = nvMolKit::BatchHardwareOptions()));

  bp::def(
    "MMFFOptimizeMoleculesConfsPerMol",
    +[](const bp::list&                       molecules,
        int                                   maxIters,
        const bp::list&                       propertiesList,
        const nvMolKit::BatchHardwareOptions& hardwareOptions) -> bp::list {
      auto molsVec = extractMolecules(molecules);
      auto props   = extractMMFFPropertiesList(propertiesList, static_cast<int>(molsVec.size()));
      auto result  = nvMolKit::MMFF::MMFFOptimizeMoleculesConfsBfgs(molsVec, maxIters, props, hardwareOptions);
      return vectorOfVectorsToList(result);
    },
    (bp::arg("molecules"),
     bp::arg("maxIters")        = 200,
     bp::arg("propertiesList"),
     bp::arg("hardwareOptions") = nvMolKit::BatchHardwareOptions()));
}
