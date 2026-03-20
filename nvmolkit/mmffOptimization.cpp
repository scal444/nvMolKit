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

#include "mmff_properties.h"
#include "bfgs_mmff.h"

template <typename T> boost::python::list vectorToList(const std::vector<T>& vec) {
  boost::python::list list;
  for (const auto& value : vec) {
    list.append(value);
  }
  return list;
}

template <typename T> boost::python::list vectorOfVectorsToList(const std::vector<std::vector<T>>& vecOfVecs) {
  boost::python::list outerList;
  for (const auto& innerVec : vecOfVecs) {
    outerList.append(vectorToList(innerVec));
  }
  return outerList;
}

nvMolKit::MMFFPropertiesNative extractMMFFProperties(const boost::python::object& obj) {
  nvMolKit::MMFFPropertiesNative props;
  if (obj.is_none()) {
    return props;
  }
  props.variant = boost::python::extract<std::string>(obj.attr("variant"));
  props.dielectricConstant = boost::python::extract<double>(obj.attr("dielectricConstant"));
  props.dielectricModel = boost::python::extract<int>(obj.attr("dielectricModel"));
  props.nonBondedThreshold = boost::python::extract<double>(obj.attr("nonBondedThreshold"));
  props.ignoreInterfragInteractions = boost::python::extract<bool>(obj.attr("ignoreInterfragInteractions"));
  props.bondTerm = boost::python::extract<bool>(obj.attr("bondTerm"));
  props.angleTerm = boost::python::extract<bool>(obj.attr("angleTerm"));
  props.stretchBendTerm = boost::python::extract<bool>(obj.attr("stretchBendTerm"));
  props.oopTerm = boost::python::extract<bool>(obj.attr("oopTerm"));
  props.torsionTerm = boost::python::extract<bool>(obj.attr("torsionTerm"));
  props.vdwTerm = boost::python::extract<bool>(obj.attr("vdwTerm"));
  props.eleTerm = boost::python::extract<bool>(obj.attr("eleTerm"));
  return props;
}

BOOST_PYTHON_MODULE(_mmffOptimization) {
  boost::python::def(
    "MMFFOptimizeMoleculesConfs",
    +[](const boost::python::list&            molecules,
        int                                   maxIters,
        const boost::python::object&          propertiesObj,
        const nvMolKit::BatchHardwareOptions& hardwareOptions) -> boost::python::list {
      // Convert Python list to std::vector<RDKit::ROMol*>
      std::vector<RDKit::ROMol*> molsVec;
      molsVec.reserve(len(molecules));

      for (int i = 0; i < len(molecules); i++) {
        RDKit::ROMol* mol = boost::python::extract<RDKit::ROMol*>(boost::python::object(molecules[i]));
        if (mol == nullptr) {
          throw std::invalid_argument("Invalid molecule at index " + std::to_string(i));
        }
        molsVec.push_back(mol);
      }

      const auto properties = extractMMFFProperties(propertiesObj);
      // Call the C++ function (uses HYBRID backend by default, which auto-selects based on max atoms)
      auto result = nvMolKit::MMFF::MMFFOptimizeMoleculesConfsBfgs(molsVec, maxIters, properties, hardwareOptions);

      // Convert result back to Python list of lists
      return vectorOfVectorsToList(result);
    },
    (boost::python::arg("molecules"),
     boost::python::arg("maxIters")           = 200,
     boost::python::arg("properties")         = boost::python::object(),
     boost::python::arg("hardwareOptions")    = nvMolKit::BatchHardwareOptions()),
    "Optimize conformers for multiple molecules using MMFF force field.\n"
    "\n"
    "Args:\n"
    "    molecules: List of RDKit molecules to optimize\n"
    "    maxIters: Maximum number of optimization iterations (default: 200)\n"
    "    properties: MMFFPropertiesNative-compatible object with forcefield settings\n"
    "    hardwareOptions: BatchHardwareOptions object with hardware settings (default: default options)\n"
    "\n"
    "Returns:\n"
    "    List of lists of energies, where each inner list contains energies for conformers of one molecule");
}
