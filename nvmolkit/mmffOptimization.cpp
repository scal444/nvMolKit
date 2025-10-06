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

BOOST_PYTHON_MODULE(_mmffOptimization) {
  boost::python::def(
    "MMFFOptimizeMoleculesConfs",
    +[](const boost::python::list&            molecules,
        int                                   maxIters,
        double                                nonBondedThreshold,
        const nvMolKit::BatchHardwareOptions& hardwareOptions,
        const std::string&                    optimizerBackend,
        const boost::python::dict&            optimizerOptionsDict) -> boost::python::list {
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

      nvMolKit::MMFF::OptimizerOptions optOptions;
      if (optimizerBackend.empty() || optimizerBackend == "BFGS" || optimizerBackend == "bfgs") {
        optOptions.backend = nvMolKit::MMFF::OptimizerOptions::Backend::BFGS;
        if (boost::python::len(optimizerOptionsDict) != 0) {
          throw std::invalid_argument("BFGS backend does not accept optimizer options");
        }
      } else if (optimizerBackend == "FIRE" || optimizerBackend == "fire") {
        optOptions.backend = nvMolKit::MMFF::OptimizerOptions::Backend::FIRE;
        const auto keys    = optimizerOptionsDict.keys();
        for (int i = 0; i < boost::python::len(keys); ++i) {
          const std::string key = boost::python::extract<std::string>(keys[i]);
          if (key == "use_masses") {
            optOptions.fireOptions.useMass = boost::python::extract<bool>(optimizerOptionsDict[key]);
          } else if (key == "dt_init") {
            optOptions.fireOptions.dtInit = boost::python::extract<double>(optimizerOptionsDict[key]);
          } else if (key == "dt_min") {
            optOptions.fireOptions.dtMin = boost::python::extract<double>(optimizerOptionsDict[key]);
          } else if (key == "dt_max") {
            optOptions.fireOptions.dtMax = boost::python::extract<double>(optimizerOptionsDict[key]);
          } else if (key == "max_step") {
            optOptions.fireOptions.maxStep = boost::python::extract<double>(optimizerOptionsDict[key]);
          } else if (key == "time_step_increment") {
            optOptions.fireOptions.timeStepIncrement = boost::python::extract<double>(optimizerOptionsDict[key]);
          } else if (key == "time_step_decrement") {
            optOptions.fireOptions.timeStepDecrement = boost::python::extract<double>(optimizerOptionsDict[key]);
          } else if (key == "n_min_for_increase") {
            optOptions.fireOptions.nMinForIncrease = boost::python::extract<int>(optimizerOptionsDict[key]);
          } else if (key == "alpha_init") {
            optOptions.fireOptions.alphaInit = boost::python::extract<double>(optimizerOptionsDict[key]);
          } else if (key == "alpha_decrement") {
            optOptions.fireOptions.alphaDecrement = boost::python::extract<double>(optimizerOptionsDict[key]);
          } else {
            throw std::invalid_argument("Unknown FIRE optimizer option: " + key);
          }
        }
      } else {
        throw std::invalid_argument("Unsupported optimizer backend: " + optimizerBackend);
      }

      // Call the C++ function
      auto result = nvMolKit::MMFF::MMFFOptimizeMoleculesConfsBfgs(molsVec,
                                                                   maxIters,
                                                                   nonBondedThreshold,
                                                                   hardwareOptions,
                                                                   optOptions);

      // Convert result back to Python list of lists
      return vectorOfVectorsToList(result);
    },
    (boost::python::arg("molecules"),
     boost::python::arg("maxIters")           = 200,
     boost::python::arg("nonBondedThreshold") = 100.0,
     boost::python::arg("hardwareOptions")    = nvMolKit::BatchHardwareOptions(),
     boost::python::arg("optimizerBackend")   = std::string("BFGS"),
     boost::python::arg("optimizerOptions")   = boost::python::dict()),
    "Optimize conformers for multiple molecules using MMFF force field.\n"
    "\n"
    "Args:\n"
    "    molecules: List of RDKit molecules to optimize\n"
    "    maxIters: Maximum number of optimization iterations (default: 200)\n"
    "    nonBondedThreshold: Radius threshold for non-bonded interactions (default: 100.0)\n"
    "    hardwareOptions: BatchHardwareOptions object with hardware settings (default: default options)\n"
    "\n"
    "Returns:\n"
    "    List of lists of energies, where each inner list contains energies for conformers of one molecule");
}
