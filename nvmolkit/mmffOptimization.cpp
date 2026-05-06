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

#include <boost/python.hpp>
#include <stdexcept>
#include <string>

#include "bfgs_mmff.h"
#include "boost_python_utils.h"
#include "minimizer/bfgs_types.h"
#include "minimizer/fire_minimizer.h"
#include "mmff_python_utils.h"

namespace {

nvMolKit::BfgsBackend parseBfgsBackend(const std::string& name) {
  if (name == "BATCHED") {
    return nvMolKit::BfgsBackend::BATCHED;
  }
  if (name == "PER_MOL" || name == "PER_MOLECULE") {
    return nvMolKit::BfgsBackend::PER_MOLECULE;
  }
  if (name == "HYBRID") {
    return nvMolKit::BfgsBackend::HYBRID;
  }
  throw std::invalid_argument("Unknown BFGS backend '" + name + "'. Expected 'BATCHED', 'PER_MOL', or 'HYBRID'.");
}

nvMolKit::FireBackend parseFireBackend(const std::string& name) {
  if (name == "BATCHED") {
    return nvMolKit::FireBackend::BATCHED;
  }
  if (name == "PER_MOL" || name == "PER_MOLECULE") {
    return nvMolKit::FireBackend::PER_MOLECULE;
  }
  if (name == "HYBRID") {
    return nvMolKit::FireBackend::HYBRID;
  }
  throw std::invalid_argument("Unknown FIRE backend '" + name + "'. Expected 'BATCHED', 'PER_MOL', or 'HYBRID'.");
}

}  // namespace

BOOST_PYTHON_MODULE(_mmffOptimization) {
  boost::python::def(
    "MMFFOptimizeMoleculesConfs",
    +[](const boost::python::list&            molecules,
        int                                   maxIters,
        const boost::python::list&            propertiesList,
        const nvMolKit::BatchHardwareOptions& hardwareOptions,
        const std::string&                    backend) -> boost::python::list {
      auto molsVec = nvMolKit::extractMolecules(molecules);

      const auto properties = nvMolKit::extractMMFFPropertiesList(propertiesList, static_cast<int>(molsVec.size()));
      const auto result     = nvMolKit::MMFF::MMFFOptimizeMoleculesConfsBfgs(molsVec,
                                                                         maxIters,
                                                                         properties,
                                                                         hardwareOptions,
                                                                         parseBfgsBackend(backend));

      return nvMolKit::vectorOfVectorsToList(result);
    },
    (boost::python::arg("molecules"),
     boost::python::arg("maxIters")        = 200,
     boost::python::arg("properties")      = boost::python::list(),
     boost::python::arg("hardwareOptions") = nvMolKit::BatchHardwareOptions(),
     boost::python::arg("backend")         = std::string("HYBRID")),
    "Optimize conformers for multiple molecules using MMFF force field with BFGS.\n"
    "\n"
    "Args:\n"
    "    molecules: List of RDKit molecules to optimize\n"
    "    maxIters: Maximum number of optimization iterations (default: 200)\n"
    "    properties: MMFFProperties-compatible object with forcefield settings\n"
    "    hardwareOptions: BatchHardwareOptions object with hardware settings (default: default options)\n"
    "    backend: BFGS kernel backend: 'BATCHED', 'PER_MOL', or 'HYBRID' (default: 'HYBRID')\n"
    "\n"
    "Returns:\n"
    "    List of lists of energies, where each inner list contains energies for conformers of one molecule");

  // TODO(remove-before-pr): the fireDebugOutput parameter exists for the
  // benchmark scripts under benchmarks/ that produce per-step trajectories
  // (alpha, dt, power, energy) for diagnostic plots. It will be removed when
  // the FIRE work is opened as a PR; the benchmark scripts will either be
  // deleted or rewritten to not depend on it at that time.
  boost::python::def(
    "MMFFOptimizeMoleculesConfsFire",
    +[](const boost::python::list&            molecules,
        int                                   maxIters,
        const nvMolKit::FireOptions&          fireOptions,
        const boost::python::list&            propertiesList,
        const nvMolKit::BatchHardwareOptions& hardwareOptions,
        const boost::python::object&          fireDebugOutput,
        const std::string&                    backend) -> boost::python::list {
      auto       molsVec    = nvMolKit::extractMolecules(molecules);
      const auto properties = nvMolKit::extractMMFFPropertiesList(propertiesList, static_cast<int>(molsVec.size()));

      std::vector<std::vector<nvMolKit::FireDebugOutput>>  debugStorage;
      std::vector<std::vector<nvMolKit::FireDebugOutput>>* debugPtr = nullptr;
      if (fireDebugOutput.ptr() != Py_None) {
        if (!PyList_Check(fireDebugOutput.ptr())) {
          throw std::invalid_argument("fireDebugOutput must be a list when provided");
        }
        debugPtr = &debugStorage;
      }

      const auto result = nvMolKit::MMFF::MMFFOptimizeMoleculesConfsFire(molsVec,
                                                                         maxIters,
                                                                         fireOptions,
                                                                         properties,
                                                                         hardwareOptions,
                                                                         debugPtr,
                                                                         parseFireBackend(backend));

      if (debugPtr != nullptr) {
        boost::python::list outer;
        for (const auto& molEntries : debugStorage) {
          boost::python::list inner;
          for (const auto& entry : molEntries) {
            boost::python::dict dict;
            dict["alphas"]   = nvMolKit::vectorToList(entry.alphas);
            dict["dt"]       = nvMolKit::vectorToList(entry.dt);
            dict["powers"]   = nvMolKit::vectorToList(entry.powers);
            dict["energies"] = nvMolKit::vectorToList(entry.energies);
            inner.append(dict);
          }
          outer.append(inner);
        }
        fireDebugOutput.attr("clear")();
        fireDebugOutput.attr("extend")(outer);
      }

      return nvMolKit::vectorOfVectorsToList(result);
    },
    (boost::python::arg("molecules"),
     boost::python::arg("maxIters")        = 200,
     boost::python::arg("fireOptions")     = nvMolKit::FireOptions(),
     boost::python::arg("properties")      = boost::python::list(),
     boost::python::arg("hardwareOptions") = nvMolKit::BatchHardwareOptions(),
     boost::python::arg("fireDebugOutput") = boost::python::object(),
     boost::python::arg("backend")         = std::string("HYBRID")),
    "Optimize conformers using the FIRE 2.0 minimizer.\n"
    "\n"
    "Args:\n"
    "    molecules: List of RDKit molecules to optimize\n"
    "    maxIters: Maximum number of FIRE iterations (default: 200)\n"
    "    fireOptions: FireOptions instance controlling the algorithm parameters\n"
    "    properties: MMFFProperties-compatible object with forcefield settings\n"
    "    hardwareOptions: BatchHardwareOptions object with hardware settings\n"
    "    fireDebugOutput: Optional empty list that will be populated with\n"
    "        per-iteration FIRE state for diagnostic benchmarking. EXPERIMENTAL\n"
    "        and slated for removal before PR. Only valid with backend='BATCHED'.\n"
    "    backend: FIRE kernel backend: 'BATCHED', 'PER_MOL', or 'HYBRID' (default: 'HYBRID')\n"
    "\n"
    "Returns:\n"
    "    List of lists of energies, where each inner list contains energies for conformers of one molecule");
}
