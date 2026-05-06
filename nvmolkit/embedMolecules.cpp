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

#include <GraphMol/DistGeomHelpers/Embedder.h>

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

#include "boost_python_utils.h"
#include "etkdg.h"
#include "minimizer/bfgs_types.h"

static boost::python::list getGpuIdsPy(nvMolKit::BatchHardwareOptions& opts) {
  return nvMolKit::vectorToList(opts.gpuIds);
}

static void setGpuIds(nvMolKit::BatchHardwareOptions& opts, const boost::python::object& iterable) {
  std::vector<int> converted;
  using namespace boost::python;
  // Prefer fast sequence path
  if (PySequence_Check(iterable.ptr())) {
    Py_ssize_t n = PySequence_Size(iterable.ptr());
    converted.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
      object item(handle<>(borrowed(PySequence_GetItem(iterable.ptr(), i))));
      converted.push_back(extract<int>(item));
    }
  } else {
    // Fallback: try generic iterable
    stl_input_iterator<int> it(iterable), end;
    for (; it != end; ++it) {
      converted.push_back(*it);
    }
  }
  opts.gpuIds.swap(converted);
}

BOOST_PYTHON_MODULE(_embedMolecules) {
  // Expose BatchHardwareOptions struct to Python
  boost::python::class_<nvMolKit::BatchHardwareOptions>("BatchHardwareOptions")
    .def(boost::python::init<>())
    .def_readwrite("preprocessingThreads", &nvMolKit::BatchHardwareOptions::preprocessingThreads)
    .def_readwrite("batchSize", &nvMolKit::BatchHardwareOptions::batchSize)
    .def_readwrite("batchesPerGpu", &nvMolKit::BatchHardwareOptions::batchesPerGpu)
    .add_property("gpuIds", &getGpuIdsPy, &setGpuIds);

  boost::python::enum_<nvMolKit::MinimizerKind>("MinimizerKind")
    .value("BFGS", nvMolKit::MinimizerKind::BFGS)
    .value("FIRE", nvMolKit::MinimizerKind::FIRE);

  boost::python::def(
    "EmbedMolecules",
    +[](const boost::python::list&                  molecules,
        const RDKit::DGeomHelpers::EmbedParameters& params,
        int                                         confsPerMolecule,
        int                                         maxIterations,
        const nvMolKit::BatchHardwareOptions&       hardwareOptions,
        nvMolKit::MinimizerKind                     minimizerKind,
        const boost::python::object&                failuresOut) {
      auto molsVec = nvMolKit::extractMolecules(molecules);

      std::vector<std::vector<int16_t>>* failuresPtr   = nullptr;
      std::vector<std::string>*          stageNamesPtr = nullptr;
      std::vector<std::vector<int16_t>>  failuresStorage;
      std::vector<std::string>           stageNamesStorage;
      if (failuresOut.ptr() != Py_None) {
        failuresPtr   = &failuresStorage;
        stageNamesPtr = &stageNamesStorage;
      }

      nvMolKit::embedMolecules(molsVec,
                               params,
                               confsPerMolecule,
                               maxIterations,
                               false,
                               failuresPtr,
                               hardwareOptions,
                               nvMolKit::BfgsBackend::HYBRID,
                               minimizerKind,
                               stageNamesPtr,
                               nvMolKit::FireBackend::HYBRID);

      if (failuresOut.ptr() != Py_None) {
        boost::python::dict outDict = boost::python::extract<boost::python::dict>(failuresOut);
        boost::python::list nameList;
        for (const auto& name : stageNamesStorage) {
          nameList.append(name);
        }
        boost::python::list perStageList;
        for (const auto& stage : failuresStorage) {
          boost::python::list perConfList;
          for (const auto& count : stage) {
            perConfList.append(static_cast<int>(count));
          }
          perStageList.append(perConfList);
        }
        outDict["stage_names"] = nameList;
        outDict["counts"]      = perStageList;
      }
    },
    (boost::python::arg("molecules"),
     boost::python::arg("params"),
     boost::python::arg("confsPerMolecule") = 1,
     boost::python::arg("maxIterations")    = -1,
     boost::python::arg("hardwareOptions")  = nvMolKit::BatchHardwareOptions(),
     boost::python::arg("minimizerKind")    = nvMolKit::MinimizerKind::BFGS,
     boost::python::arg("failuresOut")      = boost::python::object()),
    "Embed multiple molecules with multiple conformers using ETKDG.\n"
    "\n"
    "Args:\n"
    "    molecules: List of RDKit molecules to embed\n"
    "    params: RDKit EmbedParameters object with embedding settings\n"
    "    confsPerMolecule: Number of conformers to generate per molecule (default: 1)\n"
    "    maxIterations: Maximum iterations, -1 for auto (default: -1)\n"
    "    hardwareOptions: BatchHardwareOptions object with hardware settings (default: default options)\n"
    "    minimizerKind: Selects the inner minimizer (BFGS or FIRE). Default BFGS preserves\n"
    "                   historical behavior. The BFGS and FIRE backends both run with HYBRID\n"
    "                   kernel selection (per-mol for small molecules, batched for large).\n"
    "    failuresOut: Optional dict; if provided, populated with keys 'stage_names' (list of\n"
    "                 stage names in pipeline order) and 'counts' (list of per-stage lists of\n"
    "                 per-conformer failure counts, indexed mol_id * confsPerMolecule).\n"
    "\n"
    "Returns:\n"
    "    None (molecules are modified in-place with generated conformers)");
}
