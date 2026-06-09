// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <boost/python/stl_iterator.hpp>

#include "src/hardware_options.h"
#include "src/minimizer/fire_minimizer.h"

namespace {

boost::python::list getGpuIdsPy(nvMolKit::BatchHardwareOptions& opts) {
  boost::python::list ids;
  for (const auto id : opts.gpuIds) {
    ids.append(id);
  }
  return ids;
}

void setGpuIds(nvMolKit::BatchHardwareOptions& opts, const boost::python::object& iterable) {
  std::vector<int> converted;
  using namespace boost::python;

  if (PySequence_Check(iterable.ptr())) {
    const Py_ssize_t n = PySequence_Size(iterable.ptr());
    converted.reserve(static_cast<std::size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
      object item(handle<>(borrowed(PySequence_GetItem(iterable.ptr(), i))));
      converted.push_back(extract<int>(item));
    }
  } else {
    stl_input_iterator<int> it(iterable), end;
    for (; it != end; ++it) {
      converted.push_back(*it);
    }
  }

  opts.gpuIds.swap(converted);
}

}  // namespace

BOOST_PYTHON_MODULE(_types) {
  boost::python::class_<nvMolKit::BatchHardwareOptions>("BatchHardwareOptions")
    .def(boost::python::init<>())
    .def_readwrite("preprocessingThreads", &nvMolKit::BatchHardwareOptions::preprocessingThreads)
    .def_readwrite("batchSize", &nvMolKit::BatchHardwareOptions::batchSize)
    .def_readwrite("batchesPerGpu", &nvMolKit::BatchHardwareOptions::batchesPerGpu)
    .add_property("gpuIds", &getGpuIdsPy, &setGpuIds);

  boost::python::class_<nvMolKit::FireOptions>("FireOptions")
    .def(boost::python::init<>())
    .def_readwrite("dtInit", &nvMolKit::FireOptions::dtInit)
    .def_readwrite("dtMinFactor", &nvMolKit::FireOptions::dtMinFactor)
    .def_readwrite("dtMaxFactor", &nvMolKit::FireOptions::dtMaxFactor)
    .def_readwrite("dMax", &nvMolKit::FireOptions::dMax)
    .def_readwrite("timeStepIncrement", &nvMolKit::FireOptions::timeStepIncrement)
    .def_readwrite("timeStepDecrement", &nvMolKit::FireOptions::timeStepDecrement)
    .def_readwrite("nMinForIncrease", &nvMolKit::FireOptions::nMinForIncrease)
    .def_readwrite("alphaInit", &nvMolKit::FireOptions::alphaInit)
    .def_readwrite("alphaDecrement", &nvMolKit::FireOptions::alphaDecrement)
    .def_readwrite("useMass", &nvMolKit::FireOptions::useMass)
    .def_readwrite("gradTol", &nvMolKit::FireOptions::gradTol)
    .def_readwrite("takeHalfStepBack", &nvMolKit::FireOptions::takeHalfStepBack)
    .def_readwrite("abcCorrection", &nvMolKit::FireOptions::abcCorrection)
    .def_readwrite("stuckDetectionEnabled", &nvMolKit::FireOptions::stuckDetectionEnabled)
    .def_readwrite("stuckEnergyRelTol", &nvMolKit::FireOptions::stuckEnergyRelTol)
    .def_readwrite("stuckStreakLength", &nvMolKit::FireOptions::stuckStreakLength)
    .def_readwrite("stuckEvalEveryNPolls", &nvMolKit::FireOptions::stuckEvalEveryNPolls);
}
