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

#include "src/minimizer/fire_options.h"
#include "src/precision_options.h"

namespace {

std::string getPrecisionMode(const nvMolKit::PrecisionOptions& options) {
  return nvMolKit::precisionModeName(options.mode);
}

void setPrecisionMode(nvMolKit::PrecisionOptions& options, const std::string& mode) {
  options.mode = nvMolKit::parsePrecisionMode(mode);
}

#define DTYPE_PROPERTY(Name, Member)                                        \
  std::string get##Name(const nvMolKit::PrecisionOptions& o) {              \
    return nvMolKit::precisionDTypeName(o.Member);                          \
  }                                                                         \
  void set##Name(nvMolKit::PrecisionOptions& o, const std::string& value) { \
    o.Member = nvMolKit::parsePrecisionDType(value);                        \
  }
DTYPE_PROPERTY(ForcefieldParameterStorage, forcefieldParameterStorage)
DTYPE_PROPERTY(CoordinateStorage, forcefieldCoordinateStorage)
DTYPE_PROPERTY(GradientStorage, forcefieldGradientStorage)
DTYPE_PROPERTY(HessianStorage, hessianStorage)
DTYPE_PROPERTY(MinimizerStateStorage, minimizerStateStorage)
DTYPE_PROPERTY(ForcefieldCompute, forcefieldCompute)
DTYPE_PROPERTY(MinimizerCompute, minimizerCompute)
DTYPE_PROPERTY(ReductionCompute, reductionCompute)
#undef DTYPE_PROPERTY

std::string getFloatMath(const nvMolKit::PrecisionOptions& o) {
  return nvMolKit::floatMathModeName(o.floatMath);
}
void setFloatMath(nvMolKit::PrecisionOptions& o, const std::string& value) {
  o.floatMath = nvMolKit::parseFloatMathMode(value);
}

}  // namespace

BOOST_PYTHON_MODULE(_types) {
  boost::python::class_<nvMolKit::PrecisionOptions>("NativePrecisionOptions")
    .def(boost::python::init<>())
    .add_property("mode", &getPrecisionMode, &setPrecisionMode)
    .add_property("forcefieldParameterStorage", &getForcefieldParameterStorage, &setForcefieldParameterStorage)
    .add_property("forcefieldCoordinateStorage", &getCoordinateStorage, &setCoordinateStorage)
    .add_property("forcefieldGradientStorage", &getGradientStorage, &setGradientStorage)
    .add_property("hessianStorage", &getHessianStorage, &setHessianStorage)
    .add_property("minimizerStateStorage", &getMinimizerStateStorage, &setMinimizerStateStorage)
    .add_property("forcefieldCompute", &getForcefieldCompute, &setForcefieldCompute)
    .add_property("minimizerCompute", &getMinimizerCompute, &setMinimizerCompute)
    .add_property("reductionCompute", &getReductionCompute, &setReductionCompute)
    .add_property("floatMath", &getFloatMath, &setFloatMath);

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
    .def_readwrite("gradTol", &nvMolKit::FireOptions::gradTol, "Convergence threshold on sqrt(sum(grad^2)) per system.")
    .def_readwrite("takeHalfStepBack", &nvMolKit::FireOptions::takeHalfStepBack)
    .def_readwrite("abcCorrection", &nvMolKit::FireOptions::abcCorrection)
    .def_readwrite("stuckDetectionEnabled", &nvMolKit::FireOptions::stuckDetectionEnabled)
    .def_readwrite("stuckEnergyRelTol", &nvMolKit::FireOptions::stuckEnergyRelTol)
    .def_readwrite("stuckStreakLength", &nvMolKit::FireOptions::stuckStreakLength)
    .def_readwrite("stuckEvalEveryNPolls", &nvMolKit::FireOptions::stuckEvalEveryNPolls);
}
