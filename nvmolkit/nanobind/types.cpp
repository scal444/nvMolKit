// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>

#include "src/minimizer/fire_options.h"

namespace nb = nanobind;

NB_MODULE(_types, module) {
  nb::class_<nvMolKit::FireOptions>(module, "FireOptions")
    .def(nb::init<>())
    .def_rw("dtInit", &nvMolKit::FireOptions::dtInit)
    .def_rw("dtMinFactor", &nvMolKit::FireOptions::dtMinFactor)
    .def_rw("dtMaxFactor", &nvMolKit::FireOptions::dtMaxFactor)
    .def_rw("dMax", &nvMolKit::FireOptions::dMax)
    .def_rw("timeStepIncrement", &nvMolKit::FireOptions::timeStepIncrement)
    .def_rw("timeStepDecrement", &nvMolKit::FireOptions::timeStepDecrement)
    .def_rw("nMinForIncrease", &nvMolKit::FireOptions::nMinForIncrease)
    .def_rw("alphaInit", &nvMolKit::FireOptions::alphaInit)
    .def_rw("alphaDecrement", &nvMolKit::FireOptions::alphaDecrement)
    .def_rw("useMass", &nvMolKit::FireOptions::useMass)
    .def_rw("gradTol", &nvMolKit::FireOptions::gradTol, "Convergence threshold on sqrt(sum(grad^2)) per system.")
    .def_rw("takeHalfStepBack", &nvMolKit::FireOptions::takeHalfStepBack)
    .def_rw("abcCorrection", &nvMolKit::FireOptions::abcCorrection)
    .def_rw("stuckDetectionEnabled", &nvMolKit::FireOptions::stuckDetectionEnabled)
    .def_rw("stuckEnergyRelTol", &nvMolKit::FireOptions::stuckEnergyRelTol)
    .def_rw("stuckStreakLength", &nvMolKit::FireOptions::stuckStreakLength)
    .def_rw("stuckEvalEveryNPolls", &nvMolKit::FireOptions::stuckEvalEveryNPolls);
}
