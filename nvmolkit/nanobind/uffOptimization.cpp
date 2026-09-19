// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>

#include "nvmolkit/nanobind/device_result_python.h"
#include "nvmolkit/nanobind/forcefield_python_utils.h"
#include "src/minimizer/fire_minimizer.h"
#include "src/minimizer/uff_minimize.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

enum class MinimizerKind {
  BFGS,
  FIRE
};

MinimizerKind parseMinimizerKind(const std::string& name) {
  if (name == "BFGS" || name == "bfgs") {
    return MinimizerKind::BFGS;
  }
  if (name == "FIRE" || name == "fire") {
    return MinimizerKind::FIRE;
  }
  throw std::invalid_argument("Unknown minimizerKind '" + name + "'. Expected 'BFGS' or 'FIRE'.");
}

}  // namespace

NB_MODULE(_uffOptimization, module) {
  nb::module_::import_("nvmolkit._embedMolecules");
  nb::module_::import_("nvmolkit._types");

  module.def(
    "UFFOptimizeMoleculesConfs",
    [](const nb::list&                       molecules,
       const int                             maxIters,
       const nb::list&                       vdwThresholds,
       const nb::list&                       ignoreInterfragInteractions,
       const nvMolKit::BatchHardwareOptions& hardwareOptions,
       const std::string&                    minimizerKind,
       const nvMolKit::FireOptions&          fireOptions) {
      auto       moleculeVector = nvMolKit::NanobindForcefield::extractMolecules(molecules);
      const int  moleculeCount  = static_cast<int>(moleculeVector.size());
      const auto thresholdVector =
        nvMolKit::NanobindForcefield::extractDoubleList(vdwThresholds, moleculeCount, "vdwThreshold");
      const auto ignoreVector = nvMolKit::NanobindForcefield::extractBoolList(ignoreInterfragInteractions,
                                                                              moleculeCount,
                                                                              "ignoreInterfragInteractions");
      return parseMinimizerKind(minimizerKind) == MinimizerKind::FIRE ?
               nvMolKit::UFF::UFFOptimizeMoleculesConfsFire(moleculeVector,
                                                            maxIters,
                                                            fireOptions,
                                                            thresholdVector,
                                                            ignoreVector,
                                                            hardwareOptions) :
               nvMolKit::UFF::UFFOptimizeMoleculesConfsBfgs(moleculeVector,
                                                            maxIters,
                                                            thresholdVector,
                                                            ignoreVector,
                                                            hardwareOptions);
    },
    "molecules"_a,
    "maxIters"_a,
    "vdwThresholds"_a,
    "ignoreInterfragInteractions"_a,
    "hardwareOptions"_a = nvMolKit::BatchHardwareOptions(),
    "minimizerKind"_a   = std::string("BFGS"),
    "fireOptions"_a     = nvMolKit::FireOptions(),
    "Optimize conformers for multiple molecules using UFF force field.");

  module.def(
    "UFFOptimizeMoleculesConfsDevice",
    [](const nb::list&                       molecules,
       const int                             maxIters,
       const nb::list&                       vdwThresholds,
       const nb::list&                       ignoreInterfragInteractions,
       const nvMolKit::BatchHardwareOptions& hardwareOptions,
       const int                             targetGpu,
       const std::string&                    minimizerKind,
       const nvMolKit::FireOptions&          fireOptions) -> nb::object {
      auto       moleculeVector = nvMolKit::NanobindForcefield::extractMolecules(molecules);
      const int  moleculeCount  = static_cast<int>(moleculeVector.size());
      const auto thresholdVector =
        nvMolKit::NanobindForcefield::extractDoubleList(vdwThresholds, moleculeCount, "vdwThreshold");
      const auto ignoreVector = nvMolKit::NanobindForcefield::extractBoolList(ignoreInterfragInteractions,
                                                                              moleculeCount,
                                                                              "ignoreInterfragInteractions");
      auto       result       = parseMinimizerKind(minimizerKind) == MinimizerKind::FIRE ?
                                  nvMolKit::UFF::UFFMinimizeMoleculesConfsFire(moleculeVector,
                                                                   maxIters,
                                                                   fireOptions,
                                                                   thresholdVector,
                                                                   ignoreVector,
                                                                               {},
                                                                   hardwareOptions,
                                                                   nvMolKit::CoordinateOutput::DEVICE,
                                                                   targetGpu) :
                                  nvMolKit::UFF::UFFMinimizeMoleculesConfs(moleculeVector,
                                                               maxIters,
                                                               1e-4,
                                                               thresholdVector,
                                                               ignoreVector,
                                                                           {},
                                                               hardwareOptions,
                                                               nvMolKit::CoordinateOutput::DEVICE,
                                                               targetGpu);
      if (!result.device.has_value()) {
        throw std::runtime_error("UFFMinimizeMoleculesConfs(DEVICE) returned no device result");
      }
      return nvMolKit::nanobind_bindings::buildOwningDevice3DResult(*result.device);
    },
    "molecules"_a,
    "maxIters"_a,
    "vdwThresholds"_a,
    "ignoreInterfragInteractions"_a,
    "hardwareOptions"_a,
    "targetGpu"_a,
    "minimizerKind"_a = std::string("BFGS"),
    "fireOptions"_a   = nvMolKit::FireOptions(),
    "Optimize conformers with UFF and return device-resident results.");
}
